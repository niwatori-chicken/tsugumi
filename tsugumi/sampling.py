"""Token choice on the host from the main model's top-k logits (model.TOPK per row, computed on the device):
sampling, and repetition penalties (also for greedy decoding).

Sampling presets (Qwen3.8 model card) for requests that do not set `temperature`: see PRESETS."""
import os

import torch

from .model import TOPK

# Qwen3.8-27B model card, "best practices". presence_penalty 1.5 in the non-thinking mode is there "to reduce
# endless repetition"; generation_config.json has the thinking values without the penalty.
PRESETS = {
    "thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
    "instruct": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
    "completion": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
    # Inkling 2 Mini (generation_config.json: do_sample, temperature 1.0; no top-p / top-k)
    "inkling": dict(temperature=1.0),
}
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty")


class Sampler:
    """Picks a token from a row's top-k logits.

    temperature > 0: sampling with top_k (<= TOPK; 0 = TOPK), min_p, top_p, in that order. seed=None draws a
    random seed. temperature == 0: greedy (only useful with a penalty; plain greedy needs no Sampler).

    presence_penalty / frequency_penalty (OpenAI semantics, as vLLM: over the generated tokens only, not the
    prompt): logit -= presence * (count > 0) + frequency * count. The generated tokens so far are recorded with
    observe(); pick(..., extra) adds the tokens between them and the row (the drafts verified before it), so
    every row of a speculative step sees exactly its own prefix and speculative decoding stays exact.

    Only the device top-k (64) is known: penalties lower logits, so a token outside it could end up above a
    penalized one. With top_k 20 that needs more than 44 of the 64 to be penalized below the 64th; the error is
    then confined to the tail of the distribution."""

    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, presence_penalty=0.0,
                 frequency_penalty=0.0):
        self.t = float(temperature or 0.0)
        if self.t < 0:
            raise ValueError("temperature must be >= 0")
        self.top_p = min(float(top_p), 1.0) if top_p is not None else 1.0  # <= 0: the top token only
        self.top_k = min(int(top_k), TOPK) if top_k and int(top_k) > 0 else TOPK
        self.min_p = float(min_p or 0.0)
        self.presence = float(presence_penalty or 0.0)
        self.frequency = float(frequency_penalty or 0.0)
        for name, x in (("presence_penalty", self.presence), ("frequency_penalty", self.frequency)):
            if not -2.0 <= x <= 2.0:
                raise ValueError(f"{name} must be in [-2, 2]")
        self.penalized = bool(self.presence or self.frequency)
        if self.t == 0 and not self.penalized:
            raise ValueError("temperature 0 without a penalty: use no sampler (greedy)")
        self.counts = {}  # generated token -> count
        self.g = torch.Generator()
        self.g.manual_seed(int(seed) if seed is not None else int.from_bytes(os.urandom(8), "little") >> 1)

    @property
    def greedy(self):
        return self.t == 0

    def observe(self, t):
        """Record a generated token (for the penalties)."""
        if self.penalized:
            self.counts[t] = self.counts.get(t, 0) + 1

    def adjust(self, vals, idx, extra=()):
        """Penalized logits, sorted again (descending): (vals, idx)."""
        if not self.penalized:
            return vals, idx
        ex = {}
        for t in extra:
            ex[t] = ex.get(t, 0) + 1
        if not self.counts and not ex:
            return vals, idx
        n = torch.tensor([self.counts.get(t, 0) + ex.get(t, 0) for t in idx.tolist()], dtype=torch.float32)
        v = vals.float() - self.presence * (n > 0).float() - self.frequency * n
        order = torch.sort(v, descending=True, stable=True).indices
        return v[order], idx[order]

    def probs(self, vals):
        """Sorted logits [K] -> probabilities over the same K entries (zeros where cut off)."""
        x = vals[: self.top_k].double() / self.t
        p = torch.softmax(x, dim=0)
        if self.min_p > 0:
            p = torch.where(p >= self.min_p * p[0], p, torch.zeros_like(p))
        if self.top_p < 1.0:
            c = torch.cumsum(p, 0) / p.sum()
            keep = int(torch.searchsorted(c, torch.tensor(self.top_p, dtype=c.dtype)).item()) + 1
            p[keep:] = 0
        out = torch.zeros(vals.shape[0], dtype=torch.float64)
        out[: p.shape[0]] = p / p.sum()
        return out

    def pick(self, vals, idx, extra=()):
        """Token for one row; extra: tokens generated after the observed ones and before this row."""
        v, i = self.adjust(vals, idx, extra)
        if self.greedy:  # the engine's tie-break: lowest token id among the equal maxima
            return int(i[v == v[0]].min())
        j = int(torch.multinomial(self.probs(v), 1, generator=self.g).item())
        return int(i[j])

    def __call__(self, vals, idx):
        return self.pick(vals, idx)


def resolve(body, mode, defaults="model"):
    """Sampling parameters of a request. mode: "thinking" / "instruct" / "completion". defaults="model": a request
    without `temperature` gets PRESETS[mode], where the fields it does set win; "greedy": no preset (the old
    behaviour, greedy unless temperature > 0). A request that sets `temperature` (also 0) gets exactly its fields.
    Returns (params dict or None for plain greedy, preset name or None)."""
    given = {k: body.get(k) for k in SAMPLING_KEYS if body.get(k) is not None}
    preset = None
    if "temperature" not in given and defaults == "model":
        preset = mode
        params = dict(PRESETS[mode])
        params.update(given)
    else:
        params = given
    params.setdefault("temperature", 0.0)  # a penalty alone: greedy with the penalty
    if not float(params.get("temperature") or 0) > 0 and not (params.get("presence_penalty") or
                                                            params.get("frequency_penalty")):
        return None, preset
    return params, preset
