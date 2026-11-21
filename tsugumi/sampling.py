"""Token sampling on the host from the main model's top-k logits (model.TOPK per row, computed on the device)."""
import os

import torch

from .model import TOPK


class Sampler:
    """temperature > 0, then top_k (<= TOPK; 0 = TOPK), min_p, top_p, in that order. seed=None draws a random seed.
    Tokens outside the device top-k (64) are never sampled; with the usual settings (top_k 20, top_p 0.8-0.95)
    this changes nothing."""

    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, seed=None):
        if temperature <= 0:
            raise ValueError("temperature must be > 0 (use no sampler for greedy)")
        self.t = float(temperature)
        self.top_p = float(top_p) if top_p is not None else 1.0
        self.top_k = min(int(top_k), TOPK) if top_k and top_k > 0 else TOPK
        self.min_p = float(min_p or 0.0)
        self.g = torch.Generator()
        self.g.manual_seed(int(seed) if seed is not None else int.from_bytes(os.urandom(8), "little") >> 1)

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

    def __call__(self, vals, idx):
        j = int(torch.multinomial(self.probs(vals), 1, generator=self.g).item())
        return int(idx[j])
