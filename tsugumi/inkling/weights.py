"""Reader for the original Inkling 2 Mini checkpoint (model-00001-of-00001.safetensors, names `model.llm.*`).

Experts are stored quantized and dequantized here (as transformers' Inkling2LoopDequantizeExperts does):
- NVFP4 (layers with `*_amax`): two E2M1 codes per byte (even element in the low nibble), one float8 (E4M3) scale per
  16 values along the input dimension, and a per-expert factor amax / outer_scale_denominator.
- MXFP8 (the others): float8 (E4M3) values with one power-of-two scale 2 ** (s - 127) per 32 values.
w13 rows interleave gate and up (gate = rows 0::2, up = rows 1::2); w2 is down [hidden, inter].
"""
import glob
import json
import os

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_BYTE = torch.arange(256)
BYTE_TO_PAIR = torch.stack((E2M1[_BYTE & 0xF], E2M1[_BYTE >> 4]), dim=-1)  # [256, 2] (low nibble first)

P = "model.llm."


class Checkpoint:
    def __init__(self, model_dir):
        self.dir = model_dir
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
            files = sorted(set(wm.values()))
        else:
            files = [os.path.basename(p) for p in glob.glob(os.path.join(model_dir, "model*.safetensors"))]
            wm = None
        self.handles = {fn: safe_open(os.path.join(model_dir, fn), framework="pt") for fn in files}
        self.where = {}
        for fn, h in self.handles.items():
            for k in h.keys():
                self.where[k] = fn

    def has(self, name):
        return (P + name) in self.where

    def raw(self, name):
        k = P + name
        return self.handles[self.where[k]].get_tensor(k)

    def get(self, name, dtype=torch.float32):
        return self.raw(name).to(dtype)

    def get_slice(self, name):
        k = P + name
        return self.handles[self.where[k]].get_slice(k)

    @property
    def loop_range(self):
        import json as _j
        with open(os.path.join(self.dir, "config.json")) as f:
            lc = _j.load(f).get("text_config", {}).get("loop_config", {})
        return lc.get("core_start", 6), lc.get("core_end", 18)

    # ---- experts -------------------------------------------------------------------------------------------
    def is_nvfp4(self, layer):
        return self.has(f"layers.{layer}.mlp.experts.w13_weight_amax")

    def _dequant(self, layer, which, experts):
        """Dequantized [len(experts), rows, cols] float32 of w13 / w2 for the given expert indices (a slice)."""
        base = f"layers.{layer}.mlp.experts.{which}"
        w = self.get_slice(base + "_weight")[experts]
        s = self.get_slice(base + "_weight_scale")[experts]
        if self.is_nvfp4(layer):
            amax = self.raw(base + "_weight_amax")[experts].float()
            den = self.raw(base + "_weight_outer_scale_denominator").float()
            n, rows, half = w.shape
            v = BYTE_TO_PAIR[w.to(torch.int64)].view(n, rows, half * 2 // 16, 16)
            v = v * s.float()[..., None]
            v = v * (amax / den)[:, None, None, None]
            return v.view(n, rows, half * 2)
        n, rows, cols = w.shape
        v = w.float().view(n, rows, cols // 32, 32) * torch.exp2(s.float() - 127)[..., None]
        return v.view(n, rows, cols)

    def experts(self, layer, start, stop):
        """(gate [n, inter, hidden], up [n, inter, hidden], down [n, hidden, inter]) float32 for experts start..stop."""
        w13 = self._dequant(layer, "w13", slice(start, stop))
        w2 = self._dequant(layer, "w2", slice(start, stop))
        return w13[:, 0::2], w13[:, 1::2], w2

    def expert(self, layer, e):
        g, u, d = self.experts(layer, e, e + 1)
        return g[0], u[0], d[0]
