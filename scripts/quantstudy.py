"""Weight-only quantization error study (CPU) for TT-native block formats.

Emulates the tt-metal host packer (blockfloat_common.cpp): groups of 16 values that are
contiguous along the last dim of the ttnn weight [K=in, N=out] (i.e. 16 output channels at
one input index), shared exponent = max exponent, sign + mantissa with the hidden bit
(3 bits for bfp4, 7 for bfp8), round-to-nearest-even, saturating.

Usage: python scripts/quantstudy.py <shard.safetensors> [name-filter]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import sys

import torch
from safetensors import safe_open


def bfp_quant(w, mant_bits, group=16, exp_shift=0):
    """w: [..., N] grouped along last dim. exp_shift>0 lowers the shared exponent (clips)."""
    shp = w.shape
    g = w.reshape(-1, group).float()
    amax = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    E = torch.floor(torch.log2(amax)) - exp_shift
    step = torch.pow(2.0, E - (mant_bits - 1))
    qmax = 2**mant_bits - 1
    q = torch.round(g.abs() / step).clamp(max=qmax)  # torch.round = half-to-even
    return (torch.sign(g) * q * step).reshape(shp)


def bfp_quant_best(w, mant_bits, group=16, shifts=(0, 1)):
    """Per group pick the exponent (shared or lowered = clipped) with least squared error."""
    shp = w.shape
    g = w.reshape(-1, group).float()
    best = None
    best_err = None
    for s in shifts:
        q = bfp_quant(g, mant_bits, group, s).reshape(-1, group)
        err = ((q - g) ** 2).sum(dim=1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            m = err < best_err
            best = torch.where(m, q, best)
            best_err = torch.where(m, err, best_err)
    return best.reshape(shp)


def q_sym(w, bits, group, scale_dtype=torch.float16):
    """Symmetric RTN with a float scale per group (Q4_0-like when bits=4, group=32)."""
    shp = w.shape
    g = w.reshape(-1, group).float()
    qmax = 2 ** (bits - 1) - 1  # e.g. 7 for 4 bit (15 levels used + -8 unused)
    amax = g.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    s = (amax / qmax).to(scale_dtype).float()
    q = torch.round(g / s).clamp(-qmax - 1, qmax)
    return (q * s).reshape(shp)


def q_asym(w, bits, group):
    """Asymmetric RTN with scale+min per group (Q4_1 / Q3_K-ish without super-blocks)."""
    shp = w.shape
    g = w.reshape(-1, group).float()
    lo = g.amin(dim=1, keepdim=True)
    hi = g.amax(dim=1, keepdim=True)
    s = ((hi - lo) / (2**bits - 1)).clamp_min(1e-30).half().float()
    lo = lo.half().float()
    q = torch.round((g - lo) / s).clamp(0, 2**bits - 1)
    return (q * s + lo).reshape(shp)


def rel(w, q):
    return (((w - q) ** 2).sum() / (w**2).sum()).item()


METHODS = {
    "bfp8": (8.5, lambda w: bfp_quant(w, 7)),
    "bfp4": (4.5, lambda w: bfp_quant(w, 3)),
    "bfp4best": (4.5, lambda w: bfp_quant_best(w, 3)),
    "q4_0(g32)": (4.5, lambda w: q_sym(w, 4, 32)),
    "q4_1(g32)": (5.0, lambda w: q_asym(w, 4, 32)),
    "q3sym(g16)": (4.0, lambda w: q_sym(w, 3, 16)),
    "q3asym(g32)": (4.0, lambda w: q_asym(w, 3, 32)),
    "bfp2": (2.5, lambda w: bfp_quant(w, 1)),
}


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    print("name".ljust(58), " ".join(f"{m:>11s}" for m in METHODS))
    print("bpw".ljust(58), " ".join(f"{b:>11.2f}" for b, _ in METHODS.values()))
    with safe_open(path, "pt") as f:
        for k in sorted(f.keys()):
            if filt and filt not in k:
                continue
            if not k.endswith(".weight"):
                continue
            t = f.get_tensor(k)
            if t.ndim != 2 or min(t.shape) < 256:
                continue
            w = t.float().T.contiguous()  # ttnn layout [in, out]: groups along out
            if w.shape[1] % 32:
                continue
            errs = [rel(w, fn(w)) for _, fn in METHODS.values()]
            print(k[:58].ljust(58), " ".join(f"{e:11.2e}" for e in errs), flush=True)


if __name__ == "__main__":
    main()
