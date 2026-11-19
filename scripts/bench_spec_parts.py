"""Per-component device time in T-row mode (each part traced alone)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import sys
import torch
import ttnn

from bench_parts import timeit
from qe.model import Precision, Qwen35Engine
import qe.paths as qe_paths

MODEL = qe_paths.MODEL
CACHE = qe_paths.CACHE
_L1 = ttnn.L1_MEMORY_CONFIG


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=4096, n_layers=4, log=lambda *a: None)
        eng.init_spec(T)
        eng.reset_spec()
        eng.set_inputs_spec(list(range(100, 100 + T)), 10, 1)
        x = eng.x_in
        G, A = eng.layers[0], eng.layers[3]
        g, at, sh = G["mix"], A["mix"], eng.sh
        h = eng._norm(x, G["in_norm"])
        y = g.w_in(h, out_mem=_L1)
        qd = g.qkv_dim
        B = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, qd), memory_config=_L1)
        def conv_part():
            cat = ttnn.concat([g.Hc, g.Bp], dim=2, memory_config=_L1)
            hn = ttnn.matmul(sh.m_hist, cat, memory_config=_L1, compute_kernel_config=sh.sel_ck)
            ttnn.copy(hn, g.Hc)
            cat = ttnn.concat([g.Hc, B], dim=2, memory_config=_L1)
            conv = None
            for j in range(4):
                sj = ttnn.matmul(sh.m_shift[j], cat, memory_config=_L1, compute_kernel_config=sh.sel_ck)
                conv = ttnn.multiply(sj, g.taps[j], memory_config=_L1) if conv is None else ttnn.mac(sj, g.taps[j], conv)
            conv = ttnn.silu(conv, memory_config=_L1)
            ttnn.copy(B, g.Bp)
            return conv
        conv = conv_part()
        zd, nv = qd + g.vd, g.Nv
        def ab_part():
            ab = ttnn.typecast(ttnn.slice(y, (0, 0, 0, zd), (1, 1, 32, zd + 2 * nv), memory_config=_L1), ttnn.float32)
            u = ttnn.add(ab, g.ab_bias, memory_config=_L1)
            ge = ttnn.exp(ttnn.multiply(ttnn.softplus(u, memory_config=_L1), g.ab_coef, memory_config=_L1), memory_config=_L1)
            be = ttnn.sigmoid(u, memory_config=_L1)
            return ge, be
        ge, be = ab_part()
        o = g.fused_T(conv, y, ge, be, g.consts_T, g.snap, sh.idx)
        parts = {
            "GDN forward_spec": lambda: g.forward_spec(h, sh),
            "  w_in matmul": lambda: g.w_in(h, out_mem=_L1),
            "  conv part": conv_part,
            "  ab part": ab_part,
            "  fused kernel": lambda: g.fused_T(conv, y, ge, be, g.consts_T, g.snap, sh.idx),
            "  w_out matmul": lambda: g.w_out(o, out_mem=_L1),
            "attn forward_spec": lambda: at.forward_spec(h, sh),
            "  qkvg matmul": lambda: at.wqkvg(h, out_mem=_L1),
            "MLP": lambda: G["mlp"](h),
            "norm": lambda: eng._norm(x, G["in_norm"]),
            "to_resid(5376)": lambda: eng._to_resid(g.w_out(o, out_mem=_L1)),
            "lm_head+argmax": lambda: eng._logits_argmax(eng._lm_head_rows(h)),
        }
        for name, fn in parts.items():
            try:
                print(f"{name:22s} {timeit(dev, fn):8.1f} us", flush=True)
            except Exception as e:
                print(f"{name:22s} FAILED {str(e).splitlines()[0][:150]}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
