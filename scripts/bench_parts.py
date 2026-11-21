"""Per-component device time of the decode step (each part traced alone, 20 reps)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import sys
import time

import torch
import ttnn

from tsugumi.model import Precision, Qwen35Engine
import tsugumi.paths as tsugumi_paths

MODEL = tsugumi_paths.MODEL
CACHE = tsugumi_paths.CACHE
_L1 = ttnn.L1_MEMORY_CONFIG


def timeit(dev, fn, n=20):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / n
    ttnn.release_trace(dev, tid)
    return dt * 1e6


def main():
    prec = sys.argv[1] if len(sys.argv) > 1 else "bfp4"
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision(prec), f"{CACHE}/{prec}", max_seq=4096, n_layers=4,
                           log=lambda *a: None)
        eng.reset()
        eng.set_inputs(100, 5)
        x = eng.x_in
        G, A = eng.layers[0], eng.layers[3]
        h = ttnn.rms_norm(x, weight=G["in_norm"], epsilon=eng.eps, memory_config=_L1)
        parts = {
            "rms_norm": lambda: ttnn.rms_norm(x, weight=G["in_norm"], epsilon=eng.eps, memory_config=_L1),
            "residual add": lambda: ttnn.add(x, h, memory_config=_L1),
            "GDN mixer": lambda: G["mix"].forward_decode(h),
            "GDN in matmul": lambda: G["mix"].w_in(h),
            "attn mixer": lambda: A["mix"].forward_decode(h, eng.cur_pos, eng.cos, eng.sin),
            "attn qkvg matmul": lambda: A["mix"].wqkvg(h),
            "MLP": lambda: G["mlp"](h),
            "MLP gate matmul": lambda: G["mlp"].w1(h, out_mem=None),
            "MLP down matmul": lambda: G["mlp"].w2(ttnn.pad(ttnn.slice(h, (0, 0, 0, 0), (1, 1, 32, 5120)), [(0, 0), (0, 0), (0, 0), (0, 17408 - 5120)], 0.0)),
            "lm_head (8 chunks)": lambda: [lin(h, out_mem=ttnn.DRAM_MEMORY_CONFIG) for _, _, lin in eng.lm_chunks],
        }
        for name, fn in parts.items():
            try:
                print(f"{name:20s} {timeit(dev, fn):8.1f} us", flush=True)
            except Exception as e:
                print(f"{name:20s} FAILED {str(e).splitlines()[0][:150]}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
