"""Sweep the DRAM-sharded decode matmul settings (DecodeLinear: workers per bank, in0_block_w, in0 cores) for the
model's weight shapes on this board; prints time and % of the DRAM roofline per setting, best first.

  python scripts/mm_sweep.py [--shapes gdn_out,attn_o,mlp_down] [--bw 512] [--rows 32]
The defaults of DecodeLinear were chosen on a P100A (7 DRAM banks); a P150A has 8."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import argparse
import itertools
import time

import torch
import ttnn

from tsugumi.ops import DecodeLinear, TILE

SHAPES = {"mlp_gate_up": (5120, 17408), "mlp_down": (17408, 5120), "gdn_in": (5120, 16480), "gdn_out": (6144, 5120),
          "attn_qkvg": (5120, 14336), "attn_o": (6144, 5120)}
B4 = 576 / 1024


def timed(dev, fn, n=50):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / n
    ttnn.release_trace(dev, tid)
    return dt


def in0_options(k_tiles):
    out = []
    for c in range(64, 0, -1):
        if k_tiles % c == 0 and any(c % r == 0 and c // r <= 8 for r in range(1, 9)):
            out.append(c)
    return out[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="gdn_out,mlp_down,mlp_gate_up,gdn_in,attn_qkvg")
    ap.add_argument("--bw", type=float, default=512.0)
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0, trace_region_size=128 << 20, l1_small_size=24576)
    try:
        print(f"DRAM banks: {dev.dram_grid_size().x}", flush=True)
        for name in a.shapes.split(","):
            K, N = SHAPES[name]
            ideal = K * N * B4 / (a.bw * 1e9)
            default = DecodeLinear(dev, None, ttnn.bfloat4_b, shape=(K, N), no_weight=True)
            dflt = (3 if K <= 5120 else 2, default.pc.in0_block_w, default.in0_cores)
            res = []
            w = torch.randn(K, N) * 0.02
            for workers in (1, 2, 3):  # the op accepts 1..3
                lin0 = DecodeLinear(dev, w, ttnn.bfloat4_b, workers=workers, name=name)  # weights for this padding
                for cores in in0_options(K // TILE):
                    pk = K // TILE // cores
                    for bw_ in sorted({d for d in (1, 2, 3, 4, 6, 8, pk) if pk % d == 0}):
                        try:
                            lin = DecodeLinear(dev, None, ttnn.bfloat4_b, workers=workers, in0_block_w=bw_,
                                               in0_cores=cores, shape=(K, N), no_weight=True)
                            lin.w = lin0.w
                            x = ttnn.from_torch(torch.randn(1, 1, TILE, K) * 0.1, dtype=ttnn.bfloat16,
                                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=lin.xmem)
                            f = lambda: ttnn.deallocate(ttnn.linear(x, lin.w, program_config=lin.pc,
                                                                    compute_kernel_config=lin.ck,
                                                                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG))
                            res.append((timed(dev, f), workers, bw_, cores))
                            ttnn.deallocate(x)
                        except Exception as e:  # unsupported combination (CB size, shard shape, ...)
                            pass
                ttnn.deallocate(lin0.w)
            res.sort()
            d = next((r for r in res if (r[1], r[2], r[3]) == dflt), None)
            print(f"\n{name} K={K} N={N}: ideal {ideal * 1e6:.1f} us; current default (workers {dflt[0]}, "
                  f"in0_block_w {dflt[1]}, in0 cores {dflt[2]}): "
                  f"{d[0] * 1e6:.1f} us ({100 * ideal / d[0]:.0f}%)" if d else f"\n{name}: default not measured",
                  flush=True)
            for t, wk, b, c in res[:6]:
                print(f"   workers {wk} in0_block_w {b} in0 cores {c:2d}: {t * 1e6:7.1f} us  "
                      f"({100 * ideal / t:3.0f}% of peak)", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
