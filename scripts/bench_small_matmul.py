"""Fixed cost of small B=1 matmuls (for the MoE port, doc/12 H2): time per op vs weight size, under trace replay.

  python scripts/bench_small_matmul.py [--iters 200] [--dtype bfp4]
Each case replays ITERS back-to-back ops in one trace, so the time includes the gap between ops on the device.
Modes: dl    DecodeLinear as tsugumi uses it (x pre-sharded, output stays L1 width-sharded)
       dl+o  DecodeLinear with out_mem=L1 interleaved (one more op, as most call sites)
       mm    plain ttnn.matmul, weight DRAM interleaved, x / out L1 interleaved
Then fits t = a + bytes / BW over the dl rows (a = fixed cost per op)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import argparse
import time

import torch
import ttnn
from tsugumi.ops import DecodeLinear, compute_cfg

BYTES = {"bfp4": 576 / 1024, "bfp8": 1088 / 1024}
DT = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
SHAPES = [  # (K, N): MoE expert sizes (hidden 2048, expert dim 256..1536) and dense ones for reference
    (2048, 256), (2048, 512), (2048, 768), (2048, 1536), (2048, 3072), (2048, 6144), (2048, 12288),
    (768, 2048), (1536, 2048), (512, 2048),
    (5120, 1024), (5120, 5120), (5120, 17408), (17408, 5120),
]


def timed(dev, fn, iters):
    o = fn()
    ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(iters):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        best = min(best, (time.perf_counter() - t0) / iters)
    ttnn.release_trace(dev, tid)
    return best


def sweep(dev, dt, iters, shapes=None):
    """DecodeLinear's K loop costs about 0.85 us per in0 block; fewer in0 cores and wider blocks cut the count."""
    print(f"{'shape':12s} {'in0c':>4s} {'bw':>3s} {'blocks':>6s} {'us':>7s}")
    for K, N in shapes or [(2048, 768), (2048, 1536), (2048, 3072), (768, 2048), (1536, 2048), (5120, 5120)]:
        w = torch.randn(K, N) * 0.02
        kt = K // 32
        for c in [c for c in (64, 48, 32, 24, 16, 12, 8, 6, 4) if kt % c == 0]:
            pk = kt // c
            for bw in sorted({d for d in (1, 2, 4, 8, 16, pk) if pk % d == 0}):
                try:
                    L = DecodeLinear(dev, w, dt, in0_cores=c, in0_block_w=bw)
                except Exception:
                    continue
                x = ttnn.from_torch(torch.randn(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=L.xmem)
                try:
                    t = timed(dev, lambda: L(x, out_mem=None), iters)
                    print(f"{K:5d}x{N:<6d} {c:4d} {bw:3d} {kt // bw:6d} {t * 1e6:7.1f}", flush=True)
                except Exception as e:
                    print(f"{K:5d}x{N:<6d} {c:4d} {bw:3d} FAILED {str(e).splitlines()[0][:80]}", flush=True)
                ttnn.deallocate(L.w)
                ttnn.deallocate(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dtype", default="bfp4")
    ap.add_argument("--sweep", action="store_true", help="vary in0 cores and in0_block_w on the MoE shapes instead")
    ap.add_argument("--shapes", default="", help="for --sweep: KxN,KxN,... (default: MoE expert shapes)")
    a = ap.parse_args()
    dt = DT[a.dtype]
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    rows = []
    if a.sweep:
        try:
            sweep(dev, dt, a.iters, [tuple(map(int, x.split('x'))) for x in a.shapes.split(',')] if a.shapes else None)
        finally:
            ttnn.close_device(dev)
        return
    try:
        # baseline: the cheapest op we can issue (elementwise add on one [32, 2048] tile row in L1)
        xb = ttnn.from_torch(torch.randn(1, 1, 32, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
        t = timed(dev, lambda: ttnn.add(xb, xb, memory_config=ttnn.L1_MEMORY_CONFIG), a.iters)
        print(f"{'add [32,2048] L1':24s} {t * 1e6:7.1f} us   (baseline op)", flush=True)
        ttnn.deallocate(xb)
        print(f"{'shape':14s} {'MB':>6s} {'dl us':>7s} {'GB/s':>6s} {'dl+o us':>8s} {'mm us':>7s} {'GB/s':>6s}")
        for K, N in SHAPES:
            w = torch.randn(K, N) * 0.02
            L = DecodeLinear(dev, w, dt)
            x = ttnn.from_torch(torch.randn(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=L.xmem)
            t_dl = timed(dev, lambda: L(x, out_mem=None), a.iters)
            t_dlo = timed(dev, lambda: L(x, out_mem=ttnn.L1_MEMORY_CONFIG), a.iters)
            ttnn.deallocate(L.w)
            ttnn.deallocate(x)
            wi = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            xi = ttnn.from_torch(torch.randn(1, 1, 32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)
            ck = compute_cfg()
            t_mm = timed(dev, lambda: ttnn.matmul(xi, wi, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ck),
                         a.iters)
            ttnn.deallocate(wi)
            ttnn.deallocate(xi)
            mb = K * L.Np * BYTES[a.dtype] / 1e6
            mbi = K * N * BYTES[a.dtype] / 1e6
            rows.append((mb * 1e6, t_dl))
            print(f"{K:5d}x{N:<6d}  {mb:6.2f} {t_dl * 1e6:7.1f} {mb / 1e3 / t_dl:6.0f} {t_dlo * 1e6:8.1f} "
                  f"{t_mm * 1e6:7.1f} {mbi / 1e3 / t_mm:6.0f}", flush=True)
    finally:
        ttnn.close_device(dev)
    # least squares t = a + b * bytes over the DecodeLinear rows
    X = torch.tensor([[1.0, b] for b, _ in rows], dtype=torch.float64)
    y = torch.tensor([t for _, t in rows], dtype=torch.float64)
    a0, b0 = torch.linalg.lstsq(X, y).solution.tolist()
    print(f"fit (dl): fixed {a0 * 1e6:.1f} us + bytes / {1 / b0 / 1e9:.0f} GB/s")


if __name__ == "__main__":
    main()
