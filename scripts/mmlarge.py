"""Large-M matmul against the engine's bfp4 DRAM width-sharded weights (prefill with T >= 128).
  python scripts/mmlarge.py [M ...]
Compares: dram-sharded program (per_core_M = M/32), minimal_matmul (default config), ttnn.linear (auto)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import math, sys, time, torch, ttnn
from tsugumi.ops import DecodeLinear

Ms = [int(a) for a in sys.argv[1:]] or [32, 128, 256, 512, 1024]
SHAPES = {  # name: (K, N)
    "gdn_in": (5120, 16480), "gdn_out": (6144, 5120), "mlp_gate": (5120, 17408), "mlp_down": (17408, 5120),
    "attn_qkvg": (5120, 2 * 6144 + 2 * 1024), "attn_o": (6144, 5120),
}
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)


def tm(f, n=10):
    out = f(); ttnn.deallocate(out); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [f() for _ in range(2)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t) / n / 2 * 1e3
    ttnn.release_trace(dev, tid)
    for o in outs:
        ttnn.deallocate(o)
    return dt


try:
    for name, (K, N) in SHAPES.items():
        L = DecodeLinear(dev, torch.randn(K, N) * 0.02, ttnn.bfloat4_b, name=name)
        x0 = torch.randn(max(Ms), K)
        wref = ttnn.to_torch(L.w).float()[:, :N]
        for M in Ms:
            xd = ttnn.from_torch(x0[:M].reshape(1, 1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ref = x0[:M] @ wref
            flop = 2 * M * K * N
            res = []
            ck = L.ck
            cands = {
                "mm_min": lambda: ttnn.experimental.minimal_matmul(xd, L.w, compute_kernel_config=ck,
                                                                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                                   dtype=ttnn.bfloat16),
                "linear": lambda: ttnn.linear(xd, L.w, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                              dtype=ttnn.bfloat16),
            }
            for cn, f in cands.items():
                try:
                    y = f()
                    yt = ttnn.to_torch(y).float().reshape(M, -1)[:, :N]
                    ttnn.deallocate(y)
                    p = torch.corrcoef(torch.stack([yt.flatten(), ref.flatten()]))[0, 1].item()
                    ms = tm(f)
                    res.append(f"{cn} {ms:7.3f}ms {flop / ms / 1e9:6.1f}TF pcc{p:.4f}")
                except Exception as e:
                    res.append(f"{cn} FAIL {str(e).splitlines()[0][:80]}")
            print(f"{name:10s} K={K:5d} N={N:5d} M={M:5d}: " + " | ".join(res), flush=True)
            ttnn.deallocate(xd)
        ttnn.deallocate(L.w)
finally:
    ttnn.close_device(dev)
