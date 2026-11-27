"""MoEGemm (kernels/moe_gemm_*.cpp) vs torch: rows sorted by expert with real routing counts (one T=2048 chunk of
moe_route_stats.py output), Qwen3.6-35B-A3B sizes (gate_up K=2048 N=1024, down K=512 N=2048), 256 experts, bfp4.
  python scripts/test_moe_gemm.py [--layer 20]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import math
import sys
import time

import torch
import ttnn
from tsugumi.moe_gemm import MoEGemm

E = 256
layer = int(sys.argv[sys.argv.index("--layer") + 1]) if "--layer" in sys.argv else 20
d = torch.load("/mnt/HDD/eval/moe_routes_q36.pt")
i, j = d["names"].index("ja prose"), d["names"].index("code py")
r = torch.cat([d["routes"][i], d["routes"][j]], 1)[layer, :2048]
cnt = torch.bincount(r.reshape(-1).long(), minlength=E)
tiles = [math.ceil(int(c) / 32) for c in cnt]
offs = [0]
for t in tiles[:-1]:
    offs.append(offs[-1] + t)
R = sum(tiles) * 32
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
try:
    NNS = {"gate_up": [int(x) for x in (sys.argv[sys.argv.index("--nn-gu") + 1] if "--nn-gu" in sys.argv else "8").split(",")],
           "down": [int(x) for x in (sys.argv[sys.argv.index("--nn-dn") + 1] if "--nn-dn" in sys.argv else "8").split(",")]}
    for name, K, N, NN in [(n, K, N, nn) for n, K, N in (("gate_up", 2048, 1024), ("down", 512, 2048)) for nn in NNS[n]]:
        g = torch.Generator().manual_seed(0)
        W = (torch.randn(E, K, N, generator=g) * 0.05).bfloat16()
        X = torch.randn(R, K, generator=g).bfloat16()
        Wt = ttnn.from_torch(W.reshape(1, 1, E * K, N), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev)
        Xt = ttnn.from_torch(X.reshape(1, 1, R, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        Yt = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                            ttnn.DRAM_MEMORY_CONFIG)
        Wq = ttnn.to_torch(Wt).float().reshape(E, K, N)  # bfp4-rounded weights for the reference
        ref = torch.empty(R, N)
        for e in range(E):
            a, b = offs[e] * 32, (offs[e] + tiles[e]) * 32
            if b > a:
                ref[a:b] = X[a:b].float() @ Wq[e]
        used = int((cnt > 0).sum())
        opt = sys.argv[sys.argv.index("--rtct") + 1] if "--rtct" in sys.argv else "2x2"
        for RT_, CT_ in [tuple(map(int, x.split("x"))) for x in opt.split(",")]:
            mg = MoEGemm(dev, K, N, NN=NN, RT=RT_, CT=CT_)
            units = mg.units(tiles, offs)
            mg(Xt, Wt, Yt, units)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(5):
                mg(Xt, Wt, Yt, units)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / 5
            got = ttnn.to_torch(Yt).float().reshape(R, N)
            pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            print(f"{name} NN={NN} RT={RT_} CT={CT_}: R={R} rows ({used} experts, {len(units)} units), "
                  f"{dt * 1e3:.2f} ms, PCC {pcc:.6f}, max err {err:.1e}; "
                  f"floor {used * K * N * 576 / 1024 / 1e9 / 390 * 1e3:.2f} ms (weights once)", flush=True)
        for t in (Wt, Xt, Yt):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(dev)
