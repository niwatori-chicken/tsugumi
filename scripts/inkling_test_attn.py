"""kernels/ink_attn_*.cpp (q / rel / sink must be in L1: the reader reads 32-byte face rows) (tsugumi/inkling/attn_kernel.py) vs torch: ring (sliding) and full layouts, several
positions (before / after the ring wraps, inside / beyond the bias window).

    python scripts/inkling_test_attn.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import ttnn  # noqa: E402

from tsugumi.inkling.attn_kernel import InkAttention  # noqa: E402

NH, NKV, HD, W = 24, 8, 128, 1024
R, RA = W + 32, 1152
MAXS = 8192
SA = MAXS + W
scale = HD ** -0.5


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def ref(q, K, V, rel, sink, P0, ring):
    """q [NH, HD], K/V [NKV, L, HD] (cache layout), rel [NH, wr] (1/scale units), sink [NH] (1/scale units)."""
    L = K.shape[1]
    keys = torch.arange(L)
    if ring:
        c = P0 % R
        dist = (c - keys) % R
        valid = (keys < R) & (dist <= min(P0, W - 1))
        bias = rel[:, dist.clamp(max=rel.shape[1] - 1)]
    else:
        dist = P0 - keys
        valid = (keys <= P0) & (keys < MAXS)
        bias = torch.where(((dist >= 0) & (dist < W))[None], rel[:, dist.clamp(0, W - 1)], torch.zeros(()))
    kk = K.repeat_interleave(NH // NKV, 0)
    vv = V.repeat_interleave(NH // NKV, 0)
    s = (torch.einsum("hd,hld->hl", q, kk) + bias) * scale
    s = s.masked_fill(~valid[None], float("-inf"))
    s = torch.cat([s, (sink * scale)[:, None]], -1)
    p = torch.softmax(s, -1)[:, :-1]
    return torch.einsum("hl,hld->hd", p, vv)


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        for ring, L, sink_slot in ((True, RA, R), (False, SA, MAXS)):
            att = InkAttention(dev, ring=ring, bsl=L, r=R, w=W, sink=sink_slot, wr=(R if ring else W))
            torch.manual_seed(0)
            K = torch.randn(NKV, L, HD)
            V = torch.randn(NKV, L, HD)
            K[:, sink_slot] = 0
            V[:, sink_slot] = 0
            kc = ttnn.from_torch(K[None], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
            vc = ttnn.from_torch(V[None], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
            Kq = ttnn.to_torch(kc).float()[0]
            Vq = ttnn.to_torch(vc).float()[0]
            for P0 in ((5, 63, 64, 700, 1055, 1056, 3000, 5000) if ring else (0, 5, 100, 1023, 1024, 2047, 5000, 8191)):
                q = torch.randn(NH, HD) * 0.5
                wr = R if ring else W
                rel = torch.randn(NH, wr) * 8
                sink = torch.randn(NH) * 20
                qt = ttnn.from_torch(q.view(1, 1, NH, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                rt = ttnn.from_torch(rel.view(1, 1, NH, wr), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                st = ttnn.from_torch(sink.view(1, 1, 1, NH), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                pt = ttnn.from_torch(torch.full((8,), P0, dtype=torch.int32), dtype=ttnn.int32, device=dev)
                n = att.chunks(P0)
                npt = ttnn.from_torch(torch.full((8,), n * att.C - 1, dtype=torch.int32), dtype=ttnn.int32, device=dev)
                o = att(qt, kc, vc, rt, st, pt, npt)
                oo = ttnn.to_torch(o).float()[0, 0, 0].view(NH, HD)
                r_ = ref(q.to(torch.bfloat16).float(), Kq, Vq, rel.to(torch.bfloat16).float(),
                         sink.to(torch.bfloat16).float(), P0, ring)
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                for _ in range(20):
                    att(qt, kc, vc, rt, st, pt, npt)
                ttnn.synchronize_device(dev)
                dt = (time.perf_counter() - t) / 20 * 1e6
                print(f"{'ring' if ring else 'full'} P0={P0:5d} chunks {n:3d}: pcc {pcc(oo, r_):.5f} "
                      f"max|d| {(oo - r_).abs().max():.4f}  ({dt:.0f} us untraced)", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
