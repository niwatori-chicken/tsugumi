"""Unit test: fused post-conv GDN kernel vs torch (random data). python scripts/test_gdn_fused.py T [steps]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys
import time

import torch
import torch.nn.functional as F
import ttnn

from tsugumi.gdn_kernel import GDNFusedKernel

NK, NV, D = 16, 48, 128
KD, VD = NK * D, NV * D
QKV = 2 * KD + VD


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    try:
        dbg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        kern = GDNFusedKernel(dev, NV, NK, T, QKV, KD, dbg=dbg)
        w = 1 + 0.2 * torch.randn(D)
        C = GDNFusedKernel.make_consts(dev, w, T)
        S = torch.randn(NV, D, D) * 0.05
        snap = torch.zeros(1, T * NV, D, D)
        slot = T - 1  # tests commit all T tokens each step
        snap[0, slot * NV:(slot + 1) * NV] = S
        St = ttnn.from_torch(snap, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        IDX = GDNFusedKernel.make_idx(dev, slot)
        for step in range(steps):
            x = torch.randn(T, QKV).bfloat16().float()
            z = torch.randn(T, VD).bfloat16().float()
            eg = torch.exp(-torch.rand(T, NV) * 0.5)
            beta = torch.rand(T, NV)
            # reference
            q = (F.normalize(x[:, :KD].view(T, NK, D), dim=-1, eps=1e-6) * D**-0.5).repeat_interleave(3, 1)
            k = F.normalize(x[:, KD:2 * KD].view(T, NK, D), dim=-1, eps=1e-6).repeat_interleave(3, 1)
            v = x[:, 2 * KD:].view(T, NV, D)
            outs = []
            for t in range(T):
                S = S * eg[t][:, None, None]
                kv = torch.einsum("hk,hkv->hv", k[t], S)
                S = S + k[t][:, :, None] * ((v[t] - kv) * beta[t][:, None])[:, None, :]
                o = torch.einsum("hk,hkv->hv", q[t], S)
                o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * w
                outs.append((o * F.silu(z[t].view(NV, D))).reshape(-1))
            ref = torch.stack(outs)
            X = torch.zeros(1, 1, 32, QKV)
            X[0, 0, :T] = x
            Y = torch.zeros(1, 1, 32, QKV + VD + 96)
            Y[0, 0, :T, QKV:QKV + VD] = z
            GB = torch.zeros(1, 1, 32, 96)
            GB[0, 0, :T, :NV] = eg
            GB[0, 0, :T, NV:] = beta
            Xt = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
            Yt = ttnn.from_torch(Y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
            GBt = ttnn.from_torch(GB, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            out = kern(Xt, Yt, GBt, GBt, C, St, IDX)
            got = ttnn.to_torch(out)[0, 0, :T].float()
            if dbg:
                full = ttnn.to_torch(out)[0, 0].float()  # [32, 6144]; head 0 = cols 0..127
                h0 = full[:, :128]
                print("dbg head0 tile0 rows0-2 cols0-6:\n", h0[:3, :7], "\nrow31:", h0[31, :4],
                      "\nnan count", torch.isnan(full).sum().item(), "inf", torch.isinf(full).sum().item())
                kn = F.normalize(x[:, KD:KD + 128], dim=-1)
                print("ref kn row0:", kn[0, :7], " ref q row0", x[0, :7])
                break
            s = ttnn.to_torch(St)[0, slot * NV:(slot + 1) * NV]
            print(f"step {step}: out pcc {pcc(got, ref):.6f} maxerr {(got - ref).abs().max():.3e} "
                  f"S pcc {pcc(s, S):.6f}", flush=True)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(20):
            kern(Xt, Yt, GBt, GBt, C, St, IDX, OUT=out)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        print(f"fused kernel T={T}: {(time.perf_counter() - t0) / 20 * 1e6:.1f} us")
        ttnn.release_trace(dev, tid)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
