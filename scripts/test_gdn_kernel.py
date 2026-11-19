"""Unit test of the fused GDN recurrence kernel vs a torch fp32 reference (random data).

python scripts/test_gdn_kernel.py T [steps] [--nofp32]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import sys
import time

import torch
import torch.nn.functional as F
import ttnn

from qe.gdn_kernel import GDNRecKernel

H, D = 48, 128


def ref(q, k, v, beta, g, S):
    """q,k,v [T,H,D] (q,k normalized/scaled), beta,g [T,H], S [H,D,D] -> o [T,H,D], S."""
    S = S.clone()
    outs = []
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv = torch.einsum("hk,hkv->hv", k[t], S)
        delta = (v[t] - kv) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        outs.append(torch.einsum("hk,hkv->hv", q[t], S))
    return torch.stack(outs), S


def tt(x, dev):
    return ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    steps = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    fp32 = "--nofp32" not in sys.argv
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    try:
        kern = GDNRecKernel(dev, H, T, fp32_unpack=fp32)
        E = torch.zeros(1, 1, T * 32, 32)
        for t in range(T):
            E[0, 0, t * 32 + t, :] = 1
        Et = tt(E, dev)
        S = torch.randn(H, D, D) * 0.05
        St = tt(S.unsqueeze(0), dev)
        for step in range(steps):
            q = F.normalize(torch.randn(T, H, D), dim=-1) * D**-0.5
            k = F.normalize(torch.randn(T, H, D), dim=-1)
            v = torch.randn(T, H, D)
            beta = torch.rand(T, H)
            g = -torch.rand(T, H) * 0.5
            o_ref, S = ref(q, k, v, beta, g, S)

            def rows(x):  # [T,H,D] -> [1,H,32,D]
                out = torch.zeros(1, H, 32, D)
                out[0, :, :T] = x.transpose(0, 1)
                return out

            Q, K, V = rows(q), rows(k), rows(v)
            KT = K.transpose(-1, -2).contiguous()
            B = torch.zeros(1, H, 32, 32)
            B[0, :, :T, :] = beta.T[:, :, None]
            G = torch.zeros(1, H, T * 32, 32)
            for t in range(T):
                G[0, :, t * 32:(t + 1) * 32, :] = g[t].exp()[:, None, None]
            O = kern(tt(Q, dev), tt(K, dev), tt(KT, dev), tt(V, dev), tt(B, dev), tt(G, dev), Et, St)
            o = ttnn.to_torch(O)[0, :, :T].transpose(0, 1)
            s = ttnn.to_torch(St)[0]
            print(f"step {step}: o pcc {pcc(o, o_ref):.6f} maxerr {(o - o_ref).abs().max():.2e}  "
                  f"S pcc {pcc(s, S):.6f} maxerr {(s - S).abs().max():.2e}", flush=True)
        # timing
        args = [tt(Q, dev), tt(K, dev), tt(KT, dev), tt(V, dev), tt(B, dev), tt(G, dev), Et, St]
        Ot = kern(*args)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(20):
            kern(*args, O=Ot)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        print(f"kernel T={T}: {(time.perf_counter() - t0) / 20 * 1e6:.1f} us")
        ttnn.release_trace(dev, tid)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
