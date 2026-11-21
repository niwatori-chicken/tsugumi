"""bfp8_b KV cache (todo 4b): the KV write kernel and the GQA attention kernel.

  1. KVWriteKernel (bfp8): T rows written at various positions (crossing tile and block boundaries, shuffled page
     table) must equal the host packer's bf16 -> bfp8_b conversion bit for bit, and leave the other rows untouched.
  2. GQAAttention on bfp8 caches vs a torch reference on the bfp8-rounded K/V (T = 4 and 32, long contexts).
  python scripts/test_kv_bfp8.py [max_seq]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import sys, torch, ttnn
from tsugumi.kv_kernel import KVWriteKernel
from tsugumi.gqa_attn import GQAAttention

NH, NKV, HD, BS = 24, 4, 256, 64
B8 = ttnn.bfloat8_b
dev = ttnn.open_device(device_id=0, l1_small_size=24576)


def bfp8_round(X):
    """Host packer's bf16 -> bfp8_b -> float (exponents are per 16 values of a row, so any [.., 32k, 32m] works)."""
    return ttnn.to_torch(ttnn.from_torch(X.bfloat16(), dtype=B8, layout=ttnn.TILE_LAYOUT)).float()


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


ok = True
try:
    maxseq = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    nb = maxseq // BS
    torch.manual_seed(0)
    perm = torch.randperm(nb)
    ptd = ttnn.from_torch(perm.to(torch.int32).view(1, nb), dtype=ttnn.int32, device=dev)

    def logical(cache):  # device [nb, NKV, BS, HD] (physical) -> [maxseq, NKV, HD] logical order
        P = ttnn.to_torch(cache).float()
        return P[perm].permute(0, 2, 1, 3).reshape(maxseq, NKV, HD)

    print("1. KV write (bfp8)", flush=True)
    kc = ttnn.zeros([nb, NKV, BS, HD], dtype=B8, layout=ttnn.TILE_LAYOUT, device=dev)
    vc = ttnn.zeros([nb, NKV, BS, HD], dtype=B8, layout=ttnn.TILE_LAYOUT, device=dev)
    Kref = torch.zeros(maxseq, NKV, HD)
    Vref = torch.zeros(maxseq, NKV, HD)
    for T, starts in ((4, [0, 4, 30, 62, 63, 1000, 4093]), (32, [96, 2000, 4032]), (1, [7, 5000])):
        W = KVWriteKernel(dev, T, NKV, HD, BS, nb, B8)
        posd = ttnn.from_torch(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        for P0 in starts:
            if P0 + T > maxseq:
                continue
            # values with a wide range: some rows tiny, some large, exact zeros and negatives
            k = torch.randn(T, NKV, HD) * torch.logspace(-3, 2, T).view(T, 1, 1)
            v = torch.randn(T, NKV, HD)
            k[:, :, :5] = 0
            src = lambda X: ttnn.from_torch(torch.nn.functional.pad(X, (0, 0, 0, 32 - NKV)).view(1, T, 32, HD),
                                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.arange(P0, P0 + T, dtype=torch.int32), dtype=ttnn.int32), posd)
            W(src(k), src(v), kc, vc, posd, ptd)
            pad32 = lambda X: torch.nn.functional.pad(X.permute(1, 0, 2), (0, 0, 0, 32 - T))  # [NKV, 32, HD]
            Kref[P0:P0 + T] = bfp8_round(pad32(k))[:, :T].permute(1, 0, 2)
            Vref[P0:P0 + T] = bfp8_round(pad32(v))[:, :T].permute(1, 0, 2)
    Kd, Vd = logical(kc), logical(vc)
    for name, d, r in (("K", Kd, Kref), ("V", Vd, Vref)):
        diff = (d != r).sum().item()
        print(f"  {name}: mismatching values {diff} / {r.numel()}", flush=True)
        ok &= diff == 0

    print("2. GQA attention on bfp8 caches", flush=True)
    K = torch.randn(maxseq, NKV, HD)
    V = torch.randn(maxseq, NKV, HD)

    def pg(X):
        L = X.view(nb, BS, NKV, HD).permute(0, 2, 1, 3).contiguous()
        P = torch.empty_like(L)
        P[perm] = L
        return P

    kc8 = ttnn.from_torch(pg(K), dtype=B8, layout=ttnn.TILE_LAYOUT, device=dev)
    vc8 = ttnn.from_torch(pg(V), dtype=B8, layout=ttnn.TILE_LAYOUT, device=dev)
    K8, V8 = logical(kc8), logical(vc8)
    rep = NH // NKV
    for T in (4, 32):
        A = GQAAttention(dev, T, nblocks=nb, kv_dtype=B8)
        posd = ttnn.from_torch(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        for P0 in ([0, 37, 1000, 4093, maxseq - 8] if T == 4 else [0, 1024, maxseq - 64]):
            if P0 + T > maxseq:
                continue
            q = torch.randn(T, NH, HD) * 2
            qs = (q / 16).view(1, T, NH, HD)
            if T == 32:
                qs = qs.permute(0, 2, 1, 3).contiguous()
            qd = ttnn.from_torch(qs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.arange(P0, P0 + T, dtype=torch.int32), dtype=ttnn.int32), posd)
            got = ttnn.to_torch(A(qd, kc8, vc8, posd, ptd)).float()[0, 0, :T].view(T, NH, HD)
            qb = q.bfloat16().float()
            r = torch.zeros(T, NH, HD)
            for t in range(T):
                n = P0 + t + 1
                for h in range(NH):
                    s = (qb[t, h] @ K8[:n, h // rep].T) / 16
                    r[t, h] = torch.softmax(s, -1) @ V8[:n, h // rep]
            p = pcc(got, r)
            print(f"  T={T} P0={P0}: pcc {p:.5f} maxerr {(got - r).abs().max():.4f} nan {torch.isnan(got).sum().item()}",
                  flush=True)
            ok &= p > 0.999
finally:
    ttnn.close_device(dev)
print("ALL OK" if ok else "FAILED")
