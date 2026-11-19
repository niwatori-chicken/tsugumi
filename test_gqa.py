"""GQAAttention vs torch reference.  python test_gqa.py"""
import sys, time, torch, ttnn
from qe.gqa_attn import GQAAttention
NH, NKV, HD, BS = 24, 4, 256, 64
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
def ref(q, K, V, P0, T):
    out = torch.zeros(T, NH, HD)
    rep = NH // NKV
    for t in range(T):
        n = P0 + t + 1
        for h in range(NH):
            k, v = K[:n, h // rep], V[:n, h // rep]
            s = (q[t, h] @ k.T) / 16
            out[t, h] = torch.softmax(s, -1) @ v
    return out
def pcc(a, b): return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
try:
    maxseq = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    nb = maxseq // BS
    torch.manual_seed(0)
    K = torch.randn(maxseq, NKV, HD); V = torch.randn(maxseq, NKV, HD)
    pg = lambda X: X.view(nb, BS, NKV, HD).permute(0, 2, 1, 3).contiguous()
    kc = ttnn.from_torch(pg(K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vc = ttnn.from_torch(pg(V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    Kb, Vb = K.bfloat16().float(), V.bfloat16().float()
    for T in (4, 32):
        A = GQAAttention(dev, T, splits=int(sys.argv[2]) if len(sys.argv) > 2 else None)
        posd = ttnn.from_torch(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        for P0 in ([0, 5, 37, 1000, 4093, 8000, 16000] if T == 4 else [0, 32, 1024, 4064, 16352]):
            if P0 + T > maxseq: continue
            q = torch.randn(T, NH, HD) * 2
            qs = (q / 16).view(1, T, NH, HD)
            if T == 32: qs = qs.permute(0, 2, 1, 3).contiguous()
            qd = ttnn.from_torch(qs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.arange(P0, P0 + T, dtype=torch.int32), dtype=ttnn.int32), posd)
            o = A(qd, kc, vc, posd)
            got = ttnn.to_torch(o).float()[0, 0, :T].view(T, NH, HD)
            r = ref(q.bfloat16().float(), Kb, Vb, P0, T)
            pad = ttnn.to_torch(o).float()[0, 0, T:].abs().max().item() if T < 32 else 0
            print(f"T={T} P0={P0}: pcc {pcc(got, r):.5f} maxerr {(got - r).abs().max():.4f} nan {torch.isnan(got).sum().item()} pad {pad}", flush=True)
finally:
    ttnn.close_device(dev)
