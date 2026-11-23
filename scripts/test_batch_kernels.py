"""Unit test: the lane (batched decode) variants of the custom kernels vs torch references.
  python scripts/test_batch_kernels.py [lanes ...]     (default: 2 4 8)
GDN fused kernel (each lane of 4 rows from its own state slot), GQA attention (each lane its own positions and page
table) and the KV write kernel (same)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys

import torch
import torch.nn.functional as F
import ttnn

from tsugumi.gdn_kernel import GDNFusedKernel
from tsugumi.gqa_attn import GQAAttention
from tsugumi.kv_kernel import KVWriteKernel

NK, NV, D = 16, 48, 128
KD, VD = NK * D, NV * D
QKV = 2 * KD + VD
SEG = 4
NH, NKV, HD, BS = 24, 4, 256, 64


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def gdn_ref(x, z, eg, beta, S, w):
    T = x.shape[0]
    q = (F.normalize(x[:, :KD].view(T, NK, D), dim=-1, eps=1e-6) * D**-0.5).repeat_interleave(3, 1)
    k = F.normalize(x[:, KD:2 * KD].view(T, NK, D), dim=-1, eps=1e-6).repeat_interleave(3, 1)
    v = x[:, 2 * KD:].view(T, NV, D)
    outs, snaps = [], []
    for t in range(T):
        S = S * eg[t][:, None, None]
        kv = torch.einsum("hk,hkv->hv", k[t], S)
        S = S + k[t][:, :, None] * ((v[t] - kv) * beta[t][:, None])[:, None, :]
        snaps.append(S)
        o = torch.einsum("hk,hkv->hv", q[t], S)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        outs.append((o * F.silu(z[t].view(NV, D))).reshape(-1))
    return torch.stack(outs), snaps


def test_gdn(dev, NU, groups=1):
    T = NU * SEG
    kern = GDNFusedKernel(dev, NV, NK, T, QKV, KD, seg=SEG, groups=groups)
    w = 1 + 0.2 * torch.randn(D)
    C = GDNFusedKernel.make_consts(dev, w, T)
    nslot = T
    S0 = [torch.randn(NV, D, D) * 0.05 for _ in range(NU)]
    snap = torch.zeros(1, nslot * NV, D, D)
    # lane u starts from slot u*SEG + (u % SEG) (as after a step that committed u % SEG + 1 rows)
    slots = [u * SEG + (u % SEG) for u in range(NU)]
    for u in range(NU):
        snap[0, slots[u] * NV:(slots[u] + 1) * NV] = S0[u]
    St = ttnn.from_torch(snap, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    IDX = GDNFusedKernel.make_idx(dev, 0)
    GDNFusedKernel.set_idx(IDX, slots)
    x = torch.randn(T, QKV).bfloat16().float()
    z = torch.randn(T, VD).bfloat16().float()
    eg = torch.exp(-torch.rand(T, NV) * 0.5)
    beta = torch.rand(T, NV)
    X = torch.zeros(1, 1, 32, QKV)
    X[0, 0, :T] = x
    Y = torch.zeros(1, 1, 32, QKV + VD + 96)
    Y[0, 0, :T, QKV:QKV + VD] = z
    GB = torch.zeros(1, 1, 32, 96)
    GB[0, 0, :T, :NV] = eg
    GB[0, 0, :T, NV:] = beta
    L1 = ttnn.L1_MEMORY_CONFIG
    Xt = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    Yt = ttnn.from_torch(Y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
    GBt = ttnn.from_torch(GB, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.to_torch(kern(Xt, Yt, GBt, GBt, C, St, IDX))[0, 0].float()
    pad = out[T:].abs().max().item() if T < 32 else 0.0
    if pad != 0 or torch.isnan(out).any():
        print(f"  GDN lanes={NU} groups={groups}: padding rows max {pad}, nan {torch.isnan(out).sum().item()}")
        return False
    sn = ttnn.to_torch(St)[0].float()
    ok = True
    for u in range(NU):
        r = slice(u * SEG, (u + 1) * SEG)
        ref, snaps = gdn_ref(x[r], z[r], eg[r], beta[r], S0[u], w)
        p_o = pcc(out[r], ref)
        p_s = min(pcc(sn[(u * SEG + t) * NV:(u * SEG + t + 1) * NV], snaps[t]) for t in range(SEG))
        ok &= p_o > 0.999 and p_s > 0.9999
        print(f"  GDN lanes={NU} groups={groups} lane {u}: out pcc {p_o:.6f}  snapshots min pcc {p_s:.6f}")
    return ok


def gqa_ref(q, K, V, P0, T):
    out = torch.zeros(T, NH, HD)
    rep = NH // NKV
    for t in range(T):
        n = P0 + t + 1
        for h in range(NH):
            s = (q[t, h] @ K[:n, h // rep].T) / 16
            out[t, h] = torch.softmax(s, -1) @ V[:n, h // rep]
    return out


def test_gqa_kv(dev, NU, maxseq=4096):
    nb = maxseq // BS
    T = SEG
    kvd = ttnn.bfloat8_b
    # one shared physical pool of nb blocks; lane u owns the blocks perm[u::NU] (logical block b -> perm[u + b*NU])
    lanes_blocks = nb // NU
    perm = torch.randperm(nb)
    pts = [perm[u::NU][:lanes_blocks] for u in range(NU)]
    Ks = [torch.randn(lanes_blocks * BS, NKV, HD) for _ in range(NU)]
    Vs = [torch.randn(lanes_blocks * BS, NKV, HD) for _ in range(NU)]
    Kp = torch.zeros(nb, NKV, BS, HD)
    Vp = torch.zeros(nb, NKV, BS, HD)
    for u in range(NU):
        for b in range(lanes_blocks):
            Kp[pts[u][b]] = Ks[u][b * BS:(b + 1) * BS].permute(1, 0, 2)
            Vp[pts[u][b]] = Vs[u][b * BS:(b + 1) * BS].permute(1, 0, 2)
    kc = ttnn.from_torch(Kp, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=dev)
    vc = ttnn.from_torch(Vp, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=dev)
    # the reference sees the cache as stored (bfp8 rounding): read it back
    Kq = ttnn.to_torch(kc).float()
    Vq = ttnn.to_torch(vc).float()

    def lane_kv(u, src):
        return torch.cat([src[pts[u][b]].permute(1, 0, 2) for b in range(lanes_blocks)], 0)

    ptd = []
    for u in range(NU):
        row = torch.zeros(nb, dtype=torch.int32)
        row[:lanes_blocks] = pts[u].to(torch.int32)
        ptd.append(ttnn.from_torch(row.view(1, nb), dtype=ttnn.int32, device=dev))
    P0s = [min(37 + 997 * u, lanes_blocks * BS - T - 1) for u in range(NU)]
    posd = [ttnn.from_torch(torch.arange(p, p + T, dtype=torch.int32), dtype=ttnn.int32, device=dev) for p in P0s]
    ok = True
    # ---- KV write: new rows at P0..P0+T-1 of every lane ----
    kvw = KVWriteKernel(dev, T, NKV, HD, BS, nb, kvd, lanes=NU)
    knew = torch.randn(NU * T, NKV, HD)
    vnew = torch.randn(NU * T, NKV, HD)
    pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, 32 - NKV)).view(1, NU * T, 32, HD)
    ks = ttnn.from_torch(pad(knew), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vs = ttnn.from_torch(pad(vnew), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ksl = ttnn.slice(ks, (0, 0, 0, 0), (1, NU * T, NKV, HD))
    vsl = ttnn.slice(vs, (0, 0, 0, 0), (1, NU * T, NKV, HD))
    kvw(ksl, vsl, kc, vc, posd, ptd)
    Kq2 = ttnn.to_torch(kc).float()
    Vq2 = ttnn.to_torch(vc).float()
    for u in range(NU):
        K2, V2 = lane_kv(u, Kq2), lane_kv(u, Vq2)
        r = slice(P0s[u], P0s[u] + T)
        e = max((K2[r] - knew[u * T:(u + 1) * T]).abs().max().item(),
                (V2[r] - vnew[u * T:(u + 1) * T]).abs().max().item())
        # the other rows are unchanged
        K1 = lane_kv(u, Kq)
        same = torch.equal(torch.cat([K1[:P0s[u]], K1[P0s[u] + T:]]), torch.cat([K2[:P0s[u]], K2[P0s[u] + T:]]))
        ok &= e < 0.1 and same
        print(f"  KV write lanes={NU} lane {u} (pos {P0s[u]}): max err {e:.4f} (bfp8), other rows unchanged {same}")
    # ---- GQA ----
    A = GQAAttention(dev, T, nblocks=nb, kv_dtype=kvd, lanes=NU)
    q = torch.randn(NU * T, NH, HD) * 2
    qd = ttnn.from_torch((q / 16).view(1, NU * T, NH, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(A(qd, kc, vc, posd, ptd)).float()[0, 0]
    for u in range(NU):
        r = gqa_ref(q[u * T:(u + 1) * T].bfloat16().float(), lane_kv(u, Kq2), lane_kv(u, Vq2), P0s[u], T)
        got = o[u * T:(u + 1) * T].view(T, NH, HD)
        p = pcc(got, r)
        ok &= p > 0.999
        print(f"  GQA lanes={NU} lane {u} (pos {P0s[u]}, splits {A.S}, DT {A.DT}): pcc {p:.6f}")
    return ok


def main():
    lanes = [int(a) for a in sys.argv[1:]] or [2, 4, 8]
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        ok = True
        for NU in [1] + lanes:
            ok &= test_gdn(dev, NU)
            if NU >= 2:
                ok &= test_gdn(dev, NU, groups=2)
            ok &= test_gqa_kv(dev, NU)
        print("ALL OK" if ok else "FAILED")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
