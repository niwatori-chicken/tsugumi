"""Large-T attention (LargePrefill._attn) vs a torch fp32 reference with the dequantized weights (layer 3).
  python test_attn_large.py [T] [N]"""
import json, sys, torch, ttnn
import qe.paths as qe_paths
from qe.attn import AttnDecode
from qe.model import LazySD, Qwen35Engine
from qe.prefill import LargePrefill

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
try:
    tc = json.load(open(f"{qe_paths.MODEL}/config.json"))["text_config"]
    sd = LazySD(qe_paths.MODEL)
    A = AttnDecode(dev, sd, "model.language_model.layers.3.self_attn.", tc, 4096, wdtype=ttnn.bfloat4_b,
                   cache_prefix=f"{qe_paths.CACHE}/bfp4/L3.attn")
    A.init_spec(4)
    e = Qwen35Engine.__new__(Qwen35Engine)
    e.HD, e.rope_dim, e.theta, e.H = 256, 64, 1e7, 5120
    e.inv_freq = 1.0 / (e.theta ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    lp = LargePrefill.__new__(LargePrefill)
    lp.eng, lp.dev, lp.BS = e, dev, 64
    g = dev.compute_with_storage_grid_size(); lp.grid = (g.x, g.y)
    lp.sdpa_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                  fp32_dest_acc_en=True, packer_l1_acc=False)
    nbk = 4096 // 64
    pt_full = ttnn.from_torch(torch.arange(nbk, dtype=torch.int32).view(1, nbk), dtype=ttnn.int32, device=dev)
    torch.manual_seed(0)
    xs = torch.randn(N, 5120).bfloat16().float()
    outs = []
    for pos in range(0, N, T):
        up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        cs = [e._rope_rows(pos + i) for i in range(T)]
        cos = up(torch.cat([c for c, _ in cs], 2)); sin = up(torch.cat([s for _, s in cs], 2))
        pt = ttnn.from_torch(torch.arange(pos // 64, pos // 64 + T // 64, dtype=torch.int32).view(1, -1), dtype=ttnn.int32, device=dev)
        o = lp._attn(A, up(xs[pos:pos + T].view(1, 1, T, 5120)), T, pos, cos, sin, pt, pt_full, lp._sdpa_cfg(T, pos))
        outs.append(ttnn.to_torch(o).float().view(T, -1)[:, :5120])
    got = torch.cat(outs)
    # reference
    W = ttnn.to_torch(A.wqkvg.w).float()[:, : (2 * 24 + 8) * 256]
    Wo = ttnn.to_torch(A.wo.w).float()[:, :5120]
    y = xs @ W
    q = y[:, :6144].view(N, 24, 256); k = y[:, 6144:7168].view(N, 4, 256); v = y[:, 7168:8192].view(N, 4, 256)
    gate = y[:, 8192:14336]
    qn = ttnn.to_torch(A.q_norm).float().reshape(-1)[:256]; kn = ttnn.to_torch(A.k_norm).float().reshape(-1)[:256]
    rms = lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    q, k = rms(q, qn), rms(k, kn)
    cs = [e._rope_rows(p) for p in range(N)]
    cos = torch.cat([c for c, _ in cs], 2).view(N, 1, 256); sin = torch.cat([s for _, s in cs], 2).view(N, 1, 256)
    def rot(x):
        r = torch.zeros_like(x); h = 32
        r[..., :h] = -x[..., h:64]; r[..., h:64] = x[..., :h]
        return r
    q = q * cos + rot(q) * sin; k = k * cos + rot(k) * sin
    # elementwise: device q/k (after norm + rope) and the K cache vs the reference
    for pos in range(0, N, T):
        cs_ = [e._rope_rows(pos + i) for i in range(T)]
        up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        qd, kd, vd, gd = lp._qkv_heads(A, up(xs[pos:pos + T].view(1, 1, T, 5120)), T, up(torch.cat([c for c, _ in cs_], 2)), up(torch.cat([s_ for _, s_ in cs_], 2)))
        qh = ttnn.to_torch(qd).float()[0].permute(1, 0, 2); kh = ttnn.to_torch(kd).float()[0].permute(1, 0, 2)
        for nm, a_, b_ in (("q", qh, q[pos:pos + T]), ("k", kh, k[pos:pos + T])):
            err = (a_ - b_).abs().amax((1, 2))
            print(f"chunk@{pos} {nm}: maxerr {err.max():.3f} (row {int(err.argmax())})  ref max {b_.abs().max():.2f}  per-64-row max err {[round(float(err[i:i+64].max()), 2) for i in range(0, T, 256)]}", flush=True)
    kc = ttnn.to_torch(A.kp).float()[: N // 64].permute(1, 0, 2, 3).reshape(4, N, 256).permute(1, 0, 2)
    err = (kc - k).abs().amax((1, 2))
    print(f"K cache vs ref: maxerr {err.max():.3f} at pos {int(err.argmax())}", flush=True)
    kk = k.repeat_interleave(6, 1); vv = v.repeat_interleave(6, 1)
    s = torch.einsum("qhd,khd->hqk", q, kk) / 16
    s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool), 1), float("-inf"))
    o = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv).reshape(N, 6144)
    ref = (o * torch.sigmoid(gate)) @ Wo
    for a in range(0, N, 256):
        gg, rr = got[a:a + 256], ref[a:a + 256]
        p = torch.corrcoef(torch.stack([gg.flatten(), rr.flatten()]))[0, 1].item()
        print(f"rows {a:5d}-{a + 255}: pcc {p:.5f}  maxerr {(gg - rr).abs().max():.3f}  ref max {rr.abs().max():.2f}", flush=True)
finally:
    ttnn.close_device(dev)
