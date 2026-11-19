"""GDN decode: correctness vs HF torch recurrent reference (random weights) and traced timing."""
import sys
import time

import torch
import torch.nn.functional as F
import ttnn

from qe.gdn import GDNDecode, GDNWeights

CFG = dict(linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128, linear_value_head_dim=128,
           linear_conv_kernel_dim=4)
H = 5120


def ref_step(w, x, conv_state, rec):
    """HF math for one token. x [H]; conv_state [C, K] (oldest..newest); rec [Nv,Dk,Dv]."""
    Nk, Nv, Dk, Dv = 16, 48, 128, 128
    kd, vd = Nk * Dk, Nv * Dv
    qkv = w.qkv.float() @ x
    z = w.z.float() @ x
    a = w.a.float() @ x
    b = w.b.float() @ x
    cs = torch.cat([conv_state[:, 1:], qkv[:, None]], dim=1)
    c = F.silu((cs * w.conv.squeeze(1).float()).sum(-1))
    q, k, v = c[:kd].view(Nk, Dk), c[kd:2 * kd].view(Nk, Dk), c[2 * kd:].view(Nv, Dv)
    q = q.repeat_interleave(3, 0)
    k = k.repeat_interleave(3, 0)
    q = F.normalize(q, dim=-1, eps=1e-6) * Dk**-0.5
    k = F.normalize(k, dim=-1, eps=1e-6)
    beta = b.sigmoid()
    g = -w.A_log.float().exp() * F.softplus(a + w.dt_bias.float())
    rec = rec * g.exp()[:, None, None]
    kv = (rec * k[:, :, None]).sum(1)
    delta = (v - kv) * beta[:, None]
    rec = rec + k[:, :, None] * delta[:, None, :]
    o = (rec * q[:, :, None]).sum(1)  # [Nv, Dv]
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * w.norm.float()
    o = o * F.silu(z.view(Nv, Dv))
    return w.out.float() @ o.reshape(-1), cs, rec


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    qkvd = 2 * 2048 + 6144
    s = lambda *sh: torch.randn(*sh) * (sh[-1] ** -0.5)
    sd = {
        "in_proj_qkv.weight": s(qkvd, H), "in_proj_z.weight": s(6144, H), "in_proj_a.weight": s(48, H),
        "in_proj_b.weight": s(48, H), "conv1d.weight": torch.randn(qkvd, 1, 4) * 0.5,
        "dt_bias": torch.randn(48) * 0.5, "A_log": torch.rand(48) * 2, "norm.weight": 1 + 0.1 * torch.randn(128),
        "out_proj.weight": s(H, 6144),
    }
    sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
    w = GDNWeights(sd, "")
    dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[sys.argv[1] if len(sys.argv) > 1 else "bfp8"]
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    try:
        g = GDNDecode(dev, w, CFG, wdtype=dtype)
        conv = torch.zeros(qkvd, 4)
        rec = torch.zeros(48, 128, 128)
        for step in range(4):
            x = torch.randn(H)
            ref, conv, rec = ref_step(w, x, conv, rec)
            xt = ttnn.from_torch(F.pad(x.view(1, 1, 1, H), (0, 0, 0, 31)).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            out = g.forward_decode_fused(xt) if "--fused" in sys.argv else g.forward_decode(xt)
            got = ttnn.to_torch(out)[0, 0, 0, :H]
            st = ttnn.to_torch(g.rec_state)[0]
            print(f"step {step}: out pcc {pcc(ref, got):.5f}  state pcc {pcc(rec, st):.5f}", flush=True)
        # timing
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        fwd = g.forward_decode_fused if "--fused" in sys.argv else g.forward_decode
        for _ in range(20):
            o = fwd(xt)
            ttnn.deallocate(o)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        print(f"GDN decode layer: {(time.perf_counter() - t0) / 20 * 1e6:.1f} us")
        ttnn.release_trace(dev, tid)
        return
        qkv, z, a, b = g.st_proj(xt)
        conv = g.st_conv(qkv)
        q, k, v, beta, gg = g.st_prep(conv, a, b)
        o = g.st_rec(q, k, v, beta, gg)
        stages = {
            "proj(matmul+slices)": lambda: g.st_proj(xt),
            "matmul_in only": lambda: g.w_in(xt),
            "conv": lambda: g.st_conv(qkv),
            "prep": lambda: g.st_prep(conv, a, b),
            "rec": lambda: g.st_rec(q, k, v, beta, gg),
            "post(norm,gate,matmul)": lambda: g.st_post(o, z),
            "matmul_out only": lambda: g.w_out(ttnn.pad(ttnn.reshape(o, (1, 1, 1, 6144)), [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)),
        }
        for name, fn in stages.items():
            fn()
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(20):
                fn()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            print(f"  {name:24s} {(time.perf_counter() - t0) / 20 * 1e6:7.1f} us", flush=True)
            ttnn.release_trace(dev, tid)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
