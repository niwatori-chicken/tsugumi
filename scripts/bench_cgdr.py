"""ttnn.transformer.chunk_gated_delta_rule for the Qwen3.8-27B GDN shape (Nk 16, Nv 48, Dk=Dv 128):
accuracy vs a token-by-token torch recurrence (with an initial state) and trace timing per T.
  python scripts/bench_cgdr.py [T ...]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import sys, time, torch, ttnn

Nk, Nv, D = 16, 48, 128
G = Nv // Nk
Ts = [int(a) for a in sys.argv[1:]] or [32, 64, 128, 256, 512, 1024]


def ref(q, k, v, g, beta, s0, scale):
    """q,k [T,Nk,D] (already l2-normed), v [T,Nv,D], g/beta [T,Nv], s0 [Nv,D,D] -> o [T,Nv,D], S."""
    q = q.repeat_interleave(G, 1).double() * scale
    k = k.repeat_interleave(G, 1).double()
    v, g, beta, S = v.double(), g.double(), beta.double(), s0.double().clone()
    o = torch.empty_like(v)
    for t in range(q.shape[0]):
        S = S * g[t].exp()[:, None, None]
        kv = torch.einsum("hk,hkv->hv", k[t], S)
        S = S + torch.einsum("hk,hv->hkv", k[t], (v[t] - kv) * beta[t][:, None])
        o[t] = torch.einsum("hk,hkv->hv", q[t], S)
    return o.float(), S.float()


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def consts(dev, c=32):
    ii, jj = torch.arange(32)[:, None], torch.arange(32)[None]
    lo_i, lo_j = ii < 16, jj < 16
    m = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], 1)
    up = lambda t: ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    return up(torch.eye(c)), up(torch.tril(torch.ones(c, c))), up(torch.ones(c, c)), up(m)


dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
try:
    cs = consts(dev)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for T in Ts:
        torch.manual_seed(0)
        nrm = lambda x: x / x.norm(dim=-1, keepdim=True)
        q = nrm(torch.randn(T, Nk, D)).bfloat16().float()
        k = nrm(torch.randn(T, Nk, D)).bfloat16().float()
        v = torch.randn(T, Nv, D).bfloat16().float()
        beta = torch.sigmoid(torch.randn(T, Nv))
        g = -torch.nn.functional.softplus(torch.randn(T, Nv)) * 0.3
        s0 = 0.1 * torch.randn(Nv, D, D)
        scale = D ** -0.5
        tq, tk, tv = up(q[None], ttnn.bfloat16), up(k[None], ttnn.bfloat16), up(v[None], ttnn.bfloat16)
        tg, tb = up(g[None], ttnn.float32), up(beta[None], ttnn.float32)
        ts = up(s0[None], ttnn.float32)
        f = lambda: ttnn.transformer.chunk_gated_delta_rule(
            tq, tk, tv, tg, tb, scale=scale, initial_state=ts, output_final_state=True, chunk_size=32,
            output_head_major=True, eye=cs[0], tril=cs[1], ones=cs[2], masks=cs[3])
        o, fs = f()
        o_t = ttnn.to_torch(o).float().reshape(Nv, T, D).transpose(0, 1)
        s_t = ttnn.to_torch(fs).float().reshape(Nv, D, D)
        if T <= 256:
            ro, rs = ref(q, k, v, g, beta, s0, scale)
            acc = f"pcc o {pcc(o_t, ro):.6f} S {pcc(s_t, rs):.6f} maxerr o {(o_t - ro).abs().max():.2e}"
        else:
            acc = ""
        ttnn.deallocate(o); ttnn.deallocate(fs)
        # timing (trace)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        outs = [f() for _ in range(4)]
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        n = 10
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / n / 4 * 1e3
        ttnn.release_trace(dev, tid)
        for a, b in outs:
            ttnn.deallocate(a); ttnn.deallocate(b)
        print(f"T={T:5d}: {ms:7.3f} ms/layer  ({ms / T * 1e3:6.1f} us/token, x48 = {ms * 48:6.1f} ms)  {acc}", flush=True)
        for t in (tq, tk, tv, tg, tb, ts):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(dev)
