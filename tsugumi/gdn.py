"""Gated DeltaNet layer (Qwen3.5/3.8 linear_attention) — single-device decode.

Ported from qwen36/tt/gdn/tp.py forward_decode (TP=1, no all-reduce), using DecodeLinear.
Math reference: transformers Qwen3_5GatedDeltaNet (recurrent path).
"""
import torch
import ttnn

from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)

from .gdn_kernel import GDNFusedKernel
from .ops import DecodeLinear, has_cache

_L1 = ttnn.L1_MEMORY_CONFIG


class GDNWeights:
    """Host-side (torch) GDN weights in HF orientation."""

    def __init__(self, sd, prefix):
        self._sd, self._p = sd, prefix

    _names = dict(qkv="in_proj_qkv.weight", z="in_proj_z.weight", a="in_proj_a.weight", b="in_proj_b.weight",
                  conv="conv1d.weight", dt_bias="dt_bias", A_log="A_log", norm="norm.weight", out="out_proj.weight")

    def __getattr__(self, n):  # lazy: big weights are only read when the cache is missing
        if n in GDNWeights._names:
            return self._sd[self._p + GDNWeights._names[n]]
        raise AttributeError(n)


class GDNDecode:
    def __init__(self, dev, w: GDNWeights, cfg, wdtype=ttnn.bfloat4_b, out_dtype=None, cache_prefix=None):
        self.dev = dev
        self.Nk = cfg["linear_num_key_heads"]
        self.Nv = cfg["linear_num_value_heads"]
        self.Dk = cfg["linear_key_head_dim"]
        self.Dv = cfg["linear_value_head_dim"]
        self.Kc = cfg["linear_conv_kernel_dim"]
        self.kd = self.Nk * self.Dk
        self.vd = self.Nv * self.Dv
        self.qkv_dim = 2 * self.kd + self.vd
        self.scale = self.Dk**-0.5
        H = cfg.get("hidden_size", 5120)
        cp = (lambda s: f"{cache_prefix}.{s}") if cache_prefix else (lambda s: None)
        nin = self.qkv_dim + self.vd + 2 * self.Nv
        win = wout = None
        if not has_cache(cp("in"), wdtype):
            win = torch.cat([w.qkv, w.z, w.a, w.b], dim=0).T.contiguous().to(torch.bfloat16)  # [H, qkv+z+2Nv]
        if not has_cache(cp("out"), out_dtype or wdtype):
            wout = w.out.T.contiguous().to(torch.bfloat16)
        self.w_in = DecodeLinear(dev, win, wdtype, cache_file=cp("in"), name="gdn_in", shape=(H, nin))
        self.w_out = DecodeLinear(dev, wout, out_dtype or wdtype, cache_file=cp("out"), name="gdn_out",
                                  shape=(self.vd, H))
        taps = w.conv.squeeze(1).float()  # [C, K]

        def row(t, dt=ttnn.bfloat16):
            return ttnn.from_torch(
                t.reshape(1, 1, -1).to(torch.float32 if dt == ttnn.float32 else torch.bfloat16),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
            )

        self.taps = [row(taps[:, j]) for j in range(self.Kc)]
        self.dt_bias = row(w.dt_bias.float())
        self.neg_exp_A = row(-w.A_log.float().exp())
        # fused-kernel path
        self.fused = GDNFusedKernel(dev, self.Nv, self.Nk, 1, self.qkv_dim, self.kd)
        self._norm_w_host = w.norm.float().clone()
        self.fused_consts = GDNFusedKernel.make_consts(dev, self._norm_w_host, 1)
        nv = self.Nv
        self.ab_bias = row(torch.cat([w.dt_bias.float(), torch.zeros(nv)]), ttnn.float32)
        self.ab_coef = row(torch.cat([-w.A_log.float().exp(), torch.zeros(nv)]), ttnn.float32)
        self.norm_w = ttnn.from_torch(
            w.norm.reshape(1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev
        )
        self.reset_state()

    def reset_state(self):
        z = lambda shape, dt=ttnn.bfloat16: ttnn.from_torch(
            torch.zeros(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.dev
        )
        self.conv_states = [z((1, 1, self.qkv_dim)) for _ in range(self.Kc)]
        self.conv_states4 = [z((1, 1, 32, self.qkv_dim)) for _ in range(self.Kc)]
        self.rec_state = z((1, self.Nv, self.Dk, self.Dv), ttnn.float32)

    def set_state_from_torch(self, conv_state, rec_state):
        """conv_state: [qkv_dim, Kc] (HF layout, last column newest), rec_state: [Nv, Dk, Dv]."""
        for j in range(self.Kc):
            t = ttnn.from_torch(
                conv_state[:, j].reshape(1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(t, self.conv_states[j])
            t4 = torch.zeros(1, 1, 32, self.qkv_dim, dtype=torch.bfloat16)
            t4[0, 0, 0] = conv_state[:, j].to(torch.bfloat16)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(t4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                                            self.conv_states4[j])
        t = ttnn.from_torch(rec_state.reshape(1, self.Nv, self.Dk, self.Dv).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(t, self.rec_state)

    # ---- decode, split in stages so each can be timed on its own ----
    def st_proj(self, x):
        Nv = self.Nv
        y = self.w_in(x, out_mem=_L1)  # [1,1,32,Np]
        y = ttnn.reshape(y, (1, 32, y.shape[-1]))
        y1 = ttnn.slice(y, (0, 0, 0), (1, 1, y.shape[-1]), memory_config=_L1)
        ttnn.deallocate(y)
        qd, zd = self.qkv_dim, self.qkv_dim + self.vd
        qkv = ttnn.slice(y1, (0, 0, 0), (1, 1, qd), memory_config=_L1)
        z = ttnn.slice(y1, (0, 0, qd), (1, 1, zd), memory_config=_L1)
        a = ttnn.slice(y1, (0, 0, zd), (1, 1, zd + Nv), memory_config=_L1)
        b = ttnn.slice(y1, (0, 0, zd + Nv), (1, 1, zd + 2 * Nv), memory_config=_L1)
        ttnn.deallocate(y1)
        return qkv, z, a, b

    def st_conv(self, qkv):
        st = self.conv_states
        for j in range(self.Kc - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.Kc - 1])
        conv = ttnn.multiply(st[0], self.taps[0], memory_config=_L1)
        for j in range(1, self.Kc):
            conv = ttnn.mac(st[j], self.taps[j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)
        return conv

    def st_prep(self, conv, a, b):
        Nk, Nv, Dk, Dv, kd = self.Nk, self.Nv, self.Dk, self.Dv, self.kd
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, 1, kd)), (1, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, 1, 2 * kd)), (1, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, 1, self.qkv_dim)), (1, Nv, Dv))
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        q = ttnn.reshape(q, (1, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(k, (1, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (1, 1, Nv, Dv), memory_config=_L1)
        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (1, 1, Nv))
        sp = ttnn.add(a, self.dt_bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])
        g = ttnn.multiply(self.neg_exp_A, sp, memory_config=_L1)
        g = ttnn.reshape(g, (1, 1, Nv))
        return q, k, v, beta, g

    def st_rec(self, q, k, v, beta, g):
        o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
            q, k, v, beta, g, scale=self.scale, initial_state=self.rec_state, device=self.dev, high_precision=True
        )
        ttnn.copy(new_rec, self.rec_state)
        ttnn.deallocate(new_rec)
        return o

    def st_post(self, o, z):
        Nv, Dv = self.Nv, self.Dv
        o = ttnn.reshape(o, (1, Nv, Dv))
        o = ttnn.rms_norm(o, weight=self.norm_w, epsilon=1e-6, memory_config=_L1)
        o = ttnn.reshape(o, (1, 1, self.vd))
        gated = ttnn.multiply(o, ttnn.silu(z, memory_config=_L1), memory_config=_L1)
        gated = ttnn.reshape(gated, (1, 1, 1, self.vd))
        gated = ttnn.pad(gated, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        return self.w_out(gated, out_mem=_L1)

    # ---------------- T-row (verify / chunked prefill) path ----------------
    def init_spec(self, T, snap_all=True, slots=None):
        """T-row mode. First call (speculative T) allocates conv history/prev-block rows and T state snapshots;
        later calls (e.g. a 32-row prefill with snap_all=False) only add a kernel for that T."""
        z = lambda shape, dt=ttnn.bfloat16: ttnn.from_torch(torch.zeros(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                                           device=self.dev)
        if not hasattr(self, "kern_T"):
            self.Hc = z((1, 1, 32, self.qkv_dim))
            self.Bp = z((1, 1, 32, self.qkv_dim))
            # state snapshots, slot-major (slot * Nv + head); shared by every T-row mode, so sized for the largest
            # verify T (a T=4 step after a T=8 step reads slot a_prev - 1 <= 7)
            self.snap = z((1, (slots or T) * self.Nv, self.Dk, self.Dv), ttnn.float32)
            self.kern_T, self.consts_by_T = {}, {}
        self.kern_T[T] = GDNFusedKernel(self.dev, self.Nv, self.Nk, T, self.qkv_dim, self.kd, snap_all=snap_all)
        self.consts_by_T[T] = GDNFusedKernel.make_consts(self.dev, self._norm_w_host, T)

    def init_batch(self, key, T, seg, max_lanes):
        """Batched decode (tsugumi/batch.py): T = lanes x seg rows. Lane u keeps its conv history in Hc_b rows
        u*3.. / Bp_b rows u*seg.. and its state snapshots in snap_b slots u*seg.. (separate from the single-sequence
        modes, so a prompt can be prefilled while the lanes decode)."""
        z = lambda shape, dt=ttnn.bfloat16: ttnn.from_torch(torch.zeros(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                                           device=self.dev)
        if not hasattr(self, "snap_b"):
            self.Hc_b = z((1, 1, 32, self.qkv_dim))
            self.Bp_b = z((1, 1, 32, self.qkv_dim))
            self.snap_b = z((1, max_lanes * seg * self.Nv, self.Dk, self.Dv), ttnn.float32)
        # 2+ lanes: the lanes in 2 groups on 2 x Nv cores (TSUGUMI_GDN_GROUPS); the recurrence is the per-lane cost
        from .env import env
        g = int(env("GDN_GROUPS", "2"))
        g = g if T // seg >= g and (T // seg) % g == 0 else 1
        self.kern_T[key] = GDNFusedKernel(self.dev, self.Nv, self.Nk, T, self.qkv_dim, self.kd, seg=seg, groups=g)
        self.consts_by_T[key] = GDNFusedKernel.make_consts(self.dev, self._norm_w_host, T)

    def reset_spec(self):
        zero = lambda t: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype, layout=ttnn.TILE_LAYOUT), t)
        zero(self.Hc)
        zero(self.Bp)
        zero(self.snap)

    def forward_spec(self, x, sh):
        """x [1,1,32,H] rows 0..T-1 = tokens. sh: shared step tensors (m_hist, m_shift[4], idx)."""
        Hc, Bp, snap = (self.Hc_b, self.Bp_b, self.snap_b) if sh.batch else (self.Hc, self.Bp, self.snap)
        y = self.w_in(x, out_mem=_L1)
        qd, zd, nv = self.qkv_dim, self.qkv_dim + self.vd, self.Nv
        B = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, qd), memory_config=_L1)
        cat = ttnn.concat([Hc, Bp], dim=2, memory_config=_L1)
        hn = ttnn.matmul(sh.m_hist, cat, memory_config=_L1, compute_kernel_config=sh.sel_ck)
        ttnn.deallocate(cat)
        ttnn.copy(hn, Hc)
        ttnn.deallocate(hn)
        cat = ttnn.concat([Hc, B], dim=2, memory_config=_L1)
        conv = None
        for j in range(self.Kc):
            sj = ttnn.matmul(sh.m_shift[j], cat, memory_config=_L1, compute_kernel_config=sh.sel_ck)
            conv = ttnn.multiply(sj, self.taps[j], memory_config=_L1) if conv is None else ttnn.mac(sj, self.taps[j], conv)
            ttnn.deallocate(sj)
        ttnn.deallocate(cat)
        conv = ttnn.silu(conv, memory_config=_L1)
        ttnn.copy(B, Bp)
        ttnn.deallocate(B)
        ab = ttnn.typecast(ttnn.slice(y, (0, 0, 0, zd), (1, 1, 32, zd + 2 * nv), memory_config=_L1), ttnn.float32)
        u = ttnn.add(ab, self.ab_bias, memory_config=_L1)
        ttnn.deallocate(ab)
        ge = ttnn.exp(ttnn.multiply(ttnn.softplus(u, memory_config=_L1), self.ab_coef, memory_config=_L1),
                      memory_config=_L1)
        be = ttnn.sigmoid(u, memory_config=_L1)
        ttnn.deallocate(u)
        o = self.kern_T[sh.key](conv, y, ge, be, self.consts_by_T[sh.key], snap, sh.idx)
        ttnn.deallocate(conv)
        ttnn.deallocate(ge)
        ttnn.deallocate(be)
        ttnn.deallocate(y)
        out = self.w_out(o, out_mem=_L1)
        ttnn.deallocate(o)
        return out

    def forward_decode_fused(self, x):
        """T=1 decode with the fused post-conv kernel. x [1,1,32,H] (row 0 valid)."""
        y = self.w_in(x, out_mem=_L1)  # [1,1,32,Np] interleaved L1
        qd, zd, nv = self.qkv_dim, self.qkv_dim + self.vd, self.Nv
        qkv = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, qd), memory_config=_L1)
        st = self.conv_states4
        for j in range(self.Kc - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.Kc - 1])
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], self.taps[0], memory_config=_L1)
        for j in range(1, self.Kc):
            conv = ttnn.mac(st[j], self.taps[j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)
        ab = ttnn.typecast(ttnn.slice(y, (0, 0, 0, zd), (1, 1, 32, zd + 2 * nv), memory_config=_L1), ttnn.float32)
        u = ttnn.add(ab, self.ab_bias, memory_config=_L1)
        ttnn.deallocate(ab)
        ge = ttnn.exp(ttnn.multiply(ttnn.softplus(u, memory_config=_L1), self.ab_coef, memory_config=_L1),
                      memory_config=_L1)
        be = ttnn.sigmoid(u, memory_config=_L1)
        ttnn.deallocate(u)
        o = self.fused(conv, y, ge, be, self.fused_consts, self.rec_state)
        ttnn.deallocate(conv)
        ttnn.deallocate(ge)
        ttnn.deallocate(be)
        ttnn.deallocate(y)
        out = self.w_out(o, out_mem=_L1)
        ttnn.deallocate(o)
        return out

    def forward_decode(self, x):
        """x: [1,1,32,H] (row 0 valid). Returns [1,1,32,Np_out] in L1."""
        qkv, z, a, b = self.st_proj(x)
        conv = self.st_conv(qkv)
        q, k, v, beta, g = self.st_prep(conv, a, b)
        o = self.st_rec(q, k, v, beta, g)
        return self.st_post(o, z)
