"""Large-T prompt prefill (hundreds to thousands of rows per step) for Qwen35Engine.

The T-row mode (forward_spec) is built for verify-sized T: DRAM-sharded matmuls that stream the weights
once per step, and a GDN kernel that walks the tokens one by one. For a long prompt that caps out at
~300 tok/s. Here, per chunk of T rows (eager ops, activations in DRAM):
  - matmuls: minimal_matmul on the same bfp4 DRAM-sharded weights (compute-bound from T ~ 1-2K)
  - GDN: causal conv via ttnn.conv1d, recurrence via ttnn.transformer.chunk_gated_delta_rule (chunk-parallel)
  - attention: nlp_create_qkv_heads -> norms/rope -> paged_fill_cache -> chunked SDPA (flash, paged KV)
  - MTP: only its KV is needed for the prompt rows (fc -> norm -> k/v -> fill cache)
On exit it leaves the engine in the state the T-row mode expects after a prefill step:
GDN conv history in Hc rows 0..2, recurrent state in snapshot slot 0, a_prev = 0.
"""
import math
import os
import time

import torch
import ttnn

from .env import env
from .page_copy import PageCopy

DRAM = ttnn.DRAM_MEMORY_CONFIG
_DEBUG = env("PREFILL_DEBUG", "0") == "1"


def _dbg(dev, msg):
    if _DEBUG:
        ttnn.synchronize_device(dev)
        print(f"[prefill-debug] {msg}", flush=True)


class LargePrefill:
    """Create before eng.capture_all(): its constant tensors must not be allocated after a trace capture
    (a replay can overwrite buffers allocated after the capture)."""

    def __init__(self, eng, t_max=1024, t_min=64):
        self.eng, self.dev = eng, eng.dev
        self.t_max, self.t_min = t_max, t_min
        self.BS = 64
        gdn = next(L["mix"] for L in eng.layers if L["type"] == "linear_attention")
        self.Nv, self.Nk, self.Dk, self.Dv = gdn.Nv, gdn.Nk, gdn.Dk, gdn.Dv
        self.qkv_dim, self.vd, self.kd = gdn.qkv_dim, gdn.vd, gdn.kd
        self.Kc = gdn.Kc
        self.state_pages = self.Nv * (self.Dk // 32) * (self.Dv // 32)
        self.pcopy = PageCopy(self.dev, self.state_pages, 4096)
        self.sel_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                       fp32_dest_acc_en=True, packer_l1_acc=False)
        self.fp32_ck = self.sel_ck
        self.norm_ck = self.sel_ck  # default rms_norm config: ~1% error per row
        # SDPA: "hifi4" = q/k chunk 128, HiFi4 (KV-bandwidth bound at long context); "hifi2" = q chunk 256, HiFi2
        # (KV read halved, 22% faster at 32K-123K)
        self.sdpa_mode = env("PREFILL_SDPA", "hifi2")
        fid = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}[self.sdpa_mode]
        self.sdpa_qmax = 256 if self.sdpa_mode == "hifi2" else 128
        self.sdpa_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                                        fp32_dest_acc_en=True, packer_l1_acc=False)
        g = self.dev.compute_with_storage_grid_size()
        self.grid = (g.x, g.y)
        up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.dev)
        # per-head sum / broadcast matrices for the gated RMSNorm in [T, Nv*Dv] layout
        nvp = math.ceil(self.Nv / 32) * 32
        hs = torch.zeros(self.vd, nvp)
        for h in range(self.Nv):
            hs[h * self.Dv:(h + 1) * self.Dv, h] = 1.0 / self.Dv
        self.head_mean = up(hs.view(1, 1, self.vd, nvp))  # 1/Dv entries: exact in bf16
        hb = (hs.T * self.Dv).contiguous()  # [nvp, vd] 0/1 head -> columns
        self.gnorm_w = {}  # per layer: head broadcast with the gated-norm weight folded in [1,1,nvp,vd]
        for i, L in enumerate(eng.layers):
            if L["type"] == "linear_attention":
                m = L["mix"]
                self.gnorm_w[i] = up((hb * m._norm_w_host.float().repeat(self.Nv)[None]).view(1, 1, nvp, self.vd))
        self._shift_by_T = {}
        T = t_min
        while T <= t_max:  # persistent tensors must exist before the engine's traces are captured
            self._shift_matrix(T)
            T *= 2
        self.consts = self._cgdr_consts()
        from .conv_kernel import CausalConvKernel

        self.convk = CausalConvKernel(self.dev, self.qkv_dim)
        self.timing = {}
        self.keep = None  # list: keep each chunk's final-normed hidden (eval)
        # SDPA chunk start as a device tensor: one program for every start position (an int start compiles a new
        # program per position, i.e. after the trace capture on every new conversation turn)
        self.start_t = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)

    # ---------------- helpers ----------------
    def _cgdr_consts(self, c=32):
        ii, jj = torch.arange(32)[:, None], torch.arange(32)[None]
        lo_i, lo_j = ii < 16, jj < 16
        m = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], 1)
        up = lambda t: ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=self.dev)
        return up(torch.eye(c)), up(torch.tril(torch.ones(c, c))), up(torch.ones(c, c)), up(m)

    MM_CK = {}

    @classmethod
    def _mm(cls, x, lin, n=None):
        """minimal_matmul on the decode weights; fp32 accumulation (env TSUGUMI_PREFILL_MM_FP32=0: the decode config)."""
        f = lin.ck.math_fidelity
        ck = cls.MM_CK.get(f)
        if ck is None:
            import os

            fp32 = env("PREFILL_MM_FP32", "1") == "1"
            ck = lin.ck if not fp32 else ttnn.WormholeComputeKernelConfig(
                math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
            cls.MM_CK[f] = ck
        y = ttnn.experimental.minimal_matmul(x, lin.w, compute_kernel_config=ck, memory_config=DRAM,
                                             dtype=ttnn.bfloat16)
        n = n if n is not None else lin.N
        if n and y.shape[-1] != n:
            y2 = ttnn.slice(y, (0, 0, 0, 0), (1, 1, y.shape[-2], n), memory_config=DRAM)
            ttnn.deallocate(y)
            y = y2
        return y

    def _norm(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.eng.eps, memory_config=DRAM, compute_kernel_config=self.norm_ck)

    def _shift_matrix(self, T):
        """[T, T+32] selecting row 31+j of cat([prev tile (32 rows); x (T rows)]): row j <- x[j-1] (prev[31] for j=0)."""
        if T not in self._shift_by_T:
            S = torch.zeros(1, 1, T, T + 32)
            for j in range(T):
                S[0, 0, j, 31 + j] = 1.0
            self._shift_by_T[T] = ttnn.from_torch(S, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        return self._shift_by_T[T]

    def _row_select(self, x, rows, at=0):
        """[1,1,32,W] whose row at+i = x[rows[i]] (other rows 0; x: [1,1,R,W] TILE)."""
        R = x.shape[-2]
        M = torch.zeros(1, 1, 32, R)
        for i, r in enumerate(rows):
            M[0, 0, at + i, r] = 1.0
        Mt = ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        y = ttnn.matmul(Mt, x, memory_config=DRAM, compute_kernel_config=self.sel_ck)
        ttnn.deallocate(Mt)
        return y

    # ---------------- GDN ----------------
    def _conv(self, i, m, qkv, T, n):
        """Causal conv + SiLU over [carry; qkv]; carry <- input rows n-3..n-1 (rows 29..31 of a tile)."""
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, self.qkv_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.dev, DRAM)
        self.convk(qkv, self.carry[i], m.taps, out, x_width=qkv.shape[-1])
        M = torch.zeros(1, 1, 32, T)
        for r in range(self.Kc - 1):
            M[0, 0, 32 - (self.Kc - 1) + r, n - (self.Kc - 1) + r] = 1.0
        Mt = ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        c = ttnn.matmul(Mt, qkv, memory_config=DRAM, compute_kernel_config=self.sel_ck)
        ttnn.deallocate(Mt)
        if c.shape[-1] != self.qkv_dim:
            c2 = ttnn.slice(c, (0, 0, 0, 0), (1, 1, 32, self.qkv_dim), memory_config=DRAM)
            ttnn.deallocate(c)
            c = c2
        ttnn.deallocate(self.carry[i])
        self.carry[i] = c
        return out

    def _gdn(self, i, m, h, T, n, mask):
        qd, zd, nv = self.qkv_dim, self.qkv_dim + self.vd, self.Nv
        y = self._mm(h, m.w_in, n=False)  # [1,1,T,Np]: qkv | z | a | b | pad (the conv kernel reads qkv in place)
        z = ttnn.slice(y, (0, 0, 0, qd), (1, 1, T, zd), memory_config=DRAM)
        ab = ttnn.typecast(ttnn.slice(y, (0, 0, 0, zd), (1, 1, T, zd + 2 * nv), memory_config=DRAM), ttnn.float32)
        u = ttnn.add(ab, m.ab_bias, memory_config=DRAM)  # [a + dt_bias | b]
        ttnn.deallocate(ab)
        g = ttnn.multiply(ttnn.softplus(u, memory_config=DRAM), m.ab_coef, memory_config=DRAM)  # [g | 0]
        beta = ttnn.sigmoid(u, memory_config=DRAM)  # [. | beta]
        ttnn.deallocate(u)
        if mask is not None:  # padded rows: identity state update
            g = ttnn.multiply(g, mask, memory_config=DRAM)
            beta = ttnn.multiply(beta, mask, memory_config=DRAM)
        g2 = ttnn.reshape(ttnn.slice(g, (0, 0, 0, 0), (1, 1, T, nv), memory_config=DRAM), (1, T, nv))
        b2 = ttnn.reshape(ttnn.slice(beta, (0, 0, 0, nv), (1, 1, T, 2 * nv), memory_config=DRAM), (1, T, nv))
        ttnn.deallocate(g)
        ttnn.deallocate(beta)
        _dbg(self.dev, 'gdn pre-conv done')
        conv = self._conv(i, m, y, T, n)
        _dbg(self.dev, 'conv done')
        ttnn.deallocate(y)
        kd = self.kd
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0, 0), (1, 1, T, kd), memory_config=DRAM), (1, T, kd))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, 0, kd), (1, 1, T, 2 * kd), memory_config=DRAM), (1, T, kd))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 0, 2 * kd), (1, 1, T, qd), memory_config=DRAM), (1, T, self.vd))
        ttnn.deallocate(conv)
        eye, tril, ones, masks = self.consts
        o, fs = ttnn.transformer.chunk_gated_delta_rule(
            q, k, v, g2, b2, scale=m.scale, initial_state=self.state[i], output_final_state=True, chunk_size=32,
            eye=eye, tril=tril, ones=ones, masks=masks,
            use_mcast=env("CGDR_MCAST", "1") == "1")
        _dbg(self.dev, 'cgdr done')
        for t in (q, k, v, g2, b2):
            ttnn.deallocate(t)
        ttnn.deallocate(self.state[i])
        self.state[i] = fs
        # gated RMSNorm per head in the [T, Nv*Dv] layout: o * rsqrt(mean_head(o^2) + eps) * w * silu(z)
        o = ttnn.to_layout(ttnn.reshape(o, (1, 1, T, self.vd)), ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        sq = ttnn.multiply(o, o, memory_config=DRAM)
        ms = ttnn.matmul(sq, self.head_mean, memory_config=DRAM, compute_kernel_config=self.fp32_ck, dtype=ttnn.float32)
        ttnn.deallocate(sq)
        r = ttnn.rsqrt(ttnn.add(ms, 1e-6, memory_config=DRAM), memory_config=DRAM)
        ttnn.deallocate(ms)
        rw = ttnn.matmul(r, self.gnorm_w[i], memory_config=DRAM, compute_kernel_config=self.fp32_ck,
                         dtype=ttnn.bfloat16)  # rsqrt * w per column
        ttnn.deallocate(r)
        t = ttnn.multiply(z, rw, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
        ttnn.deallocate(z)
        ttnn.deallocate(rw)
        gated = ttnn.multiply(o, t, memory_config=DRAM)
        ttnn.deallocate(o)
        ttnn.deallocate(t)
        out = self._mm(gated, m.w_out, n=self.eng.H)
        ttnn.deallocate(gated)
        return out

    # ---------------- attention ----------------
    def _qkv_heads(self, A, h, T, cos, sin):
        NH, NKV, HD = A.NH, A.NKV, A.HD
        y = self._mm(h, A.wqkvg, n=(2 * NH + 2 * NKV) * HD)
        qkv = ttnn.slice(y, (0, 0, 0, 0), (1, 1, T, (NH + 2 * NKV) * HD), memory_config=DRAM)
        gate = ttnn.slice(y, (0, 0, 0, (NH + 2 * NKV) * HD), (1, 1, T, (2 * NH + 2 * NKV) * HD), memory_config=DRAM)
        ttnn.deallocate(y)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=NH, num_kv_heads=NKV, transpose_k_heads=False,
                                                         memory_config=DRAM)
        ttnn.deallocate(qkv)
        q = self._rope(A, ttnn.rms_norm(q, weight=A.q_norm, epsilon=1e-6, memory_config=DRAM,
                                        compute_kernel_config=self.norm_ck), cos, sin)
        k = self._rope(A, ttnn.rms_norm(k, weight=A.k_norm, epsilon=1e-6, memory_config=DRAM,
                                        compute_kernel_config=self.norm_ck), cos, sin)
        return q, k, v, gate

    @staticmethod
    def _rope(A, x, cos, sin):
        r = ttnn.add(ttnn.multiply(x, cos, memory_config=DRAM),
                     ttnn.multiply(ttnn.matmul(x, A.P, memory_config=DRAM), sin, memory_config=DRAM),
                     memory_config=DRAM)
        ttnn.deallocate(x)
        return r

    def _fill_kv(self, A, k, v, pt_chunk):
        if A.kp.dtype != k.dtype:  # bfp8 cache: paged_fill_cache copies tiles as they are
            k8, v8 = ttnn.typecast(k, A.kp.dtype), ttnn.typecast(v, A.vp.dtype)
            ttnn.experimental.paged_fill_cache(A.kp, k8, pt_chunk, batch_idx=0)
            ttnn.experimental.paged_fill_cache(A.vp, v8, pt_chunk, batch_idx=0)
            ttnn.deallocate(k8)
            ttnn.deallocate(v8)
            return
        ttnn.experimental.paged_fill_cache(A.kp, k, pt_chunk, batch_idx=0)
        ttnn.experimental.paged_fill_cache(A.vp, v, pt_chunk, batch_idx=0)

    def _attn(self, A, h, T, pos, cos, sin, pt_chunk, pt_full, sdpa_cfg):
        q, k, v, gate = self._qkv_heads(A, h, T, cos, sin)
        self._fill_kv(A, k, v, pt_chunk)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            q, A.kp, A.vp, pt_full, chunk_start_idx_tensor=self.start_t, scale=A.scale, program_config=sdpa_cfg,
            compute_kernel_config=self.sdpa_ck, memory_config=DRAM)
        ttnn.deallocate(q)
        o = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)  # [1,1,T,NH*HD]
        o = ttnn.multiply(o, ttnn.sigmoid(gate, memory_config=DRAM), memory_config=DRAM)
        ttnn.deallocate(gate)
        out = self._mm(o, A.wo, n=self.eng.H)
        ttnn.deallocate(o)
        return out

    def _mlp(self, mlp, h):
        g = self._mm(h, mlp.w1, n=False)
        u = self._mm(h, mlp.w3, n=False)
        a = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        if a.shape[-1] != mlp.w2.K:
            a2 = ttnn.slice(a, (0, 0, 0, 0), (1, 1, a.shape[-2], mlp.w2.K), memory_config=DRAM)
            ttnn.deallocate(a)
            a = a2
        out = self._mm(a, mlp.w2, n=self.eng.H)
        ttnn.deallocate(a)
        return out

    # ---------------- state hand-off with the T-row mode ----------------
    def _enter(self):
        e = self.eng
        sh = e.sh_by[e.T_dec]
        from .model import SpecShared

        a_prev = e._a_prev
        slot = max(a_prev - 1, 0) if e._last_snap_all else 0
        H = SpecShared.hist_matrix(a_prev)  # rows 0..2 = previous 3 inputs
        K1 = self.Kc - 1
        Hs = torch.zeros_like(H)
        Hs[0, 0, 32 - K1:] = H[0, 0, :K1]
        Mh = ttnn.from_torch(Hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        self.carry, self.state = {}, {}
        for i, L in enumerate(e.layers):
            if L["type"] != "linear_attention":
                continue
            m = L["mix"]
            cat = ttnn.concat([m.Hc, m.Bp], dim=2, memory_config=DRAM)
            self.carry[i] = ttnn.matmul(Mh, cat, memory_config=DRAM, compute_kernel_config=self.sel_ck)
            ttnn.deallocate(cat)
            st = ttnn.allocate_tensor_on_device(ttnn.Shape([1, self.Nv, self.Dk, self.Dv]), ttnn.float32,
                                                ttnn.TILE_LAYOUT, self.dev, DRAM)
            self.pcopy.set_offsets(slot * self.state_pages, 0)
            self.pcopy(m.snap, st)
            self.state[i] = st
        ttnn.deallocate(Mh)
        del sh

    def _exit(self):
        e = self.eng
        K1 = self.Kc - 1
        M = torch.zeros(1, 1, 32, 32)
        for r in range(K1):
            M[0, 0, r, 32 - K1 + r] = 1.0
        Mt = ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        for i, L in enumerate(e.layers):
            if L["type"] != "linear_attention":
                continue
            m = L["mix"]
            c = ttnn.matmul(Mt, self.carry[i], memory_config=DRAM, compute_kernel_config=self.sel_ck)
            ttnn.copy(c, m.Hc)
            ttnn.deallocate(c)
            ttnn.deallocate(self.carry[i])
            self.pcopy.set_offsets(0, 0)
            self.pcopy(self.state[i], m.snap)
            ttnn.deallocate(self.state[i])
        ttnn.deallocate(Mt)
        self.carry, self.state = {}, {}
        e._a_prev = 0
        e._last_snap_all = False

    def warmup(self):
        """Run every chunk bucket once (full and partial) before the engine captures its traces. Ops allocate
        device buffers on their first run for a shape (e.g. chunk_gated_delta_rule); if that happens after a
        trace capture, a later trace replay can overwrite them and the next prefill hangs."""
        e = self.eng
        mtp = e.mtp is not None
        T = self.t_min
        while T <= self.t_max:
            for n in (T, T - 1):
                e.reset_spec()
                self.run([0] * n, start=0, with_mtp=mtp)
                # resumed after the T-row mode at a start that is 64 (not 128) aligned: the 64-chunk SDPA config,
                # the previous-hidden pick for MTP, and the T-row state hand-off
                e.reset_spec()
                e.prefill_spec([0] * 64, start=0, with_mtp=mtp)
                self.run([0] * n, start=64, with_mtp=mtp)
                if n == T and T > self.sdpa_qmax // 2 and self.sdpa_qmax > 128:
                    # start % 256 == 128: q chunk 128 instead of 256 (a separate SDPA program for T >= 256)
                    e.reset_spec()
                    e.prefill_spec([0] * 128, start=0, with_mtp=mtp)
                    self.run([0] * n, start=128, with_mtp=mtp)
            T *= 2
        e.reset_spec()  # a prompt spanning two chunks (the previous chunk's last rows feed the next chunk's MTP)
        self.run([0] * (self.t_max + self.t_min), start=0, with_mtp=mtp)
        e.reset_spec()

    # ---------------- snapshots (conversation reuse) ----------------
    def save_state(self):
        """Host copy of the GDN recurrent states and conv histories at the current position. The KV cache is not
        copied: positions before this point stay valid as long as the same conversation continues."""
        self._enter()
        snap = {i: (ttnn.to_torch(self.state[i]), ttnn.to_torch(self.carry[i])) for i in self.state}
        self._exit()  # also leaves the T-row state in canonical form (slot 0, Hc rows 0..2, a_prev = 0)
        return snap

    def restore_state(self, snap):
        up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=DRAM)
        self.state = {i: up(st, ttnn.float32) for i, (st, _) in snap.items()}
        self.carry = {i: up(c, ttnn.bfloat16) for i, (_, c) in snap.items()}
        self._exit()

    # ---------------- one chunk ----------------
    def _bucket(self, n):
        T = self.t_min
        while T < n and T < self.t_max:
            T *= 2
        return T

    def _sdpa_cfg(self, T, pos):
        def fit(c):
            while c > 32 and (T % c or pos % c):
                c //= 2
            return c
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, exp_approx_mode=False,
                                      q_chunk_size=fit(self.sdpa_qmax), k_chunk_size=fit(128))

    def _chunk(self, toks, pos, prev_hid, with_mtp):
        """Rows toks at positions pos.. (pos % 64 == 0). Returns (final-normed hidden [1,1,T,H], T, n)."""
        e, dev = self.eng, self.dev
        n = len(toks)
        T = self._bucket(n)
        up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        xe = torch.zeros(1, 1, T, e.H, dtype=torch.bfloat16)
        xe[0, 0, :n] = e.embed_rows(toks)
        x = up(xe)
        mask = None
        if n < T:
            mk = torch.zeros(1, 1, T, 1)
            mk[0, 0, :n] = 1.0
            mask = up(mk, ttnn.float32)
        cs = [e._rope_rows(pos + i) for i in range(T)]
        cos = up(torch.cat([c for c, _ in cs], dim=2))  # [1,1,T,HD]
        sin = up(torch.cat([s_ for _, s_ in cs], dim=2))
        nb = T // self.BS
        e.pager.ensure(pos + T)  # padded rows write KV too
        blk = e.pager.blocks[pos // self.BS: pos // self.BS + nb]
        pt_chunk = ttnn.from_torch(torch.tensor(blk, dtype=torch.int32).view(1, nb), dtype=ttnn.int32, device=dev)
        pt_full = e.pager.pt1
        sdpa_cfg = self._sdpa_cfg(T, pos)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32),
                                        self.start_t)
        for i, L in enumerate(e.layers):
            h = self._norm(x, L["in_norm"])
            if L["type"] == "linear_attention":
                a = self._gdn(i, L["mix"], h, T, n, mask)
            else:
                a = self._attn(L["mix"], h, T, pos, cos, sin, pt_chunk, pt_full, sdpa_cfg)
            ttnn.deallocate(h)
            _dbg(self.dev, f'layer {i} mix done')
            x2 = ttnn.add(x, a, memory_config=DRAM)
            ttnn.deallocate(a)
            ttnn.deallocate(x)
            x = x2
            h = self._norm(x, L["post_norm"])
            m = self._mlp(L["mlp"], h)
            ttnn.deallocate(h)
            x2 = ttnn.add(x, m, memory_config=DRAM)
            ttnn.deallocate(m)
            ttnn.deallocate(x)
            x = x2
        _dbg(self.dev, 'layers done')
        hid = self._norm(x, e.final_norm)
        ttnn.deallocate(x)
        _dbg(self.dev, 'final norm done')
        if with_mtp and e.mtp is not None:
            self._mtp_kv(xe, hid, prev_hid, T, pos, cos, sin, pt_chunk)
        for t in (cos, sin, pt_chunk) + ((mask,) if mask is not None else ()):
            ttnn.deallocate(t)
        return hid, T, n

    def _mtp_kv(self, xe, hid, prev_hid, T, pos, cos, sin, pt_chunk):
        """MTP row j at position pos+j = (main hidden at pos+j-1, token at pos+j): only its KV is needed."""
        e, mtp = self.eng, self.eng.mtp
        cat = ttnn.concat([prev_hid, hid], dim=2, memory_config=DRAM)
        hs = ttnn.matmul(self._shift_matrix(T), cat, memory_config=DRAM, compute_kernel_config=self.sel_ck)
        ttnn.deallocate(cat)
        _dbg(self.dev, 'mtp shift done')
        emb = ttnn.from_torch(xe, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        a = self._norm(emb, mtp.n_emb)
        b = self._norm(hs, mtp.n_hid)
        ttnn.deallocate(emb)
        ttnn.deallocate(hs)
        c2 = ttnn.concat([a, b], dim=-1, memory_config=DRAM)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        x = self._mm(c2, mtp.fc, n=e.H)
        ttnn.deallocate(c2)
        _dbg(self.dev, 'mtp fc done')
        h = self._norm(x, mtp.in_norm)
        ttnn.deallocate(x)
        q, k, v, gate = self._qkv_heads(mtp.attn, h, T, cos, sin)
        ttnn.deallocate(h)
        ttnn.deallocate(q)
        ttnn.deallocate(gate)
        _dbg(self.dev, 'mtp qkv done')
        self._fill_kv(mtp.attn, k, v, pt_chunk)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        _dbg(self.dev, 'mtp fill done')

    def logits_host(self, hid, r0, r1, step=128):
        """Full-vocab logits of rows [r0, r1) of a kept hidden [1,1,T,H] -> torch float [r1-r0, V]."""
        e, out = self.eng, []
        for a in range(r0 - r0 % 32, r1, step):
            b = min(a + step, hid.shape[-2])
            hs = ttnn.slice(hid, (0, 0, a, 0), (1, 1, b, e.H), memory_config=DRAM)
            parts = []
            for lo, hi, lin in e.lm_chunks:
                y = self._mm(hs, lin, n=hi - lo)
                parts.append(ttnn.to_torch(y).float().reshape(b - a, hi - lo))
                ttnn.deallocate(y)
            ttnn.deallocate(hs)
            out.append(torch.cat(parts, 1))
        lg = torch.cat(out, 0)
        off = r0 - (r0 - r0 % 32)
        return lg[off: off + (r1 - r0)]

    # ---------------- driver ----------------
    def run(self, ids, start=0, with_mtp=True):
        """Prefill ids at positions start.. (start % 64 == 0). Returns (next token, next position, MTP draft)."""
        e = self.eng
        assert start % self.BS == 0
        t0 = time.perf_counter()
        _dbg(self.dev, f'run start={start} n={len(ids)} a_prev={e._a_prev} snap_all={e._last_snap_all}')
        self._enter()
        _dbg(self.dev, 'entered')
        prev_hid = None
        if with_mtp and e.mtp is not None:
            prev_hid = ttnn.from_torch(torch.zeros(1, 1, 32, e.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=self.dev)
            if start > 0 and getattr(e, "last_spec", None) is not None and e._a_prev > 0:
                # main hidden of the last row processed by the T-row mode
                ttnn.deallocate(prev_hid)
                prev_hid = self._row_select(e.last_spec[2], [e._a_prev - 1], at=31)
        pos, p = start, 0
        hid = None
        while p < len(ids):
            n = min(self.t_max, len(ids) - p)
            if hid is not None and self.keep is None:
                ttnn.deallocate(hid)
            hid, T, n = self._chunk(ids[p:p + n], pos, prev_hid, with_mtp)
            if self.keep is not None:
                self.keep.append((hid, n))
            if prev_hid is not None and p + n < len(ids):
                ttnn.deallocate(prev_hid)
                prev_hid = ttnn.slice(hid, (0, 0, T - 32, 0), (1, 1, T, e.H), memory_config=DRAM)
            pos += n
            p += n
            if e.progress is not None:  # e.g. the server's hang backstop: a 250k-token prefill takes minutes
                e.progress()
        self._exit()
        # last row -> logits / argmax, and hid_src row 0 for the MTP draft
        hl = self._row_select(hid, [n - 1])
        if self.keep is None:
            ttnn.deallocate(hid)
        if prev_hid is not None:
            ttnn.deallocate(prev_hid)
        e.use_T(e.T_dec)
        if getattr(e, "hid_src", None) is not None:
            ttnn.copy(hl, e.hid_src)
        logits, tk = e._logits_topk(e._lm_head_rows(hl))
        ttnn.deallocate(hl)
        e.last_topk = e.topk_to_host(tk, 1)
        nxt = e._pick_row(0)  # before the MTP run (it overwrites hid_src)
        ttnn.deallocate(logits)
        for t in tk:
            ttnn.deallocate(t)
        draft = None
        if with_mtp and e.mtp is not None:
            draft = e.mtp.run([nxt], pos, [0])[0]
            e._mtp_last_rows = 1
        self.timing = dict(t=time.perf_counter() - t0, n=len(ids))
        return nxt, pos, draft
