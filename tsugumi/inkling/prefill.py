"""Chunked prefill for the Inkling engine: T = 32 prompt tokens per step (rows = tokens), at any start position.

Same weights, caches and convolution histories as the decode step (engine.py); the differences:
- attention residuals per token: the normalized block states are stacked on dim 1 ([1, n, 32, H]); scores =
  sum(B * q), softmax over the blocks, out = sum(B * p) over dim 1;
- convolutions over the 32 rows and the 3 history rows: X_lag = MALL [x; hist] (one matmul for the 4 lags), the new
  history = MH [x; hist] (MH picks the last 3 real rows, written by the host per chunk);
- K / V of the 32 rows are written into the (non-paged) caches by kernels/kv_write.cpp (one block of the whole
  cache), padding rows to a scratch slot;
- attention by matmuls, batched over the KV heads (q heads 3g..3g+2 x 32 tokens = 96 rows per KV head), with a
  per-token mask in head-major layout [1, NH, 32, S]: the relative bias gathered by distance (ttnn.gather, index =
  the distance of each slot from each token), the validity (window / causal), and the sink logit put on a slot whose
  K and V stay zero;
- MoE with the dense routing weights [32, E] and ttnn.sparse_matmul over the union of the chunk's experts.
"""
import math

import torch
import ttnn

from ..kv_kernel import KVWriteKernel

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
NEG = -1e9
U, UWP = ttnn.UnaryOpType, ttnn.UnaryWithParam
T = 32


class PrefillMixin:
    def init_prefill(self):
        c = self.cfg
        H, R, RA, SA = c.hidden, self.R, self.RA, self.SA
        NKV, HD = c.n_kv, c.head_dim
        k = c.conv_k
        # X_lag[t] = x[t - lag] (t >= lag) or hist row lag - t
        mall = torch.zeros(1, 1, k * T, 2 * T)
        for lag in range(k):
            for t in range(T):
                if t >= lag:
                    mall[0, 0, lag * T + t, t - lag] = 1.0
                else:
                    mall[0, 0, lag * T + t, T + (lag - t)] = 1.0
        self.P_MALL = self.tt(mall)
        self.P_TCOL = self.tt(torch.arange(T).float().view(1, 1, T, 1), ttnn.float32)
        er = torch.zeros(1, 1, 32, RA)
        er[0, 0, 0, R] = 1.0
        self.P_ER = self.tt(er)  # row 0 -> the ring sink slot
        es = torch.zeros(1, 1, 32, SA)
        es[0, 0, 0, self.max_seq] = 1.0
        self.P_ES = self.tt(es)  # row 0 -> the full-cache sink slot
        sf = torch.zeros(1, 1, 1, RA)
        sf[0, 0, 0, R] = -NEG  # cancels the invalid mark of the ring sink slot
        self.P_SINKFIX = self.tt(sf, ttnn.float32)
        ars = torch.full((SA,), 1e7)  # sink slot: far in the past (valid, outside the window); scratch etc.: invalid
        ars[:self.max_seq] = torch.arange(self.max_seq).float()
        ars[self.max_seq] = -1e7
        self.P_ARS = self.tt(ars.view(1, 1, 1, SA), ttnn.float32)
        # dynamic conv: D[lag][t, c] = dyn[t, head(c) * 4 + lag] / sqrt(2k + 1)
        el = torch.zeros(1, k, 32, NKV * HD)
        for lag in range(k):
            for h in range(NKV):
                el[0, lag, h * k + lag, h * HD:(h + 1) * HD] = 1.0 / math.sqrt(2 * k + 1)
        self.P_ELAG = self.tt(el)
        # per-chunk inputs
        self.P_X0 = self.tt(torch.zeros(1, 1, T, H))
        self.P_POSR = ttnn.from_torch(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.P_POSF = ttnn.from_torch(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.P_PT0 = ttnn.from_torch(torch.zeros(1, 8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.P_C0 = self.tt(torch.zeros(1, 1, 1, 1), ttnn.float32)  # p0 % R
        self.P_THR = self.tt(torch.zeros(1, 1, T, 1), ttnn.float32)  # min(P_t, W - 1)
        self.P_PCOL = self.tt(torch.zeros(1, 1, T, 1), ttnn.float32)  # P_t
        self.P_TAU = self.tt(torch.ones(1, T, 1, 1))  # tau(P_t), [1, T, 1, 1] (token-major q / r)
        self.P_MH = self.tt(torch.zeros(1, 1, T, 2 * T))
        self.P_SEL = self.tt(torch.zeros(1, 1, T, T))
        self.kvw_ring = KVWriteKernel(self.dev, T, NKV, HD, RA, 1, ttnn.bfloat8_b)
        self.kvw_full = KVWriteKernel(self.dev, T, NKV, HD, SA, 1, ttnn.bfloat8_b)
        self.p_pc_gu = self._spc(2 * c.inter, (9, 8), 6)
        self.p_pc_dn = self._spc(H, (9, 8), 6)
        for d in self.layers:
            self._prefill_layer_consts(d)
        self.p_qf = ttnn.slice(self.Qf, (0, 0, 0, 0), (1, 1, 1, H))

    def _spc(self, n, cores, bw):
        nc = cores[0] * cores[1]
        pcn = math.ceil(math.ceil(n / 32) / nc)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*cores), in0_block_w=bw, out_subblock_h=1,
            out_subblock_w=1, out_block_h=1, out_block_w=pcn, per_core_M=1, per_core_N=pcn, fuse_batch=False,
            fused_activation=None, mcast_in0=True)

    def _prefill_layer_consts(self, d):
        c = self.cfg
        H, k = c.hidden, c.conv_k
        qrow = lambda Q, r: ttnn.slice(Q, (0, 0, r, 0), (1, 1, r + 1, H))
        if "Qa" in d:
            d["p_qa"] = [qrow(d["Qa"], r) for r in range(d["nqa"])]
        d["p_qm"] = qrow(d["Qm"], 0)
        if "Qmsg" in d:
            d["p_qmsg"] = [qrow(d["Qmsg"], r) for r in range(4)]
        for name in ("tk", "tv", "tkv", "ta", "tm"):  # taps rows 0..3 -> [1, 4, 1, C]
            if name in d:
                t = ttnn.to_torch(d[name]).float()[0, 0, :k]
                d["p_" + name] = self.tt(t.view(1, k, 1, -1))
        # the relative bias by distance: [1, NH, 32, R] (sliding; distances >= W are 0) / [.., W] (full)
        d["p_proj"] = ttnn.slice(d["proj"], (0, 0, 0, 0), (1, 1, c.d_rel, d["proj"].shape[-1]))

    # ------------------------------------------------------------------------------------------------ pieces
    def p_attn_res(self, q, B):
        """Per-token attention residual: q [1,1,1,H] (pre-scaled), B [1, n, 32, H] normalized states."""
        s = ttnn.sum(ttnn.multiply(B, q, memory_config=L1), dim=-1, keepdim=True, memory_config=L1)  # [1,n,32,1]
        s = ttnn.permute(s, (0, 3, 2, 1), memory_config=L1)  # [1,1,32,n]
        p = ttnn.softmax(s, dim=-1, memory_config=L1, numeric_stable=True)
        p = ttnn.permute(p, (0, 3, 2, 1), memory_config=L1)  # [1,n,32,1]
        return ttnn.sum(ttnn.multiply(B, p, memory_config=L1), dim=1, keepdim=True, memory_config=L1)

    def p_stack(self, B, x_norm):
        return x_norm if B is None else ttnn.concat([B, x_norm], dim=1, memory_config=L1)

    def p_conv(self, x, hist, taps4, dyn4=None):
        XH = ttnn.concat([x, hist], dim=-2, memory_config=L1)  # [1,1,64,C]
        Y = ttnn.reshape(self.mm(self.P_MALL, XH), (1, self.cfg.conv_k, T, x.shape[-1]))
        taps = taps4 if dyn4 is None else ttnn.add(dyn4, taps4, memory_config=L1)
        y = ttnn.sum(ttnn.multiply(Y, taps, memory_config=L1), dim=1, keepdim=True, memory_config=L1)
        self.mm(self.P_MH, XH, optional_output_tensor=hist, memory_config=hist.memory_config())
        return y

    def p_l2clip(self, x, thr, tau=None):
        ss = ttnn.sum(ttnn.multiply(x, x, memory_config=L1), dim=-1, keepdim=True, memory_config=L1)
        f = ttnn.unary_chain(ss, [UWP(U.RSQRT), UWP(U.MUL_UNARY_SFPU, thr), UWP(U.MINIMUM, 1.0)], memory_config=L1)
        if tau is not None:
            f = ttnn.multiply(f, tau, memory_config=L1)
        return ttnn.multiply(x, f, memory_config=L1)

    def p_attention(self, d, st, q_in, k_in, v_in, pc):
        c = self.cfg
        NH, NKV, HD, DR = c.n_heads, c.n_kv, c.head_dim, c.d_rel
        sliding = d["sliding"]
        nq, nkv = d["nq"], d["nkv"]
        fused = q_in is k_in and k_in is v_in
        if fused:
            y = d["Lqkv"](q_in, out_mem=L1)
            yq = y
            yk = ttnn.slice(y, (0, 0, 0, nq), (1, 1, T, nq + nkv), memory_config=L1)
            yv = ttnn.slice(y, (0, 0, 0, nq + nkv), (1, 1, T, nq + 2 * nkv), memory_config=L1)
        else:
            yq = d["Lq"](q_in, out_mem=L1)
            yk = d["Lk"](k_in, out_mem=L1)
            yv = d["Lv"](v_in, out_mem=L1)
        part = lambda y, lo, hi, nh, hd: ttnn.reshape(ttnn.slice(y, (0, 0, 0, lo), (1, 1, T, hi), memory_config=L1),
                                                      (1, T, nh, hd))
        tau = None if sliding else self.P_TAU
        q = ttnn.multiply(part(yq, 0, NH * HD, NH, HD), d["qscale"], memory_config=L1)
        q = self.p_l2clip(q, c.qk_clip, tau)  # [1, T, NH, HD]
        gate = part(yq, NH * HD, 2 * NH * HD, NH, HD)
        o0 = 2 * NH * HD
        r = ttnn.multiply(part(yq, o0, o0 + NH * DR, NH, DR), d["rscale"], memory_config=L1)
        r = self.p_l2clip(r, c.rel_clip, tau)
        r = ttnn.permute(r, (0, 2, 1, 3), memory_config=L1)  # [1, NH, T, DR]
        sink = ttnn.slice(yq, (0, 0, 0, o0 + NH * DR), (1, 1, T, o0 + NH * DR + NH), memory_config=L1)  # [1,1,T,NH]
        sink = ttnn.reshape(ttnn.transpose(sink, -2, -1, memory_config=L1), (1, NH, T, 1))
        sink = ttnn.pad(sink, [(0, 0), (0, 0), (0, 0), (0, 31)], 0.0)  # [1, NH, T, 32], col 0 = sink / scale
        C = NKV * HD
        if sliding:
            if fused:
                kvraw = ttnn.slice(y, (0, 0, 0, nq), (1, 1, T, nq + 2 * C), memory_config=L1)
            else:
                kvraw = ttnn.concat([ttnn.slice(yk, (0, 0, 0, 0), (1, 1, T, C), memory_config=L1),
                                     ttnn.slice(yv, (0, 0, 0, 0), (1, 1, T, C), memory_config=L1)], dim=-1,
                                    memory_config=L1)
            kv = self.p_conv(kvraw, st["hkv"], d["p_tkv"])
            k = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, T, C), memory_config=L1)
            v = ttnn.slice(kv, (0, 0, 0, C), (1, 1, T, 2 * C), memory_config=L1)
        else:
            kraw = ttnn.slice(yk, (0, 0, 0, 0), (1, 1, T, C), memory_config=L1)
            vraw = ttnn.slice(yv, (0, 0, 0, 0), (1, 1, T, C), memory_config=L1)
            def dyn4(y):
                dd = ttnn.slice(y, (0, 0, 0, NKV * HD), (1, 1, T, NKV * HD + 32), memory_config=L1)
                return self.mm(ttnn.repeat(dd, (1, c.conv_k, 1, 1), memory_config=L1), self.P_ELAG)
            k = self.p_conv(kraw, st["hk"], d["p_tk"], dyn4(yk))
            v = self.p_conv(vraw, st["hv"], d["p_tv"], dyn4(yv))
        k = self.p_l2clip(ttnn.reshape(k, (1, T, NKV, HD)), c.qk_clip)
        v = self.rmsu(ttnn.reshape(v, (1, T, NKV, HD)))
        if sliding:
            self.kvw_ring(k, v, st["kc"], st["vc"], self.P_POSR, self.P_PT0)
        else:
            self.kvw_full(k, v, st["kc"], st["vc"], self.P_POSF, self.P_PT0)
        # mask [1, NH, T, S] (units of 1 / scale)
        rel = self.mm(r, d["p_proj"])  # [1, NH, T, R or W] by distance
        if sliding:
            rel = ttnn.gather(rel, -1, index=pc["IDXR"])  # [1, NH, T, RA] by slot
            mask = ttnn.add(rel, pc["VR"], memory_config=L1)
            mask = ttnn.add(mask, self.mm(sink, self.P_ER), memory_config=DRAM)
        else:
            rel = ttnn.gather(rel, -1, index=pc["IDXF"])  # [1, NH, T, W] by position % W
            rep = ttnn.repeat(rel, (1, 1, 1, self.SA // c.window), memory_config=DRAM)
            mask = ttnn.add(ttnn.multiply(rep, pc["BWIN"], memory_config=DRAM), pc["AV"], memory_config=DRAM)
            mask = ttnn.add(mask, self.mm(sink, self.P_ES, memory_config=DRAM), memory_config=DRAM)
        S = mask.shape[-1]
        mask = ttnn.reshape(mask, (1, NKV, (NH // NKV) * T, S))
        qg = ttnn.reshape(ttnn.permute(q, (0, 2, 1, 3), memory_config=L1), (1, NKV, (NH // NKV) * T, HD))
        s = self.mm(qg, st["kc"], transpose_b=True, memory_config=DRAM)  # [1, NKV, 96, S]
        s = ttnn.multiply(ttnn.add(s, mask, memory_config=DRAM), self.scale, memory_config=DRAM)
        p = ttnn.softmax(s, dim=-1, memory_config=DRAM, numeric_stable=True)
        o = self.mm(p, st["vc"], memory_config=L1)  # [1, NKV, 96, HD]
        o = ttnn.permute(ttnn.reshape(o, (1, NH, T, HD)), (0, 2, 1, 3), memory_config=L1)  # [1, T, NH, HD]
        o = ttnn.multiply(o, gate, input_tensor_b_activations=[UWP(U.SIGMOID)], memory_config=L1)
        return d["Lo"](ttnn.reshape(o, (1, 1, T, NH * HD)), out_mem=L1)

    def p_moe(self, d, x):
        c = self.cfg
        logits = self.mm(x, d["Wg"], dtype=ttnn.float32)
        s = ttnn.sigmoid(logits, memory_config=L1)
        sb = ttnn.add(s, d["gbias"], memory_config=L1)
        vals, _ = ttnn.topk(sb, c.top_k, dim=-1, largest=True, sorted=True, memory_config=L1)
        kth = ttnn.slice(vals, (0, 0, 0, c.top_k - 1), (1, 1, T, c.top_k), memory_config=L1)
        w = ttnn.multiply(s, ttnn.ge(sb, kth, memory_config=L1), memory_config=L1)  # dense routing weights
        kk = c.route_scale * c.gate_softcap * c.up_softcap
        w = ttnn.multiply(w, ttnn.sum(w, dim=-1, keepdim=True, memory_config=L1),
                          input_tensor_b_activations=[UWP(U.RECIP), UWP(U.MUL_UNARY_SFPU, kk)], memory_config=L1)
        w = ttnn.typecast(w, ttnn.bfloat16)
        sp = ttnn.to_layout(ttnn.max(w, dim=-2, keepdim=True, memory_config=L1), ttnn.ROW_MAJOR_LAYOUT)
        E, I = c.n_experts, c.inter
        gu = ttnn.sparse_matmul(x, d["Egu"], sparsity=sp, is_input_b_sparse=True, memory_config=DRAM,
                                program_config=self.p_pc_gu, compute_kernel_config=d["eck"])
        gu = ttnn.reshape(gu, (1, E, T, 2 * I))
        g = ttnn.slice(gu, (0, 0, 0, 0), (1, E, T, I), memory_config=DRAM)
        u = ttnn.slice(gu, (0, 0, 0, I), (1, E, T, 2 * I), memory_config=DRAM)
        act = ttnn.multiply(g, g, input_tensor_a_activations=[UWP(U.MUL_UNARY_SFPU, 1.0 / c.gate_softcap),
                                                              UWP(U.TANH)],
                            input_tensor_b_activations=[UWP(U.SIGMOID)], memory_config=DRAM)
        act = ttnn.multiply(act, u, input_tensor_b_activations=[UWP(U.MUL_UNARY_SFPU, 1.0 / c.up_softcap),
                                                                UWP(U.TANH)], memory_config=DRAM)
        dn = ttnn.sparse_matmul(act, d["Edn"], sparsity=sp, is_input_a_sparse=True, is_input_b_sparse=True,
                                memory_config=DRAM, program_config=self.p_pc_dn, compute_kernel_config=d["eck"])
        dn = ttnn.reshape(dn, (1, E, T, c.hidden))
        wp = ttnn.permute(w, (0, 3, 2, 1), memory_config=L1)  # [1, E, T, 1]
        return ttnn.sum(ttnn.multiply(dn, wp, memory_config=DRAM), dim=1, keepdim=True, memory_config=L1)

    def p_layer(self, a, h, B, pc, msgs=None):
        c = self.cfg
        _, i, _ = self.apps[a]
        d, st = self.layers[i], self.state[a]
        N = self.p_stack(B, self.rmsu(h))
        if d["sliding"]:
            ai = self.p_attn_res(d["p_qa"][0], N) if i > 0 else h
            if msgs is None:
                q_in = k_in = v_in = ttnn.rms_norm(ai, weight=d["ln"], epsilon=c.eps, memory_config=L1)
            else:
                na = self.rmsu(ai)
                q_in, k_in, v_in = (ttnn.rms_norm(self.msg_gate(d, wh, ai, m, na, rows=True), weight=d["ln"], epsilon=c.eps,
                                                  memory_config=L1)
                                    for wh, m in zip(("attn", "k", "v"), msgs[:3]))
        else:
            if i > 0:
                qa, ka, va = (self.p_attn_res(q, N) for q in d["p_qa"])
            else:
                qa = ka = va = h
            if msgs is not None:
                qa = self.msg_gate(d, "attn", qa, msgs[0], rows=True)
                ka = self.msg_gate(d, "k", ka, msgs[1], rows=True)
                va = self.msg_gate(d, "v", va, msgs[2], rows=True)
            q_in = ttnn.rms_norm(qa, weight=d["lnq"], epsilon=c.eps, memory_config=L1)
            k_in = ttnn.rms_norm(ka, weight=d["lnk"], epsilon=c.eps, memory_config=L1)
            v_in = ttnn.rms_norm(va, weight=d["lnv"], epsilon=c.eps, memory_config=L1)
        o = self.p_attention(d, st, q_in, k_in, v_in, pc)
        o = self.p_conv(o, st["ha"], d["p_ta"])
        if c.boundary(i):
            B = N
            res = o
        else:
            res = ttnn.add(h, o, memory_config=L1)
        m_in = self.p_attn_res(d["p_qm"], self.p_stack(B, self.rmsu(res)))
        if msgs is not None:
            m_in = self.msg_gate(d, "mlp", m_in, msgs[3], rows=True)
        m = self.p_moe(d, ttnn.rms_norm(m_in, weight=d["lnm"], epsilon=c.eps, memory_config=L1))
        m = self.p_conv(m, st["hm"], d["p_tm"])
        return ttnn.add(res, m, memory_config=L1), B

    def p_consts(self):
        """Per-chunk mask inputs, from P_C0 (p0 % R), P_THR, P_PCOL."""
        c = self.cfg
        W, R = c.window, self.R
        pc = {}
        d0 = ttnn.neg(ttnn.sub(self.ARW, self.P_C0, memory_config=L1), memory_config=L1)  # (c0 - s), [1,1,1,RA]
        d0 = ttnn.add(d0, ttnn.multiply(ttnn.lt(d0, 0.0, memory_config=L1), float(R), memory_config=L1),
                      memory_config=L1)
        dt = ttnn.add(d0, self.P_TCOL, memory_config=L1)  # [1,1,T,RA] distance of slot s from token t
        dt = ttnn.subtract(dt, ttnn.multiply(ttnn.ge(dt, float(R), memory_config=L1), float(R), memory_config=L1),
                           memory_config=L1)
        valid = ttnn.le(dt, self.P_THR, memory_config=L1)
        vr = ttnn.add(ttnn.multiply(ttnn.subtract(valid, 1.0, memory_config=L1), -NEG, memory_config=L1),
                      self.P_SINKFIX, memory_config=L1)
        pc["VR"] = ttnn.reshape(ttnn.typecast(vr, ttnn.bfloat16), (1, 1, T, self.RA))
        idx = ttnn.clamp(dt, min=0.0, max=float(R - 1), memory_config=L1)
        pc["IDXR"] = ttnn.repeat(ttnn.typecast(idx, ttnn.uint32), (1, c.n_heads, 1, 1), memory_config=L1)
        # full layers: distance by position % W; window / causal masks over the cache
        arw = ttnn.slice(self.ARW, (0, 0, 0, 0), (1, 1, 1, W), memory_config=L1)  # s' = 0..W-1
        df = ttnn.sub(self.P_PCOL, arw, memory_config=L1)  # P_t - s'
        df = ttnn.subtract(df, ttnn.multiply(ttnn.floor(ttnn.multiply(df, 1.0 / W, memory_config=L1),
                                                        memory_config=L1), float(W), memory_config=L1),
                           memory_config=L1)  # (P_t - s') mod W
        df = ttnn.clamp(df, min=0.0, max=float(W - 1), memory_config=L1)
        pc["IDXF"] = ttnn.repeat(ttnn.typecast(df, ttnn.uint32), (1, c.n_heads, 1, 1), memory_config=L1)
        dp = ttnn.sub(self.P_PCOL, self.P_ARS, memory_config=DRAM)  # P_t - p, [1,1,T,SA]
        win = ttnn.multiply(ttnn.ge(dp, 0.0, memory_config=DRAM), ttnn.lt(dp, float(W), memory_config=DRAM),
                            memory_config=DRAM)
        pc["BWIN"] = ttnn.typecast(win, ttnn.bfloat16)
        pc["AV"] = ttnn.typecast(ttnn.multiply(ttnn.lt(dp, 0.0, memory_config=DRAM), NEG, memory_config=DRAM),
                                 ttnn.bfloat16)
        return pc

    def p_forward(self, logits):
        c = self.cfg
        pc = self.p_consts()
        h, B, a = self.P_X0, None, 0
        for _ in range(c.loop_start):
            h, B = self.p_layer(a, h, B, pc)
            a += 1
        pre = (h, B)
        for _ in range(c.n_loop):
            h, B = self.p_layer(a, h, B, pc)
            a += 1
        for _ in range(1, c.passes):
            NM = ttnn.concat([ttnn.slice(B, (0, 2, 0, 0), (1, 3, T, c.hidden), memory_config=L1), self.rmsu(h)],
                             dim=1, memory_config=L1)
            h, B = pre
            for j in range(c.n_loop):
                msgs = [self.p_attn_res(q, NM) for q in self.layers[c.loop_start + j]["p_qmsg"]]
                h, B = self.p_layer(a, h, B, pc, msgs)
                a += 1
        for _ in range(c.loop_end, c.n_layers):
            h, B = self.p_layer(a, h, B, pc)
            a += 1
        if not logits:
            return None
        f = self.p_attn_res(self.p_qf, self.p_stack(B, self.rmsu(h)))
        f = ttnn.rms_norm(f, weight=self.lnf, epsilon=c.eps, memory_config=L1)
        return self.head(self.mm(self.P_SEL, f))  # the last real row -> row 0

    # ------------------------------------------------------------------------------------------------ host side
    def p_write_inputs(self, tokens, p0):
        c = self.cfg
        n = len(tokens)
        assert 1 <= n <= T
        e = torch.stack([self.embed[t:t + 1][0] for t in tokens]).float()
        e = e * torch.rsqrt(e.square().mean(-1, keepdim=True) + c.eps) * self.embed_norm
        x = torch.zeros(1, 1, T, c.hidden)
        x[0, 0, :n] = e
        P = [p0 + t for t in range(T)]
        posr = [P[t] % self.R if t < n else self.R + 1 for t in range(T)]
        posf = [P[t] if t < n else self.max_seq + 1 for t in range(T)]
        mh = torch.zeros(1, 1, T, 2 * T)
        for r in range(1, c.conv_k):  # new hist row r = x at position p0 + n - r
            j = n - r
            mh[0, 0, r, j if j >= 0 else T + (r - n)] = 1.0
        sel = torch.zeros(1, 1, T, T)
        sel[0, 0, 0, n - 1] = 1.0
        put = lambda t, v, dt: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT if dt != ttnn.int32 else ttnn.ROW_MAJOR_LAYOUT), t)
        put(self.P_X0, x, ttnn.bfloat16)
        put(self.P_POSR, torch.tensor(posr, dtype=torch.int32), ttnn.int32)
        put(self.P_POSF, torch.tensor(posf, dtype=torch.int32), ttnn.int32)
        put(self.P_C0, torch.full((1, 1, 1, 1), float(p0 % self.R)), ttnn.float32)
        put(self.P_THR, torch.tensor([float(min(p, c.window - 1)) for p in P]).view(1, 1, T, 1), ttnn.float32)
        put(self.P_PCOL, torch.tensor([float(p) for p in P]).view(1, 1, T, 1), ttnn.float32)
        put(self.P_TAU, torch.tensor([c.tau(p) for p in P]).view(1, T, 1, 1), ttnn.bfloat16)
        put(self.P_MH, mh, ttnn.bfloat16)
        put(self.P_SEL, sel, ttnn.bfloat16)

    def prefill(self, tokens, logits=True, traced=True, progress=None):
        """Feed tokens from self.pos on, 32 per step. Returns the top-k after the last one if logits."""
        out = None
        for k in range(0, len(tokens), T):
            chunk = tokens[k:k + T]
            last = k + T >= len(tokens)
            want = logits and last
            if self.pos + len(chunk) > self.max_seq:
                raise RuntimeError(f"context full ({self.max_seq} tokens)")
            self.p_write_inputs(chunk, self.pos)
            key = ("prefill", want)
            if traced and key in self.traces:
                tid, o = self.traces[key]
                ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
            else:
                o = self.p_forward(want)
            self.pos += len(chunk)
            if want:
                out = self.read_topk(o)
            if progress is not None:
                progress()
        return out
