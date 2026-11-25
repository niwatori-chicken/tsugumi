"""Chunked prefill for the Inkling engine: T = 64 (or 32) prompt tokens per step (rows = tokens), at any start position.

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

from ..env import env
from ..kv_kernel import KVWriteKernel, KVWriteRows
from .mask_kernel import MaskBuilder

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
NEG = -1e9
U, UWP = ttnn.UnaryOpType, ttnn.UnaryWithParam
# prompt tokens per prefill step (rows = tokens): 512 (default), 256, 128, 64 or 32 (TSUGUMI_INKLING_PREFILL_T). The
# per-step cost of most prefill ops grows much less than the rows (max_seq 32768, 4k prompt: 770 tok/s at 512, ~380
# at 128); a short remainder costs one full step (27 tokens: 0.57 s at 512, ~0.4 s at 128)
T = int(env("INKLING_PREFILL_T", "512"))
assert T in (32, 64, 128, 256, 512)
# the full layers' attention runs on blocks of TS query rows (the [rows, context] scores of 512 rows would not fit)
TS = min(T, 128)
KZ = 128  # zero K / V rows after a full layer's bucket (sink column + padding: S a multiple of the SDPA key chunk)
USE_SDPA = env("INKLING_PREFILL_SDPA", "1") == "1"
# where the prefill's activations go: DRAM at T = 128 (in L1 the residual stream / q / k / v starve the
# attention softmax's CBs)
PL = L1 if T <= 64 else DRAM


class PrefillMixin:
    def init_prefill(self):
        c = self.cfg
        H, R, RA, SA = c.hidden, self.R, self.RA, self.SA
        NKV, HD = c.n_kv, c.head_dim
        k = c.conv_k
        # X_lag[t] = x[t - lag] (t >= lag) or hist row lag - t
        mall = torch.zeros(1, 1, k * T, T + 32)  # XH = [x (T rows); hist (32 rows)]
        for lag in range(k):
            for t in range(T):
                if t >= lag:
                    mall[0, 0, lag * T + t, t - lag] = 1.0
                else:
                    mall[0, 0, lag * T + t, T + (lag - t)] = 1.0
        self.P_MALL = self.tt(mall)
        # convolution by row tiles (p_conv): XP[i] = [x tile i - 1 (hist for i = 0); x tile i] (64 rows); row l * 32 + r
        # of SHIFT @ XP[i] = x[32 i + r - l]
        sh = torch.zeros(1, 1, k * 32, 64)
        for lag in range(k):
            for r in range(32):
                sh[0, 0, lag * 32 + r, 32 + r - lag] = 1.0
        self.P_SHIFT = self.tt(sh)
        fl = torch.zeros(1, 1, 32, 32)  # hist row j (x at p0 - j) -> row 32 - j (the previous tile's last rows)
        for j in range(1, k):
            fl[0, 0, 32 - j, j] = 1.0
        self.P_FLIP = self.tt(fl)
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
        # context buckets for the full layers' prefill attention: a chunk whose positions are all below S_b attends to
        # cache slots [0, S_b) + 32 zero K / V rows (the sink at column S_b) instead of all max_seq + W slots; one
        # trace per bucket (TSUGUMI_INKLING_PREFILL_BUCKETS, comma separated, multiples of the window; "" = off)
        W = c.window
        bl = env("INKLING_PREFILL_BUCKETS", "4096,8192,16384")
        self.p_buckets = [b for b in (int(x) for x in bl.split(",") if x) if b < self.max_seq]
        assert all(b % W == 0 and b >= T for b in self.p_buckets)
        self.p_bucket = None
        self.P_ARS_B, self.P_ES_B = {}, {}
        for b in self.p_buckets:
            a_ = torch.full((b + 32,), 1e7)
            a_[:b] = torch.arange(b).float()
            a_[b] = -1e7
            self.P_ARS_B[b] = self.tt(a_.view(1, 1, 1, b + 32), ttnn.float32)
            e_ = torch.zeros(1, 1, 32, b + 32)
            e_[0, 0, 0, b] = 1.0
            self.P_ES_B[b] = self.tt(e_)
        # zero K / V rows after a bucket's slots (the sink column, then padding to the SDPA key chunk)
        self.P_KZ = ttnn.from_torch(torch.zeros(1, NKV, KZ, HD), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                    device=self.dev, memory_config=DRAM)
        self.P_RZ = self.tt(torch.zeros(1, c.n_heads, TS, 32), mem=DRAM)
        self.P_VALID = self.tt(torch.zeros(1, 1, T, 1), mem=DRAM)  # 1 for the chunk's real rows (MoE routing)
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
        # [p0 % R, p0 % W, p0]: chunk start for kernels/ink_mask.cpp
        self.P_BASE = ttnn.from_torch(torch.zeros(1, 8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.maskb = MaskBuilder(self.dev)
        g = self.dev.compute_with_storage_grid_size()
        self.p_sdpa = {n: ttnn.SDPAProgramConfig(compute_with_storage_grid_size=g, q_chunk_size=128 if n > 128 else 64,
                                                 k_chunk_size=128, exp_approx_mode=False) for n in {T, TS}}
        self.P_C0 = self.tt(torch.zeros(1, 1, 1, 1), ttnn.float32)  # p0 % R
        self.P_THR = self.tt(torch.zeros(1, 1, T, 1), ttnn.float32)  # min(P_t, W - 1)
        self.P_PCOL = self.tt(torch.zeros(1, 1, T, 1), ttnn.float32)  # P_t
        self.P_TAU = self.tt(torch.ones(1, T, 1, 1))  # tau(P_t), [1, T, 1, 1] (token-major q / r)
        self.P_MH = self.tt(torch.zeros(1, 1, 32, T + 32))  # XH -> the new 32-row history
        self.P_SEL = self.tt(torch.zeros(1, 1, 32, T))  # the last real row -> row 0 (32 rows for the head)
        # K / V rows -> the caches (head-major sources, one read-modify-write per cache tile on every core)
        self.kvw_ring = KVWriteRows(self.dev, T, NKV, HD, RA, ttnn.bfloat8_b)
        self.kvw_full = KVWriteRows(self.dev, T, NKV, HD, SA, ttnn.bfloat8_b)
        self.p_pc_gu = self._spc(2 * c.inter, (9, 8), 6)
        self.p_pc_dn = self._spc(H, (9, 8), 6)
        # output of the prefill MoE kernel (DRAM: written as whole tiles, consumed by ttnn ops)
        self.p_moe_out = self.tt(torch.zeros(1, 1, T, H), mem=DRAM)
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
        # columns reversed for kernels/ink_mask.cpp
        pr = ttnn.to_torch(d["proj"]).float()[:, :, :c.d_rel]
        if USE_SDPA:  # masks in units of 1 (see p_attention)
            pr = pr * self.scale
        d["p_proj"] = self.tt(torch.flip(pr, dims=[-1]))

    # ------------------------------------------------------------------------------------------------ pieces
    def p_attn_res(self, q, B):
        """Per-token attention residual: q [1,1,1,H] (pre-scaled), B [1, n, 32, H] normalized states."""
        s = ttnn.sum(ttnn.multiply(B, q, memory_config=PL), dim=-1, keepdim=True, memory_config=PL)  # [1,n,32,1]
        s = ttnn.permute(s, (0, 3, 2, 1), memory_config=PL)  # [1,1,32,n]
        p = ttnn.softmax(s, dim=-1, memory_config=PL, numeric_stable=True)
        p = ttnn.permute(p, (0, 3, 2, 1), memory_config=PL)  # [1,n,32,1]
        return ttnn.sum(ttnn.multiply(B, p, memory_config=PL), dim=1, keepdim=True, memory_config=PL)

    def p_stack(self, B, x_norm):
        return x_norm if B is None else ttnn.concat([B, x_norm], dim=1, memory_config=PL)

    def p_conv(self, x, hist, taps4, dyn4=None):
        k, C = self.cfg.conv_k, x.shape[-1]
        XH = ttnn.concat([x, hist], dim=-2, memory_config=PL)  # [1,1,T+32,C] (the new history)
        if T <= 32:
            Y = ttnn.reshape(self.mm(self.P_MALL, XH), (1, k, T, C))
            taps = taps4 if dyn4 is None else ttnn.add(dyn4, taps4, memory_config=PL)
            y = ttnn.sum(ttnn.multiply(Y, taps, memory_config=PL), dim=1, keepdim=True, memory_config=PL)
        else:  # by row tiles: [NT, 64, C] pairs (previous tile, tile), one [128, 64] shift matrix
            NT = T // 32
            prev = ttnn.concat([self.mm(self.P_FLIP, hist), ttnn.slice(x, (0, 0, 0, 0), (1, 1, T - 32, C),
                                                                       memory_config=PL)], dim=-2, memory_config=PL)
            XP = ttnn.concat([ttnn.reshape(prev, (1, NT, 32, C)), ttnn.reshape(x, (1, NT, 32, C))], dim=2,
                             memory_config=PL)
            Y = ttnn.reshape(self.mm(self.P_SHIFT, XP), (NT, k, 32, C))  # [i, lag, r]: x[32 i + r - lag]
            if dyn4 is None:
                taps = taps4  # [1, k, 1, C]
            else:  # [1, k, T, C] -> [NT, k, 32, C]
                taps = ttnn.add(ttnn.permute(ttnn.reshape(dyn4, (k, NT, 32, C)), (1, 0, 2, 3), memory_config=PL),
                                taps4, memory_config=PL)
            y = ttnn.reshape(ttnn.sum(ttnn.multiply(Y, taps, memory_config=PL), dim=1, keepdim=True, memory_config=PL),
                             (1, 1, T, C))
        self.mm(self.P_MH, XH, optional_output_tensor=hist, memory_config=hist.memory_config())
        return y

    def p_l2clip(self, x, thr, tau=None):
        ss = ttnn.sum(ttnn.multiply(x, x, memory_config=PL), dim=-1, keepdim=True, memory_config=PL)
        f = ttnn.unary_chain(ss, [UWP(U.RSQRT), UWP(U.MUL_UNARY_SFPU, thr), UWP(U.MINIMUM, 1.0)], memory_config=PL)
        if tau is not None:
            f = ttnn.multiply(f, tau, memory_config=PL)
        return ttnn.multiply(x, f, memory_config=PL)

    def p_attention(self, d, st, q_in, k_in, v_in, pc):
        c = self.cfg
        NH, NKV, HD, DR = c.n_heads, c.n_kv, c.head_dim, c.d_rel
        sliding = d["sliding"]
        nq, nkv = d["nq"], d["nkv"]
        fused = q_in is k_in and k_in is v_in
        if fused:
            y = d["Lqkv"](q_in, out_mem=PL)
            yq = y
            yk = ttnn.slice(y, (0, 0, 0, nq), (1, 1, T, nq + nkv), memory_config=PL)
            yv = ttnn.slice(y, (0, 0, 0, nq + nkv), (1, 1, T, nq + 2 * nkv), memory_config=PL)
        else:
            yq = d["Lq"](q_in, out_mem=PL)
            yk = d["Lk"](k_in, out_mem=PL)
            yv = d["Lv"](v_in, out_mem=PL)
        part = lambda y, lo, hi, nh, hd: ttnn.reshape(ttnn.slice(y, (0, 0, 0, lo), (1, 1, T, hi), memory_config=PL),
                                                      (1, T, nh, hd))
        tau = None if sliding else self.P_TAU
        q = ttnn.multiply(part(yq, 0, NH * HD, NH, HD), d["qscale"], memory_config=PL)
        q = self.p_l2clip(q, c.qk_clip, tau)  # [1, T, NH, HD]
        gate = part(yq, NH * HD, 2 * NH * HD, NH, HD)
        o0 = 2 * NH * HD
        r = ttnn.multiply(part(yq, o0, o0 + NH * DR, NH, DR), d["rscale"], memory_config=PL)
        r = self.p_l2clip(r, c.rel_clip, tau)
        r = ttnn.permute(r, (0, 2, 1, 3), memory_config=PL)  # [1, NH, T, DR]
        sink = ttnn.slice(yq, (0, 0, 0, o0 + NH * DR), (1, 1, T, o0 + NH * DR + NH), memory_config=PL)  # [1,1,T,NH]
        sink = ttnn.reshape(ttnn.transpose(sink, -2, -1, memory_config=PL), (1, NH, T, 1))
        sink = ttnn.pad(sink, [(0, 0), (0, 0), (0, 0), (0, 31)], 0.0)  # [1, NH, T, 32], col 0 = sink / scale
        if USE_SDPA:  # the SDPA adds the mask after scaling q k: masks in units of 1 (the bias: p_proj pre-scaled)
            sink = ttnn.multiply(sink, self.scale, memory_config=PL)
        C = NKV * HD
        if sliding:
            if fused:
                kvraw = ttnn.slice(y, (0, 0, 0, nq), (1, 1, T, nq + 2 * C), memory_config=PL)
            else:
                kvraw = ttnn.concat([ttnn.slice(yk, (0, 0, 0, 0), (1, 1, T, C), memory_config=PL),
                                     ttnn.slice(yv, (0, 0, 0, 0), (1, 1, T, C), memory_config=PL)], dim=-1,
                                    memory_config=PL)
            kv = self.p_conv(kvraw, st["hkv"], d["p_tkv"])
            k = ttnn.slice(kv, (0, 0, 0, 0), (1, 1, T, C), memory_config=PL)
            v = ttnn.slice(kv, (0, 0, 0, C), (1, 1, T, 2 * C), memory_config=PL)
        else:
            kraw = ttnn.slice(yk, (0, 0, 0, 0), (1, 1, T, C), memory_config=PL)
            vraw = ttnn.slice(yv, (0, 0, 0, 0), (1, 1, T, C), memory_config=PL)
            def dyn4(y):
                dd = ttnn.slice(y, (0, 0, 0, NKV * HD), (1, 1, T, NKV * HD + 32), memory_config=PL)
                return self.mm(ttnn.repeat(dd, (1, c.conv_k, 1, 1), memory_config=PL), self.P_ELAG)
            k = self.p_conv(kraw, st["hk"], d["p_tk"], dyn4(yk))
            v = self.p_conv(vraw, st["hv"], d["p_tv"], dyn4(yv))
        k = self.p_l2clip(ttnn.reshape(k, (1, T, NKV, HD)), c.qk_clip)
        v = self.rmsu(ttnn.reshape(v, (1, T, NKV, HD)))
        hm = lambda t: ttnn.permute(t, (0, 2, 1, 3), memory_config=PL)  # [1, NKV, T, HD]
        if sliding:
            self.kvw_ring(hm(k), hm(v), st["kc"], st["vc"], self.P_POSR)
        else:
            self.kvw_full(hm(k), hm(v), st["kc"], st["vc"], self.P_POSF)
        kc, vc = st["kc"], st["vc"]
        # mask [1, NH, rows, S] (units of 1 / scale)
        rel = self.mm(r, d["p_proj"])  # [1, NH, T, R or W] by distance
        if sliding:
            # [1, NH, T, RA] by ring slot (kernels/ink_mask.cpp: bias / validity / sink by selection)
            blocks = [(0, T, self.maskb(rel, sink, self.P_BASE, 0, self.RA, c.window, self.R, M=self.R))]
        else:
            b = self.p_bucket
            if b is not None:  # slots [0, b) + 32 (sink at b)
                kc = ttnn.concat([ttnn.slice(kc, (0, 0, 0, 0), (1, NKV, b, HD), memory_config=DRAM), self.P_KZ],
                                 dim=-2, memory_config=DRAM)
                vc = ttnn.concat([ttnn.slice(vc, (0, 0, 0, 0), (1, NKV, b, HD), memory_config=DRAM), self.P_KZ],
                                 dim=-2, memory_config=DRAM)
            blocks = []
            S_, sc = (self.SA, self.max_seq) if b is None else (b + KZ, b)  # cache columns, sink column
            for i0 in range(0, T, TS):  # query rows i0 .. i0 + TS
                blocks.append((i0, TS, self.maskb(rel, sink, self.P_BASE, 1, S_, c.window, sc, m0=i0 // 32, rows=TS)))
        os_ = []
        for i0, n, mask in blocks:
            if USE_SDPA:  # fused q k + mask + softmax + p v by key chunks (GQA)
                qb = ttnn.slice(q, (0, i0, 0, 0), (1, i0 + n, NH, HD), memory_config=PL) if n < T else q
                qh = ttnn.permute(qb, (0, 2, 1, 3), memory_config=DRAM)  # [1, NH, n, HD]
                o = ttnn.transformer.scaled_dot_product_attention(
                    qh, kc, vc, attn_mask=mask, is_causal=False, scale=self.scale, program_config=self.p_sdpa[n],
                    compute_kernel_config=self.hi, memory_config=DRAM)
                os_.append(ttnn.permute(o, (0, 2, 1, 3), memory_config=PL))  # [1, n, NH, HD]
                continue
            S = mask.shape[-1]
            mask = ttnn.reshape(mask, (1, NKV, (NH // NKV) * n, S))
            qb = ttnn.slice(q, (0, i0, 0, 0), (1, i0 + n, NH, HD), memory_config=PL) if n < T else q
            qg = ttnn.reshape(ttnn.permute(qb, (0, 2, 1, 3), memory_config=PL), (1, NKV, (NH // NKV) * n, HD))
            s = self.mm(qg, kc, transpose_b=True, memory_config=DRAM)  # [1, NKV, 3n, S]
            s = ttnn.multiply(ttnn.add(s, mask, memory_config=DRAM), self.scale, memory_config=DRAM)
            p = ttnn.softmax(s, dim=-1, memory_config=DRAM, numeric_stable=True)
            o = self.mm(p, vc, memory_config=PL)  # [1, NKV, 3n, HD]
            os_.append(ttnn.permute(ttnn.reshape(o, (1, NH, n, HD)), (0, 2, 1, 3), memory_config=PL))  # [1,n,NH,HD]
        o = os_[0] if len(os_) == 1 else ttnn.concat(os_, dim=1, memory_config=PL)  # [1, T, NH, HD]
        o = ttnn.multiply(o, gate, input_tensor_b_activations=[UWP(U.SIGMOID)], memory_config=PL)
        return d["Lo"](ttnn.reshape(o, (1, 1, T, NH * HD)), out_mem=PL)

    def p_moe(self, d, x):
        c = self.cfg
        logits = self.mm(x, d["Wg"], dtype=ttnn.float32)
        s = ttnn.sigmoid(logits, memory_config=PL)
        sb = ttnn.add(s, d["gbias"], memory_config=PL)
        kk = c.route_scale * c.gate_softcap * c.up_softcap
        if "Mgu" in d and T > 128:  # kernels/ink_moex_*: only the routed (token, expert) pairs, X rows by token
            _, ids = ttnn.topk(sb, c.top_k, dim=-1, largest=True, sorted=True, memory_config=DRAM)
            sg = ttnn.gather(s, -1, index=ids, memory_config=DRAM)  # [T, 8] the chosen experts' sigmoids
            wk = ttnn.multiply(sg, ttnn.sum(sg, dim=-1, keepdim=True, memory_config=DRAM),
                               input_tensor_b_activations=[UWP(U.RECIP), UWP(U.MUL_UNARY_SFPU, kk)],
                               memory_config=DRAM)
            wk = ttnn.multiply(ttnn.typecast(wk, ttnn.bfloat16), self.P_VALID, memory_config=DRAM)  # padding: skipped
            xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            return d["moek"].prefill_moex(xr, ids, wk, d["Mgu"], d["Mdn"], self.p_moe_out, nb=8)
        vals, _ = ttnn.topk(sb, c.top_k, dim=-1, largest=True, sorted=True, memory_config=PL)
        kth = ttnn.slice(vals, (0, 0, 0, c.top_k - 1), (1, 1, T, c.top_k), memory_config=PL)
        w = ttnn.multiply(s, ttnn.ge(sb, kth, memory_config=PL), memory_config=PL)  # dense routing weights
        w = ttnn.multiply(w, ttnn.sum(w, dim=-1, keepdim=True, memory_config=PL),
                          input_tensor_b_activations=[UWP(U.RECIP), UWP(U.MUL_UNARY_SFPU, kk)], memory_config=PL)
        w = ttnn.typecast(w, ttnn.bfloat16)
        sp = ttnn.to_layout(ttnn.max(w, dim=-2, keepdim=True, memory_config=PL), ttnn.ROW_MAJOR_LAYOUT)
        if "Mgu" in d:  # kernels/ink_moep_*: the used experts in batches, act, routing weights and sum in one program
            # compact: each expert's routed tokens packed into row tiles (kernels/ink_moec_*; TSUGUMI_INKLING_MOE_COMPACT=0:
            # all rows of every used expert, ink_moep_*)
            if T > 64 or env("INKLING_MOE_COMPACT", "1") == "1":
                # 128 rows: X in bfp8 (its 72 x 4 tiles must fit each MoE core's L1)
                xk = ttnn.typecast(x, ttnn.bfloat8_b) if T > 64 else x
                return d["moek"].prefill_compact(xk, w, sp, d["Mgu"], d["Mdn"], self.p_moe_out)
            return d["moek"].prefill(x, w, sp, d["Mgu"], d["Mdn"], self.p_moe_out)
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
        wp = ttnn.permute(w, (0, 3, 2, 1), memory_config=PL)  # [1, E, T, 1]
        return ttnn.sum(ttnn.multiply(dn, wp, memory_config=DRAM), dim=1, keepdim=True, memory_config=PL)

    def p_layer(self, a, h, B, pc, msgs=None):
        c = self.cfg
        _, i, _ = self.apps[a]
        d, st = self.layers[i], self.state[a]
        N = self.p_stack(B, self.rmsu(h))
        if d["sliding"]:
            ai = self.p_attn_res(d["p_qa"][0], N) if i > 0 else h
            if msgs is None:
                q_in = k_in = v_in = ttnn.rms_norm(ai, weight=d["ln"], epsilon=c.eps, memory_config=PL)
            else:
                na = self.rmsu(ai)
                q_in, k_in, v_in = (ttnn.rms_norm(self.msg_gate(d, wh, ai, m, na, rows=True), weight=d["ln"], epsilon=c.eps,
                                                  memory_config=PL)
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
            q_in = ttnn.rms_norm(qa, weight=d["lnq"], epsilon=c.eps, memory_config=PL)
            k_in = ttnn.rms_norm(ka, weight=d["lnk"], epsilon=c.eps, memory_config=PL)
            v_in = ttnn.rms_norm(va, weight=d["lnv"], epsilon=c.eps, memory_config=PL)
        o = self.p_attention(d, st, q_in, k_in, v_in, pc)
        o = self.p_conv(o, st["ha"], d["p_ta"])
        if c.boundary(i):
            B = N
            res = o
        else:
            res = ttnn.add(h, o, memory_config=PL)
        m_in = self.p_attn_res(d["p_qm"], self.p_stack(B, self.rmsu(res)))
        if msgs is not None:
            m_in = self.msg_gate(d, "mlp", m_in, msgs[3], rows=True)
        m = self.p_moe(d, ttnn.rms_norm(m_in, weight=d["lnm"], epsilon=c.eps, memory_config=PL))
        m = self.p_conv(m, st["hm"], d["p_tm"])
        return ttnn.add(res, m, memory_config=PL), B

    def p_consts(self):
        """Per-chunk device inputs shared by the layers (the masks are built per layer by kernels/ink_mask.cpp)."""
        return {}

    def p_forward(self, logits):
        self.mm_mem = PL
        try:
            return self._p_forward(logits)
        finally:
            self.mm_mem = L1

    def _p_forward(self, logits):
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
            NM = ttnn.concat([ttnn.slice(B, (0, 2, 0, 0), (1, 3, T, c.hidden), memory_config=PL), self.rmsu(h)],
                             dim=1, memory_config=PL)
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
        f = ttnn.rms_norm(f, weight=self.lnf, epsilon=c.eps, memory_config=PL)
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
        mh = torch.zeros(1, 1, 32, T + 32)
        for r in range(1, c.conv_k):  # new hist row r = x at position p0 + n - r
            j = n - r
            mh[0, 0, r, j if j >= 0 else T + (r - n)] = 1.0
        sel = torch.zeros(1, 1, 32, T)
        sel[0, 0, 0, n - 1] = 1.0
        put = lambda t, v, dt: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT if dt != ttnn.int32 else ttnn.ROW_MAJOR_LAYOUT), t)
        put(self.P_X0, x, ttnn.bfloat16)
        put(self.P_VALID, (torch.arange(T) < n).float().view(1, 1, T, 1), ttnn.bfloat16)
        put(self.P_POSR, torch.tensor(posr, dtype=torch.int32), ttnn.int32)
        put(self.P_POSF, torch.tensor(posf, dtype=torch.int32), ttnn.int32)
        put(self.P_C0, torch.full((1, 1, 1, 1), float(p0 % self.R)), ttnn.float32)
        bs = torch.zeros(1, 8, dtype=torch.int32)
        bs[0, 0], bs[0, 1], bs[0, 2] = p0 % self.R, p0 % c.window, p0
        put(self.P_BASE, bs, ttnn.int32)
        put(self.P_THR, torch.tensor([float(min(p, c.window - 1)) for p in P]).view(1, 1, T, 1), ttnn.float32)
        put(self.P_PCOL, torch.tensor([float(p) for p in P]).view(1, 1, T, 1), ttnn.float32)
        put(self.P_TAU, torch.tensor([c.tau(p) for p in P]).view(1, T, 1, 1), ttnn.bfloat16)
        put(self.P_MH, mh, ttnn.bfloat16)
        put(self.P_SEL, sel, ttnn.bfloat16)

    def prefill(self, tokens, logits=True, traced=True, progress=None):
        """Feed tokens from self.pos on, T per step. Returns the top-k after the last one if logits."""
        out = None
        for k in range(0, len(tokens), T):
            chunk = tokens[k:k + T]
            last = k + T >= len(tokens)
            want = logits and last
            if self.pos + len(chunk) > self.max_seq:
                raise RuntimeError(f"context full ({self.max_seq} tokens)")
            self.p_write_inputs(chunk, self.pos)
            self.p_bucket = next((b for b in self.p_buckets if self.pos + T <= b), None)
            key = ("prefill", True, self.p_bucket)  # traced with the lm_head only (see Engine.capture)
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
