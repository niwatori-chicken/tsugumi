"""Inkling 2 Mini decode engine on one Blackhole card (ttnn), one token per step, every op on the device so the
whole step can be traced.

Layout: activations are [1, 1, 32, C] tiles whose row 0 is the token (other rows are don't-care and are never
read into row 0: every row mix goes through a constant 0/1 matrix that only picks the rows that matter).

- Attention residuals (block states): the normalized block states are rows of a [32, H] matrix N; a residual is
  softmax(Q N^T + colmask) N with the learned queries Q as rows (1 or 3 per layer, 4 for the loop messages).
- Short convolutions: per application a [32, C] history (rows 1..3 = the 3 previous inputs); A = hist + E0 x,
  y = ONES0 (A * taps), hist <- SHIFT A.
- Attention: ttnn decode SDPA with an additive mask (the relative position bias, divided by the softmax scale)
  and the per-token sink logits (also / scale). Sliding layers keep a 1024-slot ring KV (slot = pos % 1024); the
  bias, computed in distance order, is permuted to slot order with a 0/1 matrix built on the device from the
  position (PERM = (DS == c), DS[d, s] = (d + s) % 1024). Full layers keep a linear KV of max_seq positions; the
  bias (window of 1024) is the ring bias repeated over the blocks of 1024, times the window mask.
- MoE: router in float32 on the device, top-8 (ttnn.topk), then ttnn.sparse_matmul in its indexed mode (only the
  8 selected experts' weights are read).
"""
import math
import time

import torch
import ttnn

from ..env import env
from ..kv_kernel import KVWriteKernel
from ..ops import DecodeLinear, cache_path, has_cache
from .attn_kernel import InkAttention
from .prep_kernel import InkPrep
from .act_kernel import InkAct, InkOGate
from .ares_kernel import InkAres
from .conv_kernel import InkConv
from .moe_kernel import InkMoE
from .gate_kernel import InkGate, InkGate1
from .route_kernel import InkRoute, InkRouter
from .config import InklingConfig
from .prefill import T as PT
from .prefill import PrefillMixin
from .convert import convert_layer, expert_dtype, expert_files
from .weights import Checkpoint

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
NEG = -1e9
U, UWP = ttnn.UnaryOpType, ttnn.UnaryWithParam
TOPK = 64
LM_CHUNKS = 8


def _ck(fid, fp32=True, approx=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32,
                                            packer_l1_acc=False)


class Engine(PrefillMixin):
    def __init__(self, dev, model_dir, cache_dir, max_seq=32768, log=print, wdtype=ttnn.bfloat8_b, experts=None,
                 prefill=True):
        self.dev, self.log = dev, log
        self.cfg = c = InklingConfig.load(model_dir)
        self.ck = Checkpoint(model_dir)
        self.cache = cache_dir
        W = c.window
        assert max_seq % W == 0, f"max_seq must be a multiple of {W}"
        self.max_seq = max_seq
        # ring KV of the sliding layers: R = W + PT slots (a PT-token prefill chunk is written before its
        # attention; the positions it evicts are older than any of its tokens' windows), then a sink slot (K = V = 0;
        # the prefill attention puts the per-token sink logits into the mask there) and a scratch slot (padding rows of
        # a chunk write there), padded to RA (multiple of 128). Full layers: max_seq + one block of W (sink at max_seq,
        # scratch at max_seq + 1).
        # PT: prefill chunk (32, 64 or 128); RA: R + the sink and scratch slots, in 128-key chunks (9 for PT <= 64)
        self.R, self.SA = W + PT, max_seq + W
        self.RA = (self.R + 2 + 127) // 128 * 128
        assert self.R + 2 <= self.RA
        self.wdtype = wdtype
        self.moek = None  # InkMoE (created with the first bfp8 / bfp4 expert layer)
        self.moeks = {}  # per weight format
        self.hi = _ck(ttnn.MathFidelity.HiFi4)
        self.mm_mem = L1  # self.mm output placement (prefill.p_forward sets DRAM at T = 128)
        self.hi2 = _ck(ttnn.MathFidelity.HiFi2)
        self.scale = c.head_dim ** -0.5
        if experts is None:
            experts = env("INKLING_EXPERTS", "bfp4")
        # bfp4 (default): decode +10%, prefill +3%, top-1 = bfp8 at 95.9% of teacher-forced positions (mostly
        # near-ties); auto: bfp8 if the DRAM holds all-bfp8 experts (~24.2 GiB), else mixed
        if experts == "auto":
            v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
            experts = "bfp8" if v.total_bytes_free_per_bank * v.num_banks >= 30 * (1 << 30) else "mixed"
        self.experts = experts
        log(f"experts: {experts}")
        t0 = time.perf_counter()
        self._constants()
        self.layers = [self._load_layer(i) for i in range(c.n_layers)]
        # the relative-bias projections of all sliding / full layers stacked: proj @ PERM is computed once per step
        # for all of them (rel = r @ (proj @ PERM)) instead of permuting each layer's bias
        for kind in ("sliding", "full"):
            ls = [d for d in self.layers if d["sliding"] == (kind == "sliding")]
            for j, d in enumerate(ls):
                d["prow"] = j * c.d_rel
            setattr(self, f"PROJ_{kind}", ttnn.concat([d["proj"] for d in ls], dim=-2))
        self._load_globals()
        self.apps = c.schedule()
        self._alloc_state()
        self.has_prefill = prefill
        if prefill:
            self.init_prefill()
        log(f"weights loaded in {time.perf_counter() - t0:.0f}s")
        self.traces = {}
        self.pos = 0
        self.dbg = None
        self.dbg_cur = {}

    # ------------------------------------------------------------------------------------------------ helpers
    def tt(self, x, dt=ttnn.bfloat16, mem=DRAM, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(x.contiguous(), dtype=dt, layout=layout, device=self.dev, memory_config=mem)

    def tt_cached(self, name, fn, dt=ttnn.bfloat16, mem=DRAM):
        """Small weights: cached as .tensorbin too (the checkpoint is not read when everything is cached)."""
        f = f"{self.cache}/{name}"
        if has_cache(f, dt):
            return ttnn.to_device(ttnn.load_tensor(cache_path(f, dt)), self.dev, memory_config=mem)
        t = ttnn.from_torch(fn().contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(cache_path(f, dt), t)
        return ttnn.to_device(t, self.dev, memory_config=mem)

    def w(self, name):
        return self.ck.get(name)

    def mm(self, a, b, **kw):
        kw.setdefault("memory_config", self.mm_mem)  # DRAM inside the 128-row prefill
        kw.setdefault("compute_kernel_config", self.hi)
        return ttnn.matmul(a, b, **kw)

    # ------------------------------------------------------------------------------------------------ setup
    def _constants(self):
        c = self.cfg
        H, W = c.hidden, c.window

        def m32(pairs):
            m = torch.zeros(1, 1, 32, 32)
            for r, s in pairs:
                m[0, 0, r, s] = 1.0
            return self.tt(m)

        self.E0 = m32([(0, 0)])
        self.SH = [m32([(r, 0)]) for r in range(8)]  # row 0 -> row r
        self.SEL = [m32([(0, r)]) for r in range(8)]  # row r -> row 0
        self.MOV20 = m32([(0, 2)])
        self.SHIFT = m32([(r + 1, r) for r in range(c.conv_k - 1)])
        self.ONES0 = m32([(0, r) for r in range(c.conv_k)])
        self.colmask = []
        for n in range(9):
            m = torch.full((1, 1, 32, 32), NEG)
            m[..., :n] = 0.0
            self.colmask.append(self.tt(m))
        R, RA, SA = self.R, self.RA, self.SA
        # PERM[d, s] = [(d + s) % R == c]: distance d -> ring slot s (slots >= R never match)
        ds = torch.full((R, RA), -1.0)
        ds[:, :R] = ((torch.arange(R)[:, None] + torch.arange(R)[None, :]) % R).float()
        self.DS = self.tt(ds[None, None], ttnn.float32)
        # full layers: the bias in position % W order (period W, repeated over the blocks of W)
        dsf = ((torch.arange(W)[:, None] + torch.arange(W)[None, :]) % W).float()
        self.DSF = self.tt(dsf[None, None], ttnn.float32)
        arw = torch.full((RA,), -1e6)  # slots >= R: huge distance (invalid)
        arw[:R] = torch.arange(R).float()
        self.ARW = self.tt(arw[None, None, None], ttnn.float32)
        ars = torch.full((SA,), 1e7)  # positions >= max_seq (prefill sink / scratch slots): never valid in decode
        ars[:self.max_seq] = torch.arange(self.max_seq).float()
        self.ARS = self.tt(ars[None, None, None], ttnn.float32)
        # dynamic conv helpers: D[lag, c] = dyn[head(c) * 4 + lag]
        nk, hd, k = c.n_kv, c.head_dim, c.conv_k
        rr = torch.zeros(1, 1, 32, nk * hd)
        rr[0, 0, 0] = 1.0
        self.ROW0ONES = self.tt(rr)
        eh = torch.zeros(1, 1, 32, nk * hd)
        for j in range(nk * k):
            eh[0, 0, j, (j // k) * hd:(j // k + 1) * hd] = 1.0 / math.sqrt(2 * k + 1)  # conv normalizer folded in
        self.EH = self.tt(eh)
        lm = torch.zeros(1, 1, 32, 32)
        for j in range(nk * k):
            lm[0, 0, j % k, j] = 1.0
        self.LAG = self.tt(lm)
        # per-step inputs (host writes them before each step)
        self.X0 = self.tt(torch.zeros(1, 1, 32, H))
        self.POS_I = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.RING_I = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.D_RING = ttnn.from_torch(torch.zeros(8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.D_POSF = ttnn.from_torch(torch.zeros(8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.D_PT0 = ttnn.from_torch(torch.zeros(1, 8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.POSF = self.tt(torch.zeros(1, 1, 1, 1), ttnn.float32)
        self.THR = self.tt(torch.zeros(1, 1, 1, 1), ttnn.float32)  # min(pos, window - 1)
        self.CF = self.tt(torch.zeros(1, 1, 1, 1), ttnn.float32)  # pos % R (ring slot)
        self.CW = self.tt(torch.zeros(1, 1, 1, 1), ttnn.float32)  # pos % W
        self.TAU = self.tt(torch.ones(1, 1, 1, 1))
        self.SPARSITY = self.tt(torch.ones(1, 1, 1, c.n_experts), layout=ttnn.ROW_MAJOR_LAYOUT)
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, hd), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        self.sdpa_ring = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                                q_chunk_size=0, k_chunk_size=128)  # RA = 9 x 128
        self.sdpa_full = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                                q_chunk_size=0, k_chunk_size=256)
        # float32 accumulation everywhere: bf16 accumulation over K = 1152 / 2304 biases the outputs by 1-4%
        self.lofi = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                                     fp32_dest_acc_en=True, packer_l1_acc=True)
        self.hifi2 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                      fp32_dest_acc_en=True, packer_l1_acc=True)

        def spc(n, cores, bw):
            nc = cores[0] * cores[1]
            pcn = math.ceil(math.ceil(n / 32) / nc)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*cores), in0_block_w=bw, out_subblock_h=1,
                out_subblock_w=1, out_block_h=1, out_block_w=pcn, per_core_M=1, per_core_N=pcn, fuse_batch=False,
                fused_activation=None, mcast_in0=True)

        self.pc_gu = spc(2 * c.inter, (9, 8), 6)  # scripts: 139 us gate_up / 75 us down for 8 bfp8 experts
        self.pc_dn = spc(H, (9, 8), 6)

    def _fp32acc(self, lin):
        """DecodeLinear with float32 accumulation (its default accumulates in bf16)."""
        fid = ttnn.MathFidelity.LoFi if self.wdtype == ttnn.bfloat4_b else ttnn.MathFidelity.HiFi2
        lin.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                                  packer_l1_acc=True)
        return lin

    def _conv_taps(self, w_ck4):
        """Static conv weight [C, 4] (index 3 = current token) -> taps [1,1,32,C] with row r = lag r, the residual
        (+1 on lag 0) and the 1/sqrt(k+1) normalizer folded in: y = ONES0 (A * taps) is the conv output."""
        k = w_ck4.shape[1]
        t = torch.zeros(1, 1, 32, w_ck4.shape[0])
        for r in range(k):
            t[0, 0, r] = w_ck4[:, k - 1 - r]
        t[0, 0, 0] += 1.0
        return t / math.sqrt(k + 1)

    def _load_layer(self, i):
        c = self.cfg
        H, NH, NKV, HD, DR = c.hidden, c.n_heads, c.n_kv, c.head_dim, c.d_rel
        a = f"layers.{i}.attn."
        L = f"layers.{i}."
        sliding = c.sliding(i)
        cp = lambda s: f"{self.cache}/L{i}.{s}"
        d = dict(i=i, sliding=sliding)
        sc = self.scale

        def lin(name, n, k, build):
            w = None if has_cache(cp(name), self.wdtype) else build().to(torch.bfloat16)
            return self._fp32acc(DecodeLinear(self.dev, w, self.wdtype, cache_file=cp(name), name=f"L{i}.{name}",
                                              shape=(k, n)))

        nq = NH * HD * 2 + NH * DR + NH
        d["Lq"] = lin("q", nq, H, lambda: torch.cat([
            self.w(a + "wq_du.weight"), self.w(a + "gate_attn.weight"), self.w(a + "wr_du.weight"),
            self.w(a + "attn_sink_proj.weight") / sc]).T)
        if sliding:
            d["Lk"] = lin("k", NKV * HD, H, lambda: self.w(a + "wk_dv.weight").T)
            d["Lv"] = lin("v", NKV * HD, H, lambda: self.w(a + "wv_dv.weight").T)
        else:
            d["Lk"] = lin("k", NKV * HD + 32, H, lambda: torch.cat(
                [self.w(a + "wk_dv.weight"), self.w(a + "k_sconv.weight_proj")]).T)
            d["Lv"] = lin("v", NKV * HD + 32, H, lambda: torch.cat(
                [self.w(a + "wv_dv.weight"), self.w(a + "v_sconv.weight_proj")]).T)
        nkv = NKV * HD + (0 if sliding else 32)
        nqa = nq + (-nq) % 32  # q | gate | r | sink padded to 32 columns: k and v start on column tiles
        d["nq"], d["nkv"] = nqa, nkv
        d["Lqkv"] = lin("qkv2", nqa + 2 * nkv, H, lambda: torch.cat(
            [self.w(a + "wq_du.weight"), self.w(a + "gate_attn.weight"), self.w(a + "wr_du.weight"),
             self.w(a + "attn_sink_proj.weight") / sc, torch.zeros(nqa - nq, H), self.w(a + "wk_dv.weight")]
            + ([] if sliding else [self.w(a + "k_sconv.weight_proj")]) + [self.w(a + "wv_dv.weight")]
            + ([] if sliding else [self.w(a + "v_sconv.weight_proj")])).T)
        d["Lo"] = lin("o", H, NH * HD, lambda: self.w(a + "wo_ud.weight").T)
        t = lambda name, fn, dt=ttnn.bfloat16: self.tt_cached(f"L{i}.{name}", fn, dt)
        d["qscale"] = t("qscale", lambda: self.w(a + "q_l2_clip.scale").view(1, 1, NH, HD))
        d["rscale"] = t("rscale", lambda: self.w(a + "rel_l2_clip.scale").view(1, 1, NH, DR))
        if sliding:
            d["proj"] = t("projR", lambda: torch.nn.functional.pad(self.w(a + "rel_logits_proj.proj") / sc,
                                                                    (0, self.R - c.window)).view(1, 1, DR, -1))
        else:
            d["proj"] = t("proj", lambda: (self.w(a + "rel_logits_proj.proj") / sc).view(1, 1, DR, -1))
        if sliding:
            d["tk"] = t("ctk", lambda: self._conv_taps(self.w(a + "k_sconv.weight")[:, 0]))
            d["tv"] = t("ctv", lambda: self._conv_taps(self.w(a + "v_sconv.weight")[:, 0]))
            d["ln"] = t("ln", lambda: self.w(L + "attn_norm.weight").view(1, H))
        else:
            def st(name):  # static taps (+1 on lag 0 for the residual) / sqrt(2k + 1)
                x = torch.zeros(1, 1, 32, NKV * HD)
                x[0, 0, :c.conv_k] = self.w(a + name)
                x[0, 0, 0] += 1.0
                return x / math.sqrt(2 * c.conv_k + 1)
            d["tk"] = t("ctk", lambda: st("k_sconv.static_w"))
            d["tv"] = t("ctv", lambda: st("v_sconv.static_w"))
            d["lnq"] = t("lnq", lambda: self.w(L + "q_input_norm.weight").view(1, H))
            d["lnk"] = t("lnk", lambda: self.w(L + "k_input_norm.weight").view(1, H))
            d["lnv"] = t("lnv", lambda: self.w(L + "v_input_norm.weight").view(1, H))
        if sliding:
            d["tkv"] = self.tt(torch.cat([ttnn.to_torch(d["tk"]), ttnn.to_torch(d["tv"])], dim=-1).float())
        d["ta"] = t("cta", lambda: self._conv_taps(self.w(L + "attn_sconv.weight")[:, 0]))
        d["tm"] = t("ctm", lambda: self._conv_taps(self.w(L + "mlp_sconv.weight")[:, 0]))
        d["lnm"] = t("lnm", lambda: self.w(L + "mlp_norm.weight").view(1, H))
        hs = H ** -0.5

        def queries(name, rows):
            def f():
                q = torch.zeros(1, 1, 32, H)
                q[0, 0, :rows.shape[0]] = rows * hs
                return q
            return t(name, f)

        if i > 0:
            d["Qa"] = queries("Qa", self.w(f"attn_res.attn_queries.{i - 1}").view(-1, H))
            d["nqa"] = 1 if sliding else 3
            d["qa_row"] = ttnn.slice(d["Qa"], (0, 0, 0, 0), (1, 1, 1, H))
        d["Qm"] = queries("Qm", self.w(f"attn_res.mlp_queries.{i}").view(-1, H))
        d["qm_row"] = ttnn.slice(d["Qm"], (0, 0, 0, 0), (1, 1, 1, H))
        d["Wg"] = t("router", lambda: self.w(L + "mlp.gate.weight").T.reshape(1, 1, H, -1))
        # [E, H] float32 for the router as multiply + row sums (a K=2304, N=128 matmul runs on 4 cores, 53 us)
        d["WgT"] = t("routerT", lambda: self.w(L + "mlp.gate.weight").reshape(1, 1, -1, H), ttnn.float32)
        d["gbias"] = t("router_bias", lambda: self.w(L + "mlp.gate.bias").view(1, 1, 1, -1), ttnn.float32)
        if c.is_loop(i):
            j = i - c.loop_start
            qa = self.w("middle_cycle_cross_pass_queries") if not has_cache(cp("Qmsg"), ttnn.bfloat16) else None
            kv = self.w("middle_cycle_cross_pass_kv_queries") if qa is not None else None
            d["Qmsg"] = queries("Qmsg", torch.stack([qa[2 * j], kv[2 * j], kv[2 * j + 1], qa[2 * j + 1]])
                                if qa is not None else None)
            for wh in ("attn", "k", "v", "mlp"):
                g = f"layers.{i}.controllers.{wh}."
                d[f"g_{wh}_d"] = t(f"g_{wh}_d", lambda: self.w(g + "down_weight").T.reshape(1, 1, 2 * H, -1))
                d[f"g_{wh}_dT"] = t(f"g_{wh}_dT", lambda: self.w(g + "down_weight").reshape(1, 1, -1, 2 * H))
                d[f"g_{wh}_fg"] = t(f"g_{wh}_fg", lambda: torch.cat(
                    [self.w(g + "forget_weight"), self.w(g + "gain_weight")]).T.reshape(1, 1, c.gate_rank, -1))
        # experts
        edt = expert_dtype(self.ck, i, self.experts)
        if not all(has_cache(f, edt) for f in expert_files(self.cache, i)):
            convert_layer(self.ck, self.cache, i, log=self.log, policy=self.experts)
        fgu, fdn = expert_files(self.cache, i)
        d["Egu"] = ttnn.to_device(ttnn.load_tensor(cache_path(fgu, edt)), self.dev, memory_config=DRAM)
        d["Edn"] = ttnn.to_device(ttnn.load_tensor(cache_path(fdn, edt)), self.dev, memory_config=DRAM)
        if edt in (ttnn.bfloat8_b, ttnn.bfloat4_b) and env("INKLING_MOE_KERNEL", "1") == "1":
            # one-program MoE kernels (decode and 32-row prefill) on a bank-contiguous copy of the experts; the
            # interleaved tensors are freed (DRAM cannot hold both). One kernel object per weight format.
            if edt not in self.moeks:
                self.moeks[edt] = InkMoE(self.dev, c.n_experts, c.hidden, c.inter, c.gate_softcap, c.up_softcap,
                                         wdtype=edt)
            self.moek = self.moeks[edt]
            d["moek"] = self.moek
            d["Mgu"], d["Mdn"] = self.moek.prepare(d["Egu"], d["Edn"])
            ttnn.deallocate(d.pop("Egu"))
            ttnn.deallocate(d.pop("Edn"))
        # LoFi drops mantissa bits of bfp8 weights; HiFi2 covers them (bfp4 needs only LoFi)
        d["eck"] = self.lofi if edt == ttnn.bfloat4_b else self.hifi2
        if i % 4 == 3:
            self.log(f"loaded layer {i + 1}/{c.n_layers}")
        return d

    def _load_globals(self):
        c = self.cfg
        H = c.hidden
        self.Qf = self.tt_cached("Qfinal", lambda: torch.cat(
            [self.w("attn_res.output_query").view(1, 1, 1, H) * H ** -0.5, torch.zeros(1, 1, 31, H)], 2))
        self.lnf = self.tt_cached("norm", lambda: self.w("norm.weight").view(1, H))
        self.qf_row = ttnn.slice(self.Qf, (0, 0, 0, 0), (1, 1, 1, H))
        # lm_head in LM_CHUNKS chunks of equal width (the DRAM-sharded matmul needs a bounded N per core); the
        # vocabulary is zero-padded to LM_CHUNKS * cw (padded ids are dropped on the host)
        V = c.vocab_unpadded
        self.cw = cw = math.ceil(V / LM_CHUNKS / 32) * 32
        self.lm_chunks = []
        full = None
        for j in range(LM_CHUNKS):
            cf = f"{self.cache}/lm_head.{j}"
            w = None
            if not has_cache(cf, self.wdtype):
                if full is None:
                    full = self.ck.get_slice("unembed.weight")[:V]
                part = full[j * cw: min(V, (j + 1) * cw)].float()
                w = torch.zeros(H, cw, dtype=torch.bfloat16)
                w[:, :part.shape[0]] = part.T.to(torch.bfloat16)
            self.lm_chunks.append(self._fp32acc(DecodeLinear(self.dev, w, self.wdtype, cache_file=cf,
                                                             name=f"lm_head.{j}", shape=(H, cw))))
        self.embed = self.ck.get_slice("embed.weight")
        self.embed_norm = self.w("embed_norm.weight")

    def _alloc_state(self):
        c = self.cfg
        NKV, HD, H = c.n_kv, c.head_dim, c.hidden
        self.state = []
        for a, i, p in self.apps:
            S = self.RA if c.sliding(i) else self.SA
            z = lambda n: ttnn.zeros([1, 1, 32, n], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
            st = dict(
                kc=ttnn.zeros([1, NKV, S, HD], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.dev),
                vc=ttnn.zeros([1, NKV, S, HD], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.dev),
                ha=z(H), hm=z(H))
            if c.sliding(i):  # k and v convolved together (one conv over 2 * NKV * HD channels)
                st["hkv"] = z(2 * NKV * HD)
            else:
                st["hk"], st["hv"] = z(NKV * HD), z(NKV * HD)
            self.state.append(st)
        self.zeros = {n: ttnn.zeros([1, 1, 32, n], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
                      for n in (NKV * HD, 2 * NKV * HD, H)}
        self.kvw1_ring = KVWriteKernel(self.dev, 1, NKV, HD, self.RA, 1, ttnn.bfloat8_b)
        # decode attention kernels (bias / validity / sink in the mask; full layers read up to the position only)
        # chunks of 128 keys; 9 splits per KV head for the ring (17 chunks), 13 for full layers (long contexts)
        self.att_ring = InkAttention(self.dev, ring=True, bsl=self.RA, r=self.R, w=c.window, sink=self.R, wr=self.R,
                                     chunk_tiles=4, splits=self.RA // 128)
        self.att_full = InkAttention(self.dev, ring=False, bsl=self.SA, r=self.R, w=c.window, sink=self.max_seq,
                                     wr=c.window, chunk_tiles=4, splits=13)
        self.D_NPR = ttnn.from_torch(torch.zeros(8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        # q / r / k / v preparation (scale, clips, convs, rms, history) of a token in one program
        self.prep = InkPrep(self.dev, NH=c.n_heads, NKV=NKV, HD=HD, DR=c.d_rel, qk_clip=c.qk_clip,
                            rel_clip=c.rel_clip, eps=c.eps)
        # MoE routing (sigmoid, bias, top-k, weights) in one program; route_scale and the two softcap factors
        # (act = gc*tanh(g/gc) * sigmoid(g) * uc*tanh(u/uc)) are in the weights
        self.route = InkRoute(self.dev, E=c.n_experts, K=c.top_k,
                              kscale=c.route_scale * c.gate_softcap * c.up_softcap)
        self.router = InkRouter(self.dev, E=c.n_experts, K=c.top_k, kscale=c.route_scale * c.gate_softcap * c.up_softcap,
                                H=c.hidden) if env("INKLING_ROUTER", "1") == "1" else None
        # message gates: one program (in-program barrier; TSUGUMI_INKLING_GATE1=0: the two-program kernel)
        G = InkGate1 if env("INKLING_GATE1", "1") == "1" else InkGate
        self.gate = G(self.dev, H=c.hidden, R=c.gate_rank, eps=c.eps)
        # post-attention / post-MLP conv + residual in one program, into persistent L1 buffers: residual and output
        # of application a rotate over 3 buffers (never the input's), the loop input (reused at every pass) has its own
        self.cvk = InkConv(self.dev, c.hidden)
        self.fuse_moe = True  # act kernel + expert combine in the post-MLP conv (False: ttnn ops, for comparisons)
        # (unused, and its 590 KB of L1 not allocated, when the MoE kernels run)
        self.act = InkAct(self.dev, c.top_k, c.inter, c.gate_softcap, c.up_softcap) \
            if any("Mgu" not in d for d in self.layers) else None
        # attention output gate -> the o projections' (shared) width-sharded input layout. Off by default: no faster
        # in the full step (the slice / multiply / conversion it replaces overlap their neighbours' dispatch) and its
        # persistent sharded buffer (12 KB per core) breaks the L1 budget of the prefill programs
        self.ogate = None
        if env("INKLING_OGATE", "0") == "1":
            lo_mem = [d["Lo"].xmem for d in self.layers]
            assert all(m == lo_mem[0] for m in lo_mem)
            self.lo_in = ttnn.from_torch(torch.zeros(1, 1, 32, c.n_heads * HD), dtype=ttnn.bfloat16,
                                         layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=lo_mem[0])
            self.ogate = InkOGate(self.dev, c.n_heads * HD // 32)
        # message path of the sliding layers: the q / k / v gate outputs packed in rows 0 / 1 / 2 of one tile, so that
        # one rms norm and the fused qkv linear serve all three
        self.qkv3 = ttnn.from_torch(torch.zeros(1, 1, 32, c.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=self.dev, memory_config=L1)
        zb = lambda mem=L1: ttnn.from_torch(torch.zeros(1, 1, 32, c.hidden), dtype=ttnn.bfloat16,
                                            layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=mem)
        self.cvbuf = [zb(DRAM) for _ in range(3)]  # DRAM: read as whole tiles only; L1 is tight in prefill
        self.cvpre = ttnn.from_torch(torch.zeros(1, 1, 32, c.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=self.dev, memory_config=DRAM)
        # block attention residual (+ rms norm with weight) in two programs: per-layer query / norm tensors, the
        # block matrices of the boundary applications in their own DRAM buffers (they live until the next boundary)
        self.ares = InkAres(self.dev, c.hidden, c.eps)
        for i, d in enumerate(self.layers):
            if i > 0:
                d["arQ"] = self.ares.queries(ttnn.to_torch(d["Qa"]).float()[0, 0, :d["nqa"]])
                d["arLN"] = [self.ares.norm(d[k]) for k in (("ln",) if d["sliding"] else ("lnq", "lnk", "lnv"))]
            d["arQm"] = self.ares.queries(ttnn.to_torch(d["Qm"]).float()[0, 0, :1])
            d["arLNm"] = [self.ares.norm(d["lnm"])]
        xm = [d[k].xmem for d in self.layers for k in ("Lqkv", "Lq", "Lk", "Lv") if k in d] + \
            [lin.xmem for lin in self.lm_chunks]
        assert all(m == xm[0] for m in xm), "linears with K = hidden need one input sharding"
        self.ares.set_sharded(xm[0])  # attention / final norms straight into the linears' input layout
        for d in self.layers:
            if "Qmsg" in d:
                d["arQmsg"] = self.ares.queries(ttnn.to_torch(d["Qmsg"]).float()[0, 0, :4])
        self.hmsg = self.ares.zbuf(DRAM)  # the previous pass's output (messages of the next pass)
        self.msgbuf = [self.ares.zbuf(DRAM) for _ in range(4)]
        self.arQf = self.ares.queries(ttnn.to_torch(self.Qf).float()[0, 0, :1])
        self.arLNf = [self.ares.norm(self.lnf)]
        self.artout = {a: self.ares.zbuf(DRAM) for a, (_, i, _) in enumerate(self.apps) if i > 0 and c.boundary(i)}
        C = NKV * HD
        for d in self.layers:
            if d["sliding"]:
                tkv = ttnn.to_torch(d["tkv"]).float()
                d["ptaps"] = InkPrep.taps_tensor(self.dev, tkv[..., :C], tkv[..., C:])
            else:
                d["ptaps"] = InkPrep.taps_tensor(self.dev, d["tk"], d["tv"])
        self.D_NPF = ttnn.from_torch(torch.zeros(8, dtype=torch.int32), dtype=ttnn.int32, device=self.dev)
        self.kvw1_full = KVWriteKernel(self.dev, 1, NKV, HD, self.SA, 1, ttnn.bfloat8_b)
        # one snapshot of the position-independent state (ring KVs of the sliding layers, convolution histories):
        # with it, the state after a prompt can be restored after the reply was generated (the full-attention KV
        # needs no copy: positions past the snapshot are masked)
        self.snap = []
        for (a, i, p), st in zip(self.apps, self.state):
            keys = ("kc", "vc", "hkv", "ha", "hm") if c.sliding(i) else ("hk", "hv", "ha", "hm")
            self.snap.append({k: ttnn.zeros(st[k].shape, dtype=st[k].dtype, layout=ttnn.TILE_LAYOUT, device=self.dev)
                              for k in keys})
        self.snap_pos = None

    def reset(self):
        """New sequence: clear the convolution histories (KV entries past the position are masked). Copies from
        zero tensors allocated before the trace capture (no device allocation here)."""
        for st in self.state:
            for k in ("hk", "hv", "hkv", "ha", "hm"):
                if k in st:
                    ttnn.copy(self.zeros[st[k].shape[-1]], st[k])
        self.pos = 0

    def snapshot_save(self):
        for st, sn in zip(self.state, self.snap):
            for k, t in sn.items():
                ttnn.copy(st[k], t)
        self.snap_pos = self.pos

    def snapshot_load(self):
        for st, sn in zip(self.state, self.snap):
            for k, t in sn.items():
                ttnn.copy(t, st[k])
        self.pos = self.snap_pos

    # ------------------------------------------------------------------------------------------------ pieces
    def rmsu(self, x):
        mc = DRAM if x.memory_config().buffer_type == ttnn.BufferType.DRAM else L1  # 128-row prefill: DRAM
        return ttnn.rms_norm(x, epsilon=self.cfg.eps, memory_config=mc)

    def rowdot(self, WT, x):
        """x row 0 [1, K] @ WT^T, WT [N, K] (small N): multiply + row sums on many cores. Returns [1,1,1,N]."""
        r = ttnn.slice(x, (0, 0, 0, 0), (1, 1, 1, x.shape[-1]), memory_config=L1)
        if WT.dtype != r.dtype:
            r = ttnn.typecast(r, WT.dtype)
        s = ttnn.sum(ttnn.multiply(WT, r, memory_config=L1), dim=-1, keepdim=True, memory_config=L1)  # [1,1,N,1]
        return ttnn.transpose(s, -2, -1, memory_config=L1)

    def attn_res1(self, q, N, n):
        """attn_res with one query (q [1,1,1,H] pre-scaled): scores as row sums (the K=2304, N=32 matmul runs on one
        core). All 32 output rows hold the result."""
        s = ttnn.transpose(ttnn.sum(ttnn.multiply(N, q, memory_config=L1), dim=-1, keepdim=True, memory_config=L1),
                           -2, -1, memory_config=L1)  # [1,1,1,32]
        p = ttnn.softmax(ttnn.add(self.colmask[n], s, memory_config=L1), dim=-1, memory_config=L1,
                         numeric_stable=True)  # [32, 32], equal rows
        return self.mm(p, N)

    def attn_res(self, Q, N, n):
        """softmax(Q N^T + mask(n)) N; Q rows = queries (pre-scaled), N rows 0..n-1 = normalized states."""
        s = self.mm(Q, N, transpose_b=True)
        s = ttnn.add(s, self.colmask[n], memory_config=L1)
        p = ttnn.softmax(s, dim=-1, memory_config=L1, numeric_stable=True)
        return self.mm(p, N)

    def stack(self, NB, nb, x_norm):
        """Block matrix with x_norm (row 0) appended as row nb."""
        t = self.mm(self.SH[nb], x_norm)
        return t if NB is None else ttnn.add(NB, t, memory_config=L1)

    def conv(self, x, hist, taps, dyn=None):
        """Short conv over [x_t (row 0), hist rows 1..3] (residual and normalizer are in the taps); updates hist."""
        A = ttnn.add(hist, self.mm(self.E0, x), memory_config=L1)
        if dyn is not None:
            taps = ttnn.add(taps, dyn, memory_config=L1)
        y = self.mm(self.ONES0, ttnn.multiply(A, taps, memory_config=L1))
        self.mm(self.SHIFT, A, optional_output_tensor=hist, memory_config=hist.memory_config())
        return y

    def dyn_taps(self, d32):
        """[1,1,32,32] (row 0 = dynamic conv weights, col head*4 + lag) -> [1,1,32,1024] (row lag)."""
        z = ttnn.transpose(self.mm(self.E0, d32), -2, -1, memory_config=L1)  # col 0 = weights
        r = self.mm(z, self.ROW0ONES)
        return self.mm(self.LAG, ttnn.multiply(r, self.EH, memory_config=L1))

    def l2clip(self, x, thr, tau=None):
        """x [1,1,n,d]: x * min(thr / |x|, 1) (* tau)."""
        ss = ttnn.sum(ttnn.multiply(x, x, memory_config=L1), dim=-1, keepdim=True, memory_config=L1)
        f = ttnn.unary_chain(ss, [UWP(U.RSQRT), UWP(U.MUL_UNARY_SFPU, thr), UWP(U.MINIMUM, 1.0)], memory_config=L1)
        if tau is not None:
            f = ttnn.multiply(f, tau, memory_config=L1)
        return ttnn.multiply(x, f, memory_config=L1)

    def row(self, y, lo, hi, shape=None):
        t = ttnn.slice(y, (0, 0, 0, lo), (1, 1, 1, hi), memory_config=L1)
        return t if shape is None else ttnn.reshape(t, shape, memory_config=L1)

    def msg_gate(self, d, wh, h, msg, nh=None, rows=False):
        """(1 - tanh(F z)) h + tanh(G z) msg, z = D [norm(h), norm(msg)]; nh = norm(h) if already computed.
        rows=False (decode): z from row 0 only (multiply + row sums); rows=True (prefill): z per row (matmul)."""
        if not rows:  # decode: kernels/ink_gz_*, ink_go_* (z, the norms, f / g and the mix in two programs)
            return self.gate(h, msg, d[f"g_{wh}_d"], d[f"g_{wh}_fg"], wh)
        nh = self.rmsu(h) if nh is None else nh
        mc = DRAM if h.memory_config().buffer_type == ttnn.BufferType.DRAM else L1  # 128-row prefill: DRAM
        cat = ttnn.concat([nh, self.rmsu(msg)], dim=-1, memory_config=mc)
        z = self.mm(cat, d[f"g_{wh}_d"]) if rows else self.rowdot(d[f"g_{wh}_dT"], cat)
        fg = self.mm(z, d[f"g_{wh}_fg"])  # [1,1,1,2H]: broadcast over the rows of h / msg
        H = self.cfg.hidden
        f = ttnn.slice(fg, (0, 0, 0, 0), (1, 1, fg.shape[-2], H), memory_config=mc)
        g = ttnn.slice(fg, (0, 0, 0, H), (1, 1, fg.shape[-2], 2 * H), memory_config=mc)
        out = ttnn.multiply(f, h, input_tensor_a_activations=[UWP(U.TANH), UWP(U.RSUB, 1.0)], memory_config=mc)
        return ttnn.add(out, ttnn.multiply(g, msg, input_tensor_a_activations=[UWP(U.TANH)], memory_config=mc),
                        memory_config=mc)

    def attention(self, d, st, q_in, k_in, v_in, sc, rows=(0, 0, 0)):
        c = self.cfg
        NH, NKV, HD, DR = c.n_heads, c.n_kv, c.head_dim, c.d_rel
        sliding = d["sliding"]
        nq, nkv = d["nq"], d["nkv"]
        fused = q_in is k_in and k_in is v_in
        # decode: the linears' width-sharded L1 outputs are read as they are (the kernels use tensor accessors)
        om = L1 if self.dbg is not None else None
        # (the prep kernel reads q, k and v with the accessor of q's tensor: separate sharded outputs would need
        # their own accessors, so the separate linears of the full layers keep interleaved outputs)
        omq = om if fused else L1
        if fused:  # one linear for q, k and v
            y = d["Lqkv"](q_in, out_mem=omq)
            yq = y
        else:
            yq = d["Lq"](q_in, out_mem=omq)
            yk = d["Lk"](k_in, out_mem=omq)
            yv = d["Lv"](v_in, out_mem=omq)
        o0 = 2 * NH * HD
        sink = self.row(yq, o0 + NH * DR, o0 + NH * DR + NH)  # [1,1,1,NH] / scale (unused with the sink page)
        C = NKV * HD
        if fused:  # column tiles of k / v in the fused output (tile aligned)
            ks, ko, vs, vo = y, nq // 32, y, (nq + nkv) // 32
        else:
            ks, ko, vs, vo = yk, 0, yv, 0
        if sliding:
            hist = dict(hist_k=st["hkv"], hk_off=0, hist_v=st["hkv"], hv_off=C // 32)
        else:
            hist = dict(hist_k=st["hk"], hk_off=0, hist_v=st["hv"], hv_off=0, k_dyn=(ks, ko + C // 32),
                        v_dyn=(vs, vo + C // 32))
        # one program: q * scale -> clip (* tau), r likewise, k / v convolutions (history shifted) -> clip / rms
        q, r, k, v = self.prep(yq, 0, o0 // 32, ks, ko, vs, vo, taps=d["ptaps"], qscale=d["qscale"],
                               rscale=d["rscale"], tau=None if sliding else self.TAU, rows=rows, **hist)
        # one kernel writes this token's K and V rows (kernels/kv_write.cpp; the cache as a single block)
        if sliding:
            self.kvw1_ring(k, v, st["kc"], st["vc"], self.D_RING, self.D_PT0)
        else:
            self.kvw1_full(k, v, st["kc"], st["vc"], self.D_POSF, self.D_PT0)
        rel = self.mm(r, d["proj"])  # [1,1,NH,R or W]: the relative bias by distance (units of 1 / scale)
        att = self.att_ring if sliding else self.att_full
        if self.dbg is None and env("INKLING_SINKP", "1") == "1":  # sink logits from their tile of the linear output
            assert (o0 + NH * DR) % 32 == 0
            o = att(q, st["kc"], st["vc"], rel, yq, self.D_POSF, self.D_NPR if sliding else self.D_NPF,
                    sink_page=(o0 + NH * DR) // 32)
        else:
            o = att(q, st["kc"], st["vc"], rel, sink, self.D_POSF, self.D_NPR if sliding else self.D_NPF)  # row 0
        if self.dbg is not None:
            g = lambda t: ttnn.to_torch(t).float()
            self.dbg_cur.update(q=g(q)[0, 0, :NH], k=g(k)[0, 0, :NKV], v=g(v)[0, 0, :NKV], r=g(r)[0, 0, :NH],
                                attn=g(o)[0, 0, 0])
        if self.dbg is None and self.ogate is not None:  # o * sigmoid(gate) -> Lo's sharded input
            o = self.ogate(o, yq, NH * HD // 32, self.lo_in)
        else:
            o = ttnn.multiply(o, ttnn.slice(yq, (0, 0, 0, NH * HD), (1, 1, 1, 2 * NH * HD), memory_config=L1),
                              input_tensor_b_activations=[UWP(U.SIGMOID)], memory_config=L1)
        return d["Lo"](o, out_mem=om)

    def moe(self, d, x, combine=True):
        """Routed experts of row 0. combine=False: returns (down projections [1, K, 32, H], weights [1, K, 1, 1]) for
        the post-MLP conv kernel, which does the weighted sum."""
        c = self.cfg
        if self.router is not None:  # kernels/ink_router_*: logits + routing in one program
            ids, wcol = self.router(x, d["Wg"], d["gbias"])
        else:
            logits = self.rowdot(d["WgT"], x)  # [1,1,1,E] float32
            ids, wcol = self.route(logits, d["gbias"])  # uint16 [1,1,1,K] expert ids, bf16 [1,K,1,1] weights
        if "Mgu" in d:  # kernels/ink_moe_*: experts, act and combine in one program (row 0, combined)
            return d["moek"](x, ids, wcol, d["Mgu"], d["Mdn"])
        gu = ttnn.sparse_matmul(x, d["Egu"], sparsity=self.SPARSITY, indices=ids, is_input_a_sparse=False,
                                is_input_b_sparse=True, memory_config=L1, program_config=self.pc_gu,
                                compute_kernel_config=d["eck"])
        I = c.inter
        if self.dbg is None and self.fuse_moe:
            act = self.act(gu)  # kernels/ink_act_*: slices + softcapped SwiGLU, row 0
        else:
            gu = ttnn.reshape(gu, (1, c.top_k, 32, 2 * I))
            g = ttnn.slice(gu, (0, 0, 0, 0), (1, c.top_k, 32, I), memory_config=L1)
            u = ttnn.slice(gu, (0, 0, 0, I), (1, c.top_k, 32, 2 * I), memory_config=L1)
            act = ttnn.multiply(g, g, input_tensor_a_activations=[UWP(U.MUL_UNARY_SFPU, 1.0 / c.gate_softcap),
                                                                  UWP(U.TANH)],
                                input_tensor_b_activations=[UWP(U.SIGMOID)], memory_config=L1)
            act = ttnn.multiply(act, u, input_tensor_b_activations=[UWP(U.MUL_UNARY_SFPU, 1.0 / c.up_softcap),
                                                                    UWP(U.TANH)], memory_config=L1)
        dn = ttnn.sparse_matmul(act, d["Edn"], sparsity=self.SPARSITY, indices=ids, is_input_a_sparse=True,
                                is_input_b_sparse=True, memory_config=L1, program_config=self.pc_dn,
                                compute_kernel_config=d["eck"])
        dn = ttnn.reshape(dn, (1, c.top_k, 32, c.hidden))
        if not combine:
            return dn, wcol
        return ttnn.sum(ttnn.multiply(dn, wcol, memory_config=L1), dim=1, keepdim=True, memory_config=L1)

    def layer(self, a, h, NB, nb, sc, msgs=None):
        """One layer application. Returns (h_out, NB', nb')."""
        c = self.cfg
        _, i, _ = self.apps[a]
        d, st = self.layers[i], self.state[a]
        fused = self.dbg is None
        rows = (0, 0, 0)  # source rows of the q / k / v linear inputs
        if fused and i > 0:
            # one kernel pair: stack, attention residual(s), and the input norms when there are no messages
            tb = self.artout.get(a)
            ais, xns, N = self.ares(h, NB, nb, d["arQ"], d["nqa"], None if msgs is not None else d["arLN"], tb,
                                    sharded=msgs is None and d["nqa"] == 1)
            if msgs is None:
                q_in, k_in, v_in = (xns * 3)[:3] if d["sliding"] else xns
            elif d["sliding"]:  # the three gate outputs in rows 0 / 1 / 2 of one tile: one rms norm, one linear
                ai = ais[0]
                for r, kd in enumerate(("attn", "k", "v")):
                    self.gate(ai, msgs[r], d[f"g_{kd}_d"], d[f"g_{kd}_fg"], kd, out=self.qkv3, orow=r)
                q_in = k_in = v_in = ttnn.rms_norm(self.qkv3, weight=d["ln"], epsilon=c.eps, memory_config=L1)
                rows = (0, 1, 2)
            else:
                q_in, k_in, v_in = (ttnn.rms_norm(self.msg_gate(d, kd, ais[r], msgs[r]), weight=d[ln], epsilon=c.eps,
                                                  memory_config=L1)
                                    for r, (kd, ln) in enumerate((("attn", "lnq"), ("k", "lnk"), ("v", "lnv"))))
        else:
            nh = self.rmsu(h)
            N = self.stack(NB, nb, nh)  # blocks + [h]
            if d["sliding"]:
                ai = self.attn_res1(d["qa_row"], N, nb + 1) if i > 0 else h
                if msgs is None:
                    q_in = k_in = v_in = ttnn.rms_norm(ai, weight=d["ln"], epsilon=c.eps, memory_config=L1)
                else:
                    na = self.rmsu(ai)
                    q_in = ttnn.rms_norm(self.msg_gate(d, "attn", ai, msgs[0], na), weight=d["ln"], epsilon=c.eps,
                                         memory_config=L1)
                    k_in = ttnn.rms_norm(self.msg_gate(d, "k", ai, msgs[1], na), weight=d["ln"], epsilon=c.eps,
                                         memory_config=L1)
                    v_in = ttnn.rms_norm(self.msg_gate(d, "v", ai, msgs[2], na), weight=d["ln"], epsilon=c.eps,
                                         memory_config=L1)
            else:
                if i > 0:
                    O = self.attn_res(d["Qa"], N, nb + 1)
                    qa, ka, va = (self.mm(self.SEL[r], O) for r in range(3))
                else:
                    qa = ka = va = h
                if msgs is not None:
                    qa = self.msg_gate(d, "attn", qa, msgs[0])
                    ka = self.msg_gate(d, "k", ka, msgs[1])
                    va = self.msg_gate(d, "v", va, msgs[2])
                q_in = ttnn.rms_norm(qa, weight=d["lnq"], epsilon=c.eps, memory_config=L1)
                k_in = ttnn.rms_norm(ka, weight=d["lnk"], epsilon=c.eps, memory_config=L1)
                v_in = ttnn.rms_norm(va, weight=d["lnv"], epsilon=c.eps, memory_config=L1)
        o = self.attention(d, st, q_in, k_in, v_in, sc, rows)
        fused = self.dbg is None
        if fused:
            rb, ob = self.cvbuf[(2 * a) % 3], self.cvbuf[(2 * a + 1) % 3]
            if a == c.loop_start - 1:
                ob = self.cvpre
            res = self.cvk(o, st["ha"], d["ta"], None if c.boundary(i) else h, rb)
            if c.boundary(i):
                NB, nb = N, nb + 1
        else:
            o = self.conv(o, st["ha"], d["ta"])
            self.dbg_cur["attn_out"] = ttnn.to_torch(o).float()[0, 0, 0]
            if c.boundary(i):
                NB, nb = N, nb + 1
                res = o
            else:
                res = ttnn.add(h, o, memory_config=L1)
        if fused:
            ais, xns, _ = self.ares(res, NB, nb, d["arQm"], 1, None if msgs is not None else d["arLNm"])
            if msgs is None:
                x_m = xns[0]
            else:
                x_m = ttnn.rms_norm(self.msg_gate(d, "mlp", ais[0], msgs[3]), weight=d["lnm"], epsilon=c.eps,
                                    memory_config=L1)
        else:
            m_in = self.attn_res1(d["qm_row"], self.stack(NB, nb, self.rmsu(res)), nb + 1)
            if msgs is not None:
                m_in = self.msg_gate(d, "mlp", m_in, msgs[3])
            x_m = ttnn.rms_norm(m_in, weight=d["lnm"], epsilon=c.eps, memory_config=L1)
        if fused and self.fuse_moe and "Mgu" not in d:
            dn, wcol = self.moe(d, x_m, combine=False)
            out = self.cvk(dn, st["hm"], d["tm"], res, ob, wcol)  # expert combine + conv + residual
        elif fused:
            out = self.cvk(self.moe(d, x_m), st["hm"], d["tm"], res, ob)
        else:
            m = self.moe(d, x_m)
            m = self.conv(m, st["hm"], d["tm"])
            out = ttnn.add(res, m, memory_config=L1)
        if self.dbg is not None:  # untraced debugging: row 0 of every application's output
            self.dbg_cur.update(mlp_out=ttnn.to_torch(m).float()[0, 0, 0], h_out=ttnn.to_torch(out).float()[0, 0, 0])
            self.dbg.append(self.dbg_cur)
            self.dbg_cur = {}
        return out, NB, nb

    def step_consts(self):
        """Per-step tensors derived on the device (none since the decode attention kernel builds its masks)."""
        return {}

    def forward(self, logits=True):
        c = self.cfg
        sc = self.step_consts()
        h = self.X0
        NB, nb = None, 0
        a = 0
        for _ in range(c.loop_start):
            h, NB, nb = self.layer(a, h, NB, nb, sc)
            a += 1
        pre = (h, NB, nb)
        for _ in range(c.n_loop):
            h, NB, nb = self.layer(a, h, NB, nb, sc)
            a += 1
        for _ in range(1, c.passes):
            # messages from the previous pass: its block state appended at loop_start + block and its output
            if self.dbg is None:  # attention-residual kernel, 4 queries: [block state, norm(h)] -> 4 messages
                NB0 = self.mm(self.MOV20, NB)
                ttnn.copy(h, self.hmsg)
            else:
                NM = ttnn.add(self.mm(self.MOV20, NB), self.mm(self.SH[1], self.rmsu(h)), memory_config=L1)
            h, NB, nb = pre
            for j in range(c.n_loop):
                if self.dbg is None:
                    self.ares(self.hmsg, NB0, 1, self.layers[c.loop_start + j]["arQmsg"], 4, outs=self.msgbuf)
                    msgs = self.msgbuf
                else:
                    M = self.attn_res(self.layers[c.loop_start + j]["Qmsg"], NM, 2)
                    msgs = [self.mm(self.SEL[r], M) for r in range(4)]
                h, NB, nb = self.layer(a, h, NB, nb, sc, msgs)
                a += 1
        for _ in range(c.loop_end, c.n_layers):
            h, NB, nb = self.layer(a, h, NB, nb, sc)
            a += 1
        if not logits:
            return None
        if self.dbg is None:
            f = self.ares(h, NB, nb, self.arQf, 1, self.arLNf, sharded=True)[1][0]
        else:
            f = self.attn_res1(self.qf_row, self.stack(NB, nb, self.rmsu(h)), nb + 1)
            f = ttnn.rms_norm(f, weight=self.lnf, epsilon=c.eps, memory_config=L1)
        return self.head(f)

    def head(self, f):
        """lm_head of row 0 -> per-chunk top-k (values, ids) [1, chunks, 1, TOPK] (see read_topk)."""
        outs = []
        xm = self.lm_chunks[0].xmem
        if f.memory_config() != xm:  # once for all chunks (they share the input layout)
            f = ttnn.to_memory_config(f, xm)
        for lin in self.lm_chunks:
            o = lin(f, out_mem=None)  # bf16 (a float32 top-k over the vocabulary costs 7 ms)
            outs.append(ttnn.slice(o, (0, 0, 0, 0), (1, 1, 1, self.cw), memory_config=L1))
            ttnn.deallocate(o)
        st = ttnn.concat(outs, dim=1, memory_config=L1)  # [1, chunks, 1, cw]
        vals, idx = ttnn.topk(st, TOPK, dim=-1, largest=True, sorted=True, memory_config=L1)
        return vals, idx

    # ------------------------------------------------------------------------------------------------ host side
    def write_inputs(self, token, pos):
        c = self.cfg
        e = self.embed[token:token + 1].float()
        e = e * torch.rsqrt(e.square().mean(-1, keepdim=True) + c.eps) * self.embed_norm
        x = torch.zeros(1, 1, 32, c.hidden)
        x[0, 0, 0] = e[0]
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.X0)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32),
                                        self.POS_I)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([pos % self.R], dtype=torch.int32),
                                                        dtype=ttnn.int32), self.RING_I)
        for t, v in ((self.D_RING, pos % self.R), (self.D_POSF, pos),
                     (self.D_NPR, self.att_ring.chunks(pos) * self.att_ring.C - 1),
                     (self.D_NPF, self.att_full.chunks(pos) * self.att_full.C - 1)):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([v] * 8, dtype=torch.int32),
                                                            dtype=ttnn.int32), t)
        for t, v in ((self.POSF, float(pos)), (self.CF, float(pos % self.R)), (self.CW, float(pos % c.window)),
                     (self.THR, float(min(pos, c.window - 1)))):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full((1, 1, 1, 1), v), dtype=ttnn.float32,
                                                            layout=ttnn.TILE_LAYOUT), t)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.full((1, 1, 1, 1), c.tau(pos)), dtype=ttnn.bfloat16,
                                                        layout=ttnn.TILE_LAYOUT), self.TAU)

    def capture(self):
        """Trace every step variant: decode with / without the lm_head, prefill with / without it. All variants
        run once (compile) before the first capture: an op's program-owned device buffers (created when it is
        first compiled) must not be allocated while a trace exists, or they can land in that trace's scratch
        memory and be overwritten when it replays (seen as a hang of the other trace)."""
        variants = [("decode", True, None), ("decode", False, None)]
        if self.has_prefill:  # one pair per context bucket of the prefill attention (None: the whole cache)
            # with the lm_head only (~2 ms of a 64-token step): a no-logits copy per bucket would double the trace memory
            variants += [("prefill", True, b) for b in [None] + self.p_buckets]

        def inputs(kind):
            self.write_inputs(0, 0) if kind == "decode" else self.p_write_inputs([0], 0)

        def run(kind, logits, b):
            if kind == "decode":
                return self.forward(logits)
            self.p_bucket = b
            return self.p_forward(logits)

        for kind, logits, b in variants:
            inputs(kind)
            run(kind, logits, b)
        ttnn.synchronize_device(self.dev)
        for kind, logits, b in variants:
            inputs(kind)
            ttnn.synchronize_device(self.dev)
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            out = run(kind, logits, b)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            ttnn.synchronize_device(self.dev)
            self.traces[logits if kind == "decode" else ("prefill", logits, b)] = (tid, out)
        self.p_bucket = None
        self.reset()

    def step(self, token, logits=True, traced=True):
        """Process `token` at position self.pos. Returns (top-64 soft-capped logits, ids) if logits."""
        if self.pos >= self.max_seq:
            raise RuntimeError(f"context full ({self.max_seq} tokens)")
        self.write_inputs(token, self.pos)
        if traced and logits in self.traces:
            tid, out = self.traces[logits]
            ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        else:
            out = self.forward(logits)
        self.pos += 1
        if not logits:
            return None
        return self.read_topk(out)

    def read_topk(self, out):
        C = len(self.lm_chunks)
        vv = ttnn.to_torch(out[0]).float().reshape(C, -1)[:, :TOPK]
        ii = ttnn.to_torch(out[1]).long().reshape(C, -1)[:, :TOPK] + torch.arange(C)[:, None] * self.cw
        vv, ii = vv.reshape(-1), ii.reshape(-1)
        keep = ii < self.cfg.vocab_unpadded
        vv, ii = vv[keep], ii[keep]
        v, j = torch.topk(vv, min(TOPK, vv.numel()))
        cap = self.cfg.logit_softcap
        return torch.tanh(v / cap) * cap, ii[j]
