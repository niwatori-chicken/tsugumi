"""Lean B=1 decode engine for Qwen3.5/3.8 dense hybrids (GDN + gated attention) on one P100A."""
import json
import math
import os
import time

import torch
import ttnn
from safetensors import safe_open

from .attn import AttnDecode
from .env import env
from .gdn import GDNDecode, GDNWeights
from .ops import DecodeLinear, has_cache

_L1 = ttnn.L1_MEMORY_CONFIG
TOPK = 64  # main-model logits kept per row for sampling (top_k is capped at this)
LM_PREFIX = "model.language_model."


class LazySD:
    """Read tensors on demand from a sharded safetensors checkpoint."""

    def __init__(self, path):
        self.path = path
        self.map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._open = {}

    def __getitem__(self, k):
        f = self.map[k]
        if f not in self._open:
            self._open[f] = safe_open(os.path.join(self.path, f), "pt")
        return self._open[f].get_tensor(k)

    def __contains__(self, k):
        return k in self.map


class Precision:
    """Weight dtypes per tensor group. Names: mlp_gu, mlp_down, gdn_in, gdn_out, attn_qkvg, attn_o, lm_head."""

    PRESETS = {
        # qwen36 demo default (bfp4 gate/up, bfp8 everything else)
        "mixed": dict(mlp_gu="bfp4", mlp_down="bfp8", gdn_in="bfp8", gdn_out="bfp8", attn_qkvg="bfp8",
                      attn_o="bfp8", lm_head="bfp8"),
        "bfp4": dict(mlp_gu="bfp4", mlp_down="bfp4", gdn_in="bfp4", gdn_out="bfp4", attn_qkvg="bfp4",
                     attn_o="bfp4", lm_head="bfp8"),
    }
    DT = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}

    def __init__(self, preset="mixed", **over):
        self.d = dict(self.PRESETS[preset])
        self.d.update(over)

    def __getitem__(self, k):
        return self.DT[self.d[k]]


class MLPDecode:
    def __init__(self, dev, sd, prefix, H, I, prec, cache_prefix):
        cp = lambda s: f"{cache_prefix}.{s}"
        gu = prec["mlp_gu"]
        dn = prec["mlp_down"]
        w1 = None if has_cache(cp("gate"), gu) else sd[prefix + "gate_proj.weight"].T.contiguous().to(torch.bfloat16)
        self.w1 = DecodeLinear(dev, w1, gu, cache_file=cp("gate"), shape=(H, I), name="mlp_gate")
        del w1
        w3 = None if has_cache(cp("up"), gu) else sd[prefix + "up_proj.weight"].T.contiguous().to(torch.bfloat16)
        self.w3 = DecodeLinear(dev, w3, gu, cache_file=cp("up"), shape=(H, I), name="mlp_up")
        del w3
        w2 = None if has_cache(cp("down"), dn) else sd[prefix + "down_proj.weight"].T.contiguous().to(torch.bfloat16)
        self.w2 = DecodeLinear(dev, w2, dn, cache_file=cp("down"), shape=(I, H), name="mlp_down")

    def __call__(self, x):
        g = self.w1(x, out_mem=None)  # width-sharded L1
        u = self.w3(x, out_mem=None)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                          memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        if h.shape[-1] != self.w2.K:
            h2 = ttnn.slice(ttnn.to_memory_config(h, _L1), (0, 0, 0, 0), (1, 1, 32, self.w2.K), memory_config=_L1)
            ttnn.deallocate(h)
            h = h2
        return self.w2(h, out_mem=_L1)


class SpecShared:
    """Per-step device inputs shared by all layers in T-row mode (fixed addresses for trace)."""

    def __init__(self, dev, T, HD, max_seq, pager, block_size=64):
        self.T = T
        bf = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        self.m_hist = bf(torch.zeros(1, 1, 32, 64))
        self.m_shift = [bf(self.shift_matrix(T, j)) for j in range(4)]
        from .gdn_kernel import GDNFusedKernel

        self.idx = GDNFusedKernel.make_idx(dev, 0)
        self.cur_pos = ttnn.from_torch(torch.zeros(T, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        nb = max_seq // block_size
        self.page_table = pager.table_T(T)  # shared with every mode; rewritten by the pager
        self.page_table_1 = pager.pt1
        self.cur_pos_rows = [ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=dev)
                             for _ in range(T)]
        self.gqa = None
        if env("GQA_ATTN", "1") == "1":
            from .gqa_attn import GQAAttention

            from .attn import kv_dtype

            self.gqa = GQAAttention(dev, T, HD=HD, block_size=block_size, nblocks=nb, kv_dtype=kv_dtype())
        self.cos = bf(torch.ones(1, T, 1, HD))
        self.sin = bf(torch.zeros(1, T, 1, HD))
        self.sel_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                       fp32_dest_acc_en=True, packer_l1_acc=False)

    @staticmethod
    def _row(k):  # index k of s = [hist0, hist1, hist2, block0, ...] -> row of the [hist(32 rows); block(32 rows)] concat
        return k if k < 3 else 32 + k - 3

    @classmethod
    def shift_matrix(cls, T, j):
        M = torch.zeros(1, 1, 32, 64)
        for t in range(T):
            M[0, 0, t, cls._row(t + j)] = 1.0
        return M

    @classmethod
    def hist_matrix(cls, a):
        M = torch.zeros(1, 1, 32, 64)
        for r in range(3):
            M[0, 0, r, cls._row(a + r)] = 1.0
        return M


class MTPHead:
    """Qwen3.5 MTP (NextN) layer: fc(cat(norm(emb), norm(hidden))) -> full-attention decoder layer -> norm.
    Shares embedding and lm_head with the main model. Runs in T-row mode with its own paged KV."""

    def __init__(self, eng, prec, cache_dir, draft_lm_bfp8=False):
        dev, sd, tc = eng.dev, eng.sd, eng.cfg
        self.eng = eng
        p = "mtp."
        nw = eng._norm_w_fn
        self.n_emb = nw(p + "pre_fc_norm_embedding.weight")
        self.n_hid = nw(p + "pre_fc_norm_hidden.weight")
        self.in_norm = nw(p + "layers.0.input_layernorm.weight")
        self.post_norm = nw(p + "layers.0.post_attention_layernorm.weight")
        self.final = nw(p + "norm.weight")
        c = os.path.join(cache_dir, "mtp")
        H = eng.H
        dt = prec["mlp_gu"]
        wfc = None if has_cache(c + ".fc", dt) else sd[p + "fc.weight"].T.contiguous().to(torch.bfloat16)
        self.fc = DecodeLinear(dev, wfc, dt, cache_file=c + ".fc", shape=(2 * H, H), name="mtp_fc")
        self.attn = AttnDecode(dev, sd, p + "layers.0.self_attn.", tc, eng.max_seq, wdtype=prec["attn_qkvg"],
                               cache_prefix=c + ".attn")
        self.mlp = MLPDecode(dev, sd, p + "layers.0.mlp.", H, eng.I, prec, c + ".mlp")
        self.x_emb = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev, memory_config=eng.resid_mem)
        self.m_sel = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev)
        self.trace = None
        self.out = None
        self.traces = {}
        # draft lm_head: bfp4 copy of the shared lm_head (half the bytes of the bfp8 verify head), by default only the
        # rows of the reduced draft vocabulary (tsumi/draft_vocab.pt, 63,168 frequent tokens: 1/4 of the bytes; the
        # draft index is mapped back on the host). The main model verifies over the full vocabulary either way.
        self.lm_chunks = []
        self.vocab_map = None
        vf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_vocab.pt")
        if not draft_lm_bfp8 and env("DRAFT_VOCAB", "1") == "1" and os.path.exists(vf):
            ids = torch.load(vf)
            self.vocab_map = ids.tolist()
            n = len(self.vocab_map)
            tag = f"{n}.{int(ids.sum()) % 1000003}"
            lm = None
            for lo in range(0, n, 31584):  # same chunk width as the full lm_head
                hi = min(lo + 31584, n)
                cf = os.path.join(cache_dir, f"lm_head_draft_sub.{tag}.{lo}")
                w = None
                if not has_cache(cf, ttnn.bfloat4_b):
                    if lm is None:
                        lm = sd["lm_head.weight"]
                    w = lm[ids[lo:hi].long()].T.contiguous().to(torch.bfloat16)
                self.lm_chunks.append((lo, hi, DecodeLinear(dev, w, ttnn.bfloat4_b, cache_file=cf, shape=(H, hi - lo),
                                                            name="lm_draft_sub")))
                del w
            del lm
        lm = None
        for lo, hi, lin in ([] if draft_lm_bfp8 or self.vocab_map else eng.lm_chunks):
            cf = os.path.join(cache_dir, f"lm_head_draft.{lo}")
            w = None
            if not has_cache(cf, ttnn.bfloat4_b):
                if lm is None:
                    lm = sd["lm_head.weight"]
                w = lm[lo:hi].T.contiguous().to(torch.bfloat16)
            self.lm_chunks.append((lo, hi, DecodeLinear(dev, w, ttnn.bfloat4_b, cache_file=cf, shape=(H, hi - lo),
                                                        name="lm_draft")))
        del lm
        if draft_lm_bfp8:
            self.lm_chunks = eng.lm_chunks

    def init_spec(self, T):
        self.attn.init_spec(T)

    def forward(self):
        e = self.eng
        hid = ttnn.matmul(self.m_sel, e.hid_src, memory_config=_L1, compute_kernel_config=e.sh.sel_ck)
        hid_s = ttnn.to_memory_config(hid, e.resid_mem)
        ttnn.deallocate(hid)
        a = e._norm(self.x_emb, self.n_emb)
        b = e._norm(hid_s, self.n_hid)
        ttnn.deallocate(hid_s)
        cat = ttnn.concat([ttnn.to_memory_config(a, _L1), ttnn.to_memory_config(b, _L1)], dim=-1, memory_config=_L1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        x = e._to_resid(self.fc(cat, out_mem=_L1))
        ttnn.deallocate(cat)
        h = e._norm(x, self.in_norm)
        att = e._to_resid(self.attn.forward_spec(h, e.sh))
        ttnn.deallocate(h)
        x2 = ttnn.add(x, att, memory_config=e.resid_mem)
        ttnn.deallocate(x)
        ttnn.deallocate(att)
        h = e._norm(x2, self.post_norm)
        mm = e._to_resid(self.mlp(h))
        ttnn.deallocate(h)
        x3 = ttnn.add(x2, mm, memory_config=e.resid_mem)
        ttnn.deallocate(x2)
        ttnn.deallocate(mm)
        h = e._norm(x3, self.final)
        ttnn.deallocate(x3)
        hidden = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(hidden, e.hid_src)
        ttnn.deallocate(hidden)
        logits, am = e._logits_argmax(e._lm_head_rows(h, self.lm_chunks))
        ttnn.deallocate(h)
        return logits, am

    def capture(self, compile_first=True):
        if compile_first:
            self.forward()
        tid = ttnn.begin_trace_capture(self.eng.dev, cq_id=0)
        out = self.forward()
        ttnn.end_trace_capture(self.eng.dev, tid, cq_id=0)
        self.traces[self.eng.T] = (tid, out)
        self.trace, self.out = tid, out

    def run(self, tokens, pos, sel_rows):
        """tokens: embeddings for rows (positions pos..); sel_rows[i] = hid_src row feeding MTP row i."""
        e = self.eng
        x = torch.zeros(1, 1, 32, e.H, dtype=torch.bfloat16)
        for i, t in enumerate(tokens):
            x[0, 0, i] = e.embed[t]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=e.resid_mem), self.x_emb)
        M = torch.zeros(1, 1, 32, 32)
        for i, r in enumerate(sel_rows):
            M[0, 0, i, r] = 1.0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.m_sel)
        e._set_positions(pos)
        tr = self.traces.get(e.T)
        if tr is not None:
            ttnn.execute_trace(e.dev, tr[0], cq_id=0, blocking=False)
            out = tr[1]
        else:
            out = self.forward()
        am = ttnn.to_torch(out[1]).reshape(-1)[: e.T]
        if self.vocab_map is not None:
            return [self.vocab_map[int(v)] for v in am]
        return [int(v) for v in am]


class Qwen35Engine:
    def __init__(self, dev, model_dir, prec: Precision, cache_dir, max_seq=4096, n_layers=None, log=print):
        self.dev = dev
        self.sampler = None  # set by Generator.generate for sampled decoding; None = greedy
        self.last_topk = None
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        tc = cfg.get("text_config", cfg)
        self.cfg = tc
        self.H = tc["hidden_size"]
        self.I = tc["intermediate_size"]
        self.V = tc["vocab_size"]
        self.HD = tc["head_dim"]
        self.rope_dim = int(self.HD * tc.get("partial_rotary_factor", 0.25))
        rp = tc.get("rope_parameters", {})
        self.theta = rp.get("rope_theta", tc.get("rope_theta", 1e7))
        self.layer_types = tc["layer_types"][: n_layers or tc["num_hidden_layers"]]
        # page tables are int32 rows of max_seq/64 entries and must be a multiple of 32 bytes: round to 512 tokens
        max_seq = -(-max_seq // 512) * 512
        self.max_seq = max_seq
        self.eps = tc["rms_norm_eps"]
        sd = LazySD(model_dir)
        self.sd = sd
        os.makedirs(cache_dir, exist_ok=True)
        tag = lambda k: prec.d[k]

        def norm_w(key):
            w = (1.0 + sd[key].float()).reshape(1, 1, -1, 32)
            return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=dev)

        self._norm_w_fn = norm_w
        self.prec = prec
        self.cache_dir = cache_dir

        # residual stream: width-sharded over the K=hidden in0 grid of the decode matmuls (40 cores for 5120)
        probe = DecodeLinear(dev, None, ttnn.bfloat8_b, shape=(self.H, 32 * 7 * 3), name="probe", no_weight=True)
        self.resid_mem = probe.xmem
        rc = probe.in0_cores
        grid = self.resid_mem.shard_spec.grid.bounding_box()
        gx, gy = grid.end.x - grid.start.x + 1, grid.end.y - grid.start.y + 1
        bw = self.H // rc // 32
        self.norm_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy], subblock_w=bw, block_h=1, block_w=bw, inplace=False)

        t0 = time.time()
        self.layers = []
        for i, lt in enumerate(self.layer_types):
            p = f"{LM_PREFIX}layers.{i}."
            c = os.path.join(cache_dir, f"L{i}")
            L = dict(type=lt, in_norm=norm_w(p + "input_layernorm.weight"),
                     post_norm=norm_w(p + "post_attention_layernorm.weight"))
            if lt == "linear_attention":
                L["mix"] = GDNDecode(dev, GDNWeights(sd, p + "linear_attn."), tc, wdtype=prec["gdn_in"],
                                     out_dtype=prec["gdn_out"], cache_prefix=c + ".gdn")
            else:
                L["mix"] = AttnDecode(dev, sd, p + "self_attn.", tc, max_seq, wdtype=prec["attn_qkvg"],
                                      cache_prefix=c + ".attn")
                if prec["attn_o"] != prec["attn_qkvg"]:
                    raise NotImplementedError("separate attn_o dtype")
            L["mlp"] = MLPDecode(dev, sd, p + "mlp.", self.H, self.I, prec, c + ".mlp")
            self.layers.append(L)
            log(f"layer {i} ({lt}) loaded, {time.time() - t0:.0f}s")
        self.final_norm = norm_w(LM_PREFIX + "norm.weight")
        # lm_head split in chunks (DRAM-sharded matmul needs a bounded per-core N)
        self.lm_chunks = []
        n_chunks = 8
        cs = math.ceil(self.V / n_chunks / 32) * 32
        lm = None
        for j in range(n_chunks):
            lo, hi = j * cs, min(self.V, (j + 1) * cs)
            cf = os.path.join(cache_dir, f"lm_head.{j}")
            if not has_cache(cf, prec["lm_head"]):
                if lm is None:
                    lm = sd["lm_head.weight"]
                w = lm[lo:hi].T.contiguous().to(torch.bfloat16)
            else:
                w = None
            self.lm_chunks.append((lo, hi, DecodeLinear(dev, w, prec["lm_head"], cache_file=cf, shape=(self.H, hi - lo),
                                                        name="lm_head")))
        del lm
        self.embed = sd[LM_PREFIX + "embed_tokens.weight"].to(torch.bfloat16)  # host-side lookup
        log(f"model loaded in {time.time() - t0:.0f}s")

        # per-step device inputs (fixed addresses for trace)
        self.x_in = ttnn.from_torch(torch.zeros(1, 1, 32, self.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=dev, memory_config=self.resid_mem)
        self.cur_pos = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        self.cos = ttnn.from_torch(torch.ones(1, 1, 1, self.HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev)
        self.sin = ttnn.from_torch(torch.zeros(1, 1, 1, self.HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev)
        inv = 1.0 / (self.theta ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float64) / self.rope_dim))
        self.inv_freq = inv
        self.trace_id = None
        self.logits_out = None
        self.fused_gdn = env('FUSED_GDN', '1') == '1'
        # chunked SDPA prefill (needs aligned chunk starts); superseded by the GQA kernel (TSUMI_GQA_ATTN=1)
        self._chunked_prefill = env('CHUNKED_PREFILL', '1') == '1' and \
            env('GQA_ATTN', '1') != '1'
        self._kv_rowwise = False

    # ---- host-side step inputs ----
    def _rope_rows(self, pos):
        f = pos * self.inv_freq
        emb = torch.cat([f, f])
        cos = torch.ones(self.HD, dtype=torch.float64)
        sin = torch.zeros(self.HD, dtype=torch.float64)
        cos[: self.rope_dim] = emb.cos()
        sin[: self.rope_dim] = emb.sin()
        return cos.float().view(1, 1, 1, -1), sin.float().view(1, 1, 1, -1)

    def set_inputs(self, token, pos):
        x = torch.zeros(1, 1, 32, self.H, dtype=torch.bfloat16)
        x[0, 0, 0] = self.embed[token]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=self.resid_mem), self.x_in)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32),
                                        self.cur_pos)
        c, s = self._rope_rows(pos)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.cos)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.sin)

    # ---- device graph ----
    def _norm(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.eps, program_config=self.norm_pc, memory_config=self.resid_mem)

    def _to_resid(self, y):
        """[1,1,32,Np] (any L1 layout) -> [1,1,32,H] width-sharded like the residual."""
        if y.shape[-1] == self.H and y.memory_config() == self.resid_mem:
            return y
        if y.memory_config().is_sharded():
            yi = ttnn.to_memory_config(y, _L1)
            ttnn.deallocate(y)
            y = yi
        r = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 32, self.H), memory_config=self.resid_mem)
        ttnn.deallocate(y)
        return r

    def forward(self):
        x = self.x_in
        for L in self.layers:
            h = self._norm(x, L["in_norm"])
            if L["type"] == "linear_attention":
                a = L["mix"].forward_decode_fused(h) if self.fused_gdn else L["mix"].forward_decode(h)
            else:
                a = L["mix"].forward_decode(h, self.cur_pos, self.cos, self.sin)
            ttnn.deallocate(h)
            a = self._to_resid(a)
            x2 = ttnn.add(x, a, memory_config=self.resid_mem)
            ttnn.deallocate(a)
            if x is not self.x_in:
                ttnn.deallocate(x)
            x = x2
            h = self._norm(x, L["post_norm"])
            m = self._to_resid(L["mlp"](h))
            ttnn.deallocate(h)
            x2 = ttnn.add(x, m, memory_config=self.resid_mem)
            ttnn.deallocate(m)
            ttnn.deallocate(x)
            x = x2
        h = self._norm(x, self.final_norm)
        ttnn.deallocate(x)
        outs = []
        for lo, hi, lin in self.lm_chunks:
            o = lin(h, out_mem=_L1)
            o1 = ttnn.slice(o, (0, 0, 0, 0), (1, 1, 1, hi - lo), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            outs.append(o1)
        ttnn.deallocate(h)
        logits = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,1,V]
        for o in outs:
            ttnn.deallocate(o)
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rows = getattr(self, "T", 1)
        if rows < 32:  # argmax cost is per row (~50us each); only the first T rows are real
            r2 = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, rows, rm.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(rm)
            rm = r2
        am = ttnn.argmax(rm, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rm)
        return logits, am

    def logits_to_host(self, outs):
        return ttnn.to_torch(outs[0]).reshape(-1)[: self.V].float()

    def argmax_to_host(self, outs):
        return int(ttnn.to_torch(outs[1]).reshape(-1)[0])

    # ---------------- T-row mode (chunked prefill / speculative verify) ----------------
    def init_spec(self, T, mtp=False):
        self.T_dec = T
        from .kvpager import KVPager

        self.pager = KVPager(self.dev, self.max_seq // 64)
        self.sh_by = {T: SpecShared(self.dev, T, self.HD, self.max_seq, self.pager)}
        self.spec_traces, self.spec_outs = {}, {}
        self._last_snap_all = True
        self.use_T(T)
        for L in self.layers:
            L["mix"].init_spec(T)
        self.hid_src = ttnn.from_torch(torch.zeros(1, 1, 32, self.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=self.dev)
        self.T_pre = None
        self.mtp = None
        if mtp:
            mp = env("MTP_PREC", "bfp4")
            if mp == "bfp8":
                prec = Precision("bfp4", mlp_gu="bfp8", mlp_down="bfp8", attn_qkvg="bfp8", attn_o="bfp8")
                cache = self.cache_dir.rsplit("/", 1)[0] + "/mtp_bfp8"
            else:
                prec, cache = self.prec, self.cache_dir
            self.mtp = MTPHead(self, prec, cache, draft_lm_bfp8=(mp == "bfp8"))
            self.mtp.init_spec(T)
        self.h_last = None
        self.use_T(T)

    def init_prefill(self, TP=32):
        """Add a TP-row mode for prompt prefill (GDN kernel writes only the final state)."""
        self.T_pre = TP
        self.sh_by[TP] = SpecShared(self.dev, TP, self.HD, self.max_seq, self.pager)
        for L in self.layers:
            if L["type"] == "linear_attention":
                L["mix"].init_spec(TP, snap_all=False)
            else:
                L["mix"].init_spec(TP, chunked=self._chunked_prefill)
        if self.mtp is not None:
            self.mtp.init_spec(TP)  # MTP rows start at pos+1 (unaligned): keeps the per-row decode SDPA

    def use_T(self, T):
        self.T = T
        self.sh = self.sh_by[T]
        self.spec_trace = self.spec_traces.get(T)
        self.spec_out = self.spec_outs.get(T)

    def reset_spec(self):
        for L in self.layers:
            if L["type"] == "linear_attention":
                L["mix"].reset_spec()
        self._a_prev = 0
        self._last_snap_all = True

    def set_inputs_spec(self, tokens, pos, a_prev):
        """tokens: list (<= T) for positions pos..; a_prev: rows committed by the previous step (0 after reset)."""
        T, sh = self.T, self.sh
        x = torch.zeros(1, 1, 32, self.H, dtype=torch.bfloat16)
        for i, t in enumerate(tokens):
            x[0, 0, i] = self.embed[t]
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=self.resid_mem), self.x_in)
        self._set_positions(pos)
        h = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay)
        ttnn.copy_host_to_device_tensor(h(SpecShared.hist_matrix(a_prev)), sh.m_hist)
        from .gdn_kernel import GDNFusedKernel

        # slot holding the state after the last committed row of the previous step
        GDNFusedKernel.set_idx(sh.idx, max(a_prev - 1, 0) if self._last_snap_all else 0)

    def _set_positions(self, pos):
        T, sh = self.T, self.sh
        self.pager.ensure(pos + T)  # rows pos..pos+T-1 write KV
        h = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay)
        ttnn.copy_host_to_device_tensor(h(torch.arange(pos, pos + T, dtype=torch.int32), ttnn.int32,
                                          ttnn.ROW_MAJOR_LAYOUT), sh.cur_pos)
        # row 0's position (chunk start for chunked SDPA); the other rows are only used by kv_rowwise
        for i in range(T if self._kv_rowwise else 1):
            ttnn.copy_host_to_device_tensor(h(torch.tensor([pos + i], dtype=torch.int32), ttnn.int32,
                                              ttnn.ROW_MAJOR_LAYOUT), sh.cur_pos_rows[i])
        cs = [self._rope_rows(pos + i) for i in range(T)]
        ttnn.copy_host_to_device_tensor(h(torch.cat([c for c, _ in cs], dim=1)), sh.cos)
        ttnn.copy_host_to_device_tensor(h(torch.cat([s_ for _, s_ in cs], dim=1)), sh.sin)

    def forward_spec(self):
        x = self.x_in
        for L in self.layers:
            h = self._norm(x, L["in_norm"])
            a = L["mix"].forward_spec(h, self.sh)
            ttnn.deallocate(h)
            a = self._to_resid(a)
            x2 = ttnn.add(x, a, memory_config=self.resid_mem)
            ttnn.deallocate(a)
            if x is not self.x_in:
                ttnn.deallocate(x)
            x = x2
            h = self._norm(x, L["post_norm"])
            m = self._to_resid(L["mlp"](h))
            ttnn.deallocate(h)
            x2 = ttnn.add(x, m, memory_config=self.resid_mem)
            ttnn.deallocate(m)
            ttnn.deallocate(x)
            x = x2
        h = self._norm(x, self.final_norm)
        ttnn.deallocate(x)
        hidden = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)  # final normed hidden (MTP input)
        if getattr(self, "hid_src", None) is not None:
            ttnn.copy(hidden, self.hid_src)
        outs = []
        for lo, hi, lin in self.lm_chunks:
            o = lin(h, out_mem=_L1)
            o1 = ttnn.slice(o, (0, 0, 0, 0), (1, 1, 32, hi - lo), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            outs.append(o1)
        ttnn.deallocate(h)
        logits, tk = self._logits_topk(outs)
        return logits, tk, hidden

    def _lm_head_rows(self, h, chunks=None):
        outs = []
        for lo, hi, lin in (chunks or self.lm_chunks):
            o = lin(h, out_mem=_L1)
            o1 = ttnn.slice(o, (0, 0, 0, 0), (1, 1, 32, hi - lo), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            outs.append(o1)
        return outs

    def _logits_topk(self, outs):
        """Main-model rows -> (logits, (values, indices)) of the TOPK largest logits per row (all 32 rows; ~1.2 ms,
        +0.8 ms over the argmax path). Greedy takes index 0; sampling works on the top-k on the host."""
        logits = self._concat_chunks(outs)
        v, i = ttnn.topk(logits, TOPK, dim=-1, largest=True, sorted=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, (v, i)

    def topk_to_host(self, tk, rows):
        v, i = tk
        return (ttnn.to_torch(v).reshape(-1, TOPK)[:rows].float(), ttnn.to_torch(i).reshape(-1, TOPK)[:rows].long())

    @staticmethod
    def _greedy(v, i):
        """argmax with ttnn.argmax's tie-break (lowest token id among the equal bf16 maxima); topk orders ties
        differently, and ties are common in bf16."""
        return int(i[v == v[0]].min())

    def _pick_row(self, r):
        """Token for row r of the last top-k: greedy or the active sampler."""
        v, i = self.last_topk
        return self._greedy(v[r], i[r]) if self.sampler is None else self.sampler(v[r], i[r])

    @staticmethod
    def _concat_chunks(outs):
        if len(outs) == 1:  # concat of one tensor returns it: do not free it
            return outs[0]
        logits = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,32,V]
        for o in outs:
            ttnn.deallocate(o)
        return logits

    def _logits_argmax(self, outs):
        logits = self._concat_chunks(outs)
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rows = getattr(self, "T", 1)
        if rows < 32:  # argmax cost is per row (~50us each); only the first T rows are real
            r2 = ttnn.slice(rm, (0, 0, 0, 0), (1, 1, rows, rm.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(rm)
            rm = r2
        am = ttnn.argmax(rm, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(rm)
        return logits, am

    def capture_spec(self, compile_first=True):
        if compile_first:
            self.forward_spec()
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        out = self.forward_spec()
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        self.spec_traces[self.T], self.spec_outs[self.T] = tid, out
        self.use_T(self.T)

    def capture_all(self):
        """Compile every mode eagerly first, then capture all traces back to back (no eager allocations
        between captures: buffers allocated after a capture can alias that trace's buffers)."""
        modes = sorted(self.sh_by.keys(), reverse=True)
        for T in modes:
            self.use_T(T)
            self.set_inputs_spec([0], 0, 0)
            self.forward_spec()
            if self.mtp is not None:
                self.mtp.run([0], 0, [0])
        for T in modes:
            self.use_T(T)
            self.capture_spec(compile_first=False)
            if self.mtp is not None:
                self.mtp.capture(compile_first=False)
        self.use_T(self.T_dec)
        self.reset_spec()

    def run_spec(self, tokens, pos, a_prev):
        """Process up to T tokens at positions pos.. ; returns argmax per row (list of T ints)."""
        self.set_inputs_spec(tokens, pos, a_prev)
        if self.spec_trace is not None:
            ttnn.execute_trace(self.dev, self.spec_trace, cq_id=0, blocking=False)
            out = self.spec_out
        else:
            out = self.forward_spec()
        self.last_spec = out
        self._last_snap_all = (self.T != self.T_pre)
        self.last_topk = self.topk_to_host(out[1], self.T)
        v, i = self.last_topk
        return [self._greedy(v[r], i[r]) for r in range(self.T)]

    def prefill_spec(self, ids, start=0, with_mtp=False):
        """Chunked prefill of ids at positions start.. : full T_pre-row chunks first, the rest in T_dec rows.
        With MTP, each chunk also runs the MTP head over the chunk (fills its KV). Returns
        (argmax after the last prompt token, next position, MTP draft for the position after that or None)."""
        a_prev = self._a_prev
        pos = 0
        last = draft = None
        while pos < len(ids):
            T = self.T_pre if (self.T_pre and len(ids) - pos >= self.T_pre) else self.T_dec
            mis = (start + pos) % self.T_pre if (self.T_pre and self._chunked_prefill) else 0
            if T == self.T_pre and mis:  # chunked SDPA needs chunk starts aligned to T_pre
                T = self.T_dec
            n = min(T, len(ids) - pos, (self.T_pre - mis) if mis else T)
            self.use_T(T)
            chunk = ids[pos:pos + n]
            self.run_spec(chunk, start + pos, a_prev)
            # the last chunk's pick is the next token (sampled if a sampler is set; earlier chunks draw nothing, so
            # a seed gives the same text however the prompt is split); before the MTP run (it overwrites hid_src)
            last = self._pick_row(len(chunk) - 1) if pos + len(chunk) == len(ids) else -1
            if with_mtp:
                toks = [ids[pos + i + 1] if pos + i + 1 < len(ids) else last for i in range(len(chunk))]
                mam = self.mtp.run(toks, start + pos + 1, list(range(len(chunk))))
                draft = mam[len(chunk) - 1]
                self._mtp_last_rows = len(chunk)
            a_prev = len(chunk)
            pos += len(chunk)
        self._a_prev = a_prev
        self.use_T(self.T_dec)
        return last, start + pos, draft

    def init_large_prefill(self, t_max=1024):
        """Large-T prompt prefill (tsumi/prefill.py); used by prefill() for prompts of 64+ tokens."""
        from .prefill import LargePrefill

        self.large = LargePrefill(self, t_max=t_max)
        self.large.warmup()

    def prefill(self, ids, start=0, with_mtp=False):
        """Prompt prefill: the large-T path when available (chunk starts aligned to 64 by first running the
        T-row mode up to the boundary), else prefill_spec. Same return value as prefill_spec."""
        lp = getattr(self, "large", None)
        k = (-start) % 64
        if lp is None or len(ids) - k < 64:
            return self.prefill_spec(ids, start=start, with_mtp=with_mtp)
        if k:
            smp, self.sampler = self.sampler, None  # only the final token is picked with the sampler
            try:
                self.prefill_spec(ids[:k], start=start, with_mtp=with_mtp)
            finally:
                self.sampler = smp
        return lp.run(ids[k:], start=start + k, with_mtp=with_mtp)

    def capture(self):
        self.forward()  # compile
        self.trace_id = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self.logits_out = self.forward()
        ttnn.end_trace_capture(self.dev, self.trace_id, cq_id=0)

    def run(self, token, pos):
        self.set_inputs(token, pos)
        if self.trace_id is not None:
            ttnn.execute_trace(self.dev, self.trace_id, cq_id=0, blocking=False)
            return self.logits_out
        return self.forward()

    def step(self, token, pos):
        """One decode step, returns float logits [V]."""
        return self.logits_to_host(self.run(token, pos))

    def step_greedy(self, token, pos):
        return self.argmax_to_host(self.run(token, pos))

    def reset(self):
        for L in self.layers:
            if L["type"] == "linear_attention":
                m = L["mix"]
                z = torch.zeros(m.qkv_dim, m.Kc)
                m.set_state_from_torch(z, torch.zeros(m.Nv, m.Dk, m.Dv))
