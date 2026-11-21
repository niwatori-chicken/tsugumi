"""Gated full attention (Qwen3.5/3.8 full_attention layers) — single-device B=1 decode.

HF reference: Qwen3_5Attention. q_proj gives per head [q(256) | gate(256)]; q/k RMSNorm with (1+w);
partial RoPE (rotate_half on the first rope_dim dims); out = attn * sigmoid(gate) -> o_proj.
"""
import torch
import ttnn

from .env import env
from .kv_kernel import KVWriteKernel
from .ops import DecodeLinear, has_cache

_L1 = ttnn.L1_MEMORY_CONFIG


def kv_dtype():
    """KV cache format (TSUGUMI_KV_DTYPE): bf16 (68 KiB/token over 17 layers) or bfp8 (36 KiB/token)."""
    return {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[env("KV_DTYPE", "bfp8")]


def rope_rot_matrix(head_dim, rope_dim):
    """P such that x @ P == rotate_half(x[:rope_dim]) (zero on the pass-through dims)."""
    P = torch.zeros(head_dim, head_dim)
    h = rope_dim // 2
    for i in range(h):
        P[i + h, i] = -1.0  # out[i] = -x[i+h]
        P[i, i + h] = 1.0  # out[i+h] = x[i]
    return P


class AttnDecode:
    def __init__(self, dev, sd, prefix, cfg, max_seq, wdtype=ttnn.bfloat8_b, cache_prefix=None):
        self.dev = dev
        self.NH = cfg["num_attention_heads"]
        self.NKV = cfg["num_key_value_heads"]
        self.HD = cfg["head_dim"]
        self.rope_dim = int(self.HD * cfg.get("partial_rotary_factor", 0.25))
        self.max_seq = max_seq
        NH, NKV, HD = self.NH, self.NKV, self.HD
        g = lambda n: sd[prefix + n]
        cp = (lambda s: f"{cache_prefix}.{s}") if cache_prefix else (lambda s: None)
        w = wo = None
        if not has_cache(cp("qkvg"), wdtype):
            qg = g("q_proj.weight").view(NH, 2, HD, -1)
            wq, wgate = qg[:, 0].reshape(NH * HD, -1), qg[:, 1].reshape(NH * HD, -1)
            w = torch.cat([wq, g("k_proj.weight"), g("v_proj.weight"), wgate], 0).T.contiguous().to(torch.bfloat16)
        if not has_cache(cp("o"), wdtype):
            wo = g("o_proj.weight").T.contiguous().to(torch.bfloat16)
        self.wqkvg = DecodeLinear(dev, w, wdtype, cache_file=cp("qkvg"), name="attn_qkvg",
                                  shape=(cfg["hidden_size"], NH * HD * 2 + 2 * NKV * HD))
        self.wo = DecodeLinear(dev, wo, wdtype, cache_file=cp("o"), name="attn_o", shape=(NH * HD, cfg["hidden_size"]))

        def tt(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.to(torch.bfloat16), dtype=dt, layout=layout, device=dev)

        self.q_norm = tt((1.0 + g("q_norm.weight").float()).reshape(1, HD))
        # same weight with the softmax scale folded in (exact: HD**-0.5 = 1/16), for the GQA kernel path
        self.q_norm_s = tt((1.0 + g("q_norm.weight").float()).reshape(1, HD) * HD**-0.5)
        self.k_norm = tt((1.0 + g("k_norm.weight").float()).reshape(1, HD))
        self.P = tt(rope_rot_matrix(HD, self.rope_dim))
        self.k_cache = self.v_cache = None  # non-paged T=1 cache, allocated on first use (see _ensure_kv)
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, HD), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                               q_chunk_size=0, k_chunk_size=128)
        self.sdpa_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                        fp32_dest_acc_en=True, packer_l1_acc=False)
        self.scale = HD**-0.5

    def _ensure_kv(self):
        if self.k_cache is None:
            z = torch.zeros(1, self.NKV, self.max_seq, self.HD)
            self.k_cache = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
            self.v_cache = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)

    # ---------------- T-row path: T "users" sharing one paged KV cache ----------------
    def init_spec(self, T, block_size=64, chunked=False):
        """chunked=True: this T is a prefill chunk size; use flash-style chunked SDPA (reads the KV once per
        chunk instead of once per row). Requires the chunk's first position to be a multiple of T."""
        NKV, HD = self.NKV, self.HD
        if chunked:
            self.chunked_T = T
            g = self.dev.compute_with_storage_grid_size()
            self.sdpa_cfg_c = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False,
                                                     q_chunk_size=T, k_chunk_size=T)
        nb = self.max_seq // block_size
        if not hasattr(self, "kvw_T"):
            self.kv_dtype = kv_dtype()
            # zero-filled on the device (unwritten positions inside a chunk are masked, but 0 * NaN would not be)
            self.kp = ttnn.zeros([nb, NKV, block_size, HD], dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT,
                                 device=self.dev)
            self.vp = ttnn.zeros([nb, NKV, block_size, HD], dtype=self.kv_dtype, layout=ttnn.TILE_LAYOUT,
                                 device=self.dev)
            self.kvw_T = {}
        self.kvw_T[T] = KVWriteKernel(self.dev, T, NKV, HD, block_size, nb, self.kv_dtype)
        # one user per paged_update_cache call: users of one step share KV tiles, and the op does a
        # read-modify-write of whole tiles per user core, so concurrent users in one tile clobber each other.
        self.kv_shard_1 = ttnn.create_sharded_memory_config(
            shape=(32, HD), core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        g = self.dev.compute_with_storage_grid_size()
        self.sdpa_cfg_T = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False,
                                                 q_chunk_size=0, k_chunk_size=0)

    def forward_spec(self, x, sh):
        """x [1,1,32,H] rows 0..T-1; sh.cur_pos [T] int32, sh.page_table [T,nb], sh.cos/sin [1,T,1,HD]."""
        NH, NKV, HD, T = self.NH, self.NKV, self.HD, sh.T
        y = self.wqkvg(x, out_mem=_L1)
        def part(lo, hi, nh):
            t = ttnn.slice(y, (0, 0, 0, lo), (1, 1, T, hi), memory_config=_L1)
            return ttnn.reshape(t, (1, T, nh, HD), memory_config=_L1)
        q = part(0, NH * HD, NH)
        k = part(NH * HD, (NH + NKV) * HD, NKV)
        v = part((NH + NKV) * HD, (NH + 2 * NKV) * HD, NKV)
        gate = part((NH + 2 * NKV) * HD, (2 * NH + 2 * NKV) * HD, NH)
        gqa = getattr(sh, "gqa", None)
        if gqa is not None:  # gate in the [rows, NH*HD] layout the GQA kernel output uses
            gate = ttnn.slice(y, (0, 0, 0, (NH + 2 * NKV) * HD), (1, 1, 32, (2 * NH + 2 * NKV) * HD),
                              memory_config=_L1)
        ttnn.deallocate(y)
        q = ttnn.rms_norm(q, weight=self.q_norm_s if gqa is not None else self.q_norm, epsilon=1e-6,
                          memory_config=_L1)
        k = ttnn.rms_norm(k, weight=self.k_norm, epsilon=1e-6, memory_config=_L1)
        cos, sin = sh.cos, sh.sin
        q = ttnn.add(ttnn.multiply(q, cos, memory_config=_L1),
                     ttnn.multiply(ttnn.matmul(q, self.P, memory_config=_L1), sin, memory_config=_L1),
                     memory_config=_L1)
        k = ttnn.add(ttnn.multiply(k, cos, memory_config=_L1),
                     ttnn.multiply(ttnn.matmul(k, self.P, memory_config=_L1), sin, memory_config=_L1),
                     memory_config=_L1)
        if getattr(self, "kv_rowwise", False):
            for i in range(T):
                for src, cache in ((k, self.kp), (v, self.vp)):
                    r = ttnn.slice(src, (0, i, 0, 0), (1, i + 1, NKV, HD), memory_config=_L1)
                    r = ttnn.to_memory_config(ttnn.pad(r, [(0, 0), (0, 0), (0, 32 - NKV), (0, 0)], 0.0),
                                              self.kv_shard_1)
                    ttnn.experimental.paged_update_cache(cache, r, update_idxs_tensor=sh.cur_pos_rows[i],
                                                         page_table=sh.page_table_1)
                    ttnn.deallocate(r)
        else:
            self.kvw_T[T](k, v, self.kp, self.vp, sh.cur_pos, sh.page_table_1)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        if gqa is not None:
            if T == 32:
                q = ttnn.permute(q, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, NH, 32, HD]
            else:
                q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            o = gqa(q, self.kp, self.vp, sh.cur_pos, sh.page_table_1)  # [1, 1, 32, NH*HD], rows >= T zero
            ttnn.deallocate(q)
            o = ttnn.multiply(o, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(gate)
            return self.wo(o, out_mem=_L1)
        if T == getattr(self, "chunked_T", None):
            q = ttnn.permute(q, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, NH, T, HD]
            o = ttnn.transformer.chunked_scaled_dot_product_attention(
                q, self.kp, self.vp, sh.page_table_1, chunk_start_idx_tensor=sh.cur_pos_rows[0], scale=self.scale,
                program_config=self.sdpa_cfg_c, compute_kernel_config=self.sdpa_ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(q)
            o = ttnn.permute(o, (0, 2, 1, 3), memory_config=_L1)  # [1, T, NH, HD]
            return self._spec_out(o, gate, T)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, self.kp, self.vp, page_table_tensor=sh.page_table, cur_pos_tensor=sh.cur_pos, scale=self.scale,
            program_config=self.sdpa_cfg_T, compute_kernel_config=self.sdpa_ck, memory_config=_L1)
        ttnn.deallocate(q)
        return self._spec_out(o, gate, T)

    def _spec_out(self, o, gate, T):
        NH, HD = self.NH, self.HD
        o = ttnn.multiply(o, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
        ttnn.deallocate(gate)
        o = ttnn.reshape(o, (1, 1, T, NH * HD), memory_config=_L1)
        if T != 32:
            o = ttnn.pad(o, [(0, 0), (0, 0), (0, 32 - T), (0, 0)], 0.0)
        return self.wo(o, out_mem=_L1)

    def forward_decode(self, x, cur_pos, cos, sin):
        """x [1,1,32,H]; cur_pos int32 [1]; cos/sin [1,1,1,HD] (1/0 on pass dims). Returns [1,1,32,Np]."""
        NH, NKV, HD = self.NH, self.NKV, self.HD
        y = self.wqkvg(x, out_mem=_L1)
        q = ttnn.slice(y, (0, 0, 0, 0), (1, 1, 1, NH * HD), memory_config=_L1)
        k = ttnn.slice(y, (0, 0, 0, NH * HD), (1, 1, 1, (NH + NKV) * HD), memory_config=_L1)
        v = ttnn.slice(y, (0, 0, 0, (NH + NKV) * HD), (1, 1, 1, (NH + 2 * NKV) * HD), memory_config=_L1)
        gate = ttnn.slice(y, (0, 0, 0, (NH + 2 * NKV) * HD), (1, 1, 1, (2 * NH + 2 * NKV) * HD), memory_config=_L1)
        ttnn.deallocate(y)
        q = ttnn.reshape(q, (1, 1, NH, HD), memory_config=_L1)
        k = ttnn.reshape(k, (1, 1, NKV, HD), memory_config=_L1)
        v = ttnn.reshape(v, (1, 1, NKV, HD), memory_config=_L1)
        q = ttnn.rms_norm(q, weight=self.q_norm, epsilon=1e-6, memory_config=_L1)
        k = ttnn.rms_norm(k, weight=self.k_norm, epsilon=1e-6, memory_config=_L1)
        # rope: x*cos + (x@P)*sin
        q = ttnn.add(ttnn.multiply(q, cos, memory_config=_L1),
                     ttnn.multiply(ttnn.matmul(q, self.P, memory_config=_L1), sin, memory_config=_L1),
                     memory_config=_L1)
        k = ttnn.add(ttnn.multiply(k, cos, memory_config=_L1),
                     ttnn.multiply(ttnn.matmul(k, self.P, memory_config=_L1), sin, memory_config=_L1),
                     memory_config=_L1)
        k = ttnn.to_memory_config(ttnn.pad(k, [(0, 0), (0, 0), (0, 32 - NKV), (0, 0)], 0.0), self.kv_shard)
        v = ttnn.to_memory_config(ttnn.pad(v, [(0, 0), (0, 0), (0, 32 - NKV), (0, 0)], 0.0), self.kv_shard)
        self._ensure_kv()
        ttnn.experimental.paged_update_cache(self.k_cache, k, update_idxs_tensor=cur_pos)
        ttnn.experimental.paged_update_cache(self.v_cache, v, update_idxs_tensor=cur_pos)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q, self.k_cache, self.v_cache, cur_pos_tensor=cur_pos, scale=self.scale,
            program_config=self.sdpa_cfg, compute_kernel_config=self.sdpa_ck, memory_config=_L1)
        ttnn.deallocate(q)
        gate = ttnn.reshape(gate, (1, 1, NH, HD), memory_config=_L1)
        o = ttnn.multiply(o, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
        o = ttnn.reshape(o, (1, 1, 1, NH * HD), memory_config=_L1)
        o = ttnn.pad(o, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        return self.wo(o, out_mem=_L1)
