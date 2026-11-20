"""The Qwen3.5 vision encoder on the device (27 ViT blocks + 2x2 merger), one image at a time.

Layout choices:
- heads are 72 wide, not a tile multiple: q/k/v are padded to 96 per head with zero weight columns (and the
  attention output projection gets zero rows there), so nlp_create_qkv_heads / nlp_concat_heads work as is and
  the padding never needs slicing. The softmax scale stays 1/sqrt(72).
- 2-D RoPE (rotate_half over the 72 dims): x * cos + (x @ P) * sin per head, cos/sin from the host.
- the MLP's 4304 is padded to 4320 (zero columns / rows; gelu(0) = 0).
- the patches are padded to a multiple of 512 rows (always with a key mask, so one program per bucket); padded
  rows are dropped at the end. Every bucket must be compiled before the decode traces are captured (warmup()): a
  program first run after the capture was corrupted by the next trace replay and hung the device on its next run
  (its device-side data lands in memory the trace uses). The processor orders patches so that each 2x2 merge group is 4 consecutive rows: the merger
  is a reshape [Np, 1152] -> [Np/4, 4608].
- position embeddings (bilinear interpolation of a 48x48 table) and cos/sin are computed on the host.
Weights bfp8 (about 0.5 GB), activations bf16, matmuls with fp32 accumulation."""
import math
import os
import time

import torch
import ttnn

from .model import LazySD
from .ops import has_cache, load_or_convert

DRAM = ttnn.DRAM_MEMORY_CONFIG
VP = "model.visual."
HDP = 96  # padded head width
ROWS = 128  # SDPA chunk
BUCKET = 512  # patch rows padded to a multiple of this: few shapes, all compiled by warmup() before the traces


_DBG = os.environ.get("TSUMI_VIT_DEBUG") == "1"


def _dbg(dev, msg):
    if _DBG:
        ttnn.synchronize_device(dev)
        print(f"[vit] {msg}", flush=True)


def _up(t, dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=dev, memory_config=DRAM)


class VisionTT:
    def __init__(self, dev, model_dir, cache_dir, wdtype=ttnn.bfloat8_b, log=print):
        from transformers import AutoConfig

        t0 = time.time()
        cfg = AutoConfig.from_pretrained(model_dir).vision_config
        self.dev, self.cfg = dev, cfg
        H, NH = cfg.hidden_size, cfg.num_heads
        HD = H // NH
        self.H, self.NH, self.HD = H, NH, HD
        self.I = cfg.intermediate_size
        self.Ip = -(-self.I // 32) * 32
        self.merge = cfg.spatial_merge_size
        self.out = cfg.out_hidden_size
        sd = LazySD(model_dir)
        c = os.path.join(cache_dir, "vision")
        os.makedirs(cache_dir, exist_ok=True)

        def W(name, w_kn):  # weight [K, N] (torch orientation already transposed), cached
            cf = f"{c}.{name}"
            return load_or_convert(dev, None if has_cache(cf, wdtype) else w_kn(), wdtype, DRAM, cf)

        bias = lambda b: _up(b.float().view(1, -1).to(torch.bfloat16), dev)
        norm = lambda w: ttnn.from_torch(w.float().to(torch.bfloat16).reshape(1, 1, -1, 32), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=DRAM)

        def pad_heads_cols(w, b):  # w [NH*HD, K] (out, in) for one of q/k/v -> [K, NH*HDP], bias [NH*HDP]
            K = w.shape[1]
            wp = torch.zeros(NH, HDP, K)
            wp[:, :HD] = w.float().view(NH, HD, K)
            bp = torch.zeros(NH, HDP)
            bp[:, :HD] = b.float().view(NH, HD)
            return wp.view(NH * HDP, K).T, bp.view(-1)

        pe = sd[VP + "patch_embed.proj.weight"]  # [H, 3, 2, 16, 16]
        self.w_patch = W("patch", lambda: pe.float().reshape(H, -1).T.to(torch.bfloat16))
        self.b_patch = bias(sd[VP + "patch_embed.proj.bias"])
        self.pos_table = sd[VP + "pos_embed.weight"].float()  # host: bilinear interpolation per image
        self.blocks = []
        for i in range(cfg.depth):
            p = f"{VP}blocks.{i}."
            wq, bq = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]

            def qkv_w(wq=wq, bq=bq):
                parts = [pad_heads_cols(wq[s * H:(s + 1) * H], bq[s * H:(s + 1) * H]) for s in range(3)]
                return torch.cat([w for w, _ in parts], dim=1).to(torch.bfloat16)

            parts_b = [pad_heads_cols(wq[s * H:(s + 1) * H], bq[s * H:(s + 1) * H])[1] for s in range(3)]

            def proj_w(p=p):  # [H(out), H(in)] -> [NH*HDP, H] with zero rows at the head padding
                w = sd[p + "attn.proj.weight"].float()
                wp = torch.zeros(NH, HDP, H)
                wp[:, :HD] = w.T.reshape(NH, HD, H)
                return wp.view(NH * HDP, H).to(torch.bfloat16)

            def fc1_w(p=p):
                w = torch.zeros(H, self.Ip)
                w[:, :self.I] = sd[p + "mlp.linear_fc1.weight"].float().T
                return w.to(torch.bfloat16)

            def fc2_w(p=p):
                w = torch.zeros(self.Ip, H)
                w[:self.I] = sd[p + "mlp.linear_fc2.weight"].float().T
                return w.to(torch.bfloat16)

            b1 = torch.zeros(self.Ip)
            b1[:self.I] = sd[p + "mlp.linear_fc1.bias"].float()
            self.blocks.append(dict(
                n1w=norm(sd[p + "norm1.weight"]), n1b=norm(sd[p + "norm1.bias"]),
                n2w=norm(sd[p + "norm2.weight"]), n2b=norm(sd[p + "norm2.bias"]),
                qkv=W(f"b{i}.qkv", qkv_w), qkv_b=bias(torch.cat(parts_b)),
                proj=W(f"b{i}.proj", proj_w), proj_b=bias(sd[p + "attn.proj.bias"]),
                fc1=W(f"b{i}.fc1", fc1_w), fc1_b=bias(b1),
                fc2=W(f"b{i}.fc2", fc2_w), fc2_b=bias(sd[p + "mlp.linear_fc2.bias"]),
            ))
        m = VP + "merger."
        self.m_nw, self.m_nb = norm(sd[m + "norm.weight"]), norm(sd[m + "norm.bias"])
        self.m_fc1 = W("m.fc1", lambda: sd[m + "linear_fc1.weight"].float().T.to(torch.bfloat16))
        self.m_fc1_b = bias(sd[m + "linear_fc1.bias"])
        self.m_fc2 = W("m.fc2", lambda: sd[m + "linear_fc2.weight"].float().T.to(torch.bfloat16))
        self.m_fc2_b = bias(sd[m + "linear_fc2.bias"])
        # rotate_half over the first HD dims of each padded head: (x @ P)[j] = -x[j + HD/2] (j < HD/2), x[j - HD/2]
        P = torch.zeros(HDP, HDP)
        h2 = HD // 2
        for j in range(h2):
            P[j + h2, j] = -1.0
            P[j, j + h2] = 1.0
        self.P = _up(P, dev)
        self.inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HD // 2, 2, dtype=torch.float) / (HD // 2)))
        self.mm_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                      fp32_dest_acc_en=True, packer_l1_acc=True)
        self.hi_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                      fp32_dest_acc_en=True, packer_l1_acc=False)
        self.max_patches = 0  # set by warmup(); larger images would compile new programs after the traces
        grid = dev.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        log(f"vision encoder on the device: {cfg.depth} blocks, {time.time() - t0:.0f}s")

    # ---------------- host-side inputs ----------------
    def _host_inputs(self, grid_thw):
        from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids

        side = int(self.cfg.num_position_embeddings ** 0.5)
        idx, wts = get_vision_bilinear_indices_and_weights(grid_thw, num_grid_per_side=side,
                                                           spatial_merge_size=self.merge)
        pos = (self.pos_table[idx] * wts[:, :, None]).sum(0)  # [N, H]
        rc = get_vision_position_ids(grid_thw, self.merge)  # [N, 2] (row, col)
        f = (rc.unsqueeze(-1).float() * self.inv_freq).flatten(1)  # [N, HD/2]
        emb = torch.cat([f, f], dim=-1)  # [N, HD]
        cos = torch.zeros(emb.shape[0], HDP)
        sin = torch.zeros(emb.shape[0], HDP)
        cos[:, :self.HD], sin[:, :self.HD] = emb.cos(), emb.sin()
        return pos, cos, sin

    # ---------------- device ----------------
    def _mm(self, x, w, b, act=None):
        return ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, fused_activation=act, compute_kernel_config=self.mm_ck,
                                                memory_config=DRAM, dtype=ttnn.bfloat16)

    def _ln(self, x, w, b):
        return ttnn.layer_norm(x, weight=w, bias=b, epsilon=1e-6, memory_config=DRAM, compute_kernel_config=self.hi_ck)

    def _rope(self, x, cos, sin):
        r = ttnn.matmul(x, self.P, memory_config=DRAM, compute_kernel_config=self.hi_ck)
        a = ttnn.multiply(x, cos, memory_config=DRAM)
        r2 = ttnn.multiply(r, sin, memory_config=DRAM)
        ttnn.deallocate(r)
        ttnn.deallocate(x)
        out = ttnn.add(a, r2, memory_config=DRAM)
        ttnn.deallocate(a)
        ttnn.deallocate(r2)
        return out

    def _block(self, B, x, cos, sin, mask, Np):
        h = self._ln(x, B["n1w"], B["n1b"])
        _dbg(self.dev, "ln1")
        qkv = self._mm(h, B["qkv"], B["qkv_b"])
        _dbg(self.dev, "qkv")
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=self.NH, num_kv_heads=self.NH,
                                                         transpose_k_heads=False, memory_config=DRAM)
        ttnn.deallocate(qkv)
        _dbg(self.dev, "heads")
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        _dbg(self.dev, "rope")
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, exp_approx_mode=False,
                                     q_chunk_size=ROWS, k_chunk_size=ROWS)
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False,
                                                          scale=1.0 / math.sqrt(self.HD), program_config=cfg,
                                                          compute_kernel_config=self.hi_ck, memory_config=DRAM)
        for t in (q, k, v):
            ttnn.deallocate(t)
        _dbg(self.dev, "sdpa")
        o2 = ttnn.experimental.nlp_concat_heads(o, memory_config=DRAM)
        ttnn.deallocate(o)
        a = self._mm(o2, B["proj"], B["proj_b"])
        ttnn.deallocate(o2)
        x2 = ttnn.add(x, a, memory_config=DRAM)
        ttnn.deallocate(a)
        ttnn.deallocate(x)
        h = self._ln(x2, B["n2w"], B["n2b"])
        m = self._mm(h, B["fc1"], B["fc1_b"])
        ttnn.deallocate(h)
        mg = ttnn.gelu(m, variant=ttnn.GeluVariant.Tanh, memory_config=DRAM)
        ttnn.deallocate(m)
        y = self._mm(mg, B["fc2"], B["fc2_b"])
        ttnn.deallocate(mg)
        _dbg(self.dev, "mlp")
        x3 = ttnn.add(x2, y, memory_config=DRAM)
        ttnn.deallocate(y)
        ttnn.deallocate(x2)
        return x3

    def warmup(self, max_pixels):
        """Compile every bucket up to max_pixels per image (call before capturing the decode traces)."""
        nmax = max_pixels // (self.cfg.patch_size ** 2)
        self.max_patches = -(-nmax // BUCKET) * BUCKET
        for Np in range(BUCKET, -(-nmax // BUCKET) * BUCKET + 1, BUCKET):
            n = Np - 4  # a padded bucket (the mask always applies)
            self.forward(torch.zeros(n, 3 * self.cfg.temporal_patch_size * self.cfg.patch_size ** 2),
                         torch.tensor([[1, 2, n // 2]]))
        ttnn.synchronize_device(self.dev)
        return nmax

    def check(self, n_patches):
        if n_patches > self.max_patches:
            raise ValueError(f"image: {n_patches} patches, more than the warmed-up {self.max_patches}")

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw):
        """pixel_values [N, 1536] (HF processor), grid_thw [[1, h, w]] -> [N/4, out] bf16 on the host."""
        dev = self.dev
        N = pixel_values.shape[0]
        Np = -(-N // BUCKET) * BUCKET
        pos, cos, sin = self._host_inputs(grid_thw)
        pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, Np - N))
        xp = _up(pad(pixel_values.float()).view(1, 1, Np, -1).to(torch.bfloat16), dev)
        pos_t = _up(pad(pos).view(1, 1, Np, -1).to(torch.bfloat16), dev)
        cos_t = _up(pad(cos).view(1, 1, Np, HDP).to(torch.bfloat16), dev)
        sin_t = _up(pad(sin).view(1, 1, Np, HDP).to(torch.bfloat16), dev)
        mk = torch.zeros(1, 1, Np, Np)
        mk[..., N:] = float("-inf")
        mask = _up(mk.to(torch.bfloat16), dev)
        _dbg(dev, "inputs uploaded")
        x0 = self._mm(xp, self.w_patch, self.b_patch)
        _dbg(dev, "patch embed")
        ttnn.deallocate(xp)
        x = ttnn.add(x0, pos_t, memory_config=DRAM)
        ttnn.deallocate(x0)
        ttnn.deallocate(pos_t)
        for B in self.blocks:
            x = self._block(B, x, cos_t, sin_t, mask, Np)
        for t in (cos_t, sin_t, mask):
            ttnn.deallocate(t)
        h = self._ln(x, self.m_nw, self.m_nb)
        ttnn.deallocate(x)
        g = self.merge ** 2
        hr = ttnn.reshape(h, (1, 1, Np // g, self.H * g))
        y = self._mm(hr, self.m_fc1, self.m_fc1_b)
        ttnn.deallocate(h)  # (hr may be a view of h)
        yg = ttnn.gelu(y, memory_config=DRAM)
        ttnn.deallocate(y)
        z = self._mm(yg, self.m_fc2, self.m_fc2_b)
        ttnn.deallocate(yg)
        out = ttnn.to_torch(z).reshape(Np // g, -1)[: N // g, : self.out].to(torch.bfloat16)
        ttnn.deallocate(z)
        return out
