"""CPU reference of the Inkling 2 DSpark draft (`mtp.safetensors`), written from vLLM's
vllm/models/inkling2/nvidia/dspark.py (main, 2026-11): an anchor + mask block through a 3-layer dense transformer
over context K/V from mixed target features, then a recurrent (`rnn_hidden`) Markov head.

Target features per position: the inputs of layer visits target_layer_ids[:-1] (the AttnRes partial block, i.e. the
residual stream h entering that application) and the final-norm output (the lm_head input), concatenated.

    dr = DraftRef(model_dir, ckpt)          # ckpt: the target Checkpoint (embedding / lm_head are shared)
    drafts = dr.draft(feats[:p], anchor, p) # 7 greedy draft tokens for positions p+1..p+7
"""
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms(x, w=None, eps=1e-6):
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x if w is None else x * w


def rope(x, pos, theta):
    """neox RoPE: x [T, H, D], pos [T]."""
    D = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    f = pos.float()[:, None] * inv[None]  # [T, D/2]
    cos, sin = f.cos()[:, None], f.sin()[:, None]
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class DraftRef:
    def __init__(self, model_dir, ckpt):
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        self.mc = c["mtp_config"]
        t = c["text_config"]
        self.softcap = t["final_logit_softcapping"]
        self.V = t["unpadded_vocab_size"]
        self.ck = ckpt
        h = safe_open(os.path.join(model_dir, "mtp.safetensors"), framework="pt")
        self.w = {k: h.get_tensor(k).float() for k in h.keys()}
        mc = self.mc
        self.H, self.nh, self.nkv, self.hd = mc["hidden_size"], mc["num_attention_heads"], mc["num_key_value_heads"], \
            mc["head_dim"]
        self.eps, self.theta = mc["rms_norm_eps"], mc["rope_theta"]
        self.L = mc["num_hidden_layers"]
        self.local = set(mc["local_layer_ids"])
        self.window = mc["sliding_window_size"]
        self.block = mc["block_size"]
        self.rank = mc["markov_rank"]
        self.alpha = torch.softmax(self.w["fuse_weight"], dim=-1)  # [L + 1, taps]
        self.lm = ckpt.get_slice("unembed.weight")[: self.V].float()  # [V, H]
        self.embed = ckpt.get_slice("embed.weight")
        self.embed_norm = ckpt.get("embed_norm.weight")
        mh = "markov_head."
        self.fold_w = self.lm @ self.w[mh + "hidden_out.weight"]  # [V, r]
        self.fold_b = self.lm @ self.w[mh + "hidden_out.bias"]  # [V]
        self.ctx = None  # per layer (K [P, nkv, hd], V), positions 0..P-1

    def mix(self, feats, row, norm_w):
        """feats [P, taps * H] -> rmsnorm(sum_i alpha[row, i] feats_i) [P, H]."""
        taps = feats.view(feats.shape[0], -1, self.H)
        return rms((taps * self.alpha[row][None, :, None]).sum(1), norm_w, self.eps)

    def set_context(self, feats):
        """Context K / V of positions 0..P-1 for every layer."""
        P = feats.shape[0]
        pos = torch.arange(P)
        self.ctx = []
        for i in range(self.L):
            a = f"layers.{i}."
            cx = self.mix(feats, i, self.w[a + "dflare_norm.weight"])
            k = (cx @ self.w[a + "self_attn.k_proj_ctx.weight"].T).view(P, self.nkv, self.hd)
            k = rope(rms(k, self.w[a + "self_attn.k_norm_ctx.weight"], self.eps), pos, self.theta)
            v = (cx @ self.w[a + "self_attn.v_proj_ctx.weight"].T).view(P, self.nkv, self.hd)
            self.ctx.append((k, v))

    def attn(self, i, x, p0):
        a = f"layers.{i}.self_attn."
        B = x.shape[0]
        pos = torch.arange(p0, p0 + B)
        q = rope(rms((x @ self.w[a + "q_proj.weight"].T).view(B, self.nh, self.hd), self.w[a + "q_norm.weight"],
                     self.eps), pos, self.theta)
        k = rope(rms((x @ self.w[a + "k_proj.weight"].T).view(B, self.nkv, self.hd), self.w[a + "k_norm.weight"],
                     self.eps), pos, self.theta)
        v = (x @ self.w[a + "v_proj.weight"].T).view(B, self.nkv, self.hd)
        kc, vc = self.ctx[i]
        kc, vc = kc[:p0], vc[:p0]
        if i in self.local:  # the window before the block start
            kc, vc = kc[max(0, p0 - self.window):], vc[max(0, p0 - self.window):]
        K = torch.cat([kc, k]).repeat_interleave(self.nh // self.nkv, dim=1)  # bidirectional within the block
        Vv = torch.cat([vc, v]).repeat_interleave(self.nh // self.nkv, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, K) * self.hd ** -0.5
        o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), Vv).reshape(B, -1)
        return o @ self.w[a + "o_proj.weight"].T

    @torch.no_grad()
    def hidden(self, anchor, p0):
        """Draft hidden states of the block (anchor at position p0, then masks) [block, H]."""
        e = self.embed[anchor:anchor + 1].float()
        x = self.w["mask_embed"][None].repeat(self.block, 1)
        x[0] = rms(e, self.embed_norm, self.eps)[0]
        res = x
        for i in range(self.L):
            a = f"layers.{i}."
            h = rms(res, self.w[a + "input_layernorm.weight"], self.eps)
            res = res + self.attn(i, h, p0)
            h = rms(res, self.w[a + "post_attention_layernorm.weight"], self.eps)
            g = h @ self.w[a + "mlp.gate_proj.weight"].T
            u = h @ self.w[a + "mlp.up_proj.weight"].T
            res = res + (F.silu(g) * u) @ self.w[a + "mlp.down_proj.weight"].T
        return rms(res, self.w["norm.weight"], self.eps)

    @torch.no_grad()
    def draft(self, feats, anchor, p0, markov=True):
        """feats [p0, taps * H] (positions < p0, committed); anchor = token at p0. Returns block - 1 draft tokens
        (positions p0 + 1 ..) by greedy Markov sampling."""
        self.set_context(feats)
        hid = self.hidden(anchor, p0)[1:]  # mask rows -> tokens at p0 + 1 ..
        base = hid @ self.lm.T  # [n, V]
        if not markov:
            return (torch.tanh(base / self.softcap) * self.softcap).argmax(-1).tolist()
        mh = "markov_head."
        W, b = self.w[mh + "rnn_proj.weight"], self.w[mh + "rnn_proj.bias"]
        r = self.rank
        f_rnn = self.mix(feats[-1:], self.L, self.w[mh + "dflare_norm.weight"])  # last committed token
        ctx = torch.cat([hid, f_rnn.expand_as(hid)], dim=-1) @ W[:, 2 * r:].T + b  # [n, 3r]
        state = torch.zeros(r)
        prev = anchor
        out = []
        for i in range(hid.shape[0]):
            raw = ctx[i] + torch.cat([state, self.w[mh + "markov_w1.weight"][prev]]) @ W[:, :2 * r].T
            gate, cand, o = raw[:r], raw[r:2 * r], raw[2 * r:]
            g = torch.sigmoid(gate)
            state = g * state + (1 - g) * torch.tanh(cand)
            bias = self.fold_b + self.fold_w @ torch.tanh(o)
            logits = torch.tanh((base[i] + bias) / self.softcap) * self.softcap
            prev = int(logits.argmax())
            out.append(prev)
        return out
