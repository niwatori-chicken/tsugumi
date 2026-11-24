"""CPU reference of Inkling 2 Mini (float32, whole sequence, no cache), written from transformers'
modeling_inkling2_loop.py (v5.24). Used to check the device engine (scripts/inkling_check.py).

    ref = Reference(model_dir)
    logits = ref.forward(ids)            # [T, vocab_unpadded] (soft-capped)
    ref.forward(ids, trace=d)            # d gets per-application intermediates (for layer-by-layer comparison)
"""
import math

import torch
import torch.nn.functional as F

from .config import InklingConfig
from .weights import Checkpoint


def rms(x, w=None, eps=1e-6):
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x if w is None else x * w


def l2clip(x, thr):
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (thr / (n + 1e-6)).clamp(max=1.0)


def softcap(x, c):
    return torch.tanh(x / c) * c


def tau(cfg, pos):
    """[T] log scaling factors."""
    return 1.0 + cfg.log_alpha * torch.log(((pos + 1).float() / cfg.log_n_floor).clamp(min=1.0))


class Reference:
    def __init__(self, model_dir, ckpt=None, cfg=None):
        self.cfg = cfg or InklingConfig.load(model_dir)
        self.ck = ckpt or Checkpoint(model_dir)
        self._w = {}

    def w(self, name):
        t = self._w.get(name)
        if t is None:
            t = self._w[name] = self.ck.get(name)
        return t

    # ---- pieces ------------------------------------------------------------------------------------------------
    def attn_res(self, query, blocks, h):
        """Block attention residual: queries [nq, H] over normalized [blocks..., h]. Returns [nq] list of [T, H]."""
        st = rms(torch.stack(blocks + [h], dim=1))  # [T, n, H]
        q = query.view(-1, st.shape[-1])
        p = torch.softmax(torch.einsum("tnh,qh->tqn", st, q) * st.shape[-1] ** -0.5, dim=-1)
        out = torch.einsum("tqn,tnh->tqh", p, st)
        return list(out.unbind(1))

    def conv(self, x, w):
        """Static short conv with residual: x [T, C], w [C, 4] (w[:, 3] weights the current token)."""
        k = w.shape[1]
        xp = F.pad(x.T[None], (k - 1, 0))
        out = F.conv1d(xp, w[:, None, :], groups=x.shape[1])[0].T
        return (out + x) / math.sqrt(k + 1)

    def dconv(self, x, dyn_in, static_w, proj):
        """Dynamic short conv: taps = static_w[lag, c] + (dyn_in @ proj.T)[head(c), lag]; lag 0 = current token."""
        T, C = x.shape
        k = static_w.shape[0]
        heads = proj.shape[0] // k
        dyn = (dyn_in @ proj.T).view(T, heads, 1, k)
        wts = static_w.T.reshape(heads, C // heads, k) + dyn  # [T, heads, C/heads, k]
        xp = F.pad(x, (0, 0, k - 1, 0))  # [T + k - 1, C]
        win = torch.stack([xp[k - 1 - lag: k - 1 - lag + T] for lag in range(k)], dim=-1)  # [T, C, k]
        out = (win.view(T, heads, C // heads, k) * wts).sum(-1).reshape(T, C)
        return (out + x) / math.sqrt(2 * k + 1)

    def attention(self, i, q_in, k_in, v_in, pos, tr=None):
        c = self.cfg
        a = f"layers.{i}.attn."
        T = q_in.shape[0]
        sliding = c.sliding(i)
        q = (q_in @ self.w(a + "wq_du.weight").T).view(T, c.n_heads, c.head_dim)
        q = l2clip(q * self.w(a + "q_l2_clip.scale"), c.qk_clip)
        k = k_in @ self.w(a + "wk_dv.weight").T
        v = v_in @ self.w(a + "wv_dv.weight").T
        if sliding:
            k = self.conv(k, self.w(a + "k_sconv.weight")[:, 0])
            v = self.conv(v, self.w(a + "v_sconv.weight")[:, 0])
        else:
            k = self.dconv(k, k_in, self.w(a + "k_sconv.static_w"), self.w(a + "k_sconv.weight_proj"))
            v = self.dconv(v, v_in, self.w(a + "v_sconv.static_w"), self.w(a + "v_sconv.weight_proj"))
        k = l2clip(k.view(T, c.n_kv, c.head_dim), c.qk_clip)
        v = rms(v.view(T, c.n_kv, c.head_dim), eps=c.eps)
        r = (q_in @ self.w(a + "wr_du.weight").T).view(T, c.n_heads, c.d_rel)
        r = l2clip(r * self.w(a + "rel_l2_clip.scale"), c.rel_clip)
        if not sliding:
            tq = tau(c, pos)[:, None, None]
            q = q * tq
            r = r * tq
        ext = c.window if sliding else c.rel_extent
        rel = r @ self.w(a + "rel_logits_proj.proj")  # [T, H, ext]
        dist = pos[:, None] - pos[None, :]  # [Tq, Tk]
        inb = (dist >= 0) & (dist < ext)
        bias = rel.gather(-1, dist.clamp(0, ext - 1)[:, None, :].expand(T, c.n_heads, T))
        bias = bias.masked_fill(~inb[:, None, :], 0.0)  # [Tq, H, Tk]
        allowed = (dist >= 0) & ((dist < c.window) if sliding else True)
        rep = c.n_heads // c.n_kv
        kk = k.repeat_interleave(rep, dim=1)  # [T, H, D]
        vv = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("qhd,khd->qhk", q, kk) * c.head_dim ** -0.5 + bias
        s = s.masked_fill(~allowed[:, None, :], float("-inf"))
        sink = (q_in @ self.w(a + "attn_sink_proj.weight").T)[:, :, None]  # [T, H, 1]
        p = torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1]
        o = torch.einsum("qhk,khd->qhd", p, vv).reshape(T, -1)
        g = torch.sigmoid(q_in @ self.w(a + "gate_attn.weight").T)
        out = (o * g) @ self.w(a + "wo_ud.weight").T
        if tr is not None:
            tr.update(q=q, k=k, v=v, r=r, attn=o)
        return out

    def moe(self, i, x, tr=None):
        c = self.cfg
        m = f"layers.{i}.mlp."
        logits = x @ self.w(m + "gate.weight").T
        sc = torch.sigmoid(logits)
        idx = torch.topk(sc + self.w(m + "gate.bias"), c.top_k, dim=-1).indices  # [T, k]
        wt = sc.gather(1, idx)
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * c.route_scale
        out = torch.zeros_like(x)
        for e in idx.unique().tolist():
            tok, kpos = torch.where(idx == e)
            g, u, d = self.ck.expert(i, e)
            xe = x[tok]
            gg, uu = xe @ g.T, xe @ u.T
            act = softcap(gg, c.gate_softcap) * torch.sigmoid(gg) * softcap(uu, c.up_softcap)
            out.index_add_(0, tok, (act @ d.T) * wt[tok, kpos, None])
        if tr is not None:
            tr.update(router_idx=idx, router_w=wt)
        return out

    def msg_gate(self, i, which, h, msg):
        g = f"layers.{i}.controllers.{which}."
        z = torch.cat([rms(h), rms(msg)], dim=-1) @ self.w(g + "down_weight").T
        f = torch.tanh(z @ self.w(g + "forget_weight").T)
        mg = torch.tanh(z @ self.w(g + "gain_weight").T)
        return (1 - f) * h + mg * msg

    def query(self, kind, i):
        """attention_input / mlp_input / final queries ([nq, H])."""
        if kind == "attn":
            return self.w(f"attn_res.attn_queries.{i - 1}").view(-1, self.cfg.hidden)
        if kind == "mlp":
            return self.w(f"attn_res.mlp_queries.{i}").view(-1, self.cfg.hidden)
        return self.w("attn_res.output_query").view(-1, self.cfg.hidden)

    def messages(self, j, prev_blocks, prev_h):
        """Messages (q, k, v, mlp) for loop layer j from the previous pass."""
        qa = self.w("middle_cycle_cross_pass_queries")
        kv = self.w("middle_cycle_cross_pass_kv_queries")
        queries = torch.stack([qa[2 * j], kv[2 * j], kv[2 * j + 1], qa[2 * j + 1]])
        return self.attn_res(queries, prev_blocks, prev_h)

    def layer(self, i, h, blocks, pos, msgs=None, tr=None):
        c = self.cfg
        L = f"layers.{i}."
        res0 = h
        if c.sliding(i):
            a = self.attn_res(self.query("attn", i), blocks, h)[0] if i > 0 else h
            ln = self.w(L + "attn_norm.weight")
            if msgs is None:
                q_in = k_in = v_in = rms(a, ln, c.eps)
            else:
                q_in = rms(self.msg_gate(i, "attn", a, msgs[0]), ln, c.eps)
                k_in = rms(self.msg_gate(i, "k", a, msgs[1]), ln, c.eps)
                v_in = rms(self.msg_gate(i, "v", a, msgs[2]), ln, c.eps)
        else:
            if i > 0:
                qa, ka, va = self.attn_res(self.query("attn", i), blocks, h)
            else:
                qa = ka = va = h
            if msgs is not None:
                qa = self.msg_gate(i, "attn", qa, msgs[0])
                ka = self.msg_gate(i, "k", ka, msgs[1])
                va = self.msg_gate(i, "v", va, msgs[2])
            q_in = rms(qa, self.w(L + "q_input_norm.weight"), c.eps)
            k_in = rms(ka, self.w(L + "k_input_norm.weight"), c.eps)
            v_in = rms(va, self.w(L + "v_input_norm.weight"), c.eps)
        o = self.attention(i, q_in, k_in, v_in, pos, tr)
        o = self.conv(o, self.w(L + "attn_sconv.weight")[:, 0])
        if c.boundary(i):
            blocks = blocks + [res0]
            res = o
        else:
            res = res0 + o
        m_in = self.attn_res(self.query("mlp", i), blocks, res)[0]
        if msgs is not None:
            m_in = self.msg_gate(i, "mlp", m_in, msgs[3])
        m = self.moe(i, rms(m_in, self.w(L + "mlp_norm.weight"), c.eps), tr)
        m = self.conv(m, self.w(L + "mlp_sconv.weight")[:, 0])
        if tr is not None:
            tr.update(attn_out=o, mlp_out=m)
        return res + m

    def embed(self, ids):
        e = self.ck.get_slice("embed.weight")
        x = torch.stack([e[t:t + 1][0] for t in ids]).float()
        return rms(x, self.w("embed_norm.weight"), self.cfg.eps)

    @torch.no_grad()
    def forward(self, ids, trace=None, last_only=False):
        c = self.cfg
        T = len(ids)
        pos = torch.arange(T)
        h = self.embed(ids)
        blocks = []
        apps = []

        def run(i, h, blocks, msgs=None):
            tr = {} if trace is not None else None
            new = self.layer(i, h, blocks, pos, msgs, tr)
            if c.boundary(i):
                blocks = blocks + [h]
            if trace is not None:
                tr.update(layer=i, h_in=h, h_out=new)
                apps.append(tr)
            return new, blocks

        for i in range(c.loop_start):
            h, blocks = run(i, h, blocks)
        pre_h, pre_b = h, blocks
        npre = len(pre_b)
        for i in range(c.loop_start, c.loop_end):
            h, blocks = run(i, h, blocks)
        for _ in range(1, c.passes):
            prev_b, prev_h = blocks[npre + 1:], h
            h, blocks = pre_h, pre_b
            for j, i in enumerate(range(c.loop_start, c.loop_end)):
                h, blocks = run(i, h, blocks, self.messages(j, prev_b, prev_h))
        for i in range(c.loop_end, c.n_layers):
            h, blocks = run(i, h, blocks)
        h = self.attn_res(self.query("final", 0), blocks, h)[0]
        h = rms(h, self.w("norm.weight"), c.eps)
        if last_only:
            h = h[-1:]
        logits = h @ self.ck.get_slice("unembed.weight")[: c.vocab_unpadded].float().T
        if trace is not None:
            trace["apps"] = apps
            trace["final"] = h
        return softcap(logits, c.logit_softcap)
