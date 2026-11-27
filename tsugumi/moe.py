"""Sparse MoE block of Qwen3.5/3.6-MoE (e.g. Qwen3.6-35B-A3B). Work in progress (branch moe-probe, doc/12).

Method B (this file, for correctness and a baseline): every row goes through all E routed experts as a few dense
matmuls, and the router's weights (zero for the experts not in a row's top k) are applied in between, so the shapes
do not depend on the routing and both the T-row decode trace and the large-T prefill trace work unchanged.

Experts are stored in groups of G: per group gate [H, G*I], up [H, G*I] (expert j of the group in columns j*I..)
and down [G*I, H] (expert j in rows j*I..; K padded to the gate's padded N). For a group,
  a = silu(x @ gate) * (x @ up) * (R_g @ X_g),   out += a @ down
where R_g [rows, G] are the routing weights of the group's experts and X_g [G, Np] the 0/1 matrix that spreads
expert j's weight over its I columns. The sum over the group's experts is the down matmul's K loop.
Routing weights: softmax over the logits >= the k-th largest one (== softmax over all, renormalized over top k).
TSUGUMI_MOE_ROUTED=0 skips the routed experts (shared expert + router only)."""
import torch
import ttnn

from .env import env
from .ops import DecodeLinear, has_cache

_L1 = ttnn.L1_MEMORY_CONFIG
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def make_mlp(dev, sd, prefix, H, tc, prec, cache_prefix):
    """The MLP of a decoder layer: MoEDecode when the checkpoint has a router, else the dense MLPDecode."""
    from .model import MLPDecode
    if prefix + "gate.weight" in sd:
        return MoEDecode(dev, sd, prefix, H, tc, prec, cache_prefix)
    return MLPDecode(dev, sd, prefix, H, tc["intermediate_size"], prec, cache_prefix)


class MoEDecode:
    def __init__(self, dev, sd, prefix, H, tc, prec, cache_prefix):
        from .model import MLPDecode
        self.dev, self.H = dev, H
        self.E, self.k = tc["num_experts"], tc["num_experts_per_tok"]
        I = self.I = tc["moe_intermediate_size"]
        self.shared = MLPDecode(dev, sd, prefix + "shared_expert.", H, tc["shared_expert_intermediate_size"], prec,
                                cache_prefix + ".shared")
        # router logits and the shared expert's gate logit in one matmul: [H, E + 1] (bfp8: the choice of experts is
        # discrete, so the router is kept finer than the bfp4 weights)
        cf = cache_prefix + ".router"
        rdt = ttnn.bfloat8_b
        w = None
        if not has_cache(cf, rdt):
            w = torch.cat([sd[prefix + "gate.weight"], sd[prefix + "shared_expert_gate.weight"]], 0).T.contiguous()
            w = w.to(torch.bfloat16)
        self.router = DecodeLinear(dev, w, rdt, cache_file=cf, shape=(H, self.E + 1), name="moe_router")
        self.routed = env("MOE_ROUTED", "1") == "1"
        self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                   fp32_dest_acc_en=True, packer_l1_acc=True)
        self.groups = []
        if not self.routed:
            return
        G = self.G = int(env("MOE_GROUP", "32"))
        gu_dt, dn_dt = prec["mlp_gu"], prec["mlp_down"]
        gu = dn = None
        for g in range(self.E // G):
            cg = f"{cache_prefix}.ex{G}.{g}"
            need = not all(has_cache(f"{cg}.{s}", d) for s, d in (("gate", gu_dt), ("up", gu_dt), ("down", dn_dt)))
            wg = wu = wd = None
            if need:
                if gu is None:
                    gu = sd[prefix + "experts.gate_up_proj"]  # [E, 2I, H]
                    dn = sd[prefix + "experts.down_proj"]  # [E, H, I]
                blk = gu[g * G:(g + 1) * G]  # [G, 2I, H]
                wg = blk[:, :I].permute(2, 0, 1).reshape(H, G * I).contiguous().to(torch.bfloat16)
                wu = blk[:, I:].permute(2, 0, 1).reshape(H, G * I).contiguous().to(torch.bfloat16)
            lg = DecodeLinear(dev, wg, gu_dt, cache_file=f"{cg}.gate", shape=(H, G * I), name="moe_gate")
            lu = DecodeLinear(dev, wu, gu_dt, cache_file=f"{cg}.up", shape=(H, G * I), name="moe_up")
            Kp = lg.Np
            if need:
                wd = dn[g * G:(g + 1) * G].permute(0, 2, 1).reshape(G * I, H).to(torch.bfloat16)
                wd = torch.nn.functional.pad(wd, (0, 0, 0, Kp - G * I))
            ld = DecodeLinear(dev, wd, dn_dt, cache_file=f"{cg}.down", shape=(Kp, H), name="moe_down")
            del wg, wu, wd
            self.groups.append((lg, lu, ld))
        del gu, dn
        X = torch.zeros(1, 1, 32, Kp)
        for j in range(G):
            X[0, 0, j, j * I:(j + 1) * I] = 1.0
        self.X = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # rows >= G zero

    # ---- routing: [rows, E] weights, zero outside each row's top k ----
    def _route(self, logits, mem):
        val, _ = ttnn.topk(logits, k=self.k, dim=-1, largest=True, sorted=True)
        kth = ttnn.slice(val, (0, 0, 0, self.k - 1), (1, 1, val.shape[-2], self.k), memory_config=mem)
        ttnn.deallocate(val)
        keep = ttnn.ge(logits, kth, memory_config=mem)  # 1.0 / 0.0
        ttnn.deallocate(kth)
        # masked logits: logits - 1e4 * (1 - keep)
        pen = ttnn.multiply(ttnn.rsub(keep, 1.0, memory_config=mem), -1e4, memory_config=mem)
        ttnn.deallocate(keep)
        ml = ttnn.add(logits, pen, memory_config=mem)
        ttnn.deallocate(pen)
        R = ttnn.softmax(ml, dim=-1, memory_config=mem, compute_kernel_config=self.ck)
        ttnn.deallocate(ml)
        return R

    def _spread(self, R, g, rows, mem):
        """[rows, Np]: routing weight of group g's expert j over its I columns (0 in the padding)."""
        G = self.G
        Rg = ttnn.slice(R, (0, 0, 0, g * G), (1, 1, rows, (g + 1) * G), memory_config=mem)
        if G < 32:
            Rg = ttnn.pad(Rg, [(0, 0), (0, 0), (0, 0), (0, 32 - G)], 0.0)
        S = ttnn.matmul(Rg, self.X, memory_config=mem, compute_kernel_config=self.ck)
        ttnn.deallocate(Rg)
        return S

    def __call__(self, x):
        """T-row decode: x [1,1,32,H] (residual layout) -> [1,1,32,Np(H)] L1."""
        r = ttnn.to_memory_config(self.router(x, out_mem=None), _L1)  # [1,1,32,Np]: E logits, shared gate, pad
        logits = ttnn.slice(r, (0, 0, 0, 0), (1, 1, 32, self.E), memory_config=_L1)
        sg = ttnn.slice(r, (0, 0, 0, self.E), (1, 1, 32, self.E + 1), memory_config=_L1)
        ttnn.deallocate(r)
        R = self._route(logits, _L1)
        ttnn.deallocate(logits)
        s = self.shared(x)  # [1,1,32,Np(H)] L1
        if s.shape[-1] != self.H:
            s2 = ttnn.slice(s, (0, 0, 0, 0), (1, 1, 32, self.H), memory_config=_L1)
            ttnn.deallocate(s)
            s = s2
        gt = ttnn.sigmoid(sg, memory_config=_L1)
        ttnn.deallocate(sg)
        out = ttnn.multiply(s, gt, memory_config=_L1)
        ttnn.deallocate(s)
        ttnn.deallocate(gt)
        if not self.groups:
            ttnn.deallocate(R)
            return out
        for g, (lg, lu, ld) in enumerate(self.groups):
            a = lg(x, out_mem=_L1)
            u = lu(x, out_mem=_L1)
            au = ttnn.multiply(a, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=_L1)
            ttnn.deallocate(a)
            ttnn.deallocate(u)
            S = self._spread(R, g, 32, _L1)
            w = ttnn.multiply(au, S, memory_config=_L1)
            ttnn.deallocate(au)
            ttnn.deallocate(S)
            d = ld(w, out_mem=_L1)
            ttnn.deallocate(w)
            if d.shape[-1] != out.shape[-1]:  # padded N of the shared and routed down projections can differ
                d2 = ttnn.slice(d, (0, 0, 0, 0), (1, 1, 32, out.shape[-1]), memory_config=_L1)
                ttnn.deallocate(d)
                d = d2
            out2 = ttnn.add(out, d, memory_config=_L1)
            ttnn.deallocate(out)
            ttnn.deallocate(d)
            out = out2
        ttnn.deallocate(R)
        return out

    def prefill(self, h, pf):
        """Large-T prefill: h [1,1,T,H] DRAM -> [1,1,T,H] DRAM (pf: the LargePrefill, for its matmul helpers)."""
        T, H = h.shape[-2], self.H
        r = pf._mm(h, self.router, n=False)
        logits = ttnn.slice(r, (0, 0, 0, 0), (1, 1, T, self.E), memory_config=_DRAM)
        sg = ttnn.slice(r, (0, 0, 0, self.E), (1, 1, T, self.E + 1), memory_config=_DRAM)
        ttnn.deallocate(r)
        R = self._route(logits, _DRAM)
        ttnn.deallocate(logits)
        s = pf._mlp(self.shared, h)  # [1,1,T,H]
        gt = ttnn.sigmoid(sg, memory_config=_DRAM)
        ttnn.deallocate(sg)
        out = ttnn.multiply(s, gt, memory_config=_DRAM)
        ttnn.deallocate(s)
        ttnn.deallocate(gt)
        for g, (lg, lu, ld) in enumerate(self.groups):
            a = pf._mm(h, lg, n=False)
            u = pf._mm(h, lu, n=False)
            au = ttnn.multiply(a, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=_DRAM)
            ttnn.deallocate(a)
            ttnn.deallocate(u)
            S = self._spread(R, g, T, _DRAM)
            w = ttnn.multiply(au, S, memory_config=_DRAM)
            ttnn.deallocate(au)
            ttnn.deallocate(S)
            d = pf._mm(w, ld, n=H)
            ttnn.deallocate(w)
            out2 = ttnn.add(out, d, memory_config=_DRAM)
            ttnn.deallocate(out)
            ttnn.deallocate(d)
            out = out2
        ttnn.deallocate(R)
        return out
