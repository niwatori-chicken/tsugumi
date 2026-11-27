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
        self.impl = env("MOE_IMPL", "C")
        if self.impl == "C":
            self._init_c(dev, sd, prefix, H, prec, cache_prefix)
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

    # ---- method C: rows sorted by expert, grouped GEMM (tsugumi/moe_gemm.py) ----
    def _init_c(self, dev, sd, prefix, H, prec, cache_prefix):
        from .ops import load_or_convert
        from .moe_gemm import MoEGemm
        E, I = self.E, self.I
        gu = dn = None
        cg, cd = f"{cache_prefix}.exC.gu", f"{cache_prefix}.exC.dn"
        dt = ttnn.bfloat4_b
        w = None
        if not has_cache(cg, dt):
            gu = sd[prefix + "experts.gate_up_proj"]  # [E, 2I, H]: gate rows then up rows
            w = gu.transpose(1, 2).reshape(1, 1, E * H, 2 * I).contiguous().to(torch.bfloat16)
        self.w_gu = load_or_convert(dev, w, dt, _DRAM, cg)
        del w, gu
        w = None
        if not has_cache(cd, dt):
            dn = sd[prefix + "experts.down_proj"]  # [E, H, I]
            w = dn.transpose(1, 2).reshape(1, 1, E * I, H).contiguous().to(torch.bfloat16)
        self.w_dn = load_or_convert(dev, w, dt, _DRAM, cd)
        del w, dn
        self.mm_gu = MoEGemm(dev, H, 2 * I, NN=int(env("MOE_NN_GU", "16")), RT=1, CT=4)
        self.mm_dn = MoEGemm(dev, I, H, NN=int(env("MOE_NN_DN", "32")), RT=1, CT=4)
        self.zrows = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev)

    def _routed_c(self, h, logits):
        """h [1,1,T,H] DRAM TILE, logits [1,1,T,E] -> routed experts' output [1,1,T,H] DRAM (host round trip for
        the schedule)."""
        import numpy as np
        T, H, E, k, I = h.shape[-2], self.H, self.E, self.k, self.I
        val, idx = ttnn.topk(logits, k=k, dim=-1, largest=True, sorted=True)
        w = ttnn.softmax(val, dim=-1, compute_kernel_config=self.ck)
        ttnn.deallocate(val)
        ix = ttnn.to_torch(idx).reshape(T, k).to(torch.int64).numpy()
        ttnn.deallocate(idx)
        # schedule: rows sorted by expert, each expert's rows padded to whole row tiles
        ef = ix.reshape(-1)
        order = np.argsort(ef, kind="stable")
        cnt = np.bincount(ef, minlength=E)
        tiles = (cnt + 31) // 32
        toff = np.concatenate([[0], np.cumsum(tiles)[:-1]])
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        rank = np.arange(len(ef)) - start[ef[order]]
        pos = np.empty(len(ef), dtype=np.int64)
        pos[order] = toff[ef[order]] * 32 + rank
        Rm = -(-(T * k + 31 * E) // 32) * 32  # fixed per T: every row count fits
        src = np.full(Rm, T, dtype=np.int32)  # rows not used: a zero row (index T)
        src[pos] = np.repeat(np.arange(T), k)
        up = lambda a, dt, lay: ttnn.from_torch(torch.from_numpy(a), dtype=dt, layout=lay, device=self.dev)
        src_t = up(src.reshape(1, Rm), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pos_t = up(pos.reshape(T, k).T.copy().astype(np.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)  # [k, T]
        units = self.mm_gu.units(tiles, toff)
        # gather the rows: X[r] = h[src[r]]
        hz = ttnn.concat([h, self.zrows], dim=2, memory_config=_DRAM)
        hz_rm = ttnn.reshape(ttnn.to_layout(hz, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM), (T + 32, H))
        ttnn.deallocate(hz)
        X = ttnn.embedding(src_t, hz_rm, layout=ttnn.TILE_LAYOUT, memory_config=_DRAM)
        ttnn.deallocate(hz_rm)
        ttnn.deallocate(src_t)
        X = ttnn.reshape(X, (1, 1, Rm, H))
        al = lambda: ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Rm, I]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                    self.dev, _DRAM)
        g, u = al(), al()
        self.mm_gu(X, self.w_gu, g, units, Y2=u)  # gate and up halves straight into two tensors
        ttnn.deallocate(X)
        a2 = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=_DRAM)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        Y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, Rm, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.dev, _DRAM)
        self.mm_dn(a2, self.w_dn, Y, self.mm_dn.units(tiles, toff))
        ttnn.deallocate(a2)
        # combine: out[t] = sum_j Y[pos[t, j]]
        Yr = ttnn.reshape(ttnn.to_layout(Y, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM), (Rm, H))
        ttnn.deallocate(Y)
        O = ttnn.embedding(pos_t, Yr, layout=ttnn.TILE_LAYOUT, memory_config=_DRAM)  # [k, T, H]
        ttnn.deallocate(Yr)
        ttnn.deallocate(pos_t)
        O = ttnn.reshape(O, (1, k, T, H))
        # routing weights [T, k] -> [1, k, T, 1], applied per gathered row
        wT = ttnn.reshape(ttnn.to_layout(ttnn.transpose(w, -2, -1), ttnn.ROW_MAJOR_LAYOUT), (1, k, T, 1))
        wT = ttnn.to_layout(wT, ttnn.TILE_LAYOUT)
        ttnn.deallocate(w)
        Ow = ttnn.multiply(O, wT, memory_config=_DRAM)
        ttnn.deallocate(O)
        ttnn.deallocate(wT)
        out = ttnn.sum(Ow, dim=1, keepdim=True, memory_config=_DRAM)
        ttnn.deallocate(Ow)
        return out

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
        if self.routed and self.impl == "C":
            return self._call_c(x, logits, sg)
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

    def _call_c(self, x, logits, sg):
        """T-row decode with method C (eager: the schedule needs a host round trip; no trace)."""
        s = self.shared(x)
        if s.shape[-1] != self.H:
            s2 = ttnn.slice(s, (0, 0, 0, 0), (1, 1, 32, self.H), memory_config=_L1)
            ttnn.deallocate(s)
            s = s2
        gt = ttnn.sigmoid(sg, memory_config=_L1)
        ttnn.deallocate(sg)
        out = ttnn.multiply(s, gt, memory_config=_L1)
        ttnn.deallocate(s)
        ttnn.deallocate(gt)
        h = ttnn.to_memory_config(x, _DRAM)
        lg = ttnn.to_memory_config(logits, _DRAM)
        ttnn.deallocate(logits)
        r = self._routed_c(h, lg)
        ttnn.deallocate(h)
        ttnn.deallocate(lg)
        o2 = ttnn.add(out, ttnn.to_memory_config(r, _L1), memory_config=_L1)
        ttnn.deallocate(out)
        ttnn.deallocate(r)
        return o2

    def prefill(self, h, pf):
        """Large-T prefill: h [1,1,T,H] DRAM -> [1,1,T,H] DRAM (pf: the LargePrefill, for its matmul helpers)."""
        T, H = h.shape[-2], self.H
        r = pf._mm(h, self.router, n=False)
        logits = ttnn.slice(r, (0, 0, 0, 0), (1, 1, T, self.E), memory_config=_DRAM)
        sg = ttnn.slice(r, (0, 0, 0, self.E), (1, 1, T, self.E + 1), memory_config=_DRAM)
        ttnn.deallocate(r)
        if self.routed and self.impl == "C":
            s = pf._mlp(self.shared, h)
            gt = ttnn.sigmoid(sg, memory_config=_DRAM)
            ttnn.deallocate(sg)
            out = ttnn.multiply(s, gt, memory_config=_DRAM)
            ttnn.deallocate(s)
            ttnn.deallocate(gt)
            rr = self._routed_c(h, logits)
            ttnn.deallocate(logits)
            o2 = ttnn.add(out, rr, memory_config=_DRAM)
            ttnn.deallocate(out)
            ttnn.deallocate(rr)
            return o2
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
