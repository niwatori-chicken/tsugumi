"""Sparse MoE block of Qwen3.5/3.6-MoE (e.g. Qwen3.6-35B-A3B) for the T-row decode path. Work in progress (branch
moe-probe, doc/12): the shared expert, its sigmoid gate and the router (top-k + softmax over the chosen logits) run
on device; the routed experts are not wired yet (TSUGUMI_MOE_ROUTED=0 measures everything but them)."""
import torch
import ttnn

from .env import env
from .ops import DecodeLinear, has_cache

_L1 = ttnn.L1_MEMORY_CONFIG


def make_mlp(dev, sd, prefix, H, tc, prec, cache_prefix):
    """The MLP of a decoder layer: MoEDecode when the checkpoint has a router, else the dense MLPDecode."""
    from .model import MLPDecode
    if prefix + "gate.weight" in sd:
        return MoEDecode(dev, sd, prefix, H, tc, prec, cache_prefix)
    return MLPDecode(dev, sd, prefix, H, tc["intermediate_size"], prec, cache_prefix)


class MoEDecode:
    def __init__(self, dev, sd, prefix, H, tc, prec, cache_prefix):
        from .model import MLPDecode
        self.E, self.k = tc["num_experts"], tc["num_experts_per_tok"]
        self.shared = MLPDecode(dev, sd, prefix + "shared_expert.", H, tc["shared_expert_intermediate_size"], prec,
                                cache_prefix + ".shared")
        # router logits and the shared expert's gate logit in one matmul: [H, E + 1] (bfp8: the choice of experts
        # is discrete, so the router is kept finer than the bfp4 weights)
        cf = cache_prefix + ".router"
        rdt = ttnn.bfloat8_b
        w = None
        if not has_cache(cf, rdt):
            w = torch.cat([sd[prefix + "gate.weight"], sd[prefix + "shared_expert_gate.weight"]], 0).T.contiguous()
            w = w.to(torch.bfloat16)
        self.router = DecodeLinear(dev, w, rdt, cache_file=cf, shape=(H, self.E + 1), name="moe_router")
        self.routed = env("MOE_ROUTED", "0") == "1"

    def __call__(self, x):
        r = ttnn.to_memory_config(self.router(x, out_mem=None), _L1)  # [1,1,32,Np]: E logits, shared gate, pad
        logits = ttnn.slice(r, (0, 0, 0, 0), (1, 1, 32, self.E), memory_config=_L1)
        sg = ttnn.slice(r, (0, 0, 0, self.E), (1, 1, 32, self.E + 1), memory_config=_L1)
        ttnn.deallocate(r)
        val, idx = ttnn.topk(logits, k=self.k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(logits)
        # softmax over all E then renormalized over the top k == softmax over the top-k logits
        wts = ttnn.softmax(val, dim=-1)
        ttnn.deallocate(val)
        s = self.shared(x)  # [1,1,32,Np(H)] L1
        g = ttnn.sigmoid(sg, memory_config=_L1)
        ttnn.deallocate(sg)
        out = ttnn.multiply(s, g, memory_config=_L1)
        ttnn.deallocate(s)
        ttnn.deallocate(g)
        if self.routed:
            raise NotImplementedError("routed experts (doc/12 H3)")
        ttnn.deallocate(wts)
        ttnn.deallocate(idx)
        return out
