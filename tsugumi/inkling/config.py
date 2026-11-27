"""Inkling 2 Mini configuration (config.json `text_config`), with the derived values the engine needs."""
import json
import os
from dataclasses import dataclass, field


@dataclass
class InklingConfig:
    hidden: int = 2304
    n_layers: int = 24
    vocab: int = 201024
    vocab_unpadded: int = 200058
    n_heads: int = 24
    n_kv: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    window: int = 1024
    log_n_floor: int = 8192
    log_alpha: float = 0.1
    eps: float = 1e-6
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    local_layers: list = field(default_factory=list)
    conv_k: int = 4
    logit_softcap: float = 30.0
    block: int = 6  # n_layers_per_attn_res_block
    n_experts: int = 128
    top_k: int = 8
    inter: int = 1152
    gate_softcap: float = 4.0
    up_softcap: float = 25.0
    route_scale: float = 8.0
    loop_start: int = 6
    loop_end: int = 18
    passes: int = 3
    gate_rank: int = 64
    eos: int = 200006

    @classmethod
    def load(cls, model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            c = json.load(f)
        t = c.get("text_config", c)
        lc = t.get("loop_config", {})
        return cls(
            hidden=t["hidden_size"], n_layers=t["num_hidden_layers"], vocab=t["vocab_size"],
            vocab_unpadded=t["unpadded_vocab_size"], n_heads=t["num_attention_heads"],
            n_kv=t["num_key_value_heads"], head_dim=t["head_dim"], d_rel=t["d_rel"], rel_extent=t["rel_extent"],
            window=t["sliding_window_size"], log_n_floor=t["log_scaling_n_floor"],
            log_alpha=t["log_scaling_alpha"], eps=t["rms_norm_eps"], qk_clip=t["qk_l2_clip_threshold"],
            rel_clip=t["rel_l2_clip_threshold"], local_layers=list(t["local_layer_ids"]),
            conv_k=t["sconv_kernel_size"], logit_softcap=t["final_logit_softcapping"],
            block=t["n_layers_per_attn_res_block"], n_experts=t["n_routed_experts"],
            top_k=t["num_experts_per_tok"], inter=t["intermediate_size"], gate_softcap=t["mlp_gate_softcap"],
            up_softcap=t["mlp_up_softcap"], route_scale=t["route_scale"],
            loop_start=lc.get("core_start", 6), loop_end=lc.get("core_end", 18), passes=lc.get("num_passes", 3),
            gate_rank=lc.get("gate_rank", 64), eos=c.get("eos_token_id", t.get("eos_token_id", 200006)),
        )

    def sliding(self, i):
        return i in self.local_layers

    def is_loop(self, i):
        return self.loop_start <= i < self.loop_end

    def boundary(self, i):
        return i % self.block == 0

    @property
    def n_loop(self):
        return self.loop_end - self.loop_start

    def schedule(self):
        """The layer applications of one forward pass, in order: (app index, layer index, pass index).
        App index = the cache slot (KV, convolution states) of that application. Pass 0 for the layers outside
        the loop."""
        apps = []
        for i in range(self.loop_start):
            apps.append((i, 0))
        for p in range(self.passes):
            for i in range(self.loop_start, self.loop_end):
                apps.append((i, p))
        for i in range(self.loop_end, self.n_layers):
            apps.append((i, 0))
        return [(a, i, p) for a, (i, p) in enumerate(apps)]

    def tau(self, pos):
        """Log scaling of the full-attention queries / relative states at position pos."""
        import math
        return 1.0 + self.log_alpha * math.log(max((pos + 1) / self.log_n_floor, 1.0))
