"""Reference perplexity on CPU (bf16 checkpoint computed in fp32), streaming one decoder layer at a time from
the safetensors (the full model does not fit in host RAM). Same protocol and outputs as eval_ppl.py.

    python scripts/ref_ppl.py IDS.pt OUT.pt [--sub 4] [--topk 64]

All chunks go through layer i before layer i+1 is loaded; hidden states (N chunks x C x 5120 fp32) stay in RAM.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import argparse
import math
import time

import torch

import tsumi.paths as tsumi_paths
from tsumi.model import LM_PREFIX, LazySD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--sub", type=int, default=4, help="chunks per layer call")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0, help="debug: only the first N layers")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoConfig
    from transformers.initialization import no_init_weights
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
                                                               Qwen3_5TextRotaryEmbedding)

    cfg = AutoConfig.from_pretrained(tsumi_paths.MODEL).text_config
    cfg._attn_implementation = "sdpa"
    sd = LazySD(tsumi_paths.MODEL)
    data = torch.load(a.ids)
    names = list(data)
    ids = torch.cat([data[n] for n in names], 0)  # [N, C]
    N, C = ids.shape
    emb = sd[LM_PREFIX + "embed_tokens.weight"]
    h = emb[ids].float()  # [N, C, H]
    del emb
    rot = Qwen3_5TextRotaryEmbedding(config=cfg)
    pos = torch.arange(C)[None].expand(a.sub, C)
    t_start = time.time()
    with torch.no_grad():
        for i in range(a.layers or cfg.num_hidden_layers):
            t0 = time.time()
            with no_init_weights():
                layer = Qwen3_5DecoderLayer(cfg, i)
            p = f"{LM_PREFIX}layers.{i}."
            layer.load_state_dict({k: sd[p + k].float() for k in layer.state_dict()}, assign=True, strict=True)
            layer.eval()
            t1 = time.time()
            for b in range(0, N, a.sub):
                x = h[b:b + a.sub]
                pp = pos[: x.shape[0]]
                pe = rot(x, pp[None].expand(3, -1, -1))
                h[b:b + a.sub] = layer(x, position_embeddings=pe, attention_mask=None, position_ids=pp,
                                       past_key_values=None)
            del layer
            print(f"layer {i} ({cfg.layer_types[i]}): load {t1 - t0:.0f}s, run {time.time() - t1:.0f}s, "
                  f"total {time.time() - t_start:.0f}s", flush=True)
        norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.load_state_dict({"weight": sd[LM_PREFIX + "norm.weight"].float()})
        W = sd["lm_head.weight"].float()  # [V, H]
        first = C // 2
        res, k = {}, 0
        for name in names:
            out = []
            for n in range(data[name].shape[0]):
                x = norm(h[k, first - 1: C - 1])
                lp = torch.log_softmax(x @ W.T, -1)
                tgt = ids[k, first:C]
                tl = lp.gather(1, tgt[:, None])[:, 0]
                tv, ti = lp.topk(a.topk, -1)
                out.append(dict(tgt=tl, top_v=tv, top_i=ti.int()))
                print(f"[{name} {n}] ppl {math.exp(-tl.mean()):.3f}", flush=True)
                k += 1
            allt = torch.cat([o["tgt"] for o in out])
            print(f"[{name}] PPL {math.exp(-allt.mean()):.4f} over {len(allt)} tokens", flush=True)
            res[name] = out
    torch.save(res, a.out)


if __name__ == "__main__":
    main()
