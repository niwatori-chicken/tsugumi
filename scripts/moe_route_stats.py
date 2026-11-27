"""Which experts a MoE model routes to on real text (doc/12 H4/H5): how fast the union of experts grows over T
consecutive tokens (speculative verification), across B unrelated requests (batch decode), and how uneven the
per-expert row counts are in a prefill chunk.

  python scripts/moe_route_stats.py collect [MODEL_DIR] [OUT.pt]   (CPU, one layer at a time: fits in 31 GB RAM)
  python scripts/moe_route_stats.py analyze [OUT.pt]
Text: /mnt/HDD/eval/decode_seqs.pt (prompt + Qwen3.8-27B continuation per category; same tokenizer family)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import json
import math
import sys
import time

import torch

MODEL = "/mnt/HDD/Qwen36_35BA3B"
OUT = "/mnt/HDD/eval/moe_routes_q36.pt"
SEQS = "/mnt/HDD/eval/decode_seqs.pt"


def collect(model_dir, out):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m

    torch.set_num_threads(_os.cpu_count())
    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg._attn_implementation = "sdpa"
    wmap = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    pre = "model.language_model."

    def tensors(prefix):
        files = sorted({f for k, f in wmap.items() if k.startswith(prefix)})
        sd = {}
        for f in files:
            with safe_open(f"{model_dir}/{f}", "pt") as h:
                for k in h.keys():
                    if k.startswith(prefix):
                        sd[k[len(prefix):]] = h.get_tensor(k)
        return sd

    seqs = torch.load(SEQS)
    names = list(seqs)
    ids = [torch.tensor(seqs[n][0] + seqs[n][1]) for n in names]
    n_prompt = [len(seqs[n][0]) for n in names]
    emb = tensors(pre + "embed_tokens.")["weight"]
    hs = [emb[x].float()[None] for x in ids]
    del emb
    rot = m.Qwen3_5MoeTextRotaryEmbedding(cfg)
    pes, pids = [], []
    for x in ids:
        T = len(x)
        pid = torch.arange(T)[None]
        pes.append(rot(hs[0], pid[None].expand(3, 1, T)))
        pids.append(pid)
    routes = [[] for _ in names]  # [seq][layer] -> int16 [T, k]
    logits_top = [[] for _ in names]  # router probs of the chosen experts, fp16 [T, k]
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = m.Qwen3_5MoeDecoderLayer(cfg, i)
        missing, unexpected = layer.load_state_dict(tensors(f"{pre}layers.{i}."), strict=False)
        assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)
        layer = layer.float().eval()
        cap = {}
        orig = layer.mlp.gate.forward

        def hook(h, _orig=orig):
            lg, sc, ix = _orig(h)
            cap["ix"], cap["sc"] = ix, sc
            return lg, sc, ix

        layer.mlp.gate.forward = hook
        with torch.no_grad():
            for s in range(len(names)):
                hs[s] = layer(hs[s], position_embeddings=pes[s], attention_mask=None, position_ids=pids[s])
                if isinstance(hs[s], tuple):
                    hs[s] = hs[s][0]
                routes[s].append(cap["ix"].to(torch.int16))
                logits_top[s].append(cap["sc"].to(torch.float16))
        del layer
        print(f"layer {i:2d} ({cfg.layer_types[i]}): {time.time() - t0:.0f}s", flush=True)
    # sanity check of the CPU forward: next-token top-1 agreement with the Qwen3.8-27B continuation
    nw = tensors(pre + "norm.")["weight"].float()
    lm = tensors("lm_head.")["weight"]
    agree = []
    with torch.no_grad():
        for s, name in enumerate(names):
            h = hs[s][0, n_prompt[s] - 1:-1]
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1 + nw)
            pred = torch.cat([(h[j:j + 256].bfloat16() @ lm.T).argmax(-1) for j in range(0, len(h), 256)])
            a = (pred == ids[s][n_prompt[s]:]).float().mean().item()
            agree.append(a)
            print(f"  {name:12s} top-1 agreement with the continuation {a:.1%}", flush=True)
    torch.save({"agree": agree, "names": names, "n_prompt": n_prompt, "E": cfg.num_experts, "k": cfg.num_experts_per_tok,
                "routes": [torch.stack(r, 0) for r in routes],  # [L, T, k]
                "scores": [torch.stack(r, 0) for r in logits_top]}, out)
    print("saved", out)


def union_mean(r, T):
    """r [L, N, k] -> mean over layers and windows of |union of experts over T consecutive tokens|."""
    L, N, k = r.shape
    if N < T:
        return float("nan")
    oh = torch.zeros(L, N, int(r.max()) + 1, dtype=torch.int32)
    oh.scatter_(2, r.long(), 1)
    cs = torch.cat([torch.zeros(L, 1, oh.shape[2], dtype=torch.int32), oh.cumsum(1)], 1)
    win = cs[:, T:] - cs[:, :-T]  # [L, N-T+1, E] count per expert in each window
    return (win > 0).sum(-1).float().mean().item()


def analyze(path):
    d = torch.load(path)
    E, k = d["E"], d["k"]
    indep = lambda T: E * (1 - (1 - k / E) ** T)
    Ts = [1, 2, 4, 8, 16]
    print(f"E={E} k={k}. Mean number of distinct experts per layer over T consecutive tokens (generated part):")
    print(f"{'':12s}" + "".join(f"{'T=' + str(T):>8s}" for T in Ts))
    print(f"{'independent':12s}" + "".join(f"{indep(T):8.1f}" for T in Ts))
    gen = []
    agg = {T: [] for T in Ts}
    for name, p, r in zip(d["names"], d["n_prompt"], d["routes"]):
        g = r[:, p:]
        gen.append(g)
        row = []
        for T in Ts:
            u = union_mean(g, T)
            row.append(u)
            agg[T].append((u, g.shape[1]))
        print(f"{name:12s}" + "".join(f"{u:8.1f}" for u in row))
    allm = {T: sum(u * n for u, n in agg[T]) / sum(n for _, n in agg[T]) for T in Ts}
    print(f"{'all (wtd)':12s}" + "".join(f"{allm[T]:8.1f}" for T in Ts))
    print("ratio to T=1:" + "".join(f"{allm[T] / allm[1]:8.2f}" for T in Ts))
    # per layer, T=4
    L = gen[0].shape[0]
    per_layer = [sum(union_mean(g[l:l + 1], 4) * g.shape[1] for g in gen) / sum(g.shape[1] for g in gen) for l in range(L)]
    print("T=4 union per layer:", " ".join(f"{u:.0f}" for u in per_layer))
    # batch: B tokens from B different sequences at random positions
    torch.manual_seed(0)
    print("batch (B unrelated requests, one token each):" + "".join(
        f"  B={B}: {sum(((torch.zeros(L, E, dtype=torch.int32).scatter_(1, torch.cat([g[:, torch.randint(g.shape[1], (1,))].squeeze(1) for g in [gen[j] for j in torch.randperm(len(gen))[:B].tolist()]], 1).long(), 1)) > 0).sum(-1).float().mean().item() for _ in range(200)) / 200:.1f}"
        for B in (2, 4, 8)))
    # prefill: per-expert rows in a chunk of the longest sequences (prompt + generation)
    print("prefill chunk (all tokens of a sequence): rows per expert, mean over layers")
    for name, r in zip(d["names"], d["routes"]):
        Lr, N, _ = r.shape
        if N < 512:
            continue
        cnt = torch.zeros(Lr, E).scatter_add_(1, r.reshape(Lr, -1).long(), torch.ones(Lr, N * k))
        mean = N * k / E
        print(f"  {name:12s} N={N:5d}: mean {mean:6.1f}, max {cnt.max(1).values.mean():6.1f} "
              f"({cnt.max(1).values.mean() / mean:.1f}x), p90 {cnt.quantile(0.9, dim=1).mean():6.1f}, "
              f"experts with 0 rows {(cnt == 0).sum(1).float().mean():5.1f}, "
              f"rows in the top 10% experts {cnt.sort(1, descending=True).values[:, :E // 10].sum(1).mean() / (N * k):.0%}")


if __name__ == "__main__":
    import os
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "collect":
        collect(sys.argv[2] if len(sys.argv) > 2 else MODEL, sys.argv[3] if len(sys.argv) > 3 else OUT)
    else:
        analyze(sys.argv[2] if len(sys.argv) > 2 else OUT)
