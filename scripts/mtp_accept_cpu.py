"""MTP draft acceptance of a Qwen3.5-family model, estimated on CPU with teacher forcing (doc/12): how many of k
greedy MTP drafts the main model would accept per step. Works for the MoE (Qwen3.6-35B-A3B) and, as a calibration
against the tsugumi engine's measured acceptance, for the dense Qwen3.8-27B.

  python scripts/mtp_accept_cpu.py collect MODEL_DIR OUT.pt   (main model, one layer at a time; saves final normed
                                                               hidden + greedy prediction per position)
  python scripts/mtp_accept_cpu.py mtp MODEL_DIR OUT.pt [K=3]  (MTP chain of depth K over the saved hiddens)
Text: /mnt/HDD/eval/decode_seqs.pt.

Estimator: a step starts at position t where the text continues greedily (x[t+1] == main argmax at t). Draft i
(i = 1..K) is the MTP's argmax after i chained MTP passes; it is accepted when it equals the main argmax at t+i and
all earlier drafts were accepted. Inputs of draft i are the text token x[t+i] (= the previous draft when accepted)
and the previous MTP output hidden. Approximation: in pass i, the MTP attention sees the pass-i entries of earlier
positions (the engine sees pass-1 entries of the verified prefix), so deeper drafts are slightly off."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import json
import sys
import time

import torch

SEQS = "/mnt/HDD/eval/decode_seqs.pt"


def family(model_dir):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg._attn_implementation = "sdpa"
    if "moe" in cfg.model_type:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
        P = "Qwen3_5Moe"
    else:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        P = "Qwen3_5"
    return cfg, m, P


def loader(model_dir):
    from safetensors import safe_open
    wmap = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]

    def tensors(prefix):
        sd = {}
        for f in sorted({f for k, f in wmap.items() if k.startswith(prefix)}):
            with safe_open(f"{model_dir}/{f}", "pt") as h:
                for k in h.keys():
                    if k.startswith(prefix):
                        sd[k[len(prefix):]] = h.get_tensor(k)
        return sd
    return tensors


def load_seqs():
    seqs = torch.load(SEQS)
    names = list(seqs)
    return names, [torch.tensor(seqs[n][0] + seqs[n][1]) for n in names], [len(seqs[n][0]) for n in names]


def rope(m, P, cfg, T, offset=0):
    rot = getattr(m, P + "TextRotaryEmbedding")(cfg)
    pid = torch.arange(T)[None] + offset
    return rot(torch.zeros(1, 1, cfg.hidden_size), pid[None].expand(3, 1, T)), pid


def layer_of(m, P, cfg, i, sd):
    layer = getattr(m, P + "DecoderLayer")(cfg, i)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)
    return layer.float().eval()


def run_layer(layer, h, pe, pid):
    o = layer(h, position_embeddings=pe, attention_mask=None, position_ids=pid)
    return o[0] if isinstance(o, tuple) else o


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())


def argmax_lm(h, lm):
    return torch.cat([(h[j:j + 256].bfloat16() @ lm.T).argmax(-1) for j in range(0, len(h), 256)])


def collect(model_dir, out):
    torch.set_num_threads(_os.cpu_count())
    cfg, m, P = family(model_dir)
    tensors = loader(model_dir)
    pre = "model.language_model."
    names, ids, n_prompt = load_seqs()
    emb = tensors(pre + "embed_tokens.")["weight"]
    hs = [emb[x].float()[None] for x in ids]
    del emb
    ropes = [rope(m, P, cfg, len(x)) for x in ids]
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = layer_of(m, P, cfg, i, tensors(f"{pre}layers.{i}."))
        with torch.no_grad():
            for s in range(len(names)):
                hs[s] = run_layer(layer, hs[s], *ropes[s])
        del layer
        print(f"layer {i:2d}: {time.time() - t0:.0f}s", flush=True)
    nw = tensors(pre + "norm.")["weight"]
    lm = tensors("lm_head.")["weight"]
    hid, pred = [], []
    with torch.no_grad():
        for s, name in enumerate(names):
            h = rms(hs[s][0], nw, cfg.rms_norm_eps)
            p = argmax_lm(h, lm)
            hid.append(h.bfloat16())
            pred.append(p)
            g = slice(n_prompt[s] - 1, len(ids[s]) - 1)
            print(f"  {name:12s} main argmax == text continuation: {(p[g] == ids[s][g.start + 1:]).float().mean():.1%}",
                  flush=True)
    torch.save({"names": names, "ids": ids, "n_prompt": n_prompt, "hidden": hid, "pred": pred}, out)
    print("saved", out)


def mtp(model_dir, path, K=3):
    torch.set_num_threads(_os.cpu_count())
    cfg, m, P = family(model_dir)
    tensors = loader(model_dir)
    d = torch.load(path)
    names, ids, n_prompt, hid, pred = d["names"], d["ids"], d["n_prompt"], d["hidden"], d["pred"]
    t = tensors("mtp.")
    fc = t["fc.weight"].float()
    n_emb, n_hid, n_fin = t["pre_fc_norm_embedding.weight"], t["pre_fc_norm_hidden.weight"], t["norm.weight"]
    full = cfg.layer_types.index("full_attention")
    layer = layer_of(m, P, cfg, full, {k[len("layers.0."):]: v for k, v in t.items() if k.startswith("layers.0.")})
    emb = tensors("model.language_model.embed_tokens.")["weight"]
    lm = tensors("lm_head.")["weight"]
    eps = cfg.rms_norm_eps
    tot = {"steps": 0, "acc": 0, "pos": [[0, 0] for _ in range(K)]}
    print(f"{'':12s} {'steps':>6s} {'drafts/step':>11s}  per-draft acceptance (conditional)")
    for s, name in enumerate(names):
        x, N, g0 = ids[s], len(ids[s]), n_prompt[s] - 1
        h_in = hid[s].float()  # [N, H]: main hidden at position t (predicts x[t+1])
        drafts = []
        with torch.no_grad():
            for i in range(1, K + 1):
                # pass i: row t takes hidden (main for i=1, previous MTP output otherwise) and token x[t+i]
                n = N - i
                tok = x[i:i + n]
                inp = torch.cat([rms(emb[tok].float(), n_emb, eps), rms(h_in[:n], n_hid, eps)], -1) @ fc.T
                pe, pid = rope(m, P, cfg, n, offset=i)
                o = run_layer(layer, inp[None], pe, pid)[0]
                h_in = rms(o, n_fin, eps)
                drafts.append(argmax_lm(h_in, lm))  # row t: draft for position t+i+1
        steps, acc = 0, 0
        pos = [[0, 0] for _ in range(K)]
        for t0 in range(g0, N - 1 - K):
            if x[t0 + 1] != pred[s][t0]:
                continue  # the text left the greedy path here: not a step start
            steps += 1
            for i in range(1, K + 1):
                pos[i - 1][1] += 1
                if drafts[i - 1][t0] != pred[s][t0 + i]:
                    break
                pos[i - 1][0] += 1
                acc += 1
                if x[t0 + i + 1] != pred[s][t0 + i]:
                    break  # text no longer greedy: later drafts were fed text, not the accepted draft
        tot["steps"] += steps
        tot["acc"] += acc
        for i in range(K):
            tot["pos"][i][0] += pos[i][0]
            tot["pos"][i][1] += pos[i][1]
        print(f"{name:12s} {steps:6d} {acc / max(steps, 1):11.2f}  " +
              " / ".join(f"{a / max(b, 1):.2f}" for a, b in pos), flush=True)
    print(f"{'all':12s} {tot['steps']:6d} {tot['acc'] / tot['steps']:11.2f}  " +
          " / ".join(f"{a / max(b, 1):.2f}" for a, b in tot["pos"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        mtp(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
