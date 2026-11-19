"""Compare engine logits against HF (CPU fp32) for the first N layers (+final norm + lm_head).

python cmp_hf.py --layers 4 [--prec mixed]
"""
import argparse
import json
import sys

import torch
import ttnn

from qe.model import LM_PREFIX, LazySD, Precision, Qwen35Engine
import qe.paths as qe_paths

MODEL = qe_paths.MODEL
CACHE = qe_paths.CACHE


def hf_logits(n_layers, ids):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL).text_config
    cfg.num_hidden_layers = n_layers
    cfg.layer_types = cfg.layer_types[:n_layers]
    cfg._attn_implementation = "eager"
    from transformers.initialization import no_init_weights  # noqa
    with no_init_weights():
        m = Qwen3_5ForCausalLM(cfg)
    sd = LazySD(MODEL)
    new = {}
    for k in m.state_dict().keys():
        src = k.replace("model.", LM_PREFIX, 1) if k.startswith("model.") else k
        new[k] = sd[src].float()
    m.load_state_dict(new, assign=True, strict=True)
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor([ids]), use_cache=False)
    return out.logits[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--prec", default="mixed")
    ap.add_argument("--T", type=int, default=0, help="T-row mode (chunked) instead of T=1 decode")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is")["input_ids"]
    ref = hf_logits(a.layers, ids)
    print("HF done", ref.shape, flush=True)
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision(a.prec), f"{CACHE}/{a.prec}", max_seq=1024, n_layers=a.layers,
                           log=lambda *x: None)
        eng.reset()
        agree = 0
        if a.T:
            eng.init_spec(a.T)
            eng.reset_spec()
            rows = []
            a_prev, pos = 0, 0
            while pos < len(ids):
                ch = ids[pos:pos + a.T]
                eng.run_spec(ch, pos, a_prev)
                lg_all = ttnn.to_torch(eng.last_spec[0])[0, 0, :len(ch), : eng.V].float()
                rows += [lg_all[i] for i in range(len(ch))]
                a_prev = len(ch)
                pos += len(ch)
        for p, t in enumerate(ids):
            lg = rows[p] if a.T else eng.step(t, p)
            r = ref[p]
            pcc = torch.corrcoef(torch.stack([lg, r]))[0, 1].item()
            agree += int(lg.argmax() == r.argmax())
            if p < 3 or p == len(ids) - 1 or p % 5 == 0:
                print(f"pos {p:3d} pcc {pcc:.5f} top1 tt={tok.decode([int(lg.argmax())])!r} hf={tok.decode([int(r.argmax())])!r}")
        print(f"top1 agreement {agree}/{len(ids)}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
