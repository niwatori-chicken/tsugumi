"""How many tokens per step would the Inkling 2 DSpark draft (mtp.safetensors) give? Greedy, on real replies.

    python scripts/inkling_mtp_accept.py [--new 160] [--no-device]
1. greedy replies to a few chat prompts on the device (or loads them from the cache file with --no-device),
2. the CPU reference (tsugumi/inkling/ref.py) gives the target features of every position (the inputs of layer
   visits 5 / 23 / 41 and the final-norm output),
3. speculative decoding is simulated with the CPU draft (tsugumi/inkling/draft_ref.py): anchor = the target's token,
   7 drafts, accepted = the leading drafts equal to the target's greedy tokens; tokens per verify step = accepted + 1.
"""
import argparse
import collections
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsugumi.env import env  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))
PROMPTS = [
    ("日本の四季について、それぞれの特徴を短く説明してください。", "none"),
    ("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", "none"),
    ("Tell me a short story about a robot who learns to paint.", "none"),
    ("次の文章を丁寧語に直してください: 明日は雨が降るから、会議はオンラインにする。資料は前の日に送っておく。", "low"),
]


def generate(prompts, n_new):
    import ttnn
    from transformers import AutoTokenizer

    from tsugumi.inkling.chat import encode
    from tsugumi.inkling.engine import Engine
    from tsugumi.inkling.generator import InklingGenerator
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    out = []
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=8192, log=lambda *a: None)
        eng.capture()
        gen = InklingGenerator(eng)
        for text, effort in prompts:
            ids = encode(tok, [{"role": "user", "content": text}], effort)
            reply = list(gen.generate(ids, n_new))
            out.append((ids, reply))
            print(f"generated {len(reply)}: {tok.decode(reply)[:120]!r}", flush=True)
    finally:
        ttnn.close_device(dev)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=160)
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--save", default="/tmp/inkling_mtp_seqs.pt")
    a = ap.parse_args()
    if a.no_device:
        seqs = torch.load(a.save)
    else:
        seqs = generate(PROMPTS, a.new)
        torch.save(seqs, a.save)
    torch.set_num_threads(16)
    from tsugumi.inkling.draft_ref import DraftRef
    from tsugumi.inkling.ref import Reference
    ref = Reference(MODEL)
    dr = DraftRef(MODEL, ref.ck)
    visits = dr.mc["target_layer_ids"][:-1]
    totals = {"markov": [], "no markov": []}
    for ids, reply in seqs:
        seq = ids + [t for t in reply]
        t0 = time.perf_counter()
        tr = {}
        lg = ref.forward(seq, trace=tr)
        feats = torch.cat([tr["apps"][v]["h_in"] for v in visits] + [tr["final"]], dim=-1)  # [T, 4H]
        agree = sum(int(lg[len(ids) - 1 + k].argmax()) == reply[k] for k in range(len(reply))) / len(reply)
        print(f"reference forward {len(seq)} tokens {time.perf_counter() - t0:.0f}s, greedy agreement with the "
              f"device {agree:.3f}", flush=True)
        for mode in totals:
            p, steps = len(ids), []
            while p + dr.block <= len(seq):
                d = dr.draft(feats[:p], seq[p], p, markov=(mode == "markov"))
                k = 0
                while k < len(d) and d[k] == seq[p + 1 + k]:
                    k += 1
                steps.append(k + 1)
                p += k + 1
            totals[mode] += steps
            print(f"  {mode:9s}: {len(steps)} steps, {sum(steps) / len(steps):.2f} tokens/step, "
                  f"distribution {sorted(collections.Counter(steps).items())}", flush=True)
    for mode, st in totals.items():
        print(f"ALL {mode}: {sum(st) / len(st):.2f} tokens/step over {len(st)} verify steps")


if __name__ == "__main__":
    main()
