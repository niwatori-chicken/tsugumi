"""Would speculative decoding with the Inkling 2 DSpark draft pay off on this engine? Greedy, on real replies.

    python scripts/inkling_mtp_estimate.py [--new 160] [--no-device] [--step-ms 31.3 --moe-ms 9.5 --draft-ms 3]

1. greedy replies to a few chat prompts on the device (cached in --save; --no-device reuses them),
2. the CPU reference (tsugumi/inkling/ref.py) gives every position's target features and router choices,
3. for k = 1..7 drafts, speculative decoding is simulated with the CPU draft (tsugumi/inkling/draft_ref.py):
   tokens per verify step, and the number of distinct experts the k + 1 verified tokens use per layer visit
   (the MoE reads them all: its cost is bandwidth),
4. cost model of one verify step: (step - moe) * (1 + growth * k) + moe * distinct / 8 + draft, against the
   current one-token step.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsugumi.env import env  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
PROMPTS = [
    ("日本の四季について、それぞれの特徴を短く説明してください。", "none"),
    ("Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.", "none"),
    ("Tell me a short story about a robot who learns to paint.", "none"),
    ("次の文章を丁寧語に直してください: 明日は雨が降るから、会議はオンラインにする。資料は前の日に送っておく。", "low"),
    ("Explain the difference between TCP and UDP in a few bullet points.", "low"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=int, default=160)
    ap.add_argument("--no-device", action="store_true")
    ap.add_argument("--save", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                   "tsugumi-cache", "inkling_mtp_seqs.pt"))
    ap.add_argument("--step-ms", type=float, default=31.3, help="current one-token decode step")
    ap.add_argument("--moe-ms", type=float, default=9.5, help="its MoE part (8 experts per layer visit)")
    ap.add_argument("--draft-ms", type=float, default=3.0, help="assumed draft cost per verify step")
    a = ap.parse_args()
    if a.no_device:
        seqs = torch.load(a.save)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from inkling_mtp_accept import generate  # noqa: E402
        seqs = generate(PROMPTS, a.new)
        torch.save(seqs, a.save)
    torch.set_num_threads(16)
    from tsugumi.inkling.draft_ref import DraftRef
    from tsugumi.inkling.ref import Reference
    ref = Reference(MODEL)
    dr = DraftRef(MODEL, ref.ck)
    visits = dr.mc["target_layer_ids"][:-1]
    KS = (1, 2, 3, 4, 5, 7)
    steps = {k: [] for k in KS}      # tokens per verify step
    distinct = {k: [] for k in KS}   # mean distinct experts per layer visit, per verify step
    for ids, reply in seqs:
        seq = ids + list(reply)
        t0 = time.perf_counter()
        tr = {}
        ref.forward(seq, trace=tr)
        feats = torch.cat([tr["apps"][v]["h_in"] for v in visits] + [tr["final"]], dim=-1)
        route = torch.stack([app["router_idx"] for app in tr["apps"]])  # [apps, T, 8]
        drafts = {}
        for k in KS:
            p = len(ids)
            while p + k + 1 <= len(seq):
                if p not in drafts:
                    drafts[p] = dr.draft(feats[:p], seq[p], p, markov=True)
                d = drafts[p][:k]
                acc = 0
                while acc < k and d[acc] == seq[p + 1 + acc]:
                    acc += 1
                steps[k].append(acc + 1)
                win = route[:, p:p + k + 1]  # the k + 1 verified positions
                distinct[k].append(sum(len(set(w.flatten().tolist())) for w in win) / win.shape[0])
                p += acc + 1
        print(f"sequence of {len(seq)} tokens ({len(reply)} generated): {time.perf_counter() - t0:.0f}s", flush=True)
    nonmoe = a.step_ms - a.moe_ms
    print(f"\ncurrent: {a.step_ms:.1f} ms / token ({1000 / a.step_ms:.1f} tok/s); model: non-MoE {nonmoe:.1f} ms, "
          f"MoE {a.moe_ms:.1f} ms per 8 experts, draft {a.draft_ms:.1f} ms")
    print(f"{'k':>2} {'tok/step':>8} {'experts':>8} | {'verify ms':>18} | {'ms/token':>14} | {'speedup':>12}")
    print(f"{'':>2} {'':>8} {'/visit':>8} | {'opt':>8} {'pess':>9} | {'opt':>6} {'pess':>7} | {'opt':>5} {'pess':>6}")
    for k in KS:
        tps = sum(steps[k]) / len(steps[k])
        u = sum(distinct[k]) / len(distinct[k])
        moe = a.moe_ms * u / 8
        v_opt = nonmoe + moe + a.draft_ms                       # non-MoE parts cost the same for k + 1 tokens
        v_pes = nonmoe * (1 + 0.15 * k) + moe + a.draft_ms      # +15% per extra token
        print(f"{k:>2} {tps:>8.2f} {u:>8.1f} | {v_opt:>8.1f} {v_pes:>9.1f} | {v_opt / tps:>6.1f} {v_pes / tps:>7.1f} | "
              f"{a.step_ms / (v_opt / tps):>5.2f} {a.step_ms / (v_pes / tps):>6.2f}")


if __name__ == "__main__":
    main()
