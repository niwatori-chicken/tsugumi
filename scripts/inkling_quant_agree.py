"""Expert precision vs quality: teacher-forced top-1 predictions of the device engine over saved replies
(scripts/inkling_mtp_estimate.py --save), for one expert policy; run twice and compare with --against.

    TSUGUMI_INKLING_EXPERTS=bfp8 python scripts/inkling_quant_agree.py --out /path/bfp8.pt
    TSUGUMI_INKLING_EXPERTS=bfp4 python scripts/inkling_quant_agree.py --out /path/bfp4.pt --against /path/bfp8.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import ttnn  # noqa: E402

from tsugumi.env import env  # noqa: E402
from tsugumi.inkling.engine import Engine  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))
SEQS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "tsugumi-cache",
                    "inkling_mtp_seqs.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--against")
    a = ap.parse_args()
    seqs = torch.load(SEQS)
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    res = []
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=8192, log=lambda *x: None, prefill=False)
        eng.capture()
        for ids, reply in seqs:
            seq = ids + list(reply)
            eng.reset()
            top = []
            for k, t in enumerate(seq[:-1]):
                out = eng.step(t, logits=k >= len(ids) - 1)
                if out is not None:
                    vals, tids = out
                    v = vals.float().tolist()
                    top.append((int(tids[0]), v[0] - v[1], int(tids[1])))  # top-1, gap to top-2, top-2
            res.append(top)
    finally:
        ttnn.close_device(dev)
    torch.save(res, a.out)
    n = sum(len(t) for t in res)
    own = sum(int(tp[0] == s) for (ids, reply), tops in zip(seqs, res) for tp, s in zip(tops, reply))
    print(f"{n} positions; top-1 = the saved (bfp8, greedy) reply: {own}/{n} ({own / n:.3%})")
    if a.against:
        ref = torch.load(a.against)
        diff = [(x, y) for r1, r2 in zip(res, ref) for x, y in zip(r1, r2) if x[0] != y[0]]
        print(f"top-1 differs from {os.path.basename(a.against)} at {len(diff)}/{n} positions ({len(diff) / n:.2%})")
        gaps = sorted(y[1] for x, y in diff)
        if gaps:
            print(f"  the other run's top-2 gap at those positions: median {gaps[len(gaps) // 2]:.3f}, "
                  f"max {gaps[-1]:.3f}; > 0.5 at {sum(g > 0.5 for g in gaps)}, > 1.0 at {sum(g > 1.0 for g in gaps)}")


if __name__ == "__main__":
    main()
