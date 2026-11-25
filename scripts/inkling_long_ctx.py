"""Inkling 2 Mini: prefill and decode speed against the context length (traced, as the server runs).

    python scripts/inkling_long_ctx.py [--lens 512,4096,8192,16384,30000] [--new 48] [--max-seq 32768]

For each length: a prompt of that many tokens (repeated natural text), the 32-row prefill, then NEW decode steps
(greedy over the non-eos tokens, so that every step runs); prints prefill tok/s and decode tok/s / ms per token.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ttnn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from tsugumi.env import env  # noqa: E402
from tsugumi.inkling.engine import Engine  # noqa: E402
from tsugumi.inkling.generator import InklingGenerator  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))
TEXT = ("The history of the city goes back many centuries. Merchants came from distant lands, bringing spices, "
        "silk and stories; scholars argued in its libraries, and the river carried boats day and night. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="512,4096,8192,16384,30000")
    ap.add_argument("--new", type=int, default=48)
    ap.add_argument("--max-seq", type=int, default=32768)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    unit = tok.encode(TEXT, add_special_tokens=False)
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=a.max_seq, log=lambda *x: None)
        eng.capture()
        eos = 200006

        def pick(vals, ids):  # greedy over everything but eos
            for v, i in zip(vals.tolist(), ids.tolist()):
                if int(i) != eos:
                    return int(i)
            return int(ids[0])
        print(f"{'context':>8} | {'prefill':>16} | {'decode':>22}")
        for n in [int(x) for x in a.lens.split(",")]:
            ids = (unit * (n // len(unit) + 1))[:n]
            gen = InklingGenerator(eng)
            st = {}
            times = []
            t = time.perf_counter()
            for k, _ in enumerate(gen.generate(ids, a.new, stats=st, sampler=pick)):
                now = time.perf_counter()
                if k > 0:
                    times.append(now - t)
                t = now
            dec = times[4:] if len(times) > 8 else times  # skip the first steps after the prefill
            ms = sum(dec) / len(dec) * 1e3
            print(f"{n:>8} | {st['t_prefill']:6.1f} s {n / st['t_prefill']:6.0f} tok/s | {ms:6.1f} ms "
                  f"{1000 / ms:6.1f} tok/s", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
