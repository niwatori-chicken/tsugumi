"""Inkling 2 Mini: two chat turns through the generator (prompt reuse via the snapshot), traced, as the server runs
them. Checks that the second turn runs (no hang) and prints both replies.

    python scripts/inkling_multiturn.py [--untraced]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")
import ttnn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from tsugumi.env import env  # noqa: E402
from tsugumi.inkling.chat import ReplyParser, encode  # noqa: E402
from tsugumi.inkling.engine import Engine  # noqa: E402
from tsugumi.inkling.generator import InklingGenerator  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))


def reply(tok, gen, ids, n):
    p = ReplyParser(tok)
    out = {"content": "", "reasoning_content": ""}
    st = {}
    t = time.perf_counter()
    for x in gen.generate(ids, n, stats=st):
        for f, v in p.feed(x):
            if f in out:
                out[f] += v
    for f, v in p.finish():
        if f in out:
            out[f] += v
    return out, st, time.perf_counter() - t


def main():
    traced = "--untraced" not in sys.argv
    tok = AutoTokenizer.from_pretrained(MODEL)
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=int(env("MAX_SEQ", "8192")))
        if traced:
            eng.capture()
        if not traced:
            eng.traces = {}
        gen = InklingGenerator(eng)
        msgs = [{"role": "user", "content": "Name three primary colors, comma separated."}]
        for turn in range(3):
            ids = encode(tok, msgs, "low")
            out, st, dt = reply(tok, gen, ids, 120)
            print(f"turn {turn}: prompt {len(ids)} reused {st.get('reused')} ({st.get('reuse')}) prefilled "
                  f"{st.get('prefilled')} in {st.get('t_prefill', 0):.2f}s, total {dt:.1f}s: {out['content']!r}",
                  flush=True)
            msgs += [{"role": "assistant", "content": out["content"]},
                     {"role": "user", "content": ["And three secondary colors?", "Which one is your favorite?"][turn % 2]}]
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
