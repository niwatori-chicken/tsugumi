"""T-row mode test: chunked prefill + plain greedy decode where each step commits 1 row (a=1).
Compares against the T=1 engine output text. python scripts/run_spec_test.py [--layers N] [--T 4]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import argparse
import sys
import time

import ttnn

from tsugumi.model import Precision, Qwen35Engine
import tsugumi.paths as tsugumi_paths

MODEL = tsugumi_paths.MODEL
CACHE = tsugumi_paths.CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", default="bfp4")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--prompt", default="日本の首都はどこですか？一文で答えてください。")
    ap.add_argument("--prompt-file", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if a.prompt_file:
        a.prompt = open(a.prompt_file, errors="ignore").read()
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                  enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision(a.prec), f"{CACHE}/{a.prec}", max_seq=4096, n_layers=a.layers,
                           log=lambda *x: None)
        eng.init_spec(a.T)
        eng.reset_spec()
        if not a.no_trace:
            eng.set_inputs_spec([0], 0, 0)
            eng.capture_spec()
            eng.reset_spec()
        t0 = time.time()
        nxt, pos, _ = eng.prefill_spec(ids)
        ttnn.synchronize_device(dev)
        print(f"prefill {len(ids)} tok in {time.time() - t0:.2f}s (T={a.T})")
        out = [nxt]
        a_prev = eng._a_prev
        times = []
        for _ in range(a.max_new):
            t1 = time.perf_counter()
            am = eng.run_spec([out[-1]], pos, a_prev)
            times.append(time.perf_counter() - t1)
            a_prev = 1
            pos += 1
            out.append(am[0])
            if am[0] in (tok.eos_token_id, 248046, 248044):
                break
        print("OUTPUT:", tok.decode(out))
        st = sorted(times)
        print(f"T={a.T} step median {st[len(st)//2]*1e3:.1f} ms")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
