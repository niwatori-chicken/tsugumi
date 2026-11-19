"""Greedy generation with the lean engine (prefill = token-by-token decode for now).

python scripts/run_gen.py [--prec mixed|bfp4] [--layers N] [--max-new 64] [--prompt "..."] [--no-trace]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import argparse
import sys
import time

import torch
import ttnn

from qe.model import Precision, Qwen35Engine
import qe.paths as qe_paths

MODEL = qe_paths.MODEL
CACHE = qe_paths.CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", default="mixed")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--prompt", default="日本の首都はどこですか？一文で答えてください。")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if a.raw:
        ids = tok(a.prompt)["input_ids"]
    else:
        ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                      enable_thinking=False)
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
    print(f"prompt tokens: {len(ids)}")

    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    try:
        cache = f"{CACHE}/{a.prec}"
        eng = Qwen35Engine(dev, MODEL, Precision(a.prec), cache, max_seq=a.max_seq, n_layers=a.layers)
        eng.reset()
        if not a.no_trace:
            t0 = time.time()
            eng.set_inputs(0, 0)
            eng.capture()
            eng.reset()
            print(f"trace captured in {time.time() - t0:.1f}s")
        t0 = time.time()
        for p, t in enumerate(ids[:-1]):
            eng.run(t, p)
        ttnn.synchronize_device(dev)
        tp = time.time() - t0
        print(f"prefill (token-by-token) {len(ids) - 1} tok in {tp:.2f}s")
        out = []
        cur = ids[-1]
        pos = len(ids) - 1
        times = []
        for _ in range(a.max_new):
            t1 = time.perf_counter()
            cur = eng.step_greedy(cur, pos)
            times.append(time.perf_counter() - t1)
            out.append(cur)
            pos += 1
            if cur in (tok.eos_token_id, 248046, 248044):
                break
        print("OUTPUT:", tok.decode(out))
        st = sorted(times[2:]) if len(times) > 4 else times
        if eng.trace_id is not None:
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(dev, eng.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            dtr = (time.perf_counter() - t1) / 20
            t1 = time.perf_counter()
            for _ in range(20):
                eng.argmax_to_host(eng.logits_out)
            dlg = (time.perf_counter() - t1) / 20
            t1 = time.perf_counter()
            for _ in range(20):
                eng.set_inputs(cur, pos)
            dsi = (time.perf_counter() - t1) / 20
            print(f"device forward {dtr*1e3:.1f} ms, argmax_to_host {dlg*1e3:.1f} ms, set_inputs {dsi*1e3:.1f} ms")
        print(f"decode: {len(out)} tok, median {st[len(st)//2]*1e3:.1f} ms/tok = {1/st[len(st)//2]:.1f} tok/s")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
