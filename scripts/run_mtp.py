"""Greedy speculative decoding with the built-in MTP head (draft k = T-1 tokens per step).

python scripts/run_mtp.py [--T 4] [--max-new 200] [--prompt ...] [--no-trace] [--compare]
--compare: also run plain greedy (a=1 per step) and check the outputs are identical.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import argparse
import faulthandler
import signal
import sys
import time

import ttnn

from tsumi.model import Precision, Qwen35Engine
import tsumi.paths as tsumi_paths

MODEL = tsumi_paths.MODEL
CACHE = tsumi_paths.CACHE
EOS = (248046, 248044)


def generate(eng, ids, max_new, k, stats):
    tp0 = time.perf_counter()
    cur, p, d = eng.prefill_spec(ids, with_mtp=True)
    stats["t_prefill"] = time.perf_counter() - tp0
    a_prev = eng._a_prev
    last_rows = eng._mtp_last_rows
    out = [cur]
    drafts = [d]
    q = p + 1
    for j in range(1, k):
        am = eng.mtp.run([drafts[-1]], q, [last_rows - 1 if j == 1 else 0])
        drafts.append(am[0])
        q += 1
    while len(out) < max_new:
        t0 = time.perf_counter()
        y = eng.run_spec([cur] + drafts, p, a_prev)
        t1 = time.perf_counter()
        n = 0
        while n < k and drafts[n] == y[n]:
            n += 1
        stats.setdefault("pos_ok", [0] * k)
        for i in range(n):
            stats["pos_ok"][i] += 1
        new = drafts[:n] + [y[n]]
        stats["steps"] += 1
        stats["accepted"] += n
        stats["t_main"] += t1 - t0
        out += new
        if any(t in EOS for t in new):
            break
        # MTP: rows 0..n: token at p+1+i, hidden row i
        toks = drafts[:n] + [y[n]]
        am = eng.mtp.run(toks, p + 1, list(range(n + 1)))
        nd = [am[n]]
        q = p + n + 2
        for j in range(1, k):
            am = eng.mtp.run([nd[-1]], q, [n if j == 1 else 0])
            nd.append(am[0])
            q += 1
        stats["t_mtp"] += time.perf_counter() - t1
        cur = y[n]
        p = p + n + 1
        drafts = nd
        a_prev = n + 1
    return out


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", default="bfp4")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--prefill-T", type=int, default=32)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--prompt", default="Pythonでフィボナッチ数列を返す関数を書いてください。")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if a.prompt_file:
        a.prompt = open(a.prompt_file, errors="ignore").read()
    ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                  enable_thinking=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision(a.prec), f"{CACHE}/{a.prec}", max_seq=4096, log=lambda *x: None)
        eng.init_spec(a.T, mtp=True)
        if a.prefill_T:
            eng.init_prefill(a.prefill_T)
        eng.reset_spec()
        if not a.no_trace:
            eng.capture_all()
        stats = dict(steps=0, accepted=0, t_main=0.0, t_mtp=0.0)
        t0 = time.time()
        out = generate(eng, ids, a.max_new, a.T - 1, stats)
        dt = time.time() - t0
        print("OUTPUT:", tok.decode(out))
        s = stats
        print(f"tokens {len(out)} in {dt:.2f}s (incl. prefill {len(ids)} tok in {stats['t_prefill']:.2f}s = "
              f"{len(ids)/stats['t_prefill']:.0f} tok/s)")
        if s["steps"]:
            print(f"steps {s['steps']}, accepted drafts/step {s['accepted']/s['steps']:.2f}, "
                  f"tokens/step {(s['accepted']+s['steps'])/s['steps']:.2f}, "
                  f"main {s['t_main']/s['steps']*1e3:.1f} ms/step, mtp {s['t_mtp']/s['steps']*1e3:.1f} ms/step, "
                  f"decode rate {(s['accepted']+s['steps'])/(s['t_main']+s['t_mtp']):.1f} tok/s")
            print("acceptance by draft position:", [f"{v/s['steps']:.2f}" for v in s.get("pos_ok", [])])
        if not a.no_trace:
            for name, tid in [(f"main T={T}", t) for T, t in eng.spec_traces.items()] + \
                    [(f"mtp T={T}", t[0]) for T, t in eng.mtp.traces.items()]:
                ttnn.synchronize_device(dev)
                t1 = time.perf_counter()
                for _ in range(10):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                print(f"device {name} trace: {(time.perf_counter() - t1) / 10 * 1e3:.2f} ms")
            t1 = time.perf_counter()
            for _ in range(10):
                eng.set_inputs_spec([1, 2, 3, 4], 100, 2)
            print(f"host set_inputs_spec: {(time.perf_counter() - t1) / 10 * 1e3:.2f} ms")
            t1 = time.perf_counter()
            for _ in range(10):
                eng.topk_to_host(eng.spec_out[1], eng.T)
            print(f"host read top-k: {(time.perf_counter() - t1) / 10 * 1e3:.2f} ms")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
