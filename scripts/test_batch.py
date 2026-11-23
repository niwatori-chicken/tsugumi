"""Batched decode end to end: the same prompts generated one at a time (Generator.generate, the classic path) and
concurrently through the batch scheduler (tsugumi/batch.py); greedy outputs are compared, then the throughput with
1..LANES concurrent requests is measured.
  TSUGUMI_MODEL_DIR=... TSUGUMI_CACHE_DIR=... python scripts/test_batch.py [lanes=4] [max_new=128] [max_seq=16384]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import os
import sys
import threading
import time

os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "60")
import ttnn
from transformers import AutoTokenizer

from tsugumi.batch import END, BatchMode, BatchScheduler, Job
from tsugumi.env import env
from tsugumi.generator import Generator
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.paths import CACHE, MODEL as MODEL_DIR

PROMPTS = [
    "Pythonでクイックソートを実装して、計算量も説明してください。",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "日本の四季について、それぞれの特徴を説明してください。",
    "Explain how a transformer neural network works, step by step.",
    "C言語で連結リストを反転する関数を書いてください。",
    "What are the main differences between TCP and UDP?",
    "富士山について小学生向けに説明してください。",
    "Write a haiku sequence about autumn rain, then explain each one.",
]


def main():
    lanes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    max_seq = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    dev = ttnn.open_device(device_id=0, trace_region_size=int(env("TRACE_MB", "768")) << 20, l1_small_size=24576)
    try:
        t0 = time.time()
        eng = Qwen35Engine(dev, MODEL_DIR, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=max_seq,
                           log=lambda m: None)
        tv = int(env("VERIFY_T", "0"))
        eng.init_spec(4, mtp=True, verify_T=tv or None)
        eng.init_prefill(32)
        if tv:
            eng.init_verify(tv)
        if env("LARGE_PREFILL", "0") == "1":
            eng.init_large_prefill(int(env("LARGE_T", "1024")))
        BatchMode(eng, lanes)
        eng.capture_all()
        v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        print(f"engine ready in {time.time() - t0:.0f}s, DRAM free "
              f"{v.total_bytes_free_per_bank * v.num_banks / (1 << 30):.2f} GiB", flush=True)
        gen = Generator(eng)
        ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                       enable_thinking=False) for p in PROMPTS[:max(lanes, 2)]]
        ids = [x if isinstance(x, list) else x["input_ids"] for x in ids]

        # ---- reference: one at a time, classic path ----
        ref = []
        for x in ids:
            gen.reset()
            t1 = time.perf_counter()
            out = list(gen.generate(x, max_new=max_new))
            ref.append(out)
            print(f"single: {len(out)} tok, {len(out) / (time.perf_counter() - t1):.1f} tok/s (incl. prefill)",
                  flush=True)
        gen.reset()

        lock = threading.Lock()
        sched = BatchScheduler(gen, lock, log=print)

        def run_jobs(xs):
            jobs = [Job(x, max_new, tag=str(j)) for j, x in enumerate(xs)]
            outs = [[] for _ in jobs]
            t1 = time.perf_counter()
            for j in jobs:
                sched.submit(j)
            for j, o in zip(jobs, outs):
                while True:
                    it = j.get()
                    if it is END:
                        break
                    if isinstance(it, Exception):
                        raise it
                    o.append(it[0])
            dt = time.perf_counter() - t1
            return outs, jobs, dt

        # ---- correctness: all prompts concurrently ----
        outs, jobs, dt = run_jobs(ids)
        for j, (o, r) in enumerate(zip(outs, ref)):
            m = next((n for n, (a, b) in enumerate(zip(o, r)) if a != b), min(len(o), len(r)))
            st = jobs[j].stats
            print(f"job {j}: {len(o)} tok, same as single for the first {m} tokens"
                  f"{' (identical)' if o == r else ''}; tokens/step "
                  f"{(st['accepted'] + st['steps']) / max(st['steps'], 1):.2f}", flush=True)
            if o != r:
                print("   single :", repr(tok.decode(r[max(0, m - 5):m + 15])))
                print("   batched:", repr(tok.decode(o[max(0, m - 5):m + 15])))
        # ---- conversation reuse after a batched turn ----
        x2 = ids[0] + outs[0] + tok.encode("<|im_end|>\n<|im_start|>user\n続けて<|im_end|>\n<|im_start|>assistant\n",
                                           add_special_tokens=False)
        o2, j2, _ = run_jobs([x2])
        print(f"follow-up turn: reuse {j2[0].stats.get('reuse')} ({j2[0].stats.get('reused')} tokens), "
              f"prefilled {j2[0].stats.get('prefilled')}", flush=True)
        # ---- throughput ----
        for n in sorted({1, 2, lanes} | ({4} if lanes >= 4 else set())):
            gen.reset()
            eng.batch.prof.clear()
            outs, jobs, dt = run_jobs(ids[:n])
            pr = eng.batch.prof
            ns = max(pr["steps"], 1)
            print(f"   per step: {pr['step'] / ns * 1e3:.1f} ms = main {pr['main'] / ns * 1e3:.1f} + mtp "
                  f"{pr['mtp'] / ns * 1e3:.1f} + host {(pr['step'] - pr['main'] - pr['mtp']) / ns * 1e3:.1f}")
            tot = sum(len(o) for o in outs)
            print(f"concurrent {n}: {tot} tokens in {dt:.2f}s = {tot / dt:.1f} tok/s total, "
                  f"{tot / dt / n:.1f} per request (incl. prefill); last step {sched.step_time * 1e3:.1f} ms",
                  flush=True)
        sched.stop()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
