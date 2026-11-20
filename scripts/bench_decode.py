"""Decode speed over a fixed prompt set (greedy, server configuration), for comparing optimizations.

  python scripts/bench_decode.py [max_new=256] [--reps 2]
Prompts: Japanese / English prose, code, a tool-free chat answer, with and without thinking. Prints per prompt the
decode rate and tokens per verify step, then the total (tokens / decode time) and the mean of per-prompt rates.
Settings come from the environment like the server (TSUMI_KV_DTYPE, TSUMI_DRAFT_VOCAB, ...)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import sys, time
import ttnn
from transformers import AutoTokenizer
import tsumi.paths as tsumi_paths
from tsumi.model import Precision, Qwen35Engine
from tsumi.generator import Generator

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAX_NEW = int(args[0]) if args else 256
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2
PROMPTS = [
    ("ja prose", "量子コンピュータの仕組みを高校生向けに説明してください。", False),
    ("ja chat", "週末に東京で一人で楽しめる過ごし方を5つ提案して、それぞれ理由も書いてください。", False),
    ("en prose", "Explain how vaccines train the immune system, in plain language.", False),
    ("en story", "Write a short story about a lighthouse keeper.", False),
    ("code py", "Pythonでクイックソートを実装して、計算量を説明してください。", False),
    ("code ts", "Write a TypeScript function that debounces another function, with types and a usage example.", False),
    ("think ja", "1から100までの整数のうち、3でも5でも割り切れない数はいくつありますか？", True),
    ("think en", "A train leaves at 9:40 and arrives at 13:15. How long is the trip in minutes?", True),
]
tok = AutoTokenizer.from_pretrained(tsumi_paths.MODEL)
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsumi_paths.MODEL, Precision("bfp4"), f"{tsumi_paths.CACHE}/bfp4", max_seq=16384,
                       log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.init_large_prefill(1024); eng.capture_all()
    gen = Generator(eng)
    tot_n = tot_t = 0.0
    rates = []
    for name, text, think in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=think)
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        best = None
        for _ in range(REPS):
            gen.reset()
            st = {}
            t0 = time.perf_counter()
            n = 0
            for t in gen.generate(ids, max_new=MAX_NEW, stats=st):
                if n == 0:
                    t1 = time.perf_counter()
                n += 1
            dt = time.perf_counter() - t1
            r = (n - 1) / dt
            if best is None or r > best[0]:
                best = (r, n - 1, dt, (st["accepted"] + st["steps"]) / max(st["steps"], 1))
        r, n, dt, tps = best
        tot_n += n
        tot_t += dt
        rates.append(r)
        print(f"  {name:9s} {n:4d} tok  {r:5.1f} tok/s  {tps:.2f} tok/step", flush=True)
    print(f"total {tot_n / tot_t:.1f} tok/s, mean of prompts {sum(rates) / len(rates):.1f} tok/s", flush=True)
finally:
    ttnn.close_device(dev)
