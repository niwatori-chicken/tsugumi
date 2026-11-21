"""Decode speed over a fixed prompt set (greedy, server configuration), for comparing optimizations.

  python scripts/bench_decode.py [max_new=256] [--reps 2] [--copy | --all]
--copy: copy-heavy prompts only (code edit, proofreading, rewriting); --all: both sets. --dump FILE saves
the prompt ids and generated tokens (greedy output does not depend on the drafts: for offline simulations).
Prompts: Japanese / English prose, code, a tool-free chat answer, with and without thinking. Prints per prompt the
decode rate and tokens per verify step, then the total (tokens / decode time) and the mean of per-prompt rates.
Settings come from the environment like the server (TSUGUMI_KV_DTYPE, TSUGUMI_DRAFT_VOCAB, ...)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import sys, time
import ttnn
from transformers import AutoTokenizer
import tsugumi.paths as tsugumi_paths
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.generator import Generator
from tsugumi.env import env

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
_here = _os.path.dirname(_os.path.abspath(__file__))
_kv = open(_os.path.join(_here, "..", "tsugumi", "kvpager.py")).read()
_ja = open("/mnt/HDD/eval/ja_eval.txt").read()[:900]
_en = open("/mnt/HDD/eval/en_eval.txt").read()[:1500]
COPY = [  # copy-heavy replies (code edits, rewrites): where prompt-lookup drafts help
    ("edit code", "次の Python ファイルの `ensure` メソッドに、引数と例外を説明する docstring を追加して、"
                  "ファイル全体をコードブロックで出力してください。\n\n```python\n" + _kv + "```", False),
    ("fix ja", "次の文章の誤字脱字があれば直して、全文をそのまま出力してください。説明は不要です。\n\n" + _ja, False),
    ("edit en", "Rewrite the following text, replacing every occurrence of the word \"the\" with \"THE\". "
                "Output only the full rewritten text.\n\n" + _en, False),
]
if "--copy" in sys.argv:
    PROMPTS = COPY
elif "--all" in sys.argv:
    PROMPTS = PROMPTS + COPY
tok = AutoTokenizer.from_pretrained(tsugumi_paths.MODEL)
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsugumi_paths.MODEL, Precision("bfp4"), f"{tsugumi_paths.CACHE}/bfp4", max_seq=16384,
                       log=lambda *a: None)
    tv = int(env("VERIFY_T", "8"))
    eng.init_spec(4, mtp=True, verify_T=tv or None); eng.init_prefill(32)
    if tv:
        eng.init_verify(tv)
    eng.init_large_prefill(1024); eng.capture_all()
    gen = Generator(eng)
    tot_n = tot_t = 0.0
    dump = {}
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
            outs = []
            for t in gen.generate(ids, max_new=MAX_NEW, stats=st):
                if n == 0:
                    t1 = time.perf_counter()
                n += 1
                outs.append(t)
            dump[name] = (list(ids), outs)
            dt = time.perf_counter() - t1
            r = (n - 1) / dt
            if best is None or r > best[0]:
                best = (r, n - 1, dt, (st["accepted"] + st["steps"]) / max(st["steps"], 1), st)
        r, n, dt, tps, st = best
        tot_n += n
        tot_t += dt
        rates.append(r)
        ngs = f"  n-gram steps {st.get('ngram_steps', 0)}/{st['steps']}" if "ngram_steps" in st else ""
        print(f"  {name:9s} {n:4d} tok  {r:5.1f} tok/s  {tps:.2f} tok/step{ngs}", flush=True)
    print(f"total {tot_n / tot_t:.1f} tok/s, mean of prompts {sum(rates) / len(rates):.1f} tok/s", flush=True)
    if "--dump" in sys.argv:  # prompt ids and generated tokens per prompt (for offline draft simulations)
        import torch
        torch.save(dump, sys.argv[sys.argv.index("--dump") + 1])
finally:
    ttnn.close_device(dev)
