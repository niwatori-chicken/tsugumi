"""Large-T prefill vs the T=32 path: next token, MTP draft, greedy continuation, timing.
  python scripts/test_prefill_large.py [N tokens] [t_max] [max_new]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys, time, torch
import ttnn
from transformers import AutoTokenizer
import tsugumi.paths as tsugumi_paths
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.generator import Generator

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
TMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
tok = AutoTokenizer.from_pretrained(tsugumi_paths.MODEL)
text = open("/mnt/HDD/eval/ja_wiki.txt").read()
body = tok.encode(text)[: N - 40]
msgs = [{"role": "user", "content": tok.decode(body) + "\n\n上の文章を3文で要約してください。"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
ids = ids if isinstance(ids, list) else ids["input_ids"]
print("[prompt]", len(ids), flush=True)
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsugumi_paths.MODEL, Precision("bfp4"), f"{tsugumi_paths.CACHE}/bfp4", max_seq=16384,
                       log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.init_large_prefill(TMAX)
    eng.capture_all(); eng.reset_spec()
    gen = Generator(eng)
    res = {}
    for name, use_large in (("large", True), ("t32", False), ("large2", True)):
        lp = eng.large
        if not use_large:
            eng.large = None
        gen.reset()
        st = {}
        t0 = time.perf_counter()
        out = []
        for t in gen.generate(ids, max_new=NEW, stats=st):
            if not out:
                t1 = time.perf_counter()
            out.append(t)
        t2 = time.perf_counter()
        eng.large = lp
        res[name] = out
        print(f"[{name}] prefill {t1 - t0:.2f}s ({len(ids) / (t1 - t0):.0f} tok/s)  decode {(len(out) - 1) / (t2 - t1):.1f} tok/s"
              f"  accepted {st.get('accepted')}/{st.get('steps')}", flush=True)
        print("   ", repr(tok.decode(out)), flush=True)
    a, b = res["large2"], res["t32"]
    same = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    print(f"[agree] first {same} of {min(len(a), len(b))} tokens identical", flush=True)
finally:
    ttnn.close_device(dev)
