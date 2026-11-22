"""Needle test + speed at long context.  python scripts/longctx.py N_TOKENS"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import os, sys, time, random, glob
import ttnn
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.generator import Generator
from transformers import AutoTokenizer
import tsugumi.paths as tsugumi_paths
from tsugumi.env import env
MODEL = tsugumi_paths.MODEL; CACHE = tsugumi_paths.CACHE
N = int(sys.argv[1])
tok = AutoTokenizer.from_pretrained(MODEL)
files = sorted(glob.glob(os.environ.get("TT_METAL_HOME", ".") + "/ttnn/cpp/ttnn/operations/**/*.hpp", recursive=True))
random.Random(0).shuffle(files)
buf, n = [], 0
for f in files:
    s = open(f, errors="ignore").read()
    buf.append(s); n += len(tok.encode(s))
    if n > N: break
needle = "\n// NOTE: the secret password for the vault is 'blue-heron-4729'.\n"
buf.insert(len(buf) // 2, needle)
text = "".join(buf)
q = sys.argv[3] if len(sys.argv) > 3 else "\n\nAbove is a large code dump. Somewhere in it is a comment with the secret password for the vault. What is the password? Answer with the password only."
ids = tok.apply_chat_template([{"role": "user", "content": text + q}], add_generation_prompt=True, enable_thinking=False)
ids = ids if isinstance(ids, list) else ids["input_ids"]
print(f"[prompt] {len(ids)} tokens", flush=True)
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=int(env("MAX_SEQ", "262144")), log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32)
    if env("LARGE_PREFILL", "1") == "1":
        eng.init_large_prefill(int(env("LARGE_T", "1024")))
    eng.capture_all()
    g = Generator(eng); st = {}; out = []; t0 = time.perf_counter(); t1 = None
    for t in g.generate(ids, max_new=int(sys.argv[2]) if len(sys.argv) > 2 else 60, stats=st):
        if t1 is None: t1 = time.perf_counter()
        out.append(t)
    t2 = time.perf_counter()
    print(f"[prefill] {len(ids)} tok in {t1-t0:.1f}s = {len(ids)/(t1-t0):.0f} tok/s", flush=True)
    print(f"[decode] {len(out)} tok in {t2-t1:.2f}s = {len(out)/(t2-t1):.1f} tok/s, steps {st['steps']}, acc/step {st['accepted']/max(st['steps'],1):.2f}", flush=True)
    print("[out]", tok.decode(out), flush=True)
    # main verify trace time at this context
    for T, tid in eng.spec_traces.items():
        ttnn.synchronize_device(dev); a = time.perf_counter()
        for _ in range(5): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev); print(f"[trace] main T={T}: {(time.perf_counter()-a)/5*1e3:.1f} ms", flush=True)
finally:
    ttnn.close_device(dev)
