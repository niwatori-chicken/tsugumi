"""Dump per-position argmax during prefill (teacher forcing) for comparing prefill paths. python scripts/prefill_agree.py OUT.pt"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import sys, glob, random, torch
import ttnn
from qe.model import Precision, Qwen35Engine
from transformers import AutoTokenizer
import qe.paths as qe_paths
MODEL = qe_paths.MODEL; CACHE = qe_paths.CACHE
tok = AutoTokenizer.from_pretrained(MODEL)
text = open(sys.argv[2] if len(sys.argv) > 2 else _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "README.md")).read()
ids = tok.encode(text)[:6000]
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=16384, log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.capture_all(); eng.reset_spec()
    rec = []; orig = eng.run_spec
    def wrap(chunk, pos, a):
        am = orig(chunk, pos, a); rec.extend(am[:len(chunk)]); return am
    eng.run_spec = wrap
    # prefill in two calls with an unaligned split to exercise the alignment path
    eng.prefill_spec(ids[:1001], start=0)
    eng.prefill_spec(ids[1001:], start=1001)
    torch.save(dict(ids=ids, am=rec), sys.argv[1]); print("[n]", len(rec), len(ids))
finally:
    ttnn.close_device(dev)
