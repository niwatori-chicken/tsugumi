"""Load the server configuration with a given max_seq and report DRAM usage.  python scripts/memprobe.py 131072 [--no-large]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import sys, time
import ttnn
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.generator import Generator
from transformers import AutoTokenizer
import tsugumi.paths as tsugumi_paths
MODEL = tsugumi_paths.MODEL; CACHE = tsugumi_paths.CACHE
ms = int(sys.argv[1])
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
def mem(tag):
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    G = 1 << 30; n = v.num_banks
    print(f"[mem] {tag}: total {v.total_bytes_per_bank*n/G:.2f} GiB, used {v.total_bytes_allocated_per_bank*n/G:.2f}, "
          f"free {v.total_bytes_free_per_bank*n/G:.2f}, largest contiguous free {v.largest_contiguous_bytes_free_per_bank*n/G:.2f}", flush=True)
try:
    mem("open")
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=ms, log=lambda *a: None)
    mem("weights")
    eng.init_spec(4, mtp=True, verify_T=8); eng.init_prefill(32); eng.init_verify(8)  # server configuration
    mem("spec+prefill")
    if len(sys.argv) < 3 or sys.argv[2] != "--no-large":
        eng.init_large_prefill(1024)
        mem("large prefill (after warmup)")
    eng.capture_all()
    mem("traced")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.apply_chat_template([{"role": "user", "content": "日本で一番高い山は？一文で。"}], add_generation_prompt=True, enable_thinking=False)
    ids = ids if isinstance(ids, list) else ids["input_ids"]
    g = Generator(eng); out = []
    for t in g.generate(ids, max_new=40):
        out.append(t)
    print("[out]", tok.decode(out), flush=True)
    mem("after a request")
finally:
    ttnn.close_device(dev)
