"""Load the server configuration with a given max_seq and report DRAM usage.  python memprobe.py 131072"""
import sys, time
import ttnn
from qe.model import Precision, Qwen35Engine
from qe.generator import Generator
from transformers import AutoTokenizer
import qe.paths as qe_paths
MODEL = qe_paths.MODEL; CACHE = qe_paths.CACHE
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
    eng.init_spec(4, mtp=True); eng.init_prefill(32)
    mem("spec+prefill")
    eng.capture_all()
    mem("traced")
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.apply_chat_template([{"role": "user", "content": "日本で一番高い山は？一文で。"}], add_generation_prompt=True, enable_thinking=False)
    ids = ids if isinstance(ids, list) else ids["input_ids"]
    g = Generator(eng); out = []
    for t in g.generate(ids, max_new=40):
        out.append(t)
    print("[out]", tok.decode(out), flush=True)
finally:
    ttnn.close_device(dev)
