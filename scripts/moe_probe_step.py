"""Device time of one decode step of a MoE checkpoint (Qwen3.6-35B-A3B) on the tsugumi engine (branch moe-probe,
doc/12): the verify trace for T rows and one MTP draft pass, replayed back to back.

  python scripts/moe_probe_step.py [MODEL_DIR=/mnt/HDD/Qwen36_35BA3B] [T=4]
With TSUGUMI_MOE_ROUTED=0 (default) the routed experts are skipped: the time is "everything but the experts"."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import os
import sys
import time

import ttnn
from tsugumi.model import Precision, Qwen35Engine

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = args[0] if args else "/mnt/HDD/Qwen36_35BA3B"
T = int(args[1]) if len(args) > 1 else 4
CACHE = "/mnt/HDD/models/qe_cache/" + os.path.basename(MODEL.rstrip("/"))


def replay(dev, tid, n=30):
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n


dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    t0 = time.time()
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=4096,
                       log=lambda m: print(m, flush=True) if "model loaded" in m else None)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    eng.init_spec(T, mtp=True)
    eng.capture_all()
    if True:
        eng.set_inputs_spec(list(range(1000, 1000 + T)), 100, 1)
        tv = replay(dev, eng.spec_trace)
        tm = replay(dev, eng.mtp.traces[T][0])
        print(f"T={T}: verify {tv * 1e3:.2f} ms, MTP pass {tm * 1e3:.2f} ms", flush=True)
finally:
    ttnn.close_device(dev)
