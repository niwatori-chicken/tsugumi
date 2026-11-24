"""Device time per op of one Inkling 2 Mini decode step (with the lm_head).

  TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 python -m tracy -r -p scripts/inkling_profile.py
  python scripts/summarize_profile.py --segments 1
Runs the step eagerly (the same kernels as the trace) after two warm-up steps."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ttnn  # noqa: E402

from tsugumi.env import env  # noqa: E402
from tsugumi.inkling.engine import Engine  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))
dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
try:
    eng = Engine(dev, MODEL, CACHE, max_seq=int(env("MAX_SEQ", "8192")), log=lambda *a: None)
    eng.reset()
    for t in (1000, 1001):
        eng.step(t, logits=True, traced=False)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    print("[profile] step", flush=True)
    eng.step(1002, logits=True, traced=False)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
finally:
    ttnn.close_device(dev)
