"""Device time per op of one batched decode step (main model + one MTP run), for kernel-level optimization.

  scripts/../llm-bench/qwen38_profiler_python.sh -m tracy -r -p scripts/profile_batch_ops.py [lanes=8] [P0=300]
  python scripts/summarize_profile.py      (newest device profile: per-op totals of the main and the MTP segment)
Same as profile_step.py, but for the batched mode with `lanes` busy lanes (run eagerly: same kernels and device
times as the trace, not the same gaps between ops). lanes=1 is the one-lane batched mode."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys

import ttnn

from tsugumi.batch import BatchMode
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.paths import CACHE, MODEL

lanes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
P0 = int(sys.argv[2]) if len(sys.argv) > 2 else 300
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=16384, log=lambda *a: None)
    eng.init_spec(4, mtp=True)
    bm = BatchMode(eng, lanes)
    key = ("b", lanes)
    eng.use_T(key)
    toks = list(range(1000, 1004))
    rows = [(toks, P0, []) for _ in range(lanes)]

    def main():
        bm.set_inputs(key, rows, [1] * lanes)
        eng.forward_spec()

    def mtp():
        bm.set_mtp_inputs(key, rows, [list(range(4))] * lanes)
        eng.mtp.forward()

    for _ in range(2):  # compile
        main()
        mtp()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)  # drop the warm-up records
    print(f"[profile] main step, {lanes} lanes", flush=True)
    main()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    print("[profile] mtp step", flush=True)
    mtp()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
finally:
    ttnn.close_device(dev)
