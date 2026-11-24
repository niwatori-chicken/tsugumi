"""Device time per op of one large-T prefill chunk (T rows at position P0), for finding what to fuse.

  python -m tracy -r -p scripts/profile_prefill.py [T=2048] [P0=0]
  python scripts/summarize_profile.py --segments 1      (the profiled chunk is the last segment)
  python scripts/profile_prefill.py [T] [P0] --wall     (wall time of the same chunk; not under tracy)
Two warm-up runs, then one chunk with host synchronizations around it (a > 2 ms gap splits the segments)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys, time
import torch
import ttnn
import tsugumi.paths as tsugumi_paths
from tsugumi.model import Precision, Qwen35Engine

args = [a for a in sys.argv[1:] if not a.startswith("--")]
T = int(args[0]) if args else 2048
P0 = int(args[1]) if len(args) > 1 else 0
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsugumi_paths.MODEL, Precision("bfp4"), f"{tsugumi_paths.CACHE}/bfp4",
                       max_seq=max(8192, P0 + T + 128), log=lambda *a: None)
    eng.init_spec(4, mtp=True)
    eng.init_large_prefill(T)  # includes the warm-up of every bucket
    lp = eng.large
    ids = torch.randint(1000, 100000, (P0 + T,)).tolist()

    def chunk():
        eng.reset_spec()
        if P0:
            lp.run(ids[:P0], start=0, with_mtp=True)
        ttnn.synchronize_device(dev)
        time.sleep(0.05)
        t0 = time.perf_counter()
        lp.run(ids[P0:P0 + T], start=P0, with_mtp=True)
        ttnn.synchronize_device(dev)
        return time.perf_counter() - t0

    for _ in range(2):
        w = chunk()
    print(f"T={T} P0={P0}: wall {w * 1e3:.1f} ms ({T / w:.0f} tok/s)", flush=True)
    if "--wall" not in sys.argv:
        ttnn.ReadDeviceProfiler(dev)
        time.sleep(0.05)
        chunk()
        time.sleep(0.05)
        ttnn.ReadDeviceProfiler(dev)
finally:
    ttnn.close_device(dev)
