"""Device time per op of one decode step (T-row verify + MTP drafts), for finding what to optimize.

  python -m tracy -r -p scripts/profile_step.py [T=4] [P0=300]
  python scripts/profile_step.py [T] [P0] --traced     (traced time of the same step; not under tracy)
  python scripts/summarize_profile.py      (reads the newest ops_perf_results CSV)
Runs the step eagerly (same kernels as the trace; the device times per op are the same, the gaps between ops are
not) after two warm-up runs, so the profiled region is exactly one main step and one MTP step. Also prints the
traced time of the same step for comparison."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys, time
import ttnn
import tsugumi.paths as tsugumi_paths
from tsugumi.model import Precision, Qwen35Engine

args = [a for a in sys.argv[1:] if not a.startswith("--")]
T = int(args[0]) if args else 4
P0 = int(args[1]) if len(args) > 1 else 300
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsugumi_paths.MODEL, Precision("bfp4"), f"{tsugumi_paths.CACHE}/bfp4", max_seq=4096,
                       log=lambda *a: None)
    eng.init_spec(T, mtp=True)
    eng.reset_spec()
    toks = list(range(1000, 1000 + T))
    for _ in range(2):  # compile
        eng.set_inputs_spec(toks, P0, 1)
        eng.forward_spec()
        eng.mtp.run(toks[:1], P0 + 1, [0])
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)  # drop the warm-up records
    eng.set_inputs_spec(toks, P0, 1)
    print("[profile] main step", flush=True)
    eng.forward_spec()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    print("[profile] mtp step", flush=True)
    eng.mtp.run(toks[:1], P0 + 1, [0])
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    if "--traced" not in sys.argv:
        raise SystemExit
    # traced time of the same step
    eng.capture_spec(compile_first=False)
    eng.mtp.capture(compile_first=False)
    for name, f in (("main", lambda: eng.run_spec(toks, P0, 1)), ("mtp", lambda: eng.mtp.run(toks[:1], P0 + 1, [0]))):
        f()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            f()
        ttnn.synchronize_device(dev)
        print(f"[traced] {name}: {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms", flush=True)
finally:
    ttnn.close_device(dev)
