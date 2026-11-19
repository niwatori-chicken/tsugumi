"""Large-T prefill: time per component (synchronized) for one chunk. python bench_prefill_large.py [T] [P0]"""
import sys, time, collections, torch, ttnn
import qe.paths as qe_paths
from qe.model import Precision, Qwen35Engine
T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
P0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, qe_paths.MODEL, Precision("bfp4"), f"{qe_paths.CACHE}/bfp4", max_seq=max(4096, P0 + T + 64),
                       log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_large_prefill(max(T, 64)); eng.reset_spec()
    lp = eng.large
    acc = collections.defaultdict(float)
    def timed(name, f):
        def g(*a, **k):
            ttnn.synchronize_device(dev); t = time.perf_counter()
            r = f(*a, **k); ttnn.synchronize_device(dev); acc[name] += time.perf_counter() - t
            return r
        return g
    ids = torch.randint(1000, 100000, (P0 + T,)).tolist()
    lp.run(ids[:T], start=0, with_mtp=True)  # compile
    for nm in ("_gdn", "_attn", "_mlp", "_norm", "_mtp_kv", "_enter", "_exit"):
        setattr(lp, nm, timed(nm, getattr(lp, nm)))
    lp.convk.__call__ = timed("conv", lp.convk.__call__)
    for rep in range(2):
        acc.clear(); eng.reset_spec()
        if P0:
            lp.run(ids[:P0], start=0, with_mtp=True)
            acc.clear()
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        lp.run(ids[P0:P0 + T], start=P0, with_mtp=True)
        ttnn.synchronize_device(dev); tot = time.perf_counter() - t0
    print(f"T={T} P0={P0}: total {tot*1e3:.0f} ms ({T/tot:.0f} tok/s)")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8s} {v*1e3:7.1f} ms")
finally:
    ttnn.close_device(dev)
