"""DRAM peak and speed of the large-T prefill for a chunk size (server configuration: max_seq, device ViT, traces).

  python scripts/measure_prefill_dram.py T_MAX [N_TOKENS ...]

The peak is sampled (in a separate, slower run) before every ttnn.deallocate during the prefill (temporaries are freed explicitly, so the high
water mark is seen just before a free) and is reported relative to the DRAM in use before the prefill.
Env: TSUGUMI_MAX_SEQ (262144), TSUGUMI_VISION_DEVICE (1), TSUGUMI_REPS (3: the DRAM run + 2 timed runs)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import glob, os, random, sys, time
import ttnn
from transformers import AutoTokenizer
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.env import env
import tsugumi.paths as tsugumi_paths

MODEL = tsugumi_paths.MODEL; CACHE = tsugumi_paths.CACHE
t_max = int(sys.argv[1])
lens = [int(a) for a in sys.argv[2:]] or [8192, 32768]
G, M = 1 << 30, 1 << 20

tok = AutoTokenizer.from_pretrained(MODEL)
files = sorted(glob.glob(os.environ.get("TT_METAL_HOME", ".") + "/ttnn/cpp/ttnn/operations/**/*.hpp", recursive=True))
random.Random(0).shuffle(files)
corpus = []
for f in files:
    corpus += tok.encode(open(f, errors="ignore").read())
    if len(corpus) > max(lens):
        break

dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)


def view():
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    n = v.num_banks
    return v.total_bytes_allocated_per_bank * n, v.total_bytes_free_per_bank * n, v.largest_contiguous_bytes_free_per_bank * n


def mem(tag):
    u, f, c = view()
    print(f"[mem] {tag}: used {u / G:.3f} GiB, free {f / G:.3f}, largest contiguous free {c / G:.3f}", flush=True)
    return u


peak = {"used": 0, "minfree": 1 << 62, "mincont": 1 << 62}
_dealloc = ttnn.deallocate


def sampling_dealloc(t, *a, **k):
    u, f, c = view()
    peak["used"] = max(peak["used"], u)
    peak["minfree"] = min(peak["minfree"], f)
    peak["mincont"] = min(peak["mincont"], c)
    return _dealloc(t, *a, **k)


try:
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=int(env("MAX_SEQ", "262144")),
                       log=lambda *a: None)
    if env("VISION_DEVICE", "1") == "1":
        from tsugumi.vision_tt import VisionTT
        vt = VisionTT(dev, MODEL, f"{CACHE}/bfp4", log=lambda *a: None)
    eng.init_spec(4, mtp=True, verify_T=8); eng.init_prefill(32); eng.init_verify(8)
    mem("before large prefill")
    t0 = time.perf_counter()
    eng.init_large_prefill(t_max)
    print(f"[warmup] t_max {t_max}: {time.perf_counter() - t0:.1f}s", flush=True)
    if env("VISION_DEVICE", "1") == "1":
        vt.warmup(1 << 20)
    eng.capture_all()
    base = mem("traced (server state)")
    for n in lens:
        ids = corpus[:n]
        for rep in range(int(env("REPS", "3"))):  # 1st: DRAM sampling (slow: the memory view per free); 2nd, 3rd: speed
            eng.reset_spec()
            peak.update(used=0, minfree=1 << 62, mincont=1 << 62)
            if rep == 0:
                ttnn.deallocate = sampling_dealloc
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            eng.prefill(ids, start=0, with_mtp=True)
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            ttnn.deallocate = _dealloc
            if rep:
                print(f"[prefill] t_max {t_max}, {n} tok, run {rep}: {dt:.2f}s = {n / dt:.0f} tok/s", flush=True)
                continue
            print(f"[dram] t_max {t_max}, {n} tok: peak work {(peak['used'] - base) / M:.0f} MiB, min free {peak['minfree'] / G:.3f} GiB, "
                  f"min largest contiguous {peak['mincont'] / G:.3f} GiB", flush=True)
    mem("end")
finally:
    ttnn.deallocate = _dealloc
    ttnn.close_device(dev)
