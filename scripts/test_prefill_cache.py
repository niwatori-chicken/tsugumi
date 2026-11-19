"""After warmup + trace capture, prefills at many start positions / lengths, with decode steps in between, must not
add programs to the program cache (a program compiled after the capture can allocate buffers that a trace replay
overwrites, which hung the engine). Also checks the replies stay sane.  python scripts/test_prefill_cache.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import faulthandler, random, time, torch, ttnn
faulthandler.dump_traceback_later(900, exit=True)
import qe.paths as qe_paths
from qe.model import Precision, Qwen35Engine
from qe.generator import Generator

dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, qe_paths.MODEL, Precision("bfp4"), f"{qe_paths.CACHE}/bfp4", max_seq=16384, log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32)
    t0 = time.perf_counter(); eng.init_large_prefill(1024); print(f"[warmup] {time.perf_counter() - t0:.1f}s", flush=True)
    eng.capture_all()
    gen = Generator(eng)
    base = dev.num_program_cache_entries()
    rnd = random.Random(0)
    text = torch.load("/mnt/HDD/eval/eval_ids.pt")["ja"].reshape(-1).tolist()  # 12288 tokens
    ids, src = text[:300], 300
    for step in range(14):
        n_new = rnd.choice([5, 40, 70, 130, 300, 700, 1500])
        ids = ids + text[src:src + n_new]
        src += n_new
        out = []
        st = {}
        for t in gen.generate(ids, max_new=rnd.choice([3, 20]), stats=st):
            if not out:
                after_prefill = dev.num_program_cache_entries() - base
            out.append(t)
        ids = ids + out  # the conversation continues from the reply (exact reuse next time)
        print(f"[{step}] len {len(ids)} (+{n_new} new) reuse={st.get('reuse')} reused={st.get('reused')}  "
              f"program cache after prefill +{after_prefill}, after decode +{dev.num_program_cache_entries() - base}",
              flush=True)
    print("[done]", flush=True)
finally:
    ttnn.close_device(dev)
