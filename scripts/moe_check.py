"""Correctness of the MoE engine (branch moe-probe, doc/12) against the CPU reference of scripts/mtp_accept_cpu.py:
teacher-forced next-token argmax over decode_seqs texts, through the large-T prefill (method B) and through the
T-row verify mode, compared with the CPU bf16 model's argmax (/mnt/HDD/eval/mtp_hid_q36.pt). Also times one
prefill chunk.

  python scripts/moe_check.py [MODEL_DIR=/mnt/HDD/Qwen36_35BA3B] [--names "ja prose,code py"] [--decode 64]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import os
import sys
import time

import torch
import ttnn
from tsugumi.model import Precision, Qwen35Engine

args = [a for i, a in enumerate(sys.argv[1:]) if not a.startswith("--") and not sys.argv[i].startswith("--")]
MODEL = args[0] if args else "/mnt/HDD/Qwen36_35BA3B"
CACHE = "/mnt/HDD/models/qe_cache/" + os.path.basename(MODEL.rstrip("/"))
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
NAMES = opt("--names", "ja prose,code py").split(",")
NDEC = int(opt("--decode", "64"))
ref = torch.load("/mnt/HDD/eval/mtp_hid_q36.pt")

dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    t0 = time.time()
    eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=8192,
                       log=lambda m: print(m, flush=True) if ("loaded" in m and ("/40" in m and m.split()[2].split("/")[0] in ("1", "10", "20", "30", "40")) or "model loaded" in m) else None)
    print(f"engine up in {time.time() - t0:.0f}s", flush=True)
    eng.init_spec(4, mtp=False)
    eng.init_large_prefill(2048)
    for name in NAMES:
        s = ref["names"].index(name)
        ids, p0, pred = ref["ids"][s].tolist(), ref["n_prompt"][s], ref["pred"][s]
        N = len(ids)
        # large-T prefill over the whole text, logits of every generated position
        eng.reset_spec()
        eng.large.keep = []
        ttnn.synchronize_device(dev)
        t1 = time.perf_counter()
        eng.large.run(ids, start=0, with_mtp=False)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t1
        am, pos = [], 0
        for hid, n in eng.large.keep:
            lo, hi = max(p0 - 1, pos), min(N - 1, pos + n)
            if lo < hi:
                am.append(eng.large.logits_host(hid, lo - pos, hi - pos).argmax(-1))
            pos += n
            ttnn.deallocate(hid)
        eng.large.keep = None
        am = torch.cat(am)
        r = pred[p0 - 1:N - 1]
        print(f"{name}: prefill {N} tok in {dt:.2f}s ({N / dt:.0f} tok/s); argmax == CPU bf16 "
              f"{(am == r).float().mean():.1%}, == text {(am == torch.tensor(ids[p0:])).float().mean():.1%}", flush=True)
        # T-row mode: prompt by large prefill, then teacher-forced rows of the continuation, 4 at a time
        eng.reset_spec()
        eng.large.run(ids[:p0], start=0, with_mtp=False)
        got, pos, a_prev = [], p0, eng._a_prev
        toks = ids[p0:p0 + NDEC]
        for j in range(0, len(toks), 4):
            out = eng.run_spec(toks[j:j + 4], pos, a_prev)
            got += out[:len(toks[j:j + 4])]
            pos += len(toks[j:j + 4])
            a_prev = len(toks[j:j + 4])
        g = torch.tensor(got[:len(toks) - 1])
        print(f"  T-row mode, {len(g)} rows: argmax == CPU bf16 {(g == pred[p0:p0 + len(g)]).float().mean():.1%}",
              flush=True)
finally:
    ttnn.close_device(dev)
