"""Where the time of a batched decode step goes, per lane count.

  python scripts/profile_batch.py engine [lanes=8] [P0=300]   traced main / MTP step of every batched mode, all lanes busy
  python scripts/profile_batch.py kernels [P0=300]            the per-lane kernels alone (traced, 20 calls each):
                                                            GDN fused (x48 layers), GQA attention and KV write (x17)
The per-lane kernels are what grows with the lane count; the matmuls read the same weights for any row count."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys
import time

import torch
import ttnn

N_GDN, N_ATT = 48, 17  # GDN layers; attention layers + the MTP layer
SEG = 4


def timed(dev, fn, n=20):
    """Mean seconds of fn() replayed from a trace of n calls."""
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / n
    ttnn.release_trace(dev, tid)
    return dt


def kernels(P0):
    from tsugumi.gdn_kernel import GDNFusedKernel
    from tsugumi.gqa_attn import GQAAttention
    from tsugumi.kv_kernel import KVWriteKernel
    NK, NV, D = 16, 48, 128
    KD, VD = NK * D, NV * D
    QKV = 2 * KD + VD
    NH, NKV, HD, BS = 24, 4, 256, 64
    nb = 4096 // BS
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    L1 = ttnn.L1_MEMORY_CONFIG
    try:
        up = lambda t, dt=ttnn.bfloat16, mc=None: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                                                  memory_config=mc)
        X = up(torch.randn(1, 1, 32, QKV) * 0.1, mc=L1)
        Y = up(torch.randn(1, 1, 32, QKV + VD + 96) * 0.1, mc=L1)
        GB = up(torch.rand(1, 1, 32, 96), ttnn.float32)
        kc = ttnn.zeros([nb, NKV, BS, HD], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        vc = ttnn.zeros([nb, NKV, BS, HD], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        print(f"{'lanes':>5} {'GDN/layer':>10} {'x48':>8} {'2 groups x48':>13} {'GQA/layer':>10} {'KVw/layer':>10} "
              f"{'x17 both':>9}  (ms)")
        for n in (1, 2, 4, 8):
            T = n * SEG
            kern = GDNFusedKernel(dev, NV, NK, T, QKV, KD, seg=SEG)
            C = GDNFusedKernel.make_consts(dev, torch.ones(D), T)
            S = up(torch.zeros(1, T * NV, D, D), ttnn.float32)
            IDX = GDNFusedKernel.make_idx(dev, 0)
            GDNFusedKernel.set_idx(IDX, [u * SEG for u in range(n)])
            out = kern(X, Y, GB, GB, C, S, IDX)
            t_gdn = timed(dev, lambda: kern(X, Y, GB, GB, C, S, IDX, OUT=out))
            t_gdn2 = None
            if n >= 2:
                k2 = GDNFusedKernel(dev, NV, NK, T, QKV, KD, seg=SEG, groups=2)
                o2 = k2(X, Y, GB, GB, C, S, IDX)
                t_gdn2 = timed(dev, lambda: k2(X, Y, GB, GB, C, S, IDX, OUT=o2))
                ttnn.deallocate(o2)
            pos = [ttnn.from_torch(torch.arange(P0, P0 + SEG, dtype=torch.int32), dtype=ttnn.int32, device=dev)
                   for _ in range(n)]
            pts = [ttnn.from_torch(torch.arange(nb, dtype=torch.int32).view(1, nb), dtype=ttnn.int32, device=dev)
                   for _ in range(n)]
            A = GQAAttention(dev, SEG, nblocks=nb, kv_dtype=ttnn.bfloat8_b, lanes=n)
            q = up(torch.randn(1, T, NH, HD) * 0.1)
            t_gqa = timed(dev, lambda: A(q, kc, vc, pos, pts))
            kvw = KVWriteKernel(dev, SEG, NKV, HD, BS, nb, ttnn.bfloat8_b, lanes=n)
            ks = ttnn.slice(up(torch.randn(1, T, 32, HD)), (0, 0, 0, 0), (1, T, NKV, HD))
            t_kvw = timed(dev, lambda: kvw(ks, ks, kc, vc, pos, pts))
            g2 = f"{t_gdn2 * N_GDN * 1e3:13.2f}" if t_gdn2 else f"{'-':>13}"
            print(f"{n:5d} {t_gdn * 1e3:10.3f} {t_gdn * N_GDN * 1e3:8.2f} {g2} {t_gqa * 1e3:10.3f} {t_kvw * 1e3:10.3f} "
                  f"{(t_gqa + t_kvw) * N_ATT * 1e3:9.2f}", flush=True)
            for t in [S, IDX, C, out, q, ks] + pos + pts:
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(dev)


def engine(lanes, P0):
    from tsugumi.batch import BatchMode
    from tsugumi.model import Precision, Qwen35Engine
    from tsugumi.paths import CACHE, MODEL
    dev = ttnn.open_device(device_id=0, trace_region_size=1024 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=16384, log=lambda m: None)
        eng.init_spec(4, mtp=True)
        eng.init_prefill(32)
        bm = BatchMode(eng, lanes)
        eng.capture_all()
        toks = list(range(1000, 1000 + SEG))
        # single-sequence T=4 step for reference
        eng.use_T(4)

        def single():
            eng.run_spec(toks, P0, 1)
            eng.mtp.run(toks, P0 + 1, list(range(SEG)))
        single()
        t0 = time.perf_counter()
        for _ in range(10):
            single()
        print(f"single T=4 (main + 1 MTP): {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms", flush=True)
        print(f"{'lanes':>5} {'main':>8} {'MTP':>7} {'main+3 MTP':>11} {'host prep':>10}  (ms, all lanes busy)")
        for key in bm.keys():
            n = key[1]
            rows = [(toks, P0, []) for _ in range(n)]
            eng.use_T(key)
            t_prep = t_main = t_mtp = 0.0
            for it in range(12):
                t0 = time.perf_counter()
                bm.set_inputs(key, rows, [1] * n)
                t1 = time.perf_counter()
                bm.run_main(key)
                t2 = time.perf_counter()
                bm.set_mtp_inputs(key, rows, [list(range(SEG))] * n)
                t3 = time.perf_counter()
                bm.run_mtp(key)
                t4 = time.perf_counter()
                if it >= 2:
                    t_prep += (t1 - t0) + (t3 - t2)
                    t_main += t2 - t1
                    t_mtp += t4 - t3
            k = 10
            print(f"{n:5d} {t_main / k * 1e3:8.1f} {t_mtp / k * 1e3:7.1f} {(t_main + 3 * t_mtp) / k * 1e3:11.1f} "
                  f"{t_prep / k * 1e3:10.1f}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "engine"
    if what == "kernels":
        kernels(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        engine(int(sys.argv[2]) if len(sys.argv) > 2 else 8, int(sys.argv[3]) if len(sys.argv) > 3 else 300)
