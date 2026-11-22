"""Speed: GQAAttention vs current ops at long context.  python scripts/bench_gqa.py [P0]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys, time, torch, ttnn
from tsugumi.gqa_attn import GQAAttention
NH, NKV, HD, BS = 24, 4, 256, 64
P0 = int(sys.argv[1]) if len(sys.argv) > 1 else 123904
maxseq = 131072; nb = maxseq // BS
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
def tm(f, n=20):
    f(); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(4): f()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(n): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev); dt = (time.perf_counter() - t) / n / 4 * 1e3
    ttnn.release_trace(dev, tid); return dt
try:
    z = torch.randn(nb, NKV, BS, HD)
    kc = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vc = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    for T in (4, 32):
        ptg = ttnn.from_torch(torch.arange(nb, dtype=torch.int32).view(1, nb), dtype=ttnn.int32, device=dev)
        posd = ttnn.from_torch(torch.arange(P0, P0 + T, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        q = ttnn.from_torch(torch.randn(1, T, NH, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        qg = ttnn.from_torch(torch.randn(1, NH, T, HD) if T == 32 else torch.randn(1, T, NH, HD), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2):
            A = GQAAttention(dev, T, fidelity=fid, nblocks=nb)
            print(f"T={T} P0={P0} gqa {fid}: {tm(lambda: A(qg, kc, vc, posd, ptg)):.2f} ms", flush=True)
        pt = ttnn.from_torch(torch.arange(nb, dtype=torch.int32).repeat(T, 1), dtype=ttnn.int32, device=dev)
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)
        f = lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(q, kc, vc, page_table_tensor=pt, cur_pos_tensor=posd, scale=1/16, program_config=cfg, compute_kernel_config=ck)
        print(f"T={T} decode-sdpa (T users): {tm(f):.2f} ms", flush=True)
finally:
    ttnn.close_device(dev)
