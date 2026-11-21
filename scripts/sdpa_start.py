"""chunked SDPA: start index as int (legacy) vs device tensor (chunk_start_idx_tensor) — program cache growth,
speed and agreement.  python scripts/sdpa_start.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import time, torch, ttnn

T, NB = 1024, 2560  # 160K positions of page table
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
try:
    g = dev.compute_with_storage_grid_size()
    kc = ttnn.from_torch(torch.randn(NB, 4, 64, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vc = ttnn.from_torch(torch.randn(NB, 4, 64, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    q = ttnn.from_torch(3 * torch.randn(1, 24, T, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    pt = ttnn.from_torch(torch.arange(NB, dtype=torch.int32).view(1, NB), dtype=ttnn.int32, device=dev)
    st = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=dev)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=False)
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False,
                                 q_chunk_size=int(_os.environ.get("C", "128")), k_chunk_size=int(_os.environ.get("C", "128")))
    legacy = lambda p: ttnn.transformer.chunked_scaled_dot_product_attention(
        q, kc, vc, pt, chunk_start_idx=p, scale=1 / 16, program_config=cfg, compute_kernel_config=ck)

    def flex(p):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([p], dtype=torch.int32), dtype=ttnn.int32), st)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            q, kc, vc, pt, chunk_start_idx_tensor=st, scale=1 / 16, program_config=cfg, compute_kernel_config=ck)

    starts = [0, 1024, 2048, 3072, 30720, 122880]
    for name, f in (("legacy", legacy), ("tensor", flex)):
        n0 = dev.num_program_cache_entries()
        res = {}
        for p in starts:
            o = f(p); ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(3):
                ttnn.deallocate(o); o = f(p)
            ttnn.synchronize_device(dev)
            res[p] = ((time.perf_counter() - t) / 3 * 1e3, ttnn.to_torch(o).float())
            ttnn.deallocate(o)
        print(f"{name}: program cache +{dev.num_program_cache_entries() - n0} entries for {len(starts)} starts; "
              + "  ".join(f"P0={p}: {res[p][0]:.1f}ms" for p in starts), flush=True)
        if name == "legacy":
            ref = res
        else:
            for p in starts:
                a, b = res[p][1], ref[p][1]
                print(f"   P0={p}: tensor vs legacy max diff {(a - b).abs().max():.2e}", flush=True)
finally:
    ttnn.close_device(dev)
