"""ttnn.sparse_matmul for the MoE prefill (doc/12 "prefill"): tokens sorted by expert and cut into blocks of M rows,
each block marked with its one expert in the sparsity tensor ([1, NB, 1, E], built on device in a real engine), so
every block is one [M, K] x [K, N] matmul with that expert's weights. The question is how much a block costs, and
so how much re-reading an expert's weights once per block costs against reading every expert once.

  python scripts/bench_sparse_matmul.py [--M 32,64,128] [--layers 5,20,35]
Block counts come from real routing (moe_route_stats.py output, two texts concatenated to one T=2048 chunk).
Sizes: Qwen3.6-35B-A3B gate_up (K=2048, N=1024) and down (K=512, N=2048), 256 experts, bfp4."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import argparse
import math
import time

import torch
import ttnn

ROUTES = "/mnt/HDD/eval/moe_routes_q36.pt"
E, KSEL, T = 256, 8, 2048


def real_counts(layers):
    d = torch.load(ROUTES)
    i, j = d["names"].index("ja prose"), d["names"].index("code py")
    r = torch.cat([d["routes"][i], d["routes"][j]], 1)[:, :T]  # [L, T, k]
    return {l: torch.bincount(r[l].reshape(-1).long(), minlength=E) for l in layers}


def pc_for(dev, M, N, K):
    """The 1D mcast factory needs a rectangle of exactly Nt / per_core_N cores: per_core_N = 1, 8 columns."""
    g = dev.compute_with_storage_grid_size()
    Nt = N // 32
    pn = 1
    while Nt // pn > 8 * g.y or (Nt // pn) % 8:
        pn += 1
    bw = max(d for d in (1, 2, 4, 8) if (K // 32) % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, Nt // pn // 8), in0_block_w=bw, out_subblock_h=1,
        out_subblock_w=1, out_block_h=1, out_block_w=pn, per_core_M=M // 32, per_core_N=pn, fuse_batch=False,
        fused_activation=None, mcast_in0=True)


def timed(dev, fn, iters=5):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(iters):
        fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / iters
    ttnn.release_trace(dev, tid)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", default="32,64,128")
    ap.add_argument("--layers", default="5,20,35")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    counts = real_counts(layers)
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    try:
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                              fp32_dest_acc_en=False, packer_l1_acc=True)
        for name, K, N in (("gate_up", 2048, 1024), ("down", 512, 2048)):
            g = torch.Generator().manual_seed(0)
            wb = (torch.randn(1, E, K, N, generator=g) * 0.05).bfloat16()
            w = ttnn.from_torch(wb, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            exp_mb = K * N * 576 / 1024 / 1e6
            print(f"== {name} K={K} N={N}: one expert {exp_mb:.2f} MB", flush=True)
            for M in [int(x) for x in a.M.split(",")]:
                for l in layers:
                    c = counts[l]
                    blocks = [e for e in range(E) for _ in range(math.ceil(int(c[e]) / M))]
                    NB = len(blocks)
                    used = int((c > 0).sum())
                    x = torch.randn(1, NB, M, K, generator=g).bfloat16()
                    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    sp = torch.zeros(1, NB, 1, E)
                    sp[0, torch.arange(NB), 0, torch.tensor(blocks)] = 1.0
                    spt = ttnn.from_torch(sp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
                    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, NB, M, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                         dev, ttnn.DRAM_MEMORY_CONFIG)
                    pc = pc_for(dev, M, N, K)
                    f = lambda: ttnn.sparse_matmul(xt, w, sparsity=spt, nnz=NB, program_config=pc,
                                                   compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                   optional_output_tensor=out)
                    try:
                        dt = timed(dev, f)
                    except Exception as ex:
                        print(f"  M={M} layer {l}: FAILED {str(ex).splitlines()[0][:150]}", flush=True)
                        continue
                    # correctness on a few blocks
                    got = ttnn.to_torch(out).float()
                    errs = []
                    for b in (0, NB // 2, NB - 1):
                        ref = x[0, b].float() @ wb[0, blocks[b]].float()
                        errs.append(torch.corrcoef(torch.stack([got[0, b].flatten(), ref.flatten()]))[0, 1].item())
                    read_gb = NB * exp_mb / 1e3
                    print(f"  M={M:3d} layer {l:2d}: {NB:3d} blocks ({used} experts used, rows {int(c.sum())}), "
                          f"{dt * 1e3:6.2f} ms = {dt / NB * 1e6:5.1f} us/block, weights read {read_gb:.2f} GB "
                          f"({read_gb / dt:.0f} GB/s), all-used-once floor {used * exp_mb / 1e3 / 390 * 1e3:.2f} ms, "
                          f"PCC {min(errs):.4f}", flush=True)
                    for t in (xt, spt, out):
                        ttnn.deallocate(t)
            ttnn.deallocate(w)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
