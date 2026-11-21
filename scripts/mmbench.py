"""B=1 decode matmul bandwidth bench on P100A (DRAM-sharded weights, trace replay).

Usage: python scripts/mmbench.py [--workers 1,2] [--dtypes bfp4,bfp8] [--shapes qwen27]
Prints time per matmul and effective weight bandwidth.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import argparse
import math
import time

import torch
import ttnn

TILE = 32

SHAPES = {
    # name: (K, N)
    "mlp_gate_up": (5120, 17408),
    "mlp_down": (17408, 5120),
    "gdn_in": (5120, 16480),
    "gdn_out": (6144, 5120),
    "attn_qkv": (5120, 14336),
    "attn_o": (6144, 5120),
    "lm_head": (5120, 248320),
}

BYTES_PER_EL = {"bfp4": 576 / 1024, "bfp8": 1088 / 1024, "bf16": 2.0}
DT = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


def in0_cores(k_tiles, max_cores=64):
    for c in range(max_cores, 0, -1):
        if k_tiles % c == 0:
            for rows in range(1, 9):
                if c % rows == 0 and c // rows <= 8:
                    return c, rows, c // rows
    raise ValueError(k_tiles)


def bench(dev, K, N, dtype, workers, iters=50, check=False, bw_override=0, fid=None):
    banks = dev.dram_grid_size().x
    n_tiles = math.ceil(N / TILE)
    shard_w = math.ceil(math.ceil(n_tiles / banks) / workers) * workers
    Np = shard_w * banks * TILE
    w = torch.randn(K, N) * 0.02
    wp = torch.nn.functional.pad(w, (0, Np - N))
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    wmem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (K, shard_w * TILE), ttnn.ShardOrientation.ROW_MAJOR),
    )
    tw = ttnn.from_torch(wp, dtype=DT[dtype], layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wmem)

    k_tiles = K // TILE
    c, rows, cols = in0_cores(k_tiles)
    xmem = ttnn.create_sharded_memory_config(
        (TILE, K // c),
        ttnn.CoreGrid(y=rows, x=cols),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = torch.randn(1, 1, TILE, K)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=xmem)

    per_core_k = k_tiles // c
    # in0_block_w: largest divisor of per_core_k... keep it simple: per_core_k (one shard per block)
    in0_block_w = per_core_k
    while in0_block_w > 1 and (in0_block_w * shard_w // workers) > 2 * 64 * 4:
        # keep in1 CB modest
        for d in range(in0_block_w - 1, 0, -1):
            if per_core_k % d == 0:
                in0_block_w = d
                break
    if bw_override:
        in0_block_w = bw_override
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=1,
        per_core_N=math.ceil(Np / TILE / c),
        fused_activation=None,
        num_workers_per_dram_bank=workers,
    )
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=fid or (ttnn.MathFidelity.LoFi if dtype == "bfp4" else ttnn.MathFidelity.HiFi2),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    omem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    def run():
        return ttnn.linear(tx, tw, program_config=pc, memory_config=omem, compute_kernel_config=ck, dtype=ttnn.bfloat16)

    out = run()
    if check:
        ref = x[0, 0] @ wp
        got = ttnn.to_torch(out)[0, 0]
        pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten().float()]))[0, 1].item()
    else:
        pcc = float("nan")
    out.deallocate()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(iters):
        o = run()
        o.deallocate()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    dt = (time.perf_counter() - t0) / iters
    ttnn.release_trace(dev, tid)
    tw.deallocate()
    tx.deallocate()
    gb = K * Np * BYTES_PER_EL[dtype] / 1e9
    return dt, gb / dt, pcc, c, in0_block_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2")
    ap.add_argument("--dtypes", default="bfp4,bfp8")
    ap.add_argument("--only", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bw", default="0")
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    print("grid", dev.compute_with_storage_grid_size(), "dram banks", dev.dram_grid_size())
    try:
        for name, (K, N) in SHAPES.items():
            if a.only and name not in a.only.split(","):
                continue
            for dt in a.dtypes.split(","):
                for wk, bwo in [(w_, b_) for w_ in map(int, a.workers.split(",")) for b_ in map(int, a.bw.split(","))]:
                    try:
                        t, bw, pcc, c, bw0 = bench(dev, K, N, dt, wk, check=a.check, bw_override=bwo)
                        print(f"{name:12s} {K:6d}x{N:<6d} {dt:5s} w{wk} b{bwo}  {t*1e6:8.1f}us  {bw:6.1f}GB/s  pcc={pcc:.4f} in0cores={c} bw={bw0}", flush=True)
                    except Exception as e:
                        print(f"{name:12s} {dt} w{wk} FAILED: {str(e).splitlines()[0][:200]}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
