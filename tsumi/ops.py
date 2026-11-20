"""Building blocks for the lean B<=32 decode engine on a single P100A.

DecodeLinear: weight [K, N] stored DRAM WIDTH_SHARDED over the 7 banks (padded so that
`workers` readers per bank each get an equal slice), activation [1,1,M<=32,K] L1 width
sharded, DRAM-sharded matmul (measured ~415 GB/s for bfp4 at workers=3, in0_block_w=2).
"""
import math
import os

import torch
import ttnn

TILE = 32


def _in0_grid(k_tiles, max_cores=64):
    for c in range(max_cores, 0, -1):
        if k_tiles % c == 0:
            for rows in range(1, 9):
                if c % rows == 0 and c // rows <= 8:
                    return c, rows, c // rows
    raise ValueError(k_tiles)


def compute_cfg(fidelity=ttnn.MathFidelity.LoFi, fp32_acc=False, approx=False, l1_acc=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=approx, fp32_dest_acc_en=fp32_acc, packer_l1_acc=l1_acc
    )


def cache_path(cache_file, dtype):
    return f"{cache_file}.{str(dtype).split('.')[-1]}.tensorbin"


def has_cache(cache_file, dtype):
    return cache_file is not None and os.path.exists(cache_path(cache_file, dtype))


def load_or_convert(dev, w, dtype, mem, cache_file=None, pad_n=None):
    """Weight tensor on device, cached as a .tensorbin (keeps the memory config)."""
    if has_cache(cache_file, dtype):
        return ttnn.to_device(ttnn.load_tensor(cache_path(cache_file, dtype)), dev, memory_config=mem)
    assert w is not None, f"no weight and no cache for {cache_file}"
    if pad_n is not None and pad_n != w.shape[-1]:
        w = torch.nn.functional.pad(w, (0, pad_n - w.shape[-1]))
    t = ttnn.from_torch(w.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        ttnn.dump_tensor(cache_path(cache_file, dtype), t)
    return ttnn.to_device(t, dev, memory_config=mem)


class DecodeLinear:
    """y = x @ W for decode (M <= 32 rows). W given as torch [K, N] (ttnn orientation)."""

    def __init__(self, dev, w_kn, dtype, workers=None, in0_block_w=None, fidelity=None, cache_file=None, name="",
                 shape=None, no_weight=False):
        K, N = w_kn.shape if w_kn is not None else shape
        self.K, self.N, self.name = K, N, name
        banks = dev.dram_grid_size().x
        if workers is None:
            workers = 3 if K <= 5120 else 2  # measured best on P100 (lowbit/mmbench.py)
        n_tiles = math.ceil(N / TILE)
        shard_w = math.ceil(math.ceil(n_tiles / banks) / workers) * workers
        self.Np = shard_w * banks * TILE
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        self.wmem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (K, shard_w * TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.w = None if no_weight else load_or_convert(dev, w_kn, dtype, self.wmem, cache_file, pad_n=self.Np)
        k_tiles = K // TILE
        c, rows, cols = _in0_grid(k_tiles)
        self.in0_cores = c
        self.xmem = ttnn.create_sharded_memory_config(
            (TILE, K // c),
            ttnn.CoreGrid(y=rows, x=cols),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        per_core_k = k_tiles // c
        if in0_block_w is None:
            in0_block_w = 2 if per_core_k % 2 == 0 else per_core_k
        self.pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=math.ceil(self.Np / TILE / c),
            fused_activation=None,
            num_workers_per_dram_bank=workers,
        )
        if fidelity is None:
            fidelity = ttnn.MathFidelity.LoFi if dtype == ttnn.bfloat4_b else ttnn.MathFidelity.HiFi2
        self.ck = compute_cfg(fidelity)

    def __call__(self, x, out_mem=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16):
        """x: [1,1,32,K] tile layout, any memory config. Returns [1,1,32,Np] in out_mem."""
        xs = x if x.memory_config() == self.xmem else ttnn.to_memory_config(x, self.xmem)
        y = ttnn.linear(
            xs,
            self.w,
            program_config=self.pc,
            compute_kernel_config=self.ck,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=dtype,
        )
        if xs is not x:
            ttnn.deallocate(xs)
        if out_mem is not None:
            y2 = ttnn.to_memory_config(y, out_mem)
            ttnn.deallocate(y)
            y = y2
        return y
