"""Launcher for kernels/ink_relshift.cpp: the relative bias by distance -> by cache slot (a per-row circular shift),
on every core with both data-movement RISC-Vs."""
import os

import ttnn

from ..gqa_attn import _cbd

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


class RelShift:
    def __init__(self, dev):
        self.dev = dev
        g = dev.compute_with_storage_grid_size()
        self.coords = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})

    def __call__(self, rel, base, k, ow, m0=0, rows=None, mem=ttnn.DRAM_MEMORY_CONFIG, mod=None, dbg=0):
        """rel [1, NH, T_in, L] bf16 tiles, columns REVERSED (rel[.., y] = the bias at distance L - 1 - y); base: int32 row-major [1, 8] (base = base[k]); returns [1, NH, rows, ow]
        (rows m0 * 32 .. + rows of rel): out[h, t, s] = rel[h, t, (base + t - s) mod L] (s < L), rel[.., L - 1] beyond."""
        _, NH, Ti, L = rel.shape
        rows = rows or Ti
        mod = mod or L  # out[h, t, s] = rel[h, t, (base + t - s) mod M] (s < M; columns >= L and slots >= M: rel[.., L - 1])
        LT, OT, MTi, MTo = L // 32, ow // 32, Ti // 32, rows // 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, NH, rows, ow]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.dev, mem)
        nu, nr = NH * MTo, 2 * len(self.coords)
        bf = ttnn.bfloat16
        cbs = [_cbd(self.cores, 0, LT, bf), _cbd(self.cores, 1, 2, bf), _cbd(self.cores, 2, LT, bf),
               _cbd(self.cores, 3, 2, bf)]
        acc = []
        for t in (rel, out, base):
            acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        kds = []
        for which, (cb0, cfg) in enumerate(((0, ttnn.ReaderConfigDescriptor()), (2, ttnn.WriterConfigDescriptor()))):
            rt = [(xy, [rel.buffer_address(), out.buffer_address(), base.buffer_address(), k, m0, MTi, MTo,
                        2 * i + which, nr, nu, mod, dbg]) for i, xy in enumerate(self.coords)]
            kds.append(ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "ink_relshift.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                compile_time_args=[LT, OT, cb0] + acc, runtime_args=rt, config=cfg))
        ttnn.generic_op([rel, out, base], ttnn.ProgramDescriptor(kernels=kds, cbs=cbs))
        return out
