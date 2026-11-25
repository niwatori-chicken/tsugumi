"""Launcher for kernels/ink_mask.cpp: the prefill attention mask built by selection on every core."""
import os

import ttnn

from ..gqa_attn import _cbd

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


class MaskBuilder:
    """Launcher for kernels/ink_mask.cpp: the prefill attention mask by selection (relative bias inside the window,
    0 / NEG outside, the sink logit on its column), on every core with both data-movement RISC-Vs."""

    def __init__(self, dev):
        self.dev = dev
        g = dev.compute_with_storage_grid_size()
        self.coords = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})

    def __call__(self, rel, sink, base, mode, ow, W, sc, M=0, m0=0, rows=None, mem=ttnn.DRAM_MEMORY_CONFIG):
        """rel [1, NH, T_in, L] (bias by distance, columns reversed), sink [1, NH, T_in, 32] (column 0), base int32
        [1, 8] ([0] p0 % R, [2] p0). mode 0: ring slots (distance mod M), 1: positions. Returns [1, NH, rows, ow]."""
        _, NH, Ti, L = rel.shape
        rows = rows or Ti
        LT, OT, MTi, MTo = L // 32, ow // 32, Ti // 32, rows // 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, NH, rows, ow]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.dev, mem)
        nu, nr = NH * MTo, 2 * len(self.coords)
        bf = ttnn.bfloat16
        cbs = [_cbd(self.cores, 0, LT + 1, bf), _cbd(self.cores, 1, 2, bf), _cbd(self.cores, 2, LT + 1, bf),
               _cbd(self.cores, 3, 2, bf)]
        acc = []
        for t in (rel, sink, out, base):
            acc += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        kds = []
        for which, (cb0, cfg) in enumerate(((0, ttnn.ReaderConfigDescriptor()), (2, ttnn.WriterConfigDescriptor()))):
            rt = [(xy, [rel.buffer_address(), sink.buffer_address(), out.buffer_address(), base.buffer_address(), m0,
                        MTi, MTo, 2 * i + which, nr, nu, M, W, sc]) for i, xy in enumerate(self.coords)]
            kds.append(ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "ink_mask.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                compile_time_args=[LT, OT, cb0, mode] + acc, runtime_args=rt, config=cfg))
        ttnn.generic_op([rel, sink, out, base], ttnn.ProgramDescriptor(kernels=kds, cbs=cbs))
        return out
