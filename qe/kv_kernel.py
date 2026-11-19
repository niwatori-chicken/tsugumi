"""Launcher for kernels/kv_write.cpp: write T K/V rows into the paged bf16 cache (identity page table)."""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


class KVWriteKernel:
    def __init__(self, dev, T, NKV, HD, block_size):
        self.dev, self.T, self.NKV, self.HDT, self.BS = dev, T, NKV, HD // 32, block_size
        grid = dev.compute_with_storage_grid_size()
        n = NKV * self.HDT
        self.assign = [(i // self.HDT, i % self.HDT) for i in range(n)]
        # use cores away from (0,0).. to not matter; any free core works
        self.coords = [(i % grid.x, grid.y - 1 - i // grid.x) for i in range(n)]
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in self.coords])
        self.cbs = [ttnn.CBDescriptor(total_size=3 * 2048, core_ranges=self.cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])]
        self.posb = ((T * 4 + 31) // 32) * 32

    def __call__(self, ksrc, vsrc, kc, vc, pos):
        ins = [ksrc, vsrc, kc, vc, pos]
        ct = [self.T, self.NKV, self.HDT, self.BS, self.posb]
        for t in ins:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        rt = [((x, y), addrs + [h, c]) for (x, y), (h, c) in zip(self.coords, self.assign)]
        kd = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "kv_write.cpp"),
                                   source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                   compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=[kd], cbs=self.cbs))
