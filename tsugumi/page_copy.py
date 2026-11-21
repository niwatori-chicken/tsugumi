"""Launcher for kernels/page_copy.cpp: copy a page range between interleaved DRAM tensors (same page size).
Offsets live in a small device tensor, so a captured trace can be re-pointed by writing it."""
import os

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


class PageCopy:
    def __init__(self, dev, npages, page_bytes, ncores=16, nbuf=8):
        self.dev, self.n, self.pb, self.nbuf = dev, npages, page_bytes, nbuf
        grid = dev.compute_with_storage_grid_size()
        ncores = min(ncores, npages)
        self.coords = [(i % grid.x, i // grid.x) for i in range(ncores)]
        per = (npages + ncores - 1) // ncores
        self.ranges = [(min(i * per, npages), min((i + 1) * per, npages)) for i in range(ncores)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                                        for x, y in self.coords])
        self.cbs = [
            ttnn.CBDescriptor(total_size=nbuf * page_bytes, core_ranges=self.cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=page_bytes)]),
            ttnn.CBDescriptor(total_size=64, core_ranges=self.cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=64)]),
        ]
        self.off = ttnn.from_torch(torch.zeros(1, 16, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                   device=dev)

    def set_offsets(self, soff, doff):
        t = torch.zeros(1, 16, dtype=torch.int32)
        t[0, 0], t[0, 1] = soff, doff
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self.off)

    def __call__(self, src, dst):
        ins = [src, dst, self.off]
        ct = [self.pb, self.nbuf]
        for t in ins:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        rt = [((x, y), addrs + [lo, hi]) for (x, y), (lo, hi) in zip(self.coords, self.ranges)]
        kd = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "page_copy.cpp"),
                                   source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                   compile_time_args=ct, runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=[kd], cbs=self.cbs))
        return dst
