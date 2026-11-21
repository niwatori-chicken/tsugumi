"""Test-only: a kernel that never finishes (kernels/hang.cpp), for checking hang detection and recovery."""
import os

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


def run_hang_kernel(dev, spin=True):
    """Enqueue the kernel on core (0, 0) and wait for the device. With spin=True the wait never ends normally."""
    x = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    y = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    kd = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "hang.cpp"),
                               source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core,
                               compile_time_args=[], runtime_args=[((0, 0), [int(spin), 0])],
                               config=ttnn.ReaderConfigDescriptor())
    ttnn.generic_op([x, y], ttnn.ProgramDescriptor(kernels=[kd], cbs=[]))
    ttnn.synchronize_device(dev)
