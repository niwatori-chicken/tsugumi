"""Inkling decode MoE routing in one program (kernels/ink_route_*.cpp, one core): sigmoid of the router logits,
+ the expert bias, the top-K experts (ties: lowest id), weights s / sum(s) * kscale. Replaces sigmoid / add / topk /
gather / normalization / typecasts / layout changes (~110 us of ops per layer application).

    route = InkRoute(dev, E=128, K=8, kscale=route_scale * softcaps)
    ids, wcol = route(logits, bias)   # ids uint16 row-major [1,1,1,K] (L1), wcol bf16 [1,K,1,1] (L1)
      logits [1,1,1,E] float32 (L1 or DRAM), bias [1,1,1,E] float32 (DRAM)
"""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


class InkRoute:
    def __init__(self, dev, E=128, K=8, kscale=1.0):
        self.dev, self.E, self.K, self.kscale = dev, E, K, kscale
        self.nt = E // 32
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(4, grid.y - 1)]
        cores = _cores(self.coords)
        f32 = ttnn.float32
        self.cbs = [_cbd(cores, 0, self.nt, f32), _cbd(cores, 1, self.nt, f32), _cbd(cores, 2, self.nt, f32),
                    _cbd(cores, 3, self.nt, f32), _cbd(cores, 4, 1, ttnn.bfloat16)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
        L1 = ttnn.L1_MEMORY_CONFIG
        self.ids = ttnn.from_torch(torch.zeros(1, 1, 1, K, dtype=torch.int32), dtype=ttnn.uint16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=L1)
        self.wcol = ttnn.from_torch(torch.zeros(1, K, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=L1)

    def __call__(self, logits, bias):
        cores = _cores(self.coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        (x, y), = self.coords
        kb = struct.unpack("<I", struct.pack("<f", self.kscale))[0]
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_route_reader.cpp", ttnn.TensorAccessorArgs(logits).get_compile_time_args() +
              ttnn.TensorAccessorArgs(bias).get_compile_time_args(),
              [((x, y), [logits.buffer_address(), bias.buffer_address(), self.nt])], ttnn.ReaderConfigDescriptor()),
            K("ink_route_writer.cpp", ttnn.TensorAccessorArgs(self.ids).get_compile_time_args() +
              ttnn.TensorAccessorArgs(self.wcol).get_compile_time_args(),
              [((x, y), [self.ids.buffer_address(), self.wcol.buffer_address(), self.nt, self.K, kb])],
              ttnn.WriterConfigDescriptor()),
            K("ink_route_compute.cpp", [], [((x, y), [self.nt])], self.ccfg),
        ], cbs=self.cbs)
        ttnn.generic_op([logits, bias, self.ids, self.wcol], prog)
        return self.ids, self.wcol
