"""Inkling decode MoE: the softcapped SwiGLU of the gate_up output in one program (kernels/ink_act_*.cpp):
act = tanh(g / gc) * sigmoid(g) * tanh(u / uc), g / u = the two column halves of each expert slot's output; row 0
of every tile only (the token), gathered 32 tiles to a dense tile per core. Replaces 2 slices + 2 multiplies.

    act = InkAct(dev, K=8, I=1152, gc=4, uc=25)
    a = act(gu)   # gu [.., K, 32, 2I] (L1) -> persistent [1, K, 32, I] (L1)
"""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class InkAct:
    def __init__(self, dev, K, I, gc, uc):
        self.dev, self.IT, self.gc, self.uc = dev, I // 32, gc, uc
        self.total = K * self.IT
        ncores = (self.total + 31) // 32
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, i // grid.x) for i in range(ncores)]
        cores = _cores(self.coords)
        bf = ttnn.bfloat16
        self.cbs = [_cbd(cores, 0, 1, bf), _cbd(cores, 1, 1, bf), _cbd(cores, 16, 1, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
        self.out = ttnn.from_torch(torch.zeros(1, K, 32, I), dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)

    def __call__(self, gu):
        cores = _cores(self.coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        r_rt, c_rt, w_rt = [], [], []
        for i, xy in enumerate(self.coords):
            o0 = i * 32
            n = min(32, self.total - o0)
            r_rt.append((xy, [gu.buffer_address(), o0, n, self.IT]))
            c_rt.append((xy, [1, fbits(1.0 / self.gc), fbits(1.0 / self.uc)]))
            w_rt.append((xy, [self.out.buffer_address(), o0, n]))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_act_reader.cpp", ttnn.TensorAccessorArgs(gu).get_compile_time_args(), r_rt,
              ttnn.ReaderConfigDescriptor()),
            K("ink_act_writer.cpp", ttnn.TensorAccessorArgs(self.out).get_compile_time_args(), w_rt,
              ttnn.WriterConfigDescriptor()),
            K("ink_act_compute.cpp", [], c_rt, self.ccfg),
        ], cbs=self.cbs)
        ttnn.generic_op([gu, self.out], prog)
        return self.out


class InkOGate:
    """Attention output gate of one token (kernels/ink_og_*.cpp): out = o * sigmoid(g), row 0 of every tile, with g
    a column range of another tensor (the q linear's output). Rows gathered NR tiles to a dense tile per core.
    Written straight into `out` (e.g. the o projection's width-sharded input): replaces slice + multiply (+ the
    linear's input conversion).

    og = InkOGate(dev, n_tiles=96)
    og(o, g, goff, out)   # o [1,1,32,N] (L1), g (L1) with the gate at column tile goff, out [1,1,32,N]
    """

    def __init__(self, dev, n_tiles, NR=8):
        self.dev, self.n, self.NR = dev, n_tiles, NR
        ncores = (n_tiles + NR - 1) // NR
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, i // grid.x) for i in range(ncores)]
        cores = _cores(self.coords)
        bf = ttnn.bfloat16
        self.cbs = [_cbd(cores, 0, 1, bf), _cbd(cores, 1, 1, bf), _cbd(cores, 2, NR, bf), _cbd(cores, 16, 1, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c

    def __call__(self, o, g, goff, out):
        cores = _cores(self.coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        r_rt, w_rt = [], []
        for i, xy in enumerate(self.coords):
            o0 = i * self.NR
            n = min(self.NR, self.n - o0)
            r_rt.append((xy, [o.buffer_address(), g.buffer_address(), o0, n, goff]))
            w_rt.append((xy, [out.buffer_address(), o0, n]))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_og_reader.cpp", ttnn.TensorAccessorArgs(o).get_compile_time_args() +
              ttnn.TensorAccessorArgs(g).get_compile_time_args(), r_rt, ttnn.ReaderConfigDescriptor()),
            K("ink_og_writer.cpp", ttnn.TensorAccessorArgs(out).get_compile_time_args(), w_rt,
              ttnn.WriterConfigDescriptor()),
            K("ink_og_compute.cpp", [], [(xy, []) for xy in self.coords], self.ccfg),
        ], cbs=self.cbs)
        ins = [o] + ([g] if g is not o else []) + [out]
        ttnn.generic_op(ins, prog)
        return out
