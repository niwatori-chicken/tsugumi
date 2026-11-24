"""Inkling decode: short conv + residual of one token in one program (kernels/ink_conv_*.cpp), split over column
tiles: out = conv(x; hist, taps) [+ b], hist shifted in place. Replaces the 5 ops of Engine.conv and the residual add.

    conv = InkConv(dev, C=2304)
    out = conv(x, hist, taps, b, out)   # x, b: [1,1,32,C] row 0 (L1); hist, taps [1,1,32,C] (DRAM); out: L1 buffer
    out = conv(dn, hist, taps, b, out, wcol)   # MoE: conv input = sum_k wcol_k dn_k (the expert combine)
"""
import os

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


class InkConv:
    def __init__(self, dev, C, ncores=24, k=4):
        self.dev, self.CT = dev, C // 32
        assert self.CT % ncores == 0
        self.NT = self.CT // ncores
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, grid.y - 2 - i // grid.x) for i in range(ncores)]
        cores = _cores(self.coords)
        bf = ttnn.bfloat16
        NT = self.NT
        self.cbs = [_cbd(cores, 0, NT, bf), _cbd(cores, 1, NT, bf), _cbd(cores, 2, NT, bf), _cbd(cores, 3, NT, bf),
                    _cbd(cores, 4, 3, bf), _cbd(cores, 5, 1, bf), _cbd(cores, 6, 1, bf), _cbd(cores, 16, 2, bf),
                    _cbd(cores, 17, 2, bf), _cbd(cores, 7, 1, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
        m = torch.zeros(1, 3, 32, 32)
        m[0, 0, 0, 0] = 1.0  # E0
        m[0, 1, 0, :k] = 1.0  # ONES0
        for r in range(k - 1):
            m[0, 2, r + 1, r] = 1.0  # SHIFT
        self.consts = ttnn.from_torch(m, dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, x, hist, taps, b, out, wcol=None):
        """wcol (MoE mode): x = the down projections [1, K, 32, C] (L1), wcol [1, K, 1, 1] bf16 tiles (L1, value at
        [0, 0]): the conv input is sum_k wcol_k x_k."""
        cores = _cores(self.coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        bb = b if b is not None else x
        r_ct = []
        for t in (x, hist, taps, bb, self.consts, wcol if wcol is not None else x):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        moe = 0 if wcol is None else x.shape[-3]
        w_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args() + ttnn.TensorAccessorArgs(hist).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        for i, (cx, cy) in enumerate(self.coords):
            j0 = i * self.NT
            r_rt.append(((cx, cy), [x.buffer_address(), hist.buffer_address(), taps.buffer_address(), bb.buffer_address(),
                                    self.consts.buffer_address(), j0, self.NT, int(b is not None), moe,
                                    (wcol if wcol is not None else x).buffer_address(), self.CT]))
            c_rt.append(((cx, cy), [self.NT, int(b is not None)]))
            w_rt.append(((cx, cy), [out.buffer_address(), hist.buffer_address(), j0, self.NT]))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_conv_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            K("ink_conv_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
            K("ink_conv_compute.cpp", [], c_rt, self.ccfg),
        ], cbs=self.cbs)
        ins = [x, hist, taps] + ([b] if b is not None else []) + [self.consts] + ([wcol] if wcol is not None else [])
        ttnn.generic_op(ins + [out], prog)
        return out
