"""Inkling message gate of one token in two programs (kernels/ink_gz_*.cpp, ink_go_*.cpp):
  A (NC cores, split over the 2H inputs): z parts = [h | m] tiles @ D (= down_weight^T) and sums of squares
  B (NB cores, split over the H outputs): z = z_h / rms(h) + z_m / rms(m); f, g = tanh(z @ [F | G]) for its
    columns; out = (1 - f) h + g m
Replaces ~13 ops (rms norms, concat, the row dot, the 64 -> 2H matmul, slices, tanh multiplies).

    gate = InkGate(dev, H=2304, R=64)
    out = gate(h, m, D, FG, kind)   # h, m [1,1,32,H] (row 0, L1); D [1,1,2H,R], FG [1,1,R,2H] (DRAM)
"""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class InkGate:
    def __init__(self, dev, H=2304, R=64, eps=1e-6, NC=18, NB=24, kinds=("attn", "k", "v", "mlp")):
        self.dev, self.H, self.R, self.eps = dev, H, R, eps
        self.HT, self.RT = H // 32, R // 32
        assert (2 * self.HT) % NC == 0 and NC % 2 == 0 and self.HT % NB == 0
        self.NC, self.NB, self.KT, self.CPT = NC, NB, 2 * self.HT // NC, self.HT // NB
        grid = dev.compute_with_storage_grid_size()
        self.a_coords = [(i % grid.x, i // grid.x) for i in range(NC)]
        self.b_coords = [(i % grid.x, i // grid.x) for i in range(NB)]
        bf, f32 = ttnn.bfloat16, ttnn.float32
        ca, cb = _cores(self.a_coords), _cores(self.b_coords)
        KT, RT, CPT = self.KT, self.RT, self.CPT
        self.a_cbs = [_cbd(ca, 0, KT, bf), _cbd(ca, 1, KT * RT, bf), _cbd(ca, 2, 1, bf), _cbd(ca, 3, KT, bf),
                      _cbd(ca, 16, 2, f32)]
        self.b_cbs = [_cbd(cb, 0, NC * (RT + 1), f32), _cbd(cb, 1, 2 * RT * CPT, bf), _cbd(cb, 2, CPT, bf),
                      _cbd(cb, 3, CPT, bf), _cbd(cb, 4, 2 * (RT + 1), f32), _cbd(cb, 5, 2, f32), _cbd(cb, 6, RT, bf),
                      _cbd(cb, 7, 2, f32), _cbd(cb, 16, 2, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
        self.part = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32 * NC * (RT + 1), 32]), f32, ttnn.TILE_LAYOUT,
                                                   dev, ttnn.DRAM_MEMORY_CONFIG)
        # DRAM: whole tiles only, consumed by rms norms (L1 is tight for the prefill programs)
        self.out = {k: ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG) for k in kinds}
        sc = torch.zeros(32, 32)
        sc[0, :] = 1.0
        sc[16, :] = 1.0
        self.sc = ttnn.from_torch(sc, dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, h, m, D, FG, kind):
        out = self.out[kind]
        KT, RT, CPT, NC, HT = self.KT, self.RT, self.CPT, self.NC, self.HT
        K = lambda src, ct, rt, cfg, cores: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        ca = _cores(self.a_coords)
        r_ct = ttnn.TensorAccessorArgs(h).get_compile_time_args() + ttnn.TensorAccessorArgs(D).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(self.sc).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        for c, (x, y) in enumerate(self.a_coords):
            src = h if c < NC // 2 else m
            t0 = (c % (NC // 2)) * KT
            r_rt.append(((x, y), [src.buffer_address(), D.buffer_address(), self.sc.buffer_address(), t0, c * KT, KT, RT]))
            c_rt.append(((x, y), [KT, RT]))
            w_rt.append(((x, y), [self.part.buffer_address(), c, RT]))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_gz_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor(), ca),
            K("ink_gz_writer.cpp", ttnn.TensorAccessorArgs(self.part).get_compile_time_args(), w_rt,
              ttnn.WriterConfigDescriptor(), ca),
            K("ink_gz_compute.cpp", [], c_rt, self.ccfg, ca),
        ], cbs=self.a_cbs)
        ttnn.generic_op([h, m, D, self.sc, self.part], prog)
        cb = _cores(self.b_coords)
        r_ct = ttnn.TensorAccessorArgs(self.part).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(FG).get_compile_time_args() + ttnn.TensorAccessorArgs(h).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        for b, (x, y) in enumerate(self.b_coords):
            j0 = b * CPT
            r_rt.append(((x, y), [self.part.buffer_address(), FG.buffer_address(), h.buffer_address(),
                                  m.buffer_address(), NC, RT, j0, CPT, 2 * HT, HT]))
            c_rt.append(((x, y), [NC, RT, CPT, fbits(1.0 / self.H), fbits(self.eps)]))
            w_rt.append(((x, y), [out.buffer_address(), j0, CPT]))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_go_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor(), cb),
            K("ink_go_writer.cpp", ttnn.TensorAccessorArgs(out).get_compile_time_args(), w_rt,
              ttnn.WriterConfigDescriptor(), cb),
            K("ink_go_compute.cpp", [], c_rt, self.ccfg, cb),
        ], cbs=self.b_cbs)
        ttnn.generic_op([self.part, FG, h, m, out], prog)
        return out
