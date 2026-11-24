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
            ttnn.TensorAccessorArgs(self.sc).get_compile_time_args() + ttnn.TensorAccessorArgs(m).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        for c, (x, y) in enumerate(self.a_coords):
            src = h if c < NC // 2 else m
            t0 = (c % (NC // 2)) * KT
            r_rt.append(((x, y), [src.buffer_address(), D.buffer_address(), self.sc.buffer_address(), t0, c * KT, KT, RT,
                                  int(src is m)]))
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
            ttnn.TensorAccessorArgs(FG).get_compile_time_args() + ttnn.TensorAccessorArgs(h).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(m).get_compile_time_args()
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


class InkGate1:
    """The message gate in ONE program on NC cores (kernels/ink_gate_*.cpp): phase A's partials stay in each core's
    CB and are pulled (row 0 only) by every core after an in-program barrier, instead of a DRAM round trip and a
    second program. Same call as InkGate."""

    def __init__(self, dev, H=2304, R=64, eps=1e-6, NC=24, kinds=("attn", "k", "v", "mlp")):
        self.dev, self.H, self.R, self.eps = dev, H, R, eps
        self.HT, self.RT = H // 32, R // 32
        assert (2 * self.HT) % NC == 0 and NC % 2 == 0 and self.HT % NC == 0 and NC <= 32
        self.NC, self.KT, self.CPT = NC, 2 * self.HT // NC, self.HT // NC
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, i // grid.x) for i in range(NC)]
        W = lambda x, y: dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        self.noc = [W(x, y) for x, y in self.coords]
        bf, f32 = ttnn.bfloat16, ttnn.float32
        cc = _cores(self.coords)
        KT, RT, CPT = self.KT, self.RT, self.CPT
        self.cbs = [_cbd(cc, 0, KT, bf), _cbd(cc, 1, KT * RT, bf), _cbd(cc, 2, 1, bf), _cbd(cc, 3, KT, bf),
                    _cbd(cc, 4, RT + 1, f32), _cbd(cc, 5, 2 * RT * CPT, bf), _cbd(cc, 6, CPT, bf), _cbd(cc, 7, CPT, bf),
                    _cbd(cc, 8, RT + 1, f32), _cbd(cc, 9, 2, bf), _cbd(cc, 10, 2, f32), _cbd(cc, 11, RT, bf),
                    _cbd(cc, 12, 2, bf), _cbd(cc, 13, 2, f32), _cbd(cc, 16, 2, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
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
        cores = _cores(self.coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        r_ct = []
        for t in (h, m, D, FG, self.sc):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        nocxy = [v for n in self.noc for v in (n.x, n.y)]
        r_rt, c_rt, w_rt = [], [], []
        for c, xy in enumerate(self.coords):
            is_m = int(c >= NC // 2)
            t0 = (c % (NC // 2)) * KT
            j0 = c * CPT
            r_rt.append((xy, [h.buffer_address(), m.buffer_address(), D.buffer_address(), FG.buffer_address(),
                              self.sc.buffer_address(), c, t0, is_m, c * KT, KT, RT, j0, CPT, 2 * HT, HT, NC] + nocxy))
            c_rt.append((xy, [KT, RT, CPT, fbits(1.0 / self.H), fbits(self.eps)]))
            w_rt.append((xy, [out.buffer_address(), j0, CPT, RT, self.noc[0].x, self.noc[0].y]))
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=cores, initial_value=0)]
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_gate_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            K("ink_gate_writer.cpp", ttnn.TensorAccessorArgs(out).get_compile_time_args(), w_rt,
              ttnn.WriterConfigDescriptor()),
            K("ink_gate_compute.cpp", [], c_rt, self.ccfg),
        ], semaphores=sems, cbs=self.cbs)
        seen, ins = set(), []
        for t in (h, m, D, FG, self.sc):
            if id(t) not in seen:
                seen.add(id(t))
                ins.append(t)
        ttnn.generic_op(ins + [out], prog)
        return out
