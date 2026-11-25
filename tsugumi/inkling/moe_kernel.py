"""Inkling decode MoE of one token in one program (kernels/ink_moe_*.cpp): the 8 routed experts' gate_up, the
softcapped SwiGLU, down and the weighted combine on 36 cores, with one in-program barrier (the act tiles are
shared through a small L1 buffer). The expert weights are kept in a bank-contiguous DRAM layout
(kernels/ink_moe_reorder.cpp, built on the device from the interleaved tensors) so that each core streams long
contiguous reads from one bank at a time (~415 GB/s vs ~320 GB/s for sparse_matmul on interleaved weights).

    moe = InkMoE(dev, E=128, H=2304, I=1152, gc=4, uc=25)
    gu, dn = moe.prepare(Egu, Edn)          # interleaved [1, E, H, 2I] / [1, E, I, H] bfp8 -> layout tensors
    out = moe(x, ids, wcol, gu, dn)         # x [1,1,32,H] (row 0, L1); ids uint16 [1,1,1,8]; wcol [1,8,1,1]
"""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")
PLACEMENTS = {  # 36 cores (logical (x, y)); core c streams from DRAM banks c % 4 and c % 4 + 4
    # near*: per bank pair, the cores with the fastest single-core reads on P150A (at most * per row)
    "near11": [(2, 9), (1, 1), (1, 6), (2, 3), (0, 9), (0, 1), (3, 7), (0, 4), (3, 9), (2, 1), (2, 6), (8, 4), (1, 9),
               (3, 1), (0, 6), (9, 4), (5, 9), (5, 1), (4, 6), (6, 4), (9, 9), (7, 1), (3, 6), (3, 3), (8, 9), (10, 1),
               (5, 6), (4, 3), (6, 9), (6, 1), (7, 6), (8, 3), (7, 9), (8, 1), (0, 7), (3, 4)],
    "near5": [(2, 9), (5, 1), (1, 6), (2, 3), (0, 9), (7, 1), (3, 7), (0, 4), (3, 9), (10, 1), (2, 6), (8, 4), (1, 9),
              (6, 1), (0, 6), (9, 4), (5, 9), (8, 1), (4, 6), (6, 4), (0, 1), (3, 0), (3, 6), (3, 3), (1, 1), (2, 0),
              (0, 7), (4, 3), (3, 1), (5, 0), (1, 7), (8, 3), (2, 1), (9, 0), (5, 7), (3, 4)],
    "near3": [(2, 9), (2, 1), (1, 6), (2, 3), (0, 9), (5, 1), (3, 7), (0, 4), (3, 9), (7, 1), (2, 6), (8, 4), (0, 1),
              (3, 0), (4, 6), (9, 4), (1, 1), (2, 0), (0, 7), (3, 3), (3, 1), (5, 0), (1, 7), (4, 3), (1, 5), (5, 2),
              (6, 1), (10, 1), (4, 4), (0, 3), (9, 1), (4, 1), (5, 3), (0, 6), (8, 1), (2, 2)],
    "rect6x6": [(c % 6, c // 6) for c in range(36)],
    "rect9x4": [(c % 9, c // 9) for c in range(36)],
    "spread": [((c * 3) % 11, (c * 3) // 11) for c in range(36)],
    "cols": [(x, y) for x in (0, 2, 4, 6, 8, 10) for y in (0, 2, 3, 5, 7, 9)],
    "stride2": [((c * 2) % 11, (c * 2) // 11 * 1) for c in range(36)],
    "stride3b": [((c * 3 + 1) % 11, (c * 3 + 1) // 11) for c in range(36)],
    "stride3c": [((c * 3) % 11, 9 - (c * 3) // 11) for c in range(36)],
    "checker": [(x, y) for y in range(10) for x in range(11) if (x + y) % 3 == 0][:36],
}


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class InkMoE:
    def __init__(self, dev, E, H, I, gc, uc, K=8, placement=None, ring=(12, 4, 3), NB=4, wdtype=ttnn.bfloat8_b):
        assert (H, I, K) == (2304, 1152, 8)
        self.dev, self.E, self.H, self.I, self.gc, self.uc = dev, E, H, I, gc, uc
        self.ring = ring  # weight blocks: tiles per block, blocks in the CB, blocks in flight
        assert 36 % ring[0] == 0 and (ring[0] * ring[1]) % 24 == 0 and ring[2] < ring[1]
        self.coords = PLACEMENTS[placement or os.environ.get("TSUGUMI_MOE_PLACEMENT", "stride2")]
        self.NC = len(self.coords)
        assert self.NC == I // 32 and len(set(self.coords)) == self.NC
        self.wdtype = wdtype
        self.pb = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[wdtype]  # bytes per weight tile
        cores = _cores(self.coords)
        bf, b8 = ttnn.bfloat16, wdtype
        self.cbs = [_cbd(cores, 0, 72, bf), _cbd(cores, 1, ring[0] * ring[1], b8), _cbd(cores, 2, 36, bf), _cbd(cores, 3, K, bf),
                    _cbd(cores, 4, 2, bf), _cbd(cores, 5, 2, bf), _cbd(cores, 7, 1, bf), _cbd(cores, 8, 1, bf),
                    _cbd(cores, 9, 1, bf), _cbd(cores, 10, 1, bf), _cbd(cores, 11, 2 * K, bf),
                    _cbd(cores, 16, 2, bf)]
        c = ttnn.ComputeConfigDescriptor()
        # as the sparse matmuls: HiFi2 covers bfp8's mantissa, bfp4 needs only LoFi
        c.math_fidelity = ttnn.MathFidelity.HiFi2 if wdtype == ttnn.bfloat8_b else ttnn.MathFidelity.LoFi
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c

        f32 = ttnn.float32
        self.NB = NB  # prefill: experts per batch (one barrier per batch)
        # prefill CBs for MT row tiles (32 tokens each)
        self.pcbs = lambda MT: [
            _cbd(cores, 0, 72 * MT, bf), _cbd(cores, 1, ring[0] * ring[1], b8), _cbd(cores, 2, 72 * MT, bf),
            _cbd(cores, 4, MT, bf), _cbd(cores, 5, MT, bf), _cbd(cores, 6, 4 * MT, bf), _cbd(cores, 7, 1, bf),
            _cbd(cores, 10, 1, bf), _cbd(cores, 11, 1, bf), _cbd(cores, 12, 1, bf),
            _cbd(cores, 13, 2 * NB * MT, bf), _cbd(cores, 14, 3 * MT, bf), _cbd(cores, 15, 2 * MT, f32),
            _cbd(cores, 16, 2 * MT, bf)]
        L1 = ttnn.L1_MEMORY_CONFIG
        z = lambda w: ttnn.from_torch(torch.zeros(1, 1, 32, w), dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                      memory_config=L1)
        self.gather = z(I)
        self.out = z(H)
        W = lambda x, y: dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        self.noc = [W(x, y) for x, y in self.coords]

    def _layout(self, T):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                ttnn.ShardSpec(grid, (32, 32 * T), ttnn.ShardOrientation.ROW_MAJOR))
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32 * T * 8]), self.wdtype, ttnn.TILE_LAYOUT,
                                              self.dev, mem)

    def _reorder(self, src, kind):
        T = 72 if kind == 0 else 36
        dst = self._layout(self.E * 9 * T)
        grid = self.dev.compute_with_storage_grid_size()
        coords = [(x, y) for y in range(grid.y) for x in range(grid.x)][:64]
        cores = _cores(coords)
        items = self.E * 72
        rt = [(xy, [src.buffer_address(), dst.buffer_address(), kind, i, len(coords), items, self.pb])
              for i, xy in enumerate(coords)]
        k = ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "ink_moe_reorder.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
                                  compile_time_args=ttnn.TensorAccessorArgs(src).get_compile_time_args(),
                                  runtime_args=rt, config=ttnn.ReaderConfigDescriptor())
        p = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cbd(cores, 0, T, self.wdtype)])
        ttnn.generic_op([src, dst], p)
        return dst

    def prepare(self, Egu, Edn):
        assert Egu.dtype == self.wdtype and Edn.dtype == self.wdtype
        return self._reorder(Egu, 0), self._reorder(Edn, 1)

    def __call__(self, x, ids, wcol, gu, dn, stage=3):
        cores = _cores(self.coords)
        Kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        r_ct = list(self.ring)
        for t in (x, ids, wcol, self.gather):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = ttnn.TensorAccessorArgs(self.gather).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(self.out).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        nocxy = [v for n in self.noc for v in (n.x, n.y)]
        for c, xy in enumerate(self.coords):
            r_rt.append((xy, [x.buffer_address(), ids.buffer_address(), wcol.buffer_address(), gu.buffer_address(),
                              dn.buffer_address(), c, self.gather.buffer_address(), self.pb, stage, self.NC] + nocxy))
            c_rt.append((xy, [fbits(1.0 / self.gc), fbits(1.0 / self.uc), stage, (c // 4) % 2]))
            w_rt.append((xy, [self.gather.buffer_address(), self.out.buffer_address(), c, self.noc[0].x, self.noc[0].y,
                              self.NC, 0, 0, 0, 0, 0, stage]))
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=cores, initial_value=0)]
        prog = ttnn.ProgramDescriptor(kernels=[
            Kd("ink_moe_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            Kd("ink_moe_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
            Kd("ink_moe_compute.cpp", [], c_rt, self.ccfg),
        ], semaphores=sems, cbs=self.cbs)
        ttnn.generic_op([x, ids, wcol, gu, dn, self.gather, self.out], prog)
        return self.out

    def prefill(self, x, route, used, gu, dn, out, dbg=0):
        """T = 32 MT token rows: x [1,1,T,H]; route [1,1,T,E] bf16 tiles (dense routing weights, 0 = not routed); used
        row-major bf16 [1,1,1,E] (max over the tokens); out [1,1,T,H] (L1 or DRAM)."""
        cores = _cores(self.coords)
        Kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        MT = x.shape[-2] // 32
        assert x.shape[-2] == 32 * MT and 1 <= MT <= 2  # 2 MT float32 DST tiles
        r_ct = list(self.ring) + [MT]
        for t in (x, route, used):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        nocxy = [v for n in self.noc for v in (n.x, n.y)]
        r_rt, c_rt, w_rt = [], [], []
        for c, xy in enumerate(self.coords):
            r_rt.append((xy, [x.buffer_address(), route.buffer_address(), used.buffer_address(), gu.buffer_address(),
                              dn.buffer_address(), c, self.pb, self.NB, self.NC] + nocxy + [dbg]))
            c_rt.append((xy, [fbits(1.0 / self.gc), fbits(1.0 / self.uc), (c // 4) % 2, self.NB, MT, dbg]))
            w_rt.append((xy, [out.buffer_address(), c, self.noc[0].x, self.noc[0].y, self.NB, MT]))
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=cores, initial_value=0)]
        kc = Kd("ink_moep_compute.cpp", [], c_rt, self.ccfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            Kd("ink_moep_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            Kd("ink_moep_writer.cpp", ttnn.TensorAccessorArgs(out).get_compile_time_args(), w_rt,
               ttnn.WriterConfigDescriptor()),
            kc,
        ], semaphores=sems, cbs=self.pcbs(MT))
        ttnn.generic_op([x, route, used, gu, dn, out], prog)
        return out

    def prefill_compact(self, x, route, used, gu, dn, out, dbg=0):
        """As prefill, with each expert's routed tokens packed into compact row tiles (kernels/ink_moec_*): the act,
        the act-tile exchange and the down projection run on ceil(n_e / 32) tiles instead of T / 32 (an expert
        takes ~5 of 64 tokens), the rows go back to their tokens with 0/1 placement matmuls."""
        cores = _cores(self.coords)
        MT = x.shape[-2] // 32
        assert x.shape[-2] == 32 * MT and 1 <= MT <= 2
        bf, f32 = ttnn.bfloat16, ttnn.float32
        NB, ring = self.NB, self.ring
        ls = ((self.E * (32 * MT + 1) + 31) // 32) * 32
        cbs = [_cbd(cores, 0, 72 * MT, bf), _cbd(cores, 1, ring[0] * ring[1], self.wdtype),
               _cbd(cores, 2, 72 * MT, bf), _cbd(cores, 3, NB * MT * MT, bf), _cbd(cores, 4, 2 * MT, bf),
               _cbd(cores, 5, 2 * MT, bf), _cbd(cores, 6, 4 * MT, bf), _cbd(cores, 7, 1, bf),
               _cbd(cores, 8, 2 * MT, bf), _cbd(cores, 9, 2 * MT, bf), _cbd(cores, 10, 1, bf),
               _cbd(cores, 11, 1, bf), _cbd(cores, 12, 1, bf), _cbd(cores, 13, 2 * NB * MT, bf),
               _cbd(cores, 14, 3 * MT, bf), _cbd(cores, 15, 2 * MT, f32), _cbd(cores, 16, 2 * MT, bf),
               ttnn.CBDescriptor(total_size=ls, core_ranges=cores, format_descriptors=[
                   ttnn.CBFormatDescriptor(buffer_index=17, data_format=bf, page_size=ls)]),
               _cbd(cores, 18, 2 * NB * MT, bf)]
        Kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        r_ct = list(ring) + [MT]
        for t in (x, route, used):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        nocxy = [v for n in self.noc for v in (n.x, n.y)]
        r_rt, c_rt, w_rt = [], [], []
        for c, xy in enumerate(self.coords):
            r_rt.append((xy, [x.buffer_address(), route.buffer_address(), used.buffer_address(), gu.buffer_address(),
                              dn.buffer_address(), c, self.pb, NB, self.NC] + nocxy + [dbg]))
            c_rt.append((xy, [fbits(1.0 / self.gc), fbits(1.0 / self.uc), (c // 4) % 2, NB, MT, dbg]))
            w_rt.append((xy, [out.buffer_address(), c, self.noc[0].x, self.noc[0].y, NB, MT]))
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=cores, initial_value=0)]
        prog = ttnn.ProgramDescriptor(kernels=[
            Kd("ink_moec_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            Kd("ink_moec_writer.cpp", ttnn.TensorAccessorArgs(out).get_compile_time_args(), w_rt,
               ttnn.WriterConfigDescriptor()),
            Kd("ink_moec_compute.cpp", [], c_rt, self.ccfg),
        ], semaphores=sems, cbs=cbs)
        ttnn.generic_op([x, route, used, gu, dn, out], prog)
        return out
