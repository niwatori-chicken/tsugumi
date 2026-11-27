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


class InkRouter:
    """Router logits + routing in ONE program (kernels/ink_router_*.cpp): 12 cores each compute a K third of one
    32-expert logits tile (bf16 x @ bf16 router weights, float32 accumulation: the same products as the float32
    row dots), core 0 sums the partials in float32 (pulled from the cores' CBs after an in-program barrier) and
    routes as InkRoute. Replaces rowdot (slice / typecast / multiply / sum / transpose) + InkRoute.

    router = InkRouter(dev, E=128, K=8, kscale=..., H=2304)
    ids, wcol = router(x, Wg, bias)   # x [1,1,32,H] (row 0); Wg [1,1,H,E] bf16; bias [1,1,1,E] float32
    """

    def __init__(self, dev, E=128, K=8, kscale=1.0, H=2304, KS=3):
        self.dev, self.E, self.K, self.kscale, self.H, self.KS = dev, E, K, kscale, H, KS
        self.NT, self.HT = E // 32, H // 32
        assert self.HT % KS == 0
        self.KTC = self.HT // KS
        self.NC = self.NT * KS
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, grid.y - 1 - i // grid.x) for i in range(self.NC)]
        W = lambda x, y: dev.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        self.noc = [W(x, y) for x, y in self.coords]
        cores = _cores(self.coords)
        bf, f32 = ttnn.bfloat16, ttnn.float32
        self.cbs = [_cbd(cores, 0, self.KTC, bf), _cbd(cores, 1, self.KTC, bf), _cbd(cores, 2, self.NT, f32),
                    _cbd(cores, 3, self.NT, f32), _cbd(cores, 4, 1, f32), _cbd(cores, 5, self.NC, f32),
                    _cbd(cores, 6, self.NT, f32), _cbd(cores, 7, 1, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        utd = [ttnn.UnpackToDestMode.Default] * 64
        utd[5] = utd[6] = ttnn.UnpackToDestMode.UnpackToDestFp32  # partial sums and the bias in full float32
        c.unpack_to_dest_mode = utd
        self.ccfg = c
        L1 = ttnn.L1_MEMORY_CONFIG
        self.ids = ttnn.from_torch(torch.zeros(1, 1, 1, K, dtype=torch.int32), dtype=ttnn.uint16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=L1)
        self.wcol = ttnn.from_torch(torch.zeros(1, K, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=L1)

    def __call__(self, x, Wg, bias):
        cores = _cores(self.coords)
        Kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        kb = struct.unpack("<I", struct.pack("<f", self.kscale))[0]
        nocxy = [v for n in self.noc for v in (n.x, n.y)]
        r_ct = []
        for t in (x, Wg, bias):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        for c, xy in enumerate(self.coords):
            n, ks = c // self.KS, c % self.KS
            r_rt.append((xy, [x.buffer_address(), Wg.buffer_address(), bias.buffer_address(), c, ks * self.KTC,
                              self.KTC, n, self.NT, self.NC] + nocxy))
            c_rt.append((xy, [self.KTC, int(c == 0), self.NT, self.KS]))
            w_rt.append((xy, [self.ids.buffer_address(), self.wcol.buffer_address(), self.NT, self.K, kb, int(c == 0),
                              self.noc[0].x, self.noc[0].y]))
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=cores, initial_value=0)]
        prog = ttnn.ProgramDescriptor(kernels=[
            Kd("ink_router_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            Kd("ink_router_writer.cpp", ttnn.TensorAccessorArgs(self.ids).get_compile_time_args() +
               ttnn.TensorAccessorArgs(self.wcol).get_compile_time_args(), w_rt, ttnn.WriterConfigDescriptor()),
            Kd("ink_router_compute.cpp", [], c_rt, self.ccfg),
        ], semaphores=sems, cbs=self.cbs)
        ttnn.generic_op([x, Wg, bias, self.ids, self.wcol], prog)
        return self.ids, self.wcol
