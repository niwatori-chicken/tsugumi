"""Launcher for kernels/conv_*.cpp: causal depthwise conv (kernel 4) + SiLU over T rows (large-T prefill)."""
import math
import os

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


def shift_tiles(K=4):
    """P_j, Q_j (32x32): output row t of a tile takes input row t-(K-1)+j, from the previous tile (P) or this one (Q)."""
    P = torch.zeros(K, 32, 32)
    Q = torch.zeros(K, 32, 32)
    for j in range(K):
        for t in range(32):
            r = t + j - (K - 1)
            if r < 0:
                P[j, t, r + 32] = 1.0
            else:
                Q[j, t, r] = 1.0
    return torch.cat([P, Q], 0).reshape(1, 1, 2 * K * 32, 32)


class CausalConvKernel:
    def __init__(self, dev, C):
        self.dev, self.C, self.CTL = dev, C, C // 32
        g = dev.compute_with_storage_grid_size()
        self.gx, self.ncore = g.x, g.x * g.y
        self.pq = ttnn.from_torch(shift_tiles(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        self.ck = ttnn.ComputeConfigDescriptor()
        self.ck.math_fidelity = ttnn.MathFidelity.HiFi4
        self.ck.fp32_dest_acc_en = True
        self.ck.math_approx_mode = False
        self._plans = {}

    def _plan(self, NT):
        if NT not in self._plans:
            nseg = max(1, math.ceil(NT * self.CTL / (self.ncore * 32)))  # ~32 row tiles per unit at most
            rps = math.ceil(NT / nseg)
            nseg = math.ceil(NT / rps)
            nu = nseg * self.CTL
            ncore = min(self.ncore, nu)
            per, rem = divmod(nu, ncore)
            ranges, u = [], 0
            for k in range(ncore):
                n = per + (1 if k < rem else 0)
                ranges.append((u, u + n))
                u += n
            coords = [(k % self.gx, k // self.gx) for k in range(ncore)]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
            bf, f32 = ttnn.bfloat16, ttnn.float32
            cb = lambda i, n, dt: ttnn.CBDescriptor(total_size=n * (4096 if dt == f32 else 2048), core_ranges=cores,
                                                    format_descriptors=[ttnn.CBFormatDescriptor(
                                                        buffer_index=i, data_format=dt,
                                                        page_size=4096 if dt == f32 else 2048)])
            cbs = [cb(0, 3, bf), cb(1, 4, bf), cb(2, 8, bf), cb(3, 4, f32), cb(4, 2, bf), cb(5, 4, f32)]
            self._plans[NT] = (rps, coords, cores, ranges, cbs)
        return self._plans[NT]

    def __call__(self, x, carry, taps, out, x_width=None, kd=None):
        """x [1,1,T,>=C] (the first C columns are the conv input), carry [1,1,32,C] (rows 29..31),
        taps: 4 x [1,1,32,C] (row 0), out [1,1,T,C], or a tuple (q, k, v) of [1,T,kd], [1,T,kd], [1,T,C-2kd]."""
        NT = x.shape[-2] // 32
        XW = (x_width or self.C) // 32
        rps, coords, cores, ranges, cbs = self._plan(NT)
        ins = [x, carry] + list(taps) + [self.pq]
        ct = [NT, self.CTL, rps]
        r_ct = ct + [XW]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        outs = list(out) if isinstance(out, (tuple, list)) else [out]
        writer = "conv_writer3.cpp" if len(outs) == 3 else "conv_writer.cpp"
        w_ct = ct + ([kd // 32] if len(outs) == 3 else [])
        for t in outs:
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        oaddrs = [t.buffer_address() for t in outs]
        K = lambda src, cta, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=cta, runtime_args=rt, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            K("conv_reader.cpp", r_ct, [(c, addrs + [a, b]) for c, (a, b) in zip(coords, ranges)],
              ttnn.ReaderConfigDescriptor()),
            K(writer, w_ct, [(c, oaddrs + [a, b]) for c, (a, b) in zip(coords, ranges)],
              ttnn.WriterConfigDescriptor()),
            K("conv_compute.cpp", ct, [(c, [a, b]) for c, (a, b) in zip(coords, ranges)], self.ck),
        ], cbs=cbs)
        ttnn.generic_op(ins + outs, prog)
        return out


class RowSelectKernel:
    """out [1,1,32,C] = MA @ xa[row tile ra] + MB @ xb[row tile rb] over the first C columns (xa, xb: [1,1,R,>=C],
    MA, MB: 0/1 row-selection tiles in m [1,1,64,32]). Reads 2 row tiles instead of a whole tensor; the row tiles are
    runtime arguments, so one program serves every position."""

    def __init__(self, dev, C):
        self.dev, self.CT = dev, C // 32
        g = dev.compute_with_storage_grid_size()
        ncore = min(g.x * g.y, self.CT)
        per, rem = divmod(self.CT, ncore)
        self.ranges, u = [], 0
        for k in range(ncore):
            n = per + (1 if k < rem else 0)
            self.ranges.append((u, u + n))
            u += n
        self.coords = [(k % g.x, k // g.x) for k in range(ncore)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                                        for x, y in self.coords])
        cb = lambda i, n: ttnn.CBDescriptor(total_size=n * 2048, core_ranges=self.cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
        self.cbs = [cb(0, 4), cb(1, 2), cb(2, 2)]
        self.ck = ttnn.ComputeConfigDescriptor()
        self.ck.math_fidelity = ttnn.MathFidelity.HiFi4
        self.ck.fp32_dest_acc_en = True
        self.ck.math_approx_mode = False

    def __call__(self, xa, ra, xb, rb, m, out):
        ins = [xa, xb, m]
        r_ct = [xa.shape[-1] // 32, xb.shape[-1] // 32]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        K = lambda src, cta, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=cta, runtime_args=rt, config=cfg)
        cr = list(zip(self.coords, self.ranges))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("rowsel_reader.cpp", r_ct, [(c, addrs + [ra, rb, a, b]) for c, (a, b) in cr],
              ttnn.ReaderConfigDescriptor()),
            K("rowsel_writer.cpp", ttnn.TensorAccessorArgs(out).get_compile_time_args(),
              [(c, [out.buffer_address(), a, b]) for c, (a, b) in cr], ttnn.WriterConfigDescriptor()),
            K("rowsel_compute.cpp", [], [(c, [a, b]) for c, (a, b) in cr], self.ck),
        ], cbs=self.cbs)
        ttnn.generic_op(ins + [out], prog)
        return out
