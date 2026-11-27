"""Launcher for kernels/gnorm_*.cpp: the GDN gated RMSNorm of the large-T prefill, straight from the head-major
output of chunk_gated_delta_rule (output_head_major=True, [NV, T, 128] TILE) to the [1,1,T,NV*128] input of the
out-projection. [NV, T, 128] -> [T, NV*128] is only a permutation of whole tiles, so the reader just addresses the
tiles; this replaces untilize / permute / reshape / tilize and the six ops of the norm."""
import math
import os

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


def gnorm_consts(dev, norm_w, eps=1e-6):
    """[1,1,6*32,32] fp32: ones/128, eps, the 128 norm weights as 4 row-replicated tiles."""
    C = torch.zeros(1, 1, 6 * 32, 32)
    C[0, 0, 0:32] = 1.0 / 128
    C[0, 0, 32:64] = eps
    w = norm_w.float()
    for j in range(4):
        C[0, 0, (2 + j) * 32:(3 + j) * 32] = w[j * 32:(j + 1) * 32][None, :]
    return ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)


class GatedNormKernel:
    def __init__(self, dev, NV, zoff_cols):
        self.dev, self.NV, self.zoff = dev, NV, zoff_cols // 32
        g = dev.compute_with_storage_grid_size()
        self.gx, self.ncore = g.x, g.x * g.y
        self.ck = ttnn.ComputeConfigDescriptor()
        self.ck.math_fidelity = ttnn.MathFidelity.HiFi4
        self.ck.fp32_dest_acc_en = True
        self.ck.math_approx_mode = False
        self._plans = {}

    def _plan(self, NT, odt):
        key = (NT, odt)
        if key not in self._plans:
            nu = NT * self.NV
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
            size = lambda dt: 4096 if dt == f32 else 2048
            cb = lambda i, n, dt: ttnn.CBDescriptor(total_size=n * size(dt), core_ranges=cores,
                                                    format_descriptors=[ttnn.CBFormatDescriptor(
                                                        buffer_index=i, data_format=dt, page_size=size(dt))])
            cbs = [cb(0, 8, odt), cb(1, 8, bf), cb(2, 6, f32), cb(3, 4, f32), cb(4, 1, f32), cb(5, 8, bf)]
            self._plans[key] = (coords, cores, ranges, cbs, size(odt))
        return self._plans[key]

    def __call__(self, o, y, consts, out):
        """o [NV, T, 128] TILE (bf16 or fp32), y [1,1,T,>=zoff+NV*128] bf16 (z at column zoff), consts
        (gnorm_consts), out [1,1,T,NV*128] bf16."""
        NT = o.shape[-2] // 32
        coords, cores, ranges, cbs, otb = self._plan(NT, o.dtype)
        ins = [o, y, consts]
        r_ct = [NT, self.NV, y.shape[-1] // 32, self.zoff, otb]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        w_ct = [self.NV] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        K = lambda src, cta, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=cta, runtime_args=rt, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            K("gnorm_reader.cpp", r_ct, [(c, addrs + [a, b]) for c, (a, b) in zip(coords, ranges)],
              ttnn.ReaderConfigDescriptor()),
            K("gnorm_writer.cpp", w_ct, [(c, [out.buffer_address(), a, b]) for c, (a, b) in zip(coords, ranges)],
              ttnn.WriterConfigDescriptor()),
            K("gnorm_compute.cpp", [], [(c, [a, b]) for c, (a, b) in zip(coords, ranges)], self.ck),
        ], cbs=cbs)
        ttnn.generic_op(ins + [out], prog)
        return out
