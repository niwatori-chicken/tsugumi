"""Inkling decode: block attention residual (+ rms norm with weight) of one token in two programs
(kernels/ink_ar_*.cpp). With T = [blocks (rows 0..nb-1, normalized) | h raw (row nb) | queries (rows 31, 30, ..)]:
  phase 1 (NC1 cores): Gram partials T T^T over column tiles
  phase 2 (NC2 cores): G = sum; scores, rms(h), softmax and the output rms from G (scalar, on the writer);
    AI_k = sum_j w_kj N_j, XN_k = rms_norm(AI_k) * ln_k, TOUT = stack with row nb = rmsnorm(h)
Replaces rms_norm(h), the stack (matmul + add), attn_res1 (6 ops) and the rms_norm with weight.

    ar = InkAres(dev, H=2304, eps=1e-6)
    QT = ar.queries(Q)          # Q [nq, H] pre-scaled queries -> DRAM tile tensor
    LN = ar.norm(w)             # [H] norm weight -> DRAM tile tensor (all rows)
    ai, xn, tout = ar(h, NB, nb, QT, nq, lns=[LN..] or None, tout=buffer or None)  # ai or xn is None
"""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class InkAres:
    def __init__(self, dev, H=2304, eps=1e-6, NC1=12, NC2=24, maxnb=5):
        self.dev, self.H, self.eps = dev, H, eps
        HT = H // 32
        assert HT % NC1 == 0 and HT % NC2 == 0
        self.NC1, self.NC2, self.NT1, self.NT2 = NC1, NC2, HT // NC1, HT // NC2
        grid = dev.compute_with_storage_grid_size()
        self.c1 = [(i % grid.x, i // grid.x) for i in range(NC1)]
        self.c2 = [(i % grid.x, i // grid.x) for i in range(NC2)]
        bf, f32 = ttnn.bfloat16, ttnn.float32
        k1, k2 = _cores(self.c1), _cores(self.c2)
        N1, N2 = self.NT1, self.NT2
        self.cbs1 = [_cbd(k1, 0, N1, bf), _cbd(k1, 1, N1, bf), _cbd(k1, 2, N1, bf), _cbd(k1, 3, 1, bf),
                     _cbd(k1, 5, N1, bf), _cbd(k1, 8, 1, f32)]
        self.cbs2 = [_cbd(k2, 0, N2, bf), _cbd(k2, 1, N2, bf), _cbd(k2, 2, N2, bf), _cbd(k2, 3, 1, bf),
                     _cbd(k2, 4, 2, f32), _cbd(k2, 5, N2, bf), _cbd(k2, 6, 2, bf), _cbd(k2, 8, 1, f32),
                     _cbd(k2, 9, 9, bf), _cbd(k2, 10, 2, bf), _cbd(k2, 16, 2, bf), _cbd(k2, 17, 2, bf),
                     _cbd(k2, 18, 2, bf)]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
        L1 = ttnn.L1_MEMORY_CONFIG
        self.part = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32 * NC1, 32]), f32, ttnn.TILE_LAYOUT, dev, L1)
        z = lambda mem=L1: ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                           memory_config=mem)
        self.AI = [z()] + [z(ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]  # 1, 2: full layers only (whole tiles)
        self.XN = self.AI  # one of the two is written (XN with norm weights, AI without)
        self.XS = None  # set_sharded(): XN in the input sharding of the next (DRAM-sharded) linear
        self.zbuf = z
        sh = torch.zeros(1, 1, 32 * maxnb, 32)
        for r in range(maxnb):
            sh[0, 0, 32 * r + r, 0] = 1.0
        self.sh = ttnn.from_torch(sh, dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def set_sharded(self, mem):
        """Buffers for XN written directly in a linear's width-sharded input layout (no conversion op)."""
        self.XS = [ttnn.from_torch(torch.zeros(1, 1, 32, self.H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self.dev, memory_config=mem)]  # one query only (L1 is tight in prefill)

    def queries(self, Q):
        Q = Q.float().reshape(-1, self.H)
        t = torch.zeros(1, 1, 32, self.H)
        for k in range(Q.shape[0]):
            t[0, 0, 31 - k] = Q[k]
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def norm(self, w):
        w = (w if isinstance(w, torch.Tensor) else ttnn.to_torch(w)).float().reshape(-1, w.shape[-1])[0, :self.H]
        return ttnn.from_torch(w.view(1, 1, 1, -1).expand(1, 1, 32, self.H).contiguous(), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _prog(self, phase, x, NB, nb, QT, nq, lns, tout, outs):
        coords, NT = (self.c1, self.NT1) if phase == 1 else (self.c2, self.NT2)
        cores = _cores(coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        nbt = NB if NB is not None else x
        ln = (list(lns) + [lns[0]] * 3)[:3] if lns else [QT] * 3
        r_ct = []
        for t in (x, nbt, QT, self.sh, self.part, ln[0]):
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        to = tout if tout is not None else self.AI[0]
        use_ln = int(bool(lns))
        ai = (list(outs[:nq]) + [outs[0]] * 4)[:4]
        w_ct = ttnn.TensorAccessorArgs(self.part).get_compile_time_args()
        for t in ai + [to]:  # one accessor per output: the buffers may live in different memories
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        r_rt, c_rt, w_rt = [], [], []
        for i, (cx, cy) in enumerate(coords):
            j0 = i * NT
            r_rt.append(((cx, cy), [x.buffer_address(), nbt.buffer_address(), QT.buffer_address(),
                                    self.sh.buffer_address(), self.part.buffer_address()]
                         + [t.buffer_address() for t in ln]
                         + [j0, NT, int(NB is not None), nb, phase, self.NC1, nq, use_ln]))
            c_rt.append(((cx, cy), [phase, NT, int(NB is not None), self.NC1, nq, use_ln, int(tout is not None)]))
            w_rt.append(((cx, cy), [phase, self.part.buffer_address(), i, nb, nq, use_ln, int(tout is not None),
                                    fbits(1.0 / self.H), fbits(self.eps)]
                         + [t.buffer_address() for t in ai] + [to.buffer_address(), j0, NT]))
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_ar_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            K("ink_ar_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
            K("ink_ar_compute.cpp", [], c_rt, self.ccfg),
        ], cbs=self.cbs1 if phase == 1 else self.cbs2)
        ins = [x] + ([NB] if NB is not None else []) + [QT, self.sh, self.part]
        if phase == 2:
            seen = set()
            for t in (lns or []):
                if id(t) not in seen:
                    seen.add(id(t))
                    ins.append(t)
            ins = ins + list(outs[:nq]) + ([tout] if tout is not None else [])
        ttnn.generic_op(ins, prog)

    def __call__(self, x, NB, nb, QT, nq=1, lns=None, tout=None, sharded=False, outs=None):
        """sharded: XN into the set_sharded() buffers (only with norm weights); outs: output buffers (default the
        persistent AI buffers; up to 4 queries with outs)."""
        assert nb < 5 and (tout is None or nq == 1) and (not sharded or (lns and nq == 1))
        assert nq <= (4 if outs is not None else 3)
        if outs is None:
            outs = self.XS if sharded else self.AI
        self._prog(1, x, NB, nb, QT, nq, lns, tout, outs)
        self._prog(2, x, NB, nb, QT, nq, lns, tout, outs)
        return (None if lns else outs[:nq]), (outs[:nq] if lns else None), tout
