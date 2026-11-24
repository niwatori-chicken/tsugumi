"""Inkling decode attention (one token) with the relative bias, the validity and the sink built into the mask inside
the kernel (kernels/ink_attn_reader.cpp + ink_attn_compute.cpp, combined by gqa_comb_*.cpp).

Replaces, per layer application: the bias permutation / repetition over the cache, the mask tensors, the sink
reshaping and the decode SDPA; full layers only read the keys up to the current position (the SDPA with a mask read
the whole cache every step).

    att = InkAttention(dev, ring=True, bsl=RA, r=R, w=W, sink=R)
    out = att(q, kc, vc, rel, sinkv, pos, npos)   # [1, 1, 32, NH * HD] bf16, row 0
      q     [1, 1, NH, HD] bf16 (not pre-scaled);  kc / vc [1, NKV, bsl, HD]
      rel   [1, 1, NH, wr] bf16, bias by distance (units of 1 / scale);  sinkv [1, 1, 1, NH] (units of 1 / scale)
      pos   int32 [8], pos[0] = position;  npos int32 [8], npos[0] = chunks * C - 1 (read by the combine)
    q, rel and sinkv must be in L1 (the reader reads 32-byte face rows of them; DRAM needs 64-byte alignment).
"""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


class InkAttention:
    def __init__(self, dev, ring, bsl, r, w, sink, NH=24, NKV=8, HD=128, chunk_tiles=2, kv_dtype=ttnn.bfloat8_b,
                 scale=None, wr=None, splits=None):
        self.dev, self.ring, self.bsl, self.R, self.W, self.sink = dev, ring, bsl, r, w, sink
        self.NH, self.NKV, self.HD = NH, NKV, HD
        self.GH, self.HDT, self.RT, self.CT = NH // NKV, HD // 32, 1, chunk_tiles
        self.C = chunk_tiles * 32
        self.kvtb = 1088 if kv_dtype == ttnn.bfloat8_b else 2048
        self.wrt = (wr if wr is not None else (r if ring else w)) // 32
        self.scale = scale if scale is not None else HD ** -0.5
        grid = dev.compute_with_storage_grid_size()
        ncore = grid.x * grid.y
        self.S = splits or ncore // NKV
        nm = NKV * self.S
        self.m_coords = [(n % grid.x, n // grid.x) for n in range(nm)]
        self.m_assign = [(n % NKV, n // NKV) for n in range(nm)]
        mc = _cores(self.m_coords)
        RT, CT, HDT = self.RT, self.CT, self.HDT
        bf, f32 = ttnn.bfloat16, ttnn.float32
        self.m_cbs = [
            _cbd(mc, 0, RT * HDT, bf), _cbd(mc, 1, 2 * CT * HDT, kv_dtype), _cbd(mc, 2, 2 * CT * HDT, kv_dtype),
            _cbd(mc, 3, 2 * RT * CT, bf), _cbd(mc, 4, 1, bf), _cbd(mc, 5, 1, bf), _cbd(mc, 6, RT * CT, f32),
            _cbd(mc, 7, RT * CT, bf), _cbd(mc, 8, RT, f32), _cbd(mc, 9, RT, f32), _cbd(mc, 10, RT, f32),
            _cbd(mc, 11, RT, f32), _cbd(mc, 12, RT, f32), _cbd(mc, 13, RT * HDT, f32), _cbd(mc, 14, RT * HDT, f32),
            _cbd(mc, 16, 2, f32), _cbd(mc, 17, RT, f32), _cbd(mc, 18, RT, f32), _cbd(mc, 19, 2, f32),
            _cbd(mc, 20, HDT, bf), _cbd(mc, 21, self.wrt + 1, bf),
        ]
        cfg = lambda fid: _ccfg(fid)
        self.m_ccfg = cfg(ttnn.MathFidelity.HiFi4)
        self.DT = 1
        while NKV * RT * (HDT // self.DT) > ncore:
            self.DT *= 2
        nd = HDT // self.DT
        nc = NKV * RT * nd
        self.c_coords = [(n % grid.x, n // grid.x) for n in range(nc)]
        self.c_assign = [(n // (RT * nd), (n // nd) % RT, n % nd) for n in range(nc)]
        cc = _cores(self.c_coords)
        S, DT = self.S, self.DT
        self.c_cbs = [
            _cbd(cc, 0, S, f32), _cbd(cc, 1, S, f32), _cbd(cc, 2, 2 * DT, f32), _cbd(cc, 3, 1, f32),
            _cbd(cc, 4, S, f32), _cbd(cc, 5, 1, f32), _cbd(cc, 6, S, f32), _cbd(cc, 7, 2 * DT, f32),
            _cbd(cc, 8, DT, bf), _cbd(cc, 9, 1, bf),
        ]
        self.c_ccfg = cfg(ttnn.MathFidelity.HiFi4)
        self.part = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, S * NKV * RT * 32, (HDT + 2) * 32]), f32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        self.out = ttnn.from_torch(torch.zeros(1, 1, 32, NH * HD), dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sc = torch.zeros(32, 32)
        sc[0, :] = 1.0
        sc[16, :] = 1.0
        self.sc = ttnn.from_torch(sc, dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cf = torch.cat([torch.full((32, 32), -1e30), torch.zeros(32, 32)], 0)
        self.cf = ttnn.from_torch(cf, dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def chunks(self, pos):
        """Number of key chunks the kernel processes at this position (for the combine's npos)."""
        C = self.C
        nn = min(pos + 1, self.R) if self.ring else pos + 1
        n_norm = (nn + C - 1) // C
        sb = (self.sink // C) * C
        return n_norm + (1 if sb >= n_norm * C else 0)

    def __call__(self, q, kc, vc, rel, sinkv, pos, npos):
        RT, CT, HDT, S, NKV = self.RT, self.CT, self.HDT, self.S, self.NKV
        K = lambda src, ct, rt, cfg, cores: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        ins = [q, kc, vc, pos, self.sc, self.cf, rel, sinkv]
        r_ct = [RT, CT, S, NKV, self.GH, HDT, self.kvtb, self.bsl, int(self.ring), self.R, self.W, self.sink,
                self.wrt]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        ad = [t.buffer_address() for t in ins]
        w_ct = [RT, HDT, NKV] + ttnn.TensorAccessorArgs(self.part).get_compile_time_args()
        pa = self.part.buffer_address()
        mc = _cores(self.m_coords)
        scale_bits = struct.unpack("<I", struct.pack("<f", self.scale))[0]
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_attn_reader.cpp", r_ct, [((x, y), ad + [g, s]) for (x, y), (g, s) in
                                            zip(self.m_coords, self.m_assign)], ttnn.ReaderConfigDescriptor(), mc),
            K("gqa_attn_writer.cpp", w_ct, [((x, y), [pa, g, s]) for (x, y), (g, s) in
                                            zip(self.m_coords, self.m_assign)], ttnn.WriterConfigDescriptor(), mc),
            K("ink_attn_compute.cpp", [RT, CT, HDT, scale_bits], [((x, y), []) for (x, y) in self.m_coords],
              self.m_ccfg, mc),
        ], cbs=self.m_cbs)
        ttnn.generic_op(ins + [self.part], prog)
        cc = _cores(self.c_coords)
        cr_ct = [RT, HDT, NKV, S, self.DT, 1, self.C, 32] + \
            ttnn.TensorAccessorArgs(self.part).get_compile_time_args() + ttnn.TensorAccessorArgs(npos).get_compile_time_args()
        cw_ct = [1, self.GH, HDT, self.DT] + ttnn.TensorAccessorArgs(self.out).get_compile_time_args()
        oa = self.out.buffer_address()
        prog = ttnn.ProgramDescriptor(kernels=[
            K("gqa_comb_reader.cpp", cr_ct, [((x, y), [pa, npos.buffer_address(), g, r, dg, 0])
                                             for (x, y), (g, r, dg) in zip(self.c_coords, self.c_assign)],
              ttnn.ReaderConfigDescriptor(), cc),
            K("gqa_comb_writer.cpp", cw_ct, [((x, y), [oa, g, r, dg, 0])
                                             for (x, y), (g, r, dg) in zip(self.c_coords, self.c_assign)],
              ttnn.WriterConfigDescriptor(), cc),
            K("gqa_comb_compute.cpp", [self.DT, S], [((x, y), []) for (x, y) in self.c_coords], self.c_ccfg, cc),
        ], cbs=self.c_cbs)
        ttnn.generic_op([self.part, npos, self.out], prog)
        return self.out


def _ccfg(fid):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c
