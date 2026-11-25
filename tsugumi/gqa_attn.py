"""GQA split-K causal attention over the paged KV cache for T query rows (kernels/gqa_attn_*.cpp, gqa_comb_*.cpp).

Each (kv head g, split s) core streams its share of the keys once and serves all GH q heads x T rows of the
group; a second op combines the S partial softmaxes. Replaces T separate decode-SDPA "users", each of which
re-reads the whole KV cache.

  q   : normed + roped and pre-scaled by 1/sqrt(HD), bf16 TILE DRAM. T < 32: [1, T, NH, HD];
        T == 32: [1, NH, 32, HD] (heads permuted to dim 1)
  kc/vc : [blocks, NKV, BS, HD] TILE paged caches, bf16 or bfp8_b (kv_dtype);  pt : int32 row-major [1, nblocks] page table
  pos : int32 [T] row positions (consecutive; POS[0] = first row)
Returns [1, 1, 32, NH*HD] bf16 (rows >= T zero) — rows = tokens, ready for the gate and o_proj.

lanes > 1 (batched decode): q holds lanes x T rows ([1, lanes*T, NH, HD]), lane u in rows u*T..; pos and pt are
lists (one per lane: its positions and page table). One program serves every lane; each lane gets S = cores /
(NKV * lanes) splits. Rows of the output: lane u in rows u*T.. .
"""
import math
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


def _cbd(cores, index, ntiles, dtype):
    ps = {ttnn.float32: 4096, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(dtype, 2048)
    return ttnn.CBDescriptor(total_size=ntiles * ps, core_ranges=cores, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=ps)])


def _cores(coords):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])


class GQAAttention:
    def __init__(self, dev, T, NH=24, NKV=4, HD=256, block_size=64, chunk_tiles=2, splits=None,
                 fidelity=ttnn.MathFidelity.HiFi4, nblocks=4096, kv_dtype=ttnn.bfloat16, lanes=1):
        assert T == 32 or (16 % T == 0 and T >= 2)
        assert lanes == 1 or (T < 32 and lanes * T <= 32)
        self.lanes = NU = lanes
        self.dev, self.T, self.NH, self.NKV, self.HD, self.BS = dev, T, NH, NKV, HD, block_size
        self.kv_dtype = kv_dtype
        self.kvtb = 1088 if kv_dtype == ttnn.bfloat8_b else 2048
        self.GH = NH // NKV
        self.HDT = HD // 32
        self.RT = (self.GH * T + 31) // 32
        self.CT = chunk_tiles
        assert block_size % 32 == 0
        grid = dev.compute_with_storage_grid_size()
        ncore = grid.x * grid.y
        self.S = splits or ncore // (NKV * NU)
        self.qmode = 1 if T == 32 else 0
        self.posb = ((T * 4 + 31) // 32) * 32
        self.ptb = ((nblocks * 4 + 31) // 32) * 32  # page table bytes (one row-major page)
        # main: core n -> (lane, g, s)
        nm = NKV * self.S
        self.m_coords = [(n % grid.x, n // grid.x) for n in range(nm * NU)]
        self.m_assign = [(n // nm, n % NKV, (n % nm) // NKV) for n in range(nm * NU)]
        mc = _cores(self.m_coords)
        RT, CT, HDT = self.RT, self.CT, self.HDT
        bf, f32 = ttnn.bfloat16, ttnn.float32
        self.m_cbs = [
            _cbd(mc, 0, RT * HDT, bf),  # Q
            _cbd(mc, 1, 2 * CT * HDT, kv_dtype),  # K
            _cbd(mc, 2, 2 * CT * HDT, kv_dtype),  # V
            _cbd(mc, 3, RT * CT, bf),  # mask
            _cbd(mc, 4, 1, bf),  # scaler
            _cbd(mc, 5, 1, bf),  # info
            _cbd(mc, 6, RT * CT, f32),  # S
            _cbd(mc, 7, RT * CT, bf),  # P
            _cbd(mc, 8, RT, f32), _cbd(mc, 9, RT, f32),  # m ping-pong
            _cbd(mc, 10, RT, f32), _cbd(mc, 11, RT, f32),  # l ping-pong
            _cbd(mc, 12, RT, f32),  # alpha
            _cbd(mc, 13, RT * HDT, f32), _cbd(mc, 14, RT * HDT, f32),  # O ping-pong
            _cbd(mc, 16, 2, f32),  # final O
            _cbd(mc, 17, RT, f32),  # final m
            _cbd(mc, 18, RT, f32),  # final l
            _cbd(mc, 19, 2, f32),  # consts
            _cbd(mc, 20, HDT, bf),  # reader scratch
            ttnn.CBDescriptor(total_size=self.ptb, core_ranges=mc, format_descriptors=[  # page table
                ttnn.CBFormatDescriptor(buffer_index=21, data_format=ttnn.uint32, page_size=self.ptb)]),
        ]
        self.m_ccfg = self._ccfg(fidelity)
        # combine: core -> (g, r, dg), DT value tiles each
        self.DT = 1
        while NU * NKV * RT * (HDT // self.DT) > ncore:
            self.DT *= 2
        nd = HDT // self.DT
        nc = NKV * RT * nd
        assert nc * NU <= ncore
        self.c_coords = [(n % grid.x, n // grid.x) for n in range(nc * NU)]
        self.c_assign = [(n // nc, (n % nc) // (RT * nd), ((n % nc) // nd) % RT, n % nd) for n in range(nc * NU)]
        cc = _cores(self.c_coords)
        S, DT = self.S, self.DT
        self.c_cbs = [
            _cbd(cc, 0, S, f32), _cbd(cc, 1, S, f32), _cbd(cc, 2, 2 * DT, f32),
            _cbd(cc, 3, 1, f32), _cbd(cc, 4, S, f32), _cbd(cc, 5, 1, f32), _cbd(cc, 6, S, f32),
            _cbd(cc, 7, 2 * DT, f32), _cbd(cc, 8, DT, bf), _cbd(cc, 9, 1, bf),
        ]
        self.c_ccfg = self._ccfg(ttnn.MathFidelity.HiFi4)
        self.part = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, NU * S * NKV * RT * 32, (HDT + 2) * 32]), f32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        import torch
        self.out = ttnn.from_torch(torch.zeros(1, 1, 32, NH * HD), dtype=bf, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sc = torch.zeros(32, 32)
        sc[0, :] = 1.0
        sc[16, :] = 1.0  # row 0 of every face
        self.sc = ttnn.from_torch(sc, dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cf = torch.cat([torch.full((32, 32), -1e30), torch.zeros(32, 32)], 0)
        self.cf = ttnn.from_torch(cf, dtype=f32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def _ccfg(fid):
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = fid
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        return c

    def __call__(self, q, kc, vc, pos, pt):
        T, RT, CT, HDT, S, NKV = self.T, self.RT, self.CT, self.HDT, self.S, self.NKV
        pos = list(pos) if isinstance(pos, (list, tuple)) else [pos]
        pt = list(pt) if isinstance(pt, (list, tuple)) else [pt]
        assert len(pos) == len(pt) == self.lanes
        K = lambda src, ct, rt, cfg, cores: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        # ---- main ----
        ins = [q, kc, vc, pos[0], self.sc, self.cf, pt[0]]
        r_ct = [T, RT, CT, S, self.BS, NKV, self.posb, self.GH, HDT, self.qmode, self.ptb, self.kvtb]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()

        def addrs(u):
            return [t.buffer_address() for t in (q, kc, vc, pos[u], self.sc, self.cf, pt[u])]
        w_ct = [RT, HDT, NKV] + ttnn.TensorAccessorArgs(self.part).get_compile_time_args()
        pa = self.part.buffer_address()
        mc = _cores(self.m_coords)
        r_rt = [((x, y), addrs(u) + [g, s, u * T]) for (x, y), (u, g, s) in zip(self.m_coords, self.m_assign)]
        w_rt = [((x, y), [pa, g, u * S + s]) for (x, y), (u, g, s) in zip(self.m_coords, self.m_assign)]
        c_rt = [((x, y), []) for (x, y) in self.m_coords]
        prog = ttnn.ProgramDescriptor(kernels=[
            K("gqa_attn_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor(), mc),
            K("gqa_attn_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor(), mc),
            K("gqa_attn_compute.cpp", [RT, CT, HDT], c_rt, self.m_ccfg, mc),
        ], cbs=self.m_cbs)
        extra = pos[1:] + pt[1:]  # the other lanes' tensors (read through runtime addresses)
        ttnn.generic_op(ins + extra + [self.part], prog)
        # ---- combine ----
        cc = _cores(self.c_coords)
        cr_ct = [RT, HDT, NKV, S, self.DT, T, CT * 32, self.posb] + \
            ttnn.TensorAccessorArgs(self.part).get_compile_time_args() + ttnn.TensorAccessorArgs(pos[0]).get_compile_time_args()
        cw_ct = [T, self.GH, HDT, self.DT] + ttnn.TensorAccessorArgs(self.out).get_compile_time_args()
        oa = self.out.buffer_address()
        cr_rt = [((x, y), [pa, pos[u].buffer_address(), g, r, dg, u * S])
                 for (x, y), (u, g, r, dg) in zip(self.c_coords, self.c_assign)]
        cw_rt = [((x, y), [oa, g, r, dg, u * T]) for (x, y), (u, g, r, dg) in zip(self.c_coords, self.c_assign)]
        cc_rt = [((x, y), []) for (x, y) in self.c_coords]
        prog = ttnn.ProgramDescriptor(kernels=[
            K("gqa_comb_reader.cpp", cr_ct, cr_rt, ttnn.ReaderConfigDescriptor(), cc),
            K("gqa_comb_writer.cpp", cw_ct, cw_rt, ttnn.WriterConfigDescriptor(), cc),
            K("gqa_comb_compute.cpp", [self.DT, S], cc_rt, self.c_ccfg, cc),
        ], cbs=self.c_cbs)
        ttnn.generic_op([self.part] + pos + [self.out], prog)
        return self.out
