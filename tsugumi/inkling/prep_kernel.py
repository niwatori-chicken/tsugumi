"""Inkling decode: q / r / k / v preparation of one token in one program (kernels/ink_prep_*.cpp), one core per role:
  q: x 24 heads x 128 (row 0 of the linear output) * qscale -> l2clip (qk_clip) [* tau]   -> QO [1,1,24,128]
  r: 24 x 16 * rscale -> l2clip (rel_clip) [* tau]                                     -> RO [1,1,24,16]
  k: conv over (x, history) with the taps (+ dynamic taps for full layers) -> l2clip     -> KO [1,1,8,128]
  v: the same conv -> rms norm                                                          -> VO [1,1,8,128]
and the conv history (rows 1..3 of the [32, C] history tensors) shifted in place. Replaces ~25 small ops.
Sources must be L1 (32-byte face-row reads); the history, taps, scales and tau are DRAM."""
import os
import struct

import torch
import ttnn

from ..gqa_attn import _cbd, _cores

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernels")


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class InkPrep:
    def __init__(self, dev, NH=24, NKV=8, HD=128, DR=16, qk_clip=22.627416997969522, rel_clip=8.0, eps=1e-6):
        self.dev = dev
        self.NH, self.NKV, self.HD, self.DR = NH, NKV, HD, DR
        self.qk_clip, self.rel_clip, self.eps = qk_clip, rel_clip, eps
        grid = dev.compute_with_storage_grid_size()
        # roles split over cores by heads (role, first head, heads): q 6 x 4, r 2 x 12, k 4 x 2, v 4 x 2
        self.split = [(0, h, 4) for h in range(0, 24, 4)] + [(1, 0, 12), (1, 12, 12)] + \
            [(2, h, 2) for h in range(0, 8, 2)] + [(3, h, 2) for h in range(0, 8, 2)]
        if NH != 24 or NKV != 8:
            self.split = [(r, 0, NH if r < 2 else NKV) for r in range(4)]
        self.coords = [(i % grid.x, grid.y - 1 - i // grid.x) for i in range(len(self.split))]
        cores = _cores(self.coords)
        bf, f32 = ttnn.bfloat16, ttnn.float32
        self.cbs = [
            _cbd(cores, 0, 4, bf), _cbd(cores, 1, 12, bf), _cbd(cores, 2, 16, bf), _cbd(cores, 3, 4, bf),
            _cbd(cores, 4, 4, bf), _cbd(cores, 5, 1, bf), _cbd(cores, 6, 1, bf), _cbd(cores, 7, 4, f32),
            _cbd(cores, 8, 4, f32), _cbd(cores, 9, 4, bf), _cbd(cores, 11, 1, f32), _cbd(cores, 12, 1, f32),
            _cbd(cores, 16, 4, bf), _cbd(cores, 20, 6, bf),
        ]
        c = ttnn.ComputeConfigDescriptor()
        c.math_fidelity = ttnn.MathFidelity.HiFi4
        c.fp32_dest_acc_en = True
        c.math_approx_mode = False
        self.ccfg = c
        L1 = ttnn.L1_MEMORY_CONFIG
        z = lambda r, w: ttnn.from_torch(torch.zeros(1, 1, r, w), dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                         memory_config=L1)
        self.QO, self.RO, self.KO, self.VO = z(NH, HD), z(NH, DR), z(NKV, HD), z(NKV, HD)
        sc = torch.zeros(32, 32)
        sc[0, :] = 1.0
        sc[16, :] = 1.0
        self.sc = ttnn.from_torch(sc, dtype=bf, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def taps_tensor(dev, tk, tv):
        """Conv taps [1,1,32,C] (row = lag) of k and v -> DRAM [1, 8, 32, 128] in head-row layout: entry
        (kv * 4 + lag), row h = the taps of head h's 128 channels."""
        out = torch.zeros(1, 8, 32, 128)
        for j, t in enumerate((tk, tv)):
            t = (t if isinstance(t, torch.Tensor) else ttnn.to_torch(t)).float().reshape(-1, t.shape[-1])  # [32, C]
            nkv = t.shape[1] // 128
            for lag in range(4):
                out[0, j * 4 + lag, :nkv] = t[lag].view(nkv, 128)
        return ttnn.from_torch(out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, q_src, q_off, r_off, k_src, k_off, v_src, v_off, hist_k, hk_off, hist_v, hv_off, taps,
                 qscale, rscale, tau=None, k_dyn=None, v_dyn=None):
        """*_src: L1 tile tensors (row 0 = the token); *_off: column tile offsets; hist_*: DRAM [1,1,32,C] histories
        with column tile offsets; taps: taps_tensor(); tau: [1,1,1,1] DRAM or None; *_dyn: (src, col tile) of the
        32 dynamic conv weights (full layers) or None. Returns (QO, RO, KO, VO)."""
        NH, NKV, HD, DR = self.NH, self.NKV, self.HD, self.DR
        dummy_d = self.sc
        roles = [
            # src, soff, nrow, hw, nt, conv, hist, hoff, toff, dyn(src, off), scale, mode, thr, out
            (q_src, q_off, NH, HD, HD // 32, None, 0, 0, None, qscale, 0, self.qk_clip, self.QO),
            (q_src, r_off, NH, DR, 1, None, 0, 0, None, rscale, 0, self.rel_clip, self.RO),
            (k_src, k_off, NKV, HD, HD // 32, hist_k, hk_off, 0, k_dyn, None, 0, self.qk_clip, self.KO),
            (v_src, v_off, NKV, HD, HD // 32, hist_v, hv_off, 16, v_dyn, None, 1, 0.0, self.VO),
        ]
        r_rt, c_rt, w_rt = [], [], []
        orr = lambda t: (t if t is not None else dummy_d).buffer_address()
        for (x, y), (role, h0, hn) in zip(self.coords, self.split):
            src, soff, nrow, hw, nt, hist, hoff, toff, dyn, scl, mode, thr, out = roles[role]
            use_tau = int(tau is not None and hist is None)  # q and r of full layers
            r_rt.append(((x, y), [src.buffer_address(), soff, nrow, hw, nt, int(hist is not None), orr(hist), hoff,
                                  taps.buffer_address(), toff, int(dyn is not None),
                                  (dyn[0] if dyn is not None else src).buffer_address(),
                                  dyn[1] if dyn is not None else 0, int(scl is not None), orr(scl), use_tau, orr(tau),
                                  self.sc.buffer_address(), h0, hn]))
            c_rt.append(((x, y), [nt, int(hist is not None), int(dyn is not None), int(scl is not None), use_tau,
                                  mode, fbits(thr), fbits(thr * thr), fbits(1.0 / hw), fbits(self.eps),
                                  fbits(1.0 / 3.0)]))
            w_rt.append(((x, y), [out.buffer_address(), nt, h0, hn, int(hn == nrow)]))
        dram = [t for t in (hist_k, hist_v, taps, qscale, rscale, tau, self.sc) if t is not None]
        l1 = [q_src, k_src, v_src] + [d[0] for d in (k_dyn, v_dyn) if d is not None]
        r_ct = ttnn.TensorAccessorArgs(q_src).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(taps).get_compile_time_args()
        w_ct = ttnn.TensorAccessorArgs(self.QO).get_compile_time_args()
        cores = _cores(self.coords)
        K = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=ct, runtime_args=rt, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            K("ink_prep_reader.cpp", r_ct, r_rt, ttnn.ReaderConfigDescriptor()),
            K("ink_prep_writer.cpp", w_ct, w_rt, ttnn.WriterConfigDescriptor()),
            K("ink_prep_compute.cpp", [], c_rt, self.ccfg),
        ], cbs=self.cbs)
        seen, ins = set(), []
        for t in l1 + dram:
            if id(t) not in seen:
                seen.add(id(t))
                ins.append(t)
        ttnn.generic_op(ins + [self.QO, self.RO, self.KO, self.VO], prog)
        return self.QO, self.RO, self.KO, self.VO
