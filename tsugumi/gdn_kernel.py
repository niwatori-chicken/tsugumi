"""Python launcher for the fused multi-token GDN recurrence kernel (kernels/gdn_rec_*.cpp).

Tensors (fp32, TILE, DRAM interleaved), H heads, T tokens (rows padded to 32):
  Q, K, V : [1, H, 32, 128]   rows = tokens (q pre-normalized and scaled, k normalized)
  KT      : [1, H, 128, 32]   K transposed (cols = tokens)
  B       : [1, H, 32, 32]    row t = beta_t (all columns)
  G       : [1, H, T*32, 32]  tile t = exp(g_t) everywhere
  E       : [1, 1, T*32, 32]  tile t = row t ones (constant)
  S       : [1, H, 128, 128]  state, updated in place
Returns O [1, H, 32, 128] (row t = o_t).
"""
import os

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")
TILE_BYTES = 4096  # fp32


def _cb(index, ntiles, cores, dtype=ttnn.float32):
    return ttnn.CBDescriptor(
        total_size=ntiles * TILE_BYTES,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=TILE_BYTES)],
    )


class GDNRecKernel:
    def __init__(self, dev, n_heads, T, heads_per_core=1, fp32_unpack=True):
        self.dev = dev
        self.H = n_heads
        self.T = T
        self.hpc = heads_per_core
        n_cores = n_heads // heads_per_core
        grid = dev.compute_with_storage_grid_size()
        coords = [(i % grid.x, i // grid.x) for i in range(n_cores)]
        self.coords = coords
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords])
        c = self.cores
        self.cbs = [
            _cb(0, 16, c), _cb(1, 16, c),  # S ping-pong
            _cb(2, 4, c), _cb(3, 4, c), _cb(4, 4, c), _cb(5, 4, c),  # Q K KT V
            _cb(6, 1, c), _cb(7, T, c), _cb(8, T, c),  # B G E
            _cb(9, 4, c), _cb(10, 4, c), _cb(11, 4, c),  # D, O ping-pong
            _cb(16, 4, c), _cb(17, 4, c),  # S out, O out
        ]
        utd = [ttnn.UnpackToDestMode.Default] * 64
        if fp32_unpack:
            for i in (0, 1, 7):  # copy_tile sources that must keep full fp32
                utd[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.ccfg = ttnn.ComputeConfigDescriptor()
        self.ccfg.math_fidelity = ttnn.MathFidelity.HiFi4
        self.ccfg.fp32_dest_acc_en = True
        self.ccfg.math_approx_mode = False
        self.ccfg.unpack_to_dest_mode = utd

    def __call__(self, Q, K, KT, V, B, G, E, S, O=None):
        if O is None:
            O = ttnn.allocate_tensor_on_device(ttnn.Shape([1, self.H, 32, 128]), ttnn.float32, ttnn.TILE_LAYOUT,
                                               self.dev, ttnn.DRAM_MEMORY_CONFIG)
        ins = [Q, K, KT, V, B, G, E, S]
        r_ct = [self.T, self.hpc]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = [self.hpc]
        for t in (S, O):
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        r_rt, w_rt, c_rt = [], [], []
        for n, (x, y) in enumerate(self.coords):
            h0 = n * self.hpc
            r_rt.append(((x, y), addrs + [h0]))
            w_rt.append(((x, y), [S.buffer_address(), O.buffer_address(), h0]))
            c_rt.append(((x, y), []))
        kd = [
            ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "gdn_rec_reader.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores, compile_time_args=r_ct, runtime_args=r_rt,
                config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "gdn_rec_writer.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores, compile_time_args=w_ct, runtime_args=w_rt,
                config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=os.path.join(KDIR, "gdn_rec_compute.cpp"),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.cores, compile_time_args=[self.T, self.hpc], runtime_args=c_rt,
                config=self.ccfg),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kd, cbs=self.cbs)
        ttnn.generic_op(ins + [O], prog)
        return O


class GDNFusedKernel:
    """Post-conv fused GDN: l2norm(k), multi-token recurrence, gated RMSNorm. See kernels/gdn_fused_*.cpp.

    X  [1,1,32,qkv_dim] bf16 (conv+silu output, rows = tokens), Y [1,1,32,Np] bf16 (in-proj output, z part),
    GB [1,1,32,>=2*NV] fp32 (cols 0..NV-1 exp(g), NV..2NV-1 beta), C consts (make_consts), S state fp32.
    Returns OUT [1,1,32,NV*128] bf16 ready for the out projection.
    """

    def __init__(self, dev, NV, NK, T, qkv_dim, kd, heads_per_core=1, dbg=0, snap_all=True, seg=None):
        """seg: rows per lane (batched decode: T/seg independent sequences, lane u in rows u*seg.., starting from
        state slot IDX[u] and writing slots u*seg..u*seg+seg-1). None: one sequence of T rows."""
        self.seg = seg or T
        assert T % self.seg == 0 and T // self.seg <= 16
        self.dbg = dbg
        self.snap_all = snap_all
        self.dev, self.NV, self.NK, self.T, self.hpc = dev, NV, NK, T, heads_per_core
        self.rep = NV // NK
        self.koff = kd // 32
        self.voff = 2 * kd // 32
        self.zoff = qkv_dim // 32
        n_cores = NV // heads_per_core
        grid = dev.compute_with_storage_grid_size()
        self.coords = [(i % grid.x, i // grid.x) for i in range(n_cores)]
        c = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in self.coords])
        self.cores = c
        bf = lambda i, n: ttnn.CBDescriptor(total_size=n * 2048, core_ranges=c, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)])
        self.nc = 9 + T
        self.cbs = [
            _cb(0, 16, c), _cb(1, 16, c), bf(2, 4), bf(3, 4), _cb(4, 4, c), bf(5, 4), _cb(6, 1, c), _cb(7, T, c),
            _cb(8, self.nc, c), _cb(9, 4, c), _cb(10, 4, c), _cb(11, 4, c), bf(12, 4), _cb(13, 4, c), _cb(14, 4, c),
            _cb(15, 1, c), _cb(16, 4, c), bf(17, 4), _cb(18, 4, c), _cb(19, 16, c), _cb(20, 2, c),
        ]
        self.ccfg = ttnn.ComputeConfigDescriptor()
        self.ccfg.math_fidelity = ttnn.MathFidelity.HiFi4
        self.ccfg.fp32_dest_acc_en = True
        self.ccfg.math_approx_mode = False

    @staticmethod
    def make_idx(dev, value=0):
        import torch

        return ttnn.from_torch(torch.full((1, 16), value, dtype=torch.int32), dtype=ttnn.uint32,
                               layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    @staticmethod
    def set_idx(idx_tensor, value):
        """value: one slot for every lane, or a list (slot of lane u)."""
        import torch

        t = torch.full((1, 16), value if isinstance(value, int) else 0, dtype=torch.int32)
        if not isinstance(value, int):
            t[0, :len(value)] = torch.tensor(value, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                                        idx_tensor)

    @staticmethod
    def make_consts(dev, norm_w, T):
        import torch

        C = torch.zeros(1, 1, (9 + T) * 32, 32)
        C[0, 0, 0:32] = 1.0
        C[0, 0, 32:64] = 1.0 / 128
        C[0, 0, 64:96] = 1e-6
        for j in range(4):
            C[0, 0, (3 + j) * 32:(4 + j) * 32] = norm_w.float()[j * 32:(j + 1) * 32][None, :]
        C[0, 0, 7 * 32:8 * 32] = 128.0
        C[0, 0, 8 * 32:9 * 32] = 128e-6
        for t in range(T):
            C[0, 0, (9 + t) * 32 + t] = 1.0
        return ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    def __call__(self, X, Y, GE, BE, C, S, IDX, OUT=None):
        if OUT is None:
            OUT = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, self.NV * 128]), ttnn.bfloat16,
                                                 ttnn.TILE_LAYOUT, self.dev, ttnn.L1_MEMORY_CONFIG)
        ins = [X, Y, GE, BE, C, S, IDX]
        r_ct = [self.T, self.hpc, self.rep, self.koff, self.voff, self.zoff, self.NV, self.nc, self.seg]
        for t in ins:
            r_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        w_ct = [self.hpc, self.T if self.snap_all else 1, self.NV]
        for t in (S, OUT):
            w_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        addrs = [t.buffer_address() for t in ins]
        r_rt, w_rt, c_rt = [], [], []
        for n, (x, y) in enumerate(self.coords):
            h0 = n * self.hpc
            r_rt.append(((x, y), addrs + [h0]))
            w_rt.append(((x, y), [S.buffer_address(), OUT.buffer_address(), h0]))
            c_rt.append(((x, y), []))
        kd = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "gdn_fused_reader.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "gdn_spec_writer.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "gdn_fused_compute.cpp"),
                                  source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=self.cores,
                                  compile_time_args=[self.T, self.hpc, self.seg], runtime_args=c_rt,
                                  config=self.ccfg, defines=[("DBG", str(self.dbg)), ("SNAP_ALL", "1" if self.snap_all else "0")]),
        ]
        ttnn.generic_op(ins + [OUT], ttnn.ProgramDescriptor(kernels=kd, cbs=self.cbs))
        return OUT
