"""Launcher for kernels/moe_gemm_*.cpp: grouped GEMM over rows sorted by expert (MoE prefill, method C, doc/12).

Y[rows of e] = X[rows of e] @ W[e], X [1,1,R,K] bf16, W [1,1,E*K,N] bfp4 (experts stacked along K), Y [1,1,R,N].
The schedule is a list of units (e, r0, nr, n0): row tiles r0..r0+nr-1 belong to expert e, output column tiles
n0..n0+NN-1. Each core reads a unit's [K, NN] weight slice once and streams the unit's row tiles through it.
Units are spread over all cores by a greedy balance; the schedule goes in runtime arguments (no recompile)."""
import os

import numpy as np

import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels")


class MoEGemm:
    def __init__(self, dev, K, N, NN=8, U=16, nr_max=8, CT=2, RT=2, wbuf=None, xbuf=2):
        self.dev, self.Kt, self.Nt, self.NN, self.U, self.nr_max = dev, K // 32, N // 32, NN, U, nr_max
        self.CT, self.RT = CT, RT
        assert NN % CT == 0
        assert self.Nt % NN == 0
        g = dev.compute_with_storage_grid_size()
        self.coords = [(x, y) for y in range(g.y) for x in range(g.x)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                                        for x, y in self.coords])
        self.wtb = 576  # bfp4_b tile
        cb = lambda i, n, dt, ps: ttnn.CBDescriptor(total_size=n * ps, core_ranges=self.cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=ps)])
        if wbuf is None:  # double-buffer the weight slice when it fits next to the X rows (1.5 MB of L1)
            wbuf = 2 if 2 * self.Kt * NN * self.wtb + xbuf * RT * self.Kt * 2048 < 1300 * 1024 else 1
        self.cbs = [cb(0, xbuf * RT * self.Kt, ttnn.bfloat16, 2048),
                    cb(1, wbuf * self.Kt * NN, ttnn.bfloat4_b, self.wtb), cb(2, 2 * RT * CT, ttnn.bfloat16, 2048)]
        self.ck = ttnn.ComputeConfigDescriptor()
        self.ck.math_fidelity = ttnn.MathFidelity.LoFi
        self.ck.fp32_dest_acc_en = True
        self.ck.math_approx_mode = False

    def units(self, tiles_per_expert, offsets):
        """tiles_per_expert[e] row tiles of expert e starting at row tile offsets[e] -> int32 array [n, 4] of
        (e, r0, nr, n0)."""
        t = np.asarray(tiles_per_expert, dtype=np.int64)
        o = np.asarray(offsets, dtype=np.int64)
        nch = (t + self.nr_max - 1) // self.nr_max  # row chunks per expert
        e = np.repeat(np.arange(len(t)), nch)
        first = np.repeat(np.cumsum(nch) - nch, nch)
        a = (np.arange(len(e)) - first) * self.nr_max  # chunk start within the expert
        nr = np.minimum(self.nr_max, t[e] - a)
        r0 = o[e] + a
        ns = self.Nt // self.NN
        out = np.empty((len(e) * ns, 4), dtype=np.int64)
        out[:, 0] = np.repeat(e, ns)
        out[:, 1] = np.repeat(r0, ns)
        out[:, 2] = np.repeat(nr, ns)
        out[:, 3] = np.tile(np.arange(ns) * self.NN, len(e))
        return out

    def __call__(self, X, W, Y, units, Y2=None):
        """Y2: when given, output columns >= N/2 go there (Y and Y2 [1,1,R,N/2])."""
        nc = len(self.coords)
        units = np.asarray(units, dtype=np.int64).reshape(-1, 4)
        n = len(units)
        # balance: biggest first, dealt to the cores in a snake order (close to greedy, all in numpy)
        order = np.argsort(-units[:, 2], kind="stable")
        i = np.arange(n)
        rnd, p = i // nc, i % nc
        core = np.where(rnd % 2 == 0, p, nc - 1 - p)
        if rnd[-1] + 1 > self.U if n else False:  # too many units per core for one launch: split the schedule
            half = n // 2
            self(X, W, Y, units[:half], Y2)
            return self(X, W, Y, units[half:], Y2)
        A = np.zeros((nc, self.U, 4), dtype=np.int64)
        A[core, rnd] = units[order]
        cnt = np.bincount(core, minlength=nc)
        A = A.reshape(nc, -1)
        ax, aw, ay = X.buffer_address(), W.buffer_address(), Y.buffer_address()
        Rt = X.shape[-2] // 32
        rows = A.tolist()
        cl = cnt.tolist()
        rd = [(c, [ax, aw, cl[j]] + rows[j] + [Rt]) for j, c in enumerate(self.coords)]
        ay2 = (Y2 if Y2 is not None else Y).buffer_address()
        wr = [(c, [ay, cl[j]] + rows[j] + [ay2]) for j, c in enumerate(self.coords)]
        cp = [(c, [cl[j]] + rows[j]) for j, c in enumerate(self.coords)]
        r_ct = [self.Kt, self.Nt, self.NN, self.U, self.wtb, self.RT, int(W.layout == ttnn.ROW_MAJOR_LAYOUT),
                int(os.environ.get("MOEGEMM_DBG_R", "0"))]
        r_ct += ttnn.TensorAccessorArgs(X).get_compile_time_args() + ttnn.TensorAccessorArgs(W).get_compile_time_args()
        w_ct = [self.Nt, self.NN, self.CT, self.RT, int(os.environ.get("MOEGEMM_DBG_W", "0")), int(Y2 is not None),
                self.U] + ttnn.TensorAccessorArgs(Y).get_compile_time_args() + \
            ttnn.TensorAccessorArgs(Y2 if Y2 is not None else Y).get_compile_time_args()
        K = lambda src, cta, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=os.path.join(KDIR, src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores, compile_time_args=cta, runtime_args=rt, config=cfg)
        prog = ttnn.ProgramDescriptor(kernels=[
            K("moe_gemm_reader.cpp", r_ct, rd, ttnn.ReaderConfigDescriptor()),
            K("moe_gemm_writer.cpp", w_ct, wr, ttnn.WriterConfigDescriptor()),
            K("moe_gemm_compute.cpp", [self.Kt, self.NN, self.CT, self.RT, int(os.environ.get("MOEGEMM_SKIP", "0"))],
              cp, self.ck),
        ], cbs=self.cbs)
        ttnn.generic_op([X, W, Y] + ([Y2] if Y2 is not None else []), prog)
        return Y
