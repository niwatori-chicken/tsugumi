"""Batched decode: several requests share each step (one read of the weights for all of them).

Decode is bound by reading the weights (~40 ms of a ~60 ms step), and a step already pads its 4 rows (current
token + 3 MTP drafts) to a 32-row tile. Here the 32 rows hold up to 8 *lanes* of 4 rows, one lane per request: the
matmuls cost the same, and only the per-sequence parts grow (the GDN recurrence walks every row, the GQA kernel
reads each lane's KV, the KV write).

Layout of a batched mode with L lanes (key ("b", L), T = 4 L rows), lane u:
  rows u*4 .. u*4+3                 x_in, hid_src, cos/sin, logits (rows of the other lanes are independent)
  GDN conv history                  Hc_b rows u*3 .. u*3+2, Bp_b rows u*4 .. (see SpecShared for the 1-lane form)
  GDN state snapshots               snap_b slots u*4 .. u*4+3; the step starts from slot IDX[u]
  KV                                its conversation's blocks, through its own page table pt[u] and positions pos[u]
These are separate from the single-sequence modes (Hc, Bp, snap, pt1), which stay in use for prompt prefill: a new
request is prefilled there (between batched steps), then its state is moved into a free lane (`join`). When a
request ends, its lane state goes back to the host (`save_lane`) and its conversation to the Generator's list, so a
follow-up request reuses it as before.

Modes are traced for L = 1, 2, 4, .. up to the lane count (a step uses the smallest that covers the highest busy
lane); unused lanes run on zero rows and write their KV into a reserved trash block.
"""
import collections
import threading
import time

import torch
import ttnn

from .env import env
from .generator import Conversation
from .model import SpecShared
from .page_copy import PageCopy

DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG


def lane_counts(max_lanes):
    out, n = [], 1
    while n < max_lanes:
        out.append(n)
        n *= 2
    return out + [max_lanes]


class BatchShared:
    """Per-step device inputs of one batched mode (the batch counterpart of SpecShared)."""

    batch = True

    def __init__(self, bm, lanes):
        eng, dev, seg = bm.eng, bm.eng.dev, bm.seg
        self.lanes, self.seg, self.T = lanes, seg, lanes * seg
        self.key = ("b", lanes)
        bf = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        self.m_hist = bf(bm.hist_matrix([0] * lanes))
        self.m_shift = [bf(bm.shift_matrix(lanes, j)) for j in range(4)]
        from .gdn_kernel import GDNFusedKernel

        self.idx = GDNFusedKernel.make_idx(dev, 0)
        self.kv_pos = bm.pos[:lanes]
        self.kv_pt = bm.pt[:lanes]
        self.cos = bf(torch.ones(1, self.T, 1, eng.HD))
        self.sin = bf(torch.zeros(1, self.T, 1, eng.HD))
        from .attn import kv_dtype
        from .gqa_attn import GQAAttention

        self.gqa = GQAAttention(dev, seg, HD=eng.HD, block_size=64, nblocks=bm.nb, kv_dtype=kv_dtype(), lanes=lanes)
        self.sel_ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                       fp32_dest_acc_en=True, packer_l1_acc=False)


class BatchMode:
    """Engine side: the batched modes, the lane hand-off ops and one batched step. Create after
    init_spec / init_prefill / init_verify / init_large_prefill and before capture_all (sets eng.batch)."""

    def __init__(self, eng, max_lanes):
        assert eng.mtp is not None, "batched decode needs the MTP head"
        assert eng.sh_by[eng.T_dec].gqa is not None, "batched decode needs the GQA kernel (TSUGUMI_GQA_ATTN=1)"
        self.eng, self.dev = eng, eng.dev
        self.seg = eng.T_dec
        assert max_lanes * self.seg <= 32 and max_lanes * 3 <= 32
        self.max_lanes = max_lanes
        self.nb = eng.max_seq // 64
        pager = eng.pager
        # a block that is never handed out: idle lanes write their (ignored) KV rows here
        self.trash = pager.free.pop(0)
        self.trash_row = torch.full((self.nb,), self.trash, dtype=torch.int32)
        i32 = lambda t: ttnn.from_torch(t, dtype=ttnn.int32, device=self.dev)
        self.pos = [i32(torch.arange(self.seg, dtype=torch.int32)) for _ in range(max_lanes)]
        self.pt = [i32(self.trash_row.view(1, -1)) for _ in range(max_lanes)]
        self.gdns = [L["mix"] for L in eng.layers if L["type"] == "linear_attention"]
        self.attns = [L["mix"] for L in eng.layers if L["type"] != "linear_attention"] + [eng.mtp.attn]
        self.counts = lane_counts(max_lanes)
        for n in self.counts:
            sh = BatchShared(self, n)
            eng.sh_by[sh.key] = sh
            for m in self.gdns:
                m.init_batch(sh.key, sh.T, self.seg, max_lanes)
            for a in self.attns:
                a.init_batch(sh.key, self.seg, n)
        m = self.gdns[0]
        self.Nv, self.Dk, self.Dv, self.qkv = m.Nv, m.Dk, m.Dv, m.qkv_dim
        self.state_pages = m.Nv * (m.Dk // 32) * (m.Dv // 32)
        self.pcopy = PageCopy(self.dev, self.state_pages, 4096)
        self.zero_state = ttnn.from_torch(torch.zeros(1, m.Nv, m.Dk, m.Dv), dtype=ttnn.float32,
                                          layout=ttnn.TILE_LAYOUT, device=self.dev)
        self.state_io = StateIO(self)
        self.prof = collections.Counter()  # seconds spent in the main / MTP trace replays (incl. the host reads)
        eng.batch = self
        eng.state_io = self.state_io

    def keys(self):
        return [("b", n) for n in self.counts]

    # ---------------- selection matrices ----------------
    def _row(self, u, k):
        """index k of lane u's s = [hist0, hist1, hist2, block0, ...] -> row of the [Hc_b; block] concat."""
        return u * 3 + k if k < 3 else 32 + u * self.seg + k - 3

    def shift_matrix(self, lanes, j):
        M = torch.zeros(1, 1, 32, 64)
        for u in range(lanes):
            for t in range(self.seg):
                M[0, 0, u * self.seg + t, self._row(u, t + j)] = 1.0
        return M

    def hist_matrix(self, a):
        """new Hc_b from [Hc_b; Bp_b]: lane u's history after it committed a[u] rows of its last step."""
        M = torch.zeros(1, 1, 32, 64)
        for u, au in enumerate(a):
            for r in range(3):
                M[0, 0, u * 3 + r, self._row(u, au + r)] = 1.0
        return M

    # ---------------- compile / warm-up (before the trace capture) ----------------
    def compile_mode(self, key):
        eng = self.eng
        n = key[1]
        self.set_inputs(key, [None] * n)
        eng.forward_spec()
        self.set_mtp_inputs(key, [None] * n)
        eng.mtp.forward()

    def warmup_handoff(self):
        """Run every eager hand-off op once (programs must not be compiled after the trace capture)."""
        eng = self.eng
        eng.reset_spec()
        self.join(0)
        self.save_lane(0, 1)
        snap = self.state_io.save()
        self.state_io.restore(snap)
        eng.reset_spec()
        for m in self.gdns:  # leave the lane buffers clean
            for t in (m.Hc_b, m.Bp_b):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(torch.zeros(tuple(t.shape)), dtype=t.dtype, layout=ttnn.TILE_LAYOUT), t)

    # ---------------- lane hand-off ----------------
    def join(self, u):
        """Move the single-sequence state (after a prefill / T-row steps) into lane u (lane a = 0 after this)."""
        eng = self.eng
        a = eng._a_prev
        slot = max(a - 1, 0) if eng._last_snap_all else 0
        H = SpecShared.hist_matrix(a)
        M = torch.zeros(1, 1, 32, 96)
        for r in range(32):
            if r // 3 != u or r >= 3 * self.max_lanes:
                M[0, 0, r, r] = 1.0
        for r in range(3):
            M[0, 0, u * 3 + r, 32 + int(H[0, 0, r].argmax())] = 1.0
        Md = ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        ck = eng.sh_by[eng.T_dec].sel_ck
        for m in self.gdns:
            cat = ttnn.concat([m.Hc_b, m.Hc, m.Bp], dim=2, memory_config=DRAM)
            hn = ttnn.matmul(Md, cat, memory_config=DRAM, compute_kernel_config=ck)
            ttnn.deallocate(cat)
            ttnn.copy(hn, m.Hc_b)
            ttnn.deallocate(hn)
        ttnn.deallocate(Md)
        self.pcopy.set_offsets(slot * self.state_pages, u * self.seg * self.state_pages)
        for m in self.gdns:
            self.pcopy(m.snap, m.snap_b)

    def save_lane(self, u, a):
        """Host copy of lane u's GDN state after it committed a rows of its last step, in the format of
        LargePrefill.save_state ({layer: (state [1,Nv,Dk,Dv], conv history in rows 29..31 of [1,1,32,qkv])})."""
        M = torch.zeros(1, 1, 32, 64)
        for r in range(3):
            M[0, 0, 29 + r, self._row(u, a + r)] = 1.0
        Md = ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        ck = self.eng.sh_by[self.eng.T_dec].sel_ck
        self.pcopy.set_offsets((u * self.seg + max(a - 1, 0)) * self.state_pages, 0)
        out = {}
        for i, L in enumerate(self.eng.layers):
            if L["type"] != "linear_attention":
                continue
            m = L["mix"]
            st = ttnn.allocate_tensor_on_device(ttnn.Shape([1, self.Nv, self.Dk, self.Dv]), ttnn.float32,
                                                ttnn.TILE_LAYOUT, self.dev, DRAM)
            self.pcopy(m.snap_b, st)
            cat = ttnn.concat([m.Hc_b, m.Bp_b], dim=2, memory_config=DRAM)
            c = ttnn.matmul(Md, cat, memory_config=DRAM, compute_kernel_config=ck)
            ttnn.deallocate(cat)
            out[i] = (ttnn.to_torch(st), ttnn.to_torch(c))
            ttnn.deallocate(st)
            ttnn.deallocate(c)
        ttnn.deallocate(Md)
        return out

    def set_table(self, u, blocks):
        row = self.trash_row.clone()
        if blocks:
            row[:len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(row.view(1, -1), dtype=ttnn.int32), self.pt[u])

    # ---------------- step inputs ----------------
    def _rope(self, mrope, pos):
        """cos, sin rows [HD] of position pos of a sequence with the M-RoPE spans mrope (Qwen35Engine.rope_pos)."""
        eng = self.eng
        rp = pos
        if mrope and pos >= mrope[0][0]:
            for k0, n, s, gw, delta in reversed(mrope):
                if pos >= k0:
                    rp = (s, s + (pos - k0) // gw, s + (pos - k0) % gw) if pos < k0 + n else pos + delta
                    break
        if isinstance(rp, tuple):
            f = torch.tensor(rp, dtype=torch.float64)[eng.mrope_axis] * eng.inv_freq
        else:
            f = rp * eng.inv_freq
        emb = torch.cat([f, f])
        cos = torch.ones(eng.HD, dtype=torch.float64)
        sin = torch.zeros(eng.HD, dtype=torch.float64)
        cos[: eng.rope_dim] = emb.cos()
        sin[: eng.rope_dim] = emb.sin()
        return cos.float(), sin.float()

    def _positions(self, sh, rows):
        """rows[u] = (tokens, first position, mrope) or None (idle lane). Embeddings [32, H], cos/sin, lane pos."""
        eng, seg = self.eng, self.seg
        x = torch.zeros(1, 1, 32, eng.H, dtype=torch.bfloat16)
        cos = torch.ones(1, sh.T, 1, eng.HD)
        sin = torch.zeros(1, sh.T, 1, eng.HD)
        toks_all, rows_all, txt_pos, txt_rows = [], [], [], []
        for u, r in enumerate(rows):
            if r is None:
                p0 = 0
            else:
                toks, p0, mrope = r
                toks_all += toks
                rows_all += range(u * seg, u * seg + len(toks))
                for t in range(seg):
                    if mrope and p0 + t >= mrope[0][0]:  # image spans: the general (slow) path
                        c, s_ = self._rope(mrope, p0 + t)
                        cos[0, u * seg + t, 0] = c
                        sin[0, u * seg + t, 0] = s_
                    else:
                        txt_pos.append(p0 + t)
                        txt_rows.append(u * seg + t)
        if toks_all:
            x[0, 0, rows_all] = eng.embed_rows(toks_all)
        if txt_pos:  # text rows: plain RoPE, vectorized
            f = torch.tensor(txt_pos, dtype=torch.float64)[:, None] * eng.inv_freq[None]
            emb = torch.cat([f, f], 1)
            cos[0, txt_rows, 0, : eng.rope_dim] = emb.cos().float()
            sin[0, txt_rows, 0, : eng.rope_dim] = emb.sin().float()
        for u, r in enumerate(rows):
            p0 = 0 if r is None else r[1]
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.arange(p0, p0 + seg, dtype=torch.int32), dtype=ttnn.int32), self.pos[u])
        h = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(h(cos), sh.cos)
        ttnn.copy_host_to_device_tensor(h(sin), sh.sin)
        return x

    def set_inputs(self, key, rows, a=None):
        """Main-model step: rows[u] = (tokens <= seg, position, mrope) or None; a[u] = rows lane u committed in its
        previous step (0 after join)."""
        eng = self.eng
        sh = eng.sh_by[key]
        a = a or [0] * sh.lanes
        x = self._positions(sh, rows)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=eng.resid_mem), eng.x_in)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.hist_matrix(a), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sh.m_hist)
        from .gdn_kernel import GDNFusedKernel

        GDNFusedKernel.set_idx(sh.idx, [u * self.seg + max(au - 1, 0) for u, au in enumerate(a)])

    def set_mtp_inputs(self, key, rows, sel=None):
        """MTP run: rows as in set_inputs; sel[u] = hid_src rows (lane-local) feeding the lane's rows."""
        eng, seg = self.eng, self.seg
        sh = eng.sh_by[key]
        x = self._positions(sh, rows)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=eng.resid_mem),
            eng.mtp.x_emb)
        M = torch.zeros(1, 1, 32, 32)
        for u, s in enumerate(sel or []):
            for i, r in enumerate(s or []):
                M[0, 0, u * seg + i, u * seg + r] = 1.0
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                                        eng.mtp.m_sel)

    def run_main(self, key):
        """Replay the main batched trace; returns the top-k (values, indices) [T, 64] on the host."""
        eng = self.eng
        t0 = time.perf_counter()
        ttnn.execute_trace(self.dev, eng.spec_traces[key], cq_id=0, blocking=False)
        out = eng.spec_outs[key]
        r = eng.topk_to_host(out[1], eng.sh_by[key].T)
        self.prof["main"] += time.perf_counter() - t0
        return r

    def run_mtp(self, key):
        """Replay the batched MTP trace; returns the draft argmax per row (full-vocabulary ids)."""
        eng = self.eng
        tid, out = eng.mtp.traces[key]
        t0 = time.perf_counter()
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        am = ttnn.to_torch(out[1]).reshape(-1)[: eng.sh_by[key].T]
        self.prof["mtp"] += time.perf_counter() - t0
        vm = eng.mtp.vocab_map
        return [vm[int(v)] if vm is not None else int(v) for v in am]


class StateIO:
    """Host save / restore of the single-sequence GDN state (like LargePrefill.save_state / restore_state, which
    are used when the engine has the large prefill): lets the Generator keep several conversations in batch mode."""

    def __init__(self, bm):
        self.bm = bm

    def save(self):
        eng, bm = self.bm.eng, self.bm
        if getattr(eng, "large", None) is not None:
            return eng.large.save_state()
        a = eng._a_prev
        slot = max(a - 1, 0) if eng._last_snap_all else 0
        H = SpecShared.hist_matrix(a)
        Hs = torch.zeros_like(H)
        Hs[0, 0, 29:] = H[0, 0, :3]
        Md = ttnn.from_torch(Hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bm.dev)
        ck = eng.sh_by[eng.T_dec].sel_ck
        bm.pcopy.set_offsets(slot * bm.state_pages, 0)
        out = {}
        for i, L in enumerate(eng.layers):
            if L["type"] != "linear_attention":
                continue
            m = L["mix"]
            st = ttnn.allocate_tensor_on_device(ttnn.Shape([1, bm.Nv, bm.Dk, bm.Dv]), ttnn.float32,
                                                ttnn.TILE_LAYOUT, bm.dev, DRAM)
            bm.pcopy(m.snap, st)
            cat = ttnn.concat([m.Hc, m.Bp], dim=2, memory_config=DRAM)
            c = ttnn.matmul(Md, cat, memory_config=DRAM, compute_kernel_config=ck)
            ttnn.deallocate(cat)
            out[i] = (ttnn.to_torch(st), ttnn.to_torch(c))
            ttnn.deallocate(st)
            ttnn.deallocate(c)
        ttnn.deallocate(Md)
        return out

    def restore(self, snap):
        """Make snap the single-sequence state (slot 0, Hc rows 0..2, a_prev = 0)."""
        eng, bm = self.bm.eng, self.bm
        if getattr(eng, "large", None) is not None:
            return eng.large.restore_state(snap)
        M = torch.zeros(1, 1, 32, 32)
        for r in range(3):
            M[0, 0, r, 29 + r] = 1.0
        Md = ttnn.from_torch(M, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bm.dev)
        ck = eng.sh_by[eng.T_dec].sel_ck
        bm.pcopy.set_offsets(0, 0)
        for i, (st, carry) in snap.items():
            m = eng.layers[i]["mix"]
            sd = ttnn.from_torch(st, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=bm.dev, memory_config=DRAM)
            bm.pcopy(sd, m.snap)
            ttnn.deallocate(sd)
            cd = ttnn.from_torch(carry, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=bm.dev,
                                 memory_config=DRAM)
            c = ttnn.matmul(Md, cd, memory_config=DRAM, compute_kernel_config=ck)
            ttnn.copy(c, m.Hc)
            ttnn.deallocate(c)
            ttnn.deallocate(cd)
        ttnn.deallocate(Md)
        eng._a_prev = 0
        eng._last_snap_all = False

    def zero_slot0(self):
        """Fresh conversation: only slot 0 of the single-sequence snapshots is read after reset_spec."""
        bm = self.bm
        bm.pcopy.set_offsets(0, 0)
        for m in bm.gdns:
            bm.pcopy(bm.zero_state, m.snap)


# ======================================================================================================
# request-level: lanes and the scheduler
# ======================================================================================================
END = object()


class Job:
    """One request. The scheduler thread puts (token, (values, indices)) items, then END (or an exception)."""

    def __init__(self, ids, max_new, stats=None, sampler=None, think=None, tag="?"):
        self.ids, self.max_new, self.sampler, self.think, self.tag = ids, max_new, sampler, think, tag
        self.stats = stats if stats is not None else {}
        self.q = collections.deque()
        self.cv = threading.Condition()
        self.cancelled = False
        self.generated = 0
        self.since = time.time()
        self.state = "waiting"

    def put(self, item):
        with self.cv:
            self.q.append(item)
            self.cv.notify()

    def get(self, timeout=None):
        with self.cv:
            while not self.q:
                if not self.cv.wait(timeout):
                    return None
            return self.q.popleft()

    def cancel(self):
        self.cancelled = True


class Lane:
    def __init__(self, u, job, conv, mrope):
        self.u, self.job, self.conv, self.mrope = u, job, conv, mrope
        self.p = self.cur = None
        self.drafts, self.kind = [], "mtp"
        self.a = 0
        think = job.think
        self.st = dict(n=0, think=think is not None, n_think=0, n_ans=0)
        self.queue = []  # forced tokens (think budget)

    def account(self, t, eos):
        """Counts the token just emitted; True when the request must stop (Generator._generate.account)."""
        st, think = self.st, self.job.think
        st["n"] += 1
        self.job.generated += 1
        if st["think"]:
            st["n_think"] += 1
            if t == think.end:
                st["think"] = False
        elif think is not None:
            st["n_ans"] += 1
        if t in eos:
            return True
        return (st["n_ans"] if think is not None else st["n"]) >= self.job.max_new


class BatchScheduler:
    """Owns the device: prefills new requests in the single-sequence modes (Generator._prefill, interleaved with
    batched steps), moves them into lanes, and runs batched steps over all lanes.

    lock: held around every use of the device (shared with e.g. the vision encoder); arm(): called once per
    device action (hang backstop); on_error(e): device errors (the server exits after them)."""

    def __init__(self, gen, lock, arm=None, disarm=None, on_error=None, log=print):
        self.gen, self.eng, self.bm = gen, gen.eng, gen.eng.batch
        self.lock = lock
        self.arm, self.disarm = arm or (lambda: None), disarm or (lambda: None)
        self.on_error, self.log = on_error, log
        self.k = self.bm.seg - 1
        self.lanes = [None] * self.bm.max_lanes
        self.pending = collections.deque()
        self.cv = threading.Condition()
        self.stopping = False
        self.dead = False
        self.prefilling = None
        # while a prompt is prefilled, run a batched step at least every PREFILL_SLICE seconds of prefill
        self.slice = float(env("PREFILL_SLICE", "0.15"))
        self._last_step = 0.0
        self.eng.between_chunks = self._between_chunks
        self.thread = threading.Thread(target=self._loop, name="tsugumi-batch", daemon=True)
        self.thread.start()

    # ---- API ----
    def submit(self, job):
        with self.cv:
            self.pending.append(job)
            self.cv.notify()

    def stop(self):
        with self.cv:
            self.stopping = True
            self.cv.notify()

    def jobs(self):
        """Busy and waiting jobs (for /health)."""
        out = [ln.job for ln in self.lanes if ln is not None]
        if self.prefilling is not None:
            out.append(self.prefilling)
        return out, list(self.pending)

    # ---- loop ----
    def _loop(self):
        while True:
            with self.cv:
                while not self.stopping and not self.pending and not any(self.lanes):
                    self.disarm()
                    self.cv.wait()
                if self.stopping:
                    break
                job = None
                if self.pending and None in self.lanes:
                    job = self.pending.popleft()
            try:
                if job is not None:
                    if job.cancelled:
                        job.put(END)
                        continue
                    with self.lock:
                        self.arm()
                        t0 = time.perf_counter()
                        self._admit(job)
                        self.bm.prof["admit"] += time.perf_counter() - t0
                        self.bm.prof["admits"] += 1
                if any(self.lanes):
                    with self.lock:
                        self.arm()
                        self._step()
            except Exception as e:  # device error: fail every request; the server exits
                self.dead = True
                for ln in self.lanes:
                    if ln is not None:
                        ln.job.put(e)
                if job is not None:
                    job.put(e)
                for j in self.pending:
                    j.put(e)
                if self.on_error is not None:
                    self.on_error(e)
                self.log(f"batch scheduler stopped: {type(e).__name__}: {e}")
                return
        for ln in self.lanes:  # shutdown: end everything
            if ln is not None:
                ln.job.put(END)
        for j in self.pending:
            j.put(END)

    def _between_chunks(self):
        """Called by the prefill between chunks (no eager device tensors alive): keep the lanes going."""
        if any(self.lanes) and time.perf_counter() - self._last_step >= self.slice:
            key = self.eng.key
            try:
                self.arm()
                self._step()
            finally:
                self.eng.use_T(key)

    # ---- admission: prefill in the single-sequence modes, first token, drafts, join a lane ----
    def _admit(self, job):
        eng, gen, k = self.eng, self.gen, self.k
        job.state = "prefill"
        self.prefilling = job
        eng.use_T(eng.T_dec)
        eng.sampler = job.sampler
        try:
            nxt, d, last_rows, info = gen._prefill(job.ids)
            v, i = eng.last_topk
            row = (v[eng.last_pick_row], i[eng.last_pick_row])
        finally:
            eng.sampler = None
            self.prefilling = None
        job.stats.update(info)
        job.stats.setdefault("steps", 0)
        job.stats.setdefault("accepted", 0)
        if job.think is not None:
            job.stats.update(think_tokens=0, think_forced=False)
        u = self.lanes.index(None)
        lane = Lane(u, job, gen.cur, list(eng.mrope))
        lane.p, lane.cur = len(job.ids), nxt
        job.put((nxt, row))
        if lane.account(nxt, gen.eos) or job.cancelled:
            job.put(END)  # the conversation stays the Generator's current one (single-sequence state)
            return
        # drafts for the first batched step: MTP chain in the single-sequence mode (its KV is still mapped there)
        lane.drafts, lane.kind = self._plan(lane, d, single=dict(q=lane.p + 1, row=last_rows - 1))
        # move into lane u: GDN state, KV page table; the Generator gets a fresh current conversation
        self.bm.join(u)
        conv = gen.cur
        gen.convs.remove(conv)
        conv.state = None
        c = Conversation()
        gen.convs.append(c)
        gen.cur = c
        eng.pager.activate(c.blocks)
        eng.reset_spec()
        self.bm.set_table(u, conv.blocks)
        lane.nblocks = len(conv.blocks)
        lane.a = 0
        job.state = "decode"
        self.lanes[u] = lane

    def _plan(self, lane, d1, single=None, ds=None):
        """Drafts for the lane's next step: forced tokens (think budget), else d1 + MTP chain. With single=dict(q,
        row), the chain runs now in the single-sequence mode; else returns ([d1], "mtp") and the batched chain
        extends it."""
        job, eng, k = lane.job, self.eng, self.k
        think = job.think
        if think is not None and lane.st["think"] and not lane.queue and lane.st["n_think"] >= think.budget:
            lane.queue.extend(think.force)
            job.stats["think_forced"] = True
        if lane.queue:
            return lane.queue[:k], "forced"
        ds = [d1]
        if single is not None:
            q, row = single["q"], single["row"]
            for j in range(1, k):
                am = eng.mtp.run([ds[-1]], q, [row if j == 1 else 0])
                ds.append(am[0])
                q += 1
        return ds, "mtp"

    def _ensure(self, lane, end_pos):
        blocks = lane.conv.blocks
        need = min((end_pos + 63) // 64, self.bm.nb)
        pager = self.eng.pager
        if len(blocks) >= need:
            return
        while len(blocks) < need:
            if not pager.free and not (pager.on_full and pager.on_full()):
                raise MemoryError(f"KV pool exhausted ({pager.nb} blocks shared by all requests; raise --max-seq)")
            blocks.append(pager.free.pop())
        self.bm.set_table(lane.u, blocks)

    def _leave(self, lane):
        """End the lane's request: its GDN state to the host and its conversation back to the Generator."""
        gen, bm = self.gen, self.bm
        u = lane.u
        conv = lane.conv
        t0 = time.perf_counter()
        if gen.multi:
            conv.state = bm.save_lane(u, lane.a)
            gen.tick += 1
            conv.last_used = gen.tick
            gen.convs.append(conv)
            while len(gen.convs) > gen.max_convs:
                gen._drop(min((x for x in gen.convs if x is not gen.cur), key=lambda x: x.last_used))
        else:
            self.eng.pager.release(conv.blocks)
        bm.set_table(u, [])
        self.lanes[u] = None
        lane.job.put(END)
        bm.prof["leave"] += time.perf_counter() - t0
        bm.prof["leaves"] += 1

    def stats(self):
        """Mean times (ms) of the scheduler's actions since the last call."""
        p = self.bm.prof
        out = dict(step_ms=p["step"] / max(p["steps"], 1) * 1e3, steps=p["steps"],
                   admit_ms=p["admit"] / max(p["admits"], 1) * 1e3, admits=p["admits"],
                   leave_ms=p["leave"] / max(p["leaves"], 1) * 1e3, leaves=p["leaves"])
        p.clear()
        return out

    # ---- one batched step over all lanes ----
    def _step(self):
        eng, bm, gen, k, seg = self.eng, self.bm, self.gen, self.k, self.bm.seg
        t0 = time.perf_counter()
        for ln in self.lanes:  # requests stopped by their consumer: the committed state is that of the last step
            if ln is not None and ln.job.cancelled:
                self._leave(ln)
        busy = [u for u, ln in enumerate(self.lanes) if ln is not None]
        if not busy:
            return
        n_l = next(n for n in bm.counts if n > busy[-1])
        key = ("b", n_l)
        eng.use_T(key)
        lanes = self.lanes[:n_l]
        rows, a = [], []
        for ln in lanes:
            if ln is None:
                rows.append(None)
                a.append(0)
                continue
            try:
                self._ensure(ln, ln.p + 2 * seg + 2)
            except MemoryError as e:
                ln.job.put(e)
                self._leave_error(ln)
                rows.append(None)
                a.append(0)
                continue
            rows.append(([ln.cur] + ln.drafts, ln.p, ln.mrope))
            a.append(ln.a)
        lanes = self.lanes[:n_l]
        bm.set_inputs(key, rows, a)
        v, i = bm.run_main(key)
        # accept / emit
        mtp_rows, sel, cont = [None] * n_l, [None] * n_l, []
        for ln in lanes:
            if ln is None:
                continue
            u, job = ln.u, ln.job
            kk = len(ln.drafts)
            r0 = u * seg
            pick = (lambda r: eng._greedy(v[r], i[r])) if job.sampler is None else (lambda r: job.sampler(v[r], i[r]))
            if ln.kind == "forced":
                n = kk
                del ln.queue[:kk]
                new = ln.drafts + [ln.queue.pop(0) if ln.queue else pick(r0 + n)]
            else:
                n = 0
                while n < kk and ln.drafts[n] == pick(r0 + n):
                    n += 1
                new = ln.drafts[:n] + [pick(r0 + n)]
            job.stats["steps"] += 1
            if ln.kind != "forced":
                job.stats["accepted"] += n
            stop = False
            for j, t in enumerate(new):
                job.put((t, (v[r0 + j], i[r0 + j])))
                if ln.account(t, gen.eos):
                    stop = True
                    break
            # committed to the lane's state: cur + drafts[:n]; when the request stops at new[j] (j < n), keep
            # only what was emitted before it (the state after row j), so the next prompt extends it exactly
            c = min(n, j) if stop else n
            ln.conv.processed += [ln.cur] + ln.drafts[:c]
            ln.a = c + 1
            if job.think is not None:
                job.stats["think_tokens"] = ln.st["n_think"]
            if stop or job.cancelled:
                self._leave(ln)
                continue
            ln.new, ln.n = new, n
            mtp_rows[u] = (new, ln.p + 1, ln.mrope)
            sel[u] = list(range(n + 1))
            cont.append(ln)
        if cont:
            # MTP over the committed rows (fills its KV; the last row's argmax is the first draft)
            bm.set_mtp_inputs(key, mtp_rows, sel)
            am = bm.run_mtp(key)
            chain = []
            for ln in cont:
                ln.drafts, ln.kind = self._plan(ln, am[ln.u * seg + ln.n])
                ln.q = ln.p + ln.n + 2
                chain.append(ln)
            for j in range(1, k):
                rows_j, sel_j = [None] * n_l, [None] * n_l
                for ln in chain:
                    tok = ln.drafts[-1] if ln.kind == "mtp" else 0
                    rows_j[ln.u] = ([tok], ln.q + j - 1, ln.mrope)
                    sel_j[ln.u] = [ln.n if j == 1 else 0]
                bm.set_mtp_inputs(key, rows_j, sel_j)
                am = bm.run_mtp(key)
                for ln in chain:
                    if ln.kind == "mtp":
                        ln.drafts.append(am[ln.u * seg])
            for ln in cont:
                ln.cur = ln.new[-1]
                ln.p = ln.p + ln.n + 1
        self._last_step = time.perf_counter()
        self.step_time = self._last_step - t0
        bm.prof["step"] += self.step_time
        bm.prof["steps"] += 1

    def _leave_error(self, ln):
        self.eng.pager.release(ln.conv.blocks)
        self.bm.set_table(ln.u, [])
        self.lanes[ln.u] = None
        ln.job.put(END)
