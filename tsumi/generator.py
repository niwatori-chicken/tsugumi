"""Speculative generation loop (MTP drafts) with conversation-prefix reuse, for the server and CLI tools.

Greedy (temperature 0) is exact: the output equals plain greedy decoding of the engine.
"""
import os
import time

from .env import env

EOS_DEFAULT = (248046, 248044)
IM_START = 248045


class Conversation:
    """One conversation's slot: its KV blocks (in the shared pool), the tokens in its state, the GDN state saved
    when it was switched out, and the snapshot at the start of its last assistant turn."""

    def __init__(self):
        self.blocks = []
        self.processed = []
        self.state = None  # host GDN state at len(processed); None while active (the state is on the device)
        self.snap = None  # (tokens, host state)
        self.last_used = 0

    def reuse(self, ids):
        """(reusable prefix length, how) for prompt ids."""
        n = len(self.processed)
        if n and len(ids) > n and ids[:n] == self.processed:
            return n, "exact"
        if self.snap is not None and len(ids) > len(self.snap[0]) and ids[: len(self.snap[0])] == self.snap[0]:
            return len(self.snap[0]), "snapshot"
        return 0, "none"


class Generator:
    """Several conversations share the KV pool (one page table switched per request) when the engine has the
    large-T prefill (needed for state save/restore); otherwise a single conversation."""

    def __init__(self, eng, eos=EOS_DEFAULT, im_start=IM_START, max_convs=None):
        self.eng = eng
        self.eos = set(eos)
        self.im_start = im_start
        self.k = eng.T_dec - 1
        self.multi = getattr(eng, "large", None) is not None and env("MULTI_CONV", "1") == "1"
        self.max_convs = max_convs or int(env("MAX_CONVS", "8"))
        self.convs = []
        self.cur = None
        self.tick = 0
        eng.pager.on_full = self._evict
        eng.pager.release(eng.pager.blocks)  # blocks mapped by warmup / earlier direct prefill calls
        self._new_conv()

    # the active conversation's fields
    @property
    def processed(self):
        return self.cur.processed

    @processed.setter
    def processed(self, v):
        self.cur.processed = v

    @property
    def snap(self):
        return self.cur.snap

    @snap.setter
    def snap(self, v):
        self.cur.snap = v

    def _new_conv(self):
        c = Conversation()
        self.convs.append(c)
        self.cur = c
        self.eng.pager.activate(c.blocks)
        self.eng.reset_spec()
        while len(self.convs) > self.max_convs:  # host memory: drop the least recently used
            self._drop(min((x for x in self.convs if x is not c), key=lambda x: x.last_used))
        return c

    def _drop(self, c):
        self.eng.pager.release(c.blocks)
        self.convs.remove(c)

    def _evict(self):
        """KV pool full: free the least recently used other conversation."""
        others = [x for x in self.convs if x is not self.cur]
        if not others:
            return False
        self._drop(min(others, key=lambda x: x.last_used))
        return True

    def _switch_out(self):
        c = self.cur
        if c.processed:
            c.state = self.eng.large.save_state()

    def reset(self):
        """Forget every conversation."""
        for c in list(self.convs):
            self._drop(c)
        self._new_conv()

    def _prefill(self, ids):
        """Bring the state to `ids` (all prompt tokens processed). Returns (next token, MTP draft, rows of last
        MTP chunk, prefill info).

        Reuse, best first: (1) the prompt extends exactly what was processed (previous prompt + generated tokens);
        (2) it extends the snapshot taken before the last <|im_start|> of the previous prompt, i.e. the start of
        the previous assistant turn. (2) covers templates that rewrite the assistant turn (thinking blocks are
        dropped from history) and clients that re-encode or edit the reply. The GDN state cannot be rewound, so
        every prompt is prefilled in two parts with a snapshot in between.
        With several conversations, the one with the longest reusable prefix is switched in (its page table and
        saved GDN state); no match starts a new conversation (the others keep their KV until evicted)."""
        eng = self.eng
        t0 = time.perf_counter()
        cands = [self.cur] + [c for c in self.convs if c is not self.cur] if self.multi else [self.cur]
        best, (start, how) = self.cur, (0, "none")
        for c in cands:
            r = c.reuse(ids)
            if r[0] > start:
                best, (start, how) = c, r
        switched = False
        if best is not self.cur:  # another conversation: swap its KV pages and GDN state in
            self._switch_out()
            self.cur = best
            eng.pager.activate(best.blocks)
            if how == "exact":
                eng.large.restore_state(best.state)
            switched = True
        if how == "snapshot":
            eng.large.restore_state(self.cur.snap[1])
        elif how == "none":
            if self.multi and self.cur.processed:  # keep the current conversation, start a new one
                self._switch_out()
                self._new_conv()
                switched = True
            else:
                self.cur.snap = None
                self.cur.processed = []
                eng.reset_spec()
        self.cur.state = None
        self.tick += 1
        self.cur.last_used = self.tick
        t_switch = time.perf_counter() - t0
        k = len(ids) - 1 - ids[::-1].index(self.im_start) if self.im_start in ids else -1
        can_snap = getattr(eng, "large", None) is not None and env("SNAPSHOT", "1") == "1"
        if can_snap and k > start:
            smp, eng.sampler = eng.sampler, None  # the first part's next token is not used
            try:
                eng.prefill(ids[start:k], start=start, with_mtp=True)
            finally:
                eng.sampler = smp
            t1 = time.perf_counter()
            self.snap = (list(ids[:k]), eng.large.save_state())
            t_snap = time.perf_counter() - t1
            nxt, pos, draft = eng.prefill(ids[k:], start=k, with_mtp=True)
        else:
            t_snap = 0.0
            nxt, pos, draft = eng.prefill(ids[start:], start=start, with_mtp=True)
        info = dict(reused=start, reuse=how, prefilled=len(ids) - start, t_prefill=time.perf_counter() - t0,
                    t_snapshot=t_snap, switched=switched, t_switch=t_switch, convs=len(self.convs),
                    kv_blocks=eng.pager.used())
        self.processed = list(ids)
        return nxt, draft, eng._mtp_last_rows, info

    def generate(self, ids, max_new=512, stats=None, sampler=None):
        """Yields generated token ids (speculative with MTP drafts).

        sampler=None is greedy. With a sampler (tsumi/sampling.py), every verified row is sampled from the main
        model's top-k and a draft is accepted when the sample equals it. The drafts are deterministic (MTP argmax),
        so this is exact speculative sampling: the output has the same distribution as sampling token by token."""
        self.eng.sampler = sampler
        try:
            yield from self._generate(ids, max_new, stats)
        finally:
            self.eng.sampler = None

    def _generate(self, ids, max_new, stats):
        eng, k = self.eng, self.k
        cur, d, last_rows, info = self._prefill(ids)
        if stats is not None:
            stats.update(info)
            stats.setdefault("steps", 0)
            stats.setdefault("accepted", 0)
        p = len(ids)
        a_prev = eng._a_prev
        yield cur
        n_out = 1
        if cur in self.eos:
            return
        drafts = [d]
        q = p + 1
        for j in range(1, k):
            am = eng.mtp.run([drafts[-1]], q, [last_rows - 1 if j == 1 else 0])
            drafts.append(am[0])
            q += 1
        while n_out < max_new:
            y = eng.run_spec([cur] + drafts, p, a_prev)
            if eng.sampler is not None:
                y = [eng._pick_row(r) for r in range(k + 1)]
            n = 0
            while n < k and drafts[n] == y[n]:
                n += 1
            new = drafts[:n] + [y[n]]
            if stats is not None:
                stats["steps"] += 1
                stats["accepted"] += n
            # committed to the main state: cur + drafts[:n]
            self.processed += [cur] + drafts[:n]
            stop = False
            for t in new:
                try:
                    yield t
                except GeneratorExit:  # the consumer stopped (client disconnected): same state as a normal stop
                    eng._a_prev = n + 1
                    raise
                n_out += 1
                if t in self.eos or n_out >= max_new:
                    stop = True
                    break
            if stop:
                # the main state holds cur + drafts[:n]; the next prompt must re-feed from there
                eng._a_prev = n + 1
                return
            am = eng.mtp.run(new, p + 1, list(range(n + 1)))
            nd = [am[n]]
            q = p + n + 2
            for j in range(1, k):
                am = eng.mtp.run([nd[-1]], q, [n if j == 1 else 0])
                nd.append(am[0])
                q += 1
            cur = y[n]
            p = p + n + 1
            drafts = nd
            a_prev = n + 1
        eng._a_prev = a_prev
