"""Speculative generation loop (MTP drafts) with conversation-prefix reuse, for the server and CLI tools.

Greedy (temperature 0) is exact: the output equals plain greedy decoding of the engine.
"""
import os
import time

from .env import env

EOS_DEFAULT = (248046, 248044)
IM_START = 248045


class NGramIndex:
    """Prompt-lookup drafts: where the last N tokens of the context occurred before, propose what followed them.

    Copy-heavy replies (code edits, quoting the prompt, rewriting a text) repeat long stretches of the context;
    there the proposals are right far more often than the MTP drafts, and they cost nothing to make."""

    def __init__(self, n=3):
        self.n = n
        self.toks = []
        self.ends = {}  # n-gram -> end indices (exclusive) of its occurrences, in order

    def extend(self, tokens):
        n, toks, ends = self.n, self.toks, self.ends
        for t in tokens:
            toks.append(t)
            if len(toks) >= n:
                ends.setdefault(tuple(toks[-n:]), []).append(len(toks))

    def propose(self, k, max_match=32):
        """(k tokens that followed the most recent earlier occurrence of the current suffix, length of the
        matching suffix up to max_match), or ([], 0) if the last N tokens never occurred before."""
        toks, L = self.toks, len(self.toks)
        if L < self.n:
            return [], 0
        occ = self.ends.get(tuple(toks[-self.n:]), ())
        for e in reversed(occ):
            if e < L:
                m = self.n
                while m < max_match and e - m - 1 >= 0 and toks[e - m - 1] == toks[L - m - 1]:
                    m += 1
                return toks[e:e + k], m
        return [], 0


class ThinkBudget:
    """Cap on the thinking: after `budget` tokens inside <think> (the reply starts there), the tokens `force`
    (a closing sentence and </think>, ending with `end`) are inserted and the model goes on with the answer. The
    check runs between steps, so the thinking can run up to one step (T_verify - 1 tokens) past the budget. With a
    budget, max_new counts the answer tokens only (after `end`)."""

    def __init__(self, budget, end, force):
        self.budget, self.end, self.force = max(int(budget), 0), int(end), list(force)
        if self.end not in self.force:
            raise ValueError("the forced tokens must contain the end-of-thinking token")


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
        self.multi = (getattr(eng, "large", None) is not None or getattr(eng, "state_io", None) is not None) and \
            env("MULTI_CONV", "1") == "1"
        self.max_convs = max_convs or int(env("MAX_CONVS", "8"))
        # prompt-lookup drafts: index n-grams of NGRAM tokens (0 = off); use them when the context's suffix matches
        # an earlier stretch over at least NGRAM_MIN tokens, verified in the longer T_verify mode when the engine has one
        self.ngram_n = int(env("NGRAM", "3"))
        self.ngram_min = int(env("NGRAM_MIN", "6"))
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

    def _save(self):
        io = getattr(self.eng, "state_io", None)
        return io.save() if io is not None else self.eng.large.save_state()

    def _restore(self, snap):
        io = getattr(self.eng, "state_io", None)
        return io.restore(snap) if io is not None else self.eng.large.restore_state(snap)

    def _switch_out(self):
        c = self.cur
        if c.processed:
            c.state = self._save()

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
        eng.set_sequence(ids)  # M-RoPE positions of the image tokens (virtual ids, tsugumi/vision.py)
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
                self._restore(best.state)
            switched = True
        if how == "snapshot":
            self._restore(self.cur.snap[1])
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

    def generate(self, ids, max_new=512, stats=None, sampler=None, think=None, row_cb=None):
        """Yields generated token ids (speculative with MTP drafts).

        think: a ThinkBudget (the reply starts inside <think>). row_cb(values, indices): called before each yielded
        token with the main model's top-k logits of the row that predicted it (for logprobs).
        sampler=None is greedy. With a sampler (tsugumi/sampling.py), every verified row is sampled from the main
        model's top-k and a draft is accepted when the sample equals it. The drafts are deterministic (MTP argmax),
        so this is exact speculative sampling: the output has the same distribution as sampling token by token."""
        self.eng.sampler = sampler
        try:
            yield from self._generate(ids, max_new, stats, think, row_cb)
        finally:
            self.eng.sampler = None
            self.eng.use_T(self.eng.T_dec)  # a step may have left the engine in the T_verify mode

    def _generate(self, ids, max_new, stats, think=None, row_cb=None):
        eng, k = self.eng, self.k
        cur, d, last_rows, info = self._prefill(ids)
        if stats is not None:
            stats.update(info)
            stats.setdefault("steps", 0)
            stats.setdefault("accepted", 0)
            if think is not None:
                stats.update(think_tokens=0, think_forced=False)
        p = len(ids)
        a_prev = eng._a_prev
        # limits: without a think budget max_new counts every token; with one, the answer tokens after </think>
        # (the thinking is bounded by the budget, the forced tokens and one step of overshoot)
        st = dict(n=0, think=think is not None, n_think=0, n_ans=0)
        queue = []  # forced tokens still to insert (think budget)

        def account(t):
            """Counts the token just yielded; True when generation must stop."""
            st["n"] += 1
            if eng.sampler is not None:
                eng.sampler.observe(t)  # repetition penalties count the generated tokens
            if st["think"]:
                st["n_think"] += 1
                if t == think.end:
                    st["think"] = False
            elif think is not None:
                st["n_ans"] += 1
            if t in self.eos:
                return True
            return (st["n_ans"] if think is not None else st["n"]) >= max_new

        if row_cb is not None:
            v, i = eng.last_topk
            row_cb(v[eng.last_pick_row], i[eng.last_pick_row])
        yield cur
        if account(cur):
            return
        ng = None
        if self.ngram_n:
            ng = NGramIndex(self.ngram_n)  # built after the first token went out: no cost to the time to first token
            ng.extend(ids)
            ng.extend([cur])
            if stats is not None:
                stats.setdefault("ngram_steps", 0)
                stats.setdefault("ngram_accepted", 0)

        TV = eng.T_verify
        kl = (TV or eng.T_dec) - 1  # drafts in an n-gram step

        def plan(d1, q, row):
            """Drafts for the next step and how to verify them: forced tokens (think budget) are accepted whatever
            the model predicts; else the prompt-lookup proposal if the context's suffix repeats an earlier stretch
            (>= ngram_min tokens), else d1 + MTP chained drafts (T_dec mode). Returns (drafts, mode T, kind)."""
            if think is not None and st["think"] and not queue and st["n_think"] >= think.budget:
                queue.extend(think.force)
                if stats is not None:
                    stats["think_forced"] = True
            if queue:
                take = queue[:kl]
                return take, (TV or eng.T_dec), "forced"
            if ng is not None:
                prop, m = ng.propose(kl)
                if any(t >= eng.n_embed for t in prop):  # never draft image tokens
                    prop = prop[:next(j for j, t in enumerate(prop) if t >= eng.n_embed)]
                if prop and m >= self.ngram_min:
                    return prop + [prop[-1]] * (kl - len(prop)), (TV or eng.T_dec), "ngram"
            ds = [d1]
            for j in range(1, k):
                am = eng.mtp.run([ds[-1]], q, [row if j == 1 else 0])
                ds.append(am[0])
                q += 1
            return ds, eng.T_dec, "mtp"

        drafts, Tm, kind = plan(d, p + 1, last_rows - 1)
        while True:
            eng.use_T(Tm)
            kk = len(drafts)
            y = eng.run_spec([cur] + drafts, p, a_prev)
            if eng.sampler is not None:  # row r: its prefix is the generated tokens + drafts[:r] (penalties)
                v, i = eng.last_topk
                y = [eng.sampler.pick(v[r], i[r], drafts[:r]) for r in range(kk + 1)]
            if kind == "forced":  # all inserted; the token after them is the next forced one or the model's
                n = kk
                del queue[:kk]
                new = drafts + [queue.pop(0) if queue else y[n]]
            else:
                n = 0
                while n < kk and drafts[n] == y[n]:
                    n += 1
                new = drafts[:n] + [y[n]]
            if stats is not None:
                stats["steps"] += 1
                if kind != "forced":
                    stats["accepted"] += n
                if kind == "ngram":
                    stats["ngram_steps"] += 1
                    stats["ngram_accepted"] += n
            # committed to the main state: cur + drafts[:n]
            self.processed += [cur] + drafts[:n]
            stop = False
            for j, t in enumerate(new):
                if row_cb is not None:
                    v, i = eng.last_topk
                    row_cb(v[j], i[j])
                try:
                    yield t
                except GeneratorExit:  # the consumer stopped (client disconnected): same state as a normal stop
                    eng._a_prev = n + 1
                    raise
                if account(t):
                    stop = True
                    break
            if stop:
                # the main state holds cur + drafts[:n]; the next prompt must re-feed from there
                eng._a_prev = n + 1
                if stats is not None and think is not None:
                    stats["think_tokens"] = st["n_think"]
                return
            if stats is not None and think is not None:
                stats["think_tokens"] = st["n_think"]
            if ng is not None:
                ng.extend(new)
            am = eng.mtp.run(new, p + 1, list(range(n + 1)))  # always (in this step's mode): fills the MTP KV
            eng.use_T(eng.T_dec)
            drafts, Tm, kind = plan(am[n], p + n + 2, n)
            cur = new[-1]
            p = p + n + 1
            a_prev = n + 1
