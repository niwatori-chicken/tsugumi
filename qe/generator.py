"""Speculative generation loop (MTP drafts) with conversation-prefix reuse, for the server and CLI tools.

Greedy (temperature 0) is exact: the output equals plain greedy decoding of the engine.
"""
import os
import time

EOS_DEFAULT = (248046, 248044)
IM_START = 248045


class Generator:
    def __init__(self, eng, eos=EOS_DEFAULT, im_start=IM_START):
        self.eng = eng
        self.eos = set(eos)
        self.im_start = im_start
        self.processed = []  # tokens whose effect is in the main model state (positions 0..len-1)
        self.snap = None  # (tokens, host state) saved just before the last <|im_start|> of the previous prompt
        self.k = eng.T_dec - 1

    def reset(self):
        self.eng.reset_spec()
        self.processed = []
        self.snap = None

    def _prefill(self, ids):
        """Bring the state to `ids` (all prompt tokens processed). Returns (next token, MTP draft, rows of last
        MTP chunk, prefill info).

        Reuse, best first: (1) the prompt extends exactly what was processed (previous prompt + generated tokens);
        (2) it extends the snapshot taken before the last <|im_start|> of the previous prompt, i.e. the start of
        the previous assistant turn. (2) covers templates that rewrite the assistant turn (thinking blocks are
        dropped from history) and clients that re-encode or edit the reply. The GDN state cannot be rewound, so
        every prompt is prefilled in two parts with a snapshot in between."""
        eng = self.eng
        n_old = len(self.processed)
        how = "none"
        if n_old and len(ids) > n_old and ids[:n_old] == self.processed:
            start, how = n_old, "exact"
        elif self.snap is not None and len(ids) > len(self.snap[0]) and ids[: len(self.snap[0])] == self.snap[0]:
            eng.large.restore_state(self.snap[1])
            start, how = len(self.snap[0]), "snapshot"
        else:
            self.reset()
            start = 0
        t0 = time.perf_counter()
        k = len(ids) - 1 - ids[::-1].index(self.im_start) if self.im_start in ids else -1
        can_snap = getattr(eng, "large", None) is not None and os.environ.get("QE_SNAPSHOT", "1") == "1"
        if can_snap and k > start:
            eng.prefill(ids[start:k], start=start, with_mtp=True)
            t1 = time.perf_counter()
            self.snap = (list(ids[:k]), eng.large.save_state())
            t_snap = time.perf_counter() - t1
            nxt, pos, draft = eng.prefill(ids[k:], start=k, with_mtp=True)
        else:
            t_snap = 0.0
            nxt, pos, draft = eng.prefill(ids[start:], start=start, with_mtp=True)
        info = dict(reused=start, reuse=how, prefilled=len(ids) - start, t_prefill=time.perf_counter() - t0,
                    t_snapshot=t_snap)
        self.processed = list(ids)
        return nxt, draft, eng._mtp_last_rows, info

    def generate(self, ids, max_new=512, stats=None):
        """Yields generated token ids (greedy, speculative with MTP)."""
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
                yield t
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
