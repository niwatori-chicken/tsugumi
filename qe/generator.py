"""Speculative generation loop (MTP drafts) with conversation-prefix reuse, for the server and CLI tools.

Greedy (temperature 0) is exact: the output equals plain greedy decoding of the engine.
"""
import time

EOS_DEFAULT = (248046, 248044)


class Generator:
    def __init__(self, eng, eos=EOS_DEFAULT):
        self.eng = eng
        self.eos = set(eos)
        self.processed = []  # tokens whose effect is in the main model state (positions 0..len-1)
        self.k = eng.T_dec - 1

    def reset(self):
        self.eng.reset_spec()
        self.processed = []

    def _prefill(self, ids):
        """Bring the state to `ids[:-0]` (all prompt tokens processed). Reuses the previous conversation
        when the new prompt extends exactly what was processed. Returns (next token, MTP draft, rows of last
        MTP chunk, prefill info)."""
        eng = self.eng
        n_old = len(self.processed)
        if n_old and len(ids) > n_old and ids[:n_old] == self.processed:
            new = ids[n_old:]
            start = n_old
        else:
            self.reset()
            new, start = ids, 0
        t0 = time.perf_counter()
        nxt, pos, draft = eng.prefill_spec(new, start=start, with_mtp=True)
        info = dict(reused=start, prefilled=len(new), t_prefill=time.perf_counter() - t0)
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
