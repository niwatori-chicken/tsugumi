"""Generation loop for the Inkling engine: prompt reuse, prefill, sampling.

The device state holds the tokens self.seq (positions 0..len-1). A request whose prompt extends self.seq only
prefills the rest; otherwise the snapshot taken at the end of the previous prompt is restored when that prompt
is a prefix of the new one (a chat turn: the previous reply, re-rendered without its thinking, follows it), else
the state is reset.
"""
import time

from .prefill import T as _PT

# shorter prompt remainders go one token per decode step (a 32-row prefill step costs ~5 decode steps, a 64-row
# one ~9)
PREFILL_MIN = {32: 6, 64: 10, 128: 12, 256: 16, 512: 24, 1024: 32}[_PT]  # a step ~ this many decode steps


def _greedy(vals, ids):
    """argmax; among equal (bf16) maxima the lowest token id"""
    return int(ids[vals == vals[0]].min())


class InklingGenerator:
    def __init__(self, eng, eos=(200006,), progress=None):
        self.eng = eng
        self.eos = set(eos)
        self.seq = []
        self.last = None  # (vals, ids) after the last processed token
        self.snap_seq = None
        self.snap_last = None
        self.progress = progress  # called during long prefills (hang backstop re-arm)

    def _prepare(self, ids, stats):
        """Bring the device state to a prefix of ids; returns the number of prompt tokens already in it."""
        e = self.eng
        n = len(ids)
        if self.seq and len(self.seq) <= n and ids[:len(self.seq)] == self.seq and \
                (len(self.seq) < n or self.last is not None):
            stats["reuse"] = "continued"
            return len(self.seq)
        if self.snap_seq is not None and len(self.snap_seq) <= n and ids[:len(self.snap_seq)] == self.snap_seq:
            e.snapshot_load()
            self.seq = list(self.snap_seq)
            self.last = self.snap_last
            stats["reuse"] = "snapshot"
            return len(self.seq)
        e.reset()
        self.seq, self.last = [], None
        stats["reuse"] = None
        return 0

    def generate(self, ids, max_new, stats=None, sampler=None, row_cb=None):
        """Yields generated token ids (the eos token too, last). stats gets reused / prefilled / t_prefill / steps.
        row_cb(vals, ids) is called with the top-k logits that each yielded token was picked from."""
        stats = {} if stats is None else stats
        e = self.eng
        t0 = time.perf_counter()
        start = self._prepare(ids, stats)
        stats["reused"] = start
        todo = ids[start:]
        if getattr(e, "has_prefill", False) and len(todo) >= PREFILL_MIN:  # T tokens per step
            self.last = e.prefill(todo, logits=True, progress=self.progress)
            self.seq += todo
        else:
            for k, t in enumerate(todo):
                last = k == len(todo) - 1
                out = e.step(t, logits=last)
                self.seq.append(t)
                if last:
                    self.last = out
        stats["prefilled"] = len(todo)
        if todo:  # the state after the prompt: restorable after the reply
            e.snapshot_save()
            self.snap_seq, self.snap_last = list(self.seq), self.last
        stats["t_prefill"] = time.perf_counter() - t0
        steps = 0
        try:
            for _ in range(max_new):
                vals, tids = self.last
                tok = sampler(vals, tids) if sampler is not None else _greedy(vals, tids)
                if row_cb is not None:
                    row_cb(vals, tids)
                steps += 1
                yield tok
                if tok in self.eos:
                    break
                self.last = e.step(tok, logits=True)
                self.seq.append(tok)
        finally:
            stats["steps"] = steps
