"""Compare eval_ppl.py results against the ref_ppl.py reference.

    python scripts/compare_ppl.py REF.pt RUN.pt [RUN2.pt ...]

Per language: PPL, its change vs the reference, mean KL(ref || run) over the reference's top-K tokens
(tokens outside the run's saved top-K get the run's smallest saved log-prob, an upper bound on q),
and how often the argmax agrees with the reference.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import math
import sys

import torch


def stats(ref, run):
    kl, agree, n, cover = 0.0, 0, 0, 0.0
    for r, q in zip(ref, run):
        pv, pi = r["top_v"].float(), r["top_i"].long()  # [P, K]
        qv, qi = q["top_v"].float(), q["top_i"].long()  # [P, Kq]
        P, V = pv.shape[0], int(max(pi.max(), qi.max())) + 1
        qfull = qv[:, -1:].expand(P, V).clone()
        qfull.scatter_(1, qi, qv)
        lq = qfull.gather(1, pi)
        p = pv.exp()
        kl += (p * (pv - lq)).sum().item()
        cover += p.sum().item()
        agree += (pi[:, 0] == qi[:, 0]).sum().item()
        n += P
    return kl / n, agree / n, cover / n


def ppl(res):
    t = torch.cat([o["tgt"].float() for o in res])
    return math.exp(-t.mean().item())


def main():
    ref = torch.load(sys.argv[1])
    runs = {p: torch.load(p) for p in sys.argv[2:]}
    for lang in ref:
        pr = ppl(ref[lang])
        print(f"== {lang}: reference PPL {pr:.4f}")
        for name, run in runs.items():
            pq = ppl(run[lang])
            kl, ag, cov = stats(ref[lang], run[lang])
            print(f"  {name}: PPL {pq:.4f} ({(pq / pr - 1) * 100:+.2f}%), KL {kl:.4f} nats "
                  f"(ref top-K mass {cov:.3f}), top-1 agree {ag * 100:.1f}%")


if __name__ == "__main__":
    main()
