"""Perplexity of the engine on fixed token chunks (llama-perplexity protocol: each chunk starts fresh,
only the second half is scored).  python scripts/eval_ppl.py IDS.pt OUT.pt [--prec bfp4]

IDS.pt: {name: LongTensor [N, C]}. Saves per chunk: target log-probs of scored positions and the top-K
log-probs (values, indices) for KL against a reference (see ref_ppl.py)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import argparse
import math
import time

import torch
import ttnn

import tsugumi.paths as tsugumi_paths
from tsugumi.model import Precision, Qwen35Engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("out")
    ap.add_argument("--prec", default="bfp4")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--set", nargs="*", default=[], help="per-group dtype overrides, e.g. mlp_down=bfp8")
    ap.add_argument("--cache", default=None, help="cache subdir (default: --prec)")
    ap.add_argument("--large", type=int, default=0, help="use the large-T prefill path with this t_max")
    a = ap.parse_args()
    data = torch.load(a.ids)
    C = next(iter(data.values())).shape[1]
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    res = {}
    try:
        prec = Precision(a.prec, **dict(kv.split("=") for kv in a.set))
        eng = Qwen35Engine(dev, tsugumi_paths.MODEL, prec, f"{tsugumi_paths.CACHE}/{a.cache or a.prec}", max_seq=C + 64,
                           log=lambda *x: None)
        eng.init_spec(4, mtp=False)
        eng.init_prefill(32)
        if a.large:
            eng.init_large_prefill(a.large)
        eng.capture_all()
        V = eng.V
        orig = eng.run_spec
        rows = []

        def wrap(chunk, pos, ap_):
            am = orig(chunk, pos, ap_)
            lg = ttnn.to_torch(eng.last_spec[0])[0, 0, : len(chunk), :V].float()
            rows.append(lg)
            return am

        eng.run_spec = wrap
        for name, ids in data.items():
            out = []
            for n in range(ids.shape[0]):
                t0 = time.time()
                rows.clear()
                eng.reset_spec()
                eng._a_prev = 0
                if a.large:
                    eng.large.keep = []
                    eng.large.run(ids[n].tolist(), start=0, with_mtp=False)
                    pos = 0
                    for hid, nn in eng.large.keep:  # rows of the scored range only
                        lo, hi = max(pos, C // 2 - 1), min(pos + nn, C - 1)
                        if hi > lo:
                            rows.append(eng.large.logits_host(hid, lo - pos, hi - pos))
                        ttnn.deallocate(hid)
                        pos += nn
                    eng.large.keep = None
                    lg = torch.cat([torch.zeros(C // 2 - 1, V)] + rows, 0)  # [C-1, V]; only the scored rows are real
                else:
                    eng.prefill_spec(ids[n].tolist(), start=0)
                    lg = torch.cat(rows, 0)  # [C, V]
                first = C // 2
                lp = torch.log_softmax(lg[first - 1: C - 1], -1)  # predicts tokens first..C-1
                tgt = ids[n, first:C]
                tl = lp.gather(1, tgt[:, None])[:, 0]
                tv, ti = lp.topk(a.topk, -1)
                out.append(dict(tgt=tl, top_v=tv.half(), top_i=ti.int()))
                print(f"[{name} {n}] ppl {math.exp(-tl.mean()):.3f} ({time.time() - t0:.0f}s)", flush=True)
            allt = torch.cat([o["tgt"] for o in out])
            print(f"[{name}] PPL {math.exp(-allt.mean()):.4f} over {len(allt)} tokens", flush=True)
            res[name] = out
        torch.save(res, a.out)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
