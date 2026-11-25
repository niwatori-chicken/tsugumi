"""Inkling 2 Mini: device engine vs the CPU reference (tsugumi/inkling/ref.py).

    python scripts/inkling_check.py [--prompt TEXT] [--chat] [--layers] [--traced] [--gen N]

Feeds the prompt one token per step, compares the last position's top logits with the reference; --layers also
compares every layer application's output at the last position; --gen N greedy-generates N tokens on the
device and checks them against the reference (teacher-forced).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsugumi.env import env  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "..", "tsugumi-cache", "inkling"))


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Japan is")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--traced", action="store_true")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=8192)
    ap.add_argument("--prefill", action="store_true", help="feed the prompt with the 32-token prefill steps")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the prompt text N times (long prompts)")
    a = ap.parse_args()
    import ttnn
    from transformers import AutoTokenizer

    from tsugumi.inkling.engine import Engine
    from tsugumi.inkling.ref import Reference

    tok = AutoTokenizer.from_pretrained(MODEL)
    a.prompt = " ".join([a.prompt] * a.repeat)
    if a.chat:
        ids = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=True,
                                      add_generation_prompt=True)
        ids = list(ids["input_ids"] if not isinstance(ids, list) else ids)
    else:
        ids = tok.encode(a.prompt, add_special_tokens=False)
    print(f"{len(ids)} prompt tokens")
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=512 << 20)
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=a.max_seq)
        if a.traced:
            t = time.perf_counter()
            eng.capture()
            print(f"capture {time.perf_counter() - t:.1f}s")
        eng.reset()
        if a.prefill:
            t = time.perf_counter()
            vals, idx = eng.prefill(ids, logits=True, traced=a.traced)
            ttnn.synchronize_device(dev)
            tp = time.perf_counter() - t
            print(f"chunked prefill {len(ids)} tokens {tp:.2f}s ({len(ids) / tp:.1f} tok/s)")
        else:
            t = time.perf_counter()
            for p, x in enumerate(ids[:-1]):
                eng.step(x, logits=False, traced=a.traced)
            ttnn.synchronize_device(dev)
            tp = time.perf_counter() - t
            eng.dbg = [] if (a.layers and not a.traced) else None
            t = time.perf_counter()
            vals, idx = eng.step(ids[-1], logits=True, traced=a.traced)
            print(f"prefill {len(ids) - 1} steps {tp:.2f}s ({tp / max(1, len(ids) - 1) * 1e3:.1f} ms/step), "
                  f"last step {(time.perf_counter() - t) * 1e3:.1f} ms")
        dev_layers = eng.dbg
        eng.dbg = None
        gen = []
        if a.gen:
            nxt = int(idx[0])
            t = time.perf_counter()
            for _ in range(a.gen):
                gen.append(nxt)
                v2, i2 = eng.step(nxt, logits=True, traced=a.traced)
                nxt = int(i2[0])
            dt = time.perf_counter() - t
            print(f"generated {a.gen} in {dt:.2f}s ({a.gen / dt:.1f} tok/s): {tok.decode(gen)!r}")
    finally:
        ttnn.close_device(dev)

    torch.set_num_threads(16)
    ref = Reference(MODEL)
    tr = {} if dev_layers else None
    lg = ref.forward(ids, trace=tr)[-1]
    rv, ri = lg.topk(10)
    print("ref top:", [(tok.decode([i]), round(v, 2)) for v, i in zip(rv.tolist(), ri.tolist())])
    print("dev top:", [(tok.decode([i]), round(v, 2)) for v, i in zip(vals[:10].tolist(), idx[:10].tolist())])
    common = [i for i in idx.tolist() if i < lg.shape[0]]
    print(f"top-1 match: {int(idx[0]) == int(ri[0])}, pcc over the device top-{len(common)}: "
          f"{pcc(vals[:len(common)], lg[common]):.4f}, max |d| {(vals[:len(common)] - lg[common]).abs().max():.3f}")
    if dev_layers:
        for k, (dl, rp) in enumerate(zip(dev_layers, tr["apps"])):
            parts = []
            for name in ("q", "k", "v", "r", "attn", "attn_out", "mlp_out", "h_out"):
                if name in dl and name in rp:
                    x, r = dl[name].flatten(), rp[name][-1].flatten()
                    parts.append(f"{name} {pcc(x, r):.4f}/{((x - r).norm() / r.norm()).item():.3f}")
            print(f"app {k:2d} L{rp['layer']:2d} (pcc/rel err): " + "  ".join(parts))
    if gen:
        full = ids + gen
        lgs = ref.forward(full)
        ok = sum(int(lgs[len(ids) - 1 + k].argmax()) == gen[k] for k in range(len(gen)))
        print(f"greedy agreement with the reference (teacher-forced): {ok}/{len(gen)}")
        for k in range(len(gen)):
            row = lgs[len(ids) - 1 + k]
            if int(row.argmax()) != gen[k]:
                top = row.topk(2)
                print(f"  step {k}: ref {tok.decode([int(top.indices[0])])!r} {top.values[0]:.3f}, dev pick "
                      f"{tok.decode([gen[k]])!r} {row[gen[k]]:.3f} (ref top-2 gap {top.values[0] - top.values[1]:.3f})")


if __name__ == "__main__":
    main()
