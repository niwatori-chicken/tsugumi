"""Sampling (tsugumi/sampling.py + Generator(sampler=...)).

  1. seed: the same seed gives the same text, other seeds differ.
  2. exactness of speculative sampling: tokens 2 and 3 sampled with MTP drafts (several tokens per verify step) vs.
     one token per call (each token from a fresh prefill). Total variation distance between the two empirical
     distributions, next to the distance between two halves of the pooled samples (what noise alone gives).
  3. speed and accepted drafts per step: greedy vs temperature 0.7 / 1.0.
  4. repetition penalties: greedy + presence / frequency penalty with drafts (MTP and n-gram) gives exactly the
     tokens of one token per call (the penalty of every row must see its own prefix); then speed.
  python scripts/test_sampling.py [N=300]      (EXACT_PROMPT=... for another prompt in part 2; PARTS=4 for some parts)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import collections, faulthandler, random, sys, time
import ttnn
from transformers import AutoTokenizer
import tsugumi.paths as tsugumi_paths
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.generator import Generator
from tsugumi.sampling import Sampler

faulthandler.dump_traceback_later(3600, exit=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
tok = AutoTokenizer.from_pretrained(tsugumi_paths.MODEL)


def enc(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
    return ids if isinstance(ids, list) else ids["input_ids"]


def tv(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsugumi_paths.MODEL, Precision("bfp4"), f"{tsugumi_paths.CACHE}/bfp4", max_seq=16384,
                       log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.init_large_prefill(1024); eng.capture_all()
    gen = Generator(eng)
    base = dev.num_program_cache_entries()

    def run(ids, n, sampler=None):
        st = {}
        t0 = time.perf_counter()
        out = list(gen.generate(ids, max_new=n, stats=st, sampler=sampler))
        return out, time.perf_counter() - t0, st

    parts = _os.environ.get("PARTS", "1234")

    if "4" in parts:
        print("4. penalties: speculative = one token per call (greedy, deterministic)", flush=True)
        for text, kw in (("「りんご」という単語を10回、読点で区切って繰り返してください。", dict(presence_penalty=1.5)),
                         ("次の文をそのまま3回書いてください: 猫は夜に静かに歩く。", dict(presence_penalty=1.5)),
                         ("短い物語を書いてください。主人公は猫です。", dict(frequency_penalty=0.5)),
                         ("短い物語を書いてください。主人公は猫です。", dict(temperature=0.7, top_p=0.8, top_k=20,
                                                                    presence_penalty=1.5, seed=3))):
            S = enc(text)
            n = 48
            kw0 = dict(kw)
            seed = kw0.pop("seed", None)
            t = kw0.pop("temperature", 0.0)
            gen.reset()
            spec, _, st = run(S, n, Sampler(t, seed=seed, **kw0))
            if t == 0:
                s1 = Sampler(0.0, **kw0)
                ref = []
                for j in range(len(spec)):
                    ref.append(run(S + ref, 1, s1)[0][0])
                    if ref[-1] in gen.eos:
                        break
                plain, _, _ = run(S, n)
                print(f"  {kw}: identical {spec == ref} ({len(spec)} tok, accepted/step "
                      f"{st['accepted'] / max(st['steps'], 1):.2f}, n-gram steps {st.get('ngram_steps', 0)}); "
                      f"differs from no penalty: {spec != plain[:len(spec)]}", flush=True)
                if spec != ref:
                    k = next(j for j in range(min(len(spec), len(ref))) if spec[j] != ref[j])
                    print(f"    first difference at {k}: {tok.decode(spec[:k + 1])!r} vs {tok.decode(ref[:k + 1])!r}")
            else:
                print(f"  {kw}: {len(spec)} tok, accepted/step {st['accepted'] / max(st['steps'], 1):.2f}", flush=True)
            print("   ", tok.decode(spec)[:90].replace("\n", " "), flush=True)
        R = enc("Pythonでクイックソートを実装して、計算量を説明してください。")
        for name, smp in (("greedy", None), ("greedy pp 1.5", Sampler(0.0, presence_penalty=1.5)),
                          ("t=0.7 top_p 0.8 top_k 20 pp 1.5", Sampler(0.7, 0.8, 20, seed=0, presence_penalty=1.5))):
            gen.reset()
            out, dt, st = run(R, 300, smp)
            print(f"  speed {name}: {len(out)} tok, {len(out) / (dt - st['t_prefill']):.1f} tok/s, "
                  f"accepted/step {st['accepted'] / max(st['steps'], 1):.2f}", flush=True)
        if parts == "4":
            raise SystemExit(0)

    print("1. seed", flush=True)
    P = enc("短い物語を書いてください。主人公は猫です。")
    a = run(P, 60, Sampler(0.8, 0.95, 20, seed=1))[0]
    b = run(P, 60, Sampler(0.8, 0.95, 20, seed=1))[0]
    c = run(P, 60, Sampler(0.8, 0.95, 20, seed=2))[0]
    print(f"  same seed identical: {a == b}   other seed differs: {a != c}", flush=True)
    print("  seed 1:", tok.decode(a)[:80].replace("\n", " "), flush=True)
    print("  seed 2:", tok.decode(c)[:80].replace("\n", " "), flush=True)

    print(f"2. exactness (N={N}, temperature 1.0, top_k 64)", flush=True)
    Q = enc(_os.environ.get("EXACT_PROMPT", "Name one animal. Answer with a single word."))
    rng = random.Random(0)
    spec, ref = [], []
    t0 = time.perf_counter()
    for i in range(N):
        out, _, _ = run(Q, 3, Sampler(1.0, seed=rng.getrandbits(62)))
        spec.append(tuple(out[:3]))
        s = Sampler(1.0, seed=rng.getrandbits(62))
        seq = []
        for j in range(3):  # one token per call: always from a prefill of the prompt + tokens so far
            seq.append(run(Q + seq, 1, s)[0][0])
            if seq[-1] in gen.eos:
                break
        ref.append(tuple(seq))
        if i % 50 == 49:
            print(f"  {i + 1}/{N} {time.perf_counter() - t0:.0f}s", flush=True)
    pool = spec + ref
    rng.shuffle(pool)
    h = len(pool) // 2
    E = 248046  # <|im_end|>; a finished reply has no token 3
    for name, f in (("token 1", lambda x: x[0]), ("token 2", lambda x: x[1] if len(x) > 1 else E),
                    ("token 3", lambda x: x[2] if len(x) > 2 else E), ("tokens 1-3", lambda x: x)):
        d = tv([f(x) for x in spec], [f(x) for x in ref])
        noise = tv([f(x) for x in pool[:h]], [f(x) for x in pool[h:]])
        show = lambda xs: [(tok.decode(list(k) if isinstance(k, tuple) else [k]), n)
                           for k, n in collections.Counter(f(x) for x in xs).most_common(4)]
        print(f"  {name}: TV(spec, ref) {d:.3f}   TV(half, half) {noise:.3f}\n    spec {show(spec)}\n    ref  {show(ref)}",
              flush=True)

    print("3. speed", flush=True)
    R = enc("Pythonでクイックソートを実装して、計算量を説明してください。")
    for name, smp in (("greedy", None), ("t=0.7 top_p 0.8 top_k 20", Sampler(0.7, 0.8, 20, seed=0)),
                      ("t=1.0", Sampler(1.0, seed=0))):
        gen.reset()
        out, dt, st = run(R, 300, smp)
        steps = max(st["steps"], 1)
        print(f"  {name}: {len(out)} tok, {len(out) / (dt - st['t_prefill']):.1f} tok/s, "
              f"accepted/step {st['accepted'] / steps:.2f}", flush=True)
    print(f"program cache +{dev.num_program_cache_entries() - base}", flush=True)
finally:
    ttnn.close_device(dev)
