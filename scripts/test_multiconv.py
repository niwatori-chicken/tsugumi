"""Several conversations sharing the KV pool (Generator slots + KVPager).

  1. A and B alternate (A1 B1 A2 B2 A3 B3); every reply must equal the reply of running A alone / B alone.
  2. With a small pool (max_seq 8192), a third conversation C evicts the least recently used one; going back to
     the evicted conversation must re-prefill and still answer the same.
Also prints reuse kind, switch time, TTFT and program cache growth after the trace capture.
  python scripts/test_multiconv.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import faulthandler, time, ttnn
faulthandler.dump_traceback_later(1500, exit=True)
from transformers import AutoTokenizer
import tsumi.paths as tsumi_paths
from tsumi.model import Precision, Qwen35Engine
from tsumi.generator import Generator

tok = AutoTokenizer.from_pretrained(tsumi_paths.MODEL)
enc = lambda m: (lambda x: x if isinstance(x, list) else x["input_ids"])(
    tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False))
wiki = open("/mnt/HDD/eval/ja_wiki.txt").read()
en = open("/mnt/HDD/eval/en_eval.txt").read()
CONVS = {
    "A": [wiki[:5000] + "\n\nこの文章の要点を1文で。", "標高は何メートル？", "最後に噴火したのはいつ？"],
    "B": [en[:9000] + "\n\nSummarize the text above in one sentence.", "Who is mentioned first?", "Translate your summary into Japanese."],
    "C": [wiki[20000:25000] + "\n\nこの文章の主題は？", "もう少し詳しく。"],
}
NEW = 60


def ask(gen, hist, msg):
    hist.append({"role": "user", "content": msg})
    ids = enc(hist)
    st, out, t0, t1 = {}, [], time.perf_counter(), None
    for t in gen.generate(ids, max_new=NEW, stats=st):
        if t1 is None:
            t1 = time.perf_counter()
        out.append(t)
    ans = tok.decode([t for t in out if t not in gen.eos])
    hist.append({"role": "assistant", "content": ans})
    return out, t1 - t0, st, len(ids)


def run_schedule(gen, schedule, dev, base, label):
    hist = {k: [] for k in CONVS}
    turn = {k: 0 for k in CONVS}
    outs = {k: [] for k in CONVS}
    for k in schedule:
        out, ttft, st, n = ask(gen, hist[k], CONVS[k][turn[k]])
        turn[k] += 1
        outs[k].append(out)
        print(f"  [{label}] {k}{turn[k]}: prompt {n}  reuse={st['reuse']} reused={st['reused']} switched={st['switched']} "
              f"switch {st['t_switch'] * 1e3:.0f}ms  ttft {ttft:.2f}s  convs {st['convs']} kv blocks {st['kv_blocks']}  "
              f"cache +{dev.num_program_cache_entries() - base}", flush=True)
    return outs


def engine(max_seq):
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    eng = Qwen35Engine(dev, tsumi_paths.MODEL, Precision("bfp4"), f"{tsumi_paths.CACHE}/bfp4", max_seq=max_seq,
                       log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.init_large_prefill(1024); eng.capture_all()
    return dev, eng


dev, eng = engine(16384)
try:
    gen = Generator(eng)
    base = dev.num_program_cache_entries()
    print("1. alone", flush=True)
    alone = {}
    for k in ("A", "B"):
        gen.reset()
        alone[k] = run_schedule(gen, [k] * 3, dev, base, "alone")[k]
    gen.reset()
    print("1. interleaved", flush=True)
    inter = run_schedule(gen, ["A", "B", "A", "B", "A", "B"], dev, base, "inter")
    for k in ("A", "B"):
        for j in range(3):
            same = alone[k][j] == inter[k][j]
            print(f"  {k}{j + 1}: identical to alone = {same}", flush=True)
finally:
    ttnn.close_device(dev)

dev, eng = engine(8192)  # 128 blocks: A + B fill most of it, C forces an eviction
try:
    gen = Generator(eng)
    base = dev.num_program_cache_entries()
    print("2. eviction (max_seq 8192)", flush=True)
    ev = run_schedule(gen, ["A", "B", "C", "A"], dev, base, "evict")
    print(f"  A2 after eviction identical to alone = {ev['A'][1] == alone['A'][1]}", flush=True)
finally:
    ttnn.close_device(dev)
