"""Conversation reuse: 3 turns (thinking off / on) through Generator; reuse kind, TTFT, snapshot time,
and the reply with reuse vs from scratch.  python scripts/test_reuse.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import faulthandler, sys, time, torch, ttnn
faulthandler.dump_traceback_later(90, repeat=True, file=sys.stderr)
from transformers import AutoTokenizer
import tsumi.paths as tsumi_paths
from tsumi.model import Precision, Qwen35Engine
from tsumi.generator import Generator

tok = AutoTokenizer.from_pretrained(tsumi_paths.MODEL)
enc = lambda m, th: (lambda x: x if isinstance(x, list) else x["input_ids"])(
    tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=th))
doc = open("/mnt/HDD/eval/ja_wiki.txt").read()[:5000]
turns = [doc + "\n\nこの文章の要点を1文で。", "標高は何メートル？", "最後に噴火したのはいつ？"]
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    eng = Qwen35Engine(dev, tsumi_paths.MODEL, Precision("bfp4"), f"{tsumi_paths.CACHE}/bfp4", max_seq=16384, log=lambda *a: None)
    eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.init_large_prefill(1024); eng.capture_all()
    gen = Generator(eng)
    base = dev.num_program_cache_entries()

    def run(ids, max_new):
        st = {}; out = []; t0 = time.perf_counter(); t1 = None
        for t in gen.generate(ids, max_new=max_new, stats=st):
            if t1 is None: t1 = time.perf_counter()
            out.append(t)
        return out, t1 - t0, st

    def reply_text(out, th):
        txt = tok.decode([t for t in out if t not in gen.eos])
        if th and "</think>" in txt:
            return txt.split("</think>", 1)[1].strip()  # client keeps only the answer
        return txt
    for th in ((False,) if "--off" in sys.argv else (False, True)):
        gen.reset()
        msgs = []
        for j, u in enumerate(turns):
            msgs.append({"role": "user", "content": u})
            ids = enc(msgs, th)
            torch.save(ids, f'/mnt/HDD/eval/reuse_ids_{int(th)}_{j}.pt')
            out, ttft, st = run(ids, 400 if th else 120)
            ans = reply_text(out, th)
            print(f"[think={th} turn {j + 1}] prompt {len(ids)}  reuse={st['reuse']} reused={st['reused']}  "
                  f"ttft {ttft:.2f}s  snapshot {st['t_snapshot'] * 1e3:.0f}ms  cache +{dev.num_program_cache_entries() - base}  -> {ans[:60]!r}", flush=True)
            if j == 2:  # same prompt from scratch
                gen.reset()
                out2, ttft2, st2 = run(ids, 400 if th else 120)
                same = out == out2
                n = next((i for i in range(min(len(out), len(out2))) if out[i] != out2[i]), min(len(out), len(out2)))
                print(f"   from scratch: ttft {ttft2:.2f}s  identical={same} (first {n} tokens equal)", flush=True)
            msgs.append({"role": "assistant", "content": ans})
finally:
    ttnn.close_device(dev)
