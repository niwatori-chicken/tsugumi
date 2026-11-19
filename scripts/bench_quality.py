"""Quality benchmarks on the engine (greedy, zero-shot chat). Data under /mnt/HDD/eval/bench (see scripts/README.md).

  python scripts/bench_quality.py TASK OUT.jsonl [--n N] [--prec bfp4] [--set k=v ...] [--cache bfp4] [--think]

TASK: jcqa (JCommonsenseQA, choice index), jsquad (JSQuAD, extract span), jemhop (JEMHopQA, closed-book short answer),
      mgsm (MGSM-ja, last number), humaneval (HumanEval+, writes samples for EvalPlus), gpqa (GPQA Diamond, letter).
Prompts follow llm-jp-eval's instructions for the JGLUE-style tasks. Per item: prediction, gold, score, tokens, time.
Resumes: items already in OUT are skipped.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import argparse, json, random, re, time, unicodedata
from collections import Counter

_os.environ.setdefault("QE_MULTI_CONV", "0")  # every item is a fresh conversation
_os.environ.setdefault("QE_SNAPSHOT", "0")
D = "/mnt/HDD/eval/bench"


def norm(s):
    return unicodedata.normalize("NFKC", s).strip().strip("。.「」『』\"' ")


def char_f1(pred, gold):
    p, g = Counter(norm(pred)), Counter(norm(gold))
    common = sum((p & g).values())
    if not common:
        return 0.0
    pr, rc = common / sum(p.values()), common / sum(g.values())
    return 2 * pr * rc / (pr + rc)


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", unicodedata.normalize("NFKC", s))
    if not nums:
        return None
    try:
        v = float(nums[-1].replace(",", "").rstrip("."))
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def load(task, n, seed=0):
    items = []
    if task == "jcqa":
        for l in open(f"{D}/jglue/jcommonsenseqa-v1.3_valid-v1.3.json"):
            r = json.loads(l)
            ch = ",".join(f"{i}.{r[f'choice{i}']}" for i in range(5))
            items.append(dict(id=r["q_id"], gold=str(r["label"]), max_new=8, prompt=(
                "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号（例：0）で"
                "するものとします。 回答となる数値をint型で返し、他には何も含めないことを厳守してください。\n\n"
                f"質問：{r['question']}\n選択肢：{ch}")))
    elif task == "jsquad":
        d = json.load(open(f"{D}/jglue/jsquad-v1.3_valid-v1.3.json"))
        for art in d["data"]:
            for para in art["paragraphs"]:
                ctx = para["context"]
                for qa in para["qas"]:
                    items.append(dict(id=qa["id"], gold=qa["answers"][0]["text"], max_new=32, prompt=(
                        "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。 それ以外には何も含めない"
                        f"ことを厳守してください。\n\n文章：{ctx}\n質問：{qa['question']}")))
    elif task == "jemhop":
        for r in json.load(open(f"{D}/jemhopqa/dev.json")):
            items.append(dict(id=r["qid"], gold=r["answer"], max_new=48, prompt=(
                "質問を入力とし、回答を出力してください。回答の他には何も含めないことを厳守してください。\n\n"
                f"質問：{r['question']}")))
    elif task == "mgsm":
        import pandas as pd
        df = pd.read_parquet(f"{D}/mgsm/ja/test-00000-of-00001.parquet")
        for i, r in df.iterrows():
            items.append(dict(id=int(i), gold=r["answer_number"], max_new=None, prompt=(
                f"{r['question']}\n\nステップごとに考えて解き、最後の行に「答え: <数値>」の形で答えてください。")))
    elif task == "humaneval":
        for l in open(f"{D}/humanevalplus/test.jsonl"):
            r = json.loads(l)
            items.append(dict(id=r["task_id"], gold=None, max_new=1024, entry=r["entry_point"], code_prompt=r["prompt"],
                              prompt=("Complete the following Python function. Respond with the complete function "
                                      "(including the signature and any needed imports) in a single ```python code block."
                                      f"\n\n```python\n{r['prompt']}```")))
    elif task == "gpqa":
        import pandas as pd
        df = pd.read_csv(f"{D}/gpqa/gpqa_diamond.csv")
        rnd = random.Random(seed)
        for i, r in df.iterrows():
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4)); rnd.shuffle(order)
            letters = "ABCD"
            gold = letters[order.index(0)]
            body = "\n".join(f"{letters[j]}) {str(opts[k]).strip()}" for j, k in enumerate(order))
            items.append(dict(id=int(i), gold=gold, max_new=None, prompt=(
                "Answer the following multiple choice question. The last line of your response should be of the "
                f"following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.\n\n"
                f"{str(r['Question']).strip()}\n\n{body}")))
    if n and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def score(task, it, text):
    if task == "jcqa":
        m = re.search(r"\d", unicodedata.normalize("NFKC", text))
        return float(bool(m) and m.group(0) == it["gold"]), {}
    if task in ("jsquad", "jemhop"):
        pred = text.strip().split("\n")[0]
        if it["gold"] in ("YES", "NO"):  # zero-shot answers come in Japanese
            p0 = norm(pred)
            pred = "YES" if p0.startswith("はい") else "NO" if p0.startswith("いいえ") else pred
        return char_f1(pred, it["gold"]), {"em": float(norm(pred) == norm(it["gold"]))}
    if task == "mgsm":
        tail = text.split("答え")[-1] if "答え" in text else text
        v = last_number(tail)
        return float(v is not None and abs(float(v) - float(it["gold"])) < 1e-6), {"pred_num": v}
    if task == "gpqa":
        m = re.findall(r"Answer\s*[:：]\s*\(?([ABCD])\)?", text)
        return float(bool(m) and m[-1] == it["gold"]), {"pred_letter": m[-1] if m else None}
    if task == "humaneval":
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
        code = blocks[0] if blocks else text
        if f"def {it['entry']}" not in code:
            code = it["code_prompt"] + code
        head = it["code_prompt"].split("def ")[0]  # imports of the prompt
        return None, {"solution": head + "\n" + code}
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--prec", default="bfp4")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-new", type=int, default=0, help="override (thinking tasks)")
    ap.add_argument("--max-seq", type=int, default=16384)
    a = ap.parse_args()
    items = load(a.task, a.n)
    done = set()
    if _os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[{a.task}] {len(items)} items, {len(todo)} to run", flush=True)
    if not todo:
        return summarize(a.out, a.task)

    import ttnn
    from transformers import AutoTokenizer
    import qe.paths as qe_paths
    from qe.model import Precision, Qwen35Engine
    from qe.generator import Generator

    tok = AutoTokenizer.from_pretrained(qe_paths.MODEL)
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    try:
        prec = Precision(a.prec, **dict(kv.split("=") for kv in a.set))
        eng = Qwen35Engine(dev, qe_paths.MODEL, prec, f"{qe_paths.CACHE}/{a.cache or a.prec}", max_seq=a.max_seq,
                           log=lambda *x: None)
        eng.init_spec(4, mtp=True); eng.init_prefill(32); eng.init_large_prefill(1024); eng.capture_all()
        gen = Generator(eng)
        default_new = 4096 if a.think else 768
        with open(a.out, "a") as f:
            for j, it in enumerate(todo):
                ids = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True,
                                              enable_thinking=a.think)
                ids = ids if isinstance(ids, list) else ids["input_ids"]
                max_new = a.max_new or it["max_new"] or default_new
                max_new = min(max_new, a.max_seq - len(ids) - 8)
                t0 = time.perf_counter()
                gen.reset()
                out = [t for t in gen.generate(ids, max_new=max_new) if t not in gen.eos]
                dt = time.perf_counter() - t0
                text = tok.decode(out)
                full = text
                if a.think and "</think>" in text:
                    text = text.split("</think>", 1)[1]
                s, extra = score(a.task, it, text.strip())
                rec = dict(id=it["id"], score=s, gold=it["gold"], pred=text.strip()[-2000:], n_out=len(out),
                           truncated=len(out) >= max_new, sec=round(dt, 2), think=a.think, prec=a.prec, set=a.set,
                           **extra)
                if a.think:
                    rec["think_tokens"] = len(tok.encode(full.split("</think>", 1)[0])) if "</think>" in full else len(out)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if j % 20 == 0 or a.think:
                    print(f"  {j + 1}/{len(todo)} score={s} n_out={len(out)} {dt:.1f}s", flush=True)
    finally:
        ttnn.close_device(dev)
    summarize(a.out, a.task)


def summarize(path, task):
    recs = [json.loads(l) for l in open(path)]
    if task == "humaneval":
        sp = path.replace(".jsonl", "_samples.jsonl")
        with open(sp, "w") as f:
            for r in recs:
                f.write(json.dumps({"task_id": r["id"], "solution": r["solution"]}) + "\n")
        print(f"[humaneval] {len(recs)} samples -> {sp} (score with EvalPlus)")
        return
    sc = [r["score"] for r in recs]
    msg = f"[{task}] n={len(sc)} score={sum(sc) / len(sc):.4f}"
    if recs and "em" in recs[0]:
        msg += f" em={sum(r['em'] for r in recs) / len(recs):.4f}"
    msg += f" truncated={sum(r['truncated'] for r in recs)} avg_out={sum(r['n_out'] for r in recs) / len(recs):.0f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
