"""LLM-as-a-judge benchmarks: Japanese MT-Bench (80 questions x 2 turns) and ELYZA-tasks-100, through any
OpenAI-compatible server (tsugumi, llama-server), judged by Gemini.

  python scripts/judge_bench.py gen BENCH ANSWERS.jsonl --api http://127.0.0.1:8010 [--think] [--max-tokens N]
  python scripts/judge_bench.py judge BENCH ANSWERS.jsonl JUDGED.jsonl [--judge gemini-4-hydrogen]
  python scripts/judge_bench.py summary BENCH JUDGED.jsonl [JUDGED2.jsonl ...]
  python scripts/judge_bench.py corrupt BENCH ANSWERS.jsonl OUT.jsonl [--n 20]   (judge sanity check)
BENCH: jmtbench | elyza. Data under /mnt/HDD/eval/bench/{jmtbench,elyza} (Stability-AI FastChat jp-stable
question_full.jsonl + GPT-4 reference answers + judge prompts; elyza/ELYZA-tasks-100 test.csv).

Generation is greedy for every category (FastChat samples some categories; greedy keeps runs reproducible and
comparable between engines). Both commands resume: items already in the output file are skipped.

Judging (single-answer grading, FastChat):
- MT-Bench turn 1: the Japanese judge prompts (single-v1, single-math-v1 with the reference answer for math /
  reasoning / coding); turn 2: FastChat's multi-turn prompts (single-v1-multi-turn, single-math-v1-multi-turn), with
  the Japanese prompts' rule that the reply must be in the user's language. Score 1-10 from "[[N]]".
- ELYZA-tasks-100: the gold answer and the per-task grading criteria (eval_aspect) with ELYZA's 1-5 rubric.
With --judge gemini-decision-1 (Decisions API: no generation, calibrated probabilities over given levels), the same
prompt is the shared input and one "score" question asks for the rating; the score is the probability-weighted
expected level (1-10 / 1-5, fractional), and the per-level probabilities are stored. A token-per-minute limiter keeps
the free tier (50k TPM) from returning 429.
The judge sees only the public questions and the model's answers. The API key is read from GEMINI_API_KEY or
~/.config/gemini_api_key and never printed."""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

D = "/mnt/HDD/eval/bench"
NEED_REF = {"math", "reasoning", "coding"}


# ---------------- data ----------------
def load(bench):
    if bench == "jmtbench":
        qs = [json.loads(l) for l in open(f"{D}/jmtbench/question_full.jsonl") if l.strip()]
        refs = {r["question_id"]: r["choices"][0]["turns"] for r in map(json.loads, open(f"{D}/jmtbench/gpt-4.jsonl"))}
        return [dict(id=q["question_id"], category=q["category"], turns=q["turns"], ref=refs.get(q["question_id"]))
                for q in qs]
    if bench == "elyza":
        rows = list(csv.DictReader(open(f"{D}/elyza/test.csv", encoding="utf-8")))
        return [dict(id=i, category="elyza", turns=[r["input"]], gold=r["output"], aspect=r["eval_aspect"])
                for i, r in enumerate(rows)]
    raise SystemExit(f"unknown bench {bench}")


def done_ids(path, key="id"):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)[key] for l in open(path) if l.strip()}


# ---------------- generation ----------------
def chat(api, messages, max_tokens, think, model="default", timeout=1800):
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0,
                chat_template_kwargs={"enable_thinking": think})
    r = urllib.request.Request(api.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                               {"content-type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as f:
        d = json.loads(f.read())
    m = d["choices"][0]["message"]
    return dict(text=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
                finish=d["choices"][0].get("finish_reason"), tokens=d.get("usage", {}).get("completion_tokens"),
                seconds=time.time() - t)


def gen(a):
    items = load(a.bench)
    seen = done_ids(a.out)
    with open(a.out, "a") as out:
        for it in items:
            if it["id"] in seen:
                continue
            msgs, answers = [], []
            for q in it["turns"]:
                msgs.append({"role": "user", "content": q})
                r = chat(a.api, msgs, a.max_tokens, a.think)
                msgs.append({"role": "assistant", "content": r["text"]})
                answers.append(r)
            rec = dict(id=it["id"], category=it["category"], answers=[x["text"] for x in answers],
                       reasoning=[x["reasoning"] for x in answers], finish=[x["finish"] for x in answers],
                       tokens=[x["tokens"] for x in answers], seconds=[round(x["seconds"], 2) for x in answers],
                       think=a.think)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{it['id']} {it['category']}: " + ", ".join(f"{t} tok {s:.0f}s {f}" for t, s, f in
                                                                zip(rec["tokens"], rec["seconds"], rec["finish"])),
                  flush=True)


# ---------------- judging ----------------
def api_key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        p = os.path.expanduser("~/.config/gemini_api_key")
        if os.path.exists(p):
            k = open(p).read().strip()
    if not k:
        raise SystemExit("no Gemini API key: set GEMINI_API_KEY or write it to ~/.config/gemini_api_key")
    return k


def gemini(prompt, model, key, system=None, retries=8):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    wait = 10
    for i in range(retries):
        r = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json",
                                                                    "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(r, timeout=300) as f:
                d = json.loads(f.read())
            parts = d["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 503) and i < retries - 1:
                print(f"  judge HTTP {e.code}, retry in {wait}s", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 300)
                continue
            raise SystemExit(f"judge HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if i < retries - 1:
                time.sleep(wait)
                continue
            raise


class TpmLimiter:
    """Keep the input tokens of the last 60 s under a budget; the size of the next call is estimated from the
    tokens-per-character ratio of the calls so far."""

    def __init__(self, tpm):
        self.tpm, self.log, self.ratio = tpm, [], 1.0

    def wait(self, chars):
        need = chars * self.ratio
        while True:
            now = time.time()
            self.log = [(t, n) for t, n in self.log if now - t < 60]
            if not self.log or sum(n for _, n in self.log) + need <= self.tpm:
                return
            time.sleep(60 - (now - self.log[0][0]) + 0.5)

    def add(self, chars, tokens):
        self.log.append((time.time(), tokens))
        self.ratio = max(self.ratio * 0.7 + 0.3 * tokens / max(chars, 1), 0.3)


MT_LEVELS = ["1: 非常に悪い"] + [str(i) for i in range(2, 10)] + ["10: 非常に優れている"]
ELYZA_LEVELS = ["1点: 誤っている、指示に従えていない", "2点: 誤っているが、方向性は合っている",
                "3点: 部分的に誤っている、部分的に合っている", "4点: 合っている", "5点: 役に立つ"]


def decision(text, levels, instructions, key, limiter, retries=8):
    body = {"model": "gemini-decision-1", "input": text,
            "questions": {"rating": {"type": "score", "instructions": instructions, "criteria": levels}}}
    wait = 10
    for i in range(retries):
        limiter.wait(len(text))
        r = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/decisions",
                                   json.dumps(body).encode(), {"content-type": "application/json",
                                                               "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(r, timeout=300) as f:
                d = json.loads(f.read())
            limiter.add(len(text), d.get("usage", {}).get("total_input_tokens", len(text)))
            ans = d["answers"]["rating"]
            probs = {int(k) + 1: round(v, 4) for k, v in ans["probabilities"].items()}
            return ans["score"] + 1, dict(sorted(probs.items()))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 503) and i < retries - 1:
                print(f"  judge HTTP {e.code}, retry in {wait}s", flush=True)
                time.sleep(wait)
                wait = min(wait * 2, 300)
                continue
            raise SystemExit(f"judge HTTP {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if i < retries - 1:
                time.sleep(wait)
                continue
            raise


def judge_prompts():
    ja = {d["name"]: d for d in map(json.loads, open(f"{D}/jmtbench/judge_ja_prompts.jsonl"))}
    en = {d["name"]: d for d in map(json.loads, open(f"{D}/jmtbench/judge_prompts.jsonl"))}
    return ja, en


LANG_RULE = ("The assistant's reply must be in the same language as the user's question (Japanese); "
             "replies in another language should be penalised.")


def mt_prompts(it, ans, ja, en):
    """[(turn, system, prompt)] for both turns of one MT-Bench item."""
    ref = it["category"] in NEED_REF
    q, r = it["turns"], it.get("ref") or ["", ""]
    out = []
    p = ja["single-math-v1" if ref else "single-v1"]
    kw = dict(question=q[0], answer=ans[0])
    if ref:
        kw["ref_answer_1"] = r[0]
    out.append((1, p["system_prompt"], p["prompt_template"].format(**kw)))
    p = en["single-math-v1-multi-turn" if ref else "single-v1-multi-turn"]
    kw = dict(question_1=q[0], answer_1=ans[0], question_2=q[1], answer_2=ans[1])
    if ref:
        kw.update(ref_answer_1=r[0], ref_answer_2=r[1])
    out.append((2, p["system_prompt"] + " " + LANG_RULE, p["prompt_template"].format(**kw)))
    return out


ELYZA_PROMPT = """あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{input}

# 正解例
{gold}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{aspect}

# 回答
{answer}
"""


def parse_score(text, bench):
    if bench == "jmtbench":
        m = re.findall(r"\[\[(\d+(?:\.\d+)?)\]\]", text) or re.findall(r"\[(\d+(?:\.\d+)?)\]", text)
        return float(m[-1]) if m else None
    m = re.findall(r"[1-5]", text.strip())
    return float(m[0]) if m else None


def judge(a):
    key = api_key()
    items = {it["id"]: it for it in load(a.bench)}
    ja, en = judge_prompts() if a.bench == "jmtbench" else (None, None)
    dec = a.judge.startswith("gemini-decision")
    limiter = TpmLimiter(a.tpm)
    seen = {(r["id"], r["turn"]) for r in map(json.loads, open(a.judged))} if os.path.exists(a.judged) else set()
    seen_ids = set()
    with open(a.judged, "a") as out:
        for rec in map(json.loads, open(a.answers)):
            it = items[rec["orig_id"] if "orig_id" in rec else rec["id"]]
            if a.limit and len(seen_ids | {rec["id"]}) > a.limit:
                break
            seen_ids.add(rec["id"])
            if a.bench == "jmtbench":
                jobs = mt_prompts(it, rec["answers"], ja, en)
            else:
                jobs = [(1, None, ELYZA_PROMPT.format(input=it["turns"][0], gold=it["gold"], aspect=it["aspect"],
                                                      answer=rec["answers"][0]))]
            for turn, system, prompt in jobs:
                if (rec["id"], turn) in seen:
                    continue
                extra = {}
                if dec:
                    text = (system + "\n\n" if system else "") + prompt
                    if a.bench == "jmtbench":
                        instr = ("上記の指示に従い、評価対象のAIアシスタントの回答" +
                                 ("（2ターン目）" if turn == 2 else "") + "を1から10の尺度で評価してください。")
                        s, probs = decision(text, MT_LEVELS, instr, key, limiter)
                    else:
                        s, probs = decision(text, ELYZA_LEVELS, "上記の採点基準に従い、回答を1から5点で採点してください。",
                                            key, limiter)
                    text, extra = None, dict(probs=probs)
                    s = round(s, 3)
                else:
                    text = gemini(prompt, a.judge, key, system)
                    s = parse_score(text, a.bench)
                out.write(json.dumps(dict(id=rec["id"], category=rec["category"], turn=turn, score=s,
                                          judge=a.judge, judgment=text, **extra, **{k: rec[k] for k in ("variant",)
                                                                                    if k in rec}),
                                     ensure_ascii=False) + "\n")
                out.flush()
                print(f"{rec['id']} {rec['category']} turn {turn}: {s}", flush=True)
                time.sleep(a.sleep)


def summary(a):
    for path in a.judged:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        ok = [r for r in rows if r["score"] is not None]
        by = {}
        for r in ok:
            by.setdefault(r["category"], []).append(r["score"])
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"{path}: {len(ok)}/{len(rows)} scored, mean {mean([r['score'] for r in ok]):.2f}")
        if any("variant" in r for r in ok):
            vs = {}
            for r in ok:
                vs.setdefault(r.get("variant", "good"), []).append(r["score"])
            for v in vs:
                print(f"  variant {v:10s} {mean(vs[v]):.2f} (n={len(vs[v])})")
            continue
        if a.bench == "jmtbench":
            for t in (1, 2):
                print(f"  turn {t}: {mean([r['score'] for r in ok if r['turn'] == t]):.2f}")
            for c in sorted(by):
                print(f"  {c:11s} {mean(by[c]):.2f} (n={len(by[c])})")


def corrupt(a):
    """Copy the first N answers and add broken versions: the judge should rank good > each variant.
    truncated: first 30% of each answer; refusal: a refusal; wrongnum: every digit shifted by +3 mod 10;
    english: a fixed English reply that does not answer."""
    recs = [json.loads(l) for l in open(a.answers) if l.strip()][:a.n]
    shift = str.maketrans("0123456789", "3456789012")
    var = {"good": lambda t: t,
           "truncated": lambda t: t[:max(len(t) * 3 // 10, 1)],
           "refusal": lambda t: "申し訳ありませんが、その質問にはお答えできません。",
           "wrongnum": lambda t: t.translate(shift),
           "english": lambda t: "Sure! Here is a short answer. I think it depends on many factors."}
    with open(a.out, "w") as out:
        for v, f in var.items():
            for r in recs:
                o = dict(r, id=f"{r['id']}:{v}", orig_id=r["id"], variant=v, answers=[f(t) for t in r["answers"]])
                out.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(recs)} items x {len(var)} variants -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("bench")
    g.add_argument("out")
    g.add_argument("--api", default="http://127.0.0.1:8010")
    g.add_argument("--think", action="store_true")
    g.add_argument("--max-tokens", type=int, default=2048)
    j = sub.add_parser("judge")
    j.add_argument("bench")
    j.add_argument("answers")
    j.add_argument("judged")
    j.add_argument("--judge", default="gemini-4-hydrogen")
    j.add_argument("--sleep", type=float, default=1.0, help="seconds between judge calls (free-tier rate limits)")
    j.add_argument("--tpm", type=int, default=45000, help="input-token budget per minute (decision judge)")
    j.add_argument("--limit", type=int, default=0, help="judge only the first N items")
    c = sub.add_parser("corrupt")
    c.add_argument("bench")
    c.add_argument("answers")
    c.add_argument("out")
    c.add_argument("--n", type=int, default=20)
    s = sub.add_parser("summary")
    s.add_argument("bench")
    s.add_argument("judged", nargs="+")
    a = ap.parse_args()
    {"gen": gen, "judge": judge, "summary": summary, "corrupt": corrupt}[a.cmd](a)


if __name__ == "__main__":
    main()
