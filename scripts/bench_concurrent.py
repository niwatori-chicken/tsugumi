"""Throughput with concurrent clients against a running server (OpenAI chat API, streaming).
  python scripts/bench_concurrent.py [--url http://127.0.0.1:8010] [--max-tokens 256] [--clients 1,2,4,8] [--rounds 1]
Each client sends --rounds requests back to back (different prompts). Prints, per concurrency level, the total
generated tokens / wall time, the mean per-request decode speed and the mean time to first token."""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "Pythonでクイックソートを実装して、計算量も説明してください。",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "日本の四季について、それぞれの特徴を説明してください。",
    "Explain how a transformer neural network works, step by step.",
    "C言語で連結リストを反転する関数を書いてください。",
    "What are the main differences between TCP and UDP?",
    "富士山について小学生向けに説明してください。",
    "Write a haiku sequence about autumn rain, then explain each one.",
    "SQLのJOINの種類を例とともに説明してください。",
    "Describe the water cycle in detail.",
    "RustとGoの違いを比較してください。",
    "Give me a 7-day beginner running plan.",
    "量子コンピュータの基本原理をわかりやすく説明してください。",
    "Explain the CAP theorem with examples.",
    "おすすめの朝ごはんのレシピを3つ教えてください。",
    "Write a bash script that backs up a directory with rotation.",
]


def one(url, prompt, max_tokens, key, res):
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                       "stream": True, "temperature": 0}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"content-type": "application/json",
                                                                                     **({"authorization": f"Bearer {key}"} if key else {})})
    t0 = time.perf_counter()
    first, n, timings = None, 0, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("timings"):
                timings = d["timings"]
            ch = d.get("choices") or [{}]
            if ch[0].get("delta", {}).get("content") and first is None:
                first = time.perf_counter() - t0
    res.append(dict(t=time.perf_counter() - t0, ttft=first or 0.0, n=(timings or {}).get("completion_tokens", 0),
                    tps=(timings or {}).get("decode_tps", 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--clients", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--key", default="")
    a = ap.parse_args()
    k = 0
    for c in [int(x) for x in a.clients.split(",")]:
        res = []

        def client(j):
            nonlocal k
            for r in range(a.rounds):
                p = PROMPTS[(j * a.rounds + r + k) % len(PROMPTS)]
                one(a.url, p, a.max_tokens, a.key, res)

        t0 = time.perf_counter()
        th = [threading.Thread(target=client, args=(j,)) for j in range(c)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.perf_counter() - t0
        k += c * a.rounds
        tot = sum(r["n"] for r in res)
        print(f"clients {c}: {len(res)} requests, {tot} tokens in {wall:.1f}s = {tot / wall:.1f} tok/s total; "
              f"per request: decode {sum(r['tps'] for r in res) / len(res):.1f} tok/s, "
              f"TTFT mean {sum(r['ttft'] for r in res) / len(res):.2f}s max {max(r['ttft'] for r in res):.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
