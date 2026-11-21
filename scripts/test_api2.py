"""Checks the thinking budget, logprobs, /v1/completions and /tokenize against a running server.

  python scripts/test_api2.py [URL=http://127.0.0.1:8010]
Greedy everywhere, so streamed and non-streamed replies must match exactly."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import json
import sys
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010").rstrip("/")
fails = []


def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as e:
        return {"_status": e.code, **json.loads(e.read())}


def post_stream(path, body):
    r = urllib.request.Request(URL + path, json.dumps(dict(body, stream=True)).encode(),
                               {"content-type": "application/json"})
    out = []
    with urllib.request.urlopen(r, timeout=600) as f:
        for line in f:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


HARD = "1から1000までの整数のうち、各桁の数字の和が10になるものはいくつありますか？"
EASY = "2+3は？数字だけ答えて。"
chat = lambda q, **kw: dict(messages=[{"role": "user", "content": q}], **kw)
think_on = {"enable_thinking": True}

print("thinking budget")
r = post("/v1/chat/completions", chat(HARD, max_tokens=200, thinking_budget=64, chat_template_kwargs=think_on))
t, m = r["timings"], r["choices"][0]["message"]
check("budget reached", t.get("think_forced") is True, f"think_tokens {t.get('think_tokens')}")
check("thinking within budget + forced + one step", 64 <= t["think_tokens"] <= 64 + 7 + 25, str(t["think_tokens"]))
check("closing sentence in the reasoning", m.get("reasoning_content", "").rstrip().endswith("directly now."))
check("answer after the cap", bool((m.get("content") or "").strip()), repr((m.get("content") or "")[:80]))
ans = r["usage"]["completion_tokens"] - t["think_tokens"]
check("max_tokens counts the answer only", ans <= 200 and (r["choices"][0]["finish_reason"] == "stop" or ans >= 199),
      f"answer tokens {ans}, finish {r['choices'][0]['finish_reason']}")
s = post_stream("/v1/chat/completions", chat(HARD, max_tokens=200, thinking_budget=64, chat_template_kwargs=think_on))
sc = "".join(c["choices"][0]["delta"].get("content") or "" for c in s if c.get("choices"))
sr = "".join(c["choices"][0]["delta"].get("reasoning_content") or "" for c in s if c.get("choices"))
check("stream = non-stream (content, reasoning)", sc == m["content"] and sr == m["reasoning_content"])
r0 = post("/v1/chat/completions", chat(HARD, max_tokens=200, thinking_budget=0, chat_template_kwargs=think_on))
check("budget 0", r0["timings"]["think_forced"] and r0["timings"]["think_tokens"] <= 7 + 25 + 1,
      str(r0["timings"]["think_tokens"]))
r2 = post("/v1/chat/completions", chat(EASY, max_tokens=300, thinking_budget=4000, chat_template_kwargs=think_on))
check("budget not reached", r2["timings"]["think_forced"] is False and "5" in (r2["choices"][0]["message"]["content"] or ""),
      f"think_tokens {r2['timings']['think_tokens']}, {r2['choices'][0]['message']['content']!r}")
r3 = post("/v1/chat/completions", chat(EASY, max_tokens=300, chat_template_kwargs=think_on))
check("no budget: same reply as a budget not reached",
      r3["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"])
check("bad budget -> 400", post("/v1/chat/completions", chat(EASY, thinking_budget="x",
                                                             chat_template_kwargs=think_on)).get("_status") == 400)

print("chat logprobs")
q = "日本の首都について一文で説明してください。"
a = post("/v1/chat/completions", chat(q, max_tokens=60))
b = post("/v1/chat/completions", chat(q, max_tokens=60, logprobs=True, top_logprobs=3))
ca, cb = a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"]
check("logprobs do not change the reply", ca == cb)
lp = b["choices"][0]["logprobs"]["content"]
check("one entry per completion token", len(lp) == b["usage"]["completion_tokens"],
      f"{len(lp)} vs {b['usage']['completion_tokens']}")
check("entry bytes join to the content", bytes(x for e in lp for x in e["bytes"]).decode() == cb)
check("logprobs <= 0, top 3, chosen = top 1 (greedy)",
      all(e["logprob"] <= 1e-9 and len(e["top_logprobs"]) == 3 and e["top_logprobs"][0]["logprob"] >= e["logprob"] - 1e-9
          for e in lp))
print("    first entries:", [(e["token"], round(e["logprob"], 3)) for e in lp[:5]])
s = post_stream("/v1/chat/completions", chat(q, max_tokens=60, logprobs=True, top_logprobs=3))
slp = [e for c in s if c.get("choices") and c["choices"][0].get("logprobs") for e in c["choices"][0]["logprobs"]["content"]]
check("stream logprobs = non-stream", [(e["token"], e["logprob"]) for e in slp] == [(e["token"], e["logprob"]) for e in lp])
bt = post("/v1/chat/completions", chat(EASY, max_tokens=100, logprobs=True, chat_template_kwargs=think_on))
blp = bt["choices"][0]["logprobs"]["content"]
check("thinking: logprobs cover the content only",
      bytes(x for e in blp for x in e["bytes"]).decode().strip() == bt["choices"][0]["message"]["content"].strip(),
      f"{len(blp)} entries")

print("completions")
p = "The three primary colors are"
c1 = post("/v1/completions", dict(prompt=p, max_tokens=24))
c2 = post_stream("/v1/completions", dict(prompt=p, max_tokens=24))
text = c1["choices"][0]["text"]
check("text", len(text) > 5 and c1["object"] == "text_completion", repr(text))
check("stream = non-stream", "".join(c["choices"][0]["text"] for c in c2 if c.get("choices")) == text)
check("default max_tokens 16", post("/v1/completions", dict(prompt=p))["usage"]["completion_tokens"] <= 16)
ids = post("/tokenize", dict(prompt=p))["tokens"]
c3 = post("/v1/completions", dict(prompt=ids, max_tokens=24))
check("token-id prompt = text prompt", c3["choices"][0]["text"] == text)
c4 = post("/v1/completions", dict(prompt=p, max_tokens=24, logprobs=2))
L = c4["choices"][0]["logprobs"]
check("legacy logprobs", c4["choices"][0]["text"] == text and len(L["tokens"]) == c4["usage"]["completion_tokens"] and
      all(len(x) <= 2 for x in L["top_logprobs"]) and L["text_offset"][0] == 0, f"{len(L['tokens'])} tokens")
c5 = post("/v1/completions", dict(prompt=p, max_tokens=24, stop=[","]))
check("stop", "," not in c5["choices"][0]["text"] and c5["choices"][0]["text"] == text.split(",")[0],
      repr(c5["choices"][0]["text"]))
check("bad prompt -> 400", post("/v1/completions", dict(prompt={"x": 1})).get("_status") == 400)

print("tokenize")
t1 = post("/tokenize", dict(prompt="こんにちは、世界"))
check("detokenize round trip", post("/detokenize", dict(tokens=t1["tokens"]))["prompt"] == "こんにちは、世界")
tm = post("/tokenize", dict(messages=[{"role": "user", "content": q}]))
check("messages count = chat prompt_tokens", tm["count"] == a["usage"]["prompt_tokens"],
      f"{tm['count']} vs {a['usage']['prompt_tokens']}")
check("max_model_len", tm["max_model_len"] > 0)

print("FAILED: " + ", ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
