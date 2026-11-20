"""Tool calling through the server (OpenAI API): a small agent loop with fake tools.

  python scripts/test_tools.py [URL]      (default http://127.0.0.1:8010; start the server first)
Cases: non-stream / stream, thinking off / on, parallel calls, tool_choice none / required / named.
Prints each turn's tool calls, the final answer, and the server's prompt reuse (the `timings` field)."""
import json
import sys
import time
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010") + "/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "指定した都市の現在の天気と気温 (摂氏) を返す",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "都市名"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calc", "description": "四則演算の式を計算する",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
]
WEATHER = {"東京": ("晴れ", 18.5), "大阪": ("くもり", 21.0), "札幌": ("雪", -2.0)}


def run_tool(name, args):
    if name == "get_weather":
        w = WEATHER.get(args.get("city"), ("不明", None))
        return json.dumps({"city": args.get("city"), "weather": w[0], "temp_c": w[1]}, ensure_ascii=False)
    if name == "calc":
        try:
            return str(eval(args["expr"], {"__builtins__": {}}))
        except Exception as e:
            return f"error: {e}"
    return "unknown tool"


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        if not body.get("stream"):
            return json.load(r)
        msg, calls, fin = {"content": "", "reasoning_content": ""}, {}, None
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d["choices"][0]
            delta = ch["delta"]
            for k in ("content", "reasoning_content"):
                msg[k] += delta.get(k) or ""
            for tc in delta.get("tool_calls") or []:  # accumulate like an OpenAI client
                c = calls.setdefault(tc["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                c["id"] += tc.get("id") or ""
                c["function"]["name"] += tc["function"].get("name") or ""
                c["function"]["arguments"] += tc["function"].get("arguments") or ""
            fin = ch["finish_reason"] or fin
        m = {"role": "assistant", "content": msg["content"] or None}
        if calls:
            m["tool_calls"] = [calls[i] for i in sorted(calls)]
        if msg["reasoning_content"]:
            m["reasoning_content"] = msg["reasoning_content"]
        return {"choices": [{"message": m, "finish_reason": fin}], "timings": {}}


def agent(label, question, stream=False, thinking=False, tool_choice=None, max_turns=6):
    print(f"=== {label}", flush=True)
    msgs = [{"role": "user", "content": question}]
    for turn in range(max_turns):
        body = {"messages": msgs, "tools": TOOLS, "stream": stream, "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if tool_choice and turn == 0:
            body["tool_choice"] = tool_choice
        t0 = time.time()
        r = post(body)
        ch = r["choices"][0]
        m = ch["message"]
        t = r.get("timings") or {}
        info = f"{time.time() - t0:.1f}s prompt {t.get('prompt_tokens', '?')} reused {t.get('reused', '?')}"
        calls = m.get("tool_calls") or []
        print(f"  turn {turn + 1} [{ch['finish_reason']}] {info}"
              + (f" think {len(m.get('reasoning_content', ''))} chars" if m.get("reasoning_content") else ""), flush=True)
        if m.get("content"):
            print("    content:", m["content"].strip().replace("\n", " ")[:200], flush=True)
        msgs.append({k: v for k, v in m.items() if k != "reasoning_content"})
        if not calls:
            return m.get("content")
        for c in calls:
            args = json.loads(c["function"]["arguments"])
            res = run_tool(c["function"]["name"], args)
            print(f"    call {c['function']['name']}({args}) -> {res}", flush=True)
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": res})
    print("  (max turns)")


Q = "東京と大阪の今の気温を調べて、どちらが何度高いか計算して教えてください。"
agent("non-stream, thinking off", Q)
agent("stream, thinking off", Q, stream=True)
agent("non-stream, thinking on", Q, thinking=True)
agent("stream, thinking on", "札幌の天気は？", stream=True, thinking=True)
agent("tool_choice none", "札幌の天気は？", tool_choice="none")
agent("tool_choice required", "こんにちは", tool_choice="required", max_turns=2)
agent("tool_choice named", "1234*5678 は？", tool_choice={"type": "function", "function": {"name": "calc"}})
agent("no tool needed", "富士山の標高は？一文で。")
