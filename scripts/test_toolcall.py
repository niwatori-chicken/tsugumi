"""Tool-call parsing (tsugumi/toolcall.py), host only: whole text vs. every split into streamed pieces, type
conversion, unterminated calls, and a template round trip (parsed calls rendered back give the same text).
  python scripts/test_toolcall.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import json
from tsugumi.toolcall import ToolStreamParser, prepare_messages, forced_prefix

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "detail": {"type": "boolean"},
    "opts": {"type": "object"}}}}},
         {"type": "function", "function": {"name": "run", "parameters": {"type": "object", "properties": {
             "cmd": {"type": "string"}}}}}]

CALL1 = "<tool_call>\n<function=get_weather>\n<parameter=city>\n東京\n</parameter>\n<parameter=days>\n3\n</parameter>\n" \
        "<parameter=detail>\nTrue\n</parameter>\n<parameter=opts>\n{\"unit\": \"C\"}\n</parameter>\n</function>\n</tool_call>"
CALL2 = "<tool_call>\n<function=run>\n<parameter=cmd>\nls -la\n  echo \"a<b\"\n</parameter>\n</function>\n</tool_call>"

CASES = [
    ("plain", "こんにちは。\n\n今日はいい天気です。", "こんにちは。\n\n今日はいい天気です。",
     []),
    ("call only", CALL1, "", [("get_weather", {"city": "東京", "days": 3, "detail": True, "opts": {"unit": "C"}})]),
    ("text + 2 calls", "調べます。\n\n" + CALL1 + "\n" + CALL2, "調べます。",
     [("get_weather", {"city": "東京", "days": 3, "detail": True, "opts": {"unit": "C"}}),
      ("run", {"cmd": "ls -la\n  echo \"a<b\""})]),
    ("text with <tool", "a <tool and <tool_ca b", "a <tool and <tool_ca b", []),
    ("unterminated", "x\n" + CALL2[:-len("\n</parameter>\n</function>\n</tool_call>") - 3], "x",
     [("run", {"cmd": "ls -la\n  echo \"a"})]),
]


def run(text, pieces, prefix=""):
    p = ToolStreamParser(TOOLS, prefix=prefix)
    ev = []
    for s in pieces:
        ev += p.feed(s)
    ev += p.finish()
    content = "".join(x for k, x in ev if k == "content")
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for k, c in ev if k == "tool_call"]
    return content, calls


ok = True
for name, text, want_c, want_calls in CASES:
    splits = [[text]] + [[text[:i], text[i:]] for i in range(1, len(text))] + [list(text)]
    bad = 0
    for pieces in splits:
        got = run(text, pieces)
        if got != (want_c, want_calls):
            bad += 1
            if bad == 1:
                print(f"  {name}: split {[len(x) for x in pieces][:5]} -> {got}")
    print(f"{name}: {len(splits) - bad}/{len(splits)} splits ok")
    ok &= bad == 0

# forced prefix: the reply starts inside the call
pre = forced_prefix({"type": "function", "function": {"name": "run"}})
got = run(CALL2[len(pre):], [CALL2[len(pre):]], prefix=pre)
print("forced prefix:", got == ("", [("run", {"cmd": "ls -la\n  echo \"a<b\""})]))
ok &= got == ("", [("run", {"cmd": "ls -la\n  echo \"a<b\""})])

# round trip through the template: arguments come back from clients as JSON strings
from transformers import AutoTokenizer
import tsugumi.paths as tsugumi_paths
tok = AutoTokenizer.from_pretrained(tsugumi_paths.MODEL)
content, calls = run(CASES[2][1], [CASES[2][1]])
p = ToolStreamParser(TOOLS)
ev = p.feed(CASES[2][1]) + p.finish()
oa_calls = [c for k, c in ev if k == "tool_call"]
msgs = prepare_messages([{"role": "user", "content": "q"},
                         {"role": "assistant", "content": content, "tool_calls": oa_calls},
                         {"role": "tool", "tool_call_id": oa_calls[0]["id"], "content": "晴れ"}])
r = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, enable_thinking=False)
seg = r[r.index("<|im_start|>assistant\n"):]
rendered = seg[seg.index("</think>\n\n") + len("</think>\n\n"):seg.index("<|im_end|>")]
same = rendered == "調べます。\n\n" + CALL1.replace("True", "true") + "\n" + CALL2
print("template round trip:", same)
if not same:
    print(repr(rendered))
ok &= same
print("ALL OK" if ok else "FAILED")
