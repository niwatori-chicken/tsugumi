"""Inkling 2 chat format: prompt rendering (the model's chat template) and parsing of the reply token stream.

A reply is a sequence of messages, each `<|message_model|>[name]<|content_X|>...<|end_message|>`, then
`<|content_model_end_sampling|>` (eos). Content kinds: `<|content_thinking|>` (reasoning), `<|content_text|>` (the
answer), `<|content_invoke_tool_json|>` ({"name": ..., "args": {...}}, a tool call).
"""
import json
import uuid

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def prepare(messages):
    """OpenAI messages -> template messages: developer -> system, tool-call arguments (JSON strings) -> dicts,
    None content -> ""."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                c = dict(c)
                f = dict(c.get("function") or {})
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        a = json.loads(a) if a.strip() else {}
                    except json.JSONDecodeError:
                        a = {}
                f["arguments"] = a if isinstance(a, dict) else {}
                c["function"] = f
                calls.append(c)
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def effort_value(effort):
    """reasoning_effort (name or number in [0, 0.99]) for the template, or None for its default (0.9)."""
    if effort is None:
        return None
    if isinstance(effort, str):
        e = effort.strip().lower()
        if e not in EFFORTS:
            raise ValueError(f"reasoning_effort: one of {', '.join(EFFORTS)} or a number in [0, 0.99]")
        return e
    v = float(effort)
    if not 0.0 <= v <= 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    return v


def encode(tok, messages, effort=None, tools=None):
    kw = {} if effort is None else {"reasoning_effort": effort}
    ids = tok.apply_chat_template(prepare(messages), tools=tools, tokenize=True, add_generation_prompt=True, **kw)
    return list(ids["input_ids"] if not isinstance(ids, list) else ids)


class ReplyParser:
    """Token ids -> events: ("reasoning_content", text), ("content", text), ("tool_call", dict). Text is decoded
    incrementally per message (incomplete UTF-8 held back)."""

    def __init__(self, tok):
        self.tok = tok
        ids = lambda n: tok.convert_tokens_to_ids(n)
        self.MODEL = ids("<|message_model|>")
        self.END = ids("<|end_message|>")
        self.EOS = ids("<|content_model_end_sampling|>")
        self.kinds = {ids("<|content_thinking|>"): "reasoning_content", ids("<|content_text|>"): "content",
                      ids("<|content_invoke_tool_json|>"): "tool", ids("<|content_invoke_tool_text|>"): "tool_text",
                      ids("<|content_xml|>"): "content"}
        self.special = set(tok.all_special_ids) | {self.MODEL, self.END, self.EOS} | set(self.kinds)
        self.mode = None  # None (between messages), "name", or a content kind
        self.seg = []  # tokens of the current message's content
        self.off = (0, 0)  # (prefix offset, read offset) of the incremental decode
        self.name = []

    def _text(self, final=False):
        """New text of the current message: only the last few tokens are re-decoded (O(n) for long replies)."""
        po, ro = self.off
        prefix = self.tok.decode(self.seg[po:ro])
        text = self.tok.decode(self.seg[po:])
        if len(text) <= len(prefix) or (not final and text.endswith("�")):
            return ""
        self.off = (ro, len(self.seg))
        return text[len(prefix):]

    def _close(self):
        ev = []
        if self.mode in ("reasoning_content", "content"):
            d = self._text(final=True)
            if d:
                ev.append((self.mode, d))
        elif self.mode in ("tool", "tool_text"):
            ev += self._tool(self.tok.decode(self.seg))
        self.mode, self.seg, self.off = None, [], (0, 0)
        return ev

    def _tool(self, text):
        name = self.tok.decode(self.name).strip()
        args = {}
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                name = obj.get("name") or name
                args = obj.get("args", obj.get("arguments", {}))
        except json.JSONDecodeError:
            args = {"_raw": text}
        return [("tool_call", dict(id="call_" + uuid.uuid4().hex[:24], type="function",
                                   function=dict(name=name, arguments=json.dumps(args, ensure_ascii=False))))]

    def feed(self, t):
        """One token -> list of events."""
        if t == self.EOS:
            return self._close()
        if t == self.MODEL:
            ev = self._close()
            self.mode, self.name = "name", []
            return ev
        if t == self.END:
            return self._close()
        if t in self.kinds:
            ev = self._close() if self.mode not in (None, "name") else []
            self.mode, self.seg, self.off = self.kinds[t], [], (0, 0)
            return ev
        if self.mode == "name":
            self.name.append(t)
            return []
        if t in self.special:
            return []
        if self.mode is None:  # text outside a message: treat as answer text
            self.mode, self.seg, self.off = "content", [], (0, 0)
        self.seg.append(t)
        if self.mode in ("reasoning_content", "content"):
            d = self._text()
            return [(self.mode, d)] if d else []
        return []

    def finish(self):
        return self._close()
