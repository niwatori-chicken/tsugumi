"""OpenAI-style tool calling on top of Qwen's chat template.

The model writes calls in the template's XML format (one or more blocks, after optional text):

    <tool_call>
    <function=NAME>
    <parameter=ARG>
    value (may span lines)
    </parameter>
    </function>
    </tool_call>

Parameter values are text; they are converted with the tool's JSON schema (strings stay as written, other types
are parsed as JSON when possible). `<tool_call>` is a single token, so it arrives in one piece when streaming.
"""
import json
import re
import uuid

OPEN, CLOSE = "<tool_call>", "</tool_call>"
_FUNC = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)(?:\n?</parameter>|$)", re.S)


def prepare_messages(messages):
    """Client messages -> template messages: `developer` -> `system`, tool-call arguments given as JSON strings
    (OpenAI format) -> dicts (the template iterates over them)."""
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


def _param_types(tools):
    """{function name: {parameter: JSON schema type}}"""
    types = {}
    for t in tools or []:
        f = t.get("function", t)
        props = ((f.get("parameters") or {}).get("properties") or {})
        types[f.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return types


def _convert(value, typ):
    if typ == "string" or (isinstance(typ, list) and "string" in typ and len(typ) == 1):
        return value
    v = value.strip()
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        pass
    low = v.lower()
    if low in ("true", "false"):  # Python-style booleans / None
        return low == "true"
    if low in ("none", "null"):
        return None
    return value


def parse_call(block, types):
    """Body of one <tool_call> block -> OpenAI tool call dict, or None if no function is named."""
    m = _FUNC.search(block)
    if not m:
        return None
    name = m.group(1).strip()
    pt = types.get(name, {})
    args = {}
    for p in _PARAM.finditer(m.group(2)):
        k = p.group(1).strip()
        args[k] = _convert(p.group(2), pt.get(k))
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class ToolStreamParser:
    """Splits streamed assistant content into text and tool calls.

    feed(text) / finish() return a list of events: ("content", str) or ("tool_call", call dict). Text that could
    be the start of `<tool_call>` is held back; text after the first call is dropped (the template allows
    reasoning only before the calls)."""

    def __init__(self, tools, prefix=""):
        self.types = _param_types(tools)
        self.buf = ""
        self.in_call = False
        self.calls = []
        self.pending_ws = ""  # whitespace before a possible tool call (the template trims it)
        if prefix:
            self.feed(prefix)

    def _content(self, s, ev):
        if not s:
            return
        if self.calls:
            return  # nothing but calls after the first call
        s = self.pending_ws + s
        self.pending_ws = ""
        body = s.rstrip()
        self.pending_ws = s[len(body):]
        if body:
            ev.append(("content", body))

    def feed(self, text):
        self.buf += text
        ev = []
        while True:
            if not self.in_call:
                i = self.buf.find(OPEN)
                if i < 0:
                    keep = 0  # longest suffix that is a prefix of OPEN
                    for n in range(min(len(OPEN) - 1, len(self.buf)), 0, -1):
                        if OPEN.startswith(self.buf[-n:]):
                            keep = n
                            break
                    self._content(self.buf[:len(self.buf) - keep], ev)
                    self.buf = self.buf[len(self.buf) - keep:]
                    return ev
                self._content(self.buf[:i], ev)
                self.buf = self.buf[i + len(OPEN):]
                self.in_call = True
            j = self.buf.find(CLOSE)
            if j < 0:
                return ev
            self._emit(self.buf[:j], ev)
            self.buf = self.buf[j + len(CLOSE):]
            self.in_call = False

    def _emit(self, block, ev):
        c = parse_call(block, self.types)
        if c:
            self.calls.append(c)
            self.pending_ws = ""
            ev.append(("tool_call", c))

    def finish(self):
        """End of generation: an unterminated call (e.g. max_tokens) is parsed as far as it got."""
        ev = []
        if self.in_call:
            self._emit(self.buf, ev)
        else:
            self._content(self.buf, ev)
        self.buf = ""
        self.in_call = False
        return ev


def forced_prefix(tool_choice):
    """Text to put at the start of the reply for tool_choice "required" / {"function": {"name": ...}}."""
    if tool_choice == "required":
        return OPEN + "\n"
    if isinstance(tool_choice, dict) and (tool_choice.get("function") or {}).get("name"):
        return f"{OPEN}\n<function={tool_choice['function']['name']}>\n"
    return ""
