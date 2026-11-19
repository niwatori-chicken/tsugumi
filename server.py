"""OpenAI-compatible chat server for Qwen3.8-27B on one P100A (lean B=1 engine + MTP speculative decoding).

    python server.py [--port 8010] [--max-seq 159744]

Endpoints: POST /v1/chat/completions (stream or not), GET /v1/models, GET /health.
Decoding is greedy (temperature/top_p are accepted but ignored). One request at a time.
`chat_template_kwargs.enable_thinking` controls thinking (default: false); <think> content goes to
`reasoning_content`.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
import qe.paths as qe_paths


MODEL_DIR = qe_paths.MODEL
CACHE = qe_paths.CACHE
MODEL_NAME = "qwen3.8-27b-tt"


class Backend:
    def __init__(self, max_seq, T=4, prefill_T=32):
        import ttnn
        from transformers import AutoTokenizer

        from qe.generator import Generator
        from qe.model import Precision, Qwen35Engine

        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
        self.eng = Qwen35Engine(self.dev, MODEL_DIR, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=max_seq,
                                log=lambda *a: None)
        self.eng.init_spec(T, mtp=True)
        self.eng.init_prefill(prefill_T)
        if os.environ.get("QE_LARGE_PREFILL", "1") == "1":  # must come before capture_all
            self.eng.init_large_prefill(int(os.environ.get("QE_LARGE_T", "1024")))
        self.eng.capture_all()
        self.gen = Generator(self.eng)
        self.max_seq = max_seq
        self.lock = threading.Lock()

    def encode(self, messages, enable_thinking):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return ids

    def stream(self, ids, max_new):
        """Yields (text_delta, done_info). Text is decoded incrementally; incomplete UTF-8 is held back."""
        out = []
        sent = ""
        stats = {}
        t0 = time.perf_counter()
        t_first = None
        for t in self.gen.generate(ids, max_new=max_new, stats=stats):
            if t_first is None:
                t_first = time.perf_counter()
            if t in self.gen.eos:
                continue  # let the generator finish so it records the committed state
            out.append(t)
            text = self.tok.decode(out)
            if text.endswith("�"):
                continue
            if len(text) > len(sent):
                yield text[len(sent):], None
                sent = text
        text = self.tok.decode(out)
        if len(text) > len(sent):
            yield text[len(sent):], None
        dt = time.perf_counter() - (t_first or t0)
        steps = max(stats.get("steps", 0), 1)
        yield "", dict(prompt_tokens=len(ids), completion_tokens=len(out), reused=stats.get("reused", 0),
                       ttft=(t_first or t0) - t0, decode_tps=len(out) / dt if dt > 0 else 0.0,
                       tokens_per_step=(stats.get("accepted", 0) + steps) / steps)


class ThinkSplitter:
    """Routes text inside <think>...</think> to reasoning_content."""

    def __init__(self, in_think):
        self.in_think = in_think
        self.buf = ""

    def feed(self, s):
        self.buf += s
        res = []
        while self.buf:
            tag = "</think>" if self.in_think else "<think>"
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    res.append(("reasoning_content" if self.in_think else "content", self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.in_think = not self.in_think
                continue
            keep = 0  # hold back a possible partial tag
            for k in range(1, len(tag)):
                if self.buf.endswith(tag[:k]):
                    keep = k
            emit = self.buf[: len(self.buf) - keep]
            if emit:
                res.append(("reasoning_content" if self.in_think else "content", emit))
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return res


def make_app(backend):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        messages = body.get("messages", [])
        for m in messages:
            if m.get("role") == "developer":
                m["role"] = "system"
        kw = body.get("chat_template_kwargs") or {}
        thinking = bool(kw.get("enable_thinking", False))
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        ids = backend.encode(messages, thinking)
        max_new = max(1, min(max_new, backend.max_seq - len(ids) - 8))
        if len(ids) + 8 >= backend.max_seq:
            return JSONResponse({"error": {"message": f"prompt too long ({len(ids)} tokens)"}}, status_code=400)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        # the template opens <think> itself when thinking is enabled
        in_think = thinking

        def chunk(delta, finish=None, usage=None):
            d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                d["usage"] = usage
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        if body.get("stream"):
            def gen():
                with backend.lock:
                    sp = ThinkSplitter(in_think)
                    yield chunk({"role": "assistant"})
                    info = None
                    for delta, done in backend.stream(ids, max_new):
                        if done:
                            info = done
                            break
                        for field, txt in sp.feed(delta):
                            yield chunk({field: txt})
                    usage = dict(prompt_tokens=info["prompt_tokens"], completion_tokens=info["completion_tokens"],
                                 total_tokens=info["prompt_tokens"] + info["completion_tokens"])
                    yield chunk({}, finish="stop", usage=usage)
                    yield "data: [DONE]\n\n"
                    print(f"[req] {info}", flush=True)

            return StreamingResponse(gen(), media_type="text/event-stream")

        with backend.lock:
            sp = ThinkSplitter(in_think)
            parts = {"content": "", "reasoning_content": ""}
            info = None
            for delta, done in backend.stream(ids, max_new):
                if done:
                    info = done
                    break
                for field, txt in sp.feed(delta):
                    parts[field] += txt
        print(f"[req] {info}", flush=True)
        msg = {"role": "assistant", "content": parts["content"]}
        if parts["reasoning_content"]:
            msg["reasoning_content"] = parts["reasoning_content"]
        return {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": dict(prompt_tokens=info["prompt_tokens"], completion_tokens=info["completion_tokens"],
                              total_tokens=info["prompt_tokens"] + info["completion_tokens"]),
                "timings": info}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--max-seq", type=int, default=159744)  # large-T prefill needs ~140MB of working DRAM
    a = ap.parse_args()
    import uvicorn

    t0 = time.time()
    backend = Backend(a.max_seq)
    print(f"[server] ready in {time.time() - t0:.0f}s on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(make_app(backend), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
