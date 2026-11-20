"""OpenAI-compatible chat server for Qwen3.8-27B on one P100A (lean B=1 engine + MTP speculative decoding).

    python server.py [--port 8010] [--max-seq 262144]

Endpoints: POST /v1/chat/completions (stream or not), GET /v1/models, GET /health, GET / (a minimal chat web UI).
Decoding is greedy unless the request sets temperature > 0; then top_p, top_k (<= 64), min_p and seed apply
(exact speculative sampling, tsumi/sampling.py). One request at a time.
`chat_template_kwargs.enable_thinking` controls thinking (default: false); <think> content goes to
`reasoning_content`. Other chat_template_kwargs (reasoning_effort, preserve_thinking) go to the template.
Tool calling: `tools`, `tool_choice` (auto / none; required / a named function with thinking off), assistant
`tool_calls` and `role: tool` messages in the history; calls are returned as `tool_calls` (tsumi/toolcall.py).

Hang handling: a device wait that makes no progress for TT_METAL_OPERATION_TIMEOUT_SECONDS (default here 30)
raises; the request gets an error, /health turns 503 and the process exits with code 75 (EXIT_HANG). A
faulthandler backstop (TSUMI_HANG_BACKSTOP seconds without a new token, default 300) kills the process if a
wait hangs without raising. run_server.sh resets the device and restarts the server after either.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
import faulthandler
import tsumi.paths as tsumi_paths
from tsumi.env import env
from tsumi.sampling import Sampler
from tsumi.toolcall import ToolStreamParser, forced_prefix, prepare_messages


MODEL_DIR = tsumi_paths.MODEL
CACHE = tsumi_paths.CACHE
MODEL_NAME = "qwen3.8-27b-tt"
EXIT_HANG = 75
# must be set before tt-metal reads its runtime options (first ttnn import)
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")


class Backend:
    def __init__(self, max_seq, T=4, prefill_T=32):
        import ttnn
        from transformers import AutoTokenizer

        from tsumi.generator import Generator
        from tsumi.model import Precision, Qwen35Engine

        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
        self.eng = Qwen35Engine(self.dev, MODEL_DIR, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=max_seq,
                                log=lambda *a: None)
        self.eng.init_spec(T, mtp=True)
        self.eng.init_prefill(prefill_T)
        if env("LARGE_PREFILL", "1") == "1":  # must come before capture_all
            self.eng.init_large_prefill(int(env("LARGE_T", "1024")))
        self.eng.capture_all()
        self.gen = Generator(self.eng)
        self.max_seq = max_seq
        self.lock = threading.Lock()
        self.dead = None  # reason string once the device is considered hung
        self.backstop = int(env("HANG_BACKSTOP", "300"))

    def _arm(self):
        faulthandler.dump_traceback_later(self.backstop, exit=True)

    def fail(self, e):
        """Device error during a request: stop serving and exit so run_server.sh can reset the device."""
        if self.dead is None:
            lines = str(e).splitlines() or [""]
            msg = next((l for l in lines if "TIMEOUT" in l), lines[0])  # the TT_THROW header line says little
            self.dead = f"{type(e).__name__}: {msg.strip()[:300]}"
            print(f"[server] device error, exiting with {EXIT_HANG} after this response: {self.dead}", flush=True)
            threading.Timer(1.0, os._exit, [EXIT_HANG]).start()  # let the error response go out first

    def debug_hang(self):
        from tsumi.hang_kernel import run_hang_kernel
        run_hang_kernel(self.dev)

    def _tokens(self, ids, max_new, stats, sampler=None):
        """Generator tokens; re-arms the backstop per token. Device errors go to fail() and are re-raised."""
        try:
            for t in self.gen.generate(ids, max_new=max_new, stats=stats, sampler=sampler):
                self._arm()
                yield t
        except Exception as e:
            self.fail(e)
            raise

    def encode(self, messages, enable_thinking, tools=None, template_kwargs=None):
        ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                           enable_thinking=enable_thinking, **(template_kwargs or {}))
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return ids

    def stream(self, ids, max_new, sampler=None):
        """Yields (text_delta, done_info). Text is decoded incrementally; incomplete UTF-8 is held back."""
        out = []
        sent = ""
        stats = {}
        t0 = time.perf_counter()
        t_first = None
        self._arm()
        try:
            for t in self._tokens(ids, max_new, stats, sampler):
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
        finally:
            faulthandler.cancel_dump_traceback_later()
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
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI()

    @app.get("/")
    def webui():  # minimal chat page (webui/index.html)
        return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "index.html"))

    @app.get("/health")
    def health():
        if backend.dead:
            return JSONResponse({"status": "error", "error": backend.dead}, status_code=503)
        return {"status": "ok"}

    def unavailable():
        return JSONResponse({"error": {"message": f"device error, server restarting: {backend.dead}",
                                       "type": "server_error"}}, status_code=503)

    if env("DEBUG_HANG", "0") == "1":  # test-only (scripts/test_server_hang.sh)
        @app.post("/debug/hang")
        def debug_hang():
            with backend.lock:
                backend._arm()
                try:
                    backend.debug_hang()
                except Exception as e:
                    backend.fail(e)
                    return unavailable()
                finally:
                    faulthandler.cancel_dump_traceback_later()
            return {"status": "no hang"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if backend.dead:
            return unavailable()
        body = await req.json()
        kw = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kw.pop("enable_thinking", False))
        tool_choice = body.get("tool_choice") or "auto"
        tools = body.get("tools") or None
        if tool_choice == "none":
            tools = None
        # tool_choice "required" / a named function: start the reply inside a call (thinking off only; with
        # thinking on the reply starts in <think>, so it is treated as "auto")
        forced = forced_prefix(tool_choice) if tools and not thinking else ""
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
        try:
            ids = backend.encode(prepare_messages(body.get("messages", [])), thinking, tools, kw)
        except Exception as e:  # template errors (bad roles, no user message, ...)
            return JSONResponse({"error": {"message": f"chat template: {e}", "type": "invalid_request_error"}},
                                status_code=400)
        if forced:
            ids = ids + backend.tok.encode(forced, add_special_tokens=False)
        max_new = max(1, min(max_new, backend.max_seq - len(ids) - 8))
        if len(ids) + 8 >= backend.max_seq:
            return JSONResponse({"error": {"message": f"prompt too long ({len(ids)} tokens)"}}, status_code=400)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        temperature = float(body.get("temperature") or 0.0)
        try:
            if temperature > 0:
                Sampler(temperature, body.get("top_p"), body.get("top_k"), body.get("min_p"), body.get("seed"))
        except (TypeError, ValueError) as e:
            return JSONResponse({"error": {"message": f"sampling parameters: {e}", "type": "invalid_request_error"}},
                                status_code=400)

        def make_sampler():  # per request; None = greedy
            if temperature <= 0:
                return None
            return Sampler(temperature, body.get("top_p"), body.get("top_k"), body.get("min_p"), body.get("seed"))

        def events():
            """(field, value) pairs: reasoning_content / content text, tool_call dicts, then ("done", info)."""
            sp = ThinkSplitter(thinking)  # the template opens <think> itself when thinking is enabled
            tp = ToolStreamParser(tools, prefix=forced) if tools else None
            for delta, done in backend.stream(ids, max_new, make_sampler()):
                if done:
                    if tp:
                        yield from tp.finish()
                    yield "done", done
                    return
                for field, txt in sp.feed(delta):
                    if tp and field == "content":
                        yield from tp.feed(txt)
                    else:
                        yield field, txt

        def usage(info):
            return dict(prompt_tokens=info["prompt_tokens"], completion_tokens=info["completion_tokens"],
                        total_tokens=info["prompt_tokens"] + info["completion_tokens"])

        def chunk(delta, finish=None, usage=None, timings=None):
            d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": MODEL_NAME,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                d["usage"] = usage
            if timings:
                d["timings"] = timings  # non-standard (prompt reuse, TTFT, decode speed); the web UI shows it
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        def error_event():
            return "data: " + json.dumps({"error": {"message": f"device error: {backend.dead}"}}) + "\n\n"

        if body.get("stream"):
            def gen():
                with backend.lock:
                    if backend.dead:
                        yield error_event()
                        return
                    yield chunk({"role": "assistant"})
                    info, ncalls = None, 0
                    try:
                        for field, v in events():
                            if field == "done":
                                info = v
                            elif field == "tool_call":
                                yield chunk({"tool_calls": [dict(index=ncalls, **v)]})
                                ncalls += 1
                            else:
                                yield chunk({field: v})
                    except Exception:
                        yield error_event()
                        return
                    yield chunk({}, finish="tool_calls" if ncalls else "stop", usage=usage(info), timings=info)
                    yield "data: [DONE]\n\n"
                    print(f"[req] {info} tool_calls={ncalls}", flush=True)

            async def agen():
                """Runs the blocking generator in worker threads, one chunk at a time, so the event loop stays free.
                When the client disconnects, Starlette cancels this task: the finally closes the generator, which
                ends the generation where it is and releases the device lock (left suspended, it would hold the
                lock until garbage collection)."""
                it = gen()
                try:
                    while True:
                        c = await run_in_threadpool(next, it, None)  # waits for the running step if cancelled
                        if c is None:
                            break
                        yield c
                finally:
                    it.close()

            return StreamingResponse(agen(), media_type="text/event-stream")

        def run():  # blocking: in a worker thread, not on the event loop
            with backend.lock:
                if backend.dead:
                    return None
                parts = {"content": "", "reasoning_content": ""}
                calls, info = [], None
                try:
                    for field, v in events():
                        if field == "done":
                            info = v
                        elif field == "tool_call":
                            calls.append(v)
                        else:
                            parts[field] += v
                except Exception:
                    return None
            return parts, calls, info

        res = await run_in_threadpool(run)
        if res is None:
            return unavailable()
        parts, calls, info = res
        print(f"[req] {info} tool_calls={len(calls)}", flush=True)
        msg = {"role": "assistant", "content": parts["content"] if parts["content"] or not calls else None}
        if parts["reasoning_content"]:
            msg["reasoning_content"] = parts["reasoning_content"]
        if calls:
            msg["tool_calls"] = calls
        return {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                "usage": usage(info), "timings": info}
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--max-seq", type=int, default=262144)  # the model limit; bfp8 KV (9 GiB) leaves ~1.8 GiB free
    a = ap.parse_args()
    import uvicorn

    t0 = time.time()
    backend = Backend(a.max_seq)
    print(f"[server] ready in {time.time() - t0:.0f}s on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(make_app(backend), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
