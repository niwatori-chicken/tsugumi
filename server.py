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
from tsumi.log import get_logger
from tsumi.sampling import Sampler
from tsumi.toolcall import ToolStreamParser, forced_prefix, prepare_messages


MODEL_DIR = tsumi_paths.MODEL
CACHE = tsumi_paths.CACHE
MODEL_NAME = "qwen3.8-27b-tt"
log = get_logger()
EXIT_HANG = 75
# must be set before tt-metal reads its runtime options (first ttnn import)
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")


class Backend:
    def __init__(self, max_seq, T=4, prefill_T=32):
        import ttnn
        from transformers import AutoTokenizer

        from tsumi.generator import Generator
        from tsumi.model import Precision, Qwen35Engine

        t_all = time.perf_counter()

        class stage:  # "name ..." then "name done in Xs"
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                self.t = time.perf_counter()
                log.info(f"{self.name} ...")

            def __exit__(self, *exc):
                if exc[0] is None:
                    log.info(f"{self.name}: done in {time.perf_counter() - self.t:.1f}s")

        def eng_log(msg):  # engine messages: one line per 8 layers at INFO, every layer at DEBUG
            if msg.startswith("loaded layer"):
                i = int(msg.split()[2].split("/")[0])
                (log.info if i % 8 == 0 else log.debug)(msg)
            else:
                log.info(msg)

        with stage("tokenizer"):
            self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        with stage("open device"):
            self.dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
        with stage(f"load weights ({CACHE}/bfp4; 2-3 min when not in the page cache)"):
            self.eng = Qwen35Engine(self.dev, MODEL_DIR, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=max_seq,
                                    log=eng_log)
        tv = int(env("VERIFY_T", "8"))  # longer verify mode for prompt-lookup drafts (0 = off)
        with stage(f"T-row modes (verify T={T}{f', {tv}' if tv else ''}, prefill T={prefill_T}) and MTP head"):
            self.eng.init_spec(T, mtp=True, verify_T=tv or None)
            self.eng.init_prefill(prefill_T)
            if tv:
                self.eng.init_verify(tv)
        if env("LARGE_PREFILL", "1") == "1":  # must come before capture_all
            with stage("large-T prefill and warm-up (compiles every chunk shape)"):
                self.eng.init_large_prefill(int(env("LARGE_T", "1024")))
        with stage("trace capture"):
            self.eng.capture_all()
        self.gen = Generator(self.eng)
        v = ttnn.get_memory_view(self.dev, ttnn.BufferType.DRAM)
        G = 1 << 30
        log.info(f"DRAM: used {v.total_bytes_allocated_per_bank * v.num_banks / G:.2f} GiB, "
                 f"free {v.total_bytes_free_per_bank * v.num_banks / G:.2f} GiB; "
                 f"startup {time.perf_counter() - t_all:.0f}s")
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
            log.error(f"device error, exiting with {EXIT_HANG} after this response: {self.dead}")
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
        """Yields (text_delta, done_info). Text is decoded incrementally (only the last few tokens are re-decoded,
        so long replies stay O(n)); incomplete UTF-8 is held back."""
        out = []
        stats = {}
        t0 = time.perf_counter()
        t_first = None
        eos = False
        prefix_off = read_off = 0  # decode out[prefix_off:] and emit what goes beyond out[prefix_off:read_off]
        self._arm()
        try:
            for t in self._tokens(ids, max_new, stats, sampler):
                if t_first is None:
                    t_first = time.perf_counter()
                if t in self.gen.eos:
                    eos = True
                    continue  # let the generator finish so it records the committed state
                out.append(t)
                prefix = self.tok.decode(out[prefix_off:read_off])
                text = self.tok.decode(out[prefix_off:])
                if text.endswith("\ufffd") or len(text) <= len(prefix):
                    continue
                yield text[len(prefix):], None
                prefix_off, read_off = read_off, len(out)
        finally:
            faulthandler.cancel_dump_traceback_later()
        prefix = self.tok.decode(out[prefix_off:read_off])
        text = self.tok.decode(out[prefix_off:])
        if len(text) > len(prefix):
            yield text[len(prefix):], None
        dt = time.perf_counter() - (t_first or t0)
        steps = max(stats.get("steps", 0), 1)
        yield "", dict(prompt_tokens=len(ids), completion_tokens=len(out), reused=stats.get("reused", 0),
                       reuse=stats.get("reuse"), switched=stats.get("switched", False),
                       ttft=(t_first or t0) - t0, decode_tps=len(out) / dt if dt > 0 else 0.0,
                       tokens_per_step=(stats.get("accepted", 0) + steps) / steps,
                       ngram_steps=stats.get("ngram_steps", 0), steps=stats.get("steps", 0),
                       finish="stop" if eos else "length", total=time.perf_counter() - t0)


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


def log_done(tag, info, ncalls=0):
    """One line per finished request."""
    reuse = f"reused {info['reused']} ({info['reuse']}{', switched' if info.get('switched') else ''})" \
        if info.get("reused") else "no reuse"
    ng = f", n-gram {info['ngram_steps']}/{info['steps']} steps" if info.get("ngram_steps") else ""
    fin = "tool_calls" if ncalls else info["finish"]
    log.info(f"req {tag}: done, {fin}; prompt {info['prompt_tokens']} ({reuse}), ttft {info['ttft']:.2f}s, "
             f"{info['completion_tokens']} tok {info['decode_tps']:.1f} tok/s {info['tokens_per_step']:.2f} tok/step"
             f"{ng}" + (f", {ncalls} tool call{'s' if ncalls > 1 else ''}" if ncalls else "") +
             f", {info['total']:.1f}s")


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
        tag = rid[-6:]
        params = [f"prompt {len(ids)} tok", f"max {max_new}"] + (["thinking"] if thinking else []) + \
            ([f"temp {body.get('temperature')}"] if (body.get("temperature") or 0) > 0 else []) + \
            ([f"tools {len(tools)}"] if tools else []) + (["stream"] if body.get("stream") else [])
        ignored = [k for k in ("stop", "presence_penalty", "frequency_penalty", "logprobs", "response_format", "n",
                               "logit_bias") if body.get(k) not in (None, False, 0, 1, [], "") or
                   (k == "n" and (body.get("n") or 1) > 1)]
        log.info(f"req {tag}: {', '.join(params)}" + (" (waiting: busy)" if backend.lock.locked() else "") +
                 (f"; ignored parameters: {', '.join(ignored)}" if ignored else ""))
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
                    yield chunk({}, finish="tool_calls" if ncalls else info["finish"], usage=usage(info), timings=info)
                    yield "data: [DONE]\n\n"
                    log_done(tag, info, ncalls)

            async def agen():
                """Runs the blocking generator in worker threads, one chunk at a time, so the event loop stays free.
                When the client disconnects, Starlette cancels this task: the finally closes the generator, which
                ends the generation where it is and releases the device lock (left suspended, it would hold the
                lock until garbage collection)."""
                it = gen()
                done = False
                try:
                    while True:
                        c = await run_in_threadpool(next, it, None)  # waits for the running step if cancelled
                        if c is None:
                            done = True
                            break
                        yield c
                finally:
                    it.close()
                    if not done:
                        log.info(f"req {tag}: client disconnected, generation stopped")

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
        log_done(tag, info, len(calls))
        msg = {"role": "assistant", "content": parts["content"] if parts["content"] or not calls else None}
        if parts["reasoning_content"]:
            msg["reasoning_content"] = parts["reasoning_content"]
        if calls:
            msg["tool_calls"] = calls
        return {"id": rid, "object": "chat.completion", "created": created, "model": MODEL_NAME,
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else info["finish"]}],
                "usage": usage(info), "timings": info}
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--max-seq", type=int, default=262144)  # the model limit; bfp8 KV (9 GiB) leaves ~1.8 GiB free
    a = ap.parse_args()
    import uvicorn

    log_settings(a)
    t0 = time.time()
    backend = Backend(a.max_seq)
    log.info(f"ready in {time.time() - t0:.0f}s: http://{a.host}:{a.port}/ (web UI), /v1/chat/completions")
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(f"listening on {a.host} without authentication: anyone on the network can use the server")
    uvicorn.run(make_app(backend), host=a.host, port=a.port, log_level="warning")


def log_settings(a):
    """The settings in effect (TSUMI_* / QE_* environment, flags), one line each."""
    g = lambda k, d: env(k, d)
    ngram = "off" if g("NGRAM", "3") == "0" else f"when >= {g('NGRAM_MIN', '6')} tokens repeat"
    items = [
        ("model", MODEL_DIR), ("weights cache", CACHE), ("listen", f"{a.host}:{a.port}"),
        ("max_seq", f"{a.max_seq} tokens"),
        ("KV cache", g("KV_DTYPE", "bfp8")),
        ("prefill", f"large T {g('LARGE_T', '1024')} (on={g('LARGE_PREFILL', '1')}), SDPA {g('PREFILL_SDPA', 'hifi2')}"),
        ("drafts", f"MTP (vocab {'63,168' if g('DRAFT_VOCAB', '1') == '1' else 'full'}), n-gram {ngram}, "
                   f"verify T 4 / {g('VERIFY_T', '8')}"),
        ("conversations", f"multi={g('MULTI_CONV', '1')} max={g('MAX_CONVS', '8')} snapshot={g('SNAPSHOT', '1')}"),
        ("hang detection", f"device timeout {os.environ.get('TT_METAL_OPERATION_TIMEOUT_SECONDS')}s, "
                           f"backstop {g('HANG_BACKSTOP', '300')}s"),
    ]
    log.info("settings:")
    for k, v in items:
        log.info(f"  {k:15s} {v}")


if __name__ == "__main__":
    main()
