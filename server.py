"""OpenAI-compatible chat server for Qwen3.8-27B on one P100A (lean B=1 engine + MTP speculative decoding).

    python server.py [--port 8010] [--max-seq 262144]

Endpoints: POST /v1/chat/completions and /v1/completions (stream or not), POST /tokenize and /detokenize,
GET /v1/models, GET /health, GET / (a minimal chat web UI).
Decoding is greedy unless the request sets temperature > 0; then top_p, top_k (<= 64), min_p and seed apply
(exact speculative sampling, tsugumi/sampling.py). One request at a time.
`chat_template_kwargs.enable_thinking` controls thinking (default: false); <think> content goes to
`reasoning_content`. `thinking_budget` (or chat_template_kwargs.thinking_budget; default --thinking-budget) caps
the thinking tokens: at the cap a closing sentence and </think> are inserted (THINK_FORCE), and max_tokens then
counts the answer only. Other chat_template_kwargs (reasoning_effort, preserve_thinking) go to the template.
Tool calling: `tools`, `tool_choice` (auto / none; required / a named function with thinking off), assistant
`tool_calls` and `role: tool` messages in the history; calls are returned as `tool_calls` (tsugumi/toolcall.py).
Images: OpenAI `image_url` content parts (data: URL or http(s)); the ViT runs on the device (tsugumi/vision_tt.py,
TSUGUMI_VISION_DEVICE=0: on the CPU) under the device lock, and
each image becomes its own virtual token ids, so conversation reuse tells images apart.
`logprobs` / `top_logprobs` (<= 20; completions: logprobs = n): log-softmax over the device top-64 logits (a
little too high; tokens outside it, only forced ones, get -9999); chat logprobs cover the content, not the thinking.

Authentication: with --api-key / TSUGUMI_API_KEY (comma separated) or --api-key-file (one key per line), every
endpoint except GET / (the web UI page) and GET /health needs `Authorization: Bearer <key>` (OpenAI clients send it
from their api_key setting); others get 401. Without keys the server is open, as before (fine on 127.0.0.1).
/health without a valid key shows only status / busy / waiting; with one (or without auth) also the running
request and `board`: temperatures, power, fan and clock from `tt-smi -s` every TSUGUMI_BOARD_POLL seconds (default
10, 0 = off; tsugumi/board.py).

Shutdown (SIGINT / SIGTERM): stop accepting requests, end a running generation at its next token, close the device.

Hang handling: a device wait that makes no progress for TT_METAL_OPERATION_TIMEOUT_SECONDS (default here 30)
raises; the request gets an error, /health turns 503 and the process exits with code 75 (EXIT_HANG). A
faulthandler backstop (TSUGUMI_HANG_BACKSTOP seconds without a new token, default 300) kills the process if a
wait hangs without raising. run_server.sh resets the device and restarts the server after either.
"""
import argparse
import hmac
import json
import os
import signal
import sys
import threading
import time
import uuid
import faulthandler
import tsugumi.paths as tsugumi_paths
from tsugumi.board import BoardMonitor
from tsugumi.env import env
from tsugumi.log import get_logger
from tsugumi.sampling import Sampler
from tsugumi.toolcall import ToolStreamParser, forced_prefix, prepare_messages


MODEL_DIR = tsugumi_paths.MODEL
CACHE = tsugumi_paths.CACHE
MODEL_NAMES = [os.path.basename(os.path.normpath(MODEL_DIR))]  # set from --served-model-name in main()
log = get_logger()
EXIT_HANG = 75
# inserted when the thinking reaches its budget (the wording Qwen suggests for budget-capped thinking)
THINK_FORCE = ("\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
               "directly now.\n</think>\n\n")
THINK_BUDGET = [None]  # server default (--thinking-budget); None = no cap
INKLING_EFFORT = [None]  # Inkling: default reasoning_effort (--reasoning-effort); None = the template's (0.9)
# must be set before tt-metal reads its runtime options (first ttnn import)
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")


class Backend:
    def __init__(self, max_seq, T=4, prefill_T=32):
        import ttnn
        from transformers import AutoTokenizer

        from tsugumi.generator import Generator
        from tsugumi.model import Precision, Qwen35Engine

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
            # batched decode (TSUGUMI_BATCH lanes) traces 2 more graphs per lane count (1, 2, 4, ..)
            self.lanes = max(1, int(env("BATCH", "1")))
            trace_mb = int(env("TRACE_MB", "512" if self.lanes == 1 else "1024"))
            self.dev = ttnn.open_device(device_id=0, trace_region_size=trace_mb << 20, l1_small_size=24576)
        with stage(f"load weights ({CACHE}/bfp4; 2-3 min when not in the page cache)"):
            self.eng = Qwen35Engine(self.dev, MODEL_DIR, Precision("bfp4"), f"{CACHE}/bfp4", max_seq=max_seq,
                                    log=eng_log)
        self.vision_tt = None
        if env("IMAGES", "1") == "1" and env("VISION_DEVICE", "1") == "1":  # before the traces (persistent buffers)
            from tsugumi.vision_tt import VisionTT
            with stage("vision encoder weights (bfp8, ~0.5 GiB)"):
                self.vision_tt = VisionTT(self.dev, MODEL_DIR, f"{CACHE}/bfp4", log=log.info)
        tv = int(env("VERIFY_T", "8"))  # longer verify mode for prompt-lookup drafts (0 = off)
        with stage(f"T-row modes (verify T={T}{f', {tv}' if tv else ''}, prefill T={prefill_T}) and MTP head"):
            self.eng.init_spec(T, mtp=True, verify_T=tv or None)
            self.eng.init_prefill(prefill_T)
            if tv:
                self.eng.init_verify(tv)
        if env("LARGE_PREFILL", "1") == "1":  # must come before capture_all
            with stage("large-T prefill and warm-up (compiles every chunk shape)"):
                self.eng.init_large_prefill(int(env("LARGE_T", "1024")))
        if self.lanes > 1:  # batched decode (tsugumi/batch.py): before capture_all
            from tsugumi.batch import BatchMode, lane_counts
            with stage(f"batched decode: {self.lanes} lanes (modes {lane_counts(self.lanes)})"):
                BatchMode(self.eng, self.lanes)
        # every image size bucket, compiled before the traces (tsugumi/vision_tt.py); TSUGUMI_VISION_WARMUP=0 only to
        # reproduce the hang (e.g. to check the capture guard)
        if self.vision_tt is not None and env("VISION_WARMUP", "1") == "1":
            with stage("vision encoder warm-up"):
                self.vision_tt.warmup(int(env("IMAGE_MAX_PIXELS", str(1 << 20))))
        elif self.vision_tt is not None:
            self.vision_tt.max_patches = 1 << 30
        with stage("trace capture"):
            self.eng.capture_all()
        self.gen = Generator(self.eng)
        self.lock = threading.Lock()
        self.sched = None  # batch scheduler (owns the device; requests do not take the lock for their whole run)
        self.vision = None
        if env("IMAGES", "1") == "1":  # image input; the embeddings are looked up by virtual token ids
            from tsugumi.vision import ImageTokens, VisionEncoder
            with stage("image preprocessing" + (" and CPU vision encoder" if self.vision_tt is None else "")):
                self.vision = VisionEncoder(MODEL_DIR, log=log.info, tt=self.vision_tt, device_lock=self.lock,
                                            on_device_error=self.fail, after_device=lambda w: self.check_capture(w))
            self.eng.images = ImageTokens(self.eng.n_embed, int(env("IMAGE_CACHE_TOKENS", "65536")))
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self._force_ids = None
        self._tok_cache = {}
        self._byte_dec = byte_decoder()
        v = ttnn.get_memory_view(self.dev, ttnn.BufferType.DRAM)
        G = 1 << 30
        log.info(f"DRAM: used {v.total_bytes_allocated_per_bank * v.num_banks / G:.2f} GiB, "
                 f"free {v.total_bytes_free_per_bank * v.num_banks / G:.2f} GiB; "
                 f"startup {time.perf_counter() - t_all:.0f}s")
        self.max_seq = max_seq
        self.waiting = 0      # requests waiting for the device
        self.stopping = False  # shutdown requested: running generations end at the next token
        self.active = None    # the running request: dict(id, prompt, generated, since)
        self.dead = None  # reason string once the device is considered hung
        self.backstop = int(env("HANG_BACKSTOP", "300"))
        # debug (TSUGUMI_CAPTURE_GUARD=1): after each use of the device, warn when a program was compiled or DRAM stayed
        # allocated after the trace capture: both have hung the device on a later trace replay (doc/11)
        self.guard = None
        if env("CAPTURE_GUARD", "0") == "1":
            self.guard = dict(progs=self.dev.num_program_cache_entries(), dram=self._dram_used(), last=None)
            log.info(f"capture guard: {self.guard['progs']} programs, {self.guard['dram'] / (1 << 20):.1f} MiB DRAM "
                     f"after the trace capture")
        self.eng.progress = self._arm  # long prefills: re-armed per chunk, not only per generated token
        if self.lanes > 1:
            from tsugumi.batch import BatchScheduler
            arm = lambda: faulthandler.dump_traceback_later(self.backstop, exit=True)
            self.sched = BatchScheduler(self.gen, self.lock, arm=arm, disarm=faulthandler.cancel_dump_traceback_later,
                                        on_error=self.fail, log=log.info)
            self.eng.progress = arm

    def _dram_used(self):
        import ttnn
        v = ttnn.get_memory_view(self.dev, ttnn.BufferType.DRAM)
        return v.total_bytes_allocated_per_bank * v.num_banks

    def check_capture(self, where):
        """Capture guard (debug): new programs / DRAM kept since the trace capture. Call with the device lock held."""
        g = self.guard
        if g is None or self.dead:
            return
        progs, dram = self.dev.num_program_cache_entries(), self._dram_used()
        state = (progs - g["progs"], dram - g["dram"])
        if state != g["last"] and (state[0] > 0 or state[1] > (1 << 20)):  # less DRAM than at the capture is fine
            what = ([f"{state[0]} programs compiled"] if state[0] > 0 else []) + \
                   ([f"{state[1] / (1 << 20):.1f} MiB more DRAM allocated"] if state[1] > (1 << 20) else [])
            log.warning(f"capture guard ({where}): {' and '.join(what)} since the trace capture: risk of a hang on a "
                        f"later trace replay (doc/11)")
        g["last"] = state

    def _arm(self):
        if self.sched is not None:  # batched: the scheduler arms the backstop per device step
            return
        faulthandler.dump_traceback_later(self.backstop, exit=True)

    def _disarm(self):
        if self.sched is None:
            faulthandler.cancel_dump_traceback_later()

    def fail(self, e):
        """Device error during a request: stop serving and exit so run_server.sh can reset the device."""
        if self.dead is None:
            lines = str(e).splitlines() or [""]
            msg = next((l for l in lines if "TIMEOUT" in l), lines[0])  # the TT_THROW header line says little
            self.dead = f"{type(e).__name__}: {msg.strip()[:300]}"
            log.error(f"device error, exiting with {EXIT_HANG} after this response: {self.dead}")
            threading.Timer(1.0, os._exit, [EXIT_HANG]).start()  # let the error response go out first

    def shutdown(self, wait=30, guard=60):
        """Stop a running generation at its next token, wait for the device, close it."""
        import ttnn

        self.stopping = True
        t0 = time.perf_counter()
        if self.sched is not None:
            self.sched.stop()
            self.sched.thread.join(timeout=wait)
        if not self.lock.acquire(timeout=wait):
            log.warning(f"a generation did not stop within {wait}s; closing the device anyway")
        faulthandler.dump_traceback_later(guard, exit=True)  # if closing itself hangs
        try:
            ttnn.close_device(self.dev)
            log.info(f"device closed ({time.perf_counter() - t0:.1f}s)")
        finally:
            faulthandler.cancel_dump_traceback_later()

    def debug_hang(self):
        from tsugumi.hang_kernel import run_hang_kernel
        run_hang_kernel(self.dev)

    def _tokens(self, ids, max_new, stats, sampler=None, think=None, row_cb=None):
        """Generator tokens; re-arms the backstop per token. Device errors go to fail() and are re-raised."""
        if self.sched is not None:
            yield from self._tokens_batched(ids, max_new, stats, sampler, think, row_cb)
            return
        try:
            for t in self.gen.generate(ids, max_new=max_new, stats=stats, sampler=sampler, think=think,
                                       row_cb=row_cb):
                self._arm()
                yield t
        except Exception as e:
            self.fail(e)
            raise

    def _tokens_batched(self, ids, max_new, stats, sampler, think, row_cb):
        """The request runs in a lane of the batch scheduler; tokens come through the job's queue."""
        from tsugumi.batch import END, Job
        job = Job(ids, max_new, stats=stats, sampler=sampler, think=think)
        self.sched.submit(job)
        try:
            while True:
                it = job.get(timeout=1.0)
                if it is None:
                    if self.dead:
                        raise RuntimeError(self.dead)
                    continue
                if it is END:
                    return
                if isinstance(it, Exception):
                    raise it
                t, row = it
                if row_cb is not None:
                    row_cb(*row)
                yield t
        finally:
            job.cancel()  # the consumer stopped (stop string, disconnect, shutdown): the lane ends at its next step

    def jobs_info(self):
        """Running / waiting requests of the batch scheduler (for /health)."""
        busy, waiting = self.sched.jobs()
        return [dict(state=j.state, prompt_tokens=len(j.ids), generated=j.generated,
                     seconds=round(time.time() - j.since, 1)) for j in busy], len(waiting)

    def encode(self, messages, enable_thinking, tools=None, template_kwargs=None, images=None, embed=True):
        """Prompt ids; each image becomes its virtual ids (encoded on the CPU unless cached). images: a dict
        that gets n / tokens / encoded / seconds. embed=False only sizes the images (for /tokenize)."""
        ids = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                           enable_thinking=enable_thinking, **(template_kwargs or {}))
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        return self._expand_images(ids, messages, images, embed)

    def _expand_images(self, ids, messages, info, embed):
        from tsugumi.vision import IMAGE_PAD, image_sources, load_image

        srcs = image_sources(messages)
        if ids.count(IMAGE_PAD) != len(srcs):
            raise ValueError(f"{ids.count(IMAGE_PAD)} image placeholders for {len(srcs)} images")
        if not srcs:
            return ids
        if self.vision is None:
            raise ValueError("image input is off (TSUGUMI_IMAGES=0)")
        reg, t0 = self.eng.images, time.perf_counter()
        out, k, enc, ntok = [], 0, 0, 0
        for t in ids:
            if t != IMAGE_PAD:
                out.append(t)
                continue
            data = load_image(srcs[k])
            k += 1
            key = self.vision.key(data)
            vids = reg.lookup(key, need_emb=embed)
            if vids is None:
                if embed:
                    e, (gh, gw) = self.vision.encode(data)
                    vids = reg.ids(key, e.shape[0], gh, gw)
                    reg.put(key, e)
                    enc += 1
                else:
                    gh, gw = self.vision.grid(data)
                    vids = reg.ids(key, gh * gw, gh, gw)
            out += vids
            ntok += len(vids)
        if embed and ntok > reg.max_tokens // 2:
            raise ValueError(f"images: {ntok} tokens in one request (limit {reg.max_tokens // 2}, "
                             f"TSUGUMI_IMAGE_CACHE_TOKENS / 2)")
        if info is not None:
            info.update(n=len(srcs), tokens=ntok, encoded=enc, seconds=time.perf_counter() - t0)
        return out

    def think_budget(self, budget):
        """ThinkBudget for a request (tsugumi/generator.py): THINK_FORCE text then </think>."""
        from tsugumi.generator import ThinkBudget
        if self._force_ids is None:
            self._force_ids = self.tok.encode(THINK_FORCE, add_special_tokens=False)
        return ThinkBudget(budget, self.think_end, self._force_ids)

    def token_info(self, t):
        """(text, bytes) of one token, cached."""
        r = self._tok_cache.get(t)
        if r is None:
            piece = self.tok.convert_ids_to_tokens(t)
            try:  # byte-level BPE pieces map back to the raw bytes
                b = bytes(self._byte_dec[c] for c in piece)
            except (KeyError, TypeError):  # special tokens
                b = (piece or "").encode()
            r = self._tok_cache[t] = (b.decode("utf-8", errors="replace"), list(b))
        return r

    def stream(self, ids, max_new, sampler=None, stop_check=None, think=None, rows=False):
        """Yields (text_delta, done_info, token rows). Text is decoded incrementally (only the last few tokens are
        re-decoded, so long replies stay O(n)); incomplete UTF-8 is held back. With rows=True, the token rows of each
        delta are (token, top-k logits, top-k ids) of the tokens it covers (for logprobs), else []."""
        import collections
        rowq = collections.deque() if rows else None
        pending = []
        out = []
        stats = {}
        t0 = time.perf_counter()
        t_first = None
        eos = False
        prefix_off = read_off = 0  # decode out[prefix_off:] and emit what goes beyond out[prefix_off:read_off]
        stopped = False
        self._arm()
        toks = self._tokens(ids, max_new, stats, sampler, think,
                            (lambda v, i: rowq.append((v, i))) if rows else None)
        try:
            for t in toks:
                row = rowq.popleft() if rows else None
                if t_first is None:
                    t_first = time.perf_counter()
                if t in self.gen.eos:
                    eos = True
                    continue  # let the generator finish so it records the committed state
                out.append(t)
                if rows:
                    pending.append((t,) + row)
                prefix = self.tok.decode(out[prefix_off:read_off])
                text = self.tok.decode(out[prefix_off:])
                if text.endswith("\ufffd") or len(text) <= len(prefix):
                    continue
                if self.active is not None:
                    self.active["generated"] = len(out)
                yield text[len(prefix):], None, pending
                pending = []
                prefix_off, read_off = read_off, len(out)
                if self.stopping or (stop_check is not None and stop_check()):  # stop string / shutdown
                    stopped = True
                    break
        finally:
            toks.close()  # an early end still records the committed state (Generator handles GeneratorExit)
            self._disarm()
        prefix = self.tok.decode(out[prefix_off:read_off])
        text = self.tok.decode(out[prefix_off:])
        if len(text) > len(prefix) or pending:
            yield text[len(prefix):], None, pending
        dt = time.perf_counter() - (t_first or t0)
        steps = max(stats.get("steps", 0), 1)
        th = dict(think_tokens=stats.get("think_tokens", 0), think_forced=stats.get("think_forced", False)) \
            if think is not None else {}
        yield "", dict(**th,prompt_tokens=len(ids), completion_tokens=len(out), reused=stats.get("reused", 0),
                       reuse=stats.get("reuse"), switched=stats.get("switched", False),
                       prefilled=stats.get("prefilled", 0), t_prefill=stats.get("t_prefill", 0.0),
                       prefill_tps=stats.get("prefilled", 0) / stats["t_prefill"] if stats.get("t_prefill") else 0.0,
                       ttft=(t_first or t0) - t0, decode_tps=len(out) / dt if dt > 0 else 0.0,
                       tokens_per_step=(stats.get("accepted", 0) + steps) / steps,
                       ngram_steps=stats.get("ngram_steps", 0), steps=stats.get("steps", 0),
                       finish="stop" if eos or stopped else "length", total=time.perf_counter() - t0), []


def model_type(model_dir):
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return json.load(f).get("model_type", "")
    except OSError:
        return ""


class InklingBackend(Backend):
    """Inkling 2 Mini (tsugumi/inkling/): one request at a time, decode-only engine (the prompt is fed one token per
    step). Shares the device / hang / shutdown handling and the completions stream of Backend; chat goes through
    chat_events (the reply's special tokens -> reasoning_content / content / tool_calls)."""
    kind = "inkling"

    def __init__(self, max_seq):
        import ttnn
        from transformers import AutoTokenizer

        from tsugumi.inkling.chat import ReplyParser
        from tsugumi.inkling.engine import Engine
        from tsugumi.inkling.generator import InklingGenerator

        t_all = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.lanes = 1
        trace_mb = int(env("TRACE_MB", "1024"))  # 512-row prefill traces (one per context bucket): ~650 MB
        self.dev = ttnn.open_device(device_id=0, trace_region_size=trace_mb << 20, l1_small_size=24576)
        t = time.perf_counter()
        self.eng = Engine(self.dev, MODEL_DIR, CACHE, max_seq=max_seq, log=log.info,
                          experts=env("INKLING_EXPERTS", "auto"))
        log.info(f"engine: {time.perf_counter() - t:.0f}s; trace capture ...")
        t = time.perf_counter()
        self.eng.capture()
        log.info(f"trace capture: done in {time.perf_counter() - t:.1f}s")
        self.gen = InklingGenerator(self.eng, eos=(self.eng.cfg.eos,), progress=self._arm)
        self.parser = lambda: ReplyParser(self.tok)
        self.lock = threading.Lock()
        self.sched = None
        self.vision = None
        self.think_end = None
        self._tok_cache = {}
        self._byte_dec = byte_decoder()
        self.max_seq = max_seq
        self.waiting = 0
        self.stopping = False
        self.active = None
        self.dead = None
        self.backstop = int(env("HANG_BACKSTOP", "300"))
        self.guard = None
        v = ttnn.get_memory_view(self.dev, ttnn.BufferType.DRAM)
        G = 1 << 30
        log.info(f"DRAM: used {v.total_bytes_allocated_per_bank * v.num_banks / G:.2f} GiB, "
                 f"free {v.total_bytes_free_per_bank * v.num_banks / G:.2f} GiB; "
                 f"startup {time.perf_counter() - t_all:.0f}s")

    def _tokens(self, ids, max_new, stats, sampler=None, think=None, row_cb=None):
        try:
            for t in self.gen.generate(ids, max_new, stats=stats, sampler=sampler,
                                       row_cb=(lambda v, i: row_cb(v, i)) if row_cb else None):
                self._arm()
                yield t
        except Exception as e:
            self.fail(e)
            raise

    def think_budget(self, budget):
        return None

    def encode(self, messages, enable_thinking, tools=None, template_kwargs=None, images=None, embed=True):
        from tsugumi.inkling.chat import effort_value, encode
        if any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") in
               ("image", "image_url", "input_image") for p in m["content"]) for m in messages):
            raise ValueError("image input is not supported for Inkling yet")
        effort = effort_value((template_kwargs or {}).get("reasoning_effort"))
        return encode(self.tok, messages, effort, tools)

    def chat_events(self, ids, max_new, sampler, stop, rows=False):
        """(field, value): reasoning_content / content text, tool_call dicts, logprobs rows (content tokens), then
        ("done", info). stop: a StopMatcher for the content."""
        p = self.parser()
        stats = {}
        t0 = time.perf_counter()
        t_first, n, eos, stopped = None, 0, False, False
        pending = []
        rowq = []
        self._arm()
        toks = self._tokens(ids, max_new, stats, sampler, None, (lambda v, i: rowq.append((v, i))) if rows else None)
        try:
            for t in toks:
                row = rowq.pop(0) if rows else None
                if t_first is None:
                    t_first = time.perf_counter()
                n += 1
                if self.active is not None:
                    self.active["generated"] = n
                if t in self.gen.eos:
                    eos = True
                    continue
                in_content = p.mode == "content" and t not in p.special
                for field, v in p.feed(t):
                    if field == "content":
                        v = stop.feed(v)
                        if v:
                            yield field, v
                    else:
                        yield field, v
                if rows and in_content and not stop.hit:
                    pending.append((t,) + row)
                    yield "logprobs", pending
                    pending = []
                if stop.hit or self.stopping:
                    stopped = True
                    break
        finally:
            toks.close()
            self._disarm()
        for field, v in p.finish():
            if field == "content":
                v = stop.feed(v)
            if v:
                yield field, v
        tail = stop.finish()
        if tail:
            yield "content", tail
        dt = time.perf_counter() - (t_first or t0)
        yield "done", dict(prompt_tokens=len(ids), completion_tokens=n, reused=stats.get("reused", 0),
                           reuse=stats.get("reuse"), switched=False, prefilled=stats.get("prefilled", 0),
                           t_prefill=stats.get("t_prefill", 0.0),
                           prefill_tps=stats.get("prefilled", 0) / stats["t_prefill"] if stats.get("t_prefill") else 0.0,
                           ttft=(t_first or t0) - t0, decode_tps=n / dt if dt > 0 else 0.0, tokens_per_step=1.0,
                           ngram_steps=0, steps=stats.get("steps", 0),
                           finish="stop" if eos or stopped else "length", total=time.perf_counter() - t0)


def byte_decoder():
    """Inverse of the GPT-2 byte-level BPE alphabet (piece character -> byte)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class StopMatcher:
    """OpenAI `stop`: cut the reply before the first stop string (not included). Text that could be the start of a
    stop string is held back until it is decided."""

    def __init__(self, stops):
        self.stops = [x for x in ([stops] if isinstance(stops, str) else (stops or [])) if x][:4]
        self.hold = max((len(x) for x in self.stops), default=1) - 1
        self.buf = ""
        self.hit = False

    def feed(self, text):
        if not self.stops:
            return text
        if self.hit:
            return ""
        self.buf += text
        cut = min((i for i in (self.buf.find(x) for x in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            self.hit = True
            out, self.buf = self.buf[:cut], ""
            return out
        n = len(self.buf) - self.hold
        out, self.buf = (self.buf[:n], self.buf[n:]) if n > 0 else ("", self.buf)
        return out

    def finish(self):
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


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
    if "think_tokens" in info:
        ng += f", thinking {info['think_tokens']} tok" + (" (budget reached)" if info["think_forced"] else "")
    log.info(f"req {tag}: done, {fin}; prompt {info['prompt_tokens']} ({reuse}), ttft {info['ttft']:.2f}s, "
             f"{info['completion_tokens']} tok {info['decode_tps']:.1f} tok/s {info['tokens_per_step']:.2f} tok/step"
             f"{ng}" + (f", {ncalls} tool call{'s' if ncalls > 1 else ''}" if ncalls else "") +
             f", {info['total']:.1f}s")


def load_api_keys(arg, path):
    keys = [k.strip() for k in (arg or "").split(",") if k.strip()]
    if path:
        keys += [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
    return keys


def make_app(backend, api_keys=(), board=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI()
    keys = [k.encode() for k in api_keys]
    OPEN = {"/", "/health"}

    def given_key(req):
        h = req.headers.get("authorization", "")
        return (h[7:] if h[:7].lower() == "bearer " else "").strip().encode()

    def key_ok(given):
        # compare against every key in constant time, so the timing does not tell which one is close
        ok = False
        for k in keys:
            ok |= hmac.compare_digest(given, k)
        return ok

    if keys:
        @app.middleware("http")
        async def auth(req, call_next):
            if req.url.path in OPEN and req.method in ("GET", "HEAD"):
                return await call_next(req)
            given = given_key(req)
            if not key_ok(given):
                log.warning(f"rejected {req.method} {req.url.path} from {req.client.host if req.client else '?'}: "
                            f"{'no' if not given else 'wrong'} API key")
                return JSONResponse({"error": {"message": "invalid or missing API key (Authorization: Bearer ...)",
                                               "type": "invalid_request_error", "code": "invalid_api_key"}},
                                    status_code=401)
            return await call_next(req)

    @app.get("/")
    def webui():  # minimal chat page (webui/index.html)
        return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui", "index.html"))

    @app.get("/health")
    def health(req: Request):
        if backend.dead:
            return JSONResponse({"status": "error", "error": backend.dead}, status_code=503)
        if backend.sched is not None:  # batched decode: several requests at once
            busy, waiting = backend.jobs_info()
            if keys and not key_ok(given_key(req)):
                return {"status": "ok", "busy": bool(busy), "waiting": waiting, "auth": True}
            res = {"status": "ok", "busy": bool(busy), "waiting": waiting, "lanes": backend.lanes, "active": busy}
            if req.query_params.get("sched"):  # scheduler timings since the last ?sched=1 (and reset them)
                res["sched"] = backend.sched.stats()
            if board is not None:
                res["board"] = board.latest()
            return res
        if keys and not key_ok(given_key(req)):
            # stays open for health checks, but the details (running request, board) need a key
            return {"status": "ok", "busy": backend.active is not None, "waiting": backend.waiting,
                    "auth": True}
        a = backend.active
        res = {"status": "ok", "busy": a is not None, "waiting": backend.waiting,
               "active": None if a is None else dict(id=a["id"], prompt_tokens=a["prompt"],
                                                     generated=a["generated"],
                                                     seconds=round(time.time() - a["since"], 1))}
        if board is not None:
            res["board"] = board.latest()
        return res

    async def acquire(req, tag, n_prompt):
        """Wait for the device without blocking the event loop; give up if the client disconnects meanwhile."""
        if backend.sched is not None:  # batched: the scheduler queues the request itself
            return True
        backend.waiting += 1
        try:
            while True:
                if await run_in_threadpool(backend.lock.acquire, True, 0.25):
                    backend.active = dict(id=tag, prompt=n_prompt, generated=0, since=time.time())
                    return True
                if await req.is_disconnected():
                    log.info(f"req {tag}: client disconnected while waiting")
                    return False
        finally:
            backend.waiting -= 1

    def release():
        if backend.sched is not None:
            return
        tag = (backend.active or {}).get("id", "?")
        backend.active = None
        try:
            backend.check_capture(f"req {tag}")
        finally:
            backend.lock.release()

    def unavailable():
        return JSONResponse({"error": {"message": f"device error, server restarting: {backend.dead}",
                                       "type": "server_error"}}, status_code=503)

    if env("DEBUG_HANG", "0") == "1":  # test-only (scripts/test_server_hang.sh)
        @app.post("/debug/hang")
        def debug_hang():
            with backend.lock:  # test-only path: plain blocking lock
                backend._arm()
                try:
                    backend.debug_hang()
                except Exception as e:
                    backend.fail(e)
                    return unavailable()
                finally:
                    faulthandler.cancel_dump_traceback_later()
            return {"status": "no hang"}

    started = int(time.time())

    @app.get("/v1/models")
    def models():
        if getattr(backend, "kind", "") == "inkling":
            meta = dict(weights=f"experts {backend.eng.experts}, other bfp8", kv_cache="bfp8",
                        max_seq=backend.max_seq, model_dir=MODEL_DIR, engine="tsugumi (Inkling 2, Tenstorrent Blackhole)")
        else:
            meta = dict(weights="bfp4", kv_cache=env("KV_DTYPE", "bfp8"), max_seq=backend.max_seq,
                        model_dir=MODEL_DIR, engine="tsugumi (Tenstorrent P100A)")
        return {"object": "list", "data": [{"id": n, "object": "model", "created": started, "owned_by": "tsugumi",
                                            "meta": meta} for n in MODEL_NAMES]}

    def bad_request(msg):
        return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=400)

    def logprob_entries(rows, top_n):
        """Chat-format logprobs for (token, top-k logits, top-k ids) rows. The softmax is taken over the device
        top-k (64) only, so the values are slightly too high; a token outside it (a forced one) gets -9999."""
        import torch
        res = []
        for t, v, i in rows:
            lp = v.double() - torch.logsumexp(v.double(), 0)
            hit = (i == t).nonzero()
            text, b = backend.token_info(t)
            e = dict(token=text, logprob=float(lp[hit[0, 0]]) if len(hit) else -9999.0, bytes=b)
            e["top_logprobs"] = [dict(token=backend.token_info(int(i[j]))[0], logprob=float(lp[j]),
                                      bytes=backend.token_info(int(i[j]))[1]) for j in range(min(top_n, len(i)))]
            res.append(e)
        return res

    def common(body, ids, tag, params, max_default):
        """Shared request checks: (max_new, sampler factory, stops, think budget, error response or None)."""
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or max_default)
        temperature = float(body.get("temperature") or 0.0)
        try:
            if temperature > 0:
                Sampler(temperature, body.get("top_p"), body.get("top_k"), body.get("min_p"), body.get("seed"))
        except (TypeError, ValueError) as e:
            return None, None, None, None, bad_request(f"sampling parameters: {e}")
        room = backend.max_seq - len(ids) - 8
        if room < 1:
            return None, None, None, None, bad_request(f"prompt too long ({len(ids)} tokens)")
        think = params.pop("_think", None)
        if think is not None:  # answer tokens + thinking (budget, forced tokens, one step of overshoot) must fit
            room -= len(think.force) + 8
            think.budget = max(0, min(think.budget, room - 1))
            room -= think.budget
        max_new = max(1, min(max_new, room))

        def make_sampler():  # per request; None = greedy
            if temperature <= 0:
                return None
            return Sampler(temperature, body.get("top_p"), body.get("top_k"), body.get("min_p"), body.get("seed"))

        stops = body.get("stop")
        p = [f"prompt {len(ids)} tok", f"max {max_new}"] + params.pop("_pre", [])
        if think is not None:
            p.append(f"thinking budget {think.budget}")
        if temperature > 0:
            p.append(f"temp {temperature}")
        p += params.pop("_post", [])
        if stops:
            p.append(f"stop {len([stops] if isinstance(stops, str) else stops)}")
        if body.get("logprobs"):
            p.append("logprobs")
        if body.get("stream"):
            p.append("stream")
        ignored = [k for k in ("presence_penalty", "frequency_penalty", "response_format", "logit_bias", "echo",
                               "suffix", "best_of") if body.get(k) not in (None, False, 0, 1, [], "")] + \
            (["n"] if (body.get("n") or 1) > 1 else [])
        if backend.sched is not None:
            busy, nw = backend.jobs_info()
            wait = f" ({len(busy)}/{backend.lanes} lanes busy, {nw} waiting)" if busy else ""
        else:
            wait = f" (waiting: busy, {backend.waiting} ahead)" if backend.lock.locked() else ""
        log.info(f"req {tag}: {', '.join(p)}" + wait +
                 (f"; ignored parameters: {', '.join(ignored)}" if ignored else ""))
        return max_new, make_sampler, stops, think, None

    def usage(info):
        return dict(prompt_tokens=info["prompt_tokens"], completion_tokens=info["completion_tokens"],
                    total_tokens=info["prompt_tokens"] + info["completion_tokens"])

    def error_event():
        return "data: " + json.dumps({"error": {"message": f"device error: {backend.dead}"}}) + "\n\n"

    def streaming(req, tag, n_prompt, gen):
        """SSE response for a blocking generator of SSE strings that runs with the device lock held."""
        async def agen():
            """Runs the blocking generator in worker threads, one chunk at a time, so the event loop stays free.
            When the client disconnects, Starlette cancels this task: the finally closes the generator, which
            ends the generation where it is and releases the device lock (left suspended, it would hold the
            lock until garbage collection)."""
            if not await acquire(req, tag, n_prompt):
                return
            it = gen()
            step_lock = threading.Lock()  # held while a worker thread runs one step of `it`

            def step():
                with step_lock:
                    return next(it, None)

            def cleanup():  # close the generation (records its committed state), then free the device
                with step_lock:
                    try:
                        it.close()
                    finally:
                        release()

            done = False
            try:
                while True:
                    c = await run_in_threadpool(step)
                    if c is None:
                        done = True
                        break
                    yield c
            finally:
                if step_lock.acquire(blocking=False):  # no step running: clean up now
                    step_lock.release()
                    cleanup()
                else:  # cancelled (shutdown) while a worker thread is still in a step: clean up after it
                    threading.Thread(target=cleanup, daemon=True).start()
                if not done:
                    log.info(f"req {tag}: " + ("stopped for shutdown" if backend.stopping
                                               else "client disconnected, generation stopped"))

        return StreamingResponse(agen(), media_type="text/event-stream")

    async def blocking(req, tag, n_prompt, run):
        """Runs run() in a worker thread with the device lock held: (result, None) or (None, error response)."""
        if not await acquire(req, tag, n_prompt):
            return None, JSONResponse({"error": {"message": "client disconnected"}}, status_code=499)
        try:
            res = await run_in_threadpool(run)
        finally:
            release()
        return (None, unavailable()) if res is None else (res, None)

    def top_n(body, key, limit=20):
        v = body.get(key)
        return max(0, min(int(v or 0), limit))

    async def chat_inkling(req, body):
        """Inkling 2: reasoning_effort (OpenAI-style, or chat_template_kwargs.reasoning_effort; enable_thinking false
        -> "none"); the reply's messages are split by their content tokens (tsugumi/inkling/chat.py)."""
        from tsugumi.inkling.chat import effort_value, encode
        model_name = body.get("model") if body.get("model") in MODEL_NAMES else MODEL_NAMES[0]
        kw = dict(body.get("chat_template_kwargs") or {})
        effort = body.get("reasoning_effort", kw.get("reasoning_effort"))
        if effort is None and kw.get("enable_thinking") is False:
            effort = "none"
        if effort is None:
            effort = INKLING_EFFORT[0]
        try:
            effort = effort_value(effort)
        except (TypeError, ValueError) as e:
            return bad_request(str(e))
        tools = body.get("tools") or None
        if (body.get("tool_choice") or "auto") == "none":
            tools = None
        try:
            msgs = body.get("messages", [])
            if any(isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") in
                   ("image", "image_url", "input_image") for p in m["content"]) for m in msgs):
                raise ValueError("image input is not supported for Inkling yet")
            ids = await run_in_threadpool(encode, backend.tok, msgs, effort, tools)
        except Exception as e:
            return bad_request(f"chat template: {e}")
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        tag = rid[-6:]
        params = {"_pre": [f"effort {effort if effort is not None else 'default'}"],
                  "_post": [f"tools {len(tools)}"] if tools else []}
        max_new, make_sampler, stops, _, err = common(body, ids, tag, params, 4096)
        if err is not None:
            return err
        want_lp = bool(body.get("logprobs"))
        ntop = top_n(body, "top_logprobs")
        created = int(time.time())

        def events():
            for field, v in backend.chat_events(ids, max_new, make_sampler(), StopMatcher(stops), rows=want_lp):
                if field == "logprobs":
                    yield field, logprob_entries(v, ntop)
                else:
                    yield field, v

        def chunk(delta, finish=None, usage=None, timings=None, logprobs=None):
            d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if logprobs is not None:
                d["choices"][0]["logprobs"] = {"content": logprobs}
            if usage:
                d["usage"] = usage
            if timings:
                d["timings"] = timings
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        if body.get("stream"):
            def gen():
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
                        elif field == "logprobs":
                            yield chunk({}, logprobs=v)
                        else:
                            yield chunk({field: v})
                except Exception:
                    yield error_event()
                    return
                yield chunk({}, finish="tool_calls" if ncalls else info["finish"], usage=usage(info), timings=info)
                yield "data: [DONE]\n\n"
                log_done(tag, info, ncalls)

            return streaming(req, tag, len(ids), gen)

        def run():
            if backend.dead:
                return None
            parts = {"content": "", "reasoning_content": ""}
            calls, lps, info = [], [], None
            try:
                for field, v in events():
                    if field == "done":
                        info = v
                    elif field == "tool_call":
                        calls.append(v)
                    elif field == "logprobs":
                        lps += v
                    else:
                        parts[field] += v
            except Exception:
                return None
            return parts, calls, lps, info

        res, err = await blocking(req, tag, len(ids), run)
        if err is not None:
            return err
        parts, calls, lps, info = res
        log_done(tag, info, len(calls))
        msg = {"role": "assistant", "content": parts["content"] if parts["content"] or not calls else None}
        if parts["reasoning_content"]:
            msg["reasoning_content"] = parts["reasoning_content"]
        if calls:
            msg["tool_calls"] = calls
        choice = {"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else info["finish"]}
        if want_lp:
            choice["logprobs"] = {"content": lps}
        return {"id": rid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [choice], "usage": usage(info), "timings": info}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if backend.dead:
            return unavailable()
        body = await req.json()
        if getattr(backend, "kind", "") == "inkling":
            return await chat_inkling(req, body)
        # one model: any requested name is accepted; an alias is echoed back, anything else gets the first name
        model_name = body.get("model") if body.get("model") in MODEL_NAMES else MODEL_NAMES[0]
        kw = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kw.pop("enable_thinking", False))
        tb = body.get("thinking_budget", kw.pop("thinking_budget", None))
        if tb is None:
            tb = THINK_BUDGET[0]
        try:
            think = backend.think_budget(int(tb)) if thinking and tb is not None and int(tb) >= 0 else None
        except (TypeError, ValueError):
            return bad_request(f"thinking_budget: expected an integer, got {tb!r}")
        tool_choice = body.get("tool_choice") or "auto"
        tools = body.get("tools") or None
        if tool_choice == "none":
            tools = None
        # tool_choice "required" / a named function: start the reply inside a call (thinking off only; with
        # thinking on the reply starts in <think>, so it is treated as "auto")
        forced = forced_prefix(tool_choice) if tools and not thinking else ""
        imgs = {}
        try:  # in a worker thread: a long prompt takes ~1 s to tokenize, an image 1-15 s to encode
            ids = await run_in_threadpool(backend.encode, prepare_messages(body.get("messages", [])), thinking,
                                          tools, kw, imgs)
        except Exception as e:  # template errors (bad roles, no user message, ...), bad images
            return bad_request(f"chat template / images: {e}")
        if forced:
            ids = ids + backend.tok.encode(forced, add_special_tokens=False)
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        tag = rid[-6:]
        params = {"_pre": ["thinking"] if thinking and think is None else [], "_think": think,
                  "_post": ([f"tools {len(tools)}"] if tools else []) +
                           ([f"images {imgs['n']} ({imgs['tokens']} tok, {imgs['encoded']} encoded in "
                             f"{imgs['seconds']:.1f}s)"] if imgs else [])}
        max_new, make_sampler, stops, think, err = common(body, ids, tag, params, 2048)
        if err is not None:
            return err
        want_lp = bool(body.get("logprobs"))
        ntop = top_n(body, "top_logprobs")
        created = int(time.time())

        def events():
            """(field, value) pairs: reasoning_content / content text, tool_call dicts, logprobs entry lists, then
            ("done", info). Stop strings apply to the content (not to the thinking); logprobs cover the content
            tokens (not the thinking)."""
            sp = ThinkSplitter(thinking)  # the template opens <think> itself when thinking is enabled
            tp = ToolStreamParser(tools, prefix=forced) if tools else None
            sm = StopMatcher(stops)

            def content(txt):
                txt = sm.feed(txt) if txt else sm.finish()
                if txt:
                    yield from (tp.feed(txt) if tp else [("content", txt)])

            for delta, done, rows in backend.stream(ids, max_new, make_sampler(), stop_check=lambda: sm.hit,
                                                    think=think, rows=want_lp):
                if done:
                    yield from content("")
                    if tp:
                        yield from tp.finish()
                    if imgs:  # image encoding (before the prefill, not in ttft)
                        done["images"] = dict(imgs)
                    yield "done", done
                    return
                for field, txt in sp.feed(delta):
                    if field == "content":
                        yield from content(txt)
                    else:
                        yield field, txt
                if rows and not sp.in_think and not sm.hit:
                    rows = [r for r in rows if r[0] != backend.think_end]
                    if rows:
                        yield "logprobs", logprob_entries(rows, ntop)

        def chunk(delta, finish=None, usage=None, timings=None, logprobs=None):
            d = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if logprobs is not None:
                d["choices"][0]["logprobs"] = {"content": logprobs}
            if usage:
                d["usage"] = usage
            if timings:
                d["timings"] = timings  # non-standard (prompt reuse, TTFT, decode speed); the web UI shows it
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        if body.get("stream"):
            def gen():  # runs with the device lock held (taken by streaming())
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
                        elif field == "logprobs":
                            yield chunk({}, logprobs=v)
                        else:
                            yield chunk({field: v})
                except Exception:
                    yield error_event()
                    return
                yield chunk({}, finish="tool_calls" if ncalls else info["finish"], usage=usage(info), timings=info)
                yield "data: [DONE]\n\n"
                log_done(tag, info, ncalls)

            return streaming(req, tag, len(ids), gen)

        def run():  # blocking, in a worker thread, with the device lock held
            if backend.dead:
                return None
            parts = {"content": "", "reasoning_content": ""}
            calls, lps, info = [], [], None
            try:
                for field, v in events():
                    if field == "done":
                        info = v
                    elif field == "tool_call":
                        calls.append(v)
                    elif field == "logprobs":
                        lps += v
                    else:
                        parts[field] += v
            except Exception:
                return None
            return parts, calls, lps, info

        res, err = await blocking(req, tag, len(ids), run)
        if err is not None:
            return err
        parts, calls, lps, info = res
        log_done(tag, info, len(calls))
        msg = {"role": "assistant", "content": parts["content"] if parts["content"] or not calls else None}
        if parts["reasoning_content"]:
            msg["reasoning_content"] = parts["reasoning_content"]
        if calls:
            msg["tool_calls"] = calls
        choice = {"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else info["finish"]}
        if want_lp:
            choice["logprobs"] = {"content": lps}
        return {"id": rid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [choice], "usage": usage(info), "timings": info}

    def prompt_ids(body, key="prompt"):
        """A text prompt or a list of token ids (or a one-element list of either) -> token ids."""
        p = body.get(key)
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
            p = p[0]
        if isinstance(p, str):
            return backend.tok.encode(p, add_special_tokens=False)
        if isinstance(p, list) and p and all(isinstance(x, int) for x in p):
            return list(p)
        raise ValueError(f"`{key}`: expected one string or one list of token ids")

    @app.post("/v1/completions")
    async def completions(req: Request):
        """Plain text completion (no chat template): the prompt is continued as is."""
        if backend.dead:
            return unavailable()
        body = await req.json()
        model_name = body.get("model") if body.get("model") in MODEL_NAMES else MODEL_NAMES[0]
        try:
            ids = await run_in_threadpool(prompt_ids, body)
        except ValueError as e:
            return bad_request(str(e))
        rid = "cmpl-" + uuid.uuid4().hex[:24]
        tag = rid[-6:]
        max_new, make_sampler, stops, _, err = common(body, ids, tag, {"_pre": ["completions"]}, 16)
        if err is not None:
            return err
        nlp = body.get("logprobs")
        nlp = None if nlp is None or nlp is False else max(0, min(int(nlp), 20))
        created = int(time.time())

        def legacy(rows, offset):
            """Completions-format logprobs (tokens, token_logprobs, top_logprobs, text_offset)."""
            es = logprob_entries(rows, nlp)
            out = dict(tokens=[e["token"] for e in es], token_logprobs=[e["logprob"] for e in es],
                       top_logprobs=[{x["token"]: x["logprob"] for x in e["top_logprobs"]} for e in es],
                       text_offset=[])
            for e in es:
                out["text_offset"].append(offset)
                offset += len(e["token"])
            return out

        def events():
            """("text", str, logprobs or None) ..., then ("done", info, None)."""
            sm = StopMatcher(stops)
            off = 0
            for delta, done, rows in backend.stream(ids, max_new, make_sampler(), stop_check=lambda: sm.hit,
                                                    rows=nlp is not None):
                txt = sm.finish() if done else sm.feed(delta)
                lp = legacy(rows, off) if rows and not sm.hit else None
                if txt or lp:
                    yield "text", txt, lp
                    off += len(txt)
                if done:
                    yield "done", done, None
                    return

        def chunk(text, finish=None, lp=None, info=None):
            d = {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                 "choices": [{"index": 0, "text": text, "logprobs": lp, "finish_reason": finish}]}
            if info is not None:
                d["usage"], d["timings"] = usage(info), info
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        if body.get("stream"):
            def gen():
                if backend.dead:
                    yield error_event()
                    return
                info = None
                try:
                    for kind, v, lp in events():
                        if kind == "done":
                            info = v
                        else:
                            yield chunk(v, lp=lp)
                except Exception:
                    yield error_event()
                    return
                yield chunk("", finish=info["finish"], info=info)
                yield "data: [DONE]\n\n"
                log_done(tag, info)

            return streaming(req, tag, len(ids), gen)

        def run():
            if backend.dead:
                return None
            text, lps, info = "", None, None
            try:
                for kind, v, lp in events():
                    if kind == "done":
                        info = v
                        continue
                    text += v
                    if lp is not None:
                        if lps is None:
                            lps = lp
                        else:
                            for k in lps:
                                lps[k] += lp[k]
            except Exception:
                return None
            if nlp is not None and lps is None:
                lps = dict(tokens=[], token_logprobs=[], top_logprobs=[], text_offset=[])
            return text, lps, info

        res, err = await blocking(req, tag, len(ids), run)
        if err is not None:
            return err
        text, lps, info = res
        log_done(tag, info)
        return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": text, "logprobs": lps, "finish_reason": info["finish"]}],
                "usage": usage(info), "timings": info}

    @app.post("/tokenize")
    async def tokenize(req: Request):
        """vLLM-style: {"prompt": text} or {"messages": [...], "add_generation_prompt": true,
        "chat_template_kwargs": {...}, "tools": [...]} -> {"count", "max_model_len", "tokens"}."""
        body = await req.json()
        try:
            if "messages" in body:
                kw = dict(body.get("chat_template_kwargs") or {})
                thinking = bool(kw.pop("enable_thinking", False))
                ids = await run_in_threadpool(backend.encode, prepare_messages(body["messages"]), thinking,
                                              body.get("tools") or None, kw, None, False)
            else:
                ids = await run_in_threadpool(prompt_ids, body)
        except Exception as e:
            return bad_request(str(e))
        return {"count": len(ids), "max_model_len": backend.max_seq, "tokens": ids}

    @app.post("/detokenize")
    async def detokenize(req: Request):
        body = await req.json()
        t = body.get("tokens")
        if not isinstance(t, list) or not all(isinstance(x, int) for x in t):
            return bad_request("`tokens`: expected a list of token ids")
        return {"prompt": backend.tok.decode(t)}
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    # Qwen: 262144 (the model limit; bfp8 KV 9 GiB leaves ~1.8 GiB free). Inkling: 32768 (its full-attention layers
    # read the whole max_seq KV every step, so a larger value slows every token)
    ap.add_argument("--max-seq", type=int, default=None)
    ap.add_argument("--served-model-name", default=env("SERVED_MODEL_NAME", ""),
                    help="name(s) in the API, comma separated (default: the model directory name, e.g. Qwen3.8-27B)")
    ap.add_argument("--thinking-budget", type=int, default=int(env("THINKING_BUDGET", "-1")),
                    help="default cap on thinking tokens when a request sets none (-1: no cap)")
    ap.add_argument("--reasoning-effort", default=env("REASONING_EFFORT", ""),
                    help="Inkling: default reasoning_effort when a request sets none (none/minimal/low/medium/high/"
                         "xhigh or 0-0.99; default: the template's 0.9)")
    ap.add_argument("--api-key", default=env("API_KEY", ""),
                    help="require Authorization: Bearer <key>; several keys comma separated (env TSUGUMI_API_KEY)")
    ap.add_argument("--api-key-file", default=env("API_KEY_FILE", ""),
                    help="file with one API key per line (env TSUGUMI_API_KEY_FILE)")
    a = ap.parse_args()
    import uvicorn
    api_keys = load_api_keys(a.api_key, a.api_key_file)
    a.auth = f"{len(api_keys)} API key(s)" if api_keys else "off"

    THINK_BUDGET[0] = a.thinking_budget if a.thinking_budget >= 0 else None
    inkling = model_type(MODEL_DIR).startswith("inkling")
    if a.max_seq is None:
        a.max_seq = 32768 if inkling else 262144
    if a.reasoning_effort:
        from tsugumi.inkling.chat import effort_value
        e = a.reasoning_effort
        INKLING_EFFORT[0] = effort_value(float(e) if e.replace(".", "", 1).isdigit() else e)

    names = [n.strip() for n in a.served_model_name.split(",") if n.strip()]
    if names:
        MODEL_NAMES[:] = names

    log_settings(a, inkling)
    t0 = time.time()
    backend = InklingBackend(a.max_seq) if inkling else Backend(a.max_seq)
    board = BoardMonitor().start()  # after the device is up (not during its init); tt-smi works alongside the engine
    log.info(f"ready in {time.time() - t0:.0f}s: http://{a.host}:{a.port}/ (web UI), /v1/chat/completions")
    if a.host not in ("127.0.0.1", "localhost", "::1") and not api_keys:
        log.warning(f"listening on {a.host} without authentication: anyone on the network can use the server")
    # uvicorn handles SIGINT / SIGTERM: stop accepting, give open requests a few seconds, then return here
    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):  # end running generations at their next token, before the grace period
            backend.stopping = True
            super().handle_exit(sig, frame)

    server = Server(uvicorn.Config(make_app(backend, api_keys, board), host=a.host, port=a.port, log_level="warning",
                                   timeout_graceful_shutdown=5))
    # uvicorn re-raises the signal it caught after shutting down, with the handler that was installed before it:
    # with the default handler that kills the process before the device is closed, so install a harmless one
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: log.info(f"{signal.Signals(signum).name} received"))
    server.run()
    if backend.dead:  # device error path: the process exits with EXIT_HANG from fail()
        return
    log.info("shutting down")
    board.stop()
    backend.shutdown()
    log.info("bye")


def log_settings(a, inkling=False):
    """The settings in effect (TSUGUMI_* / TSUMI_* / QE_* environment, flags), one line each."""
    g = lambda k, d: env(k, d)
    if inkling:
        items = [
            ("model", f"{MODEL_DIR} (Inkling 2)"), ("served as", ", ".join(MODEL_NAMES)), ("weights cache", CACHE),
            ("listen", f"{a.host}:{a.port}"), ("auth", a.auth), ("max_seq", f"{a.max_seq} tokens"),
            ("experts", g("INKLING_EXPERTS", "auto")),
            ("effort", a.reasoning_effort or "template default (0.9)"),
            ("hang detection", f"device timeout {os.environ.get('TT_METAL_OPERATION_TIMEOUT_SECONDS')}s, "
                               f"backstop {g('HANG_BACKSTOP', '300')}s"),
        ]
        log.info("settings:")
        for k, v in items:
            log.info(f"  {k:15s} {v}")
        return
    ngram = "off" if g("NGRAM", "3") == "0" else f"when >= {g('NGRAM_MIN', '6')} tokens repeat"
    items = [
        ("model", MODEL_DIR), ("served as", ", ".join(MODEL_NAMES)), ("weights cache", CACHE),
        ("listen", f"{a.host}:{a.port}"), ("auth", a.auth),
        ("max_seq", f"{a.max_seq} tokens"),
        ("thinking", "no default budget" if a.thinking_budget < 0 else f"default budget {a.thinking_budget} tokens"),
        ("KV cache", g("KV_DTYPE", "bfp8")),
        ("prefill", f"large T {g('LARGE_T', '1024')} (on={g('LARGE_PREFILL', '1')}), SDPA {g('PREFILL_SDPA', 'hifi2')}"),
        ("drafts", f"MTP (vocab {'63,168' if g('DRAFT_VOCAB', '1') == '1' else 'full'}), n-gram {ngram}, "
                   f"verify T 4 / {g('VERIFY_T', '8')}"),
        ("images", "off" if g("IMAGES", "1") != "1" else
                   f"on (ViT on the {'device' if g('VISION_DEVICE', '1') == '1' else 'CPU'}, max {g('IMAGE_MAX_PIXELS', str(1 << 20))} pixels/image, "
                   f"cache {g('IMAGE_CACHE_TOKENS', '65536')} tokens, URL fetch={g('IMAGE_FETCH', '1')})"),
        ("batch", "off (one request at a time)" if int(g("BATCH", "1")) <= 1 else
                  f"{g('BATCH', '1')} lanes (concurrent requests share each step)"),
        ("conversations", f"multi={g('MULTI_CONV', '1')} max={g('MAX_CONVS', '8')} snapshot={g('SNAPSHOT', '1')}"),
        ("hang detection", f"device timeout {os.environ.get('TT_METAL_OPERATION_TIMEOUT_SECONDS')}s, "
                           f"backstop {g('HANG_BACKSTOP', '300')}s"),
    ]
    log.info("settings:")
    for k, v in items:
        log.info(f"  {k:15s} {v}")


if __name__ == "__main__":
    main()
