"""Image input: the Qwen3.5 vision encoder (on the CPU, HF implementation) and the image-token registry.

Image tokens are given virtual ids >= the embedding table size: each distinct image (by content hash and size
setting) gets its own range, the same on every request. The prompt then says which image it holds, so the
conversation reuse (token-id prefix match) tells images apart, and the engine looks the embeddings up by id.

Positions (M-RoPE): text tokens have the same t/h/w position; an image of gh x gw merged tokens starting at
position s has t = s, h = s + row, w = s + col, and the text after it continues at s + max(gh, gw). So past an
image the RoPE position falls behind the KV position; `layout` gives the spans the engine needs for that."""
import base64
import bisect
import hashlib
import io
import os
import threading
import time
from collections import OrderedDict

import torch

from .env import env
from .model import LazySD

VISION_PREFIX = "model.visual."
IMAGE_PAD = 248056


class ImageTokens:
    """Virtual ids of the images seen so far and their embeddings (LRU on the host, by token count)."""

    def __init__(self, base, max_tokens=65536):
        self.base = base
        self.next = base
        self.by_key = {}  # key -> first id
        self.starts = []  # first ids in order (ranges are allocated upwards)
        self.info = {}  # first id -> (n, gh, gw)
        self.emb = OrderedDict()  # first id -> [n, H] bf16
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

    def ids(self, key, n, gh, gw):
        """The id range of an image (allocated on first sight)."""
        with self.lock:
            b = self.by_key.get(key)
            if b is None:
                b = self.next
                self.next += n
                self.by_key[key] = b
                self.starts.append(b)
                self.info[b] = (n, gh, gw)
            return list(range(b, b + n))

    def lookup(self, key, need_emb=True):
        """Id range of a known image (its embeddings kept: refreshed in the LRU), else None."""
        with self.lock:
            b = self.by_key.get(key)
            if b is None or (need_emb and b not in self.emb):
                return None
            if b in self.emb:
                self.emb.move_to_end(b)
            return list(range(b, b + self.info[b][0]))

    def put(self, key, emb):
        b = self.by_key[key]
        with self.lock:
            self.emb[b] = emb
            self.emb.move_to_end(b)
            tot = sum(e.shape[0] for e in self.emb.values())
            while tot > self.max_tokens and len(self.emb) > 1:
                _, e = self.emb.popitem(last=False)
                tot -= e.shape[0]

    def _find(self, t):
        b = self.starts[bisect.bisect_right(self.starts, t) - 1]
        return b, t - b

    def rows(self, toks):
        """Embedding rows for virtual ids."""
        out = []
        for t in toks:
            b, j = self._find(t)
            e = self.emb.get(b)
            if e is None:
                raise KeyError("image embedding evicted before its prefill")
            out.append(e[j])
        return torch.stack(out)

    def layout(self, ids):
        """M-RoPE spans of the images in ids: [(kv start, n, rope start, gw, rope delta after the image)]."""
        spans = []
        if not ids or max(ids) < self.base:
            return spans
        delta = 0  # rope position - kv position for text at this point
        i = 0
        L = len(ids)
        while i < L:
            t = ids[i]
            if t < self.base:
                i += 1
                continue
            b, j = self._find(t)
            n, gh, gw = self.info[b]
            if j != 0 or ids[i:i + n] != list(range(b, b + n)):
                raise ValueError("image tokens out of order")
            s = i + delta
            delta = s + max(gh, gw) - (i + n)
            spans.append((i, n, s, gw, delta))
            i += n
        return spans


def image_sources(messages):
    """Image URLs in the order the chat template renders them (OpenAI `image_url` parts, or `image`)."""
    out = []
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for it in c:
            if isinstance(it, dict) and ("image" in it or "image_url" in it or it.get("type") in ("image", "image_url")):
                u = it.get("image_url", it.get("image"))
                out.append(u.get("url") if isinstance(u, dict) else u)
    return out


def load_image(url, max_bytes=32 << 20, timeout=30):
    """bytes of a data: URL (base64) or an http(s) URL."""
    if not isinstance(url, str) or not url:
        raise ValueError("image: expected a URL string")
    if url.startswith("data:"):
        head, _, b64 = url.partition(",")
        if ";base64" not in head:
            raise ValueError("image: data URLs must be base64")
        return base64.b64decode(b64)
    if url.startswith(("http://", "https://")):
        if env("IMAGE_FETCH", "1") != "1":
            raise ValueError("image: fetching URLs is off (TSUMI_IMAGE_FETCH=0); send a data: URL")
        import urllib.request
        r = urllib.request.Request(url, headers={"User-Agent": "tsumi"})
        with urllib.request.urlopen(r, timeout=timeout) as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"image: larger than {max_bytes >> 20} MiB")
        return data
    raise ValueError("image: expected a data: or http(s) URL")


class VisionEncoder:
    """Image -> [n, 5120] embeddings, n = (h/32) * (w/32): the HF preprocessing on the CPU, then the ViT + 2x2 merger
    on the device (tt: a tsumi.vision_tt.VisionTT, run under device_lock) or on the CPU (float32, HF model)."""

    def __init__(self, model_dir, max_pixels=None, min_pixels=None, threads=None, log=print, tt=None,
                 device_lock=None, on_device_error=None, after_device=None):
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.initialization import no_init_weights
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        t0 = time.time()
        cfg = AutoConfig.from_pretrained(model_dir).vision_config
        self.tt, self.device_lock, self.on_device_error = tt, device_lock, on_device_error
        self.after_device = after_device
        self.model = None
        if tt is None:
            cfg._attn_implementation = "sdpa"
            with no_init_weights():
                m = Qwen3_5VisionModel(cfg)
            sd = LazySD(model_dir)
            m.load_state_dict({k: sd[VISION_PREFIX + k].float() for k in m.state_dict()}, assign=True, strict=True)
            self.model = m.eval()
        self.proc = AutoImageProcessor.from_pretrained(model_dir)
        self.merge = cfg.spatial_merge_size
        self.unit = cfg.patch_size * self.merge  # pixels per merged token side (32)
        self.max_pixels = int(max_pixels or env("IMAGE_MAX_PIXELS", str(1024 * 1024)))
        self.min_pixels = int(min_pixels or env("IMAGE_MIN_PIXELS", str(64 * 1024)))
        self.threads = threads
        self.lock = threading.Lock()  # one image at a time (uses every core)
        log(f"vision encoder ({'device' if tt is not None else 'CPU'}) ready in {time.time() - t0:.0f}s "
            f"(max {self.max_pixels} pixels/image)")

    def key(self, data):
        """Cache key of an image (its bytes and the size setting)."""
        return hashlib.sha256(data + f"|{self.min_pixels}|{self.max_pixels}".encode()).hexdigest()

    def preprocess(self, data):
        """bytes -> (pixel patches, merged grid (gh, gw))."""
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        out = self.proc(images=[img], size={"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels},
                        return_tensors="pt")
        g = out["image_grid_thw"]
        return out["pixel_values"], g, (int(g[0, 1]) // self.merge, int(g[0, 2]) // self.merge)

    def grid(self, data):
        return self.preprocess(data)[2]

    @torch.no_grad()
    def encode(self, data):
        """bytes -> ([n, H] bf16 embeddings, (gh, gw))."""
        pv, g, hw = self.preprocess(data)
        if self.tt is not None:
            self.tt.check(pv.shape[0])
            with self.device_lock:
                try:
                    out = self.tt.forward(pv, g)
                    if self.after_device is not None:
                        self.after_device("image encode")
                    return out, hw
                except Exception as e:
                    if self.on_device_error is not None:
                        self.on_device_error(e)
                    raise
        with self.lock:
            nt = torch.get_num_threads()
            if self.threads:
                torch.set_num_threads(self.threads)
            try:
                out = self.model(pv.float(), grid_thw=g, return_dict=True)
            finally:
                torch.set_num_threads(nt)
        return out.pooler_output.to(torch.bfloat16), hw
