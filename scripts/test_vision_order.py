"""Device ViT inside the server configuration (engine, traces, generator): encode / generate / encode ...

  python scripts/test_vision_order.py [--no-gen] [--vit-only] [--no-warmup]
Without the warm-up (ViT programs first compiled after the trace capture) the second encode after a generation
hung the device (2026-11-20).
Finds which step hangs (TT_METAL_OPERATION_TIMEOUT_SECONDS=30 turns a hang into an exception)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsugumi`
import os
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "30")
import io
import sys
import time

import ttnn
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

import tsugumi.paths as P
from tsugumi.generator import Generator
from tsugumi.model import Precision, Qwen35Engine
from tsugumi.vision import IMAGE_PAD, ImageTokens, VisionEncoder
from tsugumi.vision_tt import VisionTT


def img(color, size=(512, 384)):
    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).ellipse([128, 96, 384, 288], fill=color)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def step(name, f):
    t = time.perf_counter()
    print(f"-> {name}", flush=True)
    r = f()
    print(f"   {name}: {time.perf_counter() - t:.2f}s", flush=True)
    return r


dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
try:
    tok = AutoTokenizer.from_pretrained(P.MODEL)
    eng = Qwen35Engine(dev, P.MODEL, Precision("bfp4"), f"{P.CACHE}/bfp4", max_seq=262144, log=lambda *a: None)
    vt = VisionTT(dev, P.MODEL, f"{P.CACHE}/bfp4")
    vit_only = "--vit-only" in sys.argv
    if "--no-warmup" not in sys.argv:  # every bucket compiled before the traces are captured (as the server does)
        step("ViT warm-up", lambda: vt.warmup(1 << 20))
    if not vit_only:
        eng.init_spec(4, mtp=True, verify_T=8)
        eng.init_prefill(32)
        eng.init_verify(8)
        eng.init_large_prefill(1024)
        eng.capture_all()
    gen = Generator(eng) if not vit_only else None
    enc = VisionEncoder(P.MODEL, tt=vt, device_lock=__import__("threading").Lock())
    eng.images = ImageTokens(eng.n_embed)
    sizes = [(512, 384), (512, 384), (640, 480), (1024, 1024), (300, 900), (512, 384), (1024, 1024), (640, 480)]
    for i, color in enumerate(["red", "blue", "green", "red", "orange", "purple", "black", "pink"]):
        data = img(color, sizes[i])
        e, (gh, gw) = step(f"encode {i} ({color})", lambda: enc.encode(data))
        if vit_only or "--no-gen" in sys.argv:
            continue
        key = enc.key(data)
        vids = eng.images.ids(key, e.shape[0], gh, gw)
        eng.images.put(key, e)
        ids = tok.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "何色？"}]}],
                                      add_generation_prompt=True, enable_thinking=False)
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        k = ids.index(IMAGE_PAD)
        ids = ids[:k] + vids + ids[k + 1:]
        out = step(f"generate {i}", lambda: list(gen.generate(ids, max_new=8)))
        print("  ", repr(tok.decode(out)), flush=True)
    print("all steps done")
finally:
    ttnn.close_device(dev)
