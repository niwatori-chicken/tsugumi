"""Device ViT (tsugumi/vision_tt.py) against the HF CPU encoder (fp32): image-token embeddings, speed, DRAM.

  python scripts/test_vision_tt.py [WxH ...]   (default: 320x256 640x480 1024x1024 1000x333)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import io
import random
import sys
import time

import torch
import ttnn
from PIL import Image, ImageDraw

import tsugumi.paths as tsugumi_paths
from tsugumi.vision import VisionEncoder
from tsugumi.vision_tt import VisionTT


def image(w, h, seed=0):
    rnd = random.Random(seed)
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    for _ in range(40):
        x, y = rnd.randrange(w), rnd.randrange(h)
        d.rectangle([x, y, x + rnd.randrange(10, w // 3), y + rnd.randrange(10, h // 3)],
                    fill=tuple(rnd.randrange(256) for _ in range(3)))
    d.text((10, 10), "tsugumi vision 2026", fill="black")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


sizes = [tuple(map(int, a.split("x"))) for a in sys.argv[1:]] or [(320, 256), (640, 480), (1024, 1024), (1000, 333)]
cpu = VisionEncoder(tsugumi_paths.MODEL)
dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
G = 1 << 30
mem = lambda: ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
try:
    m0 = mem()
    vt = VisionTT(dev, tsugumi_paths.MODEL, f"{tsugumi_paths.CACHE}/bfp4",
                  wdtype=ttnn.bfloat16 if _os.environ.get("VT_BF16") else ttnn.bfloat8_b)
    m1 = mem()
    print(f"DRAM for the weights: {(m1.total_bytes_allocated_per_bank - m0.total_bytes_allocated_per_bank) * m1.num_banks / G:.3f} GiB")
    for w, h in sizes:
        data = image(w, h)
        pv, g, hw = cpu.preprocess(data)
        t = time.perf_counter()
        ref, _ = cpu.encode(data)
        t_cpu = time.perf_counter() - t
        out = vt.forward(pv, g)  # first call compiles
        ts = []
        for _ in range(2):
            t = time.perf_counter()
            out = vt.forward(pv, g)
            ts.append(time.perf_counter() - t)
        a, b = out.float(), ref.float()
        pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        rows = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        print(f"{w}x{h}: {out.shape[0]} tokens (grid {hw}), pcc {pcc:.5f}, row cosine min {rows.min():.4f} mean {rows.mean():.4f}, "
              f"rel err {((a - b).norm() / b.norm()):.4f}; device {min(ts):.3f}s vs CPU {t_cpu:.2f}s", flush=True)
    m2 = mem()
    print(f"DRAM after: allocated {(m2.total_bytes_allocated_per_bank - m1.total_bytes_allocated_per_bank) * m2.num_banks / G:.3f} GiB more than after loading")
finally:
    ttnn.close_device(dev)
