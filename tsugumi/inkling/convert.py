"""Expert weights in the device layout, cached as .tensorbin (ttnn.dump_tensor) under CACHE/L{i}.experts.*.

gate_up: [1, E, hidden, 2 * inter] (columns: gate | up), down: [1, E, inter, hidden], in the format expert_dtype
picks (bfp8 or bfp4; see there).

    python -m tsugumi.inkling.convert MODEL_DIR CACHE_DIR [bfp8|mixed|bfp4]   # converts the layers not cached yet
"""
import os
import sys
import time

import torch
import ttnn

from ..ops import cache_path
from .weights import Checkpoint

CHUNK = 16  # experts dequantized at a time (float32: 16 x 7.96M x 4 B = 0.5 GB)


def expert_dtype(ck, layer, policy="bfp8"):
    """Device format of a layer's experts. bfp4 re-quantizes NVFP4 with ~11% weight error (bfp8: 0.7%).
    policy: bfp8 (all), mixed (bfp8 for the loop layers, which run 3 times, and the MXFP8 layers), bfp4 (only the
    MXFP8 layers in bfp8)."""
    if not ck.is_nvfp4(layer) or policy == "bfp8":
        return ttnn.bfloat8_b
    if policy == "mixed":
        loop = ck.loop_range
        return ttnn.bfloat8_b if loop[0] <= layer < loop[1] else ttnn.bfloat4_b
    return ttnn.bfloat4_b


def expert_files(cache, layer):
    return f"{cache}/L{layer}.experts.gate_up", f"{cache}/L{layer}.experts.down"


def expert_tensors(ck, layer):
    """(gate_up [1, E, H, 2I], down [1, E, I, H]) bfloat16."""
    E = ck.raw(f"layers.{layer}.mlp.gate.bias").shape[0]
    gu, dn = [], []
    for s in range(0, E, CHUNK):
        g, u, d = ck.experts(layer, s, min(E, s + CHUNK))
        gu.append(torch.cat([g.transpose(1, 2), u.transpose(1, 2)], dim=2).to(torch.bfloat16))
        dn.append(d.transpose(1, 2).contiguous().to(torch.bfloat16))
    return torch.cat(gu)[None], torch.cat(dn)[None]


def convert_layer(ck, cache, layer, log=print, policy="bfp8"):
    dt = expert_dtype(ck, layer, policy)
    files = expert_files(cache, layer)
    if all(os.path.exists(cache_path(f, dt)) for f in files):
        return False
    t0 = time.perf_counter()
    gu, dn = expert_tensors(ck, layer)
    os.makedirs(cache, exist_ok=True)
    for f, w in zip(files, (gu, dn)):
        t = ttnn.from_torch(w, dtype=dt, layout=ttnn.TILE_LAYOUT)
        tmp = cache_path(f + ".tmp", dt)
        ttnn.dump_tensor(tmp, t)
        os.replace(tmp, cache_path(f, dt))
    log(f"experts layer {layer} ({str(dt).split('.')[-1]}): {time.perf_counter() - t0:.0f}s")
    return True


def main():
    model, cache = sys.argv[1], sys.argv[2]
    policy = sys.argv[3] if len(sys.argv) > 3 else "bfp8"
    torch.set_num_threads(int(os.environ.get("TSUGUMI_CONVERT_THREADS", "8")))
    ck = Checkpoint(model)
    layers = sorted({int(k.split(".")[3]) for k in ck.where if ".mlp.experts." in k})
    for i in layers:
        convert_layer(ck, cache, i, policy=policy)
    print("done")


if __name__ == "__main__":
    main()
