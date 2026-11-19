"""Attention layer: T=1 sequential decode vs T-row paged path on the same inputs (layer 3 real weights)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import qe`
import sys

import torch
import ttnn

from qe.attn import AttnDecode
from qe.model import LazySD, SpecShared, Qwen35Engine
import json
import qe.paths as qe_paths

MODEL = qe_paths.MODEL
CACHE = qe_paths.CACHE + "/bfp4"


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    try:
        tc = json.load(open(f"{MODEL}/config.json"))["text_config"]
        sd = LazySD(MODEL)
        at = AttnDecode(dev, sd, "model.language_model.layers.3.self_attn.", tc, 1024, wdtype=ttnn.bfloat4_b,
                        cache_prefix=f"{CACHE}/L3.attn")
        at.init_spec(T, chunked=len(sys.argv) > 3 and sys.argv[3] == "chunked")
        from qe.kvpager import KVPager
        import random
        pager = KVPager(dev, 1024 // 64)
        random.Random(0).shuffle(pager.free)  # non-identity logical -> physical blocks
        sh = SpecShared(dev, T, 256, 1024, pager)
        torch.manual_seed(0)
        xs = torch.randn(N, 5120)
        eng = Qwen35Engine.__new__(Qwen35Engine)
        eng.HD, eng.rope_dim, eng.theta = 256, 64, 1e7
        eng.inv_freq = 1.0 / (eng.theta ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
        cos1 = ttnn.from_torch(torch.ones(1, 1, 1, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        sin1 = ttnn.from_torch(torch.zeros(1, 1, 1, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cp1 = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, device=dev)
        ref = []
        for p in range(N):
            x = torch.zeros(1, 1, 32, 5120)
            x[0, 0, 0] = xs[p]
            c, s = Qwen35Engine._rope_rows(eng, p)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), cos1)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sin1)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([p], dtype=torch.int32), dtype=ttnn.int32), cp1)
            o = at.forward_decode(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), cp1, cos1, sin1)
            ref.append(ttnn.to_torch(o)[0, 0, 0, :5120].float())
        for c0 in range(0, N, T):
            pager.ensure(c0 + T)
            x = torch.zeros(1, 1, 32, 5120)
            x[0, 0, :T] = xs[c0:c0 + T]
            cs = [Qwen35Engine._rope_rows(eng, c0 + i) for i in range(T)]
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.cat([a for a, _ in cs], 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sh.cos)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.cat([b for _, b in cs], 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sh.sin)
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.arange(c0, c0 + T, dtype=torch.int32), dtype=ttnn.int32), sh.cur_pos)
            for i in range(T):
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.tensor([c0 + i], dtype=torch.int32), dtype=ttnn.int32), sh.cur_pos_rows[i])
            o = at.forward_spec(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), sh)
            got = ttnn.to_torch(o)[0, 0, :T, :5120].float()
            for i in range(T):
                print(f"pos {c0 + i}: pcc {pcc(got[i], ref[c0 + i]):.5f} |got| {got[i].norm():.3f} |ref| {ref[c0+i].norm():.3f} nan {torch.isnan(got[i]).sum().item()}/{torch.isnan(ref[c0+i]).sum().item()}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
