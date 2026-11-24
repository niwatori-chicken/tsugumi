"""kernels/ink_prep_*.cpp (tsugumi/inkling/prep_kernel.py) vs the ttnn-op path of the decode engine, on real layer
weights: q / r / k / v and the updated conv history, sliding and full layers, fused and separate linears.

    python scripts/inkling_test_prep.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import ttnn  # noqa: E402

from tsugumi.env import env  # noqa: E402
from tsugumi.inkling.engine import DRAM, L1, Engine  # noqa: E402
from tsugumi.inkling.prep_kernel import InkPrep  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=4096, log=lambda *a: None, prefill=False)
        c = eng.cfg
        NH, NKV, HD, DR = c.n_heads, c.n_kv, c.head_dim, c.d_rel
        C = NKV * HD
        prep = InkPrep(dev)
        torch.manual_seed(0)
        for layer in (7, 11):
            d = eng.layers[layer]
            sliding = d["sliding"]
            if sliding:
                tkv = ttnn.to_torch(d["tkv"]).float()
                taps = InkPrep.taps_tensor(dev, tkv[..., :C], tkv[..., C:])
            else:
                taps = InkPrep.taps_tensor(dev, d["tk"], d["tv"])
            for fused in (True, False):
                x = eng.tt(torch.randn(1, 1, 32, c.hidden), mem=L1)
                hist0 = torch.randn(1, 1, 32, 2 * C if sliding else C) * 0.5
                hist0[:, :, 0] = 0
                hist0[:, :, 4:] = 0
                mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                               memory_config=DRAM)
                if sliding:
                    hA, hB = mk(hist0), mk(hist0)
                else:
                    hA, hB = (mk(hist0), mk(hist0 * 0.7)), (mk(hist0), mk(hist0 * 0.7))
                nq, nkv = d["nq"], d["nkv"]
                # ---- reference: the ttnn ops of Engine.attention ----
                if fused:
                    y = d["Lqkv"](x, out_mem=L1)
                    yq = y
                    yk = ttnn.slice(y, (0, 0, 0, nq), (1, 1, 32, nq + nkv), memory_config=L1)
                    yv = ttnn.slice(y, (0, 0, 0, nq + nkv), (1, 1, 32, nq + 2 * nkv), memory_config=L1)
                else:
                    yq, yk, yv = d["Lq"](x, out_mem=L1), d["Lk"](x, out_mem=L1), d["Lv"](x, out_mem=L1)
                tau = eng.TAU
                q = eng.l2clip(ttnn.multiply(eng.row(yq, 0, NH * HD, (1, 1, NH, HD)), d["qscale"], memory_config=L1),
                               c.qk_clip, None if sliding else tau)
                o0 = 2 * NH * HD
                r = eng.l2clip(ttnn.multiply(eng.row(yq, o0, o0 + NH * DR, (1, 1, NH, DR)), d["rscale"],
                                             memory_config=L1), c.rel_clip, None if sliding else tau)
                if sliding:
                    kvraw = ttnn.concat([ttnn.slice(yk, (0, 0, 0, 0), (1, 1, 32, C)),
                                         ttnn.slice(yv, (0, 0, 0, 0), (1, 1, 32, C))], dim=-1, memory_config=L1)
                    kv = eng.conv(kvraw, hA, d["tkv"])
                    kr, vr = eng.row(kv, 0, C), eng.row(kv, C, 2 * C)
                else:
                    dk = eng.dyn_taps(ttnn.slice(yk, (0, 0, 0, C), (1, 1, 32, C + 32), memory_config=L1))
                    dv = eng.dyn_taps(ttnn.slice(yv, (0, 0, 0, C), (1, 1, 32, C + 32), memory_config=L1))
                    kr = eng.row(eng.conv(ttnn.slice(yk, (0, 0, 0, 0), (1, 1, 32, C), memory_config=L1), hA[0],
                                          d["tk"], dk), 0, C)
                    vr = eng.row(eng.conv(ttnn.slice(yv, (0, 0, 0, 0), (1, 1, 32, C), memory_config=L1), hA[1],
                                          d["tv"], dv), 0, C)
                k = eng.l2clip(ttnn.reshape(kr, (1, 1, NKV, HD)), c.qk_clip)
                v = eng.rmsu(ttnn.reshape(vr, (1, 1, NKV, HD)))
                ref = [ttnn.to_torch(t).float() for t in (q, r, k, v)]
                # ---- kernel ----
                if fused:
                    ko, vo = nq // 32, (nq + nkv) // 32
                    ksrc = vsrc = y
                else:
                    ko = vo = 0
                    ksrc, vsrc = yk, yv
                if sliding:
                    args = dict(hist_k=hB, hk_off=0, hist_v=hB, hv_off=C // 32)
                else:
                    args = dict(hist_k=hB[0], hk_off=0, hist_v=hB[1], hv_off=0,
                                k_dyn=(ksrc, ko + C // 32), v_dyn=(vsrc, vo + C // 32))
                outs = prep(yq, 0, o0 // 32, ksrc, ko, vsrc, vo, taps=taps, qscale=d["qscale"], rscale=d["rscale"],
                            tau=None if sliding else tau, **args)
                got = [ttnn.to_torch(t).float() for t in outs]
                names = ("q", "r", "k", "v")
                res = []
                for n, a_, b_ in zip(names, got, ref):
                    a_ = a_.reshape(-1, a_.shape[-1])[: b_.reshape(-1, b_.shape[-1]).shape[0]]
                    res.append(f"{n} pcc {pcc(a_, b_.reshape(a_.shape)):.5f}")
                hs = [(hA, hB)] if sliding else [(hA[0], hB[0]), (hA[1], hB[1])]
                for j, (h1, h2) in enumerate(hs):
                    a_, b_ = ttnn.to_torch(h1).float()[0, 0, :4], ttnn.to_torch(h2).float()[0, 0, :4]
                    res.append(f"hist{j} maxdiff {(a_ - b_).abs().max():.4f}")
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                for _ in range(10):
                    prep(yq, 0, o0 // 32, ksrc, ko, vsrc, vo, taps=taps, qscale=d["qscale"], rscale=d["rscale"],
                         tau=None if sliding else tau, **args)
                ttnn.synchronize_device(dev)
                print(f"layer {layer} ({'sliding' if sliding else 'full'}, {'fused' if fused else 'separate'}): "
                      + ", ".join(res) + f"  ({(time.perf_counter() - t) / 10 * 1e6:.0f} us untraced)", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
