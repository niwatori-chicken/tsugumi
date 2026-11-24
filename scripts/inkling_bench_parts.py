"""Traced time of the parts of an Inkling 2 Mini decode step (the device profiler needs a Tracy build).

    python scripts/inkling_bench_parts.py
Each part is captured in its own trace, repeated as often as it runs per step, and timed; the sum is compared
with the full step."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import ttnn  # noqa: E402

from tsugumi.env import env  # noqa: E402
from tsugumi.inkling.engine import L1, Engine  # noqa: E402

MODEL = env("INKLING_DIR", os.path.expanduser("~/models/Inkling-2-Mini"))
CACHE = env("INKLING_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
                                          "tsugumi-cache", "inkling"))


def timed(dev, fn, n=5):
    fn()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t) / n
    ttnn.release_trace(dev, tid)
    return dt


def main():
    dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20, l1_small_size=24576)
    try:
        eng = Engine(dev, MODEL, CACHE, max_seq=int(env("MAX_SEQ", "8192")), log=lambda *a: None)
        c = eng.cfg
        eng.reset()
        eng.write_inputs(1000, 5)
        H = c.hidden
        x = eng.tt(torch.randn(1, 1, 32, H) * 0.1, mem=L1)
        d_s, d_f = eng.layers[7], eng.layers[11]
        st_s, st_f = eng.state[7], eng.state[11]
        res = {}
        sc = {}

        def consts():
            sc.update(eng.step_consts())
        res["step consts x1"] = timed(dev, consts)
        sc.update(eng.step_consts())
        res["moe x48"] = timed(dev, lambda: [eng.moe(d_s, x) for _ in range(48)])
        res["attention sliding x40"] = timed(dev, lambda: [eng.attention(d_s, st_s, x, x, x, sc) for _ in range(40)])
        res["attention full x8"] = timed(dev, lambda: [eng.attention(d_f, st_f, x, x, x, sc) for _ in range(8)])
        res["conv 2304 x96"] = timed(dev, lambda: [eng.conv(x, st_s["ha"], d_s["ta"]) for _ in range(96)])
        N = eng.stack(eng.stack(None, 0, eng.rmsu(x)), 1, eng.rmsu(x))
        res["attn_res1 x96"] = timed(dev, lambda: [eng.attn_res1(d_s["qa_row"], N, 2) for _ in range(96)])
        res["attn_res (3-4 queries) x32"] = timed(dev, lambda: [eng.attn_res(d_s["Qa"], N, 2) for _ in range(32)])
        res["rms_norm x200"] = timed(dev, lambda: [eng.rmsu(x) for _ in range(200)])
        res["msg_gate x96"] = timed(dev, lambda: [eng.msg_gate(d_s, "attn", x, x) for _ in range(96)])
        res["Lq x48"] = timed(dev, lambda: [d_s["Lq"](x, out_mem=L1) for _ in range(48)])
        res["Lo x48"] = timed(dev, lambda: [d_s["Lo"](ttnn.pad(ttnn.slice(x, (0, 0, 0, 0), (1, 1, 32, 2304)),
                                                                [(0, 0), (0, 0), (0, 0), (0, 768)], 0.0),
                                                       out_mem=L1) for _ in range(48)])
        res["full step (lm_head)"] = timed(dev, lambda: eng.forward(True), n=3)
        res["full step (no lm_head)"] = timed(dev, lambda: eng.forward(False), n=3)
        for k, v in res.items():
            print(f"{k:28s} {v * 1e3:8.2f} ms")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
