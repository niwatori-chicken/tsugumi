"""Expert selection inside a trace (doc/12 H3) with tt-metal's DRAMStreamingExpertsMatmul (deepseek_v3_b1):
all E experts of one projection live in DRAM back to back, an index tensor in L1 says which k to read, and the
kernel computes the addresses on device. Checks that changing only the index tensor between replays of one trace
selects other experts (vs torch), and times the op at Qwen3.6-35B-A3B sizes.

  python scripts/bench_expert_matmul.py [--m 1] [--iters 50]
Sizes: gate_up K=2048 N=1024 (gate and up of one expert), down K=512 N=2048; 256 experts, k=8, bfp4."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
_sys.path.insert(0, _os.path.expanduser("~/tt-metal"))
import argparse
import time

import torch
import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_experts_matmul.op import DRAMStreamingExpertsMatmul


def shuffle_tensor_tiles(t, tile, banks):
    """[K, Np] -> per-bank shards with K tiles contiguous per N column (tt-metal's test helper)."""
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles as f
    return f(t.reshape(1, *t.shape), tile, banks).reshape(t.shape)


def setup(dev, K, N, E, k, m, seed):
    tile = 32
    cores = dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    nc, banks = len(cores), dev.dram_grid_size().x
    Np = -(-N // (tile * banks)) * tile * banks
    per = Np // banks
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, m, K, generator=g).bfloat16().float()
    xm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(grid, [m, K], ttnn.ShardOrientation.ROW_MAJOR))
    xt = ttnn.from_torch(x.repeat(1, 1, nc, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=xm, tile=ttnn.Tile([m, tile]))
    wm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                           ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(
                               banks - 1, dev.dram_grid_size().y - 1))}), [K, per], ttnn.ShardOrientation.ROW_MAJOR))
    ws, w_host = [], []
    for e in range(E):  # one tensor per expert, allocated back to back (the kernel offsets by e * expert size)
        w = (torch.randn(K, Np, generator=g) * 0.05).bfloat16()
        w_host.append(w)
        ws.append(ttnn.from_torch(shuffle_tensor_tiles(w, tile, banks).reshape(1, 1, K, Np).contiguous(),
                                  dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wm))
    im = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR))
    idx = ttnn.from_torch(torch.zeros(nc, 16, dtype=torch.int32).to(torch.uint16), dtype=ttnn.uint16,
                          layout=ttnn.TILE_LAYOUT, device=dev, memory_config=im, tile=ttnn.Tile([1, 16]))
    om = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                           ttnn.ShardSpec(grid, (m, per * k), ttnn.ShardOrientation.ROW_MAJOR))
    out = ttnn.from_torch(torch.zeros(1, 1, m, Np * k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=om, tile=ttnn.Tile([m, tile]))
    return dict(x=x, xt=xt, ws=ws, w_host=w_host, idx=idx, im=im, out=out, nc=nc, banks=banks, per=per, Np=Np, k=k,
                m=m, K=K)


def set_index(S, sel):
    t = torch.zeros(S["nc"], 16, dtype=torch.int32)
    t[:, :len(sel)] = torch.tensor(sel, dtype=torch.int32)
    h = ttnn.from_torch(t.to(torch.uint16), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([1, 16]))
    ttnn.copy_host_to_device_tensor(h, S["idx"])


def golden(S, sel):
    """Output layout: core c holds [e0 cols c*per.., e1 cols c*per.., ...] (per expert, its slice of N)."""
    k, per = S["k"], S["per"]
    res = [S["x"] @ S["w_host"][e].float() for e in sel]
    out = torch.zeros(1, 1, S["m"], S["nc"] * per * k)
    for c in range(S["nc"]):
        for j in range(k):
            out[..., (c * k + j) * per:(c * k + j + 1) * per] = res[j][..., c * per:(c + 1) * per]
    return out


def run(S):
    return DRAMStreamingExpertsMatmul.op(S["xt"], S["ws"][0], S["out"], index_tensor=S["idx"], fp32_dest_acc_en=True,
                                         math_fidelity=ttnn.MathFidelity.LoFi, subblock_k=S["K"] // 32,
                                         selected_experts_k=S["k"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20, l1_small_size=24576)
    try:
        for name, K, N in (("gate_up", 2048, 1024), ("down", 512, 2048)):
            t0 = time.time()
            S = setup(dev, K, N, 256, 8, a.m, seed=1)
            print(f"{name}: {S['nc']} cores / {S['banks']} banks, N padded {S['Np']}, upload {time.time() - t0:.0f}s",
                  flush=True)
            sels = [[3, 77, 128, 200, 5, 90, 255, 0], [10, 11, 12, 13, 250, 251, 64, 32]]
            set_index(S, sels[0])
            run(S)
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            run(S)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            for sel in sels + sels[::-1]:  # same trace, only the index tensor changes
                set_index(S, sel)
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                got = ttnn.to_torch(S["out"]).float()
                ref = golden(S, sel)
                pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                print(f"  experts {sel}: PCC {pcc:.4f}", flush=True)
            ttnn.release_trace(dev, tid)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            for _ in range(a.iters):
                run(S)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            dt = (time.perf_counter() - t0) / a.iters
            ttnn.release_trace(dev, tid)
            mb = 8 * K * S["Np"] * 576 / 1024 / 1e6
            print(f"  m={a.m}: {dt * 1e6:.1f} us for 8 experts ({mb:.1f} MB, {mb / 1e3 / dt:.0f} GB/s)", flush=True)
            for w in S["ws"]:
                ttnn.deallocate(w)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
