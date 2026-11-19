"""CausalConvKernel vs torch (bf16 inputs, fp64 ref). python test_conv.py"""
import time, torch, ttnn
from qe.conv_kernel import CausalConvKernel
C = 10240
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=64 << 20)
try:
    K = CausalConvKernel(dev, C)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for T in (64, 256, 1024, 2048):
        torch.manual_seed(0)
        x = torch.randn(T, C).bfloat16().float()
        hist = torch.randn(3, C).bfloat16().float()
        w = (torch.randn(C, 4) * 0.5).bfloat16().float()
        carry = torch.zeros(32, C); carry[29:] = hist
        taps = []
        for j in range(4):
            t = torch.zeros(32, C); t[0] = w[:, j]; taps.append(up(t.view(1, 1, 32, C)))
        xd, cd = up(x.view(1, 1, T, C)), up(carry.view(1, 1, 32, C))
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, C]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        K(xd, cd, taps, out)
        y = ttnn.to_torch(out).float().view(T, C)
        xe = torch.cat([hist, x]).double()
        ref = sum(w[:, j].double() * xe[j:j + T] for j in range(4))
        ref = torch.nn.functional.silu(ref).float()
        err = (y - ref).abs().max().item()
        pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(10): K(xd, cd, taps, out)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / 10 * 1e3
        print(f"T={T}: pcc {pcc:.6f} maxerr {err:.3e}  {ms:.3f} ms (eager)", flush=True)
finally:
    ttnn.close_device(dev)
