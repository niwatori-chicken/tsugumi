"""GatedNormKernel (kernels/gnorm_*.cpp) vs torch fp64: head-major GDN output -> gated RMSNorm in [T, NV*128].
  python scripts/test_gnorm.py [T=2048]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import sys, time
import torch
import ttnn
from tsugumi.gnorm_kernel import GatedNormKernel, gnorm_consts

T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
NV, ZOFF, NP = 48, 10240, 16480 + 32 * 5  # Qwen3.8-27B GDN: z after qkv (10240 columns); in-proj padded width
torch.manual_seed(0)
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
ok = True
try:
    w = torch.randn(128).to(torch.bfloat16) * 0.3 + 1.0
    consts = gnorm_consts(dev, w)
    k = GatedNormKernel(dev, NV, ZOFF)
    for odt in (ttnn.float32, ttnn.bfloat16):
        o = (torch.randn(NV, T, 128) * torch.rand(NV, T, 1) * 3).to(torch.bfloat16)
        y = torch.randn(1, 1, T, NP).to(torch.bfloat16)
        ot = ttnn.from_torch(o.float() if odt == ttnn.float32 else o, dtype=odt, layout=ttnn.TILE_LAYOUT, device=dev)
        yt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, NV * 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                             ttnn.DRAM_MEMORY_CONFIG)
        k(ot, yt, consts, out)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(10):
            k(ot, yt, consts, out)
        ttnn.synchronize_device(dev)
        dt = (time.perf_counter() - t0) / 10
        got = ttnn.to_torch(out).double().reshape(T, NV, 128)
        od = o.double().permute(1, 0, 2)  # [T, NV, 128]
        z = y[0, 0, :, ZOFF:ZOFF + NV * 128].double().reshape(T, NV, 128)
        ref = od * torch.rsqrt(od.pow(2).mean(-1, keepdim=True) + 1e-6) * w.double() * torch.nn.functional.silu(z)
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        good = pcc > 0.9999
        ok &= good
        print(f"[{'ok' if good else 'NG'}] o {str(odt).split('.')[-1]}: PCC {pcc:.6f}, max err {err:.2e} (of max), "
              f"{dt * 1e3:.3f} ms", flush=True)
        for t in (ot, yt, out):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(dev)
print("all passed" if ok else "FAILED")
