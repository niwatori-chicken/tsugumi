"""chunked SDPA: speed vs k_chunk at long context, and correctness when chunk start is not k_chunk-aligned."""
import sys, time, torch, ttnn
dev = ttnn.open_device(device_id=0, l1_small_size=24576)
NH, NKV, HD, BS = 24, 4, 256, 64
def run(L, start, kc, check):
    nb = ((L + 32 + BS - 1) // BS + 8) // 8 * 8
    torch.manual_seed(0)
    K = torch.randn(nb * BS, NKV, HD) ; V = torch.randn(nb * BS, NKV, HD)
    pg = lambda X: X.view(nb, BS, NKV, HD).permute(0, 2, 1, 3).contiguous()
    kp = ttnn.from_torch(pg(K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    vp = ttnn.from_torch(pg(V), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    pt = ttnn.from_torch(torch.arange(nb, dtype=torch.int32).view(1, nb), dtype=ttnn.int32, device=dev)
    cs = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, device=dev)
    q = torch.randn(1, NH, 32, HD)
    qt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    g = dev.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=32, k_chunk_size=kc)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    f = lambda: ttnn.transformer.chunked_scaled_dot_product_attention(qt, kp, vp, pt, chunk_start_idx_tensor=cs, scale=HD**-0.5, program_config=cfg, compute_kernel_config=ck)
    o = f(); ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(5): o = f()
    ttnn.synchronize_device(dev); dt = (time.perf_counter() - t) / 5 * 1e3
    msg = f"L={L} start={start} kc={kc}: {dt:.2f} ms"
    if check:
        got = ttnn.to_torch(o).float()
        Kf, Vf = K[: start + 32].float(), V[: start + 32].float()
        rep = NH // NKV
        Kh = Kf.permute(1, 0, 2).repeat_interleave(rep, 0); Vh = Vf.permute(1, 0, 2).repeat_interleave(rep, 0)
        s = (q[0].float() @ Kh.transpose(1, 2)) * HD**-0.5
        mask = torch.arange(start + 32)[None, :] > (start + torch.arange(32))[:, None]
        s = s.masked_fill(mask, -1e9)
        ref = torch.softmax(s, -1) @ Vh
        pcc = torch.corrcoef(torch.stack([got[0].flatten(), ref.flatten()]))[0, 1].item()
        msg += f" pcc {pcc:.5f}"
    print(msg, flush=True)
    for x in (kp, vp): ttnn.deallocate(x)
try:
    for kc in (32, 64, 128, 256, 512):
        run(123904, 123904, kc, False)
    for kc in (32, 128, 256):
        for start in (1920, 1952, 1984):
            run(2048, start, kc, True)
finally:
    ttnn.close_device(dev)
