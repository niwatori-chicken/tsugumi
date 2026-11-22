"""Image input against HF (CPU fp32): M-RoPE positions, then the first N layers' logits with an image in the prompt.

  python scripts/cmp_hf_mm.py [--layers 8] [--size 320x256] [--no-device]
1. positions: the engine's rope_pos for every prompt token (and generated ones) vs HF get_rope_index.
2. logits: HF text model (N layers, inputs_embeds with the image embeddings, 3-D position ids) vs the engine
   in the T-row mode (per row) and the large-T prefill (last row)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import argparse
import io
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

import tsugumi.paths as tsugumi_paths
from tsugumi.model import LM_PREFIX, LazySD
from tsugumi.vision import IMAGE_PAD, ImageTokens, VisionEncoder

MODEL = tsugumi_paths.MODEL


def test_image(w, h):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([w // 8, h // 8, w // 2, h // 2], fill="red")
    d.ellipse([w // 2, h // 2, w - 10, h - 10], fill="blue")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def hf_positions(cfg, ids_hf, grids):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    fake = SimpleNamespace(config=cfg,
                           get_vision_position_ids=lambda *a, **k: Qwen3_5Model.get_vision_position_ids(None, *a, **k))
    t = torch.tensor([ids_hf])
    mm = (t == IMAGE_PAD).int()
    pos, delta = Qwen3_5Model.get_rope_index(fake, t, mm, image_grid_thw=torch.tensor(grids))
    return pos[:, 0], int(delta)


def hf_logits(n_layers, emb, pos3):
    from transformers import AutoConfig
    from transformers.initialization import no_init_weights
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL).text_config
    cfg.num_hidden_layers = n_layers
    cfg.layer_types = cfg.layer_types[:n_layers]
    cfg._attn_implementation = "eager"
    with no_init_weights():
        m = Qwen3_5ForCausalLM(cfg)
    sd = LazySD(MODEL)
    m.load_state_dict({k: sd[k.replace("model.", LM_PREFIX, 1) if k.startswith("model.") else k].float()
                       for k in m.state_dict()}, assign=True, strict=True)
    m.eval()
    with torch.no_grad():
        return m(inputs_embeds=emb[None].float(), position_ids=pos3[:, None], use_cache=False).logits[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--size", default="320x256")
    ap.add_argument("--no-device", action="store_true")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    enc = VisionEncoder(MODEL)
    w, h = map(int, a.size.split("x"))
    imgs = [test_image(w, h), test_image(h, w)]  # two images: positions after one, between both
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "この画像と"},
                                         {"type": "image"}, {"type": "text", "text": "この画像の違いは？"}]}]
    ids_hf = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    ids_hf = ids_hf if isinstance(ids_hf, list) else ids_hf["input_ids"]
    reg = ImageTokens(base=cfg.text_config.vocab_size)
    embs, grids, ids, ids_exp = [], [], [], []
    k = 0
    for t in ids_hf:
        if t == IMAGE_PAD:
            e, (gh, gw) = enc.encode(imgs[k])
            key = enc.key(imgs[k])
            vids = reg.ids(key, e.shape[0], gh, gw)
            reg.put(key, e)
            embs.append(e)
            grids.append([1, gh * 2, gw * 2])
            ids += vids
            ids_exp += [IMAGE_PAD] * len(vids)
            k += 1
        else:
            ids.append(t)
            ids_exp.append(t)
    print(f"prompt {len(ids)} tokens, image grids {grids}")

    # 1. positions
    pos3, delta = hf_positions(cfg, ids_exp, grids)

    from tsugumi.model import Qwen35Engine

    class Probe:  # rope_pos / set_sequence / _rope_rows without a device
        rope_pos = Qwen35Engine.rope_pos
    pr = Probe()
    pr.images, pr.mrope = reg, []
    Qwen35Engine.set_sequence(pr, ids)
    bad = 0
    for p in range(len(ids) + 5):
        r = Qwen35Engine.rope_pos(pr, p)
        r = r if isinstance(r, tuple) else (r, r, r)
        want = tuple(int(x) for x in pos3[:, p]) if p < len(ids) else (p + delta,) * 3
        bad += r != want
    print(f"[{'ok' if not bad else 'FAIL'}] rope positions ({bad} differ), delta {delta}")
    # cos/sin rows vs the HF rotary embedding (interleaved M-RoPE)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    rot = Qwen3_5TextRotaryEmbedding(cfg.text_config)
    hc, hs = rot(torch.zeros(1, dtype=torch.float32), pos3[:, None])
    pr.HD = cfg.text_config.head_dim
    pr.rope_dim = int(pr.HD * cfg.text_config.rope_parameters.get("partial_rotary_factor", 0.25))
    th = cfg.text_config.rope_parameters["rope_theta"]
    pr.inv_freq = 1.0 / (th ** (torch.arange(0, pr.rope_dim, 2, dtype=torch.float64) / pr.rope_dim))
    sec = cfg.text_config.rope_parameters["mrope_section"]
    ax = torch.zeros(len(pr.inv_freq), dtype=torch.long)
    for k_, off in ((1, 1), (2, 2)):
        ax[off: sec[k_] * 3: 3] = k_
    pr.mrope_axis = ax
    err = max(float((Qwen35Engine._rope_rows(pr, p)[0].view(-1)[: pr.rope_dim] - hc[0, p]).abs().max()) +
              float((Qwen35Engine._rope_rows(pr, p)[1].view(-1)[: pr.rope_dim] - hs[0, p]).abs().max())
              for p in range(len(ids)))
    print(f"[{'ok' if err < 1e-4 else 'FAIL'}] cos/sin rows vs HF rotary (max error {err:.2e})")

    # embeddings: text rows from the table, image rows from the encoder
    E = LazySD(MODEL)[LM_PREFIX + "embed_tokens.weight"].to(torch.bfloat16)
    emb = torch.empty(len(ids), E.shape[1], dtype=torch.bfloat16)
    tx = torch.tensor(ids) < E.shape[0]
    emb[tx] = E[torch.tensor(ids)[tx]]
    emb[~tx] = torch.cat(embs)
    if a.no_device:
        return
    ref = hf_logits(a.layers, emb, pos3)
    print("HF done", flush=True)

    import ttnn
    from tsugumi.model import Precision
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=24576)
    try:
        eng = Qwen35Engine(dev, MODEL, Precision("bfp4"), f"{tsugumi_paths.CACHE}/bfp4", max_seq=2048,
                           n_layers=a.layers, log=lambda *x: None)
        eng.images = reg
        eng.init_spec(32)
        eng.init_prefill(32)
        eng.set_sequence(ids)
        # T-row mode, every row
        eng.use_T(32)
        eng.reset_spec()
        rows, a_prev, pos = [], 0, 0
        import ttnn as _t
        while pos < len(ids):
            ch = ids[pos:pos + 32]
            eng.run_spec(ch, pos, a_prev)
            lg = _t.to_torch(eng.last_spec[0])[0, 0, :len(ch), : eng.V].float()
            rows += [lg[i] for i in range(len(ch))]
            a_prev = len(ch)
            pos += len(ch)
        agree, pccs = 0, []
        for p in range(len(ids)):
            pccs.append(torch.corrcoef(torch.stack([rows[p], ref[p]]))[0, 1].item())
            agree += int(rows[p].argmax() == ref[p].argmax())
        img_rows = [p for p in range(len(ids)) if ids[p] >= eng.n_embed]
        txt_after = [p for p in range(len(ids)) if ids[p] < eng.n_embed and p > img_rows[0]]
        mean = lambda xs: sum(xs) / len(xs)
        print(f"T-row mode: top1 agreement {agree}/{len(ids)}, pcc min {min(pccs):.4f} mean {mean(pccs):.4f}, "
              f"image rows {mean([pccs[p] for p in img_rows]):.4f}, text after the image {mean([pccs[p] for p in txt_after]):.4f}")
        last = rows[-1]
        print(f"  last row: tt {tok.decode([int(last.argmax())])!r} hf {tok.decode([int(ref[-1].argmax())])!r}")
        # the same without M-RoPE (plain positions): the text after the image should get worse
        eng.mrope = []
        eng.reset_spec()
        rows2, a_prev, pos = [], 0, 0
        while pos < len(ids):
            ch = ids[pos:pos + 32]
            eng.run_spec(ch, pos, a_prev)
            lg = _t.to_torch(eng.last_spec[0])[0, 0, :len(ch), : eng.V].float()
            rows2 += [lg[i] for i in range(len(ch))]
            a_prev = len(ch)
            pos += len(ch)
        p2 = [torch.corrcoef(torch.stack([rows2[p], ref[p]]))[0, 1].item() for p in txt_after]
        print(f"  (control, plain RoPE: text after the image pcc {mean(p2):.4f})")
        # large-T prefill, last row
        eng.set_sequence(ids)
        eng.init_large_prefill(256)
        eng.reset_spec()
        eng.pager.release(eng.pager.blocks)
        eng.use_T(32)
        eng.prefill(ids, 0, with_mtp=False)
        v, i = eng.last_topk
        r = ref[-1]
        print(f"large prefill: top1 tt {tok.decode([int(i[0, 0])])!r} hf {tok.decode([int(r.argmax())])!r}; "
              f"top-8 tt {i[0, :8].tolist()} hf {r.topk(8).indices.tolist()}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
