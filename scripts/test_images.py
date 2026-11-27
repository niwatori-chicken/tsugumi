"""Image input against a running server: answers about drawn images, two images, reuse, streaming, errors.

  python scripts/test_images.py [URL=http://127.0.0.1:8010]
Greedy everywhere."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tsugumi`
import base64
import io
import json
import random
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010").rstrip("/")
fails = []


def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=900) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as e:
        return {"_status": e.code, **json.loads(e.read())}


def post_stream(path, body):
    r = urllib.request.Request(URL + path, json.dumps(dict(body, stream=True)).encode(),
                               {"content-type": "application/json"})
    out = []
    with urllib.request.urlopen(r, timeout=900) as f:
        for line in f:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                out.append(json.loads(line[6:]))
    return out


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        fails.append(name)


NONCE = tuple(random.randrange(256) for _ in range(3))  # a different image every run: no state from earlier runs


def data_url(im):
    im.putpixel((0, 0), NONCE)
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def shape(color, kind, size=(512, 384)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    box = [size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4]
    (d.ellipse if kind == "circle" else d.rectangle)(box, fill=color)
    return im


def text_image(text, size=(1000, 240)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 80)
    except OSError:
        font = ImageFont.load_default(size=80)
    d.text((40, 60), text, fill="black", font=font)
    return im


def ask(parts, max_tokens=80, **kw):
    content = [{"type": "image_url", "image_url": {"url": p}} if p.startswith("data:") else {"type": "text", "text": p}
               for p in parts]
    kw.setdefault("temperature", 0)  # greedy: the checks compare outputs
    return dict(messages=[{"role": "user", "content": content}], max_tokens=max_tokens, **kw)


def answer(r):
    return (r.get("choices") or [{}])[0].get("message", {}).get("content") or json.dumps(r)[:200]


red = data_url(shape("red", "circle"))
blue = data_url(shape("blue", "square"))
num = data_url(text_image("Hello World 2026"))
ja = data_url(text_image("吾輩は猫である", (800, 240)))

print("single image")
t0 = time.time()
r = post("/v1/chat/completions", ask([red, "この画像に描かれている図形の色と形を一言で答えて。"]))
a = answer(r)
check("red circle", "赤" in a and ("円" in a or "丸" in a), f"{a!r} ({time.time() - t0:.1f}s)")
print("    timings:", {k: r.get("timings", {}).get(k) for k in ("reused", "prefilled", "t_prefill")})
r2 = post("/v1/chat/completions", ask([blue, "この画像に描かれている図形の色と形を一言で答えて。"]))
a2 = answer(r2)
check("blue square (same size: no false reuse)", "青" in a2 and ("四角" in a2 or "正方形" in a2 or "長方形" in a2), repr(a2))
r3 = post("/v1/chat/completions", ask([num, "画像の文字をそのまま書き写して。"]))
check("read text (en)", "Hello World 2026" in answer(r3), repr(answer(r3)))
r3 = post("/v1/chat/completions", ask([ja, "画像の文字をそのまま書き写して。"]))
check("read text (ja)", "吾輩は猫である" in answer(r3), repr(answer(r3)))

print("two images")
r4 = post("/v1/chat/completions", ask(["1枚目:", red, "2枚目:", blue, "青い図形があるのは何枚目？数字だけ答えて。"]))
check("which one is blue", "2" in answer(r4), repr(answer(r4)))

print("conversation reuse")
q = ask([red, "この画像の図形は何色？"], max_tokens=40)
first = post("/v1/chat/completions", q)
q2 = dict(q, messages=q["messages"] + [{"role": "assistant", "content": answer(first)},
                                       {"role": "user", "content": "その色を英語で言うと？"}])
second = post("/v1/chat/completions", q2)
tm = second.get("timings", {})
check("second turn reuses the image prefix", (tm.get("reused") or 0) > 200 and "red" in answer(second).lower(),
      f"reused {tm.get('reused')}, prefilled {tm.get('prefilled')}, {answer(second)!r}")
q3 = dict(q2, messages=[dict(q2["messages"][0], content=[{"type": "image_url", "image_url": {"url": blue}},
                                                         {"type": "text", "text": "この画像の図形は何色？"}])]
          + q2["messages"][1:])
third = post("/v1/chat/completions", q3)
check("other image, same history: no reuse past the image", (third.get("timings", {}).get("reused") or 0) < 30,
      f"reused {third.get('timings', {}).get('reused')}")

print("stream")
s = post_stream("/v1/chat/completions", ask([red, "この画像に描かれている図形の色と形を一言で答えて。"]))
sc = "".join(c["choices"][0]["delta"].get("content") or "" for c in s if c.get("choices"))
check("stream = non-stream", sc == a, repr(sc))

print("tokenize / errors")
tk = post("/tokenize", dict(messages=ask([red, "この画像に描かれている図形の色と形を一言で答えて。"])["messages"]))
check("tokenize count = prompt_tokens", tk.get("count") == r["usage"]["prompt_tokens"],
      f"{tk.get('count')} vs {r['usage']['prompt_tokens']}")
bad = post("/v1/chat/completions", ask(["data:image/png;base64,AAAA", "何？"]))
check("broken image -> 400", bad.get("_status") == 400, json.dumps(bad)[:150])
bad2 = post("/v1/chat/completions", ask(["<|image_pad|> これは？"]))
check("placeholder text without an image -> 400", bad2.get("_status") == 400, json.dumps(bad2)[:150])

print("FAILED: " + ", ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
