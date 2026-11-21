"""Translate a Markdown article (English -> Japanese) block by block with the PLaMo 3 Translate server
(plamo_port/serve_plamo.sh, OpenAI-compatible, port 8000).

  python scripts/translate_md.py docs/article/part1.md docs/article/part1.ja.md [--api http://127.0.0.1:8000]
      [--bilingual] [--jobs 8] [--form plain|polite] [--glossary terms.tsv] [--context TEXT|@file] [--history N]

- Kept as is: fenced code blocks, HTML comments, blank lines, table separator rows, cells without letters.
- Translated: headings, paragraphs, list items (with their continuation lines), table cells (one request per cell).
- Inline code, URLs and [[PLACEHOLDERS]] are checked after translation: a block that lost one is reported (and
  marked with <!-- CHECK --> in the output), so it can be fixed by hand.
- --bilingual puts the English source of each translated block in an HTML comment above it (for reviewing).
- PLaMo 3 Translate's chat template options: --form (plain = 常体, polite = 敬体), --glossary (TSV: source<TAB>target
  per line, '#' comments; only the terms that occur in a block are sent with it), --context (text, or @path).
- --history N: translate the headings and paragraphs in order, passing the previous N (source, translation) pairs as
  conversation history (consistent wording across blocks; sequential, so slower). Table cells stay parallel.
Needs only the standard library."""
import argparse
import concurrent.futures as cf
import json
import re
import urllib.request

KEEP_TOKENS = re.compile(r"`[^`]+`|\[\[[^\]]+\]\]|https?://\S+")


def blocks(lines):
    """[(kind, [lines])]: kind in code, comment, blank, heading, table, text."""
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(("code", lines[i:j + 1]))
            i = j + 1
        elif l.lstrip().startswith("<!--"):
            j = i
            while "-->" not in lines[j]:
                j += 1
            out.append(("comment", lines[i:j + 1]))
            i = j + 1
        elif not l.strip():
            out.append(("blank", [l]))
            i += 1
        elif l.startswith("#"):
            out.append(("heading", [l]))
            i += 1
        elif l.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(("table", lines[i:j]))
            i = j
        else:  # a paragraph, or one list item with its indented continuation lines
            j = i + 1
            while (j < len(lines) and lines[j].strip() and not lines[j].startswith(("#", "|", "```"))
                   and not re.match(r"\s*([-*]|\d+\.)\s", lines[j])):
                j += 1
            out.append(("text", lines[i:j]))
            i = j
    return out


OPTS = {}


def translate(api, text, history=()):
    kw = {"target_lang": "ja", "source_lang": "en"}
    if OPTS.get("form"):
        kw["form"] = OPTS["form"]
    if OPTS.get("context"):
        kw["context"] = OPTS["context"]
    terms = [dict(source=a, target=b) for a, b in OPTS.get("glossary", [])  # whole words (plural / -d allowed)
             if re.search(r"\b" + re.escape(a) + r"(s|es|d)?\b", text, re.I)]
    if terms:
        kw["glossary"] = terms
    msgs = []
    for en, ja in history:
        msgs += [{"role": "user", "content": en}, {"role": "assistant", "content": ja}]
    body = dict(model="plamo", messages=msgs + [{"role": "user", "content": text}],
                chat_template_kwargs=kw, temperature=0, max_tokens=4096)
    models = json.loads(urllib.request.urlopen(api + "/v1/models", timeout=30).read())
    body["model"] = models["data"][0]["id"]
    r = urllib.request.Request(api + "/v1/chat/completions", json.dumps(body).encode(),
                               {"content-type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as f:
        return json.loads(f.read())["choices"][0]["message"]["content"].strip()


def split_prefix(s):
    """Markdown prefix ('## ', '- ', '1. ', indentation, '**...**' kept with the text) and the rest."""
    m = re.match(r"(\s*(?:#+\s|[-*]\s|\d+\.\s)?)(.*)", s, re.S)
    return m.group(1), m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--bilingual", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--form", choices=["plain", "polite"])
    ap.add_argument("--glossary")
    ap.add_argument("--context")
    ap.add_argument("--history", type=int, default=0)
    a = ap.parse_args()
    OPTS["form"] = a.form
    if a.context:
        OPTS["context"] = open(a.context[1:], encoding="utf-8").read().strip() if a.context.startswith("@") else a.context
    if a.glossary:
        OPTS["glossary"] = [tuple(l.rstrip("\n").split("\t", 1)) for l in open(a.glossary, encoding="utf-8")
                            if l.strip() and not l.startswith("#") and "\t" in l]
    bl = blocks(open(a.src, encoding="utf-8").read().split("\n"))

    jobs = {}  # key -> english text
    for bi, (kind, ls) in enumerate(bl):
        if kind in ("heading", "text"):
            pre, rest = split_prefix(" ".join(x.strip() if k else x for k, x in enumerate(ls)))
            jobs[(bi, None)] = rest
        elif kind == "table":
            for ri, row in enumerate(ls):
                if re.fullmatch(r"\|[\s:|-]+\|?", row.strip()):
                    continue
                for ci, cell in enumerate(row.strip().strip("|").split("|")):
                    if re.search(r"[A-Za-z]{2}", KEEP_TOKENS.sub("", cell)):
                        jobs[(bi, (ri, ci))] = cell.strip()
    seq = [k for k in jobs if k[1] is None] if a.history else []
    par = [k for k in jobs if k not in set(seq)]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        fut = ex.map(lambda k: translate(a.api, jobs[k]), par)
        hist, done = [], {}
        for k in seq:  # in document order, with the previous pairs as history
            done[k] = translate(a.api, jobs[k], hist[-a.history:])
            hist.append((jobs[k], done[k]))
        done.update(zip(par, fut))

    bad, out = [], []
    for bi, (kind, ls) in enumerate(bl):
        if kind in ("code", "comment", "blank"):
            out += ls
        elif kind in ("heading", "text"):
            en = jobs[(bi, None)]
            ja = done[(bi, None)]
            pre = split_prefix(ls[0])[0]
            lost = [t for t in KEEP_TOKENS.findall(en) if t not in ja]
            if a.bilingual:
                out.append("<!-- " + en.replace("-->", "- ->") + " -->")
            if lost:
                bad.append((en[:60], lost))
                out.append("<!-- CHECK: lost " + ", ".join(lost).replace("-->", "- ->") + " -->")
            out.append(pre + ja)
        else:  # table
            for ri, row in enumerate(ls):
                cells = row.strip().strip("|").split("|")
                new = [" " + done[(bi, (ri, ci))] + " " if (bi, (ri, ci)) in done else c for ci, c in enumerate(cells)]
                out.append("|" + "|".join(new) + "|")
    open(a.dst, "w", encoding="utf-8").write("\n".join(out))
    print(f"{len(jobs)} pieces translated -> {a.dst}")
    for en, lost in bad:
        print(f"CHECK {en!r}...: lost {lost}")


if __name__ == "__main__":
    main()
