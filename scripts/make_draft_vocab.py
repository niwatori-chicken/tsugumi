"""Build the reduced draft vocabulary (tsumi/draft_vocab.pt): the N most frequent tokens of a mixed corpus.

The MTP draft head only proposes tokens from this set (its lm_head is 4x smaller at the default 63,168 tokens); the main model still verifies
over the full vocabulary, so outputs do not change, only how often a draft is right.
  python scripts/make_draft_vocab.py [N=63168]     (HELDOUT_TEXT=file adds a held-out text)
Train text: Japanese Wikipedia, WikiText-2, JCommonsenseQA, GPQA questions, HumanEval+ prompts / solutions, tt-metal
Python and C++ sources, the model's own answers from the attn8 / mixed benchmark runs, and public corpora in
/mnt/HDD/eval/corpus (TinyStories valid, WikiText-103 part 1, OpenAssistant oasst1 validation) when present. Held out: the bfp4 runs'
answers (MGSM-ja, HumanEval, GPQA reasoning summaries, JSQuAD) and ja_eval / en_eval. Prints the share of
held-out tokens inside the set (an upper bound on how often the drafts could still be right)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # lowbit/ for `import tsumi`
import collections, csv, glob, json, os, random, sys
import torch
from transformers import AutoTokenizer
import tsumi.paths as tsumi_paths

N = int(sys.argv[1]) if len(sys.argv) > 1 else 63168
E = "/mnt/HDD/eval"
tok = AutoTokenizer.from_pretrained(tsumi_paths.MODEL)


def jsonl(path, key):
    try:
        return [str(json.loads(l).get(key) or "") for l in open(path)]
    except FileNotFoundError:
        return []


# groups are weighted equally (token frequencies normalized within each group), so the large English corpora do
# not crowd out Japanese and code
groups = collections.defaultdict(list)
groups["ja"] += [open(f"{E}/ja_wiki.txt").read()]
with open(f"{E}/bench/jglue/jcommonsenseqa-v1.3_train-v1.3.json") as f:
    groups["ja"] += [l for l in f]
groups["en"] += [open(f"{E}/wikitext2_test.txt").read()]
with open(f"{E}/bench/gpqa/gpqa_diamond.csv") as f:
    for r in csv.DictReader(f):
        groups["en"] += [r.get("Question", ""), r.get("Correct Answer", ""), r.get("Explanation", "")]
for l in open(f"{E}/bench/humanevalplus/test.jsonl"):
    d = json.loads(l)
    groups["code"] += [d.get("prompt", ""), d.get("canonical_solution", "")]
home = os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal"))
src = sorted(glob.glob(f"{home}/ttnn/ttnn/**/*.py", recursive=True)) + \
    sorted(glob.glob(f"{home}/models/tt_transformers/**/*.py", recursive=True)) + \
    sorted(glob.glob(f"{home}/ttnn/cpp/ttnn/operations/**/*.cpp", recursive=True))
random.Random(0).shuffle(src)
size = 0
for p in src:
    s_ = open(p, errors="ignore").read()
    groups["code"].append(s_)
    size += len(s_)
    if size > 6_000_000:
        break
# public corpora in /mnt/HDD/eval/corpus (general English, fiction, multilingual chat incl. Japanese)
C = f"{E}/corpus"
if os.path.exists(f"{C}/TinyStoriesV2-GPT4-valid.txt"):
    groups["fiction"].append(open(f"{C}/TinyStoriesV2-GPT4-valid.txt").read()[:8_000_000])
try:
    import pyarrow.parquet as pq
    wt = pq.read_table(f"{C}/wikitext-103-raw-v1/train-00000-of-00002.parquet").column("text").to_pylist()
    groups["en"].append("".join(wt)[:30_000_000])
    for f in glob.glob(f"{C}/data/validation-*.parquet"):
        groups["chat"] += pq.read_table(f).column("text").to_pylist()
except (ImportError, FileNotFoundError) as e:
    print("skipping parquet corpora:", e)
for v in ("attn8", "mixed"):
    for t in ("mgsm", "humaneval", "gpqa", "jsquad", "jemhop"):
        groups["model"] += jsonl(f"{E}/quality/{v}/{t}.jsonl", "pred")
held = [open(f"{E}/ja_eval.txt").read(), open(f"{E}/en_eval.txt").read()]
held_by = {t: jsonl(f"{E}/quality/bfp4/{t}.jsonl", "pred") for t in ("mgsm", "humaneval", "gpqa", "jsquad")}
extra = os.environ.get("HELDOUT_TEXT")  # e.g. a generated English story
if extra:
    held_by["extra"] = [open(extra).read()]

score = collections.Counter()
ntr, distinct = 0, set()
for g, texts in groups.items():
    c = collections.Counter()
    for t in texts:
        c.update(tok.encode(t, add_special_tokens=False))
    n = sum(c.values())
    ntr += n
    distinct |= set(c)
    print(f"  group {g}: {n} tokens, {len(c)} distinct")
    for i, k in c.items():
        score[i] += k / n
cnt = score
special = [tok.convert_tokens_to_ids(t) for t in
           ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>")]
special = [i for i in special if isinstance(i, int) and i >= 0]
ids = list(dict.fromkeys(special + [i for i, _ in cnt.most_common(N)]))[:N]
ids = sorted(ids)
print(f"train: {ntr} tokens, {len(distinct)} distinct; kept {len(ids)}")
S = set(ids)
for name, texts in [("ja_eval + en_eval", held)] + list(held_by.items()):
    t = [i for s in texts for i in tok.encode(s, add_special_tokens=False)]
    if t:
        print(f"  held out {name}: {len(t)} tokens, in set {100 * sum(i in S for i in t) / len(t):.2f}%")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tsumi", "draft_vocab.pt")
torch.save(torch.tensor(ids, dtype=torch.int32), out)
print("wrote", out)
