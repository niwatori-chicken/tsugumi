"""Break one profiled decode step (scripts/profile_batch_ops.py or profile_step.py) into categories per layer and
compare the weight matmuls with the DRAM roofline.

  python scripts/analyze_step_profile.py CSV [--bw 512]   (CSV: cpp_device_perf_report.csv of the profiled run)
The main-model step is the second-to-last segment (the last is the MTP run). Layers are split at the RMSNorms on
the 40-core residual grid (input and post-attention norm of every layer)."""
import collections
import csv
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BW = float(sys.argv[sys.argv.index("--bw") + 1]) if "--bw" in sys.argv else 512.0  # GB/s (P150A GDDR6)
rows = []
with open(args[0]) as f:
    for r in csv.DictReader(f):
        try:
            rows.append(dict(name=r["OP NAME"].replace("DeviceOperation", ""), cores=int(r["CORE COUNT"]),
                             dur=int(r["DEVICE KERNEL DURATION [ns]"]) / 1e3,
                             start=int(r["DEVICE KERNEL START CYCLE"]), end=int(r["DEVICE KERNEL END CYCLE"])))
        except (ValueError, KeyError):
            pass
rows.sort(key=lambda r: r["start"])
ghz = sum(r["end"] - r["start"] for r in rows) / max(sum(r["dur"] * 1e3 for r in rows), 1)
segs, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if (b["start"] - a["end"]) / ghz > 2e6:
        segs.append(cur)
        cur = []
    cur.append(b)
segs.append(cur)
main, mtp = segs[-2], segs[-1]

# split into layers at the 40-core LayerNorms: [in_norm .. ) mixer, [post_norm .. ) MLP
norms = [i for i, r in enumerate(main) if r["name"] == "LayerNorm" and r["cores"] == 40]
parts = [main[a:b] for a, b in zip(norms, norms[1:] + [len(main)])]
tail = parts.pop()  # final norm, lm_head, top-k
cat = collections.defaultdict(float)
cnt = collections.Counter()
MB = 1e6
# weight bytes (bfp4: 576 B per 32x32 tile; bfp8 lm_head: 1088 B)
B4, B8 = 576 / 1024, 1088 / 1024
W = dict(gdn_in=5120 * 16480 * B4, gdn_out=6144 * 5120 * B4, attn_qkvg=5120 * 14336 * B4, attn_o=6144 * 5120 * B4,
         mlp_gate=5120 * 17408 * B4, mlp_up=5120 * 17408 * B4, mlp_down=17408 * 5120 * B4, lm_head=5120 * 248320 * B8)
wsum = collections.defaultdict(float)
for k in range(0, len(parts), 2):
    mix, mlp = parts[k], parts[k + 1]
    is_gdn = sum(r["name"] == "GenericOp" for r in mix) == 1  # the GDN kernel; attention: KV write + GQA main + combine
    big = [r for r in mix if r["name"] == "Matmul" and r["dur"] > 20]
    if is_gdn:
        cat["gdn: in proj"] += big[0]["dur"]; wsum["gdn_in"] += big[0]["dur"]
        cat["gdn: out proj"] += big[-1]["dur"]; wsum["gdn_out"] += big[-1]["dur"]
        for r in mix:
            if r in big:
                continue
            if r["name"] == "GenericOp" and r["dur"] > 90:
                cat["gdn: recurrence kernel"] += r["dur"]
            elif r["name"] == "Matmul":
                cat["gdn: conv selection matmuls"] += r["dur"]
            else:
                cat["gdn: small ops"] += r["dur"]
                cnt["gdn small ops"] += 1
    else:
        cat["attn: qkvg proj"] += big[0]["dur"]; wsum["attn_qkvg"] += big[0]["dur"]
        cat["attn: o proj"] += big[-1]["dur"]; wsum["attn_o"] += big[-1]["dur"]
        gen = [r for r in mix if r["name"] == "GenericOp"]
        for r in mix:
            if r in big:
                continue
            if r["name"] == "GenericOp":
                cat["attn: KV write" if r is gen[0] else "attn: GQA kernels"] += r["dur"]
            else:
                cat["attn: small ops"] += r["dur"]
                cnt["attn small ops"] += 1
    bm = [r for r in mlp if r["name"] == "Matmul" and r["dur"] > 20]
    for nm, r in zip(("mlp_gate", "mlp_up", "mlp_down"), bm):
        cat["mlp: " + nm[4:]] += r["dur"]
        wsum[nm] += r["dur"]
    for r in mlp:
        if r not in bm:
            cat["mlp: small ops"] += r["dur"]
            cnt["mlp small ops"] += 1
for r in tail:
    if r["name"] == "Matmul" and r["dur"] > 20:
        cat["lm_head"] += r["dur"]; wsum["lm_head"] += r["dur"]
    elif "Topk" in r["name"]:
        cat["top-k"] += r["dur"]
    else:
        cat["final norm / logits small ops"] += r["dur"]
total = sum(r["dur"] for r in main)
span = (main[-1]["end"] - main[0]["start"]) / ghz / 1e3
print(f"main step: {len(main)} ops, kernel sum {total / 1e3:.2f} ms (span {span / 1e3:.2f} ms, eager), "
      f"{len(parts) // 2} layers")
for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
    n = cnt.get(k.split(":")[0] + " small ops", 0) if "small" in k else 0
    print(f"  {k:34s} {v / 1e3:7.2f} ms  {100 * v / total:5.1f}%" + (f"   ({n} ops)" if n else ""))
print(f"\nweight matmuls vs the DRAM roofline at {BW:.0f} GB/s:")
tw = tt = 0.0
nl = dict(gdn_in=48, gdn_out=48, attn_qkvg=16, attn_o=16, mlp_gate=64, mlp_up=64, mlp_down=64, lm_head=1)
for k, t in wsum.items():
    b = W[k] * nl[k]
    ideal = b / (BW * 1e3)  # us
    tw += ideal
    tt += t
    print(f"  {k:10s} {b / MB / 1e3:6.2f} GB  ideal {ideal / 1e3:6.2f} ms  measured {t / 1e3:6.2f} ms  "
          f"({100 * ideal / t:4.0f}% of peak)")
print(f"  {'all':10s}            ideal {tw / 1e3:6.2f} ms  measured {tt / 1e3:6.2f} ms  ({100 * tw / tt:4.0f}% of peak)")
print(f"\nMTP run: {len(mtp)} ops, kernel sum {sum(r['dur'] for r in mtp) / 1e3:.2f} ms")
