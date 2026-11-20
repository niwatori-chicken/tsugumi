"""Summarize the device profile written by `python -m tracy -r -p scripts/profile_step.py`.

  python scripts/summarize_profile.py [CSV] [--segments N] [--seq]
Reads tt-metal's cpp_device_perf_report.csv (default: $TT_METAL_HOME/generated/profiler/.logs/), splits the ops
into segments at host synchronizations (start gaps > 2 ms), and for the last N segments (default 2: main step,
MTP step) prints device kernel time per op name (sum, count, mean) plus the sum vs. the first-to-last span.
--seq also prints the op sequence of the first segment shown (index, name, cores, µs)."""
import collections
import csv
import os
import sys

args, it = [], iter(sys.argv[1:])
for a in it:
    if a == "--segments":
        next(it)
    elif not a.startswith("--"):
        args.append(a)
path = args[0] if args else os.path.join(os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal")),
                                         "generated/profiler/.logs/cpp_device_perf_report.csv")
nseg = int(sys.argv[sys.argv.index("--segments") + 1]) if "--segments" in sys.argv else 2
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        try:
            rows.append(dict(name=r["OP NAME"], cores=int(r["CORE COUNT"]), dur=int(r["DEVICE KERNEL DURATION [ns]"]),
                             start=int(r["DEVICE KERNEL START CYCLE"]), end=int(r["DEVICE KERNEL END CYCLE"])))
        except (ValueError, KeyError):
            pass
rows.sort(key=lambda r: r["start"])
ghz = sum(r["end"] - r["start"] for r in rows) / max(sum(r["dur"] for r in rows), 1)  # cycles per ns
segs, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if (b["start"] - a["end"]) / ghz > 2e6:
        segs.append(cur)
        cur = []
    cur.append(b)
segs.append(cur)
print(f"{len(rows)} ops, {len(segs)} segments, clock {ghz:.3f} GHz; showing the last {nseg}")
for si, seg in enumerate(segs[-nseg:]):
    tot = sum(r["dur"] for r in seg)
    span = (seg[-1]["end"] - seg[0]["start"]) / ghz
    print(f"\n== segment {len(segs) - nseg + si}: {len(seg)} ops, kernel sum {tot / 1e6:.2f} ms, span {span / 1e6:.2f} ms")
    by = collections.defaultdict(lambda: [0, 0])
    for r in seg:
        k = r["name"] if r["name"] != "GenericOpDeviceOperation" else f"GenericOp[{r['cores']} cores]"
        by[k][0] += r["dur"]
        by[k][1] += 1
    for k, (d, n) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        if d / tot < 0.002:
            continue
        print(f"  {k:48s} {d / 1e6:7.2f} ms  {100 * d / tot:5.1f}%  x{n:<5d} {d / n / 1e3:7.1f} us")
    if "--seq" in sys.argv and si == 0:
        for i, r in enumerate(seg):
            print(f"    {i:5d} {r['name'][:40]:40s} {r['cores']:4d} {r['dur'] / 1e3:8.1f}")
