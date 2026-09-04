#!/usr/bin/env python3
"""Readout for the turn-arithmetic stage-2 ablation (panel 2, rank 5).

Parses tools/train.c logs written with --holdout (and --holdout-buckets)
and reports, per arm and seed, the held-out correction CE per iteration,
its minimum (with the iteration it was reached at), the deck-bucket
breakdown at that iteration and the per-bucket minima; then per arm the
mean and spread over seeds and the F-vs-Z comparison the protocol asks
for (data/probes/turnfeat_2026-09-04.txt: adopt only if F beats Z on the
deck<=8 held-out correction CE by more than the seed spread without
losing on deck>14).

    python3 tools/turnfeat_readout.py LOG [LOG ...]

Log names are expected to carry the arm and seed as <arm>_s<seed>.log
(e.g. F_s1.log, Z_s3.log); anything else is reported under its basename.
"""
import os
import re
import sys
from collections import OrderedDict

RE_MAIN = re.compile(r"it(\d+) held-out: corr ce ([\d.]+) top1 ([\d.]+)% \(n=(\d+)\) \| rest ce ([\d.]+) top1 ([\d.]+)% \(n=(\d+)\)")
RE_BUCK = re.compile(r"it(\d+) held-out (corr|rest) by deck: (.*)")
RE_ONE = re.compile(r"(deck<=8|deck 9-14|deck>14) ce ([\d.]+) top1 ([\d.]+)% \(n=(\d+)\)")
BUCKETS = ("deck<=8", "deck 9-14", "deck>14")


def parse(path):
    """-> {it: {'corr': (ce, top1, n), 'rest': (...), 'corr_b': {bucket: (ce, top1, n)}, ...}}"""
    its = OrderedDict()
    with open(path) as f:
        for line in f:
            m = RE_MAIN.search(line)
            if m:
                it = int(m.group(1))
                d = its.setdefault(it, {})
                d["corr"] = (float(m.group(2)), float(m.group(3)), int(m.group(4)))
                d["rest"] = (float(m.group(5)), float(m.group(6)), int(m.group(7)))
                continue
            m = RE_BUCK.search(line)
            if m:
                it = int(m.group(1))
                d = its.setdefault(it, {})
                key = m.group(2) + "_b"
                d[key] = {b: (float(ce), float(t1), int(n)) for b, ce, t1, n in RE_ONE.findall(m.group(3))}
    return its


def arm_seed(path):
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]+)_s(\d+)\.log$", base)
    return (m.group(1), int(m.group(2))) if m else (base, 0)


def fmt(x):
    return "   -  " if x is None else f"{x:6.3f}"


def main(paths):
    runs = OrderedDict()
    for p in paths:
        its = parse(p)
        if not its:
            print(f"{p}: no held-out lines found", file=sys.stderr)
            continue
        runs[arm_seed(p)] = (p, its)

    summary = {}   # (arm, seed) -> dict of readouts
    for (arm, seed), (p, its) in runs.items():
        print(f"== {arm} seed {seed}  ({p})")
        have_b = any("corr_b" in d for d in its.values())
        head = "  it   corr ce  top1%    n  | rest ce"
        if have_b:
            head += "  ||  corr " + "  ".join(f"{b:>9s}" for b in BUCKETS)
        print(head)
        best_it, best_ce = None, None
        bmin = {b: None for b in BUCKETS}
        for it, d in its.items():
            if "corr" not in d:
                continue
            ce, t1, n = d["corr"]
            row = f"  {it:2d}   {ce:6.3f}  {t1:5.1f}  {n:5d} | {d['rest'][0]:6.3f}"
            if have_b:
                cb = d.get("corr_b", {})
                row += "  ||       " + "  ".join(fmt(cb[b][0]) if b in cb else fmt(None) for b in BUCKETS)
                for b in BUCKETS:
                    if b in cb and it > 0 and (bmin[b] is None or cb[b][0] < bmin[b][0]):
                        bmin[b] = (cb[b][0], it)
            print(row)
            if it > 0 and (best_ce is None or ce < best_ce):
                best_it, best_ce = it, ce
        if best_it is None:
            print("  (no trained iteration)")
            continue
        d = its[best_it]
        s = {"min_ce": best_ce, "min_it": best_it, "it0": its[0]["corr"][0] if 0 in its else None}
        line = f"  min held-out corr ce {best_ce:.3f} at it{best_it}"
        if s["it0"] is not None:
            line += f" (it0 {s['it0']:.3f}, delta {best_ce - s['it0']:+.3f})"
        if have_b and "corr_b" in d:
            cb = d["corr_b"]
            line += " | at that it: " + ", ".join(f"{b} {cb[b][0]:.3f} (n={cb[b][2]})" for b in BUCKETS if b in cb)
            line += " | per-bucket minima: " + ", ".join(f"{b} {bmin[b][0]:.3f}@it{bmin[b][1]}" for b in BUCKETS if bmin[b])
            for b in BUCKETS:
                s["at_min_" + b] = cb[b][0] if b in cb else None
                s["min_" + b] = bmin[b][0] if bmin[b] else None
        print(line)
        summary[(arm, seed)] = s

    # per-arm aggregate
    arms = OrderedDict()
    for (arm, seed), s in summary.items():
        arms.setdefault(arm, []).append(s)
    if not arms:
        return

    def agg(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        return mean, min(vals), max(vals), len(vals)

    print("\n== per arm (over seeds): mean [min, max] (n)")
    keys = ["min_ce"] + ["at_min_" + b for b in BUCKETS] + ["min_" + b for b in BUCKETS]
    for arm, ss in arms.items():
        parts = []
        for k in keys:
            a = agg([s.get(k) for s in ss])
            if a:
                parts.append(f"{k} {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] ({a[3]})")
        print(f"  {arm}: " + " | ".join(parts))

    if "F" in arms and "Z" in arms:
        print("\n== F vs Z (protocol: F must beat Z on deck<=8 by more than the seed spread and not lose on deck>14)")
        for k, label in (("min_ce", "overall min corr ce"), ("at_min_deck<=8", "deck<=8 at the overall min"),
                         ("min_deck<=8", "deck<=8 per-bucket min"), ("at_min_deck>14", "deck>14 at the overall min"),
                         ("min_deck>14", "deck>14 per-bucket min")):
            f = agg([s.get(k) for s in arms["F"]])
            z = agg([s.get(k) for s in arms["Z"]])
            if not f or not z:
                continue
            spread = max(f[2] - f[1], z[2] - z[1])
            gain = z[0] - f[0]
            verdict = "F better by more than the spread" if gain > spread else (
                "F better, within the spread" if gain > 0 else "F not better")
            print(f"  {label:28s}: F {f[0]:.3f} [{f[1]:.3f},{f[2]:.3f}]  Z {z[0]:.3f} [{z[1]:.3f},{z[2]:.3f}]"
                  f"  gain (Z-F) {gain:+.3f}  seed spread {spread:.3f}  -> {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
