#!/usr/bin/env python3
"""Fit prior-aware override thresholds from calib TSV rows.

For every (state, non-top candidate) pair the data holds the cheap
96-world estimate the play-time selection actually sees and a 1024-world
sampled-playout oracle.  The question the fit answers, per prior-gap
bucket and ply band: how big must the cheap edge dm96 be before the
oracle expects the switch to gain points (E[dm1k | dm96 = m] = 0)?

The threshold surface is then tested against the log-linear form
m*(gap) = lambda * log(p_top/p_c) that the prior_w0/w1 spec fields
implement, and per-band lambdas are reduced to the pw0/pw1 pair.

    python3 tools/calfit.py 'scratch/cal*_c*.tsv'
"""
import glob
import math
import sys

LR_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.6, 99.0]
PLY_EDGES = [0, 10, 20, 30, 44, 999]


def load(pattern):
    rows = []
    for f in glob.glob(pattern):
        with open(f) as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            idx = {h: i for i, h in enumerate(hdr)}
            for ln in fh:
                p = ln.rstrip("\n").split("\t")
                if len(p) != len(hdr):
                    continue
                try:
                    pt, pc = float(p[idx["prio_top"]]), float(p[idx["prio_c"]])
                    if pc <= 0 or pt <= 0:
                        continue
                    rows.append((
                        int(p[idx["ply"]]),
                        math.log(pt / pc),
                        float(p[idx["q96_c"]]) - float(p[idx["q96_top"]]),
                        float(p[idx["q1k_c"]]) - float(p[idx["q1k_top"]]),
                    ))
                except (ValueError, KeyError):
                    continue
    return rows


def linreg(xs, ys):
    n = len(xs)
    if n < 30:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / max(n - 2, 1)
    seb = math.sqrt(s2 / sxx)
    return a, b, seb, n


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "cal*_c*.tsv")
    print(f"{len(rows)} candidate rows loaded")
    print("\nfalse-positive rate of the cheap search (dm96>0 but oracle says worse),")
    print("by prior-gap (nats) and ply band -- the raw case for prior handicaps:")
    print("band      " + "".join(f"  lr[{LR_EDGES[i]:.1f},{LR_EDGES[i+1]:.1f})" for i in range(len(LR_EDGES) - 1)))
    for b in range(len(PLY_EDGES) - 1):
        line = f"ply {PLY_EDGES[b]:3d}-{PLY_EDGES[b+1]:3d}"
        for i in range(len(LR_EDGES) - 1):
            sub = [r for r in rows
                   if PLY_EDGES[b] <= r[0] < PLY_EDGES[b + 1]
                   and LR_EDGES[i] <= r[1] < LR_EDGES[i + 1] and r[2] > 0]
            if len(sub) < 20:
                line += "        --"
                continue
            fp = sum(1 for r in sub if r[3] < 0) / len(sub)
            line += f"   {fp*100:5.1f}%({len(sub):4d})"
        print(line)

    print("\nthreshold fit m* per (ply band, prior-gap bucket):")
    print("m* = cheap edge where E[oracle edge]=0, from E[dm1k|dm96]=a+b*dm96")
    lam_by_band = []
    for b in range(len(PLY_EDGES) - 1):
        pts = []
        for i in range(len(LR_EDGES) - 1):
            sub = [r for r in rows
                   if PLY_EDGES[b] <= r[0] < PLY_EDGES[b + 1]
                   and LR_EDGES[i] <= r[1] < LR_EDGES[i + 1]]
            fit = linreg([r[2] for r in sub], [r[3] for r in sub])
            if not fit:
                continue
            a, slope, seb, n = fit
            if slope <= 2 * seb:          # cheap edge carries no oracle signal
                mstar = None
            else:
                mstar = max(0.0, -a / slope)
            mlr = sum(r[1] for r in sub) / len(sub)
            pts.append((mlr, mstar, n))
            ms = "  none" if mstar is None else f"{mstar:6.2f}"
            print(f"  ply {PLY_EDGES[b]:3d}+ lr~{mlr:4.2f}: m*={ms}  "
                  f"(slope {slope:.2f}+-{seb:.2f}, n={n})")
        usable = [(lr, m, n) for lr, m, n in pts if m is not None and lr > 0.05]
        if usable:
            num = sum(m * lr * n for lr, m, n in usable)
            den = sum(lr * lr * n for lr, m, n in usable)
            lam = num / den if den > 0 else 0.0
            lam_by_band.append((0.5 * (PLY_EDGES[b] + min(PLY_EDGES[b + 1], 60)), lam))
            print(f"  ply band {PLY_EDGES[b]}-{PLY_EDGES[b+1]}: lambda = {lam:.2f} pts/nat"
                  + "   residuals: " + " ".join(
                      f"{(m - lam*lr):+.1f}" for lr, m, n in usable))
    if len(lam_by_band) >= 2:
        xs = [p for p, _ in lam_by_band]
        ys = [l for _, l in lam_by_band]
        fit = linreg(xs, ys) if len(xs) >= 30 else None
        # small-N fallback: simple 2-point line through first/last band
        pw0 = ys[0] - (ys[-1] - ys[0]) / (xs[-1] - xs[0]) * xs[0]
        pw1 = pw0 + (ys[-1] - ys[0]) / (xs[-1] - xs[0]) * 44.0
        print(f"\nsuggested spec fields: pw0={pw0:.2f} pw1={pw1:.2f}"
              f"  (lambda per band: {[(round(p), round(l, 2)) for p, l in lam_by_band]})")


if __name__ == "__main__":
    main()
