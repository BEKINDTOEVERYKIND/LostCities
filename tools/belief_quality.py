#!/usr/bin/env python3
"""How good is the learned opponent-hand inference?

Reads an analyze.c JSON dump (which records, per ply, the network's belief
probabilities for the mover's unseen cards next to the omniscient truth) and
reports discrimination and calibration.  A "prior" baseline -- every unseen
card equally likely, hand_size / pool_size -- shows how much of the skill is
actual inference rather than counting.
"""
import json
import sys


def main(path):
    d = json.load(open(path))
    rows = []          # (predicted p, actually held)
    prior_rows = []
    for ply in d["plies"]:
        b = ply.get("belief")
        if not b:
            continue
        # the dump keeps the top-14 cards by belief; that biases calibration
        # upward, so restrict the comparison to what is recorded and say so
        for c in b["cards"]:
            rows.append((c["p"], 1 if c["held"] else 0))

    if not rows:
        print("no belief records in", path)
        return 1

    n = len(rows)
    held = sum(h for _, h in rows)

    # AUC by rank comparison
    pos = sorted(p for p, h in rows if h)
    neg = sorted(p for p, h in rows if not h)
    if pos and neg:
        import bisect
        wins = 0.0
        for p in pos:
            lo = bisect.bisect_left(neg, p)
            hi = bisect.bisect_right(neg, p)
            wins += lo + 0.5 * (hi - lo)
        auc = wins / (len(pos) * len(neg))
    else:
        auc = float("nan")

    print(f"{n} belief records over {len(d['plies'])} plies "
          f"(top-14 unseen cards per ply); {held} actually held")
    print(f"AUC (held vs not, higher = better discrimination): {auc:.3f}")
    print("calibration (predicted -> observed frequency):")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        hi = lo + 0.2
        bucket = [(p, h) for p, h in rows if lo <= p < hi]
        if len(bucket) < 25:
            continue
        obs = sum(h for _, h in bucket) / len(bucket)
        mean_p = sum(p for p, _ in bucket) / len(bucket)
        print(f"  {lo:.1f}-{hi:.1f}: n={len(bucket):5d}  mean predicted "
              f"{mean_p:.2f}  observed {obs:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/analysis.json"))
