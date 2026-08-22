#!/usr/bin/env python3
"""Score an agent spec against the reviewer-flagged decision suite.

For every manifest row whose second column is a .state file, the decision is
replayed N times with fresh RNG seeds through the given spec (via the
decreplay harness) and the tally of chosen moves is compared with the
manifest's better/worse verdict:

    python3 tools/suite.py <decreplay-binary> <spec> [seeds=20]

Rows referencing .moves files are legacy (replayable only through qpair's
round-0 reconstruction) and are skipped with a note.  Output: one line per
probe -- %better / %worse / %other -- and a summary.  "better" and "worse"
match modulo wager-copy identity (any Yx counts as Yx).
"""
import subprocess
import sys

SUITS = "YBWGR"


def to_harness(mv):
    """'W4 p deck' -> ('W4', 'play', 0); 'R6 p W' -> ('R6', 'play', 3)"""
    card, act, draw = mv.split()
    act = "play" if act == "p" else "discard"
    d = 0 if draw == "deck" else SUITS.index(draw) + 1
    return card, act, d


def main():
    harness, spec = sys.argv[1], sys.argv[2]
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    rows = []
    for line in open("data/probes/manifest.tsv"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        rows.append(f)
    nb = nw = 0
    scored = 0
    for f in rows:
        name, src, _seed, _ply, better, worse = f[0], f[1], f[2], f[3], f[4], f[5]
        if not src.endswith(".state"):
            print(f"{name:20s} legacy (.moves row), skipped")
            continue
        out = subprocess.run(
            [harness, f"data/probes/{src}", "0", str(seeds), spec],
            capture_output=True, text=True, timeout=7200).stdout
        tally = {}
        for ln in out.strip().splitlines():
            try:
                mvpart, cnt = ln.rsplit(":", 1)
                card, act, d = mvpart.split()[0], mvpart.split()[1], int(mvpart.split()[2].split("-")[1])
                tally[(card, act, d)] = int(cnt.strip().split("/")[0])
            except (ValueError, IndexError):
                continue
        b = tally.get(to_harness(better), 0)
        w = tally.get(to_harness(worse), 0)
        o = seeds - b - w
        verdict = "PASS" if b > w and w == 0 else ("ok" if b >= w else "FAIL")
        print(f"{name:20s} better {b:2d}/{seeds}  worse {w:2d}/{seeds}  other {o:2d}  {verdict}")
        nb += b
        nw += w
        scored += 1
    print(f"\n{scored} probes: better {nb}, worse {nw} "
          f"({100.0 * nb / (scored * seeds):.0f}% / {100.0 * nw / (scored * seeds):.0f}% of decisions)")


if __name__ == "__main__":
    main()
