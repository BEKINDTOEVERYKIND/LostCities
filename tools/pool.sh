#!/bin/sh
# pool.sh -- external-opponent regression tripwire (no adoption authority).
#
# Plays SPEC against a fixed pool of non-champion opponents on shared deal
# seeds and writes per-opponent W/L/D, win% and margin.  Pre-registered
# alarm (data/probes/upgrade_panel_2026-09-01.md, rank 1): any per-opponent
# win% drop >= 2 SE (>= 7 points at 200 games) against the recorded
# baseline puts a hold on that spec.  Weak-alien margin is reported, never
# gated (the win-trained champion deliberately gives back margin vs heur).
#
#   tools/pool.sh SPEC OUTFILE [WORKDIR]
#
# Resumable: one chunk file per (opponent, seed); rerun to fill gaps.
set -u
SPEC=$1; OUT=$2; WD=${3:-/tmp/lc_pool}
mkdir -p "$WD"
cd "$(dirname "$0")/.." || exit 1
run() {  # name spec games-per-chunk chunks seedbase
  name=$1; opp=$2; n=$3; chunks=$4; base=$5
  i=0
  while [ $i -lt $chunks ]; do
    f="$WD/${name}_$((base+i)).txt"
    [ -f "$f" ] || ./bin/arena -a "$SPEC" -b "$opp" -n "$n" -r 3 -t 4 -s $((base+i)) > "$f.tmp" && mv "$f.tmp" "$f"
    i=$((i+1))
  done
}
run big1    "rollout:data/big1.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1"     5 40 7000
run s2      "rollout:data/s2.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1"       5 40 7100
run sym1    "rollout:data/sym1.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1"     5 40 7200
run oldbest "rollout:data/old_best.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1" 5 40 7300
run m0      "rollout:data/m0.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1"       5 20 7400
run heur    "heur"                                                            5 20 7500
python3 - "$WD" "$SPEC" > "$OUT" <<'PY'
import sys, glob, re, math, os
wd, spec = sys.argv[1], sys.argv[2]
print("pool tripwire for", spec)
for name in ("big1","s2","sym1","oldbest","m0","heur"):
    w=l=d=0; pa=[]; pc=[]
    for f in sorted(glob.glob(os.path.join(wd, name+"_*.txt"))):
        t=open(f).read()
        m=re.search(r'W/L/D (\d+)/(\d+)/(\d+)', t)
        if not m: continue
        w+=int(m.group(1)); l+=int(m.group(2)); d+=int(m.group(3))
        m2=re.search(r'points/game ([\d.-]+) vs ([\d.-]+)', t)
        if m2: pa.append(float(m2.group(1))); pc.append(float(m2.group(2)))
    g=w+l+d
    if not g: print(f"{name:8s} no games"); continue
    sc=(w+0.5*d)/g; se=math.sqrt(sc*(1-sc)/g)
    mg=(sum(pa)/len(pa)-sum(pc)/len(pc)) if pa else float('nan')
    print(f"{name:8s} games {g:4d}  W/L/D {w}/{l}/{d}  win {sc*100:5.1f}% (SE {se*100:.1f})  margin {mg:+.1f}")
PY
