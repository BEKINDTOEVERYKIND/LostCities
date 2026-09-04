#!/bin/bash
# Turn-arithmetic block, panel 2 rank 5, STAGE 2 ablation (prepared, not run).
# Protocol: data/probes/turnfeat_2026-09-04.txt.  F (new rows trainable) vs
# Z (new rows frozen), 3 seeds each, anchored corpus, deal-keyed 10% holdout,
# readout = held-out correction CE per iteration (minimum over iterations)
# with the deck<=8 / 9-14 / >14 breakdown (--holdout-buckets).
#
# Binary: the worktree build (FEAT_DIM 592; loads data/best.bin with the 36
# new w1 rows zero, saves v6).  Corpus: data/anchor_c20.smp + data/corr21c.smp
# merged by tools/merge_samples.py into data/corr21a.smp (another job); if it
# does not exist yet, merge first:
#   python3 tools/merge_samples.py data/corr21a.smp data/anchor_c20.smp data/corr21c.smp
#
# Rates (the belx-style recipe, tools/belief.c xtrain): the NEW rows at the
# recipe's full 1e-4, the inherited net at 2% of it.  --rowlr LO:HI:SCALE
# multiplies the step of w1 rows [LO,HI) by SCALE, everything else steps at
# --lr, so "--lr 2e-6 --rowlr 556:592:50" = rows 556..591 at 1e-4, rest at
# 2e-6.  Z uses the same base so the arms differ only in whether the new rows
# move (--freeze-rows restores them after every step; it composes with
# --rowlr, verified byte-identical in the smoke test).
#
# Cost: each arm is 8 x 1500 steps x 512 = 6.1M sample-gradients + 9 held-out
# evaluations; ~5 min per run on 4 threads (25-30 min for all six).  Run when
# the miner is done with the box (both want 4 threads).
set -eu
cd /home/user/LostCities             # the corpus lives in the main repo's data/
BIN=/home/user/LostCities/.claude/worktrees/agent-ac9465eeaa4515a49/bin/train
OUT=${OUT:-/tmp/claude-0/-home-user-LostCities/a6899ed3-62e1-5e6d-b08d-0674a82a78fa/scratchpad/stage2}
mkdir -p "$OUT"
CORPUS=data/corr21a.smp
COMMON="--init data/best.bin --data $CORPUS --holdout 0.1 --holdout-buckets \
        --iters 8 --steps 1500 --batch 512 --aug 1 --vw 0 --pw 1 --bw 1 --eval 0 --threads 4"

for s in 1 2 3; do
    # arm F: new rows at the full 1e-4, inherited net at 2% (2e-6)
    $BIN $COMMON --lr 2e-6 --rowlr 556:592:50 --seed $s \
         --out $OUT/F_s$s.bin > $OUT/F_s$s.log 2>&1
    # arm Z: same base rate, new rows frozen (stay exactly zero)
    $BIN $COMMON --lr 2e-6 --freeze-rows 556:592 --seed $s \
         --out $OUT/Z_s$s.bin > $OUT/Z_s$s.log 2>&1
done

# readout: per arm and seed the minimum held-out correction CE over the
# iterations, the deck-bucket breakdown at that iteration and per-bucket
# minima, then the F-vs-Z comparison against the seed spread
python3 /home/user/LostCities/.claude/worktrees/agent-ac9465eeaa4515a49/tools/turnfeat_readout.py \
    $OUT/F_s1.log $OUT/F_s2.log $OUT/F_s3.log $OUT/Z_s1.log $OUT/Z_s2.log $OUT/Z_s3.log \
    | tee $OUT/readout.txt

# Optional reference pair at the STANDARD anchored-recipe base (everything at
# 1e-4), if the question "does the block help on top of the recipe as run
# for c20" is wanted as well; same readout script (name them F1_s*/Z1_s*):
#   $BIN $COMMON --lr 1e-4 --seed $s                        --out $OUT/F1_s$s.bin > $OUT/F1_s$s.log
#   $BIN $COMMON --lr 1e-4 --freeze-rows 556:592 --seed $s  --out $OUT/Z1_s$s.bin > $OUT/Z1_s$s.log
