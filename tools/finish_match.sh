#!/bin/sh
# Final evaluation of the match-trained agent.  Waits for the PPO run (or an
# early stop), promotes the checkpoint, and measures everything that matters:
# progress over the imitation start, the heuristic baseline, the search gain,
# and the belief-sampling ablation.  All 3-round paired matches.
set -e
cd "$(dirname "$0")/.."

while pgrep -x rl >/dev/null; do sleep 15; done
cp data/m2.bin data/best.bin

echo "=== final policy vs imitation start (400 paired matches) ==="
./bin/arena -a policy:data/best.bin -b policy:data/ref_m0.bin -n 400 -t 4 -r 3 -s 101

echo
echo "=== final policy vs heuristic (300 paired matches) ==="
./bin/arena -a policy:data/best.bin -b heur -n 300 -t 4 -r 3 -s 102

echo
echo "=== rollout search on top (60 paired matches) ==="
./bin/arena -a rollout:data/best.bin:96:5 -b policy:data/best.bin -n 60 -t 4 -r 3 -s 103

echo
echo "=== belief-sampled worlds vs uniform worlds (60 paired matches) ==="
./bin/arena -a rollout:data/best.bin:96:5 -b rolloutu:data/best.bin:96:5 -n 60 -t 4 -r 3 -s 104

echo
echo "=== ladder (single rounds, 200 pairs) ==="
./bin/ladder -n 200 -t 4 random heur policy:data/ref_m0.bin policy:data/best.bin
