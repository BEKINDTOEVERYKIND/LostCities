#!/bin/sh
# Wait for the third PPO phase, then keep its result only if it actually beats
# the model it started from over a large paired match.
set -e
cd "$(dirname "$0")/.."

while pgrep -x rl >/dev/null; do sleep 15; done

echo "=== phase 3 result vs its starting point ==="
./bin/arena -a policy:data/rl3.bin -b policy:data/ref3.bin -n 800 -t 4

M=$(./bin/arena -a policy:data/rl3.bin -b policy:data/ref3.bin -n 800 -t 4 -s 99 -q | awk '{print $1}')
case "$M" in
    -*) echo "phase 3 did not improve ($M); keeping the previous model" ;;
    *)  cp data/rl3.bin data/best.bin; echo "phase 3 improved by $M points/game; best.bin updated" ;;
esac

echo
echo "=== final agent vs the fixed references ==="
./bin/arena -a policy:data/best.bin -b heur -n 400 -t 4
./bin/arena -a rollout:data/best.bin:128:4 -b policy:data/best.bin -n 120 -t 4
