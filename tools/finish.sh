#!/bin/sh
# Wait for the running trainer, pick the checkpoint that actually measured best
# against the frozen reference, then run the final evaluation.
set -e
cd "$(dirname "$0")/.."

while pgrep -x train >/dev/null; do sleep 10; done

# choose between the expert-iteration result and the model it started from
BEST=data/rl2.bin
if [ -f data/ex1.bin ]; then
    M=$(./bin/arena -a policy:data/ex1.bin -b policy:data/ref2.bin -n 400 -t 4 -q | awk '{print $1}')
    echo "expert iteration vs its starting point: $M points/game"
    case "$M" in
        -*) BEST=data/rl2.bin ;;
        *)  BEST=data/ex1.bin ;;
    esac
fi
cp "$BEST" data/best.bin
echo "best model: $BEST"

./tools/eval_final.sh 300 data/best.bin
