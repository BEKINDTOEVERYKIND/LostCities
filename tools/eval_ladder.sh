#!/bin/sh
# Round robin over the training checkpoints, to check that each stage really
# beats the one before it rather than only beating the fixed baseline.
set -e
cd "$(dirname "$0")/.."
N="${1:-250}"
./bin/ladder -n "$N" -t 4 \
  heur \
  policy:data/base.bin \
  policy:data/rl1.bin.it10 \
  policy:data/rl1.bin.it25 \
  policy:data/rl1.bin.it40 \
  policy:data/rl1.bin.it60
