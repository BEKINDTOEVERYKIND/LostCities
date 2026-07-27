# Lost Cities AI

A from-scratch Lost Cities engine, neural network, and self-play training
pipeline, written in C with no external dependencies.

Lost Cities (Reiner Knizia) is a two-player imperfect-information card game.
Five suits of twelve cards (three wagers and the numbers 2-10). Each turn you
play a card to one of your own expeditions or discard it, then draw from the
deck or from any discard pile. Expeditions must ascend; wagers must come before
numbers. An expedition scores `(sum of numbers - 20) x (1 + wagers)`, plus 20
if it holds eight or more cards, and only if you opened it at all. The game
ends when the last deck card is drawn.

## Layout

```
src/lc.[ch]         rules engine: state, move generation, scoring
src/features.[ch]   information-set encoding (sparse binary + dense scalars)
src/net.[ch]        two-headed network, forward/backward, Adam, save/load
src/heuristic.[ch]  hand-crafted projection evaluation (baseline, bootstrap)
src/search.[ch]     determinized MCTS with network priors and values
src/agent.[ch]      move-selection policies over a player's information set
src/match.[ch]      paired-deal match runner
src/spec.[ch]       agent command-line specs
tools/train.c       expert iteration / imitation trainer
tools/rl.c          PPO self-play trainer  <- this is what produced the agent
tools/arena.c       head-to-head match with error bars
tools/ladder.c      round robin with fitted Elo
tools/probe.c       value-network diagnostics
tools/bench.c       throughput of every hot path
tools/play.c        play against the agent in a terminal
tests/test_engine.c rule and invariant tests
```

Build and test:

```
make
make test
```

## The agent

One network serves both roles:

* **Trunk** — 432 input features to 256 to 128, ReLU. The input layer is
  sparse: 300 of the features are one-hot card planes (in my hand, in my
  expedition, in yours, in a discard pile, on top of a pile), so the first
  layer is mostly column additions rather than multiplies.
* **Value head** — expected final score margin from the perspective player's
  point of view.
* **Policy head** — a move is `(card, play-or-discard, draw source)`. Rather
  than one logit for each of the 720 combinations, the head keeps one logit per
  `(card, disposition)` and one per draw source and adds them, with the softmax
  taken over legal moves only. That shares statistics across combinations a
  flat head would have to learn separately, and it is cheaper: a turn touches
  at most 16 play components and 6 draw components.

Everything an agent sees is its own information set. Deck draws are unknown at
decision time, so they are averaged over the cards the player has not seen. The
same sample of unseen cards is reused across all candidate moves of a turn --
common random numbers, which removes nearly all the sampling noise from the
*comparison* between moves.

## How it was trained, and what did not work

The training story is worth recording, because the first two approaches failed
for the same measurable reason.

**Attempt 1: value network, one-ply greedy.** Regress the final score margin,
then pick the move whose resulting position scores best. This reached a
respectable R² but played barely better than random. The diagnostic
(`tools/probe.c`) showed why: top-1 agreement with the baseline heuristic was
2.9%, and the network picked deck draws 12% of the time against the
heuristic's 51%, so it stalled, recycling discard piles instead of finishing
games.

**Attempt 2: distil the heuristic's value function.** Regressing `heur_eval`
directly -- a clean, deterministic target -- fit to 4.6 points RMS out of a
58-point spread. The greedy policy on that near-perfect copy still lost by 71
points a game to the heuristic it was copying.

That is the central fact about this game: **candidate moves differ by one or
two points while a finished game's margin has a standard deviation near 60.**
No value function accurate enough to rank moves by one-ply lookahead is
learnable from outcomes, and a search whose leaves use such a value function
inherits the problem -- measured, not assumed: determinized MCTS over the
distilled network scored +0.4 ± 3.1 points against the raw policy it was
seeded from.

**What works: optimise the policy directly.** A policy head predicts the
choice instead of the values behind it, and the value head is demoted to a
baseline, where its errors cancel instead of corrupting the ranking. The
network is initialised by imitating the heuristic, then improved by PPO
self-play: both seats are the same network, moves are sampled from the policy,
and advantages come from lambda-returns against the value baseline. Self-play
generation costs one forward pass per ply, around 3000 games per second on four
cores, so the data is always fresh and on-policy.

Notably, the trained agent stopped stalling on its own. Self-play games fell
from 113 plies to around 57 -- close to the natural length of the game -- once
the policy learned that dragging a game out is not worth what it costs.

## Reproducing

```
# 1. imitate the heuristic to get a sane starting policy
./bin/train --gen heur --gen-switch 99 --iters 4 --games 3000 --steps 12000 \
            --lr 1.5e-3 --tau 0.5 --out data/base.bin

# 2. improve it by self-play
./bin/rl --init data/base.bin --iters 60 --games 5000 --lr 3e-4 --out data/rl1.bin
```

## Playing and measuring

```
./bin/play -a policy:data/best.bin              # play a game yourself
./bin/arena -a policy:data/best.bin -b heur -n 500
./bin/ladder -n 300 random heur policy:data/base.bin policy:data/best.bin
```

Agent specs: `random`, `heur`, `policy:PATH[:temp]`, `net:PATH[:samples]`
(one-ply value greedy), `mcts:PATH[:dets[:sims[:root_width[:node_width]]]]`.

Matches are played over paired deals -- every deal is played twice with the
seats swapped -- so reported margins are not polluted by deal luck.
