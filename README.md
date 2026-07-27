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
src/rollout.c       policy improvement by paired playouts in sampled worlds
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
tools/eval_ladder.sh, tools/eval_final.sh   reproduce the tables below
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
from 113 plies to around 48 -- close to the shortest the deck allows -- once the
policy worked out that dragging a game out mostly helps whoever is behind.
Nothing in the training told it that; the ply cap in the engine exists only to
stop degenerate early policies from looping forever, and trained self-play never
approaches it.

Three smaller stages follow the first PPO run, each worth a few points and each
kept only after beating its own starting point over hundreds of paired deals: a
second PPO phase at a lower learning rate (+6.0), a round of distilling the
rollout operator below back into the policy head (+2.3), and a third PPO phase
(+2.2). Distillation flattened out after about two rounds -- the operator's edge
is around +6 points and only a fraction of it survives being compressed into a
single forward pass, which is why it stays worth running the search at play
time.

## Results

Every number below is from paired deals -- each deal played twice with the
seats swapped -- so deal luck cancels. Percentages are the share of games won
(draws count a half); margins are mean points per game.

| agent | Elo | margin vs heuristic (points/game) |
| --- | ---: | ---: |
| `policy:best` (one forward pass per move) | **1366** | **+97.4 ± 1.8** (97.2% of games) |
| `policy:rl2` (after PPO phase 2) | 1327 | +87.1 ± 2.0 |
| `policy:rl1` (after PPO phase 1) | 1299 | +88.7 ± 1.9 |
| `heur` (hand-crafted projection evaluation) | 820 | 0 |
| `policy:base` (imitation of the heuristic) | 792 | −9.2 ± 2.4 |
| `random` | 0 | −139.5 ± 1.7 |

(The margin column is not monotone between `rl1` and `rl2` -- their scores
against a much weaker third party are noisy and not the right comparison. The
head-to-head result below is: `rl2` beats `rl1` by +6.0 ± 1.3.)

Elo comes from a round robin of 400 paired deals per pairing, fitted by
Bradley-Terry and anchored at random play. The rollout agent is not in this
table: a rollout move costs thousands of forward passes, so it is measured
separately below rather than in a full round robin. The final policy is about
**545 Elo above the hand-crafted heuristic**, winning 97% of games against it by
an average of 97 points.

Each stage also beats the one before it head to head, which is the check that
matters -- an agent can drift away from a fixed baseline without getting better:

```
                  margin of the later stage over the earlier one
base -> rl1            +91.7 +- 2.3    (imitation -> PPO phase 1)
rl1  -> rl2             +6.0 +- 1.3    (PPO phase 2)
rl2  -> best            +6.6 +- 1.1    (rollout distillation, then PPO phase 3)
```

**Search on top.** Rollout search over the trained policy adds a further
**+5.5 ± 1.9 points a game** (56.7% of games) for about 30 ms of thinking per
move, at 128 sampled worlds and 4 candidates.

**Against the classical approach.** The standard recipe for imperfect
information card games is a hand-crafted evaluation plus perfect-information
Monte Carlo over sampled worlds (`hrollout`). Here it is worth measuring
because it isolates what the learning bought:

```
hrollout      vs heur          -5.3 +- 5.4    (PIMC over the heuristic: no gain)
policy:best   vs hrollout     +93.4 +- 4.9    (96.7% of games)
```

Sampling worlds does not rescue a weak rollout policy -- the heuristic's
playouts stall, so the estimates it produces describe stalling, not good play.
The same operator over the trained policy is worth +5.5. The rollout is only as
good as the policy driving it.

## Reproducing

```
# 1. imitate the heuristic, to start from something that plays legally and
#    sensibly instead of from noise
./bin/train --gen heur --gen-switch 99 --iters 4 --games 3000 --steps 12000 \
            --lr 1.5e-3 --tau 0.5 --out data/base.bin

# 2. PPO self-play: this is where nearly all the strength comes from
./bin/rl --init data/base.bin --iters 60 --games 5000 --lr 3e-4 --ent 0.004 \
         --eval 400 --out data/rl1.bin

# 3. a second, gentler PPO phase, measured against the frozen phase-1 model
cp data/rl1.bin.it60 data/ref1.bin
./bin/rl --init data/ref1.bin --ref policy:data/ref1.bin --iters 80 --games 6000 \
         --lr 2e-4 --ent 0.006 --out data/rl2.bin

# 4. distil the rollout operator back into the policy head
cp data/rl2.bin data/ref2.bin
./bin/train --init data/ref2.bin --gen selfrollout:48:4 --iters 4 --games 1200 \
            --steps 3000 --lr 2e-4 --tau 3.0 --lambda 0.7 \
            --ref policy:data/ref2.bin --out data/ex1.bin
```

Total training time was about two hours on four CPU cores. `tools/eval_ladder.sh`
and `tools/eval_final.sh` reproduce the tables above.

## Playing and measuring

```
./bin/play -a policy:data/best.bin                     # play a game yourself
./bin/play -a rollout:data/best.bin:128:4              # against the strong one
./bin/arena -a policy:data/best.bin -b heur -n 500
./bin/ladder -n 300 random heur policy:data/base.bin policy:data/best.bin
./bin/bench                                            # throughput of each path
./bin/probe -n data/best.bin -p heur                   # value head diagnostics
```

Agent specs:

| spec | what it does |
| --- | --- |
| `random` | uniform legal move |
| `heur` | one-ply greedy on the hand-crafted projection evaluation |
| `policy:PATH[:temp]` | policy head, argmax (or sampled at `temp` > 0) |
| `net:PATH[:samples]` | one-ply greedy on the value head (kept for comparison; weak, see above) |
| `rollout:PATH[:worlds[:candidates[:floor]]]` | policy plus paired playouts -- the strongest configuration |
| `hrollout[:worlds[:candidates]]` | heuristic plus perfect-information Monte Carlo, no network |
| `mcts:PATH[:dets[:sims[:rw[:nw]]]]` | determinized PUCT search with network priors and values |

`data/best.bin` (620 KB) is the trained network and is checked in, so the
agent is playable straight after `make` with no training run.

Matches are played over paired deals -- every deal is played twice with the
seats swapped -- so reported margins are not polluted by deal luck.

## Honest limits

* There is no public Lost Cities benchmark bot or human game corpus to test
  against offline, so "strong" here means: 545 Elo over a competent
  hand-crafted player, 97% of games against the classical
  heuristic-plus-Monte-Carlo approach, and monotone head-to-head improvement
  across every training stage. It is not a measurement against known human
  experts.
* Training is pure self-play after the imitation start, so the policy is tuned
  to an equilibrium against itself. Its large margins against three quite
  different opponents (random, heuristic, PIMC) are evidence that it did not
  merely overfit to its own quirks, but a genuinely alien opponent could still
  find something.
* One round is scored, not the three-round match the boxed game describes.
  Maximising expected margin per round is the right building block for that,
  since match totals add up, but no across-round risk management is modelled.
