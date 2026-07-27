# Lost Cities AI

A from-scratch Lost Cities engine, neural network, and self-play training
pipeline for the full competitive game — three-round matches with cumulative
scoring — written in C with no external dependencies.

Lost Cities (Reiner Knizia) is a two-player imperfect-information card game.
Five suits of twelve cards (three wagers and the numbers 2-10). Each turn you
play a card to one of your own expeditions or discard it, then draw from the
deck or from the top of any discard pile (never the pile you just discarded
to). Expeditions must ascend; wagers must come before numbers. An expedition
scores `(sum of numbers - 20) x (1 + wagers)`, plus 20 if it holds eight or
more cards, and only if you opened it at all. A round ends when the deck runs
out; a competitive match is three rounds, totals win, and the first player
alternates by round.

## Layout

```
src/lc.[ch]         rules engine: state, move generation, scoring, match context
src/features.[ch]   information-set encoding (556 inputs, sparse + dense)
src/net.[ch]        three-headed network, forward/backward, Adam, save/load
src/heuristic.[ch]  hand-crafted projection evaluation (baseline, bootstrap)
src/search.[ch]     determinized MCTS with network priors and values
src/rollout.c       rollout policy improvement over belief-sampled worlds
src/agent.[ch]      move-selection policies; belief-weighted determinization
src/match.[ch]      paired-deal match runner (single rounds or full matches)
src/spec.[ch]       agent command-line specs
tools/rl.c          PPO self-play trainer over full matches  <- the main trainer
tools/train.c       imitation / expert-iteration trainer (+ dataset dump/load)
tools/arena.c       head-to-head matches with error bars (-r 3 for full matches)
tools/ladder.c      round robin with fitted Elo
tools/analyze.c     per-ply JSON dump: state, values, policy, search, beliefs
tools/referee.py    numpy port of engine+net, verified to ~1e-6 against the C
tools/dumpfeat.c    parity reference dumper for the referee
tools/verify_transcript.py  independent replay/audit of printed transcripts
tools/showgame.c    replayable match transcripts, re-scored independently
tools/play.c        play against the agent in a terminal
web/viewer.html     self-contained analysis console (published as an artifact)
tests/test_engine.c rule, information, and match invariant tests
```

Build and test: `make && make test`.

## What the agent knows and how it decides

**Information tracking.** The state records, beyond the public board:

* the deck count (a direct network input, with endgame flags),
* *known cards*: every card taken from a discard pile is drawn face up, so
  until it is played again the opponent provably holds it. The engine tracks
  this both ways, the encoder exposes both planes ("cards I know they hold",
  "cards of mine they know about"), and the world-sampler treats known cards
  as certainties, never as unknowns.

**Learned opponent inference.** A third network head predicts, for every card
whose location the player cannot pin down, the probability that the opponent
holds it. It is trained on self-play states where the true opponent hand is
known — every position is a free supervised example — so it learns behavioural
inference from the same data that trains the policy: an opponent who opened
Yellow with a wager is Yellow-heavy; one who discarded a low Blue early is
unlikely to be building Blue; a card that has sat untaken in a pile through
many of their turns is a card they do not want. The determinized search then
samples opponent hands from this posterior (Gumbel-top-k over the belief
logits) instead of uniformly, so every rollout plays against plausible
opponents rather than random ones. The same inference reaches the raw policy
implicitly: the policy/value heads share the trunk with the belief head and
see all the same signals.

**Match play, not just round play.** The network's inputs include the round
number and the cumulative score difference, and PPO's terminal reward is the
match margin *plus a win bonus*, so being 40 up in round three genuinely
changes what the policy should (and does) optimise: protect the win rather
than maximise expectation, and gamble when behind.

**Stalling.** Drawing a useless card from a pile to deny the opponent a turn
of deck progress is in the action space, and nothing hand-crafted decides it:
the policy learns from match outcomes when a stall is worth more than the
tempo it gives away. The engine's only concession is a 300-ply safety cap per
round (real rules allow unbounded mutual pile-recycling), which sane play
never approaches.

## Why the architecture looks like this

The value-only approach fails measurably in this game: candidate moves differ
by one or two points while a finished round's margin has a standard deviation
near 60, so no value function learnable from outcomes can rank moves by
one-ply lookahead — a near-perfect distillation of the hand-crafted
evaluation (4.6 pts RMS) still lost by 71 points a game to the evaluation it
copied, and search built on that value function was no stronger than its own
prior. What works is predicting decisions directly: a factored policy head
(one logit per card-and-disposition plus one per draw source, softmax over
legal moves), with the value head serving as a PPO baseline where its errors
cancel, and search done by *rollouts*: play each candidate move out to the end
of the round with the policy in belief-sampled worlds, sharing worlds across
candidates so the comparison is paired.

Training is: imitate the heuristic for a sane start (it knows nothing about
match context or beliefs), then PPO over full three-round matches with the
belief head learning on the side. The trunk is 556 -> 512 -> 256.

## Results

All numbers are 3-round paired matches (each triple of deals played twice with
seats swapped) unless stated. Margins are total match points; "wins" are match
wins with draws counting half.

| comparison | margin/match | match wins |
| --- | ---: | ---: |
| final policy vs imitation start | **+204.8 ± 4.0** | **96.6%** (400 pairs) |
| final policy vs heuristic | **+159.8 ± 4.1** | **95.5%** (300 pairs) |
| rollout search vs raw policy | **+26.5 ± 4.8** | **63.3%** (60 pairs) |
| belief-sampled vs uniform worlds | −3.7 ± 5.4 | 45.8% (60 pairs) |

The training trajectory (evaluated vs the frozen imitation start every 3
iterations): the PPO run climbs from parity to a peak of ~+205/match around
iteration 51, then over-optimises into stall-heavy play and falls back to
+66 by iteration 130. The shipped model is the peak checkpoint, selected by a
5-way 300-pair tournament and confirmed head-to-head against its neighbours
(+5.5 ± 1.9 over iteration 45, +2.9 ± 1.8 over iteration 48). Checkpoint
selection matters: the *last* iterate of a PPO run is not the best one.

The belief-sampling ablation is a null result: the rollout agent is no
stronger (and no weaker, within noise) when its imagined worlds come from the
learned posterior instead of uniform sampling. The likely reason is that the
policy driving the playouts shares its trunk with the belief head, so the same
inference already shapes every playout decision; making the sampled hands more
realistic adds little on top. The head still earns its place through what it
demonstrably knows:

**Belief quality** (tools/belief_quality.py over the embedded analysis game):
AUC 0.734 for "does the opponent hold this card", with calibration close to
the diagonal (predicted 29% -> observed 28%, predicted 88% -> observed 93%).
In the analysis console you can watch it work: mid-game, most of its top-ranked
cards really are in the opponent's hand.

## Reproducing

```
# 1. imitation start (heuristic plays the rounds; ~15 min on 4 cores)
./bin/train --gen heur --gen-switch 99 --rounds 3 --iters 4 --games 2500 \
            --steps 15000 --batch 512 --lr 1e-3 --tau 0.5 --out data/m0.bin

# 2. PPO over full matches with belief learning (~2 h on 4 cores)
./bin/rl --init data/m0.bin --ref policy:data/m0.bin --rounds 3 --winbonus 15 \
         --iters 130 --games 900 --epochs 1 --lr 2.5e-4 --ent 0.003 --out data/m1.bin
```

## Playing, analysing, measuring

```
./bin/play -a rollout:data/best.bin:128:4          # play against the agent
./bin/showgame -a policy:data/best.bin -r 3        # full match transcript
python3 tools/verify_transcript.py <transcript>    # independent rules audit
./bin/analyze -a rollout:data/best.bin:96:5 -r 3 > data/analysis.json
./bin/arena -a policy:data/best.bin -b heur -n 300 -r 3
python3 tools/referee.py match NETA NETB --pairs 400 --rounds 3
```

The analysis console (`web/viewer.html`, embedded game included) replays a
match ply by ply: board, both hands (marked where publicly known), the policy
distribution, rollout Q values per candidate, the network's belief about the
opponent's hidden hand next to the omniscient truth, and the value trajectory
across all three rounds.

Agent specs: `random`, `heur`, `policy:PATH[:temp]`,
`rollout:PATH[:worlds[:cands]]` (strongest), `rolloutu:...` (uniform-world
ablation of the belief sampler), `mcts:PATH[...]`, `net:PATH` (kept as the
negative result it is).

All matches are paired: every deal (all three of them, in match mode) is
played twice with the seats swapped, so deal luck cancels.

## Honest limits

* There is no public Lost Cities benchmark bot or human game corpus to
  measure against offline; strength claims are relative (baselines, earlier
  stages, ablations), not against known human experts.
* The belief head conditions on the current information set, which carries
  most but not all behavioural evidence (the exact order of past actions is
  not encoded).
* Training is pure self-play after the imitation start; margins over
  qualitatively different opponents argue against self-overfitting, but a
  genuinely alien style could still find something.
