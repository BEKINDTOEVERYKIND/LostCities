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
tools/qpair.c       paired rollout Q for any named moves at a replayed position
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
number and the cumulative score difference. Training is staged: early PPO
rewards `margin + win bonus` (the dense margin signal teaches point play),
and the finishing phase switches to `0.05 x margin + 50 x match result`
(--mw / --winbonus in tools/rl.c) so that winning is nearly all that matters.
Being 40 up in round three genuinely changes what the policy optimises:
protect the win rather than maximise expectation, and gamble when behind.

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

**Symmetry.** Nothing in the rules distinguishes the five colors, and the
three wager copies of a suit are identical cards — the state space is 600x
redundant (5! suit relabelings x 3! wager relabelings per suit). This is
exploited at two levels. At decision time, moves that differ only in which
held wager copy they use are compacted everywhere it matters: the rollout
candidate stage folds duplicate wager moves into one (summing their policy
priors, so "play a wager" keeps its full mass in candidate selection and the
duplicates stop wasting candidate slots), and the endgame solver skips the
isomorphic subtrees outright (C and browser both). At training time,
`train --aug 1` presents every sample under a fresh random relabeling of
suits and wager copies (`lc_permute`), forcing the net toward color-invariant
strategy instead of learning each suit slot separately — on small
corrections datasets this multiplies effective coverage by orders of
magnitude. The feature planes themselves remain per-card-id (a relabel-
invariant encoding would invalidate every trained net); augmentation makes
the net invariant to the labels rather than making the labels disappear.

Measured: retrofitting invariance onto a mature net does not pay as a
fine-tune. c10 (the c9 corrections recipe + `--aug 1` as the only change)
collapsed to 30% vs the champion's policy by iteration 5 as the trunk's
label-specific structure was torn up, then recovered to a plateau of only
45-48% by iteration 20 -- the augmentation machinery itself was validated
separately (20k mined samples: permuted targets always legal in permuted
states, all invariants hold), so the shortfall is the size of the
reorganization, not a bug. The right home for augmentation is training
where the net is still plastic: both trainers accept `--aug 1` (`rl.c`
keeps the PPO ratio honest because a from-scratch net stays near-symmetric
throughout, so old and new policy agree across relabelings).

**The from-scratch test confirmed the thesis decisively.** The full
reproduction pipeline rerun with `--aug 1` at every stage (imitation ->
130 PPO iterations -> win-dominated finishing, identical budgets
otherwise) produced `data/sym1.bin`: **61.4% ± 1.7% match wins, +22.0 ±
2.3 points/game over c9** (the previous strongest raw policy) and 63.7%
over the champion's policy at final selection -- the largest single jump
in policy strength in the project, from the same compute the asymmetric
pipeline used. The probe scorecard shows why: long-standing habit leaks
the old nets put ~0% prior on the correct move now carry majority prior
(dead_discard 0.92, stall_habit 0.97, skip_B7 0.999). sym1 is
`data/policy_best.bin`, `data/best.bin` (see the search section: it also
won the search slot 68.3% ± 2.7%), and the browser opponent. Symmetry was
worth roughly what a color-blind observer would guess: five suits' worth
of data for every weight instead of one.

And the corrections loop compounds on the symmetric base: c11 (mining the
new champion's own games -- the stall class had collapsed from thousands
of detections to 46 -- then the proven corrections recipe with `--aug 1`)
fine-tuned with no c10-style collapse, exactly as the symmetry argument
predicts for an already-invariant net, and its best iteration beats sym1's
raw policy **57.6% ± 1.7%, +15.3 ± 1.7 points/game** (400 pairs) with a
healthy probe scorecard. `data/policy_best.bin` = c11 and it powers the
browser opponent -- and c11 swept the search slot too: **55.9% ± 2.5%
match wins, margin +7.4 ± 2.4 per game** over 400 pairs as a search agent
against sym1 under the post-review-fix binary, so `data/best.bin` = c11
across every surface.

Scale asks a different question and got a clear answer: s2 (the identical
from-scratch augmented pipeline with a DOUBLED PPO budget, 260 iterations)
final-selected at **49.1% ± 1.8% vs c11** -- parity with a champion built
from half the self-play compute plus two corrections cycles.  Raw
self-play has hit diminishing returns at this net size; the compounding
loop of mine-your-own-mistakes -> gated labels -> augmented fine-tune is
where strength is still being found.  And it keeps paying: c12 (third
corrections generation, mined from c11's games with the isomorphism-fixed
miner) beats c11 **53.2% ± 1.8%, +7.9 ± 1.7 points/game** (400 pairs).
The champion lineage is old -> sym1 (+63.7%) -> c11 (+57.6%) -> c12
(+53.2%) -> c13 (+52.6%, margin +4.7 ± 1.7), each step over the previous
champion, with per-generation gains shrinking as the mistake classes
empty out; `data/best.bin` = `data/policy_best.bin` = c13.  The loop then
converged on schedule: c14 (fifth cycle, same recipe) confirmed at
**46.6% ± 1.8%** against c13 over 400 pairs -- the detectable mistake
classes are empty, and further cycles just churn.  c13 stands as the
final corrections-loop champion for this architecture; the remaining
gains live elsewhere (wider trunk, richer detectors, human review).
Residual invariance, measured: across 8 random relabelings of real
mid-game states the champion's top-move probability still varies with a
6.6% standard deviation and the argmax flips 27% of the time (near-ties
dominating the flips) -- augmentation got most of the way to invariance,
not all of it.  Test-time symmetrization (averaging the policy over a few
relabelings at the search's candidate stage, negligible cost next to the
playouts) is the natural next experiment for whoever picks this up.

**The trust check, run both ways.** When a reviewer doubted the lineage
("too many horrible plays to believe it improved"), the current champion
was measured directly against the ORIGINAL pre-symmetry champion recovered
from git history: **71.4% ± 1.6% (+42.5 pts/game) as raw policies** and
**68.2% ± 2.3% (+32.2 ± 3.3 pts/game) as full search agents** over 400
paired matches each, with transitive intermediate rungs and a reproduced
sym1 measurement.  The doubted games were real, though: an adversarial
review of the embedded match confirmed three indefensible moves -- one an
artifact of a weakened display config, two exposing the endgame
draw-sequencing weakness that the misorder/deckburn detector classes and
the ov_draw override mode now target.

## Results

All numbers are 3-round paired matches (each triple of deals played twice with
seats swapped) unless stated. Margins are total match points; "wins" are match
wins with draws counting half.

| comparison | margin/match | match wins |
| --- | ---: | ---: |
| win-training continuations vs each predecessor | 63.6% / 61.5% / 57.9% | (500/400/400 pairs) |
| shipped champion vs heuristic | **+173.7 ± 3.6** | **98.0%** (300 pairs) |
| margin-trained champion vs imitation start | +204.8 ± 4.0 | 96.6% (400 pairs) |
| rollout search vs raw policy (margin-trained) | +26.5 ± 4.8 | 63.3% (60 pairs) |
| belief-sampled vs uniform worlds | −3.7 ± 5.4 | 45.8% (60 pairs) |

The shipped model is the *win-trained* one. Training ends with a phase whose
return is `0.05 x margin + 50 x match result`, so winning dominates: a 5%
chance to steal the match outranks a certain narrow loss even at a terrible
expected margin, exactly as competitive play demands. That phase converted
margin into wins -- against the heuristic it gives back ~15 points of margin
relative to the margin-trained champion while winning matches it previously
lost, and it beats that champion head-to-head 63.6% of the time.

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

## When to search, and when the policy alone is enough

Instrumented over 6,697 self-play decisions (tools/searchcmp.c): when the
policy's top move already carries >= 0.95 probability -- 59% of all decisions
-- rollout search disagrees with it only 3-7% of the time, for a mean gain of
0.1-0.2 points; below 0.95 confidence, disagreement is 39-81% and the mean
gain per decision is 1-5 points. Confidence is the dominant variable: the
pattern barely moves across rounds, deck phase, or match closeness (low-
confidence late-deck decisions have the largest tail, up to ~5 points).

The rollout agent therefore takes a gate parameter -- skip the search when the
policy's confidence is already >= the gate (`rollout:NET:worlds:cands:floor:gate`).
Measured (3-round paired matches vs the raw policy):

| configuration | margin/match | match wins | speed |
| --- | ---: | ---: | ---: |
| full rollout, 96 worlds | **+30.0 ± 5.1** | **69.5%** (50 pairs) | 0.8 matches-games/s |
| gate 0.85 (searches ~23% of plies) | +17.1 ± 3.9 | 57.5% (50 pairs) | 1.9 |
| gate 0.95 (searches ~41% of plies) | +14.3 ± 4.6 | 60.5% (50 pairs) | 1.4 |

Head-to-head, the 0.95 gate loses -3.6 ± 4.2 per match to the ungated search
(43.8% over 40 pairs). The lesson cuts both ways: per *decision* the
high-confidence searches look worthless, but there are ~40 of them per match
per side and their 0.15-point slivers add up to most of the gap -- so gating
is a compute trade, not a free lunch.

**The candidate floor cuts both ways.** Candidates come from the policy, and
moves below a 2% prior are pruned -- so when the policy is *certain*, the
"search" has one candidate and can only confirm it, never overrule it. A
replayed position made this concrete: the policy put 100% on a discard, and a
paired re-evaluation (tools/qpair.c, 4000 shared worlds) showed a wager it
had written off was better -- +2.9 ± 0.6 with the net that played the game
(robust to sampled playouts and to search-driven continuations). The leak
family recurs, smaller, in the current champion: in the analogous position
of the embedded game its written-off wager play measures +0.6 to +1.0 over
the 100%-prior discard under three estimators -- real, but below what a
96-world play-time search can resolve, which makes it a training target,
not a search target. But *forcing*
the floor open is worse than the disease: full rollout with `min_cand` 3
scored only **42.8% ± 3.5%** (-10.8 ± 4.2/match, 100 pairs) against the
baseline. A 96-world Q difference carries ±2-4 points of paired noise, most
true gaps between a near-certain policy move and its alternatives are
smaller than that, and taking the argmax of several noisy estimates
systematically flatters the winner. So `min_cand` selects among noise, while
`eval_cand` (the analysis setting) evaluates and *reports* extra candidates
without letting them be selected -- the viewer shows what written-off moves
were worth at zero strength cost.

**Where the search earns its keep: late, not early** (all vs the raw policy,
3-round paired matches):

| search window (plies of each round) | margin/match | match wins |
| --- | ---: | ---: |
| everywhere (150 pairs) | +10.6 ± 3.0 | 51.5% ± 2.9% |
| only plies >= 14 (150 pairs) | **+11.4 ± 2.4** | **56.2% ± 2.9%** |
| only plies < 14 (200 pairs) | +4.6 ± 2.4 | 53.4% ± 2.5% |
| only plies < 14, forced 3 candidates (200 pairs) | -4.7 ± 3.1 | 50.1% ± 2.5% |

Restricting the search to the mid/late round loses nothing -- it matches or
beats searching everywhere while skipping ~30% of the searched plies, and
the direct head-to-head confirms it: late-only vs full search is a dead
heat, +0.6 ± 3.0/match, 49.7% ± 2.9% (150 pairs). Early search contributes
little, and *aggressive* early search (forced candidates) contributes
nothing at all. The mechanism shows up clearly at the opening
ply of the embedded game: three different first moves measure within ±0.5
points of each other at 8000 worlds under three different estimators.
Early-round moves are often near-equivalent in true value, so there is
little for a rollout to find, and its noise can only hurt; late-round
positions diverge sharply and have short, accurately-evaluated horizons.
(An earlier 50-pair run put full search at +30.0 ± 5.1 / 69.5%; the
run-to-run spread between that and the 150-pair number above is itself a
caution about small evaluation batches.)

**The search reports its own noise.** Every reported Q carries the standard
error of its paired difference against the chosen move -- a gap under ~2 of
those is sampling noise, which at 96 worlds means most gaps under ~4-8
points; the analysis dump uses 512 worlds to make the displayed numbers
meaningful. In the final round the dump also reports each candidate's match
win fraction over the playouts (the last round decides the match exactly,
so point EV stops being the objective there). Selecting by that win
fraction is available (`win_q`) but off by default, because it measured no
better than margin selection -- 50.4% ± 0.8% match wins pooled over 2,000
head-to-head pairs (a 300-pair run at 48.0% ± 2.0% and a 1,700-pair
confirmation at 50.8% ± 0.9%), while costing 1.3 ± 0.6 points of margin:
decided finals tie on win%, close finals make a 96-world win fraction a
noisy binomial estimate, and the win-trained policy already carries the
clutch behaviour into every playout. The same lesson as the candidate
floor, from the other direction: at fixed compute, the statistically
efficient objective beats the theoretically right one.

**Dominated discards are pruned by rule** (on by default): with a card in
hand that neither player can ever legally play -- provable from public
expedition tops alone -- discarding any live card with the same draw is a
strictly riskier gift, so those moves leave the candidate set and the
playout argmax. Guards: draw legality (a card cannot go onto the pile
drawn from this turn) and dedup among equally dead cards. Exposure
analysis over an analysed game found the rule could fire on 40% of plies
with zero cases where the dead discard buried a live pile top; the A/B
measured strength-neutral, 50.5% ± 2.0% over 300 pairs, while skipping
work and never gifting a usable card for no reason.

**Expert iteration** (tools/train.c --gen selfrollout, Q-softmax targets
over searched-plus-advisory candidates): twelve iterations from the
champion moved every targeted confidently-wrong prior -- four probe
positions went from 0% prior on the better move to 25-36% -- and flipped
the sequencing watch-probe toward optimal ordering. It did not, however,
produce a stronger agent: 48.2% ± 2.9% search-vs-search against the
champion; the blanket soft targets give up more sharpness than the fixed
leaks return. The refined recipe (corrections only at statistically
significant search-policy disagreements, KL-anchored elsewhere) is the
open training direction.

**The significance-gated override is a measured gain** -- the discipline
blanket forcing lacked. Advisory candidates (eval_cand) may take the move
only when they lead the best policy-plausible candidate by more than
`override_k` paired standard errors, the statistical signature of a
confidently-wrong prior rather than noise. A/B at k=3 with four evaluated
candidates: **+6.35 ± 1.88 per match, 52.5% ± 2.0%** over 300 pairs
against the previous maximum-strength config -- the first strength
improvement since the shipped champion, at ~1.7x search compute.

Expert review of an override-enabled game then exposed two further gates
the SE test needs. (1) `override_min` points (default 4): the SE gate is
world-count-dependent in the wrong direction -- more worlds shrink noise
but sharpen *bias*, so at 512 worlds a 3-SE gate fired on ~1-point stall-
and discard-flavoured playout bias; in the reviewed game every override
gap over 4 points was one the reviewer endorsed and every graded blunder
was under 2.5. (2) Sampled confirmation: the surviving gap must also hold
at half the floor under stochastically-sampled continuations, because
deterministic playouts repeat knife-edge downstream decisions across all
paired worlds -- one position produced a +5.0 ± 0.14 argmax gap for
discarding over a free scoring play that sampling collapsed to +0.6.
Regenerating the reviewed game under the full gates removed every
reviewer-graded blunder while keeping the overrides the reviewer agreed
with.

The corrections loop (mine -> gate-respecting labels -> anchored fine-tune,
tools/mine.c) produced its first net gain: c9 beat the then-champion 52-55%
policy-vs-policy with every reviewer-flagged prior improved on the held-out
probe suite, while search-vs-search pooled to exact parity (49.6% ± 1.8%,
400 pairs) -- the search already extracted what the policy fix supplied.
For a while the nets split by surface (best.bin for search, c9 for raw
policy); the symmetric from-scratch net ended the split.  sym1 swept both
slots: policy-vs-policy 61.4% ± 1.7% over c9, and search-vs-search
**68.3% ± 2.7%, +32.9 ± 3.4 points/game** over the old champion (150
pairs, full gates) -- unlike c9's, its gains AMPLIFY under search, which
means the value and belief heads improved along with the policy.
data/best.bin and data/policy_best.bin are both sym1 now.

Sampled playouts (playout_sample, spec field 14) A/B'd against argmax at
the full config: 49.4% ± 2.0% over 300 pairs -- a tie.  Unbiased
continuations cost nothing in strength, so analysis and training labels
use them; match play keeps argmax with the sampled confirmation gate.

Exact endgame solving (solve_deck, spec field 15) A/B'd at deck <= 2
against the full config: **48.7% ± 2.9% match wins, margin +1.98 ± 2.13
per game** over 150 pairs (6 seeds pooled).  The signature is telling:
the solver banks a couple of extra points per game in terminal positions
but they do not convert to match wins -- the gated 96-world search already
avoids the endgame blunders that matter, so at match level the feature is
neutral and stays OUT of the strength spec.  It stays where exactness is
the point: analysis labels (se = 0 ground truth near the deck's end) and
the browser opponent, whose raw policy otherwise has no search to save it
from last-card arithmetic errors.  Each solve_deck decision is bounded by
one 4M-node budget across all moves and worlds, falling back to normal
search on exhaustion -- an unbounded per-solve version could pin a thread
for hours on one rare wide-hand endgame.

Recommended settings: **maximum strength**
`rollout:NET:96:5:0.02:0:1:14:0:4:0:1:3` (search from ply 14, four
candidates evaluated, dominated discards pruned, 3-SE advisory override);
**gate 0.85 for real-time play**; raw policy for bulk generation.
Analysis uses `rollout:NET:512:5:0.02:0:1:0:0:4:0:1:3` -- the same
selection rules at 512 worlds, searched at every ply for display.

## Reproducing

```
# 1. imitation start (heuristic plays the rounds; ~15 min on 4 cores)
./bin/train --gen heur --gen-switch 99 --rounds 3 --iters 4 --games 2500 \
            --steps 15000 --batch 512 --lr 1e-3 --tau 0.5 --out data/m0.bin

# 2. PPO over full matches with belief learning (~2 h on 4 cores)
./bin/rl --init data/m0.bin --ref policy:data/m0.bin --rounds 3 --winbonus 15 \
         --iters 130 --games 900 --epochs 1 --lr 2.5e-4 --ent 0.003 --out data/m1.bin

# 3. finishing phase: win-dominated reward (~40 min)
./bin/rl --init <peak of step 2> --ref policy:<same> --rounds 3 \
         --winbonus 50 --mw 0.05 --iters 80 --games 900 --epochs 1 \
         --lr 1.5e-4 --ent 0.002 --lambda 0.9 --out data/w1.bin
# then select the checkpoint by MATCH WIN RATE over a 500-pair validation
```

## Playing, analysing, measuring

```
./bin/play -a rollout:data/best.bin:128:4          # play against the agent
./bin/showgame -a policy:data/best.bin -r 3        # full match transcript
python3 tools/verify_transcript.py <transcript>    # independent rules audit
./bin/analyze -a rollout:data/best.bin:512:5:0.02:0:1:0:0:4 -r 3 > data/analysis.json
./bin/arena -a policy:data/best.bin -b heur -n 300 -r 3
python3 tools/referee.py match NETA NETB --pairs 400 --rounds 3
# what was move X worth at ply N of an analysed game? (paired, with SE)
./bin/qpair -n data/best.bin -s SEED -f moves.txt -p N -w 4000 \
            -c "Y2 d deck" -c "W4 p deck"
```

The analysis console (`web/viewer.html`, embedded game included) replays a
match ply by ply: board, both hands (marked where publicly known), the policy
distribution, rollout Q values per candidate, the network's belief about the
opponent's hidden hand next to the omniscient truth, and the value trajectory
across all three rounds.

Agent specs: `random`, `heur`, `policy:PATH[:temp]`,
`rollout:PATH[:worlds[:cands[:floor[:gate[:minc[:plylo[:plyhi[:evalc[:winq]]]]]]]]]`
(strongest), `rolloutu:...` (uniform-world
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
* tools/blunders.py tallies outcome-level events -- expeditions that finished
  negative, wagered expeditions that finished deep negative, discards the
  opponent took at once. These are *style statistics*, not error rates: under
  optimal play every one of them is non-zero (a good gamble that fails still
  shows up in the tally), and with no optimal-play reference there is no
  "correct" value to compare against. They are only useful for watching style
  drift between versions (e.g. the agent hands the opponent far fewer
  immediately-useful discards than the heuristic, 2.5 vs 17.9 per match), and
  say nothing about whether any individual count is too high.
* Win-focused continuation training converged after three rounds: a fourth
  continuation stayed flat at 50-52% against its predecessor through 63
  iterations and was abandoned. Further gains likely need a bigger change
  (deeper search at training time, a larger trunk, or an opponent pool)
  rather than more of the same recipe.
