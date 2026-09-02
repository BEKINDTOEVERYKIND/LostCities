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
the ov_draw override mode now target.  Measured: the ov_draw override
(same-action draw variants through relaxed gates) scored **47.2% ± 2.5%,
margin -2.7 ± 2.4** over 400 pairs -- rejected.  The sampled-confirmation
gate's habit-pricing is on net protective: loosening it admits more
determinism-bias artifacts than real endgame edges.  The sequencing class
is a POLICY problem, addressed by the detector-driven corrections cycle,
not a gate problem.  That cycle also reported honestly: c15 (33,735
samples mined with the new detectors -- deckburn fired ~8x/game) confirmed
at **47.2% ± 1.8%** vs c13, no promotion.  The likely mechanism closes the
loop on the estimator schism: the labeler's own sampled-confirmation gate
sides with the deck-burn in most flagged states (it prices the policy's
habits), so the corpus cannot teach the pile-extension it was mined to
teach.  Fixing the sequencing class needs a label source that understands
turn arithmetic -- deeper exact solving or a larger net -- not more of the
same recipe.  c13 stands as champion.

**The exact-label test settles it.** The "deeper exact solving" branch was
then built and measured.  A transposition table went into the endgame
solver (2.5-3x fewer nodes, verified byte-identical values against the
table-off solver), and a one-root-solve-per-world voting mode
(`lc_solve_root` + `solve_vote`) made exact labeling affordable where
per-move exact averaging is not: one alpha-beta from the root prices a
whole belief world (6.5-31M nodes at deck 5) because rising alpha refutes
bad moves cheaply, where full-window per-move solving costs 12-55M nodes
PER MOVE.  corr10 (33,624 samples, 320 games, deck<=5 plies labeled by
solver vote instead of gated search) trained c16: every fine-tune
iteration evaluated below parity, and the best iteration confirmed at
**42.7% ± 2.9%** vs c13 over 300 paired games (margin -12.9) -- stopped
early, no promotion.  Two independent label sources -- the gated search
(c15) and the exact solver (c16) -- have now failed the same way, which
acquits the labels and convicts the recipe: fine-tuning on one-hot
targets drawn only from detector-flagged positions degrades global play
(the flagged distribution is not the play distribution, and the
interference cost exceeds the endgame gain; the exact slice, ~10% of the
corpus, cannot outvote it).  The corrections loop is closed for this
architecture.  Exact endgame strength, if it matters, belongs at
DECISION time (`solve_deck`, measured match-neutral) or inside ordinary
self-play training data -- not in flagged-corpus fine-tunes.  The
remaining strength levers are a wider trunk (needs dynamic net sizing:
the C structs are compile-time sized, the JS side is already dynamic)
and test-time symmetrization at the candidate stage.  c13 stands as
champion.

**The wider trunk, measured.** Runtime net sizing was then built (the
weight-file header always carried h1/h2; the Net struct became a shell
over one contiguous block in the old layout, so a 512x256 file
round-trips byte-identically and an arena chunk replays game-for-game --
35% faster, the block being kinder to the cache), and the full sym1
pipeline (imitation -> 130 PPO -> 80 win-finishing, `--aug 1` everywhere,
identical budgets) was rerun at 1024x512 -- 2.56x the weights.  The wide
net trained normally (87-93% vs its own imitation start through PPO,
finishing added its usual match-win reshaping), and its best checkpoint
(`data/big1.bin`) confirmed at **41.8%** vs c13 as a raw policy (300
pairs) and **46.3% ± 1.8%** (margin -15.2) as a full search agent over
800 games.  No promotion; c13 stands.  Two readings, both recorded: the
narrow one is that width alone at a FIXED training budget loses to four
generations of corrections -- 2.56x capacity fed the same 130+80
iterations is plausibly undertrained, and resuming b1's PPO for another
130+ iterations is the cheap follow-up this leaves on the table.  The
interesting one is the config split: as a search agent the wide net sits
5 points closer to c13 than its raw policy does, i.e. its value/belief
trunk carries relatively more of its strength -- a hybrid agent (champion
policy priors, wide-net playout evaluation) is the other recorded
follow-up, pending an Agent that can carry two nets.  Either way the
sizing infrastructure is permanent: any width now loads, trains, and
plays from the same binaries.  The undertraining reading was then tested
directly: 130 MORE PPO iterations on the wide net (260 total, twice the
recipe's budget) moved its vs-champion-policy probe from 37.4% to 39.2%
-- statistically flat across six late checkpoints (34-40%, SE ~2.4 each).
Doubling the training did not unlock the doubled capacity; width scaling
under this recipe is closed, matching the s2 result one width down.
The inference-side signal was then isolated with a hybrid agent
(`rollouth:MAIN:BELIEF` -- MAIN keeps priors, candidates and playouts,
BELIEF's head alone steers world sampling): champion policy + wide-net
beliefs measured **49.7% ± 1.8%** (margin -6.5) against the plain
champion over 800 games.  Parity, not gain -- the config split was the
search structure flattering the wide net, not a superior belief head.
The wide-trunk direction is fully closed: policy 41.8%, search 46.3%,
doubled PPO flat, beliefs 49.7%.  The rollouth machinery stays (any two
nets can now split roles in one agent).  c13 stands as champion.

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
`rollouth:data/best.bin:data/belief_best.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1:0:0:0:0:0:2:0:120:0:120`
(search from ply 14, four candidates evaluated, dominated discards
pruned, 3-SE advisory override, 1-SE selection gate, belief specialist
steering world sampling -- adopted at 52.31% over 800 games, see the
belief-deployment section -- no draw-variant expansion (search evaluates
only policy-ranked moves, adopted on the reviewer's directive at
50.44%/800 games, 1.47x cheaper), and the policy prior and value
symmetrized EXACTLY over all 120 suit relabelings before candidates
form (`sym_k`, field 25 -- 120 or more enumerates the permutations,
smaller values sample that many: first adopted at K=8 on record suite
390/151 and 51.00%/800 games under the reviewer's standing rule that
the agent must respect the rules' symmetries wherever that is free,
then made exact when `tools/symres` showed the K=8 average still let
the top candidate depend on the relabeling draw on 10% of states and
moved the gate value by ~1 point; exact costs ~3 ms per decision), and
the belief logits that sample opponent hands symmetrized the same way
(`sym_bel`, field 27: inference skill up in every phase, cached per
decision so it is slightly cheaper than the raw path; the three wager
copies of a suit are additionally pooled exactly, since relabelings
only average copy symmetry -- a reviewer caught two Yx copies
displayed 5 points apart));  the analysis dump and viewer show these
symmetrized beliefs, priors and values from the deciding agent, not
the raw head;
plain `rollout:NET:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1:0:0:0:0:0:2:0:120:0:120`
when only one network file is on hand;
**review games** are played at full strength, cost no object:
`rollouth:data/best.bin:data/belief_best.bin:512:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1:0:0:0:0:0:2:0:120:0:120`
(the same rules at 512 worlds);
**gate 0.85 for real-time play**; raw policy for bulk generation.
Analysis uses `rollout:NET:512:5:0.02:0:1:0:0:4:0:1:3` -- the same
selection rules at 512 worlds, searched at every ply for display.
Showcase/review games are PLAYED at full strength, cost no object
(reviewer doctrine): the recommended spec with dets raised to 512 --
`rollouth:data/best.bin:data/belief_best.bin:512:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1:0:0:0:0:0:2`
-- so a reviewed blunder is a real policy/search failure, never the
96-world sampling noise the wall-clock tournament spec accepts.

Field 17 (`sel_k`) is the selection gate, added after a reviewer caught
the showcase game gifting a B wager to an opponent holding a wager-only
B expedition.  The eligible-candidate argmax had no noise protection:
the two dead wager copies dedup-folded into a 3.7% prior that crossed
the 2% floor, and at 96 worlds a candidate measuring **3.13 ± 0.52
points worse** at 4000 worlds still won the argmax in ~5% of seeds --
the game hit that tail.  With the gate a non-top candidate must lead
the policy top by `sel_k` paired SEs or the top plays.  Replays at the
caught position: the gift drops to 0/40 seeds while the position's
genuinely better override (the safe wager discard, +2.61 ± 0.24) still
plays.  Measured head-to-head at k=1: **53.8% ± 1.8% match wins over
800 games** against the ungated spec (margin -6.7: it wins more matches
on fewer aggregate points -- the signature of removing rare
catastrophes at a small cost in average sharpness, and match wins are
the objective).  k=1.5 measured 52.0% ± 1.8% over its own 800 games --
also above parity, but the stricter gate suppresses more of the good
overrides than it saves in blunders; k=1 stands.

A c20-era reviewer catch (showcase ply 16: a 76%-prior, deep-search-
best play G8 lost the move to a noise qualifier on one 96-world
stream, a ~5%-of-seeds tail) prompted two attempts to close the sel_k
gate's residual noise, both refuted by the probe suite before any
match gate: a fresh-batch confirmation veto suppressed twelve
reviewer-verified good overrides to remove six noise cases (the k=1.5
signature -- real overrides are mostly SMALL true leads that fail an
independent margin retest nearly as often as flukes), and pooled
contested-ply deepening (sel_deep, field 24, kept as an experimental
mode) moved the suite from 341/148 to 342/160.  The single-batch k=1
gate stands as the measured optimum of five variants; the tail is the
price of the small-lead overrides.  What shipped instead is audit
honesty: `analyze -p` now records the play agent's own search
(`playsearch`) and the viewer displays it as the decision record
beside the deeper advisory analysis -- the reviewer's confusion came
from reading a 512-world advisory table as if it had chosen the move.
Full protocol in `data/probes/selconf_gate_2026-08-29.txt`.

**The solver-vote endgame, measured -- and a divergence worth naming.**
A reviewer's stall catches were refuted by EXACT solving (8/8 sampled
worlds at three deck-4/5 positions say draw from the deck) while
4000-world playouts scored the stalls BETTER by up to +9.2: the playout
policy stalls on both seats, so round extensions price its own habit as
profit.  Decision-time fix attempt: at deck<=5, one root alpha-beta per
belief world and a majority vote (spec fields 15/18/19).  On the probe
suite it does exactly what it promises -- the two solver-covered stall
probes flip from 1/20 and 0/20 correct to 18/20 and 20/20.  In the
400-pair match gate it scores **47.1% ± 1.8%** (margin -12.3): NO
adoption.  The lesson is the divergence itself: winning the flagged
positions does not win matches when the aggregation is variance-blind
-- a majority of per-world optima ignores margin magnitude, so a move
that is +1 in five worlds outvotes one that is +40 in four, and the
carried round margins pay for it.  Next: the same solver through the
win-aware AVERAGING block (per-move exact values, lexicographic
wins-then-margin) at a real budget -- the configuration E14 tested only
at starved 4M budgets before the transposition table existed.

That rematch was run: solvedeck 5 through the averaging block at a 60M
node budget (spec tail `:5:0:1:0:60`), 400 pairs / 800 games against
the adopted spec.  Result **49.2% ± 3.5%** (95% CI), point margin +1.0:
measured-neutral.  The magnitude-aware aggregation fixes the vote's
pathology (it recovers the ~6-point match loss the vote suffered) and
still flips the solver-covered stall probes, but converts none of that
into match wins -- by deck<=5 the games it changes are almost all
already decided, and each endgame decision costs minutes of solving.
The adopted fast spec stands.  Verdict for the flagged-stall class:
decision-time exactness is the wrong lever at this depth; what remains
of the stall flaw lives at deck 6-8, outside affordable solver reach,
and is a policy-training problem.

**Doubled world sampling (dets 192), measured.**  The other open lever:
2x belief-world samples per decision (192 vs the adopted 96), targeting
the small-edge class -- reviewer-flagged decisions where the true gap
is ~1-3 points and 96 worlds leaves it under the noise gate.  800 games
against the adopted spec: **51.4% ± 3.5%** (95% CI), point margin
**+4.7**.  Short of the 52% adoption gate, so the 96-world spec stands
as the recommendation, but the signature differs from a null result:
the point margin was +4 to +8 in every interim read (E14TT's collapsed
to ~0), i.e. 192 worlds plays measurably sharper per point and doubles
decision time, yet converts little of it into match wins -- consistent
with the sel_k noise gate (calibrated at 96 worlds' SE) absorbing most
of the extra resolution.  A sel_k recalibration at 192 is the natural
follow-up if the time budget ever allows a slow spec.

**Stall-shaped policy training (`rl --stallpen`), measured.**  The
training-side attack on the flagged stall class: a soft penalty (0.8
pts) subtracted from the RL return whenever the acting player draws
from a pile at deck<=8, folded into the standard finishing-phase
fine-tune from best.bin.  Behaviorally it works: the reviewer-probe
suite's flagged-worse rate collapses from 34% to 13% of decisions
(flagged-better rises 42%->44%), and the exact-solver-refuted stall at
probe 91 flips to a deck draw 11/12.  At the 800-game match gate under
the full search spec: **50.3% ± 3.5%**, margin -3.2 -- measured-neutral,
best.bin stands.  The instructive part is the suite/match divergence in
the OTHER direction from solver-vote: fixing the flagged behaviors did
not win matches, which says the search layer was already absorbing most
of the stall cost at decision time -- the remaining flagged-worse
choices are real but cheap.  The flag stays in the trainer (off by
default) as the template for shaping other flagged classes.

**Gift-shaped policy training (`rl --giftpen`), measured.**  The same
template applied to the second flagged class: a soft penalty (0.8 pts)
for discarding a wager while the opponent has no number cards down in
that suit and could still play it -- soft because the reviewer is
explicit that such discards are sometimes correct.  The fine-tune posts
the best probe-suite score of any net so far (47% flagged-better / 14%
flagged-worse; the adopted baseline is 42%/34%) with raw-policy evals
at parity.  Match gate: **50.8% ± 3.5%** over 800 games, margin +1.6 --
measured-neutral again.  Two shaped classes now show the same pattern:
the flagged behaviors are trainable away at zero match cost, and
neither buys wins because the search layer already prices most of the
damage out at decision time.  Follow-up: one combined run
(`--stallpen 0.8 --giftpen 0.8`) to fold both corrections into a single
net -- at parity, the cleaner-behaving net is the better artifact, and
it is a candidate for adoption on non-regression grounds.

**The combined run, measured -- and the adoption decision.**  Stacking
both penalties (`--stallpen 0.8 --giftpen 0.8`) over-shaped: the suite's
flagged-better rate fell to 41% (below even the unshaped baseline's
42%), one previously-passing probe regressed, and the match gate read
**49.3% ± 3.5%** (margin +0.5) -- not adopted.  One class of shaping at
a time is what the recipe tolerates.

Final call: **best.bin is now the gift-shaped net** (the previous one
is preserved as `data/best_pre_gp.bin`).  This is a behavior-quality
adoption, documented as such: at match level the two nets are
statistically indistinguishable (50.8% ± 3.5 over 800 games), and on
the reviewer-flagged decision suite the gift-shaped net strictly
dominates (47% flagged-better / 14% flagged-worse vs 42%/34%),
including the exact-solver-confirmed stall at probe 91.  Where the
match objective is a wash, the net that makes fewer of the flagged
blunders is the better artifact -- and the better opponent against a
field that punishes those blunders harder than self-play does.  The
recommended spec string is UNCHANGED; only the contents of
`data/best.bin` changed.

**Prior-aware override thresholds (spec fields 20-21), calibrated and
measured.**  The idea: how much search EV it takes to overrule the
policy should scale with how lopsided the policy is.  A 4% candidate
against a 95% favourite must clear a much higher bar than a 45%
candidate against 55%, and a 1% move must beat the 4% move, not just
the leader.  Implemented as a prior tax on candidate selection --
candidates compare on `EV + lambda(ply) * log(prior)` with `lambda`
interpolated from field 20 (`pw0`, ply 0) to field 21 (`pw1`, ply 44+),
applied uniformly to the initial argmax, the sel_k gate, and the
advisory override layer; `0:0` (the default) reproduces the old
behavior bit-for-bit.  The thresholds were fit empirically, not chosen:
`tools/calib.c` sampled ~93k candidate rows from 640 self-play games
(sampled and argmax trajectories), scoring every candidate with both
the play-time 96-world estimator and a 1024-world sampled-playout
oracle, and `tools/calfit.py` regressed, per prior-gap bucket and ply
band, how large a cheap-search edge must be before the oracle expects
the switch to gain points (full record in
`data/probes/calib_fit_2026-08-24.txt`).  The surface is cleanly
log-linear in log(p_top/p_c) and falls monotonically with ply -- 0.65
pts/nat at plies 0-10 down to 0.06 past ply 44 -- confirming both of
the motivating intuitions: certainty deserves deference, and it
deserves *more* deference early, which is exactly why search had been
restricted to ply>=14 in the first place.  Deployment fit: `pw0=0.7
pw1=0.2`.

Three pre-registered 800-game arms against the adopted spec, all with
the tax on: at the current `plylo` 14 the tax is redundant -- 49.3% ±
3.5 -- because the sel_k paired-SE gate already suppresses the same
false positives at those plies.  The interesting arm moved `plylo` down
to 8, using the tax as the safety net the early plies previously
lacked: 54.6% ± 3.5 over the first 800 games.  A confirmatory
1600-game final gate on that arm, pre-registered at a 52% adoption
bar, came back **49.7% ± 2.4%** (margin -0.4); pooled over all 2400
games the plylo-8 arm reads 51.3% ± 2.0 -- suggestive, not
significant, and below the bar.  Verdict: measured-neutral, the
recommended spec is unchanged.  What survives is the calibrated
threshold surface itself (the empirical answer to "how much should a
95/4 split cost to overturn"), the fields to deploy it, and a sharp
negative result: with sel_k in place, prior taxes and earlier search
buy nothing detectable at 96 worlds.  The one open door is a slow-spec
pairing -- at dets 192+ where sel_k's SEs shrink, the early plies get
cheaper to search and the 800-game signal may be real; that experiment
should pre-register a larger sample from the start.

**The 2026-08-25 reviewer corpus, and the stall-stacked champion.**  A
reviewer pass over a full-strength self-play game (both sides played by
the recommended spec) produced 21 flagged plies, archived verbatim in
`data/probes/reviews.md` with 18 replayable probe rows (`fs1_*` in the
manifest) held out as evaluation ground truth -- never trained on.
Three behavior classes emerged: endgame stalls (still present despite
the earlier stall-shaped run, which predated the gift-shaped adoption),
wager-clutch (policy holding a wager the opponent provably cannot play,
~1% prior on the correct discard five turns running), and pile-draw
refusal (rejecting a useful discard-pile draw the deeper search ranks
first).  The classes were attacked one at a time, per the combined-run
lesson.

The winner is the sequential stack: `--stallpen 0.8` fine-tuned FROM
the gift-shaped champion.  Suite: **342 better / 186 worse** across all
42 probes vs the champion's 300/268 -- the best aggregate of any net,
with the flagged stalls fixed outright (probes 47/49: 0-for-20 to
20-for-20) and unexpected spillover onto untrained classes (pile-draws
17/3 and 18/2, plus probes 21, 71, 96).  Honest costs: suit_order_72
regressed 19/1 to 3/17, play_70 and save_17 worsened.  Match gate:
first 800 games 49.31% +- 3.5, a pre-registered 800-game extension
52.81%, **pooled 1600 games 51.06% +- 2.4** (margin +0.19) -- above
the 49.5% neutrality bar, so the behavior-dominant net is adopted:
**best.bin is now the stall-stacked net** (previous champion preserved
as `data/best_pre_st.bin`).  The recommended spec string is UNCHANGED.

The other two classes are documented attempts: `rl --safewd` (a soft
bonus for discarding a wager the opponent provably cannot play, the
complement of `--giftpen`) fixed its literal probe (fs1_wager_53
0/20 to 20/0) but left the clutch arc unmoved and regressed the
aggregate to 251/246 -- not gated, retained in the trainer for a retry
from the new champion.  Spec field 22 (`sel_draw`: same-action draw
variants clear the sel_k gate at half k, mirroring ov_draw one layer
up) measured suite-neutral with a mild gain on the draw class; its
match arm is parked.  The reviewer's architecture note -- stop spending
worlds on per-pile draw variants of popular plays, spend them on good
candidate plays -- is an open work item.

**The world-allocation follow-up (`draw_filter`, spec field 22-23 era),
measured.**  The reviewer's ply-18 note -- stop spending worlds on every
pile-draw variant of the popular plays, spend them on real candidates --
turned out to be quantitatively right: the advisory draw-variant
expansion was consuming 29% of searched-decision time at the match
spec.  Spec field 23 (`draw_filter`) admits a pile-draw variant only
when that pile's top card is playable by the mover; every historical
useful-draw probe is bit-identical under it, and the freed budget fits
128 worlds inside the old 96-world wall-clock with time to spare.  The
reallocated spec (filter + dets 128) posted the best suite aggregate
yet (357 better / 169 worse) and its pre-registered 800-game arm read
**50.69% +- 3.5** against the adopted spec -- inside the neutral band,
below the 52% bar a spec change must clear, so the recommended spec is
unchanged.  What stands: the waste is real and the filter removes it
safely (`filter at 96 worlds` is a pure ~29% speed win with identical
probe decisions, relevant the day a time budget matters), but at fixed
96-vs-128 worlds the extra determinizations do not convert to match
wins -- consistent with the earlier dets-192 result, and one more data
point that this engine's match ceiling is policy-bound, not
worlds-bound.  Field 23 later gained a terminal value on the reviewer's
direct instruction: `draw_filter=2` suppresses the expansion entirely,
so search evaluates only the policy-ranked moves (variants were
advisory-only -- their sole path to the move was the override -- and
the corrections labeler keeps the full expansion offline, teaching the
policy to RANK good draws rather than having match search brute-force
them; do not combine =2 with ov_draw, which the mode would starve).
All 18 held-out reviewer probes give byte-identical 20-seed tallies
under =2 at ~30% less searched-ply time.

**The wager-clutch class resists reward shaping (two measured negatives).**
The 2026-08-25 review's remaining big class -- clinging to a wager the
opponent provably cannot play -- got a trainer mechanism (`rl --safewd`,
a soft bonus for the safe unload, the complement of `--giftpen`) and two
fine-tune attempts from the stall-stacked champion.  Both failed the
suite precondition, and the failure mode is the finding: a 40-iteration
shaped fine-tune moves non-target behavior more than the target class
gains.  The plain retry reverted the just-adopted stall fixes outright
(probes 47/49 from 20-for-20 back to 0-for-20); anchoring with an active
`--stallpen` during the safewd run still let probe 49 revert while
worse-rates rose elsewhere (342/186 champion vs 345/211 candidate).
Sequential shaping, in other words, does not compose at this scale --
each pass pays interference on every previously shaped class.  The
corrections loop is the tool built against exactly this (search-labeled
distributions at flagged states, policy-agree anchors everywhere else),
and the review's class is now its eighth miner detector (wager-clutch,
tools/mine.c class 8) -- with the older detector classes measurably
refilled since c13 by the shaped passes, a fresh corrections cycle has
material on every front.

**c17: the fourth corrections generation -- and a new champion.**  The
corrections loop, restarted with the review-derived wager-clutch
detector (miner class 8) and the discovery that the shaped fine-tunes
had refilled the older detector classes, mined 400 champion self-play
games into 53k gate-respecting labels (~250 wclutch corrections per 50
games) and fine-tuned the champion in dataset mode with symmetry
augmentation.  The result is the largest gain since the original
c-series: **59.0% +- 1.6 (+15.4 points/game) as raw policies** over
1000 validation games, and -- the part that matters -- the gain
survives the search layer: **54.69% +- 3.4, margin +8.8 over 800
games** at the full match spec, the first decisive search-level gate
win since sel_k.  best.bin is now c17 (previous champion preserved as
`data/best_pre_c17.bin`); the recommended spec string is unchanged.
The suite record is stated honestly: 354 better / 196 worse against
the prior champion's 342/186 -- big fixes (the reviewer's fs1_wager_53
0/20 to 20/0, stall_89, save_17, two long-standing zero-prior probes
to 20/0) alongside real regressions (selk_gift_13 to 0/20,
fs1_open_11, drawGx_27), the familiar rebalancing signature of
corrections training; the regressed rows are the natural targets of
the next mining round, and the match gate says the trade is strongly
net-positive.

**c18: generation five, and the loop keeps paying.**  Mined from c17's
own games with two detector upgrades -- the gift counter-class (the
missing other side of the wager-discard boundary, after c17's clutch
training over-generalized and regressed the founding gift probe) and
late-deck stall coverage -- the corr18 corpus (67k labels) fine-tuned
c17 into c18: **53.5% +- 1.6 (+5.2 points/game) as raw policies** and
**51.88% +- 3.5 over 800 games** at the full match spec.  best.bin is
now c18 (c17 preserved as `data/best_pre_c18.bin`); the recommended
spec string is unchanged.  The suite posts the lowest worse-count in
the corpus's history -- 319 better / 143 worse against c17's 354/196 --
with the reviewer's ply-11 wager-clutch probe fixed outright (0/20 to
20/0), the gift blunder stopped, and premature_B10 healed 19/1; honest
costs: selk_stall_89 and drawGx_29 regressed, prior_76 dipped, and the
ply-13 variant of the clutch arc remains the one reliably-wrong member
of its class.  Two generations in two days, both driven by the same
human review: new detector classes are what reopens the converged
corrections loop.

**The belief head, isolated and improved (reviewer-directed).**  The
opponent-hand inference head had only ever trained as a side objective
of the policy trainers; `tools/belief.c` isolates it -- corpus
generation from champion self-play, honest metrics (BCE skill over the
counting prior and within-decision AUC on the STRICTLY-UNKNOWN card
set, known-card bookkeeping excluded; adversarially reviewed for
leakage before any compute), and head-only training on the frozen
trunk.  Findings, all on an untouched fresh-seed 1000-game test
corpus: (1) the linear head is already at capacity -- retuning it on
1.8M samples cannot beat the RL equilibrium (4.8% skill, AUC 0.639) --
so the bottleneck is the shared trunk representation, not the head;
(2) a SPECIALIST net (full c18 clone fine-tuned belief-only, pw=0
vw=0, on 6.75M frozen-champion self-play samples) lifts inference to
**6.4% skill / AUC 0.653** with gains in every phase (early 2.3->3.7,
mid 7.7->9.4, late 4.9->6.8) and cleaner calibration; (3) doubling
the specialist's width from scratch does worse (6.0%, overfits) --
the warm-started champion trunk, not capacity, carries the signal.
The winner is committed as `data/belief_best.bin` and deploys without
touching a single play decision via the hybrid spec
(`rollouth:data/best.bin:data/belief_best.bin:<usual tail>` -- main
net keeps priors, candidates and playouts; the specialist only steers
world sampling).  Whether sharper worlds convert to match wins is a
separate gated question; the earlier hybrid experiment used the wide
net's un-specialized head and measured neutral, and this one is
strictly better at the only job the slot has.

**Belief, round two: the history features the snapshot erased.**  The
State now tracks which player last discarded each card still in a pile
and how long each player has passed over the current pile tops
(disc_by/passed, maintained in lc_apply, never read by play logic), and
an extended-input specialist (`belx`, tools/belief.c) consumes them as
130 extra feature rows, warm-started from the standard specialist so
it begins at the identical function.  Training required per-group
learning rates -- a uniform rate destroyed the inherited representation
faster than the new signals paid (holdout 6.4% -> 4.1% in two epochs),
while frozen-base training proved the new features add nothing through
frozen pathways alone; the working recipe is full rate on the new rows
with the base at 2% of it.  Final, on the untouched fresh-seed corpus:
**6.7% skill / AUC 0.654** vs the standard specialist's 6.4%/0.653 and
the champion head's 4.8%/0.639, with the late game (where sampling
matters most) improving most: 4.9% -> 7.2%.  The instructive negative:
discard-origin and decline signals, the intuitive core of human hand
inference, add only ~0.3 skill points -- self-play pile contents plus
expedition state already imply most of who discarded what.  Cumulative
belief arc: +40% relative inference skill over the shipped head, all
without touching a play decision; `data/belx_final.blx` is the
artifact, engine integration (world sampling from a BelX net) is the
scoped follow-up if deployment is wanted.

**Belief deployment gates: sharper worlds DO convert to match wins.**
Engine integration first: the rollout world sampler dispatches to the
belief net with the extended-format (`belx`) runtime alongside the
standard-net path (src/belx.[ch], determinize_bx, spec.c auto-detecting
.blx files by magic in the `rollouth` slot), adversarially reviewed --
the one substantive finding, that .state probe files carry no
disc_by/passed history and would probe a belx spec off-distribution,
is closed by decreplay refusing such specs.  Then two pre-registered
800-game gates against the adopted spec, verdicts declared before any
game ran (>=52% adopt, 49.5-52% neutral, <49.5% negative; both above
52% -> higher scorer wins).  Gate A, the standard-format specialist
`data/belief_best.bin` in the hybrid slot: **52.31% +- 3.46, points
margin +2.86** -- clears the bar.  Gate B, the extended-format
`data/belx_final.blx` (better at inference: 6.7% vs 6.4% skill):
**50.88% +- 3.46, margin +0.73** -- measured-neutral, and the history
features that need live disc_by/passed state buy nothing at 96 worlds
that the snapshot specialist doesn't already provide.  Per
pre-registration the recommended match spec became the Gate A hybrid
(the first spec-string change since the selection gate era):

    rollouth:data/best.bin:data/belief_best.bin:96:5:0.02:0:1:14:0:4:0:1:3:4:0:0:0:1

(superseded the same day by the draw_filter=2 tail -- see the
recommended-settings section for the current string)

Same search, same priors, same candidates and playouts -- only the
sampled opponent hands are drawn from the specialist.  The belief
program's chain thus closes end-to-end: +33% relative inference skill
shipped as a +2.3-point match-strength gain with zero play-logic
changes.  Full protocol and numbers in
`data/probes/belief_gates_2026-08-28.txt`.

**Corrections gen-6, the belief-labeled round: a clean measured
negative.**  The natural compounding step -- give the corrections
labeler the belief specialist for its 256-world sampling
(`mine --belief`), the same upgrade that won the match gate for search
-- produced a corpus and a candidate that FAILED at every level.  The
labeler now disagrees with the policy on 54% of flagged states (36,938
samples from 200 games, hedge-dominated at 3,654 corrections), the
anchored fine-tune degrades monotonically past one iteration (it1
+0.75 to it8 -7.29 at 400 policy pairs, while policy CE falls 1.60 to
1.26 -- memorization, not learning), and the it1 checkpoint lost its
800-game search gate **47.62% +- 3.46, margin -2.35**
(`data/c19_cand.bin`, not adopted; c18 stands).  The instructive
part: sharper worlds make a better SAMPLER but not automatically a
better TEACHER -- re-judging every mistake class under a different
world model shifts the label distribution wholesale, and the anchored
student fights its anchor everywhere at once instead of learning
targeted corrections.  The control (same recipe, standard labeler) is
the next measurement: it separates "the belief labeler hurt" from
"the corrections loop is mined out at c18."

**The control round rewrites that story: c20, generation six, adopted.**
The control corpus (standard labeler, seed 20260830) came back
statistically indistinguishable from the belief-labeled one -- 48.5%
flagged / 54.7% corrected vs 49.0%/54.3%, matching class counts -- so
the belief labeler never meaningfully moved the labels; the high
disagreement is intrinsic to mining the champion's own gen-6 games.
Training degraded the same way past its peak (it2 50.4% sliding to it8
46.6%), but the it2 checkpoint told a different story from c19's it1:
Gate F read **50.25% +- 3.46 over 800 games** (401 W / 397 L,
statistically neutral) and the suite -- both nets scored under the
standing no-expansion spec, per the comparability rule -- moved from
c18's 333 better / 152 worse to **341 better / 148 worse**.  That is
the pre-registered net-swap bar (neutrality + suite dominance) met on
both prongs: best.bin is now c20 (c18 preserved as
`data/best_pre_c20.bin`).  The pair of rounds is the real finding:
with an identical corpus profile, the c19/c20 difference reduces to
checkpoint choice and label-world noise -- the loop still pays, but at
gen 6 its yield has shrunk from c17's +8.8-margin gate win to a suite
edge inside match noise, and the per-round trainable signal is one to
two iterations before memorization.  Deeper cycles need a new mistake
CLASS (a detector the suite doesn't yet check), not another pass of
the same nine.

**The upgrade panel (2026-09-01) and its first two experiments.**  A
four-lens proposal panel with hostile critics (protocol and ranked plan
in `data/probes/upgrade_panel_2026-09-01.md`) put two search-side items
at the top; both were built and measured the same day, both against the
c20 champion at the standing spec.

*Test-time symmetrization* (`sym_k`, spec field 25): the rules are
invariant to relabeling suits and wager copies but the trained policy is
not -- `tools/symtest` measures the c20 champion's raw argmax flipping on
25% of relabelings and the K=8-averaged policy disagreeing with the raw
one on 17% of states (36% when the raw top is under 0.40; only 0.7% of
decisions are above 0.95 any more).  Averaging the prior and value over
K random relabelings before candidates form costs +1% wall-clock and
posts **the best suite aggregate on record, 390 better / 151 worse vs
341/148**, yet the 800-game gate reads **51.00% +- 3.46, margin +0.31**
-- above parity, inside the neutral band, short of the 52% spec-change
bar.  Adopted on the reviewer's call: the agent must respect the rules'
symmetries wherever doing so is free, and the 52% bar exists to keep
noise out of the spec, not to keep correctness out of it.

*Exact symmetrization* (`sym_k`, `sym_bel` = 120): the sampled form
had a residue that the suite exposed.  Once the relabelings were drawn
from a stream seeded by the information set (so a display and the
decision it explains compute the same numbers), every seed of a suite
probe shared one draw of 8 relabelings, and the 20-seed sanity read
moved en bloc per probe (326/173 vs 368/162: 19/0 -> 0/0 on one probe,
7/1 -> 20/0 on another) -- the instrument could no longer average over
draws, and neither could the agent.  `tools/symres` measures the
deployed decision function against itself on a relabeled copy of the
same state: at K=8 the top candidate still depended on the draw on 10%
of states, the prior moved by 0.06 total variation and the gate value
by ~1 point; enumerating all 120 suit permutations with equal weight
(Lehmer unranking in `lc_sym_relabel`, no raw term on top) brings that
to 99.4% agreement, 0.004 TV and 0.06 points -- the remainder is the
sampled wager-copy relabeling, which the nets see as distinct cards
and which is pooled exactly afterwards.  Cost: 120 policy and 120
belief forwards per decision, ~13 us each, inside run-to-run timing
noise.  Adopted as the exact form of the standing rule.  The 20-seed
suite then read 322/175 against 368/162, and the diagnosis is a
finding about the instrument: on the probes that moved, the 96-world
search's own best is the policy top under raw, sampled and exact
symmetrization alike; the reviewer's move sits second in a near-tied
prior, and the old sampled relabelings had been randomizing which
near-tied move became candidate 0, kept by the sel_k gate's status
quo.  The suite is built from positions where the policy top was
wrong, so it rewards deviation from the policy top -- noise included
-- and penalizes noise removal (sel_deep and the calibrated sampler
showed the same signature).  **Gate K**, 800 games, exact vs the
sampled-8 regime, pre-registered at a 49.5% floor: **50.38% +/-
3.46, paired margin -2.00 +/- 1.56** -- match-neutral, exact stands.
Protocol, per-probe table and measurements in
`data/probes/symk_gate_2026-08-29.txt`.

*Calibrated belief sampling* (`bel_samp`, field 26): the panel's critics
reproduced a real defect -- Gumbel-top-k on the belief logits is a
Plackett-Luce draw whose inclusion probabilities are not the head's
trained marginals, so late-round worlds collapse onto duplicate
over-confident hands (`belief sampeval`: late-phase sampled skill -1.8%,
below the counting prior; top calibration bin 0.96 predicted -> 0.83
observed; 28% duplicate hands per 96 draws).  A fixed-size conditional-
Bernoulli draw on the shifted marginals restores late skill to +5.2%
with calibration within 0.03 of the head's own -- and then LOSES on the
40-seed suite (660/311 and 657/320 vs the standing 687/314, worse
column flat).  The over-confident worlds had been acting as variance
reduction for the paired candidate comparison: calibrated worlds widen
the paired SE, the sel_k and override gates fire less, and fewer of the
reviewer-verified non-top overrides get through at fixed worlds.
Fidelity up, resolution down; refuted at the suite step, no match gate
run.  Both modes stay in the engine as documented knobs (protocols in
`data/probes/symk_gate_2026-08-29.txt` and
`data/probes/belsamp_gate_2026-09-01.txt`).

**The play x draw head (net v5), measured.**  The factored policy head
could not condition the draw source on the action at all -- P(draw |
action) was identical across actions up to legality, which is why the
corrections miner had to drop draw-only labels (any draw pressure
became a global reweighting).  v5 adds sixteen state gates and a
rank-16 bilinear play x draw term with U zero at init, so every
existing champion loads as the identical function (verified
byte-identical) while the term trains from the first step.  An
8-iteration anchored fine-tune from c20 (`data/xhead_cand.bin`; the
anchored recipe itself measured on four arms in
`data/probes/anchor_recipe_2026-09-01.txt`: it removes the gen-6
interference and adds no signal alone) gave the head exactly the
capability it was built for -- on the reviewer's fs1_p27 the draw
conditional now depends on the action (0.315 vs 0.367 for taking the
Green wager after playing G4 vs discarding R4; c20 was 0.455/0.456) --
and the raw policy scores **51.0% +- 1.6, +3.54 +- 1.41 points/game**
over c20 with the suite at 384/154 vs 341/148.  Under search it reads
**49.25% +- 3.46, margin +0.52** over 800 games: a quarter-point under
the net-swap bar, not adopted.  The c9 lesson again -- a policy-level
gain the paired-world search already extracts.  Everything stays
(referee and browser ports included); the open door is PPO from the
extended net, which the closed raw-self-play result never had.
Protocol in `data/probes/xhead_gate_2026-09-01.txt`.

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
