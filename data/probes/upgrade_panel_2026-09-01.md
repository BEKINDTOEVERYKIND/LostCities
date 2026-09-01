# Upgrade panel, 2026-09-01

Four independent proposers (search/inference, training/data, representation/engine,
tournament objective), each critiqued by a hostile statistician and a hostile engineer
who ran code against the repo, then one synthesis.  Closed directions from the README
were enforced as off-limits.  Ranked by (expected match-win gain x survival probability
at an 800-game gate) / cost.  Verbatim panel text follows; measurement claims inside are
the critics' own reproductions and are to be re-verified before adoption.

## Rank 1: Rules/harness audit of the other team's transcripts, plus a report-only external-opponent tripwire (no adoption authority)

**Mechanism.** Two pieces of the lens-4 harness proposal, stripped of its adoption rule. (a)
Audit: run tools/verify_transcript.py (already checks first-player alternation
by round, 44 deck draws per round, pile-draw legality, discard-then-draw
restriction, independent scoring) over transcripts produced by the OTHER
team's harness as soon as any exist; write a small converter from their log
format to this repo's showgame format. (b) Tripwire: a resumable shell wrapper
in the gate-supervisor style (80 chunks x `bin/arena -n 5 -r 3 -t 4 -s
<chunk>`, see scratchpad/gateH_supervisor.sh) that plays the standing spec
against rollout:data/big1.bin, rollout:data/s2.bin, rollout:data/sym1.bin,
rollout:data/old_best.bin (200 games each), plus 100 games vs
rollout:data/m0.bin and raw-policy games vs heur; writes
data/probes/pool_<date>.txt with per-opponent W/L/D, win%, margin on shared
deal seeds.

**Why.** Not a strength lever, and the ratio formula is degenerate for it: cost is
minutes for the audit, and the numerator is protection of every number in the
README. Both lens-4 critics independently called the audit 'the single
highest-expected-value item in the whole list' and 'do first' -- a rules
mismatch in the opponent's harness (first-player order, last-deck-draw round
end, 8-card bonus, ply caps) would swamp every 1-3 point gain below. The pool
half survives only as a regression instrument: both critics rejected the
proposed '+2 points' adoption rule (0.8 SE on two 800-game estimates; ~20%
false-adopt under a true zero, 50% power at a true +2), showed the 'near-
peers' are the same lineage beaten 64-77% by the raw c20 policy and the
'aliens' (heur, m0, hrollout) are saturated at ~100% wins, and noted the 68.2%
vs old_best reproduction target is stale (c13-era spec). So the pool cannot
separate 'mirror-neutral because the situation never arises' from 'does
nothing', but it can catch a >=2-SE per-opponent regression before the
tournament does, and the sym_k arm (Gate H, now 50.09% at 540 games with the
best suite ever) is a free first customer.

**Implementation Sketch.** Audit: tools/verify_transcript.py as-is + a ~40-line converter for the other
harness's log format; run on >=20 of their transcripts; record
data/probes/rules_audit_<date>.txt. Tripwire: tools/pool.sh (~30 lines) around
`bin/arena -q`, chunked and resumable like gateH_supervisor.sh; baseline the
standing spec rollouth:data/best.bin:data/belief_best.bin:96:5:0.02:0:1:14:0:4
:0:1:3:4:0:0:0:1:0:0:0:0:0:2 once (seeds pre-registered), ~1.6 h on 4 cores
(measured 21 s/game mirror, 34 s/game vs big1, 33 s/game vs rollout:m0); re-
run for any spec/net that reaches the 49.5% mirror band. Weak-alien margin is
reported, never gated (the win-trained champion deliberately gives back ~15
margin points vs heur). Sequence after Gate H releases the four cores.

**Gate.** Audit: pass = zero rule violations on >=20 other-harness transcripts; any
violation is a tournament-blocking finding. Tripwire: no adoption authority;
pre-registered alarm = any per-opponent win% drop >= 2 SE (>= 7 points at 200
games) vs data/probes/pool_baseline.txt on shared seeds triggers a hold on
that spec. Sanity: re-baseline vs old_best rather than reproduce the stale
68.2%.

## Rank 2: Calibrated fixed-size world sampler (conditional-Bernoulli) with once-per-decision belief logits and optional K-relabeling belief averaging

**Mechanism.** determinize_b/determinize_bx (src/agent.c:70-174) draw the opponent's unknown
cards by Gumbel-top-k on the belief logits, i.e. a Plackett-Luce top-k whose
inclusion probabilities are odds-based, not the per-card sigmoid marginals the
head was trained on (tools/rl.c:276-291, tools/belief.c cand_set are per-card
BCE). Late in the game (need ~7 of ~15) the worlds collapse onto over-
confident copies of the head's favourite hand; the trunk is also recomputed
per world on an unchanged root state (agent.c:81-86, ~192 wasted trunk passes
per searched decision). Fix: compute logits once per decision; shift them so
the marginals sum to `need` (logit shift, not multiplicative rescale; clamp
targets into [1e-3, 1-1e-3]); sample each world by the sequential fixed-size
rule using an elementary-symmetric-polynomial DP (n<=52, k<=8, O(n*k) per
world); deck order uniform as now. Optionally average the sigmoid over K
random suit/wager relabelings (lc_perm_map/lc_permute, map back) before
sampling, the belief-head analogue of sym_k at K trunk passes per decision
instead of per world. Refactor into logits-then-shared-sampler so the .blx
path and Gate B' use the same code.

**Why.** The best-supported item: three of four critics PURSUE, one MAYBE ('ship it as
engineering'). Every critic reproduced the defect on the engine's own
determinize_b code: per-card |sampled inclusion - sigma| 0.011-0.014 / 0.028 /
0.053-0.054 early/mid/late, late PL-sampled skill -1.5% to +0.5% vs head
4.5-5.4%, calibration bins 0.85->0.72-0.74, ~25% duplicate hands per 96 late
draws, sampled opponents holding ~9% more playable-now cards than real ones;
plain CB recovers 4.3% late and CPS 5.7-6.6% on the bel6k tail. It is
mechanistically distinct from every closed belief item (all changed the MODEL
or the world COUNT; none checked the sampler) and it is strictly cheaper at
runtime. Honest bounds from the critics: (i) the belief slot's total leverage
is unresolved -- Gate A's +2.31 is a 1.3-SE read equal to the expected maximum
of six null gates, belx (better head) read 50.88, and one critic's uniform-vs-
belief probe check barely moved the paired sel_k statistics -- so this most
likely reads neutral and is adopted on fidelity grounds; (ii) IPF calibration
did not beat plain CB in one rerun (3.6 vs 4.3 late) so build plain CB first;
(iii) hidden gate coupling -- calibrated worlds are more diverse, so paired
SEs widen and sel_k=1 / the 3-SE override fire less often, silently shifting
the operating point of gates calibrated under the sharp sampler; (iv) drop
stratified/weighted worlds (breaks the iid paired-SE formula). Frame-
randomized playouts (lens 1 #2) were NOT carried: one critic rejected it (MSE
unchanged at fixed worlds; the identity frame IS what the gate opponent
plays), the other made it contingent on Gate H adopting, and Gate H is at
50.09% -- only the belief-side K average survives, folded in here.

**Implementation Sketch.** Spec: two new tail fields after sym_k (field 25): 26 = sampler mode (0 =
Gumbel, bit-identical RNG stream; 1 = CB; 2 = CB+IPF), 27 = belief K (0 =
off). Files: src/agent.c -- split determinize_b/_bx into (logits once) +
shared `belief_sample(const float *logit, uint8_t *unseen, int n, int need,
Rng*, State*)` with a per-decision cache (unseen list, need, weights, 53x9
suffix ESP table) and fallbacks to the Gumbel path for need >= n, n > 60, or
non-convergence (0.65% of decisions in the prototype); src/rollout.c
sample_world (four call sites: main batch :285/:340/:487, sampled confirmation
:665) builds the cache lazily; keep determinize_b/_bx signatures for
src/search.c:170 and tools/qpair.c:323. tools/belief.c: add `belief sampeval
NET BELNET STATES.bst FROMGAME` (port of scratchpad corpustest 'samp' /
sampdiag2) reporting sampled-inclusion BCE/skill/AUC and calibration bins by
phase, plus mean paired SE and sel_k/override fire rates on the 42 suite
states under both samplers. Prototype of the sampler exists in
scratchpad/corpustest.c (logit_fix_sum, cps_calibrate, cps_sample). ~150-200
lines, ~1 day; runtime slightly cheaper than today.

**Gate.** Pre-registered before any game: (1) sampeval on data/bel6k.bst games >= 5900
-- sampled-inclusion BCE within 0.003 of the head's, calibration bins within
0.03 of the diagonal, late sampled skill >= 4% (from ~0); (2) suite via
tools/suite.py under the standing spec + `:0:0:1:0` (fields 24-27) not worse
than the standing record (341/148; 390/151 only if Gate H clears 52%, which at
50.09%/540 it will not), with paired-SE and fire-rate deltas recorded; (3) ONE
800-game bin/arena -r 3 gate vs the standing spec on the project's fidelity-
fix rule, declared as such: >= 49.5% plus suite non-regression adopts, >= 52%
is a bonus, < 49.5% negative; (4) Gate B': data/belx_final.blx under the new
sampler as a single 800-game arm (the only belief arm whose delivered late
skill jump is larger than Gate A's), 52% bar, acknowledging an 800-game null
leaves +2% inside the CI. Honest expectation: +0.3 to +1.5 match points; ~70%
adopted under the fidelity rule, ~25% >= 52%.

## Rank 3: State-gated low-rank play x draw-source interaction term in the policy head (function-preserving, LoRA-style init)

**Mechanism.** net_policy_act (src/net.c:159-171) computes logit = play[card*2+disc] +
draw[src], so P(src | action) is identical across actions up to legality: the
head cannot say 'take the Green wager if I discard R4, not if I play G4 onto
Green'. Add g = W_g a2 (r=16 from the 256 trunk), logit += sum_j g_j *
U[card*2+disc][j] * V[src][j] (U [120][16], V [6][16], W_g [16][256]). Init V
and W_g at the existing s4 random scale with a fixed seed and U = 0: the
champion's function is preserved exactly at load and the gradient to U (d*g*V)
is nonzero from step one. Backward adds ~25 lines (dU, dV, dg -> dW_g and the
d2 contribution); net_adam_step, grad accumulation in rl.c/train.c and
net_copy are flat over the block. File format: new section after wbel, hdr[5]
bumped to 5, v4 files load through the existing shorter-prefix path with the
seeded init. Train it with the anchored dataset recipe from rank 4 on a
corrections corpus that KEEPS draw-only labels (plain `mine`, class-2 halving
retained, no --filter).

**Why.** Double-reject rebutted explicitly: both rejections of the lens-2 version hit
(a) a zero-initialised bilinear term whose gradient is identically zero and
(b) a pre-check that counted state-level draw misjudgments the additive head
can already express. Neither applies to this lens-3 formulation, which both of
its critics PURSUE: asymmetric init makes it trainable while bit-identical at
load, and the ceiling argument is replaced by a structural one already written
in the codebase -- tools/mine.c:325-329 explains that draw-only corrections
had to be dropped because 'the 6-logit factored draw head turns any net draw-
direction pressure into a global reweighting (measured as ply inflation and a
17-point collapse)'. The corrections loop has been structurally unable to
teach state-specific draws since day one; the fs1_drawGx_27/29 oscillation
across c17->c18->c20 (one fixed as the other regresses) is the signature of a
head that can only move one shared conditional, and the README's gen-6 verdict
is that the loop needs a new CLASS. distprobe on fs1_p27 reproduces P(take G |
action) = 0.455 for five different actions with the self-blocking 'play G4,
take G' at 0.194 above the reviewer's 'disc R4, take G' at 0.152. Honest
magnitude from the critics: self-blocking mass is ~1% (1.6% of candidate
slots, 6.7% of states); every search-side draw fix measured neutral
(draw_filter 50.44%, ov_draw 47.2%, sel_draw parked); the fs1_p27 failure is
partly a wrong PLAY logit the term cannot fix; and the lineage's raw-to-search
conversion ratio (c17 +15.4 raw -> 54.7%, c18 +5.2 -> 51.9%, c20 ~0 -> 50.25%)
puts the search-level effect at 51-52%. So this is a net-swap-rule candidate
whose real payoff is unblocking a label class the corpora already contain, not
a 52% gate win.

**Implementation Sketch.** src/net.h/net.c: section bookkeeping in net_nfloat/net_wire, forward in
net_policy_act (+16 MACs per move, one 16x256 gate per decision), backward,
seeded init in the hdr[5] < 5 load branch, save as v5. tools/referee.py (numpy
head at ~line 354) and web/play.html (JS section layout lines 371-383, head
416-418) ported; bin/dumpfeat parity re-verified to 1e-6. Unit test: v4-loaded
c20 with U=0 gives byte-identical 20-seed suite tallies. Training: `train
--init data/best.bin --data <corr with draw labels + anchor> --aug 1 --vw 0
--pw 1 --bw 1 --lr 1e-4 --steps 1500 --batch 512 --iters 8 --ref
policy:data/best.bin --eval 400`, checkpoint by validation (the new
6k-parameter head will memorise the corpus's draw labels first). ~1 day build
plus one fine-tune cycle; runtime +~1%.

**Gate.** (1) Function preservation: zero-U extended c20 byte-identical suite tallies.
(2) Mechanism after training: on fs1_p27 P(take G | disc R4) exceeds P(take G
| play G4) by a wide margin (small dump tool in tools/analyze.c); corpus self-
blocking mass from ~1.0% to < 0.2%; ply count per match unchanged (the global-
reweighting signature must not appear). (3) Suite: draw rows pre-registered as
the target set (fs1_drawGx_27/29, g424_drawY_112, g424_G8Y_118,
g424_takeG6_126, c13m_grab_20), everything else as non-regression vs 341/148;
watch suit_order_72, fs1_play_70, fs1_prior_76 for the interference signature.
(4) Primary readout: raw-policy 1000-game bin/arena vs policy:data/best.bin
(SE 1.6%, where c18's +5.2 was clearly resolved). (5) Search gate at the
tournament spec, 800 games minimum (1600 preferred), adopt on the pre-
registered net-swap rule (>= 49.5% + suite dominance + raw-policy win), 52% a
bonus. Expectation: raw +2-5 pts, search 51-52%, ~20% >= 52%, ~40% net-swap
adoption.

## Rank 4: Play-distribution self-distillation anchor for corrections fine-tunes, with an anchor-only control arm

**Mechanism.** Build an anchor corpus from the champion's own play distribution with the
existing trainer: `./bin/train --init data/best.bin --gen selfpolicy --iters 1
--games 600 --steps 0 --eval 0 --sample-plies 80 --dump data/anchor_c20.smp`
(tools/train.c AG_POLICY generator, lines ~168-187: target = top-12 of c20's
own policy, both perspectives, true opponent hand for the belief head). Merge
with the corrections corpus via tools/merge_samples.py (~4.5:1 by sample
count) and run the c17-c20 dataset recipe with the value loss OFF: `train
--init data/best.bin --data data/corr20a.smp --aug 1 --vw 0 --pw 1 --bw 1 --lr
1e-4 --steps 1500 --batch 512 --iters 8 --ref policy:data/best.bin --eval
400`. Three arms, identical seeds: anchor+corr20, anchor only, corr20 alone
(plus a corr20-alone arm at ~300 steps/iter so 'no it2 peak-then-slide' is not
just slower training).

**Why.** MAYBE/PURSUE. Both critics verified the factual basis: the c17-c20 supervisor
scripts trained on corrNN.smp alone, the only anchor was mine.c's agree-
samples (17.8% by count), and corrstat reproduces (c20 absorbed ~1/5 of
corr20; corrected states sit at mean top-prob 0.45). The PURSUE critic's
corrections are load-bearing and are folded in: the anchor carries lambda-
return value targets on a different scale from the win-trained head so --vw 0
must be explicit (default vw=1 deforms the trunk); c17-c20 used --batch 512,
not 256; --sample-plies is match-cumulative so the default 24 leaves rounds
2-3 pure argmax; and under --aug 1 the anchor target is computed on the
unpermuted state and trained on a relabeled one, so the anchor-only arm is a
symmetrization-distillation corpus and a mandatory confound control. The MAYBE
critic's objection caps the expectation: the anchor removes interference but
adds no signal, gen-6 labels are the labeler's ungated 256-world argmax at
near-tie states (mine.c sets no sel_k), so its best case is parity, and the
record's conversion ratio says a fine-tune needs ~57%+ at 400 policy pairs to
have any chance at 52%. Gate H at 50.09% with the best suite on record also
removes the 'distilled sym_k converts to wins' upside the PURSUE critic hoped
for. Ranked here because it is nearly free (~5 min per 8-iteration arm), is
the training recipe rank 3 depends on, and settles the 'memorization' belief.
The lens-2 posterior-target distillation of the 512-world search was deferred
rather than ranked: both critics rated it MAYBE only after m_min is raised to
the validated 4-point floor and a solver kill-test on selk_stall_43/91/93 is
added, corrected the cost to ~3 h per iteration (c20 has 0.7% of window plies
at >= 0.95 confidence, so nearly every ply is searched), and put under 20% on
52%; revisit it only if rank 3's head plus this anchor produce a policy-vs-c20
read >= 56%.

**Implementation Sketch.** Zero engine code. tools/train.c flags as above; tools/merge_samples.py for the
merge; per-iteration policy-vs-c20 at 400 pairs via --ref/--eval. Optional
replay arm (corr17/corr18 through a ~40-line 248->280-byte State converter +
`mine --filter` against c20) is low value; the existing beldump*.smp corpora
(6.7M c18-era play-distribution samples) are an alternative anchor source
after the same conversion. Sequence after Gate H releases the cores; suite ~1
h; a match gate only if the trigger below fires.

**Gate.** Pre-registered: (1) training-curve signature -- anchor arms must not show the
it2-peak-then-monotone-slide while the corr20-alone control reproduces c20's
curve; (2) trigger for any further spend: policy-vs-c20 >= 56% at 400 pairs on
the selected checkpoint (the record's ratio for a plausible 52% search read);
(3) tools/suite.py under the standing spec >= 341/148 with selk_gift_13,
drawGx_27/29, selk_stall_89, fs1_open_11 called out; (4) 800-game bin/arena at
the tournament spec vs c20, standard bars (>= 52% adopt; 49.5-52% + suite
dominance = net-swap; < 49.5% negative). Expectation: gate-neutral most
likely, ~10% on 52%; the deliverable is a usable anchored recipe and a clean
answer on memorization.

## Rank 5: Variance-aware match-utility aggregation in round 1 -- instrumentation first, build only on a quantitative go/no-go

**Mechanism.** rollout_move (src/rollout.c, `usew = lastround && a->win_q` at ~:527)
aggregates per-world playout margins val[c][d] directly in rounds 0-1; win
awareness exists only as the closed hard 0/1 win_q in the last round. Map each
world's margin to a points-equivalent utility u_d = sigma_r * sqrt(2*pi) *
Phi((cum_p - cum_o + m_d) / sigma_r), sigma_r = fitted SD of the remaining
rounds' total margin (measured 39-41 with one round left, 54-58 with two),
applied at the accumulation point so the eligible argmax, the sel_k paired SE,
the override paired SE / points floor and sel_deep pooling all inherit it; the
sampled-confirmation block must apply the same per-world transform or its
0.5*override_min floor compares different units. Last round: arm A keeps
margin, arm B uses a soft step (sigma ~8). Two tail spec fields
util_sigma_r1:util_sigma_last, 0:0 = bit-identical.

**Why.** MAYBE/MAYBE, and the two critiques agree on the same arithmetic, which is why
it sits fifth and only as an instrumented probe: round 0 is inert by
construction (cum diff is 0, so u is odd about 0 and E[u] ~ E[m]), the whole
effect lives in round 1 at |carried| >= 20 and is second-order (0.5*u''*dVar,
~1 point-equivalent for candidates whose across-world SDs differ by 2 points),
the affected decision count is ~0.6 per player-match, and the honest +0.3-1%
sits 3-4x below what even a 1600-game gate resolves (SE 1.25%). Both critics
independently reproduced the calibration (cumhist: round SD 39.0-40.6, two-
round SD 55.3-57.7; Phi(mid/sigma) within 2-4 points of every leader-wins
bin), so the transform is principled and zero-cost -- but win_q, the same
objective with a step function, read 50.4 +- 0.8 over 4000 games. One critic
notes arm B (soft last round) is the direct test of the binomial-noise
explanation and should run first, not conditionally on A. Because it is an
objective change, a neutral read must NOT adopt (unlike rank 2's fidelity
fix).

**Implementation Sketch.** Stage 1 (~40 lines + minutes of compute): sigma refit from 400 arena matches
at the match spec (per-round margins logged from tools/arena.c), then
instrument 200 self-play matches at the match spec: on searched round-1 plies
record whether the utility choice differs from the margin choice after sel_k
and the utility gap, and stats->q stays in margin with utility reported
separately. Stage 2 only if the trigger fires: ~60-80 lines in src/rollout.c
(per-world transform in the main batch, sel_deep second batch and sampled
confirmation; erf already linked), next free tail fields after rank 2's 26/27.

**Gate.** Go/no-go before any game: >= 3% of searched round-1 decisions change by >=
override_min utility points AND (changed decisions per player-match x mean
win-probability delta at the carried margin) >= 2% per match; expectation is
that it stops here. If it proceeds: tools/suite.py near byte-identical (probe
states mostly carry cum 0 0; better >= 333, worse <= 156); 1600-game bin/arena
-r 3 vs the standing spec, disjoint seeds, arm B first, >= 52% adopt /
49.5-52% neutral-and-NOT-adopted / < 49.5% negative, judged on W/L.
Expectation: +0.3-1% if real, ~30% of reaching the gate, ~20% of clearing 52%
given it does.

## Rank 6: bel_tau logit tempering field plus `belief eval --opp`, with the style-robust belief retrain deferred until real-opponent transcripts exist

**Mechanism.** (a) Spec field bel_tau scaling the belief logits in
determinize_b/determinize_bx (key = tau * clamped_logit + Gumbel; tau = 1
today bit-identical, 0 = uniform worlds) -- in rank 2's refactor this becomes
a scale on the hoisted logits before calibration. (b) Land
scratchpad/beldiag.c as `belief eval --opp SPEC` in tools/belief.c: skill over
the counting prior, AUC and per-phase BCE of the deployed head when the
champion faces a given opponent style, and a mode that scores it on external
transcripts with true hands. (c) Only if (b) shows a negative-skill phase
against the actual tournament opponent's style: extend the belief corpus
generator (train.c gen_worker seats one Agent on both sides; add a second
Agent and seat assignment, ~40 lines) with an opponent mixture, retrain the
specialist belief-only (`train --data --pw 0 --vw 0 --bw 1` warm-started from
data/belief_best.bin) and ship it through the rollouth belief slot exactly as
Gate A did.

**Why.** MAYBE/MAYBE from both lens-4 critics, and their agreement is the design: the
deficit is real and reproduced (data/belief_best.bin -1.4 to -1.6% skill vs
heur with late-round -5.3 to -7.1%, -0.4 to -0.7% late vs m0, while 5.8-8.2%
across the self-play lineage), but its match value is bounded by the whole
belief deployment gain (Gate A +2.3, a 1.3-SE read), it is only realisable
against styles the champion already beats ~100% on disk, and part of the skill
spread is opponent predictability rather than distribution shift (5.1-5.3% vs
policy:best.bin:1.0, which IS the training distribution). Both critics
converge on the same recommendation: land the ~10-line hedge and the eval tool
now, run the diagnostic on the other team's transcripts the moment rank 1's
audit yields any (their states with true hands are exactly what beldiag
needs), pick tau on that evidence, and spend the hours on a mixed retrain only
if the real opponent's style shows a deficit -- guessing the style with heur
is the wrong prior for an unknown tournament bot. Explicitly dropped from this
lens after double rejections: the always-take opponent seat (a 4-10x
miscalibrated, seat-asymmetric estimator bias whose probe flips were shown to
come from the round-length channel, not gift pricing, with its deck>=9 gate
fitted on the suite) and pool-mined corrections / pool PPO at the policy level
(class-2-dominated corpus the miner already halves and filters; unsatisfiable
+2 bar against opponents beaten ~100%; value baseline trained on the pool
agent's returns). The lens-2 opponent-pool PPO fine-tune (REJECT/MAYBE) is
likewise out: under the finishing reward the return vs weak members is near-
constant (+~55) and globally standardised advantages turn the pool games into
a self-imitation gradient; the only variant either critic would entertain is a
league of frozen self-lineage checkpoints under the margin reward, which
cannot buy alien-style robustness and inherits the interference record of
every PPO-from-champion fine-tune.

**Implementation Sketch.** src/agent.c: tau applied to the clamped logit (10 lines), spec tail field
after rank 2's (default 1.0, bit-identical); tools/belief.c: `eval --opp SPEC`
(~60 lines, port of beldiag) plus a transcript-scoring mode reusing the rank 1
converter; corpus/retrain path per (c) only on evidence. No play-logic change;
the hybrid slot is the proven deployment path.

**Gate.** No match gate for (a)/(b). Pre-registered inference thresholds for any
deployment: self-skill on fresh-seed corpora >= 7.2%; skill vs the real
opponent's transcripts > 0 in every phase; held-out styles
(rollout:data/big1.bin, hrollout) not worse than today. A retrained specialist
goes through Gate A's protocol: 800-game hybrid gate vs the standing spec at
>= 49.5% (the mirror cannot see the benefit) plus the inference thresholds.
Never deploy tau < 1 on pool or mirror match evidence -- both are below the
resolution of the effect; deploy it only on a measured negative-skill phase
against the actual opponent.
