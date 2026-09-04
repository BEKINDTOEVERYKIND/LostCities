# Execution plan — LostCities improvement round, 2026-09-04

State of the machine at time of writing: no arena running (Gate P finished 18:25 UTC, `GATEP_DONE`); HEAD is `102e5be` "arena: dynamic pair queue with per-pair play seeds" — i.e. the C half of proposal 1 is ALREADY COMMITTED (src/match.c:44-53 pulls pairs from an atomic counter and seeds the play RNG per pair; verified identical results at -t 4 and -t 2). Only the supervisor half remains. Corpora on disk: data/anchor_c20.smp, data/corr20.smp, data/corr20a.smp, data/bel6k.bst, data/beltest.bst. Last gate seeds used: 5600-5679 (Gate P). Next free seed blocks: 5700+.

Ranking is by expected value per CPU-day after the critiques (throughput multipliers first because they cost ~0 CPU and every later item inherits them; then strength candidates ordered by P(adopt) x gain / CPU-days, each with a hard cheap kill).

---

## Rank 1 — Gate runner at full utilization (supervisor half; C half already landed)

**(a) Mechanism / files.** C change is done (`src/match.c` worker(), commit 102e5be). Remaining: every gate template still runs `bin/arena -n 5 -r 3 -t 4` (e.g. `/tmp/claude-0/-home-user-LostCities/a6899ed3-62e1-5e6d-b08d-0674a82a78fa/scratchpad/gateP_supervisor.sh:11`). Change the template to `-n 8 -r 3 -t 4` with 50 chunks per 800-game gate (50 chunk seeds x 8 pairs x 2 games), and the same in `tools/pool.sh`. Record `runner=102e5be, chunk=8 pairs` in every gate file from now on. Optional hygiene: `tools/arena.c -q` should print per-pair scores (needed for rank-1's determinism check and for the honest paired SE, see deferred item "honest gate statistics (a)").

**(b) Pre-test (<2 CPU-h).** (i) Determinism: one chunk, seed 5700, `-n 8 -t 4` vs `-n 8 -t 1`, identical totals (already shown for -t 4 vs -t 2 in the commit; ~3 min + 8 min). (ii) Throughput: time 2 chunks of `-n 8 -t 4` at the standing string on the idle box; pre-registered target <= 34 s per 10-game equivalent (archive: 43-45 s). Fail = revert to -n 5 and investigate.

**(c) Pre-registration text.** "Runner change only (commit 102e5be + chunk 8). No strength gate: decision-preserving by construction. Validation: (1) chunk seed 5700 identical at -t 1 and -t 4; (2) next 800-game gate records s/chunk; adopt the template if <= 34 s/10 games. Historical seeds are not byte-replayable under the new seeding; archives stay valid as recorded."

**(d) Dependencies / concurrency.** None. 15 minutes, do it first; every gate below assumes it (800 games ~42-46 min alone on the box).

---

## Rank 2 — Register-tiled `net_trunk` (decision-preserving ~1.25-1.35x on every search workload)

**(a) Mechanism / files.** `src/net.c` net_trunk (lines ~135-160): single-state re-tiling only — keep the a1 zero-row skip and dense-feature skip, keep per-output accumulation order over i (bias, sparse idx in order, dense j in order, layer-2 i in order) so sums are bit-identical under the project's `-O3 -march=x86-64-v3 -ffast-math`; hold a tile of 64 output accumulators in registers across i (both critics measured this exact form at 9.4-10.8 us vs 13.3-14.1 us, bit-identical on 1024 real states). NO lockstep restructuring of `src/rollout.c` (measured ~0 incremental gain with AVX2; dropped). ~40-80 lines.

**(b) Pre-test (<2 CPU-h).** Microbench: 1024 states from data/bel6k.bst, old vs new trunk, assert `memcmp` of a2 == 0 and report us/forward (target <= 11 us). Then `tools/suite.py`-style decreplay identity: all 42 manifest probes x 20 seeds byte-identical to the current binary (the 087d1bc / drawexp precedent; ~10 min single-threaded). If identity fails on any probe, fall back to explicit fixed-order accumulation; if that also fails, it becomes a strength change (800 games at the >= 49.5% decision-preserving bar) — do not ship silently.

**(c) Pre-registration text.** "`data/probes/trunk_tile_<date>.txt`: decision-preserving kernel rewrite. Acceptance = (1) bit-identical a2 on 1024 states; (2) 20-seed decreplay tallies identical on all 42 probes; (3) wall-clock per game recorded before/after at the standing string (`arena -n 8 -t 4`, seed 5701, 2 chunks). No strength gate unless identity fails."

**(d) Dependencies / concurrency.** None; half a day of engineering on an idle box, validation ~1 CPU-h. Do it in parallel with rank 3's pre-check (both are single-thread jobs). Everything downstream (mining, gates, distillation generation) inherits ~25-35%.

---

## Rank 3 — Significance-gated, estimator-matched corrections labeler (gen-7 from c20), pilot first

**(a) Mechanism / files.** `tools/mine.c` worker (lines 211-229): the labeler Agent currently sets only `lab.dets`, `lab.gate=0`, `lab.eval_cand=4`, `lab.override_k=3`, `lab.playout_sample=1`; `agent_default` (src/agent.c:7-33) leaves `sel_k=sym_k=sym_bel=bel_samp=0`. Add: `lab.sel_k` (arm k=1.0 and k=1.5), `lab.sym_k=120`, `lab.sym_bel=120`, `lab.bel_samp=1`, `lab.net_b=data/belief_best.bin`; KEEP `draw_filter=0` (the advisory draw expansion is guarded by `eval_cand>0 && draw_filter<2` in rollout.c, and the README keeps it offline on purpose) and keep eval_cand 4 + the 3-SE/4-pt override. A correction (dup 4) is written only when a non-top candidate clears BOTH the sel_k bar and the 4-point floor (the two-gate logic the record validated); flagged-but-agree states remain confirmations. Also log `dm`, paired SE, class, ply to a side file for every written label (~20 lines) — this is the diagnostic the critics say is the real deliverable. Optional arm: gain-weighted dup `reps = 1 + min(3, floor(dm/override_min))`.

**(b) Pre-test (<2 CPU-h).** Two pilots of 30 games each (k=1.0, k=1.5) with the logging on, 4 threads, ~20-25 min each. Read: corrected % of flagged (currently 54.7%), per-class counts, dm/SE distribution, share >= 2 SE and >= 4 pts, hedge-class share. KILL if (i) fewer than ~40 gated corrections per 30 games (projects < 500 per 400 games) or (ii) the median existing ungated correction already clears 1 SE at 256 worlds (premise dead: labels were not noise). Either outcome is recorded in the README as the "noise vs mined-out" measurement.

**(c) Pre-registration text.** "`data/probes/gen7_labeler_<date>.txt`. Arms: labeler k=1.0 vs k=1.5, 400 c20 self-play games each only if the pilot passes; train with the anchored recipe (`train --init data/best.bin --data <anchor_c20 + corr21> --aug 1 --vw 0 --pw 1 --bw 1 --lr 1e-4 --batch 512 --steps 1500 --iters 8`). Checkpoint = held-out label CE minimum (see rank 3 note below) confirmed by ONE 1000-game raw arena vs `policy:data/best.bin`; proceed to the search gate only at >= 54% raw (SE 1.6). Suite = tripwire only (fs1_* never trained on). Gate Q: 800 games, seeds 5800-5849 (50 chunks x 8 pairs), candidate as main net in the standing string vs standing; >= 52% adopt, 49.5-52% + suite dominance + raw win = net-swap rule, < 49.5% negative; pool tripwire on adoption."

**(d) Dependencies / concurrency.** Wants rank 2 (mining is 100% playout forwards). Pilot: 1 CPU-h, can share the box with rank 2's microbench. Full run: mining ~2-2.5 h wall on 4 cores (~10 CPU-h), train 40 min, raw arena 10 min, gate ~45 min => ~1 CPU-day if it runs to the gate, ~3 CPU-h if the pilot kills it. Reuse its labeler fields in rank 4.

Note (fine-tune hygiene, folded in from the "recipe hygiene" proposal): add `--holdout FRAC` to `tools/train.c` with a hash-of-`State.deck` game split (contiguous tails are wrong for merged corpora) and print held-out correction CE and anchor CE every iteration (~60 lines); use the CE minimum as a memorization tripwire and pick ONE checkpoint for the 1000-game confirm instead of max-of-8 400-pair reads. Calibrate it retrospectively on the existing arm_A..D and xhead it1..it8 checkpoints (minutes) before pre-registering any CE-based selection.

---

## Rank 4 — Gated search distillation on the tournament agent's own play distribution (weakened form)

**(a) Mechanism / files.** Generate self-play at the exact tournament string on both seats (train.c's spec_parse generator branch, `train.c:546-554`, AG_ROLLOUT branch 131-155 already traverses by the gated choice and emits the gated one-hot) BUT label with rank 3's validated labeler (256 worlds, sampled playouts, eval_cand 4, sel_k + 4-pt floor), not the 96-world play-time choice. Target rule: one-hot (dup 2, not 4) only where the gated labeler choice differs from the symmetrized policy top AND clears both gates; everywhere else (agree plies, unsearched plies < 14) the policy's own top-12 soft distribution as self-anchor (the selftarget form of mine.c:370-400) — never one-hot on the argmax. Filter draw-only qualifiers with `mine --filter`'s rule (the class mine.c already drops). Implementation: `mine --all` mode (no detect(), traversal by the tournament agent's move after an early sampled window; ~100-150 lines) plus corpus-stat dump. Drop arm C (solver labels at deck<=3: measured zero match leverage twice).

**(b) Pre-test (<2 CPU-h).** 40 tournament-string matches with the labeler on (~1.5 CPU-h with rank 2). Corpus go/no-go: qualifier rate per searched ply 5-20%; share of qualifiers >= 2 SE >= 50%; draw-only share after filter < 5%; class histogram from detect() in report-only mode comparable to gens 4-6; plies per match unchanged. KILL if qualifier rate > 25% (noise regime of the 48.2% expert iteration) or if a fresh-batch relabel of 200 qualifiers flips > 35% of them.

**(c) Pre-registration text.** "`data/probes/distill_<date>.txt`. Arm A: 500 matches at the tournament string, validated labeler, targets as above, anchored recipe with held-out CE, one 1000-game raw arena; search gate only at >= 54% raw (panel trigger, adjusted for the 0.3-0.5 raw-to-search ratio). Gate R: 800 games, seeds 5900-5949, candidate vs standing, >= 52% adopt / 49.5-52% + suite dominance net-swap / < 49.5% negative. Tripwires: fs1_47/49/53, selk_stall_89/91 (estimator-schism check: the search prices its own stall habit). Arm B (192-world labels) only if arm A reads >= 54% raw."

**(d) Dependencies / concurrency.** After rank 3's labeler fields exist (same code); generation ~4-5 h wall on 4 cores for 500 matches at 256 worlds (~33 CPU-s/game measured at the current string, less with rank 2). ~1.5 CPU-days end to end; sequence its generation overnight after rank 3's gate.

---

## Rank 5 — Turn-arithmetic feature block through a function-preserving v6 loader (supervised ablation only, no gate unless it earns one)

**(a) Mechanism / files.** `src/net.c` net_load (~line 390): accept `hdr[1] < FEAT_DIM` only for a v6 header, read the old w1 prefix rows, zero-pad new rows, bump save version (the v5 U=0 / belx_from_net trick); `src/features.c` feat_extract: append ~30 rows at index >= 556: my_turns=(deck_left+1)/2, opp_turns=deck_left/2 (/22), last_draw_is_mine=deck_left&1, one-hot deck_left 0..14, nplay_total and clip(nplay_total - my_turns), per suit: unseen-above-my-top count, unseen-above-opp-top count, playable-in-hand minus my_turns (per-suit rows permute with the suit relabeling so `--aug 1` and sym_k stay exact). `net_adam_step`: add a row-range lr scale (~15 lines) for the belx-style 2%-base / full-new-rows rate, and `--freeze-rows LO:HI` for the control arm. `tools/referee.py` and `web/play.html` ports only on adoption. Note BELX_XDIM = FEAT_DIM+130: give .blx a compat refusal (not in the spec).

**(b) Pre-test (<2 CPU-h).** Step 0: extended c20 with zero rows replays the 20-seed suite byte-identically (10 min). Step 1 (the real test, before any training): solver-bucketed error probe — on the 40k late states (scratchpad acts.bin / data/bel6k.bst), c20 top-1 error vs `lc_solve_root` (src/solver.c) at deck <= 5, bucketed by parity and by sign(nplay - turns_left); PROCEED only if the parity-sensitive bucket's error rate is elevated over the rest by more than its binomial SE. Then the 3-seed F (rows trainable) vs Z (rows frozen) anchored fine-tune on corr20a/anchor_c20 with the rank-3 held-out split; readout = held-out correction CE in the deck<=8 bucket, class-7 bucket, and parity linearly decodable from a2 > 0.95 on arm F. The critics note anchor labels come from a parity-blind function, so exclude deck<=14 anchor states from the anchor slice or anchor on search labels there.

**(c) Pre-registration text.** "`data/probes/turnfeat_<date>.txt`. Stage 0 byte-identity; Stage 1 solver-bucketed error probe (go if parity bucket error - rest > 1 SE); Stage 2 F vs Z, 3 seeds, adopt-to-stage-3 only if F beats Z on deck<=8 held-out CE by more than the 3-seed spread AND does not lose on deck>14 AND parity probe > 0.95; Stage 3 1000-game raw arena >= 54%; Stage 4 Gate S 800 games, seeds 6000-6049, candidate as main net, standard bars. Belief-slot arm dropped unless Stage 2 shows belief CE gain."

**(d) Dependencies / concurrency.** ~1 day engineering, ~3-6 CPU-h of training if Stage 1 passes (fine-tune arms are 5 min each; the box can run them while rank 4's generation waits). Rank 3's --holdout harness is a prerequisite. Realistic P(52%) ~10-15%; keep for the loader infrastructure and the probe answer even if it stops at Stage 1.

---

## Rank 6 — Racing deepening (sel_deep=2: second batch for survivors only)

**(a) Mechanism / files.** `src/rollout.c` batch loop (lines 596-612, `if (a->sel_deep && want == reps)`): with sel_deep=2, after batch 1 trigger only on a POSITIVE lead (same trigger as sel_deep=1; the "contested-but-negative" trigger is dropped — it buys batches that almost never change a move); survivors S = {cand 0} + {eligible c with deficit vs the leader < 1.5 paired SE}; per-candidate `nrep[c]`, pooled dm/SE in the sel_k block (line 653+) over the common 192 worlds for survivors; non-survivors keep 1x stats and are excluded from selection. sel_deep 0/1 bit-identical. ~80 lines + spec.c doc. Arm S (plylo 22) is dropped from this item — separate question with its own negative history.

**(b) Pre-test (<2 CPU-h).** Extend scratchpad plyprof.c (2 matches, ~15 min) to log, per searched decision, the survivor-set size under the 1.5-SE rule and whether batch 1 triggered; compute the expected wall-clock multiplier = 1 + sum(trigger x |S|/ncand x time share). PROCEED only if the multiplier <= 1.35 and the survivor set changes the pooled pick vs sel_deep=1 on < 3% of triggered decisions (i.e. it is a cost reduction of Gate M's estimand, not a new estimand). Then 2-chunk wall-clock at seed 5750.

**(c) Pre-registration text.** "`data/probes/racing_<date>.txt`. Suite tripwire 20 seeds (consistent if within 13 of standing on both columns). Gate T: 1600 games, seeds 6100-6199 (100 chunks x 8 pairs), A = standing string with sel_deep field = 2, B = standing; primary readout the 52% bar at 1600 games (CI +/-2.45), paired margin secondary; report s/chunk both sides. >= 52% adopt (at the recorded wall-clock multiplier); 49.5-52% = neutral, recorded as the third read of the resolution family and the family closed; < 49.5% negative."

**(d) Dependencies / concurrency.** Half a day of code; gate ~1.5-1.8 h wall on 4 cores with rank 1+2 (1.3x on one side). Run it in a slot when no training job is on the box (it needs all 4 cores). P(>= 52%) ~30%; it re-gates Gate M's estimand at lower cost, which is why it ranks last of six.

---

## Suggested 4-core schedule (day 1 → day 3)

- **Now (box idle):** rank 1 template edit + 2-chunk timing (15 min, 4 cores) → rank 3 pilot k=1.0 then k=1.5 (4 cores, ~45 min) while rank 2's kernel is written (no CPU) → rank 2 microbench + decreplay identity (1 core, ~15 min) → rebuild bin/ with rank 2.
- **Day 1 evening/night:** if the rank 3 pilot passes, mine 400 games x 2 arms (4 cores, ~4 h with rank 2); else start rank 4's 40-match pre-test. Rank 5 Stage 0/1 (solver probe, single core) can share the box with mining.
- **Day 2:** rank 3 train + held-out CE + 1000-game raw arena (30-50 min); Gate Q if >= 54% raw (~45 min). Then rank 4 generation (4-5 h). Rank 5 Stage 2 arms in the gaps (5 min each). Rank 6 code in parallel (no CPU).
- **Day 3:** rank 4 train/confirm/Gate R; rank 6 pre-test + Gate T (1600 games, ~1.7 h) in a 4-core slot.

Rule for the whole plan: one job that needs 4 cores at a time (gate, mining, generation); single-core diagnostics (decreplay, probes, microbench) may overlap. Every gate file records runner commit, chunk size, and s/chunk.

---

## Surviving proposals not in the top 6 (deferred, one line each, nothing dropped silently)

- **Honest paired gate statistics + sequential looks:** ship part (a) now as hygiene (per-pair win score with n-1 variance in `src/match.c:117` and the margin SE; aggregator pools per-pair sums; re-report archived gates with CI +/-2.8) — it changes no decision; part (b) (300/600 looks, 1200 continuation) requires the owner to amend the 800-game rule, so it is proposed, not implemented.
- **Inference-headroom bound (omniscient-hand ablation):** run 800 games (not 400; the 54/56 bands are sub-SE at 400) as a README ceiling measurement in an idle slot, no funding rule attached; the follow-on likelihood sampler is gated on the offline diagnostic below, not on this number.
- **One-step opponent-action likelihood correction:** offline gain re-measured at pooled +0.3-0.4 / late +0.6-1.0 skill (belx-sized) at +10-45% wall-clock; park; revisit only if the 800-game omniscient bound reads >= 55% AND the search-corpus lik pre-test on scratchpad search80.bst shows pooled >= +1.0.
- **Deployment-distribution belief corpus:** measured already this session — specialist reads 8.4-8.8% skill on search-agent states vs 5.5% on policy self-play (surplus, not deficit); the re-fit branch is dead; keep the search-state corpus only as the substrate for the likelihood pre-test.
- **Belief specialist fine-tuned on search play:** same measurement kills its premise; record the number, no run.
- **Oracle-regret instrument:** do the no-tool fallback only (hold rows in data/probes/manifest.tsv, suite reported in two halves, ~3 h, no admission authority); the full tool is 1.5-2 days for an instrument whose margin readout decouples from wins (sel_k: 53.8% at -6.7 pts) — build only if a sweep of spec fields is scheduled.
- **Held-out ablation harness / repprobe:** the --holdout + --freeze-rows half is absorbed into rank 3/5; repprobe.c lands only if rank 5 passes Stage 1.
- **Prior-gap-faded sel_k:** the four motivating probes print no [sel] line (search argmax already = candidate 0), so the fade cannot move them; only an ~8%-of-decisions population count (30 min plyprof) could revive it, at the plain 52% bar with no symmetry-rule adoption path.
- **Playout-ply budget (horizon-scaled worlds):** a softer dets-192 (measured 51.4%); one gate at most after rank 6 reads, continuous form only, neutral closes the family.
- **PPO from the x-head + unshaped PPO-from-c20 control:** no code, ~75 min wall; run as an idle-time probe with a single-checkpoint 1000-game confirm as the only gate trigger (>= 53%); five champion-start PPO runs on record all read parity, so expectations are informational.
- **Discard-pile order planes:** attach as a zero-budget arm P inside rank 5's Stage 2 only; never gated alone (decision-relevant in ~2% of late decisions).
- **Function-preserving trunk deepening:** run the flat-net PPO control (above) first; build the identity-initialised layer only if that control moves and stalls; needs a section-aware weight decay (net_adam_step's flat decay would destroy the identity init).
- **Joint-structure belief sampler (negative pre-test):** record in the README as an OPEN small residual (cite scratchpad/rc_joint_m1_s40.txt with the mode-0 control), not as a closed line; drop the "AR can only hurt" sentence and the 0.01 bar.
- **Opening-phase corrections class 10:** as written it is a phase filter (fires on 63% of early plies, yield at the null floor); the salvageable kernel is a rule-derived wager-foreclosure detector (mine.c class 6 skips wagers) with the 4-pt floor — queue behind rank 3.
- **Gate admission ladder:** would have blocked c18 (53.5% raw) and c20; keep only as a non-binding queue-ordering note and the already-standing "suite is a tripwire" rule.

## Killed proposals (one line each)

- **Regression-adjusted paired estimates (control variates from sampler inclusion probabilities):** novel and free at play time, but the paired-SE/argmax-determinism structure means the variance it removes is not the variance that limits selection; no path to a 52% read, killed at critique.
- **Reply-aware discard pricing (1-ply minimax opponent reply inside worlds):** nearest precedent ("always-take opponent seat") was double-rejected in the 2026-09-01 panel as a seat-asymmetric, 4-10x miscalibrated estimator whose probe flips came from the round-length channel; the proposal omitted that record.
- **Sampled-continuation confirmation for sel_k qualifiers:** three near-neighbours at the same layer failed (prior-aware threshold 49.3%, fresh-batch veto refuted, sel_k 1.5 < k=1); same bias, same layer, no new angle.
- **Persistent particle filter over full opponent history:** strictly downstream of the ungated one-step likelihood correction (K=2 lookback measured only +0.3-0.5 late skill over K=1 and halves ESS); nothing to build until the one-step version earns a gate.