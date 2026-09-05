/* rollout.c -- policy improvement by playing candidate moves out.
 *
 * The policy network is sharp, so a turn usually has two to four moves worth
 * considering.  For each of them we sample a world consistent with what the
 * mover knows -- the opponent's hand and the deck order -- and play the game to
 * the end with the same policy driving both seats, then compare the final
 * margins.
 *
 * Two properties make this work where the value network did not:
 *
 *  - The estimate comes from real finished games, so it never inherits the
 *    value head's inability to separate moves that differ by a point or two.
 *  - Each sampled world is shared by every candidate, and the playouts are
 *    deterministic given the world, so the *difference* between candidates is
 *    measured on identical futures.  That pairing is what makes a few hundred
 *    samples enough to resolve small differences.
 *
 * Rollouts also avoid the strategy fusion that spoils determinized tree search:
 * inside a sampled world each side still chooses from its own information set,
 * because the policy only ever sees the features of the player to move.
 */
#include "search.h"
#include "agent.h"
#include "heuristic.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_CAND 8
/* racing deepening (sel_deep=2): an eligible candidate survives into the
 * second world batch only while its batch-1 deficit against the batch-1
 * leader is under this many paired standard errors */
#define RACE_K 1.5

/* Rank the legal moves of s for the player to move.  With a network that is
 * the policy head; without one it is the hand-crafted evaluation, which gives
 * the classical "heuristic + perfect-information Monte Carlo" baseline. */
static int rank_moves(const Net *net, const State *s, Move *mv, float *score)
{
    if (net) return policy_probs(net, s, mv, score, NULL);
    int n = lc_moves(s, mv);
    for (int i = 0; i < n; i++) score[i] = heur_move_value_det(s, mv[i]);
    return n;
}

/* Play s out to the end of the round, returning the round margin for player
 * p.  In the final round of a match the round's end decides the match, so
 * *winpts gets the match result (1 win, 0.5 draw, 0 loss) from the carried
 * cumulative totals; in earlier rounds it gets -1 (margin is the only
 * available objective there, and it doubles as the natural proxy).
 * srng != NULL samples the policy instead of argmaxing it: deterministic
 * playouts repeat every knife-edge downstream decision identically across
 * paired worlds, which can manufacture large fake Q gaps with tiny paired
 * errors; sampling breaks that correlation. */
static int playout(const Net *net, State *s, int p, int prune, Rng *srng,
                   double *winpts)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = rank_moves(net, s, mv, score);
        if (n <= 0) break;
        /* fold identical wager copies here too: the argmax below otherwise
         * compares each copy's SPLIT probability against undivided rivals
         * and systematically underplays wagers in every playout */
        if (n > 1) n = lc_dedup_wagers(s, mv, score, n, net != NULL);
        uint64_t dead = prune ? (lc_dead_cards(s) & s->hand[s->turn]) : 0;
        int best = -1;
        if (srng && net) {
            float w[MAX_MOVES];
            float tot = 0.0f;
            for (int i = 0; i < n; i++) {
                w[i] = (dead && lc_discard_dominated(s, mv[i], dead)) ? 0.0f : score[i];
                tot += w[i];
            }
            if (tot > 0.0f) best = sample_index(w, n, srng);
        }
        if (best < 0) {
            for (int i = 0; i < n; i++) {
                if (dead && lc_discard_dominated(s, mv[i], dead)) continue;
                if (best < 0 || score[i] > score[best]) best = i;
            }
        }
        if (best < 0) best = 0;
        lc_apply(s, mv[best]);
    }
    int sp = lc_score(s, p), so = lc_score(s, p ^ 1);
    if (winpts) {
        if (s->round == MATCH_ROUNDS - 1) {
            int tp = s->cum[p] + sp, to = s->cum[p ^ 1] + so;
            *winpts = tp > to ? 1.0 : (tp == to ? 0.5 : 0.0);
        } else *winpts = -1.0;
    }
    return sp - so;
}


/* world sampling dispatch: extended-format specialist wins if configured */
static void sample_world(const struct Agent *a, const State *st, int p,
                         Rng *rng, State *out)
{
    if (a->omni) { determinize_omni(st, p, rng, out); return; }   /* measurement only */
    if (a->no_belief) { determinize_b(st, p, rng, NULL, out); return; }
    if (a->bel_samp > 0) {
        determinize_bm(st, p, rng, a->net_b ? a->net_b : a->net, a->bx, a->bel_samp, a->sym_bel, out);
        return;
    }
    if (a->sym_bel > 0) {
        determinize_bsym(st, p, rng, a->net_b ? a->net_b : a->net, a->bx, a->sym_bel, out);
        return;
    }
    if (a->bx) { determinize_bx(st, p, rng, a->bx, out); return; }
    determinize_b(st, p, rng, a->net_b ? a->net_b : a->net, out);
}

/* is the top card of discard pile s playable by player p right now?
 * (wager: blocked by any own number in the suit; number: blocked by any
 * own played number of equal-or-higher rank) */
static int pile_top_playable(const State *st, int p, int s)
{
    if (st->pile_n[s] == 0) return 0;
    int card = st->pile[s][st->pile_n[s] - 1];
    int r = card % 12;
    uint32_t nums = (uint32_t)(st->played[p] >> (s * 12 + 3)) & 0x1FFu;
    if (r < 3) return nums == 0;
    return (nums >> (r - 3)) == 0;
}


/* test-time symmetrization (sym_k, spec field 25): the rules are invariant
 * to relabeling the five suits and a suit's three wager copies, but the
 * trained net is only approximately so (measured on the c20 champion over
 * 3,845 argmax self-play states: the top move's probability moves with an
 * SD of 0.08 across relabelings, the raw argmax flips on 25% of them, and
 * the K=8-averaged policy disagrees with the raw one on 17% of states --
 * 36% when the raw top is under 0.40).  Everything downstream -- the
 * candidate set, the 2% floor, candidate 0's identity for the sel_k gate,
 * the ply-window policy move -- was inheriting that label noise.  Average
 * the policy (and value) over K random relabelings, mapping every move
 * back through the inverse map with wager copies folded; the folded mass
 * is split evenly over the copies actually held so the dedup fold below
 * recovers it exactly.  Cost: K forward passes per decision, nothing
 * beside the playouts.  Playouts themselves stay raw (K forwards per
 * playout ply would not). */
static int sym_key(Move m)
{
    int c = m.card;
    if (CARD_IS_WAGER(c)) c = CARD_SUIT(c) * NRANK;
    return c + 60 * m.discard + 120 * m.draw;
}

static void symmetrize_priors(const Net *net, const State *st, Rng *rng_unused, int K,
                              Move *mv, float *prob, int n, float *value)
{
    static _Thread_local float acc[720];
    static _Thread_local uint8_t cnt[720];
    /* relabelings are drawn from a stream seeded by the information set:
     * the symmetrized prior is then a fixed function of the state (a
     * display and the decision agree exactly) and the match Rng stream is
     * not consumed by symmetrization */
    (void)rng_unused;
    Rng lrng;
    rng_seed(&lrng, infoset_hash(st, st->turn) ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(K + 1)));
    Rng *rng = &lrng;
    /* K >= LC_SYM_EXACT: the 120 suit relabelings are enumerated and the
     * raw (identity) term is not added on top, so every suit labeling has
     * the same weight and the prior is exactly suit-invariant */
    const int exact = K >= LC_SYM_EXACT, R = lc_sym_count(K);
    memset(acc, 0, sizeof acc);
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < n; i++) { int k = sym_key(mv[i]); if (!exact) acc[k] += prob[i]; cnt[k]++; }
    double vsum = exact ? 0.0 : *value;
    int m = exact ? 0 : 1;
    for (int k = 0; k < R; k++) {
        int sp[NSUIT], wp[NSUIT][WAGERS_PER_SUIT];
        lc_sym_relabel(rng, K, k, sp, wp);
        uint8_t map[NCARD], inv[NCARD];
        lc_perm_map(sp, wp, map);
        for (int c = 0; c < NCARD; c++) inv[map[c]] = (uint8_t)c;
        int invsuit[NSUIT];
        for (int s = 0; s < NSUIT; s++) invsuit[sp[s]] = s;
        State ps = *st;
        lc_permute(&ps, map);
        Move pm[MAX_MOVES];
        float pp[MAX_MOVES], pv = 0.0f;
        int pn = policy_probs(net, &ps, pm, pp, &pv);
        if (pn <= 0) continue;
        for (int i = 0; i < pn; i++) {
            Move b;
            b.card = inv[pm[i].card];
            b.discard = pm[i].discard;
            b.draw = pm[i].draw == 0 ? 0 : (uint8_t)(invsuit[pm[i].draw - 1] + 1);
            acc[sym_key(b)] += pp[i];
        }
        vsum += pv;
        m++;
    }
    if (m == 0) return;
    for (int i = 0; i < n; i++) {
        int k = sym_key(mv[i]);
        prob[i] = acc[k] / (float)m / (float)(cnt[k] ? cnt[k] : 1);
    }
    *value = (float)(vsum / m);
}

/* The policy prior (and value) exactly as this agent's candidate stage
 * sees them: raw policy_probs, symmetrized over sym_k relabelings when the
 * agent is configured so.  For displays that must show what the deciding
 * agent used rather than the raw head. */
/* The value head from perspective q, with the real player to move, averaged
 * over the same state-seeded relabelings when sym_k is set.  (Flipping the
 * turn to q and reading the mover's value is a different, off-distribution
 * quantity: +3-4 points on average.) */
float agent_value(const struct Agent *a, const State *st, int q)
{
    Features f;
    feat_extract(st, q, &f);
    double v = net_value(a->net, &f) * VAL_SCALE;
    int K = a->sym_k;
    if (K <= 0) return (float)v;
    const int exact = K >= LC_SYM_EXACT, R = lc_sym_count(K);
    if (exact) v = 0.0;
    Rng lrng;
    rng_seed(&lrng, infoset_hash(st, q) ^ (0xC2B2AE3D27D4EB4FULL * (uint64_t)(K + 1)));
    for (int k = 0; k < R; k++) {
        int sp[NSUIT], wp[NSUIT][WAGERS_PER_SUIT];
        lc_sym_relabel(&lrng, K, k, sp, wp);
        uint8_t map[NCARD];
        lc_perm_map(sp, wp, map);
        State ps = *st;
        lc_permute(&ps, map);
        feat_extract(&ps, q, &f);
        v += net_value(a->net, &f) * VAL_SCALE;
    }
    return (float)(v / (R + (exact ? 0 : 1)));
}

int agent_policy_probs(const struct Agent *a, const State *st, Rng *rng,
                       Move *mv, float *prob, float *value)
{
    float v = 0.0f;
    int n = policy_probs(a->net, st, mv, prob, &v);
    if (a->sym_k > 0 && n > 1) symmetrize_priors(a->net, st, rng, a->sym_k, mv, prob, n, &v);
    if (value) *value = v;
    return n;
}

Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats)
{
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    float value = 0.0f;
    int n;
    if (a->net) {
        n = policy_probs(a->net, st, mv, prob, &value);
        if (a->sym_k > 0 && n > 1)
            symmetrize_priors(a->net, st, rng, a->sym_k, mv, prob, n, &value);
    } else {
        DrawSamples ds;
        draw_samples_init(st, st->turn, rng, 6, &ds);
        n = lc_moves(st, mv);
        for (int i = 0; i < n; i++) prob[i] = move_value_heur(st, mv[i], &ds);
    }
    /* identical wager copies generate isomorphic moves: keep the lowest held
     * copy per (suit, disposition, draw) and give it the siblings' policy
     * mass, so duplicates neither waste candidate slots nor split priors
     * (heuristic scores are values, not mass -- those are not summed) */
    if (n > 1)
        n = lc_dedup_wagers(st, mv, prob, n, a->net != NULL);
    /* dominated-discard pruning: with a dead card in hand, gifting any live
     * card (same draw) is a strictly worse class of move -- drop those before
     * they cost candidate slots or playout gifts */
    if (a->prune_dom && n > 1) {
        uint64_t dead = lc_dead_cards(st);
        if (dead & st->hand[st->turn]) {
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (lc_discard_dominated(st, mv[i], dead)) continue;
                mv[k] = mv[i];
                prob[k] = prob[i];
                k++;
            }
            if (k > 0) n = k;
        }
    }
    /* NOTE: a prune of opponent-playable wager discards was tried here and
     * REVERTED at the reviewer's direction: the class is sometimes correct
     * (every alternative worse, or an endgame where the pickup cannot be
     * afforded), and category bans are the wrong tool.  The wager-gift
     * blunders (probes c13m_gift_19, selk_gift_13) are POLICY weaknesses,
     * recorded in the suite until the policy itself learns them. */
    if (n <= 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = n;
            if (n == 1) {
                stats->mv[0] = mv[0]; stats->visits[0] = 1; stats->q[0] = value;
                stats->se[0] = 0.0; stats->qw[0] = -1.0;
                stats->prio[0] = n == 1 ? 1.0 : -1.0;
            }
            stats->value = value;
        }
        return mv[0];
    }

    /* exact endgame: with few deck cards the whole tree is solvable, so
     * every legal move gets an exact value inside each sampled world and
     * the argmax of the averages plays -- no priors, no playout noise, and
     * no way for a confidently-wrong policy to forfeit terminal points.
     * One node budget covers the WHOLE decision (all moves x all worlds):
     * a rare wide-hand endgame can blow any per-solve cap by hours, so on
     * exhaustion the decision falls through to the normal search instead
     * of trusting truncated bounds */
    if (a->solve_deck > 0 && st->deck_left <= a->solve_deck) {
        double ssum[MAX_MOVES];
        for (int i = 0; i < n; i++) ssum[i] = 0.0;
        const int sp = st->turn;
        int sreps = a->dets > 0 ? a->dets : 1;
        if (sreps > 32) sreps = 32;
        /* labeling runs may raise the per-decision budget by env; match
         * play keeps the 4M default (LC_SOLVE_BUDGET, nodes) */
        static long sbudget_cfg = -1;
        if (sbudget_cfg < 0) {
            const char *e = getenv("LC_SOLVE_BUDGET");
            sbudget_cfg = e ? atol(e) : 4 * 1000 * 1000L;
            if (sbudget_cfg < 100000) sbudget_cfg = 100000;
        }
        long sbudget = a->solve_budget > 0 ? a->solve_budget : sbudget_cfg;
        /* labeling mode: one root solve per world, vote for the PV move
         * (see agent.h solve_vote).  Worlds that exhaust the shared budget
         * don't vote; too few completed worlds falls through to search. */
        if (a->solve_vote) {
            int votes[MAX_MOVES];
            double vsum[MAX_MOVES];
            for (int i = 0; i < n; i++) { votes[i] = 0; vsum[i] = 0.0; }
            int vreps = sreps < 9 ? sreps : 9;
            int done = 0;
            for (int d = 0; d < vreps && sbudget > 0; d++) {
                State world;
                sample_world(a, st, sp, rng, &world);
                Move bm;
                int v = lc_solve_root(&world, &sbudget, &bm);
                if (sbudget <= 0) break;
                /* map the PV move onto the deduped list modulo the
                 * wager-copy isomorphism: the solver searches the lowest
                 * held copy, the dedup fold may have kept another */
                int idx = -1;
                for (int i = 0; i < n && idx < 0; i++) {
                    int same_card = mv[i].card == bm.card ||
                                    (CARD_IS_WAGER(mv[i].card) && CARD_IS_WAGER(bm.card) &&
                                     CARD_SUIT(mv[i].card) == CARD_SUIT(bm.card) &&
                                     mv[i].discard == bm.discard);
                    if (same_card && mv[i].discard == bm.discard && mv[i].draw == bm.draw)
                        idx = i;
                }
                if (idx >= 0) { votes[idx]++; vsum[idx] += v; done++; }
            }
            if (done >= 3) {
                int vb = 0;
                for (int i = 1; i < n; i++)
                    if (votes[i] > votes[vb] ||
                        (votes[i] == votes[vb] && vsum[i] > vsum[vb])) vb = i;
                if (out_value) *out_value = votes[vb] ? (float)(vsum[vb] / votes[vb]) : value;
                if (stats) {
                    int keep = n < MAX_MOVES ? n : MAX_MOVES;
                    stats->n = keep;
                    for (int i = 0; i < keep; i++) {
                        stats->mv[i] = mv[i];
                        stats->visits[i] = votes[i];
                        stats->q[i] = votes[i] ? vsum[i] / votes[i] : 0.0;
                        stats->se[i] = 0.0;
                        stats->qw[i] = -1.0;
                        stats->prio[i] = -1.0;
                    }
                    stats->value = votes[vb] ? (float)(vsum[vb] / votes[vb]) : value;
                }
                return mv[vb];
            }
            /* not enough exact worlds: fall through to the normal search
             * (skip the per-move averaging path -- if the budget could not
             * finish 3 root solves it cannot finish n x worlds solves) */
        } else {
            int solved = 1;
            /* in the final round the match objective is WINS, not margin: an
             * exact +5 that still loses the match must not beat an exact +12
             * that wins it.  Solved values are noise-free, so the lexicographic
             * (wins, then margin) order is safe here in a way it is not for
             * sampled playouts. */
            const int slast = st->round >= MATCH_ROUNDS - 1;
            const int scumd = (int)st->cum[sp] - (int)st->cum[sp ^ 1];
            double swin[MAX_MOVES];
            for (int i = 0; i < n; i++) swin[i] = 0.0;
            for (int d = 0; d < sreps && solved; d++) {
                State world;
                sample_world(a, st, sp, rng, &world);
                for (int i = 0; i < n; i++) {
                    State s = world;
                    lc_apply(&s, mv[i]);
                    int sm = lc_solve_budget(&s, sp, &sbudget);
                    ssum[i] += sm;
                    if (slast)
                        swin[i] += scumd + sm > 0 ? 1.0 : (scumd + sm == 0 ? 0.5 : 0.0);
                    if (sbudget <= 0) { solved = 0; break; }
                }
            }
            if (solved) {
                int sbest = 0;
                for (int i = 1; i < n; i++) {
                    if (slast) {
                        if (swin[i] > swin[sbest] ||
                            (swin[i] == swin[sbest] && ssum[i] > ssum[sbest])) sbest = i;
                    } else if (ssum[i] > ssum[sbest]) sbest = i;
                }
                if (stats) {
                    int keep = n < MAX_MOVES ? n : MAX_MOVES;
                    stats->n = keep;
                    for (int i = 0; i < keep; i++) {
                        stats->mv[i] = mv[i];
                        stats->visits[i] = sreps;
                        stats->q[i] = ssum[i] / sreps;
                        stats->se[i] = 0.0;
                        stats->qw[i] = -1.0;
                        stats->prio[i] = -1.0;
                    }
                    stats->value = (float)(ssum[sbest] / sreps);
                }
                if (out_value) *out_value = (float)(ssum[sbest] / sreps);
                return mv[sbest];
            }
            }
    }

    /* ply window: outside it the raw policy plays (see agent.h) */
    if ((a->ply_lo > 0 && st->nply < a->ply_lo) ||
        (a->ply_hi > 0 && st->nply >= a->ply_hi)) {
        int top = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[top]) top = i;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->mv[0] = mv[top];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0; stats->qw[0] = -1.0;
            stats->prio[0] = prob[top];
            stats->value = value;
        }
        return mv[top];
    }

    /* confidence gate: when the policy is already near-certain, searching can
     * only confirm it or override it with noise -- return the policy move and
     * spend the compute where decisions are actually contested */
    if (a->gate > 0.0f) {
        int top = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[top]) top = i;
        if (prob[top] >= a->gate) {
            if (out_value) *out_value = value;
            if (stats) {
                stats->n = 1;
                stats->mv[0] = mv[top];
                stats->visits[0] = 0;
                stats->q[0] = value;
                stats->se[0] = 0.0; stats->qw[0] = -1.0;
            stats->prio[0] = prob[top];
                stats->value = value;
            }
            return mv[top];
        }
    }

    /* action-level candidates (draw_filter 3): the policy head factors
     * P(move) = P(action) P(draw), so when it is sure of the action the
     * joint ranking fills the candidate slots with that action's other
     * draw sources instead of other plays.  Rank by action mass (summed
     * over draw sources) and let each action carry its most probable
     * draw: the search compares plays; the draw source is the policy's. */
    if (a->draw_filter >= 3 && a->net && n > 1) {
        float top_p[MAX_MOVES];
        int k = 0;
        for (int i = 0; i < n; i++) {
            int j;
            for (j = 0; j < k; j++)
                if (mv[j].card == mv[i].card && mv[j].discard == mv[i].discard) break;
            if (j < k) {
                if (prob[i] > top_p[j]) { mv[j].draw = mv[i].draw; top_p[j] = prob[i]; }
                prob[j] += prob[i];
            } else {
                mv[k] = mv[i];
                prob[k] = prob[i];
                top_p[k] = prob[i];
                k++;
            }
        }
        n = k;
    }
    /* candidates: the most likely moves, cut off once the policy stops caring */
    int order[MAX_MOVES];
    for (int i = 0; i < n; i++) order[i] = i;
    int ncand = a->root_width < MAX_CAND ? a->root_width : MAX_CAND;
    if (ncand > n) ncand = n;
    for (int i = 0; i < ncand; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (prob[order[j]] > prob[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    int nsorted = ncand;                     /* prefix of order[] that is sorted */
    if (a->net) {
        float floor_p = a->cand_floor > 0.0f ? a->cand_floor : 0.02f;
        int keep = a->min_cand > 1 ? a->min_cand : 1;
        if (keep > ncand) keep = ncand;
        while (ncand > keep && prob[order[ncand - 1]] < floor_p) ncand--;
    }
    /* a lone candidate has no decision to make: return it without the world
     * sweep (decision-preserving -- the sweep could only confirm it; ~3% of
     * searched plies at the 2% floor) */
    if (ncand <= 1 && a->eval_cand <= 0) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->mv[0] = mv[order[0]];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0; stats->qw[0] = -1.0;
            stats->prio[0] = prob[order[0]];
            stats->value = value;
        }
        return mv[order[0]];
    }
    /* advisory candidates: evaluated and reported but never selected, so an
     * analysis dump can show what the search thinks of moves the policy has
     * written off without letting that opinion change the game (forcing the
     * floor open for *selection* measured 42.8% vs the baseline) */
    int neval = ncand;
    if (a->eval_cand > neval) {
        neval = a->eval_cand < nsorted ? a->eval_cand : nsorted;
    }
    /* draw-variant expansion: "same action, another draw" shares every
     * sampled world, and those variants are the alternatives an analyst
     * asks about most -- make sure the two top-prior actions have all
     * their legal draw sources on the board.  But each variant still buys
     * a full dets x playout sweep, and the expansion feeds only the
     * advisory layer, whose sole path to the move is the override (full
     * bar unless ov_draw discounts it) -- the reviewer's verdict is that
     * pricing every draw source of an already-chosen action is compute
     * the top policy plays should get instead (draw_filter=2 drops the
     * expansion entirely; combining that with ov_draw starves ov_draw of
     * the very variants it exists to rescue, so don't). */
    if (a->eval_cand > 0 && a->draw_filter < 2) {
        for (int t = 0; t < 2 && t < neval; t++) {
            Move top = mv[order[t]];
            for (int i = 0; i < n && neval < MAX_CAND; i++) {
                if (mv[i].card != top.card || mv[i].discard != top.discard) continue;
                /* draw_filter (spec field 23): expand a pile-draw variant
                 * only when that pile's top is playable by the mover --
                 * evaluating every pile after a popular play burns playouts
                 * on draws nobody wants (the reviewer's ply-18 note), while
                 * the useful-draw wins this expansion exists for all involve
                 * a top card the mover could play */
                if (a->draw_filter && mv[i].draw != 0 &&
                    !pile_top_playable(st, st->turn, mv[i].draw - 1)) continue;
                int seen = 0;
                for (int c = 0; c < neval; c++) if (order[c] == i) { seen = 1; break; }
                if (!seen) order[neval++] = i;
            }
        }
    }

    double sum[MAX_CAND], sumw[MAX_CAND];
    for (int i = 0; i < neval; i++) { sum[i] = 0.0; sumw[i] = 0.0; }
    const int p = st->turn;
    int reps = a->dets > 0 ? a->dets : 1;
    int lastround = st->round == MATCH_ROUNDS - 1;
    /* sel_deep doubles the worlds on contested plies; val is laid out at
     * the maximum width up front so pooling extends rows in place */
    const int vstride = a->sel_deep ? reps * 2 : reps;
    double *val = (double *)malloc(sizeof(double) * (size_t)neval * (size_t)vstride);
    /* racing deepening (sel_deep=2, see agent.h): the second batch is
     * bought only for the SURVIVORS of the first -- candidate 0 plus every
     * eligible candidate within RACE_K paired SEs of the batch-1 leader.
     * The rest keep their batch-1 statistics (alive[c]=0, nrep[c]=reps)
     * and leave the selection; every survivor pair shares the same 2*reps
     * worlds, so the paired statistics below stay exact.  In modes 0/1
     * every candidate is alive with nrep[c] == reps throughout.  sum1[]
     * snapshots the batch-1 sums so a survivor can still be paired with a
     * batch-1-only candidate over their common worlds (override, stats). */
    const int reps0 = reps;
    int alive[MAX_CAND], nrep[MAX_CAND];
    double sum1[MAX_CAND];
    for (int c = 0; c < neval; c++) { alive[c] = 1; sum1[c] = 0.0; }
    int triggered = 0, nsurv = neval;
    static int race_dbg = -1;
    if (race_dbg < 0) race_dbg = getenv("LC_RACE_DEBUG") != NULL;

    int want = reps, done = 0;
    while (done < want) {
        for (int d = done; d < want; d++) {
            State world;
            sample_world(a, st, p, rng, &world);
            uint64_t wseed = 0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1) ^ rng->s[0];
            for (int c = 0; c < neval; c++) {
                if (!alive[c]) continue;     /* sel_deep=2: dropped after batch 1 */
                State s = world;             /* same world for every candidate */
                lc_apply(&s, mv[order[c]]);
                double w;
                Rng pr;
                if (a->playout_sample) rng_seed(&pr, wseed);   /* same seed per world */
                int m = playout(a->net_p ? a->net_p : a->net, &s, p, a->prune_dom,
                                a->playout_sample ? &pr : NULL, &w);
                if (val) val[(size_t)c * vstride + d] = m;
                sum[c] += m;
                if (w >= 0.0) sumw[c] += w;
            }
        }
        done = want;
        if (done == reps0)
            for (int c = 0; c < neval; c++) sum1[c] = sum[c];
        /* contested-ply deepening (sel_deep, spec field 24): when any
         * eligible candidate outscores the policy top on the first batch,
         * the selection decision is live -- buy a second batch and decide
         * everything on pooled statistics.  More data moves the trade the
         * sel_k bar alone cannot: the pooled SE shrinks, so a candidate
         * with a real small lead qualifies MORE often while a
         * single-batch fluke (the reviewer's ply-16 catch: a move 1.9
         * points worse at 512 worlds played off one 96-world stream)
         * qualifies less.  A fresh-batch veto was tried first and
         * refuted on the probe suite -- it suppressed twice as many
         * reviewer-verified good overrides as the noise it removed, the
         * same signature that sank sel_k=1.5.  Cost: one extra batch
         * only on contested plies. */
        if (a->sel_deep && want == reps) {
            for (int c = 1; c < ncand; c++)
                if (sum[c] > sum[0]) { want = reps * 2; break; }
            triggered = want > reps;
            /* racing (sel_deep=2): the second batch goes to candidate 0
             * and to every eligible candidate whose batch-1 deficit
             * against the batch-1 leader is under RACE_K paired SEs --
             * anyone further back cannot overtake on one more batch and
             * would only buy playouts.  Advisory candidates (c >= ncand)
             * are never raced.  Without a usable SE (one world, or no
             * val buffer) the batch stays full, i.e. mode 1. */
            if (triggered && a->sel_deep >= 2 && val && reps > 1) {
                int lead = 0;
                for (int c = 1; c < ncand; c++) if (sum[c] > sum[lead]) lead = c;
                nsurv = 0;
                for (int c = 0; c < neval; c++) {
                    int keep = c == 0 || c == lead;
                    if (!keep && c < ncand) {
                        double dm = (sum[lead] - sum[c]) / reps;
                        double v2 = 0.0;
                        for (int d = 0; d < reps; d++) {
                            double x = val[(size_t)lead * vstride + d]
                                     - val[(size_t)c * vstride + d] - dm;
                            v2 += x * x;
                        }
                        keep = dm < RACE_K * sqrt(v2 / (reps - 1) / reps);
                    }
                    alive[c] = keep;
                    nsurv += keep;
                }
            }
        }
    }
    for (int c = 0; c < neval; c++) nrep[c] = alive[c] ? done : reps0;
    reps = done;
    if (race_dbg && a->sel_deep)
        fprintf(stderr, "[race] mode %d nply %d ncand %d neval %d trig %d surv %d share %.3f\n",
                a->sel_deep, st->nply, ncand, neval, triggered,
                triggered ? nsurv : 0, triggered ? (double)nsurv / neval : 0.0);

    /* In the final round the playouts decide the match, so pick by match
     * wins with margin as the tiebreak -- a 5% shot at stealing the match
     * outranks a certain narrow loss regardless of expected points.  In
     * earlier rounds margin is all a round-end playout can know. */
    int usew = lastround && a->win_q;
    /* prior-aware selection (prior_w0/w1): candidates compete on
     * EV + lambda(ply)*log(prior), so the EV edge needed to overrule the
     * policy grows with the prior gap -- log(p_top/p_cand) is ~3.2 nats for
     * 95% vs 4% but ~0.2 for 55% vs 45% -- and a low-prior candidate must
     * beat a mid-prior one on the same handicapped score.  lambda is
     * interpolated across the ply so early-game policy trust and endgame
     * search trust can differ.  Margin path only: the final-round win-first
     * rule keeps its measured lexicographic form. */
    double lam = 0.0;
    if (a->prior_w0 != 0.0f || a->prior_w1 != 0.0f) {
        double t = st->nply >= 44 ? 1.0 : (double)st->nply / 44.0;
        lam = a->prior_w0 + (a->prior_w1 - a->prior_w0) * t;
    }
    double pscore[MAX_CAND];
    for (int c = 0; c < neval; c++) {
        double pr = prob[order[c]];
        if (pr < 1e-4) pr = 1e-4;
        pscore[c] = sum[c] / nrep[c] + lam * log(pr);
    }
    int best = 0;
    for (int c = 1; c < ncand; c++) {
        if (!alive[c]) continue;             /* sel_deep=2: out of the race */
        if (usew ? (sumw[c] > sumw[best] ||
                    (sumw[c] == sumw[best] && sum[c] > sum[best]))
                 : (lam != 0.0 ? pscore[c] > pscore[best]
                               : sum[c] > sum[best])) best = c;
    }
    /* selection gate (sel_k, see agent.h): candidate 0 is the policy's top
     * choice after dedup and pruning; any other eligible candidate keeps the
     * move only if its paired margin lead over candidate 0 clears sel_k
     * standard errors.  Among qualifiers the usual rule picks; if none
     * qualify the policy top plays.  The gate is on margins even in win_q
     * mode -- the win signal is coarser, not less noisy. */
    if (a->sel_k > 0.0f && val && reps > 1 && best != 0) {
        int pick = 0;
        for (int c = 1; c < ncand; c++) {
            if (!alive[c]) continue;         /* sel_deep=2: batch-1 only, cannot win */
            double dm = (sum[c] - sum[0]) / reps;
            if (dm <= 0.0) continue;
            double v2 = 0.0;
            for (int d = 0; d < reps; d++) {
                double x = val[(size_t)c * vstride + d] - val[(size_t)0 * vstride + d] - dm;
                v2 += x * x;
            }
            double sed = sqrt(v2 / (reps - 1) / reps);
            /* same-action draw variants clear at half sel_k (see agent.h
             * sel_draw): only the draw source differs from the policy's own
             * top choice, and its prior over draw sources is its least
             * trustworthy output */
            float sk = a->sel_k;
            if (a->sel_draw &&
                mv[order[c]].card == mv[order[0]].card &&
                mv[order[c]].discard == mv[order[0]].discard)
                sk *= 0.5f;
            if (getenv("LC_OV_DEBUG"))
                fprintf(stderr, "[sel] cand %d dm %.2f sed %.2f need >%.2f: %s\n",
                        c, dm, sed, sk * sed,
                        dm > sk * sed ? "QUALIFY" : "reject");
            if (dm <= sk * sed) continue;
            /* under prior-aware selection the candidate must also clear the
             * log-prior handicap, not just the noise gate */
            if (lam != 0.0 && !usew && pscore[c] <= pscore[0]) continue;
            if (pick == 0 ||
                (usew ? (sumw[c] > sumw[pick] ||
                         (sumw[c] == sumw[pick] && sum[c] > sum[pick]))
                      : (lam != 0.0 ? pscore[c] > pscore[pick]
                                    : sum[c] > sum[pick]))) pick = c;
        }
        best = pick;
    }
    /* significance-gated override: an advisory candidate may take the move
     * only when its lead over the eligible best exceeds override_k paired
     * standard errors AND override_min points.  The SE gate rejects noise
     * (what blanket forcing lacked: it overrode on any gap and lost 42.8%);
     * the points gate rejects playout bias, which more worlds sharpen
     * rather than shrink.  The reference is the eligible best, fixed, and
     * the highest-Q qualifier wins -- chaining comparisons through interim
     * winners made the outcome depend on candidate order. */
    if (a->override_k > 0.0f && val && reps > 1) {
        int elig = best;
        for (int c = ncand; c < neval; c++) {
            /* an advisory candidate holds batch-1 worlds only; against a
             * raced survivor the pairing runs over their common prefix */
            const int nr = nrep[c] < nrep[elig] ? nrep[c] : nrep[elig];
            double dm = ((nr == nrep[c] ? sum[c] : sum1[c])
                       - (nr == nrep[elig] ? sum[elig] : sum1[elig])) / nr;
            if (dm <= 0.0) continue;
            double v2 = 0.0;
            for (int d = 0; d < nr; d++) {
                double x = val[(size_t)c * vstride + d] - val[(size_t)elig * vstride + d] - dm;
                v2 += x * x;
            }
            double sed = sqrt(v2 / (nr - 1) / nr);
            /* same-action draw variants may qualify at half k (see agent.h
             * ov_draw): the action is the policy's own choice, only the
             * draw source differs, and their paired SE is structurally
             * inflated by future divergence */
            float k = a->override_k;
            if (a->ov_draw &&
                mv[order[c]].card == mv[order[elig]].card &&
                mv[order[c]].discard == mv[order[elig]].discard)
                k *= 0.5f;
            if (getenv("LC_OV_DEBUG"))
                fprintf(stderr, "[ov] cand %d dm %.2f sed %.2f need >%.2f and >%.2f: %s\n",
                        c, dm, sed, k * sed, a->override_min,
                        (dm > k * sed && dm > a->override_min) ? "QUALIFY" : "reject");
            /* prior-aware selection extends to the advisory layer: these
             * are the lowest-prior candidates of all, so without this they
             * would be the only layer paying no prior tax -- and since the
             * handicap can select an eligible best with a lower raw sum,
             * their bar would actually DROP when lambda rose */
            if (lam != 0.0 && pscore[c] <= pscore[elig]) {
                if (getenv("LC_OV_DEBUG"))
                    fprintf(stderr, "[ov] cand %d blocked by prior handicap "
                            "(pscore %.2f <= %.2f)\n", c, pscore[c], pscore[elig]);
                continue;
            }
            if (dm > k * sed && dm > a->override_min &&
                (lam != 0.0 ? pscore[c] > pscore[best]
                            : sum[c] / nrep[c] > sum[best] / nrep[best])) best = c;
        }
        /* sampled confirmation: a qualifying gap must survive stochastic
         * continuations at half the floor, or it was determinism bias --
         * measured concretely: a +5.0 +- 0.14 argmax gap that collapsed to
         * +0.6 under sampling, from one knife-edge downstream decision
         * repeating across every paired world */
        if (best != elig) {
            /* ov_draw=2: same-action draw variants skip the sampled
             * confirmation.  The sampled gate prices the policy's OWN
             * continuation habits; for a draw-source choice that gate can
             * veto an objectively winning turn-extension because the
             * policy later squanders it (measured: argmax +4.2 +- 0.5 vs
             * sampled -2.0 +- 0.5 on the same decision at 3000 worlds,
             * and the unsampled line demonstrably lost the match).
             * Whether trusting argmax here helps MATCH play is an A/B
             * question, hence a mode rather than the default. */
            int skip_conf = a->ov_draw >= 2 &&
                            mv[order[best]].card == mv[order[elig]].card &&
                            mv[order[best]].discard == mv[order[elig]].discard;
            if (!skip_conf) {
                double ds = 0.0;
                for (int d = 0; d < reps; d++) {
                    State world;
                    sample_world(a, st, p, rng, &world);
                    uint64_t wseed = 0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1) ^ rng->s[0];
                    Rng r1, r2;
                    rng_seed(&r1, wseed);
                    rng_seed(&r2, wseed);
                    State sa = world, sb = world;
                    lc_apply(&sa, mv[order[best]]);
                    lc_apply(&sb, mv[order[elig]]);
                    ds += playout(a->net_p ? a->net_p : a->net, &sa, p, a->prune_dom, &r1, NULL)
                        - playout(a->net_p ? a->net_p : a->net, &sb, p, a->prune_dom, &r2, NULL);
                }
                if (getenv("LC_OV_DEBUG"))
                    fprintf(stderr, "[ov] confirm best %d vs elig %d: sampled ds %.2f need >=%.2f: %s\n",
                            best, elig, ds / reps, 0.5 * a->override_min,
                            (ds / reps < 0.5 * a->override_min) ? "REVERT" : "CONFIRMED");
                if (ds / reps < 0.5 * a->override_min) best = elig;
            }
        }
    }
    float bestq = (float)(sum[best] / nrep[best]);
    if (stats) {
        stats->n = neval;
        for (int c = 0; c < neval; c++) {
            stats->mv[c] = mv[order[c]];
            stats->visits[c] = nrep[c];
            stats->q[c] = sum[c] / nrep[c];
            stats->qw[c] = lastround ? sumw[c] / nrep[c] : -1.0;
            stats->prio[c] = prob[order[c]];
            double v = 0.0;
            /* paired against the chosen move over their common worlds: a
             * dropped candidate (sel_deep=2) has batch-1 worlds only */
            const int nr = nrep[c] < nrep[best] ? nrep[c] : nrep[best];
            if (val && nr > 1) {
                double sc = nr == nrep[c] ? sum[c] : sum1[c];
                double sb = nr == nrep[best] ? sum[best] : sum1[best];
                double mean = (sc - (c == best ? 0.0 : sb)) / nr;
                for (int d = 0; d < nr; d++) {
                    double x = val[(size_t)c * vstride + d]
                             - (c == best ? 0.0 : val[(size_t)best * vstride + d]);
                    v += (x - mean) * (x - mean);
                }
                v = sqrt(v / (nr - 1) / nr);
            }
            stats->se[c] = v;
        }
        stats->value = bestq;
    }
    free(val);
    if (out_value) *out_value = bestq;
    return mv[order[best]];
}
