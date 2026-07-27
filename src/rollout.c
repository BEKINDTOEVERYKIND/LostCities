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

#define MAX_CAND 8

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

/* Play s out to the end, returning the margin for player p. */
static int playout(const Net *net, State *s, int p)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = rank_moves(net, s, mv, score);
        if (n <= 0) break;
        int best = 0;
        for (int i = 1; i < n; i++) if (score[i] > score[best]) best = i;
        lc_apply(s, mv[best]);
    }
    return lc_score(s, p) - lc_score(s, p ^ 1);
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
    } else {
        DrawSamples ds;
        draw_samples_init(st, st->turn, rng, 6, &ds);
        n = lc_moves(st, mv);
        for (int i = 0; i < n; i++) prob[i] = move_value_heur(st, mv[i], &ds);
    }
    if (n <= 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = n;
            if (n == 1) { stats->mv[0] = mv[0]; stats->visits[0] = 1; stats->q[0] = value; }
            stats->value = value;
        }
        return mv[0];
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
    if (a->net) {
        float floor_p = a->cand_floor > 0.0f ? a->cand_floor : 0.02f;
        while (ncand > 1 && prob[order[ncand - 1]] < floor_p) ncand--;
    }

    double sum[MAX_CAND];
    for (int i = 0; i < ncand; i++) sum[i] = 0.0;
    const int p = st->turn;
    int reps = a->dets > 0 ? a->dets : 1;

    for (int d = 0; d < reps; d++) {
        State world;
        determinize(st, p, rng, &world);
        for (int c = 0; c < ncand; c++) {
            State s = world;                 /* same world for every candidate */
            lc_apply(&s, mv[order[c]]);
            sum[c] += playout(a->net, &s, p);
        }
    }

    int best = 0;
    for (int c = 1; c < ncand; c++) if (sum[c] > sum[best]) best = c;
    float bestq = (float)(sum[best] / reps);
    if (stats) {
        stats->n = ncand;
        for (int c = 0; c < ncand; c++) {
            stats->mv[c] = mv[order[c]];
            stats->visits[c] = reps;
            stats->q[c] = sum[c] / reps;
        }
        stats->value = bestq;
    }
    if (out_value) *out_value = bestq;
    return mv[order[best]];
}
