/* solver.c -- exact endgame play for the last cards of a round.
 *
 * Near deck exhaustion the game tree is tiny and terminal values are exact
 * round margins, yet a policy prior can still be confidently wrong there
 * (observed: 94% on a play that forfeits points with one card left in the
 * deck -- pure arithmetic, no hidden information involved).  This is
 * alpha-beta over a PERFECT-INFORMATION state: the caller determinizes a
 * world first, the solver plays both sides optimally inside it, and the
 * agent averages exact values across belief-sampled worlds (classical PIMC
 * with exact leaves instead of policy playouts).
 *
 * Two bounds keep pathological pile-recycling finite: a ply cap (capped
 * lines return their current margin), and a per-side allowance of one
 * unproductive pile draw per line -- drawing a card you cannot play only
 * stalls, and unlimited recycling explodes the tree while adding nothing
 * an optimal endgame line uses.  A node budget backstops everything; on
 * exhaustion the caller sees a non-positive budget and can fall back to
 * its normal search. */
#include "lc.h"

static int solve_ab(State *st, int p, int alpha, int beta, int cap,
                    long *budget, int stalls0, int stalls1)
{
    if (st->over || st->nply >= cap || --*budget <= 0)
        return lc_score(st, p) - lc_score(st, p ^ 1);
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv);
    if (n == 0)
        return lc_score(st, p) - lc_score(st, p ^ 1);

    /* order: plays before discards, deck draws before pile draws -- ending
     * lines first makes the alpha-beta window close quickly */
    int keyv[MAX_MOVES];
    for (int i = 0; i < n; i++)
        keyv[i] = mv[i].discard * 1000 + (mv[i].draw != 0) * 100
                - CARD_VALUE(mv[i].card) * (1 + st->exp_wager[st->turn][CARD_SUIT(mv[i].card)]);
    for (int i = 1; i < n; i++) {
        Move m = mv[i];
        int key = keyv[i];
        int j = i - 1;
        while (j >= 0 && keyv[j] > key) {
            mv[j + 1] = mv[j];
            keyv[j + 1] = keyv[j];
            j--;
        }
        mv[j + 1] = m;
        keyv[j + 1] = key;
    }

    const int maxing = st->turn == p;
    const int mover = st->turn;
    const uint64_t hand = st->hand[mover];
    int best = maxing ? -32000 : 32000;
    int any = 0;
    for (int i = 0; i < n; i++) {
        /* identical wager copies span isomorphic subtrees: only the lowest
         * held copy is searched */
        int c = mv[i].card;
        if (CARD_IS_WAGER(c) && CARD_RANK(c) > 0 &&
            (hand & (((1ULL << CARD_RANK(c)) - 1) << (CARD_SUIT(c) * NRANK))))
            continue;
        int ns0 = stalls0, ns1 = stalls1;
        if (mv[i].draw > 0) {
            int s2 = mv[i].draw - 1;
            int top = st->pile[s2][st->pile_n[s2] - 1];
            int playable = CARD_IS_WAGER(top)
                               ? st->exp_top[mover][CARD_SUIT(top)] == 0
                               : CARD_VALUE(top) > st->exp_top[mover][CARD_SUIT(top)];
            if (!playable) {
                if (mover == 0) { if (ns0 <= 0) continue; ns0--; }
                else            { if (ns1 <= 0) continue; ns1--; }
            }
        }
        any = 1;
        State s = *st;
        lc_apply(&s, mv[i]);
        int v = solve_ab(&s, p, alpha, beta, cap, budget, ns0, ns1);
        if (maxing) {
            if (v > best) best = v;
            if (best > alpha) alpha = best;
        } else {
            if (v < best) best = v;
            if (best < beta) beta = best;
        }
        if (alpha >= beta) break;
    }
    if (!any) {   /* every legal move was a barred stall: take the first */
        State s = *st;
        lc_apply(&s, mv[0]);
        return solve_ab(&s, p, alpha, beta, cap, budget, stalls0, stalls1);
    }
    return best;
}

/* Exact margin for p from the perfect-information state st.  budget bounds
 * total nodes; on exhaustion remaining lines evaluate at their current
 * margin and *budget goes non-positive so the caller can fall back. */
int lc_solve_budget(const State *st, int p, long *budget)
{
    State s = *st;
    int cap = st->nply + st->deck_left + 6;
    if (cap > LC_MAX_PLIES) cap = LC_MAX_PLIES;
    return solve_ab(&s, p, -32000, 32000, cap, budget, 1, 1);
}

int lc_solve(const State *st, int p)
{
    long budget = 50 * 1000 * 1000L;
    return lc_solve_budget(st, p, &budget);
}
