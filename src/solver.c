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
 * its normal search.
 *
 * Transposition table: endgame lines are permutations of a small move
 * multiset, so distinct paths hit identical states constantly (measured
 * before the table: 73M nodes for ONE root move at deck 6 -- deep labeling
 * was infeasible).  A state's expedition arrays are fully derived from the
 * played masks (ascending-play rule), so the key hashes hands, played
 * masks, live pile stacks, the remaining deck slice, mover, nply, the
 * root's ply cap, perspective and the stall allowances -- everything the
 * returned value can depend on.  Entries verify the full 64-bit key, are
 * always-replaced, and persist across solves (stale entries cannot match:
 * the key covers the world).  Thread-local so mining threads don't race. */
#include "lc.h"
#include <stdlib.h>

typedef struct { uint64_t key; int16_t val; uint8_t flag; uint16_t mv; } TTE;
enum { TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };
#define TT_BITS 21
#define TT_SIZE (1UL << TT_BITS)
static _Thread_local TTE *tt_tab;

static inline uint64_t tt_mix(uint64_t h, uint64_t x)
{
    x *= 0x9E3779B97F4A7C15ULL;
    x ^= x >> 32;
    h ^= x;
    h *= 0xFF51AFD7ED558CCDULL;
    return h;
}

static uint64_t tt_hash(const State *st, int p, int cap, int stalls0, int stalls1)
{
    uint64_t h = 0x8000000080004021ULL;
    h = tt_mix(h, st->hand[0]);
    h = tt_mix(h, st->hand[1]);
    h = tt_mix(h, st->played[0]);
    h = tt_mix(h, st->played[1]);
    h = tt_mix(h, st->discarded);
    for (int s = 0; s < NSUIT; s++) {
        /* only live entries: pile_n-- on pickup leaves stale bytes above;
         * chunk every 10 cards -- 12 six-bit ids overflow one word */
        uint64_t w = (uint64_t)st->pile_n[s];
        int k = 0;
        for (int i = 0; i < st->pile_n[s]; i++) {
            w = (w << 6) | st->pile[s][i];
            if (++k == 9) { h = tt_mix(h, w); w = 0; k = 0; }
        }
        h = tt_mix(h, w ^ ((uint64_t)s << 60));
    }
    uint64_t w = 0;
    int k = 0;
    for (int i = 0; i < st->deck_left; i++) {
        w = (w << 6) | st->deck[st->deck_pos + i];
        if (++k == 10) { h = tt_mix(h, w); w = 0; k = 0; }
    }
    h = tt_mix(h, w ^ ((uint64_t)k << 60));
    h = tt_mix(h, (uint64_t)st->turn | ((uint64_t)st->nply << 8) |
                  ((uint64_t)p << 24) | ((uint64_t)cap << 26) |
                  ((uint64_t)stalls0 << 42) | ((uint64_t)stalls1 << 46));
    return h ? h : 1;   /* 0 means "empty slot" */
}

static int solve_ab(State *st, int p, int alpha, int beta, int cap,
                    long *budget, int stalls0, int stalls1)
{
    if (st->over || st->nply >= cap || --*budget <= 0)
        return lc_score(st, p) - lc_score(st, p ^ 1);

    static _Thread_local int tt_off = -1;
    if (tt_off < 0) {
        const char *e = getenv("LC_SOLVER_TT");
        tt_off = e && *e == '0';   /* kill-switch doubles as an A/B lever */
    }
    if (!tt_tab && !tt_off) tt_tab = (TTE *)calloc(TT_SIZE, sizeof(TTE));
    uint64_t key = 0;
    uint16_t ttmv = 0xFFFF;   /* MOVE_PACK(0,0,0) is a real move: sentinel out of range */
    TTE *te = NULL;
    if (tt_tab) {
        key = tt_hash(st, p, cap, stalls0, stalls1);
        te = &tt_tab[key & (TT_SIZE - 1)];
        if (te->key == key) {
            ttmv = te->mv;
            if (te->flag == TT_EXACT) return te->val;
            if (te->flag == TT_LOWER) { if (te->val >= beta) return te->val; if (te->val > alpha) alpha = te->val; }
            else                      { if (te->val <= alpha) return te->val; if (te->val < beta) beta = te->val; }
            if (alpha >= beta) return te->val;
        }
    }
    const int alpha0 = alpha, beta0 = beta;
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
    /* the move that proved best here last time is searched first: even a
     * bound-only entry usually remembers the refutation */
    if (ttmv != 0xFFFF) {
        for (int i = 0; i < n; i++) {
            if (MOVE_PACK(mv[i]) == ttmv) {
                Move tm = mv[i];
                memmove(&mv[1], &mv[0], sizeof(Move) * (size_t)i);
                mv[0] = tm;
                break;
            }
        }
    }

    const int maxing = st->turn == p;
    const int mover = st->turn;
    const uint64_t hand = st->hand[mover];
    int best = maxing ? -32000 : 32000;
    uint16_t bestmv = 0xFFFF;
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
            if (v > best) { best = v; bestmv = MOVE_PACK(mv[i]); }
            if (best > alpha) alpha = best;
        } else {
            if (v < best) { best = v; bestmv = MOVE_PACK(mv[i]); }
            if (best < beta) beta = best;
        }
        if (alpha >= beta) break;
    }
    if (!any) {   /* every legal move was a barred stall: take the first */
        State s = *st;
        lc_apply(&s, mv[0]);
        return solve_ab(&s, p, alpha, beta, cap, budget, stalls0, stalls1);
    }
    if (te && *budget > 0) {
        /* budget guard: it only ever decreases, so positive here proves no
         * child was budget-truncated -- a truncated value stored in a table
         * that outlives this solve would corrupt a later one.  Fail-soft
         * bound classification is node-type agnostic: a result outside the
         * ORIGINAL window is a bound, inside it is exact */
        te->key = key;
        te->val = (int16_t)best;
        te->flag = best <= alpha0 ? TT_UPPER : (best >= beta0 ? TT_LOWER : TT_EXACT);
        te->mv = bestmv;
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

/* One alpha-beta from the root for the MOVER: returns the exact value of
 * optimal play in this world and writes the move achieving it.  Alpha rises
 * as moves resolve, so refuted moves cost a fraction of a full-window
 * solve -- this is what makes per-world exact labeling affordable where
 * per-move exact averaging is not.  On budget exhaustion (*budget <= 0)
 * the result is truncated and the caller must discard it. */
int lc_solve_root(const State *st, long *budget, Move *out)
{
    int cap = st->nply + st->deck_left + 6;
    if (cap > LC_MAX_PLIES) cap = LC_MAX_PLIES;
    const int p = st->turn;
    Move mv[MAX_MOVES];
    State root = *st;
    int n = lc_moves(&root, mv);
    if (n <= 0) return 0;
    *out = mv[0];

    int keyv[MAX_MOVES];
    for (int i = 0; i < n; i++)
        keyv[i] = mv[i].discard * 1000 + (mv[i].draw != 0) * 100
                - CARD_VALUE(mv[i].card) * (1 + st->exp_wager[p][CARD_SUIT(mv[i].card)]);
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

    const uint64_t hand = st->hand[p];
    int best = -32000, alpha = -32000, any = 0;
    for (int i = 0; i < n; i++) {
        int c = mv[i].card;
        if (CARD_IS_WAGER(c) && CARD_RANK(c) > 0 &&
            (hand & (((1ULL << CARD_RANK(c)) - 1) << (CARD_SUIT(c) * NRANK))))
            continue;
        int ns0 = 1, ns1 = 1;
        if (mv[i].draw > 0) {
            int s2 = mv[i].draw - 1;
            int top = st->pile[s2][st->pile_n[s2] - 1];
            int playable = CARD_IS_WAGER(top)
                               ? st->exp_top[p][CARD_SUIT(top)] == 0
                               : CARD_VALUE(top) > st->exp_top[p][CARD_SUIT(top)];
            if (!playable) ns0 = 0;   /* the mover spent the allowance */
        }
        if (p == 1) { int t = ns0; ns0 = ns1; ns1 = t; }
        any = 1;
        State s = *st;
        lc_apply(&s, mv[i]);
        int v = solve_ab(&s, p, alpha, 32000, cap, budget, ns0, ns1);
        if (v > best) { best = v; *out = mv[i]; }
        if (best > alpha) alpha = best;
        if (*budget <= 0) return best;
    }
    if (!any) {
        State s = *st;
        lc_apply(&s, mv[0]);
        *out = mv[0];
        return solve_ab(&s, p, -32000, 32000, cap, budget, 1, 1);
    }
    return best;
}
