#include "features.h"

void feat_extract(const State *st, int p, Features *f)
{
    const int o = p ^ 1;
    int n = 0;

    /* --- card planes -------------------------------------------------- */
    uint64_t mask;
    mask = st->hand[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(0 * NCARD + c); }
    mask = st->played[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(1 * NCARD + c); }
    mask = st->played[o];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(2 * NCARD + c); }
    mask = st->discarded;
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(3 * NCARD + c); }
    for (int s = 0; s < NSUIT; s++)
        if (st->pile_n[s] > 0) f->idx[n++] = (uint16_t)(4 * NCARD + st->pile[s][st->pile_n[s] - 1]);
    /* cards taken face up: I know the opponent holds these, and they know I
     * hold mine -- both facts shape what is worth discarding or racing for */
    mask = st->known[o];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(5 * NCARD + c); }
    mask = st->known[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; f->idx[n++] = (uint16_t)(6 * NCARD + c); }
    f->nidx = n;

    /* --- dense ---------------------------------------------------------- */
    float *d = f->dense;
    for (int i = 0; i < FEAT_DENSE; i++) d[i] = 0.0f;

    uint64_t unseen = ~(st->hand[p] | st->played[0] | st->played[1] | st->discarded
                        | st->known[o])
                      & ((1ULL << NCARD) - 1);

    int my_started = 0, op_started = 0, my_score = 0, op_score = 0;
    int nplay_total = 0;
    int play_cnt_s[NSUIT], uns_above_me[NSUIT], uns_above_op[NSUIT];

    for (int s = 0; s < NSUIT; s++) {
        float *v = d + s * SUIT_FEATS;
        int mytop = st->exp_top[p][s], optop = st->exp_top[o][s];
        int msc = lc_exp_score(st, p, s), osc = lc_exp_score(st, o, s);
        my_score += msc; op_score += osc;
        if (st->exp_n[p][s]) my_started++;
        if (st->exp_n[o][s]) op_started++;

        v[0]  = st->exp_n[p][s] ? 1.0f : 0.0f;
        v[1]  = st->exp_wager[p][s] * (1.0f / 3.0f);
        v[2]  = st->exp_n[p][s] * (1.0f / 12.0f);
        v[3]  = st->exp_sum[p][s] * (1.0f / 54.0f);
        v[4]  = mytop * (1.0f / 10.0f);
        v[5]  = msc * (1.0f / 50.0f);
        v[6]  = st->exp_n[o][s] ? 1.0f : 0.0f;
        v[7]  = st->exp_wager[o][s] * (1.0f / 3.0f);
        v[8]  = st->exp_n[o][s] * (1.0f / 12.0f);
        v[9]  = st->exp_sum[o][s] * (1.0f / 54.0f);
        v[10] = optop * (1.0f / 10.0f);
        v[11] = osc * (1.0f / 50.0f);
        v[12] = st->pile_n[s] * (1.0f / 12.0f);
        if (st->pile_n[s] > 0) {
            int tc = st->pile[s][st->pile_n[s] - 1];
            v[13] = CARD_VALUE(tc) * (1.0f / 10.0f);
            v[14] = CARD_IS_WAGER(tc) ? 1.0f : 0.0f;
        }

        int hand_cnt = 0, play_cnt = 0, play_sum = 0, hand_wag = 0;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
            if ((st->hand[p] >> c) & 1ULL) {
                hand_cnt++;
                if (CARD_IS_WAGER(c)) {
                    hand_wag++;
                    if (mytop == 0) { play_cnt++; }
                } else if (CARD_VALUE(c) > mytop) {
                    play_cnt++; play_sum += CARD_VALUE(c);
                }
            }
        }
        play_cnt_s[s] = play_cnt;
        nplay_total += play_cnt;
        v[15] = hand_cnt * (1.0f / 12.0f);
        v[16] = play_cnt * (1.0f / 12.0f);
        v[17] = play_sum * (1.0f / 54.0f);
        v[18] = hand_wag * (1.0f / 3.0f);

        int uns_cnt = 0, uns_mine = 0, uns_opp = 0, uns_nme = 0, uns_nop = 0;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
            if ((unseen >> c) & 1ULL) {
                uns_cnt++;
                int val = CARD_VALUE(c);
                if (!CARD_IS_WAGER(c)) {
                    if (val > mytop) { uns_mine += val; uns_nme++; }
                    if (val > optop) { uns_opp += val; uns_nop++; }
                }
            }
        }
        uns_above_me[s] = uns_nme;
        uns_above_op[s] = uns_nop;
        v[19] = uns_cnt * (1.0f / 12.0f);
        v[20] = uns_mine * (1.0f / 54.0f);
        v[21] = uns_opp * (1.0f / 54.0f);
        /* how many more cards this expedition needs for the 8 card bonus */
        v[22] = st->exp_n[p][s] ? (8 - st->exp_n[p][s] > 0 ? (8 - st->exp_n[p][s]) * 0.125f : 0.0f) : 0.0f;
        v[23] = st->exp_n[o][s] ? (8 - st->exp_n[o][s] > 0 ? (8 - st->exp_n[o][s]) * 0.125f : 0.0f) : 0.0f;
    }

    float *g = d + NSUIT * SUIT_FEATS;
    g[0]  = st->deck_left * (1.0f / 44.0f);
    g[1]  = (st->turn == p) ? 1.0f : 0.0f;
    g[2]  = my_score * (1.0f / 50.0f);
    g[3]  = op_score * (1.0f / 50.0f);
    g[4]  = my_started * 0.2f;
    g[5]  = op_started * 0.2f;
    g[6]  = st->nply * 0.01f;
    g[7]  = st->hand_n[p] * 0.125f;
    g[8]  = st->hand_n[o] * 0.125f;
    g[9]  = 1.0f;
    g[10] = st->deck_left <= 5 ? 1.0f : 0.0f;
    g[11] = st->deck_left <= 12 ? 1.0f : 0.0f;
    /* match context: which round this is and where the match stands */
    g[12] = st->round == 0 ? 1.0f : 0.0f;
    g[13] = st->round == 1 ? 1.0f : 0.0f;
    g[14] = st->round >= 2 ? 1.0f : 0.0f;
    float cm = (float)(st->cum[p] - st->cum[o]) * 0.01f;
    if (cm > 1.5f) cm = 1.5f;
    if (cm < -1.5f) cm = -1.5f;
    g[15] = cm;

    /* --- turn arithmetic (layout documented in features.h) ------------
     * The round ends when deck_left reaches 0 after a draw: with only deck
     * draws the mover gets (deck_left+1)/2 more turns and draws last iff
     * deck_left is odd; a pile draw leaves deck_left alone and so hands
     * the last draw to the other side. */
    float *t = d + FEAT_DENSE_V5;
    const int dl = st->deck_left;
    const int mover = (st->turn == p);
    const int mover_turns = (dl + 1) / 2, other_turns = dl / 2;
    const int my_turns = mover ? mover_turns : other_turns;
    const int opp_turns = mover ? other_turns : mover_turns;
    const int last_mine = (dl & 1) ^ (mover ? 0 : 1);
    t[0] = my_turns * (1.0f / 22.0f);
    t[1] = opp_turns * (1.0f / 22.0f);
    t[2] = last_mine ? 1.0f : 0.0f;
    if (dl <= 14) t[3 + dl] = 1.0f;
    t[18] = nplay_total * 0.125f;
    int press = nplay_total - my_turns;
    if (press > 8) press = 8;
    if (press < -8) press = -8;
    t[19] = press * 0.125f;
    t[20] = last_mine ? 0.0f : 1.0f;
    for (int s = 0; s < NSUIT; s++) {
        float *u = t + 21 + 3 * s;
        u[0] = uns_above_me[s] * (1.0f / 9.0f);
        u[1] = uns_above_op[s] * (1.0f / 9.0f);
        int ps = play_cnt_s[s] - my_turns;
        if (ps > 8) ps = 8;
        if (ps < -8) ps = -8;
        u[2] = ps * 0.125f;
    }
}
