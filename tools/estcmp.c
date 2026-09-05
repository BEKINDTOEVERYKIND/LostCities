/* estcmp -- leaf-role estimator error of the tournament search.
 *
 * On stored search-play states (deck 6-14, nply >= 14, >= 2 tournament
 * candidates) the play-time estimator ranks the policy's top-5 by paired
 * greedy playouts of the main net.  This tool scores the same candidates
 * with two references that share nothing with that leaf policy's habits:
 *   ref A: W belief worlds, each continued to the round end by the STANDING
 *          SEARCH AGENT playing both seats (worlds paired across candidates);
 *   ref B: the search's own estimator with SAMPLED playouts at many worlds
 *          (the calib.c oracle form).
 * One TSV row per (state, non-top candidate): dQ_est, se_est, dQ_refA,
 * se_refA, dQ_refB, se_refB, plus the state's deck/ply and the moves.
 *
 *   estcmp EST_SPEC REFB_SPEC corpus.bst FIRST COUNT WORLDS_A SEED OUT.tsv
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define BST_MAGIC 0x4C424554u
typedef struct { State st; uint16_t game; } Rec;
static int same_move(Move a, Move b) {
    int ca = a.card, cb = b.card;
    int sc = ca == cb || (CARD_IS_WAGER(ca) && CARD_IS_WAGER(cb) && CARD_SUIT(ca) == CARD_SUIT(cb));
    return sc && a.discard == b.discard && a.draw == b.draw;
}
static int find_cand(const SearchStats *s, Move m) { for (int i = 0; i < s->n; i++) if (same_move(s->mv[i], m)) return i; return -1; }
static void mvname(Move m, char *out) {
    char c[8]; lc_card_name(m.card, c);
    sprintf(out, "%s%s%d", c, m.discard ? "d" : "p", m.draw);
}
int main(int argc, char **argv) {
    if (argc < 9) { fprintf(stderr, "usage: estcmp EST_SPEC REFB_SPEC corpus.bst FIRST COUNT WORLDS_A SEED OUT.tsv\n"); return 1; }
    Agent est, refb; spec_parse(argv[1], &est); spec_parse(argv[2], &refb);
    FILE *f = fopen(argv[3], "rb"); uint32_t h[2]; uint64_t count;
    if (!f || fread(h, sizeof h, 1, f) != 1 || fread(&count, sizeof count, 1, f) != 1 || h[0] != BST_MAGIC || h[1] != sizeof(Rec)) { fprintf(stderr, "bad corpus\n"); return 1; }
    long first = atol(argv[4]), want = atol(argv[5]); int WA = atoi(argv[6]);
    uint64_t seed = strtoull(argv[7], NULL, 10);
    FILE *out = fopen(argv[8], "w");
    fprintf(out, "state\tdeck\tnply\tcand\tmove0\tmove\tprio\tdQ_est\tse_est\tdQ_refA\tse_refA\tdQ_refB\tse_refB\n");
    Rng rng; rng_seed(&rng, seed);
    long idx = 0, done = 0; Rec r;
    const Net *bnet = est.net_b ? est.net_b : est.net;
    while (fread(&r, sizeof r, 1, f) == 1 && done < want) {
        const State *st = &r.st;
        if (st->over || st->nply < 14 || st->deck_left > 14 || st->deck_left < 6) continue;
        if (idx++ < first) continue;
        const int p = st->turn;
        SearchStats se_, sb_; float v;
        Rng r1; rng_seed(&r1, seed ^ (0x9E37ULL * (uint64_t)(idx + 1)));
        rollout_move(&est, st, &r1, &v, &se_);
        if (se_.n < 2) continue;
        Rng r2; rng_seed(&r2, seed ^ (0xC2B2ULL * (uint64_t)(idx + 1)));
        rollout_move(&refb, st, &r2, &v, &sb_);
        /* ref A: WA worlds, standing agent both seats, paired across candidates */
        double sumA[MAX_MOVES] = {0}, sqA[MAX_MOVES] = {0};
        double *valA = calloc((size_t)se_.n * WA, sizeof(double));
        for (int w = 0; w < WA; w++) {
            State world;
            Rng wr; rng_seed(&wr, seed ^ (0xD1B5ULL * (uint64_t)(idx + 1)) ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(w + 1)));
            determinize_bm(st, p, &wr, bnet, est.bx, est.bel_samp, est.sym_bel, &world);
            for (int c = 0; c < se_.n; c++) {
                State s = world;
                lc_apply(&s, se_.mv[c]);
                Rng cr; rng_seed(&cr, seed ^ (0xA5A5ULL * (uint64_t)(idx + 1)) ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(w + 1)));
                int guard = 0;
                while (!s.over && guard++ < 400) { Move m = rollout_move(&est, &s, &cr, NULL, NULL); lc_apply(&s, m); }
                double mg = lc_score(&s, p) - lc_score(&s, p ^ 1);
                valA[(size_t)c * WA + w] = mg;
                sumA[c] += mg; sqA[c] += mg * mg;
            }
        }
        char m0[16], mc[16]; mvname(se_.mv[0], m0);
        for (int c = 1; c < se_.n; c++) {
            mvname(se_.mv[c], mc);
            double dA = (sumA[c] - sumA[0]) / WA, vA = 0;
            for (int w = 0; w < WA; w++) { double x = valA[(size_t)c * WA + w] - valA[w] - dA; vA += x * x; }
            double seA = WA > 1 ? sqrt(vA / (WA - 1) / WA) : 0;
            int cb = find_cand(&sb_, se_.mv[c]), c0b = find_cand(&sb_, se_.mv[0]);
            double dB = (cb >= 0 && c0b >= 0) ? sb_.q[cb] - sb_.q[c0b] : NAN;
            double seB = (cb >= 0) ? sb_.se[cb] : NAN;
            fprintf(out, "%ld\t%d\t%d\t%d\t%s\t%s\t%.3f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n",
                    idx, st->deck_left, st->nply, c, m0, mc, se_.prio[c], se_.q[c] - se_.q[0], se_.se[c], dA, seA, dB, seB);
        }
        fflush(out);
        free(valA);
        done++;
        fprintf(stderr, "state %ld (deck %d) done: %d candidates\n", idx, st->deck_left, se_.n);
    }
    fclose(out);
    return 0;
}
