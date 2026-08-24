/* calib -- data collection for prior-aware override thresholds.
 *
 * Samples decision points from policy self-play and scores every candidate
 * move twice: with the cheap match-strength estimator (96 paired worlds,
 * deterministic playouts -- exactly what the play-time selection sees) and
 * with a far deeper oracle (1024 worlds, SAMPLED playouts, which breaks the
 * knife-edge determinism bias the sampled-confirmation gate was built for).
 * One TSV row per (state, non-top candidate) pair records the priors and both
 * estimates; the analysis side then fits, per prior-gap bucket and ply band,
 * how large a cheap-search edge must be before the oracle agrees the
 * policy's top move is actually inferior -- the empirical threshold surface
 * behind the prior_w0/w1 spec fields.
 *
 *   calib NET GAMES SEED OUT.tsv [SPROB=0.15]
 *
 * Single-threaded by design: run several instances with different seeds in
 * parallel and concatenate the outputs.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s NET GAMES SEED OUT.tsv [SPROB] [ARGMAX]\n", argv[0]);
        return 1;
    }
    const char *netpath = argv[1];
    int games = atoi(argv[2]);
    uint64_t seed = (uint64_t)strtoull(argv[3], NULL, 10);
    const char *outpath = argv[4];
    double sprob = argc > 5 ? atof(argv[5]) : 0.15;
    /* trajectory mode: 0 = temperature-1 policy sampling (diverse),
     * 1 = policy argmax (closer to deployment's strong trajectories);
     * collect both and let the analysis check the fits agree */
    int adv_argmax = argc > 6 ? atoi(argv[6]) : 0;

    char spec96[256], spec1k[256];
    /* wide, floorless-ish, gate-free candidate sets; no sel/override logic so
     * the stats are the raw per-candidate means the selection layer consumes */
    snprintf(spec96, sizeof spec96,
             "rollout:%s:96:8:0.005:0:8:0:0:0:0:1:0:0:0:0:0:0", netpath);
    snprintf(spec1k, sizeof spec1k,
             "rollout:%s:1024:8:0.005:0:8:0:0:0:0:1:0:0:1:0:0:0", netpath);
    Agent cheap, oracle;
    spec_parse(spec96, &cheap);
    spec_parse(spec1k, &oracle);
    if (!cheap.net || !oracle.net) { fprintf(stderr, "bad net\n"); return 1; }
    const Net *net = cheap.net;

    FILE *out = fopen(outpath, "w");
    if (!out) { perror(outpath); return 1; }
    fprintf(out, "game\tply\tround\tdeck\tcand\tprio_top\tprio_c\t"
                 "q96_top\tq96_c\tq1k_top\tq1k_c\n");

    Rng rng; rng_seed(&rng, seed);
    Move mv[MAX_MOVES];
    float pr[MAX_MOVES];
    long rows = 0, states = 0;

    for (int g = 0; g < games; g++) {
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < MATCH_ROUNDS; rd++) {
            State st;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)(cum[0] > 320 ? 320 : (cum[0] < -320 ? -320 : cum[0]));
            st.cum[1] = (int16_t)(cum[1] > 320 ? 320 : (cum[1] < -320 ? -320 : cum[1]));
            st.turn = (uint8_t)(rd & 1);
            while (!st.over) {
                int n = policy_probs(net, &st, mv, pr, NULL);
                if (n <= 0) break;
                int nf = n > 1 ? lc_dedup_wagers(&st, mv, pr, n, 1) : n;
                if (nf >= 2 && (double)(rng_next(&rng) >> 11) * (1.0 / 9007199254740992.0) < sprob) {
                    SearchStats s96, s1k;
                    Rng r1, r2;
                    uint64_t ss = rng_next(&rng);
                    rng_seed(&r1, ss);
                    rng_seed(&r2, ss ^ 0xD1B54A32D192ED03ULL);
                    rollout_move(&cheap, &st, &r1, NULL, &s96);
                    rollout_move(&oracle, &st, &r2, NULL, &s1k);
                    /* candidate 0 is the top-prior move in both agents; join
                     * the rest by move identity in case pruning diverged */
                    if (s96.n >= 2 && s1k.n >= 2 &&
                        s96.mv[0].card == s1k.mv[0].card &&
                        s96.mv[0].discard == s1k.mv[0].discard &&
                        s96.mv[0].draw == s1k.mv[0].draw) {
                        states++;
                        for (int c = 1; c < s96.n; c++) {
                            int j = -1;
                            for (int k = 1; k < s1k.n; k++) {
                                if (s1k.mv[k].card == s96.mv[c].card &&
                                    s1k.mv[k].discard == s96.mv[c].discard &&
                                    s1k.mv[k].draw == s96.mv[c].draw) { j = k; break; }
                            }
                            if (j < 0) continue;
                            fprintf(out, "%d\t%d\t%d\t%d\t%d\t%.5f\t%.5f\t"
                                         "%.3f\t%.3f\t%.3f\t%.3f\n",
                                    g, st.nply, st.round, st.deck_left, c,
                                    s96.prio[0], s96.prio[c],
                                    s96.q[0], s96.q[c], s1k.q[0], s1k.q[j]);
                            rows++;
                        }
                        fflush(out);
                    }
                }
                /* advance: sampled (diverse) or argmax (deployment-like) */
                int pick;
                if (adv_argmax) {
                    pick = 0;
                    for (int i = 1; i < nf; i++) if (pr[i] > pr[pick]) pick = i;
                } else {
                    pick = sample_index(pr, nf, &rng);
                }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
        }
        fprintf(stderr, "game %d done, %ld states %ld rows\n", g, states, rows);
    }
    fclose(out);
    return 0;
}
