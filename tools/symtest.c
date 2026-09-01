/* symtest -- residual color/wager asymmetry of a policy, and what
 * test-time symmetrization would change.
 *
 * Nothing in the rules distinguishes the five suits or a suit's three
 * wager copies, so the policy's move distribution should be invariant to
 * relabeling them.  Training augmentation (--aug 1) makes it approximately
 * so; this measures the residue on real mid-game states: how much the
 * top move's probability moves across K random relabelings, how often the
 * argmax flips, and how often the K-averaged (symmetrized) policy picks a
 * different move than the raw one -- the rate at which a symmetrized
 * candidate stage would change what the search treats as the policy's
 * choice.
 *
 *   symtest [-n NET] [-g GAMES] [-k RELABELS] [-s SEED] [-p MINPLY]
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

static int key_of(Move m)
{
    int c = m.card;
    if (CARD_IS_WAGER(c)) c = CARD_SUIT(c) * NRANK;   /* fold copies */
    return c + 60 * m.discard + 120 * m.draw;
}

static void random_perm(Rng *r, int *p, int n)
{
    for (int i = 0; i < n; i++) p[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)rng_below(r, (uint32_t)i + 1);
        int t = p[i]; p[i] = p[j]; p[j] = t;
    }
}

int main(int argc, char **argv)
{
    const char *netpath = "data/best.bin";
    int games = 30, K = 8, minply = 14, argmax = 0;
    uint64_t seed = 777;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) minply = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-a")) argmax = 1;
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (net_load(net, netpath)) { fprintf(stderr, "cannot load %s\n", netpath); return 1; }
    Rng rng; rng_seed(&rng, seed);

    long nstates = 0, flips = 0, symdiff = 0, conf95 = 0, conf95_lost = 0;
    double sd_sum = 0.0, mean_top = 0.0, mean_symtop = 0.0;
    long hist_flip_by_conf[4] = {0}, hist_n_by_conf[4] = {0};

    for (int g = 0; g < games; g++) {
        int cum[2] = {0, 0};
        for (int rd = 0; rd < MATCH_ROUNDS; rd++) {
            State st;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)cum[0]; st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(rd & 1);
            while (!st.over) {
                Move mv[MAX_MOVES]; float pr[MAX_MOVES];
                int n = policy_probs(net, &st, mv, pr, NULL);
                if (n <= 0) break;
                int nf = n > 1 ? lc_dedup_wagers(&st, mv, pr, n, 1) : n;
                if (st.nply >= minply) {
                    /* base */
                    int top = 0;
                    for (int i = 1; i < nf; i++) if (pr[i] > pr[top]) top = i;
                    int topkey = key_of(mv[top]);
                    float ptop = pr[top];
                    /* accumulate symmetrized probs by key */
                    static float acc[720]; static float relp[64];
                    memset(acc, 0, sizeof acc);
                    for (int i = 0; i < nf; i++) acc[key_of(mv[i])] += pr[i];
                    double s1 = ptop, s2 = (double)ptop * ptop;
                    int nrel = 0;
                    for (int k = 0; k < K; k++) {
                        int sp[NSUIT], wp[NSUIT][WAGERS_PER_SUIT];
                        random_perm(&rng, sp, NSUIT);
                        for (int s = 0; s < NSUIT; s++) random_perm(&rng, wp[s], WAGERS_PER_SUIT);
                        uint8_t map[NCARD], inv[NCARD];
                        lc_perm_map(sp, wp, map);
                        for (int c = 0; c < NCARD; c++) inv[map[c]] = (uint8_t)c;
                        int invsuit[NSUIT];
                        for (int s = 0; s < NSUIT; s++) invsuit[sp[s]] = s;
                        State ps = st;
                        lc_permute(&ps, map);
                        Move pm[MAX_MOVES]; float pp[MAX_MOVES];
                        int pn = policy_probs(net, &ps, pm, pp, NULL);
                        if (pn <= 0) continue;
                        int pnf = pn > 1 ? lc_dedup_wagers(&ps, pm, pp, pn, 1) : pn;
                        float p_of_top = 0.0f;
                        for (int i = 0; i < pnf; i++) {
                            Move bm;
                            bm.card = inv[pm[i].card];
                            bm.discard = pm[i].discard;
                            bm.draw = pm[i].draw == 0 ? 0 : (uint8_t)(invsuit[pm[i].draw - 1] + 1);
                            int kk = key_of(bm);
                            acc[kk] += pp[i];
                            if (kk == topkey) p_of_top += pp[i];
                        }
                        relp[nrel++] = p_of_top;
                        s1 += p_of_top; s2 += (double)p_of_top * p_of_top;
                        /* argmax under this relabeling, mapped back */
                        int pt = 0;
                        for (int i = 1; i < pnf; i++) if (pp[i] > pp[pt]) pt = i;
                        Move bt; bt.card = inv[pm[pt].card]; bt.discard = pm[pt].discard;
                        bt.draw = pm[pt].draw == 0 ? 0 : (uint8_t)(invsuit[pm[pt].draw - 1] + 1);
                        if (key_of(bt) != topkey) flips++;
                    }
                    int m = nrel + 1;
                    double mean = s1 / m, var = s2 / m - mean * mean;
                    sd_sum += sqrt(var > 0 ? var : 0);
                    mean_top += ptop;
                    /* symmetrized argmax */
                    int bestk = -1; float bestv = -1.0f;
                    for (int kk = 0; kk < 720; kk++) if (acc[kk] > bestv) { bestv = acc[kk]; bestk = kk; }
                    mean_symtop += bestv / m;
                    if (bestk != topkey) symdiff++;
                    int cb = ptop >= 0.95f ? 3 : (ptop >= 0.7f ? 2 : (ptop >= 0.4f ? 1 : 0));
                    hist_n_by_conf[cb]++;
                    if (bestk != topkey) hist_flip_by_conf[cb]++;
                    if (ptop >= 0.95f) { conf95++; if (bestv / m < 0.95f) conf95_lost++; }
                    nstates++;
                    (void)relp;
                }
                /* advance by sampling the policy (diverse states) */
                float u = rng_float(&rng), c = 0.0f; int pick = n - 1;
                if (argmax) { pick = 0; for (int i = 1; i < n; i++) if (pr[i] > pr[pick]) pick = i; }
                else for (int i = 0; i < n; i++) { c += pr[i]; if (u < c) { pick = i; break; } }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0); cum[1] += lc_score(&st, 1);
        }
    }
    printf("states %ld (ply>=%d), relabelings %d\n", nstates, minply, K);
    printf("mean top prob %.3f; mean SD of top-move prob across relabelings %.3f\n",
           mean_top / nstates, sd_sum / nstates);
    printf("per-relabeling argmax flip rate %.1f%%\n", 100.0 * flips / ((double)nstates * K));
    printf("symmetrized argmax != raw argmax: %.1f%% of states\n", 100.0 * symdiff / nstates);
    const char *lab[4] = {"<0.40", "0.40-0.70", "0.70-0.95", ">=0.95"};
    for (int b = 0; b < 4; b++)
        printf("  raw-top-prob %-10s: %5ld states, symmetrized argmax differs %.1f%%\n",
               lab[b], hist_n_by_conf[b], hist_n_by_conf[b] ? 100.0 * hist_flip_by_conf[b] / hist_n_by_conf[b] : 0.0);
    printf("raw >=0.95 states: %ld; of those symmetrized prob <0.95: %ld (%.1f%%)\n",
           conf95, conf95_lost, conf95 ? 100.0 * conf95_lost / conf95 : 0.0);
    return 0;
}
