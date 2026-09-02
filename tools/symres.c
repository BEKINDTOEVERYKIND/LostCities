/* symres -- residual relabeling asymmetry of a deployed agent's decision
 * function: the symmetrized prior, value and belief an Agent actually
 * consumes under a spec, compared with themselves on a randomly relabeled
 * copy of the same state, mapped back.  symtest measures the raw net;
 * this measures what the search is handed.  Three relabeling classes are
 * reported separately: suits only, wager copies only, both.
 *
 *   symres SPEC [-g GAMES] [-p MINPLY] [-s SEED]
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

static int key_of(Move m)
{
    int c = m.card;
    if (CARD_IS_WAGER(c)) c = CARD_SUIT(c) * NRANK;
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

typedef struct {
    long n, top_agree;
    double sum_maxd, sum_tv, max_maxd, sum_dval, max_dval, sum_dv2, max_dv2;
    double sum_bl_mean, sum_bl_max, max_bl_max, sum_bp_tv, max_bp_tv;
} Stat;

static void softmax(const float *l, int n, double *p)
{
    double mx = -1e30, z = 0.0;
    for (int i = 0; i < n; i++) if (l[i] > mx) mx = l[i];
    for (int i = 0; i < n; i++) { p[i] = exp(l[i] - mx); z += p[i]; }
    for (int i = 0; i < n; i++) p[i] /= z;
}

/* compare the agent's decision numbers on st and on its relabeling */
static void measure(const Agent *ag, const State *st, Rng *rng, int suits, int copies, Stat *S)
{
    int p = st->turn;
    int sp[NSUIT], wp[NSUIT][WAGERS_PER_SUIT];
    if (suits) random_perm(rng, sp, NSUIT); else for (int i = 0; i < NSUIT; i++) sp[i] = i;
    for (int s = 0; s < NSUIT; s++) {
        if (copies) random_perm(rng, wp[s], WAGERS_PER_SUIT);
        else for (int i = 0; i < WAGERS_PER_SUIT; i++) wp[s][i] = i;
    }
    uint8_t map[NCARD], inv[NCARD];
    lc_perm_map(sp, wp, map);
    for (int c = 0; c < NCARD; c++) inv[map[c]] = (uint8_t)c;
    int invsuit[NSUIT];
    for (int s = 0; s < NSUIT; s++) invsuit[sp[s]] = s;
    State ps = *st;
    lc_permute(&ps, map);

    /* prior + value (the candidate stage's numbers) */
    Move mv[MAX_MOVES], pm[MAX_MOVES];
    float pr[MAX_MOVES], pp[MAX_MOVES], v1 = 0.0f, v2 = 0.0f;
    Rng dummy; rng_seed(&dummy, 1);
    int n1 = agent_policy_probs(ag, st, &dummy, mv, pr, &v1);
    int n2 = agent_policy_probs(ag, &ps, &dummy, pm, pp, &v2);
    if (n1 <= 1 || n2 <= 1) return;
    static double acc1[720], acc2[720];
    memset(acc1, 0, sizeof acc1); memset(acc2, 0, sizeof acc2);
    for (int i = 0; i < n1; i++) acc1[key_of(mv[i])] += pr[i];
    for (int i = 0; i < n2; i++) {
        Move b;
        b.card = inv[pm[i].card];
        b.discard = pm[i].discard;
        b.draw = pm[i].draw == 0 ? 0 : (uint8_t)(invsuit[pm[i].draw - 1] + 1);
        acc2[key_of(b)] += pp[i];
    }
    double maxd = 0.0, tv = 0.0; int t1 = 0, t2 = 0;
    for (int k = 0; k < 720; k++) {
        double d = fabs(acc1[k] - acc2[k]);
        if (d > maxd) maxd = d;
        tv += d;
        if (acc1[k] > acc1[t1]) t1 = k;
        if (acc2[k] > acc2[t2]) t2 = k;
    }
    tv *= 0.5;
    S->n++;
    S->sum_maxd += maxd; if (maxd > S->max_maxd) S->max_maxd = maxd;
    S->sum_tv += tv;
    if (t1 == t2) S->top_agree++;
    double dv = fabs(v1 - v2);
    S->sum_dval += dv; if (dv > S->max_dval) S->max_dval = dv;
    /* the display value head, mover's perspective */
    double dv2 = fabs(agent_value(ag, st, p) - agent_value(ag, &ps, p));
    S->sum_dv2 += dv2; if (dv2 > S->max_dv2) S->max_dv2 = dv2;

    /* belief logits about the opponent's hand, matched per original card */
    uint8_t u1[NCARD], u2[NCARD];
    float l1[NCARD], l2[NCARD];
    int nb1 = agent_belief_logits(ag, st, p, &dummy, u1, l1);
    int nb2 = agent_belief_logits(ag, &ps, p, &dummy, u2, l2);
    if (nb1 > 0 && nb1 == nb2) {
        float byc1[NCARD], byc2[NCARD];
        for (int c = 0; c < NCARD; c++) byc1[c] = byc2[c] = -1e9f;
        for (int i = 0; i < nb1; i++) byc1[u1[i]] = l1[i];
        for (int i = 0; i < nb2; i++) byc2[inv[u2[i]]] = l2[i];
        float a1[NCARD], a2[NCARD]; int m = 0;
        double bl_sum = 0.0, bl_max = 0.0;
        for (int c = 0; c < NCARD; c++) {
            if (byc1[c] < -1e8f || byc2[c] < -1e8f) continue;
            a1[m] = byc1[c]; a2[m] = byc2[c]; m++;
            double d = fabs(byc1[c] - byc2[c]);
            bl_sum += d; if (d > bl_max) bl_max = d;
        }
        if (m > 0) {
            double p1[NCARD], p2[NCARD], btv = 0.0;
            softmax(a1, m, p1); softmax(a2, m, p2);
            for (int i = 0; i < m; i++) btv += fabs(p1[i] - p2[i]);
            btv *= 0.5;
            S->sum_bl_mean += bl_sum / m;
            S->sum_bl_max += bl_max; if (bl_max > S->max_bl_max) S->max_bl_max = bl_max;
            S->sum_bp_tv += btv; if (btv > S->max_bp_tv) S->max_bp_tv = btv;
        }
    }
}

static void report(const char *lab, const Stat *S)
{
    if (!S->n) { printf("%-12s no states\n", lab); return; }
    double n = (double)S->n;
    printf("%-12s prior: mean max|dp| %.4f (max %.4f)  mean TV %.4f  top agree %.2f%%\n",
           lab, S->sum_maxd / n, S->max_maxd, S->sum_tv / n, 100.0 * S->top_agree / n);
    printf("%-12s value: mean |dv| %.3f (max %.3f) pts;  display value mean |dv| %.3f (max %.3f)\n",
           "", S->sum_dval / n, S->max_dval, S->sum_dv2 / n, S->max_dv2);
    printf("%-12s belief: mean |dlogit| %.4f, mean max|dlogit| %.4f (max %.4f), softmax TV mean %.4f (max %.4f)\n",
           "", S->sum_bl_mean / n, S->sum_bl_max / n, S->max_bl_max, S->sum_bp_tv / n, S->max_bp_tv);
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s SPEC [-g GAMES] [-p MINPLY] [-s SEED]\n", argv[0]); return 2; }
    const char *spec = argv[1];
    int games = 20, minply = 14; uint64_t seed = 7;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-g") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) minply = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
    }
    /* the exact enumeration must produce 120 distinct suit permutations */
    {
        Rng r; rng_seed(&r, 1);
        static uint8_t seen[3125];
        int distinct = 0;
        for (int k = 0; k < LC_SYM_EXACT; k++) {
            int sp[NSUIT], wp[NSUIT][WAGERS_PER_SUIT];
            lc_sym_relabel(&r, LC_SYM_EXACT, k, sp, wp);
            int code = 0, ok = 1, used = 0;
            for (int i = 0; i < NSUIT; i++) { code = code * 5 + sp[i]; if (sp[i] < 0 || sp[i] >= NSUIT || (used >> sp[i]) & 1) ok = 0; used |= 1 << sp[i]; }
            if (ok && !seen[code]) { seen[code] = 1; distinct++; }
        }
        printf("exact suit enumeration: %d distinct permutations of %d\n", distinct, LC_SYM_EXACT);
    }
    Agent ag;
    memset(&ag, 0, sizeof ag);
    spec_parse(spec, &ag);
    if (!ag.net) { fprintf(stderr, "spec has no net\n"); return 1; }
    printf("spec %s\n", spec);
    Rng rng; rng_seed(&rng, seed);
    Stat Ss, Sc, Sb;
    memset(&Ss, 0, sizeof Ss); memset(&Sc, 0, sizeof Sc); memset(&Sb, 0, sizeof Sb);
    long nst = 0;
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
                int n = policy_probs(ag.net, &st, mv, pr, NULL);
                if (n <= 0) break;
                if (st.nply >= minply) {
                    measure(&ag, &st, &rng, 1, 0, &Ss);
                    measure(&ag, &st, &rng, 0, 1, &Sc);
                    measure(&ag, &st, &rng, 1, 1, &Sb);
                    nst++;
                }
                float u = rng_float(&rng), c = 0.0f; int pick = n - 1;
                for (int i = 0; i < n; i++) { c += pr[i]; if (u < c) { pick = i; break; } }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0); cum[1] += lc_score(&st, 1);
        }
    }
    printf("states %ld (ply >= %d)\n", nst, minply);
    report("suits", &Ss);
    report("copies", &Sc);
    report("both", &Sb);
    return 0;
}
