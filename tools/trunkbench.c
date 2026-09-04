/* trunkbench -- bit-identity and timing of the register-tiled net_trunk
 * against the reference (pre-tiling) loop.
 *
 *   trunkbench [-n NET] [-N NSTATES] [-r REPS] [-s SEED] [-p MINPLY]
 *
 * States come from self-play with the raw policy (moves sampled from it,
 * the same loop tools/symtest.c uses); the feature vector of every
 * position at ply >= MINPLY is kept until NSTATES are collected.  For each
 * of them net_trunk_ref (a verbatim copy of the loop the tiled kernel
 * replaced) and net_trunk must agree byte for byte on a1 and a2 -- that is
 * the acceptance criterion; a "close" result is a failure.  The same check
 * then runs on random nets whose widths are not tile multiples, covering
 * the remainder path.  Finally each kernel is timed over REPS forwards
 * cycling through the states, in alternating rounds so machine noise hits
 * both alike; the minimum round is the number to compare.
 *
 * Exit status is non-zero on any mismatch, so the tool doubles as a test.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

/* verbatim copy of net_trunk as it stood before the register-tiled kernel */
static void net_trunk_ref(const Net *n, const Features *f, NetAct *act)
{
    const int H1 = n->h1, H2 = n->h2;
    float h1[NET_H1_MAX];
    for (int h = 0; h < H1; h++) h1[h] = n->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = n->w1 + (size_t)f->idx[k] * H1;
        for (int h = 0; h < H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        const float *w = n->w1 + (size_t)(FEAT_BIN + j) * H1;
        for (int h = 0; h < H1; h++) h1[h] += x * w[h];
    }
    for (int h = 0; h < H1; h++) act->a1[h] = h1[h] > 0.0f ? h1[h] : 0.0f;

    float h2[NET_H2_MAX];
    for (int h = 0; h < H2; h++) h2[h] = n->b2[h];
    for (int i = 0; i < H1; i++) {
        float a = act->a1[i];
        if (a == 0.0f) continue;
        const float *w = n->w2 + (size_t)i * H2;
        for (int h = 0; h < H2; h++) h2[h] += a * w[h];
    }
    for (int h = 0; h < H2; h++) act->a2[h] = h2[h] > 0.0f ? h2[h] : 0.0f;
}

typedef void (*trunk_fn)(const Net *, const Features *, NetAct *);

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

/* both kernels on every state; the whole a1/a2 arrays are compared (they
 * are pre-filled with a sentinel, so a stray write past the width shows
 * up too).  Returns the number of states that differ. */
static int check_identity(const Net *net, const Features *F, int ns, const char *label)
{
    int bad = 0;
    for (int i = 0; i < ns; i++) {
        NetAct a, b;
        memset(&a, 0xA5, sizeof a);
        memset(&b, 0xA5, sizeof b);
        net_trunk_ref(net, &F[i], &a);
        net_trunk(net, &F[i], &b);
        int d1 = memcmp(a.a1, b.a1, sizeof a.a1) != 0;
        int d2 = memcmp(a.a2, b.a2, sizeof a.a2) != 0;
        if (d1 || d2) {
            if (bad < 3) {
                printf("  state %d: %s%s differ", i, d1 ? "a1 " : "", d2 ? "a2" : "");
                const float *ra = d1 ? a.a1 : a.a2, *rb = d1 ? b.a1 : b.a2;
                int lim = d1 ? net->h1 : net->h2;
                for (int h = 0; h < lim; h++)
                    if (ra[h] != rb[h] || memcmp(&ra[h], &rb[h], sizeof(float))) {
                        printf(" (first at %d: ref %.9g new %.9g)", h, ra[h], rb[h]);
                        break;
                    }
                printf("\n");
            }
            bad++;
        }
    }
    printf("%-28s %d/%d states bit-identical (a1 and a2)%s\n",
           label, ns - bad, ns, bad ? "   ** MISMATCH **" : "");
    return bad;
}

static double time_kernel(trunk_fn fn, const Net *net, const Features *F, int ns, long reps, float *sink)
{
    NetAct act;
    float s = 0.0f;
    double t0 = now_us();
    for (long r = 0; r < reps; r++) {
        fn(net, &F[r % ns], &act);
        s += act.a2[0];
    }
    double t1 = now_us();
    *sink += s;
    return (t1 - t0) / (double)reps;
}

static void print_loadavg(void)
{
    FILE *fp = fopen("/proc/loadavg", "r");
    char buf[128];
    if (fp && fgets(buf, sizeof buf, fp)) printf("loadavg %s", buf);
    if (fp) fclose(fp);
}

int main(int argc, char **argv)
{
    const char *netpath = "data/best.bin";
    int nstates = 1024, minply = 14;
    long reps = 20000;
    uint64_t seed = 777;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "-N") && i + 1 < argc) nstates = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atol(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) minply = atoi(argv[++i]);
        else { fprintf(stderr, "usage: trunkbench [-n NET] [-N NSTATES] [-r REPS] [-s SEED] [-p MINPLY]\n"); return 2; }
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (net_load(net, netpath)) { fprintf(stderr, "cannot load %s\n", netpath); return 1; }
    printf("net %s: h1 %d h2 %d\n", netpath, net->h1, net->h2);

    /* ---- states: raw-policy self-play, features of every position at
     * ply >= minply from the mover's point of view ---- */
    Features *F = (Features *)malloc((size_t)nstates * sizeof(Features));
    if (!F) return 1;
    Rng rng; rng_seed(&rng, seed);
    int ns = 0, games = 0;
    long nidx_sum = 0, dense_nz = 0;
    while (ns < nstates) {
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < MATCH_ROUNDS && ns < nstates; rd++) {
            State st;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)cum[0]; st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(rd & 1);
            while (!st.over && ns < nstates) {
                Move mv[MAX_MOVES]; float pr[MAX_MOVES];
                int n = policy_probs(net, &st, mv, pr, NULL);
                if (n <= 0) break;
                if (st.nply >= minply) {
                    feat_extract(&st, st.turn, &F[ns]);
                    nidx_sum += F[ns].nidx;
                    for (int j = 0; j < FEAT_DENSE; j++) dense_nz += F[ns].dense[j] != 0.0f;
                    ns++;
                }
                float u = rng_float(&rng), c = 0.0f; int pick = n - 1;
                for (int i = 0; i < n; i++) { c += pr[i]; if (u < c) { pick = i; break; } }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0); cum[1] += lc_score(&st, 1);
        }
        games++;
    }
    printf("states %d from %d self-play games (ply >= %d): mean %.1f sparse idx, %.1f non-zero dense\n",
           ns, games, minply, (double)nidx_sum / ns, (double)dense_nz / ns);

    /* ---- identity ---- */
    int bad = check_identity(net, F, ns, "loaded net");
    /* widths that are not tile multiples exercise the remainder kernels;
     * one full-tile and one tiny-width net too */
    static const int dims[][2] = { { 200, 100 }, { 72, 40 }, { 576, 320 }, { 8, 8 }, { 1024, 512 } };
    for (size_t d = 0; d < sizeof dims / sizeof dims[0]; d++) {
        Net r;
        if (net_alloc(&r, dims[d][0], dims[d][1]) != 0) { fprintf(stderr, "alloc %dx%d failed\n", dims[d][0], dims[d][1]); return 1; }
        net_init(&r, 0x7A11ULL + d);
        char label[64];
        snprintf(label, sizeof label, "random net %dx%d", dims[d][0], dims[d][1]);
        bad += check_identity(&r, F, ns, label);
        net_free(&r);
    }
    printf("identity: %s\n", bad ? "FAILED" : "OK -- new kernel is bit-identical to the reference");

    /* ---- timing ---- */
    float sink = 0.0f;
    time_kernel(net_trunk_ref, net, F, ns, 2000, &sink);   /* warm the caches */
    time_kernel(net_trunk, net, F, ns, 2000, &sink);
    const int rounds = 3;
    double best_old = 1e30, best_new = 1e30, sum_old = 0.0, sum_new = 0.0;
    for (int r = 0; r < rounds; r++) {
        double to = time_kernel(net_trunk_ref, net, F, ns, reps, &sink);
        double tn = time_kernel(net_trunk, net, F, ns, reps, &sink);
        printf("round %d: old %.2f us/forward, new %.2f us/forward\n", r + 1, to, tn);
        if (to < best_old) best_old = to;
        if (tn < best_new) best_new = tn;
        sum_old += to; sum_new += tn;
    }
    printf("%ld forwards x %d rounds: old %.2f us (best) %.2f (mean); new %.2f us (best) %.2f (mean); speedup %.2fx (best)\n",
           reps, rounds, best_old, sum_old / rounds, best_new, sum_new / rounds, best_old / best_new);
    print_loadavg();
    if (sink == 12345.678f) printf("(sink)\n");
    free(F);
    net_free(net);
    free(net);
    return bad ? 1 : 0;
}
