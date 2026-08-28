/* belief -- dedicated measurement and head-only training for the opponent-
 * hand inference head.
 *
 * The belief head has only ever been trained as a side objective of the
 * policy trainers.  This tool isolates it: generate a large state corpus
 * from champion self-play, measure prediction quality properly, and train
 * ONLY the head weights (wbel/bbel) on the frozen trunk -- the policy and
 * value heads stay bit-identical, so play decisions cannot change; better
 * beliefs reach play solely through world sampling.
 *
 *   belief gen   NET GAMES SEED OUT.bst
 *   belief eval  NET STATES.bst FROMGAME        (metrics on games >= FROMGAME)
 *   belief train NET STATES.bst OUT.bin EPOCHS LR HOLDGAMES
 *
 * Metrics are reported for two candidate sets:
 *   trainset  -- every card not visible to the mover (the trainers' set;
 *                includes opponent cards known from face-up pile draws,
 *                which the features encode outright)
 *   unknown   -- the strictly-unknown subset (known[opp] excluded): the
 *                honest measure of INFERENCE rather than bookkeeping
 * against the counting prior p = hidden_opp_cards / n_candidates.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/features.h"
#include "../src/agent.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BST_MAGIC 0x4C424554u

typedef struct { State st; uint16_t game; } Rec;

static long load_bst(const char *path, Rec **out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    uint32_t h[2]; uint64_t count;
    if (fread(h, sizeof h, 1, f) != 1 || fread(&count, sizeof count, 1, f) != 1 ||
        h[0] != BST_MAGIC || h[1] != sizeof(Rec)) {
        fprintf(stderr, "%s: not a compatible state corpus\n", path); exit(1);
    }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, sizeof h + sizeof count, SEEK_SET);
    if (count > (uint64_t)(fsz - (long)(sizeof h + sizeof count)) / sizeof(Rec)) {
        fprintf(stderr, "%s: count exceeds file size\n", path); exit(1);
    }
    Rec *r = (Rec *)malloc(sizeof(Rec) * count);
    if (fread(r, sizeof(Rec), count, f) != count) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    *out = r;
    return (long)count;
}

static void gen(const Net *net, int games, uint64_t seed, const char *outp)
{
    if (games < 1 || games > 65535) { fprintf(stderr, "gen: GAMES must be 1..65535 (uint16 game ids)\n"); exit(1); }
    FILE *out = fopen(outp, "wb");
    if (!out) { perror(outp); exit(1); }
    uint32_t h[2] = { BST_MAGIC, sizeof(Rec) };
    uint64_t count = 0;
    fwrite(h, sizeof h, 1, out); fwrite(&count, sizeof count, 1, out);
    Rng rng; rng_seed(&rng, seed);
    Move mv[MAX_MOVES];
    float pr[MAX_MOVES];
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
                Rec r; r.st = st; r.game = (uint16_t)g;
                fwrite(&r, sizeof r, 1, out);
                count++;
                int n = policy_probs(net, &st, mv, pr, NULL);
                if (n <= 0) break;
                lc_apply(&st, mv[sample_index(pr, n, &rng)]);
            }
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
        }
        if ((g + 1) % 200 == 0) fprintf(stderr, "gen %d/%d games, %llu states\n",
                                        g + 1, games, (unsigned long long)count);
    }
    fseek(out, sizeof h, SEEK_SET);
    fwrite(&count, sizeof count, 1, out);
    fclose(out);
    printf("wrote %llu states from %d games to %s\n",
           (unsigned long long)count, games, outp);
}

/* candidate cards for perspective p; returns n, fills cards/labels/known-flag */
static int cand_set(const State *st, int p, uint8_t *cards, uint8_t *lab, uint8_t *isknown)
{
    int o = p ^ 1, n = 0;
    uint64_t vis = st->hand[p] | st->played[0] | st->played[1] | st->discarded;
    uint64_t cands = ~vis & ((1ULL << NCARD) - 1);
    while (cands) {
        int c = __builtin_ctzll(cands); cands &= cands - 1;
        cards[n] = (uint8_t)c;
        lab[n] = (uint8_t)((st->hand[o] >> c) & 1ULL);
        isknown[n] = (uint8_t)((st->known[o] >> c) & 1ULL);
        n++;
    }
    return n;
}

typedef struct {
    double bce, pbce;      /* model and prior BCE sums */
    long n, held;
    double auc_w; long auc_pairs;      /* pooled within-decision pairwise wins */
    long caln[10]; double calp[10], calh[10];
} Acc;

static void acc_add(Acc *a, const float *prob, const uint8_t *lab, int n, float prior)
{
    for (int i = 0; i < n; i++) {
        float p = prob[i];
        if (p < 1e-6f) p = 1e-6f;
        if (p > 1.0f - 1e-6f) p = 1.0f - 1e-6f;
        a->bce += lab[i] ? -log(p) : -log(1.0f - p);
        float q = prior < 1e-6f ? 1e-6f : (prior > 1.0f - 1e-6f ? 1.0f - 1e-6f : prior);
        a->pbce += lab[i] ? -log(q) : -log(1.0f - q);
        a->n++; a->held += lab[i];
        int b = (int)(p * 10.0f); if (b > 9) b = 9;
        a->caln[b]++; a->calp[b] += p; a->calh[b] += lab[i];
    }
    /* in-state AUC pairs: held vs not-held within the same decision */
    for (int i = 0; i < n; i++) {
        if (!lab[i]) continue;
        for (int j = 0; j < n; j++) {
            if (lab[j]) continue;
            a->auc_pairs++;
            if (prob[i] > prob[j]) a->auc_w += 1.0;
            else if (prob[i] == prob[j]) a->auc_w += 0.5;
        }
    }
}

static void acc_print(const char *name, const Acc *a)
{
    printf("  %-9s n=%-9ld held=%.3f  BCE %.4f  (prior %.4f, skill %.1f%%)  AUC %.4f\n",
           name, a->n, a->n ? (double)a->held / a->n : 0.0,
           a->n ? a->bce / a->n : 0.0, a->n ? a->pbce / a->n : 0.0,
           a->pbce > 0 ? 100.0 * (1.0 - a->bce / a->pbce) : 0.0,
           a->auc_pairs ? a->auc_w / a->auc_pairs : 0.0);
}

static void eval_net(const Net *net, const Rec *recs, long nrec, int fromgame, int verbose)
{
    Acc all = { 0 }, unk = { 0 }, phase[3] = { { 0 } };
    Features f;
    NetAct act;
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD];
    float logit[NCARD], prob[NCARD], up[NCARD];
    uint8_t ulab[NCARD];
    for (long i = 0; i < nrec; i++) {
        if (recs[i].game < fromgame) continue;
        const State *st = &recs[i].st;
        for (int p = 0; p < 2; p++) {
            int n = cand_set(st, p, cards, lab, isk);
            if (n < 2) continue;
            feat_extract(st, p, &f);
            net_trunk(net, &f, &act);
            net_belief_act(net, &act, cards, n, logit);
            int held = 0;
            for (int k = 0; k < n; k++) {
                float l = logit[k];
                if (l > 15.0f) l = 15.0f;
                if (l < -15.0f) l = -15.0f;
                prob[k] = 1.0f / (1.0f + expf(-l));
                held += lab[k];
            }
            float prior = (float)held / (float)n;
            acc_add(&all, prob, lab, n, prior);
            /* strictly-unknown subset: exclude cards known via face-up draws */
            int un = 0, uheld = 0;
            for (int k = 0; k < n; k++) {
                if (isk[k]) continue;
                up[un] = prob[k]; ulab[un] = lab[k]; uheld += lab[k]; un++;
            }
            if (un >= 2) {
                float uprior = (float)uheld / (float)un;
                acc_add(&unk, up, ulab, un, uprior);
                int ph = st->deck_left >= 30 ? 0 : (st->deck_left >= 15 ? 1 : 2);
                acc_add(&phase[ph], up, ulab, un, uprior);
            }
        }
    }
    printf("eval on games >= %d:\n", fromgame);
    acc_print("trainset", &all);
    acc_print("unknown", &unk);
    acc_print("  early", &phase[0]);
    acc_print("  mid", &phase[1]);
    acc_print("  late", &phase[2]);
    if (verbose) {
        printf("  calibration (unknown set, predicted -> observed):\n");
        for (int b = 0; b < 10; b++)
            if (unk.caln[b] >= 200)
                printf("    %.1f-%.1f: n=%-8ld mean %.3f  observed %.3f\n",
                       b / 10.0, (b + 1) / 10.0, unk.caln[b],
                       unk.calp[b] / unk.caln[b], unk.calh[b] / unk.caln[b]);
    }
}

static void train_head(Net *net, const Rec *recs, long nrec, const char *outp,
                       int epochs, float lr, int holdgames)
{
    int maxg = 0;
    for (long i = 0; i < nrec; i++) if (recs[i].game > maxg) maxg = recs[i].game;
    if (holdgames < 1 || holdgames > maxg) {
        fprintf(stderr, "train: HOLDGAMES must be 1..%d\n", maxg); exit(1);
    }
    int cut = maxg + 1 - holdgames;
    printf("train on games < %d, hold out %d..%d\n", cut, cut, maxg);

    const int H2 = net->h2;
    size_t nw = (size_t)NCARD * H2 + NCARD;
    float *m = (float *)calloc(nw, sizeof(float));
    float *v = (float *)calloc(nw, sizeof(float));
    Features f;
    NetAct act;
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD];
    float logit[NCARD];

    /* pre-index training records for cheap shuffling */
    long *idx = (long *)malloc(sizeof(long) * nrec);
    long ntr = 0;
    for (long i = 0; i < nrec; i++) if (recs[i].game < cut) idx[ntr++] = i;
    Rng rng; rng_seed(&rng, 0xBE11EF);

    for (int e = 1; e <= epochs; e++) {
        /* Fisher-Yates */
        for (long i = ntr - 1; i > 0; i--) {
            long j = (long)(rng_next(&rng) % (uint64_t)(i + 1));
            long tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        double bce = 0; long bn = 0;
        for (long ii = 0; ii < ntr; ii++) {
            const State *st = &recs[idx[ii]].st;
            for (int p = 0; p < 2; p++) {
                int n = cand_set(st, p, cards, lab, isk);
                if (n < 2) continue;
                feat_extract(st, p, &f);
                net_trunk(net, &f, &act);
                net_belief_act(net, &act, cards, n, logit);
                float scale = 1.0f / (float)n;
                for (int k = 0; k < n; k++) {
                    float l = logit[k];
                    if (l > 15.0f) l = 15.0f;
                    if (l < -15.0f) l = -15.0f;
                    float pr = 1.0f / (1.0f + expf(-l));
                    bce += lab[k] ? -log(pr + 1e-6f) : -log(1.0f - pr + 1e-6f);
                    bn++;
                    float g = scale * (pr - (float)lab[k]);
                    /* Adam, per-parameter, belief weights only */
                    int c = cards[k];
                    float *wrow = net->wbel + (size_t)c * H2;
                    float *mr = m + (size_t)c * H2, *vr = v + (size_t)c * H2;
                    /* uncorrected Adam by choice: the warm-start head plus a
                     * small lr makes the ~10-update overshoot transient
                     * negligible (reviewed and quantified: <=6.5x for the
                     * first ~0.1%% of epoch 1), and per-card correction would
                     * need per-card step counters for no measurable gain */
                    for (int hh = 0; hh < H2; hh++) {
                        float gr = g * act.a2[hh];
                        mr[hh] = 0.9f * mr[hh] + 0.1f * gr;
                        vr[hh] = 0.999f * vr[hh] + 0.001f * gr * gr;
                        wrow[hh] -= lr * mr[hh] / (sqrtf(vr[hh]) + 1e-8f);
                    }
                    float gb = g;
                    float *mb = m + (size_t)NCARD * H2 + c, *vb = v + (size_t)NCARD * H2 + c;
                    *mb = 0.9f * *mb + 0.1f * gb;
                    *vb = 0.999f * *vb + 0.001f * gb * gb;
                    net->bbel[c] -= lr * *mb / (sqrtf(*vb) + 1e-8f);
                }
            }
        }
        printf("epoch %d: train BCE %.4f over %ld card-preds\n", e, bn ? bce / bn : 0.0, bn);
        eval_net(net, recs, nrec, cut, e == epochs);
        fflush(stdout);
        net_save(net, outp);
    }
    free(m); free(v); free(idx);
    printf("saved head-tuned net to %s (trunk/policy/value untouched)\n", outp);
}


/* ---- belx: extended-input belief specialist --------------------------
 * A standalone MLP whose input extends the engine features with the two
 * planes the snapshot encoding erases: who discarded each card still in
 * a pile, and how long each player has passed over the current pile tops.
 * Warm-started from a standard-net specialist (new rows zero), so at
 * init it computes the identical function and training can only add the
 * new information. */

#define XBIN  (2 * NCARD)                       /* disc_by[o], disc_by[p] */
#define XDENSE 10                               /* passed[o][s], passed[p][s] */
#define XDIM  (FEAT_DIM + XBIN + XDENSE)
#define BLX_MAGIC 0x42454C58u

typedef struct {
    int h1, h2;
    float *blk;
    float *w1, *b1, *w2, *b2, *wb, *bb;
} BelX;

static size_t belx_nfloat(int h1, int h2)
{
    return (size_t)XDIM * h1 + h1 + (size_t)h1 * h2 + h2 + (size_t)NCARD * h2 + NCARD;
}

static void belx_wire(BelX *x)
{
    float *p = x->blk;
    x->w1 = p; p += (size_t)XDIM * x->h1;
    x->b1 = p; p += x->h1;
    x->w2 = p; p += (size_t)x->h1 * x->h2;
    x->b2 = p; p += x->h2;
    x->wb = p; p += (size_t)NCARD * x->h2;
    x->bb = p;
}

static void belx_alloc(BelX *x, int h1, int h2)
{
    x->h1 = h1; x->h2 = h2;
    x->blk = (float *)calloc(belx_nfloat(h1, h2), sizeof(float));
    if (!x->blk) { fprintf(stderr, "belx: out of memory\n"); exit(1); }
    belx_wire(x);
}

static void belx_save(const BelX *x, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    uint32_t h[4] = { BLX_MAGIC, (uint32_t)x->h1, (uint32_t)x->h2, XDIM };
    fwrite(h, sizeof h, 1, f);
    fwrite(x->blk, sizeof(float), belx_nfloat(x->h1, x->h2), f);
    fclose(f);
}

static int belx_load(BelX *x, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[4];
    if (fread(h, sizeof h, 1, f) != 1 || h[0] != BLX_MAGIC || h[3] != XDIM) { fclose(f); return -2; }
    belx_alloc(x, (int)h[1], (int)h[2]);
    if (fread(x->blk, sizeof(float), belx_nfloat(x->h1, x->h2), f) != belx_nfloat(x->h1, x->h2)) { fclose(f); return -3; }
    fclose(f);
    return 0;
}

/* warm start from a standard Net specialist: identical function at init */
static void belx_from_net(BelX *x, const Net *n)
{
    belx_alloc(x, n->h1, n->h2);
    for (int r = 0; r < FEAT_DIM; r++)
        memcpy(x->w1 + (size_t)r * x->h1, n->w1 + (size_t)r * n->h1, sizeof(float) * n->h1);
    memcpy(x->b1, n->b1, sizeof(float) * n->h1);
    memcpy(x->w2, n->w2, sizeof(float) * (size_t)n->h1 * n->h2);
    memcpy(x->b2, n->b2, sizeof(float) * n->h2);
    memcpy(x->wb, n->wbel, sizeof(float) * (size_t)NCARD * n->h2);
    memcpy(x->bb, n->bbel, sizeof(float) * NCARD);
}

/* extended feature build: engine features plus the history planes */
typedef struct {
    uint16_t idx[128];
    int nidx;
    float dense[FEAT_DENSE + XDENSE];   /* rows FEAT_BIN.. and XROW_DENSE.. */
} XFeat;

static void xfeat(const State *st, int p, Features *base, XFeat *xf)
{
    const int o = p ^ 1;
    feat_extract(st, p, base);
    int n = 0;
    uint64_t mask = st->disc_by[o];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; xf->idx[n++] = (uint16_t)(FEAT_DIM + c); }
    mask = st->disc_by[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; xf->idx[n++] = (uint16_t)(FEAT_DIM + NCARD + c); }
    xf->nidx = n;
    for (int s = 0; s < NSUIT; s++) {
        float po = st->passed[o][s] * (1.0f / 8.0f); if (po > 1.0f) po = 1.0f;
        float pp = st->passed[p][s] * (1.0f / 8.0f); if (pp > 1.0f) pp = 1.0f;
        xf->dense[s] = po;
        xf->dense[NSUIT + s] = pp;
    }
}

typedef struct { float a1[NET_H1_MAX], a2[NET_H2_MAX]; } XAct;

static void belx_trunk(const BelX *x, const Features *f, const XFeat *xf, XAct *act)
{
    const int H1 = x->h1, H2 = x->h2;
    float h1[NET_H1_MAX];
    for (int h = 0; h < H1; h++) h1[h] = x->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = x->w1 + (size_t)f->idx[k] * H1;
        for (int h = 0; h < H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float v = f->dense[j];
        if (v == 0.0f) continue;
        const float *w = x->w1 + (size_t)(FEAT_BIN + j) * H1;
        for (int h = 0; h < H1; h++) h1[h] += v * w[h];
    }
    for (int k = 0; k < xf->nidx; k++) {
        const float *w = x->w1 + (size_t)xf->idx[k] * H1;
        for (int h = 0; h < H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < XDENSE; j++) {
        float v = xf->dense[j];
        if (v == 0.0f) continue;
        const float *w = x->w1 + (size_t)(FEAT_DIM + XBIN + j) * H1;
        for (int h = 0; h < H1; h++) h1[h] += v * w[h];
    }
    for (int h = 0; h < H1; h++) act->a1[h] = h1[h] > 0.0f ? h1[h] : 0.0f;
    float h2[NET_H2_MAX];
    for (int h = 0; h < H2; h++) h2[h] = x->b2[h];
    for (int i = 0; i < H1; i++) {
        float a = act->a1[i];
        if (a == 0.0f) continue;
        const float *w = x->w2 + (size_t)i * H2;
        for (int h = 0; h < H2; h++) h2[h] += a * w[h];
    }
    for (int h = 0; h < H2; h++) act->a2[h] = h2[h] > 0.0f ? h2[h] : 0.0f;
}

static void belx_logits(const BelX *x, const XAct *act, const uint8_t *cards, int nc, float *lg)
{
    const int H2 = x->h2;
    for (int i = 0; i < nc; i++) {
        const float *w = x->wb + (size_t)cards[i] * H2;
        float v = x->bb[cards[i]];
        for (int h = 0; h < H2; h++) v += w[h] * act->a2[h];
        lg[i] = v;
    }
}

/* eval mirror of eval_net for BelX */
static void xeval_net(const BelX *x, const Rec *recs, long nrec, int fromgame, int verbose)
{
    Acc all = { 0 }, unk = { 0 }, phase[3] = { { 0 } };
    Features f; XFeat xf; XAct act;
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD];
    float logit[NCARD], prob[NCARD], up[NCARD];
    uint8_t ulab[NCARD];
    for (long i = 0; i < nrec; i++) {
        if (recs[i].game < fromgame) continue;
        const State *st = &recs[i].st;
        for (int p = 0; p < 2; p++) {
            int n = cand_set(st, p, cards, lab, isk);
            if (n < 2) continue;
            xfeat(st, p, &f, &xf);
            belx_trunk(x, &f, &xf, &act);
            belx_logits(x, &act, cards, n, logit);
            int held = 0;
            for (int k = 0; k < n; k++) {
                float l = logit[k];
                if (l > 15.0f) l = 15.0f;
                if (l < -15.0f) l = -15.0f;
                prob[k] = 1.0f / (1.0f + expf(-l));
                held += lab[k];
            }
            acc_add(&all, prob, lab, n, (float)held / (float)n);
            int un = 0, uheld = 0;
            for (int k = 0; k < n; k++) {
                if (isk[k]) continue;
                up[un] = prob[k]; ulab[un] = lab[k]; uheld += lab[k]; un++;
            }
            if (un >= 2) {
                float uprior = (float)uheld / (float)un;
                acc_add(&unk, up, ulab, un, uprior);
                int ph = st->deck_left >= 30 ? 0 : (st->deck_left >= 15 ? 1 : 2);
                acc_add(&phase[ph], up, ulab, un, uprior);
            }
        }
    }
    printf("xeval on games >= %d:\n", fromgame);
    acc_print("trainset", &all);
    acc_print("unknown", &unk);
    acc_print("  early", &phase[0]);
    acc_print("  mid", &phase[1]);
    acc_print("  late", &phase[2]);
    if (verbose) {
        printf("  calibration (unknown set):\n");
        for (int b = 0; b < 10; b++)
            if (unk.caln[b] >= 200)
                printf("    %.1f-%.1f: n=%-8ld mean %.3f  observed %.3f\n",
                       b / 10.0, (b + 1) / 10.0, unk.caln[b],
                       unk.calp[b] / unk.caln[b], unk.calh[b] / unk.caln[b]);
    }
}

/* full-backprop belief-only training of the BelX net, single-threaded
 * Adam, per-decision 1/n loss scale as everywhere else */
static void xtrain(BelX *x, const Rec *recs, long nrec, const char *outp,
                   int epochs, float lr, int holdgames)
{
    int maxg = 0;
    for (long i = 0; i < nrec; i++) if (recs[i].game > maxg) maxg = recs[i].game;
    if (holdgames < 1 || holdgames > maxg) { fprintf(stderr, "bad HOLDGAMES\n"); exit(1); }
    int cut = maxg + 1 - holdgames;
    printf("xtrain on games < %d, hold out %d..%d\n", cut, cut, maxg);

    size_t nw = belx_nfloat(x->h1, x->h2);
    float *m = (float *)calloc(nw, sizeof(float));
    float *v = (float *)calloc(nw, sizeof(float));
    long *idx = (long *)malloc(sizeof(long) * nrec);
    long ntr = 0;
    for (long i = 0; i < nrec; i++) if (recs[i].game < cut) idx[ntr++] = i;
    Rng rng; rng_seed(&rng, 0xB31EFF);
    const int H1 = x->h1, H2 = x->h2;

    Features f; XFeat xf; XAct act;
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD];
    float lg[NCARD];
    float d2[NET_H2_MAX], d1[NET_H1_MAX];

    #define ADAM(off, g) do { \
        size_t _o = (off); float _g = (g); \
        m[_o] = 0.9f * m[_o] + 0.1f * _g; \
        v[_o] = 0.999f * v[_o] + 0.001f * _g * _g; \
        x->blk[_o] -= lr * m[_o] / (sqrtf(v[_o]) + 1e-8f); \
    } while (0)

    for (int e = 1; e <= epochs; e++) {
        for (long i = ntr - 1; i > 0; i--) {
            long j = (long)(rng_next(&rng) % (uint64_t)(i + 1));
            long t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        double bce = 0; long bn = 0;
        for (long ii = 0; ii < ntr; ii++) {
            const State *st = &recs[idx[ii]].st;
            for (int p = 0; p < 2; p++) {
                int n = cand_set(st, p, cards, lab, isk);
                if (n < 2) continue;
                xfeat(st, p, &f, &xf);
                belx_trunk(x, &f, &xf, &act);
                belx_logits(x, &act, cards, n, lg);
                float scale = 1.0f / (float)n;
                for (int h = 0; h < H2; h++) d2[h] = 0.0f;
                size_t off_wb = (size_t)XDIM * H1 + H1 + (size_t)H1 * H2 + H2;
                size_t off_bb = off_wb + (size_t)NCARD * H2;
                for (int k = 0; k < n; k++) {
                    float l = lg[k];
                    if (l > 15.0f) l = 15.0f;
                    if (l < -15.0f) l = -15.0f;
                    float pr = 1.0f / (1.0f + expf(-l));
                    bce += lab[k] ? -log(pr + 1e-6f) : -log(1.0f - pr + 1e-6f);
                    bn++;
                    float g = scale * (pr - (float)lab[k]);
                    int c = cards[k];
                    const float *wrow = x->wb + (size_t)c * H2;
                    for (int h = 0; h < H2; h++) {
                        d2[h] += g * wrow[h];
                        ADAM(off_wb + (size_t)c * H2 + h, g * act.a2[h]);
                    }
                    ADAM(off_bb + c, g);
                }
                for (int h = 0; h < H2; h++) if (act.a2[h] == 0.0f) d2[h] = 0.0f;
                for (int h = 0; h < H1; h++) d1[h] = 0.0f;
                size_t off_w2 = (size_t)XDIM * H1 + H1;
                size_t off_b2 = off_w2 + (size_t)H1 * H2;
                for (int h = 0; h < H2; h++) if (d2[h] != 0.0f) ADAM(off_b2 + h, d2[h]);
                for (int i2 = 0; i2 < H1; i2++) {
                    float a = act.a1[i2];
                    const float *w2r = x->w2 + (size_t)i2 * H2;
                    float acc2 = 0.0f;
                    for (int h = 0; h < H2; h++) acc2 += d2[h] * w2r[h];
                    d1[i2] = a > 0.0f ? acc2 : 0.0f;
                    if (a != 0.0f)
                        for (int h = 0; h < H2; h++)
                            if (d2[h] != 0.0f) ADAM(off_w2 + (size_t)i2 * H2 + h, d2[h] * a);
                }
                size_t off_b1 = (size_t)XDIM * H1;
                for (int h = 0; h < H1; h++) if (d1[h] != 0.0f) ADAM(off_b1 + h, d1[h]);
                for (int k = 0; k < f.nidx; k++) {
                    size_t r = (size_t)f.idx[k] * H1;
                    for (int h = 0; h < H1; h++) if (d1[h] != 0.0f) ADAM(r + h, d1[h]);
                }
                for (int j = 0; j < FEAT_DENSE; j++) {
                    float vv = f.dense[j];
                    if (vv == 0.0f) continue;
                    size_t r = (size_t)(FEAT_BIN + j) * H1;
                    for (int h = 0; h < H1; h++) if (d1[h] != 0.0f) ADAM(r + h, d1[h] * vv);
                }
                for (int k = 0; k < xf.nidx; k++) {
                    size_t r = (size_t)xf.idx[k] * H1;
                    for (int h = 0; h < H1; h++) if (d1[h] != 0.0f) ADAM(r + h, d1[h]);
                }
                for (int j = 0; j < XDENSE; j++) {
                    float vv = xf.dense[j];
                    if (vv == 0.0f) continue;
                    size_t r = (size_t)(FEAT_DIM + XBIN + j) * H1;
                    for (int h = 0; h < H1; h++) if (d1[h] != 0.0f) ADAM(r + h, d1[h] * vv);
                }
            }
            if ((ii + 1) % 100000 == 0) { fprintf(stderr, "e%d %ld/%ld\n", e, ii + 1, ntr); }
        }
        printf("xepoch %d: train BCE %.4f over %ld card-preds\n", e, bn ? bce / bn : 0.0, bn);
        xeval_net(x, recs, nrec, cut, e == epochs);
        fflush(stdout);
        belx_save(x, outp);
    }
    #undef ADAM
    free(m); free(v); free(idx);
    printf("saved belx net to %s\n", outp);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage:\n  %s gen NET GAMES SEED OUT.bst\n"
                        "  %s eval NET STATES.bst FROMGAME\n"
                        "  %s train NET STATES.bst OUT.bin EPOCHS LR HOLDGAMES\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    Net net;
    int needs_net = strcmp(argv[1], "xeval") != 0 && strcmp(argv[1], "xresume") != 0;
    if (needs_net && net_load(&net, argv[2]) != 0) { fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    if (!strcmp(argv[1], "gen") && argc >= 6) {
        gen(&net, atoi(argv[3]), strtoull(argv[4], NULL, 10), argv[5]);
    } else if (!strcmp(argv[1], "eval") && argc >= 5) {
        Rec *r; long n = load_bst(argv[3], &r);
        eval_net(&net, r, n, atoi(argv[4]), 1);
    } else if (!strcmp(argv[1], "train") && argc >= 8) {
        Rec *r; long n = load_bst(argv[3], &r);
        train_head(&net, r, n, argv[4], atoi(argv[5]), (float)atof(argv[6]), atoi(argv[7]));
    } else if (!strcmp(argv[1], "xtrain") && argc >= 8) {
        /* argv[2] = warm-start SPEC net; xtrain SPEC STATES OUT EPOCHS LR HOLD */
        BelX x; belx_from_net(&x, &net);
        Rec *r; long n = load_bst(argv[3], &r);
        xtrain(&x, r, n, argv[4], atoi(argv[5]), (float)atof(argv[6]), atoi(argv[7]));
    } else if (!strcmp(argv[1], "xresume") && argc >= 8) {
        /* xresume BLXFILE STATES OUT EPOCHS LR HOLD (argv[2] reused as blx path) */
        BelX x;
        if (belx_load(&x, argv[2]) != 0) { fprintf(stderr, "cannot load blx %s\n", argv[2]); return 1; }
        Rec *r; long n = load_bst(argv[3], &r);
        xtrain(&x, r, n, argv[4], atoi(argv[5]), (float)atof(argv[6]), atoi(argv[7]));
    } else if (!strcmp(argv[1], "xeval") && argc >= 5) {
        BelX x;
        if (belx_load(&x, argv[2]) != 0) { fprintf(stderr, "cannot load blx %s\n", argv[2]); return 1; }
        Rec *r; long n = load_bst(argv[3], &r);
        xeval_net(&x, r, n, atoi(argv[4]), 1);
    } else {
        fprintf(stderr, "bad arguments\n");
        return 1;
    }
    return 0;
}
