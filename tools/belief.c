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
#include "../src/belx.h"
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


/* sampeval: does the WORLD SAMPLER reproduce what the head knows?  For each
 * held-out state, draw M worlds with determinize_bm under the given mode
 * and score the per-card inclusion FREQUENCY against the true opponent
 * hand exactly as eval scores the head's marginals (BCE skill over the
 * counting prior, within-decision AUC, calibration), on the strictly
 * unknown card set, by deck phase.  Also reports duplicate-hand rate and
 * the mean number of playable-now cards in sampled vs real hands. */
static int samp_playable(const State *st, int p, int c)
{
    int s = CARD_SUIT(c);
    return CARD_IS_WAGER(c) ? (st->exp_top[p][s] == 0) : (CARD_VALUE(c) > st->exp_top[p][s]);
}

static void sampeval(const Net *net, const Net *bnet, const Rec *recs, long nrec,
                     int fromgame, int mode, int M, uint64_t seed, int stride, int symk)
{
    Acc unk = { 0 }, phase[3] = { { 0 } };
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD], ulab[NCARD];
    float freq[NCARD], up[NCARD];
    double dup[3] = { 0 }, dupn[3] = { 0 }, play_s[3] = { 0 }, play_t[3] = { 0 };
    Rng rng; rng_seed(&rng, seed);
    long states = 0;
    for (long i = 0; i < nrec; i += (stride > 0 ? stride : 1)) {
        if (recs[i].game < fromgame) continue;
        const State *st = &recs[i].st;
        for (int p = 0; p < 2; p++) {
            int o = p ^ 1;
            int n = cand_set(st, p, cards, lab, isk);
            if (n < 2) continue;
            int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
            if (need <= 0) continue;
            double cnt[NCARD] = { 0 };
            uint64_t seen_hands[256];
            int nseen = 0, dups = 0;
            double psum = 0.0;
            for (int m = 0; m < M; m++) {
                State w;
                if (symk > 0 && mode == 0) determinize_bsym(st, p, &rng, bnet ? bnet : net, NULL, symk, &w);
                else determinize_bm(st, p, &rng, bnet ? bnet : net, NULL, mode, &w);
                uint64_t h = w.hand[o];
                for (int k = 0; k < n; k++) if ((h >> cards[k]) & 1ULL) cnt[k] += 1.0;
                int found = 0;
                for (int q = 0; q < nseen; q++) if (seen_hands[q] == h) { found = 1; break; }
                if (found) dups++; else if (nseen < 256) seen_hands[nseen++] = h;
                uint64_t hh = h;
                while (hh) { int c = __builtin_ctzll(hh); hh &= hh - 1; psum += samp_playable(st, o, c); }
            }
            int ph = st->deck_left >= 30 ? 0 : (st->deck_left >= 15 ? 1 : 2);
            dup[ph] += (double)dups / M; dupn[ph] += 1.0;
            play_s[ph] += psum / M;
            { uint64_t hh = st->hand[o]; double t = 0; while (hh) { int c = __builtin_ctzll(hh); hh &= hh - 1; t += samp_playable(st, o, c); } play_t[ph] += t; }
            int un = 0, uheld = 0;
            for (int k = 0; k < n; k++) {
                if (isk[k]) continue;
                freq[k] = (float)(cnt[k] / M);
                up[un] = freq[k]; ulab[un] = lab[k]; uheld += lab[k]; un++;
            }
            if (un >= 2) {
                float uprior = (float)uheld / (float)un;
                acc_add(&unk, up, ulab, un, uprior);
                acc_add(&phase[ph], up, ulab, un, uprior);
            }
            states++;
        }
    }
    printf("sampeval mode %d symk %d, M=%d worlds/state, %ld states (games >= %d):\n", mode, symk, M, states, fromgame);
    acc_print("unknown", &unk);
    acc_print("  early", &phase[0]);
    acc_print("  mid", &phase[1]);
    acc_print("  late", &phase[2]);
    const char *pn[3] = { "early", "mid", "late" };
    for (int b = 0; b < 3; b++)
        if (dupn[b] > 0)
            printf("  %-5s duplicate-hand rate %.3f  playable-now cards: sampled %.2f vs real %.2f\n",
                   pn[b], dup[b] / dupn[b], play_s[b] / dupn[b], play_t[b] / dupn[b]);
    printf("  calibration (unknown set, sampled frequency -> observed):\n");
    for (int b = 0; b < 10; b++)
        if (unk.caln[b] >= 200)
            printf("    %.1f-%.1f: n=%-8ld mean %.3f  observed %.3f\n",
                   b / 10.0, (b + 1) / 10.0, unk.caln[b],
                   unk.calp[b] / unk.caln[b], unk.calh[b] / unk.caln[b]);
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



/* eval mirror of eval_net for the extended-format specialist */
static void xeval_net(const BelX *x, const Rec *recs, long nrec, int fromgame, int verbose)
{
    Acc all = { 0 }, unk = { 0 }, phase[3] = { { 0 } };
    Features f; BelXFeat xf; NetAct act;
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD];
    float logit[NCARD], prob[NCARD], up[NCARD];
    uint8_t ulab[NCARD];
    for (long i = 0; i < nrec; i++) {
        if (recs[i].game < fromgame) continue;
        const State *st = &recs[i].st;
        for (int p = 0; p < 2; p++) {
            int n = cand_set(st, p, cards, lab, isk);
            if (n < 2) continue;
            belx_feat(st, p, &f, &xf);
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
                   int epochs, float lr, float basescale, int holdgames)
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

    Features f; BelXFeat xf; NetAct act;
    uint8_t cards[NCARD], lab[NCARD], isk[NCARD];
    float lg[NCARD];
    float d2[NET_H2_MAX], d1[NET_H1_MAX];

    /* per-group learning rate: inherited parameters (base w1 rows and all
     * of b1/w2/b2/wb/bb) move at lr*basescale; the NEW feature rows
     * (w1 rows FEAT_DIM..BELX_XDIM) get the full lr.  The uniform-rate run
     * destroyed the inherited representation faster than the new signals
     * paid (holdout 6.4%% -> 4.1%% in two epochs). */
    const size_t newrow_lo = (size_t)FEAT_DIM * x->h1;
    const size_t newrow_hi = (size_t)BELX_XDIM * x->h1;
    #define ADAM(off, g) do { \
        size_t _o = (off); float _g = (g); \
        float _lr = (_o >= newrow_lo && _o < newrow_hi) ? lr : lr * basescale; \
        if (_lr != 0.0f) { \
            m[_o] = 0.9f * m[_o] + 0.1f * _g; \
            v[_o] = 0.999f * v[_o] + 0.001f * _g * _g; \
            x->blk[_o] -= _lr * m[_o] / (sqrtf(v[_o]) + 1e-8f); \
        } \
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
                belx_feat(st, p, &f, &xf);
                belx_trunk(x, &f, &xf, &act);
                belx_logits(x, &act, cards, n, lg);
                float scale = 1.0f / (float)n;
                for (int h = 0; h < H2; h++) d2[h] = 0.0f;
                size_t off_wb = (size_t)BELX_XDIM * H1 + H1 + (size_t)H1 * H2 + H2;
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
                size_t off_w2 = (size_t)BELX_XDIM * H1 + H1;
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
                size_t off_b1 = (size_t)BELX_XDIM * H1;
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
                for (int j = 0; j < BELX_XDENSE; j++) {
                    float vv = xf.dense[j];
                    if (vv == 0.0f) continue;
                    size_t r = (size_t)(FEAT_DIM + BELX_XBIN + j) * H1;
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
                        "  %s sampeval NET BELNET|- STATES.bst FROMGAME MODE [M] [SEED] [STRIDE] [SYMK]\n"
                        "  %s train NET STATES.bst OUT.bin EPOCHS LR HOLDGAMES\n",
                argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    Net net;
    int needs_net = strcmp(argv[1], "xeval") != 0 && strcmp(argv[1], "xresume") != 0;
    if (needs_net && net_load(&net, argv[2]) != 0) { fprintf(stderr, "cannot load %s\n", argv[2]); return 1; }
    if (!strcmp(argv[1], "gen") && argc >= 6) {
        gen(&net, atoi(argv[3]), strtoull(argv[4], NULL, 10), argv[5]);
    } else if (!strcmp(argv[1], "sampeval") && argc >= 7) {
        /* sampeval NET BELNET|- STATES.bst FROMGAME MODE [M] [SEED] */
        Net *bn = NULL;
        if (strcmp(argv[3], "-")) {
            bn = (Net *)malloc(sizeof(Net));
            if (net_load(bn, argv[3])) { fprintf(stderr, "cannot load %s\n", argv[3]); return 1; }
        }
        Rec *r; long n = load_bst(argv[4], &r);
        sampeval(&net, bn, r, n, atoi(argv[5]), atoi(argv[6]),
                 argc > 7 ? atoi(argv[7]) : 96, argc > 8 ? strtoull(argv[8], NULL, 10) : 4242,
                 argc > 9 ? atoi(argv[9]) : 1, argc > 10 ? atoi(argv[10]) : 0);
    } else if (!strcmp(argv[1], "eval") && argc >= 5) {
        Rec *r; long n = load_bst(argv[3], &r);
        eval_net(&net, r, n, atoi(argv[4]), 1);
    } else if (!strcmp(argv[1], "train") && argc >= 8) {
        Rec *r; long n = load_bst(argv[3], &r);
        train_head(&net, r, n, argv[4], atoi(argv[5]), (float)atof(argv[6]), atoi(argv[7]));
    } else if (!strcmp(argv[1], "xtrain") && argc >= 9) {
        /* xtrain SPECNET STATES OUT EPOCHS LR BASESCALE HOLD */
        BelX x; belx_from_net(&x, &net);
        Rec *r; long n = load_bst(argv[3], &r);
        xtrain(&x, r, n, argv[4], atoi(argv[5]), (float)atof(argv[6]), (float)atof(argv[7]), atoi(argv[8]));
    } else if (!strcmp(argv[1], "xresume") && argc >= 9) {
        /* xresume BLXFILE STATES OUT EPOCHS LR BASESCALE HOLD */
        BelX x;
        if (belx_load(&x, argv[2]) != 0) { fprintf(stderr, "cannot load blx %s\n", argv[2]); return 1; }
        Rec *r; long n = load_bst(argv[3], &r);
        xtrain(&x, r, n, argv[4], atoi(argv[5]), (float)atof(argv[6]), (float)atof(argv[7]), atoi(argv[8]));
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
