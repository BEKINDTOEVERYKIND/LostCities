/* rl -- self-play policy optimisation (PPO) for the Lost Cities network.
 *
 * Why policy gradient rather than expert iteration: candidate moves in this
 * game differ by one or two points while a finished game's margin swings by
 * sixty, so no value function accurate enough to rank moves by one-ply
 * lookahead is learnable, and a search built on such a value function is no
 * stronger than the policy that seeded it (measured, not assumed).  What does
 * work is improving the policy directly from played outcomes: the value head
 * only has to serve as a baseline, where its errors cancel instead of
 * corrupting the ranking.
 *
 * Both seats are the same network, so every game yields training data from two
 * points of view.  Generation is one forward pass per ply, which is fast enough
 * that the data is always fresh and on-policy.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/heuristic.h"
#include "../src/match.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    State st;
    float vtarget;    /* lambda-return, points, perspective player's view */
    float adv;        /* advantage, points (actor samples only)           */
    float oldp;       /* policy probability of the move actually played   */
    uint16_t chosen;  /* packed move                                      */
    uint8_t persp;
    uint8_t actor;    /* 1 when persp is the player who moved             */
} RLSample;

typedef struct {
    const Net *net;
    int games;
    uint64_t seed;
    int thread, nthread;
    RLSample *out;
    size_t nout, cap;
    double plies, absmargin, score_sum, entropy;
    long entropy_n;
    int done;
    float lambda;
    float temp;
} GenJob;

static _Thread_local State chain[LC_MAX_PLIES + 2];
static _Thread_local float chain_v[2][LC_MAX_PLIES + 2];
static _Thread_local uint16_t chain_mv[LC_MAX_PLIES + 2];
static _Thread_local float chain_p[LC_MAX_PLIES + 2];

static void *gen_worker(void *arg)
{
    GenJob *j = (GenJob *)arg;
    Rng rng; rng_seed(&rng, j->seed + 0x9E3779B9ULL * (uint64_t)(j->thread + 1));
    Features f;
    Move mv[MAX_MOVES];
    float pr[MAX_MOVES];

    for (int g = j->thread; g < j->games; g += j->nthread) {
        State st;
        lc_deal(&st, &rng);
        int T = 0;
        while (!st.over && T < LC_MAX_PLIES) {
            chain[T] = st;
            int n = policy_probs(j->net, &st, mv, pr, NULL);
            double h = 0.0;
            for (int i = 0; i < n; i++) if (pr[i] > 1e-9f) h -= pr[i] * log(pr[i]);
            j->entropy += h; j->entropy_n++;

            int c;
            if (j->temp != 1.0f) {
                float w[MAX_MOVES];
                for (int i = 0; i < n; i++) w[i] = powf(pr[i], 1.0f / j->temp);
                c = sample_index(w, n, &rng);
            } else {
                c = sample_index(pr, n, &rng);
            }
            chain_mv[T] = MOVE_PACK(mv[c]);
            chain_p[T] = pr[c];
            T++;
            lc_apply(&st, mv[c]);
        }
        int score[2] = { lc_score(&st, 0), lc_score(&st, 1) };

        /* values of every state from both points of view */
        for (int p = 0; p < 2; p++)
            for (int t = 0; t < T; t++) {
                feat_extract(&chain[t], p, &f);
                chain_v[p][t] = net_value(j->net, &f) * VAL_SCALE;
            }

        for (int p = 0; p < 2; p++) {
            float G = (float)(score[p] - score[p ^ 1]);
            for (int t = T - 1; t >= 0; t--) {
                if (t < T - 1) G = (1.0f - j->lambda) * chain_v[p][t + 1] + j->lambda * G;
                if (j->nout >= j->cap) continue;
                RLSample *s = &j->out[j->nout++];
                s->st = chain[t];
                s->persp = (uint8_t)p;
                s->vtarget = G;
                if (chain[t].turn == p) {
                    s->actor = 1;
                    s->chosen = chain_mv[t];
                    s->oldp = chain_p[t];
                    s->adv = G - chain_v[p][t];
                } else {
                    s->actor = 0;
                    s->chosen = 0;
                    s->oldp = 1.0f;
                    s->adv = 0.0f;
                }
            }
        }
        j->plies += st.nply;
        j->absmargin += fabs((double)(score[0] - score[1]));
        j->score_sum += score[0] + score[1];
        j->done++;
    }
    return NULL;
}

/* ---------------- optimisation ---------------------------------------- */

typedef struct {
    const Net *net;
    Net *grad;
    const RLSample *buf;
    const int *idx;
    int from, to;
    float clip, vcoef, entcoef, advscale;
    double ploss, vloss, clipped;
    int pn;
} OptJob;

static void *opt_worker(void *arg)
{
    OptJob *t = (OptJob *)arg;
    net_zero(t->grad);
    double ploss = 0, vloss = 0, clipped = 0;
    int pn = 0;
    Features f;
    NetAct act;
    Move mv[MAX_MOVES];
    uint16_t pk[MAX_MOVES];
    float logit[MAX_MOVES], prob[MAX_MOVES], dlog[MAX_MOVES];

    for (int i = t->from; i < t->to; i++) {
        const RLSample *s = &t->buf[t->idx[i]];
        feat_extract(&s->st, s->persp, &f);
        net_trunk(t->net, &f, &act);

        float v = net_value_act(t->net, &act);
        float y = s->vtarget / VAL_SCALE;
        float e = v - y;
        vloss += (double)e * e;
        float dvalue = 2.0f * e * t->vcoef;

        int n = 0;
        if (s->actor) {
            n = lc_moves(&s->st, mv);
            int ci = -1;
            for (int k = 0; k < n; k++) {
                pk[k] = MOVE_PACK(mv[k]);
                if (pk[k] == s->chosen) ci = k;
            }
            if (ci < 0) { net_backward(t->net, &f, &act, dvalue, pk, NULL, 0, t->grad); continue; }
            net_policy_act(t->net, &act, pk, n, logit);
            float mx = logit[0];
            for (int k = 1; k < n; k++) if (logit[k] > mx) mx = logit[k];
            float sum = 0.0f;
            for (int k = 0; k < n; k++) { prob[k] = expf(logit[k] - mx); sum += prob[k]; }
            float inv = 1.0f / sum;
            float ent = 0.0f;
            for (int k = 0; k < n; k++) {
                prob[k] *= inv;
                if (prob[k] > 1e-9f) ent -= prob[k] * logf(prob[k]);
            }
            float A = s->adv * t->advscale;
            float ratio = prob[ci] / (s->oldp > 1e-9f ? s->oldp : 1e-9f);
            float lo = 1.0f - t->clip, hi = 1.0f + t->clip;
            /* PPO: gradient flows only when the unclipped branch is the
             * binding one, which is what stops a single batch from moving the
             * policy too far off the data it was collected under. */
            int use = 1;
            if (ratio > hi && A > 0.0f) use = 0;
            if (ratio < lo && A < 0.0f) use = 0;
            if (!use) clipped += 1.0;
            ploss += -(double)(ratio < lo ? lo : (ratio > hi ? hi : ratio)) * A;
            float gsurr = use ? -A * ratio : 0.0f;
            for (int k = 0; k < n; k++) {
                float dsurr = gsurr * ((k == ci ? 1.0f : 0.0f) - prob[k]);
                float dent = t->entcoef * prob[k] * (logf(prob[k] + 1e-9f) + ent);
                dlog[k] = dsurr + dent;
            }
            pn++;
            net_backward(t->net, &f, &act, dvalue, pk, dlog, n, t->grad);
        } else {
            net_backward(t->net, &f, &act, dvalue, pk, NULL, 0, t->grad);
        }
    }
    t->ploss = ploss; t->vloss = vloss; t->pn = pn; t->clipped = clipped;
    return NULL;
}

static void grad_accumulate(Net *dst, Net *const *src, int n)
{
    float *d = (float *)dst;
    size_t nw = sizeof(Net) / sizeof(float);
    for (int k = 1; k < n; k++) {
        const float *s = (const float *)src[k];
        for (size_t i = 0; i < nw; i++) d[i] += s[i];
    }
}

int main(int argc, char **argv)
{
    const char *out_path = "data/rl.bin";
    const char *init_path = "data/base.bin";
    const char *ref_spec = "heur";
    int iters = 30, games = 4000, nthread = 4, batch = 512, epochs = 2;
    int eval_pairs = 400, eval_every = 1;
    float lr = 3e-4f, wd = 1e-7f, lambda = 0.85f, clip = 0.2f;
    float vcoef = 1.0f, entcoef = 0.004f, temp = 1.0f;
    uint64_t seed = 7;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        #define ARG(name) (!strcmp(k, name) && i + 1 < argc)
        if (ARG("--out")) out_path = argv[++i];
        else if (ARG("--init")) init_path = argv[++i];
        else if (ARG("--ref")) ref_spec = argv[++i];
        else if (ARG("--iters")) iters = atoi(argv[++i]);
        else if (ARG("--games")) games = atoi(argv[++i]);
        else if (ARG("--threads")) nthread = atoi(argv[++i]);
        else if (ARG("--batch")) batch = atoi(argv[++i]);
        else if (ARG("--epochs")) epochs = atoi(argv[++i]);
        else if (ARG("--lr")) lr = (float)atof(argv[++i]);
        else if (ARG("--lambda")) lambda = (float)atof(argv[++i]);
        else if (ARG("--clip")) clip = (float)atof(argv[++i]);
        else if (ARG("--vcoef")) vcoef = (float)atof(argv[++i]);
        else if (ARG("--ent")) entcoef = (float)atof(argv[++i]);
        else if (ARG("--temp")) temp = (float)atof(argv[++i]);
        else if (ARG("--wd")) wd = (float)atof(argv[++i]);
        else if (ARG("--eval")) eval_pairs = atoi(argv[++i]);
        else if (ARG("--eval-every")) eval_every = atoi(argv[++i]);
        else if (ARG("--seed")) seed = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown option %s\n", k); return 1; }
        #undef ARG
    }

    Net *net = (Net *)malloc(sizeof(Net));
    Net *frozen = (Net *)malloc(sizeof(Net));
    Adam *adam = (Adam *)calloc(1, sizeof(Adam));
    if (net_load(net, init_path) != 0) { fprintf(stderr, "cannot load %s\n", init_path); return 1; }
    printf("initialised from %s\n", init_path);

    Agent ref;
    spec_parse(ref_spec, &ref);

    Net **grads = (Net **)calloc((size_t)nthread, sizeof(Net *));
    for (int i = 0; i < nthread; i++) grads[i] = (Net *)malloc(sizeof(Net));

    size_t cap = (size_t)games * 210;
    RLSample *buf = (RLSample *)malloc(sizeof(RLSample) * cap);
    if (!buf) { fprintf(stderr, "sample buffer allocation failed\n"); return 1; }
    int *order = (int *)malloc(sizeof(int) * cap);

    printf("ppo: %d iters x %d games, batch %d, %d epochs, lr %.1e, lambda %.2f, ent %.4f\n",
           iters, games, batch, epochs, lr, lambda, entcoef);
    fflush(stdout);

    for (int it = 1; it <= iters; it++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(frozen, net, sizeof(Net));

        GenJob *jobs = (GenJob *)calloc((size_t)nthread, sizeof(GenJob));
        pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
        size_t per = cap / (size_t)nthread;
        for (int i = 0; i < nthread; i++) {
            jobs[i].net = frozen;
            jobs[i].games = games;
            jobs[i].seed = seed * 7919ULL + (uint64_t)it * 104729ULL;
            jobs[i].thread = i; jobs[i].nthread = nthread;
            jobs[i].out = buf + per * (size_t)i;
            jobs[i].cap = per;
            jobs[i].lambda = lambda;
            jobs[i].temp = temp;
        }
        for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, gen_worker, &jobs[i]);
        for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

        /* compact the per-thread blocks into one contiguous array */
        size_t n = 0;
        for (int i = 0; i < nthread; i++) {
            if (jobs[i].out != buf + n) memmove(buf + n, jobs[i].out, jobs[i].nout * sizeof(RLSample));
            n += jobs[i].nout;
        }
        double plies = 0, absm = 0, pts = 0, ent = 0;
        long entn = 0;
        int gdone = 0;
        for (int i = 0; i < nthread; i++) {
            plies += jobs[i].plies; absm += jobs[i].absmargin; pts += jobs[i].score_sum;
            ent += jobs[i].entropy; entn += jobs[i].entropy_n;
            gdone += jobs[i].done;
        }
        free(jobs); free(th);

        /* standardise advantages */
        double am = 0, av = 0;
        long an = 0;
        for (size_t i = 0; i < n; i++) if (buf[i].actor) { am += buf[i].adv; an++; }
        am /= (an ? an : 1);
        for (size_t i = 0; i < n; i++) if (buf[i].actor) { double d = buf[i].adv - am; av += d * d; }
        av = sqrt(av / (an ? an : 1)) + 1e-6;
        for (size_t i = 0; i < n; i++) if (buf[i].actor) buf[i].adv = (float)((buf[i].adv - am) / av);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double gs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("iter %2d: %d games %.1fs (%.0f g/s), %zu samples, plies %.1f, "
               "points/game %.1f, |margin| %.1f, entropy %.2f, adv sd %.1f\n",
               it, gdone, gs, gdone / gs, n, plies / gdone, pts / (2 * gdone),
               absm / gdone, ent / entn, av);
        fflush(stdout);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        Rng r; rng_seed(&r, seed + 31ULL * (uint64_t)it);
        double pl = 0, vl = 0, cl = 0;
        long pn = 0, steps = 0;
        for (int ep = 0; ep < epochs; ep++) {
            for (size_t i = 0; i < n; i++) order[i] = (int)i;
            for (size_t i = n - 1; i > 0; i--) {
                uint32_t jx = rng_below(&r, (uint32_t)i + 1);
                int t = order[i]; order[i] = order[jx]; order[jx] = t;
            }
            for (size_t off = 0; off + (size_t)batch <= n; off += (size_t)batch) {
                OptJob tj[64];
                pthread_t tt[64];
                int nt = nthread > 64 ? 64 : nthread;
                int chunk = (batch + nt - 1) / nt;
                for (int i = 0; i < nt; i++) {
                    tj[i].net = net; tj[i].grad = grads[i]; tj[i].buf = buf;
                    tj[i].idx = order + off;
                    tj[i].from = i * chunk > batch ? batch : i * chunk;
                    tj[i].to = (i + 1) * chunk > batch ? batch : (i + 1) * chunk;
                    tj[i].clip = clip; tj[i].vcoef = vcoef; tj[i].entcoef = entcoef;
                    tj[i].advscale = 1.0f;
                }
                for (int i = 0; i < nt; i++) pthread_create(&tt[i], NULL, opt_worker, &tj[i]);
                for (int i = 0; i < nt; i++) pthread_join(tt[i], NULL);
                for (int i = 0; i < nt; i++) { pl += tj[i].ploss; vl += tj[i].vloss; pn += tj[i].pn; cl += tj[i].clipped; }
                grad_accumulate(grads[0], grads, nt);
                net_adam_step(net, grads[0], adam, lr, 1.0f / (float)batch, wd);
                steps++;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ts = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("         %ld updates in %.1fs: value rmse %.1f pts, surrogate %.4f, clipped %.1f%%\n",
               steps, ts, sqrt(vl / ((double)steps * batch)) * VAL_SCALE,
               pn ? pl / pn : 0.0, pn ? 100.0 * cl / pn : 0.0);
        fflush(stdout);

        char path[512];
        net_save(net, out_path);
        snprintf(path, sizeof path, "%s.it%d", out_path, it);
        net_save(net, path);

        if (eval_pairs > 0 && (it % eval_every == 0 || it == iters)) {
            Agent cur;
            agent_default(&cur, AG_POLICY, net);
            MatchResult mr;
            match_run(&cur, &ref, eval_pairs, nthread, 20260727 + (uint64_t)it, &mr);
            printf("         vs %s: margin %+.2f +- %.2f, score %.1f%%, plies %.0f\n",
                   ref_spec, mr.margin, mr.margin_se, 100 * mr.winrate, mr.plies);
            fflush(stdout);
        }
    }
    return 0;
}
