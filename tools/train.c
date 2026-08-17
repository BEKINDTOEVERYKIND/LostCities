/* train -- self-play training for the two-headed Lost Cities network.
 *
 * Each iteration generates games with the current expert (the hand-crafted
 * policy at first, then determinized search over the network), records every
 * information state with a policy target and a value target, and fits the
 * network to them.  A better network makes the next expert stronger: expert
 * iteration.
 *
 * Value targets are lambda-returns rather than raw game outcomes.  A finished
 * game's margin has a standard deviation near 60 points while good and bad
 * moves differ by a couple of points, so bootstrapping is not optional here.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/heuristic.h"
#include "../src/match.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define PI_K 12   /* policy target entries kept per sample */

typedef struct {
    State st;
    float target;        /* value target, points, perspective player's view */
    uint8_t persp;
    uint8_t npi;
    uint16_t pmv[PI_K];  /* packed moves */
    float ppr[PI_K];     /* target probabilities */
} Sample;

typedef struct {
    Sample *buf;
    size_t cap, n, head;
} Replay;

static void replay_init(Replay *r, size_t cap)
{
    r->buf = (Sample *)malloc(sizeof(Sample) * cap);
    if (!r->buf) { fprintf(stderr, "replay buffer allocation failed\n"); exit(1); }
    r->cap = cap; r->n = 0; r->head = 0;
}

static void replay_push(Replay *r, const Sample *s)
{
    r->buf[r->head] = *s;
    r->head = (r->head + 1) % r->cap;
    if (r->n < r->cap) r->n++;
}

/* ---------------- self-play generation ------------------------------- */

typedef struct {
    Agent agent;
    int games;           /* matches per thread quota */
    uint64_t seed;
    int thread, nthread;
    Sample *out;
    size_t nout, cap;
    double sum_margin, sum_abs, plies;
    int done;
    float lambda;
    float tau;           /* softmax temperature for value-based experts */
    float winbonus;      /* terminal bonus for winning the match, points */
    int sample_plies;    /* sample from the target policy below this ply */
    int rounds;
    const Net *net;
} GenJob;

/* Keep the PI_K largest entries of a distribution, renormalized. */
static void topk(const Move *mv, const float *pr, int n, uint16_t *omv, float *opr, uint8_t *on)
{
    int idx[MAX_MOVES];
    for (int i = 0; i < n; i++) idx[i] = i;
    int k = n < PI_K ? n : PI_K;
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (pr[idx[j]] > pr[idx[best]]) best = j;
        int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
    }
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += pr[idx[i]];
    if (sum <= 0.0f) sum = 1.0f;
    for (int i = 0; i < k; i++) {
        omv[i] = MOVE_PACK(mv[idx[i]]);
        opr[i] = pr[idx[i]] / sum;
    }
    for (int i = k; i < PI_K; i++) { omv[i] = 0; opr[i] = 0.0f; }
    *on = (uint8_t)k;
}

#define CHAIN_MAX (MATCH_ROUNDS * LC_MAX_PLIES + 4)
static _Thread_local State chain[CHAIN_MAX];
static _Thread_local uint16_t chain_pmv[CHAIN_MAX][PI_K];
static _Thread_local float chain_ppr[CHAIN_MAX][PI_K];
static _Thread_local uint8_t chain_npi[CHAIN_MAX];
static _Thread_local float chain_sval[CHAIN_MAX];
static _Thread_local uint8_t chain_hasv[CHAIN_MAX];

static void *gen_worker(void *arg)
{
    GenJob *j = (GenJob *)arg;
    Rng rng; rng_seed(&rng, j->seed + 0x1234567ULL * (uint64_t)(j->thread + 1));
    Features f;

    for (int g = j->thread; g < j->games; g += j->nthread) {
        int T = 0;
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < j->rounds; rd++) {
        State st;
        lc_deal(&st, &rng);
        st.round = (uint8_t)rd;
        st.cum[0] = (int16_t)(cum[0] > 320 ? 320 : (cum[0] < -320 ? -320 : cum[0]));
        st.cum[1] = (int16_t)(cum[1] > 320 ? 320 : (cum[1] < -320 ? -320 : cum[1]));
        st.turn = (uint8_t)(rd & 1);
        int Tstop = T + LC_MAX_PLIES;
        while (!st.over && T < Tstop) {
            chain[T] = st;
            Move mv[MAX_MOVES];
            float pr[MAX_MOVES];
            int n = 0;
            int gated_idx = -1;
            chain_hasv[T] = 0;

            if (j->agent.kind == AG_ROLLOUT) {
                SearchStats ss;
                float sv = 0.0f;
                Move gm = rollout_move(&j->agent, &st, &rng, &sv, &ss);
                n = ss.n;
                /* Both behavior and targets follow the GATED selection rule
                 * (rollout_move's returned move), not the raw Q-argmax --
                 * the ungated argmax selects among paired noise and measured
                 * 42.8% against the baseline (the c1-c8 label-bug family).
                 * The tau softmax over Q survives only as the exploration
                 * distribution for early-ply sampling. */
                double mx = -1e30;
                for (int i = 0; i < n; i++) if (ss.q[i] > mx) mx = ss.q[i];
                double sum = 0.0;
                int gi = -1;
                for (int i = 0; i < n; i++) {
                    mv[i] = ss.mv[i];
                    pr[i] = (float)exp((ss.q[i] - mx) / j->tau);
                    sum += pr[i];
                    if (mv[i].card == gm.card && mv[i].discard == gm.discard &&
                        mv[i].draw == gm.draw) gi = i;
                }
                for (int i = 0; i < n; i++) pr[i] /= (float)sum;
                gated_idx = gi;
                chain_sval[T] = sv; chain_hasv[T] = 1;
            } else if (j->agent.kind == AG_MCTS) {
                SearchStats ss;
                float sv = 0.0f;
                search_move(&j->agent, &st, &rng, &sv, &ss);
                n = ss.n;
                double tot = 0.0;
                for (int i = 0; i < n; i++) tot += ss.visits[i];
                for (int i = 0; i < n; i++) {
                    mv[i] = ss.mv[i];
                    pr[i] = tot > 0 ? (float)(ss.visits[i] / tot) : 1.0f / (float)n;
                }
                chain_sval[T] = sv; chain_hasv[T] = 1;
            } else if (j->agent.kind == AG_POLICY) {
                n = policy_probs(j->net, &st, mv, pr, NULL);
            } else {
                float val[MAX_MOVES];
                n = agent_move_values(&j->agent, &st, &rng, mv, val);
                float mx = -1e30f;
                for (int i = 0; i < n; i++) if (val[i] > mx) mx = val[i];
                float sum = 0.0f;
                for (int i = 0; i < n; i++) { pr[i] = expf((val[i] - mx) / j->tau); sum += pr[i]; }
                for (int i = 0; i < n; i++) pr[i] /= sum;
            }

            if (gated_idx >= 0) {
                /* target = one-hot on the gated selection (the validated
                 * labeling rule); pr stays the exploration distribution */
                float tgt[MAX_MOVES];
                for (int i = 0; i < n; i++) tgt[i] = (i == gated_idx) ? 1.0f : 0.0f;
                topk(mv, tgt, n, chain_pmv[T], chain_ppr[T], &chain_npi[T]);
            } else {
                topk(mv, pr, n, chain_pmv[T], chain_ppr[T], &chain_npi[T]);
            }

            int chosen;
            if (T < j->sample_plies) chosen = sample_index(pr, n, &rng);
            else if (gated_idx >= 0) chosen = gated_idx;
            else { chosen = 0; for (int i = 1; i < n; i++) if (pr[i] > pr[chosen]) chosen = i; }

            T++;
            lc_apply(&st, mv[chosen]);
        }
        cum[0] += lc_score(&st, 0);
        cum[1] += lc_score(&st, 1);
        j->plies += st.nply;
        }   /* rounds */
        int score[2] = { cum[0], cum[1] };

        for (int p = 0; p < 2; p++) {
            float G = (float)(score[p] - score[p ^ 1]);
            if (score[p] > score[p ^ 1]) G += j->winbonus;
            else if (score[p] < score[p ^ 1]) G -= j->winbonus;
            for (int t = T - 1; t >= 0; t--) {
                if (t < T - 1 && j->lambda < 0.999f) {
                    float vnext;
                    if (chain_hasv[t + 1]) {
                        /* search values are THIS-ROUND margins while G is
                         * match-scale: shift by the known cumulative
                         * context (future-round margin unknowable here) */
                        float rv = (chain[t + 1].turn == p) ? chain_sval[t + 1] : -chain_sval[t + 1];
                        vnext = rv + (float)(chain[t + 1].cum[p] - chain[t + 1].cum[p ^ 1]);
                    } else {
                        feat_extract(&chain[t + 1], p, &f);
                        vnext = net_value(j->net, &f) * VAL_SCALE;
                    }
                    G = (1.0f - j->lambda) * vnext + j->lambda * G;
                }
                if (j->nout >= j->cap) continue;
                Sample *s = &j->out[j->nout++];
                s->st = chain[t];
                s->persp = (uint8_t)p;
                s->target = G;
                /* a policy target only exists for the player who moved */
                if (chain[t].turn == p) {
                    s->npi = chain_npi[t];
                    memcpy(s->pmv, chain_pmv[t], sizeof(s->pmv));
                    memcpy(s->ppr, chain_ppr[t], sizeof(s->ppr));
                } else {
                    s->npi = 0;
                }
            }
        }
        j->sum_margin += score[0] - score[1];
        j->sum_abs += fabs((double)(score[0] - score[1]));
        j->done++;
    }
    return NULL;
}

/* ---------------- training ------------------------------------------- */

typedef struct {
    const Net *net;
    Net *grad;
    const Replay *rp;
    const size_t *idx;
    int from, to;
    float pw;
    float vw;            /* value-loss weight: 0 for policy/belief-only
                            fine-tunes -- a win-trained value head predicts
                            on a return scale far from margin targets, and
                            the resulting gradients deform the shared trunk
                            (the c1/c2 collapse mechanism) */
    float bw;            /* belief BCE weight (rl.c trains this head too;
                            without it a fine-tune drifts the shared trunk
                            out from under the belief head) */
    int aug;             /* symmetry augmentation: present each sample under
                            a random relabeling of the 5 interchangeable
                            suits and 3 identical wager copies per suit */
    Rng rng;
    double vloss, ploss;
    int pn;
} TrainJob;

static void *train_worker(void *arg)
{
    TrainJob *t = (TrainJob *)arg;
    net_zero(t->grad);
    double vloss = 0.0, ploss = 0.0;
    int pn = 0;
    Features f;
    NetAct act;
    Move mv[MAX_MOVES];
    uint16_t pk[MAX_MOVES];
    float logit[MAX_MOVES], prob[MAX_MOVES], dlog[MAX_MOVES], tgt[MAX_MOVES];
    uint8_t bcard[NCARD];
    float blogit[NCARD], dbel[NCARD];

    Sample augs;
    for (int i = t->from; i < t->to; i++) {
        const Sample *s = &t->rp->buf[t->idx[i]];
        if (t->aug) {
            augs = *s;
            int sperm[NSUIT], wperm[NSUIT][WAGERS_PER_SUIT];
            for (int a = 0; a < NSUIT; a++) sperm[a] = a;
            for (int a = NSUIT - 1; a > 0; a--) {
                int b = (int)(rng_next(&t->rng) % (uint64_t)(a + 1));
                int tm = sperm[a]; sperm[a] = sperm[b]; sperm[b] = tm;
            }
            for (int su = 0; su < NSUIT; su++) {
                for (int a = 0; a < WAGERS_PER_SUIT; a++) wperm[su][a] = a;
                for (int a = WAGERS_PER_SUIT - 1; a > 0; a--) {
                    int b = (int)(rng_next(&t->rng) % (uint64_t)(a + 1));
                    int tm = wperm[su][a]; wperm[su][a] = wperm[su][b]; wperm[su][b] = tm;
                }
            }
            uint8_t map[NCARD];
            lc_perm_map(sperm, wperm, map);
            lc_permute(&augs.st, map);
            for (int a = 0; a < augs.npi; a++)
                augs.pmv[a] = lc_permute_pack(augs.pmv[a], map);
            s = &augs;
        }
        feat_extract(&s->st, s->persp, &f);
        net_trunk(t->net, &f, &act);

        float v = net_value_act(t->net, &act);
        float y = s->target / VAL_SCALE;
        float e = v - y;
        vloss += (double)e * e;

        int n = 0;
        if (s->npi > 0) {
            n = lc_moves(&s->st, mv);
            for (int k = 0; k < n; k++) { pk[k] = MOVE_PACK(mv[k]); tgt[k] = 0.0f; }
            for (int a = 0; a < s->npi; a++)
                for (int k = 0; k < n; k++)
                    if (pk[k] == s->pmv[a]) { tgt[k] = s->ppr[a]; break; }
            net_policy_act(t->net, &act, pk, n, logit);
            float mx = logit[0];
            for (int k = 1; k < n; k++) if (logit[k] > mx) mx = logit[k];
            float sum = 0.0f;
            for (int k = 0; k < n; k++) { prob[k] = expf(logit[k] - mx); sum += prob[k]; }
            float inv = 1.0f / sum;
            for (int k = 0; k < n; k++) prob[k] *= inv;
            for (int k = 0; k < n; k++) {
                if (tgt[k] > 0.0f) ploss -= (double)tgt[k] * log((double)prob[k] + 1e-9);
                dlog[k] = t->pw * (prob[k] - tgt[k]);
            }
            pn++;
        }
        /* belief head, exactly as rl.c trains it: the sample's state stores
         * the true opponent hand, a free supervised label */
        int nb = 0;
        if (t->bw > 0.0f) {
            const State *st = &s->st;
            int p = s->persp, o = p ^ 1;
            uint64_t vis = st->hand[p] | st->played[0] | st->played[1] | st->discarded;
            uint64_t cands = ~vis & ((1ULL << NCARD) - 1);
            while (cands) {
                int cc = __builtin_ctzll(cands);
                cands &= cands - 1;
                bcard[nb++] = (uint8_t)cc;
            }
            net_belief_act(t->net, &act, bcard, nb, blogit);
            float scale = t->bw / (float)(nb > 0 ? nb : 1);
            for (int k = 0; k < nb; k++) {
                float lab = ((st->hand[o] >> bcard[k]) & 1ULL) ? 1.0f : 0.0f;
                float l = blogit[k];
                if (l > 15.0f) l = 15.0f;
                if (l < -15.0f) l = -15.0f;
                float pr2 = 1.0f / (1.0f + expf(-l));
                dbel[k] = scale * (pr2 - lab);
            }
        }
        net_backward(t->net, &f, &act, 2.0f * e * t->vw, pk, s->npi > 0 ? dlog : NULL, n,
                     nb > 0 ? bcard : NULL, nb > 0 ? dbel : NULL, nb, t->grad);
    }
    t->vloss = vloss; t->ploss = ploss; t->pn = pn;
    return NULL;
}

static void grad_accumulate(Net *dst, Net *const *src, int n)
{
    float *d = dst->blk;
    size_t nw = dst->nfloat;
    for (int k = 1; k < n; k++) {
        const float *s = src[k]->blk;
        for (size_t i = 0; i < nw; i++) d[i] += s[i];
    }
}

int main(int argc, char **argv)
{
    const char *out_path = "data/net.bin";
    const char *init_path = NULL;
    const char *gen_spec = NULL;
    const char *ref_spec = "heur";
    const char *eval_spec = NULL;
    int iters = 10, games = 2000, nthread = 4, batch = 256, steps = 6000;
    int eval_pairs = 300;
    size_t bufcap = 800000;
    float lr = 1e-3f, wd = 1e-7f, tau = 1.0f, pw = 1.0f, lambda = 0.75f, bw = 1.0f, vw = 1.0f;
    int aug = 0;
    float winbonus = 15.0f;
    int rounds = MATCH_ROUNDS;
    int sample_plies = 24;
    uint64_t seed = 1;
    int keep_lr_flat = 0;
    int gen_switch = 1;
    int gen_dets = 12, gen_sims = 100, gen_rw = 12, gen_nw = 8;
    int h1 = NET_H1_DEF, h2 = NET_H2_DEF;   /* from-scratch width; --init overrides from file */
    const char *dump_path = NULL;   /* write generated samples to this file  */
    const char *data_path = NULL;   /* train from this file, no generation   */

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        #define ARG(name) (!strcmp(k, name) && i + 1 < argc)
        if (ARG("--out")) out_path = argv[++i];
        else if (ARG("--init")) init_path = argv[++i];
        else if (ARG("--gen")) gen_spec = argv[++i];
        else if (ARG("--ref")) ref_spec = argv[++i];
        else if (ARG("--eval-agent")) eval_spec = argv[++i];
        else if (ARG("--iters")) iters = atoi(argv[++i]);
        else if (ARG("--games")) games = atoi(argv[++i]);
        else if (ARG("--threads")) nthread = atoi(argv[++i]);
        else if (ARG("--batch")) batch = atoi(argv[++i]);
        else if (ARG("--steps")) steps = atoi(argv[++i]);
        else if (ARG("--buffer")) bufcap = (size_t)atol(argv[++i]);
        else if (ARG("--lr")) lr = (float)atof(argv[++i]);
        else if (ARG("--tau")) tau = (float)atof(argv[++i]);
        else if (ARG("--pw")) pw = (float)atof(argv[++i]);
        else if (ARG("--vw")) vw = (float)atof(argv[++i]);
        else if (ARG("--wd")) wd = (float)atof(argv[++i]);
        else if (ARG("--lambda")) lambda = (float)atof(argv[++i]);
        else if (ARG("--sample-plies")) sample_plies = atoi(argv[++i]);
        else if (ARG("--winbonus")) winbonus = (float)atof(argv[++i]);
        else if (ARG("--rounds")) rounds = atoi(argv[++i]);
        else if (ARG("--eval")) eval_pairs = atoi(argv[++i]);
        else if (ARG("--seed")) seed = strtoull(argv[++i], NULL, 10);
        else if (ARG("--gen-switch")) gen_switch = atoi(argv[++i]);
        else if (ARG("--gen-dets")) gen_dets = atoi(argv[++i]);
        else if (ARG("--gen-sims")) gen_sims = atoi(argv[++i]);
        else if (ARG("--gen-rw")) gen_rw = atoi(argv[++i]);
        else if (ARG("--gen-nw")) gen_nw = atoi(argv[++i]);
        else if (ARG("--dump")) dump_path = argv[++i];
        else if (ARG("--data")) data_path = argv[++i];
        else if (ARG("--bw")) bw = (float)atof(argv[++i]);
        else if (ARG("--aug")) aug = atoi(argv[++i]);
        else if (ARG("--h1")) h1 = atoi(argv[++i]);
        else if (ARG("--h2")) h2 = atoi(argv[++i]);
        else if (!strcmp(k, "--flat-lr")) keep_lr_flat = 1;
        else { fprintf(stderr, "unknown option %s\n", k); return 1; }
        #undef ARG
    }

    Net *net = (Net *)calloc(1, sizeof(Net));
    Adam *adam = (Adam *)calloc(1, sizeof(Adam));
    if (init_path) {
        if (net_load(net, init_path) != 0) { fprintf(stderr, "cannot load %s\n", init_path); return 1; }
        printf("loaded initial network from %s (%dx%d)\n", init_path, net->h1, net->h2);
    } else {
        if (net_alloc(net, h1, h2) != 0) { fprintf(stderr, "bad width %dx%d\n", h1, h2); return 1; }
        net_init(net, seed * 977 + 13);
    }
    if (net_alloc_like(&adam->m, net) != 0 || net_alloc_like(&adam->v, net) != 0) {
        fprintf(stderr, "adam allocation failed\n"); return 1;
    }

    /* Sample files: header {magic, sizeof(Sample), PI_K, 0, count(u64)} then
     * raw samples.  States and targets are architecture independent, which is
     * what lets an old network's play teach a differently sized one. */
    #define SMP_MAGIC 0x4C435344u
    Replay rp;
    if (data_path) {
        FILE *f = fopen(data_path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", data_path); return 1; }
        uint32_t h[4]; uint64_t count;
        if (fread(h, sizeof h, 1, f) != 1 || fread(&count, sizeof count, 1, f) != 1 ||
            h[0] != SMP_MAGIC || h[1] != sizeof(Sample) || h[2] != PI_K) {
            fprintf(stderr, "%s is not a compatible sample file\n", data_path); return 1;
        }
        replay_init(&rp, (size_t)count);
        if (fread(rp.buf, sizeof(Sample), (size_t)count, f) != (size_t)count) {
            fprintf(stderr, "short read from %s\n", data_path); return 1;
        }
        fclose(f);
        rp.n = (size_t)count;
        printf("loaded %zu samples from %s\n", rp.n, data_path);
    } else {
        replay_init(&rp, bufcap);
    }
    FILE *dumpf = NULL;
    uint64_t dumped = 0;
    if (dump_path) {
        dumpf = fopen(dump_path, "wb");
        if (!dumpf) { fprintf(stderr, "cannot open %s\n", dump_path); return 1; }
        uint32_t h[4] = { SMP_MAGIC, sizeof(Sample), PI_K, 0 };
        fwrite(h, sizeof h, 1, dumpf);
        fwrite(&dumped, sizeof dumped, 1, dumpf);
    }

    Agent ref;
    spec_parse(ref_spec, &ref);

    Net **grads = (Net **)calloc((size_t)nthread, sizeof(Net *));
    for (int i = 0; i < nthread; i++) {
        grads[i] = (Net *)calloc(1, sizeof(Net));
        if (net_alloc_like(grads[i], net) != 0) { fprintf(stderr, "grad allocation failed\n"); return 1; }
    }

    size_t sample_cap = (size_t)games * 200 * (size_t)rounds;
    Sample *genbuf = (Sample *)malloc(sizeof(Sample) * sample_cap);
    if (!genbuf) { fprintf(stderr, "generation buffer allocation failed\n"); return 1; }

    printf("network %d-%d-%d value+policy, %zu weights, replay %zu MB\n",
           FEAT_DIM, net->h1, net->h2, net->nfloat,
           bufcap * sizeof(Sample) / (1024 * 1024));
    fflush(stdout);

    for (int it = 1; it <= iters; it++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        if (data_path) goto training;   /* dataset mode: nothing to generate */

        Agent gen;
        if (it <= gen_switch && !init_path) {
            agent_default(&gen, AG_HEUR, NULL);
        } else if (!gen_spec || !strcmp(gen_spec, "selfmcts")) {
            agent_default(&gen, AG_MCTS, net);
            gen.dets = gen_dets; gen.sims = gen_sims;
            gen.root_width = gen_rw; gen.node_width = gen_nw;
        } else if (!strcmp(gen_spec, "heur")) {
            agent_default(&gen, AG_HEUR, NULL);
        } else if (!strcmp(gen_spec, "self")) {
            agent_default(&gen, AG_NET, net);
        } else if (!strcmp(gen_spec, "selfpolicy")) {
            agent_default(&gen, AG_POLICY, net);
        } else if (!strncmp(gen_spec, "selfrollout", 11)) {
            /* live-net rollout generator, same parameter tail as the rollout
             * spec: dets:cands:floor:gate:minc:plylo:plyhi:evalc.  evalc
             * matters for expert iteration: advisory candidates enter the
             * Q-softmax target, so the trainer can pull probability toward
             * moves the policy has written off while the generated games
             * still play the selection rule. */
            agent_default(&gen, AG_ROLLOUT, net);
            char tmp[128];
            snprintf(tmp, sizeof tmp, "%s", gen_spec);
            char *save = NULL;
            strtok_r(tmp, ":", &save);
            char *v;
            if ((v = strtok_r(NULL, ":", &save))) gen.dets = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.root_width = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.cand_floor = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.gate = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.min_cand = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.ply_lo = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.ply_hi = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) gen.eval_cand = atoi(v);
        } else {
            spec_parse(gen_spec, &gen);
            gen.net = net;
        }

        GenJob *jobs = (GenJob *)calloc((size_t)nthread, sizeof(GenJob));
        pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
        size_t per = sample_cap / (size_t)nthread;
        for (int i = 0; i < nthread; i++) {
            jobs[i].agent = gen;
            jobs[i].games = games;
            jobs[i].seed = seed * 1000003ULL + (uint64_t)it * 7919ULL;
            jobs[i].thread = i; jobs[i].nthread = nthread;
            jobs[i].out = genbuf + per * (size_t)i;
            jobs[i].cap = per;
            jobs[i].lambda = lambda;
            jobs[i].tau = tau;
            jobs[i].winbonus = winbonus;
            jobs[i].rounds = rounds;
            jobs[i].sample_plies = sample_plies;
            jobs[i].net = net;
        }
        for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, gen_worker, &jobs[i]);
        for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

        if (dumpf) {
            for (int i = 0; i < nthread; i++) {
                fwrite(jobs[i].out, sizeof(Sample), jobs[i].nout, dumpf);
                dumped += jobs[i].nout;
            }
        }

        size_t added = 0;
        double ga = 0, gp = 0;
        int gdone = 0;
        for (int i = 0; i < nthread; i++) {
            for (size_t k = 0; k < jobs[i].nout; k++) replay_push(&rp, &jobs[i].out[k]);
            added += jobs[i].nout;
            ga += jobs[i].sum_abs; gp += jobs[i].plies;
            gdone += jobs[i].done;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double gen_secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("iter %2d [%s]: %d games in %.1fs (%.1f g/s), %zu new samples, buffer %zu, "
               "|margin| %.1f, plies %.1f\n",
               it, gen.name ? gen.name : "?", gdone, gen_secs, gdone / gen_secs, added, rp.n,
               ga / gdone, gp / gdone);
        fflush(stdout);
        free(jobs); free(th);

    training:
        clock_gettime(CLOCK_MONOTONIC, &t0);
        Rng r; rng_seed(&r, seed + 555ULL * (uint64_t)it);
        size_t *idx = (size_t *)malloc(sizeof(size_t) * (size_t)batch);
        double vl = 0, pl = 0;
        long pnt = 0;
        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < batch; i++) idx[i] = rng_next(&r) % rp.n;
            TrainJob tj[64];
            pthread_t tt[64];
            int nt = nthread > 64 ? 64 : nthread;
            int chunk = (batch + nt - 1) / nt;
            for (int i = 0; i < nt; i++) {
                tj[i].net = net; tj[i].grad = grads[i]; tj[i].rp = &rp; tj[i].idx = idx;
                tj[i].from = i * chunk > batch ? batch : i * chunk;
                tj[i].to = (i + 1) * chunk > batch ? batch : (i + 1) * chunk;
                tj[i].pw = pw;
                tj[i].bw = bw;
                tj[i].vw = vw;
                tj[i].aug = aug;
                for (int q = 0; q < 4; q++) tj[i].rng.s[q] = rng_next(&r) | 1ULL;
            }
            for (int i = 0; i < nt; i++) pthread_create(&tt[i], NULL, train_worker, &tj[i]);
            for (int i = 0; i < nt; i++) pthread_join(tt[i], NULL);
            for (int i = 0; i < nt; i++) { vl += tj[i].vloss; pl += tj[i].ploss; pnt += tj[i].pn; }
            grad_accumulate(grads[0], grads, nt);
            float cur_lr = lr;
            if (!keep_lr_flat) {
                float frac = (float)s / (float)steps;
                cur_lr = lr * (0.5f * (1.0f + cosf(3.14159265f * frac)) * 0.9f + 0.1f);
            }
            net_adam_step(net, grads[0], adam, cur_lr, 1.0f / (float)batch, wd);
        }
        free(idx);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double tr_secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("            trained %d steps in %.1fs: value rmse %.1f pts, policy ce %.3f\n",
               steps, tr_secs, sqrt(vl / ((double)steps * batch)) * VAL_SCALE,
               pnt ? pl / pnt : 0.0);
        fflush(stdout);

        char path[512];
        snprintf(path, sizeof path, "%s", out_path);
        net_save(net, path);
        snprintf(path, sizeof path, "%s.it%d", out_path, it);
        net_save(net, path);

        if (eval_pairs > 0) {
            Agent cur;
            if (eval_spec && !strcmp(eval_spec, "mcts")) {
                agent_default(&cur, AG_MCTS, net);
                cur.dets = gen_dets; cur.sims = gen_sims;
            } else {
                agent_default(&cur, AG_POLICY, net);
            }
            MatchResult mr;
            match_run_r(&cur, &ref, eval_pairs, nthread, 777 + (uint64_t)it, rounds, &mr);
            printf("            %s vs %s: margin %+.2f +- %.2f, score %.1f%%, plies %.0f\n",
                   cur.name, ref_spec, mr.margin, mr.margin_se, 100 * mr.winrate, mr.plies);
            fflush(stdout);
        }
    }
    if (dumpf) {
        fseek(dumpf, sizeof(uint32_t) * 4, SEEK_SET);
        fwrite(&dumped, sizeof dumped, 1, dumpf);
        fclose(dumpf);
        printf("dumped %llu samples to %s\n", (unsigned long long)dumped, dump_path);
    }
    return 0;
}
