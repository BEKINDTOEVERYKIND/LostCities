#include "match.h"
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    const Agent *a, *b;
    int pairs, thread, nthread;
    uint64_t seed;
    double sum, sumsq, wins, losses, draws, points_a, points_b, plies;
    int done;
} Job;

static void play_one(const Agent *first, const Agent *second, const uint8_t deck[NCARD],
                     Rng *rng, int *score0, int *score1, double *plies)
{
    State st;
    lc_deal_from_deck(&st, deck);
    const Agent *ag[2] = { first, second };
    while (!st.over) {
        Move m = agent_move(ag[st.turn], &st, rng);
        lc_apply(&st, m);
    }
    *score0 = lc_score(&st, 0);
    *score1 = lc_score(&st, 1);
    *plies += st.nply;
}

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    Rng rng; rng_seed(&rng, j->seed * 6364136223846793005ULL + 1442695040888963407ULL + (uint64_t)j->thread);
    for (int g = j->thread; g < j->pairs; g += j->nthread) {
        Rng deal_rng; rng_seed(&deal_rng, j->seed + 0x5DEECE66DULL * (uint64_t)(g + 1));
        uint8_t deck[NCARD];
        for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
        for (int i = NCARD - 1; i > 0; i--) {
            uint32_t k = rng_below(&deal_rng, (uint32_t)i + 1);
            uint8_t t = deck[i]; deck[i] = deck[k]; deck[k] = t;
        }
        int s0, s1;
        double pair = 0.0;
        play_one(j->a, j->b, deck, &rng, &s0, &s1, &j->plies);
        j->points_a += s0; j->points_b += s1;
        pair += s0 - s1;
        if (s0 > s1) j->wins++; else if (s0 < s1) j->losses++; else j->draws++;
        play_one(j->b, j->a, deck, &rng, &s0, &s1, &j->plies);
        j->points_a += s1; j->points_b += s0;
        pair += s1 - s0;
        if (s1 > s0) j->wins++; else if (s1 < s0) j->losses++; else j->draws++;
        j->sum += pair;
        j->sumsq += pair * pair;
        j->done++;
    }
    return NULL;
}

void match_run(const Agent *a, const Agent *b, int pairs, int nthread,
               uint64_t seed, MatchResult *out)
{
    if (nthread < 1) nthread = 1;
    Job *jobs = (Job *)calloc((size_t)nthread, sizeof(Job));
    pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
    for (int i = 0; i < nthread; i++) {
        jobs[i].a = a; jobs[i].b = b; jobs[i].pairs = pairs;
        jobs[i].thread = i; jobs[i].nthread = nthread; jobs[i].seed = seed;
    }
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, worker, &jobs[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

    double sum = 0, sumsq = 0, w = 0, l = 0, d = 0, pa = 0, pb = 0, pl = 0;
    int done = 0;
    for (int i = 0; i < nthread; i++) {
        sum += jobs[i].sum; sumsq += jobs[i].sumsq;
        w += jobs[i].wins; l += jobs[i].losses; d += jobs[i].draws;
        pa += jobs[i].points_a; pb += jobs[i].points_b; pl += jobs[i].plies;
        done += jobs[i].done;
    }
    free(jobs); free(th);
    if (done == 0) { memset(out, 0, sizeof(*out)); return; }
    double ngames = 2.0 * done;
    double mean_pair = sum / done;
    double var = sumsq / done - mean_pair * mean_pair;
    if (var < 0) var = 0;
    out->pairs = done;
    out->games = (int)ngames;
    out->margin = mean_pair / 2.0;
    out->margin_se = sqrt(var / done) / 2.0;
    out->winrate = (w + 0.5 * d) / ngames;
    out->winrate_se = sqrt(out->winrate * (1.0 - out->winrate) / ngames);
    out->points_a = pa / ngames;
    out->points_b = pb / ngames;
    out->plies = pl / ngames;
    out->wins = w; out->losses = l; out->draws = d;
}
