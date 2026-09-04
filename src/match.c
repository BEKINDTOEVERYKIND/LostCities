#include "match.h"
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    const Agent *a, *b;
    int pairs, thread, nthread, rounds;
    uint64_t seed;
    double sum, sumsq, wins, losses, draws, points_a, points_b, plies;
    double wsum, wsumsq;    /* per-pair win score (0, .25, .5, .75, 1) */
    int done;
    int *next;          /* shared pair counter: threads pull pairs dynamically */
} Job;

/* One full match: `rounds` deals with cumulative context, the first player
 * alternating by round.  decks[r] supplies the deal for round r so a paired
 * rematch sees identical cards.  Scores are totals over all rounds for the
 * seats passed as first/second. */
static void play_one(const Agent *first, const Agent *second, int rounds,
                     uint8_t decks[][NCARD], Rng *rng,
                     int *score0, int *score1, double *plies)
{
    const Agent *ag[2] = { first, second };
    int cum[2] = { 0, 0 };
    for (int r = 0; r < rounds; r++) {
        State st;
        lc_deal_from_deck(&st, decks[r]);
        st.round = (uint8_t)r;
        st.cum[0] = (int16_t)cum[0];
        st.cum[1] = (int16_t)cum[1];
        st.turn = (uint8_t)(r & 1);
        while (!st.over) {
            Move m = agent_move(ag[st.turn], &st, rng);
            lc_apply(&st, m);
        }
        cum[0] += lc_score(&st, 0);
        cum[1] += lc_score(&st, 1);
        *plies += st.nply;
    }
    *score0 = cum[0];
    *score1 = cum[1];
}

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    /* pairs are pulled from a shared counter (no thread idles while another
     * finishes a long pair), and every pair's play stream is seeded by the
     * pair index, so a run is reproducible whatever the scheduling */
    for (;;) {
        int g = __atomic_fetch_add(j->next, 1, __ATOMIC_RELAXED);
        if (g >= j->pairs) break;
        Rng rng; rng_seed(&rng, j->seed * 6364136223846793005ULL + 1442695040888963407ULL + 0x9E3779B97F4A7C15ULL * (uint64_t)(g + 1));
        Rng deal_rng; rng_seed(&deal_rng, j->seed + 0x5DEECE66DULL * (uint64_t)(g + 1));
        uint8_t decks[MATCH_ROUNDS][NCARD];
        for (int r = 0; r < j->rounds; r++) {
            for (int i = 0; i < NCARD; i++) decks[r][i] = (uint8_t)i;
            for (int i = NCARD - 1; i > 0; i--) {
                uint32_t k = rng_below(&deal_rng, (uint32_t)i + 1);
                uint8_t t = decks[r][i]; decks[r][i] = decks[r][k]; decks[r][k] = t;
            }
        }
        int s0, s1;
        double pair = 0.0, ws = 0.0;
        play_one(j->a, j->b, j->rounds, decks, &rng, &s0, &s1, &j->plies);
        j->points_a += s0; j->points_b += s1;
        pair += s0 - s1;
        if (s0 > s1) { j->wins++; ws += 1.0; } else if (s0 < s1) j->losses++; else { j->draws++; ws += 0.5; }
        play_one(j->b, j->a, j->rounds, decks, &rng, &s0, &s1, &j->plies);
        j->points_a += s1; j->points_b += s0;
        pair += s1 - s0;
        if (s1 > s0) { j->wins++; ws += 1.0; } else if (s1 < s0) j->losses++; else { j->draws++; ws += 0.5; }
        j->sum += pair;
        j->sumsq += pair * pair;
        ws *= 0.5;
        j->wsum += ws;
        j->wsumsq += ws * ws;
        j->done++;
    }
    return NULL;
}

void match_run_r(const Agent *a, const Agent *b, int pairs, int nthread,
                 uint64_t seed, int rounds, MatchResult *out)
{
    if (nthread < 1) nthread = 1;
    if (rounds < 1) rounds = 1;
    if (rounds > MATCH_ROUNDS) rounds = MATCH_ROUNDS;
    Job *jobs = (Job *)calloc((size_t)nthread, sizeof(Job));
    pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
    int next = 0;
    for (int i = 0; i < nthread; i++) {
        jobs[i].next = &next;
        jobs[i].a = a; jobs[i].b = b; jobs[i].pairs = pairs;
        jobs[i].thread = i; jobs[i].nthread = nthread; jobs[i].seed = seed;
        jobs[i].rounds = rounds;
    }
    for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, worker, &jobs[i]);
    for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

    double sum = 0, sumsq = 0, w = 0, l = 0, d = 0, pa = 0, pb = 0, pl = 0, ws = 0, wss = 0;
    int done = 0;
    for (int i = 0; i < nthread; i++) {
        sum += jobs[i].sum; sumsq += jobs[i].sumsq;
        ws += jobs[i].wsum; wss += jobs[i].wsumsq;
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
    {
        double m = ws / done, v = done > 1 ? (wss - done * m * m) / (done - 1) : 0.0;
        if (v < 0) v = 0;
        out->winrate_se_paired = sqrt(v / done);
    }
    out->points_a = pa / ngames;
    out->points_b = pb / ngames;
    out->plies = pl / ngames;
    out->wins = w; out->losses = l; out->draws = d;
}

void match_run(const Agent *a, const Agent *b, int pairs, int nthread,
               uint64_t seed, MatchResult *out)
{
    match_run_r(a, b, pairs, nthread, seed, 1, out);
}
