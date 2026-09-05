/* match.h -- paired-deal match runner shared by the arena and the trainer. */
#ifndef MATCH_H
#define MATCH_H

#include "agent.h"
#include <stdio.h>

typedef struct {
    int pairs, games;
    double margin, margin_se;   /* per game, from agent a's point of view */
    double winrate, winrate_se;         /* binomial SE over games (the record's convention) */
    double winrate_se_paired;           /* SE from the per-pair win scores: pairs are the
                                           independent units, so this is the honest one */
    double points_a, points_b, plies;
    double wins, losses, draws;
} MatchResult;

/* rounds = 1 gives single-deal games; rounds = MATCH_ROUNDS gives the full
 * competitive format, cumulative totals, alternating first player, margins
 * and winrate reported per match. */
/* optional per-pair log (arena -P): one line per pair, see match.c worker */
void match_set_pairlog(FILE *f);

void match_run_r(const Agent *a, const Agent *b, int pairs, int nthread,
                 uint64_t seed, int rounds, MatchResult *out);
void match_run(const Agent *a, const Agent *b, int pairs, int nthread,
               uint64_t seed, MatchResult *out);

#endif
