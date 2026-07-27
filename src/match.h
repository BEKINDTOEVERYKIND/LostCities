/* match.h -- paired-deal match runner shared by the arena and the trainer. */
#ifndef MATCH_H
#define MATCH_H

#include "agent.h"

typedef struct {
    int pairs, games;
    double margin, margin_se;   /* per game, from agent a's point of view */
    double winrate, winrate_se;
    double points_a, points_b, plies;
    double wins, losses, draws;
} MatchResult;

void match_run(const Agent *a, const Agent *b, int pairs, int nthread,
               uint64_t seed, MatchResult *out);

#endif
