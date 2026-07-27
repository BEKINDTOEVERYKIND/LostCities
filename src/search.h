/* search.h -- determinized Monte Carlo tree search.
 *
 * The hidden information (opponent hand, deck order) is resampled many times;
 * each sample turns the game into a perfect information problem that a small
 * PUCT tree search solves with network leaf values and heuristic priors.  Root
 * statistics are pooled over all samples.
 */
#ifndef SEARCH_H
#define SEARCH_H

#include "lc.h"
#include "net.h"

struct Agent;

typedef struct {
    int n;
    Move mv[MAX_MOVES];
    double visits[MAX_MOVES];
    double q[MAX_MOVES];   /* mean value in points, mover's view */
    float value;           /* pooled root value in points        */
} SearchStats;

Move search_move(const struct Agent *a, const State *st, Rng *rng,
                 float *out_value, SearchStats *stats);
/* Policy improvement by paired playouts from sampled worlds (rollout.c). */
Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats);

#endif
