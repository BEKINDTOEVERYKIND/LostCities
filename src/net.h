/* net.h -- two-headed network with a sparse input layer.
 *
 * Input  : FEAT_DIM features (FEAT_BIN sparse binary + FEAT_DENSE scalars)
 * Trunk  : NET_H1 -> NET_H2, ReLU
 * Value  : one scalar, the expected final score margin of the perspective
 *          player, in units of VAL_SCALE points.
 * Policy : a move is (card, play-or-discard, draw source), so instead of one
 *          logit for each of the 720 combinations the head keeps one logit per
 *          (card, disposition) and one per draw source and adds them:
 *              logit(m) = play[card*2+disc] + draw[src]
 *          The softmax runs over the legal moves only.  This shares statistics
 *          across combinations that a flat head would have to learn one by one,
 *          and it is cheaper: a turn touches at most 16 play components and 6
 *          draw components rather than ~50 full rows.
 *
 * The policy head exists because ranking moves by a value function alone is
 * hopeless here: candidate moves differ by one or two points, far below the
 * accuracy any regression on game outcomes can reach.  Predicting the choice
 * directly sidesteps that.
 */
#ifndef NET_H
#define NET_H

#include "features.h"

#define NET_H1 512
#define NET_H2 256
#define NET_NPLAY (NCARD * 2)   /* card x (play|discard) */
#define NET_NDRAW (NSUIT + 1)   /* deck or one of five piles */
#define VAL_SCALE 50.0f

typedef struct {
    float w1[FEAT_DIM][NET_H1];
    float b1[NET_H1];
    float w2[NET_H1][NET_H2];
    float b2[NET_H2];
    float w3[NET_H2];                  /* value head  */
    float b3;
    float wplay[NET_NPLAY][NET_H2];    /* policy: which card, and whether played */
    float bplay[NET_NPLAY];
    float wdraw[NET_NDRAW][NET_H2];    /* policy: where to draw from */
    float bdraw[NET_NDRAW];
} Net;

typedef struct {
    float a1[NET_H1];
    float a2[NET_H2];
} NetAct;

typedef struct {
    Net m, v;
    long t;
} Adam;

void  net_init(Net *n, uint64_t seed);
void  net_zero(Net *n);

/* trunk only; fills act */
void  net_trunk(const Net *n, const Features *f, NetAct *act);
float net_value_act(const Net *n, const NetAct *act);
/* logits for the given packed moves */
void  net_policy_act(const Net *n, const NetAct *act, const uint16_t *mv, int nmv, float *logits);

/* convenience: trunk + value */
float net_value(const Net *n, const Features *f);

/* Accumulate gradients for one sample.  dvalue is dLoss/dvalue_output;
 * dlogit[i] is dLoss/dlogit for move mv[i] (may be NULL). */
void  net_backward(const Net *n, const Features *f, const NetAct *act,
                   float dvalue, const uint16_t *mv, const float *dlogit, int nmv,
                   Net *g);
void  net_adam_step(Net *n, const Net *g, Adam *a, float lr, float scale, float wd);
int   net_save(const Net *n, const char *path);
int   net_load(Net *n, const char *path);

#endif
