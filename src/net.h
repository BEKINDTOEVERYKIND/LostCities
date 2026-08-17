/* net.h -- two-headed network with a sparse input layer.
 *
 * Input  : FEAT_DIM features (FEAT_BIN sparse binary + FEAT_DENSE scalars)
 * Trunk  : h1 -> h2, ReLU (widths are RUNTIME properties of the net, read
 *          from the weight file's header; the compiled maxima below only
 *          bound stack scratch buffers)
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
 *
 * Belief : one logit per card, trained to predict whether the opponent holds
 *          it (the true hand is known when self-play data is generated, so the
 *          labels are free).  This is where behavioural inference lives: the
 *          trunk sees what the opponent has committed to and thrown away, and
 *          the head learns what that implies about the cards they kept.  The
 *          determinized search samples opponent hands from this posterior
 *          instead of uniformly.
 *
 * Storage: one contiguous float block, sections in file order (w1, b1, w2,
 * b2, w3, b3, wplay, bplay, wdraw, bdraw, wbel, bbel), so the flattened
 * Adam/accumulate loops and the on-disk layout survive the move to runtime
 * sizing unchanged -- a 512x256 file saved by this code is byte-identical
 * to one saved by the fixed-size code it replaces.
 */
#ifndef NET_H
#define NET_H

#include "features.h"

#define NET_H1_MAX 2048          /* stack scratch bound, not a net property */
#define NET_H2_MAX 1024
#define NET_H1_DEF 512           /* the architecture every champion so far used */
#define NET_H2_DEF 256
#define NET_NPLAY (NCARD * 2)    /* card x (play|discard) */
#define NET_NDRAW (NSUIT + 1)    /* deck or one of five piles */
#define VAL_SCALE 50.0f

typedef struct {
    int h1, h2;
    size_t nfloat;   /* total floats in blk */
    float *blk;      /* single allocation; the pointers below index into it */
    float *w1;       /* [FEAT_DIM][h1], row stride h1 */
    float *b1;       /* [h1] */
    float *w2;       /* [h1][h2], row stride h2 */
    float *b2;       /* [h2] */
    float *w3;       /* [h2]  value head */
    float *b3;       /* [1] */
    float *wplay;    /* [NET_NPLAY][h2] */
    float *bplay;    /* [NET_NPLAY] */
    float *wdraw;    /* [NET_NDRAW][h2] */
    float *bdraw;    /* [NET_NDRAW] */
    float *wbel;     /* [NCARD][h2], appended last so files without it load */
    float *bbel;     /* [NCARD] */
} Net;

typedef struct {
    float a1[NET_H1_MAX];
    float a2[NET_H2_MAX];
} NetAct;

typedef struct {
    Net m, v;
    long t;
} Adam;

/* allocate/free the weight block; alloc zeroes it */
int   net_alloc(Net *n, int h1, int h2);
void  net_free(Net *n);
int   net_alloc_like(Net *dst, const Net *src);
void  net_copy(Net *dst, const Net *src);      /* dims must already match */

void  net_init(Net *n, uint64_t seed);         /* requires net_alloc first */
void  net_zero(Net *n);                        /* zero the weights, keep dims */

/* trunk only; fills act */
void  net_trunk(const Net *n, const Features *f, NetAct *act);
float net_value_act(const Net *n, const NetAct *act);
/* logits for the given packed moves */
void  net_policy_act(const Net *n, const NetAct *act, const uint16_t *mv, int nmv, float *logits);
/* belief logits (opponent holds card?) for the given card ids */
void  net_belief_act(const Net *n, const NetAct *act, const uint8_t *cards, int nc, float *logits);

/* convenience: trunk + value */
float net_value(const Net *n, const Features *f);

/* Accumulate gradients for one sample.  dvalue is dLoss/dvalue_output;
 * dlogit[i] is dLoss/dlogit for move mv[i]; dbel[i] likewise for belief card
 * bc[i].  Either head may be skipped by passing NULL. */
void  net_backward(const Net *n, const Features *f, const NetAct *act,
                   float dvalue, const uint16_t *mv, const float *dlogit, int nmv,
                   const uint8_t *bc, const float *dbel, int nb,
                   Net *g);
void  net_adam_step(Net *n, const Net *g, Adam *a, float lr, float scale, float wd);
int   net_save(const Net *n, const char *path);
/* loads into a fresh shell (allocates the block; does not free a prior one) */
int   net_load(Net *n, const char *path);

#endif
