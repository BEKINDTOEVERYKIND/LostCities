/* belx.h -- the extended-input belief specialist.
 *
 * A standalone MLP for opponent-hand inference whose input is the engine
 * feature encoding plus the behavioral-history planes the snapshot
 * otherwise erases (State.disc_by / State.passed): who discarded each card
 * still in a pile, and how long each player has passed over the current
 * pile tops.  Trained by tools/belief.c (xtrain); consumed at play time by
 * belief-weighted determinization (rollouth with a .blx second path).
 */
#ifndef BELX_H
#define BELX_H

#include "lc.h"
#include "features.h"
#include "net.h"

#define BELX_XBIN   (2 * NCARD)                       /* disc_by[o], disc_by[p] */
#define BELX_XDENSE 10                                /* passed[o][s], passed[p][s] */
#define BELX_XDIM   (FEAT_DIM + BELX_XBIN + BELX_XDENSE)
/* the layout before the turn-arithmetic block (data/belx_final.blx):
 * belx_load reads such a file with the block's rows zero, belx_save
 * always writes BELX_XDIM */
#define BELX_XDIM_V5 (FEAT_DIM_V5 + BELX_XBIN + BELX_XDENSE)
#define BELX_MAGIC  0x42454C58u

typedef struct BelX {
    int h1, h2;
    float *blk;
    float *w1, *b1, *w2, *b2, *wb, *bb;   /* w1: [BELX_XDIM][h1] */
} BelX;

/* extra feature rows beyond the base encoding */
typedef struct {
    uint16_t idx[128];              /* binary rows in [FEAT_DIM, FEAT_DIM+BELX_XBIN) */
    int nidx;
    float dense[BELX_XDENSE];       /* rows FEAT_DIM+BELX_XBIN .. BELX_XDIM */
} BelXFeat;

size_t belx_nfloat(int h1, int h2);
void   belx_wire(BelX *x);
void   belx_alloc(BelX *x, int h1, int h2);
void   belx_save(const BelX *x, const char *path);
int    belx_load(BelX *x, const char *path);
void   belx_from_net(BelX *x, const Net *n);   /* warm start; identical function */

void   belx_feat(const State *st, int p, Features *base, BelXFeat *xf);
void   belx_trunk(const BelX *x, const Features *f, const BelXFeat *xf, NetAct *act);
void   belx_logits(const BelX *x, const NetAct *act, const uint8_t *cards, int nc, float *lg);

#endif
