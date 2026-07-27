/* features.h -- information-set encoding for the value network.
 *
 * A feature vector describes the game from the point of view of one player:
 * that player's own hand plus everything public.  Nothing about the
 * opponent's hidden cards leaks in.  The vector is split into a sparse
 * binary part (card planes, encoded as a list of active indices) and a dense
 * part (engineered per-suit and global scalars), which lets the first network
 * layer skip almost all of its multiplies.
 */
#ifndef FEATURES_H
#define FEATURES_H

#include "lc.h"

/* card planes: my hand / my expeditions / their expeditions / discarded /
 * pile tops / cards I know they hold / my cards they know about */
#define FEAT_PLANES 7
#define FEAT_BIN (FEAT_PLANES * NCARD) /* 420 */
#define SUIT_FEATS 24
#define GLOBAL_FEATS 16
#define FEAT_DENSE (NSUIT * SUIT_FEATS + GLOBAL_FEATS) /* 136 */
#define FEAT_DIM (FEAT_BIN + FEAT_DENSE)               /* 556 */
#define FEAT_MAX_ACTIVE 184

typedef struct {
    int nidx;
    uint16_t idx[FEAT_MAX_ACTIVE];
    float dense[FEAT_DENSE];
} Features;

/* Encode st from player p's point of view. */
void feat_extract(const State *st, int p, Features *f);

#endif
