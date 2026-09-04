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
/* turn-arithmetic block (2026-09-04, panel 2 rank 5), appended AFTER the
 * global block so every earlier index is unchanged and a pre-block weight
 * file loads as the identical function with these w1 rows zero (net_load).
 * Layout, offsets relative to FEAT_BIN + FEAT_DENSE_V5:
 *   0  my_turns      turns I still get = (deck_left+1)/2 if I am the mover
 *                    else deck_left/2, scaled /22
 *   1  opp_turns     the other count, /22
 *   2  last_draw_is_mine   1 if I make the round's last draw under deck
 *                    draws: the mover draws last iff deck_left is odd
 *   3..17            one-hot deck_left 0..14
 *   18 nplay_total   cards in my hand playable now (sum of the per-suit
 *                    play_cnt), /8
 *   19 clip(nplay_total - my_turns, -8, 8)/8   turn pressure
 *   20 stall_last_is_mine   who draws last if the MOVER takes from a pile
 *                    this turn (deck_left unchanged, so the parity flips
 *                    relative to a deck draw) = 1 - last_draw_is_mine
 *   21 + 3*s + {0,1,2}  per suit s (suit order, so lc_permute relabels
 *                    them with the suits): unseen number cards above my
 *                    top (/9), unseen number cards above the opponent's
 *                    top (/9), clip(play_cnt[s] - my_turns, -8, 8)/8
 * When the perspective player is the mover the first three reduce to
 * (deck_left+1)/2, deck_left/2 and deck_left & 1 exactly. */
#define TURN_FEATS (21 + 3 * NSUIT)                       /* 36 */
#define FEAT_DENSE_V5 (NSUIT * SUIT_FEATS + GLOBAL_FEATS) /* 136: pre-block */
#define FEAT_DIM_V5 (FEAT_BIN + FEAT_DENSE_V5)            /* 556: pre-block */
#define FEAT_DENSE (FEAT_DENSE_V5 + TURN_FEATS)           /* 172 */
#define FEAT_DIM (FEAT_BIN + FEAT_DENSE)                  /* 592 */
#define FEAT_MAX_ACTIVE 184

typedef struct {
    int nidx;
    uint16_t idx[FEAT_MAX_ACTIVE];
    float dense[FEAT_DENSE];
} Features;

/* Encode st from player p's point of view. */
void feat_extract(const State *st, int p, Features *f);

#endif
