#ifndef HEURISTIC_H
#define HEURISTIC_H

#include "lc.h"

/* Hand-crafted projection evaluation, in points, from p's point of view. */
float heur_eval(const State *st, int p);

/* Value of a move for the player to move.
 *
 * The two variants differ in what is known about the deck.  _det is for
 * perfect-information (determinized) states, where the drawn card is simply
 * taken from the deck.  _is is for real play, where a deck draw is an unknown
 * card and has to be averaged over the cards the mover has not seen -- getting
 * this wrong makes deck draws look worse than they are and the agent stalls,
 * recycling discard piles instead of ever finishing the game. */
float heur_move_value_det(const State *st, Move m);
float heur_move_value_is(const State *st, Move m, Rng *rng, int samples);

#endif
