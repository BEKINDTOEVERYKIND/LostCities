/* agent.h -- move selection policies.
 *
 * Every agent sees only its own information set: the deck order and the
 * opponent's hand are never read, and deck draws are handled by sampling from
 * the set of cards the agent has not seen.
 */
#ifndef AGENT_H
#define AGENT_H

#include "lc.h"
#include "net.h"

typedef enum {
    AG_RANDOM = 0,
    AG_HEUR,     /* one-ply greedy on the hand-crafted evaluation */
    AG_NET,      /* one-ply greedy on the value head             */
    AG_POLICY,   /* single forward pass, argmax of the policy head */
    AG_MCTS,     /* determinized MCTS, network priors and values  */
    AG_ROLLOUT   /* candidate moves played out in sampled worlds   */
} AgentKind;

typedef struct Agent {
    AgentKind kind;
    const Net *net;
    int draw_samples;   /* deck-draw samples per decision (AG_NET)         */
    float temp;         /* >0: sample instead of taking the best move      */
    float eps;          /* probability of a uniformly random legal move    */
    /* AG_MCTS */
    int dets;           /* determinizations                                */
    int sims;           /* simulations per determinization                 */
    int root_width;     /* root moves kept after prior pruning             */
    int node_width;     /* interior moves kept                             */
    float cpuct;
    float cand_floor;   /* AG_ROLLOUT: ignore candidates below this policy  */
    int min_cand;       /* AG_ROLLOUT: but always keep at least this many --
                           a sharp prior otherwise leaves the search a single
                           candidate, able to confirm the policy but never to
                           correct it (0/1 = floor applies unconditionally) */
    int ply_lo, ply_hi; /* AG_ROLLOUT: search only when
                           ply_lo <= nply (< ply_hi if ply_hi > 0); outside
                           the window the raw policy plays.  For measuring
                           WHERE in a round the search actually earns its
                           keep (0,0 = search everywhere) */
    int eval_cand;      /* AG_ROLLOUT: evaluate (and report in stats) at
                           least this many candidates, but selection stays
                           restricted to the floor-passing set -- analysis
                           gets Q values for written-off moves without the
                           measured strength cost of letting 96-world noise
                           overrule a near-certain policy (0 = off) */
    int win_q;          /* AG_ROLLOUT: in the final round, select by match
                           wins over the playouts (margin as tiebreak)
                           instead of by margin.  Principled -- the last
                           round's playouts decide the match exactly -- but
                           measured NO BETTER than margin selection (48.0%
                           +- 2.0% head-to-head, 300 pairs): decided finals
                           tie on win%, close finals make a 96-world win
                           fraction a noisy binomial, and the win-trained
                           policy already carries the clutch behaviour.
                           Default off; the win fraction is still computed
                           and reported (SearchStats.qw) either way. */
    int prune_dom;      /* AG_ROLLOUT: drop discards dominated by a dead-card
                           discard (lc_discard_dominated) from candidates and
                           playout argmax -- frees candidate slots and stops
                           playouts gifting live cards when a dead one is in
                           hand */
    float override_k;   /* AG_ROLLOUT: let an advisory (eval_cand) candidate
                           take the move when it beats the eligible best by
                           more than this many paired standard errors
                           (0 = advisory candidates never selected) */
    int solve_deck;     /* AG_ROLLOUT: when deck_left <= this, replace the
                           whole candidate machinery with exact play: solve
                           every legal move to round end inside each
                           belief-sampled world (alpha-beta, no net calls)
                           and take the argmax of the exact averages.
                           0 = off. */
    int solve_vote;     /* AG_ROLLOUT labeling mode: with solve_deck set,
                           solve each belief world ONCE from the root (a
                           single alpha-beta whose cutoffs skip refuted
                           moves) and vote across worlds for the PV move,
                           margin-sum tiebreak.  Exact per-move averages
                           cost n_moves x worlds full-window solves --
                           measured 12-55M nodes PER MOVE at deck 5, out of
                           reach of any labeling budget -- while one root
                           solve prices a whole world.  Within one world
                           margin-argmax IS wins-then-margin argmax (win is
                           a monotone threshold on margin), so the final
                           round needs no special case.  Worlds that blow
                           the budget don't vote; fewer than 3 completed
                           worlds falls through to the normal search. */
    int playout_sample; /* AG_ROLLOUT: sample the policy in playouts instead
                           of argmaxing it (common per-world seeds keep the
                           candidate comparison paired).  Argmax repeats
                           every knife-edge downstream decision across all
                           worlds, which can manufacture large fake Q gaps
                           with tiny paired errors; sampling trades a
                           little variance for unbiasedness. */
    float override_min; /* AG_ROLLOUT: ...AND by at least this many points.
                           The SE gate alone is world-count-dependent in the
                           wrong direction: more worlds shrink the noise but
                           not the playout BIAS, so at 512 worlds a 3-SE
                           gate fires on ~1-point bias artifacts (measured:
                           stall- and discard-flavoured overrides an expert
                           reviewer graded as blunders).  Points are the
                           bias's own units.  Default 4. */
    float gate;         /* AG_ROLLOUT: skip the search entirely when the
                           policy's top move already has >= this probability
                           (0 = always search) */
    int no_belief;      /* AG_ROLLOUT ablation: sample worlds uniformly      */
    long solve_budget;  /* AG_ROLLOUT: per-decision node budget for the
                           solve_deck block (spec field 19, in millions;
                           0 = the LC_SOLVE_BUDGET env / 4M default).  The
                           pre-transposition-table budget could not finish a
                           deck-5 decision; with the table a root-vote at
                           deck 5 fits in tens of millions. */
    float sel_k;        /* AG_ROLLOUT selection gate: a non-top-prior ELIGIBLE
                           candidate may replace the policy top only when its
                           paired lead exceeds sel_k standard errors.  The
                           advisory override always had this protection; the
                           eligible argmax had none, so a low-prior candidate
                           could take the move on pure world noise (observed:
                           a 2%-prior wager gift, 3.1 +- 0.5 points WORSE at
                           4000 worlds, played from a ~5% noise tail at 96).
                           0 = raw argmax (old behavior). */
    float prior_w0;     /* AG_ROLLOUT prior-aware selection (spec fields
                           20/21): candidates are compared by
                           EV + lambda(ply)*log(prior), lambda linearly
                           interpolated from prior_w0 at ply 0 to prior_w1 at
                           ply 44 (clamped past 44).  The EV edge a non-top
                           candidate needs to overrule the policy therefore
                           scales with the prior gap -- a 4% candidate against
                           a 95% top needs lambda*log(95/4) extra points where
                           a 45%-vs-55% split needs almost none -- and a 1%
                           candidate must beat the 4% one on the same
                           handicapped score before it can take the move.
                           The sel_k paired-SE gate still applies on top
                           (priors price model belief, SEs price sampling
                           noise; they guard different failure modes).
                           0/0 = plain EV argmax (previous behavior). */
    float prior_w1;     /* lambda at ply 44+, see prior_w0 */
    const Net *net_b;   /* AG_ROLLOUT hybrid: when set, THIS net's belief
                           head steers world sampling while `net` keeps the
                           policy/priors/playouts (spec kind "rollouth").
                           NULL = use `net` for both, the normal case.  Born
                           from the wide-trunk config split: the 1024x512
                           net's search results sat 5 points closer to the
                           champion than its raw policy, suggesting its
                           trunk's strength lives in inference, not moves. */
    int ov_draw;        /* AG_ROLLOUT: draw variants of the SAME action
                           qualify for the override at HALF override_k.
                           Rationale: the 42.8% forcing disaster was about
                           written-off ACTIONS entering selection; a draw
                           variant keeps the chosen action and reconsiders
                           only the draw source -- the policy's weakest
                           head -- and its paired SE is inherently larger
                           (different draws genuinely diverge the futures),
                           so the full-k gate demands ~7+ points and real
                           ~5-point endgame edges are unreachable (observed:
                           a trailing player burning the deck instead of a
                           free dead-pile draw with q +12.5 vs +7.2).
                           Points floor and sampled confirmation still
                           apply.  0 = off (measured default until A/B'd). */
    const char *name;
} Agent;

/* Cards standing in for the unknown top of the deck.
 *
 * The set of cards a player has not seen is the same after every candidate
 * move of a turn (the card that leaves the hand was never unseen), so one
 * sample can be shared by all of them.  Reusing it is a common-random-numbers
 * trick: it removes almost all of the sampling noise from the *comparison*
 * between moves, which is what the choice depends on. */
#define MAX_DRAW_SAMPLES 24
typedef struct {
    uint8_t card[MAX_DRAW_SAMPLES];
    int n;
} DrawSamples;

void  draw_samples_init(const State *st, int p, Rng *rng, int k, DrawSamples *ds);
float move_value_net(const Net *net, const State *st, Move m, const DrawSamples *ds);
float move_value_heur(const State *st, Move m, const DrawSamples *ds);

void agent_default(Agent *a, AgentKind k, const Net *net);

/* Policy head evaluated on st for the player to move.  Fills mv[] with the
 * legal moves and prob[] with the normalized policy; returns the count. */
int  policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

/* Evaluate every legal move from st for the player to move.  Returns the
 * number of moves and fills mv[] and val[] (values in points, mover's view). */
int  agent_move_values(const Agent *a, const State *st, Rng *rng, Move *mv, float *val);
Move agent_move(const Agent *a, const State *st, Rng *rng);

/* Sample an index from weights[0..n) (already non-negative, sum > 0). */
int  sample_index(const float *w, int n, Rng *rng);

/* Build a determinization of st consistent with p's information: cards the
 * opponent is known to hold are pinned, and the rest of their hand and the
 * deck order are resampled from the unseen cards.  With a network, the
 * opponent's unknown cards are drawn from the belief head's posterior (what
 * their play so far implies they kept) instead of uniformly; net == NULL
 * falls back to uniform. */
void determinize(const State *st, int p, Rng *rng, State *out);
void determinize_b(const State *st, int p, Rng *rng, const Net *net, State *out);

#endif
