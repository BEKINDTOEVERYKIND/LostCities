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
                           and reported (SearchStats.qw) either way. 
                           2 = smooth match objective in EVERY round (see
                           rollout.c win_value): each world's margin m is
                           scored as 100*Phi((cum_diff + m) / s_left), s_left^2 =
                           rounds_after * ROUND_SD^2 + FINAL_SD^2, so rounds
                           1-2 reduce to points except in blowouts and the
                           final round becomes a soft step; selection, the
                           sel_k gate and the override run unchanged on that
                           statistic (win_q 1's lexicographic final-round pick
                           only ever acted as a tiebreak among margin
                           qualifiers).  Reported q stays in points. */
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
    int sel_draw;       /* AG_ROLLOUT (spec field 22): same-action draw
                           variants clear the sel_k selection gate at HALF
                           sel_k, mirroring ov_draw's rationale one layer up:
                           the card and disposition are the policy's own top
                           choice, only the draw source differs, the paired SE
                           is structurally inflated by post-draw divergence,
                           and the policy's prior over draw sources is its
                           least trustworthy output (the reviewer's pile-draw
                           refusals at 2026-08-25 plies 27/29 sat inside the
                           eligible set and lost to the full-k gate at 96
                           worlds while 512 worlds ranked them first by +6).
                           0 = off (previous behavior). */
    int draw_filter;    /* AG_ROLLOUT (spec field 23): restricts the advisory
                           draw-variant expansion of top actions, a strength
                           scale.  1: add a pile-draw variant only when that
                           pile's top card is playable by the mover (the
                           reviewer's ply-18 note: evaluating every pile draw
                           after a popular play is pure waste, and every
                           historical useful-draw win -- takeG6, drawY, the Gx
                           grabs -- involved a playable top).  2: no expansion
                           at all; search evaluates only policy-ranked moves
                           (the reviewer's directive: the compute belongs to
                           the top policy plays; expansion variants were
                           advisory-only, reachable solely through the
                           override).  Do not combine 2 with ov_draw -- the
                           mode would starve the override of the variants it
                           exists to rescue.  Cuts per-decision playout cost;
                           the freed budget can be spent as more worlds
                           (higher dets at the same wall-clock).  0 = expand
                           all legal draw sources (previous behavior).  3: action-level
                           candidates -- moves are grouped by (card,
                           disposition), ranked by the group's summed prior,
                           and each group is searched with its most probable
                           draw source (the policy head factors P(move) =
                           P(action) P(draw), so a confident action would
                           otherwise fill the slots with its own alternative
                           draws instead of other plays).  Implies 2. */
    int sel_deep;       /* AG_ROLLOUT (spec field 24): when any eligible
                           candidate outscores the policy top on the first
                           world batch, run a second batch of dets worlds
                           and make every selection/override decision on the
                           pooled 2x statistics.  One-batch/one-SE sel_k
                           qualification against several alternatives passes
                           noise on a tail of seeds (the reviewer's ply-16
                           catch); pooling ADDS DATA instead of raising the
                           bar, so real small leads qualify more often while
                           flukes qualify less -- the trade a stiffer sel_k
                           (k=1.5, measured worse) and a fresh-batch veto
                           (probe-refuted: suppressed 12 reviewer-verified
                           good overrides to remove 6 noise cases) both got
                           wrong.  Costs one extra batch on contested plies
                           only (1 = this mode).
                           2 = RACING deepening: same trigger (an eligible
                           candidate leads the policy top after batch 1),
                           but the second batch is bought only for the
                           SURVIVORS -- candidate 0 plus every eligible
                           candidate whose batch-1 deficit against the
                           batch-1 leader is under 1.5 paired SEs
                           (rollout.c RACE_K).  Survivors are decided on
                           pooled 2x statistics exactly as in mode 1
                           (they share the same 2*dets worlds, so the
                           paired sel_k gate is unchanged); the others keep
                           their batch-1 statistics (SearchStats visits
                           shows each candidate's own world count), are
                           reported, and cannot be selected.  Mode 1
                           measured +3.6 paired points at ~1.7x wall-clock
                           (Gate M); racing keeps the pooled decision on
                           the candidates that can still win and skips
                           the playouts on those that cannot. */
    int sym_bel;        /* AG_ROLLOUT (spec field 27): average the belief
                           logits used for world sampling over this many
                           random suit/wager relabelings (the belief-head
                           analogue of sym_k; K trunk forwards per decision,
                           cached, so it also removes the per-world trunk
                           forward).  0 = raw logits.  >= 120 (LC_SYM_EXACT)
                           enumerates the 120 suit relabelings: exactly
                           suit-invariant, copies still sampled then pooled. */
    int omni;           /* AG_ROLLOUT (spec field 28, after sym_bel):
                           MEASUREMENT ONLY.  1 = every world carries the
                           opponent's TRUE hand (the deck order is still
                           sampled from the remaining unseen cards).  This
                           is cheating and can never be deployed; it exists
                           to bound what any hand-inference improvement
                           could deliver at the current search (see
                           data/probes/omni_bound_2026-09-05.txt).  0 = off. */
    const Net *net_p;   /* AG_ROLLOUT (spec field 29, after omni): a net used
                           ONLY inside the playouts (the greedy/sampled
                           continuations of both seats in every world).  The
                           root keeps a->net for priors, candidates and the
                           displayed value, so the evaluated set is unchanged
                           and only the estimator's leaves move.  Path, or
                           "0"/absent = the main net (bit-identical to before
                           the field existed). */
    int sym_play;       /* AG_ROLLOUT (spec field 30, after net_p): 1 = every
                           world is played out under its own random suit
                           relabeling (wager copies too), drawn from the
                           world's seed so the world stream itself is
                           unchanged.  The playout policy is the one
                           unsymmetrized component of the estimator; one
                           frame per world makes the estimator exactly
                           suit-invariant in expectation at zero forward
                           cost (a permute per world).  Every candidate
                           shares the world AND the frame, so the pairing is
                           exact; the round margin is frame-invariant.
                           0 = identity frame (bit-identical). */
    int bel_samp;       /* AG_ROLLOUT (spec field 26): belief world sampler.
                           0 = Gumbel-top-k on the belief logits (the
                           original path, a Plackett-Luce draw whose
                           inclusion probabilities are NOT the head's trained
                           marginals -- late-game worlds collapse onto
                           duplicate over-confident hands, measured); 1 =
                           fixed-size conditional-Bernoulli on the marginals
                           shifted to sum to the hand size; 2 = as 1 with the
                           weights calibrated so inclusion == marginals.  The
                           per-decision cache also removes the per-world trunk
                           forward.  See determinize_bm. */
    int sym_k;          /* AG_ROLLOUT (spec field 25): average the policy
                           prior and value over this many random suit/wager
                           relabelings before candidates are formed (see
                           symmetrize_priors in rollout.c).  0 = raw policy.
                           Removes the residual label noise measured at 25%
                           argmax flips on the c20 champion; costs K forward
                           passes per decision.  >= 120 (LC_SYM_EXACT)
                           enumerates the 120 suit relabelings instead of
                           sampling them: the prior and value are then exactly
                           suit-invariant (wager-copy relabelings stay sampled
                           from the state-seeded stream; copies are folded
                           exactly by sym_key). */
    const struct BelX *bx;  /* AG_ROLLOUT hybrid, extended-format flavour:
                           when set, world sampling uses this belx net's
                           inference (with the behavioral-history features)
                           instead of any Net belief head; takes precedence
                           over net_b.  Set by a rollouth spec whose second
                           path is a .blx file. */
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
struct BelX;
void determinize_bx(const State *st, int p, Rng *rng, const struct BelX *bx, State *out);
/* omniscient-hand world (Agent.omni): the opponent's real hand, the deck
 * a uniform shuffle of the remaining unseen cards.  Measurement only. */
void determinize_omni(const State *st, int p, Rng *rng, State *out);
/* belief world with the sampler selected by mode (Agent.bel_samp): 0 = the
 * Gumbel-top-k draws above (bit-identical), 1 = conditional-Bernoulli on
 * the shifted marginals, 2 = the same with calibrated weights.  Uses bx
 * when given, else net.  Per-decision cache keyed by the information set. */
void determinize_bm(const State *st, int p, Rng *rng, const Net *net,
                    const struct BelX *bx, int mode, int symK, State *out);
/* Gumbel-top-k world from belief logits averaged over K random suit/wager
 * relabelings (Agent.sym_bel), cached per decision.  K<=0 = plain path. */
uint64_t infoset_hash(const State *st, int p);
float agent_value(const struct Agent *a, const State *st, int q);
int  agent_belief_logits(const struct Agent *a, const State *st, int p, Rng *rng,
                         uint8_t *unseen, float *logit);
int  agent_policy_probs(const struct Agent *a, const State *st, Rng *rng,
                        Move *mv, float *prob, float *value);
void determinize_bsym(const State *st, int p, Rng *rng, const Net *net,
                      const struct BelX *bx, int K, State *out);

#endif
