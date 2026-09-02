#include "agent.h"
#include "belx.h"
#include "heuristic.h"
#include "search.h"
#include <math.h>

void agent_default(Agent *a, AgentKind k, const Net *net)
{
    memset(a, 0, sizeof(*a));
    a->kind = k;
    a->net = net;
    a->draw_samples = 6;
    a->temp = 0.0f;
    a->eps = 0.0f;
    a->dets = 16;
    a->sims = 160;
    a->root_width = 14;
    a->node_width = 8;
    a->cpuct = 1.4f;
    a->cand_floor = 0.02f;
    a->min_cand = 1;
    a->ply_lo = 0;
    a->ply_hi = 0;
    a->eval_cand = 0;
    a->win_q = 0;
    a->prune_dom = 1;   /* measured strength-neutral (50.5% +- 2.0%, 300
                           pairs) and provably near-dominant; cheaper
                           searches, no free gifts.  Spec field 11 = 0
                           reverts. */
    a->override_k = 0.0f;
    a->override_min = 4.0f;
    a->playout_sample = 0;
    a->solve_deck = 0;
    switch (k) {
    case AG_RANDOM: a->name = "random"; break;
    case AG_HEUR:   a->name = "heuristic"; break;
    case AG_NET:    a->name = "net1"; break;
    case AG_POLICY: a->name = "policy"; break;
    case AG_MCTS:   a->name = "mcts"; break;
    case AG_ROLLOUT: a->name = "rollout"; a->dets = 128; a->root_width = 4; break;
    }
}

void determinize(const State *st, int p, Rng *rng, State *out)
{
    *out = *st;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);   /* already excludes cards known to be held */
    for (int i = n - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = unseen[i]; unseen[i] = unseen[j]; unseen[j] = t;
    }
    const int o = p ^ 1;
    /* the opponent certainly holds every card they took face up */
    out->hand[o] = st->known[o];
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    int k = 0;
    while (need-- > 0) out->hand[o] |= 1ULL << unseen[k++];
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    while (k < n) out->deck[d++] = unseen[k++];
    out->deck_left = (uint8_t)d;
}

/* Sample the opponent's unknown cards from the belief posterior using
 * Gumbel-top-k on the logits: an exact draw from the Plackett-Luce
 * distribution the logits induce, so likelier hands appear in more worlds. */
void determinize_b(const State *st, int p, Rng *rng, const Net *net, State *out)
{
    if (!net) { determinize(st, p, rng, out); return; }
    *out = *st;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    const int o = p ^ 1;
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    if (need <= 0 || n == 0) { determinize(st, p, rng, out); return; }

    Features f;
    feat_extract(st, p, &f);
    NetAct act;
    net_trunk(net, &f, &act);
    float logit[NCARD];
    net_belief_act(net, &act, unseen, n, logit);

    /* keys = logit + Gumbel noise; the top `need` keys form the hand */
    float key[NCARD];
    int order[NCARD];
    for (int i = 0; i < n; i++) {
        /* clamp u into (0,1) strictly: at the RNG's 24-bit max, u+eps
         * rounds to exactly 1.0f and logf(-logf(1)) = logf(-0) = -inf,
         * making key = +inf -- which silently FORCES this card into the
         * sampled hand regardless of its belief logit */
        float u = rng_float(rng);
        if (u < 1e-7f) u = 1e-7f;
        if (u > 0.999999f) u = 0.999999f;
        float l = logit[i];
        if (l > 15.0f) l = 15.0f;
        if (l < -15.0f) l = -15.0f;
        key[i] = l - logf(-logf(u));
        order[i] = i;
    }
    for (int i = 0; i < need; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (key[order[j]] > key[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    out->hand[o] = st->known[o];
    for (int i = 0; i < need; i++) out->hand[o] |= 1ULL << unseen[order[i]];
    /* remaining cards form the deck in random order */
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    for (int i = need; i < n; i++) out->deck[d++] = unseen[order[i]];
    for (int i = d - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = out->deck[i]; out->deck[i] = out->deck[j]; out->deck[j] = t;
    }
    out->deck_left = (uint8_t)d;
}


/* determinize_b with the extended-format specialist supplying the logits:
 * identical Gumbel-top-k sampling, only the inference model differs */

/* ---- calibrated fixed-size belief sampling (bel_samp, spec field 26) ----
 *
 * The belief head is trained as per-card Bernoulli marginals: sigma(l_c) is
 * "the probability the opponent holds c".  Gumbel-top-k on those logits is
 * a Plackett-Luce draw of `need` cards, whose INCLUSION probabilities are
 * not the marginals -- they are odds-shaped and, when need is a large share
 * of the unseen pool, pile mass onto the head's favourites (measured on the
 * c20 champion: per-card |sampled inclusion - sigma| 0.011 early / 0.028
 * mid / 0.054 late; ~25% duplicate hands per 96 late draws; sampled
 * opponents holding ~9% more playable-now cards than real ones; late-phase
 * sampled skill ~0 against the head's ~5%).  The fix is a fixed-size
 * conditional-Bernoulli draw: shift the logits so the marginals sum to
 * `need`, take w = pi/(1-pi), and draw sequentially with suffix elementary
 * symmetric polynomials so every world has exactly `need` cards and the
 * inclusion probabilities are (mode 1) close to, or (mode 2, weights
 * iteratively calibrated) equal to, the shifted marginals.  The shift, the
 * weights and the ESP table depend only on the mover's information set, so
 * they are computed once per decision and cached across the world batch --
 * which also removes the trunk forward the old path repeated per world. */

#define BS_MAXN 64
#define BS_MAXK 10

typedef struct {
    uint64_t key;
    int n, need, mode;
    uint8_t unseen[NCARD];
    double w[BS_MAXN];
    double S[BS_MAXN][BS_MAXK];   /* suffix ESPs of w */
    int valid;
} BelSampCache;

static _Thread_local BelSampCache bs_cache;

static uint64_t bs_key(const Features *f, int p, int need, int mode)
{
    uint64_t h = 1469598103934665603ULL;
#define MIX(b) (h = (h ^ (uint64_t)(b)) * 1099511628211ULL)
    MIX(p); MIX(need); MIX(mode); MIX(f->nidx);
    for (int i = 0; i < f->nidx; i++) MIX(f->idx[i]);
    const unsigned char *d = (const unsigned char *)f->dense;
    for (size_t i = 0; i < sizeof(f->dense); i++) MIX(d[i]);
#undef MIX
    return h;
}

/* hash of player p's information set at st: everything the agent may
 * legitimately condition on.  Symmetrization draws its random relabelings
 * from an Rng seeded by this, so the averaged prior/belief is a fixed
 * function of the state -- a display and the decision it explains compute
 * literally the same numbers, and the game's own Rng stream is untouched. */
uint64_t infoset_hash(const State *st, int p)
{
    Features f;
    feat_extract(st, p, &f);
    return bs_key(&f, p, 0, 7);
}

static void bs_suffix_esp(const double *w, int n, int k, double S[BS_MAXN][BS_MAXK])
{
    for (int j = 0; j <= k; j++) S[n][j] = (j == 0) ? 1.0 : 0.0;
    for (int m = n - 1; m >= 0; m--) {
        S[m][0] = 1.0;
        for (int j = 1; j <= k; j++) S[m][j] = S[m + 1][j] + w[m] * S[m + 1][j - 1];
    }
}

/* inclusion probabilities of the sequential draw with weights w, size k */
static void bs_incl(const double *w, int n, int k, double *pi)
{
    double P[BS_MAXN][BS_MAXK], S[BS_MAXN][BS_MAXK];
    for (int j = 0; j <= k; j++) P[0][j] = (j == 0) ? 1.0 : 0.0;
    for (int m = 1; m <= n; m++) {
        P[m][0] = 1.0;
        for (int j = 1; j <= k; j++) P[m][j] = P[m - 1][j] + w[m - 1] * P[m - 1][j - 1];
    }
    bs_suffix_esp(w, n, k, S);
    double ek = S[0][k];
    for (int c = 0; c < n; c++) {
        double f = 0.0;
        for (int i = 0; i <= k - 1; i++) f += P[c][i] * S[c + 1][k - 1 - i];
        pi[c] = ek > 0 ? w[c] * f / ek : 0.0;
    }
}

/* build the cache: shifted marginals -> weights (optionally calibrated so
 * the draw's inclusion probabilities match them) -> suffix ESPs.
 * Returns 0 when the draw is degenerate (fall back to the Gumbel path). */
static int bs_build(BelSampCache *c, const float *logit, const uint8_t *unseen,
                    int n, int need, int mode)
{
    if (n > BS_MAXN - 1 || need >= n || need < 1 || need > BS_MAXK - 2) return 0;
    double pi[BS_MAXN];
    /* logit shift so the marginals sum to need (bisection) */
    double lo = -20.0, hi = 20.0;
    for (int it = 0; it < 60; it++) {
        double d = 0.5 * (lo + hi), sum = 0.0;
        for (int i = 0; i < n; i++) {
            float l = logit[i];
            if (l > 15.0f) l = 15.0f;
            if (l < -15.0f) l = -15.0f;
            sum += 1.0 / (1.0 + exp(-((double)l + d)));
        }
        if (sum > need) hi = d; else lo = d;
    }
    double d = 0.5 * (lo + hi);
    for (int i = 0; i < n; i++) {
        float l = logit[i];
        if (l > 15.0f) l = 15.0f;
        if (l < -15.0f) l = -15.0f;
        pi[i] = 1.0 / (1.0 + exp(-((double)l + d)));
        if (pi[i] < 1e-3) pi[i] = 1e-3;
        if (pi[i] > 1.0 - 1e-3) pi[i] = 1.0 - 1e-3;
        c->w[i] = pi[i] / (1.0 - pi[i]);
    }
    if (mode >= 2) {
        double ph[BS_MAXN];
        for (int it = 0; it < 40; it++) {
            bs_incl(c->w, n, need, ph);
            double md = 0.0, g = 0.0;
            for (int i = 0; i < n; i++) {
                double dd = fabs(ph[i] - pi[i]);
                if (dd > md) md = dd;
                if (ph[i] > 1e-12) c->w[i] *= pi[i] / ph[i];
                g += log(c->w[i]);
            }
            g = exp(-g / n);
            for (int i = 0; i < n; i++) c->w[i] *= g;
            if (md < 1e-6) break;
        }
    }
    bs_suffix_esp(c->w, n, need, c->S);
    if (!(c->S[0][need] > 0.0)) return 0;
    memcpy(c->unseen, unseen, (size_t)n);
    c->n = n; c->need = need; c->mode = mode; c->valid = 1;
    return 1;
}

/* one world from the cache: exactly `need` cards for the opponent, the
 * rest of the unseen pool as a uniformly shuffled deck */
static void bs_draw(const BelSampCache *c, const State *st, int o, Rng *rng, State *out)
{
    *out = *st;
    out->hand[o] = st->known[o];
    uint8_t rest[NCARD];
    int nr = 0, r = c->need;
    for (int m = 0; m < c->n; m++) {
        int take = 0;
        if (r > 0) {
            double den = c->S[m][r];
            double pin = den > 0.0 ? c->w[m] * c->S[m + 1][r - 1] / den : 0.0;
            if (rng_float(rng) < pin) take = 1;
        }
        if (take) { out->hand[o] |= 1ULL << c->unseen[m]; r--; }
        else rest[nr++] = c->unseen[m];
    }
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    for (int i = 0; i < nr; i++) out->deck[i] = rest[i];
    for (int i = nr - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = out->deck[i]; out->deck[i] = out->deck[j]; out->deck[j] = t;
    }
    out->deck_left = (uint8_t)nr;
}


/* ---- symmetrized belief logits (sym_bel, spec field 27) ----
 * The belief head, like the policy, is only approximately invariant to
 * relabeling suits and wager copies.  Average its logits over K random
 * relabelings: permute the state, score the same unseen cards under their
 * permuted ids, and accumulate per original card.  Wager copies of a suit
 * are exchangeable, so averaging over copy permutations also equalizes
 * their marginals.  Cached per decision on the information set, which
 * also removes the per-world trunk forward of the plain Gumbel path. */
typedef struct {
    uint64_t key;
    int n, valid;
    uint8_t unseen[NCARD];
    float logit[NCARD];
} BelSymCache;
static _Thread_local BelSymCache bsym_cache;

static void belief_logits_raw(const Net *net, const struct BelX *bx, const State *st, int p,
                              const uint8_t *cards, int n, float *logit)
{
    Features f;
    NetAct act;
    if (bx) {
        BelXFeat xf;
        belx_feat(st, p, &f, &xf);
        belx_trunk(bx, &f, &xf, &act);
        belx_logits(bx, &act, cards, n, logit);
    } else {
        feat_extract(st, p, &f);
        net_trunk(net, &f, &act);
        net_belief_act(net, &act, cards, n, logit);
    }
}

static const float *belief_logits_sym(const Net *net, const struct BelX *bx, const State *st,
                                      int p, int K, Rng *rng, const uint8_t *unseen, int n)
{
    Features f;
    feat_extract(st, p, &f);
    uint64_t base = bs_key(&f, p, n, 1000 + K);
    /* the cache key tells belief sources apart; the relabeling seed must
     * NOT (pointer values differ run to run and would make the averaged
     * beliefs irreproducible across processes) */
    uint64_t key = base ^ (bx ? (uint64_t)(uintptr_t)bx * 0x9E3779B97F4A7C15ULL
                              : (uint64_t)(uintptr_t)net * 0xC2B2AE3D27D4EB4FULL);
    BelSymCache *c = &bsym_cache;
    if (c->valid && c->key == key && c->n == n) return c->logit;
    /* relabelings come from a state-seeded stream, not the caller's Rng:
     * deterministic per information set, and the match Rng is untouched */
    Rng lrng;
    rng_seed(&lrng, base ^ 0x5851F42D4C957F2DULL);
    rng = &lrng;
    double acc[NCARD];
    float tmp[NCARD];
    belief_logits_raw(net, bx, st, p, unseen, n, tmp);
    for (int i = 0; i < n; i++) acc[i] = tmp[i];
    for (int k = 0; k < K; k++) {
        int sp[NSUIT], wp[NSUIT][WAGERS_PER_SUIT];
        for (int i = 0; i < NSUIT; i++) sp[i] = i;
        for (int i = NSUIT - 1; i > 0; i--) { int j = (int)rng_below(rng, (uint32_t)i + 1); int t = sp[i]; sp[i] = sp[j]; sp[j] = t; }
        for (int su = 0; su < NSUIT; su++) {
            for (int i = 0; i < WAGERS_PER_SUIT; i++) wp[su][i] = i;
            for (int i = WAGERS_PER_SUIT - 1; i > 0; i--) { int j = (int)rng_below(rng, (uint32_t)i + 1); int t = wp[su][i]; wp[su][i] = wp[su][j]; wp[su][j] = t; }
        }
        uint8_t map[NCARD], pc[NCARD];
        lc_perm_map(sp, wp, map);
        State ps = *st;
        lc_permute(&ps, map);
        for (int i = 0; i < n; i++) pc[i] = map[unseen[i]];
        belief_logits_raw(net, bx, &ps, p, pc, n, tmp);
        for (int i = 0; i < n; i++) acc[i] += tmp[i];
    }
    for (int i = 0; i < n; i++) c->logit[i] = (float)(acc[i] / (K + 1));
    /* exact copy symmetry: the three wager copies of a suit are the same
     * card, so a belief that one copy is held is a belief about any of
     * them -- K random relabelings only approximate that (copies landed
     * 0.01-0.02 apart in probability); pooling the averaged logits over
     * the unseen copies of each suit makes it exact at no cost */
    for (int su = 0; su < NSUIT; su++) {
        double sum = 0.0; int cnt = 0;
        for (int i = 0; i < n; i++)
            if (CARD_IS_WAGER(unseen[i]) && CARD_SUIT(unseen[i]) == su) { sum += c->logit[i]; cnt++; }
        if (cnt > 1)
            for (int i = 0; i < n; i++)
                if (CARD_IS_WAGER(unseen[i]) && CARD_SUIT(unseen[i]) == su) c->logit[i] = (float)(sum / cnt);
    }
    memcpy(c->unseen, unseen, (size_t)n);
    c->n = n; c->key = key; c->valid = 1;
    return c->logit;
}

/* Gumbel-top-k world from a given logit vector (the tail of determinize_b) */
static void gumbel_fill(const State *st, int o, int need, const uint8_t *unseen, int n,
                        const float *logit, Rng *rng, State *out)
{
    *out = *st;
    float key[NCARD];
    int order[NCARD];
    for (int i = 0; i < n; i++) {
        float u = rng_float(rng);
        if (u < 1e-7f) u = 1e-7f;
        if (u > 0.999999f) u = 0.999999f;
        float l = logit[i];
        if (l > 15.0f) l = 15.0f;
        if (l < -15.0f) l = -15.0f;
        key[i] = l - logf(-logf(u));
        order[i] = i;
    }
    for (int i = 0; i < need; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (key[order[j]] > key[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    out->hand[o] = st->known[o];
    for (int i = 0; i < need; i++) out->hand[o] |= 1ULL << unseen[order[i]];
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    for (int i = need; i < n; i++) out->deck[d++] = unseen[order[i]];
    for (int i = d - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = out->deck[i]; out->deck[i] = out->deck[j]; out->deck[j] = t;
    }
    out->deck_left = (uint8_t)d;
}

/* The belief logits the agent's world sampler uses at this state, for
 * every card the mover cannot place (lc_unseen order): the agent's belief
 * source (belx / belief net / main net) and, when sym_bel is set, the
 * relabeling-averaged logits -- so a display built on this shows exactly
 * what the deciding agent believes, wager copies included (identical
 * copies must carry identical probability; the raw head does not). */
int agent_belief_logits(const struct Agent *a, const State *st, int p, Rng *rng,
                        uint8_t *unseen, float *logit)
{
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    if (n == 0) return 0;
    const Net *bnet = a->net_b ? a->net_b : a->net;
    if (a->no_belief || (!bnet && !a->bx)) {
        /* uniform sampler: every unseen card is held with probability
         * need/n -- report that marginal, not p=0.5 */
        int need = (int)st->hand_n[p ^ 1] - __builtin_popcountll(st->known[p ^ 1]);
        float q = n > 0 ? (float)need / (float)n : 0.5f;
        if (q < 1e-4f) q = 1e-4f;
        if (q > 1.0f - 1e-4f) q = 1.0f - 1e-4f;
        for (int i = 0; i < n; i++) logit[i] = logf(q / (1.0f - q));
        return n;
    }
    /* bel_samp modes (field 26) additionally shift these logits so the
     * marginals sum to the hand size before sampling; the display shows
     * the unshifted head, which is the deciding quantity for the adopted
     * spec (bel_samp=0) */
    if (a->sym_bel > 0) {
        const float *lg = belief_logits_sym(bnet, a->bx, st, p, a->sym_bel, rng, unseen, n);
        for (int i = 0; i < n; i++) logit[i] = lg[i];
    } else {
        belief_logits_raw(bnet, a->bx, st, p, unseen, n, logit);
    }
    return n;
}

void determinize_bsym(const State *st, int p, Rng *rng, const Net *net,
                      const struct BelX *bx, int K, State *out)
{
    if (K <= 0 || (!net && !bx)) {
        if (bx) determinize_bx(st, p, rng, bx, out); else determinize_b(st, p, rng, net, out);
        return;
    }
    const int o = p ^ 1;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    if (need <= 0 || n == 0) { determinize(st, p, rng, out); return; }
    const float *lg = belief_logits_sym(net, bx, st, p, K, rng, unseen, n);
    gumbel_fill(st, o, need, unseen, n, lg, rng, out);
}

void determinize_bm(const State *st, int p, Rng *rng, const Net *net,
                    const struct BelX *bx, int mode, State *out)
{
    if (mode <= 0 || (!net && !bx)) {
        if (bx) determinize_bx(st, p, rng, bx, out);
        else determinize_b(st, p, rng, net, out);
        return;
    }
    const int o = p ^ 1;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    if (need <= 0 || n == 0) { determinize(st, p, rng, out); return; }

    Features f;
    BelXFeat xf;
    if (bx) belx_feat(st, p, &f, &xf); else feat_extract(st, p, &f);
    uint64_t key = bs_key(&f, p, need, mode) ^ (bx ? 0x9E3779B97F4A7C15ULL : 0);
    BelSampCache *c = &bs_cache;
    if (!(c->valid && c->key == key)) {
        float logit[NCARD];
        NetAct act;
        if (bx) { belx_trunk(bx, &f, &xf, &act); belx_logits(bx, &act, unseen, n, logit); }
        else    { net_trunk(net, &f, &act);     net_belief_act(net, &act, unseen, n, logit); }
        c->valid = 0;
        if (!bs_build(c, logit, unseen, n, need, mode)) {
            /* degenerate pool (need >= n etc.): the Gumbel path handles it */
            if (bx) determinize_bx(st, p, rng, bx, out);
            else determinize_b(st, p, rng, net, out);
            return;
        }
        c->key = key;
    }
    bs_draw(c, st, o, rng, out);
}

void determinize_bx(const State *st, int p, Rng *rng, const struct BelX *bx, State *out)
{
    if (!bx) { determinize(st, p, rng, out); return; }
    *out = *st;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    const int o = p ^ 1;
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    if (need <= 0 || n == 0) { determinize(st, p, rng, out); return; }

    Features f;
    BelXFeat xf;
    NetAct act;
    belx_feat(st, p, &f, &xf);
    belx_trunk(bx, &f, &xf, &act);
    float logit[NCARD];
    belx_logits(bx, &act, unseen, n, logit);

    float key[NCARD];
    int order[NCARD];
    for (int i = 0; i < n; i++) {
        float u = rng_float(rng);
        if (u < 1e-7f) u = 1e-7f;
        if (u > 0.999999f) u = 0.999999f;
        float l = logit[i];
        if (l > 15.0f) l = 15.0f;
        if (l < -15.0f) l = -15.0f;
        key[i] = l - logf(-logf(u));
        order[i] = i;
    }
    for (int i = 0; i < need; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (key[order[j]] > key[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    out->hand[o] = st->known[o];
    for (int i = 0; i < need; i++) out->hand[o] |= 1ULL << unseen[order[i]];
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    for (int i = need; i < n; i++) out->deck[d++] = unseen[order[i]];
    for (int i = d - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = out->deck[i]; out->deck[i] = out->deck[j]; out->deck[j] = t;
    }
    out->deck_left = (uint8_t)d;
}

void draw_samples_init(const State *st, int p, Rng *rng, int k, DrawSamples *ds)
{
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    if (k > MAX_DRAW_SAMPLES) k = MAX_DRAW_SAMPLES;
    if (k < 1) k = 1;
    if (k >= n) {
        for (int i = 0; i < n; i++) ds->card[i] = unseen[i];
        ds->n = n;
        return;
    }
    /* sample k distinct cards: partial Fisher-Yates */
    for (int i = 0; i < k; i++) {
        uint32_t j = i + rng_below(rng, (uint32_t)(n - i));
        uint8_t t = unseen[i]; unseen[i] = unseen[j]; unseen[j] = t;
        ds->card[i] = unseen[i];
    }
    ds->n = k;
}

float move_value_net(const Net *net, const State *st, Move m, const DrawSamples *ds)
{
    const int p = st->turn;
    State base = *st;
    lc_apply_play(&base, m);
    Features f;

    if (m.draw > 0) {
        State s2 = base;
        lc_apply_draw(&s2, m, -1);   /* the pile top is public */
        if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        feat_extract(&s2, p, &f);
        return net_value(net, &f) * VAL_SCALE;
    }
    if (ds->n == 0) {
        State s2 = base;
        lc_apply_draw(&s2, m, -1);
        if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        feat_extract(&s2, p, &f);
        return net_value(net, &f) * VAL_SCALE;
    }
    float sum = 0.0f;
    for (int i = 0; i < ds->n; i++) {
        State s2 = base;
        lc_apply_draw(&s2, m, ds->card[i]);
        if (s2.over) {
            sum += (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        } else {
            feat_extract(&s2, p, &f);
            sum += net_value(net, &f) * VAL_SCALE;
        }
    }
    return sum / (float)ds->n;
}

float move_value_heur(const State *st, Move m, const DrawSamples *ds)
{
    const int p = st->turn;
    State base = *st;
    lc_apply_play(&base, m);

    if (m.draw > 0 || ds->n == 0) {
        State s2 = base;
        lc_apply_draw(&s2, m, -1);
        if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        return heur_eval(&s2, p);
    }
    float sum = 0.0f;
    for (int i = 0; i < ds->n; i++) {
        State s2 = base;
        lc_apply_draw(&s2, m, ds->card[i]);
        sum += s2.over ? (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1)) : heur_eval(&s2, p);
    }
    return sum / (float)ds->n;
}

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value)
{
    int n = lc_moves(st, mv);
    if (n == 0) return 0;
    uint16_t pk[MAX_MOVES];
    for (int i = 0; i < n; i++) pk[i] = MOVE_PACK(mv[i]);
    Features f;
    feat_extract(st, st->turn, &f);
    NetAct act;
    net_trunk(net, &f, &act);
    if (value) *value = net_value_act(net, &act) * VAL_SCALE;
    float lg[MAX_MOVES];
    net_policy_act(net, &act, pk, n, lg);
    float mx = lg[0];
    for (int i = 1; i < n; i++) if (lg[i] > mx) mx = lg[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { prob[i] = expf(lg[i] - mx); sum += prob[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) prob[i] *= inv;
    return n;
}

int sample_index(const float *w, int n, Rng *rng)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += w[i];
    float r = rng_float(rng) * sum;
    for (int i = 0; i < n; i++) { r -= w[i]; if (r <= 0.0f) return i; }
    return n - 1;
}

int agent_move_values(const Agent *a, const State *st, Rng *rng, Move *mv, float *val)
{
    int n = lc_moves(st, mv);
    if (a->kind == AG_RANDOM) {
        for (int i = 0; i < n; i++) val[i] = 0.0f;
        return n;
    }
    DrawSamples ds;
    draw_samples_init(st, st->turn, rng, a->draw_samples, &ds);
    for (int i = 0; i < n; i++) {
        if (a->kind == AG_HEUR) val[i] = move_value_heur(st, mv[i], &ds);
        else                    val[i] = move_value_net(a->net, st, mv[i], &ds);
    }
    return n;
}

static Move pick_from_values(const Agent *a, Move *mv, float *val, int n, Rng *rng)
{
    if (a->eps > 0.0f && rng_float(rng) < a->eps) return mv[rng_below(rng, (uint32_t)n)];
    if (a->temp > 0.0f) {
        float best = -1e30f;
        for (int i = 0; i < n; i++) if (val[i] > best) best = val[i];
        float sum = 0.0f, w[MAX_MOVES];
        for (int i = 0; i < n; i++) { w[i] = expf((val[i] - best) / a->temp); sum += w[i]; }
        float r = rng_float(rng) * sum;
        for (int i = 0; i < n; i++) { r -= w[i]; if (r <= 0.0f) return mv[i]; }
        return mv[n - 1];
    }
    float best = -1e30f;
    int nbest = 0;
    Move bm = mv[0];
    for (int i = 0; i < n; i++) {
        if (val[i] > best + 1e-6f) { best = val[i]; bm = mv[i]; nbest = 1; }
        else if (val[i] > best - 1e-6f) { nbest++; if (rng_below(rng, (uint32_t)nbest) == 0) bm = mv[i]; }
    }
    return bm;
}

Move agent_move(const Agent *a, const State *st, Rng *rng)
{
    Move mv[MAX_MOVES];
    float val[MAX_MOVES];
    if (a->kind == AG_MCTS) return search_move(a, st, rng, NULL, NULL);
    if (a->kind == AG_ROLLOUT) return rollout_move(a, st, rng, NULL, NULL);
    if (a->kind == AG_POLICY) {
        float prob[MAX_MOVES];
        int n = policy_probs(a->net, st, mv, prob, NULL);
        if (a->eps > 0.0f && rng_float(rng) < a->eps) return mv[rng_below(rng, (uint32_t)n)];
        if (a->temp > 0.0f) {
            if (a->temp != 1.0f)
                for (int i = 0; i < n; i++) prob[i] = powf(prob[i], 1.0f / a->temp);
            return mv[sample_index(prob, n, rng)];
        }
        int best = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[best]) best = i;
        return mv[best];
    }
    int n = agent_move_values(a, st, rng, mv, val);
    if (a->kind == AG_RANDOM) return mv[rng_below(rng, (uint32_t)n)];
    return pick_from_values(a, mv, val, n, rng);
}
