/* parityprobe -- is the raw policy blind to endgame turn arithmetic?
 *
 * Generates late-round states by sampled raw-policy self-play (as symtest
 * does), keeps every non-terminal state with deck_left <= D, and labels
 * each with the exact solver over the ACTUAL (perfect-information) world.
 *
 * Labeling (default, "hybrid"): lc_solve_root gives the exact mover value
 * V* and a PV move.  If the policy's top-1 (after lc_dedup_wagers fold) is
 * the PV move it is optimal (loss 0).  Otherwise the top-1 is applied and
 * the child solved full-window for the mover with lc_solve_budget; error
 * iff that value < V*, loss = V* - value.  The child solve grants a fresh
 * stall allowance and (for pile draws) one extra ply of cap, so its value
 * is >= the root-consistent one: errors and losses are LOWER bounds --
 * conservative.  A child value above V* (extra freedom found a better
 * line) is clamped to loss 0 and counted.
 * With -m every legal move's child is solved (self-consistent per-move
 * labels: error iff top-1 value < max child value) -- ~5x the cost.
 *
 * Buckets: deck_left value, deck parity (odd = the mover draws the last
 * deck card when both sides draw from the deck; the round ends when the
 * deck is empty after a draw, lc_apply_draw), A/B = mover holds more
 * playable cards than turns left ((deck_left+1)/2) vs not, their cross,
 * whether the solver's best move draws from a pile, and whether it is a
 * pure stall (an unplayable pile card: parity manipulation and nothing
 * else).  Per bucket: N, top-1 error rate, binomial SE, mean point loss,
 * the rate of errors costing >= 2 points, the mean top-1 prob.
 *
 *   parityprobe [-n NET] [-g GAMES] [-s SEED] [-d MAXDECK] [-b NODES]
 *               [-T CPUSECS] [-m] [-v] [-D]
 *   -A SPEC: also play each included state with the deployed search agent
 *       (spec_parse) and score its move against the same solver labels;
 *       -G N solves deck 4-5 states only in the first N games (cost cap),
 *       -x pA -y pB include A / B states with these probabilities
 *       (solver-best=pile states are always included), -S seed2 drives
 *       inclusion, the agent and permutations (the self-play RNG is
 *       untouched, so the states are the main run's).
 *   -P K: for deck<=3 states whose solver-best is a pure stall, re-solve
 *       under K random permutations of the undrawn deck (hands fixed) and
 *       under all d! orders; lenient = the same stall stays value-equal to
 *       V*, strict = the re-solved PV is still a pure stall.
 *   -D: dump one line per solved state:
 *       S game round ply deck odd nplay turns A err loss best_pile best_stall ptop top_stall vstar vtop pv_agree
 *   -b: per-state node budget (all solves of one state); default
 *       LC_SOLVE_BUDGET env or 20M.  States that exhaust it are dropped.
 *   -T: stop generating once this much CPU time is used and report.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

typedef struct {
    const char *name;
    long n, err, err2, pile_best, stall_best;
    double loss, ptop;
} Bucket;

static void badd(Bucket *b, int err, int loss, float ptop, int pile_best, int stall_best)
{
    b->n++;
    if (err) b->err++;
    if (loss >= 2) b->err2++;
    b->loss += loss;
    b->ptop += ptop;
    b->pile_best += pile_best;
    b->stall_best += stall_best;
}
static double brate(const Bucket *b) { return b->n ? (double)b->err / (double)b->n : 0.0; }
static double bse(const Bucket *b)
{
    double p = brate(b);
    return b->n ? sqrt(p * (1.0 - p) / (double)b->n) : 0.0;
}
static double brate2(const Bucket *b) { return b->n ? (double)b->err2 / (double)b->n : 0.0; }
static double bse2(const Bucket *b)
{
    double p = brate2(b);
    return b->n ? sqrt(p * (1.0 - p) / (double)b->n) : 0.0;
}
static void bprint(const Bucket *b)
{
    printf("  %-24s %6ld  %5ld  %6.2f%% +/- %5.2f  %6.3f   %6.2f%% +/- %5.2f   %5.3f  %5.1f%%  %5.1f%%\n",
           b->name, b->n, b->err, 100.0 * brate(b), 100.0 * bse(b),
           b->n ? b->loss / (double)b->n : 0.0,
           100.0 * brate2(b), 100.0 * bse2(b),
           b->n ? b->ptop / (double)b->n : 0.0,
           b->n ? 100.0 * (double)b->pile_best / (double)b->n : 0.0,
           b->n ? 100.0 * (double)b->stall_best / (double)b->n : 0.0);
}
/* a minus b, elementwise (for "everything else" complements) */
static Bucket bdiff(const Bucket *a, const Bucket *b, const char *name)
{
    Bucket r = *a;
    r.name = name;
    r.n -= b->n; r.err -= b->err; r.err2 -= b->err2; r.pile_best -= b->pile_best;
    r.stall_best -= b->stall_best; r.loss -= b->loss; r.ptop -= b->ptop;
    return r;
}
static void bcompare(const char *label, const Bucket *a, const Bucket *b)
{
    double pa = brate(a), pb = brate(b);
    double sea = bse(a), sediff = sqrt(sea * sea + bse(b) * bse(b));
    printf("  %-34s %6.2f%% - %6.2f%% = %+6.2f pts; bucket SE %.2f (%+.2fx), diff SE %.2f (%+.2fx) -> %s\n",
           label, 100.0 * pa, 100.0 * pb, 100.0 * (pa - pb), 100.0 * sea,
           sea > 0 ? (pa - pb) / sea : 0.0, 100.0 * sediff,
           sediff > 0 ? (pa - pb) / sediff : 0.0,
           (pa - pb) > sea ? "ELEVATED (> 1 bucket SE)" : "not elevated");
    double qa = brate2(a), qb = brate2(b);
    double sea2 = bse2(a), sediff2 = sqrt(sea2 * sea2 + bse2(b) * bse2(b));
    printf("  %-34s %6.2f%% - %6.2f%% = %+6.2f pts; bucket SE %.2f (%+.2fx), diff SE %.2f (%+.2fx) -> %s\n",
           "   (errors costing >= 2 pts)", 100.0 * qa, 100.0 * qb, 100.0 * (qa - qb), 100.0 * sea2,
           sea2 > 0 ? (qa - qb) / sea2 : 0.0, 100.0 * sediff2,
           sediff2 > 0 ? (qa - qb) / sediff2 : 0.0,
           (qa - qb) > sea2 ? "ELEVATED (> 1 bucket SE)" : "not elevated");
    printf("  %-34s %6.3f vs %6.3f\n", "   (mean point loss)",
           a->n ? a->loss / (double)a->n : 0.0, b->n ? b->loss / (double)b->n : 0.0);
}

typedef struct {
    const char *name;
    long n, er, ea, e2r, e2a, changed;
    double lr, la, sd, sd2, ld, ld2;
} PBucket;
static void padd(PBucket *b, int er, int lr, int ea, int la, int changed)
{
    b->n++; b->er += er; b->ea += ea; b->e2r += lr >= 2; b->e2a += la >= 2;
    b->lr += lr; b->la += la; b->changed += changed;
    double d = (double)er - (double)ea; b->sd += d; b->sd2 += d * d;
    double dl = (double)lr - (double)la; b->ld += dl; b->ld2 += dl * dl;
}
static double prate(long e, long n) { return n ? (double)e / (double)n : 0.0; }
static double pse(long e, long n) { double p = prate(e, n); return n ? sqrt(p * (1 - p) / (double)n) : 0.0; }
static double paired_se(double sd, double sd2, long n)
{
    if (n < 2) return 0.0;
    double var = (sd2 - sd * sd / (double)n) / (double)(n - 1);
    return var > 0 ? sqrt(var / (double)n) : 0.0;
}
static void pprint(const PBucket *b)
{
    printf("  %-24s %5ld  %6.2f%% +/- %5.2f  %6.2f%% +/- %5.2f  %+6.2f +/- %5.2f  %6.3f %6.3f  %+6.3f +/- %5.3f  %5.1f%%  %5.1f%%  %5.1f%%\n",
           b->name, b->n, 100 * prate(b->er, b->n), 100 * pse(b->er, b->n),
           100 * prate(b->ea, b->n), 100 * pse(b->ea, b->n),
           100 * (b->n ? b->sd / (double)b->n : 0.0), 100 * paired_se(b->sd, b->sd2, b->n),
           b->n ? b->lr / (double)b->n : 0.0, b->n ? b->la / (double)b->n : 0.0,
           b->n ? b->ld / (double)b->n : 0.0, paired_se(b->ld, b->ld2, b->n),
           100 * prate(b->e2r, b->n), 100 * prate(b->e2a, b->n), 100 * prate(b->changed, b->n));
}

static int same_action(Move a, Move b)
{
    int sc = a.card == b.card ||
             (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card) &&
              CARD_SUIT(a.card) == CARD_SUIT(b.card));
    return sc && a.discard == b.discard && a.draw == b.draw;
}

/* a pile draw whose card the mover cannot play: pure stall */
static int is_stall(const State *st, Move m)
{
    if (m.draw == 0) return 0;
    int s = m.draw - 1, p = st->turn;
    if (st->pile_n[s] <= 0) return 0;
    int top = st->pile[s][st->pile_n[s] - 1];
    int playable = CARD_IS_WAGER(top) ? st->exp_top[p][CARD_SUIT(top)] == 0
                                      : CARD_VALUE(top) > st->exp_top[p][CARD_SUIT(top)];
    return !playable;
}

/* Solve st with the undrawn deck reordered by perm (indices into the
 * remaining slice); returns 1 if the stall move keeps value-equal to V*
 * (lenient), sets *strict if the re-solved PV is itself a pure stall,
 * -1 on budget exhaustion. */
static int stall_under_order(const State *st, Move sbest, const int *perm, long budget_cfg, int *strict)
{
    State s = *st;
    int d = st->deck_left, p = st->turn;
    for (int i = 0; i < d; i++) s.deck[st->deck_pos + i] = st->deck[st->deck_pos + perm[i]];
    long b = budget_cfg;
    Move pv;
    int v = lc_solve_root(&s, &b, &pv);
    if (b <= 0) return -1;
    *strict = is_stall(&s, pv);
    if (same_action(pv, sbest)) return 1;
    State c = s;
    lc_apply(&c, sbest);
    int vs = lc_solve_budget(&c, p, &b);
    if (b <= 0) return -1;
    return vs >= v;
}
/* Heap's algorithm over n <= 5 elements, calling fn per order */
static void all_orders(int n, int *a, int k, const State *st, Move sbest, long budget_cfg,
                       int *nopt, int *nstrict, int *nexh, int *ntot)
{
    if (k == 1) {
        int strict = 0;
        int r = stall_under_order(st, sbest, a, budget_cfg, &strict);
        (*ntot)++;
        if (r < 0) (*nexh)++; else { *nopt += r; *nstrict += strict; }
        return;
    }
    all_orders(n, a, k - 1, st, sbest, budget_cfg, nopt, nstrict, nexh, ntot);
    for (int i = 0; i < k - 1; i++) {
        int j = (k & 1) ? 0 : i;
        int t = a[j]; a[j] = a[k - 1]; a[k - 1] = t;
        all_orders(n, a, k - 1, st, sbest, budget_cfg, nopt, nstrict, nexh, ntot);
    }
}

static double cpu_secs(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }

int main(int argc, char **argv)
{
    const char *netpath = "data/best.bin";
    int games = 400, maxdeck = 5, verbose = 0, permove = 0, dump = 0, g45 = -1, nperm = 0;
    const char *agspec = NULL;
    float pA = 0.29f, pB = 0.20f;
    uint64_t seed2 = 4242ULL;
    uint64_t seed = 20260904ULL;
    long budget_cfg = -1;
    double tlimit = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) maxdeck = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) budget_cfg = atol(argv[++i]);
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) tlimit = atof(argv[++i]);
        else if (!strcmp(argv[i], "-m")) permove = 1;
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (!strcmp(argv[i], "-D")) dump = 1;   /* one line per solved state, for post-processing */
        else if (!strcmp(argv[i], "-A") && i + 1 < argc) agspec = argv[++i];
        else if (!strcmp(argv[i], "-G") && i + 1 < argc) g45 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-x") && i + 1 < argc) pA = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) pB = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) seed2 = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-P") && i + 1 < argc) nperm = atoi(argv[++i]);
    }
    if (budget_cfg <= 0) {
        const char *e = getenv("LC_SOLVE_BUDGET");
        budget_cfg = e ? atol(e) : 20 * 1000 * 1000L;
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (net_load(net, netpath)) { fprintf(stderr, "cannot load %s\n", netpath); return 1; }
    Rng rng; rng_seed(&rng, seed);
    Rng rng2; rng_seed(&rng2, seed2);
    Agent ag;
    memset(&ag, 0, sizeof ag);
    if (agspec) spec_parse(agspec, &ag);
    if (g45 < 0) g45 = games;

    enum { B_ALL, B_ODD, B_EVEN, B_A, B_B, B_ODD_A, B_ODD_B, B_EVEN_A, B_EVEN_B,
           B_PILEBEST, B_DECKBEST, B_STALLBEST, B_TOPSTALL, B_D0, B_D1, B_D2, B_D3, B_D4, B_D5, B_D6, B_D7, B_D8, NB };
    Bucket bk[NB];
    memset(bk, 0, sizeof bk);
    bk[B_ALL].name = "all";
    bk[B_ODD].name = "deck odd (mover last)";
    bk[B_EVEN].name = "deck even (opp last)";
    bk[B_A].name = "A: nplay > turns";
    bk[B_B].name = "B: nplay <= turns";
    bk[B_ODD_A].name = "odd x A";
    bk[B_ODD_B].name = "odd x B";
    bk[B_EVEN_A].name = "even x A";
    bk[B_EVEN_B].name = "even x B";
    bk[B_PILEBEST].name = "solver best: pile draw";
    bk[B_DECKBEST].name = "solver best: deck draw";
    bk[B_STALLBEST].name = "solver best: pure stall";
    bk[B_TOPSTALL].name = "policy top-1: pure stall";
    enum { P_ALL, P_ODD, P_EVEN, P_A, P_B, P_PILE, P_DECK, P_STALL, P_ROB, P_NOTROB, P_ROBALL, P_D1, P_D2, P_D3, P_D4, P_D5, NP };
    PBucket pb[NP];
    memset(pb, 0, sizeof pb);
    pb[P_ALL].name = "all (stratified)"; pb[P_ODD].name = "deck odd"; pb[P_EVEN].name = "deck even";
    pb[P_A].name = "A: nplay > turns"; pb[P_B].name = "B: nplay <= turns";
    pb[P_PILE].name = "solver best: pile draw"; pb[P_DECK].name = "solver best: deck draw";
    pb[P_STALL].name = "solver best: pure stall";
    pb[P_ROB].name = "stall robust (>=6/8)"; pb[P_NOTROB].name = "stall not robust";
    pb[P_ROBALL].name = "stall optimal all d! orders";
    pb[P_D1].name = "deck_left = 1"; pb[P_D2].name = "deck_left = 2"; pb[P_D3].name = "deck_left = 3";
    pb[P_D4].name = "deck_left = 4"; pb[P_D5].name = "deck_left = 5";
    long nsearched = 0, ag_exh = 0, ag_clamp = 0;
    long nstall3 = 0, nrob = 0, nrob_strict = 0, nroball = 0, nroball_strict = 0, nrob_half = 0, perm_exh = 0, ident_fail = 0;
    double t_agent = 0.0, t_perm = 0.0;
    static char dn[9][16];
    for (int d = 0; d <= 8; d++) { snprintf(dn[d], sizeof dn[d], "deck_left = %d", d); bk[B_D0 + d].name = dn[d]; }

    long nseen = 0, nexh = 0, nsolved = 0, nodes_total = 0, nodes_root = 0, nodes_child = 0;
    long child_solves = 0, child_above_root = 0, ties = 0, pv_agree = 0;
    long root_mismatch = 0;
    long nmoves_total = 0, games_done = 0;
    long seen_d[9] = {0}, exh_d[9] = {0}, nodes_d[9] = {0};
    double t_cpu0 = cpu_secs(), t_solve = 0.0;
    time_t t_wall0 = time(NULL);
    int stop = 0;

    for (int g = 0; g < games && !stop; g++) {
        int cum[2] = {0, 0};
        for (int rd = 0; rd < MATCH_ROUNDS && !stop; rd++) {
            State st;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)cum[0]; st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(rd & 1);
            while (!st.over) {
                Move mv[MAX_MOVES]; float pr[MAX_MOVES];
                int n = policy_probs(net, &st, mv, pr, NULL);
                if (n <= 0) break;
                if (st.deck_left <= maxdeck && (st.deck_left <= 3 || g < g45)) {
                    nseen++;
                    const int d = st.deck_left;
                    if (d <= 8) seen_d[d]++;
                    Move fmv[MAX_MOVES]; float fpr[MAX_MOVES];
                    memcpy(fmv, mv, sizeof(Move) * (size_t)n);
                    memcpy(fpr, pr, sizeof(float) * (size_t)n);
                    int nf = n > 1 ? lc_dedup_wagers(&st, fmv, fpr, n, 1) : n;
                    int top = 0;
                    for (int i = 1; i < nf; i++) if (fpr[i] > fpr[top]) top = i;

                    const int p = st.turn;
                    long budget = budget_cfg;
                    double ts0 = cpu_secs();
                    Move rbest;
                    int vroot = lc_solve_root(&st, &budget, &rbest);
                    long used = budget_cfg - budget;
                    nodes_root += used;
                    int exhausted = budget <= 0;
                    int vstar = vroot, vtop = vroot, err = 0, loss = 0;
                    Move sbest = rbest;
                    int agree = same_action(rbest, fmv[top]);
                    if (!exhausted) {
                        if (permove) {
                            int val[MAX_MOVES];
                            int vbest = -32000, ibest = -1;
                            for (int i = 0; i < nf; i++) {
                                State c = st;
                                lc_apply(&c, fmv[i]);
                                val[i] = lc_solve_budget(&c, p, &budget);
                                child_solves++;
                                if (budget <= 0) { exhausted = 1; break; }
                                if (val[i] > vbest) { vbest = val[i]; ibest = i; }
                            }
                            if (!exhausted) {
                                vstar = vbest; vtop = val[top]; sbest = fmv[ibest];
                                if (vroot != vbest) root_mismatch++;
                                /* prefer to report a deck-draw optimum when several moves tie */
                                if (sbest.draw > 0)
                                    for (int i = 0; i < nf; i++)
                                        if (val[i] == vbest && fmv[i].draw == 0) { sbest = fmv[i]; break; }
                            }
                        } else if (!agree) {
                            State c = st;
                            lc_apply(&c, fmv[top]);
                            vtop = lc_solve_budget(&c, p, &budget);
                            child_solves++;
                            if (budget <= 0) exhausted = 1;
                            else if (vtop > vroot) { child_above_root++; vtop = vroot; }
                        }
                    }
                    nodes_child += (budget_cfg - (budget > 0 ? budget : 0)) - used;
                    t_solve += cpu_secs() - ts0;
                    nodes_total += budget_cfg - (budget > 0 ? budget : 0);
                    if (d <= 8) nodes_d[d] += budget_cfg - (budget > 0 ? budget : 0);
                    if (exhausted) {
                        nexh++;
                        if (d <= 8) exh_d[d]++;
                    } else {
                        nsolved++;
                        nmoves_total += nf;
                        loss = vstar - vtop;
                        err = loss > 0;
                        if (agree) pv_agree++;
                        else if (!err) ties++;
                        int pile_best = sbest.draw > 0;
                        int stall_best = is_stall(&st, sbest);
                        int top_stall = is_stall(&st, fmv[top]);
                        /* mover's turn arithmetic */
                        int odd = d & 1;
                        int turns_left = (d + 1) / 2;
                        uint8_t cards[HAND_SIZE];
                        int nc = lc_hand_cards(&st, p, cards);
                        int nplay = 0;
                        for (int i = 0; i < nc; i++) {
                            int c = cards[i], s = CARD_SUIT(c);
                            int playable = CARD_IS_WAGER(c) ? (st.exp_top[p][s] == 0)
                                                            : (CARD_VALUE(c) > st.exp_top[p][s]);
                            nplay += playable;
                        }
                        int bucketA = nplay > turns_left;
                        float pt = fpr[top];
                        badd(&bk[B_ALL], err, loss, pt, pile_best, stall_best);
                        badd(&bk[odd ? B_ODD : B_EVEN], err, loss, pt, pile_best, stall_best);
                        badd(&bk[bucketA ? B_A : B_B], err, loss, pt, pile_best, stall_best);
                        badd(&bk[odd ? (bucketA ? B_ODD_A : B_ODD_B) : (bucketA ? B_EVEN_A : B_EVEN_B)], err, loss, pt, pile_best, stall_best);
                        badd(&bk[pile_best ? B_PILEBEST : B_DECKBEST], err, loss, pt, pile_best, stall_best);
                        if (stall_best) badd(&bk[B_STALLBEST], err, loss, pt, pile_best, stall_best);
                        if (top_stall) badd(&bk[B_TOPSTALL], err, loss, pt, pile_best, stall_best);
                        if (d <= 8) badd(&bk[B_D0 + d], err, loss, pt, pile_best, stall_best);
                        /* ---- deployed search agent on the same state ---- */
                        int included = 0, err_a = 0, loss_a = 0, changed = 0, ag_ok = 0;
                        if (agspec) {
                            included = pile_best || rng_float(&rng2) < (bucketA ? pA : pB);
                            if (included) {
                                double ta0 = cpu_secs();
                                Move am = agent_move(&ag, &st, &rng2);
                                t_agent += cpu_secs() - ta0;
                                changed = !same_action(am, fmv[top]);
                                int va = vroot;
                                ag_ok = 1;
                                if (!same_action(am, sbest)) {
                                    if (same_action(am, fmv[top])) va = vtop;
                                    else {
                                        long b2 = budget_cfg;
                                        State c = st;
                                        lc_apply(&c, am);
                                        va = lc_solve_budget(&c, p, &b2);
                                        if (b2 <= 0) { ag_ok = 0; ag_exh++; }
                                        else if (va > vroot) { ag_clamp++; va = vroot; }
                                    }
                                }
                                if (ag_ok) {
                                    nsearched++;
                                    loss_a = vroot - va; err_a = loss_a > 0;
                                    padd(&pb[P_ALL], err, loss, err_a, loss_a, changed);
                                    padd(&pb[odd ? P_ODD : P_EVEN], err, loss, err_a, loss_a, changed);
                                    padd(&pb[bucketA ? P_A : P_B], err, loss, err_a, loss_a, changed);
                                    padd(&pb[pile_best ? P_PILE : P_DECK], err, loss, err_a, loss_a, changed);
                                    if (stall_best) padd(&pb[P_STALL], err, loss, err_a, loss_a, changed);
                                    if (d >= 1 && d <= 5) padd(&pb[P_D1 + d - 1], err, loss, err_a, loss_a, changed);
                                }
                            }
                        }
                        /* ---- stall-label robustness to deck order ---- */
                        int nopt8 = -1, nstrict8 = 0, nopt_all = -1, nstrict_all = 0, ntot_all = 0;
                        if (nperm > 0 && d <= 3 && stall_best) {
                            double tp0 = cpu_secs();
                            nstall3++;
                            int perm[8], ex8 = 0;
                            nopt8 = 0;
                            for (int k = 0; k < nperm; k++) {
                                for (int i = 0; i < d; i++) perm[i] = i;
                                for (int i = d - 1; i > 0; i--) {
                                    int j = (int)rng_below(&rng2, (uint32_t)i + 1);
                                    int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
                                }
                                int strict = 0;
                                int r = stall_under_order(&st, sbest, perm, budget_cfg, &strict);
                                if (r < 0) ex8++; else { nopt8 += r; nstrict8 += strict; }
                            }
                            perm_exh += ex8;
                            int a[8];
                            for (int i = 0; i < d; i++) a[i] = i;
                            int exa = 0;
                            nopt_all = 0;
                            all_orders(d, a, d, &st, sbest, budget_cfg, &nopt_all, &nstrict_all, &exa, &ntot_all);
                            perm_exh += exa;
                            /* identity order (first of Heap's) must reproduce the label */
                            {
                                int ide[8], strict = 0;
                                for (int i = 0; i < d; i++) ide[i] = i;
                                if (stall_under_order(&st, sbest, ide, budget_cfg, &strict) != 1) ident_fail++;
                            }
                            int robust = nopt8 >= 6;
                            if (robust) nrob++;
                            if (nstrict8 >= 6) nrob_strict++;
                            if (nopt8 * 2 >= nperm) nrob_half++;
                            int roball = ntot_all > 0 && exa == 0 && nopt_all == ntot_all;
                            if (roball) nroball++;
                            if (ntot_all > 0 && exa == 0 && nstrict_all == ntot_all) nroball_strict++;
                            if (ag_ok) {
                                padd(&pb[robust ? P_ROB : P_NOTROB], err, loss, err_a, loss_a, changed);
                                if (roball) padd(&pb[P_ROBALL], err, loss, err_a, loss_a, changed);
                            }
                            t_perm += cpu_secs() - tp0;
                            if (dump)
                                printf("R %d %d %d %d %d %d %d %d %d %d %d\n", g, rd, (int)st.nply, d, nopt8, nstrict8, nopt_all, nstrict_all, ntot_all,
                                       ag_ok ? err_a : -1, err);
                        }
                        if (dump)
                            printf("S %d %d %d %d %d %d %d %d %d %d %d %d %.3f %d %d %d %d\n", g, rd, (int)st.nply, d, odd, nplay, turns_left,
                                   bucketA, err, loss, pile_best, stall_best, pt, top_stall, vstar, vtop, agree);
                        if (dump && included && ag_ok)
                            printf("G %d %d %d %d %d %d %d %d\n", g, rd, (int)st.nply, d, err, loss, err_a, loss_a);
                        if (verbose && err) {
                            char b1[48], b2[48];
                            lc_move_name(&st, fmv[top], b1);
                            lc_move_name(&st, sbest, b2);
                            printf("ERR g%d r%d ply%d deck%d %s nplay=%d turns=%d: policy %s (p=%.2f, v=%d) solver %s (v=%d) loss %d%s\n",
                                   g, rd, (int)st.nply, d, odd ? "odd" : "even", nplay, turns_left,
                                   b1, pt, vtop, b2, vstar, loss, stall_best ? " [stall]" : "");
                        }
                    }
                    if (tlimit > 0 && cpu_secs() - t_cpu0 > tlimit) { stop = 1; break; }
                }
                /* advance by sampling the policy (diverse states) */
                float u = rng_float(&rng), c = 0.0f; int pick = n - 1;
                for (int i = 0; i < n; i++) { c += pr[i]; if (u < c) { pick = i; break; } }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0); cum[1] += lc_score(&st, 1);
        }
        if (!stop) games_done++;
        if ((g + 1) % 25 == 0)
            fprintf(stderr, "  %d games, %ld states seen, %ld solved, %ld exhausted, cpu %.0fs\n",
                    g + 1, nseen, nsolved, nexh, cpu_secs() - t_cpu0);
    }
    double cpu = cpu_secs() - t_cpu0;
    double wall = difftime(time(NULL), t_wall0);

    printf("parityprobe: net %s seed %llu games %d (completed %ld) maxdeck %d per-state budget %ld nodes, mode %s\n",
           netpath, (unsigned long long)seed, games, games_done, maxdeck, budget_cfg,
           permove ? "per-move child solves" : "hybrid (root solve + top-1 child solve)");
    printf("states seen %ld, solved %ld, exhausted %ld (%.2f%%); cpu %.1fs (solver %.1fs), wall %.0fs, %.2f solved states/cpu-s\n",
           nseen, nsolved, nexh, nseen ? 100.0 * (double)nexh / (double)nseen : 0.0, cpu, t_solve, wall,
           cpu > 0 ? (double)nsolved / cpu : 0.0);
    printf("nodes: total %ld (%.0f per state: root %.0f, child %.0f); %ld child solves; legal moves after wager fold %.1f per state\n",
           nodes_total, nseen ? (double)nodes_total / (double)nseen : 0.0,
           nseen ? (double)nodes_root / (double)nseen : 0.0,
           nseen ? (double)nodes_child / (double)nseen : 0.0, child_solves,
           nsolved ? (double)nmoves_total / (double)nsolved : 0.0);
    printf("per deck_left: ");
    for (int dd = 0; dd <= maxdeck && dd <= 8; dd++)
        printf("d%d seen %ld exh %ld (%.1f%%) %.2fM nodes/state; ", dd, seen_d[dd], exh_d[dd],
               seen_d[dd] ? 100.0 * (double)exh_d[dd] / (double)seen_d[dd] : 0.0,
               seen_d[dd] ? (double)nodes_d[dd] / (double)seen_d[dd] / 1e6 : 0.0);
    printf("\n");
    if (permove)
        printf("cross-check: lc_solve_root value != max child value in %ld of %ld states (child model is more permissive)\n",
               root_mismatch, nsolved);
    printf("policy top-1 == solver PV move: %ld of %ld (%.1f%%); top-1 != PV but value-equal (tie): %ld; child value above root V* (clamped): %ld\n",
           pv_agree, nsolved, nsolved ? 100.0 * (double)pv_agree / (double)nsolved : 0.0, ties, child_above_root);
    printf("\n  %-24s %6s  %5s  %-17s  %-7s  %-18s  %5s  %s  %s\n", "bucket", "N", "err", "top-1 error +/- SE", "meanloss", "err(>=2pt) +/- SE", "ptop", "best=pile", "best=stall");
    bprint(&bk[B_ALL]);
    bprint(&bk[B_ODD]);
    bprint(&bk[B_EVEN]);
    bprint(&bk[B_A]);
    bprint(&bk[B_B]);
    bprint(&bk[B_ODD_A]);
    bprint(&bk[B_ODD_B]);
    bprint(&bk[B_EVEN_A]);
    bprint(&bk[B_EVEN_B]);
    bprint(&bk[B_PILEBEST]);
    bprint(&bk[B_DECKBEST]);
    bprint(&bk[B_STALLBEST]);
    bprint(&bk[B_TOPSTALL]);
    for (int dd = 0; dd <= maxdeck && dd <= 8; dd++) bprint(&bk[B_D0 + dd]);
    printf("\ncomparisons (rate difference vs the sensitive bucket's own binomial SE, and vs the SE of the difference):\n");
    bcompare("deck odd vs even", &bk[B_ODD], &bk[B_EVEN]);
    bcompare("A (nplay>turns) vs B", &bk[B_A], &bk[B_B]);
    Bucket rest_odd_a = bdiff(&bk[B_ALL], &bk[B_ODD_A], "not odd x A");
    bcompare("odd x A vs everything else", &bk[B_ODD_A], &rest_odd_a);
    Bucket rest_even_a = bdiff(&bk[B_ALL], &bk[B_EVEN_A], "not even x A");
    bcompare("even x A vs everything else", &bk[B_EVEN_A], &rest_even_a);
    bcompare("solver-best pile draw vs deck", &bk[B_PILEBEST], &bk[B_DECKBEST]);
    Bucket rest_stall = bdiff(&bk[B_ALL], &bk[B_STALLBEST], "not stall-best");
    bcompare("solver-best pure stall vs rest", &bk[B_STALLBEST], &rest_stall);
    if (agspec) {
        printf("\nsearch agent: %s\n", agspec);
        printf("searched %ld states (stratified: solver-best=pile always, A with p=%.2f, B with p=%.2f; deck 4-5 only in games < %d), agent CPU %.1fs (%.3f s/decision), agent child solves exhausted %ld, clamped %ld\n",
               nsearched, pA, pB, g45, t_agent, nsearched ? t_agent / (double)nsearched : 0.0, ag_exh, ag_clamp);
        printf("  %-24s %5s  %-17s  %-17s  %-16s  %-13s  %-16s  %s  %s  %s\n", "bucket", "N", "raw err +/- SE", "agent err +/- SE",
               "paired diff+/-SE", "loss raw agt", "paired dloss+/-SE", "raw>=2", "agt>=2", "agt!=top1");
        for (int i = 0; i < NP; i++) if (pb[i].n) pprint(&pb[i]);
    }
    if (nperm > 0) {
        printf("\nstall-label robustness (deck<=3 states whose solver-best is a pure stall): %ld states, %d random orders each + all d! orders; permutation solves exhausted %ld; identity-order label failures %ld\n",
               nstall3, nperm, perm_exh, ident_fail);
        printf("  stall stays optimal in >= 6/%d random orders: %ld (%.1f%%) [lenient: same stall value-equal to V*]; re-solved PV is a pure stall in >= 6/%d: %ld (%.1f%%) [strict]\n",
               nperm, nrob, nstall3 ? 100.0 * (double)nrob / (double)nstall3 : 0.0, nperm, nrob_strict, nstall3 ? 100.0 * (double)nrob_strict / (double)nstall3 : 0.0);
        printf("  optimal in >= half of the random orders: %ld (%.1f%%); optimal under ALL d! orders: %ld (%.1f%%) lenient, %ld (%.1f%%) strict\n",
               nrob_half, nstall3 ? 100.0 * (double)nrob_half / (double)nstall3 : 0.0,
               nroball, nstall3 ? 100.0 * (double)nroball / (double)nstall3 : 0.0,
               nroball_strict, nstall3 ? 100.0 * (double)nroball_strict / (double)nstall3 : 0.0);
    }
    printf("cpu total %.1fs: solver %.1fs, agent %.1fs, permutations %.1fs\n", cpu_secs() - t_cpu0, t_solve, t_agent, t_perm);
    return 0;
}
