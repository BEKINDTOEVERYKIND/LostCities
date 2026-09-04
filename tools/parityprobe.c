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
 *   -D: dump one line per solved state:
 *       S game round ply deck odd nplay turns A err loss best_pile best_stall ptop top_stall vstar vtop pv_agree
 *   -b: per-state node budget (all solves of one state); default
 *       LC_SOLVE_BUDGET env or 20M.  States that exhaust it are dropped.
 *   -T: stop generating once this much CPU time is used and report.
 */
#include "../src/lc.h"
#include "../src/net.h"
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

static double cpu_secs(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }

int main(int argc, char **argv)
{
    const char *netpath = "data/best.bin";
    int games = 400, maxdeck = 5, verbose = 0, permove = 0, dump = 0;
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
    }
    if (budget_cfg <= 0) {
        const char *e = getenv("LC_SOLVE_BUDGET");
        budget_cfg = e ? atol(e) : 20 * 1000 * 1000L;
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (net_load(net, netpath)) { fprintf(stderr, "cannot load %s\n", netpath); return 1; }
    Rng rng; rng_seed(&rng, seed);

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
                if (st.deck_left <= maxdeck) {
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
                        if (dump)
                            printf("S %d %d %d %d %d %d %d %d %d %d %d %d %.3f %d %d %d %d\n", g, rd, (int)st.nply, d, odd, nplay, turns_left,
                                   bucketA, err, loss, pile_best, stall_best, pt, top_stall, vstar, vtop, agree);
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
    return 0;
}
