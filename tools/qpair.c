/* qpair -- paired rollout Q comparison for chosen candidate moves at a
 * replayed position.
 *
 * The analysis dump only searches the moves the policy proposes, so a move
 * the policy assigns ~0% never gets a Q value even when a human wants to see
 * one.  This tool answers "what was move X actually worth there?": it replays
 * a recorded game (same seed, same moves, so the hidden deck is identical) to
 * a given ply, then evaluates any moves you name with the same machinery the
 * rollout agent uses -- shared belief-sampled worlds, argmax-policy playouts
 * to the end of the round -- and reports each candidate's Q with a standard
 * error, plus the *paired* difference against the first candidate, which is
 * the number that actually decides between moves.
 *
 * The moves file is one move per line, "CARD ACT DRAW" (e.g. "Y2 d deck",
 * "Gx p Y"); plies 1..P-1 are replayed, matching the "n" field of the
 * analysis JSON.  Multi-round replay follows the match loop of analyze.c.
 *
 *   ./bin/qpair -n NET.bin -s SEED -f moves.txt -p 5 -w 4000 \
 *               -c "Y2 d deck" -c "W4 p deck"
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>

#define MAXC 8

/* find CARD (by display name) in p's hand; any wager of the suit matches */
static int find_card(const State *st, int p, const char *name)
{
    char b[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, b);
        if (!strcasecmp(b, name) && ((st->hand[p] >> c) & 1ULL)) return c;
    }
    return -1;
}

static int parse_move(const State *st, int p, const char *cs, const char *as,
                      const char *ws, Move *out)
{
    static const char SUIT_CH[NSUIT + 1] = "YBWGR";
    int card = find_card(st, p, cs);
    if (card < 0) { fprintf(stderr, "qpair: %s not in hand\n", cs); return 0; }
    int disc = (as[0] == 'd' || as[0] == 'D');
    int draw = -1;
    if (!strcasecmp(ws, "deck")) draw = 0;
    else for (int s = 0; s < NSUIT; s++)
        if (ws[0] == SUIT_CH[s] || ws[0] == SUIT_CH[s] + 32) draw = s + 1;
    if (draw < 0) { fprintf(stderr, "qpair: bad draw '%s'\n", ws); return 0; }
    out->card = (uint8_t)card;
    out->discard = (uint8_t)disc;
    out->draw = (uint8_t)draw;
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv);
    for (int i = 0; i < n; i++)
        if (mv[i].card == out->card && mv[i].discard == out->discard &&
            mv[i].draw == out->draw) return 1;
    fprintf(stderr, "qpair: %s %s %s is not legal here\n", cs, as, ws);
    return 0;
}

/* identical to the playout in rollout.c: argmax policy to the end of round */
static int playout(const Net *net, State *s, int p)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = policy_probs(net, s, mv, score, NULL);
        if (n <= 0) break;
        int best = 0;
        for (int i = 1; i < n; i++) if (score[i] > score[best]) best = i;
        lc_apply(s, mv[best]);
    }
    return lc_score(s, p) - lc_score(s, p ^ 1);
}

int main(int argc, char **argv)
{
    const char *netpath = NULL, *movespath = NULL;
    uint64_t seed = 1;
    int target = 1, worlds = 2000;
    const char *cand_str[MAXC];
    int ncand = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) movespath = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) target = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) worlds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc && ncand < MAXC) cand_str[ncand++] = argv[++i];
        else {
            fprintf(stderr, "usage: %s -n NET -s SEED -f MOVES -p PLY [-w WORLDS] -c \"CARD p|d DRAW\" [-c ...]\n", argv[0]);
            return 1;
        }
    }
    if (!netpath || !movespath || ncand == 0) { fprintf(stderr, "qpair: need -n, -f and at least one -c\n"); return 1; }

    Net *net = (Net *)malloc(sizeof(Net));
    if (!net || net_load(net, netpath)) { fprintf(stderr, "qpair: cannot load %s\n", netpath); return 1; }

    FILE *mf = fopen(movespath, "r");
    if (!mf) { fprintf(stderr, "qpair: cannot open %s\n", movespath); return 1; }

    /* replay, following the match loop of analyze.c / match.c */
    Rng rng;
    rng_seed(&rng, seed);
    int cum[2] = { 0, 0 }, rd = 0;
    State st;
    lc_deal(&st, &rng);
    st.round = 0;
    st.turn = 0;
    char line[64], cs[16], as[16], ws[16];
    for (int ply = 1; ply < target; ply++) {
        if (st.over) {
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
            rd++;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)(cum[0] > 320 ? 320 : (cum[0] < -320 ? -320 : cum[0]));
            st.cum[1] = (int16_t)(cum[1] > 320 ? 320 : (cum[1] < -320 ? -320 : cum[1]));
            st.turn = (uint8_t)(rd & 1);
        }
        if (!fgets(line, sizeof line, mf) || sscanf(line, "%15s %15s %15s", cs, as, ws) != 3) {
            fprintf(stderr, "qpair: moves file ends before ply %d\n", target);
            return 1;
        }
        Move m;
        if (!parse_move(&st, st.turn, cs, as, ws, &m)) { fprintf(stderr, "  (at ply %d)\n", ply); return 1; }
        lc_apply(&st, m);
    }
    fclose(mf);
    if (st.over) { fprintf(stderr, "qpair: round already over at ply %d\n", target); return 1; }

    const int p = st.turn;
    char b[8];
    printf("position: ply %d, round %d, player %d to move, deck %d, cum [%d,%d]\nhand:", target, st.round, p, st.deck_left, cum[0], cum[1]);
    uint8_t hc[HAND_SIZE];
    int hn = lc_hand_cards(&st, p, hc);
    for (int i = 0; i < hn; i++) { lc_card_name(hc[i], b); printf(" %s", b); }
    printf("\n");

    Move cand[MAXC];
    for (int c = 0; c < ncand; c++) {
        if (sscanf(cand_str[c], "%15s %15s %15s", cs, as, ws) != 3 ||
            !parse_move(&st, p, cs, as, ws, &cand[c])) {
            fprintf(stderr, "qpair: bad candidate '%s'\n", cand_str[c]);
            return 1;
        }
    }

    /* the policy's own opinion of each candidate, for context */
    Move pmv[MAX_MOVES];
    float prob[MAX_MOVES], value;
    int nleg = policy_probs(net, &st, pmv, prob, &value);
    printf("value head: %+.1f   policy priors:", value * VAL_SCALE);
    for (int c = 0; c < ncand; c++) {
        float pr = 0.0f;
        for (int i = 0; i < nleg; i++)
            if (pmv[i].card == cand[c].card && pmv[i].discard == cand[c].discard &&
                pmv[i].draw == cand[c].draw) pr = prob[i];
        printf("  [%s] %.4f", cand_str[c], pr);
    }
    printf("\n");

    double *val = (double *)malloc(sizeof(double) * (size_t)ncand * (size_t)worlds);
    for (int d = 0; d < worlds; d++) {
        State world;
        determinize_b(&st, p, &rng, net, &world);
        for (int c = 0; c < ncand; c++) {
            State s = world;               /* same world for every candidate */
            lc_apply(&s, cand[c]);
            val[c * worlds + d] = playout(net, &s, p);
        }
    }

    printf("\n%-16s %10s %8s     %s\n", "candidate", "Q(round)", "+-SE", "paired diff vs first");
    double m0 = 0.0;
    for (int c = 0; c < ncand; c++) {
        double mean = 0.0, var = 0.0, dmean = 0.0, dvar = 0.0;
        for (int d = 0; d < worlds; d++) mean += val[c * worlds + d];
        mean /= worlds;
        if (c == 0) m0 = mean;
        for (int d = 0; d < worlds; d++) {
            double e = val[c * worlds + d] - mean;
            var += e * e;
            double dd = val[c * worlds + d] - val[0 * worlds + d];
            double de = dd - (mean - m0);
            dvar += de * de;
        }
        var /= (worlds - 1);
        dvar /= (worlds - 1);
        dmean = mean - m0;
        printf("%-16s %+10.2f %8.2f", cand_str[c], mean, sqrt(var / worlds));
        if (c > 0) printf("     %+.2f +- %.2f", dmean, sqrt(dvar / worlds));
        printf("\n");
    }
    free(val);
    free(net);
    return 0;
}
