/* rootchan -- root-channel population count for a candidate MAIN net.
 * On stored search-play states (nply >= 14, deck_left <= 14, >= 2 tournament
 * candidates): (A) does the candidate net's symmetrized (sym_k 120) top move
 * differ from c20's?  (B) among those, does the STANDING search (c20 main,
 * tournament string) pick something other than the candidate's top?
 * A x B = the share of searched decisions where the swap can change the
 * played move through the root channel.
 *   rootchan STANDING_SPEC CAND_SPEC corpus.bst MAXSTATES SEED
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BST_MAGIC 0x4C424554u
typedef struct { State st; uint16_t game; } Rec;
static int same_move(Move a, Move b) {
    int ca = a.card, cb = b.card;
    int same_card = ca == cb || (CARD_IS_WAGER(ca) && CARD_IS_WAGER(cb) && CARD_SUIT(ca) == CARD_SUIT(cb));
    return same_card && a.discard == b.discard && a.draw == b.draw;
}
static Move top_of(const Agent *a, const State *st, Rng *rng, int *ncand) {
    Move mv[MAX_MOVES]; float pr[MAX_MOVES], v;
    int n = agent_policy_probs(a, st, rng, mv, pr, &v);
    int best = 0, nc = 0;
    for (int i = 0; i < n; i++) { if (pr[i] > pr[best]) best = i; if (pr[i] >= 0.02f) nc++; }
    *ncand = nc;
    return mv[best];
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    Agent std, cand; spec_parse(argv[1], &std); spec_parse(argv[2], &cand);
    FILE *f = fopen(argv[3], "rb"); uint32_t h[2]; uint64_t count;
    if (!f || fread(h, sizeof h, 1, f) != 1 || fread(&count, sizeof count, 1, f) != 1 || h[0] != BST_MAGIC || h[1] != sizeof(Rec)) { fprintf(stderr, "bad corpus\n"); return 1; }
    long maxs = atol(argv[4]); Rng rng; rng_seed(&rng, strtoull(argv[5], NULL, 10));
    long seen = 0, elig = 0, diff = 0, notplayed = 0, outside = 0;
    int bydeck_e[16] = {0}, bydeck_d[16] = {0}, bydeck_np[16] = {0};
    Rec r;
    while (fread(&r, sizeof r, 1, f) == 1 && elig < maxs) {
        seen++;
        const State *st = &r.st;
        if (st->over || st->nply < 14 || st->deck_left > 14 || st->deck_left < 1) continue;
        int nc0, nc1;
        Move t0 = top_of(&std, st, &rng, &nc0);
        Move t1 = top_of(&cand, st, &rng, &nc1);
        if (nc0 < 2 && nc1 < 2) continue;
        elig++; bydeck_e[st->deck_left]++;
        if (same_move(t0, t1)) continue;
        diff++; bydeck_d[st->deck_left]++;
        /* is the candidate's top inside c20's tournament candidate set at all? */
        SearchStats ss; float v;
        Move played = rollout_move(&std, st, &rng, &v, &ss);
        int inside = 0;
        for (int i = 0; i < ss.n; i++) if (same_move(ss.mv[i], t1)) inside = 1;
        if (!inside) outside++;
        if (!same_move(played, t1)) { notplayed++; bydeck_np[st->deck_left]++; }
    }
    printf("states %ld eligible %ld  A(top differs) %ld (%.1f%%)  B(standing search does not play cand top | A) %ld (%.1f%%)  outside c20 top-5: %ld  AxB = %.2f%% of searched decisions\n",
           seen, elig, diff, 100.0 * diff / (elig ? elig : 1), notplayed, 100.0 * notplayed / (diff ? diff : 1), outside, 100.0 * notplayed / (elig ? elig : 1));
    printf("by deck_left (elig/differs/not-played):");
    for (int d = 1; d <= 14; d++) printf(" %d:%d/%d/%d", d, bydeck_e[d], bydeck_d[d], bydeck_np[d]);
    printf("\n");
    return 0;
}
