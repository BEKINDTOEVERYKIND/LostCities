/* policydump -- every legal move's policy probability at a probed state.
 *
 *   policydump NET STATE.state
 *
 * The decreplay state loader is reused (same reconstruction).  Output is
 * one line per legal move, probability descending, wager copies folded.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value);

static int name_id_free(const char *nm, uint64_t used)
{
    char b[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, b);
        if (!strcasecmp(b, nm) && !((used >> c) & 1ULL)) return c;
    }
    return -1;
}

static int load_state(const char *path, State *st)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    memset(st, 0, sizeof *st);
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        if (!strcmp(tok, "turn")) st->turn = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "round")) st->round = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "nply")) st->nply = (uint16_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "deck_left")) st->deck_left = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "cum")) {
            int a = atoi(strtok(NULL, " \n")), b = atoi(strtok(NULL, " \n"));
            st->cum[0] = (int16_t)a; st->cum[1] = (int16_t)b;
        } else if (!strncmp(tok, "hand", 4) && tok[4] >= '0' && tok[4] <= '1') {
            int pl = tok[4] - '0'; char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used); if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c; st->hand[pl] |= 1ULL << c; st->hand_n[pl]++;
            }
        } else if (!strncmp(tok, "known", 5) && tok[5] >= '0' && tok[5] <= '1') {
            int pl = tok[5] - '0'; char *w;
            while ((w = strtok(NULL, " \n"))) {
                char b[8];
                for (int c = 0; c < NCARD; c++) {
                    lc_card_name(c, b);
                    if (!strcasecmp(b, w) && ((st->hand[pl] >> c) & 1ULL) && !((st->known[pl] >> c) & 1ULL)) {
                        st->known[pl] |= 1ULL << c; break;
                    }
                }
            }
        } else if (!strcmp(tok, "exp")) {
            int pl = atoi(strtok(NULL, " \n")); int s = atoi(strtok(NULL, " \n")); char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used); if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c; st->played[pl] |= 1ULL << c; st->exp_n[pl][s]++;
                if (CARD_IS_WAGER(c)) st->exp_wager[pl][s]++;
                else { int v = CARD_VALUE(c); if (v > st->exp_top[pl][s]) st->exp_top[pl][s] = (uint8_t)v; st->exp_sum[pl][s] = (uint8_t)(st->exp_sum[pl][s] + v); }
            }
        } else if (!strcmp(tok, "pile")) {
            int s = atoi(strtok(NULL, " \n")); char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used); if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c; st->pile[s][st->pile_n[s]++] = (uint8_t)c; st->discarded |= 1ULL << c;
            }
        }
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s NET STATE.state\n", argv[0]); return 1; }
    Net *net = (Net *)malloc(sizeof(Net));
    if (net_load(net, argv[1])) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    State st;
    if (!load_state(argv[2], &st)) { fprintf(stderr, "bad state file\n"); return 1; }
    Move mv[MAX_MOVES]; float pr[MAX_MOVES], v;
    int n = policy_probs(net, &st, mv, pr, &v);
    n = lc_dedup_wagers(&st, mv, pr, n, 1);
    int ord[MAX_MOVES];
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (pr[ord[j]] > pr[ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    const char *S = "YBWGR";
    printf("value %.1f  (%d legal moves after folding)\n", v * VAL_SCALE, n);
    for (int i = 0; i < n; i++) {
        char nm[8]; lc_card_name(mv[ord[i]].card, nm);
        printf("  %-3s %-7s draw-%c  %.4f\n", nm, mv[ord[i]].discard ? "discard" : "play",
               mv[ord[i]].draw ? S[mv[ord[i]].draw - 1] : 'D', pr[ord[i]]);
    }
    return 0;
}
