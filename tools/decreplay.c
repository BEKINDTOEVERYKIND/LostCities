/* decreplay -- replay one probed decision under an agent spec, many seeds.
 *
 * The probe-suite harness (tools/suite.py): load a statedump.py .state file,
 * run the spec's move selection NSEEDS times with fresh RNG streams, and
 * print the tally of chosen moves by display identity -- wager copies fold
 * into one name, so "any Yx counts as Yx" exactly as the manifest verdicts
 * are written.
 *
 *   decreplay STATE.state BASE_SEED NSEEDS SPEC
 *
 * Output, one line per distinct chosen move:
 *   R4 discard draw-0 : 12/20
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* state-file loading, same reconstruction as tools/qpair.c -S */

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
            st->cum[0] = (int16_t)(a > 320 ? 320 : (a < -320 ? -320 : a));
            st->cum[1] = (int16_t)(b > 320 ? 320 : (b < -320 ? -320 : b));
        } else if (!strncmp(tok, "hand", 4) && tok[4] >= '0' && tok[4] <= '1') {
            int pl = tok[4] - '0';
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->hand[pl] |= 1ULL << c;
                st->hand_n[pl]++;
            }
        } else if (!strncmp(tok, "known", 5) && tok[5] >= '0' && tok[5] <= '1') {
            int pl = tok[5] - '0';
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                char b[8];
                for (int c = 0; c < NCARD; c++) {
                    lc_card_name(c, b);
                    if (!strcasecmp(b, w) && ((st->hand[pl] >> c) & 1ULL) &&
                        !((st->known[pl] >> c) & 1ULL)) {
                        st->known[pl] |= 1ULL << c;
                        break;
                    }
                }
            }
        } else if (!strcmp(tok, "exp")) {
            int pl = atoi(strtok(NULL, " \n"));
            int s = atoi(strtok(NULL, " \n"));
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->played[pl] |= 1ULL << c;
                st->exp_n[pl][s]++;
                if (CARD_IS_WAGER(c)) st->exp_wager[pl][s]++;
                else {
                    int v = CARD_VALUE(c);
                    if (v > st->exp_top[pl][s]) st->exp_top[pl][s] = (uint8_t)v;
                    st->exp_sum[pl][s] = (uint8_t)(st->exp_sum[pl][s] + v);
                }
            }
        } else if (!strcmp(tok, "pile")) {
            int s = atoi(strtok(NULL, " \n"));
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->pile[s][st->pile_n[s]++] = (uint8_t)c;
                st->discarded |= 1ULL << c;
            }
        }
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s STATE.state BASE_SEED NSEEDS SPEC\n", argv[0]);
        return 1;
    }
    State st;
    if (!load_state(argv[1], &st)) {
        fprintf(stderr, "decreplay: bad state file %s\n", argv[1]);
        return 1;
    }
    uint64_t base = (uint64_t)strtoull(argv[2], NULL, 10);
    int nseeds = atoi(argv[3]);
    Agent ag;
    spec_parse(argv[4], &ag);

    char keys[64][24];
    int counts[64], nkey = 0;
    for (int s = 0; s < nseeds; s++) {
        Rng rng;
        rng_seed(&rng, base + 0x9E3779B97F4A7C15ULL * (uint64_t)(s + 1));
        Move m = agent_move(&ag, &st, &rng);
        char nm[8];
        lc_card_name(m.card, nm);
        char key[24];
        snprintf(key, sizeof key, "%s %s draw-%d", nm,
                 m.discard ? "discard" : "play", m.draw);
        int k = -1;
        for (int i = 0; i < nkey; i++) if (!strcmp(keys[i], key)) { k = i; break; }
        if (k < 0 && nkey < 64) { k = nkey++; snprintf(keys[k], sizeof keys[k], "%s", key); counts[k] = 0; }
        if (k >= 0) counts[k]++;
    }
    for (int i = 0; i < nkey; i++)
        printf("%s : %d/%d\n", keys[i], counts[i], nseeds);
    return 0;
}
