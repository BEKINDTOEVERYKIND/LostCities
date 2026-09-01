#include "spec.h"
#include "belx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAMES 16
static char g_names[MAX_NAMES][96];
static int g_nname = 0;

static Net *load_net(const char *path)
{
    Net *n = (Net *)malloc(sizeof(Net));
    if (!n) { fprintf(stderr, "out of memory\n"); exit(1); }
    int r = net_load(n, path);
    if (r != 0) { fprintf(stderr, "cannot load net '%s' (error %d)\n", path, r); exit(1); }
    return n;
}

void spec_parse(const char *spec, Agent *a)
{
    char buf[512];
    snprintf(buf, sizeof buf, "%s", spec);
    char *save = NULL;
    char *tok = strtok_r(buf, ":", &save);
    if (!tok) { fprintf(stderr, "empty agent spec\n"); exit(1); }
    if (!strcmp(tok, "random")) { agent_default(a, AG_RANDOM, NULL); return; }
    if (!strcmp(tok, "heur"))   { agent_default(a, AG_HEUR, NULL); return; }
    if (!strcmp(tok, "hrollout")) {
        /* classical baseline: hand-crafted evaluation with perfect-information
         * Monte Carlo over sampled worlds, no network anywhere */
        agent_default(a, AG_ROLLOUT, NULL);
        a->name = "hrollout";
        char *v;
        if ((v = strtok_r(NULL, ":", &save))) a->dets = atoi(v);
        if ((v = strtok_r(NULL, ":", &save))) a->root_width = atoi(v);
        return;
    }
    if (!strcmp(tok, "net") || !strcmp(tok, "mcts") || !strcmp(tok, "policy") ||
        !strcmp(tok, "rollout") || !strcmp(tok, "rolloutu") || !strcmp(tok, "rollouth")) {
        int is_mcts = !strcmp(tok, "mcts");
        int is_policy = !strcmp(tok, "policy");
        int is_hybrid = !strcmp(tok, "rollouth");
        int is_rollout = !strcmp(tok, "rollout") || !strcmp(tok, "rolloutu") || is_hybrid;
        int is_uniform = !strcmp(tok, "rolloutu");
        char *path = strtok_r(NULL, ":", &save);
        if (!path) { fprintf(stderr, "agent '%s' needs a network path\n", tok); exit(1); }
        Net *n = load_net(path);
        agent_default(a, is_rollout ? AG_ROLLOUT :
                         (is_mcts ? AG_MCTS : (is_policy ? AG_POLICY : AG_NET)), n);
        if (is_hybrid) {
            /* rollouth:NETMAIN:NETBELIEF:<the usual rollout fields> -- the
             * second net's belief head steers world sampling only */
            char *bpath = strtok_r(NULL, ":", &save);
            if (!bpath) { fprintf(stderr, "rollouth needs a second (belief) network path\n"); exit(1); }
            /* extended-format specialist (.blx magic) or a standard net */
            BelX *bx = (BelX *)malloc(sizeof(BelX));
            int br = belx_load(bx, bpath);
            if (br == 0) { a->bx = bx; }
            else if (br == -2) { free(bx); a->net_b = load_net(bpath); }
            else { fprintf(stderr, "cannot load belief file '%s'\n", bpath); exit(1); }
        }
        char *v;
        if (is_rollout) {
            a->no_belief = is_uniform;
            if ((v = strtok_r(NULL, ":", &save))) a->dets = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->root_width = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->cand_floor = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->gate = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->min_cand = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->ply_lo = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->ply_hi = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->eval_cand = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->win_q = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->prune_dom = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->override_k = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->override_min = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->playout_sample = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->solve_deck = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->ov_draw = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sel_k = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->solve_vote = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->solve_budget = atol(v) * 1000000L;
            if ((v = strtok_r(NULL, ":", &save))) a->prior_w0 = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->prior_w1 = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sel_draw = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->draw_filter = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sel_deep = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sym_k = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->bel_samp = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sym_bel = atoi(v);
            /* a malformed spec must degrade to a working agent, not to an
             * uninitialized-read of sum[0] in the candidate loop */
            if (a->dets < 1) a->dets = 1;
            if (a->root_width < 1) a->root_width = 1;
            /* negative draw_filter would read as truthy (= mode 1) in the
             * expansion filter, silently enabling a restriction next to a
             * doc that says 0 = expand all */
            if (a->draw_filter < 0) a->draw_filter = 0;
        } else if (is_policy) {
            if ((v = strtok_r(NULL, ":", &save))) a->temp = (float)atof(v);
        } else if (is_mcts) {
            if ((v = strtok_r(NULL, ":", &save))) a->dets = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sims = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->root_width = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->node_width = atoi(v);
        } else {
            if ((v = strtok_r(NULL, ":", &save))) a->draw_samples = atoi(v);
        }
        if (g_nname < MAX_NAMES) {
            snprintf(g_names[g_nname], sizeof g_names[0], "%s", spec);
            a->name = g_names[g_nname++];
        }
        return;
    }
    fprintf(stderr, "unknown agent kind '%s'\n", tok);
    exit(1);
}
