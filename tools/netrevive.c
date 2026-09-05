/* netrevive -- function-preserving revival of dead trunk units.
 *
 * Census: over the states of a .bst corpus, a trunk unit is DEAD if its
 * ReLU activation never exceeds zero.  Revival re-seeds each dead unit at
 * half init scale and zeroes every weight that READS it (a2 unit: the
 * value/policy/draw/belief/gate head columns; a1 unit: its w2 row), so the
 * net computes exactly the same function while the unit can train again.
 *
 *   netrevive IN.bin corpus.bst MAXSTATES SEED OUT.bin
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/features.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define BST_MAGIC 0x4C424554u
typedef struct { State st; uint16_t game; } Rec;
static float frand(Rng *r) { return rng_float(r) * 2.0f - 1.0f; }
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: netrevive IN.bin corpus.bst MAXSTATES SEED OUT.bin\n"); return 1; }
    Net *n = calloc(1, sizeof(Net));
    if (net_load(n, argv[1])) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    FILE *f = fopen(argv[2], "rb"); uint32_t h[2]; uint64_t count;
    if (!f || fread(h, sizeof h, 1, f) != 1 || fread(&count, sizeof count, 1, f) != 1 || h[0] != BST_MAGIC || h[1] != sizeof(Rec)) { fprintf(stderr, "bad corpus\n"); return 1; }
    long maxs = atol(argv[3]);
    const int h1 = n->h1, h2 = n->h2;
    float *max1 = calloc(h1, sizeof(float)), *max2 = calloc(h2, sizeof(float));
    long seen = 0; Rec r;
    while (fread(&r, sizeof r, 1, f) == 1 && seen < maxs) {
        if (r.st.over) continue;
        Features ft; feat_extract(&r.st, r.st.turn, &ft);
        NetAct act; net_trunk(n, &ft, &act);
        for (int i = 0; i < h1; i++) if (act.a1[i] > max1[i]) max1[i] = act.a1[i];
        for (int i = 0; i < h2; i++) if (act.a2[i] > max2[i]) max2[i] = act.a2[i];
        seen++;
    }
    fclose(f);
    int d1 = 0, d2 = 0;
    for (int i = 0; i < h1; i++) if (max1[i] <= 0.0f) d1++;
    for (int i = 0; i < h2; i++) if (max2[i] <= 0.0f) d2++;
    printf("census over %ld states: a1 dead %d of %d, a2 dead %d of %d\n", seen, d1, h1, d2, h2);
    Rng rng; rng_seed(&rng, strtoull(argv[4], NULL, 10));
    float s1 = sqrtf(2.0f / (float)FEAT_DIM) * 0.5f, s2 = sqrtf(2.0f / (float)h1) * 0.5f;
    for (int i = 0; i < h1; i++) if (max1[i] <= 0.0f) {
        for (int k = 0; k < FEAT_DIM; k++) n->w1[(size_t)k * h1 + i] = frand(&rng) * s1;
        n->b1[i] = 0.0f;
        for (int j = 0; j < h2; j++) n->w2[(size_t)i * h2 + j] = 0.0f;   /* nothing reads it yet */
    }
    for (int j = 0; j < h2; j++) if (max2[j] <= 0.0f) {
        for (int i = 0; i < h1; i++) n->w2[(size_t)i * h2 + j] = frand(&rng) * s2;
        n->b2[j] = 0.0f;
        n->w3[j] = 0.0f;
        for (int k = 0; k < NET_NPLAY; k++) n->wplay[(size_t)k * h2 + j] = 0.0f;
        for (int k = 0; k < NET_NDRAW; k++) n->wdraw[(size_t)k * h2 + j] = 0.0f;
        for (int k = 0; k < NCARD; k++) n->wbel[(size_t)k * h2 + j] = 0.0f;
        if (n->wg) for (int k = 0; k < NET_XR; k++) n->wg[(size_t)k * h2 + j] = 0.0f;
    }
    if (net_save(n, argv[5])) { fprintf(stderr, "cannot save %s\n", argv[5]); return 1; }
    printf("revived %d a1 + %d a2 units -> %s\n", d1, d2, argv[5]);
    return 0;
}
