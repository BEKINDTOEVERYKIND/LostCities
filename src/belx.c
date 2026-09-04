#include "belx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t belx_nfloat(int h1, int h2)
{
    return (size_t)BELX_XDIM * h1 + h1 + (size_t)h1 * h2 + h2 + (size_t)NCARD * h2 + NCARD;
}

void belx_wire(BelX *x)
{
    float *p = x->blk;
    x->w1 = p; p += (size_t)BELX_XDIM * x->h1;
    x->b1 = p; p += x->h1;
    x->w2 = p; p += (size_t)x->h1 * x->h2;
    x->b2 = p; p += x->h2;
    x->wb = p; p += (size_t)NCARD * x->h2;
    x->bb = p;
}

void belx_alloc(BelX *x, int h1, int h2)
{
    x->h1 = h1; x->h2 = h2;
    x->blk = (float *)calloc(belx_nfloat(h1, h2), sizeof(float));
    if (!x->blk) { fprintf(stderr, "belx: out of memory\n"); exit(1); }
    belx_wire(x);
}

void belx_save(const BelX *x, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    uint32_t h[4] = { BELX_MAGIC, (uint32_t)x->h1, (uint32_t)x->h2, BELX_XDIM };
    fwrite(h, sizeof h, 1, f);
    fwrite(x->blk, sizeof(float), belx_nfloat(x->h1, x->h2), f);
    fclose(f);
}

int belx_load(BelX *x, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t h[4];
    if (fread(h, sizeof h, 1, f) != 1 || h[0] != BELX_MAGIC) { fclose(f); return -2; }
    if ((h[3] != BELX_XDIM && h[3] != BELX_XDIM_V5) ||
        h[1] < 1 || h[1] > NET_H1_MAX || h[2] < 1 || h[2] > NET_H2_MAX) {
        fprintf(stderr, "belx '%s': dims h1=%u h2=%u xdim=%u vs compiled xdim=%d (or the v5 prefix %d)"
                        " -- refused\n", path, h[1], h[2], h[3], BELX_XDIM, BELX_XDIM_V5);
        fclose(f);
        return -4;
    }
    belx_alloc(x, (int)h[1], (int)h[2]);
    const size_t H1 = (size_t)x->h1;
    if (h[3] == BELX_XDIM) {
        if (fread(x->blk, sizeof(float), belx_nfloat(x->h1, x->h2), f) !=
            belx_nfloat(x->h1, x->h2)) { fclose(f); return -3; }
    } else {
        /* pre-turn-block file: its w1 is [FEAT_DIM_V5 base rows][XBIN+XDENSE
         * extra rows]; ours has the turn block between them.  Read the base
         * rows, leave the turn rows zero (they contribute nothing, so the
         * loaded specialist is the identical function -- the same trick as
         * net_load), then the extra rows and the rest of the block. */
        size_t base = (size_t)FEAT_DIM_V5 * H1;
        size_t extra = (size_t)(BELX_XBIN + BELX_XDENSE) * H1;
        size_t after = belx_nfloat(x->h1, x->h2) - (size_t)BELX_XDIM * H1;
        if (fread(x->w1, sizeof(float), base, f) != base ||
            fread(x->w1 + (size_t)FEAT_DIM * H1, sizeof(float), extra, f) != extra ||
            fread(x->b1, sizeof(float), after, f) != after) { fclose(f); return -3; }
    }
    fclose(f);
    return 0;
}

void belx_from_net(BelX *x, const Net *n)
{
    belx_alloc(x, n->h1, n->h2);
    for (int r = 0; r < FEAT_DIM; r++)
        memcpy(x->w1 + (size_t)r * x->h1, n->w1 + (size_t)r * n->h1, sizeof(float) * n->h1);
    memcpy(x->b1, n->b1, sizeof(float) * n->h1);
    memcpy(x->w2, n->w2, sizeof(float) * (size_t)n->h1 * n->h2);
    memcpy(x->b2, n->b2, sizeof(float) * n->h2);
    memcpy(x->wb, n->wbel, sizeof(float) * (size_t)NCARD * n->h2);
    memcpy(x->bb, n->bbel, sizeof(float) * NCARD);
}

void belx_feat(const State *st, int p, Features *base, BelXFeat *xf)
{
    const int o = p ^ 1;
    feat_extract(st, p, base);
    int n = 0;
    uint64_t mask = st->disc_by[o];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; xf->idx[n++] = (uint16_t)(FEAT_DIM + c); }
    mask = st->disc_by[p];
    while (mask) { int c = __builtin_ctzll(mask); mask &= mask - 1; xf->idx[n++] = (uint16_t)(FEAT_DIM + NCARD + c); }
    xf->nidx = n;
    for (int s = 0; s < NSUIT; s++) {
        float po = st->passed[o][s] * (1.0f / 8.0f); if (po > 1.0f) po = 1.0f;
        float pp = st->passed[p][s] * (1.0f / 8.0f); if (pp > 1.0f) pp = 1.0f;
        xf->dense[s] = po;
        xf->dense[NSUIT + s] = pp;
    }
}

void belx_trunk(const BelX *x, const Features *f, const BelXFeat *xf, NetAct *act)
{
    const int H1 = x->h1, H2 = x->h2;
    float h1[NET_H1_MAX];
    for (int h = 0; h < H1; h++) h1[h] = x->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = x->w1 + (size_t)f->idx[k] * H1;
        for (int h = 0; h < H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float v = f->dense[j];
        if (v == 0.0f) continue;
        const float *w = x->w1 + (size_t)(FEAT_BIN + j) * H1;
        for (int h = 0; h < H1; h++) h1[h] += v * w[h];
    }
    for (int k = 0; k < xf->nidx; k++) {
        const float *w = x->w1 + (size_t)xf->idx[k] * H1;
        for (int h = 0; h < H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < BELX_XDENSE; j++) {
        float v = xf->dense[j];
        if (v == 0.0f) continue;
        const float *w = x->w1 + (size_t)(FEAT_DIM + BELX_XBIN + j) * H1;
        for (int h = 0; h < H1; h++) h1[h] += v * w[h];
    }
    for (int h = 0; h < H1; h++) act->a1[h] = h1[h] > 0.0f ? h1[h] : 0.0f;
    float h2[NET_H2_MAX];
    for (int h = 0; h < H2; h++) h2[h] = x->b2[h];
    for (int i = 0; i < H1; i++) {
        float a = act->a1[i];
        if (a == 0.0f) continue;
        const float *w = x->w2 + (size_t)i * H2;
        for (int h = 0; h < H2; h++) h2[h] += a * w[h];
    }
    for (int h = 0; h < H2; h++) act->a2[h] = h2[h] > 0.0f ? h2[h] : 0.0f;
}

void belx_logits(const BelX *x, const NetAct *act, const uint8_t *cards, int nc, float *lg)
{
    const int H2 = x->h2;
    for (int i = 0; i < nc; i++) {
        const float *w = x->wb + (size_t)cards[i] * H2;
        float v = x->bb[cards[i]];
        for (int h = 0; h < H2; h++) v += w[h] * act->a2[h];
        lg[i] = v;
    }
}
