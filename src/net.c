#include "net.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* the block is laid out in this order (matches the on-disk format and the
 * pre-runtime-sizing struct layout exactly) */
static size_t net_nfloat(int h1, int h2)
{
    return (size_t)FEAT_DIM * h1 + h1
         + (size_t)h1 * h2 + h2
         + h2 + 1
         + (size_t)NET_NPLAY * h2 + NET_NPLAY
         + (size_t)NET_NDRAW * h2 + NET_NDRAW
         + (size_t)NCARD * h2 + NCARD;
}

static void net_wire(Net *n)
{
    float *p = n->blk;
    n->w1 = p;    p += (size_t)FEAT_DIM * n->h1;
    n->b1 = p;    p += n->h1;
    n->w2 = p;    p += (size_t)n->h1 * n->h2;
    n->b2 = p;    p += n->h2;
    n->w3 = p;    p += n->h2;
    n->b3 = p;    p += 1;
    n->wplay = p; p += (size_t)NET_NPLAY * n->h2;
    n->bplay = p; p += NET_NPLAY;
    n->wdraw = p; p += (size_t)NET_NDRAW * n->h2;
    n->bdraw = p; p += NET_NDRAW;
    n->wbel = p;  p += (size_t)NCARD * n->h2;
    n->bbel = p;  p += NCARD;
    n->nfloat = (size_t)(p - n->blk);
}

int net_alloc(Net *n, int h1, int h2)
{
    if (h1 < 1 || h1 > NET_H1_MAX || h2 < 1 || h2 > NET_H2_MAX) return -1;
    n->h1 = h1;
    n->h2 = h2;
    n->blk = (float *)calloc(net_nfloat(h1, h2), sizeof(float));
    if (!n->blk) return -1;
    net_wire(n);
    return 0;
}

void net_free(Net *n)
{
    free(n->blk);
    n->blk = NULL;
    n->nfloat = 0;
}

int net_alloc_like(Net *dst, const Net *src)
{
    return net_alloc(dst, src->h1, src->h2);
}

void net_copy(Net *dst, const Net *src)
{
    memcpy(dst->blk, src->blk, src->nfloat * sizeof(float));
}

static float gauss(Rng *r)
{
    float u1 = rng_float(r) + 1e-7f, u2 = rng_float(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

void net_init(Net *n, uint64_t seed)
{
    const int H1 = n->h1, H2 = n->h2;
    Rng r; rng_seed(&r, seed);
    float s1 = sqrtf(2.0f / (float)FEAT_DIM);
    for (int i = 0; i < FEAT_DIM; i++)
        for (int h = 0; h < H1; h++) n->w1[(size_t)i * H1 + h] = gauss(&r) * s1;
    for (int h = 0; h < H1; h++) n->b1[h] = 0.0f;
    float s2 = sqrtf(2.0f / (float)H1);
    for (int i = 0; i < H1; i++)
        for (int h = 0; h < H2; h++) n->w2[(size_t)i * H2 + h] = gauss(&r) * s2;
    for (int h = 0; h < H2; h++) n->b2[h] = 0.0f;
    float s3 = sqrtf(1.0f / (float)H2);
    for (int h = 0; h < H2; h++) n->w3[h] = gauss(&r) * s3;
    *n->b3 = 0.0f;
    float s4 = 0.1f * sqrtf(1.0f / (float)H2);
    for (int i = 0; i < NET_NPLAY; i++) {
        for (int h = 0; h < H2; h++) n->wplay[(size_t)i * H2 + h] = gauss(&r) * s4;
        n->bplay[i] = 0.0f;
    }
    for (int i = 0; i < NET_NDRAW; i++) {
        for (int h = 0; h < H2; h++) n->wdraw[(size_t)i * H2 + h] = gauss(&r) * s4;
        n->bdraw[i] = 0.0f;
    }
    for (int i = 0; i < NCARD; i++) {
        for (int h = 0; h < H2; h++) n->wbel[(size_t)i * H2 + h] = gauss(&r) * s4;
        n->bbel[i] = 0.0f;
    }
}

/* random-init only the belief head (for upgrading older files) */
static void net_init_belief(Net *n, uint64_t seed)
{
    const int H2 = n->h2;
    Rng r; rng_seed(&r, seed);
    float s4 = 0.1f * sqrtf(1.0f / (float)H2);
    for (int i = 0; i < NCARD; i++) {
        for (int h = 0; h < H2; h++) n->wbel[(size_t)i * H2 + h] = gauss(&r) * s4;
        n->bbel[i] = 0.0f;
    }
}

void net_zero(Net *n)
{
    memset(n->blk, 0, n->nfloat * sizeof(float));
}

void net_trunk(const Net *n, const Features *f, NetAct *act)
{
    const int H1 = n->h1, H2 = n->h2;
    float h1[NET_H1_MAX];
    for (int h = 0; h < H1; h++) h1[h] = n->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = n->w1 + (size_t)f->idx[k] * H1;
        for (int h = 0; h < H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        const float *w = n->w1 + (size_t)(FEAT_BIN + j) * H1;
        for (int h = 0; h < H1; h++) h1[h] += x * w[h];
    }
    for (int h = 0; h < H1; h++) act->a1[h] = h1[h] > 0.0f ? h1[h] : 0.0f;

    float h2[NET_H2_MAX];
    for (int h = 0; h < H2; h++) h2[h] = n->b2[h];
    for (int i = 0; i < H1; i++) {
        float a = act->a1[i];
        if (a == 0.0f) continue;
        const float *w = n->w2 + (size_t)i * H2;
        for (int h = 0; h < H2; h++) h2[h] += a * w[h];
    }
    for (int h = 0; h < H2; h++) act->a2[h] = h2[h] > 0.0f ? h2[h] : 0.0f;
}

float net_value_act(const Net *n, const NetAct *act)
{
    float o = *n->b3;
    for (int h = 0; h < n->h2; h++) o += act->a2[h] * n->w3[h];
    return o;
}

static inline float dot_h2(const float *w, const float *a, int h2)
{
    float o = 0.0f;
    for (int h = 0; h < h2; h++) o += a[h] * w[h];
    return o;
}

void net_policy_act(const Net *n, const NetAct *act, const uint16_t *mv, int nmv, float *logits)
{
    const int H2 = n->h2;
    float pl[NET_NPLAY], dr[NET_NDRAW];
    uint8_t hp[NET_NPLAY] = { 0 }, hd[NET_NDRAW] = { 0 };
    for (int i = 0; i < nmv; i++) {
        int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
        int id = MOVE_DRAW(mv[i]);
        if (!hp[ip]) { pl[ip] = n->bplay[ip] + dot_h2(n->wplay + (size_t)ip * H2, act->a2, H2); hp[ip] = 1; }
        if (!hd[id]) { dr[id] = n->bdraw[id] + dot_h2(n->wdraw + (size_t)id * H2, act->a2, H2); hd[id] = 1; }
        logits[i] = pl[ip] + dr[id];
    }
}

void net_belief_act(const Net *n, const NetAct *act, const uint8_t *cards, int nc, float *logits)
{
    const int H2 = n->h2;
    for (int i = 0; i < nc; i++)
        logits[i] = n->bbel[cards[i]] + dot_h2(n->wbel + (size_t)cards[i] * H2, act->a2, H2);
}

float net_value(const Net *n, const Features *f)
{
    NetAct act;
    net_trunk(n, f, &act);
    return net_value_act(n, &act);
}

void net_backward(const Net *n, const Features *f, const NetAct *act,
                  float dvalue, const uint16_t *mv, const float *dlogit, int nmv,
                  const uint8_t *bc, const float *dbel, int nb,
                  Net *g)
{
    const int H1 = n->h1, H2 = n->h2;
    float d2[NET_H2_MAX];
    for (int h = 0; h < H2; h++) d2[h] = 0.0f;

    if (dvalue != 0.0f) {
        *g->b3 += dvalue;
        for (int h = 0; h < H2; h++) {
            g->w3[h] += dvalue * act->a2[h];
            d2[h] += dvalue * n->w3[h];
        }
    }
    if (dlogit) {
        /* Sum the per-move gradient into the components it is built from, then
         * touch each component's weight row once. */
        float sp[NET_NPLAY], sd[NET_NDRAW];
        int plist[MAX_MOVES], dlist[NET_NDRAW];
        int np = 0, nd = 0;
        uint8_t hp[NET_NPLAY] = { 0 }, hd[NET_NDRAW] = { 0 };
        for (int i = 0; i < nmv; i++) {
            int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
            int id = MOVE_DRAW(mv[i]);
            if (!hp[ip]) { hp[ip] = 1; sp[ip] = 0.0f; plist[np++] = ip; }
            if (!hd[id]) { hd[id] = 1; sd[id] = 0.0f; dlist[nd++] = id; }
            sp[ip] += dlogit[i];
            sd[id] += dlogit[i];
        }
        for (int k = 0; k < np; k++) {
            int ip = plist[k];
            float d = sp[ip];
            if (d == 0.0f) continue;
            float *gw = g->wplay + (size_t)ip * H2;
            const float *w = n->wplay + (size_t)ip * H2;
            g->bplay[ip] += d;
            for (int h = 0; h < H2; h++) { gw[h] += d * act->a2[h]; d2[h] += d * w[h]; }
        }
        for (int k = 0; k < nd; k++) {
            int id = dlist[k];
            float d = sd[id];
            if (d == 0.0f) continue;
            float *gw = g->wdraw + (size_t)id * H2;
            const float *w = n->wdraw + (size_t)id * H2;
            g->bdraw[id] += d;
            for (int h = 0; h < H2; h++) { gw[h] += d * act->a2[h]; d2[h] += d * w[h]; }
        }
    }
    if (dbel) {
        for (int i = 0; i < nb; i++) {
            float dv = dbel[i];
            if (dv == 0.0f) continue;
            int card = bc[i];
            float *gw = g->wbel + (size_t)card * H2;
            const float *w = n->wbel + (size_t)card * H2;
            g->bbel[card] += dv;
            for (int h = 0; h < H2; h++) { gw[h] += dv * act->a2[h]; d2[h] += dv * w[h]; }
        }
    }
    for (int h = 0; h < H2; h++) if (act->a2[h] <= 0.0f) d2[h] = 0.0f;

    float d1[NET_H1_MAX];
    for (int h = 0; h < H2; h++) g->b2[h] += d2[h];
    for (int i = 0; i < H1; i++) {
        float a = act->a1[i];
        if (a != 0.0f) {
            float *gw = g->w2 + (size_t)i * H2;
            const float *w = n->w2 + (size_t)i * H2;
            float acc = 0.0f;
            for (int h = 0; h < H2; h++) { gw[h] += a * d2[h]; acc += w[h] * d2[h]; }
            d1[i] = acc;
        } else {
            d1[i] = 0.0f;
        }
    }
    for (int h = 0; h < H1; h++) g->b1[h] += d1[h];
    for (int k = 0; k < f->nidx; k++) {
        float *gw = g->w1 + (size_t)f->idx[k] * H1;
        for (int h = 0; h < H1; h++) gw[h] += d1[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        float *gw = g->w1 + (size_t)(FEAT_BIN + j) * H1;
        for (int h = 0; h < H1; h++) gw[h] += x * d1[h];
    }
}

void net_adam_step(Net *n, const Net *g, Adam *a, float lr, float scale, float wd)
{
    a->t++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)a->t);
    float bc2 = 1.0f - powf(b2, (float)a->t);
    float step = lr * sqrtf(bc2) / bc1;

    float *w = n->blk, *gm = a->m.blk, *gv = a->v.blk;
    const float *gr = g->blk;
    size_t nw = n->nfloat;
    for (size_t i = 0; i < nw; i++) {
        float grad = gr[i] * scale + wd * w[i];
        gm[i] = b1 * gm[i] + (1.0f - b1) * grad;
        gv[i] = b2 * gv[i] + (1.0f - b2) * grad * grad;
        w[i] -= step * gm[i] / (sqrtf(gv[i]) + eps);
    }
}

#define NET_MAGIC 0x4C435651U /* "LCVQ" */

int net_save(const Net *n, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    uint32_t hdr[6] = { NET_MAGIC, FEAT_DIM, (uint32_t)n->h1, (uint32_t)n->h2, NET_NPLAY, 4 };
    fwrite(hdr, sizeof(hdr), 1, fp);
    fwrite(n->blk, n->nfloat * sizeof(float), 1, fp);
    fclose(fp);
    return 0;
}

int net_load(Net *n, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint32_t hdr[6];
    if (fread(hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); return -1; }
    /* width comes from the file now; only the feature space and the policy
     * head's structural dims must match the build */
    if (hdr[0] != NET_MAGIC || hdr[1] != FEAT_DIM ||
        hdr[4] != NET_NPLAY) { fclose(fp); return -2; }
    if (net_alloc(n, (int)hdr[2], (int)hdr[3]) != 0) { fclose(fp); return -3; }
    if (hdr[5] >= 4) {
        if (fread(n->blk, n->nfloat * sizeof(float), 1, fp) != 1) {
            fclose(fp); net_free(n); return -1;
        }
    } else {
        /* older file without the belief head: load the prefix, init the rest */
        size_t belief = (size_t)NCARD * n->h2 + NCARD;
        size_t prefix = (n->nfloat - belief) * sizeof(float);
        if (fread(n->blk, prefix, 1, fp) != 1) { fclose(fp); net_free(n); return -1; }
        net_init_belief(n, 0xBE11EFULL);
    }
    fclose(fp);
    return 0;
}
