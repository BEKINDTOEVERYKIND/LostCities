#include "net.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static float gauss(Rng *r)
{
    float u1 = rng_float(r) + 1e-7f, u2 = rng_float(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

void net_init(Net *n, uint64_t seed)
{
    Rng r; rng_seed(&r, seed);
    float s1 = sqrtf(2.0f / (float)FEAT_DIM);
    for (int i = 0; i < FEAT_DIM; i++)
        for (int h = 0; h < NET_H1; h++) n->w1[i][h] = gauss(&r) * s1;
    for (int h = 0; h < NET_H1; h++) n->b1[h] = 0.0f;
    float s2 = sqrtf(2.0f / (float)NET_H1);
    for (int i = 0; i < NET_H1; i++)
        for (int h = 0; h < NET_H2; h++) n->w2[i][h] = gauss(&r) * s2;
    for (int h = 0; h < NET_H2; h++) n->b2[h] = 0.0f;
    float s3 = sqrtf(1.0f / (float)NET_H2);
    for (int h = 0; h < NET_H2; h++) n->w3[h] = gauss(&r) * s3;
    n->b3 = 0.0f;
    float s4 = 0.1f * sqrtf(1.0f / (float)NET_H2);
    for (int i = 0; i < NET_NPLAY; i++) {
        for (int h = 0; h < NET_H2; h++) n->wplay[i][h] = gauss(&r) * s4;
        n->bplay[i] = 0.0f;
    }
    for (int i = 0; i < NET_NDRAW; i++) {
        for (int h = 0; h < NET_H2; h++) n->wdraw[i][h] = gauss(&r) * s4;
        n->bdraw[i] = 0.0f;
    }
}

void net_zero(Net *n)
{
    memset(n, 0, sizeof(*n));
}

void net_trunk(const Net *n, const Features *f, NetAct *act)
{
    float h1[NET_H1];
    for (int h = 0; h < NET_H1; h++) h1[h] = n->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = n->w1[f->idx[k]];
        for (int h = 0; h < NET_H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        const float *w = n->w1[FEAT_BIN + j];
        for (int h = 0; h < NET_H1; h++) h1[h] += x * w[h];
    }
    for (int h = 0; h < NET_H1; h++) act->a1[h] = h1[h] > 0.0f ? h1[h] : 0.0f;

    float h2[NET_H2];
    for (int h = 0; h < NET_H2; h++) h2[h] = n->b2[h];
    for (int i = 0; i < NET_H1; i++) {
        float a = act->a1[i];
        if (a == 0.0f) continue;
        const float *w = n->w2[i];
        for (int h = 0; h < NET_H2; h++) h2[h] += a * w[h];
    }
    for (int h = 0; h < NET_H2; h++) act->a2[h] = h2[h] > 0.0f ? h2[h] : 0.0f;
}

float net_value_act(const Net *n, const NetAct *act)
{
    float o = n->b3;
    for (int h = 0; h < NET_H2; h++) o += act->a2[h] * n->w3[h];
    return o;
}

static inline float dot_h2(const float *w, const float *a)
{
    float o = 0.0f;
    for (int h = 0; h < NET_H2; h++) o += a[h] * w[h];
    return o;
}

void net_policy_act(const Net *n, const NetAct *act, const uint16_t *mv, int nmv, float *logits)
{
    float pl[NET_NPLAY], dr[NET_NDRAW];
    uint8_t hp[NET_NPLAY] = { 0 }, hd[NET_NDRAW] = { 0 };
    for (int i = 0; i < nmv; i++) {
        int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
        int id = MOVE_DRAW(mv[i]);
        if (!hp[ip]) { pl[ip] = n->bplay[ip] + dot_h2(n->wplay[ip], act->a2); hp[ip] = 1; }
        if (!hd[id]) { dr[id] = n->bdraw[id] + dot_h2(n->wdraw[id], act->a2); hd[id] = 1; }
        logits[i] = pl[ip] + dr[id];
    }
}

float net_value(const Net *n, const Features *f)
{
    NetAct act;
    net_trunk(n, f, &act);
    return net_value_act(n, &act);
}

void net_backward(const Net *n, const Features *f, const NetAct *act,
                  float dvalue, const uint16_t *mv, const float *dlogit, int nmv,
                  Net *g)
{
    float d2[NET_H2];
    for (int h = 0; h < NET_H2; h++) d2[h] = 0.0f;

    if (dvalue != 0.0f) {
        g->b3 += dvalue;
        for (int h = 0; h < NET_H2; h++) {
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
            float *gw = g->wplay[ip];
            const float *w = n->wplay[ip];
            g->bplay[ip] += d;
            for (int h = 0; h < NET_H2; h++) { gw[h] += d * act->a2[h]; d2[h] += d * w[h]; }
        }
        for (int k = 0; k < nd; k++) {
            int id = dlist[k];
            float d = sd[id];
            if (d == 0.0f) continue;
            float *gw = g->wdraw[id];
            const float *w = n->wdraw[id];
            g->bdraw[id] += d;
            for (int h = 0; h < NET_H2; h++) { gw[h] += d * act->a2[h]; d2[h] += d * w[h]; }
        }
    }
    for (int h = 0; h < NET_H2; h++) if (act->a2[h] <= 0.0f) d2[h] = 0.0f;

    float d1[NET_H1];
    for (int h = 0; h < NET_H2; h++) g->b2[h] += d2[h];
    for (int i = 0; i < NET_H1; i++) {
        float a = act->a1[i];
        if (a != 0.0f) {
            float *gw = g->w2[i];
            const float *w = n->w2[i];
            float acc = 0.0f;
            for (int h = 0; h < NET_H2; h++) { gw[h] += a * d2[h]; acc += w[h] * d2[h]; }
            d1[i] = acc;
        } else {
            d1[i] = 0.0f;
        }
    }
    for (int h = 0; h < NET_H1; h++) g->b1[h] += d1[h];
    for (int k = 0; k < f->nidx; k++) {
        float *gw = g->w1[f->idx[k]];
        for (int h = 0; h < NET_H1; h++) gw[h] += d1[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        float *gw = g->w1[FEAT_BIN + j];
        for (int h = 0; h < NET_H1; h++) gw[h] += x * d1[h];
    }
}

void net_adam_step(Net *n, const Net *g, Adam *a, float lr, float scale, float wd)
{
    a->t++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)a->t);
    float bc2 = 1.0f - powf(b2, (float)a->t);
    float step = lr * sqrtf(bc2) / bc1;

    float *w = (float *)n, *gm = (float *)&a->m, *gv = (float *)&a->v;
    const float *gr = (const float *)g;
    size_t nw = sizeof(Net) / sizeof(float);
    for (size_t i = 0; i < nw; i++) {
        float grad = gr[i] * scale + wd * w[i];
        gm[i] = b1 * gm[i] + (1.0f - b1) * grad;
        gv[i] = b2 * gv[i] + (1.0f - b2) * grad * grad;
        w[i] -= step * gm[i] / (sqrtf(gv[i]) + eps);
    }
}

#define NET_MAGIC 0x4C435650U /* "LCVP" */

int net_save(const Net *n, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    uint32_t hdr[6] = { NET_MAGIC, FEAT_DIM, NET_H1, NET_H2, NET_NPLAY, 3 };
    fwrite(hdr, sizeof(hdr), 1, fp);
    fwrite(n, sizeof(Net), 1, fp);
    fclose(fp);
    return 0;
}

int net_load(Net *n, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint32_t hdr[6];
    if (fread(hdr, sizeof(hdr), 1, fp) != 1) { fclose(fp); return -1; }
    if (hdr[0] != NET_MAGIC || hdr[1] != FEAT_DIM || hdr[2] != NET_H1 ||
        hdr[3] != NET_H2 || hdr[4] != NET_NPLAY) { fclose(fp); return -2; }
    if (fread(n, sizeof(Net), 1, fp) != 1) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}
