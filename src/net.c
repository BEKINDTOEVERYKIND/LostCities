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
         + (size_t)NCARD * h2 + NCARD
         + (size_t)NET_XR * h2 + (size_t)NET_NPLAY * NET_XR + (size_t)NET_NDRAW * NET_XR;
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
    n->wg = p;    p += (size_t)NET_XR * n->h2;
    n->xu = p;    p += (size_t)NET_NPLAY * NET_XR;
    n->xv = p;    p += (size_t)NET_NDRAW * NET_XR;
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

static void net_init_xhead(Net *n, uint64_t seed);
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
    net_init_xhead(n, seed ^ 0x5848ULL);
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

/* interaction head init: V and the gate rows at the head scale, U zero so
 * the term contributes nothing until training moves it */
static void net_init_xhead(Net *n, uint64_t seed)
{
    const int H2 = n->h2;
    Rng r; rng_seed(&r, seed);
    float s4 = 0.1f * sqrtf(1.0f / (float)H2);
    for (size_t i = 0; i < (size_t)NET_XR * H2; i++) n->wg[i] = gauss(&r) * s4;
    for (size_t i = 0; i < (size_t)NET_NPLAY * NET_XR; i++) n->xu[i] = 0.0f;
    for (size_t i = 0; i < (size_t)NET_NDRAW * NET_XR; i++) n->xv[i] = gauss(&r) * 0.1f;
}

void net_zero(Net *n)
{
    memset(n->blk, 0, n->nfloat * sizeof(float));
}

/* ---- trunk forward -------------------------------------------------------
 *
 * Every search workload is dominated by this forward pass, and the pass is
 * bound by load/store traffic rather than arithmetic: the plain loop streams
 * the whole accumulator vector through L1 once per active input row (load,
 * add, store all h1 floats for each of the ~100 sparse rows and ~130 dense
 * rows, then all h2 floats for each of the ~250 live layer-2 rows).  It is
 * re-tiled here so a block of NET_TILE outputs sits in registers across the
 * entire input loop -- bias in, every active row folded in, ReLU, one store
 * -- which divides the accumulator traffic by the tile width.
 *
 * Decision-preserving: for every output element the summation order is
 * exactly the reference loop's -- bias, then the sparse indices in list
 * order, then the non-zero dense inputs in index order, then (layer 2) the
 * non-zero activations in index order -- and the ReLU is the same C
 * expression, so under the project's -ffast-math build the activations are
 * bit-identical to the reference loop.  tools/trunkbench.c checks that on
 * real states and on odd widths, and must stay green whenever this code is
 * touched.  Widths are runtime properties: whatever does not fill a whole
 * tile goes through the range kernels, which are the reference loop
 * restricted to a column range and also the entire path on compilers
 * without the GNU vector extension. */

/* reference loop, layer 1, output columns [h0, h1) */
static void trunk1_range(const Net *n, const Features *f, float *a1, int h0, int h1)
{
    const int H1 = n->h1;
    float acc[NET_H1_MAX];
    for (int h = h0; h < h1; h++) acc[h] = n->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = n->w1 + (size_t)f->idx[k] * H1;
        for (int h = h0; h < h1; h++) acc[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        const float *w = n->w1 + (size_t)(FEAT_BIN + j) * H1;
        for (int h = h0; h < h1; h++) acc[h] += x * w[h];
    }
    for (int h = h0; h < h1; h++) a1[h] = acc[h] > 0.0f ? acc[h] : 0.0f;
}

/* reference loop, layer 2, output columns [h0, h1) */
static void trunk2_range(const Net *n, const float *a1, float *a2, int h0, int h1)
{
    const int H1 = n->h1, H2 = n->h2;
    float acc[NET_H2_MAX];
    for (int h = h0; h < h1; h++) acc[h] = n->b2[h];
    for (int i = 0; i < H1; i++) {
        float a = a1[i];
        if (a == 0.0f) continue;
        const float *w = n->w2 + (size_t)i * H2;
        for (int h = h0; h < h1; h++) acc[h] += a * w[h];
    }
    for (int h = h0; h < h1; h++) a2[h] = acc[h] > 0.0f ? acc[h] : 0.0f;
}

#if defined(__GNUC__) && !defined(NET_TRUNK_SCALAR)
#define NET_TILE 64            /* outputs per register tile: 8 x 8-wide accumulators */
#define NET_TV (NET_TILE / 8)
typedef float v8sf __attribute__((vector_size(32)));
/* unaligned, aliasing view of eight consecutive floats of a weight row */
typedef float v8su __attribute__((vector_size(32), aligned(4), may_alias));
#define V8(p) (*(const v8su *)(p))

/* ReLU of a finished tile into out[0..NET_TILE): the tile is spilled to a
 * scratch row and passed through the reference expression, so the compiler
 * lowers it exactly as it lowers the range kernel's ReLU */
static inline void tile_relu(const v8sf *acc, float *out)
{
    float tmp[NET_TILE];
    for (int r = 0; r < NET_TV; r++) *(v8su *)(tmp + 8 * r) = acc[r];
    for (int h = 0; h < NET_TILE; h++) out[h] = tmp[h] > 0.0f ? tmp[h] : 0.0f;
}

/* layer 1 over the leading ntile full tiles; returns the columns done */
static int trunk1_tiles(const Net *n, const Features *f, float *a1, int ntile)
{
    const int H1 = n->h1;
    /* the non-zero dense inputs, gathered once rather than once per tile */
    int nd = 0;
    float dx[FEAT_DENSE];
    const float *dw[FEAT_DENSE];
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        dx[nd] = x;
        dw[nd] = n->w1 + (size_t)(FEAT_BIN + j) * H1;
        nd++;
    }
    for (int t = 0; t < ntile; t++) {
        const int h0 = t * NET_TILE;
        v8sf acc[NET_TV];
        for (int r = 0; r < NET_TV; r++) acc[r] = V8(n->b1 + h0 + 8 * r);
        for (int k = 0; k < f->nidx; k++) {
            const float *w = n->w1 + (size_t)f->idx[k] * H1 + h0;
            for (int r = 0; r < NET_TV; r++) acc[r] += V8(w + 8 * r);
        }
        for (int d = 0; d < nd; d++) {
            const float x = dx[d];
            const v8sf xv = { x, x, x, x, x, x, x, x };
            const float *w = dw[d] + h0;
            for (int r = 0; r < NET_TV; r++) acc[r] += xv * V8(w + 8 * r);
        }
        tile_relu(acc, a1 + h0);
    }
    return ntile * NET_TILE;
}

/* layer 2 over the leading ntile full tiles; returns the columns done */
static int trunk2_tiles(const Net *n, const float *a1, float *a2, int ntile)
{
    const int H1 = n->h1, H2 = n->h2;
    /* the live (non-zero) activations, gathered once */
    int nz = 0;
    float za[NET_H1_MAX];
    uint16_t zi[NET_H1_MAX];
    for (int i = 0; i < H1; i++) {
        float a = a1[i];
        if (a == 0.0f) continue;
        za[nz] = a;
        zi[nz] = (uint16_t)i;
        nz++;
    }
    for (int t = 0; t < ntile; t++) {
        const int h0 = t * NET_TILE;
        v8sf acc[NET_TV];
        for (int r = 0; r < NET_TV; r++) acc[r] = V8(n->b2 + h0 + 8 * r);
        for (int z = 0; z < nz; z++) {
            const float a = za[z];
            const v8sf av = { a, a, a, a, a, a, a, a };
            const float *w = n->w2 + (size_t)zi[z] * H2 + h0;
            for (int r = 0; r < NET_TV; r++) acc[r] += av * V8(w + 8 * r);
        }
        tile_relu(acc, a2 + h0);
    }
    return ntile * NET_TILE;
}
#endif

void net_trunk(const Net *n, const Features *f, NetAct *act)
{
    const int H1 = n->h1, H2 = n->h2;
    int h = 0;
#if defined(__GNUC__) && !defined(NET_TRUNK_SCALAR)
    h = trunk1_tiles(n, f, act->a1, H1 / NET_TILE);
#endif
    if (h < H1) trunk1_range(n, f, act->a1, h, H1);
    h = 0;
#if defined(__GNUC__) && !defined(NET_TRUNK_SCALAR)
    h = trunk2_tiles(n, act->a1, act->a2, H2 / NET_TILE);
#endif
    if (h < H2) trunk2_range(n, act->a1, act->a2, h, H2);
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

/* is any U entry non-zero?  Called only when the cheap sentinel entries are
 * zero, i.e. on freshly extended nets, where it is a 1920-float scan. */
static int net_xu_nonzero(const Net *n)
{
    for (size_t i = 0; i < (size_t)NET_NPLAY * NET_XR; i++) if (n->xu[i] != 0.0f) return 1;
    return 0;
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
    /* play x draw interaction: the additive head cannot say "take the Green
     * wager if I discard R4 but not if I play G4 onto Green" -- P(draw |
     * action) is identical across actions up to legality.  A rank-NET_XR
     * bilinear term gated by the state gives each (action, draw) pair its
     * own conditional at +NET_XR MACs per move.  Skipped entirely while
     * the U block is all-zero (a freshly extended v4 net), which keeps
     * such a net bit-identical to its v4 self. */
    if (n->xu[0] != 0.0f || n->xu[1] != 0.0f || n->xu[NET_NPLAY * NET_XR - 1] != 0.0f ||
        net_xu_nonzero(n)) {
        float g[NET_XR];
        for (int j = 0; j < NET_XR; j++) g[j] = dot_h2(n->wg + (size_t)j * H2, act->a2, H2);
        for (int i = 0; i < nmv; i++) {
            int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
            int id = MOVE_DRAW(mv[i]);
            const float *u = n->xu + (size_t)ip * NET_XR, *v = n->xv + (size_t)id * NET_XR;
            float x = 0.0f;
            for (int j = 0; j < NET_XR; j++) x += g[j] * u[j] * v[j];
            logits[i] += x;
        }
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
        /* play x draw interaction gradients: dU[ip][j] += d g_j V[id][j],
         * dV[id][j] += d g_j U[ip][j], dg_j = sum_m d_m U V, then dW_g and
         * the trunk share.  Always computed in training: U grows from zero
         * through exactly this path. */
        {
            float gv[NET_XR], dg[NET_XR];
            for (int j = 0; j < NET_XR; j++) { gv[j] = dot_h2(n->wg + (size_t)j * H2, act->a2, H2); dg[j] = 0.0f; }
            for (int i = 0; i < nmv; i++) {
                float d = dlogit[i];
                if (d == 0.0f) continue;
                int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
                int id = MOVE_DRAW(mv[i]);
                const float *u = n->xu + (size_t)ip * NET_XR, *v = n->xv + (size_t)id * NET_XR;
                float *gu = g->xu + (size_t)ip * NET_XR, *gvv = g->xv + (size_t)id * NET_XR;
                for (int j = 0; j < NET_XR; j++) {
                    gu[j] += d * gv[j] * v[j];
                    gvv[j] += d * gv[j] * u[j];
                    dg[j] += d * u[j] * v[j];
                }
            }
            for (int j = 0; j < NET_XR; j++) {
                if (dg[j] == 0.0f) continue;
                float *gw = g->wg + (size_t)j * H2;
                const float *w = n->wg + (size_t)j * H2;
                for (int h = 0; h < H2; h++) { gw[h] += dg[j] * act->a2[h]; d2[h] += dg[j] * w[h]; }
            }
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
    uint32_t hdr[6] = { NET_MAGIC, FEAT_DIM, (uint32_t)n->h1, (uint32_t)n->h2, NET_NPLAY, 5 };
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
    size_t xsec = (size_t)NET_XR * n->h2 + (size_t)NET_NPLAY * NET_XR + (size_t)NET_NDRAW * NET_XR;
    if (hdr[5] >= 5) {
        if (fread(n->blk, n->nfloat * sizeof(float), 1, fp) != 1) {
            fclose(fp); net_free(n); return -1;
        }
    } else if (hdr[5] >= 4) {
        /* v4: no interaction section -- load the prefix, seed V and W_g,
         * leave U zero so the function is unchanged */
        size_t prefix = (n->nfloat - xsec) * sizeof(float);
        if (fread(n->blk, prefix, 1, fp) != 1) { fclose(fp); net_free(n); return -1; }
        net_init_xhead(n, 0x58484541ULL);
    } else {
        /* older file without the belief head: load the prefix, init the rest */
        size_t belief = (size_t)NCARD * n->h2 + NCARD;
        size_t prefix = (n->nfloat - belief - xsec) * sizeof(float);
        if (fread(n->blk, prefix, 1, fp) != 1) { fclose(fp); net_free(n); return -1; }
        net_init_belief(n, 0xBE11EFULL);
        net_init_xhead(n, 0x58484541ULL);
    }
    fclose(fp);
    return 0;
}
