/* mine -- class-conditioned error mining with trustworthy labels.
 *
 * Expert review of analysed games found the policy's mistakes cluster into
 * mechanically detectable classes: playing a card that forecloses obtainable
 * value while a zero-skip play exists, refusing a pile top it could profitably
 * play, stalling with plenty of its own plays left, discarding onto a pile
 * top it wants, and hedging onto dominated discards.  Blanket expert
 * iteration paid more in blurred sharpness than the fixed leaks returned, so
 * this tool builds a *corrections-only* corpus instead: cheap sampled
 * self-play visits states, the detectors flag class instances, and only
 * flagged states are labeled -- by the estimator configuration the review
 * validated (sampled playouts against determinism bias, dominated-discard
 * pruning, draw-variant expansion, the points-and-significance override
 * gates).  A state becomes a training sample only when that search actually
 * disagrees with the policy's argmax; training mixes these with plain
 * self-policy anchor games so the rest of the policy stays put.
 *
 *   ./bin/mine --net data/best.bin --games 400 --threads 4 \
 *              --out data/corr.smp [--dets 256] [--dup 4]
 *
 * --confirm IN OUT: independent confirmation pass over a mined corpus --
 * every correction is re-labeled on a fresh world stream and kept only if
 * the fresh run makes the same gated choice with a lead >= --cfloor points
 * over candidate 0 (see confirm_mode).
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/net.h"
#include "../src/search.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI_K 12
#define SMP_MAGIC 0x4C435344u

typedef struct {
    State st;
    float target;
    uint8_t persp;
    uint8_t npi;
    uint16_t pmv[PI_K];
    float ppr[PI_K];
} Sample;

/* can player p legally play card c right now? */
static int playable(const State *st, int p, int c)
{
    int s = CARD_SUIT(c);
    return CARD_IS_WAGER(c) ? (st->exp_top[p][s] == 0)
                            : (CARD_VALUE(c) > st->exp_top[p][s]);
}

/* values a play of c would skip that the mover cannot rule out obtaining:
 * neither played nor discarded (own-hand skips count -- playing over your
 * own lower card is exactly the misorder being mined) */
static int skip_cost(const State *st, int c)
{
    if (CARD_IS_WAGER(c)) return 0;
    int p = st->turn, s = CARD_SUIT(c);
    int n = 0;
    uint64_t gone = st->played[0] | st->played[1] | st->discarded;
    for (int v = st->exp_top[p][s] + 1; v < CARD_VALUE(c); v++) {
        int id = s * NRANK + v + 1;
        if (!((gone >> id) & 1ULL)) n++;
    }
    return n;
}

/* class detectors over the policy distribution; returns a bitmask */
static int detect(const State *st, const Move *mv, const float *pr, int n)
{
    const int p = st->turn;
    int cls = 0;
    int top = 0;
    for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;

    /* playable-now cards in hand, and the best zero-skip play's prob mass */
    uint8_t hc[HAND_SIZE];
    int nh = lc_hand_cards(st, p, hc);
    int nplay = 0, zero_skip_exists = 0;
    for (int i = 0; i < nh; i++)
        if (playable(st, p, hc[i])) {
            nplay++;
            if (!CARD_IS_WAGER(hc[i]) && CARD_VALUE(hc[i]) >= 2 && skip_cost(st, hc[i]) == 0)
                zero_skip_exists = 1;
        }

    /* 1: top move plays over obtainable value while a zero-skip play exists */
    if (!mv[top].discard && skip_cost(st, mv[top].card) >= 1 && zero_skip_exists)
        cls |= 1;

    /* 2: a pile top is a playable number WORTH TAKING for the mover (decent
     * value or a wagered expedition), yet almost no policy mass draws it */
    for (int s = 0; s < NSUIT; s++) {
        if (st->pile_n[s] == 0) continue;
        int t = st->pile[s][st->pile_n[s] - 1];
        if (CARD_IS_WAGER(t) || !playable(st, p, t)) continue;
        if (CARD_VALUE(t) < 4 && st->exp_wager[p][s] == 0) continue;
        float mass = 0.0f;
        for (int i = 0; i < n; i++) if (mv[i].draw == s + 1) mass += pr[i];
        if (mass < 0.05f) { cls |= 2; break; }
    }

    /* 3: top move stalls (pile-draws a card the mover cannot play) with
     * plenty of own plays and deck left */
    if (mv[top].draw != 0 && (st->deck_left >= 8 ? nplay >= 3 : nplay >= 2)) {
        int t = st->pile[mv[top].draw - 1][st->pile_n[mv[top].draw - 1] - 1];
        if (!playable(st, p, t)) cls |= 4;
    }

    /* 4: top move discards onto a pile whose current top the mover can play
     * -- burying its own draw option */
    if (mv[top].discard) {
        int s = CARD_SUIT(mv[top].card);
        if (st->pile_n[s] > 0 && playable(st, p, st->pile[s][st->pile_n[s] - 1]))
            cls |= 8;
    }

    /* 5: visible probability on a dominated discard */
    uint64_t dead = lc_dead_cards(st);
    if (dead & st->hand[p]) {
        for (int i = 0; i < n; i++)
            if (pr[i] >= 0.01f && lc_discard_dominated(st, mv[i], dead)) { cls |= 16; break; }
    }

    /* 6: out-of-order same-expedition play -- the top move plays card B
     * while a LOWER playable card A of the same suit sits in hand.  B-first
     * permanently kills A (ascending only); A-first keeps both.  Skipping A
     * to save a turn is sometimes right, which is exactly why these states
     * need a searched label instead of a habit (observed: G7 over held G6
     * at 5 deck stranded 6 points and flipped a 133-131 match). */
    if (!mv[top].discard && !CARD_IS_WAGER(mv[top].card)) {
        int bs = CARD_SUIT(mv[top].card), bv = CARD_VALUE(mv[top].card);
        for (int i = 0; i < nh; i++) {
            int c = hc[i];
            if (CARD_IS_WAGER(c) || CARD_SUIT(c) != bs || c == mv[top].card) continue;
            if (CARD_VALUE(c) < bv && playable(st, p, c)) { cls |= 32; break; }
        }
    }

    /* 7: deck burn while turn-constrained with a free pile extension --
     * the mover has more guaranteed plays than remaining turns, some pile
     * top is dead to BOTH sides (drawing it costs nothing), yet the top
     * move burns a deck card, shortening the very round the mover needs
     * lengthened (observed at 10 deck: q +12.5 for the free extension vs
     * +7.2 for the deck burn, and the deck line lost the match). */
    if (mv[top].draw == 0 && st->deck_left <= 14) {
        int turns_left = (st->deck_left + 1) / 2;
        if (nplay > turns_left) {
            for (int s = 0; s < NSUIT; s++) {
                if (st->pile_n[s] == 0) continue;
                if ((dead >> st->pile[s][st->pile_n[s] - 1]) & 1ULL) { cls |= 64; break; }
            }
        }
    }
    /* 8: wager-clutch -- the mover holds a wager the opponent provably
     * cannot play (a number is down on their suit expedition) while the
     * mover's own expedition in the suit is empty and their holding is
     * thin, yet the policy puts almost no mass on the safe unload.
     * The 2026-08-25 review flagged this five consecutive turns at ~1%
     * prior on the correct discard; the label decides keep-vs-unload. */
    if (st->deck_left >= 6) {
        for (int i = 0; i < nh; i++) {
            int c = hc[i];
            if (!CARD_IS_WAGER(c)) continue;
            int s = CARD_SUIT(c);
            if (st->exp_n[p][s] != 0) continue;
            if (st->exp_top[p ^ 1][s] == 0) continue;
            int own = 0;
            for (int k = 0; k < nh; k++)
                if (!CARD_IS_WAGER(hc[k]) && CARD_SUIT(hc[k]) == s) own++;
            if (own > 2) continue;
            float mass = 0.0f;
            for (int k = 0; k < n; k++)
                if (mv[k].discard && CARD_IS_WAGER(mv[k].card) &&
                    CARD_SUIT(mv[k].card) == s) mass += pr[k];
            if (mass < 0.05f) { cls |= 128; break; }
        }
    }
    /* 9: gift -- the top move discards a card the opponent can play right
     * now AND plausibly wants (a wager, a card onto a wagered expedition,
     * or value >= 5).  The c17 cycle taught the safe wager unload without
     * a counter-signal and the gift probe regressed to 0/20; this is the
     * missing other side of the wager/number discard boundary, and the
     * label decides case by case (gifts are sometimes right, per the
     * reviewer -- category bans stay out). */
    if (mv[top].discard) {
        int c = mv[top].card, s = CARD_SUIT(c);
        if (playable(st, p ^ 1, c) &&
            (CARD_IS_WAGER(c) || st->exp_wager[p ^ 1][s] > 0 || CARD_VALUE(c) >= 5))
            cls |= 256;
    }
    return cls;
}

typedef struct {
    const Net *net;
    const Net *net_b;   /* optional belief specialist for labeler worlds */
    int games, thread, nthread, dets, dup, solvedeck;
    float selk;         /* labeler selection gate (0 = ungated argmax) */
    float pfloor;       /* a correction must lead the policy top by this many points */
    int sym;            /* estimator-matched labeler: sym_k/sym_bel 120, calibrated sampler */
    int winq;           /* win-aware last-round selection in the labeler */
    uint64_t seed;
    FILE *out, *log;
    pthread_mutex_t *lk;
    long flagged, corrected, hedged, plies, written;
    long bycls[9];
} Job;

/* the validated labeler (gens 6-7), shared by the miner and --confirm */
static void labeler_init(Agent *lab, const Job *j)
{
    agent_default(lab, AG_ROLLOUT, j->net);
    /* belief-improved labeling (gen-6): the specialist's sharper hand
     * inference steers the labeler's world sampling, same division of
     * labor as the adopted rollouth match spec -- policy, priors and
     * playouts stay the champion's */
    lab->net_b = j->net_b;
    lab->dets = j->dets;
    lab->root_width = 5;
    lab->gate = 0.0f;
    lab->eval_cand = 4;
    lab->override_k = 3.0f;      /* override_min 4 and prune_dom on by default */
    lab->playout_sample = 1;
    /* exact endgame labels: at deck_left <= solvedeck the labeler solves
     * belief worlds instead of estimating them -- vote mode, one root
     * solve per world (grant the solver a real budget via LC_SOLVE_BUDGET
     * or it falls back to search immediately) */
    lab->solve_deck = j->solvedeck;
    lab->solve_vote = j->solvedeck > 0;
    /* gen-7 labeler (panel 2, rank 3): the label is the move the deployed
     * selection rule would play -- the sel_k paired-SE gate protects the
     * policy top, priors and beliefs are exactly symmetrized and worlds
     * come from the calibrated sampler, as in the match spec */
    lab->sel_k = j->selk;
    if (j->sym) { lab->sym_k = 120; lab->sym_bel = 120; lab->bel_samp = 1; lab->sym_play = 1; }
    /* gen-8: win-aware labels -- in the last round candidates compare on
     * win probability (the engine's win_q path), the match objective */
    lab->win_q = j->winq;
}

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    Rng rng;
    rng_seed(&rng, j->seed + 77777ULL * (uint64_t)(j->thread + 1));

    Agent lab;
    labeler_init(&lab, j);

    Features f;
    for (int g = j->thread; g < j->games; g += j->nthread) {
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < MATCH_ROUNDS; rd++) {
            State st;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)(cum[0] > 320 ? 320 : (cum[0] < -320 ? -320 : cum[0]));
            st.cum[1] = (int16_t)(cum[1] > 320 ? 320 : (cum[1] < -320 ? -320 : cum[1]));
            st.turn = (uint8_t)(rd & 1);
            while (!st.over) {
                Move mv[MAX_MOVES];
                float pr[MAX_MOVES];
                int n = policy_probs(j->net, &st, mv, pr, NULL);
                if (n <= 0) break;
                j->plies++;

                int cls = detect(&st, mv, pr, n);
                /* the dominant class must not monopolize the corpus: a
                 * corpus that only ever says "take the pile" teaches the
                 * draw head a direction, not a judgment (measured: ply
                 * inflation 143->161 and a 15-point h2h collapse) */
                if (cls == 2 && (rng_next(&rng) & 1)) cls = 0;
                if (cls) {
                    j->flagged++;
                    SearchStats ss;
                    memset(&ss, 0, sizeof ss);
                    float sval = 0.0f;
                    Move lm = rollout_move(&lab, &st, &rng, &sval, &ss);
                    /* the policy's preferred ACTION: fold identical wager
                     * copies first (their split mass otherwise loses the
                     * argmax), and compare modulo the copy isomorphism --
                     * "played the other Yx" is agreement, not a correction
                     * worth 4x dup weight */
                    Move fmv[MAX_MOVES];
                    float fpr[MAX_MOVES];
                    memcpy(fmv, mv, sizeof(Move) * (size_t)n);
                    memcpy(fpr, pr, sizeof(float) * (size_t)n);
                    int fn = lc_dedup_wagers(&st, fmv, fpr, n, 1);
                    int top = 0;
                    for (int i = 1; i < fn; i++) if (fpr[i] > fpr[top]) top = i;
                    int same_card = lm.card == fmv[top].card ||
                                    (CARD_IS_WAGER(lm.card) && CARD_IS_WAGER(fmv[top].card) &&
                                     CARD_SUIT(lm.card) == CARD_SUIT(fmv[top].card));
                    int agree = same_card && lm.discard == fmv[top].discard &&
                                lm.draw == fmv[top].draw;
                    /* with a symmetrized labeler the reference is ITS candidate 0
                     * (the symmetrized policy top after dedup and pruning), not
                     * the raw argmax: a label that merely restates the
                     * symmetrized top is a confirmation, not a search-driven
                     * correction (measured: 24% of the ungated corrections
                     * carried a zero lead for exactly this reason) */
                    if (j->sym && ss.n >= 1) {
                        Move c0 = ss.mv[0];
                        int sc = lm.card == c0.card ||
                                 (CARD_IS_WAGER(lm.card) && CARD_IS_WAGER(c0.card) &&
                                  CARD_SUIT(lm.card) == CARD_SUIT(c0.card));
                        agree = sc && lm.discard == c0.discard && lm.draw == c0.draw;
                    }
                    /* paired lead of the labeled move over the policy top,
                     * from the labeler's own stats (candidate 0 is the
                     * policy top after dedup and pruning) */
                    double dm = 0.0, dse = 0.0;
                    int kc = -1, k0 = 0;
                    for (int c = 0; c < ss.n; c++) {
                        if (ss.mv[c].card == lm.card && ss.mv[c].discard == lm.discard && ss.mv[c].draw == lm.draw) kc = c;
                    }
                    if (kc >= 0) { dm = ss.q[kc] - ss.q[k0]; dse = ss.se[kc] > ss.se[k0] ? ss.se[kc] : ss.se[k0]; }
                    int hedge = !agree && j->pfloor > 0.0f && dm < j->pfloor;
                    if (j->log) {
                        pthread_mutex_lock(j->lk);
                        fprintf(j->log, "%d %d %d %d %.2f %.2f %d\n", (int)st.nply, (int)st.deck_left, cls, agree ? 1 : (hedge ? 2 : 0), dm, dse, ss.n);
                        pthread_mutex_unlock(j->lk);
                    }
                    if (hedge) j->hedged++;
                    /* corrections AND confirmations: the same flagged state
                     * where the search agrees with the policy is the
                     * counterweight that keeps a class from becoming a
                     * direction */
                    if (!hedge) {
                        if (!agree) {
                            j->corrected++;
                            for (int b = 0; b < 9; b++) if (cls & (1 << b)) j->bycls[b]++;
                        }
                        Sample s;
                        memset(&s, 0, sizeof s);
                        s.st = st;
                        s.persp = st.turn;
                        feat_extract(&st, st.turn, &f);
                        s.target = net_value(j->net, &f) * VAL_SCALE; /* zero value grad */
                        /* the target is the GATED choice -- the move the
                         * validated selection rule actually plays.  The
                         * first corpus generation softmaxed raw Q over all
                         * evaluated candidates, whose argmax is the ungated
                         * Q-maximum: exactly the estimator measured harmful
                         * (min_cand, 42.8%), and the reason five fine-tune
                         * variants collapsed while self-target and no-data
                         * controls sat at parity. */
                        s.pmv[0] = MOVE_PACK(lm);
                        s.ppr[0] = 1.0f;
                        s.npi = 1;
                        pthread_mutex_lock(j->lk);
                        int reps = agree ? 1 : j->dup;
                        for (int r = 0; r < reps; r++)
                            fwrite(&s, sizeof s, 1, j->out);
                        j->written += reps;
                        pthread_mutex_unlock(j->lk);
                    }
                }

                /* diverse traversal: sample the policy early, argmax later */
                int pick;
                if (st.nply < 30) pick = sample_index(pr, n, &rng);
                else { pick = 0; for (int i = 1; i < n; i++) if (pr[i] > pr[pick]) pick = i; }
                lc_apply(&st, mv[pick]);
            }
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
        }
    }
    return NULL;
}

/* --filter IN OUT: keep only samples whose labeled action (card + play/
 * discard) differs from the net's argmax action.  Draw-only corrections are
 * dropped: the 6-logit factored draw head turns any net draw-direction
 * pressure into a global reweighting (measured as ply inflation and a
 * 17-point collapse), while action corrections are state-specific. */
static int filter_mode(const Net *net, const char *inp, const char *outp)
{
    FILE *fi = fopen(inp, "rb");
    if (!fi) { fprintf(stderr, "mine: cannot open %s\n", inp); return 1; }
    uint32_t h[4];
    uint64_t cnt;
    if (fread(h, sizeof h, 1, fi) != 1 || fread(&cnt, sizeof cnt, 1, fi) != 1 ||
        h[0] != SMP_MAGIC || h[1] != sizeof(Sample)) { fprintf(stderr, "mine: bad file\n"); return 1; }
    FILE *fo = fopen(outp, "wb");
    if (!fo) { fprintf(stderr, "mine: cannot open %s\n", outp); return 1; }
    uint64_t kept = 0;
    fwrite(h, sizeof h, 1, fo);
    fwrite(&kept, sizeof kept, 1, fo);
    Sample s;
    while (fread(&s, sizeof s, 1, fi) == 1) {
        Move mv[MAX_MOVES];
        float pr[MAX_MOVES];
        int n = policy_probs(net, &s.st, mv, pr, NULL);
        int top = 0;
        for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;
        int tb = 0;
        for (int i = 1; i < s.npi; i++) if (s.ppr[i] > s.ppr[tb]) tb = i;
        int tcard = s.pmv[tb] % 60, tdisc = (s.pmv[tb] / 60) % 2;
        if (tcard != mv[top].card || tdisc != mv[top].discard) {
            fwrite(&s, sizeof s, 1, fo);
            kept++;
        }
    }
    fseek(fo, sizeof h, SEEK_SET);
    fwrite(&kept, sizeof kept, 1, fo);
    fclose(fi); fclose(fo);
    printf("mine --filter: kept %llu action-level corrections of %llu\n",
           (unsigned long long)kept, (unsigned long long)cnt);
    return 0;
}

/* --selftarget IN OUT: same states, targets replaced by the net's own
 * policy top-k.  A structurally identical but semantically null corpus:
 * if training on THIS collapses, the sample pipeline is broken; if it is
 * inert, the labels are the poison. */
static int selftarget_mode(const Net *net, const char *inp, const char *outp)
{
    FILE *fi = fopen(inp, "rb");
    FILE *fo = fopen(outp, "wb");
    if (!fi || !fo) { fprintf(stderr, "mine: open failed\n"); return 1; }
    uint32_t h[4]; uint64_t cnt;
    if (fread(h, sizeof h, 1, fi) != 1 || fread(&cnt, sizeof cnt, 1, fi) != 1) return 1;
    fwrite(h, sizeof h, 1, fo); fwrite(&cnt, sizeof cnt, 1, fo);
    Sample s;
    while (fread(&s, sizeof s, 1, fi) == 1) {
        Move mv[MAX_MOVES]; float pr[MAX_MOVES];
        int n = policy_probs(net, &s.st, mv, pr, NULL);
        int idx[MAX_MOVES];
        for (int i = 0; i < n; i++) idx[i] = i;
        int k = n < PI_K ? n : PI_K;
        for (int i = 0; i < k; i++) {
            int b = i;
            for (int j2 = i + 1; j2 < n; j2++) if (pr[idx[j2]] > pr[idx[b]]) b = j2;
            int t = idx[i]; idx[i] = idx[b]; idx[b] = t;
        }
        float tot = 0.0f;
        for (int i = 0; i < k; i++) tot += pr[idx[i]];
        if (tot <= 0.0f) tot = 1.0f;
        for (int i = 0; i < k; i++) { s.pmv[i] = MOVE_PACK(mv[idx[i]]); s.ppr[i] = pr[idx[i]] / tot; }
        for (int i = k; i < PI_K; i++) { s.pmv[i] = 0; s.ppr[i] = 0.0f; }
        s.npi = (uint8_t)k;
        fwrite(&s, sizeof s, 1, fo);
    }
    fclose(fi); fclose(fo);
    printf("mine --selftarget: rewrote %llu samples\n", (unsigned long long)cnt);
    return 0;
}

/* same move modulo the wager-copy isomorphism ("played the other Yx") */
static int move_iso(Move a, Move b)
{
    int sc = a.card == b.card ||
             (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card) &&
              CARD_SUIT(a.card) == CARD_SUIT(b.card));
    return sc && a.discard == b.discard && a.draw == b.draw;
}

/* the labeler's candidate 0 for a stored state, recomputed without a
 * world sweep: with --sym the exactly symmetrized policy top after the
 * wager-copy fold and dominated-discard pruning, formed as rollout_move
 * forms it (the symmetrization is state-seeded, so this IS the candidate
 * 0 the miner compared against); without --sym the raw folded argmax the
 * pre-sym miner used as its reference */
static Move labeler_ref(const Agent *lab, int sym, const State *st, Rng *rng)
{
    Move mv[MAX_MOVES];
    float pr[MAX_MOVES];
    int n = agent_policy_probs(lab, st, rng, mv, pr, NULL);
    if (n > 1) n = lc_dedup_wagers(st, mv, pr, n, 1);
    if (sym && lab->prune_dom && n > 1) {
        uint64_t dead = lc_dead_cards(st);
        if (dead & st->hand[st->turn]) {
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (lc_discard_dominated(st, mv[i], dead)) continue;
                mv[k] = mv[i]; pr[k] = pr[i]; k++;
            }
            if (k > 0) n = k;
        }
    }
    int top = 0;
    for (int i = 1; i < n; i++) if (pr[i] > pr[top]) top = i;
    return mv[top];
}

/* lead of move m over candidate 0 in a labeler's stats (dse = the larger
 * of the two SEs, as the mining log); 0 if m was not evaluated */
static int label_lead(const SearchStats *ss, Move m, double *dm, double *dse)
{
    *dm = 0.0; *dse = 0.0;
    for (int c = 0; c < ss->n; c++)
        if (move_iso(ss->mv[c], m)) {
            *dm = ss->q[c] - ss->q[0];
            *dse = ss->se[c] > ss->se[0] ? ss->se[c] : ss->se[0];
            return 1;
        }
    return 0;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* --confirm IN OUT: independent confirmation of a mined corpus's
 * corrections.  The miner writes a correction when ONE labeler run's gated
 * choice differs from candidate 0 and leads it by the points floor; that
 * lead is the best of several candidates on one 256-world batch, so it is
 * inflated by selection (the rank-4 pre-test relabeled 50 such qualifiers
 * on fresh worlds: half flipped back to candidate 0, only 36% re-cleared
 * the floor).  This pass re-runs the SAME labeler on every correction's
 * state with an independent world stream and keeps the sample only if the
 * fresh run makes the same gated choice (wager-copy isomorphism) AND the
 * stored move's fresh lead over candidate 0 is >= cfloor points.
 * Everything that is not a correction -- confirmations (one-hot on
 * candidate 0), anchor games, soft targets -- passes through unchanged,
 * and the dup copies of a correction (identical consecutive samples) are
 * judged once.  Log line per distinct correction:
 * "nply deck kept dm_fresh dse_fresh". */
static int confirm_mode(const Job *j, float cfloor, const char *inp, const char *outp)
{
    FILE *fi = fopen(inp, "rb");
    if (!fi) { fprintf(stderr, "mine: cannot open %s\n", inp); return 1; }
    uint32_t h[4];
    uint64_t cnt;
    if (fread(h, sizeof h, 1, fi) != 1 || fread(&cnt, sizeof cnt, 1, fi) != 1 ||
        h[0] != SMP_MAGIC || h[1] != sizeof(Sample)) { fprintf(stderr, "mine: bad file\n"); return 1; }
    FILE *fo = fopen(outp, "wb");
    if (!fo) { fprintf(stderr, "mine: cannot open %s\n", outp); return 1; }
    uint64_t written = 0;
    fwrite(h, sizeof h, 1, fo);
    fwrite(&written, sizeof written, 1, fo);

    Agent lab;
    labeler_init(&lab, j);
    Rng rng;
    Sample s, prev;
    int have_prev = 0, prev_corr = 0, prev_keep = 1;
    long idx = 0, seen = 0, kept = 0, flip = 0, floor = 0, refmis = 0, notfound = 0;
    double *lead_all = (double *)malloc(sizeof(double) * (size_t)(cnt ? cnt : 1));
    double *lead_kept = (double *)malloc(sizeof(double) * (size_t)(cnt ? cnt : 1));
    while (fread(&s, sizeof s, 1, fi) == 1) {
        int corr = 0, keep = 1;
        if (have_prev && memcmp(&s, &prev, sizeof s) == 0) {
            corr = prev_corr; keep = prev_keep;          /* a dup copy: same verdict */
        } else {
            if (s.npi == 1 && s.ppr[0] >= 0.999f) {
                Move tgt;
                tgt.card = MOVE_CARD(s.pmv[0]);
                tgt.discard = MOVE_DISC(s.pmv[0]);
                tgt.draw = MOVE_DRAW(s.pmv[0]);
                /* independent stream per correction: the seed mixed with
                 * the sample's index, never the miner's game stream */
                rng_seed(&rng, j->seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(idx + 1));
                Move ref = labeler_ref(&lab, j->sym, &s.st, &rng);
                if (!move_iso(tgt, ref)) {
                    corr = 1;
                    SearchStats ss;
                    memset(&ss, 0, sizeof ss);
                    float sv = 0.0f;
                    Move lm = rollout_move(&lab, &s.st, &rng, &sv, &ss);
                    if (ss.n >= 1 && !move_iso(ss.mv[0], ref)) refmis++;
                    double dm, dse;
                    if (!label_lead(&ss, tgt, &dm, &dse)) notfound++;
                    lead_all[seen++] = dm;
                    if (!move_iso(lm, tgt)) { keep = 0; flip++; }
                    else if (dm < cfloor) { keep = 0; floor++; }
                    else { lead_kept[kept++] = dm; }
                    if (j->log)
                        fprintf(j->log, "%d %d %d %.2f %.2f\n", (int)s.st.nply, (int)s.st.deck_left, keep, dm, dse);
                }
            }
            prev = s; have_prev = 1; prev_corr = corr; prev_keep = keep;
        }
        if (!corr || keep) { fwrite(&s, sizeof s, 1, fo); written++; }
        idx++;
    }
    fseek(fo, sizeof h, SEEK_SET);
    fwrite(&written, sizeof written, 1, fo);
    fclose(fi); fclose(fo);
    if (j->log) fflush(j->log);

    printf("mine --confirm: %ld distinct corrections seen, %ld confirmed (%.1f%%), %ld dropped by flip (fresh gated choice differs), %ld dropped by floor (same choice, fresh lead < %.1f)\n",
           seen, kept, 100.0 * kept / (seen ? seen : 1), flip, floor, cfloor);
    for (int pass = 0; pass < 2; pass++) {
        double *v = pass ? lead_kept : lead_all;
        long n = pass ? kept : seen;
        if (n <= 0) continue;
        qsort(v, (size_t)n, sizeof(double), cmp_double);
        long ge2 = 0, ge4 = 0;
        for (long i = 0; i < n; i++) { if (v[i] >= 2.0) ge2++; if (v[i] >= 4.0) ge4++; }
        double med = n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        printf("      fresh lead of the stored move over candidate 0, %s (%ld): median %.2f, share >= 2: %.0f%%, share >= 4: %.0f%%\n",
               pass ? "confirmed set" : "all corrections", n, med, 100.0 * ge2 / n, 100.0 * ge4 / n);
    }
    if (refmis || notfound)
        printf("      NOTE: recomputed reference != fresh candidate 0 on %ld corrections; stored move absent from the fresh candidates on %ld\n",
               refmis, notfound);
    printf("      wrote %llu of %llu samples to %s\n",
           (unsigned long long)written, (unsigned long long)cnt, outp);
    free(lead_all); free(lead_kept);
    return 0;
}

int main(int argc, char **argv)
{
    const char *netpath = "data/best.bin", *outpath = "data/corr.smp";
    const char *beliefpath = NULL;
    const char *filter_in = NULL, *filter_out = NULL, *self_in = NULL, *self_out = NULL;
    const char *confirm_in = NULL, *confirm_out = NULL;
    int games = 200, nthread = 4, dets = 256, dup = 4, solvedeck = 0, sym = 0, winq = 0;
    float selk = 0.0f, pfloor = 0.0f, cfloor = 2.0f;
    const char *logpath = NULL;
    uint64_t seed = 20260729;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--net") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outpath = argv[++i];
        else if (!strcmp(argv[i], "--games") && i + 1 < argc) games = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthread = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dets") && i + 1 < argc) dets = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dup") && i + 1 < argc) dup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--solvedeck") && i + 1 < argc) solvedeck = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--belief") && i + 1 < argc) beliefpath = argv[++i];
        else if (!strcmp(argv[i], "--selk") && i + 1 < argc) selk = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--pfloor") && i + 1 < argc) pfloor = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sym")) sym = 1;
        else if (!strcmp(argv[i], "--winq")) winq = 1;
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) logpath = argv[++i];
        else if (!strcmp(argv[i], "--filter") && i + 2 < argc) { filter_in = argv[++i]; filter_out = argv[++i]; }
        else if (!strcmp(argv[i], "--selftarget") && i + 2 < argc) { self_in = argv[++i]; self_out = argv[++i]; }
        else if (!strcmp(argv[i], "--confirm") && i + 2 < argc) { confirm_in = argv[++i]; confirm_out = argv[++i]; }
        else if (!strcmp(argv[i], "--cfloor") && i + 1 < argc) cfloor = (float)atof(argv[++i]);
        else { fprintf(stderr, "usage: %s [--net N] [--out F] [--games G] [--threads T] [--dets D] [--dup K]\n"
                               "       [--belief B] [--selk K] [--pfloor P] [--sym] [--log F] [--solvedeck D]\n"
                               "       [--filter IN OUT] [--selftarget IN OUT] [--confirm IN OUT [--cfloor P]]\n", argv[0]); return 1; }
    }
    Net *net = (Net *)malloc(sizeof(Net));
    if (!net || net_load(net, netpath)) { fprintf(stderr, "mine: cannot load %s\n", netpath); return 1; }
    Net *net_b = NULL;
    if (beliefpath) {
        net_b = (Net *)malloc(sizeof(Net));
        if (!net_b || net_load(net_b, beliefpath)) { fprintf(stderr, "mine: cannot load %s\n", beliefpath); return 1; }
    }
    if (filter_in) return filter_mode(net, filter_in, filter_out);
    if (self_in) return selftarget_mode(net, self_in, self_out);
    if (confirm_in) {
        /* the same labeler configuration the corpus was mined with */
        Job cj;
        memset(&cj, 0, sizeof cj);
        cj.net = net; cj.net_b = net_b; cj.dets = dets; cj.solvedeck = solvedeck;
        cj.selk = selk; cj.pfloor = pfloor; cj.sym = sym; cj.seed = seed;
        cj.log = logpath ? fopen(logpath, "w") : NULL;
        int rc = confirm_mode(&cj, cfloor, confirm_in, confirm_out);
        if (cj.log) fclose(cj.log);
        return rc;
    }

    FILE *out = fopen(outpath, "wb");
    if (!out) { fprintf(stderr, "mine: cannot open %s\n", outpath); return 1; }
    uint32_t h[4] = { SMP_MAGIC, sizeof(Sample), PI_K, 0 };
    uint64_t count = 0;
    fwrite(h, sizeof h, 1, out);
    fwrite(&count, sizeof count, 1, out);

    FILE *logf = logpath ? fopen(logpath, "w") : NULL;
    pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
    Job jobs[64];
    pthread_t th[64];
    if (nthread > 64) nthread = 64;
    for (int i = 0; i < nthread; i++) {
        memset(&jobs[i], 0, sizeof(Job));
        jobs[i].net = net; jobs[i].net_b = net_b;
        jobs[i].games = games; jobs[i].thread = i;
        jobs[i].nthread = nthread; jobs[i].dets = dets; jobs[i].dup = dup;
        jobs[i].solvedeck = solvedeck;
        jobs[i].selk = selk; jobs[i].pfloor = pfloor; jobs[i].sym = sym; jobs[i].winq = winq; jobs[i].log = logf;
        jobs[i].seed = seed; jobs[i].out = out; jobs[i].lk = &lk;
        pthread_create(&th[i], NULL, worker, &jobs[i]);
    }
    long fl = 0, co = 0, pl = 0, bc[9] = { 0 };
    for (int i = 0; i < nthread; i++) {
        pthread_join(th[i], NULL);
        fl += jobs[i].flagged; co += jobs[i].corrected; pl += jobs[i].plies;
        for (int b = 0; b < 9; b++) bc[b] += jobs[i].bycls[b];
    }
    long wr = 0;
    for (int i = 0; i < nthread; i++) wr += jobs[i].written;
    count = (uint64_t)wr;
    fseek(out, sizeof h, SEEK_SET);
    fwrite(&count, sizeof count, 1, out);
    fclose(out);
    long hd = 0;
    for (int i = 0; i < nthread; i++) hd += jobs[i].hedged;
    printf("mine: %d games, %ld plies, %ld flagged (%.1f%%), %ld corrected (%.1f%% of flagged), %ld hedged (gated lead under the points floor, not written)\n",
           games, pl, fl, 100.0 * fl / (pl ? pl : 1), co, 100.0 * co / (fl ? fl : 1), hd);
    printf("      by class: skip %ld, pile-refusal %ld, stall %ld, burial %ld, hedge %ld, misorder %ld, deckburn %ld, wclutch %ld, gift %ld\n",
           bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], bc[6], bc[7], bc[8]);
    printf("      wrote %llu samples (dup %d) to %s\n",
           (unsigned long long)count, dup, outpath);
    return 0;
}
