// backtest/kernels.c — boucles chaudes de la stratégie (compilées automatiquement par native.py avec gcc).
// Tout est écrit pour une logique BAISSIÈRE (short). Le cas haussier = miroir : prix inversés (o,h,l,c -> -o,-l,-h,-c).
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

// ---------------------------------------------------------------- MSS baissier après sweep
// H1 = sommet confirmé (k bougies plus basses de chaque côté), H2 = plus haut qui dépasse H1 (sweep, peut être provisoire),
// L2 = plus bas (mèche comprise) entre H1 et H2 (bougies h1i+1 .. h2i), cassure = bougie BAISSIÈRE qui CLÔTURE sous L2,
// cassure en <= ratio x distance (en bougies) : spmode 0 = entre les 2 sommets (h2i-h1i), spmode 1 = entre L2 et H2 (h2i-li).
// Correction minimale : si (H1 - L2) < max(min_corr_pct % de H1, min_corr_atr x ATR), on remonte vers un H1 PLUS ANCIEN
// (pivot encore balayé par H2, au plus max_back pivots en arrière) dont la correction H1 -> L2 est assez grande ; sinon pas de MSS.
static void l2_of(int h1i, int h2i, int trough, const double *l, const int *low_i, const double *low_p, int nlw, double *cL, int *cLi) {
    if (trough) {
        int lo = 0; while (lo < nlw && low_i[lo] <= h1i) lo++;
        int hi = lo; while (hi < nlw && low_i[hi] < h2i) hi++;
        if (hi <= lo) { *cL = NAN; *cLi = -1; }
        else { int j = lo; for (int q = lo + 1; q < hi; q++) if (low_p[q] < low_p[j]) j = q; *cL = low_p[j]; *cLi = low_i[j]; }
    } else {
        int j = h1i + 1;
        for (int q = h1i + 2; q <= h2i; q++) if (l[q] < l[j]) j = q;
        *cL = l[j]; *cLi = j;
    }
}

int mss_events(int n, const double *h, const double *l, const double *c, const double *o, int k, double ratio,
               int spmode, int trough_pivot, double min_sweep_pct, const double *atr, double min_corr_pct, double min_corr_atr,
               int max_back, signed char *ev, int *o_h1i, int *o_h2i, int *o_li, double *o_h1p, double *o_h2p, double *o_l2p) {
    int *piv_i = malloc(sizeof(int) * (n + 1)), *low_i = malloc(sizeof(int) * (n + 1));
    double *piv_p = malloc(sizeof(double) * (n + 1)), *low_p = malloc(sizeof(double) * (n + 1));
    int npv = 0, nlw = 0, runM_i = -1, last_h1 = -1, last_h2 = -1, ck_h1 = -2, ck_h2 = -2, ck_nl = -2, cLi = -1, cnt = 0;
    int fh1i = -1; double fh1p = NAN;
    double runM = -INFINITY, cL = NAN, ms = min_sweep_pct / 100.0;
    int corr_on = (min_corr_pct > 0 || min_corr_atr > 0);
    for (int t = 0; t < n; t++) {
        ev[t] = 0;
        int p = t - k;
        if (p >= k) {
            double lmin = l[p - k];
            for (int q = p - k + 1; q < p; q++) if (l[q] < lmin) lmin = l[q];
            double rmin = l[p + 1];
            for (int q = p + 2; q <= t; q++) if (l[q] < rmin) rmin = l[q];
            if (l[p] < lmin && l[p] <= rmin) { low_i[nlw] = p; low_p[nlw] = l[p]; nlw++; }
            double lmax = h[p - k];
            for (int q = p - k + 1; q < p; q++) if (h[q] > lmax) lmax = h[q];
            double rmax = h[p + 1];
            for (int q = p + 2; q <= t; q++) if (h[q] > rmax) rmax = h[q];
            if (h[p] > lmax && h[p] >= rmax) {
                piv_i[npv] = p; piv_p[npv] = h[p]; npv++;
                runM = h[p + 1]; runM_i = p + 1;
                for (int q = p + 2; q <= t; q++) if (h[q] > runM) { runM = h[q]; runM_i = q; }
            } else if (npv > 0 && h[t] > runM) { runM = h[t]; runM_i = t; }
        } else if (npv > 0 && h[t] > runM) { runM = h[t]; runM_i = t; }
        if (npv == 0) continue;
        int ia = piv_i[npv - 1], h1i, h2i, q1; double pa = piv_p[npv - 1], h1p, h2p;
        if (runM > pa && fabs(runM - pa) >= fabs(pa) * ms) { h1i = ia; h1p = pa; h2i = runM_i; h2p = runM; q1 = npv - 1; }
        else if (npv >= 2 && pa > piv_p[npv - 2] && fabs(pa - piv_p[npv - 2]) >= fabs(piv_p[npv - 2]) * ms) {
            h1i = piv_i[npv - 2]; h1p = piv_p[npv - 2]; h2i = ia; h2p = pa; q1 = npv - 2;
        } else continue;
        int nl_key = trough_pivot ? nlw : 0;
        if (h1i != ck_h1 || h2i != ck_h2 || nl_key != ck_nl) {
            ck_h1 = h1i; ck_h2 = h2i; ck_nl = nl_key;
            fh1i = h1i; fh1p = h1p;
            l2_of(fh1i, h2i, trough_pivot, l, low_i, low_p, nlw, &cL, &cLi);
            if (corr_on) {
                double a_ = isnan(atr[h2i]) ? 0.0 : atr[h2i];
                int q = q1, tries = 0;
                while (1) {
                    double thr = fmax(min_corr_pct / 100.0 * fabs(fh1p), min_corr_atr * a_);
                    if (!isnan(cL) && (fh1p - cL) >= thr) break;
                    q--; tries++;
                    while (q >= 0 && !(piv_p[q] < h2p && (h2p - piv_p[q]) >= fabs(piv_p[q]) * ms)) q--;
                    if (q < 0 || tries > max_back) { cL = NAN; break; }
                    fh1i = piv_i[q]; fh1p = piv_p[q];
                    l2_of(fh1i, h2i, trough_pivot, l, low_i, low_p, nlw, &cL, &cLi);
                }
            }
        }
        if (isnan(cL)) continue;
        if (t > h2i && c[t] < cL && !(fh1i == last_h1 && h2i == last_h2)) {
            last_h1 = fh1i; last_h2 = h2i;
            double ref = spmode == 0 ? (double)(h2i - fh1i) : (double)(h2i - cLi);
            if (c[t] < o[t] && (double)(t - h2i) <= ratio * ref) {
                ev[t] = 1; o_h1i[t] = fh1i; o_h2i[t] = h2i; o_li[t] = cLi; o_h1p[t] = fh1p; o_h2p[t] = h2p; o_l2p[t] = cL; cnt++;
            }
        }
    }
    free(piv_i); free(low_i); free(piv_p); free(low_p);
    return cnt;
}

// ---------------------------------------------------------------- mitigation des FVG par des bougies HTF CLÔTURÉES
// type 0 : zone baissière (offre), mitigée dès qu'une bougie atteint zlo + depth*(zhi-zlo) par le haut (depth 0 = simple touche)
// type 1 : zone haussière (demande), mitigée dès qu'une bougie atteint zhi - depth*(zhi-zlo) par le bas
// Seules comptent les bougies qui s'OUVRENT après la formation de la zone. out = heure de clôture de la bougie qui mitige (INT64_MAX = jamais).
void mitigate(int nz, const double *zlo, const double *zhi, const int64_t *form, int type, double depth, int nm,
              const double *mh, const double *ml, const int64_t *mopen, const int64_t *mclose, int64_t *out) {
    for (int z = 0; z < nz; z++) {
        int a = 0, b = nm;
        while (a < b) { int m = (a + b) / 2; if (mopen[m] < form[z]) a = m + 1; else b = m; }
        out[z] = INT64_MAX;
        double lvl_hi = zlo[z] + depth * (zhi[z] - zlo[z]), lvl_lo = zhi[z] - depth * (zhi[z] - zlo[z]);
        for (int m = a; m < nm; m++) {
            if ((type == 0 && mh[m] >= lvl_hi) || (type == 1 && ml[m] <= lvl_lo)) { out[z] = mclose[m]; break; }
        }
    }
}

// ---------------------------------------------------------------- événements MSS -> ordres limites
// Pour chaque MSS : (1) H2 doit être DANS une zone HTF non mitigée, (2) premier FVG baissier de la jambe H2 -> cassure,
// (3) entrée = niveau choisi du FVG, SL = au-dessus de H2 (+ marge ATR), (4) TP = RR fixe ou prochain FVG HTF non mitigé.
// P : 0 marge zone (ATR), 1 niveau d'entrée dans le FVG (0 = bord proche, 1 = bord lointain), 2 marge SL (ATR), 3 mode TP (0 RR, 1 FVG),
//     4 RR, 5 RR mini (mode FVG), 6 bord du FVG visé (0 proche .. 1 lointain), 7 RR maxi (plafond), 8 taille mini du FVG (ATR)
int build_signals(int ne, const int *ev_t, const int *ev_h1i, const int *ev_h2i, const double *ev_h1p, const double *ev_h2p,
                  int nc, const int64_t *cclose, const double *ch, const double *cl, const double *cc, const double *catr,
                  int np, const double *plo, const double *phi, const int64_t *pform, const int64_t *pmit,
                  int no, const double *olo, const double *ohi, const int64_t *oform, const int64_t *omit,
                  const double *P, signed char *ok, double *entry, double *sl, double *tp, int *zid) {
    int *act = malloc(sizeof(int) * (np + 1)), *acto = malloc(sizeof(int) * (no + 1));
    int na = 0, pp = 0, nao = 0, po = 0, cnt = 0;
    for (int e = 0; e < ne; e++) {
        ok[e] = 0;
        int t = ev_t[e], h1i = ev_h1i[e], h2i = ev_h2i[e]; int64_t T = cclose[t], T_h1 = cclose[h1i];
        double H1 = ev_h1p[e], H2 = ev_h2p[e], atr = catr[t];
        while (pp < np && pform[pp] <= T) act[na++] = pp++;
        if (P[3] > 0.5) {
            while (po < no && oform[po] <= T) acto[nao++] = po++;
            for (int a = 0; a < nao;) { if (omit[acto[a]] <= T) acto[a] = acto[--nao]; else a++; }
        }
        if (!(atr > 0)) continue;
        // H2 doit être le maximum entre H1 et la bougie de cassure t
        int max_ok = 1;
        for (int m = h1i; m <= t; m++) { if (ch[m] > H2 + 1e-12) { max_ok = 0; break; } }
        if (!max_ok) continue;
        int z = -1;
        // Une zone est valide si :
        // 1) formée AVANT H1 (pform[q] <= T_h1) et NON mitigée par une bougie 1h clôturée avant H1 (pmit[q] > T_h1)
        // 2) H2 est dans la FVG (plo[q] <= H2 <= phi[q] + P[0]*atr)
        // 3) H2 est le maximum absolu dans la FVG depuis sa formation jusqu'à la cassure t
        for (int a = 0; a < na; a++) {
            int q = act[a];
            if (pform[q] <= T_h1 && pmit[q] > T_h1 && plo[q] <= H2 && H2 <= phi[q] + P[0] * atr) {
                int lo_m = 0, hi_m = h1i;
                while (lo_m < hi_m) { int mid = (lo_m + hi_m) / 2; if (cclose[mid] <= pform[q]) lo_m = mid + 1; else hi_m = mid; }
                int z_max_ok = 1;
                for (int m = lo_m; m < h1i; m++) { if (ch[m] > H2 + 1e-12) { z_max_ok = 0; break; } }
                if (z_max_ok) { z = q; break; }
            }
        }
        if (z < 0) continue;
        int j = -1;
        for (int q = h2i; q + 2 <= t; q++) if (ch[q + 2] < cl[q] && (cl[q] - ch[q + 2]) >= P[8] * atr) { j = q; break; }
        if (j < 0) continue;
        double zl = ch[j + 2], zh = cl[j];
        double en = zl + P[1] * (zh - zl), s = H2 + P[2] * atr;
        if (!(en > cc[t]) || !(s > en)) continue;
        double risk = s - en, tpv;
        if (P[3] < 0.5) tpv = en - P[4] * risk;
        else {
            double best = -INFINITY;
            for (int a = 0; a < nao; a++) {
                int q = acto[a]; double px = ohi[q] - P[6] * (ohi[q] - olo[q]);
                if (px <= en - P[5] * risk && px > best) best = px;
            }
            if (best == -INFINITY) continue;
            tpv = best;
        }
        double max_rr = (P[7] > 0.0 && P[7] < 8.0) ? P[7] : 8.0;
        if (en - tpv > max_rr * risk) tpv = en - max_rr * risk;
        ok[e] = 1; entry[e] = en; sl[e] = s; tp[e] = tpv; zid[e] = z; cnt++;
    }
    free(act); free(acto);
    return cnt;
}

// ---------------------------------------------------------------- simulation (ordres limites, un seul trade à la fois)
// Règles prudentes (comme ton bot) : SL avant TP si les deux dans la même bougie ; gap = sortie à l'ouverture ;
// risque fixe en % du capital (frais compris), levier plafonné ; BE actif à partir de la bougie suivante ;
// ordre limite : rempli si la bougie touche le prix ; annulé si le TP est touché avant remplissage ou après ttl bougies.
// Sur la bougie de remplissage : seul le SL est testé (ordre des prix inconnu -> cas défavorable).
int simulate(int n, const double *o, const double *h, const double *l, const double *c,
             int ns, const int *sidx, const double *sentry, const double *ssl, const double *stp,
             double capital, double risk_pct, double fee, double lev, double be_rr, int ttl,
             const signed char *noentry, const signed char *fexit,
             double *equity, int *t_si, int *t_fi, int *t_xi, double *t_entry, double *t_exit, double *t_sl, double *t_tp,
             int *t_reason, double *t_pnl, double *t_r, double *t_bal) {
    double bal = capital; int nt = 0, busy = -1, eqpos = 0;
    for (int s = 0; s < ns; s++) {
        int i0 = sidx[s]; if (i0 < busy) continue;
        double e = sentry[s], sl = ssl[s], tp = stp[s];
        int fill = -1, i;
        for (i = i0 + 1; i < n && (ttl <= 0 || i - i0 <= ttl); i++) {
            if (noentry[i]) break;                 // week-end / annonce : l'ordre limite est annulé
            if (h[i] >= e) { fill = i; break; }
            if (l[i] <= tp) break;
        }
        if (fill < 0) { busy = i; continue; }
        double risk_dist = sl - e, loss_unit = risk_dist + fabs(e) * fee + fabs(sl) * fee;
        double risk_cash = bal * risk_pct / 100.0, qty = risk_cash / loss_unit;
        if (qty * fabs(e) > bal * lev) { busy = fill; continue; }
        for (int k = eqpos; k < fill; k++) equity[k] = bal;
        double entry_fee = qty * fabs(e) * fee; bal -= entry_fee;
        double cur_sl = sl, xp = c[n - 1]; int be_done = 0, xi = n - 1, reason = 3;
        for (int j = fill; j < n; j++) {
            if (h[j] >= cur_sl) { xp = (j > fill) ? fmax(cur_sl, o[j]) : cur_sl; reason = be_done ? 2 : 0; xi = j; break; }
            if (j > fill && l[j] <= tp) { xp = fmin(tp, o[j]); reason = 1; xi = j; break; }
            if (fexit[j]) { xp = c[j]; reason = 4; xi = j; break; }          // fermeture avant le week-end
            if (be_rr > 0 && !be_done && j > fill && l[j] <= e - be_rr * risk_dist) { cur_sl = e; be_done = 1; }
            equity[j] = bal + (e - c[j]) * qty;
        }
        double exit_fee = qty * fabs(xp) * fee, gross = (e - xp) * qty;
        bal += gross - exit_fee;
        equity[xi] = bal; eqpos = xi + 1; busy = xi;
        t_si[nt] = s; t_fi[nt] = fill; t_xi[nt] = xi; t_entry[nt] = e; t_exit[nt] = xp; t_sl[nt] = sl; t_tp[nt] = tp;
        t_reason[nt] = reason; t_pnl[nt] = gross - entry_fee - exit_fee; t_r[nt] = (gross - entry_fee - exit_fee) / risk_cash; t_bal[nt] = bal;
        nt++;
    }
    for (int k = eqpos; k < n; k++) equity[k] = bal;
    return nt;
}
