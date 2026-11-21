"""Task 520: specification and influence tests, as EW sums and EW moments.

The recursion they sit beside is `specification.rs`: a test of what a fit is missing is an
auxiliary regression read from EW centred moments of the out-of-sample residual `e` and
regressors `phi`, `LM = n R^2` at Kish's `n = (sum w)^2 / sum w^2`, `R^2 = c' Sxx^-1 c / Syy`
from `C` of the vector `(e^2, phi)`; each scored row (w > 0) enters once, a zero-weight row ages.

    ARCH LM (Engle 1982)   phi = (e_{t-1}^2 ... e_{t-L}^2) the last L SCORED squared residuals
                           (a ring of 8 L bytes); a row enters once the ring is full.      df = L
    White (1980)           phi = (x_i, x_i x_j for i <= j), q = k + k(k+1)/2 regressors,
                           df = q; `interaction_terms=False` keeps only the squares, q = 2k.
    Spiegelhalter (1986)   z = sum w (y - p)(1 - 2p) / sqrt(sum w^2 (1-2p)^2 p (1-p))
                           over EW sums; the weights squared in the denominator keep Var z = 1.
    DFBETAS, Cook's D      from the row's leverage `h = w x' G^-1 x` against the Gram `G` before it
                           and the out-of-sample residual `e`: `Delta b = G^-1 x w e/(1+h)`,
                           `v^2 = w e^2/(1+h)` the recursive residual (the `influence` model's own).
"""

from __future__ import annotations

import math
import warnings
from collections import deque

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_arch, het_white
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.api as sm

from .common import EwMoments, dt_stream, lam_of, table


def r2_from(ew: EwMoments):
    C = ew.C
    syy = C[0, 0]
    if not syy > 0.0:
        return math.nan
    c = C[1:, 0]
    return float(c @ np.linalg.solve(C[1:, 1:], c) / syy)


def kish(ew):
    return ew.W * ew.W / ew.Q if ew.Q > 0 else math.nan


class Arch:
    def __init__(self, L):
        self.L, self.ew, self.ring = L, EwMoments(L + 1), deque(maxlen=L)

    def step(self, e, lam, w):
        if w == 0.0:  # ages, and the ring takes scored rows only
            self.ew.update(None, lam, 0.0)
            return
        if len(self.ring) == self.L:
            u = np.concatenate([[e * e], np.array(self.ring)[::-1]])
            self.ew.update(u, lam, w)
        else:
            self.ew.update(None, lam, 0.0)
        self.ring.append(e * e)

    def lm(self):
        return kish(self.ew) * r2_from(self.ew)

    def state_bytes(self):
        L = self.L
        return 8 * ((L + 1) * (L + 2) // 2 + (L + 1) + 2 + L)


def white_terms(x, full=True):
    k = len(x)
    if not full:  # statsmodels: the squares only (the constant is the intercept)
        return np.asarray(x, float) ** 2
    sq = [x[i] * x[j] for i in range(k) for j in range(i, k)]
    return np.concatenate([x, sq])


class White:
    def __init__(self, k, full=True):
        self.k, self.full = k, full
        self.q = k + k * (k + 1) // 2 if full else k
        self.ew = EwMoments(self.q + 1)

    def step(self, e, x, lam, w):
        self.ew.update(np.concatenate([[e * e], white_terms(x, self.full)]) if w > 0 else None, lam, w)

    def lm(self):
        return kish(self.ew) * r2_from(self.ew)

    def state_bytes(self):
        d = self.q + 1
        return 8 * (d * (d + 1) // 2 + d + 2)


class Spiegelhalter:
    """S1 = sum w' a, S2 = sum w'^2 b, S2n = sum w' b: EW sums, `w'` the decayed weights."""

    def __init__(self):
        self.s1 = self.s2 = self.s2n = 0.0

    def step(self, y, p, lam, w):
        self.s1 *= lam
        self.s2 *= lam * lam
        self.s2n *= lam
        if w > 0:
            self.s1 += w * (y - p) * (1 - 2 * p)
            self.s2 += w * w * (1 - 2 * p) ** 2 * p * (1 - p)
            self.s2n += w * (1 - 2 * p) ** 2 * p * (1 - p)

    def z(self, naive=False):
        d = self.s2n if naive else self.s2
        return self.s1 / math.sqrt(d) if d > 0 else math.nan


class RlsInfl:
    """Weighted RLS in the Gram form, with each row's influence read against the fit before it."""

    def __init__(self, k):
        self.G = np.zeros((k, k))
        self.b = np.zeros(k)
        self.Wc = 0.0  # sum of weights of scored recursive residuals
        self.Qc = 0.0  # sum w e_in^2 (whitened recursive RSS)
        self.nrows = 0

    def step(self, x, y, lam, w):
        """Returns (dfbetas, cooks, h) for the row if it is scored, else None. Updates."""
        k = len(x)
        out = None
        Gd = lam * self.G
        if w > 0 and self.nrows >= k:
            Ginv_x = np.linalg.solve(Gd, x)
            h = w * float(x @ Ginv_x)
            beta = np.linalg.solve(Gd, lam * self.b)
            e = y - float(x @ beta)
            v2 = w * e * e / (1.0 + h)
            if self.Wc > 0:
                db = Ginv_x * w * e / (1.0 + h)
                Gf_inv_diag = np.diag(np.linalg.inv(Gd)) - w * Ginv_x**2 / (1.0 + h)
                s2_ext = self.Qc / self.Wc  # the fit without the row
                s2_full = (lam * self.Qc + v2) / (lam * self.Wc + w)
                dfb = db / (math.sqrt(s2_ext) * np.sqrt(Gf_inv_diag))
                out = (dfb, (w * e * e * h / (1.0 + h)) / (k * s2_full), h)
            self.Qc = lam * self.Qc + v2
            self.Wc = lam * self.Wc + w
        else:
            self.Qc *= lam
            self.Wc *= lam
        self.G = Gd + (w * np.outer(x, x) if w > 0 else 0.0)
        self.b = lam * self.b + (w * x * y if w > 0 else 0.0)
        if w > 0:
            self.nrows += 1
        return out


def _resid_stream(seed, n, k, hetero=False, arch=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k))
    sd = np.ones(n)
    if hetero:
        sd = np.sqrt(0.5 + 0.8 * x[:, 0] ** 2)
    e = rng.standard_normal(n) * sd
    if arch:
        for t in range(1, n):
            e[t] = rng.standard_normal() * math.sqrt(0.3 + 0.6 * e[t - 1] ** 2)
    return x, e


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    # ---------------- ARCH LM vs het_arch
    rows = []
    for L in (1, 4, 10):
        g = 0.0
        for sd in range(seeds):
            _, e = _resid_stream(sd, 300, 1, arch=True)
            a = Arch(L)
            for t in range(300):
                a.step(e[t], 1.0, 1.0)
            ref = het_arch(e, nlags=L)[0]
            g = max(g, abs(a.lm() - ref) / ref)
        rows.append([f"lam=1, unit w, L={L}, 300 rows", g])
    # window by retraction (100 rows) at L=4
    L = 4
    g = 0.0
    for sd in range(seeds):
        _, e = _resid_stream(sd, 300, 1, arch=True)
        # retraction: the contribution of row s (full-lag row) leaves at weight -1 when it ages out
        a = Arch(L)
        us = {}
        for t in range(300):
            if t >= L:
                us[t] = np.concatenate([[e[t] ** 2], e[t - L : t][::-1] ** 2])
            a.step(e[t], 1.0, 1.0)
            old = t - 100
            if old in us:
                a.ew.update(us[old], 1.0, -1.0)
            if t in (150, 299):
                win = e[t - 99 : t + 1]
                ref = het_arch(win, nlags=L)[0]
                # the window's rows with full lags inside the window only: rows old+1.. have lags in the stream, not the window
                # so compare against the stream-lag version: lags may cross the boundary (documented)
                u = np.array([us[s] for s in range(t - 99, t + 1) if s in us])
                y, Xl = u[:, 0], u[:, 1:]
                res = sm.OLS(y, sm.add_constant(Xl)).fit()
                ref2 = len(y) * res.rsquared
                g = max(g, abs(a.lm() - ref2) / ref2)
    rows.append(["window 100 by retraction vs OLS on the window's rows, L=4 (lags cross the boundary)", g])
    table("520a ARCH LM vs statsmodels `het_arch(resid, nlags=L)` (LM statistic, relative gap)", ["case", "largest gap"], rows)

    # decay/weights vs two-pass, zero-weight rows removed (scored rows only in the ring)
    g = 0.0
    for sd in range(seeds):
        _, e = _resid_stream(sd, 250, 1, arch=True)
        dt, w = dt_stream(sd, 250)
        L = 3
        a = Arch(L)
        for t in range(250):
            a.step(e[t], 1.0 if t == 0 else lam_of(dt[t], 40.0), w[t])
        # two-pass: scored rows, lags over scored rows, weights omega
        idx = [t for t in range(250) if w[t] > 0]
        lam_ = [1.0 if t == 0 else lam_of(dt[t], 40.0) for t in range(250)]
        om = {}
        acc = 1.0
        for t in range(249, -1, -1):
            om[t] = w[t] * acc
            acc *= lam_[t]
        rowsu, oms = [], []
        for j in range(L, len(idx)):
            s = idx[j]
            rowsu.append(np.concatenate([[e[s] ** 2], [e[idx[j - l]] ** 2 for l in range(1, L + 1)]]))
            oms.append(om[s])
        U, o = np.array(rowsu), np.array(oms)
        W = o.sum(); m = o @ U / W
        Cc = ((U - m) * o[:, None]).T @ (U - m) / W
        r2 = Cc[1:, 0] @ np.linalg.solve(Cc[1:, 1:], Cc[1:, 0]) / Cc[0, 0]
        ref = W * W / (o**2).sum() * r2
        g = max(g, abs(a.lm() - ref) / ref)
    table("520b ARCH LM under decay (half-life 40), weights, 10% zero-weight rows, a zero-weight first row, vs the two-pass weighted regression (statsmodels has no weights here)",
          ["case", "largest gap"], [["L=3", g]])

    # ---------------- White vs het_white
    rows = []
    for k in (2, 4, 8):
        for full in (True, False):
            g = 0.0
            for sd in range(max(5, seeds // 2)):
                x, e = _resid_stream(sd, 250, k, hetero=True)
                wh = White(k, full)
                for t in range(250):
                    wh.step(e[t], x[t], 1.0, 1.0)
                exog = sm.add_constant(x)
                ref = het_white(e, exog, interaction_terms=full)[0]
                g = max(g, abs(wh.lm() - ref) / ref)
            rows.append([f"k={k}, interaction_terms={full}, q={k + k * (k + 1) // 2 if full else k} regressors", g, wh.state_bytes()])
    table("520c White's LM vs statsmodels `het_white(resid, exog, interaction_terms=)` (exog with a constant; relative gap; state bytes of the accumulator)",
          ["case", "largest gap", "state bytes"], rows)
    # window of 80 by retraction, k = 3
    g = 0.0
    k = 3
    for sd in range(max(5, seeds // 2)):
        x, e = _resid_stream(sd, 250, k, hetero=True)
        wh = White(k)
        for t in range(250):
            wh.step(e[t], x[t], 1.0, 1.0)
            if t >= 80:
                wh.ew.update(np.concatenate([[e[t - 80] ** 2], white_terms(x[t - 80])]), 1.0, -1.0)
        t = 249
        ref = het_white(e[t - 79 : t + 1], sm.add_constant(x[t - 79 : t + 1]))[0]
        g = max(g, abs(wh.lm() - ref) / ref)
    print(f"\n520c' White over a window of 80 by retraction vs `het_white` on the window: {g:.2e}")
    # null size under decay
    rej = 0
    nrep = 300
    k = 3
    for sd in range(nrep):
        x, e = _resid_stream(1000 + sd, 600, k)
        wh = White(k)
        for t in range(600):
            wh.step(e[t], x[t], 0.5 ** (1 / 100.0), 1.0)
        rej += wh.lm() > stats.chi2.ppf(0.95, wh.q)
    print(f"520c'' White's size at the 5% value under half-life 100 rows, Kish n, iid homoskedastic, k=3 (q=9): {rej / nrep:.3f} over {nrep} streams (binomial sd 0.013)")
    # cost cap
    rows = []
    for k in (5, 10, 20, 21, 30):
        q = k + k * (k + 1) // 2
        d = q + 1
        by = 8 * (d * (d + 1) // 2 + d + 2)
        rows.append([k, q, by, "over the 256 KB cap" if by > 256 * 1024 else "within"])
    table("520d White's accumulator `8 ((q+1)(q+2)/2 + q + 3)` bytes, `q = k + k(k+1)/2`, against the 256 KB per-group cap", ["k", "q", "bytes", "cap"], rows)

    # ---------------- Spiegelhalter vs MAPIE
    from mapie.metrics.calibration import spiegelhalter_p_value, spiegelhalter_statistic

    rows = []
    for miscal in (0.0, 0.8):
        g = gp = gp2 = 0.0
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            n = 400
            p_true = rng.uniform(0.05, 0.95, n)
            y = (rng.random(n) < p_true).astype(float)
            p = np.clip(p_true + miscal * (0.5 - p_true) * 0.3, 0.01, 0.99)
            sp = Spiegelhalter()
            for t in range(n):
                sp.step(y[t], p[t], 1.0, 1.0)
            z = sp.z()
            zr = spiegelhalter_statistic(y, p)
            g = max(g, abs(z - zr) / max(1.0, abs(zr)))
            pv = float(stats.norm.sf(z))  # one-sided, MAPIE's
            gp = max(gp, abs(pv - spiegelhalter_p_value(y, p)))
            pv2 = 2 * min(pv, 1 - pv)
            gp2 = max(gp2, abs(pv2 - 2 * stats.norm.sf(abs(zr))))
        rows.append([f"lam=1, unit w, shrink toward .5: {miscal}", g, gp, gp2])
    table("520e Spiegelhalter's z vs MAPIE `spiegelhalter_statistic` (relative), one-sided p vs `spiegelhalter_p_value` (absolute), two-sided p vs `2 sf(|z|)`",
          ["case", "z gap", "one-sided p gap", "two-sided p gap"], rows)
    # decay with weights: null distribution of z, principled vs naive denominator
    rows = []
    zs_p, zs_n = [], []
    for sd in range(300):
        rng = np.random.default_rng(5000 + sd)
        n = 500
        p = rng.uniform(0.05, 0.95, n)
        y = (rng.random(n) < p).astype(float)
        dt = np.ones(n)
        sp = Spiegelhalter()
        lam = 0.5 ** (1 / 30.0)
        for t in range(n):
            sp.step(y[t], p[t], lam, 1.0 + (t % 3) * 0.5)
        zs_p.append(sp.z())
        zs_n.append(sp.z(naive=True))
    rows.append(["`sum w^2 b` (squared weights): the form proposed", float(np.std(zs_p)), float(np.mean(zs_p))])
    rows.append(["`sum w b` (the weights not squared): the unit-weight formula read naively", float(np.std(zs_n)), float(np.mean(zs_n))])
    table("520f z under H0 (calibrated p), half-life 30 rows, weights 1/1.5/2, 500 rows, 300 streams", ["denominator", "sd of z (should be 1)", "mean"], rows)
    # zero-weight rows
    g = 0.0
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        n = 200
        p = rng.uniform(0.05, 0.95, n)
        y = (rng.random(n) < p).astype(float)
        dt, w = dt_stream(sd, n)
        a = Spiegelhalter()
        for t in range(n):
            a.step(y[t], p[t], 1.0 if t == 0 else lam_of(dt[t], 25.0), w[t])
        b = Spiegelhalter()
        carry = 0.0
        for t in range(n):
            carry += 0.0 if t == 0 else dt[t]
            if w[t] > 0:
                b.step(y[t], p[t], lam_of(carry, 25.0), w[t])
                carry = 0.0
        g = max(g, abs(a.z() - b.z()) / abs(a.z()) if a.z() else 0)
    print(f"\n520f' zero-weight rows (10%, first included) vs gaps folded: z relative gap {g:.1e}")

    # ---------------- DFBETAS / Cook's vs OLSInfluence
    rows = []
    for label, wt, hl in (("lam=1, unit w", False, np.inf), ("lam=1, w in [.5,2] (divisor corrected)", True, np.inf), ("half-life 40, w, zero-weight rows (divisor corrected)", True, 40.0)):
        gd = gc = 0.0
        for sd in range(seeds):
            k = 3
            rng = np.random.default_rng(sd)
            n = 120
            x = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
            y = x @ rng.standard_normal(k) + rng.standard_normal(n)
            dt, w = dt_stream(sd, n, zero_first=hl != np.inf, zero_frac=0.1 if hl != np.inf else 0.0)
            if not wt:
                w = np.ones(n)
            r = RlsInfl(k)
            outs = {}
            for t in range(n):
                lam = 1.0 if (t == 0 or hl == np.inf) else lam_of(dt[t], hl)
                o = r.step(x[t], y[t], lam, w[t])
                if o is not None and t in (40, 79, 119):
                    outs[t] = (o, r.Wc, r.nrows)
            for t, (o, Wc, nr) in outs.items():
                lam_ = np.array([1.0 if (s == 0 or hl == np.inf) else lam_of(dt[s], hl) for s in range(t + 1)])
                om = np.empty(t + 1)
                acc = 1.0
                for s in range(t, -1, -1):
                    om[s] = w[s] * acc
                    acc *= lam_[s]
                keep = om > 0
                # whitened OLS: statsmodels' own WLS influence is not the weighted leverage
                sw = np.sqrt(om[keep])
                res = sm.OLS(y[: t + 1][keep] * sw, x[: t + 1][keep] * sw[:, None]).fit()
                infl = OLSInfluence(res)
                dfb_sm = infl.dfbetas[-1]
                ck_sm = infl.cooks_distance[0][-1]
                nobs = int(keep.sum())
                # statsmodels' s^2 divides by (n - k) rows, rheon's by the scored weight sum Wc (here Wc = sum of omegas)
                # rescale: rheon s^2 = Q/W_scored; statsmodels s^2 = Q/(n-1-k) for the external one and Q'/(n-k) full
                n_ext = nobs - 1 - k
                n_full = nobs - k
                # Wc here is the scored weight sum INCLUDING this row; the external one excludes it
                Wc_ext = Wc - om[-1]
                dfb_mine = o[0] * math.sqrt(n_ext / Wc_ext)
                ck_mine = o[1] * (n_full / Wc)
                gd = max(gd, float(np.max(np.abs(dfb_mine - dfb_sm)) / np.max(np.abs(dfb_sm))))
                gc = max(gc, abs(ck_mine - ck_sm) / ck_sm)
        rows.append([label, gd, gc])
    table("520g DFBETAS and Cook's distance of the newest row vs `OLSInfluence` on the prefix (OLS on the rows whitened by the EW weights; k=3; rows 40, 79, 119; relative gaps, DFBETAS to the row's largest entry; "
          "where weights are not 1 the oracle's `s^2` divides by `n - k` rows and rheon's `influence` by the scored weight, corrected by their ratio)",
          ["case", "DFBETAS", "Cook's D"], rows)
    print("\n520 state added (per slot, per accumulator):")
    wrows = []
    for L in (1, 4, 10, 30):
        wrows.append([f"ARCH LM, L={L}", Arch(L).state_bytes()])
    for k in (5, 10, 20):
        wrows.append([f"White full, k={k}", White(k).state_bytes()])
    wrows.append(["Spiegelhalter", 8 * 3])
    wrows.append(["DFBETAS / Cook beside `influence`", "8 k (diag of the inverse Gram, refreshed with the solve), 8 (W_c, Q_c shared with `influence`)"])
    table("520h state bytes", ["test", "bytes"], wrows)
