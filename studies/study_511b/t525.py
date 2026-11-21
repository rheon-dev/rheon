"""Task 525: smaller options.

* `kmeans` with given (fixed) centres: distances to every centre, the assignment, and an EW inertia (the EW mean of the row's squared
  distance to its nearest centre, read before the row folds in). Oracle: `KMeans(init=C, n_init=1)` frozen at `C`: `transform`, `predict`, `score`.
* `ew_class`: fixed class priors; a regulariser that does not fade (`reg_param`: `M_c = (1 - r) S_c + r I`, with `S_c` the class covariance, divisor `W_c`);
  `var_smoothing` (the diagonal shape's `eps = vs * max_i var_i(all rows)`, the total variance read from the class moments). Oracles:
  `GaussianNB(priors, var_smoothing)`, `QuadraticDiscriminantAnalysis(priors, reg_param)`.
* `holt`: `forecast(h) = l + h b` and a known initial level and trend (the textbook gains from row 1: the state starts with the saturated weight
  `W = 1/(1 - lam)`, so the first gain is `alpha`, not 1). Oracle: `statsmodels` `Holt(initialization_method='known').fit(optimized=False)`.
* Holm beside Bonferroni (`multipletests(method='holm')`).
* a window two-sample (Kolmogorov-Smirnov) test for feature health: the last `r` values against the older ones in a ring (`scipy.stats.ks_2samp`).
Not built, oracle MISSING (C1): `deco`'s less biased alternative to `u` (Engle and Kelly 2012; the paper's formula is the only oracle).
"""

from __future__ import annotations

import math
import warnings
from collections import deque

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from .common import EwMoments, dt_stream, lam_of, omegas, table


# ----------------------------------------------------------------------------- kmeans
class FixedKMeans:
    def __init__(self, C):
        self.C = np.asarray(C, float)
        self.W = 0.0
        self.m = 0.0  # EW mean of d^2_min

    def step(self, x, lam, w):
        d2 = np.sum((self.C - x) ** 2, axis=1)
        j = int(np.argmin(d2))  # first minimum wins
        out = (np.sqrt(d2), j, math.sqrt(d2[j]), self.m)  # inertia read before the row
        Wo = lam * self.W
        W2 = Wo + w
        self.W = W2
        if w > 0 and W2 > 0:
            self.m = self.m + (w / W2) * (d2[j] - self.m)
        return out


def report_kmeans(seeds):
    from sklearn.cluster import KMeans

    rows = []
    for K, p in ((3, 2), (8, 5), (20, 16)):
        g_t = g_s = g_ew = 0
        mism = 0
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            C = rng.standard_normal((K, p)) * 3
            n = 300
            X = rng.standard_normal((n, p)) * 2 + C[rng.integers(K, size=n)]
            dt, w = dt_stream(sd, n)
            ref = KMeans(n_clusters=K, init=C, n_init=1, max_iter=1).fit(C)  # one point per centre: the centres stay C
            assert np.allclose(ref.cluster_centers_, C, rtol=0, atol=1e-12)
            ref.cluster_centers_ = C.copy()  # exactly the given centres (the one-point fit leaves rounding)
            km = FixedKMeans(C)
            ds, labs = [], []
            for t in range(n):
                lam = 1.0 if t == 0 else lam_of(dt[t], 30.0)
                dall, j, dj, inertia = km.step(X[t], lam, w[t])
                ds.append(dall)
                labs.append(j)
            g_t = max(g_t, float(np.max(np.abs(np.array(ds) - ref.transform(X)))))
            mism += int(np.sum(np.array(labs) != ref.predict(X)))
            om = omegas(dt, w, 30.0, n - 1)
            sk = -ref.score(X, sample_weight=om)  # sum omega d^2
            g_ew = max(g_ew, abs(km.m * km.W - sk) / sk)
        rows.append([f"K = {K}, p = {p}", g_t, mism, g_ew])
    table("525a `kmeans` with given centres (fixed): distances to every centre vs `KMeans(init=C, n_init=1).transform` (Euclidean, not squared; absolute), the assignment vs `predict` (mismatches over 300 rows x seeds), "
          "and the EW inertia `m W` (half-life 30, weights, 10% zero-weight rows, a zero-weight first row) vs `-score(X, sample_weight=omega)` (relative)",
          ["case", "distances", "assignment mismatches", "EW inertia"], rows)
    print("\n525a state added: the centres `8 K p` bytes (configuration, but state once they are the model's: K = 20, p = 16: 2.5 KB); the EW inertia `8` and its weight `8`; the `K` distances are output slots, not state. "
          "Zero-weight rows age `W`, learn nothing. Learning from given centres has no exact oracle here: C1 did not re-fetch `MiniBatchKMeans.partial_fit`, and `KMeans` runs full Lloyd; not studied.")


# ----------------------------------------------------------------------------- ew_class
class EwClass:
    def __init__(self, k, ncls):
        self.k, self.ncls = k, ncls
        self.acc = [EwMoments(k) for _ in range(ncls)]
        self.n = np.zeros(ncls)

    def step(self, x, c, lam, w):
        for j in range(self.ncls):
            if j != c:
                self.acc[j].update(None, lam, 0.0)
        self.acc[c].update(x, lam, w)
        self.n = np.array([a.W for a in self.acc])

    def total_var(self):
        N = self.n.sum()
        m = sum(a.W * a.m for a in self.acc) / N
        tot = sum(a.W * (np.diag(a.C) + (a.m - m) ** 2) for a in self.acc) / N
        return tot

    def posterior(self, x, shape, priors=None, reg=0.0, var_smoothing=0.0, unbiased=False):
        ll = []
        N = self.n.sum()
        eps = var_smoothing * np.max(self.total_var()) if var_smoothing else 0.0
        for j, a in enumerate(self.acc):
            pi = (a.W / N) if priors is None else priors[j]
            if shape == "diag":
                v = np.diag(a.C) + eps
                lj = -0.5 * np.sum(np.log(2 * np.pi * v) + (x - a.m) ** 2 / v)
            else:
                c = a.unbiased_factor() if unbiased else 1.0
                S = c * a.C
                S = (1 - reg) * S + reg * np.eye(self.k)
                d = x - a.m
                lj = -0.5 * (self.k * math.log(2 * math.pi) + np.linalg.slogdet(S)[1] + d @ np.linalg.solve(S, d))
            ll.append(math.log(pi) + lj)
        ll = np.array(ll)
        return np.exp(ll - logsumexp(ll))


def make_class_stream(sd, n, k, ncls, imbalance=True):
    rng = np.random.default_rng(sd)
    means = rng.standard_normal((ncls, k)) * 1.5
    A = [rng.standard_normal((k, k)) * 0.4 + np.eye(k) for _ in range(ncls)]
    pr = rng.dirichlet(np.ones(ncls) * (1.0 if imbalance else 50.0))
    pr = np.maximum(pr, 0.05)
    pr /= pr.sum()
    y = rng.choice(ncls, size=n, p=pr)
    X = np.array([means[c] + rng.standard_normal(k) @ A[c] for c in y])
    return X, y


def report_class(seeds):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB

    rows = []
    # QDA: priors and reg_param
    for reg in (0.0, 0.1, 0.5):
        g = 0.0
        for sd in range(seeds):
            k, ncls, n = 4, 3, 400
            X, y = make_class_stream(sd, n, k, ncls)
            priors = np.array([0.5, 0.3, 0.2])
            ec = EwClass(k, ncls)
            for t in range(n):
                ec.step(X[t], y[t], 1.0, 1.0)
            qda = QuadraticDiscriminantAnalysis(priors=priors, reg_param=reg, tol=1e-12).fit(X, y)
            Xq = np.random.default_rng(sd + 3).standard_normal((15, k)) * 2
            ref = qda.predict_proba(Xq)
            mine = np.array([ec.posterior(x, "full", priors=priors, reg=reg, unbiased=False) for x in Xq])
            g = max(g, float(np.max(np.abs(ref - mine))))
        rows.append([f"QDA, `priors` fixed, `reg_param` = {reg}, `lam = 1`, unit w", g])
    # GaussianNB: priors, var_smoothing
    for vs in (1e-9, 1e-2, 0.5):
        g = 0.0
        for sd in range(seeds):
            k, ncls, n = 5, 3, 400
            X, y = make_class_stream(sd, n, k, ncls)
            priors = np.array([0.2, 0.3, 0.5])
            ec = EwClass(k, ncls)
            for t in range(n):
                ec.step(X[t], y[t], 1.0, 1.0)
            nb = GaussianNB(priors=priors, var_smoothing=vs).fit(X, y)
            Xq = np.random.default_rng(sd + 3).standard_normal((15, k)) * 2
            ref = nb.predict_proba(Xq)
            mine = np.array([ec.posterior(x, "diag", priors=priors, var_smoothing=vs) for x in Xq])
            g = max(g, float(np.max(np.abs(ref - mine))))
        rows.append([f"GaussianNB, `priors` fixed, `var_smoothing` = {vs}, `lam = 1`, unit w", g])
    table("525b `ew_class` options vs `QuadraticDiscriminantAnalysis(priors, reg_param)` and `GaussianNB(priors, var_smoothing)` `predict_proba` (absolute gap in the posteriors, 15 query rows, 20 seeds; "
          "both use the biased divisor `n`: scikit-learn 1.9.1's QDA svd solver takes `scaling = S**2 / n_samples`)", ["case", "largest gap"], rows)

    # weights: GaussianNB's epsilon ignores weights (the source: np.var(X, axis=0) unweighted); reproduce by setting var_smoothing so eps matches
    g_w = g_eps = 0.0
    for sd in range(seeds):
        k, ncls, n = 5, 3, 300
        X, y = make_class_stream(sd, n, k, ncls)
        rng = np.random.default_rng(sd + 9)
        w = rng.uniform(0.5, 2.0, n)
        ec = EwClass(k, ncls)
        for t in range(n):
            ec.step(X[t], y[t], 1.0, w[t])
        priors = np.array([0.2, 0.3, 0.5])
        vs = 0.05
        eps_mine = vs * np.max(ec.total_var())
        eps_sk = vs * np.max(np.var(X, axis=0))
        g_eps = max(g_eps, abs(eps_mine - eps_sk) / eps_sk)
        vs_match = eps_mine / np.max(np.var(X, axis=0))  # the var_smoothing that gives sklearn rheon's weighted eps
        nb = GaussianNB(priors=priors, var_smoothing=vs_match).fit(X, y, sample_weight=w)
        Xq = np.random.default_rng(sd + 3).standard_normal((15, k)) * 2
        mine = np.array([ec.posterior(x, "diag", priors=priors, var_smoothing=vs) for x in Xq])
        g_w = max(g_w, float(np.max(np.abs(nb.predict_proba(Xq) - mine))))
    print(f"\n525b' with row weights in [.5, 2]: GaussianNB's weighted fit vs `diag` with `var_smoothing` read from the weighted total variance: posterior gap {g_w:.1e} when sklearn is given the `var_smoothing` that reproduces "
          f"rheon's epsilon (they differ by up to {g_eps:.2f} relative if one passes the same `var_smoothing`: sklearn's `epsilon_ = var_smoothing * max(np.var(X, axis=0))` is **unweighted** and taken over the rows of each `fit` or `partial_fit` call, C1's nuance, confirmed in the source).")

    # decay: against a two-pass weighted definition (sklearn has no decay)
    g = 0.0
    for sd in range(seeds):
        k, ncls, n = 4, 3, 250
        X, y = make_class_stream(sd, n, k, ncls)
        dt, w = dt_stream(sd, n)
        ec = EwClass(k, ncls)
        for t in range(n):
            ec.step(X[t], y[t], 1.0 if t == 0 else lam_of(dt[t], 40.0), w[t])
        om = omegas(dt, w, 40.0, n - 1)
        priors = np.array([0.5, 0.3, 0.2])
        Xq = np.random.default_rng(sd + 3).standard_normal((10, k)) * 2
        for x in Xq:
            mine = ec.posterior(x, "full", priors=priors, reg=0.2)
            ll = []
            for c in range(ncls):
                idx = np.where((y == c) & (om > 0))[0]
                o = om[idx]
                W = o.sum()
                if W == 0:
                    ll.append(-np.inf)
                    continue
                m = o @ X[idx] / W
                S = ((X[idx] - m) * o[:, None]).T @ (X[idx] - m) / W
                S = 0.8 * S + 0.2 * np.eye(k)
                d = x - m
                ll.append(math.log(priors[c]) - 0.5 * (k * math.log(2 * math.pi) + np.linalg.slogdet(S)[1] + d @ np.linalg.solve(S, d)))
            ll = np.array(ll)
            g = max(g, float(np.max(np.abs(mine - np.exp(ll - logsumexp(ll))))))
    table("525b'' fixed priors and `reg_param` under decay (half-life 40), weights, 10% zero-weight rows, a zero-weight first row vs the two-pass weighted definition (sklearn has no decay)", ["case", "largest gap"], [["`full`, `reg_param` .2, priors fixed", g]])

    # the permanence of reg_param versus the fading ridge: the ridge scale s after n rows (lam = 1, unit weights) is 1/n
    scales = []
    for n_c in (6, 50, 400):
        ew = EwMoments(3)
        rng = np.random.default_rng(1)
        for t in range(n_c):
            ew.update(rng.standard_normal(3), 1.0, 1.0)
        scales.append(ew.s)
    print("\n525b-3 the existing ridge `precision_prior * s_c` fades with the class own rows (`s = a s`): "
          f"s = {scales[0]:.4f} / {scales[1]:.4f} / {scales[2]:.5f} after 6 / 50 / 400 rows (= 1/n at `lam = 1`); "
          "`reg_param` is a configured constant share of `I` in `M_c = (1 - r) S_c + r I` for as long as the class runs, and adds no state.")
    print("525b state added: fixed priors `8 n_classes` bytes (configuration); `reg_param`, `var_smoothing` 8 each; the total variance `eps` is read from the class moments each solve (`O(n_classes k)`), **no new state**.")


# ----------------------------------------------------------------------------- holt
class HoltTextbook:
    """The textbook gains (`holt.rs`): l' = pred + g e, b' = b + h e with `g = w/(lam_l W + w)`, `h = w/(lam_b V + w) g / s`, W and V the weights gathered. `known=(l0, b0, W0, V0)`
    starts from a given state with its weight, so the first gains are `alpha` and `beta` rather than 1; the line-fit gain (the larger-gain rule) is off under a known start."""

    def __init__(self, known=None, ll=None, lb=None):
        self.l = self.b = 0.0
        self.W = self.V = 0.0
        self.s = 0.0
        self.seen = False
        self.lam_l, self.lam_b = ll, lb
        if known is not None:
            self.l, self.b, self.W, self.V = known
            self.seen = True

    def forecast(self, h):
        return self.l + h * self.b

    def step(self, y, d, w=1.0):
        """`d` the clock since the last row; zero weight is transparent (its `d` joins the next row's `s`)."""
        self.s += d
        if w == 0.0:
            return None
        s = self.s
        pred = self.l + self.b * s
        ll_s = self.lam_l**s
        lb_s = self.lam_b**s
        if not self.seen:
            self.l, self.b, self.W, self.V = y, 0.0, w, w
            self.seen = True
            self.s = 0.0
            return pred
        e = y - pred
        g = w / (ll_s * self.W + w)
        h = (w / (lb_s * self.V + w)) * g / s if s > 0 else 0.0
        self.W = ll_s * self.W + w
        self.V = lb_s * self.V + w
        self.l = pred + g * e
        self.b = self.b + h * e
        self.s = 0.0
        return pred


def report_holt(seeds):
    from statsmodels.tsa.holtwinters import Holt

    rows = []
    for alpha, beta in ((0.3, 0.1), (0.6, 0.3), (0.1, 0.05)):
        g_fit = g_lvl = g_tr = g_fc = 0.0
        for sd in range(seeds):
            rng = np.random.default_rng(sd)
            n = 120
            y = 5 + 0.2 * np.arange(n) + np.cumsum(0.1 * rng.standard_normal(n)) + rng.standard_normal(n)
            l0, b0 = 4.0, 0.15
            res = Holt(y, initialization_method="known", initial_level=l0, initial_trend=b0).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
            lam_l, lam_b = 1 - alpha, 1 - beta
            # saturated weights: the first gain is alpha, beta
            Wsat, Vsat = 1.0 / (1 - lam_l), 1.0 / (1 - lam_b)
            m = HoltTextbook(known=(l0, b0, Wsat, Vsat), ll=lam_l, lb=lam_b)
            preds = []
            lv, tr = [], []
            for t in range(n):
                preds.append(m.step(y[t], 1.0))
                lv.append(m.l)
                tr.append(m.b)
            g_fit = max(g_fit, float(np.max(np.abs(np.array(preds) - res.fittedvalues))))
            g_lvl = max(g_lvl, float(np.max(np.abs(np.array(lv) - res.level))))
            g_tr = max(g_tr, float(np.max(np.abs(np.array(tr) - res.trend))))
            for h in (1, 5, 30):
                g_fc = max(g_fc, abs(m.forecast(h) - res.forecast(h)[-1]))
        rows.append([f"alpha = {alpha}, beta = {beta}", g_fit, g_lvl, g_tr, g_fc])
    table("525c `holt` from a known initial level and trend with the textbook gains from row 1 (the start carries the saturated weight `1/(1 - lam)`) vs `Holt(initialization_method='known', initial_level, initial_trend).fit(smoothing_level, smoothing_trend, optimized=False)` "
          "(unit clock; 120 rows; absolute gaps in the one-step forecasts, the level, the trend, and `forecast(h)` for h = 1, 5, 30)",
          ["smoothing", "one-step forecasts", "level", "trend", "forecast(h)"], rows)

    # zero weight transparent: fold
    g = 0.0
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        n = 100
        y = 3 + 0.1 * np.arange(n) + rng.standard_normal(n)
        dt, w = dt_stream(sd, n)
        dt = np.ones(n) * 1.0
        a = HoltTextbook(known=(2.0, 0.1, 10.0, 20.0), ll=0.9, lb=0.95)
        for t in range(n):
            a.step(y[t], dt[t], w[t])
        b = HoltTextbook(known=(2.0, 0.1, 10.0, 20.0), ll=0.9, lb=0.95)
        carry = 0.0
        for t in range(n):
            carry += dt[t]
            if w[t] > 0:
                b.step(y[t], carry, w[t])
                carry = 0.0
        g = max(g, abs(a.l - b.l), abs(a.b - b.b))
    print(f"\n525c' zero-weight rows are transparent (10%, first included): their clock joins the next row's `s`; against the stream without them and the gaps folded, level and trend differ by {g:.1e}")
    print("525c state added: 0 (the known state is the existing `l`, `b`, `W`, `V`; its weights are two configuration numbers). `forecast(h)` reads `l + h b` and is an output, not state; at `h = s` it is the pred the model already emits. "
          "Where it is not exact: statsmodels has no clock, so the comparison is on a unit clock; and the line-fit gain (`g = max(g_f, g_t)`, which keeps the unseeded model a least-squares line early on) is switched off under a known start, "
          "since it would dominate the first rows with a gain of about `1/n`: a decision, not a default change (a known start is new).")


# ----------------------------------------------------------------------------- Holm
def holm_reject(p, alpha):
    m = len(p)
    order = np.argsort(p)
    rej = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def holm_adjusted(p):
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def report_holm(seeds):
    from statsmodels.stats.multitest import multipletests

    rows = []
    for m in (3, 10, 45):
        ag_dec = ag_adj = ag_bon = 0
        mism = 0
        for sd in range(seeds * 5):
            rng = np.random.default_rng(sd)
            p = rng.uniform(size=m) ** rng.choice([1, 3, 8], size=m)  # a mix of nulls and small p-values
            r1, a1, _, _ = multipletests(p, alpha=0.05, method="holm")
            mism += int(np.sum(holm_reject(p, 0.05) != r1))
            ag_adj = max(ag_adj, float(np.max(np.abs(holm_adjusted(p) - a1))))
            rb, ab, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
            ag_bon = max(ag_bon, float(np.max(np.abs(np.minimum(1.0, p * m) - ab))))
            assert not np.any(rb & ~r1)  # Holm rejects everything Bonferroni does
        rows.append([f"m = {m} pairs", mism, ag_adj, ag_bon])
    table("525d Holm's step-down vs `multipletests(method='holm')` (rejections: mismatches over 100 p-value sets; adjusted p-values: absolute gap) and Bonferroni `min(1, m p)` vs `method='bonferroni'` (C1: `multipletests`' default is `'hs'`, Holm-Sidak, so Holm must be passed by name). "
          "Holm rejected a superset of Bonferroni's on every set", ["case", "rejection mismatches", "Holm adjusted p", "Bonferroni adjusted p"], rows)

    # corrchange's monitor: Kolmogorov quantile per pair at alpha/m; Holm = ordered Kolmogorov quantiles at alpha/(m - j)
    nrep = 4000
    rows = []
    for m, shifted in ((10, 0), (10, 3), (45, 5)):
        fw_b = fw_h = 0
        pw_b = pw_h = 0.0
        for sd in range(nrep):
            rng = np.random.default_rng(10000 + sd)
            # per-pair sup statistics: Kolmogorov under H0; a shifted pair's statistic is larger
            sup = stats.kstwobign.rvs(size=m, random_state=rng)
            sup[:shifted] = sup[:shifted] + 1.6
            p = stats.kstwobign.sf(sup)
            rb = p <= 0.05 / m
            rh = holm_reject(p, 0.05)
            # the quantile form (crit at 1 - alpha/(m - j)) agrees with the p form
            order = np.argsort(-sup)
            rq = np.zeros(m, bool)
            for j, i in enumerate(order):
                if sup[i] >= stats.kstwobign.isf(0.05 / (m - j)):
                    rq[i] = True
                else:
                    break
            assert np.array_equal(rq, rh)
            if shifted == 0:
                fw_b += rb.any()
                fw_h += rh.any()
            else:
                fw_b += rb[shifted:].any()
                fw_h += rh[shifted:].any()
                pw_b += rb[:shifted].mean()
                pw_h += rh[:shifted].mean()
        rows.append([f"m = {m} pairs, {shifted} shifted", fw_b / nrep, fw_h / nrep, pw_b / nrep if shifted else "", pw_h / nrep if shifted else ""])
    table("525d' `corrchange`'s `monitor` pairs: the Kolmogorov sup statistic per pair, Bonferroni (critical value at `1 - alpha/m`, today) against Holm (the ordered statistics against the quantiles at `1 - alpha/(m - j)`: asserted equal to the p-value step-down), 4000 replications at alpha .05: "
          "family-wise error (a null pair flagged) and the mean share of shifted pairs flagged (a shift of +1.6 in the statistic)", ["case", "FWER Bonferroni", "FWER Holm", "power Bonferroni", "power Holm"], rows)
    print("\n525d state added: 0 (a step-down over the pairs' current statistics, `O(m log m)` per read; `m = k(k-1)/2`: 190 pairs at k = 20). It changes no default: `alpha_adjust` gains `\"holm\"` beside `\"bonferroni\"` and `\"none\"`. "
          "A per-row monitor repeated over time is not a one-shot family, so both are conservative-by-construction only per read; unchanged from Bonferroni's.")


# ----------------------------------------------------------------------------- KS window test
class KSWindow:
    """A ring of the last `window` values; the test compares the last `r` against the `window - r` before them."""

    def __init__(self, window, r):
        self.window, self.r = window, r
        self.ring = deque(maxlen=window)

    def step(self, x):
        self.ring.append(x)
        if len(self.ring) < self.window:
            return None
        v = np.array(self.ring)
        recent, ref = v[-self.r :], v[: -self.r]
        # KS distance by a merge of the sorted samples
        a, b = np.sort(recent), np.sort(ref)
        allv = np.concatenate([a, b])
        cdf1 = np.searchsorted(a, allv, side="right") / len(a)
        cdf2 = np.searchsorted(b, allv, side="right") / len(b)
        return float(np.max(np.abs(cdf1 - cdf2)))

    def state_bytes(self):
        return 8 * self.window


def report_ks(seeds):
    rows = []
    for window, r in ((100, 30), (300, 50), (1000, 100)):
        g = gp = 0.0
        for sd in range(max(5, seeds // 2)):
            rng = np.random.default_rng(sd)
            n = 2 * window
            x = np.concatenate([rng.standard_normal(n // 2), rng.standard_normal(n // 2) * 1.5 + 0.3])
            x = np.round(x, 1) if sd % 2 else x  # ties on half the seeds
            ks = KSWindow(window, r)
            for t in range(n):
                d = ks.step(x[t])
                if d is not None and t % 37 == 0:
                    v = x[t - window + 1 : t + 1]
                    res = stats.ks_2samp(v[-r:], v[:-r])
                    g = max(g, abs(d - res.statistic))
        rows.append([f"window {window}, recent {r}", g, 8 * window])
    table("525e window two-sample KS: the last `r` values against the `window - r` before them (a deterministic split) vs `scipy.stats.ks_2samp` on the same two samples (statistic, absolute; ties on half of the seeds); ring bytes", ["case", "statistic gap", "state bytes"], rows)
    # river's KSWIN reference
    try:
        import inspect

        from river.drift import KSWIN

        src = inspect.getsource(KSWIN.update)
        print("\n525e' river's `KSWIN.update` (the reference, not exact: it draws `stat_size` values AT RANDOM from the older part): " + " | ".join(s.strip() for s in src.splitlines() if "ks_2samp" in s or "p_value" in s or "stat" in s and "<" in s)[:400])
    except Exception as e:  # pragma: no cover
        print(f"\n525e' river's KSWIN not available here: {type(e).__name__}")
    print("525e state added: a ring of `window` values, `8 window` bytes per feature (window 1000: 8 KB a feature; k = 20 features: 160 KB); the 256 KB per-group cap holds `window <= 32768 / k` values per feature. "
          "Cost `O(window log window)` per read (a sorted structure makes it `O(log window)`). The test is two-sided on a hard window, no decay; a zero-weight row enters no ring.")


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    report_kmeans(seeds)
    report_class(seeds)
    report_holt(seeds)
    report_holm(seeds)
    report_ks(seeds)
