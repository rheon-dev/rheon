"""Task 522: `hmm`.

Beside `hmm.rs`'s Hamilton filter (before the row, `p~ = p Pi`, uniform `p` before the first; `f_l = N(x | mu_l, S_l)`;
`loglik = ln sum p~ f`; `p = p~ f / sum`; counts `A += w xi`, `xi_kl = p_k(t-1) Pi_kl f_l / sum`; `Pi = (A + tau)/sum(A + tau)`;
state accumulators take the row at weight `w p_l`). A zero-weight row leaves the filtered `p` where it was (the Rust test
`a_zero_weight_row_is_clock_alone`), so here it is skipped by the filter and the smoother's rings.

    fixed-lag smoothing     gamma_{t-L | t} ~ p_{t-L} * beta,  beta_t = 1, beta_{s-1} = Pi (f_s * beta_s), s = t .. t-L+1
                            from a ring of the last L likelihood vectors and L + 1 filtered vectors
    spherical covariance    S_l = s_l I, `s_l' = a s_l + a b |delta|^2 / d`  (a scalar a state, not d(d+1)/2)
    partial learning        `params` letters s t m c: which of start / transitions / means / covariances learn
    prior weights           given means worth `kappa` rows; given variances `covars_prior`, `covars_weight`
                            (hmmlearn's `means_weight`, `means_prior`, `covars_prior`, `covars_weight`)
"""

from __future__ import annotations

import math
import warnings
from collections import deque

import numpy as np
from hmmlearn import hmm as hl

from .common import table


def logdens(x, mu, cov, kind):
    d = len(x)
    if kind == "spherical":
        s = float(cov)
        return -0.5 * (d * math.log(2 * math.pi * s) + float(np.sum((x - mu) ** 2)) / s)
    sign, ld = np.linalg.slogdet(cov)
    r = x - mu
    return -0.5 * (d * math.log(2 * math.pi) + ld + float(r @ np.linalg.solve(cov, r)))


class Filter:
    """Hamilton filter + fixed-lag smoother at fixed parameters."""

    def __init__(self, Pi, means, covs, kind, lag=0):
        self.Pi, self.means, self.covs, self.kind, self.lag = np.asarray(Pi, float), means, covs, kind, lag
        self.K = len(means)
        self.p = np.full(self.K, 1.0 / self.K)
        self.fring = deque(maxlen=lag)  # likelihood vectors (K each), newest last
        self.pring = deque(maxlen=lag + 1)  # filtered vectors, newest last
        self.rows = 0

    def step(self, x):
        pt = self.p @ self.Pi
        lf = np.array([logdens(x, self.means[l], self.covs[l], self.kind) for l in range(self.K)])
        mx = lf.max()
        f = np.exp(lf - mx)
        s = float(pt @ f)
        ll = math.log(s) + mx
        self.p = pt * f / s
        self.fring.append(f)
        self.pring.append(self.p.copy())
        self.rows += 1
        sm = None
        if self.lag and len(self.pring) == self.lag + 1:
            beta = np.ones(self.K)
            for f_s in list(self.fring)[::-1]:  # s = t .. t-L+1
                beta = self.Pi @ (f_s * beta)
                beta = beta / beta.sum()
            g = self.pring[0] * beta
            sm = g / g.sum()
        return ll, self.p.copy(), pt, sm

    def state_bytes(self):
        return 8 * (self.K * (2 * self.lag + 1) + self.K)  # f ring L K, p ring (L+1) K, p


class FilterAdv(Filter):
    """The half-size form: the likelihood ring and ONE filtered vector, advanced a row at a time (`8 K (L + 1)` bytes)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.alpha_old = np.full(self.K, 1.0 / self.K)

    def step(self, x):
        pt = self.p @ self.Pi
        lf = np.array([logdens(x, self.means[l], self.covs[l], self.kind) for l in range(self.K)])
        mx = lf.max()
        f = np.exp(lf - mx)
        s = float(pt @ f)
        ll = math.log(s) + mx
        self.p = pt * f / s
        if len(self.fring) == self.lag:  # alpha_{t-L} from alpha_{t-L-1} and f_{t-L}
            a = (self.alpha_old @ self.Pi) * self.fring[0]
            self.alpha_old = a / a.sum()
        self.fring.append(f)
        sm = None
        if self.lag and len(self.fring) == self.lag and self.rows >= self.lag:
            beta = np.ones(self.K)
            for f_s in list(self.fring)[::-1]:
                beta = self.Pi @ (f_s * beta)
                beta = beta / beta.sum()
            g = self.alpha_old * beta
            sm = g / g.sum()
        self.rows += 1
        return ll, self.p.copy(), pt, sm

    def state_bytes(self):
        return 8 * self.K * (self.lag + 1) + 8 * self.K


def make_model(K, d, kind):
    rng = np.random.default_rng(5)
    Pi = rng.dirichlet(np.ones(K) * 4, size=K) * 0.4 + 0.6 * np.eye(K)
    Pi /= Pi.sum(1, keepdims=True)
    means = rng.standard_normal((K, d)) * 1.5
    if kind == "spherical":
        covs = rng.uniform(0.5, 1.5, K)
    else:
        covs = np.stack([(lambda a: a @ a.T + 0.3 * np.eye(d))(rng.standard_normal((d, d)) * 0.5) for _ in range(K)])
    return Pi, means, covs


def sample(Pi, means, covs, kind, n, seed):
    rng = np.random.default_rng(seed)
    K, d = means.shape
    z = np.empty(n, int)
    z[0] = rng.integers(K)
    for t in range(1, n):
        z[t] = rng.choice(K, p=Pi[z[t - 1]])
    X = np.empty((n, d))
    for t in range(n):
        if kind == "spherical":
            X[t] = means[z[t]] + math.sqrt(covs[z[t]]) * rng.standard_normal(d)
        else:
            X[t] = rng.multivariate_normal(means[z[t]], covs[z[t]])
    return X, z


def hmm_fixed(Pi, means, covs, kind):
    K, d = means.shape
    m = hl.GaussianHMM(n_components=K, covariance_type=kind, params="", init_params="")
    m.n_features = d
    m.startprob_ = Pi.sum(0) / K  # `p~` of the first row is (1/K) 1' Pi
    m.transmat_ = Pi
    m.means_ = means
    m.covars_ = covs
    return m


def report(seeds: int = 20) -> None:
    warnings.filterwarnings("ignore")
    # ---------------- A. filter and fixed-lag smoother at fixed parameters vs hmmlearn
    rows = []
    for kind in ("full", "spherical"):
        for K, d in ((2, 2), (3, 3)):
            g_ll = g_sm = g_pre = g_adv = 0.0
            for sd in range(max(4, seeds // 4)):
                Pi, mu, cv = make_model(K, d, kind)
                X, _ = sample(Pi, mu, cv, kind, 150, sd)
                # zero-weight rows: the filter skips them; the oracle sees the scored rows
                rng = np.random.default_rng(sd + 1)
                w = (rng.random(150) >= 0.1)
                w[0] = False
                Xs = X[w]
                m = hmm_fixed(Pi, mu, cv, kind)
                lags = (0, 1, 3, 8)
                flt = {L: Filter(Pi, mu, cv, kind, lag=L) for L in lags}
                adv = {L: FilterAdv(Pi, mu, cv, kind, lag=L) for L in lags if L}
                tot = {L: 0.0 for L in lags}
                prefix_ll = []
                for t in range(len(Xs)):
                    for L in lags:
                        ll, p, pt, sm = flt[L].step(Xs[t])
                        tot[L] += ll
                        if L:
                            _, _, _, sm2 = adv[L].step(Xs[t])
                            if sm is not None:
                                g_adv = max(g_adv, float(np.max(np.abs(sm - sm2))))
                        if sm is not None and t in (40, 80, len(Xs) - 1) and L > 0:
                            ref = m.predict_proba(Xs[: t + 1])[t - L]
                            g_sm = max(g_sm, float(np.max(np.abs(sm - ref))))
                        if L == 0 and t in (40, 80, len(Xs) - 1):
                            ref = m.predict_proba(Xs[: t + 1])[t]  # smoothed at lag 0 = the filtered p at the last row
                            g_pre = max(g_pre, float(np.max(np.abs(p - ref))))
                g_ll = max(g_ll, abs(tot[0] - m.score(Xs)) / abs(tot[0]))
            rows.append([f"{kind}, K = {K}, d = {d}", g_ll, g_pre, g_sm, g_adv])
    table("522a Hamilton filter and fixed-lag smoothing at fixed parameters vs hmmlearn `score` (summed `ln p(x_t | x_<t)`, relative), `predict_proba(X[:t+1])[t]` (the filtered p of the last row, absolute) "
          "and `predict_proba(X[:t+1])[t-L]` for L = 1, 3, 8 (absolute); `startprob_ = (1/K) 1'Pi`; 10% zero-weight rows skipped, a zero-weight first row",
          ["case", "loglik", "filtered p", "smoothed p (fixed lag)", "advance form vs two-ring form"], rows)

    # ---------------- B. spherical learning vs hmmlearn M-step on the same responsibilities
    # run the online filter + learner (means and spherical variance, no transition learning), recording responsibilities
    rows = []
    for K, d in ((2, 2), (3, 4)):
        g_m = g_c = 0.0
        for sd in range(max(4, seeds // 4)):
            Pi, mu, cv = make_model(K, d, "spherical")
            X, _ = sample(Pi, mu, cv, "spherical", 400, sd)
            out = learn_run(X, Pi, mu.copy(), cv.copy(), kappa=1e-12, cprior=0.0, cweight=0.0, half_life=math.inf)
            means_on, var_on, gam = out["means"], out["var"], out["gam"]
            # hmmlearn's M-step on the same responsibilities: stats post / obs / obs**2
            ref_m, ref_c = hmm_mstep(X, gam, mu, kappa=1e-12, mprior=mu, cprior=0.0, cweight=0.0)
            g_m = max(g_m, float(np.max(np.abs(means_on - ref_m)) / np.max(np.abs(ref_m))))
            g_c = max(g_c, float(np.max(np.abs(var_on - ref_c[:, 0])) / np.max(ref_c[:, 0])))
        rows.append([f"K = {K}, d = {d}, no prior (weight ~0)", g_m, g_c])
    table("522b spherical state learned online (mean form at weight `w p_l`, `s' = a s + a b |delta|^2/d`) vs hmmlearn's `_do_mstep` (`covariance_type=\"spherical\"`, `params=\"mc\"`) on the same recorded responsibilities (relative, 400 rows)",
          ["case", "means", "spherical variance"], rows)

    # ---------------- C. prior weights vs hmmlearn
    rows = []
    for kappa, cprior, cweight in ((1.0, 0.01, 1.0), (5.0, 0.2, 3.0), (20.0, 1.0, 10.0)):
        g_m = g_c = 0.0
        for sd in range(max(4, seeds // 4)):
            K, d = 3, 3
            Pi, mu, cv = make_model(K, d, "spherical")
            X, _ = sample(Pi, mu, cv, "spherical", 400, sd)
            out = learn_run(X, Pi, mu.copy(), cv.copy(), kappa=kappa, cprior=cprior, cweight=cweight, half_life=math.inf)
            ref_m, ref_c = hmm_mstep(X, out["gam"], mu, kappa=kappa, mprior=mu, cprior=cprior, cweight=cweight)
            g_m = max(g_m, float(np.max(np.abs(out["means"] - ref_m)) / np.max(np.abs(ref_m))))
            g_c = max(g_c, float(np.max(np.abs(out["var"] - ref_c[:, 0])) / np.max(ref_c[:, 0])))
        rows.append([f"means_weight = {kappa}, covars_prior = {cprior}, covars_weight = {cweight}", g_m, g_c])
    table("522c prior weights for given states: given means worth `kappa` rows (mean form started at `W = kappa`), given variances `covars_prior` / `covars_weight`, vs hmmlearn's `means_prior` / `means_weight` / `covars_prior` / `covars_weight` M-step on the same responsibilities (relative, `half_life = inf`)",
          ["setting", "means", "spherical variance"], rows)

    # decay: the prior fades
    out = learn_run(sample(*make_model(2, 2, "spherical"), "spherical", 600, 3)[0], *make_model(2, 2, "spherical"), kappa=20.0, cprior=0.0, cweight=0.0, half_life=30.0)
    print(f"\n522c' under a half-life 30 the prior mean weight fades as `kappa 2^(-t/30)`: after 600 rows state weights {np.round(out['W'], 3)} (prior share {20.0 * 2 ** (-600 / 30):.1e}); "
          "so the prior is a starting point here, where hmmlearn's is permanent (the same distinction `hmm` already makes for given states).")

    # ---------------- D. partial learning: params letters
    rows = []
    for letters in ("t", "m", "c", "mc", "tmc"):
        errs = {"online": [], "hmmlearn": []}
        for sd in range(max(3, seeds // 5)):
            K, d = 2, 2
            Pi, mu, cv = make_model(K, d, "spherical")
            Pi = np.array([[0.9, 0.1], [0.2, 0.8]])
            mu = np.array([[-1.5, -1.5], [1.5, 1.5]])
            cv = np.array([1.0, 1.0])
            X, z = sample(Pi, mu, cv, "spherical", 4000, 100 + sd)
            # online, starting from a perturbed copy of the truth, learning the `letters` only
            Pi0 = np.array([[0.7, 0.3], [0.4, 0.6]])
            mu0 = mu + 0.6
            cv0 = cv * 1.8
            out = learn_run(X, Pi0 if "t" in letters else Pi, mu0 if "m" in letters else mu, cv0 if "c" in letters else cv, kappa=1.0, cprior=0.0, cweight=0.0,
                            half_life=math.inf, learn_t="t" in letters, learn_m="m" in letters, learn_c="c" in letters, tau=1.0)
            # frozen parts must be untouched
            if "t" not in letters:
                assert np.array_equal(out["Pi"], Pi)
            if "m" not in letters:
                assert np.array_equal(out["means"], mu)
            if "c" not in letters:
                assert np.array_equal(out["var"], cv)
            e_on = 0.0
            if "t" in letters:
                e_on = max(e_on, float(np.max(np.abs(out["Pi"] - Pi))))
            if "m" in letters:
                e_on = max(e_on, float(np.max(np.abs(out["means"] - mu))))
            if "c" in letters:
                e_on = max(e_on, float(np.max(np.abs(out["var"] - cv))))
            m = hl.GaussianHMM(n_components=2, covariance_type="spherical", params=letters, init_params="", n_iter=100, tol=1e-6)
            m.n_features = d
            m.startprob_ = np.array([0.5, 0.5])
            m.transmat_ = Pi0 if "t" in letters else Pi
            m.means_ = mu0 if "m" in letters else mu
            # hmmlearn 0.3.3: a spherical `covars_` set as (K,) breaks EM (the M-step stores (K, d)); set the tiled form
            m._covars_ = np.tile((cv0 if "c" in letters else cv)[:, None], (1, d))
            if letters == "c":  # hmmlearn 0.3.3 accumulates `post` and `obs` only when 'm' is in params: `params="c"` diverges
                errs["online"].append(0.0)
                errs["hmmlearn"].append(float("nan"))
                errs["online"][-1] = max(abs(out["var"] - cv))
                continue
            m.fit(X)
            e_hl = 0.0
            if "t" in letters:
                e_hl = max(e_hl, float(np.max(np.abs(m.transmat_ - Pi))))
            if "m" in letters:
                e_hl = max(e_hl, float(np.max(np.abs(m.means_ - mu))))
            if "c" in letters:
                e_hl = max(e_hl, float(np.max(np.abs(m._covars_[:, 0] - cv))))
            errs["online"].append(e_on)
            errs["hmmlearn"].append(e_hl)
        hv = float(np.mean(errs["hmmlearn"]))
        rows.append([f"params = {letters!r}", float(np.mean(errs["online"])), "defect: diverges (below)" if math.isnan(hv) else hv])
    table("522d partial learning: the learned parameters' largest error against the generating values after 4000 rows (mean over 4 streams; start from a perturbed copy of the truth): one online pass learning only the letters, and hmmlearn's EM (`params=letters`, 100 iterations, which scores the filtered-vs-smoothed gap). The frozen parts are asserted bit-equal to what was given",
          ["letters", "online pass", "hmmlearn EM"], rows)

    # state
    srows = []
    for K, L in ((2, 5), (3, 20), (5, 100), (8, 1000)):
        srows.append([f"fixed-lag smoother, K = {K}, lag {L}: two rings / advance form", f"{Filter(np.eye(K), np.zeros((K, 1)), np.ones(K), 'spherical', L).state_bytes()} / {FilterAdv(np.eye(K), np.zeros((K, 1)), np.ones(K), 'spherical', L).state_bytes()}"])
    for d in (2, 10, 50):
        srows.append([f"covariance per state, d = {d}: full / diag / spherical bytes", f"{8 * d * (d + 1) // 2} / {8 * d} / 8"])
    table("522e state bytes (smoother: `8 K (2 L + 2)`: the likelihood ring and the filtered ring and `p`)", ["feature", "bytes"], srows)


def learn_run(X, Pi, means, var, *, kappa, cprior, cweight, half_life, learn_t=False, learn_m=True, learn_c=True, tau=1.0):
    """Online filter + learner for the spherical hmm. Returns the parameters and the recorded responsibilities."""
    K, d = means.shape
    lam = 1.0 if math.isinf(half_life) else 0.5 ** (1.0 / half_life)
    mean = means.copy()
    W = np.full(K, kappa)  # state weight incl. the prior mass
    G = np.zeros(K)  # data mass alone (the covariance's divisor)
    C = np.zeros(K)  # sum of squared deviations / d, mean form: C = M2 / W, kept as a mean (divisor W)
    A = np.zeros((K, K))
    p = np.full(K, 1.0 / K)
    gam = []
    Pi_cur = Pi.copy()
    V = var.copy()
    prior_var_term = cprior  # per feature, added to the numerator
    cvd0 = max(cweight - 1.0, 0.0)
    for t in range(len(X)):
        x = X[t]
        Pi_cur = (A + tau) / (A + tau).sum(1, keepdims=True) if learn_t else Pi
        pt = p @ Pi_cur
        lf = np.array([logdens(x, mean[l], V[l], "spherical") for l in range(K)])
        f = np.exp(lf - lf.max())
        pn = pt * f / (pt @ f)
        if learn_t:
            xi = (p[:, None] * Pi_cur) * f[None, :]
            xi /= xi.sum()
            A = lam * A + xi
        gam.append(pn.copy())
        for l in range(K):
            w = pn[l]
            Wo = lam * W[l]
            W2 = Wo + w
            if learn_m or learn_c:
                a = Wo / W2 if W2 > 0 else 0.0
                b = w / W2 if W2 > 0 else 0.0
                dl = x - mean[l]
                if learn_c:
                    C[l] = a * C[l] + a * b * float(dl @ dl) / d
                if learn_m:
                    mean[l] = mean[l] + b * dl
            W[l] = W2
            G[l] = lam * G[l] + w
            if learn_c:
                # variance = (prior + W C)/(max(cw-1,0) + G)   [per-feature mean of hmmlearn's (covars_prior + c_n)/c_d]
                V[l] = (prior_var_term + W[l] * C[l]) / max(cvd0 + G[l], 1e-5) if (cprior > 0 or cweight > 0 or True) else V[l]
        p = pn
    return {"means": mean, "var": V, "gam": np.array(gam), "Pi": Pi_cur, "W": W}


def hmm_mstep(X, gam, mu0, *, kappa, mprior, cprior, cweight):
    K, d = mu0.shape
    m = hl.GaussianHMM(n_components=K, covariance_type="spherical", params="mc", init_params="",
                       means_prior=mprior, means_weight=kappa, covars_prior=cprior, covars_weight=cweight)
    m.n_features = d
    m.startprob_ = np.full(K, 1.0 / K)
    m.transmat_ = np.full((K, K), 1.0 / K)
    m.means_ = mu0.copy()
    m.covars_ = np.ones(K)
    stats = {
        "nobs": 1,
        "start": np.zeros(K),
        "trans": np.zeros((K, K)),
        "post": gam.sum(0),
        "obs": gam.T @ X,
        "obs**2": gam.T @ (X**2),
    }
    m._do_mstep(stats)
    return m.means_, m._covars_
