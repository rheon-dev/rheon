"""Study 506: a Gaussian mixture by exponentially weighted online EM.

numpy only (scipy for the Hungarian assignment and a chi-square quantile,
scikit-learn for the batch oracle). No data files: every stream is drawn
from a seeded generator (5 seeds, 0 to 4, unless ``--seeds`` says otherwise).

    .venv/bin/python scripts/study_506.py --quick          # every check, short; < 2 min
    .venv/bin/python scripts/study_506.py                  # every check, full; ~15 min
    .venv/bin/python scripts/study_506.py decay collapse   # named sections only
    .venv/bin/python scripts/study_506.py --cell full 20 8 20000   # one cell of `sklearn`

Sections, in order: ``compare`` (k = 2, K = 2 against scikit-learn),
``passes`` (repeated passes), ``zero`` (zero-weight rows, without and with
decay), ``timing``, ``decay`` (against the weighted EM longhand),
``sklearn`` (larger k and K), ``collapse`` (a component's weight going to
zero), ``state`` (bytes and flops), ``tolerances`` (the proposed test
tolerances). The full ``sklearn`` section's K = 8, k = 20 cells take
minutes each: run the full study under ``nohup``.

Each section prints its table and records the study's findings as checks
(the forms agree, separated components stay within the batch SE, the sum
form fails under collapse, the stabilised floor revives, ...); the run ends
with every check's result and exits non-zero if one failed.

Notation follows rheon: ``k`` is the number of features, ``K`` the number of
mixture components. The notes are docs/records/STUDY-506-MIXTURES.md.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

LOG_2PI = math.log(2.0 * math.pi)
REG = 1e-6
W0 = 1.0  # the initialisation enters at one row's weight, as hmm's means=/covs=
TAU = 0.01  # the stabilised floor's pseudo-count

CHECKS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


# --------------------------------------------------------------------------
# The recursion (Cappe and Moulines 2009, sufficient statistics, with a
# forgetting factor in place of their step size), vectorised over components.
# --------------------------------------------------------------------------
class Mix:
    """Sum form: per component S0 (scalar), S1 (k), S2 (k x k, or k squares).

    Row (x, w), with lam the decay over the row's clock:
      E: r_j = pi_j N(x; mu_j, Sigma_j) / sum_l pi_l N(x; mu_l, Sigma_l),
         the parameters read from the state *before* the row.
      S0_j <- lam S0_j + w r_j
      S1_j <- lam S1_j + w r_j x
      S2_j <- lam S2_j + w r_j x x'
    M (read on demand, never stored):
      pi_j = S0_j / sum S0,  mu_j = S1_j / S0_j,
      Sigma_j = S2_j / S0_j - mu_j mu_j' + reg I.

    cov      'full' (S2: K x k x k) or 'diag' (S2: K x k, the squares only).
    guard    'fallback': a component with S0_j = 0 reads the initial
             parameters, a state with sum S0 = 0 the initial weights;
             'none': the bare M-step, S1/S0 and S2/S0 - mu mu' as written.
    tau      stabilised floor: tau pseudo-rows per component at the prior
             (the initial parameters), which do NOT decay:
               pi_j  = (S0_j + tau) / (sum S0 + K tau)
               mu_j  = (S1_j + tau m0_j) / (S0_j + tau)
               Sig_j = (S2_j + tau (C0_j + m0_j m0_j')) / (S0_j + tau)
                       - mu_j mu_j' + reg I
    fading   with tau > 0: the pseudo-rows decay with the statistics
             (hmm's precision_scale form), tau_t = tau lam^t.
    reseed   (eps, seed_w): after a row, a component with pi_j < eps is
             reseeded at that row when the row is an outlier to every
             component (min Mahalanobis^2 > chi2 quantile 1 - 1e-6): S0_j =
             seed_w, mean x, covariance the pooled one.
    A row with w = 0 (or a refused row: w < 0, non-finite w or x) does no
    E-step at all and only decays.
    """

    def __init__(
        self,
        init,
        lam=1.0,
        reg=REG,
        cov="full",
        w0=W0,
        guard="fallback",
        tau=0.0,
        fading=False,
        reseed=None,
    ):
        w, m, c = (np.asarray(a, float).copy() for a in init)
        self.K, self.k = m.shape
        self.cov, self.lam, self.reg = cov, float(lam), float(reg)
        if cov == "diag" and c.ndim == 3:
            c = np.einsum("jaa->ja", c).copy()
        self.fb_w, self.fb_m, self.fb_c = w, m, c
        self.guard, self.tau, self.fading, self.reseed = guard, float(tau), fading, reseed
        self.tau_t = self.tau
        s0 = w0 * w
        self.S0 = s0.copy()
        self.S1 = s0[:, None] * m
        self.S2 = (
            s0[:, None, None] * (c + m[:, :, None] * m[:, None, :])
            if cov == "full"
            else s0[:, None] * (c + m * m)
        )
        self.reseeds = 0
        if reseed:
            self.chi_cut = chi2.ppf(1 - 1e-6, self.k)

    def stats(self):
        return [self.S0, self.S1, self.S2]

    def _outer(self, m):
        return m[:, :, None] * m[:, None, :] if self.cov == "full" else m * m

    def params(self):
        S0, S1, S2 = self.S0, self.S1, self.S2
        if self.tau_t > 0:
            t = self.tau_t
            S0 = S0 + t
            S1 = S1 + t * self.fb_m
            S2 = S2 + t * (self.fb_c + self._outer(self.fb_m))
        tot = S0.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.guard == "none":
                w = S0 / tot
                m = S1 / S0[:, None]
                c = S2 / (S0[:, None, None] if self.cov == "full" else S0[:, None]) - self._outer(m)
            else:
                w = S0 / tot if tot > 0 else self.fb_w.copy()
                ok = S0 > 0
                m = self.fb_m.copy()
                c = self.fb_c.copy()
                m[ok] = S1[ok] / S0[ok, None]
                if self.cov == "full":
                    c[ok] = S2[ok] / S0[ok, None, None] - self._outer(m[ok])
                else:
                    c[ok] = S2[ok] / S0[ok, None] - m[ok] ** 2
        c = c + self.reg * np.eye(self.k) if self.cov == "full" else c + self.reg
        return w, m, c

    def logdens(self, x, w, m, c):
        """log pi_j + log N(x; mu_j, Sigma_j), and the Mahalanobis^2."""
        d = x - m
        if self.cov == "full":
            L = np.linalg.cholesky(c)  # raises on a matrix that is not PD
            z = np.linalg.solve(L, d[:, :, None])[:, :, 0]
            q = (z * z).sum(1)
            ld = np.log(np.einsum("jaa->ja", L)).sum(1)
        else:
            q = (d * d / c).sum(1)
            ld = 0.5 * np.log(c).sum(1)
        with np.errstate(divide="ignore"):
            return np.log(w) - 0.5 * (self.k * LOG_2PI + q) - ld, q

    def step(self, x, weight, lam=None):
        lam = self.lam if lam is None else lam
        if not (weight >= 0 and np.isfinite(weight) and np.all(np.isfinite(x))):
            weight = 0.0  # a refused row: the clock only
        if weight == 0.0:
            self.S0 *= lam
            self.S1 *= lam
            self.S2 *= lam
            if self.fading:
                self.tau_t *= lam
            return None
        w, m, c = self.params()
        lp, q = self.logdens(x, w, m, c)
        e = np.exp(lp - lp.max())
        r = e / e.sum()
        wr = weight * r
        self.S0 = lam * self.S0 + wr
        self.S1 = lam * self.S1 + wr[:, None] * x
        if self.cov == "full":
            self.S2 = lam * self.S2 + wr[:, None, None] * np.outer(x, x)
        else:
            self.S2 = lam * self.S2 + wr[:, None] * (x * x)
        if self.fading:
            self.tau_t *= lam
        if self.reseed:
            eps, sw = self.reseed
            pi = self.S0 / self.S0.sum()
            dead = np.flatnonzero(pi < eps)
            if dead.size and q.min() > self.chi_cut:
                j = dead[0]
                live = pi >= eps
                pooled = np.tensordot(w[live] / w[live].sum(), c[live], 1) - (
                    self.reg * np.eye(self.k) if self.cov == "full" else self.reg
                )
                self.S0[j] = sw
                self.S1[j] = sw * x
                self.S2[j] = sw * (pooled + (np.outer(x, x) if self.cov == "full" else x * x))
                self.reseeds += 1
        return r


class MixMean(Mix):
    """The same recursion held as (W_j, mu_j, C_j), hmm's `EwCov` form, full
    covariance:  w_j = w r_j,  W' = lam W + w_j,  a = lam W / W',
    b = w_j / W',  d = x - mu,  mu' = mu + b d,  C' = a C + a b d d'.
    Algebraically S0 = W, S1 = W mu, S2 = W (C + mu mu'); free of the
    S2 / S0 - mu mu' cancellation. W' <= 0 skips the component.

    tau: the stabilised floor read in centred coordinates (the sum form's
    read, rearranged so nothing cancels), with T = W_j + tau, d = mu_j - m0_j:
      pi_j  = (W_j + tau) / (sum W + K tau)
      mu_j' = (W_j mu_j + tau m0_j) / T
      Sig_j = (W_j C_j + tau C0_j) / T + (W_j tau / T^2) d d' + reg I
    """

    def __init__(self, init, lam=1.0, reg=REG, w0=W0, tau=0.0):
        w, m, c = (np.asarray(a, float).copy() for a in init)
        self.K, self.k = m.shape
        self.cov, self.lam, self.reg, self.tau = "full", float(lam), float(reg), float(tau)
        self.fb_w, self.fb_m, self.fb_c = w, m.copy(), c.copy()
        self.W, self.mu, self.C = w0 * w, m, c

    def stats(self):
        return [self.W, self.mu, self.C]

    def params(self):
        if self.tau > 0:
            W, t = self.W, self.tau
            T = W + t
            d = self.mu - self.fb_m
            w = T / T.sum()
            mu = (W[:, None] * self.mu + t * self.fb_m) / T[:, None]
            C = (W[:, None, None] * self.C + t * self.fb_c) / T[:, None, None] + (W * t / T**2)[
                :, None, None
            ] * (d[:, :, None] * d[:, None, :])
            return w, mu, C + self.reg * np.eye(self.k)
        tot = self.W.sum()
        w = self.W / tot if tot > 0 else self.fb_w.copy()
        return w, self.mu.copy(), self.C + self.reg * np.eye(self.k)

    def step(self, x, weight, lam=None):
        lam = self.lam if lam is None else lam
        if not (weight >= 0 and np.isfinite(weight) and np.all(np.isfinite(x))):
            weight = 0.0
        if weight == 0.0:
            self.W *= lam  # mu, C: ratios, unmoved
            return None
        w, m, c = self.params()
        lp, _ = self.logdens(x, w, m, c)
        e = np.exp(lp - lp.max())
        r = e / e.sum()
        for j in range(self.K):
            wj = weight * r[j]
            Wn = lam * self.W[j] + wj
            if Wn <= 0:
                continue
            a, b = lam * self.W[j] / Wn, wj / Wn
            d = x - self.mu[j]
            self.mu[j] = self.mu[j] + b * d
            self.C[j] = a * self.C[j] + a * b * np.outer(d, d)
            self.W[j] = Wn
        return r


# --------------------------------------------------------------------------
# Data, initialisation, oracles
# --------------------------------------------------------------------------
SCENARIOS = {
    # k = 2, K = 2. Means 4.0 and 3.4 standard deviations apart along each axis.
    "separated": dict(
        w=[0.35, 0.65],
        m=[[0.0, 0.0], [4.0, 3.0]],
        c=[[[1.0, 0.5], [0.5, 1.0]], [[1.5, -0.4], [-0.4, 0.8]]],
    ),
    # The same shapes, means about one standard deviation apart.
    "overlapping": dict(
        w=[0.35, 0.65],
        m=[[0.0, 0.0], [1.5, 1.0]],
        c=[[[1.0, 0.5], [0.5, 1.0]], [[1.5, -0.4], [-0.4, 0.8]]],
    ),
}


def truth(scn):
    return np.array(scn["w"]), np.array(scn["m"]), np.array(scn["c"])


def make_mixture(k, K, seed, cov="full", min_dist=4.0):
    """Separated components: covariance eigenvalues in [0.5, 1.5] (so a
    standard deviation of at most 1.22), means at least min_dist apart
    (3.3 SD or more; the k = 2 separated pair is 4 SD), weights
    proportional to 1 + j/(K-1) (max/min = 2)."""
    rng = np.random.default_rng(10_000 + seed)
    side = max(2.0 * min_dist, min_dist * K ** (1.0 / k) * 1.5)
    while True:
        m = rng.uniform(0, side, size=(K, k))
        dist = np.sqrt(((m[:, None] - m[None]) ** 2).sum(-1)) + np.eye(K) * 1e9
        if dist.min() >= min_dist:
            break
    c = np.empty((K, k, k))
    for j in range(K):
        ev = rng.uniform(0.5, 1.5, size=k)
        if cov == "full":
            q, _ = np.linalg.qr(rng.standard_normal((k, k)))
            c[j] = (q * ev) @ q.T
        else:
            c[j] = np.diag(ev)
    w = 1.0 + np.arange(K) / max(K - 1, 1)
    return w / w.sum(), m, c


def draw(mix, n, seed):
    """n i.i.d. rows of the mixture (w, m, c)."""
    w, m, c = mix
    rng = np.random.default_rng(seed)
    z = rng.choice(len(w), size=n, p=w)
    x = np.empty((n, m.shape[1]))
    for j in range(len(w)):
        idx = z == j
        x[idx] = rng.multivariate_normal(m[j], c[j], size=idx.sum())
    return x


def kmeans_init(x, K, seed, cov="full"):
    """scikit-learn's own `init_params='kmeans'`: one k-means run, hard
    labels, one M-step. Weights, means, raw covariances (no reg)."""
    lab = KMeans(n_clusters=K, n_init=1, random_state=seed).fit(x).labels_
    n, k = x.shape
    w, m, c = np.empty(K), np.empty((K, k)), np.empty((K, k, k))
    for j in range(K):
        xj = x[lab == j]
        w[j] = len(xj) / n
        m[j] = xj.mean(0)
        d = xj - m[j]
        c[j] = d.T @ d / len(xj)
        if cov == "diag":
            c[j] = np.diag(np.diag(c[j]))
    return w, m, c


def sk_fit(x, init, reg, cov, seed, n_init=None, tol=1e-10):
    """scikit-learn's GaussianMixture from `init` (weights, means, raw
    covariances; reg added to form the precisions), or its own best of
    n_init starts. Returns the fit (full covariances), converged, mean
    log-likelihood."""
    k = x.shape[1]
    kw = dict(
        n_components=len(init[0]),
        covariance_type=cov,
        reg_covar=reg,
        tol=tol,
        max_iter=5000,
        random_state=seed,
    )
    if n_init is None:
        w, m, c = init
        if cov == "full":
            prec = np.linalg.inv(c + reg * np.eye(k))
        else:
            prec = 1.0 / (np.einsum("jaa->ja", c) + reg)
        gm = GaussianMixture(weights_init=w, means_init=m, precisions_init=prec, **kw).fit(x)
    else:
        gm = GaussianMixture(n_init=n_init, **kw).fit(x)
    cv = gm.covariances_
    if cov == "diag":
        cv = np.array([np.diag(v) for v in cv])
    return (gm.weights_, gm.means_, cv), gm.converged_, gm.score(x)


def weighted_em(x, v, init, reg, cov="full", max_iter=20000, tol=1e-11):
    """The longhand oracle: batch EM over every row with fixed row weights
    v_t (lam^(t_end - t) under decay), responsibilities recomputed each
    iteration, run until no parameter moves by more than tol."""
    w, m, c = (np.asarray(a, float).copy() for a in init)
    K, k = m.shape
    c = c + reg * np.eye(k)
    for it in range(max_iter):
        lp = np.empty((len(x), K))
        for j in range(K):
            d = x - m[j]
            if cov == "full":
                L = np.linalg.cholesky(c[j])
                z = np.linalg.solve(L, d.T)
                lp[:, j] = (
                    np.log(w[j]) - 0.5 * (k * LOG_2PI + (z * z).sum(0)) - np.log(np.diag(L)).sum()
                )
            else:
                dv = np.diag(c[j])
                lp[:, j] = (
                    np.log(w[j])
                    - 0.5 * (k * LOG_2PI + (d * d / dv).sum(1))
                    - 0.5 * np.log(dv).sum()
                )
        top = lp.max(1, keepdims=True)
        r = np.exp(lp - top)
        r /= r.sum(1, keepdims=True)
        vr = r * v[:, None]
        nk = vr.sum(0)
        w2 = nk / nk.sum()
        m2 = (vr.T @ x) / nk[:, None]
        c2 = np.empty_like(c)
        for j in range(K):
            d = x - m2[j]
            cj = (vr[:, j, None] * d).T @ d / nk[j]
            c2[j] = (cj if cov == "full" else np.diag(np.diag(cj))) + reg * np.eye(k)
        move = max(np.abs(w2 - w).max(), np.abs(m2 - m).max(), np.abs(c2 - c).max())
        w, m, c = w2, m2, c2
        if move < tol:
            return (w, m, c), it + 1
    return (w, m, c), -1


def match(ref_means, means):
    """The Hungarian assignment on squared mean distances."""
    cost = ((ref_means[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    return col


def gaps(ref, est, n):
    """max |est - ref| over components and entries, in units of the batch
    standard error with n_j = n pi_j: mean sqrt(Sigma_aa / n_j), covariance
    entry sqrt((S_aa S_bb + S_ab^2) / n_j) (Gaussian), weight
    sqrt(pi(1-pi)/n). Also the absolute gaps."""
    rw, rm, rc = ref
    perm = match(rm, est[1])
    ew, em, ec = est[0][perm], est[1][perm], est[2][perm]
    nj = n * rw
    se_m = np.sqrt(np.einsum("jaa->ja", rc) / nj[:, None])
    dg = np.einsum("jaa->ja", rc)
    se_c = np.sqrt((dg[:, :, None] * dg[:, None, :] + rc**2) / nj[:, None, None])
    se_w = np.sqrt(rw * (1 - rw) / n)
    return dict(
        mean=float(np.max(np.abs(em - rm) / se_m)),
        cov=float(np.max(np.abs(ec - rc) / se_c)),
        weight=float(np.max(np.abs(ew - rw) / se_w)),
        mean_abs=float(np.max(np.abs(em - rm))),
        cov_abs=float(np.max(np.abs(ec - rc))),
        weight_abs=float(np.max(np.abs(ew - rw))),
    )


def as_full(p, cov):
    w, m, c = p
    if cov == "diag":
        c = np.array([np.diag(v) for v in c])
    return w, m, c


def run_stream(g, x):
    for row in x:
        g.step(row, 1.0)
    return g.params()


def mean_loglik(x, params):
    w, m, c = params
    lp = np.empty((len(x), len(w)))
    for j in range(len(w)):
        L = np.linalg.cholesky(c[j])
        z = np.linalg.solve(L, (x - m[j]).T)
        lp[:, j] = (
            np.log(w[j]) - 0.5 * (x.shape[1] * LOG_2PI + (z * z).sum(0)) - np.log(np.diag(L)).sum()
        )
    top = lp.max(1, keepdims=True)
    return float(np.mean(top[:, 0] + np.log(np.exp(lp - top).sum(1))))


def summ(vals):
    v = np.asarray(vals)
    return f"{np.median(v):.3g} / {v.max():.3g}"


def summarise(rows, key):
    vals = {f: np.array([r[key][f] for r in rows]) for f in ("mean", "cov", "weight")}
    return "  ".join(f"{f} med {np.median(v):6.2f} max {v.max():6.2f}" for f, v in vals.items())


def maxf(rows, fields=("mean", "cov", "weight")):
    return max(r[f] for r in rows for f in fields)


def tol_of(x):
    """1.5 times x, rounded up to 0.05 (the previous studies' rule)."""
    return math.ceil(1.5 * x / 0.05 - 1e-9) * 0.05


# --------------------------------------------------------------------------
# compare: k = 2, K = 2 against scikit-learn (part 1)
# --------------------------------------------------------------------------
def run_compare(name, n, seed):
    scn = SCENARIOS[name]
    x = draw(truth(scn), n, seed)
    init = kmeans_init(x, 2, seed)
    ref, conv, _ = sk_fit(x, init, REG, "full", seed, tol=1e-12)
    assert conv
    sf, mf = Mix(init), MixMean(init)
    for row in x:
        sf.step(row, 1.0)
        mf.step(row, 1.0)
    p1 = sf.params()
    form_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(p1, mf.params(), strict=True))
    # Second pass, the statistics carried on (lam = 1: pass 1 keeps half the weight).
    p2 = run_stream(sf, x)
    # Second pass, the statistics restarted from pass 1's parameters at W0.
    p2r = run_stream(Mix((p1[0], p1[1], p1[2] - REG * np.eye(2))), x)
    return dict(
        batch_vs_truth=gaps(ref, truth(scn), n),
        pass1=gaps(ref, p1, n),
        pass2=gaps(ref, p2, n),
        pass2_restart=gaps(ref, p2r, n),
        form_diff=form_diff,
    )


def sec_compare(quick, seeds):
    ns = [2000] if quick else [2000, 20000]
    print("gaps: max over components and entries of |online - batch|, in batch SE;")
    print("median and max over the seeds")
    for name in SCENARIOS:
        for n in ns:
            t = time.perf_counter()
            rows = [run_compare(name, n, s) for s in seeds]
            print(f"-- {name}, k = 2, K = 2, N = {n}  ({time.perf_counter() - t:.1f} s)")
            for key in ("batch_vs_truth", "pass1", "pass2", "pass2_restart"):
                print(f"  {key:15s} {summarise(rows, key)}")
            absg = {f: max(r["pass1"][f + "_abs"] for r in rows) for f in ("mean", "cov", "weight")}
            print(f"  pass1 absolute max: {absg}")
            fd = max(r["form_diff"] for r in rows)
            check(f"sum and mean forms agree ({name}, N = {n})", fd < 1e-11, f"{fd:.1e}")
            p1 = [r["pass1"] for r in rows]
            if name == "separated":
                check(
                    f"separated, N = {n}: one pass within 1 batch SE of scikit-learn",
                    maxf(p1) < 1.0,
                    f"max {maxf(p1):.2f}",
                )
            else:
                med = float(np.median([r["mean"] for r in p1]))
                check(
                    f"overlapping, N = {n}: one pass is not within the batch SE (a limit)",
                    med > 3.0,
                    f"median mean gap {med:.2f}",
                )


def passes_curve(name, n, seed, n_passes):
    """Carried-on lam = 1 statistics over repeated passes: the gap to batch
    EM after each pass, and the mean log-likelihood shortfall per row."""
    x = draw(truth(SCENARIOS[name]), n, seed)
    init = kmeans_init(x, 2, seed)
    ref, _, _ = sk_fit(x, init, REG, "full", seed, tol=1e-12)
    ll_ref = mean_loglik(x, ref)
    g = Mix(init)
    out = []
    for _ in range(n_passes):
        p = run_stream(g, x)
        out.append((gaps(ref, p, n), ll_ref - mean_loglik(x, p)))
    return out


def sec_passes(quick, seeds):
    print("carried-on passes, N = 2000: median over seeds of the max gap (SE)")
    print("and of the batch's mean log-likelihood lead per row")
    for name in SCENARIOS:
        curves = [passes_curve(name, 2000, s, 8) for s in seeds]
        leads = []
        for i in (0, 1, 3, 7):
            gm = np.median([c[i][0]["mean"] for c in curves])
            gc = np.median([c[i][0]["cov"] for c in curves])
            gw = np.median([c[i][0]["weight"] for c in curves])
            dl = float(np.median([c[i][1] for c in curves]))
            leads.append(dl)
            print(
                f"  {name:11s} pass {i + 1}: mean {gm:5.2f} cov {gc:5.2f}"
                f" weight {gw:5.2f}  loglik lead {dl:.2e}"
            )
        if name == "separated":
            check(
                "separated: the log-likelihood lead falls with each doubling of passes",
                all(a > b for a, b in zip(leads, leads[1:], strict=False)),
                " > ".join(f"{v:.1e}" for v in leads),
            )


# --------------------------------------------------------------------------
# zero: zero-weight rows, without decay (part 1) and with it (part 2)
# --------------------------------------------------------------------------
def sec_zero(quick, seeds):
    # lam = 1, k = 2, K = 2 (part 1).
    init = kmeans_init(draw(truth(SCENARIOS["separated"]), 500, 0), 2, 0)
    x = draw(truth(SCENARIOS["separated"]), 200, 1)
    for cls in (Mix, MixMean):
        a, b = cls(init), cls(init)
        for i, row in enumerate(x):
            a.step(row, 1.0)
            b.step(row, 1.0)
            if i % 3 == 0:
                b.step(x[(i * 7) % len(x)], 0.0)
        same = all(np.array_equal(u, v) for u, v in zip(a.stats(), b.stats(), strict=True))
        check(f"{cls.__name__}, lam = 1: zero-weight rows leave the state bit-identical", same)
        g = cls(init, lam=0.99)
        g.step(x[0], np.nan)
        check(
            f"{cls.__name__}: a refused row (NaN weight) only ages the clock",
            np.all(np.isfinite(np.concatenate([np.ravel(u) for u in g.stats()]))),
        )
    # lam = 0.99, k = 3, K = 2 (part 2).
    lam = 0.99
    x = draw(make_mixture(3, 2, 1), 400, 1)
    init = kmeans_init(x, 2, 1)
    for name, mk in (
        ("sum", lambda w0: Mix(init, lam, w0=w0)),
        ("mean", lambda w0: MixMean(init, lam, w0=w0)),
        ("sum tau=0.01", lambda w0: Mix(init, lam, w0=w0, tau=TAU)),
    ):
        # (a) an empty state, a zero-weight first row: nothing moves, and the
        # stream after it is the stream without it, to the bit.
        a, b = mk(0.0), mk(0.0)
        a.step(x[0], 0.0)
        empty_zero = all(np.array_equal(u, np.zeros_like(u)) for u in a.stats()[:1])
        for row in x[1:]:
            a.step(row, 1.0)
            b.step(row, 1.0)
        empty_bits = all(np.array_equal(u, v) for u, v in zip(a.stats(), b.stats(), strict=True))
        nan_free = all(np.all(np.isfinite(p)) for p in a.params())
        # (b) an initialised state, a zero-weight first row: the clock advances
        # (each weight is lam times its value, to the bit).
        g = mk(W0)
        before = g.stats()[0].copy()
        p0 = g.params()
        g.step(x[0], 0.0)
        clock = np.array_equal(g.stats()[0], lam * before)
        pmove = max(float(np.abs(u - v).max()) for u, v in zip(p0, g.params(), strict=True))
        # (c) a zero-weight row after every third row: the same as the stream
        # without them with the decay lam^2 over each gap (the clock form).
        a, b = mk(W0), mk(W0)
        for i, row in enumerate(x):
            a.step(row, 1.0)
            if i % 3 == 0 and i < len(x) - 1:
                a.step(x[(7 * i) % len(x)], 0.0)
            b.step(row, 1.0, lam=lam * lam if (i > 0 and (i - 1) % 3 == 0) else lam)
        rel = max(  # worst entry, relative to its array's largest magnitude
            float(np.max(np.abs(u - v)) / np.max(np.abs(v)))
            for u, v in zip(a.stats(), b.stats(), strict=True)
        )
        print(
            f"  lam = {lam}, {name}: first-row parameter move {pmove:.2g},"
            f" interleaved vs lam^2 clock max rel {rel:.2g}, n_eff {a.stats()[0].sum():.1f}"
        )
        check(f"{name}: empty state, zero-weight first row stays exactly zero", empty_zero)
        check(f"{name}: and the stream after it is bit-identical, finite", empty_bits and nan_free)
        check(
            f"{name}: zero-weight row on a fitted state decays each weight by lam, to the bit",
            clock,
        )
        bound = 1e-4 if "tau" in name else 1e-13
        check(f"{name}: and moves the parameters by < {bound:g}", pmove < bound, f"{pmove:.2g}")
        if "tau" not in name:  # under tau the read sees the gap: the floor does not decay
            check(
                f"{name}: zero-weight rows = the lam^2 clock to 1e-13 relative",
                rel < 1e-13,
                f"{rel:.2g}",
            )


def sec_timing(quick, seeds):
    x = draw(truth(SCENARIOS["separated"]), 2000, 3)
    init = kmeans_init(x, 2, 3)
    for cls in (Mix, MixMean):
        g = cls(init)
        t = time.perf_counter()
        run_stream(g, x)
        print(
            f"  {cls.__name__}: {(time.perf_counter() - t) / len(x) * 1e6:.0f} us a row"
            " (k = 2, K = 2; interpreter bound)"
        )


# --------------------------------------------------------------------------
# decay: against the weighted EM longhand (part 2)
# --------------------------------------------------------------------------
def sec_decay(quick, seeds):
    ns = (2000,) if quick else (2000, 20000)
    x = draw(make_mixture(2, 2, 0), 2000, 0)
    init = kmeans_init(x, 2, 0)
    lh, _ = weighted_em(x, np.ones(len(x)), init, REG)
    sk, _, _ = sk_fit(x, init, REG, "full", 0)
    d = max(np.abs(a - b).max() for a, b in zip(lh, sk, strict=True))
    check("the longhand equals scikit-learn at lam = 1", d < 1e-12, f"{d:.1e}")
    print(
        "gap online vs longhand at the stream's end, in SE of the decay-weighted"
        " sample (Kish n = (sum v)^2 / sum v^2); median / max over the seeds;"
        f" means, covariances, weights; and with a stabilised tau = {TAU}"
    )
    meds = []
    for k in (2, 5):
        for lam in (0.999, 0.99):
            for n in ns:
                t0 = time.perf_counter()
                res, res_tau = [], []
                for s in seeds:
                    x = draw(make_mixture(k, 2, s), n, 100 + s)
                    init = kmeans_init(x, 2, s)
                    v = lam ** np.arange(n - 1, -1, -1, dtype=float)
                    nk = v.sum() ** 2 / (v * v).sum()
                    ref, _ = weighted_em(x, v, init, REG)
                    res.append(gaps(ref, run_stream(Mix(init, lam), x), nk))
                    res_tau.append(gaps(ref, run_stream(Mix(init, lam, tau=TAU), x), nk))
                line = "  ".join(summ([r[f] for r in res]) for f in ("mean", "cov", "weight"))
                lt = "  ".join(summ([r[f] for r in res_tau]) for f in ("mean", "cov", "weight"))
                print(
                    f"  k={k} lam={lam} N={n:5d} kish n={nk:7.1f}: {line} | tau: {lt}"
                    f" ({time.perf_counter() - t0:.1f} s)",
                    flush=True,
                )
                meds.append(max(float(np.median([r[f] for r in res])) for f in ("mean", "cov")))
                if n == 20000:
                    check(
                        f"decay k={k} lam={lam} N=20000: online within 0.5 SE of the longhand",
                        maxf(res) < 0.5,
                        f"max {maxf(res):.2f}",
                    )
    check(
        "decay: every cell's median gap to the longhand under 0.2 SE",
        max(meds) < 0.2,
        f"{max(meds):.3f}",
    )


# --------------------------------------------------------------------------
# sklearn: larger k and K against scikit-learn (part 2; part 3's two cells)
# --------------------------------------------------------------------------
def sklearn_cell(cov, k, K, n, seeds, w0s, best5=True):
    t0 = time.perf_counter()
    rs = {w0: [] for w0 in w0s}
    same, us = 0, []
    for s in seeds:
        x = draw(make_mixture(k, K, s, cov), n, 200 + s)
        init = kmeans_init(x, K, s, cov)
        ref, _, ll = sk_fit(x, init, REG, cov, s)
        if best5:
            _, _, llb = sk_fit(x, init, REG, cov, s, n_init=5)
            same += abs(ll - llb) < 1e-6
        for w0 in rs:
            g = Mix(init, 1.0, cov=cov, w0=w0)
            t1 = time.perf_counter()
            p = as_full(run_stream(g, x), cov)
            us.append((time.perf_counter() - t1) / n * 1e6)
            rs[w0].append(gaps(ref, p, n))
    cols = " | ".join(
        f"W0={w0:g}: " + "  ".join(summ([r[f] for r in v]) for f in ("mean", "cov", "weight"))
        for w0, v in rs.items()
    )
    b5 = f" | best5 same {same}/{len(seeds)}" if best5 else ""
    print(
        f"  {cov:4s} k={k:2d} K={K} N={n:5d}: {cols}{b5}"
        f" | {np.median(us):.0f} us ({time.perf_counter() - t0:.1f} s)",
        flush=True,
    )
    return rs


def sec_sklearn(quick, seeds):
    print(
        "gap online (1 pass, lam = 1) vs scikit-learn GaussianMixture from the same"
        " k-means init (reg_covar 1e-6), batch SE units, median / max over the seeds,"
        " max over entries of means / covariances / weights; online started at"
        " W0 = 1 row and at W0 = 20 K rows; 'best5 same' = seeds where"
        " scikit-learn's best of n_init = 5 own starts reached the shared-init fit;"
        " us = Python us a row"
    )
    if quick:
        cells = [(c, 5, K, 4000) for c in ("diag", "full") for K in (2, 4)]
    else:
        cells = [
            (c, k, K, n)
            for c in ("diag", "full")
            for k in (5, 20)
            for K in (2, 4, 8)
            for n in (4000, 20000)
        ]
    for cov, k, K, n in cells:
        rs = sklearn_cell(cov, k, K, n, seeds, (1.0, 20.0 * K), best5=not quick)
        if quick:
            heavy = rs[20.0 * K]
            check(
                f"{cov} k=5 K={K} N=4000, W0 = 20K: within 1 SE of scikit-learn",
                maxf(heavy) < 1.0,
                f"max {maxf(heavy):.2f}",
            )
            if K == 4:
                light = maxf(rs[1.0])
                check(
                    f"{cov} k=5 K=4: a one-row start lands far from it on some seed (> 5 SE)",
                    light > 5.0,
                    f"max {light:.1f}",
                )


# --------------------------------------------------------------------------
# collapse: one component absent for a long stretch, under decay (part 2)
# --------------------------------------------------------------------------
def sec_collapse(quick, seeds, seed=0):
    # The quick run shortens the clock: lam = 0.98 reaches the subnormals in
    # about 36,000 rows, lam = 0.99 in about 72,000.
    lam, n1, n_gap, n3 = (0.98, 5000, 40000, 5000) if quick else (0.99, 20000, 80000, 20000)
    k, K = 2, 3
    w, m, c = make_mixture(k, K, seed)
    rng = np.random.default_rng(seed)

    def seg(n, comps):
        z = rng.choice(comps, size=n, p=w[comps] / w[comps].sum())
        out = np.empty((n, k))
        for j in comps:
            idx = z == j
            out[idx] = rng.multivariate_normal(m[j], c[j], size=idx.sum())
        return out

    x = np.vstack([seg(n1, [0, 1, 2]), seg(n_gap, [0, 1]), seg(n3, [0, 1, 2])])
    init = kmeans_init(x[:n1], K, seed)
    j3 = int(np.argmin(((init[1] - m[2]) ** 2).sum(1)))  # the missing one, in the fit's labels
    variants = {
        "sum form, guard": Mix(init, lam),
        "sum form, no guard": Mix(init, lam, guard="none"),
        "mean form (hmm EwCov)": MixMean(init, lam),
        f"fading tau={TAU} (hmm-like)": Mix(init, lam, tau=TAU, fading=True),
        f"stabilised tau={TAU}": Mix(init, lam, tau=TAU),
        f"mean form, stabilised tau={TAU}": MixMean(init, lam, tau=TAU),
        "reseed (pi<1e-6, outlier)": Mix(init, lam, reseed=(1e-6, 1.0)),
    }
    print(
        f"k = 2, K = 3, lam = {lam}: rows 0-{n1} all three components; the third"
        f" absent for {n_gap} rows; back for {n3}. The missing component's least S0 (or W),"
        " its pi at the gap's end, and at the end its pi and mean error; the first failure."
    )
    out, ends = {}, {}
    for name, g in variants.items():
        t0 = time.perf_counter()
        fail, s0_min, pi_gap = None, np.inf, None
        for t, row in enumerate(x):
            try:
                g.step(row, 1.0)
            except np.linalg.LinAlgError:
                fail = t
                break
            s0_min = min(s0_min, g.stats()[0][j3])
            if t == n1 + n_gap - 1:
                pi_gap = g.params()[0][j3]
        end = None
        if fail is None:  # the missing component's weight and mean error at the end
            p = g.params()
            end = (float(p[0][j3]), float(np.abs(p[1][j3] - m[2]).max()))
        out[name] = (fail, s0_min, pi_gap, end)
        ends[name] = g.params() if fail is None else None
        res = (
            f"FAILS at row {fail} (Sigma not PD)"
            if fail is not None
            else f"finite; at the end its pi = {end[0]:.3g} (true {w[2]:.3f}),"
            f" |mu - truth| = {end[1]:.2f}"
        )
        reseeds = f", reseeds {g.reseeds}" if getattr(g, "reseed", None) else ""
        print(
            f"  {name:28s} min S0 {s0_min:.2g}, pi at gap end {pi_gap}; {res}{reseeds}"
            f" ({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )
    sub = 2.3e-308
    check(
        "collapse: the absent component's S0 sinks into the subnormals but never reaches 0",
        0 < out["sum form, guard"][1] < sub,
        f"{out['sum form, guard'][1]:.2g}",
    )
    for name in ("sum form, guard", "sum form, no guard", f"fading tau={TAU} (hmm-like)"):
        check(
            f"collapse: {name} fails inside the gap (S2/S0 - mu mu' not PD)",
            out[name][0] is not None and n1 < out[name][0] < n1 + n_gap,
            f"row {out[name][0]}",
        )
    mf = out["mean form (hmm EwCov)"]
    check(
        "collapse: the mean form survives but the component stays dead",
        mf[0] is None and mf[3][0] < 0.01,
        f"pi at the end {mf[3][0] if mf[3] else None}",
    )
    for name in (f"stabilised tau={TAU}", f"mean form, stabilised tau={TAU}"):
        r = out[name]
        check(
            f"collapse: {name} stays finite and revives the component",
            r[0] is None and r[3][0] > 0.2 and r[3][1] < 0.5,
            f"pi {r[3][0]:.3f}, |mu - truth| {r[3][1]:.2f}" if r[3] else "failed",
        )
    a, b = ends[f"stabilised tau={TAU}"], ends[f"mean form, stabilised tau={TAU}"]
    if a is not None and b is not None:
        d = max(float(np.abs(u - v).max()) for u, v in zip(a, b, strict=True))
        check(
            "collapse: the stabilised read agrees in the sum and the mean form",
            d < 1e-9,
            f"{d:.1e}",
        )
    # The reseed is no substitute for tau. It fires only on a row that is an
    # outlier to every component: at lam = 0.99 none comes before the sum
    # form's failure (0 reseeds, the same row fails); at lam = 0.98 it fires
    # once in the gap but the returning rows fall inside the chi-square cut
    # of a live component, which takes them, so the component stays dead.
    r = out["reseed (pi<1e-6, outlier)"]
    check(
        "collapse: the reseed does not revive the component (fails or stays dead)",
        r[0] is not None or r[3][0] < 0.01,
        f"fails at row {r[0]}" if r[0] is not None else f"finite, its pi {r[3][0]:.2g}",
    )
    # A component that starts with S0 = 0 exactly.
    init0 = (np.array([0.5, 0.5, 0.0]), init[1], init[2])
    p = Mix(init0, lam, guard="none").params()
    check("S0 = 0 from the start, no guard: mu = 0/0 = NaN", np.isnan(p[1][2]).all())
    for name, g in (("guard", Mix(init0, lam)), ("stabilised tau", Mix(init0, lam, tau=TAU))):
        ok = all(np.all(np.isfinite(a)) for a in g.params())
        g.step(x[0], 1.0)
        ok = ok and all(np.all(np.isfinite(a)) for a in g.params())
        check(f"S0 = 0 from the start, {name}: finite", ok)


# --------------------------------------------------------------------------
# state: bytes and flops a row, formula against the arrays the state holds
# --------------------------------------------------------------------------
def state_bytes(k, K, cov):
    return 8 * K * (1 + k + (k * (k + 1) // 2 if cov == "full" else k))


def sec_state(quick, seeds):
    print(
        "bytes = 8 K (1 + k + k(k+1)/2) full, 8 K (1 + 2k) diag; flops a row:"
        " full E = K (k^3/3 + k^2 + 4k) (refactor) or K (2k^2 + 4k) (held factor),"
        " update K (k^2 + 3k); diag E = K 4k, update K 4k"
    )
    print("| k | K | full bytes | diag bytes | full flops (refactor / held) | diag flops |")
    for k in (2, 5, 20):
        for K in (2, 4, 8):
            fr = K * (k**3 // 3 + k * k + 4 * k) + K * (k * k + 3 * k)
            fh = K * (2 * k * k + 4 * k) + K * (k * k + 3 * k)
            print(
                f"| {k} | {K} | {state_bytes(k, K, 'full')} | {state_bytes(k, K, 'diag')}"
                f" | {fr} / {fh} | {K * 8 * k} |"
            )
    ok = True
    for k in (2, 5, 20):
        for K in (2, 4, 8):
            for cov in ("full", "diag"):
                g = Mix(make_mixture(k, K, 0, cov), cov=cov)
                s2 = g.S2.size if cov == "diag" else K * k * (k + 1) // 2  # packed symmetric
                ok &= 8 * (g.S0.size + g.S1.size + s2) == state_bytes(k, K, cov)
    check("state: the formula counts the arrays the recursion holds (S2 packed)", ok)


# --------------------------------------------------------------------------
# tolerances: separated components, N = 20000, k = 5, per K and covariance
# --------------------------------------------------------------------------
def tol_cell(cov, k, K, n, seeds):
    """Online (W0 = 20 K rows) against scikit-learn from the same start. A
    seed is 'well posed' when the shared start's batch fit is also
    scikit-learn's best of n_init = 5 own starts (mean log-likelihood within
    1e-6); otherwise the batch fit is itself a poor local optimum. Also the
    EM iterations scikit-learn takes from the online fit back to its own."""
    rows = []
    for s in seeds:
        x = draw(make_mixture(k, K, s, cov), n, 200 + s)
        init = kmeans_init(x, K, s, cov)
        ref, _, ll = sk_fit(x, init, REG, cov, s)
        _, _, llb = sk_fit(x, init, REG, cov, s, n_init=5)
        p = as_full(run_stream(Mix(init, 1.0, cov=cov, w0=20.0 * K), x), cov)
        g = gaps(ref, p, n)
        prec = np.linalg.inv(p[2]) if cov == "full" else 1.0 / np.einsum("jaa->ja", p[2])
        back = GaussianMixture(
            K,
            covariance_type=cov,
            reg_covar=REG,
            tol=1e-10,
            max_iter=5000,
            weights_init=p[0],
            means_init=p[1],
            precisions_init=prec,
        ).fit(x)
        g.update(well_posed=abs(ll - llb) < 1e-6, seed=s, refit_iters=back.n_iter_)
        rows.append(g)
    return rows


def sec_tolerances(quick, seeds):
    print(
        "online (1 pass, lam = 1, W0 = 20 K rows) vs scikit-learn from the same start,"
        " separated components, k = 5, N = 20000, batch SE; max over the regular seeds"
        " of the max over entries; proposed tolerance = 1.5 x that, rounded up to 0.05;"
        " EM iterations from the online fit back to scikit-learn's"
    )
    Ks = (2, 4) if quick else (2, 4, 8)
    sd = list(seeds)[:2] if quick else list(seeds)
    for cov in ("diag", "full"):
        for K in Ks:
            t0 = time.perf_counter()
            rows = tol_cell(cov, 5, K, 20000, sd)
            # Regular: well posed, and EM from the online fit returns in a few
            # iterations (2 to 20 seen); the rest come back in 1,000 or more
            # (the online pass ended on a ridge, another attribution).
            ins = [r for r in rows if r["well_posed"] and r["refit_iters"] < REGULAR_ITERS]
            out = [r for r in rows if r not in ins]
            mx = {
                f: max(r[f] for r in ins) if ins else float("nan")
                for f in ("mean", "cov", "weight")
            }
            tl = {f: tol_of(v) for f, v in mx.items()}
            iters = [r["refit_iters"] for r in ins]
            outs = ", ".join(
                f"seed {r['seed']} at {r['mean']:.3g} / {r['cov']:.3g} / {r['weight']:.3g}"
                f" ({'' if r['well_posed'] else 'not well posed, '}"
                f"{r['refit_iters']} EM iterations back)"
                for r in out
            )
            print(
                f"  {cov:4s} K={K}: max means {mx['mean']:.3f} cov {mx['cov']:.3f}"
                f" weights {mx['weight']:.3f} -> tolerance {tl['mean']:.2f} / {tl['cov']:.2f}"
                f" / {tl['weight']:.2f}; EM back {min(iters)}-{max(iters)} iterations;"
                f" {len(ins)} regular of {len(rows)}; other: {outs or 'none'}"
                f" ({time.perf_counter() - t0:.1f} s)",
                flush=True,
            )
            if quick:
                prop = PROPOSED[(cov, K)]
                worst = max(
                    r[f] / prop[i] for r in ins for i, f in enumerate(("mean", "cov", "weight"))
                )
                check(
                    f"tolerances {cov} K={K}: seeds {sd} within the proposed {prop}",
                    bool(ins) and worst <= 1.0,
                    f"worst {worst:.2f} of the tolerance",
                )


REGULAR_ITERS = 100

# The proposed tolerances (means, covariances, weights) in batch SE, from
# `tolerances --seeds 50` (the regular seeds of 50); the quick run checks
# them on seeds 0 and 1.
PROPOSED = {
    ("diag", 2): (1.10, 1.40, 0.65),
    ("diag", 4): (1.20, 1.45, 1.05),
    ("diag", 8): (1.25, 1.90, 1.45),
    ("full", 2): (1.15, 1.50, 0.70),
    ("full", 4): (1.85, 2.05, 1.25),
    ("full", 8): (2.65, 4.10, 2.05),
}

SECTIONS = {
    "compare": sec_compare,
    "passes": sec_passes,
    "zero": sec_zero,
    "timing": sec_timing,
    "decay": sec_decay,
    "sklearn": sec_sklearn,
    "collapse": sec_collapse,
    "state": sec_state,
    "tolerances": sec_tolerances,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sections", nargs="*", help=f"default: all of {', '.join(SECTIONS)}")
    ap.add_argument("--quick", action="store_true", help="short runs of every section")
    ap.add_argument("--seeds", type=int, default=5, help="seeds 0 .. n-1 (default 5)")
    ap.add_argument(
        "--cell",
        nargs=4,
        metavar=("COV", "k", "K", "N"),
        help="one cell of the sklearn section, e.g. full 20 8 20000",
    )
    args = ap.parse_args()
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        ap.error(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    seeds = range(args.seeds)
    t_all = time.perf_counter()
    if args.cell:
        cov, k, K, n = args.cell[0], int(args.cell[1]), int(args.cell[2]), int(args.cell[3])
        sklearn_cell(cov, k, K, n, seeds, (1.0, 20.0 * K))
        return 0
    for name in args.sections or list(SECTIONS):
        print(f"\n===== {name}", flush=True)
        t = time.perf_counter()
        SECTIONS[name](args.quick, seeds)
        print(f"({name}: {time.perf_counter() - t:.1f} s)", flush=True)
    bad = [c for c in CHECKS if not c[1]]
    print(
        f"\n{len(CHECKS) - len(bad)} of {len(CHECKS)} checks passed"
        f" in {time.perf_counter() - t_all:.0f} s"
    )
    for name, _, detail in bad:
        print(f"  FAILED: {name}: {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
