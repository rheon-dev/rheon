"""Study 506, part 1: a Gaussian mixture by exponentially weighted online EM.

numpy only (scipy for the Hungarian assignment, scikit-learn for the batch
oracle). No data files: every stream is drawn from a seeded generator.

    uv run --no-project python scripts/study_506.py          # all checks, ~20 s
    uv run --no-project python scripts/study_506.py --quick  # N = 2000 only

Notation follows rheon: ``k`` is the number of features, ``K`` the number of
mixture components. The notes are docs/records/STUDY-506-MIXTURES.md.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------
# The recursion, sufficient-statistic form (Cappe and Moulines 2009, with a
# forgetting factor in place of their step size).
# --------------------------------------------------------------------------
class OnlineGMM:
    """State: per component S0 (scalar), S1 (k), S2 (k x k, symmetric).

    Row (x, w), with lam the decay over the row's clock:
      E: r_j = pi_j N(x; mu_j, Sigma_j) / sum_l pi_l N(x; mu_l, Sigma_l),
         the parameters read from the state *before* the row.
      S0_j <- lam S0_j + w r_j
      S1_j <- lam S1_j + w r_j x
      S2_j <- lam S2_j + w r_j x x'
    M (read on demand, never stored):
      pi_j = S0_j / sum S0,  mu_j = S1_j / S0_j,
      Sigma_j = S2_j / S0_j - mu_j mu_j' + reg I.
    Guards: a component with S0_j = 0 reads the fallback (the
    initialisation's) parameters; a state with sum S0 = 0 reads the
    fallback weights; a row with w = 0 does no E-step at all and only decays.
    """

    def __init__(self, lam, reg, weights, means, covs_raw, w0):
        self.lam = float(lam)
        self.reg = float(reg)
        self.fb_w = np.asarray(weights, float).copy()
        self.fb_m = np.asarray(means, float).copy()
        self.fb_c = np.asarray(covs_raw, float).copy()
        K, k = self.fb_m.shape
        self.K, self.k = K, k
        s0 = w0 * self.fb_w
        self.S0 = s0.copy()
        self.S1 = s0[:, None] * self.fb_m
        self.S2 = s0[:, None, None] * (
            self.fb_c + self.fb_m[:, :, None] * self.fb_m[:, None, :]
        )

    def params(self):
        tot = self.S0.sum()
        w = self.S0 / tot if tot > 0 else self.fb_w.copy()
        m = self.fb_m.copy()
        c = self.fb_c.copy()
        for j in range(self.K):
            if self.S0[j] > 0:
                m[j] = self.S1[j] / self.S0[j]
                c[j] = self.S2[j] / self.S0[j] - np.outer(m[j], m[j])
        c = c + self.reg * np.eye(self.k)
        return w, m, c

    def logdens(self, x, w, m, c):
        out = np.empty(self.K)
        for j in range(self.K):
            L = np.linalg.cholesky(c[j])
            z = np.linalg.solve(L, x - m[j])
            out[j] = -0.5 * (self.k * LOG_2PI + z @ z) - np.log(np.diag(L)).sum()
        with np.errstate(divide="ignore"):
            return out + np.log(w)

    def step(self, x, weight, lam=None):
        lam = self.lam if lam is None else lam
        if not (weight >= 0 and np.isfinite(weight) and np.all(np.isfinite(x))):
            weight = 0.0  # a refused row: the clock only
        if weight == 0.0:
            self.S0 *= lam
            self.S1 *= lam
            self.S2 *= lam
            return None
        w, m, c = self.params()
        lp = self.logdens(x, w, m, c)
        top = lp.max()
        e = np.exp(lp - top)
        r = e / e.sum()
        wr = weight * r
        self.S0 = lam * self.S0 + wr
        self.S1 = lam * self.S1 + wr[:, None] * x
        self.S2 = lam * self.S2 + wr[:, None, None] * np.outer(x, x)
        return r


class OnlineGMMMeanForm(OnlineGMM):
    """The same recursion held as (W_j, mu_j, C_j), hmm's `EwCov` form.

    w_j = w r_j,  W' = lam W + w_j,  a = lam W / W',  b = w_j / W',
    d = x - mu,  mu' = mu + b d,  C' = a C + a b d d'.
    Algebraically S0 = W, S1 = W mu, S2 = W (C + mu mu'); free of the
    S2 / S0 - mu mu' cancellation.
    """

    def __init__(self, lam, reg, weights, means, covs_raw, w0):
        super().__init__(lam, reg, weights, means, covs_raw, w0)
        self.W = w0 * self.fb_w.copy()
        self.mu = self.fb_m.copy()
        self.C = self.fb_c.copy()

    def params(self):
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
        lp = self.logdens(x, w, m, c)
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
# Data, initialisation, the batch oracle
# --------------------------------------------------------------------------
SCENARIOS = {
    # Means 4.0 and 3.4 standard deviations apart along each axis.
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


def draw(scn, n, seed):
    rng = np.random.default_rng(seed)
    w = np.array(scn["w"])
    z = rng.choice(len(w), size=n, p=w)
    x = np.empty((n, len(scn["m"][0])))
    for j in range(len(w)):
        idx = z == j
        x[idx] = rng.multivariate_normal(scn["m"][j], scn["c"][j], size=idx.sum())
    return x


def kmeans_init(x, K, seed):
    """sklearn's own `init_params='kmeans'`: one k-means run, hard labels,
    one M-step. Returns weights, means, raw covariances (no reg)."""
    lab = KMeans(n_clusters=K, n_init=1, random_state=seed).fit(x).labels_
    n, k = x.shape
    w = np.empty(K)
    m = np.empty((K, k))
    c = np.empty((K, k, k))
    for j in range(K):
        xj = x[lab == j]
        w[j] = len(xj) / n
        m[j] = xj.mean(0)
        d = xj - m[j]
        c[j] = d.T @ d / len(xj)
    return w, m, c


def batch(x, init, reg, seed):
    w, m, c = init
    prec = np.linalg.inv(c + reg * np.eye(x.shape[1]))
    gm = GaussianMixture(
        n_components=len(w),
        covariance_type="full",
        reg_covar=reg,
        tol=1e-12,
        max_iter=5000,
        weights_init=w,
        means_init=m,
        precisions_init=prec,
        random_state=seed,
    ).fit(x)
    assert gm.converged_
    return gm.weights_, gm.means_, gm.covariances_


def match(ref_means, means):
    cost = ((ref_means[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(cost)
    return col


def gaps(ref, est, n):
    """max |est - ref| over components and entries, in units of the batch
    standard error: mean sqrt(Sigma_aa / (n pi)), covariance entry
    sqrt((S_aa S_bb + S_ab^2) / (n pi)) (Gaussian), weight sqrt(pi(1-pi)/n)."""
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


def truth(scn):
    return np.array(scn["w"]), np.array(scn["m"]), np.array(scn["c"])


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
REG = 1e-6
W0 = 1.0  # the initialisation enters at one row's weight, as hmm's means=/covs=


def run_compare(name, n, seed):
    scn = SCENARIOS[name]
    x = draw(scn, n, seed)
    init = kmeans_init(x, 2, seed)
    ref = batch(x, init, REG, seed)
    sf = OnlineGMM(1.0, REG, *init, W0)
    mf = OnlineGMMMeanForm(1.0, REG, *init, W0)
    for row in x:
        sf.step(row, 1.0)
        mf.step(row, 1.0)
    p1 = sf.params()
    form_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(p1, mf.params()))
    # Second pass, the statistics carried on (lam = 1: pass 1 keeps half the weight).
    for row in x:
        sf.step(row, 1.0)
    p2 = sf.params()
    # Second pass, the statistics restarted from pass 1's parameters at W0.
    rs = OnlineGMM(1.0, REG, p1[0], p1[1], p1[2] - REG * np.eye(2), W0)
    for row in x:
        rs.step(row, 1.0)
    p2r = rs.params()
    return dict(
        batch_vs_truth=gaps(ref, truth(scn), n),
        pass1=gaps(ref, p1, n),
        pass2=gaps(ref, p2, n),
        pass2_restart=gaps(ref, p2r, n),
        form_diff=form_diff,
    )


def check_zero_weight():
    out = {}
    init = kmeans_init(draw(SCENARIOS["separated"], 500, 0), 2, 0)
    x = draw(SCENARIOS["separated"], 200, 1)
    for cls in (OnlineGMM, OnlineGMMMeanForm):
        # (a) an empty state (w0 = 0), the first row of weight 0, then data.
        g = cls(1.0, REG, *init, 0.0)
        snap = [a.copy() for a in (g.S0, g.S1, g.S2)]
        g.step(x[0], 0.0)
        same_first = all(np.array_equal(a, b) for a, b in zip(snap, (g.S0, g.S1, g.S2)))
        if cls is OnlineGMMMeanForm:
            same_first = same_first and np.array_equal(g.W, np.zeros(2))
        for row in x[1:50]:
            g.step(row, 1.0)
        nan_free = all(np.all(np.isfinite(p)) for p in g.params())
        # (b) lam = 1: zero-weight rows interleaved leave the state bit-identical.
        a = cls(1.0, REG, *init, W0)
        b = cls(1.0, REG, *init, W0)
        for i, row in enumerate(x):
            a.step(row, 1.0)
            b.step(row, 1.0)
            if i % 3 == 0:
                b.step(x[(i * 7) % len(x)], 0.0)
        attrs = ("W", "mu", "C") if cls is OnlineGMMMeanForm else ("S0", "S1", "S2")
        bit_same = all(np.array_equal(getattr(a, t), getattr(b, t)) for t in attrs)
        # (c) lam < 1: a zero-weight row only decays; the parameters it reads do
        # not move (the mean form to the bit, the sum form to rounding).
        c = cls(0.99, REG, *init, W0)
        for row in x[:100]:
            c.step(row, 1.0)
        before = c.params()
        c.step(x[100], 0.0)
        after = c.params()
        param_move = max(float(np.max(np.abs(p - q))) for p, q in zip(before, after))
        out[cls.__name__] = dict(
            empty_first_row_unchanged=same_first,
            nan_free_after=nan_free,
            lam1_bit_identical=bit_same,
            lam099_param_move=param_move,
        )
    return out


def per_row_timing(n=2000):
    x = draw(SCENARIOS["separated"], n, 3)
    init = kmeans_init(x, 2, 3)
    out = {}
    for cls in (OnlineGMM, OnlineGMMMeanForm):
        g = cls(1.0, REG, *init, W0)
        t = time.perf_counter()
        for row in x:
            g.step(row, 1.0)
        out[cls.__name__] = (time.perf_counter() - t) / n * 1e6
    return out


def mean_loglik(x, params):
    w, m, c = params
    lp = np.empty((len(x), len(w)))
    for j in range(len(w)):
        L = np.linalg.cholesky(c[j])
        z = np.linalg.solve(L, (x - m[j]).T)
        lp[:, j] = (
            np.log(w[j]) - 0.5 * (x.shape[1] * LOG_2PI + (z * z).sum(0))
            - np.log(np.diag(L)).sum()
        )
    top = lp.max(1, keepdims=True)
    return float(np.mean(top[:, 0] + np.log(np.exp(lp - top).sum(1))))


def passes_curve(name, n, seed, n_passes):
    """Carried-on lam = 1 statistics over repeated passes: the gap to batch
    EM after each pass, and the mean log-likelihood shortfall per row."""
    scn = SCENARIOS[name]
    x = draw(scn, n, seed)
    init = kmeans_init(x, 2, seed)
    ref = batch(x, init, REG, seed)
    ll_ref = mean_loglik(x, ref)
    g = OnlineGMM(1.0, REG, *init, W0)
    out = []
    for _ in range(n_passes):
        for row in x:
            g.step(row, 1.0)
        p = g.params()
        out.append((gaps(ref, p, n), ll_ref - mean_loglik(x, p)))
    return out


def summarise(rows, key):
    vals = {f: np.array([r[key][f] for r in rows]) for f in ("mean", "cov", "weight")}
    return "  ".join(
        f"{f} med {np.median(v):6.2f} max {v.max():6.2f}" for f, v in vals.items()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ns = [2000] if args.quick else [2000, 20000]
    seeds = range(5)
    print("Gaps are max over components and entries of |online - batch|,")
    print("in batch standard errors; median and max over 5 seeds.\n")
    for name in SCENARIOS:
        for n in ns:
            t = time.perf_counter()
            rows = [run_compare(name, n, s) for s in seeds]
            print(f"== {name}, k = 2, K = 2, N = {n}  ({time.perf_counter() - t:.1f} s)")
            for key in ("batch_vs_truth", "pass1", "pass2", "pass2_restart"):
                print(f"  {key:15s} {summarise(rows, key)}")
            absg = {
                f: max(r["pass1"][f + "_abs"] for r in rows)
                for f in ("mean", "cov", "weight")
            }
            print(f"  pass1 absolute max: {absg}")
            fd = max(r["form_diff"] for r in rows)
            print(f"  sum form vs mean form, max |param diff| after pass 1: {fd:.2e}")
    print("\n== carried-on passes, N = 2000: median over seeds of the max gap (SE)")
    print("   and of the batch's mean log-likelihood lead per row")
    for name in SCENARIOS:
        curves = [passes_curve(name, 2000, s, 8) for s in seeds]
        for i in (0, 1, 3, 7):
            gm = np.median([c[i][0]["mean"] for c in curves])
            gc = np.median([c[i][0]["cov"] for c in curves])
            gw = np.median([c[i][0]["weight"] for c in curves])
            dl = np.median([c[i][1] for c in curves])
            print(
                f"  {name:11s} pass {i + 1}: mean {gm:5.2f} cov {gc:5.2f}"
                f" weight {gw:5.2f}  loglik lead {dl:.2e}"
            )
    print("\n== zero-weight rows")
    for cls, res in check_zero_weight().items():
        print(f"  {cls}: {res}")
    print("\n== per-row Python time (us, k = 2, K = 2):", per_row_timing())
    K, k = 2, 2
    floats = K * (1 + k + k * (k + 1) // 2)
    print(f"== state: {floats} f64 = {8 * floats} bytes (K (1 + k + k(k+1)/2))")


if __name__ == "__main__":
    main()
