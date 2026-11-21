"""Study 506, part 2: decay, larger k and K, collapse, zero-weight rows, state.

numpy only (scipy for the Hungarian assignment, scikit-learn for the batch
oracle). No data files: every stream is drawn from a seeded generator.

    .venv/bin/python scripts/study_506_part2.py decay     # ~60 s
    .venv/bin/python scripts/study_506_part2.py sklearn   # ~3 min (nohup it)
    .venv/bin/python scripts/study_506_part2.py collapse  # ~30 s
    .venv/bin/python scripts/study_506_part2.py zero      # ~1 s
    .venv/bin/python scripts/study_506_part2.py state     # ~10 s

Notation follows rheon and part 1 (scripts/study_506.py): ``k`` features,
``K`` mixture components. The notes are docs/records/STUDY-506-MIXTURES.md.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_506 import gaps  # noqa: E402  (part 1: Hungarian match + SE units)

LOG_2PI = math.log(2.0 * math.pi)
REG = 1e-6
W0 = 1.0


# --------------------------------------------------------------------------
# The recursion, vectorised over components, full or diagonal covariance.
# --------------------------------------------------------------------------
class Mix:
    """Sum form (S0, S1, S2) of part 1, vectorised, with options:

    cov      'full' (S2: K x k x k) or 'diag' (S2: K x k, the squares only).
    guard    'fallback': part 1's guard (S0_j = 0 reads the initial params);
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


class MixMean:
    """Part 1's mean form (W_j, mu_j, C_j), hmm's EwCov, full covariance."""

    def __init__(self, init, lam=1.0, reg=REG, w0=W0):
        w, m, c = (np.asarray(a, float).copy() for a in init)
        self.K, self.k = m.shape
        self.lam, self.reg = float(lam), float(reg)
        self.fb_w = w
        self.W, self.mu, self.C = w0 * w, m, c

    def params(self):
        tot = self.W.sum()
        w = self.W / tot if tot > 0 else self.fb_w.copy()
        return w, self.mu.copy(), self.C + self.reg * np.eye(self.k)

    def step(self, x, weight, lam=None):
        lam = self.lam if lam is None else lam
        if weight == 0.0:
            self.W *= lam
            return None
        w, m, c = self.params()
        L = np.linalg.cholesky(c)
        z = np.linalg.solve(L, (x - m)[:, :, None])[:, :, 0]
        with np.errstate(divide="ignore"):
            lp = np.log(w) - 0.5 * (z * z).sum(1) - np.log(np.einsum("jaa->ja", L)).sum(1)
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
def make_mixture(k, K, seed, cov="full", min_dist=4.0):
    """Separated components: covariance eigenvalues in [0.5, 1.5] (so a
    standard deviation of at most 1.22), means at least min_dist apart
    (3.3 SD or more; part 1's separated pair was 4 SD), weights
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


def draw(mix, n, seed, present=None):
    """present: optional (n,) boolean mask per component row-range handled by
    the caller; here a plain i.i.d. draw."""
    w, m, c = mix
    rng = np.random.default_rng(seed)
    z = rng.choice(len(w), size=n, p=w)
    x = np.empty((n, m.shape[1]))
    for j in range(len(w)):
        idx = z == j
        x[idx] = rng.multivariate_normal(m[j], c[j], size=idx.sum())
    return x, z


def kmeans_init(x, K, seed, cov="full"):
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


def sk_fit(x, init, reg, cov, seed, n_init=None):
    k = x.shape[1]
    kw = dict(
        n_components=len(init[0]),
        covariance_type=cov,
        reg_covar=reg,
        tol=1e-10,
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


def as_full(p, cov):
    w, m, c = p
    if cov == "diag":
        c = np.array([np.diag(v) for v in c])
    return w, m, c


def run_stream(g, x):
    for row in x:
        g.step(row, 1.0)
    return g.params()


def summ(vals):
    v = np.asarray(vals)
    return f"{np.median(v):.3g} / {v.max():.3g}"


# --------------------------------------------------------------------------
# 1. Decay against the weighted EM longhand
# --------------------------------------------------------------------------
def decay(seeds=range(5), ns=(2000, 20000), lams=(0.999, 0.99), ks=(2, 5), K=2):
    # The longhand checked at lam = 1 against scikit-learn (same init).
    mix = make_mixture(2, 2, 0)
    x, _ = draw(mix, 2000, 0)
    init = kmeans_init(x, 2, 0)
    lh, _ = weighted_em(x, np.ones(len(x)), init, REG)
    sk, _, _ = sk_fit(x, init, REG, "full", 0)
    print(
        "longhand vs sklearn at lam = 1, max |param diff|:",
        f"{max(np.abs(a - b).max() for a, b in zip(lh, sk, strict=True)):.2e}",
    )
    print(
        "gap online vs longhand at the stream's end, in SE of the decay-weighted"
        " sample (Kish n = (sum v)^2 / sum v^2); median / max over 5 seeds;"
        " columns: means, covariances, weights; and with a stabilised tau = 0.01"
    )
    for k in ks:
        for lam in lams:
            for n in ns:
                t0 = time.perf_counter()
                res, res_tau, iters = [], [], []
                for s in seeds:
                    mix = make_mixture(k, K, s)
                    x, _ = draw(mix, n, 100 + s)
                    init = kmeans_init(x, K, s)
                    v = lam ** np.arange(n - 1, -1, -1, dtype=float)
                    nk = v.sum() ** 2 / (v * v).sum()
                    ref, it = weighted_em(x, v, init, REG)
                    iters.append(it)
                    res.append(gaps(ref, run_stream(Mix(init, lam), x), nk))
                    res_tau.append(gaps(ref, run_stream(Mix(init, lam, tau=0.01), x), nk))
                line = "  ".join(summ([r[f] for r in res]) for f in ("mean", "cov", "weight"))
                lt = "  ".join(summ([r[f] for r in res_tau]) for f in ("mean", "cov", "weight"))
                print(
                    f"k={k} lam={lam} N={n:5d} kish n={nk:7.1f}: {line} | tau: {lt}"
                    f"  (EM iters {min(iters)}-{max(iters)}, {time.perf_counter() - t0:.1f} s)"
                )


# --------------------------------------------------------------------------
# 2. Larger k and K against scikit-learn
# --------------------------------------------------------------------------
def vs_sklearn(seeds=range(5), ns=(4000, 20000), ks=(5, 20), Ks=(2, 4, 8), covs=("diag", "full")):
    print(
        "gap online (1 pass, lam = 1) vs sklearn GaussianMixture from the same"
        " k-means init (reg_covar 1e-6), batch SE units, median / max over 5 seeds,"
        " max over entries of means / covariances / weights; online started at"
        " W0 = 1 row (part 1) and at W0 = 20 K rows; 'best5 same' = seeds where"
        " sklearn's best of n_init = 5 own starts reached the shared-init fit;"
        " us = Python us a row"
    )
    for cov in covs:
        for k in ks:
            for K in Ks:
                for n in ns:
                    t0 = time.perf_counter()
                    rs = {1.0: [], 20.0 * K: []}
                    same, us = 0, []
                    for s in seeds:
                        mix = make_mixture(k, K, s, cov)
                        x, _ = draw(mix, n, 200 + s)
                        init = kmeans_init(x, K, s, cov)
                        ref, conv, ll = sk_fit(x, init, REG, cov, s)
                        _, _, llb = sk_fit(x, init, REG, cov, s, n_init=5)
                        same += abs(ll - llb) < 1e-6
                        for w0 in rs:
                            g = Mix(init, 1.0, cov=cov, w0=w0)
                            t1 = time.perf_counter()
                            p = as_full(run_stream(g, x), cov)
                            us.append((time.perf_counter() - t1) / n * 1e6)
                            rs[w0].append(gaps(ref, p, n))
                    cols = " | ".join(
                        f"W0={w0:g}: "
                        + "  ".join(summ([r[f] for r in v]) for f in ("mean", "cov", "weight"))
                        for w0, v in rs.items()
                    )
                    print(
                        f"{cov:4s} k={k:2d} K={K} N={n:5d}: {cols} | best5 same {same}/5"
                        f" | {np.median(us):.0f} us ({time.perf_counter() - t0:.1f} s)",
                        flush=True,
                    )


# --------------------------------------------------------------------------
# 3. Collapse: one component never sampled for a long stretch, under decay
# --------------------------------------------------------------------------
def collapse(lam=0.99, n1=20000, n_gap=80000, n3=20000, seed=0):
    k, K = 2, 3
    mix = make_mixture(k, K, seed)
    w, m, c = mix
    rng = np.random.default_rng(seed)

    def seg(n, comps):
        ww = w[comps] / w[comps].sum()
        z = rng.choice(comps, size=n, p=ww)
        out = np.empty((n, k))
        for j in comps:
            idx = z == j
            out[idx] = rng.multivariate_normal(m[j], c[j], size=idx.sum())
        return out

    x = np.vstack([seg(n1, [0, 1, 2]), seg(n_gap, [0, 1]), seg(n3, [0, 1, 2])])
    init = kmeans_init(x[:n1], K, seed)
    perm = [int(np.argmin(((init[1] - m[j]) ** 2).sum(1))) for j in range(K)]
    j3 = perm[2]  # the component that goes missing, in the fit's labels
    variants = {
        "sum form, part 1 guard": Mix(init, lam),
        "sum form, no guard": Mix(init, lam, guard="none"),
        "mean form (hmm EwCov)": MixMean(init, lam),
        "fading tau=0.01 (hmm-like)": Mix(init, lam, tau=0.01, fading=True),
        "stabilised tau=0.01": Mix(init, lam, tau=0.01),
        "reseed (pi<1e-6, outlier)": Mix(init, lam, reseed=(1e-6, 1.0)),
    }
    print(
        f"k = 2, K = 3, lam = {lam}: rows 0-{n1} all three components; the third"
        f" absent for {n_gap} rows; back for {n3}. Missing component's S0 (or W),"
        " pi, |mu - truth| at the end of the gap and at the end; first failure."
    )
    for name, g in variants.items():
        t0 = time.perf_counter()
        fail, s0_min, s0_zero_at, pi_gap, mu_gap = None, np.inf, None, None, None
        for t, row in enumerate(x):
            try:
                g.step(row, 1.0)
            except np.linalg.LinAlgError as e:
                fail = f"row {t}: LinAlgError ({e})"
                break
            s0 = g.S0[j3] if hasattr(g, "S0") else g.W[j3]
            s0_min = min(s0_min, s0)
            if s0 == 0 and s0_zero_at is None:
                s0_zero_at = t
            if t == n1 + n_gap - 1:
                p = g.params()
                pi_gap, mu_gap = p[0][j3], np.abs(p[1][j3] - m[2]).max()
                s0_gap = s0
            if t % 1000 == 0:
                p = g.params()
                if not all(np.all(np.isfinite(a)) for a in p):
                    fail = f"row {t}: non-finite parameters"
                    break
        if fail is None:
            p = g.params()
            back = int(np.argmin(((p[1] - m[2]) ** 2).sum(1)))
            end = (
                f"end: nearest comp to truth pi={p[0][back]:.3f} (true {w[2] / 1:.3f}),"
                f" |mu-truth|={np.abs(p[1][back] - m[2]).max():.3f}, finite"
            )
        else:
            end = ""
        print(
            f"  {name:28s} min S0={s0_min:.3g} zero at {s0_zero_at}; gap end: S0="
            f"{s0_gap if pi_gap is not None else float('nan'):.3g} pi={pi_gap} |mu-truth|="
            f"{mu_gap}; {fail or end}"
            f"{'; reseeds ' + str(g.reseeds) if getattr(g, 'reseed', None) else ''}"
            f" ({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )
    # Forced zero responsibility: a component that starts with S0 = 0.
    init0 = (np.array([0.5, 0.5, 0.0]), init[1], init[2])
    for name, g in (
        ("forced S0=0, no guard", Mix(init0, lam, guard="none")),
        ("forced S0=0, part 1 guard", Mix(init0, lam)),
        ("forced S0=0, stabilised tau", Mix(init0, lam, tau=0.01)),
    ):
        try:
            p = g.params()
            ok = all(np.all(np.isfinite(a)) for a in p)
            g.step(x[0], 1.0)
            ok2 = all(np.all(np.isfinite(a)) for a in g.params())
            print(f"  {name:28s} params finite before row {ok}, after {ok2}; mu_3 = {p[1][2]}")
        except np.linalg.LinAlgError as e:
            print(f"  {name:28s} mu_3 = {p[1][2]}; row 0: LinAlgError ({e})")


# --------------------------------------------------------------------------
# 4. Zero-weight rows under decay
# --------------------------------------------------------------------------
def zero_weight(lam=0.99):
    mix = make_mixture(3, 2, 1)
    x, _ = draw(mix, 400, 1)
    init = kmeans_init(x, 2, 1)
    out = {}
    for name, mk in (
        ("sum", lambda w0: Mix(init, lam, w0=w0)),
        ("mean", lambda w0: MixMean(init, lam, w0=w0)),
        ("sum tau=0.01", lambda w0: Mix(init, lam, w0=w0, tau=0.01)),
    ):
        stats = (lambda g: [g.S0, g.S1, g.S2]) if name != "mean" else (lambda g: [g.W, g.mu, g.C])
        # (a) empty state, a zero-weight first row: nothing moves, and the
        # stream after it is the stream without it, to the bit.
        a, b = mk(0.0), mk(0.0)
        a.step(x[0], 0.0)
        empty_same = all(np.array_equal(u, np.zeros_like(u)) for u in stats(a)[:1])
        for row in x[1:]:
            a.step(row, 1.0)
            b.step(row, 1.0)
        empty_bits = all(np.array_equal(u, v) for u, v in zip(stats(a), stats(b), strict=True))
        # (b) an initialised state, a zero-weight first row: the clock advances
        # (each weight is lam times its value, to the bit), the parameters it
        # reads do not move.
        g = mk(W0)
        before = [u.copy() for u in stats(g)]
        p0 = g.params()
        g.step(x[0], 0.0)
        clock = np.array_equal(stats(g)[0], lam * before[0])
        pmove = max(float(np.abs(u - v).max()) for u, v in zip(p0, g.params(), strict=True))
        # (c) a zero-weight row after every third row: the same as the stream
        # without them with the decay lam^2 over each gap (the clock form).
        a, b = mk(W0), mk(W0)
        for i, row in enumerate(x):
            a.step(row, 1.0)
            if i % 3 == 0 and i < len(x) - 1:
                a.step(x[(7 * i) % len(x)], 0.0)
            b.step(row, 1.0, lam=lam * lam if (i > 0 and (i - 1) % 3 == 0) else lam)
        rel = max(
            float(np.max(np.abs(u - v) / np.maximum(np.abs(v), 1e-300)))
            for u, v in zip(stats(a), stats(b), strict=True)
        )
        neff = float(stats(a)[0].sum())
        out[name] = dict(
            empty_first_row_zero=empty_same,
            empty_then_bit_identical=empty_bits,
            first_row_decays_by_lam_bitwise=clock,
            first_row_param_move=pmove,
            interleaved_vs_lam2_max_rel=rel,
            n_eff=neff,
        )
    for k_, v in out.items():
        print(f"  {k_}: {v}")


# --------------------------------------------------------------------------
# 5. State size and cost
# --------------------------------------------------------------------------
def state_table():
    print(
        "bytes = 8 K (1 + k + k(k+1)/2) full, 8 K (1 + 2k) diag; flops a row:"
        " full E = K (k^3/3 + k^2 + 4k) (refactor) or K (2k^2 + 4k) (held factor),"
        " update K (k^2 + 3k); diag E = K 4k, update K 4k"
    )
    print("| k | K | full bytes | diag bytes | full flops (refactor / held) | diag flops |")
    for k in (5, 20):
        for K in (2, 4, 8):
            fb = 8 * K * (1 + k + k * (k + 1) // 2)
            db = 8 * K * (1 + 2 * k)
            fr = K * (k**3 // 3 + k * k + 4 * k) + K * (k * k + 3 * k)
            fh = K * (2 * k * k + 4 * k) + K * (k * k + 3 * k)
            df = K * 8 * k
            print(f"| {k} | {K} | {fb} | {db} | {fr} / {fh} | {df} |")


if __name__ == "__main__":
    what = sys.argv[1:] or ["zero", "state", "decay", "collapse", "sklearn"]
    if what[0] == "sklearn" and len(what) == 3:  # e.g. `sklearn full 20`: one slice
        vs_sklearn(covs=(what[1],), ks=(int(what[2]),))
        sys.exit()
    for w in what:
        print(f"\n===== {w}", flush=True)
        t = time.perf_counter()
        {
            "decay": decay,
            "sklearn": vs_sklearn,
            "collapse": collapse,
            "zero": zero_weight,
            "state": state_table,
        }[w]()
        print(f"({w}: {time.perf_counter() - t:.1f} s)", flush=True)
