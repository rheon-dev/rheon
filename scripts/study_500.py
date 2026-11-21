"""Study 500: online GLMs (log link) by recursive IRLS, against statsmodels and glum.

One script for the whole study (parts 1 to 3; the record is
docs/records/STUDY-500-GLM.md):

- checks       self-checks: zero-weight rows, the sums form against the increment
               form, the recursion against the same recursion driven by statsmodels'
               family objects, zero counts, Gamma y = 0
- variants     Poisson, Gamma and Tweedie (p = 1.5), each online variant against
               statsmodels (unpenalised IRLS with freq_weights for bse; the ridge path
               with alpha matched) and glum (alpha matched):
                 A        ridge R = 1, no start-up refit (the plain recursion)
                 B        R = 10, no refit
                 C        R = 1, refit of the first 200 rows (the baseline)
                 periodic C plus a stored window of the last 2000 rows; every 1000 rows
                          the window's rows are refitted to convergence, starting from
                          the online estimate, with the rows that have left the window
                          kept as their frozen quadratic, and re-scored at that fit
                 replace  C plus the same window; every 1000 rows beta is replaced by
                          the ridge fit of the window's rows alone
                 sweeps   C, then after the start-up 2 Newton steps per row on the
                          stored window of the last 200 rows (older rows frozen)
- binomial     the log-link binomial (not well posed) and the logit link (extra)
- dispersion   variant C's gap against the Tweedie dispersion (0.5, 2, 8), k = 20
- separation   Poisson (log link) and logit separation
- decay        decay with half-lives 500 and 5000 rows
- window       an exact rolling window of 2000 rows
- cap          a cap on the per-row change in eta in place of the refit (k = 20)

The online estimator takes one Fisher-scoring (IRLS) step per row at the fit
before the row and keeps the sums

    G_t = lambda G_{t-1} + w_t W_t x_t x_t'     W_t = (dmu/deta)^2 / V(mu) at beta_{t-1}
    h_t = lambda h_{t-1} + w_t W_t z_t x_t      z_t = eta_t + (y_t - mu_t) / (dmu/deta)
    beta_t = (G_t + R I)^{-1} h_t

with eta_t = x_t' beta_{t-1}, mu_t = exp(eta_t) (lambda = 1: no decay). For the
log link dmu/deta = mu, so W = mu (Poisson, V = mu), W = 1 (Gamma, V = mu^2) and
W = mu^(2-p) (Tweedie, V = mu^p). The prediction mu_t is emitted before the
update (out of sample). A zero-weight row learns nothing (under decay it still
advances the clock).

The batch fits minimise the same penalised objective in mean form,
    (1 / sum w) * sum w_i * halfdev_i(beta) + alpha / 2 * |beta|^2,  alpha = R / sum w,
which is the online objective sum w_i halfdev_i + R/2 |beta|^2 divided by sum w.

Data are generated from seeded numpy generators; nothing is read from disk.

Run:  .venv/bin/python scripts/study_500.py --quick                 (every section, short)
      .venv/bin/python scripts/study_500.py variants --seeds 50 --procs 3 \\
          --variants C periodic replace sweeps --out <dir>/variants.json
      .venv/bin/python scripts/study_500.py all --out <dir>/all.json    (everything, long)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ETA_CLIP = 30.0  # |eta| bound before exp(); every clip is counted
TW_P = 1.5  # Tweedie variance power
TW_PHI = 2.0  # Tweedie dispersion of the generated data (about 27% zeros)
FAMILY_CODE = {"poisson": 0, "gamma": 1, "tweedie": 2}
NS = (1000, 10000, 100000)
QUICK_NS = (1000, 3000)


# ---------------------------------------------------------------- data


def make_data(
    family: str, k: int, n: int, seed: int, gamma_shape: float = 2.0, phi: float = TW_PHI
):
    """Intercept + (k-1) N(0,1) features; eta has sd about 0.5 around 0.5."""
    rng = np.random.default_rng([500, seed, k, FAMILY_CODE[family]])
    X = np.empty((n, k))
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((n, k - 1))
    beta = np.empty(k)
    beta[0] = 0.5
    beta[1:] = rng.normal(0.0, 0.5 / np.sqrt(k - 1), k - 1)
    mu = np.exp(X @ beta)
    if family == "poisson":
        y = rng.poisson(mu).astype(float)
    elif family == "gamma":
        y = rng.gamma(gamma_shape, mu / gamma_shape)
    else:  # Tweedie 1 < p < 2: a Poisson number of Gamma amounts (compound Poisson-Gamma)
        lam = mu ** (2 - TW_P) / (phi * (2 - TW_P))
        shape = (2 - TW_P) / (TW_P - 1)
        scale = phi * (TW_P - 1) * mu ** (TW_P - 1)
        N = rng.poisson(lam)
        y = np.zeros(n)
        pos = N > 0
        y[pos] = rng.gamma(N[pos] * shape, scale[pos])
    return X, y, beta


# ---------------------------------------------------------------- families


def inv_link(fam, eta):
    if fam == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    return np.exp(eta)


def irls_terms(fam, eta, mu, y):
    """Fisher weight W = (dmu/deta)^2 / V(mu) and working response z."""
    if fam == "poisson":
        return mu, eta + (y - mu) / mu
    if fam == "gamma":
        return np.ones_like(mu) if np.ndim(mu) else 1.0, eta + (y - mu) / mu
    if fam == "tweedie":
        return mu ** (2 - TW_P), eta + (y - mu) / mu
    if fam == "binlog":  # log-link binomial: dmu/deta = mu, V = mu (1 - mu)
        return mu / (1.0 - mu), eta + (y - mu) / mu
    if fam == "logit":
        v = mu * (1.0 - mu)
        return v, eta + (y - mu) / v
    raise ValueError(fam)


def halfdev(fam, eta, mu, y):
    """Half unit deviance up to a term free of beta."""
    if fam == "poisson":
        return mu - y * eta
    if fam == "gamma":
        return y / mu + eta
    if fam == "tweedie":  # -y mu^(1-p)/(1-p) + mu^(2-p)/(2-p)
        return -y * mu ** (1 - TW_P) / (1 - TW_P) + mu ** (2 - TW_P) / (2 - TW_P)
    if fam == "logit":
        return np.logaddexp(0.0, eta) - y * eta
    if fam == "binlog":  # NaN (rejected by the line search) where mu > 1
        with np.errstate(all="ignore"):
            return -(y * eta + (1.0 - y) * np.log1p(-mu))
    raise ValueError(fam)


def sm_family(fam):
    import statsmodels.api as sm

    L = sm.families.links
    return {
        "poisson": lambda: sm.families.Poisson(L.Log()),
        "gamma": lambda: sm.families.Gamma(L.Log()),
        "tweedie": lambda: sm.families.Tweedie(L.Log(), var_power=TW_P),
        "binlog": lambda: sm.families.Binomial(L.Log()),
        "logit": lambda: sm.families.Binomial(L.Logit()),
    }[fam]()


def sm_terms(fam):
    """The same W and z from statsmodels' family objects (the row-by-row oracle):
    W = family.weights(mu) = 1 / (g'(mu)^2 V(mu)), z = eta + (y - mu) g'(mu)."""
    f = sm_family(fam)

    def terms(_fam, eta, mu, y):
        mu_a = np.atleast_1d(mu)
        W = float(f.weights(mu_a)[0])
        z = float(eta + (y - mu_a[0]) * f.link.deriv(mu_a)[0])
        return W, z

    return terms


# ---------------------------------------------------------------- online


class OnlineGLM:
    """Recursive IRLS (log or logit link) with an optional start-up refit, decay,
    an exact rolling window and a cap on the per-row change in eta.

    State: G (k x k), h (k), beta (k), weight_sum, plus the refit buffer (only
    until it fills) and the window's rows (with a window)."""

    def __init__(self, k, fam, ridge=1.0, refit_rows=0, lam=1.0, window=0, cap=None, terms=None):
        self.k, self.fam, self.ridge = k, fam, ridge
        self.refit_rows, self.lam, self.window, self.cap = refit_rows, lam, window, cap
        self.terms = terms or irls_terms
        self.G = np.zeros((k, k))
        self.h = np.zeros(k)
        self.beta = np.zeros(k)
        self.weight_sum = 0.0
        self.t = 0
        self._R = ridge * np.eye(k)
        self._buf = []  # refit buffer: (t, x, y, w)
        self._win = deque()  # window: [x, y, w, W, z]
        self.clipped = 0
        self.capped = 0
        self.bad_w = 0  # rows whose Fisher weight came out non-finite or negative

    def step(self, x, y, w=1.0):
        eta = float(x @ self.beta)
        if abs(eta) > ETA_CLIP:
            self.clipped += 1
            eta = float(np.clip(eta, -ETA_CLIP, ETA_CLIP))
        mu = np.float64(inv_link(self.fam, eta))  # numpy scalar: 1/0 gives inf, not an exception
        pred = mu  # out of sample
        self.t += 1
        moved = False
        if self.lam < 1.0:  # the clock advances on every row, zero weight included
            self.G *= self.lam
            self.h *= self.lam
            self.weight_sum *= self.lam
            moved = True
        if self.window:
            self._win.append([x, y, w, 0.0, 0.0])
            if len(self._win) > self.window:
                xo, _, wo, Wo, zo = self._win.popleft()
                if wo != 0.0:
                    self.G -= (wo * Wo) * np.outer(xo, xo)
                    self.h -= (wo * Wo * zo) * xo
                    self.weight_sum -= wo
                    moved = True
        if w == 0.0:
            if moved:  # R > 0: G + R I is positive definite, nothing divides by zero
                self.beta = np.linalg.solve(self.G + self._R, self.h)
            return pred
        W, z = self.terms(self.fam, eta, mu, y)
        if not np.isfinite(W) or W < 0 or not np.isfinite(z):
            self.bad_w += 1
        if self.window:
            self._win[-1][3], self._win[-1][4] = W, z
        self.G += (w * W) * np.outer(x, x)
        self.h += (w * W * z) * x
        self.weight_sum += w
        prop = np.linalg.solve(self.G + self._R, self.h)
        if self.cap is not None:
            de = float(x @ (prop - self.beta))
            if abs(de) > self.cap:
                self.capped += 1
                prop = self.beta + (prop - self.beta) * (self.cap / abs(de))
                self.h = (self.G + self._R) @ prop
        self.beta = prop
        if self.refit_rows:
            self._buf.append((self.t, x.copy(), y, w))
            if len(self._buf) == self.refit_rows:
                self._refit()
        return pred

    def _refit(self):
        """Start-up remedy: once the buffer holds refit_rows weighted rows, fit them
        to convergence and rebuild G, h at that fit; the buffer is then dropped."""
        ts = np.array([b[0] for b in self._buf])
        Xb = np.array([b[1] for b in self._buf])
        yb = np.array([b[2] for b in self._buf])
        wb = np.array([b[3] for b in self._buf]) * self.lam ** (self.t - ts)
        beta = irls_ridge(Xb, yb, wb, self.fam, self.ridge)
        eta = np.clip(Xb @ beta, -ETA_CLIP, ETA_CLIP)
        mu = inv_link(self.fam, eta)
        W, z = irls_terms(self.fam, eta, mu, yb)
        W = np.broadcast_to(W, eta.shape)
        self.G = (Xb * (wb * W)[:, None]).T @ Xb
        self.h = (self.G + self._R) @ beta
        self.beta = beta
        if self.window:  # re-score the buffered rows still in the window at the refit
            byt = {int(t): i for i, t in enumerate(ts)}
            first = self.t - len(self._win) + 1
            for j, row in enumerate(self._win):
                i = byt.get(first + j)
                if i is not None:
                    row[3], row[4] = float(W[i]), float(z[i])
        self._buf = []
        self.refit_rows = 0


class WindowRefitGLM:
    """Variant C (start-up refit of the first `refit_rows` rows) plus a stored
    window of the last `win` rows (x, y, w and the W_i, z_i each was last scored at).

    The rows that have left the window are kept only as their frozen quadratic
    (G_out, h_out: sum of w W x x' and w W z x at their last scoring), so the
    objective the window refit and the sweeps minimise is

        F(b) = 1/2 b' G_out b - h_out' b + sum_{i in window} w_i halfdev_i(b) + R/2 |b|^2.

    mode "rescore": the plain recursion row by row; every `every` rows, Newton with
        step halving on F to convergence from the online beta, then every window row
        is re-scored at that fit: G = G_out + sum_win w W(b) x x', h = (G + R I) b.
    mode "replace": the plain recursion; every `every` rows beta is replaced by the
        ridge fit of the window's rows alone (started at the online beta) and
        h = (G + R I) beta (G keeps every row).
    mode "sweeps": after the start-up, `newton` Newton steps (with step halving) on F
        per row; the row that leaves the window is frozen at the current fit.
    """

    def __init__(
        self, k, fam, ridge=1.0, refit_rows=200, win=2000, every=1000, newton=None, mode="rescore"
    ):
        assert win >= refit_rows > 0
        self.k, self.fam, self.ridge = k, fam, ridge
        self.refit_rows, self.win, self.every = refit_rows, win, every
        self.newton, self.mode = newton, mode
        self.G = np.zeros((k, k))
        self.h = np.zeros(k)
        self.beta = np.zeros(k)
        self.G_out = np.zeros((k, k))
        self.h_out = np.zeros(k)
        self.Xw = np.zeros((win, k))
        self.yw = np.zeros(win)
        self.ww = np.zeros(win)
        self.Wsc = np.zeros(win)
        self.zsc = np.zeros(win)
        self.t = 0
        self.n_in = 0
        self.started = False
        self._R = ridge * np.eye(k)
        self.clipped = 0
        self.iters = 0  # Newton iterations after the start-up
        self.refits = 0
        self.halvings = 0

    def step(self, x, y, w=1.0):
        eta = float(x @ self.beta)
        if abs(eta) > ETA_CLIP:
            self.clipped += 1
            eta = float(np.clip(eta, -ETA_CLIP, ETA_CLIP))
        mu = np.float64(np.exp(eta))
        pred = mu  # out of sample
        slot = self.t % self.win
        if self.n_in == self.win:  # the oldest row leaves: it becomes frozen
            wo = self.ww[slot]
            if wo != 0.0:
                xo = self.Xw[slot]
                if self.mode == "sweeps" and self.started:  # frozen at the current fit
                    eo = float(np.clip(xo @ self.beta, -ETA_CLIP, ETA_CLIP))
                    Wo, zo = irls_terms(self.fam, eo, np.exp(eo), self.yw[slot])
                else:
                    Wo, zo = self.Wsc[slot], self.zsc[slot]
                self.G_out += (wo * Wo) * np.outer(xo, xo)
                self.h_out += (wo * Wo * zo) * xo
        else:
            self.n_in += 1
        self.Xw[slot], self.yw[slot], self.ww[slot] = x, y, w
        self.t += 1
        if self.mode == "sweeps" and self.started:
            self.beta = self._newton(self.beta, self.newton)
            return pred
        if w != 0.0:
            W, z = irls_terms(self.fam, eta, mu, y)
            self.Wsc[slot], self.zsc[slot] = W, z
            self.G += (w * W) * np.outer(x, x)
            self.h += (w * W * z) * x
            self.beta = np.linalg.solve(self.G + self._R, self.h)
        else:
            self.Wsc[slot] = self.zsc[slot] = 0.0
        if not self.started:
            if self.t == self.refit_rows:  # variant C's refit (G_out is still zero)
                n = self.n_in
                b = irls_ridge(self.Xw[:n], self.yw[:n], self.ww[:n], self.fam, self.ridge)
                self._rescore(b)
                self.started = True
        elif self.mode in ("rescore", "replace") and self.t % self.every == 0:
            self.refits += 1
            n = self.n_in
            if self.mode == "replace":
                b = irls_ridge(
                    self.Xw[:n], self.yw[:n], self.ww[:n], self.fam, self.ridge, start=self.beta
                )
                self.beta = b
                self.h = (self.G + self._R) @ b
            else:
                self._rescore(self._newton(self.beta, None))
        return pred

    def _rescore(self, b):
        n = self.n_in
        Xw, ww = self.Xw[:n], self.ww[:n]
        eta = np.clip(Xw @ b, -ETA_CLIP, ETA_CLIP)
        W, z = irls_terms(self.fam, eta, np.exp(eta), self.yw[:n])
        W = np.broadcast_to(W, eta.shape)
        self.Wsc[:n], self.zsc[:n] = W, z
        self.G = self.G_out + (Xw * (ww * W)[:, None]).T @ Xw
        self.beta = b
        self.h = (self.G + self._R) @ b

    def _newton(self, b, steps, tol=1e-13):
        n = self.n_in
        X, yv, wv = self.Xw[:n], self.yw[:n], self.ww[:n]
        Go, ho, R, fam = self.G_out, self.h_out, self.ridge, self.fam

        def F(c):
            e = np.clip(X @ c, -ETA_CLIP, ETA_CLIP)
            return float(
                0.5 * c @ Go @ c - ho @ c + wv @ halfdev(fam, e, np.exp(e), yv) + 0.5 * R * c @ c
            )

        f = F(b)
        for _ in range(steps or 100):
            eta = np.clip(X @ b, -ETA_CLIP, ETA_CLIP)
            W, z = irls_terms(fam, eta, np.exp(eta), yv)
            W = np.broadcast_to(W, eta.shape)
            g = ho - Go @ b + X.T @ (wv * W * (z - eta)) - R * b
            H = Go + (X * (wv * W)[:, None]).T @ X + self._R
            d = np.linalg.solve(H, g)
            t, fn = 1.0, f
            while t > 1e-12:
                fn = F(b + t * d)
                if fn <= f + 1e-14 * (1 + abs(f)):  # at the optimum F moves by rounding only
                    break
                t *= 0.5
                self.halvings += 1
            b = b + t * d
            self.iters += 1
            done = abs(f - fn) <= tol * (1 + abs(f)) and np.abs(t * d).max() < 1e-11
            f = fn
            if steps is None and done:
                break
        return b

    def state_floats(self):
        """Floats the state holds after the start-up (the symmetric matrices packed)."""
        k, W = self.k, self.win
        sym = k * (k + 1) // 2
        if self.mode == "sweeps":  # G_out, h_out, beta, the window's x, y, w
            return sym + 2 * k + W * (k + 2)
        if self.mode == "replace":  # G, h, beta, the window's x, y, w
            return sym + 2 * k + W * (k + 2)
        return 2 * sym + 3 * k + W * (k + 4)  # G, G_out, h, h_out, beta, x, y, w, W_i, z_i


# ---------------------------------------------------------------- batch


def irls_ridge(X, y, w, fam, ridge, tol=1e-13, maxiter=300, start=None):
    """Penalised IRLS, sum form: sum w_i halfdev_i + R/2 |beta|^2, step halving."""
    k = X.shape[1]
    if start is None:
        beta = np.zeros(k)
        ym = max(float(np.average(y, weights=w)), 1e-12)
        beta[0] = np.log(ym) if fam != "logit" else np.log(ym / max(1 - ym, 1e-12))
    else:
        beta = np.array(start, dtype=float)

    def obj(b):
        eta = np.clip(X @ b, -ETA_CLIP, ETA_CLIP)
        return float(w @ halfdev(fam, eta, inv_link(fam, eta), y) + 0.5 * ridge * b @ b)

    f = obj(beta)
    for _ in range(maxiter):
        eta = np.clip(X @ beta, -ETA_CLIP, ETA_CLIP)
        mu = inv_link(fam, eta)
        W, z = irls_terms(fam, eta, mu, y)
        W = np.broadcast_to(W, eta.shape)
        g = X.T @ (w * W * (z - eta)) - ridge * beta
        H = (X * (w * W)[:, None]).T @ X + ridge * np.eye(k)
        d = np.linalg.solve(H, g)
        t, fn = 1.0, f
        while t > 1e-12:
            fn = obj(beta + t * d)
            if fn <= f:
                break
            t *= 0.5
        beta = beta + t * d
        done = abs(f - fn) <= tol * (1 + abs(f)) and np.abs(t * d).max() < 1e-11
        f = fn
        if done:
            break
    return beta


def batch_refs(X, y, fam, ridge, use_glum=True):
    """Every batch fit of the n rows (all weights 1) the online fit is compared with.

    ref      penalised IRLS (sum form, ridge R), tolerance 1e-13
    sm       statsmodels GLM(freq_weights=1) unpenalised IRLS: params and bse
             (Gamma and Tweedie with the Pearson scale)
    sm_ridge statsmodels' ridge path (fit_regularized, L1_wt=0, alpha = R/n), a BFGS
             minimisation of -loglike/nobs + alpha/2 |beta|^2; loglike is replaced by
             -deviance/2 from statsmodels' family object (Gamma's and Tweedie's own
             loglike profile the scale or use a series), score is statsmodels' own
             at scale 1
    glum     GeneralizedLinearRegressor, alpha = R/n, no intercept column of its own
    """
    import statsmodels.api as sm

    n = len(y)
    alpha = ridge / n
    ones = np.ones(n)
    out = {"ref": irls_ridge(X, y, ones, fam, ridge)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(y, X, family=sm_family(fam), freq_weights=ones).fit(
            method="IRLS", tol=1e-12, maxiter=200
        )
        out["sm"], out["bse"] = np.asarray(f.params), np.asarray(f.bse)
        mr = sm.GLM(y, X, family=sm_family(fam), freq_weights=ones)
        fo, sc = mr.family, mr.score
        mr.loglike = lambda p, scale=None: -0.5 * fo.deviance(y, mr.predict(p))
        mr.score = lambda p, scale=None: sc(p, scale=1.0)
        r = mr.fit_regularized(alpha=alpha, L1_wt=0.0, start_params=out["sm"])
    out["sm_ridge"] = np.asarray(r.params)
    grad = -np.asarray(sc(out["sm_ridge"], scale=1.0)) / n + alpha * out["sm_ridge"]
    out["sm_ridge_grad"] = float(np.abs(grad).max())
    if use_glum:
        out["glum"] = glum_fit(X, y, fam, ones, ridge)
    return out


def glum_fit(X, y, fam, wts, ridge):
    from glum import GeneralizedLinearRegressor, TweedieDistribution

    keep = wts > 0
    family = {"logit": "binomial", "binlog": "binomial"}.get(fam, fam)
    if fam == "tweedie":
        family = TweedieDistribution(TW_P)
    g = GeneralizedLinearRegressor(
        family=family,
        link="log" if fam != "logit" else "logit",
        alpha=ridge / wts.sum(),
        l1_ratio=0.0,
        fit_intercept=False,  # intercept is column 0 and penalised like the rest
        gradient_tol=1e-10,
        max_iter=1000,
    )
    g.fit(X[keep], y[keep], sample_weight=wts[keep])
    return np.asarray(g.coef_)


def sm_bse(X, y, fam, wts):
    """Unpenalised statsmodels IRLS with the decay weights as freq_weights and as
    var_weights (same point estimate), plus the sandwich (HC0) SE computed by hand."""
    import statsmodels.api as sm

    keep = wts > 0
    X, y, wts = X[keep], y[keep], wts[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = sm.GLM(y, X, family=sm_family(fam), freq_weights=wts).fit(tol=1e-12, maxiter=200)
        v = sm.GLM(y, X, family=sm_family(fam), var_weights=wts).fit(tol=1e-12, maxiter=200)
    b = np.asarray(f.params)
    eta = X @ b
    mu = np.exp(eta)
    W, z = irls_terms(fam, eta, mu, y)
    W = np.broadcast_to(W, eta.shape)
    sc = (wts * W * (z - eta))[:, None] * X  # per-row score
    A = np.linalg.inv((X * (wts * W)[:, None]).T @ X)
    sand = A @ (sc.T @ sc) @ A
    return {
        "b": b,
        "bse_freq": np.asarray(f.bse),
        "bse_var": np.asarray(v.bse),
        "bse_sand": np.sqrt(np.diag(sand)),
        "params_diff": float(np.abs(b - np.asarray(v.params)).max()),
    }


# ---------------------------------------------------------------- checks


def check_zero_weight(family: str, k: int = 5, n: int = 300) -> None:
    """A zero-weight row (including the first) leaves the state bit-identical."""
    X, y, _ = make_data(family, k, n, seed=999)
    a, b = OnlineGLM(k, family, 1.0), OnlineGLM(k, family, 1.0)
    rng = np.random.default_rng(7)
    zero = rng.random(n) < 0.2
    zero[0] = True
    for i in range(n):
        b.step(X[i], y[i], 0.0 if zero[i] else 1.0)
        if not zero[i]:
            a.step(X[i], y[i], 1.0)
    assert np.array_equal(a.beta, b.beta) and np.array_equal(a.G, b.G)
    assert np.array_equal(a.h, b.h) and np.all(np.isfinite(b.beta))


def check_newton_form(family: str, k: int = 5, n: int = 500) -> tuple[float, int]:
    """The sums form equals the increment form beta += (G+R)^{-1} w x s_t,
    s_t = W_t (z_t - eta_t) = (y - mu) (dmu/deta) / V(mu). The plain recursion
    (no refit) diverges on some seeds (Tweedie at seed 998), so the check takes the
    first seed from 998 down on which it does not clip; returns (max |dbeta|, seed)."""
    for seed in range(998, 900, -1):
        X, y, _ = make_data(family, k, n, seed=seed)
        o = OnlineGLM(k, family, 1.0)
        beta = np.zeros(k)
        G = np.eye(k)  # G + R I with R = 1
        worst = 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n):
                x = X[i]
                eta = x @ beta
                W, z = irls_terms(family, eta, np.exp(eta), y[i])
                G = G + W * np.outer(x, x)
                o.step(x, y[i])
                if o.clipped or not np.isfinite(G).all():
                    break
                beta = beta + np.linalg.solve(G, x * W * (z - eta))
                worst = max(worst, float(np.abs(beta - o.beta).max()))
        if not o.clipped and np.isfinite(G).all():
            return worst, seed
    raise AssertionError("no seed without a clip")


def check_family_oracle(family: str, k: int = 5, n: int = 2000) -> float:
    """The recursion against the same recursion with W and z taken from
    statsmodels' family objects (family.weights, link.deriv): max |dbeta| over the
    path, refit of the first 200 rows included."""
    X, y, _ = make_data(family, k, n, seed=995)
    a = OnlineGLM(k, family, 1.0, refit_rows=200)
    b = OnlineGLM(k, family, 1.0, refit_rows=200, terms=sm_terms(family))
    worst = 0.0
    for i in range(n):
        a.step(X[i], y[i])
        b.step(X[i], y[i])
        worst = max(worst, float(np.abs(a.beta - b.beta).max()))
    return worst


def checks_section(quick=False):
    out = {"part1": {}}
    for fam in ("poisson", "gamma", "tweedie"):
        check_zero_weight(fam)
        worst, seed = check_newton_form(fam)
        out["part1"][fam] = {
            "zero_weight_bit_identical": True,
            "sums_vs_increment_max_dbeta": worst,
            "sums_vs_increment_seed": seed,
            "statsmodels_family_oracle_max_dbeta": check_family_oracle(
                fam, n=500 if quick else 2000
            ),
        }
    rng = np.random.default_rng([500, 4])
    for fam in ("poisson", "gamma", "tweedie"):
        k, n = 5, 3000
        X, y, _ = make_data(fam, k, n, seed=997)
        if fam == "poisson":  # many zero counts: a low mean, and the first row a zero count
            X[:, 0] = 1.0
            y = rng.poisson(np.exp(-2.0 + 0.3 * X[:, 1])).astype(float)
            y[0] = 0.0
        zero = rng.random(n) < 0.2
        zero[0] = zero[1] = True  # the first two rows carry zero weight
        y_z = y.copy()
        y_z[zero] = 7.0  # what a zero-weight row's y is must not matter
        rec = {"zero_y_share": float((y == 0).mean()), "zero_weight_rows": int(zero.sum())}
        for name, kw in (("plain", dict(refit_rows=0)), ("refit200", dict(refit_rows=200))):
            a = OnlineGLM(k, fam, **kw)
            b = OnlineGLM(k, fam, **kw)
            for i in range(n):
                b.step(X[i], y_z[i], 0.0 if zero[i] else 1.0)
                if not zero[i]:
                    a.step(X[i], y[i], 1.0)
            rec[name + "_bit_identical_to_dropped"] = bool(
                np.array_equal(a.beta, b.beta) and np.array_equal(a.G, b.G)
            )
            rec[name + "_finite"] = bool(np.all(np.isfinite(b.beta)) and np.all(np.isfinite(b.G)))
        # decay: a zero-weight row advances the clock, so the target is the batch fit
        # with those rows at weight 0 (ages still counted), not the fit with them dropped
        lam = 0.5 ** (1 / 500)
        b = OnlineGLM(k, fam, refit_rows=200, lam=lam)
        for i in range(n):
            b.step(X[i], y_z[i], 0.0 if zero[i] else 1.0)
        wts = np.where(zero, 0.0, lam ** np.arange(n - 1, -1, -1, dtype=float))
        ref = irls_ridge(X[~zero], y[~zero], wts[~zero], fam, 1.0)
        bse = sm_bse(X, y, fam, wts)["bse_freq"]
        rec["decay500_finite"] = bool(np.all(np.isfinite(b.beta)))
        rec["decay500_gap_se_vs_zero_weighted_batch"] = float((np.abs(b.beta - ref) / bse).max())
        # window: a zero-weight row occupies a slot; target = batch on the window's rows
        Wn = 1000
        b = OnlineGLM(k, fam, refit_rows=200, window=Wn)
        for i in range(n):
            b.step(X[i], y_z[i], 0.0 if zero[i] else 1.0)
        sl = slice(n - Wn, n)
        keep = ~zero[sl]
        ref = irls_ridge(X[sl][keep], y[sl][keep], np.ones(keep.sum()), fam, 1.0)
        bse = sm_bse(X[sl][keep], y[sl][keep], fam, np.ones(keep.sum()))["bse_freq"]
        rec["window1000_finite"] = bool(np.all(np.isfinite(b.beta)))
        rec["window1000_gap_se_vs_batch"] = float((np.abs(b.beta - ref) / bse).max())
        out[fam] = rec
    # Gamma with y = 0 (outside its support): what the online fit and the batch fits do
    X, y, _ = make_data("gamma", 5, 2000, seed=996)
    y[::50] = 0.0
    o, _, _ = online_path(X, y, "gamma", (), ridge=1.0, refit_rows=200)
    _, err, msgs = capture(sm_fit(X, y, "gamma"))
    _, gerr, gmsgs = capture(lambda: glum_fit(X, y, "gamma", np.ones(len(y)), 1e-9))
    out["gamma_y0"] = {
        "online_finite": bool(np.all(np.isfinite(o.beta))),
        "sm_err": err,
        "sm_warnings": msgs,
        "glum_err": gerr,
        "glum_warnings": gmsgs,
    }
    return out


# ---------------------------------------------------------------- variants (parts 1 and 3)

VARIANTS = {
    "A": dict(cls="plain", ridge=1.0, refit_rows=0),
    "B": dict(cls="plain", ridge=10.0, refit_rows=0),
    "C": dict(cls="plain", ridge=1.0, refit_rows=200),
    "periodic": dict(cls="window", mode="rescore", win=2000, every=1000, newton=None),
    "replace": dict(cls="window", mode="replace", win=2000, every=1000, newton=None),
    "sweeps": dict(cls="window", mode="sweeps", win=200, every=1, newton=2),
}


def make_online(variant, k, fam):
    v = VARIANTS[variant]
    if v["cls"] == "plain":
        return OnlineGLM(k, fam, v["ridge"], v["refit_rows"]), v["ridge"]
    return WindowRefitGLM(k, fam, 1.0, 200, v["win"], v["every"], v["newton"], v["mode"]), 1.0


def run_variants(job):
    """One (family, k, seed) stream: every variant's online pass, then the batch fits
    at each checkpoint, computed once and shared by the variants of the same ridge."""
    fam, k, seed, variants, ns, use_glum = job
    n = max(ns)
    X, y, beta_true = make_data(fam, k, n, seed)
    snaps, info = {}, {}
    for v in variants:
        o, ridge = make_online(v, k, fam)
        snaps[v] = {}
        t0 = time.perf_counter()
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n):
                o.step(X[i], y[i])
                if i + 1 in ns:
                    snaps[v][i + 1] = o.beta.copy()
        info[v] = {
            "ridge": ridge,
            "online_s": time.perf_counter() - t0,
            "clipped": o.clipped,
            "newton_iters": getattr(o, "iters", 0),
            "refits": getattr(o, "refits", 0),
            "halvings": getattr(o, "halvings", 0),
            "state_floats": o.state_floats() if hasattr(o, "state_floats") else None,
        }
    out = {"fam": fam, "k": k, "seed": seed, "variants": info, "at": {}}
    for m in ns:
        Xm, ym = X[:m], y[:m]
        refs = {}
        rec = {}
        for v in variants:
            ridge = info[v]["ridge"]
            if ridge not in refs:
                refs[ridge] = batch_refs(Xm, ym, fam, ridge, use_glum)
            r = refs[ridge]
            bo, se = snaps[v][m], r["bse"]
            d = {}
            for name in ("ref", "sm", "sm_ridge", "glum"):
                if name in r:
                    d[f"{name}_abs"] = float(np.abs(bo - r[name]).max())
                    d[f"{name}_se"] = float((np.abs(bo - r[name]) / se).max())
            d["true_online_se"] = float((np.abs(bo - beta_true) / se).max())
            d["true_batch_se"] = float((np.abs(r["sm"] - beta_true) / se).max())
            rec[v] = d
        for ridge, r in refs.items():
            rec[f"floor_R{ridge:g}"] = {
                "sm_ridge_grad": r["sm_ridge_grad"],
                "sm_ridge_vs_ref_se": float((np.abs(r["sm_ridge"] - r["ref"]) / r["bse"]).max()),
                "glum_vs_ref_se": float((np.abs(r["glum"] - r["ref"]) / r["bse"]).max())
                if "glum" in r
                else None,
                "sm_vs_ref_se": float((np.abs(r["sm"] - r["ref"]) / r["bse"]).max()),
                "bse_max": float(r["bse"].max()),
            }
        out["at"][m] = rec
    return out


def summarise_variants(rows, ns):
    """Mean and max over seeds (non-finite counted as failed) per family, k, variant, n."""
    tab = {}
    for r in rows:
        for m in ns:
            rec = r["at"][m] if m in r["at"] else r["at"][str(m)]
            for v, d in rec.items():
                if v.startswith("floor"):
                    continue
                for key, val in d.items():
                    tab.setdefault((r["fam"], r["k"], v, m), {}).setdefault(key, []).append(val)
    out = []
    for (fam, k, v, m), d in sorted(tab.items()):
        row = {"fam": fam, "k": k, "variant": v, "n": m, "seeds": len(d["ref_se"])}
        for key, vals in d.items():
            a = np.asarray(vals, dtype=float)
            ok = np.isfinite(a) & (a <= 10) if key.endswith("_se") else np.isfinite(a)
            row[key] = [
                float(a[ok].mean()) if ok.any() else None,
                float(a[ok].max()) if ok.any() else None,
                int((~ok).sum()),
            ]
        out.append(row)
    return out


def dispersion_section(seeds=5, n=100000, phis=(0.5, 2.0, 8.0)):
    """Variant C's gap (in batch SE) against the Tweedie dispersion phi, k = 20: the
    frozen-weight bias grows like phi while the SE grows like sqrt(phi)."""
    import statsmodels.api as sm

    out = {}
    for phi in phis:
        gaps, zeros = [], []
        for seed in range(seeds):
            X, y, _ = make_data("tweedie", 20, n, seed, phi=phi)
            zeros.append(float((y == 0).mean()))
            o = OnlineGLM(20, "tweedie", 1.0, 200)
            with np.errstate(over="ignore", invalid="ignore"):
                for i in range(n):
                    o.step(X[i], y[i])
            ref = irls_ridge(X, y, np.ones(n), "tweedie", 1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = sm.GLM(y, X, family=sm_family("tweedie")).fit(tol=1e-12, maxiter=200)
            gaps.append(float((np.abs(o.beta - ref) / f.bse).max()))
        out[phi] = {"zero_share": float(np.mean(zeros)), "gap_se": gaps}
        print(
            f"dispersion phi={phi}: zeros {np.mean(zeros):.2f} C gap SE mean/max "
            f"{np.mean(gaps):.3f}/{np.max(gaps):.3f}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------- decay / window (part 2)


def run_stream(job):
    """One (family, k, seed) stream under decay or a window, compared at ns."""
    fam, k, seed, mode, par, ns = job
    n = max(ns)
    X, y, _ = make_data(fam, k, n, seed)
    lam = 0.5 ** (1.0 / par) if mode == "decay" else 1.0
    o = OnlineGLM(k, fam, 1.0, 200, lam=lam, window=par if mode == "window" else 0)
    snaps = {}
    t0 = time.perf_counter()
    for i in range(n):
        o.step(X[i], y[i])
        if i + 1 in ns:
            snaps[i + 1] = (o.beta.copy(), o.weight_sum)
    t_on = time.perf_counter() - t0
    out = {"fam": fam, "k": k, "seed": seed, "mode": mode, "par": par, "online_s": t_on, "at": {}}
    for m in ns:
        bo, ws_on = snaps[m]
        if mode == "decay":
            wts = lam ** np.arange(m - 1, -1, -1, dtype=float)
            lo = 0
        else:
            lo = max(0, m - par)
            wts = np.ones(m - lo)
        Xm, ym = X[lo:m], y[lo:m]
        b_ref = irls_ridge(Xm, ym, wts, fam, 1.0)
        b_gl = glum_fit(Xm, ym, fam, wts, 1.0)
        s = sm_bse(Xm, ym, fam, wts)
        se = s["bse_freq"]
        out["at"][m] = {
            "gap_abs": float(np.abs(bo - b_ref).max()),
            "gap_se": float((np.abs(bo - b_ref) / se).max()),
            "gap_se_glum": float((np.abs(bo - b_gl) / se).max()),
            "glum_vs_ref_se": float((np.abs(b_gl - b_ref) / se).max()),
            "gap_se_sm_unpen": float((np.abs(bo - s["b"]) / se).max()),
            "var_over_freq": float(np.median(s["bse_var"] / se)),
            "sand_over_freq": float(np.median(s["bse_sand"] / se)),
            "freq_var_params_diff": s["params_diff"],
            "weight_sum_online": ws_on,
            "weight_sum_batch": float(wts.sum()),
        }
    return out


# ---------------------------------------------------------------- step cap (part 2)


def run_cap(job):
    fam, k, seed, cap, lam_hl, n = job
    X, y, _ = make_data(fam, k, n, seed)
    lam = 0.5 ** (1.0 / lam_hl) if lam_hl else 1.0
    o = OnlineGLM(k, fam, 1.0, 0, lam=lam, cap=cap)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n):
            o.step(X[i], y[i])
    wts = lam ** np.arange(n - 1, -1, -1, dtype=float)
    b_ref = irls_ridge(X, y, wts, fam, 1.0)
    s = sm_bse(X, y, fam, wts)
    return {
        "fam": fam,
        "k": k,
        "seed": seed,
        "cap": cap,
        "hl": lam_hl,
        "gap_abs": float(np.abs(o.beta - b_ref).max()),
        "gap_se": float((np.abs(o.beta - b_ref) / s["bse_freq"]).max()),
        "clipped": o.clipped,
        "capped": o.capped,
        "finite": bool(np.all(np.isfinite(o.beta))),
    }


# ---------------------------------------------------------------- binomial / separation (part 2)


def capture(fn):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            r = fn()
            err = None
        except Exception as e:  # noqa: BLE001 - reported, not handled
            r, err = None, f"{type(e).__name__}: {e}"[:200]
    msgs = sorted({f"{w.category.__name__}: {str(w.message)[:120]}" for w in ws})
    return r, err, msgs


def sm_fit(X, y, fam):
    import statsmodels.api as sm

    return lambda: sm.GLM(y, X, family=sm_family(fam)).fit(maxiter=100)


def online_path(X, y, fam, ns, **kw):
    o = OnlineGLM(X.shape[1], fam, **kw)
    preds = np.empty(len(y))
    snaps = {}
    for i in range(len(y)):
        preds[i] = o.step(X[i], y[i])
        if i + 1 in ns:
            snaps[i + 1] = o.beta.copy()
    return o, preds, snaps


def online_report(X, y, fam, **kw):
    """The online path, recorded whatever happens (the log-link binomial starts at
    beta = 0, i.e. mu = 1, where its Fisher weight mu / (1 - mu) is infinite)."""
    o = OnlineGLM(X.shape[1], fam, **kw)
    preds = np.full(len(y), np.nan)
    first_nan = None
    with np.errstate(all="ignore"):
        for i in range(len(y)):
            try:
                preds[i] = o.step(X[i], y[i])
            except np.linalg.LinAlgError as e:
                return {"error_at_row": i + 1, "error": str(e), "bad_fisher_weight_rows": o.bad_w}
            if first_nan is None and not np.all(np.isfinite(o.beta)):
                first_nan = i + 1
    return {
        "beta": o.beta.tolist(),
        "first_row_beta_nonfinite": first_nan,
        "preds_above_1": int((preds > 1).sum()),
        "bad_fisher_weight_rows": o.bad_w,
        "finite": bool(np.all(np.isfinite(o.beta))),
        "clipped": o.clipped,
    }


def binomial_section(n=20000):
    out = {}
    # (a) log-link binomial, well inside (0, 1)
    rng = np.random.default_rng([500, 2, 1])
    k = 5
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta = np.array([-1.5, 0.3, -0.2, 0.15, 0.1])
    mu = np.exp(X @ beta)
    out["binlog_true_mu_max"] = float(mu.max())
    ok = mu < 1
    X, mu = X[ok], mu[ok]
    y = (rng.random(len(mu)) < mu).astype(float)
    out["binlog_rows_kept"] = int(ok.sum())
    res, err, msgs = capture(sm_fit(X, y, "binlog"))
    out["binlog_sm"] = {
        "err": err,
        "warnings": msgs,
        "params": None if res is None else np.asarray(res.params).tolist(),
        "max_fitted_mu": None if res is None else float(np.max(res.fittedvalues)),
    }
    res, err, msgs = capture(lambda: glum_fit(X, y, "binlog", np.ones(len(y)), 1e-9))
    out["binlog_glum"] = {
        "err": err,
        "warnings": msgs,
        "coef": None if res is None else res.tolist(),
    }
    import statsmodels.api as sm

    start = np.r_[np.log(y.mean()), np.zeros(k - 1)]
    res, err, msgs = capture(
        lambda: sm.GLM(y, X, family=sm_family("binlog")).fit(start_params=start, maxiter=100)
    )
    b_ref = irls_ridge(X, y, np.ones(len(y)), "binlog", 1.0)
    out["binlog_sm_started"] = {
        "err": err,
        "warnings": msgs,
        "params": None if res is None else np.asarray(res.params).tolist(),
        "bse": None if res is None else np.asarray(res.bse).tolist(),
    }
    out["binlog_ridge_irls"] = b_ref.tolist()
    for refit in (0, 200):
        out[f"binlog_online_refit{refit}"] = online_report(
            X, y, "binlog", ridge=1.0, refit_rows=refit
        )
    # the logit link on the same rows, as a labelled extra (not the log link)
    out["logit_extra_online_refit200"] = online_report(X, y, "logit", ridge=1.0, refit_rows=200)
    b_ref = irls_ridge(X, y, np.ones(len(y)), "logit", 1.0)
    out["logit_extra_gap_abs_vs_ridge_batch"] = float(
        np.abs(np.array(out["logit_extra_online_refit200"]["beta"]) - b_ref).max()
    )
    res, err, msgs = capture(sm_fit(X, y, "logit"))
    if res is not None:
        out["logit_extra_gap_se_vs_sm"] = float(
            (
                np.abs(
                    np.array(out["logit_extra_online_refit200"]["beta"]) - np.asarray(res.params)
                )
                / np.asarray(res.bse)
            ).max()
        )
    # (b) the same features, a truth whose mu would exceed 1 on some rows
    beta2 = np.array([-0.5, 0.6, 0.0, 0.0, 0.0])
    rng = np.random.default_rng([500, 2, 2])
    X2 = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    p = np.minimum(np.exp(X2 @ beta2), 1.0)
    y2 = (rng.random(n) < p).astype(float)
    out["binlog_capped_truth_rows_mu_ge_1"] = int((np.exp(X2 @ beta2) >= 1).sum())
    res, err, msgs = capture(sm_fit(X2, y2, "binlog"))
    out["binlog2_sm"] = {
        "err": err,
        "warnings": msgs,
        "max_fitted_mu": None if res is None else float(np.max(res.fittedvalues)),
    }
    res, err, msgs = capture(lambda: glum_fit(X2, y2, "binlog", np.ones(n), 1e-9))
    out["binlog2_glum"] = {"err": err, "warnings": msgs}
    out["binlog2_online_refit200"] = online_report(X2, y2, "binlog", ridge=1.0, refit_rows=200)
    return out


def separation_section(ns=NS):
    """Poisson (log link, in scope): x1 is a 0/1 indicator and y = 0 on every row
    with x1 = 1, so the MLE's coefficient on x1 is -infinity. Logit (extra):
    y = 1 iff x1 > 0, complete separation, slope +infinity."""
    out = {}
    for fam in ("poisson", "logit"):
        rng = np.random.default_rng([500, 3, 0 if fam == "poisson" else 1])
        n, k = max(ns), 5
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        if fam == "poisson":
            X[:, 1] = (X[:, 1] > 0).astype(float)
            mu = np.exp(0.5 + 0.3 * X[:, 2])
            y = np.where(X[:, 1] == 1.0, 0.0, rng.poisson(mu).astype(float))
        else:
            y = (X[:, 1] > 0).astype(float)
        o, preds, snaps = online_path(X, y, fam, ns, ridge=1.0, refit_rows=200)
        rec = {"online_finite": bool(np.all(np.isfinite(o.beta))), "clipped": o.clipped, "at": {}}
        for m in ns:
            Xm, ym = X[:m], y[:m]
            r_sm, err_sm, w_sm = capture(sm_fit(Xm, ym, fam))
            ones = np.ones(m)
            r_gl0, err_gl0, w_gl0 = capture(
                lambda a=Xm, b=ym, c=fam, d=ones: glum_fit(a, b, c, d, 1e-12)
            )
            b_ref = irls_ridge(Xm, ym, np.ones(m), fam, 1.0)
            r_gl, err_gl, _ = capture(lambda a=Xm, b=ym, c=fam, d=ones: glum_fit(a, b, c, d, 1.0))
            rec["at"][m] = {
                "online_b1": float(snaps[m][1]),
                "ridge_ref_b1": float(b_ref[1]),
                "glum_ridge_b1": None if r_gl is None else float(r_gl[1]),
                "online_vs_ridge_abs": float(np.abs(snaps[m] - b_ref).max()),
                "sm_b1": None if r_sm is None else float(np.asarray(r_sm.params)[1]),
                "sm_bse1": None if r_sm is None else float(np.asarray(r_sm.bse)[1]),
                "sm_converged": None if r_sm is None else bool(r_sm.converged),
                "sm_err": err_sm,
                "sm_warnings": w_sm,
                "glum_unpen_b1": None if r_gl0 is None else float(r_gl0[1]),
                "glum_unpen_err": err_gl0,
                "glum_unpen_warnings": w_gl0,
            }
        out[fam] = rec
    return out


# ---------------------------------------------------------------- driver


def _summ(rows, key_fields, vals):
    tab = {}
    for r in rows:
        key = tuple(r[f] for f in key_fields)
        for m, rec in r["at"].items():
            for v in vals:
                tab.setdefault(key + (int(m),), {}).setdefault(v, []).append(rec[v])
    return [
        {
            **dict(zip(key_fields + ("n",), key, strict=True)),
            **{v: [float(np.mean(x)), float(np.max(x))] for v, x in d.items()},
        }
        for key, d in sorted(tab.items())
    ]


def _fmt(x, f=".3f"):
    return "  -  " if x is None else format(x, f)


def _mm(row, key):
    v = row.get(key)
    if v is None:
        return "n/a"
    return f"{_fmt(v[0])}/{_fmt(v[1])}" + (f" ({v[2]} failed)" if v[2] else "")


def print_variants(summary):
    for row in summary:
        mm = {key: _mm(row, key) for key in ("sm_ridge_se", "sm_se", "glum_se", "ref_se")}
        print(
            f"{row['fam']:7s} k={row['k']:2d} {row['variant']:8s} n={row['n']:6d} "
            f"[{row['seeds']} seeds] SE mean/max: sm_ridge {mm['sm_ridge_se']} "
            f"sm {mm['sm_se']} glum {mm['glum_se']} ref {mm['ref_se']} | "
            f"abs max sm_ridge {_fmt(row['sm_ridge_abs'][1], '.2e')}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sections", nargs="*", default=["all"])
    ap.add_argument("--quick", action="store_true", help="every section, short: about 30 seconds")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--families", nargs="+", default=["poisson", "gamma", "tweedie"])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--no-glum", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    secs = set(a.sections)
    if "all" in secs:
        secs = {
            "checks",
            "variants",
            "dispersion",
            "binomial",
            "separation",
            "decay",
            "window",
            "cap",
        }
    ns = QUICK_NS if a.quick else NS
    seeds = 1 if a.quick else a.seeds
    use_glum = not a.no_glum
    res = {"args": vars(a)}
    if a.out and os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f) | {"args": vars(a)}
    t0 = time.perf_counter()
    if "checks" in secs:
        res["checks"] = checks_section(a.quick)
        print("checks", json.dumps(res["checks"], indent=1, default=str), flush=True)
    if "binomial" in secs:
        res["binomial"] = binomial_section(5000 if a.quick else 20000)
        print("binomial", json.dumps(res["binomial"], indent=1), flush=True)
    if "dispersion" in secs:
        res["dispersion"] = dispersion_section(1 if a.quick else 5, max(ns))
    if "separation" in secs:
        res["separation"] = separation_section(ns)
        print("separation", json.dumps(res["separation"], indent=1), flush=True)
    var_jobs = []
    if "variants" in secs:
        # plus a checkpoint half-way between two window refits (they fall on multiples
        # of 1000 rows, so every other checkpoint sees the state just after one)
        vns = tuple(sorted(set(ns) | {max(ns) - 500}))
        var_jobs = [
            (f, k, s, tuple(a.variants), vns, use_glum)
            for s in range(seeds)
            for f in a.families
            for k in a.ks
        ]
    stream_jobs = []
    if "decay" in secs:
        stream_jobs += [
            (f, k, s, "decay", hl, ns)
            for hl in (500, 5000)
            for f in ("poisson", "gamma")
            for k in a.ks
            for s in range(seeds)
        ]
    if "window" in secs:
        stream_jobs += [
            (f, k, s, "window", 2000, ns)
            for f in ("poisson", "gamma")
            for k in a.ks
            for s in range(seeds)
        ]
    cap_cfgs = ((None, 0), (1.0, 0), (2.0, 0), (None, 5000), (1.0, 5000))
    cap_jobs = []
    if "cap" in secs:
        cap_jobs = [
            (f, 20, s, cap, hl, max(ns))
            for f in ("poisson", "gamma")
            for s in range(seeds)
            for cap, hl in cap_cfgs
        ]
    # spawn, and one future per job: a dead worker raises instead of hanging the run.
    # One job per worker process: a long-lived worker grows by about 250 MB a job
    # (5 GB after some 20 jobs at 100k rows; the cause is in the batch libraries'
    # or the allocator's keeping, not in the online state) until it is killed.
    # Each result is also appended to <out>.partial.jsonl as it arrives.
    partial = a.out + ".partial.jsonl" if a.out else None
    with ProcessPoolExecutor(
        a.procs, mp_context=mp.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        futs = {pool.submit(run_variants, j): "variants" for j in var_jobs}
        futs |= {pool.submit(run_stream, j): "streams" for j in stream_jobs}
        futs |= {pool.submit(run_cap, j): "cap" for j in cap_jobs}
        got = {"variants": [], "streams": [], "cap": []}
        for fut in as_completed(futs):
            kind = futs[fut]
            r = fut.result()
            got[kind].append(r)
            if partial:
                with open(partial, "a") as pf:
                    pf.write(json.dumps({"kind": kind, **r}, default=str) + "\n")
            el = f"[{time.perf_counter() - t0:.0f}s]"
            if kind == "variants":
                last = r["at"][max(ns)]
                print(
                    f"variants {r['fam']} k={r['k']} s={r['seed']} "
                    + " ".join(
                        f"{v}={last[v]['sm_ridge_se']:.3g}/{r['variants'][v]['online_s']:.1f}s"
                        for v in r["variants"]
                    )
                    + f" {el}",
                    flush=True,
                )
            elif kind == "streams":
                last = r["at"][max(ns)]
                print(
                    f"{r['mode']} {r['par']} {r['fam']} k={r['k']} s={r['seed']} "
                    f"gap_se={last['gap_se']:.3g} glum={last['gap_se_glum']:.3g} {el}",
                    flush=True,
                )
            else:
                print(f"cap {r} {el}", flush=True)
    if got["variants"]:
        res["variants"] = got["variants"]
        res["variants_summary"] = summarise_variants(got["variants"], var_jobs[0][4])
        print_variants(res["variants_summary"])
    if got["streams"]:
        rows = got["streams"]
        res["streams"] = rows
        res["streams_summary"] = _summ(
            rows,
            ("mode", "par", "fam", "k"),
            (
                "gap_abs",
                "gap_se",
                "gap_se_glum",
                "glum_vs_ref_se",
                "gap_se_sm_unpen",
                "var_over_freq",
                "sand_over_freq",
                "freq_var_params_diff",
            ),
        )
        for row in res["streams_summary"]:
            print(
                f"{row['mode']:6s} {row['par']:5d} {row['fam']:7s} k={row['k']:2d} "
                f"n={row['n']:6d} abs max {row['gap_abs'][1]:.2e} SE mean/max "
                f"{row['gap_se'][0]:.3f}/{row['gap_se'][1]:.3f} glum max "
                f"{row['gap_se_glum'][1]:.3f} glum-ref {row['glum_vs_ref_se'][1]:.1e} "
                f"var/freq {row['var_over_freq'][0]:.3f} sand/freq {row['sand_over_freq'][0]:.3f}"
            )
    if got["cap"]:
        rows = got["cap"]
        res["cap"] = rows
        for f in ("poisson", "gamma"):
            for cap, hl in cap_cfgs:
                sel = [r for r in rows if r["fam"] == f and r["cap"] == cap and r["hl"] == hl]
                fail = sorted(r["seed"] for r in sel if not (r["gap_se"] <= 10))
                gs = [r["gap_se"] for r in sel if r["gap_se"] <= 10]
                print(
                    f"cap {f} k=20 cap={cap} hl={hl}: fail {len(fail)}/{len(sel)} seeds {fail}; "
                    f"ok-seed SE mean/max {np.mean(gs) if gs else float('nan'):.3g}/"
                    f"{max(gs) if gs else float('nan'):.3g}",
                    flush=True,
                )
    res["elapsed_s"] = time.perf_counter() - t0
    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
            )
    print(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
