"""Study 500, part 2: online GLMs (log link) under decay, a rolling window, a
step cap, separation, the log-link binomial and zero-weight rows.

Builds on scripts/study_500.py (part 1: the recursion, the data generator and
the penalised IRLS used by the start-up refit). The baseline is part 1's
variant C: ridge R = 1 and a refit of the first 200 rows.

The recursion, with decay lambda (lambda = 1: none) and a constant ridge R:

    G_t = lambda G_{t-1} + w_t W_t x_t x_t'      (G, h decay even on a zero-weight row:
    h_t = lambda h_{t-1} + w_t W_t z_t x_t        the clock advances, nothing is learnt)
    beta_t = (G_t + R I)^{-1} h_t

so its converged target is the batch fit with row weights w_i lambda^(T-i) and
the same R in sum form, i.e. alpha = R / sum_i w_i lambda^(T-i) in mean form.

Window: the last `window` rows are kept with the W_i, z_i they were scored at;
the row leaving the window has exactly its own contribution subtracted.

Step cap: the change in this row's eta, x_t' (beta_t - beta_{t-1}), is scaled
down to at most `cap` in absolute value; h is then reset to (G + R I) beta so
the next row's step is the increment form from the capped beta.

Data come from seeded numpy generators; nothing is read from disk.

Run:  .venv/bin/python scripts/study_500_part2.py all --out <dir>/part2.json
      (sections: checks, decay, window, cap, binomial, separation)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_500 import make_data  # noqa: E402

ETA_CLIP = 30.0
NS = (1000, 10000, 100000)


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
    if fam == "binlog":  # log-link binomial: dmu/deta = mu, V = mu (1 - mu)
        return mu / (1.0 - mu), eta + (y - mu) / mu
    if fam == "logit":
        v = mu * (1.0 - mu)
        return v, eta + (y - mu) / v
    raise ValueError(fam)


def halfdev(fam, eta, mu, y):
    if fam == "poisson":
        return mu - y * eta
    if fam == "gamma":
        return y / mu + eta
    if fam == "logit":
        return np.logaddexp(0.0, eta) - y * eta
    if fam == "binlog":  # NaN (rejected by the line search) where mu > 1
        with np.errstate(all="ignore"):
            return -(y * eta + (1.0 - y) * np.log1p(-mu))
    raise ValueError(fam)


# ---------------------------------------------------------------- online


class OnlineGLM2:
    def __init__(self, k, fam, ridge=1.0, refit_rows=200, lam=1.0, window=0, cap=None):
        self.k, self.fam, self.ridge = k, fam, ridge
        self.refit_rows, self.lam, self.window, self.cap = refit_rows, lam, window, cap
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
        W, z = irls_terms(self.fam, eta, mu, y)
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
        ts = np.array([b[0] for b in self._buf])
        Xb = np.array([b[1] for b in self._buf])
        yb = np.array([b[2] for b in self._buf])
        wb = np.array([b[3] for b in self._buf]) * self.lam ** (self.t - ts)
        beta = irls_ridge2(Xb, yb, wb, self.fam, self.ridge)
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


# ---------------------------------------------------------------- batch


def irls_ridge2(X, y, w, fam, ridge, tol=1e-13, maxiter=300):
    """Penalised IRLS, sum form: sum w_i halfdev_i + R/2 |beta|^2, step halving."""
    k = X.shape[1]
    beta = np.zeros(k)
    ym = max(float(np.average(y, weights=w)), 1e-12)
    beta[0] = np.log(ym) if fam != "logit" else np.log(ym / max(1 - ym, 1e-12))

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


def sm_family(fam):
    import statsmodels.api as sm

    L = sm.families.links
    return {
        "poisson": lambda: sm.families.Poisson(L.Log()),
        "gamma": lambda: sm.families.Gamma(L.Log()),
        "binlog": lambda: sm.families.Binomial(L.Log()),
        "logit": lambda: sm.families.Binomial(L.Logit()),
    }[fam]()


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


def glum_fit(X, y, fam, wts, ridge):
    from glum import GeneralizedLinearRegressor

    keep = wts > 0
    g = GeneralizedLinearRegressor(
        family={"logit": "binomial", "binlog": "binomial"}.get(fam, fam),
        link="log" if fam != "logit" else "logit",
        alpha=ridge / wts.sum(),
        l1_ratio=0.0,
        fit_intercept=False,
        gradient_tol=1e-10,
        max_iter=1000,
    )
    g.fit(X[keep], y[keep], sample_weight=wts[keep])
    return np.asarray(g.coef_)


# ---------------------------------------------------------------- 2. decay / 3. window


def run_stream(job):
    """One (family, k, seed) stream under decay or a window, compared at NS."""
    fam, k, seed, mode, par = job
    n = max(NS)
    X, y, _ = make_data(fam, k, n, seed)
    lam = 0.5 ** (1.0 / par) if mode == "decay" else 1.0
    o = OnlineGLM2(k, fam, 1.0, 200, lam=lam, window=par if mode == "window" else 0)
    snaps = {}
    t0 = time.perf_counter()
    for i in range(n):
        o.step(X[i], y[i])
        if i + 1 in NS:
            snaps[i + 1] = (o.beta.copy(), o.weight_sum)
    t_on = time.perf_counter() - t0
    out = {"fam": fam, "k": k, "seed": seed, "mode": mode, "par": par, "online_s": t_on, "at": {}}
    for m in NS:
        bo, ws_on = snaps[m]
        if mode == "decay":
            wts = lam ** np.arange(m - 1, -1, -1, dtype=float)
            lo = 0
        else:
            lo = max(0, m - par)
            wts = np.ones(m - lo)
        Xm, ym = X[lo:m], y[lo:m]
        b_ref = irls_ridge2(Xm, ym, wts, fam, 1.0)
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


# ---------------------------------------------------------------- 5. step cap


def run_cap(job):
    fam, k, seed, cap, lam_hl = job
    n = max(NS)
    X, y, _ = make_data(fam, k, n, seed)
    lam = 0.5 ** (1.0 / lam_hl) if lam_hl else 1.0
    o = OnlineGLM2(k, fam, 1.0, 0, lam=lam, cap=cap)
    for i in range(n):
        o.step(X[i], y[i])
    wts = lam ** np.arange(n - 1, -1, -1, dtype=float)
    b_ref = irls_ridge2(X, y, wts, fam, 1.0)
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


# ---------------------------------------------------------------- 1. binomial / separation


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
    o = OnlineGLM2(X.shape[1], fam, **kw)
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
    o = OnlineGLM2(X.shape[1], fam, **kw)
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


def binomial_section():
    out = {}
    # (a) log-link binomial, well inside (0, 1): truth mu in about (0.05, 0.6)
    rng = np.random.default_rng([500, 2, 1])
    n, k = 20000, 5
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
    b_ref = irls_ridge2(X, y, np.ones(len(y)), "binlog", 1.0)
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
    b_ref = irls_ridge2(X, y, np.ones(len(y)), "logit", 1.0)
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


SEP_FAMS = ("poisson", "logit")


def separation_section():
    """Poisson (log link, in scope): x1 is a 0/1 indicator and y = 0 on every row
    with x1 = 1, so the MLE's coefficient on x1 is -infinity. Logit (extra):
    y = 1 iff x1 > 0, complete separation, slope +infinity."""
    out = {}
    ns = (1000, 10000, 100000)
    for fam in SEP_FAMS:
        rng = np.random.default_rng([500, 3, 0 if fam == "poisson" else 1])
        n, k = max(ns), 5
        X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
        if fam == "poisson":  # x1 an indicator; every row with x1 = 1 has y = 0
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
            b_ref = irls_ridge2(Xm, ym, np.ones(m), fam, 1.0)
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


# ---------------------------------------------------------------- 4. zero weights


def checks_section():
    out = {}
    rng = np.random.default_rng([500, 4])
    for fam in ("poisson", "gamma"):
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
        rec = {"zero_count_share": float((y == 0).mean()), "zero_weight_rows": int(zero.sum())}
        for name, kw in (
            ("plain", dict(refit_rows=0)),
            ("refit200", dict(refit_rows=200)),
        ):
            a = OnlineGLM2(k, fam, **kw)
            b = OnlineGLM2(k, fam, **kw)
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
        b = OnlineGLM2(k, fam, refit_rows=200, lam=lam)
        for i in range(n):
            b.step(X[i], y_z[i], 0.0 if zero[i] else 1.0)
        wts = np.where(zero, 0.0, lam ** np.arange(n - 1, -1, -1, dtype=float))
        ref = irls_ridge2(X[~zero], y[~zero], wts[~zero], fam, 1.0)
        bse = sm_bse(X, y, fam, wts)["bse_freq"]
        rec["decay500_finite"] = bool(np.all(np.isfinite(b.beta)))
        rec["decay500_gap_se_vs_zero_weighted_batch"] = float((np.abs(b.beta - ref) / bse).max())
        # window: a zero-weight row occupies a slot; target = batch on the window's rows,
        # zero-weight ones at weight 0
        Wn = 1000
        b = OnlineGLM2(k, fam, refit_rows=200, window=Wn)
        for i in range(n):
            b.step(X[i], y_z[i], 0.0 if zero[i] else 1.0)
        sl = slice(n - Wn, n)
        keep = ~zero[sl]
        ref = irls_ridge2(X[sl][keep], y[sl][keep], np.ones(keep.sum()), fam, 1.0)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="+")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    secs = set(a.sections)
    if "all" in secs:
        secs = {"checks", "binomial", "separation", "decay", "window", "cap"}
    res = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            res = json.load(f)
    t0 = time.perf_counter()
    if "checks" in secs:
        res["checks"] = checks_section()
        print("checks", json.dumps(res["checks"], indent=1), flush=True)
    if "binomial" in secs:
        res["binomial"] = binomial_section()
        print("binomial", json.dumps(res["binomial"], indent=1), flush=True)
    if "separation" in secs:
        res["separation"] = separation_section()
        print("separation", json.dumps(res["separation"], indent=1), flush=True)
    jobs = []
    if "decay" in secs:
        jobs += [
            (f, k, s, "decay", hl)
            for hl in (500, 5000)
            for f in ("poisson", "gamma")
            for k in (5, 20)
            for s in range(a.seeds)
        ]
    if "window" in secs:
        jobs += [
            (f, k, s, "window", 2000)
            for f in ("poisson", "gamma")
            for k in (5, 20)
            for s in range(a.seeds)
        ]
    cap_jobs = []
    if "cap" in secs:
        cap_jobs = [
            (f, 20, s, cap, hl)
            for f in ("poisson", "gamma")
            for s in range(a.seeds)
            for cap, hl in ((None, 0), (1.0, 0), (2.0, 0), (None, 5000), (1.0, 5000))
        ]
    # spawn, and one future per job: a dead worker raises instead of hanging the run
    with ProcessPoolExecutor(a.procs, mp_context=mp.get_context("spawn")) as pool:
        if jobs:
            rows = []
            for fut in as_completed([pool.submit(run_stream, j) for j in jobs]):
                r = fut.result()
                rows.append(r)
                last = r["at"][max(NS)]
                print(
                    f"{r['mode']} {r['par']} {r['fam']} k={r['k']} s={r['seed']} "
                    f"{r['online_s']:.1f}s gap_se={last['gap_se']:.3g} "
                    f"glum={last['gap_se_glum']:.3g} [{time.perf_counter() - t0:.0f}s]",
                    flush=True,
                )
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
                    f"n={row['n']:6d} "
                    f"abs max {row['gap_abs'][1]:.2e} SE mean/max {row['gap_se'][0]:.3f}/"
                    f"{row['gap_se'][1]:.3f} glum max {row['gap_se_glum'][1]:.3f} "
                    f"glum-ref {row['glum_vs_ref_se'][1]:.1e} "
                    f"var/freq {row['var_over_freq'][0]:.3f} "
                    f"sand/freq {row['sand_over_freq'][0]:.3f}"
                )
        if cap_jobs:
            rows = []
            for fut in as_completed([pool.submit(run_cap, j) for j in cap_jobs]):
                rows.append(fut.result())
                print("cap job", rows[-1], flush=True)
            res["cap"] = rows
            for f in ("poisson", "gamma"):
                for cap, hl in ((None, 0), (1.0, 0), (2.0, 0), (None, 5000), (1.0, 5000)):
                    sel = [r for r in rows if r["fam"] == f and r["cap"] == cap and r["hl"] == hl]
                    fail = sorted(r["seed"] for r in sel if not (r["gap_se"] <= 10))
                    gs = [r["gap_se"] for r in sel if r["gap_se"] <= 10]
                    capped = [r["capped"] for r in sorted(sel, key=lambda r: r["seed"])]
                    print(
                        f"cap {f} k=20 cap={cap} hl={hl}: fail {len(fail)}/{len(sel)} "
                        f"seeds {fail}; ok-seed SE max {max(gs) if gs else float('nan'):.3g}; "
                        f"capped rows {capped}",
                        flush=True,
                    )
    res["elapsed_s"] = time.perf_counter() - t0
    with open(a.out, "w") as f:
        json.dump(
            res, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
        )
    print(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
