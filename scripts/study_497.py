"""Study 497: extended and unscented Kalman filters for GLM observations, in numpy.

The model is the random-walk-coefficient regression of the ``kalman`` model with
a non-Gaussian observation drawn from a GLM family (the record is
docs/records/STUDY-497-EKF-UKF.md)::

    beta_t = beta_{t-1} + u_t,          u_t ~ N(0, q I)       (per clock unit)
    y_t | beta_t ~ family(mu_t),        mu_t = g^{-1}(x_t' beta_t)

Families: ``gaussian`` (identity link, the linear Kalman filter), ``poisson``
(log), ``binomial`` (logit, y in {0, 1}) and ``gamma`` (log), with dispersion
``phi`` (1 for Poisson and binomial, a fixed 1/shape for Gamma, sigma^2 for
Gaussian). A row has a weight ``w``; its observation variance is
``phi V(mu) / w``. A row of weight 0, or a null target, observes nothing.

Per row, with ``D`` the clock since the covariance last took an observation
(the prior counts as observed at clock 0, so the first row has ``D = 1``)::

    P^- = P / lam^D + q D^2 I              (the forgetting factor, then the noise)
    pred = g^{-1}(x' b)        (EKF; the UKF's pred is its sigma-point mean)

and, only on a row that observes its target, the EKF (linearised at the prior
mean ``b``)::

    eta = x' b,  mu = g^{-1}(eta),  m = dmu/deta(eta),  R = phi V(mu) / w
    s = m^2 x' P^- x + R,   K = m P^- x / s
    b <- b + K (y - mu),    P <- P^- - s K K'       ("standard" form)
                            P <- (I - K H) P^- (I - K H)' + R K K',  H = m x'
                                                    ("joseph" form, filterpy's)

or the UKF, with sigma points chi_i = b, b +- [chol(c P^-)]_j (c = n + lambda)
and weights Wm, Wc (Merwe's scaled set or Julier's)::

    z_i = g^{-1}(x' chi_i),  zbar = sum Wm_i z_i
    R = phi V(g^{-1}(x' b)) / w       ("plugin", as the EKF)
     or phi sum Wm_i V(z_i) / w       ("expected": E[Var(y | beta)])
    s = sum Wc_i (z_i - zbar)^2 + R,  C = sum Wc_i (chi_i - b)(z_i - zbar)
    K = C / s,  b <- b + K (y - zbar),  P <- P^- - s K K'

then ``D <- 0``. A row that observes nothing only adds its clock to ``D``: the
noise is charged once per observation for the whole gap, and the forgetting
``lam^D`` is the same over one gap as over any cut of it. The weight sum
(``n_eff``) is emitted before the row's own decay and update, ``n <- lam^d n + w``.
Every prediction is emitted before the row's target is read.

numpy only for the filters. filterpy (oracle 1), statsmodels and scipy (oracle 2,
the static case) are imported only in the comparison functions. The grid filter
(k = 2) and the bootstrap particle filter are the second oracle for q > 0 and are
written from the definition, in numpy.

Data are generated from seeded numpy generators; nothing is read from disk.

Run (the study's venv holds numpy, scipy, filterpy and statsmodels; the record
says how it was made):

    .venv-497/bin/python scripts/study_497.py part1 --out <dir>
    .venv-497/bin/python scripts/study_497.py part2 --out <dir>
    ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- families

ETA_MAX_LOG = 30.0  # |eta| clamp for the log link (exp(30) = 1.07e13)
ETA_MAX_LOGIT = 36.0  # |eta| clamp for the logit link (1 - sigmoid(36) = 2.3e-16)


@dataclass(frozen=True)
class Family:
    name: str
    inv: Callable[[np.ndarray], np.ndarray]  # g^{-1}
    dmu: Callable[[np.ndarray], np.ndarray]  # d mu / d eta, as a function of eta
    var: Callable[[np.ndarray], np.ndarray]  # V(mu)
    eta_max: float  # the clamp's bound (inf: no clamp)


def _sigmoid(eta):
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    e = np.exp(eta[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _dsigmoid(eta):
    p = _sigmoid(eta)
    return p * (1.0 - p)


FAMILIES: dict[str, Family] = {
    "gaussian": Family(
        "gaussian",
        lambda e: np.asarray(e, float),
        lambda e: np.ones_like(e, float),
        lambda m: np.ones_like(m, float),
        math.inf,
    ),
    "poisson": Family("poisson", np.exp, np.exp, lambda m: m, ETA_MAX_LOG),
    "binomial": Family("binomial", _sigmoid, _dsigmoid, lambda m: m * (1.0 - m), ETA_MAX_LOGIT),
    "gamma": Family("gamma", np.exp, np.exp, lambda m: m * m, ETA_MAX_LOG),
}


# --------------------------------------------------------------------------- sigma points


def merwe_weights(n: int, alpha: float, beta: float, kappa: float):
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    wm = np.full(2 * n + 1, 0.5 / c)
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return c, wm, wc


def julier_weights(n: int, kappa: float):
    c = n + kappa
    wm = np.full(2 * n + 1, 0.5 / c)
    wm[0] = kappa / c
    return c, wm, wm.copy()


def sigma_points(b: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
    """(2n+1, n): b, b + L[:, j], b - L[:, j] with L L' = c P (lower Cholesky)."""
    n = b.size
    L = np.linalg.cholesky(c * P)
    chi = np.empty((2 * n + 1, n))
    chi[0] = b
    chi[1 : n + 1] = b + L.T
    chi[n + 1 :] = b - L.T
    return chi


# --------------------------------------------------------------------------- the filter


@dataclass
class KalmanGLM:
    """EKF or UKF with a GLM observation and a random-walk state (module doc)."""

    k: int
    family: str = "poisson"
    filt: str = "ekf"  # "ekf" or "ukf"
    q: float = 0.0
    lam: float = 1.0
    phi: float = 1.0
    p0: float = 1.0
    b0: np.ndarray | None = None
    form: str = "standard"  # EKF covariance update: "standard", "joseph" or "joseph_rank1"
    points: tuple = ("merwe", 1.0, 2.0, 0.0)  # ("merwe", alpha, beta, kappa) or ("julier", kappa)
    ukf_r: str = "expected"  # "plugin" or "expected"
    clamp: bool = True
    b: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    D: float = field(init=False, default=0.0)
    n: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.fam = FAMILIES[self.family]
        self.b = np.zeros(self.k) if self.b0 is None else np.array(self.b0, float)
        self.P = self.p0 * np.eye(self.k)
        self.I = np.eye(self.k)
        if self.filt == "ukf":
            if self.points[0] == "merwe":
                self.c, self.wm, self.wc = merwe_weights(self.k, *self.points[1:])
            else:
                self.c, self.wm, self.wc = julier_weights(self.k, *self.points[1:])

    # the state, for the size count: b (k), P (k x k, symmetric), D, n
    def state_floats(self, packed: bool = True) -> int:
        k = self.k
        return k + (k * (k + 1) // 2 if packed else k * k) + 2

    def _eta(self, eta):
        if self.clamp and math.isfinite(self.fam.eta_max):
            return np.clip(eta, -self.fam.eta_max, self.fam.eta_max)
        return eta

    def _dmu(self, eta_raw):
        # linearised at the clamped eta: mu, dmu/deta and V(mu) are all read at
        # clamp(eta), so a row beyond the bound still pulls eta back (the clamp's
        # own derivative, 0 there, would freeze the filter for good)
        return float(self.fam.dmu(self._eta(np.asarray(eta_raw))))

    def prior(self, D: float):
        P = self.P / self.lam**D if self.lam != 1.0 else self.P.copy()
        if self.q:
            P = P + (self.q * D * D) * self.I
        return P

    def step(self, x: np.ndarray, y: float, w: float, d: float = 1.0) -> dict:
        D = self.D + d
        Pp = self.prior(D)
        b = self.b
        fam = self.fam
        n_emit = self.n
        self.n = (self.lam**d) * self.n + w
        observes = w > 0 and y is not None and math.isfinite(y)
        if self.filt == "ekf":
            eta_raw = float(x @ b)
            mu = float(fam.inv(self._eta(np.asarray(eta_raw))))
            pred = mu
            if observes:
                m = self._dmu(eta_raw)
                R = self.phi * float(fam.var(np.asarray(mu))) / w
                Px = Pp @ x
                s = m * m * float(x @ Px) + R
                K = (m / s) * Px
                self.b = b + K * (y - mu)
                if self.form == "standard":
                    P = Pp - s * np.outer(K, K)
                elif self.form == "joseph_rank1":
                    # the Joseph form expanded for a scalar observation, O(k^2):
                    # (I - K h')P(I - K h')' + R K K' = P - K u' - u K' + s K K', u = P h
                    u = m * Px
                    KuT = np.outer(K, u)
                    P = Pp - KuT - KuT.T + s * np.outer(K, K)
                else:
                    IKH = self.I - np.outer(K, m * x)
                    P = IKH @ Pp @ IKH.T + R * np.outer(K, K)
                self.P = 0.5 * (P + P.T) if self.form != "joseph" else P
        else:
            chi = sigma_points(b, Pp, self.c)
            z = fam.inv(self._eta(chi @ x))
            zbar = float(self.wm @ z)
            pred = zbar
            if observes:
                if self.ukf_r == "plugin":
                    R = self.phi * float(fam.var(fam.inv(self._eta(np.asarray(float(x @ b)))))) / w
                else:
                    R = self.phi * float(self.wm @ fam.var(z)) / w
                dz = z - zbar
                s = float(self.wc @ (dz * dz)) + R
                C = (chi - b).T @ (self.wc * dz)
                K = C / s
                self.b = b + K * (y - zbar)
                P = Pp - s * np.outer(K, K)
                self.P = 0.5 * (P + P.T)
        if observes:
            self.D = 0.0
        else:
            self.D = D
        return {"pred": pred, "n_eff": n_emit, "observes": observes}


def run_filter(f: KalmanGLM, X, y, w):
    T = len(y)
    preds = np.empty(T)
    B = np.empty((T, f.k))
    Ps = np.empty((T, f.k, f.k))
    neff = np.empty(T)
    for t in range(T):
        r = f.step(X[t], y[t], w[t])
        preds[t] = r["pred"]
        neff[t] = r["n_eff"]
        B[t] = f.b
        Ps[t] = f.P
    return {"pred": preds, "b": B, "P": Ps, "n_eff": neff}


# --------------------------------------------------------------------------- data


def make_data(
    family: str,
    k: int,
    T: int,
    seed: int,
    q_true: float = 1e-4,
    phi: float = 1.0,
    zero_frac: float = 0.1,
    var_frac: float = 0.1,
    first_zero: bool = False,
    eta_scale: float = 1.0,
):
    """Intercept plus k - 1 N(0, 1) features; beta follows a random walk of
    variance q_true per row from a start whose non-intercept part has variance
    eta_scale^2 / (k - 1), so Var(x'beta) is about eta_scale^2 whatever k.
    Weights: 1, except zero_frac of rows at 0 and var_frac at U(0.5, 2)."""
    rng = np.random.default_rng(seed)
    X = np.empty((T, k))
    X[:, 0] = 1.0
    X[:, 1:] = rng.standard_normal((T, k - 1))
    beta = np.empty((T, k))
    b = np.empty(k)
    b[0] = {"poisson": 1.0, "binomial": 0.0, "gamma": 0.5, "gaussian": 0.5}[family]
    b[1:] = rng.standard_normal(k - 1) * eta_scale / math.sqrt(max(k - 1, 1))
    for t in range(T):
        b = b + math.sqrt(q_true) * rng.standard_normal(k)
        beta[t] = b
    eta = np.einsum("tk,tk->t", X, beta)
    if family == "gaussian":
        y = eta + math.sqrt(phi) * rng.standard_normal(T)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family == "binomial":
        y = (rng.random(T) < _sigmoid(eta)).astype(float)
    elif family == "gamma":
        shape = 1.0 / phi
        y = rng.gamma(shape, np.exp(eta) / shape)
    u = rng.random(T)
    w = np.ones(T)
    w[u < zero_frac] = 0.0
    sel = (u >= zero_frac) & (u < zero_frac + var_frac)
    w[sel] = rng.uniform(0.5, 2.0, sel.sum())
    if first_zero:
        w[0] = 0.0
    return X, y, w, beta


# --------------------------------------------------------------------------- comparison helpers


def ulps(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    sp = np.spacing(np.maximum(np.abs(a), np.abs(b)))
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.abs(a - b) / sp
    u[(a == b)] = 0.0
    return u


def gap(mine: dict, ref: dict) -> dict:
    """Largest differences, each relative to the reference's own size on that row:
    pred |a - r| / |r|; b max_j |a_j - r_j| / max_j |r_j|; P the same over P's
    entries. "rel_run" divides the largest |a - r| by the largest |r| of the run
    (a b vector passing near 0 makes the per-row "rel" large for b); "abs" is the
    plain largest |a - r|."""
    out = {}
    for key in ("pred", "b", "P"):
        if key not in ref:
            continue
        a, r = np.asarray(mine[key], float), np.asarray(ref[key], float)
        ok = np.isfinite(r)
        if key == "pred":
            d = np.abs(a - r)[ok]
            rel = d / np.maximum(np.abs(r[ok]), 1e-300)
        else:
            T = r.shape[0]
            d = np.abs(a - r).reshape(T, -1)
            scale = np.max(np.abs(r).reshape(T, -1), axis=1)
            rel = np.max(d, axis=1) / np.maximum(scale, 1e-300)
        out[key] = {
            "abs": float(np.max(d)) if d.size else 0.0,
            "rel": float(np.max(rel)) if rel.size else 0.0,
            # the largest difference over the largest reference value in the run
            "rel_run": float(np.max(d) / max(float(np.max(np.abs(r[ok]))), 1e-300))
            if d.size
            else 0.0,
        }
    return out


# --------------------------------------------------------------------------- oracle 1: filterpy


def _qeff(P, lam, q, D, k):
    """The process noise that turns filterpy's P + Q into P / lam^D + q D^2 I."""
    Q = (q * D * D) * np.eye(k)
    if lam != 1.0:
        Q = Q + (1.0 / lam**D - 1.0) * P
    return Q


def filterpy_kf(X, y, w, k, q, lam, phi, p0, b0=None):
    """Linear Gaussian: filterpy.kalman.KalmanFilter, with its own fading factor
    alpha = lam^(-D/2) for the forgetting and Q = q D^2 I."""
    from filterpy.kalman import KalmanFilter

    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.x = (np.zeros(k) if b0 is None else np.array(b0, float)).reshape(k, 1)
    kf.P = p0 * np.eye(k)
    kf.F = np.eye(k)
    T = len(y)
    out = {"pred": np.empty(T), "b": np.empty((T, k)), "P": np.empty((T, k, k))}
    D = 0.0
    for t in range(T):
        D += 1.0
        out["pred"][t] = float(X[t] @ kf.x[:, 0])
        if w[t] > 0:
            kf.alpha = lam ** (-D / 2.0)
            kf.Q = (q * D * D) * np.eye(k)
            kf.predict()
            kf.update(np.array([[y[t]]]), R=np.array([[phi / w[t]]]), H=X[t].reshape(1, k))
            D = 0.0
        out["b"][t] = kf.x[:, 0]
        out["P"][t] = kf.P
    return out


def filterpy_ekf(X, y, w, family, k, q, lam, phi, p0, b0=None, clamp=True):
    """filterpy.kalman.ExtendedKalmanFilter: F = I, Q = (lam^-D - 1) P + q D^2 I,
    Hx(b) = g^{-1}(x'b), HJacobian(b) = (dmu/deta)(x'b) x', R = phi V(Hx(b)) / w,
    all evaluated at the prior mean (the filter's x when update is called)."""
    from filterpy.kalman import ExtendedKalmanFilter

    fam = FAMILIES[family]
    em = fam.eta_max if clamp else math.inf

    def ce(e):
        return min(max(e, -em), em)

    ekf = ExtendedKalmanFilter(dim_x=k, dim_z=1)
    ekf.x = (np.zeros(k) if b0 is None else np.array(b0, float)).reshape(k, 1)
    ekf.P = p0 * np.eye(k)
    ekf.F = np.eye(k)
    T = len(y)
    out = {"pred": np.empty(T), "b": np.empty((T, k)), "P": np.empty((T, k, k))}
    D = 0.0
    for t in range(T):
        D += 1.0
        xt = X[t]
        out["pred"][t] = float(fam.inv(np.asarray(ce(float(xt @ ekf.x[:, 0])))))
        if w[t] > 0:
            ekf.Q = _qeff(ekf.P, lam, q, D, k)
            ekf.predict()

            def Hx(b, xt=xt):
                return np.array([[float(fam.inv(np.asarray(ce(float(xt @ b[:, 0])))))]])

            def HJ(b, xt=xt):
                m = float(fam.dmu(np.asarray(ce(float(xt @ b[:, 0])))))
                return (m * xt).reshape(1, k)

            mu = Hx(ekf.x)[0, 0]
            R = np.array([[phi * float(fam.var(np.asarray(mu))) / w[t]]])
            ekf.update(np.array([[y[t]]]), HJ, Hx, R=R)
            D = 0.0
        out["b"][t] = ekf.x[:, 0]
        out["P"][t] = ekf.P
    return out


def filterpy_ukf(
    X,
    y,
    w,
    family,
    k,
    q,
    lam,
    phi,
    p0,
    b0=None,
    points=("merwe", 1.0, 2.0, 0.0),
    ukf_r="expected",
    clamp=True,
    redraw=False,
):
    """filterpy.kalman.UnscentedKalmanFilter: fx(b) = b, Q as filterpy_ekf,
    hx(b) = g^{-1}(x'b), R = phi V(.) / w at the plug-in prior mean or averaged over
    filterpy's own sigma points (sigmas_f, drawn after predict from the predicted P)
    with Wm. The sigma points: MerweScaledSigmaPoints(alpha, beta, kappa) or
    JulierSigmaPoints(kappa), each with scipy's Cholesky (filterpy's default).
    redraw=True replaces sigmas_f by points drawn from the prior (x, P) that
    the numpy filter uses (the exact prior, not filterpy's UT of it)."""
    from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints, UnscentedKalmanFilter

    fam = FAMILIES[family]
    em = fam.eta_max if clamp else math.inf
    if points[0] == "merwe":
        pts = MerweScaledSigmaPoints(k, alpha=points[1], beta=points[2], kappa=points[3])
    else:
        pts = JulierSigmaPoints(k, kappa=points[1])

    def fx(b, dt):
        return b

    holder = {"x": None}

    def hx(b):
        e = float(holder["x"] @ b)
        return np.array([float(fam.inv(np.asarray(min(max(e, -em), em))))])

    ukf = UnscentedKalmanFilter(dim_x=k, dim_z=1, dt=1.0, hx=hx, fx=fx, points=pts)
    ukf.x = np.zeros(k) if b0 is None else np.array(b0, float)
    ukf.P = p0 * np.eye(k)
    T = len(y)
    out = {"pred": np.full(T, np.nan), "b": np.empty((T, k)), "P": np.empty((T, k, k))}
    D = 0.0
    for t in range(T):
        D += 1.0
        xt = X[t]
        holder["x"] = xt
        if w[t] > 0:
            P_post = ukf.P.copy()
            x_post = ukf.x.copy()
            ukf.Q = _qeff(ukf.P, lam, q, D, k)
            ukf.predict()
            if redraw:
                Pex = P_post / lam**D if lam != 1.0 else P_post.copy()
                Pex = Pex + (q * D * D) * np.eye(k)
                Pex = 0.5 * (Pex + Pex.T)  # the numpy filter keeps P symmetric
                ukf.x = x_post
                ukf.P = Pex
                ukf.sigmas_f = pts.sigma_points(x_post, Pex)
            zs = np.array([hx(s)[0] for s in ukf.sigmas_f])
            out["pred"][t] = float(pts.Wm @ zs)
            if ukf_r == "plugin":
                R = phi * float(fam.var(np.asarray(hx(ukf.x)[0]))) / w[t]
            else:
                R = phi * float(pts.Wm @ fam.var(zs)) / w[t]
            ukf.update(np.array([y[t]]), R=np.array([[R]]))
            D = 0.0
        out["b"][t] = ukf.x
        out["P"][t] = ukf.P
    return out


# --------------------------------------------------------------------------- oracle 2


def loglik(family: str, y: float, eta, phi: float):
    """The row's log-likelihood in eta, up to terms free of eta (exponential-family form)."""
    if family == "gaussian":
        return -0.5 * (y - eta) ** 2 / phi
    if family == "poisson":
        return y * eta - np.exp(eta)
    if family == "binomial":
        return y * eta - np.logaddexp(0.0, eta)
    if family == "gamma":
        return (-y * np.exp(-eta) - eta) / phi
    raise ValueError(family)


def grid_filter(X, y, w, family, q, lam, phi, p0, lo, hi, n):
    """The exact filter for k = 2 on an n x n grid over [lo_j, hi_j], written from
    the definition: the forgetting as power discounting (density^(lam^D), which is
    P / lam^D for a Gaussian), the random walk as a convolution with N(0, q D^2 I)
    applied in Fourier space (exact for the grid's trigonometric interpolant), and
    Bayes' rule with the row's likelihood raised to its weight. Emits, per row, the
    posterior mean and covariance of beta and the predictive mean E[mu] under the
    prior of the row (before its target is read)."""
    g0 = np.linspace(lo[0], hi[0], n)
    g1 = np.linspace(lo[1], hi[1], n)
    h0, h1 = g0[1] - g0[0], g1[1] - g1[0]
    B0, B1 = np.meshgrid(g0, g1, indexing="ij")
    om0 = 2 * np.pi * np.fft.fftfreq(n, d=h0)
    om1 = 2 * np.pi * np.fft.rfftfreq(n, d=h1)
    O2 = om0[:, None] ** 2 + om1[None, :] ** 2
    logd = -0.5 * (B0**2 + B1**2) / p0  # prior N(0, p0 I)
    fam = FAMILIES[family]
    T = len(y)
    out = {"pred": np.empty(T), "b": np.empty((T, 2)), "P": np.empty((T, 2, 2))}
    D = 0.0

    def prior_density(D):
        ld = logd * (lam**D) if lam != 1.0 else logd
        dens = np.exp(ld - ld.max())
        if q:
            dens = np.fft.irfft2(np.fft.rfft2(dens) * np.exp(-0.5 * q * D * D * O2), s=(n, n))
            dens = np.maximum(dens, 0.0)
        return dens / dens.sum()

    def moments(dens):
        m0, m1 = (dens * B0).sum(), (dens * B1).sum()
        c00 = (dens * (B0 - m0) ** 2).sum()
        c11 = (dens * (B1 - m1) ** 2).sum()
        c01 = (dens * (B0 - m0) * (B1 - m1)).sum()
        return np.array([m0, m1]), np.array([[c00, c01], [c01, c11]])

    cur_b, cur_P = moments(np.exp(logd - logd.max()) / np.exp(logd - logd.max()).sum())
    for t in range(T):
        D += 1.0
        dens = prior_density(D)
        eta = X[t, 0] * B0 + X[t, 1] * B1
        out["pred"][t] = float((dens * fam.inv(eta)).sum())
        if w[t] > 0:
            ll = np.log(np.maximum(dens, 1e-320)) + w[t] * loglik(family, y[t], eta, phi)
            logd = ll - ll.max()
            post = np.exp(logd)
            post /= post.sum()
            cur_b, cur_P = moments(post)
            D = 0.0
        out["b"][t] = cur_b
        out["P"][t] = cur_P
    return out


def particle_filter(X, y, w, family, q, phi, p0, N, seed):
    """Bootstrap particle filter (random walk only, lam = 1), written from the
    definition: particles from the prior N(0, p0 I); on each observing row the
    particles move by N(0, q D^2 I), are weighted by the row's likelihood to the
    power of its weight, and are resampled (systematic) when the effective sample
    size falls under N / 2. Emits the weighted posterior mean and the predictive
    mean of mu under the moved particles."""
    rng = np.random.default_rng(seed)
    T, k = X.shape
    fam = FAMILIES[family]
    parts = rng.standard_normal((N, k)) * math.sqrt(p0)
    logw = np.zeros(N)
    out = {"pred": np.empty(T), "b": np.empty((T, k)), "var": np.empty((T, k)), "ess_min": float(N)}
    D = 0.0
    wts = np.full(N, 1.0 / N)
    for t in range(T):
        D += 1.0
        # the predictive mean at this row: the particles moved by the gap so far,
        # drawn afresh for the prediction only when the row does not observe
        moved = parts + rng.standard_normal((N, k)) * (math.sqrt(q) * D) if q else parts
        eta = moved @ X[t]
        out["pred"][t] = float(wts @ fam.inv(np.clip(eta, -700, 700)))
        if w[t] > 0:
            parts = moved
            logw = np.log(np.maximum(wts, 1e-320)) + w[t] * loglik(family, y[t], eta, phi)
            logw -= logw.max()
            wts = np.exp(logw)
            wts /= wts.sum()
            ess = 1.0 / float(wts @ wts)
            out["ess_min"] = min(out["ess_min"], ess)
            if ess < N / 2:
                u = (rng.random() + np.arange(N)) / N
                idx = np.minimum(np.searchsorted(np.cumsum(wts), u), N - 1)
                parts = parts[idx]
                wts = np.full(N, 1.0 / N)
            D = 0.0
        m = wts @ parts
        out["b"][t] = m
        out["var"][t] = wts @ ((parts - m) ** 2)
    return out


# --------------------------------------------------------------------------- part 1


U = 2.0**-52  # the unit roundoff's double: a 1-ulp change of a value near 1


def perturbed(f: KalmanGLM) -> KalmanGLM:
    """The same filter with its prior covariance moved by one ulp: the gap between
    the two runs is the stream's own amplification of a rounding difference, the
    band a second implementation's gap is read against."""
    f.P = f.P * (1.0 + U)
    return f


def p0_for(k: int, kind: str) -> float:
    """'calibrated': x'P0x about 0.25 for N(0, 1) features (an eta prior sd of
    0.5); 'wide': P0 = I, x'P0x about k."""
    return 0.25 / k if kind == "calibrated" else 1.0


def part1(args) -> dict:
    res: dict = {"gaussian": [], "poisson_ekf": []}
    T = args.rows
    for k in (2, 5):
        for q, lam in ((1e-4, 1.0), (0.0, 0.99), (1e-4, 0.995)):
            for first_zero in (False, True):
                seed = 1000 + 10 * k + int(first_zero)
                # Gaussian: the numpy EKF (identity link) against filterpy's linear
                # KalmanFilter (fading alpha) and its ExtendedKalmanFilter
                X, y, w, _ = make_data(
                    "gaussian", k, T, seed, q_true=1e-4, phi=0.5, first_zero=first_zero
                )
                p0 = 1.0
                mine = run_filter(
                    KalmanGLM(k, family="gaussian", q=q, lam=lam, phi=0.5, p0=p0, form="joseph"),
                    X,
                    y,
                    w,
                )
                band = run_filter(
                    perturbed(
                        KalmanGLM(k, family="gaussian", q=q, lam=lam, phi=0.5, p0=p0, form="joseph")
                    ),
                    X,
                    y,
                    w,
                )
                kf = filterpy_kf(X, y, w, k, q, lam, 0.5, p0)
                ek = filterpy_ekf(X, y, w, "gaussian", k, q, lam, 0.5, p0)
                fs = run_filter(
                    KalmanGLM(k, family="gaussian", q=q, lam=lam, phi=0.5, p0=p0, form="standard"),
                    X,
                    y,
                    w,
                )
                res["gaussian"].append(
                    {
                        "k": k,
                        "q": q,
                        "lam": lam,
                        "first_zero": first_zero,
                        "p0": p0,
                        "vs_KalmanFilter": gap(mine, kf),
                        "vs_EKF": gap(mine, ek),
                        "standard_vs_KalmanFilter": gap(fs, kf),
                        "band": gap(band, mine),
                    }
                )
                # Poisson EKF, at a calibrated and a wide prior
                X, y, w, _ = make_data("poisson", k, T, seed, q_true=1e-4, first_zero=first_zero)
                for kind in ("calibrated", "wide"):
                    p0 = p0_for(k, kind)

                    def mk(k=k, q=q, lam=lam, p0=p0):
                        return KalmanGLM(k, family="poisson", q=q, lam=lam, p0=p0, form="joseph")

                    mine = run_filter(mk(), X, y, w)
                    band = run_filter(perturbed(mk()), X, y, w)
                    std = run_filter(
                        KalmanGLM(k, family="poisson", q=q, lam=lam, p0=p0, form="standard"),
                        X,
                        y,
                        w,
                    )
                    ref = filterpy_ekf(X, y, w, "poisson", k, q, lam, 1.0, p0)
                    res["poisson_ekf"].append(
                        {
                            "k": k,
                            "q": q,
                            "lam": lam,
                            "first_zero": first_zero,
                            "prior": kind,
                            "p0": p0,
                            "joseph_vs_filterpy": gap(mine, ref),
                            "standard_vs_filterpy": gap(std, ref),
                            "band": gap(band, mine),
                            "max_pred": float(np.max(mine["pred"])),
                            "clamped_rows": int(
                                np.sum(
                                    np.abs(np.log(np.maximum(mine["pred"], 1e-300)))
                                    >= ETA_MAX_LOG - 1e-9
                                )
                            ),
                        }
                    )
    # out of sample: changing row t's target and weight changes no prediction
    # up to and including row t
    X, y, w, _ = make_data("poisson", 5, 300, 7, first_zero=True)
    base = run_filter(KalmanGLM(5, family="poisson", q=1e-4, lam=0.995), X, y, w)
    worst = 0.0
    for t in (0, 1, 50, 299):
        y2, w2 = y.copy(), w.copy()
        y2[t] = y[t] + 17.0
        w2[t] = 3.0
        alt = run_filter(KalmanGLM(5, family="poisson", q=1e-4, lam=0.995), X, y2, w2)
        worst = max(worst, float(np.max(np.abs(alt["pred"][: t + 1] - base["pred"][: t + 1]))))
    res["out_of_sample"] = {"max_pred_change_up_to_t": worst}
    # a zero-weight row moves nothing but the clock; a run of them is one gap
    X, y, w, _ = make_data("poisson", 2, 200, 8, zero_frac=0.3, first_zero=True)
    f1 = KalmanGLM(2, family="poisson", q=1e-3, lam=0.99)
    r1 = run_filter(f1, X, y, w)
    keep = w > 0
    f2 = KalmanGLM(2, family="poisson", q=1e-3, lam=0.99)
    # the same stream with the zero-weight rows removed and their clock folded
    # into the next observed row's d
    d = 0.0
    bs = []
    for t in range(len(y)):
        d += 1.0
        if keep[t]:
            f2.step(X[t], y[t], w[t], d=d)
            d = 0.0
            bs.append(f2.b.copy())
    res["zero_weight_rows_are_a_gap"] = {
        "max_abs_b": float(np.max(np.abs(r1["b"][keep] - np.array(bs)))),
        "first_row_zero_b_unchanged": bool(np.all(r1["b"][0] == 0.0)),
    }
    return res


# --------------------------------------------------------------------------- part 2

POINTS = {
    "merwe(1,2,0)": ("merwe", 1.0, 2.0, 0.0),
    "merwe(0.5,2,0)": ("merwe", 0.5, 2.0, 0.0),
    "julier(1)": ("julier", 1.0),
}
PHI = {"gaussian": 0.5, "poisson": 1.0, "binomial": 1.0, "gamma": 0.5}


def part2(args) -> dict:
    """The UKF for every family and the EKF for binomial and Gamma, against filterpy."""
    res: dict = {"ekf": [], "ukf": [], "ukf_vs_kf": []}
    T = args.rows
    for family in ("gaussian", "poisson", "binomial", "gamma"):
        phi = PHI[family]
        for k in (2, 5):
            for q, lam in ((1e-4, 1.0), (0.0, 0.99), (1e-4, 0.995)):
                for first_zero in (False, True):
                    seed = 2000 + 10 * k + int(first_zero) + 100 * list(PHI).index(family)
                    X, y, w, _ = make_data(
                        family, k, T, seed, q_true=1e-4, phi=phi, first_zero=first_zero
                    )
                    p0 = p0_for(k, "calibrated")
                    head = {"family": family, "k": k, "q": q, "lam": lam, "first_zero": first_zero}
                    if family in ("binomial", "gamma"):

                        def mk(form, k=k, q=q, lam=lam, p0=p0, phi=phi, family=family):
                            return KalmanGLM(
                                k, family=family, q=q, lam=lam, phi=phi, p0=p0, form=form
                            )

                        mine = run_filter(mk("joseph"), X, y, w)
                        std = run_filter(mk("standard"), X, y, w)
                        band = run_filter(perturbed(mk("joseph")), X, y, w)
                        ref = filterpy_ekf(X, y, w, family, k, q, lam, phi, p0)
                        res["ekf"].append(
                            head
                            | {
                                "joseph_vs_filterpy": gap(mine, ref),
                                "standard_vs_filterpy": gap(std, ref),
                                "band": gap(band, mine),
                                "clamped": _clamped(mine["pred"], family),
                            }
                        )
                    for pname, pts in POINTS.items():
                        r_forms = ("expected", "plugin") if family != "gaussian" else ("plugin",)
                        if pname != "merwe(1,2,0)" and family in ("gaussian", "gamma"):
                            continue
                        for ukf_r in r_forms:

                            def mk(
                                k=k,
                                q=q,
                                lam=lam,
                                p0=p0,
                                phi=phi,
                                family=family,
                                pts=pts,
                                ukf_r=ukf_r,
                            ):
                                return KalmanGLM(
                                    k,
                                    family=family,
                                    filt="ukf",
                                    q=q,
                                    lam=lam,
                                    phi=phi,
                                    p0=p0,
                                    points=pts,
                                    ukf_r=ukf_r,
                                )

                            mine = run_filter(mk(), X, y, w)
                            band = run_filter(perturbed(mk()), X, y, w)
                            ref = filterpy_ukf(
                                X, y, w, family, k, q, lam, phi, p0, points=pts, ukf_r=ukf_r
                            )
                            ref_rd = filterpy_ukf(
                                X,
                                y,
                                w,
                                family,
                                k,
                                q,
                                lam,
                                phi,
                                p0,
                                points=pts,
                                ukf_r=ukf_r,
                                redraw=True,
                            )
                            res["ukf"].append(
                                head
                                | {
                                    "points": pname,
                                    "R": ukf_r,
                                    "vs_filterpy": gap(mine, ref),
                                    "vs_filterpy_exact_prior": gap(mine, ref_rd),
                                    "band": gap(band, mine),
                                    "clamped": _clamped(mine["pred"], family),
                                }
                            )
                            if family == "gaussian":
                                kf = filterpy_kf(X, y, w, k, q, lam, phi, p0)
                                res["ukf_vs_kf"].append(head | {"gap": gap(mine, kf)})
    return res


def _clamped(pred, family) -> int:
    pred = np.asarray(pred)
    if family in ("poisson", "gamma"):
        return int(np.sum(np.abs(np.log(np.maximum(pred, 1e-300))) >= ETA_MAX_LOG - 1e-9))
    if family == "binomial":
        return int(np.sum((pred <= 1e-15) | (pred >= 1 - 1e-15)))
    return 0


def summarise2(res):
    print("part 2 (largest relative gap over cases; 'band' is the 1-ulp self-perturbation)")
    for fam in ("binomial", "gamma"):
        for c in (0, 1):

            def sel(r, fam=fam, c=c):
                return r["family"] == fam and (r["clamped"] > 0) == bool(c)

            n = sum(1 for r in res["ekf"] if sel(r))
            if not n:
                continue
            tag = f"EKF {fam} [{'clamped' if c else 'unclamped'}, {n} cases]"
            print(
                f"  {tag} joseph vs filterpy:  ",
                _worst(res["ekf"], "joseph_vs_filterpy", where=sel),
            )
            print(
                f"  {tag} standard vs filterpy:",
                _worst(res["ekf"], "standard_vs_filterpy", where=sel),
            )
            print(f"  {tag} band:                ", _worst(res["ekf"], "band", where=sel))
    print(
        "  UKF gaussian vs KalmanFilter (pred abs):",
        _worst(res["ukf_vs_kf"], "gap"),
        _worst(res["ukf_vs_kf"], "gap", "abs"),
    )
    for fam in ("gaussian", "poisson", "binomial", "gamma"):
        for pname in POINTS:
            for rf in ("expected", "plugin"):
                for c in (0, 1):

                    def sel(r, fam=fam, pname=pname, rf=rf, c=c):
                        return (
                            r["family"] == fam
                            and r["points"] == pname
                            and r["R"] == rf
                            and (r["clamped"] > 0) == bool(c)
                        )

                    n = sum(1 for r in res["ukf"] if sel(r))
                    if not n:
                        continue
                    tag = f"UKF {fam} {pname} R={rf} [{'clamped' if c else 'ok'} {n}]"
                    print(
                        f"  {tag}: vs filterpy",
                        _worst(res["ukf"], "vs_filterpy", where=sel),
                        " exact prior",
                        _worst(res["ukf"], "vs_filterpy_exact_prior", where=sel),
                        " band",
                        _worst(res["ukf"], "band", where=sel),
                    )


# --------------------------------------------------------------------------- part 3


def _sd_units(filt: dict, exact: dict, rows) -> dict:
    """The filter's b against the exact posterior mean in the exact posterior's own
    sd, its sd over the exact sd, and its pred against the exact predictive mean."""
    sd = np.sqrt(np.einsum("tii->ti", exact["P"][rows]))
    z = np.abs(filt["b"][rows] - exact["b"][rows]) / sd
    ratio = np.sqrt(np.einsum("tii->ti", filt["P"][rows])) / sd
    pr = np.abs(filt["pred"][rows] - exact["pred"][rows]) / np.abs(exact["pred"][rows])
    return {
        "z_max": float(z.max()),
        "z_median": float(np.median(z)),
        "z_last": float(z[-1].max()),
        "sd_ratio_min": float(ratio.min()),
        "sd_ratio_max": float(ratio.max()),
        "sd_ratio_median": float(np.median(ratio)),
        "pred_rel_median": float(np.median(pr)),
        "pred_rel_max": float(pr.max()),
        "pred_abs_max": float(np.max(np.abs(filt["pred"][rows] - exact["pred"][rows]))),
    }


def _grid_case(family, q, lam, T, seed, n_max):
    phi = PHI[family]
    k = 2
    p0 = p0_for(k, "calibrated")
    X, y, w, beta = make_data(family, k, T, seed, q_true=max(q, 1e-5), phi=phi, eta_scale=0.5)
    ekf = run_filter(KalmanGLM(k, family=family, q=q, lam=lam, phi=phi, p0=p0), X, y, w)
    ukf = run_filter(KalmanGLM(k, family=family, filt="ukf", q=q, lam=lam, phi=phi, p0=p0), X, y, w)
    # the grid: wide enough for the prior, the truth and both filters (8 sd), fine
    # enough for the narrowest posterior (UKF's sd / 6), at most n_max a side
    lo, hi, sdmin = [], [], math.inf
    for j in range(k):
        cols = [beta[:, j], ekf["b"][:, j], ukf["b"][:, j]]
        sdj = np.sqrt(ukf["P"][:, j, j])
        sdmin = min(sdmin, float(sdj.min()))
        lo.append(min(float(np.min(c)) for c in cols) - 8 * float(sdj.max()) - 0.5)
        hi.append(max(float(np.max(c)) for c in cols) + 8 * float(sdj.max()) + 0.5)
    lo[0] = min(lo[0], -6 * math.sqrt(p0))
    lo[1] = min(lo[1], -6 * math.sqrt(p0))
    hi[0] = max(hi[0], 6 * math.sqrt(p0))
    hi[1] = max(hi[1], 6 * math.sqrt(p0))
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    n = int(min(n_max, math.ceil(6 * span / sdmin)))
    t0 = time.perf_counter()
    grid = grid_filter(X, y, w, family, q, lam, phi, p0, lo, hi, n)
    secs = time.perf_counter() - t0
    coarse = grid_filter(X, y, w, family, q, lam, phi, p0, lo, hi, int(0.75 * n))
    rows = np.arange(T)
    out = {
        "family": family,
        "q": q,
        "lam": lam,
        "n": n,
        "spacing_over_min_sd": span / n / sdmin,
        "grid_seconds": secs,
        "grid_band": _sd_units(coarse, grid, rows),
        "ekf": _sd_units(ekf, grid, rows),
        "ukf": _sd_units(ukf, grid, rows),
    }
    if family == "gaussian":
        kf = filterpy_kf(X, y, w, k, q, lam, phi, p0)
        out["grid_vs_KalmanFilter"] = _sd_units(grid, kf, rows)
    return out


def _pf_case(family, T, seed, N, reps):
    phi = PHI[family]
    k, q, lam = 5, 1e-4, 1.0
    p0 = p0_for(k, "calibrated")
    X, y, w, _ = make_data(family, k, T, seed, q_true=q, phi=phi)
    ekf = run_filter(KalmanGLM(k, family=family, q=q, lam=lam, phi=phi, p0=p0), X, y, w)
    ukf = run_filter(KalmanGLM(k, family=family, filt="ukf", q=q, lam=lam, phi=phi, p0=p0), X, y, w)
    t0 = time.perf_counter()
    runs = [particle_filter(X, y, w, family, q, phi, p0, N, seed=10_000 + r) for r in range(reps)]
    secs = time.perf_counter() - t0
    mean_b = np.mean([r["b"] for r in runs], axis=0)
    mean_pred = np.mean([r["pred"] for r in runs], axis=0)
    var_b = np.mean([r["var"] for r in runs], axis=0)
    mc_sd = np.std([r["b"] for r in runs], axis=0, ddof=1) / math.sqrt(reps)
    exact = {"b": mean_b, "P": np.array([np.diag(v) for v in var_b]), "pred": mean_pred}
    rows = np.arange(min(50, T - 1), T)  # after the first 50 rows, where the PF settles
    sd = np.sqrt(var_b[rows])
    mc_sd = mc_sd[rows]
    return {
        "family": family,
        "N": N,
        "reps": reps,
        "pf_seconds": secs,
        "ess_min": min(r["ess_min"] for r in runs),
        "pf_mc_sd_over_post_sd_max": float(np.max(mc_sd / sd)),
        "pf_mc_sd_over_post_sd_median": float(np.median(mc_sd / sd)),
        "ekf": _sd_units(ekf, exact, rows),
        "ukf": _sd_units(ukf, exact, rows),
        # the gap beyond the PF's own band, 4 Monte Carlo sd of its mean, in
        # posterior sd; and the gap on the rows where the PF is sharp (mc sd < 0.05 sd)
        **{
            f"{name}_beyond_4mc_max": float(
                np.max(np.maximum(np.abs(f["b"][rows] - mean_b[rows]) - 4 * mc_sd, 0.0) / sd)
            )
            for name, f in (("ekf", ekf), ("ukf", ukf))
        },
        **{
            f"{name}_z_max_where_sharp": float(
                np.max((np.abs(f["b"][rows] - mean_b[rows]) / sd)[mc_sd / sd < 0.05])
            )
            for name, f in (("ekf", ekf), ("ukf", ukf))
        },
        "sharp_fraction": float(np.mean(mc_sd / sd < 0.05)),
    }


def static_map(X, y, w, family, phi, p0):
    """The posterior mode with q = 0, lam = 1: maximise statsmodels' GLM
    log-likelihood (var_weights = w, scale = phi) minus |b|^2 / (2 p0) by Newton's
    method on statsmodels' own score and observed Hessian. Returns the mode and the
    Laplace covariance (the inverse of the negative Hessian there)."""
    import statsmodels.api as sm

    fams = {
        "poisson": sm.families.Poisson(),
        "binomial": sm.families.Binomial(),
        "gamma": sm.families.Gamma(sm.families.links.Log()),
        "gaussian": sm.families.Gaussian(),
    }
    keep = w > 0
    model = sm.GLM(y[keep], X[keep], family=fams[family], var_weights=w[keep])
    k = X.shape[1]
    b = np.zeros(k)
    for _ in range(100):
        g = model.score(b, scale=phi) - b / p0
        H = model.hessian(b, scale=phi, observed=True) - np.eye(k) / p0
        step = np.linalg.solve(H, g)
        b = b - step
        if np.max(np.abs(step)) < 1e-14 * max(1.0, np.max(np.abs(b))):
            break
    H = model.hessian(b, scale=phi, observed=True) - np.eye(k) / p0
    g = model.score(b, scale=phi) - b / p0
    return b, np.linalg.inv(-H), float(np.max(np.abs(g)))


def _static_case(family, k, T, seed, checkpoints):
    phi = PHI[family]
    p0 = p0_for(k, "calibrated")
    X, y, w, _ = make_data(family, k, T, seed, q_true=0.0, phi=phi)
    out = {"family": family, "k": k, "rows": {}}
    filt = {
        "ekf": run_filter(KalmanGLM(k, family=family, phi=phi, p0=p0), X, y, w),
        "ukf": run_filter(KalmanGLM(k, family=family, filt="ukf", phi=phi, p0=p0), X, y, w),
    }
    for t in checkpoints:
        b, C, grad = static_map(X[:t], y[:t], w[:t], family, phi, p0)
        sd = np.sqrt(np.diag(C))
        row = {"map_grad_max": grad}
        for name, f in filt.items():
            row[name] = {
                "z_max": float(np.max(np.abs(f["b"][t - 1] - b) / sd)),
                "sd_ratio_min": float(np.min(np.sqrt(np.diag(f["P"][t - 1])) / sd)),
                "sd_ratio_max": float(np.max(np.sqrt(np.diag(f["P"][t - 1])) / sd)),
            }
        out["rows"][str(t)] = row
    return out


def _zero_lead(family, filt, k, lead=25, T=400, seed=31):
    """A run of zero-weight rows at the start: b and P stay at the prior to the
    bit, n_eff stays 0, nothing is NaN, and the run equals the stream with those
    rows removed and their clock folded into the first observed row."""
    phi = PHI[family]
    p0 = p0_for(k, "calibrated")
    X, y, w, _ = make_data(family, k, T, seed, phi=phi)
    w[:lead] = 0.0
    y[3] = np.nan  # a null target inside the run as well

    def mk():
        return KalmanGLM(k, family=family, filt=filt, q=1e-3, lam=0.99, phi=phi, p0=p0)

    f = mk()
    res = run_filter(f, X, y, w)
    g = mk()
    d = 0.0
    bs = []
    for t in range(T):
        d += 1.0
        if w[t] > 0:
            g.step(X[t], y[t], w[t], d=d)
            d = 0.0
            bs.append(g.b.copy())
    keep = w > 0
    return {
        "family": family,
        "filt": filt,
        "k": k,
        "b_at_prior": bool(np.all(res["b"][:lead] == 0.0)),
        "P_at_prior": bool(np.all(res["P"][:lead] == p0 * np.eye(k))),
        "n_eff_zero": bool(np.all(res["n_eff"][: lead + 1] == 0.0)),
        "all_finite": bool(np.all(np.isfinite(res["b"])) and np.all(np.isfinite(res["pred"]))),
        "fold_max_abs_b": float(np.max(np.abs(res["b"][keep] - np.array(bs)))),
    }


def _sweep20(family, filt, T, seed):
    phi = PHI[family]
    k, q, lam = 20, 1e-4, 0.995
    p0 = p0_for(k, "calibrated")
    X, y, w, _ = make_data(family, k, T, seed, q_true=q, phi=phi, first_zero=True)

    def mk():
        if filt == "ekf":
            return KalmanGLM(k, family=family, q=q, lam=lam, phi=phi, p0=p0, form="joseph")
        return KalmanGLM(k, family=family, filt="ukf", q=q, lam=lam, phi=phi, p0=p0)

    mine = run_filter(mk(), X, y, w)
    band = run_filter(perturbed(mk()), X, y, w)
    t0 = time.perf_counter()
    if filt == "ekf":
        ref = filterpy_ekf(X, y, w, family, k, q, lam, phi, p0)
        ref_rd = None
    else:
        ref = filterpy_ukf(X, y, w, family, k, q, lam, phi, p0)
        ref_rd = filterpy_ukf(X, y, w, family, k, q, lam, phi, p0, redraw=True)
    secs = time.perf_counter() - t0
    out = {
        "family": family,
        "filt": filt,
        "k": k,
        "vs_filterpy": gap(mine, ref),
        "band": gap(band, mine),
        "filterpy_seconds": secs,
    }
    if ref_rd is not None:
        out["vs_filterpy_exact_prior"] = gap(mine, ref_rd)
    return out


def part3(args) -> dict:
    res: dict = {"sweep20": [], "zero_lead": [], "grid": [], "pf": [], "static": []}
    fams = ("poisson", "binomial", "gamma")
    for i, family in enumerate(fams):
        for filt in ("ekf", "ukf"):
            res["sweep20"].append(_sweep20(family, filt, args.rows, 3000 + i))
            for k in (2, 5, 20):
                res["zero_lead"].append(_zero_lead(family, filt, k))
    for i, family in enumerate(("gaussian",) + fams):
        for q, lam in ((1e-4, 1.0), (0.0, 0.99), (1e-4, 0.995)):
            res["grid"].append(_grid_case(family, q, lam, args.grid_rows, 4000 + i, args.grid_n))
            print("grid", family, q, lam, file=sys.stderr, flush=True)
    for i, family in enumerate(fams):
        res["pf"].append(_pf_case(family, args.grid_rows, 5000 + i, args.particles, 4))
        print("pf", family, file=sys.stderr, flush=True)
    for i, family in enumerate(fams):
        for k in (2, 5, 20):
            res["static"].append(
                _static_case(family, k, args.rows, 6000 + 10 * i + k, (200, args.rows))
            )
    return res


# --------------------------------------------------------------------------- part 3: divergence


def _run_safe(f: KalmanGLM, X, y, w) -> dict:
    """run_filter that records the first row a filter fails on (a Cholesky of a
    non-positive-definite P, a non-positive innovation variance, a non-finite b)
    instead of raising."""
    T = len(y)
    out = {
        "pred": np.full(T, np.nan),
        "b": np.full((T, f.k), np.nan),
        "P": np.full((T, f.k, f.k), np.nan),
        "fail_row": None,
        "fail": None,
    }
    for t in range(T):
        try:
            with np.errstate(all="ignore"):
                r = f.step(X[t], y[t], w[t])
        except np.linalg.LinAlgError as e:
            out["fail_row"], out["fail"] = t, f"LinAlgError: {e}"
            break
        out["pred"][t], out["b"][t], out["P"][t] = r["pred"], f.b, f.P
        if not (np.all(np.isfinite(f.b)) and np.all(np.isfinite(f.P))):
            out["fail_row"], out["fail"] = t, "non-finite state"
            break
        if np.min(np.linalg.eigvalsh(f.P)) <= 0:
            out["fail_row"], out["fail"] = t, "P not positive definite"
            break
    return out


def _clamp_rows(f: KalmanGLM, X, b_path) -> int:
    eta = np.einsum("tk,tk->t", X[1:], b_path[:-1])
    return int(np.sum(np.abs(eta[np.isfinite(eta)]) >= f.fam.eta_max))


def _lost(b, P, beta, z=4.0) -> bool:
    """The last row's b more than z of its own sd from the truth in some coefficient."""
    sd = np.sqrt(np.diag(P))
    return bool(np.any(np.abs(b - beta) / sd > z))


def divergence(args) -> dict:
    res: dict = {"overshoot": [], "spike": [], "separation": [], "small_alpha": []}
    T = args.div_rows
    # D1: over seeds, how often each filter reaches the clamp, and how often it ends
    # more than 4 of its own sd from the true coefficients
    for family in ("poisson", "binomial", "gamma"):
        phi = PHI[family]
        for k in (2, 5):
            for prior in ("calibrated", "wide"):
                p0 = p0_for(k, prior)
                for filt in ("ekf", "ukf"):
                    clamped = lost = failed = 0
                    for seed in range(args.seeds):
                        X, y, w, beta = make_data(family, k, T, 7000 + seed, q_true=1e-4, phi=phi)
                        f = KalmanGLM(k, family=family, filt=filt, q=1e-4, phi=phi, p0=p0)
                        r = _run_safe(f, X, y, w)
                        if r["fail_row"] is not None:
                            failed += 1
                            continue
                        clamped += _clamp_rows(f, X, r["b"]) > 0
                        lost += _lost(r["b"][-1], r["P"][-1], beta[-1])
                    res["overshoot"].append(
                        {
                            "family": family,
                            "k": k,
                            "prior": prior,
                            "filt": filt,
                            "seeds": args.seeds,
                            "reached_clamp": clamped,
                            "lost_track": lost,
                            "failed": failed,
                        }
                    )
    # D2: feature spikes: beta = (1, 0.1) fixed, x1 = 60, 300 and 1000 on rows
    # T/4, T/2 and 3T/4 (true eta 7, 31 and 101; there y is the true mean, rounded,
    # past eta 30), a filter that has drifted (b1 starts at 0.4) at q = 1e-4 from
    # the calibrated prior; with and without the clamp
    rng = np.random.default_rng(77)
    k = 2
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    spikes = [T // 4, T // 2, 3 * T // 4]
    X[spikes, 1] = (60.0, 300.0, 1000.0)
    beta = np.array([1.0, 0.1])
    eta_true = X @ beta
    y = rng.poisson(np.exp(np.minimum(eta_true, 30.0))).astype(float)
    big = eta_true > 30
    y[big] = np.round(np.exp(eta_true[big]))
    w = np.ones(T)
    for filt in ("ekf", "ukf"):
        for clamp in (True, False):
            # a filter that has drifted: b1 starts 0.3 off, P0 calibrated
            f = KalmanGLM(
                k,
                family="poisson",
                filt=filt,
                q=1e-4,
                p0=p0_for(k, "calibrated"),
                b0=np.array([1.0, 0.4]),
                clamp=clamp,
            )
            r = _run_safe(f, X, y, w)
            eta_spike = [float(X[s] @ r["b"][s - 1]) for s in spikes]
            after = {}
            for s in spikes:
                if r["fail_row"] is not None and r["fail_row"] <= s:
                    break
                err = np.abs(r["b"][s:] - beta) / np.sqrt(np.einsum("tii->ti", r["P"][s:]))
                back = np.where(np.all(err < 2, axis=1))[0]
                after[str(s)] = {
                    "b_after": r["b"][s].tolist(),
                    "P11_after": float(r["P"][s][1, 1]),
                    "rows_to_within_2sd": int(back[0]) if back.size else None,
                }
            res["spike"].append(
                {
                    "filt": filt,
                    "clamp": clamp,
                    "eta_at_spike": eta_spike,
                    "fail_row": r["fail_row"],
                    "fail": r["fail"],
                    "after": after,
                    "final_b": r["b"][-1].tolist() if r["fail_row"] is None else None,
                }
            )
    # D3: separation, binomial logit: y = 1[x1 > 0], over 10 T rows
    T = 10 * T
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    y = (X[:, 1] > 0).astype(float)
    w = np.ones(T)
    for q, lam in ((0.0, 1.0), (1e-4, 1.0), (0.0, 0.99)):
        for filt in ("ekf", "ukf"):
            f = KalmanGLM(2, family="binomial", filt=filt, q=q, lam=lam, p0=p0_for(2, "calibrated"))
            r = _run_safe(f, X, y, w)
            marks = [t for t in (10, 100, 1000, T // 2, T - 1) if t < T]
            res["separation"].append(
                {
                    "filt": filt,
                    "q": q,
                    "lam": lam,
                    "fail_row": r["fail_row"],
                    "fail": r["fail"],
                    "b1": {str(t): float(r["b"][t][1]) for t in marks},
                    "sd1": {str(t): float(math.sqrt(r["P"][t][1, 1])) for t in marks},
                    "clamp_rows": _clamp_rows(f, X, r["b"]) if r["fail_row"] is None else None,
                }
            )
    T = T // 10
    # D4: the UKF with a small alpha has a negative centre weight; the expected R and
    # the innovation variance can then go negative (binomial: V is not convex)
    for alpha in (1.0, 0.5, 0.1, 1e-3):
        for family in ("poisson", "binomial"):
            fails = []
            for seed in range(10):
                X, y, w, beta = make_data(family, 5, T, 8000 + seed, q_true=1e-4)
                f = KalmanGLM(
                    5,
                    family=family,
                    filt="ukf",
                    q=1e-4,
                    p0=p0_for(5, "calibrated"),
                    points=("merwe", alpha, 2.0, 0.0),
                )
                r = _run_safe(f, X, y, w)
                fails.append(r["fail_row"])
            res["small_alpha"].append(
                {
                    "alpha": alpha,
                    "family": family,
                    "w0_mean": float(merwe_weights(5, alpha, 2.0, 0.0)[1][0]),
                    "failed": sum(x is not None for x in fails),
                    "of": len(fails),
                    "first_fail_rows": [x for x in fails if x is not None][:5],
                }
            )
    return res


def summarise_div(res):
    print("divergence")
    for r in res["overshoot"]:
        print(
            "  overshoot",
            {
                k: r[k]
                for k in (
                    "family",
                    "k",
                    "prior",
                    "filt",
                    "reached_clamp",
                    "lost_track",
                    "failed",
                    "seeds",
                )
            },
        )
    for r in res["spike"]:
        print(
            "  spike",
            r["filt"],
            "clamp" if r["clamp"] else "no clamp",
            "eta",
            [round(e, 2) for e in r["eta_at_spike"]],
            "fail",
            r["fail_row"],
            r["fail"],
            {s: v["rows_to_within_2sd"] for s, v in r["after"].items()},
        )
    for r in res["separation"]:
        print(
            "  separation",
            r["filt"],
            r["q"],
            r["lam"],
            "b1",
            r["b1"],
            "sd1",
            {k: round(v, 4) for k, v in r["sd1"].items()},
            "fail",
            r["fail_row"],
            "clamp rows",
            r["clamp_rows"],
        )
    for r in res["small_alpha"]:
        print("  small alpha", r)


# --------------------------------------------------------------------------- part 4


def state_bytes(k: int, packed: bool) -> int:
    """Persisted state per target: b (k), P (k(k+1)/2 packed or k^2 full), D, n."""
    return 8 * (k + (k * (k + 1) // 2 if packed else k * k) + 2)


def flops(k: int, filt: str, form: str = "standard") -> int:
    """Multiply-adds counted as 2 flops, per observing row, excluding the link
    and variance functions (counted separately as evaluations)."""
    prior = k * k + k  # P / lam^D (or none) and + q D^2 on the diagonal
    if filt == "ekf":
        base = prior + 2 * k * k + 2 * k + 2 * k  # Px, x'Px, K, b
        upd = {"standard": 2 * k * k, "joseph_rank1": 6 * k * k, "joseph": 4 * k**3 + 4 * k * k}
        return base + upd[form]
    n = 2 * k + 1
    chol = k**3 // 3 + k * k  # c P^- and its Cholesky
    return prior + chol + n * k + 2 * n * k + 2 * n + 2 * n * k + 2 * k + 2 * k * k


def _time_filter(make, X, y, w, budget_s: float = 5.0) -> float:
    """Microseconds per row, the best of repeated passes within budget_s."""
    best = math.inf
    t_end = time.perf_counter() + budget_s
    while time.perf_counter() < t_end:
        f = make()
        t0 = time.perf_counter()
        for t in range(len(y)):
            f.step(X[t], y[t], w[t])
        best = min(best, (time.perf_counter() - t0) / len(y) * 1e6)
    return best


def part4(args) -> dict:
    res: dict = {"cost": [], "rank1": []}
    for k in (2, 5, 20, 50):
        X, y, w, _ = make_data("poisson", k, 400, 9000 + k, zero_frac=0.0, var_frac=0.0)
        p0 = p0_for(k, "calibrated")
        row = {
            "k": k,
            "state_bytes_packed": state_bytes(k, True),
            "state_bytes_full": state_bytes(k, False),
            "ukf_scratch_bytes": 8 * ((2 * k + 1) * k + 2 * (2 * k + 1) + k * k),
        }
        for name, kw in (
            ("ekf_standard", {"filt": "ekf", "form": "standard"}),
            ("ekf_joseph_rank1", {"filt": "ekf", "form": "joseph_rank1"}),
            ("ekf_joseph", {"filt": "ekf", "form": "joseph"}),
            ("ukf", {"filt": "ukf"}),
        ):

            def make(kw=kw, k=k, p0=p0):
                return KalmanGLM(k, family="poisson", q=1e-4, lam=0.995, p0=p0, **kw)

            row[name + "_us_per_row_numpy"] = _time_filter(make, X, y, w, args.time_budget)
            row[name + "_flops"] = flops(k, kw["filt"], kw.get("form", "standard"))
        row["ukf_link_evals"] = 2 * k + 1
        row["ekf_link_evals"] = 1
        res["cost"].append(row)
        print("cost", k, file=sys.stderr, flush=True)
    # the rank-1 Joseph form against filterpy's matrix Joseph form, on the part 1
    # Poisson streams (both priors), and against the numpy matrix form
    for k in (2, 5):
        for q, lam in ((1e-4, 1.0), (0.0, 0.99), (1e-4, 0.995)):
            for first_zero in (False, True):
                seed = 1000 + 10 * k + int(first_zero)
                X, y, w, _ = make_data(
                    "poisson", k, args.rows, seed, q_true=1e-4, first_zero=first_zero
                )
                for kind in ("calibrated", "wide"):
                    p0 = p0_for(k, kind)
                    r1 = run_filter(
                        KalmanGLM(k, family="poisson", q=q, lam=lam, p0=p0, form="joseph_rank1"),
                        X,
                        y,
                        w,
                    )
                    jm = run_filter(
                        KalmanGLM(k, family="poisson", q=q, lam=lam, p0=p0, form="joseph"), X, y, w
                    )
                    ref = filterpy_ekf(X, y, w, "poisson", k, q, lam, 1.0, p0)
                    band = run_filter(
                        perturbed(
                            KalmanGLM(k, family="poisson", q=q, lam=lam, p0=p0, form="joseph")
                        ),
                        X,
                        y,
                        w,
                    )
                    res["rank1"].append(
                        {
                            "k": k,
                            "q": q,
                            "lam": lam,
                            "first_zero": first_zero,
                            "prior": kind,
                            "clamped": _clamped(jm["pred"], "poisson"),
                            "rank1_vs_filterpy": gap(r1, ref),
                            "rank1_vs_joseph": gap(r1, jm),
                            "band": gap(band, jm),
                        }
                    )
    return res


def summarise4(res):
    print("part 4")
    for r in res["cost"]:
        print(
            f"  k={r['k']}: state {r['state_bytes_packed']} B packed, "
            f"{r['state_bytes_full']} B full; UKF scratch {r['ukf_scratch_bytes']} B"
        )
        for name in ("ekf_standard", "ekf_joseph_rank1", "ekf_joseph", "ukf"):
            print(
                f"     {name:18s} {r[name + '_us_per_row_numpy']:8.1f} us/row (numpy), "
                f"{r[name + '_flops']:8d} flops"
            )
    for c in (0, 1):

        def sel(r, c=c):
            return (r["clamped"] > 0) == bool(c)

        n = sum(1 for r in res["rank1"] if sel(r))
        tag = "clamped" if c else "unclamped"
        print(
            f"  rank-1 Joseph [{tag}, {n}] vs filterpy",
            _worst(res["rank1"], "rank1_vs_filterpy", where=sel),
            "vs matrix Joseph",
            _worst(res["rank1"], "rank1_vs_joseph", where=sel),
            "band",
            _worst(res["rank1"], "band", where=sel),
        )


# --------------------------------------------------------------------------- driver


def _worst(rows: list, key: str, field_: str = "mixed", where=None) -> dict:
    """The largest gap over the selected cases: "mixed" reads pred and P per row and
    b against the run's largest value (gap's doc)."""
    out = {}
    for r in rows:
        if where is not None and not where(r):
            continue
        for q in ("pred", "b", "P"):
            if q in r[key]:
                f = field_ if field_ != "mixed" else ("rel_run" if q == "b" else "rel")
                out[q] = max(out.get(q, 0.0), r[key][q][f])
    return {q: float(f"{v:.3g}") for q, v in out.items()}


def summarise1(res):
    print("part 1 (largest relative gap over cases; 'band' is the 1-ulp self-perturbation)")
    g = res["gaussian"]
    print("  gaussian joseph vs KalmanFilter:", _worst(g, "vs_KalmanFilter"))
    print("  gaussian joseph vs EKF:         ", _worst(g, "vs_EKF"))
    print("  gaussian standard vs KF:        ", _worst(g, "standard_vs_KalmanFilter"))
    print("  gaussian band:                  ", _worst(g, "band"))
    p = res["poisson_ekf"]
    for kind in ("calibrated", "wide"):
        sel = lambda r, kind=kind: r["prior"] == kind  # noqa: E731
        print(
            f"  poisson EKF [{kind}] joseph vs filterpy:  ",
            _worst(p, "joseph_vs_filterpy", where=sel),
        )
        print(
            f"  poisson EKF [{kind}] standard vs filterpy:",
            _worst(p, "standard_vs_filterpy", where=sel),
        )
        print(f"  poisson EKF [{kind}] band:                ", _worst(p, "band", where=sel))
        print(
            f"  poisson EKF [{kind}] max pred, clamped rows:",
            max(r["max_pred"] for r in p if sel(r)),
            max(r["clamped_rows"] for r in p if sel(r)),
        )
    print("  out of sample:", res["out_of_sample"])
    print("  zero-weight rows:", res["zero_weight_rows_are_a_gap"])


def summarise3(res):
    print("part 3")
    for r in res["sweep20"]:
        extra = r.get("vs_filterpy_exact_prior")
        print(
            f"  k=20 {r['filt']} {r['family']}: vs filterpy",
            _worst([r], "vs_filterpy"),
            "exact prior",
            _worst([r], "vs_filterpy_exact_prior") if extra else "-",
            "band",
            _worst([r], "band"),
            f"filterpy {r['filterpy_seconds']:.1f}s",
        )
    bad = [
        r
        for r in res["zero_lead"]
        if not (
            r["b_at_prior"]
            and r["P_at_prior"]
            and r["n_eff_zero"]
            and r["all_finite"]
            and r["fold_max_abs_b"] == 0.0
        )
    ]
    print(f"  zero-weight lead: {len(res['zero_lead'])} cases, failing: {bad}")
    keys = (
        "z_max",
        "z_median",
        "z_last",
        "sd_ratio_min",
        "sd_ratio_max",
        "pred_rel_median",
        "pred_rel_max",
    )
    for r in res["grid"]:
        print(
            f"  grid {r['family']} q={r['q']} lam={r['lam']} n={r['n']} "
            f"h/sd={r['spacing_over_min_sd']:.2f} ({r['grid_seconds']:.0f}s)"
        )
        for name in ("grid_band", "ekf", "ukf", "grid_vs_KalmanFilter"):
            if name in r:
                print(f"     {name:22s}", {k: float(f"{r[name][k]:.3g}") for k in keys})
    for r in res["pf"]:
        print(
            f"  pf {r['family']} N={r['N']}x{r['reps']} ess_min={r['ess_min']:.0f} "
            f"mc_sd/sd max={r['pf_mc_sd_over_post_sd_max']:.3g} "
            f"median={r['pf_mc_sd_over_post_sd_median']:.3g} ({r['pf_seconds']:.0f}s)"
        )
        for name in ("ekf", "ukf"):
            print(f"     {name:22s}", {k: float(f"{r[name][k]:.3g}") for k in keys})
            if f"{name}_beyond_4mc_max" in r:
                print(
                    f"     {name} beyond 4 mc sd: {r[name + '_beyond_4mc_max']:.3g}; "
                    f"z max where the PF is sharp ({r['sharp_fraction']:.2f} of cells): "
                    f"{r[name + '_z_max_where_sharp']:.3g}"
                )
    for r in res["static"]:
        print(f"  static {r['family']} k={r['k']}:", json.dumps(r["rows"]))


PARTS = {
    "part1": (part1, summarise1),
    "part2": (part2, summarise2),
    "part3": (part3, summarise3),
    "divergence": (divergence, summarise_div),
    "part4": (part4, summarise4),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_497"))
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--grid-rows", type=int, default=1000)
    ap.add_argument("--grid-n", type=int, default=500)
    ap.add_argument("--particles", type=int, default=50000)
    ap.add_argument("--div-rows", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--time-budget", type=float, default=4.0, help="seconds per timed case")
    ap.add_argument("--force", action="store_true", help="recompute a part whose result exists")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.part}.json")
    run, summarise = PARTS[args.part]
    if os.path.exists(path) and not args.force:
        with open(path) as fh:
            res = json.load(fh)
        print(f"(cached: {path})")
    else:
        t0 = time.perf_counter()
        res = run(args)
        res["_seconds"] = time.perf_counter() - t0
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
    summarise(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
