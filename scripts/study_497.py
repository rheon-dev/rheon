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
    form: str = "standard"  # EKF covariance update: "standard" or "joseph"
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
                else:
                    IKH = self.I - np.outer(K, m * x)
                    P = IKH @ Pp @ IKH.T + R * np.outer(K, K)
                self.P = 0.5 * (P + P.T) if self.form == "standard" else P
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
    entries. "abs" is the plain largest |a - r|."""
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


# --------------------------------------------------------------------------- driver


def _worst(rows: list, key: str, field_: str = "rel", where=None) -> dict:
    out = {}
    for r in rows:
        if where is not None and not where(r):
            continue
        for q in ("pred", "b", "P"):
            if q in r[key]:
                out[q] = max(out.get(q, 0.0), r[key][q][field_])
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


PARTS = {"part1": (part1, summarise1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("--out", default=os.path.join(".cache", "study_497"))
    ap.add_argument("--rows", type=int, default=2000)
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
