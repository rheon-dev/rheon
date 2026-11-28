"""Extended and unscented Kalman filters on ``kalman`` against filterpy and
exact posteriors.

Oracle checks:
- Linear Gaussian identity EKF and UKF (``scalar_index`` and ``merwe``,
  ``expected`` and ``plugin``) against ``filterpy.kalman.KalmanFilter``.
- EKF across all nine nonlinear ``(family, link)`` pairs (random walk and
  reverting) against ``filterpy.kalman.ExtendedKalmanFilter``.
- UKF ``merwe`` against ``filterpy.kalman.UnscentedKalmanFilter`` with
  ``MerweScaledSigmaPoints`` (exact linear prior handed over each row).
- UKF ``scalar_index`` against ``filterpy.kalman.UnscentedKalmanFilter`` with
  ``JulierSigmaPoints(k, kappa=3 - k)`` and a feature-aligned ``sqrt_method``.
- Fixed-lag smoothing (``smooth_lag``) with EKF and UKF against
  ``filterpy.kalman.rts_smoother`` on the filtered steps.
- Exact ``k = 2`` grid posterior confirming ``scalar_index`` accuracy over
  ``merwe`` and feature-order invariance at ``k = 20``.
- Stream contracts (out-of-sample, zero-weight gap folding, chunk invariance,
  state save/load and JSON export/import, target support refusals, metrics,
  and spec refusals).
- 50-seed ``oracle_grid`` registered with ``ledger.guard``.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import polars as pl
import pytest
from filterpy.kalman import (
    ExtendedKalmanFilter,
    JulierSigmaPoints,
    KalmanFilter,
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
    rts_smoother,
)

import ledger
from rheon import polars as rp

TIER = "mixed"

ETA_CLIP = 30.0


# --- family math matching family.rs -----------------------------------------


def mean_of(link: str, eta: float, var_power: float = 1.5) -> float:
    if link == "identity":
        return float(eta)
    if link == "log":
        return float(math.exp(max(-ETA_CLIP, min(ETA_CLIP, float(eta)))))
    if link == "logit":
        e = max(-ETA_CLIP, min(ETA_CLIP, float(eta)))
        return float(1.0 / (1.0 + math.exp(-e)))
    if link == "inverse":
        return float(1.0 / eta)
    if link == "power":
        return float(eta ** (1.0 / (1.0 - var_power)))
    raise ValueError(link)


def inv_link_deriv_of(link: str, mu: float, var_power: float = 1.5) -> float:
    if link == "identity":
        return 1.0
    if link == "log":
        return float(mu)
    if link == "logit":
        return float(mu * (1.0 - mu))
    if link == "inverse":
        return float(-(mu * mu))
    if link == "power":
        return float((mu**var_power) / (1.0 - var_power))
    raise ValueError(link)


def var_of(family: str, mu: float, var_power: float = 1.5) -> float:
    if family == "gaussian":
        return 1.0
    if family == "poisson":
        return float(mu)
    if family == "binomial":
        return float(mu * (1.0 - mu))
    if family == "gamma":
        return float(mu * mu)
    if family == "tweedie":
        return float(mu**var_power)
    raise ValueError(family)


def gap_noise_scalar(d: float, revert_hl: float | None) -> float:
    if revert_hl is None or math.isinf(revert_hl):
        return d * d
    theta = math.log(2.0) / revert_hl
    g = -math.expm1(-theta * d) / theta
    return g * g


def phi_scalar(d: float, revert_hl: float | None) -> float:
    if revert_hl is None or math.isinf(revert_hl):
        return 1.0
    return float(2.0 ** (-d / revert_hl))


# --- synthetic streams ------------------------------------------------------


def make_stream(
    family: str,
    link: str,
    *,
    k_feat: int = 2,
    n: int = 200,
    seed: int = 0,
    var_power: float = 1.5,
    dispersion: float = 1.0,
    obs_var: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a well-conditioned stream ``(X_raw, y, t, w)`` in the family's
    support (``X_raw`` has shape ``(n, k_feat)``, without intercept)."""
    tag = zlib.crc32(f"{family}:{link}:{var_power}".encode()) & 0xFFFF
    rng = np.random.default_rng([497, seed, k_feat, tag])
    X = rng.normal(0.0, 0.35, size=(n, k_feat))
    dt = np.concatenate([[0.0], rng.choice([0.5, 1.0, 1.5, 2.0], size=n - 1)])
    t = np.cumsum(dt)
    w = rng.uniform(0.6, 1.8, size=n)

    if link in ("inverse", "power"):
        b0 = np.zeros(k_feat + 1)
        b0[0] = 1.1
        b0[1:] = rng.normal(0.0, 0.12 / math.sqrt(k_feat), size=k_feat)
    else:
        b0 = np.zeros(k_feat + 1)
        b0[0] = 0.25 if link != "logit" else 0.1
        b0[1:] = rng.normal(0.0, 0.35 / math.sqrt(k_feat), size=k_feat)

    Z = np.column_stack([np.ones(n), X])
    eta = Z @ b0
    if link in ("inverse", "power"):
        eta = np.clip(eta, 0.4, 3.0)
    mu = np.array([mean_of(link, float(e), var_power) for e in eta])

    if family == "gaussian":
        y = mu + rng.normal(0.0, math.sqrt(obs_var), size=n)
    elif family == "poisson":
        y = rng.poisson(mu).astype(float)
    elif family == "binomial":
        y = (rng.random(n) < mu).astype(float)
    elif family == "gamma":
        shape = 1.0 / dispersion
        y = rng.gamma(shape, mu * dispersion)
    elif family == "tweedie":
        if 1.0 < var_power < 2.0:
            lam = (mu ** (2.0 - var_power)) / (dispersion * (2.0 - var_power))
            shape = (2.0 - var_power) / (var_power - 1.0)
            scale = dispersion * (var_power - 1.0) * (mu ** (var_power - 1.0))
            cnt = rng.poisson(lam)
            y = np.zeros(n)
            pos = cnt > 0
            y[pos] = rng.gamma(cnt[pos] * shape, scale[pos])
        else:
            # Inverse-Gaussian-like positive amounts with mean mu and variance dispersion * mu^p
            v = dispersion * (mu**var_power)
            shape = (mu * mu) / v
            y = rng.gamma(shape, v / mu)
    else:
        raise ValueError(family)
    return X, y, t, w


def run_rheon(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    **spec_kw: object,
) -> dict[str, np.ndarray]:
    k_feat = X.shape[1]
    feats = [f"x{i}" for i in range(k_feat)]
    df = pl.DataFrame({**{feats[i]: X[:, i] for i in range(k_feat)}, "y": y, "t": t, "w": w})
    spec = rp.spec.kalman(
        "k",
        targets=["y"],
        features=feats,
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=float("inf"),
        min_weight=0.0,
        standardize=False,
        coef_every=0,
        emit_se_coef=True,
        **spec_kw,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["k"]
    res: dict[str, np.ndarray] = {
        "pred": out.struct.field("pred_y").to_numpy(),
        "coef": np.vstack(out.struct.field("coef").to_list()),
        "se_coef": np.vstack(out.struct.field("se_coef").to_list()),
    }
    if "smooth_lag" in spec_kw and spec_kw["smooth_lag"] is not None:
        res["smoothed_coef"] = out.struct.field("smoothed_coef").to_list()
        if spec_kw.get("emit_smoothed_se"):
            res["smoothed_se_coef"] = out.struct.field("smoothed_se_coef").to_list()
        res["smoothed_for"] = out.struct.field("smoothed_for_y").to_list()
    return res


# --- filterpy oracle runners ------------------------------------------------


def run_filterpy_kf(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    q: np.ndarray,
    obs_var: float,
    p0: float,
    revert_hl: float | None = None,
) -> dict[str, np.ndarray]:
    n, k_feat = X.shape
    k = k_feat + 1
    Z = np.column_stack([np.ones(n), X])
    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.x = np.zeros(k)
    kf.P = np.eye(k) * (p0 * obs_var)
    preds = np.full(n, np.nan)
    coefs = np.empty((n, k))
    ses = np.empty((n, k))
    for i in range(n):
        d = 0.0 if i == 0 else float(t[i] - t[i - 1])
        ph = phi_scalar(d, revert_hl)
        gn = gap_noise_scalar(d, revert_hl)
        kf.F = np.eye(k) * ph
        kf.Q = np.diag(q * gn)
        kf.predict()
        if i >= 1:
            preds[i] = float(Z[i] @ kf.x)
        kf.H = Z[i : i + 1]
        kf.R = np.array([[obs_var / w[i]]])
        kf.update(np.array([y[i]]))
        coefs[i] = kf.x
        ses[i] = np.sqrt(np.diag(kf.P))
    return {"pred": preds, "coef": coefs, "se_coef": ses}


def run_filterpy_ekf(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    family: str,
    link: str,
    q: np.ndarray,
    p0: float,
    var_power: float = 1.5,
    dispersion: float = 1.0,
    obs_var: float | None = None,
    revert_hl: float | None = None,
) -> dict[str, np.ndarray]:
    n, k_feat = X.shape
    k = k_feat + 1
    Z = np.column_stack([np.ones(n), X])
    ekf = ExtendedKalmanFilter(dim_x=k, dim_z=1)
    ekf.x = np.zeros(k)
    if link in ("inverse", "power"):
        ekf.x[0] = 1.0
    ekf.P = np.eye(k) * p0
    r_scale = obs_var if family == "gaussian" else dispersion
    assert r_scale is not None

    preds = np.full(n, np.nan)
    coefs = np.empty((n, k))
    ses = np.empty((n, k))
    Ps = np.empty((n, k, k))
    Fs = np.empty((n, k, k))
    Qs = np.empty((n, k, k))

    for i in range(n):
        d = 0.0 if i == 0 else float(t[i] - t[i - 1])
        ph = phi_scalar(d, revert_hl)
        gn = gap_noise_scalar(d, revert_hl)
        ekf.F = np.eye(k) * ph
        ekf.Q = np.diag(q * gn)
        Fs[i] = ekf.F.copy()
        Qs[i] = ekf.Q.copy()
        ekf.predict()
        zi = Z[i]
        eta = float(zi @ ekf.x)
        mu = mean_of(link, eta, var_power)
        if i >= 1:
            preds[i] = mu
        m_jac = inv_link_deriv_of(link, mu, var_power)
        r_eff = (r_scale * var_of(family, mu, var_power)) / float(w[i])

        def h_jac(_x: np.ndarray, _zi: np.ndarray = zi, _m: float = m_jac) -> np.ndarray:
            return (_m * _zi)[None, :]

        def h_x(_x: np.ndarray, _mu: float = mu) -> np.ndarray:
            return np.array([_mu])

        ekf.update(np.array([y[i]]), HJacobian=h_jac, Hx=h_x, R=np.array([[r_eff]]))
        ekf.P = 0.5 * (ekf.P + ekf.P.T)
        coefs[i] = ekf.x
        Ps[i] = ekf.P
        ses[i] = np.sqrt(np.diag(ekf.P))
    return {"pred": preds, "coef": coefs, "se_coef": ses, "P": Ps, "F": Fs, "Q": Qs}


def _aligned_sqrt_factory(z_holder: list[np.ndarray]):
    """Householder-aligned square root ``A`` with ``A.T @ A = M`` whose first
    row points along ``M @ z`` so every other sigma point projects onto the
    mean along ``z`` (making filterpy's ``JulierSigmaPoints(k, 3 - k)``
    algebraically equal to the 3-point Gauss-Hermite scalar-index rule)."""

    def sqrt_fn(M: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(M)
        z = z_holder[0]
        v = L.T @ z
        nv = float(np.linalg.norm(v))
        if nv == 0.0:
            return L.T
        u = v / nv
        e0 = np.zeros_like(u)
        e0[0] = 1.0
        sign = -1.0 if u[0] >= 0.0 else 1.0
        w = u - sign * e0
        nw2 = float(w @ w)
        Q = np.eye(len(u)) if nw2 == 0.0 else np.eye(len(u)) - 2.0 * np.outer(w, w) / nw2
        return Q.T @ L.T

    return sqrt_fn


def run_filterpy_ukf(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    family: str,
    link: str,
    q: np.ndarray,
    p0: float,
    ukf_rule: str = "merwe",
    ukf_noise: str = "expected",
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    var_power: float = 1.5,
    dispersion: float = 1.0,
    obs_var: float | None = None,
    revert_hl: float | None = None,
) -> dict[str, np.ndarray]:
    n, k_feat = X.shape
    k = k_feat + 1
    Z = np.column_stack([np.ones(n), X])
    z_holder: list[np.ndarray] = [Z[0]]

    if ukf_rule == "merwe":
        pts = MerweScaledSigmaPoints(k, alpha=alpha, beta=beta, kappa=kappa)
    elif ukf_rule == "scalar_index":
        pts = JulierSigmaPoints(k, kappa=3.0 - k, sqrt_method=_aligned_sqrt_factory(z_holder))
    else:
        raise ValueError(ukf_rule)

    def fx(x: np.ndarray, _dt: float) -> np.ndarray:
        return x

    def hx(x: np.ndarray) -> np.ndarray:
        return np.array([mean_of(link, float(z_holder[0] @ x), var_power)])

    ukf = UnscentedKalmanFilter(dim_x=k, dim_z=1, dt=1.0, hx=hx, fx=fx, points=pts)
    b = np.zeros(k)
    if link in ("inverse", "power"):
        b[0] = 1.0
    is_gauss_id = family == "gaussian" and link == "identity"
    r_scale = obs_var if family == "gaussian" else dispersion
    assert r_scale is not None
    P = np.eye(k) * (p0 * obs_var if is_gauss_id else p0)

    preds = np.full(n, np.nan)
    coefs = np.empty((n, k))
    ses = np.empty((n, k))
    Ps = np.empty((n, k, k))
    Fs = np.empty((n, k, k))
    Qs = np.empty((n, k, k))

    for i in range(n):
        d = 0.0 if i == 0 else float(t[i] - t[i - 1])
        ph = phi_scalar(d, revert_hl)
        gn = gap_noise_scalar(d, revert_hl)
        F_i = np.eye(k) * ph
        Q_i = np.diag(q * gn)
        Fs[i] = F_i
        Qs[i] = Q_i
        b_minus = ph * b
        P_minus = (ph * ph) * P + Q_i
        P_minus = 0.5 * (P_minus + P_minus.T)
        z_holder[0] = Z[i]

        # Hand filterpy the exact linear prior `(b⁻, P⁻)` and its sigma points
        # so the test is invariant across filterpy 1.4.5 and 2.x.
        ukf.x = b_minus.copy()
        ukf.P = P_minus.copy()
        ukf.sigmas_f = pts.sigma_points(b_minus, P_minus)
        sigmas_h = np.array([hx(s) for s in ukf.sigmas_f])
        zbar = float(pts.Wm @ sigmas_h[:, 0])
        if i >= 1:
            preds[i] = zbar
        if ukf_noise == "expected":
            v_bar = float(
                pts.Wm @ np.array([var_of(family, float(sh[0]), var_power) for sh in sigmas_h])
            )
        else:
            mu0 = mean_of(link, float(Z[i] @ b_minus), var_power)
            v_bar = var_of(family, mu0, var_power)
        r_eff = (r_scale * v_bar) / float(w[i])
        ukf.update(np.array([y[i]]), R=np.array([[r_eff]]))
        b = ukf.x.copy()
        P = 0.5 * (ukf.P + ukf.P.T)
        coefs[i] = b
        Ps[i] = P
        ses[i] = np.sqrt(np.diag(P))
    return {"pred": preds, "coef": coefs, "se_coef": ses, "P": Ps, "F": Fs, "Q": Qs}


def max_rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0)))


# --- 1. Linear Gaussian identity vs filterpy.KalmanFilter -------------------


@pytest.mark.parametrize(
    ("flt", "kw"),
    [
        ("ekf", {}),
        ("ukf", {"ukf_rule": "scalar_index", "ukf_noise": "expected"}),
        ("ukf", {"ukf_rule": "scalar_index", "ukf_noise": "plugin"}),
        ("ukf", {"ukf_rule": "merwe", "ukf_noise": "expected"}),
        ("ukf", {"ukf_rule": "merwe", "ukf_noise": "plugin"}),
    ],
)
@pytest.mark.parametrize("revert_hl", [None, 40.0])
def test_gaussian_identity_ekf_and_ukf_match_filterpy_kf(flt, kw, revert_hl):
    X, y, t, w = make_stream("gaussian", "identity", k_feat=3, n=150, seed=11, obs_var=0.4)
    q = np.full(4, 2e-4)
    got = run_rheon(
        X,
        y,
        t,
        w,
        filter=flt,
        family="gaussian",
        link="identity",
        q=q.tolist(),
        obs_var=0.4,
        p0=1.2,
        revert_half_life=revert_hl,
        **kw,
    )
    want = run_filterpy_kf(X, y, t, w, q=q, obs_var=0.4, p0=1.2, revert_hl=revert_hl)
    assert max_rel(got["coef"], want["coef"]) < 1e-13
    assert max_rel(got["se_coef"], want["se_coef"]) < 1e-13
    assert np.isnan(got["pred"][0])
    assert float(np.max(np.abs(got["pred"][1:] - want["pred"][1:]))) < 1e-13


# --- 2. EKF across all 9 nonlinear (family, link) pairs vs filterpy EKF -----


NONLINEAR_PAIRS = [
    ("poisson", "log", None, 1.0, None, 1.0),
    ("binomial", "logit", None, 1.0, None, 1.0),
    ("gamma", "log", None, 0.5, None, 1.0),
    ("gamma", "inverse", None, 0.5, None, 0.05),
    ("tweedie", "log", 1.5, 0.8, None, 1.0),
    ("tweedie", "power", 1.5, 0.8, None, 0.05),
    ("tweedie", "power", 2.5, 0.8, None, 0.05),
    ("gaussian", "log", None, None, 0.25, 0.5),
    ("gaussian", "logit", None, None, 0.04, 0.5),
]


@pytest.mark.parametrize(
    ("family", "link", "var_power", "dispersion", "obs_var", "p0"), NONLINEAR_PAIRS
)
@pytest.mark.parametrize("revert_hl", [None, 50.0])
def test_ekf_matches_filterpy_ekf_all_families_and_links(
    family, link, var_power, dispersion, obs_var, p0, revert_hl
):
    X, y, t, w = make_stream(
        family,
        link,
        k_feat=2,
        n=160,
        seed=7,
        var_power=var_power or 1.5,
        dispersion=dispersion or 1.0,
        obs_var=obs_var or 0.25,
    )
    q = np.full(3, 1e-4)
    kw: dict[str, object] = {
        "family": family,
        "link": link,
        "filter": "ekf",
        "q": q.tolist(),
        "p0": p0,
        "revert_half_life": revert_hl,
    }
    if var_power is not None:
        kw["var_power"] = var_power
    if dispersion is not None:
        kw["dispersion"] = dispersion
    if obs_var is not None:
        kw["obs_var"] = obs_var

    got = run_rheon(X, y, t, w, **kw)
    want = run_filterpy_ekf(
        X,
        y,
        t,
        w,
        family=family,
        link=link,
        q=q,
        p0=p0,
        var_power=var_power or 1.5,
        dispersion=dispersion or 1.0,
        obs_var=obs_var,
        revert_hl=revert_hl,
    )
    assert max_rel(got["coef"], want["coef"]) < 1e-12
    assert max_rel(got["se_coef"], want["se_coef"]) < 1e-12
    assert max_rel(got["pred"][1:], want["pred"][1:]) < 1e-12


# --- 3. UKF Merwe vs filterpy.UnscentedKalmanFilter -------------------------


@pytest.mark.parametrize(
    ("family", "link", "obs_var", "disp", "ukf_noise", "alpha"),
    [
        ("poisson", "log", None, 1.0, "expected", 1.0),
        ("poisson", "log", None, 1.0, "plugin", 1.0),
        ("poisson", "log", None, 1.0, "expected", 0.5),
        ("binomial", "logit", None, 1.0, "expected", 1.0),
        ("gamma", "log", None, 0.5, "expected", 1.0),
        ("gaussian", "logit", 0.05, None, "expected", 1.0),
    ],
)
def test_ukf_merwe_matches_filterpy_ukf(family, link, obs_var, disp, ukf_noise, alpha):
    X, y, t, w = make_stream(
        family,
        link,
        k_feat=3,
        n=160,
        seed=19,
        dispersion=disp or 1.0,
        obs_var=obs_var or 0.25,
    )
    q = np.full(4, 1e-4)
    kw: dict[str, object] = {
        "family": family,
        "link": link,
        "filter": "ukf",
        "ukf_rule": "merwe",
        "ukf_noise": ukf_noise,
        "ukf_alpha": alpha,
        "ukf_beta": 2.0,
        "ukf_kappa": 0.0,
        "q": q.tolist(),
        "p0": 0.5,
        "revert_half_life": 60.0,
    }
    if disp is not None:
        kw["dispersion"] = disp
    if obs_var is not None:
        kw["obs_var"] = obs_var
    got = run_rheon(X, y, t, w, **kw)
    want = run_filterpy_ukf(
        X,
        y,
        t,
        w,
        family=family,
        link=link,
        q=q,
        p0=0.5,
        ukf_rule="merwe",
        ukf_noise=ukf_noise,
        alpha=alpha,
        beta=2.0,
        kappa=0.0,
        dispersion=disp or 1.0,
        obs_var=obs_var,
        revert_hl=60.0,
    )
    assert max_rel(got["coef"], want["coef"]) < 2e-14
    assert max_rel(got["se_coef"], want["se_coef"]) < 5e-14
    assert max_rel(got["pred"][1:], want["pred"][1:]) < 5e-14


# --- 4. UKF scalar_index vs filterpy JulierSigmaPoints(k, 3 - k) aligned ----


@pytest.mark.parametrize(
    ("family", "link", "obs_var", "disp", "ukf_noise"),
    [
        ("poisson", "log", None, 1.0, "expected"),
        ("poisson", "log", None, 1.0, "plugin"),
        ("binomial", "logit", None, 1.0, "expected"),
        ("gamma", "log", None, 0.5, "expected"),
        ("gaussian", "logit", 0.05, None, "expected"),
    ],
)
def test_ukf_scalar_index_matches_filterpy_julier_aligned(family, link, obs_var, disp, ukf_noise):
    X, y, t, w = make_stream(
        family,
        link,
        k_feat=4,
        n=160,
        seed=23,
        dispersion=disp or 1.0,
        obs_var=obs_var or 0.25,
    )
    q = np.full(5, 1e-4)
    kw: dict[str, object] = {
        "family": family,
        "link": link,
        "filter": "ukf",
        "ukf_rule": "scalar_index",
        "ukf_noise": ukf_noise,
        "q": q.tolist(),
        "p0": 0.5,
        "revert_half_life": 50.0,
    }
    if disp is not None:
        kw["dispersion"] = disp
    if obs_var is not None:
        kw["obs_var"] = obs_var
    got = run_rheon(X, y, t, w, **kw)
    want = run_filterpy_ukf(
        X,
        y,
        t,
        w,
        family=family,
        link=link,
        q=q,
        p0=0.5,
        ukf_rule="scalar_index",
        ukf_noise=ukf_noise,
        dispersion=disp or 1.0,
        obs_var=obs_var,
        revert_hl=50.0,
    )
    assert max_rel(got["coef"], want["coef"]) < 2e-14
    assert max_rel(got["se_coef"], want["se_coef"]) < 5e-14
    assert max_rel(got["pred"][1:], want["pred"][1:]) < 5e-14


# --- 5. smooth_lag composition vs filterpy.rts_smoother ---------------------


@pytest.mark.parametrize(
    ("flt", "ukf_rule"),
    [("ekf", None), ("ukf", "scalar_index"), ("ukf", "merwe")],
)
def test_smooth_lag_ekf_and_ukf_match_filterpy_rts_smoother(flt, ukf_rule):
    X, y, t, w = make_stream("poisson", "log", k_feat=2, n=60, seed=31)
    q = np.full(3, 1e-3)
    lag = 6
    kw: dict[str, object] = {
        "family": "poisson",
        "link": "log",
        "filter": flt,
        "q": q.tolist(),
        "p0": 0.5,
        "revert_half_life": 40.0,
        "smooth_lag": lag,
        "emit_smoothed_se": True,
    }
    if ukf_rule is not None:
        kw["ukf_rule"] = ukf_rule
    got = run_rheon(X, y, t, w, **kw)
    if flt == "ekf":
        fp = run_filterpy_ekf(X, y, t, w, family="poisson", link="log", q=q, p0=0.5, revert_hl=40.0)
    else:
        fp = run_filterpy_ukf(
            X,
            y,
            t,
            w,
            family="poisson",
            link="log",
            q=q,
            p0=0.5,
            ukf_rule=ukf_rule or "scalar_index",
            revert_hl=40.0,
        )

    # At each emission row `i >= lag`, the fixed-lag smoother at `i - lag` equals
    # filterpy's `rts_smoother` run on observations `0..=i` read at index `i - lag`.
    # Note: filterpy's `rts_smoother` uses `Fs[k]` and `Qs[k]` as the transition
    # from `k` to `k + 1`, whereas our `fp["F"][k + 1]` / `fp["Q"][k + 1]` is that step.
    for i in range(lag, len(y)):
        idx = i - lag
        assert got["smoothed_for"][i] == pytest.approx(float(t[idx]), rel=0.0, abs=1e-12)
        Fs_sub = np.empty((i + 1, 3, 3))
        Qs_sub = np.empty((i + 1, 3, 3))
        Fs_sub[:-1] = fp["F"][1 : i + 1]
        Qs_sub[:-1] = fp["Q"][1 : i + 1]
        Fs_sub[-1] = np.eye(3)
        Qs_sub[-1] = np.zeros((3, 3))
        ms, Ps, _, _ = rts_smoother(fp["coef"][: i + 1], fp["P"][: i + 1], Fs_sub, Qs_sub)
        got_c = np.asarray(got["smoothed_coef"][i], dtype=float)
        got_se = np.asarray(got["smoothed_se_coef"][i], dtype=float)
        want_c = ms[idx]
        want_se = np.sqrt(np.diag(Ps[idx]))
        assert max_rel(got_c, want_c) < 1e-12
        assert max_rel(got_se, want_se) < 1e-12


# --- 6. Exact k = 2 grid posterior & k = 20 feature-order invariance --------


def _exact_grid_binomial(
    X: np.ndarray, y: np.ndarray, q: float, p0: float, n_grid: int = 81
) -> tuple[np.ndarray, np.ndarray]:
    """Exact 2D grid Bayes filter for ``k = 2`` (intercept + 1 slope) binomial
    logit with random walk ``Q = q I`` and prior ``N(0, p0 I)``."""
    lim = 3.5
    axis = np.linspace(-lim, lim, n_grid)
    h = axis[1] - axis[0]
    B0, B1 = np.meshgrid(axis, axis, indexing="ij")
    log_p = -0.5 * (B0 * B0 + B1 * B1) / p0
    log_p -= np.max(log_p)
    prob = np.exp(log_p)
    prob /= np.sum(prob)

    # 1D Gaussian diffusion kernel for `q` per step
    if q > 0.0:
        r = max(2, int(math.ceil(5.0 * math.sqrt(q) / h)))
        offs = np.arange(-r, r + 1) * h
        kern = np.exp(-0.5 * (offs * offs) / q)
        kern /= np.sum(kern)
    else:
        kern = None

    means = np.empty((len(y), 2))
    stds = np.empty((len(y), 2))
    for i in range(len(y)):
        if i > 0 and kern is not None:
            # Separable 2D convolution
            tmp = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 0, prob)
            prob = np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 1, tmp)
            prob /= np.sum(prob)
        eta = B0 + B1 * X[i, 0]
        p_y1 = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        lik = p_y1 if y[i] > 0.5 else (1.0 - p_y1)
        prob = prob * lik
        prob /= np.sum(prob)
        m0 = float(np.sum(prob * B0))
        m1 = float(np.sum(prob * B1))
        v0 = float(np.sum(prob * (B0 - m0) ** 2))
        v1 = float(np.sum(prob * (B1 - m1) ** 2))
        means[i] = [m0, m1]
        stds[i] = [math.sqrt(max(v0, 1e-18)), math.sqrt(max(v1, 1e-18))]
    return means, stds


def test_scalar_index_matches_exact_grid_and_is_feature_order_invariant():
    # Exact k = 2 binomial grid filter: both rules track the exact posterior
    # closely and scalar_index is closer than merwe.
    rng = np.random.default_rng(4970)
    n = 150
    X = rng.normal(0.0, 1.0, size=(n, 1))
    b_true = np.array([0.2, 0.8])
    p_true = 1.0 / (1.0 + np.exp(-(b_true[0] + b_true[1] * X[:, 0])))
    y = (rng.random(n) < p_true).astype(float)
    t = np.arange(n, dtype=float)
    w = np.ones(n)

    exact_m, exact_s = _exact_grid_binomial(X, y, q=1e-4, p0=1.0)
    sc = run_rheon(
        X, y, t, w, family="binomial", filter="ukf", ukf_rule="scalar_index", q=[1e-4, 1e-4], p0=1.0
    )
    mw = run_rheon(
        X, y, t, w, family="binomial", filter="ukf", ukf_rule="merwe", q=[1e-4, 1e-4], p0=1.0
    )
    z_sc = float(np.max(np.abs(sc["coef"] - exact_m) / exact_s))
    z_mw = float(np.max(np.abs(mw["coef"] - exact_m) / exact_s))
    assert z_sc < 0.15
    assert z_mw < 0.25
    assert z_sc < z_mw

    # Feature-order invariance at k = 20 on Poisson: reversing feature order
    # leaves scalar_index predictions invariant to machine epsilon.
    X20, y20, t20, w20 = make_stream("poisson", "log", k_feat=20, n=100, seed=99)
    q20 = [1e-4] * 21
    fwd = run_rheon(
        X20, y20, t20, w20, family="poisson", filter="ukf", ukf_rule="scalar_index", q=q20, p0=0.5
    )
    rev = run_rheon(
        X20[:, ::-1],
        y20,
        t20,
        w20,
        family="poisson",
        filter="ukf",
        ukf_rule="scalar_index",
        q=q20,
        p0=0.5,
    )
    assert max_rel(fwd["pred"][1:], rev["pred"][1:]) < 1e-13


# --- 7. Stream contracts: OOS, zero-weight, chunk invariance, state I/O -----


@pytest.mark.parametrize(
    ("flt", "extra"),
    [
        ("ekf", {}),
        ("ukf", {"ukf_rule": "scalar_index"}),
        ("ukf", {"ukf_rule": "merwe"}),
    ],
)
def test_stream_contracts_oos_zero_weight_chunks_and_state_io(flt, extra):
    X, y, t, w = make_stream("poisson", "log", k_feat=2, n=60, seed=41)
    df = pl.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "y": y, "t": t, "w": w})
    spec = rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=50.0,
        min_weight=0.0,
        family="poisson",
        filter=flt,
        q=[1e-4, 1e-4, 1e-4],
        coef_every=0,
        emit_metrics=True,
        emit_calibration=True,
        emit_breaks=True,
        **extra,
    )

    # Full run
    full = rp.ModelBank([spec]).fit_predict(df)["k"]
    full_pred = full.struct.field("pred_y").to_numpy()
    # Hit rate is null for Poisson (positive target, no sign to hit)
    assert full.struct.field("hit_rate_y").null_count() == len(df)

    # Out-of-sample (Rule 2): changing row 30's y and weight moves no pred on rows 0..=30.
    y_mut = y.copy()
    w_mut = w.copy()
    y_mut[30] = 500.0
    w_mut[30] = 9.0
    df_mut = df.with_columns(pl.Series("y", y_mut), pl.Series("w", w_mut))
    mut_pred = rp.ModelBank([spec]).fit_predict(df_mut)["k"].struct.field("pred_y").to_numpy()
    np.testing.assert_array_equal(full_pred[:31], mut_pred[:31])

    # Chunk invariance (Rule 3): 7-row chunks equal full run to the bit.
    bank_chunk = rp.ModelBank([spec])
    parts = [
        bank_chunk.fit_predict(df.slice(i, 7))["k"].struct.field("pred_y").to_numpy()
        for i in range(0, len(df), 7)
    ]
    np.testing.assert_array_equal(full_pred, np.concatenate(parts))

    # Save/load bytes continues bit-for-bit, and write_json emits valid JSON.
    bank_a = rp.ModelBank([spec])
    bank_a.fit_predict(df.slice(0, 30))
    raw_bytes = bank_a.save_bytes()
    json_text = bank_a.write_json()
    assert json_text is not None and f'"{flt}"' in json_text
    bank_b = rp.ModelBank.load_bytes(raw_bytes)
    tail = df.slice(30, 30)
    tail_b = bank_b.fit_predict(tail)["k"].struct.field("pred_y").to_numpy()
    np.testing.assert_array_equal(full_pred[30:], tail_b)

    # Zero-weight rows inserted inside a gap leave predictions unchanged (unstandardized).
    spec_raw = rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=float("inf"),
        min_weight=0.0,
        standardize=False,
        family="poisson",
        filter=flt,
        q=[1e-4, 1e-4, 1e-4],
        **extra,
    )
    clean_p = rp.ModelBank([spec_raw]).fit_predict(df)["k"].struct.field("pred_y").to_numpy()
    t_mid = float(0.5 * (t[19] + t[20]))
    zero_row = pl.DataFrame({"x0": [9.0], "x1": [-7.0], "y": [99.0], "t": [t_mid], "w": [0.0]})
    df_zw = pl.concat([df.slice(0, 20), zero_row, df.slice(20, len(df) - 20)])
    zw_p = rp.ModelBank([spec_raw]).fit_predict(df_zw)["k"].struct.field("pred_y").to_numpy()
    np.testing.assert_array_equal(clean_p[:20], zw_p[:20])
    np.testing.assert_array_equal(clean_p[20:], zw_p[21:])

    # Out-of-support target refused with row named.
    y_bad = y.copy()
    y_bad[12] = -1.0
    df_bad = df.with_columns(pl.Series("y", y_bad))
    with pytest.raises(ValueError, match=r"row 12.*family"):
        rp.ModelBank([spec]).fit_predict(df_bad)


def test_binomial_kalman_uses_threshold_hit_rate():
    X, y, t, w = make_stream("binomial", "logit", k_feat=2, n=60, seed=55)
    df = pl.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "y": y, "t": t, "w": w})
    spec = rp.spec.kalman(
        "b",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=50.0,
        min_weight=0.0,
        family="binomial",
        filter="ukf",
        q=[1e-4, 1e-4, 1e-4],
        emit_metrics=True,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["b"]
    hr = out.struct.field("hit_rate_y").drop_nulls().to_numpy()
    assert len(hr) > 10
    assert np.all((hr >= 0.0) & (hr <= 1.0))


# --- 8. Spec refusals -------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        (
            {"family": "poisson", "q": [0.0, 0.0]},
            'filter="linear" applies only to family="gaussian"',
        ),
        (
            {"family": "binomial", "link": "log", "filter": "ekf", "q": [0.0, 0.0]},
            "the binomial family takes the link logit",
        ),
        (
            {"family": "poisson", "filter": "ekf", "coef_half_life": 50.0},
            "coef_half_life cannot be used with family=",
        ),
        (
            {"family": "poisson", "filter": "ekf"},
            "needs an explicit q",
        ),
        (
            {"family": "gaussian", "link": "log", "filter": "ekf", "q": [0.0, 0.0]},
            "needs a fixed obs_var",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "obs_var": 1.0},
            "obs_var cannot be set with family=",
        ),
        (
            {"family": "gaussian", "coef_half_life": 50.0, "dispersion": 1.0},
            "dispersion is for the non-Gaussian families",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "dispersion": 0.0},
            "dispersion must be finite and above 0",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "var_power": 1.5},
            "var_power is the Tweedie family's power",
        ),
        (
            {"family": "tweedie", "filter": "ekf", "q": [0.0, 0.0], "var_power": 2.0},
            "var_power must be above 1, finite and not 2",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "share_p": True},
            "share_p cannot be used with a nonlinear observation",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "forecast_spans": [5.0]},
            "forecast_spans is only supported for",
        ),
        (
            {
                "family": "poisson",
                "filter": "ekf",
                "q": [0.0, 0.0],
                "covariance_form": "square_root",
            },
            'covariance_form="square_root" applies only to filter="linear"',
        ),
        (
            {
                "family": "gamma",
                "link": "inverse",
                "filter": "ekf",
                "q": [0.0],
                "fit_intercept": False,
            },
            "the inverse link gives a mean only where",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "ukf_rule": "merwe"},
            'ukf_rule applies only under filter="ukf"',
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "ukf_noise": "plugin"},
            'ukf_noise applies only under filter="ukf"',
        ),
        (
            {"family": "poisson", "filter": "ukf", "q": [0.0, 0.0], "ukf_alpha": 0.5},
            'ukf_alpha applies only under ukf_rule="merwe"',
        ),
        (
            {
                "family": "poisson",
                "filter": "ukf",
                "ukf_rule": "merwe",
                "q": [0.0, 0.0],
                "ukf_kappa": -2.0,
            },
            "Merwe sigma-point scale must be finite and above 0",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "max_error_inflation": 2.0},
            "max_error_inflation",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "emit_error_inflation": True},
            "emit_error_inflation",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "emit_influence": True},
            "emit_influence",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "emit_specification": True},
            "emit_specification",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "emit_tails": True},
            "emit_tails",
        ),
    ],
)
def test_nonlinear_kalman_spec_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        rp.spec.kalman("m", targets=["y"], features=["x0"], half_life=50.0, **kw)


# --- 9. 50-seed oracle_grid + ledger.guard ----------------------------------

GRID_CASES = [
    ("poisson", "log", "ekf", None),
    ("poisson", "log", "ukf", "scalar_index"),
    ("poisson", "log", "ukf", "merwe"),
    ("binomial", "logit", "ekf", None),
    ("binomial", "logit", "ukf", "scalar_index"),
    ("binomial", "logit", "ukf", "merwe"),
    ("gamma", "log", "ekf", None),
    ("gamma", "log", "ukf", "scalar_index"),
    ("gamma", "log", "ukf", "merwe"),
]


def _grid_case_name(family: str, flt: str, rule: str | None) -> str:
    return f"{family}_{flt}" if rule is None else f"{family}_{flt}_{rule}"


def _run_grid_seed(
    family: str, link: str, flt: str, rule: str | None, seed: int
) -> tuple[float, float]:
    X, y, t, w = make_stream(
        family, link, k_feat=2, n=120, seed=seed, dispersion=0.5 if family == "gamma" else 1.0
    )
    q = np.full(3, 1e-4)
    disp = 0.5 if family == "gamma" else 1.0
    kw: dict[str, object] = {
        "family": family,
        "link": link,
        "filter": flt,
        "q": q.tolist(),
        "p0": 0.5,
        "dispersion": disp,
    }
    if rule is not None:
        kw["ukf_rule"] = rule
    got = run_rheon(X, y, t, w, **kw)
    if flt == "ekf":
        want = run_filterpy_ekf(X, y, t, w, family=family, link=link, q=q, p0=0.5, dispersion=disp)
    else:
        want = run_filterpy_ukf(
            X,
            y,
            t,
            w,
            family=family,
            link=link,
            q=q,
            p0=0.5,
            ukf_rule=rule or "scalar_index",
            dispersion=disp,
        )
    return max_rel(got["coef"], want["coef"]), max_rel(got["pred"][1:], want["pred"][1:])


ledger.guard(
    "filterpy_grid",
    lambda: [
        run_rheon(
            *make_stream(
                fam, lnk, k_feat=2, n=120, seed=s, dispersion=0.5 if fam == "gamma" else 1.0
            ),
            family=fam,
            link=lnk,
            filter=flt,
            q=[1e-4, 1e-4, 1e-4],
            p0=0.5,
            dispersion=0.5 if fam == "gamma" else 1.0,
            **({"ukf_rule": rule} if rule is not None else {}),
        )["coef"][-1]
        for fam, lnk, flt, rule in GRID_CASES
        for s in (0, 1)
    ],
    oracles=["filterpy"],
)


@pytest.mark.oracle_grid(reason="50 seeds x 3 families x 3 nonlinear filters against filterpy")
@pytest.mark.parametrize(("family", "link", "flt", "rule"), GRID_CASES)
def test_nonlinear_kalman_50_seed_filterpy_grid(family, link, flt, rule):
    name = _grid_case_name(family, flt, rule)
    b_gaps, p_gaps = zip(
        *(_run_grid_seed(family, link, flt, rule, s) for s in range(50)), strict=True
    )
    ledger.record("filterpy_grid", f"{name}_coef", b_gaps)
    ledger.record("filterpy_grid", f"{name}_pred", p_gaps)
    b_lim = 1e-12 if flt == "ekf" else 2e-14
    p_lim = 1e-12 if flt == "ekf" else 5e-14
    assert max(b_gaps) < b_lim, (name, max(b_gaps))
    assert max(p_gaps) < p_lim, (name, max(p_gaps))
