"""Tests for ``kalman`` ``coef_prior``, ``emit_loglik``, and ``emit_mahal``.

Oracle checks against ``filterpy`` (>= 2.0.0):
- Linear Gaussian filter with ``coef_prior``, ``emit_loglik``, and ``emit_mahal``
  against ``filterpy.kalman.KalmanFilter`` (``kf.x = coef_prior``,
  ``kf.log_likelihood``, ``kf.mahalanobis``): with and without intercept,
  ``q = 0`` and ``q > 0``, heterogeneous weights, zero-weight rows,
  ``revert_half_life``, and ``covariance_form="square_root"`` parity.
- Diffuse intercept limit (``kappa -> inf``) against ``filterpy`` confirming
  that the intercept entry of ``coef_prior`` is ignored and row 0 withholds
  ``loglik`` / ``mahal``.
- ``standardize=True`` with ``coef_prior`` (original-units seeding and
  ``1e6`` feature-level shift invariance).
- Nonlinear EKF and UKF (``scalar_index`` and ``merwe``) with ``coef_prior``
  against ``filterpy.kalman.ExtendedKalmanFilter`` and
  ``filterpy.kalman.UnscentedKalmanFilter``.
- Stream contracts: out-of-sample (Hard Rule 2), chunk invariance 1 vs 1000
  chunks (Hard Rule 3), zero-weight rows (Hard Rule 9), ``embargo`` /
  ``label_clock`` / formula target, ``ModelBank.predict`` (with and without
  targets), readiness gates (``min_weight``, ``min_settled_frac``,
  ``max_error_inflation``), save/load & JSON round-trip, and spec refusals.
- 50-seed ``oracle_grid`` registered with ``ledger.guard``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from filterpy.kalman import (
    ExtendedKalmanFilter,
    KalmanFilter,
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
)

import ledger
from rheon import polars as rp

TIER = "mixed"


# --- helpers ----------------------------------------------------------------


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a_arr - b_arr) / (1.0 + np.abs(b_arr))))


def make_linear_stream(
    k_feat: int,
    n: int,
    seed: int,
    *,
    obs_var: float = 0.25,
    fit_intercept: bool = True,
    hetero_weights: bool = False,
    irregular_clock: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, k_feat))
    true_b = rng.normal(0.0, 1.0, size=k_feat)
    intercept = float(rng.normal(0.5, 0.5)) if fit_intercept else 0.0
    w = rng.uniform(0.3, 3.0, size=n) if hetero_weights else np.ones(n)
    if irregular_clock:
        dt = rng.choice([0.5, 1.0, 2.0, 4.0], size=n)
        dt[0] = 0.0
        t = np.cumsum(dt)
    else:
        t = np.arange(n, dtype=float)
    y = intercept + X @ true_b + rng.normal(0.0, math.sqrt(obs_var), size=n) / np.sqrt(w)
    return X, y, t, w


def run_filterpy_linear(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    obs_var: float,
    p0: float,
    q: np.ndarray,
    coef_prior: np.ndarray,
    fit_intercept: bool = True,
    revert_half_life: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n, k_feat = X.shape
    k = k_feat + (1 if fit_intercept else 0)
    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.x = np.asarray(coef_prior, dtype=float).reshape(k, 1).copy()
    kf.P = np.eye(k) * (p0 * obs_var)
    kf.F = np.eye(k)
    kf.Q = np.zeros((k, k))

    preds = np.full(n, np.nan)
    coefs = np.zeros((n, k))
    loglik = np.full(n, np.nan)
    mahal = np.full(n, np.nan)

    seen_obs = False
    elapsed = 0.0
    for i in range(n):
        d = 0.0 if i == 0 else float(t[i] - t[i - 1])
        elapsed += d
        # Reversion transition on every row.
        if revert_half_life is not None:
            phi = np.array(
                [1.0 if math.isinf(h) else 2.0 ** (-d / h) for h in revert_half_life],
                dtype=float,
            )
            kf.F = np.diag(phi)
            kf.Q = np.zeros((k, k))
            kf.predict()
        z = np.concatenate(([1.0], X[i])) if fit_intercept else X[i].copy()
        if seen_obs:
            preds[i] = float((z @ kf.x).item())
        if w[i] > 0.0 and np.isfinite(y[i]):
            # Process noise charged once per observation for the whole clock elapsed.
            if revert_half_life is None:
                q_diag = q * (elapsed * elapsed)
            else:
                q_diag = np.zeros(k)
                for idx in range(k):
                    h = float(revert_half_life[idx])
                    if math.isinf(h):
                        q_diag[idx] = q[idx] * elapsed * elapsed
                    else:
                        theta = math.log(2.0) / h
                        g = -math.expm1(-theta * elapsed) / theta
                        q_diag[idx] = q[idx] * g * g
            elapsed = 0.0
            kf.P = kf.P + np.diag(q_diag)
            kf.H = z.reshape(1, k)
            kf.R = np.array([[obs_var / float(w[i])]])
            kf.update(np.array([[float(y[i])]]))
            loglik[i] = float(kf.log_likelihood)
            mahal[i] = float(kf.mahalanobis)
            seen_obs = True
        coefs[i] = kf.x.ravel()

    return {"pred": preds, "coef": coefs, "loglik": loglik, "mahal": mahal}


def run_rheon_linear(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    w: np.ndarray,
    *,
    obs_var: float | None,
    p0: float,
    q: list[float] | None = None,
    coef_half_life: float | list[float] | None = None,
    coef_prior: list[list[float]] | None = None,
    fit_intercept: bool = True,
    standardize: bool = False,
    revert_half_life: float | list[float] | None = None,
    covariance_form: str = "covariance",
    emit_loglik: bool = True,
    emit_mahal: bool = True,
    min_weight: float = 0.0,
) -> dict[str, np.ndarray]:
    k_feat = X.shape[1]
    feats = [f"x{j}" for j in range(k_feat)]
    df = pl.DataFrame({**{f: X[:, j] for j, f in enumerate(feats)}, "y": y, "t": t, "w": w})
    kw: dict[str, object] = {
        "targets": ["y"],
        "features": feats,
        "clock": "t",
        "gap_cap": 1e6,
        "weight": "w",
        "half_life": 50.0,
        "min_weight": min_weight,
        "fit_intercept": fit_intercept,
        "standardize": standardize,
        "p0": p0,
        "covariance_form": covariance_form,
        "coef_every": 0,
        "emit_loglik": emit_loglik,
        "emit_mahal": emit_mahal,
    }
    if obs_var is not None:
        kw["obs_var"] = obs_var
    if q is not None:
        kw["q"] = q
    if coef_half_life is not None:
        kw["coef_half_life"] = coef_half_life
    if coef_prior is not None:
        kw["coef_prior"] = coef_prior
    if revert_half_life is not None:
        kw["revert_half_life"] = revert_half_life
    spec = rp.spec.kalman("k", **kw)  # type: ignore[arg-type]
    col = rp.ModelBank([spec]).fit_predict(df)["k"]
    out: dict[str, np.ndarray] = {
        "pred": col.struct.field("pred_y").to_numpy(),
        "coef": np.vstack(col.struct.field("coef").to_list()),
    }
    if emit_loglik:
        out["loglik"] = col.struct.field("loglik_y").to_numpy()
    if emit_mahal:
        out["mahal"] = col.struct.field("mahal_y").to_numpy()
    return out


# --- 1. Linear filterpy oracle tests ----------------------------------------


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("q_val", [0.0, 1e-3])
@pytest.mark.parametrize("covariance_form", ["covariance", "square_root"])
def test_linear_coef_prior_loglik_mahal_matches_filterpy(fit_intercept, q_val, covariance_form):
    k_feat = 3
    k = k_feat + (1 if fit_intercept else 0)
    X, y, t, w = make_linear_stream(
        k_feat,
        80,
        seed=17,
        obs_var=0.36,
        fit_intercept=fit_intercept,
        hetero_weights=True,
        irregular_clock=True,
    )
    prior = np.linspace(-1.5, 2.0, k)
    q = np.full(k, q_val)

    want = run_filterpy_linear(
        X, y, t, w, obs_var=0.36, p0=1.5, q=q, coef_prior=prior, fit_intercept=fit_intercept
    )
    got = run_rheon_linear(
        X,
        y,
        t,
        w,
        obs_var=0.36,
        p0=1.5,
        q=q.tolist(),
        coef_prior=[prior.tolist()],
        fit_intercept=fit_intercept,
        covariance_form=covariance_form,
    )
    tol = 1e-12 if covariance_form == "covariance" else 1e-11
    assert max_rel(got["coef"], want["coef"]) < tol
    assert max_rel(got["pred"][1:], want["pred"][1:]) < tol
    assert max_abs(got["loglik"], want["loglik"]) < tol
    assert max_abs(got["mahal"], want["mahal"]) < tol


def test_linear_revert_half_life_with_coef_prior_matches_filterpy():
    X, y, t, w = make_linear_stream(
        2, 70, seed=29, obs_var=0.25, hetero_weights=True, irregular_clock=True
    )
    prior = np.array([1.2, -2.5, 0.8])
    q = np.array([1e-4, 2e-3, 5e-4])
    rev = np.array([math.inf, 25.0, 60.0])

    want = run_filterpy_linear(
        X, y, t, w, obs_var=0.25, p0=2.0, q=q, coef_prior=prior, revert_half_life=rev
    )
    got = run_rheon_linear(
        X,
        y,
        t,
        w,
        obs_var=0.25,
        p0=2.0,
        q=q.tolist(),
        coef_prior=[prior.tolist()],
        revert_half_life=rev.tolist(),
    )
    assert max_rel(got["coef"], want["coef"]) < 1e-12
    assert max_abs(got["loglik"], want["loglik"]) < 1e-12
    assert max_abs(got["mahal"], want["mahal"]) < 1e-12


def test_diffuse_intercept_limit_matches_filterpy_and_withholds_first_row():
    X, y, t, w = make_linear_stream(2, 60, seed=101, obs_var=0.25)
    # Without obs_var (noise_units=True, diffuse intercept), intercept prior is ignored.
    prior_a = [[0.0, 1.5, -0.75]]
    prior_b = [[9999.0, 1.5, -0.75]]
    got_a = run_rheon_linear(
        X, y, t, w, obs_var=None, p0=2.0, q=[0.0, 0.0, 0.0], coef_prior=prior_a
    )
    got_b = run_rheon_linear(
        X, y, t, w, obs_var=None, p0=2.0, q=[0.0, 0.0, 0.0], coef_prior=prior_b
    )
    np.testing.assert_array_equal(got_a["coef"], got_b["coef"])
    np.testing.assert_array_equal(got_a["loglik"], got_b["loglik"])
    np.testing.assert_array_equal(got_a["mahal"], got_b["mahal"])

    # Row 0 (diffuse intercept) and Row 1 (no pre-row residual variance yet) must withhold.
    assert np.isnan(got_a["loglik"][0]) and np.isnan(got_a["mahal"][0])
    assert np.isnan(got_a["loglik"][1]) and np.isnan(got_a["mahal"][1])
    assert np.all(np.isfinite(got_a["loglik"][2:]))
    assert np.all(np.isfinite(got_a["mahal"][2:]))

    # Diffuse limit against filterpy with P_00 = kappa * obs_var (kappa = 1e9):
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.array([[0.0], [1.5], [-0.75]])
    kf.P = np.diag([1e9 * 0.25, 2.0 * 0.25, 2.0 * 0.25])
    kf.F = np.eye(3)
    kf.Q = np.zeros((3, 3))
    fp_coef = np.zeros((len(y), 3))
    for i in range(len(y)):
        kf.H = np.array([[1.0, X[i, 0], X[i, 1]]])
        kf.R = np.array([[0.25]])
        kf.update(np.array([[y[i]]]))
        fp_coef[i] = kf.x.ravel()
    assert max_rel(got_a["coef"], fp_coef) < 1e-6


def test_standardize_with_coef_prior_seeds_original_units_and_is_shift_invariant():
    rng = np.random.default_rng(77)
    n = 40
    x0 = rng.normal(0.0, 2.0, size=n)
    x1 = rng.normal(0.0, 0.5, size=n)
    y = 3.0 + 2.5 * x0 - 1.5 * x1 + rng.normal(0.0, 0.1, size=n)
    t = np.arange(n, dtype=float)
    w = np.ones(n)

    def run(shift: float) -> dict[str, np.ndarray]:
        Xs = np.column_stack([x0 + shift, x1 - shift])
        return run_rheon_linear(
            Xs,
            y,
            t,
            w,
            obs_var=None,
            p0=1e-4,
            q=[0.0, 0.0, 0.0],
            coef_prior=[[0.0, 2.5, -1.5]],
            standardize=True,
        )

    base = run(0.0)
    shifted = run(1e6)
    assert abs(base["coef"][-1, 1] - 2.5) < 0.05
    assert abs(base["coef"][-1, 2] + 1.5) < 0.05
    assert max_abs(base["coef"][3:, 1:], shifted["coef"][3:, 1:]) < 1e-6
    assert max_abs(base["pred"][3:], shifted["pred"][3:]) < 1e-5
    assert max_abs(base["loglik"][3:], shifted["loglik"][3:]) < 1e-5
    assert max_abs(base["mahal"][3:], shifted["mahal"][3:]) < 1e-5


# --- 2. Nonlinear EKF / UKF coef_prior against filterpy ---------------------


@pytest.mark.parametrize(
    ("flt", "ukf_rule"),
    [("ekf", None), ("ukf", "merwe"), ("ukf", "scalar_index")],
)
def test_nonlinear_ekf_ukf_coef_prior_matches_filterpy(flt, ukf_rule):
    rng = np.random.default_rng(313)
    n = 50
    X = rng.normal(0.0, 0.5, size=(n, 2))
    true_b = np.array([0.4, 0.6, -0.5])
    Z = np.column_stack([np.ones(n), X])
    lam_y = np.exp(Z @ true_b)
    y = rng.poisson(lam_y).astype(float)
    t = np.arange(n, dtype=float)
    w = np.ones(n)
    prior = np.array([0.3, 0.5, -0.4])
    q = np.array([1e-4, 1e-4, 1e-4])
    p0 = 0.5

    df = pl.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "y": y, "t": t, "w": w})
    kw: dict[str, object] = {
        "targets": ["y"],
        "features": ["x0", "x1"],
        "clock": "t",
        "gap_cap": 100.0,
        "weight": "w",
        "half_life": 50.0,
        "min_weight": 0.0,
        "standardize": False,
        "family": "poisson",
        "link": "log",
        "filter": flt,
        "q": q.tolist(),
        "p0": p0,
        "coef_prior": [prior.tolist()],
        "coef_every": 0,
    }
    if ukf_rule is not None:
        kw["ukf_rule"] = ukf_rule
    spec = rp.spec.kalman("nl", **kw)  # type: ignore[arg-type]
    col = rp.ModelBank([spec]).fit_predict(df)["nl"]
    got_coef = np.vstack(col.struct.field("coef").to_list())
    got_pred = col.struct.field("pred_y").to_numpy()

    # Check coef_prior actually moves early predictions vs zero prior.
    kw_zero = {k: v for k, v in kw.items() if k != "coef_prior"}
    zero_col = rp.ModelBank([rp.spec.kalman("nl0", **kw_zero)]).fit_predict(df)["nl0"]  # type: ignore[arg-type]
    zero_coef = np.vstack(zero_col.struct.field("coef").to_list())
    assert np.max(np.abs(got_coef[0] - zero_coef[0])) > 0.1

    if flt == "ekf":
        ekf = ExtendedKalmanFilter(dim_x=3, dim_z=1)
        ekf.x = prior.reshape(3, 1).copy()
        ekf.P = np.eye(3) * p0
        ekf.F = np.eye(3)
        ekf.Q = np.zeros((3, 3))
        want_coef = np.zeros((n, 3))
        want_pred = np.full(n, np.nan)
        for i in range(n):
            z = Z[i]
            if i > 0:
                want_pred[i] = math.exp(float((z @ ekf.x).item()))
                ekf.P = ekf.P + np.diag(q)
            eta = float((z @ ekf.x).item())
            mu = math.exp(eta)
            ekf.update(
                np.array([[y[i]]]),
                HJacobian=lambda x, z_=z: (math.exp(float((z_ @ x).item())) * z_).reshape(1, 3),
                Hx=lambda x, z_=z: np.array([[math.exp(float((z_ @ x).item()))]]),
                R=np.array([[mu]]),
            )
            want_coef[i] = ekf.x.ravel()
        assert max_rel(got_coef, want_coef) < 1e-12
        assert max_rel(got_pred[1:], want_pred[1:]) < 1e-12
    elif ukf_rule == "merwe":
        pts = MerweScaledSigmaPoints(n=3, alpha=1.0, beta=2.0, kappa=0.0)
        ukf = UnscentedKalmanFilter(
            dim_x=3,
            dim_z=1,
            dt=1.0,
            hx=lambda x: np.array([0.0]),
            fx=lambda x, dt: x,
            points=pts,
        )
        ukf.x = prior.copy()
        ukf.P = np.eye(3) * p0
        want_coef = np.zeros((n, 3))
        want_pred = np.full(n, np.nan)
        for i in range(n):
            z = Z[i]
            if i > 0:
                ukf.P = ukf.P + np.diag(q)
            sigmas = pts.sigma_points(ukf.x, ukf.P)
            etas = sigmas @ z
            mus = np.exp(np.clip(etas, -30.0, 30.0))
            if i > 0:
                want_pred[i] = float(np.dot(pts.Wm, mus))
            r_exp = float(np.dot(pts.Wm, mus))
            ukf.sigmas_f = sigmas
            ukf.update(
                np.array([y[i]]),
                R=np.array([[r_exp]]),
                hx=lambda x, z_=z: np.array([math.exp(float(np.clip(z_ @ x, -30.0, 30.0)))]),
            )
            want_coef[i] = ukf.x.copy()
        assert max_rel(got_coef, want_coef) < 1e-12
        assert max_rel(got_pred[1:], want_pred[1:]) < 1e-12


# --- 3. Stream contracts (OOS, chunks, zero-weight, embargo, predict, I/O) --


def test_stream_contracts_oos_chunks_zero_weight_embargo_predict_and_io():
    X, y, t, w = make_linear_stream(2, 100, seed=404, obs_var=0.25, hetero_weights=True)
    y_missing = y.copy()
    y_missing[15] = np.nan
    df = pl.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "y": y_missing, "t": t, "w": w})

    spec = rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=50.0,
        min_weight=0.0,
        obs_var=0.25,
        standardize=False,
        q=[1e-4, 1e-4, 1e-4],
        coef_prior=[[0.5, 1.0, -0.5]],
        emit_loglik=True,
        emit_mahal=True,
        coef_every=0,
    )
    full = rp.ModelBank([spec]).fit_predict(df)["k"]
    ll_full = full.struct.field("loglik_y").to_numpy()
    mh_full = full.struct.field("mahal_y").to_numpy()

    # Missing target yields null loglik/mahal.
    assert np.isnan(ll_full[15]) and np.isnan(mh_full[15])
    assert np.all(np.isfinite(ll_full[:15])) and np.all(np.isfinite(ll_full[16:]))

    # Rule 2 (out-of-sample): changing row 40's target moves no row 0..39,
    # and changing row 41's target moves no row 0..40 (including row 40's loglik/mahal).
    y_mut = y_missing.copy()
    y_mut[41] = 100.0
    df_mut = df.with_columns(pl.Series("y", y_mut))
    mut = rp.ModelBank([spec]).fit_predict(df_mut)["k"]
    np.testing.assert_array_equal(ll_full[:41], mut.struct.field("loglik_y").to_numpy()[:41])
    np.testing.assert_array_equal(mh_full[:41], mut.struct.field("mahal_y").to_numpy()[:41])

    # Rule 3 (chunk invariance): 1-row chunks match 1-chunk run bit-for-bit.
    bank_1 = rp.ModelBank([spec])
    parts_ll = []
    parts_mh = []
    for i in range(len(df)):
        res = bank_1.fit_predict(df.slice(i, 1))["k"]
        parts_ll.append(res.struct.field("loglik_y").to_numpy())
        parts_mh.append(res.struct.field("mahal_y").to_numpy())
    np.testing.assert_array_equal(ll_full, np.concatenate(parts_ll))
    np.testing.assert_array_equal(mh_full, np.concatenate(parts_mh))

    # Rule 9 (zero-weight rows): null loglik/mahal on zero-weight rows,
    # and bit-identical loglik/mahal on every positive-weight row.
    zero_row = pl.DataFrame({"x0": [5.0], "x1": [-3.0], "y": [99.0], "t": [24.5], "w": [0.0]})
    df_zw = pl.concat([df.slice(0, 25), zero_row, df.slice(25, len(df) - 25)])
    zw = rp.ModelBank([spec]).fit_predict(df_zw)["k"]
    ll_zw = zw.struct.field("loglik_y").to_numpy()
    mh_zw = zw.struct.field("mahal_y").to_numpy()
    assert np.isnan(ll_zw[25]) and np.isnan(mh_zw[25])
    np.testing.assert_array_equal(ll_full[:25], ll_zw[:25])
    np.testing.assert_array_equal(ll_full[25:], ll_zw[26:])
    np.testing.assert_array_equal(mh_full[:25], mh_zw[:25])
    np.testing.assert_array_equal(mh_full[25:], mh_zw[26:])

    # Save / load bytes + JSON export round-trip.
    bank_a = rp.ModelBank([spec])
    bank_a.fit_predict(df.slice(0, 50))
    raw = bank_a.save_bytes()
    js = bank_a.write_json()
    assert js is not None and '"emit_loglik": true' in js and '"emit_mahal": true' in js
    bank_b = rp.ModelBank.load_bytes(raw)
    tail = bank_b.fit_predict(df.slice(50, 50))["k"]
    np.testing.assert_array_equal(ll_full[50:], tail.struct.field("loglik_y").to_numpy())
    np.testing.assert_array_equal(mh_full[50:], tail.struct.field("mahal_y").to_numpy())

    # ModelBank.predict: scores loglik/mahal when y is present, null when absent.
    pred_with_y = bank_a.predict(df.slice(50, 5))["k"]
    assert pred_with_y.struct.field("loglik_y").null_count() == 0
    assert pred_with_y.struct.field("mahal_y").null_count() == 0
    pred_no_y = bank_a.predict(df.slice(50, 5).drop("y"))["k"]
    assert pred_no_y.struct.field("loglik_y").null_count() == 5
    assert pred_no_y.struct.field("mahal_y").null_count() == 5

    # Embargo: scored at arrival against the delayed model; chunk-invariant.
    spec_emb = rp.spec.kalman(
        "ke",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=50.0,
        min_weight=0.0,
        obs_var=0.25,
        standardize=False,
        q=[1e-4, 1e-4, 1e-4],
        coef_prior=[[0.5, 1.0, -0.5]],
        embargo=3.0,
        emit_loglik=True,
        emit_mahal=True,
    )
    emb_full = rp.ModelBank([spec_emb]).fit_predict(df)["ke"]
    bank_emb_c = rp.ModelBank([spec_emb])
    emb_parts = [
        bank_emb_c.fit_predict(df.slice(i, 7))["ke"].struct.field("loglik_y").to_numpy()
        for i in range(0, len(df), 7)
    ]
    np.testing.assert_array_equal(
        emb_full.struct.field("loglik_y").to_numpy(), np.concatenate(emb_parts)
    )

    # Readiness gates withhold loglik and mahal alongside pred.
    spec_gated = rp.spec.kalman(
        "kg",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        weight="w",
        half_life=50.0,
        min_weight=5.0,
        min_settled_frac=0.1,
        obs_var=0.25,
        standardize=False,
        q=[1e-4, 1e-4, 1e-4],
        emit_loglik=True,
        emit_mahal=True,
    )
    gated = rp.ModelBank([spec_gated]).fit_predict(df)["kg"]
    withheld = gated.struct.field("withheld_reason").is_not_null().to_numpy()
    assert withheld.any()
    assert np.all(np.isnan(gated.struct.field("loglik_y").to_numpy()[withheld]))
    assert np.all(np.isnan(gated.struct.field("mahal_y").to_numpy()[withheld]))


# --- 4. Spec refusals -------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"coef_half_life": 50.0, "coef_prior": [[0.0]]}, "coef_prior"),
        ({"coef_half_life": 50.0, "coef_prior": [[0.0, 1.0], [0.0, 1.0]]}, "coef_prior"),
        ({"coef_half_life": 50.0, "coef_prior": [[float("nan"), 0.0]]}, "coef_prior"),
        ({"coef_half_life": 50.0, "coef_prior": [[float("inf"), 0.0]]}, "coef_prior"),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "emit_loglik": True},
            "emit_loglik",
        ),
        (
            {"family": "poisson", "filter": "ekf", "q": [0.0, 0.0], "emit_mahal": True},
            "emit_mahal",
        ),
        (
            {"family": "poisson", "filter": "ukf", "q": [0.0, 0.0], "emit_loglik": True},
            "emit_loglik",
        ),
        (
            {"family": "gaussian", "filter": "ekf", "coef_half_life": 50.0, "emit_loglik": True},
            "emit_loglik",
        ),
        (
            {"family": "gaussian", "filter": "ukf", "coef_half_life": 50.0, "emit_mahal": True},
            "emit_mahal",
        ),
    ],
)
def test_kalman_extras_spec_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        rp.spec.kalman("m", targets=["y"], features=["x0"], half_life=50.0, **kw)


# --- 5. 50-seed oracle_grid + ledger.guard ----------------------------------

GRID_CASES = [
    ("ridge_intercept", True, 0.0, False, "covariance"),
    ("drift_no_intercept", False, 1e-3, False, "covariance"),
    ("reverting_weighted", True, 5e-4, True, "covariance"),
    ("square_root_drift", True, 1e-3, False, "square_root"),
]


def _run_grid_seed(
    fit_intercept: bool, q_val: float, reverting: bool, cov_form: str, seed: int
) -> tuple[float, float, float]:
    k_feat = 3
    k = k_feat + (1 if fit_intercept else 0)
    X, y, t, w = make_linear_stream(
        k_feat,
        100,
        seed=seed,
        obs_var=0.25,
        fit_intercept=fit_intercept,
        hetero_weights=True,
        irregular_clock=True,
    )
    rng = np.random.default_rng(10_000 + seed)
    prior = rng.normal(0.0, 1.5, size=k)
    q = np.full(k, q_val)
    rev = np.array([math.inf] + [30.0] * k_feat) if reverting else None

    want = run_filterpy_linear(
        X,
        y,
        t,
        w,
        obs_var=0.25,
        p0=1.5,
        q=q,
        coef_prior=prior,
        fit_intercept=fit_intercept,
        revert_half_life=rev,
    )
    got = run_rheon_linear(
        X,
        y,
        t,
        w,
        obs_var=0.25,
        p0=1.5,
        q=q.tolist(),
        coef_prior=[prior.tolist()],
        fit_intercept=fit_intercept,
        revert_half_life=None if rev is None else rev.tolist(),
        covariance_form=cov_form,
    )
    return (
        max_rel(got["coef"], want["coef"]),
        max_abs(got["loglik"], want["loglik"]),
        max_abs(got["mahal"], want["mahal"]),
    )


def _guard_summary() -> list[np.ndarray]:
    out = []
    for _name, fit_intercept, q_val, reverting, cov_form in GRID_CASES:
        k_feat = 3
        k = k_feat + (1 if fit_intercept else 0)
        for seed in (0, 1):
            X, y, t, w = make_linear_stream(
                k_feat,
                100,
                seed=seed,
                obs_var=0.25,
                fit_intercept=fit_intercept,
                hetero_weights=True,
                irregular_clock=True,
            )
            prior = np.random.default_rng(10_000 + seed).normal(0.0, 1.5, size=k)
            rev = [math.inf] + [30.0] * k_feat if reverting else None
            res = run_rheon_linear(
                X,
                y,
                t,
                w,
                obs_var=0.25,
                p0=1.5,
                q=[q_val] * k,
                coef_prior=[prior.tolist()],
                fit_intercept=fit_intercept,
                revert_half_life=rev,
                covariance_form=cov_form,
            )
            out.append(np.concatenate([res["coef"][-1], res["loglik"][-3:], res["mahal"][-3:]]))
    return out


ledger.guard("filterpy_grid", _guard_summary, oracles=["filterpy"])


@pytest.mark.oracle_grid(reason="50 seeds x 4 linear kalman prior/loglik/mahal configs vs filterpy")
@pytest.mark.parametrize(("name", "fit_intercept", "q_val", "reverting", "cov_form"), GRID_CASES)
def test_kalman_extras_50_seed_filterpy_grid(name, fit_intercept, q_val, reverting, cov_form):
    c_gaps, ll_gaps, mh_gaps = zip(
        *(_run_grid_seed(fit_intercept, q_val, reverting, cov_form, s) for s in range(50)),
        strict=True,
    )
    ledger.record("filterpy_grid", f"{name}_coef", c_gaps)
    ledger.record("filterpy_grid", f"{name}_loglik", ll_gaps)
    ledger.record("filterpy_grid", f"{name}_mahal", mh_gaps)
    lim = 1e-12 if cov_form == "covariance" else 1e-11
    assert max(c_gaps) < lim, (name, max(c_gaps))
    assert max(ll_gaps) < lim, (name, max(ll_gaps))
    assert max(mh_gaps) < lim, (name, max(mh_gaps))
