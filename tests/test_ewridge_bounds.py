"""Tests and 50-seed SciPy oracles for bounded/fixed-sum ``ewridge`` and ``rheon.gram.solve``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

import ledger
import rheon
import rheon.polars as rp
from rheon import gram as rg

TIER = "mixed"
GRID_SEEDS = range(50)


def _decay_weights(
    times: np.ndarray,
    weights: np.ndarray,
    half_life: float,
) -> np.ndarray:
    """Exact exponentially decayed row weights at the end of a stream."""
    if math.isinf(half_life):
        return weights.astype(float).copy()
    dt = times[-1] - times
    return weights.astype(float) * np.power(0.5, dt / half_life)


def _scipy_ridge_qp(
    x: np.ndarray,
    y: np.ndarray,
    row_w: np.ndarray,
    *,
    ridge: float,
    standardize: bool,
    fit_intercept: bool,
    ridge_scale: str = "mean",
    coef_prior: np.ndarray | None = None,
    prior_scale: float = 1.0,
    coef_min: float | list[float] | None = None,
    coef_max: float | list[float] | None = None,
    coef_sum: float | None = None,
) -> tuple[np.ndarray, float]:
    """Independent third-party QP oracle via ``scipy.optimize.minimize`` (SLSQP).

    Computes the exact weighted ridge objective directly from ``(x, y, row_w)``
    without calling any ``rheon`` code, returning ``(coef, optimal_obj)``.
    """
    n, k = x.shape
    w_sum = float(np.sum(row_w))
    w_norm = row_w / w_sum

    lo = (
        np.full(k, -np.inf)
        if coef_min is None
        else (np.full(k, float(coef_min)) if np.isscalar(coef_min) else np.asarray(coef_min, float))
    )
    hi = (
        np.full(k, np.inf)
        if coef_max is None
        else (np.full(k, float(coef_max)) if np.isscalar(coef_max) else np.asarray(coef_max, float))
    )

    if ridge_scale == "sum":
        z = np.column_stack([np.ones(n), x]) if fit_intercept else x
        d = z.shape[1]
        h = (z.T * row_w) @ z + prior_scale * ridge * np.eye(d)
        c = z.T @ (row_w * y)
        if coef_prior is not None:
            c = c + prior_scale * ridge * coef_prior
        bounds = ([(None, None)] if fit_intercept else []) + [
            (None if np.isneginf(a) else float(a), None if np.isposinf(b) else float(b))
            for a, b in zip(lo, hi, strict=True)
        ]
        off = 1 if fit_intercept else 0
        cons = []
        if coef_sum is not None:
            s_target = float(coef_sum)
            a_eq = np.zeros(d)
            a_eq[off:] = 1.0
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda v, a=a_eq, s=s_target: float(a @ v - s),
                    "jac": lambda _v, a=a_eq: a,
                }
            )
        x0 = np.zeros(d)
        if coef_sum is not None:
            x0[off:] = coef_sum / k
        x0[off:] = np.clip(x0[off:], lo, hi)
        res = minimize(
            lambda v: (0.5 * float(v @ (h @ v)) - float(c @ v), h @ v - c),
            x0,
            method="SLSQP",
            jac=True,
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        assert res.success, f"SLSQP failed: {res.message}"
        return res.x, float(res.fun)

    if fit_intercept:
        mx = w_norm @ x
        my = float(w_norm @ y)
        xc = x - mx
        yc = y - my
        a_mat = (xc.T * w_norm) @ xc
        rhs = xc.T @ (w_norm * yc)
    else:
        mx = np.zeros(k)
        my = 0.0
        a_mat = (x.T * w_norm) @ x
        rhs = x.T @ (w_norm * y)

    diag = np.diag(a_mat)
    pen = np.where(diag > 0.0, ridge * diag, 0.0) if standardize else np.full(k, ridge)
    h = a_mat + np.diag(pen)
    c = rhs.copy()
    if coef_prior is not None:
        slope_prior = coef_prior[1:] if fit_intercept else coef_prior
        c = c + pen * slope_prior

    # Equilibrate for SciPy SLSQP numerical stability on extreme feature scales.
    scale = np.sqrt(np.maximum(np.diag(h), 0.0))
    scale = np.where(scale > 0.0, scale, 1.0)
    h_s = h / np.outer(scale, scale)
    c_s = c / scale
    bounds_s = [
        (
            None if np.isneginf(a) else float(a * sc),
            None if np.isposinf(b) else float(b * sc),
        )
        for a, b, sc in zip(lo, hi, scale, strict=True)
    ]
    cons_s = []
    if coef_sum is not None:
        s_target = float(coef_sum)
        a_eq = 1.0 / scale
        cons_s.append(
            {
                "type": "eq",
                "fun": lambda u, a=a_eq, s=s_target: float(a @ u - s),
                "jac": lambda _u, a=a_eq: a,
            }
        )

    # Feasible initial guess in original units, then scaled.
    b0_init = np.zeros(k)
    if coef_sum is not None:
        b0_init[:] = coef_sum / k
    b0_init = np.clip(b0_init, lo, hi)
    u0 = b0_init * scale

    # Normalize objective scale so SLSQP's Fortran line search stays well-conditioned.
    obj_norm = max(
        1.0, float(np.linalg.norm(c_s, ord=np.inf)), float(np.linalg.norm(h_s, ord=np.inf))
    )
    h_n = h_s / obj_norm
    c_n = c_s / obj_norm
    res = minimize(
        lambda u: (0.5 * float(u @ (h_n @ u)) - float(c_n @ u), h_n @ u - c_n),
        u0,
        method="SLSQP",
        jac=True,
        bounds=bounds_s,
        constraints=cons_s,
        options={"ftol": 1e-13, "maxiter": 1500},
    )
    if not res.success:
        tc_cons = (
            [LinearConstraint(np.asarray(1.0 / scale)[None, :], float(coef_sum), float(coef_sum))]
            if coef_sum is not None
            else ()
        )
        tc_bounds = Bounds(
            np.where(np.isneginf(lo), -np.inf, lo * scale),
            np.where(np.isposinf(hi), np.inf, hi * scale),
        )
        res = minimize(
            lambda u: (0.5 * float(u @ (h_n @ u)) - float(c_n @ u), h_n @ u - c_n),
            u0,
            method="trust-constr",
            jac=True,
            hess=lambda _u: h_n,
            bounds=tc_bounds,
            constraints=tc_cons,
            options={"gtol": 1e-12, "xtol": 1e-13, "maxiter": 2000},
        )
    assert res.success, f"SciPy QP failed: {res.message}"
    slopes = np.clip(res.x / scale, lo, hi)
    obj = 0.5 * float(slopes @ (h @ slopes)) - float(c @ slopes)
    if fit_intercept:
        b0 = my - float(mx @ slopes)
        return np.concatenate([[b0], slopes]), obj
    return slopes, obj


def _kkt_error(
    h: np.ndarray,
    c: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    coef_sum: float | None,
    x: np.ndarray,
) -> float:
    """Exact KKT optimality residual for ``min 0.5 x' H x - c' x``."""
    g = h @ x - c
    bound_err = float(
        max(
            0.0,
            np.max(np.where(np.isfinite(lo), lo - x, 0.0)),
            np.max(np.where(np.isfinite(hi), x - hi, 0.0)),
        )
    )
    tol_b = 1e-8 * (1.0 + np.abs(x))
    at_lo = np.isfinite(lo) & (x <= lo + tol_b)
    at_hi = np.isfinite(hi) & (x >= hi - tol_b)
    interior = ~at_lo & ~at_hi

    if coef_sum is None:
        eq_err = 0.0
        mu = 0.0
    else:
        eq_err = abs(float(np.sum(x)) - coef_sum)
        if np.any(interior):
            mu = float(np.mean(g[interior]))
        else:
            mu_hi = float(np.min(g[at_lo])) if np.any(at_lo) else math.inf
            mu_lo = float(np.max(g[at_hi])) if np.any(at_hi) else -math.inf
            if math.isfinite(mu_lo) and math.isfinite(mu_hi):
                mu = 0.5 * (mu_lo + mu_hi)
            elif math.isfinite(mu_lo):
                mu = mu_lo
            elif math.isfinite(mu_hi):
                mu = mu_hi
            else:
                mu = 0.0

    r = g - mu
    stat_err = float(np.max(np.abs(r[interior]))) if np.any(interior) else 0.0
    dual_lo = float(np.max(np.maximum(-r[at_lo], 0.0))) if np.any(at_lo) else 0.0
    dual_hi = float(np.max(np.maximum(r[at_hi], 0.0))) if np.any(at_hi) else 0.0
    scale = max(1.0, float(np.linalg.norm(h, ord=np.inf)), float(np.linalg.norm(c, ord=np.inf)))
    return max(bound_err, eq_err, stat_err / scale, dual_lo / scale, dual_hi / scale)


def _seed_problem(seed: int) -> dict[str, Any]:
    """Deterministic problem specification for one seed of the 50-seed oracle grid."""
    rng = np.random.default_rng(10_000 + seed)
    k = [3, 4, 5, 6][seed % 4]
    n = 60 + (seed % 5) * 15
    fit_intercept = (seed % 2) == 0
    standardize = ((seed // 2) % 2) == 1
    half_life = math.inf if (seed % 3) == 0 else 25.0
    ridge = [0.0, 1e-4, 0.05, 0.4][seed % 4]

    scales = np.power(10.0, rng.uniform(-2.0, 2.0, size=k))
    means = rng.uniform(-5.0, 5.0, size=k)
    a_gen = rng.standard_normal((k, k))
    x = (rng.standard_normal((n, k)) @ a_gen / math.sqrt(k)) * scales + means
    beta_true = rng.uniform(-1.5, 1.5, size=k) / scales
    y = (1.5 if fit_intercept else 0.0) + x @ beta_true + 0.25 * rng.standard_normal(n)
    times = np.cumsum(rng.uniform(0.5, 2.0, size=n))
    weights = rng.uniform(0.3, 3.0, size=n)

    regime = seed % 5
    coef_min: float | list[float] | None = None
    coef_max: float | list[float] | None = None
    coef_sum: float | None = None
    if regime == 0:
        coef_min = 0.0
        coef_sum = 1.0
    elif regime == 1:
        coef_min = 0.0
    elif regime == 2:
        coef_min = -0.5
        coef_max = 0.5
    elif regime == 3:
        coef_min = [0.0 if i % 2 == 0 else -0.1 for i in range(k)]
        coef_max = [round(1.5 / k + 0.15 * (i % 2), 4) for i in range(k)]
        coef_sum = 1.0
    else:
        coef_max = 0.35

    return {
        "k": k,
        "n": n,
        "fit_intercept": fit_intercept,
        "standardize": standardize,
        "half_life": half_life,
        "ridge": ridge,
        "x": x,
        "y": y,
        "times": times,
        "weights": weights,
        "regime": regime,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "coef_sum": coef_sum,
    }


def _grid_rheon_outputs() -> list[tuple[list[float], list[float]]]:
    """Run rheon's bounded ewridge and gram.solve on all 50 seeds (for oracle ledger guard)."""
    out: list[tuple[list[float], list[float]]] = []
    for seed in GRID_SEEDS:
        p = _seed_problem(seed)
        k = p["k"]
        feat_names = [f"x{i}" for i in range(k)]
        df = pl.DataFrame(
            {
                "t": p["times"],
                "w": p["weights"],
                "y": p["y"],
                **{feat_names[i]: p["x"][:, i] for i in range(k)},
            }
        )
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=feat_names,
            clock="t",
            weight="w",
            half_life=p["half_life"],
            gap_cap=1e6,
            ridge=p["ridge"],
            standardize=p["standardize"],
            fit_intercept=p["fit_intercept"],
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_min=p["coef_min"],
            coef_max=p["coef_max"],
            coef_sum=p["coef_sum"],
        )
        bank = rp.ModelBank([spec])
        bank.fit(df)
        ew_beta = bank.coef("m")["coef"].to_list()
        g = bank.gram("m")[0]
        g_beta = rg.solve(
            g,
            ridge=p["ridge"],
            target="y",
            standardize=p["standardize"],
            coef_min=p["coef_min"],
            coef_max=p["coef_max"],
            coef_sum=p["coef_sum"],
        ).tolist()
        out.append((ew_beta, g_beta))
    return out


ledger.guard(
    "ewridge_bounds_oracle",
    _grid_rheon_outputs,
    oracles=["scipy"],
    bits=52,
)


@pytest.mark.oracle_grid(
    reason="50 seeds comparing bounded/fixed-sum ewridge and gram.solve to SciPy SLSQP and KKT"
)
def test_ewridge_bounds_50_seed_scipy_oracle() -> None:
    """50-seed third-party SciPy SLSQP oracle + KKT + gram.solve bit-identity."""
    rheon_vals = _grid_rheon_outputs()
    kkt_errs: list[float] = []
    obj_gaps: list[float] = []
    coef_rel_errs: list[float] = []

    for seed, (ew_list, g_list) in zip(GRID_SEEDS, rheon_vals, strict=True):
        p = _seed_problem(seed)
        k = p["k"]
        x = p["x"]
        y = p["y"]
        times = p["times"]
        weights = p["weights"]
        fit_intercept = p["fit_intercept"]
        standardize = p["standardize"]
        half_life = p["half_life"]
        ridge = p["ridge"]
        coef_min = p["coef_min"]
        coef_max = p["coef_max"]
        coef_sum = p["coef_sum"]

        got = np.asarray(ew_list, dtype=np.float64)
        from_gram = np.asarray(g_list, dtype=np.float64)

        # 1. Exact bit-identity with rheon.gram.solve
        assert np.array_equal(got.view(np.uint64), from_gram.view(np.uint64)), (
            f"seed {seed}: ewridge vs gram.solve bit mismatch: {got} vs {from_gram}"
        )

        # 2. Independent SciPy SLSQP oracle directly from raw rows
        row_w = _decay_weights(times, weights, half_life)
        ref, ref_obj = _scipy_ridge_qp(
            x,
            y,
            row_w,
            ridge=ridge,
            standardize=standardize,
            fit_intercept=fit_intercept,
            coef_min=coef_min,
            coef_max=coef_max,
            coef_sum=coef_sum,
        )

        off = 1 if fit_intercept else 0
        got_slopes = got[off:]
        w_norm = row_w / float(np.sum(row_w))
        if fit_intercept:
            xc = x - (w_norm @ x)
            yc = y - float(w_norm @ y)
            a_mat = (xc.T * w_norm) @ xc
            rhs = xc.T @ (w_norm * yc)
        else:
            a_mat = (x.T * w_norm) @ x
            rhs = x.T @ (w_norm * y)
        pen = np.where(np.diag(a_mat) > 0.0, ridge * np.diag(a_mat), 0.0) if standardize else ridge
        h = a_mat + np.diag(pen if isinstance(pen, np.ndarray) else np.full(k, pen))

        lo_arr = (
            np.full(k, -np.inf)
            if coef_min is None
            else (np.full(k, float(coef_min)) if np.isscalar(coef_min) else np.asarray(coef_min))
        )
        hi_arr = (
            np.full(k, np.inf)
            if coef_max is None
            else (np.full(k, float(coef_max)) if np.isscalar(coef_max) else np.asarray(coef_max))
        )
        kkt = _kkt_error(h, rhs, lo_arr, hi_arr, coef_sum, got_slopes)
        got_obj = 0.5 * float(got_slopes @ (h @ got_slopes)) - float(rhs @ got_slopes)
        obj_scale = max(1.0, abs(ref_obj), abs(got_obj))
        obj_gap = max(0.0, got_obj - ref_obj) / obj_scale
        coef_scale = max(1.0, float(np.linalg.norm(ref)))
        coef_rel = float(np.linalg.norm(got - ref)) / coef_scale

        kkt_errs.append(kkt)
        obj_gaps.append(obj_gap)
        coef_rel_errs.append(coef_rel)

        assert kkt < 1e-8, f"seed {seed} (regime={p['regime']}): KKT error {kkt}"
        assert obj_gap < 1e-9, f"seed {seed}: got_obj={got_obj} > ref_obj={ref_obj}"
        assert coef_rel < 1e-4, f"seed {seed}: coef rel err {coef_rel} ({got} vs {ref})"

    ledger.record("ewridge_bounds_oracle", "kkt_err", kkt_errs, floor=1e-10)
    ledger.record("ewridge_bounds_oracle", "obj_gap", obj_gaps, floor=1e-12)
    ledger.record("ewridge_bounds_oracle", "coef_rel_err", coef_rel_errs, floor=1e-6)


def test_ewridge_bounds_unconstrained_bit_identical() -> None:
    """Unconstrained and trivial-bound ewridge stay bit-identical and keep LS uncertainty."""
    rng = np.random.default_rng(7)
    n = 40
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    s_plain = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=20.0,
        emit_se_coef=True,
        emit_robust_se=True,
        emit_error_inflation=True,
    )
    s_triv = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=20.0,
        coef_min=-math.inf,
        coef_max=math.inf,
        emit_se_coef=True,
        emit_robust_se=True,
        emit_error_inflation=True,
    )
    assert "coef_min" not in s_plain["model"]
    assert "coef_max" not in s_plain["model"]
    assert "coef_sum" not in s_plain["model"]

    b1 = rp.ModelBank([s_plain])
    b2 = rp.ModelBank([s_triv])
    assert b1.specs[0] == s_plain
    out1 = b1.fit_predict(df)
    out2 = b2.fit_predict(df)
    assert out1.equals(out2)
    assert "support_coef" in rp.spec.output_fields(s_plain)


def test_ewridge_bounds_readiness_and_refusals() -> None:
    """Bounded ewridge uses min_weight readiness, omits support_coef, and refuses LS SEs."""
    rng = np.random.default_rng(11)
    n = 20
    df = pl.DataFrame(
        {
            "x0": rng.standard_normal(n),
            "x1": rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=50.0,
        coef_min=0.0,
        coef_sum=1.0,
        coef_every=0.0,
    )
    assert "support_coef" not in rp.spec.output_fields(spec)

    bank = rp.ModelBank([spec])
    assert bank.specs[0] == spec
    out = bank.fit_predict(df)
    reasons = out["m"].struct.field("withheld_reason").to_list()
    # k=2 + intercept=1 -> default min_weight = 3.0; under half_life=50, decayed weight_sum
    # crosses 3.0 after 4 rows (rows 0..3 withheld as below_min_weight, row 4 ready).
    assert reasons[:4] == ["below_min_weight"] * 4
    assert reasons[4] is None
    assert "support_coef" not in out["m"].struct.fields

    # max_error_inflation=inf is accepted on bounded ewridge.
    spec_inf = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=50.0,
        coef_min=0.0,
        max_error_inflation=math.inf,
    )
    rp.ModelBank([spec_inf])

    # Refusals on bounded ewridge
    base: dict[str, Any] = dict(
        targets=["y"],
        features=["x0", "x1"],
        half_life=50.0,
        coef_min=0.0,
        coef_sum=1.0,
    )
    for bad_kw, expected_word in [
        ({"emit_se_coef": True}, "emit_se_coef"),
        ({"emit_robust_se": True}, "emit_robust_se"),
        ({"emit_error_inflation": True}, "emit_error_inflation"),
        ({"emit_influence": True}, "emit_influence"),
        ({"max_error_inflation": 2.0}, "max_error_inflation"),
        ({"feature_sets": {"a": ["x0"]}}, "feature_sets"),
        ({"constraints": ([[1.0, 1.0]], [1.0])}, "constraints"),
        ({"coef_max": 0.4}, "coef_sum"),
        ({"coef_min": 0.6, "coef_max": 0.4}, "coef_min"),
        ({"coef_min": [0.0]}, "coef_min"),
        ({"coef_sum": math.inf}, "coef_sum"),
    ]:
        with pytest.raises(ValueError, match=expected_word):
            rp.ModelBank([rp.spec.ewridge("bad", **{**base, **bad_kw})])


def test_ewridge_bounds_collinear_and_constant_features_zero_ridge() -> None:
    """PSD / rank-deficient H at ridge=0 (collinear + constant features) solves cleanly."""
    rng = np.random.default_rng(19)
    n = 80
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    df = pl.DataFrame(
        {
            "x0": x0,
            "x1": x1,
            "x_dup": x0,
            "x_const": np.full(n, 3.5),
            "y": 0.5 + 0.8 * x0 + 0.2 * x1 + 0.05 * rng.standard_normal(n),
        }
    )
    for standardize in (False, True):
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x_dup", "x_const"],
            half_life=math.inf,
            ridge=0.0,
            standardize=standardize,
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_min=0.0,
            coef_sum=1.0,
        )
        bank = rp.ModelBank([spec])
        bank.fit(df)
        assert bank.solve_failures().filter(pl.col("solve_failures") > 0).height == 0
        got = bank.coef("m")["coef"].to_numpy()
        assert np.all(np.isfinite(got))
        slopes = got[1:]
        assert np.all(slopes >= -1e-10)
        assert abs(float(np.sum(slopes)) - 1.0) < 1e-10

        x_mat = df.select(["x0", "x1", "x_dup", "x_const"]).to_numpy()
        y_vec = df["y"].to_numpy()
        _, ref_obj = _scipy_ridge_qp(
            x_mat,
            y_vec,
            np.ones(n),
            ridge=0.0,
            standardize=standardize,
            fit_intercept=True,
            coef_min=0.0,
            coef_sum=1.0,
        )
        xc = x_mat - x_mat.mean(axis=0)
        yc = y_vec - y_vec.mean()
        a_mat = (xc.T @ xc) / n
        rhs = (xc.T @ yc) / n
        got_obj = 0.5 * float(slopes @ (a_mat @ slopes)) - float(rhs @ slopes)
        assert got_obj <= ref_obj + 1e-10


def test_ewridge_bounds_prior_window_multi_target_embargo_label_clock_zero_weight() -> None:
    """Bounded ewridge works with prior, window, grid, embargo, label_clock, and Rule 9."""
    rng = np.random.default_rng(29)
    n = 70
    x = rng.standard_normal((n, 3))
    y0 = 0.4 + 0.9 * x[:, 0] - 0.3 * x[:, 1] + 0.5 * x[:, 2] + 0.1 * rng.standard_normal(n)
    y1 = -0.2 - 0.4 * x[:, 0] + 0.8 * x[:, 1] + 0.6 * x[:, 2] + 0.1 * rng.standard_normal(n)
    t = np.arange(n, dtype=float)
    w = rng.uniform(0.5, 2.0, size=n)

    # 1. coef_prior on both ridge_scale="mean" and ridge_scale="sum" vs SciPy SLSQP
    prior = [[0.2, 0.5, 0.3, 0.2]]
    df1 = pl.DataFrame({"t": t, "w": w, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y0": y0})
    for rscale in ("mean", "sum"):
        spec = rp.spec.ewridge(
            "m",
            targets=["y0"],
            features=["x0", "x1", "x2"],
            clock="t",
            gap_cap=1000.0,
            weight="w",
            half_life=math.inf,
            ridge=0.5,
            ridge_scale=rscale,
            coef_prior=prior,
            solve_every=0.0,
            max_rows_between_solves=1,
            coef_min=0.0,
            coef_max=0.7,
            coef_sum=1.0,
        )
        bank = rp.ModelBank([spec])
        bank.fit(df1)
        got = bank.coef("m")["coef"].to_numpy()
        ref, _ = _scipy_ridge_qp(
            x,
            y0,
            w,
            ridge=0.5,
            standardize=False,
            fit_intercept=True,
            ridge_scale=rscale,
            coef_prior=np.asarray(prior[0]),
            prior_scale=1.0,
            coef_min=0.0,
            coef_max=0.7,
            coef_sum=1.0,
        )
        assert np.allclose(got, ref, atol=1e-6), f"ridge_scale={rscale}: {got} vs {ref}"

    # 2. Hard window + ridge grid + multi-target + embargo + label_clock + Rule 9 zero-weight rows
    df_multi = pl.DataFrame(
        {
            "t": t,
            "avail": t + 2.0,
            "w": w,
            "x0": x[:, 0],
            "x1": x[:, 1],
            "x2": x[:, 2],
            "y0": y0,
            "y1": y1,
        }
    )
    spec_win = rp.spec.ewridge(
        "win",
        targets=["y0", "y1"],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=1000.0,
        weight="w",
        half_life=20.0,
        window_size=25.0,
        ridge=[0.01, 0.5],
        standardize=True,
        embargo=2.0,
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_min=0.0,
        coef_sum=1.0,
    )
    spec_lc = rp.spec.ewridge(
        "lc",
        targets=["y0", "y1"],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=1000.0,
        label_clock="avail",
        weight="w",
        half_life=20.0,
        ridge=[0.01, 0.5],
        solve_every=0.0,
        max_rows_between_solves=1,
        coef_min=-0.2,
        coef_max=0.8,
        coef_sum=1.0,
    )
    b_clean = rp.ModelBank([spec_win, spec_lc])
    out_clean = b_clean.fit_predict(df_multi)

    # Interleave zero-weight poison rows (Rule 9) and confirm identical predictions on real rows.
    rows_poison = []
    for i in range(n):
        rows_poison.append(
            {
                "t": float(t[i]),
                "avail": float(t[i] + 2.0),
                "w": float(w[i]),
                "x0": float(x[i, 0]),
                "x1": float(x[i, 1]),
                "x2": float(x[i, 2]),
                "y0": float(y0[i]),
                "y1": float(y1[i]),
                "real": True,
            }
        )
        if i % 7 == 3:
            rows_poison.append(
                {
                    "t": float(t[i]),
                    "avail": float(t[i] + 2.0),
                    "w": 0.0,
                    "x0": 1e5,
                    "x1": -1e5,
                    "x2": 1e5,
                    "y0": 1e8,
                    "y1": -1e8,
                    "real": False,
                }
            )
    df_poison = pl.DataFrame(rows_poison)
    b_poison = rp.ModelBank([spec_win, spec_lc])
    out_poison = b_poison.fit_predict(df_poison).filter(pl.col("real")).drop("real")
    assert out_clean["win"].equals(out_poison["win"])
    assert out_clean["lc"].equals(out_poison["lc"])


def test_ewridge_bounds_chunk_and_save_load_invariance() -> None:
    """Chunking and save/load preserve bounded ewridge outputs to the bit."""
    rng = np.random.default_rng(37)
    n = 90
    df = pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "x0": rng.standard_normal(n),
            "x1": 2.0 * rng.standard_normal(n),
            "x2": 0.5 * rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        clock="t",
        gap_cap=1000.0,
        half_life=25.0,
        ridge=0.05,
        standardize=True,
        coef_min=0.0,
        coef_max=[0.7, 0.8, 0.9],
        coef_sum=1.0,
        coef_every=0.0,
    )
    full = rp.ModelBank([spec]).fit_predict(df)

    b_chunk = rp.ModelBank([spec])
    parts = []
    splits = [0, 7, 23, 51, 68, n]
    for idx, (a, b) in enumerate(zip(splits[:-1], splits[1:], strict=True)):
        parts.append(b_chunk.fit_predict(df.slice(a, b - a)))
        if idx == 2:
            raw = b_chunk.save_bytes()
            b_chunk = rp.ModelBank.load_bytes(raw)
    chunked = pl.concat(parts)
    assert full.equals(chunked)


def test_gram_solve_bounds_and_refusals() -> None:
    """rheon.gram.solve handles ridge grids, subsets, within pooling, and refuses conflicts."""
    rng = np.random.default_rng(53)
    n = 120
    df = pl.DataFrame(
        {
            "g": ["a"] * (n // 2) + ["b"] * (n // 2),
            "x0": rng.standard_normal(n),
            "x1": 3.0 * rng.standard_normal(n),
            "x2": 0.4 * rng.standard_normal(n),
            "y": rng.standard_normal(n),
        }
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        group="g",
        half_life=math.inf,
    )
    bank = rp.ModelBank([spec])
    bank.fit(df)
    gs = bank.gram("m")

    # 1. Ridge grid + feature subset
    g0 = gs[0]
    grid = rg.solve(
        g0,
        ridge=[0.0, 0.05, 0.5],
        target="y",
        features=["x0", "x2"],
        standardize=True,
        coef_min=0.0,
        coef_sum=1.0,
    )
    assert grid.shape == (3, 4)
    # Column x1 (index 2) is omitted -> 0.0; active slopes x0 (index 1) + x2 (index 3) sum to 1.0
    assert np.all(grid[:, 2] == 0.0)
    assert np.all(grid[:, [1, 3]] >= -1e-12)
    assert np.allclose(grid[:, 1] + grid[:, 3], 1.0, atol=1e-11)
    # Scalar solve is bit-identical to the matching grid row
    single = rg.solve(
        g0,
        ridge=0.05,
        target="y",
        features=["x0", "x2"],
        standardize=True,
        coef_min=0.0,
        coef_sum=1.0,
    )
    assert np.array_equal(single.view(np.uint64), grid[1].view(np.uint64))

    # 2. Within-pooled Gram: intercept is NaN, slopes match demeaned-by-group SciPy oracle
    gw = rg.merge(gs, within=True)
    got_w = rg.solve(gw, ridge=0.1, target="y", coef_min=0.0, coef_sum=1.0)
    assert math.isnan(got_w[0])
    # Demean each group manually and compare with SciPy SLSQP through origin
    x_dm = np.vstack(
        [
            df.filter(pl.col("g") == grp).select(["x0", "x1", "x2"]).to_numpy()
            - df.filter(pl.col("g") == grp).select(["x0", "x1", "x2"]).to_numpy().mean(axis=0)
            for grp in ("a", "b")
        ]
    )
    y_dm = np.concatenate(
        [
            df.filter(pl.col("g") == grp)["y"].to_numpy()
            - df.filter(pl.col("g") == grp)["y"].to_numpy().mean()
            for grp in ("a", "b")
        ]
    )
    ref_w, _ = _scipy_ridge_qp(
        x_dm,
        y_dm,
        np.ones(n),
        ridge=0.1,
        standardize=False,
        fit_intercept=False,
        coef_min=0.0,
        coef_sum=1.0,
    )
    assert np.allclose(got_w[1:], ref_w, atol=1e-6)

    # 3. Refusals
    with pytest.raises(ValueError, match="constraints"):
        rg.solve(
            g0,
            constraints=([[1.0, 1.0, 1.0]], [1.0]),
            coef_min=0.0,
        )
    with pytest.raises(ValueError, match="coef_sum"):
        rg.solve(g0, coef_min=0.0, coef_max=0.2, coef_sum=1.0)
    with pytest.raises(ValueError, match="coef_min"):
        rg.solve(g0, coef_min=[0.0, 0.0])
    with pytest.raises(TypeError, match="coef_min"):
        rg.solve(g0, coef_min="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="coef_sum"):
        rg.solve(g0, coef_sum=True)  # type: ignore[arg-type]
    assert hasattr(rheon, "ReadinessWarning")
