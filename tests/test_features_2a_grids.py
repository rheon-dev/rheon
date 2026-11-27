"""Fifty-seed oracle grids for ``ew_cov`` raw_cov/loglik, exact rolling window
operators, extended specification/influence diagnostics, and forecast comparison
tests.

Each group registers its outputs with ``ledger.guard`` and records its 50-seed
worst and 95th-percentile oracle errors in ``oracles.toml``.
"""

from __future__ import annotations

import mpmath
import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from dieboldmariano import dm_test
from mapie.exchangeability_testing.bounds import conjugate_mixture_empirical_bernstein_bound
from mapie.metrics.calibration import spiegelhalter_statistic
from scipy import special as sp_special
from scipy import stats as sp_stats
from sklearn.covariance import EmpiricalCovariance
from statsmodels.stats.diagnostic import het_arch, het_white
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.sandwich_covariance import S_hac_simple, weights_bartlett

import ledger
from rheon import _rheon as _native
from rheon import polars as rp

TIER = "oracle_grid"

pytestmark = pytest.mark.oracle_grid(reason="50-seed oracle grids against third-party references")

SEEDS = range(50)
GUARD_SEEDS = range(3)


# --- 1. ew_cov raw_cov and loglik vs scikit-learn & scipy --------------------


def _ewcov_frame(seed: int, n: int = 160, k: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(517_000 + seed)
    a = rng.normal(size=(k, k))
    cov = a @ a.T / k + 0.5 * np.eye(k)
    x = rng.normal(size=(n, k)) @ np.linalg.cholesky(cov).T + rng.uniform(-1.0, 2.0, k)
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def _ewcov_run(seed: int) -> pl.DataFrame:
    spec = rp.spec.ew_cov(
        "m",
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        stats=["raw_cov", "loglik"],
        precision_prior=0.05,
    )
    return rp.ModelBank([spec]).fit_predict(_ewcov_frame(seed))["m"].struct.unnest()


ledger.guard(
    "ew_cov_raw_loglik",
    lambda: [_ewcov_run(s) for s in GUARD_SEEDS],
    oracles=["scikit-learn", "scipy"],
)


def test_ew_cov_raw_cov_and_loglik_fifty_seeds():
    raw_err: list[float] = []
    ll_err: list[float] = []
    prior = 0.05
    for seed in SEEDS:
        df = _ewcov_frame(seed)
        x = df.to_numpy()
        out = _ewcov_run(seed)
        t = df.height - 1
        want_raw = EmpiricalCovariance(assume_centered=True).fit(x[:t]).covariance_
        r_gap = 0.0
        for i in range(3):
            for j in range(i, 3):
                got = out[f"raw_cov_x{i}_x{j}"][t]
                r_gap = max(r_gap, abs(got - want_raw[i, j]) / max(1.0, abs(want_raw[i, j])))
        raw_err.append(r_gap)
        mu = x[:t].mean(axis=0)
        c_factor = t / (t - 1.0)
        cov = np.cov(x[:t], rowvar=False, ddof=1) + (c_factor * prior / t) * np.eye(3)
        want_ll = float(sp_stats.multivariate_normal.logpdf(x[t], mean=mu, cov=cov))
        ll_err.append(abs(out["loglik"][t] - want_ll) / max(1.0, abs(want_ll)))
    ledger.record("ew_cov_raw_loglik", "raw_cov_rel_err", raw_err, floor=1e-12)
    ledger.record("ew_cov_raw_loglik", "loglik_rel_err", ll_err, floor=1e-12)
    assert max(raw_err) < 1e-11
    assert max(ll_err) < 1e-11


# --- 2. exact window operators vs Polars -------------------------------------


def _win_frame(seed: int, n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(518_000 + seed)
    t = np.cumsum(rng.integers(0, 4, size=n)).astype(np.int64)
    x = np.round(rng.normal(size=n), 2)
    return pl.DataFrame({"t": t, "x": x})


def _win_ops_run(seed: int) -> pl.DataFrame:
    df = _win_frame(seed)
    ops = [
        rp.ops.rolling_quantile(
            pl.col("x"), quantile=0.35, window_size=12, interpolation=interp
        ).alias(f"q_{interp}")
        for interp in ("nearest", "higher", "lower", "midpoint", "linear", "equiprobable")
    ] + [
        rp.ops.rolling_rank(pl.col("x"), window_size=12, method=meth).alias(f"r_{meth}")
        for meth in ("average", "min", "max", "dense")
    ]
    out_clock = rp.stream.with_windows(df.lazy(), *ops, clock="t", gap_cap=100).collect()
    out_ewm = rp.stream.with_windows(
        df.select("x").lazy(),
        rp.ops.ewm_mean(pl.col("x"), half_life=15.0, adjust=True).alias("ewm"),
    ).collect()
    return out_clock.with_columns(out_ewm["ewm"])


ledger.guard(
    "window_ops_exact",
    lambda: [_win_ops_run(s) for s in GUARD_SEEDS],
    oracles=["polars"],
    bits=52,
)


def test_window_ops_exact_match_polars_fifty_seeds():
    q_err: list[float] = []
    r_err: list[float] = []
    e_err: list[float] = []
    for seed in SEEDS:
        df = _win_frame(seed)
        got = _win_ops_run(seed)
        qg = 0.0
        for interp in ("nearest", "higher", "lower", "midpoint", "linear", "equiprobable"):
            want = df.select(
                pl.col("x")
                .rolling_quantile_by("t", window_size="12i", quantile=0.35, interpolation=interp)
                .alias("w")
            )["w"]
            diff = np.nanmax(np.abs(got[f"q_{interp}"].to_numpy() - want.to_numpy()))
            qg = max(qg, float(diff))
        q_err.append(qg)

        rg = 0.0
        for meth in ("average", "min", "max", "dense"):
            want = df.select(
                pl.col("x")
                .rolling_rank_by("t", window_size="12i", method=meth)
                .cast(pl.Float64)
                .alias("w")
            )["w"]
            diff = np.nanmax(np.abs(got[f"r_{meth}"].to_numpy() - want.to_numpy()))
            rg = max(rg, float(diff))
        r_err.append(rg)

        want_e = df.select(
            pl.col("x").ewm_mean(half_life=15.0, adjust=True, ignore_nulls=False).alias("w")
        )["w"]
        e_err.append(float(np.nanmax(np.abs(got["ewm"].to_numpy() - want_e.to_numpy()))))

    ledger.record("window_ops_exact", "rolling_quantile_abs_err", q_err, floor=1e-15)
    ledger.record("window_ops_exact", "rolling_rank_abs_err", r_err, floor=1e-15)
    ledger.record("window_ops_exact", "ewm_mean_adjust_abs_err", e_err, floor=1e-14)
    assert max(q_err) == 0.0
    assert max(r_err) == 0.0
    assert max(e_err) < 1e-14


# --- 3. specification & influence diagnostics vs statsmodels & mapie ---------


def _diag_frames(seed: int, n: int = 180) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(520_000 + seed)
    x = rng.normal(size=(n, 3))
    y = 0.5 + x @ np.array([1.2, -0.7, 0.4]) + 0.6 * rng.normal(size=n)
    reg = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.7 * x[:, 0] - 0.5 * x[:, 1])))
    yb = rng.binomial(1, p).astype(float)
    clf = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": yb})
    return reg, clf


def _diag_run(seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    reg, clf = _diag_frames(seed)
    s_reg = rp.spec.ewridge(
        "r",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=float("inf"),
        ridge=0.0,
        solve_every=0.0,
        max_error_inflation=float("inf"),
        emit_arch_lm=True,
        arch_lm_lags=4,
        emit_white=True,
        specification_half_life=float("inf"),
        emit_cooks_distance=True,
        emit_dfbetas=True,
        influence_half_life=float("inf"),
    )
    s_clf = rp.spec.ftrl(
        "c",
        targets=["y"],
        features=["x0", "x1"],
        half_life=float("inf"),
        emit_spiegelhalter=True,
        calibration_half_life=float("inf"),
    )
    out_r = rp.unnest(rp.ModelBank([s_reg]).fit_predict(reg), [s_reg])
    out_c = rp.ModelBank([s_clf]).fit_predict(clf)["c"].struct.unnest()
    return out_r, out_c


ledger.guard(
    "spec_influence_ext",
    lambda: [_diag_run(s) for s in GUARD_SEEDS],
    oracles=["statsmodels", "mapie"],
)


def test_specification_and_influence_fifty_seeds():
    arch_err: list[float] = []
    white_err: list[float] = []
    spieg_err: list[float] = []
    cook_err: list[float] = []
    dfb_err: list[float] = []
    for seed in SEEDS:
        reg, clf = _diag_frames(seed)
        out_r, out_c = _diag_run(seed)
        t = reg.height - 1
        e = out_r["resid_y"].to_numpy()
        valid = np.flatnonzero(np.isfinite(e[:t]))
        x = reg.select(["x0", "x1", "x2"]).to_numpy()

        want_arch = float(het_arch(e[valid], nlags=4, ddof=0)[0])
        arch_err.append(abs(out_r["arch_lm_y"][t] - want_arch) / max(1.0, abs(want_arch)))

        want_white = float(het_white(e[valid], sm.add_constant(x[valid]))[0])
        white_err.append(abs(out_r["white_y"][t] - want_white) / max(1.0, abs(want_white)))

        preds = out_c["pred_y"].to_numpy()
        vc = np.flatnonzero(np.isfinite(preds[:t]))
        want_sp = float(spiegelhalter_statistic(clf["y"].to_numpy()[vc], preds[vc]))
        spieg_err.append(abs(out_c["spiegelhalter_y"][t] - want_sp) / max(1.0, abs(want_sp)))

        infl = OLSInfluence(sm.OLS(reg["y"].to_numpy(), sm.add_constant(x)).fit())
        want_c = float(infl.cooks_distance[0][t])
        cook_err.append(abs(out_r["cooks_distance_y"][t] - want_c) / max(1e-12, abs(want_c)))
        got_d = np.array(
            [
                out_r["dfbetas_y_intercept"][t],
                out_r["dfbetas_y_x0"][t],
                out_r["dfbetas_y_x1"][t],
                out_r["dfbetas_y_x2"][t],
            ]
        )
        want_d = infl.dfbetas[t]
        dfb_err.append(float(np.max(np.abs(got_d - want_d) / np.maximum(1.0, np.abs(want_d)))))

    ledger.record("spec_influence_ext", "arch_lm_rel_err", arch_err, floor=1e-10)
    ledger.record("spec_influence_ext", "white_rel_err", white_err, floor=1e-10)
    ledger.record("spec_influence_ext", "spiegelhalter_rel_err", spieg_err, floor=1e-10)
    ledger.record("spec_influence_ext", "cooks_distance_rel_err", cook_err, floor=1e-9)
    ledger.record("spec_influence_ext", "dfbetas_rel_err", dfb_err, floor=1e-9)
    assert max(arch_err) < 1e-8
    assert max(white_err) < 1e-8
    assert max(spieg_err) < 1e-9
    assert max(cook_err) < 1e-8
    assert max(dfb_err) < 1e-8


# --- 4. forecast comparison & special functions vs mpmath/scipy/dm/sm/mapie --


def _cmp_frame(seed: int, n: int = 160) -> pl.DataFrame:
    rng = np.random.default_rng(521_000 + seed)
    ra = 0.9 * rng.normal(size=n)
    rb = 1.05 * rng.normal(size=n) + 0.05
    d_bnd = np.clip(0.15 + 0.35 * rng.normal(size=n), -1.0, 1.0)
    return pl.DataFrame(
        {
            "a": pl.DataFrame({"resid_y": ra}).select(pl.struct(pl.all()).alias("a"))["a"],
            "b": pl.DataFrame({"resid_y": rb}).select(pl.struct(pl.all()).alias("b"))["b"],
            "d_bnd": d_bnd,
        }
    )


def _cmp_run(seed: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = _cmp_frame(seed)
    dm = rp.eval.diebold_mariano(
        df, a="a", b="b", loss="pinball", quantile=0.75, lags=2, hln=True, reference="t"
    )
    gw = rp.eval.giacomini_white(df, a="a", b="b", lags=3, instrument_lags=1)
    ms = rp.eval.mean_seqtest(
        df,
        targets=["d_bnd"],
        bound=1.0,
        method="cm_eb",
        alpha=0.05,
        v_opt=1.0,
        running_intersection=True,
    )["mean_seqtest"].struct.unnest()
    return dm, gw, ms


ledger.guard(
    "forecast_comparison",
    lambda: [_cmp_run(s) for s in GUARD_SEEDS],
    oracles=["mpmath", "scipy", "statsmodels", "dieboldmariano", "mapie"],
)


def test_forecast_comparison_and_special_fifty_seeds():
    mpmath.mp.dps = 40
    dm_err: list[float] = []
    gw_err: list[float] = []
    cmeb_err: list[float] = []
    spec_err: list[float] = []
    for seed in SEEDS:
        df = _cmp_frame(seed)
        dm, gw, ms = _cmp_run(seed)
        ra = df["a"].struct.field("resid_y").to_numpy()
        rb = df["b"].struct.field("resid_y").to_numpy()
        z = np.zeros_like(ra)

        def pinball(u: float, v: float) -> float:
            e = u - v
            return 0.75 * e if e >= 0.0 else -0.25 * e

        rst, _rp_val = dm_test(
            z, -ra, -rb, loss=pinball, h=3, harvey_correction=True, variance_estimator="bartlett"
        )
        want_dmp = float(2.0 * sp_stats.t.sf(abs(rst), df=len(ra) - 1))
        dm_err.append(
            max(
                abs(dm["dm"][0] + rst) / max(1.0, abs(rst)),
                abs(dm["p_value"][0] - want_dmp),
            )
        )

        d = rb * rb - ra * ra
        y = d[1:]
        H = np.column_stack([np.ones(len(y)), d[:-1]])
        Z = H * y[:, None]
        S = S_hac_simple(Z, nlags=3, weights_func=weights_bartlett)
        zs = Z.sum(axis=0)
        want_gw = float(zs @ np.linalg.solve(S, zs))
        want_gwp = float(sp_stats.chi2.sf(want_gw, df=2))
        gw_err.append(
            max(
                abs(gw["gw"][0] - want_gw) / max(1.0, abs(want_gw)),
                abs(gw["p_value"][0] - want_gwp),
            )
        )

        r = (df["d_bnd"].to_numpy() + 1.0) * 0.5
        want_lo = (
            2.0
            * conjugate_mixture_empirical_bernstein_bound(
                r, v_opt=1.0, alpha=0.05, bound_side="lower", running_intersection=True
            )
            - 1.0
        )
        want_hi = (
            2.0
            * conjugate_mixture_empirical_bernstein_bound(
                r, v_opt=1.0, alpha=0.05, bound_side="upper", running_intersection=True
            )
            - 1.0
        )
        cmeb_err.append(
            float(
                max(
                    np.max(np.abs(ms["lower_d_bnd"].to_numpy() - want_lo)),
                    np.max(np.abs(ms["upper_d_bnd"].to_numpy() - want_hi)),
                )
            )
        )

        # Spot-check special functions on seed-varied points vs mpmath & scipy.
        rng = np.random.default_rng(999_000 + seed)
        t_val = float(rng.uniform(-6.0, 6.0))
        df_t = float(rng.uniform(1.0, 200.0))
        x_chi = float(rng.uniform(0.1, 40.0))
        df_c = float(rng.integers(1, 25))
        e_t = abs(_native.special_student_t_sf(t_val, df_t) - float(sp_stats.t.sf(t_val, df_t)))
        e_c = abs(_native.special_chi2_sf(x_chi, df_c) - float(sp_stats.chi2.sf(x_chi, df_c)))
        a_b, b_b, x_b = (
            float(rng.uniform(0.5, 40.0)),
            float(rng.uniform(0.5, 10.0)),
            float(rng.uniform(0.05, 0.95)),
        )
        e_b = abs(
            _native.special_betainc(a_b, b_b, x_b)
            - float(mpmath.betainc(a_b, b_b, 0, x_b, regularized=True))
        )
        e_g = abs(
            _native.special_gammaincc(df_c * 0.5, x_chi * 0.5)
            - float(sp_special.gammaincc(df_c * 0.5, x_chi * 0.5))
        )
        spec_err.append(max(e_t, e_c, e_b, e_g))

    ledger.record("forecast_comparison", "dm_max_err", dm_err, floor=1e-11)
    ledger.record("forecast_comparison", "gw_max_err", gw_err, floor=1e-11)
    ledger.record("forecast_comparison", "cm_eb_abs_err", cmeb_err, floor=1e-10)
    ledger.record("forecast_comparison", "special_abs_err", spec_err, floor=1e-12)
    assert max(dm_err) < 1e-10
    assert max(gw_err) < 1e-10
    assert max(cmeb_err) < 1e-9
    assert max(spec_err) < 1e-11
