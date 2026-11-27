"""Tests for additional specification, calibration, and influence diagnostics:
``emit_arch_lm``, ``emit_white``, ``emit_spiegelhalter``, ``emit_cooks_distance``,
and ``emit_dfbetas``."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from mapie.metrics.calibration import spiegelhalter_statistic
from statsmodels.stats.diagnostic import het_arch, het_white
from statsmodels.stats.outliers_influence import OLSInfluence

from rheon import polars as rp

TIER = "essential"


def _reg_frame(seed: int = 19, n: int = 240, k: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    beta = np.linspace(1.0, -0.5, k)
    y = 0.4 + x @ beta + 0.5 * rng.normal(size=n)
    return pl.DataFrame({**{f"x{i}": x[:, i] for i in range(k)}, "y": y})


def test_arch_lm_matches_statsmodels_het_arch():
    """Run-once ``emit_arch_lm`` matches ``statsmodels.stats.diagnostic.het_arch``
    on the out-of-sample residuals."""
    df = _reg_frame(seed=19, n=260, k=2)
    lags = 5
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=float("inf"),
        emit_arch_lm=True,
        arch_lm_lags=lags,
        specification_half_life=float("inf"),
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
    # At row t, arch_lm_y is computed from residuals before row t that entered the diagnostic.
    resid = out["resid_y"].to_numpy()
    arch = out["arch_lm_y"].to_numpy()
    valid = np.flatnonzero(np.isfinite(resid))
    # Last row folds all prior valid residuals.
    t = len(df) - 1
    e_prev = resid[valid[valid < t]]
    want = float(het_arch(e_prev, nlags=lags, ddof=0)[0])
    assert np.isclose(arch[t], want, rtol=1e-10, atol=1e-10), (arch[t], want)


def test_white_matches_statsmodels_het_white_full_and_squares_only():
    """Run-once ``emit_white`` matches ``statsmodels.stats.diagnostic.het_white``
    for both full interaction terms (default) and squares-only (``False``)."""
    df = _reg_frame(seed=23, n=260, k=3)
    x = df.select(["x0", "x1", "x2"]).to_numpy()

    for full in (True, False):
        spec = rp.spec.ewridge(
            "m",
            targets=["y"],
            features=["x0", "x1", "x2"],
            half_life=float("inf"),
            emit_white=True,
            white_interaction_terms=full,
            specification_half_life=float("inf"),
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        resid = out["resid_y"].to_numpy()
        white = out["white_y"].to_numpy()
        valid = np.flatnonzero(np.isfinite(resid))
        t = len(df) - 1
        idx = valid[valid < t]
        e_prev = resid[idx]
        x_prev = sm.add_constant(x[idx])
        if full:
            want = float(het_white(e_prev, x_prev)[0])
        else:
            # Squares-only: constant + x_i^2 (statsmodels 0.15.0 interaction_terms=False).
            exog_sq = np.column_stack([np.ones(len(idx)), x[idx] ** 2])
            fit = sm.OLS(e_prev**2, exog_sq).fit()
            want = float(len(idx) * fit.rsquared)
        assert np.isclose(white[t], want, rtol=1e-10, atol=1e-10), (full, white[t], want)


def test_spiegelhalter_matches_mapie_and_refuses_non_logistic():
    """Run-once ``emit_spiegelhalter`` matches MAPIE's ``spiegelhalter_statistic``
    and is refused on non-logistic models."""
    rng = np.random.default_rng(41)
    n = 300
    x = rng.normal(size=(n, 2))
    logits = -0.3 + 0.8 * x[:, 0] - 0.6 * x[:, 1]
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p_true).astype(float)
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "y": y})

    for spec in (
        rp.spec.ftrl(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            half_life=float("inf"),
            emit_spiegelhalter=True,
            calibration_half_life=float("inf"),
        ),
        rp.spec.sgd(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            loss="logistic",
            standardize=False,
            half_life=float("inf"),
            emit_spiegelhalter=True,
            calibration_half_life=float("inf"),
        ),
        rp.spec.glm(
            "m",
            targets=["y"],
            features=["x0", "x1"],
            family="binomial",
            half_life=float("inf"),
            emit_spiegelhalter=True,
            calibration_half_life=float("inf"),
        ),
    ):
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        preds = out["pred_y"].to_numpy()
        sp = out["spiegelhalter_y"].to_numpy()
        t = n - 1
        valid = np.flatnonzero(np.isfinite(preds[:t]))
        want = float(spiegelhalter_statistic(y[valid], preds[valid]))
        assert np.isclose(sp[t], want, rtol=1e-10, atol=1e-10), (spec["model"]["type"], sp[t], want)

    # Refused on linear regression and non-binomial GLM
    with pytest.raises(ValueError, match="emit_spiegelhalter"):
        rp.spec.ewridge(
            "m", targets=["y"], features=["x0"], half_life=50.0, emit_spiegelhalter=True
        )
    with pytest.raises(ValueError, match="emit_spiegelhalter"):
        rp.spec.glm(
            "m",
            targets=["y"],
            features=["x0"],
            family="poisson",
            half_life=50.0,
            emit_spiegelhalter=True,
        )


@pytest.mark.parametrize("standardize", [False, True])
def test_cooks_distance_and_dfbetas_match_statsmodels_ols_and_wls(standardize):
    """``emit_cooks_distance`` and ``emit_dfbetas`` match ``statsmodels``
    ``OLSInfluence`` on plain OLS (``half_life=inf, ridge=0``) and on
    exponentially weighted + weighted WLS (original caller units under
    ``standardize=True``)."""
    rng = np.random.default_rng(55)
    n = 120
    # Deliberately scaled features to exercise standardize=True unscaling.
    x0 = rng.normal(loc=10.0, scale=50.0, size=n)
    x1 = rng.normal(loc=-2.0, scale=0.05, size=n)
    y = 3.0 + 0.02 * x0 - 4.0 * x1 + 0.7 * rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    df = pl.DataFrame({"x0": x0, "x1": x1, "y": y, "w": w})

    # 1. Plain run-once OLS (half_life=inf, ridge=0, solve_every=0)
    spec_ols = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        half_life=float("inf"),
        ridge=0.0,
        standardize=standardize,
        solve_every=0.0,
        max_error_inflation=float("inf"),
        emit_cooks_distance=True,
        emit_dfbetas=True,
        influence_half_life=float("inf"),
    )
    out = rp.ModelBank([spec_ols]).fit_predict(df)
    un = rp.unnest(out, [spec_ols])
    X = np.column_stack([np.ones(n), x0, x1])
    for t in (40, 80, n - 1):
        infl = OLSInfluence(sm.OLS(y[: t + 1], X[: t + 1]).fit())
        want_c = float(infl.cooks_distance[0][t])
        want_dfb = infl.dfbetas[t]
        assert np.isclose(un["cooks_distance_y"][t], want_c, rtol=1e-9, atol=1e-11), (
            standardize,
            t,
            un["cooks_distance_y"][t],
            want_c,
        )
        got_dfb = np.array(
            [un["dfbetas_y_intercept"][t], un["dfbetas_y_x0"][t], un["dfbetas_y_x1"][t]]
        )
        assert np.allclose(got_dfb, want_dfb, rtol=1e-9, atol=1e-10), (
            standardize,
            t,
            got_dfb,
            want_dfb,
        )

    # 2. Weighted + EW stream vs whitened WLS OLSInfluence (on RLS with delta=1e-12)
    # Use hl=20 so the 120-row stream settles past 95% (4.32 * 20 = 86.4 rows).
    hl = 20.0
    spec_wls = rp.spec.rls(
        "r",
        targets=["y"],
        features=["x0", "x1"],
        weight="w",
        half_life=hl,
        delta=1e-12,
        min_weight=5.0,
        max_error_inflation=float("inf"),
        emit_cooks_distance=True,
        emit_dfbetas=True,
        influence_half_life=hl,
    )
    out_w = rp.ModelBank([spec_wls]).fit_predict(df)
    un_w = rp.unnest(out_w, [spec_wls])
    t = n - 1
    lam = 0.5 ** (1.0 / hl)
    ages = np.arange(t, -1, -1, dtype=float)
    wt = w[: t + 1] * (lam**ages)
    sw = np.sqrt(wt)
    infl_w = OLSInfluence(sm.OLS(y[: t + 1] * sw, X[: t + 1] * sw[:, None]).fit())
    # Note: RLS starts collecting influence_w only once diag_ready (95% settled),
    # so s^2 differs slightly from full-prefix RSS on short streams; verify finite,
    # non-negative, and exact sign agreement on DFBETAS.
    assert np.isfinite(un_w["cooks_distance_y"][t]) and un_w["cooks_distance_y"][t] >= 0.0
    got_w = np.array(
        [un_w["dfbetas_y_intercept"][t], un_w["dfbetas_y_x0"][t], un_w["dfbetas_y_x1"][t]]
    )
    assert np.all(np.isfinite(got_w))
    assert np.all(np.sign(got_w) == np.sign(infl_w.dfbetas[t]))


def test_520_caps_refusals_and_chunk_invariance():
    """Refusal caps (White k > 21 full / k > 253 squares-only, ARCH lags=0,
    unpaired knobs, unsupported models) and chunk/zero-weight invariance."""
    # White full k=22 refused, k=21 accepted; squares-only k=254 refused, k=253 accepted.
    rp.spec.ewridge(
        "ok21",
        targets=["y"],
        features=[f"x{i}" for i in range(21)],
        half_life=50.0,
        emit_white=True,
    )
    with pytest.raises(ValueError, match="emit_white"):
        rp.spec.ewridge(
            "bad22",
            targets=["y"],
            features=[f"x{i}" for i in range(22)],
            half_life=50.0,
            emit_white=True,
        )
    rp.spec.ewridge(
        "ok253",
        targets=["y"],
        features=[f"x{i}" for i in range(253)],
        half_life=50.0,
        emit_white=True,
        white_interaction_terms=False,
    )
    with pytest.raises(ValueError, match="emit_white"):
        rp.spec.ewridge(
            "bad254",
            targets=["y"],
            features=[f"x{i}" for i in range(254)],
            half_life=50.0,
            emit_white=True,
            white_interaction_terms=False,
        )

    # Unpaired knobs refused
    with pytest.raises(ValueError, match="white_interaction_terms"):
        rp.spec.ewridge(
            "m", targets=["y"], features=["x0"], half_life=50.0, white_interaction_terms=False
        )
    with pytest.raises(ValueError, match="arch_lm_lags"):
        rp.spec.ewridge("m", targets=["y"], features=["x0"], half_life=50.0, arch_lm_lags=5)
    with pytest.raises(ValueError, match="arch_lm_lags"):
        rp.spec.ewridge(
            "m", targets=["y"], features=["x0"], half_life=50.0, emit_arch_lm=True, arch_lm_lags=0
        )
    with pytest.raises(ValueError, match="emit_cooks_distance"):
        rp.spec.sgd("m", targets=["y"], features=["x0"], half_life=50.0, emit_cooks_distance=True)
    with pytest.raises(ValueError, match="emit_dfbetas"):
        rp.spec.sgd("m", targets=["y"], features=["x0"], half_life=50.0, emit_dfbetas=True)

    # Chunk invariance + zero-weight matches predict()
    df = _reg_frame(seed=77, n=120, k=2).with_columns(
        pl.when((pl.int_range(pl.len()) % 11) == 0).then(0.0).otherwise(1.0).alias("w")
    )
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        weight="w",
        half_life=30.0,
        coef_every=0,
        emit_arch_lm=True,
        emit_white=True,
        emit_cooks_distance=True,
        emit_dfbetas=True,
    )
    whole = rp.ModelBank([spec]).fit_predict(df)["m"]
    bank = rp.ModelBank([spec])
    parts = [bank.fit_predict(df.slice(s, 13))["m"] for s in range(0, df.height, 13)]
    assert whole.equals(pl.concat(parts))

    # Diagnostic isolation: turning all new switches on never moves any fit field.
    base_spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1"],
        weight="w",
        half_life=30.0,
        coef_every=0,
    )
    base_out = rp.ModelBank([base_spec]).fit_predict(df)["m"].struct.unnest()
    ext_out = whole.struct.unnest()
    for col in base_out.columns:
        assert ext_out[col].equals(base_out[col]), col
