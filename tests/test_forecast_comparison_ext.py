"""Tests for pure-Rust special functions, extended ``rp.eval.diebold_mariano``,
``rp.eval.giacomini_white``, and ``rp.eval.mean_seqtest``."""

from __future__ import annotations

import math

import mpmath
import numpy as np
import polars as pl
import pytest
from dieboldmariano import dm_test
from mapie.exchangeability_testing.bounds import (
    GammaExponentialMixtureBound,
    conjugate_mixture_empirical_bernstein_bound,
)
from scipy import special as sp_special
from scipy import stats as sp_stats
from statsmodels.stats.sandwich_covariance import S_hac_simple, weights_bartlett

from rheon import _rheon as _native
from rheon import polars as rp

TIER = "essential"


def _two_spec_frame(seed: int = 13, n: int = 240) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    ra = 0.85 * rng.normal(size=n)
    rb = 1.05 * rng.normal(size=n) + 0.08
    z_inst = rng.normal(size=n)
    return pl.DataFrame(
        {
            "g": ["g0" if i < n // 2 else "g1" for i in range(n)],
            "z_inst": z_inst,
            "a": pl.DataFrame({"resid_y": ra}).select(pl.struct(pl.all()).alias("a"))["a"],
            "b": pl.DataFrame({"resid_y": rb}).select(pl.struct(pl.all()).alias("b"))["b"],
            "y": y,
        }
    )


def test_special_functions_match_mpmath_and_scipy():
    """``betainc``, ``gammainc``, ``gammaincc``, ``student_t_sf``, ``chi2_sf``,
    and ``cm_eb_bound`` match 50-digit ``mpmath``, ``scipy.special``, and
    ``mapie`` to tight tolerances across small, moderate, and extreme arguments."""
    mpmath.mp.dps = 50
    for a, b, x in [
        (0.5, 0.5, 0.25),
        (1.5, 3.5, 0.7),
        (15.0, 0.5, 0.92),
        (500.0, 0.5, 0.995),
        (50.0, 12.0, 0.85),
    ]:
        want = float(mpmath.betainc(a, b, 0, x, regularized=True))
        got = _native.special_betainc(a, b, x)
        assert abs(got - want) <= 1e-13 * max(1.0, abs(want)), (a, b, x, got, want)

    for a, x in [
        (0.5, 0.05),
        (0.5, 10.0),
        (2.5, 1.0),
        (2.5, 15.0),
        (50.0, 48.0),
        (50.0, 70.0),
        (1000.0, 1000.0),
        (1000.0, 1050.0),
        (1e5, 1e5 + 200.0),
    ]:
        want_p = float(sp_special.gammainc(a, x))
        want_q = float(sp_special.gammaincc(a, x))
        got_p = _native.special_gammainc(a, x)
        got_q = _native.special_gammaincc(a, x)
        assert abs(got_p - want_p) <= 1e-12 * max(1e-15, want_p), (a, x, got_p, want_p)
        assert abs(got_q - want_q) <= 1e-12 * max(1e-15, want_q), (a, x, got_q, want_q)

    for df in (1.0, 2.0, 10.0, 100.0, 1e4, 1e5, 1e7):
        for t in (-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0):
            want = float(sp_stats.t.sf(t, df))
            got = _native.special_student_t_sf(t, df)
            assert abs(got - want) <= 1e-11 * max(1e-15, want), (df, t, got, want)

    for df in (1.0, 2.0, 5.0, 20.0, 100.0):
        for x in (0.1, 1.0, 5.0, 25.0, 80.0):
            want = float(sp_stats.chi2.sf(x, df))
            got = _native.special_chi2_sf(x, df)
            assert abs(got - want) <= 1e-12 * max(1e-15, want), (df, x, got, want)

    # CM-EB boundary vs MAPIE GammaExponentialMixtureBound
    for v_opt in (1.0, 25.0, 100.0):
        for alpha in (0.01, 0.05, 0.1):
            bnd = GammaExponentialMixtureBound(v_opt=v_opt, c=1.0, alpha_opt=alpha / 2.0)
            for v in (0.0, 0.5, 5.0, 50.0, 500.0):
                want = float(bnd(v=v, alpha=alpha / 2.0))
                got = _native.special_cm_eb_bound(v, v_opt, alpha)
                assert abs(got - want) <= 1e-10 * max(1.0, want), (v_opt, alpha, v, got, want)


@pytest.mark.parametrize(
    "loss_kw",
    [
        {"loss": "squared"},
        {"loss": "absolute"},
        {"loss": "pinball", "quantile": 0.8},
        {"loss": "linex", "linex_a": 0.5},
    ],
)
@pytest.mark.parametrize("lags", [0, 3])
@pytest.mark.parametrize("hln", [False, True])
def test_diebold_mariano_matches_oracle_across_losses_hln_t_and_alternatives(loss_kw, lags, hln):
    """Extended ``diebold_mariano`` matches ``dieboldmariano.dm_test`` across
    squared, absolute, pinball, and linex losses, HLN correction, t reference,
    and one-sided alternatives."""
    df = _two_spec_frame(seed=37, n=220)
    ra = df["a"].struct.field("resid_y").to_numpy()
    rb = df["b"].struct.field("resid_y").to_numpy()
    y = np.zeros_like(ra)
    # dm_test computes loss(y, f_a) - loss(y, f_b); pass y=0, f=-r so y - f = r.
    fa, fb = -ra, -rb
    kind = loss_kw["loss"]

    def lf(u: float, v: float) -> float:
        e = u - v
        if kind == "squared":
            return e * e
        if kind == "absolute":
            return abs(e)
        if kind == "pinball":
            q = float(loss_kw["quantile"])
            return q * e if e >= 0.0 else (q - 1.0) * e
        a = float(loss_kw["linex_a"])
        return math.exp(a * e) - a * e - 1.0

    got_two = rp.eval.diebold_mariano(
        df, a="a", b="b", lags=lags, hln=hln, reference="t", alternative="two-sided", **loss_kw
    )
    got_gt = rp.eval.diebold_mariano(
        df, a="a", b="b", lags=lags, hln=hln, reference="t", alternative="greater", **loss_kw
    )
    got_lt = rp.eval.diebold_mariano(
        df, a="a", b="b", lags=lags, hln=hln, reference="t", alternative="less", **loss_kw
    )
    # Note: dm_test sign is loss_a - loss_b, opposite to Rheon's loss_b - loss_a.
    want_stat, want_p2 = dm_test(
        y, fa, fb, loss=lf, h=lags + 1, harvey_correction=hln, variance_estimator="bartlett"
    )
    _, want_p_gt = dm_test(
        y,
        fa,
        fb,
        loss=lf,
        h=lags + 1,
        harvey_correction=hln,
        one_sided=True,
        variance_estimator="bartlett",
    )
    assert np.isclose(got_two["dm"][0], -want_stat, rtol=1e-11, atol=1e-11)
    # dieboldmariano's own pure-Python incomplete beta holds ~1e-10; scipy.stats.t
    # holds 1e-12.
    want_p2_scipy = float(2.0 * sp_stats.t.sf(abs(want_stat), df=len(ra) - 1))
    assert np.isclose(got_two["p_value"][0], want_p2_scipy, rtol=1e-12, atol=1e-12)
    assert np.isclose(got_two["p_value"][0], want_p2, rtol=1e-8, atol=1e-10)
    assert np.isclose(got_gt["p_value"][0], want_p_gt, rtol=1e-8, atol=1e-10)
    assert np.isclose(got_gt["p_value"][0] + got_lt["p_value"][0], 1.0, atol=1e-12)

    # Streaming LazyFrame matches DataFrame
    lazy_two = rp.eval.diebold_mariano(
        df.lazy(), a="a", b="b", lags=lags, hln=hln, reference="t", **loss_kw
    )
    assert np.isclose(lazy_two["dm"][0], got_two["dm"][0], rtol=1e-11)
    assert np.isclose(lazy_two["p_value"][0], got_two["p_value"][0], rtol=1e-11)


def test_diebold_mariano_callable_loss_half_life_hln_and_refusals():
    """Callable loss works; ``hln=True`` works beside ``half_life``; illegal
    option combinations are refused."""
    df = _two_spec_frame(seed=51, n=200)
    built_in = rp.eval.diebold_mariano(df, a="a", b="b", loss="absolute")
    custom = rp.eval.diebold_mariano(df, a="a", b="b", loss=lambda e: e.abs())
    assert built_in.equals(custom)

    # half_life + hln=True works on both DataFrame and LazyFrame
    ew_df = rp.eval.diebold_mariano(df, a="a", b="b", lags=2, half_life=30.0, hln=True)
    ew_lf = rp.eval.diebold_mariano(
        df.lazy(), a="a", b="b", lags=2, half_life=30.0, hln=True
    ).collect()
    assert np.allclose(
        ew_df["diebold_mariano"].struct.field("dm_y").drop_nulls().to_numpy(),
        ew_lf["diebold_mariano"].struct.field("dm_y").drop_nulls().to_numpy(),
        rtol=1e-11,
    )

    with pytest.raises(ValueError, match="reference"):
        rp.eval.diebold_mariano(df, a="a", b="b", half_life=30.0, reference="t")
    with pytest.raises(ValueError, match="alternative"):
        rp.eval.diebold_mariano(df, a="a", b="b", half_life=30.0, alternative="greater")
    with pytest.raises(ValueError, match="quantile"):
        rp.eval.diebold_mariano(df, a="a", b="b", loss="pinball")
    with pytest.raises(ValueError, match="linex_a"):
        rp.eval.diebold_mariano(df, a="a", b="b", loss="linex")


@pytest.mark.parametrize("instrument_lags,lags", [(0, 0), (1, 3), (2, 5)])
def test_giacomini_white_matches_statsmodels_and_streams(instrument_lags, lags):
    """``rp.eval.giacomini_white`` matches ``statsmodels`` ``S_hac_simple`` to
    1e-12 in both DataFrame and streaming LazyFrame modes, supports extra
    instruments and ``half_life``, and enforces the 256 KiB cap."""
    df = _two_spec_frame(seed=67, n=260)
    got_df = rp.eval.giacomini_white(df, a="a", b="b", lags=lags, instrument_lags=instrument_lags)
    got_lf = rp.eval.giacomini_white(
        df.lazy(), a="a", b="b", lags=lags, instrument_lags=instrument_lags
    )
    assert got_df.equals(got_lf)

    ra = df["a"].struct.field("resid_y").to_numpy()
    rb = df["b"].struct.field("resid_y").to_numpy()
    d = rb * rb - ra * ra
    y = d[instrument_lags:]
    cols = [np.ones(len(y))] + [
        d[instrument_lags - j - 1 : len(d) - j - 1] for j in range(instrument_lags)
    ]
    H = np.column_stack(cols)
    Z = H * y[:, None]
    S = S_hac_simple(Z, nlags=lags, weights_func=weights_bartlett)
    zs = Z.sum(axis=0)
    want_gw = float(zs @ np.linalg.solve(S, zs))
    want_p = float(sp_stats.chi2.sf(want_gw, df=H.shape[1]))

    assert got_df["n"][0] == len(y)
    assert got_df["df"][0] == H.shape[1]
    assert np.isclose(got_df["gw"][0], want_gw, rtol=1e-12, atol=1e-12)
    assert np.isclose(got_df["p_value"][0], want_p, rtol=1e-12, atol=1e-12)

    # Extra instruments + half_life streaming
    ew_df = rp.eval.giacomini_white(
        df,
        a="a",
        b="b",
        lags=2,
        instrument_lags=1,
        instruments=["z_inst"],
        half_life=40.0,
        group="g",
    )
    ew_lf = rp.eval.giacomini_white(
        df.lazy(),
        a="a",
        b="b",
        lags=2,
        instrument_lags=1,
        instruments=["z_inst"],
        half_life=40.0,
        group="g",
    ).collect()
    assert ew_df.equals(ew_lf)

    # Cap refusal (256 KiB): q = 30, lags = 40 -> 8 * (41 * 900 + ...) > 262_144
    with pytest.raises(ValueError, match="262144"):
        rp.eval.giacomini_white(df, a="a", b="b", lags=40, instrument_lags=29)


def test_mean_seqtest_cm_eb_matches_mapie_and_betting_and_clip_refusal():
    """``rp.eval.mean_seqtest`` matches MAPIE's ``conjugate_mixture_empirical_bernstein_bound``
    to 1e-10 (with and without ``running_intersection``), supports ``method='betting'``,
    streams chunk-invariantly, and refuses out-of-bound values unless ``clip=True``."""
    rng = np.random.default_rng(89)
    n = 150
    bound = 2.0
    d = np.clip(0.25 + 0.4 * rng.normal(size=n), -bound, bound)
    r = (d / bound + 1.0) * 0.5
    df = pl.DataFrame({"d": d})

    for ri in (False, True):
        got_df = rp.eval.mean_seqtest(
            df,
            targets=["d"],
            bound=bound,
            method="cm_eb",
            alpha=0.05,
            v_opt=1.0,
            running_intersection=ri,
        )
        got_lf = rp.eval.mean_seqtest(
            df.lazy(),
            targets=["d"],
            bound=bound,
            method="cm_eb",
            alpha=0.05,
            v_opt=1.0,
            running_intersection=ri,
        ).collect()
        assert got_df.equals(got_lf)

        st = got_df["mean_seqtest"].struct.unnest()
        want_lo_r = conjugate_mixture_empirical_bernstein_bound(
            r, v_opt=1.0, alpha=0.05, bound_side="lower", running_intersection=ri
        )
        want_hi_r = conjugate_mixture_empirical_bernstein_bound(
            r, v_opt=1.0, alpha=0.05, bound_side="upper", running_intersection=ri
        )
        want_lo = bound * (2.0 * want_lo_r - 1.0)
        want_hi = bound * (2.0 * want_hi_r - 1.0)
        assert np.allclose(st["lower_d"].to_numpy(), want_lo, rtol=1e-10, atol=1e-10)
        assert np.allclose(st["upper_d"].to_numpy(), want_hi, rtol=1e-10, atol=1e-10)

    # Betting e-process in compare mode + LazyFrame invariance
    cmp_df = _two_spec_frame(seed=93, n=120)
    b_df = rp.eval.mean_seqtest(
        cmp_df, a="a", b="b", bound=10.0, method="betting", clip=True, group="g"
    )
    b_lf = rp.eval.mean_seqtest(
        cmp_df.lazy(), a="a", b="b", bound=10.0, method="betting", clip=True, group="g"
    ).collect()
    assert b_df.equals(b_lf)
    assert set(b_df["mean_seqtest"].struct.fields) == {
        "log_e_a_y",
        "log_e_b_y",
        "mean_y",
        "reject_y",
        "n_y",
    }

    # Out-of-bound refusal when clip=False
    bad = pl.DataFrame({"d": [0.1, 2.5, -0.2]})
    with pytest.raises(rp.RefusedValueError, match="bound"):
        rp.eval.mean_seqtest(bad, targets=["d"], bound=2.0, clip=False)
    clipped = rp.eval.mean_seqtest(bad, targets=["d"], bound=2.0, clip=True)
    assert clipped["mean_seqtest"].struct.field("n_d").to_list() == [1, 2, 3]


def test_giacomini_white_kish_effective_df_nw_and_min_samples():
    """``giacomini_white`` gates on Kish effective sample size ``n_eff >= min_samples``,
    ``n_eff >= q``, and ``df_NW = n_eff / (lags + 1) >= max(q + 1, 4)``."""
    df = _two_spec_frame(seed=101, n=300)
    # half_life=1.0 -> asymptotic Kish n_eff = 3.0 < 4.0: withheld on every row.
    tiny = rp.eval.giacomini_white(
        df, a="a", b="b", lags=0, instrument_lags=0, half_life=1.0, min_samples=1
    )
    assert tiny["giacomini_white"].struct.field("gw_y").null_count() == 300

    # half_life=3.0 -> asymptotic n_eff ~ 8.66: with q=2, emits at lags=0 & min_samples=1,
    # withheld at lags=2 (df_NW <= 2.89 < 4) or min_samples=15.
    ok = rp.eval.giacomini_white(
        df, a="a", b="b", lags=0, instrument_lags=1, half_life=3.0, min_samples=1
    )
    assert ok["giacomini_white"].struct.field("gw_y").drop_nulls().len() > 0
    bad_lags = rp.eval.giacomini_white(
        df, a="a", b="b", lags=2, instrument_lags=1, half_life=3.0, min_samples=1
    )
    assert bad_lags["giacomini_white"].struct.field("gw_y").null_count() == 300
    bad_ms = rp.eval.giacomini_white(
        df, a="a", b="b", lags=0, instrument_lags=1, half_life=3.0, min_samples=15
    )
    assert bad_ms["giacomini_white"].struct.field("gw_y").null_count() == 300

    # Unweighted with q=5 (instrument_lags=4), lags=2 -> requires df_NW = n/3 >= 6 -> n >= 18.
    short = _two_spec_frame(seed=102, n=19)  # 19 - 4 = 15 complete observations (< 18)
    gw_short = rp.eval.giacomini_white(
        short, a="a", b="b", lags=2, instrument_lags=4, min_samples=1
    )
    assert gw_short["n"][0] == 15 and gw_short["gw"][0] is None
    long_df = _two_spec_frame(seed=102, n=25)  # 25 - 4 = 21 complete observations (>= 18)
    gw_long = rp.eval.giacomini_white(
        long_df, a="a", b="b", lags=2, instrument_lags=4, min_samples=1
    )
    assert gw_long["n"][0] == 21 and gw_long["gw"][0] is not None
