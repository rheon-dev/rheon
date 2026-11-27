"""Tests for ``ew_cov`` raw second-moment matrix (``raw_cov``) and Gaussian
log-likelihood (``loglik``) statistics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance

from rheon import polars as rp

TIER = "essential"


def _sample_frame(seed: int = 17, n: int = 220, k: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, k))
    corr = np.eye(k) * 0.6 + 0.4
    x = base @ np.linalg.cholesky(corr).T + np.linspace(0.5, 2.0, k)
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def test_raw_cov_matches_sklearn_empirical_covariance_uncentered():
    """``raw_cov`` matches ``EmpiricalCovariance(assume_centered=True)`` to
    round-off on both ``bias=False`` and ``bias=True`` (divisor is always W)."""
    df = _sample_frame(seed=17, n=180, k=3)
    cols = ["x0", "x1", "x2"]
    x = df.select(cols).to_numpy()
    for bias in (False, True):
        spec = rp.spec.ew_cov(
            "m",
            features=cols,
            half_life=float("inf"),
            stats=["raw_cov"],
            bias=bias,
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
        for t in (4, 15, 60, 179):
            want = EmpiricalCovariance(assume_centered=True).fit(x[:t]).covariance_
            for i in range(3):
                for j in range(i, 3):
                    got = out[f"raw_cov_x{i}_x{j}"][t]
                    assert np.isclose(got, want[i, j], rtol=1e-12, atol=1e-13), (
                        bias,
                        t,
                        i,
                        j,
                        got,
                        want[i, j],
                    )


def test_loglik_matches_scipy_multivariate_normal_and_shares_factorization():
    """``loglik`` matches ``scipy.stats.multivariate_normal.logpdf`` under the
    regularized covariance ``Sigma + precision_prior * I`` and produces the exact
    same values whether requested alone or beside ``mahal`` and ``partial_corr``."""
    df = _sample_frame(seed=29, n=200, k=3)
    cols = ["x0", "x1", "x2"]
    x = df.select(cols).to_numpy()
    prior = 0.05

    for bias in (False, True):
        spec_alone = rp.spec.ew_cov(
            "a",
            features=cols,
            half_life=float("inf"),
            stats=["loglik"],
            precision_prior=prior,
            bias=bias,
        )
        spec_mahal = rp.spec.ew_cov(
            "mh",
            features=cols,
            half_life=float("inf"),
            stats=["mahal", "loglik"],
            precision_prior=prior,
            bias=bias,
        )
        spec_shared = rp.spec.ew_cov(
            "s",
            features=cols,
            half_life=float("inf"),
            stats=["partial_corr", "mahal", "loglik"],
            precision_prior=prior,
            bias=bias,
        )
        res = rp.ModelBank([spec_alone, spec_mahal, spec_shared]).fit_predict(df)
        ll_alone = res["a"].struct.field("loglik")
        ll_mahal = res["mh"].struct.field("loglik")
        ll_shared = res["s"].struct.field("loglik")
        assert ll_alone.equals(ll_mahal)
        assert np.allclose(
            ll_alone.drop_nulls().to_numpy(),
            ll_shared.drop_nulls().to_numpy(),
            rtol=1e-13,
            atol=1e-13,
        )

        for t in (10, 50, 120, 199):
            mu = x[:t].mean(axis=0)
            c_factor = 1.0 if bias else t / (t - 1.0)
            cov_reg = np.cov(x[:t], rowvar=False, ddof=0 if bias else 1) + (
                c_factor * prior / t
            ) * np.eye(3)
            want = float(multivariate_normal.logpdf(x[t], mean=mu, cov=cov_reg))
            got = ll_alone[t]
            assert np.isclose(got, want, rtol=1e-12, atol=1e-12), (bias, t, got, want)


def test_single_feature_raw_cov_and_loglik():
    """Both ``raw_cov`` and ``loglik`` are legal for a single feature (``k == 1``)."""
    rng = np.random.default_rng(7)
    x = rng.normal(loc=1.5, scale=0.8, size=100)
    df = pl.DataFrame({"x0": x})
    prior = 0.1
    spec = rp.spec.ew_cov(
        "m",
        features=["x0"],
        half_life=float("inf"),
        stats=["raw_cov", "loglik"],
        precision_prior=prior,
        bias=True,
    )
    out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()
    assert out.columns[:2] == ["raw_cov_x0_x0", "loglik"]
    t = 80
    assert np.isclose(out["raw_cov_x0_x0"][t], np.mean(x[:t] ** 2), rtol=1e-12)
    mu = float(np.mean(x[:t]))
    var = float(np.var(x[:t], ddof=0)) + prior / t
    want_ll = float(multivariate_normal.logpdf([x[t]], mean=[mu], cov=[[var]]))
    assert np.isclose(out["loglik"][t], want_ll, rtol=1e-12)


def test_loglik_requires_precision_prior():
    """``stats=['loglik']`` without ``precision_prior`` is refused clearly."""
    with pytest.raises(ValueError, match="precision_prior"):
        rp.spec.ew_cov("m", features=["x0", "x1"], half_life=20.0, stats=["loglik"])


def test_decay_weights_zero_weight_and_chunk_invariance():
    """``raw_cov`` and ``loglik`` under finite half-life, non-unit weights,
    zero-weight rows, and chunked streaming are bit-identical."""
    rng = np.random.default_rng(101)
    n = 150
    t = np.cumsum(rng.uniform(0.5, 2.0, n))
    w = rng.uniform(0.5, 2.0, n)
    w[::10] = 0.0
    x = rng.normal(size=(n, 2)) + np.array([1.0, -0.5])
    df = pl.DataFrame({"t": t, "w": w, "x0": x[:, 0], "x1": x[:, 1]})
    spec = rp.spec.ew_cov(
        "m",
        features=["x0", "x1"],
        clock="t",
        weight="w",
        half_life=25.0,
        gap_cap=100.0,
        stats=["raw_cov", "loglik", "mahal"],
        precision_prior=0.05,
    )
    whole = rp.ModelBank([spec]).fit_predict(df)["m"]

    bank = rp.ModelBank([spec])
    parts = [bank.fit_predict(df.slice(s, 17))["m"] for s in range(0, n, 17)]
    assert whole.equals(pl.concat(parts))

    # Zero-weight rows removed from a twin stream leave every positive-weight row unchanged.
    df_pos = df.filter(pl.col("w") > 0.0)
    pos_only = rp.ModelBank([spec]).fit_predict(df_pos)["m"].struct.unnest()
    whole_pos = whole.struct.unnest().filter(pl.Series(w > 0.0))
    for col in ("raw_cov_x0_x0", "raw_cov_x0_x1", "raw_cov_x1_x1", "loglik", "mahal"):
        assert whole_pos[col].equals(pos_only[col]), col
