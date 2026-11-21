"""Asserts for the study's checks (``compare.py``); run with ``pytest studies/var_495``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import compare  # noqa: E402
from online_var import OnlineVAR, WindowVAR, run  # noqa: E402

from data import simulate  # noqa: E402


@pytest.mark.parametrize(("t", "k", "p"), compare.GRID)
def test_fit_and_granger_equal_statsmodels(t: int, k: int, p: int) -> None:
    res = compare.check_fit(t, k, p)
    for name in ("params", "sigma_u", "sigma_u_mle", "cov_params"):
        assert res[name] < 1e-12, (name, res[name])
    assert res["granger_stat"] < 1e-12
    assert res["granger_pvalue"] < 1e-10  # relative, down to p-values of 1e-177


@pytest.mark.parametrize("level", [1e4, 1e6])
def test_far_from_zero_online_beats_raw_design(level: float) -> None:
    res = compare.check_level(level)
    assert res["online_granger"] < 1e-9
    assert res["online_granger"] < res["statsmodels_granger"] / 100


def test_predictions_are_out_of_sample() -> None:
    assert compare.check_out_of_sample() < 1e-12


def test_chunk_invariance() -> None:
    assert compare.check_chunks()


@pytest.mark.parametrize("lam", [1.0, 0.99, 0.9])
def test_decay_and_weights_equal_wls(lam: float) -> None:
    res = compare.check_decay(lam)
    assert res["params"] < 1e-12 and res["sse"] < 1e-12


def test_zero_weight_first_rows_leave_no_nan() -> None:
    y, _ = simulate(200, 3, 2, seed=11)
    w = np.ones(200)
    w[:10] = 0.0
    m, pred, wsum = run(y, 2, lam=0.97, weights=w)
    assert np.all(wsum[:11] == 0.0)
    assert np.all(np.isfinite(m.sxx)) and np.all(np.isfinite(m.params()))
    assert np.all(np.isfinite(pred[30:]))


def test_weight_sum_is_before_the_row_and_its_decay() -> None:
    y, _ = simulate(50, 2, 1, seed=12)
    lam = 0.9
    _, _, wsum = run(y, 1, lam=lam)
    expect = np.concatenate([[0.0, 0.0], np.cumsum(lam ** -np.arange(48)) * lam ** np.arange(48)])
    assert np.allclose(wsum, expect[:50], rtol=1e-14)


def test_missing_rows() -> None:
    assert compare.check_missing() < 1e-12


def test_select_order() -> None:
    assert compare.check_select_order() < 1e-12


def test_grangercausalitytests_forms() -> None:
    res = compare.check_bivariate()
    assert res["stat"] < 1e-12 and res["pvalue"] < 1e-12 and res["var_f_stat"] < 1e-12
    # Same statistic, different denominator degrees of freedom: the p-values differ.
    assert res["p_ssr_ftest_0to1"] > 1.3 * res["p_var_f_0to1"]


def test_min_weight_withholds_the_fit() -> None:
    y, _ = simulate(100, 2, 1, seed=13)
    _, pred, wsum = run(y, 1, min_weight=40.0)
    assert np.all(np.isnan(pred[wsum < 40.0])) and np.all(np.isfinite(pred[wsum >= 40.0]))


def test_state_is_bounded_by_shape() -> None:
    m = OnlineVAR(4, 4)
    assert m.state_bytes() == 8 * (16 + 16 + 4 + 256 + 64 + 16 + 3)


@pytest.mark.parametrize(("t", "k", "p"), compare.SPEC_GRID)
def test_spec_grid_equals_statsmodels(t: int, k: int, p: int) -> None:
    """k = 2, 5, 10 by p = 1, 2, 5: fit, Granger, impulse responses and FEVD."""
    res = {**compare.check_fit(t, k, p), **compare.check_irf(t, k, p)}
    for name in ("params", "sigma_u", "sigma_u_mle", "cov_params", "irf", "orth_irf", "fevd"):
        assert res[name] < 1e-12, (name, res[name])
    assert res["irf_cov"] < 1e-12
    assert res["granger_stat"] < 1e-11  # worst measured 1.6e-13 (k = 5, p = 1)
    assert res["granger_pvalue"] < 1e-10


def test_window_equals_statsmodels_on_the_window() -> None:
    res = compare.check_window()
    for name in ("pred", "params", "sigma_u", "cov_params", "granger_stat"):
        assert res[name] < 1e-12, (name, res[name])
    assert res["n_eff"] == 0.0


def test_window_with_zero_weights_and_gaps_equals_wls() -> None:
    assert compare.check_window_gaps() < 1e-12


def test_window_state_size() -> None:
    m = WindowVAR(3, 2, window=100)
    assert m.state_bytes() == 8 * (300 + 100 + 6 + 3 + 36 + 18 + 9 + 4)
