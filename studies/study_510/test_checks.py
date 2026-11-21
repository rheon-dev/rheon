"""Asserts for study 510's checks (``compare.py``); run with ``pytest studies/study_510``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import compare  # noqa: E402


@pytest.mark.parametrize(("rule", "bx", "by", "window"), compare.MI_GRID)
def test_mi_equals_sklearn_and_miller_madow_its_definition(rule, bx, by, window) -> None:
    r = compare.check_mi(rule, bx, by, window, n=2000)
    assert r["sklearn"] < 1e-13
    assert r["mm_def"] < 1e-13
    assert r["mm_plus_sign"] > 1e-3  # the correction is subtracted, not added


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("bins", [8, 16, 32])
def test_gaussian_band(rho, bins) -> None:
    r = compare.check_gaussian(rho, bins, seeds=20)
    # Each seed's Miller-Madow value within 4 sd of the binned population MI, plus a fifth of
    # the correction for the second-order bias the correction leaves at many cells per row.
    assert r["mm_worst"] < 4 * r["sd"] + 0.2 * r["bias_pred"]
    # The binned value sits below the true MI by the discretization loss, never above it.
    assert -1e-12 <= r["loss"] < {8: 0.14, 16: 0.06, 32: 0.026}[bins]  # the quadrature: ~1e-14


@pytest.mark.parametrize("half_life", [5, 50, 1000])
def test_decay_equals_longhand(half_life) -> None:
    r = compare.check_decay(half_life, n=3000)
    assert r["table_rel"] < 1e-12
    assert r["mi_longhand"] < 1e-12
    assert r["mi_scipy_entropy"] < 1e-12


def test_rings_hold_exactly_their_rows() -> None:
    r = compare.check_rings(n=1200)
    assert r["rank_mismatch"] == 0
    assert r["kendall_pairs_mismatch"] == 0
    assert r["mi_table_mismatch"] == 0
    assert r["empty_window_ok"] == 1
    assert r["first_row_zero"] == 1


def test_fractional_weights_leave_residue_but_not_in_mi() -> None:
    r = compare.check_window_float(n=20_000)
    assert r["cell_residue"] < 1e-12
    assert r["mi_gap"] < 1e-13


@pytest.mark.parametrize(("window", "every"), [(50, 1), (200, 7)])
def test_ranks_equal_scipy(window, every) -> None:
    r = compare.check_ranks(window, every, n=600)
    assert r["spearman"] < 1e-14
    assert r["kendall"] < 1e-14
    assert r["xi_scipy"] < 1e-14
    assert r["xi_longhand"] < 1e-14
    assert r["xi_cont"] < 1e-14
    assert r["held_ok"] == 1
    assert r["chunks_ok"] == 1


def test_state_size_formula() -> None:
    r = compare.check_state()
    assert r["mi_decay"] == r["mi_decay_formula"]
    assert r["mi_window"] == r["mi_window_formula"]
    assert r["ranks"] == r["ranks_formula"]


def test_mi_precision() -> None:
    r = compare.check_mi_precision()
    assert r["abs_gap"] < 1e-14
