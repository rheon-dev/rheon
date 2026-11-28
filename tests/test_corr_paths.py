"""``rheon.corr`` on the paths no other test took (task 252): each refusal of
an argument of the wrong shape, by name; the summaries that are undefined
below two series (``nan``, as the docstrings say); a minimum-variance
weight whose normaliser is zero; and the Epps inversion read from a closed
row rather than a Gram mapping."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from rheon import corr
from rheon import polars as rp

TIER = "essential"

R2 = np.array([[1.0, 0.3], [0.3, 1.0]])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: corr.shrink(R2, x=np.zeros((10, 3))),
            r"rheon.corr.shrink: x must have one column for each of the matrix\'s 2 columns, "
            r"but it has shape \(10, 3\)",
        ),
        (
            lambda: corr.shrink(R2, x=np.zeros(10)),
            r"rheon.corr.shrink: x must have one column for each of the matrix\'s 2 columns, "
            r"but it has shape \(10,\)",
        ),
        (
            lambda: corr.absorption(R2, 0),
            "rheon.corr.absorption: k must be from 1 to the matrix's 2 columns, but it is 0",
        ),
        (
            lambda: corr.absorption(R2, 3),
            "rheon.corr.absorption: k must be from 1 to the matrix's 2 columns, but it is 3",
        ),
        (
            lambda: corr.absorption_shift(np.zeros(3), np.zeros(4)),
            r"rheon.corr.absorption_shift: ar_fast and ar_slow must have the same shape, but they "
            r"have \(3,\) and \(4,\)",
        ),
        (
            lambda: corr.from_spectral([1.0, 1.0], np.eye(3)),
            r"rheon.corr.from_spectral: 2 eigenvalues need 2 eigenvectors, one for each row "
            r"of vecs, but vecs has shape \(3, 3\)",
        ),
        (
            lambda: corr.from_blocks(np.eye(3), ["a", "a", "b"]),
            r"rheon.corr.from_blocks: labels name 2 blocks, so b must be a 2 by 2 matrix, "
            r"but it has shape \(3, 3\)",
        ),
        (
            lambda: corr.mp_edge(0, 10),
            "rheon.corr.mp_edge: n and m must be finite numbers above 0, but they are 0 and 10",
        ),
        (
            lambda: corr.mp_edge(10, -1),
            "rheon.corr.mp_edge: n and m must be finite numbers above 0, but they are 10 and -1",
        ),
        (
            lambda: corr.signal_share(np.zeros(3), np.ones(2)),
            "rheon.corr.signal_share: n_kish_blocks must hold one sample size for each of the 3 "
            "blocks",
        ),
        (
            lambda: corr.loss(R2, np.eye(3), "qlike"),
            r"rheon.corr.loss: fcst and real must have the same shape, but they have \(2, 2\) and "
            r"\(3, 3\)",
        ),
    ],
)
def test_an_argument_of_the_wrong_shape_is_refused_by_name(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_the_equicorrelation_of_fewer_than_two_series_is_nan():
    assert math.isnan(corr.equicorr(np.eye(1)))
    assert math.isnan(corr.equicorr_row([0.4]))
    # Two series: their one correlation.
    assert corr.equicorr(R2) == pytest.approx(0.3, rel=1e-15)


def test_a_minimum_variance_weight_with_a_zero_normaliser_has_no_loss():
    """``w = F^-1 mu / (mu' F^-1 mu)``: a normaliser of exactly zero leaves
    the loss undefined (``nan``), not an infinity or a division error. A
    positive definite forecast has one only at ``mu = 0``. An indefinite
    forecast holds no minimum-variance portfolio at all, so it has no loss
    off the zero either: ``F^-1 mu = (1, -1/2)`` here is a saddle of
    ``w' F w``, which scored 20/9 before PLAN 578h (review 2026-11-23,
    MATH-5)."""
    f = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert math.isnan(corr.loss(f, np.eye(2), "minvar", mu=[0.0, 0.0]))
    indefinite = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert math.isnan(corr.loss(indefinite, np.eye(2), "minvar", mu=[1.0, 1.0]))
    assert math.isnan(corr.loss(indefinite, np.eye(2), "minvar", mu=[1.0, 0.5]))
    # A positive definite forecast off the zero: w' R w by its definition.
    mu = np.array([1.0, 0.5])
    w = np.linalg.solve(f, mu)
    w = w / (mu @ w)
    assert corr.loss(f, np.eye(2), "minvar", mu=mu) == pytest.approx(float(w @ w), rel=1e-15)


def test_the_epps_inversion_reads_a_closed_row_as_the_gram_it_closed():
    """A group's closed row (``closed_groups``) carries its lagged
    co-moments as the Gram mapping of the open group does: inverted from
    either, the correlation at scale ``L`` is the same, to the bit."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(600)
    rows = pl.DataFrame({"x0": a, "x1": np.concatenate([[0.0], a[:-1]]) + 0.1 * a})
    spec = rp.spec.ew_cov(
        "c",
        features=["x0", "x1"],
        lam=1.0,
        stats=[],
        lags=[1, 2],
        group="g",
        group_close="monotone",
    )
    first = rows.with_columns(g=pl.lit("a"))
    closing = pl.DataFrame({"g": ["b"], "x0": [0.0], "x1": [0.0]})
    bank = rp.ModelBank([spec])
    bank.fit_predict(first)
    open_gram = bank.gram("c")[0]
    bank.fit_predict(closing)
    row = bank.closed_groups()
    assert row.height == 1 and row["group"].item() == "a"
    want = corr.epps_invert(open_gram, lags=3)
    got = corr.epps_invert(row, lags=3)
    np.testing.assert_array_equal(got, want)
    assert got[0, 1] != rp.gram.correlation(open_gram)[0, 1], "the lags moved it"


# --- review 2026-11-23, MATH-6 -------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf])
@pytest.mark.parametrize("which", ["n", "m"])
def test_the_noise_floor_refuses_a_count_that_is_not_finite(bad, which):
    """MATH-6 (1): ``n`` or ``m`` NaN passed ``n <= 0`` and gave NaN edges;
    an infinite one gave edges that bound nothing. Each is refused by name,
    by ``mp_edge`` and by ``mp_density``."""
    kw = {"n": 400, "m": 100, which: bad}
    with pytest.raises(ValueError, match=r"n and m must be finite numbers above 0"):
        corr.mp_edge(**kw)
    with pytest.raises(ValueError, match=r"n and m must be finite numbers above 0"):
        corr.mp_density([1.0], **kw)


def test_an_unsymmetric_matrix_is_read_as_its_symmetric_part():
    """MATH-6 (2): ``eigh`` and ``eigvalsh`` read the lower triangle alone,
    so ``absorption`` and ``spectral`` of a matrix whose halves differ were
    those of its lower half mirrored, the upper half unread. Each reads
    ``(r + r') / 2``, as ``nearest`` does."""
    r = np.array([[1.0, 0.9], [0.1, 1.0]])
    sym = 0.5 * (r + r.T)
    assert corr.absorption(r, 1) == pytest.approx(corr.absorption(sym, 1), rel=1e-15)
    assert corr.absorption(r, 1) == pytest.approx(0.75)
    got, want = corr.spectral(r, 2), corr.spectral(sym, 2)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-15)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-15)


def test_a_nan_off_its_mirror_is_named_where_it_is():
    """MATH-6 (3): a NaN at ``b[0, 1]`` against a number at ``b[1, 0]``
    fails the symmetry check, but ``nanargmax`` skipped it and named the
    largest of the other gaps, a pair that matched."""
    b = np.array([[1.0, math.nan, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    with pytest.raises(ValueError, match=r"b\[0, 1\] is nan and b\[1, 0\] is 0\.4"):
        corr.from_blocks(b, ["a", "b", "c"])
