"""``rheon.spec.power_variation`` against third-party libraries that compute
its terms (docs/PLAN.md task 555; docs/TESTING.md, "Libraries the package
does not depend on"):

- ``arch``'s kernel long-run covariance (``arch.covariance.kernel``,
  ``Bartlett``, ``Parzen``, ``TukeyHanning``), at ``bandwidth = H``, whose
  weights are ``k(h / (H + 1))``, divided by ``n``, centered or not: ``n``
  times it is the kernel term, plain or ``center``.
- ``hfhd``'s ``krvm`` with its own ``parzen_kernel``, the flat-top realized
  kernel ``γ_0 + Σ_h k((h - 1)/H)(γ_h + γ_-h)`` on log prices: the
  ``flat_top`` Parzen term.
- ``hfhd``'s ``tsrc``, two-scale realized variance on log prices, ``n`` the
  count of prices and no small-sample factor: composed in Polars from two
  ``scale`` terms of squares and ``n``.

The conventions written in, each library agrees with the definition in
``tests/reference_power_variation.py``; the gaps are relative to the
term's scale (``(2H + 1) Σ r²`` for a kernel sum, whose products cancel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from arch.covariance import kernel as arch_kernel
from hfhd import hf

from rheon import polars as rp

TIER = "essential"

INF = float("inf")
EPS = float(np.finfo(float).eps)


def bounced(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = 0.0005 * np.exp(np.cumsum(0.02 * rng.standard_normal(n)))
    noise = 0.0003 * rng.standard_normal(n + 1)
    return vol * rng.standard_normal(n) + np.diff(noise)


def _last(spec: dict, r: np.ndarray) -> dict:
    """The fields of the row after the last return: every return's terms."""
    df = pl.DataFrame({"r": [*map(float, r), None]})
    return rp.ModelBank([spec]).fit_predict(df).unnest("v").row(-1, named=True)


def _rel(got: float, want: float, scale: float) -> float:
    return abs(got - want) / max(abs(want), abs(scale))


ARCH = {"bartlett": "Bartlett", "parzen": "Parzen", "tukey_hanning": "TukeyHanning"}


@pytest.mark.parametrize("kernel", sorted(ARCH))
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("h", [1, 5, 20])
def test_kernel_terms_are_archs_long_run_variance_times_n(kernel, center, h):
    """``arch``'s long-run variance at ``bandwidth = H`` weights lag ``h`` by
    ``k(h / (H + 1))``, sums every lag and divides by ``n``; with
    ``center`` it takes the sample mean out. ``n`` times it is the term.
    Measured: at most 0.2 ε of the scale (centered), 0.04 ε plain."""
    r = bounced(3_000, 7 + h)
    spec = rp.spec.power_variation(
        "v",
        targets=["r"],
        terms={"k": {"kernel": kernel, "bandwidth": h, "center": center}},
        half_life=INF,
    )
    got = _last(spec, r)["k_r"]
    est = getattr(arch_kernel, ARCH[kernel])(
        r, bandwidth=float(h), center=center, df_adjust=0, force_int=False
    )
    want = float(np.asarray(est.cov.long_run).ravel()[0]) * len(r)
    scale = (2 * h + 1) * float(np.dot(r, r))
    assert _rel(got, want, scale) <= 4 * EPS, (got, want)


def _prices(r: np.ndarray) -> pd.Series:
    """Log prices whose differences are ``r`` to the bit where it can be,
    built as the running sum the returns are fed as, stamped a second
    apart."""
    y = np.concatenate([[0.0], np.cumsum(r)])
    stamps = pd.date_range("2024-01-02 09:30", periods=len(y), freq="s")
    return pd.Series(y, index=stamps)


@pytest.mark.parametrize("h", [3, 10, 40])
def test_the_flat_top_parzen_term_is_hfhds_realized_kernel(h):
    """``hfhd.krvm`` with its Parzen kernel is the flat-top realized kernel
    on the prices' differences, without a jitter. Fed the differences of
    its own prices, the term agrees to the rounding of the sum. Measured:
    at most 3 ulps of the value, 0.03 ε of the scale."""
    y = _prices(bounced(2_000, 31 + h))
    r = np.diff(y.to_numpy())
    want = float(np.asarray(hf.krvm([y], h, pairwise=False, kernel=hf.parzen_kernel)).ravel()[0])
    spec = rp.spec.power_variation(
        "v",
        targets=["r"],
        terms={"rk": {"kernel": "parzen", "bandwidth": h, "flat_top": True}},
        half_life=INF,
    )
    got = _last(spec, r)["rk_r"]
    scale = (2 * h + 1) * float(np.dot(r, r))
    assert _rel(got, want, scale) <= 4 * EPS, (got, want)


@pytest.mark.parametrize(("j", "k"), [(1, 10), (2, 25), (5, 40)])
def test_two_scale_realized_variance_composed_in_polars_is_hfhds(j, k):
    """Zhang, Mykland and Aït-Sahalia's two-scale estimator, ``[Y,Y]^(K) -
    (n̄_K / n̄_J) [Y,Y]^(J)``, ``n̄_K = (N - K + 1) / K``: ``hfhd.tsrc``
    takes ``N`` as the count of prices, ``n + 1``. Composed from the two
    ``scale`` terms of squares and ``n``, as a Polars expression. Measured:
    at most 3 ulps."""
    y = _prices(bounced(4_000, 41 + k))
    r = np.diff(y.to_numpy())
    # `tsrc` takes two series or more; a second on the same stamps leaves
    # the refresh-time grid every stamp, and the first's variance is [0, 0].
    want = float(np.asarray(hf.tsrc([y, 2.0 * y], J=j, K=k))[0, 0])
    spec = rp.spec.power_variation(
        "v",
        targets=["r"],
        terms={"fast": {"powers": [2], "scale": j}, "slow": {"powers": [2], "scale": k}},
        half_life=INF,
    )
    df = pl.DataFrame({"r": [*map(float, r), None]})
    out = rp.ModelBank([spec]).fit_predict(df).unnest("v")
    prices = pl.col("n_r") + 1
    nbar = lambda s: (prices - s + 1) / s  # noqa: E731
    tsrv = out.select(pl.col("slow_r") - nbar(k) / nbar(j) * pl.col("fast_r")).item(-1, 0)
    assert abs(tsrv - want) <= 16 * np.spacing(abs(want)), (tsrv, want)
