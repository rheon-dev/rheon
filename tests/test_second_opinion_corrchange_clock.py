"""``corrchange``'s spans on the clock (``window_size``) against the
libraries that hold its row form.

On the clock a ``monitor`` span runs from its first row through the first
at least ``window_size`` after it, and the next span starts on the row
after; a ``window`` report reads the two windows ``(t − 2D, t − D]`` and
``(t − D, t]``. ``crates/rheon-core/src/corrchange_clock_tests.rs`` holds
each clock form to its row form on the row clock, to the bit. Here, on an
irregular clock of whole units, so that rows fall exactly on a span's and
a window's edges, each span is built from the definition with numpy, and
its statistic is held to the libraries ``tests/test_corrchange.py`` holds
the row form to:

- the monitor's linear CUSUM to statsmodels' ``S_hac_simple`` (Bartlett,
  ``nlags = γ − 1``) at ``bandwidth = 3``, and at ``bandwidth = 1`` to
  ``breaks_cusumolsresid(ξ, ddof=0)``, the OLS-residual CUSUM;
- the scale-ratio check to ``breaks_cusumolsresid`` of ``x̃² − ỹ²``: where
  both CUSUMs pass the critical value the flag is withheld, and
  ``withheld_reason`` says ``scale_ratio_moved``;
- the window kind's statistic to numpy's ``corrcoef`` on each window's
  rows.

The tolerance is the row form's, 1e-12 relative (measured there at 3.6e-15
for the window's merged correlations, and to rounding for the monitor's
two-pass span).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.stats.sandwich_covariance import S_hac_simple

from rheon import polars as rp

TIER = "essential"


def _clock(n: int, seed: int) -> np.ndarray:
    """Whole units, steps of 1 to 4: rows land on a span's edge exactly."""
    return np.cumsum(np.random.default_rng([553, seed]).integers(1, 5, n)).astype(float)


def _spans(t: np.ndarray, d: float) -> list[tuple[int, int]]:
    """Each monitor span, its first and last rows: from its first row
    through the first row at least ``d`` after it, the next starting on the
    row after."""
    out, start = [], 0
    while start < len(t):
        reach = np.flatnonzero(t[start:] - t[start] >= d)
        if reach.size == 0:
            break
        end = start + int(reach[0])
        out.append((start, end))
        start = end + 1
    return out


def _span_series(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The span's influence series ``ξ`` and the scale check's ``x̃² − ỹ²``,
    ``tests/test_corrchange.py``'s, the span standardized by its mean and
    biased standard deviation."""
    z = (s - s.mean(axis=0)) / s.std(axis=0)
    rho = float((z[:, 0] * z[:, 1]).mean())
    xi = z[:, 0] * z[:, 1] - rho - rho / 2 * (z[:, 0] ** 2 - 1 + z[:, 1] ** 2 - 1)
    return xi, z[:, 0] ** 2 - z[:, 1] ** 2


def _stream(n: int, seed: int) -> pl.DataFrame:
    """A correlation that steps between 0.2 and 0.8, and the first column's
    scale doubled on alternate stretches, so spans flag, withhold and stay
    quiet."""
    rng = np.random.default_rng([553, 1, seed])
    a, e = rng.standard_normal((2, n))
    i = np.arange(n)
    rho = np.where((i // 70) % 3 == 0, 0.2, 0.8)
    x = a * np.where((i // 110) % 2 == 0, 1.0, 2.0)
    y = rho * a + np.sqrt(1 - rho**2) * e
    return pl.DataFrame({"t": _clock(n, seed), "x0": x, "x1": y})


def _run(df: pl.DataFrame, **kw) -> pl.DataFrame:
    s = rp.spec.corrchange("c", features=["x0", "x1"], clock="t", gap_cap=10.0, **kw)
    return rp.ModelBank([s]).fit_predict(df)["c"].struct.unnest()


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("bandwidth", [1, 3])
def test_a_clock_span_is_statsmodels_cusum_on_its_rows(seed, bandwidth):
    """Each span closes on its last row, and only there: its statistic is
    ``max_j |Σ_{t≤j} ξ_t| / (√T D̂)``, ``D̂²`` statsmodels'
    ``S_hac_simple(ξ, nlags=γ − 1) / T``, and at ``γ = 1`` that is
    ``breaks_cusumolsresid(ξ, ddof=0)``'s statistic. A span on which both
    it and the same CUSUM of ``x̃² − ỹ²`` pass the critical value is
    withheld, by name; one on which only the first does flags."""
    df = _stream(1500, seed)
    d = 120.0
    out = _run(df, window_size=d, bandwidth=bandwidth, alpha_adjust="none")
    t, X = df["t"].to_numpy(), df.select("x0", "x1").to_numpy()
    spans = _spans(t, d)
    assert any(t[e] - t[s] == d for s, e in spans), "a span closes on its edge exactly"
    stat = out["stat"].to_numpy().astype(float)
    assert np.flatnonzero(np.isfinite(stat)).tolist() == [e for _, e in spans]
    reason = out["withheld_reason"].cast(pl.String)
    seen = set()
    for s, e in spans:
        xi, dd = _span_series(X[s : e + 1])
        rows = len(xi)
        lrv = S_hac_simple(xi, nlags=bandwidth - 1)[0, 0] / rows
        want = np.max(np.abs(np.cumsum(xi))) / (math.sqrt(rows) * math.sqrt(lrv))
        assert stat[e] == pytest.approx(want, rel=1e-12), (s, e)
        if bandwidth != 1:
            continue
        q, _, _ = breaks_cusumolsresid(xi, ddof=0)
        assert stat[e] == pytest.approx(q, rel=1e-12)
        r, _, _ = breaks_cusumolsresid(dd, ddof=0)
        crit = out["crit"][e]
        withheld = bool(q > crit and r > crit)
        flag = out["flag"][e]
        assert flag is (None if withheld else bool(q > crit)), (s, e, q, r, crit)
        assert reason[e] == ("scale_ratio_moved" if withheld else None)
        seen.add("withheld" if withheld else str(flag))
    print(seed, bandwidth, len(spans), seen)


@pytest.mark.parametrize("norm", ["l1", "linf"])
def test_a_clock_window_is_corrcoef_on_each_windows_rows(norm):
    """A window report reads the rows of ``(t − 2D, t − D]`` against those
    of ``(t − D, t]``, ``t`` the row's own clock: its statistic is the
    ``norm`` of the difference of numpy's ``corrcoef`` over each, on every
    row reported. Whole units put rows exactly at ``t − D`` and ``t − 2D``,
    the edges' closed and open sides."""
    n, d = 900, 40.0
    df = _stream(n, 7).with_columns(pl.Series("x2", np.random.default_rng(9).standard_normal(n)))
    s = rp.spec.corrchange(
        "c", features=["x0", "x1", "x2"], kind="window", window_size=d, crit=9.0, norm=norm,
        clock="t", gap_cap=10.0,
    )  # fmt: skip
    out = rp.ModelBank([s]).fit_predict(df)["c"].struct.unnest()
    t = df["t"].to_numpy()
    X = df.select("x0", "x1", "x2").to_numpy()
    iu = np.triu_indices(3, 1)
    stat = out["stat"].to_numpy().astype(float)
    on_edge = 0
    checked = 0
    for row in np.flatnonzero(np.isfinite(stat)):
        now = t[row]
        pre = (t > now - 2 * d) & (t <= now - d)
        post = (t > now - d) & (t <= now)
        on_edge += int(np.any(t == now - d) or np.any(t == now - 2 * d))
        diff = np.abs(np.corrcoef(X[pre].T)[iu] - np.corrcoef(X[post].T)[iu])
        want = diff.sum() if norm == "l1" else diff.max()
        assert stat[row] == pytest.approx(want, rel=1e-12), row
        checked += 1
    assert checked > n // 2 and on_edge > 50, (checked, on_edge)
