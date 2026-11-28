"""``spectral`` against scipy's own windows and normalizations.

``tests/test_spectral.py`` feeds scipy the decay's amplitudes as the window
and removes the weighted mean itself. Here scipy does both its own way:

- a **rectangular window** (no decay, unit weights) is scipy's
  ``periodogram(x, fs, window="boxcar", detrend="constant",
  return_onesided=True)``, its mean removed by scipy, at every frequency of
  its grid, the Nyquist among them on an even count of rows and not on an
  odd one, under both ``scaling="density"`` (``c |X|² spacing / N``) and
  ``"spectrum"`` (``c |X|² / N²``), ``c`` 2 but 1 at the Nyquist;
- **without detrending**, the decayed amplitudes as the window, scipy's
  ``periodogram(..., detrend=False)`` of the raw rows;
- the **Lomb-Scargle** power on a rectangular window is scipy's
  ``lombscargle(t, x, 2πf, normalize="normalize", floating_mean=detrend)``
  with no weights given: the explained share of the sum of squares about
  the constant (``floating_mean=True``) or about 0 (``False``);
- the Lomb-Scargle **spread threshold**: the power is withheld where its
  two columns' determinant ``CC·SS − CS²`` is at or below ``1e-10`` of
  ``(CC + SS)² / 4``, and reported, as scipy's, just above it.

Each comparison is held to a bound derived from both computations'
rounding, written beside it, with the headroom measured.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.signal import lombscargle, periodogram

from rheon import polars as rp

TIER = "essential"

U = 2.0**-53


def _signal(n: int, seed: int, step: float) -> pl.DataFrame:
    """A level, cycles of period 8 and 3.2 clock units and noise, a row
    every ``step`` clock units."""
    rng = np.random.default_rng([17, seed])
    t = np.arange(n) * step
    y = 3.0 + np.sin(2 * np.pi * t / 8) + 0.5 * np.sin(2 * np.pi * t / 3.2 + 1) + rng.normal(size=n)
    return pl.DataFrame({"t": t, "y": y})


def _powers(spec: dict, df: pl.DataFrame, n_periods: int) -> np.ndarray:
    """The power at each period on the stream's last row, which reads every
    row before it."""
    last = rp.ModelBank([spec]).fit_predict(df).unnest("s")[-1]
    return np.array([last[f"power_y_p{j}"][0] for j in range(n_periods)], dtype=float)


def _transform_bound(rows: int, mass: float) -> float:
    """The most the transform ``X_k`` of ``rows`` rows can be off, against
    the exact one, by the rotating recursion's rounding: each row's term is
    turned by ``r_k`` once a row after it, so the oldest is turned ``rows``
    times, each turn one complex product (at most ``2√2 u``, Higham 3.6) of a
    rotation ``libm`` gives within one ulp in each part, at a frequency
    ``1 / (1 / f)`` within two; so a term's turns move it by at most ``8 u``
    a turn, times the turns. ``mass`` is ``Σ a |x|`` over the rows; the
    weighted mean's removal, ``X − m U``, doubles it. scipy's FFT
    (pocketfft) is off by ``O(u log N)`` of ``‖x‖``, far below this."""
    return 16.0 * rows * U * mass


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
@pytest.mark.parametrize(("rows", "step"), [(1000, 0.5), (999, 0.25), (512, 1.0)])
def test_a_rectangular_window_is_scipys_boxcar_periodogram(scaling, rows, step):
    """With no decay and unit weights the amplitudes are a rectangular
    window, so the power before row ``rows`` is scipy's ``periodogram`` of
    the rows before it with ``window="boxcar"`` and scipy's own
    ``detrend="constant"``, one-sided, at every frequency on its grid above
    0. On an even count the grid's last frequency is the Nyquist, where
    scipy does not double; on an odd one it is below it, where it does.

    Bound (``_transform_bound``): ``|P − P'| ≤ s c (2 |X'| δ + δ²)``, ``s``
    the scaling's factor and ``δ`` the transform's bound. Measured
    2026-11-24 (x86-64, Linux): at most 1.9e-3 of it."""
    df = _signal(rows + 1, rows, step)
    x = df["y"].to_numpy()[:rows]
    f, want = periodogram(
        x, fs=1 / step, window="boxcar", detrend="constant", scaling=scaling,
        return_onesided=True,
    )  # fmt: skip
    ks = np.arange(1, len(f))
    assert (rows % 2 == 0) == (f[-1] == 1 / (2 * step))
    spec = rp.spec.spectral(
        "s", targets=["y"], periods=list(1.0 / f[ks]), spacing=step, scaling=scaling,
        clock="t", gap_cap=10.0, half_life=float("inf"),
    )  # fmt: skip
    got = _powers(spec, df, len(ks))
    # The exact transform's modulus, read back from scipy's power.
    c = np.where(f[ks] == 1 / (2 * step), 1.0, 2.0)
    s = step / rows if scaling == "density" else 1.0 / rows**2
    modulus = np.sqrt(want[ks] / (c * s))
    delta = _transform_bound(rows, float(np.sum(np.abs(x))))
    bound = s * c * (2 * modulus * delta + delta**2)
    gap = np.abs(got - want[ks])
    assert np.all(gap <= bound), (np.argmax(gap / bound), np.max(gap / bound))
    # The bound is far inside a factor of 2, so the doubling is held: a
    # power doubled at the Nyquist, or not doubled below it, fails.
    assert np.median(bound / want[ks]) < 1e-6


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
def test_without_detrending_it_is_scipys_periodogram_of_the_raw_rows(scaling):
    """``detrend=False`` transforms the raw rows: scipy's ``periodogram``
    with ``detrend=False`` and the decay's amplitudes as the window. Under
    a window that is not rectangular the level leaks into every frequency,
    so the read differs from the detrended one by far more than the bound,
    which is ``_transform_bound`` over ``Σ a |x|``. Measured 2026-11-24:
    at most 9e-4 of it."""
    rows, step, h = 800, 0.5, 60.0
    df = _signal(rows + 1, 3, step)
    x = df["y"].to_numpy()[:rows]
    a = 0.5 ** ((rows - 1 - np.arange(rows)) * step / h)
    f, want = periodogram(x, fs=1 / step, window=a, detrend=False, scaling=scaling)
    ks = np.array([3, 25, 50, 101, 250, rows // 2])
    kw = dict(
        targets=["y"], periods=list(1.0 / f[ks]), spacing=step, scaling=scaling, clock="t",
        gap_cap=10.0, half_life=h,
    )  # fmt: skip
    raw = _powers(rp.spec.spectral("s", detrend=False, **kw), df, len(ks))
    centred = _powers(rp.spec.spectral("s", detrend=True, **kw), df, len(ks))
    c = np.where(ks == rows // 2, 1.0, 2.0)
    s = step / np.sum(a * a) if scaling == "density" else 1.0 / np.sum(a) ** 2
    modulus = np.sqrt(want[ks] / (c * s))
    delta = _transform_bound(rows, float(np.sum(a * np.abs(x))))
    bound = s * c * (2 * modulus * delta + delta**2)
    gap = np.abs(raw - want[ks])
    assert np.all(gap <= bound), np.max(gap / bound)
    # The control: the detrended read is another number, past the bound.
    assert np.any(np.abs(centred - want[ks]) > 100 * bound)


def _ls_bound(rows: int, spread: float) -> float:
    """Lomb-Scargle's bound, relative to 1, the power's scale: its sums are
    ``rows`` rotating terms each, off by ``_transform_bound`` of their own
    mass (relative ``16 rows u``); the power is a ratio of 2 × 2 forms whose
    solve amplifies a relative error by the columns' condition, about ``4 /
    spread`` for ``spread = (CC·SS − CS²) / ((CC + SS)² / 4)``; scipy's sums
    are direct, off by ``rows u``. Three such errors enter: ``48 rows u /
    spread``, doubled for scipy's side."""
    return 96.0 * rows * U * 4.0 / spread


def _spread(t: np.ndarray, period: float, detrend: bool) -> float:
    """The columns' spread at ``period`` over clocks ``t`` read from the last
    one, from the definition: the weighted (here unit) cosine and sine
    columns' determinant over the square of half their trace, each centred
    under ``detrend``."""
    th = 2 * np.pi * (t[-1] - t) / period
    c, s = np.cos(th), np.sin(th)
    if detrend:
        c, s = c - c.mean(), s - s.mean()
    cc, ss, cs = c @ c, s @ s, c @ s
    return float((cc * ss - cs * cs) / ((cc + ss) ** 2 / 4))


@pytest.mark.parametrize("detrend", [True, False])
def test_lomb_scargle_on_a_rectangular_window_is_scipys_unweighted(detrend):
    """With no decay and unit weights, the Lomb-Scargle power before row
    ``end`` is scipy's ``lombscargle`` with no ``weights`` given, its
    ``normalize="normalize"`` power (the explained share of the sum of
    squares about the fit's constant under ``floating_mean=True``, about 0
    under ``False``), on an uneven clock and at a period below twice the
    mean step. Bound: ``_ls_bound`` at each period's own spread. Measured
    2026-11-24: at most 8e-5 of it, since the bound takes each sum's worst
    case, which grows with the rows, where the realized error grows as their
    square root."""
    n = 700
    rng = np.random.default_rng(31)
    t = np.cumsum(rng.uniform(0.2, 1.8, n))
    y = 1.5 + np.sin(2 * np.pi * t / 8) + rng.normal(size=n)
    df = pl.DataFrame({"t": t, "y": y})
    periods = [8.0, 21.0, 1.3, 55.0]
    spec = rp.spec.spectral(
        "s", targets=["y"], periods=periods, method="lomb_scargle", detrend=detrend,
        clock="t", gap_cap=10.0, half_life=float("inf"),
    )  # fmt: skip
    out = rp.ModelBank([spec]).fit_predict(df).unnest("s")
    for end in (150, n - 1):
        tau, x = t[:end], y[:end]
        want = lombscargle(
            tau, x, 2 * np.pi / np.array(periods), normalize="normalize",
            floating_mean=detrend,
        )  # fmt: skip
        for j, p in enumerate(periods):
            got = out[f"power_y_p{j}"][end]
            bound = _ls_bound(end, _spread(tau, p, detrend))
            assert abs(got - want[j]) <= bound, (end, p, got, want[j], bound)


def _period_at_spread(t: np.ndarray, target: float) -> float:
    """The period whose columns' spread over ``t`` (not centred) is
    ``target``: the spread falls as the period grows past the clocks' span,
    so a bisection on the period's log finds it."""
    lo, hi = np.log(t[-1] - t[0]), np.log(1e12 * (t[-1] - t[0]))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _spread(t, float(np.exp(mid)), False) > target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def test_lomb_scargle_is_withheld_at_the_spread_threshold_and_scipys_above_it():
    """The two columns are told apart only once the phases have spread over
    enough of a cycle: the power is withheld at a spread of ``1e-10`` and
    below, and reported above it. A period three times either side of the
    threshold's (in spread) falls on each side: the spread is computed to
    about ``u / spread``, 1e-6 relative at the threshold, in the model's
    sums and in this test's. Above it, the power is scipy's within
    ``_ls_bound``, which at that spread is loose (6e-2) because the solve
    is that ill-conditioned for both sides; measured 2026-11-24 at 9.5e-7
    from scipy's, 2e-5 of the bound. Without detrending, whose centring
    would leave the columns' spread to rounding at these periods."""
    n = 400
    rng = np.random.default_rng(41)
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    y = 1.5 + 0.1 * t + rng.normal(size=n)
    df = pl.DataFrame({"t": t, "y": y})
    tau = t[: n - 1]
    above = _period_at_spread(tau, 3e-10)
    below = _period_at_spread(tau, 3e-11)
    assert _spread(tau, above, False) > 1e-10 > _spread(tau, below, False)
    spec = rp.spec.spectral(
        "s", targets=["y"], periods=[above, below], method="lomb_scargle", detrend=False,
        clock="t", gap_cap=10.0, half_life=float("inf"),
    )  # fmt: skip
    last = rp.ModelBank([spec]).fit_predict(df).unnest("s")[-1]
    got_above, got_below = last["power_y_p0"][0], last["power_y_p1"][0]
    assert got_below is None
    assert got_above is not None
    want = float(lombscargle(tau, y[: n - 1], np.array([2 * np.pi / above]), normalize="normalize"))
    assert abs(got_above - want) <= _ls_bound(n - 1, 3e-10), (got_above, want)
