"""``spectral`` through the bank: its power against scipy's ``periodogram``
of the same rows with the decay's amplitudes as the window, on an even
clock; against the definition written out with numpy on an uneven one; and
what the bank owes every model -- out-of-sample reads, chunk invariance, a
save mid-stream, zero weights and nulls, a break, its refusals and its
output schema (task 491).
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from scipy.signal import periodogram

from rheon import polars as rp

TIER = "essential"


def _signal(n: int, seed: int, step: float = 1.0) -> pl.DataFrame:
    """A level, cycles of period 8 and 3.2 clock units and noise, a row
    every ``step`` clock units."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * step
    y = 3.0 + np.sin(2 * np.pi * t / 8) + 0.5 * np.sin(2 * np.pi * t / 3.2 + 1) + rng.normal(size=n)
    return pl.DataFrame({"t": t, "y": y})


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("s")


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
@pytest.mark.parametrize("h", [float("inf"), 120.0])
def test_the_power_is_scipys_periodogram_with_the_decay_as_window(scaling, h):
    """On an even clock of spacing 0.5 (fs = 2), the power before row ``n``
    is ``periodogram(x - m, fs, window=a, detrend=False, scaling=...)`` of
    the rows before it, ``a`` the decay's amplitudes and ``m`` their
    weighted mean, at frequencies on its grid: the zero frequency, two
    others and the Nyquist."""
    n, step = 1001, 0.5
    df = _signal(n, 1, step)
    rows = n - 1
    grid = np.fft.rfftfreq(rows, d=step)
    ks = [0, 37, 250, rows // 2]
    periods = [float("inf") if k == 0 else 1.0 / grid[k] for k in ks]
    periods = [p for p in periods if np.isfinite(p)]
    spec = rp.spec.spectral(
        "s", targets=["y"], periods=periods, spacing=step, scaling=scaling,
        clock="t", gap_cap=10.0, half_life=h,
    )  # fmt: skip
    last = _out(spec, df)[-1]
    x = df["y"].to_numpy()[:rows]
    age = (rows - 1 - np.arange(rows)) * step
    a = 0.5 ** (age / h)
    m = np.sum(a * x) / np.sum(a)
    f, p = periodogram(x - m, fs=1 / step, window=a, detrend=False, scaling=scaling)
    for j, k in enumerate(ks[1:]):
        got = last[f"power_y_p{j}"][0]
        assert got == pytest.approx(p[k], rel=1e-9), (k, got, p[k])


def test_on_an_uneven_clock_it_is_the_definition():
    """With uneven steps and weights, the power is the definition written out
    with numpy: ``c |Σ a (x - m) e^{-2πi f τ}|² spacing / Σ a²``."""
    n, h = 800, 60.0
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(0.5, 1.5, n))
    w = rng.uniform(0.5, 1.5, n)
    y = np.sin(2 * np.pi * t / 8) + rng.normal(size=n)
    df = pl.DataFrame({"t": t, "y": y, "w": w})
    periods = [8.0, 3.0]
    spec = rp.spec.spectral(
        "s", targets=["y"], periods=periods, clock="t", gap_cap=10.0, weight="w", half_life=h
    )
    out = _out(spec, df)
    for end in (300, n - 1):
        got = [out[f"power_y_p{j}"][end] for j in range(2)]
        tau, x, ww = t[:end], y[:end], w[:end]
        a = ww * 0.5 ** ((tau[-1] - tau) / h)
        m = np.sum(a * x) / np.sum(a)
        for g, p in zip(got, periods, strict=True):
            z = np.sum(a * (x - m) * np.exp(-2j * np.pi * tau / p))
            assert g == pytest.approx(2 * abs(z) ** 2 / np.sum(a * a), rel=1e-9), end


def test_a_cycle_stands_out():
    df = _signal(3000, 3)
    spec = rp.spec.spectral("s", targets=["y"], periods=[8.0, 5.0, 3.2], half_life=500.0)
    last = _out(spec, df)[-1]
    p8, p5, p32 = (last[f"power_y_p{j}"][0] for j in range(3))
    assert p8 > 20 * p5 and p32 > 5 * p5


@pytest.mark.parametrize("cut", [1, 333, 998])
def test_chunks_and_a_save_go_on_to_the_bit(cut):
    df = _signal(1000, 4).with_columns(
        pl.Series("g", np.where(np.arange(1000) % 3 == 0, "a", "b")),
        pl.Series("w", np.where(np.arange(1000) % 11 == 4, 0.0, 1.5)),
    )
    s = rp.spec.spectral(
        "s", targets=["y"], periods=[8.0, 3.2], group="g", weight="w", half_life=40.0
    )
    one = rp.ModelBank([s]).fit_predict(df)
    a = rp.ModelBank([s])
    first = a.fit_predict(df.slice(0, cut))
    buf = io.BytesIO()
    a.save(buf)
    b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
    rest = df.slice(cut)
    got = a.fit_predict(rest)
    assert got.equals(b.fit_predict(rest), null_equal=True)
    assert pl.concat([first, got]).equals(one, null_equal=True)


def test_a_zero_weight_row_counts_in_no_clock():
    """Rows of weight 0, wild values among them, leave every other row's
    output as the stream without them gives it, to the bit, under decay on
    an uneven clock; they are read."""
    n = 900
    rng = np.random.default_rng(5)
    zero = rng.uniform(size=n) < 0.1
    zero[0] = True
    df = _signal(n, 6).with_columns(
        pl.Series("t", np.cumsum(rng.uniform(0.5, 1.5, n))),
        pl.Series("w", np.where(zero, 0.0, 1.0)),
        pl.when(pl.Series(zero)).then(1e6).otherwise(pl.col("y")).alias("y"),
    )
    s = rp.spec.spectral(
        "s", targets=["y"], periods=[8.0, 3.0], clock="t", gap_cap=10.0, weight="w",
        half_life=50.0,
    )  # fmt: skip
    full, thin = _out(s, df), _out(s, df.filter(~pl.Series(zero)))
    keep = full.filter(~pl.Series(zero))
    for f in ("power_y_p0", "power_y_p1", "weight_sum"):
        assert keep[f].equals(thin[f], null_equal=True), f
    assert full["power_y_p0"].filter(pl.Series(zero)).drop_nulls().len() > 50


def test_a_session_change_empties_the_transform():
    df = _signal(600, 7).with_columns(pl.Series("ses", np.repeat(["a", "b"], 300)))
    s = rp.spec.spectral(
        "s", targets=["y"], periods=[8.0], clock="t", gap_cap=10.0, session="ses",
        session_gap=1.0, half_life=float("inf"),
    )  # fmt: skip
    out = _out(s, df)
    assert out["power_y_p0"][300] is None and out["power_y_p0"][301] is not None
    second = _out(
        rp.spec.spectral("s", targets=["y"], periods=[8.0], half_life=float("inf")),
        df.slice(300),
    )
    np.testing.assert_allclose(
        out["power_y_p0"][305:].to_numpy(), second["power_y_p0"][5:].to_numpy(), rtol=1e-9
    )


def test_names_match_the_realized_struct():
    df = _signal(100, 8).with_columns(pl.col("y").alias("z"))
    spec = rp.spec.spectral("s", targets=["y", "z"], periods=[8.0, 4.0], half_life=50.0)
    out = rp.ModelBank([spec]).fit_predict(df)
    assert [f.name for f in out.schema["s"].fields] == rp.spec.output_fields(spec)
    assert rp.spec.coef_index(spec).height == 0


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"periods": []}, "at least one"),
        ({"periods": [1.9]}, "Nyquist"),
        ({"periods": [8.0], "spacing": 0.0}, "spacing"),
        ({"periods": [-8.0]}, "above 0"),
        ({"periods": [8.0], "max_state_bytes": 0}, "above 0"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.spec.spectral("s", targets=["y"], half_life=10.0, **kw)


def test_the_nyquist_itself_is_taken_and_features_are_refused():
    rp.spec.spectral("s", targets=["y"], periods=[2.0], half_life=10.0)
    with pytest.raises(TypeError, match="features"):
        rp.spec.spectral("s", targets=["y"], periods=[8.0], features=["x"])  # type: ignore[call-arg]
