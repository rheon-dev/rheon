"""Part 2: the exponentially weighted features against Polars' ``ewm_*``, the
carried alpha across a simulated change of libm, and the bias of the decayed
min and max envelopes against Polars' ``rolling_max`` / ``rolling_min``."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl

from .common import compare, masked, stream_values, ulps
from .ew import DecayedMax, EwmMean, EwmMeanBy, EwmVar, PeakEnvelope, alpha_of_half_life

Factory = Callable[[], object]


def ew_features() -> list[tuple[str, Factory, pl.Expr]]:
    """Each feature: its name, a factory for one group's state, and the Polars
    expression on the masked column. The settings are study 532's and more."""
    x = pl.col("xm")
    a = alpha_of_half_life
    out: list[tuple[str, Factory, pl.Expr]] = []
    for hl, adjust, ign, m in (
        (10, True, False, 1),
        (10, False, False, 1),
        (4, True, True, 5),
        (100, False, True, 1),
    ):
        out.append(
            (
                f"ewm_mean[hl={hl},adjust={adjust},ignore_nulls={ign},m={m}]",
                lambda hl=hl, adjust=adjust, ign=ign, m=m: EwmMean(a(hl), adjust, m, ign),
                x.ewm_mean(half_life=hl, adjust=adjust, ignore_nulls=ign, min_samples=m),
            )
        )
    for kind, hl, adjust, bias, ign, m in (
        ("std", 10, True, False, False, 1),
        ("var", 10, False, True, False, 1),
        ("std", 3, False, False, True, 1),
        ("var", 25, True, False, True, 4),
        ("std", 50, True, True, False, 2),
    ):
        expr = (x.ewm_std if kind == "std" else x.ewm_var)(
            half_life=hl, adjust=adjust, bias=bias, ignore_nulls=ign, min_samples=m
        )
        out.append(
            (
                f"ewm_{kind}[hl={hl},adjust={adjust},bias={bias},ignore_nulls={ign},m={m}]",
                lambda hl=hl, adjust=adjust, bias=bias, ign=ign, m=m, kind=kind: EwmVar(
                    a(hl), adjust, bias, m, ign, std=(kind == "std")
                ),
                expr,
            )
        )
    for hl in (25, 3):
        out.append(
            (
                f"ewm_mean_by[{hl}i]",
                lambda hl=hl: EwmMeanBy(hl),
                x.ewm_mean_by("t", half_life=f"{hl}i"),
            )
        )
    return out


def stream(df: pl.DataFrame, factory: Factory) -> list:
    gs, ts, xs = stream_values(df)
    states: dict[int, object] = {}
    out = []
    for g, t, x in zip(gs, ts, xs, strict=True):
        st = states.get(g)
        if st is None:
            st = states[g] = factory()
        out.append(st.step(x, t))  # type: ignore[attr-defined]
    return out


def vs_polars(df: pl.DataFrame) -> list[tuple[str, dict]]:
    pol = masked(df).select([e.over("g").alias(name) for name, _, e in ew_features()])
    return [(name, compare(stream(df, f), pol[name])) for name, f, _ in ew_features()]


# ---------------------------------------------------------------------------
# The carried alpha.


def _single_series(n: int, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(1.0e3 + np.cumsum(rng.standard_t(3, n)) + 0.25 * rng.standard_normal(n))


def carried_alpha(n: int = 20000) -> list[list]:
    """Save at the middle row; continue (a) from the bytes, alpha carried, and
    (b) with alpha recomputed on a machine whose libm gives one ulp more or
    less for it. Both against the uncut stream over the second half."""
    xs = _single_series(n)
    rows = []
    for name, make in (
        ("ewm_mean adjust=True hl=10", lambda a: EwmMean(a, True)),
        ("ewm_mean adjust=False hl=10", lambda a: EwmMean(a, False)),
        ("ewm_var adjust=False bias hl=10", lambda a: EwmVar(a, False, True)),
        ("ewm_std adjust=True hl=10", lambda a: EwmVar(a, True, False, std=True)),
        ("max envelope hl=10", lambda a: PeakEnvelope(a, True)),
    ):
        alpha = alpha_of_half_life(10)
        whole = make(alpha)
        ref = [whole.step(x) for x in xs]
        first = make(alpha)
        for x in xs[: n // 2]:
            first.step(x)
        saved = first.to_bytes()
        carried = make(0.5)  # whatever the spec would recompute; the load replaces it
        carried.load(saved)
        out_c = [carried.step(x) for x in xs[n // 2 :]]
        gaps = []
        e = math.exp(-math.log(2.0) / 10)
        for moved_alpha in (
            math.nextafter(alpha, math.inf),
            math.nextafter(alpha, -math.inf),
            1.0 - math.nextafter(e, math.inf),
            1.0 - math.nextafter(e, -math.inf),
        ):
            moved = make(alpha)
            moved.load(saved)
            if isinstance(moved, PeakEnvelope):
                moved.r = 1.0 - moved_alpha
            else:
                moved.alpha = moved_alpha  # type: ignore[attr-defined]
            out_m = [moved.step(x) for x in xs[n // 2 :]]
            gaps.append(int(ulps(np.array(out_m), np.array(ref[n // 2 :])).max()))
        u_c = int(ulps(np.array(out_c), np.array(ref[n // 2 :])).max())
        rows.append([name, len(saved), u_c, *gaps])
    return rows


# ---------------------------------------------------------------------------
# The envelopes against the rolling extremes.


def envelope_bias(n: int = 20000) -> list[list]:
    """The max envelope minus ``rolling_max(w)`` (and the min envelope minus
    ``rolling_min(w)``), in units of ``rolling_std(w)``, on two series: the
    random walk about 1e3 and white noise N(0, 1)."""
    rng = np.random.default_rng(11)
    series = {
        "random walk": np.array(_single_series(n)),
        "white noise": rng.standard_normal(n),
    }
    rows = []
    for sname, xs in series.items():
        s = pl.Series(xs)
        for w in (20, 100):
            rmax = s.rolling_max(w).to_numpy()
            rmin = s.rolling_min(w).to_numpy()
            rsd = s.rolling_std(w).to_numpy()
            for hl in (w / 8, w / 4, w / 2):
                a = alpha_of_half_life(hl)
                emax, emin = PeakEnvelope(a, True), PeakEnvelope(a, False)
                ymax = np.array([emax.step(float(x)) for x in xs])
                ymin = np.array([emin.step(float(x)) for x in xs])
                ok = np.isfinite(rmax) & (rsd > 0)
                ok[: 5 * w] = False  # past the start-up
                dmax = (ymax - rmax)[ok] / rsd[ok]
                dmin = (ymin - rmin)[ok] / rsd[ok]
                rows.append(
                    [
                        sname,
                        w,
                        f"{hl:g}",
                        float(np.mean(dmax)),
                        float(np.quantile(dmax, 0.05)),
                        float(np.quantile(dmax, 0.95)),
                        f"{100 * np.mean(dmax > 0):.1f}%",
                        float(np.mean(dmin)),
                        float(np.corrcoef(ymax[ok], rmax[ok])[0, 1]),
                    ]
                )
    return rows


def decayed_max_rows(n: int = 20000) -> list[list]:
    """Cormode et al.'s decayed max against ``rolling_max(w)``: on abs(N(0,1))
    (non-negative, where it is meant to be used) and on the walk about 1e3."""
    rng = np.random.default_rng(12)
    rows = []
    for sname, xs in (
        ("abs(white noise)", np.abs(rng.standard_normal(n))),
        ("random walk", np.array(_single_series(n))),
    ):
        s = pl.Series(xs)
        for w, hl in ((20, 5.0), (100, 25.0)):
            rmax = s.rolling_max(w).to_numpy()
            rsd = s.rolling_std(w).to_numpy()
            dm = DecayedMax(alpha_of_half_life(hl))
            y = np.array([dm.step(float(x)) for x in xs])
            ok = np.isfinite(rmax) & (rsd > 0)
            ok[: 5 * w] = False
            d = (y - rmax)[ok] / rsd[ok]
            rows.append(
                [
                    sname,
                    w,
                    f"{hl:g}",
                    float(np.mean(d)),
                    float(np.quantile(d, 0.05)),
                    float(np.quantile(d, 0.95)),
                    f"{100 * np.mean(y[ok] == xs[ok]):.1f}%",
                ]
            )
    return rows
