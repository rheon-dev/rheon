"""The study's runs: each feature against Polars, chunked with the state saved
and loaded between chunks, per group, and the state's bytes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from .features import (
    EwmMean,
    EwmMeanBy,
    EwmVar,
    Lag,
    Rolling,
    WindowSpec,
    alpha_of_half_life,
)
from .oracle import compare, make_frame, masked

KINDS = ("sum", "mean", "var", "min", "max")


@dataclass
class Unit:
    """One state per group: ``make()`` builds it, ``save``/``load`` move it
    through bytes, ``labels`` name its outputs and ``exprs`` are the Polars
    expressions (over ``xm``, before ``over``) each output is checked against."""

    name: str
    make: Callable[[], object]
    load: Callable[[bytes], object]
    labels: tuple[str, ...]
    exprs: tuple[pl.Expr, ...]
    timed: bool  # whether it reads the clock


def _lag(kind: str, n: int) -> Unit:
    return Unit(
        f"{kind}({n})",
        lambda: Lag(kind, n),
        lambda b: Lag.from_bytes(kind, n, b),
        (f"{kind}({n})",),
        (getattr(pl.col("xm"), kind)(n),),
        False,
    )


def _rolling_rows(w: int, min_samples: int | None = None, ddof: int = 1) -> Unit:
    ms = w if min_samples is None else min_samples
    spec = WindowSpec(rows=w, min_samples=ms, ddof=ddof)
    exprs = []
    for k in KINDS:
        kw: dict[str, object] = {"min_samples": ms}
        if k == "var":
            kw["ddof"] = ddof
        exprs.append(getattr(pl.col("xm"), f"rolling_{k}")(w, **kw))
    tag = f"w={w},m={ms}" + (f",ddof={ddof}" if ddof != 1 else "")
    return Unit(
        f"rolling[{tag}]",
        lambda: Rolling(spec, KINDS),
        lambda b: Rolling.from_bytes(spec, KINDS, b),
        tuple(f"rolling_{k}[{tag}]" for k in KINDS),
        tuple(exprs),
        False,
    )


def _rolling_by(span: int, closed: str, min_samples: int, ddof: int = 1) -> Unit:
    spec = WindowSpec(span=span, closed=closed, min_samples=min_samples, ddof=ddof)
    exprs = []
    for k in KINDS:
        kw: dict[str, object] = {"min_samples": min_samples, "closed": closed}
        if k == "var":
            kw["ddof"] = ddof
        exprs.append(getattr(pl.col("xm"), f"rolling_{k}_by")("t", f"{span}i", **kw))
    tag = f"{span}i,{closed},m={min_samples}"
    return Unit(
        f"rolling_by[{tag}]",
        lambda: Rolling(spec, KINDS),
        lambda b: Rolling.from_bytes(spec, KINDS, b),
        tuple(f"rolling_{k}_by[{tag}]" for k in KINDS),
        tuple(exprs),
        True,
    )


def _ewm_mean(
    hl: float, adjust: bool = True, ignore_nulls: bool = False, min_samples: int = 1
) -> Unit:
    a = alpha_of_half_life(hl)

    def make() -> EwmMean:
        return EwmMean(a, adjust, min_samples, ignore_nulls)

    def load(b: bytes) -> EwmMean:
        s = make()
        s.load(b)
        return s

    tag = f"hl={hl:g},adjust={adjust},ignore_nulls={ignore_nulls},m={min_samples}"
    return Unit(
        f"ewm_mean[{tag}]",
        make,
        load,
        (f"ewm_mean[{tag}]",),
        (
            pl.col("xm").ewm_mean(
                half_life=hl, adjust=adjust, ignore_nulls=ignore_nulls, min_samples=min_samples
            ),
        ),
        False,
    )


def _ewm_var(
    hl: float, std: bool, adjust: bool = True, bias: bool = False, ignore_nulls: bool = False
) -> Unit:
    a = alpha_of_half_life(hl)

    def make() -> EwmVar:
        return EwmVar(a, adjust, bias, 1, ignore_nulls, std)

    def load(b: bytes) -> EwmVar:
        s = make()
        s.load(b)
        return s

    op = "ewm_std" if std else "ewm_var"
    tag = f"hl={hl:g},adjust={adjust},bias={bias},ignore_nulls={ignore_nulls}"
    return Unit(
        f"{op}[{tag}]",
        make,
        load,
        (f"{op}[{tag}]",),
        (
            getattr(pl.col("xm"), op)(
                half_life=hl, adjust=adjust, bias=bias, ignore_nulls=ignore_nulls
            ),
        ),
        False,
    )


def _ewm_mean_by(hl: int) -> Unit:
    def load(b: bytes) -> EwmMeanBy:
        s = EwmMeanBy(hl)
        s.load(b)
        return s

    return Unit(
        f"ewm_mean_by[{hl}i]",
        lambda: EwmMeanBy(hl),
        load,
        (f"ewm_mean_by[{hl}i]",),
        (pl.col("xm").ewm_mean_by("t", half_life=f"{hl}i"),),
        True,
    )


def units() -> list[Unit]:
    return [
        _lag("shift", 1),
        _lag("shift", 5),
        _lag("diff", 1),
        _lag("diff", 5),
        _lag("pct_change", 2),
        _rolling_rows(20),
        _rolling_rows(500, min_samples=1),
        _rolling_rows(7, min_samples=3, ddof=0),
        _rolling_by(50, "right", 1),
        _rolling_by(200, "left", 3),
        _rolling_by(30, "both", 2, ddof=0),
        _rolling_by(30, "none", 1),
        _rolling_by(12, "right", 4),  # often short of min_samples: skipped updates
        _rolling_by(
            3, "left", 0
        ),  # rolling_sum_by's default min_samples: an empty window sums to 0
        _ewm_mean(10.0),
        _ewm_mean(10.0, adjust=False),
        _ewm_mean(4.0, ignore_nulls=True, min_samples=5),
        _ewm_var(10.0, std=True),
        _ewm_var(10.0, std=False, adjust=False, bias=True),
        _ewm_var(3.0, std=True, adjust=False, ignore_nulls=True),
        _ewm_mean_by(25),
    ]


# ---------------------------------------------------------------------------


def polars_outputs(df: pl.DataFrame, us: list[Unit], grouped: bool = True) -> dict[str, pl.Series]:
    exprs = []
    for u in us:
        for lab, e in zip(u.labels, u.exprs, strict=False):
            exprs.append((e.over("g") if grouped else e).alias(lab))
    out = df.select(exprs)
    return {c: out[c] for c in out.columns}


def stream(
    df: pl.DataFrame,
    us: list[Unit],
    cuts: list[int] | None = None,
    grouped: bool = True,
) -> tuple[dict[str, list[float | None]], dict[str, int]]:
    """Feed the rows in order, one state per (unit, group). At each cut, every
    state is saved to bytes, dropped, and loaded from the bytes: the next chunk
    sees only what the bytes carry. Returns the outputs and the largest saved
    size of each unit's state, per group."""
    xs = df["xm"].to_list()
    ts = df["t"].to_list()
    gs = df["g"].to_list() if grouped else [0] * len(xs)
    cutset = set(cuts or [])
    states: list[dict[int, object]] = [{} for _ in us]
    outs: dict[str, list[float | None]] = {lab: [] for u in us for lab in u.labels}
    lists = [[outs[lab] for lab in u.labels] for u in us]
    peak = {u.name: 0 for u in us}
    for i, (x, t, g) in enumerate(zip(xs, ts, gs, strict=False)):
        if i in cutset:
            for j, u in enumerate(us):
                for key, st in states[j].items():
                    b = st.to_bytes()  # type: ignore[attr-defined]
                    peak[u.name] = max(peak[u.name], len(b))
                    states[j][key] = u.load(b)
        for j, u in enumerate(us):
            st = states[j].get(g)
            if st is None:
                st = states[j][g] = u.make()
            r = st.step(x, t)  # type: ignore[attr-defined]
            if len(u.labels) == 1:
                lists[j][0].append(r)
            else:
                for lst, v in zip(lists[j], r, strict=False):
                    lst.append(v)
    return outs, peak


def random_cuts(n: int, k: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(set(rng.integers(1, n, k).tolist()))


def frame(n: int, nonfinite: bool = False, groups: int = 5) -> pl.DataFrame:
    return masked(make_frame(n, groups=groups, nonfinite=nonfinite))


def check_all(
    df: pl.DataFrame, us: list[Unit], cuts: list[int] | None, grouped: bool = True
) -> tuple[dict[str, dict[str, float | int]], dict[str, int]]:
    ours, peak = stream(df, us, cuts, grouped)
    theirs = polars_outputs(df, us, grouped)
    return {lab: compare(ours[lab], theirs[lab]) for lab in ours}, peak
