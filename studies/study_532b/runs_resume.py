"""Part 4: chunk and resume invariance, and the cost per row.

Every feature's state is saved to bytes, dropped and loaded from the bytes at
every cut, for every group. The next chunk sees nothing else. The output is
compared with the uncut stream (every feature), and with one Polars pass
where the feature equals a Polars expression: the EW features and lags on
every row; the bucketed count, min and max on the rows where the B buckets
are the span ``(t - B b, t]`` exactly."""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from .bucket import Bucketed, BucketSpec
from .common import compare, masked, stream_values
from .ew import DecayedMax, PeakEnvelope, alpha_of_half_life
from .lag import Lag
from .runs_ew import ew_features
from .runs_lag import _cuts, _expr


@dataclass
class Feature:
    name: str
    new: Callable[[], object]
    step: Callable[[object, object, int], object]
    save: Callable[[object], bytes]
    load: Callable[[bytes], object]
    oracle: pl.Expr | None = None  # Polars on the masked column, every row
    aligned: tuple[int, int] | None = None  # (b, B): compare count/min/max on aligned rows


def _bucket_feature(b: int, B: int, interp: bool) -> Feature:
    spec = BucketSpec(width=b, buckets=B, keep_previous=interp)

    def step(st: object, x: object, t: int) -> object:
        assert isinstance(st, Bucketed)
        st.step(x, t)  # type: ignore[arg-type]
        r = st.read()
        out = tuple(r[s] for s in ("count", "sum", "mean", "var", "min", "max"))
        if interp:
            ri = st.read_interpolated(t)
            out = out + (ri["count"], ri["sum"], ri["mean"])
        return out

    return Feature(
        f"bucketed[b={b},B={B}{',interp' if interp else ''}]",
        lambda: Bucketed(spec),
        step,
        lambda st: st.to_bytes(),  # type: ignore[attr-defined]
        lambda data: Bucketed.from_bytes(spec, data),
        aligned=(b, B),
    )


def _obj_feature(
    name: str, new: Callable[[], object], oracle: pl.Expr | None, load_into: Callable[[], object]
) -> Feature:
    def load(data: bytes) -> object:
        st = load_into()
        st.load(data)  # type: ignore[attr-defined]
        return st

    return Feature(
        name,
        new,
        lambda st, x, t: st.step(x, t),  # type: ignore[attr-defined]
        lambda st: st.to_bytes(),  # type: ignore[attr-defined]
        load,
        oracle,
    )


def features() -> list[Feature]:
    out = [
        _bucket_feature(5, 12, False),
        _bucket_feature(20, 12, True),
        _bucket_feature(60, 4, True),
    ]
    for name, factory, expr in ew_features():
        # The state loaded into is made with a wrong alpha: the load must replace it.
        if name.startswith("ewm_mean_by"):
            wrong = factory
        else:

            def wrong(factory: Callable[[], object] = factory) -> object:
                st = factory()
                st.alpha = 0.5  # type: ignore[attr-defined]
                return st

        out.append(_obj_feature(name, factory, expr, wrong))
    a = alpha_of_half_life(10)
    out.append(
        _obj_feature(
            "max envelope[hl=10]",
            lambda: PeakEnvelope(a, True),
            None,
            lambda: PeakEnvelope(0.5, True),
        )
    )
    out.append(
        _obj_feature(
            "min envelope[hl=10]",
            lambda: PeakEnvelope(a, False),
            None,
            lambda: PeakEnvelope(0.5, False),
        )
    )
    out.append(
        _obj_feature("decayed max[hl=10]", lambda: DecayedMax(a), None, lambda: DecayedMax(0.5))
    )
    for kind, k in (("shift", 1), ("shift", 5), ("diff", 1), ("pct_change", 20)):
        out.append(
            _obj_feature(
                f"{kind}({k})",
                lambda kind=kind, k=k: Lag(kind, k),
                _expr(kind, k),
                lambda kind=kind, k=k: Lag(kind, k),
            )
        )
    return out


def _bits(v: object) -> object:
    if v is None:
        return None
    if isinstance(v, tuple):
        return tuple(_bits(u) for u in v)
    f = float(v)  # type: ignore[arg-type]
    return "nan" if math.isnan(f) else struct.pack("<d", f)


def run_feature(
    f: Feature, gs: list[int], ts: list[int], xs: list, cuts: list[int]
) -> tuple[list, int]:
    """Stream with a save and load of every group's state at each cut; returns
    the outputs and the largest saved state seen at a cut."""
    states: dict[int, object] = {}
    out = []
    biggest = 0
    cut_set = set(cuts[1:-1])
    for i, (g, t, x) in enumerate(zip(gs, ts, xs, strict=True)):
        if i in cut_set:
            for key in list(states):
                data = f.save(states[key])
                biggest = max(biggest, len(data))
                states[key] = f.load(data)
        st = states.get(g)
        if st is None:
            st = states[g] = f.new()
        out.append(f.step(st, x, t))
    return out, biggest


def invariance(n: int, nonfinite: bool = False) -> list[list]:
    """The clean frame (nulls and zero-weight rows) and the one with NaN and
    infinities. On the second, a NaN poisons each EW state for good (as in
    Polars), so after a group's first NaN its EW outputs are all NaN and say
    little about resuming; the clean frame is the test that bites for them."""
    df = make_df(n, nonfinite)
    gs, ts, xs = stream_values(df)
    m = masked(df)
    feats = features()
    pol = {
        f.name: m.select(f.oracle.over("g").alias("p"))["p"] for f in feats if f.oracle is not None
    }
    rolling = {}
    for f in feats:
        if f.aligned:
            b, B = f.aligned
            kw = dict(window_size=f"{B * b}i", closed="right", min_samples=1)
            x = pl.col("xm")
            rolling[f.name] = m.select(
                c=x.is_not_null().cast(pl.Float64).rolling_sum_by("t", **kw).over("g"),
                mn=x.rolling_min_by("t", **kw).over("g"),
                mx=x.rolling_max_by("t", **kw).over("g"),
            )
    rows = []
    uncut = {f.name: run_feature(f, gs, ts, xs, [0, n])[0] for f in feats}
    for label, chunks in (
        ("1 chunk", 1),
        ("2 chunks", 2),
        ("17 random chunks", 17),
        ("~300 random chunks", 300),
        ("every row", n),
    ):
        cuts = list(range(n + 1)) if chunks == n else _cuts(n, chunks)
        t0 = time.perf_counter()
        eq_uncut = eq_polars = n_polars = 0
        for f in feats:
            out, _ = run_feature(f, gs, ts, xs, cuts)
            if [_bits(v) for v in out] == [_bits(v) for v in uncut[f.name]]:
                eq_uncut += 1
            if f.oracle is not None:
                n_polars += 1
                c = compare(out, pol[f.name])
                if c["validity_diff"] == 0 and c["bit_diff"] == 0:
                    eq_polars += 1
            elif f.aligned:
                n_polars += 1
                b, B = f.aligned
                idx = [i for i, t in enumerate(ts) if t % b == b - 1]
                r = rolling[f.name]
                ok = True
                for col, j in (("c", 0), ("mn", 4), ("mx", 5)):
                    c = compare([out[i][j] for i in idx], r[col].gather(idx))
                    ok &= c["validity_diff"] == 0 and c["bit_diff"] == 0
                eq_polars += ok
        rows.append(
            [
                label,
                len(cuts) - 1,
                len(feats),
                eq_uncut,
                f"{eq_polars} of {n_polars}",
                f"{time.perf_counter() - t0:.1f}",
            ]
        )
    return rows


def make_df(n: int, nonfinite: bool) -> pl.DataFrame:
    from .common import make_frame

    return make_frame(n, nonfinite=nonfinite)


# ---------------------------------------------------------------------------
# Cost per row.


def cost(n_py: int = 50_000, n_pl: int = 1_000_000) -> list[list]:
    """Best of 3. The Python state machine on one group of ``n_py`` rows (the
    step and its output); Polars on ``n_pl`` rows of one group, the matching
    expression or the nearest one."""
    from .common import make_frame

    df = make_frame(n_py, groups=1)
    _, ts, xs = stream_values(df)
    big = masked(make_frame(n_pl, groups=1, null_p=0.0, zero_w_p=0.0)) if n_pl else None
    rows = []

    def best(fn: Callable[[], None], reps: int = 3) -> float:
        out = math.inf
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            out = min(out, time.perf_counter() - t0)
        return out

    x = pl.col("xm")
    py_cases: list[tuple[str, Feature, pl.Expr | list[pl.Expr] | None]] = []
    for b, B in ((5, 12), (20, 12), (5, 60)):
        kw = dict(window_size=f"{B * b}i", closed="right")
        py_cases.append(
            (
                f"bucketed b={b}, B={B}: 6 outputs",
                _bucket_feature(b, B, False),
                [
                    x.rolling_sum_by("t", **kw),
                    x.rolling_mean_by("t", **kw),
                    x.rolling_var_by("t", **kw),
                    x.rolling_min_by("t", **kw),
                    x.rolling_max_by("t", **kw),
                ],
            )
        )
    for f in features():
        if f.name.startswith(
            (
                "ewm_mean[hl=10,adjust=True",
                "ewm_std[hl=10",
                "ewm_mean_by[25i]",
                "max envelope",
                "decayed max",
                "shift(1)",
                "shift(5)",
            )
        ):
            near = f.oracle
            if f.oracle is None:  # the envelopes: Polars' nearest is a rolling extreme
                near = x.rolling_max(20)
            py_cases.append((f.name, f, near))
    for name, f, pexpr in py_cases:

        def go(f: Feature = f) -> None:
            st = f.new()
            for t, v in zip(ts, xs, strict=True):
                f.step(st, v, t)

        py = best(go) / n_py * 1e9
        pl_ns: object = "-"
        if pexpr is not None and big is not None:
            exprs = pexpr if isinstance(pexpr, list) else [pexpr]
            exprs = [e.alias(f"o{j}") for j, e in enumerate(exprs)]
            pl_ns = best(lambda exprs=exprs: big.select(exprs)) / n_pl * 1e9
        rows.append([name, f"{py:,.0f}", pl_ns if isinstance(pl_ns, str) else f"{pl_ns:.1f}"])
    # the read alone, against B
    for B in (4, 12, 60, 240):
        st = Bucketed(BucketSpec(width=5, buckets=B))
        for t, v in zip(ts[:1000], xs[:1000], strict=True):
            st.step(v, t)
        reps = 2000
        t0 = time.perf_counter()
        for _ in range(reps):
            st.read()
        rows.append(
            [f"bucketed read alone, B={B}", f"{(time.perf_counter() - t0) / reps * 1e9:,.0f}", "-"]
        )
    return rows
