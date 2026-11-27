"""The JSON export of every kind's state, at its hardest, mid-stream.

``ModelBank.write_json`` re-reads its own output and refuses unless it gives
the state's MessagePack byte for byte (``tests/test_state_json.py``). That
check is what makes an export faithful, and it is also what turned a NaN in
a state into a refused export: ``serde_json`` writes NaN and the infinities
as ``null``, and a state field nobody had tagged by hand refused the bank.
``gmm``'s held ``score_pred`` under an embargo and ``sarima``'s rings were
the two found; this sweep finds the rest.

Each kind, as ``test_model_registry.MINIMAL`` builds it and so joining the
day it is registered, runs with every stress its builder takes: an embargo,
``min_weight`` above the first rows' weight (so rows are withheld), null and
zero-weight rows (the first row among them), two targets, and every output
switch that grows a reading which can be NaN (``fullest``'s list). At
several points in the stream the bank is exported to JSON, which must not
refuse; since the export is read back to the state's own bytes before it is
returned, a successful export *is* a JSON round trip to the bit. The bank is
also saved and loaded as MessagePack there, and the loaded bank must go on
exactly as the original does.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import polars as pl
import pytest

from rheon import _spec
from rheon import polars as rp
from test_label_delay import kind_frame
from test_model_registry import _GROWTH, MINIMAL

TIER = "mixed"

N = 300
#: Where the stream is cut: after a zero-weight first row, inside the
#: withheld start, and twice once the fits are reading.
CUTS = (1, 7, 120, 230)


def frame(kind: str) -> pl.DataFrame:
    df = kind_frame(kind, n=N, steps=np.ones(N)).with_columns(
        # A second target, null on its first row.
        z=pl.col("y").shift(1)
    )
    idx = pl.int_range(pl.len())
    zero_w = idx.is_in([0, 5, 6, 90, 91, 200])
    null_y = idx.is_in([3, 40, 41, 150])
    null_x = idx.is_in([4, 60, 61, 180])
    # `rcov` and `random_cut_forest` take a weight of 0 or 1 only; `g` is
    # `rcov`'s group, closed as it grows.
    w = pl.lit(1.0) if kind in ("rcov", "random_cut_forest") else pl.col("w")
    return df.with_columns(
        g=idx // 60,
        w=pl.when(zero_w).then(0.0).otherwise(w),
        y=pl.when(null_y).then(None).otherwise(pl.col("y")),
        x0=pl.when(null_x).then(None).otherwise(pl.col("x0")),
    )


#: The stream settings offered first, each kept where the builder takes it.
_STRESS: list[dict[str, object]] = [
    {"weight": "w"},
    {"min_weight": 10.0},
    {"embargo": 3.0},
]


def stressed(kind: str) -> tuple[dict, list[str]]:
    """``kind`` with every stress of ``_STRESS`` and every switch of
    ``_GROWTH`` its builder takes; a refused one (a ``ValueError`` or a
    ``TypeError``) is left out. Returns the spec and what it took."""
    kw: dict[str, object] = {"targets": ["y", "z"], "features": ["x0", "x1"], "half_life": 50.0}
    kw.update(MINIMAL[kind])
    if kind in ("kmeans", "hmm"):
        kw["warm_rows"] = 20
    kw = {k: v for k, v in kw.items() if v is not None}
    builder = getattr(rp.spec, kind)
    takes = set(inspect.signature(builder).parameters) | set(
        inspect.signature(_spec._common).parameters
    )
    took: list[str] = []
    for more in _STRESS + _GROWTH:
        if any(k in MINIMAL[kind] for k in more) or not set(more) <= takes:
            continue
        try:
            spec = builder("m", **{**kw, **more})
            rp.spec.output_fields(spec)
        except (TypeError, ValueError):
            continue
        kw = {**kw, **more}
        took.extend(more)
    if "targets" in kw and kw["targets"] == ["y", "z"]:
        try:
            builder("m", **kw)
        except (TypeError, ValueError):
            kw["targets"] = ["y"]
    return builder("m", **kw), took


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            k,
            marks=()
            if k in ("ewridge", "sarima", "gmm")
            else pytest.mark.extended(
                reason="a state exported and resumed at every cut, a kind each"
            ),
        )
        for k in sorted(MINIMAL)
    ],
)
def test_every_kind_exports_and_resumes_at_every_cut(kind):
    spec, _ = stressed(kind)
    df = frame(kind)
    bank = rp.ModelBank([spec], threads=1)
    outs: list[pl.DataFrame] = []
    saved: dict[int, bytes] = {}
    edges = [0, *CUTS, N]
    for lo, hi in zip(edges, edges[1:], strict=False):
        outs.append(bank.fit_predict(df[lo:hi]))
        if hi == N:
            break
        # The export reads itself back to the state's bytes or raises.
        text = bank.write_json()
        json.loads(text)
        saved[hi] = bank.save_bytes()
    text = bank.write_json()
    json.loads(text)

    # Fed in the same calls as the original: a ``coef`` is written on the
    # last row of each call, so the calls are part of what is compared.
    for cut, data in saved.items():
        resumed = rp.ModelBank.load(data, specs=[spec], threads=1)
        for i in range(edges.index(cut), len(outs)):
            got = resumed.fit_predict(df[edges[i] : edges[i + 1]])
            assert got.equals(outs[i], null_equal=True), (kind, cut, edges[i])


def test_the_sweep_meets_every_stress_somewhere():
    """The stresses are not left out everywhere: an embargo, ``min_weight``
    and a weight column each reach most kinds."""
    taken = {k: stressed(k)[1] for k in MINIMAL}
    for stress in ("weight", "min_weight", "embargo"):
        n = sum(stress in t for t in taken.values())
        assert n > len(MINIMAL) // 2, (stress, n)


#: The bank's bound on a value it learns from (``rheon_core::INPUT_BOUND``).
INPUT_BOUND = 1e100


def _with(df: pl.DataFrame, col: str, at: int, values: tuple[float, ...]) -> pl.DataFrame:
    idx = pl.int_range(pl.len())
    e = pl.col(col)
    for i, v in enumerate(values):
        e = pl.when(idx == at + i).then(v).otherwise(e)
    return df.with_columns(e.alias(col))


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            k,
            marks=()
            if k in ("ar", "ewridge")
            else pytest.mark.extended(
                reason="a state saved after rows at the input bound, a kind each"
            ),
        )
        for k in sorted(MINIMAL)
    ],
)
def test_every_kind_loads_after_a_row_at_the_input_bound(kind):
    """A row at the input bound is a legal row, so a state saved after it
    is one the model reached, and it must load and go on exactly as the
    bank that was never saved. Two cases a load once refused: `garch`'s
    projected step left its parameters at the rounding of an `ω` near
    `1e195`, and `emit_breaks` kept the square of an `ar` residual past
    `1e154` as an infinite spread (W14-15). Each kind with every stress
    its builder takes, a target and a feature at `±1e100`, alone and
    twice in a row, in the start and once fitting, saved after the bound
    row and after the row past it; a row the kind refuses is skipped, as
    its own refusal is tested beside its domain."""
    spec, _ = stressed(kind)
    base = frame(kind)
    loaded = 0
    for col in ("y", "x0"):
        for at in (10, 125, 235):
            for values in ((INPUT_BOUND,), (-INPUT_BOUND,), (INPUT_BOUND, -INPUT_BOUND)):
                df = _with(base, col, at, values)
                for cut in (at + len(values), at + len(values) + 1):
                    kept = rp.ModelBank([spec], threads=1)
                    try:
                        kept.fit_predict(df[:cut])
                        want = kept.fit_predict(df[cut:])
                    except ValueError as e:
                        # a value outside the kind's domain: a label, a
                        # count, a probability
                        assert "column" in str(e), (kind, col, at, values, e)
                        continue
                    again = rp.ModelBank([spec], threads=1)
                    again.fit_predict(df[:cut])
                    resumed = rp.ModelBank.load(again.save_bytes(), specs=[spec], threads=1)
                    got = resumed.fit_predict(df[cut:])
                    assert got.equals(want, null_equal=True), (kind, col, at, values, cut)
                    loaded += 1
    assert loaded > 0, kind
