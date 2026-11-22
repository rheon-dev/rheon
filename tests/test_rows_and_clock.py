"""Every count of rows a kind takes has its clock form (docs/PLAN.md task
538, part 2): the warm-ups, the robust refits, the cadences and the windows
that counted rows only, each beside its count.

One table, ``ITEMS``, and the same checks for each entry:

- on every clock of one unit a row -- none, a number clock in seconds, a
  datetime clock of one-second steps -- the clock form of ``D`` is the
  count it stands for (``D + 1`` rows for a start, which holds the row that
  reaches its span; ``D`` for a cadence or a window), every field of every
  row to the bit, and the count one more is not;
- on an irregular clock a start holds the rows Polars' own datetime
  arithmetic says it does (the definition, computed by the library), and a
  duration on a datetime clock is the same number of seconds on a number
  clock, to the bit;
- rows of weight 0 are in no clock: the weighted rows' outputs are those of
  the stream without them;
- chunking and a save and load inside the span move nothing;
- each form is refused where it means nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

INF = float("inf")
T0 = datetime(2024, 3, 1, 9, 30)


def frame(n: int = 400, seed: int = 0, steps_ms: np.ndarray | None = None) -> pl.DataFrame:
    """Two features in three blobs (a fourth from the middle on), a linear
    target with a few wild rows, and a count; stamped every second unless
    steps are given, as a datetime ``t`` and as seconds ``n``."""
    rng = np.random.default_rng([538, 2, seed])
    centres = np.array([[0.0, 0.0], [6.0, 1.0], [1.0, 5.0], [-4.0, 7.0]])
    pick = np.where(np.arange(n) > n // 2, rng.integers(0, 4, n), rng.integers(0, 3, n))
    x = centres[pick] + rng.normal(0.0, 0.4, (n, 2))
    y = 1.0 + 0.5 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(0.0, 0.3, n)
    y = np.where(rng.random(n) < 0.08, y + 8.0, y)
    steps = np.full(n, 1000) if steps_ms is None else steps_ms
    ms = np.concatenate([[0], np.cumsum(steps[1:])])
    return pl.DataFrame(
        {
            "t": [T0 + timedelta(milliseconds=int(v)) for v in ms],
            "n": ms / 1000.0,
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": y,
            "c": np.abs(np.round(y)),
        }
    ).with_columns(pl.col("t").cast(pl.Datetime("ns")))


def irregular(n: int = 400, seed: int = 1) -> pl.DataFrame:
    return frame(n, seed, np.random.default_rng([seed, 9]).integers(1, 3000, n))


@dataclass
class Item:
    """One count and its clock form: ``base`` the spec's own keywords,
    ``rows(n)`` the count, ``span(v)`` the clock form; a start holds ``D +
    1`` rows on a unit clock, a cadence or a window ``D``."""

    builder: str
    base: dict[str, Any]
    rows: Callable[[int], dict[str, Any]]
    span: Callable[[Any], dict[str, Any]]
    start: bool
    spans: tuple[int, ...] = (5, 40)
    fixed: dict[str, Any] = field(default_factory=dict)


ITEMS: dict[str, Item] = {
    "kmeans.warm_span": Item(
        "kmeans",
        dict(features=["x0", "x1"], n_clusters=3, half_life=200.0),
        lambda n: dict(warm_rows=n),
        lambda v: dict(warm_span=v),
        start=True,
    ),
    "kmeans.update_every": Item(
        "kmeans",
        dict(features=["x0", "x1"], n_clusters=3, half_life=200.0, warm_rows=20),
        lambda n: dict(update_every_rows=n),
        lambda v: dict(update_every=v),
        start=False,
        spans=(3, 7),
    ),
    "hmm.warm_span": Item(
        "hmm",
        dict(
            features=["x0", "x1"],
            n_components=2,
            half_life=300.0,
            covariance_type="diag",
            precision_prior=0.1,
        ),
        lambda n: dict(warm_rows=n),
        lambda v: dict(warm_span=v),
        start=True,
    ),
    # Four centers on three blobs: two share one, and merge once the fourth
    # blob appears.
    "kmeans.split_merge_every": Item(
        "kmeans",
        dict(features=["x0", "x1"], n_clusters=4, half_life=200.0, warm_rows=20),
        lambda n: dict(split_merge_every_rows=n),
        lambda v: dict(split_merge_every=v),
        start=False,
        spans=(10, 30),
    ),
}


def make(item: Item, clock: str | None, **kw: Any) -> dict:
    d: dict = dict(item.base)
    d.update(kw)
    if clock is not None:
        d["clock"] = clock
        d.setdefault("gap_cap", "1000d" if clock == "t" else 1e9)
        # The half-life in the clock's own form, so a unit clock is the rows.
        if "half_life" in d:
            d["half_life"] = as_clock(clock, d["half_life"])
    return getattr(rp.spec, item.builder)("m", **d)


def fields(out: pl.DataFrame) -> pl.DataFrame:
    """Every field but ``coef``, which a chunk's last row writes by design."""
    cols = out.select(pl.col("m").struct.unnest())
    return cols.drop("coef") if "coef" in cols.columns else cols


def run(spec: dict, df: pl.DataFrame, chunks: int = 1) -> pl.DataFrame:
    bank = rp.ModelBank([spec])
    size = -(-df.height // chunks)
    return pl.concat([fields(bank.fit_predict(c)) for c in df.iter_slices(size)])


def as_clock(clock: str | None, d: float) -> Any:
    """``d`` units of the clock: a duration on ``t``, a number otherwise."""
    if clock == "t":
        return f"{int(round(d * 1000))}ms"
    return d


@pytest.mark.parametrize("name", sorted(ITEMS))
def test_the_clock_form_of_d_is_its_count_on_a_clock_of_one_unit_a_row(name):
    item = ITEMS[name]
    df = frame()
    for d in item.spans:
        n = d + 1 if item.start else d
        want = run(make(item, None, **item.rows(n)), df)
        # The count matters here: a longer one moves some output.
        assert not run(make(item, None, **item.rows(2 * n + 3)), df).equals(want, null_equal=True)
        for clock in [None, "n", "t"]:
            got = run(make(item, clock, **item.span(as_clock(clock, d))), df)
            assert got.equals(want, null_equal=True), (clock, d)


@pytest.mark.parametrize("name", sorted(n for n in ITEMS if ITEMS[n].start))
def test_a_start_holds_the_rows_polars_puts_in_its_span(name):
    """The definition, by Polars: the rows from the first through the first
    row at least the span after it; spans a millisecond either side of a
    row's distance from the first."""
    item = ITEMS[name]
    df = irregular()
    edge = df["t"][40] - df["t"][0]
    ms1 = timedelta(milliseconds=1)
    for span in [edge - ms1, edge, edge + ms1]:
        reached = df.select((pl.col("t") - pl.col("t").first()) >= span).to_series()
        n = int(reached.arg_true()[0]) + 1
        ms = int(span / ms1)
        got = run(make(item, "t", **item.span(f"{ms}ms")), df)
        want = run(make(item, "t", **item.rows(n)), df)
        assert got.equals(want, null_equal=True), span


@pytest.mark.parametrize("name", sorted(ITEMS))
def test_a_duration_is_its_seconds_on_a_number_clock(name):
    item = ITEMS[name]
    # Whole seconds, which a number clock holds exactly as a datetime does:
    # a step a number clock rounds is that clock's, not the span's.
    df = frame(400, 3, 1000 * np.random.default_rng(11).integers(0, 4, 400))
    d = 37.0
    got = run(make(item, "t", **item.span(as_clock("t", d))), df)
    assert got.equals(run(make(item, "n", **item.span(d)), df), null_equal=True)


@pytest.mark.parametrize("name", sorted(ITEMS))
def test_rows_of_weight_zero_are_in_no_clock(name):
    item = ITEMS[name]
    df = frame(300).with_columns(w=pl.lit(1.0))
    zero = np.zeros(df.height, dtype=bool)
    zero[[0, 3, 10, 11, 12, 29, 30, 31, 77]] = True
    with_zeros = df.with_columns(w=pl.Series(np.where(zero, 0.0, 1.0)))
    s = make(item, None, weight="w", **item.span(item.spans[0]))
    got = run(s, with_zeros).filter(pl.Series(~zero))
    assert got.equals(run(s, df.filter(pl.Series(~zero))), null_equal=True)


@pytest.mark.parametrize("name", sorted(ITEMS))
@pytest.mark.parametrize("chunks", [7, 150])
def test_chunking_moves_nothing(name, chunks):
    item = ITEMS[name]
    df = irregular(450, 2)
    s = make(item, "t", **item.span(as_clock("t", 3.0 * item.spans[-1])))
    assert run(s, df, chunks).equals(run(s, df), null_equal=True)


@pytest.mark.parametrize("name", sorted(ITEMS))
@pytest.mark.parametrize("cut", [12, 90])
def test_a_state_saved_inside_the_span_goes_on_to_the_bit(name, cut, tmp_path):
    item = ITEMS[name]
    df = irregular(300, 4)
    s = make(item, "t", **item.span(as_clock("t", 1.5 * item.spans[-1])))
    want = run(s, df)
    first = rp.ModelBank([s])
    head = fields(first.fit_predict(df.head(cut)))
    path = tmp_path / "bank.msgpack"
    first.save(str(path))
    rest = fields(rp.ModelBank.load(str(path)).fit_predict(df.tail(df.height - cut)))
    assert pl.concat([head, rest]).equals(want, null_equal=True)


@pytest.mark.parametrize("name", sorted(ITEMS))
def test_the_clock_form_is_refused_where_it_means_nothing(name):
    item = ITEMS[name]
    df = frame(30)
    key = next(iter(item.span(1.0)))
    free = dict(half_life=INF)
    with pytest.raises(Exception, match="has no clock to measure it"):
        rp.ModelBank([make(item, None, **free, **item.span("5m"))]).fit_predict(df)
    with pytest.raises(Exception, match=f"{key} (?:must|is a plain number)"):
        rp.ModelBank([make(item, "t", **item.span(-3.0), gap_cap="10m")]).fit_predict(df)
    if item.start:
        both = {**item.span(10.0), **item.rows(20)}
        with pytest.raises(Exception, match="Leave one of them out"):
            rp.ModelBank([make(item, None, **both)]).fit_predict(df)


def test_every_clock_form_is_written_only_when_given():
    for item in ITEMS.values():
        bare = make(item, None)
        given = make(item, "t", **item.span("5m"))
        for key in item.span(1.0):
            assert key not in bare["model"]
        assert rp.ModelBank([given]).specs[0] == given
        assert rp.ModelBank([bare]).specs[0] == bare
