"""``refit_span``: the first rows' refit over a span of the clock, for ``glm``
and ``expectile`` (docs/PLAN.md task 538).

The span holds a target's rows from its first learned row through the first
row at least the span after it, read off the rows' exact stamps. The oracle
is that definition computed by Polars' own datetime arithmetic: the count of
rows it gives, as ``refit_rows``, is the same refit to the bit. Both sides of
the span's end; the row clock, a number clock and a temporal one; rows of
weight 0; chunking; a saved state; the cap as the rows go in; the refusals.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import rheon
from rheon import polars as rp

TIER = "essential"

INF = float("inf")
T0 = datetime(2024, 3, 1, 9, 30)


def frame(n: int = 400, seed: int = 0, steps_ms: np.ndarray | None = None) -> pl.DataFrame:
    """Two features and a positive target (a Poisson count, for ``glm``),
    stamped every second unless steps are given."""
    rng = np.random.default_rng([538, seed])
    x = rng.standard_normal((n, 2))
    y = rng.poisson(np.exp(0.4 + 0.3 * x[:, 0] - 0.2 * x[:, 1])).astype(float)
    steps = np.full(n, 1000) if steps_ms is None else steps_ms
    ms = np.concatenate([[0], np.cumsum(steps[1:])])
    return pl.DataFrame(
        {
            "t": [T0 + timedelta(milliseconds=int(v)) for v in ms],
            "n": ms / 1000.0,
            "x0": x[:, 0],
            "x1": x[:, 1],
            "y": y,
        }
    ).with_columns(pl.col("t").cast(pl.Datetime("ns")))


def make(kind: str, **kw: object) -> dict:
    d: dict = dict(targets=["y"], features=["x0", "x1"], half_life=INF)
    d.update(kw)
    if d.get("clock") is not None:
        # No step here is capped: the span reads the clock as it is.
        d.setdefault("gap_cap", "1000d" if d["clock"] == "t" else 1e9)
    if kind == "glm":
        return rp.spec.glm("m", **d)
    return rp.spec.expectile("m", expectile=0.8, **d)


def fields(out: pl.DataFrame) -> pl.DataFrame:
    """Every field but ``coef``, which a chunk's last row writes by design."""
    return out.select(pl.col("m").struct.unnest()).drop("coef")


def run(spec: dict, df: pl.DataFrame, chunks: int = 1) -> pl.DataFrame:
    bank = rp.ModelBank([spec])
    size = -(-df.height // chunks)
    return pl.concat([fields(bank.fit_predict(c)) for c in df.iter_slices(size)])


def rows_of_span(df: pl.DataFrame, span: timedelta) -> int:
    """The definition, by Polars: the rows from the first through the first
    row at least ``span`` after it."""
    reached = df.select((pl.col("t") - pl.col("t").first()) >= span).to_series()
    return int(reached.arg_true()[0]) + 1


KINDS = ["glm", "expectile"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("span", [5, 40])
def test_a_span_of_d_on_a_clock_of_one_unit_a_row_is_d_plus_one_rows(kind, span):
    """On every clock with one unit a row -- none, a number clock in seconds,
    a datetime clock of one-second steps -- ``refit_span = D`` is the refit
    of ``refit_rows = D + 1``: every field of every row, to the bit."""
    df = frame()
    want = run(make(kind, refit_rows=span + 1), df)
    for clock, value in [(None, span), ("n", float(span)), ("t", f"{span}s")]:
        got = run(make(kind, clock=clock, refit_span=value), df)
        assert got.equals(want, null_equal=True), (clock, value)


@pytest.mark.parametrize("kind", KINDS)
def test_an_irregular_clock_refits_on_the_rows_the_span_has(kind):
    """On an irregular clock the span holds the rows Polars' own datetime
    arithmetic says it does, and a span a millisecond longer or shorter than
    a row's distance from the first falls on either side of that row."""
    rng = np.random.default_rng(7)
    df = frame(500, seed=1, steps_ms=rng.integers(1, 3000, 500))
    # The distance of row 60 from the first, exactly: both sides of it.
    edge = df["t"][60] - df["t"][0]
    ms1 = timedelta(milliseconds=1)
    assert rows_of_span(df, edge) == 61
    assert rows_of_span(df, edge + ms1) == 62
    assert rows_of_span(df, edge - ms1) <= 61
    for span in [edge - ms1, edge, edge + ms1]:
        n = rows_of_span(df, span)
        ms = int(span / timedelta(milliseconds=1))
        got = run(make(kind, clock="t", refit_span=f"{ms}ms"), df)
        want = run(make(kind, refit_rows=n), df)
        assert got.equals(want, null_equal=True), (span, n)


@pytest.mark.parametrize("kind", KINDS)
def test_rows_of_weight_zero_are_no_rows_of_the_span(kind):
    """A row of weight 0 counts in no clock: on the row clock the span still
    holds ``D + 1`` rows that counted, and the weighted rows' outputs are
    those of the stream without the zero-weight rows, to the bit."""
    df = frame(300).with_columns(w=pl.lit(1.0))
    zero = np.zeros(df.height, dtype=bool)
    zero[[0, 3, 10, 11, 12, 29, 30, 31]] = True
    with_zeros = df.with_columns(w=pl.Series(np.where(zero, 0.0, 1.0)))
    s = make(kind, weight="w", refit_span=20)
    got = run(s, with_zeros).filter(pl.Series(~zero))
    want = run(s, df.filter(pl.Series(~zero)))
    assert got.equals(want, null_equal=True)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("chunks", [7, 150])
def test_chunking_moves_nothing(kind, chunks):
    df = frame(450, seed=2, steps_ms=np.random.default_rng(3).integers(1, 2500, 450))
    s = make(kind, clock="t", refit_span="2m")
    assert run(s, df, chunks).equals(run(s, df), null_equal=True)


@pytest.mark.parametrize("kind", KINDS)
def test_a_state_saved_inside_the_span_goes_on_to_the_bit(kind, tmp_path):
    df = frame(300, seed=4)
    s = make(kind, clock="t", refit_span="2m")
    whole = rp.ModelBank([s])
    want = fields(whole.fit_predict(df))
    first = rp.ModelBank([s])
    head = fields(first.fit_predict(df.head(50)))
    path = tmp_path / "bank.msgpack"
    first.save(str(path))
    rest = fields(rp.ModelBank.load(str(path)).fit_predict(df.tail(250)))
    assert pl.concat([head, rest]).equals(want, null_equal=True)


def test_glm_warns_past_half_of_its_cap_and_stops_at_it():
    """A glm refit's buffers are held to 256 KB a group: a span longer than
    the stream warns once past half, naming the group and the bytes, and the
    row that would pass the cap stops the run naming them."""
    n = 9000
    df = frame(n, seed=5).with_columns(g=pl.lit("a"))
    s = make("glm", clock="t", refit_span="30d", group="g")
    bank = rp.ModelBank([s])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(df.head(5000))
    assert len(seen) == 1
    msg = str(seen[0].message)
    assert "past half of the cap a group's refit buffers may hold (262144)" in msg
    assert "the rows inside refit_span = 30d" in msg
    with pytest.raises(Exception, match=r'group "a".*to \d+ bytes, above the 262144 bytes') as e:
        bank.fit_predict(df.tail(n - 5000))
    assert "Shorten refit_span, or give refit_rows" in str(e.value)


def test_expectile_warns_past_half_of_max_state_bytes_and_stops_at_it():
    cap = 20_000
    df = frame(1500, seed=6).with_columns(g=pl.lit("a"))
    s = make("expectile", clock="t", refit_span="30d", group="g", max_state_bytes=cap)
    bank = rp.ModelBank([s])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(df.head(400))
    assert len(seen) == 1
    assert f"past half of max_state_bytes ({cap})" in str(seen[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Exception, match=r"above max_state_bytes of 20000") as e:
            bank.fit_predict(df.tail(1100))
    assert "Shorten refit_span, or give refit_rows, or raise max_state_bytes" in str(e.value)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("kw", "words"),
    [
        (dict(refit_span=0.0), "refit_span must be a finite number of clock units above 0"),
        (dict(refit_span=-3.0), "refit_span must be a finite number of clock units above 0"),
        (dict(refit_span=10.0, refit_rows=20), "two refits of the first rows"),
        (dict(refit_span="5m"), "has no clock to measure it"),
        (dict(clock="t", refit_span=300.0, gap_cap="10m"), "is a plain number"),
    ],
)
def test_the_span_is_refused_where_it_means_nothing(kind, kw, words):
    with pytest.raises(Exception, match=words):
        rp.ModelBank([make(kind, **kw)]).fit_predict(frame(20))


def test_an_expectile_window_refuses_a_span_refit():
    with pytest.raises(Exception, match="but refit_span is set"):
        rp.ModelBank([make("expectile", window_rows=40, refit_span=10.0)])


def test_the_span_is_written_only_when_given():
    """The bank reports the dict the builder made, and a spec without a span
    names none."""
    for kind in KINDS:
        bare = make(kind)
        assert "refit_span" not in bare["model"]
        with_span = make(kind, clock="t", refit_span="5m")
        assert rp.ModelBank([with_span]).specs[0] == with_span
        assert rp.ModelBank([bare]).specs[0] == bare
