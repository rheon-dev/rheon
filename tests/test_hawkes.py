"""``hawkes`` through the bank: the forgotten fit and the marks against the
contrast's definition, and what the bank owes every model -- chunk
invariance, a save mid-stream, zero weights, nulls, its refusals and its
output schema (task 498).

The oracle for the forgotten fit is the least-squares system written from
its definition: ``G = ∫ w φ φᵀ dt`` and ``b_i = Σ w m φ(t−)`` as double sums
over pairs of events, each integral in closed form, ``O(n²)``, sharing no
recursion with the model, which reads ``G`` from ``b``. tick, the
oracle of the unweighted fit and of the window
(``tests/test_second_opinion_hawkes.py``), takes neither weights nor marks.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

INF = float("inf")


def _system(
    t: np.ndarray, marks: np.ndarray, betas: list[float], gamma: float, start: float, end: float
) -> tuple[np.ndarray, np.ndarray]:
    """``G`` and ``b`` (one row per node) from the definition, for rows at
    clocks ``t`` with one mark a node (``> 0`` an event), observed from
    ``start`` to ``end``, each instant weighted ``exp(-γ (end - t))``. An
    event excites the events of every later row and none of its own."""
    d, u = marks.shape[1], len(betas)
    k, p = d * u, 1 + d * u
    rows, nodes = np.nonzero(marks > 0)
    tk, mk = t[rows], marks[rows, nodes]

    def tail(c: float, length: np.ndarray) -> np.ndarray:
        """``∫_0^L exp(-γ (L - y) - c y) dy``."""
        if gamma == 0.0:
            return -np.expm1(-c * length) / c
        return (np.exp(-c * length) - np.exp(-gamma * length)) / (gamma - c)

    g = np.zeros((p, p))
    g[0, 0] = end - start if gamma == 0.0 else -math.expm1(-gamma * (end - start)) / gamma
    for a in range(k):
        ja, ba = divmod(a, u)
        ba = betas[ba]
        on = nodes == ja
        g[0, a + 1] = g[a + 1, 0] = np.sum(ba * mk[on] * tail(ba, end - tk[on]))
        for c in range(k):
            jc, bc = divmod(c, u)
            bc = betas[bc]
            oc = nodes == jc
            t1, t2 = tk[on][:, None], tk[oc][None, :]
            at = np.maximum(t1, t2)
            g[a + 1, c + 1] = np.sum(
                ba * bc * mk[on][:, None] * mk[oc][None, :]
                * np.exp(-ba * (at - t1) - bc * (at - t2))
                * tail(ba + bc, end - at)
            )  # fmt: skip
    b = np.zeros((d, p))
    for i in range(d):
        mine = nodes == i
        w = np.exp(-gamma * (end - tk[mine])) * mk[mine]
        b[i, 0] = w.sum()
        for a in range(k):
            ja, ua = divmod(a, u)
            src = nodes == ja
            earlier = rows[src][None, :] < rows[mine][:, None]
            lag = tk[mine][:, None] - tk[src][None, :]
            excite = betas[ua] * mk[src][None, :] * np.exp(-betas[ua] * lag)
            b[i, a + 1] = np.sum(w * np.sum(np.where(earlier, excite, 0.0), axis=1))
    return g, b


def _stream(n: int, d: int, seed: int) -> pl.DataFrame:
    """Rows on an irregular clock, some at one clock, with events of marks
    1, 2 and fractional sizes, rows of several nodes' events, and rows of
    none: a 0, a negative value or a null."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.where(rng.uniform(size=n) < 0.05, 0.0, rng.exponential(0.5, n)))
    cols: dict[str, object] = {"t": t}
    for j in range(d):
        u = rng.uniform(size=n)
        v = np.where(u < 0.3, 1.0, np.where(u < 0.33, 2.0, np.where(u < 0.36, 0.5 + u, 0.0)))
        v = np.where((u > 0.5) & (u < 0.6), -1.0, v)
        cols[f"n{j}"] = [None if x > 0.9 else y for x, y in zip(u, v, strict=True)]
    cols["w"] = np.where(np.arange(n) % 11 == 5, 0.0, 1.0 + (np.arange(n) % 3))
    cols["g"] = ["a" if i % 3 else "b" for i in range(n)]
    return pl.DataFrame(cols)


def _spec(d: int = 2, **kw: object) -> dict:
    args: dict[str, object] = {
        "targets": [f"n{j}" for j in range(d)],
        "kernel_decays": [1.0, 4.0],
        "clock": "t",
        "gap_cap": 1e9,
        "half_life": 30.0,
    }
    args.update(kw)
    if args.get("kernel_decays") is None:
        del args["kernel_decays"]
    return rp.spec.hawkes("h", **args)


def _marks(df: pl.DataFrame, d: int) -> np.ndarray:
    """Each row's marks as the model reads them: a value above 0 on a row
    of weight above 0."""
    m = np.column_stack([df[f"n{j}"].fill_null(0.0).to_numpy() for j in range(d)])
    m = np.where(m > 0, m, 0.0)
    if "w" in df.columns:
        m = np.where(df["w"].to_numpy()[:, None] > 0, m, 0.0)
    return m


@pytest.mark.parametrize("half_life", [INF, 30.0, 4.0])
def test_the_fit_is_the_contrasts_minimiser(half_life):
    """At every few rows, the fit is ``G⁻¹ b`` of the definition, forgotten
    at ``ln 2 / half_life`` per unit of the clock, with marks, rows of
    several events and zero weights."""
    df = _stream(300, 2, 1).drop("g")
    spec = _spec(half_life=half_life, weight="w", coef_every=0)
    out = rp.ModelBank([spec]).fit_predict(df)["h"].struct.field("coef")
    t, marks = df["t"].to_numpy(), _marks(df, 2)
    gamma = 0.0 if half_life == INF else math.log(2.0) / half_life
    for row in range(60, 300, 37):
        # The fit after a row of weight 0 is the one after the last row that
        # counted: the zero-weight row is in no clock.
        row = np.flatnonzero(df["w"].to_numpy()[: row + 1] > 0)[-1]
        g, b = _system(t[: row + 1], marks[: row + 1], [1.0, 4.0], gamma, t[0], t[row])
        want = np.linalg.solve(g, b.T).T.ravel()
        got = np.asarray(out[int(row)])
        err = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert err < 1e-10, (row, err)


def test_each_intensity_is_read_before_the_rows_events():
    """A row's intensity is the fit on the rows before it, observed to the
    row's clock, times their excitation at that clock: the row's own events
    are in neither."""
    df = _stream(200, 2, 2).drop("g")
    out = rp.ModelBank([_spec(weight="w")]).fit_predict(df).unnest("h")
    t, marks = df["t"].to_numpy(), _marks(df, 2)
    gamma, betas = math.log(2.0) / 30.0, [1.0, 4.0]
    for row in range(50, 200, 29):
        g, b = _system(t[:row], marks[:row], betas, gamma, t[0], t[row])
        theta = np.linalg.solve(g, b.T).T
        phi = [1.0]
        for j in range(2):
            phi.extend(
                np.sum(be * marks[:row, j] * np.exp(-be * (t[row] - t[:row]))) for be in betas
            )
        for i in range(2):
            want = float(theta[i] @ phi)
            got = out[f"intensity_n{i}"][row]
            assert abs(got - want) <= 1e-10 * (1.0 + abs(want)), (row, i, got, want)


WINDOWED = [
    {"half_life": INF, "window_events": 9},
    {"half_life": INF, "window_size": 3.0},
    {"half_life": INF, "window_size": 3.0, "closed": "both"},
]


@pytest.mark.parametrize("kw", [{}, *WINDOWED])
def test_chunk_invariance(kw):
    """The fit on every row (``coef_every=0``), so ``coef`` is held too, not
    only at a chunk's end."""
    df = _stream(600, 2, 3)
    s = _spec(group="g", weight="w", coef_every=0, **kw)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 7, 97):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 600, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize("kw", [{}, *WINDOWED])
def test_a_save_goes_on_to_the_bit(kw):
    df = _stream(500, 2, 4)
    s = _spec(group="g", weight="w", **kw)
    for cut in (1, 37, 250, 499):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_zero_weight_row_learns_none_of_its_events():
    """A row of weight 0 advances the clock and learns nothing: every
    intensity and fit is the stream's whose row carries no event."""
    df = _stream(400, 2, 5).drop("g")
    zero = pl.col("w") == 0.0
    quiet = df.with_columns(
        *(pl.when(zero).then(None).otherwise(pl.col(f"n{j}")).alias(f"n{j}") for j in range(2))
    )
    s = _spec(weight="w", coef_every=0)
    a = rp.ModelBank([s]).fit_predict(df).unnest("h").drop("weight_sum", "settled_frac")
    b = rp.ModelBank([s]).fit_predict(quiet).unnest("h").drop("weight_sum", "settled_frac")
    assert a.drop("n0", "n1").equals(b.drop("n0", "n1"), null_equal=True)


def test_a_positive_weight_scales_nothing_but_the_weight():
    """A weight gates a row's events and is no multiplier of them: every
    weight halved leaves the intensities and the fit where they were."""
    df = _stream(300, 2, 6).drop("g")
    s = _spec(weight="w", coef_every=0)
    a = rp.ModelBank([s]).fit_predict(df).unnest("h")
    b = rp.ModelBank([s]).fit_predict(df.with_columns(pl.col("w") * 0.5)).unnest("h")
    for f in ("intensity_n0", "intensity_n1", "coef"):
        assert a[f].equals(b[f], null_equal=True), f
    assert not a["weight_sum"].equals(b["weight_sum"])


def test_a_window_keeps_its_rows_with_an_event():
    """The window's fit is the definition's over the last ``window_events``
    rows with an event, observed from the clock of the one dropped last."""
    df = _stream(400, 2, 7).drop("g", "w")
    w = 25
    out = rp.ModelBank([_spec(half_life=INF, window_events=w, coef_every=0)]).fit_predict(df)
    coef = out["h"].struct.field("coef")
    t, marks = df["t"].to_numpy(), _marks(df, 2)
    events = np.flatnonzero(marks.sum(axis=1) > 0)
    for row in range(120, 400, 31):
        kept = events[events <= row][-w:]
        dropped = events[events < kept[0]]
        start = t[dropped[-1]] if len(dropped) else t[0]
        g, b = _system(t[kept], marks[kept], [1.0, 4.0], 0.0, start, t[row])
        want = np.linalg.solve(g, b.T).T.ravel()
        got = np.asarray(coef[row])
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-10, row


@pytest.mark.parametrize("closed", ["right", "both"])
def test_a_clock_window_keeps_the_events_of_its_span(closed):
    """The window of ``window_size`` clock units is the definition's fit on
    the events of the last ``window_size``, observed over ``(T - window_size,
    T]``, from the first row until there is that much clock; and ``closed``
    says whether an event exactly that old is in it. The clock is in
    quarters, so rows sit on the edge."""
    n, span = 500, 6.0
    rng = np.random.default_rng(8)
    t = np.cumsum(rng.integers(0, 6, n) * 0.25)
    df = pl.DataFrame(
        {
            "t": t,
            "n0": (rng.uniform(size=n) < 0.5).astype(float),
            "n1": (rng.uniform(size=n) < 0.3).astype(float),
        }
    )
    out = rp.ModelBank(
        [_spec(half_life=INF, window_size=span, closed=closed, coef_every=0)]
    ).fit_predict(df)
    coef = out["h"].struct.field("coef")
    marks = df.select("n0", "n1").to_numpy()
    events = np.flatnonzero(marks.sum(axis=1) > 0)
    edge = checked = 0
    for row in range(40, n, 7):
        age = t[row] - t[events[events <= row]]
        edge += int(np.sum(age == span))
        kept = events[events <= row][(age < span) if closed == "right" else (age <= span)]
        start = max(t[row] - span, t[0])
        g, b = _system(t[kept], marks[kept], [1.0, 4.0], 0.0, start, t[row])
        got = coef[row]
        if np.linalg.cond(g) > 1e8:
            # Too few events to tell the terms apart: no fit is reported.
            assert got.is_null().all(), row
            continue
        want = np.linalg.solve(g, b.T).T.ravel()
        assert np.max(np.abs(np.asarray(got) - want)) / np.max(np.abs(want)) < 1e-10, row
        checked += 1
    assert edge > 10 and checked > 30, "the stream put few events on the window's edge"


def test_the_weight_under_a_window_is_the_weight_inside_it():
    """``weight_sum`` is the weight of the event rows the window holds, as
    the last row left it: a window of events holds ``window_events`` of them,
    one of clock units those within its span; a row without an event is no
    part of it."""
    df = _stream(300, 2, 12).drop("g")
    t, marks, w = df["t"].to_numpy(), _marks(df, 2), df["w"].to_numpy()
    have = marks.sum(axis=1) > 0
    for kw in ({"window_events": 7}, {"window_size": 2.0}):
        out = rp.ModelBank([_spec(half_life=INF, weight="w", **kw)]).fit_predict(df)
        got = out["h"].struct.field("weight_sum").to_numpy()
        for row in range(1, 300):
            prior = np.flatnonzero(have[:row])
            # The window is cut at the last row that counted.
            last = np.flatnonzero(w[:row] > 0)[-1]
            held = prior[-7:] if "window_events" in kw else prior[t[last] - t[prior] < 2.0]
            assert abs(got[row] - w[held].sum()) < 1e-9, (kw, row)


def test_names_match_the_realized_struct():
    """One intensity a target, then the shared fields and ``coef``, each of
    the dtype ``output_index`` declares."""
    df = _stream(200, 3, 8)
    for kw in ({}, {"half_life": [10.0, 50.0]}):
        spec = _spec(3, **kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        assert [f.name for f in out.schema["h"].fields] == rp.spec.output_fields(spec)
        idx = rp.spec.output_index(spec)
        names = {pl.Float64: "f64", pl.List(pl.Float64): "list[f64]"}
        for f in out.schema["h"].fields:
            declared = idx.filter(pl.col("field") == f.name)["dtype"].item()
            assert names.get(f.dtype, "enum") == declared, (f, declared)
    terms = rp.spec.coef_index(_spec(2))["term"].to_list()
    assert (
        terms == ["mu", "alpha_n0_beta1", "alpha_n0_beta4", "alpha_n1_beta1", "alpha_n1_beta4"] * 2
    )


def test_two_nodes_on_the_same_rows_report_nothing():
    """Two nodes whose events fall on the same rows cannot be told apart:
    the fit is withheld, the intensities null, rather than digits of
    rounding."""
    n = 300
    t = np.arange(n, dtype=float)
    df = pl.DataFrame({"t": t, "n0": np.ones(n), "n1": np.ones(n)})
    out = rp.ModelBank([_spec()]).fit_predict(df).unnest("h")
    assert out["intensity_n0"].is_null().all()
    apart = df.with_columns(pl.Series("n1", (np.arange(n) % 2).astype(float)))
    out = rp.ModelBank([_spec()]).fit_predict(apart).unnest("h")
    assert out["intensity_n0"].drop_nulls().len() > 250


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"kernel_decays": [1.0, 1.0]}, "distinct"),
        ({"kernel_decays": [1.0, -2.0]}, "finite number above 0"),
        ({"kernel_decays": []}, "at least one decay"),
        ({"window_events": 0, "half_life": INF}, "at least 1"),
        ({"window_events": 10}, "takes no decay"),
        ({"window_size": 5.0}, "takes no decay"),
        ({"window_size": 5.0, "window_events": 4, "half_life": INF}, "are two windows"),
        ({"window_size": 0.0, "half_life": INF}, "finite span of the clock above 0"),
        ({"window_size": INF, "half_life": INF}, "window_size must be finite"),
        ({"closed": "both"}, "closed applies only under a window"),
        ({"half_life": math.log(2.0) / 1.0002}, "within 0.1% of 1"),
        ({"emit_sigma": True}, "emit_sigma"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([_spec(**kw)])


def test_features_and_a_temporal_clock_are_refused():
    with pytest.raises(TypeError, match="features"):
        rp.spec.hawkes("h", targets=["y"], kernel_decays=[1.0], half_life=10.0, features=["x"])  # type: ignore[call-arg]
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 2), "1h", eager=True
            )
        }
    ).with_columns(pl.lit(1.0).alias("y"))
    with pytest.raises(ValueError, match="kernel_decays are rates per clock unit"):
        spec = rp.spec.hawkes(
            "h", targets=["y"], kernel_decays=[1.0], clock="ts", gap_cap="1h", half_life="1d"
        )
        rp.ModelBank([spec]).fit_predict(df)
    # A duration is a window on a temporal clock, which the rates refuse.
    with pytest.raises(ValueError, match="kernel_decays are rates per clock unit"):
        spec = rp.spec.hawkes(
            "h",
            targets=["y"],
            kernel_decays=[1.0],
            clock="ts",
            gap_cap="1h",
            half_life=INF,
            window_size="1h",
        )
        rp.ModelBank([spec]).fit_predict(df)


def test_the_window_is_held_to_the_groups_state_cap():
    """The window's rows are the one part of the state the data could size,
    and ``window_events`` bounds them, so the cap is checked at the call: a
    window past ``max_state_bytes`` (256 KB by default) is refused naming the
    bytes, one past half of it warned of, and a stream far past the window
    keeps the state at the bytes the settings give."""
    import warnings

    import rheon
    from rheon._spec_point import _state_bytes

    # Two nodes, two decays: 64 bytes a row, so 4,096 rows fill 256 KB.
    assert _state_bytes(2, 2, 4_000) < 256 * 1024 < _state_bytes(2, 2, 4_700)
    with pytest.raises(
        ValueError, match=r"would hold \d+ bytes in each group, past max_state_bytes of 262144"
    ):
        rp.ModelBank([_spec(half_life=INF, window_events=4_700)])
    with pytest.warns(rheon.RheonWarning, match="past half of max_state_bytes"):
        _spec(half_life=INF, window_events=4_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _spec(half_life=INF, window_events=2_000)
        _spec(half_life=INF, window_events=4_700, max_state_bytes=1 << 20)
    with pytest.raises(
        ValueError, match="hold 848 bytes in each group, past max_state_bytes of 500"
    ):
        rp.ModelBank([_spec(half_life=INF, window_events=10, max_state_bytes=500)])
    with pytest.raises(ValueError, match="max_state_bytes must be above 0"):
        rp.ModelBank([_spec(max_state_bytes=0)])
    # Driven far past its window, the state holds what the settings say.
    w = 30
    bank = rp.ModelBank([_spec(group="g", half_life=INF, window_events=w)])
    df = _stream(3_000, 2, 9)
    sizes = []
    for i in range(0, 3_000, 500):
        bank.fit_predict(df.slice(i, 500))
        sizes.append(len(bank.save_bytes()))
    assert max(sizes[1:]) - min(sizes[1:]) < 200, sizes


def test_a_burst_in_a_clock_window_is_warned_of_once_and_refused_at_the_cap():
    """A window of clock units holds the events its span has, which the data
    decides, so the cap is checked as the rows go in: the first row to take a
    group past half of ``max_state_bytes`` warns, once, naming the group and
    the bytes; the row that would take it past the cap stops the run with an
    error naming the group and the bytes. A group that stays under says
    nothing."""
    import warnings

    import rheon
    from rheon._spec_point import _state_bytes

    # Two nodes, two decays: 64 bytes a row and a 32-byte stamp beside each.
    cap = _state_bytes(2, 2, 0) + 8 * (4 + 2) + 40 + 96 * 150
    s = _spec(group="g", half_life=INF, window_size=1_000.0, max_state_bytes=cap)

    def burst(group: str, n: int, first: int = 0) -> pl.DataFrame:
        i = np.arange(first, first + n)
        return pl.DataFrame({"g": group, "t": i * 0.01, "n0": 1.0, "n1": 1.0, "w": 1.0})

    bank = rp.ModelBank([s])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(pl.concat([burst("a", 100), burst("b", 30)]))
    assert len(seen) == 1
    assert f"past half of max_state_bytes ({cap})" in str(seen[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Under the cap and past half: said already, and not said again.
        bank.fit_predict(burst("a", 30, 100))
    with pytest.raises(Exception, match=r'group "a".*to \d+ bytes, above max_state_bytes of') as e:
        bank.fit_predict(burst("a", 100, 130))
    assert str(cap) in str(e.value)
    # The bank learned part of the chunk and stops there.
    with pytest.raises(Exception, match="cannot go on"):
        bank.fit_predict(burst("b", 1, 30))


def _grid(n: int, seed: int, weights: bool = False) -> pl.DataFrame:
    """Rows a quarter of a unit apart in steps, so that sums of steps are
    exact, with two event columns and, optionally, rows of weight 0."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(0, 6, n) * 0.25)
    cols: dict[str, object] = {
        "t": t,
        "n0": (rng.uniform(size=n) < 0.5).astype(float),
        "n1": (rng.uniform(size=n) < 0.35).astype(float),
    }
    if weights:
        cols["w"] = np.where(rng.uniform(size=n) < 0.25, 0.0, 1.0 + rng.integers(0, 3, n))
    return pl.DataFrame(cols)


@pytest.mark.parametrize(
    "kw",
    [
        {"half_life": INF},
        {"half_life": 40.0},
        {"half_life": INF, "window_events": 9},
        {"half_life": INF, "window_size": 3.0},
        {"half_life": INF, "window_size": 3.0, "closed": "both"},
    ],
)
def test_a_zero_weight_row_counts_in_no_clock(kw):
    """The stream with rows of weight 0 gives every other row, to the bit,
    the output of the stream without them: the next row steps from the last
    row that counted, the zero-weight row is in no window and none of
    ``weight_sum``. It is still predicted."""
    df = _grid(500, 21, weights=True)
    s = _spec(**kw, weight="w", coef_every=0)
    full = rp.ModelBank([s]).fit_predict(df).unnest("h")
    kept = df.filter(pl.col("w") > 0)
    thin = rp.ModelBank([s]).fit_predict(kept).unnest("h")
    mask = (df["w"] > 0).to_numpy()
    for f in ("intensity_n0", "intensity_n1", "weight_sum", "coef"):
        got = full[f].filter(pl.Series(mask))
        assert got.equals(thin[f], null_equal=True), f
    zero = full.filter(pl.Series(~mask))
    assert zero["intensity_n0"].drop_nulls().len() > 20, "a zero-weight row is still predicted"


def test_half_lives_are_the_decays_ln_2_over_them():
    """``kernel_half_lives`` on a numeric clock are numbers of its units, the
    decays ``ln 2 / half-life`` by hand, to the bit, and name their terms by
    the half-life."""
    df = _grid(400, 5)
    kw = {"half_life": INF, "coef_every": 0, "gap_cap": 1e9}
    by_half = rp.ModelBank([_spec(kernel_decays=None, kernel_half_lives=[0.5, 3.0], **kw)])
    by_rate = rp.ModelBank([_spec(kernel_decays=[math.log(2) / 0.5, math.log(2) / 3.0], **kw)])
    a, b = by_half.fit_predict(df).unnest("h"), by_rate.fit_predict(df).unnest("h")
    assert a.equals(b, null_equal=True)
    assert by_half.coef()["coef"].equals(by_rate.coef()["coef"])
    terms = by_half.coef()["term"].to_list()
    assert "alpha_n0_h0.5" in terms and "alpha_n1_h3" in terms, terms
    with pytest.raises(ValueError, match="both set the kernels"):
        rp.ModelBank([_spec(kernel_half_lives=[1.0])])
    with pytest.raises(ValueError, match="needs its kernels"):
        rp.ModelBank([_spec(kernel_decays=None)])
    with pytest.raises(ValueError, match="kernel_half_lives must be a finite span"):
        rp.ModelBank([_spec(kernel_decays=None, kernel_half_lives=[1.0, 0.0])])


@pytest.mark.parametrize("quarter", [True, False])
def test_a_temporal_clock_takes_durations(quarter):
    """On a temporal clock every span is a duration, read in seconds off the
    exact stamps: ``kernel_half_lives``, ``window_size`` and ``gap_cap`` as
    durations give, to the bit on steps of a quarter second, and within 1e-11
    on millisecond steps (where a step's seconds differ in the last bit; 1e-13
    measured, where a window holds 5 events or more), the
    output of the same model on a numeric clock of seconds with the decays
    converted by hand. Its terms are named by the half-lives as written."""
    rng = np.random.default_rng(3)
    n = 500
    steps = rng.integers(0, 6, n) * 250 if quarter else rng.integers(1, 900, n)
    ms = np.cumsum(steps)
    base = pl.DataFrame(
        {
            "n0": (rng.uniform(size=n) < 0.5).astype(float),
            "n1": (rng.uniform(size=n) < 0.35).astype(float),
            "w": np.where(rng.uniform(size=n) < 0.2, 0.0, 1.0),
        }
    )
    temporal = base.with_columns(pl.Series("t", ms).cast(pl.Datetime("ms")))
    numeric = base.with_columns(pl.Series("t", ms / 1000.0))
    common = {
        "targets": ["n0", "n1"],
        "clock": "t",
        "half_life": INF,
        "weight": "w",
        "coef_every": 0,
    }
    # A span to the microsecond-and-a-half, so that no row sits exactly on
    # its edge, where the numeric clock's rounded difference would decide
    # it the other way (the stamps decide it exactly).
    spans = (
        [("10s", "both"), ("3s", "right")]
        if quarter
        else [("9999500us", "both"), ("2999500us", "right")]
    )
    for window, closed in spans:
        spec_t = rp.spec.hawkes(
            "h",
            kernel_half_lives=["500ms", "2s"],
            window_size=window,
            closed=closed,
            gap_cap="1h",
            **common,
        )
        spec_n = rp.spec.hawkes(
            "h",
            kernel_decays=[math.log(2) / 0.5, math.log(2) / 2.0],
            window_size=int(window[:-2]) / 1e6 if window.endswith("us") else float(window[:-1]),
            closed=closed,
            gap_cap=3600.0,
            **common,
        )
        bank_t, bank_n = rp.ModelBank([spec_t]), rp.ModelBank([spec_n])
        a = bank_t.fit_predict(temporal).unnest("h")
        b = bank_n.fit_predict(numeric).unnest("h")
        for f in ("intensity_n0", "intensity_n1", "weight_sum", "coef"):
            if quarter:
                assert a[f].equals(b[f], null_equal=True), (window, f)
            else:
                # Where the window holds a few events: with fewer the fit is
                # near singular and a last-bit difference of a step moves it.
                held = (b["weight_sum"] >= 5).to_numpy()
                assert held.sum() > 150
                if f == "coef" and window[0] not in "19":
                    continue  # a few events fit ten terms: no digits to compare
                if f == "coef":

                    def rows(c: pl.Series) -> np.ndarray:
                        return np.array(
                            [
                                [np.nan if x is None else x for x in v] if v else [np.nan] * 10
                                for v in c.to_list()
                            ]
                        )

                    got, want = rows(a[f])[held], rows(b[f])[held]
                    nan = np.isnan(want)
                    assert np.array_equal(np.isnan(got), nan), (window, f)
                    assert np.max(np.abs(got[~nan] - want[~nan])) < 1e-10, (window, f)
                else:
                    x, y = a[f].to_numpy()[held], b[f].to_numpy()[held]
                    assert np.array_equal(np.isnan(x), np.isnan(y)), (window, f)
                    assert np.allclose(x, y, rtol=1e-11, atol=1e-11, equal_nan=True), (window, f)
        assert a["intensity_n0"].drop_nulls().len() > 200
        terms = bank_t.coef()["term"].to_list()
        assert "alpha_n1_h500ms" in terms and "alpha_n0_h2s" in terms, terms
