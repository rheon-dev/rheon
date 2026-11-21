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
            excite = (
                betas[ua] * mk[src][None, :] * np.exp(-betas[ua] * (tk[mine][:, None] - tk[src][None, :]))
            )  # fmt: skip
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
        cols[f"n{j}"] = [None if 0.9 < x else y for x, y in zip(u, v, strict=True)]
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
        g, b = _system(t[: row + 1], marks[: row + 1], [1.0, 4.0], gamma, t[0], t[row])
        want = np.linalg.solve(g, b.T).T.ravel()
        got = np.asarray(out[row])
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


@pytest.mark.parametrize("kw", [{}, {"half_life": INF, "window_events": 9}])
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


@pytest.mark.parametrize("kw", [{}, {"half_life": INF, "window_events": 9}])
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
    assert terms == ["mu", "alpha_n0_beta1", "alpha_n0_beta4", "alpha_n1_beta1", "alpha_n1_beta4"] * 2


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
        {"ts": pl.datetime_range(pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 2), "1h", eager=True)}
    ).with_columns(pl.lit(1.0).alias("y"))
    with pytest.raises(ValueError, match="kernel_decays is a plain number"):
        spec = rp.spec.hawkes(
            "h", targets=["y"], kernel_decays=[1.0], clock="ts", gap_cap="1h", half_life="1d"
        )
        rp.ModelBank([spec]).fit_predict(df)
