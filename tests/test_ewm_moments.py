"""``ewm_moments`` through its builder and the bank: the spec the builder
writes, the outputs' names and order, the refusals by name, and a realistic
stream read against the weighted definitions written longhand.

The third-party oracles of each quantity (scipy, statsmodels, Polars' rolling
moments) hold the NumPy reference in ``tests/test_reference_ewm_moments.py``,
and the Rust core is held to that reference; here the bank is held to the
definitions themselves on a stream with uneven weights, zero weights, a decay
on an uneven clock, a lag, and chunks of every size.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

N = 300


def _frame(n: int = N, seed: int = 554) -> pl.DataFrame:
    """A gamma column (skewed, at a level of about 6) and an AR(1) column at
    0.6, on an uneven clock, with uneven weights and two rows of weight 0,
    the first row among them."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + e[i]
    w = rng.uniform(0.5, 2.0, n)
    w[0] = 0.0
    w[57] = 0.0
    return pl.DataFrame(
        {
            "t": np.cumsum(rng.exponential(1.0, n)),
            "g": 3.0 + rng.gamma(2.0, 1.5, n),
            "ar": ar,
            "w": w,
        }
    )


def _fit(spec: dict, frame: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(frame)[spec["name"]].struct.unnest()


def _close(got: float | None, want: float, rel: float = 1e-11) -> bool:
    return got is not None and abs(got - want) <= rel * max(abs(want), 1.0)


# --- the spec the builder writes ---------------------------------------------


def test_the_builder_writes_what_is_given_and_reads_the_features_off_the_terms():
    terms = {"m3": {"g": 3}, "ac1": {"ar": 1, "ar@1": 1}, "co": {"ar": 1, "g": 1}}
    spec = rp.spec.ewm_moments("mom", terms=terms, half_life=50.0)
    assert spec["model"] == {"type": "ewm_moments", "terms": terms}
    # Every column the terms name, lags taken off, in the order first named;
    # the target the plumbing needs is the first.
    assert spec["features"] == ["g", "ar"]
    assert spec["targets"] == ["g"]
    # A copy: changing the caller's table later changes no spec.
    terms["m3"]["g"] = 4
    assert spec["model"]["terms"]["m3"] == {"g": 3}

    given = rp.spec.ewm_moments(
        "mom",
        terms={"skew": {"g": 3}},
        features=["g", "ar"],
        kind={"skew": "standardized"},
        bias=False,
        emit_se=True,
        window_rows=100,
        max_state_bytes=1 << 20,
        half_life=50.0,
    )
    assert given["model"] == {
        "type": "ewm_moments",
        "terms": {"skew": {"g": 3}},
        "kind": {"skew": "standardized"},
        "bias": False,
        "emit_se": True,
        "window_rows": 100,
    }
    assert given["max_state_bytes"] == 1 << 20
    assert given["features"] == ["g", "ar"]
    clocked = rp.spec.ewm_moments(
        "mom",
        terms={"ac": {"ar": 1, "ar@2.5": 1}},
        lag_tolerance=1.0,
        window_size=40.0,
        clock="t",
        gap_cap=50.0,
        half_life=50.0,
    )
    assert clocked["model"]["lag_tolerance"] == 1.0
    assert clocked["model"]["window_size"] == 40.0
    # A bank reports the dict each builder made.
    for s in (spec, given, clocked):
        assert rp.ModelBank([s]).specs[0] == s


def test_a_duration_lag_and_window_are_kept_as_text_on_a_temporal_clock():
    spec = rp.spec.ewm_moments(
        "mom",
        terms={"ac": {"ar": 1, "ar@5m": 1}},
        lag_tolerance="1m",
        window_size="1h",
        clock="ts",
        gap_cap="1d",
        half_life="30m",
    )
    assert spec["model"]["terms"] == {"ac": {"ar": 1, "ar@5m": 1}}
    assert spec["model"]["lag_tolerance"] == "1m"
    assert spec["model"]["window_size"] == "1h"
    assert spec["features"] == ["ar"]


def test_the_outputs_are_the_terms_in_order_then_their_errors_then_the_weight():
    terms = {"zeta": {"g": 4}, "alpha": {"g": 2}, "mid": {"g": 1, "ar": 1}}
    plain = rp.spec.ewm_moments("mom", terms=terms, half_life=50.0)
    tail = ["weight_sum", "settled_frac", "withheld_reason"]
    assert rp.spec.output_fields(plain) == ["zeta", "alpha", "mid", *tail]
    with_se = rp.spec.ewm_moments("mom", terms=terms, emit_se=True, half_life=50.0)
    want = ["zeta", "alpha", "mid", "se_zeta", "se_alpha", "se_mid", *tail]
    assert rp.spec.output_fields(with_se) == want
    out = _fit(with_se, _frame())
    assert out.columns == want
    assert all(out[c].dtype == pl.Float64 for c in want[:-1])
    # A half-life grid writes one set a half-life, each suffixed.
    grid = rp.spec.ewm_moments("mom", terms={"v": {"g": 2}}, half_life=[10.0, 100.0])
    assert rp.spec.output_fields(grid) == [
        f"{f}@{h}" for h in ("h10", "h100") for f in ("v", *tail)
    ]


# --- the refusals, by name ---------------------------------------------------

REFUSALS: list[tuple[str, dict, str]] = [
    (
        "a leg naming no feature",
        dict(terms={"m": {"q": 2}}, features=["g"]),
        'the term "m": the leg "q" names the column "q", which is not among the features',
    ),
    (
        "a lag neither a number nor a duration",
        dict(terms={"m": {"g": 1, "g@soon": 1}}),
        'the leg "g@soon" lags "g" by "soon", which is neither a number nor a duration',
    ),
    (
        "a power below 1",
        dict(terms={"v": {"g": 0}}),
        'the term "v" gives a leg the power 0, but a power is a whole number from 1',
    ),
    (
        "a negative lag",
        dict(terms={"ac": {"g": 1, "g@-1": 1}}),
        'the term "ac" lags a leg by -1, but a lag is finite and at least 0',
    ),
    (
        "a leg named twice",
        dict(terms={"ac": {"g": 1, "g@0": 1}}),
        'the term "ac" names a leg twice',
    ),
    (
        "a term with no leg",
        dict(terms={"v": {}}, features=["g"]),
        'the term "v" names no leg',
    ),
    (
        "no term",
        dict(terms={}, features=["g"]),
        "terms names no term",
    ),
    (
        "window_rows below 1",
        dict(terms={"v": {"g": 2}}, window_rows=0),
        "window_rows must be at least 1, but it is 0",
    ),
    (
        "a joint cumulant",
        dict(terms={"k": {"g": 2, "ar": 2}}, kind="cumulant"),
        'the term "k" asks for a joint cumulant of 2 legs; a cumulant is of one leg',
    ),
    (
        "bias=False past order 4",
        dict(terms={"m5": {"g": 5}}, bias=False),
        'the term "m5" asks for bias=False at order 5; the unbiased forms stop at order 4',
    ),
    (
        "a lag of part of a row without a clock",
        dict(terms={"ac": {"g": 1, "g@1.5": 1}}),
        'the term "ac" lags a leg by 1.5 rows, which is not a whole number',
    ),
    (
        "a duration lag without a clock",
        dict(terms={"ac": {"g": 1, "g@5m": 1}}),
        "terms is a duration but half_life is a plain number",
    ),
    (
        "lag_tolerance without a clock",
        dict(terms={"ac": {"g": 1, "g@1": 1}}, lag_tolerance=1.0),
        "lag_tolerance is the as-of tolerance of a lag on a clock, and this stream has none",
    ),
    (
        "window_size without a clock",
        dict(terms={"v": {"g": 2}}, window_size=10.0),
        "window_size = 10 is a span of the clock, but the spec has no clock",
    ),
    (
        "both windows",
        dict(terms={"v": {"g": 2}}, window_rows=10, window_size=10.0, clock="t", gap_cap=50.0),
        "window_rows (a count of rows) and window_size (a span of the clock) both set the window",
    ),
    (
        "a word of kind that is no kind",
        dict(terms={"v": {"g": 2}}, kind="median"),
        'kind must be "central", "raw", "standardized" or "cumulant", but it is "median"',
    ),
    (
        "a term's word of kind that is no kind",
        dict(terms={"v": {"g": 2}}, kind={"v": "median"}),
        'the term "v"\'s kind must be "central", "raw", "standardized" or "cumulant"',
    ),
    (
        "a term kind names that terms does not have",
        dict(terms={"v": {"g": 2}}, kind={"w": "raw"}),
        'kind names the term "w", which terms does not have',
    ),
    (
        "a state past max_state_bytes",
        dict(terms={"v": {"g": 2}}, window_rows=100_000, max_state_bytes=4096),
        "the state takes 2435536 bytes, over max_state_bytes (4096)",
    ),
]


@pytest.mark.parametrize(("why", "kw", "says"), REFUSALS, ids=[r[0] for r in REFUSALS])
def test_each_refusal_names_its_problem(why, kw, says):
    with pytest.raises(ValueError) as e:
        rp.spec.ewm_moments("mom", **{"half_life": 50.0, **kw})
    assert says in str(e.value), (why, str(e.value))


def test_targets_are_refused_by_name():
    with pytest.raises(TypeError, match=r"ewm_moments\(\) takes no targets"):
        rp.spec.ewm_moments("mom", terms={"v": {"g": 2}}, targets=["g"], half_life=50.0)


# --- a realistic stream against the definitions ------------------------------


def test_decayed_weighted_moments_on_an_uneven_clock_are_the_definitions():
    """Each row reads the rows before it, each weighing ``w_i 2^(-(t - t_i)/h)``:
    the central moments about their weighted mean, plug-in and unbiased, the
    raw moments and the standardized ones, against ``numpy.average``."""
    frame = _frame()
    h = 20.0
    spec = rp.spec.ewm_moments(
        "mom",
        terms={
            "m2": {"g": 2},
            "m3": {"g": 3},
            "r2": {"g": 2},
            "skew": {"g": 3},
            "kurt": {"g": 4},
            "var": {"g": 2},
            "cov": {"g": 1, "ar": 1},
        },
        kind={"r2": "raw", "skew": "standardized", "kurt": "standardized"},
        bias=True,
        clock="t",
        gap_cap=50.0,
        weight="w",
        half_life=h,
    )
    unbiased = rp.spec.ewm_moments(
        "mom", terms={"var": {"g": 2}}, clock="t", gap_cap=50.0, weight="w", half_life=h
    )
    out = _fit(spec, frame)
    var_unbiased = _fit(unbiased, frame)["var"]
    t, g, ar, w = (frame[c].to_numpy() for c in ("t", "g", "ar", "w"))
    checked = 0
    for row in range(N):
        before = w[:row] > 0
        if not before.any():
            # No weight yet: every term is withheld, the weight 0.
            assert out["m2"][row] is None and out["weight_sum"][row] == 0.0
            continue
        om = (w[:row] * np.exp2(-(t[row] - t[:row]) / h))[before]
        x, y = g[:row][before], ar[:row][before]
        mx, my = np.average(x, weights=om), np.average(y, weights=om)

        def c(p: int, x=x, mx=mx, om=om) -> float:
            return float(np.average((x - mx) ** p, weights=om))

        assert _close(out["m2"][row], c(2)), row
        assert _close(out["m3"][row], c(3)), row
        assert _close(out["r2"][row], float(np.average(x**2, weights=om))), row
        assert _close(out["cov"][row], float(np.average((x - mx) * (y - my), weights=om))), row
        if before.sum() > 1:
            assert _close(out["skew"][row], c(3) / c(2) ** 1.5), row
            assert _close(out["kurt"][row], c(4) / c(2) ** 2), row
            # Kish's correction: the variance unbiased under these weights.
            r2 = float(np.sum(om**2) / np.sum(om) ** 2)
            if 1 - r2 > 1e-6:
                assert _close(var_unbiased[row], c(2) / (1 - r2)), row
                checked += 1
    assert checked > N - 10


def test_a_lag_pairs_each_learned_row_with_the_one_before():
    """``ar@1`` without a clock is the learned row before: a row of weight 0
    is in no pair. Each pair weighs its closing row's weight, and is centred
    on the pairs' own weighted means; the standardized term is the pairs'
    weighted Pearson correlation."""
    frame = _frame()
    spec = rp.spec.ewm_moments(
        "mom",
        terms={"acov": {"ar": 1, "ar@1": 1}, "acorr": {"ar": 1, "ar@1": 1}},
        kind={"acorr": "standardized"},
        bias=True,
        weight="w",
        half_life=float("inf"),
    )
    out = _fit(spec, frame)
    ar, w = frame["ar"].to_numpy(), frame["w"].to_numpy()
    learned = np.flatnonzero(w > 0)
    for row in range(N):
        rows = learned[learned < row]
        if len(rows) < 2:
            assert out["acov"][row] is None, row
            continue
        a, b, om = ar[rows[1:]], ar[rows[:-1]], w[rows[1:]]
        ma, mb = np.average(a, weights=om), np.average(b, weights=om)
        cab = float(np.average((a - ma) * (b - mb), weights=om))
        caa = float(np.average((a - ma) ** 2, weights=om))
        cbb = float(np.average((b - mb) ** 2, weights=om))
        assert _close(out["acov"][row], cab), row
        if caa > 0 and cbb > 0:
            assert _close(out["acorr"][row], cab / math.sqrt(caa * cbb)), row
    # On 300 rows of an AR(1) at 0.6 the last row reads near it.
    assert abs(out["acorr"][-1] - 0.6) < 0.15


def test_a_row_window_reads_the_latest_learned_rows():
    frame = _frame()
    spec = rp.spec.ewm_moments(
        "mom",
        terms={"m3": {"g": 3}},
        window_rows=25,
        weight="w",
        half_life=float("inf"),
    )
    out = _fit(spec, frame)
    g, w = frame["g"].to_numpy(), frame["w"].to_numpy()
    learned = np.flatnonzero(w > 0)
    for row in range(1, N):
        rows = learned[learned < row][-25:]
        if not len(rows):
            continue
        x, om = g[rows], w[rows]
        m = np.average(x, weights=om)
        assert _close(out["m3"][row], float(np.average((x - m) ** 3, weights=om))), row
        assert _close(out["weight_sum"][row], float(om.sum()), 1e-13), row


# --- the stream's own invariants ---------------------------------------------


def test_chunks_of_any_size_give_the_same_bits():
    frame = _frame()
    spec = rp.spec.ewm_moments(
        "mom",
        terms={"skew": {"g": 3}, "ac": {"ar": 1, "ar@1.5": 1}},
        kind="standardized",
        emit_se=True,
        window_size=30.0,
        clock="t",
        gap_cap=50.0,
        weight="w",
        half_life=20.0,
    )
    whole = rp.ModelBank([spec]).fit_predict(frame)
    rng = np.random.default_rng(7)
    bank, parts, at = rp.ModelBank([spec]), [], 0
    while at < N:
        size = int(rng.integers(1, 40))
        parts.append(bank.fit_predict(frame.slice(at, size)))
        at += size
    assert pl.concat(parts).equals(whole)


def test_a_first_row_of_weight_zero_changes_no_later_row():
    frame = _frame().with_columns(pl.lit(1.0).alias("w"))
    spec = rp.spec.ewm_moments(
        "mom", terms={"m4": {"g": 4}, "co": {"g": 1, "ar": 1}}, weight="w", half_life=30.0
    )
    without = _fit(spec, frame.slice(1))
    zeroed = _fit(
        spec,
        frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(pl.col("w")).alias("w")
        ),
    )
    assert zeroed.slice(1).equals(without)
