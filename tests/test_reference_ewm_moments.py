"""The NumPy reference for ``ewm_moments`` (``tests/reference_ewm_moments.py``)
held to the third-party computations of each quantity and to the weighted
definitions written longhand (PLAN 554, part 2).

No library computes a weighted moment above order 2, so the weighted and
decayed oracle is the definition, with ``numpy.average`` doing the sums; the
unweighted special case (``w = 1``, no decay) is held to scipy and
statsmodels, which compute it.
"""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction

import numpy as np
import polars as pl
import pytest
import scipy.stats
from statsmodels.stats import moment_helpers
from statsmodels.tsa.stattools import acf, acovf

import reference_ewm_moments as rm

TIER = "essential"

N = 60


def _data(seed: int = 554, n: int = N) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    x = 3.0 + rng.gamma(2.0, 1.5, n)
    y = 0.6 * x + rng.normal(0.0, 1.0, n)
    w = rng.uniform(0.5, 2.0, n)
    w[17] = 0.0
    t = np.cumsum(rng.exponential(1.0, n))
    return {"x": list(x), "y": list(y), "w": list(w), "t": list(t)}


def _vals(outs: list[rm.Out]) -> np.ndarray:
    return np.array([np.nan if o.value is None else o.value for o in outs])


def _close(got: float, want: float, rel: float) -> bool:
    return abs(got - want) <= rel * max(abs(want), 1.0)


# --- equal weights, no decay: scipy and statsmodels --------------------------


def test_central_and_raw_moments_match_scipy_at_equal_weights():
    d = _data()
    x = np.array(d["x"])
    terms = {f"m{p}": {"x": p} for p in range(2, 7)}
    kinds = {f"m{p}": "central" for p in range(2, 7)}
    terms |= {f"r{p}": {"x": p} for p in range(2, 7)}
    kinds |= {f"r{p}": "raw" for p in range(2, 7)}
    spec = rm.EwmMoments(terms, kind=kinds, bias=True)
    out = rm.run(spec, {"x": d["x"]})
    worst = 0.0
    for i in range(N):
        for p in range(2, 7):
            want = scipy.stats.moment(x[: i + 1], order=p)
            got = out[f"m{p}"][i].value
            assert _close(got, want, 1e-12), (i, p, got, want)
            raw = scipy.stats.moment(x[: i + 1], order=p, center=0)
            worst = max(worst, abs(out[f"r{p}"][i].value - raw) / abs(raw))
    # Raw read from the central moments and the mean, against power sums:
    # 4.5e-16 here (level 6 over a spread of 2), so no power sums are kept.
    assert worst < 1e-13, worst


def test_skew_and_kurtosis_match_scipy_both_biases():
    d = _data()
    x = np.array(d["x"])
    terms = {"g1": {"x": 3}, "b2": {"x": 4}}
    biased = rm.run(rm.EwmMoments(terms, kind="standardized"), {"x": d["x"]})
    unbiased = rm.run(rm.EwmMoments(terms, kind="standardized", bias=False), {"x": d["x"]})
    assert biased["g1"][0].reason == "the spread of x is zero"
    for i in range(1, N):
        xs = x[: i + 1]
        assert _close(biased["g1"][i].value, scipy.stats.skew(xs, bias=True), 1e-12)
        k = scipy.stats.kurtosis(xs, bias=True, fisher=False)
        assert _close(biased["b2"][i].value, k, 1e-12)
    # bias=False: G1 from n = 3 and G2 from n = 4; before that the
    # condition withholds it with a reason, never a row count.
    assert (
        unbiased["g1"][1].reason == "the weight sits on too few rows for an unbiased third moment"
    )
    assert (
        unbiased["b2"][2].reason == "the weight sits on too few rows for an unbiased fourth moment"
    )
    for i in range(2, N):
        xs = x[: i + 1]
        assert _close(unbiased["g1"][i].value, scipy.stats.skew(xs, bias=False), 1e-11)
        if i >= 3:
            k = scipy.stats.kurtosis(xs, bias=False, fisher=False)
            assert _close(unbiased["b2"][i].value, k, 1e-11), i


def test_cumulants_match_kstat_and_mc2cum():
    d = _data()
    x = np.array(d["x"])
    terms = {f"k{p}": {"x": p} for p in range(2, 7)}
    plug = rm.run(rm.EwmMoments(terms, kind="cumulant", bias=True), {"x": d["x"]})
    fisher = rm.run(
        rm.EwmMoments({f"k{p}": {"x": p} for p in range(2, 5)}, kind="cumulant", bias=False),
        {"x": d["x"]},
    )
    for i in range(4, N):
        xs = x[: i + 1]
        mc = [0.0] + [scipy.stats.moment(xs, order=p) for p in range(2, 7)]
        cum = moment_helpers.mc2cum(mc)
        for p in range(2, 7):
            assert _close(plug[f"k{p}"][i].value, cum[p - 1], 1e-11), (i, p)
        for p in range(2, 5):
            assert _close(fisher[f"k{p}"][i].value, scipy.stats.kstat(xs, p), 1e-11), (i, p)


def test_unbiased_forms_are_unbiased_under_unequal_weights():
    """The expectation over every sample of a three-point distribution, at
    five unequal weights, enumerated exactly (243 samples): the unbiased
    central moments of order 2 to 4 and Fisher's k4 equal the population's."""
    support = [-1.0, 0.0, 2.0]
    probs = [0.25, 0.5, 0.25]
    weights = [1.0, 3.0, 2.0, 0.5, 4.0]
    mu = sum(p * s for p, s in zip(probs, support, strict=True))
    pairs = list(zip(probs, support, strict=True))
    pop = {k: sum(p * (s - mu) ** k for p, s in pairs) for k in (2, 3, 4)}
    pop["k4"] = pop[4] - 3.0 * pop[2] ** 2
    terms = {"c2": {"x": 2}, "c3": {"x": 3}, "c4": {"x": 4}}
    exp = {"c2": 0.0, "c3": 0.0, "c4": 0.0, "k4": 0.0}
    for idx in itertools.product(range(3), repeat=len(weights)):
        pr = math.prod(probs[i] for i in idx)
        xs = [support[i] for i in idx]
        spec = rm.EwmMoments(terms, bias=False)
        o = spec.feed({"x": xs, "w": weights})
        k4 = rm.EwmMoments({"k4": {"x": 4}}, kind="cumulant", bias=False).feed(
            {"x": xs, "w": weights}
        )
        for k in ("c2", "c3", "c4"):
            exp[k] += pr * o[k][-1].value
        exp["k4"] += pr * k4["k4"][-1].value
    assert math.isclose(exp["c2"], pop[2], rel_tol=1e-12)
    assert math.isclose(exp["c3"], pop[3], rel_tol=1e-12)
    assert math.isclose(exp["c4"], pop[4], rel_tol=1e-12)
    assert math.isclose(exp["k4"], pop["k4"], rel_tol=1e-12, abs_tol=1e-14)
    # The plug-in fourth moment is biased at these weights: the test sees it.
    plug = sum(
        math.prod(probs[i] for i in idx)
        * rm.EwmMoments({"c4": {"x": 4}}, bias=True)
        .feed({"x": [support[i] for i in idx], "w": weights})["c4"][-1]
        .value
        for idx in itertools.product(range(3), repeat=len(weights))
    )
    assert abs(plug - pop[4]) > 0.1


def test_the_unbiased_coefficients_are_exact():
    """``_e4``'s coefficients against exact rational expectations of the
    plug-in m4 and m2^2, at one weight vector and two distributions."""
    wts = [Fraction(v) for v in (2, 1, 3, 1)]
    W = sum(wts)
    p = [v / W for v in wts]
    r = [sum(q**k for q in p) for k in range(5)]
    a1, a2, b1, b2 = rm._e4(*(float(r[k]) for k in (2, 3, 4)))
    for sup, pr in (
        ([Fraction(0), Fraction(1)], [Fraction(1, 3), Fraction(2, 3)]),
        (
            [Fraction(-1), Fraction(0), Fraction(2)],
            [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)],
        ),
    ):
        m = sum(a * b for a, b in zip(sup, pr, strict=True))
        mu4 = sum(q * (s - m) ** 4 for s, q in zip(sup, pr, strict=True))
        mu22 = sum(q * (s - m) ** 2 for s, q in zip(sup, pr, strict=True)) ** 2
        e4 = e22 = Fraction(0)
        for idx in itertools.product(range(len(sup)), repeat=len(wts)):
            q = math.prod(pr[i] for i in idx)
            xs = [sup[i] for i in idx]
            mm = sum(a * b for a, b in zip(p, xs, strict=True))
            m2 = sum(a * (b - mm) ** 2 for a, b in zip(p, xs, strict=True))
            e4 += q * sum(a * (b - mm) ** 4 for a, b in zip(p, xs, strict=True))
            e22 += q * m2 * m2
        assert math.isclose(float(e4), a1 * float(mu4) + a2 * float(mu22), rel_tol=1e-14)
        assert math.isclose(float(e22), b1 * float(mu4) + b2 * float(mu22), rel_tol=1e-14)


def _full_mean_gap(x: np.ndarray, k: int) -> tuple[float, float, float]:
    """The pairs' own means and the full sample's, for lag ``k``."""
    return float(x[k:].mean()), float(x[:-k].mean()), float(x.mean())


def test_autocovariance_is_the_pairs_own_and_acovf_up_to_the_full_mean():
    """With equal weights and no decay the lag-k autocovariance is numpy's
    ``cov(x[k:], x[:-k], bias=True)``: both legs about the pairs' own means.
    statsmodels' ``acovf`` takes both about the full sample's mean ``m`` and
    divides by ``n``; the two are one identity apart,

        acovf_k = (n - k) / n * (cov_k + (m_a - m)(m_b - m)),

    with ``m_a``, ``m_b`` the pairs' leg means, and ``acf_k`` is that over
    ``acovf_0``. The identity holds to rounding; the gap itself, ``acf``
    against the pairs' Pearson correlation, is 1e-3 to 1e-2 here."""
    d = _data()
    x = np.array(d["x"])
    n = len(x)
    terms = {f"ac{k}": {"x": 1, f"x@{k}": 1} for k in (1, 2, 3)}
    terms |= {f"rho{k}": {"x": 1, f"x@{k}": 1} for k in (1, 2, 3)}
    terms["var"] = {"x": 2}
    kinds = {f"rho{k}": "standardized" for k in (1, 2, 3)}
    out = rm.run(rm.EwmMoments(terms, kind=kinds, bias=True), {"x": d["x"]})
    assert out["ac2"][1].reason == "no lagged pair yet"
    sm_cov = acovf(x, fft=False)
    sm_acf = acf(x, nlags=3, fft=False)
    gaps = []
    for k in (1, 2, 3):
        cov = out[f"ac{k}"][-1].value
        assert _close(cov, np.cov(x[k:], x[:-k], bias=True)[0, 1], 1e-13)
        ma, mb, m = _full_mean_gap(x, k)
        assert _close((n - k) / n * (cov + (ma - m) * (mb - m)), sm_cov[k], 1e-13)
        rho = out[f"rho{k}"][-1].value
        assert _close(rho, np.corrcoef(x[k:], x[:-k])[0, 1], 1e-13)
        var = out["var"][-1].value
        assert _close((n - k) / n * (cov + (ma - m) * (mb - m)) / var, sm_acf[k], 1e-13)
        gaps.append(abs(rho - sm_acf[k]))
    assert 1e-4 < max(gaps) < 0.1, gaps


# --- weights and decay: the definitions longhand -----------------------------


def _ew_weights(d, upto: int, half_life: float) -> np.ndarray:
    t = np.array(d["t"][: upto + 1])
    w = np.array(d["w"][: upto + 1])
    return w * np.exp2(-(t[-1] - t) / half_life)


def test_decayed_weighted_moments_match_numpy_on_a_clock():
    d = _data()
    x = np.array(d["x"])
    terms = {f"m{p}": {"x": p} for p in range(2, 7)} | {f"r{p}": {"x": p} for p in range(2, 7)}
    kinds = {f"r{p}": "raw" for p in range(2, 7)}
    spec = rm.EwmMoments(terms, kind=kinds, bias=True, half_life=5.0, clocked=True)
    out = rm.run(spec, d)
    for i in range(N):
        om = _ew_weights(d, i, 5.0)
        m = np.average(x[: i + 1], weights=om)
        for p in range(2, 7):
            want = np.average((x[: i + 1] - m) ** p, weights=om)
            assert _close(out[f"m{p}"][i].value, want, 1e-11), (i, p)
            want = np.average(x[: i + 1] ** p, weights=om)
            assert _close(out[f"r{p}"][i].value, want, 1e-12), (i, p)


def test_unbiased_variance_is_polars_ewm_var():
    """Decay only (``w = 1``, a row clock): ``bias=False`` is Polars'
    ``ewm_var(adjust=True, bias=False)``, Kish's correction."""
    d = _data()
    out = rm.run(rm.EwmMoments({"v": {"x": 2}}, lam=0.9), {"x": d["x"]})
    want = pl.Series(d["x"]).ewm_var(alpha=0.1, adjust=True, bias=False).to_list()
    assert (
        out["v"][0].reason == "one row carries all the weight, so the unbiased form is not defined"
    )
    for i in range(1, N):
        assert _close(out["v"][i].value, want[i], 1e-12), i


def test_cross_and_lagged_cross_moments_match_longhand():
    d = _data()
    x, y, w = np.array(d["x"]), np.array(d["y"]), np.array(d["w"])
    alphas = [(1, 1), (1, 2), (2, 2), (3, 1)]
    terms = {f"c{a}{b}": {"x": a, "y": b} for a, b in alphas}
    terms |= {f"l{a}{b}": {"x": a, "y@2": b} for a, b in alphas}
    spec = rm.EwmMoments(terms, bias=True, lam=0.95)
    out = rm.run(spec, d)
    # The learned rows (row 17 has weight 0), and their decayed weights.
    keep = w > 0
    xl, yl, wl = x[keep], y[keep], w[keep]
    n = len(xl)
    om = wl * 0.95 ** np.arange(n - 1, -1, -1)
    for a, b in alphas:
        mx, my = np.average(xl, weights=om), np.average(yl, weights=om)
        want = np.average((xl - mx) ** a * (yl - my) ** b, weights=om)
        assert _close(out[f"c{a}{b}"][-1].value, want, 1e-12)
        # The pair (x_t, y_{t-2}) over learned rows, weight w_t, about the
        # pairs' own means.
        px, py, po = xl[2:], yl[:-2], om[2:]
        mx, my = np.average(px, weights=po), np.average(py, weights=po)
        want = np.average((px - mx) ** a * (py - my) ** b, weights=po)
        assert _close(out[f"l{a}{b}"][-1].value, want, 1e-12)


# --- the degenerate cases ----------------------------------------------------


def _same(a: dict[str, list[rm.Out]], b: dict[str, list[rm.Out]]) -> bool:
    """Equal to the bit: the same values (NaN-free by construction), the same
    reasons."""
    if a.keys() != b.keys():
        return False
    for k in a:
        for p, q in zip(a[k], b[k], strict=True):
            if p.reason != q.reason:
                return False
            if (p.value is None) != (q.value is None):
                return False
            if p.value is not None and struct_bits(p.value) != struct_bits(q.value):
                return False
    return True


def struct_bits(v: float) -> bytes:
    return np.float64(v).tobytes()


SPEC = {
    "m2": {"x": 2},
    "m4": {"x": 4},
    "xy": {"x": 1, "y": 1},
    "xy2": {"x": 1, "y": 2},
    "ac1": {"x": 1, "x@1": 1},
    "ac3": {"x": 1, "x@3": 1},
}
SPEC_KINDS = {"m4": "standardized", "ac1": "standardized"}


def test_a_zero_weight_first_row_is_no_row():
    d = _data()
    d0 = {k: [v[0], *v] for k, v in d.items()}
    d0["w"][0] = 0.0
    d0["t"][0] = d["t"][0] - 5.0
    for kw in ({"half_life": 3.0, "clocked": True}, {"lam": 0.9, "window_rows": 9}):
        a = rm.run(rm.EwmMoments(SPEC, kind=SPEC_KINDS, **kw), d)
        b = rm.run(rm.EwmMoments(SPEC, kind=SPEC_KINDS, **kw), d0)
        assert b["m2"][0] == rm.Out(None, "no weight yet")
        assert b["ac1"][0] == rm.Out(None, "no lagged pair yet")
        assert _same(a, {k: v[1:] for k, v in b.items()})


def test_a_constant_input_withholds_by_the_condition_with_no_nan():
    n = 30
    data = {"x": [1e6 + 0.25] * n, "y": list(np.linspace(0.0, 1.0, n))}
    spec = rm.EwmMoments(
        SPEC | {"v": {"x": 2}, "k3": {"x": 3}},
        kind=SPEC_KINDS | {"k3": "cumulant"},
        half_life=4.0,
        se=True,
    )
    out = rm.run(spec, data)
    one = "one row carries all the weight, so the unbiased form is not defined"
    assert out["m2"][0] == out["xy"][0] == rm.Out(None, one)
    for i in range(n):
        assert out["m4"][i] == rm.Out(None, "the spread of x is zero")
        assert out["ac1"][i].value is None
        if i:
            # A variance of 0 is a value, not a withheld one.
            assert out["m2"][i] == out["xy"][i] == rm.Out(0.0, None)
            assert out["k3"][i].value == 0.0 or out["k3"][i].reason is not None
    assert out["ac1"][5].reason == "the spread of x is zero"
    flat = [v for s in spec.state_dict()["sums"] for v in s]
    assert all(math.isfinite(v) for v in flat)
    for k, col in out.items():
        assert all(o.value is None or math.isfinite(o.value) for o in col), k


def _chunks(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    out, left = [], n
    while left:
        k = int(min(left, rng.integers(1, 7)))
        out.append(k)
        left -= k
    return out


CASES = [
    {"half_life": 3.0, "clocked": True},
    {"lam": 0.9},
    {"half_life": 4.0, "window_rows": 11, "block_rows": 3},
    {"half_life": 2.0, "clocked": True, "window_clock": 7.5},
    {"lam": 0.9, "emit": "before"},
]


@pytest.mark.parametrize(
    "kw", CASES, ids=["ew-clock", "ew-rows", "win-rows", "win-clock", "before"]
)
def test_chunk_invariance_to_the_bit(kw):
    d = _data()
    d["brk"] = [i == 40 for i in range(N)]
    whole = rm.run(rm.EwmMoments(SPEC, kind=SPEC_KINDS, se=True, **kw), d)
    for seed in (1, 2):
        parts = rm.run(rm.EwmMoments(SPEC, kind=SPEC_KINDS, se=True, **kw), d, _chunks(N, seed))
        assert _same(whole, parts)
    single = rm.run(rm.EwmMoments(SPEC, kind=SPEC_KINDS, se=True, **kw), d, [1] * N)
    assert _same(whole, single)


@pytest.mark.parametrize(
    "kw", CASES, ids=["ew-clock", "ew-rows", "win-rows", "win-clock", "before"]
)
def test_a_saved_state_continues_to_the_bit(kw):
    d = _data()
    whole_spec = rm.EwmMoments(SPEC, kind=SPEC_KINDS, **kw)
    whole = rm.run(whole_spec, d)
    first = rm.EwmMoments(SPEC, kind=SPEC_KINDS, **kw)
    head = rm.run(first, {k: v[:33] for k, v in d.items()})
    saved = json.dumps(first.state_dict())
    second = rm.EwmMoments(SPEC, kind=SPEC_KINDS, **kw)
    second.load_state(json.loads(saved))
    tail = rm.run(second, {k: v[33:] for k, v in d.items()})
    assert _same(whole, {k: head[k] + tail[k] for k in whole})
    assert json.dumps(second.state_dict()) == json.dumps(whole_spec.state_dict())


def test_a_clock_lag_is_as_of_and_a_gap_past_it_leaves_no_pair():
    t = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.5, 13.0, 14.0]
    x = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0]
    brk = [False] * 7 + [True, False]
    spec = rm.EwmMoments(
        {"ac": {"x": 1, "x@1": 1}, "lagmean": {"x@1": 1}, "mean": {"x": 1}},
        kind={"lagmean": "raw", "mean": "raw"},
        bias=True,
        clocked=True,
    )
    out = rm.run(spec, {"x": x, "t": t, "brk": brk})
    # The pairs: t=1 with 0, 2 with 1, 3 with 2; t=10 has none (its as-of
    # point 9 is 6 past the latest row, at 3: a gap longer than the lag);
    # 11 with 10; 12.5 with 11 (as-of 11.5, 0.5 back); 13 has none (a break
    # cleared the buffer); 14 with 13.
    pairs = [(3.0, 1.0), (2.0, 3.0), (5.0, 2.0), (7.0, 4.0), (6.0, 7.0), (8.0, 9.0)]
    px = np.array([p[0] for p in pairs])
    py = np.array([p[1] for p in pairs])
    assert _close(out["ac"][-1].value, np.cov(px, py, bias=True)[0, 1], 1e-14)
    assert _close(out["lagmean"][-1].value, py.mean(), 1e-14)
    # Row t=10 left the lagged term where it was, and still counts for the mean.
    assert out["ac"][4] == out["ac"][3]
    assert out["mean"][4].value == pytest.approx(np.mean(x[:5]), rel=1e-15)
    assert out["ac"][0] == rm.Out(None, "no lagged pair yet")


def test_lag_tolerance_pairs_at_it_and_not_past_it():
    # Lag 2: at t = 7 the as-of point is 5 and the latest row at or before
    # it is at 4, one before the point. A tolerance of 1 pairs it (at the
    # tolerance), one of 0.5 does not; the default (the lag, 2) does.
    t = [0.0, 1.0, 4.0, 7.0]
    x = [1.0, 3.0, 2.0, 5.0]
    lagmean = {"lagmean": {"x@2": 1}}

    def last(**kw):
        spec = rm.EwmMoments(lagmean, kind="raw", bias=True, clocked=True, **kw)
        return rm.run(spec, {"x": x, "t": t})["lagmean"]

    # t = 1: as-of point -1, no row; t = 4: point 2, row 1 (1 before).
    at = last(lag_tolerance=1.0)
    assert [o.value for o in at] == [None, None, 3.0, 2.5]
    short = last(lag_tolerance=0.5)
    assert [o.value for o in short] == [None, None, None, None]
    assert [o.value for o in last()] == [o.value for o in at]
    with pytest.raises(ValueError, match="lag_tolerance"):
        rm.EwmMoments(lagmean, lag_tolerance=1.0)


def test_row_windows_match_polars_rolling():
    d = _data()
    x = np.array(d["x"])
    win = 10
    spec = rm.EwmMoments(
        {"g1": {"x": 3}, "b2": {"x": 4}, "m5": {"x": 5}},
        kind={"g1": "standardized", "b2": "standardized"},
        bias=True,
        window_rows=win,
        block_rows=3,
    )
    out = rm.run(spec, {"x": d["x"]})
    skew = pl.Series(x).rolling_skew(win, bias=True).to_list()
    kurt = pl.Series(x).rolling_kurtosis(win, bias=True, fisher=False).to_list()
    for i in range(win - 1, N):
        assert _close(out["g1"][i].value, skew[i], 1e-12), i
        assert _close(out["b2"][i].value, kurt[i], 1e-12), i
        assert _close(
            out["m5"][i].value, scipy.stats.moment(x[i - win + 1 : i + 1], order=5), 1e-12
        )


def test_clock_windows_match_longhand():
    d = _data()
    x, t, w = np.array(d["x"]), np.array(d["t"]), np.array(d["w"])
    span = 6.0
    decayed = rm.run(
        rm.EwmMoments({"m3": {"x": 3}}, bias=True, half_life=2.0, clocked=True, window_clock=span),
        d,
    )
    flat = rm.run(
        rm.EwmMoments({"v": {"x": 2}}, bias=True, clocked=True, window_clock=span),
        {"x": d["x"], "t": d["t"]},
    )
    for i in range(N):
        inside = (t > t[i] - span) & (t <= t[i]) & (w > 0)
        om = w[inside] * np.exp2(-(t[i] - t[inside]) / 2.0)
        xs = x[inside]
        if w[i] > 0:
            m = np.average(xs, weights=om)
            want = np.average((xs - m) ** 3, weights=om)
            assert _close(decayed["m3"][i].value, want, 1e-12), i
        inside = (t > t[i] - span) & (t <= t[i])
        assert _close(flat["v"][i].value, np.var(x[inside]), 1e-12), i


# --- the standard error ------------------------------------------------------


def test_the_standard_error_is_the_delta_method_at_kishs_n():
    d = _data()
    x = np.array(d["x"])
    out = rm.run(rm.EwmMoments({"m3": {"x": 3}}, bias=True, se=True), {"x": d["x"]})
    mu = [scipy.stats.moment(x, order=p) for p in range(7)]
    mu[1] = 0.0
    p = 3
    var = (
        mu[2 * p] - mu[p] ** 2 - 2 * p * mu[p - 1] * mu[p + 1] + p * p * mu[2] * mu[p - 1] ** 2
    ) / N
    assert _close(out["se_m3"][-1].value, math.sqrt(var), 1e-12)
    # Kish's n with unequal weights: the same formula over n = W^2 / sum w^2.
    out = rm.run(rm.EwmMoments({"v": {"x": 2}}, bias=True, se=True), {"x": d["x"], "w": d["w"]})
    w = np.array(d["w"])
    m = np.average(x, weights=w)
    m2, m4 = (np.average((x - m) ** k, weights=w) for k in (2, 4))
    n = w.sum() ** 2 / (w**2).sum()
    assert _close(out["se_v"][-1].value, math.sqrt((m4 - m2**2) / n), 1e-12)


def test_the_skews_standard_error_is_root_6_over_n_on_normal_rows():
    rng = np.random.default_rng(7)
    n = 3000
    x = list(rng.normal(5.0, 2.0, n))
    out = rm.run(rm.EwmMoments({"g1": {"x": 3}}, kind="standardized", se=True), {"x": x})
    assert out["se_g1"][-1].value == pytest.approx(math.sqrt(6.0 / n), rel=0.1)


# --- the state's bound and the refusals --------------------------------------


def test_the_state_is_bounded_by_the_spec_not_the_stream():
    spec = rm.EwmMoments(SPEC, kind=SPEC_KINDS, lam=0.99)
    # Summaries, 8 bytes a number: (x) to order 4 (m2, and m4
    # standardized): W, m, c2..c4, r2 = 6. (x, y) for (1, 1) and (1, 2),
    # the downward closure c11, c02, c12: W, 2 means, 3, r2 = 7. (x, x@1)
    # standardized at (1, 1): c20, c11, c02 = 7. (x, x@3) at (1, 1): c11
    # = 5. One buffer of x for both lags, as deep as the deepest: 3.
    # Three counters: the last clock, the learned rows, the ring's start.
    assert [spec.layouts[g].numbers() for g in spec.groups] == [6, 7, 7, 5]
    want = 8 * (3 + 6 + 7 + 7 + 5 + 3)
    assert spec.fixed_state_bytes() == want
    rng = np.random.default_rng(3)
    for _ in range(4):
        spec.feed({"x": list(rng.normal(size=500)), "y": list(rng.normal(size=500))})
        assert spec.state_bytes() == want
    win = rm.EwmMoments(SPEC, kind=SPEC_KINDS, lam=0.99, window_rows=20, block_rows=4)
    for _ in range(3):
        win.feed({"x": list(rng.normal(size=300)), "y": list(rng.normal(size=300))})
        assert win.state_bytes() <= win.fixed_state_bytes()


def test_refusals_name_the_problem():
    with pytest.raises(ValueError, match="bias=False at order 5"):
        rm.EwmMoments({"m5": {"x": 5}}, bias=False)
    with pytest.raises(ValueError, match="cumulant of 2 legs"):
        rm.EwmMoments({"k": {"x": 2, "y": 2}}, kind="cumulant")
    with pytest.raises(ValueError, match="over max_state_bytes"):
        rm.EwmMoments({"ac": {"x": 1, "x@100000": 1}})
    with pytest.raises(ValueError, match="lag by more than 0"):
        rm.EwmMoments({"ac": {"x": 1, "x@0": 1}})
    with pytest.raises(ValueError, match="not a whole number"):
        rm.EwmMoments({"ac": {"x": 1, "x@1.5": 1}})
    # A clock window's ring is checked as it fills.
    spec = rm.EwmMoments({"v": {"x": 2}}, clocked=True, window_clock=1e9, max_state_bytes=4096)
    with pytest.raises(ValueError, match="over max_state_bytes"):
        spec.feed({"x": [1.0] * 400, "t": [float(i) for i in range(400)]})
