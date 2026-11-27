"""E59: has the correlation structure changed?

Three tests with three nulls. The `monitor` kind is the linearised form of
Wied, Krämer & Dehling's closed-sample constancy test (docs/PLAN.md task
553), against the Kolmogorov law their statistic converges to, read on a
grid; its statistic is held to statsmodels' OLS-residual CUSUM and Bartlett
long-run variance, and its size and power to the review's measured tables.
The `sequential` kind is Wied & Galeano's detector against a stable history
(task 114) on Fisher's scale (task 553), whose critical values are held to
their Table 1 and to a simulation of the law behind it, and whose detector is
held to numpy's `arctanh` and the variance scipy's `pearsonr` puts on it. The
`window` kind measures how big a change is, against a permutation null held to
scipy's exact `permutation_test` over the blocks.
"""

import itertools
import math
import re

import numpy as np
import polars as pl
import pytest

import ledger
from rheon import polars as rp

TIER = "mixed"


def pair(n, rho, seed=0, k=2):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    x = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, k))
    return pl.DataFrame({f"x{i}": x[:, i] for i in range(k)})


def spec(**kw):
    # A span of the clock in place of the count of rows.
    d = (
        {"features": ["x0", "x1"]}
        if "window_size" in kw
        else {"features": ["x0", "x1"], "window_rows": 200}
    )
    d.update(kw)
    return rp.spec.corrchange("c", **d)


def run(df, **kw):
    return rp.ModelBank([spec(**kw)]).fit_predict(df)["c"].struct.unnest()


# --- the monitor -------------------------------------------------------------


def test_the_critical_value_is_the_kolmogorov_quantile():
    """Computed from the series, not pinned: scipy's ``kstwobign``, 1.3581 at
    5%, 1.6276 at 1%, 1.2239 at 10%, less the grid correction
    ``-zeta(1/2)/sqrt(2 pi)/sqrt(T)`` (scipy's ``zeta``): the statistic is a
    maximum over the span's ``T`` rows and the quantile is of a supremum
    over a continuum."""
    from scipy.special import zeta
    from scipy.stats import kstwobign

    shift = -zeta(0.5) / math.sqrt(2 * math.pi)
    assert shift == pytest.approx(0.5825971579390107, rel=1e-15)
    for alpha, published in ((0.05, 1.3581), (0.01, 1.6276), (0.10, 1.2239)):
        want = kstwobign.isf(alpha) - shift / math.sqrt(200)
        assert kstwobign.isf(alpha) == pytest.approx(published, abs=1e-4)
        out = run(pair(200, 0.4), window_rows=200, alpha=alpha, alpha_adjust="none")
        assert out["crit"][199] == pytest.approx(want, rel=1e-13)
    # Bonferroni over the pairs takes a smaller level, so a larger value.
    three = run(
        pair(200, 0.4, k=3),
        features=["x0", "x1", "x2"],
        window_rows=200,
        alpha=0.05,
    )
    assert three["crit"][199] > 1.3581 - shift / math.sqrt(200)


def test_nothing_is_reported_except_where_a_span_closes():
    out = run(pair(250, 0.3), window_rows=100)
    live = [i for i, v in enumerate(out["stat"].to_list()) if v is not None]
    assert live == [99, 199], live
    assert out["flag"][99] in (True, False)
    assert out["flag"][50] is None


def test_a_constant_correlation_is_not_flagged_and_a_break_is():
    """The whole point, at a break the test is built to find."""
    n = 500
    steady = run(pair(n, 0.5, seed=1), window_rows=n, alpha_adjust="none")
    assert steady["flag"][n - 1] is False, steady["stat"][n - 1]
    a = pair(n // 2, 0.2, seed=2)
    b = pair(n // 2, 0.9, seed=3)
    broken = run(pl.concat([a, b]), window_rows=n, alpha_adjust="none")
    assert broken["flag"][n - 1] is True
    assert broken["stat"][n - 1] > broken["crit"][n - 1]


def test_the_size_is_near_nominal_on_gaussian_pairs():
    """On **Gaussian** pairs the rejection rate is near the 5% it asks for.

    That is the property to hold here, and it is not quite the paper's
    table. WKD's Table 1 is for "i.i.d. bivariate `t_5` innovations" and
    reads `.040 / .035 / .041` at `T = 500` for `rho = -0.5 / 0 / 0.5`. The
    phrase does not pin the distribution down, but both readings of it, a
    shared-scale multivariate `t_5` and independent `t_5` marginals, land
    within 0.011 of that table, and neither is liberal. `docs/REGIMES.md` §2
    measures both, and Gaussian pairs, at 2000 replications. What is pinned
    here is the level the test claims for itself.

    The paper's own statistic over-rejected at `|rho| = 0.9` for `T <= 500`
    (`.142` in their own table at `T = 200`, `.26` measured at 100); the
    linear CUSUM does not, which the extended grid below holds.
    """
    n, reps, want = 500, 200, 0.05
    for rho in (0.0, 0.5):
        flags = 0
        for r in range(reps):
            out = run(pair(n, rho, seed=1000 + r), window_rows=n, alpha_adjust="none")
            # A flag withheld (null) because the scale ratio moved is not one.
            flags += int(out["flag"][n - 1] is True)
        size = flags / reps
        se = (want * (1 - want) / reps) ** 0.5
        assert abs(size - want) < 4 * se + 0.02, (rho, size, want)
        # Ten flags expected: a monitor that never flags is not "near
        # nominal", and the band above admits zero.
        assert flags >= 2, (rho, flags)


def test_the_power_is_at_least_the_papers():
    """Their Table 2: a `0.5 -> 0.7` break at `T/2` rejects `.587` of the
    time at `T = 500`.

    One-sided, and on Gaussian pairs, where this implementation is well
    above their figure (`.830` at 1000 replications). Under either reading
    of their `t_5` it is within 0.034 of it, `.582` size-adjusted for the
    shared-scale draw, measured in `docs/REGIMES.md` §3.
    """
    n, reps, want = 500, 120, 0.587
    flags = 0
    for r in range(reps):
        a = pair(n // 2, 0.5, seed=7000 + r)
        b = pair(n // 2, 0.7, seed=9000 + r)
        out = run(pl.concat([a, b]), window_rows=n, alpha_adjust="none")
        flags += int(out["flag"][n - 1] is True)
    power = flags / reps
    se = (want * (1 - want) / reps) ** 0.5
    assert power > want - 4 * se, (power, want)
    # And "well above" it, as measured: a floor of 0.7 where 0.83 is the
    # rate at 1000 replications. The paper's figure less 4 se is 0.41.
    assert power >= 0.7, (power, want)


def test_the_scalar_form_tests_the_equicorrelations_level():
    """One statistic however many columns, and it is a **mean** CUSUM: the
    equicorrelation is already one number, so a break in its level is what
    there is to find."""
    cols = [f"x{i}" for i in range(6)]
    kw = {
        "features": cols,
        "window_rows": 400,
        "scalar": True,
        "half_life": 200.0,
        "alpha_adjust": "none",
    }
    steady = run(pair(500, 0.4, seed=20, k=6), **kw)
    assert steady["stat"].drop_nulls().len() == 1, "one statistic per span"
    assert steady["flag"].drop_nulls().to_list() == [False]
    broken = run(pl.concat([pair(250, 0.1, seed=21, k=6), pair(250, 0.9, seed=22, k=6)]), **kw)
    assert broken["flag"].drop_nulls().to_list() == [True]
    assert broken["stat"].drop_nulls()[0] > steady["stat"].drop_nulls()[0]


def test_since_flag_counts_the_rows():
    """The break sits mid-span, at row 150, so the second span straddles it
    and its flag is the break's. On a span boundary, at row 200, no span
    held both regimes: the one flag fell on the last span, not the break's,
    and the restart check sat behind a condition that was false (review
    2026-10-05, TB4)."""
    a = pair(150, 0.1, seed=4)
    b = pair(250, 0.95, seed=5)
    out = run(pl.concat([a, b]), window_rows=100, alpha_adjust="none")
    rows = out.select("stat", "flag", "since_flag").drop_nulls()
    assert rows.height == 4
    # Before any flag, the count runs from the first row.
    assert rows["since_flag"][0] == 100
    # The span holding the break flags, and the count restarts after it.
    assert rows["flag"].to_list()[:2] == [False, True], rows
    assert rows["since_flag"].to_list()[1:3] == [200, 100], rows


# --- the window --------------------------------------------------------------


def test_the_window_statistic_is_the_norm_of_the_difference():
    w = 60
    a = pair(w, 0.1, seed=6, k=3)
    b = pair(w, 0.9, seed=7, k=3)
    df = pl.concat([a, b])
    out = run(
        df,
        features=["x0", "x1", "x2"],
        kind="window",
        window_rows=w,
        crit=0.5,
    )
    pre = np.corrcoef(a.to_numpy().T)
    post = np.corrcoef(b.to_numpy().T)
    iu = np.triu_indices(3, 1)
    want = np.abs(pre[iu] - post[iu]).sum()
    assert out["stat"][2 * w - 1] == pytest.approx(want, rel=1e-12)
    assert out["flag"][2 * w - 1] is True


def test_the_linf_norm_is_the_largest_pair():
    w = 50
    df = pl.concat([pair(w, 0.1, seed=8, k=3), pair(w, 0.9, seed=9, k=3)])
    cols = ["x0", "x1", "x2"]
    l1 = run(df, features=cols, kind="window", window_rows=w, crit=9.0)
    li = run(df, features=cols, kind="window", window_rows=w, crit=9.0, norm="linf")
    assert li["stat"][2 * w - 1] < l1["stat"][2 * w - 1]


def test_the_permutation_critical_value_separates_noise_from_a_break():
    """The flag rate **per row** is not `alpha`: two adjacent windows that
    slide by one row are almost the same windows, so a statistic above the
    quantile stays above it for a run of rows. What the null buys is the
    separation — measured at 9% of rows on a stationary stream against 27%
    on a broken one, with the critical value redrawn every 10 rows."""
    w = 50
    common = {
        "kind": "window",
        "window_rows": w,
        "n_perm": 100,
        "permute_every_rows": 10,
        "features": ["x0", "x1"],
    }
    steady = run(pair(400, 0.4, seed=10), **common)
    quiet = steady["flag"].drop_nulls().to_list()
    broken = run(pl.concat([pair(200, 0.0, seed=11), pair(200, 0.9, seed=12)]), **common)
    loud = broken["flag"].drop_nulls().to_list()
    assert sum(quiet) / len(quiet) < 0.15, sum(quiet) / len(quiet)
    assert sum(loud) / len(loud) > 2 * sum(quiet) / len(quiet)


@pytest.mark.parametrize("every", [1, 3, 10])
def test_the_permutation_critical_value_is_redrawn_every_permute_every_rows_reports(every):
    """S1 (docs/PLAN.md task 196): ``permute_every_rows = n`` redraws the
    critical value every ``n`` reports, as the docstring says. It was drawn
    every ``n + 1``, so ``1`` redrew every other row. Read off the ``crit``
    field: past the first ``2 * window_rows`` rows, a run of ``n`` reports
    holds one value, and the next run another."""
    w = 20
    out = run(
        pair(2 * w + 12 * every, 0.3, seed=196),
        kind="window",
        window_rows=w,
        n_perm=50,
        permute_every_rows=every,
    )
    crit = out["crit"].to_list()
    assert all(c is None for c in crit[: 2 * w]), crit[: 2 * w]
    drawn = crit[2 * w :]
    runs = [len(list(g)) for _, g in itertools.groupby(drawn)]
    assert runs[:-1] == [every] * (len(runs) - 1), runs
    assert runs[-1] <= every, runs


def test_reset_empties_the_windows_at_a_flag():
    w = 40
    df = pl.concat([pair(80, 0.0, seed=13), pair(200, 0.95, seed=14)])
    common = {"kind": "window", "window_rows": w, "crit": 0.4, "features": ["x0", "x1"]}
    kept = run(df, **common)
    cleared = run(df, change_action="reset", **common)
    # After a flag the reset run has no statistic until both windows refill.
    first = kept["flag"].to_list().index(True)
    assert cleared["stat"][first + 1] is None
    assert kept["stat"][first + 1] is not None


# --- the shared contract -----------------------------------------------------


KIND_KW = {
    "monitor": {"window_rows": 150},
    "window": {"kind": "window", "window_rows": 50, "n_perm": 30},
    "sequential": {
        "kind": "sequential",
        "window_rows": 100,
        "monitor_rows": 150,
        "boundary_gamma": 0.25,
    },
}


#: The window kind's permutation knobs off their defaults, and its reset:
#: the redraw cadence, the blocks and the generator are state that a chunk
#: boundary or a save must carry (task 278, F2).
CHUNK_KW = {
    **KIND_KW,
    "window, blocks": {
        "kind": "window",
        "window_rows": 50,
        "n_perm": 30,
        "perm_block": 7,
        "permute_every_rows": 3,
        "seed": 11,
    },
    "window, reset": {
        "kind": "window",
        "window_rows": 40,
        "n_perm": 30,
        "permute_every_rows": 5,
        "change_action": "reset",
    },
    # Task 553: each kind's span on the clock, here the row clock, whose
    # spans, histories, periods and windows are read off the rows' stamps.
    "monitor, clock": {"window_size": 149},
    "sequential, clock": {
        "kind": "sequential",
        "window_size": 99,
        "monitor_span": 149,
        "boundary_gamma": 0.25,
    },
    "window, clock": {"kind": "window", "window_size": 50, "n_perm": 30},
}


def broken():
    """1,300 rows in ten segments whose correlation alternates 0 and 0.9, so
    every kind flags and the reset fires inside each chunked run."""
    return pl.concat([pair(130, (0.0, 0.9)[i % 2], seed=40 + i) for i in range(10)])


@pytest.mark.parametrize("kind", list(CHUNK_KW))
@pytest.mark.parametrize("size", [1, 7, 13, 300, 600])
def test_chunk_invariance(kind, size):
    # One row at a time over half the frame: it holds the same boundaries,
    # five breaks, at half the essentials' time.
    df = broken().head(650 if size == 1 else 1300)
    kw = CHUNK_KW[kind]
    want = run(df, **kw)
    # Evidence the paths ran: each kind flags, and a window kind redraws.
    assert want["flag"].fill_null(False).sum() >= 2, kind
    if kw.get("kind") == "window":
        assert want["crit"].drop_nulls().n_unique() > 3, kind
    bank = rp.ModelBank([spec(**kw)])
    got = pl.concat([bank.fit_predict(df[i : i + size]) for i in range(0, df.height, size)])
    assert want.equals(got["c"].struct.unnest())


@pytest.mark.parametrize("kind", list(CHUNK_KW))
@pytest.mark.parametrize("cut", [60, 101, 170, 300, 707])
def test_save_load_mid_stream(kind, cut):
    """A state saved in a span, in a history or in a monitoring period goes
    on as the unbroken stream does; the sequential kind's critical value is
    recomputed on load, not stored. The window kind had no such test (task
    278, F2): its generator, the critical value in force and the rows since
    it was drawn are state, and a cut falls before both windows fill (60),
    on the first draw's row (101 under a window of 50) and between redraws."""
    df = broken()
    kw = CHUNK_KW[kind]
    s = spec(**kw)
    want = run(df, **kw)
    bank = rp.ModelBank([s])
    bank.fit_predict(df[:cut])
    again = rp.ModelBank.load_bytes(bank.save_bytes(), [s])
    got = again.fit_predict(df[cut:])["c"].struct.unnest()
    assert want[cut:].equals(got)


def test_a_capped_gap_abandons_the_span():
    """Task 47's signal: a span that straddles a break in the clock is not
    a span, so the ring is emptied and the statistic is deferred."""
    df = pair(300, 0.4, seed=17).with_columns(t=pl.int_range(pl.len()).cast(pl.Float64))
    kw = {"window_rows": 100, "clock": "t", "gap_cap": 5.0}
    plain = run(df, **kw)
    gapped = run(
        df.with_columns(
            t=pl.when(pl.int_range(pl.len()) >= 50).then(pl.col("t") + 500.0).otherwise(pl.col("t"))
        ),
        **kw,
    )
    assert plain["stat"][99] is not None
    assert gapped["stat"][99] is None, "the span was abandoned at the gap"
    assert gapped["stat"][149] is not None, "and a new one closed 100 rows later"


def test_a_zero_weight_row_is_not_a_row_of_the_span():
    df = pair(300, 0.4, seed=18).with_columns(w=pl.lit(1.0))
    want = run(df, window_rows=100, weight="w")
    padded = pl.concat([df[:50], df[:1].with_columns(x0=pl.lit(1e6), w=pl.lit(0.0)), df[50:]])
    got = run(padded, window_rows=100, weight="w")
    assert got["stat"][100] == want["stat"][99]


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"features": ["x0"]}, "needs at least two features"),
        ({"window_rows": 4}, "window_rows of at least 8"),
        ({"window_rows": None}, 'kind="monitor" needs window_rows, .*but it is not set'),
        ({"kind": "nope"}, 'kind must be "monitor", "sequential" or "window", but it is "nope"'),
        ({"alpha": 1.5}, "alpha must be above 0 and below 1, but it is 1.5"),
        (
            {"alpha_adjust": "nope"},
            'alpha_adjust must be "bonferroni", "holm" or "none", but it is "nope"',
        ),
        (
            {"kind": "window", "window_rows": None},
            'kind="window" needs window_rows, .*but it is not set',
        ),
        ({"kind": "window", "window_rows": 2}, "window_rows of at least 3"),
        ({"kind": "window", "window_rows": 20, "n_perm": 2}, "n_perm must be at least 20"),
        (
            {"kind": "window", "window_rows": 20, "scalar": True},
            'scalar applies only under kind="monitor" or "sequential", but kind is "window"',
        ),
        ({"half_life": 100.0}, "half_life applies to corrchange only under scalar=True"),
        ({"emit_sigma": True}, "does not apply to corrchange"),
        # docs/REVIEW-E54-E64.md CC1 and CC3: a critical value that never
        # flags or always does, and the permutation knobs.
        ({"kind": "window", "crit": float("nan")}, "crit must be a number, but it is NaN"),
        ({"kind": "window", "crit": 0.0}, "crit must be finite and above 0, but it is 0"),
        ({"kind": "sequential", "crit": -1.0}, "crit must be finite and above 0, but it is -1"),
        ({"kind": "window", "crit": float("inf")}, "crit must be finite"),
        ({"bandwidth": 0}, "bandwidth must be at least 1, but it is 0"),
        (
            {"kind": "window", "window_rows": 20, "perm_block": 0},
            r"perm_block must be from 1 to window_rows \(20\), but it is 0",
        ),
        (
            {"kind": "window", "window_rows": 20, "perm_block": 21},
            r"perm_block must be from 1 to window_rows \(20\), but it is 21",
        ),
        (
            {"kind": "window", "window_rows": 20, "permute_every_rows": 0},
            "permute_every_rows must be at least 1, but it is 0",
        ),
    ],
)
def test_a_bad_spec_is_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


@pytest.mark.parametrize(
    ("kind", "kw", "applies_to"),
    [
        ("monitor", {"crit": 2.0}, '"window" or "sequential"'),
        ("monitor", {"change_action": "reset"}, '"window"'),
        ("monitor", {"n_perm": 50}, '"window"'),
        ("monitor", {"seed": 3}, '"window"'),
        ("monitor", {"norm": "linf"}, '"window"'),
        ("monitor", {"perm_block": 2}, '"window"'),
        ("monitor", {"permute_every_rows": 10}, '"window"'),
        ("monitor", {"monitor_rows": 50}, '"sequential"'),
        ("monitor", {"boundary_gamma": 0.2}, '"sequential"'),
        ("window", {"bandwidth": 3}, '"monitor" or "sequential"'),
        # "window"'s permutation quantile is taken at `alpha`, so a spread over
        # the pairs changed nothing there (task 160, CD4).
        ("window", {"alpha_adjust": "none"}, '"monitor" or "sequential"'),
        ("window", {"monitor_rows": 50}, '"sequential"'),
        ("window", {"boundary_gamma": 0.2}, '"sequential"'),
        ("sequential", {"n_perm": 50}, '"window"'),
        ("sequential", {"change_action": "reset"}, '"window"'),
        ("sequential", {"norm": "linf"}, '"window"'),
        ("sequential", {"seed": 1}, '"window"'),
    ],
)
def test_a_parameter_of_another_kind_is_refused(kind, kw, applies_to):
    """A parameter that belongs to another kind is refused, naming the kinds
    it applies to. They were taken and ignored: a ``crit`` given to
    ``"monitor"`` changed nothing, and the docstring's promise that
    ``reset`` under ``"monitor"`` raises was not kept (task 114)."""
    param = next(iter(kw))
    # `change_action` is named with its value, `change_action="reset"`.
    named = f'{param}="{kw[param]}"' if param == "change_action" else param
    with pytest.raises(ValueError, match=f"{named} applies only under kind={applies_to}"):
        spec(kind=kind, window_rows=50, **kw)


@pytest.mark.parametrize(
    "kw", [{"boundary_gamma": 0.5}, {"boundary_gamma": 0.495}, {"boundary_gamma": -0.1}]
)
def test_the_boundary_exponent_is_at_most_0_49(kw):
    """W&G allow up to 1/2, but the critical value is solved for, and the
    solve's work grows as 1/(1/2 - gamma): minutes to hours past 0.49
    (task 160, CD1)."""
    with pytest.raises(ValueError, match="boundary_gamma must be from 0 to 0.49"):
        spec(kind="sequential", window_rows=50, **kw)


def test_the_monitoring_period_holds_a_row_the_detector_reads():
    """A pair's period of fewer than four rows would report nothing, ever:
    Fisher's ``z`` of ``k`` rows carries the information ``k - 3``. Four is
    accepted, as two is under ``scalar``, whose mean is read from its second
    row."""
    with pytest.raises(ValueError, match="monitor_rows of at least 4, but it is 3"):
        spec(kind="sequential", window_rows=50, monitor_rows=3)
    spec(kind="sequential", window_rows=50, monitor_rows=4)
    with pytest.raises(ValueError, match="monitor_rows of at least 2, but it is 1"):
        spec(kind="sequential", window_rows=50, monitor_rows=1, scalar=True)
    spec(kind="sequential", window_rows=50, monitor_rows=2, scalar=True)


# --- the sequential detector (Wied & Galeano 2013) ---------------------------


def _sup_abs_bm_cdf(x):
    """``P(sup_{0<=s<=1} |W(s)| <= x)``, the series (Feller 1951), written
    from the formula rather than read from the library."""
    k = np.arange(400)
    n = 2 * k + 1
    return float(4 / np.pi * np.sum((-1.0) ** k / n * np.exp(-(n**2) * np.pi**2 / (8 * x * x))))


def _sup_abs_bm_quantile(p):
    lo, hi = 0.0, 20.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _sup_abs_bm_cdf(mid) < p else (lo, mid)
    return 0.5 * (lo + hi)


def _sequential_crit(m, ratio, gamma=0.0, **kw):
    out = run(
        pair(m + 5, 0.3),
        kind="sequential",
        window_rows=m,
        monitor_rows=round(ratio * m),
        boundary_gamma=gamma,
        alpha_adjust="none",
        **kw,
    )
    crits = out["crit"].drop_nulls().unique()
    assert crits.len() == 1, crits
    return crits[0]


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 4.0])
def test_the_sequential_critical_value_at_gamma_zero_is_the_series(ratio):
    """Wied & Galeano's Eq. 7 at ``γ = 0``: ``sqrt(T/(1+T))`` times the 95 %
    point of ``sup|W|`` on ``[0, 1]``, 2.2414, with ``T`` the period's
    length in the detector's own time: the information ``(monitor_rows -
    3) / (window_rows - 3)`` Fisher's ``z`` carries, 1 where the two
    counts are equal."""
    t = (round(ratio * 100) - 3) / (100 - 3)
    want = (t / (1 + t)) ** 0.5 * _sup_abs_bm_quantile(0.95)
    assert _sequential_crit(100, ratio) == pytest.approx(want, abs=1e-6)
    assert _sup_abs_bm_quantile(0.95) == pytest.approx(2.2414, abs=1e-4)


#: Wied & Galeano's Table 1: 5 % critical values, simulated from 10,000
#: paths on a grid of 10,000 points, for T = 0.5, 1, 2, 4.
TABLE_1 = {
    0.0: (1.2870, 1.5578, 1.8158, 1.9980),
    0.25: (1.8001, 1.9924, 2.1684, 2.2467),
    0.45: (2.6282, 2.6844, 2.7215, 2.7660),
}


@pytest.mark.parametrize("gamma", sorted(TABLE_1))
def test_the_sequential_critical_values_are_wied_and_galeanos_table_1(gamma):
    """Within 0.035 of every cell. Theirs sit below ours in 11 of 12, since a
    grid reads the supremum low; the twelfth is above by 0.015, in a row of
    theirs off the exact ``T``-scaling by 0.027, the spread 10,000 paths
    leave."""
    for ratio, theirs in zip((0.5, 1.0, 2.0, 4.0), TABLE_1[gamma], strict=True):
        ours = _sequential_crit(100, ratio, gamma)
        assert abs(ours - theirs) < 0.035, (gamma, ratio, ours, theirs)


def test_the_boundary_law_matches_a_simulation():
    """A second opinion on the solve above ``γ = 0``, independent of it:
    20,000 Brownian paths on a grid of 4,000 points, the 95 % point of
    ``sup |W(s)| / s^0.25``. A grid reads the supremum low (by about
    ``0.58 / sqrt(4000)`` at ``γ = 0``) and the quantile of 20,000 draws is
    good to about 0.015, so the solve sits above the simulation, by no
    more than 0.05."""
    gamma, n, batch = 0.25, 4000, 2000
    rng = np.random.default_rng(114)
    s = np.arange(1, n + 1) / n
    sups = []
    for _ in range(10):
        w = np.cumsum(rng.standard_normal((batch, n)), axis=1) / np.sqrt(n)
        sups.append(np.max(np.abs(w) / s**gamma, axis=1))
    simulated = float(np.quantile(np.concatenate(sups), 0.95))
    ours = _sequential_crit(100, 1.0, gamma) / 0.5 ** (0.5 - gamma)
    assert simulated - 0.015 < ours < simulated + 0.05, (ours, simulated)


def test_nothing_is_reported_in_the_history_or_on_the_first_three_monitored_rows():
    """Fisher's ``z`` of ``k`` rows carries the information ``k - 3``: none on
    the first three monitored rows, which report nothing, as the first did
    when a correlation of two rows was compared."""
    out = run(pair(260, 0.3), kind="sequential", window_rows=100, monitor_rows=100, crit=1e9)
    live = [i for i, v in enumerate(out["stat"].to_list()) if v is not None]
    # History 0..99, monitored rows 100..102 silent, 103..199 report, then a
    # new history from row 200.
    assert live == list(range(103, 200)), live[:5]


def test_the_sequential_size_is_near_nominal_on_gaussian_pairs():
    """Under the null the share of monitoring periods that end in a flag is
    the size. On about 600 cycles of i.i.d. Gaussian pairs, ``m = 250``,
    ``T = 1``, ``γ = 0``, at 5 %: a rate from 600 cycles has a standard
    error of 0.009."""
    m = 250
    out = run(pair(300_000, 0.4, seed=31), kind="sequential", window_rows=m, alpha_adjust="none")
    live = out["stat"].is_not_null()
    periods = (live & ~live.shift(1, fill_value=False)).sum()
    flags = out["flag"].fill_null(False).sum()
    size = flags / periods
    assert periods > 500
    assert 0.025 < size < 0.075, (flags, periods, size)


def _break(m, before, after, rho0, rho1, seed):
    return pl.concat([pair(m + before, rho0, seed=seed), pair(after, rho1, seed=seed + 1)])


def test_a_sequential_break_is_flagged_and_dated():
    """History and 200 monitored rows at 0.3, then 0.8: a flag after the
    break, dated by Eq. 8 to within 50 rows of it."""
    m = 300
    out = run(_break(m, 200, 400, 0.3, 0.8, 7), kind="sequential", window_rows=m, monitor_rows=600)
    flagged = out.with_row_index("i").filter(pl.col("flag").fill_null(False))
    assert flagged.height >= 1
    first = flagged.row(0, named=True)
    tau = first["i"] - m + 1  # the flag's place in the monitoring period
    assert tau > 200, tau
    assert abs((tau - first["since_change"]) - 200) <= 50, first


def test_a_larger_boundary_exponent_finds_an_early_change_sooner():
    """The trade ``boundary_gamma`` makes, as the paper measured it: a
    change soon after the history (here 10 rows in) is flagged sooner at
    0.45 than at 0, over twenty streams."""
    m = 250
    delays = {0.0: [], 0.45: []}
    for seed in range(20):
        df = _break(m, 10, 490, 0.2, 0.8, 100 + 2 * seed)
        for gamma in delays:
            out = run(
                df, kind="sequential", window_rows=m, monitor_rows=500, boundary_gamma=gamma
            ).with_row_index("i")
            hit = out.filter(pl.col("flag").fill_null(False))
            delays[gamma].append(hit["i"][0] - m - 10 if hit.height else 500)
    assert np.median(delays[0.45]) < np.median(delays[0.0]), delays


def test_since_change_is_null_except_on_a_flag():
    for kind, kw in KIND_KW.items():
        out = run(_break(200, 150, 250, 0.1, 0.9, 3), **kw)
        flags = out["flag"].fill_null(False)
        dated = out["since_change"]
        assert dated.filter(~flags).null_count() == (~flags).sum(), kind
        assert flags.any(), kind
        assert dated.filter(flags).drop_nulls().min() >= 1, kind


# --- the units of the data ---------------------------------------------------


def test_the_monitor_statistic_is_free_of_the_columns_units():
    """A correlation is scale-free, so the monitor's statistic and its
    critical value must not move when one column is rescaled. The
    delta-method gradient carried ``σ_y²`` and ``σ_x²`` in the wrong
    places, so on anything but unit-variance columns ``D̂`` -- and with it
    ``Q`` -- depended on the data's units, and the paper's tables were
    reproduced only where the defect vanished (review 2026-09-18, S3)."""
    n = 300
    df = pair(n, 0.4, seed=19)
    scaled = df.with_columns(pl.col("x1") * 100.0)
    a = run(df, window_rows=n, alpha_adjust="none")
    b = run(scaled, window_rows=n, alpha_adjust="none")
    assert a["stat"][n - 1] is not None, "the span did not close"
    assert b["stat"][n - 1] == pytest.approx(a["stat"][n - 1], rel=1e-9)
    assert b["crit"][n - 1] == a["crit"][n - 1]
    assert b["flag"][n - 1] == a["flag"][n - 1]


def test_the_size_is_near_nominal_on_columns_of_different_scale():
    """The size test above, on ``(x, 100·y)``: the level a correlation test
    claims cannot depend on the units the columns arrive in (S3)."""
    n, reps, want = 500, 200, 0.05
    flags = 0
    for r in range(reps):
        df = pair(n, 0.5, seed=3000 + r).with_columns(pl.col("x1") * 100.0)
        out = run(df, window_rows=n, alpha_adjust="none")
        flags += int(out["flag"][n - 1])
    size = flags / reps
    se = (want * (1 - want) / reps) ** 0.5
    assert abs(size - want) < 4 * se + 0.02, (size, want)
    # The wrong gradient inflated `D̂` by ~1e4 here, so `Q` never reached the
    # critical value and the size read zero -- inside the band above.
    assert flags >= 2, flags


# --- every parameter, without a library (task 278, F2) -----------------------
#
# No third-party library computes any of the three tests, so each parameter
# is held to a relation it must keep: an equivalent spelling, a monotone
# effect, a limit with a hand-checkable answer, a definition written from the
# paper, its boundaries, and an activation test showing it moves what it
# should and nothing else.

#: Two windows of 50 over a stream whose correlation moves 0 -> 0.6 -> 0,
#: with the permutation quantile in force on most rows.
WINDOW = {"kind": "window", "window_rows": 50, "features": ["x0", "x1"]}


def three_regimes():
    return pl.concat([pair(300, 0.0, seed=1), pair(300, 0.6, seed=2), pair(300, 0.0, seed=3)])


def test_the_seed_reproduces_and_moves_only_the_critical_value():
    """``seed`` seeds the draws: the same seed twice is the same frame,
    ``None`` is 0, and so is leaving it out. Another seed moves the critical
    value and what it decides -- ``flag``, ``since_flag``, ``since_change``
    -- and never ``stat`` or ``weight_sum``, which the data alone set."""
    df = three_regimes()
    default = run(df, **WINDOW)
    assert default.equals(run(df, **WINDOW))
    assert default.equals(run(df, **WINDOW, seed=None))
    assert default.equals(run(df, **WINDOW, seed=0))
    other = run(df, **WINDOW, seed=7)
    for col in ("stat", "weight_sum"):
        assert other[col].equals(default[col]), col
    crit_a, crit_b = default["crit"].to_numpy(), other["crit"].to_numpy()
    live = np.isfinite(crit_a)
    assert np.array_equal(live, np.isfinite(crit_b))
    moved = int((crit_a[live] != crit_b[live]).sum())
    print(f"seed 0 vs 7: crit differs on {moved} of {live.sum()} rows")
    assert moved > 0.9 * live.sum()
    assert not other["flag"].equals(default["flag"])


@pytest.mark.parametrize(
    ("knob", "values"),
    [("n_perm", (20, 1000)), ("perm_block", (1, 10)), ("permute_every_rows", (1, 50))],
)
def test_each_permutation_knob_moves_the_critical_value_and_not_the_statistic(knob, values):
    """Activation: each of the permutation quantile's knobs changes the
    critical value in force, at the same seed, and leaves the statistic --
    a function of the two windows alone -- where it was."""
    df = three_regimes()
    a, b = (run(df, **WINDOW, **{knob: v}) for v in values)
    assert a["stat"].equals(b["stat"])
    crit_a, crit_b = a["crit"].to_numpy(), b["crit"].to_numpy()
    live = np.isfinite(crit_a) & np.isfinite(crit_b)
    assert live.sum() > 700
    assert (crit_a[live] != crit_b[live]).mean() > 0.9, knob


def test_n_perm_tightens_the_critical_value_around_its_limit():
    """The essentials' share of the grid below: on one pool, the first
    critical value drawn over 20 seeds spreads less at 2,000 draws than at
    20. An empirical quantile's error falls like ``1 / sqrt(n_perm)``, so the
    ratio is 10 in the limit; measured 8.2 over these 20 seeds and 7.6 over
    50 (2026-10-25)."""
    df = pair(101, 0.4, seed=280, k=3)
    kw = {"kind": "window", "window_rows": 50, "features": ["x0", "x1", "x2"]}
    spread = {}
    for n in (20, 2000):
        specs = [rp.spec.corrchange(f"s{s}", n_perm=n, seed=s, **kw) for s in range(20)]
        out = rp.ModelBank(specs).fit_predict(df)
        spread[n] = np.std([out[f"s{s}"].struct.field("crit")[100] for s in range(20)])
    print(f"crit's spread over 20 seeds: {spread}")
    assert spread[20] > 4 * spread[2000], spread


N_PERM_W, N_PERM_K = 50, 3
N_PERM_SIZES = (20, 200, 2_000, 20_000)


def n_perm_crits(n_perm: int, seeds: range) -> np.ndarray:
    """The critical value each seed draws at ``n_perm`` on the grid's pool,
    where its first window closes."""
    w, k = N_PERM_W, N_PERM_K
    df = pair(2 * w + 1, 0.4, seed=280, k=k)
    cols = [f"x{i}" for i in range(k)]
    specs = [
        rp.spec.corrchange(
            f"s{s}", features=cols, kind="window", window_rows=w, n_perm=n_perm, seed=s
        )
        for s in seeds
    ]
    out = rp.ModelBank(specs).fit_predict(df)
    return np.array([out[f"s{s}"].struct.field("crit")[2 * w] for s in seeds])


#: The ledger's inputs (tests/ledger.py): thirty seeds' critical values at
#: every size, where the grid draws a hundred.
ledger.guard(
    "n_perm",
    lambda: [n_perm_crits(n, range(30)) for n in N_PERM_SIZES],
    oracles=["numpy"],
)


@pytest.mark.oracle_grid(reason="100 seeds at four sizes of n_perm, and 400,000 reference draws")
def test_n_perm_converges_at_the_rate_of_an_empirical_quantile():
    """The critical value is the empirical ``1 - alpha`` quantile of
    ``n_perm`` permutations of the pooled rows. As ``n_perm`` grows it
    converges to the quantile of the permutation distribution, its error
    falling like ``1 / sqrt(n_perm)``.

    The reference is that distribution's quantile from 400,000 draws in
    NumPy (the statistic from ``np.corrcoef``'s definition, the quantile
    ``inverted_cdf``, which is the empirical quantile's definition), good to
    about 0.002: two reference seeds differ by 0.0023. Judged by the worst of
    100 seeds and the 95th percentile, at each size. Measured (2026-10-26,
    the critical value the ``floor(alpha (n_perm + 1))``-th largest draw),
    worst / p95: 20 draws 0.576 / 0.320 (biased high: the largest of 20,
    where the old rule's 19th of 20 gave 0.261 / 0.215, biased low); 200,
    0.168 / 0.098; 2,000, 0.038 / 0.028; 20,000, 0.017 / 0.0090 -- a slope
    of -0.52 at the 95th percentile from 200 to 20,000.
    """
    w, k, alpha = N_PERM_W, N_PERM_K, 0.05
    df = pair(2 * w + 1, 0.4, seed=280, k=k)
    pool = df.to_numpy()[: 2 * w]
    iu = np.triu_indices(k, 1)
    rng = np.random.default_rng(278)
    draws = []
    for _ in range(20):
        perm = np.argsort(rng.random((20_000, 2 * w)), axis=1)
        rows = pool[perm]
        r = [np.array([np.corrcoef(b.T)[iu] for b in half]) for half in (rows[:, :w], rows[:, w:])]
        draws.append(np.abs(r[0] - r[1]).sum(axis=1))
    limit = float(np.quantile(np.concatenate(draws), 1 - alpha, method="inverted_cdf"))

    seeds = range(100)
    worst, p95 = {}, {}
    for n in N_PERM_SIZES:
        err = np.abs(n_perm_crits(n, seeds) - limit)
        worst[n], p95[n] = float(err.max()), float(np.quantile(err, 0.95))
        ledger.record("n_perm", f"crit_error_at_{n}", err)
    print(f"limit {limit:.4f}; worst {worst}; p95 {p95}")
    # The rate: 100x the draws, a tenth of the error, at the tail; a
    # slope of -0.35 or steeper on a log-log line through 200 and 20,000.
    assert np.log(p95[20_000] / p95[200]) / np.log(100) < -0.35, p95
    assert worst[20] > worst[200] > worst[2_000] > worst[20_000], worst
    # The endpoint: 20,000 draws' worst seed within 0.03 of the limit
    # (measured 0.017; 0.0045 of it is the draws' own sd, 0.002 the
    # reference's).
    assert worst[20_000] < 0.03, worst


def test_a_block_of_a_whole_window_makes_the_quantile_the_observed_statistic():
    """``perm_block = window_rows`` leaves two blocks, so every draw is the
    two windows as they are or swapped, and the statistic is the same both
    ways: the critical value drawn after a row is that row's statistic, so
    the next row flags just when its statistic rose. A report's statistic
    is merged from its windows' block sums (task 328) and a draw's is read
    from its shuffled rows whole, so the two agree to rounding, not to the
    bit: within 1e-14 here, and the flag is held where the statistic moved
    by more than that."""
    w = 30
    out = run(three_regimes(), **{**WINDOW, "window_rows": w}, perm_block=w, permute_every_rows=1)
    stat, crit, flag = (out[c].to_list() for c in ("stat", "crit", "flag"))
    assert crit[2 * w - 1] is None
    held = 0
    for t in range(2 * w - 1, len(stat) - 1):
        assert crit[t + 1] == pytest.approx(stat[t], rel=1e-14, abs=1e-15), t
        if abs(stat[t + 1] - stat[t]) > 1e-14 * (1 + stat[t]):
            assert flag[t + 1] == (stat[t + 1] > stat[t]), t
            held += 1
    assert held > len(stat) // 2, held


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (
            {"n_perm": 19},
            "n_perm must be at least 20 for a permutation critical value, but it is 19",
        ),
        (
            {"n_perm": 2**20 + 1},
            r"n_perm must be at most 1048576, but it is 1048577: every draw is kept, and each "
            r"takes a pass over the rows\. Lower n_perm to 1048576 or below\.$",
        ),
        ({"perm_block": 31}, r"perm_block must be from 1 to window_rows \(30\), but it is 31"),
        ({"alpha": 0.0}, "alpha must be above 0 and below 1, but it is 0"),
        ({"alpha": 1.0}, "alpha must be above 0 and below 1, but it is 1"),
    ],
)
def test_the_permutation_knobs_bounds_are_refused_by_name(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(kind="window", window_rows=30, **kw)


def test_the_permutation_knobs_bounds_are_accepted():
    """20 and 2^20 draws, a block of the whole window and of one row: each
    bound is inclusive, and each draws a critical value. 2^20 draws over
    windows of 3 rows, so the ceiling runs in a second."""
    for w, kw in (
        (30, {"n_perm": 20}),
        (3, {"n_perm": 2**20}),
        (30, {"perm_block": 30}),
        (30, {"perm_block": 1}),
    ):
        out = run(pair(2 * w + 1, 0.3, seed=5), kind="window", window_rows=w, **kw)
        assert out["stat"].drop_nulls().len() == 2, kw
        assert out["crit"][2 * w] is not None, kw


def test_alpha_nests_the_flags():
    """A higher level never flags less, row by row. Under ``"monitor"`` the
    statistic does not read ``alpha`` and the Kolmogorov quantile falls as
    it rises; the check on the two columns' scale ratio is read at the same
    level, so a span past the critical value at one level is past it at a
    higher one, flagged or its flag withheld (null). Under ``"window"`` the
    same seed gives the same draws, and the level picks a lower order
    statistic of them, so the critical value falls on every row."""
    spans = pl.concat([pair(50, (0.1, 0.7)[i % 2], seed=100 + i) for i in range(80)])
    window = three_regimes()
    for frame, kw in (
        (spans, {"window_rows": 100, "alpha_adjust": "none"}),
        # 999 draws, so 0.001 can flag: fewer than ceil(1/alpha) - 1 is refused.
        (window, {**WINDOW, "permute_every_rows": 5, "n_perm": 999}),
    ):
        flags, crits = [], []
        for alpha in (0.001, 0.01, 0.05, 0.2, 0.5):
            out = run(frame, alpha=alpha, **kw)
            past = out["flag"].fill_null(out["stat"].is_not_null() & out["flag"].is_null())
            flags.append(past.fill_null(False).to_numpy())
            crits.append(out["crit"].to_numpy())
        counts = [int(f.sum()) for f in flags]
        print(kw.get("kind", "monitor"), "flags at rising alpha:", counts)
        assert counts[0] < counts[-1], counts
        for lo, hi, c_lo, c_hi in zip(flags, flags[1:], crits, crits[1:], strict=False):
            assert np.all(hi >= lo)
            live = np.isfinite(c_lo)
            assert np.all(c_hi[live] < c_lo[live])


@pytest.mark.parametrize("kind", ["monitor", "sequential"])
@pytest.mark.parametrize("k", [3, 4])
def test_bonferroni_at_alpha_is_none_at_alpha_over_the_pairs(kind, k):
    """``alpha_adjust="bonferroni"`` tests each pair at ``alpha / npairs``,
    so it is ``"none"`` at that level, to the bit, on every field."""
    cols = [f"x{i}" for i in range(k)]
    # The break mid-span, at row 375, so a span straddles it.
    df = pl.concat([pair(375, 0.2, seed=9, k=k), pair(525, 0.8, seed=10, k=k)])
    kw = {"features": cols, "kind": kind, "window_rows": 150}
    adjusted = run(df, alpha=0.05, **kw)
    plain = run(df, alpha=0.05 / (k * (k - 1) // 2), alpha_adjust="none", **kw)
    assert adjusted.equals(plain)
    assert adjusted["flag"].fill_null(False).any(), "a flag, so the level was read"
    # And it is an adjustment: "none" at alpha itself is a lower value.
    assert run(df, alpha=0.05, alpha_adjust="none", **kw)["crit"].max() < adjusted["crit"].min()


@pytest.mark.parametrize("kind", ["monitor", "sequential"])
@pytest.mark.parametrize("m", [8, 20, 21, 150])
def test_the_default_bandwidth_is_the_floor_of_ln_m(kind, m):
    """Left out, the Bartlett bandwidth is ``floor(ln window_rows)``: 2 at 8
    and at 20, 3 at 21 (``e^3`` is 20.09), 5 at 150. The explicit value is
    the same frame to the bit, and one either side of it is not, so the
    equality is the kernel reading the value."""
    df = pl.concat([pair(300, 0.2, seed=11), pair(300, 0.7, seed=12)])
    kw = {"kind": kind, "window_rows": m}
    want = math.floor(math.log(m))
    default = run(df, **kw)
    assert default.equals(run(df, **kw, bandwidth=want))
    for other in (want - 1, want + 1):
        if other >= 1:
            assert not default["stat"].equals(run(df, **kw, bandwidth=other)["stat"]), other


def _span_series(s):
    """The span's influence series as numpy writes it: ``ξₜ = x̃ₜỹₜ − ρ̂ −
    (ρ̂/2)(x̃ₜ² − 1 + ỹₜ² − 1)`` and ``dₜ = x̃ₜ² − ỹₜ²``, the span standardized
    by its mean and (biased) standard deviation."""
    z = (s - s.mean(axis=0)) / s.std(axis=0)
    rho = float((z[:, 0] * z[:, 1]).mean())
    xi = z[:, 0] * z[:, 1] - rho - rho / 2 * (z[:, 0] ** 2 - 1 + z[:, 1] ** 2 - 1)
    return xi, z[:, 0] ** 2 - z[:, 1] ** 2


@pytest.mark.parametrize("bandwidth", [1, 3, 10, None])
def test_the_monitor_statistic_is_its_definition_with_statsmodels_bartlett(bandwidth):
    """``bandwidth`` (task 320, rank 13) through the whole statistic, each
    span's linear CUSUM ``max_j |Σ_{t≤j} ξₜ| / (√T·D̂)`` (task 553): ``D̂²``
    statsmodels' Bartlett long-run variance ``S_hac_simple(ξ, nlags=γ − 1) /
    T`` of the delta-method series of Wied, Krämer and Dehling. statsmodels
    weighs lag ``l`` by ``1 − l/(nlags + 1)``, so ``nlags = γ − 1`` is this
    kernel's ``1 − l/γ``; unset, ``γ = ⌊ln T⌋``. A correlation break inside
    the third span puts its statistic past the 5% critical value."""
    from statsmodels.stats.sandwich_covariance import S_hac_simple

    span = 200
    df = pl.concat([pair(500, 0.05, seed=4), pair(300, 0.85, seed=5)])
    X = df.to_numpy()
    kw = {} if bandwidth is None else {"bandwidth": bandwidth}
    out = run(df, window_rows=span, alpha_adjust="none", **kw)
    stat = out["stat"].to_numpy().astype(float)
    closes = np.flatnonzero(np.isfinite(stat))
    assert closes.tolist() == [199, 399, 599, 799]
    gamma = bandwidth or math.floor(math.log(span))
    qs = []
    for c in closes:
        xi, _ = _span_series(X[c - span + 1 : c + 1])
        d = math.sqrt(S_hac_simple(xi, nlags=gamma - 1)[0, 0] / span)
        qs.append(np.max(np.abs(np.cumsum(xi))) / (math.sqrt(span) * d))
    np.testing.assert_allclose(stat[closes], qs, rtol=1e-12)
    crit = out["crit"][199]
    assert stat[599] > crit > max(stat[199], stat[399], stat[799]), stat[closes]


@pytest.mark.parametrize("seed", range(6))
def test_the_monitor_cusum_is_statsmodels_ols_residual_cusum(seed):
    """At ``bandwidth = 1`` the statistic is statsmodels'
    ``breaks_cusumolsresid(ξ, ddof=0)``, the OLS-residual CUSUM ``max_j
    |Σ_{t≤j} uₜ| / √(Σu²)`` (Ploberger and Krämer 1992), on the delta-method
    series: ``ξ`` sums to 0 over the span, as residuals do, and ``D̂²`` at
    one lag is ``Σξ²/T``. Its p-value is scipy's ``kstwobign``, the limit the
    critical value is read from. And the check on the scale ratio is the
    same CUSUM of ``d = x̃² − ỹ²``: a span whose correlation CUSUM passes
    the critical value flags where ``d``'s does not, and its flag is
    withheld (null) where ``d``'s does too -- on streams whose correlation
    breaks, and whose one column's scale doubles, at spans of 30 to 400
    rows."""
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    rng = np.random.default_rng(553 + seed)
    span = [30, 60, 120, 200, 300, 400][seed]
    n = 6 * span
    a, e = rng.standard_normal((2, n))
    rho = np.where((np.arange(n) // (span // 2)) % 3 == 0, 0.2, 0.8)
    x = a * np.where(np.arange(n) % (2 * span) < span // 2, 1.0, 2.0)
    y = rho * a + np.sqrt(1 - rho**2) * e
    df = pl.DataFrame({"x0": x, "x1": y})
    out = run(df, window_rows=span, bandwidth=1, alpha_adjust="none")
    stat = out["stat"].to_numpy().astype(float)
    crit = out["crit"].to_numpy().astype(float)
    closes = np.flatnonzero(np.isfinite(stat))
    assert closes.size == 6
    seen = set()
    for c in closes:
        xi, d = _span_series(np.column_stack([x, y])[c - span + 1 : c + 1])
        q, pval, _ = breaks_cusumolsresid(xi, ddof=0)
        assert stat[c] == pytest.approx(q, rel=1e-12)
        r, _, _ = breaks_cusumolsresid(d, ddof=0)
        flag = out["flag"][int(c)]
        want = bool(q > crit[c] and not r > crit[c])
        withheld = bool(q > crit[c] and r > crit[c])
        assert flag is (None if withheld else want), (c, q, r, crit[c], flag)
        seen.add("withheld" if withheld else str(want))
    print(span, seen)


@pytest.mark.parametrize("kind", ["monitor", "sequential"])
def test_a_bandwidth_of_the_whole_span_is_the_largest_accepted(kind):
    """The bandwidth is at most ``window_rows``: a span of ``m`` rows has no
    lag past ``m - 1``, so a larger one adds no lag and only flattens the
    weights toward 1, where the centred moments' lags sum to 0. ``m`` itself
    is accepted and gives a statistic; ``m + 1`` is refused by name, under
    both kinds that read it."""
    df = pair(400, 0.4, seed=13)
    out = run(df, kind=kind, window_rows=50, bandwidth=50)
    stats = out["stat"].drop_nulls()
    assert stats.len() > 0 and np.isfinite(stats.to_numpy()).all()
    with pytest.raises(
        ValueError, match=r"bandwidth must be at most window_rows \(50\), but it is 51"
    ):
        spec(kind=kind, window_rows=50, bandwidth=51)


def _sequential_break():
    return pl.concat([pair(400, 0.3, seed=4), pair(600, 0.8, seed=5)])


def test_a_sequential_crit_replaces_wied_and_galeanos_value():
    """``crit`` under ``"sequential"`` is reported on every live row as
    given, and a row flags just when its statistic is above it. Given Wied &
    Galeano's own value it is the default frame, to the bit."""
    df = _sequential_break()
    kw = {"kind": "sequential", "window_rows": 200}
    default = run(df, **kw)
    theirs = default["crit"].drop_nulls().unique()
    assert theirs.len() == 1
    assert run(df, **kw, crit=theirs[0]).equals(default)
    for crit in (0.5, 1.0, 50.0):
        live = run(df, **kw, crit=crit).filter(pl.col("stat").is_not_null())
        assert live["crit"].unique().to_list() == [crit]
        assert (live["flag"] == (live["stat"] > crit)).all(), crit


def test_a_larger_sequential_crit_flags_no_sooner():
    """A cycle ends at its first flag, so flag counts need not nest across
    values of ``crit``; what must hold is that the two runs are the same
    frame up to the smaller value's first flag, and the larger flags there
    or later, or never."""
    df = _sequential_break()
    kw = {"kind": "sequential", "window_rows": 200}
    firsts = []
    for crit in (0.5, 1.0, 2.0, 50.0):
        out = run(df, **kw, crit=crit)
        flag = out["flag"].fill_null(False)
        firsts.append((out, int(flag.arg_max()) if flag.any() else None))
    print("first flags at rising crit:", [f for _, f in firsts])
    assert firsts[0][1] is not None and firsts[-1][1] is None
    for (lo, at), (hi, later) in zip(firsts, firsts[1:], strict=False):
        if at is None:
            assert later is None
            continue
        assert lo["stat"][: at + 1].equals(hi["stat"][: at + 1])
        assert later is None or later >= at


def test_the_sequential_critical_value_rises_with_the_boundary_exponent():
    """At a fixed ``T`` the boundary is lower at the start of the period the
    larger ``boundary_gamma`` is, so its critical value is higher: Wied &
    Galeano's Table 1 reads 1.5578, 1.9924, 2.6844 at ``T = 1``."""
    crits = [_sequential_crit(100, 1.0, g) for g in (0.0, 0.1, 0.25, 0.45)]
    assert all(b > a for a, b in zip(crits, crits[1:], strict=False)), crits


@pytest.mark.extended(reason="the critical value's solve takes about 3 s at 0.49")
def test_the_largest_boundary_exponent_is_accepted_and_finite():
    at_45 = _sequential_crit(100, 1.0, 0.45)
    at_49 = _sequential_crit(100, 1.0, 0.49)
    assert np.isfinite(at_49) and at_49 > at_45, (at_45, at_49)


def test_monitor_rows_defaults_to_window_rows():
    df = pair(1000, 0.4, seed=6)
    kw = {"kind": "sequential", "window_rows": 100}
    assert run(df, **kw).equals(run(df, **kw, monitor_rows=100))
    assert not run(df, **kw).equals(run(df, **kw, monitor_rows=101))


def test_the_scalar_standardisers_half_life_moves_the_statistic():
    """``half_life`` is accepted under ``scalar`` only, and there it decays
    the standardiser and ``weight_sum``: 20, 2,000 and none give three
    statistics and three weights."""
    cols = [f"x{i}" for i in range(4)]
    df = pair(1000, 0.4, seed=7, k=4)
    outs = [
        run(df, features=cols, scalar=True, **({} if h is None else {"half_life": h}))
        for h in (None, 20.0, 2000.0)
    ]
    assert outs[0]["weight_sum"][-1] == 999.0
    for a, b in itertools.combinations(outs, 2):
        assert not a["stat"].equals(b["stat"])
        assert a["weight_sum"][-1] != b["weight_sum"][-1]


@pytest.mark.parametrize("level", [0.0, 1e8])
@pytest.mark.parametrize("norm", ["l1", "linf"])
def test_the_window_statistic_is_its_norm_on_every_row(norm, level):
    """The definition, by hand: ``norm(vech(R_pre - R_post))`` over the two
    windows ending at each row, the correlations from ``np.corrcoef``. No
    library computes the statistic, so this is the reference, on every row
    of a stream with a break, at four columns (six pairs); ``linf`` the
    largest pair, ``l1`` their sum. Windows of 40 rows over four columns are
    eight blocks of five, so the statistic is merged from the blocks' sums
    (task 328), and at a level of 1e8 with a spread of 1 each block keeps
    its mean against one of its rows: numpy's two-pass ``corrcoef`` agrees
    within 1e-12 there too."""
    w, k = 40, 4
    df = pl.concat([pair(100, 0.1, seed=8, k=k), pair(100, 0.8, seed=9, k=k)])
    df = df.with_columns(pl.all() + level)
    cols = [f"x{i}" for i in range(k)]
    out = run(df, features=cols, kind="window", window_rows=w, crit=9.0, norm=norm)
    x = df.to_numpy()
    iu = np.triu_indices(k, 1)
    reduce = np.max if norm == "linf" else np.sum
    for t in range(df.height):
        if t < 2 * w - 1:
            assert out["stat"][t] is None
            continue
        pre = np.corrcoef(x[t - 2 * w + 1 : t - w + 1].T)[iu]
        post = np.corrcoef(x[t - w + 1 : t + 1].T)[iu]
        assert out["stat"][t] == pytest.approx(reduce(np.abs(pre - post)), rel=1e-12), t


def test_reset_under_the_permutation_quantile():
    """``change_action="reset"`` without ``crit``: the quantile is drawn from
    the windows, so a reset also discards the value in force. Each flag
    empties both windows, so the next ``2 * window_rows - 1`` rows report
    nothing, the row after them reports a statistic with no critical value
    yet (drawn after it), and ``since_flag`` there counts the ``2 *
    window_rows`` rows since the flag. Between flags the statistic is a
    fresh bank's on the rows after the flag: the windows start over. The
    critical values are not: the generator draws on, so they are new draws,
    not a replay of the first ones."""
    w = 40
    df = broken()
    kw = {"kind": "window", "window_rows": w, "n_perm": 30, "permute_every_rows": 5}
    kept = run(df, **kw)
    reset = run(df, **kw, change_action="reset")
    flags = [i for i, f in enumerate(reset["flag"].to_list()) if f]
    print(f"flags: kept {kept['flag'].sum()}, reset {len(flags)} at {flags}")
    assert len(flags) >= 5 and kept["flag"].sum() > 3 * len(flags)
    redrawn = 0
    for f, nxt in zip(flags, [*flags[1:], df.height], strict=True):
        if f + 2 * w >= df.height:
            continue  # the stream ends in the refill
        assert reset["stat"][f + 1 : f + 2 * w].null_count() == 2 * w - 1, f
        assert reset["stat"][f + 2 * w] is not None and reset["crit"][f + 2 * w] is None, f
        assert reset["since_flag"][f + 2 * w] == 2 * w, f
        fresh = run(df[f + 1 : nxt + 1], **kw)
        assert fresh["stat"].equals(reset["stat"][f + 1 : nxt + 1]), f
        # `since_flag` agrees until either flags: the critical values are
        # other draws, so the two need not flag on the same row (measured
        # 2026-10-26: after the flag at 593 the fresh bank flags 5 rows
        # before the reset one does).
        hits = fresh["flag"].fill_null(False).arg_true()
        upto = int(hits[0]) if len(hits) else nxt - f
        assert fresh["since_flag"][:upto].equals(reset["since_flag"][f + 1 : f + 1 + upto]), f
        a, b = fresh["crit"].to_numpy(), reset["crit"][f + 1 : nxt + 1].to_numpy()
        live = np.isfinite(a)
        assert np.array_equal(live, np.isfinite(b)), f
        redrawn += int(np.any(a[live] != b[live]))
    assert redrawn > 0, "the generator drew on: not a fresh bank's draws"


def test_a_state_saved_across_a_reset_goes_on_to_the_bit():
    """Cut on the row after a reset, in the refill, and on the first row
    the refilled windows report; each loads and goes on as the unbroken
    stream does (the chunk legs above cover 1, 7, 13, 300 and 600 rows)."""
    w = 40
    df = broken()
    s = spec(**CHUNK_KW["window, reset"])
    want = run(df, **CHUNK_KW["window, reset"])
    first = want["flag"].to_list().index(True)
    for cut in (first + 1, first + w, first + 2 * w, first + 2 * w + 1):
        bank = rp.ModelBank([s])
        bank.fit_predict(df[:cut])
        again = rp.ModelBank.load_bytes(bank.save_bytes(), [s])
        assert want[cut:].equals(again.fit_predict(df[cut:])["c"].struct.unnest()), cut


@pytest.mark.parametrize(
    ("kw", "live"),
    [
        # A span of 8 reports on its last row, every 8 rows.
        ({"window_rows": 8}, [7, 15, 23]),
        # A history of 8 (rows 0-7), the first three monitored rows silent,
        # the fourth reports and ends the period; then the next history.
        ({"kind": "sequential", "window_rows": 8, "monitor_rows": 4}, [11, 23]),
        # Two windows of 3: the first statistic once six rows are in.
        ({"kind": "window", "window_rows": 3}, list(range(5, 24))),
    ],
)
def test_the_smallest_window_rows_report_where_they_close(kw, live):
    out = run(pair(24, 0.3, seed=1), **kw)
    assert [i for i, v in enumerate(out["stat"].to_list()) if v is not None] == live
    if kw.get("kind") == "window":
        # The quantile is drawn after the first report, so it is in force
        # from the next row.
        assert out["crit"][5] is None and out["crit"][6] is not None


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"window_rows": 7}, 'kind="monitor" needs window_rows of at least 8, but it is 7'),
        (
            {"kind": "sequential", "window_rows": 7},
            'kind="sequential" needs window_rows of at least 8, but it is 7',
        ),
        (
            {"kind": "sequential", "window_rows": 8, "monitor_rows": 3},
            "monitor_rows of at least 4",
        ),
    ],
)
def test_one_below_the_smallest_window_rows_is_refused(kw, message):
    with pytest.raises(ValueError, match=message):
        spec(**kw)


# --- task 553: the oracles of the new statistics ------------------------------


@pytest.mark.parametrize("gamma", [0.0, 0.25, 0.45])
def test_the_sequential_detector_is_fisher_z_by_numpy_and_scipy(gamma):
    """The ``"sequential"`` detector from third-party pieces: numpy's
    ``arctanh`` of scipy's ``pearsonr`` correlations of the history and of
    the monitored rows so far, each weighted by the information scipy's
    ``pearsonr(...).confidence_interval()`` puts on its Fisher ``z``
    (``1/se² = n - 3``, read back from the interval's width), and the
    history's long-run deviation from statsmodels' ``S_hac_simple``:

    ``V_k = (I_k/sqrt(I_h))·|z_k − z_h|·(1 − r_h²)/σ̂``, against ``w(I_k/I_h)``.

    On a stream whose correlation moves from 0.4 to 0.7 inside the period,
    with a ``crit`` no row reaches, so the whole period reports; the first
    three monitored rows report nothing, having no information."""
    from scipy.stats import norm, pearsonr
    from statsmodels.stats.sandwich_covariance import S_hac_simple

    m, k_rows = 120, 150
    df = pl.concat([pair(m + 60, 0.4, seed=71), pair(90, 0.7, seed=72)])
    X = df.to_numpy()
    out = run(
        df,
        kind="sequential",
        window_rows=m,
        monitor_rows=k_rows,
        boundary_gamma=gamma,
        crit=1e9,
        alpha_adjust="none",
    )
    stat = out["stat"].to_numpy().astype(float)
    z975 = norm.ppf(0.975)

    def info(a, b):
        ci = pearsonr(a, b).confidence_interval(0.95)
        return ((2 * z975) / (np.arctanh(ci.high) - np.arctanh(ci.low))) ** 2

    h = X[:m]
    r_h = pearsonr(h[:, 0], h[:, 1]).statistic
    xi, _ = _span_series(h)
    sd = math.sqrt(S_hac_simple(xi, nlags=math.floor(math.log(m)) - 1)[0, 0] / m)
    i_h = info(h[:, 0], h[:, 1])
    assert i_h == pytest.approx(m - 3, rel=1e-9)
    want = np.full(k_rows, np.nan)
    for k in range(4, k_rows + 1):
        mon = X[m : m + k]
        r_k = pearsonr(mon[:, 0], mon[:, 1]).statistic
        i_k = info(mon[:, 0], mon[:, 1])
        b = i_k / i_h
        w = (1 + b) * (b / (1 + b)) ** gamma
        v = i_k / math.sqrt(i_h) * abs(np.arctanh(r_k) - np.arctanh(r_h)) * (1 - r_h**2) / sd
        want[k - 1] = v / w
    got = stat[m : m + k_rows]
    assert np.isnan(got[:3]).all(), got[:4]
    np.testing.assert_allclose(got[3:], want[3:], rtol=1e-9)


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.31, 0.35, 0.4, 0.55, 0.65, 0.69, 0.9])
def test_the_block_permutation_null_is_scipys_exact_one(alpha):
    """``perm_block`` against scipy's ``permutation_test``, which enumerates
    the null exactly: two windows of 6 rows in blocks of 3 are four blocks,
    and their six splits into two windows are the whole null, each block one
    observation that scipy permutes by its number (the statistic does not
    read the order inside a window). The model's critical value at many
    draws is the ``1 - alpha`` quantile of that null, one of its atoms, at
    levels on both sides of the atoms' edges at 1/3 and 2/3, so the draws
    are uniform over the block splits and not merely on the right values.
    Matched to 3e-16 in the review that found the oracle."""
    from scipy import stats

    rng = np.random.default_rng(7)
    w, b = 6, 3
    n = 2 * w
    a, e = rng.standard_normal((2, n + 1))
    x = np.column_stack([a, 0.5 * a + e])
    df = pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1]})
    blocks = x[:n].reshape(n // b, b * 2)

    def stat(pre, post):
        def r(ids):
            return np.corrcoef(blocks[ids.astype(int)].reshape(-1, 2).T)[0, 1]

        return abs(r(pre) - r(post))

    ids = np.arange(n // b, dtype=float)
    res = stats.permutation_test(
        (ids[: w // b], ids[w // b :]),
        stat,
        permutation_type="independent",
        n_resamples=np.inf,
        vectorized=False,
        alternative="greater",
    )
    null = np.sort(res.null_distribution)
    assert len(null) == 6
    cdf = np.arange(1, len(null) + 1) / len(null)
    want = null[np.searchsorted(cdf, 1 - alpha - 1e-12)]
    out = run(
        df,
        kind="window",
        window_rows=w,
        perm_block=b,
        n_perm=20_000,
        alpha=alpha,
        permute_every_rows=1,
        seed=3,
    )
    # Drawn after the first report (row n - 1), in force at row n.
    assert out["crit"][n] == pytest.approx(want, abs=1e-14)


def test_a_moved_scale_ratio_withholds_the_flag_by_name():
    """The one condition the linear CUSUM does not remove: with the
    correlation constant and one column's scale doubled half way through a
    span, ``ξ``'s mean moves by ``−(ρ/2)(r_x − r_y)²`` and its CUSUM leaves
    its null law, so at ``rho = 0.9`` it passes the critical value on most
    such spans (0.93 of spans of 1,000 rows, measured). The same CUSUM of
    ``x̃² − ỹ²`` sees the ratio move, and the flag is withheld: null, named
    ``scale_ratio_moved`` by ``withheld_reason``. A correlation that breaks
    with both scales steady flags, with no reason; a span where both scales
    double together (a change of clock for the sums) is stable and flags
    neither."""
    rng = np.random.default_rng(91)
    T = 1000

    def span(rho, sx=1.0, sy=1.0):
        a, e = rng.standard_normal((2, T))
        rho = np.asarray(rho)
        x, y = a, rho * a + np.sqrt(1 - rho**2) * e
        return pl.DataFrame({"x0": x * sx, "x1": y * sy})

    step = np.where(np.arange(T) < T // 2, 1.0, 2.0)
    df = pl.concat(
        [
            span(0.9, sx=step),
            span(np.where(np.arange(T) < T // 2, 0.2, 0.8)),
            span(0.6, sx=step, sy=step),
        ]
    )
    bank = rp.ModelBank([spec(window_rows=T, alpha_adjust="none")])
    out = bank.fit_predict(df)
    c = out["c"].struct.unnest()
    reason = c["withheld_reason"].cast(pl.String)
    ends = [T - 1, 2 * T - 1, 3 * T - 1]
    assert c["stat"][ends[0]] > c["crit"][ends[0]]
    assert c["flag"][ends[0]] is None and reason[ends[0]] == "scale_ratio_moved"
    assert c["flag"][ends[1]] is True and reason[ends[1]] is None
    assert c["flag"][ends[2]] is False and reason[ends[2]] is None
    assert reason.drop_nulls().len() == 1


# --- task 553: spans on the clock ----------------------------------------------


@pytest.mark.parametrize(
    ("clock_kw", "rows_kw"),
    [
        ({"window_size": 39}, {"window_rows": 40}),
        (
            {"kind": "sequential", "window_size": 49, "monitor_span": 79},
            {"kind": "sequential", "window_rows": 50, "monitor_rows": 80},
        ),
    ],
)
def test_a_clock_span_on_rows_is_its_count_plus_one(clock_kw, rows_kw):
    """Without a clock column each row is one unit of clock, and a span of
    ``D`` units ends at the row ``D`` after its first: ``window_size = D`` is
    ``window_rows = D + 1`` (and ``monitor_span = M`` is ``monitor_rows = M +
    1``), to the bit, on a stream with breaks and a row of weight 0 in every
    seven, which counts in no clock and is in no span."""
    df = broken().with_columns(w=pl.when(pl.int_range(pl.len()) % 7 == 3).then(0.0).otherwise(1.0))
    a = run(df, weight="w", **clock_kw)
    b = run(df, weight="w", **rows_kw)
    assert a.equals(b)
    assert a["flag"].fill_null(False).sum() >= 1


def _stamped(n, seed, rho=None):
    """``n`` rows of a pair at irregular times, 0.2 s to 3 s apart, and the
    times as a datetime column ``t``."""
    rng = np.random.default_rng(seed)
    gaps = rng.uniform(0.2, 3.0, n)
    t = np.datetime64("2026-01-01T00:00:00", "ns") + (np.cumsum(gaps) * 1e9).astype(
        "timedelta64[ns]"
    )
    rho = np.full(n, 0.3) if rho is None else rho
    a, e = rng.standard_normal((2, n))
    return pl.DataFrame({"t": t, "x0": a, "x1": rho * a + np.sqrt(1 - rho**2) * e})


def test_a_monitor_span_of_the_clock_is_its_rows_by_polars_datetimes():
    """``window_size = "60s"`` on a temporal clock, against the definition by
    Polars' datetime arithmetic: a span begins at its first row and ends at
    the first row at least 60 s after it, which is its last; the report
    there is the ``window_rows`` statistic of exactly those rows, at the
    critical value of their count."""
    df = _stamped(700, 5)
    out = run(df, clock="t", gap_cap="10s", window_size="60s", alpha_adjust="none")
    t = df["t"].to_numpy()
    ends, first = [], 0
    for i in range(df.height):
        if t[i] - t[first] >= np.timedelta64(60, "s"):
            ends.append(i)
            first = i + 1
    live = np.flatnonzero(out["stat"].is_not_null().to_numpy()).tolist()
    assert live == ends
    start = 0
    for e in ends:
        rows = df[start : e + 1]
        alone = run(rows.drop("t"), window_rows=rows.height, alpha_adjust="none")
        assert out["stat"][e] == alone["stat"][rows.height - 1]
        assert out["crit"][e] == alone["crit"][rows.height - 1]
        start = e + 1


def test_a_clock_window_reads_its_two_windows_by_polars_datetimes():
    """``kind="window"`` with ``window_size = "30s"`` on a temporal clock:
    each report's two windows are the rows of ``(now − 60 s, now − 30 s]``
    and ``(now − 30 s, now]``, cut by Polars' datetime comparisons, and its
    statistic is ``|corr(pre) − corr(post)|`` by numpy; reports begin once
    the stream has run 60 s. Between two draws the held critical value is
    rescaled to each report's counts: the ratio of two reports' values is
    ``sqrt((1/n₁ + 1/n₂)/(1/n₁' + 1/n₂'))``."""
    df = _stamped(900, 6, rho=np.where(np.arange(900) < 450, 0.1, 0.8))
    out = run(
        df,
        clock="t",
        gap_cap="10s",
        kind="window",
        window_size="30s",
        n_perm=50,
        permute_every="200s",
    )
    t = df["t"].to_numpy()
    X = df.select("x0", "x1").to_numpy()
    d = np.timedelta64(30, "s")
    stat = out["stat"].to_numpy().astype(float)
    crit = out["crit"].to_numpy().astype(float)
    spread = np.full(df.height, np.nan)
    for i in range(df.height):
        now = t[i]
        if now - t[0] < 2 * d:
            assert np.isnan(stat[i]), i
            continue
        pre = (t > now - 2 * d) & (t <= now - d) & (np.arange(df.height) <= i)
        post = (t > now - d) & (np.arange(df.height) <= i)
        n1, n2 = int(pre.sum()), int(post.sum())
        spread[i] = 1 / n1 + 1 / n2
        if n1 < 2 or n2 < 2:
            continue
        want = abs(np.corrcoef(X[pre].T)[0, 1] - np.corrcoef(X[post].T)[0, 1])
        assert stat[i] == pytest.approx(want, rel=1e-12, abs=1e-15), i
    # The same draw on consecutive rows: its value moves with the counts.
    same = 0
    for i in range(1, df.height):
        if np.isfinite(crit[i]) and np.isfinite(crit[i - 1]) and np.isfinite(spread[i - 1]):
            ratio = crit[i] / crit[i - 1]
            want = math.sqrt(spread[i] / spread[i - 1])
            if abs(ratio - want) < 1e-12:
                same += 1
    assert same > 600, same


def test_a_sequential_cycle_on_the_clock_plans_its_period():
    """``kind="sequential"`` with ``window_size = "40s"`` and ``monitor_span
    = "60s"``: each cycle's history is the rows of the 40 s from its first,
    ``m`` of them by Polars' count, its period plans ``K = ⌊1.5 m⌋`` rows, and
    its critical value is Wied & Galeano's at ``T = (K − 3)/(m − 3)``: at
    ``γ = 0``, ``sqrt(T/(1+T))`` times 2.2414, a value per cycle."""
    df = _stamped(1200, 7)
    out = run(
        df,
        clock="t",
        gap_cap="10s",
        kind="sequential",
        window_size="40s",
        monitor_span="60s",
        alpha_adjust="none",
        crit=None,
    )
    crits = out["crit"].drop_nulls().unique()
    assert crits.len() > 2, crits
    q = _sup_abs_bm_quantile(0.95)
    allowed = []
    for m in range(4, 200):
        k = math.floor(m * 1.5)
        ratio = (k - 3) / (m - 3)
        allowed.append((ratio / (1 + ratio)) ** 0.5 * q)
    for c in crits:
        assert min(abs(c - a) for a in allowed) < 1e-6, c


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        (
            {"window_rows": 50, "window_size": 20},
            "window_rows and window_size are both set",
        ),
        (
            {"kind": "sequential", "window_rows": 50, "monitor_span": 20},
            "monitor_span needs window_size",
        ),
        (
            {"window_size": 20, "monitor_span": 20},
            'monitor_span applies only under kind="sequential"',
        ),
        (
            {"window_rows": 50, "max_state_bytes": 10_000},
            "max_state_bytes applies only under window_size",
        ),
        ({"window_size": 0}, "window_size must be a span of the clock above 0"),
        ({"window_size": 20, "max_state_bytes": 0}, "max_state_bytes must be above 0"),
    ],
)
def test_the_clock_forms_are_refused_by_name(kw, message):
    d = {"features": ["x0", "x1"]}
    d.update(kw)
    with pytest.raises(ValueError, match=re.escape(message)):
        rp.ModelBank([rp.spec.corrchange("c", **d)])


def test_a_clock_span_past_its_bytes_is_refused_by_the_bank():
    """The rows a span of the clock holds are the data's to number, so they
    are checked as they come: a ``window_size`` of a day over rows a second
    apart passes a ``max_state_bytes`` of 20 KB, and the bank refuses the run
    by the spec, naming ``window_size``."""
    df = _stamped(2000, 8)
    with pytest.raises(ValueError, match=r"the rows inside window_size = 1d would take its state"):
        run(df, clock="t", gap_cap="10s", window_size="1d", max_state_bytes=20_000)


# --- task 553: size and power against the review's measured tables -------------
#
# The review of 2026-11-22 measured each candidate statistic on a numpy
# replica written from the papers (10,000 replications a cell); these are its
# figures for the two that were built, the linear CUSUM against the grid-
# corrected value and the Fisher-z detector. The built model is held to them
# here: a size at most the review's plus three standard errors of this
# grid's replications, a power at least the review's less three, where the
# monitor's figures without the scale-ratio check are first taken down by
# the 5% of spans that check withholds at a correlation break (0.033 to 0.046
# measured, every T). The streams are the review's: Gaussian, a shared-scale
# multivariate t_5, a shared GARCH(1,1) volatility of persistence 0.98, W&G's
# own GARCH design, and the scales that double at the span's or the
# period's middle.


def _garch(rng, reps, n, omega, a, b, burn=500):
    z = rng.standard_normal((reps, n + burn))
    h = np.full(reps, omega / (1 - a - b))
    out = np.empty((reps, n + burn))
    for t in range(n + burn):
        out[:, t] = np.sqrt(h) * z[:, t]
        h = omega + a * out[:, t] ** 2 + b * h
    return out[:, burn:] / np.sqrt(omega / (1 - a - b))


def _streams(rng, reps, n, rho, kind):
    """The review's designs: ``x, y`` of shape ``(reps, n)``."""
    rho = np.asarray(rho, float)
    mix = lambda e1, e2, r: (e1, r * e1 + np.sqrt(1 - r**2) * e2)  # noqa: E731
    if kind == "gauss":
        return mix(*rng.standard_normal((2, reps, n)), rho)
    if kind == "t5":
        x, y = mix(*rng.standard_normal((2, reps, n)), rho)
        s = np.sqrt(rng.chisquare(5, (reps, n)) / 5)
        return x / s, y / s
    if kind == "garch_wg":
        e1 = _garch(rng, reps, n, 0.01, 0.05, 0.8)
        e2 = _garch(rng, reps, n, 0.01, 0.1, 0.75)
        a = 0.5 * (np.sqrt(1 + rho) + np.sqrt(1 - rho))
        b = 0.5 * (np.sqrt(1 + rho) - np.sqrt(1 - rho))
        return a * e1 + b * e2, b * e1 + a * e2
    if kind == "garch_hi":
        e1, e2 = rng.standard_normal((2, reps, n + 500))
        h = np.full(reps, 0.02 / (1 - 0.98))
        sd = np.empty((reps, n + 500))
        for t in range(n + 500):
            sd[:, t] = np.sqrt(h)
            h = 0.02 + 0.08 * (sd[:, t] * e1[:, t]) ** 2 + 0.90 * h
        x, y = mix(e1[:, 500:], e2[:, 500:], rho)
        return x * sd[:, 500:], y * sd[:, 500:]
    if kind in ("both_double", "one_doubles"):
        x, y = mix(*rng.standard_normal((2, reps, n)), rho)
        s = np.where(np.arange(n) < n // 2, 1.0, 2.0)
        return (x * s, y * s) if kind == "both_double" else (x * s, y)
    raise ValueError(kind)


def _grouped(x, y, **kw):
    """Each replication a group of one bank, run in one pass."""
    reps, n = x.shape
    df = pl.DataFrame(
        {
            "g": np.repeat(np.arange(reps), n),
            "i": np.tile(np.arange(n), reps),
            "x0": x.ravel(),
            "x1": y.ravel(),
        }
    )
    out = rp.ModelBank([spec(group="g", alpha_adjust="none", **kw)]).fit_predict(df)
    return pl.concat([df.select("g", "i"), out["c"].struct.unnest()], how="horizontal")


#: Monitor, the share of stable spans flagged: the review's linear CUSUM
#: against the grid-corrected value, at T = 8 / 20 / 100 / 200 / 1000; the
#: one-scale row is its guarded figure (correlation flags the ratio does not).
MONITOR_SIZE = {
    ("gauss", 0.0): (0.008, 0.017, 0.034, 0.037, 0.049),
    ("gauss", 0.9): (0.007, 0.017, 0.030, 0.036, 0.044),
    ("t5", 0.9): (0.007, 0.009, 0.024, 0.029, 0.036),
    ("garch_hi", 0.6): (0.009, 0.015, 0.034, 0.035, 0.044),
    ("both_double", 0.0): (0.010, 0.016, 0.028, 0.036, 0.048),
    ("one_doubles", 0.6): (0.009, 0.019, 0.044, 0.001, 0.000),
}

#: Monitor power, Gaussian: (from, to, where) -> {T: the review's rate}.
MONITOR_POWER = {
    (0.5, 0.7, 0.5): {100: 0.198, 200: 0.430, 500: 0.869},
    (0.9, 0.7, 0.5): {100: 0.564, 200: 0.922},
    (0.5, 0.7, 0.1): {500: 0.178},
    (0.3, 0.6, 0.9): {500: 0.142},
}

#: Sequential, the share of cycles ending in a flag, nothing changed in the
#: correlation (monitor_rows = window_rows): the review's Fisher-z figures at
#: m = 50 / 100 / 250 / 1000.
SEQUENTIAL_SIZE = {
    ("gauss", 0.0, 0.0): (0.077, 0.069, 0.055, 0.050),
    ("gauss", 0.6, 0.45): (0.071, 0.060, 0.057, 0.049),
    ("gauss", 0.9, 0.0): (0.072, 0.064, 0.058, 0.056),
    ("gauss", 0.9, 0.45): (0.068, 0.060, 0.055, 0.052),
    ("t5", 0.0, 0.0): (0.121, 0.102, 0.080, 0.071),
    ("t5", 0.9, 0.45): (0.096, 0.086, 0.067, 0.061),
    ("garch_wg", 0.5, 0.45): (0.071, 0.072, 0.062, 0.055),
}

#: Sequential power at m = 250: (from, to, monitored row of the break, gamma)
#: -> (flagged before it, flagged after it, median delay in rows).
SEQUENTIAL_POWER = {
    (0.5, 0.7, 12, 0.0): (0.000, 0.885, 106),
    (0.5, 0.7, 12, 0.45): (0.013, 0.761, 90),
    (0.9, 0.7, 62, 0.0): (0.001, 0.998, 46),
    (0.9, 0.7, 62, 0.45): (0.035, 0.962, 44),
    (0.5, -0.5, 12, 0.0): (0.000, 1.000, 29),
    (0.0, 0.5, 62, 0.0): (0.001, 0.984, 73),
}


def _band(p, reps):
    return 3 * math.sqrt(max(p, 0.002) * (1 - p) / reps)


@pytest.mark.extended(reason="the review's size and power grid: 4,000 replications a cell")
@pytest.mark.parametrize(("dgp", "rho"), list(MONITOR_SIZE))
def test_the_monitor_size_is_at_most_the_reviews(dgp, rho):
    reps = 4000
    for T, want in zip((8, 20, 100, 200, 1000), MONITOR_SIZE[(dgp, rho)], strict=True):
        rng = np.random.default_rng(553_000 + T)
        x, y = _streams(rng, reps, T, rho, dgp)
        out = _grouped(x, y, window_rows=T).filter(pl.col("i") == T - 1)
        size = float((out["flag"] == True).sum()) / reps  # noqa: E712
        assert size <= want + _band(want, reps), (dgp, rho, T, size, want)


@pytest.mark.extended(reason="the review's size and power grid: 4,000 replications a cell")
@pytest.mark.parametrize("brk", list(MONITOR_POWER))
def test_the_monitor_power_is_at_least_the_reviews(brk):
    r0, r1, at = brk
    reps = 4000
    for T, want in MONITOR_POWER[brk].items():
        rng = np.random.default_rng(553_100 + T)
        rho = np.where(np.arange(T) < int(at * T), r0, r1)
        x, y = _streams(rng, reps, T, rho, "gauss")
        out = _grouped(x, y, window_rows=T).filter(pl.col("i") == T - 1)
        power = float((out["flag"] == True).sum()) / reps  # noqa: E712
        guarded = 0.95 * want
        assert power >= guarded - _band(guarded, reps), (brk, T, power, want)


def _first_flags(x, y, m, gamma):
    out = _grouped(x, y, kind="sequential", window_rows=m, monitor_rows=m, boundary_gamma=gamma)
    f = out.filter(pl.col("flag").fill_null(False)).group_by("g").agg(pl.col("i").min())
    first = np.full(x.shape[0], -1)
    first[f["g"].to_numpy()] = f["i"].to_numpy() - m + 1
    return first


@pytest.mark.extended(reason="the review's size and power grid: 4,000 replications a cell")
@pytest.mark.parametrize(("dgp", "rho", "gamma"), list(SEQUENTIAL_SIZE))
def test_the_sequential_size_is_at_most_the_reviews(dgp, rho, gamma):
    for m, want in zip((50, 100, 250, 1000), SEQUENTIAL_SIZE[(dgp, rho, gamma)], strict=True):
        reps = 4000 if m < 1000 else 2000
        rng = np.random.default_rng(553_200 + m)
        x, y = _streams(rng, reps, 2 * m, rho, dgp)
        size = float((_first_flags(x, y, m, gamma) > 0).mean())
        assert size <= want + _band(want, reps), (dgp, rho, gamma, m, size, want)


@pytest.mark.extended(reason="the review's size and power grid: 4,000 replications a cell")
@pytest.mark.parametrize("brk", list(SEQUENTIAL_POWER))
def test_the_sequential_power_is_at_least_the_reviews(brk):
    r0, r1, k, gamma = brk
    before, after, delay = SEQUENTIAL_POWER[brk]
    m, reps = 250, 4000
    rng = np.random.default_rng(553_300 + k)
    rho = np.full(2 * m, r0)
    rho[m + k :] = r1
    x, y = _streams(rng, reps, 2 * m, rho, "gauss")
    first = _first_flags(x, y, m, gamma)
    early = float(((first > 0) & (first <= k)).mean())
    late = first[first > k]
    assert early <= before + _band(before, reps), (brk, early)
    assert late.size / reps >= after - _band(after, reps), (brk, late.size / reps)
    # The delay is a median of a few thousand: within a tenth of the review's.
    assert np.median(late - k) <= 1.1 * delay, (brk, np.median(late - k))
