"""`expectile` through the bank (docs/PLAN.md task 501): its arithmetic
against a numpy longhand of the one pass and of the exact refit, rows of
weight 0 in no clock, the windows on rows and on the clock, its state's
bytes against ``max_state_bytes``, and its refusals by name.
`tests/test_second_opinion_expectile.py` holds it to statsmodels and scipy.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import rheon
from rheon import polars as rp

TIER = "essential"
INF = float("inf")


def stream(n: int = 600, seed: int = 0, zeros: bool = False) -> pl.DataFrame:
    """Two features, a target on a line with skewed noise, a clock in
    quarter units (sums of steps are exact), and optionally rows of weight
    0 among weights of 0.5 to 1.5, the first rows among them."""
    rng = np.random.default_rng([501, seed])
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    y = 1.0 + 2.0 * x0 - x1 + rng.exponential(1.0, n)
    w = rng.uniform(0.5, 1.5, n)
    if zeros:
        w[[0, 1, 2]] = 0.0
        w[rng.random(n) < 0.15] = 0.0
    t = np.cumsum(rng.integers(1, 6, n) * 0.25)
    return pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y, "w": w})


def _spec(**kw: object) -> dict:
    kw = {
        "targets": ["y"],
        "features": ["x0", "x1"],
        "expectile": 0.8,
        "ridge": 0.0,
        "half_life": 50.0,
        "min_weight": 0.0,
        **kw,
    }
    return rp.spec.expectile("m", **kw)


def fields(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).select("m").unnest("m")


def coef_after(spec: dict, df: pl.DataFrame) -> np.ndarray:
    """The coefficients once ``df`` is learned, the intercept first."""
    bank = rp.ModelBank([spec])
    bank.fit(df)
    return bank.coef()["coef"].to_numpy()


def side(r: np.ndarray, tau: float) -> np.ndarray:
    return np.where(r > 0, tau, 1 - tau)


def test_the_one_pass_is_its_longhand():
    """The recursion of the docstring, written in numpy, row by row: each
    row's side judged at the fit before it (1/2 before the first fit), the
    mean-form sums at weight ``w a``, and ``beta = G^-1 c``."""
    df = stream(400)
    tau, h = 0.8, 50.0
    out = fields(_spec(weight="w", refit_rows=0, clock="t", gap_cap=1e9), df)
    X = np.column_stack([np.ones(df.height), df["x0"], df["x1"]])
    y, w, t = df["y"].to_numpy(), df["w"].to_numpy(), df["t"].to_numpy()
    S, G, c, beta = 0.0, np.zeros((3, 3)), np.zeros(3), None
    pred = []
    for i in range(df.height):
        z = X[i]
        p = np.nan if beta is None else float(z @ beta)
        pred.append(p)
        lam = 1.0 if i == 0 else 0.5 ** ((t[i] - t[i - 1]) / h)
        a = 0.5 if beta is None else float(side(np.array(y[i] - p), tau))
        S = lam * S + w[i] * a
        f = w[i] * a / S
        G = G + f * (np.outer(z, z) - G)
        c = c + f * (z * y[i] - c)
        if i >= 2:  # a row per coefficient, every weight here above 0
            beta = np.linalg.solve(G, c)
    got = out["pred_y"].fill_null(np.nan).to_numpy()
    np.testing.assert_allclose(got[3:], np.array(pred)[3:], rtol=1e-10, atol=1e-10)
    assert np.isnan(got[:3]).all()


def test_the_refit_of_the_first_rows_is_the_exact_fit():
    """On the row the buffer fills the fit is the exact expectile regression
    of those rows: the weighted least-squares fit whose sides agree with its
    own residuals, found here by iterating numpy's solve to a fixed point."""
    df = stream(300, 1)
    tau, rows = 0.9, 150
    got = coef_after(_spec(expectile=tau, half_life=INF, refit_rows=rows), df.head(rows))
    X = np.column_stack([np.ones(rows), df["x0"][:rows], df["x1"][:rows]])
    y = df["y"].to_numpy()[:rows]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(100):
        a = side(y - X @ b, tau)
        nb = np.linalg.solve((X * a[:, None]).T @ X, (X * a[:, None]).T @ y)
        done = np.array_equal(y - X @ nb > 0, y - X @ b > 0)
        b = nb
        if done:
            break
    np.testing.assert_allclose(got, b, rtol=1e-11)


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(refit_rows=60),
        dict(half_life=INF, window_rows=40),
        dict(half_life=INF, window_size=20.0, closed="both"),
    ],
    ids=["decay", "refit", "window_rows", "window_size"],
)
@pytest.mark.parametrize("clocked", [True, False], ids=["clock", "rows"])
def test_rows_of_weight_zero_count_in_no_clock(kw, clocked):
    """The weighted rows' outputs are those of the stream without the rows
    of weight 0, to the bit, on a clock and on the row clock; and a row of
    weight 0 is predicted as any row is."""
    df = stream(500, 2, zeros=True)
    extra = dict(clock="t", gap_cap=1e9) if clocked else {}
    s = _spec(weight="w", **{**kw, **extra})
    whole = fields(s, df)
    kept = fields(s, df.filter(pl.col("w") > 0))
    weighted = whole.filter(df["w"] > 0)
    for f in ("pred_y", "resid_y", "weight_sum"):
        assert weighted[f].equals(kept[f], null_equal=True), f
    assert kept["pred_y"].drop_nulls().len() > 300
    zero = whole.filter((df["w"] == 0) & (pl.int_range(df.height) > 100))
    assert zero["pred_y"].null_count() == 0


@pytest.mark.parametrize(
    "kw",
    [
        dict(refit_rows=80),
        dict(half_life=INF, window_rows=30),
        dict(half_life=INF, window_size=15.0),
    ],
    ids=["refit", "window_rows", "window_size"],
)
def test_chunks_and_a_save_change_nothing(kw, tmp_path):
    """One chunk, chunks of 37 rows, and a save and a load midway give the
    same output, to the bit; ``coef``, which a chunk's last row reports
    whatever its cadence, aside."""
    df = stream(400, 3)
    s = _spec(clock="t", gap_cap=1e9, **kw)
    one = fields(s, df)
    keep = [c for c in one.columns if c != "coef"]
    bank = rp.ModelBank([s])
    many = pl.concat([bank.fit_predict(df.slice(i, 37)) for i in range(0, 400, 37)])
    assert many.select("m").unnest("m").select(keep).equals(one.select(keep), null_equal=True)
    bank = rp.ModelBank([s])
    first = bank.fit_predict(df.slice(0, 200))
    path = tmp_path / "state.bin"
    bank.save(path)
    rest = rp.ModelBank.load(path).fit_predict(df.slice(200))
    split = pl.concat([first, rest]).select("m").unnest("m")
    assert split.select(keep).equals(one.select(keep), null_equal=True)


def test_a_temporal_window_takes_a_duration_as_a_numeric_one_takes_seconds():
    """``window_size="30s"`` on a temporal clock is ``30.0`` on a clock of the
    same seconds: the same rows in the window, so the same bits, steps of a
    quarter second being the same double either way."""
    df = stream(400, 4)
    ns = (df["t"] * 1e9).cast(pl.Int64) + 1_700_000_000_000_000_000
    temporal = df.with_columns(t=ns.cast(pl.Datetime("ns")))
    a = fields(_spec(half_life=INF, window_size="30s", clock="t", gap_cap="1h"), temporal)
    b = fields(_spec(half_life=INF, window_size=30.0, clock="t", gap_cap=3600.0), df)
    assert a["pred_y"].equals(b["pred_y"], null_equal=True)
    assert a["weight_sum"].equals(b["weight_sum"], null_equal=True)


def test_a_burst_in_a_clock_window_is_warned_of_once_and_refused_at_the_cap():
    """A window of clock units holds the rows its span has, which the data
    decides, so the cap is checked as the rows go in: the first row to take a
    group past half of ``max_state_bytes`` warns, once, naming the group and
    the bytes; the row that would take it past the cap stops the run with an
    error naming the group and the bytes."""
    # Two features, one target: 8 * 4 + 1 bytes a row, and a 32-byte stamp.
    fixed = 8 * (1 * (3 + 3) + 6) + 2 * 32
    cap = fixed + 65 * 150
    s = _spec(
        group="g",
        half_life=INF,
        window_size=1_000.0,
        max_state_bytes=cap,
        clock="t",
        gap_cap=1e9,
    )

    def burst(group: str, n: int, first: int = 0) -> pl.DataFrame:
        i = np.arange(first, first + n)
        rng = np.random.default_rng(first)
        return pl.DataFrame(
            {
                "g": group,
                "t": i * 0.01,
                "x0": rng.standard_normal(n),
                "x1": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )

    bank = rp.ModelBank([s])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(pl.concat([burst("a", 100), burst("b", 30)]))
    assert len(seen) == 1
    assert f"past half of max_state_bytes ({cap})" in str(seen[0].message)
    assert "the rows inside window_size" in str(seen[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bank.fit_predict(burst("a", 30, 100))
    with pytest.raises(Exception, match=r'group "a".*to \d+ bytes, above max_state_bytes of') as e:
        bank.fit_predict(burst("a", 100, 130))
    assert str(cap) in str(e.value)


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        (dict(expectile=1.0), "must be above 0 and below 1"),
        (dict(expectile=0.0), "must be above 0 and below 1"),
        (dict(window_rows=2, half_life=INF), "at least one row per coefficient"),
        (dict(window_rows=40), "takes no decay"),
        (dict(window_rows=40, half_life=INF, refit_rows=10), "nothing to add"),
        (dict(window_rows=40, window_size=3.0, half_life=INF), "two windows"),
        (dict(closed="both"), "there is no clock window"),
        (dict(max_state_bytes=0), "must be above 0"),
        (dict(max_state_bytes=(64 << 20) + 1), "at most 67108864 (64 MiB)"),
        (dict(window_rows=100_000, half_life=INF), "past max_state_bytes of 262144"),
        (dict(refit_rows=100_000), "past max_state_bytes of 262144"),
        (dict(refit_budget=0.0), "refit_budget"),
    ],
)
def test_each_setting_is_refused_by_name(kw, words):
    with pytest.raises(ValueError, match=words.replace("(", r"\(").replace(")", r"\)")):
        rp.ModelBank([_spec(**kw)])


def test_a_window_reports_the_weight_inside_it():
    """Under a window, ``weight_sum`` is the weight of the rows in it before
    the row (hard rule 8): the last 25 rows' weights."""
    df = stream(200, 5)
    out = fields(_spec(half_life=INF, window_rows=25, weight="w"), df)
    w = df["w"].to_numpy()
    want = [w[max(0, i - 25) : i].sum() for i in range(df.height)]
    np.testing.assert_allclose(out["weight_sum"].to_numpy(), want, rtol=1e-12)


def _exact_linear_stream(n: int = 4000, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pl.DataFrame({"x": x, "y": 2.0 * x + rng.normal(size=n)})


@pytest.mark.parametrize("key", ["emit_calibration", "emit_breaks"])
@pytest.mark.parametrize("tau", [0.1, 0.9, 0.5 + 1e-9])
def test_an_expectile_off_the_mean_refuses_the_mean_checks(key, tau):
    """An expectile at a level other than one half is not the mean, so its
    residuals' mean is not 0 on a stream it fits exactly: the
    Mincer-Zarnowitz Wald and the CUSUM, whose nulls say it is, rejected on
    every row of such a stream (review 2026-11-23, DIAG-11). Each is refused
    there, naming the level."""
    with pytest.raises(ValueError, match=f"{key} tests a fit of the mean, but expectile"):
        rp.ModelBank(
            [
                rp.spec.expectile(
                    "m", features=["x"], targets=["y"], half_life=50.0, expectile=tau, **{key: True}
                )
            ]
        )


def test_the_mean_expectile_keeps_every_check_at_its_level():
    """At one half the expectile is the mean, and every residual check holds
    its 5% level on a stream it fits exactly, after its first thousand rows;
    off it the tests that read no mean are taken and hold theirs too."""
    df = _exact_linear_stream()
    for tau, keys in [
        (0.5, ["emit_calibration", "emit_breaks", "emit_specification", "emit_tails"]),
        (0.9, ["emit_specification", "emit_tails"]),
    ]:
        spec = rp.spec.expectile(
            "m",
            features=["x"],
            targets=["y"],
            half_life=50.0,
            expectile=tau,
            **dict.fromkeys(keys, True),
        )
        out = rp.ModelBank([spec]).fit_predict(df)["m"].struct.unnest()[1000:]
        cut = {
            "calibration_wald_y": 5.991,
            "breusch_pagan_y": 3.841,
            "ramsey_reset_y": 5.991,
        }
        for field, c in cut.items():
            if field in out.columns:
                assert (out[field] > c).mean() < 0.1, (tau, field)
        if "cusum_y" in out.columns:
            assert (out["cusum_y"].abs() > 1.96).mean() < 0.1, tau


def test_a_clock_window_withholds_only_where_its_weight_says_so():
    """A row's ``min_weight`` gate reads the weight it reports, the
    window's before the rows that leave at the row (hard rule 8): five rows
    at 0 to 4 under a window of 5 and a floor of 3, then a row at 8, by
    which four have left. The gate read the one row left and withheld the
    prediction while ``weight_sum`` said 5 and no reason was given (review
    2026-11-23, EXP-4); it now reads the fit of the row left. A row at 20,
    by which every row has left, has no fit at any level, rather than the
    fit of the rows that left (EXP-1)."""
    rng = np.random.default_rng(578)
    for tau in (0.5, 0.9):
        df = pl.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 20.0],
                "x0": rng.standard_normal(7),
                "x1": rng.standard_normal(7),
                "y": rng.standard_normal(7),
            }
        )
        out = fields(
            _spec(
                expectile=tau,
                half_life=INF,
                window_size=5.0,
                clock="t",
                gap_cap=100.0,
                min_weight=3.0,
                ridge=1e-6,
            ),
            df,
        )
        assert out["weight_sum"][5] == 5.0
        assert out["pred_y"][5] is not None, out
        assert out["withheld_reason"][5] is None
        assert out["pred_y"][6] is None, out
