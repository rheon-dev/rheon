"""``dcc`` through the bank: the correlation recursion against its
definition, row by row, and what the bank owes every model -- out-of-sample
reads, chunk invariance, a save mid-stream, zero weights and nulls, weights
as relative evidence, its refusals, its margins as ``garch``'s, the notice
for a margin with no fourth moment, and a start over a span of the clock
held to the group's state cap (task 493).

The oracle for the recursion is Engle's (2002) definition, with
Cappiello, Engle and Sheppard's (2006) asymmetric term, written out in
numpy: ``Q_{t+1} = (1 - a - b) Q̄ - g N̄ + a z zᵀ + g n nᵀ + b Q_t``, ``Q̄`` and
``N̄`` the means of the start's rows and then the running means, ``R_t`` read
off ``Q_t``. The batch quasi-likelihood fit is the oracle for the
estimator, in ``tests/test_second_opinion_dcc.py``.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import polars as pl
import pytest
from scipy.stats import multivariate_normal

import rheon
from rheon import polars as rp

TIER = "essential"


def _z(n: int, seed: int, k: int = 3, theta=(0.04, 0.92, 0.03), rho: float = 0.5) -> np.ndarray:
    """A realistic ADCC stream of standardized residuals: ``k`` series,
    every target correlation ``rho``, ``N`` at half the target (which keeps
    the truth's intercept positive definite), 500 rows discarded."""
    rng = np.random.default_rng(seed)
    a, b, g = theta
    qbar = np.full((k, k), rho)
    np.fill_diagonal(qbar, 1.0)
    nbar = 0.5 * qbar
    q = qbar.copy()
    out = np.empty((n, k))
    for t in range(n + 500):
        d = np.sqrt(np.diag(q))
        z = np.linalg.cholesky(q / np.outer(d, d)) @ rng.standard_normal(k)
        m = np.minimum(z, 0.0)
        q = (1 - a - b) * qbar - g * nbar + a * np.outer(z, z) + g * np.outer(m, m) + b * q
        if t >= 500:
            out[t - 500] = z
    return out


def _returns(z: np.ndarray, seed: int) -> np.ndarray:
    """Each series of ``z`` scaled by its own GARCH(1,1) at ``(0.05, 0.08,
    0.9)``."""
    del seed
    s = np.full(z.shape[1], 2.5)
    e = np.zeros(z.shape[1])
    out = np.empty_like(z)
    for t in range(len(z)):
        s = 0.05 + 0.08 * e * e + 0.9 * s
        e = np.sqrt(s) * z[t]
        out[t] = e
    return out


NAMES = ["a", "b", "c"]


def _frame(x: np.ndarray, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(x)
    cols = {NAMES[j]: x[:, j] for j in range(x.shape[1])}
    return pl.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "g": np.where(np.arange(n) % 3 == 0, "p", "q"),
            **cols,
            "w": np.where(rng.uniform(size=n) < 0.1, 0.0, 1.0 + rng.integers(0, 3, n)),
        }
    )


def _spec(k: int = 3, **kw) -> dict:
    base = {"margins": "none", "start_rows": 100, "half_life": float("inf")}
    return rp.spec.dcc("m", targets=NAMES[:k], **{**base, **kw})


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")


def _longhand(z: np.ndarray, theta, start: int, half_life: float = float("inf")):
    """The correlations and log-densities the definition gives, each row's
    before the row: ``Q̄`` and ``N̄`` the decayed means of the rows learned,
    the start's at its end, ``Q`` from ``Q̄`` over the start's rows and on."""
    n, k = z.shape
    lam = 0.5 ** (1.0 / half_life)
    a, b = theta[0], theta[1]
    g = theta[2] if len(theta) > 2 else 0.0
    zz = np.einsum("ti,tj->tij", z, z)
    neg = np.minimum(z, 0.0)
    nn = np.einsum("ti,tj->tij", neg, neg)
    wts = lam ** np.arange(start - 1, -1, -1)
    qbar = np.tensordot(wts, zz[:start], 1) / wts.sum()
    nbar = np.tensordot(wts, nn[:start], 1) / wts.sum()
    big_w = wts.sum()
    q = qbar.copy()
    for t in range(start):
        q = (1 - a - b) * qbar - g * nbar + a * zz[t] + g * nn[t] + b * q
    corr = np.full((n, k * (k - 1) // 2), np.nan)
    dens = np.full(n, np.nan)
    iu = np.triu_indices(k, 1)
    for t in range(start, n):
        d = np.sqrt(np.diag(q))
        r = q / np.outer(d, d)
        corr[t] = r[iu]
        dens[t] = multivariate_normal(np.zeros(k), r).logpdf(z[t])
        big_w = lam * big_w + 1.0
        qbar = qbar + (zz[t] - qbar) / big_w
        nbar = nbar + (nn[t] - nbar) / big_w
        q = (1 - a - b) * qbar - g * nbar + a * zz[t] + g * nn[t] + b * q
    return corr, dens


@pytest.mark.parametrize(
    "kw",
    [
        {"a": 0.05, "b": 0.9},
        {"asymmetric": True, "a": 0.03, "b": 0.93, "g": 0.03},
        {"asymmetric": True, "a": 0.03, "b": 0.93, "g": 0.03, "half_life": 300.0},
    ],
)
def test_the_correlation_is_the_definition(kw):
    """With the parameters given, each row's correlations are Engle's
    recursion from the row before, and its log-density scipy's Gaussian
    density under them, to rounding: on a realistic stream, three series."""
    z = _z(1200, 3)
    df = pl.DataFrame({n: z[:, j] for j, n in enumerate(NAMES)})
    out = _out(_spec(**kw), df)
    theta = [kw["a"], kw["b"]] + ([kw["g"]] if "g" in kw else [])
    corr, dens = _longhand(z, theta, 100, kw.get("half_life", float("inf")))
    got = out.select("corr_a_b", "corr_a_c", "corr_b_c").to_numpy()
    assert np.isnan(got[:100]).all() and out["loglik"][:100].is_null().all()
    np.testing.assert_allclose(got[100:], corr[100:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out["loglik"].to_numpy()[100:], dens[100:], rtol=1e-12, atol=1e-12)
    # inside the set, the given parameters stand as given
    assert out["coef"][-1].to_list() == theta


def test_a_rows_correlation_is_read_before_the_row():
    """A row's correlation does not read the row: changing it moves its
    log-density and the next row's correlation alone."""
    z = _z(600, 4)
    df = pl.DataFrame({n: z[:, j] for j, n in enumerate(NAMES)})
    s = _spec()
    a = _out(s, df)
    b = _out(
        s,
        df.with_columns(
            pl.when(pl.int_range(600) == 400).then(4.0).otherwise(pl.col("a")).alias("a")
        ),
    )
    assert a["corr_a_b"][:401].equals(b["corr_a_b"][:401])
    assert a["loglik"][400] != b["loglik"][400]
    assert a["corr_a_b"][401] != b["corr_a_b"][401]


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"asymmetric": True, "half_life": 200.0},
        {"margins": "garch", "margin_start_rows": 60, "vol": "gjr"},
    ],
)
def test_chunk_invariance(kw):
    x = _z(700, 5)
    if kw.get("margins") == "garch":
        x = _returns(x, 5)
    df = _frame(x, 5)
    s = _spec(group="g", weight="w", coef_every=0, **kw)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 13, 211):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 700, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"asymmetric": True},
        {"margins": "garch", "margin_start_rows": 60},
        {"a": 0.04, "b": 0.9},
    ],
)
def test_a_save_goes_on_to_the_bit(kw):
    x = _z(800, 6)
    if kw.get("margins") == "garch":
        x = _returns(x, 6)
    df = _frame(x, 6)
    s = _spec(group="g", weight="w", **kw)
    for cut in (1, 150, 330, 799):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


@pytest.mark.parametrize(
    "kw", [{"asymmetric": True}, {"margins": "garch", "margin_start_rows": 500, "half_life": 400.0}]
)
def test_a_zero_weight_row_counts_in_no_clock(kw):
    """The stream with rows of weight 0 gives every other row, to the bit,
    the output of the stream without them, on an irregular clock: such a
    row learns nothing and counts in no clock. They are still read."""
    n = 2500
    x = _z(n, 7)
    if kw.get("margins") == "garch":
        x = _returns(x, 7)
    df = _frame(x, 7).drop("g").with_columns(t=pl.Series(np.cumsum(1.0 + np.arange(n) % 3)))
    s = _spec(weight="w", coef_every=0, clock="t", gap_cap=1e9, **kw)
    keep = (df["w"] > 0).to_numpy()
    full = _out(s, df)
    thin = _out(s, df.filter(pl.Series(keep)))
    for f in ("corr_a_b", "corr_b_c", "loglik", "weight_sum", "coef"):
        assert full[f].filter(pl.Series(keep)).equals(thin[f], null_equal=True), f
    read = full.filter(pl.Series(~keep))
    assert read["corr_a_b"].drop_nulls().len() > 20, "such a row is still read"


def test_a_row_with_a_series_null_steps_no_recursion():
    """A row with any series null steps neither the correlation's
    recursion nor its estimate: the correlations of every other row are the
    stream's without it, to the bit. It still counts its weight."""
    df = _frame(_z(700, 8), 8).drop("g")
    df = df.with_columns(
        pl.when(pl.int_range(700) % 17 == 5).then(None).otherwise(pl.col("b")).alias("b")
    )
    s = _spec(weight="w", coef_every=0, asymmetric=True)
    keep = df["b"].is_not_null().to_numpy()
    full = _out(s, df)
    thin = _out(s, df.filter(pl.Series(keep)))
    for f in ("corr_a_b", "corr_b_c", "loglik", "coef"):
        assert full[f].filter(pl.Series(keep)).equals(thin[f], null_equal=True), f
    assert full.filter(pl.Series(~keep))["loglik"].is_null().all()


def test_a_weight_scales_nothing_but_the_weight():
    """Every weight doubled leaves the correlations and the fit where they
    were, to rounding; ``weight_sum`` doubles."""
    df = _frame(_z(800, 9), 9).drop("g")
    s = _spec(weight="w", coef_every=0, asymmetric=True)
    a = _out(s, df)
    b = _out(s, df.with_columns(pl.col("w") * 2.0))
    np.testing.assert_allclose(
        a["corr_a_b"].to_numpy(), b["corr_a_b"].to_numpy(), rtol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(b["weight_sum"].to_numpy(), 2 * a["weight_sum"].to_numpy())


def test_the_margins_are_garchs():
    """With margins, each target's variance is the ``garch`` model's, run
    beside it on the same returns, to the bit; the correlation reports from
    the margins' start plus its own."""
    x = _returns(_z(1500, 10, k=2), 10)
    df = pl.DataFrame({"a": x[:, 0], "b": x[:, 1]})
    d = _out(
        rp.spec.dcc("m", targets=["a", "b"], half_life=float("inf"), margin_start_rows=300), df
    )
    g = _out(rp.spec.garch("m", targets=["a", "b"], start_rows=300), df)
    assert d["variance_a"].equals(g["variance_a"], null_equal=True)
    assert d["variance_b"].equals(g["variance_b"], null_equal=True)
    # the margins' start (300 rows, or twice or four times that where its fit
    # sat on a bound), then the correlation's 500
    first = d["corr_a_b"].is_not_null().arg_max()
    assert first == d["variance_b"].is_not_null().arg_max() + 500


@pytest.mark.parametrize(
    "kw, says",
    [
        ({"a": 0.05}, "Give a and b, or leave every one out"),
        ({"a": 0.05, "b": 0.9, "g": 0.01}, "set asymmetric to true"),
        ({"a": 0.5, "b": 0.5}, r"a \+ b must be at most 1 - 1e-4"),
        ({"a": -0.1, "b": 0.5}, "finite and at least 0"),
        ({"margins": "none", "vol": "gjr"}, "runs no GARCH margin"),
        (
            {"margins": "garch", "vol": "figarch", "margin_start_rows": 100},
            "margin_start_rows must be at least 2000",
        ),
        ({"start_rows": 0}, "start_rows must be at least 1"),
        ({"start_rows": 100, "start_span": 10.0}, "Leave one of them out"),
        ({"max_state_bytes": 1000}, "past max_state_bytes of 1000"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([_spec(**kw)])


def test_one_target_and_features_are_refused():
    with pytest.raises(ValueError, match="at least two targets"):
        rp.ModelBank([_spec(k=1)])
    with pytest.raises(TypeError, match="features"):
        rp.spec.dcc("m", targets=["a", "b"], features=["x"], half_life=1.0)  # type: ignore[call-arg]


def test_a_margin_with_no_fourth_moment_is_said_once():
    """A Student-t margin whose degrees of freedom are fitted at or below 4
    has residuals with no fourth moment: the bank says so once, naming the
    target, and runs on."""
    rng = np.random.default_rng(12)
    n = 1500
    x = _returns(rng.standard_t(3, size=(n, 2)) / np.sqrt(3.0), 12)
    df = pl.DataFrame({"a": x[:, 0], "b": x[:, 1]})
    s = rp.spec.dcc(
        "m", targets=["a", "b"], dist="t", margin_start_rows=1000, half_life=float("inf")
    )
    with pytest.warns(rheon.RheonWarning, match=r"margin of target \"[ab]\".*at most 4") as seen:
        rp.ModelBank([s]).fit_predict(df)
    assert sum("at most 4" in str(w.message) for w in seen) == 1


def test_a_clock_span_start_is_warned_of_once_and_refused_at_the_cap():
    """A correlation start over a span of the clock buffers the rows the
    span has, which the data decides: the first row to take a group past
    half of ``max_state_bytes`` warns, once; the row that would take it past
    the cap stops the run naming the group and the bytes."""
    # the state of 300 buffered rows of two series: the stage's 32·4 + 96,
    # 32 of the model's, 32 bytes a row
    cap = 32 * 4 + 96 + 32 + 32 * 300
    s = rp.spec.dcc(
        "m",
        targets=["a", "b"],
        margins="none",
        group="g",
        clock="t",
        gap_cap=1e9,
        start_span=1e9,
        half_life=float("inf"),
        max_state_bytes=cap,
    )

    def rows(group: str, n: int, first: int = 0) -> pl.DataFrame:
        z = _z(n, first + 1, k=2)
        i = np.arange(first, first + n)
        return pl.DataFrame({"g": group, "t": i.astype(float), "a": z[:, 0], "b": z[:, 1]})

    bank = rp.ModelBank([s])
    with pytest.warns(rheon.RheonWarning, match=r'group "a": the state holds \d+ bytes') as seen:
        bank.fit_predict(pl.concat([rows("a", 200), rows("b", 30)]))
    assert len(seen) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bank.fit_predict(rows("a", 30, 200))
    with pytest.raises(Exception, match=r'group "a".*start_span.*above max_state_bytes of') as e:
        bank.fit_predict(rows("a", 200, 230))
    assert str(cap) in str(e.value)


def test_a_start_span_on_a_temporal_clock_is_a_duration():
    z = _z(400, 13, k=2)
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2026, 1, 1), pl.datetime(2026, 1, 1, 6, 39), "1m", eager=True
            ),
            "a": z[:, 0],
            "b": z[:, 1],
        }
    )
    # the parameters given: the span alone decides where the start closes
    s = _spec(k=2, clock="ts", gap_cap="1h", start_rows=None, start_span="2h", a=0.05, b=0.9)
    out = _out(s, df)
    # the rows from the first through the first at or past two hours on
    assert out["corr_a_b"].is_not_null().arg_max() == 121
    with pytest.raises(ValueError, match="start_span"):
        rp.ModelBank([_spec(k=2, clock="ts", gap_cap="1h", start_rows=None, start_span=120.0)])


@pytest.mark.parametrize("asymmetric", [False, True])
def test_a_start_reports_only_a_recursion_its_rows_identify(asymmetric):
    """The start waits on the evidence, not on a count of rows: on series of
    one constant correlation (no recursion to find) the fit is withheld,
    each row after the start's buffer naming the wait ``not_identified``,
    and nothing is reported; on a realistic stream the same spec reports
    from its start's row."""
    null = _z(2000, 14, theta=(0.0, 0.0, 0.0))
    s = _spec(start_rows=300, asymmetric=asymmetric)
    out = _out(s, pl.DataFrame({n: null[:, j] for j, n in enumerate(NAMES)}))
    assert out["corr_a_b"].is_null().all() and out["coef"].list.get(0).is_null().all()
    reasons = out["withheld_reason"].cast(pl.String)
    assert (reasons[:300] == "below_warm_rows").all()
    assert (reasons[300:] == "not_identified").all()
    live = _z(2000, 15, theta=(0.04, 0.92, 0.04 if asymmetric else 0.0))
    out = _out(s, pl.DataFrame({n: live[:, j] for j, n in enumerate(NAMES)}))
    first = out["corr_a_b"].is_not_null().arg_max()
    assert first in (300, 600, 1200), first
    assert out["withheld_reason"][first:].is_null().all()
