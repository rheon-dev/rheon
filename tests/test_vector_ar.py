"""``vector_ar`` through the bank, held to statsmodels: its coefficients,
standard errors and Granger tests against ``VAR(y, exog).fit(p)`` and
``grangercausalitytests`` on the same rows with no decay, its order against
``select_order``, its reads (impulse responses, their standard errors, the
variance decomposition, the roots, the mean) against ``irf``, ``fevd`` and
``VARProcess``, under decay against ``WLS`` per equation and ``VARProcess``
of the decayed fit's own parameters, and the p-values against scipy's
tails; then what the bank owes every model -- chunk invariance, a save
mid-stream, zero-weight and incomplete rows, a break that empties the lags,
a window, its refusals and its output schema.

statsmodels 0.15's ``test_causality(causing=None)`` is wrong for three or
more series, so every test here names ``causing``.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from scipy import stats
from statsmodels.regression.linear_model import WLS
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VARProcess

from rheon import polars as rp

TIER = "essential"
INF = float("inf")


def _var(n: int, k: int, p: int, seed: int, *, exog: bool = False, level: float = 0.0):
    """A stable VAR(p) of ``k`` series with true cross effects -- each series
    driven by its own lags and the next series' first lag -- and, with
    ``exog``, a regressor ``x`` in every equation, after 300 rows of
    burn-in. Returns a frame of ``y0 .. y{k-1}`` (and ``x``) and ``t``."""
    rng = np.random.default_rng(seed)
    a = np.zeros((p, k, k))
    a[0] = 0.4 * np.eye(k)
    for i in range(k - 1):
        a[0, i, i + 1] = 0.25
    if p > 1:
        a[1] = -0.15 * np.eye(k)
    c = np.linspace(-1.0, 1.0, k)
    b = np.linspace(0.5, -0.5, k)
    m = n + 300
    x = rng.standard_normal(m)
    y = np.zeros((m, k))
    for t in range(p, m):
        y[t] = c + rng.standard_normal(k) + (b * x[t] if exog else 0.0)
        for j in range(p):
            y[t] += a[j] @ y[t - 1 - j]
    cols = {f"y{i}": y[300:, i] + level for i in range(k)}
    if exog:
        cols["x"] = x[300:]
    return pl.DataFrame({"t": np.arange(n, dtype=float), **cols})


def _names(k: int) -> list[str]:
    return [f"y{i}" for i in range(k)]


def _out(spec: dict, df: pl.DataFrame) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(df).unnest("v")


def _rel(got, want) -> float:
    got, want = np.asarray(got, float), np.asarray(want, float)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _params(out: pl.DataFrame, k: int) -> np.ndarray:
    """The last row's ``coef`` as statsmodels' ``params``: one column per
    equation."""
    c = np.asarray(out["coef"][-1].to_list())
    return c.reshape(k, -1).T


def _ys(df: pl.DataFrame, k: int) -> np.ndarray:
    return df.select(_names(k)).to_numpy()


def _at(names: list[str]) -> list[int]:
    """statsmodels' positions of ``y<i>`` columns: an array's series are
    named ``y1 ..`` there, so they are passed by position."""
    return [int(n[1:]) for n in names]


@pytest.mark.parametrize(("k", "p", "exog"), [(2, 1, False), (3, 2, True), (5, 2, False)])
def test_no_decay_is_statsmodels_var(k, p, exog):
    """With no decay the coefficients and their standard errors are
    ``VAR(y, exog).fit(p)``'s on the same rows: ``params`` within 1e-12 of
    the largest, ``bse`` within 1e-11, statsmodels' order of terms
    (``const``, the exog, then ``L1.y0 .. Lp.y{k-1}``)."""
    df = _var(1500, k, p, 1, exog=exog)
    spec = rp.spec.vector_ar(
        "v",
        targets=_names(k),
        lags=p,
        features=["x"] if exog else None,
        half_life=INF,
        emit_se_coef=True,
    )
    out = _out(spec, df)
    res = VAR(_ys(df, k), exog=df["x"].to_numpy()[:, None] if exog else None).fit(p)
    assert _rel(_params(out, k), res.params) < 1e-12
    se = np.asarray(out["se_coef"][-1].to_list()).reshape(k, -1).T
    assert _rel(se, res.bse) < 1e-11


def test_without_an_intercept_it_is_trend_n():
    df = _var(1200, 2, 2, 2)
    spec = rp.spec.vector_ar("v", targets=_names(2), lags=2, fit_intercept=False, half_life=INF)
    res = VAR(_ys(df, 2)).fit(2, trend="n")
    assert _rel(_params(_out(spec, df), 2), res.params) < 1e-12


def test_a_prediction_is_the_var_fitted_on_the_rows_before():
    """Hard rule 2: row ``t``'s predictions are statsmodels' one-step
    forecast from ``VAR`` fitted on the rows before ``t``."""
    df = _var(300, 3, 2, 3)
    out = _out(rp.spec.vector_ar("v", targets=_names(3), lags=2, half_life=INF), df)
    y = _ys(df, 3)
    pred = out.select([f"pred_{n}" for n in _names(3)]).to_numpy()
    for t in (20, 150, 299):
        want = VAR(y[:t]).fit(2).forecast(y[t - 2 : t], 1)[0]
        assert _rel(pred[t], want) < 1e-10, t
    # Nothing before the ring is full and the fit identified: seven
    # coefficients an equation need seven learned rows after the two lags.
    assert np.isnan(pred[:9]).all() and np.isfinite(pred[9:]).all()
    assert out["withheld_reason"][3] == "not_identified"


def _causality_specs(k: int) -> list[tuple[list[str], list[str]]]:
    names = _names(k)
    return [
        ([names[0]], [names[1]]),
        ([names[0]], names[1:]),
        ([names[1], names[2]], [names[0]]),
        ([names[2]], [names[0], names[1]]),
    ]


@pytest.mark.parametrize("kind", ["f", "wald"])
def test_granger_tests_are_test_causality(kind):
    """Each declared test, read per row, is ``test_causality(caused,
    causing, kind)`` on the same rows: the statistic within 1e-11, the
    p-value within 1e-10 of it, relative, and that p-value is scipy's tail
    at the statistic and its degrees of freedom."""
    k, p = 3, 2
    df = _var(800, k, p, 4)
    tests = _causality_specs(k)
    spec = rp.spec.vector_ar(
        "v",
        targets=_names(k),
        lags=p,
        causality=[{"caused": c, "causing": g} for c, g in tests],
        causality_kind=kind,
        half_life=INF,
    )
    bank = rp.ModelBank([spec])
    bank.fit(df)
    # The tests a row reports read the fit before it: the last fit is read
    # by the next row, or after the fact by the bank.
    res = VAR(_ys(df, k)).fit(p)
    nobs = res.nobs
    df_resid = nobs - (k * p + 1)
    for i, (caused, causing) in enumerate(tests):
        want = res.test_causality(_at(caused), _at(causing), kind=kind)
        got = bank.vector_ar(table="causality", caused=caused, causing=causing, kind=kind)
        stat, pv = got["statistic"][0], got["p_value"][0]
        assert abs(stat / want.test_statistic - 1) < 1e-11, i
        assert abs(pv / want.pvalue - 1) < 1e-10, (i, pv, want.pvalue)
        q = p * len(caused) * len(causing)
        tail = stats.f.sf(stat, q, k * df_resid) if kind == "f" else stats.chi2.sf(stat, q)
        assert abs(pv / tail - 1) < 1e-10
    # Per row: the row after the stream's last would read these; the last
    # row reads the fit of every row but its own.
    out = rp.ModelBank([spec]).fit_predict(df).unnest("v")
    res_before = VAR(_ys(df, k)[:-1]).fit(p)
    for i, (caused, causing) in enumerate(tests):
        want = res_before.test_causality(_at(caused), _at(causing), kind=kind)
        assert abs(out[f"causality{i}_stat"][-1] / want.test_statistic - 1) < 1e-11
        assert abs(out[f"causality{i}_p"][-1] / want.pvalue - 1) < 1e-10


def test_the_single_equation_tests_are_grangercausalitytests():
    """``ssr_f``, ``ssr_chi2`` and ``lr`` are ``grangercausalitytests``' three
    on the pair: each statistic within 1e-11 and its p-value within 1e-10,
    both directions, the one with no effect included."""
    p = 3
    df = _var(500, 2, p, 5)
    y = _ys(df, 2)
    gc = {e: grangercausalitytests(y[:, [e, 1 - e]], [p])[p][0] for e in (0, 1)}
    names = {"ssr_f": "ssr_ftest", "ssr_chi2": "ssr_chi2test", "lr": "lrtest"}
    for kind, theirs in names.items():
        spec = rp.spec.vector_ar("v", targets=_names(2), lags=p, half_life=INF)
        bank = rp.ModelBank([spec])
        bank.fit(df)
        for e in (0, 1):
            got = bank.vector_ar(table="causality", caused=f"y{e}", causing=f"y{1 - e}", kind=kind)
            want = gc[e][theirs]
            assert abs(got["statistic"][0] / want[0] - 1) < 1e-11, (kind, e)
            assert abs(got["p_value"][0] / want[1] - 1) < 1e-10, (kind, e)


@pytest.mark.parametrize("ic", ["aic", "bic", "hqic", "fpe"])
def test_select_is_select_order(ic):
    """``select`` picks ``VAR.select_order(p)``'s order, and the bank's
    ``order`` table is its ``ics``, every order on the same rows, within
    1e-13 of each value."""
    k, pmax = 3, 5
    df = _var(800, k, 2, 6)
    spec = rp.spec.vector_ar("v", targets=_names(k), lags=pmax, select=ic, half_life=INF)
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(df).unnest("v")
    sel = VAR(_ys(df, k)).select_order(pmax)
    table = bank.vector_ar(table="order")
    for name in ("aic", "bic", "hqic", "fpe"):
        assert _rel(table[name].to_numpy(), np.asarray(sel.ics[name])) < 1e-13, name
    want = sel.selected_orders[ic]
    assert table.filter("in_force")["order"][0] == want
    assert out["order"][-1] == want or out["order"][-2] == want


def test_the_impulse_responses_and_decomposition_are_statsmodels():
    """``irf``, ``orth_irf`` and their standard errors, ``fevd``, the roots
    and the long-run mean are ``VARResults``' on the same fit: the responses
    and shares within 1e-11, the standard errors within 1e-10."""
    k, p, h = 3, 2, 8
    df = _var(1000, k, p, 7)
    spec = rp.spec.vector_ar("v", targets=_names(k), lags=p, half_life=INF)
    bank = rp.ModelBank([spec])
    bank.fit(df)
    res = VAR(_ys(df, k)).fit(p)
    irf = res.irf(h)

    def cube(frame: pl.DataFrame, col: str, first: str) -> np.ndarray:
        frame = frame.sort(first, "response", "impulse" if "impulse" in frame.columns else "shock")
        n = frame[first].n_unique()
        return frame[col].to_numpy().reshape(n, k, k)

    plain = bank.vector_ar(table="irf", periods=h, stderr=True)
    orth = bank.vector_ar(table="orth_irf", periods=h, stderr=True)
    assert _rel(cube(plain, "value", "period"), irf.irfs) < 1e-11
    assert _rel(cube(orth, "value", "period"), irf.orth_irfs) < 1e-11
    assert _rel(cube(plain, "stderr", "period"), irf.stderr(orth=False)) < 1e-10
    assert _rel(cube(orth, "stderr", "period"), irf.stderr(orth=True)) < 1e-10
    fevd = bank.vector_ar(table="fevd", periods=h)
    got = fevd.sort("response", "horizon", "shock")["share"].to_numpy().reshape(k, h, k)
    assert _rel(got, res.fevd(h).decomp) < 1e-11
    roots = bank.vector_ar(table="stability")
    want = np.sort(np.abs(res.roots))
    assert _rel(np.sort(1.0 / roots["modulus"].to_numpy()), want) < 1e-10
    assert roots["stable"].all() and res.is_stable()
    mean = bank.vector_ar(table="mean")["mean"].to_numpy()
    assert _rel(mean, res.mean()) < 1e-11


def _decayed(df: pl.DataFrame, k: int, p: int, h: float, w: np.ndarray, t: np.ndarray):
    """Each equation's ``WLS`` on the lag design with weights ``w λ^age``,
    its age on the clock: ``params`` as statsmodels lays them out."""
    y = _ys(df, k)
    n = len(y)
    z = np.column_stack([np.ones(n - p)] + [y[p - 1 - j : n - 1 - j] for j in range(p)])
    ww = w[p:] * 0.5 ** ((t[-1] - t[p:]) / h)
    return np.column_stack([WLS(y[p:, e], z, weights=ww).fit().params for e in range(k)])


def test_decay_is_weighted_least_squares_and_its_reads_are_varprocess():
    """Under ``half_life`` on an uneven clock with row weights, the fit is
    ``WLS`` per equation with each row weighted ``w 0.5^(age / half_life)``,
    within 1e-11; and the impulse responses and decomposition read from it
    are ``VARProcess``'s of the fit's own coefficients and ``Σ_u``."""
    k, p, n, h = 3, 2, 2000, 300.0
    df = _var(n, k, p, 8)
    rng = np.random.default_rng(80)
    t = np.cumsum(rng.uniform(0.3, 1.7, n))
    w = rng.uniform(0.5, 1.5, n)
    df = df.with_columns(pl.Series("t", t), pl.Series("w", w))
    # A solve every row: the default under a half-life solves by weight,
    # and the last row's coefficients would be the last solve's.
    spec = rp.spec.vector_ar(
        "v",
        targets=_names(k),
        lags=p,
        clock="t",
        gap_cap=10.0,
        weight="w",
        half_life=h,
        max_rows_between_solves=1,
    )
    bank = rp.ModelBank([spec])
    out = bank.fit_predict(df).unnest("v")
    want = _decayed(df, k, p, h, w, t)
    assert _rel(_params(out, k), want) < 1e-11
    # The reads, from the fit's own parameters.
    hh = 6
    irf = bank.vector_ar(table="irf", periods=hh).sort("period", "response", "impulse")
    ours = irf["value"].to_numpy().reshape(hh + 1, k, k)
    coefs = np.stack([want[1 + j * k : 1 + (j + 1) * k].T for j in range(p)])
    orth = bank.vector_ar(table="orth_irf", periods=hh).sort("period", "response", "impulse")
    proc_ma = VARProcess(coefs, want[0], np.eye(k)).ma_rep(hh)
    assert _rel(ours, proc_ma) < 1e-10
    # Σ_u under decay: the residuals' weighted co-moments with Kish's count.
    y = _ys(df, k)
    z = np.column_stack([np.ones(n - p)] + [y[p - 1 - j : n - 1 - j] for j in range(p)])
    ww = w[p:] * 0.5 ** ((t[-1] - t[p:]) / h)
    resid = y[p:] - z @ want
    sse = (resid * ww[:, None]).T @ resid / ww.sum()
    nk = ww.sum() ** 2 / (ww**2).sum()
    sigma = sse * nk / (nk - (k * p + 1))
    proc = VARProcess(coefs, want[0], sigma)
    got_orth = orth["value"].to_numpy().reshape(hh + 1, k, k)
    assert _rel(got_orth, proc.orth_ma_rep(hh)) < 1e-9
    fevd = bank.vector_ar(table="fevd", periods=hh).sort("response", "horizon", "shock")
    theta2 = np.cumsum(proc.orth_ma_rep(hh)[:hh] ** 2, axis=0)
    mse = proc.mse(hh)
    decomp = (theta2 / np.diagonal(mse, axis1=1, axis2=2)[:, :, None]).swapaxes(0, 1)
    assert _rel(fevd["share"].to_numpy().reshape(k, hh, k), decomp) < 1e-9


def test_an_incomplete_row_is_dropped_and_still_predicted():
    """A row with a target null teaches nothing and is not a lag: the fit
    is ``VAR`` on the rows with those dropped (statsmodels'
    ``missing="drop"``, the lags joining across them), and the row is
    still predicted, every target, from the lags before it."""
    k, p = 3, 2
    df = _var(1500, k, p, 9)
    holes = np.arange(1500) % 17 == 5
    df = df.with_columns(pl.when(pl.Series(holes)).then(None).otherwise(pl.col("y1")).alias("y1"))
    out = _out(rp.spec.vector_ar("v", targets=_names(k), lags=p, half_life=INF), df)
    kept = _ys(df.filter(~pl.Series(holes)), k)
    assert _rel(_params(out, k), VAR(kept).fit(p).params) < 1e-12
    on_holes = out.filter(pl.Series(holes))
    for name in _names(k):
        assert on_holes[f"pred_{name}"].drop_nulls().len() > 80
    assert on_holes["resid_y1"].drop_nulls().len() == 0
    assert on_holes["resid_y0"].drop_nulls().len() > 80


def test_a_zero_weight_row_counts_in_no_clock():
    """Rows of weight 0 -- wild values among them, the first rows too --
    leave every other row's output, to the bit, as the stream without them
    gives it, under decay on an uneven clock."""
    n, k = 1500, 2
    df = _var(n, k, 2, 10)
    rng = np.random.default_rng(100)
    zero = rng.uniform(size=n) < 0.1
    zero[:3] = True
    df = df.with_columns(
        pl.Series("t", np.cumsum(rng.uniform(0.5, 1.5, n))),
        pl.Series("w", np.where(zero, 0.0, 1.0)),
        pl.when(pl.Series(zero)).then(1e6).otherwise(pl.col("y0")).alias("y0"),
    )
    spec = rp.spec.vector_ar(
        "v",
        targets=_names(k),
        lags=2,
        causality=[{"caused": "y0"}],
        clock="t",
        gap_cap=10.0,
        weight="w",
        half_life=60.0,
        coef_every=0,
    )
    full = _out(spec, df)
    thin = _out(spec, df.filter(~pl.Series(zero)))
    keep = full.filter(~pl.Series(zero))
    for f in ("pred_y0", "pred_y1", "weight_sum", "coef", "causality0_stat", "causality0_p"):
        assert keep[f].equals(thin[f], null_equal=True), f


def test_a_session_change_empties_the_lags_and_keeps_the_fit():
    """On a session change the lags empty: the next two rows of a VAR(2)
    are not predicted, and the fit at the end is least squares over each
    session's own rows, its first two only lags."""
    k = 2
    df = _var(1200, k, 2, 11).with_columns(pl.Series("s", np.repeat(["a", "b", "c"], 400)))
    spec = rp.spec.vector_ar(
        "v",
        targets=_names(k),
        lags=2,
        clock="t",
        gap_cap=10.0,
        session="s",
        session_gap=1.0,
        half_life=INF,
    )
    out = _out(spec, df)
    pred = out["pred_y0"].to_numpy()
    assert np.isnan(pred[[400, 401, 800, 801]]).all()
    assert np.isfinite(pred[[402, 802, 1199]]).all()
    y = _ys(df, k)
    rows = [
        (np.r_[1.0, y[t - 1], y[t - 2]], y[t]) for s in (0, 400, 800) for t in range(s + 2, s + 400)
    ]
    z, target = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    want = np.linalg.lstsq(z, target, rcond=None)[0]
    assert _rel(_params(out, k), want) < 1e-11


@pytest.mark.parametrize("kw", [{"half_life": INF}, {"half_life": 50.0, "weight": "w"}])
def test_chunk_invariance(kw):
    n = 700
    df = _var(n, 2, 2, 12).with_columns(
        pl.Series("g", np.where(np.arange(n) % 3 == 0, "a", "b")),
        pl.Series("w", np.where(np.arange(n) % 11 == 4, 0.0, 1.5)),
    )
    s = rp.spec.vector_ar(
        "v",
        targets=_names(2),
        lags=2,
        causality=[{"caused": "y0"}],
        select="bic",
        group="g",
        coef_every=0,
        **kw,
    )
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 13, 211):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, n, size)])
        assert one.equals(many, null_equal=True), size


def test_a_save_goes_on_to_the_bit():
    """Saved at every depth of the lags -- none, part, full -- and later, a
    bank goes on as the one it was saved from, to the bit."""
    df = _var(500, 2, 2, 13).with_columns(
        pl.Series("g", np.where(np.arange(500) % 2 == 0, "a", "b"))
    )
    s = rp.spec.vector_ar(
        "v",
        targets=_names(2),
        lags=3,
        causality=[{"caused": "y1", "causing": "y0"}],
        group="g",
        half_life=40.0,
        emit_se_coef=True,
    )
    for cut in (1, 3, 5, 8, 250, 499):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_window_is_var_on_its_rows():
    """Under ``window_size`` with no decay, the fit is ``VAR`` on the
    window's targets, each with its lags read wherever they lie: the window
    of ``W`` rows and the ``p`` before them."""
    k, p, n, w = 2, 2, 600, 150.0
    df = _var(n, k, p, 14)
    spec = rp.spec.vector_ar(
        "v", targets=_names(k), lags=p, clock="t", gap_cap=10.0, window_size=w, half_life=INF
    )
    out = _out(spec, df)
    y = _ys(df, k)
    # The window after the last row holds the rows with t in (T - 150, T]:
    # the last 150; with their lags, the last 152 rows.
    want = VAR(y[-152:]).fit(p).params
    assert _rel(_params(out, k), want) < 1e-10


def test_a_weight_scales_nothing_but_the_weight():
    df = _var(800, 2, 1, 15).with_columns(pl.Series("w", np.linspace(0.5, 2.0, 800)))
    s = rp.spec.vector_ar("v", targets=_names(2), lags=1, weight="w", half_life=100.0)
    a = _out(s, df)
    b = _out(s, df.with_columns(pl.col("w") * 2.0))
    np.testing.assert_allclose(a["pred_y0"].to_numpy(), b["pred_y0"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(b["weight_sum"].to_numpy(), 2 * a["weight_sum"].to_numpy())


def test_names_match_the_realized_struct():
    df = _var(200, 2, 2, 16, exog=True)
    for kw in (
        {},
        {"features": ["x"]},
        {"fit_intercept": False},
        {"select": "aic", "causality": [{"caused": "y0"}, {"caused": ["y1"], "causing": "y0"}]},
    ):
        spec = rp.spec.vector_ar("v", targets=_names(2), lags=2, half_life=50.0, **kw)
        out = rp.ModelBank([spec]).fit_predict(df)
        assert [f.name for f in out.schema["v"].fields] == rp.spec.output_fields(spec)
    terms = rp.spec.coef_index(
        rp.spec.vector_ar("v", targets=_names(2), lags=2, features=["x"], half_life=50.0)
    )
    assert terms.filter(target="y0")["term"].to_list() == [
        "intercept",
        "x",
        "lag1_y0",
        "lag1_y1",
        "lag2_y0",
        "lag2_y1",
    ]


def test_the_state_cap_on_both_sides():
    """A VAR(1) of two series and no features holds at most ``8·2 + 8(4·5² +
    17·5 + 64) + 8(4·5 + 16) + 8(2·2·3 + 10) + 2048`` = 4,520 bytes: taken
    at that cap, refused a byte under it, the refusal naming the bytes; ten
    series at ten lags are past the default cap."""
    kw = dict(targets=["y0", "y1"], lags=1, half_life=50.0)
    rp.ModelBank([rp.spec.vector_ar("v", max_state_bytes=4520, **kw)])
    with pytest.raises(ValueError, match="4520 bytes, past max_state_bytes of 4519"):
        rp.spec.vector_ar("v", max_state_bytes=4519, **kw)
    with pytest.raises(ValueError, match="462088 bytes, past max_state_bytes of 262144"):
        rp.spec.vector_ar("v", targets=_names(10), lags=10, half_life=50.0)


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"lags": 0}, "lags must be at least 1"),
        ({"ridge": -1.0}, "ridge must be finite and at least 0"),
        ({"ridge": 0.1, "select": "aic"}, "select compares"),
        ({"ridge": 0.1, "causality": [{"caused": "y0"}]}, "law holds for least squares"),
        ({"causality": [{"caused": "z"}]}, "not one of the targets"),
        ({"causality": [{"caused": "y0", "causing": "y0"}]}, "both caused and causing"),
        (
            {"causality": [{"caused": ["y0", "y1"], "causing": "y2"}], "causality_kind": "lr"},
            "one caused equation",
        ),
        ({"max_state_bytes": 0}, "above 0"),
        ({"embargo": 5}, "embargo does not apply to vector_ar"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.spec.vector_ar("v", targets=_names(3), **{"lags": 2, "half_life": 50.0, **kw})


def test_the_reads_refuse_by_name():
    df = _var(300, 2, 1, 17, exog=True)
    spec = rp.spec.vector_ar("v", targets=_names(2), features=["x"], lags=1, half_life=INF)
    bank = rp.ModelBank([spec])
    assert bank.vector_ar(table="irf").height == 0
    bank.fit(df.head(4))
    with pytest.raises(ValueError, match="cannot be read, since its rows do not identify"):
        bank.vector_ar(table="irf")
    bank.fit(df.slice(4))
    with pytest.raises(ValueError, match="depends on its features"):
        bank.vector_ar(table="mean")
    with pytest.raises(ValueError, match="needs caused"):
        bank.vector_ar(table="causality")
    with pytest.raises(ValueError, match="table must be"):
        bank.vector_ar(table="nope")  # type: ignore[arg-type]
    other = rp.ModelBank([rp.spec.ewridge("v", targets=["y0"], features=["x"], half_life=INF)])
    with pytest.raises(ValueError, match="no vector_ar spec"):
        other.vector_ar()


@pytest.mark.extended
def test_the_f_test_holds_its_size_under_decay():
    """Calibration: a VAR(1) of two series with unit noise and no causation
    from ``y1`` to ``y0``, 1,000 groups of 2,000 rows, the test read at each
    group's last row. At ``λ = 0.99`` (an effective 199 rows) the F test at
    5% rejects within three Monte Carlo standard errors (0.7% each) of 5%;
    at ``λ = 0.95`` (an effective 39) it is conservative, at most 5%."""
    groups, n = 1000, 2000
    rng = np.random.default_rng(18)
    a = np.array([[0.5, 0.0], [0.3, 0.4]])
    y = np.zeros((groups, n, 2))
    e = rng.standard_normal((groups, n, 2))
    for t in range(1, n):
        y[:, t] = y[:, t - 1] @ a.T + e[:, t]
    df = pl.DataFrame(
        {
            "g": np.repeat(np.arange(groups), n),
            "y0": y[:, :, 0].ravel(),
            "y1": y[:, :, 1].ravel(),
        }
    )
    for lam, low, high in ((0.99, 0.029, 0.071), (0.95, 0.0, 0.05)):
        spec = rp.spec.vector_ar(
            "v",
            targets=["y0", "y1"],
            lags=1,
            causality=[{"caused": "y0", "causing": "y1"}],
            group="g",
            lam=lam,
            max_rows_between_solves=1,
        )
        out = rp.ModelBank([spec]).fit_predict(df).unnest("v")
        last = out.with_columns(pl.Series("g", df["g"])).group_by("g").last()
        rate = float((last["causality0_p"] < 0.05).mean())
        assert low <= rate <= high, (lam, rate)
