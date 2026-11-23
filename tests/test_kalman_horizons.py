"""``kalman``'s forecasts at horizons on the clock (task 559): the prediction a
row of this row's features would get ``Δ`` past the row with no row between,
its integral over the span and its variance, as the filter would make them.

``filterpy``'s ``KalmanFilter`` is the oracle, run on ``kalman``'s documented
recursion in the form a test can write without the standardizer: features
in their own units (``standardize=False``), a given noise ``obs_var`` and a
given process noise ``q``, so ``P`` is absolute and starts at ``p0 R I``.
Every row runs ``predict`` with ``F = diag(2^(-d / r))`` and no noise; a row
that observes the target adds ``q g(D)²`` for the clock ``D`` since the last
such row (``g(D) = (1 - 2^(-D / r)) / θ``, ``D`` itself at ``r = inf``) and
then ``update`` with ``R = obs_var / w``. The forecast on row ``t`` is
``filterpy``'s ``predict`` from the state after row ``t``'s transition, with
``F = diag(2^(-Δ / r))`` and the noise of ``D + Δ``: the mean ``z' x``, the
variance ``z' P z + obs_var``. Its integral is scipy's ``quad`` of that mean
as a function of the span.
"""

from __future__ import annotations

import copy
import io
import math

import filterpy.kalman as fk
import numpy as np
import polars as pl
import pytest
from scipy import integrate

from rheon import polars as rp

TIER = "essential"

R = 0.25
Q = [1e-4, 4e-4, 1e-4]
REVERT = [math.inf, 30.0, 12.0]
SPANS = [0.0, 0.5, 3.0, 20.0]


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.3, 1.7, n))
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(1, 0.5, n)
    y = 0.5 + 1.2 * x0 - 0.7 * x1 + rng.normal(0, math.sqrt(R), n)
    w = np.where(rng.uniform(size=n) < 0.1, 0.0, 1.0)
    return pl.DataFrame({"t": t, "x0": x0, "x1": x1, "y": y, "w": w})


def _spec(**kw) -> dict:
    base: dict = dict(
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=100.0,
        half_life=200.0,
        q=Q,
        obs_var=R,
        p0=1.0,
        revert_half_life=REVERT,
        standardize=False,
        min_weight=0.0,
        coef_every=0,
        horizons=SPANS,
    )
    base.update(kw)
    return rp.spec.kalman("m", **base)


def _g(d: float, r: float) -> float:
    if math.isinf(r):
        return d
    theta = math.log(2.0) / r
    return -math.expm1(-theta * d) / theta


def _phi(d: float) -> np.ndarray:
    return np.diag([2.0 ** (-d / r) if math.isfinite(r) else 1.0 for r in REVERT])


def _oracle(df: pl.DataFrame, rows: list[int]):
    """filterpy's run, and at each of `rows` the forecasts at `SPANS`."""
    t, w, y = df["t"].to_numpy(), df["w"].to_numpy(), df["y"].to_numpy()
    Z = np.column_stack([np.ones(len(df)), df["x0"].to_numpy(), df["x1"].to_numpy()])
    kf = fk.KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.zeros((3, 1))
    kf.P = 1.0 * R * np.eye(3)
    gap, prev = 0.0, None
    out = {}
    for i in range(len(df)):
        d = 0.0 if prev is None else t[i] - prev
        prev = t[i]
        kf.predict(F=_phi(d), Q=np.zeros((3, 3)))
        gap += d
        z = Z[i]
        if i in rows:
            got = []
            for s in SPANS:
                f = copy.deepcopy(kf)
                q = np.diag([qi * _g(gap + s, r) ** 2 for qi, r in zip(Q, REVERT, strict=True)])
                f.predict(F=_phi(s), Q=q)
                mean = float(z @ f.x[:, 0])
                var = float(z @ f.P @ z) + R
                x_now = kf.x[:, 0].copy()

                def at(u: float, x_now: np.ndarray = x_now, z: np.ndarray = z) -> float:
                    return float(z @ (_phi(u) @ x_now))

                area = integrate.quad(at, 0.0, s, epsabs=0, epsrel=1e-13)[0] if s else 0.0
                got.append((mean, area, var))
            out[i] = got
        if w[i] > 0:
            q = np.diag([qi * _g(gap, r) ** 2 for qi, r in zip(Q, REVERT, strict=True)])
            kf.P = kf.P + q
            gap = 0.0
            kf.update(np.array([[y[i]]]), R=R / w[i], H=z.reshape(1, 3))
    return out


def test_the_forecasts_are_filterpys_and_their_integral_scipys():
    df = _frame(400, 1)
    rows = [60, 200, 333, 399]
    out = rp.ModelBank([_spec(weight="w")]).fit_predict(df).unnest("m")
    want = _oracle(df, rows)
    for r in rows:
        for s, (mean, area, var) in zip(SPANS, want[r], strict=True):
            label = f"h{s:g}"
            assert out[f"forecast_{label}_y"][r] == pytest.approx(mean, rel=1e-13, abs=1e-13)
            assert out[f"forecast_sum_{label}_y"][r] == pytest.approx(area, rel=1e-13, abs=1e-13)
            assert out[f"forecast_var_{label}_y"][r] == pytest.approx(var, rel=1e-13)
        assert out["forecast_h0_y"][r] == out["pred_y"][r]


def test_chunk_invariance_and_a_save_go_on_to_the_bit():
    df = _frame(500, 2).with_columns(g=pl.int_range(500) % 2)
    for spec in (
        _spec(group="g"),
        _spec(group="g", standardize=True, q=None, obs_var=None, coef_half_life=40.0),
    ):
        one = rp.ModelBank([spec]).fit_predict(df)
        for size in (1, 17, 211):
            bank = rp.ModelBank([spec])
            many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 500, size)])
            assert one.equals(many, null_equal=True), size
        for cut in (1, 250, 499):
            a = rp.ModelBank([spec])
            a.fit_predict(df.slice(0, cut))
            buf = io.BytesIO()
            a.save(buf)
            b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[spec])
            rest = df.slice(cut)
            assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


def test_a_row_of_weight_zero_is_as_if_absent():
    """A row of weight 0 learns nothing and counts in no clock: the learned
    rows' forecasts are those of the stream without it, to the rounding of
    the transition taken in two steps rather than one."""
    df = _frame(400, 3)
    spec = _spec(weight="w")
    full = rp.ModelBank([spec]).fit_predict(df).unnest("m")
    keep = df["w"].to_numpy() > 0
    thin = rp.ModelBank([spec]).fit_predict(df.filter(pl.Series(keep))).unnest("m")
    for f in [c for c in full.columns if c.startswith(("forecast", "pred"))]:
        a = full[f].filter(pl.Series(keep)).to_numpy()
        b = thin[f].to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f)


def test_a_forecast_reads_no_row_it_forecasts():
    df = _frame(300, 4)
    spec = _spec()
    a = rp.ModelBank([spec]).fit_predict(df).unnest("m")
    for r in (150, 260):
        moved = df.with_columns(
            pl.when(pl.int_range(300) == r).then(pl.col("y") * 5 + 7).otherwise(pl.col("y"))
        )
        b = rp.ModelBank([spec]).fit_predict(moved).unnest("m")
        for f in a.columns:
            if f.startswith("forecast"):
                assert a[f][: r + 1].equals(b[f][: r + 1], null_equal=True), (f, r)
                # the integral at 0 is 0, and with a given noise `P` reads no
                # target, so neither moves; the means do
                if not f.startswith(("forecast_sum_h0", "forecast_var")):
                    assert not a[f][r + 1 :].equals(b[f][r + 1 :], null_equal=True), (f, r)


def test_names():
    spec = _spec(horizons={"soon": 1.0, "later": 30.0})
    fields = rp.spec.output_fields(spec)
    for name in ("forecast_soon_y", "forecast_sum_later_y", "forecast_var_later_y"):
        assert name in fields
    out = rp.ModelBank([spec]).fit_predict(_frame(50, 5))
    assert [f.name for f in out.schema["m"].fields] == fields


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"horizons": [-1.0]}, "finite span of the clock of 0 or more"),
        ({"horizons": [2.0, 2]}, 'two horizons are labeled "h2"'),
        ({"horizons": {"sumx": 1.0}}, 'not begin with "sum"'),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(ValueError, match=says):
        rp.ModelBank([_spec(**kw)])


@pytest.mark.parametrize("kind", ["kalman", "holt"])
def test_a_short_delay_is_the_undelayed_run_with_horizons(kind):
    """A delay shorter than every clock step is the undelayed run, every
    field but ``coef`` to the bit, with the forecasts' slots after the
    predictions: a held row's record keeps one entry a slot in every part,
    the row inflation the conformal interval reads included (a record a
    part short scored nothing at its release)."""
    import test_label_delay as delay

    df = delay.short_frame(kind)
    kw: dict = {"clock": "t", "gap_cap": 1e9, **delay.DIAGNOSTICS, "coef_every": 0}
    kw["horizons"] = [0.0, 5.0]
    plain = delay.unnested(delay.kind_spec(kind, **kw), df)
    held = delay.unnested(delay.kind_spec(kind, embargo=delay.SHORT, **kw), df)
    assert held.columns == plain.columns
    assert any(c.startswith("forecast_") for c in plain.columns)
    for c in held.columns:
        if c in delay.AFTER_THE_UPDATE or c.startswith(delay.DRIFT):
            assert delay.same(held[c][1:], plain[c][:-1]), c
        else:
            assert delay.same(held[c], plain[c]), c
