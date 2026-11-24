"""``sarima`` through the bank, its parameters estimated: the estimate against
statsmodels' ``SARIMAX.fit`` on the same rows over fifty seeds of three
specs; and what the bank owes every model -- predictions before the row,
chunk invariance, a save mid-stream, its wait named while its start is
withheld, its ``coef`` names and its refusals.

The oracle is ``SARIMAX(y, order, seasonal_order, trend="n").fit()``, the
batch maximum-likelihood fit of every row. The core's filter at fixed
parameters is held to ``SARIMAX.filter`` in ``tests/test_sarima_filter.py``.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import polars as pl
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from rheon import polars as rp

TIER = "essential"

#: Each case: ``(order, seasonal_order, (phi, theta, Phi, Theta, s))``, the
#: truth with ``sigma2 = 1``.
SPECS = {
    "ar1_sma4": ((1, 0, 0), (0, 0, 1, 4), ([0.6], [], [], [-0.5], 4)),
    "arma11": ((1, 0, 1), (0, 0, 0, 0), ([0.5], [0.3], [], [], 0)),
    "ima11": ((0, 1, 1), (0, 0, 0, 0), ([], [-0.4], [], [], 0)),
}

#: The bound the start's gate accepts by default, ``se_bound``: each ARMA
#: coefficient's standard error at most 0.1, and ``sigma2``'s relative one.
SE_BOUND = 0.1


def _sim(spec: str, n: int, seed: int) -> np.ndarray:
    """``n`` values of the spec's process after 500 rows of burn-in: the
    ARMA part by its two polynomials, then integrated ``d`` times, then
    a level of 10 where it is integrated (``trend="n"`` reads no level of
    a stationary series)."""
    order, _, (phi, theta, sphi, stheta, s) = SPECS[spec]

    def lag(coefs, step, sign):
        out = np.zeros(len(coefs) * step + 1)
        out[0] = 1.0
        out[step::step] = sign * np.asarray(coefs)
        return out

    a = np.polymul(lag(phi, 1, -1.0)[::-1], lag(sphi, max(s, 1), -1.0)[::-1])[::-1]
    c = np.polymul(lag(theta, 1, 1.0)[::-1], lag(stheta, max(s, 1), 1.0)[::-1])[::-1]
    rng = np.random.default_rng(seed)
    burn = 500
    e = rng.standard_normal(n + burn)
    u = np.zeros(n + burn)
    for t in range(n + burn):
        acc = sum(c[j] * e[t - j] for j in range(len(c)) if t >= j and c[j])
        acc -= sum(a[i] * u[t - i] for i in range(1, len(a)) if t >= i and a[i])
        u[t] = acc
    y = u[burn:]
    for _ in range(order[1]):
        y = np.cumsum(y)
    return y + (10.0 if order[1] else 0.0)


def _spec(spec: str, **kw) -> dict:
    order, seasonal, _ = SPECS[spec]
    return rp.spec.sarima("m", targets=["y"], order=order, seasonal_order=seasonal, **kw)


def _out(spec: dict, y: np.ndarray) -> pl.DataFrame:
    return rp.ModelBank([spec]).fit_predict(pl.DataFrame({"y": y})).unnest("m")


def _sarimax(spec: str, y: np.ndarray):
    order, seasonal, _ = SPECS[spec]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y, order=order, seasonal_order=seasonal, trend="n").fit(disp=False)
    return np.asarray(res.params), np.asarray(res.bse)


@pytest.mark.parametrize("spec", sorted(SPECS))
def test_fifty_seeds_are_sarimax_fit_within_the_gates_bound(spec):
    """Over fifty seeds of 3,000 rows, the estimate after the last row is
    within the standard error the start's gate accepts of ``SARIMAX.fit``'s
    on the same rows: each ARMA coefficient within ``se_bound`` (0.1) of
    statsmodels', and ``sigma2`` within 0.1 of it relative. Measured: at
    worst 0.0052 (``ar1_sma4``), 0.0071 (``arma11``) and 0.0022 (``ima11``)
    on the coefficients, 0.0042 relative on ``sigma2``, so the bound is the
    gate's, not this test's. As a check finer than the gate's, each ARMA
    coefficient is also within one of ``SARIMAX``'s own standard errors of
    its estimate (0.35 of one at worst when measured). About 20 s, most of
    it ``SARIMAX.fit``."""
    for seed in range(50):
        y = _sim(spec, 3000, seed)
        out = _out(_spec(spec), y)
        assert out["pred_y"][-1] is not None, f"seed {seed}: the start never opened"
        ours = np.asarray(out["coef"][-1].to_list(), dtype=float).ravel()
        theirs, se = _sarimax(spec, y)
        gap = np.abs(ours[:-1] - theirs[:-1])
        rel = abs(ours[-1] / theirs[-1] - 1.0)
        what = f"seed {seed}: ours {ours.round(4)}, SARIMAX {theirs.round(4)}"
        assert gap.max() <= SE_BOUND, what
        assert rel <= SE_BOUND, what
        assert (gap / se[:-1]).max() <= 1.0, what


def test_predictions_are_out_of_sample():
    """Hard rule 2: changing row ``t``'s value moves no prediction up to
    and including row ``t``, before the start and after it."""
    y = _sim("ar1_sma4", 1200, 7)
    base = _out(_spec("ar1_sma4"), y)["pred_y"].to_numpy()
    for t in (100, 900):
        moved = y.copy()
        moved[t] += 50.0
        pred = _out(_spec("ar1_sma4"), moved)["pred_y"].to_numpy()
        np.testing.assert_array_equal(pred[: t + 1], base[: t + 1])


def test_chunks_and_a_save_mid_stream_change_nothing():
    """Hard rule 3: one chunk or many, and a save and load between two
    chunks, give the same rows to the bit, before and after the start
    (``coef`` on every row, as by default it is written on each chunk's
    last row)."""
    y = _sim("arma11", 1500, 3)
    df = pl.DataFrame({"y": y})
    spec = _spec("arma11", coef_every=0)
    whole = rp.ModelBank([spec]).fit_predict(df)
    bank = rp.ModelBank([spec])
    parts = [bank.fit_predict(df[i : min(i + 97, 700)]) for i in range(0, 700, 97)]
    buf = io.BytesIO()
    bank.save(buf)
    loaded = rp.ModelBank.load_bytes(buf.getvalue(), specs=[spec])
    parts.append(loaded.fit_predict(df[700:]))
    assert pl.concat(parts).equals(whole, null_equal=True)


def test_the_start_names_its_wait():
    """Until a target's start is accepted nothing is reported and the row
    says why: ``below_warm_rows`` while the differencing fills and the
    buffer is below the parameters, ``not_identified`` while a fit is
    withheld at its gate. Tightening ``se_bound`` past the standard error
    the rows give withholds the start to the end."""
    y = _sim("ar1_sma4", 1500, 3)
    out = _out(_spec("ar1_sma4"), y)
    opened = int(np.argmax(out["pred_y"].is_not_null().to_numpy()))
    assert 0 < opened < 1000
    before = out["withheld_reason"][:opened].cast(pl.String).to_list()
    assert before[0] == "below_warm_rows"
    assert set(before) <= {"below_warm_rows", "not_identified"}
    assert out["withheld_reason"][opened:].is_null().all()
    tight = _out(_spec("ar1_sma4", se_bound=0.02), y)
    assert tight["pred_y"].is_null().all()
    assert tight["withheld_reason"][-1] == "not_identified"


def test_coef_is_named_as_statsmodels_names_the_parameters():
    """``coef`` per target is ``SARIMAX(...).param_names``' order: the
    features, ``ar.L<i>``, ``ma.L<i>``, ``ar.S.L<s i>``, ``ma.S.L<s i>``,
    then ``sigma2``."""
    spec = rp.spec.sarima(
        "m", targets=["y"], features=["x"], order=(2, 1, 1), seasonal_order=(1, 0, 2, 12)
    )
    terms = rp.spec.coef_index(spec)["term"].to_list()
    rng = np.random.default_rng(0)
    model = SARIMAX(
        rng.standard_normal(50),
        exog=rng.standard_normal(50),
        order=(2, 1, 1),
        seasonal_order=(1, 0, 2, 12),
        trend="n",
    )
    assert terms == ["x", *model.param_names[1:]]


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"half_life": 50.0}, "half_life does not apply to sarima"),
        ({"se_bound": 0.0}, "se_bound must be finite and above 0"),
        ({"seasonal_order": (1, 0, 0, 1)}, "a season s of at least 2 slots"),
        ({"spacing": -1.0}, "spacing must be finite and above 0"),
    ],
)
def test_refusals_name_the_setting(kw, words):
    """A setting the model cannot take is refused when the bank is built,
    naming it."""
    args = {"targets": ["y"], "order": (1, 0, 0), **kw}
    with pytest.raises(ValueError, match=words):
        rp.ModelBank([rp.spec.sarima("m", **args)])
