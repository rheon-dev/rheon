"""The refits over a span of the clock, ``huber_refit_span`` (``huber`` and
``sgd(loss="huber")``) and ``quantile_refit_span``, against the batch fit
of the rows inside the span.

A span refit fires on the first row at least the span after a target's
first learned row, and fits the rows from the first through it. On an
irregular clock, the span set to that row's distance exactly, the stream
ending on that row, the fit read after it is the refit's: statsmodels'
converged ``RLM`` (``HuberT(1.345)``, the MAD scale) for the Huber refit,
and statsmodels' ``QuantReg`` for the quantile refit, each on those rows
alone. The span one ulp longer does not reach the row, so no refit has run
and the one-pass fit, far from either, is read: the two sides of the
span's edge.

The tolerances are the row-count refits', in ``tests/test_huber_refit.py``
and ``tests/test_quantile_refit.py``: the Huber refit to the ridge's size
(1e-5), the quantile refit within the band's smoothing of the check loss
(0.03).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

from rheon import polars as rp
from test_huber_scale import stream

TIER = "essential"
INF = float("inf")


def _clock(n: int, seed: int) -> np.ndarray:
    """An irregular clock: steps drawn from 0.3 to 2.7 units."""
    return np.cumsum(np.random.default_rng([577, seed]).uniform(0.3, 2.7, n))


def _coef(spec: dict, df: pl.DataFrame) -> np.ndarray:
    bank = rp.ModelBank([spec])
    bank.fit_predict(df)
    return np.array(bank.coef("m")["coef"].to_list())


def _edges(t: np.ndarray, row: int) -> tuple[float, float]:
    """The span that reaches ``row`` exactly, as the clock's own
    subtraction gives it, and the next float above it, which does not."""
    at = float(t[row] - t[0])
    return at, float(np.nextafter(at, INF))


@pytest.mark.parametrize("kind", ["huber", "sgd"])
@pytest.mark.parametrize(("share", "seed"), [(0.1, 0), (0.25, 1)])
def test_the_huber_span_refit_is_rlm_on_the_rows_inside_the_span(kind, share, seed):
    """The rows from the first through the first at least the span after
    it (an odd count, 301, where the weighted median is ``np.median``):
    the fit read after that row is ``RLM``'s on them, within 1e-5; with the
    span one ulp longer, the one-pass fit, more than 1e-3 off."""
    edge = 300
    df = stream(edge + 1, share, seed, level=1000.0)
    t = _clock(edge + 1, seed)
    df = df.with_columns(pl.Series("t", t))
    X = sm.add_constant(df["x"].to_numpy())
    oracle = (
        sm.RLM(df["y"].to_numpy(), X, M=HuberT(1.345))
        .fit(scale_est="mad", conv="coefs", tol=1e-12, maxiter=200)
        .params
    )
    at, past = _edges(t, edge)
    c = dict(targets=["y"], features=["x"], half_life=INF, clock="t", gap_cap=10.0)

    def model(span: float) -> dict:
        if kind == "huber":
            return rp.spec.huber("m", huber_refit_span=span, **c)
        return rp.spec.sgd("m", loss="huber", huber_refit_span=span, **c)

    refit, plain = _coef(model(at), df), _coef(model(past), df)
    assert np.abs(refit - oracle).max() < 1e-5, (refit, oracle)
    assert np.abs(plain - oracle).max() > 1e-3, (plain, oracle)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_quantile_span_refit_is_quantreg_on_the_rows_inside_the_span(seed):
    """``y = 1 + 2x − z + 0.5 Laplace``, its first five targets 1e4 off
    (``tests/test_quantile_refit.py``'s stream), on an irregular clock:
    the fit read after the span's edge row is ``QuantReg``'s on the rows
    through it, within 0.03; with the span one ulp longer, the one pass,
    its warm-up's least squares still in its sums, five times as far."""
    edge = 399
    rng = np.random.default_rng(seed)
    n = edge + 1
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    y = 1.0 + 2.0 * x - z + 0.5 * rng.laplace(size=n)
    y[:5] += 1e4
    t = _clock(n, 10 + seed)
    df = pl.DataFrame({"t": t, "x": x, "z": z, "y": y})
    X = sm.add_constant(np.column_stack([x, z]))
    oracle = np.asarray(sm.QuantReg(y, X).fit(q=0.5, p_tol=1e-10).params)
    at, past = _edges(t, edge)
    c = dict(
        targets=["y"], features=["x", "z"], half_life=INF, quantile=0.5, solve_every=0.0,
        clock="t", gap_cap=10.0,
    )  # fmt: skip
    refit = _coef(rp.spec.quantile("m", quantile_refit_span=at, **c), df)
    one_pass = _coef(rp.spec.quantile("m", quantile_refit_span=past, **c), df)
    gap, before = np.max(np.abs(refit - oracle)), np.max(np.abs(one_pass - oracle))
    assert gap < 0.03, gap
    assert before > 5 * gap, (before, gap)
