"""The jump tests of ``realized`` against R's: Barndorff-Nielsen and
Shephard's adjusted ratio test against ``highfrequency``'s ``BNSjumpTest``
and ``yuima``'s ``bns.test``, and Lee and Mykland's statistic and flag
against ``highfrequency``'s ``intradayJumpTest`` (docs/PLAN.md task 499),
through rpy2, selected by ``-m r_oracle`` alone (the ``r-oracles`` job).

The conventions, read in each source, so the comparison is of the
arithmetic (the jump study's record states each):

- ``BNSjumpTest(type = "ratio", max = TRUE, IQestimator = "TP")`` is the
  paper's adjusted ratio ``J`` with ``highfrequency``'s tripower
  quarticity, which carries ``N / (N - 2)``; ``realized``'s carries none.
- ``bns.test(r = rep(4/3, 3), type = "ratio", adj = TRUE)`` is ``J`` with the
  factor in front ``sqrt`` of the number of prices, ``n + 1``, where the
  paper has the ``n`` returns: ``sqrt((n + 1)/n)`` times the paper's.
- ``intradayJumpTest(dontIncludeLast = TRUE)`` takes each return's
  volatility from the ``K - 1`` returns before it, the paper's window, as
  ``lm_window_rows = K`` does, and reports ``c L`` (``c = sqrt(2/π)``); its
  critical value at ``alpha = 0.05`` is ``c (C_N + S_N β*)`` with
  ``β* = -ln(-ln 0.95)`` and ``N`` the number of prices, ``n + 1``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

import reference_realized as ref
from rheon import polars as rp

TIER = "essential"

pytestmark = pytest.mark.r_oracle

hf = importr("highfrequency")
C = math.sqrt(2 / math.pi)

_bns_hf = ro.r(
    "function(x) highfrequency::BNSjumpTest(x, type = 'ratio', max = TRUE,"
    " IQestimator = 'TP')$ztest"
)
_bns_yuima = ro.r(
    "function(lp) yuima::bns.test(yuima::setData(zoo::zoo(lp)), r = rep(4/3, 3),"
    " type = 'ratio', adj = TRUE)[[1]]$statistic"
)
_lm_hf = ro.r(
    """function(p, step, by, per, K) {
      idx <- as.POSIXct("2024-01-02 09:30:00", tz = "UTC") + (0:(length(p) - 1)) * step
      x <- xts::xts(p, order.by = idx); colnames(x) <- "PRICE"
      o <- highfrequency::intradayJumpTest(x, volEstimator = "RM", alignBy = by,
        alignPeriod = per, marketOpen = "09:30:00", marketClose = "16:00:00", tz = "UTC",
        RM = "rBPCov", lookBackPeriod = K, dontIncludeLast = TRUE, alpha = 0.05)
      list(z = as.numeric(o$ztest), crit = o$criticalValue)}"""
)


def _prices_and_returns(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prices from 100, their logs and returns by libm's ``log``, as R takes
    them: numpy's vectorised ``log`` differs in the last bit at a few prices,
    which a tiny return amplifies."""
    p = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    lp = np.array([math.log(x) for x in p])
    return p, lp, np.diff(lp)


def _fit(r: np.ndarray, **kw) -> pl.DataFrame:
    df = pl.DataFrame({"r": [*map(float, r), None]})
    spec = rp.spec.realized("v", targets=["r"], half_life=float("inf"), **kw)
    return rp.ModelBank([spec]).fit_predict(df).unnest("v")


def _cond_gap(got: float, want: float, rv: float, bv: float) -> float:
    cond = (rv + bv) / abs(rv - bv)
    return abs(got - want) / (cond * np.finfo(float).eps * max(abs(want), 1.0))


def _days() -> list[np.ndarray]:
    """Stochastic-volatility days with and without jumps, and days of constant
    volatility, whose ``TQ/BV²`` falls on either side of 1."""
    days = [ref.sv_returns(n, 40 + n, j) for n in (390, 2_340, 23_400) for j in (True, False)]
    for seed in range(8):
        days.append(0.001 * np.random.default_rng(100 + seed).standard_normal(390))
    return days


def test_the_ratio_test_is_highfrequencys_and_yuimas():
    """On every day the adjusted ratio test, from ``realized``'s own RV, BV
    and TQ, is ``BNSjumpTest``'s with its ``N/(N-2)`` on TQ applied, and
    ``bns.test``'s over ``sqrt((n+1)/n)``, to the conditioning of the
    difference; where the ``max`` takes 1 in both, ``BNSjumpTest`` is
    ``realized``'s ``bns`` itself. Both sides of the ``max`` and of the 5%
    test's 1.645 are met; TQ is ``rTPQuar`` over ``N/(N-2)``."""
    sides = {"max takes 1": 0, "max takes TQ/BV²": 0, "rejects": 0, "does not": 0}
    for r in _days():
        _, lp, rr = _prices_and_returns(r)
        n = len(rr)
        got = _fit(rr).row(n, named=True)
        rv, bv, tq, z = got["rv_r"], got["bv_r"], got["tq_r"], got["bns_r"]
        factor = n / (n - 2)
        tq_hf = float(np.asarray(hf.rTPQuar(ro.r.matrix(ro.FloatVector(rr), ncol=1)))[0])
        assert abs(tq - tq_hf / factor) <= 64 * np.spacing(tq), (n, tq, tq_hf)
        q_hf = tq * factor / bv**2
        want_hf = math.sqrt(n) * (1 - bv / rv) / math.sqrt(ref.THETA * max(1.0, q_hf))
        z_hf = float(_bns_hf(ro.FloatVector(rr))[0])
        assert _cond_gap(want_hf, z_hf, rv, bv) <= 64, (n, want_hf, z_hf)
        if q_hf <= 1:
            assert _cond_gap(z, z_hf, rv, bv) <= 64, (n, z, z_hf)
            sides["max takes 1"] += 1
        else:
            sides["max takes TQ/BV²"] += 1
        z_yu = float(_bns_yuima(ro.FloatVector(lp))[0])
        assert _cond_gap(z * math.sqrt((n + 1) / n), z_yu, rv, bv) <= 64, (n, z, z_yu)
        if abs(z - 1.645) > 0.01:
            assert (z > 1.645) == (z_hf > 1.645) == (z_yu > 1.645), (n, z, z_hf, z_yu)
            sides["rejects" if z > 1.645 else "does not"] += 1
    assert all(v > 0 for v in sides.values()), sides


@pytest.mark.parametrize(
    ("n", "step", "by", "per", "k"),
    [
        (390, 60, "minutes", 1, 10),
        (390, 60, "minutes", 1, 30),
        (2_340, 10, "seconds", 10, 30),
        (2_340, 10, "seconds", 10, 100),
    ],
)
def test_lee_myklands_statistic_and_flag_are_intradayjumptests(n, step, by, per, k):
    """``lm`` with ``lm_window_rows = K`` is ``intradayJumpTest``'s statistic over
    ``c`` to a few units in the last place on every return from the ``K``th;
    its flags at ``alpha = 0.05`` are the rows where ``c |lm|`` passes its
    critical value, which is the paper's for ``N = n + 1`` prices and
    ``β = 0.95``; the planted jumps are flagged and most rows are not, so
    both sides of the threshold are met. ``realized``'s own flag at
    ``lm_confidence = 0.95`` reads its own ``n``, the returns before the row, and
    agrees with ``intradayJumpTest``'s wherever the two thresholds do."""
    r = ref.sv_returns(n, 7 + k, True)
    p, _, rr = _prices_and_returns(r)
    out = _lm_hf(ro.FloatVector(p), step, by, per, k)
    z, crit = np.asarray(out[0]), float(out[1][0])
    assert abs(crit - C * ref.critical(n + 1, 0.95)) <= 1e-12 * crit
    got = _fit(rr, lm_window_rows=k, lm_confidence=0.95)
    lm = got["lm_r"].to_numpy()[:n]
    flag = got["lm_jump_r"].to_list()[:n]
    gaps = np.abs(C * lm[k:] - z[k:]) / np.spacing(np.abs(z[k:]))
    assert gaps.max() <= 8, gaps.max()
    hf_flags = set(np.flatnonzero(np.abs(z) > crit))
    ours = {i for i in range(k, n) if C * abs(lm[i]) > crit}
    assert ours == hf_flags
    assert hf_flags and len(hf_flags) < n // 20, sorted(hf_flags)
    for at in (0.3, 0.7):
        assert int(at * n) in hf_flags, at
    for i in range(k, n):
        own = C * ref.critical(i, 0.95)
        if (C * abs(lm[i]) > own) == (i in hf_flags):
            assert flag[i] == (i in hf_flags), i
