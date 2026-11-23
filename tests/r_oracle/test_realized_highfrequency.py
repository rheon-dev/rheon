"""R's ``highfrequency`` (CRAN 1.0.3, GPL >= 2), the field's reference
implementation of the realized measures, as the oracle of ``realized``'s
(docs/PLAN.md task 499), through rpy2. Selected by ``-m r_oracle`` alone,
which the ``r-oracles`` job of ``.github/workflows/oracles.yml`` runs with R
and its packages installed (pyproject.toml's ``r-oracles`` group).

Its conventions, read in its source and stated here so the comparison is of
the arithmetic: ``rRVar`` and ``rBPCov`` apply no factor; ``rMPVar(m=3,
p=2)``, the tripower variation, and ``rTPQuar``, the tripower quarticity,
apply ``N / (N - 2)``, which the papers do not, so it is divided out.
"""

from __future__ import annotations

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


def _r(f, r: np.ndarray, **kw) -> float:
    return float(np.asarray(f(ro.r.matrix(ro.FloatVector(r), ncol=1), **kw)).ravel()[0])


def _whole_day(r: np.ndarray) -> dict:
    df = pl.DataFrame({"r": [*map(float, r), None]})
    spec = rp.spec.realized("v", targets=["r"], half_life=float("inf"))
    return rp.ModelBank([spec]).fit_predict(df).unnest("v").row(len(r), named=True)


def _ulps(got: float, want: float) -> float:
    return abs(got - want) / np.spacing(abs(want))


DAYS = [(n, jumps) for n in (390, 2_340, 23_400) for jumps in (True, False)]


@pytest.mark.parametrize(("n", "jumps"), DAYS)
def test_the_measures_are_highfrequencys(n, jumps):
    """RV and BV to a few units in the last place; TV and TQ, whose 4/3 and
    2/3 powers and constants each side computes its own way, to 64."""
    r = ref.sv_returns(n, 5 + n, jumps)
    got = _whole_day(r)
    factor = n / (n - 2)
    assert _ulps(got["rv_r"], _r(hf.rRVar, r)) <= 4
    assert _ulps(got["bv_r"], _r(hf.rBPCov, r)) <= 8
    assert _ulps(got["tv_r"], _r(hf.rMPVar, r, m=3, p=2) / factor) <= 64
    assert _ulps(got["tq_r"], _r(hf.rTPQuar, r) / factor) <= 64
