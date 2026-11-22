"""``ew_cov``'s moments at a nonzero correlation, on a weighted, gapped stream,
held to numpy.

``tests/test_second_opinion.py`` holds the same moments to pandas' ``ewm`` on
two independent features, where a covariance and a correlation cross 0; this
file reads them where they do not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"


class TestTheEwMomentsAreNumpyOnAWeightedGappedStream:
    """The moments of two features that are correlated, on a stream with row
    weights, zero-weight rows and gaps in the clock, are ``numpy``'s weighted
    mean, covariance and correlation of the rows before the read, each at
    ``w * 0.5 ** (age / h)`` with the age the sum of the clock's steps since
    the row, each capped at ``gap_cap``. The pandas comparison above draws two
    independent features (a true correlation of 0, so ``cov`` and ``corr``
    cross 0 and a wrong cross term is a small number there); here the
    correlation is 0.6, the weights vary by a factor of 15 and two gaps are
    long, so a wrong weight in any one accumulator, a cap that does not
    apply, or a clock step read after the row moves a number of order 1."""

    H = 25.0

    @pytest.mark.parametrize("cap", [1e9, 5.0])
    @pytest.mark.parametrize("bias", [False, True])
    def test_mean_var_cov_corr_are_numpy_of_the_weighted_rows(self, cap, bias):
        rng = np.random.default_rng(21)
        n = 300
        x0 = 3.0 + rng.normal(0.0, 1.0, n)
        x1 = -2.0 + 0.6 * (x0 - 3.0) + 0.8 * rng.normal(0.0, 1.0, n)
        w = rng.uniform(0.2, 3.0, n)
        w[[150, 151, 210]] = 0.0
        dt = rng.uniform(0.2, 3.0, n)
        dt[0] = 0.0
        dt[[120, 200]] = [40.0, 15.0]
        t = np.cumsum(dt)
        frame = pl.DataFrame({"t": t, "w": w, "x0": x0, "x1": x1})
        spec = rp.spec.ew_cov(
            "m",
            features=["x0", "x1"],
            half_life=self.H,
            clock="t",
            weight="w",
            gap_cap=cap,
            stats=["mean", "var", "cov", "corr"],
            min_weight=0.0,
            bias=bias,
        )
        out = rp.ModelBank([spec]).fit_predict(frame)["m"].struct.unnest()
        x = np.column_stack([x0, x1])
        # A row of weight 0 is skipped: the step it would have taken joins
        # the next learned row's, and the cap applies to the joined step.
        learned = np.flatnonzero(w > 0.0)
        step = np.minimum(np.diff(t[learned], prepend=t[learned[0]]), cap)
        seen_corr = []
        for i in range(60, n, 7):
            # The read on row i is the state after the learned rows before
            # it, aged to the last of them.
            k = learned[learned < i]
            steps = step[: len(k)]
            age = np.array([steps[m + 1 :].sum() for m in range(len(k))])
            weights = w[k] * 0.5 ** (age / self.H)
            rows = x[k]
            mean = np.average(rows, axis=0, weights=weights)
            cov = np.cov(rows.T, aweights=weights, ddof=0 if bias else 1)
            corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            seen_corr.append(corr)
            np.testing.assert_allclose(
                [out["mean_x0"][i], out["mean_x1"][i]], mean, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                [out["var_x0"][i], out["var_x1"][i], out["cov_x0_x1"][i]],
                [cov[0, 0], cov[1, 1], cov[0, 1]],
                rtol=1e-10,
                atol=1e-12,
            )
            assert out["corr_x0_x1"][i] == pytest.approx(corr, rel=1e-10, abs=1e-12)
        # The stream has a correlation to get wrong: well away from 0.
        assert 0.4 < float(np.mean(seen_corr)) < 0.8
