"""``huber``'s feature sets, its sum-scale ridge and its solve schedule, held
to a library on a contaminated stream.

``tests/test_huber_options.py`` holds each option to a model of this library
that already has the property (``ewridge`` fed the rows at the Huber weights,
``rls`` at an infinite cut), and ``tests/convergence.py`` holds a plain
``huber`` to statsmodels' ``RLM`` at its limit. Neither reaches the options on
a stream with outliers against a library's answer: this does. Each reads the
model's one-pass fit on a stationary stream of 8,000 rows, 6% of them
contaminated by a one-sided shift of 8 standard deviations, and holds it to
what a library fits on the same rows:

- a smaller feature set is a weighted least-squares fit of its features at the
  Huber weights of the largest set's converged ``RLM`` fit (statsmodels);
- the sum-scale ridge is a ridge on every coefficient, the intercept's
  included, in the sums' own units (scikit-learn's ``Ridge`` with no intercept
  of its own, at the converged weights, the loop of ``convergence.huber_oracle``
  around it);
- under a solve schedule of ``k`` rows the coefficients hold for ``k`` rows and
  each block's are the batch fit of the rows before it: ``RLM`` for ``huber``,
  ``QuantReg`` for ``quantile``. The blocks are read off the predictions
  (inside a block they are one linear function of the features, to rounding),
  not assumed.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from statsmodels.robust.norms import HuberT

from rheon import polars as rp

TIER = "essential"

FEATURES = ["x0", "x1", "x2"]
BETA = np.array([2.0, -1.0, 0.5])
LEVEL = 50.0
Q3 = 0.6744897501960817  # Phi^-1(3/4): the MAD's divisor


def _rows(n: int = 8000, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """A line with noise of 0.4 and one-sided outliers on 6% of the rows."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    y = LEVEL + x @ BETA + 0.4 * rng.standard_normal(n)
    y[rng.random(n) < 0.06] += 3.2
    return x, y


def _frame(x: np.ndarray, y: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "y": y})


def _pred(spec: dict, df: pl.DataFrame, name: str = "pred_y") -> np.ndarray:
    out = rp.ModelBank([spec]).fit_predict(df)
    return out["m"].struct.field(name).to_numpy().astype(float)


def _design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def _huber_weights(z: np.ndarray, y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``HuberT``'s weights of the residuals over the median-of-``|r|`` scale
    the model's cut uses (``convergence.huber_oracle``'s)."""
    r = y - z @ b
    return np.asarray(HuberT(1.345).weights(r / (np.median(np.abs(r)) / Q3)))


class TestFeatureSets:
    def test_a_smaller_set_is_wls_at_the_largest_sets_converged_weights(self):
        x, y = _rows()
        n, tail = len(y), 200
        sets = {"one": ["x1"], "all": FEATURES}
        spec = rp.spec.huber(
            "m",
            targets=["y"],
            features=FEATURES,
            feature_sets=sets,
            half_life=float("inf"),
            min_weight=5.0,
        )
        out = rp.ModelBank([spec]).fit_predict(_frame(x, y))["m"].struct
        one = out.field("pred_y__one").to_numpy().astype(float)[-tail:]
        full = out.field("pred_y__all").to_numpy().astype(float)[-tail:]
        # The largest set is the batch Huber fit (RLM), up to the one pass:
        # the rows before the last 200 give the fit that predicts them.
        z, yy = _design(x[: n - tail]), y[: n - tail]
        zt = _design(x[-tail:])
        rlm = sm.RLM(yy, z, M=HuberT(1.345)).fit()
        assert np.max(np.abs(full - zt @ np.asarray(rlm.params))) < 0.02
        # A smaller set is a weighted fit of its own columns at those weights.
        w = _huber_weights(z, yy, np.asarray(rlm.params))
        assert (w < 1.0).sum() > 100, "the cut must bind"
        small = sm.WLS(yy, z[:, [0, 2]], weights=w).fit().params
        want = zt[:, [0, 2]] @ np.asarray(small)
        assert np.max(np.abs(one - want)) < 0.02, np.max(np.abs(one - want))
        # And it is neither the largest set's fit nor the unweighted fit of x1.
        assert np.mean(np.abs(one - full)) > 0.5
        plain = sm.OLS(yy, z[:, [0, 2]]).fit().params
        assert np.mean(np.abs(one - zt[:, [0, 2]] @ np.asarray(plain))) > 0.05


class TestSumScaleRidge:
    def test_it_is_a_ridge_on_every_coefficient_the_intercept_included(self):
        x, y = _rows()
        n = len(y)
        ridge = 400.0
        spec = rp.spec.huber(
            "m",
            targets=["y"],
            features=FEATURES,
            ridge=ridge,
            ridge_scale="sum",
            half_life=float("inf"),
            min_weight=5.0,
        )
        got = _pred(spec, _frame(x, y))
        # The oracle: the loop of `convergence.huber_oracle` (a weighted-MAD
        # scale, `HuberT`'s weights) around scikit-learn's ridge on the
        # design that holds the intercept's column, penalized like the rest.
        z, yy = _design(x[: n - 1]), y[: n - 1]
        v = np.ones(n - 1)
        b = Ridge(alpha=ridge, fit_intercept=False).fit(z, yy).coef_
        for _ in range(200):
            r = yy - z @ b
            scale = np.median(np.abs(r)) / Q3
            nxt = (
                Ridge(alpha=ridge, fit_intercept=False)
                .fit(z, yy, sample_weight=v * HuberT(1.345).weights(r / scale))
                .coef_
            )
            if np.max(np.abs(nxt - b)) <= 1e-11 * (1 + np.max(np.abs(nxt))):
                b = nxt
                break
            b = nxt
        want = float((_design(x[-1:]) @ b)[0])
        # The prior reaches the intercept: 400 against 8,000 rows pulls a level
        # of 50 down by about 2.4, far past the tolerance.
        mean_form = _pred(
            rp.spec.huber(
                "m",
                targets=["y"],
                features=FEATURES,
                ridge=ridge,
                half_life=float("inf"),
                min_weight=5.0,
            ),
            _frame(x, y),
        )
        assert abs(mean_form[-1] - want) > 1.0, "the two forms must part here"
        assert abs(got[-1] - want) < 0.1, (got[-1], want)


def _blocks(
    pred: np.ndarray, x: np.ndarray, k: int, lo: int, hi: int
) -> list[tuple[int, np.ndarray]]:
    """The solve schedule read off the predictions: the offset ``o`` at which
    every block ``[o + k m, o + k (m + 1))`` inside ``[lo, hi)`` is one linear
    function of the features, to rounding (no other offset makes them all so),
    and each block's first row with its coefficients."""
    z = _design(x)
    found = []
    for o in range(k):
        blocks = []
        ok = True
        start = lo + (o - lo) % k
        while start + k <= hi:
            seg = slice(start, start + k)
            b = np.linalg.lstsq(z[seg], pred[seg], rcond=None)[0]
            if np.max(np.abs(z[seg] @ b - pred[seg])) > 1e-7:
                ok = False
                break
            blocks.append((start, b))
            start += k
        if ok:
            found.append(blocks)
    assert len(found) == 1, f"{len(found)} offsets give blocks of {k} rows"
    return found[0]


class TestTheSolveSchedule:
    K = 100

    def test_huber_holds_each_blocks_coefficients_to_rlm_on_the_rows_before_it(self):
        x, y = _rows()
        spec = rp.spec.huber(
            "m",
            targets=["y"],
            features=FEATURES,
            half_life=float("inf"),
            solve_every=float("inf"),
            max_rows_between_solves=self.K,
            min_weight=5.0,
        )
        pred = _pred(spec, _frame(x, y))
        blocks = _blocks(pred, x, self.K, 7000, 8000)
        assert len(blocks) >= 9
        # Successive blocks have their own coefficients: a model that solved
        # on every row would give a line that moves with every row, and a
        # frozen one the same line throughout.
        assert not np.allclose(blocks[0][1], blocks[1][1], atol=1e-9)
        for start, b in blocks[::4]:
            rlm = sm.RLM(y[:start], _design(x[:start]), M=HuberT(1.345)).fit()
            assert np.max(np.abs(b - np.asarray(rlm.params))) < 0.01, (start, b, rlm.params)

    def test_quantile_holds_each_blocks_coefficients_to_quantreg_on_the_rows_before_it(self):
        rng = np.random.default_rng(12)
        n = 8000
        x = rng.standard_normal((n, 3))
        # Skewed noise, so that the 0.75 quantile's line and the mean's part.
        y = LEVEL + x @ BETA + rng.exponential(1.0, n)
        spec = rp.spec.quantile(
            "m",
            targets=["y"],
            features=FEATURES,
            quantile=0.75,
            half_life=float("inf"),
            solve_every=float("inf"),
            max_rows_between_solves=self.K,
            min_weight=5.0,
        )
        pred = _pred(spec, _frame(x, y))
        blocks = _blocks(pred, x, self.K, 7000, 8000)
        assert len(blocks) >= 9
        assert not np.allclose(blocks[0][1], blocks[1][1], atol=1e-9)
        for start, b in blocks[::4]:
            want = np.asarray(sm.QuantReg(y[:start], _design(x[:start])).fit(q=0.75).params)
            assert np.max(np.abs(b - want)) < 0.03, (start, b, want)
        # The 0.75 quantile of Exp(1) is ln 4 = 1.386 above the mean's 1.
        assert abs(blocks[-1][1][0] - (LEVEL + np.log(4.0))) < 0.15
        # The control: the least-squares line is not any block's.
        ols = np.linalg.lstsq(_design(x[: blocks[-1][0]]), y[: blocks[-1][0]], rcond=None)[0]
        assert np.max(np.abs(blocks[-1][1] - ols)) > 0.2
