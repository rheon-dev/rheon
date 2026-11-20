"""Second opinions at the numeric edges of the model core (review fable2):
the scales, weights and clocks inside the input bound where a kept product
or reciprocal left the doubles' range.

- ``kalman``'s square root at a target scale of 1e80 and 1e-85, against
  statsmodels' exact diffuse filter (C5);
- ``holt`` after a first row of subnormal weight, against statsmodels'
  ``Holt`` on the series without it (C7).

``sgd`` and ``pa`` after a long clock gap are held to an exact longhand of
the standardization in Rust (``crates/rheon-core/src/sgd.rs``,
``a_row_after_a_long_gap_is_standardized_as_exact_arithmetic_does``) and
through the bank in ``tests/test_standardize_under_decay.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

#: ``holt``'s half-lives and the row from which its gains are statsmodels'
#: fixed ones to the last digit read (``tests/test_second_opinion.py``,
#: ``TestHoltAcrossAMissingObservation``): ``0.5 ** (1000 / 25)`` is 1e-12.
H_LEVEL, H_TREND, SETTLED = 6.0, 25.0, 1000


def _rising(n: int) -> np.ndarray:
    """A rising trend with a random walk about it."""
    rng = np.random.default_rng(5)
    return 3.0 + 0.4 * np.arange(n) + np.cumsum(rng.normal(0.0, 0.5, n))


@pytest.mark.parametrize("scale", [1.0, 2.0**266, 2.0**-282], ids=["unit", "1e80", "1e-85"])
def test_kalmans_square_root_is_statsmodels_exact_diffuse_filter_at_any_scale(scale):
    """The square root against statsmodels' exact diffuse filter at a
    target scale of 1, ``2**266`` (about 1.2e80) and ``2**-282`` (about
    1.6e-85), with ``obs_var`` given: ``R = obs_var / w``, every slot's
    prior ``p0 R`` at ``p0 = 1e15``, vague enough that the exact diffuse
    filter is its limit (review fable2, C5). Carlson's step of the square
    root formed ``sqrt(alpha * next)``, a product of order ``R**2``, which
    overflowed at a noise near 1e154 and dropped its cross term -- on
    a5c4dcef 2.2 off here, relative -- and underflowed near 1e-154, 0.64 off.

    A power of two scales every number of an exact filter exactly, so the
    predictions over the scale are the unit scale's: statsmodels' at the
    unit scale is the oracle at each. Its own filter is that at ``2**266``,
    to the bit, and 0.29 off it at ``2**-282``, where it too squares the
    noise. Measured (2026-11-20): the square root 4.1e-15 of the
    prediction's size off it at every scale; the covariance form 1.1e-2, the
    digits a vague prior costs it, whatever the scale."""
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    rng = np.random.default_rng(282)
    n = 400
    x = rng.standard_normal((n, 2)) * [1.0, 0.5] + [0.5, 1.0]
    y = 5.0 + 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    kf = KalmanFilter(k_endog=1, k_states=3)
    kf.bind(y[:, None])
    kf["design"] = np.column_stack([np.ones(n), x]).T[None, :, :]
    kf["obs_cov"] = (0.01 / w)[None, None, :]
    kf["transition"] = np.eye(3)
    kf["selection"] = np.eye(3)
    kf["state_cov"] = np.zeros((3, 3))
    kf.initialize_diffuse()
    res = kf.filter()
    s = res.nobs_diffuse
    assert s == 3
    want = res.forecasts[0][s:]
    frame = pl.DataFrame({"t": np.arange(n, dtype=float), "y": scale * y, "w": w}).with_columns(
        **{f"x{i}": pl.Series(x[:, i]) for i in range(2)}
    )
    spec = rp.spec.kalman(
        "k",
        targets=["y"],
        features=["x0", "x1"],
        clock="t",
        gap_cap=1e9,
        coef_half_life=float("inf"),
        p0=1e15,
        obs_var=0.01 * scale * scale,
        standardize=False,
        weight="w",
        half_life=100.0,
        min_weight=0.0,
        covariance_form="square_root",
    )
    got = rp.ModelBank([spec], threads=1).fit_predict(frame).unnest("k")["pred_y"]
    got = got.to_numpy().astype(float)[s:] / scale
    off = float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1.0)))
    assert off < 1e-11, off


@pytest.mark.parametrize("tiny", [5e-324, 1e-310, 2e-308])
def test_a_first_row_of_subnormal_weight_leaves_statsmodels_holt(tiny):
    """A first row whose weight is below the smallest normal double
    teaches nothing, as a weight of 0 teaches nothing (review fable2,
    C7): from the second row on, the stream is statsmodels' ``Holt`` on
    the series without it, from the row where the gains have settled.
    Before, the line fit's ``1/Q`` was infinite on the second row and
    every later prediction ``NaN``."""
    import statsmodels.tsa.holtwinters as holtwinters

    n = SETTLED + 200
    y = _rising(n)
    spec = rp.spec.holt(
        "h",
        targets=["y"],
        half_life=H_LEVEL,
        trend_half_life=H_TREND,
        weight="w",
        min_weight=0.0,
    )
    w = np.ones(n)
    w[0] = tiny
    frame = pl.DataFrame({"y": y, "w": w})
    pred = rp.ModelBank([spec]).fit_predict(frame)["h"].struct.field("pred_y").to_numpy()
    res = holtwinters.Holt(
        y[1:], initialization_method="known", initial_level=y[1], initial_trend=0.0
    ).fit(
        smoothing_level=1.0 - 0.5 ** (1.0 / H_LEVEL),
        smoothing_trend=1.0 - 0.5 ** (1.0 / H_TREND),
        optimized=False,
    )
    fitted = np.asarray(res.fittedvalues)
    settled = slice(SETTLED, None)
    assert np.all(np.isfinite(pred[2:]))
    np.testing.assert_allclose(pred[1:][settled], fitted[settled], rtol=1e-12)
