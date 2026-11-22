"""The autoregression's builder, :func:`ar`: a part of :mod:`rheon.spec`,
split from ``_spec.py`` to keep it under the repository's file-size cap.
Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def ar(
    name: str,
    *,
    targets: TargetList,
    lags: int | list[int],
    features: list[str] | None = None,
    ridge: float | None = None,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """An autoregression for each target, AR(p): the target regressed on its
    own last values, and on the features if any, by exponentially weighted
    least squares.

    Use it to forecast a series from its own recent past: the next value of
    a load, a price change or a sensor reading, one row ahead. The model
    keeps each target's last values itself, so the input needs no lag
    columns. With no decay its coefficients are those of ordinary least
    squares on the lag design, statsmodels' ``AutoReg`` on the same rows;
    with a ``half_life`` it follows a series whose dynamics drift. Each
    target is its own series, with its own lags and fit; the features, if
    any, are exogenous regressors shared by every target.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`rheon.spec`. ``half_life`` (or ``lam``) is on
    the clock, in clock units or a duration on a temporal clock;
    ``fit_intercept`` fits the constant ``c``, statsmodels' ``trend="c"``,
    and ``False`` fits none, its ``trend="n"``. A row's ``weight`` scales
    its evidence. A row of weight 0, or whose target is null, is not a lag:
    the target's series is its learned values, so the next row's lags are
    the ones before it, as if it were not there, which is statsmodels'
    ``missing="drop"``. It is still predicted.

    This model's own parameters:

    ``lags``
        The order ``p``, for the lags ``1, ..., p``, or a list of lags, each
        at least 1 and strictly increasing, such as ``[1, 2, 24]``. Required.
        A lag counts learned rows of the target, not clock units: the value
        ``ℓ`` rows back has no meaning on a clock, so the lags are positions
        in the series and only the decay is on the clock. A session change
        or a gap past ``gap_cap`` empties the lags, since the rows on either
        side are no longer adjacent; the fit is kept, and the target's next
        ``max(lags)`` values refill them before it learns again.
    ``features``
        Exogenous regressors, the same for every target: their coefficients
        follow the lags'. Default none, a pure autoregression.
    ``ridge``
        A ridge penalty on the lag and feature coefficients, on the scale of
        the weighted mean of ``z zᵀ``; never on the constant. Default 0,
        least squares.
    ``max_state_bytes``
        The bytes a group's state may hold: per target ``8 max(lags)`` for
        its lags and at most ``8 (4k² + 16k + 64)`` for its fit, ``k`` the
        coefficients a target has. The state's size is fixed by these
        settings, never by the stream's length, so a spec past it is
        refused when it is built. Default 256 KB, ``262144``; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#ar
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#ar>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction from the target's last values and the
        row's features, read before the row is learned, and ``y - pred``
        where the target is not null. Null until the target has
        ``max(lags)`` values and a fit, and below ``min_weight``.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay: every row's, the first ``max(lags)`` included.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).
    ``coef``
        Per target after the row: ``intercept`` (with ``fit_intercept``),
        ``lag1``, ``lag2``, ... for each lag in order, then the features.
        Null until the target has a fit. :func:`coef_index` names them.

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); and ``ValueError`` naming the
    problem for ``lags`` empty, holding a 0, not strictly increasing or past
    ``2**20``, a ``ridge`` below 0 or not finite, and a state past
    ``max_state_bytes``, or a ``max_state_bytes`` of 0 or past 64 MiB.

    .. rubric:: The update

    For target ``y`` with lags ``ℓ_1 < ... < ℓ_q``, the design of row ``t``
    is ``z_t = (y_{t-ℓ_1}, ..., y_{t-ℓ_q}, x_t)``, its lags read from the
    target's last learned values. Per row, ``λ = 0.5 ** (Δclock /
    half_life)``, weight ``w``, ``W`` the weight of every row and ``W_y``
    the target's own:

    .. code-block:: text

        pred_t = c + β · z_t                   from the fit before the row
        W'     = λ W + w                       weight_sum reads W
        W_y'   = λ W_y + w                     a learned row of the target only
        S'     = (λ W_y S + w z zᵀ) / W_y'     the weighted mean of z zᵀ, centred
        r'     = (λ W_y r + w z y) / W_y'      the weighted mean of z y, centred
        β      = (S + ridge I)⁻¹ r             solved after every learned row
        c      = ȳ - β · z̄                     the weighted means

    A learned row of a target is one with a weight above 0 and the target
    present. It is appended to the target's last values; it enters ``S``
    and ``r`` only once ``max(lags)`` values precede it, since before that
    its lags do not exist (statsmodels' ``hold_back``). Every other row only
    ages the fit by ``λ``. The moments are kept centred, so a series far
    from zero loses nothing to its level.

    .. rubric:: Example

    This code simulates ``y = 2 + 0.6 y₋₁ - 0.3 y₋₂ + e`` and prints the
    fit after 20,000 rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        y = [5.0, 5.0]
        for e in rng.standard_normal(20_000):
            y.append(2.0 + 0.6 * y[-1] - 0.3 * y[-2] + e)
        pl.DataFrame({"y": y}).write_parquet("series.parquet")

        model = rp.spec.ar("a", targets=["y"], lags=2, half_life=float("inf"))
        bank = rp.ModelBank([model])
        bank.fit(pl.scan_parquet("series.parquet"))
        coef = {row["term"]: row["coef"] for row in bank.coef().iter_rows(named=True)}
        print(f"lag1: true 0.6, found {coef['lag1']:.2f}")
        print(f"lag2: true -0.3, found {coef['lag2']:.2f}")

    .. code-block:: text

        lag1: true 0.6, found 0.60
        lag2: true -0.3, found -0.31

    .. rubric:: See also

    - :func:`ewridge`, the same least squares on lag columns made in the
      query, for lags across columns or on the clock.
    - :func:`holt`, for a level and trend rather than a regression on lags.
    - :func:`garch`, for the variance of a series rather than its value.

    .. rubric:: Notes

    **Exact against the library.** With no decay and ``ridge=0`` the
    coefficients are least squares on the rows after the first
    ``max(lags)``, as ``AutoReg(y, lags, trend="c", exog=x).fit()`` computes
    them, to rounding: within ``1e-9`` relative on a well-conditioned
    design over 20,000 rows. Under decay they are weighted least squares
    with each row weighted ``w · λ^age``, its age on the clock.

    **Early rows.** A target's fit has as many coefficients as its constant,
    lags and features, and fewer learned rows than that leave least squares
    without a unique answer: the default ``min_weight`` withholds the
    prediction until the weight holds that many rows. A small ``ridge``
    makes the early fits defined.
    """
    model: dict[str, Any] = {"type": "ar", "lags": lags}
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if ridge is not None:
        model["ridge"] = ridge
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    return _common(name, model, targets=targets, features=features or [], **common)
