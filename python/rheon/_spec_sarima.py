"""The seasonal ARIMA builder, :func:`sarima`: a part of :mod:`rheon.spec`,
split from ``_spec.py`` to keep it under the repository's file-size cap.
Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def sarima(
    name: str,
    *,
    targets: TargetList,
    order: tuple[int, int, int] | list[int],
    seasonal_order: tuple[int, int, int, int] | list[int] | None = None,
    features: list[str] | None = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
    se_bound: float = 0.1,
    spacing: float | Duration | None = None,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A seasonal ARIMA model with exogenous regressors, SARIMAX, for each
    target, its parameters estimated online.

    Use it to predict a series one slot ahead from its own past, its past
    errors and its season, such as an hourly load with a daily cycle: the
    model differences the series itself, keeps the lags it reads and
    estimates the coefficients as the stream runs. Its parameters are
    statsmodels' ``SARIMAX(order, seasonal_order, trend="n")``'s.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``gap_cap``, ``min_weight`` and ``group``, are in :mod:`rheon.spec`.
    The model does not decay: ``half_life`` and ``lam`` are refused, and
    every learned row counts alike, a row's ``weight`` its share of the
    estimate. A row of weight 0, or whose target is null, is a slot not
    observed: it keeps its place in the series, so the seasonal lags after
    it stay aligned, and is predicted through. ``fit_intercept`` has nothing
    to act on, ``SARIMAX``'s ``trend="n"``: a constant is a feature of ones.

    This model's own parameters:

    ``order``
        ``(p, d, q)``: the autoregressive order, the differences taken and
        the moving-average order. Required. Each a count of slots.
    ``seasonal_order``
        ``(P, D, Q, s)``: the seasonal orders and the season ``s``, in slots,
        at least 2 where ``P``, ``D`` or ``Q`` is above 0. Default none,
        ``(0, 0, 0, 0)``.
    ``features``
        Exogenous regressors, the same for every target, entering as
        ``SARIMAX``'s ``exog``: differenced with the series, their
        coefficients estimated with the polynomials'. Default none.
    ``enforce_stationarity``, ``enforce_invertibility``
        Hold the autoregressive factors stationary and the moving-average
        ones invertible, through ``SARIMAX``'s transforms. Default on.
    ``se_bound``
        The largest standard error the start's fit is accepted with: each
        ARMA coefficient's as it is (they are unit-free), each regressor's
        coefficient's times the regressor's standard deviation over the
        innovations', and ``sigma2``'s relative to ``sigma2``, so the gate
        does not depend on the target's units. Finite and above 0. Default
        0.1: measured over five specs and eight seeds each at 10,000 rows,
        every start at 0.1 ended within 0.4 of ``SARIMAX.fit``'s standard
        errors of its parameters, where a bound of 0.15 or 0.2 left a spec
        whose seasonal factors nearly cancel 4.5 to 8 off. Smaller waits
        longer for a better start.
    ``spacing``
        Clock units a slot, a number of the clock column's units or a
        duration on a temporal clock: a step of the clock of ``n`` times
        this advances the series ``n`` slots, the ones between not
        observed. Default 1, one slot a clock unit.
    ``max_state_bytes``
        The bytes a group's state may hold. Before its start a target
        buffers its slots, ``9 (2 + k)`` bytes each for ``k`` features; a
        warning is given past half of this, and the slot that would pass
        it is refused. After the start a target holds
        ``9 (n + 2 + n² + L (6 + 2k))`` bytes, ``n = k + p + q + P + Q``
        and ``L = max(p + sP, q + sQ, 1)``. Default 256 KB, ``262144``; at
        most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#sarima
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#sarima>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the one-slot prediction of the series, read before the
        row is learned, and ``y - pred`` where the target is not null. Null
        until the target's start is accepted, and below ``min_weight``.
    ``weight_sum``
        The weight of every row before this one, with no decay.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).
    ``coef``
        Per target after the row, in statsmodels' order: the features'
        coefficients, ``ar.L1``, ..., ``ma.L1``, ..., ``ar.S.L<s>``, ...,
        ``ma.S.L<s>``, ..., then ``sigma2``. Null until the target's start
        is accepted. :func:`coef_index` names them.

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); and ``ValueError`` naming the
    problem for an order past the lag ceiling, a seasonal order with ``s``
    under 2, a ``spacing`` not finite and above 0, an ``se_bound`` not
    finite and above 0, and a state past ``max_state_bytes``.

    .. rubric:: The update

    With ``B`` the lag of one slot, the differenced series is ``w_t = (1 -
    B)^d (1 - B^s)^D y_t``, and the model is

    .. code-block:: text

        phi(B) Phi(B^s) (w_t - xw_t' beta) = theta(B) Theta(B^s) e_t,   e_t ~ (0, sigma2)

    with ``xw`` the features differenced alike. Each target's slots past the
    differencing's fill are buffered until Kish's count of them,
    ``(Σw)² / Σw²``, passes the parameters (``sigma2`` among them) plus
    ``d + sD``. Then the buffer's conditional likelihood, ``sigma2``
    concentrated, ``l = -W/2 ln(S/W)`` with ``S = Σ w e²`` and ``W = Σ w``,
    is maximized in ``SARIMAX``'s unconstrained coordinates ``x``. The fit
    is accepted only off every stationarity and invertibility bound,
    converged, and with every standard error at most ``se_bound``;
    otherwise the target keeps buffering and tries again once Kish's count
    has grown by half. Then the buffer is replayed at the fit and dropped,
    and each learned row of weight ``w`` steps by recursive prediction
    error, with ``psi = -de/dx`` and ``J = dpar/dx``:

    .. code-block:: text

        pred_t  = xw_t' beta - Σ A_i z_{t-i} + Σ C_j e_{t-j},   z = w - xw' beta
        V'      = V + w,   a = w / V'
        R'      = R + a (J'psi psi'J - R)
        x'      = x + a R'^-1 J'psi e_t
        sigma2' = sigma2 + a (e_t² - sigma2)

    where ``A`` and ``C`` are the coefficients of ``phi(B) Phi(B^s)`` and
    ``theta(B) Theta(B^s)`` past their first. ``pred`` adds the differencing
    back: ``y_t``'s prediction is ``ŵ_t`` plus the part of ``w_t`` that the
    past values give.

    .. rubric:: Example

    This code simulates an AR(1) with a seasonal moving average at a
    season of 4 and prints the fit after 3,000 rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        e = rng.standard_normal(3_004)
        u = [0.0] * 4
        for t in range(4, 3_004):
            u.append(0.6 * u[-1] + e[t] - 0.5 * e[t - 4])
        pl.DataFrame({"y": u[4:]}).write_parquet("series.parquet")

        model = rp.spec.sarima("s", targets=["y"], order=(1, 0, 0), seasonal_order=(0, 0, 1, 4))
        bank = rp.ModelBank([model])
        bank.fit(pl.scan_parquet("series.parquet"))
        coef = {row["term"]: row["coef"] for row in bank.coef().iter_rows(named=True)}
        print(f"ar.L1: true 0.6, found {coef['ar.L1']:.1f}")
        print(f"ma.S.L4: true -0.5, found {coef['ma.S.L4']:.1f}")

    .. code-block:: text

        ar.L1: true 0.6, found 0.6
        ma.S.L4: true -0.5, found -0.5

    .. rubric:: See also

    - :func:`ar`, least squares on lags under decay, with no moving average.
    - :func:`garch`, for the variance of a series rather than its value.

    .. rubric:: Notes

    **Against the library.** The start is ``SARIMAX``'s conditional fit;
    the steps after it are one pass, so the parameters are not the batch
    fit's, but they track it: over fifty seeds the estimate is held within
    the standard errors the gate accepts of ``SARIMAX.fit``'s on the same
    rows. The prediction is the conditional recursion's, not the exact
    Kalman filter's, so its state is linear in the season.
    """
    model: dict[str, Any] = {
        "type": "sarima",
        "order": list(order),
        "enforce_stationarity": enforce_stationarity,
        "enforce_invertibility": enforce_invertibility,
        "se_bound": se_bound,
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if seasonal_order is not None:
        model["seasonal_order"] = list(seasonal_order)
    if spacing is not None:
        model["spacing"] = spacing
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    return _common(name, model, targets=targets, features=features or [], **common)
