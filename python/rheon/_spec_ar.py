"""The time-series builders, :func:`ar` and :func:`spectral`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import them from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
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
    select: Literal["aic", "bic"] | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """An autoregression for each target, fitted by exponentially weighted
    least squares on the target's own last values.

    Use it to predict a series one row ahead from its own recent past, such
    as the next value of a load or a sensor reading. The model keeps each
    target's last values itself, so the input needs no lag columns. With no
    decay its coefficients are statsmodels' ``AutoReg``'s on the same rows:
    an AR(p), or any list of lags.

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
    ``missing="drop"``. It is still predicted. ``embargo`` is refused: under
    a delay the rows it predicts would read older lags than the rows it
    learns from.

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
        Exogenous inputs, the same for every target: their coefficients
        follow the lags'. Each target is its own series, with its own lags
        and fit. Default none, a pure autoregression.
    ``ridge``
        A ridge penalty on the lag and feature coefficients, on the scale of
        the weighted mean of ``z zᵀ``; never on the constant. Default 0,
        least squares.
    ``max_state_bytes``
        The bytes a group's state may hold: per target ``8 max(lags)`` for
        its lags and at most ``8 (4k² + 17k + 64)`` for its fit, ``k`` the
        coefficients a target has. The state's size is fixed by these
        settings, never by the stream's length, so a spec past it is
        refused when it is built. Default 256 KB, ``262144``; at most 64 MiB.
    ``select``
        Choose each target's order after every learned row: ``"aic"`` or
        ``"bic"`` picks the ``q`` in ``0, ..., p`` whose lags ``1, ..., q``
        (with the features) minimize the criterion, and the prediction and
        ``coef`` are that order's, its lags past ``q`` reported as 0. Every
        order is fitted on the same rows, those after the first ``p``
        values, as statsmodels' ``ar_select_order`` fits them. The count
        in the criterion is Kish's effective number of rows, ``(Σw)² /
        Σw²`` over the target's learned rows at their decayed weights: the
        row count with no decay and unit weights. Needs ``lags`` as an order
        and ``ridge`` 0. Default none, every lag.
    ``window_size``
        Clock units of history each target's fit is read from, a hard
        cutoff: a row older than this is in no fit, as :func:`ewridge`'s
        ``window_size`` (a number of the clock's units, or a duration on a
        temporal clock). Under decay the rows inside are still weighted by
        it. The lags a row reads may be older than the window: a row's lags
        are part of the row. As rows leave, a fit can lose the spread it
        needs, so under a window each target's definiteness is checked on
        every row, and its predictions are withheld while it does not hold.
        Default none.
    ``closed``, ``snapshot_every``, ``max_rows_between_snapshots``
        The window's edge and the cadence of the snapshots it is cut from,
        as :func:`ewridge`'s.
    ``window_budget``
        The MiB a group's snapshots may take, every target's together, each
        target's fit a share of it; past it they thin or refuse the run, as
        :func:`ewridge`'s. Default ``{"refuse": 256}``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#ar
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#ar>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction from the target's last values and the
        row's features, read before the row is learned, and ``y - pred``
        where the target is not null. Null until the target has
        ``max(lags)`` values and its fit is defined (below), and below
        ``min_weight``.
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
    ``2**20``, a ``ridge`` below 0 or not finite, ``select`` beside lags that
    are not ``1`` to an order or beside a ridge, and a state past
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
        S'     = (λ W_y S + w z zᵀ) / W_y'     the weighted mean of z zᵀ, centered
        r'     = (λ W_y r + w z y) / W_y'      the weighted mean of z y, centered
        β      = (S + ridge I)⁻¹ r             solved after every learned row
        c      = ȳ - β · z̄                     the weighted means

    A learned row of a target is one with a weight above 0 and the target
    present. It is appended to the target's last values; it enters ``S``
    and ``r`` only once ``max(lags)`` values precede it, since before that
    its lags do not exist (statsmodels' ``hold_back``). Every other row only
    ages the fit by ``λ``. The moments are kept centered, so a series far
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

    **Early rows.** Until a target's lags and features have spread in every
    direction, least squares has no unique answer. So a target's fit is read
    only once the system it solves, its Gram plus the ridge, is positive
    definite: a Cholesky factor with every pivot above ``1e-10`` of its diagonal. The
    guard is that condition, not a count of rows: on a well-spread series
    it opens at as many learned rows as coefficients, and on a feature that
    copies a lag it never opens. ``min_weight`` defaults to 0, with this
    gate in its place.

    **A trend or a season.** There is no ``trend`` or ``seasonal``
    parameter: a model sees the clock with every gap past ``gap_cap``
    shortened, so a time or a phase read from it would drift after each
    long gap. Make them as columns in the query from the clock column, such
    as ``pl.col("t")`` for a trend and the sine and cosine of
    ``2 * math.pi * pl.col("t") / 24`` for a daily cycle on an hourly clock,
    and pass them in ``features``.

    **Standard errors.** ``emit_se_coef`` reports each coefficient's
    standard error as :func:`ewridge` does, from the same fit: the
    coefficients' variance over the noise, from the fit's own factor, times
    the residual variance the bank keeps.
    """
    model: dict[str, Any] = {
        "type": "ar",
        "lags": lags,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if ridge is not None:
        model["ridge"] = ridge
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    if select is not None:
        model["select"] = select
    return _common(name, model, targets=targets, features=features or [], **common)


@_checked
def spectral(
    name: str,
    *,
    targets: TargetList,
    periods: list[float | Duration],
    spacing: float | Duration | None = None,
    scaling: Literal["density", "spectrum", "lomb_scargle"] = "density",
    detrend: bool = True,
    max_state_bytes: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """The power of each target at chosen periods, from an exponentially
    weighted Fourier transform, read as scipy's periodogram.

    Use it to watch how strong a cycle is as a stream runs: a daily rhythm in
    a load, a weekly one in sales, a vibration's line in a sensor. Each row
    reports, per target and period, the power before the row, under the
    spec's ``half_life``. With no decay on an even clock it is scipy's
    ``periodogram`` of the rows so far at those frequencies.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``weight`` and ``group``, are in
    :mod:`rheon.spec`. A row's ``weight`` scales its amplitude in the
    transform. A row of weight 0, or whose target is null, adds nothing; it
    is still read. A session change or a gap past ``gap_cap`` empties the
    transform, since the phase across it is not known.

    This model's own parameters:

    ``periods``
        The periods to transform at, each in clock units, or a duration on
        a temporal clock, at least twice ``spacing`` (the Nyquist limit),
        or under ``"lomb_scargle"`` any finite period. Their inverses are
        the frequencies. Required; at most 4,096.
    ``spacing``
        The clock units between rows, scipy's ``1/fs``: the density's scale
        and the place of the Nyquist frequency. Default 1, a row a clock
        unit, which is the stream's own clock without a clock column.
    ``scaling``
        ``"density"``, the default and scipy's, ``|X|² spacing / Σa²``, or
        ``"spectrum"``, ``|X|² / (Σa)²``, each doubled but at the zero
        frequency and the Nyquist; or ``"lomb_scargle"``, the generalized
        Lomb-Scargle power, which needs no even clock: per period, the
        share of the target's weighted variance that a sinusoid of that
        period explains, between 0 and 1, from the weighted least squares
        fit of ``a cos + b sin`` (and a constant with ``detrend``). It is
        scipy's ``lombscargle(normalize=True, floating_mean=detrend)`` with
        the amplitudes as ``weights``. Where the rows' phases have not yet
        spread over enough of a cycle to tell a cosine from a sine, or the
        target has not varied, the power is null.
    ``detrend``
        Remove the weighted mean before the transform, as scipy's
        ``detrend="constant"`` removes the mean. Default True.
    ``max_state_bytes``
        The bytes a group's state may hold: ``8 (4F + 3)`` a target for
        ``F`` periods, ``8 (6F + 4)`` under ``"lomb_scargle"``. Default
        256 KB; at most 64 MiB.
    ``window_size``
        Clock units of history the transform holds, a hard cutoff: the
        sliding-window transform of the rows less than this old, a number of
        the clock's units or a duration on a temporal clock. With no decay
        it is scipy's ``periodogram`` of the window's rows; under decay the
        rows inside are still weighted by it. A session change or a capped
        gap empties the window too. Default none.
    ``closed``, ``snapshot_every``, ``max_rows_between_snapshots``
        The window's edge and the cadence of the snapshots it is cut from,
        as :func:`ewridge`'s. Each snapshot holds the state and ``16 F``
        bytes more.
    ``window_budget``
        The MiB a group's snapshots may take; past it they thin or refuse
        the run, as :func:`ewridge`'s. Default ``{"refuse": 256}``.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#spectral
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#spectral>`_
    gives each field's type and when it is null:

    ``power_<t>_p<k>``
        Per target and period, ``k`` its place in ``periods``: the power
        before the row. Null before the target's first row.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for ``periods`` empty, longer than 4,096, or holding a period
    not finite, not above 0 or below twice ``spacing`` (but under
    ``"lomb_scargle"``, where any finite period is taken), a ``spacing`` not
    above 0, ``scaling`` not one of its words, and a state past
    ``max_state_bytes``.

    .. rubric:: The update

    Per row of clock step ``Δ``, weight ``w`` and target ``x``, with
    ``λ = 0.5 ** (Δ / half_life)`` and, per frequency ``f``, the rotation
    ``r = e^{2πi f Δ}``:

    .. code-block:: text

        X ← λ r X + w x        U ← λ r U + w
        A ← λ A + w            B ← λ B + w x        Q ← λ² Q + w²
        P = c |X − m U|² spacing / Q     (density; m = B / A with detrend)

    ``X`` is the transform of the rows, each at amplitude ``w λ^age``,
    referred to the current clock; ``U`` is the amplitudes' own, which
    removes the mean. ``c`` is 2, and 1 at the zero frequency and the
    Nyquist. Under ``"lomb_scargle"`` the model keeps two sums more, the
    amplitudes' transform at twice each frequency and ``Σ a x²``:

    .. code-block:: text

        V ← λ r² V + w         S ← λ S + w x²
        CC = (A + Re V)/2      SS = (A − Re V)/2      CS = Im V / 2
        YC = Re X              YS = Im X              YY = S
        P = (SS YC² − 2 CS YC YS + CC YS²) / ((CC SS − CS²) YY)

    with each centered first under ``detrend`` (``CC −= (Re U)²/A``,
    ``YC −= B Re U/A``, ``YY −= B²/A`` and so on). The ``sin`` and ``cos``
    are the same bits on every platform.
    The cost is two of them a frequency where the clock's step changes,
    none where it does not, and ``O(F)`` arithmetic a row.

    Under ``window_size`` the sums are referred to an origin fixed on the
    clock instead of the current row: a row ``σ`` clock units past it adds
    ``w x e^{-2πi f σ}``, and the origin moves to the row, the sums turned
    by ``e^{2πi f σ}``, each time a snapshot is taken. The window is the
    live sums less the snapshot taken before its first row, decayed
    forward and turned by the product of the turns since. That costs a
    ``sin`` and a ``cos`` a frequency on every row, and the turned sums keep
    their digits on long streams: over ten million rows with no decay the
    power is within ``1.5e-8`` of the window's periodogram.

    .. rubric:: Example

    This code finds a daily cycle in an hourly series:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(1)
        t = np.arange(24 * 60, dtype=float)
        y = np.sin(2 * np.pi * t / 24) + rng.standard_normal(t.size)
        pl.DataFrame({"y": y}).write_parquet("hourly.parquet")

        spec = rp.spec.spectral("s", targets=["y"], periods=[24.0, 10.0], half_life=float("inf"))
        out = pl.scan_parquet("hourly.parquet").rheon.fit_predict([spec]).collect()
        last = out["s"][-1]
        print(f"24 h: {last['power_y_p0']:.1f}, 10 h: {last['power_y_p1']:.1f}")

    .. code-block:: text

        24 h: 684.0, 10 h: 1.4

    .. rubric:: See also

    - :func:`ar`, to forecast a series from its own past.
    - :func:`ew_cov`, for a series' variance rather than its cycles.
    """
    model: dict[str, Any] = {
        "type": "spectral",
        "periods": periods,
        "spacing": spacing,
        "scaling": scaling,
        "detrend": detrend,
        "max_state_bytes": max_state_bytes,
        "window_size": window_size,
        "closed": closed,
        "snapshot_every": snapshot_every,
        "max_rows_between_snapshots": max_rows_between_snapshots,
        "window_budget": window_budget,
    }
    return _common(name, model, targets=targets, features=[], **common)
