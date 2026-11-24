"""The trend model's builder, :func:`holt`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration, duration_text
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def holt(
    name: str,
    *,
    targets: TargetList,
    trend_half_life: float | Duration | None = None,
    trend: bool = True,
    forecast_spans: list[float | Duration] | dict[str, float | Duration] | None = None,
    seasonal: Literal["add", "mul"] | None = None,
    seasonal_periods: int | None = None,
    period: float | Duration | None = None,
    seasonal_half_life: float | Duration | None = None,
    damping_half_life: float | Duration | None = None,
    start: Literal["warm", "known", "heuristic"] | None = None,
    initial_level: float | list[float] | None = None,
    initial_trend: float | list[float] | None = None,
    initial_seasonal: list[float] | list[list[float]] | None = None,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Holt's linear trend, which extrapolates the target's own level and slope.

    Use it as the baseline a model with features should beat: it takes no
    features, so a regression that cannot outperform "the series is going up
    at about this rate" has features that are not earning their place. Run
    it in the same bank as the regression, with ``emit_sigma=True`` on both,
    and compare their ``sigma_<slot>`` fields. Or add a :func:`seqtest`
    whose ``a`` and ``b`` name the two specs, to test which predicts closer.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`rheon.spec`.

    The level forgets at the spec's ``half_life`` (or ``lam``), in clock
    units, ``inf`` included: one parameter under one name, as for every
    model. ``level_half_life`` is refused, with a message naming
    ``half_life``. ``fit_intercept`` has nothing to act on, since the level
    is the prediction's intercept.

    ``seasonal`` adds a season, ``damping_half_life`` damps the trend, and
    ``start`` chooses where the fit starts: Holt-Winters and the fixed
    forms of exponential smoothing (*Seasons and damping* in Notes). The
    alternative for a seasonal index is a column of the phase (the hour of
    the day, say), made with Polars' ``with_columns`` in the query before
    the bank and passed as ``group=``: the bank then fits a level and trend
    for each phase.

    This model's own parameters:

    ``trend_half_life``
        How fast the trend forgets, in clock units, above 0. Default four
        times the level's half-life, so the trend forgets four times more
        slowly than the level. ``inf`` is the whole history's drift, not a
        trend pinned at zero: the line fit's gain on the slope, which is the
        weighted least-squares slope exactly only when the level does not
        decay either. Beside a finite level half-life the slope learns from
        residuals against a level that forgets, and lands near the line fit's:
        0.2% to 1% of it off at a level half-life of 5 over 3,000 rows.
    ``trend``
        ``False`` fits the level alone: the trend is held at zero and the
        prediction is flat, which is simple exponential smoothing. The level
        is then the weighted mean of the observations, each at its weight
        times ``0.5 ** (age / half_life)``, which is pandas'
        ``ewm(adjust=True)`` for unit weights. ``trend_half_life`` is refused
        beside it. Default ``True``.
    ``forecast_spans``
        The forecasts to report, each a span of the clock ``Δ`` past the
        row read: a number of clock units (rows without a clock), or a
        duration such as ``"30d"`` on a temporal clock. Each gives the
        forecast ``Δ`` past the row and its integral over the span, read
        before the row is learned, as ``pred_<t>`` is; at ``Δ = 0`` the
        forecast is ``pred_<t>`` itself. A list labels each horizon's fields
        ``h<Δ>`` for a number and by the duration as written
        (``forecast_spans=[5, 20]`` gives ``forecast_h5_<t>`` and
        ``forecast_sum_h5_<t>``); a dict names them. They are spans of the
        clock, never counts of rows: the trend is per clock unit, and the
        clock of the rows to come is not known. There is no forecast
        variance: the variance of exponential smoothing's forecast (with
        additive errors and trend) counts the innovations of the rows
        within the span and assumes fixed gains, which holds here only on a
        clock of evenly spaced rows of equal weight, once the gains have
        stopped moving, and the model keeps no noise variance of its own. At most
        64, each finite and 0 or more; a name must not begin with
        ``"sum"``. Refused beside ``seasonal``, ``damping_half_life`` or a
        ``start`` other than ``"warm"``, whose forecasts are not a line to
        integrate. Default none.
    ``seasonal``
        ``"add"`` adds a seasonal value to the level, ``"mul"`` multiplies
        the level by a seasonal factor; under ``"mul"`` every value must
        stay above 0. Default no season. Needs ``start`` ``"known"`` or
        ``"heuristic"``.
    ``seasonal_periods``
        ``m``, the number of values a season is held in, at least 2.
    ``period``
        ``P``, the season's span in clock units (a duration on a temporal
        clock): each value covers ``P / m`` of it. Default ``m``, one value
        per clock unit, which is one per row without a clock column.
    ``seasonal_half_life``
        How fast a seasonal value forgets, in clock units: learned once a
        season on an even clock, at ``γ = 1 - 0.5 ** (P / h)``. ``inf``
        holds the values where they start. Required with ``seasonal``.
    ``damping_half_life``
        Damps the trend by ``φ = 0.5 ** (1 / h)`` a clock unit, so the
        forecast levels off. Default ``inf``, undamped. Needs ``start``
        ``"known"`` or ``"heuristic"``, and the trend on.
    ``start``
        ``"warm"`` (the default) starts from no weight, as above, and is
        the start for the forms without a season or damping only.
        ``"known"`` starts from ``initial_level``, ``initial_trend`` and
        ``initial_seasonal``, with every gain the fixed one from the first
        row. ``"heuristic"`` holds a target's first rows (10 without a
        season, else 5 seasons, or more where ``m`` is small), fits the
        initial states to them as statsmodels' ``initialization_method=
        "heuristic"`` does, then learns those rows from those states; they
        are withheld as ``below_warm_rows``.
    ``initial_level``, ``initial_trend``, ``initial_seasonal``
        The known start's states, the state one step before the first row:
        a number per target (or one number for one target); the seasonal
        values a list of ``m`` per target (or one list), phase 0 first,
        phase 0 being the first row's.
    ``max_state_bytes``
        The bytes the model's state may hold, the heuristic start's held
        rows included: 256 KB by default, at most 64 MiB. A season takes
        16 bytes a value a target, and the heuristic start 36 bytes a held
        row a target until it is fitted.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#holt
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#holt>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction, and ``y - pred`` where the target is not
        null.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's predictions were withheld, null where they were not:
        a reason of the shared gates (:mod:`rheon.spec`), such as
        ``below_min_weight`` on the first row under the default
        ``min_weight``. A target's first observation has no prediction,
        since it sets the level: with ``min_weight=0``, ``pred_<t>`` and
        ``withheld_reason`` both read null there.
    ``forecast_<h>_<t>``, ``forecast_sum_<h>_<t>``
        Per target and horizon: the forecast ``Δ`` past the row,
        ``l + b * (s + Δ)``, and its integral over the span,
        ``Δ * (l + b * (s + Δ / 2))``, from the level and trend before the
        row. Null where ``pred_<t>`` is.
    ``coef``
        ``[level, trend]`` per target after the row is learned, the two
        numbers a prediction reads, which :func:`coef_index` names. Null
        for a target not yet observed, which has no level to report. The
        trend is 0 under ``trend=False``.

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take, and for
    ``level_half_life``, whose name is ``half_life``; and ``ValueError``
    naming the problem for a ``trend_half_life`` not above 0,
    ``trend_half_life`` beside ``trend=False``, and a horizon below 0, not
    finite, given twice, more than 64 of them or a name beginning with
    ``"sum"``.

    .. rubric:: The update

    Per row and target, ``s`` is the clock since the target was last
    observed, this row's step included, each step capped at ``gap_cap`` (or
    ``session_gap`` at a session change), so ``s`` is the row's own step on
    a stream with no gaps. ``w`` is the row's weight, ``e = y - pred`` the
    residual, ``l`` and ``b`` the level and the trend, and ``W``,
    ``V`` the weight the level and the trend have gathered. Each decays on
    its own half-life: ``λ_l = 0.5 ** (s / half_life)`` (``lam ** s`` under
    ``lam``), the spec's own, and
    ``λ_b`` likewise at ``trend_half_life``. ``S0``, ``a`` and ``Q`` are the
    line fit's weight, mean age and spread of ages (*The line fit* in
    Notes):

    .. code-block:: text

        pred = l + b * s                          extrapolate s clock units ahead
        textbook:  g_t = w / (λ_l * W + w)        W' = λ_l * W + w
                   h_t = w / (λ_b * V + w) * g / s
                                                  V' = λ_b * V + w   (s > 0)
        line fit:  S0' = S0 + w
                   a'  = S0 * (a + s) / S0'
                   Q'  = Q + S0 * w * (a + s)**2 / S0'
                   g_f = w * (1 / S0' + a'**2 / Q')
                   h_f = w * a' / Q'
        l' = pred + g * e      g = g_f under half_life = inf, else max(g_f, g_t)
        b' = b + h * e         h = h_f under trend_half_life = inf or at s = 0,
                               else max(h_f, h_t)

    The forecasts at the requested spans extrapolate the level and trend before the
    row, ``Δ`` clock units past it, and integrate them over the span; nothing
    decays across the span, since the half-lives weigh the evidence behind
    ``l`` and ``b``, not the extrapolation:

    .. code-block:: text

        f(Δ) = l + b * (s + Δ)                       f(0) = pred
        F(Δ) = Δ * (l + b * (s + Δ / 2))             the integral of f over [0, Δ]

    Under ``trend=False`` the line fit is not used: ``g = g_t`` at every
    half-life, ``inf`` included, and ``b = 0``.

    A target's first observation sets ``l = y``, ``W = S0 = w`` and
    ``a = Q = 0``, and predicts nothing. The trend starts at 0.

    .. rubric:: Example

    This code fits a series that starts at 10 and rises 0.5 a row, and
    prints the level and trend found at the last row beside the truth:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        t = np.arange(2_000)
        y = 10.0 + 0.5 * t + rng.standard_normal(2_000)
        pl.DataFrame({"y": y}).write_parquet("trend.parquet")

        holt = rp.spec.holt("holt", targets=["y"], half_life=200.0)  # no features
        bank = rp.ModelBank([holt])
        bank.fit(pl.scan_parquet("trend.parquet"))  # learn every row, keep no output
        coef = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
        print(f"level: true {10.0 + 0.5 * t[-1]:.1f}, found {coef['level']:.1f}")
        print(f"trend: true 0.50, found {coef['trend']:.2f}")

    .. code-block:: text

        level: true 1009.5, found 1009.5
        trend: true 0.50, found 0.50

    .. rubric:: See also

    - :func:`seqtest`, to test whether a regression predicts closer than
      this baseline.
    - :func:`ewridge`, a regression on features to set against it.
    - :meth:`rheon.polars.ModelBank.predict`, which extrapolates the level
      and trend to the row's clock.

    .. rubric:: Notes

    **The textbook gains.** The textbook form is Holt's recursion in its
    error-correction form, whose smoothing constants are ``1 - λ_l`` on the
    level and ``1 - λ_b`` on the trend once the weights reach their steady
    state. Its gains here are weighted means, as every running sum here is:
    a row at weight ``w`` counts ``w`` times. Where they win, ``l'`` is
    ``(λ_l * W * pred + w * y) / (λ_l * W + w)``, and ``b'`` the weighted
    mean of ``b`` and the slope the level moved at, ``(l' - l) / s``. Once
    the weights reach their steady state they are those fixed constants.

    **The line fit.** The line fit's gains are those of the weighted
    least-squares line through every observation so far, on the clock, read
    at the last. ``S0`` is its weight, ``a`` the observations' weighted mean
    age and ``Q`` the spread of their ages about it, none decayed. With every
    observation at one clock (``Q = 0``) no slope is determined:
    ``g_f = w / S0'`` and ``h_f = 0``.

    **Which gain a row takes.** A row takes the larger gain, on the level and
    on the slope. Under ``half_life = inf`` the level takes the line fit's
    gain, so without decay the model is the weighted line fit, to rounding
    (statsmodels' ``WLS`` agrees to 1e-15). Under ``trend_half_life = inf``
    the slope takes the line fit's gain. Under a half-life the line fit's
    gains fall as ``1/n`` and ``1/n**2`` (``n`` the number of
    observations), and the textbook's reach their
    fixed values. So after a warm-up of a few half-lives every row takes the
    textbook's, and statsmodels' ``Holt`` agrees from there. The warm-up
    follows a new series sooner than the textbook would. Without the line
    fit, gains of ``1/n`` on both level and trend would pin the trend to the
    first observation. The level would
    carry that observation's noise times ``ln n``: 11 off a flat series
    after 40,000 rows without decay, at a noise of 0.5, the worst of 50
    seeds.

    **Rows at one clock, and rows that learn nothing.** A row at the last
    row's clock is a second observation the level takes in. The textbook
    trend holds, since a move over no clock has no slope, and the line
    fit's slope moves by its own gain, as a least-squares slope does. A row
    with a null target or a zero weight leaves ``l`` and ``b`` where the
    last observation put them and carries its clock to the next one. So it
    gives the same level, trend and predictions as if it were absent. A row
    with a null target still adds its weight to ``weight_sum``.

    **Seasons and damping.** Under ``start = "known"`` or
    ``"heuristic"`` the gains are the fixed ones: ``α = 1 - λ_l`` over the
    row's step ``s`` at weight 1. A weighted row's gain is the weighted
    mean's whose held weight is the one rows at this step saturate at,
    ``w * α / (λ + w * α)``. With ``L = l + φ_s * b`` the level carried
    over the step, ``φ_s = φ * (1 - φ ** s) / (1 - φ)`` (``s`` undamped),
    and ``p`` the row's phase:

    .. code-block:: text

        pred   = L + S[p]                       (add)     L * S[p]   (mul)
        l'     = α * y - α * S[p] + (1 - α) * L  (add)
        l'     = α * y / S[p] + (1 - α) * L      (mul)
        l'     = α * y + (1 - α) * L             (no season)
        b'     = β * (l' - l) / s + (1 - β) * φ ** s * b
        S'[p]  = γ * y - γ * L + (1 - γ) * S[p]  (add)
        S'[p]  = γ * y / L + (1 - γ) * S[p]      (mul)

    with ``β = 1 - λ_b`` over ``s`` and ``γ = 1 - 0.5 ** (Δ / h)``, ``Δ``
    the clock since the phase was last learned (``P`` on an even clock, and
    ``P`` the first time). The other ``m - 1`` values are untouched. On a
    row clock at weight 1 this is statsmodels' ``ExponentialSmoothing``
    with ``smoothing_level = α``, ``smoothing_trend = β``,
    ``smoothing_seasonal = γ`` and ``damping_trend = φ``, operation for
    operation.

    A row's phase is ``floor(m * (c - c0) / P) mod m``, ``c`` the row's own
    clock value and ``c0`` the first row's, in integers on a temporal or
    integer clock whose ``P`` is a whole number of its units; on a float
    clock a row exactly on a bucket's edge may round to either side. It is
    the row's raw clock value, which ``gap_cap`` and ``session_gap`` do not
    shorten, so nothing between two rows moves a phase. A row with a null
    target keeps its place, and its phase is not learned. Without a clock
    column a row counts its place, and a row of weight 0 counts in no clock,
    so the row after it takes its place: null the target instead to keep a
    row's place in the season. A row at the clock of the last learned row
    teaches nothing, since its gains, ``1 - λ ** 0``, are 0.

    Under ``"mul"`` a row whose target is 0 or below, or whose update would
    make the level or the seasonal factor 0 or below, is predicted and not
    learned, and its ``withheld_reason`` is ``target_not_positive``, a category
    only this form's ``Enum`` holds. The seasonal values are not
    renormalized.

    **Extrapolating over the clock.** The trend is per clock unit, so an
    irregular clock extrapolates the right distance.
    :meth:`rheon.polars.ModelBank.predict` extrapolates over the clock
    distance from the row the model last learned, capped by ``gap_cap``.
    """
    model: dict[str, Any] = {
        "type": "holt",
        "trend_half_life": trend_half_life,
    }
    if not trend:
        # Absent when on, as the Rust spec leaves it out, so a bank reports
        # the dict this made (`ModelBank.specs`).
        model["trend"] = False
    if forecast_spans is not None:
        # A dict is `[name, span]` pairs, each span a number or a duration's
        # text, as the Rust spec reads them.
        who = f"spec {name!r}"
        model["forecast_spans"] = (
            [[k, duration_text(v, who, "forecast_spans")] for k, v in forecast_spans.items()]
            if isinstance(forecast_spans, dict)
            else list(forecast_spans)
        )
    # The fixed forms' keys, each absent when not given, as the Rust spec
    # leaves them out.
    for key, value in (
        ("seasonal", seasonal),
        ("seasonal_periods", seasonal_periods),
        ("period", period),
        ("seasonal_half_life", seasonal_half_life),
        ("damping_half_life", damping_half_life),
        ("start", start),
        ("initial_level", initial_level),
        ("initial_trend", initial_trend),
        ("initial_seasonal", initial_seasonal),
        ("max_state_bytes", max_state_bytes),
    ):
        if value is not None:
            model[key] = value
    return _common(name, model, targets=targets, features=[], **common)
