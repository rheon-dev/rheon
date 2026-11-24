"""The Kalman filter's builder, :func:`kalman`: a part of :mod:`rheon.spec`,
split from ``_spec.py`` to keep it under the repository's file-size cap.
Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration, duration_text
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common
from rheon._spec_words import quoted, words_refusal


@_checked
def kalman(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    coef_half_life: float | Duration | list[float | Duration] | None = None,
    q: list[float] | None = None,
    obs_var: float | None = None,
    p0: float | None = None,
    share_p: bool = False,
    revert_half_life: float | Duration | list[float | Duration] | None = None,
    standardize: bool = True,
    covariance_form: Literal["covariance", "square_root"] = "covariance",
    forecast_spans: list[float | Duration] | dict[str, float | Duration] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A regression whose coefficients drift, tracked by a Kalman filter.

    Use it where a coefficient moves at a drift rate of its own, faster
    than one half-life for the whole fit can follow, or where a coefficient
    no row supports should be forgotten (``revert_half_life``). Use
    :func:`ewridge` for coefficients that hold still over a half-life, and
    :func:`rls` for an exact fit whose prior fades.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`rheon.spec`. Here ``half_life`` drives
    the standardization and the residual variance, not the coefficients'
    drift. This model's own:

    ``coef_half_life``
        How fast a coefficient may drift, as a half-life in clock units, on
        standardized features: an observation ``D`` clock units after the
        target's last adds the process noise ``(ln 2 * D / h_i) ** 2`` times
        the noise ``R_j``, with ``h_i`` the coefficient's half-life. The
        half-life is the same whether rows come every unit or every
        hundredth, and whether the target is seen on every row or one in ten
        (*The drift* in Notes). It is each standardized direction's memory
        with the others held still, and a direction of correlated features
        is learned more slowly (*Correlated features* in Notes). A scalar,
        or one value per coefficient with the intercept first where there is
        one, each above 0.
        ``inf`` gives that coefficient no drift: the filter takes it as
        constant and goes on learning it from every row. Required unless
        ``q`` is
        given, and refused beside it. Under ``standardize`` the drift is per
        standard deviation of each feature as the moments stand: a slope's
        random walk in the caller's units has a variance of
        ``R_j * (ln 2 * D / h_i) ** 2 / s_i ** 2`` per observation, ``s_i``
        the feature's standard deviation. The diagnostics' statistics read
        the fit's memory as the shortest finite entry (the model's
        ``half_life`` where every entry is ``inf``), and their readiness
        waits for the longer of the longest finite entry and ``half_life``.
        The CUSUM under ``emit_breaks``
        reads the intercept's entry as the memory at which a level in the
        target is absorbed, and ``inf`` there leaves the CUSUM a plain sum.
    ``q``
        The process noise given outright, ``q_i`` per coefficient, in place
        of the derivation from ``coef_half_life``, added as ``q_i * D ** 2``
        on each row that observes the target. One value per coefficient, the
        intercept first where there is one, each finite and at least 0.
        Exactly one of ``q`` and
        ``coef_half_life`` is given. Its units are a coefficient's squared
        per clock unit squared. Under ``standardize`` a slope is in the
        target's units per standard deviation of its feature, so ``q_i`` is
        in the target's squared units per clock unit squared. Without it, a
        slope's ``q_i`` is in the target's units over the feature's, squared,
        per clock unit squared. The intercept's is the target's squared
        units per clock unit squared either way. ``q`` is measured against
        the noise estimate, ``q_i / R_j``, which is how the estimate reaches
        the gain under ``q`` (*The noise* in Notes).
    ``obs_var``
        A fixed observation noise in place of the estimated one: the
        variance of the target about its prediction, in the target's squared
        units, finite and above 0. Default none, which estimates it as the
        exponentially weighted (EW) variance of the target's out-of-sample
        residuals. It scales ``se_coef``. Under ``coef_half_life`` the gain
        does not depend on its value, and under ``q`` it is the ``R_j`` that
        ``q`` is measured against. ``P`` is then absolute, and starts at
        ``p0 * obs_var`` on the intercept, and on each slope too where
        ``standardize=False``. Under ``standardize`` each slope's prior waits
        for its feature's scale (*The prior* in Notes).
    ``p0``
        The prior variance of each slope, in units of the noise: a ratio,
        with no units under ``standardize``. Without ``standardize`` a
        slope's prior ``p0 * R`` carries its feature's units, so ``p0`` is in
        one over the feature's squared units there. Default 1.0, a prior as
        uncertain as one observation. Finite and above 0. From about 1e8 the
        default ``covariance_form`` loses digits; ``"square_root"`` keeps
        them (*The square root* in Notes).
    ``share_p``
        Keep one ``P`` for every target, where by default each target has its
        own. Targets observed on the same rows would each carry the same
        ``P``, so one update of the ``n x n`` matrix a row then serves every
        target, ``n`` the number of coefficients. Every prediction is then
        the per-target filter's, to the bit, except under ``q``, which the
        shared ``P`` measures against the targets' mean noise. Default
        ``False``, so that each target's memory follows its own rows: where
        a target is null on rows another is present on, the shared ``P``
        takes those rows too (*A shared P* in Notes).
    ``revert_half_life``
        A reversion half-life ``kappa_i`` per coefficient, in clock units,
        above 0: between observations the coefficient shrinks toward zero by
        ``2 ** (-d / kappa_i)``, so the filter forgets a coefficient no row
        has supported for a while, where a random walk would carry it
        unchanged. Default ``inf``, the random walk, which adds nothing. A
        scalar applies to every coefficient, the intercept included. A list
        gives one per coefficient, intercept first, and ``[inf, h, h]``
        leaves the intercept a random walk. Under ``standardize`` the pull is
        toward zero in the standardized coordinates: a slope toward "no
        effect", the intercept toward "the target averages zero". A
        reverting coefficient's uncertainty stays bounded across a run of
        rows that observe nothing (*Reversion* in Notes).
    ``standardize``
        Run the filter on standardized features, so ``coef_half_life`` and
        ``p0`` mean the same thing whatever the columns' scale. Default
        ``True``. Each row is standardized against the EW means ``m_i`` and
        standard deviations ``s_i`` of the rows before it,
        ``z_i = (x_i - m_i) / s_i``. Without an intercept the features are
        scaled by their root mean square and not centered, as for
        :func:`ewridge`. The reported coefficients are in the original units
        either way, and the moments' changes move no prediction
        (*Standardizing* in Notes). With ``standardize=False``, ``q`` all
        zeros, no ``revert_half_life`` and a fixed ``obs_var``, the filter is
        exactly Bayesian linear regression. For a coefficient per current standard deviation, which
        moves with the scale, z-score the column in the query before the bank
        and give it with ``standardize = False``. `Scale a feature by its
        spread
        <https://github.com/rheon-dev/rheon/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_
        builds the column with :func:`rheon.polars.ewm_mean
        <rheon.polars.ops.ewm_mean>` and :func:`rheon.polars.ewm_std
        <rheon.polars.ops.ewm_std>`.
    ``covariance_form``
        How the filter holds ``P``. ``"covariance"``, the default, holds
        ``P`` and updates it as ``P_j <- P_j - g z' P_j``, which subtracts two
        numbers of ``P``'s size to leave one of the posterior's, so it
        rounds at ``u * p0``, ``u`` the unit roundoff. ``"square_root"``
        holds a triangular ``C`` with ``P = C C'`` and updates ``C``, which
        rounds at ``u * sqrt(p0)`` (*The square root* in Notes). Use it for
        a vague prior, ``p0`` of about 1e8 or more. It costs time on every
        row: 1.14 to 1.72 times the covariance form's with no drift, and
        2.4 to 16 times under ``coef_half_life`` or ``q``, at 5 to 50
        coefficients on arm64, since each row's drift then updates the
        factor once per coefficient. At ``p0 = 1`` the two agree to about
        1e-9 of a prediction.
    ``forecast_spans``
        The forecasts to report, each a span of the clock ``Δ`` past the
        row read: a number of clock units (rows without a clock), or a
        duration such as ``"30d"`` on a temporal clock. Each is the
        prediction a row of this row's features would get at the row's
        clock plus ``Δ``, with no row between, as the filter would make it:
        its mean, its integral over the span and its variance. At ``Δ = 0``
        the mean is ``pred_<t>`` itself. The features are held at the row's:
        the forecast is the regression's at them. Under the default random
        walk the mean does not move with ``Δ``, and the variance grows by
        the drift the span adds; ``revert_half_life`` moves the mean too. A
        list labels each horizon's fields ``h<Δ>`` for a number and by the
        duration as written; a dict names them. They are spans of the clock,
        never counts of rows: the filter's drift and reversion are per clock
        unit. At most 64, each finite and 0 or more; a name must not begin
        with ``"sum"``. Default none.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#kalman
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#kalman>`_
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
        Why the row's predictions were withheld, null where they were not.
        It reads a reason of the shared gates (:mod:`rheon.spec`),
        such as ``below_min_weight`` on the first rows under the default
        ``min_weight``. It reads ``below_warm_rows`` while no coefficient of
        the target has its prior set yet (*The prior* in Notes), as without
        an intercept before a feature's scale is usable. A target's first row
        has no prediction, since the model has learned nothing to predict
        from: with ``min_weight=0``, ``pred_<t>``
        and ``withheld_reason`` both read null there.
    ``forecast_<h>_<t>``, ``forecast_sum_<h>_<t>``, ``forecast_var_<h>_<t>``
        Per target and horizon: the prediction ``Δ`` past the row at the
        row's features, ``z' T(Δ) b``, its integral over the span, and its
        variance, from ``T(Δ) P T(Δ)'`` and the drift of the clock since the
        target's last observation plus ``Δ``, scaled as ``pred_<t>``'s noise
        is (*The update*). Null where ``pred_<t>`` is, and the variance
        before there is a noise.
    ``coef``
        Per target, the intercept (unless ``fit_intercept=False``) then one
        entry per feature, in the original units, after the row is learned (:func:`coef_index`).

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them. ``se_coef`` and
    ``error_inflation`` read the filter's own covariance, and are exact only
    under the model (*Standard errors* in Notes).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), and ``ValueError``
    naming the problem for:

    - neither or both of ``coef_half_life`` and ``q``;
    - a ``coef_half_life`` or ``revert_half_life`` whose list is neither one
      value nor one per coefficient, or with a value not above 0;
    - a ``q`` whose list is not one value per coefficient, or with a value
      below 0 or not finite;
    - an ``obs_var`` or ``p0`` not finite or not above 0;
    - ``emit_robust_se``: the filter's covariance is its own posterior,
      which ``emit_se_coef`` reports.

    .. rubric:: The update

    Per target ``j``, per row, ``d`` is the row's clock step, the clock's
    step since the last row the model did not skip, capped at ``gap_cap``
    (or ``session_gap`` at a session change). ``w`` is the row's weight,
    ``z`` its standardized features with the intercept in front, ``y_j`` the
    target, ``b_j`` the coefficients and ``kappa_i`` the
    ``revert_half_life``:

    .. code-block:: text

        b_j <- Phi b_j                     Phi = diag(2 ** (-d / kappa_i))  the reversion
        P_j <- Phi P_j Phi                                                  every row
        D_j <- D_j + d                     the clock since P_j's last observation
        P_j <- P_j + Q * D_j ** 2, D_j <- 0                                 the drift, on a row
                                                                            that observes y_j
        v    = z' P_j z + 1 / w            the innovation's variance
        g    = P_j z / v                   the gain
        b_j <- b_j + g (y_j - z' b_j)
        P_j <- P_j - g z' P_j

    ``P_j`` is the coefficients' covariance in units of the observation
    noise ``R_j``, so the covariance itself is ``R_j * P_j``. ``Q`` is the
    process noise in the same units, ``(ln 2 / h_i) ** 2`` from
    ``coef_half_life`` or ``q_i / R_j`` from ``q``. ``R_j`` is the target's
    EW residual variance, or ``obs_var``. Under ``obs_var`` the filter runs
    on the covariance itself: ``P_j`` and ``Q`` are ``R_j`` times the above,
    and ``v`` is ``z' P_j z + R_j / w``. A reverting coefficient's drift
    over a gap ``D`` is ``Q_i * r_i(D) ** 2`` in place of ``Q_i * D ** 2``,
    with ``r_i(D) = (1 - 2 ** (-D / kappa_i)) / theta_i`` and
    ``theta_i = ln 2 / kappa_i``.

    .. rubric:: Example

    This code fits a slope that steps from 1 to 3 at row 2,000, and prints
    the slope the filter found just before the step, 10 rows after it and
    100 rows after it:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        x = rng.standard_normal(4_000)
        slope = np.where(np.arange(4_000) < 2_000, 1.0, 3.0)  # the true slope
        y = slope * x + 0.2 * rng.standard_normal(4_000)
        pl.DataFrame({"x": x, "y": y}).write_parquet("step.parquet")

        k = rp.spec.kalman(
            "k", targets=["y"], features=["x"], half_life=200.0,
            coef_half_life=20.0,                  # how fast a coefficient may drift, in rows
            coef_every=0,                         # write coef on every row
        )
        out = pl.scan_parquet("step.parquet").rheon.fit_predict([k]).collect()
        coef = out["k"].struct.field("coef")
        for row in (1_999, 2_010, 2_100):
            print(f"row {row}: true slope {slope[row]:.0f}, found {coef[row][1]:.2f}")

    .. code-block:: text

        row 1999: true slope 1, found 0.96
        row 2010: true slope 3, found 1.55
        row 2100: true slope 3, found 2.95

    .. rubric:: See also

    - :func:`ewridge`, for coefficients that hold still over a half-life.
    - :func:`rls`, the fit this filter gives with no drift, a fixed noise,
      ``standardize=False``, no ``revert_half_life`` and ``half_life=inf``,
      at ``delta = 1 / p0``.
    - :meth:`rheon.polars.ModelBank.predict`, which moves the coefficients
      by ``Phi`` over the clock since the last learned row.

    .. rubric:: Notes

    **The noise.** ``R_j`` is per target, and its weight ages on every row,
    a row with no prediction or with weight 0 included. It learns from the
    filter's own prediction on every row the filter predicts, whether or not
    ``min_weight`` lets that prediction out. ``min_weight`` withholds the
    output alone, so the fit, the variance and a ``q`` read against it are
    the same at any ``min_weight``. A row with no prediction, before the
    filter has sized the target's coefficients, still moves them and adds
    to the target's weight. Only the residual variance learns nothing from
    it. A row of weight 0, or one whose target is null, ages the target's
    weight and learns nothing for it.

    Under ``coef_half_life`` the noise is a scale, not a weight: the gain
    never reads the noise estimate. The model's noise is one constant per
    target. The prior (``p0`` times it) and the drift
    (``(ln 2 * D / h) ** 2`` times it) are multiples of it, so it cancels
    from the gain. The estimate scales only ``se_coef``: ``error_inflation``
    and its gate read ``P`` in units of the noise, and wait only until the
    target has a residual variance. Under ``q`` the estimate
    reaches the gain only through ``q_i / R_j``. A filter with the estimate
    inside the gain weighs each row by the inverse of the noise as it stood
    before the row, so a burst of large residuals inflates it and the rows
    after count for less than the burst did. In a run of that filter with
    no drift, a fifth of the first 500 rows 50 too high held the slopes at
    ``[-25.5, 33.6]``, where least squares on the same rows gives
    ``[4.1, -1.4]``. The residual variance starts at the target's first
    prediction. Under ``q``, before a target's first residual the row's own
    squared innovation stands in for the noise, and a row whose innovation
    is exactly 0 adds no drift. Scaling a target by ``c`` scales its
    predictions by ``c``, at any ``p0``, when ``obs_var`` and ``q``, where
    given, are scaled by ``c ** 2``.

    **States saved with the estimate in the gain.** A state saved by a
    release that put the noise estimate inside the gain goes on as it was
    fitted, with the estimate in the gain.

    **The drift.** The drift is charged once, on the row that next observes
    the target, for the whole clock since the last one (``D_j``, or under
    ``share_p`` since any target's row), because ``Q * D ** 2`` is not additive
    over a split gap. Charged per row, a row of weight 0 inside a gap would
    shrink the gap's noise and move every later prediction, by 0.197 in the
    target's units at ``coef_half_life = 20``. And a target seen on one row
    in 10 would forget 3.6 times slower on the clock than one seen on every
    row: 270 clock units against 75, measured at the same
    ``coef_half_life``. The noise ``(ln 2 * D / h_i) ** 2`` matches the
    steady-state gain of recursive least squares under a half-life ``h_i``
    (:func:`rls`) to first order in ``D / h_i``: within 1% for observations
    closer than half a half-life, and within 4% at one.

    **Correlated features.** Under correlated features, a direction of the
    design's correlation with eigenvalue ``lambda`` is learned with a memory
    of about ``h / sqrt(lambda)`` (Ljung and Gunnarsson 1990), where
    recursive least squares under a half-life forgets every direction at
    ``h``. At a correlation of 0.9 and ``h = 50`` the memories measured 37.5
    and 160.5 clock units, against 36.3 and 158.1 predicted. Under
    ``coef_half_life`` the gain reads no noise estimate, so a step in the
    truth is learned at that memory however much it inflates the residual
    variance.

    **The prior.** Without ``obs_var`` the intercept's prior is diffuse. The
    first row that observes the target fills the intercept with its whole
    innovation, ``b_0 = y - z' b``, at a variance of ``z' P z + 1 / w``.
    That is the limit of a prior that knows nothing of the level. So a level
    in the target moves every prediction by the level, to the rounding of a
    number of its size, from the first. It needs neither ``p0`` nor a noise:
    with no process noise the filter is least squares with the ridge
    ``1 / p0`` on each slope and none on the intercept, the noise's estimate
    playing no part. Unstandardized, each slope's prior is ``p0`` from the
    start. Standardized, it is ``p0`` from the first row that observes the
    target on which its feature's scale is usable, and until then the slope
    is 0 and takes no correction. While no coefficient is sized the target
    is not predicted, and with an intercept every row after the first has
    one sized. With ``obs_var`` given, unstandardized and with no process
    noise, the filter is the ridge regression with penalty ``1 / p0``. A
    covariance whose diagonal goes below 0 is sized again at ``p0``.

    **The square root.** Under ``covariance_form="square_root"`` the filter
    holds ``C``, upper triangular with ``P_j = C C'``. A row with ``f = C'
    z``, ``a_0 = 1 / w`` (``R_j / w`` under ``obs_var``) and ``a_j =
    a_(j-1) + f_j ** 2`` updates each
    column ``C_j`` of ``C`` in turn (Carlson 1973), to the factor of the
    posterior, with nothing subtracted at ``P``'s size:

    .. code-block:: text

        C_j <- sqrt(a_(j-1) / a_j) C_j - f_j / sqrt(a_(j-1) a_j) * sum_(i<j) C_i f_i

    The drift, a slope's prior and the reversion each update ``C`` by plane
    rotations. Against the posterior solved directly, on 300 rows of three
    features with unlike means and scales, unstandardized and with no
    drift: at ``p0 = 1e12`` the covariance form was 6.7e-5 off in a
    coefficient and the square root 1.0e-13; at 1e15, 1.3e-2 against
    9.6e-14. Through the bank, the square root's predictions at ``p0`` of
    1e8, 1e10 and 1e12 came within 2.4e-9, 2.4e-11 and 2.4e-13 of those at
    1e14, the ``1 / p0`` the prior's own weight moves them by; the
    covariance form's stayed up to 0.19 apart at every one.

    **Standardizing.** The filter's state is held in the coordinates of an
    *anchor*, the moments at its last re-map, and maps the process noise and
    the priors, defined in the moments as they stand, into them exactly.
    When the moments drift from the anchor (a scale by more than twice or
    less than half, a mean by more than the anchor's scale), the state
    follows them to their new coordinates, ``b <- A b`` and
    ``P <- A P A'``:

    .. code-block:: text

        A_00 = 1,    A_0i = (m_i - m^a_i) / s^a_i,    A_ii = s_i / s^a_i

    with ``m^a``, ``s^a`` the anchor and ``m``, ``s`` the moments after the
    row. So a change in the moments moves no prediction, no predictive
    variance and no coefficient in the original units, from the first row.
    A feature reaches the filter only through its standardized value
    ``z``, so features scaled by a power of two predict the same to the
    bit. A filter read
    through the moments as they stand, the alternative, lets a finite
    half-life's own wander in the moments move every prediction. It was
    measured on ``y = 2 x + e`` with one feature ``x ~ N(0, 1)``, R²
    0.99998, 30,000 rows and both half-lives 50. Over rows 10,000 to 30,000
    its out-of-sample error stood 223 noise variances above the noise (the
    median of seeds 0 to 4, which ranged from 212 to 244). This filter stands 0.014 above it
    there, as the unstandardized one does. The anchor gives the same filter,
    to rounding, as a re-map on every row, and is a fifth cheaper a row at
    ten features (221 against 277 nanoseconds). A re-map past what a stretch
    of data makes (a scale by more than 1024 times either way, or a mean by
    more than 1024 of the anchor's scale) is not followed, and the state is
    read in the new coordinates.

    **Reversion.** A reverting coefficient's drift ``Q_i * r_i(D) ** 2`` is
    ``Q_i * D ** 2`` for a gap well under ``kappa_i``, and never more than
    ``Q_i / theta_i ** 2``. At observations one clock unit apart it is short
    of ``Q_i`` by about ``ln 2 / kappa_i`` of it. At observations ``D`` apart, a
    reverting coefficient reaches the stationary prior variance
    ``Q_i * r_i(D) ** 2 / (1 - phi_i ** 2)`` with
    ``phi_i = 2 ** (-D / kappa_i)``, a stationary AR(1) instead of an
    unbounded walk. The reversion runs on every row, a row that observes
    nothing included, being the same over a gap however it is cut. A
    prediction propagates the state by the same ``Phi`` over the row's clock
    gap. :meth:`rheon.polars.ModelBank.predict` moves the coefficients by
    ``Phi`` over the clock distance from the last learned row, capped by
    ``gap_cap``. So however far past the data, a coefficient has shrunk by
    ``2 ** (-gap_cap / kappa_i)`` at most, and the prediction is the
    intercept alone only where ``gap_cap`` spans several of the slopes'
    reversion half-lives and the intercept's is ``inf``.

    **A shared P.** ``P``'s recursion reads neither the targets nor the
    noise, since it is held in units of the noise. Every target observed on
    a row takes its gain from ``P`` as the row finds it, and ``P`` takes the
    row once, so the order of the targets moves no prediction. Where a
    target is null on rows another is present on, that target's memory
    follows the others' rows. A target first seen after the others has its
    intercept filled by its first innovation, as the diffuse prior fills
    each target's. Under ``q`` the shared ``P`` measures it against the mean
    noise over the targets that have one. A target's ``se_coef`` reads ``P``
    times its own noise.

    **Standard errors.** ``se_coef`` reads the filter's own covariance
    ``R_j * P_j``, and ``error_inflation`` is ``sqrt(1 + z' P⁻ z)``, with
    ``P⁻`` the covariance in units of the noise that the row's update starts
    from, after the reversion and the drift. Both are exact *under the
    model*, which says the coefficients walk at the noise
    ``coef_half_life`` implies. On a truth that does not move, that walk is
    variance the coefficients do not have. The standard errors then come
    out about ``sqrt(2)`` too large: a mean squared error over
    ``se_coef ** 2`` of 0.47 to 0.50 at ``coef_half_life`` 50 and 200,
    against 0.95 to 1.06 on a walk of the implied variance. And with ``R``
    the EW variance of the out-of-sample residuals, which already carry the
    estimation error, ``error_inflation`` counts that error twice: the mean
    squared error over ``R * error_inflation ** 2`` is 0.96 at
    ``coef_half_life`` 50.
    """
    model: dict[str, Any] = {
        "type": "kalman",
        "coef_half_life": coef_half_life,
        "q": q,
        "obs_var": obs_var,
        "p0": p0,
        "share_p": share_p,
        "revert_half_life": revert_half_life,
        "standardize": standardize,
        "covariance_form": covariance_form,
    }
    words = ("covariance", "square_root")
    if covariance_form not in words:
        raise ValueError(
            words_refusal(f"spec {quoted(name)}", "covariance_form", covariance_form, words)
        )
    # Written only where it is not the default, as the Rust spec skips it there.
    if covariance_form == "covariance":
        del model["covariance_form"]
    if forecast_spans is not None:
        # A dict is `[name, span]` pairs, as the Rust spec reads them.
        who = f"spec {quoted(name)}"
        model["forecast_spans"] = (
            [[k, duration_text(v, who, "forecast_spans")] for k, v in forecast_spans.items()]
            if isinstance(forecast_spans, dict)
            else list(forecast_spans)
        )
    return _common(name, model, targets=targets, features=features, **common)
