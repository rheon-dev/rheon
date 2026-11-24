"""The distribution fit's builder, :func:`dist_fit`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def dist_fit(
    name: str,
    *,
    targets: TargetList,
    family: Literal["normal", "t", "skewt"] = "normal",
    quantiles: list[float] | None = None,
    es_levels: list[float] | None = None,
    floc: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A distribution fitted to each target, exponentially weighted on the
    clock, and the quantiles, shortfalls and scores it gives.

    Use it where the question is what values a series is likely to take,
    not what its mean will be: a band to hold a reading to, a value at risk
    and the expected shortfall past it, or a score of how surprising each
    row was. Each target is fitted on its own, by weighted maximum
    likelihood over its rows, the older rows weighing less by the decay:
    the normal; Student's t where the tails are heavier than the normal's;
    or Hansen's skewed t where they are heavier and unequal too. Every
    value a row reports is read from the fit before the row is learned, so
    each is a forecast of that row.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. A row's ``weight`` scales its term in the
    likelihood, as a weighted fit's weights do; scaling every weight changes
    nothing but ``weight_sum``. A row of weight 0 counts in no clock: the
    next row decays as if it were not there, and it is in no
    ``weight_sum``; it is still read, and its ``pit`` and ``log_score`` are
    reported. A target null on a row learns nothing from it, and its own
    weight ages by the row's decay. ``min_weight`` is each target's floor
    on its own weight, the rows it was present on; default 0, since a
    target with no weight, or one whose values have all been the same,
    reports nothing anyway.

    This model's own parameters:

    ``family``
        The distribution: ``"normal"``, the default, with ``loc`` its mean
        and ``scale`` its standard deviation, as ``scipy.stats.norm`` names
        them; or ``"t"``, Student's t, with ``loc``, ``scale`` and ``df``
        (its degrees of freedom) as ``scipy.stats.t`` names them, so its
        ``scale`` is not the standard deviation, which is ``scale √(df/(df
        - 2))`` where ``df > 2``; or ``"skewt"``, Hansen's skewed t as
        ``arch``'s ``SkewStudent`` has it, with ``loc`` its mean, ``scale``
        its standard deviation, ``df`` above 2 and ``lambda`` its skew,
        between -0.99 and 0.99, 0 the t and above 0 a longer right tail.
    ``quantiles``
        The levels of the quantiles to report, each above 0 and below 1,
        such as ``[0.05, 0.95]`` for a 90% band. Default none.
    ``es_levels``
        The levels ``α`` of the lower-tail expected shortfalls to report,
        each above 0 and below 1: the mean of the values below the
        ``α`` quantile. ``0.05`` is the mean of the worst 5%. Default none.
    ``floc``
        A fixed location in place of a fitted one, as ``scipy.stats``'
        ``fit(floc=...)``: ``loc`` is then ``floc``, and ``scale`` is the
        root of the weighted mean square about it. ``0.0`` fits the size of
        a return series whose mean is taken to be 0. Finite. Default none,
        which fits ``loc``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#dist_fit
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#dist_fit>`_
    gives each field's type and when it is null:

    ``loc_<t>``, ``scale_<t>``
        Per target: the fit's location and scale, from the rows before this
        one.
    ``df_<t>``
        Under ``family="t"`` or ``"skewt"``, per target: the fit's degrees
        of freedom, between 1 (2.05 under ``"skewt"``) and infinity (the
        normal's tail); null while the shape cannot be told from the
        normal's.
    ``lambda_<t>``
        Under ``family="skewt"`` only, per target, after ``df_<t>``: the
        fit's skew, between -0.99 and 0.99; null where ``df_<t>`` is.
    ``q<level>_<t>``
        Per target and level of ``quantiles``, the quantile at that level,
        as ``q0.05_y``.
    ``es<level>_<t>``
        Per target and level of ``es_levels``, the lower-tail expected
        shortfall at that level, as ``es0.05_y``.
    ``pit_<t>``, ``log_score_<t>``
        Per target: the fit's distribution function at the row's own value,
        and the log of its density there. Null where the target is.
    ``withheld_reason_<t>``
        Per target, why the target's fields above are null, as an ``Enum``:
        ``no_weight`` (nothing learned yet, or a weight the decay has aged
        to nothing), ``min_weight`` (under the floor) or
        ``degenerate_variance`` (every value learned was the same, or equal
        to ``floc``, so the scale is 0), each of which nulls every field.
        Under ``family="t"`` or ``"skewt"``, two more: ``not_converged``
        (the fit has not started, or is further from the maximum of its
        likelihood than noise alone would put it), which nulls every field,
        and ``shape_unidentified`` (the tail, and under ``"skewt"`` the skew
        with it, cannot yet be told from the normal's), which nulls every
        field but ``loc_<t>`` and ``scale_<t>``. Null where the target reports, and on a row the
        shared gates withhold (``withheld_reason``).
    ``weight_sum``
        The weight the model has learned before the row and before the
        row's own decay, every row's whatever its targets.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for a ``family`` other than ``"normal"``, ``"t"`` or
    ``"skewt"``, a level of
    ``quantiles`` or ``es_levels`` not above 0 and below 1, and a ``floc``
    that is not finite.

    .. rubric:: The update

    Per target, with ``w`` the row's weight and ``λ = 0.5 ** (Δ /
    half_life)`` the decay over the clock ``Δ`` since the last row that
    counted, the weighted moments step as Welford's:

    .. code-block:: text

        W' = λ W + w,   a = λ W / W',   b = w / W'
        d  = y - m
        m' = m + b d
        c' = a c + a b d²

    so that, with ``ω_i`` each learned row's weight times the decay since
    it, ``m = Σ ω_i y_i / Σ ω_i`` and ``c = Σ ω_i (y_i - m)² / Σ ω_i``:

    .. code-block:: text

        loc   = m                       (floc where it is given)
        scale = √c                      (√(c + m²), the moments of y - floc, under floc)

    the normal's weighted maximum-likelihood fit, whose variance has no
    degrees-of-freedom correction, as ``scipy.stats.norm.fit`` gives it.
    From that fit, before the row is learned, with ``Φ`` the standard
    normal's distribution function and ``φ`` its density:

    .. code-block:: text

        q_p    = loc + scale Φ⁻¹(p)
        es_α   = E[y | y ≤ q_α] = loc - scale φ(Φ⁻¹(α)) / α
        z      = (y - loc) / scale
        pit    = Φ(z)
        log_score = -ln scale - ½ ln 2π - ½ z²

    Under ``family="t"``, with ``η = 1/df`` (``η = 0`` is the normal),
    ``z = (y - loc) / scale`` and ``u = η z²``, the log density is

    .. code-block:: text

        log f(y) = -ln B(df/2, ½) - ½ ln df - ln scale - ((1 + η) / (2η)) ln(1 + u)

    and the fit is the maximum of ``Σ ω_i log f(y_i)`` over ``θ = (loc, ln
    scale, η)``, ``η`` between 0 and 1, taken one row at a time. It starts
    from the method of moments, with ``κ`` the weighted excess kurtosis:

    .. code-block:: text

        η₀ = κ / (4κ + 6)  (df = 4 + 6/κ) where κ > 0, else 0
        loc₀ = the weighted mean (floc where it is given)
        scale₀ = √(c (1 - 2η₀))

    on the first row whose moments pin ``ln scale`` to a standard error of
    1/4 by the t's own information, about 19 rows of equal weight. Each row
    after it, of weight ``w``, with ``W`` the weight before it at the row's
    decay and ``I(θ)`` the t's expected information per unit weight (which
    depends on ``scale`` and ``η`` alone), moves the fit to

    .. code-block:: text

        θ' = argmax  w log f(y; θ) - ½ W (θ - θ_prev)ᵀ I(θ_prev) (θ - θ_prev)

    the maximum of the rows before it, as the quadratic their information
    gives, with the row's own log density, found by Newton's method on the
    row's observed information and stopped when the gain another step
    promises is below ``10⁻¹⁵ w``. A row far out moves the tail as far as
    its own density says, not as far as one step along its score, which
    grows as ``z⁴`` in ``η``. From the fit, before the row is learned, with
    ``F`` and ``F⁻¹`` the t's distribution function and quantile at ``df``
    (by the regularized incomplete beta function):

    .. code-block:: text

        q_p    = loc + scale F⁻¹(p)
        es_α   = loc - scale (df + F⁻¹(α)²) / (df - 1) f(F⁻¹(α)) / α
        pit    = F((y - loc) / scale)
        log_score = log f(y)

    where ``f`` is the standard t's density; ``es_α`` is ``-inf`` at ``df =
    1``, where the mean does not exist.

    Under ``family="skewt"``, Hansen's skewed t, with ``z = (y - loc) /
    scale``, the log density is

    .. code-block:: text

        c = Γ((df + 1)/2) / (√(π (df - 2)) Γ(df/2))
        a = 4 λ c (df - 2) / (df - 1),    b = √(1 + 3λ² - a²)
        x = b z + a,    q = 1 - λ where x < 0, else 1 + λ
        log f(y) = ln b + ln c - ln scale - ((df + 1)/2) ln(1 + (x/q)² / (df - 2))

    ``x`` is a t of variance 1 whose left half is scaled by ``1 - λ`` and
    right half by ``1 + λ``, and ``a`` and ``b`` are its mean and standard
    deviation. The fit is the maximum of ``Σ ω_i log f(y_i)`` over ``θ =
    (loc, ln s, η, λ)``, ``η = 1/df`` between 0 and 1/2.05 and ``λ``
    between -0.99 and 0.99, taken one row at a time as the t's. ``s = scale
    √(1 - 2η)`` is the scale of the t inside the density, as
    ``scipy.stats.t`` has it: near ``df = 2`` the standard deviation and
    the tail are nearly one direction of the likelihood, and a step on ``ln
    scale`` there could move it far for almost nothing, where a step on
    ``ln s`` is the t's. It starts from the method of moments, the skew
    left to the likelihood:

    .. code-block:: text

        loc₀ = the weighted mean (floc where it is given)
        scale₀ = the weighted standard deviation (about floc under it)
        η₀ = κ / (4κ + 6) where κ > 0, else 0,    λ₀ = 0

    on the first row whose moments pin both ``ln scale`` and ``λ`` to a
    standard error of 1/4 by the skewed t's information, about 35 rows of
    equal weight. The weighted skewness is not matched: on so few rows it
    is noisy, and the family's skewness rises slowly in ``λ``, so the
    matched skew lands far out and the fit spends half-lives walking back
    from it. Each row after it moves the fit to the same ``θ'`` as the
    t's, with ``I(θ)`` the skewed t's expected information, which is
    closed, found as the t's is, by Newton's method on the row's observed
    information, also closed. From the fit, with ``G`` the distribution function
    of the t of variance 1 at ``df`` and ``P(m) = ∫ u g(u) du`` over ``u ≤
    m`` its lower partial moment:

    .. code-block:: text

        q_p    = loc + scale (x_p - a) / b
        x_p    = (1 - λ) G⁻¹(p / (1 - λ))        where p < (1 - λ)/2
               = -(1 + λ) G⁻¹((1 - p) / (1 + λ)) otherwise
        pit    = (1 - λ) G(x / (1 - λ))          where x < 0
               = 1 - (1 + λ) G(-x / (1 + λ))     otherwise
        es_α   = loc + scale (L_α / α - a) / b
        L_α    = (1 - λ)² P(x_α / (1 - λ))       where x_α < 0
               = a + (1 + λ)² P(-x_α / (1 + λ))  otherwise
        log_score = log f(y)

    ``L_α`` is the mean of ``x`` times the indicator of ``x ≤ x_α``. With
    ``κ = √(1 - 2η)`` and ``t_df`` the standard t's density, ``P(m) = -κ
    (1 + η τ²) / (1 - η) t_df(τ)`` at ``τ = m / κ``.

    .. rubric:: Example

    This code fits a normal to a series of mean 5 and standard deviation 2,
    and prints the fit and the 5% quantile the last row was given:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        y = 5.0 + 2.0 * rng.standard_normal(20_000)
        pl.DataFrame({"y": y}).write_parquet("levels.parquet")

        fit = rp.spec.dist_fit("d", targets=["y"], quantiles=[0.05], half_life=5_000.0)
        out = rp.ModelBank([fit]).fit_predict(pl.read_parquet("levels.parquet")).unnest("d")
        last = out.row(-1, named=True)
        print(f"loc {last['loc_y']:.2f}, scale {last['scale_y']:.2f}, q0.05 {last['q0.05_y']:.2f}")

    .. code-block:: text

        loc 4.99, scale 1.98, q0.05 1.74

    .. rubric:: See also

    - :func:`ew_cov`, for the weighted moments of several columns together,
      their covariances included.
    - :func:`garch`, for a variance that is forecast from the rows' own
      clustering, rather than averaged over a fixed decay.
    - :func:`quantile`, for a quantile that depends on features.

    .. rubric:: Notes

    **The estimate.** The recursion is exact: ``loc`` and ``scale`` are the
    weighted mean and standard deviation of every learned row at its
    decayed weight, to rounding, which statsmodels' ``DescrStatsW`` with
    ``ddof=0`` computes in two passes from the same weights. The update is
    the mean form, so a series far from 0 loses no digits to a sum of
    squares.

    **The tail.** The expected shortfall is the normal's closed form, the
    mean of the fitted distribution below its ``α`` quantile, so it moves
    with the fit's ``loc`` and ``scale`` alone. A series with heavier tails
    than the normal's has a shortfall past it; ``pit`` near 0 or 1 more
    often than its level says is how that shows.

    **The state** is four numbers a target, its weight, its mean, the part
    of the mean that rounding leaves out and its second moment, and the
    shared weight and clock: 32 bytes a target, whatever the stream's
    length or the levels asked for. Under ``family="t"`` each target keeps
    15 more: the sum of its squared weights, its third and fourth moments,
    ``θ`` (3), and the convergence check's gradient (3) and curvature (6),
    another 120 bytes. Under ``family="skewt"``, 21 more, the same with
    ``θ`` and the gradient of 4 and the curvature of 10: another 168 bytes.

    **The t's checks.** Its fit reports once it is near the maximum of the
    whole weighted likelihood. Beside the fit it keeps, for every row
    learned, the row's score ``g_i`` and observed information ``J_i`` at
    the fit it was learned at, summed at their decayed weights into the
    gradient ``G = Σ ω_i [g_i - J_i (θ - θ_i)]`` of the likelihood to
    second order in each row. ``Gᵀ (W I(θ))⁻¹ G`` is twice the log
    likelihood a step to that maximum would gain, and the fit is
    ``not_converged`` while it is above ``χ²_p(0.95) Σω²/W``, what sampling
    noise alone puts between the truth and the maximum at 95% (``p`` the
    parameters fitted and not held at a bound, ``Σω²/W`` the correction for
    unequal weights). Its tail is identified while the Wald statistic
    ``η² / Var(η)``, ``Var(η) = [(W I(θ))⁻¹]_ηη Σω²/W``, is above 2.7055:
    ``η = 0`` is the normal, on the boundary of the parameters, where the
    statistic's law is ``½χ²₀ + ½χ²₁`` (Self and Liang, 1987), whose 95%
    quantile that is. On a stream that is normal, the tail stays
    ``shape_unidentified`` on most rows (18 streams of 20 at their end, in
    a test of 3,000 rows each), and ``family="normal"`` is the fit to
    read.

    **The skewed t's checks** are the t's, ``p`` counting up to 4, but for
    the shape. Its shape is identified while the joint Wald statistic ``sᵀ
    V⁻¹ s``, ``s = (η, λ)`` and ``V`` the ``(η, λ)`` block of ``(W
    I(θ))⁻¹ Σω²/W``, is above 5.1384. ``(η, λ) = (0, 0)`` is the normal:
    ``η = 0`` is on the boundary of the parameters, and Hansen's skew is
    regular at ``λ = 0``, inside them, where the information is positive
    definite. With one parameter on the boundary and one inside, the
    statistic's law is ``½χ²₁ + ½χ²₂`` (Self and Liang, 1987), whose 95%
    quantile that is. A skew the test finds with a normal tail reports
    ``df`` as infinity.
    """
    model: dict[str, Any] = {
        "type": "dist_fit",
        "family": None if family == "normal" else family,
        "quantiles": quantiles,
        "es_levels": es_levels,
        "floc": floc,
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=targets, features=[], **common)
