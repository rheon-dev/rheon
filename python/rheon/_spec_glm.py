"""The generalized linear model's builder, :func:`glm`: a part of
:mod:`rheon._spec` in a file of its own, keeping that one under
``tests/test_repo_hygiene.py``'s 250 KB cap for a source file.
:mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def glm(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    family: Literal["poisson", "gamma", "binomial", "tweedie"] = "poisson",
    link: Literal["log", "logit", "inverse", "power"] | None = None,
    var_power: float | None = None,
    ridge: float | None = None,
    refit_rows: int | None = None,
    refit_span: float | Duration | None = None,
    max_state_bytes: int | None = None,
    step_cap: float | None = None,
    refit_budget: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A generalized linear model, fitted one row at a time by reweighted least squares.

    Use it for a target whose spread grows with its mean, or that has a
    floor or a ceiling: counts (``family="poisson"``), positive amounts
    such as prices, sizes and durations (``"gamma"``), proportions and 0/1
    outcomes (``"binomial"``), and amounts with an exact zero, such as an
    insurance claim or a day's rainfall (``"tweedie"``). The prediction is
    the mean on the target's own scale, so it stays inside the family's
    range. :func:`ewridge` fits the mean as a straight line, and
    :func:`sgd` with ``loss="poisson"`` fits the same Poisson model by a
    gradient step in O(k) time, where this model takes O(k²) a row and a
    Newton step.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`rheon.spec`. A row's ``weight`` counts the
    row that many times in the likelihood, as statsmodels' ``freq_weights``
    does, and ``min_weight`` counts each target's own rows in the family's
    range.

    This model's own parameters:

    ``family``
        The target's distribution: ``"poisson"`` (counts, the default),
        ``"gamma"`` (amounts above 0), ``"binomial"`` (values from 0 to 1)
        or ``"tweedie"`` (amounts with an exact zero, at ``var_power``
        between 1 and 2; amounts above 0 above 2). A target value outside
        the family's range is refused with the row named.
    ``link``
        How the mean ``μ`` is read from ``η = x'β``:
        ``"log"`` (``μ = e^η``), ``"logit"`` (the binomial's), ``"inverse"``
        (``μ = 1/η``, the Gamma family's canonical link) or ``"power"``
        (``μ = η^(1/(1 - p))``, the Tweedie family's canonical link). Each
        family takes its own: the Poisson the log, the Gamma the log or the
        inverse, the binomial the logit and the Tweedie the log or the power.
        Default the log, and the logit for the binomial. The inverse and
        power links give a mean only where ``η > 0``: elsewhere the row is
        not predicted. They need the refit and ``fit_intercept``, since
        every coefficient at 0 gives no mean. After the refit each row is
        learned by an exact step that lands its own ``η`` above 0 (*The
        update*).
    ``var_power``
        The Tweedie family's power ``p``, the exponent of its variance
        ``μ^p``: above 1 and not 2. Default 1.5. Refused for another family.
    ``ridge``
        ``R``, the penalty ``R/2 * |β|²`` on the sum of the rows' half
        deviances, above 0. Default 1: next to a few hundred rows it moves
        the fit by a small share of a standard error, and it keeps every
        solve defined before the coefficients are determined. The intercept
        is penalized too. Under the inverse and power links the penalty is
        not free of the target's units: ``β`` is on the scale of ``1/μ`` (or
        ``μ^(1 - p)``), so a target in units a hundredth as large has
        coefficients a hundred times as large (``100^(p - 1)`` times under
        the power link), which the same ``R`` pulls toward 0 far harder: at
        3,000 rows of a Gamma, 0.06 standard errors from the unpenalized
        fit with ``η`` near 0.5, 1.1 near 5 and 36 near 50. Scale such a
        target so that ``η`` is of order 1, or lower ``ridge``.
    ``refit_rows``
        The learned rows (each of positive weight, with its target in the
        family's range) after which each target's first rows are fitted to
        convergence, once, as a batch fit does (*The start* in Notes).
        Default ``max(200, 10 * k)``, ``k`` the features plus one, up to 100
        features, held to the rows the group's cap below holds, and off
        above 100 features or where the cap holds fewer than ``2 * k``
        rows. ``0`` turns the refit off. A group's buffers hold ``8 *
        targets * refit_rows * (features + 2)`` bytes until the refit runs,
        at most ``max_state_bytes``: a count past that is refused, naming the
        bytes.
    ``refit_span``
        The same refit over a span of the clock in place of a count of rows:
        each target's rows from its first learned row through the first row
        at least ``refit_span`` after it, once they hold a row per
        coefficient. A number of the clock's units, or of rows where the
        spec has no clock, or a duration such as ``"1h"`` on a temporal
        clock; finite and above 0. The span is read off the rows' exact
        stamps (integer nanoseconds on a temporal clock), and a row of
        weight 0 is no row of it, so on a clock of one unit a row it holds
        ``refit_span + 1`` rows: the refit of ``refit_rows = refit_span +
        1``, to the bit. How many rows a span holds the data decides, so the
        buffers are checked against ``max_state_bytes`` as they fill.
        Refused beside ``refit_rows``. Default none.
    ``max_state_bytes``
        The bytes a group's refit buffers may hold, a count's when the spec is built. Under
        ``refit_span`` the data decides how many rows the buffers hold, so they
        are checked as they fill: the first row to take a group past half of
        it raises a warning, once for the group, naming the group and the
        bytes, and the row that would take it past the cap stops the run
        with an error naming the group and the bytes, the bank having
        learned part of the chunk. Default 256 KB, ``262144``; at most
        64 MiB, ``67108864``.
    ``step_cap``
        The most one learned row may move its own ``η``, above
        0, or ``inf`` for no cap. Default: none where the refit runs, and 1
        where it does not (*The start* in Notes).
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups
        together, above 0, or ``inf`` for none. Default 320. Each new group
        reserves the most its targets' buffers can hold when it first
        appears. The group that would cross the bound, and every new group
        after it, gives its refit up and keeps its one-pass fit, which
        ``bank.summary()``'s ``refit_given_up`` counts. Under the log and
        logit links such a group's rows are held to the start's step cap of
        1 (or ``step_cap``, where smaller) for good; under the inverse and
        power links the exact step learns them from every coefficient at 0,
        held to ``step_cap`` alone.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#glm
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#glm>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the predicted mean ``μ``, on the target's scale, and
        ``y - μ`` where the target is not null.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's predictions were withheld, null where they were not.
    ``coef``
        Per target, the intercept (unless ``fit_intercept=False``) then one
        entry per feature (:func:`coef_index`), on the link's scale: under
        the log link a coefficient is the log of the factor a unit of its
        feature multiplies the mean by.

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), and ``ValueError``
    naming the problem for:

    - a ``family`` or ``link`` that is not one of its words, or a link the
      family does not take;
    - a ``var_power`` not above 1, or of 2, and one given for a family other
      than the Tweedie;
    - a ``ridge`` not above 0 or not finite, and a ``step_cap`` not above 0;
    - a ``refit_rows`` whose buffers would pass ``max_state_bytes``, and the
      inverse or power link with
      ``refit_rows=0`` (and no ``refit_span``) or ``fit_intercept=False``;
    - a ``refit_span`` not finite and above 0, and one beside ``refit_rows``;
    - a ``max_state_bytes`` of 0 or above 64 MiB, and a ``refit_span``
      whose rows pass it, as the rows go in;
    - a ``refit_budget`` not above 0;
    - a target value outside the family's range, as the chunk is read.

    .. rubric:: The update

    Per row and target, ``x̃`` is ``[1, x]`` (``x`` without an intercept),
    ``w`` the row's weight, ``λ = 0.5 ** (Δ / half_life)`` the decay over
    the row's clock step ``Δ``, ``V`` the family's variance and ``g`` the
    link:

    .. code-block:: text

        η = x̃'β                     the fit before the row
        μ = g⁻¹(η)                   the prediction, before the update
        W = 1 / (g'(μ)² V(μ))        the Fisher weight
        z = η + (y - μ) g'(μ)        the working response
        G = λ G + w W x̃x̃'
        β = β + (G + R I)⁻¹ w W (z - η) x̃

    ``G`` is per target, since ``W`` depends on the target. Under the log
    and logit links ``η`` is clipped to ``±30`` before ``μ`` is formed.

    Under the log link, on each row the start's cap does not hold (no
    refit, or once it has run), an upward step that would carry the row's
    own ``η`` past ``ln y`` is scaled, as ``step_cap`` scales one, so that
    ``η`` lands instead on the one root ``u`` in ``(η, ln y]`` of the row's
    own penalized deviance, ``w f(u) + (u - η)²/(2q)``, ``f`` the half unit
    deviance at ``μ = e^u``, ``p`` the variance power (1 for the Poisson, 2
    for the Gamma) and ``δ`` the step's move of ``η``:

    .. code-block:: text

        h(u) = u - η + w q e^((1-p)u) (e^u - y) = 0
        q = x̃'(λG + R I)⁻¹x̃                      before the row
        β = β + ((u - η)/δ) (G + R I)⁻¹ w W (z - η) x̃

    Where ``step_cap`` scales the same step, the smaller scale wins. Every
    other row takes the step above unchanged.

    Under the inverse and power links, once the refit has run or been given
    up, each row instead minimises its own penalized deviance exactly, ``β =
    argmin w d(y, μ(x̃'b))/2 + (b - β)'(G + R I)(b - β)/2`` (``G`` decayed,
    ``d`` the family's unit deviance). With ``P = (G + R I)⁻¹``, ``q =
    x̃'P x̃`` and ``m = 1/(p - 1)`` (``p = 2`` for the Gamma), the row's
    ``η`` moves to the one root ``u > 0`` of ``u - η + (w q/(p - 1)) (y -
    u^(-m)) = 0``:

    .. code-block:: text

        β = β + ((u - η)/q) P x̃
        G = λ G + w min(W(η), W(u)) x̃x̃'      W(u) alone where η ≤ 0

    The root lies between ``η`` and the link at ``y``, so the step never
    passes the target, and it is above 0 from any ``η``: a target far from
    its mean cannot push the fit out of the link's domain, where the rows
    after it would have no mean and teach nothing. For the Gamma ``u = (a +
    √(a² + 4wq))/2``, ``a = η - w q y``; for the Tweedie it is found by
    Newton's method. ``step_cap`` holds ``|u - η|``.

    .. rubric:: Example

    This code fits a count whose log mean is ``0.5 + 0.3 x``, and prints
    the coefficients this model finds beside the truth:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(5)
        x = rng.standard_normal(20_000)
        y = rng.poisson(np.exp(0.5 + 0.3 * x)).astype(float)
        pl.DataFrame({"x": x, "y": y}).write_parquet("counts.parquet")

        glm = rp.spec.glm("pois", targets=["y"], features=["x"], half_life=float("inf"))
        bank = rp.ModelBank([glm])
        bank.fit(pl.scan_parquet("counts.parquet"))  # learn every row, keep no output
        coef = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
        print("true: intercept 0.50, x 0.30")
        print(f"found: intercept {coef['intercept']:.2f}, x {coef['x']:.2f}")

    .. code-block:: text

        true: intercept 0.50, x 0.30
        found: intercept 0.50, x 0.30

    .. rubric:: See also

    - :func:`sgd` with ``loss="poisson"`` or ``"logistic"``, a gradient
      step on the same likelihood in O(k) time a row.
    - :func:`ftrl`, a sparse logistic regression.
    - :func:`ewridge`, the least-squares fit, for a target whose spread does
      not grow with its mean.

    .. rubric:: Notes

    **What the fit is.** Each row takes one Fisher-scoring step on the
    penalized likelihood of every row so far, each past row's curvature
    ``W`` frozen at the fit that scored it. Without decay that is the
    penalized batch fit's IRLS, one row at a time: measured over 50 seeds
    at 20 coefficients, the largest gap to statsmodels' ``GLM`` and to glum
    on the same rows, ridge matched, was 0.47, 0.33 and 0.61 standard errors
    at 10,000 rows (Poisson, Gamma of shape 2, Tweedie at ``p = 1.5`` and
    dispersion 2) and 0.20, 0.16 and 0.31 at 100,000. The gap is a bias of
    the frozen weights, which shrinks as ``1/n`` against the standard
    error's ``1/√n``, and grows with the dispersion as its square root. Its
    error against the true coefficients equals the batch fit's to 0.01
    standard errors on average. Under decay ``G`` decays and the ridge does
    not: each step's penalty is in part one toward 0 that ages as the first
    row does, and in part a pull toward the fits before the row, which
    vanishes as the fit stops moving. So a row of weight 0 moves no
    coefficient, and the fit tends to the decayed likelihood's maximum,
    unpenalized: statsmodels' ``GLM`` with the decayed weights as
    ``freq_weights``. The refit fits its rows at their decayed weights
    under the whole ridge.

    **The start.** Before about ``k`` rows the fit is underdetermined and
    one row can move ``η`` by tens. With no remedy the recursion then runs
    away (a Gamma row scored at a very negative ``η`` has a step of order
    ``y e^|η|``) or sticks (a Poisson row scored at a too-large mean enters
    ``G`` with that mean's weight, and without decay starves its direction
    for good): 34 and 23 of 50 seeds at 20 coefficients. The refit keeps
    each target's first ``refit_rows`` learned rows, fits them to
    convergence by penalized IRLS with step halving from the intercept at
    the link of their mean, and rebuilds ``G`` at that fit. The buffer is
    then freed. While it fills, each row's step is held to a cap of 1 on its
    own ``η`` (or ``step_cap``, where smaller), so the predictions before
    the refit are sound; the refit replaces whatever those rows left. The
    step cap needs no buffer and held every seed too, but without decay its
    capped rows' weights stay in ``G``: its largest gap at 100,000 rows was
    2.9 times the refit's for the Poisson family and 1.8 times for the
    Gamma. Under decay the two are the same.

    **The target guard.** Under the log link a downward step moves ``η``
    by less than 1, but an upward one grows as ``y μ^(1-p)`` without bound
    as ``μ`` falls. With neither the refit nor a cap, over 100 seeded
    streams of 300 rows at 3 coefficients, the step without the guard
    passed ``|β|`` 10 on 0, 17, 0, 1, 7, 35 and 64 of them (Poisson, Gamma,
    Tweedie at ``p`` 1.2, 1.5, 1.8, 2.5 and 3); under the default spec, after
    its refit, one target 1000 times its draw did on 14, 41, 17, 38 and 48
    of 100 (Poisson, Gamma, Tweedie at 1.5, 2.5 and 3). With the guard none
    did. It is a floor, not a remedy: without a refit the first rows'
    curvature stays in ``G``, and the refit or a cap of 1 is what makes the
    fit accurate. The logit link needs no guard: its step moves ``η`` by
    ``w (y - μ)`` times ``x̃'(G + R I)⁻¹x̃``, at most ``w |x̃|²/R`` either
    way, and over 960 seeded streams down to means of ``e^-7`` it passed
    10 on none.

    **The inverse and power links after the refit.** Each row's exact step
    keeps its own ``η`` above 0, so a target far from its mean no longer
    leaves the rows after it without a mean. Against statsmodels' ``GLM``
    on 16 adversarial streams of 3,000 rows, 20 seeds each, the median seed
    was 0.03 to 0.21 standard errors off on the streams without a far
    target (0.58 at 20 features), and the worst 0.11 to 0.86. With one
    target 1000 times its mean the median seed was 0.8 to 1.6 off and the
    worst 6.2, with at most 16 null predictions after the refit, where the
    Fisher step left 58 standard errors and up to 933 null predictions.

    **The state** is ``k(k+1)/2 + k + 1`` numbers a target and, until its
    refit runs, ``refit_rows (features + 2)`` more: 35 KB a target at 20
    features. The buffers are the one part whose size is not the model's
    shape, and a group's are held to ``max_state_bytes`` (256 KB by default). There is
    no window: removing a row exactly would need each row in the window,
    stored with the weight it was scored at.

    **The spans.** ``half_life``, ``gap_cap`` and ``refit_span`` take a
    number (rows, with no clock) or a clock duration, as every spec's do.
    ``refit_rows`` counts rows of positive weight. A count's buffer has a
    size known when the spec is built, and is held to its cap there; a
    span's holds the rows the data puts in it, and is held to the cap as
    they come.
    """
    model: dict[str, Any] = {
        "type": "glm",
        "family": family,
        "link": link,
        "var_power": var_power,
        "ridge": ridge,
        "refit_rows": refit_rows,
        "step_cap": step_cap,
        "refit_budget": refit_budget,
    }
    # Written only when given, as the Rust spec leaves them out when absent.
    if refit_span is not None:
        model["refit_span"] = refit_span
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    return _common(name, model, targets=targets, features=features, **common)
