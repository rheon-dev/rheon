"""The volatility model's builder, :func:`garch`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def garch(
    name: str,
    *,
    targets: TargetList,
    vol: Literal["garch", "gjr", "egarch", "figarch"] = "garch",
    dist: Literal["normal", "t", "skewt"] = "normal",
    truncation: int | None = None,
    variance_targeting: bool = False,
    start_rows: int | None = None,
    start_span: float | Duration | None = None,
    horizon: int = 1,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A conditional variance for each target, from the GARCH family, fitted
    online by recursive quasi-maximum likelihood.

    Use it on returns, or on any zero-mean residual whose size clusters in
    time: each row's variance is forecast from the rows before it, and the
    process's parameters are learned one row at a time, from a start fitted
    by batch likelihood on the first rows. Each target is its own series,
    with its own parameters.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``min_weight`` and ``group``, are in :mod:`rheon.spec`; ``half_life`` and
    ``lam`` are refused, since the recursion is per observation and the
    estimate weighs every row it has learned. A row's ``weight`` scales its
    evidence: its score, its information and its term in the start's
    likelihood. Scaling every weight changes nothing but ``weight_sum``. A
    row of weight 0, or whose target is null, steps nothing: the next row's
    lags are those of the last row learned, as if it were not there; it is
    still read.

    This model's own parameters:

    ``vol``
        The process: ``"garch"``, the default, GARCH(1,1); ``"gjr"``, with a
        leverage term for negative returns; ``"egarch"``, in the logarithm
        of the variance; ``"figarch"``, fractionally integrated, with a long
        memory of the squared returns.
    ``dist``
        The innovations: ``"normal"``, the default; ``"t"``, Student's t,
        whose degrees of freedom are learned; ``"skewt"``, Hansen's skewed
        t, which learns a skew too. The t takes ``vol="garch"`` and
        ``"gjr"``, the skewed t ``"garch"`` alone: under skewed innovations
        GJR's persistence reads ``E[z² 1(z < 0)]``, not the half of ``E[z²]``
        the forecast uses.
    ``truncation``
        FIGARCH's lags: its weights over the last this many squared returns.
        Default 1,000, as ``arch``'s; 8 bytes a lag in each group.
    ``variance_targeting``
        Set the level by the running mean square, ``ω = v̄ (1 - α - β)``,
        and learn ``α`` and ``β`` alone, with a Huber-clipped score:
        ``vol="garch"`` with normal innovations only. It is a different
        estimate from the default's, ``ω`` a moment rather than a
        likelihood one, and it needs returns with a finite fourth moment. It
        is not the safer choice: on persistent series (``α + β = 0.98``) it
        forecasts worse than the default, since its level rides on one
        running mean the whole stream shares. Default False.
    ``start_rows``
        The rows the start buffers and fits, each target's own; they report
        nothing. Default 2,000, at least 50, and at least 2,000 for
        ``vol="figarch"``. A start whose fit does not converge, or sits on a
        bound of the process's parameters, waits for twice the rows and
        refits, at most twice, while ``max_state_bytes`` holds the doubled
        buffer. Refused beside ``start_span``.
    ``start_span``
        The start as a span of the clock in place of ``start_rows``, in
        clock units: a number of the clock column's units, or a duration
        such as ``"1d"`` on a temporal clock. It closes at the first row at
        or past the span from its first row, once it holds 50 rows; a refit
        waits for twice the span. How many rows that is, the data decides,
        so its bytes are checked as the rows go in (see
        ``max_state_bytes``).
    ``horizon``
        Report the variance this many rows ahead too, as ``forecast_<t>``.
        Default 1, the variance alone. ``vol="egarch"`` takes 1 only: its
        forecast past one row has no closed form.
    ``max_state_bytes``
        The bytes a group's state may hold, the start's buffer included (16
        bytes a row until it is fitted). A spec whose start or FIGARCH ring
        is past it is refused. A ``start_span`` start is checked as the rows
        go in: the first row to take a group past half of it raises a
        warning, once for the group, and the row that would take it past
        the cap stops the run with an error naming the group and the bytes.
        Default 256 KB, ``262144``; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#garch
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#garch>`_
    gives each field's type and when it is null:

    ``variance_<t>``
        Per target: the variance of the row's return given the rows before
        it, read before the row is learned. Null before the target's start
        is fitted, and below ``min_weight``.
    ``std_resid_<t>``
        Per target: the row's return over the square root of its variance,
        null where either is.
    ``forecast_<t>``
        Per target, with ``horizon`` above 1: the variance ``horizon - 1``
        rows past this one, from the same rows as ``variance_<t>``.
    ``weight_sum``
        The weight the model has learned before the row: every learned
        row's weight, the start's included, with no decay.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).
    ``coef``
        Per target, after the row: ``omega``, then ``alpha``, ``gamma``
        (``gjr``, ``egarch``) and ``beta``, or FIGARCH's ``omega``, ``phi``,
        ``d`` and ``beta``; then ``nu`` for a t, and ``lambda`` for a
        skewed t. Null before the start. :func:`coef_index` names them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for ``vol`` or ``dist`` not one of the words above, a t on
    ``egarch`` or ``figarch``, a skewed t on ``gjr``, ``variance_targeting``
    on any form but ``garch`` with normal innovations, ``truncation`` on a
    process but FIGARCH or below 1, ``start_rows`` below its floor or beside
    ``start_span``, ``start_span`` not finite or not above 0, ``horizon``
    below 1 or above 1 on ``egarch``, ``half_life`` or ``lam``, and a state
    past ``max_state_bytes``, or a ``max_state_bytes`` of 0 or past 64 MiB.

    .. rubric:: The update

    With ``s_t`` the variance of row ``t``'s return ``e_t`` given the rows
    before it, and ``z_t = e_t / sqrt(s_t)``:

    .. code-block:: text

        garch    s_t = ω + α e²_{t-1} + β s_{t-1}
        gjr      s_t = ω + α e²_{t-1} + γ 1[e_{t-1} < 0] e²_{t-1} + β s_{t-1}
        egarch   ln s_t = ω + α (|z_{t-1}| - sqrt(2/π)) + γ z_{t-1} + β ln s_{t-1}
        figarch  s_t = ω / (1 - β) + Σ_{j=1..M} λ_j e²_{t-j},
                 λ_1 = φ - β + d,  λ_j = β λ_{j-1} + δ_j - φ δ_{j-1},
                 δ_j = d Π_{i=2..j} (i - 1 - d) / i

    Before the first row every lag is ``arch``'s backcast, the
    ``0.94^i``-weighted mean of the first 75 squared returns. Each learned
    row of weight ``w_t`` steps the parameters by the score of its
    log-likelihood ``l_t = log f(z_t) - ½ ln s_t``, ``g_t = ½ w ∂ln s_t / ∂θ``
    (``w = z² - 1`` under the normal, a bounded ``w`` under a t), with the
    sensitivity ``∂ln s_t / ∂θ`` carried by its own recursion:

    .. code-block:: text

        a_t = w_t / W_t          W_t the weight learned, the start's included
        H  ← H + a_t (I_t - H)   I_t = ½ ∂ln s ∂ln sᵀ, or the score's outer product under a t
        θ  ← Proj_H(θ + a_t H⁻¹ g_t)

    ``Proj_H`` projects onto the process's admissible set, ``arch``'s with
    the persistence at most ``1 - 1e-4``, in ``H``'s own metric. A t's
    ``ν`` is stepped as ``ln(ν - 2)``, so no step crosses 2. EGARCH's guard
    keeps the filter that runs contracting: ``α ≥ |γ|`` and ``α ≤ 1``, its
    variance within a factor ``1e6`` of the 0.94-weighted mean square, as
    ``arch``'s is, and the mean of ``ln |β - (α |z| + γ z)/2|`` over the
    residuals the filter has met held below 0, a step that would break it
    cut back along the step. It agrees with ``arch`` wherever ``arch``'s own
    filter contracts, and carries no proof that the estimate it guards is
    consistent. The state is ``8 * (2k + m + n(n + 1)/2 + 9)`` bytes a target
    for ``k`` process parameters, ``m`` shape parameters and ``n`` stepped
    coordinates, EGARCH's 512 bytes of bins beside them, plus FIGARCH's
    ring, ``8 M``, whatever the stream's length.

    .. rubric:: Example

    This code simulates a GARCH(1,1) at ``(ω, α, β) = (0.05, 0.08, 0.9)``
    and prints the fit after 20,000 rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        e, s, r = 0.0, 2.5, []
        for z in rng.standard_normal(20_000):
            s = 0.05 + 0.08 * e * e + 0.9 * s
            e = np.sqrt(s) * z
            r.append(e)
        pl.DataFrame({"r": r}).write_parquet("returns.parquet")

        model = rp.spec.garch("g", targets=["r"])
        bank = rp.ModelBank([model])
        bank.fit(pl.scan_parquet("returns.parquet"))
        coef = {row["term"]: row["coef"] for row in bank.coef().iter_rows(named=True)}
        print(f"alpha: true 0.08, found {coef['alpha']:.2f}")
        print(f"beta: true 0.90, found {coef['beta']:.2f}")

    .. code-block:: text

        alpha: true 0.08, found 0.07
        beta: true 0.90, found 0.91

    .. rubric:: See also

    - :func:`ew_cov`, for an exponentially weighted variance with a fixed
      decay rather than a fitted one.
    - :func:`hawkes`, for the clustering of events rather than of sizes.

    .. rubric:: Notes

    **The estimator.** It is the recursive prediction-error method: with
    the gain ``1/W`` and a batch start, its estimate agrees with the batch
    fit of the same rows to first order, so the gap between the two,
    measured in the batch fit's standard errors, shrinks as rows arrive.
    Against ``arch``'s batch fit of 10,000 rows, over 50 simulated series a
    form, every parameter was within two of ``arch``'s standard errors on
    46 to 50 of them, and the out-of-sample log-likelihood lost to
    ``arch``'s path was at its expected size, ``(k/2)(ln(n/n₀) + 1)`` for
    ``k`` parameters and ``n₀`` rows of start.

    **The projection.** A Newton step projected onto the bounds in the
    Euclidean metric is no ascent where the bounds hold parameters the
    likelihood barely tells apart (``α + β`` near 1, FIGARCH's ``φ`` and
    ``β``); projected in ``H``'s own metric it is.

    **The start.** Its rows report nothing, and its fit is made by the
    model's own projected Newton on their likelihood, from a small grid.
    2,000 rows keep a series from starting far enough off that the
    ``1/W`` gain takes the rest of the stream to recover; a FIGARCH needs
    them for its ring of 1,000 lags to hold returns rather than the
    backcast.
    """
    model: dict[str, Any] = {"type": "garch"}
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if vol != "garch":
        model["vol"] = vol
    if dist != "normal":
        model["dist"] = dist
    if truncation is not None:
        model["truncation"] = truncation
    if variance_targeting:
        model["variance_targeting"] = True
    if start_rows is not None:
        model["start_rows"] = start_rows
    if start_span is not None:
        model["start_span"] = start_span
    if horizon != 1:
        model["horizon"] = horizon
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    return _common(name, model, targets=targets, features=[], **common)
