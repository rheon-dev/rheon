"""The dynamic correlation model's builder, :func:`dcc`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import _SKIPPED_WHEN_ABSENT, TargetList, _checked, _common

# Every model key of `dcc`'s is skipped when absent (`_SKIPPED_WHEN_ABSENT`),
# kept here rather than in `_spec.py`, which stands at the size a source file
# is held to.
_SKIPPED_WHEN_ABSENT["dcc"] = frozenset(
    {
        "asymmetric",
        "margins",
        "vol",
        "dist",
        "truncation",
        "variance_targeting",
        "margin_start_rows",
        "margin_start_span",
        "a",
        "b",
        "g",
        "start_rows",
        "start_span",
        "max_state_bytes",
    }
)


@_checked
def dcc(
    name: str,
    *,
    targets: TargetList,
    asymmetric: bool = False,
    margins: Literal["garch", "none"] = "garch",
    vol: Literal["garch", "gjr", "egarch", "figarch"] = "garch",
    dist: Literal["normal", "t", "skewt"] = "normal",
    truncation: int | None = None,
    variance_targeting: bool = False,
    margin_start_rows: int | None = None,
    margin_start_span: float | Duration | None = None,
    a: float | None = None,
    b: float | None = None,
    g: float | None = None,
    start_rows: int | None = None,
    start_span: float | Duration | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """The full correlation matrix of several return series, moving every
    row, with the parameters that drive it given or learned online.

    It is dynamic conditional correlation, DCC, or its asymmetric form,
    ADCC. Use it on returns whose correlations change with time, and rise after
    joint falls: each series is standardized by its own GARCH variance,
    and the correlation of each pair is forecast from the rows before it.
    The two or three numbers that drive the recursion are learned one row
    at a time by quasi-likelihood, from a start fitted on the first rows.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``min_weight`` and ``group``, are in :mod:`rheon.spec`. ``half_life``
    or ``lam`` is required, ``half_life="inf"`` for no decay: a decay acts
    on the clock, on the targets ``Q̄`` and ``N̄``, the weight and the
    estimator's gain, so the model tracks a correlation level that drifts.
    The recursion itself steps once per row, however long the gap before
    it. A row's ``weight`` scales its evidence. A row of weight 0, or with
    any target null, steps nothing, and is still read.

    This model's own parameters:

    ``asymmetric``
        The asymmetric model (ADCC), whose correlations also answer joint
        negative returns, through a third parameter ``g``. Default False.
    ``margins``
        ``"garch"``, the default: each target is a return series,
        standardized by its own GARCH model (the parameters below, as
        :func:`garch` takes them). ``"none"``: the targets are standardized
        residuals already, from margins fitted elsewhere.
    ``vol``, ``dist``, ``truncation``, ``variance_targeting``
        The margins' process, innovations, FIGARCH lags and variance
        targeting, as :func:`garch` takes them. Refused with
        ``margins="none"``.
    ``margin_start_rows``, ``margin_start_span``
        The margins' start, as :func:`garch`'s ``start_rows`` and
        ``start_span``. Default 2,000 rows.
    ``a``, ``b``, ``g``
        The parameters, given: held rather than learned. Give ``a`` and
        ``b``, and ``g`` for the asymmetric model, or none of them. ``a + b``
        must be at most ``1 - 1e-4``; with ``g``, the model holds ``a + b +
        δ g`` at that bound as the targets move (see the update).
    ``start_rows``
        The rows the correlation's start buffers, after the margins' start:
        the targets are their means, and the parameters their batch
        quasi-likelihood's maximum. They report no correlation. Default
        500, at least 1: the start waits on the evidence, not a count. Its
        fit is withheld (``withheld_reason`` ``not_identified``) until its
        likelihood-ratio statistic against constant correlation reaches χ²
        at 0.999 (13.8 for ``a`` and ``b``, 16.3 with ``g``), and stepped
        from only at its optimum. A start that fails either keeps its rows
        and fits again once as many new rows have come as it held: twice
        its rows while ``max_state_bytes`` holds them, then the latest rows
        it holds. No count of tries ends the wait. Under a decay give a
        half-life several times these rows: the rows' weights decay to the
        last, so the evidence a buffer holds stops growing. Refused beside
        ``start_span``.
    ``start_span``
        The correlation's start as a span of the clock in place of
        ``start_rows``: a number of the clock column's units, or a duration
        such as ``"1d"`` on a temporal clock. It closes at the first row at
        or past the span from its first row; a refit waits, as under
        ``start_rows``, for as many new rows as the try before held.
    ``max_state_bytes``
        The bytes a group's state may hold, both starts' buffers included.
        A spec whose state is past it is refused; a first span is checked as
        the rows go in, warned of once past half of it, and stopped at it.
        A start that waits after a try keeps at most the rows it holds.
        Default 256 KB, ``262144``, which holds eight series with the
        margins' default start; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#dcc
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#dcc>`_
    gives each field's type and when it is null:

    ``variance_<t>``
        With margins, per target: the GARCH variance of the row's return
        given the rows before it. Null before the margin's start.
    ``corr_<a>_<b>``
        Per pair of targets: their correlation given the rows before the
        row, read before the row is learned. Null before the correlation's
        start, and below ``min_weight``.
    ``loglik``
        The row's Gaussian log-density of its standardized residuals under
        that correlation; null where a target is.
    ``weight_sum``
        The weight the model has learned before the row, decayed.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).
    ``coef``
        After the row: ``a``, ``b`` and, for the asymmetric model, ``g``.
        Null before the start of a learned model. :func:`coef_index` names
        them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for fewer than two targets, ``margins`` not one of the words
    above or ``"none"`` beside a margin's parameter, a margin's parameter
    :func:`garch` refuses, some of ``a``, ``b`` and ``g`` but not all, ``g``
    without ``asymmetric``, a negative parameter or ``a + b`` past
    ``1 - 1e-4``, a start below its floor or given both ways, and a state
    past ``max_state_bytes``.

    .. rubric:: The update

    With ``z_t`` the row's standardized residuals and ``n_t`` their negative
    parts (``z`` where it is below 0, else 0):

    .. code-block:: text

        Q_{t+1} = (1 - a - b) Q̄ - g N̄ + a z_t z_tᵀ + g n_t n_tᵀ + b Q_t
        R_t     = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)
        Q̄ ← Q̄ + (w/W) (z zᵀ - Q̄),   N̄ ← N̄ + (w/W) (n nᵀ - N̄),   W ← λ W + w

    ``g = 0`` is DCC(1,1). The parameters are held in ``a, b, g ≥ 0`` and
    ``a + b + δ g ≤ 1 - 1e-4``, ``δ`` the largest eigenvalue of
    ``Q̄^(-1/2) N̄ Q̄^(-1/2)``: exactly where the intercept ``(1 - a - b) Q̄ -
    g N̄`` is positive semi-definite, so every ``Q_t`` is positive definite,
    and where a covariance-stationary ``Q_t`` exists given innovations with
    a finite fourth moment. A GARCH margin with Student-t degrees of freedom
    at or below 4 has none, and the bank says so once. Learned, each row of
    weight ``w`` steps the parameters by the score of its quasi-log-
    likelihood ``l_t = -½ (ln |R_t| + z_tᵀ R_t⁻¹ z_t)``:

    .. code-block:: text

        r = w / W
        H ← H + r (I_t - H)          I_t = ½ tr(R⁻¹ ∂R_i R⁻¹ ∂R_j)
        θ ← Proj_H(θ + r H⁻¹ g_t)    g_t = ½ (uᵀ ∂R_i u - tr(R⁻¹ ∂R_i)),  u = R⁻¹ z

    with ``∂R`` from ``∂Q``, which its own recursion carries, and
    ``Proj_H`` the nearest point of the set in ``H``'s metric. The state is
    ``32 k² + 96`` bytes for DCC and ``48 k² + 144`` for ADCC, learned, for
    ``k`` series (``16 k² + 64`` and ``24 k² + 72`` given), and the margins'
    168 bytes a series, plus each start's buffer until it is fitted:
    ``8 (k + 2)`` bytes a row of the correlation's, 16 a row a series of the
    margins'.

    .. rubric:: Example

    This code simulates two series whose correlation follows a DCC at
    ``(a, b) = (0.04, 0.93)``, and prints the fit after 20,000 rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        a, b, qbar = 0.04, 0.93, np.array([[1.0, 0.5], [0.5, 1.0]])
        q, rows = qbar.copy(), []
        for _ in range(20_000):
            d = np.sqrt(np.diag(q))
            z = np.linalg.cholesky(q / np.outer(d, d)) @ rng.standard_normal(2)
            q = (1 - a - b) * qbar + a * np.outer(z, z) + b * q
            rows.append(z)
        z = np.array(rows)
        pl.DataFrame({"x": z[:, 0], "y": z[:, 1]}).write_parquet("z.parquet")

        model = rp.spec.dcc(
            "c", targets=["x", "y"], margins="none", half_life="inf"
        )
        bank = rp.ModelBank([model])
        bank.fit(pl.scan_parquet("z.parquet"))
        coef = {row["term"]: row["coef"] for row in bank.coef().iter_rows(named=True)}
        print(f"a: true 0.04, found {coef['a']:.2f}")
        print(f"b: true 0.93, found {coef['b']:.2f}")

    .. code-block:: text

        a: true 0.04, found 0.04
        b: true 0.93, found 0.93

    .. rubric:: See also

    - :func:`garch`, for each series' variance alone.
    - :func:`deco`, for one average correlation across many series, at
      ``O(k)`` a row where this model's is ``O(k³)``.
    - :func:`ew_cov`, for an exponentially weighted correlation with a
      fixed decay rather than a fitted recursion.

    .. rubric:: Notes

    **The estimator.** It is the recursive prediction-error method on the
    second stage's quasi-likelihood: with the gain ``1/W`` and a batch
    start its estimate agrees with the batch fit of the same rows to first
    order. The batch fit it is held to is the one Engle's definition gives,
    the margins and their residuals fixed, ``Q̄`` and ``N̄`` the sample
    means, maximized by a general optimizer.

    **Two stages.** As in the batch method, the margins' estimation error
    is not carried into the correlation's: the quasi-likelihood is the
    second stage's alone.
    """
    model: dict[str, Any] = {"type": "dcc"}
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if asymmetric:
        model["asymmetric"] = True
    if margins != "garch":
        model["margins"] = margins
    if vol != "garch":
        model["vol"] = vol
    if dist != "normal":
        model["dist"] = dist
    if truncation is not None:
        model["truncation"] = truncation
    if variance_targeting:
        model["variance_targeting"] = True
    if margin_start_rows is not None:
        model["margin_start_rows"] = margin_start_rows
    if margin_start_span is not None:
        model["margin_start_span"] = margin_start_span
    for key, value in (("a", a), ("b", b), ("g", g)):
        if value is not None:
            model[key] = value
    if start_rows is not None:
        model["start_rows"] = start_rows
    if start_span is not None:
        model["start_span"] = start_span
    return _common(name, model, targets=targets, features=[], **common)
