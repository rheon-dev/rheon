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
    family: Literal["normal"] = "normal",
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
    likelihood over its rows, the older rows weighing less by the decay.
    Every value a row reports is read from the fit before the row is
    learned, so each is a forecast of that row.

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
        The distribution: ``"normal"``, the default and the only one, with
        ``loc`` its mean and ``scale`` its standard deviation, as
        ``scipy.stats.norm`` names them.
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
        Per target, why every field of the target above is null, as an
        ``Enum``: ``no_weight`` (nothing learned yet, or a weight the decay
        has aged to nothing), ``min_weight`` (under the floor) or
        ``degenerate_variance`` (every value learned was the same, or equal
        to ``floc``, so the scale is 0). Null where the target reports, and
        on a row the shared gates withhold (``withheld_reason``).
    ``weight_sum``
        The weight the model has learned before the row and before the
        row's own decay, every row's whatever its targets.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for a ``family`` other than ``"normal"``, a level of
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
    length or the levels asked for.
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
