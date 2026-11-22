"""The equicorrelation model's builder, :func:`deco`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import _checked, _common, _mirror_target


@_checked
def deco(
    name: str,
    *,
    features: list[str],
    dynamics: Literal["ew", "linear"] = "ew",
    alpha: float | None = None,
    beta: float | None = None,
    blocks: dict[str, list[str]] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Engle and Kelly's dynamic equicorrelation, one number for the whole correlation
    matrix of the features.

    Use it to track how correlated many series are on average, at ``O(n)``
    a row for ``n`` features, where keeping all ``n(n-1)/2`` correlations
    moving takes ``O(n²)``. Use :func:`ew_cov` for each pair's correlation,
    and this model's ``blocks`` for one number per block of features and per
    pair of blocks.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. This model reads ``features``, at least two,
    and takes no ``targets``. ``fit_intercept`` has nothing to act on, as
    there is no regression, and each switch that reads residuals is
    refused, as is ``emit_feature_health``.
    ``half_life`` or ``lam`` is required: both the standardizer and the
    level decay on it. This model's own:

    ``dynamics``
        How the level follows the rows: ``"ew"`` (the default, since it
        runs on the clock and needs no parameters of its own), the ratio of
        exponentially weighted means on the clock, or ``"linear"``, the
        averaged correlation recursion of *The update*, which steps once
        per row. ``"linear"`` needs ``alpha`` and ``beta``, and ``"ew"``
        refuses them.
    ``alpha``, ``beta``
        The linear dynamics' weights on the row and on the previous
        level, each at least 0, with ``alpha + beta < 1``. Default none:
        required under ``"linear"``, refused under ``"ew"``.
    ``blocks``
        A name to a subset of ``features``: the model then estimates one
        number per block and one per pair of blocks, which is the useful
        middle between one correlation and all of them. Every feature must
        be in exactly one block, and a block needs at least two. Default
        none: one block of every feature.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields, all but ``coef`` read from the state before the
    row, so they are safe as features for that same row.
    `docs/OUTPUTS.md#deco
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#deco>`_
    gives each field's type and when it is null:

    ``u``
        The row's own estimate of the equicorrelation, biased toward 0
        (*The update*). Null on a row where fewer than two of its block's
        columns have a positive variance before the row, or where every
        standardized value is 0.
    ``rho``
        The level as it stood before the row: the estimate of the
        equicorrelation.
    ``loglik``
        The row's Gaussian log-density in standardized coordinates under
        that level. Null on a row where any column has no positive variance
        before the row, and where the levels hold no correlation matrix that
        the density can be solved for (*Blocks* in Notes).
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row, null where nothing decays.
    ``withheld_reason``
        Why the row's fields were withheld, null where they were not:
        ``below_min_weight`` on the rows before ``weight_sum`` reaches
        ``min_weight``, then ``below_warm_rows`` while no level has a
        value, until the standardizer is warm (*The standardizer's
        warm-up* in Notes). ``u`` reads null with no reason beside it until
        a block's columns have a variance (the first two rows of a group).
    ``coef``
        The levels after the row, in the order of the ``rho`` fields:
        ``coef`` on a row is ``rho`` on the next.

    With ``K`` named blocks, ``u`` becomes ``u_<A>`` per block then
    ``u_<A>_<B>`` per pair, and ``rho`` likewise, with one ``loglik`` over
    all of them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), ``TypeError`` for
    ``targets``, and ``ValueError`` naming the problem for:

    - fewer than two features, or no ``half_life`` or ``lam``;
    - a ``dynamics`` other than ``"ew"`` or ``"linear"``;
    - ``alpha`` or ``beta`` under ``"ew"``, either missing under
      ``"linear"``, either below 0 or not a number, and
      ``alpha + beta`` of 1 or more;
    - ``blocks`` that name no block, leave a feature out of every block,
      put one in two, list one twice in a block, name a column not in
      ``features``, or hold a block of fewer than two;
    - a switch that reads residuals, such as ``emit_sigma``, and
      ``emit_feature_health``.

    .. rubric:: The update

    The row is standardized against the means ``m_i`` and variances ``v_i``
    that exponentially weighted running sums hold before the row,
    ``zeta_i = (x_i - m_i) / sqrt(v_i)``. With ``S1 = sum(zeta)`` and
    ``S2 = sum(zeta * zeta)`` over the ``n`` features, the row's estimate is
    Engle and Kelly's eq. 20:

    .. code-block:: text

        u = A / V,   A = (S1**2 - S2) / (n * (n - 1))   = mean_{i != j} zeta_i zeta_j
                     V = S2 / n                          = mean_i zeta_i**2

    which lies in ``[-1 / (n - 1), 1]``, both ends reached (their Lemma
    2.3). Their Lemma 2.4 reads ``u`` and ``V`` as the one-row Gaussian
    maximum-likelihood estimates of the equicorrelation and the variance.
    Between two blocks ``A`` and ``B`` the estimate is
    ``u_AB = C_AB / sqrt(V_A * V_B)``, ``C_AB = S1_A * S1_B / (n_A * n_B)``.

    ``u`` is a ratio over one row, and the mean of a ratio is not the
    ratio of the means: it reads toward 0 (1/3 at a true 0.6 for two
    series, about 0.41 for many, by its exact Gaussian expectation), and
    neither more series nor a longer half-life shrinks that, since every
    row sees one draw of the common factor. So the level averages ``A``
    and ``V`` first and divides once, with ``w`` the row's weight and
    ``λ = 0.5 ** (Δclock / half_life)`` its decay (``lam ** Δclock`` under
    ``lam``). ``Δclock`` is the clock's step since the last row the model
    did not skip, capped at ``gap_cap``, or ``session_gap`` at a session
    change:

    .. code-block:: text

        W' = λ * W + w,  b = w / W',  Abar' = Abar + b * (A - Abar),  Vbar' likewise
        "ew":      rho = Abar / Vbar
        "linear":  N' = (1 - alpha - beta) * Abar' + alpha * A + beta * N
                   D' = (1 - alpha - beta) * Vbar' + alpha * V + beta * D
                   rho = N / D              (N = A, D = V on the first row; held at w = 0)

    Between two blocks ``Cbar`` takes ``Abar``'s place and the level is
    ``Cbar / sqrt(Vbar_A * Vbar_B)``, or ``N / sqrt(D_A * D_B)``, each pair
    keeping its own denominators. ``"ew"`` is the weighted Gaussian
    maximum-likelihood estimate of the equicorrelation over the weighted
    window: Lemma 2.4 with the sums taken over the window instead of one
    row. ``"linear"`` is the correlation recursion of a DCC model (Engle's
    dynamic conditional correlation, their eq. 5, with correlation
    targeting) averaged over its off-diagonal and its diagonal; the
    paper's own level reads each entry's correlation before it averages,
    at ``O(n²)`` a row, and differs from this one by 0.001 to 0.003 on
    average on equicorrelated Gaussian streams of 3 to 30 series. Both levels stay in
    ``[-1 / (n - 1), 1]``, since every row has ``-V / (n - 1) <= A <= V``.

    .. rubric:: Example

    This code plants eight series whose pairs are correlated 0.2 for 10,000
    rows and 0.6 after, and prints the level before the last row of each
    half:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(6)
        c = np.repeat([0.2, 0.6], 10_000)               # the pairs' correlation, row by row
        common = rng.standard_normal(20_000)
        cols = {
            f"r{i}": np.sqrt(c) * common + np.sqrt(1 - c) * rng.standard_normal(20_000)
            for i in range(8)
        }
        pl.DataFrame(cols).write_parquet("returns.parquet")

        eq = rp.spec.deco("eq", features=list(cols), half_life=500.0)
        out = pl.scan_parquet("returns.parquet").rheon.fit_predict([eq]).collect().unnest("eq")
        print(f"correlation 0.2: rho {out['rho'][9_999]:.2f}")
        print(f"correlation 0.6: rho {out['rho'][-1]:.2f}")

    .. code-block:: text

        correlation 0.2: rho 0.22
        correlation 0.6: rho 0.60

    .. rubric:: See also

    - :func:`ew_cov`, every pair's correlation, at ``O(n²)`` a row.
    - :func:`corrchange`, a test for a change in a pair's correlation.

    .. rubric:: Notes

    **The clock.** The ``"ew"`` dynamics run on the clock. The linear one
    steps once per row, as a DCC model's does. So across a gap capped at
    ``gap_cap`` its own terms take one row's step, as across a millisecond,
    and only the means ``Abar`` and ``Vbar`` move on the clock. A row of
    weight 0 moves neither: it learns nothing, as everywhere. Otherwise the
    linear recursion has no row weights, as the paper's has none: a
    positive weight reaches ``rho`` only through the means, so rows of
    weight 0.5 and 2 move ``N`` and ``D`` by the same ``alpha`` share.

    **The standardizer's warm-up.** A row standardized against a variance
    resting on a few rows is heavy-tailed: under Gaussian rows each
    ``zeta_i`` is Student-t with ``n_e - 1`` degrees of freedom, where
    ``n_e = (sum w)**2 / sum(w**2)`` is Kish's effective size of the
    standardizer's weights. Its fourth moment, and with it the variance of
    what the row adds to ``Abar`` and ``Vbar``, exists only above 4 degrees
    of freedom, and below that one row's square can outweigh the window.
    So the level learns from a row only when its standardizer has
    ``n_e > 5``. Before that a row teaches the level nothing, and while no
    level has a value the row says so (``below_warm_rows``). With equal
    weights and no decay ``n_e`` is the row count, so the seventh row is
    the first learned. A decay whose steady state holds 5 effective rows or
    fewer, a half-life under 1.71 rows, never learns; a gap that takes the
    history holds the level where it was until the standardizer has the
    rows again.

    **Departures from the paper.** Two from Engle and Kelly, on purpose.
    Its linear dynamics, eq. 21, fit a free intercept and ``alpha + beta``
    slightly above 1 by maximum likelihood, which rescales the biased
    ``u``; a streaming model has no sample to fit them on, so this one
    averages the numerator and the denominator instead, writes the
    intercept as ``(1 - alpha - beta)`` times their means, and refuses
    ``alpha + beta >= 1``. The paper's alternative row estimate,
    ``1 - (1 / (n - 1)) * sum((zeta_i - zeta_bar)**2)``, is not computed:
    it is unbounded below, and needs the unit variances the standardizer
    only estimates. The equation and lemma numbers are those of the open
    working paper, NYU FIN-08-038 (2008).

    **A constant column.** A column whose variance is exactly zero, constant
    from its first row (the only way an exponentially weighted variance is
    exactly zero), has no standardized value. It is left out of its block's
    sums, so ``n`` counts the columns that have one. Its block reads the
    correlation among its other columns, and so does a pair it is in. A
    block left with fewer than two has no ``u`` on the row, nor a pair with
    an empty side. Nor does a row whose ``S2`` is 0. Each value keeps its own
    weight ``W``: one with no ``u`` on a row learns nothing and does not
    decay, while the others learn. ``loglik`` needs every column, and is
    null on such a row.

    **Blocks.** ``loglik`` is null where the levels hold no correlation
    matrix, or hold one too close to singular for its density to keep a
    digit: each level can be inside ``(-1, 1)`` while the blocks' levels
    together are not. Then the factor of the ``K x K`` system the density
    solves needs a shift on its diagonal, or its condition passes the point
    where the solve's forward-error bound reaches 1. Such levels come from
    the first rows a level learns from, when they rest on one or two rows.
    One block never reaches that point.
    """
    model: dict[str, Any] = {
        "type": "deco",
        "dynamics": dynamics,
        "alpha": alpha,
        "beta": beta,
        # `{}` as `[]`, refused by name as `feature_sets={}` is (YA5).
        "blocks": [[k, list(v)] for k, v in blocks.items()] if blocks is not None else None,
    }
    targets = _mirror_target(name, "deco", features, common, "its equicorrelation is over")
    return _common(name, model, targets=targets, features=features, **common)
