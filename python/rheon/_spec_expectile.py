"""The expectile regression's builder, :func:`expectile`: a part of
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
def expectile(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    expectile: float,
    ridge: float | None = None,
    refit_rows: int | None = None,
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    max_state_bytes: int | None = None,
    refit_budget: float | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Expectile regression, fitted one row at a time by reweighted least squares.

    Use it for a tail of the target that should weigh how far the rows past
    it lie, not only how many there are: the 0.9 expectile of a loss, a
    latency or a demand. Rows above the fit weigh ``tau`` in its squared
    error and rows at or below it ``1 - tau``, so ``expectile=0.5`` is the
    least-squares fit and a level near 1 an upper envelope that every large
    row pulls up. Use :func:`quantile` for the level a share of the rows fall
    below, which no single row can move far, and :func:`ewridge` for the mean
    with its diagnostics.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`rheon.spec`. A row's ``weight`` scales its
    squared error, as a weighted least-squares fit's weights do. A row of
    weight 0 counts in no clock: the next row decays as if it were not
    there, it is in no window and in no ``weight_sum``; it is still
    predicted.

    This model's own parameters:

    ``expectile``
        The level ``tau``, above 0 and below 1; required. Several levels are
        several specs: each level weighs every row by its own sides, so they
        share no sums, and a spec per level costs what one spec of several
        would.
    ``ridge``
        The penalty ``ridge * W * |slopes|²`` beside the loss, ``W`` the
        target's weight: a ridge on the mean-form Gram, in the features'
        squared units, as :func:`quantile`'s. Default ``1e-6``, which keeps
        every solve defined from the first row, a constant or duplicated
        feature included; ``0`` is the exact fit, which waits for a row per
        coefficient. Finite and at least 0.
    ``refit_rows``
        The learned rows (each of positive weight, with its target) after
        which each target's first rows are fitted exactly, once, and the
        running sums rebuilt with each row on the side the exact fit puts it
        (*One pass and the refit* in Notes). Default ``max(200, 10 * k)``,
        ``k`` the coefficients, up to 100 features, held to what
        ``max_state_bytes`` holds beside the sums, and off above 100
        features, under a window, or where fewer than ``2 * k`` rows fit.
        ``0`` turns the refit off. A group's buffers hold ``8 * targets *
        refit_rows * (features + 2)`` bytes until the refit runs: a count
        past ``max_state_bytes`` is refused, naming the bytes.
    ``window_rows``
        Fit only the last this many rows of positive weight, refit exactly
        on every row, in place of a decay: set ``half_life`` to ``inf``
        beside it. At least one row per coefficient. Each row of the window
        takes ``8 * (features + targets + 1) + targets`` bytes, checked
        against ``max_state_bytes`` when the spec is built. Default none.
    ``window_size``
        Fit only the rows of the last this long on the clock, in clock units,
        refit exactly on every row, in place of a decay: set ``half_life`` to
        ``inf`` beside it. A number of the clock's units, or of rows where
        the spec has no clock, or a duration such as ``"5m"`` on a temporal
        clock; finite and above 0. A row leaves the window when it is
        ``window_size`` old on the clock (decayed after ``gap_cap``), before
        the row that reaches that age is predicted, decided on the rows'
        exact stamps (integer nanoseconds on a temporal clock). How many rows
        that is, the data decides, so its bytes are checked as the rows go
        in (see ``max_state_bytes``). Refused beside ``window_rows``.
        Default none.
    ``closed``
        Which edge of ``window_size`` holds the row exactly that old, in
        Polars' words (``rolling_sum_by(closed=)``). ``"right"``, the
        default, keeps the rows less than ``window_size`` old; ``"both"``
        keeps that row too and is refused without ``window_size``.
    ``max_state_bytes``
        The bytes a group's state may hold: the sums, a refit buffer and a
        window's rows. A refit buffer and ``window_rows`` fix their size, so
        a spec past the cap is refused when it is built. A ``window_size``
        window holds the rows the span has, so it is checked as the rows go
        in: the first row to take a group past half of the cap raises a
        warning, once for the group, naming the group and the bytes, and the
        row that would take it past the cap stops the run with an error
        naming the group and the bytes, the bank having learned part of the
        chunk. Each row of such a window takes a stamp of 32 bytes beside
        its numbers. Default 256 KB, ``262144``; at most 64 MiB,
        ``67108864``.
    ``refit_budget``
        A bound in MiB on the refit buffers of all the spec's groups
        together, above 0, or ``inf`` for none. Default 320. Each new group
        reserves the most its targets' buffers can hold when it first
        appears. The group that would cross the bound, and every new group
        after it, gives its refit up and keeps its one-pass fit, which
        ``bank.summary()``'s ``refit_given_up`` counts.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#expectile
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#expectile>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the fitted expectile at the row's features, from the
        rows before it, and ``y`` less it where the target is not null. Null
        before the target's first fit, which needs a row per coefficient.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay; under a window, the weight of the rows in it.
    ``settled_frac``
        How far the model had come toward its steady state under decay
        before the row.
    ``withheld_reason``
        Why the row's predictions were withheld, null where they were not.
    ``coef``
        Per target, the intercept (unless ``fit_intercept=False``) then one
        entry per feature (:func:`coef_index`); null before the first fit.

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`), and ``ValueError``
    naming the problem for:

    - an ``expectile`` not above 0 and below 1, and a ``ridge`` below 0 or
      not finite;
    - a ``window_rows`` under one row per coefficient, a ``window_size`` not
      finite and above 0, the two together, ``closed="both"`` without
      ``window_size``, and a window beside a finite ``half_life`` or a
      ``refit_rows`` above 0;
    - a ``max_state_bytes`` of 0 or above 64 MiB, and a refit buffer or a
      ``window_rows`` window that would hold more than it;
    - a ``refit_budget`` not above 0;
    - a ``window_size`` window that passes ``max_state_bytes``, as the rows
      go in.

    .. rubric:: The update

    Per row and target, ``x̃`` is ``[1, x]`` (``x`` without an intercept),
    ``w`` the row's weight and ``λ = 0.5 ** (Δ / half_life)`` the decay over
    the clock ``Δ`` since the last row that counted:

    .. code-block:: text

        pred = x̃'β                     the prediction, before the update
        a    = tau if y > pred else 1 - tau     (1/2 before the first fit)
        S    = λ S + w a
        G    = G + (w a / S) (x̃x̃' - G)
        c    = c + (w a / S) (x̃ y - c)
        W    = λ W + w
        β    = (G + ridge (W / S) D)⁻¹ c

    ``G`` and ``c`` are per target, since each target's rows fall on their
    own sides; ``D`` is the slopes' diagonal, the intercept unpenalized.
    Under a window, and at the refit, ``β`` is instead the exact minimiser
    of ``Σ v a(r) r² + ridge V |slopes|²`` over the rows held, ``v`` their
    weights and ``V`` their sum:
    each pass solves the weighted least squares at the current sides and
    judges every side again, until no side changes.

    .. rubric:: Example

    This code fits the 0.9 expectile of a target whose noise is skewed, and
    prints the coefficients this model finds. statsmodels' batch fit of the
    same rows, ``RLM`` with ``MQuantileNorm(0.9, LeastSquares())``, gives
    the same to two decimals:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(7)
        x = rng.standard_normal(20_000)
        y = 1.0 + 0.5 * x + rng.exponential(1.0, 20_000)
        pl.DataFrame({"x": x, "y": y}).write_parquet("skewed.parquet")

        fit = rp.spec.expectile(
            "upper", targets=["y"], features=["x"], expectile=0.9, half_life=float("inf")
        )
        bank = rp.ModelBank([fit])
        bank.fit(pl.scan_parquet("skewed.parquet"))  # learn every row, keep no output
        coef = {r["term"]: r["coef"] for r in bank.coef().iter_rows(named=True)}
        print(f"intercept {coef['intercept']:.2f}, x {coef['x']:.2f}")

    .. code-block:: text

        intercept 3.03, x 0.51

    .. rubric:: See also

    - :func:`quantile`, the level a share of the rows fall below.
    - :func:`ewridge`, the least-squares fit, which ``expectile=0.5`` is.
    - :func:`huber`, a mean that outliers cannot pull.

    .. rubric:: Notes

    **The loss.** The expectile at ``tau`` minimises ``Σ v a(r) r²`` with
    ``a(r) = tau`` above the fit and ``1 - tau`` at or below it. The loss is
    convex and its curvature lies between ``min(tau, 1 - tau)`` and
    ``max(tau, 1 - tau)`` times least squares' everywhere, so each step is a
    weighted least-squares solve, with no band and no residual scale to
    choose. Its fit is statsmodels' ``RLM`` with
    ``MQuantileNorm(tau, LeastSquares())``, and at one level of the target
    alone it is ``scipy.stats.expectile``.

    **One pass and the refit.** A row's side is judged by its residual
    against the fit before it, and the row's own update cannot move that
    residual across zero, so one solve a row is the whole step. What one
    pass cannot do is judge an earlier row again once the fit has crossed
    it. At ``tau = 0.5`` it is the decayed least-squares fit to rounding;
    elsewhere the gap of its fitted values to the exact decayed fit,
    measured over 10 seeds with decay, was 0.02 to 0.4 times the batch
    fit's own sampling error on average and 1.2 times it at worst (500
    rows, ``tau = 0.9``). Without decay the rows judged early stay at full
    weight, and the refit of the first ``refit_rows`` rows cut the gap at
    500 rows fivefold.

    **The state** is ``k(k+1)/2 + 2k + 4`` numbers a target, ``k`` the
    coefficients, and the shared weight and clock: 0.4 KB at 5 features. A
    row takes O(k²) time and a solve of O(k³).
    Until each target's refit runs, ``refit_rows (features + 2)`` more. A
    window holds its rows instead of the sums, and refits them on every
    row: about ``passes * rows * k²`` work a row, the passes 1 to 2 on
    average warm-started at the fit before and at most 10 on awkward data.

    **The spans.** ``half_life``, ``gap_cap`` and ``window_size`` take a
    number (rows, with no clock) or a clock duration, as every spec's do.
    ``window_rows`` and ``refit_rows`` count rows of positive weight.
    """
    model: dict[str, Any] = {
        "type": "expectile",
        "expectile": expectile,
        "ridge": ridge,
        "refit_rows": refit_rows,
        "window_rows": window_rows,
        "window_size": window_size,
        "closed": None if closed == "right" else closed,
        "max_state_bytes": max_state_bytes,
        "refit_budget": refit_budget,
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=targets, features=features, **common)
