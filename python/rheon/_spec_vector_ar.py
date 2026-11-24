"""The vector autoregression builder, :func:`vector_ar`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


def _names(v: str | list[str]) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


@_checked
def vector_ar(
    name: str,
    *,
    targets: TargetList,
    lags: int,
    features: list[str] | None = None,
    ridge: float | None = None,
    max_state_bytes: int | None = None,
    select: Literal["aic", "bic", "hqic", "fpe"] | None = None,
    causality: list[dict[str, Any]] | None = None,
    causality_kind: Literal["f", "wald", "ssr_f", "ssr_chi2", "lr"] | None = None,
    solve_every: float | Duration | None = None,
    max_rows_between_solves: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    snapshot_every: float | Duration | None = None,
    max_rows_between_snapshots: int | None = None,
    window_budget: dict[Literal["thin", "refuse"], float] | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """A vector autoregression of the targets together, fitted by
    exponentially weighted least squares on their last values, with Granger
    causality tests reported on every row.

    Use it when one series' past predicts another's, or when the question
    is whether it does: the Granger test on every row, and the impulse
    responses and variance decomposition read with
    :meth:`ModelBank.vector_ar`. The model keeps the series' last values
    itself, so the input needs no lag columns. With no decay its
    coefficients and tests are statsmodels' ``VAR(y, exog).fit(p)``'s on the
    same rows.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``group`` and the
    diagnostics, are in :mod:`rheon.spec`. ``half_life`` (or ``lam``) is on
    the clock, in clock units or a duration on a temporal clock;
    ``fit_intercept`` fits the constant ``c``, statsmodels' ``trend="c"``,
    and ``False`` fits none, its ``trend="n"``. A row's ``weight`` scales
    its evidence. A row of weight 0, or with any target null, learns nothing
    and is not a lag: the next row's lags are the ones before it, as if it
    were not there, which is statsmodels' ``missing="drop"``. A row with a
    target null is still predicted, every target, where its lags are
    complete, and its weight counts in ``weight_sum``. ``embargo`` is
    refused: under a delay the rows it predicts would read older lags than
    the rows it learns from.

    This model's own parameters:

    ``lags``
        The order ``p``: the lags ``1, ..., p`` of every target, at least 1.
        Required. A lag counts learned rows, not clock units: the value
        ``ℓ`` rows back has no meaning on a clock, so the lags are positions
        in the series and only the decay and the window are on the clock. A
        session change or a gap past ``gap_cap`` empties the lags, since the
        rows on either side are no longer adjacent; the fit is kept, and the
        next ``p`` learned rows refill them before it learns again.
    ``features``
        Exogenous inputs, in every equation: their coefficients come before
        the lags', as in statsmodels' ``params``. Default none.
    ``ridge``
        A ridge penalty on the lag and feature coefficients, on the scale of
        the weighted mean of ``z zᵀ``; never on the constant. Default 0,
        least squares. Refused beside ``select`` and ``causality``, whose
        criteria and laws are least squares'.
    ``max_state_bytes``
        The bytes a group's state may hold. With ``k`` targets, ``e``
        features, ``d = e + kp`` and ``K = d + k + 1``, the state is at most
        ``8kp`` for its lags, ``8 (4K² + 17K + 64) + 8 (k - 1)(4K + 16)``
        for its fit, ``8 (2k(d + 1) + T + 10)`` for what its last solve
        read with ``T`` tests, and 2 KiB of field names. That is fixed by
        these settings, never by the stream's length, so a spec past it is
        refused when it is built, the refusal giving the bytes it needs.
        Default 256 KB, ``262144``, which takes up to 41 lags of 2 targets,
        15 of 5 and 7 of 10 (ten targets at ten lags need 462,088 bytes);
        at most 64 MiB.
    ``select``
        Choose the order at every solve: ``"aic"``, ``"bic"``, ``"hqic"`` or
        ``"fpe"`` picks the ``q`` in ``0, ..., p`` whose lags ``1, ..., q``
        (with the features) minimize the criterion, as statsmodels'
        ``VAR.select_order(p)`` computes it, every order fitted on the same
        rows, those after the first ``p``. The predictions, ``coef``, the
        tests and ``order`` are that order's, its lags past ``q`` reported as
        0. The count in each criterion is Kish's effective number of rows,
        ``(Σw)² / Σw²`` at the decayed weights: the row count with no decay
        and unit weights. Only the orders the moments identify are
        candidates (*Identification* in Notes). Needs ``ridge`` 0. Default
        none, every lag.
    ``causality``
        The Granger tests to report on every row, each a dict
        ``{"caused": ..., "causing": ...}`` of target columns (a name or a
        list): the null that the lags of ``causing`` do not enter the
        equations of ``caused``. ``causing`` left out is every target not in
        ``caused``. The two sets may not share a series. Test ``i`` writes
        ``causality<i>_stat`` and ``causality<i>_p``. Needs ``ridge`` 0.
        Default none.
    ``causality_kind``
        The statistic every test reports, from the Wald statistic ``λ`` of
        the ``q`` restrictions, the residual degrees of freedom ``df`` and
        Kish's count ``n``:

        - ``"f"``, the default: ``λ / q`` against ``F(q, k·df)``, statsmodels'
          ``test_causality(kind="f")``;
        - ``"wald"``: ``λ`` against ``χ²(q)``, its ``kind="wald"``;
        - ``"ssr_f"``: ``λ / q`` against ``F(q, df)``, statsmodels'
          ``grangercausalitytests`` ``ssr_ftest``;
        - ``"ssr_chi2"``: ``n λ / df`` against ``χ²(q)``, its
          ``ssr_chi2test``;
        - ``"lr"``: ``n ln(1 + λ / df)`` against ``χ²(q)``, its ``lrtest``.

        The last three test one caused equation, so they are refused for a
        test with more than one caused series.
    ``solve_every``, ``max_rows_between_solves``
        How often the fit solves for its coefficients, as
        :func:`ewridge`'s, with its defaults: by weight under a finite
        ``half_life``, every row under ``half_life = inf`` or ``lam``. The
        tests, the order and the standard errors are read at each solve,
        from the same moments, and held between solves as the coefficients
        are. ``solve_every`` is in clock units, or a duration on a temporal
        clock.
    ``window_size``
        Clock units of history the fit is read from, a hard cutoff: a row
        older than this is in no fit, as :func:`ewridge`'s ``window_size``
        (a number of the clock's units, or a duration on a temporal clock).
        Under decay the rows inside are still weighted by it. A row's lags
        may be older than the window: they are part of the row. As rows
        leave, the fit can lose the spread it needs, so under a window its
        identification is read again at every solve. Default none.
    ``closed``, ``snapshot_every``, ``max_rows_between_snapshots``
        The window's edge and the cadence of the snapshots it is cut from,
        as :func:`ewridge`'s. A snapshot holds about ``K² / 2 + kK`` numbers.
    ``window_budget``
        The MiB a group's snapshots may take; past it they thin or refuse
        the run, as :func:`ewridge`'s. Default ``{"refuse": 256}``.

    .. rubric:: Output

    One output column, named after the spec, whose value in each row is a
    record of these fields. `docs/OUTPUTS.md#vector_ar
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#vector_ar>`_
    gives each field's type and when it is null:

    ``pred_<t>``, ``resid_<t>``
        Per target: the prediction from the targets' last values and the
        row's features, read before the row is learned, and ``y - pred``
        where the target is not null. Null until the lags are complete and
        the fit is identified, and below ``min_weight``.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay: every row's, the first ``p`` and those with a
        target null included.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`);
        ``withheld_reason`` is ``not_identified`` while the fit is not.
    ``coef``
        Per target after the row: ``intercept`` (with ``fit_intercept``),
        the features, then ``lag1_<s>`` for every target ``s``, then
        ``lag2_<s>``, and so on: statsmodels' ``params`` column for the
        target. Null until the fit is identified. :func:`coef_index` names
        them.
    ``causality<i>_stat``, ``causality<i>_p``
        Per test in ``causality``: the statistic of ``causality_kind`` and
        its p-value, read at the last solve before the row. Null until the
        fit is identified, while the residual degrees of freedom are not
        above 0, and below ``min_weight``.
    ``order``
        Under ``select``: the order in force, read at the last solve.

    plus the fields of the diagnostics switched on, as
    :mod:`rheon.spec` describes them. ``emit_se_coef`` reports
    statsmodels' ``bse``, the standard error the tests are read with
    (*Standard errors* in Notes).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); and ``ValueError`` naming the
    problem for ``lags`` below 1 or past ``2**20``, a ``ridge`` below 0 or
    not finite, or above 0 beside ``select`` or ``causality``, a test
    naming a column that is not a target, a series twice, or one series on
    both sides, a single-equation ``causality_kind`` beside a test of more
    than one caused series, and a state past ``max_state_bytes``.

    .. rubric:: The update

    With the row's features ``x`` and the targets' lags, the design of row
    ``t`` is ``z_t = [x_t, y_{t-1}, ..., y_{t-p}]`` (``d = e + kp``
    numbers), and the model is

    .. code-block:: text

        y_t = c + B_x x_t + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t

    Every equation has the same regressors, so least squares equation by
    equation is the system's estimate, and one set of moments serves them
    all. Per row, ``λ = 0.5 ** (Δclock / half_life)``, weight ``w``, ``W``
    the weight of every row and ``W_y`` that of the learned rows:

    .. code-block:: text

        pred_t = c + B' z_t                    from the fit before the row
        W'     = λ W + w                       weight_sum reads W
        W_y'   = λ W_y + w                     a learned row only
        M'     = (λ W_y M + w v vᵀ) / W_y'     v = [z_t, y_t], centered
        B      = (S_zz + ridge I)⁻¹ S_zy       at each solve
        c      = ȳ - B' z̄
        SSE    = S_yy - S_zyᵀ B                the residual co-moments

    ``M`` holds ``S_zz``, ``S_zy`` and ``S_yy``, the weighted means of
    ``z zᵀ``, ``z yᵀ`` and ``y yᵀ`` over the learned rows, kept centered
    with the intercept (raw without it), so a series far from zero loses
    nothing to its level. A learned row has a weight above 0 and every
    target present; it is appended to the lags, and it enters ``M`` only
    once ``p`` rows precede it, since before that its lags do not exist.
    At each solve, from one Cholesky factor ``S_zz = L Lᵀ`` and ``n =
    W_y² / Σw²``, Kish's count:

    .. code-block:: text

        df  = n - (d + 1)                      the residual degrees of freedom
        Σ_u = SSE · n / df                     statsmodels' sigma_u
        λ   = df · tr(SSE_E⁻¹ βᵀ G⁻¹ β)         a test's Wald statistic

    with ``R`` the lag columns of the causing series, ``G = (S_zz⁻¹)_RR``,
    ``β = B[R, E]`` and ``SSE_E`` the residual co-moments of the caused
    equations ``E``. No restricted fit is run.

    .. rubric:: Example

    This code simulates two series where the first's past drives the
    second, and prints the test of that and of the reverse after 5,000
    rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(4)
        a, b = [0.0], [0.0]
        for _ in range(5_000):
            a.append(0.5 * a[-1] + rng.standard_normal())
            b.append(0.3 * b[-1] + 0.4 * a[-2] + rng.standard_normal())
        pl.DataFrame({"a": a, "b": b}).write_parquet("pair.parquet")

        model = rp.spec.vector_ar(
            "v",
            targets=["a", "b"],
            lags=2,
            causality=[{"caused": "b", "causing": "a"}, {"caused": "a", "causing": "b"}],
            half_life=float("inf"),
        )
        out = pl.scan_parquet("pair.parquet").rheon.fit_predict([model]).collect()
        last = out["v"][-1]
        print(f"a -> b: p = {last['causality0_p']:.0e}")
        print(f"b -> a: p = {last['causality1_p']:.2f}")

    .. code-block:: text

        a -> b: p = 1e-190
        b -> a: p = 0.98

    .. rubric:: See also

    - :func:`ar`, an autoregression of each target on its own past alone.
    - :meth:`ModelBank.vector_ar`, the impulse responses, the variance
      decomposition, stability, the long-run mean, every order's criteria
      and a test asked after the fact.
    - :func:`ewridge`, least squares on lag columns made in the query.

    .. rubric:: Notes

    **ar or vector_ar.** Use :func:`ar` when each series is predicted from
    its own past, when the series are missing on different rows (each of
    ``ar``'s targets keeps its own series), or when there are many series:
    its cost a row is ``k (p + e)²``. Use ``vector_ar`` when one series'
    past predicts another's, when the question is Granger causality, the
    impulse responses or the variance decomposition, and when the series
    arrive together: its cost a row is ``(kp + e + k)²`` on one set of
    moments. A row missing any series teaches ``vector_ar`` nothing. With
    one target and no features the two are the same fit, to rounding.

    **Exact against the library.** With no decay and ``ridge=0``, the
    coefficients, ``Σ_u``, the standard errors and every test's statistic
    are statsmodels' ``VAR(y, exog).fit(p)``'s on the rows after the first
    ``p``, to rounding: within ``1e-12`` of the largest coefficient on the
    test suite's streams. The order criteria are ``select_order(p).ics``'s,
    and the impulse responses, their standard errors and the variance
    decomposition read with :meth:`ModelBank.vector_ar` are ``irf`` and
    ``fevd``'s. Under decay the coefficients are weighted least squares
    with each row weighted ``w · λ^age``, its age on the clock.

    **Identification.** Until the lags and features have spread in every
    direction, least squares has no unique answer. So the fit is read only
    once ``S_zz`` plus the ridge is positive definite: a Cholesky factor
    with every pivot above ``1e-10`` of its diagonal. The guard is that
    condition, not a count of rows: on series that move independently it
    opens at as many learned rows as each equation has coefficients, and
    on a series that is a multiple of another it never opens. Under
    ``select`` it opens once order 1 is identified, and the candidates are
    the orders whose block of ``S_zz`` passes. ``min_weight`` defaults to
    0, with this gate in its place.

    **Standard errors and tests under decay.** Under decay the weights
    shrink the rows' share of the evidence, and Kish's count ``n`` stands
    for the number of rows: ``df = n - (d + 1)``, ``Σ_u = SSE · n / df``
    and the coefficients' covariance ``(S_zz⁻¹ / n) ⊗ Σ_u``, which are
    statsmodels' with no decay and unit weights. ``SSE / df`` alone would be
    half the noise at steady state. Measured on a VAR(1) of two series with
    unit noise and no causation, the test read at the end of each of 1,000
    streams of 2,000 rows (a Monte Carlo standard error of 0.7%), the F test
    at the 5% level rejected 4.4% of the time at ``λ = 0.99`` (an effective
    199 rows) and 3.6% at ``λ = 0.95`` (an effective 39), ``"ssr_f"`` 4.4%
    and 3.1%, with ``Σ_u``'s first diagonal entry within 0.2% of the noise's
    on average: it holds its size once the memory spans a few hundred rows,
    and is conservative below that. The tests and
    the standard errors are null until ``df`` is above 0; the predictions
    are not.

    **Reads.** :meth:`ModelBank.vector_ar` computes the impulse responses,
    their standard errors, the variance decomposition, the companion
    matrix's roots, the long-run mean, every order's criteria and a test of
    any two sets from the moments as they stand, when asked; none of them
    is kept in the state, and none is written per row.
    """
    model: dict[str, Any] = {
        "type": "vector_ar",
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
    if causality is not None:
        tests: list[dict[str, Any]] = []
        for t in causality:
            if "caused" not in t:
                msg = (
                    f"spec {name!r}: each causality test is a dict with a 'caused' key, the "
                    f"series whose equations it reads, but one is {t!r}. Add 'caused'."
                )
                raise ValueError(msg)
            one: dict[str, Any] = {"caused": _names(t["caused"])}
            causing = t.get("causing")
            if causing is not None:
                one["causing"] = _names(causing)
            tests.append(one)
        model["causality"] = tests
    if causality_kind is not None:
        model["causality_kind"] = causality_kind
    if solve_every is not None:
        model["solve_every"] = solve_every
    if max_rows_between_solves is not None:
        model["max_rows_between_solves"] = max_rows_between_solves
    return _common(name, model, targets=targets, features=features or [], **common)
