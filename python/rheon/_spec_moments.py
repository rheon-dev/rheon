"""The moments' builder, :func:`ewm_moments`: a part of :mod:`rheon._spec`
in a file of its own, keeping that one under ``tests/test_repo_hygiene.py``'s
250 KB cap for a source file. :mod:`rheon.spec` documents it with the rest.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import _SKIPPED_WHEN_ABSENT, _checked, _common, _mirror_target

# The keys the builder writes only when given, as the Rust spec skips them
# when absent; entered here, since `_spec.py` stands at the source cap.
_SKIPPED_WHEN_ABSENT["ewm_moments"] = frozenset(
    {"kind", "bias", "se", "lag_tolerance", "window_rows", "window_size", "max_state_bytes"}
)


def _leg_column(features: list[str], key: str) -> str:
    """The column a leg reads: the key itself where it is a feature's name,
    so a column whose name holds ``@`` is read whole; otherwise the text
    before the last ``@``, as the Rust side reads it."""
    if key in features:
        return key
    return key.rsplit("@", 1)[0]


def _term_columns(terms: dict[str, dict[str, int]]) -> list[str]:
    """Every column the terms read, in the order they first name them, lags
    taken off: the features a spec that names none reads."""
    out: list[str] = []
    for legs in terms.values():
        for key in legs:
            col = _leg_column([], key)
            if col not in out:
                out.append(col)
    return out


@_checked
def ewm_moments(
    name: str,
    *,
    terms: dict[str, dict[str, int]],
    features: list[str] | None = None,
    kind: Literal["central", "raw", "standardized", "cumulant"] | dict[str, str] | None = None,
    bias: bool | None = None,
    se: bool = False,
    lag_tolerance: float | Duration | None = None,
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Exponentially weighted, or windowed, moments and cross-moments of
    columns, to any order, each leg lagged or not.

    Use it for the running shape of a stream's columns, read before each
    row: a skew or a kurtosis, a cumulant, a co-skewness, an
    autocovariance or an autocorrelation at a lag, each a term you name.
    Every term is computed in one pass from a state whose size the terms,
    the lags and the window fix, whatever the stream's length. It fits no
    target: use :func:`ew_cov` for the means, covariances and the
    statistics read off them, and :func:`marginal` for a regression's
    moments of a target against each feature.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight`` and ``group``, are in
    :mod:`rheon.spec`. This model takes no ``targets``. Nothing that reads
    residuals applies, and each such switch is refused by name, as are
    ``coef_every`` and ``max_rows_between_coefs``, as there are no
    coefficients. A row's ``weight`` is its frequency weight: scaling every
    weight changes nothing but ``weight_sum``. A row of weight 0 is no row:
    it is in no sum, no lag and no window, and it is still read. This
    model's own parameters:

    ``terms``
        The terms, each a name of your choosing and a table from each leg to
        its power, a whole number from 1; the outputs take the names, in
        this order. A leg is a column, or a column lagged ``k`` back,
        written ``"x@k"``: ``k`` learned rows without a clock, and in the
        clock's units with one, a duration such as ``"x@5m"`` on a
        temporal clock. ``{"m3": {"x": 3}}`` is the third moment of ``x``,
        ``{"xy2": {"x": 1, "y": 2}}`` a cross-moment and
        ``{"ac1": {"x": 1, "x@1": 1}}`` the first autocovariance, or under
        ``kind="standardized"`` the autocorrelation. No order is capped:
        ``max_state_bytes`` bounds the state. Required.
    ``features``
        The columns the legs name. The default, ``None``, is every column
        the terms name, lags taken off, in the order they first name them.
        A leg that is a feature's name is that column, so a column whose
        name holds ``@`` is read whole.
    ``kind``
        What each term is: ``"central"`` (the default), the moment about
        the weighted means; ``"raw"``, about zero; ``"standardized"``, the
        central moment over each leg's standard deviation to its power (a
        skew, a kurtosis, a correlation); ``"cumulant"``, of one leg. One
        word for every term, or a table from a term's name to its word,
        where a term left out is ``"central"``.
    ``bias``
        ``True`` reads every term as the plug-in moment of the weighted
        rows; ``False`` as the form unbiased for rows drawn independently
        under these weights (orders 2 to 4). The default, ``None``, gives
        each term the default its statistic has in Polars: a variance
        unbiased, as ``ewm_var``'s, and every other order the plug-in, as
        ``skew``'s and ``kurtosis``'s.
    ``se``
        Also emit each term's standard error, ``se_<term>``. The state then
        keeps the moments to twice each term's order. Default ``False``.
    ``lag_tolerance``
        How far before ``t - k`` the row a clock lag ``k`` pairs with may
        sit, in clock units, or a duration on a temporal clock. The
        default, ``None``, is each lag's own length. Needs a ``clock``.
    ``window_rows``
        The moments of the latest this many learned rows, in place of every
        row since the start. Refused beside ``window_size``.
    ``window_size``
        The moments of the rows whose clock is in ``(t - window_size, t]``,
        in clock units, or a duration on a temporal clock; the edge is
        decided on the raw clocks, as Polars' ``rolling_*_by`` decides it.
        Needs a ``clock``.
    ``max_state_bytes``
        The bytes a group's state may hold, its lag buffers and window
        included. A spec whose terms, row lags or ``window_rows`` pass it is
        refused. A clock lag's buffer and a ``window_size`` ring hold the
        rows inside a span of the clock, so they are checked as the rows go
        in: the first row to take a group past half of it raises a warning,
        and the row that would take it past the cap stops the run with an
        error naming the group and the bytes. Default 256 KB, ``262144``.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row, each read before the row is learned, so a field can be a
    feature for that same row without leaking it:

    ``var``, ``m3``, ``skew``, ``ac1``, ...
        Each term, under the name ``terms`` gives it, in the order it gives
        them. Null where the term is not defined: no weight yet, no lagged
        pair yet, a weight below ``min_weight``, a standardized term of a
        leg with no spread, an unbiased form whose denominator keeps no
        digit, or a value past a double's range.
    ``se_<term>``
        Under ``se=True``: each term's standard error, after every term, in
        the same order.
    ``weight_sum``
        The weight learned before the row, decayed, and inside the window
        under one.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``targets``, which this model does not take, and for a power that is
    not a whole number; and ``ValueError`` naming the problem for no term,
    a term with no leg or a leg named twice, a leg naming a column that is
    not among ``features``, a lag below 0, a lag that is neither a number
    nor a duration, a lag of part of a row without a clock or a duration
    without one, a power below 1, a cumulant of more than one leg,
    ``bias=False`` on a term past order 4, a word of ``kind`` that is no
    kind or a term ``kind`` names that ``terms`` does not have,
    ``lag_tolerance`` or ``window_size`` without a clock, ``window_rows``
    below 1 or beside ``window_size``, and a state past
    ``max_state_bytes``, or a ``max_state_bytes`` of 0.

    .. rubric:: The update

    A term is computed over the rows at which every one of its legs has a
    value, ``z_t``. Row ``t`` holds the weight ``ω_t = w_t Π_{s>t} λ_s``,
    its own weight times every later row's decay, and with ``W = Σ ω_t``:

    .. code-block:: text

        mean          m    = Σ ω_t z_t / W
        central       μ_a  = Σ ω_t (z_t - m)^a / W        (a, the powers over the legs)
        raw           μ'_a = Σ ω_t z_t^a / W
        standardized  s_a  = μ_a / Π_j μ_{2 e_j}^(a_j / 2)
        cumulant      κ_p  = μ_p - Σ_{j=2..p-2} C(p-1, j-1) κ_j μ_{p-j}    (one leg)

    With every weight 1 and no decay these are scipy's ``moment``,
    ``skew(bias=True)`` and ``kurtosis(bias=True, fisher=False)``, and a
    lagged term of order (1, 1) is numpy's ``cov(x[k:], x[:-k], bias=True)``.
    The state keeps, for the terms over the same legs, ``W``, the means, the
    central moments ``c_a = μ_a`` below each term, and the ratios
    ``r_k = Σ ω_t^k / W^k`` (``1 / r_2`` is Kish's effective number of
    rows). A row of weight ``w`` after a decay ``λ`` merges into them by
    Pébay's rule (Pébay, Terriberry, Kolla and Bennett 2016):

    .. code-block:: text

        W'   = λ W + w        a = λ W / W'        b = w / W'        d = z - m
        c'_a = Σ_{β ≤ a} C(a, β) a c_{a-β} (-b d)^β  +  b (a d)^a     (c_0 = 1, c_1 = 0)
        r'_k = a^k r_k + b^k
        m'   = m + b d

    Every product is at the scale of the spread, so a column at ``1e6``
    with a spread of 1 keeps its fourth moment, where power sums keep no
    digit of it. Decay multiplies ``W`` alone, so a long gap moves no
    moment. Where ``W'`` is 0, a first row of weight 0, ``a`` and ``b`` are
    0/0 and the update stops there: the state stays empty, with no NaN in
    it.

    **Unbiased forms.** For rows drawn independently, the plug-in moments'
    expectations are ``E[c_2] = (1 - r_2) μ_2``,
    ``E[c_3] = (1 - 3 r_2 + 2 r_3) μ_3`` and, for order 4, a combination of
    ``μ_4`` and ``μ_2²``. Inverting them gives the unbiased weighted
    moments of Rimoldini (2014) for reliability weights; with every weight
    1 they are scipy's ``kstat``, ``skew(bias=False)`` and
    ``kurtosis(bias=False)``. A term is withheld where a denominator keeps
    no digit: one row carrying all the weight, read from the weights, not
    from a count of rows.

    **Lags.** A lag ``k`` reads its column ``k`` learned rows back, or on a
    clock the latest learned row at or before ``t - k``, as Polars'
    ``join_asof`` with ``strategy="backward"`` pairs them, and no row where
    that one is more than ``lag_tolerance`` before ``t - k``: after a gap
    longer than the lag a row closes no pair. A pair weighs its closing
    row's weight. A row with no pair adds nothing to that term, which still
    decays, and counts for every other term. A session change or a gap past
    ``gap_cap`` empties the lag buffers; the moments go on. The pairs are
    centered on their own weighted means, so the autocorrelation is the
    pairs' Pearson correlation, which differs from statsmodels' ``acf``
    (the whole stream's mean over its variance) by a few thousandths at
    the lags of an autoregressive series.

    **Windows.** The window's rows are kept, and its moments read by
    merging summaries of its parts by the same rule, so a row leaving the
    window is never subtracted and nothing drifts; each part is decayed to
    the next one's end, so decay still weighs the window's rows. The
    standard error, under ``se``, is the delta method's at Kish's
    effective size, ``sqrt(r_2 E[IF²])`` for the term's influence function
    ``IF``.

    .. rubric:: Example

    This code draws a gamma column, whose skew is 1 and kurtosis 4.5, and
    an autoregressive one whose lag-1 autocorrelation is 0.6, and prints
    what the last row reads, from a decay whose half-life is 10,000 rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(5)
        n = 50_000
        e, ar = rng.standard_normal(n), np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.6 * ar[t - 1] + e[t]
        frame = pl.DataFrame({"g": rng.gamma(4.0, size=n), "ar": ar})

        model = rp.spec.ewm_moments(
            "shape",
            terms={"skew": {"g": 3}, "kurt": {"g": 4}, "ac1": {"ar": 1, "ar@1": 1}},
            kind="standardized",
            half_life=10_000.0,
        )
        last = rp.ModelBank([model]).fit_predict(frame)["shape"][-1]
        print(f"skew: true 1.00, found {last['skew']:.2f}")
        print(f"kurtosis: true 4.50, found {last['kurt']:.2f}")
        print(f"lag-1 autocorrelation: true 0.60, found {last['ac1']:.2f}")

    .. code-block:: text

        skew: true 1.00, found 0.99
        kurtosis: true 4.50, found 4.43
        lag-1 autocorrelation: true 0.60, found 0.60

    .. rubric:: See also

    - :func:`ew_cov`, for the means, variances and correlations of the
      columns and the statistics read off them.
    - :func:`marginal`, for each feature's moments against a target.
    """
    model: dict[str, Any] = {
        "type": "ewm_moments",
        "terms": {t: dict(legs) for t, legs in terms.items()},
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if kind is not None:
        model["kind"] = dict(kind) if isinstance(kind, dict) else kind
    if bias is not None:
        model["bias"] = bias
    if se:
        model["se"] = True
    if lag_tolerance is not None:
        model["lag_tolerance"] = lag_tolerance
    if window_rows is not None:
        model["window_rows"] = window_rows
    if window_size is not None:
        model["window_size"] = window_size
    if max_state_bytes is not None:
        model["max_state_bytes"] = max_state_bytes
    if features is None:
        features = _term_columns(terms)
        if not features and "targets" not in common:
            # Terms that name no column: the Rust side's refusal names the
            # terms, where an empty `features` would be blamed otherwise.
            return _common(name, model, targets=[], features=[], **common)
    targets = _mirror_target(name, "ewm_moments", features, common, "its moments are over")
    return _common(name, model, targets=targets, features=features, **common)
