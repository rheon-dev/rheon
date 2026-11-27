"""The realized-volatility builder, :func:`realized`: a part of
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
def realized(
    name: str,
    *,
    targets: TargetList,
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    lm_window: int | None = None,
    lm_beta: float = 0.99,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Realized volatility and the jump tests on it, for each target, one
    pass.

    Use it on returns: log returns of a price sampled at intervals, a
    session's worth at a time. For each target it measures the variance
    of the returns before each row in three ways, two of them blind to
    jumps, and tests the row for a jump in two: the difference between
    the measures over the window, and the row's own return against the
    volatility before it. Each target is its own series.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``,
    ``half_life``, ``gap_cap``, ``min_weight``, ``session`` and ``group``,
    are in :mod:`rheon.spec`. ``half_life`` is how fast the measures forget,
    ``inf`` for not at all; set it to ``inf`` beside a window. A row's
    ``weight`` counts its return that many times: the measures are sums, so
    they scale with the weights, on the sum scale. The count of returns
    ``n`` both tests read is not scaled: it is the count the weights are
    worth (below). A row of weight 0
    counts in no clock: the next row steps as if it were not there, it is
    in no window, its return is in no product, and its weight is in no
    ``weight_sum``; it is still read. A null target is that target's absent
    return: its products reach across the row to its last return. A gap
    past ``gap_cap`` or a session's edge is a step of the clock like any
    other, its products reaching across it; to measure each session on its
    own, set ``session_gap="reset"``.

    This model's own parameters:

    ``window_rows``
        Measure only the returns of the last this many rows of positive
        weight, in place of a decay. A window of ``W`` returns holds the
        ``W - 1`` products of its consecutive returns and the ``W - 2``
        triples, as the papers' sums over ``W`` returns do. Each row it
        holds keeps ``16 + 56 m`` bytes for ``m`` targets, checked against
        ``max_state_bytes`` here: a window past it is refused naming the
        largest that fits. Default none.
    ``window_size``
        Measure only the returns of the last this long on the clock, in
        clock units: a number of the clock's units, or of rows where the
        spec has no clock, or a duration such as ``"30m"`` on a temporal
        clock; finite and above 0. A row leaves the window when it is
        ``window_size`` old, decided on the rows' exact stamps; a row reads
        the window as it stood after the row before it. How many rows that
        holds is the data's, so its bytes are checked as the rows go in
        (see ``max_state_bytes``). Refused beside ``window_rows``. Default
        none.
    ``closed``
        Which edge of ``window_size`` holds the row exactly that old, in
        Polars' words (``rolling_sum_by(closed=)``): ``"right"``, the
        default, drops it, so the window is ``(T - window_size, T]``;
        ``"both"`` keeps it, and is refused without ``window_size``.
    ``lm_window``
        Lee and Mykland's window ``K``, a count of returns: the volatility
        of a row's return is read from the target's ``K - 1`` returns
        before it alone, their ``K - 2`` products, whatever the window or
        decay of the other measures. It counts returns, as the paper's
        ``K`` does, so it has no form on the clock. ``16 (K - 2) + 32``
        bytes a target. At least 3, by the statistic's definition: its
        volatility is the mean of ``K - 2`` products, and a window of fewer
        than 3 returns holds none. It is no warm-up: before ``K - 1``
        returns the mean is of the products there are. Default none: the
        volatility is read from the measures' own window or decay.
    ``lm_beta``
        The confidence of Lee and Mykland's jump flag, above 0 and below 1.
        Default 0.99.
    ``max_state_bytes``
        The bytes a group's state may hold, a window's rows included.
        ``window_rows`` and ``lm_window`` fix the size, so it is checked
        here. A ``window_size`` window holds the rows the span has, so it is
        checked as the rows go in: the first row to take a group past half
        of it raises a warning, once for the group, and the row that would
        take it past the cap stops the run with an error naming the group
        and the bytes. Without a window a group holds ``8 + 144 m`` bytes.
        Default 256 KB, ``262144``; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#realized
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#realized>`_
    gives each field's type and when it is null. Every field but ``lm`` and
    ``lm_jump`` is of the returns before the row; all are null below
    ``min_weight``:

    ``rv_<t>``
        Per target: the realized variance, the sum of the squared returns.
    ``bv_<t>``
        Per target: the bipower variation, which leaves jumps out.
    ``tv_<t>``
        Per target: the tripower variation, which leaves jumps out too.
    ``tq_<t>``
        Per target: the tripower quarticity, the integrated quarticity the
        test is scaled by.
    ``bns_<t>``
        Per target: Barndorff-Nielsen and Shephard's adjusted ratio test, a
        standard normal without jumps, large and positive with them. Null
        where ``rv`` or ``bv`` is 0.
    ``lm_<t>``
        Per target: Lee and Mykland's statistic of the row's own return, the
        return over the volatility of the returns before it. ``sqrt(2/π)``
        times it is a standard normal without a jump, a z-score. Null where
        the row has no return or the volatility is 0.
    ``lm_jump_<t>``
        Per target: whether ``lm`` is a jump at confidence ``lm_beta``, by
        the paper's test (below). Null where ``lm`` is.
    ``weight_sum``
        The weight the model has accumulated before the row and before the
        row's own decay, every row's, or the weight inside the window.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`).

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for a window beside a decay, ``window_rows`` of 0 or past
    ``max_state_bytes``, ``window_size`` not finite or not above 0, both
    windows, ``closed="both"`` without ``window_size``, ``lm_window`` below
    3, ``lm_beta`` not in ``(0, 1)``, and a ``max_state_bytes`` of 0 or past
    64 MiB.

    .. rubric:: The update

    For ``n`` returns ``r_j``, ``a_j = |r_j|``, and ``μ_p = E|Z|^p`` of a
    standard normal (Barndorff-Nielsen and Shephard 2004, 2006; Lee and
    Mykland 2008):

    .. code-block:: text

        RV  = Σ r_j²
        BV  = μ_1^-2 Σ_{j≥2} a_j a_{j-1}                      μ_1^-2 = π/2
        TV  = μ_{2/3}^-3 Σ_{j≥3} (a_j a_{j-1} a_{j-2})^(2/3)
        TQ  = n μ_{4/3}^-3 Σ_{j≥3} (a_j a_{j-1} a_{j-2})^(4/3)
        bns = sqrt(n) (1 - BV/RV) / sqrt(ϑ max(1, TQ/BV²))      ϑ = π²/4 + π - 5
        lm  = r_i / σ_i,   σ_i² = the mean of a_j a_{j-1} over the products before row i

    Online, each sum decays by ``0.5 ** (Δclock / half_life)`` a row and
    then takes the row's terms at the weight of the latest row in each. ``n``
    is Kish's effective count of the target's returns' weights ``w_j``,
    decayed or in a window, ``(Σ w_j)² / Σ w_j²``, which is the count itself
    at equal weights without a decay; ``bns`` and the flag read it.
    ``TQ``'s factor is the weight ``Σ w_j`` in its place, the length of the
    period the sums are of, so ``TQ/BV²`` is 1 at a constant volatility
    whatever the weights: with unequal weights ``1 - BV/RV`` has the
    variance ``ϑ Σ w_j² / (Σ w_j)² = ϑ / n``, so ``bns`` is a standard
    normal without jumps under a decay too. No factor such as ``n/(n-1)`` is
    applied: the papers have none. The flag is Lee and Mykland's test of
    the largest ``|lm|`` among ``n`` returns, with ``c = sqrt(2/π)``:

    .. code-block:: text

        C_n = sqrt(2 ln n)/c - (ln π + ln ln n) / (2 c sqrt(2 ln n))
        S_n = 1 / (c sqrt(2 ln n))
        lm_jump = (|lm| - C_n) / S_n > -ln(-ln lm_beta)

    ``lm_beta = 0.99`` is a test at 1%, ``-ln(-ln 0.99) = 4.6001``. ``n``
    is the target's own count of returns, the ``n`` of the other measures,
    so the threshold is that of the stretch they are measured over; the
    flag is null where ``n`` is 1 or less, where ``C_n`` has no value.
    The sums are compensated pairs, so a day of 23,400 returns keeps
    the rounding of its terms, and a window takes each leaving term back
    exactly.

    .. rubric:: Example

    This code makes a day of 2,340 returns with one jump of 12 standard
    deviations at row 1,000, and prints the test over the day, past the 5%
    level of 1.645, and the one row flagged, the jump:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        r = 0.0005 * rng.standard_normal(2_340)
        r[1_000] += 0.006
        pl.DataFrame({"r": r}).write_parquet("returns.parquet")

        model = rp.spec.realized("v", targets=["r"], half_life=float("inf"), lm_window=200)
        bank = rp.ModelBank([model])
        out = bank.fit_predict(pl.read_parquet("returns.parquet")).unnest("v")
        print(f"bns over the day: {out['bns_r'][-1]:.1f}")
        print(f"flags: {out['lm_jump_r'].sum()}, at row {out['lm_jump_r'].arg_max()}")

    .. code-block:: text

        bns over the day: 2.0
        flags: 1, at row 1000

    .. rubric:: See also

    - :func:`rcov`, for realized covariance robust to microstructure noise
      (two-scale, pre-averaged and kernel estimators over a block).
    - :func:`garch`, for a conditional variance fitted to the returns.

    .. rubric:: Notes

    **The window's convention.** A window of ``W`` returns holds ``W - 1``
    products: a product is in the window only while both of its returns
    are, so the measures of a window are the papers' on its returns. Under
    a decay the first return's product is with nothing, as on a day.

    """
    model: dict[str, Any] = {"type": "realized"}
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if window_rows is not None:
        model["window_rows"] = window_rows
    if window_size is not None:
        model["window_size"] = window_size
    if closed != "right":
        model["closed"] = closed
    if lm_window is not None:
        model["lm_window"] = lm_window
    if lm_beta != 0.99:
        model["lm_beta"] = lm_beta
    return _common(name, model, targets=targets, features=[], **common)
