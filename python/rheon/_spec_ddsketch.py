"""The relative-accuracy quantile sketch's builder, :func:`ddsketch`: a part
of :mod:`rheon.spec`, split from ``_spec.py`` to keep it under the
repository's file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common

#: The default relative accuracy, ``tanh(1/128)``: ``gamma = e^(1/64)``.
DDSKETCH_ALPHA = math.tanh(1.0 / 128.0)


@_checked
def ddsketch(
    name: str,
    *,
    targets: TargetList,
    quantiles: list[float],
    alpha: float = DDSKETCH_ALPHA,
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Quantiles and expected shortfalls of each target column, from a
    DDSketch whose error is relative to the value.

    Use it for the quantiles of a column whose scale matters more than its
    rank, such as latencies, sizes, prices or losses, where a reading within
    1% of the true quantile is what is asked; for values near 0 or of both
    signs, use :func:`tdigest`. Each target gets its own sketch in each
    group, and each row reports every level from the rows before it, so the
    row's own value is never in its own quantile. It forgets by a window,
    exactly, or by a decay, rounded, or not at all.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group``
    and ``weight``, are in :mod:`rheon.spec`. ``half_life`` or ``lam``
    decays every bucket on the clock (the rounded form, below), and is
    refused beside a window; unset, the sketch never forgets. A row's
    ``weight`` is its mass in the sketch; scaling every
    weight changes nothing but ``weight_sum``. A row of weight 0 counts in
    no clock and is in no window, and a row whose target is null adds
    nothing to that target's sketch; both are still read.

    This model's own parameters:

    ``quantiles``
        The levels read, each above 0 and below 1, no two alike; two outputs
        a level and a target, the quantile (the value at risk) and its
        tail's mean (the expected shortfall). A level below 0.5 reads the
        lower tail, one above it the upper; 0.5 itself has no tail of its
        own and is refused.
    ``alpha``
        The relative accuracy, at least ``1e-6`` and below 1: every quantile
        read is within ``alpha`` of the exact one, relatively. Default
        ``tanh(1/128)``, about 0.0078, 64 buckets to a factor of ``e``; 0.01
        is the ``ddsketch`` package's default. Half the ``alpha`` doubles
        the buckets a range of values takes.
    ``window_rows``
        Read the last this many rows of positive weight alone. Its ring of
        ``window_rows (8 + 4 m)`` bytes for ``m`` targets is checked against
        ``max_state_bytes`` when the spec is built. Refused beside
        ``window_size``.
    ``window_size``
        Read the rows of the last this long on the clock alone: a number of
        the clock column's units (rows when the spec has no clock), or a
        duration on a temporal clock. The data decides how many rows that
        is, so its ring is checked as the rows go in (see
        ``max_state_bytes``).
    ``closed``
        Which edge of ``window_size`` holds the row exactly that old, in
        Polars' words (``rolling_sum_by(closed=)``). ``"right"``, the
        default, keeps the rows less than ``window_size`` old; ``"both"``
        keeps that row too and is refused without ``window_size``.
    ``max_state_bytes``
        The bytes a group's state may hold. The buckets are the data's
        log-range over ``ln gamma``, so they are checked as the rows go in:
        the first row past half of it is said in a notice, and a row that
        would pass it stops the run with an error naming the group. Default
        256 KB, ``262144``; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#ddsketch
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#ddsketch>`_
    gives each field's type and when it is null:

    ``q<level>_<t>``
        Per target and level: the quantile of the target's rows before this
        one (the window's, under a window). Null until the rows identify the
        level (see the update).
    ``es<level>_<t>``
        Per target and level: the mean of the target's rows past
        ``q<level>_<t>``, below it at a level under 0.5 and above it at one
        over 0.5 (the expected shortfall). Its sign is the values', not a
        loss's. Null where ``q<level>_<t>`` is.
    ``weight_sum``
        The weight of the rows the sketches hold before the row: every
        learned row's, or the window's.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`);
        ``withheld_reason`` is ``"level_beyond_sample"`` on a row where a
        level is null because its rows do not identify it yet.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for an empty ``quantiles``, a level not above 0 and below
    1, listed twice or of 0.5, an ``alpha`` below ``1e-6`` or not below 1, a
    ``window_rows`` of 0, a ``window_size`` not above 0 and finite, the two
    windows together, ``closed="both"`` without ``window_size``, a finite
    ``half_life`` or a ``lam`` below 1 beside a window, a ``window_rows``
    ring past
    ``max_state_bytes``, or a ``max_state_bytes`` of 0 or past 64 MiB.
    ``fit_predict`` raises when a group's buckets would pass
    ``max_state_bytes``.

    .. rubric:: The update

    **The decayed form is rounded.** Under ``half_life`` or ``lam`` every
    row, after its read, multiplies each bucket's weight, the zero count and
    the weights by the clock step's factor ``f`` before it adds itself, each
    multiply and add rounded once. So a bucket formed ``k`` rows ago holds
    its exact decayed sum within a relative ``2 k u / (1 - 2 k u)``,
    ``u = 2^-53``: about ``2.2e-12`` after 10,000 rows, far below ``alpha``.
    No row can be taken out of such a weight exactly, which is why the
    window, not the decay, is the form that drops a row; the factor is the
    platform's, so a decayed sketch agrees across platforms within its last
    bits rather than to the bit.

    Each target ``t`` keeps a DDSketch ``D_t`` (Masson, Rim and Lee, 2019,
    with the ``ddsketch`` package's logarithmic mapping). With
    ``gamma = (1 + alpha) / (1 - alpha)``, a value ``x > 0`` goes to the
    bucket ``k = ceil(log_gamma x)``, which covers ``(gamma^(k-1),
    gamma^k]`` and reads ``2 gamma^k / (1 + gamma)``, within ``alpha`` of
    every value in it; a negative value goes to a second store by its
    magnitude, and one within about ``2.2e-308`` of 0 to a count of its own. For a
    row of weight ``w`` and value ``y``:

    .. code-block:: text

        q_p,t  = the representative of the first bucket, in value order,
                 whose cumulative weight reaches p W_t
        es_p,t = (sum_{k<j} c_k r_k + (m - C_{j-1}) r_j) / m
        D_t    ← D_t + (y, w)           when w > 0 and y is present
        W_t    ← W_t + w,   S_t ← S_t + w²

    the tail mass ``m`` being ``p W_t`` from the smallest value below
    ``p = 0.5`` and ``(1 - p) W_t`` from the largest above it, ``c_k`` and
    ``r_k`` a bucket's weight and representative, ``C`` their running sum,
    and the bucket ``j`` the tail ends in counted at only the weight it
    still needs. So ``es`` is within ``alpha`` times the tail's mean
    magnitude of the exact tail mean. Under a window a row leaving takes
    its weight out of its bucket, and a bucket whose rows have all left
    holds exactly 0: the sketch is the one the window's rows alone build.
    ``log_gamma`` and ``gamma^k`` are summed from their series in plain
    arithmetic, so a row's bucket is the same on every platform.

    A level is reported where the rows identify it: with Kish's effective
    size ``n_t = W_t² / S_t``, a level is null while

    .. code-block:: text

        p (n_t + 1) < 1   or   (1 - p) (n_t + 1) < 1

    that is, while it lies past the smallest or the largest row in
    expectation. Levels of 0.25 and 0.75 are reported from the fourth row,
    0.05 and 0.95 from the twentieth, 0.01 and 0.99 from the hundredth.

    **The state** is one double a bucket from the smallest to the largest
    key held in each of two stores a target (a 32-bit row count beside each
    under a window), about 1,400 buckets a store for values spanning
    ``1e-5`` to ``1e5`` at the default ``alpha``: the data's range, never
    the stream's length. A window adds its ring.

    .. rubric:: Example

    This code reads the lower quartile and the 99% quantile of a lognormal
    column, each with the mean of its tail, as it streams, and prints the
    last row's 99% pair:

    .. code-block:: python

        import numpy as np
        import polars as pl
        from rheon import polars as rp

        rng = np.random.default_rng(3)
        df = pl.DataFrame({"y": rng.lognormal(0.0, 1.0, 20_000)})

        model = rp.spec.ddsketch("d", targets=["y"], quantiles=[0.25, 0.99])
        out = rp.ModelBank([model]).fit_predict(df).unnest("d")
        last = out.row(-1, named=True)
        print(f"q0.99: {last['q0.99_y']:.2f}, es0.99: {last['es0.99_y']:.2f}")

    .. code-block:: text

        q0.99: 10.18, es0.99: 14.95

    .. rubric:: See also

    - :func:`tdigest`, for a quantile whose error is uniform in rank.
    - ``ewm_quantile``, a decayed or windowed quantile as a window
      operator on any column.
    """
    model: dict[str, Any] = {
        "type": "ddsketch",
        "quantiles": list(quantiles),
        "alpha": None if alpha == DDSKETCH_ALPHA else alpha,
        "window_rows": window_rows,
        "window_size": window_size,
        "closed": None if closed == "right" else closed,
        "max_state_bytes": max_state_bytes,
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=targets, features=[], **common)
