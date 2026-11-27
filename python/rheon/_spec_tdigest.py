"""The streaming quantile sketch's builder, :func:`tdigest`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Unpack

from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def tdigest(
    name: str,
    *,
    targets: TargetList,
    quantiles: list[float],
    compression: float = 100.0,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Quantiles and expected shortfalls of each target column, from a
    merging t-digest whose error is uniform in rank.

    Use it for the quantiles of a column whose values sit near 0 or take
    both signs, where a sketch of relative accuracy misreads: returns,
    residuals, spreads. Each target gets its own digest in each group, and
    each row reports every level from the rows before it, so the row's own
    value is never in its own quantile. The digest never forgets: for a
    quantile that does, use ``ewm_quantile`` with a ``window_size``, the
    windowed DDSketch.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``group`` and
    ``weight``, are in :mod:`rheon.spec`; ``half_life`` and ``lam`` are
    refused, since a merging digest folds each row into a centroid's mean
    and cannot take it out again, and so has no decayed or windowed form.
    A row's ``weight`` is its mass in the digest; scaling every weight
    changes nothing but ``weight_sum``. A row of weight 0, or whose target
    is null, adds nothing to that target's digest; it is still read.

    This model's own parameters:

    ``quantiles``
        The levels read, each above 0 and below 1, no two alike; two outputs
        a level and a target, the quantile (the value at risk) and its
        tail's mean (the expected shortfall). A level below 0.5 reads the
        lower tail, one above it the upper; 0.5 itself has no tail of its
        own and is refused.
    ``compression``
        The digest's ``delta``, at least 10, with no ceiling but
        ``max_state_bytes``: at most ``delta + 1`` centroids a target, 16
        bytes each, beside a buffer of ``8 delta`` rows. Default 100,
        whose rank error was measured at most 2.0e-3 (held under 3.8e-3);
        200 halves it, at most 9.3e-4 (held under 1.8e-3), for twice the
        state. Both measured at 100,000 rows of normal,
        lognormal, Student-t(3) and Pareto(2.5) values over 20 seeds.
    ``max_state_bytes``
        The bytes a group's state may hold: each target's fullest digest
        and buffer. A spec past it is refused. Default 256 KB, ``262144``;
        at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#tdigest
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#tdigest>`_
    gives each field's type and when it is null:

    ``q<level>_<t>``
        Per target and level: the quantile of the target's rows before this
        one. Null until the rows identify the level (see the update), and
        below ``min_weight``.
    ``es<level>_<t>``
        Per target and level: the mean of the target's rows past
        ``q<level>_<t>``, below it at a level under 0.5 and above it at one
        over 0.5 (the expected shortfall), from the same digest. Its sign
        is the values', not a loss's. Null where ``q<level>_<t>`` is.
    ``weight_sum``
        The weight the model has learned before the row: every learned
        row's weight, with no decay.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`);
        ``withheld_reason`` is ``"level_beyond_sample"`` on a row where a
        level is null because its rows do not identify it yet.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); ``TypeError`` for
    ``features``, which this model does not take; and ``ValueError`` naming
    the problem for an empty ``quantiles``, a level not above 0 and below
    1, listed twice or of 0.5, a ``compression`` under 10,
    ``half_life`` or ``lam``, and a state past ``max_state_bytes``, or a
    ``max_state_bytes`` of 0 or past 64 MiB.

    .. rubric:: The update

    Each target ``t`` keeps a merging t-digest ``D_t`` (Dunning and Ertl,
    2019, scale function ``k_1``) over its rows, with the weight ``W_t``
    and the squared weights ``S_t`` of the rows it holds. For a row of
    weight ``w`` and value ``y``:

    .. code-block:: text

        q_p,t  = D_t.quantile(p)        read first: the rows before this one
        es_p,t = the centroids' mean past q_p,t, the boundary one split
        D_t   ← D_t + (y, w)           when w > 0 and y is present
        W_t   ← W_t + w,   S_t ← S_t + w²

    The digest keeps centroids ``(c, m)`` ascending by mean, each spanning
    at most 1 in ``k(q) = delta / (2π) asin(2q - 1)``, and merges a buffer
    of ``8 delta`` rows into them; the quantile interpolates between the
    centroids' centers, from the smallest row to the largest. The tail's
    mean is the partial-mass rule over the centroids: the mass
    ``m = p W_t`` from the smallest mean below 0.5, ``m = (1 - p) W_t``
    from the largest above it, each centroid wholly inside at its mean and
    the boundary one split, counted at only the weight the tail still
    needs:

    .. code-block:: text

        es_p,t = (sum_{i<j} m_i c_i + (m - M_{j-1}) c_j) / m

    It is within 4.3% of the exact tail mean, and within 1.2% at
    compression 200, on 100,000 rows of the four streams above.


    A level is reported where the rows identify it: with Kish's effective
    size ``n_t = W_t² / S_t`` (the rows themselves, for rows of equal
    weight), the ``i``-th smallest of ``n`` rows sits at level
    ``i / (n + 1)`` in expectation, so a level is null while

    .. code-block:: text

        p (n_t + 1) < 1   or   (1 - p) (n_t + 1) < 1

    that is, while it lies past the smallest or the largest row. Levels of
    0.25 and 0.75 are reported from the fourth row, 0.05 and 0.95 from the
    twentieth, 0.01 and 0.99 from the hundredth. The state is at most
    ``delta + 1`` centroids and ``8 delta`` buffered rows a target, 16 bytes each,
    whatever the stream's length.

    .. rubric:: Example

    This code reads the 5% and 95% quantiles of a normal column as it
    streams, with the mean of each tail, and prints the last row's:

    .. code-block:: python

        import numpy as np
        import polars as pl
        from rheon import polars as rp

        rng = np.random.default_rng(3)
        df = pl.DataFrame({"y": rng.standard_normal(20_000)})

        model = rp.spec.tdigest("d", targets=["y"], quantiles=[0.05, 0.95])
        out = rp.ModelBank([model]).fit_predict(df).unnest("d")
        last = out.row(-1, named=True)
        print(f"q0.05: {last['q0.05_y']:.2f}, es0.05: {last['es0.05_y']:.2f}")

    .. code-block:: text

        q0.05: -1.63, es0.05: -2.05

    .. rubric:: See also

    - ``ewm_quantile``, the decayed or windowed DDSketch, for a quantile
      that forgets, accurate relative to the value rather than in rank.
    - :func:`ew_cov`, for a decayed mean and variance.
    """
    model: dict[str, Any] = {"type": "tdigest", "quantiles": list(quantiles)}
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    if compression != 100.0:
        model["compression"] = compression
    return _common(name, model, targets=targets, features=[], **common)
