"""The isotonic regression's builder, :func:`isotonic`: a part of
:mod:`rheon.spec`, split from ``_spec.py`` to keep it under the repository's
file-size cap. Import it from :mod:`rheon.spec`.
"""

from __future__ import annotations

from typing import Any, Literal, Unpack

from rheon._duration import Duration
from rheon._kwargs import CommonKwargs
from rheon._spec import TargetList, _checked, _common


@_checked
def isotonic(
    name: str,
    *,
    targets: TargetList,
    features: list[str],
    increasing: bool = True,
    y_min: float | None = None,
    y_max: float | None = None,
    out_of_bounds: Literal["nan", "clip"] = "nan",
    window_rows: int | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "both"] = "right",
    max_state_bytes: int | None = None,
    **common: Unpack[CommonKwargs],
) -> dict[str, Any]:
    """Isotonic regression of each target on one feature: the best
    monotone function of the feature, fitted to the rows before each row.

    Use it as a monotone calibrator, such as turning a classifier's score
    into a probability, or wherever a target should only rise (or only fall)
    with a feature and nothing else is assumed of the shape: the fit is
    scikit-learn's ``IsotonicRegression`` on the same rows and weights, one
    a target in each group. Each row reads the fit of the rows before it at
    its own feature value, then is learned, so the row's own target is never
    in its own prediction.

    .. rubric:: Parameters

    The stream parameters every builder takes, such as ``clock``, ``group``
    and ``weight``, are in :mod:`rheon.spec`. ``half_life`` or ``lam``
    weighs each row by its decay; unset, every row counts alike, as in
    scikit-learn's fit. A row's ``weight`` is its weight in the fit; scaling
    every weight changes nothing but ``weight_sum``. A row of weight 0
    counts in no clock and is in no window, and a row whose target is null
    adds nothing to that target's fit; both are still read.

    This model's own parameters:

    ``features``
        The one column each target is a monotone function of, such as the
        score to calibrate.
    ``increasing``
        Fit a non-decreasing function (``True``, the default) or a
        non-increasing one (``False``).
    ``y_min``, ``y_max``
        Clip the fit below and above, as scikit-learn's do; each a finite
        number, ``y_min`` at most ``y_max``. Unset, no clip.
    ``out_of_bounds``
        What a feature value outside the range of the rows learned reads:
        ``"nan"``, scikit-learn's default, withholds it; ``"clip"`` reads the
        fit at the nearer end of the range, usually what a calibrator wants.
    ``window_rows``
        Fit the last this many rows of positive weight alone. Its rows are
        checked against ``max_state_bytes`` when the spec is built. Refused
        beside ``window_size``.
    ``window_size``
        Fit the rows of the last this long on the clock alone: a number of
        the clock column's units (rows when the spec has no clock), or a
        duration on a temporal clock. The data decides how many rows that
        is, so they are checked as the rows go in (see ``max_state_bytes``).
        Either window may sit beside a ``half_life``: the rows in the window
        are then weighed by their decay.
    ``closed``
        Which edge of ``window_size`` holds the row exactly that old, in
        Polars' words (``rolling_sum_by(closed=)``). ``"right"``, the
        default, keeps the rows less than ``window_size`` old; ``"both"``
        keeps that row too and is refused without ``window_size``.
    ``max_state_bytes``
        The bytes a group's state may hold. Without a window the state
        holds each distinct feature value once a target, 40 bytes each, and
        under ``window_size`` each row in the window; the data decides how
        many, so they are checked as the rows go in: the first row past
        half of it is said in a notice, and a row that would pass it stops
        the run with an error naming the group. Round the feature in the
        query to bound its distinct values (``pl.col("score").round(3)``
        holds at most 1,001 between 0 and 1), or give a window. Default
        256 KB, ``262144``; at most 64 MiB.

    .. rubric:: Output

    One output column, named after the spec, holds a record of these fields
    in each row. `docs/OUTPUTS.md#isotonic
    <https://github.com/rheon-dev/rheon/blob/main/docs/OUTPUTS.md#isotonic>`_
    gives each field's type and when it is null:

    ``pred_<t>``
        The fit of the target's rows before this one (the window's, under
        a window) read at the row's feature. Null before the target's first
        row, and under ``out_of_bounds="nan"`` at a feature outside the
        range of its rows.
    ``resid_<t>``
        The target minus ``pred_<t>``.
    ``weight_sum``
        The weight of the rows the fit holds before the row: every learned
        row's, decayed, or the window's.
    ``settled_frac``, ``withheld_reason``
        As every model reports them (:mod:`rheon.spec`);
        ``withheld_reason`` is ``"not_identified"`` on a row whose
        prediction is null because the rows before it do not reach its
        feature.

    .. rubric:: Raises

    As every builder does (:mod:`rheon.spec`); and ``ValueError`` naming the
    problem for ``features`` that is not one column, a ``y_min`` or
    ``y_max`` that is not finite or a ``y_min`` above ``y_max``, an
    ``out_of_bounds`` other than ``"nan"`` or ``"clip"``, a ``window_rows``
    of 0, a ``window_size`` not above 0 and finite, the two windows
    together, ``closed="both"`` without ``window_size``, a ``window_rows``
    window past ``max_state_bytes``, or a ``max_state_bytes`` of 0 or past
    64 MiB. ``fit_predict`` raises when a group's state would pass
    ``max_state_bytes``.

    .. rubric:: The update

    For target ``t`` and the rows ``(x_i, y_i, w_i)`` learned before the
    row, each weight times its decay ``d_i`` (1 without a decay):

    .. code-block:: text

        f_t  = argmin_f  sum_i w_i d_i (y_i - f(x_i))²
               subject to f non-decreasing in x (non-increasing if not increasing)
        f_t  ← min(max(f_t, y_min), y_max)
        pred = f_t at the row's x, linear between the two nearest x fitted

    The fit is computed as scikit-learn computes it. Rows at the same
    feature value are pooled first: their weights summed, their value the
    weighted mean. Then pool-adjacent-violators runs over the distinct
    values in order, merging each run of neighbors that breaks the order
    into one block at its weighted mean (scipy's implementation, line for
    line). At a feature between two fitted values the prediction is the
    line between them, and at a fitted value that value (``numpy.interp``).
    With one distinct feature value the fit is that value everywhere. Each
    learned row refits the targets it carried in full: a row costs time in
    proportion to the distinct feature values held.

    **Exact against the library.** With no decay and no repeated feature
    value, each prediction is scikit-learn's on the rows before it to the
    bit, on a platform whose scipy and numpy compute the same unfused
    products and sums (scipy's and numpy's arm64 wheels fuse a multiply and
    an add into one rounding, a last-bit difference). Under a window the
    rows at one feature value are pooled in scikit-learn's order too. A
    repeated value without a window is pooled as its rows arrive, and a
    decayed weight is one ``exp2`` of the clock, so either agrees within a
    few units in the last place.

    **The state** is, a target, each distinct feature value with its weight
    and sum, or under a window each row in it; a decay keeps each weight
    scaled from an origin on the clock that moves every 128 half-lives, so
    no weight is rescaled on every row.

    .. rubric:: Example

    This code calibrates a score whose true probability is its square, and
    prints the calibrated probability at three scores after 20,000 rows:

    .. code-block:: python

        import numpy as np

        rng = np.random.default_rng(3)
        score = rng.uniform(0.0, 1.0, 20_000).round(2)
        label = (rng.uniform(0.0, 1.0, 20_000) < score**2).astype(float)
        frame = pl.DataFrame({"score": score, "label": label})
        frame.write_parquet("scores.parquet")

        model = rp.spec.isotonic(
            "cal", targets=["label"], features=["score"], out_of_bounds="clip"
        )
        bank = rp.ModelBank([model])
        bank.fit(pl.scan_parquet("scores.parquet"))
        probe = pl.DataFrame({"score": [0.2, 0.5, 0.9], "label": [None] * 3})
        out = bank.predict(probe).unnest("cal")
        print([round(p, 2) for p in out["pred_label"]])

    .. code-block:: text

        [0.05, 0.23, 0.8]

    .. rubric:: See also

    - :func:`marginal`, for a curve of a target against a feature that need
      not be monotone.
    - :func:`ew_class`, for a probability from several features at once.
    """
    model: dict[str, Any] = {
        "type": "isotonic",
        "increasing": None if increasing else False,
        "y_min": y_min,
        "y_max": y_max,
        "out_of_bounds": None if out_of_bounds == "nan" else out_of_bounds,
        "window_rows": window_rows,
        "window_size": window_size,
        "closed": None if closed == "right" else closed,
        "max_state_bytes": max_state_bytes,
    }
    # Each absent when not given, as the Rust spec leaves it out, so a bank
    # reports the dict this made (`ModelBank.specs`).
    model = {k: v for k, v in model.items() if v is not None}
    return _common(name, model, targets=targets, features=list(features), **common)
