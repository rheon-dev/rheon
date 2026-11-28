"""The window operators: exponentially weighted (EW) means, sums, rates and
quantiles over a stream's clock, looking back or ahead, a variance and a
standard deviation looking back, and the largest and smallest value of a
window, or since an event. ``increment``, the step of a running total,
and ``tick_sign``, the sign of a price's last change, give a window
operator an input. :func:`invert` solves an equation of the row for
:func:`unknown`, in the same run: a root a row, never a window. ``run_length``
counts the rows a value has held, ``clock_lag`` and ``clock_drift`` read the
lag between an event's time and its arrival, and how fast it grows, and
``clock_step`` the steps between a time and the one before it: a stream's
health, read as columns. :func:`bspline`, :func:`natural_spline`,
:func:`hinge` and :func:`interactions` expand numeric columns into stateless
basis and interaction terms as plain Polars expressions.

Each operator returns a ``pl.Expr`` that composes with Polars' own
expressions, in which ``pl.col`` reads the current row.
:func:`rheon.polars.stream.with_windows` computes the expressions in one
pass, holding one window of rows in memory, or a spec takes one as its
target. A target that looks ahead is known only once its window has
closed, so the bank learns it then, under an ``embargo``
(:mod:`rheon.spec`) of at least the window. The guide `How a window
operator weighs the rows near each row
<https://github.com/rheon-dev/rheon/blob/main/docs/WINDOWS.md>`_ explains
each.

Every operator is defined once, by the library that has it, and held to it
in the tests. A *stamp* is a clock value, which several
rows may share. A *clock break* is a gap past ``gap_cap`` or a session
change, and a *stretch* is the rows between two clock breaks. ``tau_t`` is
row *t*'s clock with each step capped at ``gap_cap``, and ``lam = 2 ** (-1 /
half_life)``:

.. list-table::
   :header-rows: 1
   :widths: 22 44 34

   * - operator
     - defined as
     - held to
   * - :func:`ewm_mean`
     - the *time-weighted* mean by default (``adjust=None``): row *i*'s value
       is held over ``(t_{i-1}, t_i]``, a stretch's first value is held from
       before the stretch begins, and the mean over ``(tau_t - w, tau_t]``
       weighs each held value by ``integral lam**(tau_t - s) ds`` over its
       interval inside the window. With ``adjust=True``, each observation
       carries point weight ``lam**(tau_t - t_j)``
     - Polars' ``ewm_mean_by`` (default) and ``ewm_mean(adjust=True)`` (under
       ``adjust=True`` on a row-count clock)
   * - :func:`ewm_mean` with ``reverse=True``
     - the mirror: value *j* is held over ``[t_j, t_{j+1})``, until the next
       row, and weighed by ``integral lam**(s - tau_t) ds`` inside ``(tau_t,
       tau_t + w]``
     - the reversed stream's :func:`ewm_mean`
   * - :func:`ewm_sum`, and with ``reverse=True``
     - ``sum lam**|t_j - tau_t| * x_j`` over the rows of the window, each
       counted once at its own time
     - Polars' ``ewm_sum_by`` on distinct stamps. At a repeated stamp every
       row holds the stamp's whole window, so each carries the stamp's
       total, where ``ewm_sum_by``'s is a running sum
   * - :func:`ewm_rate`, and with ``reverse=True``
     - the sum over the decayed time the window covers, ``integral_0^T lam**s
       ds = half_life / ln 2 * (1 - 2 ** (-T / half_life))``, ``T`` the
       window's span inside the stretch: a quantity per unit time
     - the sum and the mass, each as above
   * - :func:`ewm_var`, :func:`ewm_std`
     - the variance about :func:`ewm_mean`, each value under the mean's
       weight ``m_i``: ``sum m_i (x_i - mu) ** 2 / V1``, ``V1 = sum m_i``.
       Unless ``bias``, it is multiplied by ``V1 ** 2 / (V1 ** 2 - V2)``,
       ``V2 = sum m_i ** 2``. :func:`ewm_std` is its root. Looking back only
     - Polars' ``ewm_var`` and ``ewm_std`` under ``adjust=False`` on a
       clock that steps by 1 a row and an input with no nulls, with Polars'
       own correction for unequal weights
   * - :func:`ewm_moments`
     - one term of the decayed weighted moments of one or more columns, each
       perhaps lagged: a central, raw or standardized moment, or a
       cumulant, each value weighing ``m_i`` as in :func:`ewm_var`. Looking
       back only
     - the weighted definitions (numpy's ``average`` of the powers), scipy's
       ``moment``, ``skew``, ``kurtosis`` and ``kstat`` at equal weights,
       and :func:`rheon.spec.ewm_moments` fed the same weights
   * - :func:`ewm_quantile`
     - the smallest ``x`` with ``W(<= x) >= q W``, each row of the window
       weighing ``lam**|t_j - tau_t|``, as in :func:`ewm_sum`, ``W`` their
       total. It is read from buckets each spanning a ratio of at most
       ``e**(1/64)``, so it is within ``tanh(1/128)``, 0.78%, of the exact
       quantile, relatively
     - numpy's ``inverted_cdf`` quantile over the window's values and
       their weights
   * - :func:`rolling_max`, :func:`rolling_min`
     - the largest or smallest value of the rows the window holds, each row
       counted once whatever its age; with no ``window_size``, every row
       since the stretch began or, under ``since``, since the last stamp
       holding an event
     - Polars' ``rolling_max_by`` and ``rolling_min_by``, bit for bit, the
       earliest row winning a tie of ``0.0`` and ``-0.0``. A NaN or an
       infinity is missing here, where Polars lets a NaN make the window NaN
       and counts an infinity
   * - :func:`rolling_quantile`
     - exact order-statistic quantile of the values in the window under any of
       Polars' six ``interpolation`` rules, with bounded state
     - Polars' ``rolling_quantile_by``, bit for bit
   * - :func:`rolling_rank`
     - exact rank of the row's own value inside its window under Polars' four
       deterministic tie methods (``"average"``, ``"min"``, ``"max"``,
       ``"dense"``), with bounded state
     - Polars' ``rolling_rank_by``, bit for bit (as ``Float64``)
   * - :func:`increment`
     - ``x_i - x_{i-1}`` within the group and session, null on a session's
       first row, and in seconds on a temporal column
     - the row before it
   * - :func:`tick_sign`
     - the sign of :func:`increment`, +1 or -1, and over a step of 0 the
       sign of the last step that was not 0: the tick rule
     - bar-forge's ``tick_rule_signs``, and Polars' ``sign`` of ``diff``
       with the last nonzero sign filled forward
   * - :func:`run_length`
     - the rows, ending at this one, whose value is unchanged from the row
       before: each row's place in its run, a run of nulls a run of its own
     - Polars' ``rle``
   * - :func:`clock_lag`, :func:`clock_drift`
     - with ``d_i = a_i - e_i`` the lag of the arrival time behind the event
       time and ``w_i = lam**(tau_t - tau_i)``, the weighted mean of ``d``,
       and the weighted least-squares slope of ``d`` on ``a``
     - numpy's weighted mean, and statsmodels' ``WLS`` under the same weights
   * - :func:`clock_step`
     - with ``s_i = t_i - t_{i-1}`` each step of a time and ``w_i =
       lam**(tau_t - tau_i)``, the weighted mean of the steps, their
       variance (unless ``bias``, multiplied by ``W ** 2 / (W ** 2 - sum
       w_i ** 2)``) or its root, or the weight of the steps of 0 or below 0
     - Polars' ``ewm_mean``, ``ewm_var`` and ``ewm_std`` of ``diff()`` on
       a clock of one unit a row, and statsmodels' ``DescrStatsW`` under the
       same weights on a clock

**Which rows a window holds follows Polars' rolling windows**
(``rolling_*_by``). A window is a set of timestamps, and ``closed`` says
which ends are in. Every row at one stamp gets the same backward window,
later rows at that stamp included, and looking ahead the mirror holds:

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - ``closed``
     - looking back
     - looking ahead
   * - ``"right"`` (the default)
     - ``(tau_t - w, tau_t]``: the stamp's own rows are in, so the output
       waits for the next distinct stamp
     - ``(tau_t, tau_t + w]``: no row at the stamp is in, and a row exactly
       ``w`` later is
   * - ``"left"``
     - ``[tau_t - w, tau_t)``: the stamp's own rows are outside, and the
       output is known as the stamp arrives
     - ``[tau_t, tau_t + w)``: every row at the row's own stamp is in, the
       row itself included
   * - ``"both"``
     - both ends in, so the output waits for the next stamp
     - both ends in: the stamp's rows, and a row exactly ``w`` later
   * - ``"none"``
     - neither end, so the output is known as the stamp arrives
     - neither end

An edge between two rows is decided from the difference of their clocks,
exact in nanoseconds on a temporal clock, as Polars decides it, at any age
of the stream. ``min_samples`` (Polars' name) nulls a window holding fewer
rows with a value. A value that is NaN, infinite or beyond 1e100 in
magnitude is read as missing, as a null is. Every model reads its inputs
within that bound, since a weight of 1e100 times a square of 1e100 is
1e300, still inside a double's range. At a row whose value is missing the
operator gives the window as it stands, where Polars' ``ewm_mean_by`` gives
null. A row is in a window by its stamp, but a mean weighs it by its held
interval inside the window. So under ``"left"`` or ``"both"``, a row exactly
one window old counts for ``min_samples`` and weighs nothing, since its
interval ends at its row. A mean with no other value in the window is null,
where a sum's default ``min_samples=0`` gives 0.0, as Polars' ``rolling_sum_by``
does.

**A weighted mean is a ratio of two sums.** A volume-weighted average price
(VWAP) is decayed notional over decayed volume, and the time mass cancels.
For one side's VWAP, put ``when/then`` inside both sums, with
``min_samples=1`` in both: a window with no such trade is then null, where a
sum of nothing is 0.0 and its ratio NaN. The operators take no ``weight=``.
This code reads ``trades`` from `Example data
<https://github.com/rheon-dev/rheon/blob/main/docs/EXAMPLE-DATA.md>`_:

.. code-block:: python

    notional = pl.col("price") * pl.col("quantity")
    buys = pl.when(pl.col("side") == "buy")
    out = rp.stream.with_windows(
        trades,
        mid_trend=rp.ops.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
        fwd_vwap_vs_mid=rp.ops.ewm_sum(notional, half_life="10s", window_size="1m",
                                       reverse=True)
                        / rp.ops.ewm_sum("quantity", half_life="10s", window_size="1m",
                                         reverse=True)
                        - pl.col("mid"),
        buy_vwap=rp.ops.ewm_sum(buys.then(notional), half_life="10s", min_samples=1)
                 / rp.ops.ewm_sum(buys.then("quantity"), half_life="10s", min_samples=1),
        clock="ts", gap_cap="5m", group="symbol",
    )

**A feature in units of its spread is a deviation over an operator.**
``(x - ewm_mean(x)) / ewm_std(x)`` is the feature's z-score. A model fitted
on it has a coefficient per standard deviation of the feature, as the
spread stands on each row, where a model on ``x`` has one per unit. Where a
target responds to a feature relative to its volatility, fit on the
z-score. Every model reads it as a column. `Scale a feature by its spread
<https://github.com/rheon-dev/rheon/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_
gives the measured gain. This code reads ``trades.parquet`` from
`Example data <https://github.com/rheon-dev/rheon/blob/main/docs/EXAMPLE-DATA.md>`_:

.. code-block:: python

    spread = {"half_life": "30m"}
    z_mid = (pl.col("mid") - rp.ops.ewm_mean("mid", **spread)) / rp.ops.ewm_std("mid", **spread)
    zscored = pl.scan_parquet("trades.parquet").rheon.with_windows(
        z_mid=z_mid, clock="ts", gap_cap="5m", group="symbol"
    ).collect()

**The clock settings are given to the call**, and every operator in it
shares them: ``clock``, ``gap_cap``, ``restart_after_step_back``, ``session``,
``session_gap`` and ``group``, in a spec's own words
(:mod:`rheon.spec`). They differ from a spec's in one way: a
session needs no ``session_gap``, since no window measures the clock's step
across a session change. A number given for it changes nothing, and only
``"reset"`` acts. A gap past ``gap_cap`` or a session change ends every
window open across it, and ``partial`` says what such a window gives. A
backward window whose span reaches before its stretch's first row is
partial too, the stream's own start included. So ``partial`` also says what
every stream's first ``window_size`` gives:

.. list-table::
   :header-rows: 1
   :widths: 16 54 30

   * - ``partial``
     - the window gives
     - the default
   * - ``"keep"``
     - the value over what it saw, the window ending at the last row seen
     - looking back
   * - ``"null"``
     - null
     - looking ahead
   * - ``"drop"``
     - the row leaves the output
     -

A reset discards a window: it gives null, and the row is never dropped.
Under ``session_gap="reset"`` a session change is a reset, so it discards
the windows rather than cuts them. A forward window under ``closed="right"``
or ``"both"`` whose far edge is the last row before a cut or a reset, or
its group's last row when the input ends, is whole: every row it covers has
arrived, so it gives its value. A forward window whose far edge lies past
the last row of the input is null. Under ``group`` a forward
window needs a clock column, since only a clock ends the windows of a group
that falls silent. ``partial`` needs a ``window_size``: without one no
window is cut short, so ``partial`` alone is refused.

An operator's input is a column name or an element-wise expression of the
row, :func:`increment` and :func:`tick_sign` included, and never another
window operator. For a
formula over an operator's output, add the inner operator as a column with
one call, and pass that column to a second. ``half_life`` and
``window_size`` are in the clock's units: numbers on a numeric clock, and
durations (``"10s"``, a ``timedelta``, ``pl.duration``) on a temporal one.
``half_life=float("inf")`` weighs the window evenly, and needs a
``window_size`` for a mean, a variance or a standard deviation.

An operator checks its own arguments when it is called, and names the one
it refuses. Each operator's *Raises* lists them.
:func:`rheon.polars.stream.with_windows` refuses the rest when it builds
its query: ``half_life=float("inf")`` with no ``window_size`` for a mean, a
variance or a standard deviation, and a window looking ahead under
``group`` with no clock column.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import polars as pl

from rheon._errors import RefusedValueError
from rheon._renamed import renamed_function
from rheon.polars._basis import bspline, hinge, interactions, natural_spline
from rheon.polars._formula import (
    UNKNOWN_COLUMN,
    invert_node,
    moments_operator,
    operator,
    profile,
    row_operator,
    seasonal_decomposition,
)
from rheon.polars._fourier import calendar_fourier, fourier

if TYPE_CHECKING:
    from rheon._duration import Duration

__all__ = [
    "bspline",
    "calendar_fourier",
    "cross_mean",
    "cross_std",
    "cross_rank",
    "cross_quantile",
    "cross_ref",
    "ewm_mean",
    "ewm_moments",
    "ewm_profile",
    "ewm_quantile",
    "ewm_rate",
    "ewm_std",
    "ewm_sum",
    "ewm_var",
    "fourier",
    "hinge",
    "increment",
    "interactions",
    "invert",
    "mstl",
    "natural_spline",
    "rolling_max",
    "rolling_min",
    "rolling_quantile",
    "rolling_rank",
    "stl",
    "tick_sign",
    "run_length",
    "clock_lag",
    "clock_drift",
    "clock_step",
    "unknown",
]


def _window(
    name: str,
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None,
    closed: Literal["right", "left", "both", "none"],
    min_samples: int,
    partial: Literal["keep", "null", "drop"] | None,
    reverse: bool,
    quantile: float | None = None,
    adjust: bool | None = None,
) -> pl.Expr:
    """The mean, sum, rate or quantile ``name`` names, looking back, or ahead
    under ``reverse``: the operator over the rows after each row, which the
    tree keeps as ``r<name>``."""
    who = f"rp.ops.{name}"
    if not isinstance(reverse, bool):
        raise TypeError(
            f"{who}: reverse must be True or False, but it is {reverse!r}. "
            "Pass reverse=True to read the rows after each row, or reverse=False."
        )
    if reverse and window_size is None:
        raise RefusedValueError(
            f"{who}: reverse=True needs window_size, but it is not set, and the rows after a "
            "row have no end without it. Set window_size to the span to read ahead."
        )
    return operator(
        f"r{name}" if reverse else name,
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        quantile=quantile,
        adjust=adjust,
        who=f"rp.ops.{name}(reverse=True)" if reverse else None,
    )


def ewm_mean(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
    adjust: bool | None = None,
) -> pl.Expr:
    """The time-weighted EW mean of a column over each row's window.

    Use it for a trailing or leading level of a value, such as the mid's recent
    mean, measured on the stream's clock. For a sum or a quantity per unit
    time, use :func:`ewm_sum` or :func:`ewm_rate`, and for the spread about
    this mean, :func:`ewm_std`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly,
        and for this operator needs ``window_size``:
        :func:`~rheon.polars.stream.with_windows`, or a spec that takes it in
        a formula target, refuses it without one.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.
    ``adjust``
        ``None`` (the default) computes the time-weighted mean (Polars'
        ``ewm_mean_by``). ``True`` computes the point-mass EW mean, weighing
        each observation by ``lam ** |tau_t - t_j|``: on a row-count clock
        (no ``clock`` column) this matches Polars' ``ewm_mean(adjust=True,
        ignore_nulls=False)`` on every non-null row, and on a clock column it
        also supports durations, ``window_size``, ``closed``, ``partial``,
        ``reverse=True``, ``gap_cap``, ``session`` and ``half_life=float("inf")``
        without ``window_size``. ``False`` is refused (leave ``adjust=None``
        for the time-weighted form).

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is a ``ValueError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = rp.stream.with_windows(
            rows,
            back=rp.ops.ewm_mean("x", half_life=1.0),        # the rows at or before each row
            ahead=rp.ops.ewm_mean("x", half_life=1.0, window_size=2.0, reverse=True),  # next 2
            clock="t", gap_cap=10.0,
        )
        polars = rows.select(pl.col("x").ewm_mean_by(pl.col("t").cast(pl.Int64), half_life="1i"))
        print(out["back"].round(4).to_list(), polars["x"].round(4).to_list())   # equal
        print(out["ahead"].round(4).to_list())   # null where a window reaches past the input's end

    .. code-block:: text

        [1.0, 1.5, 2.25, 3.125, 4.0625] [1.0, 1.5, 2.25, 3.125, 4.0625]
        [2.0, 3.0, 4.0, None, None]

    .. rubric:: See also

    - :func:`ewm_std`, the spread about this mean.
    - :func:`ewm_sum`, for a weighted mean as a ratio of two sums.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `How a window operator weighs the rows near each row
      <https://github.com/rheon-dev/rheon/blob/main/docs/WINDOWS.md>`_,
      the guide.

    .. rubric:: Notes

    Looking back, it is Polars' ``ewm_mean_by`` recursion,
    ``y_i = a_i x_i + (1 - a_i) y_{i-1}`` with
    ``a_i = 1 - lam ** (t_i - t_{i-1})``, ``lam = 2 ** (-1 / half_life)`` and
    ``y_1 = x_1``. Each value is weighed by the decayed time of the interval
    ending at its row. Under a ``window_size`` only the part of each interval
    inside the window counts.

    Looking ahead, it is the mirror: each value is held until the next row, and
    weighed by the decayed time from the row the window is read at.
    """
    return _window(
        "ewm_mean",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
        adjust=adjust,
    )


def ewm_sum(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 0,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """The EW sum of a column over each row's window.

    Use it for a decayed total, such as recent notional, and for a weighted
    mean as a ratio of two sums. For a mean weighted by time, use
    :func:`ewm_mean`, and for a total per unit time, :func:`ewm_rate`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 0, as in Polars' ``rolling_sum_by``:
        a window with no row with a value sums to 0.0. With 1 or more, such a
        window is null. An integer from 0 to ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is 0.0 where the
    window holds no row with a value, under the default ``min_samples=0``, and
    null where it holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 0 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is a ``ValueError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        s = rp.ops.ewm_sum("x", half_life=1.0)
        out = rp.stream.with_windows(rows, s=s, clock="t", gap_cap=10)
        polars = rows.select(pl.col("x").ewm_sum_by(pl.col("t").cast(pl.Int64), half_life="1i"))
        print(out["s"].round(4).to_list(), polars["x"].round(4).to_list())   # the same as Polars'

    .. code-block:: text

        [1.0, 2.5, 4.25, 6.125, 8.0625] [1.0, 2.5, 4.25, 6.125, 8.0625]

    .. rubric:: See also

    - :func:`ewm_rate`, the sum per unit of clock time.
    - :func:`ewm_mean`, the mean weighted by time.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `Make a weighted mean, such as a VWAP
      <https://github.com/rheon-dev/rheon/blob/main/docs/how-to/make-window-features.md#make-a-weighted-mean-such-as-a-vwap>`_,
      the forward VWAP built from two sums.

    .. rubric:: Notes

    Looking back it is ``sum lam ** (tau_t - t_j) * x_j`` over the rows at or
    before row *t*, less than ``window_size`` older, with
    ``lam = 2 ** (-1 / half_life)`` and ``tau_t`` the row's time. Looking ahead
    it is ``sum lam ** (t_j - tau_t) * x_j`` over the rows after it, at most
    ``window_size`` later. On distinct stamps it is Polars' ``ewm_sum_by``,
    each value counted once at its own time. At a repeated stamp every row
    carries the stamp's total, where ``ewm_sum_by``'s is a running sum.
    Polars added ``ewm_sum_by`` in py-polars 1.43.0, so the example's
    comparison with it needs that version or later.
    """
    return _window(
        "ewm_sum",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
    )


def ewm_rate(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """The EW sum of a column per unit of clock time over each row's window.

    Use it for a quantity per unit time, such as a volume rate from
    :func:`increment` of a running volume. For the total itself, use
    :func:`ewm_sum`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value, and where the
    window spans no clock time, such as at a stretch's first stamp. It is
    null too where ``partial="null"`` meets a cut window, where a forward
    window is still open at the input's end, and after a reset (the
    module's ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is a ``ValueError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # half_life=inf weighs the window evenly: the sum over the last 2 units, per unit.
        rate = rp.ops.ewm_rate("x", half_life=float("inf"), window_size=2.0)
        out = rp.stream.with_windows(rows, r=rate, clock="t", gap_cap=10.0)
        print(out["r"].round(4).to_list())   # null on row 0, whose window spans no time yet

    .. code-block:: text

        [None, 3.0, 2.5, 3.5, 4.5]

    .. rubric:: See also

    - :func:`ewm_sum`, the total it divides.
    - :func:`increment`, the steps of a running total.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `How a window operator weighs the rows near each row
      <https://github.com/rheon-dev/rheon/blob/main/docs/WINDOWS.md>`_,
      the guide.

    .. rubric:: Notes

    It is :func:`ewm_sum` over the decayed time the window covers,
    ``half_life / ln 2 * (1 - 2 ** (-T / half_life))``, with ``T`` the window's
    span inside the stretch. Looking ahead it is the sum ahead over the decayed
    time the window ahead covers.
    """
    return _window(
        "ewm_rate",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
    )


def ewm_quantile(
    input: str | pl.Expr,
    *,
    quantile: float,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
) -> pl.Expr:
    """The EW quantile of a column over each row's window.

    Use it for a level that one wild row does not move, such as a median, or
    for a robust scale (Notes). For a mean, use :func:`ewm_mean`, and for a
    standard deviation, :func:`ewm_std`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``quantile``
        The level of the quantile, above 0 and below 1, such as 0.5 for the
        median. Required.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
        Required under ``reverse=True``, since the rows after a row have no end
        without it.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"`` looking back and ``"null"`` looking ahead. It needs
        ``window_size``, since without one no window is cut short.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it, as Polars'
        ``cum_sum(reverse=True)`` runs from the end. As a spec's target, a
        window looking ahead is known only once it has closed, so the bank
        learns it under an ``embargo`` of at least ``window_size``. Default
        ``False``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``;
        - ``quantile`` is not a number.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``quantile`` is not above 0 and below 1;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is a ``ValueError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # The median of the last 3 units, weighed evenly.
        med = rp.ops.ewm_quantile("x", quantile=0.5, half_life=float("inf"), window_size=3.0)
        out = rp.stream.with_windows(rows, med=med, clock="t", gap_cap=10.0)
        print(out["med"].round(3).to_list())   # within 0.78% of the exact medians 1, 1, 2, 3, 4

    .. code-block:: text

        [1.008, 1.008, 2.016, 3.016, 4.031]

    .. rubric:: See also

    - :func:`ewm_sum`, which weighs the rows the same way.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `Quantiles and a robust scale
      <https://github.com/rheon-dev/rheon/blob/main/docs/WINDOWS.md#quantiles-and-a-robust-scale>`_,
      the guide's worked example.

    .. rubric:: Notes

    Each row of the window with a value counts once, at its own time, with the
    weight :func:`ewm_sum` gives it, ``w_j = lam ** |t_j - tau_t|``,
    ``lam = 2 ** (-1 / half_life)``. The quantile at ``q`` is the smallest
    ``x`` with ``W(<= x) >= q W``, where ``W(<= x)`` is the weight of the
    values at or below ``x`` and ``W`` the weight of all of them. That is
    numpy's ``inverted_cdf`` quantile over the window's values and weights.

    A value is read as the bucket it falls in, each bucket spanning a ratio of
    at most ``e ** (1/64)``. So the result is within ``tanh(1/128)``, 0.78%, of
    the exact quantile, relatively, at any scale and either sign. A magnitude
    below the smallest normal double counts as 0. A row whose value is missing
    counts for nothing and gives the window as it stands, as the sums do.

    **A robust scale.** The median of ``|r|`` is the median absolute
    deviation of ``r`` about 0. Divided by ``Φ⁻¹(3/4) = 0.6744897501960817``,
    it estimates a standard deviation for normal ``r``, and holds
    while up to half the window is wild, where a root mean square is moved by
    one row. For a residual, which the fit centers on 0::

        mad = rp.ops.ewm_quantile(pl.col("resid").abs(), quantile=0.5, half_life=500)
        scale = mad / 0.6744897501960817

    ``closed="left"`` leaves the row's own value out of its window. So a scale
    of a target, used as a weight on the same row, does not see the target it
    weighs.
    """
    return _window(
        "ewm_quantile",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
        quantile=quantile,
    )


def _extremum(
    name: str,
    input: str | pl.Expr,
    *,
    window_size: float | Duration | None,
    closed: Literal["right", "left", "both", "none"],
    min_samples: int,
    partial: Literal["keep", "null", "drop"] | None,
    reverse: bool,
    since: str | pl.Expr | None,
    max_state_bytes: int | None,
) -> pl.Expr:
    """The maximum or minimum ``name`` names, looking back, or ahead under
    ``reverse``, which the tree keeps as ``r<name>``."""
    who = f"rp.ops.{name}"
    if not isinstance(reverse, bool):
        raise TypeError(
            f"{who}: reverse must be True or False, but it is {reverse!r}. "
            "Pass reverse=True to read the rows after each row, or reverse=False."
        )
    if reverse and window_size is None:
        raise RefusedValueError(
            f"{who}: reverse=True needs window_size, but it is not set, and the rows after a "
            "row have no end without it. Set window_size to the span to read ahead."
        )
    return operator(
        f"r{name}" if reverse else name,
        input,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        since=since,
        max_state_bytes=max_state_bytes,
        who=f"rp.ops.{name}(reverse=True)" if reverse else None,
    )


def rolling_max(
    input: str | pl.Expr,
    *,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
    since: str | pl.Expr | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """The largest value of a column over each row's window.

    Use it for the largest value of a trailing or leading window on the
    stream's clock, or, with ``since``, for the largest since an event, such as
    a drawdown's high-water mark. With no ``window_size`` it covers every row since the
    last clock break or event, in constant memory.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("bid") - pl.col("ask")`` or
        :func:`increment`. It may hold no other window operator.
    ``window_size``
        The span of clock time a window covers: a number of clock units on a
        numeric clock, a duration (``"10s"``, a ``timedelta``,
        ``pl.duration(seconds=10)``) on a temporal one, and rows where the
        call has no clock column. The default, ``None``, covers every row
        since the last clock break (a gap past ``gap_cap`` or a session
        change) or the last ``since`` event. Required under ``reverse=True``.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change, the stream's start or a ``since`` event cuts before
        its full span. ``"keep"`` gives the largest value over what it saw,
        ``"null"`` gives null, and ``"drop"`` removes the row from the output.
        ``None`` (the default) means ``"keep"`` looking back and ``"null"``
        looking ahead. It needs ``window_size``.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it. As a spec's target, a window
        looking ahead is known only once it has closed, so the bank learns it
        under an ``embargo`` of at least ``window_size``. Default ``False``.
    ``since``
        An element-wise boolean of the row, a column name or a ``pl.Expr``:
        a row where it is true starts the window over at its stamp. Every row
        at that stamp is in the new window and every row before it is out, so
        the rows of one stamp never get two values, wherever in the stamp the
        true row falls. A null is false. It acts on this operator alone, and
        under ``group`` each group's events are its own. Default ``None``, no
        event. It cannot be combined with ``reverse=True`` yet: a window
        looking ahead would end at the next event, and whether such a window
        is whole or partial has no rule yet.
    ``max_state_bytes``
        The bytes a group's window may hold under a ``window_size``: each row
        inside the window costs a few dozen bytes (the module's Notes), and a
        window on a clock holds every row inside its span, so a burst fills
        it. Default 256 KB; at most 64 MiB. Without a clock column the window
        holds a known number of rows, and one past the cap is refused when the
        query is built; on a clock the row that takes a group's window past it
        is refused. With no ``window_size`` the state is a few doubles a
        group, whatever the stream's length.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    ``Float64`` column, or that a spec takes as its target. Its value is the
    largest of the values the window holds, and null where the window holds
    fewer than ``min_samples`` of them. It is null too where
    ``partial="null"`` meets a cut window, where a forward window is still
    open at the input's end, and after a reset (the module's ``partial``
    table). At a row whose value is missing, it gives the window as it
    stands. Under ``closed="left"`` or ``"none"`` the rows of an event's
    stamp are outside their own window, which the event has emptied, so they
    are null.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` or ``since`` is neither a column name nor a ``pl.Expr``;
        - ``window_size`` is neither a number nor a duration;
        - ``min_samples`` or ``max_state_bytes`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``window_size`` is NaN, infinite or not above 0, or text that is not
          a duration;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``max_state_bytes`` is not above 0 or is above 64 MiB;
        - ``input`` or ``since`` holds a window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``, or with ``since``. It is
        a ``ValueError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query: a ``since`` that is not a boolean, a window past
    ``max_state_bytes`` on the row clock, and a window looking ahead under
    ``group`` with no clock column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({
            "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "equity": [1.0, 3.0, 2.0, 4.0, 1.0, 2.0],
            "new_period": [False, False, False, False, True, False],
        })
        peak = rp.ops.rolling_max("equity", since="new_period")   # the high-water mark
        out = rp.stream.with_windows(
            rows,
            peak=peak,
            drawdown=pl.col("equity") - peak,
            last_3=rp.ops.rolling_max("equity", window_size=3.0),
            clock="t", gap_cap=10.0,
        )
        print(out["peak"].to_list())
        print(out["drawdown"].to_list())
        print(out["last_3"].to_list())

    .. code-block:: text

        [1.0, 3.0, 3.0, 4.0, 1.0, 2.0]
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
        [1.0, 3.0, 3.0, 4.0, 4.0, 4.0]

    .. rubric:: See also

    - :func:`rolling_min`, the smallest value.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `How a window operator weighs the rows near each row
      <https://github.com/rheon-dev/rheon/blob/main/docs/WINDOWS.md>`_,
      the guide.

    .. rubric:: Notes

    The value at row *t* is ``max(x_j for j in W_t)``, ``W_t`` the rows of
    the window with a value: Polars' ``rolling_max_by`` over a window on the
    clock. The window is the rows ``rolling_*_by`` holds under ``closed``,
    cut at the start of the stretch and, under ``since``, at the stamp of the
    last event. With no ``window_size`` it is every row since then, kept as
    one running maximum and its count, ``m_t = max(m_(t-1), x_t)``.

    Where it differs from Polars: a NaN, an infinity or a value beyond 1e100
    in magnitude is read as missing, as for every operator here, where
    Polars' ``rolling_max_by`` lets a NaN make the window NaN and counts an
    infinity. On equal values that differ only as ``0.0`` and ``-0.0`` the
    earliest row of the window wins, Polars' rolling rule, and looking ahead
    the latest. An integer input is read as a double, so one beyond ``2**53``
    in magnitude is rounded, where Polars returns an integer. Polars'
    ``cum_max`` gives null at a null row; this gives the window as it
    stands.

    **Drawdown** is a formula around it, not an operator of its own: with
    ``peak = rp.ops.rolling_max("equity", since="new_period")``, the drawdown is
    ``pl.col("equity") - peak``, 0 at a new high and negative below it, and
    ``pl.col("equity") / peak - 1`` the relative one. Its running worst is
    :func:`rolling_min` of that column, in a second call.
    """
    return _extremum(
        "rolling_max",
        input,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
        since=since,
        max_state_bytes=max_state_bytes,
    )


def rolling_min(
    input: str | pl.Expr,
    *,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
    since: str | pl.Expr | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """The smallest value of a column over each row's window.

    Use it for the smallest value of a trailing or leading window on the
    stream's clock, or, with ``since``, for the smallest since an event, such as
    a period's low. With no ``window_size`` it covers every row since the
    last clock break or event, in constant memory.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("bid") - pl.col("ask")`` or
        :func:`increment`. It may hold no other window operator.
    ``window_size``
        The span of clock time a window covers: a number of clock units on a
        numeric clock, a duration (``"10s"``, a ``timedelta``,
        ``pl.duration(seconds=10)``) on a temporal one, and rows where the
        call has no clock column. The default, ``None``, covers every row
        since the last clock break (a gap past ``gap_cap`` or a session
        change) or the last ``since`` event. Required under ``reverse=True``.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back and ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change, the stream's start or a ``since`` event cuts before
        its full span. ``"keep"`` gives the smallest value over what it saw,
        ``"null"`` gives null, and ``"drop"`` removes the row from the output.
        ``None`` (the default) means ``"keep"`` looking back and ``"null"``
        looking ahead. It needs ``window_size``.
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later,
        in place of the rows at or before it. As a spec's target, a window
        looking ahead is known only once it has closed, so the bank learns it
        under an ``embargo`` of at least ``window_size``. Default ``False``.
    ``since``
        An element-wise boolean of the row, a column name or a ``pl.Expr``:
        a row where it is true starts the window over at its stamp. Every row
        at that stamp is in the new window and every row before it is out, so
        the rows of one stamp never get two values, wherever in the stamp the
        true row falls. A null is false. It acts on this operator alone, and
        under ``group`` each group's events are its own. Default ``None``, no
        event. It cannot be combined with ``reverse=True`` yet: a window
        looking ahead would end at the next event, and whether such a window
        is whole or partial has no rule yet.
    ``max_state_bytes``
        The bytes a group's window may hold under a ``window_size``: each row
        inside the window costs a few dozen bytes (the module's Notes), and a
        window on a clock holds every row inside its span, so a burst fills
        it. Default 256 KB; at most 64 MiB. Without a clock column the window
        holds a known number of rows, and one past the cap is refused when the
        query is built; on a clock the row that takes a group's window past it
        is refused. With no ``window_size`` the state is a few doubles a
        group, whatever the stream's length.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    ``Float64`` column, or that a spec takes as its target. Its value is the
    smallest of the values the window holds, and null where the window holds
    fewer than ``min_samples`` of them. It is null too where
    ``partial="null"`` meets a cut window, where a forward window is still
    open at the input's end, and after a reset (the module's ``partial``
    table). At a row whose value is missing, it gives the window as it
    stands. Under ``closed="left"`` or ``"none"`` the rows of an event's
    stamp are outside their own window, which the event has emptied, so they
    are null.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` or ``since`` is neither a column name nor a ``pl.Expr``;
        - ``window_size`` is neither a number nor a duration;
        - ``min_samples`` or ``max_state_bytes`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``reverse`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``window_size`` is NaN, infinite or not above 0, or text that is not
          a duration;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``max_state_bytes`` is not above 0 or is above 64 MiB;
        - ``input`` or ``since`` holds a window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``, or with ``since``. It is
        a ``ValueError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query: a ``since`` that is not a boolean, a window past
    ``max_state_bytes`` on the row clock, and a window looking ahead under
    ``group`` with no clock column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0, 1, 2, 3, 4], "x": [3.0, 1.0, 2.0, 5.0, 4.0]})
        out = rp.stream.with_windows(
            rows,
            low=rp.ops.rolling_min("x", window_size=2),
            ahead=rp.ops.rolling_min("x", window_size=2, reverse=True),
            clock="t", gap_cap=10,
        )
        polars = rows.select(pl.col("x").rolling_min_by("t", window_size="2i"))
        print(out["low"].to_list(), polars["x"].to_list())   # equal
        print(out["ahead"].to_list())   # null where a window was open at the input's end

    .. code-block:: text

        [3.0, 1.0, 1.0, 2.0, 4.0] [3.0, 1.0, 1.0, 2.0, 4.0]
        [1.0, 2.0, 4.0, None, None]

    .. rubric:: See also

    - :func:`rolling_max`, the largest value.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `How a window operator weighs the rows near each row
      <https://github.com/rheon-dev/rheon/blob/main/docs/WINDOWS.md>`_,
      the guide.

    .. rubric:: Notes

    The value at row *t* is ``min(x_j for j in W_t)``, ``W_t`` the rows of
    the window with a value: Polars' ``rolling_min_by`` over a window on the
    clock. The window is the rows ``rolling_*_by`` holds under ``closed``,
    cut at the start of the stretch and, under ``since``, at the stamp of the
    last event. With no ``window_size`` it is every row since then, kept as
    one running minimum and its count, ``m_t = min(m_(t-1), x_t)``.

    Where it differs from Polars: a NaN, an infinity or a value beyond 1e100
    in magnitude is read as missing, as for every operator here, where
    Polars' ``rolling_min_by`` lets a NaN make the window NaN and counts an
    infinity. On equal values that differ only as ``0.0`` and ``-0.0`` the
    earliest row of the window wins, Polars' rolling rule, and looking ahead
    the latest. An integer input is read as a double, so one beyond ``2**53``
    in magnitude is rounded, where Polars returns an integer. Polars'
    ``cum_min`` gives null at a null row; this gives the window as it
    stands.

    **Drawdown** is a formula around it, not an operator of its own: with
    ``peak = rp.ops.rolling_max("equity", since="new_period")``, the drawdown is
    ``pl.col("equity") - peak``, 0 at a new high and negative below it, and
    ``pl.col("equity") / peak - 1`` the relative one. Its running worst is
    :func:`rolling_min` of that column, in a second call.
    """
    return _extremum(
        "rolling_min",
        input,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        reverse=reverse,
        since=since,
        max_state_bytes=max_state_bytes,
    )


def rolling_quantile(
    input: str | pl.Expr,
    *,
    quantile: float,
    window_size: float | Duration,
    interpolation: Literal[
        "nearest", "higher", "lower", "midpoint", "linear", "equiprobable"
    ] = "nearest",
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """Exact quantile of a column over each row's window.

    Use it when you need the exact rolling quantile matching Polars'
    ``rolling_quantile_by`` under any of its six ``interpolation`` rules, over
    a bounded window. For a decayed quantile with constant memory and no
    window cap, use :func:`ewm_quantile`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row. It may hold no other window operator.
    ``quantile``
        The quantile level in ``[0.0, 1.0]`` (0.5 is the median).
    ``window_size``
        The span of clock time a window covers (a positive finite number or
        duration, or rows without a clock column). Required so state stays
        bounded.
    ``interpolation``
        How a level between two sorted values is read, matching Polars:
        ``"nearest"`` (the default), ``"higher"``, ``"lower"``, ``"midpoint"``,
        ``"linear"`` or ``"equiprobable"``.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default), ``"left"``,
        ``"both"`` or ``"none"``.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value.
        Default 1.
    ``partial``
        What a window cut short gives: ``"keep"``, ``"null"``, ``"drop"`` or
        ``None`` (``"keep"`` looking back, ``"null"`` looking ahead).
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later.
        Default ``False``.
    ``max_state_bytes``
        Per-group memory cap in bytes for the window's queue and order tree.
        Default ``None`` (256 KB); at most 64 MiB.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    ``Float64`` column, or that a spec takes as its target.

    .. rubric:: Raises

    ``TypeError``
        When ``input``, ``window_size``, ``reverse``, ``quantile``,
        ``min_samples`` or ``max_state_bytes`` has the wrong type.
    ``ValueError``
        When ``quantile`` is outside ``[0.0, 1.0]``, ``window_size`` is not
        finite and positive, ``interpolation``, ``closed`` or ``partial`` is
        unknown, or ``max_state_bytes`` is outside ``[1, 64 MiB]``.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0, 1, 2, 3, 4], "x": [1.0, 4.0, 2.0, 5.0, 3.0]})
        q = rp.ops.rolling_quantile("x", quantile=0.5, window_size=3, interpolation="linear")
        out = rp.stream.with_windows(rows, med=q, clock="t", gap_cap=10)
        print(out["med"].to_list())

    .. code-block:: text

        [1.0, 2.5, 2.0, 4.0, 3.0]

    .. rubric:: See also

    - :func:`ewm_quantile`, decayed constant-memory quantile.
    - :func:`rolling_rank`, exact rank of each row inside its window.
    - :func:`rolling_max`, :func:`rolling_min`, rolling extrema.
    """
    who = "rp.ops.rolling_quantile"
    if not isinstance(reverse, bool):
        raise TypeError(
            f"{who}: reverse must be True or False, but it is {reverse!r}. "
            "Pass reverse=True to read the rows after each row, or reverse=False."
        )
    return operator(
        "rrolling_quantile" if reverse else "rolling_quantile",
        input,
        window_size=window_size,
        quantile=quantile,
        interpolation=interpolation,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        max_state_bytes=max_state_bytes,
        who=f"{who}(reverse=True)" if reverse else None,
    )


def rolling_rank(
    input: str | pl.Expr,
    *,
    window_size: float | Duration,
    method: Literal["average", "min", "max", "dense"] = "average",
    closed: Literal["right", "left", "both"] | None = None,
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    reverse: bool = False,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """Exact rank of each row's own value inside its window.

    Matches Polars' ``rolling_rank_by`` under its four deterministic tie
    methods (``"average"``, ``"min"``, ``"max"``, ``"dense"``), returned as
    ``Float64``.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row. It may hold no other window operator.
    ``window_size``
        The span of clock time a window covers (a positive finite number or
        duration, or rows without a clock column). Required so state stays
        bounded.
    ``method``
        How ties rank: ``"average"`` (the default), ``"min"``, ``"max"`` or
        ``"dense"``. ``"random"`` and ``"ordinal"`` are refused.
    ``closed``
        Which ends of the window are in. The row's own stamp must be in the
        window: ``"right"`` or ``"both"`` looking back, ``"left"`` or
        ``"both"`` looking ahead. ``None`` (the default) selects ``"right"``
        looking back and ``"left"`` looking ahead.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value.
        Default 1.
    ``partial``
        What a window cut short gives: ``"keep"``, ``"null"``, ``"drop"`` or
        ``None`` (``"keep"`` looking back, ``"null"`` looking ahead).
    ``reverse``
        ``True`` reads the rows after each row, at most ``window_size`` later.
        Default ``False``.
    ``max_state_bytes``
        Per-group memory cap in bytes for the window's queue and order tree.
        Default ``None`` (256 KB); at most 64 MiB.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    ``Float64`` column, or that a spec takes as its target. At a row whose own
    input value is missing, the rank is null. At a shared timestamp, each row
    ranks its own value against the stamp's full window.

    .. rubric:: Raises

    ``TypeError``
        When ``input``, ``window_size``, ``reverse``, ``min_samples`` or
        ``max_state_bytes`` has the wrong type.
    ``ValueError``
        When ``window_size`` is not finite and positive, ``method``,
        ``closed`` or ``partial`` is refused, or ``max_state_bytes`` is
        outside ``[1, 64 MiB]``.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0, 1, 2, 3, 4], "x": [1.0, 3.0, 2.0, 2.0, 5.0]})
        rk = rp.ops.rolling_rank("x", window_size=3, method="average")
        out = rp.stream.with_windows(rows, rk=rk, clock="t", gap_cap=10)
        print(out["rk"].to_list())

    .. code-block:: text

        [1.0, 2.0, 2.0, 1.5, 3.0]

    .. rubric:: See also

    - :func:`rolling_quantile`, exact rolling quantile.
    - :func:`cross_rank`, cross-sectional rank across groups at one stamp.
    """
    who = "rp.ops.rolling_rank"
    if not isinstance(reverse, bool):
        raise TypeError(
            f"{who}: reverse must be True or False, but it is {reverse!r}. "
            "Pass reverse=True to read the rows after each row, or reverse=False."
        )
    resolved_closed = ("left" if reverse else "right") if closed is None else closed
    return operator(
        "rrolling_rank" if reverse else "rolling_rank",
        input,
        window_size=window_size,
        method=method,
        closed=resolved_closed,
        min_samples=min_samples,
        partial=partial,
        max_state_bytes=max_state_bytes,
        who=f"{who}(reverse=True)" if reverse else None,
    )


#: The names the reversed operators had, which refuse the call naming the
#: ``reverse=True`` form that replaced them.
rewm_mean = renamed_function("rp.ops.rewm_mean", "rp.ops.ewm_mean(..., reverse=True)")
rewm_sum = renamed_function("rp.ops.rewm_sum", "rp.ops.ewm_sum(..., reverse=True)")
rewm_rate = renamed_function("rp.ops.rewm_rate", "rp.ops.ewm_rate(..., reverse=True)")


def ewm_var(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    bias: bool = False,
) -> pl.Expr:
    """The EW variance of a column about its time-weighted mean, over each row's
    window.

    Use it for the spread of a value as it stands on each row, looking back.
    For its square root, use :func:`ewm_std`. For a spread that one wild row
    does not move, use :func:`ewm_quantile` (its Notes).

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly,
        and for this operator needs ``window_size``:
        :func:`~rheon.polars.stream.with_windows`, or a spec that takes it in
        a formula target, refuses it without one.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"``. It needs ``window_size``, since without one no window
        is cut short.
    ``bias``
        ``False`` (the default) multiplies the variance by the correction for
        unequal weights, ``V1 ** 2 / (V1 ** 2 - V2)``, as Polars' ``ewm_var``
        does. ``True`` leaves it out.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands. While one row carries
    all the weight it is null under the default ``bias=False``, as Polars'
    first row is, and 0 under ``bias=True``.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``bias`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        v = rp.ops.ewm_var("x", half_life=1.0)
        out = rp.stream.with_windows(rows, v=v, clock="t", gap_cap=10)
        polars = rows.select(pl.col("x").ewm_var(half_life=1.0, adjust=False))   # 1 a row
        print(out["v"].round(4).to_list(), polars["x"].round(4).to_list())   # the same as Polars'

    .. code-block:: text

        [None, 0.5, 1.1, 1.6905, 2.1588] [None, 0.5, 1.1, 1.6905, 2.1588]

    .. rubric:: See also

    - :func:`ewm_std`, its square root.
    - :func:`ewm_mean`, the mean it is taken about.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `Scale a feature by its spread
      <https://github.com/rheon-dev/rheon/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_,
      the guide.

    .. rubric:: Notes

    Each value weighs ``m_i``, the decayed mass of its held interval inside the
    window, as in :func:`ewm_mean`. The variance is
    ``sum m_i (x_i - mu) ** 2 / V1``, with ``mu = sum m_i x_i / V1`` and
    ``V1 = sum m_i``. Unless ``bias``, it is multiplied by
    ``V1 ** 2 / (V1 ** 2 - V2)``, with ``V2 = sum m_i ** 2``: the correction
    for unequal weights that Polars' ``ewm_var`` applies to its own.

    On a clock that steps by 1 a row, with ``half_life`` in rows, and an input
    with no nulls, this is Polars' ``ewm_var(half_life=..., adjust=False,
    bias=...)``. ``adjust=False`` is the form :func:`ewm_mean` already mirrors, since
    ``ewm_mean_by`` holds a stretch's first value from before it. So the first
    row weighs ``(1 - alpha) ** (n - 1)``, with ``alpha = 1 - lam``, as
    ``adjust=False`` weighs it. A window that holds one value gives null under
    the default ``bias=False`` and 0 exactly, at any level, under ``bias=True``.
    Polars' ``ewm_var`` gives that row null too from py-polars 1.44, and 0
    before it, so on an older Polars the example's first value from Polars
    reads ``0.0``.

    With a null in the input the two differ, from the first row after it, under
    either setting of Polars' ``ignore_nulls``. This operator follows
    :func:`ewm_mean`, which follows ``ewm_mean_by``: a null row moves nothing,
    and the last value is held across the longer step to the next row with a
    value, so that row weighs the whole step. Polars' ``ewm_var`` has no such
    mode.
    """
    return operator(
        "ewm_var",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        bias=bias,
    )


def ewm_std(
    input: str | pl.Expr,
    *,
    half_life: float | Duration,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    min_samples: int = 1,
    partial: Literal["keep", "null", "drop"] | None = None,
    bias: bool = False,
) -> pl.Expr:
    """The EW standard deviation of a column, the square root of :func:`ewm_var`.

    Use it to put a feature in units of its current spread:
    ``(x - ewm_mean(x)) / ewm_std(x)`` is a z-score. For the variance itself,
    use :func:`ewm_var`.

    .. rubric:: Parameters

    ``input``
        The values the window reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("price") * pl.col("quantity")``
        or :func:`increment`. It may hold no other window operator.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at: a row ``half_life`` away weighs half as
        much. A number of clock units on a numeric clock, or a duration
        (``"10s"``, a ``timedelta``, ``pl.duration(seconds=10)``) on a temporal
        one. Required. ``float("inf")`` weighs every row of the window evenly,
        and for this operator needs ``window_size``:
        :func:`~rheon.polars.stream.with_windows`, or a spec that takes it in
        a formula target, refuses it without one.
    ``window_size``
        The span of clock time a window covers, in the same units as
        ``half_life``. The default, ``None``, covers every row since the last
        clock break (a gap past ``gap_cap`` or a session change), decayed.
    ``closed``
        Which ends of the window are in: ``"right"`` (the default, as in
        Polars' ``rolling_*_by``), ``"left"``, ``"both"`` or ``"none"``. The
        module's table of ``closed`` (:mod:`rheon.polars.ops`) gives each end,
        looking back.
    ``min_samples``
        The fewest rows with a value a window must hold to give a value, as in
        Polars' rolling functions. Default 1. An integer from 1 to
        ``2**32 - 1``.
    ``partial``
        What a window cut short gives: one that a gap past ``gap_cap``, a
        session change or the stream's start cuts before its full span.
        ``"keep"`` gives the value over what it saw, ``"null"`` gives null, and
        ``"drop"`` removes the row from the output. ``None`` (the default)
        means ``"keep"``. It needs ``window_size``, since without one no window
        is cut short.
    ``bias``
        ``False`` (the default) multiplies the variance by the correction for
        unequal weights, ``V1 ** 2 / (V1 ** 2 - V2)``, as Polars' ``ewm_var``
        does. ``True`` leaves it out.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, or that a spec takes as its target. Its value is null where the
    window holds fewer than ``min_samples`` rows with a value. It is null
    too where ``partial="null"`` meets a cut window, where a forward window
    is still open at the input's end, and after a reset (the module's
    ``partial`` table). At a row whose
    value is missing, it gives the window as it stands. It is null where
    :func:`ewm_var` is.

    .. rubric:: Raises

    ``TypeError``
        When:

        - ``input`` is neither a column name nor a ``pl.Expr``;
        - ``half_life`` or ``window_size`` is neither a number nor a duration
          (text such as ``"10s"``, a ``timedelta`` or ``pl.duration``);
        - ``min_samples`` is not an integer (a
          :class:`~rheon.WrongTypeError`, also a ``ValueError``);
        - ``bias`` is not ``True`` or ``False``.
    ``ValueError``
        When:

        - ``half_life`` is NaN or not above 0, or ``window_size`` is NaN,
          infinite or not above 0;
        - ``half_life`` or ``window_size`` is text that is not a duration,
          such as ``"abc"``;
        - ``closed`` or ``partial`` is a word it does not know, or ``partial``
          is given without ``window_size``;
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        z = (pl.col("x") - rp.ops.ewm_mean("x", half_life=1.0)) / rp.ops.ewm_std("x", half_life=1.0)
        sd = rp.ops.ewm_std("x", half_life=1.0)
        out = rp.stream.with_windows(rows, sd=sd, z=z, clock="t", gap_cap=10.0)
        polars = rows.select(pl.col("x").ewm_std(half_life=1.0, adjust=False))   # 1 a row
        print(out["sd"].round(4).to_list(), polars["x"].round(4).to_list())   # the same as Polars'
        print(out["z"].round(4).to_list())   # x in units of its spread

    .. code-block:: text

        [None, 0.7071, 1.0488, 1.3002, 1.4693] [None, 0.7071, 1.0488, 1.3002, 1.4693]
        [None, 0.7071, 0.7151, 0.673, 0.6381]

    .. rubric:: See also

    - :func:`ewm_var`, whose square root it is, and whose Notes define it.
    - :func:`ewm_mean`, the mean the spread is taken about.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.
    - `Scale a feature by its spread
      <https://github.com/rheon-dev/rheon/blob/main/docs/how-to/scale-a-feature-by-its-spread.md>`_,
      the guide's measured gain.

    .. rubric:: Notes

    It is the square root of :func:`ewm_var`, as Polars' ``ewm_std`` is the
    root of its ``ewm_var``. Before py-polars 1.44, Polars gives a window of
    one value 0 rather than null, so the example's first value from Polars
    reads ``0.0`` there. As :func:`ewm_var` says, it matches Polars' ``ewm_std``
    only on an input with no nulls.
    """
    return operator(
        "ewm_std",
        input,
        half_life=half_life,
        window_size=window_size,
        closed=closed,
        min_samples=min_samples,
        partial=partial,
        bias=bias,
    )


def ewm_moments(
    legs: dict[str, int],
    *,
    half_life: float | Duration,
    kind: Literal["central", "raw", "standardized", "cumulant"] = "central",
    bias: bool | None = None,
    window_size: float | Duration | None = None,
    lag_tolerance: float | Duration | None = None,
    partial: Literal["keep", "null", "drop"] | None = None,
) -> pl.Expr:
    """One term of the EW moments of one or more columns, each perhaps lagged,
    over each row's window.

    Use it for the running shape of a value as it stands on each row: a skew
    or a kurtosis (``{"x": 3}``, ``{"x": 4}`` under ``kind="standardized"``),
    a cumulant, a co-skewness of two columns (``{"x": 2, "y": 1}``), or an
    autocovariance or autocorrelation (``{"x": 1, "x@1": 1}``). Each call is
    one term. For the variance alone, :func:`ewm_var` is the same number by
    a shorter path. For the moments as a model's output, read before each row
    and saved with the bank's state, use :func:`rheon.spec.ewm_moments`.

    .. rubric:: Parameters

    ``legs``
        The term: a dict from each leg to its power, a whole number from 1,
        the term's order the sum of the powers. A leg is a column's name, or
        a column lagged ``k`` back, written ``"x@k"``: ``k`` in clock units on
        a numeric clock, a duration such as ``"x@5m"`` on a temporal one, and
        rows without a clock column. A key whose text after its last ``@`` is
        neither a number nor a duration is a column's name, read whole.
        Required.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the window is read at, as in :func:`ewm_var`: a number of clock
        units, or a duration on a temporal clock. Required. ``float("inf")``
        weighs every row of the window evenly and needs ``window_size``.
    ``kind``
        What the term is: ``"central"`` (the default), the moment about the
        weighted means; ``"raw"``, about zero; ``"standardized"``, the central
        moment over each leg's standard deviation to its power (a skew, a
        kurtosis, a correlation); ``"cumulant"``, of one leg.
    ``bias``
        ``True`` reads the plug-in moment of the weighted rows; ``False`` the
        form unbiased for rows drawn independently under these weights, of
        orders 2 to 4. The default, ``None``, is the statistic's default in
        Polars: a variance (a central moment or a cumulant of order 2)
        unbiased, as ``ewm_var``'s, and every other term the plug-in, as
        ``skew``'s and ``kurtosis``'s.
    ``window_size``
        The span of clock time a window covers, in the units of
        ``half_life``: the rows whose clock is in ``(tau_t - window_size,
        tau_t]``, the edge decided as Polars' ``rolling_*_by`` decides it.
        The default, ``None``, covers every row since the last clock break (a
        gap past ``gap_cap`` or a session change), decayed.
    ``lag_tolerance``
        How far before ``t - k`` the row a lag ``k`` pairs with may sit, in
        the units of ``half_life``. The default, ``None``, is the lag itself.
    ``partial``
        What a window cut short gives, as in :func:`ewm_var`: ``"keep"`` (the
        default under ``None``), ``"null"`` or ``"drop"``. It needs
        ``window_size``.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column. Its value at a row is read once every row at the row's stamp is
    in, as :func:`ewm_var`'s is (``closed="right"``). It is null where the
    term is not defined: no row yet, no lagged pair yet, a standardized term
    of a leg with no spread, an unbiased form whose denominator keeps no
    digit (one row carrying all the weight), or a value past a double's
    range. It is null too where ``partial="null"`` meets a cut window, and
    after a reset.

    .. rubric:: Raises

    ``TypeError``
        When ``legs`` is not a dict of text keys, a power is not a whole
        number (a :class:`~rheon.WrongTypeError`), ``half_life`` is not
        given, ``bias`` is not ``True``, ``False`` or ``None``, or a span is
        neither a number nor a duration.
    ``ValueError``
        When a power is below 1; a lag is below 0, or a duration lag is not
        above 0; two legs name the same column and lag; ``kind`` is not one
        of its words; a cumulant has more than one leg; a standardized term
        has order 1; ``bias=False`` meets an order above 4 outside ``raw``;
        ``half_life`` is inf without ``window_size``; ``partial`` is given
        without ``window_size``; or a span is NaN or not above 0.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a duration lag on a numeric clock.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 4.0, 3.0, 7.0]})
        out = rp.stream.with_windows(
            rows,
            m2=rp.ops.ewm_moments({"x": 2}, half_life=1.0, bias=True),
            v=rp.ops.ewm_var("x", half_life=1.0, bias=True),
            ac1=rp.ops.ewm_moments({"x": 1, "x@1": 1}, half_life=2.0, kind="standardized"),
            clock="t",
            gap_cap=10.0,
        )
        for row in out.select(pl.all().exclude("t", "x").round(4)).rows():
            print(row)   # m2 is ewm_var's; ac1 a correlation of the lagged pairs

    .. code-block:: text

        (0.0, 0.0, None)
        (0.25, 0.25, None)
        (1.6875, 1.6875, 1.0)
        (0.8594, 0.8594, 0.1767)
        (4.6836, 4.6836, 0.201)

    .. rubric:: See also

    - :func:`ewm_var`, the variance by the same weights.
    - :func:`rheon.spec.ewm_moments`, the moments as a model, many terms at
      once, read before each row.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.

    .. rubric:: Notes

    **The weights.** Each row's value is held over the interval since the
    last row with a value in every column the term reads, as in
    :func:`ewm_mean`, and weighs ``m_i``, the decayed mass of that interval:
    ``m_i = integral lam**(tau_t - s) ds`` over ``(t_{i-1}, t_i]``, with
    ``lam = 2 ** (-1 / half_life)``. A stretch's first value is held from
    before the stretch begins (``half_life / ln 2``), or over one window
    under ``window_size``, and no value's interval is longer than the
    window. Under a window a row is in it by its stamp and weighs its whole
    interval, where :func:`ewm_var` counts only the part of the oldest
    row's interval inside the window. A row at a repeated stamp holds no
    interval: it weighs 0, and a row of weight 0 is no row, in no sum, no
    lag and no window. A row missing a value in a column the term reads is
    skipped the same way.

    **The moments.** Over the rows of the window, ``z_i`` the legs' values
    at row ``i`` (a lagged leg's from its partner row), ``W = sum m_i`` and
    the weighted means ``m = sum m_i z_i / W``:

    .. code-block:: text

        central       mu_a  = sum m_i (z_i - m)^a / W         (a, the powers over the legs)
        raw           mu'_a = sum m_i z_i^a / W
        standardized  s_a   = mu_a / prod_j mu_{2 e_j}^(a_j / 2)
        cumulant      k_p   = mu_p - sum_{j=2..p-2} C(p-1, j-1) k_j mu_{p-j}    (one leg)

    These are computed in one pass by Pébay's update (Pébay, Terriberry,
    Kolla and Bennett 2016): a row of weight ``w`` after a decay ``lam``
    merges into ``W``, the means, the normalized central moments ``c_a`` and
    the ratios ``r_k = sum m_i**k / W**k`` as

    .. code-block:: text

        W'   = lam W + w        a = lam W / W'        b = w / W'        d = z - m
        c'_a = sum_{beta <= a} C(a, beta) a c_{a-beta} (-b d)^beta  +  b (a d)^a
        r'_k = a**k r_k + b**k
        m'   = m + b d

    Every product is at the scale of the spread, so a column at ``1e6``
    with a spread of 1 keeps its fourth moment. Decay multiplies ``W``
    alone, so a long gap moves no moment. A window's rows are kept, and its
    moments read by merging summaries of its parts by the same rule, so a
    row leaving the window is never subtracted.

    **Bias.** With ``r_k`` as above, the plug-in moments' expectations for
    rows drawn independently are ``E[c_2] = (1 - r_2) mu_2`` and ``E[c_3] =
    (1 - 3 r_2 + 2 r_3) mu_3``, and for order 4 a combination of ``mu_4``
    and ``mu_2**2``. Inverting them gives the unbiased weighted moments of
    Rimoldini (2014) for reliability weights: at order 2 this is
    ``V1**2 / (V1**2 - V2)`` times the plug-in, :func:`ewm_var`'s
    correction, and with every weight 1 they are scipy's ``kstat``,
    ``skew(bias=False)`` and ``kurtosis(bias=False, fisher=False)``. A term
    whose denominator keeps no digit is null: one row carrying all the
    weight, read from the weights and not from a count of rows.

    **Lags.** A lag ``k`` pairs the row at ``t`` with the latest row at or
    before ``t - k`` that the term learned, as Polars' ``join_asof`` with
    ``strategy="backward"`` pairs them, and with no row where that one is
    more than ``lag_tolerance`` before ``t - k``: after a gap longer than the
    lag a row closes no pair. A pair weighs its closing row's ``m_i``. A
    clock break starts the window and the lags over. The pairs are centered
    on their own weighted means, so ``{"x": 1, "x@1": 1}`` under
    ``kind="standardized"`` is the pairs' Pearson correlation, which differs
    from statsmodels' ``acf`` (the whole stream's mean over its variance)
    by a few thousandths on an autoregressive series.

    On a clock that steps by 1 a row and an input with no nulls, the order-2
    central term is :func:`ewm_var`, and so Polars' ``ewm_var(adjust=False)``,
    to rounding.
    """
    return moments_operator(
        legs,
        half_life=half_life,
        kind=kind,
        bias=bias,
        window_size=window_size,
        lag_tolerance=lag_tolerance,
        partial=partial,
    )


def increment(input: str | pl.Expr) -> pl.Expr:
    """The step of a column from its last value within the group and session.

    Use it to turn a running total, such as a cumulative volume, into the
    amounts each row added, before :func:`ewm_sum` or :func:`ewm_rate`. For a
    difference that is null on the row after a null too, use Polars'
    ``diff()``.

    .. rubric:: Parameters

    ``input``
        The running value: a column name, or an element-wise expression of the
        row. It may hold no window operator.

    .. rubric:: Returns

    A ``pl.Expr`` of ``x_i - x_{i-1}``, ``x_{i-1}`` the input's last value,
    that :func:`rheon.polars.stream.with_windows` computes as a column, or
    that a window operator takes as its input. It is a float, in seconds on a
    temporal column. It is null on the first row of each group (of the
    stream, without ``group``) and of each session, on a row whose input is
    null, and on a row whose clock steps back by more than
    ``restart_after_step_back``, which starts the stream over, in the row's
    group or on the stream across groups. A step back on the stream across
    groups starts every group over, so the next row of every group is null
    too.

    .. rubric:: Raises

    ``TypeError``
        ``input`` is neither a column name nor a ``pl.Expr``.
    ``ValueError``
        ``input`` holds a window operator, a part that is not element-wise, or
        a column whose name starts with ``@po:``, the prefix the operators' own
        columns carry.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "x": [1.0, None, 3.0, 4.0]})
        out = rp.stream.with_windows(rows, d=rp.ops.increment("x"), clock="t", gap_cap=10.0)
        print(out["d"].to_list())   # the null row is skipped: 3 steps from 1, the last value

    .. code-block:: text

        [None, None, 2.0, 1.0]

    .. rubric:: See also

    - :func:`ewm_sum` and :func:`ewm_rate`, which take its steps.
    - :func:`rheon.polars.stream.with_windows`, which computes the operators
      over a stream.

    .. rubric:: Notes

    A null input makes the row's increment null and is skipped, as the
    operators hold a value from the last valued row. So ``[1, null, 3]`` gives
    ``[null, null, 2]``, where ``diff()`` gives three nulls. On a time of day
    the step is in its seconds since midnight, so a step across midnight is
    negative, as Polars' ``diff`` on a ``Time`` gives.

    On an integer column, ``Int8`` to ``Int128`` or ``UInt8`` to ``UInt64``,
    the step is taken in integers, then made a float, rounded once. So a
    running count past ``2**53`` keeps its steps of 1.

    Give a sum or a rate the increments of a running total: the increments
    are the trades themselves. The difference of two decayed sums of the
    total loses precision instead. A window's notional of 1e6, read as the difference of two such
    sums near 1e10, loses about four of a double's sixteen digits.
    """
    return operator("increment", input)


def tick_sign(input: str | pl.Expr) -> pl.Expr:
    """The tick rule: the sign of a price's step from its last value, the
    last nonzero sign held over a step of 0.

    Use it to sign trades as buys (+1) or sells (-1) when the feed carries
    no side: a trade above the last price is taken as a buy, one below as a
    sell, and one at the same price as the trade before it. Multiplied by a
    size it gives the signed volume an imbalance bar
    (:func:`rheon.polars.stream.bars`) or a decayed order-flow sum
    (:func:`ewm_sum`) reads.

    .. rubric:: Parameters

    ``input``
        The price: a column name, or an element-wise expression of the row.
        It may hold no window operator, :func:`increment` or
        :func:`tick_sign`.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a Float64 column, or that a window operator takes as its input. With
    ``d_i`` the row's :func:`increment` and ``s`` the sign carried:

    .. code-block:: text

        d_i > 0:   s = +1,  the row's sign is +1
        d_i < 0:   s = -1,  the row's sign is -1
        d_i = 0:   s as it was, the row's sign is s (null while none is known)
        d_i null:  s as it was, the row's sign is null

    So the first row of each group and session has no sign, as it has no
    step; nor do the rows after it at the same price, until the price first
    moves. A row whose input is missing (null, NaN, infinite or beyond 1e100
    in magnitude) has no sign and is skipped, so the next row's step is from
    the last valued price, and a session change, or a step back of the
    clock past ``restart_after_step_back``, starts the rule over, as each
    does :func:`increment`'s.

    .. rubric:: Raises

    ``TypeError``
        ``input`` is neither a column name nor a ``pl.Expr``.
    ``ValueError``
        ``input`` holds a window operator, a part that is not element-wise, or
        a column whose name starts with ``@po:``, the prefix the operators' own
        columns carry.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                             "p": [10.0, 10.0, 10.5, 10.5, 10.25, 10.25]})
        out = rp.stream.with_windows(rows, s=rp.ops.tick_sign("p"), clock="t", gap_cap=10.0)
        print(out["s"].to_list())   # no sign until the price first moves

    .. code-block:: text

        [None, None, 1.0, 1.0, -1.0, -1.0]

    .. rubric:: See also

    - :func:`increment`, the step it reads the sign of.
    - :func:`rheon.polars.stream.bars`, whose imbalance bars sum a signed
      quantity.

    .. rubric:: Notes

    The rule is the tick test of Lee and Ready (1991), "Inferring trade
    direction from intraday data", *Journal of Finance* 46(2). Its own
    convention for a first trade, +1, is ``.fill_null(1.0)`` on the output,
    which then matches bar-forge's ``tick_rule_signs`` row for row; the
    output leaves it null instead, since no step is known. On a price grid
    many trades print at the last price, so carrying the last nonzero sign,
    rather than giving 0, keeps their volume in the signed flow.

    On a whole frame with no missing price, the same column in Polars, which
    the tests hold this to bit for bit, is the sign of ``diff`` with 0 read
    as null and the last sign filled forward (``.over`` the group, under
    ``group``)::

        d = pl.col("p").diff()
        pl.when(d != 0).then(d.sign()).forward_fill()

    Polars' form needs the whole column, where this one reads one row and
    carries two numbers a group: the last price and the last sign.
    """
    return operator("tick_sign", input)


def cross_mean(
    input: str | pl.Expr,
    *,
    weight: str | pl.Expr | None = None,
    closed: Literal["right", "left"] = "right",
    neighbors: str | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """The mean of a column across every group's row at each stamp of the clock.

    Use it for a cross-sectional level, such as the average return of every
    symbol at each timestamp, broadcast back to each symbol's row; a row's
    deviation from it, ``pl.col("r") - rp.ops.cross_mean("r")``, is its
    cross-sectional residual. :func:`cross_std` gives the spread about it.

    .. rubric:: Parameters

    ``input``
        The values reduced: a column name, or an element-wise expression of
        the row, such as ``pl.col("price") / pl.col("open") - 1`` or
        :func:`increment`. It may hold no window or cross operator.
    ``weight``
        Each row's weight: a column name or an element-wise expression of the
        row. ``None`` (the default) weighs every row 1. A row whose weight is
        null, NaN, infinite or not above 0 contributes nothing.
    ``closed``
        Which stamp a row reads. ``"right"`` (the default) is the row's own
        stamp: every group's row at that clock value, the row's own included,
        the value ``pl.col(input).mean().over(clock)`` gives on the whole
        frame. The row waits for its stamp to close, which the first row of a
        later stamp, or the end of the input, does. ``"left"`` is the stamp
        before the row's own, known as the row arrives.
    ``neighbors``
        The name of a list column giving each row's neighbors: the keys of
        the groups whose rows it reads, in place of every group's. ``None``
        (the default) reads every group. Each key counts once, a key with no
        row at the stamp read adds nothing, and the row's own group is read
        only where its list names it. It needs a ``group`` column, whose keys
        the lists name. Under ``"right"`` a neighbor's row later in the
        stamp counts. Each row of the open stamp is kept with its keys, 40
        bytes and 8 bytes plus its text a key, and under ``"left"`` each row
        of the stamp before with a value and a weight, 32 bytes, within
        ``max_state_bytes``.
    ``max_state_bytes``
        The most bytes the operator's state may hold, a whole number from 1
        to 64 MiB. Default ``None``, 262,144 (256 KB). Under ``"right"`` each
        row of a stamp is held until the stamp closes, ``8 * (outputs + 1)``
        bytes a row for the call's ``outputs`` operator columns, so the state
        grows with the rows of one stamp: the run warns once, as a
        :class:`~rheon.RheonWarning`, when a stamp takes it past half of the
        cap, and refuses the row that would take it past the cap, naming the
        row. Under ``"left"`` no row is held.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, under a ``clock``. Every row of a stamp reads the same value. It
    is null where the cross-section it reads holds fewer than two rows of
    weight with a value: one row is no cross-section. Under ``"left"`` it is
    null on the first stamp, and on the first stamp after a clock break (a
    gap past ``gap_cap``, a session change on a stream with no ``group``, or
    a step back past ``restart_after_step_back``).

    .. rubric:: Raises

    ``TypeError``
        When ``input`` or ``weight`` is neither a column name nor a
        ``pl.Expr``, or ``max_state_bytes`` is not an integer (a
        :class:`~rheon.WrongTypeError`).
    ``ValueError``
        When ``closed`` is neither ``"right"`` nor ``"left"``, ``input`` or
        ``weight`` holds a window or cross operator, or ``max_state_bytes``
        is below 1 or above 64 MiB.

    .. rubric:: Examples

    Each symbol's return less the mean return of every symbol at the same
    timestamp:

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        df = pl.DataFrame(
            {
                "t": [1, 1, 1, 2, 2, 2],
                "symbol": ["a", "b", "c", "a", "b", "c"],
                "r": [0.01, 0.03, -0.01, 0.02, 0.00, 0.01],
            }
        )
        out = rp.stream.with_windows(
            df,
            resid=pl.col("r") - rp.ops.cross_mean("r"),
            clock="t",
            gap_cap=10,
            group="symbol",
        )
        print(out["resid"].round(4).to_list())

    .. code-block:: text

        [0.0, 0.02, -0.02, 0.01, -0.01, 0.0]

    .. rubric:: Notes

    **The stamps.** The rows of a run are in clock order across groups, so
    a stamp's rows arrive together, and a step back of the clock is refused
    by row, as every window operator refuses it. A stamp's value does not
    depend on how the input was cut into chunks. A row of weight 0, or with
    no value, contributes nothing and counts in no stamp: a stamp of such
    rows alone is skipped by ``"left"``, whose next row reads the stamp
    before it.

    **The definition.** With ``w_i`` the weights of the stamp's rows of
    weight above 0 with a value ``x_i``, the mean is ``sum w_i x_i / sum
    w_i``, accumulated in Welford's form in input order. The oracle is
    Polars' ``pl.col(input).mean().over(clock)``, or ``group_by(clock)``
    and a join, on the whole frame at once.

    **A stream processor.** A cross-section needs every group's row at a
    stamp in one place, so a run partitioned by group, as a Kafka topic
    keyed by group is, computes it in one partition that reads every group.
    """
    return operator(
        "cross_mean",
        input,
        closed=closed,
        weight=weight,
        neighbors=neighbors,
        max_state_bytes=max_state_bytes,
    )


def cross_std(
    input: str | pl.Expr,
    *,
    weight: str | pl.Expr | None = None,
    closed: Literal["right", "left"] = "right",
    bias: bool = False,
    neighbors: str | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """The standard deviation of a column across every group's row at each stamp.

    Use it for a cross-sectional spread, such as the dispersion of returns
    across symbols at each timestamp, broadcast back to each symbol's row.
    :func:`cross_mean` gives the level it is about.

    .. rubric:: Parameters

    ``input``, ``weight``, ``closed``, ``neighbors``, ``max_state_bytes``
        As :func:`cross_mean` takes them.
    ``bias``
        ``False`` (the default) corrects the weighted second moment for the
        weights' count, as Polars' ``std(ddof=1)`` does at equal weights;
        ``True`` reports it as it is.

    .. rubric:: Returns

    A ``pl.Expr``, as :func:`cross_mean` returns one, null where it is null.

    .. rubric:: Raises

    ``TypeError``
        When ``input`` or ``weight`` is neither a column name nor a
        ``pl.Expr``, or ``bias`` is not ``True`` or ``False``.
    ``ValueError``
        As :func:`cross_mean` raises it.

    .. rubric:: Examples

    Each symbol's return in units of the spread of every symbol's return at
    the same timestamp:

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        df = pl.DataFrame(
            {
                "t": [1, 1, 1, 2, 2, 2],
                "symbol": ["a", "b", "c", "a", "b", "c"],
                "r": [0.01, 0.03, -0.01, 0.02, 0.00, 0.01],
            }
        )
        out = rp.stream.with_windows(
            df,
            z=(pl.col("r") - rp.ops.cross_mean("r")) / rp.ops.cross_std("r"),
            clock="t",
            gap_cap=10,
            group="symbol",
        )
        print(out["z"].round(3).to_list())

    .. code-block:: text

        [0.0, 1.0, -1.0, 1.0, -1.0, 0.0]

    .. rubric:: Notes

    With ``V1 = sum w_i``, ``V2 = sum w_i ** 2`` and the mean ``m`` of
    :func:`cross_mean`, the variance is ``sum w_i (x_i - m) ** 2 / V1``,
    times ``V1 ** 2 / (V1 ** 2 - V2)`` unless ``bias``; the standard
    deviation is its root. At equal weights the factor is ``n / (n - 1)``,
    and the value is Polars' ``pl.col(input).std().over(clock)``. Its stamps,
    weights and nulls are :func:`cross_mean`'s.
    """
    return operator(
        "cross_std",
        input,
        closed=closed,
        bias=bias,
        weight=weight,
        neighbors=neighbors,
        max_state_bytes=max_state_bytes,
    )


def cross_rank(
    input: str | pl.Expr,
    *,
    method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
    descending: bool = False,
    closed: Literal["right", "left"] = "right",
    neighbors: str | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """A row's rank among every group's row at its stamp.

    Use it for a cross-sectional rank, such as where each symbol's return
    stands among every symbol's return at the same timestamp: 1 is the
    smallest, and the cross-section's count of rows the largest.

    .. rubric:: Parameters

    ``input``, ``closed``, ``neighbors``
        As :func:`cross_mean` takes them. Under ``"right"`` the row is ranked
        among its own stamp's rows, itself included: Polars'
        ``pl.col(input).rank(method, descending=descending).over(clock)`` on
        the whole frame. Under ``"left"`` it is ranked among the stamp
        before's rows, as though it were added to them last.
    ``method``
        How tied values rank, as Polars' ``rank`` reads it: ``"average"``
        (the default) their mean rank, ``"min"`` or ``"max"`` the lowest or
        highest, ``"dense"`` the place of their value among the distinct
        values, ``"ordinal"`` each its own, the earlier row first.
    ``descending``
        ``True`` ranks the largest value 1.
    ``max_state_bytes``
        As :func:`cross_mean` takes it. A rank also keeps each row of the
        open stamp, 16 bytes a row, and under ``"left"`` the stamp before's
        distinct values, 16 bytes each.

    .. rubric:: Returns

    A ``pl.Expr``, as :func:`cross_mean` returns one. It is null where the
    row has no value, and where the cross-section it reads holds fewer than
    two rows with a value.

    .. rubric:: Raises

    ``TypeError``
        As :func:`cross_mean` raises it, or when ``descending`` is not
        ``True`` or ``False``.
    ``ValueError``
        As :func:`cross_mean` raises it, or when ``method`` is not one of
        the five words above.

    .. rubric:: Examples

    Each symbol's rank by return among every symbol at the same timestamp:

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        df = pl.DataFrame(
            {
                "t": [1, 1, 1, 2, 2, 2],
                "symbol": ["a", "b", "c", "a", "b", "c"],
                "r": [0.01, 0.03, 0.01, 0.02, 0.00, 0.01],
            }
        )
        out = rp.stream.with_windows(
            df, rank=rp.ops.cross_rank("r"), clock="t", gap_cap=10, group="symbol"
        )
        print(out["rank"].to_list())

    .. code-block:: text

        [1.5, 3.0, 1.5, 3.0, 1.0, 2.0]

    .. rubric:: Notes

    With ``l`` the rows ranked before the row's value, ``e`` the rows tied
    with it, itself among them, and ``p`` its place among those ties in
    input order, from 0: ``average`` is ``l + (e + 1) / 2``, ``min``
    ``l + 1``, ``max`` ``l + e``, ``dense`` one more than the distinct values
    ranked before it, and ``ordinal`` ``l + p + 1``. A rank takes no weight:
    it counts rows. A row with no value is no member of its stamp, and a
    stamp of such rows alone is skipped by ``"left"``, as :func:`cross_mean`
    skips it. The oracles are Polars' ``rank`` over the clock and SciPy's
    ``rankdata``.
    """
    return operator(
        "cross_rank",
        input,
        closed=closed,
        method=method,
        descending=descending,
        neighbors=neighbors,
        max_state_bytes=max_state_bytes,
    )


def cross_quantile(
    input: str | pl.Expr,
    quantile: float,
    *,
    interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear", "equiprobable"]
    | None = None,
    weight: str | pl.Expr | None = None,
    closed: Literal["right", "left"] = "right",
    neighbors: str | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """A quantile of a column across every group's row at each stamp.

    Use it for a cross-sectional level, such as the median return across
    symbols at each timestamp, broadcast back to each symbol's row: a level
    that one outlying symbol does not move as it moves :func:`cross_mean`.

    .. rubric:: Parameters

    ``input``, ``weight``, ``closed``, ``neighbors``
        As :func:`cross_mean` takes them. Under ``"right"`` the value is
        Polars' ``pl.col(input).quantile(quantile, interpolation).over(clock)``
        on the whole frame.
    ``quantile``
        The level, from 0 to 1: 0.5 is the median.
    ``interpolation``
        How a level between two values is read, as Polars reads it:
        ``"nearest"``, ``"higher"``, ``"lower"``, ``"midpoint"``,
        ``"linear"`` or ``"equiprobable"``. ``None`` (the default) is
        ``"nearest"``, Polars' own default, without ``weight``, and
        ``"equiprobable"``, the only reading with a weighted form, with it.
    ``max_state_bytes``
        As :func:`cross_mean` takes it. A quantile also keeps each row of the
        open stamp, 16 bytes a row, under either ``closed``.

    .. rubric:: Returns

    A ``pl.Expr``, as :func:`cross_mean` returns one, null where it is null.

    .. rubric:: Raises

    ``TypeError``
        As :func:`cross_mean` raises it, or when ``quantile`` is not a
        number (a :class:`~rheon.WrongTypeError`).
    ``ValueError``
        As :func:`cross_mean` raises it, or when ``quantile`` is below 0 or
        above 1, ``interpolation`` is not one of the six words above, or it
        is not ``"equiprobable"`` with ``weight``.

    .. rubric:: Examples

    Each symbol's return less the median return of every symbol at the same
    timestamp:

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        df = pl.DataFrame(
            {
                "t": [1, 1, 1, 2, 2, 2],
                "symbol": ["a", "b", "c", "a", "b", "c"],
                "r": [0.01, 0.03, -0.01, 0.02, 0.00, 0.50],
            }
        )
        out = rp.stream.with_windows(
            df,
            resid=pl.col("r") - rp.ops.cross_quantile("r", 0.5),
            clock="t",
            gap_cap=10,
            group="symbol",
        )
        print(out["resid"].round(4).to_list())

    .. code-block:: text

        [0.0, 0.02, -0.02, 0.0, -0.02, 0.48]

    .. rubric:: Notes

    With the cross-section's ``n`` values sorted, ``v[0] <= ... <=
    v[n - 1]``, and ``h = (n - 1) * quantile``: ``"nearest"`` reads
    ``v[round(h)]``, ``"lower"`` ``v[floor(h)]``, ``"higher"``
    ``v[ceil(h)]``, ``"midpoint"`` the mean of those two, ``"linear"`` the
    point ``h - floor(h)`` of the way between them, and ``"equiprobable"``
    ``v[max(ceil(n * quantile) - 1, 0)]``. With weights it is ``v[k]`` for
    the first ``k`` whose cumulative weight share ``(w_0 + ... + w_k) / sum
    w`` reaches the level, the inverse of the weighted distribution
    function: NumPy's ``quantile(method="inverted_cdf", weights=...)``.
    Its stamps, weights and nulls are :func:`cross_mean`'s.
    """
    return operator(
        "cross_quantile",
        input,
        closed=closed,
        quantile=quantile,
        interpolation=interpolation,
        weight=weight,
        neighbors=neighbors,
        max_state_bytes=max_state_bytes,
    )


def cross_ref(
    input: str | pl.Expr,
    where: pl.Expr,
    *,
    closed: Literal["right", "left"] = "right",
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """The value of one reference row at each stamp, read by every row of it.

    Use it to read a leader's value on every group's row, such as an index's
    return beside each symbol's at the same timestamp, without a join.

    .. rubric:: Parameters

    ``input``, ``closed``, ``max_state_bytes``
        As :func:`cross_mean` takes them. Under ``"right"`` a row reads the
        reference row of its own stamp, and waits for the stamp to close;
        under ``"left"`` the reference value of the last stamp before its own
        whose reference row had a value, known as the row arrives.
    ``where``
        The predicate that picks the reference row, an element-wise
        expression of the row, such as ``pl.col("symbol") == "SPY"``: a row
        where it is true (or a number other than 0) is the stamp's reference
        row.

    .. rubric:: Returns

    A ``pl.Expr``, as :func:`cross_mean` returns one. It is null where the
    stamp it reads has no reference row, or the reference row has no value.
    One row is all it reads, so it is not withheld for being one row.

    .. rubric:: Raises

    ``TypeError``
        As :func:`cross_mean` raises it, or when ``where`` is not given.
    ``ValueError``
        As :func:`cross_mean` raises it. The run raises an error naming the
        row where ``where`` picks a second row at one stamp: a stamp has one
        reference row.

    .. rubric:: Examples

    Each symbol's return less the index's return at the same timestamp:

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        df = pl.DataFrame(
            {
                "t": [1, 1, 1, 2, 2, 2],
                "symbol": ["a", "idx", "b", "idx", "a", "b"],
                "r": [0.01, 0.02, -0.01, 0.01, 0.03, 0.00],
            }
        )
        out = rp.stream.with_windows(
            df,
            excess=pl.col("r") - rp.ops.cross_ref("r", where=pl.col("symbol") == "idx"),
            clock="t",
            gap_cap=10,
            group="symbol",
        )
        print(out["excess"].round(4).to_list())

    .. code-block:: text

        [-0.01, 0.0, -0.03, 0.0, 0.02, -0.01]

    .. rubric:: Notes

    The oracle under ``"right"`` is a join on the clock with the reference
    rows, ``df.join(df.filter(where).select(clock, input), on=clock,
    how="left")``; under ``"left"``, Polars' ``join_asof`` on the reference
    rows with a value, backward and excluding an exact match. A clock break
    ends the stamp before, as for :func:`cross_mean`.
    """
    return operator("cross_ref", input, closed=closed, where=where, max_state_bytes=max_state_bytes)


def ewm_profile(
    input: str | pl.Expr,
    *,
    period: float | Duration,
    half_life: float | Duration,
    bins: int | None = None,
    harmonics: int | None = None,
    stat: Literal["mean", "var", "std", "quantile"] = "mean",
    quantile: float | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    origin: float | Duration | None = None,
    weight: str | pl.Expr | None = None,
    min_samples: int = 1,
    bias: bool | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """The EW mean, spread or quantile of a column at each row's phase of a period.

    Use it for a seasonal profile, such as the typical volatility of each
    hour of the trading day, or the delay of each hour of a transit day, read
    at each row's own hour. The profile is the input to an adjustment that
    the call composes: ``x - profile`` for an additive one, ``x / profile``
    or ``x / std_profile`` for a multiplicative one. With ``closed="left"``
    the profile a row reads holds no row of its own time, the row itself
    included, so the adjustment reads only what was known before the row.

    .. rubric:: Parameters

    ``input``
        The values the profile reads: a column name, or an element-wise
        expression of the row, such as ``pl.col("ret") ** 2`` or
        :func:`increment`. It may hold no other window operator.
    ``period``
        The span the phase repeats over: a duration such as ``"1d"`` or
        ``"1w"`` on a temporal clock, a number of clock units on a numeric
        one, or a number of rows without a clock. A calendar unit (``"1mo"``,
        ``"1q"``, ``"1y"``) is refused, since a month has no fixed length.
    ``bins``
        The binned form: how many equal phases the period is cut into, 24
        for each hour of a day, 7 for each day of a week, 168 for each hour
        of a week. A row reads its own phase's rows alone. Give ``bins``
        or ``harmonics``, one of them.
    ``harmonics``
        The harmonic form: how many Fourier harmonics fit the profile, from
        1 to 64, so ``2 · harmonics + 1`` terms. Every row reads one smooth
        curve fitted to every phase's rows, so a phase seen rarely borrows
        from the rest of the period. It takes ``stat`` ``"mean"``, ``"var"``
        or ``"std"`` and no ``bias``.
    ``half_life``
        How fast a row's weight fades with its distance in clock time from the
        row the profile is read at, whatever its phase: a row ``half_life``
        away weighs half as much, so a phase not seen for a long time weighs
        little. A number of clock units, a duration on a temporal clock, or
        ``float("inf")`` for no decay.
    ``stat``
        ``"mean"`` (the default), ``"var"``, ``"std"`` or ``"quantile"``:
        the weighted mean of each phase's rows, their weighted variance about
        it, its root, or their weighted quantile at ``quantile``, read from
        :func:`ewm_quantile`'s sketch of each phase's values, within 0.78%
        of the exact one.
    ``quantile``
        For ``stat="quantile"`` only: the level, above 0 and below 1, such as
        0.5 for the median.
    ``window_size``
        The most a row may be behind the row read and still count, in the
        same units as ``half_life``: rows further back leave their phase, as
        they leave :func:`ewm_mean`'s window. ``None`` (the default) keeps
        every row, decayed. Under a window each phase keeps its rows in the
        window, so the state grows with them.
    ``closed``
        ``"right"`` (the default), as :func:`ewm_mean` and Polars'
        ``rolling_*_by`` take it: a row reads the profile with every row of
        its own stamp in it, its own included, known once the stamp closes.
        ``"left"`` leaves the stamp's rows out, so the value is known as the
        stamp arrives, from earlier rows alone. Under a ``window_size``,
        ``"left"`` and ``"both"`` keep a row exactly ``window_size`` back,
        ``"right"`` and ``"none"`` do not; ``"both"`` holds the stamp's rows
        as ``"right"`` does and ``"none"`` leaves them out as ``"left"``
        does.
    ``origin``
        Where the period starts after its default start: a duration such as
        ``"9h30m"`` for a session that opens at 09:30, or ``"6d"`` for a
        week that starts on Sunday. The default start is midnight, the
        Monday of a week for a whole number of weeks (as Polars'
        ``dt.truncate`` starts one), and 0 on a numeric clock.
    ``weight``
        Each row's weight: a column name or an element-wise expression of the
        row. ``None`` (the default) weighs every row 1. A row whose weight is
        null, NaN, infinite or not above 0 adds nothing.
    ``min_samples``
        The rows with a value and a weight above 0 a phase needs before it is
        read, as :func:`ewm_mean` and Polars' ``rolling_*_by`` count them.
        Default 1: a phase any row has reached is read, and one none has is
        null.
    ``bias``
        For ``stat="var"`` or ``"std"`` only. ``False`` (the default)
        corrects the weighted second moment for the weights' number and
        inequality, as :func:`ewm_var` does; ``True`` reports it as it is.
    ``max_state_bytes``
        The most bytes one group's profile may hold, a whole number from 1
        to 64 MiB. Default ``None``, 262,144 (256 KB). A profile holds 56
        bytes a bin, so the default holds 4,681 bins, and a call whose
        bins pass its cap is refused before it reads a row, naming their
        bytes. Under a ``window_size`` each row in a phase's window adds 80
        bytes (104 for a quantile), and a quantile's sketches 8 bytes a
        sketch bucket. The harmonic form's fit holds ``8 · (p(p + 1)/2 + 2p
        + 3)`` bytes, ``p = 2 · harmonics + 1`` (224 at two harmonics), and
        under a ``window_size`` each row in the window adds that and 40 more
        (264 at two harmonics), its share of the Grams the window keeps. The
        run warns once, as a :class:`~rheon.RheonWarning`, when a row takes
        a group past half of the cap, and refuses the row that would take it
        past the cap, naming the row.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as a
    column, one profile per group. It is null where the row's phase holds
    fewer rows than ``min_samples``, inside the window where there is one,
    so a phase no row has reached is null; for a variance, also where one
    row's weight fills the phase. The harmonic form is null where its rows
    are fewer than ``min_samples``, where they do not pin the curve down
    (fewer distinct phases than its terms, or as good as), and, for a
    variance, where the fit of ``x²`` falls below the square of the fit of
    ``x``.

    .. rubric:: Raises

    ``TypeError``
        When ``input`` or ``weight`` is neither a column name nor a
        ``pl.Expr``, ``period``, ``half_life`` or ``origin`` is neither a
        number nor a duration, ``bins`` or ``max_state_bytes`` is not an
        integer, ``min_samples`` is not an integer (each a
        :class:`~rheon.WrongTypeError`), or ``bias`` is not ``True`` or
        ``False``.
    ``ValueError``
        When ``period``, ``half_life`` or ``window_size`` is not above 0,
        ``bins`` is below 1, ``stat`` or ``closed`` is not one of its
        words, ``quantile`` is not above 0 and below 1, ``min_samples`` is
        below 1, or ``max_state_bytes`` is below 1 or above 64 MiB. The run
        raises an error naming the row that would take a group's state past
        ``max_state_bytes``.

    .. rubric:: Examples

    The mean of each hour of the day, read at every row's own hour, with the
    hour before 09:00 seen on two days:

    .. code-block:: python

        from datetime import datetime

        import polars as pl
        from rheon import polars as rp

        df = pl.DataFrame(
            {
                "t": [
                    datetime(2026, 1, 5, 8, 30),
                    datetime(2026, 1, 5, 9, 30),
                    datetime(2026, 1, 6, 8, 15),
                    datetime(2026, 1, 6, 9, 45),
                ],
                "delay": [4.0, 1.0, 6.0, 2.0],
            }
        )
        out = rp.stream.with_windows(
            df,
            hourly=rp.ops.ewm_profile(
                "delay", period="1d", bins=24, half_life=float("inf")
            ),
            before=rp.ops.ewm_profile(
                "delay", period="1d", bins=24, half_life=float("inf"), closed="left"
            ),
            clock="t",
            gap_cap="7d",
        )
        print(out["hourly"].to_list(), out["before"].to_list())

    .. code-block:: text

        [4.0, 1.0, 5.0, 1.5] [None, None, 4.0, 1.0]

    .. rubric:: Notes

    **The phase** is the row's place in the period, read from its own clock
    alone: ``t − t.dt.truncate(period)``, Polars' own, less the origin. On a
    clock with a time zone it is read on the wall clock in the column's own
    zone, as Polars' ``dt`` namespace reads a zoned value: on the day the
    clocks go forward, 05:00 is phase 5h, though four hours have passed
    since midnight, so a rush hour stays in its own bin across the
    change. Without a clock the phase counts the group's rows of weight
    above 0. Row ``i`` falls in bin ``floor(bins · phase / period)``.

    **The statistic.** Each bin keeps the weighted moments of its rows,
    each row weighed by its weight times ``0.5 ** (age / half_life)``, its
    age the clock between it and the row read: ``V1 = Σ w``, ``V2 = Σ w²``,
    the mean ``m = Σ w x / V1`` and ``M2 = Σ w (x − m)²``, in Welford's form.
    The variance is ``M2 / V1`` times ``V1² / (V1² − V2)`` unless ``bias``.
    A bin decays by the clock whether or not a row falls in it.

    **The harmonic form** fits ``f(φ) = a₀ + Σₖ aₖ cos 2πkφ + bₖ sin
    2πkφ`` to the rows by weighted least squares, each row weighed as a
    bin's are, ``φ`` the phase as a share of the period: with the basis
    ``z(φ)``, the Gram ``G = Σ w d z zᵀ`` and ``r = Σ w d z x``, the mean
    profile is ``z(φ)ᵀ G⁻¹ r``, and the variance the fit of ``x²`` less the
    square of the fit of ``x``. The oracle is statsmodels' ``WLS`` on the
    same basis.

    **A window** keeps each phase's rows in two stacks, the oldest with the
    moments of the rows newer than it, so a row leaves by dropping its own,
    never by a subtraction that would leave a rounding behind; a quantile's
    sketches leave the same way, as :func:`ewm_quantile`'s do.

    **The state** is a group's bins, ``56 · bins`` bytes, and under a
    window the rows inside it, whatever the stream's length. The oracle is
    Polars' ``group_by`` over the phase on one chunk, with the decayed
    weights written out, and for a quantile NumPy's weighted
    ``inverted_cdf`` of each phase's values.
    """
    return profile(
        input,
        period=period,
        bins=bins,
        harmonics=harmonics,
        half_life=half_life,
        stat=stat,
        quantile=quantile,
        window_size=window_size,
        closed=closed,
        origin=origin,
        weight=weight,
        min_samples=min_samples,
        bias=bias,
        max_state_bytes=max_state_bytes,
    )


def run_length(input: str | pl.Expr, *, abs_tol: float = 0.0) -> pl.Expr:
    """The rows, ending at each row, whose value is unchanged: how long a
    value has held.

    Use it to find a frozen feed: a sensor holding its last reading, or a
    countdown stopped at one value, shows as a run that keeps growing.

    .. rubric:: Parameters

    ``input``
        The value: a column name, or an element-wise expression of the row.
        It may hold no operator.
    ``abs_tol``
        The largest difference two adjacent rows of one run may have. The
        default, 0, asks for values exactly equal.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column: on each row its place in its run, so the last row of a run
    gives the run's whole length. It is a float, and is never null.

    .. rubric:: Raises

    ``TypeError``
        ``input`` is neither a column name nor a ``pl.Expr``, or ``abs_tol`` is
        not a number.
    ``ValueError``
        ``abs_tol`` is negative, NaN or infinite, or ``input`` holds an operator,
        a part that is not element-wise, or a column whose name starts with
        ``@po:``, the prefix the operators' own columns carry.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"x": [1.0, 1.0, None, None, 1.0, 0.0, -0.0, 2.0]})
        out = rp.stream.with_windows(rows, n=rp.ops.run_length("x"))
        print(out["n"].to_list())   # 0.0 and -0.0 are one value; two nulls are one run

    .. code-block:: text

        [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0]

    .. rubric:: See also

    - :func:`increment`, the step from the last value, which is 0 inside a
      run under the default ``abs_tol``.
    - :func:`rheon.polars.stream.with_windows`, which computes the
      operators over a stream.

    .. rubric:: Notes

    Row *t*'s value is the largest ``n`` such that each of rows ``t - n + 1``
    to ``t`` is the same value as the row before it: that is Polars'
    ``rle``, the run lengths, read row by row. Two numbers are the same when
    ``abs(x_t - x_{t-1}) <= abs_tol``, compared with ``==`` under the default, so
    ``0.0`` and ``-0.0`` are one value, as ``rle`` has them. Each row is
    compared with the row before it, so under a positive ``abs_tol`` a run can
    creep, each step within ``abs_tol`` and its ends further apart.

    A null is a value of its own, as in ``rle``: rows of null are one run,
    and a null next to a number ends the run on both sides, so ``[3, null,
    3]`` gives ``[1, 1, 1]``. A NaN, an infinity or a value beyond 1e100 in
    magnitude is read as missing, as every operator reads it, so it is a null
    here: it joins a run of nulls, where ``rle`` makes a run of NaN apart.

    The run is a group's own, and starts again at a new session and where the
    clock steps back past ``restart_after_step_back``, as an
    :func:`increment` does. A gap past ``gap_cap`` leaves it as it is, since
    a value held across a gap is still held. An integer column is compared
    whole, so two neighbors past ``2**53`` stay two values, and a temporal
    one in nanoseconds, with ``abs_tol`` in seconds.
    """
    return row_operator("run_length", input, abs_tol=abs_tol)


def clock_lag(
    arrival: str | pl.Expr,
    event: str | pl.Expr,
    *,
    half_life: float | Duration,
) -> pl.Expr:
    """The decayed mean lag of each row's arrival behind its event: how late
    the rows arrive.

    Use it where each row carries two times, such as an exchange's stamp
    and the time a feed delivered it, and the lag between them says how
    fresh the stream is. :func:`clock_drift` gives how fast the lag grows.

    .. rubric:: Parameters

    ``arrival``
        The time each row arrived: a column name, or an element-wise
        expression of the row. It may hold no operator.
    ``event``
        The time each row's event happened, in the same form. Both are
        temporal, or both are numbers in one unit.
    ``half_life``
        How fast a row's weight halves on the stream's clock, in clock
        units: a number, or a duration such as ``"5m"`` on a temporal
        clock. ``float("inf")`` weighs every row the same.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column: ``sum w_i d_i / sum w_i`` over the rows of the stretch up to
    and including this one, with ``d_i = arrival_i - event_i`` and ``w_i =
    2 ** (-(tau_t - tau_i) / half_life)``. It is a float, in seconds where
    the times are temporal, and null until a row has both times.

    .. rubric:: Raises

    ``TypeError``
        ``arrival`` or ``event`` is neither a column name nor a ``pl.Expr``,
        or ``half_life`` is not given.
    ``ValueError``
        ``half_life`` is not above 0, or ``arrival`` or ``event`` holds an
        operator, a part that is not element-wise, or a column whose name
        starts with ``@po:``.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "e": [-0.5, 0.0, 0.5, 1.0]})
        out = rp.stream.with_windows(
            rows, lag=rp.ops.clock_lag("t", "e", half_life=float("inf")), clock="t", gap_cap=10.0
        )
        print(out["lag"].to_list())   # the lag grows by 0.5 a row: 0.5, 1, 1.5, 2

    .. code-block:: text

        [0.5, 0.75, 1.0, 1.25]

    .. rubric:: See also

    - :func:`clock_drift`, the lag's slope on the arrival time.
    - :func:`ewm_mean`, a time-weighted mean over a window, where this is a
      mean of each row counted once.

    .. rubric:: Notes

    The weights decay on the stream's clock, the ``clock`` of
    :func:`rheon.polars.stream.with_windows`, one unit a row without one. A
    row missing either time, null, NaN, infinite or beyond 1e100, learns
    nothing, and gives the mean as it stands. The sums are a group's own and
    start again at a clock break (a gap past ``gap_cap`` or a session change)
    and where the clock steps back past ``restart_after_step_back``.
    Two temporal times are subtracted in nanoseconds, exactly, before they
    become seconds.
    """
    return row_operator("clock_lag", arrival, event=event, half_life=half_life)


def clock_drift(
    arrival: str | pl.Expr,
    event: str | pl.Expr,
    *,
    half_life: float | Duration,
) -> pl.Expr:
    """The decayed slope of the lag of each row's arrival behind its event,
    on the arrival time: how fast the stream falls behind.

    Use it beside :func:`clock_lag` to tell a feed that is late by a steady
    amount from one whose lag grows, such as a consumer slower than its
    producer, or two clocks that drift apart.

    .. rubric:: Parameters

    ``arrival``
        The time each row arrived: a column name, or an element-wise
        expression of the row. It may hold no operator.
    ``event``
        The time each row's event happened, in the same form. Both are
        temporal, or both are numbers in one unit.
    ``half_life``
        How fast a row's weight halves on the stream's clock, in clock
        units: a number, or a duration such as ``"5m"`` on a temporal
        clock. ``float("inf")`` weighs every row the same.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column: the weighted least-squares slope of ``d_i = arrival_i -
    event_i`` on ``arrival_i`` over the rows of the stretch up to and
    including this one, each weighing ``w_i = 2 ** (-(tau_t - tau_i) /
    half_life)``. It is a float: seconds of lag a second of arrival where
    the times are temporal, so 0.001 is a lag that grows a millisecond a
    second. It is null while the rows with both times all arrived at one
    time, a single row included, since no slope is defined there.

    .. rubric:: Raises

    ``TypeError``
        ``arrival`` or ``event`` is neither a column name nor a ``pl.Expr``,
        or ``half_life`` is not given.
    ``ValueError``
        ``half_life`` is not above 0, or ``arrival`` or ``event`` holds an
        operator, a part that is not element-wise, or a column whose name
        starts with ``@po:``.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "e": [-0.5, 0.0, 0.5, 1.0]})
        out = rp.stream.with_windows(
            rows,
            drift=rp.ops.clock_drift("t", "e", half_life=float("inf")),
            clock="t",
            gap_cap=10.0,
        )
        print(out["drift"].to_list())   # the lag grows half a unit a unit of arrival

    .. code-block:: text

        [None, 0.5, 0.5, 0.5]

    .. rubric:: See also

    - :func:`clock_lag`, the lag's decayed mean.

    .. rubric:: Notes

    With ``W = sum w_i``, ``a_bar`` and ``d_bar`` the weighted means, the
    slope is ``sum w_i (a_i - a_bar)(d_i - d_bar) / sum w_i (a_i -
    a_bar) ** 2``. The two sums are kept about their running means, never
    as a mean square less a squared mean, which cancels where the times are
    far from 0 next to their spread, and the arrivals are measured from the
    stretch's first one, in nanoseconds where they are temporal. So the
    denominator is 0 exactly while every arrival is one time, and the slope
    is null there on the data's own condition, after no fixed count of rows.

    The weights decay on the stream's clock, one unit a row without one. A
    row missing either time learns nothing and gives the slope as it stands.
    The sums are a group's own and start again at a clock break (a gap past
    ``gap_cap`` or a session change) and where the clock steps back past
    ``restart_after_step_back``.
    """
    return row_operator("clock_drift", arrival, event=event, half_life=half_life)


def clock_step(
    time: str | pl.Expr,
    *,
    half_life: float | Duration,
    stat: Literal["mean", "var", "std", "zeros", "backs"] = "mean",
    bias: bool | None = None,
) -> pl.Expr:
    """A decayed statistic of the steps between each time and the one
    before it: how evenly a stream's rows arrive.

    Use it to watch a feed's cadence as a column: the mean step says how
    often rows come, the spread how regularly, and the counts how often a
    stamp repeats or goes back, such as an exchange's event times arriving
    out of order on a stream clocked by their arrival.

    .. rubric:: Parameters

    ``time``
        The time whose steps are read: a column name, or an element-wise
        expression of the row, temporal or a number. It may hold no
        operator. The stream's own clock column reads its cadence.
    ``half_life``
        How fast a step's weight halves on the stream's clock, in clock
        units: a number, or a duration such as ``"5m"`` on a temporal
        clock. ``float("inf")`` weighs every step the same.
    ``stat``
        What is read of the steps: ``"mean"`` (the default), ``"var"``,
        ``"std"``, ``"zeros"`` (the weight of the steps of exactly 0) or
        ``"backs"`` (the weight of the steps below 0).
    ``bias``
        For ``"var"`` and ``"std"`` only: ``False`` (the default) corrects
        the variance for the weights' count, as Polars' ``ewm_var`` does by
        default; ``True`` leaves it uncorrected.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column of floats, in seconds where the time is temporal. A row whose
    time is missing gives null. A mean or a spread is null until a step has
    weight, and a corrected spread while one step carries all the weight; a
    count is 0 before any step.

    .. rubric:: Raises

    ``TypeError``
        ``time`` is neither a column name nor a ``pl.Expr``, ``half_life``
        is not given, or ``bias`` is not a bool or is given for a ``stat``
        other than ``"var"`` and ``"std"``.
    ``ValueError``
        ``half_life`` is not above 0, ``stat`` is not one of its words, or
        ``time`` holds an operator, a part that is not element-wise, or a
        column whose name starts with ``@po:``.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 2.0, 5.0]})
        out = rp.stream.with_windows(
            rows,
            mean=rp.ops.clock_step("t", half_life=float("inf")),
            zeros=rp.ops.clock_step("t", half_life=float("inf"), stat="zeros"),
            clock="t",
            gap_cap=10.0,
        )
        print(out["mean"].round(3).to_list(), out["zeros"].to_list())   # steps 1, 1, 0, 3

    .. code-block:: text

        [None, 1.0, 1.0, 0.667, 1.25] [0.0, 0.0, 0.0, 1.0, 1.0]

    .. rubric:: See also

    - :func:`increment`, each step itself.
    - :func:`clock_lag` and :func:`clock_drift`, the lag between two times.

    .. rubric:: Notes

    Step ``s_i = t_i - t_{i-1}`` is taken between each time and the group's
    time before it, and weighs ``w_i = 2 ** (-(tau_t - tau_i) / half_life)``
    on the stream's clock ``tau``, one unit a row without one. With ``W =
    sum w_i`` the mean is ``sum w_i s_i / W``, the variance ``sum w_i (s_i -
    mean) ** 2 / W``, multiplied unless ``bias`` by ``W ** 2 / (W ** 2 - sum
    w_i ** 2)``, and a count ``sum w_i`` over the steps of 0, or below 0.
    Each is the decayed moments the other operators keep, updated a row at
    a time about the running mean, so a time far from 0 next to its steps
    loses no digit of their spread, and no count of rows decides when a
    value is given: one step gives a mean, and two of weight a corrected
    spread.

    A row whose time is missing (null, NaN, infinite or beyond 1e100) has no
    step and gives null, and the next time starts the steps again from
    itself: no step is taken across the missing one. The clock still decays
    the steps there. The steps are a group's own and start again at a clock
    break (a gap past ``gap_cap`` or a session change) and where the clock
    steps back past ``restart_after_step_back``. Two temporal times are
    subtracted in nanoseconds, exactly, before they become seconds.
    """
    return row_operator("clock_step", time, half_life=half_life, stat=stat, bias=bias)


def invert(
    f: pl.Expr | Callable[[pl.Expr], pl.Expr],
    equals: str | float | pl.Expr,
    lo: str | float | pl.Expr,
    hi: str | float | pl.Expr,
    *,
    increasing: bool,
    xtol: float = 2e-12,
    rtol: float = 4 * 2.0**-52,
    detail: bool = False,
) -> pl.Expr:
    """Solve an equation of the row for one unknown: the ``x`` in ``[lo, hi]``
    where ``f(x) = equals``, row by row.

    Use it to invert a closed-form function of the row's columns, such as a
    pricing formula for the volatility that gives a quoted price, where the
    function rises or falls with the unknown over the bracket. The solve
    never calls Python a row: the equation is read once, when the call is
    made, and the library evaluates it in Rust on each row. A root is given only
    where the function's own rounding fixes it to the tolerance, and is
    withheld as null otherwise, never clipped to the bracket.

    .. rubric:: Parameters

    ``f``
        The equation's left side: a ``pl.Expr`` in :func:`unknown`, such as
        ``x ** 3 + pl.col("a") * x`` with ``x = rp.ops.unknown()``, or a
        function of one argument that returns one, called once with
        :func:`unknown`. The expression may hold the unknown, columns,
        numbers, ``+ - * /``, ``**``, negation, ``abs``, ``exp``, ``log``,
        ``sqrt`` and a cast to ``pl.Float64``.
    ``equals``
        The value ``f`` must take: a column name, a number, or an expression
        of the row in the nodes ``f`` may hold, without the unknown.
    ``lo``, ``hi``
        The bracket's ends, in the forms ``equals`` takes. Each row's
        bracket is its own where an end reads a column.
    ``increasing``
        Required: ``True`` where ``f`` rises with the unknown over the
        bracket, ``False`` where it falls. The direction names the side of
        the function's range a target outside it lies on, and catches a
        function that is not as declared.
    ``xtol``
        The absolute tolerance, in the unknown's units: finite and above 0.
        Default ``2e-12``.
    ``rtol``
        The relative tolerance: finite and at least four machine epsilons,
        ``8.881784197001252e-16``, the default.
    ``detail``
        ``True`` for a struct with the root, the certified bracket, the
        evaluations it took, their bound and the reason a root is withheld, as ``withheld_reason``.
        Default ``False``: the root alone.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column, in a run that computes at least one window operator. It may
    stand alone, under arithmetic beside an operator, or as an operator's
    input, such as ``rp.ops.ewm_mean`` of the roots. Polars' ``with_columns``
    does not compute it, and a spec's formula target does not take it. The
    column is a ``Float64``, the root, null where it is withheld. Under
    ``detail`` it is a struct of ``root`` (``Float64``), ``lo`` and ``hi``
    (``Float64``, the final bracket, null where none was searched),
    ``evaluations`` and ``bound`` (``UInt32``) and ``withheld_reason`` (``String``,
    null where a root is given).

    .. rubric:: Raises

    ``TypeError``
        ``f`` is neither a ``pl.Expr`` nor a function returning one; ``equals``,
        ``lo`` or ``hi`` is neither a column name, a number nor a
        ``pl.Expr``; ``increasing`` or ``detail`` is not a bool; ``xtol``
        or ``rtol`` is not a number.
    ``ValueError``
        ``f`` does not hold the unknown; ``equals``, ``lo`` or ``hi``
        holds it; a part holds a window operator or a node the solve does not
        evaluate; the equation reads no column; ``lo`` and ``hi`` are
        numbers with ``lo`` not below ``hi``; a number is not finite;
        ``xtol`` or ``rtol`` is out of its range.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "a": [1.0, 2.0, 0.5, 1.0],
                             "y": [2.0, 3.0, None, 500.0]})
        x = rp.ops.unknown()
        cubic = x ** 3 + pl.col("a") * x   # rises with x where a >= 0
        out = rp.stream.with_windows(
            rows,
            x=rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True),
            d=rp.ops.invert(cubic, "y", -5.0, 5.0, increasing=True, detail=True),
            level=rp.ops.ewm_mean("y", half_life=1.0),   # a run computes a window operator
            clock="t", gap_cap=10.0,
        )
        print(out["x"].round(9).to_list())
        print(out["d"].struct.field("withheld_reason").to_list())   # y null, and y past f(5) = 130

    .. code-block:: text

        [1.0, 1.0, None, None]
        [None, None, 'input_missing', 'above_range']

    .. rubric:: See also

    - :func:`unknown`, the quantity solved for.
    - :func:`rheon.polars.stream.with_windows`, which computes it.
    - ``scipy.optimize.brentq``, the oracle it is held to.

    .. rubric:: Notes

    **The method.** With ``h(x) = f(x) - equals``, negated where ``f`` is
    declared decreasing, the solve keeps a bracket ``[a, b]`` with ``h(a) <
    0 < h(b)``, from ``[lo, hi]``. Each step evaluates ``h`` at one point
    and keeps the half whose ends' signs differ. The point is the ITP
    method's (Oliveira and Takahashi, ACM Transactions on Mathematical
    Software 47(1), 2021): the regula falsi point ``a - h(a) (b - a) / (h(b)
    - h(a))``, moved toward the midpoint ``m = (a + b) / 2`` by ``0.2 (b -
    a)² / (hi - lo)``, and kept within ``r_j = eps 2^(n_max - j) - (b - a) /
    2`` of ``m`` at step ``j``. That radius keeps the bracket after step ``j``
    no wider than ``eps 2^(n_max - j)``: never wider than bisection's after
    ``j - 1`` steps, while a smooth function converges as regula falsi does.

    **The tolerance and the bound.** ``eps = xtol + rtol min(|lo|, |hi|)``,
    or ``xtol`` alone where the bracket holds 0. The solve stops when ``b -
    a <= 2 eps + 8 u max(|a|, |b|)``, ``u`` the machine epsilon (or ``a``
    and ``b`` are adjacent doubles), or when an evaluation is exactly 0, and
    gives the midpoint: within ``eps + 5 u |x|`` of a sign change of the
    computed function. With ``n_max = ceil(log2((hi - lo) / (2 eps))) + 1``,
    it takes at most ``bound = n_max + 4`` evaluations: the two ends,
    ``n_max`` steps and two for the certificate below. The bound follows
    from the tolerance before the first step, so it is a property of the row
    and not a cap: ``[1e-4, 5]`` at ``xtol=2e-12`` is at most 46.

    **The certificate.** Each value carries a bound on its rounding error,
    by running error analysis: an operation adds half a unit in the last
    place of its result, and ``2^-1074`` so that a result that underflowed
    is not read as exact, to the bounds of its operands; ``exp``, ``log``
    and ``**`` add two units more. A sign counts only where the value
    exceeds its bound. Where the function is flat at the root, every point
    within ``e / |f'(x)|`` of it, ``e`` the rounding, is a root of the
    computed function, so a bracket narrowed there is not fixed by the
    function. A root is therefore given only where the final bracket's ends
    have certain signs, or the two points ``m - 2 eps`` and ``m + 2 eps``
    around its midpoint do, and that wider bracket is then the one reported.

    **What is withheld**, each reason named under ``detail``:

    .. list-table::
       :header-rows: 1
       :widths: 22 78

       * - ``withheld_reason``
         - when
       * - ``input_missing``
         - a column the equation reads, ``equals``, or a bracket end is
           null, NaN, infinite or past ``1e100`` in magnitude
       * - ``bracket_invalid``
         - ``lo >= hi`` on the row, or ``hi - lo`` overflows
       * - ``below_range``, ``above_range``
         - ``h`` has one sign at both ends, each beyond its bound: the
           target lies below or above the function's values on the bracket,
           so a monotone function has no root in it
       * - ``wrong_direction``
         - the ends' signs change against ``increasing``: the function is
           not what was declared on this row
       * - ``not_finite``
         - NaN at an end or a step, a domain error such as the log of a
           negative. An infinity has a sign, and is no such error
       * - ``not_monotone``
         - a step's value lies outside the values at the bracket's ends,
           beyond their bounds: the root may not be unique. It is checked
           at no extra evaluation, so a function that is not monotone can
           pass unseen
       * - ``not_determined``
         - the certificate fails, or an end's sign or the ends' order is
           within its rounding, so the target may lie inside the bracket

    A solve that took its whole bound without meeting the tolerance would be
    a defect of the solver, which the method rules out: the query raises an
    error naming the row's values rather than give a null.

    **Rows and state.** A row's root is a function of that row alone, so it
    is the same however the stream is chunked, and a run's saved state holds
    the equation and nothing of the rows. ``exp``, ``log`` and ``**`` call
    the platform's math library, as Polars does for the same expression, so
    across operating systems a root can differ in its last bits, or by up to
    the tolerance where a last bit moves a step to the other side.
    """
    return invert_node(
        f,
        equals,
        lo,
        hi,
        increasing=increasing,
        xtol=xtol,
        rtol=rtol,
        detail=detail,
    )


def unknown() -> pl.Expr:
    """The quantity :func:`invert` solves for, in the expression of its
    equation.

    Use it to write an equation's left side as a Polars expression, such as
    ``x ** 3 + pl.col("a") * x`` with ``x = rp.ops.unknown()``, and hand that
    expression to :func:`invert`. It composes with Polars' operators and
    the element-wise functions :func:`invert` evaluates, as a column does.

    .. rubric:: Returns

    A ``pl.Expr`` that stands for the unknown: a column under the prefix
    ``@po:``, which the window operators' own columns carry. Only the
    function :func:`invert` solves may hold it. Polars cannot compute it,
    and a formula of :func:`rheon.polars.stream.with_windows`, an invert's
    target or bracket, or a spec's target that holds it is refused when the
    query or the spec is built.

    .. rubric:: Examples

    .. code-block:: python

        x = rp.ops.unknown()
        rows = pl.DataFrame({"k": [1.0, 4.0, 9.0]})
        out = rp.stream.with_windows(
            rows,
            root=rp.ops.invert(x * x, "k", 0.0, 10.0, increasing=True),   # the square root
            level=rp.ops.ewm_mean("k", half_life=1.0),
        )
        print(out["root"].round(9).to_list())

    .. code-block:: text

        [1.0, 2.0, 3.0]

    .. rubric:: See also

    - :func:`invert`, which solves for it.
    """
    return pl.col(UNKNOWN_COLUMN)


def stl(
    input: str | pl.Expr,
    *,
    period: int,
    seasonal: int = 7,
    component: Literal["trend", "seasonal", "resid"] = "trend",
    window_rows: int | None = None,
    lag: int | None = None,
    closed: Literal["left", "right"] = "left",
    trend: int | None = None,
    low_pass: int | None = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    low_pass_jump: int = 1,
    robust: bool = False,
    inner_iter: int | None = None,
    outer_iter: int | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """A component of STL, the seasonal-trend decomposition by loess, read a few rows late.

    Use it to split a regular series into its trend, its season and the rest,
    such as an hourly load into its level, its daily shape and its surprise,
    as statsmodels' ``STL`` splits it. A decomposition revises its
    last rows as new ones arrive, so each value is read ``lag`` rows back,
    from a window of the group's last ``W`` learned rows, where the
    decomposition has stopped moving: the value of row ``t - lag`` arrives on row ``t``.

    On each row, STL of the window ``y`` (Cleveland et al., 1990): five passes
    of, with ``T = 0`` at first,

    .. code-block:: text

        C = loess of each cycle-subseries of y - T (span seasonal), extended a row each end
        L = loess (span low_pass) of the moving averages of p, p and 3 rows of C
        S = C - L,    T = loess (span trend) of y - S

    in the operations and the order of statsmodels' ``STL``, so each value is
    within the rounding of those operations of the exact decomposition, as
    statsmodels' is. The two agree to the bit where neither fuses a multiply
    and an add, and in the last bits elsewhere: statsmodels' build for macOS
    on arm64 fuses some. A row is learned when its input is present; a null
    learns nothing and counts in no lag. A clock break, a session change or a
    reset empties the group's ring.

    .. rubric:: Parameters

    ``input``
        The series: a column name, or an element-wise expression of the row.
        It may hold no other window operator.
    ``period``
        The season's length in learned rows, at least 2: 24 for a daily
        season of hourly rows.
    ``seasonal``
        The seasonal smoother's span, odd and at least 3: 7 by default.
    ``component``
        ``"trend"`` (the default), ``"seasonal"`` or ``"resid"``, the
        remainder ``x - S - T``.
    ``window_rows``
        ``W``, the learned rows each decomposition reads. ``None`` (the
        default) is ``2 h + 1``, with ``h = max p (s - 1) / 2`` the seasonal
        smoother's reach, so the row read sits at the window's center.
    ``lag``
        The learned rows back the value is read at. ``None`` (the default)
        is ``h``, plus 1 under ``closed="left"``.
    ``closed``
        ``"left"`` (the default): the window before the row, so the value
        reads no row at or after the row's own. ``"right"``: the window
        ending at the row, which reads the row's own value; a column
        decomposed this way and used to predict itself leaks its target.
    ``trend``, ``low_pass``
        The spans of the trend and of the low-pass filter, odd and above the
        period. ``None`` (the default) is statsmodels' default for each.
    ``seasonal_deg``, ``trend_deg``, ``low_pass_deg``
        The degree of each loess, 0 or 1: 1 by default.
    ``seasonal_jump``, ``trend_jump``, ``low_pass_jump``
        The rows between the points each loess fits, the rest interpolated:
        1 by default, every row.
    ``robust``, ``inner_iter``, ``outer_iter``
        statsmodels' robust fit, which downweighs outliers, and its passes:
        by default 5 inner and no outer, or 2 and 15 when robust. A robust
        fit costs about 7 times a plain one.
    ``max_state_bytes``
        The most bytes a group's ring may hold, ``8 W + 24``: 262144 (256
        KB) by default, at most 64 MiB. A window past it is refused when
        the run is built.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column: null until a group has ``W`` learned rows, then the component.

    .. rubric:: Raises

    ``TypeError``
        ``input`` is not a column name or an expression, or a setting is of
        the wrong type.
    ``ValueError``
        A setting statsmodels refuses (an even span, a trend span not above
        the period); a window under two periods; a lag outside the window; a
        ring past ``max_state_bytes``.

    .. rubric:: Examples

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        rows = pl.DataFrame({"t": range(40), "x": [float(i % 4 + i // 4) for i in range(40)]})
        out = rp.stream.with_windows(
            rows, rp.ops.stl("x", period=4).alias("trend"), clock="t", gap_cap=10.0
        )
        print(out["trend"].round(2).to_list()[-3:])

    .. code-block:: text

        [7.12, 7.38, 7.62]

    .. rubric:: See also

    - :func:`mstl`, the same with several seasons.
    - :func:`ewm_profile`, a seasonal profile read on the row itself, with no
      delay.
    """
    return seasonal_decomposition(
        input,
        periods=[period],
        windows=[seasonal],
        iterate=2,
        season=None,
        component=component,
        window_rows=window_rows,
        lag=lag,
        closed=closed,
        trend=trend,
        low_pass=low_pass,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
        low_pass_jump=low_pass_jump,
        robust=robust,
        inner_iter=inner_iter,
        outer_iter=outer_iter,
        max_state_bytes=max_state_bytes,
        mstl=False,
        who="rp.ops.stl",
    )


def mstl(
    input: str | pl.Expr,
    *,
    periods: list[int],
    windows: list[int] | None = None,
    iterate: int = 2,
    component: Literal["trend", "seasonal", "resid"] = "trend",
    season: int | None = None,
    window_rows: int | None = None,
    lag: int | None = None,
    closed: Literal["left", "right"] = "left",
    trend: int | None = None,
    low_pass: int | None = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    seasonal_jump: int = 1,
    trend_jump: int = 1,
    low_pass_jump: int = 1,
    robust: bool = False,
    inner_iter: int | None = None,
    outer_iter: int | None = None,
    max_state_bytes: int | None = None,
) -> pl.Expr:
    """A component of MSTL, the decomposition with several seasons, read a few rows late.

    Use it for a series with more than one season, such as an hourly load
    with a daily and a weekly shape, decomposed as statsmodels' ``MSTL``
    does (Bandara, Hyndman and Bergmeir, 2021): the seasons in ascending
    period, ``iterate`` times, each season added back, fitted by STL with its
    own period and span, and taken out again. The trend is the last STL's;
    the remainder is the series less every season and the trend. Each value
    is read ``lag`` learned rows back, as :func:`stl` reads it.

    .. rubric:: Parameters

    ``input``
        The series: a column name, or an element-wise expression of the row.
    ``periods``
        Each season's length in learned rows, such as ``[24, 168]`` for a
        daily and a weekly season of hourly rows. Each must be below half the
        window.
    ``windows``
        Each season's seasonal span, in the order of ``periods``. ``None``
        (the default) is statsmodels': 11, 15, ... from the shortest period.
    ``iterate``
        The passes over the seasons: 2 by default.
    ``season``
        For ``component="seasonal"``: the period whose seasonal to read.
    ``component``
        ``"trend"`` (the default), ``"seasonal"`` or ``"resid"``, the
        remainder ``x - S - T``.
    ``window_rows``
        ``W``, the learned rows each decomposition reads. ``None`` (the
        default) is ``2 h + 1``, with ``h = max p (s - 1) / 2`` the seasonal
        smoother's reach, so the row read sits at the window's center.
    ``lag``
        The learned rows back the value is read at. ``None`` (the default)
        is ``h``, plus 1 under ``closed="left"``.
    ``closed``
        ``"left"`` (the default): the window before the row, so the value
        reads no row at or after the row's own. ``"right"``: the window
        ending at the row, which reads the row's own value; a column
        decomposed this way and used to predict itself leaks its target.
    ``trend``, ``low_pass``
        The spans of the trend and of the low-pass filter, odd and above the
        period. ``None`` (the default) is statsmodels' default for each.
    ``seasonal_deg``, ``trend_deg``, ``low_pass_deg``
        The degree of each loess, 0 or 1: 1 by default.
    ``seasonal_jump``, ``trend_jump``, ``low_pass_jump``
        The rows between the points each loess fits, the rest interpolated:
        1 by default, every row.
    ``robust``, ``inner_iter``, ``outer_iter``
        statsmodels' robust fit, which downweighs outliers, and its passes:
        by default 5 inner and no outer, or 2 and 15 when robust. A robust
        fit costs about 7 times a plain one.
    ``max_state_bytes``
        The most bytes a group's ring may hold, ``8 W + 24``: 262144 (256
        KB) by default, at most 64 MiB. A window past it is refused when
        the run is built.

    .. rubric:: Returns

    A ``pl.Expr`` that :func:`rheon.polars.stream.with_windows` computes as
    a column: null until a group has ``W`` learned rows, then the component.

    .. rubric:: Raises

    ``TypeError``
        A setting is of the wrong type, or ``season`` is given without
        ``component="seasonal"``.
    ``ValueError``
        As :func:`stl`; a period not below half the window; ``season`` not
        among ``periods``.

    .. rubric:: Examples

    .. code-block:: python

        import polars as pl
        from rheon import polars as rp

        x = [float(i % 3 + 2 * (i % 5) + i // 15) for i in range(80)]
        rows = pl.DataFrame({"t": range(80), "x": x})
        three = rp.ops.mstl("x", periods=[3, 5], component="seasonal", season=3)
        out = rp.stream.with_windows(rows, three.alias("s3"), clock="t", gap_cap=10.0)
        print(out["s3"].is_not_null().sum())

    .. code-block:: text

        9

    .. rubric:: See also

    - :func:`stl`, one season.
    """
    return seasonal_decomposition(
        input,
        periods=list(periods),
        windows=None if windows is None else list(windows),
        iterate=iterate,
        season=season,
        component=component,
        window_rows=window_rows,
        lag=lag,
        closed=closed,
        trend=trend,
        low_pass=low_pass,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        low_pass_deg=low_pass_deg,
        seasonal_jump=seasonal_jump,
        trend_jump=trend_jump,
        low_pass_jump=low_pass_jump,
        robust=robust,
        inner_iter=inner_iter,
        outer_iter=outer_iter,
        max_state_bytes=max_state_bytes,
        mstl=True,
        who="rp.ops.mstl",
    )
