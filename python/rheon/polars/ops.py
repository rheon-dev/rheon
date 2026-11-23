"""The window operators: exponentially weighted (EW) means, sums, rates and
quantiles over a stream's clock, looking back or ahead, and a variance and a
standard deviation looking back. ``increment``, the step of a running total,
gives a window operator an input.

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
     - the *time-weighted* mean: row *i*'s value is held over ``(t_{i-1},
       t_i]``, a stretch's first value is held from before the stretch
       begins, and the mean
       over ``(tau_t - w, tau_t]`` weighs each held value by ``integral
       lam**(tau_t - s) ds`` over its interval inside the window
     - Polars' ``ewm_mean_by``. A burst of rows does not outweigh a quiet
       stretch, and a row at a repeated stamp holds no interval, so it moves
       no mean. No operator offers a mean that weighs each row by its count,
       the form pandas' ``ewm(times=)`` computes
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
   * - :func:`increment`
     - ``x_i - x_{i-1}`` within the group and session, null on a session's
       first row, and in seconds on a temporal column
     - the row before it

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
        mid_trend=rp.ewm_mean("mid", half_life="5s", window_size="1m") - pl.col("mid"),
        fwd_vwap_vs_mid=rp.ewm_sum(notional, half_life="10s", window_size="1m", reverse=True)
                        / rp.ewm_sum("quantity", half_life="10s", window_size="1m", reverse=True)
                        - pl.col("mid"),
        buy_vwap=rp.ewm_sum(buys.then(notional), half_life="10s", min_samples=1)
                 / rp.ewm_sum(buys.then("quantity"), half_life="10s", min_samples=1),
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
    z_mid = (pl.col("mid") - rp.ewm_mean("mid", **spread)) / rp.ewm_std("mid", **spread)
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
row, :func:`increment` included, and never another window operator. For a
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

from typing import TYPE_CHECKING, Literal

import polars as pl

from rheon._errors import RefusedValueError
from rheon._renamed import renamed_function
from rheon.polars._formula import moments_operator, operator, profile

if TYPE_CHECKING:
    from rheon._duration import Duration

__all__ = [
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
    "increment",
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
) -> pl.Expr:
    """The mean, sum, rate or quantile ``name`` names, looking back, or ahead
    under ``reverse``: the operator over the rows after each row, which the
    tree keeps as ``r<name>``."""
    who = f"rp.{name}"
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
        who=f"rp.{name}(reverse=True)" if reverse else None,
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
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = rp.stream.with_windows(
            rows,
            back=rp.ewm_mean("x", half_life=1.0),        # the rows at or before each row
            ahead=rp.ewm_mean("x", half_life=1.0, window_size=2.0, reverse=True),   # next 2 units
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
        - ``min_samples`` is below 1 or above ``2**32 - 1``;
        - ``input`` holds another window operator, a part that is not
          element-wise, or a column whose name starts with ``@po:``, the prefix
          the operators' own columns carry.
    :class:`~rheon.RefusedValueError`
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = rp.stream.with_windows(rows, s=rp.ewm_sum("x", half_life=1.0), clock="t", gap_cap=10)
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
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # half_life=inf weighs the window evenly: the sum over the last 2 units, per unit.
        rate = rp.ewm_rate("x", half_life=float("inf"), window_size=2.0)
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
        ``reverse=True`` without ``window_size``. It is both a ``ValueError``
        and a ``TypeError``.

    :func:`rheon.polars.stream.with_windows` refuses the rest when it builds
    its query, such as a window looking ahead under ``group`` with no clock
    column.

    .. rubric:: Examples

    .. code-block:: python

        rows = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        # The median of the last 3 units, weighed evenly.
        med = rp.ewm_quantile("x", quantile=0.5, half_life=float("inf"), window_size=3.0)
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

        mad = rp.ewm_quantile(pl.col("resid").abs(), quantile=0.5, half_life=500)
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


#: The names the reversed operators had, which refuse the call naming the
#: ``reverse=True`` form that replaced them.
rewm_mean = renamed_function("rp.rewm_mean", "rp.ewm_mean(..., reverse=True)")
rewm_sum = renamed_function("rp.rewm_sum", "rp.ewm_sum(..., reverse=True)")
rewm_rate = renamed_function("rp.rewm_rate", "rp.ewm_rate(..., reverse=True)")


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
        out = rp.stream.with_windows(rows, v=rp.ewm_var("x", half_life=1.0), clock="t", gap_cap=10)
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
        z = (pl.col("x") - rp.ewm_mean("x", half_life=1.0)) / rp.ewm_std("x", half_life=1.0)
        sd = rp.ewm_std("x", half_life=1.0)
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
            m2=rp.ewm_moments({"x": 2}, half_life=1.0, bias=True),
            v=rp.ewm_var("x", half_life=1.0, bias=True),
            ac1=rp.ewm_moments({"x": 1, "x@1": 1}, half_life=2.0, kind="standardized"),
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
        out = rp.stream.with_windows(rows, d=rp.increment("x"), clock="t", gap_cap=10.0)
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
    deviation from it, ``pl.col("r") - rp.cross_mean("r")``, is its
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
            resid=pl.col("r") - rp.cross_mean("r"),
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
            z=(pl.col("r") - rp.cross_mean("r")) / rp.cross_std("r"),
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
            df, rank=rp.cross_rank("r"), clock="t", gap_cap=10, group="symbol"
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
            resid=pl.col("r") - rp.cross_quantile("r", 0.5),
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
            excess=pl.col("r") - rp.cross_ref("r", where=pl.col("symbol") == "idx"),
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
    buckets: int | None = None,
    harmonics: int | None = None,
    stat: Literal["mean", "var", "std", "quantile"] = "mean",
    quantile: float | None = None,
    window_size: float | Duration | None = None,
    closed: Literal["right", "left", "both", "none"] = "right",
    offset: float | Duration | None = None,
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
    ``buckets``
        The bucketed form: how many equal phases the period is cut into, 24
        for each hour of a day, 7 for each day of a week, 168 for each hour
        of a week. A row reads its own phase's rows alone. Give ``buckets``
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
    ``offset``
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
        bytes a bucket, so the default holds 4,681 buckets, and a call whose
        buckets pass its cap is refused before it reads a row, naming their
        bytes. Under a ``window_size`` each row in a phase's window adds 80
        bytes (104 for a quantile), and a quantile's sketches 8 bytes a
        sketch bucket. The harmonic form's fit holds ``8 · (p(p + 1)/2 + 2p
        + 2)`` bytes, ``p = 2 · harmonics + 1`` (216 at two harmonics), and
        under a ``window_size`` each row in the window adds that and 40 more
        (256 at two harmonics), its share of the Grams the window keeps. The
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
        ``pl.Expr``, ``period``, ``half_life`` or ``offset`` is neither a
        number nor a duration, ``buckets`` or ``max_state_bytes`` is not an
        integer, ``min_samples`` is not an integer (each a
        :class:`~rheon.WrongTypeError`), or ``bias`` is not ``True`` or
        ``False``.
    ``ValueError``
        When ``period``, ``half_life`` or ``window_size`` is not above 0,
        ``buckets`` is below 1, ``stat`` or ``closed`` is not one of its
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
            hourly=rp.ewm_profile(
                "delay", period="1d", buckets=24, half_life=float("inf")
            ),
            before=rp.ewm_profile(
                "delay", period="1d", buckets=24, half_life=float("inf"), closed="left"
            ),
            clock="t",
            gap_cap="7d",
        )
        print(out["hourly"].to_list(), out["before"].to_list())

    .. code-block:: text

        [4.0, 1.0, 5.0, 1.5] [None, None, 4.0, 1.0]

    .. rubric:: Notes

    **The phase** is the row's place in the period, read from its own clock
    alone: ``t − t.dt.truncate(period)``, Polars' own, less the offset. On a
    clock with a time zone it is read on the wall clock in the column's own
    zone, as Polars' ``dt`` namespace reads a zoned value: on the day the
    clocks go forward, 05:00 is phase 5h, though four hours have passed
    since midnight, so a rush hour stays in its own bucket across the
    change. Without a clock the phase counts the group's rows of weight
    above 0. Row ``i`` falls in bucket ``floor(buckets · phase / period)``.

    **The statistic.** Each bucket keeps the weighted moments of its rows,
    each row weighed by its weight times ``0.5 ** (age / half_life)``, its
    age the clock between it and the row read: ``V1 = Σ w``, ``V2 = Σ w²``,
    the mean ``m = Σ w x / V1`` and ``M2 = Σ w (x − m)²``, in Welford's form.
    The variance is ``M2 / V1`` times ``V1² / (V1² − V2)`` unless ``bias``.
    A bucket decays by the clock whether or not a row falls in it.

    **The harmonic form** fits ``f(φ) = a₀ + Σₖ aₖ cos 2πkφ + bₖ sin
    2πkφ`` to the rows by weighted least squares, each row weighed as a
    bucket's are, ``φ`` the phase as a share of the period: with the basis
    ``z(φ)``, the Gram ``G = Σ w d z zᵀ`` and ``r = Σ w d z x``, the mean
    profile is ``z(φ)ᵀ G⁻¹ r``, and the variance the fit of ``x²`` less the
    square of the fit of ``x``. The oracle is statsmodels' ``WLS`` on the
    same basis.

    **A window** keeps each phase's rows in two stacks, the oldest with the
    moments of the rows newer than it, so a row leaves by dropping its own,
    never by a subtraction that would leave a rounding behind; a quantile's
    sketches leave the same way, as :func:`ewm_quantile`'s do.

    **The state** is a group's buckets, ``56 · buckets`` bytes, and under a
    window the rows inside it, whatever the stream's length. The oracle is
    Polars' ``group_by`` over the phase on one chunk, with the decayed
    weights written out, and for a quantile NumPy's weighted
    ``inverted_cdf`` of each phase's values.
    """
    return profile(
        input,
        period=period,
        buckets=buckets,
        harmonics=harmonics,
        half_life=half_life,
        stat=stat,
        quantile=quantile,
        window_size=window_size,
        closed=closed,
        offset=offset,
        weight=weight,
        min_samples=min_samples,
        bias=bias,
        max_state_bytes=max_state_bytes,
    )
