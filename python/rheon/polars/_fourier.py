"""Fourier seasonality terms of a clock column, as plain Polars expressions.

Two builders, each returning a list of ``pl.Expr``, one per term, that need
no state and no window: a row's terms are a function of its own clock
alone, so they stream in any query and are the same however the rows are
chunked.

``fourier(clock, period=P, order=K)``
    For a clock ``t`` in integer units (nanoseconds of a ``Datetime``'s or a
    ``Date``'s wall clock, or the integer itself), an origin ``t0`` and a
    period ``P`` in the same units::

        b   = (t - t0) mod P          a floor mod, in [0, P)
        r_i = (i b) mod P             exact in Int64, since P K < 2**63
        u_i = f64(r_i) / f64(P)
        sin(2 pi u_i), cos(2 pi u_i)  i = 1 .. K

    The angle stays inside ``[0, 2 pi)`` however long the stream, so its
    rounding does not grow with the clock. A phase held in a double, such
    as epoch seconds over the period, loses 1e-6 rad at a one-second period
    from the first row of today's clock.

``calendar_fourier(clock, freq=F, order=K)``
    The share of the calendar hour, day, week (Monday to Sunday), month,
    quarter or year that has elapsed at ``t``::

        s = t truncated to F,  e = s + F
        x = 2 pi (f64(t - s) / f64(e - s))     in the column's own unit
        sin(i x), cos(i x)                      i = 1 .. K

The definitions are statsmodels' ``Fourier`` and ``CalendarFourier``
(``statsmodels.tsa.deterministic``), the second to the bit.

A zoned ``Datetime`` is read on its wall clock in its own zone, so a daily
term peaks at the same local hour in summer and winter.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import polars as pl
import polars.selectors as cs

from rheon._duration import duration_text
from rheon._errors import WrongTypeError
from rheon._rheon import format_duration, parse_duration

#: ``2 * math.pi`` as a double, the factor statsmodels multiplies by.
TWO_PI = 6.283185307179586

#: Nanoseconds in a week, and from the epoch (a Thursday) to Monday 1970-01-05.
_WEEK_NS = 7 * 86_400 * 10**9
_MONDAY_NS = 4 * 86_400 * 10**9

#: The calendar units ``calendar_fourier`` takes, each as Polars writes it.
_CALENDAR = ("1h", "1d", "1w", "1mo", "1q", "1y")


def _refusal(clock: str, accepted: Any, message: str, dtype: pl.DataType) -> pl.Expr:
    """An arm that selects the clock only when its dtype is not ``accepted``,
    and then raises ``message`` when the query runs. An expression cannot see
    its column's dtype when it is built; a selector resolves against the
    schema, so for an accepted clock this arm selects nothing and leaves no
    Python function in the plan."""

    def refuse(_: pl.Series) -> pl.Series:
        raise ValueError(message)

    return (cs.by_name(clock) & ~accepted).as_expr().map_batches(refuse, return_dtype=dtype)


def _wall(clock: str, keep_unit: bool, who: str) -> pl.Expr:
    """A temporal clock column on its wall clock: a ``Date`` as is (or as
    nanoseconds), a naive ``Datetime`` as is, a zoned one with its zone
    dropped after reading the local time. Each arm selects the column only
    when its dtype matches, so exactly one arm is non-empty; a column of no
    temporal type selects the refusal's arm."""
    col = cs.by_name(clock)
    date_arm = (col & cs.date()).as_expr()
    datetime_arm = (col & cs.datetime()).as_expr().dt.replace_time_zone(None)
    if not keep_unit:
        date_arm = date_arm.cast(pl.Datetime("ns"))
        datetime_arm = datetime_arm.dt.cast_time_unit("ns")
    refused = _refusal(
        clock,
        cs.date() | cs.datetime(),
        f"{who}: clock {clock!r} is not a Date or Datetime column, but the period is a "
        "duration. Use a temporal clock, or an integer clock with an int period.",
        pl.Datetime("ns"),
    )
    return pl.max_horizontal(date_arm, datetime_arm, refused)


def _check_order(order: object, who: str) -> int:
    if isinstance(order, bool) or not isinstance(order, int):
        raise WrongTypeError(
            f"{who}: order must be an int, but it is {order!r}. Give the number of harmonics."
        )
    if order < 1:
        raise ValueError(
            f"{who}: order must be at least 1, but it is {order}. Give 1 or more harmonics."
        )
    return order


def _origin_ns(origin: object, who: str) -> int:
    """An origin of a temporal clock in nanoseconds of the naive wall clock."""
    if isinstance(origin, datetime):
        if origin.tzinfo is not None:
            raise ValueError(
                f"{who}: origin {origin!r} carries a time zone, but the terms read a zoned "
                "clock on its wall clock. Give origin as a naive datetime of that wall clock."
            )
        delta = origin - datetime(1970, 1, 1)
    elif isinstance(origin, date):
        delta = datetime(origin.year, origin.month, origin.day) - datetime(1970, 1, 1)
    else:
        raise WrongTypeError(
            f"{who}: the period is a duration, so origin must be a date or a datetime, "
            f"but it is {origin!r}. Give origin as a date or a naive datetime."
        )
    return (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1_000


def fourier(
    clock: str,
    *,
    period: int | timedelta | str | pl.Expr,
    order: int,
    origin: int | date | datetime | None = None,
) -> list[pl.Expr]:
    """Fourier terms of a fixed period: the sine and cosine of each row's phase.

    Use them as features of a regression on a seasonal series, such as the
    hour of the day of an hourly load, so the model weighs a smooth season
    in a few terms with its other features. A term reads the row's own clock
    alone, so it needs no state and no window, and it is the same however
    the stream is cut into chunks.

    The phase is reduced in whole numbers before it is turned into an angle,
    so its rounding does not grow with the clock: for a clock ``t`` in its
    integer units, an origin ``t0`` and a period ``P``,

    .. code-block:: text

        r_i = (i * (t - t0)) mod P,   u_i = r_i / P,   i = 1 .. order
        sin(2 pi u_i), cos(2 pi u_i)

    A phase held as epoch seconds in a float, the usual way, loses 1e-6
    radians at a one-second period. These terms are statsmodels'
    ``Fourier(n, 1).in_sample`` at the same phase, to the last bit.

    .. rubric:: Parameters

    ``clock``
        The clock column's name: a ``Datetime`` or ``Date`` column, read in
        nanoseconds, or an integer column, read in its own units. A zoned
        ``Datetime`` is read on its wall clock in its own zone, so a daily
        term peaks at the same local hour in summer and winter.
    ``period``
        On a temporal clock, a fixed duration: ``"1d"``, ``"1w"``, ``"90m"``
        or a ``timedelta``. On an integer clock, a whole number of its units.
        A calendar unit (``"1mo"``, ``"1q"``, ``"1y"``) has no fixed length;
        :func:`calendar_fourier` takes it.
    ``order``
        The harmonics, from 1: ``2 * order`` terms.
    ``origin``
        Where every term's phase is 0. ``None`` (the default) is 0 on an
        integer clock, and on a temporal one the epoch, or Monday 1970-01-05
        when the period is a whole number of weeks, as Polars'
        ``dt.truncate`` starts one. A naive ``date`` or ``datetime`` on a
        temporal clock, an ``int`` on an integer one.

    .. rubric:: Returns

    A list of ``pl.Expr``, one a term, named ``{clock}_sin{i}_{period}`` and
    ``{clock}_cos{i}_{period}`` in the order ``sin1, cos1, sin2, ...``, for
    ``with_columns`` or ``select``. A null clock gives null terms.

    .. rubric:: Raises

    ``TypeError``
        ``clock`` is not a column name, ``order`` is not an int, or
        ``period`` or ``origin`` is of the wrong type for the clock.
    ``ValueError``
        ``order`` is below 1; ``period`` is a calendar unit or not above 0;
        ``period * order`` reaches ``2**63``; ``origin`` carries a time
        zone. When the query runs: the clock is a float column, or of
        another type than the period's (an integer clock with a duration).

    .. rubric:: Examples

    .. code-block:: python

        from datetime import datetime

        import polars as pl
        from rheon import polars as rp

        rows = pl.DataFrame({"t": [datetime(2026, 1, 1, h) for h in (0, 6, 12, 18)]})
        out = rows.with_columns(rp.ops.fourier("t", period="1d", order=1))
        print(out["t_sin1_1d"].round(3).to_list())

    .. code-block:: text

        [0.0, 1.0, 0.0, -1.0]

    .. rubric:: See also

    - :func:`calendar_fourier`, the terms of a calendar hour, day, week,
      month, quarter or year.
    - :func:`ewm_profile`, a seasonal profile learned online.
    """
    who = "fourier"
    if not isinstance(clock, str):
        raise WrongTypeError(
            f"{who}: clock must be a column name, but it is {clock!r}. Pass the name as a str."
        )
    order = _check_order(order, who)
    if isinstance(period, (bool, float)):
        raise WrongTypeError(
            f"{who}: period must be an int on an integer clock or a duration on a temporal "
            f'one, but it is {period!r}. Give an int, or a duration such as "1d".'
        )
    if isinstance(period, int):
        p = period
        label = str(period)
        t0 = 0 if origin is None else origin
        if isinstance(t0, bool) or not isinstance(t0, int):
            raise WrongTypeError(
                f"{who}: the period is an int, so origin must be an int of the clock's units, "
                f"but it is {origin!r}. Give origin as an int."
            )
        refused = _refusal(
            clock,
            cs.integer(),
            f"{who}: clock {clock!r} is not an integer column, but the period is an int. A "
            "float clock has no exact reduction, so cast it to integer units first; give a "
            "temporal clock a duration period.",
            pl.Int64(),
        )
        t = pl.max_horizontal((cs.by_name(clock) & cs.integer()).as_expr(), refused).cast(pl.Int64)
    else:
        try:
            text = duration_text(period, who, "period")
            p = parse_duration(text)
        except ValueError as e:
            if isinstance(period, str) and period.endswith(("mo", "q", "y")):
                raise ValueError(
                    f"{who}: period {period!r} is a calendar unit, whose length varies. "
                    "Use calendar_fourier for it."
                ) from None
            if str(e).startswith(who):
                raise
            reason = str(e).rstrip(".")
            raise ValueError(
                f'{who}: period {reason}. Give a fixed duration such as "1d".'
            ) from None
        label = format_duration(p)
        t0 = (_MONDAY_NS if p % _WEEK_NS == 0 else 0) if origin is None else _origin_ns(origin, who)
        t = _wall(clock, keep_unit=False, who=who).dt.epoch("ns")
    if p <= 0:
        raise ValueError(
            f"{who}: period must be above 0, but it is {label}. Give a positive period."
        )
    if p * order >= 2**63:
        raise ValueError(
            f"{who}: period {label} times order {order} must stay below 2**63 so the "
            "reduction is exact in Int64. Lower the order."
        )
    # (t - t0) mod P, without forming t - t0, which may overflow Int64.
    base = ((t % p) - (t0 % p)) % p
    out: list[pl.Expr] = []
    for i in range(1, order + 1):
        r = ((base * i) % p).cast(pl.Float64)
        # Divided by P as a column: Polars divides a column by a scalar as a
        # product with its reciprocal, which is not the correctly rounded
        # quotient (5 of 24 hours in a day differ by an ulp, Polars 2.1.0),
        # while a column by a column is IEEE division. `r * 0.0` is 0 for
        # every finite r, and null where r is.
        x = r / (r * 0.0 + float(p)) * TWO_PI
        out.append(x.sin().alias(f"{clock}_sin{i}_{label}"))
        out.append(x.cos().alias(f"{clock}_cos{i}_{label}"))
    return out


def calendar_fourier(clock: str, *, freq: str, order: int) -> list[pl.Expr]:
    """Fourier terms of a calendar period: the share of its hour, day, week, month, quarter or year.

    Use them where a season follows the calendar, such as the day of the
    month of a payroll series or the day of the year of a temperature,
    whose periods have no fixed length. Each term reads the row's own clock
    alone, so it needs no state and no window.

    For the calendar period that holds ``t``, from its start ``s`` to the
    next start ``e``, in the column's own unit:

    .. code-block:: text

        x = 2 pi (t - s) / (e - s)
        sin(i x), cos(i x),   i = 1 .. order

    These are statsmodels' ``CalendarFourier(freq, order).in_sample``, to
    the last bit.

    .. rubric:: Parameters

    ``clock``
        The clock column's name: a ``Datetime`` or ``Date`` column. A zoned
        ``Datetime`` is read on its wall clock in its own zone.
    ``freq``
        ``"1h"``, ``"1d"``, ``"1w"`` (Monday to Sunday), ``"1mo"``, ``"1q"``
        or ``"1y"``. A fixed span of several units, such as ``"2d"``, is
        :func:`fourier`'s period.
    ``order``
        The harmonics, from 1: ``2 * order`` terms.

    .. rubric:: Returns

    A list of ``pl.Expr``, one a term, named ``{clock}_cal_sin{i}_{freq}``
    and ``{clock}_cal_cos{i}_{freq}`` in the order ``sin1, cos1, sin2,
    ...``. The names differ from :func:`fourier`'s, whose ``"1d"`` is a
    fixed span where this one's is the calendar day.

    .. rubric:: Raises

    ``TypeError``
        ``clock`` is not a column name, or ``order`` is not an int.
    ``ValueError``
        ``order`` is below 1, or ``freq`` is not one of the six units. When
        the query runs: the clock is not a ``Date`` or ``Datetime`` column.

    .. rubric:: Examples

    .. code-block:: python

        from datetime import date

        import polars as pl
        from rheon import polars as rp

        days = pl.DataFrame({"t": [date(2026, 1, 1), date(2026, 1, 16), date(2026, 2, 1)]})
        out = days.select(rp.ops.calendar_fourier("t", freq="1mo", order=1))
        print(out["t_cal_cos1_1mo"].round(3).to_list())

    .. code-block:: text

        [1.0, -0.995, 1.0]

    .. rubric:: See also

    - :func:`fourier`, the terms of a fixed period.
    """
    who = "calendar_fourier"
    if not isinstance(clock, str):
        raise WrongTypeError(
            f"{who}: clock must be a column name, but it is {clock!r}. Pass the name as a str."
        )
    order = _check_order(order, who)
    if freq not in _CALENDAR:
        raise ValueError(
            f"{who}: freq must be one of {', '.join(_CALENDAR)}, but it is {freq!r}. "
            "For a fixed span of several units, use fourier's period."
        )
    t = _wall(clock, keep_unit=True, who=who)
    start = t.dt.truncate(freq)
    gap = start.dt.offset_by(freq) - start
    ratio = (t - start).to_physical().cast(pl.Float64) / gap.to_physical().cast(pl.Float64)
    x = ratio * TWO_PI
    out: list[pl.Expr] = []
    for i in range(1, order + 1):
        xi = x * float(i)
        out.append(xi.sin().alias(f"{clock}_cal_sin{i}_{freq}"))
        out.append(xi.cos().alias(f"{clock}_cal_cos{i}_{freq}"))
    return out
