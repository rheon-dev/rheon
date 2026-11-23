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


def _wall(clock: str, keep_unit: bool) -> pl.Expr:
    """A temporal clock column on its wall clock: a ``Date`` as is (or as
    nanoseconds), a naive ``Datetime`` as is, a zoned one with its zone
    dropped after reading the local time. Each arm selects the column only
    when its dtype matches, so exactly one arm is non-empty, and a column of
    no temporal type is refused by Polars when the query runs."""
    col = cs.by_name(clock)
    date_arm = (col & cs.date()).as_expr()
    datetime_arm = (col & cs.datetime()).as_expr().dt.replace_time_zone(None)
    if not keep_unit:
        date_arm = date_arm.cast(pl.Datetime("ns"))
        datetime_arm = datetime_arm.dt.cast_time_unit("ns")
    return pl.max_horizontal(date_arm, datetime_arm)


def _check_order(order: object, who: str) -> int:
    if isinstance(order, bool) or not isinstance(order, int):
        raise WrongTypeError(f"{who}: order must be an int, but it is {order!r}.")
    if order < 1:
        raise ValueError(f"{who}: order must be at least 1, but it is {order}.")
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
            f"but it is {origin!r}."
        )
    return (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1_000


def fourier(
    clock: str,
    *,
    period: int | timedelta | str | pl.Expr,
    order: int,
    origin: int | date | datetime | None = None,
) -> list[pl.Expr]:
    """Fourier terms of a fixed period, reduced in integers (module docstring).

    ``period`` is a whole number of the clock's units on an integer clock,
    and a fixed duration (``"1d"``, ``"1w"``, ``"90m"``, a ``timedelta``) on a
    ``Datetime`` or ``Date`` clock. ``origin`` is where every term's phase is
    0: ``0`` on an integer clock; on a temporal one the epoch, or Monday
    1970-01-05 when the period is a whole number of weeks. The terms are
    named ``{clock}_sin{i}_{period}`` and ``{clock}_cos{i}_{period}``, in the
    order ``sin1, cos1, sin2, ...``; a null clock gives null terms.
    """
    who = "fourier"
    if not isinstance(clock, str):
        raise WrongTypeError(f"{who}: clock must be a column name, but it is {clock!r}.")
    order = _check_order(order, who)
    if isinstance(period, (bool, float)):
        raise WrongTypeError(
            f"{who}: period must be an int on an integer clock or a duration on a temporal "
            f"one, but it is {period!r}."
        )
    if isinstance(period, int):
        p = period
        label = str(period)
        t0 = 0 if origin is None else origin
        if isinstance(t0, bool) or not isinstance(t0, int):
            raise WrongTypeError(
                f"{who}: the period is an int, so origin must be an int of the clock's units, "
                f"but it is {origin!r}."
            )
        t = pl.max_horizontal((cs.by_name(clock) & cs.integer()).as_expr()).cast(pl.Int64)
    else:
        try:
            text = duration_text(period, who, "period")
            p = parse_duration(text)
        except ValueError as e:
            if isinstance(period, str) and period.endswith(("mo", "q", "y")):
                raise ValueError(
                    f"{who}: period {period!r} is a calendar unit, whose length varies; "
                    "calendar_fourier takes it."
                ) from None
            if str(e).startswith(who):
                raise
            raise ValueError(f"{who}: period {e}") from None
        label = format_duration(p)
        t0 = (_MONDAY_NS if p % _WEEK_NS == 0 else 0) if origin is None else _origin_ns(origin, who)
        t = _wall(clock, keep_unit=False).dt.epoch("ns")
    if p <= 0:
        raise ValueError(f"{who}: period must be above 0, but it is {label}.")
    if p * order >= 2**63:
        raise ValueError(
            f"{who}: period {label} times order {order} must stay below 2**63 so the "
            "reduction is exact in Int64; lower the order."
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
    """Fourier terms of a calendar period (module docstring), statsmodels'
    ``CalendarFourier``: ``freq`` is one of ``"1h"``, ``"1d"``, ``"1w"``
    (Monday to Sunday), ``"1mo"``, ``"1q"`` or ``"1y"``. The terms are named
    ``{clock}_cal_sin{i}_{freq}`` and ``{clock}_cal_cos{i}_{freq}``, apart from
    ``fourier``'s, whose ``"1d"`` is a fixed span where this one's is the
    calendar day."""
    who = "calendar_fourier"
    if not isinstance(clock, str):
        raise WrongTypeError(f"{who}: clock must be a column name, but it is {clock!r}.")
    order = _check_order(order, who)
    if freq not in _CALENDAR:
        raise ValueError(
            f"{who}: freq must be one of {', '.join(_CALENDAR)}, but it is {freq!r}. "
            "A fixed span of several units is fourier's period."
        )
    t = _wall(clock, keep_unit=True)
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
