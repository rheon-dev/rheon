//! The age of a carried value on the clock (PLAN 556): the clock of the row
//! it is read on less the clock of the row it came from, computed by Polars'
//! own subtraction of the clock's dtype, so the age is what `pl.col(clock) -
//! pl.col(clock_then)` gives -- its dtype and its value -- on every clock
//! `rp.stream.merge` and `rp.stream.refresh_time` take: an integer one gives
//! an integer of its dtype, a float one a float, a `Datetime`, a `Date` or a
//! `Duration` a `Duration`. A Boolean clock, which `refresh_time` reads as
//! the numbers 0 and 1, is subtracted as those numbers, since Polars
//! subtracts no Booleans.

use polars::prelude::*;

/// The dtype the clock is subtracted in: its own, a Boolean's as a float.
fn work(clock: &DataType) -> DataType {
    match clock {
        DataType::Boolean => DataType::Float64,
        d => d.clone(),
    }
}

/// The age's dtype on a clock of `clock`'s dtype: Polars' own for the
/// difference of two such clocks.
pub(crate) fn dtype(clock: &DataType) -> PolarsResult<DataType> {
    let (a, b) = (
        Series::new_empty(PlSmallStr::EMPTY, &work(clock)),
        Series::new_empty(PlSmallStr::EMPTY, &work(clock)),
    );
    Ok((&a - &b)?.dtype().clone())
}

/// `now - then`, named `name`: each a clock of one dtype, `then` null where
/// there is no value to age.
pub(crate) fn of(name: &str, now: &Series, then: &Series) -> PolarsResult<Series> {
    let w = work(now.dtype());
    // Cast only a column not yet of the dtype. Polars' Rust crate (0.55.2)
    // built without its `timezones` feature casts to `Datetime(unit, zone)`
    // and returns the column naive (`chunked_array/cast.rs`), while a column
    // of nulls alone keeps the zone (`Series::cast`'s all-null path), so the
    // two failed to find a supertype. The workspace now enables `timezones`;
    // the cast is skipped all the same, being no work.
    let as_w = |s: &Series| {
        if s.dtype() == &w {
            Ok(s.clone())
        } else {
            s.cast(&w)
        }
    };
    Ok((&as_w(now)? - &as_w(then)?)?.with_name(name.into()))
}

/// A clock of `dtype` from its physical integers: a temporal clock's in its
/// own unit (days for a `Date`), an integer clock's as themselves.
pub(crate) fn clock_of_physical(dtype: &DataType, v: Vec<Option<i64>>) -> PolarsResult<Series> {
    let ints = Int64Chunked::from_iter_options(PlSmallStr::EMPTY, v.into_iter());
    Ok(match dtype {
        DataType::Datetime(u, tz) => ints.into_datetime(*u, tz.clone()).into_series(),
        DataType::Duration(u) => ints.into_duration(*u).into_series(),
        DataType::Date => ints
            .into_series()
            .cast(&DataType::Int32)?
            .i32()?
            .clone()
            .into_date()
            .into_series(),
        d => ints.into_series().cast(&work(d))?,
    })
}

/// A clock of `dtype` from its values as floats.
pub(crate) fn clock_of_floats(dtype: &DataType, v: Vec<Option<f64>>) -> PolarsResult<Series> {
    Float64Chunked::from_iter_options(PlSmallStr::EMPTY, v.into_iter())
        .into_series()
        .cast(&work(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zoned clock cast to its own dtype keeps its zone: without Polars'
    /// `timezones` feature the cast returned it naive, and this test fails.
    #[test]
    fn a_zoned_clock_cast_to_its_own_dtype_keeps_its_zone() {
        let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
        let d = DataType::Datetime(TimeUnit::Microseconds, tz.clone());
        let now = Int64Chunked::from_vec("t".into(), vec![5, 6])
            .into_datetime(TimeUnit::Microseconds, tz)
            .into_series();
        assert_eq!(now.cast(&d).unwrap().dtype(), &d);
        let none = clock_of_physical(&d, vec![None, None]).unwrap();
        assert_eq!(none.cast(&d).unwrap().dtype(), &d);
        let diff = (&now.cast(&d).unwrap() - &none.cast(&d).unwrap()).unwrap();
        assert_eq!(diff.dtype(), &DataType::Duration(TimeUnit::Microseconds));
    }

    /// A zoned clock with no value to age on any row: nulls of the age's
    /// dtype.
    #[test]
    fn a_zoned_clock_with_nothing_to_age_gives_nulls() {
        let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
        let d = DataType::Datetime(TimeUnit::Microseconds, tz.clone());
        let now = Int64Chunked::from_vec("t".into(), vec![5, 6])
            .into_datetime(TimeUnit::Microseconds, tz)
            .into_series();
        let none = clock_of_physical(&d, vec![None, None]).unwrap();
        let age = of("a", &now, &none).unwrap();
        assert_eq!(age.dtype(), &DataType::Duration(TimeUnit::Microseconds));
        assert_eq!(age.null_count(), 2);
        let one = clock_of_physical(&d, vec![Some(1), None]).unwrap();
        let age = of("a", &now, &one).unwrap();
        assert_eq!(age.duration().unwrap().physical().get(0), Some(4));
        assert_eq!(age.null_count(), 1);
    }
}
