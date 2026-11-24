//! `age` (PLAN 556): each point gives each series' value's age, the
//! completing tick's clock less the clock of the tick the value came from.
//! Held here to that definition row by row, joint and pairs, grouped and
//! not, with missing values and ties; to Polars' own subtraction on every
//! clock dtype; across slices and resumed runs; and to its refusals. The
//! Python suite holds it against Polars' `join_asof`
//! (tests/test_stream_carry.py).

use std::collections::HashMap;

use polars::prelude::*;

use super::{RefreshCols, RefreshFile, RefreshTime};
use crate::stream::usable;

/// Four series in a seeded order on a clock with ties, two groups, values
/// missing now and then (null, NaN): `i` is the row, kept.
fn ticks(n: usize) -> DataFrame {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let names = ["a", "b", "c", "d"];
    let (mut series, mut t, mut v, mut g, mut i) = (vec![], vec![], vec![], vec![], vec![]);
    let mut clock = 0.0;
    for row in 0..n {
        let r = next();
        series.push(names[(r % 4) as usize]);
        clock += f64::from(((r >> 4) % 3) as u32) * 0.25;
        t.push(clock);
        v.push(match (r >> 8) % 7 {
            0 => None,
            1 => Some(f64::NAN),
            k => Some(k as f64),
        });
        g.push(if (r >> 20) % 4 == 0 { "y" } else { "x" });
        i.push(row as i64);
    }
    df!("series" => series, "t" => t, "v" => v, "g" => g, "i" => i).unwrap()
}

fn names() -> Vec<String> {
    ["a", "b", "c", "d"].map(str::to_string).to_vec()
}

fn cols<'a>(group: bool, keep: &'a [String]) -> RefreshCols<'a> {
    RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: group.then_some("g"),
        keep,
    }
}

fn sampler(pairs: bool, age: bool) -> RefreshTime {
    RefreshTime::new(names(), pairs).unwrap().with_age(age)
}

/// A joint grid of the two series `a` and `b`, with ages.
fn two() -> RefreshTime {
    RefreshTime::new(vec!["a".into(), "b".into()], false)
        .unwrap()
        .with_age(true)
}

/// The definition: at the completing row `i`, series `s`'s age is the
/// point's clock less the clock of `s`'s last usable tick at or before `i`
/// in its group.
fn want_age(df: &DataFrame, group: bool, i: usize, s: &str) -> f64 {
    let series = df.column("series").unwrap().str().unwrap();
    let t = df.column("t").unwrap().f64().unwrap();
    let v = df.column("v").unwrap().f64().unwrap();
    let g = df.column("g").unwrap().str().unwrap();
    let at = (0..=i)
        .rev()
        .find(|&r| {
            series.get(r) == Some(s)
                && v.get(r).is_some_and(usable)
                && (!group || g.get(r) == g.get(i))
        })
        .expect("a point's every series has ticked");
    t.get(i).unwrap() - t.get(at).unwrap()
}

#[test]
fn each_age_is_the_definition_and_the_rest_of_the_grid_is_unchanged() {
    let df = ticks(400);
    let keep = vec!["i".to_string()];
    let mut points = 0;
    for pairs in [false, true] {
        for group in [false, true] {
            let c = cols(group, &keep);
            let with = sampler(pairs, true).feed(&df, &c).unwrap();
            let without = sampler(pairs, false).feed(&df, &c).unwrap();
            let ages: Vec<String> = if pairs {
                vec!["a_age".into(), "b_age".into()]
            } else {
                names().iter().map(|n| format!("{n}_age")).collect()
            };
            let rest = with.drop_many(ages.clone());
            assert!(rest.equals_missing(&without), "{pairs} {group}");
            let row = with.column("i").unwrap().i64().unwrap();
            let pair = pairs.then(|| with.column("pair").unwrap().str().unwrap().clone());
            for p in 0..with.height() {
                let i = row.get(p).unwrap() as usize;
                let series: Vec<String> = match &pair {
                    Some(pr) => pr.get(p).unwrap().split('|').map(str::to_string).collect(),
                    None => names(),
                };
                for (col, s) in ages.iter().zip(&series) {
                    let got = with.column(col).unwrap().f64().unwrap().get(p).unwrap();
                    assert_eq!(
                        got.to_bits(),
                        want_age(&df, group, i, s).to_bits(),
                        "{col} at {p}"
                    );
                }
                points += 1;
            }
        }
    }
    assert!(points > 200, "{points} points");
}

/// One run as the Python source drives it, in chunks, stopping once
/// `limit` points are out; the output, the state, and the next input row.
fn run(
    state: Option<&[u8]>,
    pairs: bool,
    df: &DataFrame,
    from: usize,
    chunk: usize,
    limit: Option<usize>,
    c: &RefreshCols<'_>,
) -> (DataFrame, Vec<u8>, usize) {
    let mut rt = match state {
        Some(b) => {
            RefreshTime::load_bytes_with_age(b, names(), pairs, c.group.is_some(), true).unwrap()
        }
        None => sampler(pairs, true),
    };
    let rest = df.slice(from as i64, df.height() - from);
    let mut out = sampler(pairs, true).feed(&rest.clear(), c).unwrap();
    let (mut seen, mut at, mut satisfied) = (0, 0, false);
    while at < rest.height() {
        let part = rest.slice(at as i64, chunk);
        at += part.height();
        let got = rt.feed_limited(&part, c, limit.map(|l| l - seen)).unwrap();
        seen += got.height();
        out.vstack_mut(&got).unwrap();
        if limit.is_some_and(|l| seen >= l) {
            satisfied = true;
            break;
        }
    }
    if !satisfied {
        let got = rt
            .feed_limited(&rest.clear(), c, limit.map(|l| l - seen))
            .unwrap();
        out.vstack_mut(&got).unwrap();
    }
    (out, rt.save_bytes().unwrap(), from + rt.rows_fed)
}

/// A slice ending at each point, then a run resumed from its state, gives
/// the one run's grid, ages and all -- a pair grid's slice inside a tick
/// holds that tick's points with their ages -- in chunks of 1, 3 and the
/// whole input; and a loaded state saves its bytes again.
#[test]
fn slices_and_resumed_runs_give_the_one_runs_ages() {
    let df = ticks(120);
    let keep = vec!["i".to_string()];
    for pairs in [false, true] {
        for group in [false, true] {
            let c = cols(group, &keep);
            let (whole, _, _) = run(None, pairs, &df, 0, df.height(), None, &c);
            for chunk in [1, 3, df.height()] {
                for n in 0..=whole.height() {
                    let (head, s, next) = run(None, pairs, &df, 0, chunk, Some(n), &c);
                    let again = RefreshTime::load_bytes_with_age(&s, names(), pairs, group, true)
                        .unwrap()
                        .save_bytes()
                        .unwrap();
                    assert_eq!(again, s);
                    let (tail, _, _) = run(Some(&s), pairs, &df, next, chunk, None, &c);
                    let mut all = head;
                    all.vstack_mut(&tail).unwrap();
                    assert!(all.equals_missing(&whole), "{pairs} {group} {chunk} {n}");
                }
            }
        }
    }
}

/// The age is Polars' subtraction of the clock's dtype, on every clock the
/// sampler takes; a Boolean clock, read as the numbers 0 and 1, as those.
#[test]
fn the_age_is_polars_subtraction_on_every_clock_dtype() {
    let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let raw = Series::new("t".into(), [3i64, 5, 9, 12]);
    let day = |s: &Series| {
        s.cast(&DataType::Int32)
            .unwrap()
            .i32()
            .unwrap()
            .clone()
            .into_date()
            .into_series()
    };
    let clocks = [
        raw.cast(&DataType::Int32).unwrap(),
        raw.cast(&DataType::UInt8).unwrap(),
        raw.cast(&DataType::Float32).unwrap(),
        (raw.clone() * 1_000_000)
            .i64()
            .unwrap()
            .clone()
            .into_datetime(TimeUnit::Microseconds, tz)
            .into_series(),
        day(&raw),
        raw.i64()
            .unwrap()
            .clone()
            .into_duration(TimeUnit::Milliseconds)
            .into_series(),
    ];
    for t in clocks {
        let dtype = t.dtype().clone();
        let df = DataFrame::new(
            4,
            vec![
                Column::new("series".into(), ["a", "b", "a", "b"]),
                t.clone().into_column(),
                Column::new("v".into(), [1.0, 2.0, 3.0, 4.0]),
            ],
        )
        .unwrap();
        let mut rt = two();
        let c = RefreshCols {
            series: "series",
            clock: "t",
            value: "v",
            group: None,
            keep: &[],
        };
        let out = rt.feed(&df, &c).unwrap();
        let want_dtype = (&t - &t).unwrap().dtype().clone();
        assert_eq!(
            rt.schema(&c, df.schema()).get("a_age"),
            Some(&want_dtype),
            "{dtype:?}"
        );
        // Points at rows 1 and 3: a's ticks at rows 0 and 2.
        let now = t.take_slice(&[1, 3]).unwrap();
        let a = (&now - &t.take_slice(&[0, 2]).unwrap()).unwrap();
        let b = (&now - &now).unwrap();
        for (name, want) in [("a_age", a), ("b_age", b)] {
            let got = out.column(name).unwrap().as_materialized_series();
            assert!(
                got.equals(&want.with_name(name.into())),
                "{dtype:?} {name}: {got:?}"
            );
        }
    }
    // A Boolean clock: the numbers 0 and 1.
    let df = df!("series" => ["a", "b"], "t" => [false, true], "v" => [1.0, 2.0]).unwrap();
    let c = RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: None,
        keep: &[],
    };
    let out = two().feed(&df, &c).unwrap();
    let a: Vec<Option<f64>> = out.column("a_age").unwrap().f64().unwrap().iter().collect();
    assert_eq!(a, [Some(1.0)]);
}

/// A temporal state resumes on another unit: an age of a tick before the
/// save is read in the new unit, floored.
#[test]
fn a_temporal_state_resumed_on_another_unit_ages_in_it() {
    let ns = |v: &[i64], u: TimeUnit| {
        Series::new("t".into(), v)
            .i64()
            .unwrap()
            .clone()
            .into_datetime(u, None)
            .into_column()
    };
    let first = DataFrame::new(
        1,
        vec![
            Column::new("series".into(), ["a"]),
            ns(&[1_500_000], TimeUnit::Nanoseconds),
            Column::new("v".into(), [1.0]),
        ],
    )
    .unwrap();
    let c = RefreshCols {
        series: "series",
        clock: "t",
        value: "v",
        group: None,
        keep: &[],
    };
    let mut rt = two();
    rt.feed(&first, &c).unwrap();
    let state = rt.save_bytes().unwrap();
    let then = DataFrame::new(
        1,
        vec![
            Column::new("series".into(), ["b"]),
            ns(&[4], TimeUnit::Milliseconds),
            Column::new("v".into(), [2.0]),
        ],
    )
    .unwrap();
    let two_names = vec!["a".to_string(), "b".to_string()];
    let mut rt = RefreshTime::load_bytes_with_age(&state, two_names, false, false, true).unwrap();
    let out = rt.feed(&then, &c).unwrap();
    assert_eq!(out.height(), 1);
    // a's tick at 1.5 ms reads as 1 ms on a millisecond clock: 4 - 1 = 3 ms.
    let a = out
        .column("a_age")
        .unwrap()
        .duration()
        .unwrap()
        .physical()
        .get(0);
    assert_eq!(a, Some(3));
}

#[test]
fn a_state_gives_ages_only_where_it_was_saved_with_them() {
    let df = ticks(30);
    let keep: Vec<String> = Vec::new();
    let c = cols(false, &keep);
    let saved = |age: bool| {
        let mut rt = sampler(false, age);
        rt.feed(&df, &c).unwrap();
        rt.save_bytes().unwrap()
    };
    let e = RefreshTime::load_bytes_with_age(&saved(false), names(), false, false, true)
        .err()
        .unwrap();
    assert!(
        e.contains("saved with age = False, but this call gives age = True"),
        "{e}"
    );
    assert!(
        e.contains("does not hold the clock of each series' last tick"),
        "{e}"
    );
    let e = RefreshTime::load_bytes(&saved(true), names(), false, false)
        .err()
        .unwrap();
    assert!(
        e.contains("saved with age = True, but this call gives age = False"),
        "{e}"
    );
    // Without ages the state is the bytes it was: no `age` and no `ats`.
    let v = rmpv::decode::read_value(&mut &saved(false)[..]).unwrap();
    let text = format!("{v}");
    assert!(
        !text.contains("\"age\"") && !text.contains("\"ats\""),
        "{text}"
    );
    let v = rmpv::decode::read_value(&mut &saved(true)[..]).unwrap();
    let text = format!("{v}");
    assert!(
        text.contains("\"age\": true") && text.contains("\"ats\""),
        "{text}"
    );
}

/// Write `f` back as a sealed state and load it with ages.
fn load(mut f: RefreshFile, pairs: bool) -> String {
    f.checksum = Some(crate::checksum::Trailer::Patched);
    let mut bytes = rmp_serde::to_vec_named(&f).unwrap();
    crate::checksum::seal(&mut bytes);
    RefreshTime::load_bytes_with_age(&bytes, names(), pairs, false, true)
        .err()
        .unwrap_or_default()
}

#[test]
fn a_state_whose_ages_do_not_hold_together_is_damaged() {
    let df = ticks(41);
    let keep = vec!["i".to_string()];
    let c = cols(false, &keep);
    let mut rt = sampler(false, true);
    rt.feed(&df, &c).unwrap();
    let saved: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    assert_eq!(
        load(
            rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap(),
            false
        ),
        ""
    );
    // A series seen without its tick's clock, and one unseen with one.
    let mut f: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    let grid = &mut f.states[0].1[0];
    let seen = grid
        .series
        .iter()
        .position(|s| s.seen)
        .expect("a series seen");
    grid.ats[seen] = None;
    assert!(load(f, false).contains("does not hold 1 sets of 4 series"));
    let mut f: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    let grid = &mut f.states[0].1[0];
    let unseen = grid
        .series
        .iter()
        .position(|s| !s.seen)
        .expect("a series unseen");
    grid.ats[unseen] = grid.ats[seen];
    assert!(load(f, false).contains("does not hold 1 sets of 4 series"));
    // A grid without a clock a series, under ages; and one with them,
    // loaded without ages.
    let mut f: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    f.states[0].1[0].ats.pop();
    assert!(load(f, false).contains("does not hold 1 sets of 4 series"));
    let mut f = saved;
    f.age = false;
    f.checksum = Some(crate::checksum::Trailer::Patched);
    let mut bytes = rmp_serde::to_vec_named(&f).unwrap();
    crate::checksum::seal(&mut bytes);
    let e = RefreshTime::load_bytes(&bytes, names(), false, false)
        .err()
        .unwrap();
    assert!(e.contains("does not hold 1 sets of 4 series"), "{e}");
    // A pair grid's held points: each needs its values' clocks.
    let mut by_tick: HashMap<i64, usize> = HashMap::new();
    let whole = sampler(true, true).feed(&df, &c).unwrap();
    let i = whole.column("i").unwrap().i64().unwrap();
    for r in i.into_no_null_iter() {
        *by_tick.entry(r).or_default() += 1;
    }
    let at = (0..whole.height())
        .find(|&p| by_tick[&i.get(p).unwrap()] > 1 && (p == 0 || i.get(p - 1) != i.get(p)))
        .expect("a tick completing two pairs' points");
    let mut rt = sampler(true, true);
    rt.feed_limited(&df, &c, Some(at + 1)).unwrap();
    let held: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    assert!(held.held.as_ref().unwrap().ats.is_some());
    assert_eq!(
        load(
            rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap(),
            true
        ),
        ""
    );
    let mut f: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    f.held.as_mut().unwrap().ats = None;
    assert!(load(f, true).contains("are not of 6 sets of 2 series"));
    let mut f: RefreshFile = rmp_serde::from_slice(&rt.save_bytes().unwrap()).unwrap();
    f.held.as_mut().unwrap().ats.as_mut().unwrap()[0].pop();
    assert!(load(f, true).contains("are not of 6 sets of 2 series"));
    let mut f = held;
    f.held
        .as_mut()
        .unwrap()
        .ats
        .as_mut()
        .unwrap()
        .push(Vec::new());
    assert!(load(f, true).contains("are not of 6 sets of 2 series"));
}
