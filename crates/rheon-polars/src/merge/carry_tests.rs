//! `carry` (PLAN 556): each named column filled forward over the merged
//! rows within its group and day, with its age on the clock beside it. Held
//! here to the definition, row by row over a merge without `carry`, for
//! random inputs, groups and chunkings; to Polars' own subtraction for the
//! age on every clock dtype; across a save at every cut; and to its
//! refusals. The Python suite holds it against Polars' `join_asof` and
//! `forward_fill` (tests/test_stream_carry.py).

use super::tests::{cfg, chunked, drive, lcg, merge_of, names};
use super::*;

/// Random inputs on a float clock with ties within and across them: `x`
/// (Int64) in every input, `s` (String) in the first, `only_<i>` (Float64)
/// in every other; each null about half the time; a group `g` of "a", "b"
/// or null.
fn inputs(seed: u64, n: usize, len: usize) -> Vec<DataFrame> {
    let mut r = lcg(seed);
    (0..n)
        .map(|i| {
            let m = (r() as usize) % len;
            let mut t = 0.0;
            let mut ts = Vec::new();
            let mut x = Vec::new();
            let mut other = Vec::new();
            let mut g = Vec::new();
            for row in 0..m {
                t += f64::from((r() % 3) as u32) / 2.0;
                ts.push(t);
                x.push(r().is_multiple_of(2).then_some((i * 100 + row) as i64));
                other.push(r().is_multiple_of(2).then_some(row as f64 + 0.25));
                g.push(match r() % 3 {
                    0 => Some("a"),
                    1 => Some("b"),
                    _ => None,
                });
            }
            let mut df = df!("t" => ts, "x" => x, "g" => g).unwrap();
            let col = if i == 0 {
                Column::new(
                    "s".into(),
                    other
                        .iter()
                        .map(|v| v.map(|v| format!("s{v}")))
                        .collect::<Vec<_>>(),
                )
            } else {
                Column::new(format!("only_{i}").into(), other)
            };
            df.with_column(col).unwrap();
            df
        })
        .collect()
}

fn carry_cfg(carry: &[&str], by: bool) -> MergeConfig {
    let mut c = cfg("t");
    c.group = by.then(|| "g".into());
    c.carry = carry.iter().map(|s| (*s).to_string()).collect();
    c
}

/// The columns `inputs` gives that `carry` can fill: `x`, `s` and each
/// `only_<i>` an input has.
fn carryable(inputs: &[DataFrame]) -> Vec<String> {
    let mut out = vec!["x".to_string()];
    if !inputs.is_empty() {
        out.push("s".into());
    }
    for i in 1..inputs.len() {
        out.push(format!("only_{i}"));
    }
    out
}

/// The definition, over the merged rows without `carry`: per group, a
/// row's value of a carried column is its own, else the group's last
/// non-null one before it; its age the row's clock less that value's.
fn definition(plain: &DataFrame, carry: &[String], by: bool) -> Vec<(Series, Vec<Option<f64>>)> {
    let t = plain.column("t").unwrap().f64().unwrap();
    let g = plain.column("g").unwrap().str().unwrap();
    carry
        .iter()
        .map(|c| {
            let col = plain.column(c).unwrap().as_materialized_series();
            let mut last: HashMap<Option<&str>, usize> = HashMap::new();
            let mut take = Vec::new();
            let mut age = Vec::new();
            for j in 0..plain.height() {
                let key = if by { g.get(j) } else { None };
                if col.get(j).unwrap() != AnyValue::Null {
                    last.insert(key, j);
                }
                match last.get(&key) {
                    Some(&i) => {
                        take.push(Some(i as IdxSize));
                        age.push(Some(t.get(j).unwrap() - t.get(i).unwrap()));
                    }
                    None => {
                        take.push(None);
                        age.push(None);
                    }
                }
            }
            let idx = IdxCa::from_iter_options("".into(), take.into_iter());
            (col.take(&idx).unwrap(), age)
        })
        .collect()
}

fn chunkings(inputs: &[DataFrame], sizes: &[usize]) -> Vec<Vec<DataFrame>> {
    inputs.iter().map(|d| chunked(d, sizes)).collect()
}

#[test]
fn carry_is_each_groups_forward_fill_however_the_chunks_fall() {
    let mut checked = 0;
    for seed in 0..40u64 {
        let n = 1 + (seed as usize % 4);
        let ins = inputs(seed, n, 30);
        let carry = carryable(&ins);
        let carry_refs: Vec<&str> = carry.iter().map(String::as_str).collect();
        for by in [false, true] {
            let mut plain_cfg = cfg("t");
            plain_cfg.group = by.then(|| "g".into());
            let plain = drive(
                &mut merge_of(&ins, plain_cfg),
                &chunkings(&ins, &[1000]),
                None,
                false,
            );
            let want = definition(&plain, &carry, by);
            for sizes in [&[1000usize][..], &[1], &[2, 1, 3], &[7]] {
                let got = drive(
                    &mut merge_of(&ins, carry_cfg(&carry_refs, by)),
                    &chunkings(&ins, sizes),
                    None,
                    false,
                );
                assert_eq!(got.height(), plain.height());
                for (c, (values, ages)) in carry.iter().zip(&want) {
                    let filled = got.column(c).unwrap().as_materialized_series();
                    assert!(
                        filled.equals_missing(values),
                        "seed {seed} by {by} {sizes:?}: {c} is {filled:?}, not {values:?}"
                    );
                    let age: Vec<Option<f64>> = got
                        .column(&format!("{c}_age"))
                        .unwrap()
                        .f64()
                        .unwrap()
                        .iter()
                        .collect();
                    assert_eq!(&age, ages, "seed {seed} by {by} {sizes:?}: {c}_age");
                    checked += usize::from(values.null_count() < values.len());
                }
                // Every column not carried is the merge's own.
                for name in plain.get_column_names() {
                    if !carry.iter().any(|c| c == name.as_str()) {
                        assert!(
                            got.column(name)
                                .unwrap()
                                .equals_missing(plain.column(name).unwrap()),
                            "{name}"
                        );
                    }
                }
            }
        }
    }
    assert!(checked > 100, "only {checked} carried columns had values");
}

#[test]
fn a_column_with_no_null_is_its_own_value_at_age_zero() {
    let a = df!("t" => [1.0, 2.0], "x" => [5i64, 6]).unwrap();
    let b = df!("t" => [1.5], "y" => [7i64]).unwrap();
    let ins = [a.clone(), b.clone()];
    // `x` is null on `b`'s row, `y` on `a`'s: each filled from its own.
    let mut m = merge_of(&ins, carry_cfg(&["x", "y"], false));
    let got = drive(&mut m, &[vec![a.clone()], vec![b]], None, false);
    let x: Vec<Option<i64>> = got.column("x").unwrap().i64().unwrap().iter().collect();
    assert_eq!(x, [Some(5), Some(5), Some(6)]);
    let y: Vec<Option<i64>> = got.column("y").unwrap().i64().unwrap().iter().collect();
    assert_eq!(y, [None, Some(7), Some(7)]);
    let ya: Vec<Option<f64>> = got.column("y_age").unwrap().f64().unwrap().iter().collect();
    assert_eq!(ya, [None, Some(0.0), Some(0.5)]);
    // An input alone, its column never null: the column as it was, age 0.
    let mut m = merge_of(std::slice::from_ref(&a), carry_cfg(&["x"], false));
    let got = drive(&mut m, &[vec![a.clone()]], None, false);
    assert!(got.column("x").unwrap().equals(a.column("x").unwrap()));
    let xa: Vec<Option<f64>> = got.column("x_age").unwrap().f64().unwrap().iter().collect();
    assert_eq!(xa, [Some(0.0), Some(0.0)]);
}

/// Under `restart_after_step_back` a value is carried within its day: the
/// first rows of a day that started again find nothing until a value of
/// that day comes.
#[test]
fn a_value_is_carried_within_its_day_alone() {
    // `a` starts again at 20, a step back of 30 from 50; `b` ends before.
    let a = df!(
        "t" => [10i64, 50, 20, 30, 40],
        "x" => [Some(1i64), None, None, Some(4), None],
    )
    .unwrap();
    let b = df!("t" => [15i64], "y" => [9i64]).unwrap();
    let mut c = cfg("t");
    c.restart = Some(Restart::Int(5));
    c.restart_text = Some("5".into());
    c.carry = vec!["x".into()];
    let ins = [a.clone(), b.clone()];
    for sizes in [&[10usize][..], &[1]] {
        let mut m = merge_of(&ins, c.clone());
        let got = drive(&mut m, &chunkings(&ins, sizes), None, false);
        let t: Vec<Option<i64>> = got.column("t").unwrap().i64().unwrap().iter().collect();
        assert_eq!(
            t,
            [Some(10), Some(15), Some(50), Some(20), Some(30), Some(40)]
        );
        // The second day's row at 20 is past the first day's 1 at 10 in
        // clock, and finds nothing: that value is the other day's.
        let x: Vec<Option<i64>> = got.column("x").unwrap().i64().unwrap().iter().collect();
        assert_eq!(x, [Some(1), Some(1), Some(1), None, Some(4), Some(4)]);
        let age: Vec<Option<i64>> = got.column("x_age").unwrap().i64().unwrap().iter().collect();
        assert_eq!(age, [Some(0), Some(5), Some(40), None, Some(0), Some(10)]);
    }
}

/// A resumed run whose group's merged rows step back past the setting
/// starts that group again, and it carries nothing from before: its clock
/// is not comparable with the one its value was read at.
#[test]
fn a_group_whose_merged_rows_start_again_carries_nothing_from_before() {
    let mut c = cfg("t");
    c.group = Some("g".into());
    c.restart = Some(Restart::Float(10.0));
    c.restart_text = Some("10".into());
    c.carry = vec!["x".into()];
    let a1 = df!("t" => [100.0], "g" => ["k"], "x" => [Some(1i64)]).unwrap();
    let b1 = df!("t" => [40.0], "g" => ["j"], "x" => [Some(2i64)]).unwrap();
    let ins = [a1.clone(), b1.clone()];
    let mut m = merge_of(&ins, c.clone());
    let _ = drive(&mut m, &[vec![a1], vec![b1]], None, false);
    let state = m.save_bytes().unwrap();
    let a2 = df!("t" => [101.0], "g" => ["k"], "x" => [None::<i64>]).unwrap();
    let b2 =
        df!("t" => [50.0, 60.0, 101.5], "g" => ["k", "k", "k"], "x" => [None::<i64>, None, None])
            .unwrap();
    let mut m = Merge::load_bytes(
        &state,
        names(2),
        ins.iter().map(|d| d.schema().as_ref().clone()).collect(),
        c,
    )
    .unwrap();
    let got = drive(&mut m, &[vec![a2], vec![b2]], None, false);
    let t: Vec<Option<f64>> = got.column("t").unwrap().f64().unwrap().iter().collect();
    assert_eq!(t, [Some(50.0), Some(60.0), Some(101.0), Some(101.5)]);
    // Before the group started again at 50, `k` carried 1 from 100; after,
    // nothing, even at 101, past the clock the value was read at.
    let x: Vec<Option<i64>> = got.column("x").unwrap().i64().unwrap().iter().collect();
    assert_eq!(x, [None, None, None, None]);
    assert_eq!(got.column("x_age").unwrap().null_count(), 4);
}

/// The age is Polars' subtraction of the clock's dtype: its dtype and its
/// value, on every clock a merge takes.
#[test]
fn the_age_is_polars_subtraction_on_every_clock_dtype() {
    let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let raw = Series::new("t".into(), [3i64, 5, 9, 12]);
    let clocks = [
        raw.cast(&DataType::Int32).unwrap(),
        raw.cast(&DataType::UInt8).unwrap(),
        raw.cast(&DataType::UInt64).unwrap(),
        raw.cast(&DataType::Float32).unwrap(),
        raw.cast(&DataType::Float64).unwrap(),
        (raw.clone() * 1_000_000)
            .i64()
            .unwrap()
            .clone()
            .into_datetime(TimeUnit::Microseconds, tz.clone())
            .into_series(),
        raw.cast(&DataType::Int32)
            .unwrap()
            .i32()
            .unwrap()
            .clone()
            .into_date()
            .into_series(),
        raw.i64()
            .unwrap()
            .clone()
            .into_duration(TimeUnit::Milliseconds)
            .into_series(),
        raw.i64()
            .unwrap()
            .clone()
            .into_datetime(TimeUnit::Nanoseconds, None)
            .into_series(),
    ];
    for t in clocks {
        let dtype = t.dtype().clone();
        // Rows 0 and 2 carry `x`; 1 and 3 read it.
        let a = DataFrame::new(
            2,
            vec![
                t.take_slice(&[0, 2]).unwrap().into_column(),
                Column::new("x".into(), [1.5, 2.5]),
            ],
        )
        .unwrap();
        let b = DataFrame::new(2, vec![t.take_slice(&[1, 3]).unwrap().into_column()]).unwrap();
        let ins = [a.clone(), b.clone()];
        let mut m = merge_of(&ins, carry_cfg(&["x"], false));
        assert_eq!(
            m.schema().get("x_age"),
            Some(&(&t - &t).unwrap().dtype().clone()),
            "{dtype:?}"
        );
        let got = drive(&mut m, &[vec![a], vec![b]], None, false);
        let src = t.take_slice(&[0, 0, 2, 2]).unwrap();
        let want = (&t - &src).unwrap().with_name("x_age".into());
        let age = got.column("x_age").unwrap().as_materialized_series();
        assert!(age.equals(&want), "{dtype:?}: {age:?} is not {want:?}");
    }
}

/// Columns of every kind a value can be, so a state carries each form.
fn kinds_inputs() -> Vec<DataFrame> {
    let tz = TimeZone::opt_try_new(Some("Europe/Amsterdam")).unwrap();
    let n = 6;
    let some = |i: usize| i.is_multiple_of(3);
    let list = Series::new(
        "l".into(),
        (0..n)
            .map(|i| Series::new("".into(), [i as i64, -(i as i64)]))
            .collect::<Vec<_>>(),
    );
    let list = list
        .zip_with(
            &BooleanChunked::from_iter_values("".into(), (0..n).map(some)),
            &Series::full_null("l".into(), n, list.dtype()),
        )
        .unwrap();
    let a = DataFrame::new(
        n,
        vec![
            Column::new("t".into(), (0..n).map(|i| i as f64).collect::<Vec<_>>()),
            Column::new(
                "g".into(),
                (0..n).map(|i| ["p", "q"][i % 2]).collect::<Vec<_>>(),
            ),
            Column::new(
                "b".into(),
                (0..n)
                    .map(|i| some(i).then_some(i % 2 == 0))
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "i8".into(),
                (0..n)
                    .map(|i| some(i).then_some(i as i8))
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "u".into(),
                (0..n)
                    .map(|i| some(i).then_some(i as u32))
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "s".into(),
                (0..n)
                    .map(|i| some(i).then(|| format!("v{i}")))
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "d".into(),
                (0..n)
                    .map(|i| some(i).then_some(19_000 + i as i32))
                    .collect::<Vec<_>>(),
            )
            .cast(&DataType::Date)
            .unwrap(),
            Series::new(
                "z".into(),
                (0..n)
                    .map(|i| some(i).then_some(i as i64 * 1_000))
                    .collect::<Vec<_>>(),
            )
            .i64()
            .unwrap()
            .clone()
            .into_datetime(TimeUnit::Milliseconds, tz)
            .into_column(),
            Column::new(
                "w".into(),
                (0..n)
                    .map(|i| some(i).then_some(i as i64))
                    .collect::<Vec<_>>(),
            )
            .cast(&DataType::Duration(TimeUnit::Microseconds))
            .unwrap(),
            Column::new(
                "h".into(),
                (0..n)
                    .map(|i| some(i).then_some(i as i64 * 1_000_000_000))
                    .collect::<Vec<_>>(),
            )
            .cast(&DataType::Time)
            .unwrap(),
            list.into_column(),
        ],
    )
    .unwrap();
    let b = df!(
        "t" => (0..n).map(|i| i as f64 + 0.5).collect::<Vec<_>>(),
        "g" => (0..n).map(|i| ["q", "p"][i % 2]).collect::<Vec<_>>(),
    )
    .unwrap();
    vec![a, b]
}

const KINDS: [&str; 9] = ["b", "i8", "u", "s", "d", "z", "w", "h", "l"];

/// Saved at every cut -- a slice at every row, and a run held at every
/// clock -- and resumed, the two runs give the one run's rows, a value of
/// every kind carried across the save; and a loaded state saves its bytes
/// again.
#[test]
fn a_save_at_every_cut_resumes_to_the_one_run() {
    let ins = kinds_inputs();
    let c = carry_cfg(&KINDS, true);
    let schemas = || {
        ins.iter()
            .map(|d| d.schema().as_ref().clone())
            .collect::<Vec<_>>()
    };
    let whole = drive(
        &mut merge_of(&ins, c.clone()),
        &chunkings(&ins, &[100]),
        None,
        false,
    );
    let src = whole.column("source").unwrap().str().unwrap().clone();
    for limit in 1..whole.height() {
        for sizes in [&[1usize][..], &[4]] {
            let mut m = merge_of(&ins, c.clone());
            let head = drive(&mut m, &chunkings(&ins, sizes), Some(limit), false);
            assert_eq!(head.height(), limit);
            let state = m.save_bytes().unwrap();
            let again = Merge::load_bytes(&state, names(2), schemas(), c.clone()).unwrap();
            assert_eq!(
                again.save_bytes().unwrap(),
                state,
                "a loaded state saves its bytes"
            );
            let rest: Vec<DataFrame> = ins
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let took = src
                        .slice(0, limit)
                        .iter()
                        .filter(|s| *s == Some(&format!("s{i}")[..]))
                        .count();
                    d.slice(took as i64, d.height() - took)
                })
                .collect();
            let mut m = Merge::load_bytes(&state, names(2), schemas(), c.clone()).unwrap();
            let tail = drive(&mut m, &chunkings(&rest, sizes), None, false);
            let mut both = head.clone();
            both.vstack_mut(&tail).unwrap();
            assert!(both.equals_missing(&whole), "slice at {limit}, {sizes:?}");
        }
    }
    // A run that holds the rows at its last clock, cut at each clock.
    for cut in 1..6 {
        let first: Vec<DataFrame> = ins.iter().map(|d| d.head(Some(cut))).collect();
        let rest: Vec<DataFrame> = ins
            .iter()
            .map(|d| d.slice(cut as i64, d.height() - cut))
            .collect();
        let mut m = merge_of(&ins, c.clone());
        let head = drive(&mut m, &chunkings(&first, &[2]), None, true);
        let state = m.save_bytes().unwrap();
        let mut m = Merge::load_bytes(&state, names(2), schemas(), c.clone()).unwrap();
        let tail = drive(&mut m, &chunkings(&rest, &[3]), None, false);
        let mut both = head.clone();
        both.vstack_mut(&tail).unwrap();
        assert!(both.equals_missing(&whole), "cut at {cut}");
    }
}

#[test]
fn a_bad_carry_is_refused_by_name() {
    let a = df!("t" => [1.0], "x" => [1i64], "g" => ["a"]).unwrap();
    let b = df!("t" => [2.0], "y" => [1i64], "y_age" => [0.0], "g" => ["a"]).unwrap();
    let refused = |carry: &[&str], by: bool| {
        let schemas = vec![a.schema().as_ref().clone(), b.schema().as_ref().clone()];
        Merge::new(names(2), schemas, carry_cfg(carry, by))
            .err()
            .expect("refused")
            .to_string()
    };
    assert!(refused(&["nope"], false).contains("carry names column \"nope\", which no input has"));
    assert!(refused(&["t"], false).contains("the clock column"));
    assert!(refused(&["g"], true).contains("the group column"));
    assert!(refused(&["x", "x"], false).contains("names column \"x\" twice"));
    assert!(refused(&["y"], false).contains("an input's column"));
    let mut c = carry_cfg(&["x"], false);
    c.source = "x_age".into();
    let e = Merge::new(names(2), vec![a.schema().as_ref().clone(); 2], c)
        .err()
        .expect("refused")
        .to_string();
    assert!(e.contains("the source column"), "{e}");
    // Carrying `g` without `by` is a column like any other.
    assert!(
        Merge::new(
            names(2),
            vec![a.schema().as_ref().clone(), a.schema().as_ref().clone()],
            carry_cfg(&["g"], false)
        )
        .is_ok()
    );
}

#[test]
fn a_state_carries_only_into_the_carry_it_was_saved_with() {
    let a = df!("t" => [1.0, 2.0], "x" => ["p", "q"], "y" => [1i64, 2]).unwrap();
    let ins = [a.clone()];
    let schemas = || vec![a.schema().as_ref().clone()];
    let saved = |carry: &[&str]| {
        let mut m = merge_of(&ins, carry_cfg(carry, false));
        let _ = drive(&mut m, &[vec![a.clone()]], None, false);
        m.save_bytes().unwrap()
    };
    let err = |state: &[u8], carry: &[&str], schemas: Vec<Schema>| {
        Merge::load_bytes(state, names(1), schemas, carry_cfg(carry, false))
            .err()
            .expect("refused")
            .to_string()
    };
    let e = err(&saved(&[]), &["x"], schemas());
    assert!(
        e.contains("saved carrying [], but this call carries [\"x\"]"),
        "{e}"
    );
    let e = err(&saved(&["x"]), &[], schemas());
    assert!(
        e.contains("saved carrying [\"x\"], but this call carries []"),
        "{e}"
    );
    let e = err(&saved(&["x", "y"]), &["y", "x"], schemas());
    assert!(e.contains("saved carrying [\"x\", \"y\"]"), "{e}");
    // A value the column cannot hold: text where the call has a number.
    let ints = df!("t" => [1.0], "x" => [1i64], "y" => [1i64]).unwrap();
    let e = err(&saved(&["x"]), &["x"], vec![ints.schema().as_ref().clone()]);
    assert!(
        e.contains("carries a value of column \"x\" that this call's \"x\", of dtype Int64"),
        "{e}"
    );
    // A state without carry is the bytes it was: no `carry` entry at all.
    let plain = saved(&[]);
    let v = rmpv::decode::read_value(&mut &plain[..]).unwrap();
    assert!(
        v.as_map()
            .unwrap()
            .iter()
            .all(|(k, _)| k.as_str() != Some("carry"))
    );
}

/// A long stream in chunks of one row appends a value a pull: the store is
/// gathered down to the values in use whenever it passes twice the groups
/// and 1,024, so it stays bounded, and every value read after a gather is
/// the definition's.
#[test]
fn the_store_of_values_stays_bounded_over_a_long_stream() {
    let n = 5_000usize;
    let a = df!(
        "t" => (0..n).map(|i| i as f64).collect::<Vec<_>>(),
        "g" => (0..n).map(|i| ["a", "b", "c"][i % 3]).collect::<Vec<_>>(),
        "x" => (0..n).map(|i| (i % 5 != 0).then(|| format!("v{i}"))).collect::<Vec<_>>(),
    )
    .unwrap();
    let ins = [a.clone()];
    let mut plain_cfg = cfg("t");
    plain_cfg.group = Some("g".into());
    let plain = drive(
        &mut merge_of(&ins, plain_cfg),
        &[vec![a.clone()]],
        None,
        false,
    );
    let want = definition(&plain, &["x".to_string()], true);
    let mut m = merge_of(&ins, carry_cfg(&["x"], true));
    let got = drive(&mut m, &[chunked(&a, &[1])], None, false);
    assert!(
        got.column("x")
            .unwrap()
            .as_materialized_series()
            .equals_missing(&want[0].0)
    );
    // About 4,000 values were appended, one a pull: only the gathers keep
    // the store under its bound.
    let store = m.carry[0].store.len();
    assert!(store <= 2 * 3 + 1024, "the store holds {store} values");
}
