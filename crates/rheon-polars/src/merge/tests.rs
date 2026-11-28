//! The merge's unit tests: the order against a stable sort of every row by
//! its key, chunk invariance, the disorder rules, slices and resumption,
//! and the state's refusals. The Python suite holds the result against
//! Polars' own sort and `merge_sorted` (tests/test_stream_merge.py).

use super::*;

pub(super) fn cfg(clock: &str) -> MergeConfig {
    MergeConfig {
        clock: clock.into(),
        source: "source".into(),
        by: None,
        restart: None,
        restart_text: None,
        carry: Vec::new(),
        skip_step_back: false,
    }
}

pub(super) fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("s{i}")).collect()
}

fn f64_input(t: &[f64], tag: i64) -> DataFrame {
    df!(
        "t" => t,
        "x" => t.iter().enumerate().map(|(i, _)| tag * 1000 + i as i64).collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Feed each input's chunks as the Python source does, pulling until done;
/// `limit` stops after that many rows, `hold` holds the rows at the end.
pub(super) fn drive(
    m: &mut Merge,
    chunks: &[Vec<DataFrame>],
    limit: Option<usize>,
    hold: bool,
) -> DataFrame {
    let mut next = vec![0usize; chunks.len()];
    let mut out = DataFrame::empty_with_schema(m.schema());
    loop {
        let left = limit.map(|l| l - out.height());
        let (df, want) = m.pull(left, hold).unwrap();
        out.vstack_mut(&df).unwrap();
        if limit.is_some_and(|l| out.height() >= l) {
            break;
        }
        let Some(i) = want else { break };
        match chunks[i].get(next[i]) {
            Some(c) => {
                m.push(i, c.clone()).unwrap();
                next[i] += 1;
            }
            None => m.end(i),
        }
    }
    out
}

/// Each input cut into chunks at the given row counts, cycled.
pub(super) fn chunked(df: &DataFrame, sizes: &[usize]) -> Vec<DataFrame> {
    let mut out = Vec::new();
    let mut at = 0;
    let mut k = 0;
    while at < df.height() {
        let n = sizes[k % sizes.len()].max(1).min(df.height() - at);
        out.push(df.slice(at as i64, n));
        at += n;
        k += 1;
    }
    out
}

pub(super) fn merge_of(inputs: &[DataFrame], c: MergeConfig) -> Merge {
    Merge::new(
        names(inputs.len()),
        inputs.iter().map(|d| d.schema().as_ref().clone()).collect(),
        c,
    )
    .unwrap()
}

/// The oracle: every row with its key (clock, input, row), stably sorted.
fn sorted_rows(inputs: &[DataFrame]) -> Vec<(String, i64)> {
    let mut rows: Vec<(f64, usize, usize, i64)> = Vec::new();
    for (i, d) in inputs.iter().enumerate() {
        let t = d.column("t").unwrap().f64().unwrap();
        let x = d.column("x").unwrap().i64().unwrap();
        for r in 0..d.height() {
            rows.push((t.get(r).unwrap(), i, r, x.get(r).unwrap()));
        }
    }
    rows.sort_by(|a, b| (a.0, a.1, a.2).partial_cmp(&(b.0, b.1, b.2)).unwrap());
    rows.into_iter()
        .map(|(_, i, _, x)| (format!("s{i}"), x))
        .collect()
}

fn rows_of(out: &DataFrame) -> Vec<(String, i64)> {
    let s = out.column("source").unwrap().str().unwrap();
    let x = out.column("x").unwrap().i64().unwrap();
    (0..out.height())
        .map(|r| (s.get(r).unwrap().to_owned(), x.get(r).unwrap()))
        .collect()
}

pub(super) fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// Random sorted clocks with ties within and across inputs.
fn random_inputs(seed: u64, n: usize, len: usize) -> Vec<DataFrame> {
    let mut r = lcg(seed);
    (0..n)
        .map(|i| {
            let m = (r() as usize) % len;
            let mut t = 0.0;
            let v: Vec<f64> = (0..m)
                .map(|_| {
                    t += f64::from((r() % 3) as u32);
                    t
                })
                .collect();
            f64_input(&v, i as i64)
        })
        .collect()
}

#[test]
fn the_merge_is_the_stable_sort_of_every_row_however_the_chunks_fall() {
    for seed in 0..40 {
        let n = 1 + (seed as usize % 7);
        let inputs = random_inputs(seed, n, 60);
        let want = sorted_rows(&inputs);
        let mut r = lcg(seed + 1000);
        for _ in 0..6 {
            let sizes: Vec<usize> = (0..5).map(|_| 1 + (r() as usize % 17)).collect();
            let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| chunked(d, &sizes)).collect();
            let mut m = merge_of(&inputs, cfg("t"));
            let out = drive(&mut m, &chunks, None, false);
            assert_eq!(rows_of(&out), want, "seed {seed}, chunks {sizes:?}");
        }
        // And every row in one chunk.
        let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| vec![d.clone()]).collect();
        let mut m = merge_of(&inputs, cfg("t"));
        assert_eq!(rows_of(&drive(&mut m, &chunks, None, false)), want);
    }
}

#[test]
fn a_tie_goes_to_the_earlier_input_then_to_each_inputs_own_order() {
    let a = f64_input(&[1.0, 2.0, 2.0], 0);
    let b = f64_input(&[0.0, 2.0, 2.0, 3.0], 1);
    let c = f64_input(&[2.0], 2);
    let inputs = [a, b, c];
    let mut m = merge_of(&inputs, cfg("t"));
    let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| chunked(d, &[1])).collect();
    let out = drive(&mut m, &chunks, None, false);
    let got: Vec<i64> = rows_of(&out).into_iter().map(|(_, x)| x).collect();
    assert_eq!(got, [1000, 0, 1, 2, 1001, 1002, 2000, 1003]);
}

#[test]
fn signed_zeros_tie_as_a_sort_ties_them() {
    let a = f64_input(&[0.0, 1.0], 0);
    let b = f64_input(&[-0.0, 1.0], 1);
    let mut m = merge_of(&[b.clone(), a.clone()], cfg("t"));
    let out = drive(&mut m, &[vec![b], vec![a]], None, false);
    assert_eq!(
        out.column("x").unwrap().i64().unwrap().to_vec(),
        [Some(1000), Some(0), Some(1001), Some(1)]
    );
    for x in [-3.5, -0.0, 0.0, 1e-300, 7.0, f64::MAX, -f64::MAX] {
        assert_eq!(float_of(float_key(x)), if x == 0.0 { 0.0 } else { x });
    }
    let xs = [
        -f64::MAX,
        -2.0,
        -1.0,
        -1e-300,
        0.0,
        1e-300,
        1.0,
        2.0,
        f64::MAX,
    ];
    assert!(xs.windows(2).all(|w| float_key(w[0]) < float_key(w[1])));
}

#[test]
fn columns_are_unified_by_name_and_a_missing_one_is_null() {
    let a = df!("t" => [1i64, 3], "a" => [1.0, 2.0]).unwrap();
    let b = df!("w" => [0.0, 1.0], "t" => [2i64, 4], "a" => [5.0, 6.0]).unwrap();
    let mut m = Merge::new(
        names(2),
        vec![a.schema().as_ref().clone(), b.schema().as_ref().clone()],
        cfg("t"),
    )
    .unwrap();
    let names: Vec<&str> = m.schema().iter_names().map(|n| n.as_str()).collect();
    assert_eq!(names, ["t", "a", "w", "source"]);
    let out = drive(&mut m, &[vec![a], vec![b]], None, false);
    assert_eq!(out.schema().as_ref(), m.schema());
    assert_eq!(
        out.column("w").unwrap().f64().unwrap().to_vec(),
        [None, Some(0.0), None, Some(1.0)]
    );
    assert_eq!(
        out.column("a").unwrap().f64().unwrap().to_vec(),
        [Some(1.0), Some(5.0), Some(2.0), Some(6.0)]
    );
}

#[test]
fn a_bad_call_is_refused_by_name() {
    let s = |d: &DataFrame| d.schema().as_ref().clone();
    let a = df!("t" => [1i64], "a" => [1.0]).unwrap();
    let ns = df!("t" => [1.0], "a" => [1.0]).unwrap();
    let a_int = df!("t" => [1i64], "a" => [1i64]).unwrap();
    let no_t = df!("u" => [1i64]).unwrap();
    let text = df!("t" => ["x"]).unwrap();
    let err = |r: PolarsResult<Merge>| r.err().expect("refused").to_string();
    let e = err(Merge::new(names(2), vec![s(&a), s(&ns)], cfg("t")));
    assert!(
        e.contains("the clocks are not of one kind") && e.contains("\"s1\""),
        "{e}"
    );
    assert!(is_wrong_type_text(Merge::new(
        names(2),
        vec![s(&a), s(&ns)],
        cfg("t")
    )));
    let e = err(Merge::new(names(2), vec![s(&a), s(&a_int)], cfg("t")));
    assert!(
        e.contains("column \"a\" is Float64 in input \"s0\" and Int64 in input \"s1\""),
        "{e}"
    );
    let r = Merge::new(names(2), vec![s(&a), s(&no_t)], cfg("t"));
    assert!(matches!(r, Err(PolarsError::ColumnNotFound(_))));
    let e = err(Merge::new(names(1), vec![s(&text)], cfg("t")));
    assert!(e.contains("which is no clock"), "{e}");
    let e = err(Merge::new(
        vec!["a".into(), "a".into()],
        vec![s(&a), s(&a)],
        cfg("t"),
    ));
    assert!(e.contains("given twice"), "{e}");
    let e = err(Merge::new(vec![], vec![], cfg("t")));
    assert!(e.contains("no input"), "{e}");
    let mut c = cfg("t");
    c.source = "a".into();
    let e = err(Merge::new(names(1), vec![s(&a)], c));
    assert!(e.contains("already has a column named \"a\""), "{e}");
    let mut c = cfg("t");
    c.by = Some("g".into());
    assert!(matches!(
        Merge::new(names(1), vec![s(&a)], c),
        Err(PolarsError::ColumnNotFound(_))
    ));
}

fn is_wrong_type_text(r: PolarsResult<Merge>) -> bool {
    r.err().is_some_and(|e| crate::is_wrong_type(&e))
}

#[test]
fn a_bad_clock_value_is_refused_naming_its_input_and_row() {
    let bad = |t: Vec<Option<f64>>, what: &str| {
        let d = df!("t" => t, "x" => [0i64, 1, 2]).unwrap();
        let mut m = merge_of(std::slice::from_ref(&d), cfg("t"));
        m.push(0, d.slice(0, 1)).unwrap();
        let e = m.push(0, d.slice(1, 2)).unwrap_err().to_string();
        assert!(e.contains(what) && e.contains("input \"s0\""), "{e}");
    };
    bad(vec![Some(0.0), Some(1.0), None], "is null at row 2");
    bad(
        vec![Some(0.0), Some(f64::NAN), Some(1.0)],
        "is NaN at row 1",
    );
    bad(
        vec![Some(0.0), Some(1.0), Some(f64::INFINITY)],
        "is infinite at row 2",
    );
    let d = df!("t" => [1u64, u64::MAX]).unwrap();
    let mut m = merge_of(std::slice::from_ref(&d), cfg("t"));
    let e = m.push(0, d).unwrap_err().to_string();
    assert!(
        e.contains("past the largest Int64") && e.contains("row 1"),
        "{e}"
    );
}

#[test]
fn a_step_back_is_refused_with_the_banks_rule() {
    let d = f64_input(&[0.0, 5.0, 4.5, 6.0], 0);
    let mut m = merge_of(std::slice::from_ref(&d), cfg("t"));
    m.push(0, d.slice(0, 2)).unwrap();
    let e = m.push(0, d.slice(2, 2)).unwrap_err().to_string();
    assert!(
        e.contains(
            "input \"s0\": clock column \"t\" steps back by 0.5 at row 2 (counting from 0), and \
             restart_after_step_back is not set"
        ),
        "{e}"
    );
    let mut c = cfg("t");
    c.restart = Some(Restart::Float(0.5));
    c.restart_text = Some("0.5".into());
    let mut m = merge_of(std::slice::from_ref(&d), c);
    let e = m.push(0, d.clone()).unwrap_err().to_string();
    assert!(
        e.contains("no more than restart_after_step_back = 0.5"),
        "{e}"
    );
    // Past the setting it starts the input again.
    let mut c = cfg("t");
    c.restart = Some(Restart::Float(0.4));
    c.restart_text = Some("0.4".into());
    let mut m = merge_of(std::slice::from_ref(&d), c);
    let out = drive(&mut m, &[vec![d]], None, false);
    assert_eq!(out.height(), 4);
}

#[test]
fn a_temporal_step_back_is_measured_exactly() {
    let t = Series::new("t".into(), [1_000_000_007i64, 1_000_000_000])
        .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
        .unwrap();
    let d = DataFrame::new(2, vec![t.into_column()]).unwrap();
    let mut m = merge_of(std::slice::from_ref(&d), cfg("t"));
    let e = m.push(0, d).unwrap_err().to_string();
    assert!(e.contains("steps back by 7ns at row 1"), "{e}");
    let t = Series::new("t".into(), [3i32, 1])
        .cast(&DataType::Date)
        .unwrap();
    let d = DataFrame::new(2, vec![t.into_column()]).unwrap();
    let mut c = cfg("t");
    c.restart = Some(Restart::Int(2 * 86_400 * 1_000_000_000));
    c.restart_text = Some("2d".into());
    let mut m = merge_of(std::slice::from_ref(&d), c);
    let e = m.push(0, d).unwrap_err().to_string();
    assert!(
        e.contains("steps back by 2d at row 1") && e.contains("restart_after_step_back = 2d"),
        "{e}"
    );
}

#[test]
fn an_input_started_again_merges_after_every_row_before_it() {
    // Two days of two inputs, each day's clock from 0.
    let a = f64_input(&[1.0, 5.0, 9.0, 0.5, 4.0], 0);
    let b = f64_input(&[2.0, 8.0, 1.0, 6.0], 1);
    let mut c = cfg("t");
    c.restart = Some(Restart::Float(1.0));
    c.restart_text = Some("1".into());
    let inputs = [a, b];
    for sizes in [vec![1], vec![2, 3], vec![10]] {
        let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| chunked(d, &sizes)).collect();
        let mut m = merge_of(&inputs, c.clone());
        let out = drive(&mut m, &chunks, None, false);
        let got: Vec<i64> = rows_of(&out).into_iter().map(|(_, x)| x).collect();
        assert_eq!(got, [0, 1000, 1, 1001, 2, 3, 1002, 4, 1003], "{sizes:?}");
    }
    // Under `by` a group whose own rows step back by no more than the
    // setting, where its input's clock stepped back past it, is caught in
    // the merged rows, as a bank grouped by it would refuse it.
    let d = df!("g" => ["A", "B", "A"], "t" => [10.0, 20.0, 9.5], "x" => [0i64, 1, 2]).unwrap();
    c.by = Some("g".into());
    let mut m = merge_of(std::slice::from_ref(&d), c);
    m.push(0, d).unwrap();
    m.end(0);
    let e = m.pull(None, false).unwrap_err().to_string();
    assert!(
        e.contains(
            "the merged rows of g = \"A\" step back by 0.5 at input \"s0\"'s row 2 (counting \
             from 0), no more than restart_after_step_back = 1"
        ),
        "{e}"
    );
}

#[test]
fn a_slice_stops_at_its_last_row_and_a_resumed_run_goes_on() {
    for seed in 0..25 {
        let n = 2 + (seed as usize % 4);
        let inputs = random_inputs(seed + 7, n, 30);
        let whole = {
            let mut m = merge_of(&inputs, cfg("t"));
            let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| chunked(d, &[3])).collect();
            drive(&mut m, &chunks, None, false)
        };
        let total = whole.height();
        let cut = (seed as usize * 7) % (total + 1);
        let mut m = merge_of(&inputs, cfg("t"));
        let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| chunked(d, &[2, 5])).collect();
        let first = drive(&mut m, &chunks, Some(cut), false);
        assert_eq!(first.height(), cut.min(total));
        let state = m.save_bytes().unwrap();
        // Each input resumes after its rows the slice returned.
        let src = first.column("source").unwrap().str().unwrap();
        let rest: Vec<DataFrame> = inputs
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let took = src.iter().filter(|s| *s == Some(&*format!("s{i}"))).count();
                d.slice(took as i64, d.height() - took)
            })
            .collect();
        let mut m = Merge::load_bytes(
            &state,
            names(n),
            inputs.iter().map(|d| d.schema().as_ref().clone()).collect(),
            cfg("t"),
        )
        .unwrap();
        let chunks: Vec<Vec<DataFrame>> = rest.iter().map(|d| chunked(d, &[4])).collect();
        let second = drive(&mut m, &chunks, None, false);
        let mut both = first.clone();
        both.vstack_mut(&second).unwrap();
        assert_eq!(rows_of(&both), rows_of(&whole), "seed {seed}, cut {cut}");
    }
}

#[test]
fn a_run_cut_inside_a_tie_holds_the_tie_and_two_runs_give_one() {
    // The cut at t <= 2 splits the tie at 2: b's second row at 2 is in the
    // next part, and comes before c's rows at 2, which the state holds.
    let a = f64_input(&[1.0, 2.0, 3.0], 0);
    let b = f64_input(&[2.0, 2.0, 4.0], 1);
    let c = f64_input(&[0.5, 2.0, 2.0, 5.0], 2);
    let inputs = [a, b, c];
    let mut one = merge_of(&inputs, cfg("t"));
    let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| vec![d.clone()]).collect();
    let whole = drive(&mut one, &chunks, None, false);
    let parts = [(2usize, 1usize), (1, 2), (3, 1)];
    let first: Vec<DataFrame> = inputs
        .iter()
        .zip(parts)
        .map(|(d, (k, _))| d.slice(0, k))
        .collect();
    let second: Vec<DataFrame> = inputs
        .iter()
        .zip(parts)
        .map(|(d, (k, _))| d.slice(k as i64, d.height() - k))
        .collect();
    for sizes in [vec![1], vec![2], vec![9]] {
        let mut m = merge_of(&inputs, cfg("t"));
        let ch = |p: &[DataFrame]| -> Vec<Vec<DataFrame>> {
            p.iter().map(|d| chunked(d, &sizes)).collect()
        };
        let out1 = drive(&mut m, &ch(&first), None, true);
        let state = m.save_bytes().unwrap();
        let mut m = Merge::load_bytes(
            &state,
            names(3),
            inputs.iter().map(|d| d.schema().as_ref().clone()).collect(),
            cfg("t"),
        )
        .unwrap();
        // A state loaded and saved again is the same bytes.
        assert_eq!(m.save_bytes().unwrap(), state);
        let out2 = drive(&mut m, &ch(&second), None, false);
        assert!(out1.height() < 5, "the tie at 2 is held");
        let mut both = out1;
        both.vstack_mut(&out2).unwrap();
        assert_eq!(rows_of(&both), rows_of(&whole), "{sizes:?}");
    }
}

#[test]
fn a_resumed_run_on_inputs_cut_at_different_clocks_is_refused() {
    let a = f64_input(&[1.0, 3.0], 0);
    let b = f64_input(&[2.0, 4.0, 5.0], 1);
    let inputs = [a, b];
    let mut m = merge_of(&inputs, cfg("t"));
    // a cut after 1.0, b after 5.0: b's 4.0 and 5.0 go out before a's 3.0.
    let _ = drive(
        &mut m,
        &[vec![inputs[0].slice(0, 1)], vec![inputs[1].clone()]],
        None,
        true,
    );
    let state = m.save_bytes().unwrap();
    let mut m = Merge::load_bytes(
        &state,
        names(2),
        inputs.iter().map(|d| d.schema().as_ref().clone()).collect(),
        cfg("t"),
    )
    .unwrap();
    m.push(0, inputs[0].slice(1, 1)).unwrap();
    m.end(0);
    m.end(1);
    let e = m.pull(None, false).unwrap_err().to_string();
    assert!(
        e.contains("the merged rows step back by 1 at input \"s0\"'s row 0")
            && e.contains("Cut every input at the same clock"),
        "{e}"
    );
}

#[test]
fn under_by_each_groups_merged_rows_are_checked_on_their_own() {
    let s = |g: &[&str], t: &[f64], tag: i64| {
        df!("g" => g, "t" => t, "x" => (0..t.len() as i64).map(|i| tag * 1000 + i).collect::<Vec<_>>())
            .unwrap()
    };
    let a = s(&["A", "A"], &[1.0, 3.0], 0);
    let b = s(&["B", "B", "B"], &[2.0, 4.0, 5.0], 1);
    let mut c = cfg("t");
    c.by = Some("g".into());
    let schemas = || vec![a.schema().as_ref().clone(), b.schema().as_ref().clone()];
    let mut m = Merge::new(names(2), schemas(), c.clone()).unwrap();
    let _ = drive(&mut m, &[vec![a.slice(0, 1)], vec![b.clone()]], None, false);
    let state = m.save_bytes().unwrap();
    let mut m = Merge::load_bytes(&state, names(2), schemas(), c.clone()).unwrap();
    // The groups do not meet, so a's 3.0 after b's 5.0 steps back in no
    // group: a bank grouped by g reads it in order.
    let out = drive(&mut m, &[vec![a.slice(1, 1)], vec![]], None, false);
    assert_eq!(out.height(), 1);
    // The null key is a group of its own.
    let n = df!("g" => [None::<&str>, Some("A")], "t" => [1.0, 2.0], "x" => [0i64, 1]).unwrap();
    let mut m = Merge::new(names(1), vec![n.schema().as_ref().clone()], c).unwrap();
    let out = drive(&mut m, &[vec![n]], None, true);
    assert_eq!(out.height(), 1);
    let file: rmpv::Value = rmpv::decode::read_value(&mut &m.save_bytes().unwrap()[..]).unwrap();
    let groups = file
        .as_map()
        .unwrap()
        .iter()
        .find(|(k, _)| k.as_str() == Some("groups"))
        .unwrap()
        .1
        .as_array()
        .unwrap()
        .len();
    assert_eq!(groups, 1, "the held row's group is not out yet");
}

#[test]
fn a_state_loads_only_into_the_merge_it_holds() {
    let a = f64_input(&[1.0, 2.0], 0);
    let b = f64_input(&[1.5], 1);
    let inputs = [a.clone(), b.clone()];
    let mut m = merge_of(&inputs, cfg("t"));
    let _ = drive(&mut m, &[vec![a.clone()], vec![b.clone()]], None, true);
    let state = m.save_bytes().unwrap();
    let schemas = || {
        inputs
            .iter()
            .map(|d| d.schema().as_ref().clone())
            .collect::<Vec<_>>()
    };
    let err = |r: PolarsResult<Merge>| r.err().expect("refused").to_string();
    let e = err(Merge::load_bytes(
        &state,
        vec!["s1".into(), "s0".into()],
        schemas(),
        cfg("t"),
    ));
    assert!(
        e.contains("saved merging the inputs [\"s0\", \"s1\"]"),
        "{e}"
    );
    let mut c = cfg("t");
    c.by = Some("x".into());
    let e = err(Merge::load_bytes(&state, names(2), schemas(), c));
    assert!(e.contains("by = None"), "{e}");
    let ints = [
        df!("t" => [1i64], "x" => [0i64]).unwrap(),
        df!("t" => [1i64], "x" => [0i64]).unwrap(),
    ];
    let e = err(Merge::load_bytes(
        &state,
        names(2),
        ints.iter().map(|d| d.schema().as_ref().clone()).collect(),
        cfg("t"),
    ));
    assert!(e.contains("saved on a clock of dtype Float64"), "{e}");
    let other = crate::RefreshTime::new(vec!["a".into(), "b".into()], false)
        .unwrap()
        .save_bytes()
        .unwrap();
    let e = err(Merge::load_bytes(&other, names(2), schemas(), cfg("t")));
    assert!(
        e.contains("a refresh_time run's state, not a merge's"),
        "{e}"
    );
    let mut broken = state.clone();
    let n = broken.len();
    broken[n - 12] ^= 1;
    let e = err(Merge::load_bytes(&broken, names(2), schemas(), cfg("t")));
    assert!(e.contains("damaged"), "{e}");
    let e = err(Merge::load_bytes(
        b"not a state",
        names(2),
        schemas(),
        cfg("t"),
    ));
    assert!(e.contains("not a rheon state"), "{e}");
    // Another kind's loader names it; a checkpoint carries it as a stage of
    // its kind, and gives it back to load.
    let e = crate::Bank::load_bytes(&state, None)
        .err()
        .expect("refused");
    assert!(e.contains("these bytes are a merge run's state"), "{e}");
    let checkpoint = crate::checkpoint_bytes(&[("merge", &state[..])]).unwrap();
    let e = err(Merge::load_bytes(
        &checkpoint,
        names(2),
        schemas(),
        cfg("t"),
    ));
    assert!(e.contains("load_checkpoint"), "{e}");
    let stages = crate::load_checkpoint(&checkpoint).unwrap();
    assert_eq!(stages[0].kind, crate::StateKind::Merge);
    assert!(Merge::load_bytes(&stages[0].state, names(2), schemas(), cfg("t")).is_ok());
}

#[test]
fn a_limit_of_zero_takes_nothing_and_an_empty_chunk_asks_again() {
    let a = f64_input(&[1.0, 2.0], 0);
    let mut m = merge_of(std::slice::from_ref(&a), cfg("t"));
    let (df, want) = m.pull(Some(0), false).unwrap();
    assert_eq!((df.height(), want), (0, Some(0)));
    m.push(0, a.slice(0, 0)).unwrap();
    assert_eq!(m.pull(None, false).unwrap().1, Some(0));
    m.push(0, a.clone()).unwrap();
    let (df, want) = m.pull(Some(1), false).unwrap();
    assert_eq!((df.height(), want), (1, None));
    let (df, want) = m.pull(None, false).unwrap();
    assert_eq!((df.height(), want), (1, Some(0)));
    m.end(0);
    let (df, want) = m.pull(None, false).unwrap();
    assert_eq!((df.height(), want), (0, None));
    assert!(m.push(0, a).is_err());
}

/// Inputs cut at one clock, any of a tie's rows on either side of the cut
/// for each input, and run in two parts that hold the tie between them, give
/// the one run's rows, however the chunks fall; ties long enough to fill
/// several chunks included.
#[test]
fn any_cut_at_one_clock_in_two_runs_gives_the_one_run() {
    for seed in 0..60 {
        let n = 1 + (seed as usize % 5);
        let inputs = random_inputs(seed + 99, n, 40);
        let mut one = merge_of(&inputs, cfg("t"));
        let whole = drive(
            &mut one,
            &inputs.iter().map(|d| vec![d.clone()]).collect::<Vec<_>>(),
            None,
            false,
        );
        let mut r = lcg(seed + 5);
        let cut = f64::from((r() % 30) as u32);
        // Each input's first part: its rows below the cut, and any number of
        // its rows at it.
        let firsts: Vec<usize> = inputs
            .iter()
            .map(|d| {
                let t = d.column("t").unwrap().f64().unwrap();
                let below = t.iter().filter(|v| v.unwrap() < cut).count();
                let at = t.iter().filter(|v| v.unwrap() == cut).count();
                below + (r() as usize) % (at + 1)
            })
            .collect();
        let sizes: Vec<usize> = (0..3).map(|_| 1 + (r() as usize % 4)).collect();
        let part = |first: bool| -> Vec<Vec<DataFrame>> {
            inputs
                .iter()
                .zip(&firsts)
                .map(|(d, &k)| {
                    let p = if first {
                        d.slice(0, k)
                    } else {
                        d.slice(k as i64, d.height() - k)
                    };
                    chunked(&p, &sizes)
                })
                .collect()
        };
        let mut m = merge_of(&inputs, cfg("t"));
        let out1 = drive(&mut m, &part(true), None, true);
        let state = m.save_bytes().unwrap();
        let mut m = Merge::load_bytes(
            &state,
            names(n),
            inputs.iter().map(|d| d.schema().as_ref().clone()).collect(),
            cfg("t"),
        )
        .unwrap();
        assert_eq!(m.save_bytes().unwrap(), state, "seed {seed}");
        let out2 = drive(&mut m, &part(false), None, false);
        let mut both = out1;
        both.vstack_mut(&out2).unwrap();
        assert_eq!(rows_of(&both), rows_of(&whole), "seed {seed}, cut {cut}");
    }
}

/// A tie longer than a chunk, in a run that holds: the merge reads on past
/// one chunk of an input to find the tie's end, and holds it whole.
#[test]
fn a_long_tie_is_read_on_and_held_whole() {
    let a = f64_input(&[1.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0);
    let b = f64_input(&[2.0, 2.0, 2.0], 1);
    let inputs = [a, b];
    let mut m = merge_of(&inputs, cfg("t"));
    let chunks: Vec<Vec<DataFrame>> = inputs.iter().map(|d| chunked(d, &[1])).collect();
    let out = drive(&mut m, &chunks, None, true);
    assert_eq!(out.height(), 1);
    let state = m.save_bytes().unwrap();
    let mut m = Merge::load_bytes(
        &state,
        names(2),
        inputs.iter().map(|d| d.schema().as_ref().clone()).collect(),
        cfg("t"),
    )
    .unwrap();
    let rest = drive(&mut m, &[vec![], vec![]], None, false);
    assert_eq!(
        rest.column("x").unwrap().i64().unwrap().to_vec(),
        [1, 2, 3, 4, 5, 1000, 1001, 1002].map(Some)
    );
}

/// Review 2026-11-23, MERGE-4 (1, 4): `Merge::new` refuses an unreadable `by`
/// dtype upfront as a wrong type, and dtype conflict messages print parameters
/// for `Decimal`, `Struct`, `Enum`, and `Array` rather than printing identical
/// bare names on both sides.
#[test]
fn merge_new_refuses_unreadable_by_dtype_and_names_compound_dtypes() {
    for bad_dtype in [
        DataType::List(Box::new(DataType::Int64)),
        DataType::Array(Box::new(DataType::Float64), 2),
        DataType::Binary,
        DataType::Struct(vec![Field::new("a".into(), DataType::Int64)]),
    ] {
        let mut s = Schema::default();
        s.insert("t".into(), DataType::Float64);
        s.insert("g".into(), bad_dtype.clone());
        let mut c = cfg("t");
        c.by = Some("g".into());
        let err = Merge::new(vec!["a".into()], vec![s], c)
            .err()
            .expect("unreadable by dtype must be refused at build");
        assert!(crate::is_wrong_type(&err), "{bad_dtype:?}: {err}");
        let msg = err.to_string();
        assert!(msg.contains("by column \"g\""), "{bad_dtype:?}: {msg}");
    }

    // Compound dtype conflicts must not print the same bare name on both sides.
    let pairs: Vec<(DataType, DataType, &str, &str)> = vec![
        (
            DataType::Decimal(10, 2),
            DataType::Decimal(10, 4),
            "Decimal(10, 2)",
            "Decimal(10, 4)",
        ),
        (
            DataType::Struct(vec![Field::new("a".into(), DataType::Int64)]),
            DataType::Struct(vec![Field::new("b".into(), DataType::Float64)]),
            "Struct([a: Int64])",
            "Struct([b: Float64])",
        ),
        (
            DataType::Array(Box::new(DataType::Float64), 2),
            DataType::Array(Box::new(DataType::Float64), 3),
            "Array(Float64, 2)",
            "Array(Float64, 3)",
        ),
    ];
    for (da, db, want_a, want_b) in pairs {
        let mut sa = Schema::default();
        sa.insert("t".into(), DataType::Float64);
        sa.insert("v".into(), da);
        let mut sb = Schema::default();
        sb.insert("t".into(), DataType::Float64);
        sb.insert("v".into(), db);
        let err = Merge::new(vec!["a".into(), "b".into()], vec![sa, sb], cfg("t"))
            .err()
            .expect("conflicting compound dtypes must be refused")
            .to_string();
        assert!(
            err.contains(want_a) && err.contains(want_b),
            "expected {want_a} and {want_b} in: {err}"
        );
    }
}

/// Review 2026-11-23, MERGE-2: `Restart::Int` compares exact nanoseconds past
/// 104.25 days (`2^53` ns) and exact integer steps past `2^53`.
#[test]
fn restart_int_is_exact_past_104_days_and_2_to_53() {
    let ns_105d: i128 = 105 * 86_400 * 1_000_000_000;
    let t0: i64 = 1_700_000_000_000_000_000;
    let mk_ns = |step: i128| {
        let s = Series::new("t".into(), [t0, t0 - step as i64])
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap();
        DataFrame::new(2, vec![s.into_column()]).unwrap()
    };
    let mut c = cfg("t");
    c.restart = Some(Restart::Int(ns_105d + 1));
    c.restart_text = Some("105d1ns".into());
    let d_eq = mk_ns(ns_105d + 1);
    let mut m = merge_of(std::slice::from_ref(&d_eq), c.clone());
    assert!(m.push(0, d_eq).is_err(), "exact boundary must be refused");
    let d_gt = mk_ns(ns_105d + 2);
    let mut m = merge_of(std::slice::from_ref(&d_gt), c);
    m.push(0, d_gt).unwrap();

    let big: i64 = (1i64 << 53) + 1;
    let di_eq = df!("t" => [big + 10, 10i64]).unwrap();
    let di_gt = df!("t" => [big + 10, 9i64]).unwrap();
    let mut ci = cfg("t");
    ci.restart = Some(Restart::Int(i128::from(big)));
    ci.restart_text = Some(big.to_string());
    let mut mi = merge_of(std::slice::from_ref(&di_eq), ci.clone());
    assert!(
        mi.push(0, di_eq).is_err(),
        "exact 2^53+1 boundary must be refused"
    );
    let mut mi = merge_of(std::slice::from_ref(&di_gt), ci);
    mi.push(0, di_gt).unwrap();
}
