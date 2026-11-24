//! The row operators of task 564 through a window run: `run_length`,
//! `clock_lag`, `clock_drift` and `clock_step`, each computed a row at a
//! time from its group's previous rows. Each is held to its definition,
//! computed here a second way (the Python suite holds them to Polars' `rle`
//! and `ewm_*`, and statsmodels' `WLS` and `DescrStatsW`), at every
//! chunking, and across a save at every row.
use polars::prelude::*;
use rheon_polars::{WindowsConfig, WindowsRun};
use serde_json::json;

const NS: i64 = 1_000_000_000;

/// A stream of two groups, each with a value that holds for stretches, a
/// null and a NaN among them, and an event and an arrival time whose lag
/// grows; a gap past `gap_cap` and a session change on the way.
fn stream() -> DataFrame {
    let n = 240;
    let mut ts = Vec::new();
    let mut g = Vec::new();
    let mut s = Vec::new();
    let mut x: Vec<Option<f64>> = Vec::new();
    let mut ev: Vec<Option<i64>> = Vec::new();
    let mut tt: Vec<Option<f64>> = Vec::new();
    let mut t = 1_700_000_000 * NS;
    for i in 0..n {
        // A gap of 400 s past the cap of 300 at row 100; uneven steps.
        t += if i == 100 {
            400 * NS
        } else {
            (1 + (i % 3) as i64) * NS / 2
        };
        ts.push(t);
        g.push(if i % 4 == 0 { "b" } else { "a" });
        s.push(if i < 170 { 1i64 } else { 2 });
        x.push(match i % 23 {
            5 | 6 => None,
            7 => Some(f64::NAN),
            _ => Some(f64::from((i / 6) as u32 % 3)),
        });
        // The lag grows by 2 ms a second, with a wobble; one row has no
        // event time.
        let lag = 50_000_000 + (i as i64) * 1_000_000 + ((i * 7) % 5) as i64 * 3_000_000;
        ev.push((i != 33).then_some(t - lag));
        // A time of its own in seconds that repeats (a step of 0), steps
        // back now and then, and is missing twice, as a null and a NaN.
        tt.push(match i {
            _ if i % 29 == 3 => None,
            _ if i % 31 == 8 => Some(f64::NAN),
            _ => Some(f64::from((i / 3) as u32) - if i % 13 == 5 { 5.0 } else { 0.0 }),
        });
    }
    df!(
        "ts" => ts,
        "g" => g,
        "s" => s,
        "x" => x,
        "ev" => ev,
        "tt" => tt,
    )
    .unwrap()
    .lazy()
    .with_columns([
        col("ts").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)),
        col("ev").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)),
    ])
    .collect()
    .unwrap()
}

fn config(group: bool, session: bool) -> WindowsConfig {
    let mut c = json!({
        "formulas": [
            {"name": "n", "tree": ["run_length", ["col", "x"]]},
            {"name": "n_tol", "tree": ["run_length", ["col", "x"], {"tol": 1.0}]},
            {"name": "lag", "tree": ["clock_lag", ["col", "ts"], {"event": ["col", "ev"], "half_life": "20s"}]},
            {"name": "drift", "tree": ["clock_drift", ["col", "ts"], {"event": ["col", "ev"], "half_life": "20s"}]},
            {"name": "flat", "tree": ["clock_drift", ["col", "ts"], {"event": ["col", "ev"], "half_life": "inf"}]},
            {"name": "m", "tree": ["ewm_mean", ["col", "x"], {"half_life": "5s"}]},
            {"name": "dt", "tree": ["clock_step", ["col", "ts"], {"half_life": "20s", "stat": "mean"}]},
            {"name": "dt_sd", "tree": ["clock_step", ["col", "ts"], {"half_life": "20s", "stat": "std", "bias": false}]},
            {"name": "tt_var", "tree": ["clock_step", ["col", "tt"], {"half_life": "inf", "stat": "var", "bias": true}]},
            {"name": "zeros", "tree": ["clock_step", ["col", "tt"], {"half_life": "20s", "stat": "zeros"}]},
            {"name": "backs", "tree": ["clock_step", ["col", "tt"], {"half_life": "20s", "stat": "backs"}]},
        ],
        "clock": "ts",
        "gap_cap": "300s",
    });
    if group {
        c["group"] = json!("g");
    }
    if session {
        c["session"] = json!("s");
    }
    serde_json::from_value(c).unwrap()
}

fn run(cfg: &WindowsConfig, df: &DataFrame, chunk: usize) -> DataFrame {
    let mut r = WindowsRun::new(cfg.clone(), df.schema()).unwrap();
    let mut out = DataFrame::empty();
    let mut at = 0;
    while at < df.height() {
        let part = r.feed(&df.slice(at as i64, chunk), None).unwrap();
        out = if out.width() == 0 {
            part
        } else {
            out.vstack(&part).unwrap()
        };
        at += chunk;
    }
    let tail = r.finish().unwrap();
    if out.width() == 0 {
        tail
    } else {
        out.vstack(&tail).unwrap()
    }
}

fn floats(df: &DataFrame, c: &str) -> Vec<Option<f64>> {
    df.column(c).unwrap().f64().unwrap().iter().collect()
}

/// Bits, so that two NaNs and two nulls compare equal and nothing else does.
fn bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

/// Per row, its group's key and whether the row starts its group anew: a
/// new session, or, for a drift (`gap`), a step past the cap.
fn restarts(df: &DataFrame, group: bool, session: bool, gap: bool) -> Vec<(String, bool)> {
    let ts: Vec<i64> = df
        .column("ts")
        .unwrap()
        .datetime()
        .unwrap()
        .physical()
        .into_no_null_iter()
        .collect();
    let g: Vec<&str> = df
        .column("g")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(Option::unwrap)
        .collect();
    let s: Vec<i64> = df
        .column("s")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let mut last: std::collections::HashMap<String, (i64, i64)> = Default::default();
    (0..df.height())
        .map(|i| {
            let key = if group {
                g[i].to_string()
            } else {
                String::new()
            };
            let new = match last.get(&key) {
                None => true,
                Some(&(t0, s0)) => (session && s0 != s[i]) || (gap && ts[i] - t0 > 300 * NS),
            };
            last.insert(key.clone(), (ts[i], s[i]));
            (key, new)
        })
        .collect()
}

/// The run length from its definition: each row against the row before it
/// in its group, a missing value (null, NaN) one value of its own.
fn run_length_brute(df: &DataFrame, tol: f64, group: bool, session: bool) -> Vec<Option<f64>> {
    let x = floats(df, "x");
    let starts = restarts(df, group, session, false);
    let mut prev: std::collections::HashMap<String, (Option<f64>, f64)> = Default::default();
    x.iter()
        .zip(starts)
        .map(|(&v, (key, new))| {
            let v = v.filter(|f| f.is_finite());
            let n = match prev.get(&key) {
                Some(&(p, n)) if !new => {
                    let same = match (p, v) {
                        (None, None) => true,
                        (Some(a), Some(b)) => (a - b).abs() <= tol,
                        _ => false,
                    };
                    if same { n + 1.0 } else { 1.0 }
                }
                _ => 1.0,
            };
            prev.insert(key, (v, n));
            Some(n)
        })
        .collect()
}

/// The lag and the slope from their definition: per group, over the rows
/// of the stretch so far with both times, each weighing
/// `2^(-(t_now - t_i)/h)`, the weighted mean and the weighted
/// least-squares slope, in two passes about the weighted means.
fn drift_brute(
    df: &DataFrame,
    h: f64,
    group: bool,
    session: bool,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let ts: Vec<i64> = df
        .column("ts")
        .unwrap()
        .datetime()
        .unwrap()
        .physical()
        .into_no_null_iter()
        .collect();
    let ev: Vec<Option<i64>> = df
        .column("ev")
        .unwrap()
        .datetime()
        .unwrap()
        .physical()
        .iter()
        .collect();
    let starts = restarts(df, group, session, true);
    let mut rows: std::collections::HashMap<String, Vec<(i64, i64)>> = Default::default();
    let (mut lags, mut slopes) = (Vec::new(), Vec::new());
    for i in 0..df.height() {
        let (key, new) = &starts[i];
        let mine = rows.entry(key.clone()).or_default();
        if *new {
            mine.clear();
        }
        if let Some(e) = ev[i] {
            mine.push((ts[i], e));
        }
        let now = ts[i];
        let w: Vec<f64> = mine
            .iter()
            .map(|&(t, _)| {
                if h.is_infinite() {
                    1.0
                } else {
                    (-((now - t) as f64 / 1e9) / h).exp2()
                }
            })
            .collect();
        let origin = mine.first().map_or(0, |r| r.0);
        let a: Vec<f64> = mine
            .iter()
            .map(|&(t, _)| (t - origin) as f64 / 1e9)
            .collect();
        let d: Vec<f64> = mine.iter().map(|&(t, e)| (t - e) as f64 / 1e9).collect();
        let sw: f64 = w.iter().sum();
        if sw == 0.0 {
            lags.push(None);
            slopes.push(None);
            continue;
        }
        let ma = w.iter().zip(&a).map(|(w, a)| w * a).sum::<f64>() / sw;
        let md = w.iter().zip(&d).map(|(w, d)| w * d).sum::<f64>() / sw;
        let caa: f64 = (0..a.len()).map(|j| w[j] * (a[j] - ma).powi(2)).sum();
        let cad: f64 = (0..a.len()).map(|j| w[j] * (a[j] - ma) * (d[j] - md)).sum();
        lags.push(Some(md));
        slopes.push((caa > 0.0).then(|| cad / caa));
    }
    (lags, slopes)
}

/// A clock step's statistic from its definition: per group, the steps of
/// `times` (seconds) between each time and the one before it since the
/// chain last started (a session, a gap past the cap, or a missing time),
/// each weighing `2^(-(t_now - t_i)/h)` on the clock `ts`; a row with a
/// missing time reports none.
#[allow(clippy::too_many_arguments)]
fn steps_brute(
    df: &DataFrame,
    times: &[Option<f64>],
    h: f64,
    stat: &str,
    bias: bool,
    group: bool,
    session: bool,
) -> Vec<Option<f64>> {
    let ts: Vec<i64> = df
        .column("ts")
        .unwrap()
        .datetime()
        .unwrap()
        .physical()
        .into_no_null_iter()
        .collect();
    let starts = restarts(df, group, session, true);
    // Per group: the steps so far, with their clock, and the last time.
    type Chain = (Vec<(i64, f64)>, Option<f64>);
    let mut groups: std::collections::HashMap<String, Chain> = Default::default();
    let mut out = Vec::new();
    for i in 0..df.height() {
        let (key, new) = &starts[i];
        let (steps, prev) = groups.entry(key.clone()).or_default();
        if *new {
            steps.clear();
            *prev = None;
        }
        let Some(t) = times[i].filter(|t| t.is_finite()) else {
            *prev = None;
            out.push(None);
            continue;
        };
        if let Some(p) = *prev {
            steps.push((ts[i], t - p));
        }
        *prev = Some(t);
        let now = ts[i];
        let w: Vec<f64> = steps
            .iter()
            .map(|&(c, _)| {
                if h.is_infinite() {
                    1.0
                } else {
                    (-((now - c) as f64 / 1e9) / h).exp2()
                }
            })
            .collect();
        let v: Vec<f64> = steps.iter().map(|s| s.1).collect();
        let count = |keep: &dyn Fn(f64) -> bool| -> f64 {
            v.iter()
                .zip(&w)
                .filter(|(s, _)| keep(**s))
                .map(|p| p.1)
                .sum()
        };
        let sw: f64 = w.iter().sum();
        let m = v.iter().zip(&w).map(|(s, w)| s * w).sum::<f64>() / sw;
        let var = || {
            let b = v
                .iter()
                .zip(&w)
                .map(|(s, w)| w * (s - m).powi(2))
                .sum::<f64>()
                / sw;
            let r2 = w.iter().map(|w| w * w).sum::<f64>() / (sw * sw);
            if bias {
                Some(b)
            } else {
                (1.0 - r2 > 64.0 * f64::EPSILON).then(|| b / (1.0 - r2))
            }
        };
        out.push(match stat {
            "zeros" => Some(count(&|s| s == 0.0)),
            "backs" => Some(count(&|s| s < 0.0)),
            _ if sw == 0.0 => None,
            "mean" => Some(m),
            "var" => var(),
            "std" => var().map(f64::sqrt),
            _ => unreachable!("{stat}"),
        });
    }
    out
}

fn close(got: &[Option<f64>], want: &[Option<f64>], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match (g, w) {
            (Some(g), Some(w)) => assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "{what}, row {i}: {g} against {w}"
            ),
            (None, None) => {}
            _ => panic!("{what}, row {i}: {g:?} against {w:?}"),
        }
    }
}

/// Each operator is its definition on every row, grouped or not, with
/// sessions or not: a run restarts at a session and not at a gap, a
/// drift's sums at both; a null and a NaN are one missing run; a row
/// without its event time learns nothing.
#[test]
fn each_row_operator_is_its_definition() {
    let df = stream();
    for (group, session) in [(false, false), (true, false), (false, true), (true, true)] {
        let out = run(&config(group, session), &df, df.height());
        assert_eq!(out.height(), df.height());
        let what = format!("group {group}, session {session}");
        close(
            &floats(&out, "n"),
            &run_length_brute(&df, 0.0, group, session),
            0.0,
            &what,
        );
        close(
            &floats(&out, "n_tol"),
            &run_length_brute(&df, 1.0, group, session),
            0.0,
            &what,
        );
        let (lag, slope) = drift_brute(&df, 20.0, group, session);
        close(&floats(&out, "lag"), &lag, 1e-9, &what);
        close(&floats(&out, "drift"), &slope, 1e-6, &what);
        let (_, flat) = drift_brute(&df, f64::INFINITY, group, session);
        close(&floats(&out, "flat"), &flat, 1e-6, &what);
        let ts: Vec<Option<f64>> = df
            .column("ts")
            .unwrap()
            .datetime()
            .unwrap()
            .physical()
            .iter()
            .map(|t| t.map(|t| t as f64 / 1e9))
            .collect();
        let tt = floats(&df, "tt");
        let steps = |times: &[Option<f64>], h: f64, stat: &str, bias: bool| {
            steps_brute(&df, times, h, stat, bias, group, session)
        };
        close(
            &floats(&out, "dt"),
            &steps(&ts, 20.0, "mean", false),
            1e-9,
            &what,
        );
        close(
            &floats(&out, "dt_sd"),
            &steps(&ts, 20.0, "std", false),
            1e-6,
            &what,
        );
        let inf = f64::INFINITY;
        close(
            &floats(&out, "tt_var"),
            &steps(&tt, inf, "var", true),
            1e-9,
            &what,
        );
        close(
            &floats(&out, "zeros"),
            &steps(&tt, 20.0, "zeros", false),
            1e-12,
            &what,
        );
        close(
            &floats(&out, "backs"),
            &steps(&tt, 20.0, "backs", false),
            1e-12,
            &what,
        );
    }
    // The lag grows 1 ms a row, about a ms each 0.83 s, so the slope over
    // the long stretch reads near 1.2e-3.
    let out = run(&config(false, false), &df, df.height());
    let flat = floats(&out, "flat");
    let last = flat.last().unwrap().unwrap();
    assert!((last - 1.2e-3).abs() < 3e-4, "{last}");
}

/// Chunks of 1, 2, 3, 7, 50 and the whole stream give one chunk's bits.
#[test]
fn chunking_changes_no_bit() {
    let df = stream();
    for (group, session) in [(false, true), (true, true)] {
        let cfg = config(group, session);
        let whole = run(&cfg, &df, df.height());
        for chunk in [1, 2, 3, 7, 50, 1000] {
            let part = run(&cfg, &df, chunk);
            for c in [
                "n", "n_tol", "lag", "drift", "flat", "m", "dt", "dt_sd", "tt_var", "zeros",
                "backs",
            ] {
                assert_eq!(
                    bits(&floats(&part, c)),
                    bits(&floats(&whole, c)),
                    "{c} at chunks of {chunk}, group {group}"
                );
            }
        }
    }
}

/// A run saved after any row and loaded gives the run that never stopped,
/// to the bit.
#[test]
fn a_save_at_every_row_is_one_run() {
    let df = stream().slice(0, 120);
    let cfg = config(true, true);
    let whole = run(&cfg, &df, df.height());
    for cut in 1..df.height() {
        let mut a = WindowsRun::new(cfg.clone(), df.schema()).unwrap();
        let mut out = a.feed(&df.slice(0, cut), None).unwrap();
        let bytes = a.save_bytes().unwrap();
        let mut b = WindowsRun::load_bytes(&bytes, cfg.clone(), df.schema()).unwrap();
        out.vstack_mut(
            &b.feed(&df.slice(cut as i64, df.height() - cut), None)
                .unwrap(),
        )
        .unwrap();
        out.vstack_mut(&b.finish().unwrap()).unwrap();
        for c in [
            "n", "n_tol", "lag", "drift", "flat", "dt", "dt_sd", "tt_var", "zeros", "backs",
        ] {
            assert_eq!(
                bits(&floats(&out, c)),
                bits(&floats(&whole, c)),
                "{c}, saved at {cut}"
            );
        }
    }
}

/// A state with no row operator but increments keeps its bytes: no `rows`
/// is written where every row operator is an increment.
#[test]
fn a_state_of_increments_writes_no_row_state() {
    let df = stream();
    let cfg: WindowsConfig = serde_json::from_value(json!({
        "formulas": [{"name": "d", "tree": ["increment", ["col", "x"]]}],
        "clock": "ts", "gap_cap": "300s",
    }))
    .unwrap();
    let mut r = WindowsRun::new(cfg, df.schema()).unwrap();
    r.feed(&df, None).unwrap();
    let bytes = r.save_bytes().unwrap();
    assert!(
        !bytes.windows(4).any(|w| w == b"rows"),
        "no row state is written"
    );
    let mut r = WindowsRun::new(config(false, false), df.schema()).unwrap();
    r.feed(&df, None).unwrap();
    let bytes = r.save_bytes().unwrap();
    assert!(bytes.windows(4).any(|w| w == b"rows"), "a run length's is");
}

/// Integers are compared whole: two neighbors past 2^53 are two values,
/// which a column of doubles would make one.
#[test]
fn a_run_of_integers_is_compared_whole() {
    let big = 1i64 << 60;
    let df = df!("x" => [big, big, big + 1, big + 1, big + 1]).unwrap();
    let cfg: WindowsConfig = serde_json::from_value(json!({
        "formulas": [
            {"name": "n", "tree": ["run_length", ["col", "x"]]},
            {"name": "f", "tree": ["run_length", ["cast", ["col", "x"], "Float64"]]},
            {"name": "t", "tree": ["run_length", ["col", "x"], {"tol": 1.0}]},
        ],
    }))
    .unwrap();
    let out = run(&cfg, &df, 2);
    assert_eq!(floats(&out, "n"), [1.0, 2.0, 1.0, 2.0, 3.0].map(Some));
    assert_eq!(floats(&out, "f"), [1.0, 2.0, 3.0, 4.0, 5.0].map(Some));
    assert_eq!(floats(&out, "t"), [1.0, 2.0, 3.0, 4.0, 5.0].map(Some));
}

/// A temporal value's `tol` is in seconds, compared in nanoseconds: both
/// sides of it.
#[test]
fn a_temporal_runs_tol_is_in_seconds() {
    let df = df!("x" => [0i64, NS, 2 * NS + 1, 2 * NS + 1])
        .unwrap()
        .lazy()
        .with_column(col("x").cast(DataType::Datetime(TimeUnit::Nanoseconds, None)))
        .collect()
        .unwrap();
    let cfg: WindowsConfig = serde_json::from_value(json!({
        "formulas": [{"name": "n", "tree": ["run_length", ["col", "x"], {"tol": 1.0}]}],
    }))
    .unwrap();
    assert_eq!(
        floats(&run(&cfg, &df, 4), "n"),
        [1.0, 2.0, 1.0, 2.0].map(Some)
    );
}

/// What the plan refuses, by name: a drift over one temporal time and one
/// number, a run length of text, an operator inside a row operator's input,
/// and a row operator inside a window operator's.
#[test]
fn the_plan_refuses_what_it_cannot_read() {
    let df = stream()
        .lazy()
        .with_columns([lit(1.5).alias("num"), lit("a").alias("txt")])
        .collect()
        .unwrap();
    let refused = |tree: serde_json::Value| -> String {
        let cfg: Result<WindowsConfig, _> = serde_json::from_value(json!({
            "formulas": [{"name": "o", "tree": tree}], "clock": "ts", "gap_cap": "300s",
        }));
        match cfg {
            Err(e) => e.to_string(),
            Ok(cfg) => WindowsRun::new(cfg, df.schema())
                .err()
                .expect("refused")
                .to_string(),
        }
    };
    let e = refused(
        json!(["clock_drift", ["col", "ts"], {"event": ["col", "num"], "half_life": "5s"}]),
    );
    assert!(
        e.contains("must both be temporal or both be numbers"),
        "{e}"
    );
    let e = refused(json!(["run_length", ["col", "txt"]]));
    assert!(
        e.contains(
            "the run length's input must be a number or a temporal column, but it is String"
        ),
        "{e}"
    );
    let e = refused(json!(["run_length", ["increment", ["col", "num"]]]));
    assert!(e.contains("its input holds the operator increment"), "{e}");
    let e = refused(json!(["ewm_mean", ["run_length", ["col", "num"]], {"half_life": "5s"}]));
    assert!(e.contains("its input holds a window operator"), "{e}");
    let e = refused(json!(["clock_lag", ["col", "ts"], {"half_life": "5s"}]));
    assert!(e.contains("event is not set"), "{e}");
    let e = refused(json!(["clock_lag", ["col", "ts"], {"event": ["col", "ev"]}]));
    assert!(e.contains("half_life is not set"), "{e}");
    let e = refused(json!(["run_length", ["col", "num"], {"tol": -1.0}]));
    assert!(
        e.contains("tol must be a finite number of at least 0"),
        "{e}"
    );
    let e = refused(json!(["run_length", ["col", "num"], {"window_size": 3}]));
    assert!(
        e.contains("\"window_size\" is not one of its parameters"),
        "{e}"
    );
    let e = refused(json!(["clock_step", ["col", "txt"], {"half_life": "5s"}]));
    assert!(
        e.contains("the clock step's time must be a number or a temporal column"),
        "{e}"
    );
    let e = refused(json!(["clock_step", ["col", "ts"], {"stat": "mean"}]));
    assert!(e.contains("half_life is not set"), "{e}");
    let e = refused(json!(["clock_step", ["col", "ts"], {"half_life": "5s", "stat": "median"}]));
    assert!(e.contains("stat must be \"mean\""), "{e}");
    let e = refused(json!(["clock_step", ["col", "ts"], {"half_life": "5s", "bias": true}]));
    assert!(e.contains("only stat \"var\" or \"std\" takes it"), "{e}");
    let e = refused(json!(["clock_step", ["col", "ts"], {"half_life": 0.0}]));
    assert!(e.contains("half_life must be above 0"), "{e}");
    let e = refused(json!(["clock_step", ["increment", ["col", "num"]], {"half_life": "5s"}]));
    assert!(e.contains("its input holds the operator increment"), "{e}");
}

/// A running clock step emits each row on that row, never held for a later
/// one: fed a row at a time, each feed returns its own row, nothing is held
/// after it, and the value is that row's own step folded in (it differs from
/// the row before's at every row). On both sides of the gap: a step of
/// exactly `gap_cap` (10) is not past it and joins the chain; a step of
/// 10.5 is past it, so the chain starts again and that row has no step. A
/// chunk of 1, 2, 3 or the whole stream gives the same bits.
#[test]
fn a_running_clock_step_emits_each_row_on_that_row_at_both_sides_of_a_gap() {
    let t = [0.0, 1.0, 3.0, 13.0, 23.5, 24.5, 26.5];
    let df = df!("t" => t).unwrap();
    let cfg: WindowsConfig = serde_json::from_value(json!({
        "formulas": [{"name": "m", "tree": ["clock_step", ["col", "t"], {"half_life": "inf", "stat": "mean"}]}],
        "clock": "t", "gap_cap": 10.0,
    }))
    .unwrap();
    // Steps 1, 2, 10 (at the cap: kept), then 10.5 (past it: the chain
    // starts again with no step), 1, 2.
    let want = [
        None,
        Some(1.0),
        Some(1.5),
        Some(13.0 / 3.0),
        None,
        Some(1.0),
        Some(1.5),
    ];
    let mut r = WindowsRun::new(cfg.clone(), df.schema()).unwrap();
    for (i, w) in want.iter().enumerate() {
        let out = r.feed(&df.slice(i as i64, 1), None).unwrap();
        assert_eq!(out.height(), 1, "row {i} is emitted on its own feed");
        assert_eq!(r.held(), 0, "row {i} is held");
        assert_eq!(floats(&out, "m"), [*w], "row {i}");
    }
    assert_eq!(r.finish().unwrap().height(), 0, "nothing is left to finish");
    let whole = run(&cfg, &df, df.height());
    assert_eq!(floats(&whole, "m"), want);
    for chunk in [1, 2, 3] {
        assert_eq!(
            bits(&floats(&run(&cfg, &df, chunk), "m")),
            bits(&floats(&whole, "m")),
            "chunks of {chunk}"
        );
    }
}

/// A clock step on both sides of each threshold: a step of exactly 0 and
/// the smallest step either side of it on an integer time, so a zero and a
/// back are counted by sign alone; a null time reports none and starts the
/// chain again; and on the row-count clock every row decays the steps by
/// one unit.
#[test]
fn a_clock_step_counts_by_sign_and_restarts_its_chain_at_a_missing_time() {
    let df = df!("t" => [Some(5i64), Some(5), Some(6), Some(5), None, Some(1), Some(1)]).unwrap();
    let cfg: WindowsConfig = serde_json::from_value(json!({
        "formulas": [
            {"name": "z", "tree": ["clock_step", ["col", "t"], {"half_life": "inf", "stat": "zeros"}]},
            {"name": "b", "tree": ["clock_step", ["col", "t"], {"half_life": "inf", "stat": "backs"}]},
            {"name": "m", "tree": ["clock_step", ["col", "t"], {"half_life": 1.0, "stat": "mean"}]},
        ],
    }))
    .unwrap();
    let out = run(&cfg, &df, 3);
    // Steps 0, +1, -1, then none across the null, and 0.
    assert_eq!(
        floats(&out, "z"),
        [
            Some(0.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            None,
            Some(1.0),
            Some(2.0)
        ]
    );
    assert_eq!(
        floats(&out, "b"),
        [
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(1.0),
            None,
            Some(1.0),
            Some(1.0)
        ]
    );
    // Each row halves the weights before it: 0; (0/2 + 1)/1.5; ...; the
    // row after the null takes no step and reports the decayed mean.
    let m = floats(&out, "m");
    assert_eq!(m[0], None);
    assert_eq!(m[1], Some(0.0));
    close(
        &m[2..4],
        &[Some(1.0 / 1.5), Some((0.5 - 1.0) / 1.75)],
        1e-15,
        "means",
    );
    assert_eq!(m[4], None);
    assert_eq!(m[5], m[3]);
}
