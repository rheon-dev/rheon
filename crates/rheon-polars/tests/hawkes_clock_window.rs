//! `hawkes` over a window of clock units (`window_size`; docs/PLAN.md tasks
//! 498 and 538): the bank's window is the core model's, the chunking and a
//! save change nothing, and a burst of events in one group warns once and
//! then refuses at `max_state_bytes`, naming the group and the bytes.

use polars::prelude::*;
use rheon_core::{Hawkes, HawkesCfg, OnlineModel, WindowClosed};
use rheon_polars::{Bank, Spec};

fn spec(window: &str, extra: &str) -> Spec {
    let text = format!(
        r#"{{"name": "h", "model": {{"type": "hawkes", "kernel_decays": [0.8, 4.0],
            {window}}}, "targets": ["e", "f"], "features": [], "clock": "t", "group": "g",
            "half_life": "inf", "gap_cap": 1e9, {extra} "min_weight": 0.0}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// One group's rows: a clock in quarters of a unit, so that sums are exact
/// and rows sit on a window's edge, and two event columns.
fn rows(group: &str, n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut r = lcg(seed ^ group.len() as u64);
    let mut t = 0.0;
    let (mut ts, mut e, mut f) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        t += (r() * 6.0).floor() * 0.25;
        ts.push(t);
        e.push(if r() < 0.5 { 1.0 } else { 0.0 });
        f.push(if r() < 0.35 { 1.0 } else { 0.0 });
    }
    (ts, e, f)
}

fn frame(groups: &[(&str, usize)], seed: u64) -> DataFrame {
    let (mut g, mut t, mut e, mut f) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (name, n) in groups {
        let (ts, es, fs) = rows(name, *n, seed);
        g.extend(std::iter::repeat_n(name.to_string(), ts.len()));
        t.extend(ts);
        e.extend(es);
        f.extend(fs);
    }
    df!("g" => g, "t" => t, "e" => e, "f" => f).unwrap()
}

fn run(spec: &Spec, df: &DataFrame, size: usize) -> DataFrame {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let mut out: Option<DataFrame> = None;
    let mut start = 0;
    while start < df.height() {
        let piece = df.slice(start as i64, size);
        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
        start += size;
    }
    out.unwrap()
}

fn field(out: &DataFrame, name: &str) -> Vec<Option<f64>> {
    out.column("h")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect()
}

fn bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

/// The bank's intensities over a window of clock units are the core model's
/// on the same rows, to the bit, at either edge; and `weight_sum` is the
/// weight inside the window.
#[test]
fn the_banks_window_is_the_models() {
    let df = frame(&[("a", 600)], 3);
    for closed in ["right", "both"] {
        let sp = spec(&format!(r#""window_size": 3.0, "closed": "{closed}""#), "");
        let out = run(&sp, &df, 600);
        let mut m = Hawkes::new(HawkesCfg {
            n_targets: 2,
            kernel_decays: vec![0.8, 4.0],
            decay: rheon_core::Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
            window_events: None,
            window_clock: Some(3.0),
            closed: if closed == "right" {
                WindowClosed::Right
            } else {
                WindowClosed::Both
            },
            max_state_bytes: None,
        })
        .unwrap();
        let t = df.column("t").unwrap().f64().unwrap();
        let (e, f) = (
            df.column("e").unwrap().f64().unwrap(),
            df.column("f").unwrap().f64().unwrap(),
        );
        let (mut want_e, mut want_f, mut want_w) = (Vec::new(), Vec::new(), Vec::new());
        let mut last = 0.0;
        for i in 0..df.height() {
            let now = t.get(i).unwrap();
            let step = m.step(
                &[],
                &[e.get(i), f.get(i)],
                if i == 0 { 0.0 } else { now - last },
                1.0,
            );
            last = now;
            want_e.push(Some(step.pred[0]).filter(|v| v.is_finite()));
            want_f.push(Some(step.pred[1]).filter(|v| v.is_finite()));
            want_w.push(Some(step.n_eff));
        }
        assert_eq!(bits(&field(&out, "intensity_e")), bits(&want_e), "{closed}");
        assert_eq!(bits(&field(&out, "intensity_f")), bits(&want_f), "{closed}");
        assert_eq!(bits(&field(&out, "weight_sum")), bits(&want_w), "{closed}");
        let fitted = want_e.iter().filter(|v| v.is_some()).count();
        assert!(fitted > 200, "{closed}: {fitted} rows fitted");
        // The edge matters on this stream: the two edges give other fits.
        if closed == "both" {
            let right = run(&spec(r#""window_size": 3.0"#, ""), &df, 600);
            assert_ne!(
                bits(&field(&right, "intensity_e")),
                bits(&field(&out, "intensity_e"))
            );
        }
    }
}

/// Feeding a stream as one chunk or as many gives the same output, over
/// groups whose rows interleave; and a save between two chunks goes on to
/// the bit.
#[test]
fn chunking_and_a_save_change_nothing() {
    let (a, b) = (frame(&[("a", 300)], 5), frame(&[("b", 300)], 9));
    // Interleave the two groups' rows, each group's clock its own.
    let mut cols: Vec<Vec<AnyValue>> = vec![Vec::new(); 4];
    for i in 0..300 {
        for d in [&a, &b] {
            for (c, col) in cols.iter_mut().enumerate() {
                col.push(d.columns()[c].get(i).unwrap().into_static());
            }
        }
    }
    let names = ["g", "t", "e", "f"];
    let df = DataFrame::new(
        600,
        cols.into_iter()
            .zip(names)
            .map(|(v, n)| Series::from_any_values(n.into(), &v, false).unwrap().into())
            .collect(),
    )
    .unwrap();
    let sp = spec(r#""window_size": 2.5"#, "");
    let whole = run(&sp, &df, 600);
    for size in [1, 7, 64, 599] {
        let got = run(&sp, &df, size);
        for name in ["intensity_e", "intensity_f", "weight_sum"] {
            assert_eq!(
                bits(&field(&got, name)),
                bits(&field(&whole, name)),
                "{name} {size}"
            );
        }
    }
    // A save and a load mid-window.
    for cut in [1, 50, 211, 400] {
        let mut bank = Bank::new(vec![sp.clone()]).unwrap();
        bank.fit_predict(&df.slice(0, cut)).unwrap();
        let mut back = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
        let rest = df.slice(cut as i64, 600 - cut);
        let x = DataFrame::new(rest.height(), bank.fit_predict(&rest).unwrap()).unwrap();
        let y = DataFrame::new(rest.height(), back.fit_predict(&rest).unwrap()).unwrap();
        for name in ["intensity_e", "intensity_f", "weight_sum"] {
            assert_eq!(
                bits(&field(&x, name)),
                bits(&field(&y, name)),
                "{name} {cut}"
            );
            let tail = field(&whole, name)[cut..].to_vec();
            assert_eq!(
                bits(&field(&x, name)),
                bits(&tail),
                "{name} {cut}: the whole run"
            );
        }
    }
}

/// A burst of events in one group, over a span that holds them all, takes
/// its state past half of `max_state_bytes`: the group is named once, with
/// the bytes, and the chunk that takes it to the cap is refused naming the
/// group and the bytes. A group that stays under says nothing.
#[test]
fn a_burst_warns_once_and_is_refused_at_the_cap() {
    // Two nodes, two decays: 150 rows fit under the cap.
    let cap = Hawkes::clock_state_bytes(2, 2, 150);
    let sp = spec(
        &format!(r#""window_size": 1000.0, "max_state_bytes": {cap}"#),
        "",
    );
    let burst = |name: &str, n: usize| {
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.01).collect();
        df!(
            "g" => vec![name; n],
            "t" => t,
            "e" => vec![1.0; n],
            "f" => vec![1.0; n]
        )
        .unwrap()
    };
    let mut bank = Bank::new(vec![sp]).unwrap();
    // "b" holds 40 rows, under half the cap; "a" 100, past it.
    let call = burst("a", 100).vstack(&burst("b", 40)).unwrap();
    bank.fit_predict(&call).unwrap();
    let said = bank.take_notices();
    assert_eq!(said.len(), 1, "{said:#?}");
    // The first row to take the state past half of the cap.
    let first = (1..)
        .find(|&r| 2 * Hawkes::clock_state_bytes(2, 2, r) > cap)
        .unwrap();
    let held = Hawkes::clock_state_bytes(2, 2, first);
    assert!(
        said[0].starts_with(r#"spec "h" group "a": "#),
        "{}",
        said[0]
    );
    assert!(
        said[0].contains(&format!("holds {held} bytes")),
        "{}",
        said[0]
    );
    assert!(
        said[0].contains(&format!("max_state_bytes ({cap})")),
        "{}",
        said[0]
    );
    // More of the same, past the cap: no second warning, and the refusal.
    let again = df!(
        "g" => vec!["a"; 100],
        "t" => (100..200).map(|i| f64::from(i) * 0.01).collect::<Vec<_>>(),
        "e" => vec![1.0; 100],
        "f" => vec![1.0; 100]
    )
    .unwrap();
    let err = bank.fit_predict(&again).unwrap_err().to_string();
    assert!(err.contains(r#"spec "h" group "a""#), "{err}");
    let one_row = Hawkes::clock_state_bytes(2, 2, 1) - Hawkes::clock_state_bytes(2, 2, 0);
    assert!(
        err.contains(&format!("to {} bytes", cap + one_row)),
        "{err}"
    );
    assert!(err.contains(&format!("max_state_bytes of {cap}")), "{err}");
    assert!(bank.take_notices().is_empty(), "the warning is said once");
    // The bank stopped after learning part of the chunk: it goes no further.
    let e = bank.fit_predict(&again).unwrap_err().to_string();
    assert!(e.contains("cannot go on"), "{e}");
}

/// On a temporal clock every span is a duration (`kernel_half_lives`,
/// `window_size`, `gap_cap`), read in seconds off the exact stamps: the
/// output is, to the bit, that of the same model on a numeric clock of the
/// same seconds, its decays `ln 2 / half-life` converted by hand. Steps of a
/// quarter second, so that a step's seconds are the same double either way;
/// and with irregular millisecond steps, where they differ in the last bit
/// of a step, within 1e-11.
#[test]
fn a_temporal_clock_takes_durations_as_a_numeric_one_takes_seconds() {
    let n = 400;
    let mut r = lcg(11);
    for quarter in [true, false] {
        let mut ms = 1_700_000_000_000i64;
        let mut t_ms = Vec::new();
        for _ in 0..n {
            ms += if quarter {
                250 * (r() * 6.0).floor() as i64
            } else {
                1 + (r() * 900.0) as i64
            };
            t_ms.push(ms);
        }
        let e: Vec<f64> = (0..n).map(|_| f64::from(r() < 0.5)).collect();
        let f: Vec<f64> = (0..n).map(|_| f64::from(r() < 0.35)).collect();
        let temporal = df!("t" => t_ms.clone(), "e" => e.clone(), "f" => f.clone())
            .unwrap()
            .lazy()
            .with_column(col("t").cast(DataType::Datetime(TimeUnit::Milliseconds, None)))
            .collect()
            .unwrap();
        let secs: Vec<f64> = t_ms
            .iter()
            .map(|m| (*m - t_ms[0]) as f64 / 1000.0)
            .collect();
        let numeric = df!("t" => secs, "e" => e, "f" => f).unwrap();
        let on_time = |model: &str, extra: &str| -> Spec {
            serde_json::from_str(&format!(
                r#"{{"name": "h", "model": {{"type": "hawkes", {model}}}, "targets": ["e", "f"],
                    "features": [], "clock": "t", "half_life": "inf", {extra}
                    "min_weight": 0.0}}"#
            ))
            .unwrap()
        };
        let (h1, h2) = (0.5f64, 2.0f64);
        let by_hand = format!(
            r#""kernel_decays": [{}, {}], "window_size": 10.0, "closed": "both""#,
            std::f64::consts::LN_2 / h1,
            std::f64::consts::LN_2 / h2
        );
        let spec_t = on_time(
            r#""kernel_half_lives": ["500ms", "2s"], "window_size": "10s", "closed": "both""#,
            r#""gap_cap": "1h","#,
        );
        let spec_n = on_time(&by_hand, r#""gap_cap": 3600.0,"#);
        let mut bank_t = Bank::new(vec![spec_t]).unwrap();
        let mut bank_n = Bank::new(vec![spec_n]).unwrap();
        let out_t = DataFrame::new(n, bank_t.fit_predict(&temporal).unwrap()).unwrap();
        let out_n = DataFrame::new(n, bank_n.fit_predict(&numeric).unwrap()).unwrap();
        let mut fitted = 0;
        for name in ["intensity_e", "intensity_f", "weight_sum"] {
            let (a, b) = (field(&out_t, name), field(&out_n, name));
            if quarter {
                assert_eq!(bits(&a), bits(&b), "{name}");
            } else {
                for (x, y) in a.iter().zip(&b) {
                    match (x, y) {
                        (Some(x), Some(y)) => assert!(
                            (x - y).abs() <= 1e-11 * (1.0 + y.abs()),
                            "{name}: {x} against {y}"
                        ),
                        (None, None) => {}
                        other => panic!("{name}: {other:?}"),
                    }
                }
            }
            fitted += a.iter().flatten().count();
        }
        assert!(fitted > 300, "{fitted} values");
    }
}
