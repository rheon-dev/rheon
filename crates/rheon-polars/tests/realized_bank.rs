//! `realized` through the bank (docs/PLAN.md task 499): its fields are the
//! core model's on the same rows, to the bit, under a decay, a window of
//! rows and a window of clock units at either edge, `lm_jump` a Boolean;
//! the chunking and a save change nothing; and a burst in a clock window
//! warns once and then refuses at `max_state_bytes`, naming the group and
//! the bytes.

use polars::prelude::*;
use rheon_core::{Decay, OnlineModel, Realized, RealizedCfg, WindowClosed};
use rheon_polars::{Bank, Spec};

const NAMES: [&str; 7] = ["rv", "bv", "tv", "tq", "bns", "lm", "lm_jump"];

fn spec(model: &str, extra: &str) -> Spec {
    let text = format!(
        r#"{{"name": "r", "model": {{"type": "realized"{model}}}, "targets": ["a", "b"],
            "features": [], "clock": "t", "group": "g", "gap_cap": 1e9, {extra}
            "min_weight": 0.0}}"#
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
/// and rows sit on a window's edge; two return columns, the second null now
/// and then, with a jump of 12 standard deviations in each.
fn frame(groups: &[(&str, usize)], seed: u64) -> DataFrame {
    let (mut g, mut t, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (name, n) in groups {
        let mut r = lcg(seed ^ name.len() as u64);
        let mut normal = || {
            let (u, v) = (r(), r());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        };
        let mut clock = 0.0;
        for i in 0..*n {
            clock += (normal().abs() * 3.0).floor() * 0.25;
            g.push(name.to_string());
            t.push(clock);
            let jump = if i == n / 2 { 0.012 } else { 0.0 };
            a.push(Some(0.001 * normal() + jump));
            b.push((i % 7 != 3).then(|| 0.002 * normal() - jump));
        }
    }
    df!("g" => g, "t" => t, "a" => a, "b" => b).unwrap()
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

/// A field as numbers, a flag read as 1 or 0.
fn field(out: &DataFrame, name: &str) -> Vec<Option<f64>> {
    let s = out
        .column("r")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap();
    if s.dtype() == &DataType::Boolean {
        return s.bool().unwrap().iter().map(|v| v.map(f64::from)).collect();
    }
    s.f64().unwrap().iter().collect()
}

fn bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

fn cfg(decay: f64) -> RealizedCfg {
    RealizedCfg {
        n_targets: 2,
        decay: Decay::Halflife(decay),
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        lm_window_rows: None,
        lm_confidence: 0.99,
        max_state_bytes: None,
    }
}

/// The bank's fields are the core model's on the same rows, to the bit:
/// under a decay with `lm_window_rows`, a window of rows, and a window of clock
/// units under either edge; `lm_jump` is a Boolean, null where `lm` is.
#[test]
fn the_banks_fields_are_the_models() {
    let df = frame(&[("a", 800)], 3);
    let cases: [(&str, &str, RealizedCfg); 4] = [
        (
            r#", "lm_window_rows": 50"#,
            r#""half_life": 60.0,"#,
            RealizedCfg {
                lm_window_rows: Some(50),
                ..cfg(60.0)
            },
        ),
        (
            r#", "window_rows": 120"#,
            r#""half_life": "inf","#,
            RealizedCfg {
                window_rows: Some(120),
                ..cfg(f64::INFINITY)
            },
        ),
        (
            r#", "window_size": 30.0"#,
            r#""half_life": "inf","#,
            RealizedCfg {
                window_clock: Some(30.0),
                ..cfg(f64::INFINITY)
            },
        ),
        (
            r#", "window_size": 30.0, "closed": "both", "lm_confidence": 0.9"#,
            r#""half_life": "inf","#,
            RealizedCfg {
                window_clock: Some(30.0),
                closed: WindowClosed::Both,
                lm_confidence: 0.9,
                ..cfg(f64::INFINITY)
            },
        ),
    ];
    for (model, extra, c) in cases {
        let out = run(&spec(model, extra), &df, 800);
        let mut m = Realized::new(c).unwrap();
        let t = df.column("t").unwrap().f64().unwrap();
        let (a, b) = (
            df.column("a").unwrap().f64().unwrap(),
            df.column("b").unwrap().f64().unwrap(),
        );
        let mut want: Vec<Vec<Option<f64>>> = vec![Vec::new(); 14];
        let mut weight = Vec::new();
        let mut last = 0.0;
        for i in 0..df.height() {
            let now = t.get(i).unwrap();
            let s = m.step(
                &[],
                &[a.get(i), b.get(i)],
                if i == 0 { 0.0 } else { now - last },
                1.0,
            );
            last = now;
            for (k, w) in want.iter_mut().enumerate() {
                w.push(Some(s.pred[k]).filter(|v| v.is_finite()));
            }
            weight.push(Some(s.n_eff));
        }
        for (j, target) in ["a", "b"].iter().enumerate() {
            for (k, name) in NAMES.iter().enumerate() {
                let f = format!("{name}_{target}");
                assert_eq!(
                    bits(&field(&out, &f)),
                    bits(&want[7 * j + k]),
                    "{model}: {f}"
                );
            }
        }
        assert_eq!(bits(&field(&out, "weight_sum")), bits(&weight), "{model}");
        let lm = field(&out, "lm_a");
        assert!(lm.iter().filter(|v| v.is_some()).count() > 700, "{model}");
        let flags = field(&out, "lm_jump_a");
        assert!(flags.contains(&Some(1.0)), "{model}: the jump is flagged");
        assert_eq!(
            flags.iter().map(Option::is_some).collect::<Vec<_>>(),
            lm.iter().map(Option::is_some).collect::<Vec<_>>(),
            "{model}: a flag where there is a statistic"
        );
    }
    let dtype = run(&spec("", r#""half_life": 60.0,"#), &df, 800)
        .column("r")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("lm_jump_a")
        .unwrap()
        .dtype()
        .clone();
    assert_eq!(dtype, DataType::Boolean);
}

/// Feeding a stream as one chunk or as many gives the same output, over two
/// groups whose rows interleave; and a save between two chunks goes on to
/// the bit, under every form.
#[test]
fn chunking_and_a_save_change_nothing() {
    let (x, y) = (frame(&[("a", 300)], 5), frame(&[("b", 300)], 9));
    let mut cols: Vec<Vec<AnyValue>> = vec![Vec::new(); 4];
    for i in 0..300 {
        for d in [&x, &y] {
            for (c, col) in cols.iter_mut().enumerate() {
                col.push(d.columns()[c].get(i).unwrap().into_static());
            }
        }
    }
    let df = DataFrame::new(
        600,
        cols.into_iter()
            .zip(["g", "t", "a", "b"])
            .map(|(v, n)| Series::from_any_values(n.into(), &v, false).unwrap().into())
            .collect(),
    )
    .unwrap();
    let names: Vec<String> = ["a", "b"]
        .iter()
        .flat_map(|t| NAMES.iter().map(move |n| format!("{n}_{t}")))
        .chain(["weight_sum".to_string()])
        .collect();
    for (model, extra) in [
        (r#", "lm_window_rows": 40"#, r#""half_life": 25.0,"#),
        (r#", "window_rows": 50"#, r#""half_life": "inf","#),
        (
            r#", "window_size": 12.5, "lm_window_rows": 20"#,
            r#""half_life": "inf","#,
        ),
    ] {
        let sp = spec(model, extra);
        let whole = run(&sp, &df, 600);
        for size in [1, 7, 64, 599] {
            let got = run(&sp, &df, size);
            for name in &names {
                assert_eq!(
                    bits(&field(&got, name)),
                    bits(&field(&whole, name)),
                    "{model} {name} {size}"
                );
            }
        }
        for cut in [1, 50, 211, 400] {
            let mut bank = Bank::new(vec![sp.clone()]).unwrap();
            bank.fit_predict(&df.slice(0, cut)).unwrap();
            let mut back = Bank::load_bytes(&bank.save_bytes().unwrap(), None).unwrap();
            let rest = df.slice(cut as i64, 600 - cut);
            let got = DataFrame::new(rest.height(), back.fit_predict(&rest).unwrap()).unwrap();
            for name in &names {
                let tail = field(&whole, name)[cut..].to_vec();
                assert_eq!(
                    bits(&field(&got, name)),
                    bits(&tail),
                    "{model} {name} {cut}"
                );
            }
        }
    }
}

/// A burst of rows in one group, over a span that holds them all, takes its
/// state past half of `max_state_bytes`: the group is named once, with the
/// bytes, and the chunk that takes it to the cap is refused naming the group
/// and the bytes. A group that stays under says nothing.
#[test]
fn a_burst_warns_once_and_is_refused_at_the_cap() {
    let row = Realized::window_row_bytes(2, true);
    let at = |rows: usize| Realized::state_bytes(2, None, Some(rows), true);
    let cap = at(150);
    let sp = spec(
        &format!(r#", "window_size": 1000.0, "max_state_bytes": {cap}"#),
        r#""half_life": "inf","#,
    );
    let burst = |name: &str, from: usize, n: usize| {
        let t: Vec<f64> = (from..from + n).map(|i| i as f64 * 0.01).collect();
        let r: Vec<f64> = (from..from + n)
            .map(|i| 0.001 * if i % 2 == 0 { 1.0 } else { -1.5 })
            .collect();
        df!("g" => vec![name; n], "t" => t, "a" => r.clone(), "b" => r).unwrap()
    };
    let mut bank = Bank::new(vec![sp]).unwrap();
    let call = burst("a", 0, 100).vstack(&burst("b", 0, 40)).unwrap();
    bank.fit_predict(&call).unwrap();
    let said = bank.take_notices();
    assert_eq!(said.len(), 1, "{said:#?}");
    let first = (1..).find(|&r| 2 * at(r) > cap).unwrap();
    assert!(
        said[0].starts_with(r#"spec "r" group "a": "#),
        "{}",
        said[0]
    );
    assert!(
        said[0].contains(&format!("holds {} bytes", at(first))),
        "{}",
        said[0]
    );
    assert!(
        said[0].contains(&format!("max_state_bytes ({cap})")),
        "{}",
        said[0]
    );
    assert!(said[0].contains("window_size = 1000"), "{}", said[0]);
    let err = bank
        .fit_predict(&burst("a", 100, 100))
        .unwrap_err()
        .to_string();
    assert!(err.contains(r#"spec "r" group "a""#), "{err}");
    assert!(err.contains(&format!("to {} bytes", cap + row)), "{err}");
    assert!(err.contains(&format!("max_state_bytes of {cap}")), "{err}");
    assert!(bank.take_notices().is_empty(), "the warning is said once");
}
