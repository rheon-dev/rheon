//! `expectile` in the bank (docs/PLAN.md tasks 501, 537, 538 and 539): rows
//! of weight 0 count in no clock on every clock the bank reads, so the
//! weighted rows' outputs are those of the stream without them to the bit;
//! a state the settings size is refused at the call past `max_state_bytes`;
//! and a window of clock units warns once at half its cap and is refused at
//! the cap, naming the group and the bytes.

use polars::prelude::*;
use rheon_core::Expectile;
use rheon_polars::{Bank, Spec};

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn spec(model: &str, clock: &str, half_life: &str) -> Spec {
    let text = format!(
        r#"{{"name": "e", "model": {{"type": "expectile", "expectile": 0.8{model}}},
            "targets": ["y"], "features": ["x0", "x1"], {clock} "weight": "w",
            "half_life": {half_life}, "min_weight": 0.0}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn field(out: &DataFrame, name: &str) -> Vec<Option<f64>> {
    out.column("e")
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

/// The bank's rows of weight 0 count in no clock (hard rule 9): on a
/// temporal clock (the gap read once from the stamps), a float clock and
/// the row clock (where such a row is no step), under decay, the first
/// rows' refit, a window of rows and one of a span of the clock, each
/// weighted row's `pred`, `resid` and `weight_sum` are those of the same
/// stream with the rows of weight 0 left out, to the bit, in one chunk or
/// many. The stream opens with rows of weight 0 too.
#[test]
fn rows_of_weight_zero_leave_the_weighted_rows_as_they_were() {
    let n = 600;
    let mut r = lcg(501);
    let mut ms = 1_700_000_000_000i64;
    let (mut t, mut x0, mut x1, mut y, mut w) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..n {
        ms += 250 * (1 + (r() * 8.0) as i64);
        t.push(ms);
        let (a, b) = (r() * 2.0 - 1.0, r() * 2.0 - 1.0);
        x0.push(a);
        x1.push(b);
        y.push(1.0 + 2.0 * a - b + (-r().ln() - 1.0));
        w.push(if i < 3 || i % 7 == 3 || (300..320).contains(&i) {
            0.0
        } else {
            0.5 + r()
        });
    }
    let secs: Vec<f64> = t.iter().map(|m| (*m - t[0]) as f64 / 1000.0).collect();
    let base = df!("t" => t, "s" => secs, "x0" => x0, "x1" => x1, "y" => y, "w" => w.clone())
        .unwrap()
        .lazy()
        .with_column(col("t").cast(DataType::Datetime(TimeUnit::Milliseconds, None)))
        .collect()
        .unwrap();
    let weighted = base
        .clone()
        .lazy()
        .filter(col("w").gt(lit(0.0)))
        .collect()
        .unwrap();
    let kept: Vec<usize> = (0..n).filter(|&i| w[i] > 0.0).collect();
    let models = [
        ("decay", ""),
        ("refit", r#", "refit_rows": 60"#),
        ("window_rows", r#", "window_rows": 40"#),
        (
            "window_size",
            r#", "window_size": WINDOW, "closed": "both""#,
        ),
    ];
    let clocks = [
        (
            "temporal",
            r#""clock": "t", "gap_cap": "1h","#,
            r#""30s""#,
            r#""20s""#,
        ),
        (
            "float",
            r#""clock": "s", "gap_cap": 3600.0,"#,
            "30.0",
            "20.0",
        ),
        ("rows", "", "30.0", "40.0"),
    ];
    for (name, model) in models {
        for (clock_name, clock, half_life, window) in clocks {
            let windowed = name.starts_with("window");
            let model = model.replace("WINDOW", window);
            let sp = spec(&model, clock, if windowed { r#""inf""# } else { half_life });
            for size in [n, 37] {
                let run = |df: &DataFrame| {
                    let mut bank = Bank::new(vec![sp.clone()]).unwrap();
                    let mut out: Option<DataFrame> = None;
                    let mut start = 0;
                    while start < df.height() {
                        let piece = df.slice(start as i64, size);
                        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap())
                            .unwrap();
                        match out.as_mut() {
                            None => out = Some(got),
                            Some(o) => {
                                o.vstack_mut(&got).unwrap();
                            }
                        }
                        start += size;
                    }
                    out.unwrap()
                };
                let (with, without) = (run(&base), run(&weighted));
                for f in ["pred_y", "resid_y", "weight_sum"] {
                    let all = field(&with, f);
                    let picked: Vec<Option<f64>> = kept.iter().map(|&i| all[i]).collect();
                    assert_eq!(
                        bits(&picked),
                        bits(&field(&without, f)),
                        "{name} on the {clock_name} clock in chunks of {size}: {f}"
                    );
                }
                let fitted = field(&without, "pred_y").iter().flatten().count();
                assert!(fitted > 400, "{name} {clock_name}: {fitted} rows fitted");
                // A row of weight 0 is read as any row is.
                let zero_read = field(&with, "pred_y")[310];
                assert!(zero_read.is_some(), "{name} {clock_name}: row 310 not read");
            }
        }
    }
}

/// A state the settings size past `max_state_bytes` is refused as the bank
/// is built, naming the bytes: a refit buffer, a window of rows, and a cap
/// above the 64 MiB a group may set.
#[test]
fn a_state_past_its_cap_is_refused_at_the_call() {
    let refused = |model: &str, half_life: &str, words: &[&str]| {
        let e = match Bank::new(vec![spec(model, "", half_life)]) {
            Ok(_) => panic!("{model}: built"),
            Err(e) => e.to_string(),
        };
        for w in words {
            assert!(e.contains(w), "{model}: {e}");
        }
        assert!(
            e.starts_with(r#"spec "e""#) || e.contains(r#"spec "e""#),
            "{e}"
        );
    };
    refused(
        r#", "window_rows": 100000"#,
        r#""inf""#,
        &[
            "bytes a group",
            "past max_state_bytes of 262144",
            "window_rows",
        ],
    );
    refused(
        r#", "refit_rows": 100000"#,
        "30.0",
        &["past max_state_bytes of 262144", "refit_rows"],
    );
    refused(
        r#", "max_state_bytes": 67108865"#,
        "30.0",
        &["at most 67108864 (64 MiB)"],
    );
    refused(r#", "window_rows": 40"#, "30.0", &["takes no decay"]);
    // At its cap's edge: a window that fits is built.
    let fits = r#", "window_rows": 40, "max_state_bytes": 100000"#;
    assert!(Bank::new(vec![spec(fits, "", r#""inf""#)]).is_ok());
}

/// A burst of rows in one group, over a span that holds them all, takes its
/// state past half of `max_state_bytes`: the group is named once, with the
/// bytes, and the chunk that takes it to the cap is refused naming the group
/// and the bytes. A group that stays under says nothing.
#[test]
fn a_burst_in_a_clock_window_warns_once_and_is_refused_at_the_cap() {
    let cap = Expectile::clock_state_bytes(2, 1, true, 150);
    let text = format!(
        r#"{{"name": "e", "model": {{"type": "expectile", "expectile": 0.8,
            "window_size": 1000.0, "max_state_bytes": {cap}}}, "targets": ["y"],
            "features": ["x0", "x1"], "clock": "t", "gap_cap": 1e9, "group": "g", "half_life": "inf",
            "min_weight": 0.0}}"#
    );
    let sp: Spec = serde_json::from_str(&text).unwrap();
    let burst = |name: &str, from: usize, n: usize| {
        let mut r = lcg(from as u64 + name.len() as u64);
        let t: Vec<f64> = (from..from + n).map(|i| i as f64 * 0.01).collect();
        let x0: Vec<f64> = (0..n).map(|_| r()).collect();
        let x1: Vec<f64> = (0..n).map(|_| r()).collect();
        let y: Vec<f64> = (0..n).map(|i| x0[i] - x1[i] + r()).collect();
        df!("g" => vec![name; n], "t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
    };
    let mut bank = Bank::new(vec![sp]).unwrap();
    let call = burst("a", 0, 100).vstack(&burst("b", 0, 40)).unwrap();
    bank.fit_predict(&call).unwrap();
    let said = bank.take_notices();
    assert_eq!(said.len(), 1, "{said:#?}");
    let first = (1..)
        .find(|&r| 2 * Expectile::clock_state_bytes(2, 1, true, r) > cap)
        .unwrap();
    let held = Expectile::clock_state_bytes(2, 1, true, first);
    assert!(
        said[0].starts_with(r#"spec "e" group "a": "#),
        "{}",
        said[0]
    );
    assert!(
        said[0].contains(&format!("holds {held} bytes")),
        "{}",
        said[0]
    );
    assert!(
        said[0].contains("the rows inside window_size"),
        "{}",
        said[0]
    );
    let err = bank
        .fit_predict(&burst("a", 100, 100))
        .unwrap_err()
        .to_string();
    assert!(err.contains(r#"spec "e" group "a""#), "{err}");
    let one_row =
        Expectile::clock_state_bytes(2, 1, true, 1) - Expectile::clock_state_bytes(2, 1, true, 0);
    assert!(
        err.contains(&format!("to {} bytes", cap + one_row)),
        "{err}"
    );
    assert!(err.contains(&format!("max_state_bytes of {cap}")), "{err}");
    assert!(err.contains("the rows inside window_size"), "{err}");
    assert!(bank.take_notices().is_empty(), "the warning is said once");
}
