//! `kalman`'s fixed-lag smoother through the bank: `smooth_lag` builds it,
//! its fields carry what the model emitted (`smoothed_coef`,
//! `smoothed_se_coef`, each target's `smoothed_for`), the predictions are
//! the filter's to the bit, the output does not move with the chunking or
//! a save and load, the `min_weight` gate withholds the value describing a
//! row it withheld, the embargo places a released row's value on the row
//! that released it, and the settings are refused where they cannot hold.

use polars::prelude::*;
use rheon_polars::rheon_core::{ClockValue, Kalman, OnlineModel};
use rheon_polars::{AnyModel, Bank, Spec, build_models, output_index};

fn spec(model_rest: &str, rest: &str) -> Spec {
    let rest = if rest.contains("min_weight") {
        rest.to_string()
    } else {
        format!(r#", "min_weight": 0.0{rest}"#)
    };
    let rest = if rest.contains("clock") {
        format!(r#"{rest}, "gap_cap": 10"#)
    } else {
        rest
    };
    let text = format!(
        r#"{{"name": "m", "model": {{"type": "kalman", "coef_half_life": 40{model_rest}}},
            "targets": ["y", "z"], "features": ["x0", "x1"], "half_life": 200,
            "weight": "w"{rest}}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// SplitMix64 in `[0, 1)`.
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` rows: two features, a drifting truth for `y` and `z`, a weight of 0
/// on about one row in ten and each target null on about one in ten, and
/// a clock `t` one unit a row with gaps of 2 and 3.
fn frame(n: usize, seed: u64) -> DataFrame {
    let mut r = rng(seed);
    let (mut b0, mut b1) = (1.0, -0.5);
    let mut t = 0.0;
    let mut cols: [Vec<Option<f64>>; 6] = Default::default();
    for _ in 0..n {
        b0 += 0.02 * (r() - 0.5);
        b1 += 0.02 * (r() - 0.5);
        t += [1.0, 1.0, 2.0, 1.0, 3.0][(r() * 5.0) as usize];
        let (x0, x1) = (r() * 2.0 - 1.0, r() * 3.0);
        let w = if r() < 0.1 { 0.0 } else { 0.5 + r() };
        let y = (r() > 0.1).then(|| b0 * x0 + b1 * x1 + 0.3 * (r() - 0.5));
        let z = (r() > 0.1).then(|| 2.0 - b1 * x0 + 0.3 * (r() - 0.5));
        for (c, v) in cols
            .iter_mut()
            .zip([Some(t), Some(x0), Some(x1), Some(w), y, z])
        {
            c.push(v);
        }
    }
    let [t, x0, x1, w, y, z] = cols;
    df!("t" => t, "x0" => x0, "x1" => x1, "w" => w, "y" => y, "z" => z).unwrap()
}

fn feed(bank: &mut Bank, df: &DataFrame, size: usize) -> DataFrame {
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

fn run(spec: &Spec, df: &DataFrame, size: usize) -> DataFrame {
    feed(&mut Bank::new(vec![spec.clone()]).unwrap(), df, size)
}

fn field(out: &DataFrame, name: &str) -> Series {
    out.column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
}

fn lists(out: &DataFrame, name: &str) -> Vec<Option<Vec<f64>>> {
    let col = field(out, name);
    let list = col.list().unwrap();
    list.amortized_iter()
        .map(|v| v.map(|s| s.as_ref().f64().unwrap().into_no_null_iter().collect()))
        .collect()
}

fn bits(v: &[Option<Vec<f64>>]) -> Vec<Option<Vec<u64>>> {
    v.iter()
        .map(|o| o.as_ref().map(|l| l.iter().map(|x| x.to_bits()).collect()))
        .collect()
}

/// The fields come after `coef` and its standard errors, each target's
/// `smoothed_for` in the clock column's type; none without `smooth_lag`.
#[test]
fn the_fields_are_laid_out_after_coef() {
    let names = |s: &Spec| {
        output_index(s)
            .into_iter()
            .map(|f| f.field)
            .collect::<Vec<_>>()
    };
    let plain = names(&spec("", ""));
    assert!(!plain.iter().any(|n| n.starts_with("smoothed")));
    let s = spec(
        r#", "smooth_lag": 5, "emit_smoothed_se": true"#,
        r#", "emit_se_coef": true"#,
    );
    let got = names(&s);
    let at = got.iter().position(|n| n == "coef").unwrap();
    assert_eq!(
        got[at + 1..at + 6],
        [
            "se_coef",
            "smoothed_coef",
            "smoothed_se_coef",
            "smoothed_for_y",
            "smoothed_for_z"
        ]
    );
    let idx = output_index(&s);
    let f = idx.iter().find(|f| f.field == "smoothed_for_z").unwrap();
    assert_eq!(
        (f.dtype.as_str(), f.kind.as_str()),
        ("clock", "smoothed_for")
    );
    assert_eq!(f.target.as_deref(), Some("z"));
}

/// Every prediction and coefficient is the filter's to the bit with the
/// smoother on; the smoothed fields are the same in chunks of 1, 7, 333
/// and the whole frame; each row's value describes the target's
/// observation `L` before it, by the row's clock.
#[test]
fn the_smoother_rides_beside_the_filter_whatever_the_chunks() {
    let df = frame(1000, 496);
    let lag = 6;
    let on = spec(
        &format!(r#", "smooth_lag": {lag}, "emit_smoothed_se": true"#),
        r#", "clock": "t""#,
    );
    let off = spec("", r#", "clock": "t""#);
    let whole = run(&on, &df, 1000);
    let plain = run(&off, &df, 1000);
    for name in ["pred_y", "pred_z", "resid_y", "weight_sum"] {
        assert!(
            field(&whole, name).equals_missing(&field(&plain, name)),
            "{name}"
        );
    }
    assert_eq!(bits(&lists(&whole, "coef")), bits(&lists(&plain, "coef")));
    for size in [1, 7, 333] {
        let got = run(&on, &df, size);
        for name in ["smoothed_coef", "smoothed_se_coef"] {
            assert_eq!(
                bits(&lists(&got, name)),
                bits(&lists(&whole, name)),
                "{name} {size}"
            );
        }
        for t in ["y", "z"] {
            let name = format!("smoothed_for_{t}");
            assert!(
                field(&got, &name).equals_missing(&field(&whole, &name)),
                "{name} {size}"
            );
        }
    }
    // Each target's observations: rows of positive weight where it is present.
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let clock: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    for t in ["y", "z"] {
        let y = df.column(t).unwrap().f64().unwrap();
        let obs: Vec<usize> = (0..df.height())
            .filter(|&i| w[i] > 0.0 && y.get(i).is_some())
            .collect();
        let described = field(&whole, &format!("smoothed_for_{t}"));
        let described = described.f64().unwrap();
        for (n, &row) in obs.iter().enumerate() {
            let want = n.checked_sub(lag).map(|s| clock[obs[s]]);
            assert_eq!(described.get(row), want, "{t} row {row}");
        }
        let emitting: usize = described.iter().flatten().count();
        assert_eq!(emitting, obs.len() - lag);
    }
}

/// The bank's values are the model's, stepped by hand on the same rows
/// with the same labels (the row's index with no clock column).
#[test]
fn the_fields_are_what_the_model_emitted() {
    let df = frame(300, 4961);
    let s = spec(r#", "smooth_lag": 4, "emit_smoothed_se": true"#, "");
    let out = run(&s, &df, 300);
    let AnyModel::Kalman(model) = build_models(&s).unwrap().remove(0).1 else {
        panic!("a kalman")
    };
    let mut model: Kalman = *model;
    let col =
        |n: &str| -> Vec<Option<f64>> { df.column(n).unwrap().f64().unwrap().iter().collect() };
    let (x0, x1, w, y, z) = (col("x0"), col("x1"), col("w"), col("y"), col("z"));
    let got = lists(&out, "smoothed_coef");
    let got_se = lists(&out, "smoothed_se_coef");
    let mut emitted = 0;
    // A row of weight 0 the bank only scores, and with no clock column it
    // counts in no clock: the next row is one step on.
    let mut first = true;
    for i in 0..df.height() {
        if w[i] == Some(0.0) {
            assert!(got[i].is_none());
            continue;
        }
        let d = if first { 0.0 } else { 1.0 };
        first = false;
        model.label_next(ClockValue::F64(i as f64));
        model.step(
            &[x0[i].unwrap(), x1[i].unwrap()],
            &[y[i], z[i]],
            d,
            w[i].unwrap(),
        );
        let s = model.smoothed().unwrap();
        if !s.emitted.iter().any(|e| *e) {
            assert!(got[i].is_none(), "row {i}");
            continue;
        }
        emitted += 1;
        let row = got[i].as_ref().unwrap();
        let se = got_se[i].as_ref().unwrap();
        for (j, e) in s.emitted.iter().enumerate() {
            for c in 0..3 {
                let (want, at) = (s.coef[j * 3 + c], j * 3 + c);
                if *e {
                    assert_eq!(row[at].to_bits(), want.to_bits(), "row {i}");
                    let v = s.var[at];
                    assert!(v.is_nan() && se[at].is_nan() || se[at] == v.max(0.0).sqrt());
                } else {
                    assert!(row[at].is_nan());
                }
            }
        }
    }
    assert!(emitted > 200);
}

/// A bank saved mid-stream and loaded goes on to the same fields, to the
/// bit.
#[test]
fn a_bank_saved_mid_ring_goes_on_to_the_bit() {
    let df = frame(600, 4962);
    let s = spec(
        r#", "smooth_lag": 20, "emit_smoothed_se": true"#,
        r#", "clock": "t""#,
    );
    let whole = run(&s, &df, 600);
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    let first = feed(&mut bank, &df.slice(0, 257), 257);
    let bytes = bank.save_bytes().unwrap();
    let mut loaded = Bank::load_bytes(&bytes, Some(std::slice::from_ref(&s))).unwrap();
    let mut joined = first;
    joined
        .vstack_mut(&feed(&mut loaded, &df.slice(257, 343), 343))
        .unwrap();
    for name in ["smoothed_coef", "smoothed_se_coef"] {
        assert_eq!(
            bits(&lists(&joined, name)),
            bits(&lists(&whole, name)),
            "{name}"
        );
    }
    assert!(field(&joined, "smoothed_for_y").equals_missing(&field(&whole, "smoothed_for_y")));
}

/// `min_weight` withholds the early rows, and the smoothed values that
/// describe them are null in the list when their lags complete.
#[test]
fn the_min_weight_gate_withholds_the_value_describing_a_row_it_withheld() {
    let df = frame(200, 4963);
    let s = spec(
        r#", "smooth_lag": 3"#,
        r#", "clock": "t", "min_weight": 15.0"#,
    );
    let out = run(&s, &df, 200);
    let pred = field(&out, "pred_y");
    let described = field(&out, "smoothed_for_y");
    let clock = df.column("t").unwrap().f64().unwrap();
    let got = lists(&out, "smoothed_coef");
    let (mut held, mut stood) = (0, 0);
    for (i, d) in described.f64().unwrap().iter().enumerate() {
        let Some(d) = d else { continue };
        let row = (0..df.height()).find(|&r| clock.get(r) == Some(d)).unwrap();
        let withheld = pred.f64().unwrap().get(row).is_none();
        let ys = &got[i].as_ref().unwrap()[..3];
        assert_eq!(
            ys.iter().all(|v| v.is_nan()),
            withheld,
            "row {i} describes {row}"
        );
        if withheld {
            held += 1;
        } else {
            stood += 1;
        }
    }
    assert!(held > 0 && stood > 0, "{held} held, {stood} stood");
}

/// A held row is learned on the row whose clock releases it, and one row
/// can release several whose lags it completes, with one line for them:
/// `smooth_lag` is refused beside `embargo` and `label_clock`, naming each,
/// and the same spec without them is accepted and runs.
#[test]
fn the_smoother_is_refused_beside_the_embargo() {
    let mut s = spec(r#", "smooth_lag": 4"#, r#", "clock": "t", "embargo": 3"#);
    let e = s.check().unwrap_err();
    assert!(e.contains("smooth_lag cannot be used with embargo"), "{e}");
    let mut s = spec(
        r#", "smooth_lag": 4"#,
        r#", "clock": "t", "label_clock": "t""#,
    );
    let e = s.check().unwrap_err();
    assert!(
        e.contains("smooth_lag cannot be used with label_clock"),
        "{e}"
    );
    let mut held = spec("", r#", "clock": "t", "embargo": 3"#);
    held.check().unwrap();
    let mut s = spec(r#", "smooth_lag": 4"#, r#", "clock": "t""#);
    s.check().unwrap();
    let out = run(&s, &frame(100, 4964), 100);
    assert!(
        field(&out, "smoothed_for_y")
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .count()
            > 50
    );
}

/// The settings are refused where they cannot hold, each saying why.
#[test]
fn the_smoothers_settings_are_refused_where_they_cannot_hold() {
    let refused = |model_rest: &str| {
        let mut s = spec(model_rest, "");
        s.check().unwrap_err()
    };
    let e = refused(r#", "smooth_lag": 0"#);
    assert!(e.contains("smooth_lag must be at least 1"), "{e}");
    let e = refused(r#", "emit_smoothed_se": true"#);
    assert!(
        e.contains("emit_smoothed_se is set, but smooth_lag is not"),
        "{e}"
    );
    let e = refused(r#", "max_state_bytes": 1000"#);
    assert!(
        e.contains("max_state_bytes is set, but smooth_lag is not"),
        "{e}"
    );
    let e = refused(r#", "smooth_lag": 3, "covariance_form": "square_root""#);
    assert!(e.contains("needs covariance_form=\"covariance\""), "{e}");
    let e = refused(r#", "smooth_lag": 3, "max_state_bytes": 0"#);
    assert!(e.contains("max_state_bytes must be above 0"), "{e}");
    // Two targets, three coefficients, the variances: about 1.2 KB a slot.
    let big = r#", "smooth_lag": 500, "emit_smoothed_se": true"#;
    let e = refused(big);
    let bytes = spec(big, "").smoother_bytes();
    assert!(bytes > 256 * 1024, "{bytes}");
    assert!(
        e.contains(&format!(
            "smooth_lag = 500 holds up to {bytes} bytes a group"
        )),
        "{e}"
    );
    assert!(e.contains("past max_state_bytes of 262144"), "{e}");
    let mut raised = spec(
        r#", "smooth_lag": 500, "emit_smoothed_se": true, "max_state_bytes": 2000000"#,
        "",
    );
    raised.check().unwrap();
}

/// The bank reads each group's last `L` observations from the state, the
/// ones whose lag no row has completed: per target, oldest first, the
/// newest the stream's last observation, by its clock; refused for a spec
/// without a smoother.
#[test]
fn the_bank_reads_the_last_observations_from_the_state() {
    let df = frame(300, 4965);
    let lag = 5;
    let s = spec(
        &format!(r#", "smooth_lag": {lag}, "emit_smoothed_se": true"#),
        r#", "clock": "t""#,
    );
    let mut bank = Bank::new(vec![s.clone()]).unwrap();
    feed(&mut bank, &df, 300);
    let tail = bank.smoothed_tail(0, None).unwrap();
    assert_eq!(tail.height(), 2 * lag);
    let names: Vec<&str> = tail.get_column_names().iter().map(|n| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "spec",
            "group",
            "instance",
            "half_life",
            "lam",
            "target",
            "smoothed_for",
            "withheld",
            "smoothed_coef",
            "smoothed_se_coef"
        ]
    );
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let clock: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let targets = tail.column("target").unwrap().str().unwrap();
    let labels = tail.column("smoothed_for").unwrap().f64().unwrap();
    for t in ["y", "z"] {
        let y = df.column(t).unwrap().f64().unwrap();
        let obs: Vec<f64> = (0..df.height())
            .filter(|&i| w[i] > 0.0 && y.get(i).is_some())
            .map(|i| clock[i])
            .collect();
        let got: Vec<f64> = (0..tail.height())
            .filter(|&i| targets.get(i) == Some(t))
            .map(|i| labels.get(i).unwrap())
            .collect();
        assert_eq!(got, obs[obs.len() - lag..], "{t}");
    }
    let mut plain = Bank::new(vec![spec("", r#", "clock": "t""#)]).unwrap();
    feed(&mut plain, &df, 300);
    let e = plain.smoothed_tail(0, None).unwrap_err();
    assert!(e.contains("it has no fixed-lag smoother"), "{e}");
}
