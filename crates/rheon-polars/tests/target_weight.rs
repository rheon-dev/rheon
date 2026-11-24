//! `emit_target_weight` (docs/PLAN.md task 568): `weight_<target>` is the
//! weight that target's `min_weight` gate reads (CLAUDE.md rule 8, S2).
//!
//! A row whose target is null teaches that target nothing but adds to the
//! shared `weight_sum`, so a row can show `weight_sum` above `min_weight`
//! and `below_min_weight` together; the field is the number the gate read.
//! Held here: the field against the gate, on both sides of the floor, and
//! against the weight's recursion written by hand; `weight_sum` where the
//! model keeps no weight per target; chunk invariance and a save and load
//! to the bit, `Bank::last_row` included; nothing else moving when the
//! switch is on, and the spec's bytes unchanged when it is off; and the
//! refusals.

use polars::prelude::*;
use rheon_polars::{Bank, Spec};

const N: usize = 240;
const HALF_LIFE: f64 = 12.0;
/// Between `y2`'s settled weight and `y`'s, so `y2` sits below it for good
/// stretches after its runs of nulls while `weight_sum` is above it.
const FLOOR: f64 = 9.0;

/// A seeded generator in `[-1, 1)`.
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Two features; `y` always present, `y2` null in runs (rows 40-69 and
/// 150-179) and on one row in five besides; weights on `[0.5, 1.5]` with
/// one row in eleven at 0.
fn frame() -> DataFrame {
    let mut r = lcg(5680);
    let (mut x0, mut x1, mut y, mut y2, mut w) = (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..N {
        let (a, b) = (r(), r());
        x0.push(a);
        x1.push(b);
        y.push(0.5 + 1.2 * a - 0.7 * b + 0.3 * r());
        let gap = (40..70).contains(&i) || (150..180).contains(&i) || i % 5 == 3;
        y2.push((!gap).then(|| -0.4 + 0.8 * a + 0.2 * b + 0.3 * r()));
        w.push(if i % 11 == 6 { 0.0 } else { 1.0 + 0.5 * r() });
    }
    df!("x0" => x0, "x1" => x1, "y" => y, "y2" => y2, "w" => w).unwrap()
}

fn spec(model: &str, more: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y", "y2"], "features": ["x0", "x1"],
            "weight": "w", "half_life": {HALF_LIFE}, "min_weight": {FLOOR},
            "emit_target_weight": true{more}}}"#
    ))
    .unwrap_or_else(|e| panic!("{model}: {e}"))
}

/// The models that keep a weight per target, each with its gates other than
/// `min_weight` left off where it has one.
const OWN: [&str; 5] = [
    r#"{"type": "ewridge"}"#,
    r#"{"type": "rls"}"#,
    r#"{"type": "sgd", "learning_rate": 0.02}"#,
    r#"{"type": "kalman", "coef_half_life": 100.0}"#,
    r#"{"type": "huber"}"#,
];

/// `ewridge`'s noise gate is on by default; off, so `min_weight` is the
/// only gate a row meets.
fn gates_off(model: &str) -> &'static str {
    if model.contains("ewridge") {
        r#", "max_error_inflation": "inf""#
    } else {
        ""
    }
}

/// `df` through `bank` in chunks of `size` rows, stacked.
fn feed(bank: &mut Bank, df: &DataFrame, size: usize) -> DataFrame {
    let mut acc: Option<DataFrame> = None;
    let mut i = 0;
    while i < df.height() {
        let chunk = df.slice(i as i64, size.min(df.height() - i));
        let out = DataFrame::new(chunk.height(), bank.fit_predict(&chunk).unwrap()).unwrap();
        match &mut acc {
            Some(a) => {
                a.vstack_mut(&out).unwrap();
            }
            None => acc = Some(out),
        }
        i += chunk.height();
    }
    acc.unwrap()
}

fn fields(out: &DataFrame, name: &str) -> Vec<Series> {
    out.column(name)
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series()
}

fn field<'a>(fs: &'a [Series], name: &str) -> &'a Series {
    fs.iter()
        .find(|s| s.name().as_str() == name)
        .unwrap_or_else(|| panic!("no field {name}"))
}

fn f64s(s: &Series) -> Vec<Option<f64>> {
    s.f64().unwrap().iter().collect()
}

/// A series by its exact contents: a float by its bits, a list by its
/// values' bits, anything else by its text.
fn bits(s: &Series) -> Vec<String> {
    if let Ok(f) = s.f64() {
        return f
            .iter()
            .map(|v| format!("{:?}", v.map(f64::to_bits)))
            .collect();
    }
    if let Ok(l) = s.list() {
        return (0..l.len())
            .map(|i| match l.get_as_series(i) {
                None => "null".to_string(),
                Some(inner) => format!("{:?}", bits(&inner)),
            })
            .collect();
    }
    (0..s.len())
        .map(|i| format!("{:?}", s.get(i).unwrap()))
        .collect()
}

/// Each target's weight before each row, by its recursion over the rows:
/// a row of positive weight decays what came before it by one step on the
/// row clock, then adds its weight where the target is present; a row of
/// weight 0 is in no clock and adds nothing (hard rule 9).
fn by_hand(df: &DataFrame, target: &str) -> Vec<f64> {
    let lam = 0.5f64.powf(1.0 / HALF_LIFE);
    let (y, w) = (
        df.column(target).unwrap().f64().unwrap(),
        df.column("w").unwrap().f64().unwrap(),
    );
    let (mut acc, mut out, mut first) = (0.0, Vec::new(), true);
    for i in 0..df.height() {
        out.push(acc);
        let wi = w.get(i).unwrap();
        if wi > 0.0 {
            if !first {
                acc *= lam;
            }
            first = false;
            if y.get(i).is_some() {
                acc += wi;
            }
        }
    }
    out
}

#[test]
fn the_field_is_the_weight_each_targets_gate_read() {
    let df = frame();
    let want_y2 = by_hand(&df, "y2");
    let want_sum = by_hand(&df, "y");
    for model in OWN {
        let mut bank = Bank::new(vec![spec(model, gates_off(model))]).unwrap();
        let fs = fields(&feed(&mut bank, &df, N), "m");
        let (sum, wy, wy2) = (
            f64s(field(&fs, "weight_sum")),
            f64s(field(&fs, "weight_y")),
            f64s(field(&fs, "weight_y2")),
        );
        let pred_y2 = f64s(field(&fs, "pred_y2"));
        let reason = field(&fs, "withheld_reason");
        let (mut under, mut over, mut hidden) = (0, 0, 0);
        for i in 0..N {
            let (s, a, b) = (sum[i].unwrap(), wy[i].unwrap(), wy2[i].unwrap());
            let close = |got: f64, want: f64| (got - want).abs() <= 1e-12 * want.max(1.0);
            assert!(
                close(b, want_y2[i]),
                "{model}: row {i}: weight_y2 {b} by hand {}",
                want_y2[i]
            );
            assert!(
                close(s, want_sum[i]),
                "{model}: row {i}: weight_sum {s} by hand {}",
                want_sum[i]
            );
            // `y` is present on every row, so its own weight is the shared
            // one; `rls` learns only rows that carry every target, so each
            // of its targets' weights is `y2`'s.
            let shared = if model.contains("rls") { b } else { s };
            assert!(
                close(a, shared),
                "{model}: row {i}: weight_y {a}, want {shared}"
            );
            // The gate: `y2`'s prediction is withheld exactly where its own
            // weight is below the floor, whatever `weight_sum` says.
            assert_eq!(
                pred_y2[i].is_none(),
                b < FLOOR,
                "{model}: row {i}: weight_y2 {b}, pred_y2 {:?}",
                pred_y2[i]
            );
            if b < FLOOR {
                under += 1;
                assert_eq!(
                    reason.get(i).unwrap().to_string().trim_matches('"'),
                    "below_min_weight",
                    "{model}: row {i}"
                );
                if s >= FLOOR {
                    hidden += 1;
                }
            } else {
                over += 1;
            }
        }
        // Both sides of the floor, and the case the field is for: the shared
        // weight past the floor while the target's own is under it.
        assert!(
            under > 0 && over > 0 && hidden > 0,
            "{model}: {under} under, {over} over, {hidden} hidden"
        );
        // The rows just either side of it, as it is crossed after a run of
        // nulls: the last under is withheld, the next row predicted.
        let cross = (171..N).find(|&i| wy2[i].unwrap() >= FLOOR).unwrap();
        assert!(wy2[cross - 1].unwrap() < FLOOR && pred_y2[cross - 1].is_none());
        assert!(pred_y2[cross].is_some(), "{model}: row {cross}");
    }
}

/// `seqtest`, `hawkes` and `dcc` keep no weight per target: every target's
/// gate reads the shared weight, and so the field is `weight_sum`, to the bit.
#[test]
fn a_model_with_no_weight_per_target_writes_weight_sum() {
    let mut df = frame();
    // `hawkes` reads its targets as event counts.
    let counts: Vec<Option<f64>> = df
        .column("y2")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .map(|v| v.map(|v| (v.abs() * 3.0).floor()))
        .collect();
    df.with_column(Column::new("e".into(), counts)).unwrap();
    let cases = [
        (r#"{"type": "seqtest"}"#, r#"["y", "y2"]"#, "[]"),
        (
            r#"{"type": "hawkes", "kernel_decays": [1.0, 5.0]}"#,
            r#"["e"]"#,
            "[]",
        ),
        (r#"{"type": "dcc"}"#, r#"["y", "y2"]"#, "[]"),
    ];
    for (model, targets, features) in cases {
        // `seqtest` counts each row as one trial, with no decay and no
        // weight column.
        let more = if model.contains("seqtest") {
            ""
        } else {
            r#", "weight": "w", "half_life": 20.0"#
        };
        let s: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {model}, "targets": {targets}, "features": {features},
                "emit_target_weight": true{more}}}"#
        ))
        .unwrap_or_else(|e| panic!("{model}: {e}"));
        let names: Vec<String> = s.targets.iter().map(|t| format!("weight_{t}")).collect();
        let mut bank = Bank::new(vec![s]).unwrap_or_else(|e| panic!("{model}: {e}"));
        let fs = fields(&feed(&mut bank, &df, 37), "m");
        let sum = bits(field(&fs, "weight_sum"));
        assert!(sum.iter().any(|b| b != "None"), "{model}: no weight");
        for name in &names {
            assert_eq!(bits(field(&fs, name)), sum, "{model}: {name}");
        }
    }
}

/// The same bits in chunks of 1, 7 and 100 rows as in one, on a half-life
/// grid, each instance's fields suffixed as `weight_sum`'s are.
#[test]
fn the_field_is_chunk_invariant() {
    let df = frame();
    for model in OWN {
        let mut s = spec(model, "");
        s.half_life = serde_json::from_str("[6.0, 30.0]").unwrap();
        let whole = fields(&feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, N), "m");
        for name in [
            "weight_y@h6",
            "weight_y2@h6",
            "weight_y@h30",
            "weight_y2@h30",
        ] {
            assert!(field(&whole, name).null_count() == 0, "{model}: {name}");
        }
        for size in [1, 7, 100] {
            let parts = fields(
                &feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, size),
                "m",
            );
            for f in &whole {
                // `coef`'s rows follow the chunks by design.
                if f.name().contains("coef") {
                    continue;
                }
                assert_eq!(
                    bits(f),
                    bits(field(&parts, f.name())),
                    "{model}, chunks of {size}: {}",
                    f.name()
                );
            }
        }
    }
}

/// Saved part-way and loaded, a bank goes on to the bit, its last row
/// included, and saves the bytes it loaded.
#[test]
fn the_field_survives_a_save_and_load_to_the_bit() {
    let df = frame();
    for model in OWN {
        let s = spec(model, gates_off(model));
        let whole = fields(&feed(&mut Bank::new(vec![s.clone()]).unwrap(), &df, N), "m");
        // Cut inside `y2`'s second run of nulls, where its weight is short.
        let cut = 160;
        let mut first = Bank::new(vec![s.clone()]).unwrap();
        let head = feed(&mut first, &df.slice(0, cut), 50);
        let (_, last) = first.last_row(0, None).unwrap();
        let bytes = first.save_bytes().unwrap();
        let mut second = Bank::load_bytes(&bytes, Some(&[s])).unwrap();
        assert_eq!(second.save_bytes().unwrap(), bytes, "{model}: re-saved");
        let (_, loaded) = second.last_row(0, None).unwrap();
        assert_eq!(
            bits(&last.as_materialized_series().clone()),
            bits(&loaded.as_materialized_series().clone()),
            "{model}: last row"
        );
        let last_fs = last.struct_().unwrap().fields_as_series();
        assert!(
            field(&last_fs, "weight_y2").null_count() == 0,
            "{model}: last row's weight_y2"
        );
        let mut out = head;
        out.vstack_mut(&feed(&mut second, &df.slice(cut as i64, N - cut), 50))
            .unwrap();
        let resumed = fields(&out, "m");
        for name in ["weight_sum", "weight_y", "weight_y2", "pred_y2"] {
            assert_eq!(
                bits(field(&whole, name)),
                bits(field(&resumed, name)),
                "{model}: {name}"
            );
        }
    }
}

/// The switch adds its fields and moves no other bit, and off it leaves the
/// spec's serialized form as it was.
#[test]
fn the_switch_moves_nothing_else() {
    let df = frame();
    for model in OWN {
        let on = spec(model, "");
        let mut off = on.clone();
        off.emit_target_weight = false;
        let with = fields(
            &feed(&mut Bank::new(vec![on.clone()]).unwrap(), &df, 50),
            "m",
        );
        let without = fields(
            &feed(&mut Bank::new(vec![off.clone()]).unwrap(), &df, 50),
            "m",
        );
        let mut added: Vec<&str> = with
            .iter()
            .map(|f| f.name().as_str())
            .filter(|n| without.iter().all(|g| g.name().as_str() != *n))
            .collect();
        added.sort_unstable();
        assert_eq!(added, ["weight_y", "weight_y2"], "{model}");
        for f in &without {
            assert_eq!(
                bits(f),
                bits(field(&with, f.name())),
                "{model}: {}",
                f.name()
            );
        }
        // In struct order, beside `weight_sum`.
        let order: Vec<&str> = with.iter().map(|f| f.name().as_str()).collect();
        let at = order.iter().position(|n| *n == "weight_sum").unwrap();
        assert_eq!(
            &order[at..at + 3],
            ["weight_sum", "weight_y", "weight_y2"],
            "{model}"
        );
        let text = serde_json::to_string(&off).unwrap();
        assert!(!text.contains("emit_target_weight"), "{model}: {text}");
        assert!(
            serde_json::to_string(&on)
                .unwrap()
                .contains(r#""emit_target_weight":true"#)
        );
    }
}

#[test]
fn a_model_with_no_target_refuses_the_switch_by_name() {
    for (model, targets, why) in [
        (
            r#"{"type": "ew_cov", "stats": ["mean"]}"#,
            "[]",
            "learns from no target",
        ),
        (
            r#"{"type": "kmeans", "n_clusters": 2}"#,
            "[]",
            "learns from no target",
        ),
        (
            r#"{"type": "marginal"}"#,
            r#"["y"]"#,
            "writes no prediction",
        ),
    ] {
        let mut s: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {model}, "targets": {targets}, "features": ["x0", "x1"],
                "half_life": 20.0, "emit_target_weight": true}}"#
        ))
        .unwrap();
        s.fill_defaults();
        let e = Bank::new(vec![s])
            .err()
            .unwrap_or_else(|| panic!("{model}: accepted"));
        assert!(
            e.contains("emit_target_weight") && e.contains(why),
            "{model}: {e}"
        );
    }
    // A target called `sum` would write a second `weight_sum`: refused as a
    // duplicate field, as any clash of names is.
    let s: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "rls"}, "targets": ["y", "sum"],
            "features": ["x0"], "half_life": 20.0, "emit_target_weight": true}"#,
    )
    .unwrap();
    let e = Bank::new(vec![s]).err().expect("a duplicate field");
    assert!(e.contains("weight_sum"), "{e}");
}
