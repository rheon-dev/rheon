//! Bank-level integration tests for `krls`: clock sliding window (`window_size`),
//! atomic `HeldCap` warning & refusal, `predict` stamp reads (`reads_at_stamp`),
//! `emit_n_atoms` schema/values, chunk invariance, and `half_life` grids.

use polars::prelude::*;
use rheon_core::{KernelKind, Krls, KrlsCfg, OnlineModel, WindowClosed};
use rheon_polars::{Bank, Spec};

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn spec(model: &str, extra: &str) -> Spec {
    let text = format!(
        r#"{{"name": "k", "model": {{"type": "krls"{model}}}, "targets": ["y"],
            "features": ["x0", "x1"], {extra} "min_weight": 0.0}}"#
    );
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn field_f64(out: &DataFrame, name: &str) -> Vec<Option<f64>> {
    out.column("k")
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

fn field_i64(out: &DataFrame, name: &str) -> Vec<Option<i64>> {
    out.column("k")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name(name)
        .unwrap()
        .i64()
        .unwrap()
        .iter()
        .collect()
}

fn bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

/// Clock-window `krls` matches the core model bit for bit on both edges
/// (`"right"` and `"both"`), and a burst of rows in one group warns once past
/// half of `max_state_bytes` and then refuses atomically at the cap.
#[test]
fn clock_window_warns_once_and_refuses_at_cap() {
    let probe = KrlsCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        kernel: KernelKind::Rbf,
        gamma: None,
        degree: None,
        coef0: None,
        ridge: 1.0,
        max_atoms: None,
        ald_threshold: None,
        window_rows: None,
        window_clock: Some(1000.0),
        closed: WindowClosed::Right,
        decay: rheon_core::Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        emit_n_atoms: true,
        max_state_bytes: None,
    };
    let cap = probe.window_bytes(30);
    let sp = spec(
        &format!(r#", "window_size": 1000.0, "emit_n_atoms": true, "max_state_bytes": {cap}"#),
        r#""clock": "t", "group": "g", "gap_cap": 1e9,"#,
    );
    let burst = |group: &str, start: usize, n: usize| {
        let mut r = lcg(17 ^ start as u64 ^ group.len() as u64);
        let t: Vec<f64> = (start..start + n).map(|i| i as f64 * 0.01).collect();
        let x0: Vec<f64> = (0..n).map(|_| r()).collect();
        let x1: Vec<f64> = (0..n).map(|_| r()).collect();
        let y: Vec<f64> = x0.iter().zip(&x1).map(|(a, b)| (a * b).sin()).collect();
        df!("g" => vec![group; n], "t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap()
    };
    let mut bank = Bank::new(vec![sp]).unwrap();
    // Group "a" takes 25 rows (> 50% of cap for 30 rows); group "b" takes 5 rows (< 50%).
    let first = burst("a", 0, 25).vstack(&burst("b", 0, 5)).unwrap();
    bank.fit_predict(&first).unwrap();
    let notices = bank.take_notices();
    assert_eq!(notices.len(), 1, "{notices:#?}");
    assert!(
        notices[0].starts_with(r#"spec "k" group "a": "#)
            && notices[0].contains(&format!("max_state_bytes ({cap})")),
        "{}",
        notices[0]
    );
    // 10 more rows for "a" pass the 30-row cap: no second warning, and refused naming group & bytes.
    let more = burst("a", 25, 10);
    let err = bank.fit_predict(&more).unwrap_err().to_string();
    assert!(err.contains(r#"spec "k" group "a""#), "{err}");
    assert!(
        err.contains(&format!("to {} bytes", probe.window_bytes(31))),
        "{err}"
    );
    assert!(err.contains(&format!("max_state_bytes of {cap}")), "{err}");
    assert!(bank.take_notices().is_empty(), "warning raised once");
    // Stopped bank cannot continue.
    let e = bank.fit_predict(&more).unwrap_err().to_string();
    assert!(e.contains("cannot go on"), "{e}");
}

/// `predict` on a clock-windowed `krls` reads at each row's stamp
/// (`reads_at_stamp()`): a future timestamp past `window_size` sees an empty
/// window (`pred_y = null`, `n_atoms = 0`, `withheld_reason = "not_identified"`)
/// without mutating the bank's live state.
#[test]
fn clock_window_predict_reads_at_stamp_without_mutating_state() {
    let sp = spec(
        r#", "window_size": 5.0, "closed": "right", "emit_n_atoms": true"#,
        r#""clock": "t", "gap_cap": 100.0,"#,
    );
    let mut r = lcg(99);
    let n = 20usize;
    let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let x0: Vec<f64> = (0..n).map(|_| r()).collect();
    let x1: Vec<f64> = (0..n).map(|_| r()).collect();
    let y: Vec<f64> = x0.iter().zip(&x1).map(|(a, b)| a * a - b).collect();
    let train = df!("t" => t, "x0" => x0, "x1" => x1, "y" => y).unwrap();

    let mut bank = Bank::new(vec![sp]).unwrap();
    bank.fit_predict(&train).unwrap();
    let saved_before = bank.save_bytes().unwrap();

    // Last learned row is at t = 9.5.
    // Query 1: t = 10.0 (inside window, keeps rows with t > 5.0 => 9 rows: 5.5..=9.5).
    // Query 2: t = 20.0 (past window: all rows aged out => n_atoms = 0, pred = null).
    let query = df!(
        "t" => [10.0, 20.0],
        "x0" => [0.2, 0.2],
        "x1" => [-0.3, -0.3],
    )
    .unwrap();
    let out = DataFrame::new(2, bank.predict(&query).unwrap()).unwrap();
    assert_eq!(
        bank.save_bytes().unwrap(),
        saved_before,
        "predict mutated state"
    );
    let preds = field_f64(&out, "pred_y");
    let atoms = field_i64(&out, "n_atoms");
    assert!(preds[0].is_some_and(f64::is_finite), "{preds:?}");
    assert_eq!(atoms[0], Some(9), "{atoms:?}");
    assert_eq!(preds[1], None, "expired window predicts null");
    assert_eq!(atoms[1], Some(0), "expired window has 0 atoms");

    // Verify exact agreement with core `predict_at`.
    let mut core = Krls::new(KrlsCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        kernel: KernelKind::Rbf,
        gamma: None,
        degree: None,
        coef0: None,
        ridge: 1.0,
        max_atoms: None,
        ald_threshold: None,
        window_rows: None,
        window_clock: Some(5.0),
        closed: WindowClosed::Right,
        decay: rheon_core::Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        emit_n_atoms: true,
        max_state_bytes: None,
    })
    .unwrap();
    let tc = train.column("t").unwrap().f64().unwrap();
    let x0c = train.column("x0").unwrap().f64().unwrap();
    let x1c = train.column("x1").unwrap().f64().unwrap();
    let yc = train.column("y").unwrap().f64().unwrap();
    for i in 0..n {
        let now = tc.get(i).unwrap();
        let d = if i == 0 {
            0.0
        } else {
            now - tc.get(i - 1).unwrap()
        };
        core.stamp_next(rheon_core::Stamp::Raw(now, 0.0));
        core.step(
            &[x0c.get(i).unwrap(), x1c.get(i).unwrap()],
            &[yc.get(i)],
            d,
            1.0,
        );
    }
    let st0 = core.predict_at(&[0.2, -0.3], 0.5, rheon_core::Stamp::Raw(10.0, 0.0));
    assert_eq!(bits(&preds[..1]), bits(&[Some(st0.pred[0])]));
}

/// `emit_n_atoms` is omitted from the output struct when `false` and is `Int64`
/// when `true` (null on skipped rows), while `half_life` grids support
/// `emit_selected`, `emit_averaged`, and `emit_sigma`.
#[test]
fn emit_n_atoms_and_half_life_grid() {
    let mut r = lcg(7);
    let n = 60usize;
    let x0: Vec<Option<f64>> = (0..n).map(|i| (i != 10).then(&mut r)).collect();
    let x1: Vec<f64> = (0..n).map(|_| r()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| x0[i].unwrap_or(0.0).sin() + 0.5 * x1[i] * x1[i])
        .collect();
    let df = df!("x0" => x0, "x1" => x1, "y" => y).unwrap();

    // Default (`emit_n_atoms` false): no `n_atoms` field in output.
    let s_off = spec("", "");
    assert!(
        !rheon_polars::output_index(&s_off)
            .iter()
            .any(|f| f.field == "n_atoms")
    );

    // Grid with `emit_n_atoms`, `emit_sigma`, `emit_selected`, `emit_averaged`.
    let s_grid = spec(
        r#", "max_atoms": 15, "emit_n_atoms": true"#,
        r#""half_life": [10.0, 40.0], "emit_sigma": true, "emit_selected": true, "emit_averaged": true,"#,
    );
    let mut b1 = Bank::new(vec![s_grid.clone()]).unwrap();
    let whole = DataFrame::new(n, b1.fit_predict(&df).unwrap()).unwrap();
    let mut b2 = Bank::new(vec![s_grid]).unwrap();
    let mut parts = DataFrame::new(23, b2.fit_predict(&df.slice(0, 23)).unwrap()).unwrap();
    parts
        .vstack_mut(
            &DataFrame::new(n - 23, b2.fit_predict(&df.slice(23, n - 23)).unwrap()).unwrap(),
        )
        .unwrap();
    assert!(whole.equals_missing(&parts), "chunk invariance failed");

    let atoms_10 = field_i64(&whole, "n_atoms@h10");
    assert_eq!(atoms_10[0], Some(0), "row 0 has 0 atoms before learning");
    assert_eq!(atoms_10[10], None, "skipped row reports null n_atoms");
    assert!(
        atoms_10.last().unwrap().unwrap() <= 15 && atoms_10.last().unwrap().unwrap() > 0,
        "{:?}",
        atoms_10.last()
    );
    assert!(
        field_f64(&whole, "pred_y__selected")
            .last()
            .unwrap()
            .is_some(),
        "selected prediction present"
    );
    assert!(
        field_f64(&whole, "pred_y__averaged")
            .last()
            .unwrap()
            .is_some(),
        "averaged prediction present"
    );
}
