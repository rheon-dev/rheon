//! Ungated `predict_with_floor` and scored-row reads under clock windows
//! (`window_size`): every model whose read decides a window's edge on the
//! clock (`expectile`, `isotonic`, `ddsketch`, `corrchange`, `hawkes`,
//! `ewm_moments`) reads at the row's exact stamp rather than the
//! floating-point read-ahead (`last + d_clock`), on learned rows, buffered
//! (`embargo` / `label_clock`) rows, weight-0 rows, and `Bank::predict` rows.

use polars::prelude::*;
use rheon_core::{
    ChangeNorm, CorrChange, CorrChangeCfg, CorrChangeKind, DdSketchModel, DdSketchModelCfg, Decay,
    EwmMoments, EwmMomentsCfg, Expectile, ExpectileCfg, Hawkes, HawkesCfg, Isotonic, IsotonicCfg,
    IsotonicOutOfBounds, MomentKind, MomentLeg, MomentTerm, OnlineModel, Stamp, WindowClosed,
};
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

fn run_bank(spec: &Spec, df: &DataFrame, chunk: usize) -> (Vec<Series>, Vec<u8>) {
    let mut bank = Bank::new(vec![spec.clone()]).unwrap();
    let mut out: Option<DataFrame> = None;
    let mut start = 0;
    while start < df.height() {
        let piece = df.slice(start as i64, chunk);
        let got = DataFrame::new(piece.height(), bank.fit_predict(&piece).unwrap()).unwrap();
        match out.as_mut() {
            None => out = Some(got),
            Some(o) => {
                o.vstack_mut(&got).unwrap();
            }
        }
        start += chunk;
    }
    let fields = out
        .unwrap()
        .column(&spec.name)
        .unwrap()
        .struct_()
        .unwrap()
        .fields_as_series();
    (fields, bank.save_bytes().unwrap())
}

fn named<'a>(out: &'a [Series], name: &str) -> &'a Series {
    out.iter()
        .find(|s| s.name().as_str() == name)
        .unwrap_or_else(|| panic!("no field {name}"))
}

fn with_prefix<'a>(out: &'a [Series], prefix: &str) -> Vec<&'a Series> {
    let found: Vec<&Series> = out
        .iter()
        .filter(|s| s.name().as_str() == prefix || s.name().starts_with(&format!("{prefix}@")))
        .collect();
    assert!(!found.is_empty(), "no field {prefix}");
    found
}

fn bits(s: &Series) -> Vec<Option<u64>> {
    s.f64()
        .unwrap()
        .iter()
        .map(|v| v.map(f64::to_bits))
        .collect()
}

fn opt_bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

fn withheld(out: &[Series]) -> usize {
    out.iter()
        .filter(|s| s.name().starts_with("withheld_reason"))
        .map(|s| s.len() - s.null_count())
        .sum()
}

fn spread_bytes(bytes: &[u8]) -> Vec<String> {
    fn walk(v: &rmpv::Value, out: &mut Vec<String>) {
        match v {
            rmpv::Value::Map(m) => {
                for (k, v) in m {
                    if matches!(k.as_str(), Some("resid_var" | "resid_w")) {
                        out.push(format!("{v}"));
                    }
                    walk(v, out);
                }
            }
            rmpv::Value::Array(a) => a.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(
        &rmpv::decode::read_value(&mut &bytes[..]).unwrap(),
        &mut out,
    );
    assert!(!out.is_empty(), "a state with a spread");
    out
}

/// Tenths clock frame: dense tenths in `[0.0, 5.0]` (starting with explicit
/// `0.0, 0.1, 0.2, 0.3, 0.4, 0.9, 1.0, 1.2, 1.4, ...` where `0.2 + 0.7 < 0.9`,
/// `(0.2 - 0.1) + (1.2 - 0.2) > 1.1`, etc.) so clock window edges split
/// between `back_by(last, -d)` and the exact `Stamp::Raw(t, 0.0)`.
fn tenths_regression_frame(n: usize, seed: u64, with_zero_weights: bool) -> DataFrame {
    let mut r = lcg(seed);
    let prefix: [i64; 18] = [
        0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22, 25,
    ];
    let mut tenths = 0i64;
    let (mut t, mut x0, mut x1, mut y, mut w) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        if i < prefix.len() {
            tenths = prefix[i];
        } else {
            tenths += (r() * 4.0).floor() as i64;
        }
        let ts = tenths as f64 / 10.0;
        let a = r() * 2.0 - 1.0;
        let b = r() * 2.0 - 1.0;
        // Include a regime shift midway so drift detectors can also react.
        let shift = if i >= n / 2 { 1.5 } else { 0.0 };
        t.push(ts);
        x0.push(a);
        x1.push(b);
        y.push(0.4 + 1.3 * a - 0.6 * b + shift + (r() - 0.5) * 0.8);
        w.push(
            if with_zero_weights && (i == 6 || i == 9 || i == 10 || (i > 12 && i % 3 == 2)) {
                0.0
            } else {
                1.0
            },
        );
    }
    df!("t" => t, "x0" => x0, "x1" => x1, "y" => y, "w" => w).unwrap()
}

/// `expectile` under a clock `window_size` on a tenths clock: with `min_weight`
/// withholding rows (both learned and `embargo`-buffered), `sigma_y`, saved
/// residual spread, `zscore_y`, and drift flags must match `min_weight = 0.0`
/// to the bit.
#[test]
fn expectile_window_min_weight_gate_isolation() {
    let df = tenths_regression_frame(260, 101, false);
    for embargo in ["", r#", "embargo": 0.4"#] {
        for span in [0.9, 1.1, 1.5, 2.3] {
            let make = |mw: f64| -> Spec {
                let text = format!(
                    r#"{{"name": "m",
                        "model": {{"type": "expectile", "expectile": 0.75, "window_size": {span}, "closed": "right"}},
                        "targets": ["y"], "features": ["x0", "x1"],
                        "clock": "t", "half_life": "inf", "gap_cap": 100.0,
                        "min_weight": {mw}, "emit_sigma": true, "emit_zscore": true,
                        "emit_drift": true, "drift_threshold": 5.0, "drift_action": "flag"{embargo}}}"#
                );
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
            };
            let (base, base_bytes) = run_bank(&make(0.0), &df, 67);
            // `min_weight = 100.0` keeps `short == true` on every window row so
            // `predict_with_floor` runs across every edge split on the clock.
            let (gated, gated_bytes) = run_bank(&make(100.0), &df, 67);
            assert!(
                withheld(&gated) > withheld(&base),
                "span={span} embargo={embargo}: gate withheld nothing"
            );
            let b_sig = bits(named(&base, "sigma_y"));
            let g_sig = bits(named(&gated, "sigma_y"));
            let first = b_sig.iter().zip(&g_sig).position(|(a, b)| a != b);
            assert_eq!(
                first, None,
                "expectile span={span} embargo={embargo}: sigma_y moved at row {first:?}"
            );
            assert_eq!(
                spread_bytes(&base_bytes),
                spread_bytes(&gated_bytes),
                "expectile span={span} embargo={embargo}: saved spread moved"
            );
            let b_drift: Vec<_> = named(&base, "drift_y").bool().unwrap().iter().collect();
            let g_drift: Vec<_> = named(&gated, "drift_y").bool().unwrap().iter().collect();
            assert_eq!(
                b_drift, g_drift,
                "expectile span={span} embargo={embargo}: drift_flag moved"
            );
        }
    }
}

/// `expectile` under a clock `window_size` on a tenths clock: weight-0 rows in
/// `fit_predict` and rows scored via `Bank::predict` must read the window at
/// their exact stamp (not `back_by(last, -d)`).
#[test]
fn expectile_window_scored_rows_read_at_stamp() {
    let df = tenths_regression_frame(260, 202, true);
    let t: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let x0: Vec<f64> = df
        .column("x0")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let x1: Vec<f64> = df
        .column("x1")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let y: Vec<f64> = df
        .column("y")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let mut parted = 0usize;
    for span in (1..25).map(|k| k as f64 / 10.0) {
        let sp: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m",
                "model": {{"type": "expectile", "expectile": 0.75, "window_size": {span}, "closed": "right"}},
                "targets": ["y"], "features": ["x0", "x1"],
                "clock": "t", "weight": "w", "half_life": "inf", "gap_cap": 100.0,
                "min_weight": 0.0}}"#
        ))
        .unwrap();
        let (out, _) = run_bank(&sp, &df, 53);
        let got: Vec<Option<f64>> = named(&out, "pred_y").f64().unwrap().iter().collect();
        let mut m = Expectile::new(ExpectileCfg {
            n_features: 2,
            n_targets: 1,
            expectile: 0.75,
            decay: Decay::Halflife(f64::INFINITY),
            ridge: 1e-6,
            fit_intercept: true,
            min_weight: 0.0,
            refit_rows: 0,
            refit_clock: None,
            window_rows: None,
            window_clock: Some(span),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap();
        let mut last: Option<f64> = None;
        let mut want = Vec::with_capacity(df.height());
        let mut bank_p = Bank::new(vec![sp.clone()]).unwrap();
        for i in 0..df.height() {
            let d = last.map_or(0.0, |l| t[i] - l);
            let stamp = Stamp::Raw(t[i], 0.0);
            let x = [x0[i], x1[i]];
            let step = if w[i] > 0.0 {
                last = Some(t[i]);
                m.stamp_next(stamp);
                bank_p.fit_predict(&df.slice(i as i64, 1)).unwrap();
                m.step(&x, &[Some(y[i])], d, 1.0)
            } else {
                let ahead = m.predict(&x, d);
                let mut at_m = m.clone();
                at_m.stamp_next(stamp);
                let exact = at_m.step(&x, &[Some(y[i])], d, 0.0);
                parted += usize::from(ahead.pred[0].to_bits() != exact.pred[0].to_bits());
                // Also verify `Bank::predict` agrees with `exact` to the bit.
                let pred_cols = bank_p.predict(&df.slice(i as i64, 1)).unwrap();
                let pred_df = DataFrame::new(1, pred_cols).unwrap();
                let p_val = pred_df
                    .column("m")
                    .unwrap()
                    .struct_()
                    .unwrap()
                    .field_by_name("pred_y")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(0);
                let want_opt = Some(exact.pred[0]).filter(|v| v.is_finite());
                assert_eq!(
                    p_val.map(f64::to_bits),
                    want_opt.map(f64::to_bits),
                    "expectile Bank::predict span={span} row={i} t={}",
                    t[i]
                );
                exact
            };
            want.push(Some(step.pred[0]).filter(|v| v.is_finite()));
        }
        assert_eq!(
            opt_bits(&got),
            opt_bits(&want),
            "expectile weight-0 span={span}"
        );
    }
    assert!(parted > 0, "no expectile scored row sat on a tenths edge");
}

/// `isotonic` under a clock `window_size` on a tenths clock with two
/// half-lives: `min_weight` must leave `sigma_y`, saved spread, `zscore_y`,
/// `emit_selected`, `emit_averaged`, and drift flags identical to the bit
/// across both plain learned rows (including rows where trimming drops
/// `t.weight` from >= `min_weight` to < `min_weight`) and `embargo`-buffered
/// rows.
#[test]
fn isotonic_window_min_weight_gate_isolation_and_slot_ranking() {
    let df = tenths_regression_frame(260, 303, false);
    for embargo in ["", r#", "embargo": 0.4"#] {
        // Test both a threshold right inside the window's weight (so trim
        // crosses `min_weight`) and a high threshold (withholding every row).
        for mw in [4.0, 100.0] {
            for span in [0.9, 1.1, 1.5, 2.3] {
                let make = |floor: f64| -> Spec {
                    let text = format!(
                        r#"{{"name": "m",
                            "model": {{"type": "isotonic", "window_size": {span}, "closed": "right", "out_of_bounds": "clip"}},
                            "targets": ["y"], "features": ["x0"],
                            "clock": "t", "half_life": [0.8, 3.0], "gap_cap": 100.0,
                            "min_weight": {floor}, "emit_sigma": true, "emit_zscore": true,
                            "emit_selected": true, "emit_averaged": true,
                            "emit_drift": true, "drift_threshold": 5.0, "drift_action": "flag"{embargo}}}"#
                    );
                    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
                };
                let (base, base_bytes) = run_bank(&make(0.0), &df, 61);
                let (gated, gated_bytes) = run_bank(&make(mw), &df, 61);
                assert!(
                    withheld(&gated) > withheld(&base),
                    "isotonic span={span} mw={mw} embargo={embargo}: gate withheld nothing"
                );
                for s in with_prefix(&base, "sigma_y") {
                    let (a, b) = (bits(s), bits(named(&gated, s.name())));
                    let first = a.iter().zip(&b).position(|(p, q)| p != q);
                    assert_eq!(
                        first,
                        None,
                        "isotonic span={span} mw={mw} embargo={embargo}: {} moved at row {first:?}",
                        s.name()
                    );
                }
                assert_eq!(
                    spread_bytes(&base_bytes),
                    spread_bytes(&gated_bytes),
                    "isotonic span={span} mw={mw} embargo={embargo}: saved spread moved"
                );
                if mw < 10.0 {
                    let slots = with_prefix(&gated, "pred_y");
                    let both_open: Vec<bool> = (0..df.height())
                        .map(|i| slots.iter().all(|s| s.f64().unwrap().get(i).is_some()))
                        .collect();
                    if both_open.iter().any(|&e| e) {
                        for name in ["pred_y__selected", "pred_y__averaged"] {
                            let (a, b) = (bits(named(&base, name)), bits(named(&gated, name)));
                            for (i, &open) in both_open.iter().enumerate() {
                                if open {
                                    assert_eq!(
                                        a[i], b[i],
                                        "isotonic span={span} mw={mw} embargo={embargo}: {name} moved at row {i}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `isotonic` under a clock `window_size` on a tenths clock: weight-0 rows and
/// `Bank::predict` rows read at their exact stamp.
#[test]
fn isotonic_window_scored_rows_read_at_stamp() {
    let df = tenths_regression_frame(260, 404, true);
    let t: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let x0: Vec<f64> = df
        .column("x0")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let y: Vec<f64> = df
        .column("y")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let mut parted = 0usize;
    for span in (1..25).map(|k| k as f64 / 10.0) {
        let sp: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m",
                "model": {{"type": "isotonic", "window_size": {span}, "closed": "right", "out_of_bounds": "clip"}},
                "targets": ["y"], "features": ["x0"],
                "clock": "t", "weight": "w", "half_life": "inf", "gap_cap": 100.0,
                "min_weight": 0.0}}"#
        ))
        .unwrap();
        let (out, _) = run_bank(&sp, &df, 59);
        let got: Vec<Option<f64>> = named(&out, "pred_y").f64().unwrap().iter().collect();
        let got_w: Vec<Option<f64>> = named(&out, "weight_sum").f64().unwrap().iter().collect();
        let mut m = Isotonic::new(IsotonicCfg {
            n_targets: 1,
            increasing: true,
            y_min: None,
            y_max: None,
            out_of_bounds: IsotonicOutOfBounds::Clip,
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
            window_rows: None,
            window_clock: Some(span),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap();
        let mut last: Option<f64> = None;
        let mut want = Vec::with_capacity(df.height());
        let mut want_w = Vec::with_capacity(df.height());
        let mut bank_p = Bank::new(vec![sp.clone()]).unwrap();
        for i in 0..df.height() {
            let d = last.map_or(0.0, |l| t[i] - l);
            let stamp = Stamp::Raw(t[i], 0.0);
            let x = [x0[i]];
            let step = if w[i] > 0.0 {
                last = Some(t[i]);
                m.stamp_next(stamp);
                bank_p.fit_predict(&df.slice(i as i64, 1)).unwrap();
                m.step(&x, &[Some(y[i])], d, 1.0)
            } else {
                let ahead = m.predict(&x, d);
                let mut at_m = m.clone();
                at_m.stamp_next(stamp);
                let exact = at_m.step(&x, &[Some(y[i])], d, 0.0);
                parted += usize::from(
                    ahead.pred[0].to_bits() != exact.pred[0].to_bits()
                        || ahead.n_eff.to_bits() != exact.n_eff.to_bits(),
                );
                let pred_cols = bank_p.predict(&df.slice(i as i64, 1)).unwrap();
                let pred_df = DataFrame::new(1, pred_cols).unwrap();
                let p_val = pred_df
                    .column("m")
                    .unwrap()
                    .struct_()
                    .unwrap()
                    .field_by_name("pred_y")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(0);
                let p_w = pred_df
                    .column("m")
                    .unwrap()
                    .struct_()
                    .unwrap()
                    .field_by_name("weight_sum")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(0);
                let want_opt = Some(exact.pred[0]).filter(|v| v.is_finite());
                assert_eq!(
                    p_val.map(f64::to_bits),
                    want_opt.map(f64::to_bits),
                    "isotonic Bank::predict pred_y span={span} row={i}"
                );
                assert_eq!(
                    p_w.map(f64::to_bits),
                    Some(exact.n_eff.to_bits()),
                    "isotonic Bank::predict weight_sum span={span} row={i}"
                );
                exact
            };
            want.push(Some(step.pred[0]).filter(|v| v.is_finite()));
            want_w.push(Some(step.n_eff));
        }
        assert_eq!(
            opt_bits(&got),
            opt_bits(&want),
            "isotonic weight-0 pred_y span={span}"
        );
        assert_eq!(
            opt_bits(&got_w),
            opt_bits(&want_w),
            "isotonic weight-0 weight_sum span={span}"
        );
    }
    assert!(parted > 0, "no isotonic scored row sat on a tenths edge");
}

/// `ddsketch` under a clock `window_size` on a tenths clock: weight-0 rows and
/// `Bank::predict` rows read at their exact stamp.
#[test]
fn ddsketch_window_scored_rows_read_at_stamp() {
    let df = tenths_regression_frame(260, 505, true);
    let t: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let y: Vec<f64> = df
        .column("y")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let mut parted = 0usize;
    for span in (1..25).map(|k| k as f64 / 10.0) {
        let sp: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m",
                "model": {{"type": "ddsketch", "quantiles": [0.4], "alpha": 0.01, "window_size": {span}, "closed": "right"}},
                "targets": ["y"], "features": [],
                "clock": "t", "weight": "w", "half_life": "inf", "gap_cap": 100.0,
                "min_weight": 0.0}}"#
        ))
        .unwrap();
        let (out, _) = run_bank(&sp, &df, 59);
        let got: Vec<Option<f64>> = named(&out, "q0.4_y").f64().unwrap().iter().collect();
        let got_w: Vec<Option<f64>> = named(&out, "weight_sum").f64().unwrap().iter().collect();
        let mut m = DdSketchModel::new(DdSketchModelCfg {
            n_targets: 1,
            levels: vec![0.4],
            alpha: 0.01,
            decay: Decay::Halflife(f64::INFINITY),
            window_rows: None,
            window_clock: Some(span),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap();
        let mut last: Option<f64> = None;
        let mut want = Vec::with_capacity(df.height());
        let mut want_w = Vec::with_capacity(df.height());
        let mut bank_p = Bank::new(vec![sp.clone()]).unwrap();
        for i in 0..df.height() {
            let d = last.map_or(0.0, |l| t[i] - l);
            let stamp = Stamp::Raw(t[i], 0.0);
            let step = if w[i] > 0.0 {
                last = Some(t[i]);
                m.stamp_next(stamp);
                bank_p.fit_predict(&df.slice(i as i64, 1)).unwrap();
                m.step(&[], &[Some(y[i])], d, 1.0)
            } else {
                let ahead = m.predict(&[], d);
                let mut at_m = m.clone();
                at_m.stamp_next(stamp);
                let exact = at_m.step(&[], &[Some(y[i])], d, 0.0);
                parted += usize::from(
                    ahead.pred[0].to_bits() != exact.pred[0].to_bits()
                        || ahead.n_eff.to_bits() != exact.n_eff.to_bits(),
                );
                let pred_cols = bank_p.predict(&df.slice(i as i64, 1)).unwrap();
                let pred_df = DataFrame::new(1, pred_cols).unwrap();
                let p_val = pred_df
                    .column("m")
                    .unwrap()
                    .struct_()
                    .unwrap()
                    .field_by_name("q0.4_y")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(0);
                let p_w = pred_df
                    .column("m")
                    .unwrap()
                    .struct_()
                    .unwrap()
                    .field_by_name("weight_sum")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .get(0);
                let want_opt = Some(exact.pred[0]).filter(|v| v.is_finite());
                assert_eq!(
                    p_val.map(f64::to_bits),
                    want_opt.map(f64::to_bits),
                    "ddsketch Bank::predict q0.4_y span={span} row={i}"
                );
                assert_eq!(
                    p_w.map(f64::to_bits),
                    Some(exact.n_eff.to_bits()),
                    "ddsketch Bank::predict weight_sum span={span} row={i}"
                );
                exact
            };
            want.push(Some(step.pred[0]).filter(|v| v.is_finite()));
            want_w.push(Some(step.n_eff));
        }
        assert_eq!(
            opt_bits(&got),
            opt_bits(&want),
            "ddsketch weight-0 q0.4_y span={span}"
        );
        assert_eq!(
            opt_bits(&got_w),
            opt_bits(&want_w),
            "ddsketch weight-0 weight_sum span={span}"
        );
    }
    assert!(parted > 0, "no ddsketch scored row sat on a tenths edge");
}

/// `corrchange` (`window` and `monitor`) under a clock `window_size` on a
/// tenths clock: weight-0 rows and `Bank::predict` rows read at their exact
/// stamp.
#[test]
fn corrchange_window_and_monitor_scored_rows_read_at_stamp() {
    let df = tenths_regression_frame(260, 606, true);
    let t: Vec<f64> = df
        .column("t")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let x0: Vec<f64> = df
        .column("x0")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let x1: Vec<f64> = df
        .column("x1")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let w: Vec<f64> = df
        .column("w")
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    for kind in ["window", "monitor"] {
        let mut parted = 0usize;
        for span in (1..25).map(|k| k as f64 / 10.0) {
            let extra = if kind == "window" {
                r#", "crit": 0.5"#
            } else {
                ""
            };
            let sp: Spec = serde_json::from_str(&format!(
                r#"{{"name": "m",
                    "model": {{"type": "corrchange", "kind": "{kind}", "window_size": {span}{extra}}},
                    "targets": [], "features": ["x0", "x1"],
                    "clock": "t", "weight": "w", "gap_cap": 100.0,
                    "min_weight": 0.0}}"#
            ))
            .unwrap();
            let (out, _) = run_bank(&sp, &df, 59);
            let got: Vec<Option<f64>> = named(&out, "stat").f64().unwrap().iter().collect();
            let cfg = CorrChangeCfg {
                n_features: 2,
                kind: if kind == "window" {
                    CorrChangeKind::Window
                } else {
                    CorrChangeKind::Monitor
                },
                span_rows: 0,
                alpha: 0.05,
                alpha_adjust: "bonferroni".into(),
                bandwidth: None,
                scalar: false,
                decay: Decay::Halflife(f64::INFINITY),
                crit: (kind == "window").then_some(0.5),
                n_perm: 200,
                permute_every_rows: 20,
                perm_block: 1,
                norm: ChangeNorm::LInf,
                seed: 0,
                reset: false,
                monitor_rows: 0,
                boundary_gamma: 0.0,
                permute_every: None,
                span_clock: Some(span),
                monitor_clock: None,
                max_state_bytes: None,
                emit_n_flagged: false,
            };
            let mut m = CorrChange::new(cfg).unwrap();
            let mut last: Option<f64> = None;
            let mut want = Vec::with_capacity(df.height());
            let mut bank_p = Bank::new(vec![sp.clone()]).unwrap();
            for i in 0..df.height() {
                let d = last.map_or(0.0, |l| t[i] - l);
                let stamp = Stamp::Raw(t[i], 0.0);
                let x = [x0[i], x1[i]];
                let step = if w[i] > 0.0 {
                    last = Some(t[i]);
                    m.stamp_next(stamp);
                    bank_p.fit_predict(&df.slice(i as i64, 1)).unwrap();
                    m.step(&x, &[], d, 1.0)
                } else {
                    let ahead = m.predict(&x, d);
                    let mut at_m = m.clone();
                    at_m.stamp_next(stamp);
                    let exact = at_m.step(&x, &[], d, 0.0);
                    parted += usize::from(ahead.pred[0].to_bits() != exact.pred[0].to_bits());
                    let pred_cols = bank_p.predict(&df.slice(i as i64, 1)).unwrap();
                    let pred_df = DataFrame::new(1, pred_cols).unwrap();
                    let p_val = pred_df
                        .column("m")
                        .unwrap()
                        .struct_()
                        .unwrap()
                        .field_by_name("stat")
                        .unwrap()
                        .f64()
                        .unwrap()
                        .get(0);
                    let want_opt = Some(exact.pred[0]).filter(|v| v.is_finite());
                    assert_eq!(
                        p_val.map(f64::to_bits),
                        want_opt.map(f64::to_bits),
                        "corrchange({kind}) Bank::predict span={span} row={i}"
                    );
                    exact
                };
                want.push(Some(step.pred[0]).filter(|v| v.is_finite()));
            }
            assert_eq!(
                opt_bits(&got),
                opt_bits(&want),
                "corrchange({kind}) weight-0 span={span}"
            );
        }
        assert!(
            parted > 0 || kind == "monitor",
            "no corrchange({kind}) scored row sat on a tenths edge"
        );
    }

    // Dedicated `corrchange(monitor)` tenths-edge test: 7 learned rows inside
    // `[0.0, 0.2]` (so `first = 0.0`, `last = 0.2`, span `0.9` not yet closed),
    // then a weight-0 row at `t = 0.9` (`d = 0.7`, `0.2 + 0.7 < 0.9` in f64):
    // exact stamp closes the monitor span (finite `stat`); read-ahead leaves
    // it open (`NaN`).
    let mut r = lcg(607);
    let ts = vec![0.0, 0.02, 0.05, 0.08, 0.11, 0.15, 0.18, 0.2, 0.9];
    let ws = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let x0s: Vec<f64> = (0..ts.len()).map(|_| r() * 2.0 - 1.0).collect();
    let x1s: Vec<f64> = (0..ts.len())
        .map(|k| x0s[k] * if k < 4 { 0.8 } else { -0.8 } + (r() - 0.5) * 0.2)
        .collect();
    let mdf = df!("t" => ts.clone(), "x0" => x0s.clone(), "x1" => x1s.clone(), "w" => ws).unwrap();
    let sp: Spec = serde_json::from_str(
        r#"{"name": "m",
            "model": {"type": "corrchange", "kind": "monitor", "window_size": 0.9},
            "targets": [], "features": ["x0", "x1"],
            "clock": "t", "weight": "w", "gap_cap": 100.0, "min_weight": 0.0}"#,
    )
    .unwrap();
    let (out, _) = run_bank(&sp, &mdf, ts.len());
    let stat_last = named(&out, "stat").f64().unwrap().get(ts.len() - 1);
    let mut m = CorrChange::new(CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Monitor,
        span_rows: 0,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(f64::INFINITY),
        crit: None,
        n_perm: 200,
        permute_every_rows: 20,
        perm_block: 1,
        norm: ChangeNorm::LInf,
        seed: 0,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
        permute_every: None,
        span_clock: Some(0.9),
        monitor_clock: None,
        max_state_bytes: None,
        emit_n_flagged: false,
    })
    .unwrap();
    for k in 0..ts.len() - 1 {
        let d = if k == 0 { 0.0 } else { ts[k] - ts[k - 1] };
        m.stamp_next(Stamp::Raw(ts[k], 0.0));
        m.step(&[x0s[k], x1s[k]], &[], d, 1.0);
    }
    let last_i = ts.len() - 1;
    let d_edge = ts[last_i] - ts[last_i - 1];
    let ahead = m.predict(&[x0s[last_i], x1s[last_i]], d_edge);
    let exact = m.predict_at(&[x0s[last_i], x1s[last_i]], d_edge, Stamp::Raw(0.9, 0.0));
    assert!(
        ahead.pred[0].is_nan(),
        "read-ahead must miss 0.2 + 0.7 < 0.9 monitor edge"
    );
    assert!(
        exact.pred[0].is_finite(),
        "exact stamp must close monitor span at 0.9"
    );
    assert_eq!(
        stat_last.map(f64::to_bits),
        Some(exact.pred[0].to_bits()),
        "corrchange(monitor) weight-0 row must match exact stamp"
    );
    let mut bank_p = Bank::new(vec![sp]).unwrap();
    bank_p.fit_predict(&mdf.slice(0, last_i)).unwrap();
    let p_cols = bank_p.predict(&mdf.slice(last_i as i64, 1)).unwrap();
    let p_df = DataFrame::new(1, p_cols).unwrap();
    let p_stat = p_df
        .column("m")
        .unwrap()
        .struct_()
        .unwrap()
        .field_by_name("stat")
        .unwrap()
        .f64()
        .unwrap()
        .get(0);
    assert_eq!(
        p_stat.map(f64::to_bits),
        Some(exact.pred[0].to_bits()),
        "corrchange(monitor) Bank::predict must match exact stamp"
    );
}

/// Temporal (`Datetime`) clock + `label_clock` + `window_size`: the arrival
/// clock's `Stamp::Ns` origin (`t_0`) differs from the learned labels'
/// `Stamp::Ns` origin (`l_0`). Scored rows (buffered, weight-0, and
/// `Bank::predict`) and ungated `predict_with_floor` reads must use the
/// `label_learned` clock's stamp, not the arrival clock's.
#[test]
fn label_clock_temporal_window_uses_learned_stamp_origin() {
    let n = 120;
    let mut r = lcg(707);
    let base_ms = 1_700_000_000_000i64;
    let mut t_ms = Vec::with_capacity(n);
    let mut l_ms = Vec::with_capacity(n);
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let t = base_ms + (i as i64) * 500;
        // 5-second delay so `l_0 - t_0 = 5s > window_size (3s)`: if a scored
        // row used the arrival clock's `Stamp::Ns` (relative to `t_0`) against
        // a window stamped on `label_learned` (relative to `l_0`), every
        // scored row would look 5s older than it is and empty the window!
        let l = t + 5_000;
        let a = r() * 2.0 - 1.0;
        let b = r() * 2.0 - 1.0;
        t_ms.push(t);
        l_ms.push(l);
        x0.push(a);
        x1.push(b);
        y.push(0.5 + 1.5 * a - 0.7 * b + (r() - 0.5) * 0.4);
        w.push(if i > 20 && i % 5 == 0 { 0.0 } else { 1.0 });
    }
    let df = df!(
        "t" => t_ms.clone(),
        "l" => l_ms.clone(),
        "x0" => x0.clone(),
        "x1" => x1.clone(),
        "y" => y.clone(),
        "w" => w.clone()
    )
    .unwrap()
    .lazy()
    .with_columns([
        col("t").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
        col("l").cast(DataType::Datetime(TimeUnit::Milliseconds, None)),
    ])
    .collect()
    .unwrap();

    for (kind, model_json, feats) in [
        (
            "expectile",
            r#"{"type": "expectile", "expectile": 0.7, "window_size": "3s", "closed": "right"}"#,
            r#"["x0", "x1"]"#,
        ),
        (
            "isotonic",
            r#"{"type": "isotonic", "window_size": "3s", "closed": "right", "out_of_bounds": "clip"}"#,
            r#"["x0"]"#,
        ),
    ] {
        let make = |mw: f64| -> Spec {
            let text = format!(
                r#"{{"name": "m", "model": {model_json},
                    "targets": ["y"], "features": {feats},
                    "clock": "t", "label_clock": "l", "weight": "w",
                    "half_life": "inf", "gap_cap": "1h",
                    "min_weight": {mw}, "emit_sigma": true}}"#
            );
            serde_json::from_str(&text).unwrap()
        };
        let sp0 = make(0.0);
        let sp_hi = make(100.0);
        let (base, base_bytes) = run_bank(&sp0, &df, 31);
        let (gated, gated_bytes) = run_bank(&sp_hi, &df, 31);
        assert!(
            withheld(&gated) > withheld(&base),
            "{kind}: gate withheld nothing"
        );
        assert_eq!(
            bits(named(&base, "sigma_y")),
            bits(named(&gated, "sigma_y")),
            "{kind} label_clock: sigma_y moved"
        );
        assert_eq!(
            spread_bytes(&base_bytes),
            spread_bytes(&gated_bytes),
            "{kind} label_clock: saved spread moved"
        );
        // Verify non-null predictions appear once labels release (if the
        // origin were mismatched by 5s > 3s, every scored row's window would
        // be trimmed empty and pred_y would stay null!).
        let preds: Vec<Option<f64>> = named(&base, "pred_y").f64().unwrap().iter().collect();
        let finite_after = preds[20..].iter().filter(|v| v.is_some()).count();
        assert!(
            finite_after > 80,
            "{kind} label_clock: expected >80 scored predictions after warmup, got {finite_after}"
        );
        // And `Bank::predict` at a weight-0 row (row 115, which releases no
        // held labels in `fit_predict`) must match `fit_predict` to the bit.
        let check_row = 115;
        assert_eq!(w[check_row], 0.0);
        let mut bank = Bank::new(vec![sp0]).unwrap();
        bank.fit_predict(&df.slice(0, check_row)).unwrap();
        let row = df.slice(check_row as i64, 1);
        let p_cols = bank.predict(&row).unwrap();
        let p_df = DataFrame::new(1, p_cols).unwrap();
        let p_val = p_df
            .column("m")
            .unwrap()
            .struct_()
            .unwrap()
            .field_by_name("pred_y")
            .unwrap()
            .f64()
            .unwrap()
            .get(0);
        assert!(
            p_val.is_some(),
            "{kind} label_clock: Bank::predict was null"
        );
        assert_eq!(
            p_val.map(f64::to_bits),
            preds[check_row].map(f64::to_bits),
            "{kind} label_clock: Bank::predict disagreed with fit_predict on weight-0 row"
        );
    }
}

/// Seasonal `holt` reads its raw clock via `clock_next` (set before the
/// ungated `predict_with_floor` read in `run_instance`): confirm `gate_isolation`
/// holds to the bit both plain and under `embargo`, with `emit_selected` and
/// `emit_averaged`.
#[test]
fn seasonal_holt_min_weight_gate_isolation() {
    let n = 200;
    let mut r = lcg(808);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    let mut clock = 0.0;
    for i in 0..n {
        if i > 0 {
            clock += 1.0 + (r() * 2.0).floor();
        }
        let phase = 2.0 * std::f64::consts::PI * (clock % 7.0) / 7.0;
        t.push(clock);
        y.push(10.0 + 0.05 * clock + 2.0 * phase.sin() + (r() - 0.5) * 0.3);
        w.push(if i % 15 == 14 { 0.0 } else { 1.0 });
    }
    let df = df!("t" => t, "y" => y, "w" => w).unwrap();
    for embargo in ["", r#", "embargo": 3.0"#] {
        let make = |mw: f64| -> Spec {
            let text = format!(
                r#"{{"name": "m",
                    "model": {{"type": "holt", "trend": true, "seasonal": "add", "seasonal_periods": 7, "period": 7.0, "start": "heuristic"}},
                    "targets": ["y"], "features": [],
                    "clock": "t", "weight": "w", "half_life": [30.0, 80.0], "gap_cap": 100.0,
                    "min_weight": {mw}, "emit_sigma": true, "emit_selected": true, "emit_averaged": true{embargo}}}"#
            );
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
        };
        let (base, base_bytes) = run_bank(&make(0.0), &df, 47);
        let (gated, gated_bytes) = run_bank(&make(22.0), &df, 47);
        assert!(
            withheld(&gated) > withheld(&base),
            "seasonal holt embargo={embargo}: gate withheld nothing"
        );
        for s in with_prefix(&base, "sigma_y") {
            assert_eq!(
                bits(s),
                bits(named(&gated, s.name())),
                "seasonal holt embargo={embargo}: {} moved",
                s.name()
            );
        }
        assert_eq!(
            spread_bytes(&base_bytes),
            spread_bytes(&gated_bytes),
            "seasonal holt embargo={embargo}: saved spread moved"
        );
        let slots = with_prefix(&gated, "pred_y");
        let open: Vec<bool> = (0..n)
            .map(|i| slots.iter().all(|s| s.f64().unwrap().get(i).is_some()))
            .collect();
        assert!(open.iter().filter(|&&e| e).count() > 50);
        for name in ["pred_y__selected", "pred_y__averaged"] {
            let (a, b) = (bits(named(&base, name)), bits(named(&gated, name)));
            for (i, &ok) in open.iter().enumerate() {
                if ok {
                    assert_eq!(a[i], b[i], "seasonal holt embargo={embargo}: {name} at {i}");
                }
            }
        }
    }
}

/// `hawkes` and `ewm_moments` under a clock `window_size` with `min_weight > 0`
/// on a tenths clock: weight-0 rows and `Bank::predict` rows read at their
/// exact stamp and respect `min_weight`.
#[test]
fn hawkes_and_ewm_moments_window_with_min_weight_read_at_stamp() {
    let mut r = lcg(909);
    let n = 220;
    let mut tenths = 0i64;
    let (mut t, mut e, mut x, mut w) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        tenths += 1 + (r() * 3.0).floor() as i64;
        t.push(tenths as f64 / 10.0);
        e.push(if r() < 0.65 { 1.0 } else { 0.0 });
        x.push(r() * 4.0 - 2.0);
        w.push(if i % 4 == 3 { 0.0 } else { 1.0 });
    }
    let df = df!("t" => t.clone(), "e" => e.clone(), "x" => x.clone(), "w" => w.clone()).unwrap();

    // Hawkes with min_weight > 0 and clock window.
    let sp_h: Spec = serde_json::from_str(
        r#"{"name": "h",
            "model": {"type": "hawkes", "kernel_decays": [0.8, 3.0], "window_size": 1.3, "closed": "right"},
            "targets": ["e"], "features": [],
            "clock": "t", "weight": "w", "half_life": "inf", "gap_cap": 100.0,
            "min_weight": 3.0}"#,
    )
    .unwrap();
    let (out_h, _) = run_bank(&sp_h, &df, 53);
    let got_h: Vec<Option<f64>> = named(&out_h, "intensity_e").f64().unwrap().iter().collect();
    let mut mh = Hawkes::new(HawkesCfg {
        n_targets: 1,
        kernel_decays: vec![0.8, 3.0],
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 3.0,
        window_events: None,
        window_clock: Some(1.3),
        closed: WindowClosed::Right,
        max_state_bytes: None,
    })
    .unwrap();
    let mut last: Option<f64> = None;
    let mut want_h = Vec::with_capacity(n);
    for i in 0..n {
        let d = last.map_or(0.0, |l| t[i] - l);
        let stamp = Stamp::Raw(t[i], 0.0);
        let step = if w[i] > 0.0 {
            last = Some(t[i]);
            mh.stamp_next(stamp);
            mh.step(&[], &[Some(e[i])], d, 1.0)
        } else {
            mh.predict_at(&[], d, stamp)
        };
        want_h.push(Some(step.pred[0]).filter(|v| v.is_finite()));
    }
    assert_eq!(opt_bits(&got_h), opt_bits(&want_h), "hawkes min_weight");

    // EwmMoments with min_weight > 0 and clock window.
    let sp_e: Spec = serde_json::from_str(
        r#"{"name": "em",
            "model": {"type": "ewm_moments", "terms": {"var": {"x": 2}}, "window_size": 1.3},
            "targets": [], "features": ["x"],
            "clock": "t", "weight": "w", "half_life": "inf", "gap_cap": 100.0,
            "min_weight": 3.0}"#,
    )
    .unwrap();
    let (out_e, _) = run_bank(&sp_e, &df, 53);
    let got_e: Vec<Option<f64>> = named(&out_e, "var").f64().unwrap().iter().collect();
    let mut me = EwmMoments::new(EwmMomentsCfg {
        decay: Decay::Halflife(f64::INFINITY),
        terms: vec![MomentTerm {
            name: "var".into(),
            powers: vec![(MomentLeg::of(0), 2)],
            kind: MomentKind::Central,
            bias: false,
        }],
        min_weight: 3.0,
        max_state_bytes: None,
        clocked: true,
        lag_tolerance: None,
        emit_se: false,
        window_rows: None,
        window_clock: Some(1.3),
    })
    .unwrap();
    let mut last: Option<f64> = None;
    let mut want_e = Vec::with_capacity(n);
    for i in 0..n {
        let d = last.map_or(0.0, |l| t[i] - l);
        let stamp = Stamp::Raw(t[i], 0.0);
        let step = if w[i] > 0.0 {
            last = Some(t[i]);
            me.stamp_next(stamp);
            me.step(&[x[i]], &[], d, 1.0)
        } else {
            me.predict_at(&[x[i]], d, stamp)
        };
        want_e.push(Some(step.pred[0]).filter(|v| v.is_finite()));
    }
    assert_eq!(
        opt_bits(&got_e),
        opt_bits(&want_e),
        "ewm_moments min_weight"
    );
}

/// Direct `AnyModel::predict_with_floor_at` contract on `Hawkes` and
/// `EwmMoments`: honors BOTH `floor` (bypassing `min_weight` withholding
/// without mutating state) AND exact `Stamp` on a tenths clock window edge
/// where read-ahead (`predict_with_floor`) sits on the wrong side of the edge.
#[test]
fn any_model_predict_with_floor_at_gates_hawkes_and_ewm_moments() {
    use rheon_polars::AnyModel;

    // 1. Hawkes: window_clock = 0.9, events at 0.0 and 0.2, read at 0.9
    //    (0.2 + 0.7 < 0.9 in f64; exact 0.9 - 0.0 == 0.9 drops row at 0.0).
    let mut h_hi = AnyModel::Hawkes(Box::new(
        Hawkes::new(HawkesCfg {
            n_targets: 1,
            kernel_decays: vec![0.8, 3.0],
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 50.0,
            window_events: None,
            window_clock: Some(0.9),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap(),
    ));
    let mut h_lo = AnyModel::Hawkes(Box::new(
        Hawkes::new(HawkesCfg {
            n_targets: 1,
            kernel_decays: vec![0.8, 3.0],
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
            window_events: None,
            window_clock: Some(0.9),
            closed: WindowClosed::Right,
            max_state_bytes: None,
        })
        .unwrap(),
    ));
    let events: &[(f64, f64)] = &[(0.0, 1.0), (0.05, 1.0), (0.1, 1.0), (0.15, 1.0), (0.2, 1.0)];
    let mut prev = 0.0;
    for (k, &(ts, ev)) in events.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - prev };
        prev = ts;
        let st = Stamp::Raw(ts, 0.0);
        h_hi.stamp_next(st);
        h_lo.stamp_next(st);
        h_hi.step(&[], &[Some(ev)], d, 1.0);
        h_lo.step(&[], &[Some(ev)], d, 1.0);
    }
    let d_edge = 0.9 - 0.2;
    let stamp_edge = Stamp::Raw(0.9, 0.0);
    let before = h_hi.state();
    // Gated read withholds (weight 5 < 50).
    let gated = h_hi.predict(&[], &[], d_edge);
    assert!(gated.pred[0].is_nan(), "hawkes min_weight=50 must withhold");
    // Ungated read-ahead (`predict_with_floor`) vs stamp-gated (`predict_with_floor_at`):
    let ahead = h_hi.predict_with_floor(&[], &[], d_edge, 0.0);
    let at_stamp = h_hi.predict_with_floor_at(&[], &[], d_edge, Some(stamp_edge), 0.0);
    let mut twin_step = rheon_core::Step::default();
    h_lo.predict_at_into(&[], &[], d_edge, Some(stamp_edge), &mut twin_step);
    assert_eq!(h_hi.state(), before, "hawkes state must not move");
    assert!(
        at_stamp.pred[0].is_finite(),
        "hawkes at_stamp must be finite"
    );
    assert_eq!(
        at_stamp.pred[0].to_bits(),
        twin_step.pred[0].to_bits(),
        "hawkes predict_with_floor_at must equal min_weight=0 twin at stamp"
    );
    assert_ne!(
        ahead.pred[0].to_bits(),
        at_stamp.pred[0].to_bits(),
        "hawkes read-ahead must diverge on 0.2 + 0.7 < 0.9 edge"
    );

    // 2. EwmMoments: window_clock = 0.9, rows at 0.0, 0.1, 0.2, read at 0.9.
    let make_em = |mw: f64| {
        AnyModel::EwmMoments(Box::new(
            EwmMoments::new(EwmMomentsCfg {
                decay: Decay::Halflife(f64::INFINITY),
                terms: vec![MomentTerm {
                    name: "var".into(),
                    powers: vec![(MomentLeg::of(0), 2)],
                    kind: MomentKind::Central,
                    bias: false,
                }],
                min_weight: mw,
                max_state_bytes: None,
                clocked: true,
                lag_tolerance: None,
                emit_se: false,
                window_rows: None,
                window_clock: Some(0.9),
            })
            .unwrap(),
        ))
    };
    let mut e_hi = make_em(50.0);
    let mut e_lo = make_em(0.0);
    let rows: &[(f64, f64)] = &[(0.0, 100.0), (0.05, 1.0), (0.1, 2.0), (0.2, 3.0)];
    let mut prev = 0.0;
    for (k, &(ts, v)) in rows.iter().enumerate() {
        let d = if k == 0 { 0.0 } else { ts - prev };
        prev = ts;
        let st = Stamp::Raw(ts, 0.0);
        e_hi.stamp_next(st);
        e_lo.stamp_next(st);
        e_hi.step(&[v], &[], d, 1.0);
        e_lo.step(&[v], &[], d, 1.0);
    }
    let before = e_hi.state();
    let gated = e_hi.predict(&[0.0], &[], d_edge);
    assert!(
        gated.pred[0].is_nan(),
        "ewm_moments min_weight=50 must withhold"
    );
    let ahead = e_hi.predict_with_floor(&[0.0], &[], d_edge, 0.0);
    let at_stamp = e_hi.predict_with_floor_at(&[0.0], &[], d_edge, Some(stamp_edge), 0.0);
    let mut twin_step = rheon_core::Step::default();
    e_lo.predict_at_into(&[0.0], &[], d_edge, Some(stamp_edge), &mut twin_step);
    assert_eq!(e_hi.state(), before, "ewm_moments state must not move");
    assert!(
        at_stamp.pred[0].is_finite(),
        "ewm_moments at_stamp must be finite"
    );
    assert_eq!(
        at_stamp.pred[0].to_bits(),
        twin_step.pred[0].to_bits(),
        "ewm_moments predict_with_floor_at must equal min_weight=0 twin at stamp"
    );
    assert_ne!(
        ahead.pred[0].to_bits(),
        at_stamp.pred[0].to_bits(),
        "ewm_moments read-ahead must diverge on 0.2 + 0.7 < 0.9 edge"
    );
}
