//! A run's output buffers kept from an earlier run are not filled with NaN
//! again where the run writes them in full (`ChunkOut::renew_run`): what a
//! row it does not process leaves in them is never read. A chunk fed to a
//! bank holding buffers a full chunk of finite values left must give what
//! the same chunk gives a bank holding none, to the bit, rows skipped for a
//! missing feature included.
use crate::stream::ChunkOut;
use crate::{Bank, Spec};
use polars::prelude::*;

/// Spec text for the cases: grouped or not, wide or narrow, one target or
/// several, a statistic model with no target, a model that withholds rows,
/// a spec whose rows are held back (every row of which the buffers are
/// filled for, as before).
const SPECS: &[&str] = &[
    r#"{"name": "a", "model": {"type": "ew_cov"}, "features": ["x0", "x1", "x2", "x3"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0}"#,
    r#"{"name": "b", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1", "x2"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "min_weight": 8.0, "emit_sigma": true, "emit_zscore": true}"#,
    r#"{"name": "c", "model": {"type": "ewridge"}, "targets": ["y", "z"], "features": ["x0", "x1"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "group": "g", "emit_sigma": true, "conformal": 0.9}"#,
    r#"{"name": "d", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"],
        "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "embargo": 3.0, "emit_sigma": true}"#,
    r#"{"name": "e", "model": {"type": "seqtest"}, "targets": ["y"], "features": [],
        "clock": "t", "gap_cap": 100.0, "group": "g"}"#,
    r#"{"name": "f", "model": {"type": "ew_cov", "stats": ["mean", "cov", "corr"]},
        "features": ["x0", "x1", "x2"], "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "group": "g"}"#,
];

/// `n` rows from row `from`; `holes` makes every `holes`-th row's `x0`
/// missing (the bank skips it), 0 for none.
fn frame(from: usize, n: usize, holes: usize) -> DataFrame {
    let at = |i: usize| from + i;
    let x = |i: usize, c: usize| ((at(i) * (7 + 2 * c)) % 23) as f64 - 11.0 + 0.25 * c as f64;
    df!(
        "t" => (0..n).map(|i| at(i) as f64 * 1.5).collect::<Vec<_>>(),
        "g" => (0..n).map(|i| (at(i) * 5 % 3) as i64).collect::<Vec<_>>(),
        "x0" => (0..n).map(|i| {
            if holes > 0 && at(i) % holes == 1 { f64::NAN } else { x(i, 0) }
        }).collect::<Vec<_>>(),
        "x1" => (0..n).map(|i| x(i, 1)).collect::<Vec<_>>(),
        "x2" => (0..n).map(|i| x(i, 2)).collect::<Vec<_>>(),
        "x3" => (0..n).map(|i| x(i, 3)).collect::<Vec<_>>(),
        "y" => (0..n).map(|i| 0.5 * x(i, 0) - x(i, 1) + ((at(i) * 13) % 7) as f64).collect::<Vec<_>>(),
        "z" => (0..n).map(|i| x(i, 2) + ((at(i) * 3) % 5) as f64).collect::<Vec<_>>()
    )
    .unwrap()
}

fn bank_of(text: &str) -> Bank {
    let spec: Spec = serde_json::from_str(text).unwrap();
    let mut bank = Bank::new(vec![spec]).unwrap();
    bank.set_threads(Some(1)).unwrap();
    bank
}

#[test]
fn buffers_kept_from_a_run_of_finite_rows_change_nothing() {
    for text in SPECS {
        // The same chunks, to a bank that keeps its runs' buffers and to
        // one that keeps none: the second chunk of each skips rows.
        let (mut kept, mut fresh) = (bank_of(text), bank_of(text));
        let mut reused = false;
        for (from, n, holes) in [
            (0, 60, 0),
            (60, 60, 3),
            (120, 17, 2),
            (137, 60, 0),
            (197, 40, 4),
        ] {
            let df = frame(from, n, holes);
            reused |= !kept.spares.outs.is_empty();
            fresh.spares = super::Spares::default();
            let a = kept.fit_predict(&df).unwrap();
            let b = fresh.fit_predict(&df).unwrap();
            assert_eq!(a.len(), b.len());
            for (ca, cb) in a.iter().zip(&b) {
                assert!(
                    ca.as_materialized_series()
                        .equals_missing(cb.as_materialized_series()),
                    "{text}: rows {from}.. differ with buffers kept"
                );
            }
        }
        assert!(
            reused,
            "{text}: no run's buffers were kept, so nothing was tested"
        );
        assert_eq!(
            kept.save_bytes().unwrap(),
            fresh.save_bytes().unwrap(),
            "{text}: the state moved"
        );
    }
}

/// The buffers a run does not fill again are the ones it writes in full:
/// kept for a spec with nothing held back, filled for any other.
#[test]
fn only_a_spec_with_nothing_held_back_keeps_its_buffers() {
    let dense: Vec<bool> = SPECS
        .iter()
        .map(|text| ChunkOut::keeps_buffers(&serde_json::from_str(text).unwrap()))
        .collect();
    assert_eq!(dense, [true, true, true, false, true, true]);
}

/// Review 459, C10: the specs above, each its own shape, in one bank, with a
/// weight whose first row is 0 (the clock advances and nothing is learned)
/// and a spec that resets on drift, which a level shift in the target from
/// row 300 sets off (`seqtest` takes no weight, and runs without one): the
/// bank keeping its runs' buffers gives what one keeping none gives, to the
/// bit, and the same state.
#[test]
fn buffers_kept_across_several_specs_of_different_shapes_change_nothing() {
    let weighted = |text: &str| {
        let mut v: serde_json::Value = serde_json::from_str(text).unwrap();
        // `seqtest` counts trials and takes no weight.
        if v["model"]["type"] != "seqtest" {
            v["weight"] = serde_json::json!("w");
        }
        v
    };
    let mut specs: Vec<serde_json::Value> = SPECS.iter().map(|t| weighted(t)).collect();
    specs.push(weighted(
        r#"{"name": "r", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0", "x1"],
            "clock": "t", "gap_cap": 100.0, "half_life": 30.0, "min_weight": 4.0,
            "emit_drift": true, "drift_threshold": 2.0, "drift_action": "reset"}"#,
    ));
    let bank = || {
        let specs: Vec<Spec> = specs
            .iter()
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .collect();
        let mut b = Bank::new(specs).unwrap();
        b.set_threads(Some(1)).unwrap();
        b
    };
    let weighted_frame = |from: usize, n: usize, holes: usize| {
        let mut df = frame(from, n, holes);
        let w: Vec<f64> = (0..n)
            .map(|i| {
                if from + i == 0 {
                    0.0
                } else {
                    1.0 + ((from + i) % 3) as f64 * 0.5
                }
            })
            .collect();
        df.with_column(Column::new("w".into(), w)).unwrap();
        // A level shift in the target from row 300, for the drift to find.
        let y: Vec<f64> = df
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .enumerate()
            .map(|(i, v)| if from + i >= 300 { v + 40.0 } else { v })
            .collect();
        df.with_column(Column::new("y".into(), y)).unwrap();
        df
    };
    let (mut kept, mut fresh) = (bank(), bank());
    let mut reused = false;
    let mut reset = false;
    for (from, n, holes) in [
        (0, 60, 0),
        (60, 60, 3),
        (120, 17, 2),
        (137, 60, 0),
        (197, 40, 4),
        (237, 200, 0),
        (437, 200, 0),
    ] {
        let df = weighted_frame(from, n, holes);
        reused |= !kept.spares.outs.is_empty();
        fresh.spares = super::Spares::default();
        let a = kept.fit_predict(&df).unwrap();
        let b = fresh.fit_predict(&df).unwrap();
        assert_eq!(a.len(), b.len());
        for (ca, cb) in a.iter().zip(&b) {
            assert!(
                ca.as_materialized_series()
                    .equals_missing(cb.as_materialized_series()),
                "rows {from}..: column {} differs with buffers kept",
                ca.name()
            );
            if ca.name().as_str() == "r" {
                let fields = ca.struct_().unwrap().fields_as_series();
                // A row flagged as drift is a reset, under `drift_action`.
                for d in fields.iter().filter(|f| f.name().starts_with("drift")) {
                    reset |= d.bool().is_ok_and(|c| c.any());
                }
            }
        }
    }
    assert!(reused, "no run's buffers were kept, so nothing was tested");
    assert!(
        reset,
        "the drift reset never fired, so its path was not tested"
    );
    assert_eq!(
        kept.save_bytes().unwrap(),
        fresh.save_bytes().unwrap(),
        "the state moved"
    );
}
