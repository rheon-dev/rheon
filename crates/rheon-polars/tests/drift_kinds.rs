//! A bank's drift detectors of the kinds `drift_kind` adds (PLAN 508), as a
//! file holds them: a detector whose shape its rows cannot run on is
//! refused as damaged when the bank loads, not run.

use polars::prelude::*;
use rheon_polars::{Bank, Spec};
use rmpv::Value;

fn spec(kind: &str, options: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "rls"}}, "targets": ["y"], "features": ["x"],
            "half_life": 50, "emit_drift": true, "drift_kind": "{kind}",
            "drift_options": {options}}}"#
    ))
    .unwrap()
}

fn frame(n: usize) -> DataFrame {
    let x: Vec<f64> = (0..n)
        .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, x)| 2.0 * x + ((i * 31) % 17) as f64 / 40.0)
        .collect();
    df!("x" => x, "y" => y).unwrap()
}

/// An edit of a detector's map.
type Edit<'a> = &'a dyn Fn(&mut Vec<(Value, Value)>);

/// The bank's file with `edit` applied to the detector's map (the value
/// under its kind's tag), sealed again.
fn edited(bytes: &[u8], tag: &str, edit: Edit) -> Vec<u8> {
    fn walk(v: &mut Value, tag: &str, edit: Edit, n: &mut usize) {
        match v {
            Value::Map(m) => {
                for (k, v) in m.iter_mut() {
                    if k.as_str() == Some(tag)
                        && let Value::Map(inner) = v
                    {
                        edit(inner);
                        *n += 1;
                    } else {
                        walk(v, tag, edit, n);
                    }
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|v| walk(v, tag, edit, n)),
            _ => {}
        }
    }
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    let mut n = 0;
    walk(&mut v, tag, edit, &mut n);
    assert_eq!(n, 1, "one detector of the kind");
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    rheon_polars::reseal(&out)
}

fn set(m: &mut [(Value, Value)], key: &str, to: Value) {
    let slot = m
        .iter_mut()
        .find(|(k, _)| k.as_str() == Some(key))
        .unwrap_or_else(|| panic!("no {key}"));
    slot.1 = to;
}

#[test]
fn a_detector_its_rows_cannot_run_on_is_refused_at_the_load() {
    let df = frame(300);
    for (kind, options, tag, key, bad) in [
        (
            "adwin",
            r#"{"clock": 2}"#,
            "Adwin",
            "levels",
            Value::Array(vec![]),
        ),
        ("adwin", r#"{"clock": 2}"#, "Adwin", "clock", Value::from(0)),
        (
            "kswin",
            r#"{"window_size": 40, "stat_size": 10}"#,
            "Kswin",
            "stat_size",
            Value::from(30),
        ),
    ] {
        let s = spec(kind, options);
        let mut bank = Bank::new(vec![s.clone()]).unwrap();
        bank.fit_predict(&df).unwrap();
        let bytes = bank.save_bytes().unwrap();
        // The file as saved, read and written again, loads.
        let same = edited(&bytes, tag, &|_| {});
        Bank::load_bytes(&same, Some(std::slice::from_ref(&s))).unwrap();
        let damaged = edited(&bytes, tag, &|m| set(m, key, bad.clone()));
        let err = Bank::load_bytes(&damaged, Some(std::slice::from_ref(&s)))
            .err()
            .unwrap_or_else(|| panic!("{kind} with {key} = {bad} loaded"));
        assert!(
            err.contains("drift detectors do not match this spec"),
            "{kind}, {key}: {err}"
        );
    }
}
