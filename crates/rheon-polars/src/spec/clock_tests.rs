//! `Spec`'s clock tests, kept in a file of their own: `spec.rs` reached the
//! repository's 250 KB cap for a source file (`tests/test_repo_hygiene.py`)
//! with task 223 (b)'s `audit`.

use super::{CLOCK_FIELDS, ClockScale, ModelKind, Spec};
use crate::span::Span;
use std::collections::BTreeSet;

/// The fields serde knows for one JSON object, read from its
/// unknown-field message: the one list of them outside the type.
fn fields_of<T: serde::de::DeserializeOwned>(json: &str) -> Vec<String> {
    let err = match serde_json::from_str::<T>(json) {
        Ok(_) => panic!("{json} has no unknown field"),
        Err(e) => e.to_string(),
    };
    let expected = err.split("expected").nth(1).unwrap_or("");
    expected
        .split('`')
        .skip(1)
        .step_by(2)
        .map(str::to_string)
        .collect()
}

/// Whether a field reads a duration as clock units: it takes `"10m"`
/// (at most another field is then missing) and refuses a word that is
/// no duration *as* no duration, which a plain text field would take.
fn takes_duration<T: serde::de::DeserializeOwned>(json: impl Fn(&str) -> String) -> bool {
    let fits = match serde_json::from_str::<T>(&json("\"10m\"")) {
        Ok(_) => true,
        Err(e) => e.to_string().starts_with("missing field"),
    };
    let refuses = serde_json::from_str::<T>(&json("\"nonsense\""))
        .err()
        .is_some_and(|e| e.to_string().contains("is not a duration"));
    fits && refuses
}

#[test]
fn clock_fields_are_exactly_the_fields_that_take_a_duration() {
    let mut found: BTreeSet<(String, String)> = BTreeSet::new();
    let base = r#""name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"]"#;
    for f in fields_of::<Spec>(&format!("{{{base}, \"zzz\": 1}}")) {
        if takes_duration::<Spec>(|v| format!("{{{base}, \"{f}\": {v}}}")) {
            found.insert(("*".into(), f));
        }
    }
    for kind in ModelKind::KINDS {
        for f in fields_of::<ModelKind>(&format!(r#"{{"type": "{kind}", "zzz": 1}}"#)) {
            if takes_duration::<ModelKind>(|v| format!(r#"{{"type": "{kind}", "{f}": {v}}}"#)) {
                found.insert((kind.to_string(), f));
            }
        }
    }
    let table = |t: &[(&str, &[&str])]| -> BTreeSet<(String, String)> {
        t.iter()
            .flat_map(|(owner, fields)| fields.iter().map(|f| (owner.to_string(), f.to_string())))
            .collect()
    };
    let clock = table(CLOCK_FIELDS);
    assert!(
        found.len() > 15,
        "the walk found too little to mean anything: {found:?}"
    );
    assert_eq!(
        found, clock,
        "CLOCK_FIELDS against the fields that take a duration"
    );
}

/// `bocpd`'s two hazards (task 547): `hazard` is a clock parameter like
/// every other -- a number is clock units, rows without a clock, so it
/// binds a spec to a number clock and is refused beside a duration -- and
/// `hazard_rows` is a count of rows, which binds the spec to no clock and
/// stands beside durations. Task 179 read a number `hazard` as rows on
/// every clock; that reading is `hazard_rows` now.
#[test]
fn bocpds_hazard_is_a_clock_parameter_and_hazard_rows_a_count() {
    let build = |model: &str, rest: &str| -> Spec {
        let mut spec: Spec = serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "bocpd", {model}}}, "features": ["x"]{rest}}}"#
        ))
        .unwrap_or_else(|e| panic!("{model}: {e}"));
        spec.fill_defaults();
        spec
    };
    let temporal = r#", "clock": "t", "gap_cap": "1h""#;
    let number = r#", "clock": "t", "gap_cap": 300"#;
    let read = |s: &Spec| -> Vec<(&'static str, Span)> {
        s.clock_spans()
            .into_iter()
            .filter(|(f, _)| *f == "hazard")
            .collect()
    };
    // A duration: read, and bound to a temporal clock.
    let d = build(r#""hazard": "10m""#, temporal);
    assert!(read(&d).iter().all(|(_, s)| s.is_duration()) && read(&d).len() == 1);
    d.validate().unwrap();
    // A number: read as clock units, so it binds the spec to numbers.
    let n = build(r#""hazard": 250"#, number);
    assert_eq!(read(&n).len(), 1);
    assert_eq!(n.clock_scale(), Ok(ClockScale::Numbers("gap_cap")));
    n.validate().unwrap();
    assert_eq!(
        build(r#""hazard": 250"#, "").clock_scale(),
        Ok(ClockScale::Numbers("hazard"))
    );
    let err = build(r#""hazard": 250"#, temporal).validate().unwrap_err();
    assert!(err.contains("hazard is a plain number"), "{err}");
    let err = build(r#""hazard": "10m""#, "").validate().unwrap_err();
    assert!(
        err.contains("hazard is a duration, but the spec has no clock"),
        "{err}"
    );
    // A count of rows binds nothing: beside durations, numbers, or none.
    for rest in [temporal, number, ""] {
        let rows = build(r#""hazard_rows": 250"#, rest);
        assert!(read(&rows).is_empty(), "{rest}");
        rows.validate().unwrap_or_else(|e| panic!("{rest}: {e}"));
    }
    assert_eq!(
        build(r#""hazard_rows": 250"#, temporal).clock_scale(),
        Ok(ClockScale::Durations("gap_cap"))
    );
}

/// A spec whose one clock parameter under test is a duration.
fn with_duration(owner: &str, field: &str) -> Spec {
    let model = match owner {
        "*" => r#"{"type": "ewridge"}"#.to_string(),
        kind => {
            let required = match kind {
                "lasso" => r#", "penalties": [0.1]"#,
                "kalman" if field != "coef_half_life" => r#", "coef_half_life": "1h""#,
                "quantile" => r#", "quantile": 0.5"#,
                "expectile" => r#", "expectile": 0.5"#,
                "ew_class" => r#", "classes": ["a", "b"], "precision_prior": 1.0"#,
                "micro" => r#", "eps": 0.3"#,
                "kmeans" => r#", "n_clusters": 2"#,
                "hmm" => r#", "n_components": 2, "precision_prior": 1.0"#,
                "corrchange" => r#", "kind": "window", "window_rows": 20"#,
                "power_variation" => r#", "terms": [{"name": "rv", "powers": [2.0]}]"#,
                "ddsketch" => r#", "quantiles": [0.25]"#,
                "ewm_moments" => r#", "terms": {"m2": {"x": 2}}"#,
                "ar" => r#", "lags": 2"#,
                "spectral" if field != "periods" => r#", "periods": "1h""#,
                _ => "",
            };
            format!(r#"{{"type": "{kind}", "{field}": "10m"{required}}}"#)
        }
    };
    let own = if owner == "*" {
        format!(r#", "{field}": "10m""#)
    } else {
        String::new()
    };
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x"],
            "clock": "t"{own}}}"#
    ))
    .unwrap_or_else(|e| panic!("{owner}.{field}: {e}"))
}

#[test]
fn every_clock_field_is_walked() {
    for (owner, fields) in CLOCK_FIELDS {
        for &field in *fields {
            let spec = with_duration(owner, field);
            assert!(
                spec.clock_spans()
                    .iter()
                    .any(|(f, s)| *f == field && s.is_duration()),
                "{owner}.{field} is in CLOCK_FIELDS but clock_spans does not read it"
            );
        }
    }
}

fn spec(extra: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x"]{extra}}}"#
    ))
    .unwrap()
}

#[test]
fn a_spec_is_numbers_durations_or_neither() {
    let durations = spec(r#", "clock": "t", "half_life": "10m", "gap_cap": "5m""#);
    assert_eq!(
        durations.clock_scale(),
        Ok(ClockScale::Durations("half_life"))
    );
    let numbers = spec(r#", "clock": "t", "half_life": 600, "gap_cap": 300"#);
    assert_eq!(numbers.clock_scale(), Ok(ClockScale::Numbers("half_life")));
    // 0 and inf mean the same in every unit, so they bind a spec to neither.
    let free = spec(r#", "clock": "t", "half_life": "inf""#);
    assert_eq!(free.clock_scale(), Ok(ClockScale::Free));
    let beside = spec(r#", "clock": "t", "half_life": "inf", "gap_cap": "5m""#);
    assert_eq!(beside.clock_scale(), Ok(ClockScale::Durations("gap_cap")));
    assert!(durations.validate().is_ok());
}

#[test]
fn a_mixture_is_refused_naming_both_parameters() {
    let err = spec(r#", "clock": "t", "half_life": "10m", "gap_cap": 300"#)
        .validate()
        .unwrap_err();
    assert!(
        err.contains("half_life is a duration") && err.contains("gap_cap is a plain number"),
        "{err}"
    );
    let err = spec(r#", "clock": "t", "lam": 0.99, "gap_cap": "5m""#)
        .validate()
        .unwrap_err();
    assert!(err.contains("lam is a decay over one clock unit"), "{err}");
}

#[test]
fn a_duration_needs_a_clock() {
    let err = spec(r#", "half_life": "10m""#).validate().unwrap_err();
    assert!(
        err.contains("half_life is a duration, but the spec has no clock"),
        "{err}"
    );
}

#[test]
fn a_duration_is_read_in_seconds_and_names_a_grid_as_written() {
    let s = spec(r#", "clock": "t", "half_life": ["5m", "1h"], "gap_cap": "1d""#);
    let decays = s.decays().unwrap();
    let labels: Vec<&str> = decays.iter().map(|(l, _)| l.as_str()).collect();
    assert_eq!(labels, vec!["@h5m", "@h1h"]);
    assert_eq!(
        decays[0].1,
        rheon_core::Decay::Halflife(300.0),
        "a duration is read in seconds"
    );
    let dup = spec(r#", "clock": "t", "half_life": ["5m", "300s"], "gap_cap": "1d""#);
    let err = dup.decays().unwrap_err();
    assert!(
        err.contains("lists 5m and 300s, which are the same half-life"),
        "{err}"
    );
    assert!(matches!(
        s.half_life.as_ref().unwrap().spans()[0],
        Span::Duration(_)
    ));
}

/// Task 160, PB2: the cap is on the clock's step, and without a clock a
/// row is one step. A cap of 0.5 was taken and halved every decay step,
/// marked every row capped and cleared every lag at every row.
#[test]
fn gap_cap_needs_a_clock() {
    let err = spec(r#", "half_life": 10, "gap_cap": 0.5"#)
        .check()
        .unwrap_err();
    assert!(err.contains("spec \"m\": gap_cap needs clock"), "{err}");
    assert!(
        spec(r#", "clock": "t", "half_life": 10, "gap_cap": 0.5"#)
            .check()
            .is_ok()
    );
}

/// Task 173, PC1: a window target under `group` is one window core per
/// group on that group's clock, which without a clock column counts
/// its own rows, so a group that falls silent leaves its windows open
/// with nothing to cut them. Refused with `with_windows`' message, the
/// rule stated once (`clock_cfg_of`); with a clock, or without groups,
/// the same spec is a spec.
#[test]
fn a_formula_target_with_group_needs_a_clock() {
    let formula = |extra: &str| {
        serde_json::from_str::<Spec>(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "features": ["x"],
                "targets": [{{"name": "fwd", "formula": ["-", ["rewm_mean", ["col", "mid"],
                    {{"half_life": 5, "window_size": 10}}], ["col", "mid"]]}}],
                "half_life": 50, "embargo": 10{extra}}}"#
        ))
        .unwrap()
    };
    let err = formula(r#", "group": "g""#).check().unwrap_err();
    assert!(
        err.starts_with("spec \"m\": a formula target under group needs clock")
            && err.ends_with("Set clock and gap_cap, or leave group out."),
        "{err}"
    );
    assert_eq!(
        formula(r#", "group": "g", "clock": "t", "gap_cap": 100"#).check(),
        Ok(())
    );
    assert_eq!(formula("").check(), Ok(()));
    // A plain target under groups and no clock is a spec as before.
    assert_eq!(spec(r#", "half_life": 10, "group": "g""#).check(), Ok(()));
}

/// Task 160, PB3: an empty half-life grid built no model instance.
#[test]
fn an_empty_half_life_grid_is_refused() {
    let err = spec(r#", "half_life": []"#).check().unwrap_err();
    assert!(
        err.contains("spec \"m\": half_life must list at least one half-life, but it is empty"),
        "{err}"
    );
}

/// The drift detector sums each row's excess times its clock step, so
/// its threshold is sigma times clock units: a number of the clock
/// column's units, a duration on a temporal clock, and required with a
/// clock, as `gap_cap` is; 20 without one, where a row is one unit and
/// it is the classic test (task 168). The default 20 on a temporal clock
/// was 20 sigma-seconds, a unit no one chose, and fired on noise at a
/// row a minute (review 2026-10-05, CC1).
#[test]
fn drift_threshold_is_a_clock_parameter() {
    let durations = r#", "clock": "t", "half_life": "10m", "gap_cap": "5m", "emit_drift": true"#;
    let err = spec(durations).validate().unwrap_err();
    assert!(
        err.contains("spec \"m\": drift_threshold must be set when the spec has a clock"),
        "{err}"
    );
    let given = spec(&format!(r#"{durations}, "drift_threshold": "20m""#));
    assert_eq!(given.validate(), Ok(()));
    assert_eq!(
        given.drift_threshold.as_ref().map(Span::value),
        Some(1200.0),
        "a duration is read on the clock's scale, seconds"
    );
    let err = spec(&format!(r#"{durations}, "drift_threshold": 20"#))
        .validate()
        .unwrap_err();
    assert!(err.contains("drift_threshold is a plain number"), "{err}");
    let numbers = r#", "clock": "t", "half_life": 600, "gap_cap": 300, "emit_drift": true"#;
    assert_eq!(
        spec(&format!(r#"{numbers}, "drift_threshold": 20"#)).validate(),
        Ok(())
    );
    let err = spec(&format!(r#"{numbers}, "drift_threshold": "20m""#))
        .validate()
        .unwrap_err();
    assert!(err.contains("drift_threshold is a duration"), "{err}");
    for bad in ["0", "-1", "\"inf\""] {
        let err = spec(&format!(r#"{numbers}, "drift_threshold": {bad}"#))
            .validate()
            .unwrap_err();
        assert!(
            err.contains("drift_threshold must be finite and above 0, but it is"),
            "{bad}: {err}"
        );
    }
    // Without a clock a row is one unit: 20 is the classic test.
    assert_eq!(
        spec(r#", "half_life": 600, "emit_drift": true"#).validate(),
        Ok(())
    );
}

/// `coef_every` reads the clock, as `solve_every` does (task 178): a
/// number of the clock column's units or a duration on a temporal clock,
/// the mixtures refused as for every clock parameter; `0` every row and
/// free of a unit; finite and `>= 0`. `max_rows_between_coefs` is at
/// least 1, and both need a model with coefficients, `0` included, which
/// was the default and passed there.
#[test]
fn coef_every_is_a_clock_parameter_and_its_row_cap_a_count() {
    let durations = r#", "clock": "t", "half_life": "10m", "gap_cap": "5m""#;
    let given = spec(&format!(r#"{durations}, "coef_every": "15m""#));
    assert_eq!(given.validate(), Ok(()));
    assert_eq!(given.coef_every.as_ref().map(Span::value), Some(900.0));
    let err = spec(&format!(r#"{durations}, "coef_every": 900"#))
        .validate()
        .unwrap_err();
    assert!(err.contains("coef_every is a plain number"), "{err}");
    // `0` means the same in every unit, so a temporal spec takes it.
    assert_eq!(
        spec(&format!(r#"{durations}, "coef_every": 0"#)).validate(),
        Ok(())
    );
    let numbers = r#", "clock": "t", "half_life": 600, "gap_cap": 300"#;
    assert_eq!(
        spec(&format!(r#"{numbers}, "coef_every": 900"#)).validate(),
        Ok(())
    );
    let err = spec(&format!(r#"{numbers}, "coef_every": "15m""#))
        .validate()
        .unwrap_err();
    assert!(err.contains("coef_every is a duration"), "{err}");
    for bad in ["-1", "-0.5", "\"inf\""] {
        let err = spec(&format!(r#"{numbers}, "coef_every": {bad}"#))
            .validate()
            .unwrap_err();
        assert!(
            err.contains("coef_every must be finite and at least 0 clock units, but it is"),
            "{bad}: {err}"
        );
    }
    let err = spec(r#", "half_life": 60, "max_rows_between_coefs": 0"#)
        .validate()
        .unwrap_err();
    assert!(
        err.contains("max_rows_between_coefs must be at least 1, but it is 0."),
        "{err}"
    );
    assert_eq!(
        spec(r#", "half_life": 60, "coef_every": 7, "max_rows_between_coefs": 1"#).validate(),
        Ok(())
    );
    for cadence in [r#""coef_every": 0"#, r#""max_rows_between_coefs": 5"#] {
        let cov: Spec = serde_json::from_str(&format!(
            r#"{{"name": "c", "model": {{"type": "ew_cov"}}, "targets": ["x"],
                "features": ["x", "z"], "half_life": 60, {cadence}}}"#
        ))
        .unwrap();
        let err = cov.validate().unwrap_err();
        let key = cadence.split('"').nth(1).unwrap();
        assert!(
            err.contains(&format!(
                "{key} applies only to a model that reports coefficients, but ew_cov reports none"
            )),
            "{err}"
        );
    }
}

/// Task 160, PB9: no delay is no embargo, not an embargo of 0.
#[test]
fn a_zero_embargo_is_told_to_leave_it_out() {
    let err = spec(r#", "half_life": 10, "embargo": 0"#)
        .check()
        .unwrap_err();
    assert!(
        err.contains("embargo must be finite and above 0, but it is 0.")
            && err.contains("or leave it out for no delay."),
        "{err}"
    );
}
