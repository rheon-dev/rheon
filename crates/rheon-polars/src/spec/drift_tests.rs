//! `drift_kind` and `drift_options` (`spec/drift.rs`): each refusal, each
//! bound on both sides, and the detector each kind builds.

use rheon_core::DriftDetector;

use super::Spec;

fn spec(rest: &str) -> Spec {
    serde_json::from_str(&format!(
        r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
            "features": ["x"], "half_life": 10, "emit_drift": true{rest}}}"#
    ))
    .unwrap_or_else(|e| panic!("{rest}: {e}"))
}

fn refused(rest: &str) -> String {
    spec(rest).check().expect_err(rest)
}

fn passes(rest: &str) {
    spec(rest).check().unwrap_or_else(|e| panic!("{rest}: {e}"));
}

#[test]
fn a_kind_is_one_of_the_names() {
    for k in ["page_hinkley", "cusum", "adwin", "kswin"] {
        passes(&format!(r#", "drift_kind": "{k}""#));
    }
    let err = refused(r#", "drift_kind": "hddm""#);
    assert!(
        err.contains("drift_kind") && err.contains("\"adwin\""),
        "{err}"
    );
    // DDM and EDDM need their error.
    for k in ["ddm", "eddm"] {
        let err = refused(&format!(r#", "drift_kind": "{k}""#));
        assert!(err.contains("error_above is not set"), "{err}");
        passes(&format!(
            r#", "drift_kind": "{k}", "drift_options": {{"error_above": 0}}"#
        ));
    }
}

#[test]
fn a_knob_without_its_switch_or_its_kind_is_refused() {
    let off = |rest: &str| {
        serde_json::from_str::<Spec>(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x"], "half_life": 10{rest}}}"#
        ))
        .unwrap()
        .check()
        .unwrap_err()
    };
    for (rest, key) in [
        (r#", "drift_kind": "adwin""#, "drift_kind"),
        (r#", "drift_options": {"delta": 0.1}"#, "drift_options"),
    ] {
        let err = off(rest);
        assert!(
            err.contains(&format!("{key} applies only when emit_drift is on")),
            "{err}"
        );
    }
    for (kind, key) in [
        ("adwin", "drift_delta"),
        ("kswin", "drift_threshold"),
        ("ddm", "drift_delta"),
    ] {
        let err = refused(&format!(
            r#", "drift_kind": "{kind}", "{key}": 1, "drift_options": {{"error_above": 1}}"#
        ));
        assert!(
            err.contains(&format!(
                "{key} applies to drift_kind \"page_hinkley\" and \"cusum\""
            )),
            "{err}"
        );
    }
    // A key the kind does not read, and one no kind reads.
    let err = refused(r#", "drift_kind": "adwin", "drift_options": {"alpha": 0.1, "seed": 2}"#);
    assert!(
        err.contains("drift_options gives alpha, seed, which drift_kind \"adwin\" does not read"),
        "{err}"
    );
    let err = refused(r#", "drift_kind": "cusum", "drift_options": {"delta": 0.1}"#);
    assert!(err.contains("\"cusum\" reads none"), "{err}");
    let err = refused(r#", "drift_options": {"delta": 0.1}"#);
    assert!(err.contains("\"page_hinkley\" reads none"), "{err}");
    assert!(
        serde_json::from_str::<Spec>(
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"],
                "half_life": 10, "emit_drift": true, "drift_options": {"nope": 1}}"#
        )
        .is_err()
    );
}

/// Every bound, each side of it.
#[test]
fn each_setting_is_held_to_its_range_on_both_sides() {
    let adwin = |o: &str| format!(r#", "drift_kind": "adwin", "drift_options": {{{o}}}"#);
    let kswin = |o: &str| format!(r#", "drift_kind": "kswin", "drift_options": {{{o}}}"#);
    let ddm =
        |o: &str| format!(r#", "drift_kind": "ddm", "drift_options": {{"error_above": 1, {o}}}"#);
    let eddm =
        |o: &str| format!(r#", "drift_kind": "eddm", "drift_options": {{"error_above": 1, {o}}}"#);
    for (bad, good) in [
        (adwin(r#""delta": 0"#), adwin(r#""delta": 1e-9"#)),
        (adwin(r#""delta": 1"#), adwin(r#""delta": 0.999"#)),
        (adwin(r#""clock": 0"#), adwin(r#""clock": 1"#)),
        (adwin(r#""max_buckets": 0"#), adwin(r#""max_buckets": 1"#)),
        (adwin(r#""max_buckets": 65"#), adwin(r#""max_buckets": 64"#)),
        (
            adwin(r#""min_window_length": 0"#),
            adwin(r#""min_window_length": 1"#),
        ),
        (adwin(r#""max_width": 0"#), adwin(r#""max_width": 1"#)),
        (
            adwin(r#""max_state_bytes": 0"#),
            adwin(r#""max_state_bytes": 4000"#),
        ),
        (
            adwin(r#""max_state_bytes": 67108865"#),
            adwin(r#""max_state_bytes": 67108864"#),
        ),
        (kswin(r#""alpha": 0"#), kswin(r#""alpha": 0.5"#)),
        (kswin(r#""alpha": 1"#), kswin(r#""alpha": 0.999"#)),
        (kswin(r#""stat_size": 0"#), kswin(r#""stat_size": 1"#)),
        (
            kswin(r#""window_size": 59, "stat_size": 30"#),
            kswin(r#""window_size": 60, "stat_size": 30"#),
        ),
        (
            ddm(r#""drift_threshold": 0"#),
            ddm(r#""drift_threshold": 0.1"#),
        ),
        (eddm(r#""beta": 0"#), eddm(r#""beta": 0.1"#)),
        (eddm(r#""beta": 1.0001"#), eddm(r#""beta": 1"#)),
        (
            r#", "drift_kind": "ddm", "drift_options": {"error_above": -1}"#.to_string(),
            r#", "drift_kind": "ddm", "drift_options": {"error_above": 0}"#.to_string(),
        ),
    ] {
        let err = refused(&bad);
        assert!(err.contains("drift_options"), "{bad}: {err}");
        passes(&good);
    }
}

/// The detectors' bytes in a group, against `max_state_bytes`, at the cap
/// and one byte past it: a KSWIN ring of `W` rows is `8 W + 48` bytes a
/// slot, and two targets are two slots.
#[test]
fn the_bytes_are_held_to_the_cap_at_the_call() {
    // 8 * 32762 + 48 = 262,144, the default cap exactly.
    passes(r#", "drift_kind": "kswin", "drift_options": {"window_size": 32762}"#);
    let err = refused(r#", "drift_kind": "kswin", "drift_options": {"window_size": 32763}"#);
    assert!(
        err.contains("262152 bytes in each group, past max_state_bytes of 262144"),
        "{err}"
    );
    passes(
        r#", "drift_kind": "kswin", "drift_options": {"window_size": 32763, "max_state_bytes": 262152}"#,
    );
    let two = |w: u64| {
        serde_json::from_str::<Spec>(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y", "z"],
                "features": ["x"], "half_life": 10, "emit_drift": true, "drift_kind": "kswin",
                "drift_options": {{"window_size": {w}, "max_state_bytes": 2000}}}}"#
        ))
        .unwrap()
        .check()
    };
    // Two slots of 8 * 119 + 48 = 1,000 bytes.
    assert!(two(119).is_ok());
    assert!(two(120).unwrap_err().contains("on each of 2 slots"));
    // ADWIN at its max_width: 1,976 bytes at a million rows, 2,080 at the
    // next level.
    passes(r#", "drift_kind": "adwin", "drift_options": {"max_state_bytes": 1976}"#);
    let err = refused(r#", "drift_kind": "adwin", "drift_options": {"max_state_bytes": 1975}"#);
    assert!(err.contains("Lower max_width"), "{err}");
}

/// A threshold is required with a clock where the kind sums over the clock,
/// and only there.
#[test]
fn a_clock_needs_a_threshold_only_for_the_clocked_kinds() {
    let clocked = |rest: &str| {
        serde_json::from_str::<Spec>(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x"], "half_life": 10, "clock": "t", "gap_cap": 100,
                "emit_drift": true{rest}}}"#
        ))
        .unwrap()
        .check()
    };
    for kind in ["page_hinkley", "cusum"] {
        let err = clocked(&format!(r#", "drift_kind": "{kind}""#)).unwrap_err();
        assert!(err.contains("drift_threshold must be set"), "{err}");
        clocked(&format!(
            r#", "drift_kind": "{kind}", "drift_threshold": 3"#
        ))
        .unwrap();
    }
    clocked(r#", "drift_kind": "adwin""#).unwrap();
    clocked(r#", "drift_kind": "eddm", "drift_options": {"error_above": 1}"#).unwrap();
}

/// What each kind builds, its defaults among them: a CUSUM's `h` is 5
/// unless set, a Page-Hinkley's threshold 20.
#[test]
fn each_kind_builds_its_detector() {
    let built = |rest: &str| {
        let mut s = spec(rest);
        s.check().unwrap();
        s.drift_detector().unwrap()
    };
    assert_eq!(
        built(""),
        DriftDetector::PageHinkley(rheon_core::PageHinkley::new(0.5, 20.0))
    );
    assert_eq!(
        built(r#", "drift_kind": "cusum""#),
        DriftDetector::Cusum(rheon_core::Cusum::new(0.5, 5.0))
    );
    assert_eq!(
        built(r#", "drift_kind": "cusum", "drift_delta": 1, "drift_threshold": 4"#),
        DriftDetector::Cusum(rheon_core::Cusum::new(1.0, 4.0))
    );
    assert_eq!(
        built(r#", "drift_kind": "adwin""#),
        DriftDetector::Adwin(rheon_core::Adwin::new(rheon_core::AdwinCfg::default()))
    );
    assert_eq!(
        built(
            r#", "drift_kind": "adwin", "drift_options": {"delta": 0.01, "clock": 4, "max_buckets": 3, "min_window_length": 2, "grace_period": 7, "max_width": 99, "restart": false}"#
        ),
        DriftDetector::Adwin(rheon_core::Adwin::new(rheon_core::AdwinCfg {
            delta: 0.01,
            clock: 4,
            max_buckets: 3,
            min_window_length: 2,
            grace_period: 7,
            max_width: 99,
            restart: false,
        }))
    );
    assert_eq!(
        built(
            r#", "drift_kind": "kswin", "drift_options": {"alpha": 0.01, "window_size": 50, "stat_size": 10, "seed": 9, "restart": false}"#
        ),
        DriftDetector::Kswin(rheon_core::Kswin::new(rheon_core::KswinCfg {
            alpha: 0.01,
            window_size: 50,
            stat_size: 10,
            seed: 9,
            restart: false,
        }))
    );
    assert_eq!(
        built(r#", "drift_kind": "ddm", "drift_options": {"error_above": 2}"#),
        DriftDetector::Ddm(rheon_core::ErrorRate {
            error_above: 2.0,
            detector: rheon_core::Ddm::new(30, 3.0)
        })
    );
    assert_eq!(
        built(
            r#", "drift_kind": "ddm", "drift_options": {"error_above": 2, "warm_start": 5, "drift_threshold": 2}"#
        ),
        DriftDetector::Ddm(rheon_core::ErrorRate {
            error_above: 2.0,
            detector: rheon_core::Ddm::new(5, 2.0)
        })
    );
    assert_eq!(
        built(
            r#", "drift_kind": "eddm", "drift_options": {"error_above": 2, "beta": 0.8, "warm_start": 4}"#
        ),
        DriftDetector::Eddm(rheon_core::ErrorRate {
            error_above: 2.0,
            detector: rheon_core::Eddm::new(4, 0.8)
        })
    );
    assert_eq!(
        built(r#", "drift_kind": "eddm", "drift_options": {"error_above": 2}"#),
        DriftDetector::Eddm(rheon_core::ErrorRate {
            error_above: 2.0,
            detector: rheon_core::Eddm::new(30, 0.9)
        })
    );
    // A spec without the switch keeps none.
    let mut s: Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"], "half_life": 10}"#,
    )
    .unwrap();
    s.check().unwrap();
    assert!(s.drift_detector().is_none());
}

/// A spec that names no kind writes no new key: its bytes are those it
/// had.
#[test]
fn a_spec_without_a_kind_writes_neither_key() {
    let mut s = spec("");
    s.check().unwrap();
    let text = serde_json::to_string(&s).unwrap();
    assert!(
        !text.contains("drift_kind") && !text.contains("drift_options"),
        "{text}"
    );
    let mut s = spec(r#", "drift_kind": "adwin", "drift_options": {"clock": 3}"#);
    s.check().unwrap();
    let text = serde_json::to_string(&s).unwrap();
    assert!(text.contains(r#""drift_options":{"clock":3}"#), "{text}");
}
