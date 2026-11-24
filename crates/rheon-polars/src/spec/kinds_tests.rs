use super::ModelKind;

/// serde's unknown-variant error names every variant the enum has -- the
/// one place that list exists outside the enum itself.
#[test]
fn kinds_lists_every_variant_in_order() {
    let err = serde_json::from_str::<ModelKind>(r#"{"type": "nope"}"#)
        .unwrap_err()
        .to_string();
    let quoted: Vec<&str> = err.split('`').skip(1).step_by(2).collect();
    assert_eq!(quoted[0], "nope", "{err}");
    assert_eq!(
        &quoted[1..],
        ModelKind::KINDS,
        "ModelKind::KINDS is out of date"
    );
}

/// A key no spec has -- a typo in a TOML -- is refused, naming it and the
/// keys there are, rather than left at its default without a word. Both
/// levels: the spec's own keys and the model's.
#[test]
fn an_unknown_key_is_refused_at_either_level() {
    let spec = serde_json::from_str::<super::Spec>(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "halflfe": 10}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(
        spec.contains("unknown field `halflfe`") && spec.contains("`half_life`"),
        "{spec}"
    );
    let model = serde_json::from_str::<ModelKind>(r#"{"type": "ewridge", "rigde": 0.1}"#)
        .unwrap_err()
        .to_string();
    assert!(
        model.contains("unknown field `rigde`") && model.contains("`ridge`"),
        "{model}"
    );
}

/// Task 544: a window given to a `tdigest` is refused naming the windowed
/// DDSketch, from JSON and from TOML; the same unknown key on another model
/// is told nothing of it.
#[test]
fn a_tdigest_window_is_refused_naming_the_windowed_ddsketch() {
    let json = serde_json::from_str::<super::Spec>(
        r#"{"name": "m", "model": {"type": "tdigest", "quantiles": [0.25], "window_size": 20},
            "targets": ["y"], "features": []}"#,
    )
    .unwrap_err()
    .to_string();
    let json = super::name_renamed(&json);
    assert!(
        json.contains("window_size does not apply to tdigest")
            && json.contains("ewm_quantile with a window_size"),
        "{json}"
    );
    let toml = toml::from_str::<super::Spec>(
        "name = \"m\"\ntargets = [\"y\"]\n[model]\ntype = \"tdigest\"\n\
         quantiles = [0.5]\nwindow_rows = 10\n",
    )
    .unwrap_err()
    .to_string();
    let toml = super::name_renamed(&toml);
    assert!(
        toml.contains("window_rows does not apply to tdigest"),
        "{toml}"
    );
    assert!(toml.contains("the windowed DDSketch"), "{toml}");
    let other = serde_json::from_str::<super::Spec>(
        r#"{"name": "m", "model": {"type": "garch", "window_size": 20},
            "targets": ["y"], "features": []}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(!super::name_renamed(&other).contains("DDSketch"));
}

/// Task 144: an old name is refused naming the new one, from JSON and
/// from TOML, at the spec's level and the model's.
#[test]
fn an_old_name_is_refused_naming_the_new_one() {
    let json = serde_json::from_str::<super::Spec>(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
            "features": ["x"], "halflife": 10}"#,
    )
    .unwrap_err()
    .to_string();
    let json = super::name_renamed(&json);
    assert!(json.contains("halflife was renamed half_life"), "{json}");
    let toml = toml::from_str::<super::Spec>(
        "name = \"m\"\ntargets = [\"y\"]\nfeatures = [\"x\"]\nhalf_life = 10.0\n\
         [model]\ntype = \"ewridge\"\nwindow = 10.0\n",
    )
    .unwrap_err()
    .to_string();
    let toml = super::name_renamed(&toml);
    assert!(toml.contains("window was renamed window_size"), "{toml}");
    for (old, new) in super::RENAMED {
        let msg = super::name_renamed(&format!("unknown field `{old}`, expected one of `x`"));
        assert!(msg.contains(&format!("{old} was renamed {new}")), "{msg}");
    }
    assert_eq!(
        super::name_renamed("unknown field `rigde`"),
        "unknown field `rigde`"
    );
    // `rls`'s `ridge` is `delta` (task 195, N11): named where the model
    // takes `delta`, and not where another model refuses a `ridge` it
    // never had.
    let rls = serde_json::from_str::<super::Spec>(
        r#"{"name": "m", "model": {"type": "rls", "ridge": 1.0}, "targets": ["y"],
            "features": ["x"], "half_life": 10}"#,
    )
    .unwrap_err()
    .to_string();
    let rls = super::name_renamed(&rls);
    assert!(rls.contains("ridge was renamed delta"), "{rls}");
    let sgd = serde_json::from_str::<super::Spec>(
        r#"{"name": "m", "model": {"type": "sgd", "ridge": 1.0}, "targets": ["y"],
            "features": ["x"], "half_life": 10}"#,
    )
    .unwrap_err()
    .to_string();
    let sgd = super::name_renamed(&sgd);
    assert!(
        sgd.contains("unknown field `ridge`") && !sgd.contains("renamed"),
        "{sgd}"
    );
}
