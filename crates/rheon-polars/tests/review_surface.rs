//! The 2026-11-23 review's batch i (PLAN 578i): the spec's own checks, the
//! names a spec carries into the Arrow export, and the refusals that blamed
//! the wrong key. Each test failed on 51671e2e for the review's reason, or
//! says where it did not.

use polars::prelude::*;
use rheon_polars::{Bank, Spec};

/// A spec from JSON, filled, validated and built, as every door does it.
fn check(json: &str) -> Result<(), String> {
    let mut s: Spec = serde_json::from_str(json).map_err(|e| format!("did not parse: {e}"))?;
    s.check()
}

/// A supervised spec over `y` from `x0` and `x1`, with `extra` keys.
fn sup(model: &str, extra: &str) -> String {
    format!(
        r#"{{"name": "m", "model": {model}, "targets": ["y"], "features": ["x0", "x1"],
            "half_life": 10.0{extra}}}"#
    )
}

/// A spec of a model with no target over `x0`, `x1` and `x2`.
fn unsup(model: &str, extra: &str) -> String {
    format!(r#"{{"name": "m", "model": {model}, "features": ["x0", "x1", "x2"]{extra}}}"#)
}

#[track_caller]
fn refused(json: &str, want: &[&str]) -> String {
    let e = check(json).expect_err(&format!("accepted: {json}"));
    for w in want {
        assert!(e.contains(w), "wanted {w:?} in: {e}");
    }
    e
}

#[track_caller]
fn accepted(json: &str) {
    check(json).unwrap_or_else(|e| panic!("refused: {e}\n{json}"));
}

/// SPEC-1: a NUL in a feature, a feature set, a `deco` block or an
/// `ew_class` class reached a field name of the Arrow export, which panicked
/// on it after the bank had learned the chunk. Each is refused when the spec
/// is checked, as the spec's and a target's name were; so is a NUL in any
/// column a spec names.
#[test]
fn a_nul_in_any_name_a_spec_carries_is_refused() {
    let nul = "NUL character";
    // A feature: its name is in `coef`'s terms and the feature health's
    // fields.
    refused(
        &sup(r#"{"type": "ewridge"}"#, "").replace(r#""x1""#, r#""x\u0000a""#),
        &[nul, "x\\0a"],
    );
    // A feature set: `pred_y__s\0a`.
    refused(
        &sup(
            r#"{"type": "ewridge", "feature_sets": [["s\u0000a", ["x0"]], ["s2", ["x1"]]]}"#,
            "",
        ),
        &[nul],
    );
    // A block of `deco`: `u_b\0a`.
    refused(
        &unsup(
            r#"{"type": "deco", "blocks": [["b\u0000a", ["x0", "x1"]], ["c", ["x2", "x3"]]]}"#,
            r#", "half_life": 10.0"#,
        )
        .replace(r#""x1", "x2"]"#, r#""x1", "x2", "x3"]"#),
        &[nul],
    );
    // A class of `ew_class`: `p_a\0b`.
    refused(
        &sup(
            r#"{"type": "ew_class", "classes": ["a\u0000b", "c"], "precision_prior": 1.0}"#,
            "",
        ),
        &[nul],
    );
    // The columns a spec reads by name.
    for (key, value) in [
        ("clock", r#""t\u0000""#),
        ("group", r#""g\u0000""#),
        ("weight", r#""w\u0000""#),
    ] {
        let extra = format!(r#", "{key}": {value}"#);
        refused(&sup(r#"{"type": "ewridge"}"#, &extra), &[nul, key]);
    }
    accepted(&sup(
        r#"{"type": "ewridge", "feature_sets": [["s1", ["x0"]], ["s2", ["x1"]]]}"#,
        "",
    ));
}

/// SPEC-4: `kmeans`' `update_every` and `split_merge_every` and
/// `corrchange`'s `permute_every` took `inf`, which no JSON of the spec can
/// say back (`resolved_defaults` refused the spec as a defect), where
/// `pca_every` and `prune_every`, the same kind of cadence, refuse it.
#[test]
fn an_infinite_cadence_is_refused_by_name() {
    for key in ["update_every", "split_merge_every"] {
        let model = format!(r#"{{"type": "kmeans", "n_clusters": 2, "{key}": "inf"}}"#);
        refused(
            &unsup(
                &model,
                r#", "half_life": 10.0, "clock": "t", "gap_cap": 1000.0"#,
            ),
            &[key, "finite"],
        );
        let model = format!(r#"{{"type": "kmeans", "n_clusters": 2, "{key}": 5.0}}"#);
        accepted(&unsup(
            &model,
            r#", "half_life": 10.0, "clock": "t", "gap_cap": 1000.0"#,
        ));
    }
    refused(
        &unsup(
            r#"{"type": "corrchange", "kind": "window", "window_size": 20.0, "permute_every": "inf"}"#,
            r#", "clock": "t", "gap_cap": 1000.0"#,
        ),
        &["permute_every", "finite"],
    );
    accepted(&unsup(
        r#"{"type": "corrchange", "kind": "window", "window_size": 20.0, "permute_every": 5.0}"#,
        r#", "clock": "t", "gap_cap": 1000.0"#,
    ));
}

/// SPEC-5: a refit's row count given beside its span is refused whatever
/// the count, 0 among them: the span ran and the count of 0, which says
/// "no refit", was ignored.
#[test]
fn a_refit_count_beside_its_span_is_refused_at_zero_too() {
    let clocked = r#", "clock": "t", "gap_cap": 1000.0"#;
    for (model, rows, span) in [
        (r#""type": "glm""#, "refit_rows", "refit_span"),
        (
            r#""type": "expectile", "expectile": 0.8"#,
            "refit_rows",
            "refit_span",
        ),
        (r#""type": "huber""#, "huber_refit_rows", "huber_refit_span"),
        (
            r#""type": "quantile", "quantile": 0.5"#,
            "quantile_refit_rows",
            "quantile_refit_span",
        ),
        (
            r#""type": "sgd", "loss": "huber""#,
            "huber_refit_rows",
            "huber_refit_span",
        ),
    ] {
        for n in [0, 5] {
            let m = format!(r#"{{{model}, "{rows}": {n}, "{span}": 10.0}}"#);
            refused(&sup(&m, clocked), &[rows, span, "Leave one of them out"]);
        }
        let m = format!(r#"{{{model}, "{span}": 10.0}}"#);
        accepted(&sup(&m, clocked));
        let m = format!(r#"{{{model}, "{rows}": 0}}"#);
        accepted(&sup(&m, clocked));
    }
}

/// SPEC-6: a feature named `intercept` is refused only beside a fit whose
/// own constant term carries that name. Six kinds with no such term refused
/// it; `pa` and `ar`, which have one, are refused as the others are.
#[test]
fn intercept_is_refused_only_where_the_fit_names_its_own_constant() {
    let features = r#""features": ["intercept", "x1", "x2"]"#;
    let unsup_i = |model: &str, extra: &str| {
        unsup(model, extra).replace(r#""features": ["x0", "x1", "x2"]"#, features)
    };
    let sup_i = |model: &str| {
        sup(model, "").replace(
            r#""features": ["x0", "x1"]"#,
            r#""features": ["intercept", "x1"]"#,
        )
    };
    for (model, extra) in [
        (r#"{"type": "audit"}"#, ""),
        (r#"{"type": "bocpd"}"#, ""),
        (r#"{"type": "corrchange", "window_rows": 20}"#, ""),
        (r#"{"type": "deco"}"#, r#", "half_life": 10.0"#),
        (
            r#"{"type": "rcov", "block_rows": 100}"#,
            r#", "clock": "t", "gap_cap": 1000.0, "group": "g", "session": "s", "group_close": "session""#,
        ),
    ] {
        accepted(&unsup_i(model, extra));
    }
    accepted(&sup_i(r#"{"type": "marginal"}"#));
    for model in [r#"{"type": "pa"}"#, r#"{"type": "ar", "lags": [1]}"#] {
        refused(&sup_i(model), &["features lists \"intercept\""]);
    }
}

/// SPEC-7: `sgd` under a loss other than `"huber"` takes neither
/// `huber_refit_span` nor `max_state_bytes`, and the spec's own checks say
/// so beside its other parameters of the Huber loss; and a refit asked for
/// by its span is named by its span where the cut makes it least squares.
#[test]
fn sgds_huber_span_and_cap_are_the_huber_losss() {
    for (key, value) in [("huber_refit_span", "10.0"), ("max_state_bytes", "100000")] {
        let json = sup(
            &format!(r#"{{"type": "sgd", "loss": "squared", "{key}": {value}}}"#),
            r#", "clock": "t", "gap_cap": 1000.0"#,
        );
        let s: Spec = serde_json::from_str(&json).unwrap();
        let e = s
            .validate()
            .expect_err(&format!("{key}: validate accepted"));
        assert!(
            e.contains(&format!("{key} applies only under loss=\"huber\"")),
            "{e}"
        );
    }
    for (model, span) in [
        (
            r#"{"type": "sgd", "loss": "huber", "huber_delta": "inf", "huber_refit_span": 10.0}"#,
            "huber_refit_span",
        ),
        (
            r#"{"type": "huber", "huber_delta": "inf", "huber_refit_span": 10.0}"#,
            "huber_refit_span",
        ),
    ] {
        let e = refused(&sup(model, r#", "clock": "t", "gap_cap": 1000.0"#), &[span]);
        assert!(!e.contains("huber_refit_rows refits"), "{e}");
    }
}

/// SPEC-10: a level listed twice in `resid_quantiles` or `mahal_quantiles`,
/// and a kernel listed twice in `hawkes`' `kernel_half_lives`, are refused
/// by their own key, where the first two reached the output's duplicate
/// field and the third blamed `kernel_decays`, which was not given. A
/// half-life whose decay is past the doubles (1e-310; the review's 1e-308 gives
/// a decay of 6.9e307, which is finite, and is taken) is refused by its key
/// too.
#[test]
fn a_repeated_level_or_kernel_is_refused_by_its_own_key() {
    refused(
        &sup(
            r#"{"type": "ewridge"}"#,
            r#", "resid_quantiles": [0.5, 0.5]"#,
        ),
        &["resid_quantiles", "more than once"],
    );
    refused(
        &unsup(
            r#"{"type": "ew_cov", "stats": ["mean", "mahal"], "mahal_quantiles": [0.99, 0.99]}"#,
            r#", "half_life": 10.0"#,
        ),
        &["mahal_quantiles", "more than once"],
    );
    let hawkes = |hl: &str| {
        format!(
            r#"{{"name": "m", "model": {{"type": "hawkes", "kernel_half_lives": {hl}}},
                "targets": ["y"], "features": [], "clock": "t", "gap_cap": "1d", "half_life": "1h"}}"#
        )
    };
    let e = refused(&hawkes(r#"["5m", "300s"]"#), &["kernel_half_lives"]);
    assert!(!e.contains("kernel_decays must"), "{e}");
    let e = refused(
        &hawkes("[1e-310]")
            .replace(r#""half_life": "1h""#, r#""half_life": 10.0"#)
            .replace(r#""gap_cap": "1d""#, r#""gap_cap": 1000.0"#),
        &["kernel_half_lives"],
    );
    assert!(!e.contains("kernel_decays must"), "{e}");
    accepted(&hawkes(r#"["5m", "10m"]"#));
}

/// SPEC-10 (3): a `refit_budget` where the spec runs no refit bounds
/// nothing, and is refused, as `marginal`'s is under the mean: on `glm` and
/// `expectile`, which no release has carried.
#[test]
fn a_refit_budget_with_no_refit_is_refused() {
    for model in [
        r#"{"type": "glm", "refit_rows": 0, "refit_budget": 8.0}"#,
        r#"{"type": "expectile", "expectile": 0.8, "refit_rows": 0, "refit_budget": 8.0}"#,
        r#"{"type": "expectile", "expectile": 0.8, "window_rows": 50, "refit_budget": 8.0}"#,
    ] {
        refused(&sup(model, ""), &["refit_budget", "no refit"]);
    }
    for model in [
        r#"{"type": "huber", "refit_budget": 8.0}"#,
        r#"{"type": "quantile", "quantile": 0.5, "refit_budget": 8.0}"#,
        r#"{"type": "sgd", "loss": "huber", "refit_budget": 8.0}"#,
        r#"{"type": "glm", "refit_budget": 8.0}"#,
        r#"{"type": "expectile", "expectile": 0.8, "refit_budget": 8.0}"#,
    ] {
        accepted(&sup(model, ""));
    }
}

/// SPEC-10 (3), on the kinds 2.0.0 released: `huber`, `quantile` and
/// `sgd(loss="huber")` still take a `refit_budget` with no refit, because a
/// refusal would refuse to load a 2.0.0 state saved with such a spec. Left
/// for the user (`REVIEW-2026-11-23-OUTCOMES.md`, batch i); this test fails
/// once the refusal lands, and is then turned into a plain one.
#[test]
#[should_panic(expected = "accepted")]
fn a_refit_budget_with_no_refit_on_a_released_kind_is_refused() {
    for model in [
        r#"{"type": "huber", "huber_refit_rows": 0, "refit_budget": 8.0}"#,
        r#"{"type": "huber", "huber_delta": "inf", "refit_budget": 8.0}"#,
        r#"{"type": "quantile", "quantile": 0.5, "quantile_refit_rows": 0, "refit_budget": 8.0}"#,
        r#"{"type": "sgd", "loss": "huber", "huber_refit_rows": 0, "refit_budget": 8.0}"#,
    ] {
        refused(&sup(model, ""), &["refit_budget", "no refit"]);
    }
}

/// SPEC-9 (2): a bank file of another model is told by its `type`, where
/// the first differing leaf in key order (`coef_prior`) was named.
#[test]
fn specs_of_two_models_differ_at_the_type() {
    let lasso: Spec =
        serde_json::from_str(&sup(r#"{"type": "lasso", "penalties": [0.1]}"#, "")).unwrap();
    let ewridge: Spec = serde_json::from_str(&sup(r#"{"type": "ewridge"}"#, "")).unwrap();
    let bytes = Bank::new(vec![lasso]).unwrap().save_bytes().unwrap();
    let e = Bank::load_bytes(&bytes, Some(&[ewridge]))
        .err()
        .expect("loaded");
    assert!(e.contains("differs at model.type"), "{e}");
    assert!(
        e.contains(r#""lasso" in the file, "ewridge" in the bank"#),
        "{e}"
    );
}

/// SPEC-11: a formula target with no operator looking ahead is refused
/// naming every one that does, `rewm_quantile` and the reversed rolling
/// extremes among them.
#[test]
fn the_formula_refusal_names_every_operator_looking_ahead() {
    let json = sup(
        r#"{"type": "ewridge"}"#,
        r#", "clock": "t", "gap_cap": 1000.0, "embargo": 10.0"#,
    )
    .replace(
        r#""targets": ["y"]"#,
        r#""targets": [{"name": "f", "formula": ["ewm_mean", ["col", "y"], {"half_life": 5.0}]}]"#,
    );
    let e = serde_json::from_str::<Spec>(&json).unwrap_err().to_string();
    for op in [
        "rewm_mean",
        "rewm_sum",
        "rewm_rate",
        "rewm_quantile",
        "rrolling_max",
        "rrolling_min",
    ] {
        assert!(e.contains(op), "{op} not named: {e}");
    }
}

/// GLM-6: a formula target of a `glm` resolved outside its family's support
/// is refused naming the row, as a column target is, and the chunk is
/// refused whole. It was skipped by the model, which still counted its
/// weight in `weight_sum`.
#[test]
fn a_glm_formula_target_outside_its_support_is_refused_by_row() {
    let n = 40;
    let n_u32: u32 = 40;
    let t: Vec<f64> = (0..n).map(f64::from).collect();
    let x: Vec<f64> = (0..n).map(|i| f64::from(i % 7) / 7.0).collect();
    // A level that rises for 20 rows and then falls, so its forward mean
    // less the row's own is above 0 and then below.
    let mid: Vec<f64> = (0..n_u32)
        .map(|i| 100.0 + f64::from(i.min(20)) - f64::from(i.saturating_sub(20)))
        .collect();
    let df = df!("t" => t, "x" => x, "mid" => mid).unwrap();
    let formula = r#"["-", ["rewm_mean", ["col", "mid"], {"half_life": 2.0, "window_size": 4.0}], ["col", "mid"]]"#;
    let spec = |tree: &str| -> Spec {
        serde_json::from_str(&format!(
            r#"{{"name": "m", "model": {{"type": "glm", "family": "poisson", "refit_rows": 0}},
                "targets": [{{"name": "fwd", "formula": {tree}}}], "features": ["x"],
                "clock": "t", "gap_cap": 1000.0, "half_life": 50.0, "embargo": 4.0}}"#
        ))
        .unwrap()
    };
    let mut bank = Bank::new(vec![spec(formula)]).unwrap();
    bank.fit_predict(&df.slice(0, 15)).unwrap();
    let before = bank.save_bytes().unwrap();
    let e = bank
        .fit_predict_from(&df.slice(15, 25), 15)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("formula target \"fwd\" resolved to -")
            && e.contains("family=\"poisson\" takes a count"),
        "{e}"
    );
    assert_eq!(
        bank.save_bytes().unwrap(),
        before,
        "the refused chunk moved the bank"
    );
    // The same formula, shifted into the support, is learned.
    let inside = format!(r#"["+", {formula}, ["lit", 1000.0]]"#);
    let mut bank = Bank::new(vec![spec(&inside)]).unwrap();
    bank.fit_predict(&df).unwrap();
}

/// SPEC-9 (1): `hawkes`, whose intensity decays over the clock since the
/// newest learned row, and `bocpd` under a clock `hazard`, whose chance of a
/// break grows over it, are refused under `label_clock`, as `holt`,
/// `kalman`, `hmm` and `micro` are: under it that row can be far older than
/// the row predicted. `bocpd` counting rows (`hazard_rows`) is taken.
#[test]
fn label_clock_refuses_the_kinds_whose_prediction_reads_the_learned_clock() {
    let clocked = r#", "clock": "t", "gap_cap": 1000.0, "label_clock": "avail""#;
    refused(
        &format!(
            r#"{{"name": "m", "model": {{"type": "hawkes", "kernel_decays": [1.0]}},
                "targets": ["y"], "features": [], "half_life": 10.0{clocked}}}"#
        ),
        &["label_clock does not work with hawkes", "intensity decays"],
    );
    refused(
        &unsup(r#"{"type": "bocpd", "hazard": 50.0}"#, clocked),
        &["label_clock does not work with bocpd"],
    );
    accepted(&unsup(r#"{"type": "bocpd", "hazard_rows": 50.0}"#, clocked));
}
