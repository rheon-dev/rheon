//! What a spec is refused for, and the words: the spec, the parameter and
//! the value, from every door a spec comes in by (`Spec::check`, which the
//! bank, a run config and every builder go through). Review round 4
//! (2026-10-06): ceilings on what a spec value sizes (CD10, CF2, CE9),
//! parameters a mode does not read (CF6, CE4, PC6), `warm_rows` below `k`
//! (PC8), the empties (YA5), a ridge of `-0.0` (PC10), the value in every
//! refusal (PC11) and the spec's name on the core's (YA4).

use rheon_polars::{Spec, output_fields};

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

/// A spec of a model with no target over `x0` and `x1`.
fn unsup(model: &str, extra: &str) -> String {
    format!(r#"{{"name": "m", "model": {model}, "features": ["x0", "x1"]{extra}}}"#)
}

fn refused(json: &str, want: &[&str]) {
    let e = check(json).expect_err(&format!("accepted: {json}"));
    for w in want {
        assert!(e.contains(w), "wanted {w:?} in: {e}");
    }
    assert!(
        e.starts_with("spec \"m\": "),
        "the spec is not named first: {e}"
    );
}

fn accepted(json: &str) {
    check(json).unwrap_or_else(|e| panic!("refused: {e}\n{json}"));
}

/// CD10: a lag sizes a ring before the first row, and `n_perm` the draws
/// held for a quantile: 2^20 each, refused by name with the value.
/// `marginal(lags=[2^62])` panicked "capacity overflow" inside the builder.
#[test]
fn a_lag_or_n_perm_past_the_ceiling_is_refused_by_name() {
    let window = r#""kind": "window", "window_rows": 10"#;
    refused(
        &unsup(
            &format!(r#"{{"type": "corrchange", {window}, "n_perm": 1048577}}"#),
            "",
        ),
        &["n_perm must be at most 1048576, but it is 1048577"],
    );
    accepted(&unsup(
        &format!(r#"{{"type": "corrchange", {window}, "n_perm": 1048576}}"#),
        "",
    ));
    for lag in [1_048_577u64, 1 << 62] {
        refused(
            &sup(
                &format!(r#"{{"type": "marginal", "lags": [1, {lag}]}}"#),
                "",
            ),
            &[
                format!("spec \"m\": lags must list lags of at most 1048576, but it lists {lag}")
                    .as_str(),
            ],
        );
        refused(
            &unsup(
                &format!(r#"{{"type": "ew_cov", "lags": [{lag}]}}"#),
                r#", "half_life": 10.0"#,
            ),
            &[
                format!("spec \"m\": lags must list lags of at most 1048576, but it lists {lag}")
                    .as_str(),
            ],
        );
        refused(
            &sup(
                r#"{"type": "ewridge"}"#,
                &format!(r#", "emit_autocorr": true, "resid_autocorr_lag": {lag}"#),
            ),
            &[format!(
                "resid_autocorr_lag must be at most 1048576, but it asks for a lag of {lag}"
            )
            .as_str()],
        );
    }
}

/// CF2: `k` has a ceiling and the warm-up buffer a budget, each refused by
/// name with the value; micro's linkage holds a matrix of 128 MiB at most
/// (4,096 potential summaries) and `O(m)` memory past it, so its cap needs
/// no ceiling.
#[test]
fn kmeans_sizes_have_ceilings_and_micro_needs_none() {
    refused(
        &unsup(
            r#"{"type": "kmeans", "n_clusters": 65537, "warm_rows": 65537}"#,
            r#", "half_life": 10.0"#,
        ),
        &["n_clusters must be at most 65536, but it is 65537"],
    );
    refused(
        &unsup(
            r#"{"type": "kmeans", "n_clusters": 3, "warm_rows": 5592406}"#,
            r#", "half_life": 10.0"#,
        ),
        &["the first 5592406 rows, which seed the centers, would take 256.00003 MiB at 2 features"],
    );
    accepted(&unsup(
        r#"{"type": "micro", "eps": 0.3, "max_clusters": 10000000}"#,
        r#", "half_life": 10.0"#,
    ));
}

/// `hmm`'s `k` sizes its transition matrix, `k²` cells, and its states
/// before the first row: 2^10, refused by name with the value, where
/// `k = 2^62` grew memory without bound (review 2026-10-06, CF2's sibling).
#[test]
fn hmm_k_has_a_ceiling() {
    let hmm = |k: u64| {
        unsup(
            &format!(
                r#"{{"type": "hmm", "n_components": {k}, "warm_rows": {k}, "precision_prior": 0.1}}"#
            ),
            r#", "half_life": 10.0"#,
        )
    };
    for k in [1025u64, 1 << 62] {
        refused(
            &hmm(k),
            &[format!("n_components must be at most 1024, but it is {k}").as_str()],
        );
    }
    accepted(&hmm(1024));
}

/// CE9: rcov's lagged products, `(ring + 1)·k²` doubles, are held to 256
/// MiB before the first row.
#[test]
fn rcov_lagged_products_have_a_byte_ceiling() {
    let ten = (0..10)
        .map(|i| format!("\"x{i}\""))
        .collect::<Vec<_>>()
        .join(", ");
    refused(
        &format!(
            r#"{{"name": "m", "model": {{"type": "rcov", "bandwidth": 1048576}},
                "features": [{ten}], "group": "g", "group_close": "monotone"}}"#
        ),
        &[
            "the lagged covariances the model keeps would take 800 MiB",
            "over the 256 MiB",
        ],
    );
}

/// CF6 and PC8: kmeans' `dead_frac` beside `split_merge = 0`, and
/// `warm_rows` below `k`, refused by name; left out, `dead_frac` is 0 under
/// `split_merge = 0` and `warm_rows` is at least `k`.
#[test]
fn kmeans_refuses_a_dead_rule_with_no_check_and_too_few_warm_rows() {
    let km = |model: &str| unsup(model, r#", "half_life": 10.0"#);
    refused(
        &km(r#"{"type": "kmeans", "n_clusters": 3, "split_merge": 0.0, "dead_frac": 0.25}"#),
        &["dead_frac", "split_merge = 0", "0.25"],
    );
    accepted(&km(
        r#"{"type": "kmeans", "n_clusters": 3, "split_merge": 0.0}"#,
    ));
    refused(
        &km(r#"{"type": "kmeans", "n_clusters": 3, "warm_rows": 1}"#),
        &["warm_rows must be at least n_clusters (3)", "but it is 1"],
    );
    accepted(&km(r#"{"type": "kmeans", "n_clusters": 600}"#));
}

/// CE4: a parameter its mode does not read is refused by name: hmm's
/// `transition` and `transition_prior` beside `tvtp_coef`, its
/// `warm_rows`, `seed_rule` and `seed` beside given states; rcov's
/// `jitter` and `max_bandwidth` outside `"kernel"`, `theta` outside
/// `"preavg"`.
#[test]
fn a_parameter_its_mode_does_not_read_is_refused() {
    let hmm = |extra: &str| {
        format!(
            r#"{{"name": "m", "model": {{"type": "hmm", "n_components": 2, "precision_prior": 0.1{extra}}},
                "features": ["x0", "x1"], "half_life": 10.0}}"#
        )
    };
    let tvtp = r#", "exog_tvtp": "z", "tvtp_coef": [[0, 0, 0, 0], [0, 0, 0, 0]]"#;
    let given = r#", "means": [-1, -1, 1, 1], "covs": [1, 0, 0, 1, 1, 0, 0, 1]"#;
    let with_z = |s: String| s.replace(r#""features""#, r#""targets": ["z"], "features""#);
    for (param, value) in [
        ("transition", "[0.9, 0.1, 0.2, 0.8]"),
        ("transition_prior", "2.0"),
    ] {
        refused(
            &with_z(hmm(&format!(r#"{tvtp}, "{param}": {value}"#))),
            &[&format!("{param} does not apply under tvtp_coef")],
        );
    }
    accepted(&with_z(hmm(tvtp)));
    for (param, value) in [("warm_rows", "20"), ("init", "\"first\""), ("seed", "3")] {
        refused(
            &hmm(&format!(r#"{given}, "{param}": {value}"#)),
            &[&format!(
                "{param} does not apply when means and covs are given"
            )],
        );
    }
    accepted(&hmm(given));
    let rcov = |model: &str| {
        format!(
            r#"{{"name": "m", "model": {model}, "features": ["x0", "x1"], "group": "g",
                "group_close": "monotone"}}"#
        )
    };
    for (kind, param, value, owner) in [
        ("plain", "jitter", "3", "\"kernel\""),
        ("plain", "max_bandwidth", "5", "\"kernel\""),
        ("plain", "theta", "2.0", "\"preavg\""),
        ("kernel", "theta", "2.0", "\"preavg\""),
        ("preavg", "jitter", "3", "\"kernel\""),
        ("preavg", "max_bandwidth", "5", "\"kernel\""),
    ] {
        refused(
            &rcov(&format!(
                r#"{{"type": "rcov", "kind": "{kind}", "block_rows": 100, "{param}": {value}}}"#
            )),
            &[&format!(
                "{param} applies only under kind={owner}, but kind is \"{kind}\"."
            )],
        );
    }
    accepted(&rcov(
        r#"{"type": "rcov", "kind": "kernel", "block_rows": 100, "jitter": 3}"#,
    ));
    accepted(&rcov(
        r#"{"type": "rcov", "kind": "preavg", "block_rows": 100, "theta": 2.0}"#,
    ));
    // `preavg_rows` fixes the window `theta` would set, so a `theta` beside
    // it is refused, as Python's builder refuses it; each alone is taken.
    refused(
        &rcov(
            r#"{"type": "rcov", "kind": "preavg", "block_rows": 100, "theta": 0.5, "preavg_rows": 12}"#,
        ),
        &[
            "theta=0.5 has no effect beside preavg_rows, which fixes the pre-averaging span theta would set. Leave theta out.",
        ],
    );
    accepted(&rcov(
        r#"{"type": "rcov", "kind": "preavg", "block_rows": 100, "preavg_rows": 12}"#,
    ));
}

/// A `covariance_type` shape not offered is refused naming the spec, the
/// parameter, the value and the shapes there are: `hmm`'s said "unknown
/// ew_class covariance" (task 190's find, folded into task 193), and the
/// parameter is one word on every surface now (task 244, P-4).
#[test]
fn a_covariance_refusal_names_the_spec_and_the_parameter() {
    refused(
        &unsup(
            r#"{"type": "hmm", "n_components": 2, "precision_prior": 0.1, "covariance_type": "lda"}"#,
            r#", "half_life": 10.0"#,
        ),
        &["covariance_type must be \"full\", \"tied\" or \"diag\", but it is \"lda\"."],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ew_class", "classes": ["a", "b"],
            "precision_prior": 1.0, "covariance_type": "lda"}, "targets": ["y"],
            "features": ["x0"], "half_life": 10.0}"#,
        &["covariance_type must be \"full\", \"tied\" or \"diag\", but it is \"lda\"."],
    );
}

/// PC6: `kalman` takes `coef_half_life` or `q`, exactly one: `q` alone was
/// refused as a missing field, and the pair took `q` and ignored the
/// half-life.
#[test]
fn kalman_takes_exactly_one_of_coef_half_life_and_q() {
    accepted(&sup(r#"{"type": "kalman", "q": [0.0, 0.1, 0.1]}"#, ""));
    accepted(&sup(r#"{"type": "kalman", "coef_half_life": 50.0}"#, ""));
    refused(
        &sup(
            r#"{"type": "kalman", "coef_half_life": 50.0, "q": [0.0, 0.1, 0.1]}"#,
            "",
        ),
        &["spec \"m\": coef_half_life and q cannot both be set"],
    );
    refused(
        &sup(r#"{"type": "kalman"}"#, ""),
        &["kalman needs coef_half_life or q, but neither is set."],
    );
}

/// YA5: an empty `mahal_quantiles` and an empty target name are refused,
/// as an empty `resid_quantiles` and `rheon.target("")` are.
#[test]
fn the_empties_are_refused() {
    refused(
        &unsup(
            r#"{"type": "ew_cov", "stats": ["mahal"], "precision_prior": 1.0, "mahal_quantiles": []}"#,
            r#", "half_life": 10.0"#,
        ),
        &["mahal_quantiles must list at least one level, but it is empty."],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": [""], "features": ["x0"],
            "half_life": 10.0}"#,
        &["every target needs a name, but one of them is \"\""],
    );
    // An empty `lags` asked for nothing, and `marginal` took it as no lags
    // where `ew_cov` refuses it; `cross_lags = []` is a choice, the least
    // memory, and stays.
    refused(
        &unsup(
            r#"{"type": "ew_cov", "lags": []}"#,
            r#", "half_life": 10.0"#,
        ),
        &["spec \"m\": lags must hold at least one lag, but it is empty"],
    );
    refused(
        &sup(r#"{"type": "marginal", "lags": []}"#, ""),
        &[
            "spec \"m\": lags must hold at least one lag, but it is empty. Give at least one lag, such as [1], or leave lags out.",
        ],
    );
    refused(
        &sup(r#"{"type": "marginal", "lags": [], "cross_lags": []}"#, ""),
        &["lags must hold at least one lag, but it is empty"],
    );
    accepted(&sup(
        r#"{"type": "marginal", "lags": [1, 2], "cross_lags": []}"#,
        "",
    ));
    accepted(&sup(r#"{"type": "marginal", "lags": null}"#, ""));
}

/// PC10: `-0.0` is `0.0`: a ridge grid of the two is one value listed
/// twice, and a `-0.0` names its field `__r0`.
#[test]
fn minus_zero_is_zero_in_a_grid_and_a_field_name() {
    refused(
        &sup(r#"{"type": "ewridge", "ridge": [0.0, -0.0]}"#, ""),
        &[
            "ridge must list each value once, since each names its own fields, but it lists 0 more than once.",
        ],
    );
    let mut s: Spec =
        serde_json::from_str(&sup(r#"{"type": "ewridge", "ridge": [-0.0, 1.0]}"#, "")).unwrap();
    s.check().unwrap();
    let fields = output_fields(&s);
    assert!(
        fields.contains(&"pred_y__r0".to_string()) && !fields.iter().any(|f| f.contains("-0")),
        "{fields:?}"
    );
}

/// PC11: a refusal of a value names it. A sample across the shared
/// parameters and the models, each refused with `got <value>`.
#[test]
fn a_refused_value_is_named() {
    for (json, want) in [
        (
            sup(r#"{"type": "ewridge", "ridge": -1.0}"#, ""),
            "ridge must be finite and at least 0, but it is -1. Set ridge to 0 for no shrinkage, or to a finite positive value.",
        ),
        (
            sup(r#"{"type": "ewridge"}"#, r#", "min_weight": -2.0"#),
            "min_weight must be finite and at least 0, but it is -2. Set min_weight to 0 to predict from the first row, or to a finite positive value.",
        ),
        (
            sup(r#"{"type": "ewridge", "solve_every": -3.0}"#, ""),
            "solve_every must be at least 0 clock units, but it is -3",
        ),
        (
            sup(r#"{"type": "ewridge"}"#, r#", "conformal": 1.5"#),
            "conformal must be above 0 and below 1, but it is 1.5.",
        ),
        (
            sup(r#"{"type": "lasso", "penalties": [0.1], "tol": 0.0}"#, ""),
            "tol must be finite and above 0, but it is 0.",
        ),
        (
            sup(
                r#"{"type": "kalman", "coef_half_life": 50.0, "p0": -1.0}"#,
                "",
            ),
            "p0 must be finite and above 0, but it is -1.",
        ),
        (
            sup(r#"{"type": "ftrl", "alpha": 0.0}"#, ""),
            "spec \"m\": alpha must be finite and above 0, but it is 0.",
        ),
        (
            sup(r#"{"type": "sgd", "learning_rate": -0.5}"#, ""),
            "learning_rate must be finite and above 0, but it is -0.5.",
        ),
        (
            unsup(
                r#"{"type": "kmeans", "n_clusters": 2, "dead_frac": -1.0}"#,
                r#", "half_life": 10.0"#,
            ),
            "dead_frac must be at least 0 and below 0.5, but it is -1. Set dead_frac to 0 to never re-place a center whose weight has decayed, or to a positive value below 0.5.",
        ),
        (
            unsup(
                r#"{"type": "micro", "eps": -0.5}"#,
                r#", "half_life": 10.0"#,
            ),
            "spec \"m\": eps must be finite and above 0, but it is -0.5.",
        ),
        (
            r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"],
                "features": ["x0"], "half_life": [10.0, -4.0]}"#
                .to_string(),
            "half_life must be above 0, but one of its values is -4",
        ),
    ] {
        refused(&json, &[want]);
    }
}

/// YA4: a refusal raised by a model's own check names the spec, as the
/// spec's own refusals do; and a window that is not finite is refused as
/// "finite and > 0", which "> 0 (got inf)" was untrue of.
#[test]
fn a_models_refusal_names_the_spec() {
    refused(
        &sup(r#"{"type": "sgd", "clip_gradient": 0.0}"#, ""),
        &["clip_gradient must be above 0, but it is 0."],
    );
    for model in [
        r#"{"type": "lasso", "penalties": [0.1], "window_size": "inf"}"#,
        r#"{"type": "marginal", "window_size": "inf"}"#,
    ] {
        refused(
            &sup(model, ""),
            &["window_size must be finite and above 0, but it is inf"],
        );
    }
}

/// PC11: an old name is refused naming the new one, and cites nothing a
/// wheel's user does not have.
#[test]
fn a_renamed_key_cites_no_plan_task() {
    let msg = rheon_polars::name_renamed("unknown field `halflife`, expected one of `x`");
    assert!(msg.contains("halflife was renamed half_life"), "{msg}");
    assert!(!msg.contains("PLAN") && !msg.contains("task"), "{msg}");
}

// ---- Task 252: each refusal below had no test that reached it. ----

/// A count that must be at least 1, given as 0, is refused by name with
/// the value, for every parameter that sizes or schedules something.
#[test]
fn a_count_of_zero_is_refused_by_name() {
    const HL: &str = r#", "half_life": 10.0"#;
    let cases: Vec<(String, &str)> = vec![
        (
            unsup(r#"{"type": "kmeans", "n_clusters": 0}"#, HL),
            "spec \"m\": n_clusters must be at least 1, but it is 0.",
        ),
        (
            unsup(
                r#"{"type": "kmeans", "n_clusters": 2, "update_every_rows": 0}"#,
                HL,
            ),
            "update_every_rows must be at least 1, but it is 0.",
        ),
        (
            unsup(
                r#"{"type": "kmeans", "n_clusters": 2, "split_merge_every_rows": 0}"#,
                HL,
            ),
            "split_merge_every_rows must be at least 1, but it is 0.",
        ),
        (
            unsup(r#"{"type": "micro", "eps": 0.5, "max_clusters": 0}"#, HL),
            "max_clusters must be at least 1, but it is 0.",
        ),
        (
            sup(
                r#"{"type": "ewridge"}"#,
                r#", "emit_specification": true, "ljung_box_lags": 0"#,
            ),
            "ljung_box_lags must be at least 1, but it is 0.",
        ),
        (
            sup(
                r#"{"type": "ewridge"}"#,
                r#", "emit_autocorr": true, "resid_autocorr_lag": 0"#,
            ),
            "resid_autocorr_lag must be at least 1, but it is 0.",
        ),
    ];
    for (json, want) in cases {
        refused(&json, &[want]);
    }
}

/// `horizon_rows` sizes Newey and West's ring before the first row: past
/// half the lag ceiling it is refused with the ceiling, and at it accepted.
#[test]
fn a_horizon_past_half_the_lag_ceiling_is_refused() {
    let half = rheon_polars::rheon_core::MAX_LAG / 2;
    let at = |h: usize| {
        sup(
            r#"{"type": "ewridge"}"#,
            &format!(r#", "emit_calibration": true, "horizon_rows": {h}"#),
        )
    };
    refused(
        &at(half + 1),
        &[
            &format!(
                "horizon_rows must be at most {half}, but it is {}",
                half + 1
            ),
            "Lower horizon_rows to",
        ],
    );
    accepted(&at(half));
}

/// `gram_threads = 0` is refused by the reader, and a spec built in Rust
/// with it is refused by `check`, naming the parameter and the way out.
#[test]
fn a_spec_built_in_rust_with_no_gram_threads_is_refused() {
    let mut s: Spec = serde_json::from_str(&sup(r#"{"type": "ewridge"}"#, "")).unwrap();
    let rheon_polars::ModelKind::EwRidge { gram_threads, .. } = &mut s.model else {
        unreachable!("an ewridge spec");
    };
    *gram_threads = Some(rheon_polars::CountOrAuto::Count(0));
    let e = s.check().expect_err("gram_threads = 0 accepted");
    assert!(
        e.starts_with("spec \"m\": gram_threads must be at least 1 or \"auto\", but it is 0."),
        "{e}"
    );
}

/// What a model cannot take: holt's features, `ew_class`'s second target,
/// a `coef_prior` of the wrong shape, a feature set naming a column that is
/// not a feature, and `corrchange`'s sequential kind without its history.
#[test]
fn a_model_given_what_it_cannot_read_is_refused() {
    refused(
        r#"{"name": "m", "model": {"type": "holt"}, "targets": ["y"], "features": ["x0"],
            "half_life": 10.0}"#,
        &["holt takes no features, but features lists 1"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "ew_class", "classes": ["a", "b"], "precision_prior": 1.0},
            "targets": ["y", "z"], "features": ["x0"], "half_life": 10.0}"#,
        &["ew_class reads one target, the column of classes, but targets names 2 columns."],
    );
    refused(
        &sup(r#"{"type": "rls", "coef_prior": [[1.0, 2.0]]}"#, ""),
        &[
            "coef_prior must hold one list of 3 coefficients for the target, but it holds 1 list of 2. Give one list for each target, the intercept first, then one for each feature in the order of features.",
        ],
    );
    refused(
        &sup(
            r#"{"type": "ewridge", "feature_sets": [["a", ["x0", "nope"]]]}"#,
            "",
        ),
        &["feature set \"a\" lists \"nope\", which is not one of features"],
    );
    let e = check(&unsup(
        r#"{"type": "deco", "blocks": [["a", ["x0", "zz"]], ["b", ["x1", "x2"]]]}"#,
        r#", "half_life": 10.0"#,
    ))
    .expect_err("a deco block naming no feature accepted");
    assert!(e.contains("zz"), "{e}");
    let e = check(&unsup(
        r#"{"type": "corrchange", "kind": "sequential"}"#,
        "",
    ))
    .expect_err("sequential without window_rows accepted");
    assert!(
        e.contains("kind=\"sequential\" needs window_rows")
            && e.contains("the rows of history each monitored period is tested against"),
        "{e}"
    );
}

/// A formula target is refused, with the reason, by a model that does not
/// regress on a real-valued target: `seqtest` tests signs, `sgd` under the
/// logistic loss fits a probability and under the Poisson one a log rate.
/// (A comparison is a `seqtest`, refused for its signs before its sides.)
#[test]
fn a_formula_target_is_refused_by_a_model_it_does_not_apply_to() {
    let target = r#"[{"name": "fwd", "formula": ["rewm_mean", ["col", "y"], {"half_life": 5.0, "window_size": 10.0}]}]"#;
    let with = |model: &str| {
        format!(
            r#"{{"name": "m", "model": {model}, "targets": {target}, "features": ["x0"],
                "half_life": 10.0, "embargo": 10.0}}"#
        )
    };
    for (model, why) in [
        (
            r#"{"type": "sgd", "loss": "logistic"}"#,
            "fits a probability to a 0/1 target under loss=\"logistic\"",
        ),
        (
            r#"{"type": "sgd", "loss": "poisson"}"#,
            "fits a log rate to a count under loss=\"poisson\"",
        ),
    ] {
        refused(&with(model), &["a formula target", why]);
    }
    let e = check(&format!(
        r#"{{"name": "m", "model": {{"type": "seqtest"}}, "targets": {target},
            "features": []}}"#
    ))
    .expect_err("a seqtest over a formula target accepted");
    assert!(e.contains("tests the signs of its targets"), "{e}");
}

/// A session column needs a `session_gap` (or a session close), and a
/// word that is not `"reset"` is refused naming the forms it can take.
#[test]
fn a_session_without_its_rule_or_with_a_word_it_does_not_know_is_refused() {
    let clocked = r#", "clock": "t", "gap_cap": 5.0, "session": "s""#;
    let e = check(&sup(r#"{"type": "ewridge"}"#, clocked)).expect_err("no session_gap");
    assert!(
        e.contains("session needs session_gap, but session_gap is not set")
            && e.contains("group_close=\"session\""),
        "{e}"
    );
    // The reader takes no word but "reset"; a spec built in Rust is held to
    // the same.
    let mut s: Spec = serde_json::from_str(&sup(
        r#"{"type": "ewridge"}"#,
        &format!(r#"{clocked}, "session_gap": "reset""#),
    ))
    .unwrap();
    s.session_gap = Some(rheon_polars::SessionGapSpec::Word("restart".into()));
    let e = s.check().expect_err("a session_gap word accepted");
    assert!(
        e.contains("session_gap must be a number, a duration or \"reset\", but it is \"restart\""),
        "{e}"
    );
    // A negative integer is read as a number, and refused as one.
    let e = check(&sup(
        r#"{"type": "ewridge"}"#,
        &format!(r#"{clocked}, "session_gap": -2"#),
    ))
    .expect_err("a negative session_gap accepted");
    assert!(
        e.contains("session_gap must be at least 0 or \"reset\""),
        "{e}"
    );
}

/// The window operators' run says the same about a cap that is not finite,
/// and about a session and its gap, in a window's words rather than a
/// model's: a gap without a session is refused, and a session needs none.
#[test]
fn a_window_runs_clock_rules_are_said_in_a_windows_words() {
    use rheon_polars::{WindowsConfig, WindowsRun};
    let schema = polars::prelude::Schema::from_iter([
        ("t".into(), polars::prelude::DataType::Float64),
        ("s".into(), polars::prelude::DataType::String),
        ("x".into(), polars::prelude::DataType::Float64),
    ]);
    let refusal = |extra: &str| {
        let config: WindowsConfig = serde_json::from_str(&format!(
            r#"{{"formulas": [{{"name": "f", "tree": ["ewm_mean", ["col", "x"], {{"half_life": 5.0}}]}}],
                "clock": "t"{extra}}}"#
        ))
        .unwrap();
        WindowsRun::new(config, &schema)
            .err()
            .expect("the run started")
    };
    let e = refusal(r#", "gap_cap": "inf""#);
    assert!(
        e.contains("gap_cap must be finite, but it is inf: it is the longest step a window spans"),
        "{e}"
    );
    // A session needs no gap here: every window ends at a session change,
    // so no window reads the number (task 278).
    let config: WindowsConfig = serde_json::from_str(
        r#"{"formulas": [{"name": "f", "tree": ["ewm_mean", ["col", "x"], {"half_life": 5.0}]}],
            "clock": "t", "gap_cap": 5.0, "session": "s"}"#,
    )
    .unwrap();
    WindowsRun::new(config, &schema).expect("a session without a gap refused");
    let e = refusal(r#", "gap_cap": 5.0, "session_gap": 1.0"#);
    assert!(
        e.starts_with("with_windows: session_gap is set, but session is not"),
        "{e}"
    );
}

/// A model key the 1.x series forwarded is refused, naming the current one,
/// where the model takes it; a model that takes neither is told nothing,
/// and a word renamed with it is told its new spelling.
#[test]
fn a_removed_model_key_is_refused_naming_the_current_one() {
    let refused = |model: &str| -> String {
        let msg = serde_json::from_str::<rheon_polars::Spec>(&format!(
            r#"{{"name": "m", "model": {model}, "features": ["x0", "x1"], "half_life": 10.0}}"#
        ))
        .unwrap_err()
        .to_string();
        rheon_polars::name_renamed(&msg)
    };
    for (model, old, new) in [
        (r#"{"type": "kmeans", "k": 2}"#, "k", "n_clusters"),
        (r#"{"type": "hmm", "k": 2}"#, "k", "n_components"),
        (
            r#"{"type": "bocpd", "emission": "robust"}"#,
            "emission",
            "covariance_type",
        ),
        (
            r#"{"type": "corrchange", "reset_on_flag": true}"#,
            "reset_on_flag",
            "change_action",
        ),
        (
            r#"{"type": "ewridge", "window_every": 5}"#,
            "window_every",
            "snapshot_every",
        ),
    ] {
        let e = refused(model);
        assert!(e.ends_with(&format!("; {old} was renamed {new}")), "{e}");
    }
    // Another model's `k` is plain: it takes neither new name.
    let e = refused(r#"{"type": "ewridge", "k": 2}"#);
    assert!(!e.contains("was renamed"), "{e}");
    // A word renamed with them is told its new spelling.
    let mut spec: rheon_polars::Spec = serde_json::from_str(
        r#"{"name": "m", "model": {"type": "kmeans", "n_clusters": 2, "init": "kmeanspp"},
            "features": ["x0", "x1"], "half_life": 10.0}"#,
    )
    .unwrap();
    spec.fill_defaults();
    let e = spec.validate().unwrap_err();
    assert!(e.contains("the old name of \"k-means++\""), "{e}");
}

/// A word refusal from serde with an empty list of words is not one this
/// library can reword: passed on as it is (`None`).
#[test]
fn a_variant_message_with_no_words_is_not_reworded() {
    assert_eq!(
        rheon_polars::word_refusal_body("unknown variant `x`, expected `` or ``"),
        None
    );
    assert!(rheon_polars::word_refusal_body("unknown variant `x`, expected `a` or `b`").is_some());
}

/// A word a model does not know is refused naming the word, whichever of
/// the model's word keys it is given to.
#[test]
fn a_model_word_it_does_not_know_is_refused_naming_it() {
    for (model, word) in [
        (r#"{"type": "ew_cov", "stats": ["kurtosis"]}"#, "kurtosis"),
        (r#"{"type": "sgd", "loss": "hinge"}"#, "hinge"),
        (r#"{"type": "sgd", "schedule": "cosine"}"#, "cosine"),
        (r#"{"type": "pa", "mode": "pa3"}"#, "pa3"),
        (
            r#"{"type": "marginal", "feature_moments": "pooled"}"#,
            "pooled",
        ),
    ] {
        let json = if model.contains("ew_cov") {
            unsup(model, r#", "half_life": 10.0"#)
        } else {
            sup(model, "")
        };
        let e = check(&json).expect_err(model);
        assert!(e.contains(word), "{model}: {e}");
    }
}

/// What needs a decay reads `lam` as it reads a half-life: `lam < 1`
/// forgets, and `lam = 1` does not, for `min_settled_frac` and for feature
/// health alike; `lam` beside `half_life` is refused.
#[test]
fn lam_is_a_decay_where_a_decay_is_needed() {
    let at = |lam: &str, extra: &str| {
        format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": ["y"],
                "features": ["x0", "x1"], "lam": {lam}{extra}}}"#
        )
    };
    for extra in [
        r#", "min_settled_frac": 0.5"#,
        r#", "emit_feature_health": true"#,
    ] {
        accepted(&at("0.99", extra));
        let e = check(&at("1.0", extra)).expect_err(extra);
        assert!(e.starts_with("spec \"m\": "), "{e}");
    }
    let mut both: Spec = serde_json::from_str(&at("0.99", "")).unwrap();
    both.half_life = Some(rheon_polars::SpanList::One(rheon_polars::Span::Units(10.0)));
    assert_eq!(
        both.decays().unwrap_err(),
        "spec \"m\": half_life and lam cannot both be set: each sets the decay. Leave out \
         half_life, or lam."
    );
}

/// A relative target with no column or name is refused with the removal's
/// own words; a formula target's name must not be empty.
#[test]
fn a_target_table_with_nothing_to_name_is_refused() {
    let parse = |targets: &str| {
        serde_json::from_str::<Spec>(&format!(
            r#"{{"name": "m", "model": {{"type": "ewridge"}}, "targets": {targets},
                "features": ["x0"], "half_life": 10.0}}"#
        ))
        .expect_err("parsed")
        .to_string()
    };
    let e = parse(r#"[{"relative_to": "x0"}]"#);
    assert!(!e.contains("target \""), "no column to name: {e}");
    assert!(e.contains("relative"), "{e}");
    let e = parse(r#"[{"name": "", "formula": ["col", "y"]}]"#);
    assert!(e.contains("formula target: name must not be empty"), "{e}");
}

/// A refit's buffer is held at its full size from a target's first learned
/// row, so `huber_refit_rows` and `quantile_refit_rows` size an allocation
/// before any row: past 2^20 each is refused by name with the value, where
/// `2^32 - 1` beside `refit_budget = inf` reserved 172 GB and aborted. 2^20
/// itself is accepted.
#[test]
fn a_refit_row_count_past_the_ceiling_is_refused_by_name() {
    for (model, key) in [
        (r#"{"type": "huber""#, "huber_refit_rows"),
        (
            r#"{"type": "huber", "huber_scale": "rms""#,
            "huber_refit_rows",
        ),
        (r#"{"type": "sgd", "loss": "huber""#, "huber_refit_rows"),
        (
            r#"{"type": "sgd", "loss": "huber", "huber_scale": "rms""#,
            "huber_refit_rows",
        ),
        (
            r#"{"type": "quantile", "quantile": 0.5"#,
            "quantile_refit_rows",
        ),
    ] {
        refused(
            &sup(&format!(r#"{model}, "{key}": 1048577}}"#), ""),
            &[
                &format!("{key} must be at most 1048576, but it is 1048577"),
                "reserves memory for all its rows from a target's first learned row",
                &format!("Lower {key} to 1048576 or below."),
            ],
        );
        refused(
            &sup(
                &format!(r#"{model}, "{key}": 4294967295, "refit_budget": "inf"}}"#),
                "",
            ),
            &[&format!(
                "{key} must be at most 1048576, but it is 4294967295"
            )],
        );
        accepted(&sup(&format!(r#"{model}, "{key}": 1048576}}"#), ""));
    }
}

/// A spec's checks report every problem they find in one refusal: the
/// spec named once, then each problem on a line of its own, in the order
/// the checks read them (WRITING §11).
#[test]
fn every_problem_of_a_spec_is_reported_at_once() {
    let e = check(&sup(
        r#"{"type": "ewridge"}"#,
        r#", "embargo": 0, "conformal": 1.5, "drift_delta": 0.1"#,
    ))
    .expect_err("three problems accepted");
    assert_eq!(
        e,
        "spec \"m\": 3 problems.\n\
         - embargo must be finite and above 0, but it is 0. Set embargo to the delay before a \
         row's target is known, in clock units, or leave it out for no delay.\n\
         - conformal must be above 0 and below 1, but it is 1.5. Set conformal to the share of \
         rows whose target the interval should cover, such as 0.9.\n\
         - drift_delta applies only when emit_drift is on, but emit_drift is off. Turn on \
         emit_drift, or leave out drift_delta."
    );
    // Each alone is the message it was written as.
    let e = check(&sup(r#"{"type": "ewridge"}"#, r#", "drift_delta": 0.1"#)).unwrap_err();
    assert_eq!(
        e,
        "spec \"m\": drift_delta applies only when emit_drift is on, but emit_drift is off. \
         Turn on emit_drift, or leave out drift_delta."
    );
}

/// Where the checks after a problem would read what it refuses, they stop
/// there and say so: a duration with no clock leaves every clock parameter
/// without a scale, so the bad `conformal` after it is not reported yet.
#[test]
fn a_check_that_stops_early_says_so() {
    let e = check(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0"],
            "half_life": "10m", "conformal": 1.5}"#,
    )
    .expect_err("a duration with no clock accepted");
    assert!(e.starts_with("spec \"m\": "), "{e}");
    assert!(
        e.ends_with("\nFix this, then run again to check the spec's other settings."),
        "{e}"
    );
    assert!(!e.contains("conformal"), "{e}");
    // The columns are a stop too: a spec with no target reports none of
    // the checks that read its targets.
    let e = check(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": [], "features": ["x0"],
            "half_life": 10.0, "conformal": 1.5}"#,
    )
    .unwrap_err();
    assert_eq!(
        e,
        "spec \"m\": targets must name at least one column, but it is empty. Name the columns \
         to predict in targets.\nFix this, then run again to check the spec's other \
         settings."
    );
}

/// A model's own settings, checked where the spec is read, report every
/// problem at once too: `kalman` given both ways of setting its drift, a
/// negative `p0` and an `obs_var` of 0 names all three, each alone as it
/// would be.
#[test]
fn a_models_settings_report_every_problem_at_once() {
    let e = check(&sup(
        r#"{"type": "kalman", "coef_half_life": 50.0, "q": [0.0, 0.1, 0.1], "p0": -1.0, "obs_var": 0.0}"#,
        "",
    ))
    .expect_err("three problems accepted");
    assert_eq!(
        e,
        "spec \"m\": 3 problems.\n\
         - coef_half_life and q cannot both be set: q gives the drift of each coefficient \
         outright, and coef_half_life derives it. Leave out coef_half_life, or q.\n\
         - obs_var must be finite and above 0, but it is 0. Set obs_var to the target's noise \
         variance, or leave it out to estimate it from the residuals.\n\
         - p0 must be finite and above 0, but it is -1. Set p0 to each slope's starting \
         variance, as a multiple of the target's noise variance, such as the default 1."
    );
    let e = check(&sup(
        r#"{"type": "kalman", "coef_half_life": 50.0, "p0": -1.0}"#,
        "",
    ))
    .unwrap_err();
    assert_eq!(
        e,
        "spec \"m\": p0 must be finite and above 0, but it is -1. Set p0 to each slope's \
         starting variance, as a multiple of the target's noise variance, such as the default 1."
    );
    // A loss outside the words is said once: the parameters of each loss
    // are not then read against it.
    let e = check(&sup(r#"{"type": "sgd", "loss": "nope", "eps": 0.2}"#, "")).unwrap_err();
    assert!(!e.contains("problems") && e.contains("\"nope\""), "{e}");
}

/// `deco`'s blocks are held to the model's rules by the names the reader
/// wrote, every problem at once: a block that lists a feature twice and so
/// holds one, a feature in two blocks, and a feature in none.
#[test]
fn a_deco_block_problem_names_the_block_and_the_feature() {
    let deco = |blocks: &str| {
        format!(
            r#"{{"name": "m", "model": {{"type": "deco", "blocks": {blocks}}},
                "features": ["a", "b", "c", "d"], "half_life": 10.0}}"#
        )
    };
    let e = check(&deco(r#"[["u", ["a", "a"]], ["v", ["b", "a"]]]"#)).unwrap_err();
    assert_eq!(
        e,
        "spec \"m\": 4 problems.\n\
         - block \"u\" lists \"a\" more than once. List each feature once in a block.\n\
         - block \"u\" must hold at least two features, since its correlation is between them, \
         but it holds one, \"a\". Add a feature to it, or merge it with another block.\n\
         - feature \"a\" is in blocks \"u\" and \"v\", but each feature belongs to one block. \
         Remove it from all but one.\n\
         - every feature must be in exactly one block, but \"c\" and \"d\" are in none. Add \
         each to a block, or remove them from features."
    );
    assert!(check(&deco(r#"[["u", ["a", "b"]], ["v", ["c", "d"]]]"#)).is_ok());
}

/// `sgd` reports a word outside `loss`'s and one outside `schedule`'s at
/// once.
#[test]
fn sgd_reports_a_bad_loss_and_a_bad_schedule_together() {
    let e = check(&sup(
        r#"{"type": "sgd", "loss": "nope", "schedule": "nah"}"#,
        "",
    ))
    .unwrap_err();
    assert!(
        e.starts_with("spec \"m\": 2 problems.\n- loss must be ")
            && e.contains("\n- schedule must be \"constant\", \"inv_scaling\" or \"adagrad\""),
        "{e}"
    );
}

/// A half-life written two ways names both spellings, and a model that
/// takes no decay refuses a `half_life` once, by its own rule, not again for
/// its value.
#[test]
fn a_half_life_grid_names_each_spelling_and_a_model_with_no_decay_says_one_thing() {
    let e = check(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x0"],
            "clock": "t", "gap_cap": "1h", "half_life": ["5m", "300s"]}"#,
    )
    .unwrap_err();
    assert!(
        e.contains("half_life lists 5m and 300s, which are the same half-life"),
        "{e}"
    );
    let e = check(&unsup(r#"{"type": "bocpd"}"#, r#", "half_life": 0.0"#)).unwrap_err();
    assert!(
        e.starts_with("spec \"m\": half_life does not apply to bocpd") && !e.contains("problems"),
        "{e}"
    );
}

/// A duration beside lam: each form's remedy is one the reader can follow.
/// A half-life given as a duration with lam is told to leave lam out, not
/// to give the half-life it already gave; without a clock, gap_cap is left
/// out rather than given as a number of rows it has no form for.
#[test]
fn a_duration_beside_lam_is_told_the_way_out_that_works() {
    let e = check(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"],
            "clock": "t", "half_life": "10m", "lam": 0.99, "gap_cap": "1h"}"#,
    )
    .unwrap_err();
    assert!(
        e.starts_with(
            "spec \"m\": half_life is a duration, but lam is set too, and lam is a decay over one \
             clock unit, which a duration cannot give. Leave lam out."
        ),
        "{e}"
    );
    let e = check(
        r#"{"name": "m", "model": {"type": "ewridge"}, "targets": ["y"], "features": ["x"],
            "lam": 0.99, "gap_cap": "1h"}"#,
    )
    .unwrap_err();
    assert!(
        e.contains(
            "spec \"m\": gap_cap is a duration, but the spec has no clock to measure it, and lam \
             is a decay over one row. Set clock to a column of times (Datetime, Date or Duration) \
             and give half_life as a duration in place of lam, or leave gap_cap out."
        ),
        "{e}"
    );
}

/// `marginal`'s `bin_rule` and `bin_warm_rows` say how to learn the bins,
/// so without `bins` they had no effect and were taken; they are refused as
/// `bin_budget` and `bin_stat` without bins are, and taken beside `bins`
/// (task 356).
#[test]
fn marginal_learning_the_bins_without_bins_is_refused() {
    refused(
        &sup(r#"{"type": "marginal", "bin_rule": "fixed"}"#, ""),
        &[
            "bin_rule sets how to learn the bins, but bins is not set, so there are none to \
           learn. Set bins, or leave bin_rule out.",
        ],
    );
    refused(
        &sup(r#"{"type": "marginal", "bin_warm_rows": 50}"#, ""),
        &["bin_warm_rows sets how to learn the bins, but bins is not set"],
    );
    refused(
        &sup(
            r#"{"type": "marginal", "bin_rule": "quantile", "bin_warm_rows": 50}"#,
            "",
        ),
        &[
            "bin_rule and bin_warm_rows set how to learn the bins, but bins is not set, so \
           there are none to learn. Set bins, or leave bin_rule and bin_warm_rows out.",
        ],
    );
    accepted(&sup(
        r#"{"type": "marginal", "bins": 4, "bin_rule": "fixed", "bin_warm_rows": 50}"#,
        "",
    ));
}

/// Review P11: a NUL character in a spec's name or a target's name panicked
/// in the Arrow export, after the bank had learned the chunk; each is
/// refused when the spec is checked. A column target's name defaults to its
/// column, and a formula target's is its own.
#[test]
fn a_nul_in_a_spec_or_target_name_is_refused() {
    let e = check(
        &sup(r#"{"type": "ewridge"}"#, "").replace(r#""name": "m""#, r#""name": "a\u0000b""#),
    )
    .unwrap_err();
    assert!(e.contains("the spec's name holds a NUL character"), "{e}");
    refused(
        &sup(r#"{"type": "ewridge"}"#, "").replace(
            r#""targets": ["y"]"#,
            r#""targets": [{"column": "y", "name": "p\u0000q"}]"#,
        ),
        &["target \"p\\0q\" has a name holding a NUL character"],
    );
    refused(
        &sup(r#"{"type": "ewridge"}"#, "")
            .replace(r#""targets": ["y"]"#, r#""targets": ["y\u0000"]"#),
        &["holding a NUL character"],
    );
    let formula = r#""targets": [{"name": "a\u0000b", "formula": ["-", ["rewm_mean", ["col", "mid"], {"half_life": "5s", "window_size": "10s"}], ["col", "mid"]]}], "clock": "t", "embargo": "10s""#;
    let json = sup(r#"{"type": "ewridge"}"#, "")
        .replace(r#""targets": ["y"]"#, formula)
        .replace(r#""half_life": 10.0"#, r#""half_life": "10s""#);
    let e = check(&json).unwrap_err();
    assert!(e.contains("holding a NUL character"), "{e}");
}

/// Review P14: a feature named `intercept` beside `fit_intercept` gave two
/// coefficients the one name, in `coef`, the Gram and the bank's tables. It
/// is refused, and taken with `fit_intercept` off, or by a model with no
/// constant term of its own.
#[test]
fn a_feature_named_intercept_beside_the_fits_own_is_refused() {
    let features = r#""features": ["x0", "intercept"]"#;
    for model in [
        r#"{"type": "ewridge"}"#,
        r#"{"type": "rls"}"#,
        r#"{"type": "huber"}"#,
    ] {
        refused(
            &sup(model, "").replace(r#""features": ["x0", "x1"]"#, features),
            &["features lists \"intercept\", but fit_intercept is on"],
        );
        accepted(
            &sup(model, r#", "fit_intercept": false"#)
                .replace(r#""features": ["x0", "x1"]"#, features),
        );
    }
    accepted(
        &unsup(r#"{"type": "ew_cov"}"#, r#", "half_life": 10.0"#)
            .replace(r#""features": ["x0", "x1"]"#, features),
    );
}

/// `hawkes`' window is the one part of its state the data could size, and
/// `window_events` bounds it, so its bytes are known at the call: past
/// `max_state_bytes` (256 KB by default) the spec is refused naming them,
/// at it the spec is taken, and features and a missing `kernel_decays`
/// are refused by name.
#[test]
fn a_hawkes_window_past_its_state_cap_is_refused() {
    let spec = |model: &str| {
        format!(
            r#"{{"name": "m", "model": {model}, "targets": ["a", "b"], "features": [],
                "half_life": "inf"}}"#
        )
    };
    // Two nodes, two decays: 8 * 8 = 64 bytes a row (its clock, its weight,
    // two marks, four sums), 160 for the sums and 48 beside the window.
    let at = |w: usize| rheon_polars::rheon_core::Hawkes::state_bytes(2, 2, Some(w));
    assert_eq!(at(4_000), 8 * (20 + 4_000 * 8 + 6));
    let cap = 256 * 1024;
    let fits = ((cap - at(0)) / 64) as usize;
    assert!(at(fits) <= cap && at(fits + 1) > cap);
    accepted(&spec(&format!(
        r#"{{"type": "hawkes", "kernel_decays": [1.0, 5.0], "window_events": {fits}}}"#
    )));
    refused(
        &spec(&format!(
            r#"{{"type": "hawkes", "kernel_decays": [1.0, 5.0], "window_events": {}}}"#,
            fits + 1
        )),
        &[
            &format!("hold {} bytes in each group", at(fits + 1)),
            "max_state_bytes of 262144",
        ],
    );
    accepted(&spec(
        r#"{"type": "hawkes", "kernel_decays": [1.0, 5.0], "window_events": 100000,
            "max_state_bytes": 8000000}"#,
    ));
    refused(
        &spec(r#"{"type": "hawkes", "kernel_decays": [1.0, 5.0], "max_state_bytes": 0}"#),
        &["max_state_bytes must be above 0"],
    );
    refused(&spec(r#"{"type": "hawkes"}"#), &["needs its kernels"]);
    refused(
        r#"{"name": "m", "model": {"type": "hawkes", "kernel_decays": [1.0]},
            "targets": ["a"], "features": ["x"], "half_life": 10.0}"#,
        &["hawkes takes no features"],
    );
}

/// `garch`'s start and its FIGARCH ring are its state's size, known from
/// its settings at the call: past `max_state_bytes` (256 KB by default) the
/// spec is refused naming the bytes, and a decay, features, a start given
/// both ways, an unknown word and a t on a process that takes none are
/// refused by name.
#[test]
fn a_garch_past_its_state_cap_or_its_forms_is_refused() {
    let spec = |model: &str| {
        format!(r#"{{"name": "m", "model": {model}, "targets": ["a", "b"], "features": []}}"#)
    };
    accepted(&spec(r#"{"type": "garch"}"#));
    accepted(&spec(
        r#"{"type": "garch", "vol": "figarch", "truncation": 15000}"#,
    ));
    // two targets of FIGARCH: 2 (216 + 8 M) + 16 bytes, 256,448 at 16,000
    // lags and 272,448 at 17,000
    accepted(&spec(
        r#"{"type": "garch", "vol": "figarch", "truncation": 16000}"#,
    ));
    refused(
        &spec(r#"{"type": "garch", "vol": "figarch", "truncation": 17000}"#),
        &["272448 bytes", "max_state_bytes of 262144", "truncation"],
    );
    accepted(&spec(
        r#"{"type": "garch", "vol": "figarch", "truncation": 17000, "max_state_bytes": 300000}"#,
    ));
    refused(
        &spec(r#"{"type": "garch", "max_state_bytes": 67108865}"#),
        &["at most 67108864"],
    );
    refused(
        &spec(r#"{"type": "garch", "vol": "arch"}"#),
        &["vol must be"],
    );
    refused(
        &spec(r#"{"type": "garch", "dist": "ged"}"#),
        &["dist must be"],
    );
    refused(
        &spec(r#"{"type": "garch", "vol": "gjr", "dist": "skewt"}"#),
        &["does not take vol = \"gjr\""],
    );
    refused(
        &spec(r#"{"type": "garch", "start_rows": 100, "start_span": 50.0}"#),
        &["Leave one of them out"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "garch"}, "targets": ["a"], "features": [],
            "half_life": 10.0}"#,
        &["half_life does not apply to garch"],
    );
    refused(
        r#"{"name": "m", "model": {"type": "garch"}, "targets": ["a"], "features": ["x"]}"#,
        &["garch takes no features"],
    );
}

/// `hawkes`' window of clock units: a span above 0, beside no other window,
/// with `closed` only under it, a number where the kernels' decays are
/// rates, and a state whose fixed part alone fits `max_state_bytes`; what
/// the span holds is the data's, and is checked as the rows go in.
#[test]
fn a_hawkes_clock_window_is_refused_where_it_conflicts() {
    let spec = |model: &str, rest: &str| {
        format!(
            r#"{{"name": "m", "model": {model}, "targets": ["a", "b"], "features": [],
                "half_life": "inf", {rest} "clock": "t", "gap_cap": 1e9}}"#
        )
    };
    let hawkes = |more: &str| format!(r#"{{"type": "hawkes", "kernel_decays": [1.0, 5.0]{more}}}"#);
    accepted(&spec(&hawkes(r#", "window_size": 30.0"#), ""));
    accepted(&spec(
        &hawkes(r#", "window_size": 30.0, "closed": "both""#),
        "",
    ));
    for bad in ["0.0", "-1.0", r#""inf""#] {
        refused(
            &spec(&hawkes(&format!(r#", "window_size": {bad}"#)), ""),
            &["window_size must be a finite span of the clock above 0"],
        );
    }
    refused(
        &spec(&hawkes(r#", "window_size": 30.0, "window_events": 10"#), ""),
        &["are two windows, and the fit holds one"],
    );
    refused(
        &spec(&hawkes(r#", "closed": "both""#), ""),
        &["closed applies only under a window"],
    );
    // A duration is a span of a temporal clock, which a rate refuses: the
    // kernels are given by their half-lives there, as `half_life` is.
    refused(
        &spec(&hawkes(r#", "window_size": "5m""#), ""),
        &[
            "kernel_decays are rates per clock unit",
            "Give the kernels as kernel_half_lives",
        ],
    );
    let halves =
        |more: &str| format!(r#"{{"type": "hawkes", "kernel_half_lives": ["1s", "10s"]{more}}}"#);
    accepted(
        &spec(&halves(r#", "window_size": "5m", "closed": "both""#), "")
            .replace(
                r#""half_life": "inf""#,
                r#""half_life": "inf", "gap_cap": "1h""#,
            )
            .replace(r#", "gap_cap": 1e9"#, ""),
    );
    // Numbers of the clock's units for a numeric clock.
    accepted(&spec(
        r#"{"type": "hawkes", "kernel_half_lives": [0.5, 4.0], "window_size": 30.0}"#,
        "",
    ));
    // Both forms of the kernels, a duration beside a number, and a
    // half-life that is no span.
    refused(
        &spec(&hawkes(r#", "kernel_half_lives": [1.0]"#), ""),
        &["kernel_decays and kernel_half_lives both set the kernels"],
    );
    refused(
        &spec(r#"{"type": "hawkes", "kernel_half_lives": [1.0, 0.0]}"#, ""),
        &["each of kernel_half_lives must be a finite span of the clock above 0"],
    );
    refused(
        &spec(&halves(r#", "window_size": 30.0"#), ""),
        &["kernel_half_lives is a duration but gap_cap is a plain number"],
    );
    // A window forgets by dropping: a half-life beside it is refused.
    refused(
        &spec(&hawkes(r#", "window_size": 30.0"#), "")
            .replace(r#""half_life": "inf""#, r#""half_life": 10.0"#),
        &["takes no decay"],
    );
    // The fixed part alone is held to the cap.
    refused(
        &spec(
            &hawkes(r#", "window_size": 30.0, "max_state_bytes": 100"#),
            "",
        ),
        &["with no event in its window", "max_state_bytes of 100"],
    );
}
