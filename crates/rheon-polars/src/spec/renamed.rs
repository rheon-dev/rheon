//! The parameters a spec once took under another name, and the refusal that
//! names the new one in place of a bare "unknown field". Split from
//! `spec.rs` to keep it under the repository's file-size cap.

use super::{BUILDER_KEYWORDS, models};

/// The parameters task 144 renamed (docs/PLAN.md): a spec dict, a TOML file
/// or a windows config naming the old one is refused naming the new one, in
/// place of a bare "unknown field". No alias.
pub const RENAMED: &[(&str, &str)] = &[
    ("halflife", "half_life"),
    ("long_halflife", "long_half_life"),
    ("coef_halflife", "coef_half_life"),
    ("revert_halflife", "revert_half_life"),
    ("select_halflife", "select_half_life"),
    // Straight to the spec's `half_life`: `level_half_life` is refused too
    // (task 196).
    ("level_halflife", "half_life"),
    ("trend_halflife", "trend_half_life"),
    ("label_delay", "embargo"),
    ("max_dclock", "gap_cap"),
    ("window", "window_size"),
    ("min_periods", "min_weight"),
    ("emit_resid_z", "emit_zscore"),
    ("scale_features", "standardize"),
    ("on_clock_reset", "restart_after_step_back"),
    ("min_backwards_jump", "restart_after_step_back"),
    ("ridge_decay", "ridge_scale"),
    ("add_intercept", "fit_intercept"),
    ("max_cd_iters", "max_iter"),
    ("cd_tol", "tol"),
    ("reset", "change_action"),
    // Task 196 (docs/PLAN.md §18, N14): a count of rows says so, so that a
    // clock form could come later under the plain name; task 538 gave
    // `update_every`, `split_merge_every` and `permute_every` theirs.
    // N16: holt's level takes the spec's `half_life`, one knob under one
    // name.
    ("level_half_life", "half_life"),
];

/// The parameters renamed in one model, or whose old name another model
/// keeps: `rls`'s `ridge` is `delta` (docs/PLAN.md task 195, N11), and
/// `ridge` is still `ewridge`'s, `huber`'s and `quantile`'s; `kmeans`' `k` is
/// `n_clusters` and `hmm`'s `n_components`. The renames made for 1.0 (task
/// 244), read until 2.0, are refused here since. Named only where the
/// refused field's model takes the new name -- where serde's list of the
/// fields it expected holds it -- so a model with neither is told nothing
/// about another's rename.
pub const RENAMED_WHERE_EXPECTED: &[(&str, &str)] = &[
    ("ridge", "delta"),
    ("k", "n_clusters"),
    ("k", "n_components"),
    ("seed_rule", "init"),
    ("covariance", "covariance_type"),
    ("window_every", "snapshot_every"),
    ("max_rows_between_pca", "max_rows_between_pcas"),
    ("lasso_path", "penalties"),
    ("emission", "covariance_type"),
    ("prior_mean", "mean_prior"),
    ("prior_kappa", "mean_precision_prior"),
    ("prior_nu", "degrees_of_freedom_prior"),
    ("prior_scale", "covariance_prior"),
    ("span_rows", "window_rows"),
    ("reset_on_flag", "change_action"),
];

/// The values task 196 renamed (docs/PLAN.md §18): the model `type` tag
/// (N1, the builder's, the README's and the core's spelling) and an
/// `ew_cov` statistic (N10, beside `partial_corr`). A spec naming the old
/// one is refused naming the new one, as [`RENAMED`] does for a key.
pub const RENAMED_VALUES: &[(&str, &str)] = &[
    ("ew_ridge", "ewridge"),
    ("lagcorr", "lag_corr"),
    ("kmeanspp", "k-means++"),
    ("shared", "tied"),
    ("diagonal", "diag"),
];

/// `msg`, a deserialization error, with the rename named when the field it
/// refuses as unknown is an old name, or the model `type` it refuses as an
/// unknown variant is an old tag, and the table's key named when the field
/// is a builder keyword a table spells otherwise ([`BUILDER_KEYWORDS`]).
pub fn name_renamed(msg: &str) -> String {
    for (keyword, builder, key) in BUILDER_KEYWORDS {
        if msg.contains(&format!("unknown field `{keyword}`")) {
            return format!(
                "{msg}; `{keyword}` is {builder}'s builder keyword; a spec table says `{key}`"
            );
        }
    }
    // No citation: a wheel's user has no docs/PLAN.md (review 2026-10-06,
    // PC11).
    for (old, new) in RENAMED {
        if msg.contains(&format!("unknown field `{old}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    for (old, new) in RENAMED_WHERE_EXPECTED {
        if msg.contains(&format!("unknown field `{old}`")) && msg.contains(&format!("`{new}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    for (old, new) in RENAMED_VALUES {
        if msg.contains(&format!("unknown variant `{old}`")) && msg.contains(&format!("`{new}`")) {
            return format!("{msg}; {old} was renamed {new}");
        }
    }
    // A window given to a `tdigest` (whose fields are the ones serde lists
    // as expected): a merging digest has no windowed form, so the refusal
    // names the quantile that forgets (task 544).
    let tdigest = msg.contains("`compression`") && msg.contains("`quantiles`");
    for key in ["window_size", "window_rows", "span"] {
        if tdigest
            && msg.contains(&format!("unknown field `{key}`"))
            && let Err(why) = rheon_core::TDigest::check_no_span(Some(key))
        {
            return format!("{msg}; {}", models::tdigest_words(&why));
        }
    }
    msg.to_string()
}
