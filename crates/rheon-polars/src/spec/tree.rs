//! Hoeffding tree specs (`hoeffding_tree_classifier` and
//! `hoeffding_tree_regressor`): their settings, validation, and
//! `rheon-core` config builders.

use std::collections::HashSet;

use rheon_core::{
    HoeffdingTreeClassifierCfg, HoeffdingTreeRegressorCfg, LeafPrediction, Refusals, SplitCriterion,
};
use serde::{Deserialize, Serialize};

use super::{ModelKind, Spec, columns_counted};

/// Settings for an online Hoeffding tree classifier
/// (`rheon_core::HoeffdingTreeClassifier`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoeffdingTreeClassifierSpec {
    /// Ordered class labels the label column takes.
    pub classes: Vec<String>,
    /// Minimum weight accumulated in a leaf between split attempts (default 200).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grace_period: Option<f64>,
    /// Hoeffding bound confidence level in `(0, 1)` (default `1e-7`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,
    /// Tie-breaking threshold in `[0, 1]` (default `0.05`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau: Option<f64>,
    /// Split criterion: `"info_gain"` (default), `"gini"`, or `"hellinger"` (binary only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_criterion: Option<String>,
    /// Leaf prediction mode: `"nba"` (default), `"nb"`, or `"majority"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leaf_prediction: Option<String>,
    /// Minimum leaf weight before `"nb"` switches from majority class to Naive Bayes (default `0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nb_threshold: Option<f64>,
    /// Candidate split thresholds evaluated per numeric feature (default `10`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_splits: Option<usize>,
    /// Minimum weight fraction in each child branch for a split to be considered (default `0.01`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_branch_fraction: Option<f64>,
    /// Maximum majority-class share at which a leaf still attempts splits (default `0.99`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_share_to_split: Option<f64>,
    /// Include the null (no-split) candidate at merit `0.0` (default `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merit_preprune: Option<bool>,
    /// Optional maximum tree depth (`>= 1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    /// Per-group packed state cap in bytes (default 256 KB, ceiling 64 MiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// Settings for an online Hoeffding tree regressor
/// (`rheon_core::HoeffdingTreeRegressor`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoeffdingTreeRegressorSpec {
    /// Minimum weight accumulated in a leaf between split attempts (default 200).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grace_period: Option<f64>,
    /// Hoeffding bound confidence level in `(0, 1)` (default `1e-7`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,
    /// Tie-breaking threshold in `[0, 1]` (default `0.05`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau: Option<f64>,
    /// Quantization observer bucket radius (default `0.25`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// Maximum quantization slots per feature per active leaf (default `32`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_slots: Option<usize>,
    /// Minimum weight in each child branch for a split to be considered (default `5`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_samples_split: Option<f64>,
    /// Include the null (no-split) candidate at merit `0.0` (default `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub merit_preprune: Option<bool>,
    /// Optional maximum tree depth (`>= 1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_depth: Option<usize>,
    /// Per-group packed state cap in bytes (default 256 KB, ceiling 64 MiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// Parse `split_criterion` word.
pub fn split_criterion_of(word: Option<&str>) -> Result<SplitCriterion, String> {
    match word {
        None | Some("info_gain") => Ok(SplitCriterion::InfoGain),
        Some("gini") => Ok(SplitCriterion::Gini),
        Some("hellinger") => Ok(SplitCriterion::Hellinger),
        Some(other) => Err(format!(
            "hoeffding_tree_classifier: split_criterion must be \"info_gain\", \"gini\" or \
             \"hellinger\", but it is {other:?}. Use one of those words."
        )),
    }
}

/// Parse `leaf_prediction` word.
pub fn leaf_prediction_of(word: Option<&str>) -> Result<LeafPrediction, String> {
    match word {
        None | Some("nba") => Ok(LeafPrediction::NaiveBayesAdaptive),
        Some("majority") => Ok(LeafPrediction::MajorityClass),
        Some("nb") => Ok(LeafPrediction::NaiveBayes),
        Some(other) => Err(format!(
            "hoeffding_tree_classifier: leaf_prediction must be \"nba\", \"nb\" or \
             \"majority\", but it is {other:?}. Use one of those words."
        )),
    }
}

/// Build and validate a [`HoeffdingTreeClassifierCfg`] from a spec.
pub fn hoeffding_tree_classifier_cfg(spec: &Spec) -> Result<HoeffdingTreeClassifierCfg, String> {
    let ModelKind::HoeffdingTreeClassifier(t) = &spec.model else {
        return Err(format!(
            "hoeffding_tree_classifier_cfg: spec {:?} is a {} spec, not a \
             hoeffding_tree_classifier spec. Pass a hoeffding_tree_classifier spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let d = HoeffdingTreeClassifierCfg::default();
    let cfg = HoeffdingTreeClassifierCfg {
        n_features: spec.k(),
        n_classes: t.classes.len(),
        grace_period: t.grace_period.unwrap_or(d.grace_period),
        delta: t.delta.unwrap_or(d.delta),
        tau: t.tau.unwrap_or(d.tau),
        split_criterion: split_criterion_of(t.split_criterion.as_deref())?,
        leaf_prediction: leaf_prediction_of(t.leaf_prediction.as_deref())?,
        nb_threshold: t.nb_threshold.unwrap_or(d.nb_threshold),
        n_splits: t.n_splits.unwrap_or(d.n_splits),
        min_branch_fraction: t.min_branch_fraction.unwrap_or(d.min_branch_fraction),
        max_share_to_split: t.max_share_to_split.unwrap_or(d.max_share_to_split),
        merit_preprune: t.merit_preprune.unwrap_or(d.merit_preprune),
        max_depth: t.max_depth,
        min_weight: spec.min_periods_or_default(),
        max_state_bytes: spec.model_max_state_bytes(t.max_state_bytes),
    };
    cfg.validate()?;
    Ok(cfg)
}

/// Build and validate a [`HoeffdingTreeRegressorCfg`] from a spec.
pub fn hoeffding_tree_regressor_cfg(spec: &Spec) -> Result<HoeffdingTreeRegressorCfg, String> {
    let ModelKind::HoeffdingTreeRegressor(t) = &spec.model else {
        return Err(format!(
            "hoeffding_tree_regressor_cfg: spec {:?} is a {} spec, not a \
             hoeffding_tree_regressor spec. Pass a hoeffding_tree_regressor spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let d = HoeffdingTreeRegressorCfg::default();
    let cfg = HoeffdingTreeRegressorCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        grace_period: t.grace_period.unwrap_or(d.grace_period),
        delta: t.delta.unwrap_or(d.delta),
        tau: t.tau.unwrap_or(d.tau),
        radius: t.radius.unwrap_or(d.radius),
        max_slots: t.max_slots.unwrap_or(d.max_slots),
        min_samples_split: t.min_samples_split.unwrap_or(d.min_samples_split),
        merit_preprune: t.merit_preprune.unwrap_or(d.merit_preprune),
        max_depth: t.max_depth,
        min_weight: spec.min_periods_or_default(),
        max_state_bytes: spec.model_max_state_bytes(t.max_state_bytes),
    };
    cfg.validate()?;
    Ok(cfg)
}

fn prefix_lines(name: &str, prefix: &str, err: String) -> String {
    err.lines()
        .map(|l| format!("spec {name:?}: {}", l.trim_start_matches(prefix)))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Spec-level validation for `hoeffding_tree_classifier`.
pub(super) fn validate_hoeffding_tree_classifier(
    spec: &Spec,
    t: &HoeffdingTreeClassifierSpec,
    decay_key: &str,
    no: &mut Refusals,
) {
    if spec.half_life.is_some() || spec.lam.is_some() {
        no.refuse(format!(
            "spec {:?}: {decay_key} does not apply to hoeffding_tree_classifier, whose \
             Hoeffding-bound splits accumulate leaf statistics without exponential decay. \
             Leave {decay_key} out.",
            spec.name
        ));
    }
    if spec.targets.len() != 1 {
        no.refuse(format!(
            "spec {:?}: hoeffding_tree_classifier reads one target, the column of classes, but \
             targets names {}. Set targets to that column alone.",
            spec.name,
            columns_counted(spec.targets.len())
        ));
    }
    let bad_classes = t.classes.len() < 2;
    if bad_classes {
        no.refuse(format!(
            "spec {:?}: classes must list at least 2 classes, but it lists {}. List every class \
             the column of classes holds in classes.",
            spec.name,
            t.classes.len()
        ));
    }
    let mut seen = HashSet::new();
    if let Some(dup) = t.classes.iter().find(|c| !seen.insert(c.as_str())) {
        no.refuse(format!(
            "spec {:?}: classes must name each class once, but it lists {dup:?} more than once. \
             Remove the repeats from classes.",
            spec.name
        ));
    }
    if t.classes.iter().any(|c| c.is_empty()) {
        no.refuse(format!(
            "spec {:?}: classes must name each class, but one of its names is empty. Give that \
             class a name.",
            spec.name
        ));
    }
    let mut split_criterion = Ok(SplitCriterion::InfoGain);
    if t.split_criterion.is_some() {
        split_criterion = split_criterion_of(t.split_criterion.as_deref());
        if let Err(e) = &split_criterion {
            no.refuse(prefix_lines(
                &spec.name,
                "hoeffding_tree_classifier: ",
                e.clone(),
            ));
        }
    }
    let mut leaf_prediction = Ok(LeafPrediction::NaiveBayesAdaptive);
    if t.leaf_prediction.is_some() {
        leaf_prediction = leaf_prediction_of(t.leaf_prediction.as_deref());
        if let Err(e) = &leaf_prediction {
            no.refuse(prefix_lines(
                &spec.name,
                "hoeffding_tree_classifier: ",
                e.clone(),
            ));
        }
    }
    if !bad_classes && let (Ok(sc), Ok(lp)) = (split_criterion, leaf_prediction) {
        let d = HoeffdingTreeClassifierCfg::default();
        let cfg = HoeffdingTreeClassifierCfg {
            n_features: spec.k(),
            n_classes: t.classes.len(),
            grace_period: t.grace_period.unwrap_or(d.grace_period),
            delta: t.delta.unwrap_or(d.delta),
            tau: t.tau.unwrap_or(d.tau),
            split_criterion: sc,
            leaf_prediction: lp,
            nb_threshold: t.nb_threshold.unwrap_or(d.nb_threshold),
            n_splits: t.n_splits.unwrap_or(d.n_splits),
            min_branch_fraction: t.min_branch_fraction.unwrap_or(d.min_branch_fraction),
            max_share_to_split: t.max_share_to_split.unwrap_or(d.max_share_to_split),
            merit_preprune: t.merit_preprune.unwrap_or(d.merit_preprune),
            max_depth: t.max_depth,
            min_weight: spec.min_periods_or_default(),
            max_state_bytes: spec.model_max_state_bytes(t.max_state_bytes),
        };
        no.take(
            cfg.validate()
                .map_err(|e| prefix_lines(&spec.name, "hoeffding_tree_classifier: ", e)),
        );
    }
}

/// Spec-level validation for `hoeffding_tree_regressor`.
pub(super) fn validate_hoeffding_tree_regressor(spec: &Spec, decay_key: &str, no: &mut Refusals) {
    if spec.half_life.is_some() || spec.lam.is_some() {
        no.refuse(format!(
            "spec {:?}: {decay_key} does not apply to hoeffding_tree_regressor, whose \
             Hoeffding-bound splits accumulate leaf statistics without exponential decay. \
             Leave {decay_key} out.",
            spec.name
        ));
    }
    no.take(
        hoeffding_tree_regressor_cfg(spec)
            .map(|_| ())
            .map_err(|e| prefix_lines(&spec.name, "hoeffding_tree_regressor: ", e)),
    );
}
