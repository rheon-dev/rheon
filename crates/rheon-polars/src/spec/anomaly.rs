//! Streaming anomaly detector specs: `half_space_trees`
//! ([`rheon_core::HalfSpaceTrees`]) and `random_cut_forest`
//! ([`rheon_core::RandomCutForest`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{ModelKind, Spec};

/// Half-Space Trees (`rheon_core::HalfSpaceTrees`) settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HalfSpaceTreesSpec {
    /// Number of half-space trees; default `10`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_trees: Option<usize>,
    /// Depth of every full binary tree (`1..=16`); default `8`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<usize>,
    /// Positive-weight rows per tumbling mass window; default `250`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_rows: Option<usize>,
    /// Optional per-feature workspace bounds `(lo, hi)`. Unspecified features
    /// default to `(0.0, 1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<BTreeMap<String, [f64; 2]>>,
    /// Philox4x64-10 draw stream seed; default `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// State budget in bytes per group; default 2 MiB, ceiling 64 MiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// Robust Random Cut Forest (`rheon_core::RandomCutForest`) settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomCutForestSpec {
    /// Number of random cut trees; default `40`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_trees: Option<usize>,
    /// Maximum points held in each tree's FIFO ring (`>= 2`); default `256`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_size: Option<usize>,
    /// Consecutive learned rows concatenated into each point (`>= 1`); default `1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shingle_size: Option<usize>,
    /// Philox4x64-10 draw stream seed; default `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// State budget in bytes per group; default 9 MiB, ceiling 64 MiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// Resolve `limits` against `spec.features`, refusing any key not in `features`.
fn resolve_limits(spec: &Spec, h: &HalfSpaceTreesSpec) -> Result<Vec<(f64, f64)>, String> {
    if let Some(map) = &h.limits {
        for col in map.keys() {
            if !spec.features.iter().any(|f| f == col) {
                return Err(format!(
                    "half_space_trees: limits names {col:?}, which is not in features {:?}. \
                     Name only feature columns in limits.",
                    spec.features
                ));
            }
        }
    }
    let limits = spec
        .features
        .iter()
        .map(|f| {
            h.limits
                .as_ref()
                .and_then(|m| m.get(f))
                .map_or((0.0, 1.0), |[lo, hi]| (*lo, *hi))
        })
        .collect();
    Ok(limits)
}

/// Build a [`rheon_core::HalfSpaceTreesCfg`] for `spec` and Philox coordinates.
pub fn half_space_trees_cfg(
    spec: &Spec,
    group_hash: u64,
    slot: u64,
) -> Result<rheon_core::HalfSpaceTreesCfg, String> {
    let ModelKind::HalfSpaceTrees(h) = &spec.model else {
        return Err(format!(
            "half_space_trees_cfg: spec {:?} is a {} spec, not a half_space_trees spec. Pass a \
             half_space_trees spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let limits = resolve_limits(spec, h)?;
    let cfg = rheon_core::HalfSpaceTreesCfg {
        n_features: spec.k(),
        n_trees: h.n_trees.unwrap_or(10),
        height: h.height.unwrap_or(8),
        window_rows: h.window_rows.unwrap_or(250),
        limits,
        min_weight: spec.min_periods_or_default(),
        seed: h.seed.unwrap_or(0),
        group_hash,
        slot,
        max_state_bytes: h.max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}

/// Build a [`rheon_core::RandomCutForestCfg`] for `spec` and Philox coordinates.
pub fn random_cut_forest_cfg(
    spec: &Spec,
    group_hash: u64,
    slot: u64,
) -> Result<rheon_core::RandomCutForestCfg, String> {
    let ModelKind::RandomCutForest(r) = &spec.model else {
        return Err(format!(
            "random_cut_forest_cfg: spec {:?} is a {} spec, not a random_cut_forest spec. Pass a \
             random_cut_forest spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let cfg = rheon_core::RandomCutForestCfg {
        n_features: spec.k(),
        n_trees: r.n_trees.unwrap_or(40),
        tree_size: r.tree_size.unwrap_or(256),
        shingle_size: r.shingle_size.unwrap_or(1),
        min_weight: spec.min_periods_or_default(),
        seed: r.seed.unwrap_or(0),
        group_hash,
        slot,
        max_state_bytes: r.max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}
