//! A `dependence` spec: its settings, word parsing, and the
//! [`rheon_core::DependenceCfg`] it builds under a decay.

use serde::{Deserialize, Serialize};

use super::{ModelClosed, ModelKind, Span, Spec};

/// `dependence`'s settings: binned mutual information (with optional
/// Miller-Madow bias correction) and windowed rank correlations (Spearman's
/// rho, Kendall's tau-b, and Chatterjee's xi).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceSpec {
    /// Measures to compute for every `(feature, target)` pair: a non-empty
    /// subset of `"mi"`, `"mi_mm"`, `"spearman"`, `"kendall"`, `"xi"`.
    /// Absent defaults to `["mi"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measures: Option<Vec<String>>,
    /// Number of feature bins for MI (default 16, from 2 to 256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bins: Option<usize>,
    /// Number of target bins for MI (defaults to `bins`, from 2 to 256).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_bins: Option<usize>,
    /// Rule for learning MI bin edges: `"quantile"` (default) or `"fixed"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_rule: Option<String>,
    /// Positive-weight rows held to learn MI bin edges when edges are not given.
    /// Defaults to `min(1000, window_rows)` under `window_rows`, else `1000`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_warm_rows: Option<usize>,
    /// Clock span held to learn MI bin edges (at least 2 rows); mutually
    /// exclusive with `bin_warm_rows`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_warm_span: Option<Span>,
    /// Fixed interior bin edges per feature, in `features` order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_edges: Option<Vec<Vec<f64>>>,
    /// Fixed interior bin edges per target, in `targets` order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_bin_edges: Option<Vec<Vec<f64>>>,
    /// Exact row window of the last this many positive-weight rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_rows: Option<usize>,
    /// Exact clock window of the last this long on the clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_size: Option<Span>,
    /// Which edge of `window_size` holds a row exactly that old: `"right"`
    /// (default) or `"both"`. Skipped when `"right"`.
    #[serde(default, skip_serializing_if = "ModelClosed::is_right")]
    pub closed: ModelClosed,
    /// Clock span between refreshes of `"spearman"` and `"xi"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_every: Option<Span>,
    /// Learned rows between refreshes of `"spearman"` and `"xi"` (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_every_rows: Option<usize>,
    /// Maximum state bytes per group (default 256 KB, ceiling 64 MiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

impl DependenceSpec {
    /// Resolve `measures` to core enum variants, defaulting to `[Mi]`.
    pub fn resolved_measures(&self) -> Result<Vec<rheon_core::DependenceMeasure>, String> {
        use rheon_core::DependenceMeasure as M;
        let Some(words) = &self.measures else {
            return Ok(vec![M::Mi]);
        };
        if words.is_empty() {
            return Err(
                "dependence: measures must list at least one measure, but it is empty. Choose \
                 from \"mi\", \"mi_mm\", \"spearman\", \"kendall\", and \"xi\"."
                    .to_string(),
            );
        }
        let mut out = Vec::with_capacity(words.len());
        for w in words {
            let m = match w.as_str() {
                "mi" => M::Mi,
                "mi_mm" => M::MiMm,
                "spearman" => M::Spearman,
                "kendall" => M::Kendall,
                "xi" => M::Xi,
                other => {
                    return Err(format!(
                        "dependence: {}",
                        super::words_refusal(
                            "measures",
                            &["mi", "mi_mm", "spearman", "kendall", "xi"],
                            other,
                        )
                    ));
                }
            };
            if out.contains(&m) {
                return Err(format!(
                    "dependence: measures must list each measure once, but it lists {w:?} more \
                     than once. Remove the repeat from measures."
                ));
            }
            out.push(m);
        }
        Ok(out)
    }

    /// The resolved measures for schema/field generation (falls back to `[Mi]`
    /// if the spec has not been validated yet).
    pub fn measures_for_fields(&self) -> Vec<rheon_core::DependenceMeasure> {
        self.resolved_measures()
            .unwrap_or_else(|_| vec![rheon_core::DependenceMeasure::Mi])
    }
}

fn bin_rule_of(word: Option<&str>) -> Result<rheon_core::BinRule, String> {
    match word {
        None | Some("quantile") => Ok(rheon_core::BinRule::Quantile),
        Some("fixed") => Ok(rheon_core::BinRule::Fixed),
        Some(other) => Err(format!(
            "dependence: {}",
            super::words_refusal("bin_rule", &["quantile", "fixed"], other)
        )),
    }
}

/// Build and validate [`rheon_core::DependenceCfg`] for a `dependence` spec
/// under `decay`.
pub fn dependence_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::DependenceCfg, String> {
    let ModelKind::Dependence(d) = &spec.model else {
        return Err(format!(
            "dependence_cfg: spec {:?} is a {} spec, not a dependence spec. Pass dependence_cfg \
             a dependence spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let measures = d.resolved_measures()?;
    let has_mi = measures.iter().any(|m| m.is_mi());
    let mut no = rheon_core::Refusals::new();

    if !has_mi
        && (d.bins.is_some()
            || d.target_bins.is_some()
            || d.bin_rule.is_some()
            || d.bin_warm_rows.is_some()
            || d.bin_warm_span.is_some()
            || d.bin_edges.is_some()
            || d.target_bin_edges.is_some())
    {
        no.refuse(
            "dependence: binning settings (bins, target_bins, bin_rule, bin_warm_rows, \
             bin_warm_span, bin_edges, target_bin_edges) apply only when \"mi\" or \"mi_mm\" is \
             in measures. Add \"mi\" to measures, or leave those settings out.",
        );
    } else if has_mi {
        if d.bin_edges.is_some() && d.bins.is_some() {
            no.refuse(
                "dependence: bin_edges fixes the feature bins, so bins cannot also be set. Leave \
                 out bins, or leave out bin_edges to learn edges.",
            );
        }
        if d.target_bin_edges.is_some() && d.target_bins.is_some() {
            no.refuse(
                "dependence: target_bin_edges fixes the target bins, so target_bins cannot also \
                 be set. Leave out target_bins, or leave out target_bin_edges to learn edges.",
            );
        }
        if d.bin_edges.is_some()
            && d.target_bin_edges.is_some()
            && (d.bin_rule.is_some() || d.bin_warm_rows.is_some() || d.bin_warm_span.is_some())
        {
            no.refuse(
                "dependence: bin_edges and target_bin_edges fix all bin edges, so bin_rule, \
                 bin_warm_rows, and bin_warm_span have nothing to learn. Leave those learning \
                 settings out, or omit bin_edges/target_bin_edges.",
            );
        }
    }

    if d.closed == ModelClosed::Both && d.window_size.is_none() {
        no.refuse(
            "dependence: closed=\"both\" applies only when window_size is set, because it decides \
             whether a row exactly window_size clock units old stays in the window. Set \
             window_size, or leave closed at its default \"right\".",
        );
    }

    let bin_rule = match bin_rule_of(d.bin_rule.as_deref()) {
        Ok(r) => r,
        Err(e) => {
            no.refuse(e);
            rheon_core::BinRule::Quantile
        }
    };

    let bins = d.bins.unwrap_or(rheon_core::DEPENDENCE_DEFAULT_BINS);
    let target_bins = d.target_bins.unwrap_or(bins);

    let cfg = rheon_core::DependenceCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        measures,
        decay,
        min_weight: spec.min_periods_or_default(),
        bins,
        target_bins,
        bin_rule,
        bin_warm_rows: d.bin_warm_rows,
        bin_warm_span: d.bin_warm_span.as_ref().map(Span::value),
        bin_edges: d.bin_edges.clone(),
        target_bin_edges: d.target_bin_edges.clone(),
        window_rows: d.window_rows,
        window_clock: d.window_size.as_ref().map(Span::value),
        closed: d.closed.edge(),
        rank_every_rows: d.rank_every_rows,
        rank_every_clock: d.rank_every.as_ref().map(Span::value),
        max_state_bytes: d.max_state_bytes,
    };
    no.take(cfg.validate());
    no.finish()?;
    Ok(cfg)
}
