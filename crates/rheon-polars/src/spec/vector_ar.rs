//! A `vector_ar` spec: its settings, and the [`rheon_core::VectorArCfg`]
//! they build under a decay.

use serde::{Deserialize, Serialize};

use super::{ModelClosed, ModelKind, Span, Spec, WindowBudgetSpec};

/// One Granger test a `vector_ar` spec reports on every row: the lags of
/// `causing` in the equations of `caused`, each a list of target columns.
/// `causing` absent is every target not in `caused`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalitySpec {
    pub caused: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causing: Option<Vec<String>>,
}

/// `vector_ar`'s settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorArSpec {
    /// The order `p`, lags `1..=p` of every series, in learned rows.
    pub lags: usize,
    /// The ridge on the lag and feature coefficients; absent is 0, least
    /// squares.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ridge: Option<f64>,
    /// The bytes a group's state may hold. Default 256 KB, at most 64 MiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
    /// Choose the order `q` in `0..=lags` by `"aic"`, `"bic"`, `"hqic"` or
    /// `"fpe"` at every solve, with Kish's effective count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub select: Option<String>,
    /// The Granger tests reported on every row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causality: Option<Vec<CausalitySpec>>,
    /// `"f"` (the default), `"wald"`, `"ssr_f"`, `"ssr_chi2"` or `"lr"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub causality_kind: Option<String>,
    /// Clock units between solves, as `ewridge`'s.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solve_every: Option<Span>,
    /// At most this many rows between solves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows_between_solves: Option<u32>,
    /// Clock units of history the fit sees, a hard cutoff, as `ewridge`'s
    /// `window_size`.
    #[serde(default)]
    pub window_size: Option<Span>,
    /// Which edge holds a row exactly `window_size` old.
    #[serde(default)]
    pub closed: ModelClosed,
    /// Clock units between the window's snapshots.
    #[serde(default)]
    pub snapshot_every: Option<Span>,
    /// At most this many rows between the window's snapshots.
    #[serde(default)]
    pub max_rows_between_snapshots: Option<u32>,
    /// Past this many MiB the group's snapshots thin or refuse.
    #[serde(default)]
    pub window_budget: Option<WindowBudgetSpec>,
}

/// The words `select` takes.
fn select_of(word: Option<&str>) -> Result<Option<rheon_core::VectorArSelect>, String> {
    use rheon_core::VectorArSelect as S;
    Ok(match word {
        None => None,
        Some("aic") => Some(S::Aic),
        Some("bic") => Some(S::Bic),
        Some("hqic") => Some(S::Hqic),
        Some("fpe") => Some(S::Fpe),
        Some(other) => {
            return Err(format!(
                "vector_ar: select must be \"aic\", \"bic\", \"hqic\" or \"fpe\", but it is \
                 {other:?}. Use one of those words."
            ));
        }
    })
}

/// The words `causality_kind` takes.
pub fn causality_kind_of(word: Option<&str>) -> Result<rheon_core::CausalityKind, String> {
    use rheon_core::CausalityKind as K;
    Ok(match word {
        None | Some("f") => K::F,
        Some("wald") => K::Wald,
        Some("ssr_f") => K::SsrF,
        Some("ssr_chi2") => K::SsrChi2,
        Some("lr") => K::Lr,
        Some(other) => {
            return Err(format!(
                "vector_ar: causality_kind must be \"f\", \"wald\", \"ssr_f\", \"ssr_chi2\" or \
                 \"lr\", but it is {other:?}. Use one of those words."
            ));
        }
    })
}

/// The target indices `names` name, in order, or a refusal naming the one
/// that is not a target.
pub fn target_indices(spec: &Spec, names: &[String], role: &str) -> Result<Vec<usize>, String> {
    let targets: Vec<&str> = spec.targets.iter().map(|t| t.as_str()).collect();
    names
        .iter()
        .map(|n| {
            targets.iter().position(|t| t == n).ok_or_else(|| {
                format!(
                    "vector_ar: causality names {n:?} as {role}, but it is not one of the \
                     targets {targets:?}. Name a target column."
                )
            })
        })
        .collect()
}

/// A test's two sets as target indices, `causing` absent read as every
/// target not caused.
pub fn test_of(spec: &Spec, t: &CausalitySpec) -> Result<rheon_core::VectorArTest, String> {
    let caused = target_indices(spec, &t.caused, "caused")?;
    let causing = match &t.causing {
        Some(c) => target_indices(spec, c, "causing")?,
        None => (0..spec.m()).filter(|j| !caused.contains(j)).collect(),
    };
    Ok(rheon_core::VectorArTest { caused, causing })
}

/// A `vector_ar` spec's [`rheon_core::VectorArCfg`] under `decay`: its
/// series are the spec's targets and its exogenous regressors the
/// features; the solve cadence is `ewridge`'s, defaults included; every
/// parameter check is the model's.
pub fn vector_ar_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
) -> Result<rheon_core::VectorArCfg, String> {
    let ModelKind::VectorAr(v) = &spec.model else {
        return Err(format!(
            "vector_ar_cfg: spec {:?} is a {} spec, not a vector_ar spec. Pass vector_ar_cfg a \
             vector_ar spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let causality = v
        .causality
        .iter()
        .flatten()
        .map(|t| test_of(spec, t))
        .collect::<Result<Vec<_>, _>>()?;
    let cfg = rheon_core::VectorArCfg {
        n_targets: spec.m(),
        n_features: spec.k(),
        lags: v.lags,
        fit_intercept: spec.fit_intercept,
        decay,
        ridge: v.ridge.unwrap_or(0.0),
        min_weight: spec.min_periods_or_default(),
        solve_every: v.solve_every.as_ref().map_or_else(
            || spec.solve_every_default(decay),
            crate::stream::solve_every_of,
        ),
        max_rows_between_solves: v.max_rows_between_solves.unwrap_or(u32::MAX),
        solve_share: spec.solve_share_default(v.solve_every.as_ref(), decay),
        max_state_bytes: v.max_state_bytes,
        select: select_of(v.select.as_deref())?,
        causality,
        causality_kind: causality_kind_of(v.causality_kind.as_deref())?,
        window: v.window_size.as_ref().map(Span::value),
        window_every: v.snapshot_every.as_ref().map(Span::value),
        max_rows_between_snapshots: v.max_rows_between_snapshots.map(|r| r as usize),
    };
    cfg.validate()?;
    Ok(cfg)
}
