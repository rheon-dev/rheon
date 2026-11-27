//! A `bandit` spec: its settings, validation helpers, and the
//! [`rheon_core::BanditCfg`] it builds under a decay and Philox coordinates.

use serde::{Deserialize, Serialize};

use super::{ModelKind, Spec};

/// Contextual bandit (`rheon_core::Bandit`) settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BanditSpec {
    /// Ordered arm names (`>= 2` unique non-empty names).
    pub arms: Vec<String>,
    /// Policy word: `"linucb"` (default), `"lints"`, or `"eps_greedy"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    /// Exploration multiplier (`>= 0`) for `"linucb"` and `"lints"`; default `1.0`.
    /// Refused when `policy = "eps_greedy"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    /// Uniform exploration probability in `[0, 1]` for `"eps_greedy"`; default `0.1`.
    /// Refused when `policy` is `"linucb"` or `"lints"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epsilon: Option<f64>,
    /// Per-arm ridge prior `l2 > 0`; default `1.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l2: Option<f64>,
    /// Philox4x64-10 draw stream seed; default `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// State budget in bytes per group; default 256 KiB, ceiling 64 MiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// The words `policy` takes.
pub fn policy_of(word: Option<&str>) -> Result<rheon_core::BanditPolicy, String> {
    use rheon_core::BanditPolicy as P;
    match word {
        None | Some("linucb") => Ok(P::Linucb),
        Some("lints") => Ok(P::Lints),
        Some("eps_greedy") => Ok(P::EpsGreedy),
        Some(other) => Err(format!(
            "bandit: policy must be \"linucb\", \"lints\" or \"eps_greedy\", but it is \
             {other:?}. Use one of those words."
        )),
    }
}

/// True when `spec` uses `"eps_greedy"` (the policy that emits `prob`).
pub fn is_eps_greedy(b: &BanditSpec) -> bool {
    b.policy.as_deref() == Some("eps_greedy")
}

/// Validate `BanditSpec`-specific constraints that sit above `BanditCfg::validate`.
pub(super) fn validate_bandit(spec: &Spec, b: &BanditSpec, no: &mut rheon_core::Refusals) {
    let name = &spec.name;
    if b.arms.len() < 2 {
        no.refuse(format!(
            "spec {name:?}: arms must list at least 2 arms, but it lists {}. Name at least two \
             arms in arms.",
            b.arms.len()
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for a in &b.arms {
        if a.is_empty() {
            no.refuse(format!(
                "spec {name:?}: arms contains an empty arm name. Give every arm a non-empty name."
            ));
        }
        if a.contains('\0') {
            no.refuse(format!(
                "spec {name:?}: arm {a:?} holds a NUL character (\\0), which no column name of \
                 an Arrow export can hold. Rename the arm."
            ));
        }
        if !seen.insert(a.as_str()) {
            no.refuse(format!(
                "spec {name:?}: arms lists {a:?} more than once. Name each arm once in arms."
            ));
        }
    }
    if spec.targets.len() != 2 {
        no.refuse(format!(
            "spec {name:?}: bandit needs both reward and arm columns (2 target slots), but has \
             {}. Set both reward and arm.",
            spec.targets.len()
        ));
    }
    match policy_of(b.policy.as_deref()) {
        Ok(rheon_core::BanditPolicy::EpsGreedy) => {
            if b.alpha.is_some() {
                no.refuse(format!(
                    "spec {name:?}: alpha does not apply to policy=\"eps_greedy\", which explores \
                     uniformly with probability epsilon. Leave alpha out, or use policy=\"linucb\" \
                     or policy=\"lints\"."
                ));
            }
        }
        Ok(rheon_core::BanditPolicy::Linucb | rheon_core::BanditPolicy::Lints) => {
            if b.epsilon.is_some() {
                let p = b.policy.as_deref().unwrap_or("linucb");
                no.refuse(format!(
                    "spec {name:?}: epsilon does not apply to policy={p:?}, which explores using \
                     alpha. Leave epsilon out, or use policy=\"eps_greedy\"."
                ));
            }
        }
        Err(e) => no.refuse(format!(
            "spec {name:?}: {}",
            e.trim_start_matches("bandit: ")
        )),
    }
    if let Some(crate::spec::FloatOrList::List(_)) = &spec.min_weight {
        no.refuse(format!(
            "spec {name:?}: min_weight must be a single number for bandit, not a list. Give one \
             min_weight threshold."
        ));
    }
    if spec.emit_target_weight {
        no.refuse(format!(
            "spec {name:?}: emit_target_weight does not apply to bandit, which reports one shared \
             weight_sum. Leave emit_target_weight out."
        ));
    }
}

/// Build a [`rheon_core::BanditCfg`] for `spec` under `decay` and Philox coordinates.
pub fn bandit_cfg(
    spec: &Spec,
    decay: rheon_core::Decay,
    group_hash: u64,
    slot: u64,
) -> Result<rheon_core::BanditCfg, String> {
    let ModelKind::Bandit(b) = &spec.model else {
        return Err(format!(
            "bandit_cfg: spec {:?} is a {} spec, not a bandit spec. Pass a bandit spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let policy = policy_of(b.policy.as_deref())?;
    let cfg = rheon_core::BanditCfg {
        n_features: spec.k(),
        n_arms: b.arms.len(),
        fit_intercept: spec.fit_intercept,
        decay,
        policy,
        alpha: b.alpha.unwrap_or(1.0),
        epsilon: b.epsilon.unwrap_or(0.1),
        l2: b.l2.unwrap_or(1.0),
        min_weight: spec.min_periods_or_default(),
        seed: b.seed.unwrap_or(0),
        group_hash,
        slot,
        max_state_bytes: spec.model_max_state_bytes(b.max_state_bytes),
    };
    cfg.validate()?;
    Ok(cfg)
}
