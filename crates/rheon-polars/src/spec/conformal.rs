//! Conformal interval variants (`conformal` scalar or multi-level,
//! `conformal_symmetric`), standalone per-row conformal p-value
//! (`emit_conformal_p`, `conformal_p_window_rows`, `conformal_p_window_size`),
//! and top-level `max_state_bytes` budget wiring. Split from `spec.rs` to keep
//! it under the repository's file-size cap.

use rheon_core::{
    CONFORMAL_P_FIXED_BYTES, ConformalPRing, ConformalPWindow, DEFAULT_CONFORMAL_P_WINDOW_ROWS,
    DEFAULT_STATE_BYTES, STATE_BYTES_CEILING, check_max_state_bytes,
};
use serde::{Deserialize, Serialize};
use std::fmt;

use super::{ConformalRate, ModelKind, Num, Spec, num_label};

/// Conformal nominal coverage configuration: a single scalar level `0.9`
/// (preserving legacy `lo_<slot>`, `hi_<slot>`, `coverage_<slot>` field names)
/// or a sorted unique list `[0.8, 0.9, 0.95]` (emitting `lo_c<lvl>_<slot>`,
/// `hi_c<lvl>_<slot>`, `coverage_c<lvl>_<slot>`).
///
/// `Single(v)` serializes as a bare `f64` so all existing state files and specs
/// serialize identically.
#[derive(Debug, Clone, PartialEq)]
pub enum ConformalSpec {
    Single(f64),
    Multi(Vec<f64>),
}

impl Serialize for ConformalSpec {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            ConformalSpec::Single(v) => s.serialize_f64(*v),
            ConformalSpec::Multi(vs) => vs.serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for ConformalSpec {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = ConformalSpec;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a coverage level between 0 and 1, or a sorted list of coverage levels")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<ConformalSpec, E> {
                Ok(ConformalSpec::Single(v))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<ConformalSpec, E> {
                Ok(ConformalSpec::Single(v as f64))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<ConformalSpec, E> {
                Ok(ConformalSpec::Single(v as f64))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<ConformalSpec, E> {
                Num::from_word(v)
                    .map(|n| ConformalSpec::Single(n.0))
                    .ok_or_else(|| E::invalid_value(serde::de::Unexpected::Str(v), &self))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                seq: A,
            ) -> Result<ConformalSpec, A::Error> {
                Vec::<Num>::deserialize(serde::de::value::SeqAccessDeserializer::new(seq))
                    .map(|vs| ConformalSpec::Multi(vs.into_iter().map(|n| n.0).collect()))
            }
        }
        d.deserialize_any(V)
    }
}

impl ConformalSpec {
    /// Nominal coverage levels in ascending order.
    pub fn levels(&self) -> Vec<f64> {
        match self {
            ConformalSpec::Single(v) => vec![*v],
            ConformalSpec::Multi(vs) => vs.clone(),
        }
    }

    /// True when configured as a sequence of levels (even length 1).
    pub fn is_multi(&self) -> bool {
        matches!(self, ConformalSpec::Multi(_))
    }

    /// The scalar level when configured as `Single(v)`.
    pub fn single(&self) -> Option<f64> {
        match self {
            ConformalSpec::Single(v) => Some(*v),
            ConformalSpec::Multi(_) => None,
        }
    }
}

impl Spec {
    /// Nominal conformal coverage levels configured on this spec (empty if
    /// `conformal` is `None`).
    pub fn conformal_levels(&self) -> Vec<f64> {
        self.conformal
            .as_ref()
            .map_or_else(Vec::new, ConformalSpec::levels)
    }

    /// True when `conformal` was given as a sequence (`lo_c<lvl>_<slot>` naming).
    pub fn conformal_is_multi(&self) -> bool {
        self.conformal.as_ref().is_some_and(ConformalSpec::is_multi)
    }

    /// True when conformal runs on the legacy single-level symmetric path
    /// (`Persisted::conformal`), preserving 100% bit-identical behavior and
    /// state encoding.
    pub fn uses_legacy_conformal(&self) -> bool {
        self.conformal.as_ref().is_some_and(|c| !c.is_multi()) && self.conformal_symmetric
    }

    /// True when conformal runs on `Persisted::conformal_multi` (multi-level
    /// or asymmetric).
    pub fn uses_multi_conformal(&self) -> bool {
        self.conformal.is_some() && !self.uses_legacy_conformal()
    }

    /// Resolved window policy for standalone conformal p-value rings, or
    /// `None` when `emit_conformal_p` is false.
    pub fn conformal_p_window(&self) -> Option<ConformalPWindow> {
        if !self.emit_conformal_p {
            return None;
        }
        if let Some(w) = &self.conformal_p_window_size {
            let span_ns = match w {
                crate::span::Span::Duration(d) => Some(i128::from(d.nanos)),
                crate::span::Span::Units(_) => None,
            };
            Some(ConformalPWindow::Clock {
                span: w.value(),
                span_ns,
            })
        } else {
            Some(ConformalPWindow::Rows(
                self.conformal_p_window_rows
                    .unwrap_or(DEFAULT_CONFORMAL_P_WINDOW_ROWS),
            ))
        }
    }

    /// True when `self.model` carries its own model-level `max_state_bytes`
    /// field (kept for backwards compatibility with saved states and TOML).
    pub fn model_has_own_max_state_bytes(&self) -> bool {
        matches!(
            &self.model,
            ModelKind::Kalman { .. }
                | ModelKind::Huber { .. }
                | ModelKind::Quantile { .. }
                | ModelKind::Sgd { .. }
                | ModelKind::Holt { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Audit { .. }
                | ModelKind::Glm { .. }
                | ModelKind::Hawkes { .. }
                | ModelKind::Expectile { .. }
                | ModelKind::Garch { .. }
                | ModelKind::Dcc { .. }
                | ModelKind::Realized { .. }
                | ModelKind::Pca { .. }
                | ModelKind::PowerVariation { .. }
                | ModelKind::TDigest { .. }
                | ModelKind::DdSketch { .. }
                | ModelKind::EwmMoments { .. }
                | ModelKind::Ar { .. }
                | ModelKind::Isotonic { .. }
                | ModelKind::Spectral(_)
                | ModelKind::VectorAr(_)
        )
    }

    /// Effective per-group `max_state_bytes`: top-level `Spec::max_state_bytes`
    /// merged with any legacy model-level `max_state_bytes`.
    pub fn effective_max_state_bytes(&self) -> Option<u64> {
        self.max_state_bytes
            .or_else(|| self.model.model_max_state_bytes())
    }

    /// Group byte budget applied to `conformal_p` rings.
    pub fn conformal_p_cap_bytes(&self) -> u64 {
        self.effective_max_state_bytes()
            .unwrap_or(DEFAULT_STATE_BYTES)
    }

    /// Number of `(instance, slot)` rings this spec builds under
    /// `emit_conformal_p`.
    fn conformal_p_ring_count(&self) -> u64 {
        let n_models = self.decays().map_or(1, |d| d.len()).max(1);
        let nc = match &self.model {
            ModelKind::EwRidge {
                ridge,
                feature_sets,
                ..
            } => {
                let nr = ridge.as_ref().map_or(1, |r| r.to_vec().len()).max(1);
                let nfs = feature_sets.as_ref().map_or(1, Vec::len).max(1);
                nr * nfs
            }
            ModelKind::Huber { feature_sets, .. } => {
                feature_sets.as_ref().map_or(1, Vec::len).max(1)
            }
            ModelKind::Lasso { penalties, .. } => penalties.len().max(1),
            _ => 1,
        };
        let n_slots = self.m().max(1) * nc + self.forecast_slots();
        (n_models as u64).saturating_mul(n_slots as u64)
    }

    /// Per-ring byte budget for each `(instance, slot)` `ConformalPRing`.
    pub fn conformal_p_per_ring_cap(&self) -> u64 {
        let n_rings = self.conformal_p_ring_count().max(1);
        self.conformal_p_cap_bytes() / n_rings
    }

    /// Validate conformal coverage levels, `conformal_symmetric`,
    /// `conformal_rate`, `emit_conformal_p` windows, and top-level
    /// `max_state_bytes`.
    pub(super) fn validate_conformal_and_caps(&self, no: &mut rheon_core::Refusals) {
        let name = &self.name;
        match &self.conformal {
            None => {}
            Some(ConformalSpec::Single(c)) => {
                if !(*c > 0.0 && *c < 1.0) {
                    no.refuse(format!(
                        "spec {name:?}: conformal must be above 0 and below 1, but it is {c}. Set \
                         conformal to the share of rows whose target the interval should cover, \
                         such as 0.9."
                    ));
                }
            }
            Some(ConformalSpec::Multi(vs)) => {
                if vs.is_empty() {
                    no.refuse(format!(
                        "spec {name:?}: conformal cannot be an empty list. Give a coverage level \
                         in (0, 1) such as 0.9, or a sorted list such as [0.8, 0.9, 0.95]."
                    ));
                } else {
                    let bad: Vec<String> = vs
                        .iter()
                        .filter(|c| !(**c > 0.0 && **c < 1.0))
                        .map(|c| num_label(*c))
                        .collect();
                    if !bad.is_empty() {
                        no.refuse(format!(
                            "spec {name:?}: conformal levels must each be above 0 and below 1, \
                             but {} {}. Set each coverage level above 0 and below 1.",
                            if bad.len() == 1 {
                                "one of them is"
                            } else {
                                "these are not:"
                            },
                            bad.join(", ")
                        ));
                    } else if let Some(pair) = vs.windows(2).find(|w| w[0] >= w[1]) {
                        if pair[0] == pair[1] {
                            no.refuse(format!(
                                "spec {name:?}: conformal lists {} more than once. Give unique \
                                 coverage levels in ascending order (such as [0.8, 0.9, 0.95]).",
                                num_label(pair[0])
                            ));
                        } else {
                            no.refuse(format!(
                                "spec {name:?}: conformal levels must be sorted in ascending \
                                 order, but {} precedes {}. Sort conformal levels from lowest to \
                                 highest (such as [0.8, 0.9, 0.95]).",
                                num_label(pair[0]),
                                num_label(pair[1])
                            ));
                        }
                    }
                }
            }
        }
        if !self.conformal_symmetric && self.conformal.is_none() {
            no.refuse(format!(
                "spec {name:?}: conformal_symmetric=false applies only when conformal is set, \
                 but conformal is not set. Set conformal (such as 0.9 or [0.8, 0.9]), or leave \
                 conformal_symmetric at true."
            ));
        }
        if let Some(ConformalRate::Fixed(r)) = self
            .conformal_rate
            .filter(|r| !(r.base() > 0.0 && r.base().is_finite()))
        {
            no.refuse(format!(
                "spec {name:?}: conformal_rate must be finite and above 0, but it is {r}. Set \
                 conformal_rate to how fast the interval's width adapts, such as the default 0.05."
            ));
        }
        if self.conformal_rate.is_some() && self.conformal.is_none() {
            no.refuse(format!(
                "spec {name:?}: conformal_rate sets how fast the interval adapts, but conformal, \
                 its coverage, is not set. Set conformal, such as 0.9, or leave out \
                 conformal_rate."
            ));
        }

        // Conformal p-value window validation.
        if !self.emit_conformal_p {
            if self.conformal_p_window_rows.is_some() {
                no.refuse(format!(
                    "spec {name:?}: conformal_p_window_rows applies only when emit_conformal_p \
                     is on, but emit_conformal_p is off. Turn on emit_conformal_p, or leave out \
                     conformal_p_window_rows."
                ));
            }
            if self.conformal_p_window_size.is_some() {
                no.refuse(format!(
                    "spec {name:?}: conformal_p_window_size applies only when emit_conformal_p \
                     is on, but emit_conformal_p is off. Turn on emit_conformal_p, or leave out \
                     conformal_p_window_size."
                ));
            }
        } else {
            if self.conformal_p_window_rows.is_some() && self.conformal_p_window_size.is_some() {
                no.refuse(format!(
                    "spec {name:?}: conformal_p_window_rows and conformal_p_window_size are two \
                     windows for the conformal p-value ring, and it holds one. Leave one of them \
                     out."
                ));
            }
            if self.conformal_p_window_rows == Some(0) {
                no.refuse(format!(
                    "spec {name:?}: conformal_p_window_rows must be at least 1, but it is 0. Set \
                     conformal_p_window_rows to a positive row count (default \
                     {DEFAULT_CONFORMAL_P_WINDOW_ROWS})."
                ));
            }
            if let Some(w) = &self.conformal_p_window_size
                && !(w.value().is_finite() && w.value() > 0.0)
            {
                no.refuse(format!(
                    "spec {name:?}: conformal_p_window_size must be a finite span above 0, but \
                     it is {w}. Set conformal_p_window_size to a positive clock span or duration."
                ));
            }
        }

        // Range check when no model-level cap validator runs (models with
        // their own max_state_bytes run their own cap range check via
        // effective_max_state_bytes, except sgd without loss="huber" and
        // kalman without smooth_lag when emit_conformal_p is on).
        let model_checks_cap = self.model.takes_model_max_state_bytes()
            && !matches!(
                &self.model,
                ModelKind::Sgd { loss, .. } if loss.as_deref() != Some("huber") && self.emit_conformal_p
            )
            && !matches!(
                &self.model,
                ModelKind::Kalman { smooth_lag: None, .. } if self.emit_conformal_p
            );
        if let Some(cap) = self.max_state_bytes
            && !model_checks_cap
            && let Err(e) = check_max_state_bytes(self.model.kind_name(), Some(cap))
        {
            no.refuse(format!("spec {name:?}: {e}"));
        }

        // State cap check for emit_conformal_p rings.
        if self.emit_conformal_p && !self.model.predicts_no_target() {
            let cap = self.conformal_p_cap_bytes();
            if cap > 0 && cap <= STATE_BYTES_CEILING {
                let n_rings = self.conformal_p_ring_count().max(1);
                if let Some(w) = self
                    .conformal_p_window_size
                    .is_none()
                    .then(|| {
                        self.conformal_p_window_rows
                            .unwrap_or(DEFAULT_CONFORMAL_P_WINDOW_ROWS)
                    })
                    .filter(|&w| w > 0)
                {
                    let bytes = n_rings.saturating_mul(ConformalPRing::bytes_for(w));
                    if bytes > cap {
                        no.refuse(format!(
                            "spec {name:?}: emit_conformal_p with conformal_p_window_rows = {w} \
                             holds up to {bytes} bytes a group across {n_rings} slot(s), past \
                             max_state_bytes of {cap}. Lower conformal_p_window_rows, or raise \
                             max_state_bytes (at most 64 MiB)."
                        ));
                    }
                } else if self.conformal_p_window_size.is_some() {
                    let fixed = n_rings.saturating_mul(CONFORMAL_P_FIXED_BYTES);
                    if fixed > cap {
                        no.refuse(format!(
                            "spec {name:?}: emit_conformal_p holds {fixed} fixed bytes a group \
                             across {n_rings} slot(s), past max_state_bytes of {cap}. Raise \
                             max_state_bytes (at most 64 MiB)."
                        ));
                    }
                }
            }
        }
    }
}
