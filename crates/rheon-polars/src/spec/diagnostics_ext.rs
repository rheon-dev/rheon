//! Validation for residual diagnostics (calibration, breaks, robust standard
//! errors, specification tests, ARCH LM, White, Spiegelhalter, tails, and
//! influence). Extracted from `spec.rs` to keep every source file under the
//! repository size cap.

use super::{ModelKind, Spec};

impl Spec {
    /// Validate diagnostic flags, lag orders, model compatibility, and state
    /// byte caps for this spec.
    pub(super) fn validate_diagnostics(
        &self,
        name: &str,
        predicts: bool,
        needs: &dyn Fn(&str, &str) -> String,
        no: &mut rheon_core::Refusals,
    ) {
        for (on, key) in [
            (self.emit_influence, "emit_influence"),
            (self.emit_cooks_distance, "emit_cooks_distance"),
            (self.emit_dfbetas, "emit_dfbetas"),
        ] {
            if on && predicts && !self.has_row_error_inflation() {
                no.refuse(format!(
                    "spec {name:?}: {key} needs a model that reads one row's leverage, as \
                     ewridge, rls and kalman do, but {} does not: {}. Leave out {key}, or \
                     use one of those models.",
                    self.model.kind_name(),
                    self.no_leverage()
                ));
            }
        }
        if self.emit_spiegelhalter && predicts && !self.is_logistic() {
            no.refuse(format!(
                "spec {name:?}: emit_spiegelhalter tests binary probability calibration against \
                 a 0/1 target, as ftrl, sgd(loss=\"logistic\") and glm(family=\"binomial\") \
                 produce, but {} does not. Leave out emit_spiegelhalter, or use one of those \
                 models.",
                self.model.kind_name()
            ));
        }
        self.check_robust_se(no, predicts);
        self.check_max_state_bytes(no);
        if predicts {
            self.check_residual_assumptions(no);
            if matches!(self.model, ModelKind::Glm { .. }) && self.emit_white {
                no.refuse(format!(
                    "spec {name:?}: emit_white tests residuals of constant variance and a \
                     normal shape, but a glm's residuals have its family's variance, \
                     which moves with the mean, and its family's shape, so the test \
                     rejects on every row of a stream it fits exactly: White's null is a \
                     constant variance. Leave out emit_white; emit_calibration and \
                     emit_breaks read a glm's mean."
                ));
            }
        }
        if self.ljung_box_lags.is_some() && !self.emit_specification {
            no.refuse(needs("ljung_box_lags", "emit_specification"));
        }
        if let Some(l) = self.ljung_box_lags {
            let skip = self.ljung_box_skip();
            let most = rheon_core::MAX_LAG.saturating_sub(skip);
            if l == 0 {
                no.refuse(format!(
                    "spec {name:?}: ljung_box_lags must be at least 1, but it is 0. Set \
                     ljung_box_lags to 1 or more, such as the default 10, or leave it out."
                ));
            } else if l > most {
                no.refuse(if skip == 0 {
                    format!(
                        "spec {name:?}: ljung_box_lags must be at most {most}, but it is {l}: \
                         memory for every lag is set aside before the first row. Lower \
                         ljung_box_lags to {most} or below."
                    )
                } else {
                    format!(
                        "spec {name:?}: ljung_box_lags must be at most {most}, since \
                         horizon_rows={} skips the first {skip} lags and memory for every lag \
                         is set aside before the first row, but it is {l}. Lower \
                         ljung_box_lags to {most} or below, or lower horizon_rows.",
                        self.horizon()
                    )
                });
            }
        }
        if self.arch_lm_lags.is_some() && !self.emit_arch_lm {
            no.refuse(needs("arch_lm_lags", "emit_arch_lm"));
        }
        if let Some(0) = self.arch_lm_lags {
            no.refuse(format!(
                "spec {name:?}: arch_lm_lags must be at least 1, but it is 0. Set \
                 arch_lm_lags to 1 or more, such as the default {}, or leave it out.",
                rheon_core::DEFAULT_ARCH_LM_LAGS
            ));
        }
        let n_slots_all = || {
            let n_models = self.decays().map_or(1, |d| d.len());
            n_models
                .saturating_mul(self.m())
                .saturating_mul(crate::stream::combos(self).len())
        };
        if self.emit_arch_lm && predicts {
            let lags = self.arch_lm_lags_or_default();
            if lags >= 1 {
                let bytes = n_slots_all().saturating_mul(rheon_core::ArchLm::state_bytes(lags));
                if bytes > rheon_core::SPEC_DIAG_CAP_BYTES {
                    no.refuse(format!(
                        "spec {name:?}: emit_arch_lm at arch_lm_lags={lags} would hold {bytes} \
                         bytes in each group across {} slots, past the {} byte cap. Lower \
                         arch_lm_lags or reduce the grid size.",
                        n_slots_all(),
                        rheon_core::SPEC_DIAG_CAP_BYTES
                    ));
                }
            }
        }
        if self.white_interaction_terms.is_some() && !self.emit_white {
            no.refuse(needs("white_interaction_terms", "emit_white"));
        }
        if self.emit_white && predicts && !matches!(self.model, ModelKind::Glm { .. }) {
            let k = self.k();
            if k == 0 {
                no.refuse(format!(
                    "spec {name:?}: emit_white regresses squared residuals on the features and \
                     their squares, but {} has no features. Leave out emit_white.",
                    self.model.kind_name()
                ));
            } else {
                let full = self.white_interaction_terms_or_default();
                let bytes =
                    n_slots_all().saturating_mul(rheon_core::WhiteTest::state_bytes(k, full));
                if bytes > rheon_core::SPEC_DIAG_CAP_BYTES {
                    if full {
                        no.refuse(format!(
                            "spec {name:?}: emit_white with {k} features across {} slots would \
                             hold {bytes} bytes in each group, past the {} byte cap. Set \
                             white_interaction_terms=False (allows up to {} features for one \
                             slot), or reduce features to {} or fewer.",
                            n_slots_all(),
                            rheon_core::SPEC_DIAG_CAP_BYTES,
                            rheon_core::MAX_WHITE_SQUARES_K,
                            rheon_core::MAX_WHITE_FULL_K
                        ));
                    } else {
                        no.refuse(format!(
                            "spec {name:?}: emit_white with {k} features across {} slots would \
                             hold {bytes} bytes in each group, past the {} byte cap. Reduce \
                             features to {} or fewer, or reduce the grid size.",
                            n_slots_all(),
                            rheon_core::SPEC_DIAG_CAP_BYTES,
                            rheon_core::MAX_WHITE_SQUARES_K
                        ));
                    }
                }
            }
        }
        if self.horizon_rows.is_some() && self.horizon_readers().is_empty() {
            no.refuse(format!(
                "spec {name:?}: horizon_rows applies only when emit_calibration, emit_breaks, \
                 emit_specification or emit_robust_se is on, but none is. Turn on the \
                 diagnostic that should read it, or leave out horizon_rows."
            ));
        }
        if let Some(h) = self.horizon_rows.filter(|h| *h > rheon_core::MAX_LAG / 2) {
            no.refuse(format!(
                "spec {name:?}: horizon_rows must be at most {}, but it is {h}: the diagnostics \
                 read twice as many lags, and memory for every lag is set aside before the \
                 first row. Lower horizon_rows to {} or below.",
                rheon_core::MAX_LAG / 2,
                rheon_core::MAX_LAG / 2
            ));
        }
        if self.robust_se_lags.is_some() && !self.emit_robust_se {
            no.refuse(needs("robust_se_lags", "emit_robust_se"));
        }
        if let Some(l) = self.robust_se_lags {
            no.take(
                rheon_core::check_lag_ceiling("robust_se_lags", l)
                    .map_err(|e| format!("spec {name:?}: {e}")),
            );
        }
        if self.emit_error_inflation && !self.has_row_error_inflation() {
            no.refuse(format!(
                "spec {name:?}: emit_error_inflation needs a model that reads one row's \
                 leverage, as ewridge, rls and kalman do, but {} does not: {}. Leave out \
                 emit_error_inflation, or use one of those models.",
                self.model.kind_name(),
                self.no_leverage()
            ));
        }
    }
}
