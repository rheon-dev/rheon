//! What each model reads of the readiness statistics
//! (docs/WARMUP-AND-CONVERGENCE.md; docs/PLAN.md task 116): which models the
//! noise gate, its per-row field, the data shares and the coefficients'
//! standard errors exist for, and why the others have none. The settings
//! that read them are `Spec`'s fields; their refusals are `Spec::validate`'s.

use super::{ModelKind, Spec, num_label};

impl Spec {
    /// Whether this spec's model reads the noise gate's statistic
    /// ([`rheon_core::OnlineModel::error_inflation_into`]): the fits
    /// linear in the past targets whose estimation variance is known --
    /// `ewridge`, `rls` and linear-Gaussian `kalman` -- and `lasso`, from its
    /// active count (docs/PLAN.md task 116). The others are left to
    /// `min_weight`, and `max_error_inflation` is refused for them, with the
    /// reason [`Self::no_noise_statistic`] gives.
    pub fn has_error_inflation(&self) -> bool {
        match self.model {
            ModelKind::EwRidge { .. } | ModelKind::Rls { .. } | ModelKind::Lasso { .. } => true,
            ModelKind::Kalman { .. } => self.kalman_is_gaussian_identity(),
            _ => false,
        }
    }

    /// Whether this spec's model reads the statistic for one row's
    /// features ([`rheon_core::OnlineModel::row_error_inflation_into`]),
    /// the field `emit_error_inflation` writes: `ewridge` and `rls` from
    /// the factor their fit keeps, linear-Gaussian `kalman` from its `P`. Not
    /// `lasso`, which keeps no factor.
    pub fn has_row_error_inflation(&self) -> bool {
        match self.model {
            ModelKind::EwRidge { .. } | ModelKind::Rls { .. } => true,
            ModelKind::Kalman { .. } => self.kalman_is_gaussian_identity(),
            _ => false,
        }
    }

    /// Why this spec's model does not track how uncertain each prediction
    /// is, for the refusal of `max_error_inflation` (docs/PLAN.md task 116).
    pub(super) fn no_noise_statistic(&self) -> &'static str {
        match self.model {
            ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                "a gradient fit keeps no second moments of the features to read that \
                 uncertainty from"
            }
            ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                "a robust fit is not linear in its targets, so a least-squares fit's \
                 uncertainty does not hold for it"
            }
            ModelKind::Glm { .. } | ModelKind::Kalman { .. } => {
                "its prediction is a mean whose spread is its family's variance, not a \
                 least-squares fit's"
            }
            ModelKind::Expectile { .. } => {
                "an expectile fit weighs each row by the side its residual falls on, so a \
                 least-squares fit's uncertainty does not hold for it"
            }
            ModelKind::Krls(_) => {
                "a kernel fit operates in a reproducing kernel Hilbert space over a dynamic \
                 dictionary, so a linear least-squares uncertainty does not hold for it"
            }
            ModelKind::EwCov { .. }
            | ModelKind::Pca { .. }
            | ModelKind::Marginal { .. }
            | ModelKind::Deco { .. }
            | ModelKind::Rcov { .. }
            | ModelKind::CorrChange { .. } => {
                "its statistics are weighted means and covariances of the columns, not \
                 predictions of a target"
            }
            ModelKind::EwmMoments { .. } => {
                "its statistics are weighted moments of the columns, not predictions of a \
                 target"
            }
            ModelKind::Dependence(_) => {
                "its statistics are nonlinear dependence measures between features and targets, \
                 not predictions of a target"
            }
            ModelKind::HoeffdingTreeClassifier(_) | ModelKind::HoeffdingTreeRegressor(_) => {
                "a decision tree partitions the feature space into leaves rather than fitting a \
                 linear model with a Gram matrix"
            }
            _ => "it keeps no measure of a prediction's uncertainty",
        }
    }

    /// Why this spec's model cannot read one row's leverage, for the
    /// refusals of `emit_influence` and `emit_error_inflation`.
    pub(super) fn no_leverage(&self) -> &'static str {
        match self.model {
            ModelKind::Lasso { .. } => {
                "it keeps no inverse of its Gram to read one row's leverage from"
            }
            ModelKind::Huber { .. } | ModelKind::Quantile { .. } => {
                "a robust fit is not linear in its targets, so a least-squares leverage does \
                 not hold for it"
            }
            ModelKind::Glm { .. } => {
                "a generalized linear fit's leverage is on its working response, not on its \
                 target"
            }
            ModelKind::Kalman { .. } => {
                "a nonlinear Kalman filter's leverage is on its linear predictor, not on its \
                 target"
            }
            ModelKind::Expectile { .. } => {
                "an expectile fit's leverage is on its rows weighed by their sides, which \
                 change with the fit"
            }
            ModelKind::Krls(_) => {
                "a kernel fit keeps no linear feature Gram to read one row's leverage from"
            }
            ModelKind::Sgd { .. } | ModelKind::Pa { .. } | ModelKind::Ftrl { .. } => {
                "a gradient fit keeps no second moments of the features to read one row's \
                 leverage from"
            }
            _ => "it fits no coefficients of features to read one row's leverage from",
        }
    }

    /// Whether this spec's model reports its coefficients' sampling
    /// variances ([`rheon_core::OnlineModel::coef_variance`]), the field
    /// `emit_se_coef` writes: `ewridge`, `rls`, `kalman` and `ar`, whose
    /// fits are `ewridge`'s (docs/PLAN.md
    /// task 116, F), and `vector_ar`, whose are statsmodels' `bse`.
    pub fn has_se_coef(&self) -> bool {
        matches!(
            self.model,
            ModelKind::EwRidge { .. }
                | ModelKind::Rls { .. }
                | ModelKind::Kalman { .. }
                | ModelKind::Ar { .. }
                | ModelKind::VectorAr(_)
        )
    }

    /// Whether this spec's model reports `support_coef`
    /// ([`rheon_core::OnlineModel::support_coef`]): the fits whose solve
    /// inverts a Gram with a mean-form ridge on its diagonal, `ewridge`,
    /// `huber` and `quantile` (docs/WARMUP-AND-CONVERGENCE.md §2.2). Not
    /// `lasso`, an L1 penalty having no shrinkage matrix -- every active
    /// coefficient would read 1 at the default -- nor `ew_cov`, which has no
    /// coefficients.
    pub fn has_support_coef(&self) -> bool {
        matches!(
            self.model,
            ModelKind::EwRidge { .. } | ModelKind::Huber { .. } | ModelKind::Quantile { .. }
        )
    }

    /// Residual checks whose null a fit fails by construction (DIAG-11):
    /// off-mean expectiles for calibration/breaks, and non-Gaussian GLM or
    /// nonlinear Kalman families for specification/tails.
    pub(super) fn check_residual_assumptions(&self, no: &mut rheon_core::Refusals) {
        let name = &self.name;
        let off_mean = match self.model {
            ModelKind::Expectile { expectile, .. } if expectile != 0.5 => Some(expectile),
            _ => None,
        };
        if let Some(tau) = off_mean {
            for (on, key, what) in [
                (
                    self.emit_calibration,
                    "emit_calibration",
                    "its Wald test asks whether the target's mean is the prediction",
                ),
                (
                    self.emit_breaks,
                    "emit_breaks",
                    "its CUSUM sums residuals whose mean is 0 under no break",
                ),
            ] {
                if on {
                    no.refuse(format!(
                        "spec {name:?}: {key} tests a fit of the mean, but expectile {} \
                         is not the mean, so its residuals' mean is not 0 even where it \
                         fits exactly, and the test rejects on every row: {what}. Leave \
                         out {key}, or fit expectile 0.5, the mean.",
                        num_label(tau)
                    ));
                }
            }
        }
        let non_gaussian =
            matches!(self.model, ModelKind::Glm { .. }) || self.kalman_is_non_gaussian();
        if non_gaussian {
            let label = if matches!(self.model, ModelKind::Glm { .. }) {
                "a glm's"
            } else {
                "a non-Gaussian kalman's"
            };
            let mean_label = if matches!(self.model, ModelKind::Glm { .. }) {
                "a glm's"
            } else {
                "its"
            };
            for (on, key, what) in [
                (
                    self.emit_specification,
                    "emit_specification",
                    "Breusch-Pagan's null is a constant variance, and RESET's is read \
                     under one",
                ),
                (
                    self.emit_tails,
                    "emit_tails",
                    "Jarque-Bera's null is a normal residual",
                ),
            ] {
                if on {
                    no.refuse(format!(
                        "spec {name:?}: {key} tests residuals of constant variance and a \
                         normal shape, but {label} residuals have its family's variance, \
                         which moves with the mean, and its family's shape, so the test \
                         rejects on every row of a stream it fits exactly: {what}. Leave \
                         out {key}; emit_calibration and emit_breaks read {mean_label} mean."
                    ));
                }
            }
        }
    }
}
