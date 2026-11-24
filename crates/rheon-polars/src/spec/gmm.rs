//! `gmm`'s settings: a Gaussian mixture fitted by online EM
//! (`rheon_core::Gmm`), scikit-learn's `GaussianMixture` names.

use serde::{Deserialize, Serialize};

/// `gmm`'s settings. No targets: every column of interest goes in
/// `features`, and the outputs are the most responsible component, the
/// row's log-density, its Mahalanobis distance to that component and each
/// component's responsibility, read before the row is learned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GmmSpec {
    /// The number of components, at least 1.
    pub n_components: usize,
    /// `"full"` (the default: a whole covariance a component) or `"diag"`
    /// (its diagonal alone).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariance_type: Option<String>,
    /// Added to each covariance's diagonal; absent is 1e-6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reg_covar: Option<f64>,
    /// The BIC margin by which the fit with `n_components` must beat the fit
    /// with one fewer before the model starts; absent is 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_bic: Option<f64>,
    /// The seeded batch EM starts each try of the start runs, the best
    /// kept; absent is 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_init: Option<usize>,
    /// The seed of the start's k-means++ draws; absent is 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// The start's batch EM iterations at most; absent is 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iter: Option<usize>,
    /// The change in the start's mean log-likelihood at which its batch EM
    /// stops; absent is 1e-3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tol: Option<f64>,
    /// The bytes the start's buffer may hold in each group; absent is 256
    /// KiB. A buffer that reaches it without a start is withheld with its
    /// own reason and never forced to start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}
