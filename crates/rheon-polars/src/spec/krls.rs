//! A `krls` spec: its settings, and the [`rheon_core::KrlsCfg`] they build
//! under a decay.

use serde::{Deserialize, Serialize};

use super::{ModelClosed, ModelKind, Span, Spec};

/// `krls`'s settings (online kernel recursive least squares).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KrlsSpec {
    /// Kernel family: `"rbf"` (absent), `"laplacian"`, `"poly"`, or `"linear"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel: Option<String>,
    /// Inverse length scale / polynomial scale; absent is `1 / len(features)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gamma: Option<f64>,
    /// Polynomial degree (absent is `3`); refused unless `kernel = "poly"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degree: Option<u32>,
    /// Polynomial offset `coef0` (absent is `1.0`, `>= 0`); refused unless
    /// `kernel = "poly"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coef0: Option<f64>,
    /// Sum-scale ridge penalty `r > 0`; absent is `1.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ridge: Option<f64>,
    /// Maximum dictionary atoms in ALD mode (absent is `100`); refused beside
    /// a window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_atoms: Option<usize>,
    /// Relative ALD admission threshold in `[0, 1)` (absent is `1e-3`);
    /// refused beside a window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ald_threshold: Option<f64>,
    /// Sliding window of the last this many positive-weight rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_rows: Option<usize>,
    /// Sliding window of the last this long on the clock.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_size: Option<Span>,
    /// Which edge of `window_size` holds a row exactly that old (`"right"`,
    /// the default, or `"both"`).
    #[serde(default, skip_serializing_if = "ModelClosed::is_right")]
    pub closed: ModelClosed,
    /// Emit `n_atoms` (`Int64`, pre-row atom or window count). Default `false`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub emit_n_atoms: bool,
    /// Per-group state cap in bytes (default 256 KB, ceiling 64 MiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// Parse the `kernel` word into [`rheon_core::KernelKind`].
pub fn kernel_of(word: Option<&str>) -> Result<rheon_core::KernelKind, String> {
    use rheon_core::KernelKind as K;
    Ok(match word {
        None | Some("rbf") => K::Rbf,
        Some("laplacian") => K::Laplacian,
        Some("poly") => K::Poly,
        Some("linear") => K::Linear,
        Some(other) => {
            return Err(format!(
                "krls: kernel must be \"rbf\", \"laplacian\", \"poly\" or \"linear\", but it is \
                 {other:?}. Use one of those words."
            ));
        }
    })
}

/// Build and validate [`rheon_core::KrlsCfg`] from a `krls` [`Spec`] under
/// `decay`.
pub fn krls_cfg(spec: &Spec, decay: rheon_core::Decay) -> Result<rheon_core::KrlsCfg, String> {
    let ModelKind::Krls(k) = &spec.model else {
        return Err(format!(
            "spec {:?}: krls_cfg asked to build a {:?} spec. Pass a krls spec.",
            spec.name,
            spec.model.kind_name()
        ));
    };
    let cfg = rheon_core::KrlsCfg {
        n_features: spec.k(),
        n_targets: spec.m(),
        fit_intercept: spec.fit_intercept,
        kernel: kernel_of(k.kernel.as_deref())?,
        gamma: k.gamma,
        degree: k.degree,
        coef0: k.coef0,
        ridge: k.ridge.unwrap_or(rheon_core::KRLS_DEFAULT_RIDGE),
        max_atoms: k.max_atoms,
        ald_threshold: k.ald_threshold,
        window_rows: k.window_rows,
        window_clock: k.window_size.as_ref().map(Span::value),
        closed: k.closed.edge(),
        decay,
        min_weight: spec.min_periods_or_default(),
        emit_n_atoms: k.emit_n_atoms,
        max_state_bytes: k.max_state_bytes,
    };
    cfg.validate()?;
    Ok(cfg)
}
