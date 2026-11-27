//! Optional extensions to [`super::BocpdCfg`] and [`super::Bocpd`]:
//!
//! - Negative-binomial run-length hazard (`nb_hazard: Option<(usize, f64)>`).
//! - Known-variance normal emission (`obs_var: Option<Vec<f64>>`).
//! - Runtime per-group state byte cap (`max_state_bytes: Option<u64>`, default
//!   256 KiB, ceiling 64 MiB).
//! - Known-variance predictive log-density and posterior mean helpers.
//! - Suffix-trimmed `Serialize` for [`super::BocpdCfg`] and [`super::Bocpd`]
//!   preserving pre-existing fixture bytes while keeping every new combination
//!   round-trip-safe in both named and compact msgpack encodings.

use serde::{Deserialize, Serialize};

use super::hazard::NbHazardTable;
use super::{Bocpd, BocpdCfg, BocpdEmission, Run};
use crate::clock_span::{DEFAULT_STATE_BYTES, check_max_state_bytes};

/// Optional configuration extensions held in one trailing field on
/// [`BocpdCfg`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BocpdCfgExtras {
    /// Negative-binomial run-length hazard `(k, p)` with `k >= 1` and
    /// `0 < p <= 1`. Per learned row only; refused beside `hazard`,
    /// `hazard_rows`, and `hazard_col`.
    pub nb_hazard: Option<(usize, f64)>,
    /// Known observation variance per feature (`len == 1` or `len == d`),
    /// positive and finite; active under `BocpdEmission::KnownVariance`.
    pub obs_var: Option<Vec<f64>>,
    /// Per-group byte cap for live runs (default 256 KiB, ceiling 64 MiB).
    pub max_state_bytes: Option<u64>,
}

impl BocpdCfgExtras {
    pub fn is_empty(&self) -> bool {
        self.nb_hazard.is_none() && self.obs_var.is_none() && self.max_state_bytes.is_none()
    }
}

impl BocpdCfg {
    /// Negative-binomial hazard parameters `(k, p)`, if configured.
    pub fn nb_hazard(&self) -> Option<(usize, f64)> {
        self.extras.as_deref().and_then(|e| e.nb_hazard)
    }

    /// Known observation variance vector (`len == 1` or `len == d`), if
    /// configured.
    pub fn obs_var(&self) -> Option<&[f64]> {
        self.extras.as_deref().and_then(|e| e.obs_var.as_deref())
    }

    /// Known observation variance of feature `i` (caller guarantees
    /// `KnownVariance`).
    #[inline]
    pub(super) fn obs_var_at(&self, i: usize) -> f64 {
        let v = self
            .obs_var()
            .expect("KnownVariance emission always carries obs_var");
        if v.len() == 1 { v[0] } else { v[i] }
    }

    /// Configured per-group state byte cap, if any.
    pub fn max_state_bytes(&self) -> Option<u64> {
        self.extras.as_deref().and_then(|e| e.max_state_bytes)
    }

    /// Validate all fields in `self.extras` and known-variance interactions.
    pub(super) fn validate_extras(&self, no: &mut crate::Refusals) {
        let d = self.n_features;
        let ex = self.extras.as_deref();

        if let Some((k, p)) = ex.and_then(|e| e.nb_hazard) {
            if k == 0 {
                no.refuse(
                    "bocpd: hazard_k must be at least 1, but it is 0. Set hazard_k to a positive \
                     integer shape for the negative-binomial run-length distribution.",
                );
            }
            if !(p.is_finite() && p > 0.0 && p <= 1.0) {
                no.refuse(format!(
                    "bocpd: hazard_p must be finite, above 0, and at most 1, but it is {p}. Set \
                     hazard_p to a probability in (0, 1]."
                ));
            }
            if self.hazard_on_clock {
                no.refuse(
                    "bocpd: hazard_kind=\"neg_binomial\" cannot be used with clock hazard: the \
                     negative-binomial hazard counts learned rows. Leave hazard out, or use \
                     hazard_kind=\"constant\".",
                );
            }
            if self.hazard_from_row {
                no.refuse(
                    "bocpd: hazard_kind=\"neg_binomial\" cannot be used with hazard_col. Leave \
                     hazard_col out, or use hazard_kind=\"constant\".",
                );
            }
        }

        no.take(check_max_state_bytes(
            "bocpd",
            ex.and_then(|e| e.max_state_bytes),
        ));

        match (self.emission, ex.and_then(|e| e.obs_var.as_deref())) {
            (BocpdEmission::KnownVariance, None) => {
                no.refuse(
                    "bocpd: known-variance emission needs obs_var, the known observation variance \
                     per feature. Give one positive value or one per feature.",
                );
            }
            (BocpdEmission::KnownVariance, Some(v)) => {
                if v.len() != 1 && v.len() != d {
                    no.refuse(format!(
                        "bocpd: obs_var must hold 1 value or {d} values (one per feature), but it \
                         has {}. Give one positive variance or one per feature.",
                        v.len()
                    ));
                } else if let Some(&bad) = v.iter().find(|x| !(x.is_finite() && **x > 0.0)) {
                    no.refuse(format!(
                        "bocpd: obs_var must be finite and above 0, but it holds {bad}. Give \
                         positive observation variances."
                    ));
                }
                if self.prior_nu.is_some() {
                    no.refuse(
                        "bocpd: degrees_of_freedom_prior does not apply when obs_var is set, \
                         since the observation variance is known. Leave degrees_of_freedom_prior \
                         out, or leave obs_var out.",
                    );
                }
                if self.robust_beta > 0.0 {
                    no.refuse(
                        "bocpd: robust_beta cannot be used with obs_var. Leave robust_beta out, \
                         or leave obs_var out.",
                    );
                }
            }
            (_, Some(_)) => {
                no.refuse(
                    "bocpd: obs_var applies only under known-variance emission (not \
                     covariance_type=\"full\" or robust_beta). Leave obs_var out, or use \
                     covariance_type=\"diag\" without robust_beta.",
                );
            }
            _ => {}
        }
    }
}

impl Bocpd {
    /// Rebuild the derived negative-binomial hazard table from `cfg`.
    pub(super) fn build_nb_table(cfg: &BocpdCfg, init_len: usize) -> NbHazardTable {
        match cfg.nb_hazard() {
            Some((k, p)) => NbHazardTable::build(k, p, cfg.max_run, init_len),
            None => NbHazardTable::default(),
        }
    }

    /// Whether this model uses the known-variance normal emission.
    #[inline]
    pub(super) fn known_var(&self) -> bool {
        self.cfg.emission == BocpdEmission::KnownVariance
    }

    /// Bytes held by `runs` live runs under this model's emission:
    /// - `KnownVariance`: `8 * (3 + 2d)` per run (`logjoint`, `len`, `n`,
    ///   `mean`, `mean_lo`; no `m2`).
    /// - `Diag` / `Robust`: `8 * (3 + 3d)` per run.
    /// - `Gaussian`: `8 * (3 + 2d + d^2)` per run.
    pub(super) fn run_bytes(&self, runs: usize) -> u64 {
        let d = self.cfg.n_features as u64;
        let words = match self.cfg.emission {
            BocpdEmission::KnownVariance => 3u64.saturating_add(2 * d),
            BocpdEmission::Diag | BocpdEmission::Robust => 3u64.saturating_add(3 * d),
            BocpdEmission::Gaussian => 3u64
                .saturating_add(2 * d)
                .saturating_add(d.saturating_mul(d)),
        };
        (runs as u64).saturating_mul(words).saturating_mul(8)
    }

    /// Runtime cap in bytes for live runs: `max_state_bytes` (default 256 KiB).
    pub(super) fn run_cap(&self) -> u64 {
        self.cap_for_test
            .or_else(|| self.cfg.max_state_bytes())
            .unwrap_or(DEFAULT_STATE_BYTES)
    }

    /// Posterior mean of feature `i` under `KnownVariance`:
    /// `μ₀ + (n τ₀² / (s² + n τ₀²)) · (x̄ − μ₀)`.
    #[inline]
    pub(super) fn known_var_mean_at(&self, run: &Run, mu0: &[f64], psi0: &[f64], i: usize) -> f64 {
        if run.n == 0.0 {
            return mu0[i];
        }
        let d = self.cfg.n_features;
        let tau2 = psi0[i * d + i];
        let s2 = self.cfg.obs_var_at(i);
        let n_tau2 = run.n * tau2;
        let frac = n_tau2 / (s2 + n_tau2);
        mu0[i] + frac * (run.mean[i] - mu0[i])
    }

    /// `(ln π(x), ln π(mode))` for one run under `KnownVariance`:
    /// `x_i | run ~ N(m_i, v_i + s_i^2)` where `v_i = 1 / (1/τ₀² + n/s²)`.
    pub(super) fn log_predictive_known_var(
        &self,
        run: &Run,
        x: &[f64],
        mu0: &[f64],
        psi0: &[f64],
    ) -> Option<(f64, f64)> {
        let d = self.cfg.n_features;
        let ln_tau = std::f64::consts::TAU.ln();
        let mut mode = 0.0;
        let mut quad = 0.0;
        for i in 0..d {
            let tau2 = psi0[i * d + i];
            let s2 = self.cfg.obs_var_at(i);
            let denom = s2 + run.n * tau2;
            let v = (tau2 * s2) / denom;
            let pv = v + s2;
            if !(pv > 0.0 && pv.is_finite()) {
                return None;
            }
            let m = if run.n == 0.0 {
                mu0[i]
            } else {
                mu0[i] + (run.n * tau2 / denom) * (run.mean[i] - mu0[i])
            };
            let diff = x[i] - m;
            if !diff.is_finite() {
                return None;
            }
            mode += -0.5 * (ln_tau + pv.ln());
            quad += (diff * diff) / pv;
        }
        let lp = mode - 0.5 * quad;
        (lp.is_finite() && mode.is_finite()).then_some((lp, mode))
    }

    /// Predict the exact number of live runs that will remain after inserting
    /// the new empty run, stepping `len` by +1, pruning below `prune_below`,
    /// and folding runs at `max_run - 1`, without mutating `self`.
    pub(super) fn count_runs_after(&self, new_joint: &[f64], z: f64) -> usize {
        if !z.is_finite() {
            return self.runs.len() + 1;
        }
        let pb = self.cfg.prune_below;
        let band = super::held::prune_band(pb);
        let cap = (self.cfg.max_run - 1) as f64;
        // Check whether any kept run passes `prune_below` (matching `prune`:
        // if none pass, nothing is pruned).
        let any_keep = pb > 0.0
            && new_joint
                .iter()
                .any(|&l| super::held::prune_keeps(l - z, pb, band));
        let mut below_cap = 0usize;
        let mut any_at_cap = false;
        for (idx, &l) in new_joint.iter().enumerate() {
            let kept = idx == 0 || !any_keep || super::held::prune_keeps(l - z, pb, band);
            if !kept {
                continue;
            }
            let len_after = if idx == 0 {
                0.0
            } else {
                self.runs[idx - 1].len + 1.0
            };
            if len_after < cap {
                below_cap += 1;
            } else {
                any_at_cap = true;
            }
        }
        below_cap + usize::from(any_at_cap)
    }
}

/// Custom `Serialize` for [`BocpdCfg`]:
///
/// - When `extras` is `None` (or empty), writes the exact pre-existing fields
///   with the exact pre-existing skip behavior (`hazard_on_clock`, `warm_clock`),
///   so every `v61..v71` and current `v72` fixture (`bocpd_warm_span` included)
///   is byte-identical.
/// - When `extras` is `Some` (and non-empty), emits `hazard_on_clock`,
///   `warm_clock`, and `extras` without holes so compact msgpack never slides.
impl Serialize for BocpdCfg {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let has_extras = self.extras.as_deref().is_some_and(|e| !e.is_empty());
        let write_clock = has_extras || self.hazard_on_clock;
        let write_warm = has_extras || self.warm_clock.is_some();
        let len = 13 + usize::from(write_clock) + usize::from(write_warm) + usize::from(has_extras);
        let mut st = serializer.serialize_struct("BocpdCfg", len)?;
        st.serialize_field("n_features", &self.n_features)?;
        st.serialize_field("hazard", &self.hazard)?;
        st.serialize_field("hazard_from_row", &self.hazard_from_row)?;
        st.serialize_field("emission", &self.emission)?;
        st.serialize_field("prior_mean", &self.prior_mean)?;
        st.serialize_field("prior_kappa", &self.prior_kappa)?;
        st.serialize_field("prior_nu", &self.prior_nu)?;
        st.serialize_field("prior_scale", &self.prior_scale)?;
        st.serialize_field("robust_beta", &self.robust_beta)?;
        st.serialize_field("prune_below", &self.prune_below)?;
        st.serialize_field("max_run", &self.max_run)?;
        st.serialize_field("min_weight", &self.min_weight)?;
        st.serialize_field("warm_rows", &self.warm_rows)?;
        if write_clock {
            st.serialize_field("hazard_on_clock", &self.hazard_on_clock)?;
        }
        if write_warm {
            st.serialize_field("warm_clock", &self.warm_clock)?;
        }
        if has_extras {
            st.serialize_field("extras", &self.extras)?;
        }
        st.end()
    }
}
