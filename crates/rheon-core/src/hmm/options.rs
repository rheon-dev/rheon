//! Optional extensions to [`super::HmmCfg`] and [`super::Hmm`]:
//!
//! - [`HmmCovariance`] (`"full"`, `"tied"`, `"diag"`, `"spherical"`),
//!   kept separate from [`crate::Covariance`] so `ew_class` is untouched.
//! - Partial learning [`HmmParams`] (subsets of `"tmc"`).
//! - Conjugate prior weights (`means_weight`, `covariance_prior`, `covariance_weight`).
//! - Fixed-lag smoothing (`smooth_lag`, `max_state_bytes`).
//! - Suffix-trimmed `Serialize` for [`super::HmmCfg`] and [`super::Hmm`] so
//!   every pre-existing schema fixture writes byte-identical output while
//!   every combination of old and new optional fields round-trips cleanly
//!   through both named and compact msgpack encodings.

use serde::{Deserialize, Serialize};

use super::smooth::HmmSmoother;
use super::spherical::EwCovSpherical;
use super::{Hmm, HmmCfg};
use crate::Covariance;
use crate::clock_span::{DEFAULT_STATE_BYTES, STATE_BYTES_CEILING};

/// The shape of each hidden state's covariance matrix in [`Hmm`].
///
/// The first three variants match [`Covariance`] in name and wire order so
/// every saved state from earlier builds decodes identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HmmCovariance {
    #[default]
    Full,
    #[serde(rename = "tied")]
    Shared,
    #[serde(rename = "diag")]
    Diagonal,
    /// Isotropic covariance `s_l I`: one scalar variance per state (`8` bytes).
    Spherical,
}

impl From<Covariance> for HmmCovariance {
    fn from(c: Covariance) -> Self {
        match c {
            Covariance::Full => Self::Full,
            Covariance::Shared => Self::Shared,
            Covariance::Diagonal => Self::Diagonal,
        }
    }
}

impl HmmCovariance {
    /// Parse a spec's `covariance_type` word for `hmm`.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "full" => Ok(Self::Full),
            "tied" => Ok(Self::Shared),
            "diag" => Ok(Self::Diagonal),
            "spherical" => Ok(Self::Spherical),
            other => Err(format!(
                "covariance_type must be \"full\", \"tied\", \"diag\" or \"spherical\", but it \
                 is {other:?}. Use one of those words."
            )),
        }
    }
}

/// Which components of an [`Hmm`] update on each learned row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HmmParams {
    /// Transition counts `A`.
    pub t: bool,
    /// Emission means `μ`.
    pub m: bool,
    /// Emission covariances `Σ`.
    pub c: bool,
}

impl HmmParams {
    pub const NONE: Self = Self {
        t: false,
        m: false,
        c: false,
    };
    pub const ALL: Self = Self {
        t: true,
        m: true,
        c: true,
    };

    /// Whether any parameter learns.
    #[inline]
    pub fn any(self) -> bool {
        self.t || self.m || self.c
    }

    /// Parse a `params` string of letters from `"tmc"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut out = Self::NONE;
        for ch in s.chars() {
            match ch {
                's' => {
                    return Err(
                        "hmm: params cannot include 's': the initial state distribution is \
                         uniform (1/n_components) and is not learned. Use letters from \"tmc\" \
                         (transitions, means, covariances), or \"\" to learn nothing."
                            .into(),
                    );
                }
                't' => {
                    if out.t {
                        return Err(format!(
                            "hmm: params holds duplicate letter 't' in {s:?}. List each letter \
                             from \"tmc\" at most once."
                        ));
                    }
                    out.t = true;
                }
                'm' => {
                    if out.m {
                        return Err(format!(
                            "hmm: params holds duplicate letter 'm' in {s:?}. List each letter \
                             from \"tmc\" at most once."
                        ));
                    }
                    out.m = true;
                }
                'c' => {
                    if out.c {
                        return Err(format!(
                            "hmm: params holds duplicate letter 'c' in {s:?}. List each letter \
                             from \"tmc\" at most once."
                        ));
                    }
                    out.c = true;
                }
                other => {
                    return Err(format!(
                        "hmm: params holds unknown letter {other:?} in {s:?}. Use letters from \
                         \"tmc\" (transitions, means, covariances), or \"\" to learn nothing."
                    ));
                }
            }
        }
        Ok(out)
    }
}

/// Optional configuration extensions held in one trailing field on [`HmmCfg`]
/// so pre-existing states keep their exact layout and compact msgpack has no
/// interior skipped fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HmmCfgExtras {
    /// Fixed-lag smoothing lag `L >= 1` in seeded learned rows.
    pub smooth_lag: Option<usize>,
    /// Per-group byte cap for the smoother (default 256 KiB, ceiling 64 MiB).
    pub max_state_bytes: Option<u64>,
    /// Subset of `"tmc"` naming which parameters learn; `None` follows `learn`.
    pub params: Option<String>,
    /// Prior weight in rows for given `means` (default `1.0`; requires given
    /// `means` and `covs`).
    pub means_weight: Option<f64>,
    /// Conjugate prior numerator for covariances (mutually exclusive with
    /// `prior_rows`).
    pub covariance_prior: Option<Vec<f64>>,
    /// Conjugate prior weight for covariances (mutually exclusive with
    /// `prior_rows`).
    pub covariance_weight: Option<f64>,
}

impl HmmCfgExtras {
    pub fn is_empty(&self) -> bool {
        self.smooth_lag.is_none()
            && self.max_state_bytes.is_none()
            && self.params.is_none()
            && self.means_weight.is_none()
            && self.covariance_prior.is_none()
            && self.covariance_weight.is_none()
    }
}

/// Optional state extensions held in one trailing field on [`Hmm`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct HmmExtras {
    /// Per-state spherical accumulators when `covariance == Spherical`.
    pub(super) spherical_states: Vec<EwCovSpherical>,
    /// Fixed-lag smoother when `smooth_lag` is set.
    pub(super) smoother: Option<HmmSmoother>,
    /// Decayed data weight `G_l` per state when conjugate `covariance_prior` /
    /// `covariance_weight` is active; empty otherwise.
    pub(super) data_w: Vec<f64>,
}

impl HmmExtras {
    pub(super) fn is_empty(&self) -> bool {
        self.spherical_states.is_empty() && self.smoother.is_none() && self.data_w.is_empty()
    }
}

impl HmmCfg {
    /// Resolved learning flags `(t, m, c)` from `learn`, `params`, and `tvtp`.
    pub fn effective_params(&self) -> HmmParams {
        if let Some(p) = self.extras.as_deref().and_then(|e| e.params.as_deref())
            && let Ok(mut parsed) = HmmParams::parse(p)
        {
            if self.tvtp.is_some() {
                parsed.t = false;
            }
            return parsed;
        }
        if self.learn {
            HmmParams {
                t: self.tvtp.is_none(),
                m: true,
                c: true,
            }
        } else {
            HmmParams::NONE
        }
    }

    /// Configured fixed-lag smoothing lag, if any.
    pub fn smooth_lag(&self) -> Option<usize> {
        self.extras.as_deref().and_then(|e| e.smooth_lag)
    }

    /// Configured `max_state_bytes` for the smoother, if any.
    pub fn max_state_bytes(&self) -> Option<u64> {
        self.extras.as_deref().and_then(|e| e.max_state_bytes)
    }

    /// Configured `params` string, if any.
    pub fn params(&self) -> Option<&str> {
        self.extras.as_deref().and_then(|e| e.params.as_deref())
    }

    /// Prior weight on given `means`, defaulting to `1.0`.
    pub fn means_weight(&self) -> f64 {
        self.extras
            .as_deref()
            .and_then(|e| e.means_weight)
            .unwrap_or(1.0)
    }

    /// Whether conjugate covariance priors (`covariance_prior` or `covariance_weight`)
    /// are active.
    pub fn has_covariance_prior(&self) -> bool {
        self.extras
            .as_deref()
            .is_some_and(|e| e.covariance_prior.is_some() || e.covariance_weight.is_some())
    }

    /// Conjugate `covariance_weight` denominator offset:
    /// `max(cw - 1, 0)` for `spherical`/`diag`, `max(cw - d, 0)` for
    /// `full`/`tied`.
    pub(super) fn covariance_weight_offset(&self) -> f64 {
        let d = self.n_features as f64;
        let default_cw = match self.covariance {
            HmmCovariance::Spherical | HmmCovariance::Diagonal => 1.0,
            HmmCovariance::Full | HmmCovariance::Shared => d,
        };
        let cw = self
            .extras
            .as_deref()
            .and_then(|e| e.covariance_weight)
            .unwrap_or(default_cw);
        match self.covariance {
            HmmCovariance::Spherical | HmmCovariance::Diagonal => (cw - 1.0).max(0.0),
            HmmCovariance::Full | HmmCovariance::Shared => (cw - d).max(0.0),
        }
    }

    /// Scalar conjugate covariance prior entry for state `s` under `spherical`.
    pub(super) fn covariance_prior_spherical(&self, s: usize) -> f64 {
        match self
            .extras
            .as_deref()
            .and_then(|e| e.covariance_prior.as_deref())
        {
            None => self.precision_prior,
            Some(v) if v.len() == 1 => v[0],
            Some(v) => v[s],
        }
    }

    /// Diagonal conjugate covariance prior entry for state `s`, feature `i`.
    pub(super) fn covariance_prior_diag(&self, s: usize, i: usize) -> f64 {
        let d = self.n_features;
        match self
            .extras
            .as_deref()
            .and_then(|e| e.covariance_prior.as_deref())
        {
            None => self.precision_prior,
            Some(v) if v.len() == 1 => v[0],
            Some(v) if v.len() == d => v[i],
            Some(v) => v[s * d + i],
        }
    }

    /// Matrix entry `(i, j)` of the conjugate covariance prior for state `s`
    /// under `full` or `tied`.
    pub(super) fn covariance_prior_matrix(&self, s: usize, i: usize, j: usize) -> f64 {
        let d = self.n_features;
        match self
            .extras
            .as_deref()
            .and_then(|e| e.covariance_prior.as_deref())
        {
            None => {
                if i == j {
                    self.precision_prior
                } else {
                    0.0
                }
            }
            Some(v) if v.len() == 1 => {
                if i == j {
                    v[0]
                } else {
                    0.0
                }
            }
            Some(v) if v.len() == d * d => v[i * d + j],
            Some(v) => v[s * d * d + i * d + j],
        }
    }

    /// Validate all fields in `self.extras` and their interactions with the
    /// base configuration.
    pub(super) fn validate_extras(&self, no: &mut crate::Refusals) {
        let Some(ex) = self.extras.as_deref() else {
            return;
        };
        let (k, d) = (self.k, self.n_features);
        // Partial learning `params`.
        let mut parsed_params = None;
        if let Some(p) = &ex.params {
            match HmmParams::parse(p) {
                Ok(parsed) => {
                    if !self.learn && parsed.any() {
                        no.refuse(format!(
                            "hmm: learn=False conflicts with params={p:?}. Leave learn out (or \
                             set learn=True) when naming parameters to learn, or set params=\"\" \
                             to learn nothing."
                        ));
                    }
                    if self.learn && !parsed.any() {
                        no.refuse(
                            "hmm: learn=True conflicts with params=\"\". Leave learn out (or set \
                             learn=False) when params is empty, or name at least one letter of \
                             \"tmc\".",
                        );
                    }
                    if parsed.t && self.tvtp.is_some() {
                        no.refuse(
                            "hmm: params includes 't' (transitions), which does not apply under \
                             tvtp_coef: transitions come from exog_tvtp and no counts are \
                             learned. Remove 't' from params, or leave out tvtp_coef.",
                        );
                    }
                    parsed_params = Some(parsed);
                }
                Err(e) => no.refuse(e),
            }
        }
        let eff = parsed_params.unwrap_or(if self.learn {
            HmmParams {
                t: self.tvtp.is_none(),
                m: true,
                c: true,
            }
        } else {
            HmmParams::NONE
        });

        // Fixed-lag smoother.
        if let Some(lag) = ex.smooth_lag
            && lag == 0
        {
            no.refuse(
                "hmm: smooth_lag must be at least 1 row, but it is 0. Set smooth_lag to 1 or \
                 more, or leave it out for no smoother.",
            );
        }
        if let Some(cap) = ex.max_state_bytes {
            if ex.smooth_lag.is_none() && self.warm_clock.is_none() {
                no.refuse(
                    "hmm: max_state_bytes applies only with smooth_lag or warm_span, but neither \
                     is set. Set smooth_lag or warm_span, or leave max_state_bytes out.",
                );
            } else if cap == 0 || cap > STATE_BYTES_CEILING {
                no.refuse(format!(
                    "hmm: max_state_bytes must be above 0 and at most {STATE_BYTES_CEILING} (64 MiB), \
                     but it is {cap}. Give the bytes a group may hold, or leave it out for \
                     {DEFAULT_STATE_BYTES} (256 KB)."
                ));
            }
        }
        if let Some(lag) = ex.smooth_lag.filter(|&l| l > 0) {
            let cap = ex.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES);
            if cap > 0 && cap <= STATE_BYTES_CEILING {
                let bytes = HmmSmoother::bytes(k, lag);
                if bytes > cap {
                    no.refuse(format!(
                        "hmm: smooth_lag = {lag} with n_components = {k} holds {bytes} bytes a \
                         group, past max_state_bytes of {cap}. Shorten smooth_lag, or raise \
                         max_state_bytes (at most 64 MiB)."
                    ));
                }
            }
        }

        // Prior weights for given states.
        if let Some(mw) = ex.means_weight {
            if !(mw.is_finite() && mw >= 0.0) {
                no.refuse(format!(
                    "hmm: means_weight must be finite and at least 0, but it is {mw}. Set \
                     means_weight to a non-negative weight in rows, or leave it out for 1.0."
                ));
            }
            if self.means.is_none() || self.covs.is_none() {
                no.refuse(
                    "hmm: means_weight requires given means and covs, but they are not set. Give \
                     means and covs, or leave means_weight out.",
                );
            } else if !eff.m {
                no.refuse(
                    "hmm: means_weight does not apply when means are not learned. Add 'm' to \
                     params (with learn on), or leave means_weight out.",
                );
            }
        }
        let has_cp = ex.covariance_prior.is_some() || ex.covariance_weight.is_some();
        if has_cp {
            if self.prior_rows.is_some() {
                no.refuse(
                    "hmm: covariance_prior and covariance_weight cannot be used beside prior_rows: they \
                     are two covariance-prior mechanisms. Leave prior_rows out, or leave out \
                     covariance_prior and covariance_weight.",
                );
            }
            if !eff.c {
                no.refuse(
                    "hmm: covariance_prior and covariance_weight do not apply when covariances are not \
                     learned. Add 'c' to params (with learn on), or leave them out.",
                );
            }
        }
        if let Some(cw) = ex.covariance_weight
            && !(cw.is_finite() && cw >= 0.0)
        {
            no.refuse(format!(
                "hmm: covariance_weight must be finite and at least 0, but it is {cw}. Set \
                 covariance_weight to a finite non-negative value."
            ));
        }
        if let Some(cp) = &ex.covariance_prior {
            if let Some(&bad) = cp.iter().find(|v| !v.is_finite()) {
                no.refuse(format!(
                    "hmm: covariance_prior must be finite, but it holds {bad}. Replace {bad} with a \
                     finite value."
                ));
            } else {
                match self.covariance {
                    HmmCovariance::Spherical => {
                        if cp.len() != 1 && cp.len() != k {
                            no.refuse(format!(
                                "hmm: under covariance_type=\"spherical\", covariance_prior must hold \
                                 1 value or {k} values (one per state), but it has {}. Give 1 or \
                                 {k} non-negative values.",
                                cp.len()
                            ));
                        } else if let Some(&v) = cp.iter().find(|&&v| v < 0.0) {
                            no.refuse(format!(
                                "hmm: covariance_prior must be at least 0, but it holds {v}. Give \
                                 non-negative prior entries."
                            ));
                        }
                    }
                    HmmCovariance::Diagonal => {
                        if cp.len() != 1 && cp.len() != d && cp.len() != k * d {
                            no.refuse(format!(
                                "hmm: under covariance_type=\"diag\", covariance_prior must hold 1, \
                                 {d}, or {} values, but it has {}. Give 1, {d}, or {} non-negative \
                                 values.",
                                k * d,
                                cp.len(),
                                k * d
                            ));
                        } else if let Some(&v) = cp.iter().find(|&&v| v < 0.0) {
                            no.refuse(format!(
                                "hmm: covariance_prior must be at least 0, but it holds {v}. Give \
                                 non-negative prior entries."
                            ));
                        }
                    }
                    HmmCovariance::Full => {
                        if cp.len() != 1 && cp.len() != d * d && cp.len() != k * d * d {
                            no.refuse(format!(
                                "hmm: under covariance_type=\"full\", covariance_prior must hold 1, \
                                 {}, or {} values, but it has {}. Give 1 scalar or a {d} by {d} \
                                 matrix per state.",
                                d * d,
                                k * d * d,
                                cp.len()
                            ));
                        } else if cp.len() == 1 && cp[0] < 0.0 {
                            no.refuse(format!(
                                "hmm: covariance_prior must be at least 0, but it holds {}. Give a \
                                 non-negative scalar.",
                                cp[0]
                            ));
                        } else if cp.len() >= d * d {
                            let blocks = cp.len() / (d * d);
                            for b in 0..blocks {
                                check_psd_prior(
                                    &cp[b * d * d..(b + 1) * d * d],
                                    d,
                                    (blocks > 1).then_some(b),
                                    no,
                                );
                            }
                        }
                    }
                    HmmCovariance::Shared => {
                        if cp.len() != 1 && cp.len() != d * d {
                            no.refuse(format!(
                                "hmm: under covariance_type=\"tied\", covariance_prior must hold 1 or \
                                 {} values, but it has {}. Give 1 scalar or one {d} by {d} \
                                 matrix.",
                                d * d,
                                cp.len()
                            ));
                        } else if cp.len() == 1 && cp[0] < 0.0 {
                            no.refuse(format!(
                                "hmm: covariance_prior must be at least 0, but it holds {}. Give a \
                                 non-negative scalar.",
                                cp[0]
                            ));
                        } else if cp.len() == d * d {
                            check_psd_prior(cp, d, None, no);
                        }
                    }
                }
            }
        }
    }
}

/// Check that a matrix block in `covariance_prior` is symmetric and has
/// non-negative diagonal entries (and is PSD if non-zero).
fn check_psd_prior(block: &[f64], d: usize, state: Option<usize>, no: &mut crate::Refusals) {
    let who = match state {
        Some(s) => format!("for state {s} (counting from 0) "),
        None => String::new(),
    };
    let asym = (0..d)
        .flat_map(|i| ((i + 1)..d).map(move |j| (i, j)))
        .map(|(i, j)| (i, j, block[i * d + j], block[j * d + i]))
        .find(|&(_, _, a, b)| (a - b).abs() > 1e-12 * (1.0 + a.abs().max(b.abs())));
    if let Some((i, j, a, b)) = asym {
        no.refuse(format!(
            "hmm: the matrix in covariance_prior {who}must be symmetric, but entry ({i}, {j}) is {a} \
             and ({j}, {i}) is {b}. Set them equal."
        ));
        return;
    }
    if block.iter().all(|&v| v == 0.0) {
        return;
    }
    // Diagonal entries must be non-negative; if strictly positive, require SPD
    // or at least non-negative eigenvalues (tiny jitter checks PSD).
    if (0..d).any(|i| block[i * d + i] < 0.0) {
        no.refuse(format!(
            "hmm: the matrix in covariance_prior {who}must be positive semidefinite, but a diagonal \
             entry is negative. Give a positive semidefinite prior matrix."
        ));
        return;
    }
    let scale = (0..d).map(|i| block[i * d + i]).fold(0.0f64, f64::max);
    if scale > 0.0 {
        let mut nudged = block.to_vec();
        let eps = 1e-12 * scale;
        for i in 0..d {
            nudged[i * d + i] += eps;
        }
        if !matches!(crate::solve::SpdFactor::of(&nudged, d), Some(f) if f.attempts() == 0) {
            no.refuse(format!(
                "hmm: the matrix in covariance_prior {who}must be positive semidefinite, but it is \
                 not. Give a positive semidefinite prior matrix."
            ));
        }
    }
}

/// Custom suffix-trimmed `Serialize` for [`HmmCfg`]:
///
/// - When `extras` is `None` (or empty), behaves identically to the prior
///   derived `Serialize` (`warm_clock` skipped when `None`), so every frozen
///   fixture from `v61..v71` and current `v72` writes the exact same bytes.
/// - When `extras` is `Some` (and non-empty), emits `warm_clock` explicitly
///   (even if `None`) before `extras`, so compact positional msgpack never
///   slides `extras` into `warm_clock`'s slot.
impl Serialize for HmmCfg {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let has_extras = self.extras.as_deref().is_some_and(|e| !e.is_empty());
        let has_warm = has_extras || self.warm_clock.is_some();
        let len = 16 + usize::from(has_warm) + usize::from(has_extras);
        let mut st = serializer.serialize_struct("HmmCfg", len)?;
        st.serialize_field("n_features", &self.n_features)?;
        st.serialize_field("k", &self.k)?;
        st.serialize_field("decay", &self.decay)?;
        st.serialize_field("covariance", &self.covariance)?;
        st.serialize_field("precision_prior", &self.precision_prior)?;
        st.serialize_field("min_weight", &self.min_weight)?;
        st.serialize_field("learn", &self.learn)?;
        st.serialize_field("transition_prior", &self.transition_prior)?;
        st.serialize_field("transition", &self.transition)?;
        st.serialize_field("means", &self.means)?;
        st.serialize_field("covs", &self.covs)?;
        st.serialize_field("warm_rows", &self.warm_rows)?;
        st.serialize_field("seed_rule", &self.seed_rule)?;
        st.serialize_field("seed", &self.seed)?;
        st.serialize_field("tvtp", &self.tvtp)?;
        st.serialize_field("prior_rows", &self.prior_rows)?;
        if has_warm {
            st.serialize_field("warm_clock", &self.warm_clock)?;
        }
        if has_extras {
            st.serialize_field("extras", &self.extras)?;
        }
        st.end()
    }
}

/// Custom `Serialize` for [`Hmm`]:
///
/// - When `extras` is absent or empty, keeps the exact pre-existing per-field
///   `skip_serializing_if` behavior so every pre-existing schema fixture and
///   existing atomicity / compact-encoding test is unchanged.
/// - When `extras` is present and non-empty, emits all trailing optional
///   fields up through `extras` with no interior holes so compact msgpack
///   round-trips unambiguously for every combination of new options.
impl Serialize for Hmm {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let has_extras = self.extras.as_deref().is_some_and(|e| !e.is_empty());
        let w_stab = has_extras || self.stabilized.is_some();
        let w_place = has_extras || self.place.is_some();
        let w_first = has_extras || self.warm_first.is_some();
        let w_over = has_extras || self.over.is_some();
        let w_warned = has_extras || self.warned;
        let len = 10
            + usize::from(w_stab)
            + usize::from(w_place)
            + usize::from(w_first)
            + usize::from(w_over)
            + usize::from(w_warned)
            + usize::from(has_extras);
        let mut st = serializer.serialize_struct("Hmm", len)?;
        st.serialize_field("cfg", &self.cfg)?;
        st.serialize_field("states", &self.states)?;
        st.serialize_field("p", &self.p)?;
        st.serialize_field("a", &self.a)?;
        st.serialize_field("n_eff", &self.n_eff)?;
        st.serialize_field("buffer", &self.buffer)?;
        st.serialize_field("seeded", &self.seeded)?;
        st.serialize_field("solve_failures", &self.solve_failures)?;
        st.serialize_field("diag_states", &self.diag_states)?;
        st.serialize_field("factor_states", &self.factor_states)?;
        if w_stab {
            st.serialize_field("stabilized", &self.stabilized)?;
        }
        if w_place {
            st.serialize_field("place", &self.place)?;
        }
        if w_first {
            st.serialize_field("warm_first", &self.warm_first)?;
        }
        if w_over {
            st.serialize_field("over", &self.over)?;
        }
        if w_warned {
            st.serialize_field("warned", &self.warned)?;
        }
        if has_extras {
            st.serialize_field("extras", &self.extras)?;
        }
        st.end()
    }
}
