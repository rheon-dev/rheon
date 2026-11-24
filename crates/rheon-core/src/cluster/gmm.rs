//! A Gaussian mixture fitted by exponentially weighted online EM (Cappé and
//! Moulines 2009), scikit-learn's `GaussianMixture` one row at a time.
//!
//! **The step.** Each component is an exponentially weighted mean and
//! covariance (the mean form [`EwCov`], or [`EwCovDiag`] for `diag`), fed
//! the row at weight `w r_j`, `r_j` its responsibility read from the
//! parameters before the row, so a row's responsibilities and log-density
//! are out of sample:
//!
//! ```text
//! log ρ_j = log π_j − ½ (k log 2π + log det Σ̂_j + (x − μ_j)ᵀ Σ̂_j⁻¹ (x − μ_j))
//! r_j     = exp(log ρ_j − max_l log ρ_l) / Σ_l exp(log ρ_l − max_l log ρ_l)
//! log f   = max_l log ρ_l + log Σ_l exp(log ρ_l − max_l log ρ_l)
//! each j: W_j' = λ W_j + w r_j,  μ_j' = μ_j + (w r_j / W_j') (x − μ_j),
//!         C_j' = (λ W_j / W_j') (C_j + (w r_j / W_j') δ δᵀ)
//! read:   π_j = W_j / Σ_l W_l,   Σ̂_j = C_j + reg_covar · I
//! ```
//!
//! `C_j` is the weighted covariance about the component's own mean, the
//! M-step's `Σ_i r_ij (x_i − μ_j)(x_i − μ_j)ᵀ / N_j`, and `reg_covar` is
//! added to its diagonal as scikit-learn adds it. Each covariance is
//! factorized afresh from the stored moments on every row, so the factors
//! are derived state: a state loaded mid-stream refactorizes to the same
//! bits. A component of no weight takes no row (`π_j = 0`).
//!
//! **The start.** Online EM needs a start, and the start's weight decides
//! the fit, so the model buffers its learned rows (bounded by
//! `max_state_bytes`) and fits batch EM to the buffer at each doubling of
//! it: a seeded k-means++ start, then EM until the mean log-likelihood
//! moves by less than `tol`, the best of `n_init` starts kept. It starts
//! when three conditions hold, and never at a count of rows:
//!
//! 1. every component is identified: its responsibility mass holds Kish's
//!    effective count above the parameters it carries, and its covariance
//!    before `reg_covar` stays positive definite less `reg_covar · I`;
//! 2. `K` components are told apart: the fit with `K` beats the fit with
//!    `K − 1` by BIC by at least `delta_bic` (10 by default, Kass and
//!    Raftery's "very strong" evidence);
//! 3. the fit has stopped moving: the rows added since the last try score,
//!    under the new fit, at most `3 p / n` above their score under the last
//!    try's (`p` the free parameters, `n` the buffer's Kish size). At a
//!    fixed point the last fit, from half the rows, loses `p / n` out of
//!    sample on them and the new fit's in-sample optimism on them adds
//!    `p / (2 n)` (AIC's reading of a likelihood's optimism), so `3 p / n`
//!    is twice what a converged fit gains; a fit still moving gains a
//!    constant a row.
//!
//! The fit then enters at its rows' weight, its components ordered by
//! weight, largest first, ties by the first mean coordinate, and that order
//! is fixed for the rest of the run. Until then every row is withheld,
//! naming the condition that failed; a buffer that reaches its bound is
//! withheld with its own reason and never forced to start.

// Each guard compares as `!(x > 0.0)` so a NaN fails it with the rest.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use serde::{Deserialize, Serialize};

use super::summary::SplitMix64;
use crate::clock::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{EwCov, EwCovDiag, SpdFactor};

/// The shape of each component's covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GmmCovariance {
    /// A whole `k × k` matrix a component.
    Full,
    /// Its diagonal alone: `k` variances a component.
    Diag,
    /// One variance a component, the mean of its `k` variances.
    Spherical,
    /// One `k × k` matrix every component shares.
    Tied,
}

impl GmmCovariance {
    /// The free parameters of `kk` components' covariances.
    fn params(self, k: usize, kk: usize) -> usize {
        match self {
            Self::Full => kk * k * (k + 1) / 2,
            Self::Diag => kk * k,
            Self::Spherical => kk,
            Self::Tied => k * (k + 1) / 2,
        }
    }

    /// The numbers a component's covariance takes in a [`Fit`]'s `covs`:
    /// `k × k` row-major, `k` variances or one; `tied`'s one `k × k` is
    /// shared, at the front.
    fn per(self, k: usize) -> usize {
        match self {
            Self::Full => k * k,
            Self::Diag => k,
            Self::Spherical => 1,
            Self::Tied => 0,
        }
    }

    /// Where component `j`'s covariance starts in a [`Fit`]'s `covs`.
    fn at(self, k: usize, j: usize) -> usize {
        j * self.per(k)
    }
}

/// The settings of a mixture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GmmCfg {
    pub n_features: usize,
    pub n_components: usize,
    pub covariance: GmmCovariance,
    /// Added to each covariance's diagonal, as scikit-learn's `reg_covar`.
    pub reg_covar: f64,
    /// The BIC margin by which `K` components must beat `K − 1` to start.
    pub delta_bic: f64,
    /// Batch EM starts the start tries, the best kept.
    pub n_init: usize,
    /// The seed of the start's k-means++ draws.
    pub seed: u64,
    /// The start's batch EM: the most iterations, and the change in the
    /// mean log-likelihood it stops at.
    pub max_iter: usize,
    pub tol: f64,
    /// The most bytes the start's buffer may hold.
    pub max_state_bytes: u64,
    /// The decay over the clock, as every model takes it.
    pub decay: Decay,
    /// The weight before the row below which nothing is reported.
    pub min_weight: f64,
}

impl GmmCfg {
    /// scikit-learn's defaults for `K` components of `k` features.
    pub fn new(n_features: usize, n_components: usize, covariance: GmmCovariance) -> Self {
        Self {
            n_features,
            n_components,
            covariance,
            reg_covar: 1e-6,
            delta_bic: 10.0,
            n_init: 1,
            seed: 0,
            max_iter: 100,
            tol: 1e-3,
            max_state_bytes: 256 * 1024,
            decay: Decay::Halflife(f64::INFINITY),
            min_weight: 0.0,
        }
    }

    fn check(&self) -> Result<(), String> {
        if self.n_features == 0 {
            return Err(
                "gmm: n_features must be at least 1, but it is 0. Give one or more \
                        feature columns."
                    .into(),
            );
        }
        if self.n_components == 0 {
            return Err(
                "gmm: n_components must be at least 1, but it is 0. Give the number \
                        of components."
                    .into(),
            );
        }
        self.decay.check().map_err(|e| format!("gmm: {e}"))?;
        if !(self.reg_covar.is_finite() && self.reg_covar > 0.0) {
            return Err(format!(
                "gmm: reg_covar must be a finite number above 0, but it is {}. Give a small \
                 positive floor, such as 1e-6.",
                self.reg_covar
            ));
        }
        if !(self.delta_bic.is_finite() && self.delta_bic >= 0.0) {
            return Err(format!(
                "gmm: delta_bic must be a finite number of at least 0, but it is {}. Give the \
                 strength of evidence the start needs, such as 10.",
                self.delta_bic
            ));
        }
        if self.n_init == 0 || self.max_iter == 0 {
            return Err("gmm: n_init and max_iter must each be at least 1. Give 1 or more.".into());
        }
        if !(self.min_weight.is_finite() && self.min_weight >= 0.0) {
            return Err(format!(
                "gmm: min_weight must be a finite number of at least 0, but it is {}. Give 0, or \
                 the weight a fit needs before it is read.",
                self.min_weight
            ));
        }
        if self.max_state_bytes == 0 {
            return Err(
                "gmm: max_state_bytes must be at least 1. Give the most bytes the start's \
                        buffer may hold."
                    .into(),
            );
        }
        if !(self.tol.is_finite() && self.tol > 0.0) {
            return Err(format!(
                "gmm: tol must be a finite number above 0, but it is {}. Give a small positive \
                 tolerance, such as 1e-3.",
                self.tol
            ));
        }
        Ok(())
    }

    /// The free parameters of the mixture: weights, means and covariances.
    fn params(&self) -> usize {
        let (k, kk) = (self.n_features, self.n_components);
        (kk - 1) + kk * k + self.covariance.params(k, kk)
    }

    /// The bytes one buffered row holds: its features and its weight.
    fn row_bytes(&self) -> u64 {
        8 * (self.n_features as u64 + 1)
    }
}

/// Why a row is withheld before the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GmmWait {
    /// No start has been tried yet.
    WarmUp,
    /// A component of the last try held too little weight to identify it,
    /// or its covariance needed the floor to stay positive definite.
    NotIdentified,
    /// `K` components did not beat `K − 1` by `delta_bic`.
    NotSeparated,
    /// The fit moved more than its optimism between the last two tries.
    Moving,
    /// The buffer reached `max_state_bytes` without a start.
    OverCap,
}

/// The components, each a mean-form accumulator. `spherical` keeps each
/// component's `k` variances and reads their mean, which the step moves
/// exactly as the M-step's mean of the diagonal; `tied` keeps each
/// component's weight and mean, and every row's moments in `total`, from
/// which the shared covariance is read as the total less the components'
/// spread about the total mean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Comps {
    Full(Vec<EwCov>),
    Diag(Vec<EwCovDiag>),
    Spherical(Vec<EwCovDiag>),
    Tied {
        comps: Vec<EwCovDiag>,
        total: Box<EwCov>,
    },
}

impl Comps {
    fn mass(&self) -> Vec<f64> {
        match self {
            Comps::Full(c) => c.iter().map(EwCov::n_eff).collect(),
            Comps::Diag(c) | Comps::Spherical(c) | Comps::Tied { comps: c, .. } => {
                c.iter().map(EwCovDiag::n_eff).collect()
            }
        }
    }

    /// Learns a row at weight `w r_j` in each component (`tied`'s total at
    /// `w`), or ages them all where `resp` is `None`.
    fn update(&mut self, x: &[f64], lam: f64, w: f64, resp: Option<&[f64]>) {
        let at = |j: usize| resp.map_or(0.0, |r| w * r[j]);
        match self {
            Comps::Full(c) => {
                for (j, e) in c.iter_mut().enumerate() {
                    e.update(x, lam, at(j));
                }
            }
            Comps::Diag(c) | Comps::Spherical(c) => {
                for (j, e) in c.iter_mut().enumerate() {
                    e.update(x, lam, at(j));
                }
            }
            Comps::Tied { comps, total } => {
                for (j, e) in comps.iter_mut().enumerate() {
                    e.update(x, lam, at(j));
                }
                total.update(x, lam, if resp.is_some() { w } else { 0.0 });
            }
        }
    }
}

/// The read of one covariance with `reg_covar`: its factor, its variances
/// or its one variance, and its log-determinant.
enum Scorer {
    Full(SpdFactor),
    Diag(Vec<f64>, f64),
    Spherical(f64, f64),
    /// A component of no weight, never read.
    None,
}

impl Scorer {
    /// Component `j`'s scorer from `covs` (before `reg_covar`), laid out as
    /// a [`Fit`]'s are; `None` where a full or tied covariance does not
    /// factorize.
    fn of(cfg: &GmmCfg, covs: &[f64], j: usize) -> Option<Self> {
        let (k, reg) = (cfg.n_features, cfg.reg_covar);
        let at = cfg.covariance.at(k, j);
        Some(match cfg.covariance {
            GmmCovariance::Full | GmmCovariance::Tied => {
                let mut m = covs[at..at + k * k].to_vec();
                for i in 0..k {
                    m[i * k + i] += reg;
                }
                Scorer::Full(SpdFactor::of(&m, k)?)
            }
            GmmCovariance::Diag => {
                let v: Vec<f64> = covs[at..at + k].iter().map(|c| c + reg).collect();
                let ld = v.iter().map(|x| x.ln()).sum();
                Scorer::Diag(v, ld)
            }
            GmmCovariance::Spherical => {
                let v = covs[at] + reg;
                Scorer::Spherical(v, k as f64 * v.ln())
            }
        })
    }

    /// Every component's scorer, `tied`'s factor computed once.
    fn all(cfg: &GmmCfg, covs: &[f64], weights: &[f64]) -> Option<Vec<Self>> {
        let mut out = Vec::with_capacity(weights.len());
        for (j, &w) in weights.iter().enumerate() {
            if !(w > 0.0) {
                out.push(Scorer::None);
            } else if cfg.covariance == GmmCovariance::Tied
                && let Some(first) = out.iter().find(|s| matches!(s, Scorer::Full(_)))
            {
                let Scorer::Full(f) = first else {
                    unreachable!()
                };
                out.push(Scorer::Full(f.clone()));
            } else {
                out.push(Scorer::of(cfg, covs, j)?);
            }
        }
        Some(out)
    }

    fn log_det(&self) -> f64 {
        match self {
            Scorer::Full(f) => f.log_det(),
            Scorer::Diag(_, ld) | Scorer::Spherical(_, ld) => *ld,
            Scorer::None => f64::NAN,
        }
    }

    /// The squared Mahalanobis distances of `n` deviations, `n × k`.
    fn quads(&self, d: &[f64], k: usize, n: usize) -> Vec<f64> {
        match self {
            Scorer::Full(f) => f.quad_forms(d, k, n),
            Scorer::Diag(v, _) => (0..n)
                .map(|r| (0..k).map(|i| d[r * k + i].powi(2) / v[i]).sum())
                .collect(),
            Scorer::Spherical(v, _) => (0..n)
                .map(|r| (0..k).map(|i| d[r * k + i].powi(2) / v).sum())
                .collect(),
            Scorer::None => vec![f64::NAN; n],
        }
    }
}

/// A batch fit: weights, means, and covariances before `reg_covar` (full:
/// `k × k` row-major each; diag: `k` each; spherical: one each; tied: one
/// `k × k` for all), with each component's mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub weights: Vec<f64>,
    pub means: Vec<f64>,
    pub covs: Vec<f64>,
    pub mass: Vec<f64>,
    /// Each component's Kish effective count of its rows' weights `w r_j`.
    pub eff: Vec<f64>,
}

/// A row's read: each component's responsibility, the most responsible
/// component, the row's log-density, and its Mahalanobis distance to that
/// component.
#[derive(Debug, Clone, PartialEq)]
pub struct GmmRead {
    pub resp: Vec<f64>,
    pub cluster: usize,
    pub log_density: f64,
    pub dist: f64,
}

/// A Gaussian mixture by online EM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gmm {
    cfg: GmmCfg,
    /// The components once started; `None` before the start.
    comps: Option<Comps>,
    /// The start's buffer: each learned row's features and its weight,
    /// decayed to the latest row.
    buf_x: Vec<f64>,
    buf_w: Vec<f64>,
    /// The buffer's length at which the next start is tried.
    next_try: usize,
    /// The last try's fit and the buffer's length then, for condition 3.
    last_fit: Option<(Fit, usize)>,
    wait: GmmWait,
    /// The weight of the rows learned, by the shared recursion
    /// `n ← λ n + w` (hard rule 8).
    n: f64,
}

const LN_2PI: f64 = 1.837_877_066_409_345_3;

impl Gmm {
    pub fn new(cfg: GmmCfg) -> Result<Self, String> {
        cfg.check()?;
        Ok(Self {
            cfg,
            comps: None,
            buf_x: Vec::new(),
            buf_w: Vec::new(),
            next_try: 1,
            last_fit: None,
            wait: GmmWait::WarmUp,
            n: 0.0,
        })
    }

    pub fn cfg(&self) -> &GmmCfg {
        &self.cfg
    }

    pub fn started(&self) -> bool {
        self.comps.is_some()
    }

    /// Why rows are withheld, before the start; `None` once started.
    pub fn wait(&self) -> Option<GmmWait> {
        (!self.started()).then_some(self.wait)
    }

    /// The weight of the rows learned, `n ← λ n + w`, read before a row:
    /// `n_eff`, the same in every model (hard rule 8). Once started it is
    /// the components' weight `Σ_j W_j`.
    pub fn n_eff(&self) -> f64 {
        self.n
    }

    /// The bytes the start's buffer holds.
    pub fn buffer_bytes(&self) -> u64 {
        self.buf_w.len() as u64 * self.cfg.row_bytes()
    }

    /// Each component's weight `π_j`, mean and covariance with `reg_covar`
    /// (full `k × k` row-major, or the diagonal), in the fixed order.
    pub fn parameters(&self) -> Option<Fit> {
        let (k, reg) = (self.cfg.n_features, self.cfg.reg_covar);
        let mut fit = self.moments()?;
        let (per, n) = match self.cfg.covariance {
            GmmCovariance::Tied => (k * k, 1),
            c => (c.per(k), self.cfg.n_components),
        };
        for j in 0..n {
            let c = &mut fit.covs[j * per..(j + 1) * per];
            match self.cfg.covariance {
                GmmCovariance::Full | GmmCovariance::Tied => {
                    (0..k).for_each(|i| c[i * k + i] += reg);
                }
                _ => c.iter_mut().for_each(|v| *v += reg),
            }
        }
        Some(fit)
    }

    /// [`Self::parameters`] before `reg_covar`, laid out as a batch fit's.
    fn moments(&self) -> Option<Fit> {
        let (k, kk) = (self.cfg.n_features, self.cfg.n_components);
        let comps = self.comps.as_ref()?;
        let mass = comps.mass();
        let total: f64 = mass.iter().sum();
        let weights: Vec<f64> = mass
            .iter()
            .map(|m| if total > 0.0 { m / total } else { 0.0 })
            .collect();
        let mut means = Vec::with_capacity(kk * k);
        let mut covs = Vec::new();
        match comps {
            Comps::Full(c) => {
                for e in c {
                    means.extend_from_slice(e.means());
                    for i in 0..k {
                        for j in 0..k {
                            covs.push(e.cov(i, j));
                        }
                    }
                }
            }
            Comps::Diag(c) => {
                for e in c {
                    means.extend_from_slice(e.means());
                    covs.extend((0..k).map(|i| e.comoment(i)));
                }
            }
            Comps::Spherical(c) => {
                for e in c {
                    means.extend_from_slice(e.means());
                    covs.push((0..k).map(|i| e.comoment(i)).sum::<f64>() / k as f64);
                }
            }
            Comps::Tied { comps: c, total: t } => {
                // The total less the components' spread about the total mean:
                // `Σ_j π_j (μ_j − m)(μ_j − m)ᵀ`, never `m mᵀ`, which cancels.
                for e in c {
                    means.extend_from_slice(e.means());
                }
                let m = t.means();
                covs = (0..k * k).map(|i| t.cov(i / k, i % k)).collect();
                for (j, &p) in weights.iter().enumerate() {
                    let mu = &means[j * k..(j + 1) * k];
                    for a in 0..k {
                        for b in a..k {
                            let v = covs[a * k + b] - p * (mu[a] - m[a]) * (mu[b] - m[b]);
                            covs[a * k + b] = v;
                            covs[b * k + a] = v;
                        }
                    }
                }
            }
        }
        let eff = mass.clone();
        Some(Fit {
            weights,
            means,
            covs,
            mass,
            eff,
        })
    }

    /// The row's read from the current components, or `None` before the
    /// start or where no component can score it.
    pub fn read(&self, x: &[f64]) -> Option<GmmRead> {
        let comps = self.comps.as_ref()?;
        let (k, kk) = (self.cfg.n_features, self.cfg.n_components);
        let reg = self.cfg.reg_covar;
        let mass = comps.mass();
        let total: f64 = mass.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut logp = vec![f64::NEG_INFINITY; kk];
        let mut quad = vec![f64::NAN; kk];
        let mut delta = vec![0.0; k];
        let mut m = vec![0.0; k * k];
        // `spherical` and `tied` read through the moments, `tied`'s factor
        // once a row.
        let shared = match comps {
            Comps::Spherical(_) | Comps::Tied { .. } => {
                let fit = self.moments()?;
                Some(Scorer::all(&self.cfg, &fit.covs, &fit.weights)?)
            }
            _ => None,
        };
        for j in 0..kk {
            if mass[j] <= 0.0 {
                continue;
            }
            let (log_det, q) = match comps {
                Comps::Spherical(c) | Comps::Tied { comps: c, .. } => {
                    let e = &c[j];
                    for (i, d) in delta.iter_mut().enumerate() {
                        *d = e.deviation(i, x[i]);
                    }
                    let s = &shared.as_ref().expect("read above")[j];
                    (s.log_det(), s.quads(&delta, k, 1)[0])
                }
                Comps::Full(c) => {
                    let e = &c[j];
                    for (i, d) in delta.iter_mut().enumerate() {
                        *d = e.deviation(i, x[i]);
                    }
                    for i in 0..k {
                        for l in 0..k {
                            m[i * k + l] = e.cov(i, l) + if i == l { reg } else { 0.0 };
                        }
                    }
                    let f = SpdFactor::of(&m, k)?;
                    (f.log_det(), f.quad_forms(&delta, k, 1)[0])
                }
                Comps::Diag(c) => {
                    let e = &c[j];
                    let (mut log_det, mut q) = (0.0, 0.0);
                    for (i, &xi) in x.iter().enumerate() {
                        let v = e.comoment(i) + reg;
                        let d = e.deviation(i, xi);
                        log_det += v.ln();
                        q += d * d / v;
                    }
                    (log_det, q)
                }
            };
            quad[j] = q;
            logp[j] = (mass[j] / total).ln() - 0.5 * (k as f64 * LN_2PI + log_det + q);
        }
        let (resp, log_density) = normalize(&logp)?;
        let cluster = argmax(&resp);
        Some(GmmRead {
            resp,
            cluster,
            log_density,
            dist: quad[cluster].sqrt(),
        })
    }

    /// Learns a row of weight `w` after a decay `lam`: the online EM step
    /// once started, else the buffer and, at its doublings, a try of the
    /// start. A row of weight 0 decays the weights and learns nothing.
    pub fn learn(&mut self, x: &[f64], w: f64, lam: f64) {
        let read = self.read(x);
        self.learn_read(x, w, lam, read.as_ref());
    }

    /// [`Self::learn`] with the row's read already taken from the state
    /// before the row.
    fn learn_read(&mut self, x: &[f64], w: f64, lam: f64, read: Option<&GmmRead>) {
        // A history the decay took to 0, or to the subnormal range of the
        // row it meets, is forgotten (`gaps::decay_into`'s rule): the model
        // goes back to its start, with the row the first of a new buffer.
        let held = self.comps.is_some() || !self.buf_w.is_empty();
        if held && w > 0.0 && lam * self.n <= w * f64::MIN_POSITIVE {
            self.comps = None;
            self.buf_x.clear();
            self.buf_w.clear();
            self.next_try = 1;
            self.last_fit = None;
            self.wait = GmmWait::WarmUp;
            self.n = 0.0;
            return self.learn_read(x, w, 1.0, None);
        }
        self.n = lam * self.n + w;
        if let Some(comps) = self.comps.as_mut() {
            // Started: the online EM step, or, where no component could score
            // the row, an ageing of them all.
            comps.update(x, lam, w, read.map(|r| r.resp.as_slice()));
            return;
        }
        for v in &mut self.buf_w {
            *v *= lam;
        }
        if w <= 0.0 || self.wait == GmmWait::OverCap {
            return;
        }
        if self.buffer_bytes() + self.cfg.row_bytes() > self.cfg.max_state_bytes {
            self.wait = GmmWait::OverCap;
            return;
        }
        self.buf_x.extend_from_slice(x);
        self.buf_w.push(w);
        if self.buf_w.len() >= self.next_try {
            self.next_try = 2 * self.buf_w.len();
            self.try_start();
        }
    }

    /// Fits the buffer and starts if the three conditions hold.
    fn try_start(&mut self) {
        let cfg = &self.cfg;
        let (k, kk) = (cfg.n_features, cfg.n_components);
        let previous = self.last_fit.take();
        let warm = previous.as_ref().map(|(f, _)| f);
        let fit = best_fit(cfg, &self.buf_x, &self.buf_w, kk, warm);
        let n = self.buf_w.len();
        let Some(fit) = fit else {
            self.wait = GmmWait::NotIdentified;
            return;
        };
        self.last_fit = Some((fit.clone(), n));
        let kish = kish(&self.buf_w);
        if !identified(cfg, &fit) {
            self.wait = GmmWait::NotIdentified;
            return;
        }
        if kk > 1 {
            let fewer = best_fit(cfg, &self.buf_x, &self.buf_w, kk - 1, None);
            let bic = |f: &Fit, c: usize| {
                let p = (c - 1) + c * k + cfg.covariance.params(k, c);
                -2.0 * kish * mean_loglik(cfg, f, &self.buf_x, &self.buf_w) + p as f64 * kish.ln()
            };
            let separated = fewer.is_none_or(|f| bic(&f, kk - 1) - bic(&fit, kk) >= cfg.delta_bic);
            if !separated {
                self.wait = GmmWait::NotSeparated;
                return;
            }
        }
        let Some((prev, at)) = previous else {
            self.wait = GmmWait::Moving;
            return;
        };
        let (new_x, new_w) = (&self.buf_x[at * k..], &self.buf_w[at..]);
        let gain = mean_loglik(cfg, &fit, new_x, new_w) - mean_loglik(cfg, &prev, new_x, new_w);
        if !(gain <= 3.0 * cfg.params() as f64 / kish) {
            self.wait = GmmWait::Moving;
            return;
        }
        self.start(&fit);
    }

    /// Enters `fit` at its rows' weight, its components by weight, largest
    /// first, ties by the first mean coordinate.
    fn start(&mut self, fit: &Fit) {
        let (k, kk) = (self.cfg.n_features, self.cfg.n_components);
        let mut order: Vec<usize> = (0..kk).collect();
        order.sort_by(|&a, &b| {
            fit.mass[b]
                .total_cmp(&fit.mass[a])
                .then(fit.means[a * k].total_cmp(&fit.means[b * k]))
        });
        let comps = match self.cfg.covariance {
            GmmCovariance::Full => Comps::Full(
                order
                    .iter()
                    .map(|&j| {
                        let cov = &fit.covs[j * k * k..(j + 1) * k * k];
                        EwCov::at_moments(k, fit.mass[j], &fit.means[j * k..(j + 1) * k], cov)
                            .without_runs()
                    })
                    .collect(),
            ),
            GmmCovariance::Diag => Comps::Diag(
                order
                    .iter()
                    .map(|&j| {
                        let v = &fit.covs[j * k..(j + 1) * k];
                        EwCovDiag::at_moments(k, fit.mass[j], &fit.means[j * k..(j + 1) * k], v)
                            .without_runs()
                    })
                    .collect(),
            ),
            // Each of the `k` variances at the component's one: their mean,
            // which is what is read, moves as the M-step's mean does.
            GmmCovariance::Spherical => Comps::Spherical(
                order
                    .iter()
                    .map(|&j| {
                        let v = vec![fit.covs[j]; k];
                        EwCovDiag::at_moments(k, fit.mass[j], &fit.means[j * k..(j + 1) * k], &v)
                            .without_runs()
                    })
                    .collect(),
            ),
            GmmCovariance::Tied => {
                // The total's moments: the shared covariance plus the
                // components' spread about their weighted mean.
                let w: f64 = fit.mass.iter().sum();
                let mut m = vec![0.0; k];
                for j in 0..kk {
                    for (v, mu) in m.iter_mut().zip(&fit.means[j * k..(j + 1) * k]) {
                        *v += fit.mass[j] * mu;
                    }
                }
                m.iter_mut().for_each(|v| *v /= w);
                let mut c = fit.covs[..k * k].to_vec();
                for j in 0..kk {
                    let p = fit.mass[j] / w;
                    let mu = &fit.means[j * k..(j + 1) * k];
                    for a in 0..k {
                        for b in a..k {
                            let v = c[a * k + b] + p * (mu[a] - m[a]) * (mu[b] - m[b]);
                            c[a * k + b] = v;
                            c[b * k + a] = v;
                        }
                    }
                }
                let diag: Vec<f64> = (0..k).map(|i| fit.covs[i * k + i]).collect();
                Comps::Tied {
                    comps: order
                        .iter()
                        .map(|&j| {
                            let mu = &fit.means[j * k..(j + 1) * k];
                            EwCovDiag::at_moments(k, fit.mass[j], mu, &diag).without_runs()
                        })
                        .collect(),
                    total: Box::new(EwCov::at_moments(k, w, &m, &c).without_runs()),
                }
            }
        };
        self.comps = Some(comps);
        self.buf_x = Vec::new();
        self.buf_w = Vec::new();
        self.last_fit = None;
    }
}

/// The responsibilities and the log of the normalizer, about the largest
/// term; `None` where no term is finite.
fn normalize(logp: &[f64]) -> Option<(Vec<f64>, f64)> {
    let top = logp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return None;
    }
    let mut r: Vec<f64> = logp.iter().map(|&l| (l - top).exp()).collect();
    let s: f64 = r.iter().sum();
    for v in &mut r {
        *v /= s;
    }
    Some((r, top + s.ln()))
}

fn argmax(r: &[f64]) -> usize {
    let mut best = 0;
    for (j, &v) in r.iter().enumerate() {
        if v > r[best] {
            best = j;
        }
    }
    best
}

/// Kish's effective count of weights.
fn kish(w: &[f64]) -> f64 {
    let s: f64 = w.iter().sum();
    let q: f64 = w.iter().map(|v| v * v).sum();
    if q > 0.0 { s * s / q } else { 0.0 }
}

/// Each row's log-density terms under `fit` (with `reg_covar`), into
/// `logp` (`n × K`); `None` where a covariance does not factorize.
fn log_terms(cfg: &GmmCfg, fit: &Fit, xs: &[f64], logp: &mut Vec<f64>) -> Option<()> {
    let (k, kk) = (cfg.n_features, fit.weights.len());
    let n = xs.len() / k;
    logp.clear();
    logp.resize(n * kk, f64::NEG_INFINITY);
    let scorers = Scorer::all(cfg, &fit.covs, &fit.weights)?;
    let mut d = vec![0.0; n * k];
    for (j, s) in scorers.iter().enumerate() {
        if fit.weights[j] <= 0.0 {
            continue;
        }
        let mu = &fit.means[j * k..(j + 1) * k];
        let lw = fit.weights[j].ln();
        for r in 0..n {
            for i in 0..k {
                d[r * k + i] = xs[r * k + i] - mu[i];
            }
        }
        let q = s.quads(&d, k, n);
        let ld = s.log_det();
        for r in 0..n {
            logp[r * kk + j] = lw - 0.5 * (k as f64 * LN_2PI + ld + q[r]);
        }
    }
    Some(())
}

/// The weighted mean log-density of rows under `fit`; `-inf` where it
/// cannot score them.
fn mean_loglik(cfg: &GmmCfg, fit: &Fit, xs: &[f64], ws: &[f64]) -> f64 {
    let mut logp = Vec::new();
    if log_terms(cfg, fit, xs, &mut logp).is_none() {
        return f64::NEG_INFINITY;
    }
    let kk = fit.weights.len();
    let (mut s, mut t) = (0.0, 0.0);
    for (r, &w) in ws.iter().enumerate() {
        let Some((_, l)) = normalize(&logp[r * kk..(r + 1) * kk]) else {
            return f64::NEG_INFINITY;
        };
        s += w * l;
        t += w;
    }
    if t > 0.0 { s / t } else { f64::NEG_INFINITY }
}

/// The M-step from responsibilities `resp` (`n × K`): `None` where a
/// component holds no weight.
fn m_step(cfg: &GmmCfg, xs: &[f64], ws: &[f64], resp: &[f64], kk: usize) -> Option<Fit> {
    let k = cfg.n_features;
    let n = ws.len();
    let mut mass = vec![0.0; kk];
    let mut sq = vec![0.0; kk];
    let mut means = vec![0.0; kk * k];
    for r in 0..n {
        for j in 0..kk {
            let v = ws[r] * resp[r * kk + j];
            mass[j] += v;
            sq[j] += v * v;
            for i in 0..k {
                means[j * k + i] += v * xs[r * k + i];
            }
        }
    }
    if mass.iter().any(|&m| !(m > 0.0)) {
        return None;
    }
    for j in 0..kk {
        for i in 0..k {
            means[j * k + i] /= mass[j];
        }
    }
    // Each component's own covariance about its mean (`k × k` for `full`
    // and `tied`, `k` variances for `diag` and `spherical`), over its mass;
    // `spherical` then takes the variances' mean, and `tied` the components'
    // sum over the total mass, as scikit-learn's M-steps do.
    let whole = matches!(cfg.covariance, GmmCovariance::Full | GmmCovariance::Tied);
    let per = if whole { k * k } else { k };
    let mut covs = vec![0.0; kk * per];
    for r in 0..n {
        for j in 0..kk {
            let v = ws[r] * resp[r * kk + j];
            let mu = &means[j * k..(j + 1) * k];
            if whole {
                for a in 0..k {
                    let da = xs[r * k + a] - mu[a];
                    for b in 0..k {
                        covs[j * per + a * k + b] += v * da * (xs[r * k + b] - mu[b]);
                    }
                }
            } else {
                for a in 0..k {
                    let da = xs[r * k + a] - mu[a];
                    covs[j * per + a] += v * da * da;
                }
            }
        }
    }
    let total: f64 = mass.iter().sum();
    match cfg.covariance {
        GmmCovariance::Full | GmmCovariance::Diag => {
            for j in 0..kk {
                for c in &mut covs[j * per..(j + 1) * per] {
                    *c /= mass[j];
                }
            }
        }
        GmmCovariance::Spherical => {
            covs = (0..kk)
                .map(|j| covs[j * k..(j + 1) * k].iter().sum::<f64>() / (k as f64 * mass[j]))
                .collect();
        }
        GmmCovariance::Tied => {
            covs = (0..k * k)
                .map(|i| (0..kk).map(|j| covs[j * per + i]).sum::<f64>() / total)
                .collect();
        }
    }
    let weights = mass.iter().map(|m| m / total).collect();
    let eff = mass.iter().zip(&sq).map(|(m, q)| m * m / q).collect();
    Some(Fit {
        weights,
        means,
        covs,
        mass,
        eff,
    })
}

/// Batch EM on the rows from `n_init` seeded k-means++ starts, and from
/// `warm` (the last try's fit) where given, the best mean log-likelihood
/// kept; `None` where no start gives a fit.
fn best_fit(cfg: &GmmCfg, xs: &[f64], ws: &[f64], kk: usize, warm: Option<&Fit>) -> Option<Fit> {
    let n = ws.len();
    if n < kk {
        return None;
    }
    let mut rng = SplitMix64::new(cfg.seed);
    let mut best: Option<(Fit, f64)> = None;
    let mut keep = |fit: Fit| {
        let score = mean_loglik(cfg, &fit, xs, ws);
        if score.is_finite() && best.as_ref().is_none_or(|(_, b)| score > *b) {
            best = Some((fit, score));
        }
    };
    if let Some(start) = warm.filter(|f| f.weights.len() == kk)
        && let Some(fit) = em(cfg, xs, ws, start.clone())
    {
        keep(fit);
    }
    for _ in 0..cfg.n_init {
        if let Some(fit) = kmeanspp(cfg, xs, ws, kk, &mut rng).and_then(|f| em(cfg, xs, ws, f)) {
            keep(fit);
        }
    }
    best.map(|(f, _)| f)
}

/// A seeded k-means++ start over the weighted rows, each row to its nearest
/// centre, and the M-step on those labels: scikit-learn's
/// `init_params="k-means++"`.
fn kmeanspp(cfg: &GmmCfg, xs: &[f64], ws: &[f64], kk: usize, rng: &mut SplitMix64) -> Option<Fit> {
    let k = cfg.n_features;
    let n = ws.len();
    {
        let mut centres: Vec<usize> = vec![rng.choice(ws)];
        let mut d2 = vec![f64::INFINITY; n];
        while centres.len() < kk {
            let c = *centres.last().expect("one centre");
            for r in 0..n {
                let d: f64 = (0..k)
                    .map(|i| (xs[r * k + i] - xs[c * k + i]).powi(2))
                    .sum();
                d2[r] = d2[r].min(d);
            }
            let p: Vec<f64> = (0..n).map(|r| ws[r] * d2[r]).collect();
            if !(p.iter().sum::<f64>() > 0.0) {
                return None;
            }
            centres.push(rng.choice(&p));
        }
        let mut resp = vec![0.0; n * kk];
        for r in 0..n {
            let mut bj = 0;
            let mut bd = f64::INFINITY;
            for (j, &c) in centres.iter().enumerate() {
                let d: f64 = (0..k)
                    .map(|i| (xs[r * k + i] - xs[c * k + i]).powi(2))
                    .sum();
                if d < bd {
                    bd = d;
                    bj = j;
                }
            }
            resp[r * kk + bj] = 1.0;
        }
        m_step(cfg, xs, ws, &resp, kk)
    }
}

/// Batch EM from `fit` until the mean log-likelihood moves by less than
/// `tol`, or `max_iter` iterations; `None` where a step leaves a component
/// with no weight or a covariance that does not factorize.
fn em(cfg: &GmmCfg, xs: &[f64], ws: &[f64], mut fit: Fit) -> Option<Fit> {
    let kk = fit.weights.len();
    let n = ws.len();
    let mut resp = vec![0.0; n * kk];
    let mut logp = Vec::new();
    let mut ll = f64::NEG_INFINITY;
    for _ in 0..cfg.max_iter {
        log_terms(cfg, &fit, xs, &mut logp)?;
        let (mut s, mut t) = (0.0, 0.0);
        for r in 0..n {
            let (rr, l) = normalize(&logp[r * kk..(r + 1) * kk])?;
            resp[r * kk..(r + 1) * kk].copy_from_slice(&rr);
            s += ws[r] * l;
            t += ws[r];
        }
        let new_ll = s / t;
        fit = m_step(cfg, xs, ws, &resp, kk)?;
        let moved = (new_ll - ll).abs();
        ll = new_ll;
        if moved < cfg.tol {
            break;
        }
    }
    Some(fit)
}

/// Condition 1: each component's Kish count exceeds its parameters, and its
/// covariance stays positive definite less `reg_covar · I`.
fn identified(cfg: &GmmCfg, fit: &Fit) -> bool {
    let k = cfg.n_features;
    let kk = fit.weights.len();
    // A component's share of the covariance parameters: `tied`'s shared
    // matrix counts once, split over the components.
    let need = k as f64 + cfg.covariance.params(k, kk) as f64 / kk as f64;
    let reg = cfg.reg_covar;
    let pd = |c: &[f64]| {
        let mut m = c.to_vec();
        (0..k).for_each(|i| m[i * k + i] -= reg);
        strictly_pd(&m, k)
    };
    if cfg.covariance == GmmCovariance::Tied && !pd(&fit.covs[..k * k]) {
        return false;
    }
    for j in 0..kk {
        if !(fit.eff[j] > need) {
            return false;
        }
        let at = cfg.covariance.at(k, j);
        let held = match cfg.covariance {
            GmmCovariance::Full => pd(&fit.covs[at..at + k * k]),
            GmmCovariance::Diag => fit.covs[at..at + k].iter().all(|&v| v > reg),
            GmmCovariance::Spherical => fit.covs[at] > reg,
            GmmCovariance::Tied => true,
        };
        if !held {
            return false;
        }
    }
    true
}

/// Whether a symmetric matrix is positive definite, by a plain Cholesky
/// that takes no jitter.
fn strictly_pd(a: &[f64], k: usize) -> bool {
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if !(s > 0.0) {
                    return false;
                }
                l[i * k + i] = s.sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    }
    true
}

/// The slots a row reports: `cluster`, `log_density`, `dist`, then each
/// component's responsibility.
pub const GMM_LEAD_SLOTS: usize = 3;

impl Gmm {
    /// Output slot labels in emission order.
    pub fn labels(n_components: usize) -> Vec<String> {
        let mut out = vec!["cluster".to_string(), "log_density".into(), "dist".into()];
        out.extend((0..n_components).map(|j| format!("membership_{j}")));
        out
    }

    /// One row a component, in the fixed order: its weight `π_j`, its mean,
    /// then its covariance with `reg_covar` (full: the upper triangle by
    /// rows, `Σ_00, Σ_01, …, Σ_11, …`; diag: the variances; spherical: its
    /// one variance; tied: the shared upper triangle, on every row). `None`
    /// before the start.
    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        let fit = self.parameters()?;
        let (k, cov) = (self.cfg.n_features, self.cfg.covariance);
        Some(
            (0..self.cfg.n_components)
                .map(|j| {
                    let mut row = vec![fit.weights[j]];
                    row.extend_from_slice(&fit.means[j * k..(j + 1) * k]);
                    let at = cov.at(k, j);
                    match cov {
                        GmmCovariance::Full | GmmCovariance::Tied => {
                            for a in 0..k {
                                row.extend_from_slice(&fit.covs[at + a * k + a..at + (a + 1) * k]);
                            }
                        }
                        _ => row.extend_from_slice(&fit.covs[at..at + cov.per(k)]),
                    }
                    row
                })
                .collect(),
        )
    }

    /// The row's read after a decay `lam`: none where the decay leaves the
    /// history below the smallest normal double, which the row forgets.
    fn read_after(&self, x: &[f64], lam: f64) -> Option<GmmRead> {
        if lam * self.n < f64::MIN_POSITIVE {
            return None;
        }
        self.read(x)
    }

    /// The row's slots from its read, NaN where it is withheld.
    fn write(&self, read: Option<&GmmRead>, n_before: f64, pred: &mut Vec<f64>) {
        pred.clear();
        pred.resize(GMM_LEAD_SLOTS + self.cfg.n_components, f64::NAN);
        if let Some(r) = read.filter(|_| n_before >= self.cfg.min_weight) {
            pred[0] = r.cluster as f64;
            pred[1] = r.log_density;
            pred[2] = r.dist;
            pred[GMM_LEAD_SLOTS..].copy_from_slice(&r.resp);
        }
    }

    fn has_shape(&self) -> bool {
        let (k, kk) = (self.cfg.n_features, self.cfg.n_components);
        let comps = match &self.comps {
            None => true,
            Some(Comps::Full(c)) => {
                self.cfg.covariance == GmmCovariance::Full
                    && c.len() == kk
                    && c.iter().all(|e| e.has_shape(k))
            }
            Some(Comps::Diag(c)) => {
                self.cfg.covariance == GmmCovariance::Diag
                    && c.len() == kk
                    && c.iter().all(|e| e.has_shape(k))
            }
            Some(Comps::Spherical(c)) => {
                self.cfg.covariance == GmmCovariance::Spherical
                    && c.len() == kk
                    && c.iter().all(|e| e.has_shape(k))
            }
            Some(Comps::Tied { comps, total }) => {
                self.cfg.covariance == GmmCovariance::Tied
                    && comps.len() == kk
                    && comps.iter().all(|e| e.has_shape(k))
                    && total.has_shape(k)
            }
        };
        comps
            && self.buf_x.len() == self.buf_w.len() * k
            && (self.comps.is_none() || self.buf_w.is_empty())
            && self.n.is_finite()
            && self.n >= 0.0
    }
}

impl OnlineModel for Gmm {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let lam = self.cfg.decay.factor(d_clock);
        let n_before = self.n;
        // Decay moves no weight ratio, mean or covariance, so the read
        // before it is the read after it: `predict`'s.
        let read = self.read_after(x, lam);
        self.write(read.as_ref(), n_before, &mut out.pred);
        let w = if weight.is_finite() && weight > 0.0 {
            weight
        } else {
            0.0
        };
        self.learn_read(x, w, lam, read.as_ref());
        out.n_eff = n_before;
        out.extra = None;
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut pred = Vec::new();
        let lam = self.cfg.decay.factor(d_clock);
        self.write(self.read_after(x, lam).as_ref(), self.n, &mut pred);
        Step {
            pred,
            n_eff: self.n,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Gmm(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Gmm(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("gmm", m.cfg.check())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "gmm",
                        "its saved values have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "gmm",
                found: other.kind(),
            }),
        }
    }

    /// Zero: `gmm` regresses nothing, as `kmeans` does.
    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        GMM_LEAD_SLOTS + self.cfg.n_components
    }
}

#[cfg(test)]
#[path = "gmm_tests.rs"]
mod tests;
