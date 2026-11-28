//! `Ar`: an autoregression, AR(p), fit by exponentially weighted least
//! squares on lags of each target that the model keeps itself.
//!
//! # The model
//!
//! For each target `y`, with lags `ℓ_1 < … < ℓ_q` (by default `1..=p`) and
//! the row's features `x` as exogenous regressors:
//!
//! ```text
//! y_t = c + Σ_i φ_i y_{t−ℓ_i} + β' x_t + e_t
//! ```
//!
//! `c` is present with `fit_intercept` (statsmodels' `trend="c"`) and absent
//! without it (`trend="n"`).
//!
//! # The recursion
//!
//! Each target keeps a ring of its last `max(lags)` **learned** values and
//! an [`EwRidge`] over the design `z_t = (y_{t−ℓ_1}, …, y_{t−ℓ_q}, x_t)`
//! read from that ring. Per row, decay `lam = decay.factor(d_clock)`,
//! weight `w`, with `W` the weight of every row and `W_j` target `j`'s own:
//!
//! ```text
//! pred_t = [1, z_t] · β_{t−1}                 out of sample: before the update
//! W'     = lam W + w                          (n_eff, before the row: W)
//! W_j'   = lam W_j + w                        (a learned row of target j only)
//! S_j'   = (lam W_j S_j + w z z')  / W_j'     (EW mean of z z', kept centered)
//! r_j'   = (lam W_j r_j + w z y_j) / W_j'     (EW mean of z y_j, kept centered)
//! β_j    = (S_j + ridge D)⁻¹ r_j              solved after every learned row
//! ```
//!
//! `D` is the identity without the intercept's slot; `ridge = 0`, the
//! default, is ordinary least squares, which with no decay is statsmodels'
//! `AutoReg` on the same rows, to rounding. The moments are kept centred
//! (the weighted Welford step), so a series far from zero loses nothing to
//! its level.
//!
//! **A learned row** of target `j` is one with a positive weight and the
//! target present. It is pushed onto the target's ring; it is learned into
//! the fit only once the ring already holds `max(lags)` values, since before
//! that its lags do not exist (statsmodels' `hold_back = max(lags)`). Every
//! other row -- a missing target, or one of the first `max(lags)` learned
//! rows -- only ages the fit by the clock. A row of weight 0 learns nothing
//! and moves no ring: the next row's lags are the ones before it, as if it
//! were not there (statsmodels' `missing="drop"`).
//!
//! **Lags count learned rows, not clock units.** "The value `ℓ` rows back"
//! has no meaning on a clock, so the lags are positions in the series of
//! learned values, and the half-life alone is on the clock. A clock break
//! (a session change, a gap past `gap_cap`) empties every ring
//! ([`OnlineModel::clear_lags`]): the rows on either side are no longer
//! adjacent, and the series starts again, its first `max(lags)` values
//! rebuilding the ring. The fit is kept across the break.
//!
//! # The state
//!
//! Per target, the ring (`8 max(lags)` bytes) and its fit, whose size is set
//! by `k = q + n_features + 1`; never by the stream's length.
//! [`Ar::state_bytes`] bounds it, and `max_state_bytes` caps that bound at
//! construction.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::ewridge::{FitSystem, SystemKept};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwRidge, EwRidgeCfg, TargetGaps};

/// The lags an [`Ar`] takes: at least one, each at least 1, strictly
/// increasing (their order is the coefficients' order), none past
/// [`crate::MAX_LAG`].
pub fn check_lags(lags: &[usize]) -> Result<(), String> {
    let mut no = crate::Refusals::new();
    if lags.is_empty() {
        no.refuse(
            "ar: lags must hold at least one lag, but it is empty. Give the order p, or a \
             list of lags such as [1, 2].",
        );
    }
    if lags.contains(&0) {
        no.refuse(
            "ar: lags must each be at least 1, but one of them is 0, which would regress a \
             value on itself. Remove the 0 from lags.",
        );
    }
    if lags.windows(2).any(|w| w[0] >= w[1]) {
        no.refuse(format!(
            "ar: lags must be strictly increasing, since their order is the coefficients' \
             order, but it is {lags:?}. List the lags in increasing order, each once."
        ));
    }
    if let Some(&max) = lags.iter().max() {
        no.take(crate::check_lag_ceiling("lags", max).map_err(|e| format!("ar: {e}")));
    }
    no.finish()
}

/// The settings of an [`Ar`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArCfg {
    /// The series, one per target, each with its own lags and fit.
    pub n_targets: usize,
    /// Exogenous regressors, the row's features, shared by every target.
    pub n_features: usize,
    /// The lags, in learned rows: strictly increasing, each at least 1.
    /// `[1, …, p]` is AR(p).
    pub lags: Vec<usize>,
    /// The constant `c` (statsmodels' `trend="c"`); without it, `"n"`.
    pub fit_intercept: bool,
    pub decay: Decay,
    /// The ridge on the lag and feature coefficients (never the constant);
    /// 0 is least squares.
    pub ridge: f64,
    /// A target's prediction is withheld until the weight before the row
    /// reaches this.
    pub min_weight: f64,
    /// The bytes a group's state may hold; `None` is
    /// [`crate::DEFAULT_STATE_BYTES`]. The state's size is fixed by the
    /// settings, so it is checked once, when the model is built.
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// Choose the order by an information criterion after every learned
    /// row: lags `1..=q` for the `q` in `0..=max(lags)` that minimizes it,
    /// every order fitted on the same rows, the features in each. `None`
    /// fits every lag given.
    #[serde(default)]
    pub select: Option<ArSelect>,
    /// Clock units of history each fit is computed from, a hard cutoff
    /// (`ewridge`'s `window`): a row older than this is in no Gram. Under
    /// a window a fit can lose its spread as rows leave, so its
    /// definiteness is checked on every row.
    #[serde(default)]
    pub window: Option<f64>,
    /// Clock units between the window's snapshots (`ewridge`'s
    /// `window_every`).
    #[serde(default)]
    pub window_every: Option<f64>,
    /// At most this many rows between the window's snapshots.
    #[serde(default)]
    pub max_rows_between_snapshots: Option<usize>,
}

/// The criterion an [`Ar`] chooses its order by, with Kish's effective
/// count `n = W² / Σ w²` of the target's learned rows, decayed: for
/// `p` parameters and residual sum of squares `RSS` (on the weights),
/// `n ln(RSS) + 2p` (AIC) or `n ln(RSS) + p ln(n)` (BIC), each less the
/// constants every order shares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArSelect {
    Aic,
    Bic,
}

impl ArCfg {
    /// The deepest lag: the ring's depth.
    pub fn max_lag(&self) -> usize {
        self.lags.iter().copied().max().unwrap_or(0)
    }

    /// Regressors per target, the constant excluded: the lags, then the
    /// features.
    pub fn n_regressors(&self) -> usize {
        self.lags.len() + self.n_features
    }

    /// Coefficients per target: the constant (with `fit_intercept`), the
    /// lags in order, then the features.
    pub fn k_total(&self) -> usize {
        self.n_regressors() + usize::from(self.fit_intercept)
    }

    /// The cap on the state's bytes.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("ar: {e}")));
        // Features are optional: a pure autoregression has none.
        if self.n_targets == 0 {
            no.refuse(
                "ar: the model needs at least one target, the series it regresses on its own \
                 lags, but it has none. Name at least one column in targets.",
            );
        }
        no.take(check_lags(&self.lags));
        if !(self.ridge.is_finite() && self.ridge >= 0.0) {
            no.refuse(format!(
                "ar: ridge must be finite and at least 0, but it is {}. Set ridge to 0 for \
                 least squares, or to a finite positive value.",
                self.ridge
            ));
        }
        no.take(crate::model::check_min_weight("ar", self.min_weight));
        no.take(crate::clock_span::check_max_state_bytes(
            "ar",
            self.max_state_bytes,
        ));
        if self.select.is_some() {
            if self.lags.iter().enumerate().any(|(i, &l)| l != i + 1) {
                no.refuse(format!(
                    "ar: select chooses an order, lags 1 to q, so lags must be 1 to the largest \
                     order to try, but it is {:?}. Give lags as the largest order, or leave out \
                     select.",
                    self.lags
                ));
            }
            if self.ridge != 0.0 {
                no.refuse(format!(
                    "ar: select compares the orders' least-squares residuals, which a ridge of \
                     {} would not be. Set ridge to 0, or leave out select.",
                    self.ridge
                ));
            }
        }
        no.stop_here()?;
        // The window's own settings, in the words its fits refuse them by.
        no.take(
            self.ridge_cfg()
                .validate()
                .map_err(|e| e.replace("ewridge: ", "ar: ")),
        );
        no.stop_here()?;
        let bytes = Ar::state_bytes(self);
        if bytes > self.cap() {
            no.refuse(format!(
                "ar: the state of {} lags and {} features over {} targets would hold {bytes} \
                 bytes, past max_state_bytes of {}. Use fewer or shorter lags, or raise \
                 max_state_bytes.",
                self.lags.len(),
                self.n_features,
                self.n_targets,
                self.cap()
            ));
        }
        no.finish()
    }

    /// One target's fit: an [`EwRidge`] with a single target, solved after
    /// every learned row, its Gram over the target's own learned rows.
    fn ridge_cfg(&self) -> EwRidgeCfg {
        EwRidgeCfg {
            n_features: self.n_regressors(),
            n_targets: 1,
            fit_intercept: self.fit_intercept,
            decay: self.decay,
            ridge: vec![self.ridge],
            feature_sets: vec![],
            standardize: false,
            ridge_scale: false,
            coef_prior: None,
            session_shrink: None,
            long_half_life: None,
            // The gate is `Ar`'s own (`Ar::gate`): a fit that withheld its
            // predictions would age its residual variance on them alone, and
            // a floor, which withholds output only, would move the state.
            min_weight: 0.0,
            solve_every: 0.0,
            max_rows_between_solves: 1,
            solve_share: None,
            gram_block_rows: 0,
            target_gaps: TargetGaps::OwnRows,
            window: self.window,
            window_every: self.window_every,
            max_rows_between_snapshots: self.max_rows_between_snapshots,
            constraints: None,
            evidence: None,
            constraint: None,
            prior_weight: None,
            prior_include_intercept: true,
        }
    }
}

/// An AR(p) per target, with exogenous features; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ar {
    cfg: ArCfg,
    /// One fit per target.
    fits: Vec<EwRidge>,
    /// Per target, its last `max(lags)` learned values, oldest first.
    rings: Vec<VecDeque<f64>>,
    /// Per target, whether its fit has been defined: its Gram positive
    /// definite (`Ar::definite`). Its predictions are withheld until then.
    /// Once set it stays: decay scales a Gram and a row adds to it, and
    /// neither takes a definite one back to singular.
    defined: Vec<bool>,
    /// Under `select`, each target's coefficients at the order chosen
    /// after its last learned row, in [`Ar::coefficients`]' layout (zeros
    /// past the order); empty without `select`, or before a fit.
    chosen: Vec<Vec<f64>>,
    /// The row's design, kept so a row allocates nothing: not state.
    #[serde(skip)]
    z: Vec<f64>,
    /// One fit's step, likewise.
    #[serde(skip)]
    one: Step,
    /// The working space of the definiteness check and the order
    /// selection, likewise: not state.
    #[serde(skip)]
    work: Work,
}

/// The working space [`Ar`] reads its fits' systems into: kept between
/// rows so that neither the definiteness check nor the order selection
/// allocates or copies a Gram on a row. It is not state: a clone starts
/// empty and two are always equal.
#[derive(Debug, Default)]
struct Work {
    /// A window's view of a fit, the moments of the rows inside it.
    kept: SystemKept,
    /// The definiteness check's factor.
    l: Vec<f64>,
    /// The order selection's.
    scratch: SelectWork,
}

impl Clone for Work {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for Work {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// The order selection's working space: its factor, its forward solve
/// and its back solve.
#[derive(Debug, Default)]
pub(crate) struct SelectWork {
    l: Vec<f64>,
    r: Vec<f64>,
    beta: Vec<f64>,
}

impl Ar {
    pub fn new(cfg: ArCfg) -> Result<Self, String> {
        cfg.validate()?;
        let ridge = EwRidge::new(cfg.ridge_cfg())?;
        let depth = cfg.max_lag();
        Ok(Self {
            fits: vec![ridge; cfg.n_targets],
            rings: (0..cfg.n_targets)
                .map(|_| VecDeque::with_capacity(depth))
                .collect(),
            z: Vec::with_capacity(cfg.n_regressors()),
            one: Step::default(),
            work: Work::default(),
            defined: vec![false; cfg.n_targets],
            chosen: vec![Vec::new(); cfg.n_targets],
            cfg,
        })
    }

    /// The smallest share of a regressor's spread that the regressors
    /// before it may leave unexplained for the Gram to count as definite.
    /// An exactly collinear design leaves rounding, about `1e-16` to
    /// `1e-14` of it; `1e-10` is past that and below any design whose
    /// coefficients carry a digit.
    pub const DEFINITE: f64 = 1e-10;

    /// Whether the system `fit` solves is positive definite: its regressors'
    /// second moments, centred with an intercept and raw without one, plus
    /// the ridge on the diagonal, factor by Cholesky with every pivot above
    /// [`Self::DEFINITE`] of its diagonal. The guard on the fit's early
    /// rows is this condition, not a count of rows: until the lags and the
    /// features have spread in every direction, the coefficients are not
    /// determined, and the fit reports none.
    fn definite(
        sys: &FitSystem<'_>,
        intercept: bool,
        n: usize,
        ridge: f64,
        l: &mut Vec<f64>,
    ) -> bool {
        let c = sys.cov;
        let Some(off) = c.k().checked_sub(n) else {
            return false;
        };
        // The system solved: the moments plus the ridge on the diagonal.
        let at = |i: usize, j: usize| {
            let m = if intercept {
                c.cov(off + i, off + j)
            } else {
                c.raw(off + i, off + j)
            };
            if i == j { m + ridge } else { m }
        };
        l.clear();
        l.resize(n * n, 0.0);
        for i in 0..n {
            for j in 0..=i {
                let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
                let a = at(i, j) - s;
                if i == j {
                    let d = at(i, i);
                    if !(d > 0.0 && a > Self::DEFINITE * d) {
                        return false;
                    }
                    l[i * n + i] = a.sqrt();
                } else {
                    l[i * n + j] = a / l[j * n + j];
                }
            }
        }
        true
    }

    /// Keep each fit's factor, which the coefficients' standard errors are
    /// read from ([`OnlineModel::coef_variance`]), as `ewridge` keeps its
    /// own under `emit_se_coef`.
    pub fn set_keep_factor(&mut self, on: bool) {
        for fit in &mut self.fits {
            fit.set_keep_factor(on);
        }
    }

    /// Target `j`'s prediction at its chosen order for the design `z`
    /// (lags, then features); NaN before a choice.
    fn chosen_pred(&self, j: usize, z: &[f64]) -> f64 {
        let c = &self.chosen[j];
        if c.is_empty() {
            return f64::NAN;
        }
        let i0 = usize::from(self.cfg.fit_intercept);
        let base = if i0 == 1 { c[0] } else { 0.0 };
        base + c[i0..].iter().zip(z).map(|(b, v)| b * v).sum::<f64>()
    }

    /// The order the criterion `ic` chooses from `fit`'s moments, and its
    /// coefficients in [`Self::coefficients`]' layout.
    ///
    /// The fit holds the system at the largest order `P`. With the features
    /// put first and the lags after them in order, the system of order `q`
    /// is its leading block of `e + q` rows (`e` features), and the
    /// Cholesky factor of a leading block is the leading block of the
    /// factor. So one factor `S = L Lᵀ` and one forward solve `r = L⁻¹ c`
    /// give every order's residual mean square, `s_yy − Σ_{i < e+q} r_i²`,
    /// on the same rows (the learned rows past the first `P` values), and
    /// its coefficients by one back solve. `S`, `c` and `s_yy` are centred
    /// with an intercept and raw without one. `None` where the system is
    /// not definite or the target has no weight.
    pub(crate) fn select_order(
        sys: &FitSystem<'_>,
        cfg: &ArCfg,
        ic: ArSelect,
        work: &mut SelectWork,
        out: &mut Vec<f64>,
    ) -> bool {
        let Some(tm) = sys.tm else {
            return false;
        };
        let (e, p) = (cfg.n_features, cfg.lags.len());
        let n = e + p;
        let c = sys.cov;
        let Some(off) = c.k().checked_sub(n) else {
            return false;
        };
        let intercept = cfg.fit_intercept;
        // The reordered index: the features first, then the lags.
        let orig = |i: usize| if i < e { p + i } else { i - e };
        // The cross-moments the system is solved for: centred with an
        // intercept, raw without, `c + m ȳ` as the fit forms them.
        let Some(roff) = sys.cross_c.len().checked_sub(n) else {
            return false;
        };
        let rhs = |i: usize| {
            let k = roff + i;
            if intercept {
                sys.cross_c[k]
            } else {
                sys.cross_c[k] + sys.mj[k] * sys.my
            }
        };
        let (l, r) = (&mut work.l, &mut work.r);
        l.clear();
        l.resize(n * n, 0.0);
        for i in 0..n {
            for jj in 0..=i {
                let (a, b) = (off + orig(i), off + orig(jj));
                let m = if intercept { c.cov(a, b) } else { c.raw(a, b) };
                let s: f64 = (0..jj).map(|k| l[i * n + k] * l[jj * n + k]).sum();
                let v = m - s;
                if i == jj {
                    // Not above 0, NaN included: no definite system.
                    if v.is_nan() || v <= 0.0 {
                        return false;
                    }
                    l[i * n + i] = v.sqrt();
                } else {
                    l[i * n + jj] = v / l[jj * n + jj];
                }
            }
        }
        r.clear();
        r.resize(n, 0.0);
        for i in 0..n {
            let s: f64 = (0..i).map(|k| l[i * n + k] * r[k]).sum();
            r[i] = (rhs(orig(i)) - s) / l[i * n + i];
        }
        let (mean_y, var_y) = (tm.means()[0], tm.vars()[0]);
        let syy = if intercept {
            var_y
        } else {
            var_y + mean_y * mean_y
        };
        let w = sys.wj;
        let Some(kish) = tm.n_kish(&[w])[0] else {
            return false;
        };
        if !(w > 0.0 && kish > 0.0) {
            return false;
        }
        let mut best: Option<(f64, usize)> = None;
        let mut explained = r[..e].iter().map(|v| v * v).sum::<f64>();
        for q in 0..=p {
            if q > 0 {
                explained += r[e + q - 1] * r[e + q - 1];
            }
            let rss = w * (syy - explained);
            let k = (e + q + usize::from(intercept)) as f64;
            let pen = match ic {
                ArSelect::Aic => 2.0,
                ArSelect::Bic => kish.ln(),
            };
            // An order that fits exactly, its residual rounding at or below
            // 0, is the best there is: the first such wins.
            let value = if rss > 0.0 {
                kish * rss.ln() + pen * k
            } else {
                f64::NEG_INFINITY
            };
            if best.is_none_or(|(b, _)| value < b) {
                best = Some((value, q));
            }
        }
        let Some((_, q)) = best else {
            return false;
        };
        // The back solve of the leading block of order q.
        let m = e + q;
        let beta = &mut work.beta;
        beta.clear();
        beta.resize(m, 0.0);
        for i in (0..m).rev() {
            let s: f64 = (i + 1..m).map(|k| l[k * n + i] * beta[k]).sum();
            beta[i] = (r[i] - s) / l[i * n + i];
        }
        let i0 = usize::from(intercept);
        out.clear();
        out.resize(cfg.k_total(), 0.0);
        for (i, b) in beta.iter().enumerate() {
            out[i0 + orig(i)] = *b;
        }
        if intercept {
            let means = sys.cov.means();
            let moff = means.len() - n;
            let at_means: f64 = (0..m).map(|i| beta[i] * means[moff + orig(i)]).sum();
            out[0] = mean_y - at_means;
        }
        true
    }

    pub fn cfg(&self) -> &ArCfg {
        &self.cfg
    }

    /// The bytes the state holds at most, by its settings: per target the
    /// ring, `8 max(lags)`, and the fit's arrays, at most `8 (4k² + 17k +
    /// 64)` for `k = k_total` (its centred Gram and the factor it solves
    /// with, the means, cross-moments and coefficients, the coefficients at
    /// the order `select` chose, and its scalars).
    /// `tests` holds this above the bytes a state writes, fresh and after a
    /// stream.
    pub fn state_bytes(cfg: &ArCfg) -> u64 {
        let k = cfg.k_total() as u64;
        let per = 8 * cfg.max_lag() as u64 + 8 * (4 * k * k + 17 * k + 64);
        per.saturating_mul(cfg.n_targets as u64)
    }

    /// Each target's coefficients, `k_total` each: the constant (with
    /// `fit_intercept`), the lags in the order given, then the features.
    /// `None` before the first fit.
    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        let k = self.cfg.k_total();
        let mut out = Vec::with_capacity(self.fits.len());
        let mut any = false;
        for (j, (fit, &defined)) in self.fits.iter().zip(&self.defined).enumerate() {
            if self.cfg.select.is_some() {
                if defined && !self.chosen[j].is_empty() {
                    any = true;
                    out.push(self.chosen[j].clone());
                } else {
                    out.push(vec![f64::NAN; k]);
                }
                continue;
            }
            // A fit not yet defined reports none.
            match fit.coefficients().filter(|_| defined) {
                Some(c) => {
                    any = true;
                    out.push(c[0].clone());
                }
                None => out.push(vec![f64::NAN; k]),
            }
        }
        any.then_some(out)
    }

    /// The weight of every row before the next (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.fits[0].n_eff()
    }

    /// Target `j`'s ring, oldest first.
    pub fn ring(&self, j: usize) -> &VecDeque<f64> {
        &self.rings[j]
    }

    /// Whether target `j`'s ring holds every lag.
    fn full(&self, j: usize) -> bool {
        self.rings[j].len() == self.cfg.max_lag()
    }

    /// Target `j`'s design for the next row into `z`: its lags from the
    /// ring, then the features. Zeros in the lags' place while the ring is
    /// short: such a row is not learned, and its prediction is not read.
    fn design_into(
        lags: &[usize],
        ring: &VecDeque<f64>,
        depth: usize,
        x: &[f64],
        z: &mut Vec<f64>,
    ) {
        let total = lags.len() + x.len();
        if z.len() != total {
            z.resize(total, 0.0);
        }
        if ring.len() == depth {
            for (dst, &l) in z[..lags.len()].iter_mut().zip(lags) {
                *dst = ring[depth - l];
            }
        } else {
            z[..lags.len()].fill(0.0);
        }
        if !x.is_empty() {
            z[lags.len()..].copy_from_slice(x);
        }
    }

    /// `pred` withheld, every target's, while the weight before the row is
    /// below `floor` (`min_weight`), as `ewridge` gates on its shared weight.
    fn gate(pred: &mut [f64], n_eff: f64, floor: f64) {
        if n_eff < floor {
            pred.fill(f64::NAN);
        }
    }

    /// The row's report, the state untouched: target `j`'s prediction from
    /// its fit, or NaN while its ring is short, gated at `floor`.
    fn read(&self, x: &[f64], d_clock: f64, floor: f64, out: &mut Step) {
        let depth = self.cfg.max_lag();
        let mut z = Vec::with_capacity(self.cfg.n_regressors());
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets, f64::NAN);
        out.extra = None;
        out.n_eff = self.n_eff();
        for j in 0..self.cfg.n_targets {
            if !self.full(j) || !self.defined[j] {
                continue;
            }
            Self::design_into(&self.cfg.lags, &self.rings[j], depth, x, &mut z);
            out.pred[j] = match self.cfg.select {
                Some(_) => self.chosen_pred(j, &z),
                None => self.fits[j].predict(&z, d_clock).pred[0],
            };
        }
        Self::gate(&mut out.pred, out.n_eff, floor);
    }
}

impl OnlineModel for Ar {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        debug_assert_eq!(x.len(), self.cfg.n_features);
        debug_assert_eq!(y.len(), self.cfg.n_targets);
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let m = self.cfg.n_targets;
        let depth = self.cfg.max_lag();
        if out.pred.len() != m {
            out.pred.clear();
            out.pred.resize(m, f64::NAN);
        } else {
            out.pred.fill(f64::NAN);
        }
        out.extra = None;
        let learns = weight > 0.0;
        for (j, target) in y.iter().enumerate() {
            let full = self.rings[j].len() == depth;
            Self::design_into(&self.cfg.lags, &self.rings[j], depth, x, &mut self.z);
            let yj = target.filter(|_| learns);
            // Every fit takes every row, so its weight and its clock are the
            // stream's; only a row with its lags and its target teaches it.
            let taught = if full { yj } else { None };
            self.fits[j].step_into(&self.z, &[taught], d_clock, weight, &mut self.one);
            if j == 0 {
                out.n_eff = self.one.n_eff;
            }
            if full && self.defined[j] {
                out.pred[j] = match self.cfg.select {
                    Some(_) => self.chosen_pred(j, &self.z),
                    None => self.one.pred[0],
                };
            }
            // The fit a learned row taught, checked until it is defined;
            // under a window, on every row, as rows leave it.
            let check =
                (taught.is_some() && !self.defined[j]) || (self.cfg.window.is_some() && full);
            let moved = taught.is_some() || self.cfg.window.is_some();
            let choose = self.cfg.select.is_some() && moved;
            if check || choose {
                // The fit's system read in place, the window's view in the
                // kept working space: no copy of the Gram a row.
                let work = &mut self.work;
                let sys = self.fits[j].system(0, &mut work.kept);
                if check {
                    self.defined[j] = sys.as_ref().is_some_and(|sys| {
                        Self::definite(
                            sys,
                            self.cfg.fit_intercept,
                            self.cfg.n_regressors(),
                            self.cfg.ridge,
                            &mut work.l,
                        )
                    });
                }
                if let (Some(ic), true) = (self.cfg.select, choose && self.defined[j]) {
                    let chosen = &mut self.chosen[j];
                    let ok = sys.as_ref().is_some_and(|sys| {
                        Self::select_order(sys, &self.cfg, ic, &mut work.scratch, chosen)
                    });
                    if !ok {
                        chosen.clear();
                    }
                }
            }
            if let Some(v) = yj {
                let ring = &mut self.rings[j];
                if ring.len() == depth {
                    ring.pop_front();
                }
                ring.push_back(v);
            }
        }
        Self::gate(&mut out.pred, out.n_eff, self.cfg.min_weight);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read(x, d_clock, self.cfg.min_weight, &mut out);
        out
    }

    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let _ = y;
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read(x, d_clock, floor, &mut out);
        out
    }

    /// The rows behind a break are no longer the series' last values: every
    /// ring empties, and each target's fit waits for `max(lags)` learned
    /// rows before it learns again. The fits are kept.
    fn clear_lags(&mut self) {
        for ring in &mut self.rings {
            ring.clear();
        }
    }

    /// Each target's coefficients' variances over its noise, its fit's as
    /// `ewridge` reports them, target after target; NaN for a target whose
    /// fit is not defined yet.
    fn coef_variance(&self) -> Option<crate::CoefVariance> {
        let k = self.cfg.k_total();
        let mut out = Vec::with_capacity(self.fits.len());
        let mut any = false;
        for (fit, &defined) in self.fits.iter().zip(&self.defined) {
            match fit.coef_variance() {
                Some(crate::CoefVariance::PerNoise(v)) if defined => {
                    any = true;
                    out.push(v.into_iter().next().unwrap_or_else(|| vec![f64::NAN; k]));
                }
                _ => out.push(vec![f64::NAN; k]),
            }
        }
        any.then_some(crate::CoefVariance::PerNoise(out))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        let mut one = Vec::with_capacity(1);
        for fit in &self.fits {
            // The fit's own: inside the window under one.
            fit.target_n_eff_into(&mut one);
            out.push(one[0]);
        }
        true
    }

    fn stamp_next(&mut self, stamp: crate::Stamp) {
        for fit in &mut self.fits {
            fit.stamp_next(stamp);
        }
    }

    fn stamp_next_on_rows(&mut self, stamp: crate::Stamp) {
        for fit in &mut self.fits {
            fit.stamp_next_on_rows(stamp);
        }
    }

    /// One budget for the group: each target's fit takes its share, so
    /// their snapshots together stay under it.
    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        let n = self.fits.len().max(1) as f64;
        let share = budget.map(|b| match b {
            crate::WindowBudget::Thin(m) => crate::WindowBudget::Thin(m / n),
            crate::WindowBudget::Refuse(m) => crate::WindowBudget::Refuse(m / n),
        });
        for fit in &mut self.fits {
            fit.set_window_budget(share);
        }
    }

    fn set_window_closed(&mut self, closed: crate::WindowClosed) {
        for fit in &mut self.fits {
            fit.set_window_closed(closed);
        }
    }

    /// Any fit's window past its share, its bytes counted for the group:
    /// every fit takes the same rows on the same stamps, so their rings
    /// hold alike.
    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        let n = self.fits.len();
        self.fits
            .iter()
            .find_map(|f| f.window_over_budget())
            .map(|(bytes, cadence)| (bytes * n, cadence))
    }

    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        self.fits.first().and_then(|f| f.window_shadow())
    }

    fn window_could_refuse(&self, rows: usize) -> bool {
        self.fits.iter().any(|f| f.window_could_refuse(rows))
    }

    fn state(&self) -> State {
        State::new(ModelState::Ar(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Ar(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("ar", m.cfg.validate())?;
                let depth = m.cfg.max_lag();
                let want = m.cfg.ridge_cfg();
                let shaped = m.fits.len() == m.cfg.n_targets
                    && m.defined.len() == m.cfg.n_targets
                    && m.chosen.len() == m.cfg.n_targets
                    && m.chosen
                        .iter()
                        .all(|c| c.is_empty() || c.len() == m.cfg.k_total())
                    && m.rings.len() == m.cfg.n_targets
                    && m.rings.iter().all(|r| r.len() <= depth)
                    && m.fits.iter().all(|f| *f.cfg() == want);
                if !shaped {
                    return Err(crate::model::damaged(
                        "ar",
                        "its lags or its fits have the wrong shape",
                    ));
                }
                for ring in &mut m.rings {
                    ring.reserve_exact(depth - ring.len());
                }
                // Each fit through its own loader, which checks its shape and
                // rebuilds what it keeps out of the file.
                let fits = std::mem::take(&mut m.fits);
                for fit in fits {
                    let state = State::new(ModelState::EwRidge(Box::new(fit)));
                    let fit = EwRidge::restore_owned(state)
                        .map_err(|_| crate::model::damaged("ar", "a target's fit is damaged"))?;
                    m.fits.push(fit);
                }
                m.z = Vec::with_capacity(m.cfg.n_regressors());
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "ar",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
mod tests;
