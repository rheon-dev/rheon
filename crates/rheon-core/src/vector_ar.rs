//! `VectorAr`: a vector autoregression, VAR(p), of `k` series together, fit
//! by exponentially weighted least squares on lags the model keeps itself,
//! with online Granger-causality tests.
//!
//! # The model
//!
//! With the row's features `x` (`e` of them) as exogenous regressors:
//!
//! ```text
//! y_t = c + B_x x_t + A_1 y_{t−1} + … + A_p y_{t−p} + u_t,    u_t ~ (0, Σ_u)
//! ```
//!
//! `c` is present with `fit_intercept` (statsmodels' `trend="c"`) and absent
//! without it (`trend="n"`). The `k` equations share their regressors
//! `z_t = [x_t; y_{t−1}; …; y_{t−p}]` (`d = e + kp`; the features first,
//! then the lags lag by lag, statsmodels' `params` order), so least squares
//! equation by equation is the system's estimate, and every equation reads
//! one Gram.
//!
//! # The recursion
//!
//! The model keeps a ring of its last `p` **learned** rows and one
//! [`EwRidge`] over `[z_t, y_t]`, whose Gram holds the regressors' moments,
//! their cross-moments with the targets and the targets' own co-moments,
//! all exponentially weighted means kept centred (the weighted Welford
//! step). Per row, decay `lam = decay.factor(d_clock)`, weight `w`:
//!
//! ```text
//! pred_t = c + B' z_t                         out of sample: the fit before the row
//! W'     = lam W + w                          (n_eff, before the row: W)
//! S'     = (lam W_S S + w v vᵀ) / W_S'        v = [z_t, y_t], a learned row only
//! ```
//!
//! and on the rows the fit's solve schedule brings due (the wrapped
//! `ewridge`'s: `solve_every`, `max_rows_between_solves` or its weight
//! share), the coefficients `B = (S_zz + ridge I)⁻¹ S_zy` and, from one
//! Cholesky factor of `S_zz`, everything the tests read
//! ([`fit`](self::fit)'s module docs have the algebra).
//!
//! **A learned row** has a positive weight and every target present. Only
//! it enters the ring, and it teaches the fit once the ring holds `p` rows.
//! A row with a target missing is predicted, ages the fit by its clock step
//! and learns nothing (statsmodels' `missing="drop"`): the lags join across
//! it. A row of weight 0 learns nothing and moves no ring.
//!
//! **Lags count learned rows, not clock units.** The half-life and the
//! window are on the clock; "the value `ℓ` rows back" has no meaning there.
//! A clock break (a session change, a gap past `gap_cap`) empties the ring
//! ([`OnlineModel::clear_lags`]); the fit is kept, and the next `p` learned
//! rows refill the ring before it learns again.
//!
//! # Identification
//!
//! The fit reports nothing until the Gram identifies it: the Cholesky factor
//! of `S_zz` plus the ridge has every pivot above `1e-10` of its diagonal
//! (`select` needs order 1 identified). That is a condition, not a count of
//! rows. Outside a window it stays passed once passed; under a window it is
//! read again at every solve, as rows leave. The tests and the standard
//! errors also need residual degrees of freedom, `df = n − (d + 1) > 0` with
//! `n = W² / Σw²` Kish's count of the learned rows.
//!
//! # The state
//!
//! The ring (`8kp` bytes), the fit (its Gram over `d + k + 1` slots) and what
//! the last solve read; never the stream's length. [`VectorAr::state_bytes`]
//! bounds it, and `max_state_bytes` caps that bound at construction.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::ewridge::SystemKept;
use crate::model::{Extra, ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, EwRidge, EwRidgeCfg, TargetGaps};

mod fit;
pub mod reads;
mod tails;

pub use fit::VectorArSelect;
pub use reads::{VectorArFit, VectorArTable};

use fit::{Factor, Moments, Scratch, Shape};

/// One Granger test a [`VectorAr`] reports on every row: the lags of the
/// `causing` series in the equations of the `caused` ones. Indices into the
/// targets; each set non-empty, without repeats, the two disjoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorArTest {
    pub caused: Vec<usize>,
    pub causing: Vec<usize>,
}

/// Which statistic a Granger test reports, and the law its p-value is read
/// from. `λ` is the Wald statistic, `q` the number of restrictions, `df` the
/// residual degrees of freedom and `n` Kish's count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalityKind {
    /// `λ / q` against `F(q, k·df)`: statsmodels' `test_causality(kind="f")`.
    #[default]
    F,
    /// `λ` against `χ²(q)`: `test_causality(kind="wald")`.
    Wald,
    /// `λ / q` against `F(q, df)`: `grangercausalitytests`' `ssr_ftest`.
    SsrF,
    /// `n λ / df` against `χ²(q)`: its `ssr_chi2test`.
    SsrChi2,
    /// `n ln(1 + λ / df)` against `χ²(q)`: its `lrtest`.
    Lr,
}

impl CausalityKind {
    /// Whether the kind tests one caused equation alone.
    pub fn single_equation(self) -> bool {
        matches!(self, Self::SsrF | Self::SsrChi2 | Self::Lr)
    }

    /// The statistic, its p-value and its two degrees of freedom (the
    /// second NaN for a χ² law) from the Wald statistic `lambda` with `q`
    /// restrictions, `df` residual degrees of freedom, `n` Kish's count and
    /// `k` equations.
    pub fn read(self, lambda: f64, q: f64, df: f64, n: f64, k: f64) -> [f64; 4] {
        match self {
            Self::F => {
                let s = lambda / q;
                [s, tails::f_sf(s, q, k * df), q, k * df]
            }
            Self::Wald => [lambda, tails::chi2_sf(lambda, q), q, f64::NAN],
            Self::SsrF => {
                let s = lambda / q;
                [s, tails::f_sf(s, q, df), q, df]
            }
            Self::SsrChi2 => {
                let s = n * lambda / df;
                [s, tails::chi2_sf(s, q), q, f64::NAN]
            }
            Self::Lr => {
                let s = n * libm::log1p(lambda / df);
                [s, tails::chi2_sf(s, q), q, f64::NAN]
            }
        }
    }
}

/// The settings of a [`VectorAr`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorArCfg {
    /// The series, `k`, at least one.
    pub n_targets: usize,
    /// Exogenous regressors, the row's features, in every equation.
    pub n_features: usize,
    /// The order `p`, lags `1..=p`, in learned rows.
    pub lags: usize,
    /// The constant `c` (statsmodels' `trend="c"`); without it, `"n"`.
    pub fit_intercept: bool,
    pub decay: Decay,
    /// The ridge on the lag and feature coefficients (never the constant);
    /// 0, least squares, beside `select` and any test.
    pub ridge: f64,
    /// Every prediction and test is withheld until the weight before the
    /// row reaches this.
    pub min_weight: f64,
    /// The wrapped fit's solve cadence in clock units (`ewridge`'s
    /// `solve_every`); `<= 0` solves every row.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub solve_every: f64,
    /// At most this many rows with weight between solves.
    pub max_rows_between_solves: u32,
    /// A solve once the weight learned since the last reaches this share
    /// of the fit's (`ewridge`'s `solve_share`); `None` keeps the clock.
    #[serde(default)]
    pub solve_share: Option<f64>,
    /// The bytes a group's state may hold; `None` is 256 KB. The state's
    /// size is fixed by the settings, so it is checked once, when the model
    /// is built.
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// Choose the order `q` in `0..=p` that minimizes this criterion at
    /// every solve, every order fitted on the same rows.
    #[serde(default)]
    pub select: Option<VectorArSelect>,
    /// The Granger tests reported on every row.
    #[serde(default)]
    pub causality: Vec<VectorArTest>,
    /// The statistic every test reports.
    #[serde(default)]
    pub causality_kind: CausalityKind,
    /// Clock units of history the fit sees, a hard cutoff (`ewridge`'s
    /// `window`): under it the fit's identification is read at every solve.
    #[serde(default)]
    pub window: Option<f64>,
    /// Clock units between the window's snapshots.
    #[serde(default)]
    pub window_every: Option<f64>,
    /// At most this many rows between the window's snapshots.
    #[serde(default)]
    pub max_rows_between_snapshots: Option<usize>,
}

impl VectorArCfg {
    /// Regressors per equation, the constant excluded: the features, then
    /// the `kp` lags.
    pub fn n_regressors(&self) -> usize {
        self.n_features + self.n_targets * self.lags
    }

    /// Coefficients per equation: the constant (with `fit_intercept`), the
    /// features, then the lags lag by lag.
    pub fn k_total(&self) -> usize {
        self.n_regressors() + usize::from(self.fit_intercept)
    }

    /// The cap on the state's bytes.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    fn shape(&self) -> Shape {
        Shape {
            e: self.n_features,
            k: self.n_targets,
            p: self.lags,
            intercept: self.fit_intercept,
        }
    }

    fn check_test(&self, i: usize, t: &VectorArTest, no: &mut crate::Refusals) {
        let k = self.n_targets;
        for (name, set) in [("caused", &t.caused), ("causing", &t.causing)] {
            if set.is_empty() {
                no.refuse(format!(
                    "vector_ar: causality test {i} names no {name} series. Name at least one of \
                     the targets."
                ));
            }
            if let Some(&bad) = set.iter().find(|&&j| j >= k) {
                no.refuse(format!(
                    "vector_ar: causality test {i} names series {bad} as {name}, past the {k} \
                     targets. Name a target."
                ));
            }
            let mut seen = set.clone();
            seen.sort_unstable();
            if seen.windows(2).any(|w| w[0] == w[1]) {
                no.refuse(format!(
                    "vector_ar: causality test {i} names a series twice as {name}. Name each \
                     once."
                ));
            }
        }
        if t.caused.iter().any(|j| t.causing.contains(j)) {
            no.refuse(format!(
                "vector_ar: causality test {i} names a series as both caused and causing, but a \
                 series' own lags are not a test of another's. Leave it out of one of the two."
            ));
        }
        if self.causality_kind.single_equation() && t.caused.len() != 1 {
            no.refuse(format!(
                "vector_ar: causality_kind {:?} tests one caused equation, but causality test \
                 {i} names {} caused series. Name one, or use \"f\" or \"wald\", which test a \
                 set.",
                self.causality_kind,
                t.caused.len()
            ));
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("vector_ar: {e}")));
        if self.n_targets == 0 {
            no.refuse(
                "vector_ar: the model needs at least one target, the series it regresses on \
                 their lags, but it has none. Name the series in targets.",
            );
        }
        if self.lags == 0 {
            no.refuse(
                "vector_ar: lags must be at least 1, the order p of lags 1 to p, but it is 0. \
                 Give the order.",
            );
        } else {
            no.take(
                crate::check_lag_ceiling("lags", self.lags).map_err(|e| format!("vector_ar: {e}")),
            );
        }
        if !(self.ridge.is_finite() && self.ridge >= 0.0) {
            no.refuse(format!(
                "vector_ar: ridge must be finite and at least 0, but it is {}. Set ridge to 0 \
                 for least squares, or to a finite positive value.",
                self.ridge
            ));
        }
        if self.ridge != 0.0 && self.select.is_some() {
            no.refuse(format!(
                "vector_ar: select compares the orders' least-squares residuals, which a ridge \
                 of {} would not be. Set ridge to 0, or leave out select.",
                self.ridge
            ));
        }
        if self.ridge != 0.0 && !self.causality.is_empty() {
            no.refuse(format!(
                "vector_ar: a causality test's law holds for least squares, which a ridge of {} \
                 is not. Set ridge to 0, or leave out causality.",
                self.ridge
            ));
        }
        no.take(crate::model::check_min_weight("vector_ar", self.min_weight));
        no.take(crate::clock_span::check_max_state_bytes(
            "vector_ar",
            self.max_state_bytes,
        ));
        for (i, t) in self.causality.iter().enumerate() {
            self.check_test(i, t, &mut no);
        }
        no.stop_here()?;
        no.take(
            self.ridge_cfg()
                .validate()
                .map_err(|e| e.replace("ewridge: ", "vector_ar: ")),
        );
        no.stop_here()?;
        let bytes = VectorAr::state_bytes(self);
        if bytes > self.cap() {
            no.refuse(format!(
                "vector_ar: the state of {} series at {} lags with {} features would hold \
                 {bytes} bytes, past max_state_bytes of {}. Use fewer lags or series, or raise \
                 max_state_bytes.",
                self.n_targets,
                self.lags,
                self.n_features,
                self.cap()
            ));
        }
        no.finish()
    }

    /// The fit: one [`EwRidge`] over `[z, y]` with the `k` targets, its
    /// solve on the regressors `z` alone (a feature set), so its Gram holds
    /// the targets' co-moments beside the regressors'.
    fn ridge_cfg(&self) -> EwRidgeCfg {
        let d = self.n_regressors();
        EwRidgeCfg {
            n_features: d + self.n_targets,
            n_targets: self.n_targets,
            fit_intercept: self.fit_intercept,
            decay: self.decay,
            ridge: vec![self.ridge],
            feature_sets: vec![("var".into(), (0..d).collect())],
            standardize: false,
            ridge_scale: false,
            coef_prior: None,
            session_shrink: None,
            long_half_life: None,
            // The gate is the model's own: a fit that withheld its
            // predictions would age its residual variance on them alone.
            min_weight: 0.0,
            solve_every: self.solve_every,
            max_rows_between_solves: self.max_rows_between_solves,
            solve_share: self.solve_share,
            gram_block_rows: 0,
            target_gaps: TargetGaps::OwnRows,
            window: self.window,
            window_every: self.window_every,
            max_rows_between_snapshots: self.max_rows_between_snapshots,
        }
    }
}

/// What the last solve read beside the coefficients: the order in force,
/// the coefficients there under `select`, each test's Wald statistic, the
/// degrees of freedom they are read on and the coefficients' variances.
/// No value here is a logarithm or a tail: the p-values are read for
/// output only.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct LastRead {
    /// The order the reads are at: `p` without `select`, the chosen one
    /// under it; `None` where the solve identified none.
    order: Option<usize>,
    /// Under `select`, each target's coefficients at the chosen order in
    /// [`VectorAr::coefficients`]' layout, zeros past the order; empty
    /// otherwise.
    chosen: Vec<f64>,
    /// Per declared test, its Wald statistic; `None` where not defined.
    wald: Vec<Option<f64>>,
    /// Kish's count of the learned rows, and the residual degrees of
    /// freedom at the order.
    n_kish: Option<f64>,
    df: Option<f64>,
    /// statsmodels' `bse²` per target and coefficient, under
    /// `keep_factor`; empty otherwise.
    coef_var: Vec<Option<f64>>,
}

/// The bytes a state's field names take beside its numbers, which the
/// file writes as a map: measured at 1.6 KB for every shape.
const NAMES: u64 = 2048;

/// A VAR(p) with Granger tests; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorAr {
    cfg: VectorArCfg,
    fit: EwRidge,
    /// The last `p` learned rows, oldest first, `k` values each.
    ring: VecDeque<f64>,
    /// Whether the fit is identified ([`fit::DEFINITE`]): set at the first
    /// solve whose system passes and kept outside a window; under one, read
    /// at every solve. Everything is withheld until it is set.
    defined: bool,
    last: LastRead,
    /// Read the coefficients' variances at every solve, which `se_coef`
    /// reports. Configuration that travels with the state.
    #[serde(default)]
    keep_factor: bool,
    /// The working space, kept so a row allocates nothing: not state.
    #[serde(skip)]
    work: Work,
}

/// The working space a solve's read and a row's report use, kept between
/// rows: not state. A clone starts empty, and two are always equal.
#[derive(Debug, Default)]
struct Work {
    kept: SystemKept,
    mom: Moments,
    factor: Factor,
    scratch: Scratch,
    b: Vec<f64>,
    /// The row's design and the targets the fit is taught, and its step.
    z: Vec<f64>,
    taught: Vec<Option<f64>>,
    one: Step,
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

impl VectorAr {
    pub fn new(cfg: VectorArCfg) -> Result<Self, String> {
        cfg.validate()?;
        let fit = EwRidge::new(cfg.ridge_cfg())?;
        Ok(Self {
            fit,
            ring: VecDeque::with_capacity(cfg.lags * cfg.n_targets),
            defined: false,
            last: LastRead::default(),
            keep_factor: false,
            work: Work::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &VectorArCfg {
        &self.cfg
    }

    /// Read the coefficients' variances at every solve, which
    /// [`OnlineModel::coef_variance`] reports.
    pub fn set_keep_factor(&mut self, on: bool) {
        self.keep_factor = on;
    }

    /// The bytes the state holds at most, by its settings: the ring, `8kp`;
    /// the fit's arrays over `K = d + k + 1` slots, at most `8 (4K² + 17K +
    /// 64)` for its Gram and its first target and `8 (4K + 16)` for each
    /// further target; the last solve's read, at most `8 (2k(d + 1) + T +
    /// 10)` for `T` tests; and 2 KiB for the names the fields are written
    /// under. `tests` holds this above the bytes a state writes, fresh and
    /// after a stream, and within four times them.
    pub fn state_bytes(cfg: &VectorArCfg) -> u64 {
        let k = cfg.n_targets as u64;
        let d = cfg.n_regressors() as u64;
        let big_k = d + k + 1;
        let ring = 8 * k * cfg.lags as u64;
        let fit =
            8 * (4 * big_k * big_k + 17 * big_k + 64) + 8 * k.saturating_sub(1) * (4 * big_k + 16);
        let read = 8 * (2 * k * (d + 1) + cfg.causality.len() as u64 + 10);
        ring.saturating_add(fit)
            .saturating_add(read)
            .saturating_add(NAMES)
    }

    /// Whether the fit is withheld because its rows do not identify it.
    pub fn not_identified(&self) -> bool {
        !self.defined
    }

    /// The weight of every row before the next (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.fit.n_eff()
    }

    /// The ring, oldest first, `k` values a row.
    pub fn ring(&self) -> &VecDeque<f64> {
        &self.ring
    }

    /// The order in force: `p` without `select`, the chosen one under it;
    /// `None` before the fit is identified.
    pub fn order(&self) -> Option<usize> {
        if self.defined { self.last.order } else { None }
    }

    /// Values a row reports beside its predictions: a statistic and a
    /// p-value per test, then the order under `select`.
    pub fn n_extra(&self) -> usize {
        2 * self.cfg.causality.len() + usize::from(self.cfg.select.is_some())
    }

    fn full(&self) -> bool {
        self.ring.len() == self.cfg.lags * self.cfg.n_targets
    }

    /// The row's design `[x, lags, y]` into `z`: zeros in the lags' place
    /// while the ring is short (such a row is not learned, and its
    /// prediction is not read), and in a target's place where it is
    /// missing (its slot's coefficient is 0 in every equation).
    fn design_into(&self, x: &[f64], y: Option<&[Option<f64>]>, z: &mut Vec<f64>) {
        let (k, p) = (self.cfg.n_targets, self.cfg.lags);
        z.clear();
        z.extend_from_slice(x);
        if self.full() {
            for l in 1..=p {
                let at = (p - l) * k;
                z.extend(self.ring.range(at..at + k));
            }
        } else {
            z.resize(x.len() + k * p, 0.0);
        }
        match y {
            Some(y) => z.extend(y.iter().map(|v| v.unwrap_or(0.0))),
            None => z.resize(z.len() + k, 0.0),
        }
    }

    /// The prediction under `select`, from the chosen coefficients.
    fn chosen_pred(&self, z: &[f64], out: &mut [f64]) {
        let c = &self.last.chosen;
        let w = self.cfg.k_total();
        let i0 = usize::from(self.cfg.fit_intercept);
        for (j, o) in out.iter_mut().enumerate() {
            let row = &c[j * w..(j + 1) * w];
            let base = if i0 == 1 { row[0] } else { 0.0 };
            *o = base + row[i0..].iter().zip(z).map(|(b, v)| b * v).sum::<f64>();
        }
    }

    /// Whether the row's predictions are read: the ring full, the fit
    /// identified and, under `select`, an order chosen.
    fn predicts(&self) -> bool {
        self.full() && self.defined && (self.cfg.select.is_none() || !self.last.chosen.is_empty())
    }

    /// The row's tests and order, from the last solve's read, into `out`.
    fn extras_into(&self, out: &mut Vec<f64>) {
        out.clear();
        let k = self.cfg.n_targets as f64;
        let q = self.order().unwrap_or(0) as f64;
        for (i, t) in self.cfg.causality.iter().enumerate() {
            let lam = self.last.wald.get(i).copied().flatten();
            let (s, p) = match (self.defined, lam, self.last.df, self.last.n_kish) {
                (true, Some(lam), Some(df), Some(n)) => {
                    let restrictions = q * (t.causing.len() * t.caused.len()) as f64;
                    let r = self.cfg.causality_kind.read(lam, restrictions, df, n, k);
                    (r[0], r[1])
                }
                _ => (f64::NAN, f64::NAN),
            };
            out.push(s);
            out.push(p);
        }
        if self.cfg.select.is_some() {
            out.push(self.order().map_or(f64::NAN, |q| q as f64));
        }
    }

    /// Withhold everything the row reports below the floor.
    fn gate(out: &mut Step, floor: f64) {
        if out.n_eff < floor {
            out.pred.fill(f64::NAN);
            if let Some(Extra::VectorAr { values }) = out.extra.as_mut() {
                values.fill(f64::NAN);
            }
        }
    }

    /// The extra a row carries: `None` where the spec asks for nothing
    /// beside the predictions.
    fn extra_slot(&self, out: &mut Step) -> Option<Vec<f64>> {
        if self.n_extra() == 0 {
            out.extra = None;
            return None;
        }
        match out.extra.take() {
            Some(Extra::VectorAr { values }) => Some(values),
            _ => Some(Vec::with_capacity(self.n_extra())),
        }
    }

    /// The row's report, the state untouched.
    fn read(&self, x: &[f64], d_clock: f64, floor: f64, out: &mut Step) {
        let k = self.cfg.n_targets;
        let mut z = Vec::with_capacity(self.cfg.n_regressors() + k);
        self.design_into(x, None, &mut z);
        out.pred.clear();
        out.pred.resize(k, f64::NAN);
        let fit = self.fit.predict(&z, d_clock);
        out.n_eff = fit.n_eff;
        if self.predicts() {
            match self.cfg.select {
                Some(_) => self.chosen_pred(&z, &mut out.pred),
                None => out.pred.copy_from_slice(&fit.pred[..k]),
            }
        }
        out.extra = match self.extra_slot(out) {
            Some(mut v) => {
                self.extras_into(&mut v);
                Some(Extra::VectorAr { values: v })
            }
            None => None,
        };
        Self::gate(out, floor);
    }

    /// Read the fit the solve just made: its identification, its order
    /// under `select`, each test's statistic and the coefficients'
    /// variances. The moments are read in place, a window's into the kept
    /// working space.
    fn refresh(&mut self) {
        let sh = self.cfg.shape();
        let (d, k) = (self.cfg.n_regressors(), self.cfg.n_targets);
        let windowed = self.cfg.window.is_some();
        let work = &mut self.work;
        let read = self.fit.system(0, &mut work.kept).is_some_and(|sys| {
            let w = sys.wj;
            let n = sys.tm.and_then(|tm| tm.n_kish(&[w])[0]);
            match n {
                Some(n) if w > 0.0 && n > 0.0 => {
                    work.mom.read(sys.cov, d, k, self.cfg.fit_intercept, w, n)
                }
                _ => false,
            }
        });
        let last = &mut self.last;
        if !read {
            if windowed {
                self.defined = false;
            }
            *last = LastRead::default();
            return;
        }
        work.factor.of(&work.mom, self.cfg.ridge);
        let top = sh.identified(&work.factor);
        let now = match self.cfg.select {
            None => top == Some(sh.p),
            Some(_) => top.is_some_and(|t| t >= 1),
        };
        self.defined = if windowed { now } else { self.defined || now };
        let order = match (self.cfg.select, top) {
            _ if !self.defined => None,
            (None, Some(t)) if t == sh.p => Some(sh.p),
            (None, _) => None,
            (Some(ic), Some(t)) => {
                fit::choose(ic, &sh, &work.factor, &work.mom, t, &mut work.scratch)
            }
            (Some(_), None) => None,
        };
        last.order = order;
        last.n_kish = Some(work.mom.n_kish);
        last.chosen.clear();
        last.wald.clear();
        last.coef_var.clear();
        let Some(q) = order else {
            last.df = None;
            return;
        };
        let df = fit::df_resid(&sh, &work.mom, q);
        last.df = Some(df);
        let m = sh.m(q);
        if self.cfg.select.is_some() {
            let width = self.cfg.k_total();
            let i0 = usize::from(self.cfg.fit_intercept);
            work.factor.coefficients(m, &mut work.b);
            last.chosen.resize(k * width, 0.0);
            for j in 0..k {
                let row = &mut last.chosen[j * width..(j + 1) * width];
                for i in 0..m {
                    row[i0 + i] = work.b[i * k + j];
                }
                if i0 == 1 {
                    let at: f64 = (0..m).map(|i| work.b[i * k + j] * work.mom.zbar[i]).sum();
                    row[0] = work.mom.ybar[j] - at;
                }
            }
        }
        for t in &self.cfg.causality {
            let v = fit::wald(
                &sh,
                &work.factor,
                &work.mom,
                q,
                &t.caused,
                &t.causing,
                &mut work.scratch,
            );
            last.wald.push((!v.is_nan()).then_some(v));
        }
        if self.keep_factor {
            let mut out = std::mem::take(&mut work.b);
            fit::coef_variances(&sh, &work.factor, &work.mom, q, &mut work.scratch, &mut out);
            last.coef_var
                .extend(out.iter().map(|&v| (!v.is_nan()).then_some(v)));
            work.b = out;
        }
    }

    /// Each target's coefficients, `k_total` each: the constant (with
    /// `fit_intercept`), the features, then the lags lag by lag
    /// (statsmodels' `params` column for the target). `None` before the fit
    /// is identified.
    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        if !self.defined {
            return None;
        }
        let w = self.cfg.k_total();
        if self.cfg.select.is_some() {
            if self.last.chosen.is_empty() {
                return None;
            }
            return Some(self.last.chosen.chunks(w).map(<[f64]>::to_vec).collect());
        }
        let c = self.fit.coefficients()?;
        Some(c.iter().map(|row| row[..w].to_vec()).collect())
    }

    /// The fit as its moments stand, read afresh at the order in force:
    /// what the bank's reads (impulse responses, variance decompositions,
    /// stability, an ad hoc test) are computed from. `Err` with the reason
    /// where it is not identified.
    pub fn fit_now(&self) -> Result<VectorArFit, String> {
        reads::fit_now(self)
    }
}

impl OnlineModel for VectorAr {
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
        let k = self.cfg.n_targets;
        let full = self.full();
        let learns = weight > 0.0 && y.iter().all(Option::is_some);
        let mut z = std::mem::take(&mut self.work.z);
        let mut taught = std::mem::take(&mut self.work.taught);
        let mut one = std::mem::take(&mut self.work.one);
        self.design_into(x, Some(y), &mut z);
        // The report before the row: the last solve's tests and, under
        // `select`, its coefficients, read before this row's solve.
        out.pred.clear();
        out.pred.resize(k, f64::NAN);
        let predicts = self.predicts();
        if predicts && self.cfg.select.is_some() {
            self.chosen_pred(&z, &mut out.pred);
        }
        out.extra = match self.extra_slot(out) {
            Some(mut v) => {
                self.extras_into(&mut v);
                Some(Extra::VectorAr { values: v })
            }
            None => None,
        };
        // Every row reaches the fit, so its weight and its clock are the
        // stream's; only a learned row with its lags teaches it.
        taught.clear();
        taught.extend(y.iter().map(|v| if learns && full { *v } else { None }));
        let before = self.fit.solves();
        self.fit.step_into(&z, &taught, d_clock, weight, &mut one);
        out.n_eff = one.n_eff;
        if predicts && self.cfg.select.is_none() {
            out.pred.copy_from_slice(&one.pred[..k]);
        }
        if self.fit.solves() != before {
            self.refresh();
        }
        if learns {
            if full {
                for _ in 0..k {
                    self.ring.pop_front();
                }
            }
            self.ring.extend(y.iter().map(|v| v.unwrap_or(0.0)));
        }
        self.work.z = z;
        self.work.taught = taught;
        self.work.one = one;
        Self::gate(out, self.cfg.min_weight);
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

    /// The rows behind a break are no longer the series' last values: the
    /// ring empties, and the fit waits for `p` learned rows before it
    /// learns again. The fit is kept.
    fn clear_lags(&mut self) {
        self.ring.clear();
    }

    /// statsmodels' `bse²`, each coefficient's sampling variance with the
    /// noise in it: `(S⁻¹)_ii Σ_u,jj / n` for a slope of target `j`, read at
    /// the last solve. NaN past the order in force and where the degrees of
    /// freedom are not positive; `None` before the fit is identified or
    /// without `set_keep_factor`.
    fn coef_variance(&self) -> Option<crate::CoefVariance> {
        if !self.keep_factor || !self.defined || self.last.coef_var.is_empty() {
            return None;
        }
        let w = self.cfg.k_total();
        Some(crate::CoefVariance::Absolute(
            self.last
                .coef_var
                .chunks(w)
                .map(|row| row.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
                .collect(),
        ))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        // Every target is learned on the same rows: the fit's own weight
        // for each, inside the window under one.
        self.fit.target_n_eff_into(out)
    }

    fn set_solve_share(&mut self, share: Option<f64>) {
        self.cfg.solve_share = share;
        self.fit.set_solve_share(share);
    }

    fn set_first_period_by_share(&mut self, on: bool) {
        self.fit.set_first_period_by_share(on);
    }

    fn solve_share(&self) -> Option<f64> {
        self.cfg.solve_share
    }

    fn stamp_next(&mut self, stamp: crate::Stamp) {
        self.fit.stamp_next(stamp);
    }

    fn stamp_next_on_rows(&mut self, stamp: crate::Stamp) {
        self.fit.stamp_next_on_rows(stamp);
    }

    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        self.fit.set_window_budget(budget);
    }

    fn set_window_closed(&mut self, closed: crate::WindowClosed) {
        self.fit.set_window_closed(closed);
    }

    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        self.fit.window_over_budget()
    }

    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        self.fit.window_shadow()
    }

    fn window_could_refuse(&self, rows: usize) -> bool {
        self.fit.window_could_refuse(rows)
    }

    fn state(&self) -> State {
        State::new(ModelState::VectorAr(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::VectorAr(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("vector_ar", m.cfg.validate())?;
                let (k, w) = (m.cfg.n_targets, m.cfg.k_total());
                let last = &m.last;
                let shaped = m.ring.len() % k == 0
                    && m.ring.len() <= k * m.cfg.lags
                    && *m.fit.cfg() == m.cfg.ridge_cfg()
                    && (last.chosen.is_empty() || last.chosen.len() == k * w)
                    && (last.wald.is_empty() || last.wald.len() == m.cfg.causality.len())
                    && (last.coef_var.is_empty() || last.coef_var.len() == k * w)
                    && last.order.is_none_or(|q| q <= m.cfg.lags);
                if !shaped {
                    return Err(crate::model::damaged(
                        "vector_ar",
                        "its lags or its fit have the wrong shape",
                    ));
                }
                let depth = k * m.cfg.lags;
                m.ring.reserve_exact(depth - m.ring.len());
                let state = State::new(ModelState::EwRidge(Box::new(m.fit.clone())));
                m.fit = EwRidge::restore_owned(state)
                    .map_err(|_| crate::model::damaged("vector_ar", "its fit is damaged"))?;
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "vector_ar",
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
