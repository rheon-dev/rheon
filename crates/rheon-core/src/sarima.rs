//! `Sarima`: ARIMA and seasonal ARIMA with exogenous regressors, at fixed
//! parameters, each target filtered by the exact Kalman filter of its
//! state-space form.
//!
//! # The model
//!
//! statsmodels' `SARIMAX` signs and orders, `order = (p, d, q)` and
//! `seasonal_order = (P, D, Q, s)`:
//!
//! ```text
//! w_t = (1 - B)^d (1 - B^s)^D y_t                      the differenced value
//! xw_t = (1 - B)^d (1 - B^s)^D x_t                     each regressor, likewise
//! phi(B) Phi(B^s) (w_t - xw_t' beta) = theta(B) Theta(B^s) e_t,   e_t ~ (0, sigma2)
//! phi(B)   = 1 - phi_1 B - ... - phi_p B^p      Phi(B^s)   = 1 - Phi_1 B^s - ... - Phi_P B^(sP)
//! theta(B) = 1 + theta_1 B + ... + theta_q B^q  Theta(B^s) = 1 + Theta_1 B^s + ... + Theta_Q B^(sQ)
//! ```
//!
//! `B` steps one **slot**: `spacing` clock units. ARMA is `d = D = 0` with
//! no season; ARIMA, no season. Each target is its own series with its own
//! parameters; the regressors, the row's features, are shared.
//!
//! # The recursion
//!
//! The differencing is statsmodels' simple differencing (`SARIMAX(...,
//! simple_differencing=True)`): `D` seasonal differences, then `d`
//! ordinary ones, over a ring of the last `d + sD` slots. The ARMA part is
//! in Harvey's state-space form with `r = max(p + sP, q + sQ + 1)` states
//! (`T`, `R`, `Z` in `poly::Ssm`), filtered by the conventional Kalman
//! filter from the stationary start, `a_1 = 0` and `P_1` solving `P = T P
//! T' + R sigma2 R'`. Per slot `t`, with `d_t = xw_t' beta`:
//!
//! ```text
//! ŵ_t = d_t + a_0                     out of sample: before y_t is read
//! ŷ_t = ŵ_t - L_t                     L_t: w_t's differencing with y_t at 0
//! F   = P_00,  v = w_t - ŵ_t          only where w_t is observed
//! a  <- a + P_.0 v / F,   P <- P - P_.0 P_0. / F
//! a  <- T a,              P <- T P T' + R sigma2 R',  made symmetric
//! ```
//!
//! `ŷ_t` is the row's `pred`. `w_t` is observed when the slot and every
//! slot its differencing reads were observed; otherwise the update is
//! skipped and the filter predicts through the slot, as `SARIMAX` does
//! with a `NaN`.
//!
//! # Rows, weights and the clock
//!
//! - A row of weight 0, or with its target null, keeps its slot and learns
//!   nothing: the slot is not observed. Its value in the ring is its own
//!   prediction `ŷ_t`, so the rows after it still predict, and seasonal
//!   positions stay aligned. A row of weight 0 keeps nothing of its own,
//!   its features included: with regressors, its value in the ring is NaN,
//!   as a skipped slot's, and the rows whose differencing reads it report
//!   none.
//! - A clock step of `n` slots (`d_clock / spacing`, rounded) leaves the
//!   `n - 1` slots between unobserved, predicted through the same way. A
//!   skipped slot has no features, so with regressors its value is NaN,
//!   and the rows whose differencing reads it report none.
//! - A break ([`OnlineModel::clear_lags`]: a session change, a gap past
//!   `gap_cap`), or a step of more than [`MAX_GAP_SLOTS`] slots, starts
//!   each series again: the ring empties and the state returns to its
//!   stationary start. The weights are kept.
//! - `n_eff` is the weight of every row before this one, with no decay;
//!   each target's prediction is withheld until the weight of the rows
//!   that observed it reaches `min_weight`. With fixed parameters a row's
//!   weight enters these sums only: the filter conditions on an observed
//!   value whatever it weighs.
//!
//! # Estimated parameters
//!
//! With `estimate` set and no `params`, each target's parameters are
//! estimated from its own rows: its slots past the differencing's fill
//! are buffered until a start is fitted and accepted, then the buffer is
//! replayed at the fit and dropped, and each learned row steps the
//! parameters by recursive prediction error (the equations are in the
//! `est` module). A target reports nothing until its start is accepted,
//! and [`Sarima::wait`] names why: the differencing filling; Kish's count
//! of the buffer, `(sum w)^2 / sum w^2`, not past the structural minimum
//! (every parameter, `sigma2` among them, plus `d + sD`); or a fit that
//! sits on a stationarity or invertibility bound, has not converged, or
//! has a standard error past `se_bound`. `min_weight` is checked on top of
//! that, against the target's own weight, as at given parameters. A break
//! fits the start again from a new buffer. The prediction is the
//! conditional recursion's, `ŵ_t = xw_t' beta - sum_i A_i z_{t-i} + sum_j
//! C_j e_{t-j}` with `z = w - xw' beta`, not the Kalman filter's: its state
//! is linear in the season where the filter's covariance is quadratic.
//!
//! # The state
//!
//! Per target, the ring (`d + sD` values and flags), `a` (`r`) and `P`'s
//! upper triangle (`r (r + 1) / 2`; it is symmetric after every step); the
//! regressors' rings, `d + sD` each. A row steps `P` in full in a working
//! matrix of `r^2` doubles that every target shares and the state does not
//! hold. Bounded by the orders,
//! never by the stream: [`Sarima::state_bytes`], capped by
//! `max_state_bytes` when the model is built.
//!
//! Where the parameters are estimated there is no filter. Per target, the
//! ring, and the run: `9 (n + 2 + n^2 + L (6 + 2k))` bytes for
//! `n = k + p + q + P + Q` parameters and rings of
//! `L = max(p + sP, q + sQ, 1)` lags (at `(1,1,1)(1,1,1,s)` without
//! regressors, `9 (22 + 6 (s + 1))`: 1,548 bytes at `s = 24`, 9,324 at
//! `s = 168`); and until its start is accepted its buffer, `9 (2 + k)`
//! bytes a slot, checked against `max_state_bytes` as it grows (a warning
//! past half of it, and the slot that would pass it refused, after which
//! nothing more is buffered).

mod est;
mod poly;
mod series;
mod transform;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
pub use est::SarimaWait;
use poly::Ssm;
use series::{Diff, Series, Work};

/// A clock step past this many slots starts each series again rather than
/// predicting through every slot between, whose cost is one covariance
/// step (`O(r^2)`) a slot.
pub const MAX_GAP_SLOTS: usize = 4096;

/// One target's parameters, in statsmodels' `SARIMAX` order and signs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarimaParams {
    /// `beta`, one per feature.
    pub exog: Vec<f64>,
    /// `phi_1, ..., phi_p`.
    pub ar: Vec<f64>,
    /// `theta_1, ..., theta_q`.
    pub ma: Vec<f64>,
    /// `Phi_1, ..., Phi_P`.
    pub seasonal_ar: Vec<f64>,
    /// `Theta_1, ..., Theta_Q`.
    pub seasonal_ma: Vec<f64>,
    /// The innovations' variance.
    pub sigma2: f64,
}

impl SarimaParams {
    /// From statsmodels' flat vector, `SARIMAX(...).param_names`' order:
    /// the regressors' coefficients, `ar`, `ma`, `ar.S`, `ma.S`, `sigma2`.
    /// `None` when its length is not the orders'.
    pub fn from_flat(
        flat: &[f64],
        n_features: usize,
        order: [usize; 3],
        seasonal_order: [usize; 4],
    ) -> Option<Self> {
        let [p, _, q] = order;
        let [sp, _, sq, _] = seasonal_order;
        let lens = [n_features, p, q, sp, sq];
        if flat.len() != lens.iter().sum::<usize>() + 1 {
            return None;
        }
        let mut at = 0;
        let mut take = |n: usize| {
            let v = flat[at..at + n].to_vec();
            at += n;
            v
        };
        Some(Self {
            exog: take(lens[0]),
            ar: take(lens[1]),
            ma: take(lens[2]),
            seasonal_ar: take(lens[3]),
            seasonal_ma: take(lens[4]),
            sigma2: flat[flat.len() - 1],
        })
    }

    fn all(&self) -> impl Iterator<Item = &f64> {
        self.exog
            .iter()
            .chain(&self.ar)
            .chain(&self.ma)
            .chain(&self.seasonal_ar)
            .chain(&self.seasonal_ma)
            .chain(std::iter::once(&self.sigma2))
    }
}

/// The settings of a [`Sarima`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SarimaCfg {
    /// The series, one per target.
    pub n_targets: usize,
    /// Exogenous regressors, the row's features, shared by every target.
    pub n_features: usize,
    /// `(p, d, q)`.
    pub order: [usize; 3],
    /// `(P, D, Q, s)`; `s` is read only where `P`, `D` or `Q` is above 0.
    pub seasonal_order: [usize; 4],
    /// Each target's parameters.
    pub params: Vec<SarimaParams>,
    /// Clock units a slot: `B` steps this much of the clock.
    pub spacing: f64,
    /// A target's prediction is withheld until the weight of the rows that
    /// observed it reaches this.
    pub min_weight: f64,
    /// The bytes a group's state may hold; `None` is
    /// [`crate::DEFAULT_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// Estimate each target's parameters (`params` then empty), rather
    /// than filter at given ones.
    #[serde(default)]
    pub estimate: Option<SarimaEstimate>,
}

/// How the parameters are estimated: whether the autoregressive factors
/// are held stationary and the moving-average ones invertible, through
/// `SARIMAX`'s transforms (`enforce_stationarity`,
/// `enforce_invertibility`, both on by default there and here), and the
/// largest standard error a start is accepted with (`se_bound`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SarimaEstimate {
    pub enforce_stationarity: bool,
    pub enforce_invertibility: bool,
    /// The start's fit is accepted only with every standard error at most
    /// this, each in units free of the target's: an ARMA coefficient's as
    /// it is (the coefficients are unit-free), a feature's coefficient's
    /// times the feature's standard deviation over the innovations',
    /// and `sigma2`'s relative to `sigma2`. The regressors' check is a
    /// rule of its own: identical or near-collinear regressors leave the
    /// polynomials identified, so they pass the ARMA check, while the split
    /// of their effect between them is not, and a start read from such a
    /// split steps from an arbitrary point on a flat ridge. Measured with
    /// two regressors correlated 0.99995, the polynomial's standard error
    /// was 0.044 and `sigma2`'s relative one 0.083 at 365 rows, the
    /// regressors' standardized ones 3.7 there and 2.1 at 1,233, so the
    /// start was withheld; a pair correlated 0.5 was accepted at 366 rows.
    /// Default [`SarimaEstimate::SE_BOUND`], 0.1; finite and above 0.
    /// Smaller waits longer for a better start. Written to a state only
    /// where it is not the default.
    #[serde(
        default = "SarimaEstimate::default_se_bound",
        skip_serializing_if = "SarimaEstimate::is_default_se_bound"
    )]
    pub se_bound: f64,
}

impl SarimaEstimate {
    /// `se_bound`'s default. It decides how good a start the recursion
    /// steps from, and so where it ends: measured over five specs and
    /// eight seeds each, 10,000 rows, at 0.1 every start's worst
    /// coefficient ends within 0.4 of its standard errors of `SARIMAX`'s
    /// batch fit of the same rows; at 0.15 and 0.2 a spec whose seasonal
    /// factors nearly cancel ends 4.5 and 8.2 off (the table is at
    /// `est::SE_BOUND`'s measurement).
    pub const SE_BOUND: f64 = est::SE_BOUND;

    fn default_se_bound() -> f64 {
        Self::SE_BOUND
    }

    // serde's `skip_serializing_if` takes a reference.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_default_se_bound(v: &f64) -> bool {
        v.to_bits() == Self::SE_BOUND.to_bits()
    }
}

impl Default for SarimaEstimate {
    fn default() -> Self {
        Self {
            enforce_stationarity: true,
            enforce_invertibility: true,
            se_bound: Self::SE_BOUND,
        }
    }
}

impl SarimaCfg {
    /// The estimator's layout of the parameters; `None` at given ones.
    fn shape(&self) -> Option<est::Shape> {
        let e = self.estimate?;
        let [p, _, q] = self.order;
        let [sp, _, sq, _] = self.seasonal_order;
        Some(est::Shape {
            k: self.n_features,
            p,
            q,
            sp,
            sq,
            s: self.s(),
            stationary: e.enforce_stationarity,
            invertible: e.enforce_invertibility,
        })
    }

    /// The structural minimum Kish's count must pass before a start is
    /// fitted: every parameter estimated, `sigma2` with them, and the
    /// differencing's `d + sD`.
    pub fn structural_minimum(&self) -> f64 {
        let [p, _, q] = self.order;
        let [sp, _, sq, _] = self.seasonal_order;
        (self.n_features + p + q + sp + sq + 1) as f64 + self.depth() as f64
    }

    fn seasonal(&self) -> bool {
        let [sp, sd, sq, _] = self.seasonal_order;
        sp + sd + sq > 0
    }

    /// The season, 0 without one.
    fn s(&self) -> usize {
        if self.seasonal() {
            self.seasonal_order[3]
        } else {
            0
        }
    }

    fn diff(&self) -> Diff {
        Diff {
            d: self.order[1],
            big_d: self.seasonal_order[1],
            s: self.s(),
        }
    }

    /// The slots the differencing reads before the current one, `d + sD`.
    pub fn depth(&self) -> u64 {
        let s = self.s() as u64;
        (self.order[1] as u64).saturating_add((self.seasonal_order[1] as u64).saturating_mul(s))
    }

    /// The filter's states, `max(p + sP, q + sQ + 1)`.
    pub fn states(&self) -> u64 {
        let s = self.s() as u64;
        let [p, _, q] = self.order.map(|v| v as u64);
        let [sp, _, sq, _] = self.seasonal_order.map(|v| v as u64);
        let ar = p.saturating_add(s.saturating_mul(sp));
        let ma = q.saturating_add(s.saturating_mul(sq));
        ar.max(ma.saturating_add(1))
    }

    /// The cap on the state's bytes.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "sarima: the model needs at least one target, the series it filters, but it \
                 has none. Name at least one column in targets.",
            );
        }
        if self.estimate.is_some() && !self.params.is_empty() {
            no.refuse(format!(
                "sarima: params must be empty when the parameters are estimated, but it gives \
                 {} sets. Give no params, or switch estimation off to filter at them.",
                self.params.len()
            ));
        }
        if self.estimate.is_none() && self.params.len() != self.n_targets {
            no.refuse(format!(
                "sarima: params must give one set of parameters a target, {}, but it gives {}. \
                 Give one set for each target.",
                self.n_targets,
                self.params.len()
            ));
        }
        let [p, d, q] = self.order;
        let [sp, sd, sq, s] = self.seasonal_order;
        for (name, v) in [
            ("p", p),
            ("d", d),
            ("q", q),
            ("P", sp),
            ("D", sd),
            ("Q", sq),
            ("s", s),
        ] {
            no.take(crate::check_lag_ceiling(name, v).map_err(|e| format!("sarima: {e}")));
        }
        if self.seasonal() && s < 2 {
            no.refuse(format!(
                "sarima: a seasonal order needs a season s of at least 2 slots, but s is {s}. \
                 Set s to the slots in a season, or the seasonal orders P, D and Q to 0."
            ));
        }
        if !(self.spacing.is_finite() && self.spacing > 0.0) {
            no.refuse(format!(
                "sarima: spacing must be finite and above 0, but it is {}. Set spacing to the \
                 clock units between two slots, 1 on a clock that counts rows.",
                self.spacing
            ));
        }
        if let Some(e) = self.estimate
            && !(e.se_bound.is_finite() && e.se_bound > 0.0)
        {
            no.refuse(format!(
                "sarima: se_bound must be finite and above 0, but it is {}. Set it to the \
                 largest standard error a start is accepted with, 0.1 by default.",
                e.se_bound
            ));
        }
        no.take(crate::model::check_min_weight("sarima", self.min_weight));
        no.take(crate::clock_span::check_max_state_bytes(
            "sarima",
            self.max_state_bytes,
        ));
        no.stop_here()?;
        for (j, prm) in self.params.iter().enumerate() {
            let want = [self.n_features, p, q, sp, sq];
            let got = [
                prm.exog.len(),
                prm.ar.len(),
                prm.ma.len(),
                prm.seasonal_ar.len(),
                prm.seasonal_ma.len(),
            ];
            if want != got {
                no.refuse(format!(
                    "sarima: target {j}'s parameters must hold {} feature, {p} ar, {q} ma, \
                     {sp} seasonal ar and {sq} seasonal ma coefficients, but they hold {}, \
                     {}, {}, {} and {}. Give one coefficient a lag of each polynomial and one \
                     a feature.",
                    self.n_features, got[0], got[1], got[2], got[3], got[4]
                ));
                continue;
            }
            if !prm.all().all(|v| v.is_finite()) {
                no.refuse(format!(
                    "sarima: target {j}'s parameters must be finite numbers, but one is not. \
                     Give finite values."
                ));
                continue;
            }
            if prm.sigma2 <= 0.0 {
                no.refuse(format!(
                    "sarima: target {j}'s sigma2 must be above 0, but it is {}. Give the \
                     innovations' variance.",
                    prm.sigma2
                ));
            }
            if !poly::stationary(&prm.ar) {
                no.refuse(format!(
                    "sarima: target {j}'s ar coefficients {:?} are not stationary: their \
                     polynomial 1 - phi_1 B - ... has a root on or inside the unit circle, so \
                     the series has no stationary start to filter from. Give stationary \
                     coefficients, or difference the series with d.",
                    prm.ar
                ));
            }
            if !poly::stationary(&prm.seasonal_ar) {
                no.refuse(format!(
                    "sarima: target {j}'s seasonal ar coefficients {:?} are not stationary: \
                     their seasonal autoregressive polynomial has a root on or inside the unit \
                     circle, so the series has no stationary start to filter from. Give \
                     stationary coefficients, or difference the series with D.",
                    prm.seasonal_ar
                ));
            }
        }
        no.stop_here()?;
        let bytes = Sarima::state_bytes(self);
        if bytes > self.cap() {
            let what = if self.estimate.is_some() {
                format!(
                    "the estimator's rings of {} lags",
                    self.shape().map_or(0, |s| s.lags())
                )
            } else {
                format!("{} states", self.states())
            };
            no.refuse(format!(
                "sarima: the state of {} targets with {what} and {} slots of differencing \
                 would hold {bytes} bytes, past max_state_bytes of {}. Use a shorter season or \
                 lower orders, or raise max_state_bytes.",
                self.n_targets,
                self.depth(),
                self.cap()
            ));
        }
        no.finish()
    }

    /// Each target's state-space matrices.
    fn ssms(&self) -> Vec<Ssm> {
        let s = self.s();
        self.params
            .iter()
            .map(|p| Ssm::new(&p.ar, &p.ma, &p.seasonal_ar, &p.seasonal_ma, s, p.sigma2))
            .collect()
    }
}

/// A row's working values, rebuilt each slot: not state, so a clone
/// starts empty and two are always equal.
#[derive(Debug, Default)]
struct Scratch(Vec<f64>);

impl Clone for Scratch {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// ARIMA and SARIMA(X) at fixed parameters, per target; see the module
/// docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sarima {
    cfg: SarimaCfg,
    /// One per target.
    series: Vec<Series>,
    /// Per regressor, its last `d + sD` slots, oldest first (NaN for a
    /// slot the clock skipped).
    #[serde(with = "crate::humanfloat::vec_deque_f64_or_tag")]
    xs: Vec<VecDeque<f64>>,
    /// The weight of every row before the next (hard rule 8).
    n_eff: f64,
    /// Each target's estimator, where the parameters are estimated; empty
    /// at given ones.
    #[serde(default)]
    est: Vec<est::Est>,
    /// The bytes a buffered slot would have taken the state to, past the
    /// cap: from then on nothing is buffered.
    #[serde(default)]
    over: Option<u64>,
    #[serde(default)]
    warned: bool,
    #[serde(default)]
    warning: Option<u64>,
    /// Each target's state-space matrices, from `cfg`: not state.
    #[serde(skip)]
    ssm: Vec<Ssm>,
    /// The slot's differenced regressors, kept so a row allocates
    /// nothing: not state.
    #[serde(skip)]
    xw: Scratch,
    #[serde(skip)]
    work: Work,
}

impl Sarima {
    pub fn new(cfg: SarimaCfg) -> Result<Self, String> {
        cfg.validate()?;
        let ssm = cfg.ssms();
        let depth = cfg.depth() as usize;
        let mut series = Vec::with_capacity(cfg.n_targets);
        let est = if cfg.estimate.is_some() {
            series.extend((0..cfg.n_targets).map(|_| Series::new(depth, 0, &[])));
            (0..cfg.n_targets).map(|_| est::Est::new()).collect()
        } else {
            Vec::new()
        };
        for (j, m) in ssm.iter().enumerate() {
            let p0 = m.stationary_cov().ok_or_else(|| {
                format!(
                    "sarima: target {j}'s state has no stationary covariance that the doubling \
                     reached; its ar coefficients are too close to a unit root. Give \
                     coefficients further inside the stationary region."
                )
            })?;
            series.push(Series::new(depth, m.r(), &p0));
        }
        Ok(Self {
            xs: (0..cfg.n_features)
                .map(|_| VecDeque::with_capacity(depth))
                .collect(),
            series,
            n_eff: 0.0,
            est,
            over: None,
            warned: false,
            warning: None,
            xw: Scratch(Vec::with_capacity(cfg.n_features)),
            work: Work::default(),
            ssm,
            cfg,
        })
    }

    pub fn cfg(&self) -> &SarimaCfg {
        &self.cfg
    }

    /// The weight of every row before the next (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// The bytes the state holds at most, by its settings, as msgpack
    /// writes it (9 bytes a double): per target `9 (r (r + 1) / 2 + r +
    /// depth) + depth` for `P`'s triangle, `a`, the ring and its flags,
    /// and its parameters; `9 depth` a regressor; and the names and
    /// settings around them.
    ///
    /// Where the parameters are estimated there is no filter (`r` is 0 and
    /// no parameters are given); a target holds instead its run,
    /// `9 (n + 2 + n^2 + L (6 + 2k))` for `n` parameters stepped and rings
    /// of `L = max(p + sP, q + sQ, 1)` lags, and its buffer until the start
    /// is accepted, [`Self::buffer_bytes`] a slot, which is checked as it
    /// grows ([`Self::bytes`]).
    pub fn state_bytes(cfg: &SarimaCfg) -> u64 {
        let estimated = cfg.shape();
        let (r, depth) = match estimated {
            Some(_) => (0, cfg.depth()),
            None => (cfg.states(), cfg.depth()),
        };
        let [p, _, q] = cfg.order.map(|v| v as u64);
        let [sp, _, sq, _] = cfg.seasonal_order.map(|v| v as u64);
        let k = cfg.n_features as u64;
        let params = match estimated {
            Some(shape) => (est::Run::len(&shape) as u64).saturating_mul(9) + 160,
            None => 9 * (k + p + q + sp + sq + 1) + 96,
        };
        let per = r
            .saturating_mul(r.saturating_add(1))
            .saturating_div(2)
            .saturating_add(r)
            .saturating_add(depth)
            .saturating_mul(9)
            .saturating_add(depth)
            .saturating_add(params)
            .saturating_add(64);
        per.saturating_mul(cfg.n_targets as u64)
            .saturating_add(depth.saturating_mul(9).saturating_add(8).saturating_mul(k))
            .saturating_add(320)
    }

    /// The bytes one buffered slot adds to a target's state: its
    /// differenced value, its weight and its `k` differenced regressors.
    pub fn buffer_bytes(cfg: &SarimaCfg) -> u64 {
        9 * (2 + cfg.n_features as u64)
    }

    /// The bytes the state holds as it stands: [`Self::state_bytes`] and
    /// each target's buffer.
    pub fn bytes(&self) -> u64 {
        let slots: usize = self.est.iter().map(|e| e.buf.w.len()).sum();
        Self::state_bytes(&self.cfg)
            .saturating_add((slots as u64).saturating_mul(Self::buffer_bytes(&self.cfg)))
    }

    /// Why target `j` reports nothing, where its parameters are estimated
    /// and its start has not been accepted; `None` once it has, and at
    /// given parameters.
    pub fn wait(&self, j: usize) -> Option<SarimaWait> {
        let e = self.est.get(j)?;
        e.run.is_none().then_some(e.wait)
    }

    /// Target `j`'s parameters as they stand: the given ones, or the
    /// estimator's once its start is accepted (`None` before).
    pub fn params(&self, j: usize) -> Option<SarimaParams> {
        let Some(shape) = self.cfg.shape() else {
            return self.cfg.params.get(j).cloned();
        };
        let run = self.est.get(j)?.run.as_ref()?;
        let par = run.params(&shape);
        let flat: Vec<f64> = par.iter().copied().chain([run.sigma2]).collect();
        SarimaParams::from_flat(&flat, shape.k, self.cfg.order, self.cfg.seasonal_order)
    }

    /// Each target's parameters in statsmodels' order, `beta`, `phi`,
    /// `theta`, `Phi`, `Theta`, `sigma2`: the given ones, or the
    /// estimator's; NaN where a target's start is not accepted yet.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        let [p, _, q] = self.cfg.order;
        let [sp, _, sq, _] = self.cfg.seasonal_order;
        let n = self.cfg.n_features + p + q + sp + sq + 1;
        (0..self.cfg.n_targets)
            .map(|j| match self.params(j) {
                Some(prm) => prm.all().copied().collect(),
                None => vec![f64::NAN; n],
            })
            .collect()
    }

    /// Whether some target's start is still filling its differencing or
    /// buffering below the structural minimum.
    pub fn starting(&self) -> bool {
        (0..self.est.len()).any(|j| {
            matches!(
                self.wait(j),
                Some(SarimaWait::Filling | SarimaWait::Buffering)
            )
        })
    }

    /// Whether some target's start was fitted and withheld: on a bound,
    /// not converged, or a standard error past `se_bound`.
    pub fn unidentified(&self) -> bool {
        (0..self.est.len()).any(|j| {
            matches!(
                self.wait(j),
                Some(SarimaWait::NotConverged | SarimaWait::OnBound | SarimaWait::NotIdentified)
            )
        })
    }

    /// Kish's count of target `j`'s buffer, where it is buffering.
    pub fn buffered_kish(&self, j: usize) -> Option<f64> {
        let e = self.est.get(j)?;
        e.run.is_none().then(|| e.kish())
    }

    /// The slots a clock step of `d_clock` leaves unobserved before the
    /// row's own: `round(d_clock / spacing) - 1`, none for a step under a
    /// slot (the first row's, or a row on the slot of the one before).
    /// `None` past [`MAX_GAP_SLOTS`].
    fn missing_slots(&self, d_clock: f64) -> Option<usize> {
        let n = (d_clock / self.cfg.spacing).round();
        if n.is_nan() || n < 1.0 {
            return Some(0);
        }
        if n - 1.0 > MAX_GAP_SLOTS as f64 {
            return None;
        }
        Some(n as usize - 1)
    }

    /// Every series and every regressor's ring back to the start; with
    /// `cov`, each covariance to its stationary start too.
    fn restart(&mut self, cov: bool) {
        for ring in &mut self.xs {
            ring.clear();
        }
        for (series, ssm) in self.series.iter_mut().zip(&self.ssm) {
            let p0 = if cov { ssm.stationary_cov() } else { None };
            series.restart(p0.as_deref());
        }
        if self.cfg.estimate.is_some() {
            // The start is fitted again from a new buffer.
            for (series, e) in self.series.iter_mut().zip(&mut self.est) {
                series.restart(None);
                *e = est::Est::new();
            }
        }
    }

    /// One slot of target `j` where the parameters are estimated: its
    /// prediction `ŷ_t`, then the slot buffered or stepped. `y` is the
    /// slot's value where observed with a weight above 0, `xw` the slot's
    /// differenced regressors where known, `keep` whether the state keeps
    /// them (not for a row of weight 0, which keeps nothing of its own),
    /// `fill` whether a slot not observed keeps its prediction in the ring.
    fn est_slot(
        &mut self,
        j: usize,
        y: Option<f64>,
        weight: f64,
        xw: Option<&[f64]>,
        keep: bool,
        fill: bool,
    ) -> f64 {
        let kept_xw = xw.filter(|_| keep);
        let Some(shape) = self.cfg.shape() else {
            return f64::NAN;
        };
        let diff = self.cfg.diff();
        let depth = diff.depth();
        let floor = self.cfg.structural_minimum();
        let bound = self.cfg.estimate.map_or(est::SE_BOUND, |e| e.se_bound);
        let series = &mut self.series[j];
        if series.ys.len() < depth {
            series.push(y.unwrap_or(f64::NAN), y.is_some(), depth);
            return f64::NAN;
        }
        let lpart = diff.of(&series.ys, 0.0, &mut self.work);
        let w = y.and_then(|v| {
            let w = diff.of_seen(&series.ys, &series.seen, v, &mut self.work);
            (!w.is_nan()).then_some(w)
        });
        let y_hat = match self.est[j].run.as_mut() {
            Some(run) => run.slot(&shape, w, weight, xw, kept_xw) - lpart,
            None => {
                let with = self.bytes().saturating_add(Self::buffer_bytes(&self.cfg));
                let cap = self.cfg.cap();
                let mut held = crate::clock_span::HeldCap {
                    over: &mut self.over,
                    warned: &mut self.warned,
                    warning: &mut self.warning,
                };
                if held.admit(with, cap) {
                    held.note(with, cap);
                    self.est[j].buffer(&shape, w, weight, kept_xw, floor, bound);
                }
                f64::NAN
            }
        };
        let kept = if fill { y_hat } else { f64::NAN };
        self.series[j].push(y.unwrap_or(kept), y.is_some(), depth);
        y_hat
    }

    /// The slot's differenced regressors, where every one is known.
    fn known_xw(&self) -> Option<Vec<f64>> {
        self.xw
            .0
            .iter()
            .all(|v| v.is_finite())
            .then(|| self.xw.0.clone())
    }

    /// The slot's differenced regressors into `xw`: NaN where their rings
    /// are short or `x` is a skipped slot's (`None`).
    fn exog_into(&mut self, x: Option<&[f64]>) {
        let diff = self.cfg.diff();
        let depth = diff.depth();
        self.xw.0.clear();
        for (c, ring) in self.xs.iter().enumerate() {
            let v = match x {
                Some(x) if ring.len() == depth => diff.of(ring, x[c], &mut self.work),
                _ => f64::NAN,
            };
            self.xw.0.push(v);
        }
    }

    fn push_x(&mut self, x: Option<&[f64]>) {
        let depth = self.cfg.depth() as usize;
        if depth == 0 {
            return;
        }
        for (c, ring) in self.xs.iter_mut().enumerate() {
            if ring.len() == depth {
                ring.pop_front();
            }
            ring.push_back(x.map_or(f64::NAN, |x| x[c]));
        }
    }

    /// `x_t' beta` for target `j`, from `xw`.
    fn exog_part(&self, j: usize) -> f64 {
        let mut s = 0.0;
        for (x, b) in self.xw.0.iter().zip(&self.cfg.params[j].exog) {
            s += x * b;
        }
        s
    }

    /// A slot the clock skipped: nothing observed, no features.
    fn skip_slot(&mut self, cov: bool) {
        let diff = self.cfg.diff();
        self.exog_into(None);
        if self.cfg.estimate.is_some() {
            let xw = self.known_xw();
            for j in 0..self.series.len() {
                self.est_slot(j, None, 0.0, xw.as_deref(), false, true);
            }
            self.push_x(None);
            return;
        }
        for j in 0..self.series.len() {
            let d_t = self.exog_part(j);
            self.series[j].slot(&self.ssm[j], diff, None, d_t, true, cov, &mut self.work);
        }
        self.push_x(None);
    }

    /// One row: the slots its clock step skipped, then its own, its
    /// predictions gated at `floor`. With `cov` false only the means move,
    /// which is all a prediction reads.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        floor: f64,
        cov: bool,
        out: &mut Step,
    ) {
        let m = self.cfg.n_targets;
        out.pred.clear();
        out.pred.resize(m, f64::NAN);
        out.extra = None;
        out.n_eff = self.n_eff;
        match self.missing_slots(d_clock) {
            Some(k) => {
                for _ in 0..k {
                    self.skip_slot(cov);
                }
            }
            None => self.restart(cov),
        }
        let diff = self.cfg.diff();
        self.exog_into(Some(x));
        let learns = weight > 0.0;
        // A row of weight 0 keeps nothing of its own: not its regressors,
        // and so not a prediction formed from them in its value's place.
        let kept = learns || self.cfg.n_features == 0;
        if self.cfg.estimate.is_some() {
            let xw = self.known_xw();
            for (j, target) in y.iter().enumerate() {
                let yj = target.filter(|_| learns && self.over.is_none());
                let y_hat = self.est_slot(j, yj, weight, xw.as_deref(), learns, kept);
                if self.series[j].weight >= floor {
                    out.pred[j] = y_hat;
                }
                if yj.is_some() {
                    self.series[j].weight += weight;
                }
            }
            self.push_x(learns.then_some(x));
            self.n_eff += weight;
            return;
        }
        for (j, target) in y.iter().enumerate() {
            let yj = target.filter(|_| learns);
            let d_t = self.exog_part(j);
            let slot = self.series[j].slot(&self.ssm[j], diff, yj, d_t, kept, cov, &mut self.work);
            if self.series[j].weight >= floor {
                out.pred[j] = slot.y_hat;
            }
            if yj.is_some() {
                self.series[j].weight += weight;
            }
        }
        self.push_x(learns.then_some(x));
        self.n_eff += weight;
    }

    /// The row's report, the state untouched, gated at `floor`.
    fn read(&self, x: &[f64], d_clock: f64, floor: f64) -> Step {
        let mut twin = self.clone();
        let mut out = Step::default();
        let none = vec![None; self.cfg.n_targets];
        twin.advance(x, &none, d_clock, 0.0, floor, false, &mut out);
        out
    }

    /// One row of target `j` for the filter alone, outside a stream:
    /// `(ŵ_t, ŷ_t)` for a slot after `skipped` unobserved ones, `y` `None`
    /// for one not observed. What the comparison with statsmodels' filter
    /// reads.
    /// At estimated parameters there is no filter: every value is NaN.
    pub fn filter_row(&mut self, x: &[f64], y: &[Option<f64>], skipped: usize) -> Vec<(f64, f64)> {
        if self.cfg.estimate.is_some() {
            return vec![(f64::NAN, f64::NAN); y.len()];
        }
        for _ in 0..skipped {
            self.skip_slot(true);
        }
        let diff = self.cfg.diff();
        self.exog_into(Some(x));
        let mut out = Vec::with_capacity(y.len());
        for (j, &yj) in y.iter().enumerate() {
            let d_t = self.exog_part(j);
            let s = self.series[j].slot(&self.ssm[j], diff, yj, d_t, true, true, &mut self.work);
            out.push((s.w_hat, s.y_hat));
        }
        self.push_x(Some(x));
        out
    }
}

impl OnlineModel for Sarima {
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
        let floor = self.cfg.min_weight;
        self.advance(x, y, d_clock, weight, floor, true, out);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        self.read(x, d_clock, self.cfg.min_weight)
    }

    fn predict_with_floor(
        &mut self,
        x: &[f64],
        _y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        self.read(x, d_clock, floor)
    }

    /// The rows behind a break are no longer the slots before the next:
    /// each series starts again, its ring empty and its state at the
    /// stationary start. The weights are kept.
    fn clear_lags(&mut self) {
        self.restart(true);
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.series.iter().map(|s| s.weight));
        true
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state(&self) -> State {
        State::new(ModelState::Sarima(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Sarima(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("sarima", m.cfg.validate())?;
                let depth = m.cfg.depth() as usize;
                let shape = m.cfg.shape();
                let r = if shape.is_some() {
                    0
                } else {
                    m.cfg.states() as usize
                };
                let k = m.cfg.n_features;
                let estimators = match shape {
                    Some(shape) => {
                        m.est.len() == m.cfg.n_targets
                            && m.est.iter().all(|e| {
                                e.buf.weight.len() == e.buf.w.len()
                                    && e.buf.xw.len() == k * e.buf.w.len()
                                    && e.run.as_ref().is_none_or(|r| r.shaped(&shape))
                            })
                    }
                    None => m.est.is_empty(),
                };
                let shaped = estimators
                    && m.series.len() == m.cfg.n_targets
                    && m.xs.len() == m.cfg.n_features
                    && m.series.iter().all(|s| {
                        s.ys.len() == s.seen.len()
                            && s.ys.len() <= depth
                            && s.a.len() == r
                            && s.p.len() == r * (r + 1) / 2
                    })
                    && m.xs
                        .iter()
                        .all(|x| m.series.first().is_some_and(|s| x.len() == s.ys.len()));
                if !shaped {
                    return Err(crate::model::damaged(
                        "sarima",
                        "its rings or its filters have the wrong shape",
                    ));
                }
                m.ssm = m.cfg.ssms();
                m.xw = Scratch(Vec::with_capacity(m.cfg.n_features));
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "sarima",
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
mod est_tests;
#[cfg(test)]
mod tests;
