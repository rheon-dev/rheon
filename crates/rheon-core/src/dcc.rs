//! `dcc`: dynamic conditional correlation (Engle 2002) and its asymmetric
//! form (Cappiello, Engle and Sheppard 2006), a full `k × k` correlation
//! matrix of `k` return series that moves every row, with its parameters
//! given or estimated online by recursive quasi-likelihood.
//!
//! **Two stages.** Each series' variance `s_{i,t}` comes from its own
//! GARCH margin (the `garch` model's, one per target, its start and
//! estimator unchanged), and its standardized residual `z_{i,t} = e_{i,t}
//! /√s_{i,t}` is read with the variance of the row before the row is
//! learned. With `margins` absent the targets are taken as standardized
//! residuals already, for margins fitted elsewhere.
//!
//! **The recursion**, with `n_t = z_t ⊙ 1[z_t < 0]` (the negative parts):
//!
//! ```text
//! Q_{t+1} = (1 − a − b) Q̄ − g N̄ + a z_t z_tᵀ + g n_t n_tᵀ + b Q_t
//! R_t     = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}
//! ```
//!
//! `g = 0` is DCC(1,1) (`asymmetric` false, `g` absent); `R_t` is the
//! correlation reported for row `t`, from the rows before it (hard rule 2).
//! The targets are the weighted means of the rows learned, `Q̄` of `z zᵀ`
//! and `N̄` of `n nᵀ`, on the stage's weight `W` (below): `Q̄ ← Q̄ + (w/W)(z
//! zᵀ − Q̄)` after each learned row, before the next `Q` is formed.
//!
//! **The admissible set** is `a, b, g ≥ 0` and `a + b + δ g ≤ 1 − 10⁻⁴`,
//! `δ` the largest eigenvalue of `Q̄^{−1/2} N̄ Q̄^{−1/2}`: exactly where the
//! intercept `(1 − a − b) Q̄ − g N̄` is positive semi-definite, so every
//! `Q_t` is positive definite. It is also where a covariance-stationary
//! `Q_t` exists, given innovations with a finite fourth moment: the moment
//! recursions' matrices are `b I` and `b² I`, so `δ` enters only through
//! positivity. A Student-t margin's `ν ≤ 4` has no fourth moment, and is
//! said once ([`Dcc::take_heavy_tail`]). The set moves with the targets,
//! so it is held at every row: an estimate, or given parameters, that the
//! targets have left outside are projected back onto it before the next
//! `Q` is formed (given parameters are projected afresh each row, so they
//! return once the targets do). Without `g` the set is `a + b ≤ 1 − 10⁻⁴`.
//!
//! **The estimator** (`params` absent). Row `t`'s quasi-log-likelihood is
//! `l_t = −½ (ln |R_t| + z_tᵀ R_t⁻¹ z_t)`; its score is `½ (uᵀ ∂R_i u −
//! tr(R⁻¹ ∂R_i))`, `u = R⁻¹ z`, and its expected information `I_ij = ½
//! tr(R⁻¹ ∂R_i R⁻¹ ∂R_j)`, with `∂R` from `∂Q`, which its own recursion
//! carries at the parameters of its time (`dcc::rows`). Each learned row of
//! weight `w` steps
//!
//! ```text
//! W ← λ W + w,   r = w / W
//! H ← H + r (I_t − H)
//! θ ← Proj_H(θ + r H⁻¹ g_t)
//! ```
//!
//! the recursive prediction-error method, whose estimate agrees with the
//! batch fit's to first order; `Proj_H` is the nearest point of the set in
//! `H`'s metric. `λ` is the decay's factor for the row's clock step; with
//! no decay `r = 1/(n + N)` after a start of `N` rows, and with a
//! half-life it floors at `1 − λ`, a tracking estimator whose targets and
//! information forget at the same rate.
//!
//! **The start.** The stage's first `start_rows` learned rows (500 by
//! default), or the rows of its first `start_span` of the clock (up to and
//! including the first at or past the span), are buffered and report no
//! correlation; the targets are their weighted means, the parameters the
//! batch quasi-likelihood's maximum over them with `Q_1 = Q̄` (projected
//! Fisher scoring from a grid, run to its rounding, `dcc::fit`), and the
//! buffer is replayed at the fit to build `H` and the recursion. With
//! `params` given, the start only forms the targets.
//!
//! **A start reports only a recursion its rows identify**, judged by the
//! condition, not by a count of rows. Its fit is withheld unless the
//! likelihood-ratio statistic against constant correlation (`a = b = 0`,
//! and `g = 0`), at Kish's count of the weights, reaches χ² at 0.999 for
//! those restrictions -- 13.8 for DCC's two, 16.3 for ADCC's three -- and
//! the information there is positive definite; a buffer whose `Q̄` is not
//! positive definite past its rounding (fewer rows than series, a series
//! constant over them, or two the same: `dcc::mat::definite`) is withheld
//! too. A fit that passes but is not at its optimum, the gain its projected
//! Newton step predicts over `10⁻⁶` nats (the Newton decrement inside the
//! set), is not stepped from either. A try that does not start keeps its
//! buffer: the next waits for as many new rows as it held, so twice the
//! rows while the cap holds them, and from then on the latest rows the
//! cap holds, the oldest leaving as each new one comes ([`Dcc::dropped`]
//! counts them), each try on rows the last did not see. Nothing is cleared,
//! which could not cure a shortage of rows, and no count of tries ends the
//! wait. While a fit is withheld the bank names the wait `not_identified`
//! ([`Dcc::no_effect`]); a fit short of its optimum alone is the start's
//! own warm-up still. How the iteration stopped decides nothing. Under a
//! decay the evidence a buffer holds stops growing: its rows' weights are
//! decayed to the last, so Kish's count of them is near `(1 + λ)/(1 − λ)`
//! at most, 289 at a half-life of 100 rows, however many are buffered; a
//! half-life must be several times the rows that tell the recursion from a
//! constant correlation, or each try is a draw at the same odds. The GARCH
//! margins buffer their own start first (2,000 rows by default, held to
//! their own test of a variance process), so with margins the correlation
//! reports from the margins' start plus the stage's.
//!
//! **What a row reports**, before it is learned: with margins, each
//! series' variance; the correlation `R_t` of each pair `i < j`; and the
//! row's Gaussian log-density under `R_t`, `−½ (k ln 2π + ln |R_t| + z_tᵀ
//! R_t⁻¹ z_t)`, where every `z` is known. Not a number while the weight is
//! under `min_weight`, the correlation and the density before the start.
//!
//! **Rows, weights and the clock.** The recursion is per observation, as a
//! GARCH's is: `Q_{t+1}` follows row `t` however far back on the clock it
//! was. The decay acts on the clock: on the targets, the weight and the
//! estimator's gain. A row of weight 0, or with any series' residual
//! absent, does not step the recursion, the targets or the estimate; it
//! decays the weights by its clock step. `weight_sum` is the weight
//! learned before the row and before its own decay (hard rule 8); the
//! margins keep their own.
//!
//! **State**, in bytes: the stage's `8 ((2 + s) k² + e (p k² + p²) + p) +
//! 48`, `s` 1 for the asymmetric model, `p` its parameters (2 or 3), `e`
//! 1 when they are estimated: `32 k² + 96` for DCC and `48 k² + 144` for
//! ADCC, estimated, `16 k² + 64` and `24 k² + 72` given; the start's buffer,
//! `8 (k + 2)` a row until it is fitted, at most what `max_state_bytes`
//! holds beside the rest; 32 of the model's; and the
//! margins' (`garch`'s state, 168 a series for `garch` with normal
//! innovations, with its own start's buffer of 16 bytes a row a series
//! until it is fitted). The margins' start comes before the stage's, so the
//! peak is the larger of the two. A row costs `O(p k³)`: the factor and
//! inverse of `R`, and `R⁻¹ ∂R` for each parameter.

use std::cell::RefCell;

use serde::{Deserialize, Serialize};

use crate::clock_span::{HeldCap, Place, check_max_state_bytes, start_due};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, Garch, GarchCfg, GarchDist, Stamp};

mod fit;
mod mat;
mod region;
mod rows;

pub use region::PERSISTENCE_CAP as DCC_PERSISTENCE_CAP;
use region::{PERSISTENCE_CAP, delta, identity, inside, project, project_into};
use rows::{WORK, advance, quad, read, score_info};

/// Default initial batch size for the correlation stage's adaptive doubling
/// start when neither `start_rows` nor `start_span` is given: `64`, the
/// smallest power of two above the unit-information floor
/// `4 χ²₁(0.999) ≈ 43.3` for `SE ≤ 1 / (2 √χ²₁(0.999))`.
pub const DCC_START_ROWS: usize = 64;

/// Default initial batch size for the correlation stage of `n_targets`
/// series when neither `start_rows` nor `start_span` is given:
/// `max(64, next_power_of_two(2 * n_targets))`, ensuring `n_kish > k` for
/// a strictly positive definite `Q̄` on the first try while doubling cleanly
/// (`n0, 2 n0, 4 n0, …`) until LR, soundness and Fisher-SE localization
/// pass, bounded by `max_state_bytes`.
pub fn default_dcc_start_rows(n_targets: usize, _asymmetric: bool) -> usize {
    DCC_START_ROWS.max((2 * n_targets.max(2)).next_power_of_two())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DccCfg {
    /// The series, one per target; at least 2.
    pub n_targets: usize,
    /// The asymmetric model (ADCC), with `g` and `N̄`.
    #[serde(default)]
    pub asymmetric: bool,
    /// Each series' GARCH margin: its `vol`, `dist`, `truncation`,
    /// `variance_targeting`, `start_rows` and `start_span` are read; its
    /// `n_targets` is this model's, its horizon 1, its `min_weight` 0 and
    /// its cap this model's less the stage's fixed bytes. `None` takes the
    /// targets as standardized residuals already.
    #[serde(default)]
    pub margins: Option<GarchCfg>,
    /// `(a, b)` or `(a, b, g)` given, held rather than estimated.
    #[serde(default)]
    pub params: Option<Vec<f64>>,
    /// The targets', the weight's and the estimator's decay.
    pub decay: Decay,
    /// The stage's start, when `start_span` is not given.
    pub start_rows: usize,
    /// The stage's start as a span of the clock, in its units.
    #[serde(default)]
    pub start_span: Option<f64>,
    /// `weight_sum` before a row reports.
    pub min_weight: f64,
    /// The bytes a group's state may hold, the starts' buffers included;
    /// `None` is 256 KB.
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// Whether the stage's start uses the tighter adaptive RPEM handoff
    /// Bollerslev-Wooldridge SE gate (set when neither `start_rows` nor
    /// `start_span` was explicitly given). Omitted on wire when `false`.
    #[serde(default, skip_serializing_if = "is_false")]
    pub adaptive_start: bool,
}

fn is_false(v: &bool) -> bool {
    !*v
}

impl DccCfg {
    /// The parameters: `(a, b)`, or `(a, b, g)` for the asymmetric model.
    pub fn n_params(&self) -> usize {
        2 + usize::from(self.asymmetric)
    }

    /// The parameters' names, in the order `coefficients` gives them.
    pub fn names(&self) -> &'static [&'static str] {
        &["a", "b", "g"][..self.n_params()]
    }

    /// Whether the parameters are estimated rather than given.
    pub fn estimates(&self) -> bool {
        self.params.is_none()
    }

    /// The pairs `i < j`, `k (k − 1)/2`.
    pub fn n_pairs(&self) -> usize {
        self.n_targets * (self.n_targets - 1) / 2
    }

    /// A row's report: the margins' variances (with margins), the pairs'
    /// correlations, and the log-density.
    pub fn n_outputs(&self) -> usize {
        self.margins.as_ref().map_or(0, |_| self.n_targets) + self.n_pairs() + 1
    }

    fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    /// The stage's bytes with no row buffered: `8 ((2 + s) k² + e (p k² +
    /// p²) + p) + 48`.
    pub fn stage_bytes(&self) -> u64 {
        let k = self.n_targets as u64;
        let p = self.n_params() as u64;
        let s = u64::from(self.asymmetric);
        let e = u64::from(self.estimates());
        8 * ((2 + s) * k * k + e * (p * k * k + p * p) + p) + 48
    }

    /// The bytes one buffered row of the stage's start holds: `8 (k + 2)`.
    pub fn row_bytes(&self) -> u64 {
        8 * (self.n_targets as u64 + 2)
    }

    /// The margins' configuration as the model runs them.
    fn margin_cfg(&self) -> Option<GarchCfg> {
        self.margins.as_ref().map(|m| GarchCfg {
            n_targets: self.n_targets,
            horizons: Vec::new(),
            min_weight: 0.0,
            max_state_bytes: Some(self.cap().saturating_sub(self.stage_bytes() + 32).max(1)),
            ..m.clone()
        })
    }

    /// The most the state holds by the settings: the margins' start beside
    /// the stage's fixed bytes, or the margins after their start beside the
    /// stage's buffer; a span's buffers count no rows, which are checked as
    /// they come.
    pub fn peak_bytes(&self) -> u64 {
        let fixed = self.stage_bytes() + 32;
        let stage_rows = if self.start_span.is_some() {
            0
        } else {
            self.start_rows as u64
        };
        match self.margin_cfg() {
            None => fixed + stage_rows * self.row_bytes(),
            Some(m) => {
                let m_rows = if m.start_span.is_some() {
                    0
                } else {
                    m.start_rows
                };
                let first = Garch::state_bytes(&m, m_rows) + fixed;
                let second = Garch::state_bytes(&m, 0) + fixed + stage_rows * self.row_bytes();
                first.max(second)
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets < 2 {
            return Err(format!(
                "dcc: it needs at least two targets, the series whose correlation it models, but \
                 it has {}. Name two or more return columns in targets.",
                self.n_targets
            ));
        }
        if let Some(p) = &self.params {
            let want = self.n_params();
            if p.len() != want {
                return Err(format!(
                    "dcc: params must hold {want} values, ({}), but it holds {}. Give {}, or \
                     leave params out to estimate them.",
                    self.names().join(", "),
                    p.len(),
                    if self.asymmetric {
                        "a, b and g for the asymmetric model"
                    } else {
                        "a and b"
                    },
                ));
            }
            if p.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
                return Err(format!(
                    "dcc: params must be finite and at least 0, but they are {p:?}. A negative \
                     a, b or g can take the correlation matrix out of the positive definite \
                     cone. Give values of 0 or more."
                ));
            }
            let ab = p[0] + p[1];
            if ab.is_nan() || ab > PERSISTENCE_CAP {
                return Err(format!(
                    "dcc: a + b must be at most 1 - 1e-4, but it is {}. At 1 or more the \
                     recursion's intercept (1 - a - b) Qbar is not positive definite, and Q \
                     does not revert to its target. Lower a or b.",
                    ab
                ));
            }
        }
        if let Err(e) = self.decay.check() {
            return Err(format!("dcc: {e}"));
        }
        if self.start_span.is_none() && self.start_rows == 0 {
            return Err(format!(
                "dcc: start_rows must be at least 1, but it is 0. The start forms the targets \
                 and fits the parameters on these rows. Set start_rows to 1 or more; \
                 {DCC_START_ROWS} is the default."
            ));
        }
        if let Some(span) = self.start_span
            && !(span.is_finite() && span > 0.0)
        {
            return Err(format!(
                "dcc: start_span must be a finite span above 0, but it is {span}. Give the \
                 clock span the start buffers, or leave it out for start_rows."
            ));
        }
        crate::model::check_min_weight("dcc", self.min_weight)?;
        check_max_state_bytes("dcc", self.max_state_bytes)?;
        let peak = self.peak_bytes();
        if peak > self.cap() {
            return Err(format!(
                "dcc: its state would hold {peak} bytes, past max_state_bytes of {}, with {} \
                 targets: {} bytes for the correlation stage, {} a row of its start's buffer{}. \
                 Lower start_rows{}, or raise max_state_bytes.",
                self.cap(),
                self.n_targets,
                self.stage_bytes(),
                self.row_bytes(),
                if self.margins.is_some() {
                    ", and the margins' state and their start's buffer beside it"
                } else {
                    ""
                },
                if self.margins.is_some() {
                    " or the margins' start_rows"
                } else {
                    ""
                },
            ));
        }
        if let Some(m) = self.margin_cfg() {
            m.validate().map_err(|e| format!("dcc: {e}"))?;
        }
        Ok(())
    }
}

/// The correlation stage: the recursion, its targets, the estimate and
/// the start's buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Stage {
    started: bool,
    /// The parameters as they stand: not a number before an estimating
    /// start, the given ones (projected) otherwise.
    theta: Vec<f64>,
    qbar: Vec<f64>,
    /// `N̄`, the asymmetric model's; empty otherwise.
    nbar: Vec<f64>,
    /// `Q` for the next row.
    q: Vec<f64>,
    /// `∂Q/∂θ`, `p` matrices; empty with given parameters.
    dq: Vec<f64>,
    /// The estimator's information; empty with given parameters.
    h: Vec<f64>,
    /// The stage's weight, decayed, the start's included.
    w: f64,
    /// The decay since the stage last learned a row, carried to the next.
    pending: f64,
    /// The start's rows: each `z`, its weight and its decay factor from the
    /// row before. At its bound, a ring whose oldest row is at `head`.
    buffer: Vec<f64>,
    /// The ring's oldest row once the buffer is at its bound; 0 before.
    #[serde(default)]
    head: usize,
    first: Option<Stamp>,
    /// The start's fit was at its optimum by the projected step's gain.
    sound: bool,
    /// The tries of the start that did not start: a `Q̄` not positive
    /// definite, a fit its rows did not identify, or one short of its
    /// optimum.
    #[serde(default)]
    refits: u32,
    /// The rows the buffer held at the last try; 0 before one.
    #[serde(default)]
    tried: u64,
    /// The rows buffered since the last try.
    #[serde(default)]
    since: u64,
    /// The rows the buffer let go at its bound, oldest first.
    #[serde(default)]
    dropped: u64,
    /// The last try did not tell its recursion from constant correlation,
    /// or its `Q̄` lacked rank: the fit is withheld, and the start waits.
    #[serde(default)]
    withheld: bool,
    /// The last start's likelihood-ratio statistic against constant
    /// correlation, at Kish's count; not a number before one.
    #[serde(default = "nan", with = "crate::humanfloat::f64_or_tag")]
    lr: f64,
    /// Rows whose `Q` did not give a correlation, after which it restarted
    /// from the target.
    repairs: u64,
}

impl Stage {
    fn new(cfg: &DccCfg) -> Self {
        let (k, p) = (cfg.n_targets, cfg.n_params());
        let kk = k * k;
        let est = cfg.estimates();
        Self {
            started: false,
            theta: cfg.params.clone().unwrap_or_else(|| vec![f64::NAN; p]),
            qbar: vec![0.0; kk],
            nbar: if cfg.asymmetric {
                vec![0.0; kk]
            } else {
                Vec::new()
            },
            q: vec![0.0; kk],
            dq: if est { vec![0.0; p * kk] } else { Vec::new() },
            h: if est { vec![0.0; p * p] } else { Vec::new() },
            w: 0.0,
            pending: 1.0,
            buffer: Vec::new(),
            head: 0,
            first: None,
            sound: false,
            refits: 0,
            tried: 0,
            since: 0,
            dropped: 0,
            withheld: false,
            lr: f64::NAN,
            repairs: 0,
        }
    }

    fn has_shape(&self, cfg: &DccCfg) -> bool {
        let (k, p) = (cfg.n_targets, cfg.n_params());
        let kk = k * k;
        let est = cfg.estimates();
        self.theta.len() == p
            && self.qbar.len() == kk
            && self.q.len() == kk
            && self.nbar.len() == if cfg.asymmetric { kk } else { 0 }
            && self.dq.len() == if est { p * kk } else { 0 }
            && self.h.len() == if est { p * p } else { 0 }
            && self.buffer.len().is_multiple_of(k + 2)
            && (self.head == 0 || self.head < self.buffer.len() / (k + 2))
            && (!self.started || (self.buffer.is_empty() && self.tried == 0))
    }

    fn rows(&self, cfg: &DccCfg) -> usize {
        self.buffer.len() / (cfg.n_targets + 2)
    }
}

fn nan() -> f64 {
    f64::NAN
}

thread_local! {
    /// The margins' report and the row's residuals and variances, read on
    /// the row and dropped.
    static ROW: RefCell<(Step, Vec<f64>, Vec<f64>)> = RefCell::new((Step::default(), Vec::new(), Vec::new()));
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dcc {
    cfg: DccCfg,
    margins: Option<Garch>,
    stage: Stage,
    /// The weight learned, decayed.
    w_sum: f64,
    /// The stage's place on the clock, for a start of a span.
    place: Option<Place>,
    over: Option<u64>,
    warned: bool,
    warning: Option<u64>,
    /// A margin whose fitted `ν` is at most 4, not yet taken.
    heavy: Option<(usize, f64)>,
    /// That has been said.
    told: bool,
}

impl Dcc {
    pub fn new(cfg: DccCfg) -> Result<Self, String> {
        cfg.validate()?;
        let margins = match cfg.margin_cfg() {
            Some(m) => Some(Garch::new(m).map_err(|e| format!("dcc: {e}"))?),
            None => None,
        };
        Ok(Self {
            stage: Stage::new(&cfg),
            margins,
            w_sum: 0.0,
            place: cfg.start_span.map(|_| Place::default()),
            over: None,
            warned: false,
            warning: None,
            heavy: None,
            told: false,
            cfg,
        })
    }

    pub fn cfg(&self) -> &DccCfg {
        &self.cfg
    }

    /// The weight learned before the next row (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// The parameters as they stand, `(a, b)` or `(a, b, g)`: not a number
    /// before an estimating start; given ones as the set holds them.
    pub fn coefficients(&self) -> Vec<f64> {
        self.stage.theta.clone()
    }

    /// Whether the correlation stage's start has been fitted.
    pub fn started(&self) -> bool {
        self.stage.started
    }

    /// Whether the start's fit was at its optimum, by the gain its
    /// projected Newton step predicts (false before it, and with given
    /// parameters).
    pub fn start_sound(&self) -> bool {
        self.stage.sound
    }

    /// The tries of the start that did not start: a target that was not
    /// positive definite, a fit its rows did not identify, or one short of
    /// its optimum.
    pub fn refits(&self) -> u32 {
        self.stage.refits
    }

    /// The rows the start's buffer let go, oldest first, once it held all
    /// the cap holds (0 while it grows).
    pub fn dropped(&self) -> u64 {
        self.stage.dropped
    }

    /// The last start's likelihood-ratio statistic against constant
    /// correlation, at Kish's count (not a number before one, and with
    /// given parameters).
    pub fn start_lr(&self) -> f64 {
        self.stage.lr
    }

    /// Whether the model holds rows for a start, its own or its margins',
    /// and is not waiting on [`Self::no_effect`].
    pub fn starting(&self) -> bool {
        !self.stage.started && !self.no_effect()
    }

    /// Whether a fit is withheld because its rows did not identify it: the
    /// correlation's start did not tell its recursion from constant
    /// correlation, or a margin's did not tell its variance from a constant
    /// one; it buffers more rows.
    pub fn no_effect(&self) -> bool {
        !self.stage.started
            && (self.stage.withheld || self.margins.as_ref().is_some_and(Garch::no_effect))
    }

    /// Rows whose `Q` gave no correlation, after which it restarted from
    /// the target (none while the set holds).
    pub fn repairs(&self) -> u64 {
        self.stage.repairs
    }

    /// The margins, where the model runs them.
    pub fn margins(&self) -> Option<&Garch> {
        self.margins.as_ref()
    }

    /// The targets `Q̄` and `N̄` (empty for the symmetric model), row-major.
    pub fn targets(&self) -> (&[f64], &[f64]) {
        (&self.stage.qbar, &self.stage.nbar)
    }

    /// `Q` for the next row, row-major.
    pub fn q(&self) -> &[f64] {
        &self.stage.q
    }

    /// `δ`, the largest eigenvalue of `Q̄^{−1/2} N̄ Q̄^{−1/2}` (0 for the
    /// symmetric model; not a number before the start).
    pub fn delta(&self) -> f64 {
        if !self.stage.started {
            return f64::NAN;
        }
        if !self.cfg.asymmetric {
            return 0.0;
        }
        let mut work = Vec::new();
        delta(
            &self.stage.qbar,
            &self.stage.nbar,
            self.cfg.n_targets,
            &mut work,
        )
        .unwrap_or(f64::NAN)
    }

    /// `a + b + δ g` at the parameters and targets as they stand, held at
    /// or below [`DCC_PERSISTENCE_CAP`].
    pub fn persistence(&self) -> f64 {
        let t = &self.stage.theta;
        t[0] + t[1] + t.get(2).map_or(0.0, |g| g * self.delta())
    }

    /// A margin whose Student-t `ν` was found at or below 4, once: its
    /// position among the targets and `ν`. With no fourth moment the
    /// residuals' `z zᵀ` has no variance, so `Q` is not covariance
    /// stationary and the estimator's score has no variance either.
    pub fn take_heavy_tail(&mut self) -> Option<(usize, f64)> {
        self.heavy.take()
    }

    /// The bytes the state holds as it stands.
    pub fn bytes(&self) -> u64 {
        let m = self.margins.as_ref().map_or(0, Garch::bytes);
        m + self.cfg.stage_bytes() + 32 + 8 * self.stage.buffer.len() as u64
    }

    fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// Whether any start runs on a span of the clock, so the state's
    /// bytes are checked as rows go in.
    fn spans(&self) -> bool {
        self.cfg.start_span.is_some()
            || self
                .cfg
                .margins
                .as_ref()
                .is_some_and(|m| m.start_span.is_some())
    }

    /// The margins' variances and residuals for the row into `vars` and
    /// `zs` from their report `m`, or the targets themselves as `z`.
    fn residuals(
        &self,
        m: &Step,
        y: Option<&[Option<f64>]>,
        vars: &mut Vec<f64>,
        zs: &mut Vec<f64>,
    ) {
        let k = self.cfg.n_targets;
        vars.clear();
        zs.clear();
        if self.margins.is_some() {
            for j in 0..k {
                vars.push(m.pred.get(2 * j).copied().unwrap_or(f64::NAN));
                zs.push(m.pred.get(2 * j + 1).copied().unwrap_or(f64::NAN));
            }
        } else {
            vars.resize(k, f64::NAN);
            for j in 0..k {
                zs.push(
                    y.and_then(|y| y.get(j).copied().flatten())
                        .unwrap_or(f64::NAN),
                );
            }
        }
    }

    /// The row's report, from the state before it is learned.
    fn report(&self, vars: &[f64], zs: &[f64], out: &mut Step) {
        let k = self.cfg.n_targets;
        out.pred.clear();
        out.pred.resize(self.cfg.n_outputs(), f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;
        let open = self.w_sum >= self.cfg.min_weight;
        if !open {
            return;
        }
        let mut at = 0;
        if self.margins.is_some() {
            for (o, v) in out.pred[..k].iter_mut().zip(vars) {
                *o = *v;
            }
            at = k;
        }
        if !self.stage.started {
            return;
        }
        crate::scratch::with_borrow_mut(&WORK, |w| {
            let Some(ld) = read(&self.stage.q, k, w) else {
                return;
            };
            for i in 0..k {
                for j in (i + 1)..k {
                    out.pred[at] = w.r[i * k + j];
                    at += 1;
                }
            }
            if zs.iter().all(|z| z.is_finite()) {
                let qf = quad(zs, k, w);
                out.pred[at] = -0.5 * (k as f64 * libm::log(2.0 * std::f64::consts::PI) + ld + qf);
            }
        });
    }

    /// The row read with no learning: `y` where the residuals are known.
    fn read_into(&self, y: Option<&[Option<f64>]>, out: &mut Step) {
        crate::scratch::with_borrow_mut(&ROW, |(m, vars, zs)| {
            if let Some(g) = &self.margins {
                let got = match y {
                    Some(y) => g.predict_with(&[], y, 0.0),
                    None => g.predict(&[], 0.0),
                };
                *m = got;
            }
            self.residuals(m, y, vars, zs);
            self.report(vars, zs, out);
        });
    }

    /// The row `z` of weight `w` into the stage: buffered before the start,
    /// stepped after it. `f` is the decay factor of the row's clock step,
    /// `now` its stamp under a start of a span.
    fn learn(&mut self, z: &[f64], w: f64, f: f64, now: Option<Stamp>) {
        let fr = self.stage.pending * f;
        self.stage.pending = 1.0;
        if self.stage.started {
            self.step_stage(z, w, fr);
            return;
        }
        let stride = self.cfg.n_targets + 2;
        let rows = self.stage.rows(&self.cfg);
        let bound = self.bound();
        let s = &mut self.stage;
        if s.tried > 0 && rows as u64 >= bound {
            // At its bound the buffer keeps the latest rows the cap holds:
            // the oldest leaves, and the wait goes on by name.
            let at = s.head * stride;
            s.buffer[at..at + stride - 2].copy_from_slice(z);
            s.buffer[at + stride - 2] = w;
            s.buffer[at + stride - 1] = fr;
            s.head = (s.head + 1) % rows;
            s.dropped += 1;
        } else {
            s.buffer.extend_from_slice(z);
            s.buffer.push(w);
            s.buffer.push(fr);
        }
        s.since += 1;
        let rows = s.rows(&self.cfg);
        // The first try at the start's rows or span; each later one once
        // as many rows have come as the try before held: twice its rows
        // while the cap holds them, then the latest rows it holds, renewed.
        let due = if s.tried > 0 {
            s.since >= s.tried
        } else {
            match (self.cfg.start_span, now) {
                (Some(span), Some(now)) => start_due(&mut s.first, now, span, rows, 1),
                _ => rows >= self.cfg.start_rows,
            }
        };
        if due {
            self.start();
        }
    }

    /// The rows the start's buffer may hold: those the cap holds beside the
    /// rest of the state.
    fn bound(&self) -> u64 {
        let rest = self.bytes() - 8 * self.stage.buffer.len() as u64;
        (self.cfg.cap().saturating_sub(rest) / self.cfg.row_bytes()).max(1)
    }

    /// The start: the targets, the fit and the replay (the module docs).
    fn start(&mut self) {
        let (k, p) = (self.cfg.n_targets, self.cfg.n_params());
        let kk = k * k;
        let stride = k + 2;
        let mut buf = self.stage.buffer.clone();
        buf.rotate_left(self.stage.head * stride);
        let n = buf.len() / stride;
        let z: Vec<f64> = (0..n)
            .flat_map(|t| buf[t * stride..t * stride + k].iter().copied())
            .collect();
        // Each row's weight decayed to the last row, and the targets by the
        // recursion the stage runs after the start.
        let mut wts = vec![0.0; n];
        let mut carry = 1.0;
        for t in (0..n).rev() {
            wts[t] = buf[t * stride + k] * carry;
            carry *= buf[t * stride + k + 1];
        }
        let (mut qbar, mut nbar) = (
            vec![0.0; kk],
            vec![0.0; if self.cfg.asymmetric { kk } else { 0 }],
        );
        let mut big_w = 0.0;
        for t in 0..n {
            let (w, f) = (buf[t * stride + k], buf[t * stride + k + 1]);
            big_w = f * big_w + w;
            let r = w / big_w;
            let zt = &z[t * k..(t + 1) * k];
            update_targets(&mut qbar, &mut nbar, zt, r, k);
        }
        let (w_sum_d, w2_sum_d) = wts.iter().fold((0.0, 0.0), |(a, b), &w| (a + w, b + w * w));
        let n_kish = if w2_sum_d > 0.0 {
            (w_sum_d * w_sum_d) / w2_sum_d
        } else {
            0.0
        };
        let mut chol = vec![0.0; kk];
        let kish_ok = self.cfg.params.is_some() || n_kish > k as f64;
        if !(big_w > 0.0 && kish_ok && mat::definite(&qbar, k, &mut chol)) {
            // fewer effective rows than series, a series constant over the
            // rows, or two the same: no correlation to identify yet, and a
            // shortage of rows that clearing the buffer cannot cure, so it is
            // kept while the cap holds it doubled
            return self.wait(true);
        }
        let mut work = Vec::new();
        let dlt = if self.cfg.asymmetric {
            match delta(&qbar, &nbar, k, &mut work) {
                Some(d) => d,
                None => return self.wait(true),
            }
        } else {
            0.0
        };
        let c = fit::coefs(p, dlt);
        let theta = match &self.cfg.params {
            Some(given) => project(given, &identity(p), &c, PERSISTENCE_CAP),
            None => {
                let rows = fit::Rows {
                    z: &z,
                    w: &wts,
                    k,
                    qbar: &qbar,
                    nbar: &nbar,
                };
                let got = crate::scratch::with_borrow_mut(&WORK, |w| fit::fit(&rows, p, dlt, w));
                let Some(got) = got else {
                    return self.wait(true);
                };
                self.stage.lr = got.lr;
                if !got.identified() {
                    return self.wait(true);
                }
                if !got.sound() {
                    return self.wait(false);
                }
                let se_ok = if self.cfg.adaptive_start {
                    got.adaptive_se_identified()
                } else {
                    got.se_identified()
                };
                if !se_ok {
                    return self.wait(true);
                }
                self.stage.sound = true;
                got.theta
            }
        };
        // The replay at the fit: Q from Q̄, the sensitivities from 0, and
        // the information's weighted mean.
        let est = self.cfg.estimates();
        let mut q = qbar.clone();
        let mut dq = if est { vec![0.0; p * kk] } else { Vec::new() };
        let mut h = if est { vec![0.0; p * p] } else { Vec::new() };
        crate::scratch::with_borrow_mut(&WORK, |w| {
            for t in 0..n {
                let zt = &z[t * k..(t + 1) * k];
                if est && read(&q, k, w).is_some() {
                    quad(zt, k, w);
                    score_info(&q, &dq, k, p, w);
                    for (hi, wi) in h.iter_mut().zip(&w.info) {
                        *hi += wts[t] * wi;
                    }
                }
                advance(&theta, &qbar, &nbar, zt, k, &mut q, &mut dq);
            }
        });
        for v in h.iter_mut() {
            *v /= big_w;
        }
        let s = &mut self.stage;
        s.buffer = Vec::new();
        s.head = 0;
        s.first = None;
        s.tried = 0;
        s.since = 0;
        s.withheld = false;
        s.theta = theta;
        s.qbar = qbar;
        s.nbar = nbar;
        s.q = q;
        if est {
            s.dq = dq;
            s.h = h;
        }
        s.w = big_w;
        s.started = true;
    }

    /// A try that did not start: the buffer is kept, and the next try
    /// waits for as many new rows as this one held. `withheld` where its
    /// rows did not identify the recursion (or its `Q̄` lacked rank), set
    /// from this try alone: one short of its optimum only is not withheld.
    fn wait(&mut self, withheld: bool) {
        let rows = self.stage.rows(&self.cfg) as u64;
        let s = &mut self.stage;
        s.withheld = withheld;
        s.refits = s.refits.saturating_add(1);
        s.tried = rows;
        s.since = 0;
    }

    /// One learned row after the start (the module docs' step).
    fn step_stage(&mut self, z: &[f64], w: f64, f: f64) {
        let (k, p) = (self.cfg.n_targets, self.cfg.n_params());
        let est = self.cfg.estimates();
        let s = &mut self.stage;
        let big_w = f * s.w + w;
        let r = w / big_w;
        crate::scratch::with_borrow_mut(&WORK, |wk| {
            let read_ok = read(&s.q, k, wk).is_some();
            // the candidate, on the stack: a learned row allocates nothing
            // while the set holds it
            let mut from = [0.0; 3];
            match &self.cfg.params {
                Some(given) => from[..p].copy_from_slice(given),
                None => from[..p].copy_from_slice(&s.theta),
            }
            if est && read_ok {
                quad(z, k, wk);
                score_info(&s.q, &s.dq, k, p, wk);
                for (hi, wi) in s.h.iter_mut().zip(&wk.info) {
                    *hi += r * (wi - *hi);
                }
                let mut d = [0.0; 3];
                if mat::solve_into(&s.h, p, &wk.g, &mut d[..p]) {
                    for i in 0..p {
                        from[i] += r * d[i];
                    }
                }
            }
            update_targets(&mut s.qbar, &mut s.nbar, z, r, k);
            if !inside(&from[..p], &s.qbar, &s.nbar, k, &mut wk.region) {
                match delta_or_zero(&s.qbar, &s.nbar, k, &mut wk.region) {
                    Some(d) => {
                        let c = [1.0, 1.0, d];
                        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
                        let mut id = [0.0; 9];
                        for a in 0..p {
                            for b in 0..p {
                                id[a * p + b] = eye[a * 3 + b];
                            }
                        }
                        let metric: &[f64] = if est { &s.h } else { &id[..p * p] };
                        let x = from;
                        project_into(&x[..p], metric, &c[..p], PERSISTENCE_CAP, &mut from[..p]);
                    }
                    None => from[..p].copy_from_slice(&s.theta),
                }
            }
            if !read_ok {
                // a Q with no correlation: restart it from the target
                s.repairs += 1;
                s.q.copy_from_slice(&s.qbar);
                s.dq.iter_mut().for_each(|v| *v = 0.0);
            }
            advance(&from[..p], &s.qbar, &s.nbar, z, k, &mut s.q, &mut s.dq);
            s.theta.copy_from_slice(&from[..p]);
        });
        s.w = big_w;
    }
}

/// `Q̄ ← Q̄ + r (z zᵀ − Q̄)`, and `N̄` likewise on the negative parts where
/// it is kept.
fn update_targets(qbar: &mut [f64], nbar: &mut [f64], z: &[f64], r: f64, k: usize) {
    let neg = |v: f64| if v < 0.0 { v } else { 0.0 };
    for i in 0..k {
        for j in 0..k {
            let ij = i * k + j;
            qbar[ij] += r * (z[i] * z[j] - qbar[ij]);
            if !nbar.is_empty() {
                nbar[ij] += r * (neg(z[i]) * neg(z[j]) - nbar[ij]);
            }
        }
    }
}

/// `δ` for the asymmetric model, 0 for the symmetric one.
fn delta_or_zero(qbar: &[f64], nbar: &[f64], k: usize, work: &mut Vec<f64>) -> Option<f64> {
    if nbar.is_empty() {
        Some(0.0)
    } else {
        delta(qbar, nbar, k, work)
    }
}

impl OnlineModel for Dcc {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        let learns = weight > 0.0;
        let k = self.cfg.n_targets;
        // Once past the cap, or if either the margins' or stage's first
        // `start_span` would pass the cap on this row, refuse atomically before
        // advancing `place`, `margins`, `w_sum`, or `stage.pending`.
        let reject = |this: &mut Self, out: &mut Step| {
            if let Some(p) = this.place.as_mut() {
                p.discard_next();
            }
            if let Some(g) = this.margins.as_mut() {
                g.discard_next();
            }
            this.read_into(Some(y), out);
        };
        if self.over.is_some() {
            reject(self, out);
            return;
        }
        if learns
            && let Some(g) = &self.margins
            && let Some(g_total) = g.span_cap_check(&[], y, weight)
            && g_total > g.cfg().cap()
        {
            let with = g_total + self.cfg.stage_bytes() + 32 + 8 * self.stage.buffer.len() as u64;
            let cap = self.cfg.cap();
            // Record the overrun even if the margin's own cap bound it first.
            if self.held_cap().admit(with, cap) {
                self.over = Some(with);
            }
            reject(self, out);
            return;
        }
        if learns && !self.stage.started && self.cfg.start_span.is_some() && self.stage.tried == 0 {
            let with = self.bytes() + self.cfg.row_bytes();
            let cap = self.cfg.cap();
            if with > cap {
                let usable = match &self.margins {
                    None => y.len() >= k && y[..k].iter().all(|v| v.is_some_and(f64::is_finite)),
                    Some(g) => {
                        let m = g.predict_with(&[], y, 0.0);
                        (0..k).all(|j| m.pred.get(2 * j + 1).is_some_and(|v| v.is_finite()))
                    }
                };
                if usable && !self.held_cap().admit(with, cap) {
                    reject(self, out);
                    return;
                }
            }
        }
        let f = self.cfg.decay.factor(d_clock);
        let now = self.place.as_mut().map(|p| p.step(d_clock));
        crate::scratch::with_borrow_mut(&ROW, |(m, vars, zs)| {
            if let Some(g) = self.margins.as_mut() {
                g.step_into(&[], y, d_clock, weight, m);
                // the margins' own warning is the model's, said at half of
                // the whole state's cap below
                g.take_state_warning();
            }
            self.residuals(m, Some(y), vars, zs);
            self.report(vars, zs, out);
            out.n_eff = self.w_sum;
            self.w_sum = f * self.w_sum + if learns { weight } else { 0.0 };
            let usable = zs.len() == k && zs.iter().all(|v| v.is_finite());
            if learns && usable {
                self.learn(zs, weight, f, now);
            } else {
                self.stage.pending *= f;
            }
        });
        if !self.told
            && let Some(g) = &self.margins
            && g.cfg().dist != GarchDist::Normal
            && let Some((j, nu)) =
                (0..k).find_map(|j| g.nu(j).filter(|nu| *nu <= 4.0).map(|nu| (j, nu)))
        {
            self.told = true;
            self.heavy = Some((j, nu));
        }
        if self.spans() && self.over.is_none() {
            let (bytes, cap) = (self.bytes(), self.cfg.cap());
            self.held_cap().note(bytes, cap);
        }
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        let mut out = Step::default();
        self.read_into(None, &mut out);
        out
    }

    /// The log-density reads the row's residuals out of `y`.
    fn predict_with(&self, _x: &[f64], y: &[Option<f64>], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict_with(self, &[], y, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read_into(Some(y), &mut out);
        out
    }

    /// A start of a clock span learns its rows' stamps.
    fn stamp_next(&mut self, stamp: Stamp) {
        if let Some(p) = self.place.as_mut() {
            p.hand(stamp);
        }
        if let Some(g) = self.margins.as_mut() {
            g.stamp_next(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state(&self) -> State {
        State::new(ModelState::Dcc(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Dcc(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("dcc", m.cfg.validate())?;
                if !m.stage.has_shape(&m.cfg) || m.margins.is_some() != m.cfg.margins.is_some() {
                    return Err(crate::model::damaged(
                        "dcc",
                        "its running sums have the wrong shape",
                    ));
                }
                if let Some(g) = &m.margins {
                    Garch::restore(&State::new(ModelState::Garch(Box::new(g.clone()))))
                        .map_err(|_| crate::model::damaged("dcc", "its margins are damaged"))?;
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "dcc",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The series are the targets; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_outputs()
    }
}

#[cfg(test)]
mod tests;
