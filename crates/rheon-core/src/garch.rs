//! The GARCH family: a conditional variance for each target, a zero-mean
//! return series, with its parameters estimated online by recursive
//! quasi-maximum likelihood after a start fitted by batch likelihood.
//!
//! **The processes** (`vol`), with `s_t` the variance of row `t`'s residual
//! `e_t` given the rows before it, `z_t = e_t/√s_t`:
//!
//! ```text
//! garch    s_t = ω + α e²_{t−1} + β s_{t−1}
//! gjr      s_t = ω + α e²_{t−1} + γ 1[e_{t−1} < 0] e²_{t−1} + β s_{t−1}
//! egarch   ln s_t = ω + α (|z_{t−1}| − √(2/π)) + γ z_{t−1} + β ln s_{t−1}
//! figarch  s_t = ω/(1 − β) + Σ_{j=1..M} λ_j e²_{t−j}
//!          λ_1 = φ − β + d,  λ_j = β λ_{j−1} + δ_j − φ δ_{j−1},  δ_j = d Π_{i=2..j} (i − 1 − d)/i
//! ```
//!
//! FIGARCH is the fractionally integrated GARCH(1,d,1) written as an
//! ARCH(∞) and cut at `M` lags (`truncation`, 1,000 by default, `arch`'s);
//! the untruncated weights sum to 1, so its long-run variance is set by the
//! tail mass the cut leaves, and `M` is part of the model. Each process's
//! lags before the first row are `arch`'s backcast `b`, the
//! `0.94^i`-weighted mean of the first 75 squared residuals: every `e²` and
//! `s` lag `b`, GJR's asymmetric term `b/2`, EGARCH's news terms 0.
//!
//! **The innovations** (`dist`): normal; Student's t, `ν > 2`; Hansen's
//! skewed t, `−1 < λ < 1`; each standardized to mean 0 and variance 1, as
//! `arch`'s `StudentsT` and `SkewStudent` are. The t takes `garch` and
//! `gjr`, the skewed t `garch` alone: under a skewed `z` GJR's persistence
//! is `α + γ E[z² 1(z < 0)] + β`, not the `α + γ/2 + β` a symmetric `z`
//! gives.
//!
//! **The estimator.** Row `t`'s log-likelihood is `l_t = log f(z_t) − ½ ln
//! s_t`; its score in the process's parameters is `g_t = ½ w_t ∂ln
//! s_t/∂θ`, with `w = z² − 1` for the normal and the t's bounded `w`
//! (`garch::dist`), and the sensitivity `∂ln s_t/∂θ` carried by its own
//! recursion at the current parameters (`garch::filter`). Each learned row
//! of weight `w_t` steps
//!
//! ```text
//! a_t = w_t / W_t,        W_t the weight learned so far, this row and the start's included
//! H  ← H + a_t (I_t − H)
//! θ  ← Proj_H(θ + a_t H⁻¹ g_t)
//! ```
//!
//! `a_t = 1/(n + 2000)` for unit weights after a 2,000-row start: the
//! recursive prediction-error method, whose estimate agrees with the batch
//! fit's to first order. `I_t` is the normal's expected information `½ ∂ln
//! s ∂ln sᵀ`, or for a t the outer product of the whole score, the shape's
//! included (BHHH). `Proj_H` projects onto the admissible set in `H`'s own
//! metric, `argmin (y − x)ᵀ H (y − x)`, a t's shape eliminated by the Schur
//! complement: a Newton step projected in the Euclidean metric is no ascent
//! where the bounds hold parameters the likelihood barely tells apart. The
//! set is `arch`'s with the persistence at most `1 − 10⁻⁴` and `ω` at least
//! `10⁻⁶` of the running mean square. A t's `ν` is stepped as `s = ln(ν −
//! 2)`, `ν − 2 ← (ν − 2) e^{Δs}`, so no step carries `ν` across 2, and held
//! to `arch`'s `[2.05, 300]`; `λ` to `[−0.99, 0.99]`.
//!
//! **EGARCH** has no sign constraints in `arch`, and its likelihood is
//! defined only where the inverted recursion is stable. Its guard
//! (`garch::guard`) keeps the filter that runs contracting: its steps and
//! its start are projected onto `α ≥ |γ|` (the news term never negative),
//! `α ≤ 1` and `0 ≤ β ≤ 1 − 10⁻⁴`; its variance is held within `arch`'s
//! bounds, a factor `10⁶` either side of the 0.94-weighted mean of `e²`;
//! and the mean of `ln |β − (α |z| + γ z)/2|` over the standardized
//! residuals the filter has met, the recursion's Lyapunov exponent, read at
//! the parameters as they stand from 64 bins of `z`, is held below 0, a
//! step that would break it cut back along the step to where it holds. It
//! agrees with `arch` wherever `arch`'s own filter contracts, and carries no
//! proof that the estimate it guards is consistent.
//!
//! **Variance targeting** (`variance_targeting`, `garch` with normal
//! innovations): `ω = v̄ (1 − α − β)`, `v̄` the running mean of `e²`, and
//! `(α, β)` stepped alone, the score Huber-clipped, `z² − 1 → (min(z², 4)
//! − κ)/m` with `κ` and `m` the running means of `min(z², 4)` and `z² 1[z²
//! < 4]`. It is a different estimand from the free fit's: `ω` is a moment
//! estimate, not a likelihood one, so it agrees with the batch targeted fit
//! to within a share of a standard error that does not shrink with the
//! rows, and it needs returns with a finite fourth moment. The clip is
//! used only here: centred on its own running mean it has mean 0 at every
//! level of the variance, so only the target identifies the level.
//!
//! **The start.** The first `start_rows` rows learned (2,000 by default,
//! at least 2,000 for FIGARCH), or the rows of the first `start_span` of
//! the clock (at least 50, up to and including the first at or past the
//! span), are buffered and report nothing; then the model's own optimizer
//! (projected Newton, `garch::optim`) fits them by the form's likelihood
//! from a small grid, and the buffer is replayed at the fit to build `H`
//! and the running means. A fit that did not converge, or sits on a bound
//! of the process's set, waits for twice the rows and refits, at most
//! twice, while the state's cap holds the doubled buffer; the last fit is
//! stepped from either way. Steps run from the next row. The buffer is 16
//! bytes a row; after the start it is gone.
//!
//! **What a row reports**, per target, all before the row is learned
//! (hard rule 2): the variance `s_t`; the standardized residual `z_t =
//! e_t/√s_t`; and, with `horizon = h > 1`, the variance `h − 1` rows past
//! this one, each later row's `e²` read as its variance and GJR's
//! asymmetric term as half of it (`arch`'s analytic forecast): `ω + P f`
//! repeated, `P = α + γ/2 + β`, for `garch` and `gjr`, FIGARCH's weights
//! over the forecasts and the ring for `figarch`. EGARCH's is not linear in
//! its own past and has no closed form past one row, so `egarch` takes
//! `horizon = 1` only. Not a number before the start, or while the target's
//! weight is under `min_weight`.
//!
//! **Rows, weights and the clock.** The recursion is per observation: a
//! row's lags are the row before it, however far back on the clock that
//! was, so a gap, a capped gap or a session change breaks nothing and no
//! parameter of the process has a duration form. Only the start is a span,
//! of rows (`start_rows`) or of the clock (`start_span`). A row's weight
//! scales its evidence: its score and its information in `H` and the
//! running means, and its term in the start's likelihood; scaling every
//! weight changes nothing but `weight_sum`. A row of weight 0, or whose
//! target is absent, is read and learned from not at all: it does not step
//! the recursion, so the next row's lags are the last learned row's, as if
//! the row were not there (hard rule 9). `weight_sum` is the weight learned
//! before the row, and each target's own weight gates it (hard rule 8);
//! there is no decay.
//!
//! **The same numbers on every platform.** `ln`, `exp` and `ln_1p` are the
//! `libm` crate's, compiled from source, not the platform's math library,
//! whose last bits differ between macOS, glibc and Windows: the start's
//! optimizer, on a short start whose likelihood is flat along a ridge,
//! ends where its comparisons at the rounding left it, so those bits chose
//! the fit (`garch::math`). Every other operation is IEEE arithmetic in a
//! fixed order, so a row's outputs and the state are the same bits
//! everywhere.
//!
//! **State**, per target, in bytes: `8 (2k + m + n(n + 1)/2 + 9)` for `k`
//! process parameters, `m` shape parameters and `n` stepped coordinates,
//! EGARCH's 64 bins of `z` (512) beside them, plus the larger of the start's
//! buffer, 16 bytes a row until it is fitted, and FIGARCH's ring, `8 M`: 168
//! for `garch` (144 targeted), 216 for `gjr`, 728 for `egarch`, 208 and 256
//! for `garch` with a t and a skewed t, 264 for `gjr` with a t, and `216 + 8
//! M` (8,216 at `M = 1000`) for `figarch`, each with 32,000 for the default start's buffer until it is
//! fitted (64,000 or 128,000 after a refit), and 16 for the model. A row
//! costs `O(n³)` (the solve, `n ≤ 5`) and FIGARCH's `O(M)` pass for its
//! weights and their derivatives.

use std::cell::RefCell;

use serde::{Deserialize, Serialize};

use crate::Stamp;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

mod dist;
mod filter;
mod fit;
mod guard;
mod linalg;
mod math;
mod optim;
mod series;

use series::{Series, start_due};

/// The bytes a group's state may hold by default, a spec's
/// `max_state_bytes`: 256 KB.
pub const GARCH_MAX_STATE_BYTES: u64 = 256 * 1024;
/// The most `max_state_bytes` may be set to: 64 MiB.
pub const GARCH_STATE_BYTES_CEILING: u64 = 64 * 1024 * 1024;
/// The fewest rows a start fits.
pub const MIN_START_ROWS: usize = 50;
/// The fewest rows FIGARCH's start fits: with fewer, the lags of a ring of
/// 1,000 are mostly the backcast, and `d` is fitted low.
pub const FIGARCH_MIN_START_ROWS: usize = 2000;
/// FIGARCH's truncation by default, `arch`'s.
pub const FIGARCH_TRUNCATION: usize = 1000;

/// The variance process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GarchVol {
    #[default]
    Garch,
    Gjr,
    Egarch,
    Figarch,
}

impl GarchVol {
    /// The process's parameters: `(ω, α, β)`, `(ω, α, γ, β)` or `(ω, φ, d,
    /// β)`.
    pub fn k(self) -> usize {
        match self {
            GarchVol::Garch => 3,
            _ => 4,
        }
    }

    /// The parameters' names, in the order `coefficients` gives them.
    pub fn names(self) -> &'static [&'static str] {
        match self {
            GarchVol::Garch => &["omega", "alpha", "beta"],
            GarchVol::Gjr | GarchVol::Egarch => &["omega", "alpha", "gamma", "beta"],
            GarchVol::Figarch => &["omega", "phi", "d", "beta"],
        }
    }
}

/// The innovations' distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GarchDist {
    #[default]
    Normal,
    T,
    Skewt,
}

impl GarchDist {
    /// The shape's parameters: none, `ν`, or `(ν, λ)`.
    pub fn shape_len(self) -> usize {
        match self {
            GarchDist::Normal => 0,
            GarchDist::T => 1,
            GarchDist::Skewt => 2,
        }
    }

    /// The shape's names, after the process's in `coefficients`.
    pub fn names(self) -> &'static [&'static str] {
        &["nu", "lambda"][..self.shape_len()]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarchCfg {
    /// The series, one per target, each with its own parameters.
    pub n_targets: usize,
    pub vol: GarchVol,
    pub dist: GarchDist,
    /// FIGARCH's lags `M`; `None` is [`FIGARCH_TRUNCATION`]. The other
    /// processes take none.
    #[serde(default)]
    pub truncation: Option<usize>,
    /// `ω` set by the running mean square, `ω = v̄ (1 − α − β)`, with a
    /// Huber-clipped score: `garch` with normal innovations only.
    #[serde(default)]
    pub variance_targeting: bool,
    /// The start's rows, when `start_span` is not given.
    pub start_rows: usize,
    /// The start's span of the clock, in its units (seconds on a temporal
    /// clock), in place of `start_rows`. How many rows that is, the data
    /// decides.
    #[serde(default)]
    pub start_span: Option<f64>,
    /// The forecast's rows ahead; 1 reports none beyond the variance.
    pub horizon: usize,
    /// `weight_sum` before a row reports, each target's own.
    pub min_weight: f64,
    /// The bytes a group's state may hold, the start's buffer included;
    /// `None` is [`GARCH_MAX_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

impl GarchCfg {
    /// FIGARCH's `M`, 0 for the other processes.
    pub fn truncation(&self) -> usize {
        match self.vol {
            GarchVol::Figarch => self.truncation.unwrap_or(FIGARCH_TRUNCATION),
            _ => 0,
        }
    }

    /// The parameters `coefficients` reports for a target: the process's,
    /// then the shape's.
    pub fn n_params(&self) -> usize {
        self.vol.k() + self.dist.shape_len()
    }

    /// The coordinates the step moves: `(α, β)` under variance targeting,
    /// every parameter otherwise.
    pub(crate) fn n_free(&self) -> usize {
        if self.variance_targeting {
            2
        } else {
            self.n_params()
        }
    }

    /// Each target's slots in a row's report: the variance, the
    /// standardized residual, and the forecast where `horizon > 1`.
    pub fn slots(&self) -> usize {
        2 + usize::from(self.horizon > 1)
    }

    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(GARCH_MAX_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err(
                "garch: it needs at least one target, a series of returns whose variance it \
                 models. Name the return column in targets."
                    .into(),
            );
        }
        if self.dist == GarchDist::Skewt && self.vol == GarchVol::Gjr {
            return Err(
                "garch: dist = \"skewt\" does not take vol = \"gjr\". Under skewed innovations \
                 GJR's persistence reads E[z² 1(z < 0)], not the half of E[z²] a symmetric \
                 distribution gives, and neither its forecast nor arch's checks it. Set dist to \
                 \"t\", or vol to \"garch\"."
                    .into(),
            );
        }
        if self.variance_targeting
            && (self.vol != GarchVol::Garch || self.dist != GarchDist::Normal)
        {
            return Err(format!(
                "garch: variance_targeting takes vol = \"garch\" and dist = \"normal\" only, but \
                 vol is {:?} and dist {:?}. The targeted estimator and its clipped score are \
                 checked against arch's targeted fit on that form alone. Leave \
                 variance_targeting out, or set vol to \"garch\" and dist to \"normal\".",
                self.vol.name(),
                self.dist.name(),
            ));
        }
        if self.dist != GarchDist::Normal
            && matches!(self.vol, GarchVol::Egarch | GarchVol::Figarch)
        {
            return Err(format!(
                "garch: dist = {:?} takes vol = \"garch\" or \"gjr\" only, but vol is {:?}. The \
                 online estimator of a t's shape is checked against arch's fit on those two \
                 processes, and on no other. Set dist to \"normal\", or vol to \"garch\" or \"gjr\".",
                self.dist.name(),
                self.vol.name(),
            ));
        }
        if self.vol != GarchVol::Figarch && self.truncation.is_some() {
            return Err(format!(
                "garch: truncation is FIGARCH's count of lags, but vol is {:?}, which keeps one \
                 lag. Leave truncation out, or set vol to \"figarch\".",
                self.vol.name()
            ));
        }
        if self.truncation == Some(0) {
            return Err(
                "garch: truncation must be at least 1, but it is 0. Give FIGARCH's count \
                        of lags, or leave it out for 1000."
                    .into(),
            );
        }
        let least = if self.vol == GarchVol::Figarch {
            FIGARCH_MIN_START_ROWS
        } else {
            MIN_START_ROWS
        };
        if self.start_span.is_none() && self.start_rows < least {
            return Err(format!(
                "garch: start_rows must be at least {least}{}, but it is {}. The start fits \
                 every parameter by likelihood on these rows, and on fewer the fit is mostly \
                 noise that the online steps take thousands of rows to wash out{}. Set \
                 start_rows to {least} or more; 2000 is the default.",
                if self.vol == GarchVol::Figarch {
                    " for vol = \"figarch\""
                } else {
                    ""
                },
                self.start_rows,
                if self.vol == GarchVol::Figarch {
                    "; a FIGARCH's lags on fewer are mostly the backcast, and its d comes out low"
                } else {
                    ""
                },
            ));
        }
        if let Some(span) = self.start_span
            && !(span.is_finite() && span > 0.0)
        {
            return Err(format!(
                "garch: start_span must be a finite span above 0, but it is {span}. Give the \
                 clock span the start buffers, or leave it out for start_rows."
            ));
        }
        if self.horizon == 0 {
            return Err(
                "garch: horizon must be at least 1, but it is 0. Give the rows ahead \
                        of the forecast, or leave it out for 1, the variance alone."
                    .into(),
            );
        }
        if self.horizon > 1 && self.vol == GarchVol::Egarch {
            return Err(format!(
                "garch: vol = \"egarch\" forecasts one row ahead only, but horizon is {}. \
                 EGARCH's log-variance depends on |z| of each row between, so a forecast past one \
                 row has no closed form and is made only by simulation. Leave horizon out, or \
                 set vol to \"garch\", \"gjr\" or \"figarch\".",
                self.horizon
            ));
        }
        crate::model::check_min_weight("garch", self.min_weight)?;
        if let Some(cap) = self.max_state_bytes {
            if cap == 0 {
                return Err(
                    "garch: max_state_bytes must be above 0, but it is 0. Give the \
                            bytes a group's state may hold, or leave it out for 256 KB."
                        .into(),
                );
            }
            if cap > GARCH_STATE_BYTES_CEILING {
                return Err(format!(
                    "garch: max_state_bytes may be at most {GARCH_STATE_BYTES_CEILING} (64 MiB), \
                     but it is {cap}. A group's state is held in memory for every group of the \
                     bank at once. Set max_state_bytes to 64 MiB or less."
                ));
            }
        }
        let rows = if self.start_span.is_some() {
            0
        } else {
            self.start_rows
        };
        let bytes = Garch::state_bytes(self, rows);
        if bytes > self.cap() {
            return Err(format!(
                "garch: its state would hold {bytes} bytes, past max_state_bytes of {}, with \
                 {} targets{}. Lower {}, or raise max_state_bytes.",
                self.cap(),
                self.n_targets,
                if self.vol == GarchVol::Figarch {
                    format!(" and truncation {}", self.truncation())
                } else {
                    String::new()
                },
                if self.vol == GarchVol::Figarch {
                    "truncation or start_rows"
                } else {
                    "start_rows"
                },
            ));
        }
        Ok(())
    }
}

impl GarchVol {
    fn name(self) -> &'static str {
        match self {
            GarchVol::Garch => "garch",
            GarchVol::Gjr => "gjr",
            GarchVol::Egarch => "egarch",
            GarchVol::Figarch => "figarch",
        }
    }
}

impl GarchDist {
    fn name(self) -> &'static str {
        match self {
            GarchDist::Normal => "normal",
            GarchDist::T => "t",
            GarchDist::Skewt => "skewt",
        }
    }
}

thread_local! {
    /// FIGARCH's weights for a forecast, read on the row and dropped.
    static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Garch {
    cfg: GarchCfg,
    series: Vec<Series>,
    /// The weight learned, every target's rows.
    w_sum: f64,
    /// The clock summed from the rows' steps, the stamp of a row no stamp
    /// was handed for (a start of a clock span).
    own: f64,
    /// The stamp handed for the next row.
    #[serde(default)]
    next: Option<Stamp>,
    /// The bytes a buffered row would have taken the state to, past the cap.
    over: Option<u64>,
    warned: bool,
    warning: Option<u64>,
}

impl Garch {
    pub fn new(cfg: GarchCfg) -> Result<Self, String> {
        cfg.validate()?;
        Ok(Self {
            series: (0..cfg.n_targets).map(|_| Series::new(&cfg)).collect(),
            w_sum: 0.0,
            own: 0.0,
            next: None,
            over: None,
            warned: false,
            warning: None,
            cfg,
        })
    }

    pub fn cfg(&self) -> &GarchCfg {
        &self.cfg
    }

    /// The weight learned, every row's, before the next row (hard rule 8).
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// The bytes one target's series holds with `rows` buffered for its
    /// start: `8 (2k + m + n(n + 1)/2 + 9)`, EGARCH's bins, `8 · 64`, and
    /// the larger of the buffer,
    /// `16 rows`, and FIGARCH's ring, `8 M`.
    pub fn series_bytes(cfg: &GarchCfg, rows: usize) -> u64 {
        let (k, m, n) = (cfg.vol.k(), cfg.dist.shape_len(), cfg.n_free());
        let bins = if cfg.vol == GarchVol::Egarch {
            guard::BINS
        } else {
            0
        };
        let fixed = 8 * (2 * k + m + n * (n + 1) / 2 + 9 + bins) as u64;
        fixed + (16 * rows as u64).max(8 * cfg.truncation() as u64)
    }

    /// The bytes a model of `cfg` holds with `rows` buffered for each
    /// target's start: [`Self::series_bytes`] for each target, and 16 of
    /// its own.
    pub fn state_bytes(cfg: &GarchCfg, rows: usize) -> u64 {
        cfg.n_targets as u64 * Self::series_bytes(cfg, rows) + 16
    }

    /// The bytes the state holds as it stands.
    pub fn bytes(&self) -> u64 {
        self.series.iter().map(|s| s.bytes(&self.cfg)).sum::<u64>() + 16
    }

    /// Each target's parameters, the process's then the shape's (the
    /// names of [`GarchVol::names`] and [`GarchDist::names`]); not a number
    /// before its start.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        self.series.iter().map(|s| s.theta.clone()).collect()
    }

    /// Whether each target's start has been fitted.
    pub fn started(&self) -> Vec<bool> {
        self.series.iter().map(|s| s.started).collect()
    }

    /// Each target's start, as it stands: how many times it doubled its
    /// rows and refitted, and whether the fit it steps from converged off
    /// every bound of the process's set (false before the start).
    pub fn start_report(&self) -> Vec<(u32, bool)> {
        self.series.iter().map(|s| (s.refits, s.sound)).collect()
    }

    /// EGARCH's guard per target: the mean of `ln |β − (α |z| + γ z)/2|`
    /// at the parameters as they stand, over the standardized residuals the
    /// filter has met, which the steps hold below 0 (0 before the start, and
    /// for the other processes).
    pub fn invertibility(&self) -> Vec<f64> {
        self.series
            .iter()
            .map(|s| {
                if s.started && !s.zbins.is_empty() {
                    guard::mean_log_mu(&s.theta, &s.zbins)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Write the row's report for target `j` into `out` at its slots, the
    /// variance `s` given; `e` the row's residual where there is one.
    fn report(&self, j: usize, s: f64, e: Option<f64>, out: &mut [f64]) {
        let series = &self.series[j];
        let open = series.w_sum >= self.cfg.min_weight && s.is_finite();
        let base = j * self.cfg.slots();
        out[base] = if open { s } else { f64::NAN };
        out[base + 1] = match e {
            Some(e) if open => e / s.sqrt(),
            _ => f64::NAN,
        };
        if self.cfg.horizon > 1 {
            out[base + 2] = if open {
                crate::scratch::with_borrow_mut(&SCRATCH, |sc| series.forecast(&self.cfg, s, sc))
            } else {
                f64::NAN
            };
        }
    }

    /// The row's report, the state untouched; `y` where the residual is
    /// known.
    fn read_into(&self, y: Option<&[Option<f64>]>, out: &mut Step) {
        out.pred.clear();
        out.pred
            .resize(self.cfg.n_targets * self.cfg.slots(), f64::NAN);
        for j in 0..self.cfg.n_targets {
            let s = self.series[j].variance(&self.cfg);
            let e = y.and_then(|y| y.get(j).copied().flatten());
            self.report(j, s, e, &mut out.pred);
        }
        out.n_eff = self.w_sum;
        out.extra = None;
    }
}

impl OnlineModel for Garch {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A residual or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`): such a residual is absent, and such a
        // weight learns nothing. The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        let handed = self.next.take();
        let now = self.cfg.start_span.map(|_| {
            self.own += d_clock;
            handed.unwrap_or(Stamp::from(self.own))
        });
        let (n, k) = (self.cfg.n_targets, self.cfg.vol.k());
        out.pred.clear();
        out.pred.resize(n * self.cfg.slots(), f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;
        let learns = weight > 0.0;
        if learns {
            self.w_sum += weight;
        }
        let mut dl = [0.0; 4];
        for j in 0..n {
            let e = y.get(j).copied().flatten();
            let s = self.series[j].read(&self.cfg, &mut dl[..k]);
            self.report(j, s, e, &mut out.pred);
            // A row of weight 0, or with no residual for this target, is
            // not learned: the recursion does not step, so the next row
            // reads this target's lags as the last learned row left them.
            let Some(e) = e.filter(|_| learns && self.over.is_none()) else {
                continue;
            };
            let series = &mut self.series[j];
            if series.started {
                series.learn(&self.cfg, e, weight, s, &dl[..k], false);
                continue;
            }
            // The start's buffer: a clock span's rows are checked against
            // the cap as they come, the warning at half of it once.
            if self.cfg.start_span.is_some() {
                let with = self.series.iter().map(|s| s.bytes(&self.cfg)).sum::<u64>() + 16 + 16;
                if with > self.cfg.cap() {
                    self.over = Some(with);
                    continue;
                }
                if !self.warned && 2 * with > self.cfg.cap() {
                    self.warned = true;
                    self.warning = Some(with);
                }
            }
            let series = &mut self.series[j];
            if series.buffer.is_empty() {
                series.first = now;
            }
            series.buffer.push([e, weight]);
            series.w_sum += weight;
            if start_due(&self.cfg, series, now) {
                // a refit doubles this buffer: the cap must hold it
                let others: u64 = self.series.iter().map(|s| s.bytes(&self.cfg)).sum::<u64>()
                    - self.series[j].bytes(&self.cfg);
                let doubled = Self::series_bytes(&self.cfg, 2 * self.series[j].buffer.len());
                let may_double = others + doubled + 16 <= self.cfg.cap();
                self.series[j].start(&self.cfg, may_double);
            }
        }
    }

    fn predict(&self, _x: &[f64], _d_clock: f64) -> Step {
        let mut out = Step::default();
        self.read_into(None, &mut out);
        out
    }

    /// The standardized residual reads the row's residual out of `y`.
    fn predict_with(&self, _x: &[f64], y: &[Option<f64>], _d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict_with(self, &[], y, _d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read_into(Some(y), &mut out);
        out
    }

    /// A start of a clock span learns its rows' stamps: they decide where
    /// it closes.
    fn stamp_next(&mut self, stamp: Stamp) {
        if self.cfg.start_span.is_some() {
            self.next = Some(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.series.iter().map(|s| s.w_sum));
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::Garch(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Garch(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("garch", m.cfg.validate())?;
                let shaped = m.series.len() == m.cfg.n_targets
                    && m.series.iter().all(|s| s.has_shape(&m.cfg));
                if !shaped {
                    return Err(crate::model::damaged(
                        "garch",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "garch",
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
        self.cfg.n_targets * self.cfg.slots()
    }
}

#[cfg(test)]
mod tests;
