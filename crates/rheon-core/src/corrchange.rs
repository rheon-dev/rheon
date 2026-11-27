//! `corrchange`: has the correlation structure changed?
//! (docs/ENHANCEMENTS.md E59)
//!
//! Two tests, because there are two questions.
//!
//! # `monitor`: the constancy test of Wied, Krämer & Dehling (2012), linear
//!
//! A **closed-sample** fluctuation test, run span by span. Over a span of
//! `T` rows, per pair (docs/PLAN.md task 553, after the review of
//! 2026-11-22):
//!
//! ```text
//! ξₜ   = x̃ₜỹₜ − (ρ̂/2)(x̃ₜ² + ỹₜ²)
//! Q    = max_{1≤j≤T} |Σ_{t≤j} ξₜ| / (√T·D̂)
//! crit = q(1 − α) − 0.5826/√T
//! ```
//!
//! `x̃`, `ỹ` are the span's rows centred at its means and scaled by its
//! standard deviations, `ρ̂` their correlation, and `D̂` the delta-method
//! long-run standard deviation of `ρ̂`: the five raw moments `Uₜ = (x², y²,
//! x, y, xy)` centred at their span means, their Bartlett long-run
//! covariance `Σ̂` at bandwidth `γ_T = ⌊ln T⌋`, mapped to `(σ_x², σ_y², σ_xy)`
//! by `D₂` and to `ρ` by `D₃`, so `D̂² = D₃D₂Σ̂D₂'D₃'`. On the standardised
//! span `m_x = m_y = 0` and both variances are 1, so `D₃D₂ = (−ρ̂/2, −ρ̂/2,
//! 0, 0, 1)` and `D̂²` is the Bartlett long-run variance of the one series
//! `ξ`, exactly. Formed from the raw moments, `σ_x² = E[x²] − E[x]²` lost the
//! level's digits: 1.2e-5 of `D̂` at a level of 1e5, and NaN at 1e8
//! (docs/PLAN.md task 103).
//!
//! WKD's statistic is `max_{2≤j≤T} (j/√T)·|ρ̂ⱼ − ρ̂_T| / D̂`; their proof
//! reduces `j(ρ̂ⱼ − ρ̂_T)` to `Σ_{t≤j} ξₜ` plus a delta-method remainder, and
//! the limit `sup|B|` is the law of the sum. The remainder over-rejected:
//! `ρ̂ⱼ` of a few rows is `±1` at two rows whatever the data, so at `ρ =
//! 0.9` and `T = 100` a quarter of stable spans flagged, most on their first
//! rows. The sum has no remainder, so its level holds at every `T` and `ρ`
//! (0.008 to 0.049 measured, every cell). `crit` is the Kolmogorov
//! quantile less `GRID_SHIFT/√T`, the shift between a bridge's maximum over
//! `T` points and its supremum.
//!
//! The kernel is WKD's: `D̂₁ = ΣₜΣᵤ k((t−u)/γ_T)VₜVᵤ'` with `k(x) = 1 − |x|`
//! (their Appendix A.1), so lag `l` is weighted `1 − l/γ` and lag `γ` not at
//! all. Until docs/PLAN.md task 114 (the user's decision, 2026-09-28) it
//! was Newey–West's `1 − l/(γ+1)` over lags `0..=γ`. At `bandwidth = 1` only
//! lag 0 is left, and `Q` is statsmodels' `breaks_cusumolsresid(ξ)`.
//!
//! The one condition the sum does not remove: with the correlation constant
//! and the columns' scales at `r_x`, `r_y` times their span values, `E[ξ] =
//! −(ρ/2)(r_x − r_y)²`. So the same CUSUM of `x̃² − ỹ²`, which has the same
//! limit while the scale ratio holds, is read beside it, and a pair whose
//! ratio CUSUM passes `crit` withholds its flag ([`CorrChange::monitor_pair`]).
//!
//! # `sequential`: Wied & Galeano (2013), on Fisher's scale
//!
//! The paper's own form of the monitor, with a boundary function, read from
//! their SFB 823 preprint (Discussion Paper 12/2012) on 2026-09-28,
//! docs/PLAN.md task 114: a history of `m` rows taken as stable, then each
//! monitored row tested against it, against `c·w(b)`, `w(b) = (1 + b)(b/(1 +
//! b))^γ`, `c` solved (`crate::boundary`). The detector compares the
//! monitored rows' correlation with the history's on Fisher's scale, each at
//! its own finite-sample variance (task 553): `V_k = ((k − 3)/√(m − 3))·
//! |atanh ρ̂_k − atanh ρ̂_h|·(1 − ρ̂_h²)/σ̂` at `b = (k − 3)/(m − 3)`, whose
//! variance is `b + b²` exactly ([`CorrChange::sequential_stat`]).
//!
//! # On the clock
//!
//! `span_clock` (`window_size`) puts each kind's span on the clock: a
//! `monitor` span and a `sequential` history from their first row through
//! the first at least `window_size` after it ([`crate::clock_span`]); a
//! period through `monitor_clock` or `⌊m·monitor_span/window_size⌋` rows,
//! the first of the two; the two windows `(t − 2D, t − D]` and `(t − D, t]`,
//! the held permutation value rescaled to each report's `(n₁, n₂)`. Each is
//! tested at the rows it holds, with no least count; their bytes are held
//! to `max_state_bytes` as they come.
//!
//! # `window`: two adjacent windows
//!
//! `‖vech(R̂_pre − R̂_post)‖`, the size of the change in the correlation
//! matrix between two adjacent windows, against a fixed threshold or a
//! **permutation** critical value. Not a sign-flip null: negating a whole
//! row leaves every `xₜxₜ'` and so every correlation matrix exactly where
//! it was, so a sign-flip null has no spread at all. The exchangeable null
//! for "the two windows share a distribution" is a permutation of the
//! pooled rows between them, in blocks of `perm_block` so that serially
//! dependent rows do not make it too liberal. A report flags when fewer than
//! `⌊alpha·(B + 1)⌋` of the `B` draws reach its statistic: the standard
//! permutation test, whose size against draws of a report's own windows is
//! that count over `B + 1`, at most `alpha` ([`perm_rejections`]). The
//! critical value is drawn after a report and held for
//! `permute_every_rows` reports, so a later report meets draws of windows a
//! few rows older than its own, and the size is about `alpha`, not held to
//! it (5.1% measured at alpha 0.05 and 200 draws; review round 10, S6-3;
//! 4.8% to 5.2% at windows of 50 to 400 rows held for `window_rows`
//! reports, the default since task 328).
//! Too few draws to flag at all are refused.
//!
//! The report's correlations are merged from summaries of blocks of
//! `⌈k/2⌉ + 3` rows, each a count, an anchor row, the mean's offset from
//! it and the centred co-moments (`crate::window_moments`, which has the
//! rule and the scheme): `O(b·k²)` a row however long the windows, where
//! reading both afresh was `O(window_rows·k²)` (task 328; finding S3 of
//! task 326's scaling audit). The sums are rebuilt from the ring and the
//! count of rows it has taken (`ring_rows`, schema 66), so they are not
//! saved. A window of fewer than four blocks is read whole, as before:
//! there the two passes cost no more than the merges (measured, task 328).
//! The merged correlations are within `γ_K·κ_a·κ_b` of the exact ones,
//! `κ² = n·range²/M_aa`, `K = 8b + 29G + 123`, `G = ⌊window_rows/b⌋`, at any
//! level (the tests derive it); measured, at most 3.6e-15 off the exact
//! statistic where the two-pass form was 1.2e-14 off.
//!
//! Unset, `permute_every_rows` is `window_rows` (task 328): a redraw is
//! `n_perm` passes over the `2·window_rows` pooled rows, so held for a
//! window's length it costs `O(n_perm·k²)` a row whatever the window.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::since::Since;
use crate::window_moments::WindowSums;
use crate::{Decay, EwDiag, SplitMix64, Stamp};

/// Which test; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrChangeKind {
    Monitor,
    Window,
    /// Wied & Galeano's (2013) detector: a history of `span_rows` rows, then
    /// every row of a monitoring period tested against it.
    Sequential,
}

/// The norm the `window` kind measures the change in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeNorm {
    L1,
    LInf,
}

/// `P(sup|B| ≤ x) = 1 − 2·Σ_{k≥1} (−1)^{k−1}·exp(−2k²x²)`, the Kolmogorov
/// distribution: the limiting law of the `monitor` statistic.
pub fn kolmogorov_cdf(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for k in 1..200 {
        let term = (-2.0 * (k * k) as f64 * x * x).exp();
        sum += if k % 2 == 1 { term } else { -term };
        if term < 1e-18 {
            break;
        }
    }
    (1.0 - 2.0 * sum).clamp(0.0, 1.0)
}

/// `P(sup|B| > x) = 2·Σ_{k≥1} (−1)^{k−1}·exp(−2k²x²)`, the Kolmogorov
/// survival function, summed until a term no longer moves the sum: so it
/// keeps its relative precision however small the tail, where `1 −`
/// [`kolmogorov_cdf`] keeps only its absolute one. Converges in a few terms
/// for `x ≥ 1`, which is where it is read.
fn kolmogorov_sf(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0;
    for k in 1..200 {
        let term = (-2.0 * (k * k) as f64 * x * x).exp();
        sum += if k % 2 == 1 { term } else { -term };
        if term <= 0.25 * f64::EPSILON * sum {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

/// [`kolmogorov_cdf`] by the theta-function series `√(2π)/x ·
/// Σ_{k≥1} exp(−(2k−1)²π²/(8x²))` (Feller 1948, eq. 1.4), which keeps its
/// relative precision as the distribution function goes to 0, where `1 −
/// 2·Σ…` cancels. Converges in a few terms for `x ≤ 1`, which is where it
/// is read.
fn kolmogorov_cdf_theta(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x <= 0.0 {
        return 0.0;
    }
    let a = PI * PI / (8.0 * x * x);
    let mut sum = 0.0;
    for k in 1..200u32 {
        let n = f64::from(2 * k - 1);
        let term = (-(n * n) * a).exp();
        sum += term;
        if term <= 0.25 * f64::EPSILON * sum {
            break;
        }
    }
    ((2.0 * PI).sqrt() / x * sum).clamp(0.0, 1.0)
}

/// The levels at which [`kolmogorov_quantile`] leaves the bisection on `1 −
/// alpha` for a series that keeps the tail: below `KOLMOGOROV_TAIL` it
/// bisects on [`kolmogorov_sf`] against `alpha`, above `1 −
/// KOLMOGOROV_TAIL` on [`kolmogorov_cdf_theta`] against `1 − alpha` (exact
/// there, since `alpha ≥ 1/2`). Either is within about an ulp of the true
/// quantile at every level.
///
/// Between them the bisection on [`kolmogorov_cdf`] against `1 − alpha` is
/// kept, so the critical values the frozen states run at (5 % on one pair)
/// and the usual levels from 1 % to 99 % keep their bits. Its root is
/// off by the roundings of order `u = 2⁻⁵³` in `1 − alpha` and in `1 −
/// 2·Σ…`, over the density `f(x)`: relative `u/(4x²α)` in the upper tail
/// (`f ≈ 4xα`) and `u/((1 − α)(π²/(4x²) − 1))` in the lower one, `1.05e-15`
/// and `9.5e-16` at 1e-2 from either end -- a handful of ulps -- and ten
/// times that a decade further out. Measured against a 50-digit sum of the
/// series (docs/PLAN.md task 278), the kept path is `5.1e-16` off at 0.01,
/// `4.1e-16` at 0.05, `3.9e-16` at 0.9 and `1.5e-16` at 0.99; beyond it, it
/// was `3.6e-15` off at 1e-3, `5.9e-7` at 1e-12, `0.097` at 1e-20 (where
/// `1 − alpha` is 1 and the value stopped rising), `1.5e-14` at 0.999 and
/// `3.3e-12` at `1 − 1e-6`.
const KOLMOGOROV_TAIL: f64 = 1e-2;

/// The `1 − alpha` quantile of `sup|B|`, by bisection. `1.3581` at 5 %,
/// `1.6276` at 1 %, `1.2239` at 10 %.
///
/// Below `alpha = 0.01` it solves `P(sup|B| > x) = alpha` on the survival
/// series, which keeps the tail's relative precision at any level down to
/// the smallest double; above 0.99 it solves on the theta series,
/// which keeps the lower tail's. Between them it solves [`kolmogorov_cdf`]
/// `= 1 − alpha`, whose rounding is of order `1e-16` absolute: a small
/// share of a level that large.
pub fn kolmogorov_quantile(alpha: f64) -> f64 {
    if !(alpha > 0.0 && alpha < 1.0) {
        return f64::NAN;
    }
    // `below(x)`: the quantile lies above `x`.
    let bisect = |hi: f64, below: &dyn Fn(f64) -> bool| {
        let mut lo = 0.0f64;
        let mut hi = hi;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if below(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    if alpha < KOLMOGOROV_TAIL {
        // `2·exp(−2x²) ≥ alpha` up to `x = 19.31` at the smallest double.
        return bisect(20.0, &|x| kolmogorov_sf(x) > alpha);
    }
    let target = 1.0 - alpha;
    if alpha > 1.0 - KOLMOGOROV_TAIL {
        // The quantile at 0.99 is 0.441, and `F(1) = 0.73`.
        return bisect(1.0, &|x| kolmogorov_cdf_theta(x) < target);
    }
    bisect(10.0, &|x| kolmogorov_cdf(x) < target)
}

/// The Bartlett kernel `k(x) = 1 − |x|` on `[−1, 1]`.
#[inline]
fn bartlett(x: f64) -> f64 {
    let a = x.abs();
    if a < 1.0 { 1.0 - a } else { 0.0 }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrChangeCfg {
    pub n_features: usize,
    pub kind: CorrChangeKind,
    /// Rows per comparison block: the span `T` that `"monitor"` tests for
    /// constancy, or the length of each of the two adjacent blocks
    /// `"window"` compares. One number, because it is one concept -- the two
    /// kinds only differ in what they do with the block.
    pub span_rows: usize,
    /// Nominal level; under `monitor` and `sequential` the critical value is
    /// the quantile at `1 − alpha/npairs` under Bonferroni. Under
    /// `sequential`, a share below 5e-11
    /// ([`crate::boundary::MIN_SEQUENTIAL_TAIL`]) with no `crit` is refused
    /// ([`Self::validate`]).
    pub alpha: f64,
    /// `"bonferroni"` (default) or `"none"`: whether `monitor` and
    /// `sequential` split `alpha` over the pairs. Under `window` it changes
    /// nothing, since there is one statistic over all the pairs: the
    /// permutation critical value is taken at `alpha`.
    pub alpha_adjust: String,
    /// `monitor`: the Bartlett bandwidth, `⌊ln T⌋` when `None`.
    pub bandwidth: Option<usize>,
    /// `monitor`: run the CUSUM on the equicorrelation of the standardised
    /// row instead of every pair.
    pub scalar: bool,
    /// `scalar`: the standardiser's decay.
    pub decay: Decay,
    /// `window`: a fixed critical value, or `None` for the permutation one.
    pub crit: Option<f64>,
    pub n_perm: usize,
    pub permute_every_rows: usize,
    pub perm_block: usize,
    pub norm: ChangeNorm,
    pub seed: u64,
    /// Empty both rings and start over at a flag.
    pub reset: bool,
    /// `sequential`: the rows monitored after each history, W&G's `⌊mT⌋`,
    /// so their `T` is `monitor_rows / span_rows`.
    #[serde(default)]
    pub monitor_rows: usize,
    /// `sequential`: the boundary's exponent `γ` in `w(b) = (1 + b)(b/(1 +
    /// b))^γ`, `0 ≤ γ ≤ 0.49`; 0 is the straight boundary `1 + b`. W&G
    /// allow up to 1/2, but the critical value is solved for, and the
    /// solve's work grows as `1/(1/2 − γ)`.
    #[serde(default)]
    pub boundary_gamma: f64,
    /// `window`: the permutation critical value redrawn every this many
    /// clock units as well as every `permute_every_rows` reports, whichever
    /// comes first (docs/PLAN.md task 538); `0` is every report. Last, and
    /// written only when set.
    // `inf` is a setting, so its JSON is a tag (review 2026-11-23, SPEC-4:
    // written as `null`, the spec's defaults and the state's JSON refused).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "crate::humanfloat::opt_f64_or_tag"
    )]
    pub permute_every: Option<f64>,
    /// `window_size`: the span (`monitor`), the history (`sequential`) or
    /// each of the two windows (`window`) as a span of the clock, in place
    /// of `span_rows` (docs/PLAN.md task 553). Written only when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span_clock: Option<f64>,
    /// `sequential` under `span_clock`: the monitoring period as a span of
    /// the clock, `monitor_span`, in place of `monitor_rows`. Written only
    /// when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub monitor_clock: Option<f64>,
    /// The bytes a clock form's held rows may take the state to: 256 KB
    /// unset ([`crate::clock_span::DEFAULT_STATE_BYTES`]). Written only when
    /// set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

/// `−ζ(1/2)/√(2π)`: how far, in units of `1/√T`, the maximum of a Brownian
/// bridge read at `T` equally spaced points falls short of its supremum over
/// the continuum, to first order (Siegmund 1985, "Sequential Analysis",
/// §10.5; Broadie, Glasserman and Kou 1997, "A continuity correction for
/// discrete barrier options", Theorem 1). The `monitor` statistic is such a
/// maximum and its critical value the supremum's quantile, so the critical
/// value is read `GRID_SHIFT/√T` lower: a constant of the limit law, not of
/// any data.
pub const GRID_SHIFT: f64 = 0.582_597_157_939_010_7;

/// The largest `boundary_gamma` a configuration may give. W&G's boundary
/// takes `γ < 1/2`, but the critical value is solved for, and the solve's
/// work grows as `1/(1/2 − γ)`: seconds here, and hours just under 1/2
/// (review 2026-10-05, CD1).
const MAX_BOUNDARY_GAMMA: f64 = 0.49;

/// The most permutation draws `"window"` takes: the draws are held to take
/// their quantile, and each is a pass over both blocks. `2^62` panicked
/// "capacity overflow" at the first full window and `10^11` never returned
/// (review 2026-10-06, CD10); the ceiling of the lags (`MAX_LAG`).
pub const MAX_PERM: usize = 1 << 20;

/// How many of `draws` permutation statistics may reach the observed one,
/// plus one, before the window test stops flagging: `m = ⌊alpha·(draws +
/// 1)⌋`. The permutation test counts the observed statistic among the
/// draws, `p = (1 + #{T_b ≥ T}) / (B + 1)`, and rejects at `p ≤ alpha`, so
/// it flags when fewer than `m` draws reach `T`: `T` above the `(B + 1 −
/// m)`-th smallest draw. On exchangeable rows the observed statistic is one
/// more draw of the same law, its rank among the `B + 1` uniform, so the
/// size is `m / (B + 1) ≤ alpha`. A level given as a multiple of `1 / (B +
/// 1)` lands on it: the product is nudged up by four units of rounding
/// before its floor, since a decimal level is rounded to a double and the
/// product rounded again, either of which can leave it just under the
/// whole number it stands for -- `0.29 · 100` is `28.999999999999996`, and
/// so is `0.145 · 200`, which flagged with 28 draws reaching `T` where the
/// level allows 29 (size 0.28 and 0.14). A level within that nudge under a
/// multiple reads as the multiple, a size past it by four units of
/// rounding at most. At 0 no statistic can flag.
/// `x`, a small positive bound a remedy sets a value at or above, to three
/// significant figures rounded up (WRITING §11: a figure is rounded toward
/// the side that keeps the remedy true).
fn up_to_three(x: f64) -> String {
    let step = 10f64.powi(x.log10().floor() as i32 - 2);
    // A hair below the quotient first, so a value already at three figures
    // (1.5e-10) is not rounded up a step by its last bit.
    format!("{:.2e}", ((x / step) * (1.0 - 1e-12)).ceil() * step)
}

pub fn perm_rejections(alpha: f64, draws: usize) -> usize {
    // `alpha < 1`, so the product is below `draws + 1` by more than the
    // nudge (`draws ≤ MAX_PERM`, `alpha ≤ 1 − u`): at most `draws`.
    let product = alpha * (draws as f64 + 1.0);
    (product * (1.0 + 4.0 * PERM_ROUNDING)).floor() as usize
}

/// The unit roundoff of a double, `2⁻⁵³`: [`perm_rejections`]'s nudge is
/// four of it.
const PERM_ROUNDING: f64 = f64::EPSILON / 2.0;

/// The fewest draws at which the window test can flag at `alpha`: the
/// smallest `B` with [`perm_rejections`]`(alpha, B) ≥ 1`, which is
/// `⌈1/alpha⌉ − 1`; `None` past [`MAX_PERM`].
pub fn min_perm(alpha: f64) -> Option<usize> {
    if perm_rejections(alpha, MAX_PERM) == 0 {
        return None;
    }
    // Start from the closed form and settle it on the predicate itself, so
    // the two never disagree by a rounding.
    let mut b = ((1.0 / alpha).ceil() as usize)
        .saturating_sub(1)
        .min(MAX_PERM);
    while b > 0 && perm_rejections(alpha, b - 1) >= 1 {
        b -= 1;
    }
    while perm_rejections(alpha, b) == 0 {
        b += 1;
    }
    Some(b)
}

impl CorrChangeCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        no.take(self.decay.check().map_err(|e| format!("corrchange: {e}")));
        if let Some(c) = self.crit {
            // A NaN never flags and a non-positive one flags every row, both
            // in silence (docs/REVIEW-E54-E64.md CC1).
            if !(c > 0.0 && c.is_finite()) {
                no.refuse(format!(
                    "corrchange: crit must be finite and above 0, but it is {c}. Set crit to the \
                     critical value the statistic is compared with, or leave it out for the \
                     default."
                ));
            }
        }
        if self.n_features < 2 {
            no.refuse(format!(
                "corrchange: the model needs at least two features, since a correlation is \
                 between two columns, but it has {}. Name at least two columns in features.",
                self.n_features
            ));
        }
        if !(self.alpha > 0.0 && self.alpha < 1.0) {
            no.refuse(format!(
                "corrchange: alpha must be above 0 and below 1, but it is {}. Set alpha to the \
                 rate of false alarms to test at, such as 0.05.",
                self.alpha
            ));
        }
        if !["bonferroni", "none"].contains(&self.alpha_adjust.as_str()) {
            no.refuse(format!(
                "corrchange: alpha_adjust must be \"bonferroni\" or \"none\", but it is {:?}. Use \
                 one of those words.",
                self.alpha_adjust
            ));
        }
        let bandwidth = || {
            "corrchange: bandwidth must be at least 1, but it is 0. Set bandwidth to 1 or more, \
             or leave it out for the automatic choice."
        };
        // A span of the clock in place of a count of rows (docs/PLAN.md task
        // 553): above 0 and finite, and not beside the count.
        let on_clock = self.span_clock.is_some();
        for (name, span) in [
            ("window_size", self.span_clock),
            ("monitor_span", self.monitor_clock),
        ] {
            if let Some(s) = span
                && !(s > 0.0 && s.is_finite())
            {
                no.refuse(format!(
                    "corrchange: {name} must be a span of the clock above 0, but it is {s}. Set \
                     {name} to the clock units the span runs for."
                ));
            }
        }
        if on_clock && self.span_rows != 0 {
            no.refuse(format!(
                "corrchange: window_rows and window_size are both set ({} rows and {} clock \
                 units). Give one: window_rows counts rows, window_size spans the clock.",
                self.span_rows,
                self.span_clock.unwrap_or(f64::NAN)
            ));
        }
        if self.monitor_clock.is_some() {
            if self.kind != CorrChangeKind::Sequential {
                no.refuse(
                    "corrchange: monitor_span applies only under kind=\"sequential\". Leave \
                     monitor_span out, or set kind=\"sequential\".",
                );
            } else if !on_clock {
                no.refuse(
                    "corrchange: monitor_span needs window_size: the critical value plans the \
                     monitored rows as the history's rows times monitor_span over window_size. \
                     Give window_size in place of window_rows, or give monitor_rows in place of \
                     monitor_span.",
                );
            } else if self.monitor_rows != 0 {
                no.refuse(format!(
                    "corrchange: monitor_rows and monitor_span are both set ({} rows and {} \
                     clock units). Give one.",
                    self.monitor_rows,
                    self.monitor_clock.unwrap_or(f64::NAN)
                ));
            }
        }
        no.take(crate::clock_span::check_max_state_bytes(
            "corrchange",
            self.max_state_bytes,
        ));
        match self.kind {
            CorrChangeKind::Monitor => {
                if !on_clock && self.span_rows < 8 {
                    no.refuse(format!(
                        "corrchange: kind=\"monitor\" needs window_rows of at least 8, but it is \
                         {}: the statistic is a maximum over those rows. Set window_rows to 8 or \
                         more.",
                        self.span_rows
                    ));
                }
                if self.bandwidth == Some(0) {
                    no.refuse(bandwidth());
                }
            }
            CorrChangeKind::Sequential => {
                if !on_clock && self.span_rows < 8 {
                    no.refuse(format!(
                        "corrchange: kind=\"sequential\" needs window_rows of at least 8, but it \
                         is {}: the long-run variance is estimated from those rows. Set \
                         window_rows to 8 or more.",
                        self.span_rows
                    ));
                }
                // A period with no row the detector can read reports nothing,
                // ever: a pair's Fisher `z` carries the information `k − 3`,
                // none before the fourth monitored row; `scalar`'s mean needs
                // a second row to differ from the first.
                if self.monitor_clock.is_none() && !self.scalar && self.monitor_rows < 4 {
                    no.refuse(format!(
                        "corrchange: kind=\"sequential\" needs monitor_rows of at least 4, but it \
                         is {}: each pair's monitored correlation is compared on Fisher's scale, \
                         where k rows carry the information k - 3, so a period of fewer than four \
                         rows reports nothing. Set monitor_rows to 4 or more.",
                        self.monitor_rows
                    ));
                }
                if self.monitor_clock.is_none() && self.scalar && self.monitor_rows < 2 {
                    no.refuse(format!(
                        "corrchange: kind=\"sequential\" needs monitor_rows of at least 2, but it \
                         is {}: the monitored mean is compared from its second row. Set \
                         monitor_rows to 2 or more.",
                        self.monitor_rows
                    ));
                }
                // Not up to 1/2: the critical value is solved for
                // (`crate::boundary`), and the solve's work grows as 1/(1/2
                // − γ). Measured under load on a release build, 0.9 s at
                // 0.45, 2.8 s at 0.49 and 32 s at 0.499, and about an hour
                // at 0.49999 by its step count, in every process and on
                // every load (review 2026-10-05, CD1).
                if !(0.0..=MAX_BOUNDARY_GAMMA).contains(&self.boundary_gamma) {
                    no.refuse(format!(
                        "corrchange: boundary_gamma must be from 0 to {MAX_BOUNDARY_GAMMA}, but \
                         it is {}: computing the critical value takes seconds at \
                         {MAX_BOUNDARY_GAMMA} and hours just under 0.5. Set boundary_gamma to \
                         {MAX_BOUNDARY_GAMMA} or below.",
                        self.boundary_gamma
                    ));
                }
                if self.bandwidth == Some(0) {
                    no.refuse(bandwidth());
                }
                // The critical value is a quantile at `1 − alpha/npairs`
                // (`alpha` itself without Bonferroni), and below `2^-52`
                // that is 1, or a double a step or two from it: the quantile
                // of 1 is NaN, and the detector ran, reported its statistic
                // and could never flag, saying nothing (review round 4,
                // CD12). The tail cannot be read in its own right either:
                // the series and the solve both give the mass inside the
                // boundary, `1 −` the tail. And well above `2^-52` the
                // solve's quantile is not accurate: below 5e-11 it is off
                // the series at `γ = 0` by more than the 0.03 the paper's
                // table is good to, and at `γ > 0` it stopped rising with
                // `γ` (`boundary::MIN_SEQUENTIAL_TAIL`; docs/PLAN.md task
                // 195). A configured `crit` reads no `alpha`.
                let floor = crate::boundary::MIN_SEQUENTIAL_TAIL;
                // An alpha out of its range is the range check's to refuse.
                let in_range = self.alpha > 0.0 && self.alpha < 1.0;
                if in_range && self.crit.is_none() && self.pair_alpha() < floor {
                    // Under Bonferroni with several pairs each is tested at
                    // a share of alpha; with one pair the share is alpha.
                    let shared = self.alpha_adjust == "bonferroni" && self.npairs() > 1;
                    // The least alpha whose share clears the floor, rounded
                    // up so the remedy holds; none below 1 where the pairs
                    // are too many.
                    let least = floor * self.npairs() as f64;
                    let (n, share) = (self.npairs(), self.pair_alpha());
                    if !shared {
                        no.refuse(format!(
                            "corrchange: kind=\"sequential\" tests each pair at alpha = {}, \
                             below 5e-11, where the critical value cannot be computed accurately. \
                             Set alpha to 5e-11 or more, or give crit.",
                            self.alpha
                        ));
                    } else if self.alpha >= floor && least < 1.0 {
                        // alpha_adjust="none" tests each pair at alpha, which
                        // clears the floor.
                        no.refuse(format!(
                            "corrchange: kind=\"sequential\" tests each of its {n} pairs at its \
                             share of alpha, {share:.2e}, below 5e-11, where the critical value \
                             cannot be computed accurately. Set alpha to {} or more, set \
                             alpha_adjust=\"none\", or give crit.",
                            up_to_three(least)
                        ));
                    } else if self.alpha >= floor {
                        no.refuse(format!(
                            "corrchange: kind=\"sequential\" tests each of its {n} pairs at its \
                             share of alpha, {share:.2e}, below 5e-11, where the critical value \
                             cannot be computed accurately. Set alpha_adjust=\"none\", or give \
                             crit."
                        ));
                    } else if least < 1.0 {
                        no.refuse(format!(
                            "corrchange: kind=\"sequential\" tests each of its {n} pairs at its \
                             share of alpha, {share:.2e}, below 5e-11, where the critical value \
                             cannot be computed accurately. Set alpha to {} or more, or give crit.",
                            up_to_three(least)
                        ));
                    } else {
                        no.refuse(format!(
                            "corrchange: kind=\"sequential\" tests each of its {n} pairs at its \
                             share of alpha, {share:.2e}, below 5e-11, where the critical value \
                             cannot be computed accurately. Give crit, or use fewer features."
                        ));
                    }
                }
            }
            CorrChangeKind::Window => {
                if !on_clock && self.span_rows < 3 {
                    no.refuse(format!(
                        "corrchange: kind=\"window\" needs window_rows of at least 3, but it is \
                         {}. Set window_rows to 3 or more.",
                        self.span_rows
                    ));
                }
                if self.crit.is_none() && self.n_perm < 20 {
                    no.refuse(format!(
                        "corrchange: n_perm must be at least 20 for a permutation critical value, \
                         but it is {}. Set n_perm to 20 or more, or give crit.",
                        self.n_perm
                    ));
                }
                if self.n_perm > MAX_PERM {
                    no.refuse(format!(
                        "corrchange: n_perm must be at most {MAX_PERM}, but it is {}: every draw \
                         is kept, and each takes a pass over the rows. Lower n_perm to \
                         {MAX_PERM} or below.",
                        self.n_perm
                    ));
                }
                // Under a clock window the pool's rows are the data's, so
                // the block is bounded below alone.
                if on_clock && self.perm_block == 0 {
                    no.refuse(
                        "corrchange: perm_block must be at least 1, but it is 0. Set perm_block \
                         to the rows a block of the permutation holds.",
                    );
                } else if self.perm_block == 0 || (!on_clock && self.perm_block > self.span_rows) {
                    no.refuse(format!(
                        "corrchange: perm_block must be from 1 to window_rows ({}), but it is {}. \
                         Set perm_block within that range.",
                        self.span_rows, self.perm_block
                    ));
                }
                if self.permute_every_rows == 0 {
                    no.refuse(
                        "corrchange: permute_every_rows must be at least 1, but it is 0. Set \
                         permute_every_rows to 1 or more.",
                    );
                }
                if let Some(d) = self.permute_every
                    && (d.is_nan() || d < 0.0)
                {
                    no.refuse(format!(
                        "corrchange: permute_every must be at least 0 clock units, but it is \
                         {d}. Set permute_every to a span of the clock, 0 for every report, or \
                         leave it out."
                    ));
                }
                if self.scalar {
                    no.refuse(
                        "corrchange: scalar applies only under kind=\"monitor\" or \
                         \"sequential\", but kind is \"window\", whose statistic is already one \
                         number. Leave out scalar, or set kind=\"monitor\" or \"sequential\".",
                    );
                }
            }
        }
        // The window's cadence and nothing else's, as the bank's spec has
        // it: unchecked under the other kinds, a NaN or a negative span
        // was taken, and stepped a clock no draw reads (review 2026-11-23,
        // REGIME-1).
        if self.kind != CorrChangeKind::Window && self.permute_every.is_some() {
            no.refuse(format!(
                "corrchange: permute_every applies only under kind=\"window\", but kind is \"{}\". \
                 Leave permute_every out, or set kind=\"window\".",
                if self.kind == CorrChangeKind::Monitor {
                    "monitor"
                } else {
                    "sequential"
                }
            ));
        }
        no.finish()
    }

    /// Pairs the statistic is a maximum over.
    fn npairs(&self) -> usize {
        if self.scalar {
            1
        } else {
            self.n_features * (self.n_features - 1) / 2
        }
    }

    /// The level each pair is tested at: `alpha`, over the pairs under
    /// Bonferroni.
    fn pair_alpha(&self) -> f64 {
        if self.alpha_adjust == "bonferroni" {
            self.alpha / self.npairs() as f64
        } else {
            self.alpha
        }
    }

    /// The critical value at the configured rows: the Kolmogorov quantile
    /// under `monitor` (before the grid correction, [`Self::monitor_crit`]),
    /// the configured one under `window`, and under `sequential` the
    /// configured one or Wied & Galeano's for a history of `span_rows` and a
    /// period of `monitor_rows` ([`Self::sequential_crit`]); under a clock
    /// span, whose rows each cycle counts, the boundary's quantile alone.
    fn fixed_crit(&self) -> Option<f64> {
        match self.kind {
            CorrChangeKind::Monitor => Some(kolmogorov_quantile(self.pair_alpha())),
            CorrChangeKind::Window => self.crit,
            CorrChangeKind::Sequential if self.span_clock.is_some() => self.crit.or_else(|| {
                Some(crate::boundary::boundary_quantile(
                    1.0 - self.pair_alpha(),
                    self.boundary_gamma,
                ))
            }),
            CorrChangeKind::Sequential => {
                Some(self.sequential_crit(self.span_rows, self.monitor_rows))
            }
        }
    }

    /// `monitor`'s critical value for a span of `t` rows: the Kolmogorov
    /// quantile at the pair's level less the grid correction
    /// [`GRID_SHIFT`]`/√t`, since the statistic is a maximum over the span's
    /// `t` rows and the quantile is of a supremum over a continuum.
    fn monitor_crit(&self, t: usize) -> f64 {
        kolmogorov_quantile(self.pair_alpha()) - GRID_SHIFT / (t as f64).sqrt()
    }

    /// `sequential`'s critical value for a history of `m` rows and a period
    /// of `k`: the configured one, or Wied & Galeano's `c` at the period's
    /// length in the detector's own time. That is `k/m` for `scalar`'s
    /// mean, and for a pair's Fisher `z` the information `(k − 3)/(m − 3)`,
    /// at which the detector's variance is `b + b²` exactly
    /// ([`Self::sequential_stat`]). NaN where the history or the period
    /// holds no information (a ratio not above 0): no verdict.
    fn sequential_crit(&self, m: usize, k: usize) -> f64 {
        if let Some(c) = self.crit {
            return c;
        }
        let ratio = if self.scalar {
            k as f64 / m as f64
        } else {
            (k as f64 - 3.0) / (m as f64 - 3.0)
        };
        crate::boundary::sequential_crit(self.pair_alpha(), self.boundary_gamma, ratio)
    }

    /// Output slots per statistic: one per pair, or one for `scalar`.
    fn width(&self) -> usize {
        if self.scalar { 1 } else { self.n_features }
    }
}

/// `sequential`'s monitoring period: what the history left, and the
/// monitored rows so far (docs/PLAN.md task 114). Before it, while the
/// history fills, the history is the model's `ring`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Monitoring {
    /// Per pair (`a < b` in order), or the one `scalar` series: the
    /// history's correlation, or the mean of `u`.
    level: Vec<f64>,
    /// And its long-run standard deviation, the inverse of W&G's `D̂`;
    /// NaN where the history gives none (a constant column, a collinear
    /// pair), and that pair has no verdict until the next history.
    scale: Vec<f64>,
    /// The monitored rows so far, `k − 1` on the `k`-th monitored row.
    rows: Vec<Vec<f64>>,
    /// Their running means, second moments and co-moments (per pair, in
    /// the order of `level`), the mean-form (Welford) sums.
    mean: Vec<f64>,
    m2: Vec<f64>,
    cross: Vec<f64>,
    /// The history's rows, `m`, under a clock span (docs/PLAN.md task 553),
    /// where each cycle counts its own; 0 on the row clock's count, which
    /// is `span_rows`. Written only where it is not 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    history: u64,
    /// The period's rows, `K`, planned when it began under a clock span:
    /// `⌊m·monitor_span/window_size⌋`, or `monitor_rows`; 0 on the row
    /// clock's count, which is `monitor_rows`. Written only where it is
    /// not 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    horizon: u64,
    /// `atanh` of each pair's `level`: Fisher's `z` of the history's
    /// correlation, which the detector compares with the period's. Not
    /// state: a function of `level`, rebuilt on load.
    #[serde(skip)]
    z_level: Vec<f64>,
    /// The critical value for this cycle's `m` and `K`
    /// ([`CorrChangeCfg::sequential_crit`]). Not state: it calls `powf`,
    /// whose last bits are the platform's, so it is computed when the
    /// period begins and on load.
    #[serde(skip)]
    crit: f64,
}

impl Monitoring {
    /// The history's rows and the period's, `(m, K)`: the cycle's own under
    /// a clock span, else the configured counts.
    fn rows_of(&self, cfg: &CorrChangeCfg) -> (usize, usize) {
        let pick = |own: u64, set: usize| if own > 0 { own as usize } else { set };
        (
            pick(self.history, cfg.span_rows),
            pick(self.horizon, cfg.monitor_rows),
        )
    }

    /// The parts that are not state, from the parts that are.
    fn rebuild(&mut self, cfg: &CorrChangeCfg) {
        self.z_level = if cfg.scalar {
            Vec::new()
        } else {
            self.level.iter().map(|&r| libm::atanh(r)).collect()
        };
        let (m, k) = self.rows_of(cfg);
        self.crit = cfg.sequential_crit(m, k);
    }
}

/// The clock a clock form's spans are read on (docs/PLAN.md task 553):
/// where the stream stands, and the stamp handed for the next row. Kept only
/// under `window_size`. A row's key is its stamp where the caller hands one,
/// the count of the rows that counted on the row clock, and the clock summed
/// from the steps for a caller that hands none; a row of weight 0 counts in
/// no clock (hard rule 9), so on the row clock it is no step.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct SpanClock {
    /// The clock summed from every step.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    own: f64,
    /// The rows that counted: the key on the row clock.
    counted: u64,
    /// The stamp of the last row stepped, where one was handed: what a read
    /// ahead is taken from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    read: Option<Stamp>,
    /// The last stamp handed was the row clock's.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    on_rows: bool,
    /// The next row's stamp, and whether it is the row clock's. Never state.
    #[serde(skip)]
    next: Option<(Stamp, bool)>,
}

impl SpanClock {
    /// The key of a row `d_clock` after the last, read as a row that counts:
    /// the stamp handed for it, or a read ahead of the last.
    fn key(&self, d_clock: f64) -> Stamp {
        match self.next {
            Some((_, true)) => Stamp::from((self.counted + 1) as f64),
            Some((s, false)) => s,
            None if self.on_rows => Stamp::from((self.counted + 1) as f64),
            None => match self.read {
                Some(last) => crate::since::back_by(last, -d_clock),
                None => Stamp::from(self.own + d_clock),
            },
        }
    }

    /// A row stepped, `d_clock` after the last; it counts where `counts`.
    fn step(&mut self, d_clock: f64, counts: bool) {
        let handed = self.next.take();
        self.own += d_clock;
        if counts {
            self.counted += 1;
        }
        if let Some((s, rows)) = handed {
            self.read = Some(s);
            self.on_rows = rows;
        }
    }
}

/// The constancy monitor; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrChange {
    cfg: CorrChangeCfg,
    /// The span's rows (`monitor`) or the two adjacent windows (`window`,
    /// oldest first: the first `window` are `pre`).
    ring: VecDeque<Vec<f64>>,
    /// `scalar`: the standardiser the row is divided by.
    diag: EwDiag,
    n_eff: f64,
    /// Learned rows since the last flag; `None` before the first.
    since_flag: Option<u64>,
    /// The permutation critical value in force, and the rows since it was
    /// drawn.
    perm_crit: Option<f64>,
    since_perm: usize,
    rng: SplitMix64,
    /// `sequential`: the monitoring period in progress, `None` while the
    /// history fills `ring` (and under the other kinds).
    #[serde(default)]
    monitoring: Option<Monitoring>,
    /// `sequential`: the critical value, a function of the cfg alone. Not
    /// state: it calls `exp`, whose last bits are the platform's, so it is
    /// computed on construction and on load.
    #[serde(skip)]
    seq_crit: f64,
    /// `window`: the rows the ring has taken since it last started (at the
    /// first row, a flag under `reset`, or `clear_lags`). It numbers the
    /// rows, and the numbers alone fix which rows each merged sum covers
    /// ([`WindowSums`]), so a state that keeps it rebuilds the same sums.
    /// Absent, 0, under the other kinds and before schema 66, where a
    /// loaded ring is numbered from its oldest row.
    #[serde(default, skip_serializing_if = "is_zero")]
    ring_rows: u64,
    /// `window`: the merged sums of the next report's two windows, `None`
    /// where the windows are read whole. A cache of the ring: rebuilt on
    /// load, never saved.
    #[serde(skip)]
    sums: Option<WindowSums>,
    /// `window`: the step's buffers between rows ([`Work`]).
    #[serde(skip)]
    work: Work,
    /// `window` under `permute_every` (task 538): the clock since the
    /// critical value was drawn. Written only where it is kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    since_draw: Option<Since>,
    /// Under `window_size` (docs/PLAN.md task 553), and written only there:
    /// the clock the spans are read on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    clock: Option<SpanClock>,
    /// The key of the first row of the span, history or period in
    /// progress, or of the ring since it last started (`window`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    first: Option<Stamp>,
    /// `window` under `window_size`: each ring row's key, oldest first.
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    keys: VecDeque<Stamp>,
    /// `window` under `window_size`: the rows in the two windows the
    /// critical value in force was drawn over, `(n₁, n₂)`, by which it is
    /// rescaled to each report's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    perm_counts: Option<[u64; 2]>,
    /// The bytes a clock form's next row would have taken the state to,
    /// past `max_state_bytes`: from then on no row is learned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    warned: bool,
    /// The warning raised and not yet taken. Never state.
    #[serde(skip)]
    warning: Option<u64>,
}

/// `window`'s buffers between rows, so a row allocates nothing (task 335,
/// F7): the report's two windows' summaries and their working space, or
/// its two windows' correlations; the row the ring let go last, which the
/// next row is copied into; and a permutation redraw's statistics, block
/// order and pooled rows. Not state: two models with the same state are
/// the same model whatever is left here, a copy keeps none of it, and a
/// state file carries none.
#[derive(Default)]
struct Work {
    report: Vec<f64>,
    spare: Vec<f64>,
    /// The rows a `monitor` span's close let go, for the next span's rows
    /// to be copied into: a span of rows is allocated once, not at each of
    /// its rows.
    rows: Vec<Vec<f64>>,
    stats: Vec<f64>,
    order: Vec<usize>,
    /// The pooled rows, `d` numbers to a row, one after another, in the
    /// ring's order.
    pool: Vec<f64>,
    /// The pooled rows' places in `pool`, in a permuted order.
    shuffled: Vec<usize>,
}

impl Work {
    /// `r`, copied into the row the window's ring let go last, or into a
    /// row of its own length where the ring has let none go: on the window
    /// kind's first rows, and on every row of the `monitor` and
    /// `sequential` kinds, whose rings hand their rows on rather than let
    /// them go. A row of its own is allocated at its exact length, as
    /// `to_vec` does: growing an empty vector cost each of those rows about
    /// 100 instructions more under Callgrind on x86-64 (60 natively on
    /// arm64), and gave a one-column ring's row room for four (task 345).
    fn ring_row(&mut self, r: &[f64]) -> Vec<f64> {
        if self.spare.capacity() < r.len() {
            let Some(mut row) = self.rows.pop() else {
                return r.to_vec();
            };
            row.clear();
            row.extend_from_slice(r);
            return row;
        }
        let mut row = std::mem::take(&mut self.spare);
        row.clear();
        row.extend_from_slice(r);
        row
    }
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

impl std::fmt::Debug for Work {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Work")
    }
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// One pair's `monitor` report ([`CorrChange::monitor_pair`]): the
/// correlation's CUSUM, the row its maximum is at, and the scale ratio's
/// CUSUM.
#[derive(Debug, Clone, Copy)]
struct PairStat {
    stat: f64,
    at: usize,
    ratio: f64,
}

impl PairStat {
    /// No verdict: a column without spread, or a collinear pair.
    const NONE: Self = Self {
        stat: f64::NAN,
        at: 0,
        ratio: f64::NAN,
    };
}

/// A uniform draw below `n` (Lemire 2019, "Fast random integer generation in
/// an interval"): the high word of `x·n` for a 64-bit `x` from `next`.
/// Outcome `j` owns the `x` with `x·n` in `[j·2^64, (j+1)·2^64)`, which is
/// `⌊2^64/n⌋` of them, or one more when the low word of `x·n` falls below the
/// threshold `2^64 mod n`. A draw with its low word below the threshold is
/// redrawn, so every outcome keeps exactly `⌊2^64/n⌋` values of `x` and the
/// draw is exactly uniform. The threshold's division is paid only when the low
/// word is below `n`, in `n / 2^64` of the draws, and a redraw happens in
/// `(2^64 mod n) / 2^64` of them, so the generator is called once a draw to
/// within that. The remainder this replaces, `x % n`, was a 64-bit division on
/// every draw and biased by up to `n / 2^64`.
#[inline]
pub(crate) fn below_by(n: u64, mut next: impl FnMut() -> u64) -> u64 {
    debug_assert!(n > 0);
    let mut m = u128::from(next()) * u128::from(n);
    if (m as u64) < n {
        let threshold = n.wrapping_neg() % n;
        while (m as u64) < threshold {
            m = u128::from(next()) * u128::from(n);
        }
    }
    (m >> 64) as u64
}

/// [`below_by`] over the model's own generator, one call a draw but for a
/// redraw.
#[inline]
fn below(rng: &mut SplitMix64, n: u64) -> u64 {
    below_by(n, || rng.next_u64())
}

impl CorrChange {
    pub fn new(cfg: CorrChangeCfg) -> Result<Self, String> {
        cfg.validate()?;
        // Refused here and not in `validate`, which a restored state runs
        // too: a state saved with such a pair still loads, and reports no
        // critical value (task 278, ff).
        if cfg.kind == CorrChangeKind::Window
            && cfg.crit.is_none()
            && perm_rejections(cfg.alpha, cfg.n_perm) == 0
        {
            if let Some(b) = min_perm(cfg.alpha) {
                return Err(format!(
                    "corrchange: n_perm = {} draws are too few for the permutation test to flag \
                     at alpha = {}. Set n_perm to {b} or more, or give crit.",
                    cfg.n_perm, cfg.alpha
                ));
            }
            // The least alpha the most draws can flag at, rounded up so the
            // remedy holds.
            let least = up_to_three(1.0 / (MAX_PERM as f64 + 1.0));
            return Err(format!(
                "corrchange: no n_perm up to {MAX_PERM} lets the permutation test flag at alpha = \
                 {}. Set alpha to {least} or more and n_perm to {MAX_PERM}, or give crit.",
                cfg.alpha
            ));
        }
        let d = cfg.n_features;
        let seq_crit = Self::sequential_crit_of(&cfg);
        let sums = Self::sums_of(&cfg);
        Ok(Self {
            ring: VecDeque::new(),
            diag: EwDiag::new(d),
            n_eff: 0.0,
            since_flag: None,
            perm_crit: None,
            since_perm: 0,
            rng: SplitMix64::new(cfg.seed),
            monitoring: None,
            seq_crit,
            ring_rows: 0,
            sums,
            work: Work::default(),
            since_draw: cfg.permute_every.map(|_| Since::default()),
            clock: cfg.span_clock.map(|_| SpanClock::default()),
            first: None,
            keys: VecDeque::new(),
            perm_counts: None,
            over: None,
            warned: false,
            warning: None,
            cfg,
        })
    }

    /// The run-time check of a state the data sizes, over this model's
    /// fields ([`crate::clock_span::HeldCap`]).
    fn held_cap(&mut self) -> crate::clock_span::HeldCap<'_> {
        crate::clock_span::HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// `max_state_bytes`, or its default.
    fn cap(&self) -> u64 {
        self.cfg
            .max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    /// The bytes a clock form's state takes when holding `rows` rows.
    fn held_bytes_for(&self, rows: usize) -> u64 {
        let key = if self.cfg.kind == CorrChangeKind::Window {
            std::mem::size_of::<Stamp>()
        } else {
            0
        };
        let d = self.cfg.n_features;
        let fixed = std::mem::size_of::<Self>() + 8 * (d * d + 4 * d);
        (fixed + rows * (8 * self.cfg.width() + key)) as u64
    }

    /// The bytes a clock form's state takes with `extra` more rows held: the
    /// rows of the ring and of the period in progress, `8` bytes a number,
    /// each `window` row's key beside it, and the model's fixed part.
    fn held_bytes(&self, extra: usize) -> u64 {
        let rows = self.ring.len() + self.monitoring.as_ref().map_or(0, |m| m.rows.len()) + extra;
        self.held_bytes_for(rows)
    }

    /// How many `window` rows leave at `now`, read without mutating the ring.
    fn dropped_at(&self, now: Stamp) -> usize {
        match (self.cfg.kind, self.cfg.span_clock) {
            (CorrChangeKind::Window, Some(span)) => self
                .keys
                .iter()
                .take_while(|&&k| crate::WindowClosed::Right.has_left(now, k, 2.0 * span))
                .count(),
            _ => 0,
        }
    }

    /// Whether the critical value in force is due to be drawn again: every
    /// `permute_every_rows` reports, or `permute_every` clock units since
    /// it was drawn (task 538).
    fn redraw_due(&self) -> bool {
        let by_clock = match (&self.since_draw, self.cfg.permute_every) {
            (Some(s), Some(every)) => every <= 0.0 || s.reached(every),
            _ => false,
        };
        self.since_perm >= self.cfg.permute_every_rows || by_clock
    }

    /// `window`'s merged sums, or `None` where its windows are read whole:
    /// under the other kinds, and where a window holds too few blocks for
    /// merging to cost less.
    fn sums_of(cfg: &CorrChangeCfg) -> Option<WindowSums> {
        match cfg.kind {
            // A clock window's rows leave by their stamps, not by the row
            // numbers the merged blocks are cut by: read whole.
            CorrChangeKind::Window if cfg.span_clock.is_none() => {
                WindowSums::new(cfg.n_features, cfg.span_rows)
            }
            _ => None,
        }
    }

    /// Make the merged sums ready for the next report, from the ring as it
    /// stands; or forget them where the ring is too short for a report.
    fn ready_sums(&mut self) {
        let Some(sums) = &mut self.sums else {
            return;
        };
        let held = self.ring.len();
        if held + 1 < 2 * self.cfg.span_rows {
            sums.clear();
            return;
        }
        let (ring, pos) = (&self.ring, self.ring_rows);
        let front = pos - held as u64;
        sums.ready(pos, &|q| ring[(q - front) as usize].as_slice());
    }

    /// The window statistic of the report on `row` from the merged sums:
    /// [`Self::window_stat_by`]'s norm over the same pairs, in the same
    /// order.
    fn window_stat_merged(&self, sums: &WindowSums, row: &[f64], buf: &mut Vec<f64>) -> f64 {
        let d = self.cfg.n_features;
        let layout = sums.layout();
        let stride = layout.stride();
        let (ring, pos) = (&self.ring, self.ring_rows);
        let front = pos - ring.len() as u64;
        let read = |q: u64| {
            if q < pos {
                ring[(q - front) as usize].as_slice()
            } else {
                row
            }
        };
        // Both windows' summaries, then one summary, `δ`, `c·δ` and a
        // row's deviations, in a buffer kept between rows.
        buf.clear();
        buf.resize(3 * stride + 3 * d, 0.0);
        let (out, tmp) = buf.split_at_mut(2 * stride);
        sums.summaries(pos, &read, out, tmp);
        let (pre, post) = out.split_at(stride);
        let mut acc = 0.0f64;
        for a in 0..d {
            for b in (a + 1)..d {
                let (x, y) = (layout.corr(pre, b, a), layout.corr(post, b, a));
                if !(x.is_finite() && y.is_finite()) {
                    return f64::NAN;
                }
                let diff = (x - y).abs();
                match self.cfg.norm {
                    ChangeNorm::L1 => acc += diff,
                    ChangeNorm::LInf => acc = acc.max(diff),
                }
            }
        }
        acc
    }

    /// `sequential`'s critical value, or NaN under the other kinds.
    fn sequential_crit_of(cfg: &CorrChangeCfg) -> f64 {
        match cfg.kind {
            CorrChangeKind::Sequential => cfg.fixed_crit().unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }

    pub fn cfg(&self) -> &CorrChangeCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }

    /// Rows in the ring, which is the span so far or the two windows.
    pub fn depth(&self) -> usize {
        self.ring.len()
    }

    /// Output slot labels in emission order.
    pub fn labels() -> Vec<String> {
        vec![
            "stat".into(),
            "crit".into(),
            "flag".into(),
            "since_flag".into(),
            "since_change".into(),
        ]
    }

    pub fn n_outputs_for() -> usize {
        5
    }

    /// Whether a report withheld its flag: a `monitor` report whose
    /// statistic and critical value are numbers and whose flag is not, the
    /// one report that is, because every pair over the critical value is one
    /// whose two columns' scale ratio moved inside the span
    /// ([`Self::monitor_pair`]). Every other report flags 0 or 1 wherever its
    /// statistic and critical value are numbers.
    pub fn flag_withheld(pred: &[f64]) -> bool {
        pred.len() == Self::n_outputs_for()
            && pred[0].is_finite()
            && pred[1].is_finite()
            && pred[2].is_nan()
    }

    /// The sample correlation of `rows[..n]` for the pair `(a, b)`.
    fn corr_of<R: AsRef<[f64]>>(rows: &[R], n: usize, a: usize, b: usize) -> f64 {
        if n < 2 {
            return f64::NAN;
        }
        let nf = n as f64;
        let (mut sa, mut sb) = (0.0, 0.0);
        for r in &rows[..n] {
            let r = r.as_ref();
            sa += r[a];
            sb += r[b];
        }
        let (ma, mb) = (sa / nf, sb / nf);
        let (mut vaa, mut vbb, mut vab) = (0.0, 0.0, 0.0);
        for r in &rows[..n] {
            let r = r.as_ref();
            let (da, db) = (r[a] - ma, r[b] - mb);
            vaa += da * da;
            vbb += db * db;
            vab += da * db;
        }
        crate::window_moments::pair_corr(vab, vaa, vbb)
    }

    /// `D̂`, the delta-method long-run standard deviation of `ρ̂` over the
    /// span, for the pair `(a, b)`: the Bartlett long-run standard deviation
    /// of `ξ_t = x̃_t ỹ_t − (ρ̂/2)(x̃_t² + ỹ_t²)`, the rows centred at the
    /// span's means and scaled by its standard deviations (the module docs
    /// derive it from the five raw moments).
    fn long_run_sd<R: AsRef<[f64]>>(rows: &[R], a: usize, b: usize, gamma: usize) -> f64 {
        let mut buf = Vec::new();
        if Self::influence_into(rows, a, b, false, &mut buf) {
            Self::bartlett_sd(&buf, gamma)
        } else {
            f64::NAN
        }
    }

    /// [`Self::influence_into`] into vectors of their own.
    #[cfg(test)]
    fn influence<R: AsRef<[f64]>>(
        rows: &[R],
        a: usize,
        b: usize,
        ratio: bool,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut buf = Vec::new();
        Self::influence_into(rows, a, b, ratio, &mut buf).then(|| {
            let d = buf.split_off(rows.len());
            (buf, d)
        })
    }

    /// The span's pair `(a, b)` as the two series the `monitor` kind sums:
    /// `ξ_t = x̃_t ỹ_t − (ρ̂/2)(x̃_t² + ỹ_t²)`, the influence of row `t` on
    /// `ρ̂`, and, where `ratio`, `d_t = x̃_t² − ỹ_t²`, the influence of row
    /// `t` on the log of the two columns' scale ratio (to a factor of two);
    /// `x̃`, `ỹ` the rows centred at the span's means and scaled by its
    /// standard deviations. Both sum to 0 over the span by construction
    /// (`E[x̃ỹ] = ρ̂`, `E[x̃²] = E[ỹ²] = 1` over the same rows), so neither is
    /// centred again. Written over `buf`, `ξ` then `d`, one buffer for the
    /// pairs of a span: `false`, and `buf` not read, where the pair has no
    /// correlation to test: a column without spread, or `|ρ̂|` within
    /// rounding of 1, where the delta method's variance `(1 − ρ²)²·(…)` is
    /// 0.
    fn influence_into<R: AsRef<[f64]>>(
        rows: &[R],
        a: usize,
        b: usize,
        ratio: bool,
        buf: &mut Vec<f64>,
    ) -> bool {
        // No count of rows is asked for: a span too short to carry a
        // correlation is one whose column has no spread, or whose two rows
        // are collinear, and both are refused below; and a short span that
        // passes cannot flag, since `ξ` sums to 0 and so its CUSUM is at
        // most `√(j(T − j)/T)` times its lag-0 deviation (Cauchy–Schwarz),
        // `√T/2` of it (docs/PLAN.md task 553; CLAUDE.md, no row counts).
        let t = rows.len();
        let tf = t as f64;
        // The span's means as pairs (`crate::comp`): a second pass over the
        // exact differences from the first pass's mean picks up what its
        // division rounded away, a shift of the whole span that a first-order
        // term in `ξ` would otherwise carry (1e-10 of `D̂` at a level of 1e8).
        let (mut mx, mut my) = (0.0, 0.0);
        for r in rows {
            let r = r.as_ref();
            mx += r[a];
            my += r[b];
        }
        let (mx, my) = (mx / tf, my / tf);
        let (mut mx_lo, mut my_lo) = (0.0, 0.0);
        for r in rows {
            let r = r.as_ref();
            mx_lo += r[a] - mx;
            my_lo += r[b] - my;
        }
        let (mx_lo, my_lo) = (mx_lo / tf, my_lo / tf);
        let dev = |r: &R| {
            let r = r.as_ref();
            (
                crate::comp::dev(r[a], mx, mx_lo),
                crate::comp::dev(r[b], my, my_lo),
            )
        };
        // Its moments about them.
        let (mut sx2, mut sy2, mut sxy) = (0.0, 0.0, 0.0);
        for r in rows {
            let (dx, dy) = dev(r);
            sx2 += dx * dx;
            sy2 += dy * dy;
            sxy += dx * dy;
        }
        let (sx2, sy2, sxy) = (sx2 / tf, sy2 / tf, sxy / tf);
        if !(sx2 > 0.0 && sy2 > 0.0) {
            return false;
        }
        let (sx, sy) = (sx2.sqrt(), sy2.sqrt());
        let rho = sxy / (sx * sy);
        // A pair within rounding of `|ρ̂| = 1` has no long-run standard
        // deviation to divide by: the delta method's variance is
        // `(1 − ρ²)²·(…)`, 0 on a collinear pair, and this `D̂` reaches that
        // 0 where the raw-moment form's own noise stood in for it, so the
        // numerator's rounding divided by it flagged every span of `y = 2x +
        // 3` (review 2026-09-25). NaN is no verdict, as for a constant
        // column.
        if 1.0 - rho * rho <= 64.0 * f64::EPSILON {
            return false;
        }
        buf.clear();
        buf.resize(if ratio { 2 * t } else { t }, 0.0);
        let (xi, d) = buf.split_at_mut(t);
        for (i, r) in rows.iter().enumerate() {
            let (dx, dy) = dev(r);
            let (u, v) = (dx / sx, dy / sy);
            xi[i] = u * v - 0.5 * rho * (u * u + v * v);
            if ratio {
                d[i] = u * u - v * v;
            }
        }
        true
    }

    /// The Bartlett long-run standard deviation of a series whose mean is 0
    /// by construction: the lag-0 term once, each other lag's
    /// autocovariance twice, at WKD's `k(l/γ)` (their Appendix A.1), which
    /// puts lag `γ` at 0; Newey–West's `1 − l/(γ+1)` stood here until
    /// docs/PLAN.md task 114. NaN where it is not positive.
    fn bartlett_sd(s: &[f64], gamma: usize) -> f64 {
        let t = s.len();
        let tf = t as f64;
        let mut v = 0.0;
        for lag in 0..t.min(gamma) {
            let w = bartlett(lag as f64 / gamma as f64);
            let mut acf = 0.0;
            for i in lag..t {
                acf += s[i] * s[i - lag];
            }
            v += w * acf / tf * if lag > 0 { 2.0 } else { 1.0 };
        }
        if v > 0.0 { v.sqrt() } else { f64::NAN }
    }

    /// The CUSUM of a series that sums to 0 over the span, against its
    /// long-run standard deviation `sd`: `max_j |Σ_{t≤j} s_t| / (√T·sd)`,
    /// and the first `j` it is attained at. NaN where `sd` is not a positive
    /// number.
    fn cusum(s: &[f64], sd: f64) -> (f64, usize) {
        if sd.is_nan() || sd <= 0.0 {
            return (f64::NAN, 0);
        }
        let (mut run, mut best, mut at) = (0.0f64, 0.0f64, 0);
        for (i, v) in s.iter().enumerate() {
            run += v;
            if run.abs() > best {
                (best, at) = (run.abs(), i + 1);
            }
        }
        (best / ((s.len() as f64).sqrt() * sd), at)
    }

    /// `Q` for the **mean** of a one-column span: `max_j (j/√T)·|ūⱼ − ū_T|
    /// / D̂ᵤ`, the same CUSUM with the Bartlett long-run standard deviation
    /// of `u` in place of the delta-method one. That is `scalar = true`'s
    /// statistic: the equicorrelation is already one number, so there is a
    /// mean to test and no pair.
    #[cfg(test)]
    fn scalar_stat(&self, rows: &[Vec<f64>]) -> f64 {
        self.scalar_stat_at(rows).0
    }

    /// [`Self::scalar_stat`], and the `j` its maximum is attained at.
    fn scalar_stat_at<R: AsRef<[f64]>>(&self, rows: &[R]) -> (f64, usize) {
        // No count of rows: a span whose `u` has no spread has no long-run
        // deviation, and a short one cannot flag (`Self::influence`).
        let t = rows.len();
        let tf = t as f64;
        let u: Vec<f64> = rows.iter().map(|r| r.as_ref()[0]).collect();
        let mean = u.iter().sum::<f64>() / tf;
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((tf).ln().floor() as usize).max(1));
        let sd = Self::scalar_long_run_sd(&u, gamma);
        if sd.is_nan() {
            return (f64::NAN, 0);
        }
        let mut run = 0.0;
        let (mut best, mut at) = (0.0f64, 0);
        for (j, uj) in u.iter().enumerate() {
            run += uj;
            let jj = j as f64 + 1.0;
            let q = (jj / tf.sqrt()) * (run / jj - mean).abs() / sd;
            if q > best {
                (best, at) = (q, j + 1);
            }
        }
        (best, at)
    }

    /// The Bartlett long-run standard deviation of one series about its
    /// mean, at WKD's `k(l/γ)`; NaN where it is not positive.
    fn scalar_long_run_sd(u: &[f64], gamma: usize) -> f64 {
        let t = u.len();
        let tf = t as f64;
        let mean = u.iter().sum::<f64>() / tf;
        let v: Vec<f64> = u.iter().map(|x| x - mean).collect();
        let mut var = 0.0;
        for lag in 0..t.min(gamma) {
            let w = bartlett(lag as f64 / gamma as f64);
            for s in lag..t {
                let term = w * v[s] * v[s - lag] / tf;
                var += term;
                if lag > 0 {
                    var += term;
                }
            }
        }
        if var.is_nan() || var <= 0.0 {
            f64::NAN
        } else {
            var.sqrt()
        }
    }

    /// The `monitor` statistic for one pair over the span in the ring.
    #[cfg(test)]
    fn monitor_stat(&self, rows: &[Vec<f64>], a: usize, b: usize) -> f64 {
        self.monitor_pair(rows, a, b, &mut Vec::new()).stat
    }

    /// [`Self::monitor_stat`], and the `j` its maximum is attained at.
    #[cfg(test)]
    fn monitor_stat_at<R: AsRef<[f64]>>(&self, rows: &[R], a: usize, b: usize) -> (f64, usize) {
        let p = self.monitor_pair(rows, a, b, &mut Vec::new());
        (p.stat, p.at)
    }

    /// The `monitor` kind's two CUSUMs for the pair `(a, b)` over a span of
    /// `T` rows (docs/PLAN.md task 553, after the review of 2026-11-22):
    ///
    /// ```text
    /// stat  = max_j |Σ_{t≤j} ξ_t| / (√T·D̂),    D̂ the Bartlett long-run sd of ξ
    /// ratio = max_j |Σ_{t≤j} d_t| / (√T·D̂_d),  D̂_d that of d
    /// ```
    ///
    /// ([`Self::influence`] has `ξ` and `d`). `stat` is the linearised form
    /// of Wied, Krämer & Dehling's `max_j (j/√T)|ρ̂ⱼ − ρ̂_T|/D̂`: their proof
    /// reduces `j(ρ̂ⱼ − ρ̂_T)` to `Σ_{t≤j} ξ_t` plus a remainder, and the limit
    /// `sup|B|` is the law of the sum. The remainder is what over-rejected:
    /// `ρ̂ⱼ` of a handful of rows lives in `[−1, 1]` and is `±1` at two rows
    /// whatever the data, so at `|ρ| = 0.9` one span in seven flagged on its
    /// second row (26% of stable 100-row spans at the 5% level; the paper's
    /// own Table 1 has 0.142 at `T = 200`). The sum forms no prefix
    /// correlation, so it has no remainder to wait for, at any `T` and any
    /// `ρ`. `at` is the first `j` of the maximum: the last row before the
    /// change, as Wied & Galeano (2013, Eq. 8) date one from a CUSUM.
    ///
    /// `ratio` is the check on the one condition the sum does not remove.
    /// With the correlation constant and the columns' scales at `r_x`,
    /// `r_y` times their span values, `E[ξ] = −(ρ/2)(r_x − r_y)²`: `ξ` keeps
    /// its zero mean, and the limit law holds, exactly when the ratio of the
    /// two scales does not move (or `ρ = 0`). `d`'s CUSUM has the same limit
    /// when that ratio is constant, and grows with `√T` when it moves.
    /// `buf` is the span's working space, one buffer for its pairs.
    fn monitor_pair<R: AsRef<[f64]>>(
        &self,
        rows: &[R],
        a: usize,
        b: usize,
        buf: &mut Vec<f64>,
    ) -> PairStat {
        let t = rows.len();
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((t as f64).ln().floor() as usize).max(1));
        if !Self::influence_into(rows, a, b, true, buf) {
            return PairStat::NONE;
        }
        let (xi, d) = buf.split_at(t);
        let (stat, at) = Self::cusum(xi, Self::bartlett_sd(xi, gamma));
        // A ratio with no long-run spread is one that cannot have moved:
        // `|x̃| = |ỹ|` on every row (NaN, which never withholds).
        let (ratio, _) = Self::cusum(d, Self::bartlett_sd(d, gamma));
        PairStat { stat, at, ratio }
    }

    /// The pair `(a, b)`, `a < b`, at position `idx` in the order the pairs
    /// are walked (`a` outer, `b` inner).
    fn pair_at(d: usize, mut idx: usize) -> (usize, usize) {
        for a in 0..d {
            let row = d - a - 1;
            if idx < row {
                return (a, a + 1 + idx);
            }
            idx -= row;
        }
        unreachable!("pair index past the pairs")
    }

    /// `sequential`: what the history leaves the monitoring period -- per
    /// pair the history's correlation and the long-run standard deviation
    /// of it (the inverse of W&G's `D̂`, WKD's Appendix A.1 estimator on the
    /// `m` history rows, bandwidth `⌊ln m⌋` unless given), or for `scalar`
    /// the mean of `u` and its long-run standard deviation.
    fn begin_monitoring(&self, history: &[Vec<f64>]) -> Monitoring {
        let m = history.len();
        let gamma = self
            .cfg
            .bandwidth
            .unwrap_or_else(|| ((m as f64).ln().floor() as usize).max(1));
        let (mut level, mut scale) = (Vec::new(), Vec::new());
        if self.cfg.scalar {
            let u: Vec<f64> = history.iter().map(|r| r[0]).collect();
            level.push(u.iter().sum::<f64>() / m as f64);
            scale.push(Self::scalar_long_run_sd(&u, gamma));
        } else {
            let d = self.cfg.n_features;
            for a in 0..d {
                for b in (a + 1)..d {
                    level.push(Self::corr_of(history, m, a, b));
                    let sd = Self::long_run_sd(history, a, b, gamma);
                    scale.push(if sd > 0.0 { sd } else { f64::NAN });
                }
            }
        }
        let width = self.cfg.width();
        let pairs = if self.cfg.scalar { 0 } else { level.len() };
        // Under a clock span each cycle has its own `m`, and its own `K`:
        // `monitor_rows`, or the period planned at the history's rate, `K =
        // ⌊m·monitor_span/window_size⌋` (the crossing probability over a
        // shorter period is smaller, so a period the clock ends sooner keeps
        // the size at most `alpha`, however its rows arrive).
        let (history, horizon) = match (self.cfg.span_clock, self.cfg.monitor_clock) {
            (None, _) => (0, 0),
            (Some(d), Some(span)) => (m as u64, ((m as f64 * span / d).floor() as u64).max(1)),
            (Some(_), None) => (m as u64, self.cfg.monitor_rows as u64),
        };
        let mut mon = Monitoring {
            level,
            scale,
            rows: Vec::new(),
            mean: vec![0.0; width],
            m2: vec![0.0; width],
            cross: vec![0.0; pairs],
            history,
            horizon,
            z_level: Vec::new(),
            crit: f64::NAN,
        };
        mon.rebuild(&self.cfg);
        mon
    }

    /// `sequential`'s detector on the row that would be the `k`-th
    /// monitored one, over the pairs, and the pair it is largest at
    /// (docs/PLAN.md task 553, after the review of 2026-11-22): Wied &
    /// Galeano's comparison of the monitored rows' correlation with the
    /// history's, each on Fisher's scale at its own finite-sample variance,
    ///
    /// ```text
    /// V_k = ((k − 3)⁺/√(m − 3))·|z_k − z_h|·(1 − ρ̂_h²)/σ̂,   z = atanh ρ̂
    /// ```
    ///
    /// against their boundary `c·w(b)`, `w(b) = (1 + b)(b/(1 + b))^γ` (their
    /// Eq. 5) at the information `b = (k − 3)/(m − 3)`. `z_k` is the
    /// correlation of the monitored rows with this one, from the mean-form
    /// sums a step on; `z_h` and `ρ̂_h` the history's, `σ̂` the history's
    /// long-run standard deviation of `ρ̂` (the inverse of their `D̂`), so
    /// `σ̂/(1 − ρ̂_h²)` is that of `z_h`. With `Var z_k = σ_z²/(k − 3)` and
    /// `Var z_h = σ_z²/(m − 3)` the detector's variance is `b + b²`, W&G's
    /// `G(b)` exactly and not only as `m → ∞`. Their own detector, `(k/√m)
    /// |ρ̂_k − ρ̂_h|/σ̂` from `k = 2`, compared a correlation of two rows,
    /// `±1` whatever the data, with the history's: at `γ = 0.45` and
    /// `ρ = 0.6` a quarter to a third of the cycles flagged on their first
    /// rows with nothing changed, at every history up to 1000. A row of no
    /// information, `k ≤ 3`, has weight 0 and no verdict: NaN, as the first
    /// monitored row was; so is a history of `m ≤ 3`.
    ///
    /// `scalar`'s is the mean's: `(k/√m)·|ū_k − ū_h|/σ̂_u / w(k/m)`, which
    /// forms no correlation to converge.
    fn sequential_stat(&self, mon: &Monitoring, row: &[f64]) -> (f64, usize) {
        if !self.informed(mon) {
            return (f64::NAN, 0);
        }
        let k = mon.rows.len() + 1;
        let kf = k as f64;
        let (m, _) = mon.rows_of(&self.cfg);
        let m = m as f64;
        let gamma = self.cfg.boundary_gamma;
        let boundary = |b: f64| (1.0 + b) * (b / (1.0 + b)).powf(gamma);
        let width = mon.mean.len();
        let dev: Vec<f64> = (0..width).map(|i| row[i] - mon.mean[i]).collect();
        let mean: Vec<f64> = (0..width).map(|i| mon.mean[i] + dev[i] / kf).collect();
        if self.cfg.scalar {
            let v = (kf / m.sqrt()) * (mean[0] - mon.level[0]) / mon.scale[0];
            return (v.abs() / boundary(kf / m), 0);
        }
        // The information each side holds, `n − 3` on Fisher's scale.
        let (info, info_h) = (kf - 3.0, m - 3.0);
        let w = boundary(info / info_h);
        let factor = info / info_h.sqrt();
        let d = self.cfg.n_features;
        let (mut best, mut at) = (f64::NAN, 0);
        let mut idx = 0;
        for a in 0..d {
            let m2a = mon.m2[a] + dev[a] * (row[a] - mean[a]);
            for bb in (a + 1)..d {
                let m2b = mon.m2[bb] + dev[bb] * (row[bb] - mean[bb]);
                let c = mon.cross[idx] + dev[a] * (row[bb] - mean[bb]);
                // A correlation's rounding can put it a hair past `±1`, where
                // `atanh` has no value: it is `±1` there, `z` infinite, and a
                // period whose rows are collinear where the history's were
                // not has changed.
                let rho = crate::window_moments::pair_corr(c, m2a, m2b).clamp(-1.0, 1.0);
                let level = mon.level[idx];
                let z = libm::atanh(rho) - mon.z_level[idx];
                let ratio = (factor * z.abs() * (1.0 - level * level) / mon.scale[idx]) / w;
                if !ratio.is_nan() && (best.is_nan() || ratio > best) {
                    (best, at) = (ratio, idx);
                }
                idx += 1;
            }
        }
        (best, at)
    }

    /// Whether the row that would be the next monitored one carries
    /// information for the detector: for a pair's Fisher `z`, `k − 3 > 0`
    /// and `m − 3 > 0` (its variance `1/(n − 3)` is defined); for
    /// `scalar`'s mean, a second row.
    fn informed(&self, mon: &Monitoring) -> bool {
        let k = mon.rows.len() + 1;
        if self.cfg.scalar {
            return k >= 2;
        }
        let (m, _) = mon.rows_of(&self.cfg);
        k > 3 && m > 3
    }

    /// W&G's Eq. 8: on a flag at the `τ`-th monitored row, the change is
    /// dated to `k̂ = argmax_{j ≤ τ−1} j·|ρ̂_j − ρ̂_{τ−1}|`, `ρ̂_j` the
    /// correlation of the first `j` monitored rows (for `scalar`, the mean
    /// of `u`), over the rows before the flag -- the history does not enter,
    /// which they found distorted the estimate -- and the positive factor
    /// `D̂/√τ` left out of an argmax. The output is `τ − k̂`, the rows from
    /// the first changed one through the flag's; NaN below two rows.
    fn sequential_since_change(&self, mon: &Monitoring, at: usize) -> f64 {
        let n = mon.rows.len();
        let prefix: Vec<f64> = if self.cfg.scalar {
            let mut run = 0.0;
            mon.rows
                .iter()
                .enumerate()
                .map(|(j, r)| {
                    run += r[0];
                    run / (j + 1) as f64
                })
                .collect()
        } else {
            let (a, b) = Self::pair_at(self.cfg.n_features, at);
            let (mut ma, mut mb, mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            mon.rows
                .iter()
                .enumerate()
                .map(|(j, r)| {
                    let jf = (j + 1) as f64;
                    let (da, db) = (r[a] - ma, r[b] - mb);
                    ma += da / jf;
                    mb += db / jf;
                    saa += da * (r[a] - ma);
                    sbb += db * (r[b] - mb);
                    sab += da * (r[b] - mb);
                    crate::window_moments::pair_corr(sab, saa, sbb)
                })
                .collect()
        };
        let Some(&last) = prefix.last() else {
            return f64::NAN;
        };
        let (mut best, mut k_hat) = (f64::NEG_INFINITY, None);
        for (j, p) in prefix.iter().enumerate() {
            let score = (j + 1) as f64 * (p - last).abs();
            if score.is_finite() && score > best {
                (best, k_hat) = (score, Some(j + 1));
            }
        }
        match (last.is_finite(), k_hat) {
            (true, Some(k)) => (n + 1 - k) as f64,
            _ => f64::NAN,
        }
    }

    /// `sequential`: a learned row enters the history, or the monitoring
    /// period. A full history starts the monitoring; a flag or the
    /// period's last row ends it, and the next learned row starts a history.
    fn advance_sequential(&mut self, row: Vec<f64>, flagged: bool, key: Option<Stamp>) {
        let scalar = self.cfg.scalar;
        let Some(mon) = self.monitoring.as_mut() else {
            self.ring.push_back(row);
            // The history's last row: its count, or under a clock span the
            // row that reaches `window_size` from its first.
            let full = match (self.cfg.span_clock, key) {
                (Some(span), Some(now)) => {
                    let first = *self.first.get_or_insert(now);
                    crate::clock_span::span_reached(now, first, span)
                }
                _ => self.ring.len() >= self.cfg.span_rows,
            };
            if full {
                let history: Vec<Vec<f64>> = self.ring.drain(..).collect();
                self.monitoring = Some(self.begin_monitoring(&history));
                self.first = None;
            }
            return;
        };
        let (_, limit) = mon.rows_of(&self.cfg);
        // The period's own span of the clock, from its first row.
        let spanned = match (self.cfg.monitor_clock, key) {
            (Some(span), Some(now)) => {
                let first = *self.first.get_or_insert(now);
                crate::clock_span::span_reached(now, first, span)
            }
            _ => false,
        };
        // The same arithmetic as `sequential_stat`'s step, now kept.
        let kf = (mon.rows.len() + 1) as f64;
        let width = mon.mean.len();
        let dev: Vec<f64> = (0..width).map(|i| row[i] - mon.mean[i]).collect();
        for i in 0..width {
            mon.mean[i] += dev[i] / kf;
            mon.m2[i] += dev[i] * (row[i] - mon.mean[i]);
        }
        if !scalar {
            // The pairs in `level`'s order: `a` outer, `b > a` inner.
            let Monitoring { cross, mean, .. } = &mut *mon;
            let mut cross = cross.iter_mut();
            for (a, &da) in dev.iter().enumerate() {
                for (xb, mb) in row[a + 1..].iter().zip(&mean[a + 1..]) {
                    *cross.next().expect("one co-moment a pair") += da * (xb - mb);
                }
            }
        }
        mon.rows.push(row);
        if flagged || mon.rows.len() >= limit || spanned {
            self.monitoring = None;
            self.first = None;
        }
    }

    /// The row standardised by the diag, for `scalar`.
    fn scalarise(&self, x: &[f64]) -> f64 {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for (i, xi) in x.iter().enumerate() {
            let v = self.diag.var(i);
            if v.is_nan() || v <= 0.0 {
                return f64::NAN;
            }
            let r = self.diag.deviation(i, *xi) / v.sqrt();
            s1 += r;
            s2 += r * r;
        }
        let n = x.len() as f64;
        // A sum of squares past the doubles (a standardised value past
        // `1e154`): the sums again at a power of two of their own, which the
        // estimate, a ratio of the squares' degree, does not read. It was
        // NaN, and the row left out of the CUSUM (review 2026-11-23, COV-5).
        if !s2.is_finite() {
            // Not where a variance is below the normal doubles: a history
            // the decay all but forgot leaves one there, with its digits
            // gone, and the row is read as it was, with no estimate.
            if (0..x.len()).any(|i| self.diag.var(i) < f64::MIN_POSITIVE) {
                return f64::NAN;
            }
            let r = |i: usize| self.diag.deviation(i, x[i]) / self.diag.var(i).sqrt();
            let top = (0..x.len()).map(|i| r(i).abs()).fold(0.0, f64::max);
            if !(top.is_finite() && top > 0.0) {
                return f64::NAN;
            }
            let (_, e) = crate::held_median::frexp(top);
            let c = crate::sqsum::ldexp(1.0, -e);
            (s1, s2) = (0.0, 0.0);
            for i in 0..x.len() {
                let ri = r(i) * c;
                s1 += ri;
                s2 += ri * ri;
            }
        }
        if s2 > 0.0 {
            (s1 * s1 - s2) / ((n - 1.0) * s2)
        } else {
            f64::NAN
        }
    }

    /// `‖vech(R̂_pre − R̂_post)‖` over the two windows in `rows`.
    #[cfg(test)]
    fn window_stat(&self, rows: &[Vec<f64>]) -> f64 {
        let w = self.cfg.span_rows;
        if rows.len() < 2 * w {
            return f64::NAN;
        }
        self.window_stat_by(w, w, |i| &rows[i], &mut Vec::new())
    }

    /// [`Self::window_corr_into`] into a buffer of its own.
    #[cfg(test)]
    fn window_corr<'a>(&self, w: usize, row: impl Fn(usize) -> &'a [f64]) -> Vec<f64> {
        let d = self.cfg.n_features;
        let mut buf = vec![0.0; 2 * d + d * d.saturating_sub(1) / 2];
        self.window_corr_into(w, row, &mut buf);
        buf
    }

    /// The window's correlation matrix, the strict upper triangle in the
    /// pairs' order, of its `w` rows `row(0)..row(w)`: [`Self::corr_of`]'s
    /// sums, in its order, with each column's mean and squared deviations
    /// summed once rather than once for each pair it is in. The same bits.
    /// Into `buf`, `2 d + d (d − 1) / 2` long, written over: the means, the
    /// squared deviations, then the correlations, which start at `2 d`.
    fn window_corr_into<'a>(&self, w: usize, row: impl Fn(usize) -> &'a [f64], buf: &mut [f64]) {
        let d = self.cfg.n_features;
        let wf = w as f64;
        buf.fill(0.0);
        let (mean, rest) = buf.split_at_mut(d);
        let (ss, corr) = rest.split_at_mut(d);
        for i in 0..w {
            for (m, x) in mean.iter_mut().zip(row(i)) {
                *m += x;
            }
        }
        mean.iter_mut().for_each(|m| *m /= wf);
        // The pairs over slices of the row, the means and the sums, so no
        // entry is indexed and checked; each sum adds the same terms in the
        // same order. Taking each row's deviations once into a buffer first
        // cost 13-14% on arm64 (an Apple M-series core, both code layouts)
        // where it saved 10% on x86-64; this form saved 9% and 6% on arm64.
        for i in 0..w {
            let r = &row(i)[..d];
            let mut p = 0;
            for a in 0..d {
                let da = r[a] - mean[a];
                ss[a] += da * da;
                let rest = d - a - 1;
                for ((c, &rb), &mb) in corr[p..p + rest]
                    .iter_mut()
                    .zip(&r[a + 1..])
                    .zip(&mean[a + 1..])
                {
                    *c += da * (rb - mb);
                }
                p += rest;
            }
        }
        Self::normalise_pairs(d, ss, corr);
    }

    /// `‖vech(R̂_pre − R̂_post)‖` over two windows of `n1` and `n2` rows,
    /// `row(0)..row(n1)` and `row(n1)..row(n1 + n2)`, read where they are:
    /// the ring was copied whole on every row, and every pooled row on every
    /// permutation draw (docs/PLAN.md task 270, D-2). Two windows of
    /// `window_rows` each, or under `window_size` the rows each window of
    /// the clock holds (task 553); a window without two rows has no
    /// correlation, and so no statistic.
    fn window_stat_by<'a>(
        &self,
        n1: usize,
        n2: usize,
        row: impl Fn(usize) -> &'a [f64],
        buf: &mut Vec<f64>,
    ) -> f64 {
        if n1 == 0 || n2 == 0 {
            return f64::NAN;
        }
        let d = self.cfg.n_features;
        let at = 2 * d;
        // Both windows' correlations, in a buffer kept between rows.
        let len = at + d * d.saturating_sub(1) / 2;
        buf.clear();
        buf.resize(2 * len, 0.0);
        let (pre, post) = buf.split_at_mut(len);
        self.window_corr_into(n1, &row, pre);
        self.window_corr_into(n2, |i| row(n1 + i), post);
        self.change_between(pre, post)
    }

    /// `‖vech(R̂_pre − R̂_post)‖` from the two windows' buffers, each as
    /// [`Self::window_corr_into`] leaves it.
    fn change_between(&self, pre: &[f64], post: &[f64]) -> f64 {
        let at = 2 * self.cfg.n_features;
        let mut acc = 0.0f64;
        for (x, y) in pre[at..].iter().zip(&post[at..]) {
            if !(x.is_finite() && y.is_finite()) {
                return f64::NAN;
            }
            let diff = (x - y).abs();
            match self.cfg.norm {
                ChangeNorm::L1 => acc += diff,
                ChangeNorm::LInf => acc = acc.max(diff),
            }
        }
        acc
    }

    /// [`Self::window_stat_by`] over the rows `pool` holds at the places
    /// `at` gives, the first `n1` of them one window and the rest the other,
    /// `d` numbers to a row in `pool`: the pooled rows of a permutation
    /// draw, copied from the ring once for every draw and not once for each.
    fn window_stat_flat(&self, pool: &[f64], at: &[usize], n1: usize, buf: &mut Vec<f64>) -> f64 {
        let d = self.cfg.n_features;
        if n1 == 0 || n1 >= at.len() {
            return f64::NAN;
        }
        let len = 2 * d + d * d.saturating_sub(1) / 2;
        buf.clear();
        buf.resize(2 * len, 0.0);
        let (pre, post) = buf.split_at_mut(len);
        let (head, tail) = at.split_at(n1);
        Self::window_corr_flat(d, pool, head, pre);
        Self::window_corr_flat(d, pool, tail, post);
        self.change_between(pre, post)
    }

    /// [`Self::window_corr_into`] over the rows of `pool`, `d` numbers to
    /// a row, at the places `at` gives, the same sums in the same order. For
    /// up to eight columns the loops over the pairs have a length known at
    /// compile time, so they unroll and each sum stays in a register.
    fn window_corr_flat(d: usize, pool: &[f64], at: &[usize], buf: &mut [f64]) {
        let wf = at.len() as f64;
        buf.fill(0.0);
        let (mean, rest) = buf.split_at_mut(d);
        let (ss, corr) = rest.split_at_mut(d);
        macro_rules! fixed {
            ($($n:literal)+) => {
                match d {
                    $($n => Self::co_moments_fixed::<$n>(pool, at, wf, mean, ss, corr),)+
                    _ => {
                        for &i in at {
                            for (m, x) in mean.iter_mut().zip(&pool[i * d..(i + 1) * d]) {
                                *m += x;
                            }
                        }
                        mean.iter_mut().for_each(|m| *m /= wf);
                        Self::co_moments_rows(d, pool, at, mean, ss, corr);
                    }
                }
            };
        }
        fixed!(1 2 3 4 5 6 7 8);
        Self::normalise_pairs(d, ss, corr);
    }

    /// The means, the sums of squared deviations and of cross products of
    /// the rows of `pool`, `D` numbers to a row, at the places `at` gives
    /// (`wf` of them): the means first, each a sum in the order of `at`
    /// over `wf`, then the deviations about them. The pool is read as rows
    /// of `D`, so a row is one bounds check.
    fn co_moments_fixed<const D: usize>(
        pool: &[f64],
        at: &[usize],
        wf: f64,
        mean: &mut [f64],
        ss: &mut [f64],
        corr: &mut [f64],
    ) {
        let rows = pool.as_chunks::<D>().0;
        let mut m = [0.0f64; D];
        for &i in at {
            let r = &rows[i];
            for a in 0..D {
                m[a] += r[a];
            }
        }
        for x in &mut m {
            *x /= wf;
        }
        // At most 28 pairs, at eight columns.
        let mut s = [0.0f64; D];
        let mut c = [0.0f64; 28];
        for &i in at {
            let r = &rows[i];
            let mut p = 0;
            for a in 0..D {
                let da = r[a] - m[a];
                s[a] += da * da;
                for b in (a + 1)..D {
                    c[p] += da * (r[b] - m[b]);
                    p += 1;
                }
            }
        }
        mean.copy_from_slice(&m);
        ss.copy_from_slice(&s);
        corr.copy_from_slice(&c[..D * (D - 1) / 2]);
    }

    /// [`Self::co_moments_fixed`] for any number of columns.
    fn co_moments_rows(
        d: usize,
        pool: &[f64],
        at: &[usize],
        mean: &[f64],
        ss: &mut [f64],
        corr: &mut [f64],
    ) {
        for &i in at {
            let r = &pool[i * d..(i + 1) * d];
            let mut p = 0;
            for a in 0..d {
                let da = r[a] - mean[a];
                ss[a] += da * da;
                let rest = d - a - 1;
                for ((c, &rb), &mb) in corr[p..p + rest]
                    .iter_mut()
                    .zip(&r[a + 1..])
                    .zip(&mean[a + 1..])
                {
                    *c += da * (rb - mb);
                }
                p += rest;
            }
        }
    }

    /// Each cross sum over the root of the product of its two columns'
    /// squared deviations, NaN where that is not positive.
    fn normalise_pairs(d: usize, ss: &[f64], corr: &mut [f64]) {
        let mut p = 0;
        for a in 0..d {
            for b in (a + 1)..d {
                corr[p] = crate::window_moments::pair_corr(corr[p], ss[a], ss[b]);
                p += 1;
            }
        }
    }

    /// The critical value of the permutation test at `alpha`: the
    /// `(B + 1 − m)`-th smallest of the `B` finite draws of the statistic
    /// under a permutation of the pooled rows between the two windows, in
    /// blocks, with `m` = [`perm_rejections`], so that its size is at most
    /// `alpha`. `None` where `m` is 0: no statistic could flag.
    ///
    /// The pool is the ring, the first `n1` of its rows the first window:
    /// two windows of `window_rows`, or under `window_size` the rows each
    /// window of the clock holds. A permutation test needs no equal
    /// windows: on exchangeable rows every assignment of the `n1 + n2`
    /// pooled rows to windows of `n1` and `n2` is equally likely.
    fn permutation_crit(&mut self, n1: usize) -> Option<f64> {
        let n = self.ring.len();
        if n1 == 0 || n1 >= n {
            return None;
        }
        let block = self.cfg.perm_block;
        let nblocks = n.div_ceil(block);
        // The draws' buffers, kept between redraws.
        let mut work = std::mem::take(&mut self.work);
        let Work {
            report: buf,
            pool,
            stats,
            order,
            shuffled,
            ..
        } = &mut work;
        stats.clear();
        stats.reserve_exact(self.cfg.n_perm);
        order.clear();
        order.extend(0..nblocks);
        // The pooled rows, by their places in one run of numbers copied
        // from the ring, permuted in blocks.
        let d = self.cfg.n_features;
        pool.clear();
        pool.reserve_exact(n * d);
        for r in &self.ring {
            pool.extend_from_slice(&r[..d]);
        }
        shuffled.clear();
        shuffled.reserve_exact(n);
        for _ in 0..self.cfg.n_perm {
            // Fisher-Yates over the blocks, from the state's own generator
            // so the draws are a function of the data alone.
            for i in (1..nblocks).rev() {
                let j = below(&mut self.rng, i as u64 + 1) as usize;
                order.swap(i, j);
            }
            shuffled.clear();
            for &bi in order.iter() {
                let start = bi * block;
                shuffled.extend(start..(start + block).min(n));
            }
            let s = self.window_stat_flat(pool, shuffled, n1, buf);
            if s.is_finite() {
                stats.push(s);
            }
        }
        let crit = if stats.is_empty() {
            None
        } else {
            stats.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            // It was the `⌈(1 − alpha)·B⌉`-th smallest, whose size is `(B +
            // 1 − q) / (B + 1)`: 0.0952 at 20 draws and alpha 0.05, 0.0476
            // at 0.01 (task 278, batch 33 measured it; fixed in ff).
            let m = perm_rejections(self.cfg.alpha, stats.len());
            (m > 0).then(|| stats[stats.len() - m])
        };
        self.work = work;
        crit
    }

    /// The row as the ring would hold it, or `None` when the standardiser
    /// cannot yet make one (`scalar` before the first variances): `x` itself,
    /// or its one-column form in `one`. A row is read this way and copied
    /// only to enter the ring.
    fn ring_row<'a>(&self, x: &'a [f64], one: &'a mut [f64; 1]) -> Option<&'a [f64]> {
        if !self.cfg.scalar {
            return Some(x);
        }
        let u = self.scalarise(x);
        // A one-column ring: the CUSUM is over `u` itself.
        u.is_finite().then(|| {
            one[0] = u;
            &one[..]
        })
    }

    /// What this row produces, from the state as it stands: nothing except
    /// where a statistic is due.
    ///
    /// A pure function of the state and the row, so `step` and `predict`
    /// share it -- which is what makes `predict` the step's answer without
    /// the step even on the row that closes a span.
    /// A row's report: the statistic for the span **ending at this row**,
    /// the critical value in force, the flag and the rows since the last
    /// one. The row itself is always part of the span reported here -- it
    /// is reported before the update, which is what makes the flag out of
    /// sample -- so a row that will not be learned (`weight = 0`) is
    /// reported as if it would be, and then does not enter the ring, does
    /// not advance `since` for the rows after it and does not reset it if
    /// it flags. That is the reading `predict` has to give too, since
    /// `predict` does not know the weight (docs/REVIEW-E54-E64.md CC2).
    /// Under a clock form `key` is the row's place on the clock, read as a
    /// row that counts.
    fn read(&self, x: &[f64], key: Option<Stamp>) -> Vec<f64> {
        let mut pred = Vec::new();
        self.read_into(x, key, &mut pred, &mut Vec::new());
        pred
    }

    /// Whether the row keyed `key` closes the `monitor` span in progress:
    /// the span's last row by its count, or under a clock span the row that
    /// reaches `window_size` from the span's first (that row is the span's
    /// last, as every span of the clock that holds rows reads it).
    fn monitor_closes(&self, key: Option<Stamp>) -> bool {
        match (self.cfg.span_clock, key) {
            (Some(span), Some(now)) => self
                .first
                .is_some_and(|first| crate::clock_span::span_reached(now, first, span)),
            _ => self.ring.len() + 1 >= self.cfg.span_rows,
        }
    }

    /// The two windows a `window` report on the row keyed `key` reads, as
    /// `(skip, n1, n2)`: the ring's rows from `skip` on, the first `n1` the
    /// earlier window and the rest with the row the later, `n2` counting
    /// the row. `None` until both windows have run: `2·window_rows` rows
    /// with this one, or under `window_size` (docs/PLAN.md task 553) `2D`
    /// of the clock since the ring's first row, the windows then the rows
    /// of `(now − 2D, now − D]` and `(now − D, now]` (a row exactly `D` old
    /// is in the earlier window, one exactly `2D` old in neither, as Polars'
    /// `rolling_*_by` with `closed = "right"` reads a window).
    fn windows_at(&self, key: Option<Stamp>) -> Option<(usize, usize, usize)> {
        let held = self.ring.len();
        match (self.cfg.span_clock, key) {
            (Some(span), Some(now)) => {
                let first = self.first?;
                if !crate::clock_span::span_reached(now, first, 2.0 * span) {
                    return None;
                }
                let right = crate::WindowClosed::Right;
                let skip = self
                    .keys
                    .iter()
                    .take_while(|&&k| right.has_left(now, k, 2.0 * span))
                    .count();
                let split = skip
                    + self
                        .keys
                        .iter()
                        .skip(skip)
                        .take_while(|&&k| right.has_left(now, k, span))
                        .count();
                Some((skip, split - skip, held - split + 1))
            }
            _ => {
                let w = self.cfg.span_rows;
                (held + 1 >= 2 * w).then(|| (held + 1 - 2 * w, w, w))
            }
        }
    }

    /// The permutation critical value in force, for a report over windows
    /// of `n1` and `n2` rows: as drawn, or under `window_size` rescaled from
    /// the windows it was drawn over, `(n₁⁰, n₂⁰)`, by `√((1/n1 + 1/n2)/
    /// (1/n₁⁰ + 1/n₂⁰))`. A difference of two windows' correlations has a
    /// null spread of `(1 − ρ²)·√(1/n1 + 1/n2)` a pair, so the norm of the
    /// differences scales with that root: a value drawn at 100 rows a
    /// window and held while the rows come at half the rate is too small
    /// by `√2`, which turns 5% into 17%. On the row clock's windows the two
    /// are the same counts and the value is as drawn, to the bit.
    fn perm_crit_for(&self, n1: usize, n2: usize) -> Option<f64> {
        let c = self.perm_crit?;
        Some(match self.perm_counts {
            Some([a, b]) if (a, b) != (n1 as u64, n2 as u64) => {
                let spread = |p: f64, q: f64| 1.0 / p + 1.0 / q;
                c * (spread(n1 as f64, n2 as f64) / spread(a as f64, b as f64)).sqrt()
            }
            _ => c,
        })
    }

    /// [`Self::read`], written over `pred`, `buf` the window statistic's
    /// working space.
    fn read_into(&self, x: &[f64], key: Option<Stamp>, pred: &mut Vec<f64>, buf: &mut Vec<f64>) {
        pred.clear();
        pred.resize(Self::n_outputs_for(), f64::NAN);
        let mut one = [0.0];
        let Some(row) = self.ring_row(x, &mut one) else {
            return;
        };
        let since = self.since_flag.unwrap_or(0) + 1;
        match self.cfg.kind {
            CorrChangeKind::Monitor => {
                if !self.monitor_closes(key) {
                    return;
                }
                // The span's rows read where they are, the ring's and this
                // one, where each ring row was copied at every close.
                let rows: Vec<&[f64]> = self
                    .ring
                    .iter()
                    .map(Vec::as_slice)
                    .chain(std::iter::once(row))
                    .collect();
                let crit = self.cfg.monitor_crit(rows.len());
                // The largest statistic over the pairs; the flag from the
                // largest pair over the critical value whose scale ratio
                // held; and the flag withheld (NaN) where every pair over it
                // is one whose ratio moved, since there the correlation's
                // CUSUM is not on its null law ([`Self::monitor_pair`]).
                let (mut stat, mut flagged, mut at, mut held) = (f64::NAN, f64::NAN, 0, false);
                let mut work = Vec::new();
                if self.cfg.scalar {
                    let (q, j) = self.scalar_stat_at(&rows);
                    stat = q;
                    if q > crit {
                        (flagged, at) = (q, j);
                    }
                } else {
                    for a in 0..self.cfg.n_features {
                        for b in (a + 1)..self.cfg.n_features {
                            let p = self.monitor_pair(&rows, a, b, &mut work);
                            if p.stat.is_finite() && (stat.is_nan() || p.stat > stat) {
                                stat = p.stat;
                            }
                            if p.stat > crit {
                                if p.ratio > crit {
                                    held = true;
                                } else if flagged.is_nan() || p.stat > flagged {
                                    (flagged, at) = (p.stat, p.at);
                                }
                            }
                        }
                    }
                }
                let flag = flagged.is_finite() && crit.is_finite();
                pred[0] = stat;
                pred[1] = crit;
                pred[2] = if flag || !held {
                    f64::from(flag)
                } else {
                    f64::NAN
                };
                pred[3] = since as f64;
                if flag && at > 0 {
                    // The rows after the CUSUM's argmax, through this one.
                    pred[4] = (rows.len() - at) as f64;
                }
            }
            CorrChangeKind::Window => {
                // The ring and the row, less what is past the two windows,
                // read where they are.
                let Some((skip, n1, n2)) = self.windows_at(key) else {
                    return;
                };
                let held = self.ring.len();
                let ring = &self.ring;
                let stat = match &self.sums {
                    Some(sums) => self.window_stat_merged(sums, row, buf),
                    None => self.window_stat_by(
                        n1,
                        n2,
                        |i| {
                            let i = skip + i;
                            if i < held { &ring[i] } else { row }
                        },
                        buf,
                    ),
                };
                // The value **in force**: a redraw happens after the row is
                // reported, so `predict` and `step` see the same one.
                let crit = self.cfg.crit.or_else(|| self.perm_crit_for(n1, n2));
                let flag = matches!((stat.is_finite(), crit), (true, Some(c)) if stat > c);
                pred[0] = stat;
                pred[1] = crit.unwrap_or(f64::NAN);
                pred[2] = f64::from(flag);
                pred[3] = since as f64;
                if flag {
                    // The second window, by construction.
                    pred[4] = n2 as f64;
                }
            }
            CorrChangeKind::Sequential => {
                let Some(mon) = &self.monitoring else {
                    return;
                };
                // A row the detector has no information at reports nothing,
                // as the first monitored row always has: `k ≤ 3` for a
                // pair's Fisher `z`, `k = 1` for `scalar`'s mean.
                if !self.informed(mon) {
                    return;
                }
                let (stat, at) = self.sequential_stat(mon, row);
                let crit = mon.crit;
                let flag = !stat.is_nan() && crit.is_finite() && stat > crit;
                pred[0] = stat;
                pred[1] = crit;
                pred[2] = f64::from(flag);
                pred[3] = since as f64;
                if flag {
                    pred[4] = self.sequential_since_change(mon, at);
                }
            }
        }
    }

    /// Under `window_size`, the `window` kind's rows that have left both
    /// windows at the row keyed `now`, dropped: those `2D` or more old.
    fn trim_windows(&mut self, now: Stamp) {
        let Some(span) = self.cfg.span_clock else {
            return;
        };
        while self
            .keys
            .front()
            .is_some_and(|&k| crate::WindowClosed::Right.has_left(now, k, 2.0 * span))
        {
            self.keys.pop_front();
            if let Some(old) = self.ring.pop_front() {
                self.work.spare = old;
            }
        }
    }

    /// Forget every row held: a `window` ring, its keys and the critical
    /// value drawn from it.
    fn restart_windows(&mut self) {
        self.ring.clear();
        self.keys.clear();
        self.first = None;
        self.ring_rows = 0;
        self.perm_crit = None;
        self.perm_counts = None;
        self.since_perm = 0;
    }

    /// [`crate::OnlineModel::predict`] for a row whose place on the clock
    /// the caller knows, `stamp`, as [`crate::OnlineModel::stamp_next`]
    /// hands a step's. Without `window_size` (or on the row clock, whose
    /// keys count accepted rows) the stamp is not read.
    pub fn predict_at(&self, x: &[f64], d_clock: f64, stamp: Stamp) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let key = self
            .clock
            .as_ref()
            .map(|c| if c.on_rows { c.key(d_clock) } else { stamp });
        crate::Step {
            pred: self.read(x, key),
            n_eff: self.n_eff,
            extra: None,
        }
    }
}

impl crate::OnlineModel for CorrChange {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut crate::Step,
    ) {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a row with a weight was counted in `n_eff`, and
        // fed to the scalar statistic's moments, whatever its features
        // held (docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        // The row's place on a clock form's clock, read as a row that
        // counts: a row of weight 0 is reported as one would be.
        let key = self.clock.as_ref().map(|c| c.key(d_clock));
        // `predict`'s answer, read before the row: `x` is usable here. Only
        // the window kind's report reads the working space; the others are
        // given an empty one, which allocates nothing, rather than pay to
        // take the kept one and put it back on every row (task 345).
        if matches!(self.cfg.kind, CorrChangeKind::Window) {
            let mut buf = std::mem::take(&mut self.work.report);
            self.read_into(x, key, &mut out.pred, &mut buf);
            self.work.report = buf;
        } else {
            self.read_into(x, key, &mut out.pred, &mut Vec::new());
        }
        out.n_eff = self.n_eff;
        out.extra = None;
        // Check the state cap atomically before advancing clocks, trimming the
        // window, or decaying running state.
        if self.over.is_some() {
            if let Some(s) = self.since_draw.as_mut() {
                s.discard_next();
            }
            if let Some(c) = self.clock.as_mut() {
                c.next = None;
            }
            return;
        }
        if weight > 0.0
            && let Some(now) = key
        {
            let rows = self.ring.len() - self.dropped_at(now)
                + self.monitoring.as_ref().map_or(0, |m| m.rows.len())
                + 1;
            let (bytes, cap) = (self.held_bytes_for(rows), self.cap());
            if !self.held_cap().admit(bytes, cap) {
                if let Some(s) = self.since_draw.as_mut() {
                    s.discard_next();
                }
                if let Some(c) = self.clock.as_mut() {
                    c.next = None;
                }
                return;
            }
        }
        let lam = self.cfg.decay.factor(d_clock);
        if let Some(s) = self.since_draw.as_mut() {
            s.step(d_clock);
        }
        // A row of weight 0 counts in no clock (hard rule 9): on the row
        // clock it is no step, and it is in no span or window.
        if let Some(c) = self.clock.as_mut() {
            c.step(d_clock, weight > 0.0);
        }
        if weight <= 0.0 {
            // Hard rule 8: `n_eff` is the accumulated weight before the
            // row's update and before its own decay, and a zero-weight row
            // still decays it (docs/REVIEW-E54-E64.md H3).
            self.n_eff *= lam;
            if self.cfg.scalar {
                self.diag.update(x, lam, 0.0);
            }
            return;
        }
        if let Some(now) = key
            && self.cfg.kind == CorrChangeKind::Window
        {
            self.trim_windows(now);
        }
        // The row as the ring holds it, copied into the row the window's
        // ring let go last, so the copy allocates nothing.
        let mut one = [0.0];
        let row = self.ring_row(x, &mut one).map(|r| self.work.ring_row(r));
        if self.cfg.scalar {
            self.diag.update(x, lam, weight);
        }
        self.n_eff = lam * self.n_eff + weight;
        let Some(row) = row else { return };
        self.since_flag = Some(self.since_flag.unwrap_or(0) + 1);
        let flagged = out.pred[2] == 1.0;
        if flagged {
            self.since_flag = Some(0);
        }
        match self.cfg.kind {
            CorrChangeKind::Sequential => self.advance_sequential(row, flagged, key),
            CorrChangeKind::Monitor => {
                let closes = self.monitor_closes(key);
                if self.ring.is_empty() {
                    self.first = key;
                }
                self.ring.push_back(row);
                if closes {
                    // Spans are disjoint by construction: the next row
                    // starts a new one, so `reset` has nothing to add here.
                    // The span's rows are kept for the next span's.
                    self.work.rows.extend(self.ring.drain(..));
                    self.first = None;
                }
            }
            CorrChangeKind::Window => {
                if self.ring.is_empty() {
                    self.first = key;
                }
                self.ring.push_back(row);
                if let Some(now) = key {
                    self.keys.push_back(now);
                }
                self.ring_rows += 1;
                if key.is_none() {
                    let w = self.cfg.span_rows;
                    while self.ring.len() > 2 * w {
                        if let Some(old) = self.ring.pop_front() {
                            self.work.spare = old;
                        }
                    }
                }
                // The windows a draw is taken over: both full, as a report
                // reads them at this row's place.
                let drawn = match (key, self.cfg.span_clock) {
                    (Some(now), Some(span)) => self.first.is_some_and(|first| {
                        crate::clock_span::span_reached(now, first, 2.0 * span)
                    }),
                    _ => self.ring.len() >= 2 * self.cfg.span_rows,
                };
                if flagged && self.cfg.reset {
                    self.restart_windows();
                } else if self.cfg.crit.is_none() && drawn {
                    // The critical value is refreshed **after** the row is
                    // reported, so `predict` and `step` see the one in
                    // force and cannot disagree. Counted before it is
                    // compared, so a value is in force for exactly
                    // `permute_every_rows` reports: compared first, it stood
                    // for one more, and `1` redrew every other row (review
                    // 2026-10-06, CD11; docs/PLAN.md task 196, S1).
                    self.since_perm = self.since_perm.saturating_add(1);
                    if self.perm_crit.is_none() || self.redraw_due() {
                        // The earlier window's rows at this row's place: the
                        // ring holds both windows, its aged rows gone.
                        let n1 = match (key, self.cfg.span_clock) {
                            (Some(now), Some(span)) => self
                                .keys
                                .iter()
                                .take_while(|&&k| crate::WindowClosed::Right.has_left(now, k, span))
                                .count(),
                            _ => self.cfg.span_rows,
                        };
                        self.perm_crit = self.permutation_crit(n1);
                        self.perm_counts = key
                            .map(|_| [n1 as u64, (self.ring.len() - n1) as u64])
                            .filter(|_| self.perm_crit.is_some());
                        self.since_perm = 0;
                        if let Some(s) = self.since_draw.as_mut() {
                            s.restart();
                        }
                    }
                }
                self.ready_sums();
            }
        }
        if key.is_some() {
            let (bytes, cap) = (self.held_bytes(0), self.cap());
            self.held_cap().note(bytes, cap);
        }
    }

    /// The redraw's clock reads the row's stamp (task 538), and a clock
    /// form's spans (task 553).
    fn stamp_next(&mut self, stamp: crate::Stamp) {
        if let Some(s) = self.since_draw.as_mut() {
            s.stamp_next(stamp);
        }
        if let Some(c) = self.clock.as_mut() {
            c.next = Some((stamp, false));
        }
    }

    /// On the row clock the redraw's clock counts from a row before the
    /// first, as every cadence does; a clock form's span counts the rows
    /// that counted.
    fn stamp_next_on_rows(&mut self, stamp: crate::Stamp) {
        if let Some(s) = self.since_draw.as_mut() {
            s.stamp_next_on_rows(stamp);
        }
        if let Some(c) = self.clock.as_mut() {
            c.next = Some((stamp, true));
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cap()))
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> crate::Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let key = self.clock.as_ref().map(|c| c.key(d_clock));
        crate::Step {
            pred: self.read(x, key),
            n_eff: self.n_eff,
            extra: None,
        }
    }

    fn clear_lags(&mut self) {
        // The rings pair rows the clock says are no longer adjacent; a
        // `monitor` span that spans a break is not a span.
        self.restart_windows();
        self.ready_sums();
        // And a `sequential` cycle across a break is not one cycle.
        self.monitoring = None;
    }

    fn state(&self) -> crate::State {
        crate::State::new(crate::ModelState::CorrChange(Box::new(self.clone())))
    }

    fn restore(s: &crate::State) -> Result<Self, crate::StateError> {
        crate::check_schema(s)?;
        match &s.model {
            crate::ModelState::CorrChange(m) => {
                let mut m = (**m).clone();
                // First, so nothing below is read from a cfg `new` refuses:
                // the sequential critical value is solved from it.
                crate::model::check_cfg("corrchange", m.cfg.validate())?;
                // The diagnostics and every ring row at the cfg's width
                // (review 2026-09-18, B3); a `scalar` ring holds `u`.
                let d = m.cfg.n_features;
                let width = m.cfg.width();
                let on_clock = m.cfg.span_clock.is_some();
                let wrong_mon = m.monitoring.as_ref().is_some_and(|mon| {
                    let values = if m.cfg.scalar { 1 } else { m.cfg.npairs() };
                    let pairs = if m.cfg.scalar { 0 } else { values };
                    mon.level.len() != values
                        || mon.scale.len() != values
                        || mon.mean.len() != width
                        || mon.m2.len() != width
                        || mon.cross.len() != pairs
                        || mon.rows.iter().any(|r| r.len() != width)
                        || (mon.history == 0) == on_clock
                });
                // A window's ring was numbered from 66 on; before, it is
                // numbered from its oldest row, a numbering as good as any
                // other (the sums it fixes are each a window's own).
                let window = m.cfg.kind == CorrChangeKind::Window;
                if window && s.schema_version < 66 {
                    m.ring_rows = m.ring.len() as u64;
                }
                let wrong_rows = if window {
                    m.ring_rows < m.ring.len() as u64
                } else {
                    m.ring_rows != 0
                };
                // A clock form's clock is kept with it, and only there; a
                // clock window keys every ring row; the counts a value was
                // drawn over, only there.
                let wrong_clock = m.clock.is_some() != on_clock
                    || m.keys.len() != if window && on_clock { m.ring.len() } else { 0 }
                    || (m.perm_counts.is_some() && !(window && on_clock && m.perm_crit.is_some()))
                    // The clock since the last draw, exactly where the cfg
                    // draws by the clock (review 2026-11-23, REGIME-1).
                    || m.since_draw.is_some() != m.cfg.permute_every.is_some();
                if m.diag.k() != d
                    || m.ring.iter().any(|r| r.len() != width)
                    || wrong_mon
                    || wrong_rows
                    || wrong_clock
                {
                    return Err(crate::model::damaged(
                        "corrchange",
                        "its saved values have the wrong shape",
                    ));
                }
                // Configuration, not state: recomputed on load.
                m.seq_crit = Self::sequential_crit_of(&m.cfg);
                if let Some(mon) = m.monitoring.as_mut() {
                    mon.rebuild(&m.cfg);
                }
                // A cache of the ring: rebuilt from it.
                m.sums = Self::sums_of(&m.cfg);
                m.ready_sums();
                Ok(m)
            }
            other => Err(crate::StateError::WrongModel {
                expected: "corrchange",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        Self::n_outputs_for()
    }
}

#[cfg(test)]
#[path = "corrchange_span_tests.rs"]
mod span_tests;

#[cfg(test)]
#[path = "corrchange_clock_tests.rs"]
mod clock_tests;

#[cfg(test)]
#[path = "corrchange_tests.rs"]
mod tests;
