//! Stochastic gradient descent with pluggable losses (docs/ENHANCEMENTS.md E16).
//!
//! The cheap baseline: one gradient step per row, no solves, O(k) per row rather
//! than O(k²). Also the only model here that handles **count targets**, via the
//! Poisson loss with a log link — none of the exact solvers cover those.
//!
//! Every loss shares the same shape. With `eta = z·b` the linear predictor,
//! `p = link(eta)` the prediction, and `d = dL/d(eta)`:
//!
//! | loss | link | `p` | `d` |
//! |---|---|---|---|
//! | `Squared` | identity | `eta` | `p − y` |
//! | `Huber` | identity | `eta` | `clamp(p − y, ±delta·s)` |
//! | `Quantile` | identity | `eta` | `1{y < p} − tau` |
//! | `EpsilonInsensitive` | identity | `eta` | `0` if `|p − y| ≤ eps·s_y`, else `sign(p − y)` |
//! | `Poisson` | log | `exp(clamp(eta, −30, 30))` | `p − y` |
//! | `Logistic` | sigmoid | `sigmoid(eta)` | `p − clamp(y, 0, 1)` |
//!
//! then `g_i = d · z_i · w + l2 · b_i` for a slope and `g_0 = d · w` for the
//! intercept, which is not penalised; then `b_i -= lr_i · g_i`, each `g_i`
//! clipped at `±clip_gradient` first. Without `standardize`, `z = [1, x]`
//! (`x` without an intercept) and `b` is in the caller's units. The Poisson
//! link clamps its exponent at `±30`, so a rate is between `e^−30 ≈ 9.4e-14`
//! and `e^30 ≈ 1.1e13`, and a linear predictor past either end predicts as
//! that end does: the prediction and its gradient stay finite, where `exp`
//! alone overflows past 709.
//!
//! **`standardize`** takes the step on the row standardized against the
//! features' EW moments with the row admitted at one row's weight of the
//! stream (the row's map `M`; [`SgdCfg::standardize`] says why the row is
//! in, and *Learning rates* below at what weight): `z = M x̃`,
//! `x̃ = [1, x]`, `z_0 = 1` and `z_i = (x_i − m_i) / s_i`, `m_i` and `s_i` the
//! mean and standard deviation, or without an intercept `z_i = x_i / s_i`,
//! `s_i` the root of the raw second moment, uncentred (review 2026-09-12,
//! C13); a feature with no spread yet has `s_i = 1`. The fit goes through
//! two phases ([`crate::Warmup`]; docs/PLAN.md task 206):
//!
//! - **While the scaler warms up**, until Kish's count of the rows it has
//!   learned, undecayed, reaches 22: `b` is in the coordinates of the
//!   moments, `eta = z·b`, the step is the one above on `z`, and the
//!   coefficients are read out with the moments as they stand, `b_i / s_i`
//!   and `b_0 − Σ_i m_i b_i / s_i`. Over the first rows the moments rest on
//!   almost nothing, and a step against a scale a few rows old is read again
//!   at the next scale and forgotten as it settles.
//! - **On the row the count reaches 22** the coefficients are read out so
//!   once and held in the caller's units from then on; **after it**:
//!
//! ```text
//! eta  = x̃·b                                  the prediction reads no moment
//! g_i  = d · z_i · w + l2 · s_i b_i,  g_0 = d · w      (each clipped)
//! Δβ_i = −lr_i · g_i                           the step in the row's coordinates
//! Δb   = Mᵀ Δβ:   Δb_i = Δβ_i / s_i,   Δb_0 = Δβ_0 − Σ_i m_i Δb_i
//! b   += Δb
//! ```
//!
//! `x̃ᵀΔb = zᵀΔβ`, so the step moves the row's own prediction as the
//! standardized step does, and one learning rate suits every feature's
//! units, the scaler's purpose (docs/PLAN.md task 195, U2). A prediction is
//! `x̃ᵀb` and never moves because the scaler did. Read through the moments
//! as they stood, every prediction moved with the scaler's own wander under
//! a finite half-life -- the EW mean of a unit-variance feature has standard
//! deviation `sqrt((1 − λ) / (1 + λ))`, 0.083 at a half-life of 50 -- 248
//! noise variances of out-of-sample error at R² 0.99998 and a half-life of
//! 50 where the unstandardized fit paid 0.010 (review round 5, G1). A step
//! whose `Δb` is not finite -- a feature of all but no spread, `Δβ_i / s_i`
//! overflowing -- teaches nothing, as a row whose gradient is not finite
//! does (below). The ridge is on the slope in the row's coordinates, `s_i
//! b_i`, as it was on `b_i` while the scaler warmed up.
//!
//! **`delta` is in units of `s`**, the target's residual scale as the row
//! arrives, as `huber`'s `huber_delta` is (`crate::Robust`), in the scale
//! `huber_scale` names ([`crate::HuberScale`]): the EW median of `|y − p|`
//! over `Φ⁻¹(3/4)` under `Mad`, a spec's default, one [`crate::EwQuantile`]
//! per target fed the residuals and weights `s²` is below, or `√s²` under
//! `Rms`, each aged on `huber_scale_decay` (the model's decay unless
//! given); **`eps` is in units of `s_y`**, the target's own EW standard
//! deviation as the row arrives. A cut and a tube in the target's
//! own units fitted one scale and failed the others -- a target in
//! thousandths never left a tube of 0.1, so the model was passive for
//! ever, and one in thousands never entered it (docs/PLAN.md task 195,
//! review round 4 CC4). Under `huber`, `s²` is the EW mean of the squared
//! out-of-sample residual `y − p`, aged on every row on the model's clock
//! and learned from a row with the target, a weight above 0 and a
//! prediction, after the row has used it:
//!
//! ```text
//! W_s ← lam·W_s;   s² ← (W_s·s² + w·(y − p)²) / (W_s + w),   W_s ← W_s + w
//! ```
//!
//! skipped where it would not be finite, as `huber`'s is. Under
//! `epsilon_insensitive`, `s_y²` is the EW variance of `y` around its EW
//! mean over the rows with the target and a weight above 0, whatever the
//! fit, the row's own `y` joining after it is judged ([`crate::spread`]).
//! Before the target has one -- no residual yet, or every one so far
//! exactly 0, for `s`; fewer than two weighted rows, or every `y` the same,
//! for `s_y` -- there is nothing to draw a cut or a tube in: the Huber row
//! is a squared-loss row and the tube has no width, `huber`'s rule for its
//! rows before a scale. The squared loss is then equivariant in the
//! target's scale -- a target scaled by `c` fits as the unscaled one,
//! scaled by `c` -- and so is the Huber loss; the quantile and
//! epsilon-insensitive losses step by the sign of the residual, so their
//! `learning_rate` is in the target's units.
//!
//! **Why two units.** The fit starts from zero coefficients, so before the
//! intercept takes the level ([`SgdCfg::target_spread`]) its first
//! residuals are the target's whole level. Outside the Huber cut the
//! gradient is clipped, not zero, so a cut drawn wide by those residuals
//! still lets every row teach. Under `Mad` the cut narrows once the fit has
//! the level, since the median leaves those residuals behind; under `Rms`
//! it narrowed only as a decay forgot them, so without decay the loss was
//! the squared loss to the bit (C-1, review round 9: slope 3.751 under both
//! on a target at 1,000 with 5% one-sided outliers, 2.016 under `Mad`;
//! docs/PLAN.md task 271). The intercept's running-mean step now takes the
//! level from the first row, so the level no longer reaches `Rms`, though
//! the outliers still inflate it (2.557 there; `Mad` 2.013), and the refit
//! of the first rows (`SgdCfg::refit_rows`, on by default, docs/PLAN.md
//! task 277) rebuilds either scale from its converged fit's residuals. Inside
//! the tube the gradient is zero: a tube of `eps = 0.1` drawn in the
//! residual's spread on a target at 1,000 in a spread of 2 was about 100
//! wide, held every row, and without decay never narrowed, so the fit
//! stopped where it stood (docs/PLAN.md task 202). `y`'s own spread does not read the fit.
//!
//! **The tube in units of the noise** (docs/PLAN.md task 207). On a target
//! a fit predicts to `R²`, `s_y = σ_noise / √(1 − R²)`, so the tube is
//! `eps / √(1 − R²)` noise standard deviations wide: at the default 0.01,
//! 0.067 of them at R² 0.978 and 2.2 at 0.99998. Inside it the gradient is
//! zero, so a tube many noise standard deviations wide holds the fit
//! wherever it first lands inside it. Task 207's sweep (one feature, no
//! decay; the out-of-sample error above the noise, in noise variances,
//! median over seeds), `inv_scaling` at 0.5 and the constant 0.01: at R²
//! 0.978, 0.015 and 0.042 at `eps = 0.01`, 0.010 and 0.027 at 0.1; at R²
//! 0.99998, 0.20 and 0.81 at 0.01, 24 and 17 at 0.1 (0.7 to 214 and 3.4 to
//! 30 over 20 seeds). The sign-valued step moves the fit's level by the
//! rate times the row's weight -- its prediction at the features' running
//! means under `standardize`, at 0 without -- so at the constant 0.01 a
//! level the target moves to by 1,000 is 100,000 unit-weight rows away
//! whatever the tube; the level it starts at, under `target_spread`, its
//! first learned row sets (below).
//!
//! **A logistic label outside {0, 1}** is clamped into `[0, 1]`, as `ftrl`
//! clamps it; `strict_binary` instead learns nothing from it, and the bank
//! refuses the chunk naming the row. `p − y` with `y = 5` pushed the linear
//! predictor up on every row for ever (review round 4, CC9).
//!
//! **A negative count under the Poisson loss** teaches nothing, and counts
//! nothing toward its target's weight; the bank refuses the chunk naming the
//! row, as scikit-learn's `PoissonRegressor` refuses one (docs/PLAN.md task
//! 209 (e)). `p − y` with `y < 0` drove the prediction to the link's clamp,
//! `e^−30`, within a few thousand rows and held it there.
//!
//! Learning rates ([`LearningRate`]): a constant, an inverse-scaling schedule
//! that anneals with `n_eff`, or AdaGrad's per-coordinate `lr / (sqrt(G_i) + eps)`.
//! AdaGrad's accumulator and `n_eff` are both decayed on the model's clock, so
//! an annealed or adapted rate re-opens after a long gap instead of staying
//! frozen at whatever it had converged to, up to the rate and no further.
//!
//! **The row's weight in the scaler.** The scaler reads every row in the
//! stream's own unit of weight, one row's: `ŵ = Σw²/Σw` over the rows it
//! learned until the switch, the weight Kish's count gives each of them,
//! exactly 1 at unit weights ([`scaler_decay`]). The row is admitted at `ŵ`
//! while the scaler warms up and at `min(ŵ, w)` once the fit is held, and
//! the history keeps at least `ŵ·n/(n + 1)` (`n` Kish's count) and then `ŵ`
//! of weight into the row. A step taken in the row's coordinates is mapped
//! into the caller's by dividing it by the row's scale, and after a gap
//! that left the history a share `a` of the moments with the row in, that
//! scale is `sqrt(a)` times the feature's spread: without the floor
//! AdaGrad's step, at most the rate in the row's coordinates, moved a slope
//! by `lr / sqrt(a)` spreads (2e8 rates after a gap of 60 half-lives), and
//! every learner's next rows were read against a scaler of one or two rows.
//! With it the share is at least 1/2, and AdaGrad moves a slope by at most
//! `sqrt(2) · lr` of the scaler's spread before the row, `|Δb_i| · sd_i <=
//! sqrt(2) · lr`, across any gap and at any scale of the weights. A row read
//! at weight 1 against moments in the weights' units was a share of them
//! that grew as the weights shrank: at weights of 1/8 and 8 features
//! AdaGrad's intercept was pushed the wrong way on every row (an error of
//! 1,140 noise variances, where weights of 1 give 0.18), and on weights
//! falling 100-fold over the stream each later row was read as a row of the
//! warm-up's size (4,176). In the stream's own unit a stream whose weights
//! are all scaled by a power of two is read as the unscaled one, to the
//! bit, and a light row is read at its own share. The floor binds only
//! where the history's decayed weight is under one row's, `lam·W < ŵ`: at
//! a half-life of 20 rows after a gap of more than about 5 half-lives, and
//! at a half-life of a row or two now and then. The coefficients do not decay:
//! under a constant rate their memory is in rows, about `1 / (lr · E[z²])`,
//! whatever the clock between them (docs/PLAN.md task 146).
//!
//! **The target's own units** ([`SgdCfg::target_spread`], schema 64;
//! docs/PLAN.md task 278, D1, D2 and D-S). The squared and Huber losses'
//! gradient is in the target's units, so their `clip_gradient` is in units
//! of `s_y`, the target's EW standard deviation as the row arrives (the
//! spread the epsilon-insensitive tube is drawn in), `|g_i| <=
//! clip_gradient · s_y`, with no clip before `s_y` is above 0; and
//! AdaGrad's epsilon is `1e-8 · s_y`, its step 0 where `√G + eps` is. As
//! constants in the target's units a target at `2^20` had every gradient
//! clipped by the default `1e3` (its fit 0.59-6.5 of the target's spread
//! off the unscaled one, scaled), and one at `2^-20` met the epsilon at 1%
//! of `√G` (4.2e-4 off). Every other loss's gradient is a sign, a count's
//! residual or a probability's, and keeps both in its own units.
//!
//! **The target's level** (the same flag, under an identity link with an
//! intercept). The fit starts from 0, so its first residuals were the
//! target's level, which moved every slope by about `lr · L` and was
//! learned at the clip's pace (2.49 off least squares at a level of 1e3
//! after 8,000 rows; 9.8e5 off a level of 1e6). Now, with `W = Σw` the
//! target's learned weight, undecayed, the row's in it, and `a = w / W`,
//! the intercept takes the running mean's step in place of its gradient's,
//! in whichever coordinates it is held (its `z̃` is 1 in each): `−a · d`
//! under the squared and Huber losses, the residual (clipped at the cut),
//! while `a` is above the row's rate (under `inv_scaling` the rate as it
//! has fallen; AdaGrad has no shared rate, so there the warm-up alone) or
//! Kish's count of the rows, `W² / Σw²`, is at most 22
//! ([`crate::WARMUP_ROWS`], the scaler's); `a · (y − p)` under
//! the sign-valued losses, whose gradient is not in the target's units,
//! while the count is. On the target's first learned row that is its whole
//! residual and nothing else moves. AdaGrad's sum learns the intercept's
//! gradient either way. The larger of the two gains is the rule `holt`
//! keeps too; a level now costs rounding (0.0077 at either level), and a
//! target with no level loses nothing measurable: 200-row groups of a
//! zero-mean target scored R² 0.46 at rows 25-50 where they scored 0.45,
//! and 0.44 at a level of 10 per group where they scored −47,000.
//!
//! **A rate past its stability bound** (the same flag; docs/PLAN.md task
//! 278, D8). A step multiplies its own row's residual by `1 − q`, `q =
//! lr · (w · c · |z̃|² + l2)`, `c` the loss's curvature at the prediction
//! (1 for the squared loss and inside the Huber cut, 0 beyond it, `p` for
//! Poisson, `p (1 − p)` for logistic), so the residual shrinks in mean
//! square while `E[(1 − q)²] < 1`, that is `E[q²] < 2·E[q]`. Each target
//! keeps `Σρ^k q̂`, `Σρ^k q̂²` and `Σρ^k` over its stepped rows, `ρ =
//! 2^(−1/100)`, each gain after the first bounded as it enters, `q̂ =
//! min(q, max(2, 4·m))` with `m` the running mean before it (a gain of 0,
//! a row the loss is flat at, says nothing of the rate and is left out),
//! and while the second sum passes
//! twice the first its predictions are withheld, from the state before the
//! row, by `step` and `predict` alike; the fit learns every row as before,
//! so the gate is the output's alone. The bound keeps one wide row from
//! holding the output: a gain at 2 leaves its row's residual where it was
//! and adds nothing to the margin `Σ(q̂² − 2q̂)`, so while the mean is below
//! 1/2 no single row moves the verdict toward unstable, and above it a row
//! moves it by at most `16m² − 8m`; a rate whose every gain is high still
//! passes, the mean rising fourfold a row to meet it. The price: a stream
//! whose gains average below 1/2 and pass 2 on a few rows only is not
//! withheld for those rows, though the one-row rule calls it unstable. The memory is a
//! count of stepped rows because the verdict is an estimate of an
//! expectation, whose error is set by the spread of the gains, not by the
//! rate or the decay ([`GAIN_DECAY`]). Under the squared and Huber losses
//! a calm gate rests (task 314; [`StepGain`]): once its sums hold two
//! half-lives of rows and `Σq̂² ≤ Σq̂`, half the bound or less, it takes no
//! gains, so no row forms `|z̃|²` for it, but a probe's every
//! [`PROBE_ROWS`] rows; a probe past [`PROBE_WAKE`] times the resting mean,
//! or a residual past [`WAKE_SPREADS`] of the target's spread, wakes it,
//! and it rests again only after [`AWAKE_ROWS`] gains, once calm. The rule is the one-row mean-square
//! contraction `E[(1 − q)²] < 1`; on standardized normal features with an
//! intercept it bounds a constant rate at `2 (p + 1) / (p² + 4p + 1)`,
//! 0.261 at five features, where the measured edge was 0.25-0.3. Without
//! the gate, a constant rate past the bound diverged until the clip held
//! it, finite and wrong (the last 1,000 of 20,000 rows' error 3.6e4 times the
//! target's variance at 0.5), with no notice. AdaGrad's step moves a slot at most the rate a
//! row in the row's coordinates (under `standardize`, a slope at most
//! `sqrt(2)` rates of its feature's spread in the caller's units, across
//! any gap; above), and the quantile and epsilon-insensitive steps are a
//! sign times the rate: neither can grow, and none is gated.

use serde::{Deserialize, Serialize};

use crate::huber_refit::{self, RefitBuf, Rows};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::spread::TargetSpread;
use crate::{Constraint, Decay, EwDiag, Warmup};

pub(crate) mod variants;

/// Loss function, and with it the link (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SgdLoss {
    Squared,
    /// `delta` is in units of the target's residual scale
    /// ([`SgdCfg::huber_scale`]), as `robust`'s Huber is: a clipped
    /// gradient keeps learning, so a cut a fit's first residuals widen does
    /// not stop it (the module docs).
    Huber {
        delta: f64,
    },
    Quantile {
        tau: f64,
    },
    /// Ignores residuals within `eps` of the target's own EW std — the SVR
    /// loss. Not the residual's: inside the tube the gradient is zero, and
    /// a tube a fit's first residuals widen stops it (the module docs).
    EpsilonInsensitive {
        eps: f64,
    },
    /// Log link, for non-negative count targets: a negative one teaches
    /// nothing, and the bank refuses it by row (the module docs).
    Poisson,
    Logistic,
}

/// Per-coordinate step size.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningRate {
    Constant,
    /// `lr / (1 + lam·n_eff)^power`: the weight before the row decayed to
    /// it, `lam` the row's decay, so a zero-weight row in between moves
    /// nothing (hard rule 9).
    InvScaling {
        power: f64,
    },
    /// `lr / (sqrt(G_i) + eps)`, `G_i` the decayed sum of squared
    /// gradients; `eps` is `1e-8`, in units of the target's spread under
    /// [`SgdCfg::target_spread`] for the squared and Huber losses.
    AdaGrad,
    /// Bottou's `optimal` schedule matching `SGDRegressor(learning_rate="optimal")`:
    /// `eta(t) = 1 / (alpha * (t0 + t))` with `alpha = l1 + l2 > 0`,
    /// `t0 = 1 / (alpha * sqrt(1 / sqrt(alpha)))`, and per-target learned step
    /// count `t = 0, 1, 2, ...`.
    Optimal,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SgdCfg {
    pub n_features: usize,
    pub n_targets: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    pub loss: SgdLoss,
    pub learning_rate: f64,
    pub schedule: LearningRate,
    /// Ridge penalty added to the gradient. The intercept is never penalized.
    pub l2: f64,
    pub min_weight: f64,
    /// Take the gradient step on the features standardized against their
    /// own running moments (ENHANCEMENTS E24): while the moments warm up
    /// the coefficients are in their coordinates and read out in the
    /// caller's units, and from then on held in the caller's units, the
    /// step mapped into them by the row's map (the module docs; docs/PLAN.md
    /// task 206).
    ///
    /// Gradient methods are the ones that need this: a single learning rate has
    /// to suit every coordinate, so a feature measured in thousands and one
    /// measured in basis points cannot both converge. The exact solvers do not
    /// care (they standardize inside the solve, or not at all).
    ///
    /// The moments a row is standardized against **include the row**, as one
    /// more row of the stream, at one row's weight of it (the module docs,
    /// *The row's weight in the scaler*): that
    /// is sklearn's `partial_fit` then `transform`, and it bounds a
    /// standardized value by `sqrt(n_eff)` whatever the running variance
    /// happens to be (the bound is reached when the history has no spread
    /// at all). Against the moments from *before* the row -- the
    /// original E24 -- a variance estimate a few rows old can be tiny by
    /// chance, the standardized value huge, and one step with
    /// `lr · |z|² > 2` throws a coefficient where the next hundred rows do
    /// not bring it back: R² of −6.9 over rows 25–50 of 200-row groups
    /// where `ewridge` scored 0.97, and a wide fit is that on every row
    /// (docs/PLAN.md task 74). Not a leak: the row's features are known at
    /// prediction time, and the rule is about the target, which enters
    /// nothing until after the prediction. While the scaler warms up, the
    /// one standardized row serves the prediction and the gradient, so a
    /// row is standardized the way every row the coefficients were learned
    /// from was (sklearn's loop predicts against the moments from before the
    /// row instead, which is two standardizations a row and a prediction
    /// with no bound). While the scaler warms up the row is read at one
    /// row's weight of the stream, `Σw²/Σw` of the rows the scaler has
    /// learned, whatever its own: its weight applies to what it teaches, not
    /// to where it sits among the rows seen, so a prediction never depends
    /// on the row's weight and `predict` gives the step's number exactly.
    /// Once the fit is held a prediction reads no moment, and the row is
    /// read at its own weight, at most one row's, so that a light row is not
    /// a larger share of the moments than its update makes it.
    ///
    /// **The warm-up** ([`crate::Warmup`]): until Kish's count of the rows
    /// the scaler has learned reaches 22 the coefficients are read out with
    /// the moments as they stand, while each step was taken in the
    /// coordinates of its own row, so a coefficient "per unit of `x`" moves
    /// with the scaler as well as with the fit; and for a weight other than
    /// one row's the coordinates the gradient was taken in (the row at one
    /// row's weight) and the ones the projection and the coefficients use a
    /// moment later
    /// (the row at its weight) differ (review 2026-09-12, D5). From the row
    /// after the switch the coefficients are held in the caller's units and
    /// each step is mapped by its own row's map, once: a prediction is `x̃ᵀb`
    /// and reads no moment, so the scaler's moving moves none of it, and
    /// D5's gap is gone. Measured (task 206): at a half-life of 10 and R²
    /// 0.978 the slope came out 3.03 for a truth of 2 read through the
    /// moments, 2.02 held; at a half-life of 50 and R² 0.99998 the
    /// out-of-sample error was 248 noise variances above the noise, 0.010
    /// held, the unstandardized fit's.
    #[serde(default)]
    pub standardize: bool,
    /// Cap on `|gradient|` before the step. **Finite by default** (`1e3` via the
    /// spec layer), not because ordinary losses need it but because a log-link
    /// loss does: with `Poisson`, `p = exp(eta)`, so one row that pushes `eta`
    /// up makes the next gradient exponentially larger, and a constant learning
    /// rate diverges within a few thousand rows. Measured on a 30k-row Poisson
    /// stream: unclipped the intercept ran to -4e10, clipped at 1e3 it recovers
    /// the true `[0.4, 0.8]`. Under [`SgdCfg::target_spread`] the squared
    /// and Huber losses' cap is in units of the target's spread `s_y`, none
    /// before it has one, so it binds at a target's every scale alike, and
    /// rarely: a gradient of a thousand target spreads (the module docs).
    /// `inf` disables it.
    #[serde(with = "crate::humanfloat::f64_or_tag")]
    pub clip_gradient: f64,
    /// Box and/or sum constraint on the slopes, imposed by Euclidean
    /// projection after each update (ENHANCEMENTS E40); the intercept is
    /// free. The projection is taken in the space the step is taken in: the
    /// caller's units, or the standardized coordinates under
    /// `standardize`, where a caller bound `lo_i` on `c_i` is the bound
    /// `lo_i * scale_i` on `b_i` and the sum is `sum(b_i / scale_i)`. While
    /// the scaler warms up that is the coefficients held in those
    /// coordinates, projected after the scaler learns each row, every target
    /// when the scales moved; past it, the coefficients in the caller's
    /// units, projected after each step in the metric of the row's
    /// coordinates, the bounds met exactly, and the intercept keeping its
    /// standardized value `b_0 + Σ m_i b_i`, so it takes `Σ m_i (b_i − b'_i)`
    /// from the slopes' move ([`Constraint::project_in_units`]). The
    /// reported coefficients satisfy the constraint after every learned
    /// row, and the initial `0` is projected too, so a simplex starts uniform.
    #[serde(default)]
    pub constraint: Option<Constraint>,
    /// A logistic label that is not 0 or 1 is not learned from, where the
    /// default clamps it into `[0, 1]`: `ftrl`'s rule. Logistic only. The
    /// bank never hands the model such a row: it refuses the chunk, naming
    /// the row (docs/PLAN.md task 195, S4).
    #[serde(default)]
    pub strict_binary: bool,
    /// The scale the Huber cut is drawn in, `huber`'s
    /// ([`crate::HuberScale`]; the module docs). A state written before
    /// schema 63 has no such field and was fitted under the RMS, so it
    /// loads as [`crate::HuberScale::Rms`], where a spec's default under
    /// `loss = "huber"` is the MAD. Every other loss keeps
    /// [`crate::HuberScale::Rms`], which it does not read.
    #[serde(default = "crate::HuberScale::of_a_state_before_63")]
    pub huber_scale: crate::HuberScale,
    /// The Huber scale's own memory: `None` ages it on `decay`, as before
    /// schema 63. Huber only.
    #[serde(default)]
    pub huber_scale_decay: Option<Decay>,
    /// Each target's first `refit_rows` learned rows refit to the converged
    /// Huber fit, once ([`crate::huber_refit`]): the coefficients set to it
    /// and the residual scale rebuilt from its residuals, a spec's
    /// `huber_refit_rows`, by default `max(500, 10 · features)` under the
    /// Huber loss up to 100 features and off above
    /// ([`crate::default_refit_rows`]); 0 is off.
    /// Huber only. A state written before schema 64 has no such field and
    /// was fitted without a refit, so it loads as 0 and goes on as it was.
    #[serde(default)]
    pub refit_rows: u32,
    /// Whether `min_weight` holds the Huber scale's learning back as well
    /// as the output, as it did before schema 64: a row it withheld had no
    /// prediction, so its residual taught the scale nothing. A spec builds
    /// `false`: the scale learns from the prediction the fit made, whether
    /// or not `min_weight` withholds it, so a fit is the same with any
    /// `min_weight`. A state written before schema 64 has no such field
    /// and loads as `true`, going on to the bit
    /// ([`crate::RobustCfg::gate_held_learning_before_64`]).
    #[serde(default = "crate::RobustCfg::gate_held_learning_before_64")]
    pub min_weight_holds_learning: bool,
    /// Whether the fit is drawn in the target's own units and gates an
    /// unstable rate (the module docs, *The target's own units* and *A
    /// rate past its stability bound*): the squared and Huber losses'
    /// `clip_gradient` and AdaGrad's epsilon in units of the target's EW
    /// standard deviation `s_y`, the intercept taking the running mean's step
    /// while it is the larger under an identity link ([`Level`]), and each
    /// target's predictions withheld while its recent steps' gains say
    /// the rate is past its stability bound ([`Sgd::unstable`]). A spec
    /// sets it. A state written before schema 64 has no such field and was
    /// fitted without any of these, so it loads as `false` and goes on as
    /// it was, to the bit.
    #[serde(default)]
    pub target_spread: bool,
    /// Whether each MAD sketch is anchored, as
    /// [`crate::RobustCfg::mad_anchor`]: a refit's rebuilt sketch at its
    /// residuals' weighted median, the grid fixed at 1 otherwise. A spec
    /// sets it; a state written before schema 64 loads as `false` and goes
    /// on as it was.
    #[serde(default)]
    pub mad_anchor: bool,
    /// The refit of the first rows over a span of the clock in place of
    /// `refit_rows` (docs/PLAN.md task 538), as
    /// [`crate::RobustCfg::refit_clock`]: a target's rows from its first
    /// learned row through the first at least this many clock units after
    /// it, once they hold a row per coefficient, its buffers held to
    /// `max_state_bytes` a group as they grow.
    #[serde(default)]
    pub refit_clock: Option<f64>,
    /// The bytes a group's refit buffers may hold under a span of the
    /// clock: `None` is 256 KB, at most 64 MiB (docs/PLAN.md task 538).
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// L1 (lasso / elastic-net) penalty applied to slopes via Tsuruoka et al.
    /// cumulative penalty truncation (`0.0` disables L1). The intercept is
    /// never penalized.
    #[serde(default)]
    pub l1: f64,
    /// Decay-weighted Polyak–Ruppert averaging of the live iterates. When
    /// true, predictions and [`Sgd::coefficients`] serve the running average
    /// while the live iterate updates from its own predictions.
    #[serde(default)]
    pub average: bool,
    /// Karampatziakis & Langford (Vowpal Wabbit `--invariant`) importance-weight
    /// invariant squared-loss step. Squared loss with a shared rate only.
    #[serde(default)]
    pub invariant: bool,
}

impl Serialize for SgdCfg {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        struct HumanF64(f64);
        impl Serialize for HumanF64 {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                crate::humanfloat::f64_or_tag::serialize(&self.0, s)
            }
        }
        let has_new = self.l1 != 0.0 || self.average || self.invariant;
        let has_cap = has_new || self.max_state_bytes.is_some();
        let has_clock = has_cap || self.refit_clock.is_some();
        let len = 19 + usize::from(has_clock) + usize::from(has_cap) + 3 * usize::from(has_new);
        let mut st = serializer.serialize_struct("SgdCfg", len)?;
        st.serialize_field("n_features", &self.n_features)?;
        st.serialize_field("n_targets", &self.n_targets)?;
        st.serialize_field("fit_intercept", &self.fit_intercept)?;
        st.serialize_field("decay", &self.decay)?;
        st.serialize_field("loss", &self.loss)?;
        st.serialize_field("learning_rate", &self.learning_rate)?;
        st.serialize_field("schedule", &self.schedule)?;
        st.serialize_field("l2", &self.l2)?;
        st.serialize_field("min_weight", &self.min_weight)?;
        st.serialize_field("standardize", &self.standardize)?;
        st.serialize_field("clip_gradient", &HumanF64(self.clip_gradient))?;
        st.serialize_field("constraint", &self.constraint)?;
        st.serialize_field("strict_binary", &self.strict_binary)?;
        st.serialize_field("huber_scale", &self.huber_scale)?;
        st.serialize_field("huber_scale_decay", &self.huber_scale_decay)?;
        st.serialize_field("refit_rows", &self.refit_rows)?;
        st.serialize_field("min_weight_holds_learning", &self.min_weight_holds_learning)?;
        st.serialize_field("target_spread", &self.target_spread)?;
        st.serialize_field("mad_anchor", &self.mad_anchor)?;
        if has_clock {
            st.serialize_field("refit_clock", &self.refit_clock)?;
        }
        if has_cap {
            st.serialize_field("max_state_bytes", &self.max_state_bytes)?;
        }
        if has_new {
            st.serialize_field("l1", &self.l1)?;
            st.serialize_field("average", &self.average)?;
            st.serialize_field("invariant", &self.invariant)?;
        }
        st.end()
    }
}

/// The memory of the step gains' means ([`StepGain`]): `2^(-1/100)` a
/// stepped row, a half-life of 100 of them, written out so that no
/// platform's `exp2` sets a persisted bit (CLAUDE.md, no libm in the
/// state).
///
/// A count of rows, not the fit's own memory, because the gate estimates
/// an expectation: whether `E[q²] < 2·E[q]` for the rate as it stands,
/// from a sample of its rows. The ratio of two means of `N` gains has a
/// relative error of about `CV(q)/√N`, and `CV(q)` -- the spread of
/// `|z̃|²` -- is a property of the rows, not of the rate or the decay: on
/// standardized normal features it is `√(2p)/(p + 1)`, widest at one
/// feature (0.71). At this memory the gate withheld 0.9% of the rows of a
/// constant rate at 0.9 times the bound and 99% at 1.2 times at one
/// feature, and none at 0.9 and all at 1.1 at five and at twenty (the
/// recursion on 12,000 rows' gains drawn from standardized normal rows;
/// 25 rows' memory withheld 7% at 0.9 times at one feature and 87% at 1.2
/// times). The fit's own memory -- about `k / mean(q)` rows for a
/// coefficient, `1 / mean(q)` along the row -- is a handful of rows near
/// the bound, and as the gate's memory left the verdict to the last few
/// rows' spread: it withheld 6-11% of a stable rate's rows (0.77 times the
/// bound, five features) and served 33-52% of a diverging one's (1.15
/// times). The model's decay ages the target's weight and AdaGrad's sums,
/// not the coefficients, so it is no memory of the rate either. A rate
/// that changes is followed within about two half-lives: a constant rate
/// raised past the bound after 2,000 stable rows was withheld from 68 rows
/// later at 1.9 times the bound and 201 at 1.15 times; and a wide row
/// cannot hold the output for long, since each gain is bounded before it
/// enters ([`GAIN_BOUND`]).
const GAIN_DECAY: f64 = 0.993_092_495_437_035_9;

/// The largest step gain a [`StepGain`] takes in: its square stays finite,
/// and a larger one is unstable by any measure.
const GAIN_CAP: f64 = 1e150;

/// The bound on each step gain as it enters a [`StepGain`]'s sums: the
/// larger of [`NEUTRAL_GAIN`] and `GAIN_BOUND` times the running mean of
/// the gains before it. One wide row -- a feature far outside its spread,
/// a weight far above the rest -- has a gain many times the others', and
/// whole it carried the sums past the bound by itself and held them there
/// while its square faded: on raw features at a rate of 0.01, 1,075 rows
/// after one row at 100 times a feature's spread and 3,736 at 1e4 (three
/// features, 6,000 rows after it), and about 99,000 by the recursion where
/// a gain that is not a number entered as the cap. Bounded, the same probe withholds none. A row moves the
/// margin `Σ(q̂² − 2q̂)` by at most `16m² − 8m` at a mean `m` above 1/2,
/// and below it not toward unstable at all; and a rate whose every row's
/// gain is high still passes the bound, the mean rising fourfold a row to
/// meet it.
const GAIN_BOUND: f64 = 4.0;

/// The gain at which a step leaves its own row's residual where it was,
/// `(1 − q)² = 1`: a row at it adds `q² − 2q = 0` to the gate's margin, so
/// a gain bounded at it never moves the verdict. It is the bound's floor,
/// so a gain beside a mean below `1/2` can only leave the verdict where it
/// was or take it toward stable.
const NEUTRAL_GAIN: f64 = 2.0;

/// Per target, the recent steps' gains `q = lr·(w·c·|z̃|² + l2)`, `c` the
/// loss's curvature at the row's prediction, as sums decayed by
/// [`GAIN_DECAY`] a stepped row: `q` and `q²`, and the rows' weight `n`, so
/// that `q / n` is their mean ([`SgdCfg::target_spread`]; the module docs).
/// Each gain after the first enters bounded, `q̂ = min(q, max(2, 4·q/n))`
/// ([`GAIN_BOUND`]); the first, with no mean to be wide against, whole. A step multiplies its own row's residual by `1 − q`,
/// so the residual shrinks in mean square while `E[(1 − q)²] < 1`, that is
/// `E[q²] < 2·E[q]`; past it the fit diverges, and the target is
/// [`StepGain::unstable`].
///
/// Under a loss in the target's units (squared and Huber), a gate whose
/// sums hold [`REST_WEIGHT`] of rows and whose mean square is at most its
/// mean -- half the bound or less -- rests: it takes no gains, so no row
/// forms `|z̃|²` for it, and its verdict stays the stable one its sums
/// read. Every [`PROBE_ROWS`]-th row it rests it takes a probe of the
/// row's gain, and wakes where the probe passes [`PROBE_WAKE`] times the
/// sums' mean; a row whose residual passes [`WAKE_SPREADS`] of the
/// target's spread wakes it too. Awake, it takes the waking gain and every
/// one after, as before it rested, and may rest again after
/// [`AWAKE_ROWS`] of them. Under Huber a waking residual is past the cut
/// wherever the cut, `δ` residual scales, is the narrower -- at
/// `δ = 1.345`, unless the residuals spread six times wider than the
/// target -- and there the loss has no curvature: the waking gain is 0,
/// which the sums skip as they skip every such row, so the first gain the
/// gate takes is the next row's inside the cut. A rate that passes its
/// bound raises its gains, so a probe finds it within a few rows. What the
/// rest saves is the other rows' gains: at 20 features and a rate of 0.01
/// the gate rests on 95% of the rows, and the step takes 0.7 of its 36 ns
/// less on arm64, natively (task 314).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub(crate) struct StepGain {
    q: f64,
    q2: f64,
    n: f64,
    /// Whether the gate rests (above). A state written before it was kept
    /// loads awake, and the gate goes on as it did.
    #[serde(default)]
    resting: bool,
    /// Resting, the rows the gate has let pass since it rested: every
    /// [`PROBE_ROWS`]-th takes a probe of the row's gain. Awake, the gains
    /// it has taken since it woke (0 at the wake, which a waking gain then
    /// makes 1): it may rest again once they reach [`AWAKE_ROWS`]. A row
    /// whose step is refused moves it in neither meaning.
    #[serde(default)]
    rested: u32,
}

/// The gains a gate takes after it wakes before it may rest again: a
/// half-life of them ([`StepGain`]).
const AWAKE_ROWS: u32 = 100;

/// A resting gate probes every `PROBE_ROWS`-th row's gain ([`StepGain`]).
const PROBE_ROWS: u32 = 8;

/// A probe wakes the gate where its gain passes `PROBE_WAKE` times the
/// resting sums' mean ([`StepGain`]).
const PROBE_WAKE: f64 = 2.0;

/// The weight of rows a gate's sums hold before it may rest: three
/// quarters of their limit, `1 / (1 − GAIN_DECAY)` ≈ 144.8, which they
/// reach after two half-lives, 200 stepped rows ([`StepGain`]).
const REST_WEIGHT: f64 = 108.0;

/// The residual, in the target's spreads, that wakes a resting gate
/// ([`StepGain`]). A rate that has passed its bound grows its residuals
/// row by row, and passes this within a few rows of doing so; a row this
/// far off for another reason wakes the gate for no more than the rows it
/// takes to be calm again.
const WAKE_SPREADS: f64 = 8.0;

/// Per target, the weight of the rows its level has been learned from,
/// undecayed, `Σw` and `Σw²` ([`SgdCfg::target_spread`]; the module docs):
/// the running mean's gain `w / Σw`, and Kish's count of the rows,
/// `(Σw)² / Σw²`, which [`crate::WARMUP_ROWS`] the level warms up over.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub(crate) struct Level {
    w: f64,
    w2: f64,
    /// `w2`'s power of two, `Σw² = w2 · 4^w2_scale` ([`crate::sqsum`]): 0,
    /// and not written, while the weights square to doubles; at weights of
    /// `1e-200` the squares were 0 and the level never warmed (review
    /// 2026-11-23, REG-12). Schema 72; last, for the compact encoding.
    #[serde(default, skip_serializing_if = "crate::sqsum::is_zero")]
    w2_scale: i32,
}

impl Level {
    fn learn(&mut self, weight: f64) {
        self.w += weight;
        crate::sqsum::add(&mut self.w2, &mut self.w2_scale, 1.0, weight);
    }

    fn count(&self) -> f64 {
        crate::sqsum::count(self.w, self.w2, self.w2_scale).unwrap_or(0.0)
    }
}

impl StepGain {
    /// The gain `q` bounded as it enters the sums ([`GAIN_BOUND`]): at most
    /// the larger of 2 and four times the running mean before it, and at
    /// most [`GAIN_CAP`]. The first gain has no mean to be wide against and
    /// enters whole, at most the cap. A gain that is not a number is taken
    /// at the bound (`f64::min` returns the other operand).
    fn bounded(&self, q: f64) -> f64 {
        if self.n > 0.0 {
            q.min(NEUTRAL_GAIN.max(GAIN_BOUND * self.q / self.n))
                .min(GAIN_CAP)
        } else {
            q.min(GAIN_CAP)
        }
    }

    /// A gain of 0 -- a row the loss is flat at (beyond the Huber cut, a
    /// saturated logistic) with no `l2` -- leaves the sums alone: its step
    /// neither contracts nor grows its own row's residual, so it says
    /// nothing of the rate, and counted in `n` it would pull the mean, and
    /// with it the bound, down. A fit thrown so far that every row is past
    /// the cut would then dilute its own verdict to the floor of 2, which
    /// moves nothing, while the rows that still step with a curvature, each
    /// at a gain far past 2, are the evidence the rate diverges.
    ///
    /// Where `may_rest`, a gate whose sums are calm after the gain rests.
    fn learn(&mut self, q: f64, may_rest: bool) {
        if q == 0.0 {
            return;
        }
        let q = self.bounded(q);
        self.q = GAIN_DECAY * self.q + q;
        self.q2 = GAIN_DECAY * self.q2 + q * q;
        self.n = GAIN_DECAY * self.n + 1.0;
        // Awake, `rested` counts the gains taken since the gate woke; it
        // rests again only after a half-life of them, so a gate woken by a
        // probe or a wide residual reads the rows after it before it
        // judges itself calm.
        self.rested = self.rested.saturating_add(1);
        if may_rest && self.rested >= AWAKE_ROWS && self.n >= REST_WEIGHT && self.q2 <= self.q {
            self.resting = true;
            self.rested = 0;
        }
    }

    /// The recent gains' mean square above twice their mean: the step
    /// grows its own row's residual in mean square. Both sums carry the
    /// same weights, so their ratio is the means'; with none, 0 against 0.
    fn unstable(&self) -> bool {
        self.q2 > 2.0 * self.q
    }

    /// Whether the sums and rest flag are ones a stream writes (review
    /// 2026-11-23, STATE-3): finite non-negative sums, `n == 0` only when
    /// both sums are 0, `q == 0` only when `q2 == 0`, and a resting gate
    /// only once `n >= REST_WEIGHT` with `q2 <= q` (a resting gate with
    /// `n == 0` divides by 0 on its next probe).
    fn is_valid(&self) -> bool {
        let non_neg = |v: f64| v.is_finite() && v >= 0.0;
        non_neg(self.q)
            && non_neg(self.q2)
            && non_neg(self.n)
            && (self.n > 0.0 || (self.q == 0.0 && self.q2 == 0.0))
            && (self.q > 0.0 || self.q2 == 0.0)
            && (!self.resting || (self.n >= REST_WEIGHT && self.q2 <= self.q))
    }
}

/// The ridge on the mean-form system an `sgd` refit solves
/// ([`crate::huber_refit::irls`]): `huber`'s default, so that the refit
/// is the converged fit `huber` reaches on the same rows. Like `huber`'s, it
/// is on the slopes the fit is drawn in: under `standardize` the
/// standardized slope `β_i = s_i b_i`, so in the caller's units `1e-6 · s_i²`
/// on slope `i`'s diagonal, as `huber` divides its Gram by `s_i s_j` before
/// adding it. A ridge of `1e-6` in the caller's units shrank a feature whose
/// spread is near `1e-3` to about half its slope, and one near `1e-4` to a
/// hundredth (2 read 0.022), and a feature's power-of-two scale moved the
/// fit after the refit.
const REFIT_RIDGE: f64 = 1e-6;

impl SgdCfg {
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        // The decay first: every model checks it in its own `new`, where only
        // the bank's spec did (review 2026-10-05, CF5).
        no.take(self.decay.check().map_err(|e| format!("sgd: {e}")));
        no.take(crate::model::check_dims(
            "sgd",
            self.n_features,
            self.n_targets,
        ));
        if self.learning_rate <= 0.0 || self.learning_rate.is_nan() {
            no.refuse(format!(
                "sgd: learning_rate must be above 0, but it is {}. Set learning_rate to a \
                 positive step size, such as the default 0.01.",
                self.learning_rate
            ));
        }
        // NaN passes `v < 0.0` and `v <= 0.0` alike, so each bound below
        // names it: a NaN `clip_gradient` reached `f64::clamp`, which panics
        // on a NaN bound, on the first learned row; a NaN `l2`, `eps` or
        // `power` poisoned every coefficient or rate (review 2026-09-18, B4).
        if self.l2.is_nan() || self.l2 < 0.0 {
            no.refuse(format!(
                "sgd: l2 must be at least 0, but it is {}. Set l2 to 0 for no shrinkage, or \
                 to a positive value.",
                self.l2
            ));
        }
        if !self.l1.is_finite() || self.l1 < 0.0 {
            no.refuse(format!(
                "sgd: l1 must be finite and at least 0, but it is {}. Set l1 to 0 for no L1 \
                 shrinkage, or to a finite positive value.",
                self.l1
            ));
        }
        if self.clip_gradient.is_nan() || self.clip_gradient <= 0.0 {
            no.refuse(format!(
                "sgd: clip_gradient must be above 0, but it is {}. Set clip_gradient to the \
                 largest size a coordinate of the gradient may take, such as the default \
                 1000, or to inf for no clipping.",
                self.clip_gradient
            ));
        }
        match self.loss {
            // NaN too: `f64::clamp` panics on a NaN bound. `inf` clips
            // nothing, which is the squared loss.
            SgdLoss::Huber { delta } if delta <= 0.0 || delta.is_nan() => {
                no.refuse(format!(
                    "sgd: huber_delta must be above 0, but it is {delta}. Set huber_delta to \
                     a positive cut in units of the residuals' scale, or to inf for the \
                     squared loss."
                ));
            }
            SgdLoss::Quantile { tau }
                if !(0.0..=1.0).contains(&tau) || tau == 0.0 || tau == 1.0 =>
            {
                no.refuse(format!(
                    "sgd: quantile must be above 0 and below 1, but it is {tau}. Set quantile \
                     to the level to fit, such as 0.5 for the median."
                ));
            }
            SgdLoss::EpsilonInsensitive { eps } if eps.is_nan() || eps < 0.0 => {
                no.refuse(format!(
                    "sgd: eps must be at least 0, but it is {eps}. Set eps to how far a \
                     residual may fall from 0 at no cost, in standard deviations of the \
                     target, such as the default 0.01, or to 0 for none."
                ));
            }
            _ => {}
        }
        if let LearningRate::InvScaling { power } = self.schedule
            && (power.is_nan() || power < 0.0)
        {
            no.refuse(format!(
                "sgd: power must be at least 0, but it is {power}. Set power to 0 for a \
                 constant step, or to a positive value for a step that shrinks as the model \
                 learns."
            ));
        }
        if let Some(c) = &self.constraint {
            no.take(c.validate(self.n_features, "sgd"));
        }
        // The loss as the spec names it, for the refusals that read it.
        let loss = match self.loss {
            SgdLoss::Squared => "squared",
            SgdLoss::Huber { .. } => "huber",
            SgdLoss::Quantile { .. } => "quantile",
            SgdLoss::EpsilonInsensitive { .. } => "epsilon_insensitive",
            SgdLoss::Poisson => "poisson",
            SgdLoss::Logistic => "logistic",
        };
        if self.strict_binary && self.loss != SgdLoss::Logistic {
            no.refuse(format!(
                "sgd: strict_binary checks the targets of the logistic loss, but loss is \
                 \"{loss}\". Set loss=\"logistic\", or leave strict_binary out."
            ));
        }
        // The Huber cut's scale and its memory are the Huber loss's
        // (docs/PLAN.md task 271).
        if let SgdLoss::Huber { delta } = self.loss {
            if let Some(d) = self.huber_scale_decay {
                no.take(crate::model::check_scale_decay("sgd", d));
            }
            if (self.refit_rows > 0 || self.refit_clock.is_some()) && delta == f64::INFINITY {
                // The key the refit was asked for by (review 2026-11-23, SPEC-7).
                let key = if self.refit_clock.is_some() {
                    "huber_refit_span"
                } else {
                    "huber_refit_rows"
                };
                no.refuse(format!(
                    "sgd: {key} refits a target's first rows to a Huber fit, but huber_delta is \
                     inf, which makes that fit least squares. Set huber_delta to a finite cut, \
                     or leave the refit out."
                ));
            }
            no.take(crate::clock_span::check_max_state_bytes(
                "sgd",
                self.max_state_bytes,
            ));
            if let Some(d) = self.refit_clock {
                if !(d.is_finite() && d > 0.0) {
                    no.refuse(format!(
                        "sgd: huber_refit_span must be a finite number of clock units above 0, \
                         but it is {d}. Give the span of clock whose rows the refit fits, or \
                         leave it out for huber_refit_rows."
                    ));
                }
                if self.refit_rows > 0 {
                    no.refuse(format!(
                        "sgd: huber_refit_span and huber_refit_rows = {} are two refits of the \
                         first rows, and a target is refit once. Leave one of them out.",
                        self.refit_rows
                    ));
                }
            }
        } else {
            // Each one the reader set, by name.
            let given: Vec<String> = [
                ("huber_scale", self.huber_scale != crate::HuberScale::Rms),
                ("huber_scale_half_life", self.huber_scale_decay.is_some()),
                ("huber_refit_rows", self.refit_rows > 0),
                ("huber_refit_span", self.refit_clock.is_some()),
                ("max_state_bytes", self.max_state_bytes.is_some()),
            ]
            .into_iter()
            .filter(|(_, set)| *set)
            .map(|(k, _)| k.to_string())
            .collect();
            if !given.is_empty() {
                let names = crate::model::listed(&given);
                let (verb, them) = if given.len() == 1 {
                    ("applies", "it")
                } else {
                    ("apply", "them")
                };
                no.refuse(format!(
                    "sgd: {names} {verb} only under loss=\"huber\", but loss is \"{loss}\". \
                     Set loss=\"huber\", or leave {them} out."
                ));
            }
        }
        if matches!(self.schedule, LearningRate::Optimal) {
            let alpha = self.l1 + self.l2;
            if !(alpha.is_finite() && alpha > 0.0) {
                no.refuse(
                    "sgd: schedule=\"optimal\" needs a positive finite regularization strength, \
                     since its rate is derived from l1 and l2. Set l2 or l1 above 0, or use \
                     another schedule.",
                );
            }
        }
        if self.invariant {
            if self.loss != SgdLoss::Squared {
                no.refuse(format!(
                    "sgd: invariant=True applies only to loss=\"squared\", but loss is \
                     \"{loss}\". Set loss=\"squared\", or set invariant=False."
                ));
            }
            if matches!(self.schedule, LearningRate::AdaGrad) {
                no.refuse(
                    "sgd: invariant=True requires a shared learning rate, so it cannot be \
                     combined with schedule=\"adagrad\". Use schedule=\"constant\", \
                     \"inv_scaling\", or \"optimal\", or set invariant=False.",
                );
            }
        }
        // What the spec layer refuses, refused here too, so the Rust API
        // and a state file are held to it (review round 5, C6): a NaN
        // `min_weight` is never reached, and a negative one is no floor.
        // `inf` never predicts, which is legal.
        no.take(crate::model::check_min_weight("sgd", self.min_weight));
        no.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "SgdV3")]
pub struct Sgd {
    cfg: SgdCfg,
    /// Running feature means and variances, when `standardize` is on.
    scaler: Option<EwDiag>,
    /// The scaler's warm-up count, and whether it has switched the fit to
    /// the caller's units (the module docs; schema 47). Untouched without a
    /// scaler.
    warmup: Warmup,
    /// Coefficients per target: in the standardized coordinates of the
    /// moments as they stand while a scaler warms up, in the caller's
    /// units without one or after it (docs/PLAN.md task 206).
    beta: Vec<Vec<f64>>,
    /// AdaGrad accumulators per target (empty for the other schedules).
    g2: Vec<Vec<f64>>,
    w_sum: f64,
    /// Per target, the weight of the rows that carried it, decayed: what its
    /// `min_weight` is checked against (hard rule 8, docs/PLAN.md task 115
    /// (d)). `w_sum` stood in for it, so ten rows with a null target met
    /// `min_weight = 10` with every coefficient at zero.
    w_target: Vec<f64>,
    /// Per target, the EW variance of the out-of-sample residual, `s²`, and
    /// its weight: the scale `huber_delta` is in (the module docs;
    /// docs/PLAN.md task 195). Kept under the Huber loss only, and empty
    /// under every other.
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    /// Per target, the sketch of `|y − p|` whose median over
    /// [`crate::NORMAL_Q3`] is the Huber cut's scale under
    /// [`crate::HuberScale::Mad`]: fed the residuals and weights `s²` is,
    /// and aged alike. Empty under the RMS, under every other loss, and in
    /// a state written before schema 63 (docs/PLAN.md task 271).
    mad: Vec<crate::EwQuantile>,
    /// Per target, the EW mean and variance of the target itself: the unit
    /// `eps` is in (the module docs; docs/PLAN.md task 202). Kept under the
    /// epsilon-insensitive loss only, and empty under every other.
    spread: TargetSpread,
    /// Per target, the refit's buffer ([`SgdCfg::refit_rows`]): none when
    /// it is off, and in a state written before schema 64.
    refit: Vec<RefitBuf>,
    /// Per target, the recent steps' gains ([`StepGain`]), kept under
    /// [`SgdCfg::target_spread`] for a loss with a curvature and a rate
    /// shared by every slot ([`keeps_gain`]); none otherwise, and in a
    /// state written before schema 64.
    gain: Vec<StepGain>,
    /// Per target, the weight its level has been learned from ([`Level`]):
    /// the intercept takes the running mean's step while it is the larger
    /// ([`keeps_level`]; the module docs). None otherwise, and in a state
    /// written before schema 64.
    level: Vec<Level>,
    /// The standardized row `[1, z]` under a scaler; unused without one,
    /// where the model reads `x` in place.
    #[serde(skip)]
    zbuf: Vec<f64>,
    /// Once the fit is held in the caller's units, the rest of the row's
    /// map: the mean `m_i` and scale `s_i` each feature is standardized by
    /// (0 and 1 in the intercept's slot).
    #[serde(skip)]
    mbuf: Vec<f64>,
    #[serde(skip)]
    sbuf: Vec<f64>,
    /// The step in the caller's units, `Δb`, formed whole before any
    /// coefficient moves, and under AdaGrad each slot's new sum beside it;
    /// the slopes before a projection after it.
    #[serde(skip)]
    dbuf: Vec<f64>,
    #[serde(skip)]
    gbuf: Vec<f64>,
    /// The raw row `[1, x]` the scaler is updated with, under a scaler.
    #[serde(skip)]
    rawbuf: Vec<f64>,
    /// Scratch for the projection.
    #[serde(skip)]
    pbuf: crate::constraint::Scratch,
    /// Which targets stepped this row (only kept under a constraint).
    #[serde(skip)]
    learned: Vec<bool>,
    /// Per target, the label a row taught it with and the `|e|` its
    /// one-pass scale took (NaN: none), while a refit buffer is
    /// open ([`SgdCfg::refit_rows`]); empty otherwise. Held here, taken for
    /// the row and put back, so that a row allocates nothing for it.
    #[serde(skip)]
    taught: Vec<Option<(f64, f64)>>,
    /// Under a refit over a span of the clock (task 538): the model's place
    /// on the clock and the buffers' cap, each written only when it holds
    /// something, last.
    #[serde(default)]
    place: Option<crate::clock_span::Place>,
    #[serde(default)]
    over: Option<u64>,
    #[serde(default)]
    warned: bool,
    /// Optional state for `average`, `l1 > 0`, and `schedule == Optimal`;
    /// `None` when none of those options is set (zero bytes written).
    #[serde(default)]
    extras: Option<variants::SgdExtras>,
    #[serde(skip)]
    warning: Option<u64>,
    #[serde(skip)]
    cap_for_test: Option<u64>,
}

impl Serialize for Sgd {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let has_extras = self.extras.is_some();
        let has_warned = has_extras || self.warned;
        let has_over = has_warned || self.over.is_some();
        let has_place = has_over || self.place.is_some();
        let len = 14
            + usize::from(has_place)
            + usize::from(has_over)
            + usize::from(has_warned)
            + usize::from(has_extras);
        let mut st = serializer.serialize_struct("Sgd", len)?;
        st.serialize_field("cfg", &self.cfg)?;
        st.serialize_field("scaler", &self.scaler)?;
        st.serialize_field("warmup", &self.warmup)?;
        st.serialize_field("beta", &self.beta)?;
        st.serialize_field("g2", &self.g2)?;
        st.serialize_field("w_sum", &self.w_sum)?;
        st.serialize_field("w_target", &self.w_target)?;
        st.serialize_field("sig2", &self.sig2)?;
        st.serialize_field("wsig", &self.wsig)?;
        st.serialize_field("mad", &self.mad)?;
        st.serialize_field("spread", &self.spread)?;
        st.serialize_field("refit", &self.refit)?;
        st.serialize_field("gain", &self.gain)?;
        st.serialize_field("level", &self.level)?;
        if has_place {
            st.serialize_field("place", &self.place)?;
        }
        if has_over {
            st.serialize_field("over", &self.over)?;
        }
        if has_warned {
            st.serialize_field("warned", &self.warned)?;
        }
        if has_extras {
            st.serialize_field("extras", &self.extras)?;
        }
        st.end()
    }
}

/// The layout `Sgd` loads, checked on the way in: what the cfg asks for,
/// the state carries, and nothing else. (A schema-2 `scaler` was a full
/// `EwCov` read through its diagonal; the minimum schema has passed it,
/// docs/PLAN.md tasks 198 and 194-202, and [`EwDiag`] refuses an `EwCov`'s
/// fields and shape.)
#[derive(Deserialize)]
struct SgdV3 {
    cfg: SgdCfg,
    scaler: Option<EwDiag>,
    warmup: Warmup,
    beta: Vec<Vec<f64>>,
    g2: Vec<Vec<f64>>,
    w_sum: f64,
    w_target: Vec<f64>,
    sig2: Vec<f64>,
    wsig: Vec<f64>,
    #[serde(default)]
    mad: Vec<crate::EwQuantile>,
    spread: TargetSpread,
    #[serde(default)]
    refit: Vec<RefitBuf>,
    #[serde(default)]
    gain: Vec<StepGain>,
    #[serde(default)]
    level: Vec<Level>,
    #[serde(default)]
    place: Option<crate::clock_span::Place>,
    #[serde(default)]
    over: Option<u64>,
    #[serde(default)]
    warned: bool,
    #[serde(default)]
    extras: Option<variants::SgdExtras>,
}

impl TryFrom<SgdV3> for Sgd {
    type Error = String;

    fn try_from(v: SgdV3) -> Result<Self, String> {
        let (cfg, scaler, beta, g2, w_sum) = (v.cfg, v.scaler, v.beta, v.g2, v.w_sum);
        let (sig2, wsig, mad, spread) = (v.sig2, v.wsig, v.mad, v.spread);
        let (w_target, warmup, refit, gain) = (v.w_target, v.warmup, v.refit, v.gain);
        let level = v.level;
        let (place, over, warned, extras) = (v.place, v.over, v.warned, v.extras);
        let k = cfg.k_total();
        let m = cfg.n_targets;
        // What the cfg asks for, the state carries, and nothing else: a
        // scaler exactly when `standardize` is on, and AdaGrad's sums, one
        // per target, exactly under AdaGrad. A file that lost its scaler
        // loaded as a model reading raw inputs with coefficients learned on
        // standardized ones, and one that lost its sums panicked on the first
        // step (review 2026-09-12, S16).
        if cfg.standardize != scaler.is_some() {
            return Err(
                "sgd: its saved feature scaler does not match its standardize setting. \
                        Load an undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // A warm-up is a scaler's: without one it counts nothing, and a
        // state that says otherwise would read raw coefficients one way and
        // standardized ones another (docs/PLAN.md task 206).
        if scaler.is_none() && warmup != Warmup::default() {
            return Err(
                "sgd: its saved warm-up does not match its standardize setting. Load an \
                        undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        let sums = if matches!(cfg.schedule, LearningRate::AdaGrad) {
            m
        } else {
            0
        };
        if g2.len() != sums {
            let set = if sums > 0 {
                "schedule=\"adagrad\""
            } else {
                "schedule, which is not \"adagrad\""
            };
            return Err(format!(
                "sgd: its saved AdaGrad sums do not match its setting {set}. Load an undamaged \
                 copy of the state file, or rebuild it from its input."
            ));
        }
        if scaler.as_ref().is_some_and(|sc| sc.k() != k)
            || beta.len() != m
            || beta.iter().any(|b| b.len() != k)
            || g2.iter().any(|g| g.len() != k)
        {
            return Err(
                "sgd: its saved state has the wrong shape. Load an undamaged copy of the \
                        state file, or rebuild it from its input."
                    .into(),
            );
        }
        if w_target.len() != m {
            return Err(
                "sgd: its saved target weights have the wrong shape. Load an undamaged \
                        copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // The scales a loss draws in, one per target, exactly under the
        // loss that draws in it: the residual's under Huber, the target's
        // own under epsilon-insensitive (docs/PLAN.md task 202).
        let (residual, own) = scales_kept(&cfg);
        let scales = |kept: bool| if kept { m } else { 0 };
        if sig2.len() != scales(residual) || wsig.len() != scales(residual) {
            return Err(
                "sgd: its saved residual variances do not match its loss setting. Load \
                        an undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        if !spread.has_shape(scales(own)) {
            return Err(
                "sgd: its saved target spreads do not match its loss setting. Load an \
                        undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // The MAD's sketches, one per target at 0.5 exactly under it.
        let huber_scale = if residual {
            cfg.huber_scale
        } else {
            crate::HuberScale::Rms
        };
        if !crate::robust::mad_sketches_fit(huber_scale, cfg.mad_anchor, m, &mad) {
            return Err(
                "sgd: its saved residual scale does not match its huber_scale setting. Load \
                 an undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // The refit's buffers, one per target exactly when it is on, each
        // one this configuration writes.
        let refit_fits = if cfg.refit_rows == 0 && cfg.refit_clock.is_none() {
            refit.is_empty()
        } else {
            refit.len() == m
                && refit.iter().all(|b| {
                    b.fits(
                        cfg.n_features,
                        cfg.refit_cap(),
                        huber_scale == crate::HuberScale::Mad,
                    )
                })
        } && place.is_some() == cfg.refit_clock.is_some();
        if !refit_fits {
            return Err(
                "sgd: the rows it saved for its refit do not match its huber_refit_rows \
                 setting. Load an undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        // The held |e|'s tree is not state: reading each buffer built it
        // from the rows (`RefitBufFile`).
        if gain.len() != scales(keeps_gain(&cfg)) {
            return Err(
                "sgd: its saved learning-rate state does not match its settings. Load an undamaged \
                        copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        if level.len() != scales(keeps_level(&cfg))
            || !level.iter().all(|l| crate::sqsum::scale_fits(l.w2_scale))
        {
            return Err(
                "sgd: part of its saved fit does not match its settings. Load an \
                        undamaged copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        let needs_extras =
            cfg.average || cfg.l1 > 0.0 || matches!(cfg.schedule, LearningRate::Optimal);
        let extras_ok = match &extras {
            None => !needs_extras,
            Some(ex) => {
                needs_extras
                    && ex.avg.is_some() == cfg.average
                    && ex.avg.as_ref().is_none_or(|a| a.has_shape(m, k))
                    && ex.enet.is_some() == (cfg.l1 > 0.0)
                    && ex.enet.as_ref().is_none_or(|e| {
                        e.has_shape(
                            m,
                            cfg.n_features,
                            matches!(cfg.schedule, LearningRate::AdaGrad),
                        )
                    })
                    && ex.t_step.len()
                        == if matches!(cfg.schedule, LearningRate::Optimal) {
                            m
                        } else {
                            0
                        }
            }
        };
        if !extras_ok {
            return Err(
                "sgd: its saved variant state does not match its settings. Load an undamaged \
                 copy of the state file, or rebuild it from its input."
                    .into(),
            );
        }
        let sgd = Self {
            cfg,
            scaler,
            warmup,
            beta,
            g2,
            w_sum,
            w_target,
            sig2,
            wsig,
            mad,
            spread,
            refit,
            gain,
            level,
            zbuf: vec![],
            mbuf: vec![],
            sbuf: vec![],
            dbuf: vec![],
            gbuf: vec![],
            rawbuf: vec![],
            pbuf: crate::constraint::Scratch::default(),
            learned: Vec::new(),
            taught: Vec::new(),
            place,
            over,
            warned,
            extras,
            warning: None,
            cap_for_test: None,
        };
        sgd.check_loaded().map_err(|e| e.to_string())?;
        Ok(sgd)
    }
}

impl Sgd {
    pub fn new(cfg: SgdCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let m = cfg.n_targets;
        let g2 = if matches!(cfg.schedule, LearningRate::AdaGrad) {
            vec![vec![0.0; k]; m]
        } else {
            Vec::new()
        };
        let scaler = cfg.standardize.then(|| EwDiag::new(k));
        let mut beta = vec![vec![0.0; k]; m];
        if let Some(c) = &cfg.constraint {
            let off = usize::from(cfg.fit_intercept);
            let mut scratch = crate::constraint::Scratch::default();
            for b in beta.iter_mut() {
                c.project(&mut b[off..], None, &mut scratch);
            }
        }
        let (residual, own) = scales_kept(&cfg);
        let gain = vec![StepGain::default(); if keeps_gain(&cfg) { m } else { 0 }];
        let level = vec![Level::default(); if keeps_level(&cfg) { m } else { 0 }];
        let mad = if residual {
            crate::robust::mad_sketches(cfg.huber_scale, cfg.mad_anchor, m)
        } else {
            Vec::new()
        };
        let keeps_resid = !mad.is_empty();
        let residual = if residual { m } else { 0 };
        let needs_extras =
            cfg.average || cfg.l1 > 0.0 || matches!(cfg.schedule, LearningRate::Optimal);
        let extras = needs_extras.then(|| variants::SgdExtras {
            avg: cfg.average.then(|| variants::AvgState::new(&beta)),
            enet: (cfg.l1 > 0.0).then(|| {
                variants::EnetState::new(
                    m,
                    cfg.n_features,
                    matches!(cfg.schedule, LearningRate::AdaGrad),
                )
            }),
            t_step: if matches!(cfg.schedule, LearningRate::Optimal) {
                vec![0; m]
            } else {
                Vec::new()
            },
        });
        Ok(Self {
            scaler,
            warmup: Warmup::default(),
            beta,
            g2,
            w_sum: 0.0,
            w_target: vec![0.0; m],
            sig2: vec![0.0; residual],
            wsig: vec![0.0; residual],
            mad,
            spread: if own {
                TargetSpread::new(m)
            } else {
                TargetSpread::none()
            },
            refit: RefitBuf::per_target(
                cfg.refit_rows.max(u32::from(cfg.refit_clock.is_some())),
                m,
                keeps_resid,
            ),
            place: cfg.refit_clock.map(|_| crate::clock_span::Place::default()),
            over: None,
            warned: false,
            extras,
            warning: None,
            cap_for_test: None,
            gain,
            level,
            zbuf: vec![0.0; k],
            mbuf: vec![0.0; k],
            sbuf: vec![1.0; k],
            dbuf: vec![0.0; k],
            gbuf: vec![0.0; k],
            rawbuf: vec![0.0; k],
            pbuf: crate::constraint::Scratch::default(),
            learned: Vec::new(),
            taught: Vec::new(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &SgdCfg {
        &self.cfg
    }

    /// Coefficients in the caller's units. While a scaler warms up the
    /// model fits on standardized inputs, so they are unscaled here with
    /// the moments as they stand and the intercept absorbs the shift; once
    /// it has, they are the state (the module docs). When [`SgdCfg::average`]
    /// is true, returns the averaged coefficients.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        let coef = self
            .extras
            .as_ref()
            .and_then(|e| e.avg.as_ref())
            .map_or(&self.beta, |a| &a.beta);
        match &self.scaler {
            Some(sc) if !self.warmup.switched() => unscaled(coef, sc, self.cfg.fit_intercept),
            // Held in the caller's units: the state itself (the module
            // docs).
            _ => coef.clone(),
        }
    }

    /// The scaler's warm-up ([`crate::Warmup`]): how many rows it has
    /// learned, and whether the fit is held in the caller's units.
    pub fn warmup(&self) -> &Warmup {
        &self.warmup
    }

    /// Whether the fit is still the warm-up's: a scaler that has not yet
    /// switched the fit to the caller's units. `false` without one.
    pub fn scaler_warming(&self) -> bool {
        self.scaler.is_some() && !self.warmup.switched()
    }

    /// Per-slot scale: the running sd for features, 1 for the intercept and for
    /// a feature with no spread yet.
    fn scales(&self) -> Vec<f64> {
        match &self.scaler {
            None => vec![1.0; self.cfg.k_total()],
            Some(sc) => scales_of(sc, usize::from(self.cfg.fit_intercept)),
        }
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Per target, the weight of the rows that carried it: what its
    /// `min_weight` is checked against.
    pub fn target_weights(&self) -> &[f64] {
        &self.w_target
    }

    /// `p = link(eta)`; the Poisson link is `exp(clamp(eta, −30, 30))` (the
    /// module docs).
    fn link(&self, eta: f64) -> f64 {
        match self.cfg.loss {
            SgdLoss::Poisson => eta.clamp(-30.0, 30.0).exp(),
            SgdLoss::Logistic => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }
            _ => eta,
        }
    }

    /// Per target, the EW variance of the out-of-sample residual, `s²`,
    /// under the Huber loss; empty under every other (the module docs).
    pub fn sigma2(&self) -> &[f64] {
        &self.sig2
    }

    /// Per target, how its refit stands ([`SgdCfg::refit_rows`]): empty
    /// when the refit is off.
    pub fn refit_status(&self) -> Vec<crate::RefitStatus> {
        self.refit.iter().map(RefitBuf::status).collect()
    }

    /// The targets whose refit took every one of its passes
    /// ([`crate::RefitStatus::ran_to_cap`]): what a bank reports as this
    /// model's `solve_failures`, as `huber` counts the same.
    pub fn solve_failures(&self) -> u64 {
        self.refit
            .iter()
            .filter(|r| r.status().ran_to_cap())
            .count() as u64
    }

    /// Give every target's refit up before it begins, past the spec's
    /// `refit_budget` ([`SgdCfg::refit_rows`]): a bank does this to a
    /// group whose reservation the budget cannot take, and to the models
    /// a reset rebuilds for it.
    pub fn refit_give_up(&mut self) {
        huber_refit::give_up_over_budget(&mut self.refit, &mut self.mad, self.cfg.n_features);
    }

    /// Target `j`'s MAD sketch built from what its buffer held
    /// ([`crate::huber_refit::Taken::rebuild_sketch`]). Never inlined: in
    /// the step it is the ageing row's rare branch, and inlined there it
    /// moved a register spill onto every row's path (task 345).
    #[inline(never)]
    fn scale_from_held(&mut self, j: usize, t: &huber_refit::Taken) {
        t.rebuild_sketch(self.mad.get_mut(j), self.cfg.n_features);
    }

    /// The bytes this model's refit buffers can hold at most
    /// ([`crate::refit_reservation`] a target), 0 with the refit off; under
    /// a refit over a span of the clock, the group's cap.
    pub fn refit_reservation(&self) -> usize {
        if self.refit.is_empty() {
            return 0;
        }
        if self.cfg.refit_clock.is_some() {
            return usize::try_from(
                self.cfg
                    .max_state_bytes
                    .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES),
            )
            .unwrap_or(usize::MAX);
        }
        self.cfg.n_targets
            * crate::refit_reservation(
                self.cfg.n_features,
                self.cfg.refit_rows,
                !self.mad.is_empty(),
            )
    }

    /// Per target, the sketch of `|e|` the MAD scale is read from, under
    /// `huber_scale = "mad"`; empty otherwise.
    pub fn mad_sketch(&self) -> &[crate::EwQuantile] {
        &self.mad
    }

    /// Per target, the EW variance of the target itself, `s_y²`, under the
    /// epsilon-insensitive loss, and under [`SgdCfg::target_spread`] the
    /// squared and Huber losses; empty under every other (the module docs).
    pub fn target_variance(&self) -> Vec<f64> {
        self.spread.vars()
    }

    /// Whether some target's predictions are withheld because its recent
    /// steps' gains say the learning rate is past its stability bound
    /// ([`StepGain::unstable`]; the module docs). Read from the state, so
    /// the next row's prediction -- by `step` or `predict` -- is withheld
    /// exactly when this says so. `false` outside
    /// [`SgdCfg::target_spread`].
    pub fn unstable(&self) -> bool {
        self.gain.iter().any(StepGain::unstable)
    }

    /// Value and cap invariants enforced on both load paths (`SgdV3::try_from`
    /// and `OnlineModel::restore`; review 2026-11-23, STATE-3).
    fn check_loaded(&self) -> Result<(), StateError> {
        if self.over.is_some() && self.cfg.refit_clock.is_none() {
            return Err(crate::model::damaged(
                "sgd",
                "the rows it saved for its refit do not match its huber_refit_rows setting",
            ));
        }
        if !self.gain.iter().all(StepGain::is_valid) {
            return Err(crate::model::damaged(
                "sgd",
                "its saved learning-rate state holds values no stream writes",
            ));
        }
        if self.cfg.refit_clock.is_some() && self.refit.iter().any(RefitBuf::has_held) {
            let bytes: u64 = self
                .refit
                .iter()
                .fold(0u64, |acc, b| acc.saturating_add(b.held_bytes()));
            let cap = self.span_cap();
            if bytes > cap {
                return Err(crate::model::damaged(
                    "sgd",
                    format!("it holds {bytes} bytes, past max_state_bytes of {cap}"),
                ));
            }
        }
        Ok(())
    }

    /// Per target, [`Sgd::unstable`]'s verdict: empty where no gains are
    /// kept.
    pub fn unstable_targets(&self) -> Vec<bool> {
        self.gain.iter().map(StepGain::unstable).collect()
    }

    /// The unit target `j`'s gradient is in, as the row arrives, for its
    /// clip and AdaGrad's epsilon: under [`SgdCfg::target_spread`] for the
    /// squared and Huber losses the target's own EW standard deviation
    /// `s_y`, 0 before it has one; 1 otherwise, where the clip and the
    /// epsilon are in the gradient's own units (the module docs).
    fn unit(&self, j: usize) -> f64 {
        if in_target_units(&self.cfg) {
            self.spread.band(j, 1.0)
        } else {
            1.0
        }
    }

    /// The unit target `j`'s cut or tube is drawn in, as the row arrives:
    /// under Huber the residual's scale `s = √s²` once `s²` is finite and
    /// above 0; under epsilon-insensitive the target's own `s_y` once it has
    /// a spread. `None` before -- and under every other loss -- where there
    /// is no cut and no tube to draw (`huber`'s rule, `Robust`;
    /// [`crate::spread`]).
    fn scale(&self, j: usize) -> Option<f64> {
        match self.cfg.loss {
            SgdLoss::Huber { .. } => {
                if let Some(q) = self.mad.get(j) {
                    // Exact from the one-pass `|e|` the refit's buffer
                    // holds while the refit waits, as `huber`'s.
                    if let Some(s) = self.refit.get(j).and_then(RefitBuf::warm_scale) {
                        return s;
                    }
                    return crate::robust::mad_scale(q);
                }
                let s2 = self.sig2[j];
                (s2 > 0.0 && s2.is_finite()).then(|| s2.sqrt())
            }
            SgdLoss::EpsilonInsensitive { .. } => {
                let unit = self.spread.band(j, 1.0);
                (unit > 0.0).then_some(unit)
            }
            _ => None,
        }
    }

    /// `dL/d(eta)` for the configured loss, with `scale` the unit its cut
    /// or tube is drawn in ([`Sgd::scale`]; `None` before it has one: no
    /// cut, and a tube of no width).
    fn dloss(&self, p: f64, y: f64, scale: Option<f64>) -> f64 {
        let r = p - y;
        match self.cfg.loss {
            SgdLoss::Squared | SgdLoss::Poisson | SgdLoss::Logistic => r,
            SgdLoss::Huber { delta } => match scale {
                Some(s) => r.clamp(-delta * s, delta * s),
                None => r,
            },
            SgdLoss::Quantile { tau } => f64::from(y < p) - tau,
            SgdLoss::EpsilonInsensitive { eps } => {
                if r.abs() <= scale.map_or(0.0, |s| eps * s) {
                    0.0
                } else {
                    r.signum()
                }
            }
        }
    }

    /// The label a row teaches target `j` with: `y` itself, except under
    /// the logistic loss, where a label outside {0, 1} is clamped into
    /// `[0, 1]`, or under `strict_binary` teaches nothing (`None`), `ftrl`'s
    /// rule (docs/PLAN.md task 195, S4); and under the Poisson loss, where a
    /// negative count teaches nothing (docs/PLAN.md task 209 (e)).
    fn label(&self, y: f64) -> Option<f64> {
        teaches(self.cfg.loss, self.cfg.strict_binary, y).then(|| match self.cfg.loss {
            SgdLoss::Logistic => y.clamp(0.0, 1.0),
            _ => y,
        })
    }

    fn ensure_buffers(&mut self) {
        let k = self.cfg.k_total();
        if self.zbuf.len() != k {
            self.zbuf = vec![0.0; k];
        }
        if self.mbuf.len() != k {
            self.mbuf = vec![0.0; k];
            self.sbuf = vec![1.0; k];
            self.dbuf = vec![0.0; k];
            self.gbuf = vec![0.0; k];
        }
        if self.rawbuf.len() != k {
            self.rawbuf = vec![0.0; k];
        }
    }

    /// Target `j`'s step once the fit is held in the caller's units (the
    /// module docs): the gradient in the row's standardized coordinates,
    /// `g_i = d z_i w + l2 β_i` with `β_i = s_i b_i` and `g_0 = d w`, each
    /// clipped; `Δβ_i = −lr_i g_i`; mapped into the caller's units by the
    /// row's map, `Δb_i = Δβ_i / s_i` and `Δb_0 = Δβ_0 − Σ_i m_i Δb_i`. The
    /// step is formed whole, in `dbuf` (and AdaGrad's new sums in `gbuf`),
    /// before anything moves: a row any of whose gradients is not finite
    /// teaches nothing, as before the switch, and so does one any of whose
    /// coefficients the step would make infinite -- `Δβ_i / s_i` for a
    /// feature of all but no spread. `false` then.
    ///
    /// Out of line, so its loop has the registers to itself: inlined into
    /// `step_into`, whose frame holds the whole row's state, the loop read
    /// the weight and the rate from the stack and reloaded a pointer on
    /// every feature (task 288: x86-64's loop 29 instructions with three
    /// stack reads, 28 with none; `step_into`'s frame 728 bytes to 648).
    #[inline(never)]
    fn step_mapped(
        &mut self,
        j: usize,
        d: f64,
        weight: f64,
        lr_row: Option<f64>,
        (clip, ada_eps): (f64, f64),
        level_step: Option<f64>,
    ) -> bool {
        let off = usize::from(self.cfg.fit_intercept);
        let (lr0, l2) = (self.cfg.learning_rate, self.cfg.l2);
        let Self {
            beta,
            g2,
            zbuf,
            mbuf,
            sbuf,
            dbuf,
            gbuf,
            ..
        } = self;
        // Every slice cut to the coefficients' length, so that the loop
        // below indexes each without a bounds check.
        let b = beta[j].as_mut_slice();
        let n = b.len();
        let (zs, ss, ms, ds) = (&zbuf[..n], &sbuf[..n], &mbuf[..n], &mut dbuf[..n]);
        let sums: &[f64] = g2.get(j).map_or(&[], |v| v.as_slice());
        // `Δβ` for slot `i` from its gradient, its new AdaGrad sum into
        // `gbuf`; `None` where the gradient is not usable.
        let mut delta = |i: usize, g: f64| -> Option<f64> {
            match lr_row {
                Some(lr) => g.is_finite().then(|| -(lr * g)),
                None => {
                    // The sum with the square in it, not the square alone:
                    // a sum past the largest double is kept for good
                    // (REG-9).
                    let s = sums[i] + g * g;
                    if !s.is_finite() {
                        return None;
                    }
                    gbuf[i] = s;
                    Some(-adagrad_step(lr0, s, ada_eps, g))
                }
            }
        };
        let mut shift = 0.0;
        for i in off..n {
            let g = (d * zs[i] * weight + l2 * (ss[i] * b[i])).clamp(-clip, clip);
            let step = match lr_row {
                // No test of `g` itself at a shared rate: a `g` that is not
                // finite leaves `−lr·g` infinite or NaN whatever `lr`, and
                // so `Δb` whatever the scale and `b + Δb` whatever `b`,
                // which the test below refuses on the same slot.
                Some(lr) => -(lr * g),
                None => {
                    let Some(step) = delta(i, g) else {
                        return false;
                    };
                    step
                }
            };
            let db = step / ss[i];
            if !(b[i] + db).is_finite() {
                return false;
            }
            ds[i] = db;
            shift += ms[i] * db;
        }
        if off == 1 {
            // The level's step in place of the gradient's while it is the
            // larger; AdaGrad's sum learns the gradient either way.
            let Some(step) = delta(0, (d * weight).clamp(-clip, clip)) else {
                return false;
            };
            let db = level_step.unwrap_or(step) - shift;
            if !(b[0] + db).is_finite() {
                return false;
            }
            ds[0] = db;
        }
        for (bi, di) in b.iter_mut().zip(ds.iter()) {
            *bi += di;
        }
        if let Some(sums) = g2.get_mut(j) {
            sums.copy_from_slice(gbuf);
        }
        true
    }

    /// Target `j` took its step: its level's weight with the row in it
    /// becomes its own, and the step's gain joins the unstable rate's sums
    /// ([`StepGain`]). Neither moves for a row the step refused, which
    /// [`Sgd::refused_step`] handles instead.
    fn took_step(&mut self, j: usize, level: Option<Level>, gain: Option<f64>) {
        if let Some(lv) = level {
            self.level[j] = lv;
        }
        if let Some(q) = gain {
            let may_rest = in_target_units(&self.cfg);
            self.gain[j].learn(q, may_rest);
        }
    }

    /// Predict into `out` without mutating state, at `weight`'s warm-up read.
    fn predict_step_into(&self, x: &[f64], d_clock: f64, weight: f64, out: &mut Step) {
        out.n_eff = self.w_sum;
        out.extra = None;
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets, f64::NAN);
        if self.w_target.iter().any(|w| *w >= self.cfg.min_weight) {
            let off = usize::from(self.cfg.fit_intercept);
            let lam = self.cfg.decay.factor(d_clock);
            let coef = self
                .extras
                .as_ref()
                .and_then(|e| e.avg.as_ref())
                .map_or(&self.beta, |a| &a.beta);
            let predict_with = |z: &[f64], pred: &mut [f64]| {
                for (j, p) in pred.iter_mut().enumerate() {
                    if self.w_target[j] >= self.cfg.min_weight
                        && !self.gain.get(j).is_some_and(StepGain::unstable)
                    {
                        *p = self.link(dot(&coef[j], off, z));
                    }
                }
            };
            match &self.scaler {
                None => predict_with(x, &mut out.pred),
                Some(_) if self.warmup.switched() => predict_with(x, &mut out.pred),
                Some(sc) => crate::scratch::with_borrow_mut(&PREDICT_Z, |z| {
                    let (admit, floor) = self.warmup.read(weight);
                    z.clear();
                    z.extend(standardized(sc, off, x, lam, admit, floor));
                    predict_with(z, &mut out.pred);
                }),
            }
        }
    }

    /// Exact span refit bytes added by this row, computed read-only only when
    /// the fast upper bound exceeds the cap.
    #[cold]
    #[allow(clippy::too_many_arguments)]
    fn span_row_delta(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        weight: f64,
        lam: f64,
        lam_scale: f64,
        per: u64,
    ) -> u64 {
        let m = self.cfg.n_targets;
        let off = usize::from(self.cfg.fit_intercept);
        let held = self.warmup.switched();
        let (admit, floor) = self.warmup.read(weight);
        let n = self.cfg.n_features;
        let mut zbuf = Vec::new();
        let mut mbuf = Vec::new();
        let mut sbuf = Vec::new();
        if let Some(sc) = &self.scaler {
            zbuf.resize(off + n, 0.0);
            if off == 1 {
                zbuf[0] = 1.0;
            }
            if held {
                mbuf.resize(off + n, 0.0);
                sbuf.resize(off + n, 1.0);
                standardize_row(
                    sc,
                    off,
                    x,
                    lam,
                    admit,
                    floor,
                    &mut zbuf[off..],
                    &mut mbuf[off..],
                    &mut sbuf[off..],
                );
            } else {
                for (z, v) in zbuf[off..]
                    .iter_mut()
                    .zip(standardized(sc, off, x, lam, admit, floor))
                {
                    *z = v;
                }
            }
        }
        let reads_z = self.scaler.is_some() && !held;
        let z: &[f64] = if self.scaler.is_some() {
            &zbuf[off..]
        } else {
            x
        };
        let lr_row_shared = match self.cfg.schedule {
            LearningRate::Constant => Some(self.cfg.learning_rate),
            LearningRate::InvScaling { power } => {
                Some(self.cfg.learning_rate / (1.0 + lam * self.w_sum).powf(power))
            }
            LearningRate::AdaGrad | LearningRate::Optimal => None,
        };
        let (lr0, l1, l2, clip0) = (
            self.cfg.learning_rate,
            self.cfg.l1,
            self.cfg.l2,
            self.cfg.clip_gradient,
        );
        let is_optimal = matches!(self.cfg.schedule, LearningRate::Optimal);
        let is_variant = l1 > 0.0 || self.cfg.invariant || is_optimal;
        let dec = |s: f64| if lam != 1.0 { lam * s } else { s };
        let mut delta = 0u64;
        for (j, &yj_opt) in y.iter().enumerate().take(m) {
            if !self.refit[j].open() {
                continue;
            }
            let lr_row = if is_optimal {
                let t = self
                    .extras
                    .as_ref()
                    .and_then(|e| e.t_step.get(j).copied())
                    .unwrap_or(0);
                Some(variants::optimal_rate(l1 + l2, t))
            } else {
                lr_row_shared
            };
            let step_ok = (|| {
                let yj = yj_opt?;
                if weight <= 0.0 || !yj.is_finite() {
                    return None;
                }
                let yj = self.label(yj)?;
                let p = self.link(dot(&self.beta[j], off, if reads_z { z } else { x }));
                let scale = self.scale(j);
                let unit = self.unit(j);
                let clip = if unit > 0.0 {
                    clip0 * unit
                } else {
                    f64::INFINITY
                };
                let ada_eps = 1e-8 * unit;
                let d = self.dloss(p, yj, scale);
                let mut level_step = None;
                if let Some(&lv) = self.level.get(j) {
                    let first = lv.w == 0.0;
                    let mut lv = lv;
                    lv.learn(weight);
                    let gain = weight / lv.w;
                    let (step, warm) =
                        if matches!(self.cfg.loss, SgdLoss::Squared | SgdLoss::Huber { .. }) {
                            let above = lr_row.is_some_and(|lr| gain > lr);
                            (-gain * d, above || lv.count() <= crate::WARMUP_ROWS)
                        } else {
                            (gain * (yj - p), lv.count() <= crate::WARMUP_ROWS)
                        };
                    if first {
                        return (self.beta[j][0] + step).is_finite().then_some(());
                    }
                    if warm && step.is_finite() {
                        level_step = Some(step);
                    }
                }
                let b = &self.beta[j];
                let sums: &[f64] = self.g2.get(j).map_or(&[], Vec::as_slice);
                if is_variant {
                    let zz_row = off as f64 + dot(z, 0, z);
                    let u_cur = self
                        .extras
                        .as_ref()
                        .and_then(|e| e.enet.as_ref())
                        .map(|e| e.u[j].as_slice());
                    let q_cur = self
                        .extras
                        .as_ref()
                        .and_then(|e| e.enet.as_ref())
                        .map(|e| e.q[j].as_slice());
                    let mut next_g2 = if sums.is_empty() {
                        Vec::new()
                    } else {
                        sums.iter().map(|&s| dec(s)).collect()
                    };
                    let mut next_u = vec![0.0; u_cur.map_or(0, <[f64]>::len)];
                    let mut next_q = vec![0.0; q_cur.map_or(0, <[f64]>::len)];
                    let g2_dec = (!next_g2.is_empty()).then_some(next_g2.clone());
                    if held {
                        let mut next_b = vec![0.0; b.len()];
                        return variants::step_mapped_variant(
                            b,
                            off,
                            &zbuf,
                            &sbuf,
                            &mbuf,
                            d,
                            weight,
                            (l1, l2, clip, ada_eps),
                            lr_row,
                            lr0,
                            self.cfg.invariant,
                            zz_row,
                            level_step,
                            g2_dec.as_deref(),
                            u_cur,
                            q_cur,
                            &mut next_b,
                            &mut next_g2,
                            &mut next_u,
                            &mut next_q,
                        )
                        .then_some(());
                    }
                    let mut next_slopes = vec![0.0; z.len()];
                    return variants::step_direct_variant(
                        b,
                        off,
                        z,
                        d,
                        weight,
                        (l1, l2, clip, ada_eps),
                        lr_row,
                        lr0,
                        self.cfg.invariant,
                        zz_row,
                        level_step,
                        g2_dec.as_deref(),
                        u_cur,
                        q_cur,
                        &mut next_slopes,
                        &mut next_g2,
                        &mut next_u,
                        &mut next_q,
                    )
                    .map(|_| ());
                }
                if held {
                    let mut shift = 0.0;
                    for i in off..b.len() {
                        let g = (d * zbuf[i] * weight + l2 * (sbuf[i] * b[i])).clamp(-clip, clip);
                        let step = match lr_row {
                            Some(lr) => -(lr * g),
                            None => {
                                let s = dec(sums[i]) + g * g;
                                if !s.is_finite() {
                                    return None;
                                }
                                -adagrad_step(lr0, s, ada_eps, g)
                            }
                        };
                        let db = step / sbuf[i];
                        if !(b[i] + db).is_finite() {
                            return None;
                        }
                        shift += mbuf[i] * db;
                    }
                    if off == 1 {
                        let g = (d * weight).clamp(-clip, clip);
                        let step = match lr_row {
                            Some(lr) => g.is_finite().then(|| -(lr * g))?,
                            None => {
                                let s = dec(sums[0]) + g * g;
                                if !s.is_finite() {
                                    return None;
                                }
                                -adagrad_step(lr0, s, ada_eps, g)
                            }
                        };
                        if !(b[0] + level_step.unwrap_or(step) - shift).is_finite() {
                            return None;
                        }
                    }
                    return Some(());
                }
                if off == 1 {
                    let g = (d * weight).clamp(-clip, clip);
                    let ok = if lr_row.is_some() {
                        g.is_finite()
                    } else {
                        (dec(sums[0]) + g * g).is_finite()
                    };
                    if !ok {
                        return None;
                    }
                }
                let slopes_ok = if lr_row.is_some() {
                    slopes_usable(&b[off..], z, d, weight, l2, clip, None)
                } else {
                    b[off..]
                        .iter()
                        .zip(z)
                        .zip(&sums[off..])
                        .all(|((bi, &zi), &s)| {
                            let g = (d * zi * weight + l2 * *bi).clamp(-clip, clip);
                            (dec(s) + g * g).is_finite()
                        })
                };
                slopes_ok.then_some(())
            })();
            delta += if step_ok.is_some() {
                per
            } else {
                self.refit[j].age_bytes(lam, lam_scale)
            };
        }
        delta
    }

    /// Target `j`'s step was refused: a gate that rested as the row found
    /// it (`found`, its rest count then) rests again with that count, so a
    /// refused row neither counts toward a probe nor wakes the gate. The
    /// row's rest count and any wake are made on the gate in place before
    /// the step, as the gain's verdict reads them, and undone here only on
    /// this rare path: a copy of the gate taken and written back on every
    /// row cost the 20-feature squared-loss step 0.9% of its instructions
    /// on arm64, natively, and 1.9% on x86-64 under Valgrind (task 347). An
    /// awake gate moves only in [`Sgd::took_step`], so a refused row leaves
    /// it as it was.
    #[cold]
    fn refused_step(&mut self, j: usize, found: Option<u32>) {
        if let Some(rested) = found {
            let g = &mut self.gain[j];
            g.resting = true;
            g.rested = rested;
        }
    }

    /// The refit of target `j` ([`SgdCfg::refit_rows`],
    /// [`crate::huber_refit`]): the converged Huber fit of its buffered
    /// rows, by IRLS from their weighted least-squares fit, in the caller's
    /// units, becomes its coefficients -- while a scaler warms up, in the
    /// coordinates of its moments as they stand (`b_i = β_i s_i`,
    /// `b_0 = β_0 + Σ m_i β_i`), so they read out as the fit exactly -- and
    /// under a constraint is projected on it as a step is. Its residual
    /// scale is rebuilt from the fit's residuals. The learning rate's state
    /// stays: AdaGrad's sums of squared gradients keep the step sizes the
    /// rows taught, where emptied they would make the next steps throw the
    /// fit off, and `inv_scaling` reads the weight, which a refit does not
    /// move (hard rule 8). The fit takes `l2` as the steps apply it, a
    /// penalty on each standardized slope once a row ([`huber_refit::irls`]);
    /// the steps after the refit go on applying it.
    #[inline(never)]
    fn refit_target(&mut self, j: usize) {
        let SgdLoss::Huber { delta } = self.cfg.loss else {
            return;
        };
        let intercept = self.cfg.fit_intercept;
        let off = usize::from(intercept);
        let held = self.refit[j].take();
        let rows = Rows::new(&held.rows, &held.ages, self.cfg.n_features);
        // `l2` is a penalty on the standardized slope `β_i = s_i b_i` a step
        // applies once a row, so in the caller's units it is `l2 · s_i²`
        // a row; and the ridge, on the same slope, is `REFIT_RIDGE · s_i²`.
        // Without a scaler `s_i` is 1.
        let s: Vec<f64> = match &self.scaler {
            Some(sc) => scales_of(sc, off)[off..].to_vec(),
            None => vec![1.0; self.cfg.n_features],
        };
        let l2: Vec<f64> = s.iter().map(|si| self.cfg.l2 * si * si).collect();
        let ridge: Vec<f64> = if self.scaler.is_some() {
            s.iter().map(|si| REFIT_RIDGE * si * si).collect()
        } else {
            vec![REFIT_RIDGE; self.cfg.n_features]
        };
        let Some((fit, passes)) =
            huber_refit::irls(&rows, intercept, delta, self.cfg.huber_scale, &ridge, &l2)
        else {
            // No pass solved: the one-pass fit stands, and the refit says
            // so; its scale the sketch from the one-pass `|e|` held.
            self.refit[j].fail();
            self.scale_from_held(j, &held);
            return;
        };
        self.refit[j].ran(passes);
        let scale = huber_refit::rebuild_scale(&rows, self.mad.get(j), |i| {
            rows.y(i) - huber_refit::fitted(&fit, rows.x(i), intercept)
        });
        match &self.scaler {
            Some(sc) if !self.warmup.switched() => {
                let s = scales_of(sc, off);
                let mut b: Vec<f64> = fit.iter().zip(&s).map(|(f, si)| f * si).collect();
                if intercept {
                    // `b_0 = β_0 + Σ m_i β_i`, the intercept's scale being 1.
                    b[0] = fit[0];
                    for (i, f) in fit.iter().enumerate().skip(1) {
                        b[0] += f * sc.mean(i);
                    }
                }
                self.beta[j] = b;
                if let Some(c) = &self.cfg.constraint {
                    c.project(&mut self.beta[j][off..], Some(&s[off..]), &mut self.pbuf);
                }
            }
            Some(_) => {
                self.beta[j] = fit;
                if self.cfg.constraint.is_some() {
                    self.project_mapped(j);
                }
            }
            None => {
                self.beta[j] = fit;
                if let Some(c) = &self.cfg.constraint {
                    c.project(&mut self.beta[j][off..], None, &mut self.pbuf);
                }
            }
        }
        self.sig2[j] = scale.sig2;
        self.wsig[j] = scale.wsig;
        if let Some(q) = scale.mad {
            self.mad[j] = q;
        }
        if let Some(ex) = self.extras.as_mut() {
            if let Some(avg) = ex.avg.as_mut() {
                avg.beta[j].clone_from(&self.beta[j]);
            }
            if let Some(enet) = ex.enet.as_mut() {
                enet.reset_target(j);
            }
        }
    }

    /// Project target `j`'s slopes, held in the caller's units, on the
    /// constraint ([`SgdCfg::constraint`]): in the metric of the row's
    /// standardized coordinates ([`Constraint::project_in_units`]), the
    /// intercept keeping its standardized value `b_0 + Σ m_i b_i`, so it
    /// takes `Σ m_i (b_i − b'_i)` from the slopes' move.
    fn project_mapped(&mut self, j: usize) {
        let off = usize::from(self.cfg.fit_intercept);
        let Self {
            cfg,
            beta,
            pbuf,
            mbuf,
            sbuf,
            dbuf,
            ..
        } = self;
        let c = cfg.constraint.as_ref().expect("a constraint to project on");
        let b = &mut beta[j];
        dbuf[off..].copy_from_slice(&b[off..]);
        c.project_in_units(&mut b[off..], &sbuf[off..], pbuf);
        if off == 1 {
            let mut shift = 0.0;
            for i in 1..b.len() {
                shift += mbuf[i] * (dbuf[i] - b[i]);
            }
            b[0] += shift;
        }
    }
}

/// Which scales `cfg` draws in: `(residual, own)`, the residual's spread
/// for the Huber cut and the target's own for the epsilon-insensitive tube,
/// and under [`SgdCfg::target_spread`] for the squared and Huber losses'
/// clip and AdaGrad epsilon (the module docs). A state keeps each, one per
/// target, exactly where it is read.
fn scales_kept(cfg: &SgdCfg) -> (bool, bool) {
    (
        matches!(cfg.loss, SgdLoss::Huber { .. }),
        matches!(cfg.loss, SgdLoss::EpsilonInsensitive { .. }) || in_target_units(cfg),
    )
}

/// Whether `cfg`'s clip and AdaGrad epsilon are in units of the target's
/// own spread: under [`SgdCfg::target_spread`], the squared and Huber
/// losses, whose gradient is in the target's units. The other losses'
/// gradients are not: a sign, a count's residual, a probability's.
fn in_target_units(cfg: &SgdCfg) -> bool {
    cfg.target_spread && matches!(cfg.loss, SgdLoss::Squared | SgdLoss::Huber { .. })
}

/// Whether `cfg` keeps each target's step gains ([`StepGain`]): under
/// [`SgdCfg::target_spread`], for a loss with a curvature -- not the
/// quantile and epsilon-insensitive losses, whose step is a sign times the
/// rate and cannot grow -- and a rate shared by every slot, not AdaGrad's,
/// which divides each slot's step by the root of its own squared
/// gradients, so that a slot moves at most the rate a row.
fn keeps_gain(cfg: &SgdCfg) -> bool {
    cfg.target_spread
        && !cfg.invariant
        && !matches!(cfg.schedule, LearningRate::AdaGrad)
        && matches!(
            cfg.loss,
            SgdLoss::Squared | SgdLoss::Huber { .. } | SgdLoss::Poisson | SgdLoss::Logistic
        )
}

/// Whether `cfg` warms each target's level ([`Sgd::level`]): under
/// [`SgdCfg::target_spread`], with an intercept, under an identity link --
/// a prediction in the target's units, whose level the intercept carries.
/// Disabled under `invariant=True`, where the intercept participates in the
/// joint importance-invariant projection from row 0.
fn keeps_level(cfg: &SgdCfg) -> bool {
    cfg.target_spread
        && !cfg.invariant
        && cfg.fit_intercept
        && !matches!(cfg.loss, SgdLoss::Poisson | SgdLoss::Logistic)
}

/// AdaGrad's step for a slot, `lr / (√G + eps) · g`, `G` its sum of
/// squared gradients with the row's in it; 0 where the denominator is 0,
/// which with `G >= g²` needs `g²` to underflow, a gradient below `1e-154`
/// (and `eps` 0: under [`SgdCfg::target_spread`] it is `1e-8` times the
/// gradient's unit, 0 before the target has a spread).
#[inline]
fn adagrad_step(lr: f64, sum: f64, eps: f64, g: f64) -> f64 {
    let den = sum.sqrt() + eps;
    if den > 0.0 { lr / den * g } else { 0.0 }
}

/// Whether a finite target `y` teaches at all under `loss`: not a label
/// `strict_binary` refuses, nor a negative count under the Poisson loss,
/// which `p − y` would drive to the link's clamp, `e^−30`, and hold there
/// (docs/PLAN.md task 209 (e)). The bank never hands the model either: it
/// refuses the chunk, naming the row. `-0.0` is not below zero, so it is the
/// count 0.
fn teaches(loss: SgdLoss, strict_binary: bool, y: f64) -> bool {
    match loss {
        SgdLoss::Logistic if strict_binary => y == 0.0 || y == 1.0,
        SgdLoss::Poisson => y >= 0.0,
        _ => true,
    }
}

thread_local! {
    /// `predict`'s standardized row (see there).
    static PREDICT_Z: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The scaler's per-slot scale, `k` slots with `off` of them the
/// intercept: the running sd for a feature, 1 for the intercept and for a
/// feature with no spread yet; through the origin (`off = 0`) the root of
/// the raw second moment, the scale the step standardizes with (C13).
/// `pa` reads it too (docs/PLAN.md task 195, U2).
pub(crate) fn scales_of(sc: &EwDiag, off: usize) -> Vec<f64> {
    (0..sc.k())
        .map(|i| {
            if i < off {
                return 1.0;
            }
            if off == 0 {
                // No intercept: the raw second moment's scale, the
                // one the step standardizes with (C13).
                let raw = sc.raw(i);
                return if raw > 0.0 { raw.sqrt() } else { 1.0 };
            }
            let v = sc.var(i);
            if crate::variance_is_usable(v, sc.raw(i)) {
                v.sqrt()
            } else {
                1.0
            }
        })
        .collect()
}

/// Coefficients learned in the coordinates `sc` standardizes into, read
/// out in the caller's units with the moments as they stand: each slope
/// over its feature's scale, and the intercept absorbing the shift of the
/// means. `pa` reads them out the same way (docs/PLAN.md task 195, U2).
pub(crate) fn unscaled(beta: &[Vec<f64>], sc: &EwDiag, fit_intercept: bool) -> Vec<Vec<f64>> {
    let off = usize::from(fit_intercept);
    let scales = scales_of(sc, off);
    let k = scales.len();
    beta.iter()
        .map(|b| {
            let mut c = vec![0.0; k];
            for i in off..k {
                c[i] = b[i] / scales[i];
            }
            if fit_intercept {
                let mut b0 = b[0];
                for (i, ci) in c.iter().enumerate().skip(off) {
                    b0 -= ci * sc.mean(i);
                }
                c[0] = b0;
            }
            c
        })
        .collect()
}

/// Whether every slope's gradient `g_i = clip(d·z_i·w + l2·β_i)` is a
/// finite number -- under AdaGrad (`sums`, its sums of squared gradients
/// for the slopes), whether `sums_i + g_i²` is, which `g2` would keep for
/// good: a square inside the doubles whose sum is not (two gradients of
/// `1.2e154`) left the sum infinite, and NaN at the next decay of 0 (review
/// 2026-11-23, REG-9). Every slot's verdict is folded without an early
/// exit, so the check vectorizes as the update does; the verdict is a
/// short-circuiting `any`'s, which reached it a slot at a time at three
/// times the update's cost (task 250).
fn slopes_usable(
    beta: &[f64],
    z: &[f64],
    d: f64,
    weight: f64,
    l2: f64,
    clip: f64,
    sums: Option<&[f64]>,
) -> bool {
    if let Some(sums) = sums {
        beta.iter()
            .zip(z)
            .zip(sums)
            .fold(true, |ok, ((b, &zi), s)| {
                let g = (d * zi * weight + l2 * *b).clamp(-clip, clip);
                ok & (s + g * g).is_finite()
            })
    } else {
        beta.iter().zip(z).fold(true, |ok, (b, &zi)| {
            ok & (d * zi * weight + l2 * *b).clamp(-clip, clip).is_finite()
        })
    }
}

/// [`slopes_usable`]'s verdict at a shared rate `lr`, with each slope's
/// step taken into `next`: `next_i = β_i − lr·g_i`, the update's own
/// expression, so that `next` is what the update leaves in `β` to the bit.
/// `next` is written whatever the verdict; the caller copies it in only
/// when every `g_i` is finite.
#[inline]
fn slopes_stepped(
    beta: &[f64],
    z: &[f64],
    d: f64,
    weight: f64,
    (l2, clip, lr): (f64, f64, f64),
    next: &mut [f64],
) -> bool {
    beta.iter()
        .zip(z)
        .zip(next.iter_mut())
        .fold(true, |ok, ((b, &zi), nb)| {
            let g = (d * zi * weight + l2 * *b).clamp(-clip, clip);
            *nb = *b - lr * g;
            ok & g.is_finite()
        })
}

/// The decay the scaler's history takes into a row of weight `w`: `lam`, or
/// 0 where that history's share of the weight after the row, `lam·W /
/// (lam·W + w)`, is below the smallest normal double -- forgotten there, as
/// a decay that underflows to 0 forgets it from 1075 half-lives on, by the
/// rule `lasso`'s and `ewridge`'s Grams keep ([`crate::gaps::decay_into`]).
/// A zero-weight row 1074 half-lives on left the scaler a weight of a few
/// subnormal ulps beside its old moments; the next row's variance came out
/// that small times the row's squared deviation, its scale near `1e-161`,
/// and the constrained and mapped steps, which divide by it, made every
/// coefficient NaN for good (docs/PLAN.md task 270 (h)). The edge is the
/// smallest normal double because below it the history's terms in the
/// moments, `a·c` and `a·b·d²` with `a` its share, are themselves subnormal,
/// carrying fewer bits than a double has and nothing of the history but a
/// spread the row never had. `standardized`, `standardize_row` and both
/// models' updates read it, `w` the weight the row is admitted at for the
/// read and its own for the update, so what `standardized` reads is to the
/// bit what the update at the admitted weight leaves.
///
/// **The floor** (`floor`, from `Warmup::read`): the history keeps at
/// least `floor` of weight into the row, the decay taken being `max(lam,
/// min(1, floor/W))`, `W` its weight before the row, so `lam·W >= min(W,
/// floor)`. Once the fit is held `floor` is one row's weight of the stream,
/// `ŵ = Σw²/Σw` of the warm-up's rows, and the row is read at no more than
/// `ŵ` (`Warmup::read`), so the history's share of the moments
/// that include it is at least 1/2 wherever `W >= ŵ`, and the scale a row is
/// divided by at least `sd / sqrt(2)`, `sd` the scaler's own before the row.
/// Held, a step taken in the row's coordinates is mapped into the caller's
/// by dividing it by that scale, and after a long gap the history's share
/// `a` is tiny and the scale `sqrt(a)` times the feature's spread: AdaGrad's
/// step, at most the rate in the row's coordinates whatever its sum holds,
/// became `lr / sqrt(a)` feature spreads (a slope moved by 2e8 times the
/// rate after a gap of 60 half-lives, and two such gaps in a row left slopes
/// near 1e89), and every learner's next rows were read against a scaler of
/// one or two rows while the fit stayed in the caller's units. The floor is
/// the least weight that gives the bound: one row's, the most the row itself
/// is read at. It binds only where `lam·W < ŵ`: on a row after a gap of more
/// than `log2(W/ŵ)` half-lives (about 5 at a half-life of 20 rows), and now
/// and then at a half-life of a row or two, where the history holds a row
/// or two; it moves no bit of a stream on which it does not. Before the
/// switch `floor` is `ŵ·n/(n + 1)`, `n` Kish's count, which binds on no
/// regular stream of unit weights whose half-life is a row or more and keeps
/// a gap there from reading the row against nothing (`pa`'s error after a
/// gap of 60 half-lives at row 21 was 9e18 times that of the same rows with
/// no gap). Both are in the stream's own unit, so a stream whose weights are
/// all scaled by a power of two decays the same, to the bit. It composes
/// across a gap split by zero-weight rows, as `max(lam_a·lam_b·W, min(W,
/// floor))`, while `floor` is the same on the zero-weight row and the row
/// after it: a zero-weight row moves neither of the warm-up's sums. `floor`
/// 0 keeps `lam`. No weight a model emits or gates on reads the scaler's.
#[inline]
pub(crate) fn scaler_decay(sc: &EwDiag, lam: f64, w: f64, floor: f64) -> f64 {
    let history = sc.n_eff();
    // A multiply and a compare on the rows it does not bind on, every row of
    // a steady stream; the division only where it does.
    let lam = if lam * history < floor && history > 0.0 {
        lam.max((floor / history).min(1.0))
    } else {
        lam
    };
    crate::gaps::decay_into(lam, history, w)
}

/// `x` standardized against the moments of `sc` with the row itself
/// admitted at weight `admit` after the decay [`scaler_decay`] takes with
/// `floor` ([`EwDiag::including_at`];
/// `SgdCfg::standardize` says why the row is in): feature `i` sits in
/// slot `off + i` of the scaler, and is divided by that sd, or by 1 while
/// it has no spread (the per-slot scale of [`scales_of`]). Lazy: `step`
/// writes it into the model's buffer and `predict` into the thread's, with
/// no vector of scales in between. `pa` standardizes with it too.
#[inline]
pub(crate) fn standardized<'a>(
    sc: &'a EwDiag,
    off: usize,
    x: &'a [f64],
    lam: f64,
    admit: f64,
    floor: f64,
) -> impl Iterator<Item = f64> + 'a {
    let inc = sc.including_at(scaler_decay(sc, lam, admit, floor), admit);
    x.iter().enumerate().map(move |(i, &xi)| {
        let (mean, v, dev) = inc.moments(off + i, xi);
        if off == 0 {
            // No intercept to absorb a shift: scale by the raw second moment
            // and do not centre. Centring gave every prediction a hidden
            // intercept `-Σ b_i m_i / s_i` that `coefficients()` has no slot
            // for, and took a constraint's projection in coordinates shifted in
            // a way the bounds do not know about (review 2026-09-12, C13).
            let raw = v + mean * mean;
            return if raw > 0.0 { xi / raw.sqrt() } else { xi };
        }
        let scale = if crate::variance_is_usable(v, v + mean * mean) {
            v.sqrt()
        } else {
            1.0
        };
        dev / scale
    })
}

/// The row's map once the fit is held in the caller's units (the module
/// docs): [`standardized`]'s `z`, to the bit, and beside it the mean each
/// feature is centred on and the scale it is divided by -- its sd, or 1
/// while it has no spread; through the origin the root of the raw second
/// moment, or 1 where that is 0, and a mean of 0, nothing being centred.
/// Feature `i` sits in slot `off + i` of the scaler and slot `i` of `z`,
/// `mean` and `scale`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn standardize_row(
    sc: &EwDiag,
    off: usize,
    x: &[f64],
    lam: f64,
    admit: f64,
    floor: f64,
    z: &mut [f64],
    mean: &mut [f64],
    scale: &mut [f64],
) {
    let inc = sc.including_at(scaler_decay(sc, lam, admit, floor), admit);
    // Every slice cut to the row's length first, so that the loop indexes
    // each without a bounds check: six a feature, a third of the loop.
    let n = x.len();
    let (hi, lo, c) = sc.parts();
    let (hi, lo, c) = (&hi[off..off + n], &lo[off..off + n], &c[off..off + n]);
    let (z, mean, scale) = (&mut z[..n], &mut mean[..n], &mut scale[..n]);
    // Four features a step in AVX2's 256-bit lanes where the CPU has them
    // (pulp's check at run time, and its safe calls: no `unsafe` here), the
    // loop below everywhere else: on x86-64 without AVX2, on any other
    // architecture, and below `LANES_FROM` features. Baseline x86-64 has
    // only SSE2's two lanes, and the compiler does not vectorize this loop
    // even into those. Each feature is its own lane and runs `moments_of`'s
    // operations in the same order, each rounded once as IEEE 754 rounds
    // it, with no fused multiply-add, so the two paths agree to the bit.
    // A row whose history keeps less than `comp::FORGOTTEN` of the weight
    // takes its means from the row back, which the lanes do not: the loop.
    #[cfg(target_arch = "x86_64")]
    if n >= LANES_FROM
        && inc.weights().0 >= crate::comp::FORGOTTEN
        && let Some(simd) = x86_v3()
    {
        let (a, b) = inc.weights();
        pulp::Simd::vectorize(
            simd,
            StandardizeLanes {
                off,
                a,
                b,
                x,
                hi,
                lo,
                c,
                z,
                mean,
                scale,
            },
        );
        return;
    }
    standardize_scalar(&inc, off, x, hi, lo, c, z, mean, scale);
}

/// [`standardize_row`]'s loop, one feature at a time: the fallback where no
/// vector unit is dispatched to, and the reference the lanes are held to.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn standardize_scalar(
    inc: &crate::ewdiag::Including<'_>,
    off: usize,
    x: &[f64],
    hi: &[f64],
    lo: &[f64],
    c: &[f64],
    z: &mut [f64],
    mean: &mut [f64],
    scale: &mut [f64],
) {
    let n = x.len();
    let (hi, lo, c) = (&hi[..n], &lo[..n], &c[..n]);
    let (z, mean, scale) = (&mut z[..n], &mut mean[..n], &mut scale[..n]);
    for i in 0..n {
        let xi = x[i];
        let (m, v, dev) = inc.moments_of(hi[i], lo[i], c[i], xi);
        if off == 0 {
            let raw = v + m * m;
            z[i] = if raw > 0.0 { xi / raw.sqrt() } else { xi };
            mean[i] = 0.0;
            scale[i] = if raw > 0.0 { raw.sqrt() } else { 1.0 };
            continue;
        }
        let s = if crate::variance_is_usable(v, v + m * m) {
            v.sqrt()
        } else {
            1.0
        };
        z[i] = dev / s;
        mean[i] = m;
        scale[i] = s;
    }
}

/// The width from which [`standardize_row`] takes the vector lanes: below
/// it the dispatch and the features past the last full vector cost more
/// than the lanes save. Measured on x86-64 (Linux, an AVX2 CPU), the time
/// of sgd's whole row with the lanes against the loop alone: +4 to +10% at
/// 2 to 6 features, even at 8 to 12, −7% at 16, −11% at 20 and −15% at 80.
#[cfg(target_arch = "x86_64")]
const LANES_FROM: usize = 12;

/// AVX2 (with the rest of x86-64-v3) where this CPU has it, looked up once:
/// pulp's own check of the CPU's feature flags, safe to call, its answer
/// kept for every later row.
#[cfg(target_arch = "x86_64")]
pub(crate) fn x86_v3() -> Option<pulp::x86::V3> {
    static V3: std::sync::OnceLock<Option<pulp::x86::V3>> = std::sync::OnceLock::new();
    *V3.get_or_init(pulp::x86::V3::try_new)
}

/// [`standardize_row`] across vector lanes: [`crate::ewdiag::Including::moments_of`]
/// and the scale's choice, operation for operation, a lane a feature.
#[cfg(target_arch = "x86_64")]
struct StandardizeLanes<'a> {
    off: usize,
    a: f64,
    b: f64,
    x: &'a [f64],
    hi: &'a [f64],
    lo: &'a [f64],
    c: &'a [f64],
    z: &'a mut [f64],
    mean: &'a mut [f64],
    scale: &'a mut [f64],
}

#[cfg(target_arch = "x86_64")]
impl pulp::WithSimd for StandardizeLanes<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        let Self {
            off,
            a,
            b,
            x,
            hi,
            lo,
            c,
            z,
            mean,
            scale,
        } = self;
        let ab = a * b;
        let (xv, xt) = S::as_simd_f64s(x);
        let (hv, ht) = S::as_simd_f64s(hi);
        let (lv, lt) = S::as_simd_f64s(lo);
        let (cv, ct) = S::as_simd_f64s(c);
        let (zv, zt) = S::as_mut_simd_f64s(z);
        let (mv, mt) = S::as_mut_simd_f64s(mean);
        let (sv, st) = S::as_mut_simd_f64s(scale);
        let (va, vb, vab) = (simd.splat_f64s(a), simd.splat_f64s(b), simd.splat_f64s(ab));
        let (zero, one) = (simd.splat_f64s(0.0), simd.splat_f64s(1.0));
        let inf = simd.splat_f64s(f64::INFINITY);
        // `Including::deviation_of`'s choice, one for the row.
        let small = a < crate::ewdiag::SMALL_SHARE && a > 0.0;
        for i in 0..xv.len() {
            let xi = xv[i];
            // `moments_of`: d, the centred moment, Kahan's step of the mean.
            let d = simd.sub_f64s(simd.sub_f64s(xi, hv[i]), lv[i]);
            let c2 = simd.add_f64s(
                simd.mul_f64s(va, cv[i]),
                simd.mul_f64s(simd.mul_f64s(vab, d), d),
            );
            let y = simd.add_f64s(simd.mul_f64s(vb, d), lv[i]);
            let t = simd.add_f64s(hv[i], y);
            let lo2 = simd.sub_f64s(y, simd.sub_f64s(t, hv[i]));
            let m = t;
            // `clamp_rounding`: 0 where `c2 <= 0`, NaN kept.
            let v = simd.select_f64s(simd.less_than_or_equal_f64s(c2, zero), zero, c2);
            let dev = if small {
                simd.mul_f64s(va, d)
            } else {
                simd.sub_f64s(simd.sub_f64s(xi, m), lo2)
            };
            if off == 0 {
                let raw = simd.add_f64s(v, simd.mul_f64s(m, m));
                let pos = simd.greater_than_f64s(raw, zero);
                let r = simd.sqrt_f64s(raw);
                zv[i] = simd.select_f64s(pos, simd.div_f64s(xi, r), xi);
                mv[i] = zero;
                sv[i] = simd.select_f64s(pos, r, one);
            } else {
                // `variance_is_usable`: `v > 0` and finite.
                let usable = simd.and_m64s(
                    simd.greater_than_f64s(v, zero),
                    simd.less_than_f64s(simd.abs_f64s(v), inf),
                );
                let s = simd.select_f64s(usable, simd.sqrt_f64s(v), one);
                zv[i] = simd.div_f64s(dev, s);
                mv[i] = m;
                sv[i] = s;
            }
        }
        // The features past the last full vector, one at a time.
        for i in 0..xt.len() {
            let xi = xt[i];
            let d = crate::comp::dev(xi, ht[i], lt[i]);
            let c2 = a * ct[i] + a * b * d * d;
            let (mut h2, mut l2) = (ht[i], lt[i]);
            crate::comp::add(&mut h2, &mut l2, b * d);
            let v = crate::solve::clamp_rounding(c2);
            let dev = if small {
                a * d
            } else {
                crate::comp::dev(xi, h2, l2)
            };
            if off == 0 {
                let raw = v + h2 * h2;
                zt[i] = if raw > 0.0 { xi / raw.sqrt() } else { xi };
                mt[i] = 0.0;
                st[i] = if raw > 0.0 { raw.sqrt() } else { 1.0 };
            } else {
                let s = if crate::variance_is_usable(v, v + h2 * h2) {
                    v.sqrt()
                } else {
                    1.0
                };
                zt[i] = dev / s;
                mt[i] = h2;
                st[i] = s;
            }
        }
    }
}

/// Partial sums the dot product is accumulated in. A single running sum is
/// a chain of dependent additions, three or four cycles each whatever the
/// core could do in parallel; eight independent ones run at the adder's
/// throughput instead, and were most of a wide step (docs/records/PERFORMANCE-LOG.md
/// §20). The order is fixed by this constant and the code below, not by
/// the hardware, so every platform sums the same bits.
const DOT_LANES: usize = 8;

/// `beta · [1, z]`: `z` against the feature coefficients in [`DOT_LANES`]
/// interleaved partial sums (slot `i` into sum `i % DOT_LANES`), the sums
/// folded pairwise, and the intercept's coefficient -- its `z` is the
/// constant 1 -- added last. Always inlined: a step calls it twice a
/// target (the prediction, and the row's squared length for the step
/// gains), and out of line the call and its length checks were 2.5% of
/// a 20-feature row's instructions (task 284).
#[inline(always)]
fn dot(beta: &[f64], off: usize, z: &[f64]) -> f64 {
    let b = &beta[off..];
    debug_assert_eq!(b.len(), z.len());
    let mut acc = [0.0f64; DOT_LANES];
    let (zc, zr) = z.as_chunks::<DOT_LANES>();
    let (bc, br) = b.as_chunks::<DOT_LANES>();
    for (zs, bs) in zc.iter().zip(bc) {
        for l in 0..DOT_LANES {
            acc[l] += zs[l] * bs[l];
        }
    }
    for (l, (zi, bi)) in zr.iter().zip(br).enumerate() {
        acc[l] += zi * bi;
    }
    let s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    if off == 1 { beta[0] + s } else { s }
}

impl SgdCfg {
    /// The count a refit buffer fills at, or [`crate::huber_refit::OPEN`]
    /// under a refit over a span of the clock.
    fn refit_cap(&self) -> u32 {
        if self.refit_clock.is_some() {
            crate::huber_refit::OPEN
        } else {
            self.refit_rows
        }
    }
}

impl Sgd {
    /// The bytes a refit over a span of the clock may hold a group.
    fn span_cap(&self) -> u64 {
        self.cap_for_test
            .or(self.cfg.max_state_bytes)
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    /// Hold a span's buffers to `bytes` in place of their cap, for a test.
    #[cfg(test)]
    pub(crate) fn set_cap_for_test(&mut self, bytes: u64) {
        self.cap_for_test = Some(bytes);
    }
}

impl OnlineModel for Sgd {
    /// A refit over a span of the clock reads the row's stamp (task 538).
    fn stamp_next(&mut self, stamp: crate::Stamp) {
        if let Some(p) = self.place.as_mut() {
            p.hand(stamp);
        }
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        let cap = self.span_cap();
        self.warning.take().map(|b| (b, cap))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|b| (b, self.span_cap()))
    }

    /// Each target's own weight, the rows that carried it, for the bank's
    /// per-target `min_weight` gate: without it the bank checked a list of
    /// thresholds against the shared `n_eff` (task 158, E14).
    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.w_target);
        true
    }

    /// The step, into the caller's buffer (task 270, M-4).
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A value that is not usable, by the rule every model keeps
        // (`OnlineModel`): a feature that is not a number made every
        // coefficient NaN (docs/PLAN.md task 183).
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        let m = self.cfg.n_targets;
        let off = usize::from(self.cfg.fit_intercept);
        let lam = self.cfg.decay.factor(d_clock);
        // The Huber scale's own clock (`huber_scale_decay`), the model's
        // unless given: the same factor, to the bit, where it is not.
        let lam_scale = self
            .cfg
            .huber_scale_decay
            .map_or(lam, |d| d.factor(d_clock));
        // Under a span refit, check the whole row's buffer growth atomically
        // before moving any clock, weight, coefficient, scaler, or buffer.
        if self.over.is_some() {
            if let Some(p) = &mut self.place {
                p.discard_next();
            }
            self.predict_step_into(x, d_clock, weight, out);
            return;
        }
        let per_refit = 8 * (x.len() + 4) as u64 + 48 * u64::from(!self.mad.is_empty());
        if self.cfg.refit_clock.is_some() && self.refit.iter().any(RefitBuf::open) {
            let held: u64 = self.refit.iter().map(RefitBuf::held_bytes).sum();
            let limit = self.span_cap();
            let upper: u64 = (0..m)
                .filter(|&j| self.refit[j].open())
                .map(|j| {
                    let cand = weight > 0.0
                        && y[j].is_some_and(|v| v.is_finite() && self.label(v).is_some());
                    if cand {
                        per_refit
                    } else {
                        self.refit[j].age_bytes(lam, lam_scale)
                    }
                })
                .sum();
            if held + upper > limit {
                let delta = self.span_row_delta(x, y, weight, lam, lam_scale, per_refit);
                if delta > 0 {
                    let mut guard = crate::clock_span::HeldCap {
                        over: &mut self.over,
                        warned: &mut self.warned,
                        warning: &mut self.warning,
                    };
                    if !guard.admit(held + delta, limit) {
                        if let Some(p) = &mut self.place {
                            p.discard_next();
                        }
                        self.predict_step_into(x, d_clock, weight, out);
                        return;
                    }
                }
            }
        }
        // The row's place, for a refit over a span of the clock (task 538).
        let now = self.place.as_mut().map(|p| p.step(d_clock));

        // The row the model sees. Without a scaler it is `x` itself, the
        // intercept's constant 1 folded into `dot` and the update below:
        // copying `[1, x]` in front of every row was a tenth of a wide step
        // (docs/records/PERFORMANCE-LOG.md §20). With one, `[1, z]` is standardized
        // against the moments with this row admitted -- read off the scaler
        // as it stands, the way `predict` reads them (E24, and
        // `SgdCfg::standardize` for the order); the raw `[1, x]` is kept
        // to update the scaler afterwards, at the row's actual weight.
        //
        // Once the scaler has warmed up ([`crate::Warmup`]) the fit is held
        // in the caller's units: the prediction reads `x`, and the row's map
        // -- each feature's mean and scale beside `z` -- maps the step into
        // the caller's units (the module docs, docs/PLAN.md task 206).
        let held = self.warmup.switched();
        // The weight the row is read at and the weight the scaler's history
        // keeps into it, both in the stream's own units and both from the
        // warm-up as the row arrives (`Warmup::read`).
        let (admit, floor) = self.warmup.read(weight);
        if self.scaler.is_some() {
            self.ensure_buffers();
            let Self {
                scaler,
                zbuf,
                mbuf,
                sbuf,
                rawbuf,
                ..
            } = self;
            let sc = scaler.as_ref().expect("checked above");
            if off == 1 {
                zbuf[0] = 1.0;
                rawbuf[0] = 1.0;
            }
            rawbuf[off..].copy_from_slice(x);
            if held {
                standardize_row(
                    sc,
                    off,
                    x,
                    lam,
                    admit,
                    floor,
                    &mut zbuf[off..],
                    &mut mbuf[off..],
                    &mut sbuf[off..],
                );
            } else {
                for (z, v) in zbuf[off..]
                    .iter_mut()
                    .zip(standardized(sc, off, x, lam, admit, floor))
                {
                    *z = v;
                }
            }
        }
        // The coordinates the prediction reads: `z` while the scaler warms
        // up, `x` itself without one or after it.
        let reads_z = self.scaler.is_some() && !held;

        // Decay first, so a long gap re-opens an annealed or adapted rate.
        if lam != 1.0 {
            for g in self.g2.iter_mut() {
                for v in g.iter_mut() {
                    *v *= lam;
                }
            }
        }
        // `n_eff` is the weight *before* this row's update and *before* its
        // decay, which is the convention every other model reports and gates
        // on (see `EwRidgeCfg::min_weight`). Decaying it here would make
        // `min_weight` mean a slightly different number of rows for `sgd`
        // than for `ewridge`, which is exactly the kind of quiet divergence
        // the cross-model semantics suite exists to catch. Each target's
        // `min_weight` reads its own weight, the rows that carried it.
        let n_eff = self.w_sum;

        out.pred.clear();
        out.pred.resize(m, f64::NAN);
        let pred = &mut out.pred;
        if self.cfg.constraint.is_some() {
            // Skipped by serde, so sized here rather than in `new`.
            self.learned.clear();
            self.learned.resize(m, false);
        }
        let track_avg = self.extras.as_ref().is_some_and(|e| e.avg.is_some());
        let mut avg_taught = if track_avg {
            vec![false; m]
        } else {
            Vec::new()
        };
        let lr_row_shared = match self.cfg.schedule {
            LearningRate::Constant => Some(self.cfg.learning_rate),
            LearningRate::InvScaling { power } => {
                Some(self.cfg.learning_rate / (1.0 + lam * self.w_sum).powf(power))
            }
            LearningRate::AdaGrad | LearningRate::Optimal => None,
        };
        let (lr0, l1, l2, clip0) = (
            self.cfg.learning_rate,
            self.cfg.l1,
            self.cfg.l2,
            self.cfg.clip_gradient,
        );
        let is_optimal = matches!(self.cfg.schedule, LearningRate::Optimal);
        let is_variant = l1 > 0.0 || self.cfg.invariant || is_optimal;
        // The squared length of the row the step is taken on, `|z̃|²` with
        // the intercept's 1, for the step gains (`StepGain`) and `invariant`;
        // unread where none are kept.
        let mut zz: Option<f64> = None;
        // Per target, the label a row taught it with, while a refit buffer
        // is open (`SgdCfg::refit_rows`); empty otherwise, which costs a
        // row nothing once every refit has run. The model's own scratch,
        // taken for the row and put back after the refits, so that its
        // capacity is kept and a row allocates nothing.
        let mut taught = std::mem::take(&mut self.taught);
        taught.clear();
        if self.refit.iter().any(RefitBuf::open) {
            taught.resize(m, None);
        }
        for j in 0..m {
            let eta = if reads_z {
                dot(&self.beta[j], off, &self.zbuf[off..])
            } else {
                dot(&self.beta[j], off, x)
            };
            let p = self.link(eta);
            // A prediction the target's weight allows, withheld while its
            // recent steps say the rate is past its stability bound, read
            // from the state before the row as `predict` reads it. That
            // gate is the output's alone: a withheld row still teaches its
            // residual to the scales, as a predicted one does.
            let predicted = self.w_target[j] >= self.cfg.min_weight;
            if predicted && !self.gain.get(j).is_some_and(StepGain::unstable) {
                pred[j] = if let Some(avg) = self.extras.as_ref().and_then(|e| e.avg.as_ref()) {
                    let eta_avg = if reads_z {
                        dot(&avg.beta[j], off, &self.zbuf[off..])
                    } else {
                        dot(&avg.beta[j], off, x)
                    };
                    self.link(eta_avg)
                } else {
                    p
                };
            }
            let lr_row = if is_optimal {
                let t = self
                    .extras
                    .as_ref()
                    .and_then(|e| e.t_step.get(j).copied())
                    .unwrap_or(0);
                Some(variants::optimal_rate(l1 + l2, t))
            } else {
                lr_row_shared
            };
            // The cut and the tube are drawn in their scale as the row
            // arrives (the module docs). Under Huber, `s²`'s weight ages on
            // every row, as `huber`'s does, whatever the row then teaches;
            // under epsilon-insensitive the target's spread does, and
            // learns the row's `y` at once, whatever the fit. So does the
            // unit the clip and AdaGrad's epsilon are in, under
            // `target_spread`, read before it learns the row.
            let scale = self.scale(j);
            let unit = self.unit(j);
            let clip = if unit > 0.0 {
                clip0 * unit
            } else {
                f64::INFINITY
            };
            let ada_eps = 1e-8 * unit;
            if let Some(ws) = self.wsig.get_mut(j) {
                *ws *= lam_scale;
            }
            if let Some(q) = self.mad.get_mut(j) {
                q.age(lam_scale);
            }
            if !self.spread.is_empty() {
                self.spread.update(j, y[j], lam, weight);
            }
            let Some(yj) = y[j] else { continue };
            if weight <= 0.0 || !yj.is_finite() {
                continue;
            }
            let Some(yj) = self.label(yj) else { continue };
            let d = self.dloss(p, yj, scale);
            // The row's own residual joins `s²` afterwards, from the
            // prediction the fit made, as `huber`'s does, whether or not
            // `min_weight` withholds it from the output, and whether or not
            // the unstable rate's gate does. A state written before schema
            // 64 learns only from a row `min_weight` lets through, so a row
            // that gate withheld adds nothing there
            // (`SgdCfg::min_weight_holds_learning`; docs/PLAN.md task 278,
            // D10).
            let scored = predicted || !self.cfg.min_weight_holds_learning;
            // The `|e|` the one-pass scale takes from the row, NaN where it
            // takes none: the refit's buffer keeps it beside the row.
            let mut e_abs = f64::NAN;
            if !self.sig2.is_empty() && scored && p.is_finite() {
                let resid = yj - p;
                let ws_new = self.wsig[j] + weight;
                let s2 = (self.wsig[j] * self.sig2[j] + weight * resid * resid) / ws_new;
                // Skipped where it would not be finite, as `huber`'s is: an
                // `inf` scale would make every cut and tube infinite for good.
                if s2.is_finite() {
                    self.sig2[j] = s2;
                    self.wsig[j] = ws_new;
                }
                // The MAD's sketch takes the same residual at the same
                // weight (`huber`'s rule; an `|e|` that is not finite adds
                // nothing), but not while the refit's buffer holds the
                // `|e|` and the scale is read from them
                // (`RefitBuf::reads_resid`).
                if let Some(q) = self.mad.get_mut(j)
                    && !self.refit.get(j).is_some_and(RefitBuf::reads_resid)
                {
                    q.add(resid.abs(), weight);
                }
                e_abs = resid.abs();
            }
            // The target's level, while it warms up (`keeps_level`): the
            // intercept takes the running weighted mean of the residuals,
            // `b_0 += (w / Σw)·(y − p)` in whichever coordinates it is held
            // (its `z̃` is 1 in each), in place of its gradient step. On the
            // target's first learned row that is the whole residual and
            // nothing else moves, so the level never passes through the
            // slopes' steps (the module docs).
            // Both the level's weight and the step's gain are learned only
            // where the row's step is taken: a row refused below for a
            // gradient that is not a finite number teaches nothing, and
            // counted in either it would move the level's warm-up and its
            // gain, or enter the unstable rate's sums, for a step that never
            // happened.
            let mut level_step = None;
            let mut level_next = None;
            if let Some(&lv) = self.level.get(j) {
                let first = lv.w == 0.0;
                let mut lv = lv;
                lv.learn(weight);
                let gain = weight / lv.w;
                // The squared and Huber losses' gradient is the residual
                // (clipped at the cut): the running mean takes it, while its
                // gain is above the rate's. The sign-valued losses' is not in
                // the target's units: the running mean of the residual over
                // the warm-up alone.
                let (step, warm) =
                    if matches!(self.cfg.loss, SgdLoss::Squared | SgdLoss::Huber { .. }) {
                        let above = lr_row.is_some_and(|lr| gain > lr);
                        (-gain * d, above || lv.count() <= crate::WARMUP_ROWS)
                    } else {
                        (gain * (yj - p), lv.count() <= crate::WARMUP_ROWS)
                    };
                if first {
                    let b0 = self.beta[j][0] + step;
                    if b0.is_finite() {
                        self.beta[j][0] = b0;
                        self.level[j] = lv;
                        if is_optimal && let Some(ex) = self.extras.as_mut() {
                            ex.t_step[j] = ex.t_step[j].saturating_add(1);
                        }
                        if track_avg {
                            avg_taught[j] = true;
                        }
                        if let Some(t) = taught.get_mut(j) {
                            *t = Some((yj, e_abs));
                        }
                    }
                    continue;
                }
                level_next = Some(lv);
                if warm && step.is_finite() {
                    level_step = Some(step);
                }
            }
            // The step's gain along its own row, `q = lr·(w·c·|z̃|² + l2)`
            // with `c` the loss's curvature at the prediction: the rate
            // past its stability bound shows in their mean square
            // (`StepGain`). `l2` shrinks each slope by `1 − lr·l2` besides.
            // A resting gate takes no gain, and wakes on a residual past
            // `WAKE_SPREADS` of the target's spread, its unit here
            // ([`StepGain`]). The rest count and the wake are made on the
            // gate in place; `found`, a resting gate's count as the row found
            // it, lets `refused_step` undo them where the row's step is
            // refused: a refused row leaves the gate as it found it.
            let (resting, probe, found) = match self.gain.get_mut(j) {
                Some(g) if g.resting => {
                    let found = g.rested;
                    g.rested = g.rested.wrapping_add(1);
                    if (yj - p).abs() > WAKE_SPREADS * unit {
                        g.resting = false;
                        g.rested = 0;
                    }
                    (
                        g.resting,
                        g.resting && g.rested % PROBE_ROWS == 0,
                        Some(found),
                    )
                }
                _ => (false, false, None),
            };
            let step_gain = match (self.gain.is_empty() || (resting && !probe), lr_row) {
                (false, Some(lr)) => {
                    let c = match self.cfg.loss {
                        SgdLoss::Squared => 1.0,
                        SgdLoss::Huber { .. } => f64::from(d == p - yj),
                        SgdLoss::Poisson => p,
                        SgdLoss::Logistic => p * (1.0 - p),
                        SgdLoss::Quantile { .. } | SgdLoss::EpsilonInsensitive { .. } => 0.0,
                    };
                    let along = if c > 0.0 {
                        // In `dot`'s lanes: a plain running sum is a chain
                        // of dependent additions, a tenth of a wide step.
                        let zz = *zz.get_or_insert_with(|| {
                            let z: &[f64] = if self.scaler.is_some() {
                                &self.zbuf[off..]
                            } else {
                                x
                            };
                            off as f64 + dot(z, 0, z)
                        });
                        weight * c * zz
                    } else {
                        0.0
                    };
                    let q = lr * (along + l2);
                    if probe {
                        // A probe: the gain wakes the gate past twice the
                        // resting mean, and joins the sums; otherwise it is
                        // let pass.
                        let g = &mut self.gain[j];
                        if q > PROBE_WAKE * g.q / g.n {
                            g.resting = false;
                            g.rested = 0;
                            Some(q)
                        } else {
                            None
                        }
                    } else {
                        Some(q)
                    }
                }
                _ => None,
            };
            if is_variant {
                let zz_row = if self.cfg.invariant {
                    *zz.get_or_insert_with(|| {
                        let z: &[f64] = if self.scaler.is_some() {
                            &self.zbuf[off..]
                        } else {
                            x
                        };
                        off as f64 + dot(z, 0, z)
                    })
                } else {
                    0.0
                };
                let k = self.beta[j].len();
                let n = k - off;
                if self.dbuf.len() < k {
                    self.dbuf.resize(k, 0.0);
                }
                if self.gbuf.len() < k {
                    self.gbuf.resize(k, 0.0);
                }
                let u_len = self
                    .extras
                    .as_ref()
                    .and_then(|e| e.enet.as_ref())
                    .map_or(0, |e| e.u[j].len());
                let mut next_u = vec![0.0; u_len];
                let mut next_q = vec![0.0; if l1 > 0.0 { n } else { 0 }];
                let ok = if held {
                    let u_cur = self
                        .extras
                        .as_ref()
                        .and_then(|e| e.enet.as_ref())
                        .map(|e| e.u[j].as_slice());
                    let q_cur = self
                        .extras
                        .as_ref()
                        .and_then(|e| e.enet.as_ref())
                        .map(|e| e.q[j].as_slice());
                    variants::step_mapped_variant(
                        &self.beta[j],
                        off,
                        &self.zbuf[..k],
                        &self.sbuf[..k],
                        &self.mbuf[..k],
                        d,
                        weight,
                        (l1, l2, clip, ada_eps),
                        lr_row,
                        lr0,
                        self.cfg.invariant,
                        zz_row,
                        level_step,
                        self.g2.get(j).map(Vec::as_slice),
                        u_cur,
                        q_cur,
                        &mut self.dbuf[..k],
                        &mut self.gbuf[..k],
                        &mut next_u,
                        &mut next_q,
                    )
                } else {
                    let z: &[f64] = if self.scaler.is_some() {
                        &self.zbuf[off..]
                    } else {
                        x
                    };
                    let u_cur = self
                        .extras
                        .as_ref()
                        .and_then(|e| e.enet.as_ref())
                        .map(|e| e.u[j].as_slice());
                    let q_cur = self
                        .extras
                        .as_ref()
                        .and_then(|e| e.enet.as_ref())
                        .map(|e| e.q[j].as_slice());
                    if let Some(b0_step) = variants::step_direct_variant(
                        &self.beta[j],
                        off,
                        z,
                        d,
                        weight,
                        (l1, l2, clip, ada_eps),
                        lr_row,
                        lr0,
                        self.cfg.invariant,
                        zz_row,
                        level_step,
                        self.g2.get(j).map(Vec::as_slice),
                        u_cur,
                        q_cur,
                        &mut self.dbuf[off..k],
                        &mut self.gbuf[..k],
                        &mut next_u,
                        &mut next_q,
                    ) {
                        if off == 1 {
                            self.dbuf[0] = self.beta[j][0] + b0_step;
                        }
                        true
                    } else {
                        false
                    }
                };
                if ok {
                    self.beta[j].copy_from_slice(&self.dbuf[..k]);
                    if let Some(sums) = self.g2.get_mut(j) {
                        sums.copy_from_slice(&self.gbuf[..k]);
                    }
                    if let Some(ex) = self.extras.as_mut() {
                        if let Some(enet) = ex.enet.as_mut() {
                            enet.u[j].copy_from_slice(&next_u);
                            enet.q[j].copy_from_slice(&next_q);
                        }
                        if is_optimal {
                            ex.t_step[j] = ex.t_step[j].saturating_add(1);
                        }
                    }
                    self.took_step(j, level_next, step_gain);
                    if let Some(t) = taught.get_mut(j) {
                        *t = Some((yj, e_abs));
                    }
                    if held {
                        if self.cfg.constraint.is_some() {
                            self.project_mapped(j);
                        }
                    } else if let Some(l) = self.learned.get_mut(j) {
                        *l = true;
                    }
                    if track_avg {
                        avg_taught[j] = true;
                    }
                } else {
                    self.refused_step(j, found);
                }
                continue;
            }
            if held {
                // The step in the row's coordinates, mapped into the
                // caller's units by the row's map, and projected there.
                if self.step_mapped(j, d, weight, lr_row, (clip, ada_eps), level_step) {
                    self.took_step(j, level_next, step_gain);
                    if let Some(t) = taught.get_mut(j) {
                        *t = Some((yj, e_abs));
                    }
                    if self.cfg.constraint.is_some() {
                        self.project_mapped(j);
                    }
                    if track_avg {
                        avg_taught[j] = true;
                    }
                } else {
                    self.refused_step(j, found);
                }
                continue;
            }
            let z: &[f64] = if self.scaler.is_some() {
                &self.zbuf[off..]
            } else {
                x
            };
            // Per slot: `g = d * z_i * w`, plus `l2 * beta_i` off the
            // intercept, clipped; `beta_i -= lr * g`. The intercept's `z`
            // is 1 (`d * 1 * w` is `d * w` exactly), and it is not
            // penalised.
            let beta = &mut self.beta[j];
            // A row any of whose gradients is not a finite number teaches
            // nothing, and is skipped before `beta` moves, as a row with no
            // weight is; under AdaGrad so is one whose squared gradient is
            // not, which `g2` would keep for good (decay never takes an
            // `inf` away; `ftrl`'s guard). With `clip_gradient = inf`, which
            // disables the clip, two rows inside the input bound made a
            // coefficient infinite and every later prediction null, where
            // `pa`, `ftrl` and `kalman` each skip such a row (review round
            // 4, CC2). The gradients are formed here as the update forms
            // them, so a row that passes moves `beta` to the same bits.
            let usable = |g: f64| {
                if lr_row.is_some() {
                    g.is_finite()
                } else {
                    // The intercept's sum with its square in it (REG-9).
                    (self.g2[j][0] + g * g).is_finite()
                }
            };
            // At a shared rate the check and the step are one pass: each
            // slope's gradient is formed once, its new value written to
            // `dbuf` beside the verdict, and copied in once the row passes.
            // Formed twice, once for the verdict and again for the step,
            // the gradient cost a 20-feature step about 3 of its 39 ns on
            // arm64, natively, and a row through the bank about 1% (task
            // 314). The values are the update's own, so `beta` moves to the
            // same bits.
            let n = beta.len() - off;
            if self.dbuf.len() < n {
                self.dbuf.resize(n, 0.0);
            }
            let slopes_ok = match lr_row {
                Some(lr) => slopes_stepped(
                    &beta[off..],
                    z,
                    d,
                    weight,
                    (l2, clip, lr),
                    &mut self.dbuf[..n],
                ),
                None => slopes_usable(
                    &beta[off..],
                    z,
                    d,
                    weight,
                    l2,
                    clip,
                    Some(&self.g2[j][off..]),
                ),
            };
            if (off == 1 && !usable((d * weight).clamp(-clip, clip))) || !slopes_ok {
                self.refused_step(j, found);
                continue;
            }
            // The intercept's step: its gradient's at the row's rate or
            // AdaGrad's, whose sum learns the gradient either way, or in its
            // place the level's while that is the larger.
            if off == 1 {
                let g = (d * weight).clamp(-clip, clip);
                let step = match lr_row {
                    Some(lr) => -(lr * g),
                    None => {
                        let g2 = &mut self.g2[j];
                        g2[0] += g * g;
                        -adagrad_step(lr0, g2[0], ada_eps, g)
                    }
                };
                beta[0] += level_step.unwrap_or(step);
            }
            match lr_row {
                Some(_) => beta[off..].copy_from_slice(&self.dbuf[..n]),
                None => {
                    let g2 = &mut self.g2[j];
                    for ((b, g2), &zi) in beta[off..].iter_mut().zip(&mut g2[off..]).zip(z) {
                        let g = (d * zi * weight + l2 * *b).clamp(-clip, clip);
                        *g2 += g * g;
                        *b -= adagrad_step(lr0, *g2, ada_eps, g);
                    }
                }
            }
            if let Some(l) = self.learned.get_mut(j) {
                *l = true;
            }
            if track_avg {
                avg_taught[j] = true;
            }
            self.took_step(j, level_next, step_gain);
            if let Some(t) = taught.get_mut(j) {
                *t = Some((yj, e_abs));
            }
        }
        if let Some(sc) = &mut self.scaler {
            sc.update(&self.rawbuf, scaler_decay(sc, lam, weight, floor), weight);
        }
        // Project after the scaler moved: the bounds live in the caller's
        // units, and in standardized coordinates they move with the scales,
        // so every target is re-projected when the scales changed (a
        // positive weight), otherwise only the ones this row stepped. Once
        // the fit is held in the caller's units each target was projected
        // after its own step, and the scales move none of it.
        if self.cfg.constraint.is_some() && !held {
            let scales = self.scaler.is_some().then(|| self.scales());
            let rescaled = self.scaler.is_some() && weight > 0.0;
            let Self {
                cfg,
                beta,
                extras,
                pbuf,
                learned,
                ..
            } = self;
            let c = cfg.constraint.as_ref().expect("checked above");
            for (j, b) in beta.iter_mut().enumerate() {
                if rescaled || learned[j] {
                    c.project(&mut b[off..], scales.as_deref().map(|s| &s[off..]), pbuf);
                }
            }
            if rescaled && let Some(avg) = extras.as_mut().and_then(|e| e.avg.as_mut()) {
                for b in avg.beta.iter_mut() {
                    c.project(&mut b[off..], scales.as_deref().map(|s| &s[off..]), pbuf);
                }
            }
        }
        // Fold each target's post-projection live iterate into the running
        // decay-weighted average; targets that did not learn only age.
        if let Some(avg) = self.extras.as_mut().and_then(|e| e.avg.as_mut()) {
            for (j, &taught) in avg_taught.iter().enumerate().take(m) {
                if taught {
                    avg.learn(j, lam, weight, &self.beta[j]);
                } else {
                    avg.age(j, lam);
                }
            }
        }
        // The warm-up: the scaler has learned the row. On the row that
        // completes it the coefficients are read out once, with the moments
        // as they stand -- the numbers `coefficients()` gave until now --
        // and held in the caller's units from then on.
        if !held
            && let Some(sc) = &self.scaler
            && self.warmup.learn(weight)
        {
            self.beta = unscaled(&self.beta, sc, self.cfg.fit_intercept);
            if let Some(avg) = self.extras.as_mut().and_then(|e| e.avg.as_mut()) {
                avg.beta = unscaled(&avg.beta, sc, self.cfg.fit_intercept);
            }
            // The map a refit on this row projects with
            // ([`Self::project_mapped`]): a row that starts held forms it
            // as it standardizes, and this one did not, so it is the
            // scaler's as it stands. It was the `[0]`, `[1]` the buffers
            // start at, and the intercept took none of the slopes' move
            // (review 2026-11-23, REG-6).
            let scales = scales_of(sc, off);
            for (i, &scale) in scales.iter().enumerate().skip(off) {
                self.mbuf[i] = sc.mean(i);
                self.sbuf[i] = scale;
            }
        }
        // The refit's buffers take the row, a learned one or an ageing, and
        // a target whose buffer it fills is refit, after the row's own
        // prediction, step and scale (hard rule 2).
        if !taught.is_empty() {
            let cap = self.cfg.refit_cap();
            let span = self.cfg.refit_clock.zip(now);
            let least = self.cfg.k_total().max(1);
            if span.is_some() {
                let held: u64 = self.refit.iter().map(RefitBuf::held_bytes).sum();
                let delta: u64 = taught
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| self.refit[j].open())
                    .map(|(j, t)| match t {
                        Some(_) => per_refit,
                        None => self.refit[j].age_bytes(lam, lam_scale),
                    })
                    .sum();
                if delta > 0 {
                    let limit = self.span_cap();
                    let mut guard = crate::clock_span::HeldCap {
                        over: &mut self.over,
                        warned: &mut self.warned,
                        warning: &mut self.warning,
                    };
                    guard.note(held + delta, limit);
                }
            }
            for (j, t) in taught.iter().enumerate() {
                let b = &mut self.refit[j];
                let full = match t {
                    Some((yj, e)) => {
                        let full = b.learn_scored(x, *yj, *e, weight, lam, lam_scale, cap);
                        match span {
                            Some((d, now)) => b.span_due(now, d, least),
                            None => full,
                        }
                    }
                    None => {
                        if let Some(held) = b.age(lam, lam_scale, cap) {
                            self.scale_from_held(j, &held);
                        }
                        false
                    }
                };
                if full {
                    self.refit_target(j);
                }
            }
        }
        self.taught = taught;
        self.w_sum = lam * self.w_sum + weight;
        // A label `strict_binary` refuses, or a negative count, carries no
        // weight for its target, as in `ftrl`: it taught nothing.
        let (loss, strict) = (self.cfg.loss, self.cfg.strict_binary);
        crate::model::age_target_weights(
            &mut self.w_target,
            |j| y[j].is_some_and(|v| v.is_finite() && teaches(loss, strict, v)),
            lam,
            weight,
        );
        out.n_eff = n_eff;
        out.extra = None;
    }

    /// The prediction at another `min_weight` floor, set for the read and
    /// put back ([`OnlineModel::predict_with_floor`]).
    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let floor = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = OnlineModel::predict_with(self, x, y, d_clock);
        self.cfg.min_weight = floor;
        out
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.predict_step_into(x, d_clock, f64::INFINITY, &mut out);
        out
    }

    fn state(&self) -> State {
        State::new(ModelState::Sgd(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Sgd(m) => {
                let mut m = (**m).clone();
                // The shapes are checked as the state is read (`SgdV3`).
                crate::model::check_cfg("sgd", m.cfg.validate())?;
                m.check_loaded()?;
                m.ensure_buffers();
                // A build of schema 64 or earlier anchored each sketch at
                // its first residual: its sketches keep that rule
                // (`EwQuantile::keep_first_value_anchor`).
                if s.schema_version <= crate::stats::LAST_FIRST_VALUE_ANCHOR_SCHEMA {
                    m.mad
                        .iter_mut()
                        .for_each(crate::EwQuantile::keep_first_value_anchor);
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "sgd",
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
#[path = "sgd/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "sgd/refit_tests.rs"]
mod refit_tests;

#[cfg(test)]
#[path = "sgd/span_tests.rs"]
mod span_tests;

#[cfg(test)]
#[path = "sgd/gate_tests.rs"]
mod gate_tests;

#[cfg(test)]
#[path = "sgd/target_spread_tests.rs"]
mod target_spread_tests;

#[cfg(test)]
#[path = "sgd/gap_tests.rs"]
mod gap_tests;

#[cfg(test)]
#[path = "sgd/weight_unit_tests.rs"]
mod weight_unit_tests;

#[cfg(test)]
#[path = "sgd/mutant_tests.rs"]
mod mutant_tests;
