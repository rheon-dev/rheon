//! The correlation stage's start: the batch quasi-likelihood of the
//! buffered rows, maximized by projected Fisher scoring, held by the model
//! itself (no library's optimizer).
//!
//! ```text
//! f(θ) = Σ_t ŵ_t l_t(θ)                 Q_1 = Q̄, the buffer's weighted means as targets
//! G(θ) = Σ_t ŵ_t I_t(θ)                 the expected information
//! d    = (G + 10⁻¹⁰ diag G)⁻¹ ∇f
//! θ'   = Proj_G(θ + s d),  s = 1, ½, ¼, … (at most 30 halvings),
//!        the first with f(θ') ≥ f(θ) + 10⁻⁴ ∇fᵀ(θ' − θ)
//! ```
//!
//! from the best point of a small grid (each projected onto the set). It
//! runs to the rounding: it stops after the iteration that moves no
//! parameter by more than `10⁻¹⁰` or `f` by more than `10⁻¹² (1 + |f|)`,
//! when no step ascends, or at 100 iterations. How it stopped decides
//! nothing. Two checks of the point it reached do, each on the likelihood
//! with the rows' weights at Kish's count, `Σw/Σw²` times it (so scaling
//! every weight moves neither, and at unit weights it is the plain one):
//!
//! - **sound**: the gain the projected Newton step predicts there, `gᵀΔ −
//!   ½ ΔᵀGΔ` with `Δ = Proj_G(θ + G⁻¹g) − θ`, at most `10⁻⁶` nats -- the
//!   Newton decrement `½ gᵀG⁻¹g` inside the set, and on a face of it the
//!   step the face allows, so an optimum on a bound is one;
//! - **identified**: the likelihood-ratio statistic against constant
//!   correlation, `2 (f(θ̂) − f(0))`, at least χ² at 0.999 for the
//!   restrictions constant correlation puts on the recursion -- `a = b = 0`
//!   for DCC, 2, cut 13.8; `a = b = g = 0` for ADCC, 3, cut 16.3 -- and `G`
//!   positive definite. At the null `b` is not identified (`a` is 0), so the
//!   statistic's law is not quite χ² of the restrictions; the cut is the
//!   one the GARCH margins' start uses for the same ridge, and it is
//!   conservative there.

use super::mat::solve_small;
use super::region::{PERSISTENCE_CAP, project};
use super::rows::{Work, advance, quad, read, score_info};

const MAX_ITER: u32 = 100;
const MAX_HALVINGS: u32 = 30;
const ARMIJO: f64 = 1e-4;

/// The buffered rows, as the fit reads them.
pub(super) struct Rows<'a> {
    /// Row `t`'s standardized residuals at `z[t k..(t + 1) k]`.
    pub z: &'a [f64],
    /// Row `t`'s weight in the likelihood, decayed to the last row.
    pub w: &'a [f64],
    pub k: usize,
    pub qbar: &'a [f64],
    pub nbar: &'a [f64],
}

/// The value, gradient and information of the buffer's likelihood at
/// `theta`; the value alone without `grad`. `None` where a row's `R` does
/// not factor.
pub(super) fn objective(
    rows: &Rows<'_>,
    theta: &[f64],
    grad: bool,
    work: &mut Work,
) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let (k, p) = (rows.k, theta.len());
    let mut q = rows.qbar.to_vec();
    let mut dq = if grad {
        vec![0.0; p * k * k]
    } else {
        Vec::new()
    };
    let (mut f, mut g, mut h) = (0.0, vec![0.0; p], vec![0.0; p * p]);
    for (t, &wt) in rows.w.iter().enumerate() {
        let z = &rows.z[t * k..(t + 1) * k];
        let ld = read(&q, k, work)?;
        let l = -0.5 * (ld + quad(z, k, work));
        if !l.is_finite() {
            return None;
        }
        f += wt * l;
        if grad {
            score_info(&q, &dq, k, p, work);
            for (gi, wg) in g.iter_mut().zip(&work.g) {
                *gi += wt * wg;
            }
            for (hi, wi) in h.iter_mut().zip(&work.info) {
                *hi += wt * wi;
            }
        }
        advance(theta, rows.qbar, rows.nbar, z, k, &mut q, &mut dq);
    }
    Some((f, g, h))
}

struct ObjectiveMeat {
    f: f64,
    g: Vec<f64>,
    h: Vec<f64>,
    j: Vec<f64>,
}

/// Evaluate the objective, gradient, expected information `H = Σ w_t I_t`,
/// and the Kish-scaled outer product of scores `J = Σ (scale · w_t)² g_t g_tᵀ`
/// for Bollerslev-Wooldridge (1992) QML sandwich standard errors at the
/// converged start fit.
fn objective_with_meat(
    rows: &Rows<'_>,
    theta: &[f64],
    scale: f64,
    work: &mut Work,
) -> Option<ObjectiveMeat> {
    let (k, p) = (rows.k, theta.len());
    let mut q = rows.qbar.to_vec();
    let mut dq = vec![0.0; p * k * k];
    let (mut f, mut g, mut h, mut j_mat) = (0.0, vec![0.0; p], vec![0.0; p * p], vec![0.0; p * p]);
    for (t, &wt) in rows.w.iter().enumerate() {
        let z = &rows.z[t * k..(t + 1) * k];
        let ld = read(&q, k, work)?;
        let l = -0.5 * (ld + quad(z, k, work));
        if !l.is_finite() {
            return None;
        }
        f += wt * l;
        score_info(&q, &dq, k, p, work);
        let wk = scale * wt;
        let wk2 = wk * wk;
        for i in 0..p {
            let gi = work.g[i];
            g[i] += wt * gi;
            for j in 0..p {
                h[i * p + j] += wt * work.info[i * p + j];
                j_mat[i * p + j] += wk2 * gi * work.g[j];
            }
        }
        advance(theta, rows.qbar, rows.nbar, z, k, &mut q, &mut dq);
    }
    Some(ObjectiveMeat { f, g, h, j: j_mat })
}

/// A start's fit and its identification and optimality checks (the module
/// docs).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Fit {
    pub theta: Vec<f64>,
    /// The projected Newton step's predicted gain at `theta`, at Kish's
    /// count.
    pub gain: f64,
    /// The likelihood-ratio statistic against constant correlation, at
    /// Kish's count.
    pub lr: f64,
    /// `G` at `theta` is positive definite.
    pub definite: bool,
    /// Largest domain-normalized asymptotic standard error at Kish's count:
    /// `max(SE(a), SE(b))` for DCC, `max(SE(a), SE(b), δ SE(g))` for ADCC;
    /// `∞` where `scale · G` does not solve.
    pub max_se: f64,
    /// Relative asymptotic standard error of the positive intercept multiplier
    /// `1 − P = 1 − cᵀ θ` at Kish's count: `SE(cᵀ θ) / (1 − cᵀ θ)`, or `∞`
    /// when shock feedback `a + δ g` is 0 or `1 − P ≤ 0` or `scale · G` does
    /// not solve.
    pub intercept_rel_se: f64,
    /// Conservative Bollerslev-Wooldridge QML (`max(H_kish⁻¹, H_kish⁻¹ J H_kish⁻¹)`)
    /// versions of `max_se`, relative shock-feedback SE `SE(a + δ g) / (a + δ g)`,
    /// and `intercept_rel_se`, used for the adaptive RPEM handoff gate.
    pub bw_max_se: f64,
    pub bw_feedback_rel_se: f64,
    pub bw_intercept_rel_se: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    pub iterations: u32,
}

/// The gain under which a start's fit is at its optimum, in nats at
/// Kish's count. At an optimum the gain is the rounding's; a fit stopped
/// short of one predicts orders of magnitude more.
pub(super) const SOUND_GAIN: f64 = 1e-6;

/// χ² at 0.999 for 2 and 3 restrictions: `−2 ln 0.001` and
/// `chi2.ppf(0.999, 3)`.
pub(super) const LR_CUT_2: f64 = 13.815_510_557_964_274;
pub(super) const LR_CUT_3: f64 = 16.266_236_196_238_13;

/// Largest asymptotic standard error on a unit-width `[0, 1)` parameter
/// domain a start may step from: `1 / (2 Φ⁻¹(0.9995)) = 1 / (2 √χ²₁(0.999))
/// ≈ 0.15195`, the same `α = 0.001` domain-localization cut `garch` uses.
pub(super) const PARAM_SE_CUT: f64 = 0.151_951_356_363_333_54;

/// Relative expected Fisher SE ceiling on the positive intercept multiplier
/// `1 − P = 1 − cᵀ θ`: `SE(ln(1 − P)) = SE(P) / (1 − P) ≤ 2` (i.e. `1 − P ≥
/// ½ SE(P)`), holding the start away from the non-stationary IDCC wall `P = 1`,
/// matching `garch`'s `MEAN_REV_REL_SE_CUT`.
pub(super) const INTERCEPT_REL_SE_CUT: f64 = 2.0;

/// Adaptive RPEM handoff cuts (when neither `start_rows` nor `start_span` is
/// explicitly given), evaluated under conservative Bollerslev-Wooldridge QML
/// standard errors so heavy-tailed innovations (`t(5)`) wait for the extra
/// rows their larger score variance requires:
/// - `bw_max_se ≤ PARAM_SE_CUT / 2` (4× information on `(a, b[, δg])`);
/// - `bw_feedback_rel_se ≤ 1 / Φ⁻¹(0.95) ≈ 0.608` (one-sided 95% Wald
///   separation of shock feedback `a + δ g > 0`, robust to finite-sample
///   sandwich noise under heavy-tailed `t(5)` where `E[z⁸] = ∞`);
/// - `bw_intercept_rel_se ≤ INTERCEPT_REL_SE_CUT` (2.0, holding the start away
///   from the IDCC wall `P = 1`; because `Q̄` is targeted separately, `1 − P`
///   does not control the long-run scale as `ω` does in free GARCH).
const ADAPTIVE_PARAM_SE_CUT: f64 = 0.5 * PARAM_SE_CUT;
const ADAPTIVE_FEEDBACK_REL_SE_CUT: f64 = 1.0 / 1.644_853_626_951_472_2;
const ADAPTIVE_INTERCEPT_REL_SE_CUT: f64 = INTERCEPT_REL_SE_CUT;

/// The cut of the likelihood-ratio statistic for `p` parameters, each a
/// restriction of constant correlation.
pub(super) fn lr_cut(p: usize) -> f64 {
    if p == 3 { LR_CUT_3 } else { LR_CUT_2 }
}

impl Fit {
    /// At its optimum: the projected step's gain at most [`SOUND_GAIN`].
    pub(super) fn sound(&self) -> bool {
        self.gain <= SOUND_GAIN
    }

    /// Its rows tell the recursion from constant correlation (the module
    /// docs); a statistic that is not a number does not.
    pub(super) fn identified(&self) -> bool {
        self.definite && self.lr >= lr_cut(self.theta.len())
    }

    /// Every parameter is localized inside its unit domain `[0, 1)` (`a`,
    /// `b`, and `δ g` for ADCC) at `α = 0.001` (`max_se ≤ PARAM_SE_CUT`) with
    /// non-degenerate shock feedback and positive intercept separated from the
    /// `P = 1` boundary (`intercept_rel_se ≤ INTERCEPT_REL_SE_CUT`).
    pub(super) fn se_identified(&self) -> bool {
        self.max_se <= PARAM_SE_CUT && self.intercept_rel_se <= INTERCEPT_REL_SE_CUT
    }

    /// Adaptive RPEM handoff gate under conservative Bollerslev-Wooldridge QML
    /// standard errors.
    pub(super) fn adaptive_se_identified(&self) -> bool {
        self.bw_max_se <= ADAPTIVE_PARAM_SE_CUT
            && self.bw_feedback_rel_se <= ADAPTIVE_FEEDBACK_REL_SE_CUT
            && self.bw_intercept_rel_se <= ADAPTIVE_INTERCEPT_REL_SE_CUT
    }
}

/// Evaluate Gaussian variance `vᵀ H_kish⁻¹ v` and Bollerslev-Wooldridge QML
/// sandwich variance `vᵀ H_kish⁻¹ J_kish H_kish⁻¹ v` for a linear combination
/// `v`. Since `H_kish x = v`, the sandwich variance is simply `xᵀ J_kish x`.
fn quad_se_pair(h_kish: &[f64], j_kish: &[f64], p: usize, v: &[f64]) -> Option<(f64, f64)> {
    let x = solve_small(h_kish, p, v)?;
    let var_g: f64 = v.iter().zip(&x).map(|(vi, xi)| vi * xi).sum();
    if !(var_g > 0.0 && var_g.is_finite()) {
        return None;
    }
    let mut var_bw = 0.0;
    for i in 0..p {
        for j in 0..p {
            var_bw += x[i] * j_kish[i * p + j] * x[j];
        }
    }
    if !var_bw.is_finite() {
        return None;
    }
    Some((var_g.sqrt(), var_g.max(var_bw).sqrt()))
}

/// `(max_se, intercept_rel_se, bw_max_se, bw_feedback_rel_se, bw_intercept_rel_se)`
/// from expected information `h` and Kish-scaled score outer product `j_kish`.
///
/// Scaling every weight by a power of two leaves `scale · H` and `j_kish`
/// unchanged to the bit.
fn se_metrics(
    theta: &[f64],
    h: &[f64],
    j_kish: &[f64],
    p: usize,
    c: &[f64],
    scale: f64,
) -> (f64, f64, f64, f64, f64) {
    let inf5 = (
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    );
    if !(scale > 0.0 && scale.is_finite()) {
        return inf5;
    }
    let scaled: Vec<f64> = h.iter().map(|&v| v * scale).collect();
    let (mut worst, mut bw_worst) = (0.0_f64, 0.0_f64);
    for (i, &ci) in c.iter().enumerate().take(p) {
        let mut e = vec![0.0; p];
        e[i] = 1.0;
        let Some((se_g, se_bw)) = quad_se_pair(&scaled, j_kish, p, &e) else {
            return inf5;
        };
        let (sg, sb) = (ci * se_g, ci * se_bw);
        if !(sg.is_finite() && sb.is_finite()) {
            return inf5;
        }
        worst = worst.max(sg);
        bw_worst = bw_worst.max(sb);
    }
    let feedback = theta[0] + if p == 3 { c[2] * theta[2] } else { 0.0 };
    let one_minus_p = 1.0 - c.iter().zip(theta).map(|(ci, ti)| ci * ti).sum::<f64>();
    let bw_feedback_rel_se = if feedback > 0.0 && feedback.is_finite() {
        let mut cf = vec![0.0; p];
        cf[0] = 1.0;
        if p == 3 {
            cf[2] = c[2];
        }
        quad_se_pair(&scaled, j_kish, p, &cf).map_or(f64::INFINITY, |(_, se_bw)| se_bw / feedback)
    } else {
        f64::INFINITY
    };
    let (intercept_rel_se, bw_intercept_rel_se) =
        if feedback > 0.0 && one_minus_p > 0.0 && one_minus_p.is_finite() {
            quad_se_pair(&scaled, j_kish, p, c)
                .map_or((f64::INFINITY, f64::INFINITY), |(sg, sb)| {
                    (sg / one_minus_p, sb / one_minus_p)
                })
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
    (
        worst,
        intercept_rel_se,
        bw_worst,
        bw_feedback_rel_se,
        bw_intercept_rel_se,
    )
}

/// The constraint's coefficients: `(1, 1)`, or `(1, 1, δ)`.
pub(super) fn coefs(p: usize, delta: f64) -> Vec<f64> {
    if p == 3 {
        vec![1.0, 1.0, delta]
    } else {
        vec![1.0, 1.0]
    }
}

/// The grid the fit starts from: `a` of 0.01, 0.03, 0.06, `b` of 0.85, 0.93,
/// 0.97 and, for the asymmetric model, `g` of 0.01 and 0.05.
fn grid(p: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for a in [0.01, 0.03, 0.06] {
        for b in [0.85, 0.93, 0.97] {
            if p == 3 {
                for g in [0.01, 0.05] {
                    out.push(vec![a, b, g]);
                }
            } else {
                out.push(vec![a, b]);
            }
        }
    }
    out
}

/// Maximize the buffer's likelihood over the set with `δ` as the targets
/// give it; `None` where no grid point has a finite value.
pub(super) fn fit(rows: &Rows<'_>, p: usize, delta: f64, work: &mut Work) -> Option<Fit> {
    let c = coefs(p, delta);
    let eye = super::region::identity(p);
    let mut best: Option<(Vec<f64>, f64)> = None;
    for point in grid(p) {
        let th = project(&point, &eye, &c, PERSISTENCE_CAP);
        if let Some((f, _, _)) = objective(rows, &th, false, work)
            && best.as_ref().is_none_or(|(_, b)| f > *b)
        {
            best = Some((th, f));
        }
    }
    let (mut theta, _) = best?;
    let mut it = 0;
    while it < MAX_ITER {
        it += 1;
        let Some((f, g, h)) = objective(rows, &theta, true, work) else {
            break;
        };
        let mut hr = h.clone();
        for i in 0..p {
            hr[i * p + i] += 1e-10 * h[i * p + i];
        }
        let Some(d) = solve_small(&hr, p, &g) else {
            break;
        };
        let mut accepted = None;
        let mut s = 1.0;
        for _ in 0..=MAX_HALVINGS {
            let trial: Vec<f64> = theta.iter().zip(&d).map(|(t, d)| t + s * d).collect();
            let cand = project(&trial, &h, &c, PERSISTENCE_CAP);
            if cand == theta {
                break;
            }
            if let Some((fc, _, _)) = objective(rows, &cand, false, work) {
                let gain: f64 = g
                    .iter()
                    .zip(cand.iter().zip(&theta))
                    .map(|(g, (c, t))| g * (c - t))
                    .sum();
                if fc >= f + ARMIJO * gain {
                    accepted = Some((cand, fc));
                    break;
                }
            }
            s *= 0.5;
        }
        let Some((cand, fc)) = accepted else {
            break;
        };
        let small = cand.iter().zip(&theta).all(|(a, b)| (a - b).abs() <= 1e-10);
        let flat = (fc - f).abs() <= 1e-12 * (1.0 + f.abs());
        theta = cand;
        if small || flat {
            break;
        }
    }
    let (w, w2) = rows
        .w
        .iter()
        .fold((0.0, 0.0), |(a, b), w| (a + w, b + w * w));
    let scale = w / w2;
    let (
        gain,
        lr,
        definite,
        max_se,
        intercept_rel_se,
        bw_max_se,
        bw_feedback_rel_se,
        bw_intercept_rel_se,
    ) = match objective_with_meat(rows, &theta, scale, work) {
        Some(ObjectiveMeat { f, g, h, j: j_kish }) => {
            let gain = projected_gain(&theta, &g, &h, &c);
            let f0 = objective(rows, &vec![0.0; p], false, work).map_or(f64::NAN, |e| e.0);
            let mut l = vec![0.0; p * p];
            let definite = super::mat::cholesky(&h, p, &mut l);
            let (max_se, intercept_rel_se, bw_max_se, bw_feedback_rel_se, bw_intercept_rel_se) =
                if definite {
                    se_metrics(&theta, &h, &j_kish, p, &c, scale)
                } else {
                    (
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::INFINITY,
                    )
                };
            (
                gain * scale,
                2.0 * (f - f0) * scale,
                definite,
                max_se,
                intercept_rel_se,
                bw_max_se,
                bw_feedback_rel_se,
                bw_intercept_rel_se,
            )
        }
        None => (
            f64::INFINITY,
            f64::NAN,
            false,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ),
    };
    Some(Fit {
        theta,
        gain: if gain.is_nan() { f64::INFINITY } else { gain },
        lr,
        definite,
        max_se,
        intercept_rel_se,
        bw_max_se,
        bw_feedback_rel_se,
        bw_intercept_rel_se,
        iterations: it,
    })
}

/// The gain the projected Newton step predicts at `theta`: `gᵀΔ − ½
/// ΔᵀGΔ`, `Δ = Proj_G(θ + G⁻¹g) − θ`; infinite where `G` does not solve.
pub(super) fn projected_gain(theta: &[f64], g: &[f64], h: &[f64], c: &[f64]) -> f64 {
    let p = theta.len();
    let mut hr = h.to_vec();
    for i in 0..p {
        hr[i * p + i] += 1e-10 * h[i * p + i];
    }
    let Some(d) = solve_small(&hr, p, g) else {
        return f64::INFINITY;
    };
    let trial: Vec<f64> = theta.iter().zip(&d).map(|(t, d)| t + d).collect();
    let full = project(&trial, h, c, PERSISTENCE_CAP);
    let delta: Vec<f64> = full.iter().zip(theta).map(|(a, b)| a - b).collect();
    let lin: f64 = g.iter().zip(&delta).map(|(g, d)| g * d).sum();
    let quad: f64 = (0..p)
        .map(|i| delta[i] * (0..p).map(|j| h[i * p + j] * delta[j]).sum::<f64>())
        .sum();
    (lin - 0.5 * quad).abs()
}
