//! One target's Student's t fit for `dist_fit`: the weighted maximum
//! likelihood of `θ = (loc, ln scale, η = 1/ν)` stepped one row at a time.
//!
//! **The start** is the method of moments, from the weighted moments the
//! normal's fit keeps beside it and the third and fourth kept here:
//!
//! ```text
//! κ = c₄/c₂² − 3,   η₀ = κ/(4κ + 6) for κ > 0 (ν = 4 + 6/κ), else 0
//! loc₀ = the mean (or floc),   scale₀ = √(c₂ (1 − 2η₀))
//! ```
//!
//! taken on the first learned row whose checks pass: a positive, finite
//! variance (moments about `floc` under a fixed location), a finite start,
//! and a scale the start's own information pins down: the standard error of
//! `ln scale`, `√([I(θ₀)⁻¹]_{ln s} Σω²)/W`, at most 1/4. That is a precision,
//! not a count: about 19 rows of equal weight at a normal tail, more under
//! unequal weights. The rows it summarizes enter the convergence surrogate
//! below as one pseudo-row at `θ₀` of their weight `W₀`: curvature
//! `W₀ I(θ₀)`, score 0.
//!
//! **Each learned row** of weight `w > 0` moves the fit to the maximum of
//! the quadratic the rows before it make, at their decayed weight `λW` and
//! the information at the fit, plus the row's own log density, exactly:
//!
//! ```text
//! θ_n = argmax_θ  w log f(y_n; θ) − ½ λW (θ − θ_{n−1})ᵀ I(θ_{n−1}) (θ − θ_{n−1}),   η ∈ [0, 1]
//! ```
//!
//! with `I` the expected (Fisher) information per unit weight, which for
//! the t depends on the scale and the tail alone, so it is read at the fit
//! as it stands rather than summed at the fits each row was learned at (a
//! sum that would freeze `loc` after an early scale far too small: `I_loc`
//! is `1/scale²`). A row the fit finds
//! surprising moves it only as far as its own density says, where a single
//! Fisher-scoring step would follow the score's `z⁴` in `η`. The maximum is
//! found by Newton's method on the row's observed information (the
//! expected information plus the observed's excess on the diagonal where
//! the observed is not positive definite), with step halving and the tail
//! held in its bounds; it stops on the decrement, when the gain a further
//! step promises is below `10⁻¹⁵ w`, or where halving finds no gain.
//!
//! **Convergence.** With `H = W I(θ)`, the observed-curvature surrogate of every
//! row learned, `Σ ω [g_i − J_i (θ − θ_i)]` with each row's score `g_i` and
//! observed information `J_i` at the fit it was learned at (the start's
//! pseudo-row's `J` its information), is kept as its gradient `G` at the
//! current fit and its curvature: the gradient of the whole weighted log
//! likelihood to second order in each row. Its Newton decrement `Gᵀ H⁻¹ G`
//! (over the coordinates not held at a bound) is twice the log likelihood
//! a step to that maximum would gain; the fit is converged while it is at
//! most what sampling noise alone puts between the truth and the maximum at
//! 95%, `χ²_p(0.95) Σω²/W` (`Σω²/W` the Kish correction for unequal
//! weights).
//!
//! **The tail.** `η = 0` is the normal, inside the parameter set at its
//! boundary, where the information is finite and positive definite, so the
//! Wald statistic `η² / Var(η)`, `Var(η) = [H⁻¹]_ηη Σω²/W`, has the
//! boundary law `½χ²₀ + ½χ²₁` (Self and Liang 1987); the tail is
//! identified while it exceeds that law's 95% quantile, 2.7055.

use serde::{Deserialize, Serialize};

use super::student_t;
use crate::EwDiag;

/// The largest `η`, `ν = 1`: Cauchy's tail.
pub(super) const ETA_MAX: f64 = 1.0;
/// `χ²_p`'s 0.95 quantile, `p = 1, 2, 3` (`scipy.stats.chi2.ppf(0.95, p)`).
const CHI2_95: [f64; 3] = [
    3.841_458_820_694_124,
    5.991_464_547_107_979,
    7.814_727_903_251_178,
];
/// `½χ²₀ + ½χ²₁`'s 0.95 quantile, `χ²₁`'s 0.90 (`scipy.stats.chi2.ppf(0.9, 1)`).
const WALD_95: f64 = 2.705_543_454_095_404;
/// The largest standard error of the start's `ln scale` at which it is
/// taken.
const START_SE: f64 = 0.25;
/// A safety bound on one row's Newton iterations: the decrement or a failed
/// halving stops it first (two or three iterations a row, in the tests).
const MAX_NEWTON: usize = 64;
/// The smallest step fraction halving tries.
const MIN_STEP: f64 = 1.0 / 1_073_741_824.0;

/// Why a started fit reports less than all it has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TStatus {
    /// Not started, or the decrement is above the noise.
    NotConverged,
    /// Converged, but the Wald test does not tell the tail from the normal.
    ShapeUnidentified,
    /// Everything reports.
    Identified,
}

/// One target's t fit. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct TFit {
    /// `Σ ω²`, every learned row's squared weight, decayed by `λ²`.
    w2: f64,
    /// The weighted central third and fourth moments, normalized by the
    /// weight (of `y − floc` under a fixed location), beside the shared
    /// accumulator's second.
    c3: f64,
    c4: f64,
    /// The fit, from the moment start on.
    fit: Option<TState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TState {
    /// `(loc, ln scale, η)`.
    theta: [f64; 3],
    /// The observed-curvature surrogate's gradient at `theta`.
    grad: [f64; 3],
    /// Its curvature, `Σ ω J_i`, packed `[00, 01, 02, 11, 12, 22]`.
    curv: [f64; 6],
}

type Mat = [[f64; 3]; 3];

fn full(p: &[f64; 6]) -> Mat {
    [[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]]
}

/// `W I(θ)`, the expected information of the weight `w` at `θ`, packed.
fn info_at(w: f64, theta: &[f64; 3]) -> [f64; 6] {
    let [i0, i1, i2, i3] = student_t::expected_info(theta[2]);
    let s2 = libm::exp(2.0 * theta[1]);
    [w * i0 / s2, 0.0, 0.0, w * i1, w * i2, w * i3]
}

/// `[I⁻¹]_{ln s, ln s}` or `[I⁻¹]_ηη` from the `(ln s, η)` block of a
/// packed information (`loc` is orthogonal to both).
fn inv_diag(h: &[f64; 6], eta: bool) -> f64 {
    let det = h[3] * h[5] - h[4] * h[4];
    (if eta { h[3] } else { h[5] }) / det
}

fn axpy(acc: &mut [f64; 6], w: f64, p: &[f64; 6]) {
    for (a, v) in acc.iter_mut().zip(p) {
        *a += w * v;
    }
}

fn mat_vec(m: &Mat, v: &[f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// `m⁻¹ b` on the coordinates `free` names, 0 on the others; `None` where
/// that block is not positive definite.
fn chol_solve(m: &Mat, b: &[f64; 3], free: [bool; 3]) -> Option<[f64; 3]> {
    let idx: Vec<usize> = (0..3).filter(|&i| free[i]).collect();
    let n = idx.len();
    let mut l = [[0.0f64; 3]; 3];
    for i in 0..n {
        for j in 0..=i {
            let mut s = m[idx[i]][idx[j]];
            for (a, b) in l[i][..j].iter().zip(&l[j][..j]) {
                s -= a * b;
            }
            if i == j {
                if !(s > 0.0 && s.is_finite()) {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = [0.0f64; 3];
    for i in 0..n {
        let mut s = b[idx[i]];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut out = [0.0f64; 3];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k][i] * out[idx[k]];
        }
        out[idx[i]] = s / l[i][i];
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// One row's log density, scores, observed and expected information at
/// `θ`, in `θ`'s own units; `None` where any is not finite (a value so far
/// out that its powers overflow), which the fit then does not learn.
struct Eval {
    ll: f64,
    g: [f64; 3],
    obs: [f64; 6],
    exp: [f64; 6],
}

#[cfg(test)]
impl TFit {
    pub(super) fn theta(&self) -> Option<[f64; 3]> {
        self.fit.as_ref().map(|f| f.theta)
    }
}

fn eval(y: f64, theta: &[f64; 3]) -> Option<Eval> {
    let [loc, ln_s, eta] = *theta;
    let s = libm::exp(ln_s);
    let z = (y - loc) / s;
    let [g0, g1, g2] = student_t::scores(z, eta);
    let j = student_t::observed_info(z, eta);
    let [i0, i1, i2, i3] = student_t::expected_info(eta);
    let e = Eval {
        ll: student_t::log_density(z, eta) - ln_s,
        g: [g0 / s, g1, g2],
        obs: [j[0] / (s * s), j[1] / s, j[2] / s, j[3], j[4], j[5]],
        exp: [i0 / (s * s), 0.0, 0.0, i1, i2, i3],
    };
    let finite = e.ll.is_finite()
        && e.g.iter().all(|v| v.is_finite())
        && e.obs.iter().chain(&e.exp).all(|v| v.is_finite());
    finite.then_some(e)
}

/// The central third and fourth moments `(c₃, c₄)` after the row `x` of
/// weight `w > 0` and a decay of `lam`, read against the shared accumulator
/// `m` *before* it takes the row ([`TFit::learn_moments`] has the update);
/// the skewed t keeps the same two.
pub(super) fn third_fourth(c3: f64, c4: f64, m: &EwDiag, x: f64, lam: f64, w: f64) -> (f64, f64) {
    let w_new = lam * m.n_eff() + w;
    let a = lam * m.n_eff() / w_new;
    let b = w / w_new;
    let d = m.deviation(0, x);
    let c2 = m.var(0);
    let (ab, d2) = (a * b, d * d);
    let c4 = a * c4 - 4.0 * ab * d * c3
        + 6.0 * ab * b * d2 * c2
        + ab * d2 * d2 * (a * a * a + b * b * b);
    let c3 = a * c3 - 3.0 * ab * d * c2 + ab * d2 * d * (a * a - b * b);
    (c3, c4)
}

impl TFit {
    pub(super) fn new() -> Self {
        Self {
            w2: 0.0,
            c3: 0.0,
            c4: 0.0,
            fit: None,
        }
    }

    pub(super) fn started(&self) -> bool {
        self.fit.is_some()
    }

    /// `(loc, scale, η)`, once started.
    pub(super) fn params(&self) -> Option<(f64, f64, f64)> {
        self.fit
            .as_ref()
            .map(|f| (f.theta[0], libm::exp(f.theta[1]), f.theta[2]))
    }

    /// Every sum ages by the row's decay `lam`; the fit does not move.
    pub(super) fn decay(&mut self, lam: f64) {
        self.w2 *= lam * lam;
        if let Some(f) = &mut self.fit {
            for v in &mut f.curv {
                *v *= lam;
            }
            for v in &mut f.grad {
                *v *= lam;
            }
        }
    }

    /// The third and fourth moments take the row `x` (already shifted by
    /// `floc`) of weight `w > 0` after a decay of `lam`, read against the
    /// shared accumulator `m` *before* it takes the row: Pébay's weighted
    /// update,
    ///
    /// ```text
    /// a = λW/(λW + w),  b = w/(λW + w),  d = x − mean
    /// c₃' = a c₃ − 3ab d c₂ + ab d³ (a² − b²)
    /// c₄' = a c₄ − 4ab d c₃ + 6ab² d² c₂ + ab d⁴ (a³ + b³)
    /// ```
    pub(super) fn learn_moments(&mut self, m: &EwDiag, x: f64, lam: f64, w: f64) {
        self.w2 += w * w;
        (self.c3, self.c4) = third_fourth(self.c3, self.c4, m, x, lam, w);
    }

    /// The moment start, tried on a learned row once the shared accumulator
    /// `m` has taken it: whether it was taken.
    pub(super) fn try_start(&mut self, m: &EwDiag, floc: Option<f64>) -> bool {
        let (hi, lo) = m.mean_pair(0);
        let mean = hi + lo;
        let c2 = m.var(0);
        let (loc, var, kurt) = match floc {
            None => (mean, c2, self.c4 / (c2 * c2) - 3.0),
            Some(f) => {
                let raw2 = c2 + mean * mean;
                let m2 = mean * mean;
                let raw4 = self.c4 + 4.0 * mean * self.c3 + 6.0 * m2 * c2 + m2 * m2;
                (f, raw2, raw4 / (raw2 * raw2) - 3.0)
            }
        };
        if !(var > 0.0 && var.is_finite() && kurt.is_finite()) {
            return false;
        }
        let eta = if kurt > 0.0 {
            kurt / (4.0 * kurt + 6.0)
        } else {
            0.0
        };
        let ln_s = 0.5 * libm::log(var * (1.0 - 2.0 * eta));
        if !ln_s.is_finite() {
            return false;
        }
        let theta = [loc, ln_s, eta];
        let w0 = m.n_eff();
        let info = info_at(w0, &theta);
        let se2 = inv_diag(&info, false) * self.w2 / w0;
        if !(info.iter().all(|v| v.is_finite()) && se2 <= START_SE * START_SE) {
            return false;
        }
        self.fit = Some(TState {
            theta,
            grad: [0.0; 3],
            curv: info,
        });
        true
    }

    /// Learn the row `y` of weight `w > 0`, the sums already decayed and
    /// `w_before` the weight of the rows before it at the row's decay. A row
    /// whose terms are not finite at the fit is not learned.
    pub(super) fn learn(&mut self, y: f64, w: f64, w_before: f64, fixed_loc: bool) {
        let Some(st) = &mut self.fit else { return };
        let Some(mut e) = eval(y, &st.theta) else {
            return;
        };
        let h = full(&info_at(w_before, &st.theta));
        let theta = st.theta;
        let mut cur = theta;
        let delta = |c: &[f64; 3]| [0, 1, 2].map(|k| c[k] - theta[k]);
        let value = |ll: f64, c: &[f64; 3]| {
            let d = delta(c);
            let hd = mat_vec(&h, &d);
            w * ll - 0.5 * (d[0] * hd[0] + d[1] * hd[1] + d[2] * hd[2])
        };
        let mut val = value(e.ll, &cur);
        for _ in 0..MAX_NEWTON {
            let hd = mat_vec(&h, &delta(&cur));
            let gr = [0, 1, 2].map(|k| w * e.g[k] - hd[k]);
            let mut free = [!fixed_loc, true, true];
            if (cur[2] <= 0.0 && gr[2] <= 0.0) || (cur[2] >= ETA_MAX && gr[2] >= 0.0) {
                free[2] = false;
            }
            let mut m_obs = h;
            let mut m_fish = h;
            let (o, x) = (full(&e.obs), full(&e.exp));
            for i in 0..3 {
                for j in 0..3 {
                    m_obs[i][j] += w * o[i][j];
                    m_fish[i][j] += w * x[i][j];
                }
                // the observed's excess on the diagonal, where it is larger
                m_fish[i][i] += w * (o[i][i] - x[i][i]).max(0.0);
            }
            let Some(step) =
                chol_solve(&m_obs, &gr, free).or_else(|| chol_solve(&m_fish, &gr, free))
            else {
                break;
            };
            let dec: f64 = (0..3).map(|k| step[k] * gr[k]).sum();
            if dec.is_nan() || 0.5 * dec <= 1e-15 * w {
                break;
            }
            let mut t = 1.0;
            let mut moved = false;
            while t >= MIN_STEP {
                let mut c = [0, 1, 2].map(|k| cur[k] + t * step[k]);
                c[2] = c[2].clamp(0.0, ETA_MAX);
                if let Some(ec) = eval(y, &c) {
                    let v = value(ec.ll, &c);
                    if v > val {
                        cur = c;
                        val = v;
                        e = ec;
                        moved = true;
                        break;
                    }
                }
                t *= 0.5;
            }
            if !moved {
                break;
            }
        }
        // The surrogate's gradient moves with the fit along its curvature
        // (decayed, before the row joins it), then takes the row's score.
        let d = delta(&cur);
        let cd = mat_vec(&full(&st.curv), &d);
        for ((gk, ek), ck) in st.grad.iter_mut().zip(&e.g).zip(&cd) {
            *gk += w * ek - ck;
        }
        axpy(&mut st.curv, w, &e.obs);
        st.theta = cur;
    }

    /// The fit's standing, read against the target's weight `w_sum`.
    pub(super) fn status(&self, w_sum: f64, fixed_loc: bool) -> TStatus {
        let Some(st) = &self.fit else {
            return TStatus::NotConverged;
        };
        let kish = self.w2 / w_sum;
        let info = info_at(w_sum, &st.theta);
        let h = full(&info);
        let g = st.grad;
        let eta = st.theta[2];
        let mut free = [!fixed_loc, true, true];
        if (eta <= 0.0 && g[2] <= 0.0) || (eta >= ETA_MAX && g[2] >= 0.0) {
            free[2] = false;
        }
        let Some(step) = chol_solve(&h, &g, free) else {
            return TStatus::NotConverged;
        };
        let dec: f64 = (0..3).map(|k| step[k] * g[k]).sum();
        let p = free.iter().filter(|f| **f).count();
        if dec.is_nan() || dec > kish * CHI2_95[p - 1] {
            return TStatus::NotConverged;
        }
        let var = inv_diag(&info, true) * kish;
        if eta * eta > WALD_95 * var && var > 0.0 {
            TStatus::Identified
        } else {
            TStatus::ShapeUnidentified
        }
    }

    /// Whether the state's numbers are of a shape and kind a fit can hold:
    /// a restored file's check.
    pub(super) fn sound(&self) -> bool {
        self.fit
            .as_ref()
            .is_none_or(|f| (0.0..=ETA_MAX).contains(&f.theta[2]))
    }
}

#[cfg(test)]
mod tests;
