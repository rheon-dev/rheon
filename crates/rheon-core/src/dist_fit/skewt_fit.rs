//! One target's fit of Hansen's skewed t for `dist_fit`: the weighted
//! maximum likelihood of `loc` (the mean), `scale` (the standard deviation
//! `σ`), `η = 1/ν` and `λ`, stepped one row at a time (`skew_t` has the
//! density). The scheme is the t's (`t_fit`); what differs is below.
//!
//! **The coordinates** stepped are `θ = (loc, ln s, η, λ)` with `s = σ √(1
//! − 2η)`, the scale of the t inside the density as scipy writes the t, not
//! `ln σ`. Near `ν = 2` the standard deviation and the tail are nearly one
//! direction of the likelihood: the information's `(ln σ, η)` block has a
//! determinant three parts in ten thousand of its diagonal's product at
//! `ν = 2.05`, so a quadratic about the fit is cheap far along it, and a
//! row at the centre once moved `ln σ` by −20 for almost nothing. In `ln s`
//! the block is the t's own, well conditioned; a change of coordinates
//! moves no maximum, only the quadratic each row is stepped against.
//!
//! **The start** is the method of moments for the location, the scale and
//! the tail, the skew left to the likelihood. With `κ` the weighted excess
//! kurtosis of the moments the normal's fit keeps beside it and the fourth
//! kept here:
//!
//! ```text
//! loc₀ = the mean (or floc),   scale₀ = the standard deviation (about floc under it)
//! η₀ = κ/(4κ + 6) for κ > 0 (the symmetric family's kurtosis 6η/(1 − 4η)), else 0,   λ₀ = 0
//! ```
//!
//! taken on the first learned row whose checks pass: a positive, finite
//! variance (moments about `floc` under a fixed location), a finite start,
//! and standard errors of `ln scale` and of `λ` by the start's own
//! information of at most 1/4 each (about 35 rows of equal weight at a
//! normal tail, more under unequal weights). The skew is not matched to the
//! weighted skewness, though the family has a closed skewness: on so few
//! rows the sample skewness is noisy and the family's skewness rises slowly
//! in `λ`, so the matched `λ₀` lands far out (0.85 on some normal streams),
//! and the convergence check below, which holds the start's rows as a
//! quadratic about it, then carries the error of the fit's walk back for
//! half-lives. Measured on 60 normal streams of 3,000 rows (half-life
//! 1,000): matching the skewness left 3,544 rows `not_converged`, starting
//! at `λ₀ = 0` none; on the skewed streams of the tests the likelihood
//! finds the skew from 0. The rows the start summarizes enter the
//! surrogate as the t's do, one pseudo-row at `θ₀` of their weight `W₀`.
//!
//! **Each learned row** of weight `w > 0` moves the fit to the maximum of
//! the quadratic the rows before it make, at their decayed weight `λW` and
//! the information at the fit, plus the row's own log density:
//!
//! ```text
//! θ_n = argmax_θ  w log f(y_n; θ) − ½ λW (θ − θ_{n−1})ᵀ I(θ_{n−1}) (θ − θ_{n−1})
//! η ∈ [0, 1/2.05],   λ ∈ [−0.99, 0.99]
//! ```
//!
//! with `I` the closed expected information. The maximum is found as the
//! t's is: Newton's method on the row's observed information (the expected
//! plus the observed's excess on the diagonal where the observed is not
//! positive definite), with step halving and the shape held in its bounds;
//! it stops on the decrement, when the gain a further step promises is
//! below `10⁻¹⁵ w`, or where halving finds no gain.
//!
//! **Convergence.** As the t's: the observed-curvature surrogate of every
//! row learned, `Σ ω [g_i − J_i (θ − θ_i)]`, kept as its gradient `G` at
//! the current fit and its curvature; `not_converged` while `Gᵀ (W I(θ))⁻¹
//! G` exceeds `χ²_p(0.95) Σω²/W` over the coordinates not held at a bound.
//!
//! **The shape.** `(η, λ) = (0, 0)` is the normal. `λ = 0` is inside the
//! parameter set and regular there (the information is positive definite),
//! while `η = 0` is its boundary, as for the t. The shape is identified
//! while the joint Wald statistic `sᵀ V⁻¹ s`, `s = (η, λ)`, `V` the `(η, λ)`
//! block of `(W I)⁻¹ Σω²/W`, exceeds the 95% quantile of its law under the
//! normal: one parameter on the boundary and one inside make it `½χ²₁ +
//! ½χ²₂` (Self and Liang 1987), whose 95% quantile is 5.1384. A target
//! whose test does not reject is `shape_unidentified`, its `loc` and
//! `scale` reported.

use serde::{Deserialize, Serialize};

use super::skew_t::{ETA_MAX, LAM_MAX, Shape};
use super::t_fit::TStatus;
use super::t_fit::third_fourth;
use crate::EwDiag;

/// `χ²_p`'s 0.95 quantile, `p = 1, …, 4` (`scipy.stats.chi2.ppf(0.95, p)`).
const CHI2_95: [f64; 4] = [
    3.841_458_820_694_124,
    5.991_464_547_107_979,
    7.814_727_903_251_178,
    9.487_729_036_781_154,
];
/// `½χ²₁ + ½χ²₂`'s 0.95 quantile (by `scipy.optimize.brentq` on the two
/// survival functions).
const WALD2_95: f64 = 5.138_380_785_321_448_5;
/// The largest standard error of the start's `ln scale`, and of its skew
/// `λ`, at which it is taken. The skew's is the t's check added: with only
/// the scale's, the start comes on the eighth row of a normal stream, and
/// from there some streams' tails meet their bound and the fit never
/// converges in 20,000 rows.
const START_SE: f64 = 0.25;
/// A safety bound on one row's iterations: the decrement or a failed
/// halving stops it first.
const MAX_STEPS: usize = 64;
/// The smallest step fraction halving tries.
const MIN_STEP: f64 = 1.0 / 1_073_741_824.0;

type Mat = [[f64; 4]; 4];

/// One target's skewed-t fit. See the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SkewTFit {
    /// `Σ ω²`, every learned row's squared weight, decayed by `λ²`.
    w2: f64,
    /// The weighted central third and fourth moments, normalized by the
    /// weight (of `y − floc` under a fixed location).
    c3: f64,
    c4: f64,
    /// The fit, from the moment start on.
    fit: Option<SkewTState>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SkewTState {
    /// `(loc, ln s, η, λ)`, `s = σ √(1 − 2η)`.
    theta: [f64; 4],
    /// The surrogate's gradient at `theta`.
    grad: [f64; 4],
    /// Its curvature, `Σ ω J_i`, packed `[00, 01, 02, 03, 11, 12, 13,
    /// 22, 23, 33]`.
    curv: [f64; 10],
}

fn full(p: &[f64; 10]) -> Mat {
    [
        [p[0], p[1], p[2], p[3]],
        [p[1], p[4], p[5], p[6]],
        [p[2], p[5], p[7], p[8]],
        [p[3], p[6], p[8], p[9]],
    ]
}

fn mat_vec(m: &Mat, v: &[f64; 4]) -> [f64; 4] {
    [0, 1, 2, 3].map(|i| (0..4).map(|k| m[i][k] * v[k]).sum())
}

fn dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    (0..4).map(|k| a[k] * b[k]).sum()
}

/// `m⁻¹ b` on the coordinates `free` names, 0 on the others; `None` where
/// that block is not positive definite.
fn chol_solve(m: &Mat, b: &[f64; 4], free: [bool; 4]) -> Option<[f64; 4]> {
    let idx: Vec<usize> = (0..4).filter(|&i| free[i]).collect();
    let n = idx.len();
    let mut l = [[0.0f64; 4]; 4];
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
    let mut y = [0.0f64; 4];
    for i in 0..n {
        let mut s = b[idx[i]];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut out = [0.0f64; 4];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k][i] * out[idx[k]];
        }
        out[idx[i]] = s / l[i][i];
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// `ln σ`, the log standard deviation, at `θ = (loc, ln s, η, λ)`:
/// `ln s − ½ ln(1 − 2η)`.
fn ln_sd(theta: &[f64; 4]) -> f64 {
    theta[1] - 0.5 * libm::log1p(-2.0 * theta[2])
}

/// `W I(θ)`, the expected information of the weight `w` at `θ`, packed,
/// in the stepped coordinates.
fn info_at(w: f64, theta: &[f64; 4]) -> [f64; 10] {
    let sh = Shape::new(theta[2], theta[3]);
    let v = 1.0 - 2.0 * theta[2];
    to_kernel(&scaled_info(w, &sh.info(&sh.derivs()), ln_sd(theta)), v)
}

/// `Mᵀ A M` for a packed `A` in `(loc, ln σ, η, λ)`, with `M = ∂(loc, ln σ,
/// η, λ)/∂(loc, ln s, η, λ)` the identity but `∂ln σ/∂η = 1/v`, `v = 1 −
/// 2η`: the information moved to the stepped coordinates.
fn to_kernel(a: &[f64; 10], v: f64) -> [f64; 10] {
    let m = full(a);
    let mut out = [0.0; 10];
    let mut at = 0;
    for r in 0..4 {
        for c in r..4 {
            let mut x = m[r][c];
            if c == 2 {
                x += m[r][1] / v;
            }
            if r == 2 {
                x += m[1][c] / v;
            }
            if r == 2 && c == 2 {
                x += m[1][1] / (v * v);
            }
            out[at] = x;
            at += 1;
        }
    }
    out
}

/// The information at `σ = 1` times `w`, its `loc` entries moved to the
/// scale `exp(ln_s)`.
fn scaled_info(w: f64, i: &[f64; 10], ln_s: f64) -> [f64; 10] {
    let s = libm::exp(ln_s);
    let mut out = [0.0; 10];
    for (k, (o, v)) in out.iter_mut().zip(i).enumerate() {
        let v = match k {
            0 => v / (s * s),
            1..=3 => v / s,
            _ => *v,
        };
        *o = w * v;
    }
    out
}

/// One row's log density, scores, observed and expected information at
/// `θ`, in `θ`'s own units; `None` where any is not finite, which the fit
/// then does not learn.
struct Eval {
    ll: f64,
    g: [f64; 4],
    obs: [f64; 10],
    exp: [f64; 10],
}

/// The same in the stepped coordinates `(loc, ln s, η, λ)`: by the chain
/// rule through `ln σ = ln s − ½ ln(1 − 2η)`, whose second derivative in
/// `η`, `2/v²`, adds `−2 g_{ln σ}/v²` to the observed information's `η, η`.
fn eval(y: f64, theta: &[f64; 4]) -> Option<Eval> {
    let [loc, _, eta, lam] = *theta;
    let ln_s = ln_sd(theta);
    let v = 1.0 - 2.0 * eta;
    let s = libm::exp(ln_s);
    let z = (y - loc) / s;
    let sh = Shape::new(eta, lam);
    let d = sh.derivs();
    let (g, obs) = sh.row(z, &d);
    let mut obs = to_kernel(&scaled_info(1.0, &obs, ln_s), v);
    obs[7] -= 2.0 * g[1] / (v * v);
    let e = Eval {
        ll: sh.log_density(z) - ln_s,
        g: [g[0] / s, g[1], g[2] + g[1] / v, g[3]],
        obs,
        exp: to_kernel(&scaled_info(1.0, &sh.info(&d), ln_s), v),
    };
    let finite = e.ll.is_finite()
        && e.g.iter().all(|v| v.is_finite())
        && e.obs.iter().chain(&e.exp).all(|v| v.is_finite());
    finite.then_some(e)
}

/// The coordinates a step may move: `loc` unless it is fixed, and each
/// shape parameter unless it sits at a bound with the gradient pushing it
/// out.
fn free_at(theta: &[f64; 4], g: &[f64; 4], fixed_loc: bool) -> [bool; 4] {
    let held = |v: f64, lo: f64, hi: f64, g: f64| (v <= lo && g <= 0.0) || (v >= hi && g >= 0.0);
    [
        !fixed_loc,
        true,
        !held(theta[2], 0.0, ETA_MAX, g[2]),
        !held(theta[3], -LAM_MAX, LAM_MAX, g[3]),
    ]
}

fn clamp_shape(c: &mut [f64; 4]) {
    c[2] = c[2].clamp(0.0, ETA_MAX);
    c[3] = c[3].clamp(-LAM_MAX, LAM_MAX);
}

#[cfg(test)]
impl SkewTFit {
    pub(super) fn theta(&self) -> Option<[f64; 4]> {
        self.fit.as_ref().map(|f| f.theta)
    }
}

impl SkewTFit {
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

    /// `(loc, scale, η, λ)`, once started, `scale` the standard deviation.
    pub(super) fn params(&self) -> Option<(f64, f64, f64, f64)> {
        self.fit.as_ref().map(|f| {
            (
                f.theta[0],
                libm::exp(ln_sd(&f.theta)),
                f.theta[2],
                f.theta[3],
            )
        })
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
    /// shared accumulator `m` before it takes the row.
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
                // the raw moments of y − floc
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
        let v = 1.0 - 2.0 * eta;
        let theta = [loc, 0.5 * libm::log(var * v), eta, 0.0];
        let w0 = m.n_eff();
        let info = info_at(w0, &theta);
        if !(theta.iter().all(|v| v.is_finite()) && info.iter().all(|v| v.is_finite())) {
            return false;
        }
        // The variances of ln σ = ln s − ½ ln(1 − 2η) and of λ, `aᵀ (W
        // I)⁻¹ a Σω²/W` with `a` their gradients in the stepped coordinates.
        let free = [floc.is_none(), true, true, true];
        let h = full(&info);
        for a in [[0.0, 1.0, 1.0 / v, 0.0], [0.0, 0.0, 0.0, 1.0]] {
            let Some(col) = chol_solve(&h, &a, free) else {
                return false;
            };
            // a NaN variance is no start either
            let var = dot(&a, &col) * self.w2 / w0;
            if var.is_nan() || var > START_SE * START_SE {
                return false;
            }
        }
        self.fit = Some(SkewTState {
            theta,
            grad: [0.0; 4],
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
        let delta = |c: &[f64; 4]| [0, 1, 2, 3].map(|k| c[k] - theta[k]);
        let value = |ll: f64, c: &[f64; 4]| {
            let d = delta(c);
            w * ll - 0.5 * dot(&d, &mat_vec(&h, &d))
        };
        let mut val = value(e.ll, &cur);
        for _ in 0..MAX_STEPS {
            let hd = mat_vec(&h, &delta(&cur));
            let gr = [0, 1, 2, 3].map(|k| w * e.g[k] - hd[k]);
            let free = free_at(&cur, &gr, fixed_loc);
            let mut m_obs = h;
            let mut m_fish = h;
            let (o, x) = (full(&e.obs), full(&e.exp));
            for i in 0..4 {
                for j in 0..4 {
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
            let dec = dot(&step, &gr);
            if dec.is_nan() || 0.5 * dec <= 1e-15 * w {
                break;
            }
            let mut t = 1.0;
            let mut moved = false;
            while t >= MIN_STEP {
                let mut c = [0, 1, 2, 3].map(|k| cur[k] + t * step[k]);
                clamp_shape(&mut c);
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
        let cd = mat_vec(&full(&st.curv), &delta(&cur));
        for ((gk, ek), ck) in st.grad.iter_mut().zip(&e.g).zip(&cd) {
            *gk += w * ek - ck;
        }
        for (c, v) in st.curv.iter_mut().zip(&e.obs) {
            *c += w * v;
        }
        st.theta = cur;
    }

    /// The fit's standing, read against the target's weight `w_sum`.
    pub(super) fn status(&self, w_sum: f64, fixed_loc: bool) -> TStatus {
        let Some(st) = &self.fit else {
            return TStatus::NotConverged;
        };
        let kish = self.w2 / w_sum;
        let h = full(&info_at(w_sum, &st.theta));
        let g = st.grad;
        let free = free_at(&st.theta, &g, fixed_loc);
        let Some(step) = chol_solve(&h, &g, free) else {
            return TStatus::NotConverged;
        };
        let dec = dot(&step, &g);
        let p = free.iter().filter(|f| **f).count();
        if dec.is_nan() || dec > kish * CHI2_95[p - 1] {
            return TStatus::NotConverged;
        }
        // The (η, λ) block of (W I)⁻¹ over every fitted coordinate.
        let all = [!fixed_loc, true, true, true];
        let (Some(ce), Some(cl)) = (
            chol_solve(&h, &[0.0, 0.0, 1.0, 0.0], all),
            chol_solve(&h, &[0.0, 0.0, 0.0, 1.0], all),
        ) else {
            return TStatus::NotConverged;
        };
        let (vee, vel, vll) = (ce[2] * kish, 0.5 * (ce[3] + cl[2]) * kish, cl[3] * kish);
        let det = vee * vll - vel * vel;
        let (eta, lam) = (st.theta[2], st.theta[3]);
        let wald = (vll * eta * eta - 2.0 * vel * eta * lam + vee * lam * lam) / det;
        if det > 0.0 && wald > WALD2_95 {
            TStatus::Identified
        } else {
            TStatus::ShapeUnidentified
        }
    }

    /// Whether the state's numbers are of a shape and kind a fit can hold:
    /// a restored file's check.
    pub(super) fn sound(&self) -> bool {
        self.fit.as_ref().is_none_or(|f| {
            (0.0..=ETA_MAX).contains(&f.theta[2]) && (-LAM_MAX..=LAM_MAX).contains(&f.theta[3])
        })
    }
}

#[cfg(test)]
mod tests;
