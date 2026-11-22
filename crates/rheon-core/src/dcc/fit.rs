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

/// A start's fit and the two checks of it (the module docs).
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
    let (gain, lr, definite) = match objective(rows, &theta, true, work) {
        Some((f, g, h)) => {
            let gain = projected_gain(&theta, &g, &h, &c);
            let f0 = objective(rows, &vec![0.0; p], false, work).map_or(f64::NAN, |e| e.0);
            let mut l = vec![0.0; p * p];
            let definite = super::mat::cholesky(&h, p, &mut l);
            (gain * scale, 2.0 * (f - f0) * scale, definite)
        }
        None => (f64::INFINITY, f64::NAN, false),
    };
    Some(Fit {
        theta,
        gain: if gain.is_nan() { f64::INFINITY } else { gain },
        lr,
        definite,
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
