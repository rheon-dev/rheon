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
//! stops after the iteration that moves no parameter by more than `10⁻¹⁰`
//! or `f` by more than `10⁻¹² (1 + |f|)`, when no step ascends, or at 100
//! iterations. The fit is *sound* where it stopped on the first rule, or
//! where no step ascended from a point the projected Newton step moves by
//! at most `10⁻⁶`: an optimum on a face of the set, where no feasible
//! step ascends.

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

/// A start's fit and how it stopped.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Fit {
    pub theta: Vec<f64>,
    pub sound: bool,
    #[cfg_attr(not(test), allow(dead_code))]
    pub iterations: u32,
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
    let mut sound = false;
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
            // No step ascends: sound where the projected Newton step
            // barely moves, an optimum on a face of the set.
            let trial: Vec<f64> = theta.iter().zip(&d).map(|(t, d)| t + d).collect();
            let full = project(&trial, &h, &c, PERSISTENCE_CAP);
            sound = full.iter().zip(&theta).all(|(a, b)| (a - b).abs() <= 1e-6);
            break;
        };
        let small = cand.iter().zip(&theta).all(|(a, b)| (a - b).abs() <= 1e-10);
        let flat = (fc - f).abs() <= 1e-12 * (1.0 + f.abs());
        theta = cand;
        if small || flat {
            sound = true;
            break;
        }
    }
    Some(Fit {
        theta,
        sound,
        iterations: it,
    })
}
