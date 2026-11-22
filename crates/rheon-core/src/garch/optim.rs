//! The batch start's optimizer: projected Newton with Armijo backtracking on
//! the buffered rows, held by the model itself (no library's optimizer).
//!
//! ```text
//! f, g, H = objective(θ)        value, gradient, Gauss-Newton (or BHHH) matrix
//! M       = −(central difference of the analytic g), symmetrized: the observed
//!           curvature; H where it is not positive definite or a probe is not finite
//! d       = (M + 10⁻¹⁰ diag M)⁻¹ g
//! θ'      = Proj(θ + s d),  s = 1, ½, ¼, … (at most 30 halvings),
//!           the first with f(θ') ≥ f + 10⁻⁴ gᵀ(θ' − θ)
//! ```
//!
//! When no step length passes along `d`, or the projection takes the step
//! back to `θ` itself, two more directions are tried in turn: Newton on the coordinates the projection does not block (one is
//! blocked when a small move along `d` in it alone is cut by more than
//! half), then the gradient scaled by `1/diag M`. It stops after the
//! iteration that moves every parameter by at most `10⁻⁸` of its unit and
//! `f` by at most `10⁻¹⁰ (1 + |f|)`, when no direction ascends, or at 50
//! iterations. The start is the best point of a small fixed grid. The unit
//! of the variance level is the buffer's mean square, of every other
//! parameter 1, so the differences and the stopping rule scale with the
//! series.

use super::linalg::{cholesky, lu_solve, ridged};

pub(super) const MAX_ITER: u32 = 50;
const MAX_HALVINGS: u32 = 30;
const ARMIJO: f64 = 1e-4;
const TOL_X: f64 = 1e-8;
const TOL_F: f64 = 1e-10;
const RIDGE: f64 = 1e-10;

/// An objective's value, gradient and Gauss-Newton matrix (row-major).
pub(super) struct Eval {
    pub f: f64,
    pub g: Vec<f64>,
    pub h: Vec<f64>,
}

impl Eval {
    pub fn bad(p: usize) -> Self {
        Self {
            f: f64::NEG_INFINITY,
            g: vec![0.0; p],
            h: vec![0.0; p * p],
        }
    }
}

/// How a fit stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Stop {
    Converged,
    NoAscent,
    IterationCap,
}

pub(super) struct Fit {
    pub theta: Vec<f64>,
    /// How many iterations ran, and why they stopped: what the tests read
    /// of a fit.
    #[allow(dead_code)]
    pub iterations: u32,
    #[cfg_attr(not(test), allow(dead_code))]
    pub stop: Stop,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Minus the Jacobian of the analytic gradient by central differences,
/// symmetrized; `None` where a probe is not finite or it is not positive
/// definite.
fn curvature(obj: &dyn Fn(&[f64], bool) -> Eval, theta: &[f64], units: &[f64]) -> Option<Vec<f64>> {
    let p = theta.len();
    let mut m = vec![0.0; p * p];
    for j in 0..p {
        let h = 1e-5 * theta[j].abs().max(1e-2 * units[j]);
        let mut up = theta.to_vec();
        let mut dn = theta.to_vec();
        up[j] += h;
        dn[j] -= h;
        let (ep, em) = (obj(&up, true), obj(&dn, true));
        if !(ep.f.is_finite() && em.f.is_finite()) {
            return None;
        }
        for i in 0..p {
            m[i * p + j] = -(ep.g[i] - em.g[i]) / (2.0 * h);
        }
    }
    for i in 0..p {
        for j in 0..i {
            let s = 0.5 * (m[i * p + j] + m[j * p + i]);
            m[i * p + j] = s;
            m[j * p + i] = s;
        }
    }
    cholesky(&m, p).map(|_| m)
}

/// The directions tried in turn: Newton, Newton on the coordinates the
/// projection does not block, the diagonally scaled gradient.
fn directions(
    d: Vec<f64>,
    g: &[f64],
    m: &[f64],
    theta: &[f64],
    project: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Vec<Vec<f64>> {
    let p = d.len();
    let mut out = Vec::with_capacity(3);
    let free: Vec<usize> = (0..p)
        .filter(|&j| {
            let mut e = theta.to_vec();
            let step = 1e-6 * d[j];
            e[j] += step;
            d[j] != 0.0 && (project(&e)[j] - theta[j]).abs() >= 0.5 * step.abs()
        })
        .collect();
    out.push(d);
    if !free.is_empty() && free.len() < p {
        let q = free.len();
        let mut sub = vec![0.0; q * q];
        for (a, &i) in free.iter().enumerate() {
            for (b, &j) in free.iter().enumerate() {
                sub[a * q + b] = m[i * p + j];
            }
        }
        let gs: Vec<f64> = free.iter().map(|&i| g[i]).collect();
        if let Some(x) = lu_solve(&ridged(&sub, q, RIDGE), q, &gs) {
            let mut red = vec![0.0; p];
            for (a, &i) in free.iter().enumerate() {
                red[i] = x[a];
            }
            out.push(red);
        }
    }
    out.push((0..p).map(|j| g[j] / m[j * p + j].max(1e-300)).collect());
    out
}

/// Maximize `obj` over the set `project` maps onto, from the best point of
/// `grid` (each projected first). `None` when no grid point has a finite
/// value.
pub(super) fn maximize(
    obj: &dyn Fn(&[f64], bool) -> Eval,
    project: &dyn Fn(&[f64]) -> Vec<f64>,
    grid: &[Vec<f64>],
    units: &[f64],
) -> Option<Fit> {
    let mut best: Option<(Vec<f64>, f64)> = None;
    for point in grid {
        let th = project(point);
        let f = obj(&th, false).f;
        if f.is_finite() && best.as_ref().is_none_or(|(_, b)| f > *b) {
            best = Some((th, f));
        }
    }
    let mut theta = best?.0;
    let p = theta.len();
    let mut e = obj(&theta, true);
    let mut stop = Stop::IterationCap;
    let mut it = 0;
    while it < MAX_ITER {
        it += 1;
        let m = curvature(obj, &theta, units).unwrap_or_else(|| e.h.clone());
        let Some(d) = lu_solve(&ridged(&m, p, RIDGE), p, &e.g) else {
            stop = Stop::NoAscent;
            break;
        };
        let mut accepted = None;
        'dirs: for dir in directions(d, &e.g, &m, &theta, project) {
            let mut s = 1.0;
            for _ in 0..=MAX_HALVINGS {
                let trial: Vec<f64> = theta.iter().zip(&dir).map(|(t, d)| t + s * d).collect();
                let cand = project(&trial);
                // A step the projection takes back to where it started is no
                // ascent: the next direction is tried.
                if cand == theta {
                    break;
                }
                let fc = obj(&cand, false).f;
                let delta: Vec<f64> = cand.iter().zip(&theta).map(|(c, t)| c - t).collect();
                if fc.is_finite() && fc >= e.f + ARMIJO * dot(&e.g, &delta) {
                    accepted = Some((cand, fc));
                    break 'dirs;
                }
                s *= 0.5;
            }
        }
        let Some((cand, fc)) = accepted else {
            stop = Stop::NoAscent;
            break;
        };
        let small = cand
            .iter()
            .zip(&theta)
            .zip(units)
            .all(|((c, t), u)| (c - t).abs() <= TOL_X * u);
        let df = (fc - e.f).abs();
        theta = cand;
        e = obj(&theta, true);
        if small && df <= TOL_F * (1.0 + e.f.abs()) {
            stop = Stop::Converged;
            break;
        }
    }
    Some(Fit {
        theta,
        iterations: it,
        stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(a: [f64; 4], c: [f64; 2]) -> impl Fn(&[f64], bool) -> Eval {
        move |x: &[f64], _grad: bool| {
            let d = [x[0] - c[0], x[1] - c[1]];
            let ad = [a[0] * d[0] + a[1] * d[1], a[2] * d[0] + a[3] * d[1]];
            Eval {
                f: -0.5 * (d[0] * ad[0] + d[1] * ad[1]),
                g: vec![-ad[0], -ad[1]],
                h: a.to_vec(),
            }
        }
    }

    /// A concave quadratic: inside the set the fit is its maximum; with the
    /// maximum outside a bound that couples to nothing, the face's nearest
    /// point. (Where the bound couples to a coordinate the Newton step would
    /// also move, the projected step can stall short of the face's maximum,
    /// as projected Newton does; the reduced and the scaled-gradient
    /// directions take over only where no step ascends.)
    #[test]
    fn a_quadratic_is_maximized_in_its_set() {
        let obj = quadratic([2.0, 0.5, 0.5, 1.0], [0.3, 0.5]);
        let project = |x: &[f64]| vec![x[0].min(1.0), x[1]];
        let fit = maximize(&obj, &project, &[vec![0.0, 0.0]], &[1.0, 1.0]).unwrap();
        assert_eq!(fit.stop, Stop::Converged);
        assert!((fit.theta[0] - 0.3).abs() < 1e-12 && (fit.theta[1] - 0.5).abs() < 1e-12);
        let obj = quadratic([2.0, 0.0, 0.0, 1.0], [2.0, 0.5]);
        let fit = maximize(&obj, &project, &[vec![0.0, 0.0]], &[1.0, 1.0]).unwrap();
        assert_ne!(fit.stop, Stop::IterationCap);
        assert!((fit.theta[0] - 1.0).abs() < 1e-12 && (fit.theta[1] - 0.5).abs() < 1e-12);
        // the best grid point is the start; no finite one, no fit
        let bad = |_: &[f64], _: bool| Eval::bad(2);
        assert!(maximize(&bad, &project, &[vec![0.0, 0.0]], &[1.0, 1.0]).is_none());
    }
}
