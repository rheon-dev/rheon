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
//! back to `θ` itself, two more directions are tried in turn: Newton on the
//! coordinates the projection does not block (one is blocked when a small
//! move along `d` in it alone is cut by more than half), then the gradient
//! scaled by `1/diag M`. The start is the best point of a small fixed grid.
//! The unit of the variance level is the buffer's mean square, of every
//! other parameter 1, so the differences and the stopping rule scale with
//! the series.
//!
//! **It runs to its rounding.** Near the optimum a step changes `f` by less
//! than `f`'s own rounding (its terms are far larger than their sum), so a
//! comparison of `f` there is decided by the last bits of `ln` and `exp`,
//! and a stop at a fixed step size leaves the fit wherever the iteration
//! happened to be. Two rules take those decisions away from the rounding:
//!
//! - **a change too small for `f` to show is read from the slopes.** Where
//!   the quadratic model's gain `gᵀΔ − ½ ΔᵀMΔ` of a step `Δ` is under
//!   `10⁻¹⁰ (1 + |f|)`, the step stands when the trapezoid rule on the
//!   slopes at its two ends, exact for a quadratic, says `f` did not fall:
//!   `g(θ)ᵀΔ + g(θ')ᵀΔ ≥ 0`.
//! - **it stops at the rounding floor:** once the full Newton step, or a
//!   step taken, moves every parameter by at most `4ε (1 + max |θ|)` in
//!   their units; or a step taken moves them by at most `10⁻¹¹ (1 + max
//!   |θ|)` and no less than the step before (an iteration that has stopped
//!   contracting is at its rounding); or when no direction ascends, or at
//!   50 iterations.
//!
//! **The fit reports its Newton decrement**, `½ gᵀ M⁻¹ g` at the point it
//! stopped: the gain a full Newton step from there predicts, in the
//! objective's own units. At an optimum it is the rounding's, at most
//! `10⁻¹²` nats on 2,000 rows; whether a fit is one to step from is read
//! from it ([`super::fit`]), never from how the iteration stopped.

use super::linalg::{cholesky, lu_solve, ridged};

pub(super) const MAX_ITER: u32 = 50;
const MAX_HALVINGS: u32 = 30;
const ARMIJO: f64 = 1e-4;
const RIDGE: f64 = 1e-10;
/// How small a gain, relative to `1 + |f|`, the line search reads from the
/// slopes rather than from `f` itself.
const UNSEEN: f64 = 1e-10;
/// A step under this, relative to `1 + max |θ|` in the parameters' units,
/// that is no smaller than the step before ends the fit.
const STALLED: f64 = 1e-11;

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
    /// The objective at `theta`.
    pub f: f64,
    /// The Newton decrement `½ gᵀ M⁻¹ g` at `theta`, in the objective's
    /// units: infinite where `M` gives no step.
    pub decrement: f64,
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
    let mut last = f64::INFINITY;
    while it < MAX_ITER {
        it += 1;
        let m = curvature(obj, &theta, units).unwrap_or_else(|| e.h.clone());
        let Some(d) = lu_solve(&ridged(&m, p, RIDGE), p, &e.g) else {
            stop = Stop::NoAscent;
            break;
        };
        let scaled = |v: f64, u: &f64| (v / u).abs();
        let size = theta
            .iter()
            .zip(units)
            .fold(0.0_f64, |a, (t, u)| a.max(scaled(*t, u)));
        let floor = 4.0 * f64::EPSILON * (1.0 + size);
        // a full Newton step at the rounding: nothing left to take
        if d.iter().zip(units).all(|(d, u)| scaled(*d, u) <= floor) {
            stop = Stop::Converged;
            break;
        }
        let unseen = UNSEEN * (1.0 + e.f.abs());
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
                let delta: Vec<f64> = cand.iter().zip(&theta).map(|(c, t)| c - t).collect();
                let slope = dot(&e.g, &delta);
                if slope - 0.5 * quad(&m, &delta) > unseen {
                    let fc = obj(&cand, false).f;
                    if fc.is_finite() && fc >= e.f + ARMIJO * slope {
                        accepted = Some((cand, None));
                        break 'dirs;
                    }
                } else {
                    // a gain under f's rounding: the slopes at both ends
                    let ec = obj(&cand, true);
                    if ec.f.is_finite() && slope + dot(&ec.g, &delta) >= 0.0 {
                        accepted = Some((cand, Some(ec)));
                        break 'dirs;
                    }
                }
                s *= 0.5;
            }
        }
        let Some((cand, at)) = accepted else {
            stop = Stop::NoAscent;
            break;
        };
        let step = cand
            .iter()
            .zip(&theta)
            .zip(units)
            .fold(0.0_f64, |a, ((c, t), u)| a.max(scaled(c - t, u)));
        theta = cand;
        e = at.unwrap_or_else(|| obj(&theta, true));
        if step <= floor || (step <= STALLED * (1.0 + size) && step >= last) {
            stop = Stop::Converged;
            break;
        }
        last = step;
    }
    let decrement = decrement(obj, &theta, &e, units);
    Some(Fit {
        theta,
        iterations: it,
        stop,
        f: e.f,
        decrement,
    })
}

/// `xᵀ M x` for a row-major `M`.
fn quad(m: &[f64], x: &[f64]) -> f64 {
    let p = x.len();
    (0..p)
        .map(|i| x[i] * (0..p).map(|j| m[i * p + j] * x[j]).sum::<f64>())
        .sum()
}

/// The Newton decrement `½ gᵀ M⁻¹ g` at `theta`, `e` the objective there:
/// the gain a full Newton step predicts, with `M` the curvature the
/// iteration reads (the Gauss-Newton matrix where the observed one is not
/// positive definite). Infinite where neither gives a step, or `f` is not a
/// number.
fn decrement(obj: &dyn Fn(&[f64], bool) -> Eval, theta: &[f64], e: &Eval, units: &[f64]) -> f64 {
    if !e.f.is_finite() {
        return f64::INFINITY;
    }
    let p = theta.len();
    let m = curvature(obj, theta, units).unwrap_or_else(|| e.h.clone());
    lu_solve(&ridged(&m, p, RIDGE), p, &e.g)
        .map(|d| 0.5 * dot(&e.g, &d))
        .filter(|v| v.is_finite())
        .map_or(f64::INFINITY, f64::abs)
}

/// The Newton decrement inside the set: the gain the Newton step predicts
/// once it is projected onto the set in the curvature's own metric, `gᵀd −
/// ½ dᵀ M d` with `d = P_M(θ + M⁻¹ g) − θ` (`in_metric` the projection, `M`
/// as [`decrement`] reads it). Where the step stays inside it is the
/// decrement; at an optimum on a face of the set, where the gradient points
/// out of it, it is the rounding's, as an interior optimum's decrement is,
/// though the decrement there is not. Never more than the decrement: `θ`
/// is in the set, and `P_M` is the point of the set nearest the step.
/// Infinite where no step is.
pub(super) fn decrement_in(
    obj: &dyn Fn(&[f64], bool) -> Eval,
    theta: &[f64],
    units: &[f64],
    in_metric: &dyn Fn(&[f64], &[f64]) -> Vec<f64>,
) -> f64 {
    let e = obj(theta, true);
    if !e.f.is_finite() {
        return f64::INFINITY;
    }
    let p = theta.len();
    let m = ridged(
        &curvature(obj, theta, units).unwrap_or_else(|| e.h.clone()),
        p,
        RIDGE,
    );
    let Some(s) = lu_solve(&m, p, &e.g) else {
        return f64::INFINITY;
    };
    let prop: Vec<f64> = theta.iter().zip(&s).map(|(t, s)| t + s).collect();
    let d: Vec<f64> = in_metric(&prop, &m)
        .iter()
        .zip(theta)
        .map(|(y, t)| y - t)
        .collect();
    let gain = dot(&e.g, &d) - 0.5 * quad(&m, &d);
    if gain.is_finite() {
        gain.abs()
    } else {
        f64::INFINITY
    }
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

    /// The decrement is the gain a full Newton step predicts, `½ gᵀ M⁻¹ g`:
    /// on a quadratic, exactly the climb left to its maximum, `½ cᵀ A c`
    /// from 0, and at the maximum the fit reports none.
    #[test]
    fn the_decrement_is_the_newton_steps_predicted_gain() {
        let (a, c) = ([2.0, 0.5, 0.5, 1.0], [0.3, 0.5]);
        let obj = quadratic(a, c);
        let at = [0.0, 0.0];
        let want = 0.5 * (c[0] * (a[0] * c[0] + a[1] * c[1]) + c[1] * (a[2] * c[0] + a[3] * c[1]));
        let got = decrement(&obj, &at, &obj(&at, true), &[1.0, 1.0]);
        // within the 1e-10 ridge on M's diagonal (measured 2.5e-10)
        assert!((got - want).abs() < 1e-9 * want, "{got} against {want}");
        let free = |x: &[f64]| x.to_vec();
        let fit = maximize(&obj, &free, &[at.to_vec()], &[1.0, 1.0]).unwrap();
        assert!(fit.decrement < 1e-24, "{}", fit.decrement);
        assert!(fit.f.abs() < 1e-24, "{}", fit.f);
        // a probe that is not finite, and no other matrix: no step, an
        // infinite decrement; a value that is not a number, the same
        let flat = |_: &[f64], _: bool| Eval {
            f: 1.0,
            g: vec![1.0, 0.0],
            h: vec![0.0; 4],
        };
        assert_eq!(
            decrement(&flat, &at, &flat(&at, true), &[1.0, 1.0]),
            f64::INFINITY
        );
        assert_eq!(
            decrement(&obj, &at, &Eval::bad(2), &[1.0, 1.0]),
            f64::INFINITY
        );
    }

    /// The decrement inside the set: on a concave quadratic whose maximum
    /// is past a bound, at the face's nearest point it is the rounding's
    /// while the decrement is the climb to the maximum outside; short of
    /// that point along the face it is the climb along the face; inside the
    /// set, with the step inside too, it is the decrement.
    #[test]
    fn the_decrement_inside_the_set_reads_an_optimum_on_a_face() {
        let obj = quadratic([2.0, 0.0, 0.0, 1.0], [2.0, 0.5]);
        let units = [1.0, 1.0];
        // `M` is diagonal: the Euclidean clip is the projection in its metric
        let face = |x: &[f64], _: &[f64]| vec![x[0].min(1.0), x[1]];
        let free = |x: &[f64], _: &[f64]| x.to_vec();
        let at = [1.0, 0.5];
        let outside = decrement(&obj, &at, &obj(&at, true), &units);
        assert!((outside - 1.0).abs() < 1e-9, "{outside}");
        let inside = decrement_in(&obj, &at, &units, &face);
        assert!(inside < 1e-9, "{inside}");
        let short = decrement_in(&obj, &[1.0, 0.0], &units, &face);
        assert!((short - 0.125).abs() < 1e-9, "{short}");
        let at = [0.0, 0.0];
        let whole = decrement(&obj, &at, &obj(&at, true), &units);
        let free_in = decrement_in(&obj, &at, &units, &free);
        assert!((free_in - whole).abs() < 1e-9 * whole, "{free_in} {whole}");
        // no finite value, no step
        let bad = |_: &[f64], _: bool| Eval::bad(2);
        assert_eq!(decrement_in(&bad, &at, &units, &free), f64::INFINITY);
    }

    /// On an objective whose value is far larger than the climb left (here
    /// 1e6 against 1e-7), every step's gain is under the `1e-10 (1 + |f|)`
    /// the line search reads `f` to: the slopes at its two ends decide it,
    /// and the fit lands on the maximum to the parameters' rounding.
    #[test]
    fn a_gain_under_the_rounding_is_read_from_the_slopes() {
        let base = quadratic([2.0, 0.5, 0.5, 1.0], [0.3, 0.5]);
        let obj = move |x: &[f64], g: bool| {
            let mut e = base(x, g);
            // the curvature is 1e-6 of the offset's scale: steps of 1e-3
            // gain about 1e-12, under the 1e-10 relative rounding read
            e.f = 1e6 + 1e-6 * e.f;
            e.g.iter_mut().for_each(|v| *v *= 1e-6);
            e.h.iter_mut().for_each(|v| *v *= 1e-6);
            e
        };
        let free = |x: &[f64]| x.to_vec();
        let fit = maximize(&obj, &free, &[vec![0.0, 0.0]], &[1.0, 1.0]).unwrap();
        assert_eq!(fit.stop, Stop::Converged);
        assert!((fit.theta[0] - 0.3).abs() < 1e-12 && (fit.theta[1] - 0.5).abs() < 1e-12);
    }
}
