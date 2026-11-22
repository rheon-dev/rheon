//! The batch start: the likelihood of the buffered rows for each form, its
//! admissible set, its starting grid, and the fit the optimizer makes of
//! them (`optim::maximize`).
//!
//! Each objective is the row-weighted log-likelihood of the buffer, the
//! filter started at the backcast, with its gradient and Gauss-Newton
//! matrix from the same sensitivity recursion the online step carries. The
//! Gaussian forms' matrix is the expected information `½ Σ w ∂ln s ∂ln sᵀ`,
//! the t forms' the outer product of the whole score (BHHH).

use super::dist::{self, LAM_LIM, NU_HI, NU_LO};
use super::filter::{Filter, figarch_weights};
use super::linalg::HalfSpaces;
use super::math;
use super::optim::{Eval, Stop, maximize};
use super::{GarchDist, GarchVol};

/// The stationarity margin: the persistence is held at most `1 − 10⁻⁴`.
pub(super) const MARGIN: f64 = 1e-4;
/// `ω`'s floor, as a fraction of the series' mean square.
pub(super) const OMEGA_FLOOR_FRAC: f64 = 1e-6;
/// FIGARCH's `d` is held at most `1 − 10⁻⁴`.
const D_MAX: f64 = 1.0 - 1e-4;
/// The rows `arch`'s backcast reads, and its decay.
const BACKCAST_ROWS: usize = 75;
const BACKCAST_DECAY: f64 = 0.94;
const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// `arch`'s backcast: the `0.94^i`-weighted mean of the first 75 squared
/// residuals (fewer where the buffer holds fewer), weights normalized.
pub(super) fn backcast(rows: &[[f64; 2]]) -> f64 {
    let (mut num, mut den, mut w) = (0.0, 0.0, 1.0);
    for r in rows.iter().take(BACKCAST_ROWS) {
        num += r[0] * r[0] * w;
        den += w;
        w *= BACKCAST_DECAY;
    }
    num / den
}

/// GARCH(1,1)'s set, `θ = (ω, α, β)`: `ω ≥ floor`, `α, β ≥ 0`,
/// `α + β ≤ 1 − 10⁻⁴`.
pub(super) fn garch_set(floor: f64) -> HalfSpaces {
    HalfSpaces {
        a: vec![
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, -1.0, -1.0,
        ],
        b: vec![floor, 0.0, 0.0, -(1.0 - MARGIN)],
        n: 3,
    }
}

/// GJR's set, `θ = (ω, α, γ, β)`, `arch`'s: `ω ≥ floor`, `α ≥ 0`,
/// `α + γ ≥ 0`, `β ≥ 0`, `α + γ/2 + β ≤ 1 − 10⁻⁴`.
pub(super) fn gjr_set(floor: f64) -> HalfSpaces {
    HalfSpaces {
        a: vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.0, -1.0, -0.5, -1.0,
        ],
        b: vec![floor, 0.0, 0.0, 0.0, -(1.0 - MARGIN)],
        n: 4,
    }
}

/// FIGARCH's set, `θ = (ω, φ, d, β)`, `arch`'s with `d ≤ 1 − 10⁻⁴`: `ω ≥
/// floor`, `φ ≥ 0`, `0 ≤ d ≤ 1 − 10⁻⁴`, `2φ + d ≤ 1`, `0 ≤ β ≤ φ + d`. They
/// keep every weight `λ_j ≥ 0`.
pub(super) fn figarch_set(floor: f64) -> HalfSpaces {
    HalfSpaces {
        a: vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, -1.0, 0.0, //
            0.0, -2.0, -1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, -1.0,
        ],
        b: vec![floor, 0.0, 0.0, -D_MAX, -1.0, 0.0, 0.0],
        n: 4,
    }
}

/// The variance-targeted GARCH(1,1)'s free `(α, β)`: `α, β ≥ 0`, `α + β ≤ 1
/// − 10⁻⁴`.
pub(super) fn ab_set() -> HalfSpaces {
    HalfSpaces {
        a: vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
        b: vec![0.0, 0.0, -(1.0 - MARGIN)],
        n: 2,
    }
}

/// EGARCH's box, the set its start's fit and every step are projected
/// onto, `θ = (ω, α, γ, β)`: `α ≥ |γ|` (the news term is never negative),
/// `α ≤ 1`, `0 ≤ β ≤ 1 − 10⁻⁴`; `ω` free. The rest of its guard, the
/// filter's running Lyapunov exponent, is [`super::guard`]'s.
pub(super) fn egarch_set() -> HalfSpaces {
    HalfSpaces {
        a: vec![
            0.0, 1.0, -1.0, 0.0, //
            0.0, 1.0, 1.0, 0.0, //
            0.0, -1.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, -1.0,
        ],
        b: vec![0.0, 0.0, -1.0, 0.0, -(1.0 - MARGIN)],
        n: 4,
    }
}

/// Whether EGARCH's start may fit `theta` on its rows, beyond the set: the
/// guard's part that reads the data, its filter contracting on the
/// standardized residuals the filter at `theta` gives the rows
/// ([`super::guard`]).
pub(super) fn egarch_start_admits(theta: &[f64], rows: &[[f64; 2]], bc: f64) -> bool {
    let mut f = Filter::new(GarchVol::Egarch, 4, 0, bc);
    let mut counts = vec![0.0; super::guard::BINS];
    let mut dl = [0.0; 4];
    for r in rows {
        let s = f.predict(GarchVol::Egarch, theta, &mut dl);
        counts[super::guard::bin(r[0] / s.sqrt())] += r[1];
        f.push(GarchVol::Egarch, r[0], s, &dl);
    }
    super::guard::admits(theta, &counts)
}

/// The set of a process's `θ`, its `ω` at least `floor` (none for EGARCH).
pub(super) fn set_of(vol: GarchVol, floor: f64) -> HalfSpaces {
    match vol {
        GarchVol::Gjr => gjr_set(floor),
        GarchVol::Egarch => egarch_set(),
        GarchVol::Figarch => figarch_set(floor),
        _ => garch_set(floor),
    }
}

/// The shape onto `ν ∈ [2.05, 300]`, `|λ| ≤ 0.99`.
pub(super) fn clip_shape(shape: &mut [f64]) {
    shape[0] = shape[0].clamp(NU_LO, NU_HI);
    if shape.len() > 1 {
        shape[1] = shape[1].clamp(-LAM_LIM, LAM_LIM);
    }
}

/// The buffer's weighted log-likelihood under a GARCH, GJR or EGARCH `θ`
/// (all `k` parameters), with the shape `shape` (empty: normal), its
/// gradient in `(θ, shape)` and its matrix.
fn likelihood(
    vol: GarchVol,
    dist: GarchDist,
    k: usize,
    rows: &[[f64; 2]],
    bc: f64,
    x: &[f64],
    grad: bool,
) -> Eval {
    let m = x.len() - k;
    let p = k + m;
    let (theta, shape) = x.split_at(k);
    let (nu, lam) = match m {
        0 => (0.0, 0.0),
        1 => (shape[0], 0.0),
        _ => (shape[0], shape[1]),
    };
    if m > 0 && !((NU_LO..=NU_HI).contains(&nu) && lam.abs() < 1.0) {
        return Eval::bad(p);
    }
    let skew = dist == GarchDist::Skewt;
    let mut filter = Filter::new(vol, k, 0, bc);
    let mut dl = [0.0; 4];
    let dl = &mut dl[..k];
    let mut e = Eval {
        f: 0.0,
        g: vec![0.0; p],
        h: vec![0.0; p * p],
    };
    let mut row = [0.0; 6];
    for r in rows {
        let (y, w) = (r[0], r[1]);
        let s = filter.predict(vol, theta, dl);
        if !(s > 0.0 && s.is_finite() && dl.iter().all(|d| d.is_finite())) {
            return Eval::bad(p);
        }
        if m == 0 {
            e.f -= 0.5 * w * (LN_2PI + math::ln(s) + y * y / s);
            if grad {
                let res = y * y / s - 1.0;
                for i in 0..k {
                    e.g[i] += 0.5 * w * res * dl[i];
                    for j in 0..k {
                        e.h[i * p + j] += 0.5 * w * dl[i] * dl[j];
                    }
                }
            }
        } else {
            let z = y / s.sqrt();
            e.f += w * (dist::log_density(z, nu, lam) - 0.5 * math::ln(s));
            if grad {
                let (wt, s_nu, s_lam) = dist::scores(z, nu, lam, skew);
                for i in 0..k {
                    row[i] = 0.5 * wt * dl[i];
                }
                row[k] = s_nu;
                if m > 1 {
                    row[k + 1] = s_lam;
                }
                for i in 0..p {
                    e.g[i] += w * row[i];
                    for j in 0..p {
                        e.h[i * p + j] += w * row[i] * row[j];
                    }
                }
            }
        }
        filter.push(vol, y, s, dl);
    }
    if !e.f.is_finite() {
        return Eval::bad(p);
    }
    e
}

/// The buffer's weighted Gaussian log-likelihood under FIGARCH's `θ`, its
/// weights read once and each row's lags past the buffer's start the
/// backcast.
fn figarch_likelihood(rows: &[[f64; 2]], bc: f64, m: usize, theta: &[f64], grad: bool) -> Eval {
    if !(theta[0] > 0.0 && (0.0..1.0).contains(&theta[3]) && (0.0..1.0).contains(&theta[2])) {
        return Eval::bad(4);
    }
    // weights, and the tail sums past each lag the buffer reaches
    let mut wts = vec![[0.0; 4]; m + 2];
    figarch_weights(theta, m, grad, |j, lam, lp, ld, lb| {
        wts[j] = [lam, lp, ld, lb];
    });
    let mut tail = vec![[0.0; 4]; m + 2];
    for j in (1..=m).rev() {
        for c in 0..4 {
            tail[j][c] = tail[j + 1][c] + wts[j][c];
        }
    }
    let one_b = 1.0 - theta[3];
    let w0 = theta[0] / one_b;
    let mut e = Eval {
        f: 0.0,
        g: vec![0.0; 4],
        h: vec![0.0; 16],
    };
    for (t, r) in rows.iter().enumerate() {
        let reach = t.min(m);
        let mut acc = [0.0; 4];
        for (j, wj) in wts.iter().enumerate().take(reach + 1).skip(1) {
            let x = rows[t - j][0] * rows[t - j][0];
            for c in 0..4 {
                acc[c] += wj[c] * x;
            }
        }
        for c in 0..4 {
            acc[c] += bc * tail[reach + 1][c];
        }
        let h = w0 + acc[0];
        if !(h > 0.0 && h.is_finite()) {
            return Eval::bad(4);
        }
        let (y2, w) = (r[0] * r[0], r[1]);
        e.f -= 0.5 * w * (LN_2PI + math::ln(h) + y2 / h);
        if grad {
            let dl = [
                1.0 / one_b / h,
                acc[1] / h,
                acc[2] / h,
                (acc[3] + w0 / one_b) / h,
            ];
            let res = y2 / h - 1.0;
            for i in 0..4 {
                e.g[i] += 0.5 * w * res * dl[i];
                for j in 0..4 {
                    e.h[i * 4 + j] += 0.5 * w * dl[i] * dl[j];
                }
            }
        }
    }
    if !e.f.is_finite() {
        return Eval::bad(4);
    }
    e
}

/// The variance-targeted likelihood in the free `φ`, `ω = v (1 − c·φ)`:
/// the chain rule takes the full gradient `g` to `g_φ − v c g_ω`.
fn targeted(
    vol: GarchVol,
    rows: &[[f64; 2]],
    bc: f64,
    v: f64,
    c: &[f64],
    phi: &[f64],
    grad: bool,
) -> Eval {
    let q = phi.len();
    let k = q + 1;
    let mut theta = vec![v * (1.0 - c.iter().zip(phi).map(|(c, p)| c * p).sum::<f64>())];
    theta.extend_from_slice(phi);
    let full = likelihood(vol, GarchDist::Normal, k, rows, bc, &theta, grad);
    if !full.f.is_finite() {
        return Eval::bad(q);
    }
    // J (k × q): ∂θ/∂φ, first row −v c, then the identity
    let jac = |i: usize, j: usize| {
        if i == 0 {
            -v * c[j]
        } else if i - 1 == j {
            1.0
        } else {
            0.0
        }
    };
    let mut e = Eval {
        f: full.f,
        g: vec![0.0; q],
        h: vec![0.0; q * q],
    };
    for a in 0..q {
        for i in 0..k {
            e.g[a] += jac(i, a) * full.g[i];
            for b in 0..q {
                for j in 0..k {
                    e.h[a * q + b] += jac(i, a) * full.h[i * k + j] * jac(j, b);
                }
            }
        }
    }
    e
}

/// GARCH-type starting grid at mean square `v`: `α` in `{0.02, 0.05, 0.1}`,
/// GJR's `γ` in `{0, 0.1}`, the persistence in `{0.8, 0.95, 0.98}`, `ω`
/// from the variance target.
fn garch_grid(gjr: bool, v: f64) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for a in [0.02, 0.05, 0.1] {
        for g in if gjr { &[0.0, 0.1][..] } else { &[0.0][..] } {
            for p in [0.8, 0.95, 0.98] {
                let b = p - a - 0.5 * g;
                let mut th = vec![v * (1.0 - p), a];
                if gjr {
                    th.push(*g);
                }
                th.push(b);
                out.push(th);
            }
        }
    }
    out
}

/// The start's fit: the full `θ` (with the shape for a t), and whether it
/// is one to step from -- the optimizer converged, and no constraint of
/// the process holds the fit on its bound.
pub(super) struct StartFit {
    pub theta: Vec<f64>,
    pub sound: bool,
}

/// Whether `x` sits on a face of `set`: a constraint's slack within `10⁻⁹`
/// of its row's size in the parameters' units.
fn on_bound(set: &HalfSpaces, x: &[f64], units: &[f64]) -> bool {
    (0..set.b.len()).any(|i| {
        let row = &set.a[i * set.n..(i + 1) * set.n];
        let slack = row.iter().zip(x).map(|(a, x)| a * x).sum::<f64>() - set.b[i];
        let scale: f64 = row.iter().zip(units).map(|(a, u)| a.abs() * u).sum();
        slack <= 1e-9 * scale
    })
}

/// The fit of the buffer for a form, the variance level at `v` (the
/// buffer's weighted mean square). `None` when no grid point has a finite
/// likelihood.
pub(super) fn fit(
    vol: GarchVol,
    dist: GarchDist,
    targeted: bool,
    trunc: usize,
    rows: &[[f64; 2]],
    bc: f64,
    v: f64,
) -> Option<StartFit> {
    let floor = OMEGA_FLOOR_FRAC * v;
    if targeted {
        // the variance-targeted GARCH(1,1): (α, β), ω = v (1 − α − β)
        let grid: Vec<Vec<f64>> = [0.01, 0.05, 0.1, 0.2]
            .iter()
            .flat_map(|&a| [0.5, 0.7, 0.9, 0.98].map(|p| vec![a, p - a]))
            .collect();
        let obj = |x: &[f64], g: bool| self::targeted(vol, rows, bc, v, &[1.0, 1.0], x, g);
        let set = ab_set();
        let proj = |x: &[f64]| set.project(x);
        let fit = maximize(&obj, &proj, &grid, &[1.0, 1.0])?;
        let (a, b) = (fit.theta[0], fit.theta[1]);
        return Some(StartFit {
            sound: fit.stop == Stop::Converged && !on_bound(&set, &fit.theta, &[1.0, 1.0]),
            theta: vec![v * (1.0 - a - b), a, b],
        });
    }
    match vol {
        GarchVol::Egarch => {
            let set = egarch_set();
            let lv = math::ln(v);
            let mut grid = Vec::new();
            for a in [0.05, 0.1, 0.2] {
                for g in [-0.1, 0.0] {
                    for b in [0.8, 0.95, 0.98] {
                        grid.push(vec![lv * (1.0 - b), a, g, b]);
                    }
                }
            }
            let obj = |x: &[f64], g: bool| {
                if !egarch_start_admits(x, rows, bc) {
                    return Eval::bad(4);
                }
                likelihood(vol, dist, 4, rows, bc, x, g)
            };
            let proj = |x: &[f64]| set.project(x);
            let units = [1.0; 4];
            let fit = maximize(&obj, &proj, &grid, &units)?;
            Some(StartFit {
                sound: fit.stop == Stop::Converged && !on_bound(&set, &fit.theta, &units),
                theta: fit.theta,
            })
        }
        GarchVol::Figarch => {
            let set = figarch_set(floor);
            let mut grid = Vec::new();
            for d in [0.2, 0.5, 0.7] {
                for pr in [0.2, 0.5, 0.8] {
                    let phi = (1.0 - d) / 2.0 * pr;
                    for br in [0.1, 0.5, 0.9] {
                        let beta = (d + phi) * br;
                        let mut sum = 0.0;
                        figarch_weights(&[0.0, phi, d, beta], trunc, false, |_, l, _, _, _| {
                            sum += l
                        });
                        grid.push(vec![(1.0 - beta) * v * (1.0 - sum), phi, d, beta]);
                    }
                }
            }
            let obj = |x: &[f64], g: bool| figarch_likelihood(rows, bc, trunc, x, g);
            let proj = |x: &[f64]| set.project(x);
            let units = [v, 1.0, 1.0, 1.0];
            let fit = maximize(&obj, &proj, &grid, &units)?;
            Some(StartFit {
                sound: fit.stop == Stop::Converged && !on_bound(&set, &fit.theta, &units),
                theta: fit.theta,
            })
        }
        _ => {
            // GARCH and GJR, normal or with a t: every parameter free
            let gjr = vol == GarchVol::Gjr;
            let k = if gjr { 4 } else { 3 };
            let set = set_of(vol, floor);
            let shapes: &[[f64; 2]] = match dist {
                GarchDist::Normal => &[],
                GarchDist::T => &[[5.0, 0.0], [10.0, 0.0]],
                GarchDist::Skewt => &[[5.0, 0.0], [10.0, 0.0], [7.0, -0.2]],
            };
            let m = dist.shape_len();
            let mut grid = Vec::new();
            for th in garch_grid(gjr, v) {
                if m == 0 {
                    grid.push(th);
                } else {
                    for s in shapes {
                        let mut x = th.clone();
                        x.extend_from_slice(&s[..m]);
                        grid.push(x);
                    }
                }
            }
            let obj = |x: &[f64], g: bool| likelihood(vol, dist, k, rows, bc, x, g);
            let proj = |x: &[f64]| {
                let mut out = set.project(&x[..k]);
                let mut shape = x[k..].to_vec();
                if m > 0 {
                    clip_shape(&mut shape);
                }
                out.extend(shape);
                out
            };
            let mut units = vec![1.0; k + m];
            units[0] = v;
            let fit = maximize(&obj, &proj, &grid, &units)?;
            Some(StartFit {
                sound: fit.stop == Stop::Converged && !on_bound(&set, &fit.theta[..k], &units[..k]),
                theta: fit.theta,
            })
        }
    }
}
