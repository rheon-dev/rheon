//! One row of the correlation stage: `R_t` read off `Q_t`, the row's
//! quasi-log-likelihood, its score and expected information in the
//! parameters, and the step of `Q` and its sensitivities to the next row.
//!
//! ```text
//! R      = D^{−1/2} Q D^{−1/2},   D = diag(Q)
//! l      = −½ (ln |R| + zᵀR⁻¹z)
//! ∂R_ij  = ∂Q_ij /√(Q_ii Q_jj) − ½ R_ij (∂Q_ii/Q_ii + ∂Q_jj/Q_jj)
//! ∂l/∂θ_i = ½ (uᵀ ∂R_i u − tr(R⁻¹ ∂R_i)),   u = R⁻¹z
//! I_ij   = ½ tr(R⁻¹ ∂R_i R⁻¹ ∂R_j)
//!
//! ∂Q'/∂a = z zᵀ − Q̄ + b ∂Q/∂a
//! ∂Q'/∂b = Q − Q̄ + b ∂Q/∂b
//! ∂Q'/∂g = n nᵀ − N̄ + b ∂Q/∂g
//! Q'     = (1 − a − b) Q̄ − g N̄ + a z zᵀ + g n nᵀ + b Q
//! ```
//!
//! with the targets held in the derivative, as the estimator holds them.

use std::cell::RefCell;

use super::mat::{chol_inverse, chol_log_det, chol_solve, cholesky};

/// The row's scratch, kept for the thread: every buffer sized by `k` and
/// `p`, so a row allocates nothing once they have grown.
#[derive(Default)]
pub(super) struct Work {
    pub r: Vec<f64>,
    pub l: Vec<f64>,
    pub rinv: Vec<f64>,
    pub u: Vec<f64>,
    pub col: Vec<f64>,
    pub dr: Vec<f64>,
    pub x: Vec<f64>,
    pub region: Vec<f64>,
    pub g: Vec<f64>,
    pub info: Vec<f64>,
}

thread_local! {
    pub(super) static WORK: RefCell<Work> = RefCell::new(Work::default());
}

/// `R` from `Q` into `w.r`, its factor into `w.l` and its inverse into
/// `w.rinv`; returns `ln |R|`, or `None` where `Q`'s diagonal is not
/// positive or `R` is not positive definite.
pub(super) fn read(q: &[f64], k: usize, w: &mut Work) -> Option<f64> {
    let kk = k * k;
    w.r.resize(kk, 0.0);
    w.l.resize(kk, 0.0);
    w.rinv.resize(kk, 0.0);
    w.col.resize(k, 0.0);
    w.u.resize(k, 0.0);
    for i in 0..k {
        let d = q[i * k + i];
        if !(d > 0.0 && d.is_finite()) {
            return None;
        }
        w.col[i] = 1.0 / d.sqrt();
    }
    for i in 0..k {
        for j in 0..k {
            w.r[i * k + j] = if i == j {
                1.0
            } else {
                q[i * k + j] * w.col[i] * w.col[j]
            };
        }
    }
    if !cholesky(&w.r, k, &mut w.l) {
        return None;
    }
    chol_inverse(&w.l, k, &mut w.rinv, &mut w.col);
    Some(chol_log_det(&w.l, k))
}

/// `zᵀR⁻¹z` at the `R` [`read`] left, with `u = R⁻¹z` into `w.u`.
pub(super) fn quad(z: &[f64], k: usize, w: &mut Work) -> f64 {
    w.u.copy_from_slice(&z[..k]);
    chol_solve(&w.l, k, &mut w.u);
    z.iter().zip(&w.u).map(|(a, b)| a * b).sum()
}

/// The row's score into `w.g` (`p`) and expected information into
/// `w.info` (`p × p`), at the `R` and `u` [`read`] and [`quad`] left, from
/// `Q` and its sensitivities `dq` (`p` matrices of `k × k`).
pub(super) fn score_info(q: &[f64], dq: &[f64], k: usize, p: usize, w: &mut Work) {
    let kk = k * k;
    w.dr.resize(p * kk, 0.0);
    w.x.resize(p * kk, 0.0);
    w.g.resize(p, 0.0);
    w.info.resize(p * p, 0.0);
    for i in 0..k {
        w.col[i] = 1.0 / q[i * k + i].sqrt();
    }
    for t in 0..p {
        let d = &dq[t * kk..(t + 1) * kk];
        let dr = &mut w.dr[t * kk..(t + 1) * kk];
        for i in 0..k {
            for j in 0..k {
                dr[i * k + j] = if i == j {
                    0.0
                } else {
                    d[i * k + j] * w.col[i] * w.col[j]
                        - 0.5
                            * w.r[i * k + j]
                            * (d[i * k + i] / q[i * k + i] + d[j * k + j] / q[j * k + j])
                };
            }
        }
        // X_t = R⁻¹ ∂R_t
        let x = &mut w.x[t * kk..(t + 1) * kk];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for m in 0..k {
                    s += w.rinv[i * k + m] * dr[m * k + j];
                }
                x[i * k + j] = s;
            }
        }
        let mut tr = 0.0;
        for i in 0..k {
            tr += x[i * k + i];
        }
        let mut uq = 0.0;
        for i in 0..k {
            let mut s = 0.0;
            for j in 0..k {
                s += dr[i * k + j] * w.u[j];
            }
            uq += w.u[i] * s;
        }
        w.g[t] = 0.5 * (uq - tr);
    }
    for s in 0..p {
        for t in 0..=s {
            let (xs, xt) = (&w.x[s * kk..(s + 1) * kk], &w.x[t * kk..(t + 1) * kk]);
            let mut acc = 0.0;
            for a in 0..k {
                for b in 0..k {
                    acc += xs[a * k + b] * xt[b * k + a];
                }
            }
            w.info[s * p + t] = 0.5 * acc;
            w.info[t * p + s] = 0.5 * acc;
        }
    }
}

/// Step `Q` (and `dq`, where it is not empty) past the row `z` at the
/// parameters `theta` and the targets `qbar` and `nbar` (`nbar` empty for
/// the symmetric model).
pub(super) fn advance(
    theta: &[f64],
    qbar: &[f64],
    nbar: &[f64],
    z: &[f64],
    k: usize,
    q: &mut [f64],
    dq: &mut [f64],
) {
    let kk = k * k;
    let (a, b) = (theta[0], theta[1]);
    let g = theta.get(2).copied().unwrap_or(0.0);
    let asym = !nbar.is_empty();
    let neg = |v: f64| if v < 0.0 { v } else { 0.0 };
    if !dq.is_empty() {
        for i in 0..k {
            for j in 0..k {
                let ij = i * k + j;
                dq[ij] = z[i] * z[j] - qbar[ij] + b * dq[ij];
                dq[kk + ij] = q[ij] - qbar[ij] + b * dq[kk + ij];
                if asym {
                    dq[2 * kk + ij] = neg(z[i]) * neg(z[j]) - nbar[ij] + b * dq[2 * kk + ij];
                }
            }
        }
    }
    let c0 = 1.0 - a - b;
    for i in 0..k {
        for j in 0..k {
            let ij = i * k + j;
            let mut v = c0 * qbar[ij] + a * (z[i] * z[j]) + b * q[ij];
            if asym {
                v += g * (neg(z[i]) * neg(z[j]) - nbar[ij]);
            }
            q[ij] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle;

    /// `l` against faer's Gaussian density (`crate::oracle`), whose
    /// determinant and quadratic form come from an eigendecomposition: the
    /// density of `z` under `R` less the constant.
    #[test]
    fn the_row_likelihood_is_the_gaussian_density_under_r() {
        let k = 3;
        let q = [2.0, 0.6, -0.3, 0.6, 1.5, 0.2, -0.3, 0.2, 0.8];
        let z = [0.7, -1.2, 0.4];
        let mut w = Work::default();
        let ld = read(&q, k, &mut w).unwrap();
        let l = -0.5 * (ld + quad(&z, k, &mut w));
        let want = oracle::gaussian_log_density(&w.r, &z)
            + 0.5 * k as f64 * (2.0 * std::f64::consts::PI).ln();
        assert!((l - want).abs() < 1e-13, "{l} {want}");
        // a Q whose diagonal is not positive, or whose R does not factor
        assert!(read(&[0.0, 0.0, 0.0, 1.0], 2, &mut w).is_none());
        assert!(read(&[1.0, 2.0, 2.0, 1.0], 2, &mut w).is_none());
    }

    /// The score and information by central differences of the row's
    /// likelihood in `θ` through one step of the recursion, from a `Q` and
    /// sensitivities that are themselves a few rows of the recursion.
    #[test]
    fn the_score_and_information_are_the_likelihoods_derivatives() {
        let k = 3;
        let qbar = [1.0, 0.4, 0.3, 0.4, 1.0, 0.5, 0.3, 0.5, 1.0];
        let nbar = [0.5, 0.25, 0.2, 0.25, 0.5, 0.3, 0.2, 0.3, 0.5];
        let zs = [
            [0.5, -1.0, 0.3],
            [-0.8, -0.2, 1.1],
            [1.4, 0.6, -0.9],
            [-0.3, -1.5, -0.4],
        ];
        let run = |theta: &[f64]| -> (Vec<f64>, Vec<f64>) {
            let p = theta.len();
            let mut q = qbar.to_vec();
            let mut dq = vec![0.0; p * k * k];
            for z in &zs[..3] {
                advance(
                    theta,
                    &qbar,
                    if p == 3 { &nbar } else { &[] },
                    z,
                    k,
                    &mut q,
                    &mut dq,
                );
            }
            (q, dq)
        };
        let lik = |theta: &[f64]| -> f64 {
            let (q, _) = run(theta);
            let mut w = Work::default();
            let ld = read(&q, k, &mut w).unwrap();
            -0.5 * (ld + quad(&zs[3], k, &mut w))
        };
        for theta in [vec![0.04, 0.9], vec![0.03, 0.92, 0.04]] {
            let p = theta.len();
            let (q, dq) = run(&theta);
            let mut w = Work::default();
            read(&q, k, &mut w).unwrap();
            quad(&zs[3], k, &mut w);
            score_info(&q, &dq, k, p, &mut w);
            for i in 0..p {
                let h = 1e-6;
                let (mut up, mut dn) = (theta.clone(), theta.clone());
                up[i] += h;
                dn[i] -= h;
                let fd = (lik(&up) - lik(&dn)) / (2.0 * h);
                assert!(
                    (w.g[i] - fd).abs() < 1e-7 * (1.0 + fd.abs()),
                    "{i}: {} {fd}",
                    w.g[i]
                );
            }
            // The information is the expectation of the score's outer
            // product under z ~ N(0, R): ½ tr(R⁻¹∂R_i R⁻¹∂R_j), against
            // the same trace from faer's inverse.
            let rinv = oracle::inverse(&w.r);
            let kk = k * k;
            for s in 0..p {
                for t in 0..p {
                    let mat = |v: usize| -> Vec<f64> {
                        let dr = &w.dr[v * kk..(v + 1) * kk];
                        (0..kk)
                            .map(|ij| {
                                let (i, j) = (ij / k, ij % k);
                                (0..k).map(|m| rinv[i * k + m] * dr[m * k + j]).sum()
                            })
                            .collect()
                    };
                    let (xs, xt) = (mat(s), mat(t));
                    let want: f64 = 0.5
                        * (0..k)
                            .flat_map(|a| (0..k).map(move |b| (a, b)))
                            .map(|(a, b)| xs[a * k + b] * xt[b * k + a])
                            .sum::<f64>();
                    assert!((w.info[s * p + t] - want).abs() < 1e-12 * (1.0 + want.abs()));
                }
            }
        }
    }

    /// The recursion longhand, entry by entry, for both models; at `g = 0`
    /// the asymmetric one is the symmetric one to the bit.
    #[test]
    fn the_recursion_is_the_definition() {
        let k = 2;
        let qbar = [1.0, 0.5, 0.5, 1.0];
        let nbar = [0.5, 0.3, 0.3, 0.5];
        let z = [-0.7, 1.3];
        let q0 = [1.1, 0.4, 0.4, 0.9];
        let (a, b, g) = (0.05_f64, 0.9_f64, 0.03_f64);
        let mut q = q0;
        advance(&[a, b, g], &qbar, &nbar, &z, k, &mut q, &mut []);
        let n = [z[0].min(0.0), z[1].min(0.0)];
        for i in 0..2 {
            for j in 0..2 {
                let want = (1.0 - a - b) * qbar[i * 2 + j]
                    + a * (z[i] * z[j])
                    + b * q0[i * 2 + j]
                    + g * (n[i] * n[j] - nbar[i * 2 + j]);
                assert!((q[i * 2 + j] - want).abs() < 1e-15);
            }
        }
        let (mut q1, mut q2) = (q0, q0);
        let (mut d1, mut d2) = (vec![0.0; 8], vec![0.0; 12]);
        advance(&[a, b], &qbar, &[], &z, k, &mut q1, &mut d1);
        advance(&[a, b, 0.0], &qbar, &nbar, &z, k, &mut q2, &mut d2);
        assert_eq!(q1, q2);
        assert_eq!(d1[..], d2[..8]);
    }
}
