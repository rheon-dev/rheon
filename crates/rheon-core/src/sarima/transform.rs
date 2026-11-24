//! The stationarity and invertibility transforms of a lag polynomial, the
//! ones statsmodels' `SARIMAX` optimizes through (Jones 1980; Monahan
//! 1984), with their Jacobians.
//!
//! An unconstrained `u` of length `n` is taken to partial autocorrelations
//! `r_k = u_k / sqrt(1 + u_k^2)`, each inside `(-1, 1)`, and those to the
//! coefficients of a polynomial whose roots lie outside the unit circle by
//! the Durbin-Levinson recursion
//!
//! ```text
//! y[k][k] = r_k,   y[k][i] = y[k-1][i] + r_k y[k-1][k-i-1]   (i < k)
//! ```
//!
//! An autoregressive factor `1 - phi_1 B - ...` takes `phi = -y[n-1]`
//! (`constrain_stationary_univariate`); a moving-average factor `1 +
//! theta_1 B + ...` takes `theta = y[n-1]`, the same map on `-theta`, so
//! its roots too lie outside the circle and it is invertible. Every
//! stationary (invertible) polynomial has exactly one `u`.

/// The coefficients of `u`'s polynomial, `-y[n-1]` (`ar`) or `y[n-1]`, and
/// their Jacobian `d coef_i / d u_m`, row-major `n x n`.
pub(crate) fn constrain(u: &[f64], ar: bool) -> (Vec<f64>, Vec<f64>) {
    let n = u.len();
    let sign = if ar { -1.0 } else { 1.0 };
    // `y` and `dy[i][m] = d y_i / d u_m` of the stage before.
    let mut y = vec![0.0; n];
    let mut dy = vec![0.0; n * n];
    let mut ny = vec![0.0; n];
    let mut ndy = vec![0.0; n * n];
    for k in 0..n {
        let s = (1.0 + u[k] * u[k]).sqrt();
        let r = u[k] / s;
        let dr = 1.0 / (s * s * s);
        for i in 0..k {
            let j = k - i - 1;
            ny[i] = y[i] + r * y[j];
            for m in 0..n {
                ndy[i * n + m] = dy[i * n + m] + r * dy[j * n + m];
            }
            ndy[i * n + k] += dr * y[j];
        }
        ny[k] = r;
        for m in 0..n {
            ndy[k * n + m] = 0.0;
        }
        ndy[k * n + k] = dr;
        y[..=k].copy_from_slice(&ny[..=k]);
        dy[..(k + 1) * n].copy_from_slice(&ndy[..(k + 1) * n]);
    }
    for v in &mut y {
        *v *= sign;
    }
    for v in &mut dy {
        *v *= sign;
    }
    (y, dy)
}

/// The partial autocorrelations of `u`, `u_k / sqrt(1 + u_k^2)`.
pub(crate) fn partials(u: &[f64]) -> impl Iterator<Item = f64> + '_ {
    u.iter().map(|v| v / (1.0 + v * v).sqrt())
}

/// The `u` of a polynomial's coefficients: the step-down recursion run
/// back to the partial autocorrelations, then `u = r / sqrt(1 - r^2)`
/// (statsmodels' `unconstrain_stationary_univariate`). `None` where the
/// polynomial is not stationary (not invertible): a partial
/// autocorrelation is not inside `(-1, 1)`.
#[cfg(test)]
pub(crate) fn unconstrain(coef: &[f64], ar: bool) -> Option<Vec<f64>> {
    let n = coef.len();
    let sign = if ar { -1.0 } else { 1.0 };
    let mut y: Vec<f64> = coef.iter().map(|c| sign * c).collect();
    let mut r = vec![0.0; n];
    let mut next = Vec::with_capacity(n);
    for k in (0..n).rev() {
        let rk = y[k];
        let inside = rk.abs() < 1.0;
        if !inside {
            return None;
        }
        r[k] = rk;
        let scale = 1.0 - rk * rk;
        next.clear();
        next.extend((0..k).map(|i| (y[i] - rk * y[k - i - 1]) / scale));
        y.truncate(k);
        y.copy_from_slice(&next);
    }
    Some(r.iter().map(|r| r / (1.0 - r * r).sqrt()).collect())
}
