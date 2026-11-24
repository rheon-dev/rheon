//! The lag polynomials of a [`super::Sarima`], the stationarity test on
//! them, the differencing, and the state-space matrices they make.

/// `1 + sign (c_1 B^step + c_2 B^(2 step) + ...)`, its coefficients from
/// `B^0`: `sign = -1` for an autoregressive factor (`1 - phi_1 B - ...`),
/// `+1` for a moving-average one (`1 + theta_1 B + ...`).
pub(crate) fn lag_poly(coefs: &[f64], step: usize, sign: f64) -> Vec<f64> {
    let mut out = vec![0.0; coefs.len() * step + 1];
    out[0] = 1.0;
    for (i, c) in coefs.iter().enumerate() {
        out[(i + 1) * step] = sign * c;
    }
    out
}

/// The product of two polynomials, coefficients from `B^0` (a convolution,
/// numpy's `polymul` on the same coefficients).
pub(crate) fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// Whether `1 - phi_1 z - ... - phi_p z^p` has every root outside the unit
/// circle, by the step-down (Schur-Cohn) recursion: the partial
/// autocorrelations `r_p, ..., r_1` it yields are each inside `(-1, 1)` if
/// and only if it does (Monahan 1984, the inverse of the transform
/// statsmodels' `constrain_stationary_univariate` applies). Arithmetic
/// only. An empty polynomial is stationary; a NaN fails.
pub(crate) fn stationary(phi: &[f64]) -> bool {
    let mut a = phi.to_vec();
    let mut next = Vec::with_capacity(a.len());
    while let Some(&r) = a.last() {
        if r.is_nan() || r.abs() >= 1.0 {
            return false;
        }
        let k = a.len();
        let scale = 1.0 - r * r;
        next.clear();
        next.extend((0..k - 1).map(|i| (a[i] + r * a[k - 2 - i]) / scale));
        std::mem::swap(&mut a, &mut next);
    }
    true
}

/// `y_t` differenced as statsmodels' `diff` does it from the window
/// `win = (y_{t-depth}, ..., y_t)`: `big_d` seasonal differences of period
/// `s` first, then `d` ordinary ones, each a subtraction of two values the
/// stage before left, so the result is statsmodels' to the bit. A NaN in
/// the window (a slot not observed) gives NaN. `work` holds the stages.
pub(crate) fn difference(
    win: &[f64],
    d: usize,
    big_d: usize,
    s: usize,
    work: &mut Vec<f64>,
) -> f64 {
    work.clear();
    work.extend_from_slice(win);
    for _ in 0..big_d {
        let n = work.len() - s;
        for i in 0..n {
            work[i] = work[i + s] - work[i];
        }
        work.truncate(n);
    }
    for _ in 0..d {
        let n = work.len() - 1;
        for i in 0..n {
            work[i] = work[i + 1] - work[i];
        }
        work.truncate(n);
    }
    debug_assert_eq!(work.len(), 1);
    work[0]
}

/// The time-invariant matrices of the ARMA part in statsmodels' `SARIMAX`
/// form (Harvey's), with `r = max(p + sP, q + sQ + 1)` states:
///
/// ```text
/// alpha_{t+1} = T alpha_t + R e_t,   w_t = x_t' beta + Z alpha_t,   e_t ~ (0, sigma2)
/// T = [ -A_1     1 0 ... 0 ]          Z = (1, 0, ..., 0)
///     [ -A_2     0 1 ... 0 ]          R = (1, C_1, ..., C_{r-1})'
///     [  ...               ]
///     [ -A_r     0 0 ... 0 ]
/// ```
///
/// with `A(B) = phi(B) Phi(B^s)` and `C(B) = theta(B) Theta(B^s)` the
/// reduced polynomials, zero past their degrees.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Ssm {
    /// `T`'s first column: `-A_1, ..., -A_r`.
    pub(crate) t_col: Vec<f64>,
    /// `R`: `1, C_1, ..., C_{r-1}`.
    pub(crate) r_vec: Vec<f64>,
    /// The innovations' variance.
    pub(crate) sigma2: f64,
}

impl Ssm {
    /// The matrices of `phi`, `theta`, `Phi`, `Theta` at season `s`, and
    /// `sigma2`.
    pub(crate) fn new(
        ar: &[f64],
        ma: &[f64],
        sar: &[f64],
        sma: &[f64],
        s: usize,
        sigma2: f64,
    ) -> Self {
        let a = polymul(&lag_poly(ar, 1, -1.0), &lag_poly(sar, s.max(1), -1.0));
        let c = polymul(&lag_poly(ma, 1, 1.0), &lag_poly(sma, s.max(1), 1.0));
        let r = Self::states(a.len() - 1, c.len() - 1);
        let t_col = (1..=r).map(|i| a.get(i).map_or(0.0, |v| -v)).collect();
        let r_vec: Vec<f64> = (0..r).map(|i| c.get(i).copied().unwrap_or(0.0)).collect();
        Self {
            t_col,
            r_vec,
            sigma2,
        }
    }

    /// `(R sigma2 R')_ij`, formed as statsmodels forms it, `(R_i sigma2) R_j`.
    #[inline]
    pub(crate) fn rqr(&self, i: usize, j: usize) -> f64 {
        self.r_vec[i] * self.sigma2 * self.r_vec[j]
    }

    /// `max(p_full, q_full + 1)`: the states of an ARMA whose reduced
    /// polynomials have degrees `p_full` and `q_full`.
    pub(crate) fn states(p_full: usize, q_full: usize) -> usize {
        p_full.max(q_full + 1)
    }

    pub(crate) fn r(&self) -> usize {
        self.t_col.len()
    }

    /// `a <- T a`, in place: `a_i = T_i0 a_0 + a_{i+1}`.
    pub(crate) fn predict_mean(&self, a: &mut [f64]) {
        let r = a.len();
        let a0 = a[0];
        for i in 0..r {
            let next = if i + 1 < r { a[i + 1] } else { 0.0 };
            a[i] = self.t_col[i] * a0 + next;
        }
    }

    /// `P <- T P T' + R sigma2 R'`, then made symmetric as statsmodels'
    /// filter makes it (`(P + P') / 2`), in place on the full `r x r`
    /// matrix. `T` is a companion matrix, so each product is two terms an
    /// entry, `O(r^2)`, and runs in place with one saved row: `(T P)_il =
    /// T_i0 P_0l + P_{i+1,l}` reads the row below, not yet overwritten, and
    /// row 0, saved in `row0`; `(M T')_ij = M_i0 T_j0 + M_{i,j+1}` reads
    /// the entry to the right, likewise.
    pub(crate) fn predict_cov(&self, p: &mut [f64], row0: &mut Vec<f64>) {
        let r = self.r();
        row0.clear();
        row0.extend_from_slice(&p[..r]);
        for i in 0..r {
            let ti = self.t_col[i];
            for l in 0..r {
                let below = if i + 1 < r { p[(i + 1) * r + l] } else { 0.0 };
                p[i * r + l] = ti * row0[l] + below;
            }
        }
        for i in 0..r {
            let m0 = p[i * r];
            for j in 0..r {
                let right = if j + 1 < r { p[i * r + j + 1] } else { 0.0 };
                p[i * r + j] = (m0 * self.t_col[j] + right) + self.rqr(i, j);
            }
        }
        for i in 0..r {
            for j in 0..i {
                let m = (p[i * r + j] + p[j * r + i]) / 2.0;
                p[i * r + j] = m;
                p[j * r + i] = m;
            }
        }
    }

    /// The stationary covariance of the state, the `P` that solves
    /// `P = T P T' + R sigma2 R'` (statsmodels' `initialization =
    /// "stationary"`), by Smith's doubling: `P_{k+1} = P_k + A_k P_k A_k'`,
    /// `A_{k+1} = A_k^2`, from `P_0 = R sigma2 R'` and `A_0 = T`, so that
    /// `P_k` sums the first `2^k` terms of `sum_j T^j R sigma2 R' T'^j`.
    /// It stops when a step adds less than a quarter of the double's
    /// precision to every entry. `None` where 64 steps do not get there:
    /// `T` has a root on or outside the unit circle, which the settings'
    /// check refuses first.
    pub(crate) fn stationary_cov(&self) -> Option<Vec<f64>> {
        let r = self.r();
        let mut p: Vec<f64> = (0..r * r).map(|k| self.rqr(k / r, k % r)).collect();
        let mut a = vec![0.0; r * r];
        for i in 0..r {
            a[i * r] = self.t_col[i];
            if i + 1 < r {
                a[i * r + i + 1] = 1.0;
            }
        }
        let mut ap = vec![0.0; r * r];
        let mut apa = vec![0.0; r * r];
        for _ in 0..64 {
            matmul(&a, &p, &mut ap, r, false);
            matmul(&ap, &a, &mut apa, r, true);
            let scale = p.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let mut small = true;
            for (pi, add) in p.iter_mut().zip(&apa) {
                if add.abs() > 0.25 * f64::EPSILON * scale {
                    small = false;
                }
                *pi += add;
            }
            if !p.iter().all(|v| v.is_finite()) {
                return None;
            }
            if small {
                return Some(p);
            }
            matmul(&a, &a.clone(), &mut ap, r, false);
            std::mem::swap(&mut a, &mut ap);
        }
        None
    }
}

/// `out = x y` (or `x y'` with `transpose_y`), `r x r`, row-major.
fn matmul(x: &[f64], y: &[f64], out: &mut [f64], r: usize, transpose_y: bool) {
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0;
            for k in 0..r {
                let yk = if transpose_y {
                    y[j * r + k]
                } else {
                    y[k * r + j]
                };
                s += x[i * r + k] * yk;
            }
            out[i * r + j] = s;
        }
    }
}

/// The full matrix's upper triangle into `out`, each entry the mean of
/// the pair it stands for (they are equal after a symmetric step).
pub(crate) fn pack(full: &[f64], r: usize, out: &mut Vec<f64>) {
    out.clear();
    for i in 0..r {
        out.push(full[i * r + i]);
        for j in i + 1..r {
            let (a, b) = (full[i * r + j], full[j * r + i]);
            out.push(if a == b { a } else { (a + b) / 2.0 });
        }
    }
}

/// The packed matrix written out in full into `out`.
pub(crate) fn unpack(p: &[f64], r: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(r * r, 0.0);
    let mut k = 0;
    for i in 0..r {
        for j in i..r {
            out[i * r + j] = p[k];
            out[j * r + i] = p[k];
            k += 1;
        }
    }
}
