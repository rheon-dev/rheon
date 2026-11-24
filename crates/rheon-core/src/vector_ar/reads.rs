//! The bank's reads of a [`VectorAr`]'s fit: the impulse responses and
//! their standard errors, the forecast-error variance decomposition, the
//! companion matrix's eigenvalues, the long-run mean, every order's
//! information criteria and a Granger test asked after the fact. Each is a
//! function of the coefficients, `Σ_u` and the coefficients' covariance,
//! computed when asked for, from the moments as they stand: nothing here is
//! kept in the state.
//!
//! With `A_1 … A_q` the lag matrices, `Σ_u = P Pᵀ` (Cholesky, the series in
//! their given order) and `n` Kish's count:
//!
//! ```text
//! Φ_0 = I,  Φ_i = Σ_{j ≤ min(i, q)} Φ_{i−j} A_j       the impulse responses
//! Θ_i = Φ_i P                                          orthogonalized
//! ω_ej(h) = Σ_{i<h} Θ_i[e, j]² / (Σ_{i<h} Φ_i Σ_u Φ_iᵀ)[e, e]
//! Cov(vec Φ_i) = G_i Σ_α G_iᵀ,  G_i = Σ_{m<i} J (Aᵀ)^{i−1−m} ⊗ Φ_m
//! Cov(vec Θ_i) = C_i Σ_α C_iᵀ + C̄_i Σ_σ C̄_iᵀ / n
//! ```
//!
//! (Lütkepohl 2005, 3.7.5 and 3.7.8; statsmodels' `IRAnalysis.cov`), with
//! `A` the companion matrix, `Σ_α = (S⁻¹)_lags / n ⊗ Σ_u`,
//! `C_i = (Pᵀ ⊗ I) G_i`, `C̄_i = (I ⊗ Φ_i) H`, `H = ∂vec P / ∂vech Σ_u` and
//! `Σ_σ = 2 D⁺ (Σ_u ⊗ Σ_u) D⁺ᵀ`.

use super::VectorAr;
use super::fit::{self, Factor, Moments, Scratch, Shape, VectorArSelect};
use crate::ewridge::SystemKept;

/// Which of a VAR's reads the bank returns as a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorArTable {
    /// `Φ_i`, one row per period, response and impulse.
    Irf,
    /// `Θ_i = Φ_i P`.
    OrthIrf,
    /// The forecast-error variance decomposition.
    Fevd,
    /// The companion matrix's eigenvalues.
    Stability,
    /// The long-run mean `(I − Σ A_j)⁻¹ c`.
    Mean,
    /// Every order's information criteria.
    Order,
    /// One Granger test.
    Causality,
}

impl VectorArTable {
    pub fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "irf" => Self::Irf,
            "orth_irf" => Self::OrthIrf,
            "fevd" => Self::Fevd,
            "stability" => Self::Stability,
            "mean" => Self::Mean,
            "order" => Self::Order,
            "causality" => Self::Causality,
            other => {
                return Err(format!(
                    "ModelBank.vector_ar: table must be \"irf\", \"orth_irf\", \"fevd\", \
                     \"stability\", \"mean\", \"order\" or \"causality\", but it is {other:?}. \
                     Set table to one of those words."
                ));
            }
        })
    }
}

/// A VAR's fit as its moments stand ([`VectorAr::fit_now`]).
#[derive(Debug, Clone)]
pub struct VectorArFit {
    /// Series.
    pub k: usize,
    /// Features.
    pub e: usize,
    /// The order in force.
    pub order: usize,
    /// The largest order.
    pub max_order: usize,
    /// The constant per equation; zeros without `fit_intercept`.
    pub intercept: Vec<f64>,
    /// `A_1 … A_q`, each `k × k` row-major: `coefs[(l k + i) k + j]` is
    /// the coefficient of series `j` at lag `l + 1` in equation `i`.
    pub coefs: Vec<f64>,
    /// `Σ_u`, `k × k`, statsmodels' `sigma_u`: NaN where the degrees of
    /// freedom are not positive.
    pub sigma_u: Vec<f64>,
    /// `Σ_mle = SSE / W`, statsmodels' `sigma_u_mle`.
    pub sigma_u_mle: Vec<f64>,
    /// Kish's count of the learned rows, `W² / Σw²`.
    pub n_kish: f64,
    /// The residual degrees of freedom, `n − (d + 1)`.
    pub df: f64,
    /// `(S_m⁻¹)` over the lag columns, `kq × kq`.
    sinv_lags: Vec<f64>,
    shape: Shape,
    mom: Moments,
    factor: Factor,
    /// The largest identified order.
    top: usize,
}

/// `out = a · b` for row-major `n × m` and `m × r`.
fn matmul(a: &[f64], b: &[f64], n: usize, m: usize, r: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * r];
    for i in 0..n {
        for q in 0..m {
            let v = a[i * m + q];
            if v == 0.0 {
                continue;
            }
            for j in 0..r {
                out[i * r + j] += v * b[q * r + j];
            }
        }
    }
    out
}

/// `aᵀ`, `n × m` into `m × n`.
fn transpose(a: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            out[j * n + i] = a[i * m + j];
        }
    }
    out
}

/// `x` with `a x = b`, `a` square `n × n` and invertible: Gaussian
/// elimination with partial pivoting. `None` where a pivot is 0.
fn solve_general(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for c in 0..n {
        let piv = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs()))?;
        if m[piv * n + c] == 0.0 || m[piv * n + c].is_nan() {
            return None;
        }
        if piv != c {
            for j in 0..n {
                m.swap(c * n + j, piv * n + j);
            }
            x.swap(c, piv);
        }
        for i in c + 1..n {
            let f = m[i * n + c] / m[c * n + c];
            for j in c..n {
                m[i * n + j] -= f * m[c * n + j];
            }
            x[i] -= f * x[c];
        }
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| m[i * n + j] * x[j]).sum();
        x[i] = (x[i] - s) / m[i * n + i];
    }
    Some(x)
}

pub(super) fn fit_now(model: &VectorAr) -> Result<VectorArFit, String> {
    let cfg = &model.cfg;
    let sh = cfg.shape();
    let (d, k) = (cfg.n_regressors(), cfg.n_targets);
    let mut kept = SystemKept::default();
    let mut mom = Moments::default();
    let read = model.fit.system(0, &mut kept).is_some_and(|sys| {
        let w = sys.wj;
        match sys.tm.and_then(|tm| tm.n_kish(&[w])[0]) {
            Some(n) if w > 0.0 && n > 0.0 => mom.read(sys.cov, d, k, cfg.fit_intercept, w, n),
            _ => false,
        }
    });
    if !read {
        return Err(
            "it has learned no row yet: a row is learned once the order's lags before it are in. \
             Read it after more rows."
                .into(),
        );
    }
    let mut factor = Factor::default();
    factor.of(&mom, cfg.ridge);
    let mut s = Scratch::default();
    let top = sh.identified(&factor);
    let order = match (cfg.select, top) {
        (None, Some(t)) if t == sh.p => Some(sh.p),
        (Some(ic), Some(t)) if t >= 1 || model.defined => {
            fit::choose(ic, &sh, &factor, &mom, t, &mut s)
        }
        _ => None,
    };
    let Some(q) = order else {
        return Err(format!(
            "its rows do not identify the fit yet: the second moments of its lags and features \
             are not positive definite at order {}, read by Cholesky with every pivot above 1e-10 of its \
             diagonal. Read it once more rows have spread the lags in every direction.",
            if cfg.select.is_some() { 1 } else { sh.p }
        ));
    };
    let m = sh.m(q);
    let mut b = Vec::new();
    factor.coefficients(m, &mut b);
    let mut coefs = vec![0.0; q * k * k];
    for l in 0..q {
        for i in 0..k {
            for j in 0..k {
                coefs[(l * k + i) * k + j] = b[(sh.e + l * k + j) * k + i];
            }
        }
    }
    let intercept: Vec<f64> = (0..k)
        .map(|i| {
            if cfg.fit_intercept {
                mom.ybar[i] - (0..m).map(|r| b[r * k + i] * mom.zbar[r]).sum::<f64>()
            } else {
                0.0
            }
        })
        .collect();
    let mut sse = Vec::new();
    factor.sse(&mom, m, &mut sse);
    let df = fit::df_resid(&sh, &mom, q);
    let n = mom.n_kish;
    let sigma_u: Vec<f64> = sse
        .iter()
        .map(|v| if df > 0.0 { v * n / df } else { f64::NAN })
        .collect();
    let mut linv = Vec::new();
    factor.inverse_factor(m, &mut linv);
    let kq = k * q;
    let mut sinv_lags = vec![0.0; kq * kq];
    for a in 0..kq {
        for c in 0..kq {
            sinv_lags[a * kq + c] = fit::inverse_entry(&linv, m, sh.e + a, sh.e + c);
        }
    }
    Ok(VectorArFit {
        k,
        e: sh.e,
        order: q,
        max_order: sh.p,
        intercept,
        coefs,
        sigma_u,
        sigma_u_mle: sse,
        n_kish: n,
        df,
        sinv_lags,
        shape: sh,
        mom,
        factor,
        top: top.unwrap_or(0),
    })
}

impl VectorArFit {
    fn a(&self, l: usize) -> &[f64] {
        let k = self.k;
        &self.coefs[l * k * k..(l + 1) * k * k]
    }

    /// `Φ_0 … Φ_H`, `(H + 1) × k × k`: statsmodels' `ma_rep(H)`.
    pub fn ma_rep(&self, periods: usize) -> Vec<f64> {
        let k = self.k;
        let mut phi = vec![0.0; (periods + 1) * k * k];
        for i in 0..k {
            phi[i * k + i] = 1.0;
        }
        for i in 1..=periods {
            let mut acc = vec![0.0; k * k];
            for j in 1..=i.min(self.order) {
                let prev = &phi[(i - j) * k * k..(i - j + 1) * k * k];
                let p = matmul(prev, self.a(j - 1), k, k, k);
                for (o, v) in acc.iter_mut().zip(p) {
                    *o += v;
                }
            }
            phi[i * k * k..(i + 1) * k * k].copy_from_slice(&acc);
        }
        phi
    }

    /// `P`, the lower Cholesky factor of `Σ_u`; `None` where it is not
    /// positive definite.
    fn chol_sigma(&self) -> Option<Vec<f64>> {
        let mut p = self.sigma_u.clone();
        fit::cholesky(&mut p, self.k).then_some(p)
    }

    /// `Θ_i = Φ_i P`, `(H + 1) × k × k`: statsmodels' `orth_ma_rep(H)`.
    /// NaN where `Σ_u` is not positive definite.
    pub fn orth_ma_rep(&self, periods: usize) -> Vec<f64> {
        let k = self.k;
        let phi = self.ma_rep(periods);
        let Some(p) = self.chol_sigma() else {
            return vec![f64::NAN; phi.len()];
        };
        phi.chunks(k * k)
            .flat_map(|ph| matmul(ph, &p, k, k, k))
            .collect()
    }

    /// The standard error of each response, `(H + 1) × k × k` beside
    /// [`Self::ma_rep`] (or [`Self::orth_ma_rep`] with `orth`): the square
    /// root of `Cov(vec Φ_i)`'s (or `Cov(vec Θ_i)`'s) diagonal, statsmodels'
    /// `irf(H).stderr(orth)`.
    pub fn irf_stderr(&self, periods: usize, orth: bool) -> Vec<f64> {
        let (k, q) = (self.k, self.order);
        let kq = k * q;
        let n = self.n_kish;
        let phi = self.ma_rep(periods);
        let mut out = vec![0.0; (periods + 1) * k * k];
        let p = match (orth, self.chol_sigma()) {
            (true, None) => return vec![f64::NAN; out.len()],
            (_, p) => p,
        };
        // `U_a = J (Aᵀ)^a`, `k × kq`, for `a` in `0..H`: the first `k`
        // rows of the companion matrix's transpose to the power `a`.
        let mut comp_t = vec![0.0; kq * kq];
        for l in 0..q {
            for i in 0..k {
                for j in 0..k {
                    // companion[i][l k + j] = A_{l+1}[i][j], so its
                    // transpose holds it at [l k + j][i].
                    comp_t[(l * k + j) * kq + i] = self.a(l)[i * k + j];
                }
            }
        }
        for r in k..kq {
            // companion[r][r − k] = 1.
            comp_t[(r - k) * kq + r] = 1.0;
        }
        let mut pow = vec![0.0; kq * kq];
        for r in 0..kq {
            pow[r * kq + r] = 1.0;
        }
        let sig_lag: Vec<f64> = self.sinv_lags.iter().map(|v| v / n).collect();
        // `V_a = Pᵀ U_a` (or `U_a` alone), and `V_a Σ_lag`.
        let mut v = Vec::with_capacity(periods);
        let mut vs = Vec::with_capacity(periods);
        for _ in 0..periods {
            let u = &pow[..k * kq];
            let va = match &p {
                Some(p) if orth => matmul(&transpose(p, k, k), u, k, k, kq),
                _ => u.to_vec(),
            };
            vs.push(matmul(&va, &sig_lag, k, kq, kq));
            v.push(va);
            pow = matmul(&pow, &comp_t, kq, kq, kq);
        }
        // `Φ_m Σ_u`.
        let phis: Vec<Vec<f64>> = (0..=periods)
            .map(|m| matmul(&phi[m * k * k..(m + 1) * k * k], &self.sigma_u, k, k, k))
            .collect();
        for i in 1..=periods {
            for a in 0..k {
                for b in 0..k {
                    let mut s = 0.0;
                    for (m, pm) in phis.iter().enumerate().take(i) {
                        for m2 in 0..i {
                            let (x, y) = (i - 1 - m, i - 1 - m2);
                            let xa: f64 =
                                (0..kq).map(|r| vs[x][a * kq + r] * v[y][a * kq + r]).sum();
                            let pb = &phi[m2 * k * k..(m2 + 1) * k * k];
                            let yb: f64 = (0..k).map(|c| pm[b * k + c] * pb[b * k + c]).sum();
                            s += xa * yb;
                        }
                    }
                    // vec index `a k + b` is the response `b` to impulse `a`.
                    out[i * k * k + b * k + a] = s;
                }
            }
        }
        if orth && let Some(p) = &p {
            let m = self.sigma_vec_cov(p);
            let kk = k * k;
            for i in 0..=periods {
                let ph = &phi[i * kk..(i + 1) * kk];
                for a in 0..k {
                    for b in 0..k {
                        let mut s = 0.0;
                        for c in 0..k {
                            for dd in 0..k {
                                s += ph[b * k + c]
                                    * m[(a * k + c) * kk + a * k + dd]
                                    * ph[b * k + dd];
                            }
                        }
                        out[i * kk + b * k + a] += s / n;
                    }
                }
            }
        }
        out.iter().map(|v| v.max(0.0).sqrt()).collect()
    }

    /// `H Σ_σ Hᵀ`, `k² × k²`: the asymptotic covariance of `vec P` times
    /// `n`, with `H = ∂vec P / ∂vech Σ_u` from the Cholesky factor's
    /// derivative `dP = P Φ(P⁻¹ dΣ P⁻ᵀ)` (`Φ` the lower triangle with its
    /// diagonal halved) and `Σ_σ = 2 D⁺ (Σ_u ⊗ Σ_u) D⁺ᵀ`.
    fn sigma_vec_cov(&self, p: &[f64]) -> Vec<f64> {
        let k = self.k;
        let kk = k * k;
        let sig = &self.sigma_u;
        // `vech` positions, column by column: (row, col) with row >= col.
        let vech: Vec<(usize, usize)> = (0..k).flat_map(|c| (c..k).map(move |r| (r, c))).collect();
        let nv = vech.len();
        // `P⁻¹`, lower.
        let mut pinv = vec![0.0; kk];
        for c in 0..k {
            pinv[c * k + c] = 1.0 / p[c * k + c];
            for i in c + 1..k {
                let s: f64 = (c..i).map(|q| p[i * k + q] * pinv[q * k + c]).sum();
                pinv[i * k + c] = -s / p[i * k + i];
            }
        }
        let pinv_t = transpose(&pinv, k, k);
        // `H`, `k² × nv`, vec index `c k + r` for `P[r][c]`.
        let mut h = vec![0.0; kk * nv];
        for (col, &(r, c)) in vech.iter().enumerate() {
            let mut ds = vec![0.0; kk];
            ds[r * k + c] = 1.0;
            ds[c * k + r] = 1.0;
            let x = matmul(&matmul(&pinv, &ds, k, k, k), &pinv_t, k, k, k);
            let mut phi = vec![0.0; kk];
            for i in 0..k {
                for j in 0..i {
                    phi[i * k + j] = x[i * k + j];
                }
                phi[i * k + i] = 0.5 * x[i * k + i];
            }
            let dp = matmul(p, &phi, k, k, k);
            for i in 0..k {
                for j in 0..k {
                    h[(j * k + i) * nv + col] = dp[i * k + j];
                }
            }
        }
        // `D⁺`'s entries for each vech position: one vec position at 1 on
        // the diagonal, two at 1/2 off it.
        let parts = |(r, c): (usize, usize)| -> Vec<(usize, f64)> {
            if r == c {
                vec![(c * k + r, 1.0)]
            } else {
                vec![(c * k + r, 0.5), (r * k + c, 0.5)]
            }
        };
        let mut ssig = vec![0.0; nv * nv];
        for (v1, &e1) in vech.iter().enumerate() {
            for (v2, &e2) in vech.iter().enumerate() {
                let mut s = 0.0;
                for (i1, w1) in parts(e1) {
                    for (i2, w2) in parts(e2) {
                        // (Σ ⊗ Σ)[c1 k + r1][c2 k + r2] = Σ[c1][c2] Σ[r1][r2].
                        let (c1, r1, c2, r2) = (i1 / k, i1 % k, i2 / k, i2 % k);
                        s += w1 * w2 * sig[c1 * k + c2] * sig[r1 * k + r2];
                    }
                }
                ssig[v1 * nv + v2] = 2.0 * s;
            }
        }
        let hs = matmul(&h, &ssig, kk, nv, nv);
        matmul(&hs, &transpose(&h, kk, nv), kk, nv, kk)
    }

    /// The forecast-error variance decomposition, `k × H × k`: equation,
    /// horizon `1..=H`, shock, statsmodels' `fevd(H).decomp`. NaN where
    /// `Σ_u` is not positive definite.
    pub fn fevd(&self, periods: usize) -> Vec<f64> {
        let k = self.k;
        let kk = k * k;
        let theta = self.orth_ma_rep(periods);
        let phi = self.ma_rep(periods);
        let mut out = vec![0.0; k * periods * k];
        let mut cum = vec![0.0; kk];
        let mut mse = vec![0.0; kk];
        for h in 0..periods {
            let th = &theta[h * kk..(h + 1) * kk];
            for (c, t) in cum.iter_mut().zip(th) {
                *c += t * t;
            }
            let ph = &phi[h * kk..(h + 1) * kk];
            let m = matmul(
                &matmul(ph, &self.sigma_u, k, k, k),
                &transpose(ph, k, k),
                k,
                k,
                k,
            );
            for (a, b) in mse.iter_mut().zip(m) {
                *a += b;
            }
            for e in 0..k {
                for j in 0..k {
                    out[(e * periods + h) * k + j] = cum[e * k + j] / mse[e * k + e];
                }
            }
        }
        out
    }

    /// The companion matrix's eigenvalues as `(re, im)`, largest modulus
    /// first: the process is stable where every one is inside the unit
    /// circle. statsmodels' `roots` are their reciprocals. Empty at order 0.
    pub fn eigenvalues(&self) -> Result<Vec<(f64, f64)>, String> {
        let (k, q) = (self.k, self.order);
        let kq = k * q;
        if kq == 0 {
            return Ok(Vec::new());
        }
        if self.coefs.iter().any(|v| !v.is_finite()) {
            return Err(
                "its coefficients are not finite. Check the rows it learned for values past the \
                 doubles' range."
                    .into(),
            );
        }
        let comp = faer::Mat::<f64>::from_fn(kq, kq, |r, c| {
            if r < k {
                let (l, j) = (c / k, c % k);
                self.a(l)[r * k + j]
            } else if c + k == r {
                1.0
            } else {
                0.0
            }
        });
        let ev = comp.eigenvalues().map_err(|e| {
            format!(
                "its companion matrix's eigenvalues did not converge ({e:?}). Read it again \
                     after more rows."
            )
        })?;
        let mut out: Vec<(f64, f64)> = ev.iter().map(|z| (z.re, z.im)).collect();
        out.sort_by(|a, b| b.0.hypot(b.1).total_cmp(&a.0.hypot(a.1)));
        Ok(out)
    }

    /// The long-run mean `(I − Σ A_j)⁻¹ c`, statsmodels' `mean()`: `None`
    /// with features, whose values the mean would depend on, and where
    /// `I − Σ A_j` is singular (a unit root).
    pub fn mean(&self) -> Option<Vec<f64>> {
        if self.e > 0 {
            return None;
        }
        let k = self.k;
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        for l in 0..self.order {
            for (o, v) in a.iter_mut().zip(self.a(l)) {
                *o -= v;
            }
        }
        solve_general(&a, &self.intercept, k)
    }

    /// Every order's `[aic, bic, hqic, fpe]` in `0..=p`, on the same rows:
    /// statsmodels' `select_order(p).ics`. NaN for an order the moments do
    /// not identify or a value not defined.
    pub fn info_criteria(&self) -> Vec<[f64; 4]> {
        let mut s = Scratch::default();
        let ics = [
            VectorArSelect::Aic,
            VectorArSelect::Bic,
            VectorArSelect::Hqic,
            VectorArSelect::Fpe,
        ];
        (0..=self.max_order)
            .map(|q| {
                let mut row = [f64::NAN; 4];
                if q > self.top {
                    return row;
                }
                for (o, ic) in row.iter_mut().zip(ics) {
                    let v = fit::criterion(ic, &self.shape, &self.factor, &self.mom, q, &mut s);
                    *o = match (ic, v) {
                        (_, None) => f64::NAN,
                        (VectorArSelect::Fpe, Some(v)) => libm::exp(v),
                        (_, Some(v)) => v,
                    };
                }
                row
            })
            .collect()
    }

    /// The Granger test of `causing`'s lags in `caused`'s equations at the
    /// order in force: `[statistic, p-value, df_num, df_denom]` by `kind`
    /// (`df_denom` NaN for a χ² law). Every entry NaN where the test is not
    /// defined (order 0, no residual degrees of freedom).
    pub fn test(
        &self,
        caused: &[usize],
        causing: &[usize],
        kind: super::CausalityKind,
    ) -> [f64; 4] {
        let mut s = Scratch::default();
        let lam = fit::wald(
            &self.shape,
            &self.factor,
            &self.mom,
            self.order,
            caused,
            causing,
            &mut s,
        );
        if lam.is_nan() {
            return [f64::NAN; 4];
        }
        let q = (self.order * caused.len() * causing.len()) as f64;
        kind.read(lam, q, self.df, self.n_kish, self.k as f64)
    }
}
