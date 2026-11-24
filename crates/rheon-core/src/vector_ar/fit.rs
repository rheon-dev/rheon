//! The VAR's fit read from its moments: one Cholesky factor of the
//! regressors' second moments, and from it the coefficients at any order,
//! each order's residual co-moments, the order an information criterion
//! chooses, a Granger test's Wald statistic and the coefficients' standard
//! errors.
//!
//! With `z = [x; y_{t−1}; …; y_{t−p}]` (`d = e + kp` regressors, features
//! first, then the lags lag by lag), centred with an intercept and raw
//! without one, `S = E[z zᵀ]` (`d × d`), `S_zy = E[z yᵀ]` (`d × k`) and
//! `S_yy = E[y yᵀ]` (`k × k`), all exponentially weighted means over the
//! learned rows, and the factor `S = L Lᵀ`:
//!
//! ```text
//! R      = L⁻¹ S_zy                      (forward solve, d × k)
//! B_m    = L_m⁻ᵀ R_m                      the first m = e + kq rows: order q
//! SSE_m  = S_yy − R_mᵀ R_m               the residual co-moments at order q
//! ```
//!
//! The order-`q` system is the leading `m × m` block of `S`, and the factor
//! of a leading block is the leading block of the factor, so one factor and
//! one forward solve give every order's fit and residuals on the same rows.
//! Every quantity here is arithmetic and square roots alone except the
//! logarithms of the order choice, which are the `libm` crate's, the same
//! bits on every platform.

/// The smallest share of a regressor's spread that the regressors before
/// it may leave unexplained for the system to count as identified: an
/// exactly collinear design leaves rounding, about `1e-16` to `1e-14` of
/// it; `1e-10` is past that and below any design whose coefficients carry
/// a digit.
pub(crate) const DEFINITE: f64 = 1e-10;

/// The criterion an order is chosen by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorArSelect {
    Aic,
    Bic,
    Hqic,
    Fpe,
}

/// The moments a fit is read from, copied out of the carrier's Gram in the
/// regressors' order. Kept by the caller between reads, so a read
/// allocates nothing once its buffers have grown.
#[derive(Debug, Default, Clone)]
pub(crate) struct Moments {
    pub(crate) d: usize,
    pub(crate) k: usize,
    /// `S`, `d × d`, row-major.
    pub(crate) s: Vec<f64>,
    /// `S_zy`, `d × k`.
    pub(crate) szy: Vec<f64>,
    /// `S_yy`, `k × k`.
    pub(crate) syy: Vec<f64>,
    /// The regressors' means (with an intercept; zeros without).
    pub(crate) zbar: Vec<f64>,
    /// The targets' means (with an intercept; zeros without).
    pub(crate) ybar: Vec<f64>,
    /// `W`, the learned rows' decayed weight.
    pub(crate) w: f64,
    /// Kish's count `W² / Σw²` of the learned rows, decayed.
    pub(crate) n_kish: f64,
    pub(crate) intercept: bool,
}

impl Moments {
    /// Read from the Gram `cov` of `[1?, z, y]`: its regressors are the
    /// slots after the intercept's, then the `k` targets.
    pub(crate) fn read(
        &mut self,
        cov: &crate::EwCov,
        d: usize,
        k: usize,
        intercept: bool,
        w: f64,
        n_kish: f64,
    ) -> bool {
        let Some(off) = cov.k().checked_sub(d + k) else {
            return false;
        };
        self.d = d;
        self.k = k;
        self.w = w;
        self.n_kish = n_kish;
        self.intercept = intercept;
        let at = |i: usize, j: usize| {
            if intercept {
                cov.cov(off + i, off + j)
            } else {
                cov.raw(off + i, off + j)
            }
        };
        self.s.clear();
        self.s.resize(d * d, 0.0);
        for i in 0..d {
            for j in 0..=i {
                let v = at(i, j);
                self.s[i * d + j] = v;
                self.s[j * d + i] = v;
            }
        }
        self.szy.clear();
        self.szy.resize(d * k, 0.0);
        for i in 0..d {
            for j in 0..k {
                self.szy[i * k + j] = at(i, d + j);
            }
        }
        self.syy.clear();
        self.syy.resize(k * k, 0.0);
        for i in 0..k {
            for j in 0..=i {
                let v = at(d + i, d + j);
                self.syy[i * k + j] = v;
                self.syy[j * k + i] = v;
            }
        }
        let means = cov.means();
        self.zbar.clear();
        self.ybar.clear();
        if intercept {
            self.zbar.extend_from_slice(&means[off..off + d]);
            self.ybar.extend_from_slice(&means[off + d..off + d + k]);
        } else {
            self.zbar.resize(d, 0.0);
            self.ybar.resize(k, 0.0);
        }
        true
    }
}

/// The factor and forward solve of a [`Moments`], with how far the factor
/// got: `pass` leading pivots each above [`DEFINITE`] of their diagonal.
#[derive(Debug, Default, Clone)]
pub(crate) struct Factor {
    pub(crate) d: usize,
    pub(crate) k: usize,
    /// `L`, `d × d` row-major lower: valid in its first `pass` rows.
    pub(crate) l: Vec<f64>,
    /// `R = L⁻¹ S_zy`, `d × k`: valid in its first `pass` rows.
    pub(crate) r: Vec<f64>,
    pub(crate) pass: usize,
}

impl Factor {
    /// Factor `S + ridge I` by Cholesky, stopping at the first pivot at or
    /// below [`DEFINITE`] of its diagonal, and solve forward for `S_zy`.
    pub(crate) fn of(&mut self, m: &Moments, ridge: f64) {
        let (d, k) = (m.d, m.k);
        self.d = d;
        self.k = k;
        self.l.clear();
        self.l.resize(d * d, 0.0);
        self.r.clear();
        self.r.resize(d * k, 0.0);
        self.pass = 0;
        let l = &mut self.l;
        'rows: for i in 0..d {
            for j in 0..=i {
                let s: f64 = (0..j).map(|q| l[i * d + q] * l[j * d + q]).sum();
                let a0 = m.s[i * d + j] + if i == j { ridge } else { 0.0 };
                let a = a0 - s;
                if i == j {
                    if !(a0 > 0.0 && a > DEFINITE * a0) {
                        break 'rows;
                    }
                    l[i * d + i] = a.sqrt();
                } else {
                    l[i * d + j] = a / l[j * d + j];
                }
            }
            self.pass = i + 1;
        }
        for i in 0..self.pass {
            for c in 0..k {
                let s: f64 = (0..i).map(|q| l[i * d + q] * self.r[q * k + c]).sum();
                self.r[i * k + c] = (m.szy[i * k + c] - s) / l[i * d + i];
            }
        }
    }

    /// The residual co-moments at `m` regressors, `S_yy − R_mᵀ R_m`, into
    /// `out` (`k × k`).
    pub(crate) fn sse(&self, mom: &Moments, m: usize, out: &mut Vec<f64>) {
        let k = self.k;
        out.clear();
        out.extend_from_slice(&mom.syy);
        for i in 0..k {
            for j in 0..=i {
                let s: f64 = (0..m).map(|q| self.r[q * k + i] * self.r[q * k + j]).sum();
                out[i * k + j] -= s;
                if i != j {
                    out[j * k + i] = out[i * k + j];
                }
            }
        }
    }

    /// The coefficients at `m` regressors, `B_m = L_m⁻ᵀ R_m`, into `out`
    /// (`m × k`).
    pub(crate) fn coefficients(&self, m: usize, out: &mut Vec<f64>) {
        let (d, k) = (self.d, self.k);
        out.clear();
        out.resize(m * k, 0.0);
        for c in 0..k {
            for i in (0..m).rev() {
                let s: f64 = (i + 1..m).map(|q| self.l[q * d + i] * out[q * k + c]).sum();
                out[i * k + c] = (self.r[i * k + c] - s) / self.l[i * d + i];
            }
        }
    }

    /// `L_m⁻¹`, `m × m` row-major lower, into `out`.
    pub(crate) fn inverse_factor(&self, m: usize, out: &mut Vec<f64>) {
        let d = self.d;
        out.clear();
        out.resize(m * m, 0.0);
        for c in 0..m {
            out[c * m + c] = 1.0 / self.l[c * d + c];
            for i in c + 1..m {
                let s: f64 = (c..i).map(|q| self.l[i * d + q] * out[q * m + c]).sum();
                out[i * m + c] = -s / self.l[i * d + i];
            }
        }
    }
}

/// `(S_m⁻¹)_{ab}` from `L_m⁻¹` (`m × m`): `Σ_{q ≥ max(a, b)} L⁻¹_{qa} L⁻¹_{qb}`.
pub(crate) fn inverse_entry(linv: &[f64], m: usize, a: usize, b: usize) -> f64 {
    (a.max(b)..m)
        .map(|q| linv[q * m + a] * linv[q * m + b])
        .sum()
}

/// Cholesky of a small symmetric `n × n` matrix in place (lower), `false`
/// where a pivot is not positive.
pub(crate) fn cholesky(a: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|q| a[i * n + q] * a[j * n + q]).sum();
            let v = a[i * n + j] - s;
            if i == j {
                if !(v > 0.0) {
                    return false;
                }
                a[i * n + i] = v.sqrt();
            } else {
                a[i * n + j] = v / a[j * n + j];
            }
        }
        for j in i + 1..n {
            a[i * n + j] = 0.0;
        }
    }
    true
}

/// Solve `L Lᵀ X = B` in place for the columns of `b` (`n × t`), `l` a
/// lower factor from [`cholesky`].
pub(crate) fn chol_solve(l: &[f64], n: usize, b: &mut [f64], t: usize) {
    for c in 0..t {
        for i in 0..n {
            let s: f64 = (0..i).map(|q| l[i * n + q] * b[q * t + c]).sum();
            b[i * t + c] = (b[i * t + c] - s) / l[i * n + i];
        }
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|q| l[q * n + i] * b[q * t + c]).sum();
            b[i * t + c] = (b[i * t + c] - s) / l[i * n + i];
        }
    }
}

/// The settings a read needs beside the moments.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    /// Features.
    pub(crate) e: usize,
    /// Series.
    pub(crate) k: usize,
    /// The largest order.
    pub(crate) p: usize,
    pub(crate) intercept: bool,
}

impl Shape {
    /// Regressors at order `q`, the intercept excluded.
    pub(crate) fn m(&self, q: usize) -> usize {
        self.e + self.k * q
    }

    /// Coefficients a target's equation estimates at order `q`, the
    /// intercept included: statsmodels' `df_model`.
    pub(crate) fn params(&self, q: usize) -> usize {
        self.m(q) + usize::from(self.intercept)
    }

    /// The largest order the factor identifies: every pivot of its leading
    /// block passed. `None` where not even the features' block did.
    pub(crate) fn identified(&self, f: &Factor) -> Option<usize> {
        let fit = f.pass.checked_sub(self.e)?;
        Some((fit / self.k).min(self.p))
    }
}

/// The working space a read keeps between rows: not state.
#[derive(Debug, Default, Clone)]
pub(crate) struct Scratch {
    pub(crate) sse: Vec<f64>,
    pub(crate) small: Vec<f64>,
    pub(crate) linv: Vec<f64>,
    pub(crate) b: Vec<f64>,
    pub(crate) g: Vec<f64>,
    pub(crate) x: Vec<f64>,
    pub(crate) mx: Vec<f64>,
    pub(crate) rows: Vec<usize>,
}

/// `ln det(SSE_m)` at `m` regressors, or `None` where the residual
/// co-moments are not positive definite (an exact fit, or a target that
/// is a combination of the others).
fn ln_det_sse(f: &Factor, mom: &Moments, m: usize, s: &mut Scratch) -> Option<f64> {
    let k = f.k;
    f.sse(mom, m, &mut s.sse);
    s.small.clear();
    s.small.extend_from_slice(&s.sse);
    if !cholesky(&mut s.small, k) {
        return None;
    }
    Some((0..k).map(|i| 2.0 * libm::log(s.small[i * k + i])).sum())
}

/// The value of criterion `ic` at order `q`, statsmodels'
/// `VARResults.info_criteria` on Kish's count `n`: with `ld = ln det
/// Σ_mle(q)`, `f = qk² + k(e + 1)` free parameters (the intercept's `k`
/// with one) and `m = kq + e + 1` per equation,
///
/// ```text
/// aic  = ld + 2 f / n
/// bic  = ld + ln(n) f / n
/// hqic = ld + 2 ln(ln n) f / n
/// fpe  = ((n + m) / (n − m))^k e^{ld}      compared as its logarithm
/// ```
///
/// `−∞` where the order fits exactly, which is then the best; `None` where
/// the value is not defined (FPE past the residual degrees of freedom,
/// HQIC at `n <= 1`).
pub(crate) fn criterion(
    ic: VectorArSelect,
    sh: &Shape,
    f: &Factor,
    mom: &Moments,
    q: usize,
    s: &mut Scratch,
) -> Option<f64> {
    let n = mom.n_kish;
    if !(n > 0.0 && n.is_finite()) {
        return None;
    }
    let k = sh.k as f64;
    let free = (q * sh.k * sh.k + sh.k * (sh.e + usize::from(sh.intercept))) as f64;
    let ld = match ln_det_sse(f, mom, sh.m(q), s) {
        Some(ld) => ld,
        None => return Some(f64::NEG_INFINITY),
    };
    let v = match ic {
        VectorArSelect::Aic => ld + 2.0 * free / n,
        VectorArSelect::Bic => ld + libm::log(n) * free / n,
        VectorArSelect::Hqic => {
            let lln = libm::log(libm::log(n));
            ld + 2.0 * lln * free / n
        }
        VectorArSelect::Fpe => {
            let m = sh.params(q) as f64;
            let df = n - m;
            if !(df > 0.0) {
                return None;
            }
            k * libm::log((n + m) / df) + ld
        }
    };
    (!v.is_nan()).then_some(v)
}

/// The order criterion `ic` chooses among `0..=top`, the first of equal
/// values; `None` where no order's value is defined.
pub(crate) fn choose(
    ic: VectorArSelect,
    sh: &Shape,
    f: &Factor,
    mom: &Moments,
    top: usize,
    s: &mut Scratch,
) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for q in 0..=top {
        let Some(v) = criterion(ic, sh, f, mom, q, s) else {
            continue;
        };
        if best.is_none_or(|(b, _)| v < b) {
            best = Some((v, q));
        }
    }
    best.map(|(_, q)| q)
}

/// The residual degrees of freedom at order `q`: Kish's count less the
/// coefficients an equation estimates.
pub(crate) fn df_resid(sh: &Shape, mom: &Moments, q: usize) -> f64 {
    mom.n_kish - sh.params(q) as f64
}

/// The Wald statistic of "the lags `1..=q` of `causing` do not enter the
/// equations of `caused`", from the order-`q` fit: with `R` the lag columns
/// of the causing series, `G = (S_m⁻¹)_RR`, `β = B[R, E]` and
/// `SSE_E = SSE_m[E, E]`,
///
/// ```text
/// λ = df · trace(SSE_E⁻¹ βᵀ G⁻¹ β)
/// ```
///
/// statsmodels' `VARResults.test_causality` statistic before its division
/// by the number of restrictions: the Wald form `(Cb)ᵀ [C Cov(b) Cᵀ]⁻¹ (Cb)`
/// with `Cov(b) = (S⁻¹ / n) ⊗ Σ_u` and `Σ_u = SSE · n / df`. NaN where the
/// degrees of freedom are not positive or a block is not definite.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wald(
    sh: &Shape,
    f: &Factor,
    mom: &Moments,
    q: usize,
    caused: &[usize],
    causing: &[usize],
    s: &mut Scratch,
) -> f64 {
    let df = df_resid(sh, mom, q);
    if q == 0 || !(df > 0.0) {
        return f64::NAN;
    }
    let k = sh.k;
    let mut rows = std::mem::take(&mut s.rows);
    rows.clear();
    rows.extend((0..q).flat_map(|l| causing.iter().map(move |&j| sh.e + l * k + j)));
    let lam = wald_on(sh, f, mom, q, caused, &rows, df, s);
    s.rows = rows;
    lam
}

/// [`wald`] at the lag columns `rows` of the causing series.
#[allow(clippy::too_many_arguments)]
fn wald_on(
    sh: &Shape,
    f: &Factor,
    mom: &Moments,
    q: usize,
    caused: &[usize],
    rows: &[usize],
    df: f64,
    s: &mut Scratch,
) -> f64 {
    let (k, m) = (sh.k, sh.m(q));
    let (nr, ne) = (rows.len(), caused.len());
    f.inverse_factor(m, &mut s.linv);
    f.coefficients(m, &mut s.b);
    s.g.clear();
    s.g.resize(nr * nr, 0.0);
    for a in 0..nr {
        for b in 0..=a {
            let v = inverse_entry(&s.linv, m, rows[a], rows[b]);
            s.g[a * nr + b] = v;
            s.g[b * nr + a] = v;
        }
    }
    if !cholesky(&mut s.g, nr) {
        return f64::NAN;
    }
    // X = G⁻¹ β, `nr × ne`.
    s.x.clear();
    s.x.resize(nr * ne, 0.0);
    for (a, &r) in rows.iter().enumerate() {
        for (c, &e) in caused.iter().enumerate() {
            s.x[a * ne + c] = s.b[r * k + e];
        }
    }
    chol_solve(&s.g, nr, &mut s.x, ne);
    // M = βᵀ X, `ne × ne`.
    s.mx.clear();
    s.mx.resize(ne * ne, 0.0);
    for c1 in 0..ne {
        for c2 in 0..ne {
            s.mx[c1 * ne + c2] = rows
                .iter()
                .enumerate()
                .map(|(a, &r)| s.b[r * k + caused[c1]] * s.x[a * ne + c2])
                .sum();
        }
    }
    f.sse(mom, m, &mut s.sse);
    s.small.clear();
    s.small.resize(ne * ne, 0.0);
    for (c1, &e1) in caused.iter().enumerate() {
        for (c2, &e2) in caused.iter().enumerate() {
            s.small[c1 * ne + c2] = s.sse[e1 * k + e2];
        }
    }
    if !cholesky(&mut s.small, ne) {
        return f64::NAN;
    }
    chol_solve(&s.small, ne, &mut s.mx, ne);
    let trace: f64 = (0..ne).map(|c| s.mx[c * ne + c]).sum();
    df * trace
}

/// Each coefficient's sampling variance at order `q`, statsmodels' `bse²`,
/// into `out`: per target `j`, `[intercept?, regressors…]` padded with NaN
/// to `d` regressors. With `Σ_u = SSE · n / df`, a slope's is
/// `(S_m⁻¹)_ii Σ_u,jj / n` and the intercept's `(1 + z̄ᵀ S_m⁻¹ z̄) Σ_u,jj /
/// n`, each `SSE_jj / df` times the bracket. All NaN where the degrees of
/// freedom are not positive.
pub(crate) fn coef_variances(
    sh: &Shape,
    f: &Factor,
    mom: &Moments,
    q: usize,
    s: &mut Scratch,
    out: &mut Vec<f64>,
) {
    let (k, d, m) = (sh.k, sh.m(sh.p), sh.m(q));
    let i0 = usize::from(sh.intercept);
    let width = i0 + d;
    out.clear();
    out.resize(k * width, f64::NAN);
    let df = df_resid(sh, mom, q);
    if !(df > 0.0) {
        return;
    }
    f.inverse_factor(m, &mut s.linv);
    f.sse(mom, m, &mut s.sse);
    // `‖L⁻¹ z̄‖²`, the intercept's share beside its own `1`.
    let quad: f64 = (0..m)
        .map(|i| {
            let v: f64 = (0..=i).map(|q| s.linv[i * m + q] * mom.zbar[q]).sum();
            v * v
        })
        .sum();
    for j in 0..k {
        let scale = s.sse[j * k + j] / df;
        let row = &mut out[j * width..(j + 1) * width];
        if sh.intercept {
            row[0] = (1.0 + quad) * scale;
        }
        for i in 0..m {
            row[i0 + i] = inverse_entry(&s.linv, m, i, i) * scale;
        }
    }
}
