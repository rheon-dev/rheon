//! The covariance's update in two forms (task 281 (3), the user's decision
//! of 2026-11-20: the square root opt-in, the covariance form the
//! default), and the square root's side of every place the filter writes or
//! reads `P`. A file of its own because `kalman.rs` stands near the
//! source-size cap (`tests/test_repo_hygiene.py`).
//!
//! Each form takes the same row, `pz = P z`, `s = z' P z + r` with `r = R /
//! w`, the gain `g = pz / s`:
//!
//! ```text
//! covariance   P <- P - g pzᵀ                                (O(k²), the default)
//! square root  P = C Cᵀ, C upper triangular, held in place of P:
//!              f = Cᵀ z, α_0 = r, α_j = α_{j-1} + f_j²,
//!              C_·j <- sqrt(α_{j-1} / α_j) C_·j − f_j / sqrt(α_{j-1} α_j) Σ_{i<j} C_·i f_i
//!                                                          (Carlson 1973, O(k²))
//! ```
//!
//! The covariance form subtracts two numbers of `P`'s size to leave one of
//! the posterior's, so at a prior `p0` a row's update rounds at `u p0` where
//! the posterior is of order 1: at `p0 = 1e12`, digits lost to the fourth
//! place. The square root holds `C`, whose entries are of size `sqrt(p0)`,
//! so it rounds at `u sqrt(p0)`. A Joseph form, `P <- (I − g zᵀ) P (I − g
//! zᵀ)ᵀ + r g gᵀ`, was built and measured beside it and dropped: it still
//! holds `P`, so it rounds as the covariance form does, and measured 7 to
//! 10 times further off the posterior at a huge prior, 1.2 to 2.0 times as
//! slow.
//!
//! The factor is stored as `L = Cᵀ`, lower triangular and row-major, so
//! that `C`'s columns, which every update walks, are `L`'s rows, contiguous.
//! The factor's other writes: the transition `C <- Phi_a C` (`Phi_a` is
//! upper triangular, so `C` stays so), a re-map `C <- A C` (`A` too), and
//! each slot's process noise or prior `x xᵀ`, `x` on the slot and the
//! intercept, by a rank-one update of the factor (Givens rotations from the
//! last column to the first, O(k²) for each slot, so O(k³) a row that takes
//! process noise on every slot). The diffuse intercept's first row writes
//! `C`'s row 0, `[sqrt(r), −f_1, …]`, with nothing subtracted.

use serde::{Deserialize, Serialize};

use super::Kalman;

/// Which form `kalman`'s covariance is held and updated in (the module
/// doc of `kalman/forms.rs`): a spec's `covariance_form`, written in a
/// state only where it is not the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceForm {
    /// `P` itself, `P <- P − g (P z)ᵀ`: the default.
    #[default]
    Covariance,
    /// An upper-triangular `C` with `P = C Cᵀ`, by Carlson's update.
    SquareRoot,
}

impl CovarianceForm {
    /// The default, which a state or a spec does not write.
    pub fn is_covariance(&self) -> bool {
        *self == CovarianceForm::Covariance
    }
}

/// `f = Cᵀ z = L z`, `L = Cᵀ` lower triangular (row-major, as every
/// factor here is held).
pub(super) fn factor_f(l: &[f64], z: &[f64], f: &mut [f64]) {
    let k = z.len();
    for (j, fj) in f.iter_mut().enumerate() {
        *fj = l[j * k..j * k + j + 1]
            .iter()
            .zip(z)
            .map(|(a, b)| a * b)
            .sum();
    }
}

/// `pz = C f = Lᵀ f`.
pub(super) fn factor_cf(l: &[f64], f: &[f64], out: &mut [f64]) {
    let k = f.len();
    out.fill(0.0);
    for (j, fj) in f.iter().enumerate() {
        for (o, lji) in out.iter_mut().zip(&l[j * k..j * k + j + 1]) {
            *o += lji * fj;
        }
    }
}

/// Carlson's measurement update of the factor (the module doc), `f = Cᵀ z`
/// read before it, `r > 0`: `C`'s column `j` is `L`'s row `j`; `e` a
/// scratch of `k`.
pub(super) fn carlson(l: &mut [f64], f: &[f64], r: f64, e: &mut [f64]) {
    let k = f.len();
    e.fill(0.0);
    let mut alpha = r;
    for j in 0..k {
        let fj = f[j];
        let next = alpha + fj * fj;
        let a = (alpha / next).sqrt();
        let b = fj / (alpha * next).sqrt();
        for (lji, ei) in l[j * k..j * k + j + 1].iter_mut().zip(e.iter_mut()) {
            let c = *lji;
            *lji = a * c - b * *ei;
            *ei += c * fj;
        }
        alpha = next;
    }
}

/// `C Cᵀ <- C Cᵀ + x xᵀ`, by Givens rotations of `C`'s columns (`L`'s
/// rows) against `x` from the last to the first; `x` is consumed. Columns
/// past `x`'s last entry that is not 0 are left alone.
pub(super) fn factor_add(l: &mut [f64], x: &mut [f64]) {
    let k = x.len();
    let Some(top) = x.iter().rposition(|v| *v != 0.0) else {
        return;
    };
    for j in (0..=top).rev() {
        let (ljj, xj) = (l[j * k + j], x[j]);
        if xj == 0.0 {
            continue;
        }
        let h = (ljj * ljj + xj * xj).sqrt();
        let (cs, sn) = (ljj / h, xj / h);
        for (lji, xi) in l[j * k..j * k + j + 1].iter_mut().zip(x.iter_mut()) {
            let (c, v) = (*lji, *xi);
            *lji = cs * c + sn * v;
            *xi = cs * v - sn * c;
        }
    }
}

/// `P = C Cᵀ = Lᵀ L`.
pub(super) fn factor_product(l: &[f64], k: usize) -> Vec<f64> {
    let mut p = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let v: f64 = (j..k).map(|r| l[r * k + i] * l[r * k + j]).sum();
            p[i * k + j] = v;
            p[j * k + i] = v;
        }
    }
    p
}

impl Kalman {
    /// The form the covariance is held in.
    pub fn covariance_form(&self) -> CovarianceForm {
        self.form
    }

    /// This filter, its covariance held and updated in `form` (the module
    /// doc of `kalman/forms.rs`). Only before its first row: a covariance
    /// that has taken a row is not converted.
    pub fn with_covariance_form(mut self, form: CovarianceForm) -> Result<Self, String> {
        let k = self.cfg.k_total();
        if self.p.iter().any(|p| {
            (0..k * k).any(|ij| ij / k != ij % k && p[ij] != 0.0)
                || (0..k).any(|i| p[i * k + i] < 0.0)
        }) {
            return Err("kalman: the covariance's form is set before its first row".into());
        }
        if form == CovarianceForm::SquareRoot && self.form != CovarianceForm::SquareRoot {
            for p in &mut self.p {
                for i in 0..k {
                    p[i * k + i] = p[i * k + i].sqrt();
                }
            }
        }
        self.form = form;
        Ok(self)
    }

    fn factor(&self) -> bool {
        self.form == CovarianceForm::SquareRoot
    }

    /// Whether `P_ii` of covariance `pi` is not 0: the entry, or on the
    /// factor `C`'s row `i` (`L`'s column), read until an entry is not 0.
    pub(super) fn sized(&self, pi: usize, i: usize) -> bool {
        let k = self.cfg.k_total();
        let p = &self.p[pi];
        if self.factor() {
            (i..k).any(|r| p[r * k + i] != 0.0)
        } else {
            p[i * k + i] != 0.0
        }
    }

    /// Covariance `pi`'s `P`: held, or formed from the factor.
    pub(super) fn cov(&self, pi: usize) -> std::borrow::Cow<'_, [f64]> {
        if self.factor() {
            std::borrow::Cow::Owned(factor_product(&self.p[pi], self.cfg.k_total()))
        } else {
            std::borrow::Cow::Borrowed(&self.p[pi])
        }
    }

    /// `pz = P z` for covariance `pi` against `zs`, and `z' P z`: the
    /// factor's `f = Cᵀ z` kept for its update.
    pub(super) fn read_pz(&mut self, pi: usize) -> f64 {
        if self.factor() {
            factor_f(&self.p[pi], &self.zs, &mut self.fbuf);
            factor_cf(&self.p[pi], &self.fbuf, &mut self.pz);
            return self.fbuf.iter().map(|v| v * v).sum();
        }
        super::matvec(&self.p[pi], &self.zs, &mut self.pz);
        self.zs.iter().zip(&self.pz).map(|(z, p)| z * p).sum()
    }

    /// Covariance `pi` takes the row whose `pz`, gain and `s` the step
    /// holds, `r = R / w`: the diffuse intercept's exact step where it is
    /// `open`, else the form's update.
    pub(super) fn take(&mut self, pi: usize, open: bool, s: f64, r: f64) {
        let k = self.cfg.k_total();
        match (self.form, open) {
            (CovarianceForm::SquareRoot, true) => {
                // `C`'s row 0, `L`'s column 0.
                let l = &mut self.p[pi];
                l[0] = r.sqrt();
                for j in 1..k {
                    l[j * k] = -self.fbuf[j];
                }
            }
            (_, true) => super::diffuse_step(&mut self.p[pi], &self.pz, s),
            (CovarianceForm::Covariance, false) => {
                super::take_the_row(&mut self.p[pi], &self.gain, &self.pz)
            }
            (CovarianceForm::SquareRoot, false) => {
                carlson(&mut self.p[pi], &self.fbuf, r, &mut self.vbuf)
            }
        }
    }

    /// `P += v x xᵀ`-style writes of one slot `i`: `P_ii += d`, and with the
    /// intercept's coupling `cc`, `P_00 += cc² v` and `P_0i += −cc v / a`
    /// (`d = v / a²`), entry by entry; on the factor, the rank-one update
    /// by `x = sqrt(v) (e_i / a − cc e_0)`.
    pub(super) fn add_slot(&mut self, pi: usize, i: usize, v: f64, a: f64, cc: Option<f64>) {
        let k = self.cfg.k_total();
        if self.factor() {
            let mut x = std::mem::take(&mut self.xbuf);
            x.clear();
            x.resize(k, 0.0);
            let sv = v.sqrt();
            x[i] = sv / a;
            if let Some(cc) = cc {
                x[0] = -cc * sv;
            }
            factor_add(&mut self.p[pi], &mut x);
            self.xbuf = x;
            return;
        }
        let p = &mut self.p[pi];
        if let Some(cc) = cc {
            p[0] += cc * cc * v;
            let o = -cc * v / a;
            p[i] += o;
            p[i * k] += o;
        }
        p[i * k + i] += v / (a * a);
    }

    /// The process noise on the factor: [`Kalman::add_process_noise`]'s
    /// terms, a rank-one update for each slot.
    pub(super) fn factor_noise(&mut self, pi: usize, sigma2: f64, d: f64) {
        let off = usize::from(self.cfg.fit_intercept);
        let mut q = std::mem::take(&mut self.qbuf);
        self.q_held_into(sigma2, &mut q);
        let open = self.open(pi);
        for (qi, g) in q.iter_mut().zip(self.gap_factors.of(&self.cfg, d)) {
            *qi *= g;
        }
        for (i, &qd) in q.iter().enumerate() {
            if qd == 0.0 {
                continue;
            }
            if !self.cfg.standardize {
                if !(open && i < off) {
                    self.add_slot(pi, i, qd, 1.0, None);
                }
                continue;
            }
            if !self.sized(pi, i) {
                continue;
            }
            if i < off {
                self.add_slot(pi, i, qd, 1.0, None);
                continue;
            }
            let a = self.map_a[i];
            let cc = (off == 1).then(|| self.map_c[i] / a);
            self.add_slot(pi, i, qd, a, cc);
        }
        self.qbuf = q;
    }
}

/// `C <- Phi_a C` (`Phi_a` the transition, upper triangular): `C`'s row
/// 0 `phi_0 C_0· + Σ_i g_i C_i·` from the rows before the row, its row `i`
/// `phi_i C_i·`; on `L`, row by row.
pub(super) fn factor_transit(l: &mut [f64], phi: &[f64], g: Option<&[f64]>) {
    let k = phi.len();
    for j in 0..k {
        let row = &mut l[j * k..j * k + j + 1];
        if let Some(g) = g {
            let mut v = phi[0] * row[0];
            for i in 1..=j {
                v += g[i] * row[i];
            }
            row[0] = v;
        } else {
            row[0] *= phi[0];
        }
        for (v, ph) in row[1..].iter_mut().zip(&phi[1..]) {
            *v *= ph;
        }
    }
}

/// The re-map `C <- A C` (`A_00 = 1`, `A_0i = c_i`, `A_ii = a_i`, upper
/// triangular): `C`'s row 0 `C_0· + Σ_l c_l C_l·` into `u` (`L`'s column
/// 0), and whether it and every `a_i C_i·` are numbers; with `apply`,
/// written into the factor.
pub(super) fn factor_remap(
    l: &mut [f64],
    u: &mut [f64],
    a: &[f64],
    cs: &[f64],
    off: usize,
    apply: bool,
) -> bool {
    let k = a.len();
    let mut finite = true;
    for j in 0..k {
        let row = &mut l[j * k..j * k + j + 1];
        if off == 1 {
            let mut v = row[0];
            for i in 1..=j {
                v += cs[i] * row[i];
            }
            u[j] = v;
            finite &= v.is_finite();
            if apply {
                row[0] = v;
            }
        }
        for (v, ai) in row[off..].iter_mut().zip(&a[off..]) {
            finite &= (*v * ai).is_finite();
            if apply {
                *v *= ai;
            }
        }
    }
    finite
}
