//! The estimator of one target's parameters in a [`super::Sarima`] whose
//! parameters are not given: the buffer before its start, the start's fit
//! and its gate, and the recursive prediction-error step after it.
//!
//! # The recursion
//!
//! With `z_t = w_t - xw_t' beta` (the differenced value less its
//! regressors' part) and the reduced polynomials `A(B) = phi(B) Phi(B^s)`,
//! `C(B) = theta(B) Theta(B^s)`, the conditional (zero pre-sample) form
//! predicts and takes its residual
//!
//! ```text
//! ẑ_t = - sum_{i>=1} A_i z_{t-i} + sum_{j>=1} C_j e_{t-j},     ŵ_t = xw_t' beta + ẑ_t
//! e_t = w_t - ŵ_t
//! ```
//!
//! and `psi_t = -de_t/dpar` comes from filtered signals at the current
//! parameters, each a ring (`x / C(B)` is `xf_t = x_t - sum_j C_j xf_{t-j}`):
//!
//! ```text
//! u = Phi(B^s) z / C(B)     psi_{phi_i}   = u_{t-i}
//! v = phi(B) z / C(B)       psi_{Phi_i}   = v_{t-s i}
//! r = Theta(B^s) e / C(B)   psi_{theta_j} = r_{t-j}
//! g = theta(B) e / C(B)     psi_{Theta_j} = g_{t-s j}
//! f = A(B) xw / C(B)        psi_{beta_m}  = f_{m,t}
//! ```
//!
//! At fixed parameters these are the exact derivatives of the conditional
//! sum of squares (`tests.rs` holds them to a central difference of it).
//! A slot not observed takes `z_t = ẑ_t` and `e_t = 0`, its expectation;
//! a regressor not known (a skipped slot's, or one the differencing cannot
//! form) enters the signals as 0, and the slot is not observed.
//!
//! # The start, and its gate
//!
//! Each slot past the differencing's fill is buffered, observed or not, as
//! `(w_t, weight, xw_t)`, so a missing slot stays a gap. Nothing is fitted
//! until Kish's count of the observed slots, `(sum w)^2 / sum w^2`, passes
//! the structural minimum: the parameters (the regressors', the four
//! polynomials' and `sigma2`) plus `d + sD`. Past it the buffer's
//! conditional Gaussian likelihood, `sigma2` concentrated out,
//!
//! ```text
//! l(par) = -W/2 ln(S/W),   S = sum w_t e_t^2,   W = sum w_t
//! ```
//!
//! is maximized in the unconstrained coordinates `SARIMAX` optimizes
//! through (its stationarity and invertibility transforms, `transform`),
//! by the projected Newton `garch` holds, from a small grid. The fit is
//! accepted only when all three hold, read in this order: it sits on no
//! bound (where a transform is on, every partial autocorrelation at most
//! `1 - 1e-3` in size; a fit drifting to a bound has no optimum inside the
//! region to converge to, so the bound is the reason it names); it
//! converged (its Newton decrement at most `1e-6` at Kish's count of
//! rows); and its information is conditioned (each standard error finite
//! and at most the spec's `se_bound`, [`SE_BOUND`] by default, in units
//! free of the target's: a polynomial's coefficient as it is, a
//! regressor's coefficient times its regressor's standard deviation over
//! the innovations', and `sigma2` relative to itself, `se / sigma2`). Otherwise the target keeps buffering, naming the condition
//! that failed, and the fit is tried again once Kish's count has grown by
//! [`RETRY_GROWTH`]. On acceptance the buffer is replayed at the fit with
//! the parameters held, which builds the rings, `R` and the weight `V`,
//! and is dropped.
//!
//! # The step
//!
//! Each observed row of weight `w` after the start (Ljung and Söderström's
//! recursive prediction error, Gauss-Newton), in the unconstrained
//! coordinates `x` with `J = dpar/dx`:
//!
//! ```text
//! V' = V + w,   a = w / V'
//! R' = R + a (J'psi psi'J - R)
//! x' = x + a R'^{-1} J'psi e
//! sigma2' = sigma2 + a (e^2 - sigma2)
//! ```
//!
//! and the residual the rings keep is the a-posteriori one, at the new
//! parameters. `sigma2` is the conditional likelihood's (the mean square
//! of the innovations), not `SARIMAX`'s exact likelihood's parameter; the
//! two differ by the start's terms, which fade as `1/n`.

// Index loops over a matrix's rows and columns read plainest as indices.
#![allow(clippy::needless_range_loop)]

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::poly::{lag_poly, polymul};
use super::transform::{constrain, partials};
use crate::garch::linalg::{chol_solve, lu_solve};
use crate::garch::optim::{Eval, maximize};

/// A fit's Newton decrement at Kish's count of rows, in nats, at most
/// this: it is at its optimum (`garch`'s cut, six orders above the
/// rounding's decrement and five below a stop short of an optimum).
pub(crate) const SOUND_DECREMENT: f64 = 1e-6;

/// A partial autocorrelation at least `1 - BOUND_MARGIN` in size is on the
/// stationarity (invertibility) bound.
pub(crate) const BOUND_MARGIN: f64 = 1e-3;

/// The default largest standard error a start is accepted with (the
/// spec's `se_bound`), in standardized units (a polynomial's coefficient
/// as it is, `sigma2`'s relative to itself). It decides how good a
/// start the recursion steps from, and so where it ends: measured over
/// five specs and eight seeds each, 10,000 rows, the worst coefficient's
/// gap at the last row from `SARIMAX`'s batch fit of the same rows, in its
/// standard errors, and the rows the gate opened at:
///
/// ```text
/// bound   (0,1,1)(0,1,1,7)   (1,0,1)(1,0,1,24)   (1,1,1)(1,1,0,24)   (1,1,1)(1,1,1,24)   (1,0,1)
/// 0.1     148-323, 0.23      548-822, 0.28       255-370, 0.24       6240-9348, 0.38     243-548, 0.20
/// 0.15    70-148, 0.26       243-548, 0.25       127-178, 1.24       1866-6240, 4.48     108-243, 0.29
/// 0.2     49-101, 0.25       162-365, 0.74       70-127, 1.17        267-4168, 8.16      48-108, 0.51
/// ```
///
/// At 0.1 every start lands within 0.4 standard errors; above it the
/// `(1,1,1)(1,1,1,24)` spec, whose seasonal factors nearly cancel
/// (`1 - 0.4 B^24` against `1 - 0.5 B^24`), starts from fits the recursion
/// does not recover from. Its wait is the price: 6,240 to 9,348 rows,
/// whose buffer is 18 bytes a slot without regressors.
pub(crate) const SE_BOUND: f64 = 0.1;

/// Kish's count must grow by this factor between two fits of the start,
/// so a long wait costs a number of fits logarithmic in its length.
pub(crate) const RETRY_GROWTH: f64 = 1.5;

/// Why a target reports nothing: the stage it is at, in the order a
/// target passes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SarimaWait {
    /// The differencing's ring is not yet full: no differenced value.
    Filling,
    /// Kish's count of the buffer is not past the parameters and the
    /// differencing.
    Buffering,
    /// The start's fit did not reach its optimum.
    NotConverged,
    /// The start's fit sits on a stationarity or invertibility bound.
    OnBound,
    /// The start's fit has a standard error past `se_bound`.
    NotIdentified,
}

impl SarimaWait {
    /// The reason as a row names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Filling => "filling the differencing: no differenced value yet",
            Self::Buffering => {
                "buffering: Kish's count not past the parameters and the differencing"
            }
            Self::NotConverged => "start not converged",
            Self::OnBound => "start on a bound",
            Self::NotIdentified => "start not identified: standard errors past the bound",
        }
    }
}

/// The orders, the season and the regressors: what fixes the layout of
/// the parameters, `(beta, phi, theta, Phi, Theta)` in that order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    pub k: usize,
    pub p: usize,
    pub q: usize,
    pub sp: usize,
    pub sq: usize,
    pub s: usize,
    pub stationary: bool,
    pub invertible: bool,
}

impl Shape {
    /// The parameters stepped: every one but `sigma2`.
    pub(crate) fn n(&self) -> usize {
        self.k + self.p + self.q + self.sp + self.sq
    }

    /// The rings' length: the largest lag any of them is read at.
    pub(crate) fn lags(&self) -> usize {
        (self.p + self.s * self.sp)
            .max(self.q + self.s * self.sq)
            .max(1)
    }

    /// The parameters and their Jacobian `dpar/dx`, row-major `n x n`
    /// (block diagonal), from the unconstrained `x`.
    pub(crate) fn params(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.n();
        let mut par = x.to_vec();
        let mut jac = vec![0.0; n * n];
        let mut at = 0;
        let blocks = [
            (self.k, None),
            (self.p, self.stationary.then_some(true)),
            (self.q, self.invertible.then_some(false)),
            (self.sp, self.stationary.then_some(true)),
            (self.sq, self.invertible.then_some(false)),
        ];
        for (len, form) in blocks {
            match form {
                Some(ar) if len > 0 => {
                    let (c, d) = constrain(&x[at..at + len], ar);
                    par[at..at + len].copy_from_slice(&c);
                    for i in 0..len {
                        for m in 0..len {
                            jac[(at + i) * n + at + m] = d[i * len + m];
                        }
                    }
                }
                _ => {
                    for i in at..at + len {
                        jac[i * n + i] = 1.0;
                    }
                }
            }
            at += len;
        }
        (par, jac)
    }

    /// Whether `x` puts a transformed polynomial on its bound.
    pub(crate) fn on_bound(&self, x: &[f64]) -> bool {
        let mut at = self.k;
        let mut hit = false;
        for (len, on) in [
            (self.p, self.stationary),
            (self.q, self.invertible),
            (self.sp, self.stationary),
            (self.sq, self.invertible),
        ] {
            if on && partials(&x[at..at + len]).any(|r| r.abs() >= 1.0 - BOUND_MARGIN) {
                hit = true;
            }
            at += len;
        }
        hit
    }
}

/// The polynomials of a set of parameters: each factor and the two
/// products, coefficients from `B^0`.
#[derive(Debug, Clone)]
pub(crate) struct Polys {
    beta: Vec<f64>,
    a: Vec<f64>,
    c: Vec<f64>,
    phi: Vec<f64>,
    sphi: Vec<f64>,
    theta: Vec<f64>,
    stheta: Vec<f64>,
}

impl Polys {
    pub(crate) fn new(shape: &Shape, par: &[f64]) -> Self {
        let Shape { k, p, q, sp, s, .. } = *shape;
        let s = s.max(1);
        let beta = par[..k].to_vec();
        let phi = lag_poly(&par[k..k + p], 1, -1.0);
        let theta = lag_poly(&par[k + p..k + p + q], 1, 1.0);
        let sphi = lag_poly(&par[k + p + q..k + p + q + sp], s, -1.0);
        let stheta = lag_poly(&par[k + p + q + sp..], s, 1.0);
        Self {
            beta,
            a: polymul(&phi, &sphi),
            c: polymul(&theta, &stheta),
            phi,
            sphi,
            theta,
            stheta,
        }
    }
}

/// The value `lag` slots back in a ring of length `L` (`1 <= lag <= L`).
#[inline]
fn back(ring: &VecDeque<f64>, lag: usize) -> f64 {
    ring[ring.len() - lag]
}

/// `sum_{i>=0} poly_i x_{t-i}` with `x_t = now` and the rest from `ring`.
fn apply(poly: &[f64], ring: &VecDeque<f64>, now: f64) -> f64 {
    let mut acc = poly[0] * now;
    for (i, c) in poly.iter().enumerate().skip(1) {
        if *c != 0.0 {
            acc += c * back(ring, i);
        }
    }
    acc
}

/// `x_t - sum_{j>=1} C_j xf_{t-j}`: one step of the recursion `x / C(B)`.
fn over_c(c: &[f64], x: f64, ring: &VecDeque<f64>) -> f64 {
    let mut acc = x;
    for (j, cj) in c.iter().enumerate().skip(1) {
        if *cj != 0.0 {
            acc -= cj * back(ring, j);
        }
    }
    acc
}

fn push(ring: &mut VecDeque<f64>, v: f64) {
    ring.pop_front();
    ring.push_back(v);
}

/// The conditional recursion's past: each signal's last `L` values,
/// oldest first, zero before the first slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Rings {
    z: VecDeque<f64>,
    e: VecDeque<f64>,
    u: VecDeque<f64>,
    v: VecDeque<f64>,
    r: VecDeque<f64>,
    g: VecDeque<f64>,
    /// Per regressor, its differenced value and its filtered one.
    xw: Vec<VecDeque<f64>>,
    xf: Vec<VecDeque<f64>>,
}

/// One slot through the recursion: where the slot is observed, its
/// residual (`psi` is filled beside it, in the parameters' coordinates).
pub(crate) struct Pass {
    pub e: Option<f64>,
}

impl Rings {
    pub(crate) fn new(shape: &Shape) -> Self {
        let l = shape.lags();
        let zero = || VecDeque::from(vec![0.0; l]);
        Self {
            z: zero(),
            e: zero(),
            u: zero(),
            v: zero(),
            r: zero(),
            g: zero(),
            xw: (0..shape.k).map(|_| zero()).collect(),
            xf: (0..shape.k).map(|_| zero()).collect(),
        }
    }

    /// The doubles the rings hold.
    pub(crate) fn len(shape: &Shape) -> usize {
        shape.lags() * (6 + 2 * shape.k)
    }

    pub(crate) fn shaped(&self, shape: &Shape) -> bool {
        let l = shape.lags();
        [&self.z, &self.e, &self.u, &self.v, &self.r, &self.g]
            .iter()
            .all(|r| r.len() == l)
            && self.xw.len() == shape.k
            && self.xf.len() == shape.k
            && self.xw.iter().chain(&self.xf).all(|r| r.len() == l)
    }

    /// `ẑ_t`, from the rings.
    fn z_hat(&self, pol: &Polys) -> f64 {
        let mut acc = 0.0;
        for (i, ai) in pol.a.iter().enumerate().skip(1) {
            if *ai != 0.0 {
                acc -= ai * back(&self.z, i);
            }
        }
        for (j, cj) in pol.c.iter().enumerate().skip(1) {
            if *cj != 0.0 {
                acc += cj * back(&self.e, j);
            }
        }
        acc
    }

    /// `xw_t' beta`, NaN where `xw` is not known.
    fn exog(pol: &Polys, xw: Option<&[f64]>) -> f64 {
        match xw {
            Some(x) => x.iter().zip(&pol.beta).map(|(x, b)| x * b).sum(),
            None if pol.beta.is_empty() => 0.0,
            None => f64::NAN,
        }
    }

    /// The slot's prediction `ŵ_t` (NaN where its regressors are not
    /// known).
    pub(crate) fn w_hat(&self, pol: &Polys, xw: Option<&[f64]>) -> f64 {
        Self::exog(pol, xw) + self.z_hat(pol)
    }

    /// `f_{m,t}` for each regressor, its value `0` where not known.
    fn xf_now(&self, pol: &Polys, xw: Option<&[f64]>, out: &mut Vec<f64>) {
        out.clear();
        for m in 0..self.xw.len() {
            let x = xw.map_or(0.0, |x| x[m]);
            out.push(over_c(&pol.c, apply(&pol.a, &self.xw[m], x), &self.xf[m]));
        }
    }

    /// `psi_t`, the parameters' order, from the rings and `xf_t`.
    fn psi(&self, shape: &Shape, xf: &[f64], out: &mut Vec<f64>) {
        let s = shape.s.max(1);
        out.clear();
        out.extend_from_slice(xf);
        out.extend((1..=shape.p).map(|i| back(&self.u, i)));
        out.extend((1..=shape.q).map(|j| back(&self.r, j)));
        out.extend((1..=shape.sp).map(|i| back(&self.v, s * i)));
        out.extend((1..=shape.sq).map(|j| back(&self.g, s * j)));
    }

    /// The slot's values into the rings, at `pol`: `z_t`, `e_t` and the
    /// signals they make.
    fn keep(&mut self, pol: &Polys, z: f64, e: f64, xw: Option<&[f64]>, xf: &[f64]) {
        let u = over_c(&pol.c, apply(&pol.sphi, &self.z, z), &self.u);
        let v = over_c(&pol.c, apply(&pol.phi, &self.z, z), &self.v);
        let r = over_c(&pol.c, apply(&pol.stheta, &self.e, e), &self.r);
        let g = over_c(&pol.c, apply(&pol.theta, &self.e, e), &self.g);
        push(&mut self.z, z);
        push(&mut self.e, e);
        push(&mut self.u, u);
        push(&mut self.v, v);
        push(&mut self.r, r);
        push(&mut self.g, g);
        for m in 0..self.xw.len() {
            push(&mut self.xw[m], xw.map_or(0.0, |x| x[m]));
            push(&mut self.xf[m], xf[m]);
        }
    }

    /// One slot at fixed parameters: `w` the observed differenced value
    /// (`None` where not observed), `xw` its regressors (`None` where not
    /// known). Fills `psi` where the slot is observed.
    pub(crate) fn pass(
        &mut self,
        shape: &Shape,
        pol: &Polys,
        w: Option<f64>,
        xw: Option<&[f64]>,
        xf: &mut Vec<f64>,
        psi: &mut Vec<f64>,
    ) -> Pass {
        let z_hat = self.z_hat(pol);
        let w_hat = Self::exog(pol, xw) + z_hat;
        self.xf_now(pol, xw, xf);
        let e = w.filter(|_| w_hat.is_finite()).map(|w| w - w_hat);
        if e.is_some() {
            self.psi(shape, xf, psi);
        }
        let z = match e {
            Some(e) => z_hat + e,
            None => z_hat,
        };
        self.keep(pol, z, e.unwrap_or(0.0), xw, xf);
        Pass { e }
    }
}

/// The buffer's slots: `w` NaN where not observed, `xw` NaN where not
/// known, `k` a slot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Buffer {
    pub w: Vec<f64>,
    pub weight: Vec<f64>,
    pub xw: Vec<f64>,
}

impl Buffer {
    fn slots(&self) -> usize {
        self.w.len()
    }

    fn slot(&self, t: usize, k: usize) -> (Option<f64>, f64, Option<&[f64]>) {
        let w = self.w[t];
        let x = &self.xw[t * k..(t + 1) * k];
        let xw = (!x.iter().any(|v| v.is_nan())).then_some(x);
        (
            (!w.is_nan() && self.weight[t] > 0.0).then_some(w),
            self.weight[t],
            xw,
        )
    }
}

/// The buffer's conditional likelihood at `par`: its sums, and the rings
/// it leaves.
pub(crate) struct Sums {
    /// `sum w e^2`.
    pub s: f64,
    /// `sum w`, `sum w^2`, over the observed slots.
    pub w: f64,
    pub w2: f64,
    /// `sum w e psi`.
    pub g: Vec<f64>,
    /// `sum w psi psi'`, row-major.
    pub h: Vec<f64>,
    rings: Rings,
}

/// The derivatives of the conditional recursion in each parameter, carried
/// forward exactly (forward-mode): `dz` and `de` a parameter, each a ring
/// like the recursion's. From `e_t = z_t + sum_i A_i z_{t-i} - sum_j C_j
/// e_{t-j}`, with `S = sum_i (A_i dz_{t-i} + dA_i z_{t-i}) - sum_j (C_j
/// de_{t-j} + dC_j e_{t-j})`: an observed slot has `dz_t = -xw_t` for a
/// regressor's coefficient (0 otherwise) and `de_t = dz_t + S`; a slot not
/// observed has `e_t = 0`, so `de_t = 0` and `dz_t = -S`. Where no slot is
/// missing, `-de_t` is the filtered signals' `psi_t`; with missing slots it
/// also carries what a filled `z_t` owes the parameters, which the signals
/// leave out, so the start's fit reads the exact gradient of its objective.
struct Sens {
    da: Vec<Vec<f64>>,
    dc: Vec<Vec<f64>>,
    dz: Vec<VecDeque<f64>>,
    de: Vec<VecDeque<f64>>,
}

impl Sens {
    fn new(shape: &Shape, pol: &Polys) -> Self {
        let (n, l, s) = (shape.n(), shape.lags(), shape.s.max(1));
        let shifted = |poly: &[f64], by: usize, sign: f64, len: usize| {
            let mut out = vec![0.0; len];
            for (i, c) in poly.iter().enumerate() {
                if i + by < len {
                    out[i + by] = sign * c;
                }
            }
            out
        };
        let (la, lc) = (pol.a.len(), pol.c.len());
        let mut da = vec![vec![0.0; la]; n];
        let mut dc = vec![vec![0.0; lc]; n];
        let mut at = shape.k;
        for i in 1..=shape.p {
            da[at] = shifted(&pol.sphi, i, -1.0, la);
            at += 1;
        }
        for j in 1..=shape.q {
            dc[at] = shifted(&pol.stheta, j, 1.0, lc);
            at += 1;
        }
        for i in 1..=shape.sp {
            da[at] = shifted(&pol.phi, s * i, -1.0, la);
            at += 1;
        }
        for j in 1..=shape.sq {
            dc[at] = shifted(&pol.theta, s * j, 1.0, lc);
            at += 1;
        }
        let zero = || VecDeque::from(vec![0.0; l]);
        Self {
            da,
            dc,
            dz: (0..n).map(|_| zero()).collect(),
            de: (0..n).map(|_| zero()).collect(),
        }
    }

    /// One slot, read before `rings` take it: `psi` filled where it is
    /// observed.
    fn step(
        &mut self,
        k: usize,
        pol: &Polys,
        rings: &Rings,
        observed: bool,
        xw: Option<&[f64]>,
        psi: &mut Vec<f64>,
    ) {
        psi.clear();
        for m in 0..self.dz.len() {
            let mut s = 0.0;
            for (i, ai) in pol.a.iter().enumerate().skip(1) {
                s += ai * back(&self.dz[m], i) + self.da[m][i] * back(&rings.z, i);
            }
            for (j, cj) in pol.c.iter().enumerate().skip(1) {
                s -= cj * back(&self.de[m], j) + self.dc[m][j] * back(&rings.e, j);
            }
            let (dz, de) = if observed {
                let dz = match xw {
                    Some(x) if m < k => -x[m],
                    _ => 0.0,
                };
                (dz, dz + s)
            } else {
                (-s, 0.0)
            };
            if observed {
                psi.push(-de);
            }
            push(&mut self.dz[m], dz);
            push(&mut self.de[m], de);
        }
    }
}

/// The buffer's conditional likelihood's sums at `par`, with the exact
/// gradient's where `grad`, and the rings it leaves.
pub(crate) fn run(shape: &Shape, buf: &Buffer, par: &[f64], grad: bool) -> Sums {
    let n = shape.n();
    let pol = Polys::new(shape, par);
    let mut rings = Rings::new(shape);
    let mut sens = grad.then(|| Sens::new(shape, &pol));
    let (mut xf, mut psi, mut unused) = (
        Vec::with_capacity(shape.k),
        Vec::with_capacity(n),
        Vec::new(),
    );
    let mut out = Sums {
        s: 0.0,
        w: 0.0,
        w2: 0.0,
        g: vec![0.0; n],
        h: vec![0.0; n * n],
        rings: Rings::new(shape),
    };
    for t in 0..buf.slots() {
        let (w, wt, xw) = buf.slot(t, shape.k);
        if let Some(sens) = sens.as_mut() {
            let observed = w.is_some() && rings.w_hat(&pol, xw).is_finite();
            sens.step(shape.k, &pol, &rings, observed, xw, &mut psi);
        }
        let pass = rings.pass(shape, &pol, w, xw, &mut xf, &mut unused);
        let Some(e) = pass.e else { continue };
        out.s += wt * e * e;
        out.w += wt;
        out.w2 += wt * wt;
        if grad {
            for i in 0..n {
                out.g[i] += wt * e * psi[i];
                for j in 0..n {
                    out.h[i * n + j] += wt * psi[i] * psi[j];
                }
            }
        }
    }
    out.rings = rings;
    out
}

/// `J' M J` for a row-major `n x n` `M` and `J`.
fn congruent(m: &[f64], j: &[f64], n: usize) -> Vec<f64> {
    let mut mj = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            mj[a * n + b] = (0..n).map(|c| m[a * n + c] * j[c * n + b]).sum();
        }
    }
    let mut out = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            out[a * n + b] = (0..n).map(|c| j[c * n + a] * mj[c * n + b]).sum();
        }
    }
    out
}

/// `J' v`.
fn transposed(j: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|a| (0..n).map(|c| j[c * n + a] * v[c]).sum())
        .collect()
}

/// The concentrated conditional log-likelihood at the unconstrained `x`,
/// its gradient and Gauss-Newton matrix in `x`.
fn objective(shape: &Shape, buf: &Buffer, x: &[f64], grad: bool) -> Eval {
    let n = shape.n();
    let (par, jac) = shape.params(x);
    let sums = run(shape, buf, &par, grad);
    if !(sums.s.is_finite() && sums.s > 0.0 && sums.w > 0.0) {
        return Eval::bad(n);
    }
    let sigma2 = sums.s / sums.w;
    // The `libm` crate's `ln`, the same bits on every platform: the
    // optimizer's line search compares `f` at its rounding, so a platform's
    // last bit could choose its path (as `garch`'s start found).
    let f = -0.5 * sums.w * libm::log(sigma2);
    if !grad {
        return Eval {
            f,
            g: vec![0.0; n],
            h: vec![0.0; n * n],
        };
    }
    let g: Vec<f64> = sums.g.iter().map(|v| v / sigma2).collect();
    let h: Vec<f64> = sums.h.iter().map(|v| v / sigma2).collect();
    let (g, h) = (transposed(&jac, &g, n), congruent(&h, &jac, n));
    if !g.iter().chain(&h).all(|v| v.is_finite()) {
        return Eval::bad(n);
    }
    Eval { f, g, h }
}

/// The start the gate accepted: the unconstrained parameters, `sigma2`,
/// and what the replay built.
pub(crate) struct Accepted {
    pub x: Vec<f64>,
    pub sigma2: f64,
    pub v: f64,
    pub rm: Vec<f64>,
    pub rings: Rings,
}

/// The standard deviation of each regressor and of the observed `w` over
/// the buffer's observed slots with known regressors.
fn spreads(shape: &Shape, buf: &Buffer) -> (Vec<f64>, f64) {
    let k = shape.k;
    let (mut sw, mut m1, mut m2) = (0.0, vec![0.0; k + 1], vec![0.0; k + 1]);
    for t in 0..buf.slots() {
        let (Some(w), wt, Some(xw)) = buf.slot(t, k) else {
            continue;
        };
        sw += wt;
        for (i, v) in xw.iter().chain(std::iter::once(&w)).enumerate() {
            m1[i] += wt * v;
            m2[i] += wt * v * v;
        }
    }
    let sd: Vec<f64> = (0..=k)
        .map(|i| {
            if sw > 0.0 {
                let m = m1[i] / sw;
                (m2[i] / sw - m * m).max(0.0).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    (sd[..k].to_vec(), sd[k])
}

/// The regressors' coefficients by weighted least squares of `w` on `xw`:
/// the grid's start for them (0 where the normal equations are singular).
fn ols(shape: &Shape, buf: &Buffer) -> Vec<f64> {
    let k = shape.k;
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for t in 0..buf.slots() {
        let (Some(w), wt, Some(xw)) = buf.slot(t, k) else {
            continue;
        };
        for i in 0..k {
            b[i] += wt * xw[i] * w;
            for j in 0..k {
                a[i * k + j] += wt * xw[i] * xw[j];
            }
        }
    }
    lu_solve(&a, k, &b).unwrap_or_else(|| vec![0.0; k])
}

/// The grid of unconstrained starts: each polynomial coefficient at `-g`,
/// 0 and `g` (every combination up to four of them, else 0 and each one
/// alone at `±g`), the regressors' at their least squares.
fn grid(shape: &Shape, beta: &[f64]) -> Vec<Vec<f64>> {
    const G: f64 = 0.55;
    let m = shape.n() - shape.k;
    let mut out = Vec::new();
    let with = |arma: Vec<f64>| {
        let mut x = beta.to_vec();
        x.extend(arma);
        x
    };
    if m <= 4 {
        let total = 3usize.pow(m as u32);
        for code in 0..total {
            let mut c = code;
            let arma: Vec<f64> = (0..m)
                .map(|_| {
                    let v = [0.0, -G, G][c % 3];
                    c /= 3;
                    v
                })
                .collect();
            out.push(with(arma));
        }
    } else {
        out.push(with(vec![0.0; m]));
        for i in 0..m {
            for v in [-G, G] {
                let mut arma = vec![0.0; m];
                arma[i] = v;
                out.push(with(arma));
            }
        }
    }
    out
}

/// The start's fit of `buf` and its gate: `Ok` with the accepted start, or
/// `Err` with the condition it failed. `bound` is the largest standard
/// error accepted: an ARMA coefficient's as it is, a regressor's
/// standardized, `sigma2`'s relative to it.
pub(crate) fn fit(shape: &Shape, buf: &Buffer, bound: f64) -> Result<Accepted, SarimaWait> {
    let n = shape.n();
    let (sd_x, sd_w) = spreads(shape, buf);
    let beta = ols(shape, buf);
    let mut units = vec![1.0; n];
    for m in 0..shape.k {
        if sd_x[m] > 0.0 && sd_w > 0.0 {
            units[m] = sd_w / sd_x[m];
        }
    }
    let obj = |x: &[f64], g: bool| objective(shape, buf, x, g);
    let id = |x: &[f64]| x.to_vec();
    let found = maximize(&obj, &id, &grid(shape, &beta), &units).ok_or(SarimaWait::NotConverged)?;
    let x = found.theta;
    let (par, jac) = shape.params(&x);
    let sums = run(shape, buf, &par, true);
    let kish = sums.w * sums.w / sums.w2;
    let scale = kish / sums.w;
    // A fit drifting to a bound has no optimum inside the region to
    // converge to: the bound is the reason it names.
    if shape.on_bound(&x) {
        return Err(SarimaWait::OnBound);
    }
    let sound = found.decrement * scale <= SOUND_DECREMENT;
    if !sound {
        return Err(SarimaWait::NotConverged);
    }
    let sigma2 = sums.s / sums.w;
    // The information per unit weight in the parameters, `sum w psi psi' /
    // W / sigma2`; a parameter's variance at Kish's count of rows is its
    // inverse's diagonal over that count.
    let info: Vec<f64> = sums.h.iter().map(|v| v / sums.w / sigma2).collect();
    let mut se_ok = sigma2.is_finite() && sigma2 > 0.0;
    for i in 0..n {
        let mut e = vec![0.0; n];
        e[i] = 1.0;
        let col = lu_solve(&info, n, &e);
        let var = col.map_or(f64::INFINITY, |c| c[i] / kish);
        let unit = if i < shape.k {
            sd_x[i] / sigma2.sqrt()
        } else {
            1.0
        };
        let se = var.sqrt() * unit;
        if !(se.is_finite() && se <= bound) {
            se_ok = false;
        }
    }
    // `sigma2`'s relative standard error at Kish's count, `se / sigma2`,
    // from the fourth moment of the residuals: free of the target's units.
    let (mut m4, mut w) = (0.0, 0.0);
    {
        let pol = Polys::new(shape, &par);
        let mut rings = Rings::new(shape);
        let (mut xf, mut psi) = (Vec::new(), Vec::new());
        for t in 0..buf.slots() {
            let (wo, wt, xw) = buf.slot(t, shape.k);
            if let Some(e) = rings.pass(shape, &pol, wo, xw, &mut xf, &mut psi).e {
                let d = e * e - sigma2;
                m4 += wt * d * d;
                w += wt;
            }
        }
    }
    let rel = (m4 / w / kish).sqrt() / sigma2;
    if !(se_ok && rel.is_finite() && rel <= bound) {
        return Err(SarimaWait::NotIdentified);
    }
    let rm = congruent(
        &sums.h.iter().map(|v| v / sums.w).collect::<Vec<_>>(),
        &jac,
        n,
    );
    Ok(Accepted {
        x,
        sigma2,
        v: sums.w,
        rm,
        rings: sums.rings,
    })
}

/// A started target's estimator: the unconstrained parameters, `sigma2`,
/// the weight `V`, `R` (row-major, in the unconstrained coordinates) and
/// the rings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Run {
    pub x: Vec<f64>,
    pub sigma2: f64,
    pub v: f64,
    pub rm: Vec<f64>,
    pub rings: Rings,
}

impl Run {
    pub(crate) fn of(a: Accepted) -> Self {
        Self {
            x: a.x,
            sigma2: a.sigma2,
            v: a.v,
            rm: a.rm,
            rings: a.rings,
        }
    }

    /// The doubles a run holds.
    pub(crate) fn len(shape: &Shape) -> usize {
        let n = shape.n();
        n + 2 + n * n + Rings::len(shape)
    }

    pub(crate) fn shaped(&self, shape: &Shape) -> bool {
        let n = shape.n();
        self.x.len() == n && self.rm.len() == n * n && self.rings.shaped(shape)
    }

    /// One slot: the prediction `ŵ_t`, and where `w` is observed with a
    /// weight above 0, the step, the residual kept at the new parameters.
    /// `xw` is what the prediction reads; `kept`, what the rings keep: the
    /// same on a learned row, `None` on a row of weight 0, which keeps
    /// nothing of its own.
    pub(crate) fn slot(
        &mut self,
        shape: &Shape,
        w: Option<f64>,
        weight: f64,
        xw: Option<&[f64]>,
        kept: Option<&[f64]>,
    ) -> f64 {
        let n = shape.n();
        let (par, jac) = shape.params(&self.x);
        let pol = Polys::new(shape, &par);
        let w_hat = self.rings.w_hat(&pol, xw);
        let learn = w.filter(|_| weight > 0.0 && w_hat.is_finite());
        let Some(w) = learn else {
            let (mut xf, mut psi) = (Vec::new(), Vec::new());
            self.rings.pass(shape, &pol, None, kept, &mut xf, &mut psi);
            return w_hat;
        };
        let e = w - w_hat;
        let mut xf = Vec::with_capacity(shape.k);
        self.rings.xf_now(&pol, xw, &mut xf);
        let mut psi = Vec::with_capacity(n);
        self.rings.psi(shape, &xf, &mut psi);
        let pu = transposed(&jac, &psi, n);
        self.v += weight;
        let a = weight / self.v;
        for i in 0..n {
            for j in 0..n {
                let m = &mut self.rm[i * n + j];
                *m += a * (pu[i] * pu[j] - *m);
            }
        }
        let rhs: Vec<f64> = pu.iter().map(|v| v * e).collect();
        if let Some(step) = chol_solve(&self.rm, n, &rhs) {
            for (x, d) in self.x.iter_mut().zip(&step) {
                *x += a * d;
            }
        }
        self.sigma2 += a * (e * e - self.sigma2);
        // The residual kept is the a-posteriori one, at the new parameters.
        let (par, _) = shape.params(&self.x);
        let pol = Polys::new(shape, &par);
        let z_hat = self.rings.z_hat(&pol);
        let exog = Rings::exog(&pol, xw);
        let z = w - exog;
        let e_post = z - z_hat;
        self.rings.xf_now(&pol, xw, &mut xf);
        self.rings.keep(&pol, z, e_post, xw, &xf);
        w_hat
    }

    /// The parameters, `(beta, phi, theta, Phi, Theta)`.
    pub(crate) fn params(&self, shape: &Shape) -> Vec<f64> {
        shape.params(&self.x).0
    }
}

/// A target's estimator: its buffer until the start is accepted, then its
/// run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Est {
    pub buf: Buffer,
    /// `sum w` and `sum w^2` over the buffer's observed slots.
    pub sw: f64,
    pub sw2: f64,
    /// Kish's count at which the next fit is tried.
    pub next_try: f64,
    pub wait: SarimaWait,
    pub run: Option<Run>,
}

impl Est {
    pub(crate) fn new() -> Self {
        Self {
            buf: Buffer::default(),
            sw: 0.0,
            sw2: 0.0,
            next_try: 0.0,
            wait: SarimaWait::Filling,
            run: None,
        }
    }

    /// Kish's count of the buffer.
    pub(crate) fn kish(&self) -> f64 {
        if self.sw2 > 0.0 {
            self.sw * self.sw / self.sw2
        } else {
            0.0
        }
    }

    /// A slot into the buffer; then, past the structural minimum `floor`
    /// and the retry's count, the start's fit and its gate at `bound`.
    pub(crate) fn buffer(
        &mut self,
        shape: &Shape,
        w: Option<f64>,
        weight: f64,
        xw: Option<&[f64]>,
        floor: f64,
        bound: f64,
    ) {
        let seen = w.filter(|_| weight > 0.0 && (shape.k == 0 || xw.is_some()));
        self.buf.w.push(seen.unwrap_or(f64::NAN));
        self.buf
            .weight
            .push(if seen.is_some() { weight } else { 0.0 });
        match xw {
            Some(x) => self.buf.xw.extend_from_slice(x),
            None => self.buf.xw.extend((0..shape.k).map(|_| f64::NAN)),
        }
        if self.wait == SarimaWait::Filling {
            self.wait = SarimaWait::Buffering;
        }
        if seen.is_none() {
            return;
        }
        self.sw += weight;
        self.sw2 += weight * weight;
        let kish = self.kish();
        if !(kish > floor && kish >= self.next_try) {
            return;
        }
        match fit(shape, &self.buf, bound) {
            Ok(a) => {
                self.run = Some(Run::of(a));
                self.buf = Buffer::default();
            }
            Err(why) => {
                self.wait = why;
                self.next_try = kish * RETRY_GROWTH;
            }
        }
    }
}
