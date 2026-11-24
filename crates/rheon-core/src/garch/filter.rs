//! The variance filters: each process's conditional variance `s_t`, read
//! before row `t`'s residual, and its sensitivity `∂ ln s_t/∂θ`, carried by
//! its own recursion at the current parameters.
//!
//! ```text
//! GARCH(1,1)    s_t = ω + α e²_{t−1} + β s_{t−1}
//! GJR(1,1,1)    s_t = ω + α e²_{t−1} + γ 1[e_{t−1} < 0] e²_{t−1} + β s_{t−1}
//!               ∂s_t/∂θ = (1, e², 1[e < 0] e², s)_{t−1} + β ∂s_{t−1}/∂θ
//! EGARCH(1,1,1) ln s_t = ω + α (|z_{t−1}| − √(2/π)) + γ z_{t−1} + β ln s_{t−1}
//!               ∂ln s_t/∂θ = (1, |z| − √(2/π), z, ln s)_{t−1}
//!                            + (β − (α |z| + γ z)/2)_{t−1} ∂ln s_{t−1}/∂θ
//! FIGARCH(1,d,1) s_t = ω/(1 − β) + Σ_{j=1..M} λ_j e²_{t−j}
//! ```
//!
//! With inputs (GARCH-X), the row's own `x_t` enters its variance, `π'x_t`
//! added to GARCH's and GJR's `s_t` and to EGARCH's `ln s_t`, and each `π`'s
//! sensitivity is the same recursion driven by the input: `∂s_t/∂π = x_t +
//! β ∂s_{t−1}/∂π`, and EGARCH's `∂ln s_t/∂π = x_t + (β − (α |z| + γ
//! z)/2)_{t−1} ∂ln s_{t−1}/∂π`. FIGARCH takes none.
//!
//! Before the first row every lag is the backcast `b` (`arch`'s: the
//! 0.94-weighted mean of the first 75 squared residuals), GJR's asymmetric
//! term `b/2`, EGARCH's news terms 0 and its `ln s` lag `ln b`, and every
//! sensitivity 0. EGARCH's variance is held within `arch`'s bounds, a
//! factor `10⁶` either side of the 0.94-weighted mean of `e²` over the
//! rows before (from `b`), above it only logarithmically, so a step that
//! runs `α` or `γ` far off leaves a variance that is a number. FIGARCH's weights, with `P_1 = 1`,
//! `P_j = P_{j−1} (j − 1 − d)/j` and `δ_j = d P_j`:
//!
//! ```text
//! λ_1 = φ − β + d,     λ_j = β λ_{j−1} + δ_j − φ δ_{j−1}
//! ```
//!
//! and their derivatives in `(φ, d, β)` are the same linear recursion,
//! driven by `(1, −δ_1, …)`, `(1, ∂δ_j/∂d − φ ∂δ_{j−1}/∂d)` with
//! `∂P_j/∂d = −P_j Σ_{i=2..j} 1/(i − 1 − d)`, and `(−1, λ_1, …)`. They are
//! no state: one pass of `O(M)` along the ring makes them and the dot
//! products together. The ring itself, the last `M` squared residuals, is
//! FIGARCH's state. The set the parameters are held to keeps every `λ_j ≥
//! 0`; a weight the recursion's rounding leaves a hair below 0 is read as 0
//! in the variance and its forecast, so a squared residual as large as the
//! inputs allow never takes the variance below 0 (review 2026-11-23,
//! GARCH-12).

use serde::{Deserialize, Serialize};

use super::math;
use super::{GarchVol, MAX_K};

/// `E|z|` of a standard normal, EGARCH's centring of `|z|` (`arch` keeps it
/// for every distribution).
pub(super) const ABS_Z: f64 = 0.797_884_560_802_865_4;

/// EGARCH's `ln s` is held below this before it is exponentiated, so a
/// runaway step leaves a variance that is large but a number.
const LN_S_MAX: f64 = 700.0;

/// The lags a process keeps between rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Filter {
    /// GARCH and GJR: `e²`, the asymmetric term `1[e < 0] e²` and `s` of the
    /// last row; EGARCH: `z` and `ln s` of the last row, and the
    /// 0.94-weighted mean of `e²` its variance bounds are read from.
    pub lag: [f64; 3],
    /// GARCH and GJR: `∂s/∂θ` of the last row; EGARCH: `∂ln s/∂θ`. Empty for
    /// FIGARCH.
    pub sens: Vec<f64>,
    /// EGARCH: no row read yet, so the news terms are 0.
    pub fresh: bool,
    /// FIGARCH: the last `M` squared residuals, `ring[head]` the newest.
    pub ring: Vec<f64>,
    pub head: usize,
}

/// Visit FIGARCH's weights `λ_j` and their derivatives in `(φ, d, β)`,
/// `j = 1..=m`, in order: `f(j, λ_j, ∂λ_j/∂φ, ∂λ_j/∂d, ∂λ_j/∂β)`.
/// `deriv` false passes the derivatives as 0, unformed.
pub(super) fn figarch_weights(
    theta: &[f64],
    m: usize,
    deriv: bool,
    mut f: impl FnMut(usize, f64, f64, f64, f64),
) {
    let (phi, d, beta) = (theta[1], theta[2], theta[3]);
    let mut p = 1.0;
    let mut sum_inv = 0.0;
    let mut delta = d;
    let mut ddelta = 1.0; // P_1 + d ∂P_1/∂d
    let mut lam = d - beta + phi;
    let (mut l_phi, mut l_d, mut l_b) = (1.0, 1.0, -1.0);
    f(1, lam, l_phi, l_d, l_b);
    for j in 2..=m {
        let jf = j as f64;
        p *= (jf - 1.0 - d) / jf;
        let delta_new = d * p;
        let u = delta_new - phi * delta;
        let lam_new = beta * lam + u;
        if deriv {
            sum_inv += 1.0 / (jf - 1.0 - d);
            let dp = -p * sum_inv;
            let ddelta_new = p + d * dp;
            l_phi = beta * l_phi - delta;
            l_d = beta * l_d + ddelta_new - phi * ddelta;
            l_b = beta * l_b + lam;
            ddelta = ddelta_new;
        }
        delta = delta_new;
        lam = lam_new;
        f(j, lam, l_phi, l_d, l_b);
    }
}

impl Filter {
    /// The lags before the first row, at the backcast `b`.
    pub fn new(vol: GarchVol, k: usize, m: usize, b: f64) -> Self {
        match vol {
            GarchVol::Garch | GarchVol::Gjr => Self {
                lag: [b, 0.5 * b, b],
                sens: vec![0.0; k],
                fresh: false,
                ring: Vec::new(),
                head: 0,
            },
            GarchVol::Egarch => Self {
                lag: [0.0, math::ln(b), b],
                sens: vec![0.0; k],
                fresh: true,
                ring: Vec::new(),
                head: 0,
            },
            GarchVol::Figarch => Self {
                lag: [0.0; 3],
                sens: Vec::new(),
                fresh: false,
                ring: vec![b; m],
                head: 0,
            },
        }
    }

    /// No lags: a series not yet started.
    pub fn empty() -> Self {
        Self {
            lag: [0.0; 3],
            sens: Vec::new(),
            fresh: false,
            ring: Vec::new(),
            head: 0,
        }
    }

    /// The `j`-th squared residual back (`j ≥ 1`) of FIGARCH's ring.
    pub(super) fn back(&self, j: usize) -> f64 {
        let m = self.ring.len();
        self.ring[(self.head + j - 1) % m]
    }

    /// `s_t` at `theta`, and `∂ ln s_t/∂θ` into `dl` (`k` entries): the
    /// process's parameters, then one a input of `x`, the row's own inputs,
    /// whose terms `π'x` come after the process's in the sum.
    pub fn predict(&self, vol: GarchVol, theta: &[f64], x: &[f64], dl: &mut [f64]) -> f64 {
        let vk = vol.k();
        match vol {
            GarchVol::Garch | GarchVol::Gjr => {
                let beta = theta[vk - 1];
                let reg = |i: usize| match (i, vk) {
                    (0, _) => 1.0,
                    (1, _) => self.lag[0],
                    (2, 4) => self.lag[1],
                    _ if i < vk => self.lag[2],
                    _ => x[i - vk],
                };
                let s: f64 = (0..dl.len()).map(|i| theta[i] * reg(i)).sum();
                for (i, d) in dl.iter_mut().enumerate() {
                    *d = reg(i) + beta * self.sens[i];
                }
                if s > 0.0 {
                    for d in dl.iter_mut() {
                        *d /= s;
                    }
                }
                s
            }
            GarchVol::Egarch => {
                let (az, zz, coef) = if self.fresh {
                    (0.0, 0.0, theta[3])
                } else {
                    let z = self.lag[0];
                    (
                        z.abs() - ABS_Z,
                        z,
                        theta[3] - 0.5 * (theta[1] * z.abs() + theta[2] * z),
                    )
                };
                let l = self.lag[1];
                let mut lt = theta[0] + theta[1] * az + theta[2] * zz + theta[3] * l;
                for (p, v) in theta[vk..dl.len()].iter().zip(x) {
                    lt += p * v;
                }
                let head = [1.0, az, zz, l];
                for (i, d) in dl.iter_mut().enumerate() {
                    let reg = if i < vk { head[i] } else { x[i - vk] };
                    *d = reg + coef * self.sens[i];
                }
                // `arch`'s variance bounds: within a factor 10⁶ of the
                // 0.94-weighted mean of `e²` over the rows before, above it
                // only logarithmically, `hi + ln s − ln hi` in `arch`'s
                // order, which forms no `s/hi` to overflow. Below, the
                // variance does not move with the parameters; above, it
                // moves as `ln s` does, so `∂ ln s/∂θ` is `∂ ln s_raw/∂θ`
                // over the bounded `s` (0 past the exponent's cap; review
                // 2026-11-23, GARCH-5).
                let s = math::exp(lt.min(LN_S_MAX));
                let (lo, hi) = (self.lag[2] / 1e6, self.lag[2] * 1e6);
                if s < lo {
                    dl.fill(0.0);
                    return lo;
                }
                if s > hi {
                    let bounded = hi + math::ln(s) - math::ln(hi);
                    if lt < LN_S_MAX {
                        for d in dl.iter_mut() {
                            *d /= bounded;
                        }
                    } else {
                        dl.fill(0.0);
                    }
                    return bounded;
                }
                s
            }
            GarchVol::Figarch => {
                let one_b = 1.0 - theta[3];
                let w = theta[0] / one_b;
                let (mut h, mut dp, mut dd, mut db) = (w, 0.0, 0.0, 0.0);
                figarch_weights(theta, self.ring.len(), true, |j, lam, lp, ld, lb| {
                    let x = self.back(j);
                    h += lam.max(0.0) * x;
                    dp += lp * x;
                    dd += ld * x;
                    db += lb * x;
                });
                let dh = [1.0 / one_b, dp, dd, db + w / one_b];
                for (d, v) in dl.iter_mut().zip(dh) {
                    *d = v / h;
                }
                h
            }
        }
    }

    /// `s_t` at `theta` alone, without the sensitivity; `k` the variance
    /// equation's parameters, `x` the row's inputs.
    pub fn variance(&self, vol: GarchVol, theta: &[f64], x: &[f64], k: usize) -> f64 {
        if vol == GarchVol::Figarch {
            let mut h = theta[0] / (1.0 - theta[3]);
            figarch_weights(theta, self.ring.len(), false, |j, lam, _, _, _| {
                h += lam.max(0.0) * self.back(j);
            });
            return h;
        }
        let mut dl = [0.0; MAX_K];
        self.predict(vol, theta, x, &mut dl[..k])
    }

    /// The lags moved past a row of residual `e`, whose variance was `s` and
    /// sensitivity `dl`.
    pub fn push(&mut self, vol: GarchVol, e: f64, s: f64, dl: &[f64]) {
        let e2 = e * e;
        match vol {
            GarchVol::Garch | GarchVol::Gjr => {
                self.lag = [e2, if e < 0.0 { e2 } else { 0.0 }, s];
                for (d, l) in self.sens.iter_mut().zip(dl) {
                    *d = l * s;
                }
            }
            GarchVol::Egarch => {
                self.lag = [e / s.sqrt(), math::ln(s), 0.94 * self.lag[2] + 0.06 * e2];
                self.sens.copy_from_slice(dl);
                self.fresh = false;
            }
            GarchVol::Figarch => {
                let m = self.ring.len();
                self.head = (self.head + m - 1) % m;
                self.ring[self.head] = e2;
            }
        }
    }
}
