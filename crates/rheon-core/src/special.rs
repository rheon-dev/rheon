//! Special functions and statistical tail probabilities in pure Rust on
//! `libm`, with no external numerical dependencies and no `unsafe`:
//!
//! - [`ln_gamma`] and [`digamma`] (shared with `garch::dist`),
//! - [`betainc`] (regularized incomplete beta `I_x(a, b)`) and [`student_t_sf`],
//! - [`gammainc`] / [`gammaincc`] (regularized incomplete gamma `P(a, x)` /
//!   `Q(a, x)`) and [`chi2_sf`],
//! - [`cm_eb_bound`] and [`cm_eb_chunk`] (the conjugate-mixture empirical
//!   Bernstein uniform boundary of Howard, Ramdas, McAuliffe and Sekhon 2021).

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
const FPMIN: f64 = 1e-300;

/// Stirling asymptotic correction `S(x)` for `x >= 20`, where `ln Γ(x) =
/// (x - 1/2) ln x - x + ln √(2π) + S(x)` with remainder `< 1e-17`.
#[inline]
fn stirling_series(x: f64) -> f64 {
    let r = 1.0 / x;
    let r2 = r * r;
    r * (1.0 / 12.0
        + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0 + r2 * (-1.0 / 1680.0 + r2 * (1.0 / 1188.0)))))
}

/// `ln(1 + u) - u` without catastrophic cancellation near `u = 0`.
#[inline]
fn ln_1p_minus_u(u: f64) -> f64 {
    if u.abs() < 1e-3 {
        -u * u * (0.5 - u * (1.0 / 3.0 - u * (0.25 - u * (0.2 - u * (1.0 / 6.0 - u / 7.0)))))
    } else {
        libm::log1p(u) - u
    }
}

/// `ln Γ(x)` for `x > 0`; `NaN` for any non-positive or non-finite input.
pub fn ln_gamma(mut x: f64) -> f64 {
    if !(x > 0.0 && x.is_finite()) {
        return f64::NAN;
    }
    let mut shift = 0.0;
    while x < 20.0 {
        shift += libm::log(x);
        x += 1.0;
    }
    (x - 0.5) * libm::log(x) - x + LN_SQRT_2PI + stirling_series(x) - shift
}

/// Digamma `ψ(x) = d ln Γ(x) / dx` for `x > 0`; `NaN` otherwise.
pub fn digamma(mut x: f64) -> f64 {
    if !(x > 0.0 && x.is_finite()) {
        return f64::NAN;
    }
    let mut shift = 0.0;
    while x < 20.0 {
        shift += 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    let series = r2
        * (1.0 / 12.0
            - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 * (1.0 / 240.0 - r2 * (1.0 / 132.0)))));
    libm::log(x) - 0.5 * r - series - shift
}

/// Trigamma `ψ₁(x) = d² ln Γ(x) / dx²` for `x > 0`; `NaN` otherwise.
/// Pure arithmetic (no platform `libm`), shifted to `x >= 20` where the
/// omitted asymptotic term is `< 1e-17`.
pub fn trigamma(mut x: f64) -> f64 {
    if !(x > 0.0 && x.is_finite()) {
        return f64::NAN;
    }
    let mut shift = 0.0;
    while x < 20.0 {
        shift += 1.0 / (x * x);
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    let series = r
        * r2
        * (1.0 / 6.0
            + r2 * (-1.0 / 30.0
                + r2 * (1.0 / 42.0
                    + r2 * (-1.0 / 30.0 + r2 * (5.0 / 66.0 - r2 * (691.0 / 2730.0))))));
    r + 0.5 * r2 + series + shift
}

/// Cancellation-free `ψ₁(x) - ψ₁(x + 1/2)` for `x > 0`; `NaN` otherwise.
pub fn trigamma_half_diff(mut x: f64) -> f64 {
    if !(x > 0.0 && x.is_finite()) {
        return f64::NAN;
    }
    let mut shift = 0.0;
    while x < 20.0 {
        let y = x + 0.5;
        // 1/x² - 1/(x + 1/2)² = (x + 1/4) / (x² (x + 1/2)²)
        shift += (x + 0.25) / (x * x * y * y);
        x += 1.0;
    }
    let y = x + 0.5;
    let p = 1.0 / x;
    let q = 1.0 / y;
    let d = 0.5 * p * q; // 1/x - 1/y without cancellation
    let pow_diff = |m: usize| -> f64 {
        let (mut acc, mut pm, mut qk) = (0.0, p.powi((m - 1) as i32), 1.0);
        for _ in 0..m {
            acc += pm * qk;
            pm /= p;
            qk *= q;
        }
        d * acc
    };
    let tail = (1.0 / 6.0) * pow_diff(3) - (1.0 / 30.0) * pow_diff(5) + (1.0 / 42.0) * pow_diff(7)
        - (1.0 / 30.0) * pow_diff(9)
        + (5.0 / 66.0) * pow_diff(11)
        - (691.0 / 2730.0) * pow_diff(13);
    d + 0.5 * (x + 0.25) * p * p * q * q + tail + shift
}

/// Cancellation-free `ln Γ(a + b) - ln Γ(a) - ln Γ(b)` for `a > 0, b > 0`.
fn neg_ln_beta(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi >= 20.0 && lo <= hi {
        let u = lo / hi;
        // ln Γ(hi + lo) - ln Γ(hi) = lo ln(hi) + hi (ln(1+u) - u) + (lo - 0.5) ln(1+u) + S(hi+lo) - S(hi)
        let diff = lo * libm::log(hi)
            + hi * ln_1p_minus_u(u)
            + (lo - 0.5) * libm::log1p(u)
            + (stirling_series(hi + lo) - stirling_series(hi));
        diff - ln_gamma(lo)
    } else {
        ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)
    }
}

/// Modified Lentz continued fraction for `I_x(a, b)` when `x <= (a + 1) / (a + b + 2)`.
fn betainc_cf(a: f64, b: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-15;
    const MAX_ITER: usize = 4_000;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() <= EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)` given both `x` and `y = 1 - x` to
/// avoid subtraction loss near `x = 1`.
fn betainc_pair(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if !(a > 0.0 && a.is_finite() && b > 0.0 && b.is_finite() && x.is_finite() && y.is_finite())
        || x < 0.0
        || y < 0.0
    {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if y == 0.0 {
        return 1.0;
    }
    let ln_x = if y < 0.5 {
        libm::log1p(-y)
    } else {
        libm::log(x)
    };
    let ln_y = if x < 0.5 {
        libm::log1p(-x)
    } else {
        libm::log(y)
    };
    let ln_bt = neg_ln_beta(a, b) + a * ln_x + b * ln_y;
    let bt = libm::exp(ln_bt);
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * betainc_cf(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * betainc_cf(b, a, y) / b).clamp(0.0, 1.0)
    }
}

/// Regularized incomplete beta function `I_x(a, b)` for `a > 0`, `b > 0`,
/// `0 <= x <= 1`.
pub fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    betainc_pair(a, b, x, 1.0 - x)
}

/// Upper tail survival function `P(T_df > t)` of Student's t distribution with
/// `df > 0` degrees of freedom.
pub fn student_t_sf(t: f64, df: f64) -> f64 {
    if df.is_nan() || df <= 0.0 || t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return if t.is_sign_positive() { 0.0 } else { 1.0 };
    }
    if df.is_infinite() {
        return 0.5 * libm::erfc(t * INV_SQRT_2);
    }
    if t == 0.0 {
        return 0.5;
    }
    let ta = t.abs();
    // For df >= 5e5, the three-term Cornish-Fisher expansion in 1/df is
    // accurate to < 5e-13 relative across |t| <= 8 and runs in O(1) time;
    // below 5e5, `betainc_pair` uses cancellation-free (x, y) and converges
    // well within MAX_ITER.
    let tail = if df >= 5e5 {
        let z = 0.5 * libm::erfc(ta * INV_SQRT_2);
        let t2 = ta * ta;
        if t2 > 1400.0 {
            0.0
        } else {
            let phi = libm::exp(-0.5 * t2 - LN_SQRT_2PI);
            let c1 = ta * (t2 + 1.0) * 0.25;
            let t4 = t2 * t2;
            let t6 = t4 * t2;
            let c2 = ta * (3.0 * t6 - 7.0 * t4 - 5.0 * t2 - 3.0) / 96.0;
            let t8 = t4 * t4;
            let t10 = t8 * t2;
            let c3 = ta * (t10 - 11.0 * t8 + 14.0 * t6 + 6.0 * t4 - 3.0 * t2 - 15.0) / 384.0;
            let inv = 1.0 / df;
            let inv2 = inv * inv;
            (z + phi * (c1 * inv + c2 * inv2 + c3 * inv2 * inv)).clamp(0.0, 0.5)
        }
    } else {
        let t2 = ta * ta;
        if !t2.is_finite() {
            0.0
        } else {
            let denom = df + t2;
            let x = df / denom;
            let y = t2 / denom;
            0.5 * betainc_pair(0.5 * df, 0.5, x, y)
        }
    };
    if t > 0.0 { tail } else { 1.0 - tail }
}

/// Cancellation-free log prefactor `-x + a ln x - ln Γ(a)` for `a > 0, x > 0`.
#[inline]
fn ln_gamma_prefactor(a: f64, x: f64) -> f64 {
    if a >= 20.0 {
        let u = (x - a) / a;
        if u > -1.0 {
            return a * ln_1p_minus_u(u) + 0.5 * libm::log(a) - LN_SQRT_2PI - stirling_series(a);
        }
    }
    -x + a * libm::log(x) - ln_gamma(a)
}

/// Pair `(P(a, x), Q(a, x))` of regularized lower and upper incomplete gamma
/// functions for `a > 0, x >= 0`.
fn gamma_pq(a: f64, x: f64) -> (f64, f64) {
    if !(a > 0.0 && a.is_finite() && x >= 0.0) || x.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if x.is_infinite() {
        return (1.0, 0.0);
    }
    // For astronomical shape parameters (a > 1e8), Temme's leading uniform
    // asymptotic expansion is accurate to < 1e-12 in O(1) time.
    if a > 1e8 {
        let u = (x - a) / a;
        let eta2 = 2.0 * (-ln_1p_minus_u(u));
        let eta = if u >= 0.0 {
            eta2.max(0.0).sqrt()
        } else {
            -eta2.max(0.0).sqrt()
        };
        let z = eta * (0.5 * a).sqrt();
        let c0 = if u.abs() < 1e-4 {
            -1.0 / 3.0 + u * (1.0 / 12.0 - u * (23.0 / 540.0 - u * 353.0 / 12960.0))
        } else {
            1.0 / u - 1.0 / eta
        };
        let corr = libm::exp(-0.5 * a * eta2 - LN_SQRT_2PI) / a.sqrt() * c0;
        if eta >= 0.0 {
            let q = (0.5 * libm::erfc(z) + corr).clamp(0.0, 1.0);
            return (1.0 - q, q);
        } else {
            let p = (0.5 * libm::erfc(-z) - corr).clamp(0.0, 1.0);
            return (p, 1.0 - p);
        }
    }

    const EPS: f64 = 1e-15;
    const MAX_ITER: usize = 25_000;
    let lp = ln_gamma_prefactor(a, x);
    let use_series = if a < 50.0 { x < a + 1.0 } else { x < a };
    if use_series {
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * EPS {
                break;
            }
        }
        let p = (libm::exp(lp) * sum).clamp(0.0, 1.0);
        (p, 1.0 - p)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITER {
            let fi = i as f64;
            let an = -fi * (fi - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= EPS {
                break;
            }
        }
        let q = (libm::exp(lp) * h).clamp(0.0, 1.0);
        (1.0 - q, q)
    }
}

/// Regularized lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`.
pub fn gammainc(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).0
}

/// Regularized upper incomplete gamma function `Q(a, x) = Γ(a, x) / Γ(a)`.
pub fn gammaincc(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).1
}

/// Upper tail survival function `P(X > x)` of the chi-square distribution with
/// `df > 0` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if !(df > 0.0 && df.is_finite()) || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    gammaincc(0.5 * df, 0.5 * x)
}

/// Log of `P(a, x)`, returning `-inf` when `P(a, x) == 0`.
#[inline]
fn ln_gammainc(a: f64, x: f64) -> f64 {
    let p = gammainc(a, x);
    if p <= 0.0 {
        f64::NEG_INFINITY
    } else {
        libm::log(p)
    }
}

/// Precomputed constants for the gamma-exponential mixture boundary with scale
/// `c = 1` (Howard et al. 2021; MAPIE's `GammaExponentialMixtureBound`).
#[derive(Debug, Clone, Copy)]
struct CmEbConst {
    rho: f64,
    /// `ln_gamma_prefactor(rho, rho) - ln_gammainc(rho, rho)`.
    base: f64,
}

impl CmEbConst {
    fn new(v_opt: f64, alpha: f64) -> Option<Self> {
        if !(v_opt > 0.0 && v_opt.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return None;
        }
        // MAPIE passes `alpha_opt = alpha / 2`, then `_one_sided_best_rho` calls
        // `_two_sided_best_rho(v_opt, 2 * alpha_opt) = _two_sided_best_rho(v_opt, alpha)`.
        let log_term = libm::log(1.0 / alpha);
        let rho = v_opt / (2.0 * log_term + libm::log1p(2.0 * log_term));
        let base = ln_gamma_prefactor(rho, rho) - ln_gammainc(rho, rho);
        if !(rho > 0.0 && rho.is_finite() && base.is_finite()) {
            return None;
        }
        Some(Self { rho, base })
    }

    #[inline]
    fn log_supermg(&self, s: f64, v: f64) -> f64 {
        let a = v + self.rho;
        let x = s + a;
        self.base - ln_gamma_prefactor(a, x) + ln_gammainc(a, x)
    }

    fn bound(&self, v: f64, alpha_side: f64) -> f64 {
        if !(v >= 0.0 && v.is_finite() && alpha_side > 0.0 && alpha_side < 1.0) {
            return f64::NAN;
        }
        let log_threshold = libm::log(1.0 / alpha_side);
        let mut upper = v.max(1.0);
        let mut found = false;
        for _ in 0..80 {
            if self.log_supermg(upper, v) > log_threshold {
                found = true;
                break;
            }
            upper *= 2.0;
        }
        if !found {
            return f64::NAN;
        }
        let mut lo = 0.0;
        let mut hi = upper;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if self.log_supermg(mid, v) > log_threshold {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo <= 1e-12 * hi.max(1.0) {
                break;
            }
        }
        0.5 * (lo + hi)
    }
}

/// Conjugate-mixture empirical-Bernstein half-width numerator `b(v)` at
/// cumulative empirical variance `v >= 0`, tuned for intrinsic time `v_opt > 0`
/// and two-sided confidence level `alpha in (0, 1)` (each side evaluated at
/// `alpha / 2`, matching MAPIE's `GammaExponentialMixtureBound(v_opt, c=1,
/// alpha_opt=alpha/2)(v, alpha=alpha/2)`).
pub fn cm_eb_bound(v: f64, v_opt: f64, alpha: f64) -> f64 {
    match CmEbConst::new(v_opt, alpha) {
        Some(c) => c.bound(v, 0.5 * alpha),
        None => f64::NAN,
    }
}

/// Running state of a streaming conjugate-mixture empirical-Bernstein
/// confidence sequence over observations `r_t in [0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CmEbState {
    pub t: u64,
    pub s: f64,
    pub v: f64,
    pub mu_prev: f64,
    pub lower: f64,
    pub upper: f64,
}

impl Default for CmEbState {
    fn default() -> Self {
        Self {
            t: 0,
            s: 0.0,
            v: 0.0,
            mu_prev: 0.5,
            lower: 0.0,
            upper: 1.0,
        }
    }
}

/// Advance a streaming CM-EB confidence sequence over `rs` (each entry in
/// `[0, 1]`; non-finite entries leave state unchanged and emit `NaN`).
/// Returns `(lowers, uppers)` of length `rs.len()`.
pub fn cm_eb_chunk(
    rs: &[f64],
    v_opt: f64,
    alpha: f64,
    running_intersection: bool,
    state: &mut CmEbState,
) -> (Vec<f64>, Vec<f64>) {
    let n = rs.len();
    let Some(cfg) = CmEbConst::new(v_opt, alpha) else {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    };
    let alpha_side = 0.5 * alpha;
    let mut lowers = Vec::with_capacity(n);
    let mut uppers = Vec::with_capacity(n);
    for &r in rs {
        if !r.is_finite() {
            lowers.push(f64::NAN);
            uppers.push(f64::NAN);
            continue;
        }
        state.t += 1;
        state.s += r;
        let diff = r - state.mu_prev;
        state.v += diff * diff;
        let mu = state.s / (state.t as f64);
        state.mu_prev = mu;
        let bd = cfg.bound(state.v, alpha_side) / (state.t as f64);
        let lo = (mu - bd).max(0.0);
        let hi = (mu + bd).min(1.0);
        if running_intersection {
            state.lower = state.lower.max(lo);
            state.upper = state.upper.min(hi);
        } else {
            state.lower = lo;
            state.upper = hi;
        }
        lowers.push(state.lower);
        uppers.push(state.upper);
    }
    (lowers, uppers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_and_digamma_known_values() {
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-15);
        assert!((ln_gamma(2.0) - 0.0).abs() < 1e-15);
        assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-14);
        assert!((ln_gamma(6.0) - 120.0_f64.ln()).abs() < 1e-14);
        #[allow(clippy::approx_constant)]
        let euler_gamma = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + euler_gamma).abs() < 1e-14);
        assert!(ln_gamma(0.0).is_nan());
        assert!(ln_gamma(-1.0).is_nan());
        // ψ₁(1) = π²/6, ψ₁(1/2) = π²/2, recurrence ψ₁(x) - ψ₁(x+1) = 1/x²
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        assert!((trigamma(1.0) - pi2 / 6.0).abs() < 1e-14);
        assert!((trigamma(0.5) - pi2 / 2.0).abs() < 1e-14);
        for x in [0.5_f64, 1.5, 19.5, 20.5, 150.0] {
            assert!((trigamma(x) - trigamma(x + 1.0) - 1.0 / (x * x)).abs() < 1e-14);
            let direct = trigamma(x) - trigamma(x + 0.5);
            let diff = trigamma_half_diff(x);
            assert!(
                (direct - diff).abs() <= 1e-14 * trigamma(x),
                "x={x}: {direct} vs {diff}"
            );
        }
        assert!(trigamma(0.0).is_nan());
        assert!(trigamma_half_diff(-1.0).is_nan());
    }

    #[test]
    fn student_t_and_chi2_exact_special_cases() {
        // df = 1 is Cauchy: sf(t) = 0.5 - atan(t)/pi
        for t in [-3.0_f64, -1.0, 0.0, 1.0, 3.0, 10.0] {
            let want = 0.5 - t.atan() / std::f64::consts::PI;
            let got = student_t_sf(t, 1.0);
            assert!((got - want).abs() < 1e-14, "t={t}: {got} vs {want}");
        }
        // df = 2: sf(t) = 0.5 * (1 - t / sqrt(2 + t^2))
        for t in [-4.0_f64, -0.5, 0.0, 1.5, 5.0] {
            let want = 0.5 * (1.0 - t / (2.0 + t * t).sqrt());
            let got = student_t_sf(t, 2.0);
            assert!((got - want).abs() < 1e-14, "t={t}: {got} vs {want}");
        }
        // chi2(df=2) is Exp(1/2): sf(x) = exp(-x/2)
        for x in [0.0_f64, 0.5, 2.0, 10.0, 30.0] {
            let want = (-0.5 * x).exp();
            let got = chi2_sf(x, 2.0);
            assert!(
                (got - want).abs() <= 1e-14 * want.max(1e-300),
                "x={x}: {got} vs {want}"
            );
        }
        // chi2(df=1): sf(x) = erfc(sqrt(x/2))
        for x in [0.25_f64, 1.0, 3.841_458_820_694_124, 15.0] {
            let want = libm::erfc((0.5 * x).sqrt());
            let got = chi2_sf(x, 1.0);
            assert!(
                (got - want).abs() <= 1e-13 * want.max(1e-300),
                "x={x}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn cm_eb_chunk_invariance_and_bounds() {
        let rs = [0.6, 0.55, 0.7, 0.45, 0.8, 0.65, 0.5, 0.75, 0.6, 0.7];
        let mut s_whole = CmEbState::default();
        let (lo_w, hi_w) = cm_eb_chunk(&rs, 1.0, 0.05, true, &mut s_whole);

        let mut s_split = CmEbState::default();
        let (lo_1, hi_1) = cm_eb_chunk(&rs[..4], 1.0, 0.05, true, &mut s_split);
        let (lo_2, hi_2) = cm_eb_chunk(&rs[4..], 1.0, 0.05, true, &mut s_split);
        assert_eq!(lo_w, [lo_1, lo_2].concat());
        assert_eq!(hi_w, [hi_1, hi_2].concat());
        assert_eq!(s_whole, s_split);
        for (&l, &u) in lo_w.iter().zip(&hi_w) {
            assert!((0.0..=u).contains(&l) && u <= 1.0);
        }
    }
}
