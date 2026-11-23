//! `ln Γ`, `ψ`, `ψ′`, and the regularized incomplete gamma functions
//! `P(a, x) = γ(a, x)/Γ(a)`, `Q(a, x) = 1 − P(a, x)` with their inverses in
//! `x`.
//!
//! `ln Γ`, `ψ` and `ψ′` are shifted by their recurrences
//! (`ln Γ(x) = ln Γ(x + 1) − ln x`, `ψ(x) = ψ(x + 1) − 1/x`,
//! `ψ′(x) = ψ′(x + 1) + 1/x²`) to an argument of at least 20 and read from
//! their asymptotic series there, whose first term left out is below
//! `1e-17` of the value. Near `x = 1` and `x = 2`, where `ln Γ` crosses 0,
//! [`ln_gamma_1p`] keeps its relative accuracy by the Taylor series of
//! `ln Γ(1 + a)` in `a`.
//!
//! `P` and `Q` are evaluated each where it is the smaller, so the small
//! one keeps its relative accuracy and the large one is `1 −` it:
//!
//! ```text
//! a < 1, x < 1.5:   u = xᵃ/Γ(a+1),  s = Σ_{n≥1} (−x)ⁿ a/(n!(a+n))
//!                   P = u (1 + s),  Q = −expm1(a ln x − ln Γ(a+1)) − u s
//! x < a + 1:        P = xᵃ e⁻ˣ/Γ(a+1) · Σ_{n≥0} xⁿ/((a+1)…(a+n))
//! otherwise:        Q = xᵃ e⁻ˣ/Γ(a) · 1/(x+1−a− 1·(1−a)/(x+3−a− 2·(2−a)/(x+5−a− …)))
//! ```
//!
//! the continued fraction by the modified Lentz method. The prefix
//! `xᵃ e⁻ˣ/Γ(a)` is formed in logs; from `a = 20` up, where `ln Γ(a)` is
//! Stirling's series itself, as `a (ln(1+t) − t) + ½ ln a − ln √(2π) − S(a)`
//! with `t = (x − a)/a` and `S` the series' tail, so the terms of size
//! `a ln a` that cancel are never formed.
//!
//! The inverses start from the Wilson–Hilferty or the small-`a` power-law
//! guess and take Halley steps on `P` (or on `Q` in the upper tail), each
//! with the density `xᵃ⁻¹ e⁻ˣ/Γ(a)` from the same prefix.

use super::LN_SQRT_2PI;

/// `ln x`, `eˣ`, `ln(1 + x)`, `eˣ − 1`: the `libm` crate's.
#[inline]
fn ln(x: f64) -> f64 {
    libm::log(x)
}
#[inline]
fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// The tail of Stirling's series for `ln Γ(x)`, `x ≥ 20`:
/// `1/(12x) − 1/(360x³) + 1/(1260x⁵) − 1/(1680x⁷) + 1/(1188x⁹)`.
#[inline]
pub(super) fn stirling_tail(x: f64) -> f64 {
    let r = 1.0 / x;
    let r2 = r * r;
    r * (1.0 / 12.0
        + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0 + r2 * (-1.0 / 1680.0 + r2 * (1.0 / 1188.0)))))
}

/// `ln Γ(x)` for `x > 0`; `+∞` at `+∞` and at 0, NaN below 0. Below 20 the
/// recurrence sums up to twenty logarithms, so the error there is absolute,
/// up to about 100 ε, and not relative where `ln Γ` crosses 0 at 1 and 2:
/// [`ln_gamma_1p`] keeps the digits there.
pub fn ln_gamma(mut x: f64) -> f64 {
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut shift = 0.0;
    while x < 20.0 {
        shift += ln(x);
        x += 1.0;
    }
    (x - 0.5) * ln(x) - x + LN_SQRT_2PI + stirling_tail(x) - shift
}

/// The digamma function `ψ(x) = d ln Γ(x)/dx` for `x > 0`.
pub fn digamma(mut x: f64) -> f64 {
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
    ln(x) - 0.5 * r - series - shift
}

/// The trigamma function `ψ′(x) = d² ln Γ(x)/dx²` for `x > 0`:
/// `1/x + 1/(2x²) + Σₖ B₂ₖ/x²ᵏ⁺¹` from 20 up.
pub fn trigamma(mut x: f64) -> f64 {
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
            + r2 * (-1.0 / 30.0 + r2 * (1.0 / 42.0 + r2 * (-1.0 / 30.0 + r2 * (5.0 / 66.0)))));
    shift + (r + (0.5 * r2 + series))
}

/// `ζ(k) − 1` for `k = 2, …, 56`, the Taylor coefficients of `ln Γ(1 + a)`
/// past `ln(1 + a)` (`Σ_{n≥2} n⁻ᵏ`, summed to 60 digits with the
/// Euler–Maclaurin tail, then rounded).
#[allow(clippy::excessive_precision)]
const ZETA_M1: [f64; 55] = [
    0.6449340668482264,
    0.2020569031595943,
    0.08232323371113819,
    0.03692775514336993,
    0.01734306198444914,
    0.008349277381922827,
    0.00407735619794434,
    0.0020083928260822143,
    0.0009945751278180853,
    0.0004941886041194645,
    0.0002460865533080483,
    0.00012271334757848915,
    6.124813505870483e-05,
    3.058823630702049e-05,
    1.528225940865187e-05,
    7.637197637899763e-06,
    3.81729326499984e-06,
    1.908212716553939e-06,
    9.539620338727962e-07,
    4.769329867878064e-07,
    2.38450502727733e-07,
    1.1921992596531106e-07,
    5.960818905125948e-08,
    2.980350351465228e-08,
    1.4901554828365043e-08,
    7.45071178983543e-09,
    3.725334024788457e-09,
    1.862659723513049e-09,
    9.313274324196682e-10,
    4.656629065033784e-10,
    2.3283118336765053e-10,
    1.164155017270052e-10,
    5.820772087902701e-11,
    2.9103850444971e-11,
    1.4551921891041985e-11,
    7.275959835057482e-12,
    3.637979547378651e-12,
    1.818989650307066e-12,
    9.094947840263888e-13,
    4.547473783042154e-13,
    2.2737368458246524e-13,
    1.136868407680228e-13,
    5.684341987627585e-14,
    2.842170976889302e-14,
    1.4210854828031608e-14,
    7.105427395210853e-15,
    3.552713691337114e-15,
    1.7763568435791204e-15,
    8.881784210930816e-16,
    4.440892103143813e-16,
    2.220446050798042e-16,
    1.1102230251410661e-16,
    5.551115124845481e-17,
    2.775557562136124e-17,
    1.3877787809725232e-17,
];

/// `ln Γ(1 + a)` for `0 ≤ a ≤ 1`, to full relative accuracy where it
/// crosses 0 at both ends:
/// `−γ a + a − ln(1 + a) + Σ_{k≥2} (−a)ᵏ (ζ(k) − 1)/k`, whose terms fall
/// as `(a/2)ᵏ/k`. Outside `[0, 1]` it is [`ln_gamma`]`(1 + a)`.
pub fn ln_gamma_1p(a: f64) -> f64 {
    if !(0.0..=1.0).contains(&a) {
        return ln_gamma(1.0 + a);
    }
    // Horner from the highest power: Σ_{k=2}^{56} c_k (−a)^k, c_k = (ζ(k)−1)/k.
    let m = -a;
    let mut acc = 0.0;
    for (i, z) in ZETA_M1.iter().enumerate().rev() {
        acc = acc * m + z / (i + 2) as f64;
    }
    let tail = acc * m * m;
    (a - libm::log1p(a)) - std::f64::consts::EULER_GAMMA * a + tail
}

/// `ln(xᵃ e⁻ˣ/Γ(a))` for `a > 0`, `x > 0`.
fn ln_prefix(a: f64, x: f64) -> f64 {
    if a >= 20.0 {
        // ln(x/a) − (x/a − 1): by `ln(1 + t)` near x = a, where the two
        // terms cancel; by `ln(x/a)` away from it, where `ln(1 + t)` would
        // amplify `t`'s rounding near t = −1.
        let t = (x - a) / a;
        let d = if t.abs() < 0.5 {
            libm::log1p(t) - t
        } else {
            let r = x / a;
            ln(r) - (r - 1.0)
        };
        a * d + 0.5 * ln(a) - LN_SQRT_2PI - stirling_tail(a)
    } else if a < 1.0 {
        // ln Γ(a) = ln Γ(1 + a) − ln a: no shift loop for a small `a`.
        a * ln(x) - x - ln_gamma_1p(a) + ln(a)
    } else {
        a * ln(x) - x - ln_gamma(a)
    }
}

/// `xᵃ e⁻ˣ/Γ(a)` for `a > 0`, `x > 0`. Below `a = 170` (`Γ`'s overflow),
/// where each factor and the product are normal numbers, as the product of the `libm` crate's `pow`,
/// `exp` and `tgamma`, each within an ulp or two: the log form's sum
/// `a ln x − x` carries an absolute rounding of `x · ε`, which `exp` turns
/// into a relative error of `x` ulps.
fn prefix(a: f64, x: f64) -> f64 {
    if a < 170.0 && x <= 708.0 {
        let (xa, ex) = (libm::pow(x, a), exp(-x));
        let v = xa * ex / libm::tgamma(a);
        if xa.is_normal() && ex.is_normal() && v.is_normal() {
            return v;
        }
    }
    exp(ln_prefix(a, x))
}

const EPS: f64 = f64::EPSILON;
/// Lentz's floor for a vanishing denominator.
const TINY: f64 = 1e-300;
/// The iteration caps: the series and the fraction converge in
/// `O(√a)` terms near `x = a` and far faster elsewhere.
const MAX_TERMS: usize = 100_000;

/// `(P(a, x), Q(a, x))`.
fn gamma_pq(a: f64, x: f64) -> (f64, f64) {
    if a.is_nan() || x.is_nan() || a < 0.0 || x < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if a == 0.0 {
        // The limit a → 0: all the mass at 0.
        return if x > 0.0 {
            (1.0, 0.0)
        } else {
            (f64::NAN, f64::NAN)
        };
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if x == f64::INFINITY {
        return if a == f64::INFINITY {
            (f64::NAN, f64::NAN)
        } else {
            (1.0, 0.0)
        };
    }
    if a == f64::INFINITY {
        return (0.0, 1.0);
    }
    if a < f64::MIN_POSITIVE {
        // P = 1 − O(a): 1 to the last bit; Q = O(a) underflows past here.
        return (1.0, 0.0);
    }
    if a < 1.0 && x < 1.5 {
        // s = Σ_{n≥1} (−x)ⁿ a / (n! (a + n)).
        let mut term = 1.0; // (−x)ⁿ/n!
        let mut s = 0.0;
        for n in 1..MAX_TERMS {
            let nf = n as f64;
            term *= -x / nf;
            let d = term * a / (a + nf);
            s += d;
            if d.abs() <= s.abs() * EPS * 0.5 {
                break;
            }
        }
        let v = a * ln(x) - ln_gamma_1p(a);
        let u = exp(v);
        let p = u * (1.0 + s);
        let q = -libm::expm1(v) - u * s;
        return (p, q);
    }
    let pre = prefix(a, x);
    if x < a + 1.0 {
        // P = prefix · Σ_{n≥0} xⁿ/(a (a+1)…(a+n)).
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..MAX_TERMS {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del <= sum * EPS * 0.5 {
                break;
            }
        }
        let p = sum * pre;
        (p, 1.0 - p)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAX_TERMS {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() <= EPS * 0.5 {
                break;
            }
        }
        let q = pre * h;
        (1.0 - q, q)
    }
}

/// The regularized lower incomplete gamma function
/// `P(a, x) = (1/Γ(a)) ∫₀ˣ tᵃ⁻¹ e⁻ᵗ dt`, `a ≥ 0`, `x ≥ 0`: the gamma
/// distribution's CDF. `P(0, x) = 1` for `x > 0` (the limit); NaN outside
/// the domain.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).0
}

/// The regularized upper incomplete gamma function `Q(a, x) = 1 − P(a, x)`,
/// to full relative accuracy where it is small.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    gamma_pq(a, x).1
}

/// The `x` with `P(a, x) = p` (`upper` false) or `Q(a, x) = p` (`upper`
/// true), `a > 0`, `p` in `[0, 1]`.
fn gamma_inv(a: f64, p: f64, upper: bool) -> f64 {
    if a.is_nan() || p.is_nan() || !(0.0..=1.0).contains(&p) || a <= 0.0 {
        return f64::NAN;
    }
    if a == f64::INFINITY {
        return f64::NAN;
    }
    // (lower-tail probability, upper-tail probability)
    let (pl, pu) = if upper { (1.0 - p, p) } else { (p, 1.0 - p) };
    if pl == 0.0 {
        return 0.0;
    }
    if pu == 0.0 {
        return f64::INFINITY;
    }
    // The start: Wilson–Hilferty above a = 1, the leading power law (lower
    // tail) or exponential tail (upper) below.
    let mut x = if a > 1.0 && pu < 1e-3 {
        // Deep in the upper tail: Q ≈ xᵃ⁻¹ e⁻ˣ/Γ(a), solved by its fixed
        // point x = −ln(q Γ(a)) + (a − 1) ln x from above.
        let x0 = -ln(pu) - ln_gamma(a);
        let mut x = x0.max(a);
        for _ in 0..4 {
            x = (x0 + (a - 1.0) * ln(x)).max(a);
        }
        x
    } else if a > 1.0 {
        let pp = pl.min(pu);
        let t = (-2.0 * ln(pp)).sqrt();
        let mut z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if pl < 0.5 {
            z = -z;
        }
        let w = 1.0 - 1.0 / (9.0 * a) - z / (3.0 * a.sqrt());
        if w > 0.1 {
            a * w * w * w
        } else {
            // Deep in the lower tail: P ≈ xᵃ/Γ(a + 1).
            exp((ln(pl) + ln_gamma(a + 1.0)) / a)
        }
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if pl < t {
            libm::pow(pl / t, 1.0 / a)
        } else {
            1.0 - ln(pu / (1.0 - t))
        }
    };
    if x == 0.0 {
        // `pl` so small that the power law underflows: the root is below
        // the smallest double.
        return 0.0;
    }
    // Halley steps (Newton steps on the log of a tail probability below
    // 0.1), safeguarded: each iterate tightens a bracket on the root (P
    // rises with x), and a step that leaves it, or a density that has
    // underflowed, is replaced by a bisection (geometric while the bracket
    // spans more than a factor of 2).
    let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
    for _ in 0..300 {
        let (gp, gq) = gamma_pq(a, x);
        // err in the tail that is the smaller, where it has its digits
        let err = if pl < 0.5 { gp - pl } else { pu - gq };
        if err == 0.0 {
            break;
        }
        if err > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let dens = prefix(a, x) / x;
        let mut next = f64::NAN;
        if dens.is_normal() {
            next = if pl < 0.1 {
                // Newton on ln P, which is near-linear in the lower tail
                // where P itself falls by orders of magnitude a step.
                x - (ln(gp) - ln(pl)) * gp / dens
            } else if pu < 0.1 {
                // and on ln Q in the upper tail
                x + (ln(gq) - ln(pu)) * gq / dens
            } else {
                let u = err / dens;
                // Halley: f''/f' = (a − 1)/x − 1
                x - u / (1.0 - 0.5 * (u * ((a - 1.0) / x - 1.0)).min(1.0))
            };
        }
        if (next - x).abs() <= 2.0 * EPS * x {
            // converged: the step is below the rounding of `x`
            x = next;
            break;
        }
        if !(next > lo && next < hi) {
            next = bisect(lo, hi, x);
        }
        x = next;
        if hi - lo <= 2.0 * EPS * lo {
            break;
        }
    }
    x
}

/// A point inside `(lo, hi)`, `x` the last iterate: the geometric mean
/// while the bracket spans more than a factor of 2, else the midpoint; an
/// open end is stepped by a factor of 2 from `x`.
pub(super) fn bisect(lo: f64, hi: f64, x: f64) -> f64 {
    if hi == f64::INFINITY {
        x * 2.0
    } else if lo == 0.0 {
        x / 2.0
    } else if hi > 2.0 * lo {
        (lo * hi).sqrt()
    } else {
        0.5 * (lo + hi)
    }
}

/// The inverse of [`gamma_p`] in `x`: the `x ≥ 0` with `P(a, x) = p`, the
/// gamma distribution's quantile. `0` at `p = 0`, `+∞` at `p = 1`, NaN
/// for `a ≤ 0` or `p` outside `[0, 1]`.
pub fn gamma_p_inv(a: f64, p: f64) -> f64 {
    gamma_inv(a, p, false)
}

/// The inverse of [`gamma_q`] in `x`: the `x` with `Q(a, x) = q`, to full
/// accuracy in the upper tail. `+∞` at `q = 0`, `0` at `q = 1`.
pub fn gamma_q_inv(a: f64, q: f64) -> f64 {
    gamma_inv(a, q, true)
}
