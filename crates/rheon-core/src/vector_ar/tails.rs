//! The upper tails of the F and χ² distributions, which a Granger test's
//! p-value is read from: `F(d₁, d₂).sf(f) = I_x(d₂/2, d₁/2)` at
//! `x = d₂ / (d₂ + d₁ f)`, the regularized incomplete beta, and
//! `χ²(q).sf(s) = Q(q/2, s/2)`, the regularized upper incomplete gamma.
//!
//! Each is written from its definition: the incomplete beta by its
//! continued fraction (evaluated by Lentz's method), on whichever side of
//! the mean keeps the fraction converging fast and the tail's small values
//! relative, and the incomplete gamma by its series below `x = a + 1` and
//! its continued fraction above. `ln Γ`, `exp`, `ln` and `ln(1 + x)` are
//! the `libm` crate's, so a p-value is the same bits on every platform.
//!
//! These are read for output only: a p-value is never kept in the state.

/// Iterations either fraction or series may take. Each converges in about
/// `√max(a, b)` terms where the tail is far out; a test of a million
/// degrees of freedom takes about a thousand.
const MAX_ITER: usize = 100_000;

/// The relative size of a term at which a series or a fraction stops.
const EPS: f64 = 1e-16;

/// The smallest value a Lentz denominator may take before it is moved off
/// zero.
const TINY: f64 = 1e-300;

/// `ln Γ(a)` for `a > 0`.
fn ln_gamma(a: f64) -> f64 {
    libm::lgamma(a)
}

/// `I_x(a, b)`'s continued fraction (the one that converges for
/// `x < (a + 1) / (a + b + 2)`), evaluated by Lentz's method.
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// The regularized incomplete beta `I_x(a, b)` for `a, b > 0` and `x` in
/// `[0, 1]`; NaN outside that domain.
pub(crate) fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    // `x^a (1 − x)^b / B(a, b)`, in logs.
    let ln_front =
        ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * libm::log(x) + b * libm::log1p(-x);
    let front = libm::exp(ln_front);
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, 1.0 - x) / b
    }
}

/// The complement `1 − I_x(a, b) = I_{1−x}(b, a)`, read on the side that
/// keeps a small value relative: what an upper tail needs.
fn beta_inc_upper(a: f64, b: f64, x: f64) -> f64 {
    beta_inc(b, a, 1.0 - x)
}

/// The regularized upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a)` for
/// `a > 0`, `x >= 0`; NaN outside that domain.
pub(crate) fn gamma_q(a: f64, x: f64) -> f64 {
    if !(a > 0.0 && a.is_finite()) || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let ln_front = -x + a * libm::log(x) - ln_gamma(a);
    if x < a + 1.0 {
        // The series of `P(a, x)`: `e^{−x} x^a / Γ(a + 1) Σ x^n / ((a+1)…(a+n))`.
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        1.0 - sum * libm::exp(ln_front)
    } else {
        // The continued fraction of `Q(a, x)`, by Lentz's method.
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAX_ITER {
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
            if (del - 1.0).abs() < EPS {
                break;
            }
        }
        libm::exp(ln_front) * h
    }
}

/// `P(F > f)` for `F ~ F(d1, d2)`: NaN where a degree of freedom is not
/// positive or `f` is not a number, 1 at `f <= 0`.
pub(crate) fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if !(d1 > 0.0 && d2 > 0.0) || f.is_nan() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    // `x = d1 f / (d1 f + d2)`: the upper tail is `I_{1−x}(d2/2, d1/2)`,
    // with `1 − x = d2 / (d2 + d1 f)` formed directly.
    let x = d1 * f / (d1 * f + d2);
    let one_minus = d2 / (d2 + d1 * f);
    if x < 0.5 {
        beta_inc_upper(d1 / 2.0, d2 / 2.0, x)
    } else {
        beta_inc(d2 / 2.0, d1 / 2.0, one_minus)
    }
}

/// `P(X > s)` for `X ~ χ²(q)`: NaN where `q` is not positive or `s` is not
/// a number, 1 at `s <= 0`.
pub(crate) fn chi2_sf(s: f64, q: f64) -> f64 {
    if !(q > 0.0) || s.is_nan() {
        return f64::NAN;
    }
    if s <= 0.0 {
        return 1.0;
    }
    gamma_q(q / 2.0, s / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) {
        assert!(
            (a - b).abs() <= rel * b.abs(),
            "{a} against {b}: {} relative",
            (a - b).abs() / b.abs()
        );
    }

    /// `χ²(2)`'s tail is `e^{−s/2}`, in closed form.
    #[test]
    fn two_degrees_of_freedom_is_the_exponential() {
        for s in [1e-6, 0.1, 1.0, 2.0, 3.5, 10.0, 50.0, 200.0, 700.0] {
            close(chi2_sf(s, 2.0), (-s / 2.0).exp(), 1e-13);
        }
    }

    /// `Q(n, x) = e^{−x} Σ_{i<n} x^i / i!` at an integer `n`, on both
    /// sides of `x = n + 1`, where the series and the fraction part.
    #[test]
    fn an_integer_shape_is_the_poisson_sum() {
        for n in [1_u32, 2, 3, 5, 8] {
            for x in [0.3, 1.0, 2.5, 4.0, 7.0, 12.0, 30.0] {
                let mut term = 1.0;
                let mut sum = 0.0;
                for i in 0..n {
                    if i > 0 {
                        term *= x / f64::from(i);
                    }
                    sum += term;
                }
                close(gamma_q(f64::from(n), x), (-x).exp() * sum, 1e-13);
            }
        }
    }

    /// `F(2, d₂)`'s tail is `(1 + 2f/d₂)^{−d₂/2}` and `F(d₁, 2)`'s is
    /// `1 − (d₁f / (d₁f + 2))^{d₁/2}`, in closed form.
    #[test]
    fn f_with_two_degrees_of_freedom_has_a_closed_form() {
        for d2 in [1.0, 3.0, 7.5, 40.0, 1000.0] {
            for f in [0.01, 0.5, 1.0, 3.0, 20.0, 400.0] {
                close(
                    f_sf(f, 2.0, d2),
                    (1.0 + 2.0 * f / d2).powf(-d2 / 2.0),
                    1e-12,
                );
            }
        }
        for d1 in [1.0, 4.0, 9.0] {
            for f in [0.01, 0.5, 1.0, 3.0] {
                let x: f64 = d1 * f / (d1 * f + 2.0);
                close(f_sf(f, d1, 2.0), 1.0 - x.powf(d1 / 2.0), 1e-12);
            }
        }
    }

    /// `I_x(a, b) + I_{1−x}(b, a) = 1`, and `I_x(a, 1) = x^a`.
    #[test]
    fn the_incomplete_beta_keeps_its_identities() {
        for (a, b) in [(0.5, 0.5), (2.0, 3.0), (10.0, 0.7), (50.0, 80.0)] {
            for x in [0.01, 0.2, 0.5, 0.8, 0.99] {
                let s = beta_inc(a, b, x) + beta_inc(b, a, 1.0 - x);
                assert!((s - 1.0).abs() < 1e-13, "{a} {b} {x}: {s}");
            }
        }
        for a in [0.5, 1.0, 3.0, 12.0] {
            for x in [0.1, 0.5, 0.9] {
                close(beta_inc(a, 1.0, x), x.powf(a), 1e-13);
            }
        }
    }

    /// A far tail keeps its digits: `χ²(2)` at 1400 is `e^{−700}`, near the
    /// smallest normal double, and the F tail goes down with it.
    #[test]
    fn a_far_tail_is_relative() {
        close(chi2_sf(1400.0, 2.0), (-700.0_f64).exp(), 1e-12);
        let f = 1e6;
        close(
            f_sf(f, 2.0, 50.0),
            (1.0 + 2.0 * f / 50.0).powf(-25.0),
            1e-12,
        );
    }

    #[test]
    fn outside_the_domain_is_nan_and_the_edges_are_exact() {
        assert!(f_sf(1.0, 0.0, 3.0).is_nan());
        assert!(f_sf(1.0, 2.0, -1.0).is_nan());
        assert!(f_sf(f64::NAN, 2.0, 3.0).is_nan());
        assert!(chi2_sf(1.0, 0.0).is_nan());
        assert!(chi2_sf(f64::NAN, 1.0).is_nan());
        assert_eq!(f_sf(0.0, 2.0, 3.0), 1.0);
        assert_eq!(f_sf(f64::INFINITY, 2.0, 3.0), 0.0);
        assert_eq!(chi2_sf(0.0, 3.0), 1.0);
        assert_eq!(chi2_sf(f64::INFINITY, 3.0), 0.0);
        assert!(beta_inc(1.0, 1.0, 1.5).is_nan());
        assert!(gamma_q(-1.0, 1.0).is_nan());
        assert_eq!(beta_inc(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_inc(2.0, 3.0, 1.0), 1.0);
    }
}
