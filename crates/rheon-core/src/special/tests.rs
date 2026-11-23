//! The special functions against their closed forms, their identities and
//! each other (PLAN 558, slice A). The agreement with `scipy.special` over
//! grids, in ulps, is `tests/test_second_opinion_special.py`'s
//! (`TestSpecialFunctionsAreScipys`): Rust tests may not call scipy.

use super::*;

/// The distance of `got` from `want` in units of `want`'s last place.
fn ulps(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    let ulp = (want.abs().next_up() - want.abs()).max(f64::MIN_POSITIVE * f64::EPSILON);
    (got - want).abs() / ulp
}

/// The log-spaced grid `lo · r^k` up to `hi`.
fn grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let r = (hi / lo).powf(1.0 / (n - 1) as f64);
    (0..n).map(|k| lo * r.powi(k as i32)).collect()
}

#[test]
fn trigamma_meets_its_closed_values_and_is_digammas_derivative() {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    assert!(ulps(trigamma(1.0), pi2 / 6.0) <= 4.0, "{}", trigamma(1.0));
    assert!(ulps(trigamma(0.5), pi2 / 2.0) <= 4.0, "{}", trigamma(0.5));
    for x in grid(1e-3, 1e4, 60) {
        // ψ′(x) = ψ′(x + 1) + 1/x²
        let (l, r) = (trigamma(x), trigamma(x + 1.0) + 1.0 / (x * x));
        assert!((l - r).abs() <= 8.0 * f64::EPSILON * l, "{x}: {l} {r}");
        let h = 1e-4 * x;
        let num = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
        assert!((num - trigamma(x)).abs() <= 1e-6 * trigamma(x), "{x}");
    }
}

#[test]
fn ln_gamma_1p_keeps_its_digits_where_ln_gamma_crosses_zero() {
    // ln Γ(1 + a) = −γ a + O(a²); Γ(2) = 1 at a = 1; Γ(3/2) = √π/2.
    let a = 1e-12;
    let want = -std::f64::consts::EULER_GAMMA * a + 0.822_467_033_424_113_2 * a * a;
    assert!(ulps(ln_gamma_1p(a), want) <= 2.0, "{}", ln_gamma_1p(a));
    assert_eq!(ln_gamma_1p(0.0), 0.0);
    assert!(ln_gamma_1p(1.0).abs() <= 1e-16, "{}", ln_gamma_1p(1.0));
    // ln(√π/2) = ½ ln π − ln 2, to 20 digits.
    let half = -0.120_782_237_635_245_22;
    assert!(ulps(ln_gamma_1p(0.5), half) <= 4.0, "{}", ln_gamma_1p(0.5));
    // and meets ln_gamma away from the zeros
    for a in [0.1, 0.3, 0.7, 0.9] {
        assert!((ln_gamma_1p(a) - ln_gamma(1.0 + a)).abs() < 1e-14, "{a}");
    }
}

#[test]
fn the_incomplete_gamma_meets_its_closed_forms() {
    let mut worst = [0.0f64; 4];
    for x in grid(1e-6, 700.0, 200) {
        // a = 1: the exponential distribution
        worst[0] = worst[0].max(ulps(gamma_q(1.0, x), libm::exp(-x)));
        worst[1] = worst[1].max(ulps(gamma_p(1.0, x), -libm::expm1(-x)));
        // a = 1/2: P = erf(√x), Q = erfc(√x), with √x's rounding `s_lo`
        // corrected to first order (the rounded root alone moves erfc by
        // `x` ulps out here).
        let s = x.sqrt();
        let s_lo = s.mul_add(-s, x) / (2.0 * s);
        let slope = std::f64::consts::FRAC_2_SQRT_PI * libm::exp(-x) * s_lo;
        worst[2] = worst[2].max(ulps(gamma_q(0.5, x), libm::erfc(s) - slope));
        worst[3] = worst[3].max(ulps(gamma_p(0.5, x), libm::erf(s) + slope));
    }
    // The bounds are the measured worst (5, 5, 18, 6 ulps, x86-64 Linux,
    // 2026-11-23) rounded up: the closed forms are themselves the `libm`
    // crate's, within an ulp or two, and erf's argument is corrected for
    // its root's rounding only to first order.
    for (w, bound) in worst.iter().zip([8.0, 8.0, 24.0, 8.0]) {
        assert!(*w <= bound, "{w} ulps against {bound}");
    }
    // Integer a: Q(n, x) = e⁻ˣ Σ_{k<n} xᵏ/k!.
    for n in [2u32, 5, 12, 30] {
        for x in [0.5, 3.0, 11.0, 29.0, 60.0] {
            let mut term = 1.0;
            let mut sum = 1.0;
            for k in 1..n {
                term *= x / f64::from(k);
                sum += term;
            }
            let want = libm::exp(-x) * sum;
            let got = gamma_q(f64::from(n), x);
            assert!(
                (got - want).abs() <= 1e-13 * want,
                "Q({n}, {x}) {got} {want}"
            );
            assert!((gamma_p(f64::from(n), x) + got - 1.0).abs() <= 2.0 * f64::EPSILON);
        }
    }
}

#[test]
fn the_incomplete_beta_meets_its_closed_forms() {
    let mut worst = [0.0f64; 3];
    for x in grid(1e-8, 1.0 - 1e-8, 150) {
        for a in [0.05, 0.5, 2.0, 37.5] {
            // b = 1: I = xᵃ
            worst[0] = worst[0].max(ulps(beta_inc(a, 1.0, x), libm::pow(x, a)));
            // a = 1: 1 − I = (1 − x)ᵇ
            // (1 − x)ᵇ, with `1 − x` exact from ½ up
            let comp = if x >= 0.5 {
                libm::pow(1.0 - x, a)
            } else {
                libm::exp(a * libm::log1p(-x))
            };
            worst[1] = worst[1].max(ulps(beta_inc_c(1.0, a, x), comp));
        }
        // a = b = ½: the arcsine law, I = (2/π) asin(√x), read in the
        // smaller tail (1 − I at √(1 − x), exact from ½ up) with the root's
        // rounding corrected to first order.
        let (v, got) = if x < 0.5 {
            (x, beta_inc(0.5, 0.5, x))
        } else {
            (1.0 - x, beta_inc_c(0.5, 0.5, x))
        };
        let s = v.sqrt();
        let s_lo = s.mul_add(-s, v) / (2.0 * s);
        let arcsine = 2.0 / std::f64::consts::PI * (libm::asin(s) + s_lo / (1.0 - v).sqrt());
        worst[2] = worst[2].max(ulps(got, arcsine));
    }
    // Measured worst 6, 29 and 6 ulps (x86-64 Linux, 2026-11-23), rounded
    // up; the second is the continued fraction's own rounding over its
    // terms at b = 37.5.
    for (w, bound) in worst.iter().zip([8.0, 32.0, 8.0]) {
        assert!(*w <= bound, "{w} ulps against {bound}");
    }
    // The symmetry I_x(a, b) = 1 − I_{1−x}(b, a), at the t's arguments.
    for nu in [2.05, 5.0, 30.0, 300.0] {
        // x and 1 − x both exact
        for x in [0.0078125, 0.25, 0.875, 0.9990234375] {
            let (l, r) = (
                beta_inc(nu / 2.0, 0.5, x),
                beta_inc_c(0.5, nu / 2.0, 1.0 - x),
            );
            assert!((l - r).abs() <= 1e-14 * l.max(1e-300), "{nu} {x}: {l} {r}");
        }
    }
}

#[test]
fn the_inverses_return_their_arguments() {
    for a in [1e-3, 0.02, 0.5, 1.0, 3.7, 50.0, 400.0] {
        for p in [1e-300, 1e-30, 1e-8, 0.01, 0.3, 0.5, 0.7, 0.99] {
            let x = gamma_p_inv(a, p);
            assert!(x.is_finite() && x >= 0.0, "P⁻¹({a}, {p}) = {x}");
            if x > 0.0 && x.is_normal() {
                let back = gamma_p(a, x);
                assert!((back - p).abs() <= 1e-12 * p, "P({a}, P⁻¹) {back} vs {p}");
            }
            // For a small `a` the lower quantiles underflow: P(10⁻³, x) =
            // 0.3 at x ≈ 0.3¹⁰⁰⁰.
            let x = gamma_q_inv(a, p);
            assert!(x.is_finite() && x >= 0.0, "Q⁻¹({a}, {p}) = {x}");
            if x.is_normal() {
                let back = gamma_q(a, x);
                assert!((back - p).abs() <= 1e-12 * p, "Q({a}, Q⁻¹) {back} vs {p}");
            }
        }
    }
    for (a, b) in [
        (1.025, 0.5),
        (0.5, 1.025),
        (2.5, 0.5),
        (15.0, 0.5),
        (150.0, 0.5),
        (0.3, 7.0),
        (40.0, 60.0),
    ] {
        for p in [1e-200, 1e-12, 0.01, 0.3, 0.5, 0.8, 0.999] {
            // I⁻¹(½, b, 10⁻²⁰⁰) ≈ 10⁻⁴⁰⁰ underflows to 0
            let x = beta_inc_inv(a, b, p);
            assert!((0.0..=1.0).contains(&x), "I⁻¹({a}, {b}, {p}) = {x}");
            if x.is_normal() {
                let back = beta_inc(a, b, x);
                // Above ½ the root sits near 1 for a large `a`, where `x`
                // cannot hold `1 − x` to many digits: the documented call
                // for that tail is the swapped one, checked below.
                let tol = if p < 0.5 {
                    1e-12 * p
                } else {
                    1e-12 + 4.0 * f64::EPSILON * x * (b * a).sqrt() / (1.0 - x).max(1e-300).sqrt()
                };
                assert!((back - p).abs() <= tol, "I({a}, {b}, I⁻¹) {back} vs {p}");
            }
            // 1 − x = I⁻¹(b, a, 1 − p), with `1 − p` exact from ½ up
            if p >= 0.5 {
                let y = beta_inc_inv(b, a, 1.0 - p);
                let back = beta_inc(b, a, y);
                assert!(
                    (back - (1.0 - p)).abs() <= 1e-12 * (1.0 - p),
                    "{a} {b} {p}: {y}"
                );
            }
        }
    }
    for p in [1e-300, 1e-20, 0.001, 0.02425, 0.3, 0.5, 0.9, 1.0 - 1e-12] {
        let z = ndtri(p);
        let back = if p > 0.5 {
            1.0 - norm_cdf(-z)
        } else {
            norm_cdf(z)
        };
        // Φ's relative slope is about z², so z's rounding alone moves Φ by
        // z² · ε/2 relative.
        let tail = p.min(1.0 - p);
        let tol = 4.0 * f64::EPSILON * (1.0 + z * z) * tail;
        assert!((back - p).abs() <= tol, "{p}: {z} gives {back}");
        if 1.0 - (1.0 - p) == p {
            assert_eq!(ndtri(1.0 - p), -ndtri(p), "odd at {p}");
        }
    }
}

/// The edges: the limits the functions document, and no NaN anywhere in
/// the domains a fit reads (`a` in `[1/500, 500]`, `x` in `[0, ∞]`).
#[test]
fn the_edges_are_their_limits_and_the_domains_have_no_nan() {
    assert_eq!((gamma_p(0.0, 1.0), gamma_q(0.0, 1.0)), (1.0, 0.0));
    assert_eq!((gamma_p(2.0, 0.0), gamma_q(2.0, 0.0)), (0.0, 1.0));
    assert_eq!(
        (gamma_p(2.0, f64::INFINITY), gamma_q(2.0, f64::INFINITY)),
        (1.0, 0.0)
    );
    assert!(gamma_p(-1.0, 1.0).is_nan() && gamma_p(1.0, -1.0).is_nan());
    assert!(gamma_p(f64::NAN, 1.0).is_nan());
    assert_eq!(gamma_p(1e-310, 2.0), 1.0);
    assert_eq!(gamma_p_inv(2.0, 0.0), 0.0);
    assert_eq!(gamma_p_inv(2.0, 1.0), f64::INFINITY);
    assert_eq!(gamma_q_inv(2.0, 0.0), f64::INFINITY);
    assert!(gamma_p_inv(0.0, 0.5).is_nan() && gamma_p_inv(1.0, 1.5).is_nan());
    assert_eq!(
        (beta_inc(2.0, 3.0, 0.0), beta_inc(2.0, 3.0, 1.0)),
        (0.0, 1.0)
    );
    assert_eq!(beta_inc(0.0, 3.0, 0.5), 1.0);
    assert_eq!(beta_inc(3.0, 0.0, 0.5), 0.0);
    assert!(beta_inc(0.0, 0.0, 0.5).is_nan() && beta_inc(1.0, 1.0, 1.5).is_nan());
    assert_eq!(
        (beta_inc_inv(2.0, 3.0, 0.0), beta_inc_inv(2.0, 3.0, 1.0)),
        (0.0, 1.0)
    );
    assert_eq!(ln_gamma(f64::INFINITY), f64::INFINITY);
    assert_eq!((ndtri(0.0), ndtri(1.0)), (f64::NEG_INFINITY, f64::INFINITY));
    for a in grid(1.0 / 500.0, 500.0, 80) {
        for x in grid(1e-300, 1e300, 120)
            .into_iter()
            .chain([0.0, f64::INFINITY])
        {
            let (p, q) = (gamma_p(a, x), gamma_q(a, x));
            assert!(
                (0.0..=1.0).contains(&p) && (0.0..=1.0).contains(&q),
                "{a} {x}: {p} {q}"
            );
        }
        for b in [0.5, 1.0, 7.0, 150.0] {
            for x in grid(1e-300, 0.5, 60)
                .into_iter()
                .chain([0.0, 0.75, 0.999_999, 1.0])
            {
                let i = beta_inc(a, b, x);
                assert!((0.0..=1.0).contains(&i), "I({a}, {b}, {x}) = {i}");
            }
        }
        for p in [1e-300, 1e-10, 0.5, 1.0 - 1e-10] {
            assert!(!gamma_p_inv(a, p).is_nan(), "P⁻¹({a}, {p})");
            assert!(!gamma_q_inv(a, p).is_nan(), "Q⁻¹({a}, {p})");
            assert!(!beta_inc_inv(a, 0.5, p).is_nan(), "I⁻¹({a}, ½, {p})");
        }
    }
}
