use super::*;

/// scipy 1.18.1's `stats.t`, `(df, x, value)`.
const SCIPY_PPF: [(f64, f64, f64); 3] = [
    (5.0, 0.05, -2.015048373333024),
    (3.0, 0.01, -4.5407028585681335),
    (40.0, 0.999, 3.3068777140858225),
];
const SCIPY_CDF: [(f64, f64, f64); 2] = [
    (4.0, -2.0, 0.05805826175840778),
    (17.5, 1.3, 0.8947681155114872),
];
/// `stats.t.logpdf(-3.1, 2.7)` and `stats.t.expect(lambda x: x, args=(5,),
/// ub=stats.t.ppf(0.05, 5), conditional=True)`.
const SCIPY_LOGPDF: (f64, f64, f64) = (2.7, -3.1, -3.8164200543913003);
const SCIPY_ES: (f64, f64, f64) = (5.0, 0.05, -2.89012894627307);

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

/// The closed forms, read at every `η` (the series' oracle above the
/// switch, where their cancellation costs little).
fn closed(eta: f64) -> (f64, f64, f64) {
    let nu = 1.0 / eta;
    let a = -nu * nu * (0.5 * (digamma(0.5 * (nu + 1.0)) - digamma(0.5 * nu)) - 0.5 * eta);
    let iee = nu.powi(4)
        * (0.25 * (trigamma(0.5 * nu) - trigamma(0.5 * (nu + 1.0)))
            - (nu + 5.0) / (2.0 * nu * (nu + 1.0) * (nu + 3.0)));
    let c = -ln_beta(0.5 * nu, 0.5) - 0.5 * nu.ln();
    (a, iee, c)
}

/// The series agree with the closed forms where both are good, either side
/// of the switch.
#[test]
fn the_series_meet_the_closed_forms() {
    for eta in [0.02, 0.025, 0.03, 0.035, 0.05] {
        let (a, iee, c) = closed(eta);
        assert!(close(horner(&A_SERIES, eta), a, 1e-9), "A {eta}");
        assert!(close(horner(&IEE_SERIES, eta), iee, 1e-8), "I {eta}");
        assert!(
            close(horner(&C_SERIES, eta) - HALF_LN_TAU, c, 1e-12),
            "C {eta}"
        );
    }
}

/// `η = 0` is the normal, term for term.
#[test]
fn eta_zero_is_the_normal() {
    for z in [-6.0, -1.5, -0.3, 0.0, 0.7, 2.2, 5.0] {
        let lf = -HALF_LN_TAU - 0.5 * z * z;
        assert!((log_density(z, 0.0) - lf).abs() < 1e-15, "{z}");
        let [m, l, e] = scores(z, 0.0);
        let q = z * z;
        assert!((m - z).abs() < 1e-15);
        assert!((l - (q - 1.0)).abs() < 1e-14);
        assert!((e - (q * q - 2.0 * q - 1.0) / 4.0).abs() < 1e-12, "{z}");
        assert_eq!(cdf(z, 0.0), norm_cdf(z));
    }
    assert_eq!(expected_info(0.0), [1.0, 2.0, 2.0, 3.5]);
    assert_eq!(ppf(0.05, 0.0), ndtri(0.05));
}

#[test]
fn it_meets_scipy() {
    for (df, p, v) in SCIPY_PPF {
        assert!(close(ppf(p, 1.0 / df), v, 1e-12), "ppf {df} {p}");
    }
    for (df, z, v) in SCIPY_CDF {
        assert!(close(cdf(z, 1.0 / df), v, 1e-12), "cdf {df} {z}");
    }
    let (df, z, v) = SCIPY_LOGPDF;
    assert!(close(log_density(z, 1.0 / df), v, 1e-13));
    let (df, a, v) = SCIPY_ES;
    assert!(close(lower_es(a, 1.0 / df), v, 1e-10));
}

/// Each score against a central difference of the log density, across the
/// series' switch and both of `G`'s branches.
#[test]
fn the_scores_are_the_log_densitys_derivatives() {
    for eta in [1e-7, 1e-3, 0.02, 0.031, 0.032, 0.1, 0.4, 1.5] {
        for z in [-30.0, -4.0, -1.3, -0.2, 0.05, 0.9, 2.5, 7.0] {
            let [m, l, e] = scores(z, eta);
            let h = 1e-6;
            // z = (y − loc)/scale: ∂/∂loc · scale = −∂/∂z, ∂/∂ln s = −1 − z ∂/∂z
            let dz = (log_density(z + h, eta) - log_density(z - h, eta)) / (2.0 * h);
            assert!((m + dz).abs() < 1e-7, "loc {eta} {z}");
            assert!((l - (-1.0 - z * dz)).abs() < 2e-6, "ln {eta} {z}: {l}");
            let he = 1e-6 * eta.max(1e-3);
            let lo = (eta - he).max(0.0);
            let num = (log_density(z, eta + he) - log_density(z, lo)) / (eta + he - lo);
            assert!(
                (e - num).abs() < 1e-5 * num.abs().max(1.0),
                "eta {eta} {z}: {e} {num}"
            );
        }
    }
}

/// `∫ f s sᵀ` over a midpoint grid against [`expected_info`], and `∫ f s`
/// is 0 (the scores' mean).
#[test]
fn the_information_is_the_scores_variance() {
    for eta in [0.0, 1e-3, 0.03, 0.05, 0.1, 0.2] {
        let (h, lim) = (2e-3, 400.0);
        let mut acc = [0.0f64; 7];
        let mut z = -lim + 0.5 * h;
        while z < lim {
            let f = libm::exp(log_density(z, eta)) * h;
            let [m, l, e] = scores(z, eta);
            for (a, v) in acc.iter_mut().zip([m * m, l * l, l * e, e * e, m, l, e]) {
                *a += f * v;
            }
            z += h;
        }
        let want = expected_info(eta);
        for i in 0..4 {
            assert!(
                (acc[i] - want[i]).abs() < 1e-4 * want[i],
                "{eta} {i}: {} {}",
                acc[i],
                want[i]
            );
        }
        for (i, a) in acc[4..].iter().enumerate() {
            assert!(a.abs() < 1e-5, "mean {eta} {i}: {a}");
        }
    }
}

/// The quantile inverts the CDF, through the expansion's switch.
#[test]
fn the_quantile_inverts_the_cdf() {
    for eta in [1e-9, 9e-6, 1.1e-5, 1e-4, 0.03, 0.2, 0.5, 1.0, 3.0] {
        for p in [1e-6, 0.01, 0.05, 0.3, 0.5, 0.77, 0.95, 0.999] {
            let t = ppf(p, eta);
            assert!(close(cdf(t, eta), p, 1e-11), "{eta} {p}: {}", cdf(t, eta));
        }
    }
}

/// Fisher's expansion against the incomplete beta on either side of the
/// switch: its error is of order `η³`.
#[test]
fn the_expansion_meets_the_incomplete_beta() {
    for eta in [1e-5, 1e-4] {
        let nu = 1.0 / eta;
        for p in [1e-4f64, 0.01, 0.2, 0.6, 0.99] {
            let x = beta_inc_inv(0.5 * nu, 0.5, 2.0 * p.min(1.0 - p));
            let exact = (nu * (1.0 - x) / x).sqrt() * if p < 0.5 { -1.0 } else { 1.0 };
            let ex = expand(ndtri(p), eta);
            assert!((ex - exact).abs() < 1e-11, "{eta} {p}: {ex} {exact}");
        }
    }
}

/// The density integrates to 1, and the shortfall to its definition.
#[test]
fn the_density_and_shortfall_integrate() {
    for eta in [0.0, 0.01, 0.2, 0.25] {
        let h = 1e-3;
        let (mut m0, mut tail) = (0.0, 0.0);
        let a = 0.05;
        let q = ppf(a, eta);
        let mut z = -2000.0 + 0.5 * h;
        while z < 2000.0 {
            let f = libm::exp(log_density(z, eta)) * h;
            m0 += f;
            if z < q {
                tail += f * z;
            }
            z += h;
        }
        assert!((m0 - 1.0).abs() < 1e-6, "{eta}: {m0}");
        assert!((tail / a - lower_es(a, eta)).abs() < 1e-3, "{eta}");
    }
    assert_eq!(lower_es(0.05, 1.0), f64::NEG_INFINITY);
}
