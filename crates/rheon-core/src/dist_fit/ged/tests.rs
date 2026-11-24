use super::*;

/// Simpson's rule on `[a, b]` with `n` (even) panels.
fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let x = a + h * i as f64;
        s += if i % 2 == 1 { 4.0 * f(x) } else { 2.0 * f(x) };
    }
    s * h / 3.0
}

const NUS: [f64; 5] = [1.0, 1.5, 2.0, 3.0, 4.0];
const L: f64 = 60.0;
const N: usize = 240_000;

#[test]
fn the_density_integrates_to_one() {
    for nu in NUS {
        let total = simpson(|z| log_density(z, nu).exp(), -L, L, N);
        assert!((total - 1.0).abs() < 1e-7, "nu {nu}: {total}");
    }
}

#[test]
fn the_variance_is_gamma_three_over_gamma_one() {
    for nu in NUS {
        let g = 1.0 / nu;
        let var = simpson(|z| z * z * log_density(z, nu).exp(), -L, L, N);
        let want = (ln_gamma(3.0 * g) - ln_gamma(g)).exp();
        assert!(
            (var - want).abs() < 1e-6 * want.max(1.0),
            "nu {nu}: {var} vs {want}"
        );
    }
}

#[test]
fn the_nu_two_case_is_the_normal_of_variance_one_half() {
    // ν = 2 at s = 1: f(z) = e^{−z²}/√π, the normal of variance ½.
    for z in [-1.3f64, 0.0, 0.4, 2.0] {
        let want = (-z * z).exp() / std::f64::consts::PI.sqrt();
        let got = log_density(z, 2.0).exp();
        assert!((got - want).abs() < 1e-14, "{z}: {got} vs {want}");
    }
}

#[test]
fn the_scores_are_the_derivatives_of_the_log_density() {
    let h = 1e-6;
    for nu in NUS {
        for z in [-2.3, -0.7, 0.4, 1.9] {
            let s = scores(z, nu);
            // z = (y − μ)/s, so ∂/∂μ = −(1/s) ∂/∂z, with s = 1 here.
            let dz = (log_density(z + h, nu) - log_density(z - h, nu)) / (2.0 * h);
            assert!(
                (s[0] + dz).abs() < 1e-6 * dz.abs().max(1.0),
                "mu nu {nu} z {z}"
            );
            // ∂/∂ν at fixed z.
            let dnu = (log_density(z, nu + h) - log_density(z, nu - h)) / (2.0 * h);
            assert!(
                (s[2] - dnu).abs() < 1e-6 * dnu.abs().max(1.0),
                "nu nu {nu} z {z}: {} vs {dnu}",
                s[2]
            );
        }
    }
}

#[test]
fn the_expected_information_is_the_mean_outer_product_of_the_scores() {
    for nu in NUS {
        let info = expected_info(nu);
        // The μ score jumps at 0 where ν = 1, and `scores` takes 0 there by
        // convention. Each half-line is integrated on its own, starting just
        // off 0, so no node sits on the convention's point.
        let e = |f: &dyn Fn(&[f64; 3]) -> f64| {
            let g = |z: f64| {
                let s = scores(z, nu);
                f(&s) * log_density(z, nu).exp()
            };
            simpson(g, -L, -1e-9, N / 2) + simpson(g, 1e-9, L, N / 2)
        };
        let mm = e(&|s| s[0] * s[0]);
        let ss = e(&|s| (s[1]) * (s[1]));
        let sn = e(&|s| s[1] * s[2]);
        let nn = e(&|s| s[2] * s[2]);
        for (name, got, want) in [
            ("mm", info[0], mm),
            ("ss", info[3], ss),
            ("sn", info[4], sn),
            ("nn", info[5], nn),
        ] {
            assert!(
                (got - want).abs() < 1e-5 * want.abs().max(1.0),
                "nu {nu} {name}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn the_cdf_is_the_integral_of_the_density_and_the_quantile_inverts_it() {
    for nu in NUS {
        for z in [-3.0, -1.0, 0.5, 2.2] {
            let direct = simpson(|x| log_density(x, nu).exp(), -L, z, N);
            let c = cdf(z, nu);
            assert!((c - direct).abs() < 1e-6, "nu {nu} z {z}: {c} vs {direct}");
            let back = ppf(c, nu);
            assert!(
                (back - z).abs() < 1e-6 * z.abs().max(1.0),
                "nu {nu} z {z}: back {back}"
            );
        }
    }
}

#[test]
fn the_lower_shortfall_is_the_mean_of_the_tail() {
    for nu in NUS {
        for alpha in [0.05, 0.2] {
            let q = ppf(alpha, nu);
            let tail_mean = simpson(|z| z * log_density(z, nu).exp(), -L, q, N) / alpha;
            let got = lower_es(alpha, nu);
            assert!(
                (got - tail_mean).abs() < 1e-5 * tail_mean.abs().max(1.0),
                "nu {nu} alpha {alpha}: {got} vs {tail_mean}"
            );
        }
    }
}
