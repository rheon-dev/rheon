use super::*;

/// `arch` 9.0.0's `SkewStudent`, `(ν, λ)` and at `z = −2.5, −0.3, 0.4, 3.0`
/// its `loglikelihood(individual=True)` (variance 1) and `cdf`; its `ppf`
/// at `0.01, 0.3, 0.95`; and `partial_moment(1, ppf(0.05)) / 0.05`.
#[allow(clippy::type_complexity)]
const ARCH: [((f64, f64), [f64; 4], [f64; 4], [f64; 3], f64); 3] = [
    (
        (6.0, -0.2),
        [
            -3.8034583866355827,
            -0.9581679502537283,
            -0.7582216137656614,
            -5.544397483022525,
        ],
        [
            0.016492343355085724,
            0.3359678112939957,
            0.6494141033164114,
            0.9977304542930165,
        ],
        [
            -2.8781813818032704,
            -0.39714738725619836,
            1.4426312480265864,
        ],
        -2.4572945735170424,
    ),
    (
        (3.5, 0.5),
        [
            -6.27325596877375,
            -0.5165670277774499,
            -1.1676718798349008,
            -4.448874660601654,
        ],
        [
            0.0010903297371271593,
            0.43897641559611783,
            0.7596195921349691,
            0.9860289065250658,
        ],
        [-1.5226483136326219, -0.5232410222042676, 1.6776923337693037],
        -1.3693344433064496,
    ),
    (
        (30.0, 0.9),
        [
            -35.54531558189925,
            -0.897085066926007,
            -1.2655988023279496,
            -4.1216658048639285,
        ],
        [
            1.7627271319080452e-17,
            0.46321775442656654,
            0.7059228874048058,
            0.9895836213911957,
        ],
        [-1.3082141591474459, -0.6759139770453981, 1.9246686402219875],
        -1.2681086069249115,
    ),
];

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

#[test]
fn it_meets_arch() {
    for ((nu, lam), ll, cdf, ppf, es) in ARCH {
        let sh = Shape::new(1.0 / nu, lam);
        for (k, z) in [-2.5, -0.3, 0.4, 3.0].into_iter().enumerate() {
            assert!(close(sh.log_density(z), ll[k], 1e-12), "ll {nu} {lam} {z}");
            let rel = if cdf[k] < 1e-10 { 1e-9 } else { 1e-12 };
            assert!(
                (sh.cdf(z) - cdf[k]).abs() <= rel * cdf[k],
                "cdf {nu} {lam} {z}: {} {}",
                sh.cdf(z),
                cdf[k]
            );
        }
        for (k, p) in [0.01, 0.3, 0.95].into_iter().enumerate() {
            assert!(close(sh.ppf(p), ppf[k], 1e-11), "ppf {nu} {lam} {p}");
        }
        assert!(close(sh.lower_es(0.05), es, 1e-10), "es {nu} {lam}");
    }
}

/// The density integrates to 1 with mean 0 and variance 1, and its
/// skewness and excess kurtosis are [`Shape::skew_kurt`]'s (midpoint rule
/// over `[−400, 400]`), across the tail and the skew, the normal's tail
/// included.
#[test]
fn the_density_is_standardized_and_its_moments_are_closed() {
    for (eta, lam) in [
        (0.0, 0.0),
        (0.0, 0.6),
        (0.1, -0.3),
        (0.15, 0.8),
        (0.02, -0.95),
    ] {
        let sh = Shape::new(eta, lam);
        let mut m = [0.0f64; 5];
        let h = 1e-3;
        let mut z = -400.0 + 0.5 * h;
        while z < 400.0 {
            let f = libm::exp(sh.log_density(z)) * h;
            let mut p = f;
            for mk in &mut m {
                *mk += p;
                p *= z;
            }
            z += h;
        }
        assert!((m[0] - 1.0).abs() < 1e-8, "{eta} {lam}: {m:?}");
        assert!(m[1].abs() < 1e-7, "{eta} {lam}: {m:?}");
        assert!((m[2] - 1.0).abs() < 1e-6, "{eta} {lam}: {m:?}");
        let (skew, kurt) = sh.skew_kurt();
        assert!((m[3] - skew).abs() < 1e-4, "{eta} {lam}: {} {skew}", m[3]);
        // the tail at η = 0.15 (ν ≈ 6.7) is slow to integrate to its fourth
        // moment
        let tol = if eta > 0.1 { 2e-2 } else { 1e-4 };
        assert!(
            (m[4] - 3.0 - kurt).abs() < tol,
            "{eta} {lam}: {} {kurt}",
            m[4]
        );
    }
}

/// `λ = 0` is the t of variance 1, and `η = 0, λ = 0` the normal.
#[test]
fn its_special_cases_are_the_t_and_the_normal() {
    for eta in [0.0, 0.01, 0.2, 0.45] {
        let sh = Shape::new(eta, 0.0);
        assert_eq!(sh.a, 0.0);
        assert_eq!(sh.b, 1.0);
        let k = (1.0 - 2.0 * eta).sqrt();
        for z in [-3.0, -0.5, 0.0, 1.2, 4.0] {
            let t = student_t::log_density(z / k, eta) - k.ln();
            assert!((sh.log_density(z) - t).abs() < 1e-14, "{eta} {z}");
            assert!((sh.cdf(z) - student_t::cdf(z / k, eta)).abs() < 1e-15);
        }
    }
    let sh = Shape::new(0.0, 0.0);
    for z in [-2.0, 0.3] {
        let lf = -0.918_938_533_204_672_8 - 0.5 * z * z;
        assert!((sh.log_density(z) - lf).abs() < 1e-15);
    }
}

/// `log f(y; θ)` at `θ = (μ, ln σ, η, λ)`.
fn log_f(y: f64, th: [f64; 4]) -> f64 {
    let sh = Shape::new(th[2], th[3]);
    sh.log_density((y - th[0]) / th[1].exp()) - th[1]
}

/// Each score against a central difference of the log density, on both
/// pieces, at the normal's tail and near the bounds.
#[test]
fn the_scores_are_the_log_densitys_derivatives() {
    for (eta, lam) in [
        (0.0, 0.0),
        (0.0, -0.4),
        (0.2, 0.0),
        (0.25, -0.2),
        (0.05, 0.7),
        (0.4, 0.95),
        (0.01, -0.9),
    ] {
        for (mu, ls) in [(0.0, 0.0), (1.5, -0.7)] {
            let th = [mu, ls, eta, lam];
            let s = f64::exp(ls);
            let sh = Shape::new(eta, lam);
            for z in [-5.0, -1.7, -0.4, 0.15, 0.8, 2.6, 9.0] {
                // away from the kink, where the central difference straddles
                // the two pieces
                if (sh.b * z + sh.a).abs() < 1e-3 {
                    continue;
                }
                let y = mu + s * z;
                let (g, _) = sh.row(z, &sh.derivs());
                let g = [g[0] / s, g[1], g[2], g[3]];
                for k in 0..4 {
                    let h = 1e-6;
                    let (mut up, mut dn) = (th, th);
                    up[k] += h;
                    dn[k] -= h;
                    if k == 2 && eta == 0.0 {
                        // one-sided at the boundary
                        // whose error is of order h times the second
                        // derivative, which grows as z⁶
                        let num = (log_f(y, up) - log_f(y, th)) / h;
                        assert!(
                            (g[k] - num).abs() < 1e-3 * num.abs().max(1.0),
                            "{k} {eta} {lam} {z}: {} {num}",
                            g[k]
                        );
                        continue;
                    }
                    let num = (log_f(y, up) - log_f(y, dn)) / (2.0 * h);
                    assert!(
                        (g[k] - num).abs() < 1e-6 * num.abs().max(1.0),
                        "{k} {eta} {lam} {z}: {} {num}",
                        g[k]
                    );
                }
            }
        }
    }
}

/// The closed information is the expectation of the scores' outer
/// product, by quadrature over `z` (midpoint rule over `[−60, 60]`, split
/// at the kink), and positive definite.
#[test]
fn the_information_is_the_scores_outer_product() {
    for (eta, lam) in [
        (0.0, 0.0),
        (0.0, 0.5),
        (0.1, -0.3),
        (0.3, 0.6),
        (0.02, -0.9),
    ] {
        let sh = Shape::new(eta, lam);
        let d = sh.derivs();
        let mut want = [0.0f64; 10];
        let h = 2e-4;
        let kink = -sh.a / sh.b;
        for (lo, hi) in [(-60.0, kink), (kink, 60.0)] {
            let n = ((hi - lo) / h).ceil();
            let step = (hi - lo) / n;
            let mut z = lo + 0.5 * step;
            while z < hi {
                let f = libm::exp(sh.log_density(z)) * step;
                let (g, _) = sh.row(z, &d);
                let mut at = 0;
                for r in 0..4 {
                    for c in r..4 {
                        want[at] += f * g[r] * g[c];
                        at += 1;
                    }
                }
                z += step;
            }
        }
        let got = sh.info(&d);
        for k in 0..10 {
            assert!(
                (got[k] - want[k]).abs() < 2e-4 * want[k].abs().max(1.0),
                "{eta} {lam} [{k}]: {} {}",
                got[k],
                want[k]
            );
        }
        let m = [
            [got[0], got[1], got[2], got[3]],
            [got[1], got[4], got[5], got[6]],
            [got[2], got[5], got[7], got[8]],
            [got[3], got[6], got[8], got[9]],
        ];
        // positive definite: Cholesky's pivots
        let mut l = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..=i {
                let s = m[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
                if i == j {
                    assert!(s > 0.0, "{eta} {lam}: pivot {i} {s}");
                    l[i][i] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
    }
}

/// The quantile inverts the distribution function, and the shortfall is
/// the mean below the quantile, by quadrature.
#[test]
fn the_quantile_and_shortfall_are_their_definitions() {
    for (eta, lam) in [(0.0, 0.0), (0.0, -0.6), (0.2, 0.3), (0.45, -0.95)] {
        let sh = Shape::new(eta, lam);
        for p in [1e-4, 0.01, 0.2, 0.5, 0.7, 0.99] {
            let z = sh.ppf(p);
            assert!(close(sh.cdf(z), p, 1e-12), "{eta} {lam} {p}");
        }
        for alpha in [0.01, 0.05, 0.6] {
            let q = sh.ppf(alpha);
            let (lo, n) = (-400.0, 2_000_000);
            let h = (q - lo) / f64::from(n);
            let mut acc = 0.0;
            for i in 0..n {
                let z = lo + (f64::from(i) + 0.5) * h;
                acc += z * libm::exp(sh.log_density(z)) * h;
            }
            let want = acc / alpha;
            // At ν = 2.2 the part of the mean below −400, which the
            // quadrature leaves out, is about 0.3% of it.
            let tol = if eta > 0.4 { 1e-2 } else { 1e-7 };
            assert!(
                (sh.lower_es(alpha) - want).abs() < tol * want.abs(),
                "{eta} {lam} {alpha}: {} {want}",
                sh.lower_es(alpha)
            );
        }
    }
}

/// The skewness rises with `λ`, and at `λ = 0` the excess kurtosis is the
/// moment start's `6η/(1 − 4η)`.
#[test]
fn the_skewness_rises_with_the_skew_and_the_kurtosis_is_the_starts() {
    for eta in [0.0, 0.1, 0.2] {
        let mut last = f64::NEG_INFINITY;
        for i in -99..=99 {
            let (s, _) = Shape::new(eta, f64::from(i) / 100.0).skew_kurt();
            assert!(s > last, "{eta} {i}");
            last = s;
        }
        let (s, k) = Shape::new(eta, 0.0).skew_kurt();
        assert_eq!(s, 0.0);
        assert!((k - 6.0 * eta / (1.0 - 4.0 * eta)).abs() < 1e-13, "{eta}");
    }
}
