//! The shrinkage arithmetic against definitions written from the papers
//! and from scikit-learn's source, two-pass over the rows.

use super::*;
use crate::EwCov;

fn normal(g: &mut crate::SplitMix64) -> f64 {
    let u = 1.0 - g.uniform();
    let v = g.uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// `n` rows of `p` correlated, unequally scaled features about `offset`:
/// `x_i = offset + scale_i (z_i + 0.6 z_{i-1})`.
fn rows(seed: u64, n: usize, p: usize, offset: f64) -> Vec<Vec<f64>> {
    let mut g = crate::SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let z: Vec<f64> = (0..p).map(|_| normal(&mut g)).collect();
            (0..p)
                .map(|i| {
                    let prev = if i > 0 { z[i - 1] } else { 0.0 };
                    offset + (1.0 + i as f64 * 0.5) * (z[i] + 0.6 * prev)
                })
                .collect()
        })
        .collect()
}

/// The accumulator and the fourth moments after `xs` under decay `lam` and
/// weights `w`, stepped as the model steps them.
fn learn(xs: &[Vec<f64>], lam: f64, w: &[f64]) -> (EwCov, Fourth, Option<f64>) {
    let p = xs[0].len();
    let mut cov = EwCov::new(p);
    let mut fourth = Fourth::new(p);
    let mut kish = None;
    let mut cu = Vec::new();
    for (x, &wi) in xs.iter().zip(w) {
        if let Some((a, b)) = cov.mixing(lam, wi) {
            let d: Vec<f64> = (0..p).map(|i| cov.deviation(i, x[i])).collect();
            fourth.learn(cov.comoments().packed(), &d, a, b, &mut cu);
        }
        cov.update(x, lam, wi);
        if wi > 0.0 {
            kish = cov.n_kish();
        }
    }
    (cov, fourth, kish)
}

/// The weighted moments two-pass: the normalised weights `ω_k = w_k
/// λ^{age_k} / Σ`, the mean, `S`, `T = Σ ω |z|² z`, `F = Σ ω |z|⁴` and
/// Kish's `(Σ w λ^age)² / Σ (w λ^age)²`.
struct TwoPass {
    omega: Vec<f64>,
    z: Vec<Vec<f64>>,
    s: Vec<f64>,
    t: Vec<f64>,
    f: f64,
    kish: f64,
}

fn two_pass(xs: &[Vec<f64>], lam: f64, w: &[f64]) -> TwoPass {
    let n = xs.len();
    let p = xs[0].len();
    let raw: Vec<f64> = (0..n)
        .map(|k| w[k] * lam.powi((n - 1 - k) as i32))
        .collect();
    let total: f64 = raw.iter().sum();
    let sq: f64 = raw.iter().map(|v| v * v).sum();
    let omega: Vec<f64> = raw.iter().map(|v| v / total).collect();
    let m: Vec<f64> = (0..p)
        .map(|i| (0..n).map(|k| omega[k] * xs[k][i]).sum())
        .collect();
    let z: Vec<Vec<f64>> = xs
        .iter()
        .map(|x| x.iter().zip(&m).map(|(a, b)| a - b).collect())
        .collect();
    let mut s = vec![0.0; p * p];
    let mut t = vec![0.0; p];
    let mut f = 0.0;
    for k in 0..n {
        let zz: f64 = z[k].iter().map(|v| v * v).sum();
        for i in 0..p {
            t[i] += omega[k] * zz * z[k][i];
            for j in 0..p {
                s[i * p + j] += omega[k] * z[k][i] * z[k][j];
            }
        }
        f += omega[k] * zz * zz;
    }
    TwoPass {
        omega,
        z,
        s,
        t,
        f,
        kish: total * total / sq,
    }
}

fn packed(full: &[f64], p: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(row_at(p));
    for i in 0..p {
        for j in 0..=i {
            out.push(full[i * p + j]);
        }
    }
    out
}

/// Ledoit and Wolf (2004), §3.3, with the paper's norm `‖A‖² = tr(AAᵀ)/p`,
/// weighted: `m = ⟨S, I⟩`, `d² = ‖S − mI‖²`, `b̄² = (1/n) Σ ω_k ‖z_k z_kᵀ −
/// S‖²` (the paper's `(1/n²) Σ_k` at `ω_k = 1/n`), `b² = min(b̄², d²)`,
/// `s = b² / d²`.
fn paper_lw(tp: &TwoPass, p: usize, n: f64) -> f64 {
    let pf = p as f64;
    let norm2 = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>() / pf;
    let m: f64 = (0..p).map(|i| tp.s[i * p + i]).sum::<f64>() / pf;
    let mut sm = tp.s.clone();
    for i in 0..p {
        sm[i * p + i] -= m;
    }
    let d2 = norm2(&sm);
    let mut b_bar2 = 0.0;
    for (k, z) in tp.z.iter().enumerate() {
        let dev: Vec<f64> = (0..p * p)
            .map(|ij| z[ij / p] * z[ij % p] - tp.s[ij])
            .collect();
        b_bar2 += tp.omega[k] * norm2(&dev);
    }
    b_bar2 /= n;
    b_bar2.min(d2) / d2
}

/// Chen, Wiesel, Eldar and Hero (2010), Eq. 23, without the `2/p` terms, as
/// scikit-learn has it: `min((tr S² + tr² S) / ((n + 1)(tr S² − tr² S / p)), 1)`.
fn paper_oas(s: &[f64], p: usize, n: f64) -> f64 {
    let tr: f64 = (0..p).map(|i| s[i * p + i]).sum();
    let tr_s2: f64 = s.iter().map(|v| v * v).sum();
    let den = (n + 1.0) * (tr_s2 - tr * tr / p as f64);
    if den == 0.0 {
        1.0
    } else {
        ((tr_s2 + tr * tr) / den).min(1.0)
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

/// The recursion against the two-pass moments, under decay, unequal
/// weights, a zero-weight first row, zero-weight rows between, and an
/// offset of 1e3. Measured here, the worst over the cases below (seeds
/// 1-3, `p` 2, 5 and 20, both offsets): `F` 4.8e-14 relative, `T` 2.9e-13
/// of `√(F tr S)`, Kish 3.8e-15 relative; each bound is ten times its worst.
#[test]
fn the_fourth_moments_follow_the_two_pass_definition() {
    let mut worst = [0.0_f64; 3];
    for seed in 1..=3 {
        for p in [2, 5, 20] {
            for offset in [0.0, 1e3] {
                let xs = rows(seed, 300, p, offset);
                let mut g = crate::SplitMix64::new(seed + 100);
                let w: Vec<f64> = (0..xs.len())
                    .map(|k| {
                        if k == 0 || k % 7 == 3 {
                            0.0
                        } else {
                            0.2 + 2.0 * g.uniform()
                        }
                    })
                    .collect();
                let lam = 0.5f64.powf(1.0 / 40.0);
                let (cov, fourth, kish) = learn(&xs, lam, &w);
                let tp = two_pass(&xs, lam, &w);
                let tr: f64 = (0..p).map(|i| tp.s[i * p + i]).sum();
                let scale = (tp.f * tr).sqrt();
                worst[0] = worst[0].max(rel(fourth.f(), tp.f));
                for (a, b) in fourth.t().iter().zip(&tp.t) {
                    worst[1] = worst[1].max((a - b).abs() / scale);
                }
                worst[2] = worst[2].max(rel(kish.unwrap(), tp.kish));
                // The co-moments the intensity reads, for scale.
                let s = packed(&tp.s, p);
                for (a, b) in cov.comoments().packed().iter().zip(&s) {
                    assert!((a - b).abs() <= 1e-10 * tr, "{a} {b}");
                }
            }
        }
    }
    assert!(worst[0] < 5e-13, "F {}", worst[0]);
    assert!(worst[1] < 3e-12, "T {}", worst[1]);
    assert!(worst[2] < 4e-14, "Kish {}", worst[2]);
}

/// A row of weight 0 leaves the fourth moments to the bit, and so does the
/// first row (one row has no spread: both are 0 after it).
#[test]
fn a_zero_weight_row_moves_nothing_and_one_row_has_no_spread() {
    let xs = rows(4, 30, 4, 5.0);
    let lam = 0.9;
    let mut cov = EwCov::new(4);
    let mut fourth = Fourth::new(4);
    let mut cu = Vec::new();
    for (r, x) in xs.iter().enumerate() {
        for w in [1.0, 0.0] {
            let before = fourth.clone();
            if let Some((a, b)) = cov.mixing(lam, w) {
                let d: Vec<f64> = (0..4).map(|i| cov.deviation(i, x[i])).collect();
                fourth.learn(cov.comoments().packed(), &d, a, b, &mut cu);
            }
            cov.update(x, lam, w);
            if w == 0.0 {
                assert_eq!(fourth, before, "row {r}");
            }
            if r == 0 && w == 1.0 {
                assert_eq!(fourth.f(), 0.0);
                assert!(fourth.t().iter().all(|v| *v == 0.0));
            }
        }
    }
    assert!(fourth.f() > 0.0);
    // A zero-weight first row: no step at all.
    let cov = EwCov::new(3);
    assert_eq!(cov.mixing(lam, 0.0), None);
}

/// At unit weights and no decay both intensities are scikit-learn's, whose
/// arithmetic `ledoit_wolf_shrinkage` and `_oas` are written out here from
/// its source (scikit-learn 1.9.1, `sklearn/covariance/_shrunk_covariance.py`):
/// `emp_cov = Xcᵀ Xc / n`, `μ = tr / p`, `β = (Σ_k |x_k|⁴ / n − ‖S‖²) / (p n)`,
/// `δ = (‖S‖² − 2 μ tr + p μ²) / p`, `min(β, δ) / δ`; and OAS's `α =
/// mean(S²)`. The paper's definitions agree. Bound 1e-13 absolute: the
/// worst measured here is 1.0e-15 (two orders of summation of the same
/// sums), and the bound is a hundred times it.
#[test]
fn unit_weights_give_scikit_learns_intensities() {
    for seed in 5..=8 {
        for (p, n) in [(2, 40), (5, 12), (5, 200), (20, 30), (20, 400)] {
            let xs = rows(seed, n, p, 3.0);
            let w = vec![1.0; n];
            let (cov, fourth, kish) = learn(&xs, 1.0, &w);
            assert!((kish.unwrap() - n as f64).abs() < 1e-9);
            let c = cov.comoments().packed();
            let (tr, frob2) = trace_and_frobenius(c, p);
            let lw = ledoit_wolf(p, tr, frob2, fourth.f(), kish);
            let oa = oas(p, tr, frob2, kish);
            // scikit-learn's arithmetic, two-pass from the rows.
            let nf = n as f64;
            let pf = p as f64;
            let m: Vec<f64> = (0..p)
                .map(|i| xs.iter().map(|x| x[i]).sum::<f64>() / nf)
                .collect();
            let xc: Vec<Vec<f64>> = xs
                .iter()
                .map(|x| x.iter().zip(&m).map(|(a, b)| a - b).collect())
                .collect();
            let mut emp = vec![0.0; p * p];
            for x in &xc {
                for i in 0..p {
                    for j in 0..p {
                        emp[i * p + j] += x[i] * x[j] / nf;
                    }
                }
            }
            let emp_tr: f64 = (0..p).map(|i| emp[i * p + i]).sum();
            let mu = emp_tr / pf;
            let delta_: f64 = emp.iter().map(|v| v * v).sum();
            let beta_: f64 = xc
                .iter()
                .map(|x| {
                    let q: f64 = x.iter().map(|v| v * v).sum();
                    q * q
                })
                .sum();
            let beta = 1.0 / (pf * nf) * (beta_ / nf - delta_);
            let delta = (delta_ - 2.0 * mu * emp_tr + pf * mu * mu) / pf;
            let beta = beta.min(delta);
            let sk_lw = if beta == 0.0 { 0.0 } else { beta / delta };
            let alpha = delta_ / (pf * pf);
            let num = alpha + mu * mu;
            let den = (nf + 1.0) * (alpha - mu * mu / pf);
            let sk_oas = if den == 0.0 {
                1.0
            } else {
                (num / den).min(1.0)
            };
            assert!((lw - sk_lw).abs() < 1e-13, "lw {lw} {sk_lw} p={p} n={n}");
            assert!((oa - sk_oas).abs() < 1e-13, "oas {oa} {sk_oas} p={p} n={n}");
            // And the papers'.
            let tp = two_pass(&xs, 1.0, &w);
            assert!((lw - paper_lw(&tp, p, nf)).abs() < 1e-13);
            assert!((oa - paper_oas(&tp.s, p, nf)).abs() < 1e-13);
            assert!(lw > 0.0 && lw < 1.0 && oa > 0.0 && oa <= 1.0);
        }
    }
}

/// Under decay and unequal weights the intensities are the papers'
/// definitions at Kish's size, written two-pass above. Bound 1e-13
/// absolute, a hundred times the worst measured here (4.2e-16).
#[test]
fn weighted_rows_give_the_papers_intensities_at_kishs_size() {
    for seed in 9..=11 {
        for p in [2, 5, 20] {
            let xs = rows(seed, 250, p, -7.0);
            let mut g = crate::SplitMix64::new(seed);
            let w: Vec<f64> = (0..xs.len()).map(|_| 0.1 + 3.0 * g.uniform()).collect();
            let lam = 0.5f64.powf(1.0 / 25.0);
            let (cov, fourth, kish) = learn(&xs, lam, &w);
            let tp = two_pass(&xs, lam, &w);
            let c = cov.comoments().packed();
            let (tr, frob2) = trace_and_frobenius(c, p);
            let lw = ledoit_wolf(p, tr, frob2, fourth.f(), kish);
            let oa = oas(p, tr, frob2, kish);
            assert!((lw - paper_lw(&tp, p, tp.kish)).abs() < 1e-13, "{lw}");
            assert!((oa - paper_oas(&tp.s, p, tp.kish)).abs() < 1e-13, "{oa}");
        }
    }
}

/// Each side of every threshold of the two intensities.
#[test]
fn every_threshold_is_held_on_both_sides() {
    // p = 1: 0 whatever the moments.
    assert_eq!(ledoit_wolf(1, 2.0, 4.0, 100.0, Some(3.0)), 0.0);
    assert_eq!(oas(1, 2.0, 4.0, Some(3.0)), 0.0);
    // n unknown: NaN, from p = 2.
    assert!(ledoit_wolf(2, 2.0, 3.0, 9.0, None).is_nan());
    assert!(oas(2, 2.0, 3.0, None).is_nan());
    assert!(oas(2, 2.0, 3.0, Some(0.0)).is_nan());
    // S = μI (p = 2, μ = 1: tr 2, ‖S‖² 2): δ = 0, LW 0 and OAS 1.
    assert_eq!(ledoit_wolf(2, 2.0, 2.0, 9.0, Some(10.0)), 0.0);
    assert_eq!(oas(2, 2.0, 2.0, Some(10.0)), 1.0);
    // δ just above 0: p = 2, tr 2, ‖S‖² 2 + 2e-12, δ = 1e-12.
    let frob2 = 2.0 + 2e-12;
    let delta = (frob2 - 2.0) / 2.0;
    // β = (F − ‖S‖²) / (p n) above δ: the clip, s = 1 exactly.
    let f_above = frob2 + 2.0 * 10.0 * (delta * 1.5);
    assert_eq!(ledoit_wolf(2, 2.0, frob2, f_above, Some(10.0)), 1.0);
    // Below δ: the ratio, below 1.
    let f_below = frob2 + 2.0 * 10.0 * (delta * 0.5);
    let s = ledoit_wolf(2, 2.0, frob2, f_below, Some(10.0));
    assert!(s > 0.4 && s < 0.6, "{s}");
    // β at 0 (F = ‖S‖²) and below it by rounding: 0, never negative.
    assert_eq!(ledoit_wolf(2, 2.0, 3.0, 3.0, Some(10.0)), 0.0);
    assert_eq!(ledoit_wolf(2, 2.0, 3.0, 3.0 - 1e-15, Some(10.0)), 0.0);
    // Just above 0: a small positive intensity.
    let s = ledoit_wolf(2, 2.0, 3.0, 3.0 + 1e-9, Some(10.0));
    assert!(s > 0.0 && s < 1e-8, "{s}");
    // OAS's clip at 1: a ratio above 1 (n small, S near μI) and below it.
    // tr 2, ‖S‖² 2.02: α = 0.505, μ² = 1, num 1.505, den (n+1)·0.005.
    assert_eq!(oas(2, 2.0, 2.02, Some(10.0)), 1.0);
    let s = oas(2, 2.0, 2.02, Some(400.0));
    assert!(
        (s - 1.505 / (401.0 * 0.005)).abs() < 1e-12 && s < 1.0,
        "{s}"
    );
}

#[test]
fn the_shrunk_matrix_is_the_mixture() {
    // [[4, 1, 0], [1, 2, -1], [0, -1, 3]]: μ = 3.
    let c = [4.0, 1.0, 2.0, 0.0, -1.0, 3.0];
    let mut out = [0.0; 6];
    shrink_into(&c, 3, 0.25, None, &mut out);
    let want = [
        0.75 * 4.0 + 0.75,
        0.75,
        0.75 * 2.0 + 0.75,
        0.0,
        -0.75,
        0.75 * 3.0 + 0.75,
    ];
    assert_eq!(out, want);
    let t = [1.0, 0.5, 1.0, 0.0, 0.0, 2.0];
    shrink_into(&c, 3, 0.5, Some(&t), &mut out);
    assert_eq!(out, [2.5, 0.75, 1.5, 0.0, -0.5, 2.5]);
    // s = 0 and s = 1 are the moments and the target.
    shrink_into(&c, 3, 0.0, None, &mut out);
    assert_eq!(out, c);
    shrink_into(&c, 3, 1.0, Some(&t), &mut out);
    assert_eq!(out, t);
}

#[test]
fn a_fixed_intensity_and_target_are_checked() {
    let ok = |s: Shrinkage, k| s.validate("ew_cov", k);
    for good in [0.0, 0.3, 1.0] {
        assert!(
            ok(
                Shrinkage::Fixed {
                    intensity: good,
                    target: None
                },
                2
            )
            .is_ok()
        );
    }
    for bad in [-1e-12, 1.0 + 1e-12, f64::NAN, f64::INFINITY] {
        let e = ok(
            Shrinkage::Fixed {
                intensity: bad,
                target: None,
            },
            2,
        )
        .unwrap_err();
        assert!(e.contains("at least 0 and at most 1"), "{e}");
    }
    let fixed = |t: Vec<f64>| Shrinkage::Fixed {
        intensity: 0.5,
        target: Some(t),
    };
    // Wrong length.
    let e = ok(fixed(vec![1.0, 0.0]), 2).unwrap_err();
    assert!(e.contains("3 numbers, but it holds 2"), "{e}");
    // Not finite.
    let e = ok(fixed(vec![1.0, f64::NAN, 1.0]), 2).unwrap_err();
    assert!(e.contains("finite"), "{e}");
    // [[1, 2], [2, 1]]: eigenvalues 3 and -1.
    let e = ok(fixed(vec![1.0, 2.0, 1.0]), 2).unwrap_err();
    assert!(e.contains("positive semidefinite"), "{e}");
    // [[1, 1], [1, 1]]: singular, semidefinite, taken; and just past it,
    // [[1, 1 + 1e-9], ...], refused.
    assert!(ok(fixed(vec![1.0, 1.0, 1.0]), 2).is_ok());
    assert!(ok(fixed(vec![1.0, 1.0 + 1e-9, 1.0]), 2).is_err());
    // The estimators take no setting to check.
    assert!(ok(Shrinkage::LedoitWolf, 1).is_ok());
    assert!(ok(Shrinkage::Oas, 3).is_ok());
}

#[test]
fn the_intensity_reads_what_each_estimator_needs() {
    let c = [4.0, 1.0, 2.0];
    let fixed = Shrinkage::Fixed {
        intensity: 0.3,
        target: None,
    };
    assert_eq!(intensity(&fixed, &c, 2, None, None), 0.3);
    assert!(intensity(&Shrinkage::LedoitWolf, &c, 2, Some(5.0), None).is_nan());
    let (tr, f2) = trace_and_frobenius(&c, 2);
    assert_eq!((tr, f2), (6.0, 16.0 + 4.0 + 2.0));
    assert_eq!(
        intensity(&Shrinkage::Oas, &c, 2, Some(5.0), None),
        oas(2, tr, f2, Some(5.0))
    );
    assert_eq!(
        intensity(&Shrinkage::LedoitWolf, &c, 2, Some(5.0), Some(30.0)),
        ledoit_wolf(2, tr, f2, 30.0, Some(5.0))
    );
    assert!(Shrinkage::LedoitWolf.needs_fourth());
    assert!(!Shrinkage::Oas.needs_fourth());
    let st = ShrinkState::new(&Shrinkage::LedoitWolf, 3);
    assert!(st.fits(&Shrinkage::LedoitWolf, 3));
    assert!(!st.fits(&Shrinkage::LedoitWolf, 2));
    assert!(!st.fits(&Shrinkage::Oas, 3));
    assert!(ShrinkState::new(&Shrinkage::Oas, 3).fits(&Shrinkage::Oas, 3));
}
