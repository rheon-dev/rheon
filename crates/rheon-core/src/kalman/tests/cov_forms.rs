//! The covariance's update at a huge prior (task 281 (3)): the filter
//! against the posterior faer's LU solves (`crate::oracle`), which shares
//! no arithmetic with the recursion. A file of its own because `kalman.rs`
//! stands near the source-size cap (`tests/test_repo_hygiene.py`).

use super::*;

/// Three features of unlike scales and means (the intercept's column
/// nearly a multiple of the third's, as `tests/test_kalman.py`'s
/// relations frame has it), uneven weights with a zero on every 37th row,
/// and a slope that drifts.
fn rows(n: usize, seed: u64) -> Vec<([f64; 3], f64, f64)> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let x = [
                0.5 + lcg(&mut s),
                -1.0 + 2.0 * lcg(&mut s),
                3.0 + 0.5 * lcg(&mut s),
            ];
            let slope = 2.0 + (i as f64 / n as f64);
            let y = 1.0 + slope * x[0] - x[1] + 0.5 * x[2] + 0.5 * lcg(&mut s);
            let w = if i % 37 == 36 {
                0.0
            } else {
                1.0 + 0.5 * lcg(&mut s)
            };
            (x, y, w)
        })
        .collect()
}

/// The bank's form: `P` in units of the noise, the intercept diffuse,
/// unstandardized so the oracle's coordinates are the filter's; the drift
/// at the half-life `h` (`inf`: none), the noise's decay at the same.
fn units_cfg(p0: f64, h: f64) -> KalmanCfg {
    KalmanCfg {
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: 3,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(h),
        half_life: vec![h],
        q: None,
        obs_var: None,
        p0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    }
}

/// The posterior mean of the last row's coefficients, and their
/// covariance, under the filter's model written as one batch: a state
/// `b_s` per observed row, `y_s ~ N(z_s' b_s, 1 / w_s)`, the slopes'
/// prior `N(0, p0)` on the first and the intercept's diffuse, and `b_s −
/// b_{s−1} ~ N(0, q D_s² I)` with `q = (ln 2 / h)²` and `D_s` the clock
/// between the two rows (the module doc, "The drift"). Its normal
/// equations, block tridiagonal, solved whole by faer's LU; with no drift
/// one state.
fn posterior(data: &[([f64; 3], f64, f64)], p0: f64, h: f64) -> (Vec<f64>, Vec<f64>, f64) {
    const K: usize = 4;
    let obs: Vec<(usize, [f64; K], f64, f64)> = data
        .iter()
        .enumerate()
        .filter(|(_, (_, _, w))| *w > 0.0)
        .map(|(t, (x, y, w))| (t, [1.0, x[0], x[1], x[2]], *y, *w))
        .collect();
    let q = if h.is_infinite() {
        0.0
    } else {
        (std::f64::consts::LN_2 / h).powi(2)
    };
    let ns = if q == 0.0 { 1 } else { obs.len() };
    let n = ns * K;
    let mut a = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for (s, (t, z, y, w)) in obs.iter().enumerate() {
        let b = if q == 0.0 { 0 } else { s * K };
        for i in 0..K {
            rhs[b + i] += w * z[i] * y;
            for j in 0..K {
                a[(b + i) * n + b + j] += w * z[i] * z[j];
            }
        }
        if q > 0.0 && s > 0 {
            let d = (t - obs[s - 1].0) as f64;
            let inv = 1.0 / (q * d * d);
            let p = b - K;
            for i in 0..K {
                a[(b + i) * n + b + i] += inv;
                a[(p + i) * n + p + i] += inv;
                a[(b + i) * n + p + i] -= inv;
                a[(p + i) * n + b + i] -= inv;
            }
        }
    }
    for i in 1..K {
        a[i * n + i] += 1.0 / p0;
    }
    let x = crate::oracle::solve(&a, &rhs);
    let last = n - K;
    // The oracle's own error in the last coefficients, relative as `apart`
    // reads a coefficient.
    let own = crate::oracle::solve_error(&a, &rhs, &x)[last..]
        .iter()
        .zip(&x[last..])
        .map(|(e, v)| e / v.abs().max(1.0))
        .fold(0.0f64, f64::max);
    // The last block of the inverse: the covariance of the last state.
    let mut unit = vec![0.0; n * K];
    for j in 0..K {
        unit[(last + j) * K + j] = 1.0;
    }
    let cols = crate::oracle::solve_columns(&a, &unit, K);
    let cov = cols[last * K..].to_vec();
    (x[last..].to_vec(), cov, own)
}

/// The filter's last coefficients and covariance on `data`.
fn filtered(
    data: &[([f64; 3], f64, f64)],
    p0: f64,
    h: f64,
    form: CovarianceForm,
) -> (Vec<f64>, Vec<f64>) {
    let mut m = Kalman::new(units_cfg(p0, h))
        .unwrap()
        .with_covariance_form(form)
        .unwrap();
    let mut t_prev = None;
    for (t, (x, y, w)) in data.iter().enumerate() {
        let d = t_prev.map_or(0.0, |p| (t - p) as f64);
        t_prev = Some(t);
        m.step(x, &[Some(*y)], d, *w);
    }
    let cov = m.covariance_now(0).into_owned();
    (m.coefficients()[0].clone(), cov)
}

/// The largest relative distance of `got` from `want`, each against the
/// larger of its size and 1 (a coefficient), or against the largest entry
/// of `want` (a covariance, `whole`).
fn apart(got: &[f64], want: &[f64], whole: bool) -> f64 {
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let d = if whole { scale } else { w.abs().max(1.0) };
            (g - w).abs() / d
        })
        .fold(0.0f64, f64::max)
}

/// One line of the table: each form's distance from the posterior in a
/// coefficient and in the covariance, at the drift `h` and prior `p0`.
fn table_line(data: &[([f64; 3], f64, f64)], p0: f64, h: f64) -> (String, [(f64, f64); 2], f64) {
    let (want_b, want_p, own) = posterior(data, p0, h);
    let mut line = format!("h={h:>6} p0={p0:>6.0e} (oracle {own:.1e}):");
    let mut out = [(0.0, 0.0); 2];
    for (o, form) in out
        .iter_mut()
        .zip([CovarianceForm::Covariance, CovarianceForm::SquareRoot])
    {
        let (got_b, got_p) = filtered(data, p0, h, form);
        *o = (apart(&got_b, &want_b, false), apart(&got_p, &want_p, true));
        line += &format!(" {form:?} coef {:.1e} cov {:.1e} |", o.0, o.1);
    }
    (line, out, own)
}

/// The loss, and the factor's cure, with no drift: the filter is then
/// Bayesian regression, whose posterior is one `k x k` solve. Measured
/// 2026-11-20, 300 rows: the covariance form 3.8e-11 off in a coefficient
/// at `p0 = 1e6`, 6.7e-5 at 1e12 and 1.3e-2 at 1e15 (its covariance
/// 1.6e-5 and 1.8e-2 of its largest entry); a Joseph form, measured and
/// dropped, 8.8e-10, 4.7e-4 and 0.11; the factor 1.1e-13 or less at every prior, the
/// oracle's own error 5e-14 to 1e-13. The factor is held to the oracle's
/// error plus the recursion's rounding, `κ γ_n` with `κ` the posterior's
/// condition and `n` the rows, 3.0e-11 here: 270 times what it made.
#[test]
fn a_huge_prior_costs_the_covariance_form_digits_and_not_the_factor() {
    let data = rows(300, 281);
    let mut table = String::new();
    for p0 in [1.0, 1e6, 1e12, 1e15] {
        let (line, [cov, factor], own) = table_line(&data, p0, f64::INFINITY);
        table += &line;
        table += "\n";
        let (_, want_p, _) = posterior(&data, p0, f64::INFINITY);
        let kappa = crate::oracle::condition(&want_p);
        let bound = own + kappa * crate::oracle::gamma(data.len());
        assert!(
            factor.0 <= bound && factor.1 <= bound,
            "the factor at p0 = {p0:e}: {factor:?}, bound {bound:.1e}"
        );
        if p0 >= 1e12 {
            assert!(cov.0 > 1e-6, "{table}");
        }
    }
    eprintln!("{table}");
}

/// The same under drift, where the posterior is the batch of every row's
/// state, solved whole: short (`h = 20`) and long (`h = 2000`). Measured
/// 2026-11-20, 150 rows: at `h = 20` the covariance form 5.1e-5 off at
/// `p0 = 1e12` and 4.4e-2 at 1e15, a Joseph form 7.5e-4 and 0.11, the
/// factor 3.2e-12 and 1.8e-12 against the oracle's own 1.9e-11; at `h =
/// 2000` the covariance form 1.2e-4 and 2.3e-2, the factor 8.5e-10, inside
/// the oracle's own error of 1.1e-7 (the batch is ill-conditioned there: a
/// drift of `(ln 2 / 2000)²` a row ties the states 8e6 times harder than a
/// row's data). Fewer rows than the test above, as the oracle's error
/// bound is O(n³).
#[test]
#[ignore = "extended: two batch solves of 600 unknowns and their error bounds"]
fn under_drift_a_huge_prior_costs_the_covariance_form_digits_and_not_the_factor() {
    let data = rows(150, 281);
    let mut table = String::new();
    for h in [20.0, 2000.0] {
        for p0 in [1.0, 1e12, 1e15] {
            let (line, [cov, factor], own) = table_line(&data, p0, h);
            table += &line;
            table += "\n";
            let bound = own + 1e-11;
            assert!(factor.0 <= bound, "{table}");
            if p0 >= 1e12 {
                assert!(cov.0 > 1e-7, "{table}");
            }
        }
    }
    eprintln!("{table}");
}

/// Every form is the same filter where the prior is modest, through every
/// branch that writes `P`: standardized with the moments drifting far
/// enough to re-map, the transition's coupling (slopes reverting at their
/// own half-lives), the process noise, a diffuse intercept, `share_p`,
/// targets missing on some rows, weights of 0, steps of 1 and 3; and
/// unstandardized, with `obs_var`, with the noise learned outside its
/// units, without an intercept. The predictions, the last coefficients and
/// `pred_var` agree to 1e-9 of their size, and the factor stays lower
/// triangular to the bit.
#[test]
fn every_form_is_the_same_filter_at_a_modest_prior() {
    let base = KalmanCfg {
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: 3,
        n_targets: 2,
        fit_intercept: true,
        decay: Decay::Halflife(80.0),
        half_life: vec![50.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY, 30.0, f64::INFINITY, 60.0],
        standardize: true,
    };
    let cases = [
        ("standardized", base.clone()),
        (
            "share_p",
            KalmanCfg {
                share_p: true,
                ..base.clone()
            },
        ),
        (
            "one revert",
            KalmanCfg {
                revert_half_life: vec![40.0],
                ..base.clone()
            },
        ),
        (
            "no intercept",
            KalmanCfg {
                fit_intercept: false,
                revert_half_life: vec![f64::INFINITY],
                ..base.clone()
            },
        ),
        (
            "absolute",
            KalmanCfg {
                noise_units: false,
                ..base.clone()
            },
        ),
        (
            "obs_var",
            KalmanCfg {
                obs_var: Some(0.5),
                ..base.clone()
            },
        ),
        (
            "raw",
            KalmanCfg {
                standardize: false,
                revert_half_life: vec![f64::INFINITY],
                ..base.clone()
            },
        ),
        (
            "raw obs_var",
            KalmanCfg {
                standardize: false,
                obs_var: Some(2.0),
                fit_intercept: false,
                revert_half_life: vec![f64::INFINITY],
                ..base.clone()
            },
        ),
    ];
    for (name, cfg) in cases {
        let k = cfg.k_total();
        let run = |form: CovarianceForm| {
            let mut m = Kalman::new(cfg.clone())
                .unwrap()
                .with_covariance_form(form)
                .unwrap();
            let mut s = 2814u64;
            let mut preds = Vec::new();
            for i in 0..1500 {
                // The second feature's scale grows 16-fold: re-maps.
                let grow = 2f64.powf(4.0 * i as f64 / 1500.0);
                let x = [
                    lcg(&mut s),
                    2.0 + grow * lcg(&mut s),
                    0.5 * lcg(&mut s) - 1.0,
                ];
                let y0 = 1.0 + x[0] - 0.3 * x[1] / grow + 2.0 * x[2] + 0.2 * lcg(&mut s);
                let y1 = (i % 3 != 1).then(|| -x[0] + 0.1 * lcg(&mut s));
                let w = if i % 29 == 7 {
                    0.0
                } else {
                    1.0 + 0.5 * lcg(&mut s)
                };
                let d = if i == 0 {
                    0.0
                } else if i % 11 == 0 {
                    3.0
                } else {
                    1.0
                };
                preds.extend(m.step(&x, &[Some(y0), y1], d, w).pred);
            }
            let tail = m.pred_var(&[0.3, 2.5, -1.0]);
            (preds, m.coefficients().concat(), tail, m)
        };
        let (want, want_c, want_v, _) = run(CovarianceForm::Covariance);
        assert!(
            want.iter().filter(|v| v.is_finite()).count() > 2500,
            "{name}"
        );
        {
            let form = CovarianceForm::SquareRoot;
            let (got, got_c, got_v, m) = run(form);
            for (what, g, w) in [
                ("pred", &got, &want),
                ("coef", &got_c, &want_c),
                ("pred_var", &got_v, &want_v),
            ] {
                for (a, b) in g.iter().zip(w.iter()) {
                    assert_eq!(a.is_finite(), b.is_finite(), "{name} {form:?} {what}");
                    if b.is_finite() {
                        assert!(
                            (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                            "{name} {form:?} {what}: {a} {b}"
                        );
                    }
                }
            }
            if form == CovarianceForm::SquareRoot {
                for l in &m.p {
                    assert!(
                        (0..k * k).all(|ij| ij / k >= ij % k || l[ij] == 0.0),
                        "{name}"
                    );
                }
            }
        }
    }
}

/// A filter in the factor form saves its form with it, and a loaded one
/// goes on to the bit; the default form writes no field for it, so a
/// default state's bytes are those written before the field existed.
#[test]
fn a_saved_factor_continues_to_the_bit_and_the_default_writes_no_form() {
    let cfg = units_cfg(1e12, 50.0);
    let data = rows(200, 7);
    let mut a = Kalman::new(cfg.clone())
        .unwrap()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap();
    for (x, y, w) in &data[..100] {
        a.step(x, &[Some(*y)], 1.0, *w);
    }
    let bytes = rmp_serde::to_vec(&a.state()).unwrap();
    let mut b = Kalman::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert_eq!(b.covariance_form(), CovarianceForm::SquareRoot);
    for (x, y, w) in &data[100..] {
        let (pa, pb) = (
            a.step(x, &[Some(*y)], 1.0, *w).pred,
            b.step(x, &[Some(*y)], 1.0, *w).pred,
        );
        assert_eq!(pa[0].to_bits(), pb[0].to_bits());
    }
    let plain = Kalman::new(cfg).unwrap();
    let with = plain
        .clone()
        .with_covariance_form(CovarianceForm::Covariance)
        .unwrap();
    let factor = plain
        .clone()
        .with_covariance_form(CovarianceForm::SquareRoot)
        .unwrap();
    let size = |m: &Kalman| rmp_serde::to_vec(&m.state()).unwrap().len();
    assert_eq!(
        rmp_serde::to_vec(&plain.state()).unwrap(),
        rmp_serde::to_vec(&with.state()).unwrap()
    );
    assert!(size(&factor) > size(&plain));
    // A filter that has taken a row is not converted.
    let mut moved = Kalman::new(units_cfg(1.0, 50.0)).unwrap();
    for (x, y, w) in &data[..3] {
        moved.step(x, &[Some(*y)], 1.0, *w);
    }
    assert!(
        moved
            .with_covariance_form(CovarianceForm::SquareRoot)
            .is_err()
    );
}

/// `A B` for row-major `k x k` matrices.
fn mul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = (0..k).map(|l| a[i * k + l] * b[l * k + j]).sum();
        }
    }
    out
}

fn transpose(a: &[f64], k: usize) -> Vec<f64> {
    (0..k * k).map(|ij| a[(ij % k) * k + ij / k]).collect()
}

/// A random factor `L = Cᵀ`, lower triangular, with a positive diagonal.
fn lower(k: usize, s: &mut u64) -> Vec<f64> {
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        l[i * k + i] = 1.5 + lcg(s);
        for j in 0..i {
            l[i * k + j] = lcg(s);
        }
    }
    l
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= tol * scale,
            "{what}: {got:?} against {want:?}"
        );
    }
}

/// Each kernel is its definition, by matrix products written out:
/// Carlson's factor of `P
/// − pz pzᵀ / s`, a rank-one update's of `P + x xᵀ` (on `x` with zeros
/// above and below its entries), the transition's `Phi_a P Phi_aᵀ` and the
/// re-map's `A P Aᵀ`; every factor `L = Cᵀ` stays lower triangular, to the bit.
#[test]
fn each_form_kernel_is_its_definition() {
    let mut s = 2813u64;
    for k in [1usize, 2, 3, 6, 9] {
        for _ in 0..5 {
            let c = lower(k, &mut s);
            let p = mul(&transpose(&c, k), &c, k);
            let z: Vec<f64> = (0..k).map(|_| 2.0 * lcg(&mut s)).collect();
            let r = 0.3 + lcg(&mut s).abs();
            let pz: Vec<f64> = (0..k)
                .map(|i| (0..k).map(|j| p[i * k + j] * z[j]).sum())
                .collect();
            let sv = z.iter().zip(&pz).map(|(a, b)| a * b).sum::<f64>() + r;
            let post: Vec<f64> = (0..k * k)
                .map(|ij| p[ij] - pz[ij / k] * pz[ij % k] / sv)
                .collect();
            let upper_zero = |c: &[f64]| (0..k * k).all(|ij| ij / k >= ij % k || c[ij] == 0.0);

            // Carlson.
            let mut f = vec![0.0; k];
            forms::factor_f(&c, &z, &mut f);
            let mut cz = vec![0.0; k];
            forms::factor_cf(&c, &f, &mut cz);
            assert_close(&cz, &pz, 1e-14, "C Cᵀ z");
            let mut cc = c.clone();
            forms::carlson(&mut cc, &f, r, &mut vec![0.0; k]);
            assert!(upper_zero(&cc), "carlson's factor {cc:?}");
            assert_close(&forms::factor_product(&cc, k), &post, 1e-13, "carlson");

            // A rank-one update on a slot and the intercept, and on one slot.
            for (i, both) in [(k - 1, true), (k / 2, false), (0, false)] {
                let mut x = vec![0.0; k];
                x[i] = 0.7 + lcg(&mut s).abs();
                if both {
                    x[0] += lcg(&mut s);
                }
                let want: Vec<f64> = (0..k * k).map(|ij| p[ij] + x[ij / k] * x[ij % k]).collect();
                let mut cc = c.clone();
                forms::factor_add(&mut cc, &mut x.clone());
                assert!(upper_zero(&cc), "the update's factor {cc:?}");
                assert_close(&forms::factor_product(&cc, k), &want, 1e-13, "rank one");
            }

            // The transition, with and without its coupling.
            let phi: Vec<f64> = (0..k).map(|_| 0.5 + 0.4 * lcg(&mut s)).collect();
            let gc: Vec<f64> = (0..k)
                .map(|i| if i == 0 { 0.0 } else { lcg(&mut s) })
                .collect();
            for coupled in [false, true] {
                let mut t = vec![0.0; k * k];
                for i in 0..k {
                    t[i * k + i] = phi[i];
                }
                if coupled {
                    t[1..k].copy_from_slice(&gc[1..k]);
                }
                let want = mul(&mul(&t, &p, k), &transpose(&t, k), k);
                let mut cc = c.clone();
                forms::factor_transit(&mut cc, &phi, coupled.then_some(&gc[..]));
                assert!(upper_zero(&cc));
                assert_close(&forms::factor_product(&cc, k), &want, 1e-13, "transition");
            }

            // The re-map, with an intercept and without.
            for off in [0usize, 1] {
                let av: Vec<f64> = (0..k)
                    .map(|i| {
                        if i < off {
                            1.0
                        } else {
                            0.6 + lcg(&mut s).abs()
                        }
                    })
                    .collect();
                let cv: Vec<f64> = (0..k)
                    .map(|i| {
                        if i < off || off == 0 {
                            0.0
                        } else {
                            lcg(&mut s)
                        }
                    })
                    .collect();
                let mut t = vec![0.0; k * k];
                for i in 0..k {
                    t[i * k + i] = av[i];
                }
                for j in 1..k {
                    t[j] += cv[j];
                }
                let want = mul(&mul(&t, &p, k), &transpose(&t, k), k);
                let mut cc = c.clone();
                let mut u = vec![0.0; k];
                assert!(forms::factor_remap(&mut cc, &mut u, &av, &cv, off, false));
                assert_eq!(cc, c, "a check moves nothing");
                assert!(forms::factor_remap(&mut cc, &mut u, &av, &cv, off, true));
                assert!(upper_zero(&cc));
                assert_close(&forms::factor_product(&cc, k), &want, 1e-13, "re-map");
            }
        }
    }
}

/// Unstandardized in units of the noise, the factor's process noise
/// (`factor_noise`) is the covariance form's: every slot takes it but a
/// diffuse intercept, which takes none before its first row. Rows that
/// observe nothing across gaps of 2 leave the noise to the first observed
/// row, which charges the slopes with it and takes the diffuse step whole
/// in both forms, which agree on `P` after it. A factor that charged the intercept, or left a
/// slope out, would size the intercept or lose that slope's noise (the
/// changed-lines mutants pass on `i < off` there, 2026-11-20).
#[test]
fn the_factors_process_noise_leaves_a_diffuse_intercept_diffuse() {
    let cfg = KalmanCfg {
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        half_life: vec![20.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    };
    let k = 3;
    let mut forms = [CovarianceForm::Covariance, CovarianceForm::SquareRoot].map(|form| {
        Kalman::new(cfg.clone())
            .unwrap()
            .with_covariance_form(form)
            .unwrap()
    });
    for m in &mut forms {
        for _ in 0..3 {
            m.step(&[0.5, 0.5], &[Some(9.0)], 2.0, 0.0);
        }
    }
    let [cov, factor] = &forms;
    assert!(cov.open::<false>(0) && factor.open::<true>(0));
    let (x, y, w) = ([0.7, -1.3], 1e6 + 4.25, 2.0);
    for m in &mut forms {
        m.step(&x, &[Some(y)], 2.0, w);
        assert_eq!(m.beta[0][0], y, "the level, whole");
    }
    let [cov, factor] = &forms;
    for i in 1..k {
        let v = cov.p[0][i * k + i];
        assert!(v > 1.0, "slot {i} took the gaps' noise: {v}");
    }
    assert_close(&factor.cov(0), &cov.p[0], 1e-12, "after the first row");
}

/// Each form is Bayesian regression's posterior at any scale of the target
/// (review fable2, C5). With `obs_var` the noise `R` is given and the prior
/// is `p0 R I` on every slot, so the posterior mean solves `(Σ w z zᵀ + I /
/// p0) b = Σ w z y`, whatever `R`: the coefficients over the target's scale
/// are the same numbers at a scale of 1, `2²⁶⁶` (about 1.2e80) and `2⁻²⁸²`
/// (about 1.6e-85), all inside the input bound, and faer's LU solves them
/// once. Carlson's step formed `sqrt(α_{j−1} α_j)`, a product of order `R²`,
/// which overflowed at a noise near 1e154 -- its cross term 0, the factor
/// finite and wrong (a slope of +0.047 where the posterior has −0.96) --
/// and underflowed near 1e-154, the factor NaN and every prediction
/// withheld. Where the product is not a normal double the square roots are
/// taken apart now, each of a number of `R`'s order.
#[test]
fn every_form_is_the_posterior_at_any_scale_of_the_target() {
    let data = rows(300, 5);
    const K: usize = 4;
    let p0 = 1.0;
    let (mut a, mut rhs) = (vec![0.0; K * K], vec![0.0; K]);
    for (x, y, w) in &data {
        let z = [1.0, x[0], x[1], x[2]];
        for i in 0..K {
            rhs[i] += w * z[i] * y;
            for j in 0..K {
                a[i * K + j] += w * z[i] * z[j];
            }
        }
    }
    for i in 0..K {
        a[i * K + i] += 1.0 / p0;
    }
    let want = crate::oracle::solve(&a, &rhs);
    for scale in [1.0, 2f64.powi(266), 2f64.powi(-282), 1e80, 1e-85] {
        for form in [CovarianceForm::Covariance, CovarianceForm::SquareRoot] {
            let mut cfg = units_cfg(p0, f64::INFINITY);
            cfg.obs_var = Some(0.25 * scale * scale);
            let mut m = Kalman::new(cfg)
                .unwrap()
                .with_covariance_form(form)
                .unwrap();
            let mut withheld = 0;
            for (i, (x, y, w)) in data.iter().enumerate() {
                let st = m.step(x, &[Some(scale * y)], if i == 0 { 0.0 } else { 1.0 }, *w);
                withheld += usize::from(i > 0 && st.pred[0].is_nan());
            }
            let got: Vec<f64> = m.coefficients()[0].iter().map(|b| b / scale).collect();
            let off = apart(&got, &want, false);
            assert!(
                off < 1e-11 && withheld == 0,
                "{form:?} at a scale of {scale:e}: {got:?} against {want:?}, {off:.1e} apart, \
                 {withheld} predictions withheld"
            );
        }
    }
}

/// `predict` withholds where the step withholds, across a reversion gap
/// long enough to take the covariance to exactly 0 (review fable2, C6).
/// The step runs the transition, `P <- Phi P Phi` (`C <- Phi C` in the
/// square root), before it reads whether the covariance is sized; past
/// about 540 reversion half-lives (1,075 in the square root) every entry
/// underflows, the step finds it unsized and withholds, and `predict`,
/// which read the covariance as it stood before the transition, reported
/// `z·b` at a `b` shrunk to nothing. A gap short of the underflow
/// predicts in both. One half-life for every slot, and one each, which
/// couples the intercept to the slopes.
#[test]
fn predict_withholds_where_the_step_does_across_a_long_reversion_gap() {
    for form in [CovarianceForm::Covariance, CovarianceForm::SquareRoot] {
        for revert in [vec![1.0], vec![1.0, 0.8, 1.3]] {
            for gap in [20.0, 600.0, 1100.0, 2000.0] {
                let mut cfg = units_cfg(1.0, 100.0);
                cfg.n_features = 2;
                cfg.standardize = true;
                cfg.revert_half_life = revert.clone();
                let mut m = Kalman::new(cfg)
                    .unwrap()
                    .with_covariance_form(form)
                    .unwrap();
                let mut s = 3u64;
                for _ in 0..60 {
                    let x = [lcg(&mut s), 0.5 + lcg(&mut s)];
                    let y = 2.0 * x[0] - x[1] + 0.1 * lcg(&mut s) + 5.0;
                    m.step(&x, &[Some(y)], 1.0, 1.0);
                }
                for x in [[0.3, 0.4], [-0.6, 1.2]] {
                    let mut stepped = m.clone();
                    let p = m.predict(&x, gap);
                    let st = stepped.step(&x, &[Some(1.0)], gap, 1.0);
                    assert_eq!(
                        p.pred[0].to_bits(),
                        st.pred[0].to_bits(),
                        "{form:?}, reverting at {revert:?}, gap {gap}: predict {:?}, step {:?}",
                        p.pred,
                        st.pred
                    );
                    if gap <= 20.0 {
                        assert!(st.pred[0].is_finite(), "a short gap predicts");
                    }
                }
            }
        }
    }
}
