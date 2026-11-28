use super::*;
use crate::{GarchVol, oracle};

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn normal(s: &mut u64) -> f64 {
    let (u, v) = (lcg(s), lcg(s));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A realistic (A)DCC stream of standardized residuals, `n` rows after
/// 1,000 discarded: `k` series, every target correlation `rho`, the truth
/// `(a, b, g)`, innovations `L_t x_t` with `x` standard normal (or a
/// Student-t of `nu` degrees of freedom at unit variance), `L_t` the
/// Cholesky factor of `R_t`. `N` is held at the independent-normal orthant
/// value scaled by the target, which keeps the truth's intercept positive
/// definite at these parameters.
fn adcc_stream(
    k: usize,
    n: usize,
    theta: [f64; 3],
    rho: f64,
    nu: Option<f64>,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut s = seed;
    let kk = k * k;
    let mut qbar = vec![rho; kk];
    for i in 0..k {
        qbar[i * k + i] = 1.0;
    }
    let nbar: Vec<f64> = qbar.iter().map(|v| 0.5 * v).collect();
    let (a, b, g) = (theta[0], theta[1], theta[2]);
    let mut q = qbar.clone();
    let mut out = Vec::with_capacity(n);
    let mut l = vec![0.0; kk];
    for t in 0..n + 1000 {
        let mut r = vec![0.0; kk];
        for i in 0..k {
            for j in 0..k {
                r[i * k + j] = q[i * k + j] / (q[i * k + i] * q[j * k + j]).sqrt();
            }
        }
        assert!(mat::cholesky(&r, k, &mut l));
        let scale = match nu {
            Some(nu) => {
                let chi: f64 = (0..nu as usize).map(|_| normal(&mut s).powi(2)).sum();
                ((nu - 2.0) / chi).sqrt()
            }
            None => 1.0,
        };
        let x: Vec<f64> = (0..k).map(|_| normal(&mut s) * scale).collect();
        let z: Vec<f64> = (0..k)
            .map(|i| (0..=i).map(|j| l[i * k + j] * x[j]).sum())
            .collect();
        let neg: Vec<f64> = z.iter().map(|v| v.min(0.0)).collect();
        for i in 0..k {
            for j in 0..k {
                let ij = i * k + j;
                q[ij] = (1.0 - a - b) * qbar[ij] - g * nbar[ij]
                    + a * z[i] * z[j]
                    + g * neg[i] * neg[j]
                    + b * q[ij];
            }
        }
        if t >= 1000 {
            out.push(z);
        }
    }
    out
}

/// Each series of `z` scaled by its own GARCH(1,1) at `(0.05, 0.08, 0.9)`:
/// returns whose margins have a variance process to identify.
fn garch_returns(z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = z[0].len();
    let mut v = vec![1.0; k];
    let mut e = vec![0.0_f64; k];
    z.iter()
        .map(|zt| {
            for i in 0..k {
                v[i] = 0.05 + 0.08 * e[i] * e[i] + 0.9 * v[i];
                e[i] = v[i].sqrt() * zt[i];
            }
            e.clone()
        })
        .collect()
}

fn cfg(k: usize, asymmetric: bool, params: Option<Vec<f64>>) -> DccCfg {
    DccCfg {
        n_targets: k,
        asymmetric,
        margins: None,
        params,
        decay: Decay::Halflife(f64::INFINITY),
        start_rows: DCC_START_ROWS,
        start_span: None,
        min_weight: 0.0,
        max_state_bytes: None,
        adaptive_start: false,
    }
}

fn ys(z: &[f64]) -> Vec<Option<f64>> {
    z.iter().map(|v| Some(*v)).collect()
}

fn run(m: &mut Dcc, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|z| m.step(&[], &ys(z), 1.0, 1.0).pred)
        .collect()
}

/// The recursion of the module docs written longhand, from the targets of
/// the start's rows, with the running means after it, against the model's
/// reported correlations and log-densities: DCC and ADCC with given
/// parameters on a realistic stream.
#[test]
fn the_reported_correlation_is_the_recursion_written_longhand() {
    for (asym, params) in [(false, vec![0.05, 0.9]), (true, vec![0.03, 0.93, 0.03])] {
        let k = 3;
        let rows = adcc_stream(
            k,
            1200,
            [0.03, 0.93, if asym { 0.03 } else { 0.0 }],
            0.5,
            None,
            7,
        );
        let mut m = Dcc::new(cfg(k, asym, Some(params.clone()))).unwrap();
        let got = run(&mut m, &rows);
        // longhand
        let n0 = DCC_START_ROWS;
        let mean = |f: &dyn Fn(&[f64], usize, usize) -> f64, upto: usize| -> Vec<f64> {
            let mut out = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    out[i * k + j] =
                        rows[..upto].iter().map(|z| f(z, i, j)).sum::<f64>() / upto as f64;
                }
            }
            out
        };
        let zz = |z: &[f64], i: usize, j: usize| z[i] * z[j];
        let nn = |z: &[f64], i: usize, j: usize| z[i].min(0.0) * z[j].min(0.0);
        let (qb0, nb0) = (mean(&zz, n0), mean(&nn, n0));
        let g = if asym { params[2] } else { 0.0 };
        let step = |q: &mut Vec<f64>, z: &[f64], qb: &[f64], nb: &[f64]| {
            let old = q.clone();
            for i in 0..k {
                for j in 0..k {
                    let ij = i * k + j;
                    q[ij] = (1.0 - params[0] - params[1]) * qb[ij] - g * nb[ij]
                        + params[0] * z[i] * z[j]
                        + g * z[i].min(0.0) * z[j].min(0.0)
                        + params[1] * old[ij];
                }
            }
        };
        let mut q = qb0.clone();
        for z in &rows[..n0] {
            step(&mut q, z, &qb0, &nb0);
        }
        for t in n0..rows.len() {
            let r: Vec<f64> = (0..k * k)
                .map(|ij| q[ij] / (q[(ij / k) * k + ij / k] * q[(ij % k) * k + ij % k]).sqrt())
                .collect();
            let mut at = 0;
            for i in 0..k {
                for j in (i + 1)..k {
                    assert!((got[t][at] - r[i * k + j]).abs() < 1e-12, "row {t}");
                    at += 1;
                }
            }
            let dens = oracle::gaussian_log_density(&r, &rows[t]);
            assert!(
                (got[t][at] - dens).abs() < 1e-10,
                "row {t}: {} {dens}",
                got[t][at]
            );
            let (qb, nb) = (mean(&zz, t + 1), mean(&nn, t + 1));
            step(&mut q, &rows[t], &qb, &nb);
        }
        // nothing before the start
        assert!(got[..n0].iter().all(|r| r.iter().all(|v| v.is_nan())));
        // the given parameters, inside the set, stand as given
        assert_eq!(m.coefficients(), params);
    }
}

/// Out of sample: a row's value moves no report at or before it.
#[test]
fn a_rows_value_moves_no_report_up_to_it() {
    let rows = adcc_stream(2, 900, [0.04, 0.92, 0.0], 0.4, None, 3);
    let mut a = Dcc::new(cfg(2, false, None)).unwrap();
    let base = run(&mut a, &rows);
    let mut moved = rows.clone();
    moved[700][0] += 5.0;
    let mut b = Dcc::new(cfg(2, false, None)).unwrap();
    let other = run(&mut b, &moved);
    let bits = |r: &[f64]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for t in 0..700 {
        assert_eq!(bits(&base[t][..1]), bits(&other[t][..1]));
    }
    assert_eq!(base[700][0].to_bits(), other[700][0].to_bits());
    assert_ne!(base[700][1].to_bits(), other[700][1].to_bits());
    assert_ne!(base[701][0].to_bits(), other[701][0].to_bits());
}

/// The start's batch fit is a maximum of the likelihood over the set: the
/// buffer's likelihood computed by faer's Gaussian density (`crate::oracle`,
/// which shares nothing with the model's Cholesky) is no higher at any
/// feasible point a small step away.
#[test]
fn the_start_fit_maximizes_the_likelihood() {
    for (asym, seed) in [(false, 11), (true, 12)] {
        let k = 3;
        let rows = adcc_stream(
            k,
            2000,
            [0.04, 0.92, if asym { 0.04 } else { 0.0 }],
            0.5,
            None,
            seed,
        );
        let z: Vec<f64> = rows.iter().flatten().copied().collect();
        let w = vec![1.0; rows.len()];
        let kk = k * k;
        let mut qbar = vec![0.0; kk];
        let mut nbar = vec![0.0; if asym { kk } else { 0 }];
        for (t, zt) in rows.iter().enumerate() {
            update_targets(&mut qbar, &mut nbar, zt, 1.0 / (t + 1) as f64, k);
        }
        let mut work = Vec::new();
        let dlt = if asym {
            region::delta(&qbar, &nbar, k, &mut work).unwrap()
        } else {
            0.0
        };
        let p = if asym { 3 } else { 2 };
        let r = fit::Rows {
            z: &z,
            w: &w,
            k,
            qbar: &qbar,
            nbar: &nbar,
        };
        let got = crate::scratch::with_borrow_mut(&WORK, |wk| fit::fit(&r, p, dlt, wk)).unwrap();
        assert!(got.sound() && got.identified(), "{got:?}");
        let lik = |th: &[f64]| -> f64 {
            let mut q = qbar.clone();
            let mut s = 0.0;
            for zt in &rows {
                let rr: Vec<f64> = (0..kk)
                    .map(|ij| q[ij] / (q[(ij / k) * k + ij / k] * q[(ij % k) * k + ij % k]).sqrt())
                    .collect();
                s += oracle::gaussian_log_density(&rr, zt);
                advance(th, &qbar, &nbar, zt, k, &mut q, &mut []);
            }
            s
        };
        let best = lik(&got.theta);
        let c = fit::coefs(p, dlt);
        for i in 0..p {
            for h in [1e-4, -1e-4] {
                let mut th = got.theta.clone();
                th[i] += h;
                let s: f64 = c.iter().zip(&th).map(|(a, b)| a * b).sum();
                if th.iter().all(|v| *v >= 0.0) && s <= PERSISTENCE_CAP {
                    assert!(lik(&th) <= best + 1e-9, "{i} {h}: {} > {best}", lik(&th));
                }
            }
        }
        // and near the truth, on 2,000 rows
        assert!(
            (got.theta[0] - 0.04).abs() < 0.03 && (got.theta[1] - 0.92).abs() < 0.06,
            "{got:?}"
        );
    }
}

/// The online estimate on a realistic stream, normal and Student-t(6)
/// innovations: inside the set at every row, `Q` positive definite, and
/// at 20,000 rows near the truth (the Python oracle tests hold it to the
/// batch fit's standard errors).
#[test]
fn the_online_estimate_stays_in_the_set_and_finds_the_truth() {
    for (asym, nu, seed) in [(false, None, 21), (true, None, 22), (true, Some(6.0), 23)] {
        let k = 3;
        let truth = [0.04, 0.92, if asym { 0.04 } else { 0.0 }];
        let rows = adcc_stream(k, 20000, truth, 0.5, nu, seed);
        let mut m = Dcc::new(cfg(k, asym, None)).unwrap();
        for z in &rows {
            m.step(&[], &ys(z), 1.0, 1.0);
            if m.started() {
                assert!(m.persistence() <= PERSISTENCE_CAP + 1e-15);
                assert!(m.coefficients().iter().all(|v| *v >= 0.0));
            }
        }
        let th = m.coefficients();
        assert!((th[0] - truth[0]).abs() < 0.015, "{th:?}");
        assert!((th[1] - truth[1]).abs() < 0.03, "{th:?}");
        if asym {
            assert!((th[2] - truth[2]).abs() < 0.03, "{th:?}");
        }
        assert_eq!(m.repairs(), 0);
        assert!(m.start_sound());
    }
}

/// Given parameters the targets leave outside the set are projected back
/// at every row, and return once the targets do: `Q` stays positive
/// definite. Both sides of the cap at the call.
#[test]
fn given_parameters_are_held_in_the_set() {
    let k = 2;
    // g large enough that a + b + δ g passes the cap on these targets
    let rows = adcc_stream(k, 1500, [0.03, 0.93, 0.03], 0.3, None, 5);
    let mut m = Dcc::new(cfg(k, true, Some(vec![0.03, 0.9, 0.5]))).unwrap();
    for z in &rows {
        let was = m.started();
        let out = m.step(&[], &ys(z), 1.0, 1.0);
        if was {
            assert!(m.persistence() <= PERSISTENCE_CAP + 1e-15);
            assert!(out.pred[0].is_finite());
        }
    }
    let th = m.coefficients();
    assert!(th[2] < 0.5, "{th:?}");
    assert!((m.persistence() - PERSISTENCE_CAP).abs() < 1e-12);
    assert_eq!(m.repairs(), 0);
    // the call: a + b at the cap and past it
    assert!(
        cfg(2, false, Some(vec![0.5, PERSISTENCE_CAP - 0.5]))
            .validate()
            .is_ok()
    );
    let past = cfg(2, false, Some(vec![0.5, PERSISTENCE_CAP - 0.5 + 1e-12])).validate();
    assert!(past.unwrap_err().contains("a + b must be at most"));
    assert!(
        cfg(2, true, Some(vec![0.1, 0.8]))
            .validate()
            .unwrap_err()
            .contains("must hold 3")
    );
    assert!(cfg(2, false, Some(vec![-0.1, 0.8])).validate().is_err());
}

/// The asymmetric model with `g` given as 0 is the symmetric one, to the
/// bit.
#[test]
fn adcc_at_g_zero_is_dcc() {
    let rows = adcc_stream(3, 800, [0.05, 0.9, 0.0], 0.5, None, 9);
    let mut a = Dcc::new(cfg(3, false, Some(vec![0.05, 0.9]))).unwrap();
    let mut b = Dcc::new(cfg(3, true, Some(vec![0.05, 0.9, 0.0]))).unwrap();
    for z in &rows {
        let (x, y) = (a.step(&[], &ys(z), 1.0, 1.0), b.step(&[], &ys(z), 1.0, 1.0));
        assert_eq!(
            x.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

/// The start fires on its row, not one before, at any `start_rows` from 1
/// (0 refused): with the parameters given there is nothing to identify. The
/// floor `min_weight` either side.
#[test]
fn the_start_and_the_floor_hold_at_their_edges() {
    let rows = adcc_stream(2, 200, [0.04, 0.92, 0.0], 0.4, None, 4);
    // one row's `Q̄` is singular for two series: the start waits for two,
    // the condition and not a floor
    for (start, first) in [(1, 2), (2, 2), (37, 37)] {
        let mut c = cfg(2, false, Some(vec![0.05, 0.9]));
        c.start_rows = start;
        let mut m = Dcc::new(c.clone()).unwrap();
        let out = run(&mut m, &rows);
        assert!(out[first - 1][0].is_nan(), "{start}");
        assert!(out[first][0].is_finite(), "{start}");
    }
    let mut c = cfg(2, false, None);
    c.start_rows = 0;
    assert!(
        c.validate()
            .unwrap_err()
            .contains("start_rows must be at least 1")
    );
    c.start_rows = 1;
    assert!(c.validate().is_ok());
    // min_weight: the row whose weight before it is the floor reports, the
    // one before it does not
    let mut c = cfg(2, false, Some(vec![0.05, 0.9]));
    c.start_rows = 50;
    c.min_weight = 60.0;
    let mut m = Dcc::new(c).unwrap();
    let out = run(&mut m, &rows);
    assert!(out[59].iter().all(|v| v.is_nan()));
    assert!(out[60][0].is_finite());
}

/// `Q̄` from `n` rows of iid normal residuals, `k` series, as the start forms
/// it (unit weights, the running mean).
fn qbar_of(k: usize, n: usize, s: &mut u64) -> Vec<f64> {
    let mut qbar = vec![0.0; k * k];
    for t in 0..n {
        let z: Vec<f64> = (0..k).map(|_| normal(s)).collect();
        update_targets(&mut qbar, &mut [], &z, 1.0 / (t + 1) as f64, k);
    }
    qbar
}

/// The start's target is judged by its conditioning, not by the sign of a
/// rounded 0 (review W13, A1). `Q̄` of fewer rows than series is singular,
/// and its last pivot is an exact 0 computed with rounding: refused on
/// every seed, where `s > 0` once passed some (k = 4, two rows). One row
/// more spans, and is taken on every seed. With the parameters given and
/// `start_rows = 1` the start tries at 1, 2, 4 and 8 rows, keeping the
/// buffer while it lacks rank, so the first report is at the first of
/// those that holds `k` rows: k = 5 and 8 once cycled at 1, 2 and 4 rows
/// for ever, or started on a rank-4 target. Two series the same, with
/// random values (where the pivot is a rounded 0, not the true 0 of ±1),
/// never start.
#[test]
fn a_target_short_of_rank_is_refused_whatever_its_rounding() {
    let mut s = 7;
    for k in [2usize, 4, 8] {
        let mut l = vec![0.0; k * k];
        for _ in 0..300 {
            let short = qbar_of(k, k - 1, &mut s);
            assert!(!mat::definite(&short, k, &mut l), "{k}: {short:?}");
            let full = qbar_of(k, k, &mut s);
            assert!(mat::definite(&full, k, &mut l), "{k}: {full:?}");
        }
    }
    for k in [2usize, 4, 5, 8] {
        for seed in 0..40u64 {
            let mut s = 1000 + seed;
            let rows: Vec<Vec<f64>> = (0..40)
                .map(|_| (0..k).map(|_| normal(&mut s)).collect())
                .collect();
            let mut c = cfg(k, false, Some(vec![0.05, 0.9]));
            c.start_rows = 1;
            let mut m = Dcc::new(c).unwrap();
            let out = run(&mut m, &rows);
            let first = out.iter().position(|p| p[0].is_finite());
            assert_eq!(first, Some(k.next_power_of_two()), "{k} {seed}");
        }
    }
    let mut s = 3;
    for params in [None, Some(vec![0.05, 0.9])] {
        let mut c = cfg(3, false, params);
        c.start_rows = 20;
        let mut m = Dcc::new(c).unwrap();
        for _ in 0..600 {
            let (x, y) = (normal(&mut s), normal(&mut s));
            let out = m.step(&[], &ys(&[x, y, x]), 1.0, 1.0);
            assert!(out.pred[0].is_nan());
        }
        assert!(!m.started() && m.no_effect());
    }
}

/// Identification, the condition and not a count of rows: on a realistic
/// stream the start's likelihood-ratio statistic is far past its cut and
/// the fit is stepped from at the start's row; on a stream of constant
/// correlation (no dynamics to find, a realistic null) the fit is withheld,
/// the start keeps its rows and waits for twice them at each try, and
/// nothing is reported, for DCC and ADCC alike.
#[test]
fn a_start_reports_only_a_recursion_its_rows_identify() {
    for asym in [false, true] {
        let rows = adcc_stream(
            3,
            2000,
            [0.04, 0.92, if asym { 0.04 } else { 0.0 }],
            0.5,
            None,
            31,
        );
        let mut m = Dcc::new(cfg(3, asym, None)).unwrap();
        let out = run(&mut m, &rows);
        let first = out.iter().position(|p| p[0].is_finite()).unwrap();
        assert!(
            first >= DCC_START_ROWS
                && first % DCC_START_ROWS == 0
                && (first / DCC_START_ROWS).is_power_of_two(),
            "{asym}: started at {first}"
        );
        assert!(out[..first].iter().all(|p| p[0].is_nan()));
        assert!(out[first..].iter().all(|p| p[0].is_finite()));
        assert!(
            m.start_lr() > fit::lr_cut(m.cfg().n_params()),
            "{}",
            m.start_lr()
        );
        assert!(m.start_sound() && !m.no_effect());
        // the null: every target correlation 0.5, held
        for seed in [1u64, 2, 3] {
            let null = adcc_stream(3, 4000, [0.0, 0.0, 0.0], 0.5, None, seed);
            let mut c = cfg(3, asym, None);
            c.start_rows = 300;
            let mut m = Dcc::new(c).unwrap();
            let mut waits = 0;
            for (t, z) in null.iter().enumerate() {
                let out = m.step(&[], &ys(z), 1.0, 1.0);
                assert!(out.pred[0].is_nan(), "{asym} {seed} {t}");
                waits += usize::from(m.no_effect());
            }
            assert!(!m.started() && m.no_effect() && m.coefficients()[0].is_nan());
            assert!(
                m.start_lr() < fit::lr_cut(m.cfg().n_params()),
                "{}",
                m.start_lr()
            );
            // tried at 300, 600, 1,200 and 2,400 rows, the buffer kept
            assert_eq!((m.refits(), m.dropped()), (4, 0), "{asym} {seed}");
            assert_eq!(m.stage.rows(m.cfg()), 4000);
            assert!(waits > 3600);
        }
    }
}

/// Both sides of each cut: the likelihood-ratio statistic at its cut
/// identifies and a hair under it does not, for two and three parameters;
/// the projected step's gain at `10⁻⁶` nats is sound and a hair over it is
/// not; a statistic that is not a number identifies nothing, and an
/// information that is not definite neither.
#[test]
fn each_cut_holds_on_both_sides() {
    for (p, cut) in [(2, fit::LR_CUT_2), (3, fit::LR_CUT_3)] {
        let at =
            |lr: f64, gain: f64, definite: bool, max_se: f64, intercept_rel_se: f64| fit::Fit {
                theta: vec![0.05; p],
                gain,
                lr,
                definite,
                max_se,
                intercept_rel_se,
                bw_max_se: max_se,
                bw_feedback_rel_se: 0.2,
                bw_intercept_rel_se: intercept_rel_se,
                iterations: 1,
            };
        assert!(at(cut, 0.0, true, 0.05, 0.5).identified());
        assert!(!at(cut.next_down(), 0.0, true, 0.05, 0.5).identified());
        assert!(!at(f64::NAN, 0.0, true, 0.05, 0.5).identified());
        assert!(!at(cut * 2.0, 0.0, false, 0.05, 0.5).identified());
        assert!(at(cut, fit::SOUND_GAIN, true, 0.05, 0.5).sound());
        assert!(!at(cut, fit::SOUND_GAIN.next_up(), true, 0.05, 0.5).sound());
        assert!(at(cut, 0.0, true, fit::PARAM_SE_CUT, fit::INTERCEPT_REL_SE_CUT).se_identified());
        assert!(
            !at(
                cut,
                0.0,
                true,
                fit::PARAM_SE_CUT.next_up(),
                fit::INTERCEPT_REL_SE_CUT
            )
            .se_identified()
        );
        assert!(
            !at(
                cut,
                0.0,
                true,
                fit::PARAM_SE_CUT,
                fit::INTERCEPT_REL_SE_CUT.next_up()
            )
            .se_identified()
        );
        assert!(!at(cut, 0.0, true, f64::NAN, 0.5).se_identified());
        assert!(!at(cut, 0.0, true, f64::INFINITY, 0.5).se_identified());
        assert!(!at(cut, 0.0, true, 0.05, f64::NAN).se_identified());
        assert!(!at(cut, 0.0, true, 0.05, f64::INFINITY).se_identified());
    }
    // the cuts are χ² at 0.999 of 2 and 3 degrees of freedom: the survival
    // function there, 1 − P(χ² ≤ x), by the closed forms e^{−x/2} and
    // erfc(√(x/2)) + √(2x/π) e^{−x/2}
    assert!((((-fit::LR_CUT_2) / 2.0).exp() - 1e-3).abs() < 1e-15);
    let x = fit::LR_CUT_3;
    let sf3 =
        libm::erfc((x / 2.0).sqrt()) + (2.0 * x / std::f64::consts::PI).sqrt() * (-x / 2.0).exp();
    assert!((sf3 - 1e-3).abs() < 1e-12, "{sf3}");
}

/// The projected step's gain: 0 at an interior maximum and at a maximum on
/// a face of the set, where the unconstrained step leaves the set; the
/// quadratic's own gain `½ gᵀG⁻¹g` away from it.
#[test]
fn the_projected_gain_is_zero_at_an_optimum_inside_or_on_a_face() {
    let h = [2.0, 0.3, 0.3, 1.0];
    let c = [1.0, 1.0];
    // the quadratic −½ (θ − m)ᵀ H (θ − m): gradient H (m − θ)
    let grad = |th: [f64; 2], m: [f64; 2]| {
        let d = [m[0] - th[0], m[1] - th[1]];
        [h[0] * d[0] + h[1] * d[1], h[2] * d[0] + h[3] * d[1]]
    };
    let inside = [0.05, 0.9];
    assert!(fit::projected_gain(&inside, &grad(inside, inside), &h, &c) < 1e-15);
    // a maximum past a + b = cap: its nearest point in H's metric on the face
    let m = [0.3, 0.9];
    let face = region::project(&m, &h, &c, PERSISTENCE_CAP);
    assert!(fit::projected_gain(&face, &grad([face[0], face[1]], m), &h, &c) < 1e-12);
    // off the optimum, inside: ½ gᵀH⁻¹g
    let th = [0.05, 0.85];
    let g = grad(th, inside);
    let want = 0.5 * oracle::quad_form(&h, &g);
    assert!((fit::projected_gain(&th, &g, &h, &c) - want).abs() < 1e-15);
}

/// The state's bytes are the docs' formula, the call refuses past the cap
/// and takes the cap exactly.
#[test]
fn the_state_is_bounded_by_its_formula_and_the_cap() {
    for (asym, params, per) in [
        (false, None, (32, 96)),
        (true, None, (48, 144)),
        (false, Some(vec![0.05, 0.9]), (16, 64)),
        (true, Some(vec![0.03, 0.9, 0.03]), (24, 72)),
    ] {
        for k in [2, 3, 10] {
            let c = cfg(k, asym, params.clone());
            let k = k as u64;
            assert_eq!(c.stage_bytes(), per.0 * k * k + per.1, "{asym} {k}");
            let m = Dcc::new(c.clone()).unwrap();
            assert_eq!(m.bytes(), c.stage_bytes() + 32);
            assert_eq!(
                c.peak_bytes(),
                c.stage_bytes() + 32 + DCC_START_ROWS as u64 * 8 * (k + 2)
            );
        }
    }
    let mut c = cfg(3, true, None);
    let peak = c.peak_bytes();
    c.max_state_bytes = Some(peak);
    assert!(c.validate().is_ok());
    c.max_state_bytes = Some(peak - 1);
    assert!(c.validate().unwrap_err().contains("past max_state_bytes"));
    // the buffer is gone after the start
    let rows = adcc_stream(3, 1200, [0.04, 0.92, 0.04], 0.5, None, 8);
    let mut m = Dcc::new(cfg(3, true, None)).unwrap();
    for (t, z) in rows.iter().enumerate() {
        m.step(&[], &ys(z), 1.0, 1.0);
        let want = if !m.started() {
            (t as u64 + 1) * 8 * 5
        } else {
            0
        };
        assert_eq!(m.bytes(), m.cfg().stage_bytes() + 32 + want, "{t}");
    }
    assert!(m.started());
}

/// A save at every phase (in the start's buffer, at the start, after it)
/// loads and continues to the bit, with GARCH margins and without.
#[test]
fn a_save_and_load_continues_to_the_bit() {
    let rows = adcc_stream(2, 2900, [0.04, 0.92, 0.03], 0.4, None, 13);
    let mut with = cfg(2, true, None);
    with.margins = Some(GarchCfg {
        n_targets: 2,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 2000,
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
        n_inputs: 0,
        adaptive_start: false,
    });
    for c in [cfg(2, true, None), with] {
        for cut in [100, 499, 500, 1200, 2400, 2600] {
            let mut a = Dcc::new(c.clone()).unwrap();
            let (head, tail) = rows.split_at(cut);
            run(&mut a, head);
            let bytes = rmp_serde::to_vec(&a.state()).unwrap();
            let s: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut b = Dcc::restore(&s).unwrap();
            let (x, y) = (run(&mut a, tail), run(&mut b, tail));
            for (p, q) in x.iter().zip(&y) {
                assert_eq!(
                    p.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    q.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            assert_eq!(
                rmp_serde::to_vec(&a.state()).unwrap(),
                rmp_serde::to_vec(&b.state()).unwrap()
            );
        }
    }
}

/// The margins are the `garch` model's: its variances and standardized
/// residuals to the bit, run beside it on the same returns.
#[test]
fn the_margins_are_the_garch_models() {
    let mut s = 31;
    let k = 2;
    let z = adcc_stream(k, 3000, [0.04, 0.92, 0.0], 0.5, None, 17);
    // GARCH(1,1) returns around the correlated innovations
    let mut v = [1.0; 2];
    let mut e: [f64; 2] = [0.0; 2];
    let ret: Vec<Vec<f64>> = z
        .iter()
        .map(|zt| {
            for i in 0..k {
                v[i] = 0.05 + 0.08 * e[i] * e[i] + 0.9 * v[i];
                e[i] = v[i].sqrt() * zt[i];
            }
            let _ = lcg(&mut s);
            e.to_vec()
        })
        .collect();
    let gcfg = GarchCfg {
        n_targets: k,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 1000,
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
        n_inputs: 0,
        adaptive_start: false,
    };
    let mut c = cfg(k, false, None);
    c.margins = Some(gcfg.clone());
    let mut m = Dcc::new(c).unwrap();
    let mut g = Garch::new(gcfg).unwrap();
    let mut rho_from = None;
    for (t, r) in ret.iter().enumerate() {
        let a = m.step(&[], &ys(r), 1.0, 1.0);
        let b = g.step(&[], &ys(r), 1.0, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
        assert_eq!(a.pred[1].to_bits(), b.pred[2].to_bits());
        if rho_from.is_none() && a.pred[2].is_finite() {
            rho_from = Some(t);
        }
    }
    // the correlation from the margins' start and the stage's, which may
    // have waited for twice its rows
    let from = rho_from.unwrap();
    let stage_rows = from - 1000;
    assert!(
        stage_rows >= DCC_START_ROWS
            && stage_rows % DCC_START_ROWS == 0
            && (stage_rows / DCC_START_ROWS).is_power_of_two(),
        "{from}"
    );
    assert!((m.coefficients()[1] - 0.92).abs() < 0.1);
}

/// A margin whose fitted ν is at most 4 is said once; one above it is
/// not.
#[test]
fn a_margin_with_no_fourth_moment_is_said_once() {
    let k = 2;
    let mut s = 5;
    let t3 = |s: &mut u64| {
        let chi: f64 = (0..3).map(|_| normal(s).powi(2)).sum();
        normal(s) / (chi / 3.0).sqrt() / 3f64.sqrt()
    };
    let mut gcfg = GarchCfg {
        n_targets: k,
        vol: GarchVol::Garch,
        dist: GarchDist::T,
        truncation: None,
        variance_targeting: false,
        start_rows: 2000,
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
        n_inputs: 0,
        adaptive_start: false,
    };
    let mut c = cfg(k, false, None);
    c.margins = Some(gcfg.clone());
    let mut m = Dcc::new(c).unwrap();
    let mut said = 0;
    // t(3) innovations under GARCH(1,1) margins, a variance process to find
    let z: Vec<Vec<f64>> = (0..2300)
        .map(|_| (0..k).map(|_| t3(&mut s)).collect())
        .collect();
    for r in garch_returns(&z) {
        m.step(&[], &ys(&r), 1.0, 1.0);
        if let Some((_, nu)) = m.take_heavy_tail() {
            assert!(nu <= 4.0);
            said += 1;
        }
    }
    assert_eq!(said, 1);
    // t(10): not said
    gcfg.start_rows = 2000;
    let mut c = cfg(k, false, None);
    c.margins = Some(gcfg);
    let mut m = Dcc::new(c).unwrap();
    let z: Vec<Vec<f64>> = (0..2300)
        .map(|_| {
            (0..k)
                .map(|_| {
                    let chi: f64 = (0..10).map(|_| normal(&mut s).powi(2)).sum();
                    normal(&mut s) / (chi / 10.0).sqrt() * (0.8f64).sqrt()
                })
                .collect()
        })
        .collect();
    for r in garch_returns(&z) {
        m.step(&[], &ys(&r), 1.0, 1.0);
        assert!(m.take_heavy_tail().is_none());
    }
}

/// A row of weight 0, or with a residual absent, steps nothing and leaves
/// no NaN: the next row reports as if it were not there, its clock step
/// decaying the weight alone.
#[test]
fn a_row_of_no_weight_or_a_missing_series_steps_nothing() {
    let rows = adcc_stream(2, 800, [0.04, 0.92, 0.03], 0.4, None, 6);
    let mut c = cfg(2, true, None);
    c.decay = Decay::Halflife(5000.0);
    let mut a = Dcc::new(c.clone()).unwrap();
    let mut b = Dcc::new(c).unwrap();
    for (t, z) in rows.iter().enumerate() {
        if t == 650 {
            b.step(&[], &ys(z), 0.0, 0.0);
            b.step(&[], &[Some(z[0]), None], 0.0, 1.0);
        }
        let (x, y) = (a.step(&[], &ys(z), 1.0, 1.0), b.step(&[], &ys(z), 1.0, 1.0));
        assert_eq!(
            x.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{t}"
        );
    }
    // the weight a row of weight 0 leaves: decayed, not grown
    let mut m = Dcc::new(cfg(2, false, None)).unwrap();
    let first = m.step(&[], &ys(&rows[0]), 1.0, 0.0);
    assert_eq!(first.n_eff, 0.0);
    assert_eq!(m.n_eff(), 0.0);
    assert!(m.stage.qbar.iter().all(|v| *v == 0.0));
}

/// Under a decay the targets forget: a stream whose correlation turns
/// from 0.6 to -0.3 is read near -0.3 a few half-lives later, on
/// average over its last 1,000 rows,, and the gain
/// floors at `1 − λ`.
#[test]
fn a_decay_tracks_a_moving_correlation() {
    let mut s = 41;
    let k = 2;
    let mut c = cfg(k, false, Some(vec![0.03, 0.95]));
    c.decay = Decay::Halflife(200.0);
    let mut m = Dcc::new(c).unwrap();
    let mut last = 0.0;
    for t in 0..4000 {
        let rho: f64 = if t < 2000 { 0.6 } else { -0.3 };
        let (x, y) = (normal(&mut s), normal(&mut s));
        let z = [x, rho * x + (1.0 - rho * rho).sqrt() * y];
        if t >= 3000 {
            last += m.step(&[], &ys(&z), 1.0, 1.0).pred[0] / 1000.0;
        } else {
            m.step(&[], &ys(&z), 1.0, 1.0);
        }
    }
    // the mean correlation read over the last 1,000 rows
    assert!((last + 0.3).abs() < 0.1, "{last}");
    let (qbar, _) = m.targets();
    assert!((qbar[1] + 0.3).abs() < 0.2, "{qbar:?}");
    // the weight at its ceiling, 1 / (1 − λ)
    let lam = 0.5_f64.powf(1.0 / 200.0);
    assert!((m.stage.w - 1.0 / (1.0 - lam)).abs() < 2e-6 / (1.0 - lam));
}

/// A start over a span of the clock closes at its length, on the rows'
/// stamps, whatever rows it holds; its first span's buffer is checked
/// against the cap as rows come, warned of once past half and stopped at
/// the cap.
#[test]
fn a_start_on_the_clock_closes_at_its_span_and_holds_the_cap() {
    let rows = adcc_stream(2, 400, [0.04, 0.92, 0.0], 0.4, None, 2);
    // given parameters: the span alone decides where the start closes
    let mut c = cfg(2, false, Some(vec![0.05, 0.9]));
    c.start_span = Some(99.5);
    let mut m = Dcc::new(c.clone()).unwrap();
    let mut opened = None;
    for (t, z) in rows.iter().enumerate() {
        m.stamp_next(Stamp::from(t as f64 * 0.5));
        let out = m.step(&[], &ys(z), 0.5, 1.0);
        if opened.is_none() && out.pred[0].is_finite() {
            opened = Some(t);
        }
    }
    // row 199 is the first at or past 99.5 after row 0
    assert_eq!(opened, Some(200));
    // the cap: past half warned once, past the cap stopped
    let fixed = c.stage_bytes() + 32;
    c.max_state_bytes = Some(fixed + 100 * c.row_bytes());
    c.start_span = Some(1e9);
    let mut m = Dcc::new(c).unwrap();
    let mut warned = 0;
    for (t, z) in rows.iter().enumerate() {
        m.stamp_next(Stamp::from(t as f64));
        m.step(&[], &ys(z), 1.0, 1.0);
        if m.take_state_warning().is_some() {
            warned += 1;
            // the first row whose bytes pass half the cap
            assert!(2 * m.bytes() > m.cfg().max_state_bytes.unwrap());
            assert!(2 * (m.bytes() - m.cfg().row_bytes()) <= m.cfg().max_state_bytes.unwrap());
        }
        if t < 100 {
            assert!(m.state_over_cap().is_none(), "{t}");
        }
    }
    assert_eq!(warned, 1);
    assert!(m.state_over_cap().is_some());
    assert_eq!(m.stage.rows(m.cfg()), 100);
}

/// Scaling every weight changes nothing but `weight_sum`.
#[test]
fn a_weights_scale_changes_only_the_weight() {
    let rows = adcc_stream(2, 900, [0.04, 0.92, 0.03], 0.4, None, 19);
    let mut a = Dcc::new(cfg(2, true, None)).unwrap();
    let mut b = Dcc::new(cfg(2, true, None)).unwrap();
    for z in &rows {
        let x = a.step(&[], &ys(z), 1.0, 1.0);
        let y = b.step(&[], &ys(z), 1.0, 4.0);
        for (p, q) in x.pred.iter().zip(&y.pred) {
            assert!((p - q).abs() <= 1e-12 * (1.0 + p.abs()) || (p.is_nan() && q.is_nan()));
        }
        assert_eq!(4.0 * x.n_eff, y.n_eff);
    }
}

/// A buffer whose target is not positive definite (two series the same)
/// identifies no correlation: it is withheld and keeps its rows, waiting
/// for twice them while the cap holds them, since clearing cannot cure a
/// shortage of rank; nothing is reported, no NaN is left in the state. The
/// given parameters wait on it too.
#[test]
fn a_degenerate_start_begins_again() {
    for params in [None, Some(vec![0.05, 0.9])] {
        let mut m = Dcc::new({
            let mut c = cfg(2, false, params);
            c.start_rows = 50;
            c
        })
        .unwrap();
        for t in 0..400 {
            let v = if t % 2 == 0 { 1.0 } else { -1.0 };
            let out = m.step(&[], &[Some(v), Some(v)], 1.0, 1.0);
            assert!(out.pred[0].is_nan());
            if t >= 49 {
                assert!(m.no_effect() && !m.starting(), "{t}");
            }
        }
        // tried at 50, 100, 200 and 400 rows, the buffer kept
        assert_eq!((m.refits(), m.dropped()), (4, 0));
        assert_eq!(m.stage.rows(m.cfg()), 400);
        assert!(!m.started());
        assert!(m.stage.qbar.iter().all(|v| v.is_finite()));
    }
}

/// With GARCH margins, `predict_with` is the step without the step, to the
/// bit, on rows with a series absent and of weight 0 among them; `predict`
/// with no residuals the same but for the log-density.
#[test]
fn with_margins_predict_is_the_step() {
    let rows = garch_returns(&adcc_stream(2, 3000, [0.04, 0.92, 0.0], 0.5, None, 29));
    let mut c = cfg(2, false, None);
    c.start_rows = 200;
    c.margins = Some(GarchCfg {
        n_targets: 2,
        vol: GarchVol::Gjr,
        dist: GarchDist::T,
        truncation: None,
        variance_targeting: false,
        start_rows: 1000,
        start_span: None,
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
        n_inputs: 0,
        adaptive_start: false,
    });
    let mut m = Dcc::new(c).unwrap();
    let mut ready = 0;
    for (t, z) in rows.iter().enumerate() {
        let y: Vec<Option<f64>> = z
            .iter()
            .enumerate()
            .map(|(j, v)| (!(t % 13 == 5 && j == 1)).then_some(*v))
            .collect();
        let w = if t % 17 == 3 { 0.0 } else { 1.0 };
        let p = m.predict_with(&[], &y, 1.0);
        let q = m.predict(&[], 1.0);
        let s = m.step(&[], &y, 1.0, w);
        let bits = |r: &[f64]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p.pred), bits(&s.pred), "{t}");
        let n = s.pred.len();
        assert_eq!(bits(&q.pred[..n - 1]), bits(&s.pred[..n - 1]), "{t}");
        assert_eq!(p.n_eff, s.n_eff);
        if s.pred.iter().all(|v| v.is_finite()) {
            ready += 1;
        }
    }
    assert!(ready > 300, "{ready}");
}

/// No count of tries ends a start's wait (review W13, A2): the buffer is
/// bounded by `max_state_bytes` alone. On a null stream under a cap that
/// holds 700 rows the buffer grows to the cap and stays there, the oldest
/// row leaving as each new one comes, each try on rows the last did not
/// see; it is never cleared, never past the cap and never refused, and the
/// wait is named on every row after the first try. A stream that its first
/// few hundred rows do not identify, at `start_rows = 100`, is identified
/// once its buffer holds enough rows, past four times `start_rows`, where a
/// cap of two refits once cleared it at 400 rows and began again.
#[test]
fn a_start_waits_on_its_evidence_bounded_by_the_cap_alone() {
    let mut c = cfg(3, false, None);
    c.start_rows = 100;
    c.max_state_bytes = Some(c.stage_bytes() + 32 + 700 * c.row_bytes());
    let null = adcc_stream(3, 6000, [0.0, 0.0, 0.0], 0.5, None, 4);
    let mut m = Dcc::new(c.clone()).unwrap();
    let mut most = 0;
    for (t, z) in null.iter().enumerate() {
        let out = m.step(&[], &ys(z), 1.0, 1.0);
        assert!(out.pred[0].is_nan());
        let rows = m.stage.rows(m.cfg());
        assert!(rows >= most.min(700), "{t}: the buffer shrank to {rows}");
        most = most.max(rows);
        assert!(m.bytes() <= c.max_state_bytes.unwrap(), "{t}");
        assert!(m.state_over_cap().is_none(), "{t}");
        if t >= 99 {
            assert!(m.no_effect() && !m.starting(), "{t}");
        }
    }
    assert_eq!(most, 700);
    // 100, 200, 400, then 700 held at 800 rows and every 700 after
    assert_eq!(m.refits(), 3 + (6000 - 800) / 700 + 1);
    assert_eq!(m.dropped(), 6000 - 700);
    // the ring replays in order: the start's targets are the latest 700
    // rows' means, in the order they came
    let (mut ring, mut want) = (m.clone(), m.clone());
    ring.start();
    want.stage.buffer.rotate_left(want.stage.head * 5);
    want.stage.head = 0;
    want.start();
    assert_eq!(ring.stage.qbar, want.stage.qbar);
    assert_eq!(ring.stage.lr.to_bits(), want.stage.lr.to_bits());
    let mut later = 0;
    for seed in [5u64, 6, 7, 8, 9, 10] {
        let rows = adcc_stream(3, 6000, [0.02, 0.95, 0.0], 0.5, None, seed);
        let mut m = Dcc::new(c.clone()).unwrap();
        let out = run(&mut m, &rows);
        let first = out.iter().position(|p| p[0].is_finite());
        assert!(m.started() && m.start_sound(), "{seed}: {first:?}");
        later += usize::from(first.unwrap() > 400);
    }
    assert!(later > 0, "every seed started within 400 rows");
}

/// A try that is identified but short of its optimum is not withheld: the
/// wait after it is the start's own warm-up, not `not_identified`, whatever
/// the try before it found (review W13, A3).
#[test]
fn a_wait_names_its_last_try_alone() {
    let mut m = Dcc::new(cfg(3, false, None)).unwrap();
    m.wait(true);
    assert!(m.no_effect() && !m.starting());
    m.wait(false);
    assert!(!m.no_effect() && m.starting());
    assert_eq!(m.refits(), 2);
}

/// The estimator under a decay (review W13, A3): a start fitted on rows
/// whose weights decay, the weight the decayed sum of the rows' (so the
/// gain tends to its floor `1 − λ`), the information a decayed mean, and
/// the estimate
/// near the truth over a half-life long against the recursion's memory,
/// in the set throughout, DCC and ADCC alike.
#[test]
fn the_estimate_under_a_decay_finds_the_truth() {
    for (asym, seed) in [(false, 61u64), (true, 62)] {
        let k = 3;
        let truth = [0.04, 0.92, if asym { 0.04 } else { 0.0 }];
        let rows = adcc_stream(k, 30000, truth, 0.5, None, seed);
        let mut c = cfg(k, asym, None);
        c.decay = Decay::Halflife(5000.0);
        let mut m = Dcc::new(c).unwrap();
        let mut path = Vec::new();
        for (t, z) in rows.iter().enumerate() {
            m.step(&[], &ys(z), 1.0, 1.0);
            if m.started() {
                assert!(m.persistence() <= PERSISTENCE_CAP + 1e-15);
                assert!(m.coefficients().iter().all(|v| *v >= 0.0));
                if t >= 20000 {
                    path.push(m.coefficients());
                }
            }
        }
        assert!(m.started() && m.start_sound());
        let lam = 0.5_f64.powf(1.0 / 5000.0);
        // the weight of 30,000 unit rows by its closed form, which tends to
        // the ceiling 1/(1 − λ) and floors the gain at 1 − λ
        let want = (1.0 - lam.powi(30000)) / (1.0 - lam);
        assert!(
            (m.stage.w - want).abs() < 1e-9 * want,
            "{} {want}",
            m.stage.w
        );
        // the estimate's mean over the last 10,000 rows
        let mean: Vec<f64> = (0..m.cfg().n_params())
            .map(|i| path.iter().map(|p| p[i]).sum::<f64>() / path.len() as f64)
            .collect();
        assert!((mean[0] - truth[0]).abs() < 0.015, "{asym} {mean:?}");
        assert!((mean[1] - truth[1]).abs() < 0.04, "{asym} {mean:?}");
        if asym {
            assert!((mean[2] - truth[2]).abs() < 0.03, "{mean:?}");
        }
        // the information is a decayed mean of the rows', positive definite
        let p = m.cfg().n_params();
        let mut l = vec![0.0; p * p];
        assert!(mat::cholesky(&m.stage.h, p, &mut l));
        assert_eq!(m.repairs(), 0);
    }
}

/// A3: default adaptive start scales as `max(64, next_power_of_two(2 * k))`,
/// fitting waits until `n_kish > k` (even when raw row count exceeds `k`), and
/// short/noisy starts withhold under the Fisher SE gate before doubling to a
/// well-localized start.
#[test]
fn adaptive_start_and_kish_and_se_gate_hold() {
    assert_eq!(default_dcc_start_rows(2, false), 64);
    assert_eq!(default_dcc_start_rows(32, true), 64);
    assert_eq!(default_dcc_start_rows(33, false), 128);
    assert_eq!(default_dcc_start_rows(65, true), 256);

    // Concentrated weights: 64 rows have n_kish ≈ 1.0001 <= k = 3, so fitting
    // withholds as not_identified even though Q̄ is numerically positive definite.
    let rows = adcc_stream(3, 128, [0.05, 0.90, 0.0], 0.5, None, 71);
    let mut c = cfg(3, false, None);
    c.start_rows = 64;
    let mut m = Dcc::new(c).unwrap();
    for (t, z) in rows[..64].iter().enumerate() {
        let w = if t == 0 { 1e6 } else { 1.0 };
        let out = m.step(&[], &ys(z), 1.0, w);
        assert!(out.pred[0].is_nan());
    }
    assert!(!m.started() && m.no_effect(), "n_kish <= k must withhold");

    // Adaptive doubling from default_dcc_start_rows starts cleanly before 500
    // on a well-identified stream, while an explicit start_rows override stands.
    let long = adcc_stream(3, 1200, [0.05, 0.90, 0.04], 0.5, None, 73);
    for asym in [false, true] {
        let mut c = cfg(3, asym, None);
        c.start_rows = default_dcc_start_rows(3, asym);
        let mut m = Dcc::new(c).unwrap();
        let out = run(&mut m, &long);
        let first = out.iter().position(|p| p[0].is_finite()).unwrap();
        assert!(
            (64..=512).contains(&first) && first % 64 == 0 && (first / 64).is_power_of_two(),
            "asym={asym} started at {first}"
        );
        assert!(m.start_sound() && !m.no_effect());
    }

    // Adaptive RPEM handoff (`adaptive_start = true`) uses conservative
    // Bollerslev-Wooldridge QML sandwich SEs, round-trips through msgpack
    // preserving `adaptive_start`, and starts cleanly on heavy-tailed t(5)
    // streams with power-of-two weight invariance.
    let t5_rows = adcc_stream(3, 2200, [0.05, 0.90, 0.0], 0.5, Some(5.0), 79);
    let mut c_adapt = cfg(3, false, None);
    c_adapt.start_rows = default_dcc_start_rows(3, false);
    c_adapt.adaptive_start = true;
    let mut m_a = Dcc::new(c_adapt.clone()).unwrap();
    let rt: State = rmp_serde::from_slice(&rmp_serde::to_vec(&m_a.state()).unwrap()).unwrap();
    assert!(Dcc::restore(&rt).unwrap().cfg().adaptive_start);
    let mut m_b = Dcc::new(c_adapt).unwrap();
    let mut first_t5 = None;
    for (t, z) in t5_rows.iter().enumerate() {
        let pa = m_a.step(&[], &ys(z), 1.0, 1.0);
        let _pb = m_b.step(&[], &ys(z), 1.0, 4.0);
        assert_eq!(m_a.started(), m_b.started(), "t={t}");
        if pa.pred[0].is_finite() && first_t5.is_none() {
            first_t5 = Some(t);
        }
    }
    let ft = first_t5.expect("adaptive_start t(5) should start within 2200 rows");
    assert!(
        (256..=2048).contains(&ft) && ft % 64 == 0 && (ft / 64).is_power_of_two(),
        "adaptive_start t(5) started at {ft}"
    );
}
