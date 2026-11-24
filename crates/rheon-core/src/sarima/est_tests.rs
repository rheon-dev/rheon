//! The estimator: its transforms, its gradient, the start's gate on both
//! sides, and the rules every model keeps (out of sample, zero weights and
//! nulls, saves at every row, a break, the state's bound).

// Index loops over a matrix's rows and columns read plainest as indices.
#![allow(clippy::needless_range_loop)]

use super::est::{self, Buffer, Polys, Rings, Shape};
use super::transform::{constrain, partials, unconstrain};
use super::*;
use crate::model::OnlineModel;

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

fn params(ar: &[f64], ma: &[f64], sar: &[f64], sma: &[f64], sigma2: f64) -> SarimaParams {
    SarimaParams {
        exog: vec![],
        ar: ar.to_vec(),
        ma: ma.to_vec(),
        seasonal_ar: sar.to_vec(),
        seasonal_ma: sma.to_vec(),
        sigma2,
    }
}

/// `n` values of the model with `truth`'s parameters and `x` its regressors
/// (`n x k`, row-major), after a burn-in: the ARMA part by its two
/// polynomials, integrated, the regressors' part added.
fn simulate(
    order: [usize; 3],
    seasonal_order: [usize; 4],
    truth: &SarimaParams,
    x: &[f64],
    n: usize,
    seed: u64,
) -> Vec<f64> {
    let s = seasonal_order[3].max(1);
    let a = poly::polymul(
        &poly::lag_poly(&truth.ar, 1, -1.0),
        &poly::lag_poly(&truth.seasonal_ar, s, -1.0),
    );
    let c = poly::polymul(
        &poly::lag_poly(&truth.ma, 1, 1.0),
        &poly::lag_poly(&truth.seasonal_ma, s, 1.0),
    );
    let mut rng = seed;
    let burn = 500;
    let (mut u, mut e) = (Vec::new(), Vec::new());
    for t in 0..n + burn {
        e.push(truth.sigma2.sqrt() * normal(&mut rng));
        let mut v = 0.0;
        for (i, ai) in a.iter().enumerate().skip(1) {
            if t >= i {
                v -= ai * u[t - i];
            }
        }
        for (j, cj) in c.iter().enumerate() {
            if t >= j {
                v += cj * e[t - j];
            }
        }
        u.push(v);
    }
    let mut y: Vec<f64> = u[burn..].to_vec();
    for _ in 0..seasonal_order[1] {
        for t in s..n {
            y[t] += y[t - s];
        }
    }
    for _ in 0..order[1] {
        for t in 1..n {
            y[t] += y[t - 1];
        }
    }
    let k = truth.exog.len();
    for t in 0..n {
        for m in 0..k {
            y[t] += x[t * k + m] * truth.exog[m];
        }
    }
    y.iter().map(|v| v + 10.0).collect()
}

fn estimated(order: [usize; 3], seasonal_order: [usize; 4], k: usize) -> SarimaCfg {
    SarimaCfg {
        n_targets: 1,
        n_features: k,
        order,
        seasonal_order,
        params: vec![],
        spacing: 1.0,
        min_weight: 0.0,
        max_state_bytes: Some(4 << 20),
        estimate: Some(SarimaEstimate::default()),
    }
}

/// Each row's prediction, through `step` with unit clock steps and weights.
fn run(m: &mut Sarima, ys: &[Option<f64>], x: &[f64], w: &[f64]) -> Vec<f64> {
    let k = m.cfg().n_features;
    let mut out = Step::default();
    let mut preds = Vec::with_capacity(ys.len());
    for (t, y) in ys.iter().enumerate() {
        m.step_into(&x[t * k..(t + 1) * k], &[*y], 1.0, w[t], &mut out);
        preds.push(out.pred[0]);
    }
    preds
}

fn some(ys: &[f64]) -> Vec<Option<f64>> {
    ys.iter().map(|v| Some(*v)).collect()
}

/// The first row whose prediction is a number.
fn opened(preds: &[f64]) -> Option<usize> {
    preds.iter().position(|v| v.is_finite())
}

#[test]
fn the_transforms_are_statsmodels() {
    // `SARIMAX(order=(3, 0, 2)).transform_params([0.3, -1.2, 0.7, 0.4,
    // -0.9, 1.1])`, statsmodels 0.15.0.
    let (ar, _) = constrain(&[0.3, -1.2, 0.7], true);
    let (ma, _) = constrain(&[0.4, -0.9], false);
    let want_ar = [0.37394485076073986, 0.7300280421874836, -0.5734623443633283];
    let want_ma = [0.12294341221980065, -0.6689647316224496];
    for (a, b) in ar.iter().chain(&ma).zip(want_ar.iter().chain(&want_ma)) {
        assert!((a - b).abs() <= 4.0 * f64::EPSILON, "{a} {b}");
    }
    // Every image stationary by the step-down test, and the inverse.
    let mut rng = 11;
    for n in 1..=5 {
        for _ in 0..50 {
            let u: Vec<f64> = (0..n).map(|_| 1.5 * normal(&mut rng)).collect();
            let (phi, _) = constrain(&u, true);
            assert!(poly::stationary(&phi), "{phi:?}");
            let back = unconstrain(&phi, true).unwrap();
            for (a, b) in u.iter().zip(&back) {
                assert!((a - b).abs() <= 1e-8 * (1.0 + a.abs()), "{u:?} {back:?}");
            }
            // An invertible MA factor is a stationary AR factor of -theta.
            let (theta, _) = constrain(&u, false);
            let neg: Vec<f64> = theta.iter().map(|v| -v).collect();
            assert!(poly::stationary(&neg));
        }
    }
    assert!(unconstrain(&[1.0], true).is_none());
}

#[test]
fn the_jacobian_is_the_transforms_derivative() {
    let u = [0.3, -1.2, 0.7, 2.0];
    for ar in [true, false] {
        let (_, jac) = constrain(&u, ar);
        for m in 0..u.len() {
            let h = 1e-6;
            let (mut up, mut dn) = (u, u);
            up[m] += h;
            dn[m] -= h;
            let (a, _) = constrain(&up, ar);
            let (b, _) = constrain(&dn, ar);
            for i in 0..u.len() {
                let fd = (a[i] - b[i]) / (2.0 * h);
                assert!(
                    (fd - jac[i * 4 + m]).abs() < 1e-8,
                    "{i} {m} {fd} {}",
                    jac[i * 4 + m]
                );
            }
        }
    }
}

fn psi_buffer(missing: bool) -> Buffer {
    let mut rng = 5;
    let mut buf = Buffer::default();
    for t in 0..300 {
        let gap = missing && t % 37 == 5;
        buf.w.push(if gap { f64::NAN } else { normal(&mut rng) });
        buf.weight.push(if gap { 0.0 } else { 0.5 + lcg(&mut rng) });
        buf.xw.push(normal(&mut rng));
        buf.xw.push(if missing && t == 90 {
            f64::NAN
        } else {
            normal(&mut rng)
        });
    }
    buf
}

const PSI_SHAPE: Shape = Shape {
    k: 2,
    p: 1,
    q: 1,
    sp: 1,
    sq: 1,
    s: 4,
    stationary: true,
    invertible: true,
};

const PSI_PAR: [f64; 6] = [0.7, -0.4, 0.5, 0.3, 0.4, -0.5];

/// The start's gradient is exact: the conditional sum of squares'
/// gradient, `-2 sum w e psi`, against its central difference, at a
/// seasonal spec with regressors, missing slots and a regressor not known.
#[test]
fn the_starts_gradient_is_the_sum_of_squares_gradient() {
    let buf = psi_buffer(true);
    let g = est::run(&PSI_SHAPE, &buf, &PSI_PAR, true).g;
    for i in 0..6 {
        let h = 1e-6;
        let (mut up, mut dn) = (PSI_PAR, PSI_PAR);
        up[i] += h;
        dn[i] -= h;
        let s_up = est::run(&PSI_SHAPE, &buf, &up, false).s;
        let s_dn = est::run(&PSI_SHAPE, &buf, &dn, false).s;
        let fd = (s_up - s_dn) / (2.0 * h);
        let exact = -2.0 * g[i];
        assert!(
            (fd - exact).abs() <= 1e-6 * (1.0 + exact.abs()),
            "{i}: {fd} {exact}"
        );
    }
}

/// With no slot missing, the filtered signals the per-row step reads give
/// the same `psi` as the exact derivatives: they are the conditional sum of
/// squares' derivatives at fixed parameters.
#[test]
fn the_signals_are_the_exact_psi_with_no_slot_missing() {
    let buf = psi_buffer(false);
    let pol = Polys::new(&PSI_SHAPE, &PSI_PAR);
    let mut rings = Rings::new(&PSI_SHAPE);
    let (mut xf, mut psi) = (Vec::new(), Vec::new());
    let (mut g, mut s) = (vec![0.0; 6], 0.0);
    for t in 0..buf.w.len() {
        let x = &buf.xw[2 * t..2 * t + 2];
        if let Some(e) = rings
            .pass(&PSI_SHAPE, &pol, Some(buf.w[t]), Some(x), &mut xf, &mut psi)
            .e
        {
            s += buf.weight[t] * e * e;
            for i in 0..6 {
                g[i] += buf.weight[t] * e * psi[i];
            }
        }
    }
    let exact = est::run(&PSI_SHAPE, &buf, &PSI_PAR, true);
    assert_eq!(s.to_bits(), exact.s.to_bits());
    for i in 0..6 {
        assert!(
            (g[i] - exact.g[i]).abs() <= 1e-10 * (1.0 + g[i].abs()),
            "{i}: {} {}",
            g[i],
            exact.g[i]
        );
    }
}

/// Both sides of the gate: below the structural minimum nothing is
/// fitted; past it, on a series its spec identifies, a start is accepted,
/// its parameters near the truth, and every row from the next one
/// predicts.
#[test]
fn the_gate_opens_on_an_identified_start() {
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let n = 1500;
    let ys = simulate(order, seasonal, &truth, &[], n, 3);
    let c = estimated(order, seasonal, 0);
    assert_eq!(c.structural_minimum(), 7.0);
    let mut m = Sarima::new(c).unwrap();
    let w = vec![1.0; n];
    let preds = run(&mut m, &some(&ys[..6]), &[], &w);
    assert!(preds.iter().all(|v| v.is_nan()));
    // The first four rows fill the seasonal difference; the next two are
    // buffered, Kish's count 2, under the minimum of 7.
    assert_eq!(m.wait(0), Some(SarimaWait::Buffering));
    assert_eq!(m.buffered_kish(0), Some(2.0));
    let rest = run(&mut m, &some(&ys[6..]), &[], &w[6..]);
    let open = opened(&rest).expect("the start was never accepted") + 6;
    assert_eq!(m.wait(0), None);
    assert!(rest[open - 6..].iter().all(|v| v.is_finite()));
    // Opened once the standard errors came under the bound: well after the
    // structural minimum and well before the end.
    assert!((60..800).contains(&open), "opened at row {open}");
    let fit = m.params(0).unwrap();
    for (got, want) in [
        (fit.ar[0], 0.6),
        (fit.seasonal_ma[0], -0.5),
        (fit.sigma2, 1.0),
    ] {
        assert!((got - want).abs() < 0.12, "{fit:?}");
    }
    // The buffer is gone, and the state is the run's.
    assert_eq!(m.bytes(), Sarima::state_bytes(m.cfg()));
}

/// `se_bound` on both sides, on the series the gate opens on above: at the
/// default the start is accepted; tightened to 0.02, past the standard
/// error 1,500 rows can give (about `1 / sqrt(1500)`, 0.026, for `sigma2`'s
/// relative one and more for the coefficients'), it is withheld and says
/// why; loosened to 0.3 it opens earlier. The bound reads `sigma2`'s
/// standard error relative to it and the coefficients' as they are, so the
/// same series in other units (times 1,024, exact in binary) opens at the
/// same row.
#[test]
fn se_bound_is_the_gate_on_both_sides() {
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let n = 1500;
    let ys = simulate(order, seasonal, &truth, &[], n, 3);
    let w = vec![1.0; n];
    let at = |bound: f64, scale: f64| {
        let c = SarimaCfg {
            estimate: Some(SarimaEstimate {
                se_bound: bound,
                ..SarimaEstimate::default()
            }),
            ..estimated(order, seasonal, 0)
        };
        let mut m = Sarima::new(c).unwrap();
        let ys: Vec<f64> = ys.iter().map(|v| v * scale).collect();
        let preds = run(&mut m, &some(&ys), &[], &w);
        (opened(&preds), m.wait(0), m.params(0))
    };
    let (open, wait, fit) = at(SarimaEstimate::SE_BOUND, 1.0);
    let open = open.expect("accepted at the default bound");
    assert_eq!(wait, None);
    assert!((fit.unwrap().ar[0] - 0.6).abs() < 0.12);
    let (tight, wait, fit) = at(0.02, 1.0);
    assert_eq!(tight, None);
    assert_eq!(wait, Some(SarimaWait::NotIdentified));
    assert_eq!(
        wait.unwrap().name(),
        "start not identified: standard errors past the bound"
    );
    assert!(fit.is_none());
    let (loose, wait, _) = at(0.3, 1.0);
    assert_eq!(wait, None);
    assert!(loose.unwrap() < open, "{loose:?} against {open}");
    let (scaled, wait, fit) = at(SarimaEstimate::SE_BOUND, 1024.0);
    assert_eq!((scaled, wait), (Some(open), None));
    assert!((fit.unwrap().sigma2 / (1024.0 * 1024.0) - 1.0).abs() < 0.12);
}

/// A bound that is not finite and above 0 is refused, naming it.
#[test]
fn se_bound_must_be_finite_and_above_zero() {
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        let c = SarimaCfg {
            estimate: Some(SarimaEstimate {
                se_bound: bad,
                ..SarimaEstimate::default()
            }),
            ..estimated([1, 0, 0], [0, 0, 0, 0], 0)
        };
        let e = Sarima::new(c).unwrap_err();
        assert!(e.contains("se_bound must be finite and above 0"), "{e}");
    }
}

/// With regressors: their coefficients estimated beside the polynomials'.
#[test]
fn regressors_are_estimated_with_the_polynomials() {
    let order = [1, 1, 0];
    let seasonal = [0, 0, 0, 0];
    let mut truth = params(&[0.5], &[], &[], &[], 0.5);
    truth.exog = vec![2.0, -1.0];
    let n = 2000;
    let mut rng = 77;
    let x: Vec<f64> = (0..2 * n).map(|_| normal(&mut rng)).collect();
    let ys = simulate(order, seasonal, &truth, &x, n, 9);
    let mut m = Sarima::new(estimated(order, seasonal, 2)).unwrap();
    let preds = run(&mut m, &some(&ys), &x, &vec![1.0; n]);
    assert!(opened(&preds).is_some());
    let fit = m.params(0).unwrap();
    assert!((fit.exog[0] - 2.0).abs() < 0.1, "{fit:?}");
    assert!((fit.exog[1] + 1.0).abs() < 0.1, "{fit:?}");
    assert!((fit.ar[0] - 0.5).abs() < 0.1, "{fit:?}");
}

/// The withheld side, each with its reason. Two regressors that are the
/// same column: the likelihood is flat along `beta_1 + beta_2` held, its
/// information singular, so the standard errors are past the bound at any
/// length.
fn flat(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut truth = params(&[0.5], &[], &[], &[], 1.0);
    truth.exog = vec![1.0, 1.0];
    let mut rng = 31;
    let mut x = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let v = normal(&mut rng);
        x.extend([v, v]);
    }
    (simulate([1, 0, 0], [0, 0, 0, 0], &truth, &x, n, 6), x)
}

#[test]
fn a_flat_likelihood_is_not_identified() {
    let (order, seasonal, n) = ([1, 0, 0], [0, 0, 0, 0], 1500);
    let (ys, x) = flat(n);
    let mut m = Sarima::new(estimated(order, seasonal, 2)).unwrap();
    let preds = run(&mut m, &some(&ys), &x, &vec![1.0; n]);
    assert!(preds.iter().all(|v| v.is_nan()));
    assert_eq!(m.wait(0), Some(SarimaWait::NotIdentified));
    assert!(m.params(0).is_none());
}

/// A series that alternates in sign, with a little noise, is an AR(1)
/// with `phi` a hair above -1: the fit runs to the stationarity bound.
#[test]
fn a_fit_on_the_stationarity_bound_is_withheld() {
    let (order, seasonal) = ([1, 0, 0], [0, 0, 0, 0]);
    let mut rng = 4;
    let ys: Vec<f64> = (0..600)
        .map(|t| if t % 2 == 0 { 5.0 } else { -5.0 } + 0.01 * normal(&mut rng))
        .collect();
    let mut m = Sarima::new(estimated(order, seasonal, 0)).unwrap();
    let preds = run(&mut m, &some(&ys), &[], &vec![1.0; ys.len()]);
    assert!(preds.iter().all(|v| v.is_nan()));
    assert_eq!(m.wait(0), Some(SarimaWait::OnBound));
    // Held to no bound, the same rows never name one.
    let mut free = Sarima::new(SarimaCfg {
        estimate: Some(SarimaEstimate {
            enforce_stationarity: false,
            ..SarimaEstimate::default()
        }),
        ..estimated(order, seasonal, 0)
    })
    .unwrap();
    let mut out = Step::default();
    for y in &ys {
        free.step_into(&[], &[Some(*y)], 1.0, 1.0, &mut out);
        assert_ne!(free.wait(0), Some(SarimaWait::OnBound));
    }
}

/// A buffer whose likelihood has no finite value (every differenced value
/// 0, so the residuals' mean square is 0) never converges, and says so.
#[test]
fn a_fit_with_no_finite_likelihood_has_not_converged() {
    let mut m = Sarima::new(estimated([1, 1, 0], [0, 0, 0, 0], 0)).unwrap();
    let ys = vec![Some(3.0); 50];
    run(&mut m, &ys, &[], &[1.0; 50]);
    assert_eq!(m.wait(0), Some(SarimaWait::NotConverged));
    let shape = m.cfg().shape().unwrap();
    let buf = &m.est[0].buf;
    assert!(matches!(
        est::fit(&shape, buf, est::SE_BOUND),
        Err(SarimaWait::NotConverged)
    ));
}

/// A retry waits for Kish's count to grow by the stated factor: the fits
/// a long wait costs grow with the log of its length.
#[test]
fn a_withheld_start_is_tried_again_as_its_count_grows() {
    let (ys, x) = flat(1000);
    let mut m = Sarima::new(estimated([1, 0, 0], [0, 0, 0, 0], 2)).unwrap();
    let mut out = Step::default();
    let mut tries = Vec::new();
    let mut last = 0.0;
    for (t, y) in ys.iter().enumerate() {
        m.step_into(&x[2 * t..2 * t + 2], &[Some(*y)], 1.0, 1.0, &mut out);
        let next = m.est[0].next_try;
        if next != last {
            tries.push(next / est::RETRY_GROWTH);
            last = next;
        }
    }
    assert!(tries[0] > m.cfg().structural_minimum());
    for pair in tries.windows(2) {
        assert!(pair[1] >= pair[0] * est::RETRY_GROWTH, "{tries:?}");
    }
    // From the minimum's first try at 5 to 1,000 rows: log_1.5(1000 / 5)
    // is 13.1, so 14 tries at most.
    assert!(tries.len() <= 14, "{tries:?}");
}

/// Hard rule 2: changing row `t`'s value moves no prediction up to and
/// including row `t`, before the start and after it.
#[test]
fn predictions_are_out_of_sample() {
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let ys = simulate(order, seasonal, &truth, &[], 900, 3);
    let c = estimated(order, seasonal, 0);
    let base = run(
        &mut Sarima::new(c.clone()).unwrap(),
        &some(&ys),
        &[],
        &[1.0; 900],
    );
    let open = opened(&base).unwrap();
    for t in [40, open - 1, open, open + 50] {
        let mut moved = ys.clone();
        moved[t] += 100.0;
        let got = run(
            &mut Sarima::new(c.clone()).unwrap(),
            &some(&moved),
            &[],
            &[1.0; 900],
        );
        for i in 0..=t {
            assert_eq!(
                got[i].to_bits(),
                base[i].to_bits(),
                "row {i} moved by row {t}"
            );
        }
    }
}

/// Hard rule 9: a row of weight 0 and a null target are the same missing
/// slot, kept in its place; a zero-weight first row learns nothing.
#[test]
fn a_zero_weight_and_a_null_are_the_same_missing_slot() {
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let n = 900;
    let ys = simulate(order, seasonal, &truth, &[], n, 8);
    let gone = |t: usize| t == 0 || t % 13 == 7 || (300..320).contains(&t);
    let w: Vec<f64> = (0..n).map(|t| if gone(t) { 0.0 } else { 1.0 }).collect();
    let nulls: Vec<Option<f64>> = (0..n).map(|t| (!gone(t)).then_some(ys[t])).collect();
    let c = estimated(order, seasonal, 0);
    let mut a = Sarima::new(c.clone()).unwrap();
    let mut b = Sarima::new(c).unwrap();
    let pa = run(&mut a, &some(&ys), &[], &w);
    let pb = run(&mut b, &nulls, &[], &[1.0; 900]);
    assert!(opened(&pa).is_some());
    for (x, y) in pa.iter().zip(&pb) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    assert_eq!(a.params(0), b.params(0));
    // Every slot is kept: the zero-weight slots are in the count of slots
    // the estimator saw, so the seasonal positions after them stay put.
    let mut first = Sarima::new(estimated(order, seasonal, 0)).unwrap();
    let mut out = Step::default();
    first.step_into(&[], &[Some(5.0)], 1.0, 0.0, &mut out);
    assert_eq!(out.n_eff, 0.0);
    assert_eq!(first.n_eff(), 0.0);
    assert!(first.series[0].seen.iter().all(|s| !s));
}

/// Chunk invariance (hard rule 3) at the model: a save and a load at every
/// row, through the buffer, the start and the steps after it, continues to
/// the bit.
#[test]
fn a_save_at_every_row_continues_to_the_bit() {
    let order = [1, 1, 0];
    let seasonal = [0, 0, 1, 4];
    let mut truth = params(&[0.4], &[], &[], &[-0.4], 0.7);
    truth.exog = vec![1.5];
    let n = 700;
    let mut rng = 2;
    let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
    let ys = simulate(order, seasonal, &truth, &x, n, 12);
    let c = estimated(order, seasonal, 1);
    let mut whole = Sarima::new(c.clone()).unwrap();
    let mut saved = Sarima::new(c).unwrap();
    let mut out = Step::default();
    let mut opened_at = None;
    for t in 0..n {
        let w = if t % 17 == 3 {
            0.0
        } else {
            1.0 + (t % 3) as f64
        };
        let y = (t % 29 != 11).then_some(ys[t]);
        let d = if t % 41 == 40 { 3.0 } else { 1.0 };
        whole.step_into(&x[t..t + 1], &[y], d, w, &mut out);
        let a = out.pred[0];
        let bytes = rmp_serde::to_vec_named(&saved.state()).unwrap();
        let state: State = rmp_serde::from_slice(&bytes).unwrap();
        saved = Sarima::restore(&state).unwrap();
        saved.step_into(&x[t..t + 1], &[y], d, w, &mut out);
        assert_eq!(a.to_bits(), out.pred[0].to_bits(), "row {t}");
        if a.is_finite() && opened_at.is_none() {
            opened_at = Some(t);
        }
    }
    assert!(opened_at.is_some(), "the start never opened");
    assert_eq!(whole, saved);
}

/// A break starts each series again: its ring, and its start fitted again
/// from a new buffer; the weights kept.
#[test]
fn a_break_fits_the_start_again() {
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let ys = simulate(order, seasonal, &truth, &[], 1200, 3);
    let mut m = Sarima::new(estimated(order, seasonal, 0)).unwrap();
    run(&mut m, &some(&ys), &[], &[1.0; 1200]);
    assert!(m.params(0).is_some());
    let weight = m.series[0].weight;
    m.clear_lags();
    assert_eq!(m.wait(0), Some(SarimaWait::Filling));
    assert!(m.params(0).is_none());
    assert_eq!(m.series[0].weight, weight);
    let after = run(&mut m, &some(&ys[..300]), &[], &[1.0; 300]);
    assert!(after[..20].iter().all(|v| v.is_nan()));
}

/// The buffer is held under `max_state_bytes`: the state's bytes as it
/// grows, a warning past half of the cap, and the slot that would pass it
/// refused, after which nothing is buffered.
#[test]
fn the_buffer_is_held_under_the_cap() {
    let c = estimated([1, 0, 0], [0, 0, 0, 0], 2);
    let fixed = Sarima::state_bytes(&c);
    let slot = Sarima::buffer_bytes(&c);
    assert_eq!(slot, 36);
    let (ys, x) = flat(150);
    let cap = fixed + 100 * slot;
    let mut m = Sarima::new(SarimaCfg {
        max_state_bytes: Some(cap),
        ..c
    })
    .unwrap();
    let mut out = Step::default();
    let mut warned_at = None;
    for t in 0..150 {
        m.step_into(&x[2 * t..2 * t + 2], &[Some(ys[t])], 1.0, 1.0, &mut out);
        assert!(m.bytes() <= cap);
        if let Some((bytes, at)) = m.take_state_warning() {
            assert_eq!(at, cap);
            assert!(2 * bytes > cap);
            warned_at.get_or_insert(t);
        }
    }
    assert!(warned_at.is_some());
    let (bytes, at) = m.state_over_cap().unwrap();
    assert_eq!(at, cap);
    assert_eq!(bytes, fixed + 101 * slot);
    assert_eq!(m.est[0].buf.w.len(), 100);
    // The serialized state is within the bound the bytes give.
    let file = rmp_serde::to_vec_named(&m.state()).unwrap().len() as u64;
    assert!(file <= m.bytes(), "{file} > {}", m.bytes());
}

/// The state's size, as the docs state it: per target the run, `9 (n + 2 +
/// n^2 + L (6 + 2k))` bytes with `L = max(p + sP, q + sQ, 1)`, linear in
/// the season where the filter's is quadratic, and the file within it.
#[test]
fn the_state_after_the_start_is_linear_in_the_season() {
    let at = |s: usize| {
        Sarima::state_bytes(&SarimaCfg {
            n_targets: 1,
            ..estimated([1, 1, 1], [1, 1, 1, s], 0)
        })
    };
    let (a, b, c) = (at(24), at(48), at(96));
    assert_eq!(c - b, 2 * (b - a), "{a} {b} {c}");
    let shape = estimated([1, 1, 1], [1, 1, 1, 168], 0).shape().unwrap();
    assert_eq!(shape.lags(), 169);
    assert_eq!(est::Run::len(&shape), 4 + 2 + 16 + 169 * 6);
    // A running state's file is within its bound.
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let ys = simulate(order, seasonal, &truth, &[], 1200, 3);
    let mut m = Sarima::new(estimated(order, seasonal, 0)).unwrap();
    run(&mut m, &some(&ys), &[], &[1.0; 1200]);
    assert!(m.params(0).is_some());
    let file = rmp_serde::to_vec_named(&m.state()).unwrap().len() as u64;
    assert!(file <= Sarima::state_bytes(m.cfg()), "{file}");
    assert!(
        4 * file > Sarima::state_bytes(m.cfg()),
        "the bound is loose: {file}"
    );
}

/// The partial autocorrelations the bound test reads are the transform's.
#[test]
fn the_bound_reads_the_partial_autocorrelations() {
    let shape = Shape {
        k: 0,
        p: 2,
        q: 1,
        sp: 0,
        sq: 0,
        s: 0,
        stationary: true,
        invertible: true,
    };
    assert!(!shape.on_bound(&[0.5, -2.0, 1.0]));
    // |r| = 1 - 1e-3 at u = r / sqrt(1 - r^2) = 22.33...
    assert!(shape.on_bound(&[0.5, 23.0, 1.0]));
    assert!(shape.on_bound(&[0.5, 1.0, -23.0]));
    let r: Vec<f64> = partials(&[22.0]).collect();
    assert!(r[0] < 1.0 - est::BOUND_MARGIN);
    let off = Shape {
        invertible: false,
        ..shape
    };
    assert!(!off.on_bound(&[0.5, 1.0, -23.0]));
}

/// `min_weight` reads the target's own weight, on top of the gate.
#[test]
fn min_weight_is_on_top_of_the_gate() {
    let order = [1, 0, 0];
    let seasonal = [0, 1, 1, 4];
    let truth = params(&[0.6], &[], &[], &[-0.5], 1.0);
    let ys = simulate(order, seasonal, &truth, &[], 1200, 3);
    let c = estimated(order, seasonal, 0);
    let free = run(
        &mut Sarima::new(c.clone()).unwrap(),
        &some(&ys),
        &[],
        &[1.0; 1200],
    );
    let open = opened(&free).unwrap();
    let floor = (open + 100) as f64;
    let gated = run(
        &mut Sarima::new(SarimaCfg {
            min_weight: floor,
            ..c
        })
        .unwrap(),
        &some(&ys),
        &[],
        &[1.0; 1200],
    );
    let first = opened(&gated).unwrap();
    // The target's weight counts the rows that observed it before this one.
    assert_eq!(first, open + 100);
    for t in first..1200 {
        assert_eq!(gated[t].to_bits(), free[t].to_bits());
    }
}
