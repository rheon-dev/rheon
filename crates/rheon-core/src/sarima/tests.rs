use super::*;
use crate::oracle;

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

/// `(1,0,1)(1,0,1,4)`, its polynomials far enough inside the unit circle
/// that the MA(inf) weights die out within a few hundred lags.
fn arma_params() -> SarimaParams {
    SarimaParams {
        exog: vec![],
        ar: vec![0.5],
        ma: vec![0.3],
        seasonal_ar: vec![0.4],
        seasonal_ma: vec![-0.3],
        sigma2: 1.7,
    }
}

fn cfg(order: [usize; 3], seasonal_order: [usize; 4], params: SarimaParams) -> SarimaCfg {
    SarimaCfg {
        n_targets: 1,
        n_features: params.exog.len(),
        order,
        seasonal_order,
        params: vec![params],
        spacing: 1.0,
        min_weight: 0.0,
        max_state_bytes: None,
        estimate: None,
    }
}

/// `n` values of the ARMA part `A(B) u = C(B) e` by its own recursion,
/// after a burn-in, scaled by `sd`; then integrated by `(1-B)^d (1-B^s)^D`
/// and offset by `level`.
fn simulate(c: &SarimaCfg, n: usize, seed: u64, level: f64) -> Vec<f64> {
    let p = &c.params[0];
    let s = c.s().max(1);
    let a = poly::polymul(
        &poly::lag_poly(&p.ar, 1, -1.0),
        &poly::lag_poly(&p.seasonal_ar, s, -1.0),
    );
    let m = poly::polymul(
        &poly::lag_poly(&p.ma, 1, 1.0),
        &poly::lag_poly(&p.seasonal_ma, s, 1.0),
    );
    let mut rng = seed;
    let burn = 500;
    let (mut u, mut e) = (Vec::new(), Vec::new());
    for t in 0..n + burn {
        let et = p.sigma2.sqrt() * normal(&mut rng);
        e.push(et);
        let mut v = 0.0;
        for (i, ai) in a.iter().enumerate().skip(1) {
            if t >= i {
                v -= ai * u[t - i];
            }
        }
        for (j, cj) in m.iter().enumerate() {
            if t >= j {
                v += cj * e[t - j];
            }
        }
        u.push(v);
    }
    let mut y: Vec<f64> = u[burn..].to_vec();
    let [_, d, _] = c.order;
    let [_, big_d, _, _] = c.seasonal_order;
    for _ in 0..big_d {
        for t in s..y.len() {
            y[t] += y[t - s];
        }
    }
    for _ in 0..d {
        for t in 1..y.len() {
            y[t] += y[t - 1];
        }
    }
    y.iter().map(|v| v + level).collect()
}

/// The ARMA part's autocovariances `gamma_0..gamma_{n-1}` from its MA(inf)
/// weights, `gamma_k = sigma2 sum_j psi_j psi_{j+k}`, `psi` from `A psi = C`,
/// truncated where the weights have died out: the definition, sharing no
/// arithmetic with the filter or its Lyapunov start.
fn autocov(p: &SarimaParams, s: usize, n: usize) -> Vec<f64> {
    let a = poly::polymul(
        &poly::lag_poly(&p.ar, 1, -1.0),
        &poly::lag_poly(&p.seasonal_ar, s, -1.0),
    );
    let c = poly::polymul(
        &poly::lag_poly(&p.ma, 1, 1.0),
        &poly::lag_poly(&p.seasonal_ma, s, 1.0),
    );
    let terms = 40_000;
    let mut psi = vec![0.0; terms];
    for j in 0..terms {
        let mut v = c.get(j).copied().unwrap_or(0.0);
        for (i, ai) in a.iter().enumerate().skip(1) {
            if j >= i {
                v -= ai * psi[j - i];
            }
        }
        psi[j] = v;
    }
    assert!(
        psi[terms - 1].abs() < 1e-30,
        "the weights have not died out"
    );
    (0..n)
        .map(|k| p.sigma2 * oracle::dot((0..terms - k).map(|j| (psi[j], psi[j + k]))))
        .collect()
}

/// The Gaussian conditional expectation of `w_t` given the observed values
/// before it, `gamma_t' Gamma^{-1} w_obs`, solved by faer's LU: what the
/// Kalman filter from the stationary start computes, by its definition.
fn conditional_mean(gamma: &[f64], w: &[Option<f64>], t: usize) -> f64 {
    let obs: Vec<usize> = (0..t).filter(|&i| w[i].is_some()).collect();
    if obs.is_empty() {
        return 0.0;
    }
    let n = obs.len();
    let mut g = vec![0.0; n * n];
    for (a, &i) in obs.iter().enumerate() {
        for (b, &j) in obs.iter().enumerate() {
            g[a * n + b] = gamma[i.abs_diff(j)];
        }
    }
    let rhs: Vec<f64> = obs.iter().map(|&i| w[i].unwrap()).collect();
    let coef = oracle::solve(&g, &rhs);
    oracle::dot(obs.iter().zip(&coef).map(|(&i, &c)| (gamma[t - i], c)))
}

#[test]
fn the_step_down_test_is_the_unit_circle() {
    assert!(poly::stationary(&[]));
    assert!(poly::stationary(&[0.99]));
    assert!(poly::stationary(&[-0.99]));
    assert!(!poly::stationary(&[1.0]));
    assert!(!poly::stationary(&[-1.0]));
    assert!(!poly::stationary(&[f64::NAN]));
    // AR(2)'s triangle: phi_2 in (-1, 1), phi_1 + phi_2 < 1, phi_2 - phi_1 < 1.
    for (p1, p2, inside) in [
        (0.5, 0.3, true),
        (0.69, 0.3, true),
        (0.71, 0.3, false),
        (-0.69, 0.3, true),
        (-0.71, 0.3, false),
        (1.5, -0.6, true),
        (1.5, -0.4, false),
        (0.0, -0.99, true),
        (0.0, -1.0, false),
        (0.0, 0.99, true),
    ] {
        assert_eq!(poly::stationary(&[p1, p2]), inside, "{p1}, {p2}");
    }
    // A product of stationary factors is stationary: 1 - 1.2 z + 0.35 z^2
    // = (1 - 0.5 z)(1 - 0.7 z), and with a factor of 1.1 it is not.
    assert!(poly::stationary(&[1.2, -0.35]));
    assert!(!poly::stationary(&[1.6, -0.55]));
}

#[test]
fn the_lag_polynomials_multiply_out() {
    let a = poly::polymul(
        &poly::lag_poly(&[0.5], 1, -1.0),
        &poly::lag_poly(&[0.4], 4, -1.0),
    );
    assert_eq!(a, vec![1.0, -0.5, 0.0, 0.0, -0.4, 0.2]);
    let c = poly::polymul(
        &poly::lag_poly(&[0.3], 1, 1.0),
        &poly::lag_poly(&[-0.3], 4, 1.0),
    );
    let want = [1.0, 0.3, 0.0, 0.0, -0.3, 0.3 * -0.3];
    assert_eq!(c, want);
    // Harvey's form: r = max(5, 5 + 1) = 6 states.
    let m = Ssm::new(&[0.5], &[0.3], &[0.4], &[-0.3], 4, 2.0);
    assert_eq!(m.r(), 6);
    assert_eq!(m.t_col, vec![0.5, 0.0, 0.0, 0.4, -0.2, 0.0]);
    assert_eq!(m.r_vec, vec![1.0, 0.3, 0.0, 0.0, -0.3, -0.09]);
}

#[test]
fn the_differencing_is_seasonal_then_ordinary() {
    let mut work = Vec::new();
    let y = [3.0, 7.0, 4.0, 10.0, 6.0, 15.0];
    // (1 - B)(1 - B^4) y_5 = y5 - y4 - y1 + y0.
    let w = poly::difference(&y, 1, 1, 4, &mut work);
    assert_eq!(w, 15.0 - 6.0 - 7.0 + 3.0);
    // (1 - B)^2 y_2 = y2 - 2 y1 + y0.
    assert_eq!(
        poly::difference(&y[..3], 2, 0, 0, &mut work),
        4.0 - 14.0 + 3.0
    );
    assert!(poly::difference(&[1.0, f64::NAN, 2.0], 2, 0, 0, &mut work).is_nan());
    assert_eq!(poly::difference(&[5.0], 0, 0, 0, &mut work), 5.0);
}

/// To 2e-15 of the matrix's scale, and `P_00` to 2e-15 of `gamma_0`: the
/// doubling's sums are short and in a fixed order. statsmodels' start (the
/// bilinear transform and scipy's Lyapunov solver) misses both by more, at
/// `(1,0,1)(1,0,1,24)` by 1.6e-14 and 5.4e-14, which is the whole of the
/// filters' gap over that spec's first rows (`tests/test_sarima_filter.py`).
#[test]
fn the_stationary_start_solves_the_lyapunov_equation() {
    for (ar, ma, sar, sma, s) in [
        (vec![0.5], vec![0.3], vec![0.4], vec![-0.3], 4),
        (vec![0.9, -0.2], vec![], vec![0.95], vec![0.5], 12),
        (vec![0.6], vec![0.2], vec![0.7], vec![-0.4], 24),
        (vec![], vec![0.4], vec![], vec![-0.8], 24),
        (vec![], vec![], vec![], vec![], 0),
    ] {
        let m = Ssm::new(&ar, &ma, &sar, &sma, s, 1.3);
        let r = m.r();
        let p = m.stationary_cov().unwrap();
        let mut tpt = p.clone();
        let mut work = Vec::new();
        m.predict_cov(&mut tpt, &mut work);
        let scale = p.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        for i in 0..r {
            for j in 0..r {
                let off = (tpt[i * r + j] - p[i * r + j]).abs();
                assert!(off <= 2e-15 * scale, "{ar:?} {sar:?} ({i},{j}): {off:e}");
            }
        }
        // P_00 is the ARMA part's variance, gamma_0, by its definition.
        let prm = SarimaParams {
            exog: vec![],
            ar,
            ma,
            seasonal_ar: sar,
            seasonal_ma: sma,
            sigma2: 1.3,
        };
        let g0 = autocov(&prm, s.max(1), 1)[0];
        assert!((p[0] - g0).abs() <= 2e-15 * g0, "{} against {g0}", p[0]);
    }
    // A unit root has no stationary start, and is refused before one is
    // looked for.
    assert!(
        Ssm::new(&[1.0], &[], &[], &[], 0, 1.0)
            .stationary_cov()
            .is_none()
    );
}

/// The filter's `ŵ_t` is the Gaussian conditional expectation of `w_t` on
/// every observed value before it, with slots unobserved (a null target,
/// a weight of 0, a skipped slot) left out of the conditioning set: the
/// definition, by faer's LU over the autocovariances.
#[test]
fn the_filter_is_the_conditional_expectation_through_missing_slots() {
    let prm = arma_params();
    let c = cfg([1, 0, 1], [1, 0, 1, 4], prm.clone());
    let n = 60;
    let ys = simulate(&c, n, 7, 0.0);
    let gamma = autocov(&prm, 4, n);
    let missing = [3, 4, 11, 20, 21, 22, 23, 24, 40];
    let w: Vec<Option<f64>> = (0..n)
        .map(|t| (!missing.contains(&t)).then_some(ys[t]))
        .collect();
    let mut model = Sarima::new(c).unwrap();
    for t in 0..n {
        let want = conditional_mean(&gamma, &w, t);
        let (w_hat, y_hat) = model.filter_row(&[], &[w[t]], 0)[0];
        assert_eq!(w_hat, y_hat, "no differencing: the two predictions are one");
        let tol = 1e-12 * gamma[0].sqrt();
        assert!(
            (w_hat - want).abs() <= tol,
            "row {t}: {w_hat} against {want}"
        );
    }
}

/// With differencing, `ŷ_t = ŵ_t + y_t - w_t`: the differenced value's
/// prediction put back on the series' scale with the slots before it.
#[test]
fn the_differenced_prediction_is_put_back_on_the_series() {
    let mut prm = arma_params();
    prm.sigma2 = 0.5;
    let c = cfg([1, 1, 1], [1, 1, 1, 4], prm);
    let n = 80;
    let ys = simulate(&c, n, 3, 100.0);
    let mut model = Sarima::new(c.clone()).unwrap();
    let mut plain = Sarima::new(cfg([1, 0, 1], [1, 0, 1, 4], c.params[0].clone())).unwrap();
    let mut work = Vec::new();
    for t in 0..n {
        let (w_hat, y_hat) = model.filter_row(&[], &[Some(ys[t])], 0)[0];
        if t < 5 {
            assert!(w_hat.is_nan() && y_hat.is_nan(), "row {t}");
            continue;
        }
        let w = poly::difference(&ys[t - 5..=t], 1, 1, 4, &mut work);
        let (want, _) = plain.filter_row(&[], &[Some(w)], 0)[0];
        assert_eq!(w_hat.to_bits(), want.to_bits(), "row {t}");
        let back = w_hat + (ys[t] - w);
        assert!((y_hat - back).abs() <= 1e-12 * ys[t].abs(), "row {t}");
    }
}

/// Every row's prediction is formed before its value is read (hard rule 2):
/// changing row 40 moves no prediction up to and including it.
#[test]
fn predictions_are_out_of_sample() {
    let c = cfg([1, 1, 1], [1, 1, 1, 4], arma_params());
    let ys = simulate(&c, 80, 11, 5.0);
    let run = |ys: &[f64]| -> Vec<f64> {
        let mut m = Sarima::new(c.clone()).unwrap();
        ys.iter()
            .map(|&y| m.step(&[], &[Some(y)], 1.0, 1.0).pred[0])
            .collect()
    };
    let a = run(&ys);
    let mut moved = ys.clone();
    moved[40] += 100.0;
    let b = run(&moved);
    for t in 0..=40 {
        assert_eq!(a[t].to_bits(), b[t].to_bits(), "row {t}");
    }
    assert_ne!(a[41], b[41]);
}

fn run_rows(m: &mut Sarima, rows: &[(Option<f64>, f64, f64)]) -> Vec<Step> {
    rows.iter()
        .map(|&(y, d, w)| m.step(&[], &[y], d, w))
        .collect()
}

/// A row of weight 0, a null target and a slot the clock skipped are one
/// thing: a slot not observed. The rows after each predict alike, and the
/// state is the same, but for the weight.
#[test]
fn a_zero_weight_a_null_and_a_skipped_slot_keep_the_slot_alike() {
    let mut airline = arma_params();
    airline.ar.clear();
    airline.seasonal_ar.clear();
    let c = cfg([0, 1, 1], [0, 1, 1, 4], airline);
    let ys = simulate(&c, 50, 5, 20.0);
    let hole = 23;
    let rows = |how: u8| -> Vec<(Option<f64>, f64, f64)> {
        let mut out = Vec::new();
        for (t, &y) in ys.iter().enumerate() {
            match (t == hole, how) {
                (true, 0) => out.push((Some(y), 1.0, 0.0)),
                (true, 1) => out.push((None, 1.0, 1.0)),
                (true, _) => {}
                (false, 2) if t == hole + 1 => out.push((Some(y), 2.0, 1.0)),
                _ => out.push((Some(y), 1.0, 1.0)),
            }
        }
        out
    };
    let mut ms: Vec<Sarima> = (0..3).map(|_| Sarima::new(c.clone()).unwrap()).collect();
    let outs: Vec<Vec<Step>> = ms
        .iter_mut()
        .enumerate()
        .map(|(i, m)| run_rows(m, &rows(i as u8)))
        .collect();
    // The skipped slot has no row: line the third run up with the others.
    let third: Vec<&Step> = outs[2][..hole]
        .iter()
        .chain(std::iter::once(&outs[0][hole]))
        .chain(&outs[2][hole..])
        .collect();
    for t in 0..ys.len() {
        let a = outs[0][t].pred[0];
        assert_eq!(a.to_bits(), outs[1][t].pred[0].to_bits(), "row {t}");
        assert_eq!(a.to_bits(), third[t].pred[0].to_bits(), "row {t}");
    }
    assert!(outs[0][hole + 1].pred[0].is_finite());
    assert_eq!(ms[0].series[0].a, ms[1].series[0].a);
    assert_eq!(ms[0].series[0].p, ms[2].series[0].p);
    assert_eq!(ms[0].series[0].ys, ms[2].series[0].ys);
    // The weight counts what was weighed: the zero weight adds nothing,
    // the null target's row adds to n_eff and not to the target's own.
    assert_eq!(ms[0].n_eff(), 49.0);
    assert_eq!(ms[1].n_eff(), 50.0);
    assert_eq!(ms[1].series[0].weight, 49.0);
}

/// A first row of weight 0 divides by nothing and leaves every weight at 0.
#[test]
fn a_zero_weight_first_row_is_legal() {
    let c = cfg([1, 0, 1], [1, 0, 1, 4], arma_params());
    let mut m = Sarima::new(c).unwrap();
    let first = m.step(&[], &[Some(3.0)], 0.0, 0.0);
    assert_eq!(first.n_eff, 0.0);
    assert_eq!(first.pred[0], 0.0, "the stationary mean, with no gate");
    assert_eq!(m.n_eff(), 0.0);
    assert_eq!(m.series[0].weight, 0.0);
    assert!(m.series[0].a.iter().all(|v| *v == 0.0));
    let next = m.step(&[], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(next.pred[0], 0.0, "nothing was learned from the first row");
    assert!(m.series[0].p.iter().all(|v| v.is_finite()));
}

/// `min_weight` withholds a target's prediction until the weight of the rows
/// that observed it reaches it, read before the row (hard rule 8).
#[test]
fn min_weight_reads_the_targets_own_weight() {
    let c = SarimaCfg {
        n_targets: 2,
        params: vec![arma_params(), arma_params()],
        min_weight: 3.0,
        ..cfg([1, 0, 1], [1, 0, 1, 4], arma_params())
    };
    let mut m = Sarima::new(c).unwrap();
    let rows = [
        [Some(1.0), Some(1.0)],
        [Some(1.0), None],
        [Some(1.0), Some(1.0)],
        [Some(1.0), Some(1.0)],
        [Some(1.0), Some(1.0)],
    ];
    let preds: Vec<[bool; 2]> = rows
        .iter()
        .map(|y| {
            let s = m.step(&[], y, 1.0, 1.0);
            [s.pred[0].is_finite(), s.pred[1].is_finite()]
        })
        .collect();
    assert_eq!(
        preds,
        vec![
            [false, false],
            [false, false],
            [false, false],
            [true, false],
            [true, true]
        ]
    );
    let mut w = Vec::new();
    assert!(m.target_n_eff_into(&mut w));
    assert_eq!(w, vec![5.0, 4.0]);
}

/// A break starts each series again: after `clear_lags` the model predicts
/// what a new one fed the same rows does, and keeps its weights.
#[test]
fn a_break_starts_each_series_again() {
    let c = cfg([1, 1, 1], [1, 1, 1, 4], arma_params());
    let ys = simulate(&c, 60, 13, -4.0);
    let mut m = Sarima::new(c.clone()).unwrap();
    for &y in &ys[..30] {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    m.clear_lags();
    let mut fresh = Sarima::new(c.clone()).unwrap();
    for &y in &ys[30..] {
        let a = m.step(&[], &[Some(y)], 1.0, 1.0);
        let b = fresh.step(&[], &[Some(y)], 1.0, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
        assert_eq!(a.n_eff, b.n_eff + 30.0);
    }
    // A step past MAX_GAP_SLOTS is a break too.
    let mut gap = Sarima::new(c.clone()).unwrap();
    let mut fresh = Sarima::new(c).unwrap();
    for &y in &ys[..30] {
        gap.step(&[], &[Some(y)], 1.0, 1.0);
    }
    let far = (MAX_GAP_SLOTS + 2) as f64;
    for (t, &y) in ys[30..].iter().enumerate() {
        let a = gap.step(&[], &[Some(y)], if t == 0 { far } else { 1.0 }, 1.0);
        let b = fresh.step(&[], &[Some(y)], 1.0, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
    }
}

/// A gap of `n` slots is `n - 1` slots not observed, predicted through;
/// the clock's step is read in `spacing` units, rounded.
#[test]
fn a_clock_step_of_several_slots_skips_them() {
    let c = SarimaCfg {
        spacing: 0.25,
        ..cfg([1, 0, 1], [1, 0, 1, 4], arma_params())
    };
    let ys = simulate(&c, 40, 17, 0.0);
    let mut gap = Sarima::new(c.clone()).unwrap();
    let mut nulls = Sarima::new(c).unwrap();
    for (t, &y) in ys.iter().enumerate() {
        if t == 20 {
            // Three slots, a little off the grid.
            let a = gap.step(&[], &[Some(y)], 0.75 + 1e-9, 1.0);
            nulls.step(&[], &[None], 0.25, 1.0);
            nulls.step(&[], &[None], 0.25, 1.0);
            let b = nulls.step(&[], &[Some(y)], 0.25, 1.0);
            assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits());
            continue;
        }
        if (21..23).contains(&t) {
            continue;
        }
        let a = gap.step(&[], &[Some(y)], 0.25, 1.0);
        let b = nulls.step(&[], &[Some(y)], 0.25, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits(), "row {t}");
    }
}

/// `predict` is the step's report without the step, across a skipped slot
/// and under a floor.
#[test]
fn predict_is_the_step_without_the_step() {
    let c = SarimaCfg {
        min_weight: 10.0,
        ..cfg([1, 1, 1], [1, 1, 1, 4], arma_params())
    };
    let ys = simulate(&c, 40, 19, 2.0);
    let mut m = Sarima::new(c).unwrap();
    for (t, &y) in ys.iter().enumerate() {
        let d = if t % 7 == 6 { 3.0 } else { 1.0 };
        let before = m.clone();
        let p = m.predict(&[], d);
        let floor0 = m.predict_with_floor(&[], &[Some(y)], d, 0.0);
        assert_eq!(m, before, "a read leaves the state as it was");
        let s = m.step(&[], &[Some(y)], d, 1.0);
        assert_eq!(p.pred[0].to_bits(), s.pred[0].to_bits(), "row {t}");
        assert_eq!(p.n_eff, s.n_eff);
        if t >= 5 {
            assert!(floor0.pred[0].is_finite(), "row {t}");
        }
    }
}

/// With regressors, `d_t = xw_t' beta` enters the differenced prediction;
/// with every coefficient 0 the model is the one without them.
#[test]
fn regressors_enter_through_their_differences() {
    let mut prm = arma_params();
    prm.exog = vec![0.0, 0.0];
    let c = cfg([1, 1, 1], [1, 0, 1, 4], prm.clone());
    let ys = simulate(&c, 30, 23, 1.0);
    let mut zero = Sarima::new(c.clone()).unwrap();
    let mut plain = Sarima::new(cfg([1, 1, 1], [1, 0, 1, 4], arma_params())).unwrap();
    prm.exog = vec![0.7, -1.5];
    let mut with = Sarima::new(cfg([1, 1, 1], [1, 0, 1, 4], prm)).unwrap();
    let mut rng = 5;
    let mut prev = [0.0, 0.0];
    for (t, &y) in ys.iter().enumerate() {
        let x = [normal(&mut rng), normal(&mut rng)];
        let a = zero.step(&x, &[Some(y)], 1.0, 1.0).pred[0];
        let b = plain.step(&[], &[Some(y)], 1.0, 1.0).pred[0];
        assert_eq!(a.to_bits(), b.to_bits(), "row {t}");
        // The first prediction: a stationary mean of 0 plus the regressors'
        // differenced part.
        let (w_hat, _) = with.filter_row(&x, &[Some(y)], 0)[0];
        if t == 1 {
            let want = (x[0] - prev[0]) * 0.7 + (x[1] - prev[1]) * -1.5;
            assert_eq!(w_hat, 0.0 + want);
        }
        prev = x;
    }
}

/// Multiplying the series and `sigma2`'s square root by 37 multiplies
/// every prediction by 37.
#[test]
fn the_predictions_scale_with_the_series() {
    let c = cfg([1, 1, 1], [1, 1, 1, 4], arma_params());
    let ys = simulate(&c, 80, 29, 3.0);
    let mut a = Sarima::new(c.clone()).unwrap();
    let mut big = c.clone();
    big.params[0].sigma2 *= 37.0 * 37.0;
    let mut b = Sarima::new(big).unwrap();
    for (t, &y) in ys.iter().enumerate() {
        let pa = a.step(&[], &[Some(y)], 1.0, 1.0).pred[0];
        let pb = b.step(&[], &[Some(37.0 * y)], 1.0, 1.0).pred[0];
        if pa.is_nan() {
            assert!(pb.is_nan());
            continue;
        }
        assert!(
            (pb - 37.0 * pa).abs() <= 1e-11 * (37.0 * pa).abs().max(37.0),
            "row {t}"
        );
    }
}

/// A state saved at any row continues to the bit; the bytes it writes stay
/// under `state_bytes`, which is not loose.
#[test]
fn the_state_continues_to_the_bit_and_is_bounded() {
    let c = cfg([1, 1, 1], [1, 1, 1, 4], arma_params());
    let ys = simulate(&c, 40, 31, 0.0);
    let bound = Sarima::state_bytes(&c);
    let mut m = Sarima::new(c.clone()).unwrap();
    let fresh = rmp_serde::to_vec(&m.state()).unwrap().len() as u64;
    assert!(fresh <= bound, "{fresh} > {bound}");
    for (t, &yt) in ys.iter().enumerate() {
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let back: State = rmp_serde::from_slice(&bytes).unwrap();
        let mut twin = Sarima::restore(&back).unwrap();
        // Byte for byte: a ring may hold a NaN, which equals nothing.
        assert_eq!(rmp_serde::to_vec(&twin.state()).unwrap(), bytes);
        let y = (t % 9 != 4).then_some(yt);
        let a = m.step(&[], &[y], 1.0, 1.0);
        let b = twin.step(&[], &[y], 1.0, 1.0);
        assert_eq!(a.pred[0].to_bits(), b.pred[0].to_bits(), "row {t}");
    }
    let after = rmp_serde::to_vec(&m.state()).unwrap().len() as u64;
    assert!(after <= bound, "{after} > {bound}");
    assert!(after * 4 > bound, "the bound is loose: {after} of {bound}");
    // Past the cap, refused when built; exactly at it, taken.
    let tight = SarimaCfg {
        max_state_bytes: Some(bound - 1),
        ..c.clone()
    };
    assert!(Sarima::new(tight).unwrap_err().contains("max_state_bytes"));
    let at = SarimaCfg {
        max_state_bytes: Some(bound),
        ..c
    };
    assert!(Sarima::new(at).is_ok());
}

#[test]
fn settings_it_cannot_filter_are_refused() {
    let base = cfg([1, 0, 1], [1, 0, 1, 4], arma_params());
    let refused = |c: SarimaCfg, says: &str| {
        let e = Sarima::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    let mut p = arma_params();
    p.ar = vec![1.0];
    refused(
        SarimaCfg {
            params: vec![p],
            ..base.clone()
        },
        "ar coefficients [1.0] are not stationary",
    );
    let mut p = arma_params();
    p.seasonal_ar = vec![-1.2];
    refused(
        SarimaCfg {
            params: vec![p],
            ..base.clone()
        },
        "seasonal ar coefficients [-1.2] are not stationary",
    );
    let mut p = arma_params();
    p.ma.push(0.1);
    refused(
        SarimaCfg {
            params: vec![p],
            ..base.clone()
        },
        "must hold 0 feature, 1 ar, 1 ma",
    );
    let mut p = arma_params();
    p.sigma2 = 0.0;
    refused(
        SarimaCfg {
            params: vec![p],
            ..base.clone()
        },
        "sigma2 must be above 0",
    );
    let mut p = arma_params();
    p.ma[0] = f64::INFINITY;
    refused(
        SarimaCfg {
            params: vec![p],
            ..base.clone()
        },
        "must be finite",
    );
    refused(
        SarimaCfg {
            seasonal_order: [1, 0, 1, 1],
            ..base.clone()
        },
        "season s of at least 2",
    );
    refused(
        SarimaCfg {
            spacing: 0.0,
            ..base.clone()
        },
        "spacing must be finite",
    );
    refused(
        SarimaCfg {
            params: vec![],
            ..base.clone()
        },
        "one set of parameters a target",
    );
    refused(
        SarimaCfg {
            min_weight: -1.0,
            ..base.clone()
        },
        "min_weight",
    );
    // No season read without a seasonal order, whatever s says.
    assert!(
        Sarima::new(SarimaCfg {
            seasonal_order: [0, 0, 0, 0],
            params: vec![SarimaParams {
                seasonal_ar: vec![],
                seasonal_ma: vec![],
                ..arma_params()
            }],
            ..base
        })
        .is_ok()
    );
}

#[test]
fn the_flat_vector_reads_in_statsmodels_order() {
    let p = SarimaParams::from_flat(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, [1, 1, 1], [1, 1, 1, 12])
        .unwrap();
    assert_eq!(p.exog, vec![1.0]);
    assert_eq!(p.ar, vec![2.0]);
    assert_eq!(p.ma, vec![3.0]);
    assert_eq!(p.seasonal_ar, vec![4.0]);
    assert_eq!(p.seasonal_ma, vec![5.0]);
    assert_eq!(p.sigma2, 6.0);
    assert!(SarimaParams::from_flat(&[1.0], 1, [1, 0, 0], [0, 0, 0, 0]).is_none());
}
