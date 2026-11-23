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

/// A stationary AR(2) with an exogenous regressor, far from zero:
/// `y = 4 + 0.55 y₋₁ − 0.25 y₋₂ + 0.8 x + e`, its mean about 5.7, after 200
/// rows of burn-in. Returns `(x, y)`.
fn series(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut s = seed;
    let (mut y1, mut y2) = (5.0, 5.0);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..n + 200 {
        let x = 1.0 + normal(&mut s);
        let y = 4.0 + 0.55 * y1 - 0.25 * y2 + 0.8 * x + normal(&mut s);
        (y2, y1) = (y1, y);
        if t >= 200 {
            xs.push(x);
            ys.push(y);
        }
    }
    (xs, ys)
}

fn cfg(lags: Vec<usize>, n_features: usize, decay: Decay) -> ArCfg {
    ArCfg {
        n_targets: 1,
        n_features,
        lags,
        fit_intercept: true,
        decay,
        ridge: 0.0,
        min_weight: 0.0,
        max_state_bytes: None,
        select: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// The weighted least squares fit of `y_t` on `(1, y_{t−ℓ}…, x_t)` over
/// the rows `t ≥ max(lags)` of the (already gap-free) series, each row
/// weighted `w_t`, by the normal equations through faer's partial-pivot LU
/// (`crate::oracle`), which shares no arithmetic with the model's solve.
fn wls(lags: &[usize], xs: &[Vec<f64>], ys: &[f64], ws: &[f64]) -> Vec<f64> {
    let depth = *lags.iter().max().unwrap();
    let k = 1 + lags.len() + xs[0].len();
    let rows: Vec<(Vec<f64>, f64, f64)> = (depth..ys.len())
        .map(|t| {
            let mut z = vec![1.0];
            z.extend(lags.iter().map(|&l| ys[t - l]));
            z.extend_from_slice(&xs[t]);
            (z, ys[t], ws[t])
        })
        .collect();
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = oracle::sum(rows.iter().map(|(z, _, w)| w * z[i] * z[j]));
        }
        b[i] = oracle::sum(rows.iter().map(|(z, y, w)| w * z[i] * y));
    }
    oracle::solve(&a, &b)
}

fn rel_gap(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

/// With no decay the fit is ordinary least squares on the lag design,
/// statsmodels' `AutoReg(trend="c", exog=x)` on the same rows: the
/// coefficients against the normal equations solved by LU, at an AR(2) and
/// at lags `[1, 3]`, and it recovers the generating coefficients.
#[test]
fn no_decay_is_least_squares_on_the_lag_design() {
    let (x, y) = series(2000, 7);
    for lags in [vec![1, 2], vec![1, 3]] {
        let mut m = Ar::new(cfg(lags.clone(), 1, Decay::Halflife(f64::INFINITY))).unwrap();
        for t in 0..y.len() {
            m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        }
        let xs: Vec<Vec<f64>> = x.iter().map(|v| vec![*v]).collect();
        let want = wls(&lags, &xs, &y, &vec![1.0; y.len()]);
        let got = &m.coefficients().unwrap()[0];
        assert!(rel_gap(got, &want) < 1e-10, "{lags:?}: {got:?} vs {want:?}");
    }
    // And on a long stream it recovers the coefficients that made it.
    let (x, y) = series(40_000, 8);
    let mut m = Ar::new(cfg(vec![1, 2], 1, Decay::Halflife(f64::INFINITY))).unwrap();
    for t in 0..y.len() {
        m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
    }
    let got = &m.coefficients().unwrap()[0];
    for (g, (truth, tol)) in got
        .iter()
        .zip([(4.0, 0.15), (0.55, 0.02), (-0.25, 0.02), (0.8, 0.02)])
    {
        assert!((g - truth).abs() < tol, "{got:?}");
    }
}

/// Under decay, row `t` weighs `w_t λ^{age}`, its age counted on the clock,
/// every row's step included: weighted least squares with those weights.
/// Uneven steps and weights, so neither cancels.
#[test]
fn decay_is_weighted_least_squares_on_the_clock() {
    let (x, y) = series(600, 11);
    let h = 80.0;
    let mut s = 3;
    let steps: Vec<f64> = (0..y.len()).map(|_| 0.5 + lcg(&mut s)).collect();
    let ws: Vec<f64> = (0..y.len()).map(|_| 0.5 + lcg(&mut s)).collect();
    let mut m = Ar::new(cfg(vec![1, 2], 1, Decay::Halflife(h))).unwrap();
    for t in 0..y.len() {
        m.step(&[x[t]], &[Some(y[t])], steps[t], ws[t]);
    }
    let total: f64 = steps.iter().skip(1).sum();
    let mut clock = 0.0;
    let decayed: Vec<f64> = (0..y.len())
        .map(|t| {
            if t > 0 {
                clock += steps[t];
            }
            ws[t] * (-(total - clock) / h).exp2()
        })
        .collect();
    let xs: Vec<Vec<f64>> = x.iter().map(|v| vec![*v]).collect();
    let want = wls(&[1, 2], &xs, &y, &decayed);
    let got = &m.coefficients().unwrap()[0];
    assert!(rel_gap(got, &want) < 1e-9, "{got:?} vs {want:?}");
}

/// Hard rule 2: the prediction for row `t` is the fit on the rows before it
/// applied to row `t`'s lags and features; and nothing is predicted until
/// the ring holds every lag and a fit exists.
#[test]
fn a_prediction_is_the_fit_before_the_row() {
    let (x, y) = series(300, 5);
    let mut m = Ar::new(cfg(vec![1, 2], 1, Decay::Halflife(f64::INFINITY))).unwrap();
    for t in 0..y.len() {
        let before = m.coefficients();
        let step = m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        match before {
            Some(c) if t >= 2 && c[0].iter().all(|v| v.is_finite()) => {
                let c = &c[0];
                let want = c[0] + c[1] * y[t - 1] + c[2] * y[t - 2] + c[3] * x[t];
                assert!((step.pred[0] - want).abs() <= 1e-12 * want.abs(), "row {t}");
            }
            _ => assert!(step.pred[0].is_nan(), "row {t}: {:?}", step.pred),
        }
    }
    // Rows 0 and 1 fill the ring; row 2 is the first learned, with no fit
    // before it.
    let mut m = Ar::new(cfg(vec![1, 2], 1, Decay::Halflife(f64::INFINITY))).unwrap();
    for t in 0..3 {
        assert!(m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0).pred[0].is_nan());
    }
}

/// A row of weight 0 learns nothing and moves no ring: with no decay the
/// stream with such rows -- wild values, the first two rows among them --
/// ends in the state of the stream without them, coefficients and rings to
/// the bit. A missing target likewise is not a lag (statsmodels'
/// `missing="drop"`); its row only adds to the shared weight.
#[test]
fn a_zero_weight_row_or_a_missing_target_is_not_a_lag() {
    let (x, y) = series(400, 13);
    let inf = Decay::Halflife(f64::INFINITY);
    let mut clean = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    let mut zeros = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    let mut gaps = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    for (j, v) in [1e9, -3.0].into_iter().enumerate() {
        zeros.step(&[v], &[Some(v * (j as f64 + 1.0))], 1.0, 0.0);
        gaps.step(&[v], &[None], 1.0, 1.0);
    }
    for t in 0..y.len() {
        clean.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        zeros.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        gaps.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        if t % 7 == 3 {
            zeros.step(&[1e6], &[Some(-1e6)], 1.0, 0.0);
            gaps.step(&[1e6], &[None], 1.0, 1.0);
        }
    }
    assert_eq!(zeros.ring(0), clean.ring(0));
    assert_eq!(gaps.ring(0), clean.ring(0));
    let want = clean.coefficients().unwrap();
    assert_eq!(zeros.coefficients().unwrap(), want);
    assert_eq!(gaps.coefficients().unwrap(), want);
    assert_eq!(zeros.n_eff(), clean.n_eff());
    assert!(gaps.n_eff() > clean.n_eff(), "a row with weight counts");
    let mut own = Vec::new();
    assert!(gaps.target_n_eff_into(&mut own));
    let mut clean_own = Vec::new();
    clean.target_n_eff_into(&mut clean_own);
    assert_eq!(
        own, clean_own,
        "the target's own weight is its learned rows'"
    );
}

/// A stream of nothing but zero-weight rows, the first of them included,
/// leaves no NaN in the state and predicts nothing (hard rule 9).
#[test]
fn zero_weight_rows_from_the_start_leave_no_nan() {
    let mut m = Ar::new(cfg(vec![1], 1, Decay::Halflife(5.0))).unwrap();
    for v in [f64::NAN, 1.0, 2.0, 3.0] {
        let s = m.step(&[1.0], &[Some(v)], 1.0, 0.0);
        assert!(s.pred[0].is_nan());
        assert_eq!(s.n_eff, 0.0);
    }
    assert!(m.ring(0).is_empty());
    let json = serde_json::to_string(&m).unwrap();
    assert!(!json.contains("nan"), "{json}");
}

/// A break empties the ring and nothing else: the fit and the weight stay,
/// the next `max(lags)` rows refill the ring unlearned and unpredicted, and
/// from then on the model is the one that saw the series with the break's
/// rows' lags left out.
#[test]
fn a_break_empties_the_ring_and_keeps_the_fit() {
    let (x, y) = series(300, 17);
    let inf = Decay::Halflife(f64::INFINITY);
    let mut m = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    for t in 0..150 {
        m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
    }
    let (coef, n_eff) = (m.coefficients(), m.n_eff());
    m.clear_lags();
    assert!(m.ring(0).is_empty());
    assert_eq!((m.coefficients(), m.n_eff()), (coef.clone(), n_eff));
    for t in 150..152 {
        let s = m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        assert!(s.pred[0].is_nan(), "row {t} has no lags");
        assert_eq!(m.coefficients(), coef, "row {t} is not learned");
    }
    let s = m.step(&[x[152]], &[Some(y[152])], 1.0, 1.0);
    let c = &coef.unwrap()[0];
    let want = c[0] + c[1] * y[151] + c[2] * y[150] + c[3] * x[152];
    assert!((s.pred[0] - want).abs() <= 1e-12 * want.abs());
    // The fit after the break is least squares on both pieces' own rows.
    for t in 153..300 {
        m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
    }
    let xs: Vec<Vec<f64>> = x.iter().map(|v| vec![*v]).collect();
    let (a, b) = (
        wls_rows(&[1, 2], &xs, &y, 0..150),
        wls_rows(&[1, 2], &xs, &y, 150..300),
    );
    let want = oracle::solve(
        &a.0.iter().zip(&b.0).map(|(p, q)| p + q).collect::<Vec<_>>(),
        &a.1.iter().zip(&b.1).map(|(p, q)| p + q).collect::<Vec<_>>(),
    );
    assert!(rel_gap(&m.coefficients().unwrap()[0], &want) < 1e-10);
}

/// The normal equations' two sides over the piece `range` of a series, its
/// first `max(lags)` rows only lags.
fn wls_rows(
    lags: &[usize],
    xs: &[Vec<f64>],
    ys: &[f64],
    range: std::ops::Range<usize>,
) -> (Vec<f64>, Vec<f64>) {
    let depth = *lags.iter().max().unwrap();
    let k = 1 + lags.len() + xs[0].len();
    let zs: Vec<(Vec<f64>, f64)> = (range.start + depth..range.end)
        .map(|t| {
            let mut z = vec![1.0];
            z.extend(lags.iter().map(|&l| ys[t - l]));
            z.extend_from_slice(&xs[t]);
            (z, ys[t])
        })
        .collect();
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = oracle::sum(zs.iter().map(|(z, _)| z[i] * z[j]));
        }
        b[i] = oracle::sum(zs.iter().map(|(z, y)| z[i] * y));
    }
    (a, b)
}

/// Two targets are two series, each with its own lags and fit: each equals
/// a model of it alone, to the bit, with the other target's gaps elsewhere.
#[test]
fn each_target_is_its_own_series() {
    let (x, y) = series(300, 19);
    let (_, y2) = series(300, 23);
    let inf = Decay::Halflife(5.0);
    let mut both = Ar::new(ArCfg {
        n_targets: 2,
        ..cfg(vec![1, 2], 1, inf)
    })
    .unwrap();
    let mut one = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    let mut two = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    for t in 0..y.len() {
        let t2 = (t % 5 != 0).then_some(y2[t]);
        let s = both.step(&[x[t]], &[Some(y[t]), t2], 1.0, 1.0);
        let a = one.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        let b = two.step(&[x[t]], &[t2], 1.0, 1.0);
        assert_eq!(s.pred[0].to_bits(), a.pred[0].to_bits());
        assert_eq!(s.pred[1].to_bits(), b.pred[0].to_bits());
        assert_eq!(s.n_eff, a.n_eff);
    }
    let c = both.coefficients().unwrap();
    assert_eq!(c[0], one.coefficients().unwrap()[0]);
    assert_eq!(c[1], two.coefficients().unwrap()[0]);
}

/// Saved at every depth of the ring, from empty to full and past, a model
/// continues to the bit; the ring's depth is the settings', not the loaded
/// container's.
#[test]
fn a_saved_state_continues_to_the_bit_at_every_depth() {
    let (x, y) = series(60, 29);
    for cut in 0..6 {
        let mut a = Ar::new(cfg(vec![1, 3], 1, Decay::Halflife(9.0))).unwrap();
        for t in 0..cut {
            a.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec(&a.state()).unwrap();
        let mut b = Ar::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for t in cut..y.len() {
            let sa = a.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
            let sb = b.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
            assert_eq!(
                sa.pred[0].to_bits(),
                sb.pred[0].to_bits(),
                "cut {cut}, row {t}"
            );
            assert!(b.ring(0).len() <= 3);
        }
        assert_eq!(
            rmp_serde::to_vec(&a.state()).unwrap(),
            rmp_serde::to_vec(&b.state()).unwrap()
        );
    }
}

/// `state_bytes` bounds what the state writes, fresh and after a stream,
/// at several shapes; and `max_state_bytes` refuses a state past it and
/// takes one exactly at it.
#[test]
fn the_state_is_bounded_by_its_settings_and_capped() {
    let (x, y) = series(200, 31);
    for (lags, nf, m) in [
        (vec![1], 1, 1),
        (vec![1, 2, 3, 12], 1, 2),
        ((1..=20).collect(), 1, 1),
    ] {
        let c = ArCfg {
            n_targets: m,
            ..cfg(lags, nf, Decay::Halflife(50.0))
        };
        let mut model = Ar::new(c.clone()).unwrap();
        let bound = Ar::state_bytes(&c);
        let fresh = rmp_serde::to_vec(&model.state()).unwrap().len() as u64;
        assert!(fresh <= bound, "{fresh} > {bound}");
        for t in 0..y.len() {
            let ys = vec![Some(y[t]); m];
            model.step(&[x[t]], &ys, 1.0, 1.0);
        }
        let after = rmp_serde::to_vec(&model.state()).unwrap().len() as u64;
        assert!(after <= bound, "{after} > {bound}");
        assert!(
            after * 4 > bound,
            "the bound is not loose: {after} of {bound}"
        );
    }
    let c = cfg((1..=20).collect(), 1, Decay::Halflife(50.0));
    let bytes = Ar::state_bytes(&c);
    assert!(
        Ar::new(ArCfg {
            max_state_bytes: Some(bytes),
            select: None,
            ..c.clone()
        })
        .is_ok()
    );
    let err = Ar::new(ArCfg {
        max_state_bytes: Some(bytes - 1),
        select: None,
        ..c.clone()
    })
    .unwrap_err();
    assert!(err.contains("past max_state_bytes"), "{err}");
    // A ring of a million lags is past the default cap.
    let deep = cfg(vec![1, 1 << 20], 0, Decay::Halflife(50.0));
    assert!(Ar::new(deep).unwrap_err().contains("max_state_bytes"));
}

/// The settings it refuses, each by name.
#[test]
fn it_refuses_its_settings_by_name() {
    let base = cfg(vec![1, 2], 1, Decay::Halflife(10.0));
    let err = |c: ArCfg| Ar::new(c).unwrap_err();
    assert!(
        err(ArCfg {
            lags: vec![],
            ..base.clone()
        })
        .contains("lags must hold at least one")
    );
    assert!(
        err(ArCfg {
            lags: vec![0, 1],
            ..base.clone()
        })
        .contains("at least 1")
    );
    assert!(
        err(ArCfg {
            lags: vec![2, 1],
            ..base.clone()
        })
        .contains("strictly increasing")
    );
    assert!(
        err(ArCfg {
            lags: vec![1, (1 << 20) + 1],
            ..base.clone()
        })
        .contains("at most")
    );
    assert!(
        err(ArCfg {
            ridge: -1.0,
            ..base.clone()
        })
        .contains("ridge")
    );
    assert!(
        err(ArCfg {
            ridge: f64::NAN,
            ..base.clone()
        })
        .contains("ridge")
    );
    assert!(
        err(ArCfg {
            decay: Decay::Halflife(0.0),
            ..base.clone()
        })
        .contains("half_life")
    );
    assert!(
        err(ArCfg {
            max_state_bytes: Some(0),
            select: None,
            ..base.clone()
        })
        .contains("max_state_bytes")
    );
    assert!(
        err(ArCfg {
            n_targets: 0,
            ..base.clone()
        })
        .contains("at least one target")
    );
    assert!(
        Ar::new(ArCfg {
            n_features: 0,
            ..base.clone()
        })
        .is_ok(),
        "a pure AR"
    );
    assert!(
        Ar::new(ArCfg {
            ridge: 0.0,
            ..base.clone()
        })
        .is_ok()
    );
    assert!(
        Ar::new(ArCfg {
            lags: vec![1 << 20],
            max_state_bytes: Some(64 << 20),
            select: None,
            ..base
        })
        .is_ok()
    );
}

/// `min_weight` withholds a prediction until the weight before the row
/// reaches it: not one row short, and from the row it is reached.
#[test]
fn min_weight_opens_at_its_weight_and_not_before() {
    let (x, y) = series(40, 37);
    let mut m = Ar::new(ArCfg {
        min_weight: 20.0,
        ..cfg(vec![1], 1, Decay::Halflife(f64::INFINITY))
    })
    .unwrap();
    for t in 0..y.len() {
        let s = m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        assert_eq!(
            s.pred[0].is_finite(),
            s.n_eff >= 20.0,
            "row {t}: {}",
            s.n_eff
        );
    }
}

/// Without an intercept the fit is through the origin (`trend="n"`).
#[test]
fn without_an_intercept_the_fit_is_through_the_origin() {
    let (x, y) = series(500, 41);
    let mut m = Ar::new(ArCfg {
        fit_intercept: false,
        ..cfg(vec![1], 1, Decay::Halflife(f64::INFINITY))
    })
    .unwrap();
    for t in 0..y.len() {
        m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
    }
    let (mut a, mut b) = (vec![0.0; 4], vec![0.0; 2]);
    for t in 1..y.len() {
        let z = [y[t - 1], x[t]];
        for i in 0..2 {
            for j in 0..2 {
                a[i * 2 + j] += z[i] * z[j];
            }
            b[i] += z[i] * y[t];
        }
    }
    let want = oracle::solve(&a, &b);
    let got = &m.coefficients().unwrap()[0];
    assert_eq!(got.len(), 2);
    assert!(rel_gap(got, &want) < 1e-10, "{got:?} vs {want:?}");
}

/// The guard on the early rows is the fit's definiteness, not a count of
/// rows: a target predicts from the first row after its centred lag-and-
/// feature design spans every direction, which the design's own smallest
/// eigenvalue (faer's eigensolver, through `crate::oracle`) says, and a row
/// after the one it is still singular on. Both sides, on a realistic AR(2)
/// with a feature, and on a feature collinear with the lag, which never
/// defines a fit.
#[test]
fn the_early_rows_are_guarded_by_definiteness() {
    let (x, y) = series(40, 43);
    let inf = Decay::Halflife(f64::INFINITY);
    let mut m = Ar::new(cfg(vec![1, 2], 1, inf)).unwrap();
    let mut first = None;
    for t in 0..y.len() {
        let s = m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        if first.is_none() && s.pred[0].is_finite() {
            first = Some(t);
        }
    }
    // The centred design over the learned rows 2..t, by the oracle.
    let spans = |t: usize| {
        let rows: Vec<[f64; 3]> = (2..t).map(|s| [y[s - 1], y[s - 2], x[s]]).collect();
        if rows.len() < 2 {
            return false;
        }
        let n = rows.len() as f64;
        let mean: Vec<f64> = (0..3)
            .map(|i| rows.iter().map(|r| r[i]).sum::<f64>() / n)
            .collect();
        let mut a = vec![0.0; 9];
        for r in &rows {
            for i in 0..3 {
                for j in 0..3 {
                    a[i * 3 + j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                }
            }
        }
        let (vals, _) = oracle::sym_eigen(&a);
        let top = vals.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        vals.iter().all(|v| *v > 1e-9 * top)
    };
    let first = first.expect("it predicts");
    assert!(
        spans(first),
        "row {first} predicts from a design that spans"
    );
    assert!(!spans(first - 1), "and the row before it did not");
    assert_eq!(first, 6, "four learned rows for four coefficients");

    // x the lag itself: the design never spans, and the fit is never read.
    let mut m = Ar::new(cfg(vec![1], 1, inf)).unwrap();
    for t in 1..y.len() {
        let s = m.step(&[y[t - 1]], &[Some(y[t])], 1.0, 1.0);
        assert!(s.pred[0].is_nan(), "row {t}");
    }
    assert!(m.coefficients().is_none());
}

/// Each order's weighted least squares on the rows after the first `pmax`
/// values, by faer's LU (`crate::oracle`): its coefficients in `Ar`'s layout
/// (zeros past the order) and its weighted residual sum of squares.
fn order_fit(q: usize, pmax: usize, x: &[f64], y: &[f64], v: &[f64]) -> (Vec<f64>, f64) {
    let rows: Vec<(Vec<f64>, f64, f64)> = (pmax..y.len())
        .map(|t| {
            let mut z = vec![1.0];
            z.extend((1..=q).map(|l| y[t - l]));
            z.push(x[t]);
            (z, y[t], v[t])
        })
        .collect();
    let k = q + 2;
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for i in 0..k {
        for j in 0..k {
            a[i * k + j] = oracle::sum(rows.iter().map(|(z, _, w)| w * z[i] * z[j]));
        }
        b[i] = oracle::sum(rows.iter().map(|(z, y, w)| w * z[i] * y));
    }
    let beta = oracle::solve(&a, &b);
    let rss = oracle::sum(rows.iter().map(|(z, y, w)| {
        let e = y - z.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>();
        w * e * e
    }));
    let mut out = vec![0.0; pmax + 2];
    out[0] = beta[0];
    out[1..=q].copy_from_slice(&beta[1..=q]);
    out[pmax + 1] = beta[q + 1];
    (out, rss)
}

/// Order selection is, at every order, the weighted least squares of the
/// same rows: the order chosen minimizes `n ln RSS + penalty` with Kish's
/// `n = (Σv)² / Σv²`, and the coefficients reported are that order's fit.
/// No decay and decay with uneven weights, AIC and BIC, on an AR(2) with a
/// feature fitted up to order 6.
#[test]
fn select_is_the_weighted_least_squares_of_each_order() {
    let (x, y) = series(1500, 47);
    let mut s = 9;
    let ws: Vec<f64> = (0..y.len()).map(|_| 0.5 + lcg(&mut s)).collect();
    for (decay, weighted) in [(f64::INFINITY, false), (300.0, true)] {
        for ic in [ArSelect::Aic, ArSelect::Bic] {
            let mut m = Ar::new(ArCfg {
                select: Some(ic),
                ..cfg((1..=6).collect(), 1, Decay::Halflife(decay))
            })
            .unwrap();
            let w: Vec<f64> = if weighted {
                ws.clone()
            } else {
                vec![1.0; y.len()]
            };
            for t in 0..y.len() {
                m.step(&[x[t]], &[Some(y[t])], 1.0, w[t]);
            }
            let n = y.len();
            let v: Vec<f64> = (0..n)
                .map(|t| w[t] * (-((n - 1 - t) as f64) / decay).exp2())
                .collect();
            let tail = &v[6..];
            let kish = tail.iter().sum::<f64>().powi(2) / tail.iter().map(|a| a * a).sum::<f64>();
            let pen = |k: usize| match ic {
                ArSelect::Aic => 2.0 * k as f64,
                ArSelect::Bic => kish.ln() * k as f64,
            };
            let fits: Vec<(Vec<f64>, f64)> = (0..=6).map(|q| order_fit(q, 6, &x, &y, &v)).collect();
            let crit: Vec<f64> = fits
                .iter()
                .enumerate()
                .map(|(q, (_, rss))| kish * rss.ln() + pen(q + 2))
                .collect();
            let best = (0..=6)
                .min_by(|&a, &b| crit[a].total_cmp(&crit[b]))
                .unwrap();
            let got = &m.coefficients().unwrap()[0];
            let chosen = (1..=6).rev().find(|&l| got[l] != 0.0).unwrap_or(0);
            assert_eq!(chosen, best, "{ic:?} at half-life {decay}: {crit:?}");
            assert!(
                rel_gap(got, &fits[best].0) < 1e-9,
                "{got:?} vs {:?}",
                fits[best].0
            );
            assert_eq!(best, 2, "the series' own order, found");
        }
    }
}

/// Under a window, order selection reads the window's rows alone: its
/// weight and Kish's `n` are the window's, not the stream's. With no decay
/// and unit weights, each order's least squares on the rows inside the
/// window, by faer's LU, and `n` their count; the stream is fifteen windows
/// long, so the stream's weight would make `n` fifteen times too large and
/// the penalty too small.
#[test]
fn under_a_window_select_reads_the_window_alone() {
    let (x, y) = series(1500, 61);
    let w = 100.0;
    for ic in [ArSelect::Aic, ArSelect::Bic] {
        let mut m = Ar::new(ArCfg {
            select: Some(ic),
            window: Some(w),
            ..cfg((1..=6).collect(), 1, Decay::Halflife(f64::INFINITY))
        })
        .unwrap();
        for t in 0..y.len() {
            m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
            if !matches!(t, 700 | 1499) {
                continue;
            }
            let n = t + 1;
            // rows s with t - s < w
            let v: Vec<f64> = (0..n)
                .map(|s| f64::from(u8::from(t - s < w as usize)))
                .collect();
            let kish = w;
            let pen = |k: usize| match ic {
                ArSelect::Aic => 2.0 * k as f64,
                ArSelect::Bic => kish.ln() * k as f64,
            };
            let fits: Vec<(Vec<f64>, f64)> = (0..=6)
                .map(|q| order_fit(q, 6, &x[..n], &y[..n], &v))
                .collect();
            let crit: Vec<f64> = fits
                .iter()
                .enumerate()
                .map(|(q, (_, rss))| kish * rss.ln() + pen(q + 2))
                .collect();
            let best = (0..=6)
                .min_by(|&a, &b| crit[a].total_cmp(&crit[b]))
                .unwrap();
            let got = &m.coefficients().unwrap()[0];
            let chosen = (1..=6).rev().find(|&l| got[l] != 0.0).unwrap_or(0);
            assert_eq!(chosen, best, "{ic:?} at row {t}: {crit:?}");
            assert!(
                rel_gap(got, &fits[best].0) < 1e-9,
                "{ic:?} at row {t}: {got:?} vs {:?}",
                fits[best].0
            );
        }
    }
}

/// `select` takes lags 1 to its largest order and no ridge, each refused
/// beside it by name; and both are taken without it.
#[test]
fn select_refuses_gapped_lags_and_a_ridge() {
    let base = ArCfg {
        select: Some(ArSelect::Bic),
        ..cfg(vec![1, 2, 3], 1, Decay::Halflife(10.0))
    };
    assert!(Ar::new(base.clone()).is_ok());
    let gapped = ArCfg {
        lags: vec![1, 3],
        ..base.clone()
    };
    assert!(
        Ar::new(gapped.clone())
            .unwrap_err()
            .contains("lags must be 1 to the largest")
    );
    assert!(
        Ar::new(ArCfg {
            select: None,
            ..gapped
        })
        .is_ok()
    );
    let ridged = ArCfg { ridge: 0.1, ..base };
    assert!(
        Ar::new(ridged.clone())
            .unwrap_err()
            .contains("Set ridge to 0")
    );
    assert!(
        Ar::new(ArCfg {
            select: None,
            ..ridged
        })
        .is_ok()
    );
}

/// Under a hard window of `w` clock units, with no decay, the fit is least
/// squares on the learned rows less than `w` old (a row exactly `w` old
/// out, the default edge), each with its lags, which may be older than the
/// window: faer's LU on those rows, at several ends of the stream.
#[test]
fn a_window_is_least_squares_on_its_rows() {
    let (x, y) = series(400, 53);
    let w = 60.0;
    let mut m = Ar::new(ArCfg {
        window: Some(w),
        ..cfg(vec![1, 2], 1, Decay::Halflife(f64::INFINITY))
    })
    .unwrap();
    let xs: Vec<Vec<f64>> = x.iter().map(|v| vec![*v]).collect();
    for t in 0..y.len() {
        m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0);
        if matches!(t, 150 | 271 | 399) {
            // rows s with t - s < w, past the two the ring fills on
            let from = (t + 1 - w as usize).max(2);
            let (a, b) = wls_rows(&[1, 2], &xs, &y, (from - 2)..(t + 1));
            let want = oracle::solve(&a, &b);
            let got = &m.coefficients().unwrap()[0];
            assert!(rel_gap(got, &want) < 1e-9, "row {t}: {got:?} vs {want:?}");
        }
    }
}

/// Under a window a fit can lose its spread: here the feature varies, then
/// holds one value for longer than the window, so the window's system
/// stops being definite and the target stops predicting; once the feature
/// varies again inside the window, it predicts again. Without a window the
/// old rows keep it definite throughout.
#[test]
fn under_a_window_definiteness_is_checked_on_every_row() {
    let (x0, y) = series(300, 59);
    let x: Vec<f64> = (0..y.len())
        .map(|t| if (100..200).contains(&t) { 1.0 } else { x0[t] })
        .collect();
    let run = |window: Option<f64>| {
        let mut m = Ar::new(ArCfg {
            window,
            ..cfg(vec![1], 1, Decay::Halflife(f64::INFINITY))
        })
        .unwrap();
        (0..y.len())
            .map(|t| m.step(&[x[t]], &[Some(y[t])], 1.0, 1.0).pred[0].is_finite())
            .collect::<Vec<bool>>()
    };
    let windowed = run(Some(30.0));
    assert!(windowed[90], "predicts while the feature varies");
    assert!(
        !windowed[150],
        "withheld once the window holds one value of it"
    );
    assert!(
        windowed[160 - 31],
        "and not before the window has lost the spread"
    );
    assert!(
        windowed[260],
        "predicts again once it varies inside the window"
    );
    assert!(run(None)[150], "no window: the old rows keep it definite");
}
