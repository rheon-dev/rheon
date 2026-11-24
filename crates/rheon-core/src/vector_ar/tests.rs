// The oracles index rows and series by position, as the formulas do.
#![allow(clippy::needless_range_loop)]

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

/// A stable three-series VAR(2) with true cross effects and an exogenous
/// regressor, after 200 rows of burn-in, its levels away from zero:
/// series 1 drives series 0, series 2 is driven by neither. Returns
/// `(x, y)`, `y[t]` the three series' values.
fn series(n: usize, seed: u64) -> (Vec<f64>, Vec<[f64; 3]>) {
    let a1 = [[0.4, 0.3, 0.0], [0.0, 0.5, 0.1], [0.0, 0.0, 0.3]];
    let a2 = [[-0.2, 0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.2]];
    let c = [2.0, -1.0, 5.0];
    let bx = [0.6, 0.0, -0.4];
    let mut s = seed;
    let (mut y1, mut y2) = ([0.0; 3], [0.0; 3]);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..n + 200 {
        let x = 1.0 + normal(&mut s);
        let mut y = [0.0; 3];
        for i in 0..3 {
            y[i] = c[i] + bx[i] * x + normal(&mut s);
            for j in 0..3 {
                y[i] += a1[i][j] * y1[j] + a2[i][j] * y2[j];
            }
        }
        (y2, y1) = (y1, y);
        if t >= 200 {
            xs.push(x);
            ys.push(y);
        }
    }
    (xs, ys)
}

fn cfg(k: usize, p: usize, e: usize, decay: Decay) -> VectorArCfg {
    VectorArCfg {
        n_targets: k,
        n_features: e,
        lags: p,
        fit_intercept: true,
        decay,
        ridge: 0.0,
        min_weight: 0.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        max_state_bytes: None,
        select: None,
        causality: vec![],
        causality_kind: CausalityKind::F,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

const INF: Decay = Decay::Halflife(f64::INFINITY);

fn row(y: &[f64; 3], k: usize) -> Vec<Option<f64>> {
    y[..k].iter().map(|v| Some(*v)).collect()
}

/// The design row of target row `t`: `[1, x?, y_{t−1}, …, y_{t−p}]`.
fn design(x: &[f64], y: &[[f64; 3]], t: usize, k: usize, p: usize, e: usize) -> Vec<f64> {
    let mut z = vec![1.0];
    if e > 0 {
        z.push(x[t]);
    }
    for l in 1..=p {
        z.extend_from_slice(&y[t - l][..k]);
    }
    z
}

/// Weighted least squares of each series on the design over the rows
/// `rows` (each at weight `w[t]`), by the normal equations through faer's
/// partial-pivot LU (`crate::oracle`), which shares no arithmetic with the
/// model's Cholesky: one coefficient vector per series, in `coefficients`'
/// layout.
fn wls(
    x: &[f64],
    y: &[[f64; 3]],
    rows: &[usize],
    w: &[f64],
    k: usize,
    p: usize,
    e: usize,
) -> Vec<Vec<f64>> {
    let zs: Vec<Vec<f64>> = rows.iter().map(|&t| design(x, y, t, k, p, e)).collect();
    let n = zs[0].len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = oracle::sum(zs.iter().zip(rows).map(|(z, &t)| w[t] * z[i] * z[j]));
        }
    }
    (0..k)
        .map(|s| {
            let b: Vec<f64> = (0..n)
                .map(|i| oracle::sum(zs.iter().zip(rows).map(|(z, &t)| w[t] * z[i] * y[t][s])))
                .collect();
            oracle::solve(&a, &b)
        })
        .collect()
}

fn rel_gap(got: &[Vec<f64>], want: &[Vec<f64>]) -> f64 {
    let scale = want.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    got.iter()
        .flatten()
        .zip(want.iter().flatten())
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

/// With no decay the fit is ordinary least squares on the lag design
/// (statsmodels' `VAR(y, exog).fit(p)` on the same rows), at k = 2 and 3,
/// p = 1 and 2, with and without the feature; and on a long stream it
/// recovers the coefficients that made it.
#[test]
fn no_decay_is_least_squares_on_the_lag_design() {
    let (x, y) = series(800, 7);
    for (k, p, e) in [(2, 1, 0), (3, 2, 1), (3, 1, 1), (2, 3, 0)] {
        let mut m = VectorAr::new(cfg(k, p, e, INF)).unwrap();
        for t in 0..y.len() {
            m.step(&x[t..t + e], &row(&y[t], k), 1.0, 1.0);
        }
        let rows: Vec<usize> = (p..y.len()).collect();
        let want = wls(&x, &y, &rows, &vec![1.0; y.len()], k, p, e);
        let got = m.coefficients().unwrap();
        assert!(
            rel_gap(&got, &want) < 1e-11,
            "{k} {p} {e}: {got:?} vs {want:?}"
        );
    }
    let (x, y) = series(40_000, 8);
    let mut m = VectorAr::new(cfg(3, 2, 1, INF)).unwrap();
    for t in 0..y.len() {
        m.step(&x[t..=t], &row(&y[t], 3), 1.0, 1.0);
    }
    let c = m.coefficients().unwrap();
    // Equation 0: [c, x, A1[0][..], A2[0][..]].
    let want0 = [2.0, 0.6, 0.4, 0.3, 0.0, -0.2, 0.1, 0.0];
    for (g, w) in c[0].iter().zip(want0) {
        assert!((g - w).abs() < 0.06, "{:?}", c[0]);
    }
}

/// Under decay, row `t` weighs `w_t λ^{age}`, its age counted on the clock
/// over every row's step: weighted least squares with those weights, with
/// uneven steps and weights so neither cancels.
#[test]
fn decay_is_weighted_least_squares_on_the_clock() {
    let (x, y) = series(700, 11);
    let h = 90.0;
    let mut s = 3;
    let steps: Vec<f64> = (0..y.len()).map(|_| 0.5 + lcg(&mut s)).collect();
    let ws: Vec<f64> = (0..y.len()).map(|_| 0.5 + lcg(&mut s)).collect();
    let mut m = VectorAr::new(cfg(3, 2, 1, Decay::Halflife(h))).unwrap();
    for t in 0..y.len() {
        m.step(&x[t..=t], &row(&y[t], 3), steps[t], ws[t]);
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
    let rows: Vec<usize> = (2..y.len()).collect();
    let want = wls(&x, &y, &rows, &decayed, 3, 2, 1);
    assert!(rel_gap(&m.coefficients().unwrap(), &want) < 1e-9);
}

/// Hard rule 2: row `t`'s prediction is the fit on the rows before it at
/// row `t`'s lags and features; nothing is predicted until the ring holds
/// every lag and the fit is identified.
#[test]
fn a_prediction_is_the_fit_before_the_row() {
    let (x, y) = series(200, 5);
    let mut m = VectorAr::new(cfg(2, 2, 1, INF)).unwrap();
    for t in 0..y.len() {
        let before = m.coefficients();
        let step = m.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
        match before {
            Some(c) if t >= 2 => {
                let z = design(&x, &y, t, 2, 2, 1);
                for j in 0..2 {
                    let want: f64 = c[j].iter().zip(&z).map(|(a, b)| a * b).sum();
                    assert!((step.pred[j] - want).abs() <= 1e-12 * want.abs(), "row {t}");
                }
            }
            _ => assert!(step.pred.iter().all(|v| v.is_nan()), "row {t}"),
        }
    }
}

/// A row of weight 0 learns nothing and moves no ring: the stream with such
/// rows (wild values, the first rows among them) ends in the state of the
/// stream without them, to the bit. A row with a target missing is not a
/// lag and teaches nothing either, but it is predicted where its lags are
/// complete, and its weight counts in `n_eff`.
#[test]
fn a_zero_weight_row_or_an_incomplete_row_is_not_a_lag() {
    let (x, y) = series(300, 13);
    let mut clean = VectorAr::new(cfg(3, 2, 1, INF)).unwrap();
    let mut zeros = VectorAr::new(cfg(3, 2, 1, INF)).unwrap();
    let mut gaps = VectorAr::new(cfg(3, 2, 1, INF)).unwrap();
    for v in [1e9, -3.0] {
        zeros.step(&[v], &[Some(v), Some(-v), Some(2.0 * v)], 1.0, 0.0);
        let s = gaps.step(&[v], &[Some(v), None, Some(v)], 1.0, 1.0);
        assert!(s.pred.iter().all(|p| p.is_nan()), "no lags yet");
    }
    let mut predicted_incomplete = 0;
    for t in 0..y.len() {
        clean.step(&x[t..=t], &row(&y[t], 3), 1.0, 1.0);
        zeros.step(&x[t..=t], &row(&y[t], 3), 1.0, 1.0);
        gaps.step(&x[t..=t], &row(&y[t], 3), 1.0, 1.0);
        if t % 7 == 3 {
            zeros.step(&[1e6], &[Some(-1e6), Some(3.0), Some(1e3)], 1.0, 0.0);
            let s = gaps.step(&[1e6], &[Some(1e6), Some(2.0), None], 1.0, 1.0);
            let p = clean.predict(&[1e6], 1.0);
            // Predicted, from the fit and the lags of the clean stream.
            if p.pred.iter().all(|v| v.is_finite()) {
                predicted_incomplete += 1;
                for j in 0..3 {
                    assert_eq!(s.pred[j].to_bits(), p.pred[j].to_bits(), "row {t}");
                }
            }
        }
    }
    assert!(predicted_incomplete > 30);
    assert_eq!(zeros.ring(), clean.ring());
    assert_eq!(gaps.ring(), clean.ring());
    let want = clean.coefficients().unwrap();
    assert_eq!(zeros.coefficients().unwrap(), want);
    assert_eq!(gaps.coefficients().unwrap(), want);
    assert_eq!(zeros.n_eff(), clean.n_eff());
    assert!(gaps.n_eff() > clean.n_eff(), "a row with weight counts");
    let (mut own, mut clean_own) = (Vec::new(), Vec::new());
    gaps.target_n_eff_into(&mut own);
    clean.target_n_eff_into(&mut clean_own);
    assert_eq!(own, clean_own, "the targets' weight is their learned rows'");
}

/// A stream of nothing but zero-weight rows, the first of them included,
/// leaves no NaN in the state and predicts nothing (hard rule 9).
#[test]
fn zero_weight_rows_from_the_start_leave_no_nan() {
    let mut m = VectorAr::new(cfg(2, 1, 1, Decay::Halflife(5.0))).unwrap();
    for v in [1.0, 2.0, 3.0, 4.0] {
        let s = m.step(&[1.0], &[Some(v), Some(-v)], 1.0, 0.0);
        assert!(s.pred.iter().all(|p| p.is_nan()));
        assert_eq!(s.n_eff, 0.0);
    }
    assert!(m.ring().is_empty());
    let json = serde_json::to_string(&m).unwrap();
    assert!(!json.to_lowercase().contains("nan"), "{json}");
}

/// A break empties the ring and nothing else: the fit and the weight stay,
/// the next `p` rows refill the ring unlearned and unpredicted, and the
/// fit is then least squares on both pieces' own rows.
#[test]
fn a_break_empties_the_ring_and_keeps_the_fit() {
    let (x, y) = series(300, 17);
    let mut m = VectorAr::new(cfg(2, 2, 0, INF)).unwrap();
    for yt in &y[..150] {
        m.step(&[], &row(yt, 2), 1.0, 1.0);
    }
    let (coef, n_eff) = (m.coefficients(), m.n_eff());
    m.clear_lags();
    assert!(m.ring().is_empty());
    assert_eq!((m.coefficients(), m.n_eff()), (coef.clone(), n_eff));
    for (t, yt) in y.iter().enumerate().take(152).skip(150) {
        let s = m.step(&[], &row(yt, 2), 1.0, 1.0);
        assert!(s.pred.iter().all(|p| p.is_nan()), "row {t} has no lags");
        assert_eq!(m.coefficients(), coef, "row {t} is not learned");
    }
    for t in 152..300 {
        m.step(&[], &row(&y[t], 2), 1.0, 1.0);
    }
    let rows: Vec<usize> = (2..150).chain(152..300).collect();
    let want = wls(&x, &y, &rows, &vec![1.0; 300], 2, 2, 0);
    assert!(rel_gap(&m.coefficients().unwrap(), &want) < 1e-11);
}

/// Saved at every depth of the ring, from empty to full and past, with a
/// test, `select` and the variances read, a model continues to the bit;
/// the ring's depth is the settings', not the loaded container's.
#[test]
fn a_saved_state_continues_to_the_bit_at_every_depth() {
    let (x, y) = series(80, 29);
    for cut in 0..8 {
        let c = VectorArCfg {
            select: Some(VectorArSelect::Aic),
            causality: vec![VectorArTest {
                caused: vec![0],
                causing: vec![1],
            }],
            ..cfg(2, 3, 1, Decay::Halflife(30.0))
        };
        let mut a = VectorAr::new(c).unwrap();
        a.set_keep_factor(true);
        for t in 0..cut {
            a.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
        }
        let bytes = rmp_serde::to_vec_named(&a.state()).unwrap();
        let mut b = VectorAr::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for t in cut..y.len() {
            let sa = a.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
            let sb = b.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
            for (p, q) in sa.pred.iter().zip(&sb.pred) {
                assert_eq!(p.to_bits(), q.to_bits(), "cut {cut}, row {t}");
            }
            assert_eq!(format!("{:?}", sa.extra), format!("{:?}", sb.extra));
            assert!(b.ring().len() <= 6);
        }
        assert_eq!(
            rmp_serde::to_vec_named(&a.state()).unwrap(),
            rmp_serde::to_vec_named(&b.state()).unwrap()
        );
    }
}

/// `state_bytes` bounds what the state writes, fresh and after a stream, at
/// several shapes, and is not loose; `max_state_bytes` takes a state at it
/// and refuses one past it with the bytes it needs; ten series at ten lags
/// are past the default cap.
#[test]
fn the_state_is_bounded_by_its_settings_and_capped() {
    let (x, y) = series(300, 31);
    for (k, p, e, tests) in [(1, 1, 0, 0), (2, 1, 1, 1), (3, 4, 1, 2), (3, 8, 0, 0)] {
        let c = VectorArCfg {
            causality: (0..tests)
                .map(|_| VectorArTest {
                    caused: vec![0],
                    causing: vec![1],
                })
                .collect(),
            ..cfg(k, p, e, Decay::Halflife(50.0))
        };
        let mut model = VectorAr::new(c.clone()).unwrap();
        model.set_keep_factor(true);
        let bound = VectorAr::state_bytes(&c);
        let fresh = rmp_serde::to_vec_named(&model.state()).unwrap().len() as u64;
        assert!(fresh <= bound, "{fresh} > {bound}");
        for t in 0..y.len() {
            model.step(&x[t..t + e], &row(&y[t], k), 1.0, 1.0);
        }
        let after = rmp_serde::to_vec_named(&model.state()).unwrap().len() as u64;
        assert!(after <= bound, "{k} {p}: {after} > {bound}");
        assert!(
            after * 4 > bound,
            "{k} {p}: the bound is loose, {after} of {bound}"
        );
    }
    let c = cfg(3, 4, 1, Decay::Halflife(50.0));
    let bytes = VectorAr::state_bytes(&c);
    assert!(
        VectorAr::new(VectorArCfg {
            max_state_bytes: Some(bytes),
            ..c.clone()
        })
        .is_ok()
    );
    let err = VectorAr::new(VectorArCfg {
        max_state_bytes: Some(bytes - 1),
        ..c.clone()
    })
    .unwrap_err();
    assert!(
        err.contains("past max_state_bytes") && err.contains(&bytes.to_string()),
        "{err}"
    );
    let big = cfg(10, 10, 0, Decay::Halflife(50.0));
    let err = VectorAr::new(big.clone()).unwrap_err();
    let need = VectorAr::state_bytes(&big);
    assert!(err.contains(&format!("{need} bytes")), "{err}");
    assert!(VectorAr::new(cfg(10, 5, 0, Decay::Halflife(50.0))).is_ok());
}

/// The settings it refuses, each by name.
#[test]
fn it_refuses_its_settings_by_name() {
    let base = cfg(3, 2, 1, Decay::Halflife(10.0));
    let err = |c: VectorArCfg| VectorAr::new(c).unwrap_err();
    let t = |caused: Vec<usize>, causing: Vec<usize>| VectorArTest { caused, causing };
    assert!(
        err(VectorArCfg {
            lags: 0,
            ..base.clone()
        })
        .contains("lags must be at least 1")
    );
    assert!(
        err(VectorArCfg {
            n_targets: 0,
            ..base.clone()
        })
        .contains("at least one target")
    );
    assert!(
        err(VectorArCfg {
            ridge: -1.0,
            ..base.clone()
        })
        .contains("ridge")
    );
    assert!(
        err(VectorArCfg {
            ridge: 0.5,
            select: Some(VectorArSelect::Bic),
            ..base.clone()
        })
        .contains("select compares")
    );
    assert!(
        err(VectorArCfg {
            ridge: 0.5,
            causality: vec![t(vec![0], vec![1])],
            ..base.clone()
        })
        .contains("causality test's law")
    );
    for (bad, words) in [
        (t(vec![], vec![1]), "names no caused"),
        (t(vec![0], vec![3]), "past the 3 targets"),
        (t(vec![0, 0], vec![1]), "twice"),
        (t(vec![0], vec![0, 1]), "both caused and causing"),
    ] {
        let e = err(VectorArCfg {
            causality: vec![bad],
            ..base.clone()
        });
        assert!(e.contains(words), "{e}");
    }
    let e = err(VectorArCfg {
        causality: vec![t(vec![0, 1], vec![2])],
        causality_kind: CausalityKind::Lr,
        ..base.clone()
    });
    assert!(e.contains("one caused equation"), "{e}");
    assert!(
        err(VectorArCfg {
            min_weight: -1.0,
            ..base.clone()
        })
        .contains("min_weight")
    );
    assert!(
        err(VectorArCfg {
            max_state_bytes: Some(0),
            ..base.clone()
        })
        .contains("max_state_bytes")
    );
}

/// `min_weight` withholds every prediction and test while the weight before
/// the row is below it, and nothing from the row it reaches it.
#[test]
fn min_weight_opens_at_its_weight_and_not_before() {
    let (x, y) = series(120, 37);
    let c = VectorArCfg {
        min_weight: 40.0,
        causality: vec![VectorArTest {
            caused: vec![0],
            causing: vec![1],
        }],
        ..cfg(2, 1, 1, INF)
    };
    let mut gated = VectorAr::new(c.clone()).unwrap();
    let mut open = VectorAr::new(VectorArCfg {
        min_weight: 0.0,
        ..c
    })
    .unwrap();
    for t in 0..y.len() {
        let g = gated.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
        let o = open.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
        let Some(Extra::VectorAr { values: gv }) = &g.extra else {
            panic!("a test's extra")
        };
        let Some(Extra::VectorAr { values: ov }) = &o.extra else {
            panic!("a test's extra")
        };
        if g.n_eff < 40.0 {
            assert!(g.pred.iter().chain(gv).all(|v| v.is_nan()), "row {t}");
        } else {
            for (a, b) in g.pred.iter().chain(gv).zip(o.pred.iter().chain(ov)) {
                assert_eq!(a.to_bits(), b.to_bits(), "row {t}");
            }
            assert!(g.pred.iter().all(|v| v.is_finite()), "row {t}");
        }
    }
}

/// Without an intercept the fit is through the origin, on the raw moments.
#[test]
fn without_an_intercept_the_fit_is_through_the_origin() {
    let (x, y) = series(400, 41);
    let mut m = VectorAr::new(VectorArCfg {
        fit_intercept: false,
        ..cfg(2, 2, 1, INF)
    })
    .unwrap();
    for t in 0..y.len() {
        m.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
    }
    let zs: Vec<Vec<f64>> = (2..y.len())
        .map(|t| design(&x, &y, t, 2, 2, 1)[1..].to_vec())
        .collect();
    let n = zs[0].len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = oracle::sum(zs.iter().map(|z| z[i] * z[j]));
        }
    }
    let want: Vec<Vec<f64>> = (0..2)
        .map(|s| {
            let b: Vec<f64> = (0..n)
                .map(|i| oracle::sum(zs.iter().zip(2..).map(|(z, t)| z[i] * y[t][s])))
                .collect();
            oracle::solve(&a, &b)
        })
        .collect();
    assert!(rel_gap(&m.coefficients().unwrap(), &want) < 1e-11);
}

/// The early rows are guarded by identification, not a row count: a fit is
/// withheld (`not_identified`) until the regressors' moments are positive
/// definite, and a pair of series one of which is a multiple of the other
/// is never identified.
#[test]
fn the_early_rows_are_guarded_by_identification() {
    let (x, y) = series(60, 43);
    let mut m = VectorAr::new(cfg(2, 2, 1, INF)).unwrap();
    let mut first = None;
    for t in 0..y.len() {
        let s = m.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
        if first.is_none() && s.pred.iter().all(|v| v.is_finite()) {
            first = Some(t);
        }
    }
    // The design has 1 + 1 + 4 columns: identified once 6 rows are learned,
    // which is rows 2..=7, so row 8 is the first predicted.
    assert_eq!(first, Some(8));
    let mut collinear = VectorAr::new(cfg(2, 1, 0, INF)).unwrap();
    for t in 0..y.len() {
        let v = y[t][0];
        let s = collinear.step(&[], &[Some(v), Some(2.0 * v)], 1.0, 1.0);
        assert!(s.pred.iter().all(|p| p.is_nan()));
    }
    assert!(collinear.not_identified());
    assert!(collinear.coefficients().is_none());
    assert!(collinear.fit_now().unwrap_err().contains("do not identify"));
}

/// The weighted least squares of one series' equation at order `q` over
/// the rows `pmax..`: its coefficients and residual sum of squares.
fn order_rss(y: &[[f64; 3]], k: usize, q: usize, pmax: usize) -> Vec<Vec<f64>> {
    let rows: Vec<usize> = (pmax..y.len()).collect();
    let zs: Vec<Vec<f64>> = rows
        .iter()
        .map(|&t| {
            let mut z = vec![1.0];
            for l in 1..=q {
                z.extend_from_slice(&y[t - l][..k]);
            }
            z
        })
        .collect();
    let n = zs[0].len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = oracle::sum(zs.iter().map(|z| z[i] * z[j]));
        }
    }
    let betas: Vec<Vec<f64>> = (0..k)
        .map(|s| {
            let b: Vec<f64> = (0..n)
                .map(|i| oracle::sum(zs.iter().zip(&rows).map(|(z, &t)| z[i] * y[t][s])))
                .collect();
            oracle::solve(&a, &b)
        })
        .collect();
    // The residual co-moments over the rows, `k × k`.
    let resid: Vec<Vec<f64>> = zs
        .iter()
        .zip(&rows)
        .map(|(z, &t)| {
            (0..k)
                .map(|s| y[t][s] - betas[s].iter().zip(z).map(|(b, v)| b * v).sum::<f64>())
                .collect()
        })
        .collect();
    (0..k)
        .map(|i| {
            (0..k)
                .map(|j| oracle::sum(resid.iter().map(|r| r[i] * r[j])))
                .collect()
        })
        .collect()
}

/// `select` minimizes statsmodels' criterion over every order on the same
/// rows: each order's `ln det Σ_mle` from its own least-squares fit (the
/// oracle's), the penalties written from the definition; the chosen order's
/// coefficients are that order's fit, and the fit identified at order 1
/// predicts before order `p` is.
#[test]
fn select_is_the_criterion_over_each_orders_least_squares() {
    let (_, y) = series(500, 47);
    let (k, pmax) = (3, 4);
    for ic in [
        VectorArSelect::Aic,
        VectorArSelect::Bic,
        VectorArSelect::Hqic,
        VectorArSelect::Fpe,
    ] {
        let mut m = VectorAr::new(VectorArCfg {
            select: Some(ic),
            ..cfg(k, pmax, 0, INF)
        })
        .unwrap();
        for t in 0..y.len() {
            m.step(&[], &row(&y[t], k), 1.0, 1.0);
        }
        let n = (y.len() - pmax) as f64;
        let vals: Vec<f64> = (0..=pmax)
            .map(|q| {
                let sse = order_rss(&y, k, q, pmax);
                let flat: Vec<f64> = sse.iter().flatten().map(|v| v / n).collect();
                let ld = oracle::log_det(&flat);
                let free = (q * k * k + k) as f64;
                let mq = (k * q + 1) as f64;
                match ic {
                    VectorArSelect::Aic => ld + 2.0 * free / n,
                    VectorArSelect::Bic => ld + n.ln() * free / n,
                    VectorArSelect::Hqic => ld + 2.0 * n.ln().ln() * free / n,
                    VectorArSelect::Fpe => ((n + mq) / (n - mq)).powi(k as i32) * ld.exp(),
                }
            })
            .collect();
        let want = (0..=pmax)
            .min_by(|&a, &b| vals[a].total_cmp(&vals[b]))
            .unwrap();
        assert_eq!(m.order(), Some(want), "{ic:?}: {vals:?}");
        let fit = m.fit_now().unwrap();
        let ics = fit.info_criteria();
        let col = match ic {
            VectorArSelect::Aic => 0,
            VectorArSelect::Bic => 1,
            VectorArSelect::Hqic => 2,
            VectorArSelect::Fpe => 3,
        };
        for q in 0..=pmax {
            let (g, w) = (ics[q][col], vals[q]);
            assert!(
                (g - w).abs() <= 1e-10 * w.abs(),
                "{ic:?} order {q}: {g} vs {w}"
            );
        }
    }
}

/// The Granger Wald statistic of one caused equation is the restricted
/// fit's extra residual sum over the unrestricted's residual variance,
/// `(SSR_r − SSR_u) / (SSR_u / df)`, here from two separate least-squares
/// fits (the oracle's), the restricted one never run by the model; and a
/// set of caused equations' F at the null is near 1.
#[test]
fn a_granger_wald_is_the_restricted_fits_extra_residual() {
    let (_, y) = series(400, 53);
    let (k, p) = (3, 2);
    let tests = vec![
        VectorArTest {
            caused: vec![0],
            causing: vec![1],
        },
        VectorArTest {
            caused: vec![2],
            causing: vec![0, 1],
        },
    ];
    let mut m = VectorAr::new(VectorArCfg {
        causality: tests.clone(),
        causality_kind: CausalityKind::Wald,
        ..cfg(k, p, 0, INF)
    })
    .unwrap();
    let mut last = None;
    for t in 0..y.len() {
        last = Some(m.step(&[], &row(&y[t], k), 1.0, 1.0));
    }
    // The row after the stream reads the fit of every row.
    let s = m.predict(&[], 1.0);
    let Some(Extra::VectorAr { values }) = s.extra else {
        panic!("the tests' values")
    };
    assert!(last.is_some());
    let rows: Vec<usize> = (p..y.len()).collect();
    let n = rows.len() as f64;
    let df = n - (k * p + 1) as f64;
    for (i, t) in tests.iter().enumerate() {
        let e = t.caused[0];
        let ssr = |keep: &dyn Fn(usize) -> bool| -> f64 {
            let zs: Vec<Vec<f64>> = rows
                .iter()
                .map(|&r| {
                    let mut z = vec![1.0];
                    for l in 1..=p {
                        for j in 0..k {
                            if keep(j) {
                                z.push(y[r - l][j]);
                            }
                        }
                    }
                    z
                })
                .collect();
            let nn = zs[0].len();
            let mut a = vec![0.0; nn * nn];
            for i in 0..nn {
                for j in 0..nn {
                    a[i * nn + j] = oracle::sum(zs.iter().map(|z| z[i] * z[j]));
                }
            }
            let b: Vec<f64> = (0..nn)
                .map(|i| oracle::sum(zs.iter().zip(&rows).map(|(z, &r)| z[i] * y[r][e])))
                .collect();
            let beta = oracle::solve(&a, &b);
            oracle::sum(zs.iter().zip(&rows).map(|(z, &r)| {
                let f: f64 = beta.iter().zip(z).map(|(b, v)| b * v).sum();
                (y[r][e] - f) * (y[r][e] - f)
            }))
        };
        let u = ssr(&|_| true);
        let r = ssr(&|j| !t.causing.contains(&j));
        let want = (r - u) / (u / df);
        let got = values[2 * i];
        assert!(
            (got - want).abs() <= 1e-9 * want,
            "test {i}: {got} vs {want}"
        );
        let q = (p * t.causing.len()) as f64;
        let pv = values[2 * i + 1];
        assert!((pv - tails::chi2_sf(want, q)).abs() <= 1e-12, "{pv}");
    }
    // Series 1 drives series 0: that test rejects; series 2 is driven by
    // neither 0 nor 1: that one does not.
    assert!(values[1] < 1e-6, "{values:?}");
    assert!(values[3] > 1e-3, "{values:?}");
}

/// The tests and the order are read at the solve the fit's schedule brings
/// due and held between solves, as the coefficients are.
#[test]
fn the_tests_and_the_order_ride_on_the_solve_schedule() {
    let (_, y) = series(400, 59);
    let c = VectorArCfg {
        select: Some(VectorArSelect::Bic),
        causality: vec![VectorArTest {
            caused: vec![0],
            causing: vec![1],
        }],
        solve_every: f64::INFINITY,
        max_rows_between_solves: 25,
        ..cfg(2, 3, 0, INF)
    };
    let mut m = VectorAr::new(c).unwrap();
    let mut changes = 0;
    let mut prev: Option<Vec<f64>> = None;
    for t in 0..y.len() {
        let s = m.step(&[], &row(&y[t], 2), 1.0, 1.0);
        let Some(Extra::VectorAr { values }) = s.extra else {
            panic!()
        };
        if let Some(p) = &prev
            && values
                .iter()
                .zip(p)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            changes += 1;
        }
        prev = Some(values);
    }
    // One read a solve, every 25 rows with weight, and a first solve.
    assert!((14..=17).contains(&changes), "{changes}");
}

/// At one series and no features the VAR is `ar`'s AR(p): the same fit.
#[test]
fn one_series_is_the_autoregression() {
    let (_, y) = series(500, 61);
    let mut var = VectorAr::new(cfg(1, 3, 0, Decay::Halflife(60.0))).unwrap();
    let mut ar = crate::Ar::new(crate::ArCfg {
        n_targets: 1,
        n_features: 0,
        lags: vec![1, 2, 3],
        fit_intercept: true,
        decay: Decay::Halflife(60.0),
        ridge: 0.0,
        min_weight: 0.0,
        max_state_bytes: None,
        select: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    })
    .unwrap();
    for t in 0..y.len() {
        let a = var.step(&[], &[Some(y[t][0])], 1.0, 1.0);
        let b = ar.step(&[], &[Some(y[t][0])], 1.0, 1.0);
        assert_eq!(a.pred[0].is_nan(), b.pred[0].is_nan(), "row {t}");
        if a.pred[0].is_finite() {
            assert!(
                (a.pred[0] - b.pred[0]).abs() <= 1e-12 * b.pred[0].abs(),
                "row {t}"
            );
        }
        assert_eq!(a.n_eff, b.n_eff);
    }
    let (a, b) = (var.coefficients().unwrap(), ar.coefficients().unwrap());
    assert!(rel_gap(&a, &b) < 1e-12, "{a:?} vs {b:?}");
}

/// Under a window the fit is least squares on the rows inside it, each
/// row's lags read wherever they lie; and once rows leave it, the fit's
/// identification is read again at every solve.
#[test]
fn a_window_is_least_squares_on_its_rows() {
    let (x, y) = series(300, 67);
    let w = 80.0;
    let mut m = VectorAr::new(VectorArCfg {
        window: Some(w),
        ..cfg(2, 2, 1, INF)
    })
    .unwrap();
    for t in 0..y.len() {
        m.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
    }
    // The window holds the rows whose clock is in `(T − 80, T]`: the last
    // 80 rows.
    let rows: Vec<usize> = (y.len() - 80..y.len()).collect();
    let want = wls(&x, &y, &rows, &vec![1.0; y.len()], 2, 2, 1);
    assert!(rel_gap(&m.coefficients().unwrap(), &want) < 1e-9);
    // A window that a held value fills is not identified any more.
    let mut held = VectorAr::new(VectorArCfg {
        window: Some(20.0),
        ..cfg(2, 1, 0, INF)
    })
    .unwrap();
    for t in 0..100 {
        held.step(&[], &row(&y[t], 2), 1.0, 1.0);
    }
    assert!(!held.not_identified());
    for _ in 0..40 {
        held.step(&[], &[Some(1.0), Some(2.0)], 1.0, 1.0);
    }
    assert!(held.not_identified());
    let s = held.step(&[], &[Some(1.0), Some(2.0)], 1.0, 1.0);
    assert!(s.pred.iter().all(|v| v.is_nan()));
}

/// The coefficients' variances are statsmodels' `bse²`: `σ²_u,jj` times the
/// diagonal of `(Z'Z)⁻¹`, here from the oracle's inverse of the raw normal
/// equations, with `σ²_u = SSE / df`.
#[test]
fn the_coefficient_variances_are_least_squares_standard_errors() {
    let (x, y) = series(300, 71);
    let (k, p, e) = (2, 2, 1);
    let mut m = VectorAr::new(cfg(k, p, e, INF)).unwrap();
    m.set_keep_factor(true);
    for t in 0..y.len() {
        m.step(&x[t..=t], &row(&y[t], k), 1.0, 1.0);
    }
    let Some(crate::CoefVariance::Absolute(v)) = m.coef_variance() else {
        panic!("variances")
    };
    let rows: Vec<usize> = (p..y.len()).collect();
    let zs: Vec<Vec<f64>> = rows.iter().map(|&t| design(&x, &y, t, k, p, e)).collect();
    let n = zs[0].len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = oracle::sum(zs.iter().map(|z| z[i] * z[j]));
        }
    }
    let inv = oracle::inverse(&a);
    let coef = wls(&x, &y, &rows, &vec![1.0; y.len()], k, p, e);
    let df = (rows.len() - n) as f64;
    for s in 0..k {
        let sse = oracle::sum(zs.iter().zip(&rows).map(|(z, &t)| {
            let f: f64 = coef[s].iter().zip(z).map(|(b, v)| b * v).sum();
            (y[t][s] - f) * (y[t][s] - f)
        }));
        for i in 0..n {
            let want = sse / df * inv[i * n + i];
            assert!(
                (v[s][i] - want).abs() <= 1e-9 * want,
                "{s} {i}: {} vs {want}",
                v[s][i]
            );
        }
    }
}

/// The impulse responses are the moving-average recursion of the fitted
/// lag matrices, the orthogonalized ones that times `Σ_u`'s factor, and the
/// variance decomposition's shares sum to 1 at every horizon; the
/// companion's eigenvalues of a fitted VAR(1) are its lag matrix's.
#[test]
fn the_reads_are_functions_of_the_fit() {
    let (_, y) = series(600, 73);
    let mut m = VectorAr::new(cfg(3, 2, 0, INF)).unwrap();
    for t in 0..y.len() {
        m.step(&[], &row(&y[t], 3), 1.0, 1.0);
    }
    let fit = m.fit_now().unwrap();
    let k = 3;
    let phi = fit.ma_rep(6);
    let a = |l: usize, i: usize, j: usize| fit.coefs[(l * k + i) * k + j];
    for i in 0..k {
        for j in 0..k {
            assert_eq!(phi[i * k + j], if i == j { 1.0 } else { 0.0 });
            assert_eq!(phi[k * k + i * k + j], a(0, i, j));
            let two: f64 = (0..k).map(|q| a(0, i, q) * a(0, q, j)).sum::<f64>() + a(1, i, j);
            assert!((phi[2 * k * k + i * k + j] - two).abs() < 1e-14);
        }
    }
    let fevd = fit.fevd(6);
    for e in 0..k {
        for h in 0..6 {
            let s: f64 = (0..k).map(|j| fevd[(e * 6 + h) * k + j]).sum();
            assert!((s - 1.0).abs() < 1e-12, "{s}");
        }
    }
    let theta = fit.orth_ma_rep(2);
    // `Θ_0 Θ_0ᵀ = Σ_u`.
    for i in 0..k {
        for j in 0..k {
            let s: f64 = (0..k).map(|q| theta[i * k + q] * theta[j * k + q]).sum();
            assert!((s - fit.sigma_u[i * k + j]).abs() < 1e-12);
        }
    }
    let mut one = VectorAr::new(cfg(3, 1, 0, INF)).unwrap();
    for t in 0..y.len() {
        one.step(&[], &row(&y[t], 3), 1.0, 1.0);
    }
    let f1 = one.fit_now().unwrap();
    // At order 1 the companion matrix is `A_1`: its eigenvalues sum to its
    // trace, multiply to its determinant (the oracle's), and come largest
    // modulus first.
    let ev = f1.eigenvalues().unwrap();
    assert_eq!(ev.len(), 3);
    assert!(
        ev.windows(2)
            .all(|w| w[0].0.hypot(w[0].1) >= w[1].0.hypot(w[1].1))
    );
    let trace: f64 = ev.iter().map(|(r, _)| r).sum();
    let tr_a: f64 = (0..3).map(|i| f1.coefs[i * 3 + i]).sum();
    assert!((trace - tr_a).abs() < 1e-12);
    let (mut pr, mut pi) = (1.0, 0.0);
    for (r, i) in &ev {
        (pr, pi) = (pr * r - pi * i, pr * i + pi * r);
    }
    let a = &f1.coefs;
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    assert!(pi.abs() < 1e-12 && (pr - det).abs() < 1e-12, "{pr} {det}");
    // The mean solves `(I − A) μ = c`.
    let mu = f1.mean().unwrap();
    for i in 0..3 {
        let lhs: f64 = mu[i] - (0..3).map(|j| f1.coefs[i * 3 + j] * mu[j]).sum::<f64>();
        assert!((lhs - f1.intercept[i]).abs() < 1e-12);
    }
}

/// The orthogonalized responses' covariance takes `Σ_u`'s through
/// `H = ∂vec P / ∂vech Σ_u`: here against a central difference of the
/// Cholesky factor, through the period-0 standard error, which is that term
/// alone.
#[test]
fn the_orthogonal_response_covariance_differentiates_the_factor() {
    let (_, y) = series(400, 79);
    let mut m = VectorAr::new(cfg(2, 1, 0, INF)).unwrap();
    for t in 0..y.len() {
        m.step(&[], &row(&y[t], 2), 1.0, 1.0);
    }
    let fit = m.fit_now().unwrap();
    let se = fit.irf_stderr(1, true);
    // Period 0: Θ_0 = P, Var(vec P) = H Σ_σ Hᵀ / n, Σ_σ's entries
    // `σ_ik σ_jl + σ_il σ_jk`.
    let s = &fit.sigma_u;
    let chol = |s: &[f64]| {
        let mut c = s.to_vec();
        assert!(fit::cholesky(&mut c, 2));
        c
    };
    let vech = [(0, 0), (1, 0), (1, 1)];
    let h = 1e-6;
    let mut jac = [[0.0; 3]; 4];
    for (v, &(r, c)) in vech.iter().enumerate() {
        let mut up = s.clone();
        let mut dn = s.clone();
        up[r * 2 + c] += h;
        dn[r * 2 + c] -= h;
        if r != c {
            up[c * 2 + r] += h;
            dn[c * 2 + r] -= h;
        }
        let (pu, pd) = (chol(&up), chol(&dn));
        for i in 0..2 {
            for j in 0..2 {
                jac[j * 2 + i][v] = (pu[i * 2 + j] - pd[i * 2 + j]) / (2.0 * h);
            }
        }
    }
    let cov = |a: (usize, usize), b: (usize, usize)| {
        let (i, j) = a;
        let (k2, l) = b;
        s[i * 2 + k2] * s[j * 2 + l] + s[i * 2 + l] * s[j * 2 + k2]
    };
    for vi in 0..4 {
        let mut var = 0.0;
        for (a, &ea) in vech.iter().enumerate() {
            for (b, &eb) in vech.iter().enumerate() {
                var += jac[vi][a] * cov(ea, eb) * jac[vi][b];
            }
        }
        let (r, c) = (vi % 2, vi / 2);
        let want = (var / fit.n_kish).sqrt();
        let got = se[r * 2 + c];
        assert!(
            (got - want).abs() <= 1e-6 * want.max(1e-12),
            "{vi}: {got} vs {want}"
        );
    }
}

/// A state of the wrong shape is refused as damaged, never loaded to panic.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let (x, y) = series(40, 83);
    let mut m = VectorAr::new(cfg(2, 2, 1, INF)).unwrap();
    for t in 0..y.len() {
        m.step(&x[t..=t], &row(&y[t], 2), 1.0, 1.0);
    }
    let mut bad = m.clone();
    bad.ring.push_back(1.0);
    let err = VectorAr::restore(&bad.state()).unwrap_err().to_string();
    assert!(err.contains("damaged"), "{err}");
    let mut bad = m.clone();
    bad.last.chosen = vec![1.0; 3];
    assert!(VectorAr::restore(&bad.state()).is_err());
    let mut bad = m.clone();
    bad.cfg.lags = 3;
    assert!(VectorAr::restore(&bad.state()).is_err());
    assert!(VectorAr::restore(&m.state()).is_ok());
}
