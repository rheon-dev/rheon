//! Golden signatures for the options whose point is a structure the
//! fixed stream of `golden.rs` does not hold: an ill-conditioned pair of
//! features (`ewridge`'s ridge), irrelevant features among relevant ones
//! (`lasso`'s selection), a coefficient that drifts (`kalman`), a sparse
//! stream of binary indicators (`ftrl`) and a stream long enough for a decaying
//! rate to leave its start (`sgd`). Each test builds a seeded stream with the
//! structure, asserts that the model found it, and then pins the numbers, so
//! a number that moved because the structure was lost fails the assertion
//! with a reason, not a diff.
//!
//! Regenerate with `PRINT_GOLDEN=1 cargo test -p rheon-core --test
//! golden_options -- --nocapture <test>`, one test a run, and only after
//! confirming a change is intended.

use rheon_core::*;

/// Deterministic pseudo-random stream, as in `golden.rs`.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A standard normal to within the central limit theorem's reach (the sum
/// of six uniforms on [-1, 1] has variance 2): sums and one square root, so
/// the same bits everywhere.
fn gauss(s: &mut u64) -> f64 {
    (0..6).map(|_| lcg(s)).sum::<f64>() / 2f64.sqrt()
}

/// One row: features, the target, clock step, row weight.
type Row = (Vec<f64>, [Option<f64>; 1], f64, f64);

/// `n` rows of unit weight on a unit clock, row `i` drawn by `draw` into
/// its features and its target.
fn rows(n: usize, seed: u64, mut draw: impl FnMut(usize, &mut u64) -> (Vec<f64>, f64)) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let (x, y) = draw(i, &mut s);
            (x, [Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
        })
        .collect()
}

/// The output slot `pick` before each row (hard rule 2: out of sample), all
/// of them, so a test can read any stretch.
fn predictions<M: OnlineModel>(model: &mut M, rows: &[Row], pick: usize) -> Vec<f64> {
    rows.iter()
        .map(|(x, y, d, w)| model.step(x, y, *d, *w).pred[pick])
        .collect()
}

fn check(name: &str, got: &[f64], want: &[f64]) {
    if std::env::var("PRINT_GOLDEN").is_ok_and(|v| v == "1") {
        let vals: Vec<String> = got.iter().map(|v| format!("{v:?}")).collect();
        println!("GOLDEN {name}: &[{}];", vals.join(", "));
        panic!("PRINT_GOLDEN=1 printed {name} and compared nothing; run again without it");
    }
    assert_eq!(got.len(), want.len(), "{name}: wrong signature length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
            "{name}[{i}]: got {g:.17e}, want {w:.17e}"
        );
    }
}

fn ewridge_cfg(ridge: f64) -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        ridge: vec![ridge],
        feature_sets: vec![],
        standardize: false,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: rheon_core::TargetGaps::OwnRows,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// `y = x0 + x1 + 0.3 e` on two features of unit variance at a correlation
/// of 0.995: the sum of the slopes is determined by the data, their split
/// is not (the smallest eigenvalue of the Gram is 0.005 of the variance).
fn collinear_rows() -> Vec<Row> {
    rows(400, 31415, |_, s| {
        let (a, b) = (gauss(s), gauss(s));
        let x1 = 0.995 * a + (1.0f64 - 0.995 * 0.995).sqrt() * b;
        (vec![a, x1], a + x1 + 0.3 * gauss(s))
    })
}

/// `ewridge` on the collinear pair, with a ridge of 0.05 (the fit is in the
/// mean form, so the ridge is in the features' variance: ten times the
/// smallest eigenvalue, 0.005, a fortieth of the largest) against one that is not
/// there: both find the sum of the slopes, and the ridge splits it nearly
/// evenly where the almost unpenalized fit takes the split from the noise.
#[test]
fn ewridge_conditioning_golden() {
    let data = collinear_rows();
    let mut ridged = EwRidge::new(ewridge_cfg(0.05)).unwrap();
    let mut plain = EwRidge::new(ewridge_cfg(1e-9)).unwrap();
    let pred = predictions(&mut ridged, &data, 0);
    predictions(&mut plain, &data, 0);
    let (r, p) = (
        ridged.coefficients().unwrap()[0].clone(),
        plain.coefficients().unwrap()[0].clone(),
    );
    for (name, c) in [("ridged", &r), ("plain", &p)] {
        assert!((c[1] + c[2] - 2.0).abs() < 0.15, "{name}: slopes {c:?}");
    }
    assert!(
        (r[1] - r[2]).abs() < (p[1] - p[2]).abs() / 2.0,
        "ridged {r:?}, plain {p:?}"
    );
    let mut got = vec![pred[100], pred[200], pred[399]];
    got.extend(&r);
    got.extend(&p);
    check("ewridge_conditioning", &got, GOLDEN_EWRIDGE_CONDITIONING);
}

/// Six independent features, two of them in the target (`1.5 x0 - x1`) and
/// four not, noise 0.3: the lasso's correlation-form statistics `c_i` are
/// about 0 for the four, within the noise's `0.3 / sqrt(n)`.
fn sparse_truth_rows() -> Vec<Row> {
    rows(400, 2718, |_, s| {
        let x: Vec<f64> = (0..6).map(|_| gauss(s)).collect();
        let y = 1.5 * x[0] - x[1] + 0.3 * gauss(s);
        (x, y)
    })
}

fn lasso_cfg(path: Vec<f64>) -> LassoCfg {
    LassoCfg {
        min_weight_holds_learning: false,
        n_features: 6,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        lasso_path: path,
        l1_ratio: 1.0,
        select_half_life: None,
        min_weight: 3.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        max_iter: 200,
        tol: 1e-12,
        target_gaps: rheon_core::TargetGaps::OwnRows,
    }
}

/// `lasso` on the sparse truth, over a path of 0.25, 0.05 and 0: at 0.25
/// (a penalty above every irrelevant feature's `c_i`, below the relevant
/// ones) it selects exactly the two features that are in the target, shrunk
/// by the penalty; at 0 it is least squares, and keeps all six.
#[test]
fn lasso_selection_golden() {
    let data = sparse_truth_rows();
    let mut m = Lasso::new(lasso_cfg(vec![0.25, 0.05, 0.0])).unwrap();
    let pred = predictions(&mut m, &data, 0);
    let beta = m.coefficients().unwrap()[0].clone();
    let active = |b: &[f64]| (1..7).filter(|&j| b[j] != 0.0).collect::<Vec<_>>();
    assert_eq!(active(&beta[0]), vec![1, 2], "{:?}", beta[0]);
    // The soft threshold shrinks each by the penalty, in the target's units.
    assert!((beta[0][1] - 1.25).abs() < 0.1, "{:?}", beta[0]);
    assert!((beta[0][2] + 0.75).abs() < 0.1, "{:?}", beta[0]);
    // At 0.05 the four are still below it (their `c_i` are about 0.01).
    assert_eq!(active(&beta[1]), vec![1, 2], "{:?}", beta[1]);
    assert_eq!(active(&beta[2]).len(), 6, "{:?}", beta[2]);
    let mut got = vec![pred[100], pred[200], pred[399]];
    for b in &beta {
        got.extend(b);
    }
    check("lasso_selection", &got, GOLDEN_LASSO_SELECTION);
}

const GOLDEN_LASSO_SELECTION: &[f64] = &[
    -0.32212993821737873,
    1.2787717555306322,
    2.0081035534254914,
    0.043427201070217004,
    1.2656584994663498,
    -0.7557307992364797,
    0.0,
    0.0,
    0.0,
    0.0,
    0.019180878329531675,
    1.4571238961201927,
    -0.9483869824075357,
    0.0,
    0.0,
    0.0,
    0.0,
    0.013122733440251208,
    1.5056201648528282,
    -0.9966220329989521,
    0.0031364629140597177,
    0.004133534024745721,
    -0.010177542973083202,
    0.008019023864554579,
];

/// One slope that drifts from 1 to 4 over 400 rows (`0.0075 t`), one that
/// does not (-0.5), noise 0.3, features of unit variance.
fn drifting_rows() -> Vec<Row> {
    rows(400, 1618, |i, s| {
        let x = vec![gauss(s), gauss(s)];
        let y = (1.0 + 0.0075 * i as f64) * x[0] - 0.5 * x[1] + 0.3 * gauss(s);
        (x, y)
    })
}

fn kalman_cfg(half_life: Vec<f64>) -> KalmanCfg {
    KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        half_life,
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: false,
    }
}

/// `kalman` on a slope that drifts, with a random walk of half-life 30 on
/// it and the intercept and the other slope pinned, against the same filter
/// with every coefficient pinned (a Bayesian regression that believes the
/// coefficients never move): the drifting filter ends within 0.5 of the
/// slope's 4.0, the pinned one near the stream's average, and the squared
/// error of the drifting one's predictions, over every row the two both
/// predict, is under half the pinned one's.
#[test]
fn kalman_drift_golden() {
    let data = drifting_rows();
    let mut walk = Kalman::new(kalman_cfg(vec![f64::INFINITY, 30.0, f64::INFINITY])).unwrap();
    let mut pinned = Kalman::new(kalman_cfg(vec![f64::INFINITY])).unwrap();
    let (pw, pp) = (
        predictions(&mut walk, &data, 0),
        predictions(&mut pinned, &data, 0),
    );
    let (mut sw, mut sp) = (0.0, 0.0);
    for ((_, y, ..), (a, b)) in data.iter().zip(pw.iter().zip(&pp)) {
        let y = y[0].unwrap();
        if a.is_finite() && b.is_finite() {
            sw += (y - a).powi(2);
            sp += (y - b).powi(2);
        }
    }
    let (cw, cp) = (
        walk.coefficients()[0].clone(),
        pinned.coefficients()[0].clone(),
    );
    assert!((cw[1] - 4.0).abs() < 0.5, "drifting {cw:?}");
    assert!(cp[1] < 3.0, "pinned {cp:?}");
    assert!(sw < sp / 2.0, "squared error {sw} against {sp}");
    let mut got = vec![pw[100], pw[200], pw[399]];
    got.extend(&cw);
    got.extend(&cp);
    check("kalman_drift", &got, GOLDEN_KALMAN_DRIFT);
}

/// 30 binary indicators, three on a row (a sparse design, as clicks on
/// an advertisement are), and a label that is a click with a probability
/// linear in three of them: `0.5 + 0.35 f0 - 0.35 f1 + 0.2 f2`, the other 27
/// irrelevant.
fn click_rows() -> Vec<Row> {
    rows(4000, 77, |_, s| {
        let mut x = vec![0.0; 30];
        for _ in 0..3 {
            let j = (((lcg(s) + 1.0) / 2.0) * 30.0) as usize;
            x[j.min(29)] = 1.0;
        }
        let p = 0.5 + 0.35 * x[0] - 0.35 * x[1] + 0.2 * x[2];
        let y = f64::from(u8::from((lcg(s) + 1.0) / 2.0 < p));
        (x, y)
    })
}

fn ftrl_cfg(l1: f64) -> FtrlCfg {
    FtrlCfg {
        n_features: 30,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        alpha: 0.5,
        beta: 1.0,
        l1,
        l2: 0.1,
        min_weight: 3.0,
        strict_binary: true,
        loss: FtrlLoss::Logistic,
        standardize: false,
    }
}

/// `ftrl` on the sparse indicators with an L1 penalty of 30 (against the
/// accumulated gradients, which grow with the rows): the three relevant
/// indicators carry weights of the right signs and the penalty holds all 27
/// others at exactly 0, where an L1 of 0 leaves every one with a weight.
#[test]
fn ftrl_sparse_golden() {
    let data = click_rows();
    let mut sparse = Ftrl::new(ftrl_cfg(30.0)).unwrap();
    let mut dense = Ftrl::new(ftrl_cfg(0.0)).unwrap();
    let pred = predictions(&mut sparse, &data, 0);
    predictions(&mut dense, &data, 0);
    let (cs, cd) = (
        sparse.coefficients()[0].clone(),
        dense.coefficients()[0].clone(),
    );
    let zeros = |c: &[f64]| c[4..].iter().filter(|w| **w == 0.0).count();
    assert!(cs[1] > 0.0 && cs[2] < 0.0 && cs[3] > 0.0, "{cs:?}");
    assert_eq!(zeros(&cs), 27, "irrelevant weights at 0: {cs:?}");
    assert_eq!(zeros(&cd), 0, "{cd:?}");
    let mut got = vec![pred[1000], pred[2000], pred[3999]];
    got.extend(&cs[..4]);
    got.push(zeros(&cs) as f64);
    check("ftrl_sparse", &got, GOLDEN_FTRL_SPARSE);
}

const GOLDEN_FTRL_SPARSE: &[f64] = &[
    0.5,
    0.50963036300188,
    0.5451469519431211,
    0.17247551202645914,
    1.5960184529436234,
    -1.7100990607910074,
    0.9104936083420349,
    27.0,
];
const GOLDEN_EWRIDGE_CONDITIONING: &[f64] = &[
    0.40812518674401055,
    -4.446439232867013,
    -1.5643694214655681,
    -0.012086658711636403,
    0.9652590286941913,
    1.0240279760723232,
    -0.014928660382763206,
    0.7269574125869644,
    1.3107116872268973,
];
const GOLDEN_KALMAN_DRIFT: &[f64] = &[
    0.6973725610168863,
    -2.7266044886508207,
    6.039499920348319,
    0.003690949546626069,
    3.6928498645336285,
    -0.4735677978181588,
    0.00962235476644814,
    2.617090362647303,
    -0.4682553429413175,
];

/// `y = 2 x0 - x1 + 0.5` with noise 0.3 on 4,000 rows of unit-variance
/// features.
fn long_rows() -> Vec<Row> {
    rows(4000, 424242, |_, s| {
        let x = vec![gauss(s), gauss(s)];
        let y = 2.0 * x[0] - x[1] + 0.5 + 0.3 * gauss(s);
        (x, y)
    })
}

fn sgd_cfg(schedule: LearningRate) -> SgdCfg {
    SgdCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        loss: SgdLoss::Squared,
        learning_rate: 0.1,
        schedule,
        l2: 0.0,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        strict_binary: false,
        min_weight: 3.0,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
    }
}

/// `sgd` over 4,000 rows with a decaying rate (`0.1 / (1 + n)^0.5`), where
/// the fixed stream's 60 rows leave the schedule at its start: the squared
/// error of the predictions over the last quarter is under a tenth of what
/// a constant rate of 0.1 leaves (its steady-state noise, since the step
/// never shrinks), and the decaying fit ends within 0.05 of the truth.
#[test]
fn sgd_later_rows_golden() {
    let data = long_rows();
    let mut decaying = Sgd::new(sgd_cfg(LearningRate::InvScaling { power: 0.5 })).unwrap();
    let mut constant = Sgd::new(sgd_cfg(LearningRate::Constant)).unwrap();
    let (pd, pc) = (
        predictions(&mut decaying, &data, 0),
        predictions(&mut constant, &data, 0),
    );
    let tail = |p: &[f64]| -> f64 {
        let from = 3 * data.len() / 4;
        data[from..]
            .iter()
            .zip(&p[from..])
            .map(|((x, ..), p)| (p - (2.0 * x[0] - x[1] + 0.5)).powi(2))
            .sum::<f64>()
            / (data.len() - from) as f64
    };
    let c = decaying.coefficients()[0].clone();
    for (got, want) in c.iter().zip([0.5, 2.0, -1.0]) {
        assert!((got - want).abs() < 0.05, "coefficients {c:?}");
    }
    assert!(
        tail(&pd) < tail(&pc) / 10.0,
        "{} against {}",
        tail(&pd),
        tail(&pc)
    );
    let mut got = vec![pd[100], pd[1000], pd[3999]];
    got.extend(&c);
    got.push(tail(&pc));
    check("sgd_later_rows", &got, GOLDEN_SGD_LATER_ROWS);
}

const GOLDEN_SGD_LATER_ROWS: &[f64] = &[
    -1.9027506861603674,
    5.192668933381674,
    -4.464291996188404,
    0.4991811187649941,
    1.998439464198432,
    -1.000020153224453,
    0.017763595796642762,
];
