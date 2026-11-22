//! Golden values on streams that have something to find.
//!
//! `golden.rs` feeds every model the same 60 rows of independent uniform
//! features. That is right for a regression, whose target is a line in them,
//! and wrong for a model whose job is to find structure: a clusterer has no
//! clusters to find there, a regime model no regimes, a change detector no
//! change, `deco` and `rcov` nothing but a diagonal, and `hawkes` no
//! excitation. Its signatures for those models pin the arithmetic of a fit
//! that has nothing to fit, so a change that moves a number only when the
//! data has the structure (a split of a cluster, a switch of a regime, a
//! run that restarts, a kernel that excites) moves none of them.
//!
//! One stream a model here, built from the structure the model looks for and
//! generated from a seed with sums and products only, so the rows are the
//! same bits on every platform (the one exception, `hawkes`'s simulation,
//! reads `ln` and `exp`, whose last bit moves no signature past the
//! tolerance). Each test asserts first that the model found the structure
//! (the clusters the blobs are, the run length since the shift, the
//! correlation the factor makes, the parameters the simulation used), and
//! then pins the signature, which is what makes the stream well posed
//! rather than merely different: a number that moved because the structure
//! was lost fails the first assertion with a reason, not a diff.
//!
//! Regenerate with `PRINT_GOLDEN=1 cargo test -p rheon-core --test
//! golden_structured -- --nocapture <test>`, one test a run, and only after
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

/// One row: features, targets, clock step, row weight.
type Row = ([f64; 2], [Option<f64>; 1], f64, f64);

/// `n` rows of unit weight on a unit clock, row `i` drawn by `draw`.
fn rows(n: usize, seed: u64, mut draw: impl FnMut(usize, &mut u64) -> [f64; 2]) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|i| (draw(i, &mut s), [None], if i == 0 { 0.0 } else { 1.0 }, 1.0))
        .collect()
}

/// For each row of `at`, the output slots of `picks` after that row.
fn probe<M: OnlineModel>(model: &mut M, rows: &[Row], picks: &[usize], at: &[usize]) -> Vec<f64> {
    let mut out = Vec::new();
    for (i, (x, y, d, w)) in rows.iter().enumerate() {
        let step = model.step(x, y, *d, *w);
        if at.contains(&i) {
            out.extend(picks.iter().map(|&p| step.pred[p]));
        }
    }
    out
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

/// Three blobs of standard deviation 0.25 round the corners of a triangle,
/// in turn: row `i` is from blob `i % 3`.
fn blob_rows() -> Vec<Row> {
    const CENTRES: [[f64; 2]; 3] = [[-1.5, -1.0], [1.5, -1.0], [0.0, 1.8]];
    rows(60, 20241122, |i, s| {
        let c = CENTRES[i % 3];
        [c[0] + 0.25 * gauss(s), c[1] + 0.25 * gauss(s)]
    })
}

/// Rows read at 45, 46, 47 and 57, 58, 59: two visits to each of the three
/// blobs, after the warm start and the seeding are long past.
const BLOB_AT: [usize; 6] = [45, 46, 47, 57, 58, 59];

fn kmeans_cfg() -> KMeansCfg {
    KMeansCfg {
        n_features: 2,
        k: 3,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        warm_rows: 12,
        seed_rule: SeedRule::Lloyd,
        seed: 0,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 10,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.0,
        warm_clock: None,
        update_every: None,
        split_merge_every: None,
    }
}

/// `kmeans` on three separated blobs: slot 0 (the cluster) and slot 1 (the
/// distance to its centre under the standardized metric) at six rows.
#[test]
fn kmeans_blobs_golden() {
    let mut m = KMeans::new(kmeans_cfg()).unwrap();
    let got = probe(&mut m, &blob_rows(), &[0, 1], &BLOB_AT);
    // The model found the blobs: the three rows of a visit are three
    // clusters, and the second visit is the same three, in the same order.
    let cluster = |r: usize| got[2 * r];
    assert!(cluster(0) != cluster(1) && cluster(1) != cluster(2) && cluster(0) != cluster(2));
    for r in 0..3 {
        assert_eq!(cluster(r), cluster(r + 3), "blob {r} changed cluster");
    }
    // A row sits within a fraction of a standard deviation of the feature
    // (about 1.3) of its blob's centre, not across the triangle.
    for r in 0..6 {
        let dist = got[2 * r + 1];
        assert!(dist > 0.0 && dist < 0.6, "row {r}: distance {dist}");
    }
    check("kmeans_blobs", &got, GOLDEN_KMEANS_BLOBS);
}

fn micro_cfg() -> MicroCfg {
    MicroCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        min_weight: 3.0,
        eps: 0.35,
        beta_mu: 2.0,
        max_clusters: 8,
        prune_every: f64::INFINITY,
        max_rows_between_prunes: 7,
        macro_link: None,
        standardize: true,
        scale_floor: 0.0,
    }
}

/// `micro` on the same blobs: the macro label (slot 0), the distance to the
/// nearest potential summary (slot 1) and its id (slot 2).
#[test]
fn micro_blobs_golden() {
    let mut m = Micro::new(micro_cfg()).unwrap();
    let got = probe(&mut m, &blob_rows(), &[0, 1, 2], &BLOB_AT);
    // Three blobs, three micro-clusters: the ids of a visit differ and the
    // next visit finds the same ones.
    let id = |r: usize| got[3 * r + 2];
    assert!(id(0) != id(1) && id(1) != id(2) && id(0) != id(2));
    for r in 0..3 {
        assert_eq!(id(r), id(r + 3), "blob {r} changed summary");
    }
    for r in 0..6 {
        let dist = got[3 * r + 1];
        assert!(dist > 0.0 && dist < 1.0, "row {r}: {dist}");
    }
    check("micro_blobs", &got, GOLDEN_MICRO_BLOBS);
}

/// `micro_blobs_golden` with the macro step's threshold given.
#[test]
fn micro_blobs_linked_golden() {
    // The linkage threshold derived from the spacing of the summaries reads
    // three summaries that are all far apart as one shape, and labels them
    // alike (slot 0 above is 0 throughout); a threshold of 2 (in units of
    // `eps sqrt(p)`, 1.0 in the metric, short of the 2.3 between blobs)
    // keeps them three clusters.
    let mut m = Micro::new(MicroCfg {
        macro_link: Some(2.0),
        ..micro_cfg()
    })
    .unwrap();
    let got = probe(&mut m, &blob_rows(), &[0], &BLOB_AT);
    assert!(got[0] != got[1] && got[1] != got[2] && got[0] != got[2]);
    assert_eq!(got[..3], got[3..], "a blob changed label");
    check("micro_blobs_linked", &got, GOLDEN_MICRO_BLOBS_LINKED);
}

/// Two regimes of 15 rows, round and round: row `i` is from regime
/// `(i / 15) % 2`, a cloud of standard deviation 0.5 at -1.5 or at +1.5 in
/// both features.
fn regime_rows() -> Vec<Row> {
    rows(60, 7771, |i, s| {
        let m = if (i / 15) % 2 == 0 { -1.5 } else { 1.5 };
        [m + 0.5 * gauss(s), m + 0.5 * gauss(s)]
    })
}

/// `hmm` with two states told only roughly where they are (means at +-1,
/// unit covariance, as a starting guess) and left to learn: the state it
/// reports (slot 4) and the log-likelihood of the row (slot 5), at rows
/// 10 and 40 (regime 0), 20 and 55 (regime 1).
#[test]
fn hmm_regimes_golden() {
    let mut m = Hmm::new(HmmCfg {
        n_features: 2,
        k: 2,
        decay: Decay::Halflife(20.0),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_weight: 3.0,
        learn: true,
        transition_prior: 1.0,
        transition: Some(vec![0.9, 0.1, 0.1, 0.9]),
        means: Some(vec![-1.0, -1.0, 1.0, 1.0]),
        covs: Some(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
    })
    .unwrap();
    let got = probe(&mut m, &regime_rows(), &[4, 5], &[10, 20, 40, 55]);
    // The state follows the regime, one row at a time out of sample.
    let state = |r: usize| got[2 * r];
    assert_eq!(state(0), state(2), "regime 0 is one state");
    assert_eq!(state(1), state(3), "regime 1 is one state");
    assert_ne!(state(0), state(1), "the regimes are two states");
    check("hmm_regimes", &got, GOLDEN_HMM_REGIMES);
}

/// A level shift of five standard deviations at row 30: both features from
/// N(0, 0.5^2) before, from N((3, -3), 0.5^2) after.
fn shift_rows() -> Vec<Row> {
    rows(60, 31337, |i, s| {
        let (a, b) = if i < 30 { (0.0, 0.0) } else { (3.0, -3.0) };
        [a + 0.5 * gauss(s), b + 0.5 * gauss(s)]
    })
}

fn bocpd_cfg(on_the_clock: bool) -> BocpdCfg {
    BocpdCfg {
        n_features: 2,
        hazard: 30.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: Some(vec![0.0, 0.0]),
        prior_kappa: 1.0,
        prior_nu: Some(2.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-8,
        max_run: 100,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: on_the_clock,
        warm_clock: None,
    }
}

/// `bocpd` on the shift: `p_change` (slot 0) and the most likely run
/// length (slot 1) at rows 20 and 29 (before), 30, 31, 32 (the shift) and
/// 59 (a run of 30 after it).
#[test]
fn bocpd_shift_golden() {
    let mut m = Bocpd::new(bocpd_cfg(false)).unwrap();
    let at = [20, 29, 30, 31, 32, 59];
    let got = probe(&mut m, &shift_rows(), &[0, 1], &at);
    let p = |r: usize| got[2 * r];
    let run = |r: usize| got[2 * r + 1];
    // Quiet before the shift, a change called within three rows of it, and
    // a run that counts from it after it.
    assert!(
        p(0) < 0.2 && p(1) < 0.2,
        "a change called in a quiet stretch"
    );
    assert!(p(2).max(p(3)).max(p(4)) > 0.5, "the shift is not called");
    assert!((run(5) - 29.0).abs() <= 3.0, "run length {} at 59", run(5));
    check("bocpd_shift", &got, GOLDEN_BOCPD_SHIFT);
}

/// The same shift with the hazard on the clock, and the shift's own step
/// the long one: a row 20 clock units after its predecessor is likelier to
/// be a change than one 1 unit after.
#[test]
fn bocpd_shift_on_the_clock_golden() {
    let mut rows = shift_rows();
    rows[30].2 = 20.0;
    let mut m = Bocpd::new(bocpd_cfg(true)).unwrap();
    let got = probe(&mut m, &rows, &[0, 1], &[29, 30, 31, 32, 59]);
    check("bocpd_shift_clock", &got, GOLDEN_BOCPD_SHIFT_CLOCK);
}

/// Two features whose correlation is +0.9 for rows 0 to 249 and -0.9 from
/// row 250 on, each of variance 1.
// The seed is the second tried: the first drew a stable span whose
// statistic was 3.9 against a critical value of 1.36 (the test's size is
// 5%, and the draw was one in which the early rows of the span read a
// lower correlation than the rest; a replay in numpy gave the same number).
fn break_rows() -> Vec<Row> {
    rows(300, 8675309, |i, s| {
        let rho: f64 = if i < 250 { 0.9 } else { -0.9 };
        let (a, b) = (gauss(s), gauss(s));
        [a, rho * a + (1.0 - rho * rho).sqrt() * b]
    })
}

fn corrchange_cfg() -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: 2,
        kind: CorrChangeKind::Monitor,
        span_rows: 100,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: Decay::Halflife(20.0),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 0,
        boundary_gamma: 0.0,
        permute_every: None,
    }
}

/// `corrchange` on a break in the correlation: the statistic, its critical
/// value and the flag (slots 0 to 2) at the three span ends of 100 rows. The
/// break comes at row 250, inside the third span, so the first two test a
/// correlation that does not move and the third one that reverses.
#[test]
fn corrchange_break_golden() {
    let mut m = CorrChange::new(corrchange_cfg()).unwrap();
    let got = probe(&mut m, &break_rows(), &[0, 1, 2], &[99, 199, 299]);
    // The first two spans hold a correlation that does not move, the third
    // one that reverses half way: only it is flagged.
    assert_eq!([got[2], got[5], got[8]], [0.0, 0.0, 1.0], "flags {got:?}");
    check("corrchange_break", &got, GOLDEN_CORRCHANGE_BREAK);
}

/// One factor: `x_i = sqrt(rho) f + sqrt(1 - rho) e_i` for `rho = 0.6`, so
/// every pair of features has a correlation of 0.6.
fn factor_rows(n: usize) -> Vec<Row> {
    rows(n, 99, |_, s| {
        let f = gauss(s);
        let (a, b) = (gauss(s), gauss(s));
        let (lo, hi) = (0.6f64.sqrt(), 0.4f64.sqrt());
        [lo * f + hi * a, lo * f + hi * b]
    })
}

/// `deco` on the factor stream, 150 rows: `rho` (slot 1) from the busier
/// `"linear"` dynamics and `loglik` (slot 2) from `"ew"`, at rows 100, 125
/// and 149; the mean `rho` over the last 50 rows is held to the value the
/// estimator has on this stream (below).
#[test]
fn deco_factor_golden() {
    let cfg = |dynamics, alpha, beta| DecoCfg {
        n_features: 2,
        decay: Decay::Halflife(20.0),
        dynamics,
        alpha,
        beta,
        blocks: Vec::new(),
        min_weight: 3.0,
    };
    let data = factor_rows(150);
    let mut m = Deco::new(cfg(DecoDynamics::Linear, Some(0.05), Some(0.9))).unwrap();
    let mut rho_tail = 0.0;
    let mut got = Vec::new();
    for (i, (x, y, d, w)) in data.iter().enumerate() {
        let step = m.step(x, y, *d, *w);
        if i >= 100 {
            rho_tail += step.pred[1] / 50.0;
        }
        if matches!(i, 100 | 125 | 149) {
            got.push(step.pred[1]);
        }
    }
    let mut m = Deco::new(cfg(DecoDynamics::Ew, None, None)).unwrap();
    let mut ew_tail = 0.0;
    for (i, (x, y, d, w)) in data.iter().enumerate() {
        let step = m.step(x, y, *d, *w);
        if i >= 100 {
            ew_tail += step.pred[1] / 50.0;
        }
    }
    // `rho` is the average of the row estimate `u = 2 r0 r1 / (r0^2 +
    // r1^2)` of the paper's Lemma 2.3, which is biased below the correlation
    // (the paper says so): for two unit normals of correlation `r` it
    // averages `(1 - sqrt(1 - r^2)) / r`, 1/3 at 0.6. Both dynamics sit
    // there, within the sampling error of an average over about 29 rows.
    let biased = (1.0 - (1.0f64 - 0.36).sqrt()) / 0.6;
    assert!(
        (rho_tail - biased).abs() < 0.1,
        "linear: mean rho {rho_tail}, expected {biased}"
    );
    assert!(
        (ew_tail - biased).abs() < 0.1,
        "ew: mean rho {ew_tail}, expected {biased}"
    );
    let mut m = Deco::new(cfg(DecoDynamics::Ew, None, None)).unwrap();
    got.extend(probe(&mut m, &data, &[2], &[100, 125, 149]));
    check("deco_factor", &got, GOLDEN_DECO_FACTOR);
}

// --- generated; see the module docs ---
const GOLDEN_KMEANS_BLOBS: &[f64] = &[
    2.0,
    0.4647144589082067,
    0.0,
    0.4554685388493479,
    1.0,
    0.16493811854175303,
    2.0,
    0.06422529915535433,
    0.0,
    0.11622676448637302,
    1.0,
    0.3482570523632956,
];
const GOLDEN_MICRO_BLOBS: &[f64] = &[
    0.0,
    0.4647144589082067,
    0.0,
    0.0,
    0.4554685388493479,
    1.0,
    0.0,
    0.16493811854175303,
    2.0,
    0.0,
    0.06422529915535434,
    0.0,
    0.0,
    0.11622676448637301,
    1.0,
    0.0,
    0.3482570523632956,
    2.0,
];
const GOLDEN_MICRO_BLOBS_LINKED: &[f64] = &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
const GOLDEN_HMM_REGIMES: &[f64] = &[
    0.0,
    -2.033795937365689,
    1.0,
    -0.9740368513225234,
    0.0,
    -1.711628768195652,
    1.0,
    -2.8097398706368955,
];
const GOLDEN_BOCPD_SHIFT: &[f64] = &[
    0.04203844949534123,
    20.0,
    0.04068620572104181,
    29.0,
    0.9701692473882619,
    30.0,
    0.03480958907139815,
    1.0,
    0.03413359878751515,
    2.0,
    0.033404090060154275,
    29.0,
];
const GOLDEN_BOCPD_SHIFT_CLOCK: &[f64] = &[
    0.007475332411326339,
    29.0,
    0.9988627220765844,
    0.0,
    0.0014787730019219837,
    1.0,
    0.0008086697940238442,
    2.0,
    7.194893573234455e-5,
    29.0,
];
const GOLDEN_CORRCHANGE_BREAK: &[f64] = &[
    0.8466760494398093,
    1.35809863932255,
    0.0,
    0.9617703258967044,
    1.35809863932255,
    0.0,
    1.5528196731668689,
    1.35809863932255,
    1.0,
];
const GOLDEN_DECO_FACTOR: &[f64] = &[
    0.3839818221187832,
    0.39547411226222373,
    0.2576146788192692,
    -7.633940013970986,
    -2.629961732725814,
    -2.1077895811748193,
];
