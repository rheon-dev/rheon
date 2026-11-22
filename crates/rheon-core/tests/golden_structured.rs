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
