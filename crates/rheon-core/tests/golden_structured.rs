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
