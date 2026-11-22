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

/// Returns with variances 1 and a correlation of 0.7: 300 ticks.
fn return_rows() -> Vec<Row> {
    rows(300, 555, |_, s| {
        let (a, b) = (gauss(s), gauss(s));
        [a, 0.7 * a + (1.0f64 - 0.49).sqrt() * b]
    })
}

/// `rcov`'s kernel and pre-averaged estimates over one block of the return
/// stream: the three distinct entries of each, and the correlation.
#[test]
fn rcov_returns_golden() {
    let cfg = |kind, kernel_bw, preavg| RcovCfg {
        n_features: 2,
        kind,
        kernel: "parzen".into(),
        bandwidth: kernel_bw,
        jitter: 2,
        theta: 1.0,
        psd: false,
        block_rows: Some(300),
        max_bandwidth: None,
        preavg_rows: preavg,
        noise_stride: 1,
        iv_stride: 20,
    };
    let mut got = Vec::new();
    for c in [
        cfg(RcovKind::Kernel, Some(3), None),
        cfg(RcovKind::Preavg, None, Some(6)),
    ] {
        let mut m = Rcov::new(c).unwrap();
        for (x, y, d, w) in return_rows() {
            m.step(&x, &y, d, w);
        }
        let est = m.estimate();
        let cov = est.rcov.expect("a block");
        assert!(
            (cov[0] / 300.0 - 1.0).abs() < 0.3,
            "variance {}",
            cov[0] / 300.0
        );
        let corr = est.rcorr.expect("a block");
        assert!(
            (corr[1] - 0.7).abs() < 0.12,
            "correlation {}, true 0.7",
            corr[1]
        );
        got.extend([cov[0], cov[1], cov[3], corr[1]]);
    }
    check("rcov_returns", &got, GOLDEN_RCOV_RETURNS);
}

/// A Hawkes process of baseline 0.4, branching ratio 0.5 and kernel decay
/// 1.5, simulated by Ogata's thinning: 300 events, each a row on the clock
/// of mark 1.
fn hawkes_rows() -> Vec<Row> {
    let (mu, alpha, beta) = (0.4f64, 0.5f64, 1.5f64);
    let mut s = 1234567u64;
    let uniform = |s: &mut u64| (lcg(s) + 1.0) / 2.0;
    let (mut excite, mut last, mut now) = (0.0f64, 0.0f64, 0.0f64);
    let mut out: Vec<Row> = Vec::new();
    while out.len() < 300 {
        // The intensity only falls between events, so its value now bounds
        // it until the next one.
        let bound = mu + excite * (-beta * (now - last)).exp();
        now += -uniform(&mut s).max(1e-300).ln() / bound;
        let here = mu + excite * (-beta * (now - last)).exp();
        if uniform(&mut s) * bound <= here {
            excite = excite * (-beta * (now - last)).exp() + alpha * beta;
            let d = if out.is_empty() { 0.0 } else { now - last };
            last = now;
            out.push(([0.0, 0.0], [Some(1.0)], d, 1.0));
        }
    }
    out
}

/// `hawkes` fitted to the simulated process by one kernel at the true
/// decay, with no forgetting: the intensity at rows 100, 200 and 299 and
/// the fit `(mu, alpha)`, then two kernels under forgetting.
#[test]
fn hawkes_process_golden() {
    let cfg = |decays: Vec<f64>, half_life| HawkesCfg {
        n_targets: 1,
        kernel_decays: decays,
        decay: Decay::Halflife(half_life),
        min_weight: 0.0,
        window_events: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    };
    let data: Vec<Row> = hawkes_rows();
    let run = |m: &mut Hawkes| -> Vec<f64> {
        let mut got = Vec::new();
        for (i, (x, y, d, w)) in data.iter().enumerate() {
            let step = m.step(&x[..0], y, *d, *w);
            if matches!(i, 100 | 200 | 299) {
                got.push(step.pred[0]);
            }
        }
        got.extend(&m.coefficients()[0]);
        got
    };
    let mut m = Hawkes::new(cfg(vec![1.5], f64::INFINITY)).unwrap();
    let mut got = run(&mut m);
    let (mu, alpha) = (got[3], got[4]);
    assert!((mu - 0.4).abs() < 0.15, "baseline {mu}, true 0.4");
    assert!((alpha - 0.5).abs() < 0.2, "excitation {alpha}, true 0.5");
    let mut m = Hawkes::new(cfg(vec![0.8, 3.0], 150.0)).unwrap();
    got.extend(run(&mut m));
    check("hawkes_process", &got, GOLDEN_HAWKES_PROCESS);
}

/// `n` rows of unit weight on a unit clock with a target: `draw` gives the
/// features and the target of row `i`.
fn target_rows(
    n: usize,
    seed: u64,
    mut draw: impl FnMut(usize, &mut u64) -> ([f64; 2], f64),
) -> Vec<Row> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            let (x, y) = draw(i, &mut s);
            (x, [Some(y)], if i == 0 { 0.0 } else { 1.0 }, 1.0)
        })
        .collect()
}

/// The mean of `pred[0]` minus `truth(x)` over the rows from `from` on, with
/// the target never read before it is predicted.
fn mean_error<M: OnlineModel>(
    model: &mut M,
    rows: &[Row],
    truth: impl Fn(&[f64; 2]) -> f64,
    from: usize,
) -> f64 {
    let mut sum = 0.0;
    for (i, (x, y, d, w)) in rows.iter().enumerate() {
        let step = model.step(x, y, *d, *w);
        if i >= from {
            sum += step.pred[0] - truth(x);
        }
    }
    sum / (rows.len() - from) as f64
}

fn line(x: &[f64; 2]) -> f64 {
    1.5 * x[0] - 0.75 * x[1] + 0.25
}

fn robust_cfg(loss: RobustLoss, standardize: bool, decay: Decay) -> RobustCfg {
    robust_cfg_scaled(loss, standardize, decay, rheon_core::HuberScale::Rms)
}

fn robust_cfg_scaled(
    loss: RobustLoss,
    standardize: bool,
    decay: Decay,
    huber_scale: rheon_core::HuberScale,
) -> RobustCfg {
    RobustCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay,
        loss,
        ridge: 1e-4,
        standardize,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        mad_anchor: false,
        min_weight_holds_learning: false,
        refit_clock: None,
        max_state_bytes: None,
    }
}

/// The line with noise of 0.2 and, on 8% of the rows, a one-sided outlier of
/// +4 (twenty standard deviations).
fn contaminated_rows() -> Vec<Row> {
    target_rows(300, 6151, |_, s| {
        let x = [lcg(s), lcg(s)];
        let mut y = line(&x) + 0.2 * gauss(s);
        if (lcg(s) + 1.0) / 2.0 < 0.08 {
            y += 4.0;
        }
        (x, y)
    })
}

/// `huber` on the contaminated line, the golden's cut of 1.5 and no
/// forgetting: the predictions at rows 100, 200 and 299, and the mean error
/// over the last 100 rows, which is held under a third of the mean error of
/// a least-squares fit on the same rows (the outliers shift it by their
/// share of the shift, 0.32; a Huber fit's cut leaves 0.05 or so).
#[test]
fn huber_contaminated_golden() {
    let cfg = || {
        robust_cfg_scaled(
            RobustLoss::Huber { delta: 1.5 },
            false,
            Decay::Halflife(f64::INFINITY),
            rheon_core::HuberScale::Mad,
        )
    };
    let rows = contaminated_rows();
    let mut m = Robust::new(cfg()).unwrap();
    let mut got = probe(&mut m, &rows, &[0], &[100, 200, 299]);
    let robust = mean_error(&mut Robust::new(cfg()).unwrap(), &rows, line, 200);
    let ols = mean_error(&mut ols_model(), &rows, line, 200);
    assert!(ols > 0.2, "the outliers move least squares: {ols}");
    assert!(
        robust.abs() < ols / 3.0,
        "huber {robust}, least squares {ols}"
    );
    got.push(robust);
    check("huber_contaminated", &got, GOLDEN_HUBER_CONTAMINATED);
}

/// Least squares over the same rows: `ewridge` with no forgetting.
fn ols_model() -> EwRidge {
    EwRidge::new(EwRidgeCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        ridge: vec![1e-6],
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
    })
    .unwrap()
}

/// `Exp(1)` as half a chi-square on two degrees of freedom: two normals,
/// sums and products only.
fn exp1(s: &mut u64) -> f64 {
    let (a, b) = (gauss(s), gauss(s));
    0.5 * (a * a + b * b)
}

/// The line with noise `Exp(1) - 1` of scale 0.8: mean 0, skewed to the right.
fn skewed_rows() -> Vec<Row> {
    target_rows(600, 880011, |_, s| {
        let x = [lcg(s), lcg(s)];
        (x, line(&x) + 0.8 * (exp1(s) - 1.0))
    })
}

/// `quantile` at 0.7 on skewed noise with no forgetting: the fit sits above
/// the mean's line by the noise's 0.7 quantile, `0.8 (ln(1/0.3) - 1)`, which
/// is 0.16 here and not the 0 a symmetric noise would give.
#[test]
fn quantile_skewed_golden() {
    let rows = skewed_rows();
    let cfg = || {
        robust_cfg(
            RobustLoss::Quantile { tau: 0.7 },
            true,
            Decay::Halflife(f64::INFINITY),
        )
    };
    let mut m = Robust::new(cfg()).unwrap();
    let mut got = probe(&mut m, &rows, &[0], &[200, 400, 599]);
    let shift = mean_error(&mut Robust::new(cfg()).unwrap(), &rows, line, 300);
    let want = 0.8 * ((1.0f64 / 0.3).ln() - 1.0);
    assert!(
        (shift - want).abs() < 0.15,
        "shift {shift}, the quantile's {want}"
    );
    got.push(shift);
    check("quantile_skewed", &got, GOLDEN_QUANTILE_SKEWED);
}

/// `expectile` at 0.85 on the same noise, no forgetting: the fit sits above
/// the mean's line by the noise's 0.85 expectile, the root `e` of `0.85
/// exp(-e) = 0.15 (e - 1 + exp(-e))` for `Exp(1)`, less 1, times 0.8, found
/// here by bisection.
#[test]
fn expectile_skewed_golden() {
    let rows = skewed_rows();
    let cfg = ExpectileCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        expectile: 0.85,
        ridge: 1e-6,
        min_weight: 3.0,
        refit_rows: 0,
        refit_clock: None,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    };
    let mut m = Expectile::new(cfg.clone()).unwrap();
    let mut got = probe(&mut m, &rows, &[0], &[200, 400, 599]);
    let shift = mean_error(&mut Expectile::new(cfg).unwrap(), &rows, line, 300);
    let balance = |e: f64| 0.85 * (-e).exp() - 0.15 * (e - 1.0 + (-e).exp());
    let (mut lo, mut hi) = (0.0f64, 10.0f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if balance(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let want = 0.8 * (lo - 1.0);
    assert!(
        (shift - want).abs() < 0.1,
        "shift {shift}, the expectile's {want}"
    );
    got.push(shift);
    check("expectile_skewed", &got, GOLDEN_EXPECTILE_SKEWED);
}

/// A Gamma amount of shape 4 and mean `exp(0.5 + 0.8 x0 - 0.4 x1)`: the
/// mean times half the sum of eight squared normals over 4.
fn gamma_rows() -> Vec<Row> {
    target_rows(500, 4040, |_, s| {
        let x = [lcg(s), lcg(s)];
        let mean = (0.5 + 0.8 * x[0] - 0.4 * x[1]).exp();
        let chi: f64 = (0..8).map(|_| gauss(s).powi(2)).sum();
        (x, mean * chi / 8.0)
    })
}

/// A Gamma `glm` with a log link on draws from that family, no forgetting
/// and a small ridge: the mean at three rows and the coefficients, which are
/// held to the truth within the sampling error of 500 rows (shape 4).
#[test]
fn glm_gamma_golden() {
    let mut m = Glm::new(GlmCfg {
        n_features: 2,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(f64::INFINITY),
        family: GlmFamily::Gamma,
        link: GlmLink::Log,
        var_power: 1.5,
        ridge: 1e-3,
        min_weight: 3.0,
        refit_rows: 15,
        refit_clock: None,
        step_cap: f64::INFINITY,
        max_state_bytes: None,
    })
    .unwrap();
    let mut got = probe(&mut m, &gamma_rows(), &[0], &[200, 400, 499]);
    let coef = m.coefficients()[0].clone();
    for (c, t) in coef.iter().zip([0.5, 0.8, -0.4]) {
        assert!(
            (c - t).abs() < 0.15,
            "coefficients {coef:?}, truth (0.5, 0.8, -0.4)"
        );
    }
    got.extend(&coef);
    check("glm_gamma", &got, GOLDEN_GLM_GAMMA);
}

/// A series with a trend: `10 + 0.5 t` and noise of 0.5. `holt` reads the
/// target alone.
fn trend_rows() -> Vec<Row> {
    target_rows(120, 2468, |i, s| {
        ([0.0, 0.0], 10.0 + 0.5 * i as f64 + 0.5 * gauss(s))
    })
}

/// `holt` on the trending series, the golden's two half-lives: the forecast
/// at rows 40, 80 and 119 and the level and trend at the end. The trend is
/// found: its estimate is within 0.15 of the slope 0.5, and the forecast is
/// within 2 of the line.
#[test]
fn holt_trend_golden() {
    let mut m = Holt::new(HoltCfg {
        n_targets: 1,
        level_half_life: 10.0,
        trend_half_life: 40.0,
        min_weight: 3.0,
        trend: true,
    })
    .unwrap();
    let rows = trend_rows();
    let mut got = Vec::new();
    for (i, (x, y, d, w)) in rows.iter().enumerate() {
        let step = m.step(&x[..0], y, *d, *w);
        if matches!(i, 40 | 80 | 119) {
            assert!(
                (step.pred[0] - (10.0 + 0.5 * i as f64)).abs() < 2.0,
                "row {i}"
            );
            got.push(step.pred[0]);
        }
    }
    let coef = m.coefficients()[0].clone();
    assert!(
        (coef.last().copied().unwrap() - 0.5).abs() < 0.15,
        "coefficients {coef:?}"
    );
    got.extend(&coef);
    check("holt_trend", &got, GOLDEN_HOLT_TREND);
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
const GOLDEN_RCOV_RETURNS: &[f64] = &[
    324.1450477074828,
    270.7530496893218,
    358.2524549213687,
    0.7945277573015966,
    352.2796004452549,
    283.81883050579927,
    362.2519795198528,
    0.7944966584471378,
];
const GOLDEN_HAWKES_PROCESS: &[f64] = &[
    0.38134630118736423,
    1.5863115646818613,
    1.2528902701167157,
    0.4687148251549859,
    0.4652004858618119,
    0.3339130879246461,
    1.5719619973726842,
    1.2565531910258125,
    0.47986354169547724,
    0.32679293101720874,
    0.15354451470316674,
];
const GOLDEN_HUBER_CONTAMINATED: &[f64] = &[
    -0.0673238035897995,
    0.301095749294845,
    0.22378708973246048,
    0.04014623471851346,
];
const GOLDEN_QUANTILE_SKEWED: &[f64] = &[
    0.01130043329129038,
    -0.14877383228171126,
    0.6511296088025674,
    0.2649190652119157,
];
const GOLDEN_EXPECTILE_SKEWED: &[f64] = &[
    0.48561188197826677,
    0.23346226377963325,
    1.0415184309502417,
    0.6527315437575731,
];
const GOLDEN_GLM_GAMMA: &[f64] = &[
    1.3208673785014653,
    0.8341742966636372,
    1.8477776311924323,
    0.5019867023301244,
    0.742887826853017,
    -0.43220700417823393,
];
const GOLDEN_HOLT_TREND: &[f64] = &[
    29.994442248160894,
    50.06146877462335,
    69.47081536438878,
    69.52337608339845,
    0.5001413372772974,
];
