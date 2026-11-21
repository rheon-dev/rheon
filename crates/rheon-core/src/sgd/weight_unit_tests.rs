//! The scaler's unit of weight under `standardize` (`scaler_decay`,
//! `Warmup::row_weight`; review C1): `sgd` and `pa` read a row at one row's
//! weight of the stream, `ŵ = Σw²/Σw` of the warm-up's rows, while the
//! scaler warms up and at `min(ŵ, w)` once the fit is held, and the history
//! keeps `ŵ·n/(n + 1)` and then `ŵ` into a row. A row read at weight 1
//! against moments in the weights' units was a share `b` of the moments that
//! grew as the weights shrank: at weights of 1/8 and 8 features AdaGrad's
//! intercept was pushed the wrong way on every row (an error of 1,140 noise
//! variances, where weights of 1 give 0.184).
//!
//! The streams and bands are those of the check that settled the rule
//! (`.claude/coord/review-fable2/C1-RULE-A.md`, harness `c1-rule-a/lab_main.rs`):
//! `gap_tests`' features and truth, noise sd 0.5, a half-life of 20 rows
//! unless said, errors `(p − f)²/0.25` in noise variances. Weight scales
//! are powers of two, so every weight scales exactly: Kish's count of
//! weights not a power of two can round under 22 (21.999999999999996 at a
//! constant 0.1), which moves the switch by a row, a difference of its own.

use super::gap_tests::{HL, Lab, Rng, cfg, irregular, lab, largest_held_step, p99, pa_cfg};
use super::*;
use crate::{Pa, PaCfg};

/// The weight patterns of the check: `unif` 0.5 to 1.5; `zeros` the same
/// with one row in 20 of weight 0; `ln1` lognormal with sd 1 and mean 1;
/// `spike` one row in 100 at 100 times the rest, mean about 1; `down`
/// falling 100-fold over the stream.
fn weights(pat: &str, n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng(seed ^ 0x5EED);
    (0..n)
        .map(|t| match pat {
            "const" => 1.0,
            "unif" => 0.5 + r.u(),
            "zeros" => {
                let (a, b) = (r.u(), r.u());
                if a < 0.05 { 0.0 } else { 0.5 + b }
            }
            "ln1" => (r.n() - 0.5).exp(),
            "spike" => {
                if r.u() < 0.01 {
                    100.0 / 1.99
                } else {
                    1.0 / 1.99
                }
            }
            "down" => 100f64.powf(-(t as f64) / n as f64),
            _ => unreachable!("{pat}"),
        })
        .collect()
}

#[derive(Clone)]
enum Learner {
    Sgd(Box<Sgd>),
    Pa(Box<Pa>),
}

impl Learner {
    /// The check's learners: a constant rate of 0.01, AdaGrad at 0.1, and
    /// `pa` (PA-I, `C` 1, a tube of 0.01 spreads), each under a half-life
    /// of `hl` rows.
    fn new(name: &str, k: usize, hl: f64) -> Self {
        let decay = Decay::Halflife(hl);
        match name {
            "constant" => Self::Sgd(Box::new(
                Sgd::new(SgdCfg {
                    decay,
                    ..cfg(k, 0.01, LearningRate::Constant)
                })
                .unwrap(),
            )),
            "adagrad" => Self::Sgd(Box::new(
                Sgd::new(SgdCfg {
                    decay,
                    ..cfg(k, 0.1, LearningRate::AdaGrad)
                })
                .unwrap(),
            )),
            _ => Self::Pa(Box::new(Pa::new(PaCfg { decay, ..pa_cfg(k) }).unwrap())),
        }
    }

    fn step(&mut self, x: &[f64], y: f64, d: f64, w: f64) -> Step {
        match self {
            Self::Sgd(m) => m.step(x, &[Some(y)], d, w),
            Self::Pa(m) => m.step(x, &[Some(y)], d, w),
        }
    }
}

/// Every prediction of `name` over the stream at weights `ws` times
/// `scale`, a row a clock unit.
fn preds(name: &str, k: usize, hl: f64, l: &Lab, ws: &[f64], scale: f64) -> Vec<f64> {
    let mut m = Learner::new(name, k, hl);
    (0..l.y.len())
        .map(|t| {
            let d = if t == 0 { 0.0 } else { 1.0 };
            m.step(&l.x[t], l.y[t], d, ws[t] * scale).pred[0]
        })
        .collect()
}

/// The mean error over rows 500 on, in noise variances.
fn mse(p: &[f64], l: &Lab) -> f64 {
    let tail = &p[500..];
    tail.iter()
        .zip(&l.f[500..])
        .map(|(p, f)| (p - f).powi(2) / 0.25)
        .sum::<f64>()
        / tail.len() as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[(v.len() - 1) / 2]
}

/// The median over `seeds` of [`mse`] on 3,000 rows.
fn median_mse(name: &str, k: usize, hl: f64, pat: &str, seeds: u64) -> f64 {
    const N: usize = 3000;
    median(
        (1..=seeds)
            .map(|seed| {
                let l = lab(k, N, seed);
                mse(&preds(name, k, hl, &l, &weights(pat, N, seed), 1.0), &l)
            })
            .collect(),
    )
}

/// What the step reads of a row: `z`, each feature's mean and its scale,
/// against the scaler as it stands, at the weight and with the floor the
/// warm-up gives the row.
fn read(m: &Sgd, x: &[f64], lam: f64, weight: f64) -> Vec<f64> {
    let k = x.len();
    let (mut z, mut mean, mut scale) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    let sc = m.scaler.as_ref().expect("standardize");
    let (admit, floor) = m.warmup.read(weight);
    standardize_row(sc, 1, x, lam, admit, floor, &mut z, &mut mean, &mut scale);
    z.into_iter().chain(mean).chain(scale).collect()
}

/// A stream whose weights are all scaled by a power of two is read as the
/// unscaled one, to the bit: every row's `z`, means and scales, before the
/// switch and after it, across gaps of 30 half-lives at rows 10 and 300 and
/// of up to 2 half-lives now and then, zero-weight rows among the rows, at
/// scales 2^-13, 2^-7 and 2^7. The read at weight 1 against moments in the
/// weights' units made the row a share of them that grew as the weights
/// shrank: on 6a594f88 row 1 already differs (`z` -1.13 at weights of 1,
/// -0.0124 at 2^-13).
#[test]
fn the_read_is_the_same_bits_at_every_power_of_two_weight_scale() {
    for seed in 1..=3u64 {
        let l = lab(3, 600, seed);
        let ws = weights("zeros", 600, seed);
        let mut r = Rng(seed ^ 0xD7);
        let d: Vec<f64> = (0..600)
            .map(|t| match t {
                0 => 0.0,
                10 | 300 => 30.0 * HL,
                _ if r.u() < 0.03 => 2.0 * HL * r.u(),
                _ => 1.0,
            })
            .collect();
        for c in [2f64.powi(-13), 2f64.powi(-7), 2f64.powi(7)] {
            let mut one = Sgd::new(cfg(3, 0.1, LearningRate::AdaGrad)).unwrap();
            let mut scaled = one.clone();
            let mut held = 0;
            for t in 0..600 {
                let lam = one.cfg.decay.factor(d[t]);
                let (a, b) = (
                    read(&one, &l.x[t], lam, ws[t]),
                    read(&scaled, &l.x[t], lam, c * ws[t]),
                );
                for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "seed {seed}, scale {c:e}, row {t}, number {i}: {a:e} against {b:e}"
                    );
                }
                held += usize::from(one.warmup.switched());
                one.step(&l.x[t], &[Some(l.y[t])], d[t], ws[t]);
                scaled.step(&l.x[t], &[Some(l.y[t])], d[t], c * ws[t]);
            }
            assert!(held > 500, "the fit was held on {held} rows");
        }
    }
}

/// `pa`'s step is capped at `C·min(w, 1)`, so it is not invariant to the
/// weights' scale by design; at a constant weight of 1 or more the cap is
/// the same, and the read in the stream's own unit makes the fit the same:
/// every prediction of 3,000 rows at weights of 2 and 128 is the one at
/// weights of 1, to the bit. On 6a594f88, which read every row at weight 1
/// against moments of rows of 2, row 2 already differs.
#[test]
fn pa_at_constant_weights_of_1_and_more_is_the_same_fit() {
    for seed in 1..=3u64 {
        let l = lab(8, 3000, seed);
        let ones = vec![1.0; 3000];
        let base = preds("pa", 8, HL, &l, &ones, 1.0);
        for c in [2.0, 128.0] {
            let got = preds("pa", 8, HL, &l, &ones, c);
            for (t, (a, b)) in base.iter().zip(&got).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "seed {seed}, weights of {c}, row {t}: {b:e} against {a:e}"
                );
            }
        }
    }
}

/// AdaGrad reads the weights only in ratios but for its epsilon, which is
/// in the target's units: on weights scaled by `c` the fit is the fit on the
/// weights within that epsilon's reach. 8 features, weights 0.5 to 1.5,
/// 3,000 rows: every prediction at weights times 2^-7 and 2^-13 within 1e-3
/// of the target's spread of the one at the weights themselves (the check
/// measured 5.8e-7 and 3.8e-5, in proportion to `1/c`; here 3.4e-5). On
/// 6a594f88 1.7e3 spreads.
#[test]
fn adagrad_on_scaled_weights_is_the_fit_on_the_weights() {
    let mut worst = 0.0f64;
    for seed in 1..=5u64 {
        let l = lab(8, 3000, seed);
        let ws = weights("unif", 3000, seed);
        let base = preds("adagrad", 8, HL, &l, &ws, 1.0);
        let mean = l.y.iter().sum::<f64>() / l.y.len() as f64;
        let sd = (l.y.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / l.y.len() as f64).sqrt();
        for c in [2f64.powi(-7), 2f64.powi(-13)] {
            let got = preds("adagrad", 8, HL, &l, &ws, c);
            for (a, b) in base.iter().zip(&got) {
                worst = worst.max((a - b).abs() / sd);
            }
        }
    }
    println!("largest move of a prediction, in spreads of the target: {worst:e}");
    assert!(worst <= 1e-3, "{worst:e} spreads of the target");
}

/// The bound of `adagrad_moves_a_slope_at_most_sqrt2_rates_over_its_spread_across_any_gap`,
/// `|Δb_i|·sd_i <= sqrt(2)·lr` on every held row, at weights scaled by
/// 2^-7 and 128: the history keeps one row's weight of the stream, and the
/// row is read at no more than one row's, so its share of the moments is at
/// most 1/2 at any scale. Measured 1.413956 and 1.414212; on 6a594f88 (no
/// floor) 1.6e3 at 2^-7 with gaps to 16 half-lives, and under a floor of
/// weight 1 with the row read at 1 (148a032e) 5.42 at 2^-7.
#[test]
fn adagrad_moves_a_slope_at_most_sqrt2_rates_at_any_weight_scale() {
    let bound = std::f64::consts::SQRT_2 * (1.0 + 1e-12);
    for scale in [2f64.powi(-7), 128.0] {
        for double in [false, true] {
            for gmax in [16.0, 300.0, 2000.0] {
                let mut worst = 0.0f64;
                for seed in 1..=10u64 {
                    let mut rows = irregular(4, gmax, 4000, seed, double);
                    for r in &mut rows {
                        r.w *= scale;
                    }
                    let mut m = Sgd::new(cfg(4, 0.05, LearningRate::AdaGrad)).unwrap();
                    worst = worst.max(largest_held_step(&mut m, &rows));
                }
                println!(
                    "weights times {scale:e}, gaps to {gmax}, two in a row {double}: {worst:.6}"
                );
                assert!(
                    worst <= bound,
                    "weights times {scale:e}, gaps to {gmax} half-lives (two in a row: \
                     {double}): a slope moved {worst:e} rates over its spread"
                );
            }
        }
    }
}

/// Weights that fall 100-fold over the stream are read at the row's own
/// weight once the fit is held, so the rows after the warm-up are not each
/// read as a row of the warm-up's size: a fit cannot be worse for having
/// seen heavier rows first. AdaGrad and `pa`, 8 features: the median error
/// over 10 seeds within twice the constant-weight stream's (measured:
/// AdaGrad 0.174 against 0.183, `pa` 0.121 against 1.17; on 6a594f88
/// AdaGrad's was 4,176).
#[test]
fn falling_weights_fit_as_constant_weights_do() {
    for name in ["adagrad", "pa"] {
        let (down, flat) = (
            median_mse(name, 8, HL, "down", 10),
            median_mse(name, 8, HL, "const", 10),
        );
        println!("{name}: falling weights {down:.4}, constant {flat:.4}");
        assert!(
            down <= 2.0 * flat,
            "{name}: weights falling 100-fold {down:.4} against constant weights {flat:.4}"
        );
    }
}

/// A few heavy rows: one row in 100 at 100 times the rest, the mean
/// weight about 1, so `Σw²/Σw` is about 50 rows of the rest. Each light row
/// was read at weight 1, about twice its own, and at a half-life of 5 rows
/// it was a share of the moments that the sum over 8 features made too
/// large: AdaGrad's error 444 on 6a594f88, 0.69 under the rule, where
/// constant weights give 0.41. Held within 2.5 times the constant-weight
/// stream's (measured 1.67).
#[test]
fn a_few_heavy_rows_do_not_throw_the_fit() {
    let (spike, flat) = (
        median_mse("adagrad", 8, 5.0, "spike", 10),
        median_mse("adagrad", 8, 5.0, "const", 10),
    );
    println!("one row in 100 at 100 times the rest {spike:.4}, constant {flat:.4}");
    assert!(
        spike <= 2.5 * flat,
        "one row in 100 at 100 times the rest: {spike:.4} against constant weights {flat:.4}"
    );
}

/// Lognormal weights (sd 1 in the log, mean 1): `Σw²/Σw` is 2.7 mean rows,
/// so a light row is read at its own weight and a heavy one at a typical
/// row's. AdaGrad's median error over 10 seeds at 8, 16 and 32 features
/// within 1.1 times the build's before the rule (6a594f88, which read every
/// row at 1): 0.2538, 0.5629 and 1.2786 against 0.2691, 0.5975 and 1.3741,
/// measured 2026-11-20. Reading every row at a typical row's weight instead
/// made a light row's share of the moments too large: 47.8 at 32 features
/// in the check.
#[test]
fn lognormal_weights_fit_no_worse_than_before() {
    for (k, before) in [(8usize, 0.2691), (16, 0.5975), (32, 1.3741)] {
        let got = median_mse("adagrad", k, HL, "ln1", 10);
        println!("{k} features: {got:.4}, before {before}");
        assert!(
            got <= 1.1 * before,
            "{k} features, lognormal weights: {got:.4} against {before} before"
        );
    }
}

/// The mean error over rows 1-5, 6-20, 21-100 and 101-300 after row `n0`,
/// the first of them a clock gap of `gap` half-lives (`None`, the twin: a
/// regular row).
fn windows(name: &str, k: usize, l: &Lab, ws: &[f64], n0: usize, gap: Option<f64>) -> [f64; 4] {
    const WIN: [(usize, usize); 4] = [(1, 5), (6, 20), (21, 100), (101, 300)];
    let mut m = Learner::new(name, k, HL);
    let (mut sums, mut counts) = ([0.0; 4], [0usize; 4]);
    for (t, &w) in ws.iter().enumerate() {
        let d = match t {
            0 => 0.0,
            _ if t == n0 => gap.map_or(1.0, |g| g * HL),
            _ => 1.0,
        };
        let p = m.step(&l.x[t], l.y[t], d, w).pred[0];
        if t >= n0 {
            let e = if p.is_finite() {
                (p - l.f[t]).powi(2) / 0.25
            } else {
                l.f[t].powi(2) / 0.25
            };
            for (w, &(lo, hi)) in WIN.iter().enumerate() {
                if (lo..=hi).contains(&(t - n0)) {
                    sums[w] += e;
                    counts[w] += 1;
                }
            }
        }
    }
    std::array::from_fn(|w| sums[w] / counts[w].max(1) as f64)
}

/// The worst of the four windows' 99th percentile over seeds after a gap
/// at row `n0`, over the twin's, at 2 and 8 features.
fn worst_window(name: &str, pat: &str, scale: f64, n0: usize, gap: f64, seeds: u64) -> f64 {
    let mut worst = 0.0f64;
    for k in [2usize, 8] {
        let n = n0 + 301;
        let (mut g, mut t) = (vec![vec![]; 4], vec![vec![]; 4]);
        for seed in 1..=seeds {
            let l = lab(k, n, seed);
            let ws: Vec<f64> = weights(pat, n, seed).iter().map(|w| w * scale).collect();
            let (a, b) = (
                windows(name, k, &l, &ws, n0, Some(gap)),
                windows(name, k, &l, &ws, n0, None),
            );
            for w in 0..4 {
                g[w].push(a[w]);
                t[w].push(b[w]);
            }
        }
        for w in 0..4 {
            let ratio = p99(g[w].clone()) / p99(t[w].clone());
            assert!(ratio.is_finite(), "{name}, k {k}, window {w}: {ratio}");
            worst = worst.max(ratio);
        }
    }
    worst
}

/// Each learner after a gap at a row of `n0s`, of each of `gaps`
/// half-lives, held to its band over the twin ([`worst_window`], 200 seeds).
fn during_the_warm_up(n0s: &[usize], gaps: &[f64]) {
    for (name, band) in [("constant", 2.0), ("adagrad", 3.0), ("pa", 6.0)] {
        for &n0 in n0s {
            for &gap in gaps {
                for (pat, scale) in [("const", 1.0), ("unif", 2f64.powi(-7))] {
                    let worst = worst_window(name, pat, scale, n0, gap, 200);
                    println!("{name}, row {n0}, {gap} half-lives, {pat} x {scale:e}: {worst:.3}");
                    assert!(
                        worst <= band,
                        "{name}: a gap of {gap} half-lives at row {n0}, weights {pat} times \
                         {scale:e}: {worst:.3} times the twin's error, past {band}"
                    );
                }
            }
        }
    }
}

/// A long gap during the warm-up, before the fit is held: the history keeps
/// `n/(n + 1)` of one row's weight of the stream into the row, so the row is
/// not read against moments of almost nothing. A gap at row 20 or 21 (the
/// switch is at 22) of 15, 60 or 1,100 half-lives, weights of 1 and 0.5 to
/// 1.5 times 2^-7: each window's error at the 99th percentile over 200
/// seeds within 2, 3 and 6 times the twin's for a constant rate, AdaGrad
/// and `pa` (measured 1.24, 1.22 and 3.74, the check's figures; `pa` is 2.0
/// times its twin after a gap of 3 half-lives at row 20, which no floor
/// binds on: its own sensitivity). With no floor before the switch
/// (6a594f88) 42 for a constant rate after 15 half-lives at row 20 and
/// 1.1e17 after 60 at row 21 (the check: up to 9e18 for `pa`). The 99th
/// percentile of 20 seeds is their
/// largest, and `pa`'s then reads 12, so the seeds are the check's.
#[test]
#[ignore = "extended: 36 cells of 200 seeds, about 12 s"]
fn a_long_gap_during_the_warm_up_tracks_its_twin() {
    during_the_warm_up(&[20, 21], &[15.0, 60.0, 1100.0]);
}

/// The essentials' form: one gap of 60 half-lives at row 21, where each
/// learner failed by most with no floor before the switch (1.1e17 to 9e18).
#[test]
fn a_long_gap_just_before_the_switch_tracks_its_twin() {
    during_the_warm_up(&[21], &[60.0]);
}

/// The twin after a long gap past the warm-up, at weights of 0.5 to 1.5
/// times 128: the floor in the weights' units kept one unit of history beside
/// rows of 128 and left AdaGrad 44 to 149 times its twin (148a032e, the
/// check), and with no floor (6a594f88) `pa` was 64 times its twin; the
/// floor of one row of the stream holds them within the unit-weight bands
/// of `each_learner_after_a_long_gap_tracks_its_twin_with_no_gap`
/// (measured 6.2, 1.0 and 1.7 for AdaGrad, a constant rate and `pa`).
#[test]
fn a_long_gap_at_weights_of_128_tracks_its_twin() {
    for (name, band) in [("constant", 2.0), ("pa", 3.0), ("adagrad", 10.0)] {
        for gap in [30.0, 1100.0] {
            let worst = worst_window(name, "unif", 128.0, 700, gap, 20);
            println!("{name}, {gap} half-lives at weights x 128: {worst:.3}");
            assert!(
                worst <= band,
                "{name}: a gap of {gap} half-lives at weights times 128: {worst:.3} times the \
                 twin's error, past {band}"
            );
        }
    }
}

/// Hard rule 9 where the floor before the switch reads Kish's count, and at
/// a weight scale: a gap at rows 1 to 21 and 700, of 0.5 to 1,000
/// half-lives, taken whole or in 2 or 5 parts with zero-weight rows inside
/// it, at weights of 1 and of 0.5 to 1.5 times 2^-7. A zero-weight row
/// moves neither `Σw²/Σw` nor the count, and the floor composes across the
/// parts, so the predictions after it agree within 1e-10 relative and the
/// weight within 1e-14 (the check: 4.0e-12 and 4.7e-16).
#[test]
fn a_gap_split_by_zero_weight_rows_is_one_gap_at_any_row_and_scale() {
    let (mut worst, mut worst_n) = (0.0f64, 0.0f64);
    for name in ["constant", "adagrad", "pa"] {
        for (pat, scale) in [("const", 1.0), ("unif", 2f64.powi(-7))] {
            for n0 in [1usize, 3, 12, 20, 21, 700] {
                for gap in [0.5, 6.0, 40.0, 300.0, 1000.0] {
                    let l = lab(2, n0 + 101, 1);
                    let ws = weights(pat, n0 + 101, 1);
                    let run = |parts: usize| -> Vec<(f64, f64)> {
                        let mut m = Learner::new(name, 2, HL);
                        let mut out = vec![];
                        for (t, &w) in ws.iter().enumerate() {
                            let part = gap * HL / parts as f64;
                            if t == n0 {
                                for _ in 1..parts {
                                    m.step(&l.x[t], l.y[t], part, 0.0);
                                }
                            }
                            let d = match t {
                                0 => 0.0,
                                _ if t == n0 => part,
                                _ => 1.0,
                            };
                            let st = m.step(&l.x[t], l.y[t], d, w * scale);
                            out.push((st.pred[0], st.n_eff));
                        }
                        out
                    };
                    let one = run(1);
                    for parts in [2, 5] {
                        for (a, b) in one.iter().zip(run(parts)).skip(n0 + 1) {
                            assert_eq!(a.0.is_finite(), b.0.is_finite(), "{name}, row {n0}");
                            if a.0.is_finite() {
                                let rel = (a.0 - b.0).abs() / a.0.abs().max(b.0.abs()).max(1e-300);
                                worst = worst.max(rel);
                            }
                            worst_n = worst_n.max((a.1 - b.1).abs() / a.1.abs().max(b.1.abs()));
                        }
                    }
                }
            }
        }
    }
    println!("predictions {worst:e}, weight {worst_n:e}");
    assert!(worst <= 1e-10, "a prediction {worst:e} apart");
    assert!(worst_n <= 1e-14, "the weight {worst_n:e} apart");
}
