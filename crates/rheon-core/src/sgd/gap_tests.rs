//! The scaler's floor once the fit is held in the caller's units
//! (`scaler_decay`, review fable2 C1 and its rule A): after a long clock gap
//! the history keeps one row's weight of the stream into the row, and the
//! row is read at no more than one row's, so a step taken in the row's
//! coordinates is mapped into the caller's by a scale of at least the
//! feature's spread over `sqrt(2)`. Its unit of weight is tested in
//! `weight_unit_tests`.
//!
//! The streams and the expected bands are those of the check that settled
//! the rule (`.claude/coord/review-fable2/C1-ADAGRAD.md`, harness
//! `c1_bits.rs` and `c1_lab.rs`): features with spreads from 0.03 to 30
//! and means up to 3 spreads off zero, a linear truth drifting 0.002 of a
//! spread a row, noise sd 0.5, a half-life of 20 rows.

use super::*;
use crate::{Pa, PaCfg, PaMode};

/// SplitMix64, the harness's generator.
#[derive(Clone, Copy)]
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub(super) fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

pub(super) const HL: f64 = 20.0;

pub(super) fn cfg(k: usize, lr: f64, schedule: LearningRate) -> SgdCfg {
    SgdCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(HL),
        loss: SgdLoss::Squared,
        learning_rate: lr,
        schedule,
        l2: 0.0,
        min_weight: 0.0,
        standardize: true,
        strict_binary: false,
        clip_gradient: 1e3,
        constraint: None,
        huber_scale: crate::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        min_weight_holds_learning: false,
        target_spread: true,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
        average: false,
        invariant: false,
        l1: 0.0,
    }
}

pub(super) fn pa_cfg(k: usize) -> PaCfg {
    PaCfg {
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(HL),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.01,
        min_weight: 0.0,
        constraint: None,
        standardize: true,
        average: false,
    }
}

pub(super) struct Row {
    pub(super) x: Vec<f64>,
    pub(super) y: Option<f64>,
    pub(super) d: f64,
    pub(super) w: f64,
}

/// An irregular stream (`c1_bits.rs`): clock steps exponential with mean
/// 1, one row in 50 a gap uniform in `(0, gmax]` half-lives; weights 1,
/// 0.25 or 2.5, one row in 40 of weight 0; one target in 30 null. With
/// `double`, rows 1,500 and 1,501 are each a gap of `gmax` half-lives.
/// Returns the rows and each feature's true sd.
pub(super) fn irregular(k: usize, gmax: f64, n: usize, seed: u64, double: bool) -> Vec<Row> {
    let mut r = Rng(seed ^ 0xB175);
    let sd: Vec<f64> = (0..k).map(|_| 10f64.powf(3.0 * r.u() - 1.5)).collect();
    let mu: Vec<f64> = sd.iter().map(|s| s * (6.0 * r.u() - 3.0)).collect();
    let b: Vec<f64> = sd.iter().map(|s| r.n() / s).collect();
    (0..n)
        .map(|t| {
            let x: Vec<f64> = (0..k).map(|i| mu[i] + sd[i] * r.n()).collect();
            let y = 1.0 + (0..k).map(|i| b[i] * x[i]).sum::<f64>() + 0.5 * r.n();
            let d = if t == 0 {
                0.0
            } else if double && (t == 1500 || t == 1501) {
                HL * gmax
            } else if r.u() < 0.02 {
                HL * gmax * r.u()
            } else {
                -r.u().ln()
            };
            let u = r.u();
            let w = if u < 0.025 {
                0.0
            } else if u < 0.2 {
                0.25
            } else if u < 0.3 {
                2.5
            } else {
                1.0
            };
            let y = (r.u() >= 0.033).then_some(y);
            Row { x, y, d, w }
        })
        .collect()
}

/// The largest `|Δb_i| · sd_i / lr` over every row the fit is held in the
/// caller's units as it arrives, `sd_i` the scaler's own before the row.
pub(super) fn largest_held_step(m: &mut Sgd, rows: &[Row]) -> f64 {
    let lr = m.cfg.learning_rate;
    let mut worst = 0.0f64;
    for r in rows {
        let held = m.warmup.switched();
        let sd: Vec<f64> = {
            let sc = m.scaler.as_ref().expect("standardize");
            (0..r.x.len()).map(|i| sc.var(1 + i).sqrt()).collect()
        };
        let before = m.beta[0].clone();
        m.step(&r.x, &[r.y], r.d, r.w);
        assert!(m.beta[0].iter().all(|b| b.is_finite()), "a finite fit");
        if held {
            for (i, s) in sd.iter().enumerate() {
                worst = worst.max((m.beta[0][1 + i] - before[1 + i]).abs() * s / lr);
            }
        }
    }
    worst
}

/// AdaGrad's step is at most the rate in the row's coordinates whatever
/// its sum holds, and the floor keeps the scale it is divided by at least
/// `sd/sqrt(2)`, so on every held row `|Δb_i| · sd_i <= sqrt(2) · lr`,
/// exactly. 20 irregular streams of 4,000 rows with gaps to 2,000
/// half-lives, zero-weight rows among them, and the same with two gaps of
/// the longest length in a row. Measured by the check: 1.414207 under a
/// floor of weight 1 and 1.414203 under one row's weight of the stream;
/// on f9e3bf77 8.4e8 with gaps to 60 half-lives and 3.1e151 with
/// gaps to 2,000 (`results/bits.txt`).
#[test]
fn adagrad_moves_a_slope_at_most_sqrt2_rates_over_its_spread_across_any_gap() {
    let bound = std::f64::consts::SQRT_2 * (1.0 + 1e-12);
    for double in [false, true] {
        for gmax in [2.0, 16.0, 60.0, 300.0, 2000.0] {
            let mut worst = 0.0f64;
            for seed in 1..=20u64 {
                let rows = irregular(4, gmax, 4000, seed, double);
                let mut m = Sgd::new(cfg(4, 0.05, LearningRate::AdaGrad)).unwrap();
                worst = worst.max(largest_held_step(&mut m, &rows));
            }
            println!("gaps to {gmax} half-lives, two in a row {double}: {worst:.6e}");
            assert!(
                worst <= bound,
                "gaps to {gmax} half-lives (two in a row: {double}): a slope moved \
                 {worst:e} rates over its spread, past sqrt(2)"
            );
        }
    }
}

/// The truth and features of `c1_lab.rs` (`Truth::Same`), `n` rows.
pub(super) struct Lab {
    pub(super) x: Vec<Vec<f64>>,
    pub(super) y: Vec<f64>,
    pub(super) f: Vec<f64>,
}

pub(super) fn lab(k: usize, n: usize, seed: u64) -> Lab {
    let mut r = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xC1AD);
    let sd: Vec<f64> = (0..k).map(|_| 10f64.powf(3.0 * r.u() - 1.5)).collect();
    let mu: Vec<f64> = sd.iter().map(|s| s * (6.0 * r.u() - 3.0)).collect();
    let mut b: Vec<f64> = sd.iter().map(|s| r.n() / s).collect();
    let b0 = 3.0 * r.n();
    let (mut x, mut y, mut f) = (vec![], vec![], vec![]);
    for _ in 0..n {
        for i in 0..k {
            b[i] += 0.002 * r.n() / sd[i];
        }
        let row: Vec<f64> = (0..k).map(|i| mu[i] + sd[i] * r.n()).collect();
        let mean = b0 + (0..k).map(|i| b[i] * row[i]).sum::<f64>();
        y.push(mean + 0.5 * r.n());
        f.push(mean);
        x.push(row);
    }
    Lab { x, y, f }
}

#[derive(Clone)]
enum Learner {
    Sgd(Box<Sgd>),
    Pa(Box<Pa>),
}

impl Learner {
    fn coefficients(&self) -> Vec<f64> {
        match self {
            Self::Sgd(m) => m.coefficients().remove(0),
            Self::Pa(m) => m.coefficients().remove(0),
        }
    }

    fn step(&mut self, x: &[f64], y: f64, d: f64) {
        match self {
            Self::Sgd(m) => m.step(x, &[Some(y)], d, 1.0),
            Self::Pa(m) => m.step(x, &[Some(y)], d, 1.0),
        };
    }
}

/// Rows of the run after the gap whose error is kept: the gap row is 0.
const AFTER: usize = 100;

/// The out-of-sample error `(p − f)²/0.25` of rows `0..=AFTER` from row
/// `n0` on, the first of them `gaps` clock gaps (each `gap` half-lives;
/// none for the twin), the model as it stood before row `n0`.
fn after(model: &Learner, lab: &Lab, n0: usize, gap: f64, gaps: usize) -> Vec<f64> {
    let mut m = model.clone();
    (0..=AFTER)
        .map(|r| {
            let t = n0 + r;
            let c = m.coefficients();
            let p = c[0]
                + c[1..]
                    .iter()
                    .zip(&lab.x[t])
                    .map(|(b, x)| b * x)
                    .sum::<f64>();
            let d = if r < gaps { gap * HL } else { 1.0 };
            m.step(&lab.x[t], lab.y[t], d);
            (p - lab.f[t]).powi(2) / 0.25
        })
        .collect()
}

pub(super) fn p99(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * 0.99).round() as usize]
}

/// Each learner on the scaler after one long gap (or `gaps` in a row),
/// against its twin fed the same rows with no gap: the 99th percentile over
/// seeds of the mean error over rows 6 to 100 after the gap, over the
/// twin's, at most `band` in every cell of 2 and 8 features at rates 0.01
/// and 0.1 (`pa` at its own settings, whatever the rate). Returns the
/// worst ratio.
fn worst_against_the_twin(learner: &str, gap: f64, gaps: usize, seeds: u64) -> f64 {
    let n0 = 700;
    let mut worst = 0.0f64;
    for k in [2usize, 8] {
        for lr in [0.01, 0.1] {
            if learner == "pa" && lr == 0.1 {
                continue;
            }
            let (mut gapped, mut twin) = (vec![], vec![]);
            for seed in 1..=seeds {
                let lab = lab(k, n0 + AFTER + 1, seed);
                let mut m = match learner {
                    "adagrad" => Learner::Sgd(Box::new(
                        Sgd::new(cfg(k, lr, LearningRate::AdaGrad)).unwrap(),
                    )),
                    "constant" => Learner::Sgd(Box::new(
                        Sgd::new(cfg(k, lr, LearningRate::Constant)).unwrap(),
                    )),
                    _ => Learner::Pa(Box::new(Pa::new(pa_cfg(k)).unwrap())),
                };
                for t in 0..n0 {
                    m.step(&lab.x[t], lab.y[t], if t == 0 { 0.0 } else { 1.0 });
                }
                let window = |e: Vec<f64>| e[6..].iter().sum::<f64>() / (AFTER - 5) as f64;
                gapped.push(window(after(&m, &lab, n0, gap, gaps)));
                twin.push(window(after(&m, &lab, n0, 0.0, 0)));
            }
            let ratio = p99(gapped) / p99(twin);
            assert!(ratio.is_finite(), "{learner}, k {k}, lr {lr}: {ratio}");
            worst = worst.max(ratio);
        }
    }
    worst
}

/// The rows after a long gap read a scaler that holds one or two rows
/// while the fit stays in the caller's units, which hurt every learner on
/// it. Over rows 6 to 100 after a gap of 15, 30 and 1,100 half-lives, at
/// the 99th percentile over 100 seeds, each is held to its twin with no
/// gap within the check's measured factors with room: worst cell over rows
/// 2 to 100 at 1,000 seeds 6.2 for AdaGrad, 1.2 at a constant rate and 1.7
/// for `pa`, so bands of 10, 2 and 3 (C1-ADAGRAD.md, "How to test it", 3;
/// table D, rows 6-20 p99 under the floor: AdaGrad 2.12, constant 1.09,
/// `pa` 1.31). On f9e3bf77 the same cells reached 4.2e5 (AdaGrad, 30
/// half-lives), 65 (constant, 30) and 5,121 (`pa`, 30), and 3.88 and 22.0
/// already at 15 (table D).
#[test]
#[ignore = "extended: 100 seeds of four streams for each learner and gap, about 10 s"]
fn each_learner_after_a_long_gap_tracks_its_twin_with_no_gap() {
    for (learner, band) in [("constant", 2.0), ("pa", 3.0), ("adagrad", 10.0)] {
        for gap in [15.0, 30.0, 1100.0] {
            let worst = worst_against_the_twin(learner, gap, 1, 100);
            println!("{learner}, one gap of {gap} half-lives: {worst:.3}");
            assert!(
                worst <= band,
                "{learner}, one gap of {gap} half-lives: rows 6-100 after it {worst:.3} times \
                 the twin's error at the 99th percentile, past {band}"
            );
        }
    }
}

/// The essentials' form of the twin: 20 seeds, one gap of 30 half-lives,
/// the cell where each learner failed by most on f9e3bf77.
#[test]
fn each_learner_after_a_gap_of_30_half_lives_tracks_its_twin() {
    for (learner, band) in [("constant", 2.0), ("pa", 3.0), ("adagrad", 10.0)] {
        let worst = worst_against_the_twin(learner, 30.0, 1, 20);
        println!("{learner}, one gap of 30 half-lives: {worst:.3}");
        assert!(
            worst <= band,
            "{learner}: rows 6-100 after a gap of 30 half-lives {worst:.3} times the twin's \
             error, past {band}"
        );
    }
}

/// Two long gaps in a row: the first collapses the target's spread, in
/// whose units AdaGrad's epsilon is, so on f9e3bf77 the second gap's step
/// was not held at all (slopes of 1e29 after two gaps of 200 half-lives and
/// 1e89 after two of 600, `results/two_gaps.txt`). Under the floor each gap
/// row is read with the history's share at 1/2 or more, and every learner
/// tracks its twin as after one gap.
#[test]
fn two_long_gaps_in_a_row_are_one() {
    for (learner, band) in [("constant", 2.0), ("pa", 3.0), ("adagrad", 10.0)] {
        for gap in [200.0, 600.0] {
            let worst = worst_against_the_twin(learner, gap, 2, 20);
            println!("{learner}, two gaps of {gap} half-lives: {worst:.3}");
            assert!(
                worst <= band,
                "{learner}, two gaps of {gap} half-lives: rows 6-100 after them {worst:.3} \
                 times the twin's error, past {band}"
            );
        }
    }
}

/// The floor binds only where the history's decayed weight is below one
/// row's of the stream, `lam·W < ŵ` (`Warmup::read`), and on no other row
/// does it move the decay, read and update alike. On streams whose gaps stay
/// under 2 half-lives, weights from 0.5 to 1.5 with zero-weight rows among
/// them, at a half-life of 20 rows it binds on no row, so the outputs are
/// the bits they were before the floor (the check measured 0 of 80,000
/// predictions moved, `results/bits.txt`). At a half-life of 5 rows the
/// history weighs about 8 rows, and a gap of up to 2 half-lives keeps a
/// quarter of it or more, about two rows, so the floor binds only where
/// two such gaps come close together: a few held rows in 10 streams of
/// 3,000 (`.claude/coord/review-fable2/C1-RULE-A.md`, which restated the
/// premise from `lam·W < 1`, a floor in the weights' units).
/// That the floor binds on a stream with a long gap shows the probe reaches
/// it.
#[test]
fn the_floor_binds_only_where_the_history_weighs_less_than_a_row() {
    for (hl, gmax, most, binds) in [
        (20.0, 2.0, 0, false),
        (5.0, 2.0, 40, true),
        (20.0, 30.0, usize::MAX, true),
    ] {
        let mut bound = 0usize;
        for seed in 1..=10u64 {
            let mut r = Rng(seed);
            let mut m = Sgd::new(SgdCfg {
                decay: Decay::Halflife(hl),
                ..cfg(3, 0.05, LearningRate::AdaGrad)
            })
            .unwrap();
            for t in 0..3000 {
                let x = [r.n(), 10.0 + 0.1 * r.n(), 100.0 * r.n()];
                let y = 1.0 + x[0] - 0.3 * x[1] + 0.01 * x[2] + 0.5 * r.n();
                let d = if t == 0 {
                    0.0
                } else if r.u() < 0.02 {
                    hl * gmax * r.u()
                } else {
                    2.0 * r.u()
                };
                let w = if r.u() < 0.02 { 0.0 } else { 0.5 + r.u() };
                let lam = m.cfg.decay.factor(d);
                if m.warmup.switched() {
                    let sc = m.scaler.as_ref().unwrap();
                    let (admit, floor) = m.warmup.read(w);
                    let short = lam < 1.0 && lam * sc.n_eff() < floor;
                    for v in [admit, w] {
                        let (held, warm) = (
                            scaler_decay(sc, lam, v, floor),
                            scaler_decay(sc, lam, v, 0.0),
                        );
                        assert_eq!(
                            held.to_bits() != warm.to_bits(),
                            short,
                            "half-life {hl} rows, row {t}: the floor moved the decay where \
                             the history weighed {} against a floor of {floor}",
                            lam * sc.n_eff()
                        );
                    }
                    bound += usize::from(short);
                }
                m.step(&x, &[Some(y)], d, w);
            }
        }
        assert_eq!(
            bound > 0,
            binds,
            "half-life {hl} rows, gaps to {gmax} half-lives: the floor bound {bound} times"
        );
        assert!(
            bound <= most,
            "half-life {hl} rows, gaps to {gmax} half-lives: the floor bound {bound} times, \
             past {most}"
        );
    }
}

/// A gap taken in one step, or in two or five with zero-weight rows inside
/// it, leaves the fit within 1e-12 relative of itself 50 rows on (hard
/// rule 9): `max(lam, min(1, ŵ/W))` composes, as `max(lam_a·lam_b·W,
/// min(W, ŵ))`. The check measured 1.1e-15 at most, as before the floor
/// (`results/split.txt`).
#[test]
fn a_gap_split_by_zero_weight_rows_is_one_gap() {
    for schedule in [LearningRate::AdaGrad, LearningRate::Constant] {
        for gap in [1.0, 6.0, 40.0, 300.0] {
            let fit = |parts: usize| -> Vec<f64> {
                let mut r = Rng(7);
                let mut m = Sgd::new(cfg(2, 0.05, schedule)).unwrap();
                let row = |r: &mut Rng| {
                    let x = [r.n(), 5.0 + 2.0 * r.n()];
                    (x, 1.0 + x[0] - 0.2 * x[1] + 0.5 * r.n())
                };
                for t in 0..400 {
                    let (x, y) = row(&mut r);
                    m.step(&x, &[Some(y)], if t == 0 { 0.0 } else { 1.0 }, 1.0);
                }
                for _ in 1..parts {
                    m.step(&[0.3, 4.0], &[Some(0.0)], gap * HL / parts as f64, 0.0);
                }
                for t in 0..50 {
                    let (x, y) = row(&mut r);
                    let d = if t == 0 { gap * HL / parts as f64 } else { 1.0 };
                    m.step(&x, &[Some(y)], d, 1.0);
                }
                m.coefficients().remove(0)
            };
            let one = fit(1);
            for parts in [2, 5] {
                for (a, b) in one.iter().zip(fit(parts)) {
                    assert!(
                        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
                        "{schedule:?}, a gap of {gap} half-lives in {parts} parts: {a:e} \
                         against {b:e}"
                    );
                }
            }
        }
    }
}

/// AdaGrad's sums never pass the largest double (review 2026-11-23,
/// REG-9): at an infinite `clip_gradient`, two gradients of `1.2e154` each
/// pass the square's check, `1.44e308`, but their sum does not. The sum
/// was infinite, the slot's steps 0 for good, and at the next gap past the
/// decay's underflow `0·∞` made it NaN. A row whose sum would pass it is
/// refused, as one whose square would is, with or without a scaler and on
/// the intercept's slot as on a slope's.
#[test]
fn adagrads_sums_never_pass_the_largest_double() {
    for standardize in [false, true] {
        let mut c = cfg(1, 1e-60, LearningRate::AdaGrad);
        c.clip_gradient = f64::INFINITY;
        c.target_spread = false;
        c.standardize = standardize;
        c.decay = Decay::Halflife(1.0);
        let mut m = Sgd::new(c).unwrap();
        for i in 0..40 {
            // A target `1.2e54` off the prediction at the input bound: a
            // slope's gradient of `1.2e154` (and, while standardized, of
            // that over the feature's scale).
            let x = if i % 2 == 0 { 1e100 } else { -1e100 };
            let p = m.predict(&[x], 1.0).pred[0];
            let p = if p.is_finite() { p } else { 0.0 };
            let d = if i == 30 { 5000.0 } else { 1.0 };
            m.step(&[x], &[Some(p - 1.2e54)], d, 1.0);
            assert!(
                m.g2.iter().flatten().all(|v| v.is_finite()),
                "standardize {standardize}, row {i}: {:?}",
                m.g2
            );
        }
        let back = Sgd::restore(&m.state()).expect("a state that loads");
        assert_eq!(back.coefficients(), m.coefficients());
    }
}

/// A coefficient sum under `standardize` on features whose spread is near
/// `1e-155` (review 2026-11-23, REG-8): the projection's step per unit of
/// its multiplier, `1/s²`, was infinite once the fit was held in the
/// caller's units, and every coefficient NaN for good. The fit stays on the
/// sum and in the box.
#[test]
fn a_sum_on_features_near_the_smallest_doubles_holds() {
    let mut c = cfg(2, 0.05, LearningRate::AdaGrad);
    c.constraint = Some(crate::constraint::Constraint {
        lo: vec![0.0, 0.0],
        hi: vec![f64::INFINITY, f64::INFINITY],
        sum: Some(1.0),
    });
    let mut m = Sgd::new(c).unwrap();
    let mut r = Rng(5);
    for i in 0..80 {
        let x = [1e-155 * r.n(), 1e-155 * r.n()];
        let y = 1e155 * (0.3 * x[0] + 0.7 * x[1]) + 0.1 * r.n();
        m.step(&x, &[Some(y)], 1.0, 1.0);
        let b = &m.coefficients()[0];
        assert!(b.iter().all(|v| v.is_finite()), "row {i}: {b:?}");
        assert!((b[1] + b[2] - 1.0).abs() < 1e-12, "row {i}: {b:?}");
    }
}

/// Weights too small to square in a double count as rows (review
/// 2026-11-23, REG-12): the warm-up's and the level's Kish counts are
/// `(Σw)² / Σw²`, and at weights of `2^-664` each `w²` was 0, so both
/// counts stayed 0: the scaler never switched and the level never warmed.
/// Every weight times `2^-664` now counts as at unit weights, to the bit,
/// and the switch comes on the same row. (A weight is also a step size
/// here, `tests/test_weight_scale.py`, so the fits themselves part: at
/// such weights AdaGrad's epsilon outweighs every gradient, and the review's
/// 0.9 quantile held at the mean is that, not the count.)
#[test]
fn weights_too_small_to_square_count_as_rows() {
    let run = |scale: f64| {
        let mut c = cfg(1, 0.05, LearningRate::AdaGrad);
        c.loss = SgdLoss::Quantile { tau: 0.9 };
        let mut m = Sgd::new(c).unwrap();
        let mut r = Rng(9);
        (0..60)
            .map(|_| {
                let y = 1.0 + r.n();
                m.step(&[r.n()], &[Some(y)], 1.0, scale * (0.5 + r.u()));
                (
                    m.warmup().switched(),
                    m.warmup().count().to_bits(),
                    m.level[0].count().to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    let (unit, tiny) = (run(1.0), run(crate::sqsum::ldexp(1.0, -664)));
    assert_eq!(unit, tiny);
    assert!(tiny.last().is_some_and(|t| t.0), "the scaler switched");
}
