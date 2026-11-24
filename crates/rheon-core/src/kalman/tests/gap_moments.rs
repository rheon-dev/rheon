//! Review C1's follow-up for `kalman` (`EwDiag::update_keeping`): under
//! `standardize`, on a row more than a half-life after the moments last
//! moved, the moments' update reads the history at no less than the
//! row's own weight, `H = max(lam·W, w)`. `b` and `P` are held in
//! coordinates the moments define, and a history decayed to a share `a`
//! left them one row's, a scale of `sqrt(a)` of the feature's spread, past
//! what the re-map carries. The weight kept is the plain recursion's, so
//! `n_eff` is untouched (hard rule 8).
//!
//! The streams and bands are those of the check that settled the rule
//! (`.claude/coord/review-fable2/C1-RULE-A.md`, harness
//! `c1-rule-a/lab_main.rs`): features with spreads from 0.03 to 30 and
//! means up to 3 spreads off zero, a linear truth drifting 0.002 of a
//! spread a row, noise sd 0.5, a half-life of 20 rows, errors `(p − f)²/0.25`
//! in noise variances.

use super::*;

const HL: f64 = 20.0;

/// SplitMix64, the check's generator.
#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

struct Lab {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    f: Vec<f64>,
}

/// The check's stream: `k` features, `n` rows.
fn lab(k: usize, n: usize, seed: u64) -> Lab {
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

/// The check's filter: noise units, a coefficient half-life of `coef_hl`
/// (infinite: a fixed truth), the moments' half-life of 20 rows.
fn filter(k: usize, coef_hl: f64) -> Kalman {
    Kalman::new(KalmanCfg {
        min_weight_holds_learning: false,
        noise_units: true,
        n_features: k,
        n_targets: 1,
        fit_intercept: true,
        decay: Decay::Halflife(HL),
        half_life: vec![coef_hl],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 0.0,
        revert_half_life: vec![f64::INFINITY],
        standardize: true,
    })
    .unwrap()
}

fn p99(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * 0.99).round() as usize]
}

/// The mean error over rows 1-5, 6-20, 21-100 and 101-300 after row `n0`,
/// the first of them a clock gap of `gap` half-lives (`None`: a regular
/// row), from a model started afresh at `n0` where `fresh`.
fn windows(k: usize, coef_hl: f64, l: &Lab, n0: usize, gap: Option<f64>, fresh: bool) -> [f64; 4] {
    const WIN: [(usize, usize); 4] = [(1, 5), (6, 20), (21, 100), (101, 300)];
    let mut m = filter(k, coef_hl);
    let (mut sums, mut counts) = ([0.0; 4], [0usize; 4]);
    for t in 0..l.y.len() {
        if fresh && t == n0 {
            m = filter(k, coef_hl);
        }
        let d = match t {
            0 => 0.0,
            _ if t == n0 && fresh => 0.0,
            _ if t == n0 => gap.map_or(1.0, |g| g * HL),
            _ => 1.0,
        };
        let p = m.step(&l.x[t], &[Some(l.y[t])], d, 1.0).pred[0];
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

/// The worst window's 99th percentile over `seeds` after a gap of `gap`
/// half-lives at row 700, over a reference's: the same rows with no gap
/// (`fresh` false) or a model started at row 700 (`fresh`), at 2 and 8
/// features.
fn worst(coef_hl: f64, gap: f64, fresh: bool, seeds: u64) -> f64 {
    let n0 = 700;
    let mut worst = 0.0f64;
    for k in [2usize, 8] {
        let (mut g, mut r) = (vec![vec![]; 4], vec![vec![]; 4]);
        for seed in 1..=seeds {
            let l = lab(k, n0 + 301, seed);
            let (a, b) = (
                windows(k, coef_hl, &l, n0, Some(gap), false),
                windows(k, coef_hl, &l, n0, None, fresh),
            );
            for w in 0..4 {
                g[w].push(a[w]);
                r[w].push(b[w]);
            }
        }
        for w in 0..4 {
            let ratio = p99(g[w].clone()) / p99(r[w].clone());
            assert!(ratio.is_finite(), "k {k}, window {w}: {ratio}");
            worst = worst.max(ratio);
        }
    }
    worst
}

/// With no drift (`coef_half_life = inf`) nothing moves across a gap, so
/// the rows after one are the rows with none: at gaps of 30, 200 and 1,100
/// half-lives each window's error at the 99th percentile within 1% of the
/// twin's (the check, 200 seeds: equal to three digits). The moments, left
/// one row's after the gap, put `b` and `P` past what the re-map carries:
/// on 6a594f88 3.1e9 times the twin's after 30 half-lives (the check: 2e8
/// to 3e60 from 30 to 200).
#[test]
fn after_a_long_gap_a_fixed_truth_is_read_as_before_it() {
    for gap in [30.0, 200.0, 1100.0] {
        let w = worst(f64::INFINITY, gap, false, 20);
        println!("a gap of {gap} half-lives: {w:.4} of the twin's");
        assert!(
            w <= 1.01,
            "a gap of {gap} half-lives: {w:.4} times the twin's error"
        );
    }
}

/// Under a drifting truth (`coef_half_life` 200 rows) the reference is a
/// model started afresh at the gap: after one of 30 to 200 half-lives the
/// filter is within twice its error at the 99th percentile (the check: at
/// most 1.4, here 1.04; on 6a594f88 7.9e6 after 30 half-lives, and up to
/// 1e59 in the check).
#[test]
fn after_a_long_gap_a_drifting_truth_is_read_as_a_fresh_filter_reads_it() {
    for gap in [30.0, 200.0] {
        let w = worst(200.0, gap, true, 20);
        println!("a gap of {gap} half-lives: {w:.4} of a fresh filter's");
        assert!(
            w <= 2.0,
            "a gap of {gap} half-lives: {w:.4} times a fresh filter's error"
        );
    }
}

/// `kalman`'s `n_eff` is its moments' weight, so the floor is on the
/// moments alone and the weight kept is the plain recursion's, `W ← lam·W +
/// w`, to the bit, across gaps of 1 to 200 half-lives at rows of weight 0.5
/// to 1.5 and 0: a floor on the weight itself moved `weight_sum` and the
/// `min_weight` gate (the check's first prototype: 2.000 where 1.000 was
/// right on the row after a gap of 30 half-lives).
#[test]
fn the_weight_is_the_plain_recursions_across_a_gap() {
    for seed in 1..=5u64 {
        let l = lab(3, 600, seed);
        let mut r = Rng(seed ^ 0x6A);
        let mut m = filter(3, 200.0);
        let mut w_sum = 0.0f64;
        let mut bound = 0usize;
        for t in 0..600 {
            let d = match t {
                0 => 0.0,
                _ if t % 97 == 50 => HL * [1.0, 30.0, 200.0][t % 3],
                _ => 1.0,
            };
            let w = if r.u() < 0.05 { 0.0 } else { 0.5 + r.u() };
            bound += usize::from(m.keeps_history(m.moments_gap() + d) && w > 0.0);
            let st = m.step(&l.x[t], &[Some(l.y[t])], d, w);
            assert_eq!(st.n_eff.to_bits(), w_sum.to_bits(), "seed {seed}, row {t}");
            w_sum = m.cfg.decay.factor(d) * w_sum + w;
        }
        assert!(bound > 0, "seed {seed}: the floor never opened");
    }
}

/// Hard rule 9: a gap of 0.5 to 1,000 half-lives at rows 2, 5 and 700,
/// taken whole or in 2 or 5 parts with zero-weight rows inside it, at
/// weights of 1 and of 0.5 to 1.5 times 2^-7. The gate reads the clock since
/// the moments last moved, which a zero-weight row adds to and does not
/// clear, and `lam·W` is the same however the rows split it, so the
/// predictions after the gap agree within 1e-10 relative and the weight
/// within 1e-14 (the check: 1.7e-12). Past about 1,075 half-lives every
/// weight underflows to 0, and a split gap restarts where a whole one does
/// not, under any rule.
#[test]
fn a_gap_split_by_zero_weight_rows_is_one_gap() {
    let (mut worst, mut worst_n) = (0.0f64, 0.0f64);
    for scale in [1.0, 2f64.powi(-7)] {
        for n0 in [2usize, 5, 700] {
            for gap in [0.5, 6.0, 40.0, 300.0, 1000.0] {
                let l = lab(2, n0 + 101, 3);
                let mut r = Rng(0x5EED);
                let ws: Vec<f64> = (0..l.y.len())
                    .map(|_| {
                        if scale == 1.0 {
                            1.0
                        } else {
                            scale * (0.5 + r.u())
                        }
                    })
                    .collect();
                let run = |parts: usize| -> Vec<(f64, f64)> {
                    let mut m = filter(2, 200.0);
                    let mut out = vec![];
                    for (t, &w) in ws.iter().enumerate() {
                        let part = gap * HL / parts as f64;
                        if t == n0 {
                            for _ in 1..parts {
                                m.step(&l.x[t], &[Some(l.y[t])], part, 0.0);
                            }
                        }
                        let d = match t {
                            0 => 0.0,
                            _ if t == n0 => part,
                            _ => 1.0,
                        };
                        let st = m.step(&l.x[t], &[Some(l.y[t])], d, w);
                        out.push((st.pred[0], st.n_eff));
                    }
                    out
                };
                let one = run(1);
                for parts in [2, 5] {
                    for (a, b) in one.iter().zip(run(parts)).skip(n0 + 1) {
                        assert_eq!(a.0.is_finite(), b.0.is_finite(), "row {n0}, gap {gap}");
                        if a.0.is_finite() {
                            worst =
                                worst.max((a.0 - b.0).abs() / a.0.abs().max(b.0.abs()).max(1e-300));
                        }
                        worst_n = worst_n.max((a.1 - b.1).abs() / a.1.abs().max(b.1.abs()));
                    }
                }
            }
        }
    }
    println!("predictions {worst:e}, weight {worst_n:e}");
    assert!(worst <= 1e-10, "a prediction {worst:e} apart");
    assert!(worst_n <= 1e-14, "the weight {worst_n:e} apart");
}

/// The floor binds only on a row more than a half-life after the moments
/// last moved, where the history's decayed weight is below the row's own:
/// on a regular clock with weights of 0.5 to 1.5, one row in 20 of weight 0
/// (which learns nothing, so the next row is two steps after the last
/// learned one), at half-lives of 5 and 20 rows, it binds on no row, so
/// every bit is `update`'s (the check: none of 30,000 rows at those
/// half-lives; 1,346 at a half-life of one row, where a zero-weight row is
/// a gap of two half-lives). That it binds after gaps of up to 3 half-lives
/// shows the probe reaches it.
#[test]
fn the_floor_binds_only_after_more_than_a_half_life() {
    for (hl, gmax, binds) in [(20.0, 0.0, false), (5.0, 0.0, false), (20.0, 3.0, true)] {
        let mut bound = 0usize;
        for seed in 1..=5u64 {
            let l = lab(3, 2000, seed);
            let mut r = Rng(seed ^ 0xC10C);
            let mut m = Kalman::new(KalmanCfg {
                decay: Decay::Halflife(hl),
                ..filter(3, 200.0).cfg
            })
            .unwrap();
            for t in 0..2000 {
                let d = if t == 0 {
                    0.0
                } else if r.u() < 0.02 {
                    1.0 + hl * gmax * r.u()
                } else {
                    1.0
                };
                let w = if r.u() < 0.05 { 0.0 } else { 0.5 + r.u() };
                let gap = m.moments_gap() + d;
                let history = m.cfg.decay.factor(d) * m.stats.n_eff();
                bound +=
                    usize::from(m.keeps_history(gap) && w > 0.0 && history > 0.0 && history < w);
                m.step(&l.x[t], &[Some(l.y[t])], d, w);
            }
        }
        assert_eq!(
            bound > 0,
            binds,
            "half-life {hl} rows, gaps to {gmax} half-lives: the floor bound {bound} times"
        );
    }
}

/// `update_keeping` is `update` to the bit where the history's decayed
/// weight is at least the floor (or its own weight), and where it is
/// below, reads the history at the floor: the moments are `update`'s with
/// the history's share `H / (H + w)`, `H` the floor up to the history's
/// weight, within rounding, and the weight kept is `lam·W + w`, exactly.
#[test]
fn update_keeping_is_update_but_where_the_floor_binds() {
    let mut s = 91u64;
    for _ in 0..200 {
        let k = 3;
        let mut sc = EwDiag::new(k);
        for _ in 0..(1 + (lcg(&mut s).abs() * 30.0) as usize) {
            let x: Vec<f64> = (0..k).map(|i| (i as f64 + 1.0) * lcg(&mut s)).collect();
            sc.update(&x, 0.9, 0.5 + lcg(&mut s).abs());
        }
        let x: Vec<f64> = (0..k).map(|_| 3.0 * lcg(&mut s)).collect();
        let (lam, w) = (
            2f64.powf(-30.0 * lcg(&mut s).abs()),
            0.5 + lcg(&mut s).abs(),
        );
        let (w_sum, floor) = (sc.n_eff(), w);
        let (mut kept, mut plain) = (sc.clone(), sc.clone());
        let bound = kept.update_keeping(&x, lam, w, floor, w_sum);
        assert_eq!(
            bound,
            lam * w_sum < floor.min(w_sum),
            "it binds where the history is lighter, up to its own weight"
        );
        assert_eq!(
            kept.n_eff().to_bits(),
            (lam * w_sum + w).to_bits(),
            "the weight kept"
        );
        if bound && floor <= w_sum {
            // `update` with the history's weight taken to the floor.
            plain.update(&x, floor / w_sum, w);
            for i in 0..k {
                for (a, b) in [(kept.mean(i), plain.mean(i)), (kept.var(i), plain.var(i))] {
                    assert!(
                        (a - b).abs() <= 1e-14 * b.abs().max(1e-300),
                        "{a:e} against {b:e}"
                    );
                }
            }
        } else if !bound {
            plain.update(&x, lam, w);
            for i in 0..k {
                assert_eq!(kept.mean(i).to_bits(), plain.mean(i).to_bits());
                assert_eq!(kept.var(i).to_bits(), plain.var(i).to_bits());
            }
        }
    }
}
