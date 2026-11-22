//! The online form against the definitions (docs/PLAN.md task 555, part 2):
//! one pass, a decay, the windows, and the rules every model keeps.

use super::paper;
use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn normal(state: &mut u64) -> f64 {
    let (u, v) = (lcg(state), lcg(state));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// `n` returns of a log price with a stochastic volatility, observed with
/// noise: the efficient return plus the difference of two noise draws, so
/// the returns are negatively autocorrelated at lag 1 (the bid-ask bounce
/// the kernel sums are for), with a jump at 40% of the stream.
fn noisy_day(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let base = 0.01 / (n as f64).sqrt();
    let kappa = std::f64::consts::LN_2 / (n as f64 / 10.0);
    let (mut h, mut e_prev) = (0.0f64, 0.0f64);
    let mut r: Vec<f64> = (0..n)
        .map(|_| {
            h = h * (-kappa).exp() + 0.3 * (2.0 * kappa).sqrt() * normal(&mut s);
            let e = 0.5 * base * normal(&mut s);
            let x = base * h.exp() * normal(&mut s) + e - e_prev;
            e_prev = e;
            x
        })
        .collect();
    let j = 2 * n / 5;
    r[j] += 8.0 * base;
    r
}

/// The distance of `a` from `b` in units in the last place of the larger
/// of `|b|` and `scale`: a sum whose terms cancel is held to the rounding
/// of its terms' size, not of what is left.
fn gap(a: f64, b: f64, scale: f64) -> f64 {
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    let m = b.abs().max(scale.abs());
    let ulp = f64::from_bits(m.to_bits() + 1) - m;
    (a - b).abs() / ulp
}

fn mp(powers: &[f64], scale: usize, normalize: bool) -> PvTerm {
    PvTerm {
        kind: PvTermKind::Multipower {
            powers: powers.to_vec(),
            normalize,
        },
        scale,
    }
}

fn ker(kernel: PvKernel, bandwidth: usize, flat_top: bool, center: bool, scale: usize) -> PvTerm {
    PvTerm {
        kind: PvTermKind::Kernel {
            kernel,
            bandwidth,
            flat_top,
            center,
        },
        scale,
    }
}

/// Every form a term takes: the four of realized variance, bipower,
/// tripower variation and quarticity, unequal powers, each kernel, flat-top
/// and centred, and scales past 1.
fn terms() -> Vec<PvTerm> {
    let (t, q) = (2.0 / 3.0, 4.0 / 3.0);
    vec![
        mp(&[2.0], 1, false),
        mp(&[1.0, 1.0], 1, false),
        mp(&[t, t, t], 1, false),
        mp(&[q, q, q], 1, true),
        mp(&[1.0, 0.5], 1, false),
        ker(PvKernel::Parzen, 5, false, false, 1),
        ker(PvKernel::Bartlett, 3, true, false, 1),
        ker(PvKernel::TukeyHanning, 4, false, true, 1),
        ker(PvKernel::Rectangular, 2, false, false, 3),
        mp(&[2.0], 5, false),
        mp(&[1.0, 1.0], 2, false),
        ker(PvKernel::Bartlett, 2, false, true, 4),
    ]
}

fn cfg(n: usize, terms: Vec<PvTerm>, half_life: f64) -> PowerVariationCfg {
    PowerVariationCfg {
        n_targets: n,
        terms,
        decay: Decay::Halflife(half_life),
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

type Row = (f64, Vec<Option<f64>>, f64);

fn run(m: &mut PowerVariation, rows: &[Row]) -> Vec<Step> {
    rows.iter()
        .map(|(d, y, w)| m.step(&[], y, *d, *w))
        .collect()
}

fn unit_rows(r: &[f64]) -> Vec<Row> {
    r.iter().map(|&x| (1.0, vec![Some(x)], 1.0)).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn name(k: PvKernel) -> &'static str {
    match k {
        PvKernel::Bartlett => "bartlett",
        PvKernel::Parzen => "parzen",
        PvKernel::TukeyHanning => "tukey_hanning",
        PvKernel::Rectangular => "rectangular",
    }
}

/// The definition of `t` on the returns `r` at the final weights `om`,
/// `n` the count its normalization reads; and the scale its gap is read
/// against, `Σ ω R² / K` for a kernel sum (whose products cancel).
fn definition(t: &PvTerm, r: &[f64], om: &[f64], n: f64) -> (f64, f64) {
    match &t.kind {
        PvTermKind::Multipower { powers, normalize } => {
            let s = paper::multipower(r, om, powers, t.scale);
            let v = if *normalize {
                let mu: f64 = powers
                    .iter()
                    .map(|&p| series::abs_normal_moment(p))
                    .product();
                let e = powers.iter().sum::<f64>() / 2.0 - 1.0;
                if n > 0.0 {
                    s * n.powf(e) / mu
                } else {
                    f64::NAN
                }
            } else {
                s
            };
            (v, 0.0)
        }
        PvTermKind::Kernel {
            kernel,
            bandwidth,
            flat_top,
            center,
        } => {
            let c = paper::weights(name(*kernel), *bandwidth, *flat_top);
            (
                paper::kernel_sum(r, om, &c, *center, t.scale),
                paper::multipower(r, om, &[2.0], t.scale) * (1 + 2 * bandwidth) as f64,
            )
        }
    }
}

/// Each row's outputs against the definitions on the returns `held(i)`
/// before it, at their final weights; the worst gap of each output.
fn worst_gaps(
    terms: &[PvTerm],
    steps: &[Step],
    held: impl Fn(usize) -> (Vec<f64>, Vec<f64>, f64),
) -> Vec<f64> {
    let mut worst = vec![0.0f64; terms.len() + 1];
    for (i, s) in steps.iter().enumerate() {
        let (r, om, n) = held(i);
        for (k, t) in terms.iter().enumerate() {
            let (want, scale) = definition(t, &r, &om, n);
            let g = gap(s.pred[k], want, scale);
            assert!(
                g.is_finite(),
                "row {i} term {k}: {} against {want}",
                s.pred[k]
            );
            worst[k] = worst[k].max(g);
        }
        worst[terms.len()] = worst[terms.len()].max(gap(s.pred[terms.len()], n, 0.0));
    }
    worst
}

/// One pass, unit weights, no decay: every row's outputs are the
/// definitions on the returns before it. Measured over three seeds (ulps,
/// of the larger of the value and `(2H + 1) Σ R²` for a kernel sum): at
/// most 3 (the quarticity's powers), the kernel sums under 1.
#[test]
fn one_pass_is_the_definition_on_every_prefix() {
    let terms = terms();
    for seed in [1, 2, 3] {
        let r = noisy_day(600, seed);
        let mut m = PowerVariation::new(cfg(1, terms.clone(), f64::INFINITY)).unwrap();
        let steps = run(&mut m, &unit_rows(&r));
        let worst = worst_gaps(&terms, &steps, |i| {
            (r[..i].to_vec(), vec![1.0; i], i as f64)
        });
        eprintln!("one pass, seed {seed}: {worst:?}");
        for (k, w) in worst.iter().enumerate() {
            assert!(*w <= 4.0, "output {k}: {w} ulps ({worst:?})");
        }
    }
}

/// Under a decay, on an irregular clock with uneven weights: the
/// definitions at each return's weight decayed by its age taken in one
/// power. The online form multiplies by each row's factor, so its error
/// grows with the rows a half-life holds: measured 43 to 85 ulps on the
/// multipower terms and `n`, 4 to 11 on the kernel sums (read against
/// their scale), at a half-life of 50 rows over 800.
#[test]
fn decay_is_the_weighted_definition() {
    let terms = terms();
    let r = noisy_day(800, 4);
    let h = 50.0;
    let mut s = 9;
    let rows: Vec<Row> = r
        .iter()
        .map(|&x| (-lcg(&mut s).ln(), vec![Some(x)], 0.5 + lcg(&mut s)))
        .collect();
    let mut clock = vec![0.0f64; rows.len()];
    for i in 1..rows.len() {
        clock[i] = clock[i - 1] + rows[i].0;
    }
    let mut m = PowerVariation::new(cfg(1, terms.clone(), h)).unwrap();
    let steps = run(&mut m, &rows);
    let worst = worst_gaps(&terms, &steps, |i| {
        if i == 0 {
            return (vec![], vec![], 0.0);
        }
        let om: Vec<f64> = (0..i)
            .map(|j| rows[j].2 * (-std::f64::consts::LN_2 * (clock[i - 1] - clock[j]) / h).exp())
            .collect();
        let n = paper::kish(&om);
        (r[..i].to_vec(), om, n)
    });
    eprintln!("decay: {worst:?}");
    for (k, w) in worst.iter().enumerate() {
        assert!(*w <= 128.0, "output {k}: {w} ulps ({worst:?})");
    }
}

/// A window of `W` rows: the definitions on the returns of the last `W`
/// rows, every term whose returns are all inside, from 1 row up. Measured:
/// at most 3 ulps, the kernel sums under 1.
#[test]
fn a_window_of_rows_is_the_definition_on_its_returns() {
    let terms = terms();
    let r = noisy_day(500, 5);
    for w in [1usize, 2, 3, 7, 50, 400] {
        let mut c = cfg(1, terms.clone(), f64::INFINITY);
        c.window_rows = Some(w);
        let mut m = PowerVariation::new(c).unwrap();
        let steps = run(&mut m, &unit_rows(&r));
        let worst = worst_gaps(&terms, &steps, |i| {
            let lo = i.saturating_sub(w);
            (r[lo..i].to_vec(), vec![1.0; i - lo], (i - lo) as f64)
        });
        eprintln!("window {w}: {worst:?}");
        for (k, g) in worst.iter().enumerate() {
            assert!(*g <= 4.0, "window {w}, output {k}: {g} ulps ({worst:?})");
        }
    }
}

/// A window of the clock: a row reads the rows of `(t − D, t]` of the row
/// before it (`Right`), or `[t − D, t]` (`Both`), on stamps that land
/// exactly on the window's edge. Measured: at most 3 ulps.
#[test]
fn a_window_of_the_clock_is_the_definition_on_its_returns() {
    let terms = terms();
    let r = noisy_day(400, 6);
    let mut s = 21;
    let steps_d: Vec<f64> = (0..r.len())
        .map(|_| [0.25, 0.5, 1.0, 2.0][(lcg(&mut s) * 4.0) as usize])
        .collect();
    let mut clock = vec![0.0f64; r.len()];
    let mut own = 0.0;
    for i in 0..r.len() {
        own += steps_d[i];
        clock[i] = own;
    }
    for closed in [WindowClosed::Right, WindowClosed::Both] {
        let mut c = cfg(1, terms.clone(), f64::INFINITY);
        c.window_clock = Some(5.0);
        c.closed = closed;
        let mut m = PowerVariation::new(c).unwrap();
        let rows: Vec<Row> = (0..r.len())
            .map(|i| (steps_d[i], vec![Some(r[i])], 1.0))
            .collect();
        let steps = run(&mut m, &rows);
        let worst = worst_gaps(&terms, &steps, |i| {
            if i == 0 {
                return (vec![], vec![], 0.0);
            }
            let now = clock[i - 1];
            let lo = (0..i)
                .find(|&j| match closed {
                    WindowClosed::Right => now - clock[j] < 5.0,
                    _ => now - clock[j] <= 5.0,
                })
                .unwrap_or(i);
            (r[lo..i].to_vec(), vec![1.0; i - lo], (i - lo) as f64)
        });
        eprintln!("clock window {closed:?}: {worst:?}");
        for (k, g) in worst.iter().enumerate() {
            assert!(*g <= 4.0, "{closed:?}, output {k}: {g} ulps ({worst:?})");
        }
    }
}

/// Normalized, the four multipower terms are `realized`'s `rv`, `bv`, `tv`
/// and `tq` (its definitions; its arithmetic takes cube roots where this
/// takes powers). Measured (ulps): 0, 0, 5, 9 one pass; 0, 0, 9, 10 at a
/// half-life of 80 rows.
#[test]
fn normalized_terms_are_realized_measures() {
    let (t, q) = (2.0 / 3.0, 4.0 / 3.0);
    let terms = vec![
        mp(&[2.0], 1, true),
        mp(&[1.0, 1.0], 1, true),
        mp(&[t, t, t], 1, true),
        mp(&[q, q, q], 1, true),
    ];
    let r = noisy_day(1_000, 7);
    for h in [f64::INFINITY, 80.0] {
        let mut a = PowerVariation::new(cfg(1, terms.clone(), h)).unwrap();
        let mut b = crate::Realized::new(crate::RealizedCfg {
            n_targets: 1,
            decay: Decay::Halflife(h),
            min_weight: 0.0,
            window_rows: None,
            window_clock: None,
            closed: WindowClosed::Right,
            lm_window: None,
            lm_beta: crate::REALIZED_LM_BETA,
            max_state_bytes: None,
        })
        .unwrap();
        let mut worst = [0.0f64; 4];
        for (d, y, w) in unit_rows(&r) {
            let x = a.step(&[], &y, d, w);
            let z = b.step(&[], &y, d, w);
            for (k, g) in worst.iter_mut().enumerate() {
                *g = g.max(gap(x.pred[k], z.pred[k], 0.0));
            }
        }
        eprintln!("against realized, half-life {h}: {worst:?}");
        assert!(worst[..2].iter().all(|w| *w == 0.0), "{worst:?}");
        assert!(worst.iter().all(|w| *w <= 16.0), "{worst:?}");
    }
}

/// `μ_p` against the constants `realized` holds, correctly rounded from 40
/// digits: `μ_1^-2 = π/2`, `μ_{2/3}^-3`, `μ_{4/3}^-3`. Measured (ulps, the
/// power's own roundings included): 3, 5 and 7.
#[test]
fn the_moments_are_the_normals() {
    let m = series::abs_normal_moment;
    assert!(gap(m(1.0).powi(-2), crate::BV_SCALE, 0.0) <= 4.0);
    assert!(gap(m(2.0 / 3.0).powi(-3), crate::TV_SCALE, 0.0) <= 8.0);
    assert!(gap(m(4.0 / 3.0).powi(-3), crate::TQ_SCALE, 0.0) <= 12.0);
    assert_eq!((m(2.0), m(4.0), m(6.0)), (1.0, 3.0, 15.0));
}

/// A save at any point goes on to the bit: two targets, an absent return,
/// uneven weights, under a decay and under each window.
#[test]
fn a_saved_state_goes_on_to_the_bit() {
    let (r0, r1) = (noisy_day(300, 12), noisy_day(300, 13));
    let mut s = 3;
    let rows: Vec<Row> = (0..r0.len())
        .map(|i| {
            (
                [0.5, 1.0, 1.5][(lcg(&mut s) * 3.0) as usize],
                vec![Some(r0[i]), (i % 7 != 2).then_some(r1[i])],
                0.5 + lcg(&mut s),
            )
        })
        .collect();
    let forms: [(f64, Option<usize>, Option<f64>); 3] = [
        (40.0, None, None),
        (f64::INFINITY, Some(30), None),
        (f64::INFINITY, None, Some(12.0)),
    ];
    for (h, wr, wc) in forms {
        for cut in [0, 1, 2, 3, 4, 97, 250] {
            let mut c = cfg(2, terms(), h);
            (c.window_rows, c.window_clock) = (wr, wc);
            let mut m = PowerVariation::new(c).unwrap();
            run(&mut m, &rows[..cut]);
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            let mut back =
                PowerVariation::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            assert_eq!(m, back);
            for (d, y, w) in &rows[cut..] {
                let a = m.step(&[], y, *d, *w);
                let b = back.step(&[], y, *d, *w);
                assert_eq!(bits(&a.pred), bits(&b.pred), "cut {cut}");
                assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
            }
            assert_eq!(m, back, "cut {cut}");
        }
    }
}

/// A row of weight 0 learns nothing (hard rule 9): without a decay, the
/// rows after it read what they read without it, to the bit, under a
/// window too.
#[test]
fn a_row_of_weight_zero_learns_nothing() {
    let r = noisy_day(300, 14);
    let rows: Vec<Row> = r
        .iter()
        .enumerate()
        .map(|(i, &x)| (1.0, vec![Some(x)], if i % 5 == 3 { 0.0 } else { 1.0 }))
        .collect();
    let kept: Vec<Row> = rows.iter().filter(|r| r.2 > 0.0).cloned().collect();
    for wr in [None, Some(9)] {
        let mut c = cfg(1, terms(), f64::INFINITY);
        c.window_rows = wr;
        let mut a = PowerVariation::new(c.clone()).unwrap();
        let mut b = PowerVariation::new(c).unwrap();
        let sa: Vec<Step> = run(&mut a, &rows)
            .into_iter()
            .zip(&rows)
            .filter(|(_, r)| r.2 > 0.0)
            .map(|(s, _)| s)
            .collect();
        let sb = run(&mut b, &kept);
        for (x, y) in sa.iter().zip(&sb) {
            assert_eq!(bits(&x.pred), bits(&y.pred), "window {wr:?}");
        }
    }
}

/// An absent return is its own series' alone: its terms reach across the
/// row, so the target reads what a model of its present returns reads.
#[test]
fn an_absent_return_is_skipped_by_its_own_series_alone() {
    let (r0, r1) = (noisy_day(300, 15), noisy_day(300, 16));
    let rows: Vec<Row> = (0..r0.len())
        .map(|i| (1.0, vec![Some(r0[i]), (i % 4 != 1).then_some(r1[i])], 1.0))
        .collect();
    let alone: Vec<Row> = rows
        .iter()
        .filter_map(|(d, y, w)| y[1].map(|x| (*d, vec![Some(x)], *w)))
        .collect();
    let t = terms().len() + 1;
    let mut both = PowerVariation::new(cfg(2, terms(), f64::INFINITY)).unwrap();
    let mut one = PowerVariation::new(cfg(1, terms(), f64::INFINITY)).unwrap();
    let a: Vec<Step> = run(&mut both, &rows)
        .into_iter()
        .zip(&rows)
        .filter(|(_, r)| r.1[1].is_some())
        .map(|(s, _)| s)
        .collect();
    let b = run(&mut one, &alone);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(bits(&x.pred[t..]), bits(&y.pred));
    }
}

/// Out of sample: a row's own return moves none of its outputs; the next
/// row's do. `predict_with` reads what the step reads.
#[test]
fn a_row_reports_from_the_returns_before_it() {
    let r = noisy_day(200, 17);
    let mut a = PowerVariation::new(cfg(1, terms(), 30.0)).unwrap();
    let mut b = a.clone();
    run(&mut a, &unit_rows(&r[..100]));
    run(&mut b, &unit_rows(&r[..100]));
    let p = a.predict_with(&[], &[Some(5.0)], 1.0);
    let x = a.step(&[], &[Some(5.0)], 1.0, 1.0);
    let y = b.step(&[], &[Some(-0.001)], 1.0, 1.0);
    assert_eq!(bits(&x.pred), bits(&y.pred));
    assert_eq!(bits(&p.pred), bits(&x.pred));
    let x2 = a.step(&[], &[Some(0.0)], 1.0, 1.0);
    let y2 = b.step(&[], &[Some(0.0)], 1.0, 1.0);
    assert_ne!(bits(&x2.pred), bits(&y2.pred));
}

/// A target under `min_weight` reports `NaN` on every output, `n` too; the
/// other target is gated by its own weight.
#[test]
fn min_weight_gates_each_series_on_its_own_weight() {
    let mut c = cfg(2, terms(), f64::INFINITY);
    c.min_weight = 3.0;
    let mut m = PowerVariation::new(c).unwrap();
    let t = terms().len() + 1;
    for i in 0..6 {
        let s = m.step(&[], &[Some(0.01), (i < 2).then_some(0.02)], 1.0, 1.0);
        assert_eq!(s.pred[..t].iter().all(|x| x.is_nan()), i < 3, "row {i}");
        assert!(s.pred[t..].iter().all(|x| x.is_nan()), "row {i}");
    }
}

/// `n` is the count of returns, Kish's count under a decay, and the state
/// holds the numbers its byte count says, a window's rows included.
#[test]
fn n_is_the_count_and_the_state_is_the_bytes_it_says() {
    let r = noisy_day(400, 18);
    let t = terms().len();
    let mut m = PowerVariation::new(cfg(1, terms(), f64::INFINITY)).unwrap();
    let steps = run(&mut m, &unit_rows(&r));
    assert_eq!(steps[399].pred[t], 399.0);
    let counted = |m: &PowerVariation| {
        8 * (1 + m.series.iter().map(Series::numbers).sum::<usize>()) as u64
            + m.window
                .as_ref()
                .map_or(0, |w| 32 + 16 * w.rows.len() as u64)
    };
    assert_eq!(counted(&m), m.bytes());
    let mut c = cfg(1, terms(), f64::INFINITY);
    c.window_rows = Some(40);
    let mut w = PowerVariation::new(c).unwrap();
    run(&mut w, &unit_rows(&r));
    assert_eq!(counted(&w), w.bytes());
    assert_eq!(w.bytes(), w.cfg.fixed_bytes(&w.plan));
}

/// The settings each refusal names, and a window past the cap is refused
/// naming the largest that fits, which is accepted.
#[test]
fn the_settings_are_checked() {
    let refused = |c: PowerVariationCfg, says: &str| {
        let e = PowerVariation::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    refused(cfg(1, vec![], f64::INFINITY), "at least one term");
    refused(cfg(0, terms(), f64::INFINITY), "at least one target");
    refused(cfg(1, vec![mp(&[], 1, false)], f64::INFINITY), "no powers");
    refused(
        cfg(1, vec![mp(&[1.0, 0.0], 1, false)], f64::INFINITY),
        "above 0",
    );
    refused(
        cfg(1, vec![mp(&[2.0], 0, false)], f64::INFINITY),
        "at least 1, but it is 0",
    );
    refused(
        cfg(1, vec![mp(&[2.0], 2, true)], f64::INFINITY),
        "normalized at a scale",
    );
    refused(
        cfg(
            1,
            vec![ker(PvKernel::Parzen, 0, true, false, 1)],
            f64::INFINITY,
        ),
        "flat-top",
    );
    refused(
        cfg(
            1,
            vec![ker(PvKernel::Parzen, 1 << 20, false, false, 2)],
            f64::INFINITY,
        ),
        "reaches back",
    );
    let mut c = cfg(1, terms(), 10.0);
    c.window_rows = Some(5);
    refused(c, "takes no decay");
    let mut c = cfg(2, terms(), f64::INFINITY);
    c.window_rows = Some(100_000);
    let e = PowerVariation::new(c.clone()).unwrap_err();
    let fits: usize = e
        .split("largest window_rows that fits is ")
        .nth(1)
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("{e}"));
    c.window_rows = Some(fits);
    PowerVariation::new(c.clone()).unwrap();
    c.window_rows = Some(fits + 1);
    PowerVariation::new(c).unwrap_err();
}

/// A restored state of the wrong shape is refused, not read.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = PowerVariation::new(cfg(1, terms(), f64::INFINITY)).unwrap();
    run(&mut m, &unit_rows(&noisy_day(50, 19)));
    let mut bad = m.clone();
    bad.cfg.terms.pop();
    let e = PowerVariation::restore(&bad.state())
        .unwrap_err()
        .to_string();
    assert!(e.contains("wrong shape"), "{e}");
    let mut bad = m.clone();
    bad.cfg.n_targets = 2;
    assert!(PowerVariation::restore(&bad.state()).is_err());
}

/// A clock window holds the rows its span has, so it is checked as they go
/// in: past half the cap a warning, once; the row that would pass the cap
/// is not learned, nor any after it, and the overrun is reported.
#[test]
fn a_clock_window_is_held_to_its_cap_as_the_rows_go_in() {
    let r = noisy_day(400, 20);
    let mut c = cfg(1, terms(), f64::INFINITY);
    c.window_clock = Some(1.0e9);
    let fixed = PowerVariation::new(c.clone()).unwrap().bytes();
    let row = PowerVariation::window_row_bytes(&Plan::of(&c.terms), 1, true);
    c.max_state_bytes = Some(fixed + 100 * row);
    let mut m = PowerVariation::new(c).unwrap();
    let mut warned = None;
    for (i, row) in unit_rows(&r).iter().enumerate() {
        m.step(&[], &row.1, row.0, row.2);
        if let Some(w) = m.take_state_warning() {
            assert!(warned.replace(i).is_none(), "warned twice");
            assert!(2 * w.0 > w.1);
        }
    }
    assert_eq!(m.window_len(), 100);
    let (over, cap) = m.state_over_cap().expect("over the cap");
    assert!(over > cap && m.bytes() <= cap);
    assert!(warned.is_some_and(|i| i < 100));
}
