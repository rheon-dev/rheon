use super::*;
use crate::Decay;
use crate::dist_fit::{DistFamily, DistFit, DistFitCfg, DistWithheld};
use crate::model::{OnlineModel, State, Step};

fn cfg(half_life: f64) -> DistFitCfg {
    DistFitCfg {
        n_targets: 1,
        family: DistFamily::StudentT,
        decay: Decay::Halflife(half_life),
        min_weight: 0.0,
        quantiles: vec![0.01, 0.5, 0.95],
        es_levels: vec![0.05],
        floc: None,
    }
}

/// A seeded uniform on (0, 1).
struct Lcg(u64);

impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()
    }
    /// Student's t of integer `nu` degrees of freedom (0: the normal).
    fn t(&mut self, nu: u32) -> f64 {
        let z = self.normal();
        if nu == 0 {
            return z;
        }
        let chi: f64 = (0..nu).map(|_| self.normal().powi(2)).sum();
        z / (chi / f64::from(nu)).sqrt()
    }
}

fn t_rows(n: usize, seed: u64, nu: u32, loc: f64, scale: f64) -> Vec<f64> {
    let mut r = Lcg(seed);
    (0..n).map(|_| loc + scale * r.t(nu)).collect()
}

fn bits(s: &Step) -> Vec<u64> {
    s.pred.iter().map(|v| v.to_bits()).collect()
}

/// The gradient of the whole weighted log likelihood at `θ` over `rows`
/// (value, weight already decayed), and its information `Σ ω I`.
fn exact_gradient(rows: &[(f64, f64)], theta: &[f64; 3]) -> ([f64; 3], [f64; 6]) {
    let mut g = [0.0; 3];
    let mut h = [0.0; 6];
    for &(y, om) in rows {
        let e = eval(y, theta).unwrap();
        for (gk, ek) in g.iter_mut().zip(&e.g) {
            *gk += om * ek;
        }
        axpy(&mut h, om, &e.exp);
    }
    (g, h)
}

/// A long t stream: the fit near the truth, its tail identified, and the
/// fit a stationary point of the whole weighted likelihood to within a
/// small part of the noise: the exact gradient's decrement against the
/// bound the convergence test holds the surrogate's to.
#[test]
fn a_long_t_stream_is_fitted_near_the_truth_and_the_maximum() {
    let hl = 3000.0;
    let ys = t_rows(30_000, 21, 5, 2.0, 0.5);
    let mut m = DistFit::new(cfg(hl)).unwrap();
    for &y in &ys {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    assert_eq!(m.withheld(0), None);
    let (loc, scale) = m.params(0);
    let df = m.df(0).unwrap();
    assert!((loc - 2.0).abs() < 0.03, "{loc}");
    assert!((scale - 0.5).abs() < 0.03, "{scale}");
    assert!((df - 5.0).abs() < 1.5, "{df}");
    let lam = Decay::Halflife(hl).factor(1.0);
    let n = ys.len();
    let rows: Vec<(f64, f64)> = ys
        .iter()
        .enumerate()
        .map(|(i, &y)| (y, lam.powi((n - 1 - i) as i32)))
        .collect();
    let theta = m.t[0].theta().unwrap();
    let (g, h) = exact_gradient(&rows, &theta);
    let step = chol_solve(&full(&h), &g, [true; 3]).unwrap();
    let dec: f64 = (0..3).map(|k| step[k] * g[k]).sum();
    let w: f64 = rows.iter().map(|r| r.1).sum();
    let w2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
    // Measured 0.0027 against a noise bound of 3.9: held at a twentieth of
    // the bound.
    assert!(dec < 0.05 * w2 / w * CHI2_95[2], "{dec}");
}

fn theta(m: &DistFit) -> [f64; 3] {
    m.t[0].theta().unwrap()
}

/// The exact decrement of the whole weighted likelihood at the fit, over
/// `ys` at unit weights and the decay `lam` a row.
fn exact_decrement(ys: &[f64], lam: f64, th: &[f64; 3]) -> (f64, f64) {
    let n = ys.len();
    let rows: Vec<(f64, f64)> = ys
        .iter()
        .enumerate()
        .map(|(i, &y)| (y, lam.powi((n - 1 - i) as i32)))
        .collect();
    let (g, h) = exact_gradient(&rows, th);
    let step = chol_solve(&full(&h), &g, [true; 3]).unwrap();
    let w: f64 = rows.iter().map(|r| r.1).sum();
    let w2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
    ((0..3).map(|k| step[k] * g[k]).sum(), w2 / w)
}

/// One row far out on near-normal data moves the tail about as far as the
/// whole likelihood says, not as far as the score's `z⁴` would throw a
/// single Fisher step (0.33 for the first, by `(z⁴ − 2z² − 1)/4` over the
/// rows' information `1.5 W`, and the bound for the others): the fit stays
/// near the maximum after it.
#[test]
fn an_outlier_moves_the_tail_no_further_than_the_likelihood() {
    let mut ys = t_rows(2000, 5, 0, 0.0, 1.0);
    let mut m = DistFit::new(cfg(f64::INFINITY)).unwrap();
    for &y in &ys {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    for out in [8.0, 30.0, 1e3] {
        m.step(&[], &[Some(out)], 1.0, 1.0);
        ys.push(out);
        let th = theta(&m);
        let (dec, noise) = exact_decrement(&ys, 1.0, &th);
        assert!(th[2] < 0.5, "{out}: {th:?}");
        // Inside the likelihood's 95% region about its maximum, but for a
        // value a thousand standard deviations out, which a quadratic of
        // the rows before it cannot carry: measured 2.2, 3.9 and 8.6.
        if out < 100.0 {
            assert!(dec < noise * CHI2_95[2], "{out}: {dec}");
        }
    }
}

/// Normal data: the tail is seldom told from the normal's, and while it is
/// not, `loc` and `scale` report and nothing that reads the tail does.
#[test]
fn a_normal_stream_leaves_the_tail_unidentified() {
    let mut identified = 0;
    for seed in 0..20 {
        let ys = t_rows(3000, 100 + seed, 0, 1.0, 2.0);
        let mut m = DistFit::new(cfg(1000.0)).unwrap();
        for &y in &ys {
            m.step(&[], &[Some(y)], 1.0, 1.0);
        }
        let s = m.predict_with(&[], &[Some(1.0)], 1.0);
        match m.withheld(0) {
            Some(DistWithheld::ShapeUnidentified) => {
                assert!(s.pred[0].is_finite() && s.pred[1].is_finite());
                assert!(s.pred[2..].iter().all(|v| v.is_nan()));
            }
            None => identified += 1,
            other => {
                let th = m.t[0].theta().unwrap();
                let (d, n) = exact_decrement(&ys, Decay::Halflife(1000.0).factor(1.0), &th);
                panic!("{seed}: {other:?} theta {th:?} exact dec {d} noise {n}");
            }
        }
        let (loc, scale) = m.params(0);
        assert!(
            (loc - 1.0).abs() < 0.2 && (scale - 2.0).abs() < 0.2,
            "{loc} {scale}"
        );
    }
    assert!(identified <= 4, "{identified} of 20");
}

/// Before the start every slot is withheld with its reason; the start
/// comes on the first row whose moments give a positive variance and pin
/// the scale to a standard error of 1/4, here the 19th of equal weight.
#[test]
fn the_start_waits_for_a_variance_and_its_precision() {
    let mut m = DistFit::new(cfg(f64::INFINITY)).unwrap();
    assert_eq!(m.withheld(0), Some(DistWithheld::NoWeight));
    m.step(&[], &[Some(1.0)], 1.0, 1.0);
    m.step(&[], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.withheld(0), Some(DistWithheld::DegenerateVariance));
    // Alternate two values: excess kurtosis −2, the normal's tail at the
    // start, whose `[I⁻¹]_{ln s}` is 7/6, so 1/n ≤ 6/7 · 1/16.
    let mut n = 2;
    while !m.t[0].started() {
        let s = m.step(&[], &[Some(if n % 2 == 0 { 1.5 } else { 1.0 })], 1.0, 1.0);
        n += 1;
        if !m.t[0].started() {
            assert_eq!(m.withheld(0), Some(DistWithheld::NotConverged), "{n}");
            assert!(s.pred.iter().all(|v| v.is_nan()));
        }
    }
    assert_eq!(n, 19);
    let th = m.t[0].theta().unwrap();
    assert_eq!(th[2], 0.0);
    let (loc, scale) = (m.fit[0].mean(0), m.fit[0].var(0).sqrt());
    assert!((th[0] - loc).abs() < 1e-15 && (libm::exp(th[1]) - scale).abs() < 1e-15);
    let s = m.predict(&[], 1.0);
    assert!(
        s.pred[0].is_finite(),
        "converged at the start: {:?}",
        m.withheld(0)
    );
}

/// The moment start against its definition on a heavy-tailed stream: the
/// kurtosis of the weighted moments, written out longhand.
#[test]
fn the_moment_start_is_its_definition() {
    for floc in [None, Some(0.3)] {
        let ys = t_rows(40, 9, 5, 0.5, 1.0);
        let mut t = TFit::new();
        let mut f = EwDiag::new(1);
        let lam = 0.97;
        let shift = floc.unwrap_or(0.0);
        for (i, &y) in ys.iter().enumerate() {
            let w = 0.5 + (i % 3) as f64;
            t.decay(lam);
            t.learn_moments(&f, y - shift, lam, w);
            f.update_one(y - shift, lam, w);
        }
        assert!(t.try_start(&f, floc));
        let n = ys.len();
        let om: Vec<f64> = (0..n)
            .map(|i| (0.5 + (i % 3) as f64) * lam.powi((n - 1 - i) as i32))
            .collect();
        let w: f64 = om.iter().sum();
        let c = floc.unwrap_or_else(|| om.iter().zip(&ys).map(|(o, y)| o * y).sum::<f64>() / w);
        let mom = |p: i32| {
            om.iter()
                .zip(&ys)
                .map(|(o, y)| o * (y - c).powi(p))
                .sum::<f64>()
                / w
        };
        let kurt = mom(4) / mom(2).powi(2) - 3.0;
        let eta = if kurt > 0.0 {
            kurt / (4.0 * kurt + 6.0)
        } else {
            0.0
        };
        let th = t.theta().unwrap();
        assert!((th[0] - c).abs() < 1e-13, "{floc:?}");
        assert!((th[2] - eta).abs() < 1e-12, "{floc:?}: {} {eta}", th[2]);
        let scale = (mom(2) * (1.0 - 2.0 * eta)).sqrt();
        assert!((libm::exp(th[1]) - scale).abs() < 1e-12, "{floc:?}");
        let w2: f64 = om.iter().map(|o| o * o).sum();
        assert!((t.w2 - w2).abs() < 1e-12 * w2);
    }
}

fn cfg2(floc: Option<f64>) -> DistFitCfg {
    DistFitCfg {
        n_targets: 2,
        floc,
        ..cfg(60.0)
    }
}

/// Two targets of t draws, irregular clock deltas, unequal weights, a few
/// rows of weight 0, the second target absent on every fifth row.
fn stream(n: usize, seed: u64) -> Vec<(Vec<Option<f64>>, f64, f64)> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|i| {
            let y = vec![
                Some(1.0 + 0.5 * r.t(4)),
                (i % 5 != 4).then(|| -2.0 + 3.0 * r.t(3)),
            ];
            let d = 0.5 + r.u();
            let w = if i % 17 == 16 { 0.0 } else { 0.25 + r.u() };
            (y, d, w)
        })
        .collect()
}

/// Every output of a row is read from the fit before it, under both
/// locations, and each slot is its definition at that fit.
#[test]
fn every_output_is_out_of_sample_and_its_definition() {
    for floc in [None, Some(1.0)] {
        let c = cfg2(floc);
        let mut m = DistFit::new(c.clone()).unwrap();
        let mut seen = [0usize; 3];
        for (y, d, w) in stream(600, 3) {
            let before = m.predict_with(&[], &y, d);
            let reasons: Vec<_> = (0..2).map(|j| m.withheld(j)).collect();
            let params: Vec<_> = (0..2).map(|j| (m.params(j), m.df(j))).collect();
            let etas: Vec<_> = (0..2).map(|j| m.t[j].theta().map(|t| t[2])).collect();
            let s = m.step(&[], &y, d, w);
            assert_eq!(bits(&before), bits(&s));
            for (j, slots) in s.pred.chunks(c.slots()).enumerate() {
                match reasons[j] {
                    None => {
                        seen[0] += 1;
                        let ((loc, scale), df) = (params[j].0, params[j].1.unwrap());
                        let eta = etas[j].unwrap();
                        assert_eq!(slots[..3], [loc, scale, df]);
                        if let Some(l) = floc {
                            assert_eq!(loc, l);
                        }
                        for (k, p) in c.quantiles.iter().enumerate() {
                            let want = loc + scale * student_t::ppf(*p, eta);
                            assert_eq!(slots[3 + k].to_bits(), want.to_bits());
                        }
                        let es = loc + scale * student_t::lower_es(0.05, eta);
                        assert_eq!(slots[6].to_bits(), es.to_bits());
                        if let Some(v) = y[j] {
                            let z = (v - loc) / scale;
                            assert_eq!(slots[7], student_t::cdf(z, eta));
                            let ls = student_t::log_density(z, eta) - scale.ln();
                            assert!((slots[8] - ls).abs() < 1e-14 * ls.abs().max(1.0));
                        } else {
                            assert!(slots[7].is_nan() && slots[8].is_nan());
                        }
                    }
                    Some(DistWithheld::ShapeUnidentified) => {
                        seen[1] += 1;
                        assert_eq!(slots[..2], [params[j].0.0, params[j].0.1]);
                        assert!(slots[2..].iter().all(|v| v.is_nan()));
                    }
                    Some(_) => {
                        seen[2] += 1;
                        assert!(slots.iter().all(|v| v.is_nan()));
                    }
                }
            }
        }
        assert!(seen.iter().all(|n| *n > 10), "{floc:?}: {seen:?}");
    }
}

/// The stream in pieces, saved and restored through msgpack between them,
/// gives the rows of one piece to the bit, before, at and after the start.
#[test]
fn chunks_and_a_save_between_them_change_nothing() {
    for floc in [None, Some(-0.5)] {
        let rows = stream(400, 4);
        let mut whole = DistFit::new(cfg2(floc)).unwrap();
        let want: Vec<Step> = rows
            .iter()
            .map(|(y, d, w)| whole.step(&[], y, *d, *w))
            .collect();
        for cut in [1, 10, 25, 200, 399] {
            let mut m = DistFit::new(cfg2(floc)).unwrap();
            let mut got: Vec<Step> = rows[..cut]
                .iter()
                .map(|(y, d, w)| m.step(&[], y, *d, *w))
                .collect();
            let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
            let state: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut m = DistFit::restore(&state).unwrap();
            got.extend(rows[cut..].iter().map(|(y, d, w)| m.step(&[], y, *d, *w)));
            for (a, b) in want.iter().zip(&got) {
                assert_eq!(bits(a), bits(b), "cut at {cut}");
            }
            assert_eq!(m, whole);
        }
    }
}

/// A row of weight 0 learns nothing whatever its value, before and after
/// the start: the fit does not move, and every weight ages by the decay.
#[test]
fn a_zero_weight_row_learns_nothing() {
    let rows = stream(300, 5);
    for upto in [0, 5, 300] {
        let mut m = DistFit::new(cfg2(None)).unwrap();
        for (y, d, w) in &rows[..upto] {
            m.step(&[], y, *d, *w);
        }
        let mut a = m.clone();
        a.step(&[], &[Some(1e9), Some(-1e9)], 2.0, 0.0);
        let lam = cfg2(None).decay.factor(2.0);
        for j in 0..2 {
            assert_eq!(a.t[j].theta(), m.t[j].theta());
            assert_eq!(a.t[j].c4.to_bits(), m.t[j].c4.to_bits());
            assert_eq!(a.t[j].w2.to_bits(), (m.t[j].w2 * (lam * lam)).to_bits());
        }
        assert_eq!(a.n_eff(), m.n_eff() * lam);
    }
}

/// A constant target is a variance of 0: withheld, never divided by, and
/// the t never started on it.
#[test]
fn a_constant_target_is_withheld_not_divided_by() {
    let mut m = DistFit::new(cfg(30.0)).unwrap();
    for _ in 0..200 {
        let s = m.step(&[], &[Some(0.25)], 1.0, 1.0);
        assert!(s.pred.iter().all(|v| v.is_nan()));
    }
    assert_eq!(m.withheld(0), Some(DistWithheld::DegenerateVariance));
    assert!(!m.t[0].started());
}

/// A fixed location stays where it is set, and the scale and tail are
/// fitted about it.
#[test]
fn a_fixed_location_is_held() {
    let ys = t_rows(20_000, 8, 5, 0.0, 1.5);
    let mut m = DistFit::new(DistFitCfg {
        floc: Some(0.0),
        ..cfg(4000.0)
    })
    .unwrap();
    for &y in &ys {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    let (loc, scale) = m.params(0);
    assert_eq!(loc, 0.0);
    assert!((scale - 1.5).abs() < 0.08, "{scale}");
    assert!((m.df(0).unwrap() - 5.0).abs() < 1.5, "{:?}", m.df(0));
    assert_eq!(m.withheld(0), None);
}

/// A saved state whose t sums are missing, or present under the normal,
/// or whose tail is out of its bounds, is refused.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = DistFit::new(cfg2(None)).unwrap();
    for (y, d, w) in stream(100, 6) {
        m.step(&[], &y, d, w);
    }
    let mut bad = m.clone();
    bad.t.pop();
    assert!(DistFit::restore(&bad.state()).is_err());
    let mut bad = m.clone();
    bad.cfg.family = DistFamily::Normal;
    assert!(DistFit::restore(&bad.state()).is_err());
    let mut bad = m.clone();
    bad.t[0].fit.as_mut().unwrap().theta[2] = 1.5;
    assert!(DistFit::restore(&bad.state()).is_err());
    assert!(DistFit::restore(&m.state()).is_ok());
}

/// The surrogate's gradient, kept with bounded state, against the exact
/// gradient of the whole weighted likelihood at the fit: the two agree to
/// within a small part of the noise on a long stream.
#[test]
fn the_surrogate_gradient_tracks_the_exact_one() {
    let hl = 2000.0;
    let ys = t_rows(12_000, 31, 4, -1.0, 2.0);
    let mut m = DistFit::new(cfg(hl)).unwrap();
    for &y in &ys {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    let st = m.t[0].fit.as_ref().unwrap();
    let lam = Decay::Halflife(hl).factor(1.0);
    let n = ys.len();
    let rows: Vec<(f64, f64)> = ys
        .iter()
        .enumerate()
        .map(|(i, &y)| (y, lam.powi((n - 1 - i) as i32)))
        .collect();
    let (g, h) = exact_gradient(&rows, &st.theta);
    let gap = [0, 1, 2].map(|k| st.grad[k] - g[k]);
    let step = chol_solve(&full(&h), &gap, [true; 3]).unwrap();
    let dec: f64 = (0..3).map(|k| step[k] * gap[k]).sum();
    let w: f64 = rows.iter().map(|r| r.1).sum();
    let w2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
    // Measured 0.0022 against a noise scale of 0.51.
    assert!(dec < 0.05 * w2 / w, "{dec}");
}
