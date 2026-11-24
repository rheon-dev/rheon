use super::*;
use crate::Decay;
use crate::dist_fit::{DistFamily, DistFit, DistFitCfg, DistWithheld};
use crate::model::{OnlineModel, State, Step};

fn cfg(half_life: f64) -> DistFitCfg {
    DistFitCfg {
        n_targets: 1,
        family: DistFamily::SkewT,
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
    /// Hansen's skewed t of integer `nu` degrees of freedom (0: the normal's
    /// tail) and skew `lam`, mean 0 and variance 1: the t of variance 1,
    /// its magnitude scaled by `1 − λ` on the left, with probability
    /// `(1 − λ)/2`, and by `1 + λ` on the right, less its mean `a`, over its
    /// standard deviation `b`.
    fn skewt(&mut self, nu: u32, lam: f64) -> f64 {
        let mut t = self.normal();
        if nu > 0 {
            let chi: f64 = (0..nu).map(|_| self.normal().powi(2)).sum();
            t /= (chi / f64::from(nu - 2)).sqrt();
        }
        let x = if self.u() < 0.5 * (1.0 - lam) {
            -(1.0 - lam) * t.abs()
        } else {
            (1.0 + lam) * t.abs()
        };
        let eta = if nu == 0 { 0.0 } else { 1.0 / f64::from(nu) };
        let sh = Shape::new(eta, lam);
        (x - sh.a) / sh.b
    }
}

fn rows(n: usize, seed: u64, nu: u32, lam: f64, loc: f64, scale: f64) -> Vec<f64> {
    let mut r = Lcg(seed);
    (0..n).map(|_| loc + scale * r.skewt(nu, lam)).collect()
}

fn bits(s: &Step) -> Vec<u64> {
    s.pred.iter().map(|v| v.to_bits()).collect()
}

fn theta(m: &DistFit) -> [f64; 4] {
    m.skewt[0].theta().unwrap()
}

/// The exact gradient of the whole weighted log likelihood at `θ`, its
/// information `Σ ω I`, and `(Σω, Σω²)`, over `ys` at unit weights and the
/// decay `lam` a row.
fn exact(ys: &[f64], lam: f64, th: &[f64; 4]) -> ([f64; 4], [f64; 10], f64, f64) {
    let n = ys.len();
    let mut g = [0.0; 4];
    let mut h = [0.0; 10];
    let (mut w, mut w2) = (0.0, 0.0);
    for (i, &y) in ys.iter().enumerate() {
        let om = lam.powi((n - 1 - i) as i32);
        let e = eval(y, th).unwrap();
        for (gk, ek) in g.iter_mut().zip(&e.g) {
            *gk += om * ek;
        }
        for (hk, ek) in h.iter_mut().zip(&e.exp) {
            *hk += om * ek;
        }
        w += om;
        w2 += om * om;
    }
    (g, h, w, w2)
}

/// The exact decrement at the fit and the noise scale `Σω²/Σω`.
fn exact_decrement(ys: &[f64], lam: f64, th: &[f64; 4], free: [bool; 4]) -> (f64, f64) {
    let (g, h, w, w2) = exact(ys, lam, th);
    let step = chol_solve(&full(&h), &g, free).unwrap();
    (dot(&step, &g), w2 / w)
}

/// A long skewed-t stream: the fit near the truth, its shape identified,
/// and the fit a stationary point of the whole weighted likelihood to
/// within a small part of the noise.
#[test]
fn a_long_skewed_stream_is_fitted_near_the_truth_and_the_maximum() {
    let hl = 3000.0;
    for (seed, nu, lam) in [(21, 5, -0.3), (22, 8, 0.5)] {
        let ys = rows(30_000, seed, nu, lam, 2.0, 0.5);
        let mut m = DistFit::new(cfg(hl)).unwrap();
        for &y in &ys {
            m.step(&[], &[Some(y)], 1.0, 1.0);
        }
        assert_eq!(m.withheld(0), None);
        let (loc, scale) = m.params(0);
        let df = m.df(0).unwrap();
        let l = m.lambda(0).unwrap();
        assert!((loc - 2.0).abs() < 0.03, "{loc}");
        assert!((scale - 0.5).abs() < 0.03, "{scale}");
        assert!((df - f64::from(nu)).abs() < 0.3 * f64::from(nu), "{df}");
        assert!((l - lam).abs() < 0.06, "{l}");
        let lam_d = Decay::Halflife(hl).factor(1.0);
        let (dec, noise) = exact_decrement(&ys, lam_d, &theta(&m), [true; 4]);
        // Measured 0.0059 and 0.018 against a noise bound of 4.7: held at a
        // twentieth of the bound.
        assert!(dec < 0.05 * noise * CHI2_95[3], "{seed}: {dec}");
    }
}

/// Normal data: the shape is seldom told from the normal's, and while it
/// is not, `loc` and `scale` report and nothing that reads the shape does.
#[test]
fn a_normal_stream_leaves_the_shape_unidentified() {
    let mut identified = 0;
    for seed in 0..20 {
        let ys = rows(3000, 100 + seed, 0, 0.0, 1.0, 2.0);
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
                let th = theta(&m);
                let lam = Decay::Halflife(1000.0).factor(1.0);
                let (d, n) = exact_decrement(&ys, lam, &th, [true; 4]);
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
/// the scale and the skew to a standard error of 1/4: about 35 rows of
/// equal weight at a normal tail.
#[test]
fn the_start_waits_for_a_variance_and_its_precision() {
    let mut m = DistFit::new(cfg(f64::INFINITY)).unwrap();
    assert_eq!(m.withheld(0), Some(DistWithheld::NoWeight));
    m.step(&[], &[Some(1.0)], 1.0, 1.0);
    m.step(&[], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.withheld(0), Some(DistWithheld::DegenerateVariance));
    let ys = rows(200, 3, 6, 0.2, 1.0, 1.0);
    let mut n = 2;
    while !m.skewt[0].started() {
        let s = m.step(&[], &[Some(ys[n])], 1.0, 1.0);
        n += 1;
        if !m.skewt[0].started() {
            assert_eq!(m.withheld(0), Some(DistWithheld::NotConverged), "{n}");
            assert!(s.pred.iter().all(|v| v.is_nan()));
        }
    }
    assert!((25..60).contains(&n), "{n}");
    let (loc, scale) = (m.fit[0].mean(0), m.fit[0].var(0).sqrt());
    let (l, s) = m.params(0);
    assert!((l - loc).abs() < 1e-15 && (s - scale).abs() < 1e-15 * scale);
}

/// The moment start against its definition on a skewed, heavy-tailed
/// stream: the weighted mean and standard deviation, the tail from the
/// weighted kurtosis as the t's, and no skew, written out longhand.
#[test]
fn the_moment_start_is_its_definition() {
    for floc in [None, Some(0.3)] {
        let ys = rows(60, 9, 7, 0.4, 0.5, 1.0);
        let mut t = SkewTFit::new();
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
        // the stepped scale is the kernel's, the standard deviation times
        // √(1 − 2η)
        let sd = libm::exp(th[1]) / (1.0 - 2.0 * th[2]).sqrt();
        assert!((sd - mom(2).sqrt()).abs() < 1e-12, "{floc:?}");
        assert!((th[2] - eta).abs() < 1e-12, "{floc:?}: {} {eta}", th[2]);
        assert_eq!(th[3], 0.0);
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

/// Two targets of skewed t draws, irregular clock deltas, unequal weights,
/// a few rows of weight 0, the second target absent on every fifth row.
fn stream(n: usize, seed: u64) -> Vec<(Vec<Option<f64>>, f64, f64)> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|i| {
            let y = vec![
                Some(1.0 + 0.5 * r.skewt(4, -0.6)),
                (i % 5 != 4).then(|| -2.0 + 3.0 * r.skewt(3, 0.5)),
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
            let params: Vec<_> = (0..2)
                .map(|j| (m.params(j), m.df(j), m.lambda(j)))
                .collect();
            let shapes: Vec<_> = (0..2)
                .map(|j| m.skewt[j].theta().map(|t| (t[2], t[3])))
                .collect();
            let s = m.step(&[], &y, d, w);
            assert_eq!(bits(&before), bits(&s));
            for (j, slots) in s.pred.chunks(c.slots()).enumerate() {
                match reasons[j] {
                    None => {
                        seen[0] += 1;
                        let ((loc, scale), df, l) =
                            (params[j].0, params[j].1.unwrap(), params[j].2.unwrap());
                        let (eta, lam) = shapes[j].unwrap();
                        assert_eq!(slots[..4], [loc, scale, df, l]);
                        assert_eq!((df, l), (1.0 / eta, lam));
                        if let Some(f) = floc {
                            assert_eq!(loc, f);
                        }
                        let sh = Shape::new(eta, lam);
                        for (k, p) in c.quantiles.iter().enumerate() {
                            let want = loc + scale * sh.ppf(*p);
                            assert_eq!(slots[4 + k].to_bits(), want.to_bits());
                        }
                        let es = loc + scale * sh.lower_es(0.05);
                        assert_eq!(slots[7].to_bits(), es.to_bits());
                        if let Some(v) = y[j] {
                            let z = (v - loc) / scale;
                            assert_eq!(slots[8], sh.cdf(z));
                            let ls = sh.log_density(z) - scale.ln();
                            assert!((slots[9] - ls).abs() < 1e-14 * ls.abs().max(1.0));
                        } else {
                            assert!(slots[8].is_nan() && slots[9].is_nan());
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
            assert_eq!(a.skewt[j].theta(), m.skewt[j].theta());
            assert_eq!(a.skewt[j].c3.to_bits(), m.skewt[j].c3.to_bits());
            assert_eq!(a.skewt[j].c4.to_bits(), m.skewt[j].c4.to_bits());
            assert_eq!(
                a.skewt[j].w2.to_bits(),
                (m.skewt[j].w2 * (lam * lam)).to_bits()
            );
        }
        assert_eq!(a.n_eff(), m.n_eff() * lam);
    }
}

/// A constant target is a variance of 0: withheld, never divided by, and
/// the fit never started on it.
#[test]
fn a_constant_target_is_withheld_not_divided_by() {
    let mut m = DistFit::new(cfg(30.0)).unwrap();
    for _ in 0..200 {
        let s = m.step(&[], &[Some(0.25)], 1.0, 1.0);
        assert!(s.pred.iter().all(|v| v.is_nan()));
    }
    assert_eq!(m.withheld(0), Some(DistWithheld::DegenerateVariance));
    assert!(!m.skewt[0].started());
}

/// A fixed location stays where it is set, and the scale and shape are
/// fitted about it: the mean of a skewed stream is its `floc`.
#[test]
fn a_fixed_location_is_held() {
    let ys = rows(20_000, 8, 6, -0.4, 0.0, 1.5);
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
    assert!((m.df(0).unwrap() - 6.0).abs() < 2.0, "{:?}", m.df(0));
    assert!(
        (m.lambda(0).unwrap() + 0.4).abs() < 0.08,
        "{:?}",
        m.lambda(0)
    );
    assert_eq!(m.withheld(0), None);
}

/// A saved state whose sums are missing, present under another family, or
/// whose shape is out of its bounds, is refused.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = DistFit::new(cfg2(None)).unwrap();
    for (y, d, w) in stream(100, 6) {
        m.step(&[], &y, d, w);
    }
    let mut bad = m.clone();
    bad.skewt.pop();
    assert!(DistFit::restore(&bad.state()).is_err());
    for family in [DistFamily::Normal, DistFamily::StudentT] {
        let mut bad = m.clone();
        bad.cfg.family = family;
        assert!(DistFit::restore(&bad.state()).is_err());
    }
    for (k, v) in [(2, -0.1), (2, 0.6), (3, 1.0), (3, -1.2)] {
        let mut bad = m.clone();
        bad.skewt[0].fit.as_mut().unwrap().theta[k] = v;
        assert!(DistFit::restore(&bad.state()).is_err(), "{k} {v}");
    }
    assert!(DistFit::restore(&m.state()).is_ok());
}

/// The surrogate's gradient, kept with bounded state, against the exact
/// gradient of the whole weighted likelihood at the fit: the two agree to
/// within a small part of the noise on a long stream.
#[test]
fn the_surrogate_gradient_tracks_the_exact_one() {
    let hl = 2000.0;
    let ys = rows(12_000, 31, 5, 0.3, -1.0, 2.0);
    let mut m = DistFit::new(cfg(hl)).unwrap();
    for &y in &ys {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    let st = m.skewt[0].fit.as_ref().unwrap();
    let lam = Decay::Halflife(hl).factor(1.0);
    let (g, h, w, w2) = exact(&ys, lam, &st.theta);
    let gap = [0, 1, 2, 3].map(|k| st.grad[k] - g[k]);
    let step = chol_solve(&full(&h), &gap, [true; 4]).unwrap();
    let dec = dot(&step, &gap);
    // Measured 0.0022 against a noise scale of 0.51.
    assert!(dec < 0.05 * w2 / w, "{dec}");
}

/// One row far out moves the shape no further than the whole likelihood
/// says: the fit stays near the maximum after it (the exact decrement
/// measured 0.61 and 2.1 against a bound of 9.5).
#[test]
fn an_outlier_moves_the_shape_no_further_than_the_likelihood() {
    let mut ys = rows(2000, 5, 0, 0.0, 0.0, 1.0);
    let mut m = DistFit::new(cfg(f64::INFINITY)).unwrap();
    for &y in &ys {
        m.step(&[], &[Some(y)], 1.0, 1.0);
    }
    for out in [8.0, -30.0] {
        m.step(&[], &[Some(out)], 1.0, 1.0);
        ys.push(out);
        let th = theta(&m);
        let free = free_at(&th, &exact(&ys, 1.0, &th).0, false);
        let (dec, noise) = exact_decrement(&ys, 1.0, &th, free);
        assert!(th[2] < 0.3, "{out}: {th:?}");
        assert!(dec < noise * CHI2_95[3], "{out}: {dec}");
    }
}

/// Near `ν = 2` the standard deviation and the tail are nearly the same
/// direction of the likelihood (the information's `(ln σ, η)` block has a
/// determinant three parts in ten thousand of its diagonal's product at
/// `ν = 2.05`), so a row at the centre once moved `ln σ` by −20 in that
/// direction for almost nothing; the fit steps the kernel's scale `ln s`
/// instead, in which the block is the t's. A heavy, skewed stream of 4
/// degrees of freedom keeps its scale and converges.
#[test]
fn a_heavy_tail_does_not_collapse_the_scale() {
    for seed in 0..8 {
        let ys = rows(800, 300 + seed, 4, -0.5, 3.0, 1.0);
        let mut m = DistFit::new(cfg(300.0)).unwrap();
        let mut not_converged = 0;
        for &y in &ys {
            m.step(&[], &[Some(y)], 1.0, 1.0);
            if m.withheld(0) == Some(DistWithheld::NotConverged) {
                not_converged += 1;
            }
        }
        let (loc, scale) = m.params(0);
        assert!(
            (loc - 3.0).abs() < 0.2 && (scale - 1.0).abs() < 0.25,
            "{seed}: {loc} {scale}"
        );
        // measured 29 to 66 rows, the start's included
        assert!(not_converged < 100, "{seed}: {not_converged}");
    }
}
