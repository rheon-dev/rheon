use super::*;

fn cfg() -> DistFitCfg {
    DistFitCfg {
        n_targets: 2,
        family: DistFamily::Normal,
        decay: Decay::Halflife(40.0),
        min_weight: 0.0,
        quantiles: vec![0.01, 0.5, 0.95],
        es_levels: vec![0.05],
        floc: None,
    }
}

/// A seeded stream: normal draws (Box-Muller on an LCG), irregular clock
/// deltas, unequal weights, the second target absent on every fifth row.
fn stream(n: usize, seed: u64) -> Vec<(Vec<Option<f64>>, f64, f64)> {
    let mut s = seed;
    let mut u = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|i| {
            let z = (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos();
            let z2 = (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos();
            let y = vec![Some(3.0 + 2.0 * z), (i % 5 != 4).then_some(-1.0 + 0.5 * z2)];
            let d = 0.5 + u();
            let w = 0.25 + u();
            (y, d, w)
        })
        .collect()
}

fn bits(s: &Step) -> Vec<u64> {
    s.pred.iter().map(|v| v.to_bits()).collect()
}

/// The fit against the definition written out longhand: each learned row's
/// weight times the product of the decay factors since it, and the weighted
/// mean and variance of the rows the target was present on.
#[test]
fn the_fit_is_the_weighted_mean_and_variance_longhand() {
    let c = cfg();
    let rows = stream(400, 7);
    let mut m = DistFit::new(c.clone()).unwrap();
    for (t, (y, d, w)) in rows.iter().enumerate() {
        let step = m.step(&[], y, *d, *w);
        for j in 0..2 {
            // The rows before t, at their decayed weights.
            let mut num = 0.0;
            let mut den = 0.0;
            let mut sq = 0.0;
            let mut lam_since = 1.0;
            for i in (0..t).rev() {
                if let Some(v) = rows[i].0[j] {
                    let om = rows[i].2 * lam_since;
                    den += om;
                    num += om * v;
                    sq += om * v * v;
                }
                lam_since *= c.decay.factor(rows[i].1);
            }
            if den == 0.0 {
                assert!(step.pred[j * 8].is_nan());
                continue;
            }
            let loc = num / den;
            let var = sq / den - loc * loc;
            let (got_loc, got_scale) = (step.pred[j * 8], step.pred[j * 8 + 1]);
            if t >= 2 {
                assert!((got_loc - loc).abs() < 1e-11 * (1.0 + loc.abs()), "row {t}");
                assert!((got_scale * got_scale - var).abs() < 1e-9 * var, "row {t}");
            }
        }
    }
}

/// Every output of a row is a function of the fit before it: the step's
/// report equals what a model that has seen only the earlier rows predicts.
#[test]
fn every_output_is_out_of_sample() {
    let rows = stream(200, 11);
    let mut m = DistFit::new(cfg()).unwrap();
    for (y, d, w) in &rows {
        let before = m.predict_with(&[], y, *d);
        let step = m.step(&[], y, *d, *w);
        assert_eq!(bits(&before), bits(&step));
        assert_eq!(before.n_eff.to_bits(), step.n_eff.to_bits());
    }
}

/// Feeding the stream in pieces, saved and restored between them through
/// msgpack, gives the same rows to the bit as one piece.
#[test]
fn chunks_and_a_save_between_them_change_nothing() {
    let rows = stream(300, 3);
    let mut whole = DistFit::new(cfg()).unwrap();
    let want: Vec<Step> = rows
        .iter()
        .map(|(y, d, w)| whole.step(&[], y, *d, *w))
        .collect();
    for cut in [1, 17, 150, 299] {
        let mut m = DistFit::new(cfg()).unwrap();
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

/// A row of weight 0 learns nothing whatever its value: the fit does not
/// move, only the weights age by the row's decay.
#[test]
fn a_zero_weight_row_learns_nothing() {
    let rows = stream(50, 5);
    for first in [true, false] {
        let mut m = DistFit::new(cfg()).unwrap();
        if !first {
            for (y, d, w) in &rows {
                m.step(&[], y, *d, *w);
            }
        }
        let before: Vec<_> = (0..2).map(|j| m.params(j)).collect();
        let w0 = m.n_eff();
        let s = m.step(&[], &[Some(1e9), Some(-1e9)], 2.0, 0.0);
        assert_eq!(s.n_eff.to_bits(), w0.to_bits());
        for (j, &b) in before.iter().enumerate() {
            let a = m.params(j);
            assert!(a.0.to_bits() == b.0.to_bits() || (a.0.is_nan() && b.0.is_nan()));
            assert_eq!(a.1.to_bits(), b.1.to_bits());
        }
        assert_eq!(m.n_eff(), w0 * cfg().decay.factor(2.0));
        assert!(m.fit.iter().all(|f| !f.var(0).is_nan()));
    }
}

/// A constant target is a variance of exactly 0: withheld with its reason,
/// every slot not a number, and the state free of NaN; a fixed location
/// at the constant is degenerate the same way.
#[test]
fn a_constant_target_is_withheld_not_divided_by() {
    for floc in [None, Some(0.1)] {
        let mut m = DistFit::new(DistFitCfg {
            n_targets: 1,
            floc,
            ..cfg()
        })
        .unwrap();
        assert_eq!(m.withheld(0), Some(DistWithheld::NoWeight));
        for _ in 0..100 {
            let s = m.step(&[], &[Some(0.1)], 1.0, 1.0);
            assert!(s.pred.iter().all(|v| v.is_nan()));
            assert_eq!(m.withheld(0), Some(DistWithheld::DegenerateVariance));
        }
        assert!(m.params(0).1 == 0.0);
        // One different value ends it.
        m.step(&[], &[Some(0.3)], 1.0, 1.0);
        assert_eq!(m.withheld(0), None);
        assert!(
            m.step(&[], &[Some(0.1)], 1.0, 1.0)
                .pred
                .iter()
                .all(|v| v.is_finite())
        );
    }
}

/// Each target's `min_weight` is read against its own weight.
#[test]
fn min_weight_reads_each_targets_own_weight() {
    let mut m = DistFit::new(DistFitCfg {
        decay: Decay::Lam(1.0),
        min_weight: 3.5,
        ..cfg()
    })
    .unwrap();
    for i in 0..4 {
        m.step(
            &[],
            &[Some(i as f64), (i % 2 == 0).then_some(i as f64)],
            1.0,
            1.0,
        );
    }
    assert_eq!(m.withheld(0), None);
    assert_eq!(m.withheld(1), Some(DistWithheld::MinWeight));
    let mut w = Vec::new();
    assert!(m.target_n_eff_into(&mut w));
    assert_eq!(w, vec![4.0, 2.0]);
}

/// The normal's quantiles and shortfalls in closed form, and on a long
/// realistic stream the fit near the truth, the PIT near uniform and the
/// quantiles covering their levels.
#[test]
fn a_long_stream_is_calibrated() {
    let mut m = DistFit::new(DistFitCfg {
        decay: Decay::Halflife(2000.0),
        ..cfg()
    })
    .unwrap();
    let rows = stream(40_000, 13);
    let (mut pit_sum, mut n, mut below) = (0.0, 0.0, [0.0; 3]);
    for (t, (y, d, w)) in rows.iter().enumerate() {
        let s = m.step(&[], y, *d, *w);
        if t > 1000 {
            let v = y[0].unwrap();
            pit_sum += s.pred[6];
            n += 1.0;
            for (k, b) in below.iter_mut().enumerate() {
                *b += f64::from(u8::from(v <= s.pred[2 + k]));
            }
            // The median is the location; the shortfall lies below the
            // quantile at its level.
            assert_eq!(s.pred[3], s.pred[0]);
            let q05 = s.pred[0] + s.pred[1] * crate::special::ndtri(0.05);
            assert!(s.pred[5] < q05);
            // φ(Φ⁻¹(0.05)) / 0.05, as scipy's `norm.pdf(norm.ppf(0.05)) / 0.05` gives it.
            let es = s.pred[0] - 2.062_712_807_507_425_3 * s.pred[1];
            assert!((s.pred[5] - es).abs() < 1e-13 * (1.0 + es.abs()));
        }
    }
    let (loc, scale) = m.params(0);
    assert!((loc - 3.0).abs() < 0.15, "{loc}");
    assert!((scale - 2.0).abs() < 0.1, "{scale}");
    assert!((pit_sum / n - 0.5).abs() < 0.01, "{}", pit_sum / n);
    for (b, p) in below.iter().zip([0.01, 0.5, 0.95]) {
        assert!((b / n - p).abs() < 0.015, "{} against {p}", b / n);
    }
}

/// With a fixed location, the scale is the root of the weighted mean square
/// about it, and the location reported is the fixed one.
#[test]
fn a_fixed_location_fits_the_scale_about_it() {
    let mut m = DistFit::new(DistFitCfg {
        n_targets: 1,
        decay: Decay::Lam(1.0),
        floc: Some(1.0),
        ..cfg()
    })
    .unwrap();
    for (v, w) in [(2.0, 1.0), (0.0, 3.0), (4.0, 2.0)] {
        m.step(&[], &[Some(v)], 1.0, w);
    }
    let (loc, scale) = m.params(0);
    assert_eq!(loc, 1.0);
    let want = ((1.0 + 3.0 + 2.0 * 9.0) / 6.0_f64).sqrt();
    assert!((scale - want).abs() < 1e-15);
}

#[test]
fn bad_settings_are_refused() {
    for c in [
        DistFitCfg {
            n_targets: 0,
            ..cfg()
        },
        DistFitCfg {
            quantiles: vec![0.0],
            ..cfg()
        },
        DistFitCfg {
            es_levels: vec![1.0],
            ..cfg()
        },
        DistFitCfg {
            floc: Some(f64::NAN),
            ..cfg()
        },
        DistFitCfg {
            min_weight: -1.0,
            ..cfg()
        },
        DistFitCfg {
            decay: Decay::Halflife(0.0),
            ..cfg()
        },
    ] {
        assert!(DistFit::new(c).is_err());
    }
}
