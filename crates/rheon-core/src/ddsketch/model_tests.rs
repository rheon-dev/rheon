use super::*;
use crate::ddsketch::Mapping;

fn cfg(n_targets: usize, levels: &[f64]) -> DdSketchModelCfg {
    DdSketchModelCfg {
        n_targets,
        alpha: crate::EW_QUANTILE_ALPHA,
        levels: levels.to_vec(),
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    }
}

fn rows_cfg(n_targets: usize, levels: &[f64], n: usize) -> DdSketchModelCfg {
    DdSketchModelCfg {
        window_rows: Some(n),
        ..cfg(n_targets, levels)
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A bare sketch of `rows` (value, weight), read at `levels`.
fn bare(map: &Mapping, rows: &[(f64, f64)], levels: &[f64]) -> Vec<Option<(f64, f64)>> {
    let mut d = DdSketch::default();
    for &(x, w) in rows {
        d.add(map, Slot::of(map, x), w, false);
    }
    d.reads(levels)
}

/// Each row's levels and tail means are the sketch's over the rows before
/// it, to the bit; a target absent from a row adds nothing to it.
#[test]
fn each_row_reads_the_sketch_of_the_rows_before_it() {
    let levels = [0.05, 0.4, 0.95];
    let mut m = DdSketchModel::new(cfg(2, &levels)).unwrap();
    let map = m.mapping().clone();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut s = 7u64;
    for i in 0..2_000 {
        let (x, z) = (lcg(&mut s) - 0.5, 10.0 * lcg(&mut s));
        let w = 1.0 + (i % 4) as f64;
        let y = [Some(x), if i % 3 == 0 { None } else { Some(z) }];
        let step = m.step(&[], &y, 1.0, w);
        for (j, rows) in [&a, &b].into_iter().enumerate() {
            let n = {
                let (sw, s2) = rows.iter().fold((0.0, 0.0), |(p, q), r: &(f64, f64)| {
                    (p + r.1, q + r.1 * r.1)
                });
                if s2 > 0.0 { sw * sw / s2 } else { 0.0 }
            };
            let reads = bare(&map, rows, &levels);
            for (li, &p) in levels.iter().enumerate() {
                let (got, es) = (step.pred[j * 6 + li], step.pred[j * 6 + 3 + li]);
                match reads[li].filter(|_| level_identified(p, n)) {
                    Some((q, tail)) => {
                        assert_eq!(got.to_bits(), q.to_bits(), "row {i}");
                        assert_eq!(es.to_bits(), tail.to_bits(), "row {i}");
                    }
                    None => assert!(got.is_nan() && es.is_nan(), "row {i}"),
                }
            }
        }
        a.push((x, w));
        if let Some(z) = y[1] {
            b.push((z, w));
        }
    }
}

/// A row window reads, at every row, the sketch built from its last `N`
/// rows alone, to the bit (weights whole numbers, so every subtraction is
/// exact), and `n_eff` is the window's weight.
#[test]
fn a_row_window_reads_the_sketch_of_its_last_rows() {
    let levels = [0.1, 0.4, 0.9];
    let n = 37;
    let mut m = DdSketchModel::new(rows_cfg(1, &levels, n)).unwrap();
    let map = m.mapping().clone();
    let mut rows: Vec<(f64, f64)> = Vec::new();
    let mut s = 11u64;
    for i in 0..1_500 {
        let x = (lcg(&mut s) - 0.4) * 10f64.powi(i % 5 - 2);
        let w = (1 + i % 3) as f64;
        let step = m.step(&[], &[Some(x)], 1.0, w);
        let window = &rows[rows.len().saturating_sub(n)..];
        let want = bare(&map, window, &levels);
        let weight: f64 = window.iter().map(|r| r.1).sum();
        assert_eq!(step.n_eff, weight, "row {i}");
        for (li, r) in want.iter().enumerate() {
            if let Some((q, es)) = r.filter(|_| !step.pred[li].is_nan()) {
                assert_eq!(step.pred[li].to_bits(), q.to_bits(), "row {i}");
                assert_eq!(step.pred[3 + li].to_bits(), es.to_bits(), "row {i}");
            }
        }
        rows.push((x, w));
    }
    assert_eq!(m.window_len(), n);
}

/// A clock window keeps the rows less than `D` old under `Right` and those
/// exactly `D` old too under `Both`: each side of the edge, read at the row.
#[test]
fn a_clock_window_keeps_its_span_and_its_edge() {
    for (closed, kept, median) in [
        (WindowClosed::Right, 2.0, 9.0),
        (WindowClosed::Both, 3.0, 9.0),
    ] {
        let c = DdSketchModelCfg {
            window_clock: Some(3.0),
            closed,
            ..cfg(1, &[0.4])
        };
        let mut m = DdSketchModel::new(c).unwrap();
        for i in 0..10 {
            m.stamp_next(Stamp::from(f64::from(i)));
            m.step(&[], &[Some(f64::from(i + 1))], 1.0, 1.0);
        }
        // At 10 the row at 7 is exactly 3 old: `Right` reads the rows at 8
        // and 9 (values 9 and 10), `Both` the row at 7 (value 8) too.
        m.stamp_next(Stamp::from(10.0));
        let step = m.step(&[], &[Some(0.0)], 1.0, 1.0);
        assert_eq!(step.n_eff, kept, "{closed:?}");
        let map = m.mapping().clone();
        assert_eq!(step.pred[0], map.value_of(median), "{closed:?}");
    }
}

/// A row of weight 0 changes nothing, first or between rows, windowed or
/// not, and reports what `predict` does; under a clock window it reads the
/// window as it would stand at the row.
#[test]
fn a_zero_weight_row_changes_nothing() {
    for c in [cfg(1, &[0.4]), rows_cfg(1, &[0.4], 4)] {
        let mut m = DdSketchModel::new(c).unwrap();
        let before = m.clone();
        let step = m.step(&[], &[Some(5.0)], 1.0, 0.0);
        assert_eq!(m, before);
        assert!(step.pred[0].is_nan() && step.n_eff == 0.0);
        for i in 0..10 {
            m.step(&[], &[Some(f64::from(i))], 1.0, 1.0);
        }
        let before = m.clone();
        let read = m.predict(&[], 1.0);
        let step = m.step(&[], &[Some(1e9)], 1.0, 0.0);
        assert_eq!(m, before);
        assert_eq!(bits(&step.pred), bits(&read.pred));
    }
    let c = DdSketchModelCfg {
        window_clock: Some(2.0),
        ..cfg(1, &[0.4])
    };
    let mut m = DdSketchModel::new(c).unwrap();
    for i in 0..5 {
        m.stamp_next(Stamp::from(f64::from(i)));
        m.step(&[], &[Some(f64::from(10 * i))], 1.0, 1.0);
    }
    // At 6 the rows at 3 and 4 have left a window of 2 (closed right):
    // the row reads an empty window, and the state keeps them.
    let before = m.clone();
    m.stamp_next(Stamp::from(6.0));
    let step = m.step(&[], &[Some(1.0)], 2.0, 0.0);
    assert!(step.pred[0].is_nan(), "{:?}", step.pred);
    assert_eq!(m.window_len(), before.window_len());
}

/// Saved at every depth of a row window, from empty to full and past,
/// and restored, the model continues to the bit.
#[test]
fn a_restored_model_continues_to_the_bit_at_every_depth() {
    for cut in 0..12 {
        for c in [cfg(2, &[0.1, 0.4, 0.9]), rows_cfg(2, &[0.1, 0.4, 0.9], 6)] {
            let mut m = DdSketchModel::new(c).unwrap();
            let mut s = 11u64;
            let row = |s: &mut u64| [Some(lcg(s) - 0.5), Some(lcg(s) * lcg(s))];
            for _ in 0..cut {
                let y = row(&mut s);
                m.step(&[], &y, 1.0, 1.0 + lcg(&mut s));
            }
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            let state: State = rmp_serde::from_slice(&bytes).unwrap();
            let mut r = DdSketchModel::restore(&state).unwrap();
            for _ in 0..40 {
                let y = row(&mut s);
                let w = 1.0 + lcg(&mut s);
                let (a, b) = (m.step(&[], &y, 1.0, w), r.step(&[], &y, 1.0, w));
                assert_eq!(bits(&a.pred), bits(&b.pred), "cut {cut}");
                assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
            }
            assert_eq!(m, r);
        }
    }
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = DdSketchModel::new(rows_cfg(2, &[0.4], 5)).unwrap();
    m.step(&[], &[Some(1.0), Some(2.0)], 1.0, 1.0);
    let mut short = m.clone();
    short.sketches.pop();
    let mut weights = m.clone();
    weights.target_w2.push(1.0);
    let mut negative = m.clone();
    negative.target_w[0] = -1.0;
    let mut ring = m.clone();
    ring.ring.as_mut().unwrap().codes.pop_back();
    let mut unwindowed = m.clone();
    unwindowed.ring = None;
    let mut map = m.clone();
    map.map = Mapping::new(0.02).unwrap();
    for bad in [short, weights, negative, ring, unwindowed, map] {
        let err = DdSketchModel::restore(&bad.state())
            .unwrap_err()
            .to_string();
        assert!(err.contains("ddsketch") && err.contains("damaged"), "{err}");
    }
    assert!(DdSketchModel::restore(&m.state()).is_ok());
}

/// The settings refused, each on both sides of its edge.
#[test]
fn the_settings_are_checked() {
    let ok = |c: DdSketchModelCfg| DdSketchModel::new(c).map(|_| ());
    assert!(ok(cfg(1, &[0.4])).is_ok());
    assert!(
        ok(cfg(0, &[0.4]))
            .unwrap_err()
            .contains("at least one target")
    );
    assert!(ok(cfg(1, &[])).unwrap_err().contains("at least one level"));
    for p in [0.0, 1.0, -0.1, f64::NAN] {
        assert!(
            ok(cfg(1, &[p]))
                .unwrap_err()
                .contains("above 0 and below 1")
        );
    }
    assert!(ok(cfg(1, &[1e-9, 1.0 - 1e-9])).is_ok());
    assert!(ok(cfg(1, &[0.4, 0.4])).unwrap_err().contains("twice"));
    assert!(ok(cfg(1, &[0.05, 0.5])).unwrap_err().contains("lists 0.5"));
    assert!(ok(cfg(1, &[0.4999, 0.5001])).is_ok());
    for (alpha, good) in [(9.9e-7, false), (1e-6, true), (0.5, true), (1.0, false)] {
        let c = DdSketchModelCfg {
            alpha,
            ..cfg(1, &[0.4])
        };
        assert_eq!(ok(c).is_ok(), good, "{alpha}");
    }
    assert!(
        ok(rows_cfg(1, &[0.4], 0))
            .unwrap_err()
            .contains("at least 1")
    );
    assert!(ok(rows_cfg(1, &[0.4], 1)).is_ok());
    for (d, good) in [
        (0.0, false),
        (1e-9, true),
        (f64::INFINITY, false),
        (f64::NAN, false),
    ] {
        let c = DdSketchModelCfg {
            window_clock: Some(d),
            ..cfg(1, &[0.4])
        };
        assert_eq!(ok(c).is_ok(), good, "{d}");
    }
    let both = DdSketchModelCfg {
        window_clock: Some(3.0),
        ..rows_cfg(1, &[0.4], 3)
    };
    assert!(ok(both).unwrap_err().contains("not both"));
    let edge = DdSketchModelCfg {
        closed: WindowClosed::Both,
        ..rows_cfg(1, &[0.4], 3)
    };
    assert!(ok(edge).unwrap_err().contains("closed"));
    // The cap on what the settings fix: a row window at the edge, and a
    // byte under it.
    let c = rows_cfg(3, &[0.4], 1_000);
    let most = c.fixed_bytes();
    let at = |cap: u64| {
        ok(DdSketchModelCfg {
            max_state_bytes: Some(cap),
            ..rows_cfg(3, &[0.4], 1_000)
        })
    };
    assert!(at(most).is_ok());
    assert!(at(most - 1).unwrap_err().contains("max_state_bytes"));
    assert!(at(0).is_err());
}

fn decayed(half_life: f64) -> DdSketchModelCfg {
    DdSketchModelCfg {
        decay: Decay::Halflife(half_life),
        ..cfg(1, &[0.1, 0.4, 0.9])
    }
}

/// The decayed form's weights keep the decaying-sum recursion every model
/// keeps (hard rule 8): `n_eff` is the weight before the row's update and
/// its own decay, `W <- f W + w`, to the bit, rows of weight 0 handed
/// directly decaying it and learning nothing.
#[test]
fn the_decayed_weight_is_the_decaying_sum() {
    let mut m = DdSketchModel::new(decayed(20.0)).unwrap();
    let f = Decay::Halflife(20.0).factor(1.5);
    let mut want = 0.0f64;
    let mut s = 4u64;
    for i in 0..500 {
        let w = if i % 7 == 0 { 0.0 } else { 0.5 + lcg(&mut s) };
        let step = m.step(&[], &[Some(lcg(&mut s))], 1.5, w);
        assert_eq!(step.n_eff.to_bits(), want.to_bits(), "row {i}");
        want = f * want + w;
    }
    assert_eq!(m.n_eff().to_bits(), want.to_bits());
}

/// A bucket's decayed weight against its exact decayed sum, worked in
/// double-double arithmetic from the same factor: within the stated bound,
/// `2 k u / (1 - 2 k u)` for `k` rows, and the reads pick the buckets the
/// exact weights pick.
#[test]
fn a_decayed_bucket_is_its_exact_sum_within_the_rounding_bound() {
    let h = 50.0;
    let mut m = DdSketchModel::new(decayed(h)).unwrap();
    let f = Decay::Halflife(h).factor(1.0);
    let map = m.mapping().clone();
    let mut s = 8u64;
    let n = 3_000usize;
    let mut rows = Vec::new();
    for _ in 0..n {
        let (x, w) = (0.5 + 3.0 * lcg(&mut s), 1.0 + lcg(&mut s));
        m.step(&[], &[Some(x)], 1.0, w);
        rows.push((x, w));
    }
    // Double-double: (hi, lo) with the error-free product and sum.
    let two_prod = |a: f64, b: f64| {
        let p = a * b;
        (p, a.mul_add(b, -p))
    };
    let two_sum = |a: f64, b: f64| {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    };
    let mut exact: std::collections::BTreeMap<i32, (f64, f64)> = Default::default();
    for &(x, w) in &rows {
        for v in exact.values_mut() {
            let (p, e) = two_prod(v.0, f);
            let (hi, lo) = two_sum(p, e + v.1 * f);
            *v = (hi, lo);
        }
        let Slot::Pos(k) = Slot::of(&map, x) else {
            unreachable!()
        };
        let v = exact.entry(k).or_insert((0.0, 0.0));
        let (hi, lo) = two_sum(v.0, w);
        *v = (hi, lo + v.1);
    }
    let u = f64::EPSILON / 2.0;
    let bound = 2.0 * n as f64 * u / (1.0 - 2.0 * n as f64 * u);
    let mut worst = 0.0f64;
    for (k, w) in m.sketch(0).positive().buckets() {
        let (hi, lo) = exact[&k];
        let rel = ((w - hi) - lo).abs() / hi;
        worst = worst.max(rel);
        assert!(rel <= bound, "bucket {k}: {rel:e} past {bound:e}");
    }
    assert!(worst > 0.0, "a decayed sum rounds somewhere");
}

/// A decay with a window is refused, both ways round, and a half-life of 0
/// or a lam past 1 by the decay's own check; `inf` and lam 1 are no decay
/// and sit beside a window.
#[test]
fn a_decay_and_a_window_are_refused_together() {
    let ok = |c: DdSketchModelCfg| DdSketchModel::new(c).map(|_| ());
    assert!(ok(decayed(10.0)).is_ok());
    let both = DdSketchModelCfg {
        window_rows: Some(5),
        ..decayed(10.0)
    };
    assert!(ok(both).unwrap_err().contains("not both"));
    let clock = DdSketchModelCfg {
        window_clock: Some(5.0),
        decay: Decay::Lam(0.9),
        ..cfg(1, &[0.4])
    };
    assert!(ok(clock).unwrap_err().contains("not both"));
    for decay in [Decay::Halflife(f64::INFINITY), Decay::Lam(1.0)] {
        let c = DdSketchModelCfg {
            window_rows: Some(5),
            decay,
            ..cfg(1, &[0.4])
        };
        assert!(ok(c).is_ok(), "{decay:?}");
    }
    assert!(
        ok(decayed(0.0))
            .unwrap_err()
            .contains("half_life must be above 0")
    );
    let lam = DdSketchModelCfg {
        decay: Decay::Lam(1.5),
        ..cfg(1, &[0.4])
    };
    assert!(ok(lam).unwrap_err().contains("lam must be"));
}

/// Under a short half-life a bucket left behind underflows to 0 and is
/// removed: the state holds the buckets of recent rows alone, and a saved
/// state of it loads.
#[test]
fn a_bucket_that_decays_to_nothing_is_removed() {
    let mut m = DdSketchModel::new(decayed(0.5)).unwrap();
    m.step(&[], &[Some(1e6)], 1.0, 1.0);
    for _ in 0..3_000 {
        m.step(&[], &[Some(2.0)], 1.0, 1.0);
    }
    assert_eq!(m.sketch(0).buckets_len(), 1);
    assert!(DdSketchModel::restore(&m.state()).is_ok());
}

/// The buckets the data spans are checked as the rows go in: the first row
/// past half of the cap is said once, and a row that would pass the cap is
/// not learned, nor any after it; under the cap nothing is said.
#[test]
fn the_buckets_are_held_to_the_cap_as_the_rows_go_in() {
    let c = DdSketchModelCfg {
        max_state_bytes: Some(20_000),
        ..cfg(1, &[0.4])
    };
    let mut m = DdSketchModel::new(c).unwrap();
    // A value a factor `gamma` above the last each row: one bucket more a
    // row, 8 bytes each.
    let mut x = 1.0;
    let mut warned_at = None;
    let mut over_at = None;
    for i in 0..5_000 {
        x *= m.mapping().gamma();
        let before = m.clone();
        m.step(&[], &[Some(x)], 1.0, 1.0);
        if warned_at.is_none()
            && let Some((bytes, cap)) = m.take_state_warning()
        {
            assert!(2 * bytes > cap && cap == 20_000);
            warned_at = Some(i);
        }
        if m.state_over_cap().is_some() {
            over_at = Some(i);
            assert_eq!(
                m.sketch(0),
                before.sketch(0),
                "the row past the cap is not learned"
            );
            break;
        }
        assert!(m.bytes() <= 20_000);
    }
    let (w, o) = (warned_at.unwrap(), over_at.unwrap());
    assert!(w < o, "{w} {o}");
    let held = m.clone();
    m.step(&[], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(m.sketch(0), held.sketch(0), "nor any row after it");
    assert!(m.take_state_warning().is_none());
    // A stream of one scale stays far under the cap and says nothing.
    let mut quiet = DdSketchModel::new(cfg(1, &[0.4])).unwrap();
    let mut s = 4u64;
    for _ in 0..50_000 {
        quiet.step(&[], &[Some(1.0 + lcg(&mut s))], 1.0, 1.0);
    }
    assert!(quiet.take_state_warning().is_none() && quiet.state_over_cap().is_none());
}

/// Scaling every weight changes nothing but the weight: the thresholds
/// `p W` and Kish's size both scale away (a power of two rounds nothing).
#[test]
fn scaling_every_weight_changes_nothing_but_the_weight() {
    let levels = [0.05, 0.4, 0.9];
    let run = |scale: f64| {
        let mut m = DdSketchModel::new(rows_cfg(1, &levels, 50)).unwrap();
        let mut s = 3u64;
        (0..500)
            .map(|_| {
                let (v, w) = (lcg(&mut s) - 0.2, 1.0 + lcg(&mut s));
                let step = m.step(&[], &[Some(v)], 1.0, w * scale);
                (step.pred, step.n_eff)
            })
            .collect::<Vec<_>>()
    };
    let (one, quarter) = (run(1.0), run(0.25));
    for ((a, wa), (b, wb)) in one.iter().zip(&quarter) {
        assert_eq!(bits(a), bits(b));
        assert_eq!((wa * 0.25).to_bits(), wb.to_bits());
    }
}

/// After a cap breach (`over.is_some()`), row-clock `stamp_next_on_rows` reads
/// advance the hypothetical window key by 1 for positive-weight rows (`weight > 0.0`)
/// and by 0 for zero-weight rows.
#[test]
fn over_cap_row_clock_read_ahead_distinguishes_positive_and_zero_weight() {
    let mut c = cfg(1, &[0.25]);
    c.window_clock = Some(4.0);
    c.closed = WindowClosed::Right;
    c.max_state_bytes = Some(20_000);
    let mut m = DdSketchModel::new(c).unwrap();
    for i in 1..=4 {
        m.stamp_next_on_rows(Stamp::from(i as f64));
        m.step(&[], &[Some(i as f64)], 1.0, 1.0);
    }
    assert_eq!(m.n_eff(), 4.0);
    // Mark over-cap directly so the post-cap `read_ahead` branch runs on row 5.
    m.over = Some(20_001);

    // Positive weight (`weight > 0.0`): key = 4 + 1 = 5.0 -> row 1 (`1.0`) leaves (`5 - 1 >= 4`) -> n_eff = 3.0.
    m.stamp_next_on_rows(Stamp::from(5.0));
    let pos = m.step(&[], &[Some(5.0)], 1.0, 1.0);
    assert_eq!(pos.n_eff, 3.0);

    // Zero weight (`weight == 0.0`): key = 4 + 0 = 4.0 -> row 1 stays (`4 - 1 < 4`) -> n_eff = 4.0.
    m.stamp_next_on_rows(Stamp::from(5.0));
    let zero = m.step(&[], &[Some(5.0)], 1.0, 0.0);
    assert_eq!(zero.n_eff, 4.0);
}
