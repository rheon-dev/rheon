//! `TDigestModel`: the levels read before each row, the identifiability
//! rule on both sides, weights, zero-weight rows, saves and refusals
//! (PLAN 544, part 5).

use super::*;

fn cfg(n_targets: usize, levels: &[f64]) -> TDigestModelCfg {
    TDigestModelCfg {
        n_targets,
        compression: TDIGEST_COMPRESSION,
        levels: levels.to_vec(),
        max_state_bytes: None,
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Each row's levels are the digest's reads over the rows before it: the
/// model against a bare `TDigest` fed the same rows, to the bit.
#[test]
fn each_row_reads_the_digest_of_the_rows_before_it() {
    let levels = [0.05, 0.25, 0.95];
    let mut m = TDigestModel::new(cfg(2, &levels)).unwrap();
    let mut bare = [TDigest::new(100.0).unwrap(), TDigest::new(100.0).unwrap()];
    let mut s = 7u64;
    for i in 0..3_000 {
        let (a, b) = (lcg(&mut s) - 0.5, 10.0 * lcg(&mut s));
        let y = [Some(a), if i % 3 == 0 { None } else { Some(b) }];
        let step = m.step(&[], &y, 1.0, 1.0);
        for (j, d) in bare.iter().enumerate() {
            let n = m.effective_rows(j) - y[j].map_or(0.0, |_| 1.0);
            let reads = d.quantiles_and_tails(&levels);
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
        for (j, d) in bare.iter_mut().enumerate() {
            if let Some(v) = y[j] {
                d.update(v, 1.0);
            }
        }
    }
}

/// The rule on both sides of each edge: 0.5 from the first row, 0.05 and
/// 0.95 from the 19th, 0.01 from the 99th, never at 0 rows.
#[test]
fn a_level_is_identified_from_its_plotting_position_on() {
    assert!(!level_identified(0.5, 0.0));
    assert!(level_identified(0.5, 1.0));
    for p in [0.05, 0.95] {
        assert!(!level_identified(p, 18.0), "{p}");
        assert!(level_identified(p, 19.0), "{p}");
    }
    for p in [0.01, 0.99] {
        assert!(!level_identified(p, 97.9), "{p}");
        assert!(level_identified(p, 99.0), "{p}");
    }
    // In the model: the 0.05 level is withheld on the 19th row (18 before
    // it) and reported on the 20th, 0.6 from the third (`0.4 (n + 1) >= 1`).
    let mut m = TDigestModel::new(cfg(1, &[0.05, 0.6])).unwrap();
    let mut rows = Vec::new();
    for i in 0..25 {
        rows.push(m.step(&[], &[Some(i as f64)], 1.0, 1.0).pred);
    }
    assert!(rows[0].iter().all(|v| v.is_nan()));
    assert!(rows[1].iter().all(|v| v.is_nan()));
    assert!(rows[2][1].is_finite() && rows[2][0].is_nan());
    assert!(rows[18][0].is_nan());
    assert!(rows[19][0].is_finite());
    assert!(!m.beyond_sample());
    let fresh = TDigestModel::new(cfg(1, &[0.6])).unwrap();
    assert!(fresh.beyond_sample());
}

/// Scaling every weight changes nothing but the weights: the levels and
/// which are withheld read Kish's size, not the weight's units.
#[test]
fn scaling_every_weight_changes_nothing_but_the_weight() {
    let levels = [0.05, 0.4, 0.9];
    let run = |scale: f64| {
        let mut m = TDigestModel::new(cfg(1, &levels)).unwrap();
        let mut s = 3u64;
        let mut out = Vec::new();
        for _ in 0..500 {
            let (v, w) = (lcg(&mut s), 1.0 + lcg(&mut s));
            let step = m.step(&[], &[Some(v)], 1.0, w * scale);
            out.push((step.pred, step.n_eff));
        }
        out
    };
    let (one, tiny, half) = (run(1.0), run(0.001), run(0.5));
    for (((a, wa), (b, wb)), (h, _)) in one.iter().zip(&tiny).zip(&half) {
        for ((x, y), z) in a.iter().zip(b).zip(h) {
            // A scale of 0.001 rounds a merge's `(c' - c) w'/w` its own
            // way, within a few ulps; a power of two rounds nothing.
            assert_eq!(x.is_nan(), y.is_nan());
            assert!(
                x.is_nan() || (x - y).abs() <= 1e-12 * x.abs().max(1.0),
                "{x} {y}"
            );
            assert_eq!(x.to_bits(), z.to_bits());
        }
        assert!((wa * 0.001 - wb).abs() <= 1e-12 * wa);
    }
}

/// A row of weight 0 changes nothing, first or between rows, and reports
/// what `predict` does.
#[test]
fn a_zero_weight_row_changes_nothing() {
    let mut m = TDigestModel::new(cfg(1, &[0.4])).unwrap();
    let before = m.clone();
    let step = m.step(&[], &[Some(5.0)], 1.0, 0.0);
    assert_eq!(m, before);
    assert!(step.pred[0].is_nan() && step.n_eff == 0.0);
    for i in 0..10 {
        m.step(&[], &[Some(i as f64)], 1.0, 1.0);
    }
    let before = m.clone();
    let read = m.predict(&[], 1.0);
    let step = m.step(&[], &[Some(1e9)], 1.0, 0.0);
    assert_eq!(m, before);
    assert_eq!(step.pred[0].to_bits(), read.pred[0].to_bits());
}

/// Saved mid-buffer and restored, the model continues to the bit.
#[test]
fn a_restored_model_continues_to_the_bit() {
    let mut m = TDigestModel::new(cfg(2, &[0.1, 0.4, 0.9])).unwrap();
    let mut s = 11u64;
    let row = |s: &mut u64| [Some(lcg(s)), Some(lcg(s) * lcg(s))];
    for _ in 0..1_234 {
        let y = row(&mut s);
        m.step(&[], &y, 1.0, 1.0);
    }
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let state: State = rmp_serde::from_slice(&bytes).unwrap();
    let mut r = TDigestModel::restore(&state).unwrap();
    for _ in 0..2_000 {
        let y = row(&mut s);
        let (a, b) = (m.step(&[], &y, 1.0, 1.0), r.step(&[], &y, 1.0, 1.0));
        for (x, z) in a.pred.iter().zip(&b.pred) {
            assert_eq!(x.to_bits(), z.to_bits());
        }
    }
    assert_eq!(m, r);
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = TDigestModel::new(cfg(2, &[0.4])).unwrap();
    m.step(&[], &[Some(1.0), Some(2.0)], 1.0, 1.0);
    let mut short = m.clone();
    short.digests.pop();
    let mut weights = m.clone();
    weights.target_w2.push(1.0);
    let mut negative = m.clone();
    negative.target_w[0] = -1.0;
    let mut other = m.clone();
    other.digests[0] = TDigest::new(200.0).unwrap();
    for bad in [short, weights, negative, other] {
        let err = TDigestModel::restore(&bad.state()).unwrap_err().to_string();
        assert!(err.contains("tdigest") && err.contains("damaged"), "{err}");
    }
    assert!(TDigestModel::restore(&m.state()).is_ok());
}

/// The settings refused, each on both sides of its edge.
#[test]
fn the_settings_are_checked() {
    let ok = |c: TDigestModelCfg| TDigestModel::new(c).map(|_| ());
    assert!(ok(cfg(1, &[0.4])).is_ok());
    assert!(
        ok(cfg(0, &[0.4]))
            .unwrap_err()
            .contains("at least one target")
    );
    assert!(ok(cfg(1, &[])).unwrap_err().contains("at least one level"));
    for p in [0.0, 1.0, -0.1, f64::NAN] {
        let err = ok(cfg(1, &[p])).unwrap_err();
        assert!(err.contains("above 0 and below 1"), "{err}");
    }
    assert!(ok(cfg(1, &[1e-9, 1.0 - 1e-9])).is_ok());
    assert!(ok(cfg(1, &[0.4, 0.4])).unwrap_err().contains("twice"));
    // 0.5 has no tail of its own: refused by name, its neighbours taken.
    assert!(ok(cfg(1, &[0.05, 0.5])).unwrap_err().contains("0.5"));
    assert!(ok(cfg(1, &[0.4999, 0.5001])).is_ok());
    for (c, good) in [
        (9.99, false),
        (10.0, true),
        (12.5, true),
        (1000.0, true),
        (f64::INFINITY, false),
        (f64::NAN, false),
    ] {
        let mut x = cfg(1, &[0.4]);
        x.compression = c;
        assert_eq!(ok(x).is_ok(), good, "{c}");
    }
    // The cap: the fullest state at the edge, and a byte under it.
    let c = cfg(3, &[0.4]);
    let most = c.max_bytes();
    let at = |cap: u64| {
        let mut x = cfg(3, &[0.4]);
        x.max_state_bytes = Some(cap);
        ok(x)
    };
    assert!(at(most).is_ok());
    assert!(at(most - 1).unwrap_err().contains("max_state_bytes"));
    assert!(at(0).is_err());
}

/// The state's bytes never pass what the settings bound, at any length.
#[test]
fn the_state_stays_within_its_bound() {
    let c = cfg(2, &[0.01, 0.4, 0.99]);
    let most = c.max_bytes();
    let mut m = TDigestModel::new(c).unwrap();
    let mut s = 5u64;
    for i in 0..50_000 {
        m.step(&[], &[Some(lcg(&mut s)), Some(-lcg(&mut s))], 1.0, 1.0);
        if i % 997 == 0 {
            assert!(m.bytes() <= most, "{} {most}", m.bytes());
        }
    }
    assert!(m.bytes() <= most);
}

/// Each tail mean is the curve's mean over its tail: the lower below 0.5,
/// the upper above; it lies on its side of the level's read, and
/// against the exact tail mean of the rows it is within the curve's own
/// error (a smooth stream of 20,000 rows).
#[test]
fn each_tail_mean_lies_past_its_level_and_near_the_rows() {
    let levels = [0.01, 0.05, 0.4, 0.6, 0.95, 0.99];
    let mut m = TDigestModel::new(cfg(1, &levels)).unwrap();
    let mut s = 21u64;
    let mut rows = Vec::new();
    for _ in 0..20_000 {
        let v = lcg(&mut s) + lcg(&mut s) + lcg(&mut s);
        m.step(&[], &[Some(v)], 1.0, 1.0);
        rows.push(v);
    }
    rows.sort_by(f64::total_cmp);
    let step = m.predict(&[], 1.0);
    for (li, &p) in levels.iter().enumerate() {
        let (q, es) = (step.pred[li], step.pred[6 + li]);
        let k = (p * rows.len() as f64).round() as usize;
        let exact = if p < 0.5 {
            assert!(es <= q * (1.0 + 1e-12), "{p}: {es} {q}");
            rows[..k].iter().sum::<f64>() / k as f64
        } else {
            assert!(es >= q * (1.0 - 1e-12), "{p}: {es} {q}");
            rows[k..].iter().sum::<f64>() / (rows.len() - k) as f64
        };
        assert!(
            (es - exact).abs() <= 5e-3 * exact.abs(),
            "{p}: {es} vs {exact}"
        );
    }
}

/// A two-row digest's tails, worked by hand: the curve runs from the
/// minimum at rank 0 through each row's value at its centre to the maximum
/// at rank 2, and 0.4 and 0.6 are levels two rows identify.
#[test]
fn a_two_row_tail_mean_is_its_curve_by_hand() {
    let mut m = TDigestModel::new(cfg(1, &[0.4, 0.6])).unwrap();
    m.step(&[], &[Some(0.0)], 1.0, 1.0);
    m.step(&[], &[Some(4.0)], 1.0, 1.0);
    let step = m.predict(&[], 1.0);
    // The knots are (0, 0), (0.5, 0), (1.5, 4), (2, 4). Lower 0.4 is ranks
    // [0, 0.8]: 0 to rank 0.5, then up to 1.2, area 0.18, mean 0.225; upper
    // 0.6 is [1.2, 2], from 2.8 to 4 then 4, area 3.02, mean 3.775.
    let want = [1.2, 2.8, 0.225, 3.775];
    for (got, want) in step.pred.iter().zip(want) {
        assert!((got - want).abs() <= 1e-14, "{got} vs {want}");
    }
}
