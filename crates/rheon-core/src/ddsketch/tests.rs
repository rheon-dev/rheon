use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// A value in (0, 1), never 0.
fn open(s: &mut u64) -> f64 {
    loop {
        let u = lcg(s);
        if u > 0.0 {
            return u;
        }
    }
}

/// `x`'s distance to `y` in ulps.
fn ulps(x: f64, y: f64) -> u64 {
    let (a, b) = (x.to_bits() as i64, y.to_bits() as i64);
    (a - b).unsigned_abs()
}

/// The plain-arithmetic `log2`, `ln(1 + g)` and `2^y` are the math
/// library's within 4 ulps over every scale a double takes, subnormals
/// included.
#[test]
fn the_series_are_the_math_librarys_within_four_ulps() {
    let mut s = 3u64;
    for _ in 0..200_000 {
        let e = (lcg(&mut s) * 2_000.0 - 1_000.0).round() as i32;
        let x = (1.0 + lcg(&mut s)) * 2f64.powi(e);
        let (got, want) = (log2(x), x.log2());
        assert!(
            ulps(got, want) <= 4 || (got - want).abs() <= 4.0 * f64::EPSILON,
            "log2({x:e}): {got} vs {want}"
        );
        let y = lcg(&mut s) * 2_000.0 - 1_000.0;
        assert!(
            ulps(exp2(y), y.exp2()) <= 4,
            "exp2({y}): {} vs {}",
            exp2(y),
            y.exp2()
        );
    }
    for x in [5e-324, 1e-310, f64::MIN_POSITIVE, 1.0, 2.0, f64::MAX] {
        assert!((log2(x) - x.log2()).abs() <= 4.0 * f64::EPSILON * x.log2().abs().max(1.0));
    }
    for g in [1e-12, 2e-6, 0.0157, 0.3, 0.4, 0.41, 3.0, 1e6] {
        assert!(ulps(ln_1p(g), g.ln_1p()) <= 4, "ln_1p({g})");
    }
}

/// `alpha` on both sides of each edge, and the default's mapping: `gamma`
/// is `e^(1/64)`, 64 buckets to a factor of `e`.
#[test]
fn alpha_is_checked_on_both_sides() {
    for good in [
        DDSKETCH_MIN_ALPHA,
        0.01,
        crate::EW_QUANTILE_ALPHA,
        0.5,
        0.999_999,
    ] {
        assert!(Mapping::new(good).is_ok(), "{good}");
    }
    for bad in [0.0, 9.9e-7, 1.0, 1.5, -0.01, f64::NAN, f64::INFINITY] {
        let err = Mapping::new(bad).unwrap_err();
        assert!(err.contains("alpha must be at least"), "{err}");
    }
    let m = Mapping::new(crate::EW_QUANTILE_ALPHA).unwrap();
    assert!((m.gamma() - (1.0f64 / 64.0).exp()).abs() <= 4.0 * f64::EPSILON);
    assert!((m.multiplier - 64.0 * std::f64::consts::LN_2).abs() <= 1e-12);
}

/// Every value's representative is within `alpha` of it (the paper's
/// guarantee), at the default, a coarse and a fine `alpha`, from the
/// smallest magnitude a bucket takes to the largest; a key's
/// representative maps back to the key.
#[test]
fn every_value_is_within_alpha_of_its_representative() {
    for alpha in [crate::EW_QUANTILE_ALPHA, 0.05, 1e-4] {
        let m = Mapping::new(alpha).unwrap();
        let mut s = 7u64;
        for _ in 0..100_000 {
            let e = (lcg(&mut s) * 2_090.0 - 1_060.0).round() as i32;
            let x = (1.0 + lcg(&mut s)) * 2f64.powi(e);
            if x <= m.min_possible() || !x.is_finite() || x > f64::MAX / m.gamma() {
                continue;
            }
            let k = m.key(x);
            let r = m.value(k);
            assert!(
                (r - x).abs() <= alpha * x * (1.0 + 1e-12),
                "{alpha} {x:e}: {r:e}"
            );
            assert_eq!(m.key(r), k, "{alpha} {x:e}");
        }
    }
}

/// The keys run in value order, and a bucket's edges are `gamma^(k-1)`
/// (out) and `gamma^k` (in), each side of the edge.
#[test]
fn a_bucket_holds_its_upper_edge_and_not_its_lower() {
    let m = Mapping::new(0.01).unwrap();
    for k in [-30_000, -300, -1, 0, 1, 2, 77, 30_000] {
        let edge = 2f64.powf(f64::from(k) / m.multiplier);
        // Just inside the bucket and just past its edge, `1e-9` away: the
        // key's own rounding is `|k|` ulps of 1, under `3e-12` here, so
        // neither side's rounding can decide it.
        let inside = edge * (1.0 - 1e-9);
        let outside = edge * (1.0 + 1e-9);
        assert_eq!(m.key(inside), k, "{k}");
        assert_eq!(m.key(outside), k + 1, "{k}");
    }
}

/// Where a value goes: a magnitude at or under `MIN_POSITIVE gamma` to the
/// zero count, both zeros included; a negative value by its magnitude.
#[test]
fn small_magnitudes_count_as_zero_and_negatives_by_magnitude() {
    let m = Mapping::new(crate::EW_QUANTILE_ALPHA).unwrap();
    for x in [
        0.0,
        -0.0,
        5e-324,
        -1e-310,
        f64::MIN_POSITIVE,
        m.min_possible(),
    ] {
        assert_eq!(Slot::of(&m, x), Slot::Zero, "{x:e}");
    }
    let past = f64::from_bits(m.min_possible().to_bits() + 1);
    assert!(matches!(Slot::of(&m, past), Slot::Pos(_)));
    assert!(matches!(Slot::of(&m, -past), Slot::Neg(_)));
    assert_eq!(Slot::of(&m, -3.5), Slot::Neg(m.key(3.5)));
}

/// A slot's 32-bit code names it again, at the widest keys any `alpha`
/// gives, and the absent code names none.
#[test]
fn a_slots_code_names_it_again() {
    let m = Mapping::new(DDSKETCH_MIN_ALPHA).unwrap();
    let widest = m.key(f64::MAX / m.gamma());
    let narrowest = m.key(f64::from_bits(m.min_possible().to_bits() + 1));
    assert!(
        widest < 1 << 29 && narrowest > -(1 << 29),
        "{narrowest} {widest}"
    );
    for k in [narrowest, -1, 0, 1, widest] {
        for slot in [Slot::Pos(k), Slot::Neg(k)] {
            assert_eq!(Slot::decode(slot.code()), Some(slot));
        }
    }
    assert_eq!(Slot::decode(Slot::Zero.code()), Some(Slot::Zero));
    assert_eq!(Slot::decode(Slot::ABSENT), None);
}

/// The exact reads the sketch's are held to: numpy's `inverted_cdf` over
/// the weighted rows (the first value whose cumulative weight reaches
/// `p W`), and the partial-mass tail mean.
fn exact(rows: &[(f64, f64)], p: f64) -> (f64, f64) {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = sorted.iter().map(|r| r.1).sum();
    let walk = |it: &mut dyn Iterator<Item = (f64, f64)>, m: f64| {
        let (mut before, mut sum) = (0.0, 0.0);
        for (x, w) in it {
            if before + w >= m {
                return (x, (sum + (m - before) * x) / m);
            }
            before += w;
            sum += w * x;
        }
        unreachable!()
    };
    let (q, lower) = walk(&mut sorted.iter().copied(), p * total);
    if p < 0.5 {
        (q, lower)
    } else {
        (
            q,
            walk(&mut sorted.iter().rev().copied(), (1.0 - p) * total).1,
        )
    }
}

/// Each read is the exact one's bucket, and each tail mean within `alpha`
/// times the tail's mean magnitude of the exact tail mean (the paper's
/// pairing argument), on streams of both signs, with weights of 1 and of
/// many sizes.
#[test]
fn reads_are_within_alpha_of_the_exact_ones() {
    let m = Mapping::new(crate::EW_QUANTILE_ALPHA).unwrap();
    let levels = [0.01, 0.05, 0.25, 0.4, 0.6, 0.75, 0.95, 0.99];
    for (seed, weighted) in [(1u64, false), (2, true), (3, false), (4, true)] {
        let mut s = seed;
        let mut d = DdSketch::default();
        let mut rows = Vec::new();
        for _ in 0..20_000 {
            let u = open(&mut s);
            let x = match seed {
                1 | 2 => (u / (1.0 - u)).ln(),   // logistic, both signs
                _ => (1.0 - u).powf(-1.0 / 2.5), // Pareto(2.5)
            };
            let w = if weighted {
                0.1 + 3.0 * lcg(&mut s)
            } else {
                1.0
            };
            d.add(&m, Slot::of(&m, x), w, false);
            rows.push((x, w));
        }
        let reads = d.reads(&levels);
        for (&p, r) in levels.iter().zip(reads) {
            let (q, es) = r.unwrap();
            let (eq, ees) = exact(&rows, p);
            assert_eq!(q.to_bits(), m.value_of(eq).to_bits(), "{seed} {p}");
            let mut tail: Vec<f64> = rows.iter().map(|r| r.0).collect();
            tail.sort_by(f64::total_cmp);
            let n = (tail.len() as f64 * p.min(1.0 - p)) as usize;
            let part = if p < 0.5 {
                &tail[..n]
            } else {
                &tail[tail.len() - n..]
            };
            let mag = part.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
            assert!(
                (es - ees).abs() <= crate::EW_QUANTILE_ALPHA * mag * 1.05,
                "{seed} {p}: {es} vs {ees} ({mag})"
            );
            if p < 0.5 {
                assert!(es <= q, "{seed} {p}");
            } else {
                assert!(es >= q, "{seed} {p}");
            }
        }
    }
}

/// Rows taken out again in any order leave the sketch exactly as it was
/// built without them, and all taken out, empty.
#[test]
fn rows_taken_out_leave_the_sketch_of_the_rest() {
    let m = Mapping::new(0.02).unwrap();
    let mut s = 9u64;
    let rows: Vec<(f64, f64)> = (0..3_000)
        .map(|i| {
            let x = (lcg(&mut s) - 0.3) * 10f64.powi((i % 9) - 4);
            (if i % 50 == 0 { 0.0 } else { x }, (1 + i % 3) as f64)
        })
        .collect();
    let mut all = DdSketch::default();
    for &(x, w) in &rows {
        all.add(&m, Slot::of(&m, x), w, true);
    }
    let mut later = DdSketch::default();
    for &(x, w) in &rows[1_000..] {
        later.add(&m, Slot::of(&m, x), w, true);
    }
    for &(x, w) in &rows[..1_000] {
        all.remove(Slot::of(&m, x), w);
    }
    assert_eq!(all.reads(&[0.1, 0.4, 0.9]), later.reads(&[0.1, 0.4, 0.9]));
    for &(x, w) in &rows[1_000..] {
        all.remove(Slot::of(&m, x), w);
    }
    assert_eq!(all, DdSketch::default());
    assert!(all.reads(&[0.5])[0].is_none());
}

/// One value, read at every level and both tails, is itself; a zero count
/// alone reads 0.
#[test]
fn one_value_reads_its_representative() {
    let m = Mapping::new(crate::EW_QUANTILE_ALPHA).unwrap();
    let mut d = DdSketch::default();
    d.add(&m, Slot::of(&m, -2.5), 1.0, false);
    let r = m.value(m.key(2.5));
    for read in d.reads(&[0.01, 0.5, 0.99]) {
        assert_eq!(read, Some((-r, -r)));
    }
    let mut z = DdSketch::default();
    z.add(&m, Slot::Zero, 3.0, false);
    assert_eq!(z.reads(&[0.3, 0.7]), vec![Some((0.0, 0.0)); 2]);
}

/// A restored sketch that `add` and `remove` could not have made is
/// refused.
#[test]
fn a_misshapen_sketch_is_refused() {
    let m = Mapping::new(0.01).unwrap();
    let mut d = DdSketch::default();
    for x in [1.0, 5.0, -2.0, 0.0] {
        d.add(&m, Slot::of(&m, x), 1.0, true);
    }
    assert!(d.has_shape(true));
    assert!(!d.has_shape(false));
    let mut bad = d.clone();
    bad.pos.n.pop();
    assert!(!bad.has_shape(true));
    let mut bad = d.clone();
    bad.pos.n[0] = 0;
    assert!(!bad.has_shape(true));
    let mut bad = d.clone();
    bad.pos.keys.reverse();
    bad.pos.w.reverse();
    bad.pos.n.reverse();
    assert!(!bad.has_shape(true) || bad.pos.keys.len() < 2);
    let mut bad = d.clone();
    bad.zero_w = -1.0;
    assert!(!bad.has_shape(true));
    let mut bad = d;
    bad.neg.w[0] = f64::NAN;
    assert!(!bad.has_shape(true));
    assert!(
        !Mapping {
            alpha: 0.01,
            gamma: 1.5,
            multiplier: m.multiplier,
            min_possible: m.min_possible
        }
        .is_own()
    );
    assert!(m.is_own());
}

/// A store keeps its live buckets alone: two values 80 decades apart are
/// two buckets, not the range between them, and a bucket emptied by a
/// window is gone.
#[test]
fn a_store_holds_its_live_buckets_alone() {
    let m = Mapping::new(crate::EW_QUANTILE_ALPHA).unwrap();
    let mut d = DdSketch::default();
    for x in [1.0, 5.5e-80, 1.0, -3.0, -3.0e200] {
        d.add(&m, Slot::of(&m, x), 1.0, true);
    }
    assert_eq!(d.buckets_len(), 4);
    assert_eq!(d.len_with(Slot::of(&m, 1.0)), 4);
    assert_eq!(d.len_with(Slot::of(&m, 7.0)), 5);
    d.remove(Slot::of(&m, 5.5e-80), 1.0);
    assert_eq!(d.buckets_len(), 3);
    assert!(d.has_shape(true));
}
