//! `TDigest` against exact ranks, its own size rule and its own definition
//! (PLAN 544, part 2). The agreement with the `tdigest` package is in
//! `tests/test_second_opinion.py`.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn open(state: &mut u64) -> f64 {
    loop {
        let u = lcg(state);
        if u > 0.0 {
            return u;
        }
    }
}

fn gauss(s: &mut u64) -> f64 {
    let (u, v) = (open(s), lcg(s));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// The study's four streams of `n` values: normal, lognormal(0, 1),
/// Student-t(3) and classical Pareto of shape 2.5.
fn streams(seed: u64, n: usize) -> [Vec<f64>; 4] {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 544;
    let normal: Vec<f64> = (0..n).map(|_| gauss(&mut s)).collect();
    let lognormal = normal.iter().map(|z| z.exp()).collect();
    let t3 = (0..n)
        .map(|_| {
            let z = gauss(&mut s);
            let chi: f64 = (0..3).map(|_| gauss(&mut s).powi(2)).sum();
            z / (chi / 3.0).sqrt()
        })
        .collect();
    let pareto = (0..n).map(|_| open(&mut s).powf(-1.0 / 2.5)).collect();
    [normal, lognormal, t3, pareto]
}

const NAMES: [&str; 4] = ["normal", "lognormal", "t3", "pareto"];
const LEVELS: [f64; 5] = [0.01, 0.05, 0.5, 0.95, 0.99];

fn digest_of(xs: &[f64], delta: f64) -> TDigest {
    let mut d = TDigest::new(delta).unwrap();
    for &x in xs {
        d.update(x, 1.0);
    }
    d
}

/// `|F(q_hat) - p|`, `F` the stream's empirical CDF (values at most `q_hat`).
fn rank_error(sorted: &[f64], q_hat: f64, p: f64) -> f64 {
    let at_most = sorted.partition_point(|&v| v <= q_hat);
    (at_most as f64 / sorted.len() as f64 - p).abs()
}

fn largest_rank_error(delta: f64, seeds: std::ops::Range<u64>, n: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for seed in seeds {
        for (name, xs) in NAMES.iter().zip(streams(seed, n)) {
            let d = digest_of(&xs, delta);
            let mut sorted = xs.clone();
            sorted.sort_by(f64::total_cmp);
            for (p, q) in LEVELS.iter().zip(d.quantiles(&LEVELS)) {
                let e = rank_error(&sorted, q.unwrap(), *p);
                if e > worst {
                    eprintln!("delta {delta} {name} seed {seed} p {p}: {e:.3e}");
                }
                worst = worst.max(e);
            }
        }
    }
    worst
}

/// The study's tolerance at N = 100,000: rank error at most 3.8e-3 at
/// compression 100 and 1.8e-3 at 200, on four distributions and five levels.
#[test]
fn rank_error_is_within_the_studys_tolerance() {
    let e100 = largest_rank_error(100.0, 0..2, 100_000);
    let e200 = largest_rank_error(200.0, 0..2, 100_000);
    eprintln!("largest rank error: {e100:.3e} at 100, {e200:.3e} at 200");
    assert!(e100 <= 3.8e-3, "{e100}");
    assert!(e200 <= 1.8e-3, "{e200}");
}

/// The other side of the tolerance: a digest a fifth as fine breaks the
/// bound at compression 100, so the test can see a coarser digest.
#[test]
fn a_coarser_digest_breaks_the_tolerance() {
    let e = largest_rank_error(20.0, 0..1, 100_000);
    assert!(e > 3.8e-3, "{e}");
}

/// The limit computed without a math library is `k^-1(k(q) + 1)` written
/// with `asin` and `sin`, within a few ulps, and 1 past the clamp; the sine
/// and cosine of `2 pi / delta` from the series are the library's within an
/// ulp.
#[test]
fn the_limit_is_the_scale_function_definition() {
    for delta in [10.0, 100.0, 200.0, 1000.0, 100_000.0, 1e7] {
        let d = TDigest::new(delta).unwrap();
        let b = std::f64::consts::TAU / delta;
        assert!(
            (d.sin_b - b.sin()).abs() <= f64::EPSILON * b.sin(),
            "{delta}"
        );
        assert!((d.cos_b - b.cos()).abs() <= f64::EPSILON, "{delta}");
        let k = |q: f64| delta / std::f64::consts::TAU * (2.0 * q - 1.0).asin();
        let k_inv = |k: f64| {
            if k >= delta / 4.0 {
                1.0
            } else {
                ((std::f64::consts::TAU * k / delta).sin() + 1.0) / 2.0
            }
        };
        for i in 0..=1000 {
            let q = i as f64 / 1000.0;
            let want = k_inv(k(q) + 1.0);
            let got = d.limit(q);
            assert!(
                (got - want).abs() <= 1e-14,
                "delta {delta} q {q}: {got} vs {want}"
            );
        }
    }
}

/// Every centroid spans at most 1 in `k_1` (the size rule), and at most
/// `delta + 1` remain; the centroid count and bytes are the study's.
#[test]
fn every_centroid_keeps_the_size_rule_and_the_count_is_bounded() {
    for delta in [100.0, 200.0] {
        let mut most = 0;
        for (name, xs) in NAMES.iter().zip(streams(3, 100_000)) {
            let d = digest_of(&xs, delta);
            let c = d.settled();
            let total: f64 = c.iter().map(|x| x.1).sum();
            let k =
                |q: f64| delta / std::f64::consts::TAU * (2.0 * q - 1.0).clamp(-1.0, 1.0).asin();
            let mut before = 0.0;
            for &(_, w) in &c {
                let span = k((before + w) / total) - k(before / total);
                // A single row spans what it spans; a merged centroid at most 1.
                assert!(
                    w == 1.0 || span <= 1.0 + 1e-9,
                    "{name} delta {delta}: {span}"
                );
                before += w;
            }
            assert!(c.len() as f64 <= delta + 1.0, "{name}: {}", c.len());
            assert!(c.len() as f64 >= delta / 2.0 - 1.0, "{name}: {}", c.len());
            assert!(d.has_shape());
            assert!(d.bytes() as u64 <= TDigest::max_bytes(delta));
            most = most.max(c.len());
        }
        eprintln!(
            "delta {delta}: at most {most} centroids, {} bytes",
            16 * most
        );
    }
}

/// A read leaves the state as it was, so where a stream is cut into chunks
/// and read between them changes nothing: the state is the rows' alone.
#[test]
fn reads_between_chunks_change_nothing() {
    let [xs, ..] = streams(5, 5_000);
    let whole = digest_of(&xs, 100.0);
    for chunk in [1, 7, 799, 800, 801, 5_000] {
        let mut d = TDigest::new(100.0).unwrap();
        for part in xs.chunks(chunk) {
            for &x in part {
                d.update(x, 1.0);
            }
            let _ = d.quantiles(&LEVELS);
        }
        assert_eq!(
            rmp_serde::to_vec(&d).unwrap(),
            rmp_serde::to_vec(&whole).unwrap()
        );
        let (a, b) = (d.quantiles(&LEVELS), whole.quantiles(&LEVELS));
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.unwrap().to_bits(), y.unwrap().to_bits());
        }
    }
}

/// Saved mid-buffer and after a merge, a digest loads, continues to the bit
/// and saves the same bytes.
#[test]
fn a_saved_state_continues_to_the_bit() {
    let [xs, _, ys, _] = streams(7, 4_000);
    for cut in [0, 1, 799, 800, 1_234, 4_000] {
        let mut live = TDigest::new(100.0).unwrap();
        for &x in &xs[..cut] {
            live.update(x, 1.0);
        }
        let bytes = rmp_serde::to_vec(&live).unwrap();
        let mut back: TDigest = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.has_shape());
        for &y in &ys {
            live.update(y, 1.0);
            back.update(y, 1.0);
        }
        assert_eq!(
            rmp_serde::to_vec(&back).unwrap(),
            rmp_serde::to_vec(&live).unwrap()
        );
        for (a, b) in back.quantiles(&LEVELS).iter().zip(live.quantiles(&LEVELS)) {
            assert_eq!(a.unwrap().to_bits(), b.unwrap().to_bits());
        }
    }
}

/// A row of weight 0, first or later, or a weight or value that is not a
/// positive finite number, changes nothing; the smallest positive weight is
/// seen.
#[test]
fn a_zero_weight_row_changes_nothing() {
    let [xs, ..] = streams(9, 3_000);
    let plain = digest_of(&xs, 100.0);
    let mut d = TDigest::new(100.0).unwrap();
    d.update(1e9, 0.0);
    assert_eq!(d.quantile(0.5), None);
    for &x in &xs {
        d.update(x * 1e3, 0.0);
        d.update(x, -1.0);
        d.update(x, f64::NAN);
        d.update(x, f64::INFINITY);
        d.update(f64::NAN, 1.0);
        d.update(x, 1.0);
    }
    assert_eq!(
        rmp_serde::to_vec(&d).unwrap(),
        rmp_serde::to_vec(&plain).unwrap()
    );
    let mut tiny = plain.clone();
    tiny.update(1e9, f64::MIN_POSITIVE);
    assert_ne!(tiny, plain);
    assert_eq!(tiny.hi, 1e9);
}

/// A row of weight `w` reads as `w` rows of it, within the digest's error:
/// the weighted rank error against the exact weighted CDF.
#[test]
fn a_weight_counts_as_mass() {
    let [xs, ..] = streams(13, 50_000);
    let mut s = 99u64;
    let ws: Vec<f64> = xs
        .iter()
        .map(|_| 1.0 + (lcg(&mut s) * 4.0).floor())
        .collect();
    let mut d = TDigest::new(100.0).unwrap();
    for (&x, &w) in xs.iter().zip(&ws) {
        d.update(x, w);
    }
    let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ws.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = ws.iter().sum();
    for (p, q) in LEVELS.iter().zip(d.quantiles(&LEVELS)) {
        let q = q.unwrap();
        let below: f64 = pairs.iter().take_while(|v| v.0 <= q).map(|v| v.1).sum();
        assert!(
            (below / total - p).abs() <= 3.8e-3,
            "p {p}: {}",
            below / total
        );
    }
}

/// Reads before any row, with one row, and at the extremes.
#[test]
fn small_digests_read_their_rows() {
    let mut d = TDigest::new(100.0).unwrap();
    assert_eq!(d.quantile(0.5), None);
    d.update(3.0, 1.0);
    assert_eq!(d.quantile(0.01), Some(3.0));
    d.update(-1.0, 1.0);
    assert_eq!(d.quantile(0.0), Some(-1.0));
    assert_eq!(d.quantile(1.0), Some(3.0));
    assert_eq!(d.quantile(0.5), Some(1.0));
    d.reset();
    assert_eq!(d, TDigest::new(100.0).unwrap());
}

/// The compression's floor, each side of it, no ceiling, and the state
/// cap checked from both sides.
#[test]
fn compression_and_cap_are_checked_on_both_sides() {
    for good in [MIN_COMPRESSION, 10.5, 100.0, 1e6] {
        assert!(TDigest::new(good).is_ok(), "{good}");
    }
    for bad in [9.99, 1.0, 0.0, f64::NAN, f64::INFINITY, -100.0] {
        let err = TDigest::new(bad).unwrap_err();
        assert!(err.contains("at least 10"), "{err}");
    }
    let most = TDigest::max_bytes(100.0);
    assert!(TDigest::check_cap(100.0, most).is_ok());
    assert!(TDigest::check_cap(100.0, most - 1).is_err());
    assert!(TDigest::check_cap(200.0, crate::clock_span::DEFAULT_STATE_BYTES).is_ok());
    assert!(TDigest::check_cap(100_000.0, crate::clock_span::STATE_BYTES_CEILING).is_ok());
}

/// A restored state that a digest could not hold is refused.
#[test]
fn a_misshapen_state_is_refused() {
    let [xs, ..] = streams(2, 2_000);
    let d = digest_of(&xs, 100.0);
    assert!(d.has_shape());
    let mut bad = d.clone();
    bad.centroids.swap(0, 1);
    assert!(!bad.has_shape());
    let mut bad = d.clone();
    bad.sin_b = 0.5;
    assert!(!bad.has_shape());
    let mut bad = d.clone();
    bad.centroids[0].1 = 0.0;
    assert!(!bad.has_shape());
    let mut bad = d;
    bad.hi = bad.lo;
    assert!(!bad.has_shape());
}

/// PLAN 544, part 3, the coordinator's option (C): a plain digest (no span)
/// builds; a decayed or windowed one is refused, naming the field and the
/// windowed DDSketch as the form that forgets.
#[test]
fn a_span_is_refused_naming_the_windowed_ddsketch() {
    assert_eq!(TDigest::check_no_span(None), Ok(()));
    assert!(TDigest::new(100.0).is_ok());
    for field in ["half_life", "window_size", "span"] {
        let err = TDigest::check_no_span(Some(field)).unwrap_err();
        assert!(err.contains(field), "{err}");
        assert!(err.contains("never forgets"), "{err}");
        assert!(err.contains("windowed DDSketch"), "{err}");
        assert!(err.contains("ewm_quantile with a window_size"), "{err}");
    }
}
