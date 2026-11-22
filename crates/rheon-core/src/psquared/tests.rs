//! `PSquared` against the recursion written out longhand from Jain and
//! Chlamtac (1985), 1-based as the paper writes it, and against its own
//! definition before the sixth value (PLAN 544, part 1). The comparison with
//! river's `stats.Quantile`, row for row and to the bit, is in
//! `tests/test_river.py`.

use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// A uniform in (0, 1), never 0, for the transforms below.
fn open(state: &mut u64) -> f64 {
    loop {
        let u = lcg(state);
        if u > 0.0 {
            return u;
        }
    }
}

/// Seeded streams of `n` values: a normal (Box-Muller), a lognormal, a
/// classical Pareto of shape 2.5, values with many ties, a rising trend and
/// a falling one.
fn streams(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 544;
    let normal: Vec<f64> = (0..n)
        .map(|_| {
            let (u, v) = (open(&mut s), lcg(&mut s));
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        })
        .collect();
    let lognormal = normal.iter().map(|z| z.exp()).collect();
    let pareto = (0..n).map(|_| open(&mut s).powf(-1.0 / 2.5)).collect();
    let ties = (0..n).map(|_| (lcg(&mut s) * 7.0).floor() - 3.0).collect();
    let rising = (0..n).map(|i| i as f64 + lcg(&mut s)).collect();
    let falling = (0..n).map(|i| -(i as f64) * 0.5 + lcg(&mut s)).collect();
    vec![normal, lognormal, pareto, ties, rising, falling]
}

/// The paper's P², one level, 1-based (index 0 unused), positions as
/// integers; how many times each of the two predictions moved a marker.
struct Longhand {
    p: f64,
    seen: Vec<f64>,
    q: [f64; 6],
    n: [i64; 6],
    nd: [f64; 6],
    dn: [f64; 6],
    parabolic: usize,
    linear: usize,
}

impl Longhand {
    fn new(p: f64) -> Self {
        Self {
            p,
            seen: Vec::new(),
            q: [0.0; 6],
            n: [0; 6],
            nd: [0.0, 1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn: [0.0, 0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            parabolic: 0,
            linear: 0,
        }
    }

    fn add(&mut self, x: f64) {
        if self.seen.len() < 5 {
            self.seen.push(x);
            return;
        }
        if self.n[1] == 0 {
            // The sixth value: the first five, sorted, are the heights.
            let mut first = self.seen.clone();
            first.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for i in 1..=5 {
                self.q[i] = first[i - 1];
                self.n[i] = i as i64;
            }
        }
        // B1: the cell k, with q_k <= x < q_{k+1}, the extremes moved.
        let k;
        if x < self.q[1] {
            self.q[1] = x;
            k = 1;
        } else if x >= self.q[5] {
            self.q[5] = self.q[5].max(x);
            k = 4;
        } else {
            let mut j = 1;
            while !(self.q[j] <= x && x < self.q[j + 1]) {
                j += 1;
            }
            k = j;
        }
        // B2: positions above the cell, and every desired position.
        for i in k + 1..=5 {
            self.n[i] += 1;
        }
        for i in 1..=5 {
            self.nd[i] += self.dn[i];
        }
        // B3: the three middle markers, each by at most one position.
        for i in 2..=4 {
            let d = self.nd[i] - self.n[i] as f64;
            let room_up = self.n[i + 1] - self.n[i] > 1;
            let room_down = self.n[i - 1] - self.n[i] < -1;
            if (d >= 1.0 && room_up) || (d <= -1.0 && room_down) {
                let sd: i64 = if d >= 0.0 { 1 } else { -1 };
                let s = sd as f64;
                let (qm, qi, qp) = (self.q[i - 1], self.q[i], self.q[i + 1]);
                let (nm, ni, np) = (self.n[i - 1] as f64, self.n[i] as f64, self.n[i + 1] as f64);
                let candidate = qi
                    + s / (np - nm)
                        * ((ni - nm + s) * (qp - qi) / (np - ni)
                            + (np - ni - s) * (qi - qm) / (ni - nm));
                if qm < candidate && candidate < qp {
                    self.q[i] = candidate;
                    self.parabolic += 1;
                } else {
                    let j = (i as i64 + sd) as usize;
                    self.q[i] = qi + s * (self.q[j] - qi) / (self.n[j] as f64 - ni);
                    self.linear += 1;
                }
                self.n[i] += sd;
            }
        }
    }

    /// The middle height; before the sixth value, the order statistic of
    /// the values so far at `floor(min(m - 1, m p))`.
    fn get(&self) -> Option<f64> {
        if self.n[1] != 0 {
            return Some(self.q[3]);
        }
        let mut seen = self.seen.clone();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = seen.len();
        if m == 0 {
            return None;
        }
        let at = (m - 1).min((m as f64 * self.p).floor() as usize);
        Some(seen[at])
    }
}

const LEVELS: [f64; 5] = [0.05, 0.25, 0.5, 0.9, 0.95];

/// Every read, after every value, is the longhand recursion's to the bit, at
/// five levels on six kinds of stream and three seeds; both predictions are
/// taken, so each is held.
#[test]
fn every_read_is_the_longhand_recursion_to_the_bit() {
    let (mut parabolic, mut linear, mut reads) = (0, 0, 0);
    for seed in 0..3 {
        for xs in streams(seed, 3_000) {
            let mut sketch = PSquared::new(&LEVELS).unwrap();
            let mut hand: Vec<Longhand> = LEVELS.iter().map(|&p| Longhand::new(p)).collect();
            assert!((0..LEVELS.len()).all(|l| sketch.get(l).is_none()));
            for (t, &x) in xs.iter().enumerate() {
                sketch.update(x);
                for (l, h) in hand.iter_mut().enumerate() {
                    h.add(x);
                    let (a, b) = (sketch.get(l).unwrap(), h.get().unwrap());
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "seed {seed} row {t} level {l}: {a} {b}"
                    );
                    reads += 1;
                }
                assert_eq!(sketch.count(), (t + 1) as f64);
            }
            for h in &hand {
                parabolic += h.parabolic;
                linear += h.linear;
            }
        }
    }
    assert_eq!(reads, 3 * 6 * 3_000 * LEVELS.len());
    assert!(
        parabolic > 10_000 && linear > 100,
        "parabolic {parabolic}, linear {linear}"
    );
}

/// Before the sixth value, the read is the sorted values' entry at
/// `floor(min(m - 1, m p))`: the first value, then at `p = 0.5` the larger
/// of two, the middle of three; at `p = 0.95` the largest of five, which the
/// markers' middle height then replaces.
#[test]
fn before_the_sixth_value_the_read_is_an_order_statistic() {
    let mut s = PSquared::new(&[0.5, 0.95]).unwrap();
    let xs = [3.0, 1.0, 2.0, 5.0, 4.0];
    let want_median = [3.0, 3.0, 2.0, 3.0, 3.0];
    let want_95 = [3.0, 3.0, 3.0, 5.0, 5.0];
    for (i, &x) in xs.iter().enumerate() {
        s.update(x);
        assert_eq!(s.get(0), Some(want_median[i]), "median after {}", i + 1);
        assert_eq!(s.get(1), Some(want_95[i]), "0.95 after {}", i + 1);
    }
    // The sixth value, 6, lands in the top cell: the top height moves to 6
    // and its position to 6. At p = 0.5 the desired positions become
    // [1, 2.25, 3.5, 4.75, 6], none a whole place from its marker, so no
    // middle marker moves. At p = 0.95 they become [1, 3.375, 5.75, 5.875,
    // 6], and only the fourth marker has room to move (positions 4 and 6):
    // the parabola through (3, 3), (4, 4), (6, 6) gives 4 + (1/3)(2·2/2 +
    // 1·1/1) = 5, inside (3, 6), so it moves to height 5 at position 5.
    s.update(6.0);
    assert_eq!(s.get(0), Some(3.0));
    assert_eq!(s.get(1), Some(3.0));
    assert_eq!(s.markers[0].q, [1.0, 2.0, 3.0, 4.0, 6.0]);
    assert_eq!(s.markers[0].n, [1.0, 2.0, 3.0, 4.0, 6.0]);
    assert_eq!(s.markers[0].desired, [1.0, 2.25, 3.5, 4.75, 6.0]);
    assert_eq!(s.markers[1].q, [1.0, 2.0, 3.0, 5.0, 6.0]);
    assert_eq!(s.markers[1].n, [1.0, 2.0, 3.0, 5.0, 6.0]);
}

/// The markers' invariants after every value: heights ascending, from the
/// smallest value seen to the largest; positions strictly ascending whole
/// numbers from 1 to the count; the desired positions' ends at 1 and the
/// count.
#[test]
fn the_markers_keep_their_order_and_count_the_values() {
    for xs in streams(7, 2_000) {
        let mut s = PSquared::new(&LEVELS).unwrap();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (t, &x) in xs.iter().enumerate() {
            s.update(x);
            lo = lo.min(x);
            hi = hi.max(x);
            for m in &s.markers {
                assert!(m.q.windows(2).all(|w| w[0] <= w[1]), "row {t}: {:?}", m.q);
                assert_eq!((m.q[0], m.q[4]), (lo, hi));
                assert!(m.n.windows(2).all(|w| w[0] < w[1]), "row {t}: {:?}", m.n);
                assert!(m.n.iter().all(|v| v.fract() == 0.0));
                assert_eq!((m.n[0], m.n[4]), (1.0, (t + 1) as f64));
                assert_eq!((m.desired[0], m.desired[4]), (1.0, (t + 1) as f64));
            }
        }
    }
}

/// A value that is not finite is not seen: a stream with NaN and the
/// infinities among its values leaves the state it leaves without them, at
/// every step, before the sixth value and after.
#[test]
fn a_value_that_is_not_finite_is_not_seen() {
    let xs = &streams(3, 400)[0];
    let mut plain = PSquared::new(&LEVELS).unwrap();
    let mut noisy = PSquared::new(&LEVELS).unwrap();
    for (t, &x) in xs.iter().enumerate() {
        if t % 3 == 0 {
            noisy.update([f64::NAN, f64::INFINITY, f64::NEG_INFINITY][t % 9 / 3]);
        }
        plain.update(x);
        noisy.update(x);
        assert_eq!(plain, noisy, "row {t}");
    }
}

/// A state saved and loaded goes on to the bit, saved before the sixth value
/// (the first values held) and after (the markers held), and a reset goes
/// back to a new estimate at the same levels.
#[test]
fn a_saved_state_continues_to_the_bit() {
    for cut in [0, 3, 5, 6, 250] {
        let xs = &streams(11, 500)[1];
        let mut live = PSquared::new(&LEVELS).unwrap();
        for &x in &xs[..cut] {
            live.update(x);
        }
        let bytes = rmp_serde::to_vec(&live).unwrap();
        let mut back: PSquared = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.has_shape());
        assert_eq!(back, live);
        for &x in &xs[cut..] {
            live.update(x);
            back.update(x);
            for l in 0..LEVELS.len() {
                assert_eq!(live.get(l).map(f64::to_bits), back.get(l).map(f64::to_bits));
            }
        }
        assert_eq!(
            rmp_serde::to_vec(&back).unwrap(),
            rmp_serde::to_vec(&live).unwrap()
        );
    }
    let mut s = PSquared::new(&LEVELS).unwrap();
    streams(1, 50)[0].iter().for_each(|&x| s.update(x));
    s.reset();
    assert_eq!(s, PSquared::new(&LEVELS).unwrap());
    assert_eq!((s.get(0), s.count()), (None, 0.0));
}

/// The state is fixed by the levels, not the stream: 15 numbers a level
/// (120 bytes) and the level itself, the same after 10 values as after
/// 200,000.
#[test]
fn the_state_is_bounded_by_the_levels() {
    assert_eq!(std::mem::size_of::<Markers>(), 120);
    let mut s = PSquared::new(&LEVELS).unwrap();
    let mut seed = 5u64;
    for _ in 0..10 {
        s.update(lcg(&mut seed));
    }
    let early = s.bytes();
    for _ in 0..200_000 {
        s.update(lcg(&mut seed));
    }
    assert_eq!(s.bytes(), early);
    let fixed = std::mem::size_of::<PSquared>();
    assert_eq!(early, fixed + LEVELS.len() * (120 + 8));
}

/// A level at or past 0 or 1, or NaN, is refused, naming it; a state that
/// does not have the shape `new` gives is told apart.
#[test]
fn a_level_outside_zero_and_one_is_refused() {
    for bad in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
        let err = PSquared::new(&[0.5, bad]).unwrap_err();
        assert!(err.contains(&format!("{bad}")), "{err}");
    }
    let mut s = PSquared::new(&[0.5]).unwrap();
    assert!(s.has_shape());
    (0..6).for_each(|i| s.update(i as f64));
    assert!(s.has_shape());
    let mut two_sets = s.clone();
    two_sets.markers.push(two_sets.markers[0].clone());
    assert!(!two_sets.has_shape());
    let mut both = s.clone();
    both.first.push(1.0);
    assert!(!both.has_shape());
    let mut bad_level = s;
    bad_level.levels[0] = 1.0;
    assert!(!bad_level.has_shape());
    let mut six = PSquared::new(&[0.5]).unwrap();
    six.first = vec![0.0; 6];
    assert!(!six.has_shape());
}
