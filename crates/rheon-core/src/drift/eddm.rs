//! EDDM, the early drift detection method (Baena-García, del Campo-Ávila,
//! Fidalgo, Bifet, Gavaldà and Morales-Bueno 2006), as river's
//! `drift.binary.EDDM` computes it.
//!
//! Each row is an error or not. On each error, the distance `d` in rows since
//! the error before it (the first from the start) enters a running mean `p'`
//! and sample standard deviation `s'` (Welford, `ddof = 1`):
//!
//! ```text
//! k += 1;   p'_k = p'_{k−1} + (1/k)(d − p'_{k−1})
//! S_k = S_{k−1} + (d − p'_{k−1})(d − p'_k);   s' = sqrt(S_k / (k − 1)), 0 for k ≤ 1
//! ```
//!
//! Once more than `warm_start` rows have been seen: where `p' + 2s'` passes
//! its largest value so far, the largest moves; otherwise, once more than
//! `warm_start` errors have been seen, a drift is flagged where
//!
//! ```text
//! (p' + 2 s') / max(p' + 2 s') < beta
//! ```
//!
//! after which the detector starts again. river's warning level (`alpha`) is
//! not kept: nothing reads it.

use serde::{Deserialize, Serialize};

/// EDDM on one stream of errors: the module doc has the equations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Eddm {
    /// Rows, and errors, before a drift can be flagged: river's 30.
    warm_start: u64,
    /// The ratio to the largest `p' + 2s'` under which a drift is flagged:
    /// river's 0.9.
    beta: f64,
    /// Rows since the start.
    n: u64,
    /// The row of the last error, 0 for none.
    last_error: u64,
    /// Errors since the start.
    errors: u64,
    /// The distances' count, mean and sum of squared deviations.
    k: f64,
    mean: f64,
    sq: f64,
    /// The largest `p' + 2s'`, `−1` before the first.
    p2s_max: f64,
}

impl Eddm {
    pub fn new(warm_start: u64, beta: f64) -> Self {
        Self {
            warm_start,
            beta,
            n: 0,
            last_error: 0,
            errors: 0,
            k: 0.0,
            mean: 0.0,
            sq: 0.0,
            p2s_max: -1.0,
        }
    }

    /// The mean distance between errors, and the largest `p' + 2s'`.
    pub fn distances(&self) -> (f64, f64, f64) {
        (self.mean, self.sq, self.p2s_max)
    }

    /// Rows since the detector last started.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Whether a restored state keeps the shape every row relies on: a
    /// `beta` above 0 and at most 1, as a spec takes it, a last error and an
    /// error count no later and no more than the rows seen (the next
    /// error's distance is `n − last_error`), one distance an error, a
    /// finite mean and a spread at least 0, and a largest `p' + 2s'` that is
    /// finite.
    pub fn sound(&self) -> bool {
        self.beta > 0.0
            && self.beta <= 1.0
            && self.last_error <= self.n
            && self.errors <= self.n
            && self.k == self.errors as f64
            && self.mean.is_finite()
            && self.sq.is_finite()
            && self.sq >= 0.0
            && self.p2s_max.is_finite()
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.warm_start, self.beta);
    }

    /// Feed one row, an error or not. Returns whether it flags a drift, and
    /// then starts again.
    pub fn update(&mut self, error: bool) -> bool {
        self.n += 1;
        if !error {
            return false;
        }
        self.errors += 1;
        let d = (self.n - self.last_error) as f64;
        let old = self.mean;
        self.k += 1.0;
        self.mean += (1.0 / self.k) * (d - self.mean);
        self.sq += 1.0 * (d - old) * (d - self.mean);
        self.last_error = self.n;
        if self.n > self.warm_start {
            let var = if self.k > 1.0 {
                self.sq / (self.k - 1.0)
            } else {
                0.0
            };
            let p2s = self.mean + 2.0 * var.sqrt();
            if p2s > self.p2s_max {
                self.p2s_max = p2s;
            } else if self.errors > self.warm_start && p2s / self.p2s_max < self.beta {
                self.reset();
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The distances' mean and sample deviation against a longhand over the
    /// distances themselves.
    #[test]
    fn the_distances_are_the_definitions() {
        let mut e = Eddm::new(30, 0.0);
        let (mut last, mut ds) = (0u64, Vec::<f64>::new());
        let mut s = 11u64;
        for t in 1..=3000u64 {
            let err = lcg(&mut s) < 0.15;
            assert!(!e.update(err), "beta 0 never flags");
            if err {
                ds.push((t - last) as f64);
                last = t;
                let k = ds.len() as f64;
                let mean = ds.iter().sum::<f64>() / k;
                let sq: f64 = ds.iter().map(|d| (d - mean) * (d - mean)).sum();
                let (m, q, _) = e.distances();
                assert!((m - mean).abs() < 1e-9 * mean, "row {t}");
                assert!((q - sq).abs() < 1e-9 * (1.0 + sq), "row {t}");
            }
        }
    }

    #[test]
    fn errors_coming_closer_together_are_flagged() {
        // Errors every 20 rows, so every distance is 20 and `p' + 2s'` holds
        // at 20; then every 2 rows.
        let mut e = Eddm::new(30, 0.9);
        let mut seen = false;
        for i in 0..20_000u64 {
            let err = if i < 4000 {
                (i + 1) % 20 == 0
            } else {
                i % 2 == 0
            };
            if e.update(err) {
                assert!(i >= 4000, "flagged at {i}, before the change");
                seen = true;
                break;
            }
        }
        assert!(seen);
        assert_eq!(e.n(), 0, "started again");
    }

    /// The errors' warm start, both sides: where a run first flags at row
    /// `R` with `E` errors seen, a warm start of `E - 1` flags at `R` and one
    /// of `E` does not. The rows' warm start holds back every flag before it.
    #[test]
    fn the_warm_starts_hold_back_a_flag() {
        // Distances of 10, then of 1: the ratio falls under beta.
        let feed = |warm: u64| {
            let mut e = Eddm::new(warm, 0.9);
            let mut errors = 0u64;
            for i in 1..=5000u64 {
                let err = if i <= 400 { i % 10 == 0 } else { true };
                errors += u64::from(err);
                if e.update(err) {
                    return Some((i, errors));
                }
            }
            None
        };
        let (r0, e0) = feed(0).expect("a flag");
        assert!(r0 > 400 && e0 > 1);
        assert_eq!(feed(e0 - 1).map(|f| f.0), Some(r0));
        assert!(feed(e0).expect("a later flag").0 > r0);
        assert!(feed(r0 + 50).is_none_or(|f| f.0 > r0 + 50));
    }

    /// The flag is `< beta`: at `beta` exactly it is not.
    #[test]
    fn beta_itself_is_not_a_drift() {
        // Distances 4, 4 (p' + 2s' = 4, the max), then 2: p' = 10/3, s' =
        // sqrt(4/3); not tidy. Use distances 2, 2, 2 then the ratio 1: p' + 2s'
        // stays 2 and is never past the max, so the ratio is 1 = beta.
        let mut e = Eddm::new(0, 1.0);
        for i in 1..=8u64 {
            assert!(!e.update(i % 2 == 0), "ratio 1 is not below beta 1");
        }
        let mut e = Eddm::new(0, 1.0 + 1e-12);
        let flagged = (1..=8u64).any(|i| e.update(i % 2 == 0));
        assert!(flagged, "ratio 1 is below a beta past it");
    }

    #[test]
    fn a_saved_detector_goes_on_to_the_bit() {
        let mut a = Eddm::new(30, 0.9);
        let mut s = 17u64;
        for _ in 0..700 {
            a.update(lcg(&mut s) < 0.1);
        }
        let mut b: Eddm = rmp_serde::from_slice(&rmp_serde::to_vec_named(&a).unwrap()).unwrap();
        for _ in 0..3000 {
            let err = lcg(&mut s) < 0.2;
            assert_eq!(a.update(err), b.update(err));
        }
        assert_eq!(a, b);
    }

    /// The whole test written out over the distances themselves: the mean
    /// and sample deviation, `p' + 2s'` with its largest so far, and the flag
    /// below `beta` of it once more than `warm_start` rows and errors have
    /// been seen. Flags, the largest `p' + 2s'` and the row count equal the
    /// longhand's on every row, at several warm starts (0 reads the first
    /// distance, whose deviation is 0, and the second) and levels.
    #[test]
    fn the_flag_is_the_longhand_over_the_distances() {
        for (warm, beta, rate) in [
            (0u64, 0.9, 0.15),
            (3, 0.95, 0.3),
            (10, 0.9, 0.1),
            (30, 0.97, 0.2),
        ] {
            let mut e = Eddm::new(warm, beta);
            let (mut last, mut ds, mut p2s_max) = (0u64, Vec::<f64>::new(), -1.0f64);
            let (mut n, mut errors, mut flags) = (0u64, 0u64, 0);
            let mut s = 21u64;
            for t in 0..6000 {
                // A change of regime so the distances move both ways.
                let rate = if (t / 700) % 2 == 1 { rate * 2.5 } else { rate };
                let err = lcg(&mut s) < rate;
                n += 1;
                let mut expect = false;
                if err {
                    errors += 1;
                    ds.push((n - last) as f64);
                    last = n;
                    if n > warm {
                        let k = ds.len() as f64;
                        let mean = ds.iter().sum::<f64>() / k;
                        let sd = if ds.len() > 1 {
                            (ds.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (k - 1.0))
                                .sqrt()
                        } else {
                            0.0
                        };
                        let p2s = mean + 2.0 * sd;
                        if p2s > p2s_max {
                            p2s_max = p2s;
                        } else if errors > warm && p2s / p2s_max < beta {
                            expect = true;
                        }
                    }
                }
                let got = e.update(err);
                assert_eq!(got, expect, "warm {warm} row {t}");
                if expect {
                    (last, p2s_max, n, errors) = (0, -1.0, 0, 0);
                    ds.clear();
                    flags += 1;
                    assert_eq!(e.n(), 0);
                } else {
                    assert_eq!(e.n(), n, "row {t}");
                    let want = p2s_max;
                    let have = e.distances().2;
                    assert!(
                        (have - want).abs() <= 1e-9 * want.abs(),
                        "warm {warm} row {t}: {have} {want}"
                    );
                }
            }
            assert!(flags >= 3, "warm {warm}: {flags} flags");
        }
    }

    /// The first row past `warm_start` is the one that reads: an error on
    /// row `warm_start` itself leaves the largest untouched, the next
    /// sets it, and the first distance read alone has a deviation of 0.
    #[test]
    fn the_warm_start_is_strict_and_one_distance_has_no_deviation() {
        let mut e = Eddm::new(5, 0.9);
        for t in 1..=5 {
            e.update(t == 3 || t == 5);
        }
        assert_eq!(e.distances().2, -1.0, "row 5 is the warm start");
        e.update(true);
        // Distances 3, 2, 1: mean 2, deviation 1: 2 + 2 = 4.
        assert!((e.distances().2 - 4.0).abs() < 1e-12, "{:?}", e.distances());
        let mut e = Eddm::new(0, 0.9);
        e.update(false);
        e.update(false);
        e.update(true);
        assert_eq!(e.distances().2, 3.0, "one distance: 3 + 0");
        e.update(true);
        // Distances 3 and 1: mean 2, sample variance 2: 2 + 2 sqrt 2.
        assert!((e.distances().2 - (2.0 + 2.0 * 2f64.sqrt())).abs() < 1e-12);
    }
}
