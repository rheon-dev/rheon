//! The run length of unchanged values (PLAN 564, part 2): for one stream of
//! values, how many consecutive rows, ending at the current one, carried the
//! same value.
//!
//! **Definition.** Rows `1..=t` carry values `v_1..v_t`, each a number or
//! missing. The run length at row `t` is the largest `n` with
//! `same(v_{t-i}, v_{t-i+1})` for every `i` in `1..n`: the length of the run
//! of `v_t`, counted up to and including row `t`. It is Polars' `rle` read
//! row by row: row `t` reports its place in its run, so the last row of a run
//! reports the run's whole length.
//!
//! **Unchanged.** Two numbers are the same when `|a − b| ≤ tol`, `tol ≥ 0`
//! (exactly equal under the default `tol = 0`, read with `==`, so `0.0` and
//! `-0.0` are the same, as Polars' `rle` has them). Each row is compared
//! with the row before it, so under a positive `tol` a run can creep: each
//! step within `tol`, the run's ends further apart. Two missing values are
//! the same, and a missing value and a number are not: a run of missing
//! values is a run of its own, as Polars' `rle` makes a run of nulls one.
//! What is missing is the caller's to say: the window engine reads a null,
//! a NaN, an infinity and a value beyond `1e100` as missing, the rule every
//! one of its operators follows.
//!
//! **Weight.** A row of weight 0 learns nothing (rule 9): it neither extends
//! nor ends a run, and reports the run as it stands, or `None` before the
//! first row of weight above 0. Any other weight counts the row once: a run
//! length is a count of rows, not a sum of weights.
//!
//! **State.** The tolerance, the last value and the run's count: 48 bytes
//! in memory (a value is an `i128` wide), bounded by nothing the stream does.

use serde::{Deserialize, Serialize};

/// One row's value as a run reads it: missing, a double, or an integer read
/// whole (an integer column past `2^53`, or a temporal one in nanoseconds,
/// keeps every unit, where a double would make two neighbors one value).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RunValue {
    Missing,
    Float(f64),
    Int(i128),
}

impl RunValue {
    /// Whether `self` and `next`, adjacent rows, are the same value under
    /// `tol`: two missing values are, a missing value and a number are not,
    /// and two numbers are when `|self − next| ≤ tol`. Between two integers
    /// the difference is taken in integers and rounded once.
    pub fn same(self, next: RunValue, tol: f64) -> bool {
        match (self, next) {
            (RunValue::Missing, RunValue::Missing) => true,
            (RunValue::Missing, _) | (_, RunValue::Missing) => false,
            (RunValue::Int(a), RunValue::Int(b)) => a == b || (a.abs_diff(b) as f64) <= tol,
            (a, b) => {
                let (a, b) = (a.as_f64(), b.as_f64());
                a == b || (a - b).abs() <= tol
            }
        }
    }

    fn as_f64(self) -> f64 {
        match self {
            RunValue::Float(x) => x,
            RunValue::Int(i) => i as f64,
            RunValue::Missing => f64::NAN,
        }
    }
}

/// The run of one stream of values ([the module docs](self)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunLength {
    /// The largest difference two adjacent values of one run may have.
    tol: f64,
    /// The last value of weight above 0.
    last: RunValue,
    /// The current run's rows of weight above 0; 0 before the first.
    n: u64,
}

impl RunLength {
    /// A run length under `tol`.
    ///
    /// # Errors
    ///
    /// A `tol` that is negative, NaN or infinite.
    pub fn new(tol: f64) -> Result<Self, String> {
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(format!(
                "tol must be finite and at least 0, but it is {tol}. Set tol to 0 for values \
                 exactly equal, or to the largest difference two adjacent rows of one run may \
                 have."
            ));
        }
        Ok(Self {
            tol,
            last: RunValue::Missing,
            n: 0,
        })
    }

    /// The tolerance this run was built with.
    pub fn tol(&self) -> f64 {
        self.tol
    }

    /// Take row `x` of weight `w` and return its run length: the count of
    /// the run it ends, itself included. A weight of 0 (or one that is not
    /// above 0) takes nothing and returns the run as it stands, `None`
    /// before any row was taken.
    pub fn step(&mut self, x: RunValue, w: f64) -> Option<u64> {
        // `w > 0` is false for 0 and NaN alike: a row that carries no
        // weight is not a row of the run.
        if w > 0.0 {
            if self.n > 0 && self.last.same(x, self.tol) {
                self.n += 1;
            } else {
                self.n = 1;
            }
            self.last = x;
        }
        self.current()
    }

    /// The current run's length, `None` before any row of weight above 0.
    pub fn current(&self) -> Option<u64> {
        (self.n > 0).then_some(self.n)
    }

    /// Forget the run: the next row of weight above 0 starts one of 1, as
    /// the stream's first row does. A new session or a reset of the clock.
    pub fn restart(&mut self) {
        self.last = RunValue::Missing;
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use RunValue::{Float as F, Int as I, Missing as M};

    fn run(tol: f64, rows: &[(RunValue, f64)]) -> Vec<Option<u64>> {
        let mut r = RunLength::new(tol).unwrap();
        rows.iter().map(|&(x, w)| r.step(x, w)).collect()
    }

    fn ones(xs: &[RunValue]) -> Vec<(RunValue, f64)> {
        xs.iter().map(|&x| (x, 1.0)).collect()
    }

    /// Polars' `rle` on `[1, 1, null, null, 1, 0.0, -0.0, 2]`, read row by
    /// row: lengths 2, 2, 1, 2, 1, so each row's place in its run.
    #[test]
    fn a_run_is_polars_rle_read_row_by_row() {
        let got = run(
            0.0,
            &ones(&[F(1.0), F(1.0), M, M, F(1.0), F(0.0), F(-0.0), F(2.0)]),
        );
        let want = [1, 2, 1, 2, 1, 1, 2, 1].map(Some);
        assert_eq!(got, want);
    }

    /// A missing value next to a number ends the run on both sides, and a
    /// run of missing values counts as one: the first row of a stream that
    /// is missing starts a run of 1, as a number would.
    #[test]
    fn a_missing_value_is_a_run_of_its_own() {
        assert_eq!(run(0.0, &ones(&[M, M, M])), [1, 2, 3].map(Some));
        assert_eq!(run(0.0, &ones(&[F(3.0), M, F(3.0)])), [1, 1, 1].map(Some));
        assert!(M.same(M, 0.0));
        assert!(!M.same(F(0.0), 0.0) && !F(0.0).same(M, 1e300));
        assert!(!M.same(I(0), 0.0) && !I(0).same(M, 0.0));
    }

    /// Both sides of `tol`: a step of exactly `tol` stays in the run, and
    /// the next double above it ends it; a run creeps a step at a time.
    #[test]
    fn tol_is_inclusive_and_read_between_adjacent_rows() {
        let tol = 0.5;
        assert!(F(1.0).same(F(1.5), tol));
        assert!(!F(1.0).same(F(1.5f64.next_up()), tol));
        assert!(F(1.5).same(F(1.0), tol), "either direction");
        assert_eq!(
            run(tol, &ones(&[F(0.0), F(0.5), F(1.0), F(1.5), F(2.5)])),
            [1, 2, 3, 4, 1].map(Some),
            "each step is within tol, the run's ends 1.5 apart"
        );
        // At tol = 0, the smallest difference ends a run.
        assert_eq!(
            run(0.0, &ones(&[F(1.0), F(1.0f64.next_up())])),
            [1, 1].map(Some)
        );
    }

    /// Integers are compared whole: two neighbors past `2^53`, one double
    /// apart, are two values; under a `tol`, their difference is read in
    /// integers.
    #[test]
    fn integers_are_compared_whole() {
        let big = 1_i128 << 60;
        assert!(!I(big).same(I(big + 1), 0.0));
        assert!(I(big).same(I(big), 0.0));
        assert!(I(big).same(I(big + 3), 3.0) && !I(big).same(I(big + 4), 3.0));
        assert!(I(big + 3).same(I(big), 3.0), "either direction");
        assert!(
            I(i128::MAX).same(I(i128::MIN), f64::MAX),
            "the difference does not overflow"
        );
        // An integer and a double, as doubles.
        assert!(I(2).same(F(2.0), 0.0) && !I(2).same(F(2.5), 0.25) && I(2).same(F(2.5), 0.5));
    }

    /// Rule 9: a row of weight 0 neither extends nor ends a run, and reports
    /// the run as it stands; before the first row of weight it reports
    /// none. A NaN or negative weight is no weight either.
    #[test]
    fn a_row_of_weight_zero_neither_extends_nor_ends_a_run() {
        let rows = [
            (F(7.0), 0.0),
            (F(1.0), 1.0),
            (F(9.0), 0.0),
            (F(1.0), 2.5),
            (M, 0.0),
            (F(1.0), f64::NAN),
            (F(2.0), -1.0),
            (F(1.0), 1e-300),
        ];
        assert_eq!(
            run(0.0, &rows),
            [
                None,
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(3)
            ]
        );
    }

    /// A restart forgets the run: the next row starts one of 1, even at the
    /// value the run held, and a zero-weight row after it reports none.
    #[test]
    fn a_restart_forgets_the_run() {
        let mut r = RunLength::new(0.0).unwrap();
        assert_eq!(r.step(F(4.0), 1.0), Some(1));
        assert_eq!(r.step(F(4.0), 1.0), Some(2));
        assert_eq!(r.current(), Some(2));
        r.restart();
        assert_eq!(r.current(), None);
        assert_eq!(r.step(F(4.0), 0.0), None);
        assert_eq!(
            r.step(M, 1.0),
            Some(1),
            "a missing value after a restart starts a run"
        );
        assert_eq!(r.step(F(4.0), 1.0), Some(1));
        assert_eq!(r.step(F(4.0), 1.0), Some(2));
    }

    /// A state saved mid-run continues to the same lengths.
    #[test]
    fn a_saved_run_continues_to_the_same_lengths() {
        let xs = [F(1.0), F(1.0), M, I(3), I(3), F(3.0), F(-0.0), F(0.0)];
        let mut whole = RunLength::new(0.25).unwrap();
        let want: Vec<_> = xs.iter().map(|&x| whole.step(x, 1.0)).collect();
        for cut in 0..=xs.len() {
            let mut a = RunLength::new(0.25).unwrap();
            let mut got: Vec<_> = xs[..cut].iter().map(|&x| a.step(x, 1.0)).collect();
            let bytes = rmp_serde::to_vec_named(&a).unwrap();
            let mut b: RunLength = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(b, a);
            got.extend(xs[cut..].iter().map(|&x| b.step(x, 1.0)));
            assert_eq!(got, want, "cut at {cut}");
        }
        assert_eq!(whole.tol(), 0.25);
    }

    /// The state's size, as the module docs state it.
    #[test]
    fn the_state_is_48_bytes() {
        assert_eq!(std::mem::size_of::<RunLength>(), 48);
    }

    #[test]
    fn tol_must_be_finite_and_at_least_zero() {
        assert!(RunLength::new(0.0).is_ok());
        assert!(RunLength::new(f64::MAX).is_ok());
        for bad in [-1e-300, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let e = RunLength::new(bad).unwrap_err();
            assert!(e.starts_with("tol must be finite and at least 0"), "{e}");
        }
    }
}
