//! The extrema behind `rolling_max` and `rolling_min` (task 561): the
//! largest or smallest value a window holds, on the window core's queue.
//!
//! **What it computes.** The extremum of the values in the window, each a
//! row's own value (a null, NaN, an infinity or a value beyond 1e100 is no
//! value, as for every operator). The queue's sums count the values for
//! `min_samples`; the extremum is kept here, beside them, as a quantile's
//! sketch is (`window_quantile.rs`).
//!
//! **Ties.** The only equal doubles with different bits are `0.0` and
//! `-0.0`. Looking back, the earliest row of the window wins, Polars'
//! `rolling_max_by` / `rolling_min_by` rule: two values are joined oldest
//! first and the older is kept unless the newer is strictly beyond it.
//! Looking ahead, the mirror: the latest row wins, the row farthest from the
//! row the window is read at, so a forward window is the backward window of
//! the reversed stream to the bit. The join copies a value and never
//! computes one, and it is associative, so where the two stacks turn over
//! changes no bit.
//!
//! **Three shapes, as the queue has.** With no window, one running extremum
//! and its count; under a `since` event the running form keeps the current
//! stamp's rows apart (`cur`), since an event read anywhere in a stamp starts
//! the run at that stamp, its earlier rows included. A windowed queue is the
//! two-stack queue: `back`'s rows are one running extremum (`back`), and each
//! of `front`'s rows keeps the extremum from itself through `front`'s newest
//! row, built once as `back` turns over into it, so a row leaving takes its
//! own entry with it and leaves the rest exact. That is one double a row.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::{Direction, Group, KernelDef, OffForm, OpDef, Queue, empty_value, gap};
use super::{ROW, ROW_VAR, SUM, SUM_VAR};

/// A group's `max_state_bytes` for an extremum's queue where none is given:
/// 256 KB, every model's default for a state the data sizes.
pub const DEFAULT_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be: 64 MiB a group, as every model's.
pub const STATE_BYTES_CEILING: u64 = 64 << 20;

/// What an extremum asks of its kernel and of its own fields.
pub(super) fn check_op(o: usize, op: &OpDef, k: &KernelDef) -> Result<(), String> {
    if op.stat.is_extremum() && !k.half_life.is_infinite() {
        return Err(format!(
            "operator {o}: an extremum weighs no row, so its kernel's half-life must be inf. \
             Give it a kernel of its own with half_life = inf."
        ));
    }
    if k.since.is_some() && !op.stat.is_extremum() {
        return Err(format!(
            "operator {o}: only rolling_max and rolling_min take since, but this one is a {:?}. \
             Give since to an extremum, or leave it out.",
            op.stat
        ));
    }
    match (op.stat.is_extremum(), op.max_state_bytes) {
        (false, Some(_)) => Err(format!(
            "operator {o}: max_state_bytes is given, but only rolling_max and rolling_min take it. \
             Leave out max_state_bytes."
        )),
        (true, Some(0)) => Err(format!(
            "operator {o}: max_state_bytes must be above 0, but it is 0. Give the bytes a \
             group's window may hold, or leave it out for 256 KB."
        )),
        (true, Some(c)) if c > STATE_BYTES_CEILING => Err(format!(
            "operator {o}: max_state_bytes may be at most {STATE_BYTES_CEILING} (64 MiB) a group, but it is \
             {c}. Lower it: a group's window is held in memory beside every other group's."
        )),
        _ => Ok(()),
    }
}

/// A row's bytes in a queue of `n` operators, `ext` of them extrema: its
/// place, policy time, raw clock, interval end and anchor, then each
/// operator's value and sums at the queue's width, and a double an
/// extremum keeps beside them.
pub fn row_bytes(n: usize, wide: bool, ext: usize) -> usize {
    let per = if wide { ROW_VAR + SUM_VAR } else { ROW + SUM };
    8 * (5 + per * n + ext)
}

/// The bytes a queue holds in its rows, as [`row_bytes`] counts them.
pub(super) fn queue_bytes(q: &Queue) -> usize {
    let ext = q.ext.iter().flatten().count();
    q.len() * row_bytes(q.n, q.wide(), ext)
}

/// Kernel `k`'s cap on its queue's bytes a group: the least its extrema
/// give, where it holds one under a window; `None` for any other.
pub(super) fn cap_of(k: &KernelDef, ops: &[OpDef], members: &[usize]) -> Option<u64> {
    k.window_size?;
    members
        .iter()
        .filter(|&&o| ops[o].stat.is_extremum())
        .map(|&o| ops[o].max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES))
        .min()
}

impl super::Windows {
    /// Per kernel whose queue an extremum bounds: its index, a row's bytes
    /// in its queue and the cap on a group's queue (task 561). On the row
    /// clock the rows a window holds are known from its size, so a caller
    /// refuses a window past its cap before the run; on a clock they are
    /// the data's, and [`super::Windows::push`] refuses the row that takes a
    /// queue past it.
    pub fn extremum_caps(&self) -> Vec<(usize, usize, u64)> {
        (0..self.kernels.len())
            .filter_map(|ki| {
                let m = &self.members[ki];
                let cap = cap_of(&self.kernels[ki], &self.ops, m)?;
                let wide = m.iter().any(|&o| self.ops[o].stat.is_var());
                let ext = m
                    .iter()
                    .filter(|&&o| self.ops[o].stat.is_extremum())
                    .count();
                Some((ki, row_bytes(m.len(), wide, ext), cap))
            })
            .collect()
    }
}

/// A `since` event at the stamp of raw clock `off`: the window starts over
/// there. Looking back with a window, the rows of the earlier stamps leave
/// it; with none, the run before the stamp is let go (its rows at this
/// stamp stay, an event acting on its whole stamp). Under `"left"` and
/// `"none"` the stamp's own value, read as it began, is that of a window
/// holding no row.
#[allow(clippy::too_many_arguments)]
pub(super) fn event(
    k: &KernelDef,
    ops: &[OpDef],
    members: &[usize],
    g: &mut Group,
    ki: usize,
    scratch: &mut [f64],
    off: i64,
    form: OffForm,
) {
    let q = &mut g.queues[ki];
    if k.window_size.is_some() {
        let stale =
            |_: u64, _: f64, t_off: i64| gap(form, t_off, off).cmp_zero() == Ordering::Greater;
        if q.wide() {
            q.evict::<true>(k, scratch, stale);
        } else {
            q.evict::<false>(k, scratch, stale);
        }
    } else {
        q.ext.iter_mut().flatten().for_each(XSide::event);
    }
    q.since_off = Some(off);
    if !k.closed.near(Direction::Backward) {
        for (j, &o) in members.iter().enumerate() {
            g.stamp_values[ki][j] = empty_value(&ops[o]);
        }
    }
}

/// Whether a backward window read at policy time `tau`, raw clock `off`,
/// spans its whole `window_size`: inside its stretch, and since its
/// kernel's last `since` event. Inlined: every backward kernel's flush reads
/// it, and a kernel with no `window_size` answers at the first test.
#[inline]
pub(super) fn complete(k: &KernelDef, q: &Queue, tau: f64, off: i64, form: OffForm) -> bool {
    k.window_size.is_none_or(|w| {
        tau >= w
            && q.since_off
                .is_none_or(|s| gap(form, s, off).cmp_window(k) != Ordering::Less)
    })
}

/// `older` and `newer` joined: the value beyond the other, and on a tie the
/// older one, or the newer `later` (a forward window). NaN is no value.
#[inline]
pub(crate) fn pick(max: bool, later: bool, older: f64, newer: f64) -> f64 {
    if older.is_nan() {
        return newer;
    }
    if newer.is_nan() {
        return older;
    }
    let beyond = |a: f64, b: f64| if max { a > b } else { a < b };
    if later {
        if beyond(older, newer) { older } else { newer }
    } else if beyond(newer, older) {
        newer
    } else {
        older
    }
}

/// One extremum operator's part of a queue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct XSide {
    /// A maximum; a minimum when false.
    max: bool,
    /// A forward kernel's: on a tie the later row wins.
    later: bool,
    /// A windowed queue's: per row of its `front` stack, the extremum from
    /// that row through `front`'s newest (index 0 is the newest), NaN for none.
    front: Vec<f64>,
    /// A windowed queue's: the extremum of its `back` stack's rows.
    back: f64,
    /// The running form's extremum and count of values, before `cur`.
    run: f64,
    run_n: f64,
    /// Under a `since` event, the running form's current stamp's extremum and
    /// count, joined into `run` as the next stamp arrives.
    cur: f64,
    cur_n: f64,
}

impl XSide {
    pub(crate) fn new(max: bool, later: bool) -> Self {
        XSide {
            max,
            later,
            front: Vec::new(),
            back: f64::NAN,
            run: f64::NAN,
            run_n: 0.0,
            cur: f64::NAN,
            cur_n: 0.0,
        }
    }

    #[inline]
    fn pick(&self, older: f64, newer: f64) -> f64 {
        pick(self.max, self.later, older, newer)
    }

    /// A row's value `x` (NaN for none) joins: `back` in a windowed queue;
    /// the running extremum, or the current stamp's under `split` (a
    /// `since` event), in one with no window.
    #[inline]
    pub(crate) fn push(&mut self, windowed: bool, split: bool, x: f64) {
        if windowed {
            self.back = self.pick(self.back, x);
        } else if !x.is_nan() {
            if split {
                self.cur = self.pick(self.cur, x);
                self.cur_n += 1.0;
            } else {
                self.run = self.pick(self.run, x);
                self.run_n += 1.0;
            }
        }
    }

    /// `back` turns over into an empty `front`: `x(i)` is `back`'s row `i`,
    /// oldest first, of `len`.
    pub(crate) fn turn_over(&mut self, len: usize, x: impl Fn(usize) -> f64) {
        debug_assert!(self.front.is_empty(), "front turns over only when empty");
        self.front.reserve(len);
        let mut acc = f64::NAN;
        for i in (0..len).rev() {
            acc = self.pick(x(i), acc);
            self.front.push(acc);
        }
        self.back = f64::NAN;
    }

    /// `front`'s oldest row leaves.
    #[inline]
    pub(crate) fn pop(&mut self) {
        self.front.pop();
    }

    /// The running form's next stamp: the current stamp's rows join the run.
    #[inline]
    pub(crate) fn new_stamp(&mut self) {
        if self.cur_n > 0.0 {
            self.run = self.pick(self.run, self.cur);
            self.run_n += self.cur_n;
            self.cur = f64::NAN;
            self.cur_n = 0.0;
        }
    }

    /// A `since` event at the current stamp: the run before it is let go.
    #[inline]
    pub(crate) fn event(&mut self) {
        self.run = f64::NAN;
        self.run_n = 0.0;
    }

    /// The running form's extremum and count of values.
    #[inline]
    pub(crate) fn running(&self) -> (f64, f64) {
        (self.pick(self.run, self.cur), self.run_n + self.cur_n)
    }

    /// A windowed queue's extremum, with `open`, a forward kernel's open row's
    /// value, the newest, joined last; NaN for none.
    #[inline]
    pub(crate) fn windowed(&self, open: Option<f64>) -> f64 {
        let f = self.front.last().copied().unwrap_or(f64::NAN);
        let a = self.pick(f, self.back);
        match open {
            Some(x) => self.pick(a, x),
            None => a,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.front.clear();
        self.back = f64::NAN;
        self.run = f64::NAN;
        self.run_n = 0.0;
        self.cur = f64::NAN;
        self.cur_n = 0.0;
    }

    /// Whether it is an extremum of this kind, at `rows` rows of `front`.
    pub(crate) fn has_shape(&self, max: bool, later: bool, rows: usize) -> bool {
        self.max == max
            && self.later == later
            && self.front.len() == rows
            && self.run_n >= 0.0
            && self.cur_n >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tie_keeps_the_older_looking_back_and_the_later_looking_ahead() {
        let (pz, nz) = (0.0_f64, -0.0_f64);
        for max in [true, false] {
            assert_eq!(pick(max, false, nz, pz).to_bits(), nz.to_bits());
            assert_eq!(pick(max, false, pz, nz).to_bits(), pz.to_bits());
            assert_eq!(pick(max, true, nz, pz).to_bits(), pz.to_bits());
            assert_eq!(pick(max, true, pz, nz).to_bits(), nz.to_bits());
        }
        // Strictly beyond wins in either direction.
        assert_eq!(pick(true, false, 1.0, 2.0), 2.0);
        assert_eq!(pick(true, true, 2.0, 1.0), 2.0);
        assert_eq!(pick(false, false, 1.0, 2.0), 1.0);
        assert_eq!(pick(false, true, 2.0, 1.0), 1.0);
        // No value is never picked over a value.
        assert_eq!(pick(true, false, f64::NAN, -3.0), -3.0);
        assert_eq!(pick(false, true, 5.0, f64::NAN), 5.0);
        assert!(pick(true, false, f64::NAN, f64::NAN).is_nan());
    }

    /// The two stacks against a scan of the rows each window holds, at
    /// every turnover point a stream of pushes and pops reaches.
    #[test]
    fn the_two_stacks_are_a_scan_of_the_window() {
        let xs = [
            3.0,
            -0.0,
            0.0,
            f64::NAN,
            7.0,
            7.0,
            -2.0,
            0.0,
            -0.0,
            9.0,
            f64::NAN,
            1.0,
            1.0,
            -5.0,
        ];
        for max in [true, false] {
            for later in [false, true] {
                for w in 1..6 {
                    let mut s = XSide::new(max, later);
                    // The queue's rows: `front` (oldest last) then `back`.
                    let mut back: Vec<f64> = Vec::new();
                    let mut front_n = 0usize;
                    for (i, &x) in xs.iter().enumerate() {
                        back.push(x);
                        s.push(true, false, x);
                        let lo = (i + 1).saturating_sub(w);
                        // Evict the rows older than the window.
                        while i + 1 - lo < front_n + back.len() {
                            if front_n == 0 {
                                let b = std::mem::take(&mut back);
                                s.turn_over(b.len(), |k| b[k]);
                                front_n = b.len();
                            }
                            s.pop();
                            front_n -= 1;
                        }
                        let want = xs[lo..=i]
                            .iter()
                            .fold(f64::NAN, |a, &x| pick(max, later, a, x));
                        let got = s.windowed(None);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "max {max} later {later} w {w} row {i}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_event_lets_the_run_go_and_keeps_its_own_stamp() {
        let mut s = XSide::new(true, false);
        s.push(false, true, 5.0);
        s.new_stamp();
        s.push(false, true, 1.0);
        // The event arrives at the stamp's second row: the stamp's first is in.
        s.event();
        s.push(false, true, 2.0);
        assert_eq!(s.running(), (2.0, 2.0));
        s.new_stamp();
        s.push(false, true, f64::NAN);
        assert_eq!(s.running(), (2.0, 2.0));
        s.event();
        assert!(s.running().0.is_nan());
        assert_eq!(s.running().1, 0.0);
    }
}
