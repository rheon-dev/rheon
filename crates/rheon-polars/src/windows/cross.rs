//! Cross-group reductions: at each stamp, a statistic over every group's
//! row at that stamp, broadcast back to each of them.
//!
//! The window core reads one stream in clock order across its groups, so
//! the rows of one stamp are a run of consecutive rows, and a stamp is
//! closed by the first row of a later one, or by the end of the input.
//! Under `closed = "right"`, the default, a row reads its own stamp's
//! cross-section, every group's row at the stamp (its own included): the
//! row is held until the stamp closes, and every row of the stamp reads the
//! same value -- Polars' `pl.col(x).mean().over(clock)` on the whole frame.
//! Under `closed = "left"` a row reads the cross-section of the stamp before
//! its own, known as the row arrives, so nothing is held.
//!
//! **Weights.** A row's weight (1 without a `weight` input) scales what it
//! contributes; a row whose value or weight is missing, or whose weight is
//! not above 0, contributes nothing and counts in no stamp: it still reads
//! its stamp's value, and a stamp of such rows alone is no stamp, so under
//! `"left"` the row after it reads the stamp before it. A cross-section of
//! fewer than two weighted rows is withheld (null): one row is no spread
//! and no cross-section to place a row in.
//!
//! **Moments** (weighted, in input order within the stamp, Welford's form):
//!
//! ```text
//! V1' = V1 + w,   V2' = V2 + w^2,   d = x - m,   m' = m + (w / V1') d
//! M2' = M2 + w d (x - m')
//! mean = m,   var = M2 / V1 * V1^2 / (V1^2 - V2)   (bias: M2 / V1),   std = sqrt(var)
//! ```
//!
//! At unit weights the correction is `n / (n - 1)`, Polars' `std(ddof=1)`.
//!
//! **Clock events.** A step back refused by the stream's clock is refused
//! here too: the stream is in clock order. A reset discards the open stamp
//! (its rows read null) and the stamp before it. A gap past `gap_cap` or a
//! session change on the stream's clock closes the open stamp as any later
//! row does, and ends the previous stamp, so the first stamp after it reads
//! null under `"left"`, as a window does not span a clock break.

use rheon_core::ClockValue;
use serde::{Deserialize, Serialize};

use super::Meta;
use crate::stream::usable;
use std::collections::VecDeque;

/// Which stamp a cross operator reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossClosed {
    /// The row's own stamp: the row waits for the stamp to close.
    Right,
    /// The stamp before the row's own: known as the row arrives.
    Left,
}

impl CrossClosed {
    pub fn name(self) -> &'static str {
        match self {
            CrossClosed::Right => "right",
            CrossClosed::Left => "left",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "right" => Some(CrossClosed::Right),
            "left" => Some(CrossClosed::Left),
            _ => None,
        }
    }
}

/// A cross operator's statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossStat {
    Mean,
    Std,
}

/// One cross operator: its statistic, which stamp it reads, its input and
/// its weight, each an index into a row's values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossDef {
    pub stat: CrossStat,
    pub closed: CrossClosed,
    pub input: usize,
    /// `None` weighs every row 1.
    pub weight: Option<usize>,
    /// A standard deviation's: the weighted second moment about the mean as
    /// it is, uncorrected for the weights' count (Polars' `bias`).
    pub bias: bool,
    /// The most bytes the operator's state may hold for its open stamp.
    pub cap: u64,
}

/// A cross operator's `max_state_bytes` where the call sets none: 256 KB.
pub const CROSS_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be set to: 64 MiB.
pub const CROSS_STATE_BYTES_CEILING: u64 = 64 << 20;

/// The weighted moments of one stamp's rows for one operator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct Acc {
    /// Rows of weight above 0 with a value.
    n: u64,
    v1: f64,
    v2: f64,
    m: f64,
    m2: f64,
}

impl Acc {
    fn add(&mut self, x: f64, w: f64) {
        self.n += 1;
        self.v1 += w;
        self.v2 += w * w;
        let d = x - self.m;
        self.m += (w / self.v1) * d;
        self.m2 += w * d * (x - self.m);
    }

    fn value(&self, def: &CrossDef) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        match def.stat {
            CrossStat::Mean => self.m,
            CrossStat::Std => {
                let var = if def.bias {
                    self.m2 / self.v1
                } else {
                    let v1 = self.v1 * self.v1;
                    let den = v1 - self.v2;
                    if den > 0.0 {
                        self.m2 / self.v1 * (v1 / den)
                    } else {
                        f64::NAN
                    }
                };
                var.max(0.0).sqrt()
            }
        }
    }
}

/// The cross operators of one core and where their stamps stand.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cross {
    pub(super) defs: Vec<CrossDef>,
    /// The output index of the first cross operator: the window operators'
    /// count.
    pub(super) first_out: usize,
    /// The open stamp and its first row's `seq`; `None` before the first
    /// row and after a reset.
    open: Option<(Option<ClockValue>, u64)>,
    acc: Vec<Acc>,
    /// Per operator, the last closed stamp's value, for `"left"`; NaN where
    /// there is none since the last clock break.
    prev: Vec<f64>,
    /// Whether a row has taken an operator past half of its cap: said once a
    /// run, and kept with the state so a resumed run does not say it again.
    #[serde(default)]
    warned: bool,
    /// That warning, `(op, bytes, cap)`, not yet taken by the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    warning: Option<(usize, u64, u64)>,
}

/// What each operator would hold with a row ([`Cross::check_cap`]).
pub(super) struct HeldBytes(Vec<u64>);

/// The held rows a cross operator writes into.
pub(super) struct Rows<'a> {
    pub first: u64,
    pub n_outputs: usize,
    pub values: &'a mut VecDeque<f64>,
    pub meta: &'a mut VecDeque<Meta>,
}

impl Cross {
    pub fn new(defs: Vec<CrossDef>, first_out: usize) -> Self {
        let n = defs.len();
        Self {
            defs,
            first_out,
            open: None,
            acc: vec![Acc::default(); n],
            prev: vec![f64::NAN; n],
            warned: false,
            warning: None,
        }
    }

    /// The bytes each operator's state would hold with the row `seq` at
    /// `now`, `n_outputs` values a held row: under `"right"`, every row of
    /// the open stamp is held, its values and its count of open windows,
    /// until the stamp closes; under `"left"` nothing is held past the row.
    /// `Err` names the first operator the row would take past its cap.
    pub(super) fn check_cap(
        &self,
        now: Option<ClockValue>,
        seq: u64,
        n_outputs: usize,
    ) -> Result<HeldBytes, super::Refusal> {
        let rows = match self.open {
            Some((stamp, first)) if now.is_some() && stamp == now => seq - first + 1,
            _ => 1,
        };
        let per_row = 8 * n_outputs as u64 + std::mem::size_of::<Meta>() as u64;
        let mut held = Vec::with_capacity(self.defs.len());
        for (op, def) in self.defs.iter().enumerate() {
            let bytes = match def.closed {
                CrossClosed::Right => rows * per_row,
                CrossClosed::Left => 0,
            };
            if bytes > def.cap {
                return Err(super::Refusal::StateCap {
                    seq,
                    op,
                    bytes,
                    cap: def.cap,
                });
            }
            held.push(bytes);
        }
        Ok(HeldBytes(held))
    }

    /// After a row is taken: the first to take an operator past half of its
    /// cap raises the warning, once.
    pub(super) fn note_held(&mut self, held: &HeldBytes) {
        if self.warned {
            return;
        }
        if let Some((op, (&bytes, def))) = held
            .0
            .iter()
            .zip(&self.defs)
            .enumerate()
            .find(|(_, (b, d))| **b > d.cap / 2)
        {
            self.warned = true;
            self.warning = Some((op, bytes, def.cap));
        }
    }

    pub(super) fn take_warning(&mut self) -> Option<(usize, u64, u64)> {
        self.warning.take()
    }

    pub fn defs(&self) -> &[CrossDef] {
        &self.defs
    }

    /// Whether any operator reads its own stamp: then each row waits once.
    pub fn waits(&self) -> bool {
        self.defs.iter().any(|d| d.closed == CrossClosed::Right)
    }

    /// The first `seq` of the open stamp, for [`super::Windows::check`]:
    /// every row from it to the last is waiting on the stamp.
    pub(super) fn waiting_from(&self) -> Option<u64> {
        if self.waits() {
            self.open.map(|(_, s)| s)
        } else {
            None
        }
    }

    pub(super) fn has_shape(&self) -> bool {
        self.acc.len() == self.defs.len() && self.prev.len() == self.defs.len()
    }

    /// A row at `now`, before its slots are pushed: close the open stamp
    /// where `now` is a later one, or discard it on a reset; a clock break
    /// ends the stamp before.
    pub(super) fn before_row(
        &mut self,
        now: Option<ClockValue>,
        reset: bool,
        broke: bool,
        rows: &mut Rows<'_>,
        seq: u64,
    ) {
        if reset {
            self.end(rows, seq, false);
            self.prev.fill(f64::NAN);
            return;
        }
        let new_stamp = match self.open {
            Some((stamp, _)) => now.is_none() || stamp != now,
            None => true,
        };
        if new_stamp {
            self.end(rows, seq, true);
            if broke {
                self.prev.fill(f64::NAN);
            }
            self.open = Some((now, seq));
        }
    }

    /// The row `seq`, its slots pushed and its meta counting the wait: the
    /// `"left"` operators read the stamp before, and the row joins its own.
    pub(super) fn row(&mut self, values: &[f64], seq: u64, rows: &mut Rows<'_>) {
        let r = (seq - rows.first) as usize;
        for (j, def) in self.defs.iter().enumerate() {
            if def.closed == CrossClosed::Left {
                rows.values[r * rows.n_outputs + self.first_out + j] = self.prev[j];
            }
            let x = values[def.input];
            let w = def.weight.map_or(1.0, |i| values[i]);
            if usable(x) && usable(w) && w > 0.0 {
                self.acc[j].add(x, w);
            }
        }
    }

    /// The end of the input: the open stamp closes.
    pub(super) fn finish(&mut self, rows: &mut Rows<'_>, next_seq: u64) {
        self.end(rows, next_seq, true);
    }

    /// End the open stamp, whose rows are `[first, until)`: with `close`,
    /// its values go to its rows under `"right"` and become the stamp
    /// before under `"left"` (where it had a weighted row); without, its
    /// rows read null. Either way its rows stop waiting.
    fn end(&mut self, rows: &mut Rows<'_>, until: u64, close: bool) {
        let Some((_, first)) = self.open.take() else {
            return;
        };
        let waits = self.waits();
        for (j, def) in self.defs.iter().enumerate() {
            let acc = self.acc[j];
            let v = if close { acc.value(def) } else { f64::NAN };
            if def.closed == CrossClosed::Right {
                for seq in first..until {
                    let r = (seq - rows.first) as usize;
                    rows.values[r * rows.n_outputs + self.first_out + j] = v;
                }
            } else if close && acc.n > 0 {
                self.prev[j] = v;
            }
        }
        if waits {
            for seq in first..until {
                rows.meta[(seq - rows.first) as usize].open -= 1;
            }
        }
        self.acc.fill(Acc::default());
    }
}

#[cfg(test)]
mod tests {
    use super::super::Windows;
    use super::*;
    use rheon_core::ClockCfg;

    fn def(stat: CrossStat, bias: bool) -> CrossDef {
        CrossDef {
            stat,
            closed: CrossClosed::Right,
            input: 0,
            weight: Some(1),
            bias,
            cap: CROSS_STATE_BYTES,
        }
    }

    /// The weighted moments against the definition written out, on unequal
    /// weights: the mean `sum w x / sum w`, and the variance about it times
    /// `V1^2 / (V1^2 - V2)`, or not under `bias`.
    #[test]
    fn the_moments_are_the_weighted_definition() {
        let xs = [1.5, -2.0, 3.25, 0.5, 7.0];
        let ws = [0.5, 2.0, 1.0, 3.0, 0.25];
        let mut acc = Acc::default();
        for (x, w) in xs.iter().zip(&ws) {
            acc.add(*x, *w);
        }
        let v1: f64 = ws.iter().sum();
        let v2: f64 = ws.iter().map(|w| w * w).sum();
        let mean = xs.iter().zip(&ws).map(|(x, w)| x * w).sum::<f64>() / v1;
        let m2: f64 = xs
            .iter()
            .zip(&ws)
            .map(|(x, w)| w * (x - mean) * (x - mean))
            .sum();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-14 * b.abs().max(1.0);
        assert!(close(acc.value(&def(CrossStat::Mean, false)), mean));
        let unbiased = (m2 / v1 * v1 * v1 / (v1 * v1 - v2)).sqrt();
        assert!(close(acc.value(&def(CrossStat::Std, false)), unbiased));
        assert!(close(
            acc.value(&def(CrossStat::Std, true)),
            (m2 / v1).sqrt()
        ));
    }

    /// Fewer than two weighted rows is no cross-section; two is one.
    #[test]
    fn one_weighted_row_is_withheld_and_two_are_not() {
        let mut acc = Acc::default();
        acc.add(3.0, 1.0);
        assert!(acc.value(&def(CrossStat::Mean, false)).is_nan());
        assert!(acc.value(&def(CrossStat::Std, false)).is_nan());
        acc.add(5.0, 1.0);
        assert_eq!(acc.value(&def(CrossStat::Mean, false)), 4.0);
        assert_eq!(acc.value(&def(CrossStat::Std, false)), 2f64.sqrt());
    }

    /// The core with two cross operators and nothing else, fed stamps of
    /// several groups in a shuffled order: every row's value is its stamp's
    /// (or the stamp before's) moments written out, the core's own check
    /// holds after every row and after a state round trip, and the rows
    /// leave in input order once their stamp closes.
    #[test]
    fn the_core_reads_each_stamp_and_checks_after_every_row() {
        let defs = vec![
            CrossDef {
                stat: CrossStat::Mean,
                closed: CrossClosed::Right,
                input: 0,
                weight: None,
                bias: false,
                cap: CROSS_STATE_BYTES,
            },
            CrossDef {
                stat: CrossStat::Std,
                closed: CrossClosed::Left,
                input: 0,
                weight: Some(1),
                bias: true,
                cap: CROSS_STATE_BYTES,
            },
        ];
        let cfg = ClockCfg {
            gap_cap: 1e9,
            ..ClockCfg::default()
        };
        let mut core = Windows::with_cross(Vec::new(), Vec::new(), defs, cfg).unwrap();
        core.set_grouped(true);
        let mut s = 7u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); 2];
        for t in 0..40 {
            for g in 0..(1 + (t % 5)) {
                let (x, w) = (next() * 10.0, if next() < 0.2 { 0.0 } else { next() + 0.5 });
                let gi = core.group(Some(&format!("g{g}")));
                core.push(&super::super::RowIn {
                    group: gi,
                    clock: Some(ClockValue::F64(f64::from(t))),
                    session: None,
                    values: &[x, w],
                    accept: true,
                })
                .unwrap();
                rows.push((f64::from(t), x, w));
                core.check().unwrap();
                let bytes = rmp_serde::to_vec_named(&core).unwrap();
                let back: Windows = rmp_serde::from_slice(&bytes).unwrap();
                back.check().unwrap();
                assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), bytes);
                let e = core.drain();
                for (o, col) in e.values.iter().enumerate() {
                    out[o].extend(col);
                }
            }
        }
        let e = core.finish();
        for (o, col) in e.values.iter().enumerate() {
            out[o].extend(col);
        }
        assert_eq!(out[0].len(), rows.len());
        let stamp = |t: f64| -> Vec<(f64, f64)> {
            rows.iter()
                .filter(|r| r.0 == t)
                .map(|r| (r.1, r.2))
                .collect()
        };
        let mean = |v: &[(f64, f64)]| {
            if v.len() < 2 {
                return f64::NAN;
            }
            v.iter().map(|p| p.0).sum::<f64>() / v.len() as f64
        };
        let wstd = |v: &[(f64, f64)]| {
            let v: Vec<_> = v.iter().filter(|p| p.1 > 0.0).collect();
            if v.len() < 2 {
                return f64::NAN;
            }
            let v1: f64 = v.iter().map(|p| p.1).sum();
            let m = v.iter().map(|p| p.0 * p.1).sum::<f64>() / v1;
            (v.iter().map(|p| p.1 * (p.0 - m) * (p.0 - m)).sum::<f64>() / v1).sqrt()
        };
        let same = |a: f64, b: f64| {
            (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * b.abs().max(1.0)
        };
        let mut prev_weighted: Option<f64> = None;
        let mut last_t = f64::NAN;
        let mut before = None;
        for (i, r) in rows.iter().enumerate() {
            if r.0 != last_t {
                before = prev_weighted;
                if stamp(r.0).iter().any(|p| p.1 > 0.0) {
                    prev_weighted = Some(r.0);
                }
                last_t = r.0;
            }
            assert!(same(out[0][i], mean(&stamp(r.0))), "row {i}");
            let want = before.map_or(f64::NAN, |t| wstd(&stamp(t)));
            assert!(
                same(out[1][i], want),
                "row {i}: {} against {want}",
                out[1][i]
            );
        }
    }
}
