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

mod nb;
mod order;

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
    /// The row's own value's rank within the cross-section it reads.
    Rank,
    /// The quantile at the operator's level of the cross-section.
    Quantile,
    /// The value of the stamp's reference row, the one row its predicate
    /// picks: a read, not a reduction.
    Ref,
}

impl CrossStat {
    /// Whether the statistic reads the stamp's values in order: then the
    /// operator keeps them, not only their moments.
    pub fn orders(self) -> bool {
        matches!(self, CrossStat::Rank | CrossStat::Quantile)
    }
}

/// How tied values rank: Polars' `rank(method=)` but `"random"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankMethod {
    #[default]
    Average,
    Min,
    Max,
    Dense,
    Ordinal,
}

impl RankMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "average" => Some(Self::Average),
            "min" => Some(Self::Min),
            "max" => Some(Self::Max),
            "dense" => Some(Self::Dense),
            "ordinal" => Some(Self::Ordinal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Average => "average",
            Self::Min => "min",
            Self::Max => "max",
            Self::Dense => "dense",
            Self::Ordinal => "ordinal",
        }
    }
}

/// How a quantile between two values is read: Polars' `interpolation`.
/// `Equiprobable` is the inverted distribution function, the smallest value
/// whose cumulative share reaches the level, which alone has a weighted
/// form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interp {
    #[default]
    Nearest,
    Higher,
    Lower,
    Midpoint,
    Linear,
    Equiprobable,
}

impl Interp {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "nearest" => Some(Self::Nearest),
            "higher" => Some(Self::Higher),
            "lower" => Some(Self::Lower),
            "midpoint" => Some(Self::Midpoint),
            "linear" => Some(Self::Linear),
            "equiprobable" => Some(Self::Equiprobable),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Higher => "higher",
            Self::Lower => "lower",
            Self::Midpoint => "midpoint",
            Self::Linear => "linear",
            Self::Equiprobable => "equiprobable",
        }
    }
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
    /// A rank's tie rule and direction.
    #[serde(default)]
    pub rank: RankMethod,
    #[serde(default)]
    pub descending: bool,
    /// A quantile's level, in `[0, 1]`, and its interpolation.
    #[serde(default)]
    pub quantile: f64,
    #[serde(default)]
    pub interpolation: Interp,
    /// The index into a row's lists of the operator's neighbor keys: the
    /// row reads the rows of those groups alone. `None` reads every group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neighbors: Option<usize>,
    /// A reference read's predicate, an index into a row's values: a row
    /// where it is a number other than 0 is the reference row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pick: Option<usize>,
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
            CrossStat::Rank | CrossStat::Quantile | CrossStat::Ref => {
                unreachable!("read from the values")
            }
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
    /// Per operator that reads values in order: every row of the open stamp,
    /// its value and weight, in input order (NaN where it has none).
    #[serde(default)]
    open_rows: Vec<Vec<(f64, f64)>>,
    /// Per rank under `"left"`: the stamp before's members as a rank reads
    /// them, each distinct value ascending with the count at or below it.
    #[serde(default)]
    prev_rows: Vec<Vec<(f64, f64)>>,
    /// Per operator over neighbors: the open stamp's rows, each with its
    /// group, value, weight and (under `"right"`) neighbor keys.
    #[serde(default)]
    nb_open: Vec<Vec<nb::NbRow>>,
    /// Their bytes, as [`Cross::check_row`] counts them.
    #[serde(default)]
    nb_bytes: Vec<u64>,
    /// Per operator over neighbors under `"left"`: the stamp before's
    /// members, by group and then input order.
    #[serde(default)]
    nb_prev: Vec<Vec<nb::Member>>,
    /// Per reference read: the open stamp's reference row, its `seq` and
    /// value (NaN where it has none).
    #[serde(default)]
    refs: Vec<Option<(u64, f64)>>,
}

/// What each operator would hold with a row ([`Cross::check_cap`]).
pub(super) struct HeldBytes(Vec<u64>);

/// The held rows a cross operator writes into.
pub(super) struct Rows<'a> {
    pub first: u64,
    pub n_outputs: usize,
    pub values: &'a mut VecDeque<f64>,
    pub meta: &'a mut VecDeque<Meta>,
    /// The group a key names, as text, where the core has one.
    pub find: &'a dyn Fn(&str) -> Option<usize>,
}

impl Rows<'_> {
    fn slot(&self, seq: u64, out: usize) -> usize {
        (seq - self.first) as usize * self.n_outputs + out
    }
}

/// Whether a reference read's predicate picks a row: a number other than 0.
fn picks(p: f64) -> bool {
    usable(p) && p != 0.0
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
            open_rows: vec![Vec::new(); n],
            prev_rows: vec![Vec::new(); n],
            nb_open: vec![Vec::new(); n],
            nb_bytes: vec![0; n],
            nb_prev: vec![Vec::new(); n],
            refs: vec![None; n],
        }
    }

    /// The bytes each operator's state would hold with the row `seq` at
    /// `now`, `n_outputs` values a held row: under `"right"`, every row of
    /// the open stamp is held, its values and its count of open windows,
    /// until the stamp closes; under `"left"` nothing is held past the row.
    /// `Err` names the first operator the row would take past its cap.
    ///
    /// Also refused: a second reference row at one stamp.
    pub(super) fn check_row(
        &self,
        row: &super::RowIn<'_>,
        seq: u64,
        n_outputs: usize,
    ) -> Result<HeldBytes, super::Refusal> {
        let now = row.clock;
        let open_first = match self.open {
            Some((stamp, first)) if now.is_some() && stamp == now => Some(first),
            _ => None,
        };
        let same = open_first.is_some();
        let rows = open_first.map_or(1, |first| seq - first + 1);
        let per_row = 8 * n_outputs as u64 + std::mem::size_of::<Meta>() as u64;
        let mut held = Vec::with_capacity(self.defs.len());
        for (op, def) in self.defs.iter().enumerate() {
            if let (Some(p), true, Some((first, _))) = (def.pick, same, self.refs[op])
                && picks(row.values[p])
            {
                return Err(super::Refusal::CrossRefTwice { seq, op, first });
            }
            let mut bytes = match def.closed {
                CrossClosed::Right => rows * per_row,
                CrossClosed::Left => 0,
            };
            if let Some(l) = def.neighbors {
                // Each row of the open stamp, and under "right" its keys;
                // under "left" the stamp before's members.
                let keys = match def.closed {
                    CrossClosed::Right => nb::keys_bytes(row.lists[l]),
                    CrossClosed::Left => 0,
                };
                let open = if same { self.nb_bytes[op] } else { 0 };
                bytes += open + nb::ROW_BYTES + keys;
                if def.closed == CrossClosed::Left {
                    // At a new stamp, the open one becomes the stamp before
                    // where it has a member.
                    let members = self.nb_open[op].iter().filter(|r| r.member()).count();
                    let prev = if same || members == 0 {
                        self.nb_prev[op].len()
                    } else {
                        members
                    };
                    bytes += nb::MEMBER_BYTES * prev as u64;
                }
            } else if def.stat.orders() {
                // A rank or a quantile keeps the stamp's values and weights,
                // and under `"left"` the stamp before's too (a rank's
                // distinct values, at most its members).
                bytes += 16 * rows;
                if def.closed == CrossClosed::Left && def.stat == CrossStat::Rank {
                    let prev = if same {
                        self.prev_rows[op].len()
                    } else {
                        let members = self.open_rows[op]
                            .iter()
                            .filter(|&&(x, w)| !x.is_nan() && w > 0.0)
                            .count();
                        if members > 0 {
                            members
                        } else {
                            self.prev_rows[op].len()
                        }
                    };
                    bytes += 16 * prev as u64;
                }
            }
            if bytes > def.cap {
                return Err(super::Refusal::StateCap {
                    seq,
                    of: super::StateOf::Cross,
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
        let n = self.defs.len();
        self.acc.len() == n
            && self.prev.len() == n
            && self.open_rows.len() == n
            && self.prev_rows.len() == n
            && self.nb_open.len() == n
            && self.nb_bytes.len() == n
            && self.nb_prev.len() == n
            && self.refs.len() == n
            && self.defs.iter().enumerate().all(|(j, d)| {
                let ordered = d.stat.orders() && d.neighbors.is_none();
                (ordered || (self.open_rows[j].is_empty() && self.prev_rows[j].is_empty()))
                    && (d.neighbors.is_some()
                        || (self.nb_open[j].is_empty() && self.nb_prev[j].is_empty()))
                    && (d.pick.is_some() == (d.stat == CrossStat::Ref))
                    && (d.stat == CrossStat::Ref || self.refs[j].is_none())
            })
    }

    /// That each ordering operator keeps a value and a weight for every row
    /// of the open stamp, `[first, next_seq)`, and its stamp before as a
    /// table of distinct values ascending, each counting more rows than the
    /// last.
    pub(super) fn check_rows(&self, next_seq: u64) -> Result<(), String> {
        let open = self.open.map_or(0, |(_, first)| next_seq - first) as usize;
        for (j, def) in self.defs.iter().enumerate() {
            let kept = if def.neighbors.is_some() {
                Some(self.nb_open[j].len())
            } else if def.stat.orders() {
                Some(self.open_rows[j].len())
            } else {
                None
            };
            if let Some(kept) = kept
                && kept != open
            {
                return Err(format!(
                    "its cross-group operator {j} keeps {kept} rows of a stamp of {open}"
                ));
            }
            let first = next_seq - open as u64;
            let t = &self.prev_rows[j];
            let in_order = t.windows(2).all(|p| p[0].0 < p[1].0 && p[0].1 < p[1].1)
                && t.first().is_none_or(|f| f.1 >= 1.0)
                && self.nb_open[j]
                    .iter()
                    .enumerate()
                    .all(|(k, r)| r.seq == first + k as u64)
                && self.refs[j].is_none_or(|(s, _)| s >= first && s < next_seq)
                && self.nb_prev[j]
                    .windows(2)
                    .all(|p| (p[0].0, p[0].1) < (p[1].0, p[1].1));
            if !in_order {
                return Err(format!(
                    "its cross-group operator {j}'s rows are not the stamps' rows in order"
                ));
            }
        }
        Ok(())
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
            self.forget_prev();
            return;
        }
        let new_stamp = match self.open {
            Some((stamp, _)) => now.is_none() || stamp != now,
            None => true,
        };
        if new_stamp {
            self.end(rows, seq, true);
            if broke {
                self.forget_prev();
            }
            self.open = Some((now, seq));
        }
    }

    /// The row `seq`, its slots pushed and its meta counting the wait: the
    /// `"left"` operators read the stamp before, and the row joins its own.
    pub(super) fn row(&mut self, row: &super::RowIn<'_>, seq: u64, rows: &mut Rows<'_>) {
        let values = row.values;
        for (j, def) in self.defs.iter().enumerate() {
            let slot = rows.slot(seq, self.first_out + j);
            let x = values[def.input];
            let w = def.weight.map_or(1.0, |i| values[i]);
            let (x, w) = (
                if usable(x) { x } else { f64::NAN },
                if usable(w) && w > 0.0 { w } else { 0.0 },
            );
            if let Some(p) = def.pick {
                if def.closed == CrossClosed::Left {
                    rows.values[slot] = self.prev[j];
                }
                if picks(values[p]) {
                    self.refs[j] = Some((seq, x));
                }
                continue;
            }
            if let Some(l) = def.neighbors {
                let keys = nb::dedup(row.lists[l]);
                if def.closed == CrossClosed::Left {
                    let groups = nb::resolve(&keys, rows.find);
                    rows.values[slot] =
                        nb::value(&nb::gather(&self.nb_prev[j], &groups), x, None, def);
                }
                let r = nb::NbRow {
                    seq,
                    group: row.group,
                    x,
                    w,
                    keys: if def.closed == CrossClosed::Right {
                        keys
                    } else {
                        Vec::new()
                    },
                };
                self.nb_bytes[j] += nb::ROW_BYTES + nb::owned_keys_bytes(&r.keys);
                self.nb_open[j].push(r);
                continue;
            }
            if def.closed == CrossClosed::Left {
                rows.values[slot] = match def.stat {
                    CrossStat::Rank => order::rank_of(&self.prev_rows[j], x, def),
                    _ => self.prev[j],
                };
            }
            if def.stat.orders() {
                self.open_rows[j].push((x, w));
            } else if !x.is_nan() && w > 0.0 {
                self.acc[j].add(x, w);
            }
        }
    }

    /// A clock break or a reset: no stamp before the next.
    fn forget_prev(&mut self) {
        self.prev.fill(f64::NAN);
        for p in &mut self.prev_rows {
            p.clear();
        }
        for p in &mut self.nb_prev {
            p.clear();
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
            let out = self.first_out + j;
            if def.pick.is_some() {
                let read = self.refs[j].take().map_or(f64::NAN, |r| r.1);
                if def.closed == CrossClosed::Right {
                    for seq in first..until {
                        let at = rows.slot(seq, out);
                        rows.values[at] = if close { read } else { f64::NAN };
                    }
                } else if close && !read.is_nan() {
                    self.prev[j] = read;
                }
                continue;
            }
            if def.neighbors.is_some() {
                let stamp = std::mem::take(&mut self.nb_open[j]);
                self.nb_bytes[j] = 0;
                let members = nb::members_of(&stamp);
                if def.closed == CrossClosed::Right {
                    let read = if close {
                        nb::read(&stamp, &members, def, rows.find)
                    } else {
                        vec![f64::NAN; stamp.len()]
                    };
                    for (k, seq) in (first..until).enumerate() {
                        let at = rows.slot(seq, out);
                        rows.values[at] = read[k];
                    }
                } else if close && !members.is_empty() {
                    self.nb_prev[j] = members;
                }
                continue;
            }
            if def.stat.orders() {
                let stamp = std::mem::take(&mut self.open_rows[j]);
                let members = order::members(&stamp);
                if def.closed == CrossClosed::Right {
                    let read = order::read(&stamp, &members, def);
                    for (k, seq) in (first..until).enumerate() {
                        let r = (seq - rows.first) as usize;
                        rows.values[r * rows.n_outputs + self.first_out + j] =
                            if close { read[k] } else { f64::NAN };
                    }
                } else if close && !members.is_empty() {
                    if def.stat == CrossStat::Rank {
                        self.prev_rows[j] = order::table(&members);
                    } else {
                        self.prev[j] = order::quantile_of(&members, def);
                    }
                }
                continue;
            }
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
            rank: RankMethod::Average,
            descending: false,
            quantile: 0.5,
            interpolation: Interp::Nearest,
            neighbors: None,
            pick: None,
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
                rank: RankMethod::Average,
                descending: false,
                quantile: 0.5,
                interpolation: Interp::Nearest,
                neighbors: None,
                pick: None,
            },
            CrossDef {
                stat: CrossStat::Std,
                closed: CrossClosed::Left,
                input: 0,
                weight: Some(1),
                bias: true,
                cap: CROSS_STATE_BYTES,
                rank: RankMethod::Average,
                descending: false,
                quantile: 0.5,
                interpolation: Interp::Nearest,
                neighbors: None,
                pick: None,
            },
            CrossDef {
                stat: CrossStat::Rank,
                closed: CrossClosed::Right,
                input: 0,
                weight: None,
                bias: false,
                cap: CROSS_STATE_BYTES,
                rank: RankMethod::Ordinal,
                descending: true,
                quantile: 0.5,
                interpolation: Interp::Nearest,
                neighbors: None,
                pick: None,
            },
            CrossDef {
                stat: CrossStat::Quantile,
                closed: CrossClosed::Left,
                input: 0,
                weight: Some(1),
                bias: false,
                cap: CROSS_STATE_BYTES,
                rank: RankMethod::Average,
                descending: false,
                quantile: 0.3,
                interpolation: Interp::Equiprobable,
                neighbors: None,
                pick: None,
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
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); 4];
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
                    lists: &[],
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
        // The values are continuous, so no two tie: a descending ordinal
        // rank counts the larger values.
        let rank = |x: f64, v: &[(f64, f64)]| {
            if v.len() < 2 {
                return f64::NAN;
            }
            1.0 + v.iter().filter(|p| p.0 > x).count() as f64
        };
        let wq = |v: &[(f64, f64)]| {
            let mut v: Vec<_> = v.iter().copied().filter(|p| p.1 > 0.0).collect();
            if v.len() < 2 {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let total: f64 = v.iter().map(|p| p.1).sum();
            let mut cum = 0.0;
            for p in &v {
                cum += p.1;
                if cum / total >= 0.3 {
                    return p.0;
                }
            }
            v[v.len() - 1].0
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
            assert_eq!(
                out[2][i].to_bits(),
                rank(r.1, &stamp(r.0)).to_bits(),
                "row {i}"
            );
            let want = before.map_or(f64::NAN, |t| wq(&stamp(t)));
            assert_eq!(out[3][i].to_bits(), want.to_bits(), "row {i}");
        }
    }

    /// A reference read and a mean over neighbors, under both closed
    /// forms: the core checks after every row, round-trips its state to the
    /// byte, and reads the definitions.
    #[test]
    fn the_relations_read_their_rows_and_check_after_every_row() {
        let base = CrossDef {
            weight: None,
            ..def(CrossStat::Mean, false)
        };
        let defs = vec![
            CrossDef {
                stat: CrossStat::Ref,
                pick: Some(1),
                ..base.clone()
            },
            CrossDef {
                stat: CrossStat::Ref,
                pick: Some(1),
                closed: CrossClosed::Left,
                ..base.clone()
            },
            CrossDef {
                neighbors: Some(0),
                ..base.clone()
            },
            CrossDef {
                neighbors: Some(0),
                closed: CrossClosed::Left,
                ..base
            },
        ];
        let cfg = ClockCfg {
            gap_cap: 1e9,
            ..ClockCfg::default()
        };
        let mut core = Windows::with_cross(Vec::new(), Vec::new(), defs, cfg).unwrap();
        core.set_grouped(true);
        let key = |g: usize| format!("g{g}");
        // Group g reads groups g + 1 and g + 2, of five.
        let lists: Vec<Vec<Option<String>>> = (0..5)
            .map(|g| vec![Some(key((g + 1) % 5)), Some(key((g + 2) % 5))])
            .collect();
        let mut rows: Vec<(u32, usize, f64)> = Vec::new();
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); 4];
        for t in 0..30u32 {
            for g in (0..5).filter(|g| !(t as usize + g).is_multiple_of(4)) {
                let x = f64::from(t * 7 % 11) + g as f64;
                let gi = core.group(Some(&key(g)));
                let list = [Some(lists[g].as_slice())];
                core.push(&super::super::RowIn {
                    group: gi,
                    clock: Some(ClockValue::F64(f64::from(t))),
                    session: None,
                    values: &[x, if g == 0 { 1.0 } else { 0.0 }],
                    accept: true,
                    lists: &list,
                })
                .unwrap();
                rows.push((t, g, x));
                core.check().unwrap();
                let bytes = rmp_serde::to_vec_named(&core).unwrap();
                let back: Windows = rmp_serde::from_slice(&bytes).unwrap();
                back.check().unwrap();
                assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), bytes);
                for (o, col) in core.drain().values.iter().enumerate() {
                    out[o].extend(col);
                }
            }
        }
        for (o, col) in core.finish().values.iter().enumerate() {
            out[o].extend(col);
        }
        let at = |t: u32, g: usize| rows.iter().find(|r| r.0 == t && r.1 == g).map(|r| r.2);
        let mean = |t: u32, g: usize| {
            let v: Vec<f64> = [(g + 1) % 5, (g + 2) % 5]
                .iter()
                .filter_map(|&h| at(t, h))
                .collect();
            if v.len() < 2 {
                f64::NAN
            } else {
                v.iter().sum::<f64>() / 2.0
            }
        };
        let same = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12;
        let mut leader_before = f64::NAN;
        let mut stamp_before: Option<u32> = None;
        let mut last_t = u32::MAX;
        for (i, &(t, g, _)) in rows.iter().enumerate() {
            if t != last_t {
                if last_t != u32::MAX {
                    stamp_before = Some(last_t);
                    if let Some(v) = at(last_t, 0) {
                        leader_before = v;
                    }
                }
                last_t = t;
            }
            assert!(
                same(out[0][i], at(t, 0).unwrap_or(f64::NAN)),
                "row {i}: {} at {t}",
                out[0][i]
            );
            assert!(same(out[1][i], leader_before), "row {i}");
            assert!(same(out[2][i], mean(t, g)), "row {i}");
            let want = stamp_before.map_or(f64::NAN, |b| mean(b, g));
            assert!(
                same(out[3][i], want),
                "row {i}: {} against {want}",
                out[3][i]
            );
        }
    }
}
