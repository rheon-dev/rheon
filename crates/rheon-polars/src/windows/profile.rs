//! A periodic profile on the clock (`ewm_profile`): an exponentially
//! weighted statistic of a column by the row's phase in a period, read at
//! each row's own phase.
//!
//! **The phase.** A row's phase is its place in the period, read from its
//! own clock value alone, so it is the same however the stream is cut into
//! chunks:
//!
//! - a numeric clock: `(t − offset) mod P`, in integers where the clock,
//!   `P` and the offset are integers;
//! - a temporal clock (a `Datetime`, a `Date` or a `Duration`, in
//!   nanoseconds): `(t − origin − offset) mod P`, `origin` the epoch, or the
//!   Monday after it (1970-01-05) where `P` is a whole number of weeks: the
//!   phase Polars' `t − t.dt.truncate(P)` gives;
//! - a zoned `Datetime`: the same on the **wall clock** in the column's own
//!   zone, `t` plus the zone's offset at `t`, as Polars' `dt` namespace reads
//!   a zoned value. On the day the clocks go forward a day has 23 hours, and
//!   a row at 05:00 is at phase 5h, not the 4h elapsed since midnight;
//! - no clock column: the group's rows of weight above 0, counted from 0,
//!   mod `P`. A row of weight 0 counts in no clock, and is read where the
//!   clock stands, at the last such row.
//!
//! The phase falls in one of `B` equal buckets, `floor(B · phase / P)`,
//! in integers on a temporal or integer clock.
//!
//! **The statistic** of each bucket is the weighted mean, or the weighted
//! variance about it in Welford's form, of the bucket's rows, each weighed
//! by its weight `w` times `0.5^(age / half_life)`, its age taken on the
//! clock from the row the profile is read at. A bucket decays by the clock
//! whether or not a row falls in it, so one long unseen weighs little:
//!
//! ```text
//! V1' = d V1 + w,  V2' = d² V2 + w²,  m' = m + (w / V1')(x − m),
//! M2' = d M2 + w (x − m)(x − m'),     d = 0.5^((t − t_b) / half_life)
//! mean = m,  var = M2 / V1 · V1² / (V1² − V2)  (bias: M2 / V1),  std = √var
//! ```
//!
//! A row reads its own bucket. Under `closed = "right"` (and `"both"`) the
//! rows at the row's own stamp are in the profile it reads, so every row of
//! a stamp reads the same value, known once the stamp closes; under
//! `"left"` (and `"none"`) they are not, and the value is known as the
//! stamp arrives.
//!
//! **Withheld** (null): a bucket holding fewer than `min_samples` rows with
//! a value and a weight above 0, as Polars' `rolling_*_by` and the window
//! operators count them, so a bucket no row has reached is null; a variance
//! of a bucket one row's weight fills (`V1² = V2`). A row of weight 0, or
//! with no value, reads its bucket and adds nothing.

use rheon_core::{ClockValue, seconds_of_ns};
use serde::{Deserialize, Serialize};

use std::collections::VecDeque;

use super::{Closed, Direction, Group, Held, KernelDef, Refusal};
use crate::stream::usable;
use crate::window_quantile::QSide;

mod harm;

/// A nanosecond day and week.
const DAY_NS: i64 = 86_400_000_000_000;
const WEEK_NS: i64 = 7 * DAY_NS;
/// The first Monday after the epoch, 1970-01-05, where Polars' `truncate`
/// starts a whole number of weeks.
const MONDAY_NS: i64 = 4 * DAY_NS;

/// The bytes one bucket of one operator's state is counted at: its seven
/// numbers (its rows, `V1`, `V2`, `m`, `M2` and the time it was last moved,
/// which is two words as a tagged clock value).
pub const BUCKET_BYTES: u64 = 56;

/// An operator's `max_state_bytes` where the call sets none: 256 KB.
pub const PROFILE_STATE_BYTES: u64 = 256 * 1024;

/// The most `max_state_bytes` may be set to: 64 MiB.
pub const PROFILE_STATE_BYTES_CEILING: u64 = 64 << 20;

/// What the profile reads of each bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileStat {
    Mean,
    Var,
    Std,
    /// The quantile at [`ProfileDef::quantile`], from `ewm_quantile`'s
    /// sketch of each bucket's values.
    Quantile,
}

impl ProfileStat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mean" => Some(Self::Mean),
            "var" => Some(Self::Var),
            "std" => Some(Self::Std),
            "quantile" => Some(Self::Quantile),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Var => "var",
            Self::Std => "std",
            Self::Quantile => "quantile",
        }
    }
}

/// A period or an offset: nanoseconds on a temporal clock, else a number
/// of the clock's units (or of rows without a clock).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseSpan {
    Ns(i64),
    Units(f64),
}

/// One profile operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileDef {
    pub stat: ProfileStat,
    pub buckets: u32,
    pub period: PhaseSpan,
    pub offset: PhaseSpan,
    /// The clock column's time zone, whose wall clock the phase is read on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone: Option<String>,
    /// In the clock's units: seconds on a temporal clock, rows without one.
    pub half_life: f64,
    pub closed: Closed,
    pub input: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<usize>,
    /// The rows with a value a bucket needs before it is read, at least 1.
    pub min_samples: u32,
    pub bias: bool,
    /// The most bytes one group's state of this operator may hold.
    pub cap: u64,
    /// The window: rows further than this from the row read leave its
    /// bucket. `None` keeps every row, decayed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<PhaseSpan>,
    /// A quantile's level, in `(0, 1)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantile: Option<f64>,
    /// The harmonic form's harmonics `K`, where `buckets` is 0: a Fourier
    /// series of `2K + 1` terms in the phase.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub harmonics: u32,
}

fn is_zero_u32(k: &u32) -> bool {
    *k == 0
}

impl ProfileDef {
    /// The bytes one group's state of this operator holds.
    pub fn bytes(&self) -> u64 {
        if self.harmonics > 0 {
            harm::gram_bytes(self.terms())
        } else {
            u64::from(self.buckets) * BUCKET_BYTES
        }
    }

    /// The harmonic form's terms, `2K + 1`.
    fn terms(&self) -> usize {
        2 * self.harmonics as usize + 1
    }

    /// Whether the buckets keep their rows or a sketch: under a window, or
    /// for a quantile.
    fn keeps_rows(&self) -> bool {
        self.harmonics == 0 && (self.window.is_some() || self.stat == ProfileStat::Quantile)
    }

    /// The bytes a row in a bucket's window is counted at: its clock, time,
    /// value and weight, its share of the moments a front row keeps, and a
    /// quantile's three weights a front row keeps.
    fn row_bytes(&self) -> u64 {
        if self.harmonics > 0 {
            // Its clock, time, phase, value and weight, and the Gram and
            // moments a front row keeps.
            return if self.window.is_some() {
                40 + harm::gram_bytes(self.terms())
            } else {
                0
            };
        }
        let q = if self.stat == ProfileStat::Quantile {
            24
        } else {
            0
        };
        if self.window.is_some() {
            ROW_BYTES + q
        } else {
            0
        }
    }

    /// The kernel a quantile's sketches are read with.
    fn kernel(&self) -> KernelDef {
        KernelDef {
            direction: Direction::Backward,
            half_life: self.half_life,
            window_size: self.window.map(span_value),
            window_ns: match self.window {
                Some(PhaseSpan::Ns(n)) => Some(n),
                _ => None,
            },
            closed: self.closed,
        }
    }

    /// Whether a row reads the rows of its own stamp: then it waits for
    /// the stamp to close.
    pub fn waits(&self) -> bool {
        self.closed.near(Direction::Backward)
    }

    /// A definition this core can run.
    ///
    /// # Errors
    ///
    /// A shape no caller's checked call could give.
    pub fn check(&self) -> Result<(), String> {
        let positive = |s: PhaseSpan| match s {
            PhaseSpan::Ns(n) => n > 0,
            PhaseSpan::Units(u) => u.is_finite() && u > 0.0,
        };
        let same_kind = matches!(
            (self.period, self.offset),
            (PhaseSpan::Ns(_), PhaseSpan::Ns(_)) | (PhaseSpan::Units(_), PhaseSpan::Units(_))
        );
        let offset_finite = match self.offset {
            PhaseSpan::Ns(_) => true,
            PhaseSpan::Units(u) => u.is_finite(),
        };
        if (self.buckets == 0) == (self.harmonics == 0)
            || (self.harmonics > 0 && (self.stat == ProfileStat::Quantile || self.bias))
            || !positive(self.period)
            || !same_kind
            || !offset_finite
            || self.half_life.is_nan()
            || self.half_life <= 0.0
            || self.min_samples == 0
            || self.bytes() > self.cap
            || self.window.is_some_and(|w| !positive(w))
            || (self.stat == ProfileStat::Quantile)
                != self.quantile.is_some_and(|q| q > 0.0 && q < 1.0)
        {
            return Err(format!("profile operator {self:?} cannot run"));
        }
        if let Some(z) = &self.zone {
            zone_of(z)?;
        }
        Ok(())
    }
}

/// The zone of a clock column by its canonical name: what
/// [`ProfileDef::zone`] holds. Polars writes a column's zone in the form
/// chrono-tz reads (a fixed offset as `Etc/GMT-1`).
pub fn canonical_zone(name: &str) -> Result<String, String> {
    zone_of(name).map(|tz| tz.name().to_string())
}

/// A canonical zone name's rules.
fn zone_of(name: &str) -> Result<chrono_tz::Tz, String> {
    name.parse::<chrono_tz::Tz>()
        .map_err(|_| format!("the time zone {name:?} is not a canonical one"))
}

/// The offset of `zone`'s wall clock from UTC at the instant `ns`.
fn zone_offset_ns(zone: &chrono_tz::Tz, ns: i64) -> i64 {
    use chrono::{Offset, TimeZone};
    let at = chrono::DateTime::from_timestamp_nanos(ns).naive_utc();
    i64::from(zone.offset_from_utc_datetime(&at).fix().local_minus_utc()) * 1_000_000_000
}

/// A bucket's weighted moments and the time they were last moved to.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(super) struct Bucket {
    /// The rows with a value and a weight above 0 it holds.
    n: u64,
    v1: f64,
    v2: f64,
    m: f64,
    m2: f64,
    at: ClockValue,
}

impl Default for Bucket {
    fn default() -> Self {
        Self {
            n: 0,
            v1: 0.0,
            v2: 0.0,
            m: 0.0,
            m2: 0.0,
            at: ClockValue::I64(0),
        }
    }
}

/// One group's state of one profile operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct ProfState {
    buckets: Vec<Bucket>,
    /// The group's rows of weight above 0 since its last restart: the row
    /// clock's time, read only without a clock column.
    rows: i64,
    /// The bucket and the time the current stamp reads, and under `"left"`
    /// its value, read before the stamp's rows joined.
    at: Option<(u32, ClockValue)>,
    left: f64,
    /// Under a window or for a quantile, the buckets that keep their rows
    /// or a sketch, in place of `buckets`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    win: Vec<WinBucket>,
    /// The clock of the first row since the restart: the time a window's
    /// rows and a sketch are measured from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    origin: Option<ClockValue>,
    /// The bytes the state is counted at ([`ProfileDef::bytes`] and each
    /// kept row and sketch bucket).
    #[serde(default, skip_serializing_if = "is_zero")]
    bytes: u64,
    /// The harmonic form's fit, and the phase the current stamp reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    harm: Option<harm::Harm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    phi: Option<f64>,
}

fn is_zero(b: &u64) -> bool {
    *b == 0
}

impl ProfState {
    fn new(def: &ProfileDef) -> Self {
        let keeps = def.keeps_rows();
        let mut st = Self {
            buckets: if keeps || def.harmonics > 0 {
                Vec::new()
            } else {
                vec![Bucket::default(); def.buckets as usize]
            },
            rows: 0,
            at: None,
            left: f64::NAN,
            win: if keeps {
                (0..def.buckets).map(|_| WinBucket::new(def)).collect()
            } else {
                Vec::new()
            },
            origin: None,
            bytes: 0,
            harm: (def.harmonics > 0).then(|| harm::Harm::new(def)),
            phi: None,
        };
        st.bytes = def.bytes() + st.win.iter().map(|b| b.bytes(def)).sum::<u64>();
        if let Some(h) = &st.harm {
            st.bytes += h.bytes(def);
        }
        st
    }
}

/// The bytes a kept row is counted at, a quantile's three weights aside:
/// its clock, time, value and weight (32), and the moments a front row
/// keeps of the rows newer than it (48).
const ROW_BYTES: u64 = 80;

/// A window's row: its clock, its time from the origin, its value and its
/// weight.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct WRow {
    at: ClockValue,
    tau: f64,
    x: f64,
    w: f64,
}

/// Weighted moments seen from `tau`: `n` rows of weight `V1`, `V2`, the
/// mean `m` and `M2`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Mom {
    n: f64,
    v1: f64,
    v2: f64,
    m: f64,
    m2: f64,
    tau: f64,
}

impl Mom {
    const EMPTY: Mom = Mom {
        n: 0.0,
        v1: 0.0,
        v2: 0.0,
        m: 0.0,
        m2: 0.0,
        tau: f64::NAN,
    };

    /// Seen from `tau`, at or after its own.
    fn at(self, k: &KernelDef, tau: f64) -> Mom {
        if self.n == 0.0 {
            return Mom { tau, ..Mom::EMPTY };
        }
        let d = k.discount(tau - self.tau);
        Mom {
            v1: self.v1 * d,
            v2: self.v2 * d * d,
            m2: self.m2 * d,
            tau,
            ..self
        }
    }

    /// Add a row of value `x` and weight `w`, as seen from `self.tau`.
    fn add(&mut self, x: f64, w: f64) {
        self.n += 1.0;
        self.v1 += w;
        self.v2 += w * w;
        let dx = x - self.m;
        self.m += (w / self.v1) * dx;
        self.m2 += w * dx * (x - self.m);
    }

    /// Chan, Golub and LeVeque's pairwise merge of two seen from one time.
    fn merge(self, o: Mom) -> Mom {
        if o.n == 0.0 {
            return self;
        }
        if self.n == 0.0 {
            return o;
        }
        let v1 = self.v1 + o.v1;
        let d = o.m - self.m;
        Mom {
            n: self.n + o.n,
            v1,
            v2: self.v2 + o.v2,
            m: self.m + d * (o.v1 / v1),
            m2: self.m2 + o.m2 + d * d * self.v1 * o.v1 / v1,
            tau: self.tau,
        }
    }
}

/// A bucket that keeps its window's rows, in two stacks so a row leaves
/// without a subtraction, or a quantile's sketch.
///
/// The rows are oldest first. The first `front` of them are the front
/// stack: each keeps the moments of itself and every front row newer than
/// it, seen from the newest front row, so the oldest leaves by dropping
/// its own. The rest are the back stack, whose moments are kept as they
/// join. When the front empties and a row must leave, the back turns over
/// into it. A quantile's sketches are `ewm_quantile`'s, over the same two
/// stacks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct WinBucket {
    rows: VecDeque<WRow>,
    front: usize,
    suffix: VecDeque<Mom>,
    back: Mom,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    q: Option<QSide>,
    /// With no window, the rows a quantile's sketch has taken: what
    /// `min_samples` counts, where under a window it counts `rows`.
    #[serde(default, skip_serializing_if = "is_zero")]
    n: u64,
}

impl WinBucket {
    fn new(def: &ProfileDef) -> Self {
        Self {
            rows: VecDeque::new(),
            front: 0,
            suffix: VecDeque::new(),
            back: Mom::EMPTY,
            q: (def.stat == ProfileStat::Quantile).then(QSide::default),
            n: 0,
        }
    }

    /// The rows with a value and a weight it holds: inside the window, or
    /// every one taken where there is none.
    fn count(&self, def: &ProfileDef) -> u64 {
        if def.window.is_some() {
            self.rows.len() as u64
        } else {
            self.n
        }
    }

    /// The bytes it is counted at, beyond [`BUCKET_BYTES`].
    fn bytes(&self, def: &ProfileDef) -> u64 {
        self.rows.len() as u64 * def.row_bytes()
            + self.q.as_ref().map_or(0, |q| 8 * q.buckets() as u64)
    }

    /// Whether its stacks are the rows it holds.
    fn fits(&self, def: &ProfileDef) -> bool {
        let front_x = self.rows.iter().take(self.front).rev().map(|r| r.x);
        self.front <= self.rows.len()
            && (def.window.is_some() || self.rows.is_empty())
            && self.suffix.len()
                == if def.stat == ProfileStat::Quantile {
                    0
                } else {
                    self.front
                }
            && self.q.is_some() == (def.stat == ProfileStat::Quantile)
            && self.q.as_ref().is_none_or(|q| {
                q.has_shape(
                    if def.window.is_some() {
                        front_x.collect::<Vec<_>>()
                    } else {
                        Vec::new()
                    }
                    .into_iter(),
                )
            })
            && self.rows.iter().all(|r| r.w > 0.0 && r.x.is_finite())
    }

    /// Drop the rows outside the window at `now`.
    fn evict(&mut self, def: &ProfileDef, k: &KernelDef, now: ClockValue) {
        let Some(w) = def.window else {
            return;
        };
        while self
            .rows
            .front()
            .is_some_and(|r| outside(w, def.closed, r.at, now))
        {
            if self.front == 0 {
                self.turn_over(def, k);
            }
            let r = self.rows.pop_front().expect("a row to leave");
            self.suffix.pop_front();
            if let Some(q) = &mut self.q {
                q.pop(self.front - 1, r.x);
            }
            self.front -= 1;
        }
    }

    /// The back stack's rows become the front stack.
    fn turn_over(&mut self, def: &ProfileDef, k: &KernelDef) {
        let n = self.rows.len();
        self.front = n;
        if def.stat == ProfileStat::Quantile {
            let tau: Vec<f64> = self.rows.iter().map(|r| r.tau).collect();
            let q = self.q.as_mut().expect("a quantile's sketch");
            q.turn_over_weighted(k, &tau, |i| self.rows[i].x, |i| self.rows[i].w);
        } else {
            self.suffix.clear();
            if let Some(newest) = self.rows.back() {
                let mut acc = Mom {
                    tau: newest.tau,
                    ..Mom::EMPTY
                };
                let mut out = vec![Mom::EMPTY; n];
                for i in (0..n).rev() {
                    let r = self.rows[i];
                    acc.add(r.x, r.w * k.discount(newest.tau - r.tau));
                    out[i] = acc;
                }
                self.suffix = out.into();
            }
        }
        self.back = Mom::EMPTY;
    }

    /// A row joins the back stack.
    fn push(&mut self, def: &ProfileDef, k: &KernelDef, r: WRow) {
        if let Some(q) = &mut self.q {
            q.push_weighted(k, r.tau, r.x, r.w);
        } else {
            let mut b = self.back.at(k, r.tau);
            b.add(r.x, r.w);
            self.back = b;
        }
        if def.window.is_some() {
            self.rows.push_back(r);
        } else {
            self.n += 1;
        }
    }

    /// The moments of the rows it holds, seen from `tau`.
    fn moments(&self, k: &KernelDef, tau: f64) -> Mom {
        let front = self.suffix.front().map_or(Mom::EMPTY, |m| m.at(k, tau));
        front.merge(self.back.at(k, tau))
    }
}

/// Whether a row at `at` is outside a window `w` read at `now`: further
/// than `w` back, or exactly `w` under a `closed` that leaves the far end
/// out.
fn outside(w: PhaseSpan, closed: Closed, at: ClockValue, now: ClockValue) -> bool {
    let far = closed.far(Direction::Backward);
    let cmp = match (w, at, now) {
        (PhaseSpan::Ns(w), ClockValue::Ns(a), ClockValue::Ns(b)) => {
            (i128::from(b) - i128::from(a)).cmp(&i128::from(w))
        }
        (PhaseSpan::Units(w), ClockValue::I64(a), ClockValue::I64(b)) if w.fract() == 0.0 => {
            (i128::from(b) - i128::from(a)).cmp(&(w as i128))
        }
        (PhaseSpan::Units(w), a, b) => (b.seconds() - a.seconds())
            .partial_cmp(&w)
            .unwrap_or(std::cmp::Ordering::Equal),
        _ => std::cmp::Ordering::Less,
    };
    match cmp {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Equal => !far,
        std::cmp::Ordering::Less => false,
    }
}

/// A span's value in the clock's units: seconds for nanoseconds.
fn span_value(s: PhaseSpan) -> f64 {
    match s {
        PhaseSpan::Ns(n) => seconds_of_ns(i128::from(n)),
        PhaseSpan::Units(u) => u,
    }
}

/// Whether a group's profile states fit the definitions: none (a group
/// that has not started since its last restart) or one each, of its
/// buckets.
pub(super) fn fits(defs: &[ProfileDef], states: &[ProfState]) -> bool {
    states.is_empty()
        || (states.len() == defs.len()
            && states.iter().zip(defs).all(|(s, d)| {
                let n = d.buckets as usize;
                let shaped = if d.harmonics > 0 {
                    s.buckets.is_empty()
                        && s.win.is_empty()
                        && s.harm
                            .as_ref()
                            .is_some_and(|h| h.fits(d) && s.bytes == d.bytes() + h.bytes(d))
                } else if d.keeps_rows() {
                    s.buckets.is_empty()
                        && s.win.len() == n
                        && s.win.iter().all(|b| b.fits(d))
                        && s.bytes == d.bytes() + s.win.iter().map(|b| b.bytes(d)).sum::<u64>()
                } else {
                    s.buckets.len() == n && s.win.is_empty()
                };
                shaped && s.rows >= 0
            }))
}

/// The clock between `from` and `to`, in the clock's units: seconds on a
/// temporal clock.
fn age(from: ClockValue, to: ClockValue) -> f64 {
    match (from, to) {
        (ClockValue::Ns(a), ClockValue::Ns(b)) => seconds_of_ns(i128::from(b) - i128::from(a)),
        (ClockValue::I64(a), ClockValue::I64(b)) => (i128::from(b) - i128::from(a)) as f64,
        (a, b) => b.seconds() - a.seconds(),
    }
}

/// `0.5^(age / half_life)`; 1 at no age or no decay.
fn decay(def: &ProfileDef, from: ClockValue, to: ClockValue) -> f64 {
    let a = age(from, to);
    if a <= 0.0 || def.half_life.is_infinite() {
        1.0
    } else {
        (-a / def.half_life).exp2()
    }
}

/// The bucket of a phase `ph` in `[0, p)`.
fn bucket_of_int(ph: i128, p: i128, b: u32) -> u32 {
    u32::try_from(ph * i128::from(b) / p).map_or(b - 1, |k| k.min(b - 1))
}

fn bucket_of_f(ph: f64, p: f64, b: u32) -> u32 {
    let k = (ph / p * f64::from(b)).floor();
    if k.is_finite() && k >= 0.0 {
        (k as u32).min(b - 1)
    } else {
        0
    }
}

/// A row's phase in the period, `(phase, period)`: in integers where the
/// clock, the period and the offset are, else in doubles.
enum Phase {
    Int(i128, i128),
    Float(f64, f64),
}

/// The phase of a clock value (`None`: the row clock's `rows`).
fn phase(def: &ProfileDef, clock: Option<ClockValue>, rows: i64) -> Phase {
    let units = |t: f64| match (def.period, def.offset) {
        (PhaseSpan::Units(p), PhaseSpan::Units(o)) => Phase::Float((t - o).rem_euclid(p), p),
        _ => Phase::Float(0.0, 1.0),
    };
    let int = |t: i64| match (def.period, def.offset) {
        (PhaseSpan::Units(p), PhaseSpan::Units(o))
            if p.fract() == 0.0 && o.fract() == 0.0 && p.abs() < 9e18 && o.abs() < 9e18 =>
        {
            let (p, o) = (p as i128, o as i128);
            Phase::Int((i128::from(t) - o).rem_euclid(p), p)
        }
        _ => units(t as f64),
    };
    match clock {
        None => int(rows),
        Some(ClockValue::I64(t)) => int(t),
        Some(ClockValue::F64(t)) => units(t),
        Some(ClockValue::Ns(t)) => {
            let (PhaseSpan::Ns(p), PhaseSpan::Ns(o)) = (def.period, def.offset) else {
                return Phase::Float(0.0, 1.0);
            };
            let local = match def.zone.as_deref().map(zone_of) {
                Some(Ok(z)) => i128::from(t) + i128::from(zone_offset_ns(&z, t)),
                _ => i128::from(t),
            };
            let origin = if p % WEEK_NS == 0 { MONDAY_NS } else { 0 };
            let ph = (local - i128::from(origin) - i128::from(o)).rem_euclid(i128::from(p));
            Phase::Int(ph, i128::from(p))
        }
    }
}

/// The bucket of a clock value (`None`: the row clock's `rows`).
pub(super) fn bucket(def: &ProfileDef, clock: Option<ClockValue>, rows: i64) -> u32 {
    match phase(def, clock, rows) {
        Phase::Int(ph, p) => bucket_of_int(ph, p, def.buckets),
        Phase::Float(ph, p) => bucket_of_f(ph, p, def.buckets),
    }
}

/// The phase as a share of the period, in `[0, 1)`: the harmonic form's
/// argument.
fn share(def: &ProfileDef, clock: Option<ClockValue>, rows: i64) -> f64 {
    let f = match phase(def, clock, rows) {
        Phase::Int(ph, p) => ph as f64 / p as f64,
        Phase::Float(ph, p) => ph / p,
    };
    if f.is_finite() && (0.0..1.0).contains(&f) {
        f
    } else {
        0.0
    }
}

/// The profile of a windowed or sketched bucket read at `now`, `tau` from
/// the origin, its rows outside the window gone.
fn read_win(def: &ProfileDef, k: &KernelDef, b: &WinBucket, tau: f64) -> f64 {
    let enough = |n: u64| n > 0 && n >= u64::from(def.min_samples);
    if let Some(q) = &b.q {
        if !enough(b.count(def)) || q.weight(k, tau) <= 0.0 {
            return f64::NAN;
        }
        return q
            .read(k, def.quantile.expect("a quantile's level"), None)
            .unwrap_or(f64::NAN);
    }
    let m = b.moments(k, tau);
    if !enough(m.n as u64) {
        return f64::NAN;
    }
    stat_of(def, m.n, m.v1, m.v2, m.m, m.m2)
}

/// The statistic of `n` rows' weighted moments.
fn stat_of(def: &ProfileDef, n: f64, v1: f64, v2: f64, m: f64, m2: f64) -> f64 {
    match def.stat {
        ProfileStat::Mean => m,
        ProfileStat::Var | ProfileStat::Std => {
            let var = if n < 2.0 {
                // One row fills its own weight: no spread.
                if def.bias { 0.0 } else { f64::NAN }
            } else if def.bias {
                m2 / v1
            } else {
                let (a, den) = (v1 * v1, v1 * v1 - v2);
                if den > 0.0 {
                    m2 / v1 * (a / den)
                } else {
                    f64::NAN
                }
            };
            let var = if var < 0.0 { 0.0 } else { var };
            if def.stat == ProfileStat::Std {
                var.sqrt()
            } else {
                var
            }
        }
        ProfileStat::Quantile => unreachable!("read from the sketch"),
    }
}

/// The profile of a bucket. A decay scales every moment of it alike, so
/// its time drops out of every ratio read.
fn read(def: &ProfileDef, b: &Bucket, _now: ClockValue) -> f64 {
    if b.n == 0 || b.n < u64::from(def.min_samples) {
        return f64::NAN;
    }
    match def.stat {
        ProfileStat::Mean => b.m,
        ProfileStat::Quantile => unreachable!("a quantile keeps a sketch"),
        ProfileStat::Var | ProfileStat::Std => {
            // The decay scales `V1` and `M2` by `d` and `V2` by `d²`, which
            // cancel in each ratio: read undecayed, so a bucket one row
            // fills has `V1² − V2` exactly 0.
            let var = if def.bias {
                b.m2 / b.v1
            } else {
                let (a, den) = (b.v1 * b.v1, b.v1 * b.v1 - b.v2);
                if den > 0.0 {
                    b.m2 / b.v1 * (a / den)
                } else {
                    f64::NAN
                }
            };
            // `max` would make a withheld NaN 0.
            let var = if var < 0.0 { 0.0 } else { var };
            if def.stat == ProfileStat::Std {
                var.sqrt()
            } else {
                var
            }
        }
    }
}

/// Add a row of value `x` and weight `w` at time `now` to a bucket.
fn add(def: &ProfileDef, b: &mut Bucket, x: f64, w: f64, now: ClockValue) {
    let d = if b.v1 > 0.0 {
        decay(def, b.at, now)
    } else {
        1.0
    };
    b.n += 1;
    b.v1 = b.v1 * d + w;
    b.v2 = b.v2 * d * d + w * w;
    b.m2 *= d;
    let dx = x - b.m;
    b.m += (w / b.v1) * dx;
    b.m2 += w * dx * (x - b.m);
    b.at = now;
}

/// The definitions and where their outputs go, beside the held rows.
pub(super) struct Out<'a> {
    pub defs: &'a [ProfileDef],
    pub first: usize,
}

/// What a row would take one operator's state of group `g` to, in bytes:
/// its window's rows that leave at the row, the row itself and a sketch's
/// new buckets counted, nothing moved. `g` holds no state yet, or since a
/// restart, at the definitions' own bytes.
fn bytes_with(
    def: &ProfileDef,
    st: Option<&ProfState>,
    clock: Option<ClockValue>,
    values: &[f64],
) -> u64 {
    if def.harmonics > 0 {
        let Some(h) = st.and_then(|st| st.harm.as_ref()) else {
            return def.bytes();
        };
        let st = st.expect("a state holds the fit");
        let x = values[def.input];
        let w = def.weight.map_or(1.0, |i| values[i]);
        let rows = if usable(w) && w > 0.0 {
            st.rows
        } else {
            (st.rows - 1).max(0)
        };
        let now = clock.unwrap_or(ClockValue::I64(rows));
        let joins = usable(w) && w > 0.0 && usable(x) && def.window.is_some();
        return st.bytes + if joins { def.row_bytes() } else { 0 }
            - h.leaving(def, now) * def.row_bytes();
    }
    let Some(st) = st.filter(|_| def.keeps_rows()) else {
        // A fresh state: its buckets, and a quantile's two empty sketches
        // a bucket, of one sketch bucket each.
        let sketches = if def.stat == ProfileStat::Quantile {
            16
        } else {
            0
        };
        return def.bytes() + u64::from(def.buckets) * sketches;
    };
    let x = values[def.input];
    let w = def.weight.map_or(1.0, |i| values[i]);
    let weighed = usable(w) && w > 0.0;
    let rows = if weighed {
        st.rows
    } else {
        (st.rows - 1).max(0)
    };
    let now = clock.unwrap_or(ClockValue::I64(rows));
    let b = &st.win[bucket(def, clock, rows) as usize];
    let leaving = def.window.map_or(0, |win| {
        b.rows
            .iter()
            .take_while(|r| outside(win, def.closed, r.at, now))
            .count() as u64
    });
    let joins = weighed && usable(x);
    let grows = if joins {
        b.q.as_ref().map_or(0, |q| 8 * q.growth(x) as u64)
    } else {
        0
    };
    let row = if joins { def.row_bytes() } else { 0 };
    st.bytes + row + grows - leaving * def.row_bytes()
}

/// Before the row `seq` of group `g` is taken: a row that would take an
/// operator's state past its cap is refused; one past half of it is said,
/// once a run, as `(op, bytes, cap)`.
pub(super) fn check_row(
    defs: &[ProfileDef],
    g: &Group,
    clock: Option<ClockValue>,
    values: &[f64],
    seq: u64,
) -> Result<Option<(usize, u64, u64)>, Refusal> {
    let mut said = None;
    for (op, def) in defs.iter().enumerate() {
        let st = if g.restart { None } else { g.profile.get(op) };
        let bytes = bytes_with(def, st, clock, values);
        if bytes > def.cap {
            return Err(Refusal::StateCap {
                seq,
                of: super::StateOf::Profile,
                op,
                bytes,
                cap: def.cap,
            });
        }
        // A state the data sizes: a window's rows, a quantile's sketches.
        let grows = def.keeps_rows() || (def.harmonics > 0 && def.window.is_some());
        if said.is_none() && grows && bytes > def.cap / 2 {
            said = Some((op, bytes, def.cap));
        }
    }
    Ok(said)
}

/// The row `seq` of group `g`: each operator under `"left"` reads its
/// stamp's value (once a stamp), and the row joins its bucket.
pub(super) fn row(
    out: &Out<'_>,
    g: &mut Group,
    clock: Option<ClockValue>,
    values: &[f64],
    new_stamp: bool,
    seq: u64,
    held: &mut Held<'_>,
) {
    if g.profile.len() != out.defs.len() {
        g.profile = out.defs.iter().map(ProfState::new).collect();
    }
    for (j, (def, st)) in out.defs.iter().zip(g.profile.iter_mut()).enumerate() {
        let x = values[def.input];
        let w = def.weight.map_or(1.0, |i| values[i]);
        let weighed = usable(w) && w > 0.0;
        // The row clock's time: this row's place where it counts, else the
        // last counted row's.
        let rows = if weighed {
            st.rows
        } else {
            (st.rows - 1).max(0)
        };
        let now = clock.unwrap_or(ClockValue::I64(rows));
        let origin = *st.origin.get_or_insert(now);
        let tau = age(origin, now);
        let k = def.kernel();
        if let Some(h) = st.harm.as_mut() {
            if new_stamp || st.at.is_none() || clock.is_none() {
                let phi = share(def, clock, rows);
                st.at = Some((0, now));
                st.phi = Some(phi);
                let before = h.bytes(def);
                h.evict(def, &k, now);
                st.bytes = st.bytes - before + h.bytes(def);
                if !def.waits() {
                    st.left = h.read(def, &k, phi, tau);
                }
            }
            if !def.waits() {
                let r = (seq - held.first) as usize;
                held.values[r * held.n_outputs + out.first + j] = st.left;
            }
            if weighed {
                st.rows += 1;
                if usable(x) {
                    let phi = st.phi.expect("set above");
                    let before = h.bytes(def);
                    h.push(def, &k, (now, tau, phi), x, w);
                    st.bytes = st.bytes - before + h.bytes(def);
                }
            }
            continue;
        }
        if new_stamp || st.at.is_none() || clock.is_none() {
            let b = bucket(def, clock, rows);
            st.at = Some((b, now));
            if def.keeps_rows() {
                let wb = &mut st.win[b as usize];
                let before = wb.bytes(def);
                wb.evict(def, &k, now);
                st.bytes = st.bytes - before + wb.bytes(def);
            }
            if !def.waits() {
                st.left = if def.keeps_rows() {
                    read_win(def, &k, &st.win[b as usize], tau)
                } else {
                    read(def, &st.buckets[b as usize], now)
                };
            }
        }
        if !def.waits() {
            let r = (seq - held.first) as usize;
            held.values[r * held.n_outputs + out.first + j] = st.left;
        }
        if weighed {
            st.rows += 1;
            if usable(x) {
                let (b, _) = st.at.expect("set above");
                if def.keeps_rows() {
                    let wb = &mut st.win[b as usize];
                    let before = wb.bytes(def);
                    wb.push(def, &k, WRow { at: now, tau, x, w });
                    st.bytes = st.bytes - before + wb.bytes(def);
                } else {
                    add(def, &mut st.buckets[b as usize], x, w, now);
                }
            }
        }
    }
}

/// The rows of group `g`'s stamp read their profile under `"right"`.
pub(super) fn flush(out: &Out<'_>, g: &mut Group, pending: &[u64], held: &mut Held<'_>) {
    for (j, (def, st)) in out.defs.iter().zip(g.profile.iter_mut()).enumerate() {
        if !def.waits() {
            continue;
        }
        let v = match st.at {
            None => f64::NAN,
            Some((_, now)) if def.harmonics > 0 => {
                let k = def.kernel();
                let h = st.harm.as_mut().expect("a harmonic fit");
                let before = h.bytes(def);
                h.evict(def, &k, now);
                st.bytes = st.bytes - before + h.bytes(def);
                let tau = st.origin.map_or(0.0, |o| age(o, now));
                h.read(def, &k, st.phi.unwrap_or(0.0), tau)
            }
            Some((b, now)) if def.keeps_rows() => {
                let k = def.kernel();
                let wb = &mut st.win[b as usize];
                let before = wb.bytes(def);
                wb.evict(def, &k, now);
                st.bytes = st.bytes - before + wb.bytes(def);
                let tau = st.origin.map_or(0.0, |o| age(o, now));
                read_win(def, &k, &st.win[b as usize], tau)
            }
            Some((b, now)) => read(def, &st.buckets[b as usize], now),
        };
        for &seq in pending {
            let r = (seq - held.first) as usize;
            held.values[r * held.n_outputs + out.first + j] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(period: PhaseSpan, buckets: u32) -> ProfileDef {
        ProfileDef {
            stat: ProfileStat::Mean,
            buckets,
            period,
            offset: match period {
                PhaseSpan::Ns(_) => PhaseSpan::Ns(0),
                PhaseSpan::Units(_) => PhaseSpan::Units(0.0),
            },
            zone: None,
            half_life: 10.0,
            closed: Closed::Right,
            input: 0,
            weight: None,
            min_samples: 1,
            bias: false,
            cap: PROFILE_STATE_BYTES,
            window: None,
            quantile: None,
            harmonics: 0,
        }
    }

    const HOUR: i64 = 3_600_000_000_000;

    #[test]
    fn the_phase_of_a_numeric_clock_is_its_remainder() {
        let d = def(PhaseSpan::Units(24.0), 24);
        for (t, b) in [(0.0, 0), (23.5, 23), (24.0, 0), (-0.5, 23), (49.0, 1)] {
            assert_eq!(bucket(&d, Some(ClockValue::F64(t)), 0), b, "{t}");
        }
        for (t, b) in [(0, 0), (23, 23), (24, 0), (-1, 23), (49, 1)] {
            assert_eq!(bucket(&d, Some(ClockValue::I64(t)), 0), b, "{t}");
        }
        assert_eq!(bucket(&d, None, 50), 2);
    }

    /// A week starts on Monday, as Polars truncates one: 1970-01-05 is
    /// Monday, so the 4th of January (a Sunday) is the last day of a week.
    #[test]
    fn a_week_starts_on_monday() {
        let d = def(PhaseSpan::Ns(WEEK_NS), 7);
        assert_eq!(bucket(&d, Some(ClockValue::Ns(4 * DAY_NS)), 0), 0);
        assert_eq!(bucket(&d, Some(ClockValue::Ns(3 * DAY_NS)), 0), 6);
        assert_eq!(bucket(&d, Some(ClockValue::Ns(0)), 0), 3);
    }

    /// Europe/Amsterdam goes from UTC+1 to UTC+2 at 01:00 UTC on
    /// 2026-03-29: 03:00 UTC is 05:00 on the wall clock.
    #[test]
    fn a_zoned_clock_reads_the_wall_clock_across_daylight_saving() {
        let mut d = def(PhaseSpan::Ns(DAY_NS), 24);
        d.zone = Some("Europe/Amsterdam".into());
        // 2026-03-29T00:00:00Z
        let midnight_utc = 20_541 * DAY_NS;
        assert_eq!(
            bucket(&d, Some(ClockValue::Ns(midnight_utc + 3 * HOUR)), 0),
            5
        );
        assert_eq!(bucket(&d, Some(ClockValue::Ns(midnight_utc)), 0), 1);
        assert_eq!(bucket(&d, Some(ClockValue::Ns(midnight_utc - HOUR)), 0), 0);
    }

    /// The weighted moments against the definition written out.
    #[test]
    fn a_bucket_is_the_decayed_weighted_moments() {
        let mut d = def(PhaseSpan::Units(1.0), 1);
        d.stat = ProfileStat::Var;
        let rows = [
            (0.0, 1.0, 2.0),
            (1.0, 3.0, 0.5),
            (4.0, -2.0, 1.5),
            (7.5, 5.0, 1.0),
        ];
        let mut b = Bucket::default();
        for &(t, x, w) in &rows {
            add(&d, &mut b, x, w, ClockValue::F64(t));
        }
        let now = 9.0;
        let ws: Vec<f64> = rows
            .iter()
            .map(|&(t, _, w)| w * (-(now - t) / d.half_life).exp2())
            .collect();
        let v1: f64 = ws.iter().sum();
        let v2: f64 = ws.iter().map(|w| w * w).sum();
        let m = rows.iter().zip(&ws).map(|(r, w)| w * r.1).sum::<f64>() / v1;
        let m2: f64 = rows
            .iter()
            .zip(&ws)
            .map(|(r, w)| w * (r.1 - m).powi(2))
            .sum();
        let want = m2 / v1 * v1 * v1 / (v1 * v1 - v2);
        let got = read(&d, &b, ClockValue::F64(now));
        assert!((got - want).abs() < 1e-12 * want, "{got} {want}");
        d.stat = ProfileStat::Mean;
        assert!((read(&d, &b, ClockValue::F64(now)) - m).abs() < 1e-12);
        // Fewer rows than min_samples, withheld.
        d.min_samples = 5;
        assert!(read(&d, &b, ClockValue::F64(now)).is_nan());
        d.min_samples = 4;
        assert!(!read(&d, &b, ClockValue::F64(now)).is_nan());
        // One row's weight fills a variance: withheld.
        let mut one = Bucket::default();
        add(&d, &mut one, 3.0, 2.0, ClockValue::F64(0.0));
        d.stat = ProfileStat::Var;
        d.min_samples = 1;
        assert!(read(&d, &one, ClockValue::F64(1.0)).is_nan());
        assert!(read(&d, &Bucket::default(), ClockValue::F64(1.0)).is_nan());
    }

    /// Two groups through the core, under both closed forms and with ties:
    /// the core checks after every row, round-trips its state to the byte,
    /// and reads the definition.
    #[test]
    fn the_core_reads_each_phase_and_checks_after_every_row() {
        use super::super::{RowIn, Windows};
        use rheon_core::ClockCfg;
        let base = ProfileDef {
            weight: Some(1),
            half_life: 7.0,
            ..def(PhaseSpan::Units(6.0), 3)
        };
        let defs = vec![
            base.clone(),
            ProfileDef {
                closed: Closed::Left,
                stat: ProfileStat::Var,
                ..base.clone()
            },
            ProfileDef {
                closed: Closed::Both,
                stat: ProfileStat::Var,
                window: Some(PhaseSpan::Units(9.0)),
                ..base.clone()
            },
            ProfileDef {
                stat: ProfileStat::Quantile,
                quantile: Some(0.5),
                window: Some(PhaseSpan::Units(9.0)),
                ..base.clone()
            },
            ProfileDef {
                buckets: 0,
                harmonics: 1,
                stat: ProfileStat::Std,
                closed: Closed::Left,
                ..base.clone()
            },
            ProfileDef {
                buckets: 0,
                harmonics: 2,
                window: Some(PhaseSpan::Units(9.0)),
                ..base
            },
        ];
        let cfg = ClockCfg {
            gap_cap: 1e9,
            ..ClockCfg::default()
        };
        let mut core = Windows::with_profiles(Vec::new(), Vec::new(), defs.clone(), cfg).unwrap();
        core.set_grouped(true);
        let mut s = 11u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut rows: Vec<(f64, usize, f64, f64)> = Vec::new();
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); 6];
        for k in 0..120 {
            let t = f64::from(k / 2);
            let g = usize::from(next() < 0.5);
            let (x, w) = (
                next() * 10.0,
                if next() < 0.15 { 0.0 } else { next() + 0.5 },
            );
            let gi = core.group(Some(&format!("g{g}")));
            core.push(&RowIn {
                group: gi,
                clock: Some(ClockValue::F64(t)),
                session: None,
                values: &[x, w],
                accept: true,
                lists: &[],
            })
            .unwrap();
            rows.push((t, g, x, w));
            core.check().unwrap();
            let bytes = rmp_serde::to_vec_named(&core).unwrap();
            let back: Windows = rmp_serde::from_slice(&bytes).unwrap();
            back.check().unwrap();
            assert_eq!(rmp_serde::to_vec_named(&back).unwrap(), bytes);
            for (o, col) in core.drain().values.iter().enumerate() {
                out[o].extend(col);
            }
        }
        for (o, col) in core.finish().values.iter().enumerate() {
            out[o].extend(col);
        }
        // The definition: the group's rows of the same phase bucket, at or
        // before (right) or before (left) the row's stamp, decayed.
        for (i, &(t, g, _, _)) in rows.iter().enumerate() {
            let b = (t.rem_euclid(6.0) / 6.0 * 3.0).floor();
            for (o, d) in defs.iter().enumerate() {
                if d.stat == ProfileStat::Quantile || d.harmonics > 0 {
                    // Held to NumPy and statsmodels in the Python tests;
                    // here, read and checked at every row.
                    continue;
                }
                let take = |r: &&(f64, usize, f64, f64)| {
                    r.1 == g
                        && r.3 > 0.0
                        && (r.0.rem_euclid(6.0) / 6.0 * 3.0).floor() == b
                        && if d.waits() { r.0 <= t } else { r.0 < t }
                        && d.window.is_none_or(|_| t - r.0 <= 9.0)
                };
                let ws: Vec<(f64, f64)> = rows
                    .iter()
                    .filter(take)
                    .map(|r| (r.2, r.3 * (-(t - r.0) / 7.0).exp2()))
                    .collect();
                let v1: f64 = ws.iter().map(|p| p.1).sum();
                let v2: f64 = ws.iter().map(|p| p.1 * p.1).sum();
                let m = ws.iter().map(|p| p.0 * p.1).sum::<f64>() / v1;
                let m2: f64 = ws.iter().map(|p| p.1 * (p.0 - m).powi(2)).sum();
                let want = if v1 <= 0.0 {
                    f64::NAN
                } else if d.stat == ProfileStat::Mean {
                    m
                } else if v1 * v1 - v2 > 1e-12 * v1 * v1 {
                    m2 / v1 * v1 * v1 / (v1 * v1 - v2)
                } else {
                    f64::NAN
                };
                let got = out[o][i];
                assert!(
                    (got.is_nan() && want.is_nan())
                        || (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "row {i} op {o}: {got} against {want}"
                );
            }
        }
    }
}
