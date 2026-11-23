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
//! **Withheld** (null): a bucket whose weight, decayed to the row's time, is
//! 0 or below `min_weight`; a variance of a bucket one row's weight fills
//! (`V1² = V2`). A row of weight 0, or with no value, reads its bucket and
//! adds nothing.

use rheon_core::{ClockValue, seconds_of_ns};
use serde::{Deserialize, Serialize};

use super::{Closed, Direction, Group, Held};
use crate::stream::usable;

/// A nanosecond day and week.
const DAY_NS: i64 = 86_400_000_000_000;
const WEEK_NS: i64 = 7 * DAY_NS;
/// The first Monday after the epoch, 1970-01-05, where Polars' `truncate`
/// starts a whole number of weeks.
const MONDAY_NS: i64 = 4 * DAY_NS;

/// The bytes one bucket of one operator's state is counted at: its six
/// numbers (`V1`, `V2`, `m`, `M2` and the time it was last moved, which is
/// two words as a tagged clock value).
pub const BUCKET_BYTES: u64 = 48;

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
}

impl ProfileStat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mean" => Some(Self::Mean),
            "var" => Some(Self::Var),
            "std" => Some(Self::Std),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Var => "var",
            Self::Std => "std",
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
    pub min_weight: f64,
    pub bias: bool,
    /// The most bytes one group's state of this operator may hold.
    pub cap: u64,
}

impl ProfileDef {
    /// The bytes one group's state of this operator holds.
    pub fn bytes(&self) -> u64 {
        u64::from(self.buckets) * BUCKET_BYTES
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
        if self.buckets == 0
            || !positive(self.period)
            || !same_kind
            || !offset_finite
            || self.half_life.is_nan()
            || self.half_life <= 0.0
            || !(self.min_weight >= 0.0 && self.min_weight.is_finite())
            || self.bytes() > self.cap
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
    v1: f64,
    v2: f64,
    m: f64,
    m2: f64,
    at: ClockValue,
}

impl Default for Bucket {
    fn default() -> Self {
        Self {
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
}

impl ProfState {
    fn new(def: &ProfileDef) -> Self {
        Self {
            buckets: vec![Bucket::default(); def.buckets as usize],
            rows: 0,
            at: None,
            left: f64::NAN,
        }
    }
}

/// Whether a group's profile states fit the definitions: none (a group
/// that has not started since its last restart) or one each, of its
/// buckets.
pub(super) fn fits(defs: &[ProfileDef], states: &[ProfState]) -> bool {
    states.is_empty()
        || (states.len() == defs.len()
            && states
                .iter()
                .zip(defs)
                .all(|(s, d)| s.buckets.len() == d.buckets as usize && s.rows >= 0))
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

/// The bucket of a clock value (`None`: the row clock's `rows`).
pub(super) fn bucket(def: &ProfileDef, clock: Option<ClockValue>, rows: i64) -> u32 {
    let b = def.buckets;
    let units = |t: f64| match (def.period, def.offset) {
        (PhaseSpan::Units(p), PhaseSpan::Units(o)) => bucket_of_f((t - o).rem_euclid(p), p, b),
        _ => 0,
    };
    let int = |t: i64| match (def.period, def.offset) {
        (PhaseSpan::Units(p), PhaseSpan::Units(o))
            if p.fract() == 0.0 && o.fract() == 0.0 && p.abs() < 9e18 && o.abs() < 9e18 =>
        {
            let (p, o) = (p as i128, o as i128);
            bucket_of_int((i128::from(t) - o).rem_euclid(p), p, b)
        }
        _ => units(t as f64),
    };
    match clock {
        None => int(rows),
        Some(ClockValue::I64(t)) => int(t),
        Some(ClockValue::F64(t)) => units(t),
        Some(ClockValue::Ns(t)) => {
            let (PhaseSpan::Ns(p), PhaseSpan::Ns(o)) = (def.period, def.offset) else {
                return 0;
            };
            let local = match def.zone.as_deref().map(zone_of) {
                Some(Ok(z)) => i128::from(t) + i128::from(zone_offset_ns(&z, t)),
                _ => i128::from(t),
            };
            let origin = if p % WEEK_NS == 0 { MONDAY_NS } else { 0 };
            let ph = (local - i128::from(origin) - i128::from(o)).rem_euclid(i128::from(p));
            bucket_of_int(ph, i128::from(p), b)
        }
    }
}

/// The profile of a bucket read at time `now`.
fn read(def: &ProfileDef, b: &Bucket, now: ClockValue) -> f64 {
    if b.v1 <= 0.0 {
        return f64::NAN;
    }
    let d = decay(def, b.at, now);
    let v1 = b.v1 * d;
    if v1.is_nan() || v1 <= 0.0 || v1 < def.min_weight {
        return f64::NAN;
    }
    match def.stat {
        ProfileStat::Mean => b.m,
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
        if new_stamp || st.at.is_none() || clock.is_none() {
            let b = bucket(def, clock, rows);
            st.at = Some((b, now));
            if !def.waits() {
                st.left = read(def, &st.buckets[b as usize], now);
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
                add(def, &mut st.buckets[b as usize], x, w, now);
            }
        }
    }
}

/// The rows of group `g`'s stamp read their profile under `"right"`.
pub(super) fn flush(out: &Out<'_>, g: &Group, pending: &[u64], held: &mut Held<'_>) {
    for (j, (def, st)) in out.defs.iter().zip(&g.profile).enumerate() {
        if !def.waits() {
            continue;
        }
        let v = st
            .at
            .map_or(f64::NAN, |(b, now)| read(def, &st.buckets[b as usize], now));
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
            min_weight: 0.0,
            bias: false,
            cap: PROFILE_STATE_BYTES,
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
        // Below min_weight, withheld.
        d.min_weight = v1 * 1.0001;
        assert!(read(&d, &b, ClockValue::F64(now)).is_nan());
        // One row's weight fills a variance: withheld.
        let mut one = Bucket::default();
        add(&d, &mut one, 3.0, 2.0, ClockValue::F64(0.0));
        d.stat = ProfileStat::Var;
        d.min_weight = 0.0;
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
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); 2];
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
                let take = |r: &&(f64, usize, f64, f64)| {
                    r.1 == g
                        && r.3 > 0.0
                        && (r.0.rem_euclid(6.0) / 6.0 * 3.0).floor() == b
                        && if d.closed == Closed::Right {
                            r.0 <= t
                        } else {
                            r.0 < t
                        }
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
