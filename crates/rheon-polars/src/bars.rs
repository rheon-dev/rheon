//! Threshold-triggered bars (`rp.stream.bars`): one output row for each run
//! of consecutive input rows whose running quantity reaches a threshold.
//!
//! The crossing rule is `rheon_core::bars`: a count, a sum whose remainder
//! is carried as the multiples of the threshold on the running total, or a
//! signed sum that resets at each bar, each kept exactly. This module reads
//! the columns of each chunk, keeps one open bar per group with its
//! aggregates, cuts a bar short on the clock's settings, and writes the
//! closed bars as a frame and the open ones to the state file.
//!
//! # A chunk at a time
//!
//! A bar is emitted at the row that closes it, in the chunk that holds that
//! row; a chunk that closes none gives no rows. Every decision is a function
//! of the rows up to it, so one chunk or a thousand give the same bars, to
//! the bit. Ties on the clock are broken by row order: a row after the
//! closing row at the same clock opens the next bar.
//!
//! # Rows of weight 0
//!
//! A row of weight 0 is in no bar and no clock: it adds nothing to the
//! quantity, enters no aggregate, is never a bar's open or close, and cannot
//! end a gap or start a session. It is still read for the order check, since
//! an input out of order is refused whatever its weights. A weight is 0 or 1
//! only: a bar's count or high has no meaning for a weight of 2.5.
//!
//! # The bound on memory
//!
//! A group holds its open bar's accumulators, its remainder and its clock:
//! bytes fixed by the call but for the key's text, so the worst case is
//! checked against `max_state_bytes` when the call is built and again, key
//! included, when a group is first seen. The groups together are held to
//! `max_total_bytes`, with a notice once at half of it.

use std::cmp::Ordering;
use std::collections::HashMap;

use polars::prelude::*;
use rheon_core::FixedState;
use rheon_core::bars::{BarKind, Crossing, Refused as CrossRefused};
use rheon_core::exact::{Expansion, MAX_PARTS, parts_below};
use serde::{Deserialize, Serialize};

use crate::bank::{GroupKey, key_cmp, session_hash};
use crate::refresh::{ClockKind, Instant};
use crate::stream::usable;

#[cfg(test)]
mod tests;

const WHO: &str = "rp.stream.bars";

/// The version of a bars state, in the envelope every state file shares.
const BARS_VERSION: u32 = 1;

/// The most parts a running quantity holds: its values are below `2^334`,
/// twice the input bound of 1e100 that bounds a quantity and a threshold.
const RUNNING_PARTS: usize = parts_below(334);

/// A Polars aggregation a bar takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggKind {
    First,
    Last,
    Min,
    Max,
    Sum,
    /// The non-null values.
    Count,
    /// The rows, `pl.len()`.
    Len,
}

/// One aggregation: of a column (none for `Len`), written as `name`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AggSpec {
    pub kind: AggKind,
    #[serde(default)]
    pub column: Option<String>,
    pub name: String,
}

/// The threshold: a number, or a column read at each row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Threshold {
    Value(f64),
    Column(String),
}

/// The call, as `rp.stream.bars` builds it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BarsConfig {
    pub kind: BarKind,
    pub clock: String,
    pub threshold: Threshold,
    #[serde(default)]
    pub quantity: Option<String>,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub session: Option<String>,
    /// Spans in the clock's units; integer nanoseconds on a temporal clock.
    #[serde(default)]
    pub gap_cap: Option<f64>,
    #[serde(default)]
    pub max_span: Option<f64>,
    #[serde(default)]
    pub restart_after_step_back: Option<f64>,
    /// Whether a bar cut short by the clock is emitted.
    #[serde(default = "yes")]
    pub keep_partial: bool,
    /// `group_close = "monotone"`: a greater key closes the groups before it.
    #[serde(default)]
    pub monotone: bool,
    pub max_state_bytes: u64,
    pub max_total_bytes: u64,
    #[serde(default)]
    pub aggregates: Vec<AggSpec>,
    /// The call as the caller wrote it -- the expressions' text, the
    /// columns' names -- which a load compares whole.
    #[serde(default)]
    pub call: std::collections::BTreeMap<String, String>,
}

fn yes() -> bool {
    true
}

/// A value as an accumulator holds it, by the physical type of its column.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum Val {
    Null,
    F(f64),
    I(i64),
    U(u64),
    B(bool),
}

/// How a column is read: its physical values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Float,
    Int,
    UInt,
    Bool,
}

fn class_of(dtype: &DataType) -> Option<Class> {
    use DataType::*;
    Some(match dtype {
        Float32 | Float64 => Class::Float,
        Int8 | Int16 | Int32 | Int64 | Date | Datetime(..) | Duration(_) | Time => Class::Int,
        UInt8 | UInt16 | UInt32 | UInt64 => Class::UInt,
        Boolean => Class::Bool,
        _ => return None,
    })
}

/// One aggregation's accumulator over the open bar's rows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Acc {
    /// `None` before the bar's first row; the value, null included, after.
    First(Option<Val>),
    Last(Option<Val>),
    /// Over the non-null values that are not NaN; `nan` for a NaN seen,
    /// which is the answer when no other value is.
    Min {
        best: Option<Val>,
        nan: bool,
    },
    Max {
        best: Option<Val>,
        nan: bool,
    },
    /// An integer or Boolean column's sum, exactly.
    SumInt(i128),
    /// A float column's sum: the finite values exactly, NaN and the
    /// infinities apart, and a sum past `2^1020` as its sign.
    SumFloat {
        exact: Expansion,
        nan: bool,
        pos_inf: bool,
        neg_inf: bool,
        over: i8,
    },
    Count(u64),
    Len(u64),
}

impl Acc {
    fn new(kind: AggKind, class: Option<Class>) -> Self {
        match kind {
            AggKind::First => Self::First(None),
            AggKind::Last => Self::Last(None),
            AggKind::Min => Self::Min {
                best: None,
                nan: false,
            },
            AggKind::Max => Self::Max {
                best: None,
                nan: false,
            },
            AggKind::Sum if class == Some(Class::Float) => Self::SumFloat {
                exact: Expansion::new(),
                nan: false,
                pos_inf: false,
                neg_inf: false,
                over: 0,
            },
            AggKind::Sum => Self::SumInt(0),
            AggKind::Count => Self::Count(0),
            AggKind::Len => Self::Len(0),
        }
    }

    /// Whether this accumulator is of the form `kind` over `class` takes: a
    /// saved state's check.
    fn fits(&self, kind: AggKind, class: Option<Class>) -> bool {
        matches!(
            (self, kind),
            (Self::First(_), AggKind::First)
                | (Self::Last(_), AggKind::Last)
                | (Self::Min { .. }, AggKind::Min)
                | (Self::Max { .. }, AggKind::Max)
                | (Self::Count(_), AggKind::Count)
                | (Self::Len(_), AggKind::Len)
        ) || matches!(
            (self, kind, class),
            (Self::SumFloat { .. }, AggKind::Sum, Some(Class::Float))
                | (
                    Self::SumInt(_),
                    AggKind::Sum,
                    Some(Class::Int | Class::UInt | Class::Bool)
                )
        )
    }

    /// The bytes it holds at most: the enum, and a float sum's parts.
    fn worst_bytes(&self) -> u64 {
        let base = std::mem::size_of::<Self>() as u64;
        match self {
            Self::SumFloat { .. } => base + 8 * MAX_PARTS as u64,
            _ => base,
        }
    }

    fn update(&mut self, v: Val) {
        // A value min and max read: not null, not NaN.
        let ordered = !matches!(v, Val::Null) && !matches!(v, Val::F(x) if x.is_nan());
        let cmp = |a: Val, b: Val| -> Ordering {
            match (a, b) {
                (Val::F(a), Val::F(b)) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
                (Val::I(a), Val::I(b)) => a.cmp(&b),
                (Val::U(a), Val::U(b)) => a.cmp(&b),
                (Val::B(a), Val::B(b)) => a.cmp(&b),
                _ => Ordering::Equal,
            }
        };
        match self {
            Self::First(f) => {
                if f.is_none() {
                    *f = Some(v);
                }
            }
            Self::Last(l) => *l = Some(v),
            Self::Min { best, nan } => {
                if ordered {
                    if best.is_none_or(|b| cmp(v, b) == Ordering::Less) {
                        *best = Some(v);
                    }
                } else if matches!(v, Val::F(_)) {
                    *nan = true;
                }
            }
            Self::Max { best, nan } => {
                if ordered {
                    if best.is_none_or(|b| cmp(v, b) == Ordering::Greater) {
                        *best = Some(v);
                    }
                } else if matches!(v, Val::F(_)) {
                    *nan = true;
                }
            }
            Self::SumInt(s) => match v {
                Val::I(x) => *s += i128::from(x),
                Val::U(x) => *s += i128::from(x),
                Val::B(b) => *s += i128::from(b),
                _ => {}
            },
            Self::SumFloat {
                exact,
                nan,
                pos_inf,
                neg_inf,
                over,
            } => {
                if let Val::F(x) = v {
                    if x.is_nan() {
                        *nan = true;
                    } else if x == f64::INFINITY {
                        *pos_inf = true;
                    } else if x == f64::NEG_INFINITY {
                        *neg_inf = true;
                    } else if *over == 0 && !exact.add_checked(x) {
                        *over = if exact.value() + x > 0.0 { 1 } else { -1 };
                    }
                }
            }
            Self::Count(n) => {
                if v != Val::Null {
                    *n += 1;
                }
            }
            Self::Len(n) => *n += 1,
        }
    }
}

/// One group's open bar and clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Group {
    crossing: Crossing,
    /// The index of the next bar emitted.
    bar: u64,
    /// The open bar's rows.
    rows: u64,
    /// The clock of the open bar's first row.
    time_open: Option<Instant>,
    /// The clock of the group's last row of positive weight: the open bar's
    /// last row, and where a gap is measured from.
    time_last: Option<Instant>,
    /// The clock of the group's last row of any weight, for the order check.
    last_clock: Option<Instant>,
    /// The session of the group's last row of positive weight.
    session: Option<u64>,
    aggs: Vec<Acc>,
}

/// One bar to write.
struct Out {
    key: GroupKey,
    bar: u64,
    open: Instant,
    close: Instant,
    thresholds: u64,
    carry: f64,
    complete: bool,
    aggs: Vec<Acc>,
}

/// What [`Bars::save_bytes`] writes, in the envelope every state file
/// shares: the call, each group's open bar, and the high-water key of a
/// monotone close.
#[derive(Serialize, Deserialize)]
struct BarsFile {
    magic: String,
    kind: String,
    version: u32,
    config: BarsConfig,
    clock_kind: Option<ClockKind>,
    /// Sorted by key, so one state is one file, byte for byte.
    groups: Vec<(GroupKey, Group)>,
    #[serde(default)]
    high_water: Option<GroupKey>,
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    checksum: Option<crate::checksum::Trailer>,
}

/// The bars of a stream, fed a chunk at a time.
#[derive(Debug)]
pub struct Bars {
    cfg: BarsConfig,
    /// The input's dtypes, read when the call is built.
    clock_dtype: DataType,
    group_dtype: Option<DataType>,
    agg_dtypes: Vec<Option<DataType>>,
    groups: HashMap<GroupKey, Group, FixedState>,
    high_water: Option<GroupKey>,
    clock_kind: Option<ClockKind>,
    rows_fed: usize,
    /// The groups' bytes at most, summed.
    total_bytes: u64,
    warned_half: bool,
    notices: Vec<String>,
}

/// The group key's text forms a monotone close orders as numbers.
fn integer_key(dtype: &DataType) -> bool {
    dtype.is_integer()
}

impl Bars {
    /// A run of the call over an input of `input`'s columns.
    ///
    /// # Errors
    ///
    /// A column the input does not have or of a type the call cannot read,
    /// or settings outside their range: each named.
    pub fn new(cfg: BarsConfig, input: &Schema) -> Result<Self, String> {
        let col = |role: &str, name: &str| -> Result<DataType, String> {
            input.get(name).cloned().ok_or_else(|| {
                format!(
                    "{WHO}: the input has no {role} column {name:?}. Set {role} to a column of the \
                     input."
                )
            })
        };
        let clock_dtype = col("clock", &cfg.clock)?;
        if !(clock_dtype.is_numeric()
            || matches!(
                clock_dtype,
                DataType::Datetime(..) | DataType::Date | DataType::Duration(_)
            ))
        {
            return Err(format!(
                "{WHO}: clock column {:?} must be a number, a Datetime, a Date or a Duration, but \
                 it is {}. Cast it, such as with pl.col({:?}).cast(pl.Float64).",
                cfg.clock,
                crate::arrow::dtype_name(&clock_dtype),
                cfg.clock
            ));
        }
        if clock_dtype.is_integer() && !crate::arrow::fits_64(&clock_dtype) {
            return Err(crate::arrow::wide_integer_clock(
                WHO,
                &cfg.clock,
                &clock_dtype,
            ));
        }
        let group_dtype = match &cfg.group {
            Some(g) => {
                let d = col("group", g)?;
                let ok = d.is_integer()
                    || matches!(
                        d,
                        DataType::String | DataType::Date | DataType::Categorical(..)
                    );
                if !ok || (d.is_integer() && !crate::arrow::fits_64(&d)) {
                    return Err(format!(
                        "{WHO}: group column {g:?} is {}, but a group is an integer, a String, a \
                         Categorical or a Date. Cast it, such as with pl.col({g:?}).cast(pl.String).",
                        crate::arrow::dtype_name(&d)
                    ));
                }
                Some(d)
            }
            None => None,
        };
        let numeric = |role: &str, name: &str| -> Result<(), String> {
            let d = col(role, name)?;
            if d.is_numeric() || matches!(d, DataType::Boolean | DataType::Null) {
                Ok(())
            } else {
                Err(format!(
                    "{WHO}: {role} column {name:?} must be numeric, but it is {}. Cast it, such as \
                     with pl.col({name:?}).cast(pl.Float64).",
                    crate::arrow::dtype_name(&d)
                ))
            }
        };
        match (cfg.kind, &cfg.quantity) {
            (BarKind::Count, Some(_)) => {
                return Err(format!(
                    "{WHO}: kind=\"count\" counts rows, so it takes no quantity. Leave out \
                     quantity, or set kind to \"sum\" or \"imbalance\"."
                ));
            }
            (BarKind::Sum | BarKind::Imbalance, None) => {
                return Err(format!(
                    "{WHO}: kind={:?} sums a quantity, but none is given. Give quantity, a column \
                     or an expression of the row.",
                    cfg.kind.name()
                ));
            }
            (_, Some(q)) => numeric("quantity", q)?,
            _ => {}
        }
        if let Some(w) = &cfg.weight {
            numeric("weight", w)?;
        }
        if let Some(s) = &cfg.session {
            col("session", s)?;
        }
        match &cfg.threshold {
            Threshold::Value(t) => check_threshold(cfg.kind, *t).map_err(|why| {
                format!("{WHO}: threshold is {t}, but {why}. Give a threshold that is.")
            })?,
            Threshold::Column(c) => numeric("threshold", c)?,
        }
        for (key, v) in [
            ("gap_cap", cfg.gap_cap),
            ("max_span", cfg.max_span),
            ("restart_after_step_back", cfg.restart_after_step_back),
        ] {
            if let Some(v) = v
                && (v.is_nan() || v < 0.0)
            {
                return Err(format!(
                    "{WHO}: {key} must be 0 or more, but it is {v}. Set it to 0 or more, or leave \
                     it out."
                ));
            }
        }
        if cfg.monotone && cfg.group.is_none() {
            return Err(format!(
                "{WHO}: group_close=\"monotone\" closes each group when a greater key arrives, \
                 but no group is given. Give group, or leave out group_close."
            ));
        }
        let mut agg_dtypes = Vec::with_capacity(cfg.aggregates.len());
        for a in &cfg.aggregates {
            let d = match (&a.column, a.kind) {
                (None, AggKind::Len) => None,
                (Some(c), _) => {
                    let d = input.get(c).cloned().ok_or_else(|| {
                        format!(
                            "{WHO}: aggregate {:?} reads column {c:?}, which the input does not \
                             have.",
                            a.name
                        )
                    })?;
                    let cls = class_of(&d);
                    let sum_ok =
                        matches!(d, DataType::Float32 | DataType::Float64 | DataType::Boolean)
                            || (d.is_integer() && crate::arrow::fits_64(&d));
                    if cls.is_none()
                        || (a.kind == AggKind::Sum && !sum_ok)
                        || (d.is_integer() && !crate::arrow::fits_64(&d))
                    {
                        return Err(format!(
                            "{WHO}: aggregate {:?} takes the {} of a column of type {}, which a \
                             bar does not aggregate. A bar's first, last, min and max take a \
                             number, a Boolean or a temporal value, and its sum a number or a \
                             Boolean. Cast the column first.",
                            a.name,
                            agg_name(a.kind),
                            crate::arrow::dtype_name(&d)
                        ));
                    }
                    Some(d)
                }
                (None, _) => {
                    return Err(format!(
                        "{WHO}: aggregate {:?} names no column to aggregate.",
                        a.name
                    ));
                }
            };
            agg_dtypes.push(d);
        }
        let bars = Self {
            cfg,
            clock_dtype,
            group_dtype,
            agg_dtypes,
            groups: HashMap::default(),
            high_water: None,
            clock_kind: None,
            rows_fed: 0,
            total_bytes: 0,
            warned_half: false,
            notices: Vec::new(),
        };
        if bars.cfg.max_state_bytes == 0 || bars.cfg.max_total_bytes == 0 {
            return Err(format!(
                "{WHO}: max_state_bytes and max_total_bytes must be above 0. Give each a number \
                 of bytes, or leave it out for the default."
            ));
        }
        let worst = bars.group_bytes(0);
        if worst > bars.cfg.max_state_bytes {
            return Err(format!(
                "{WHO}: a group holds up to {worst} bytes with these aggregates, past \
                 max_state_bytes of {}. Raise max_state_bytes, or take fewer aggregates.",
                bars.cfg.max_state_bytes
            ));
        }
        Ok(bars)
    }

    /// The bytes a group holds at most, `key_len` bytes of key text
    /// included: the map's entry and the key, the group's fixed fields, the
    /// running quantity's parts, and each aggregate's accumulator.
    ///
    /// ```text
    /// B_group = 48 + key_len + size_of(Group) + 8·p_R + Σ_j (size_of(Acc) + 8·p_j)
    /// ```
    ///
    /// with `p_R` the parts of the running quantity (27 at most, below 2^334)
    /// and `p_j` a float sum's (40 at most), 0 for any other.
    pub fn group_bytes(&self, key_len: usize) -> u64 {
        let running = match self.cfg.kind {
            BarKind::Count => 0,
            _ => 8 * RUNNING_PARTS as u64,
        };
        let aggs: u64 = self
            .cfg
            .aggregates
            .iter()
            .zip(&self.agg_dtypes)
            .map(|(a, d)| Acc::new(a.kind, d.as_ref().and_then(class_of)).worst_bytes())
            .sum();
        48 + key_len as u64 + std::mem::size_of::<Group>() as u64 + running + aggs
    }

    /// The readiness notices since the last call: the groups past half of
    /// `max_total_bytes`.
    pub fn take_notices(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notices)
    }

    /// The output's columns.
    pub fn schema(&self) -> Schema {
        let mut out = Schema::default();
        if let (Some(g), Some(d)) = (&self.cfg.group, &self.group_dtype) {
            out.insert(g.as_str().into(), d.clone());
        }
        out.insert("bar".into(), DataType::Int64);
        out.insert("time_open".into(), self.clock_dtype.clone());
        out.insert("time_close".into(), self.clock_dtype.clone());
        out.insert("thresholds".into(), DataType::Int64);
        if self.cfg.kind == BarKind::Sum {
            out.insert("carry".into(), DataType::Float64);
        }
        out.insert("complete".into(), DataType::Boolean);
        for (a, d) in self.cfg.aggregates.iter().zip(&self.agg_dtypes) {
            out.insert(a.name.as_str().into(), agg_dtype(a.kind, d.as_ref()));
        }
        out
    }

    /// The state as bytes: the call and each group's open bar.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        let mut groups: Vec<(GroupKey, Group)> = self
            .groups
            .iter()
            .map(|(k, g)| (k.clone(), g.clone()))
            .collect();
        groups.sort_by(|a, b| a.0.cmp(&b.0));
        let file = BarsFile {
            magic: crate::envelope::MAGIC.into(),
            kind: crate::envelope::Kind::Bars.tag().into(),
            version: BARS_VERSION,
            config: self.cfg.clone(),
            clock_kind: self.clock_kind,
            groups,
            high_water: self.high_water.clone(),
            checksum: Some(crate::checksum::Trailer::Patched),
        };
        let mut out = rmp_serde::to_vec_named(&file).map_err(|e| e.to_string())?;
        crate::checksum::seal(&mut out);
        Ok(out)
    }

    /// [`Self::save_bytes`] to `path`, replacing it whole or not at all.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = self.save_bytes().map_err(std::io::Error::other)?;
        crate::atomic::write(path, &bytes)
    }

    /// A run that goes on from a saved state, which must have been saved by
    /// the same call.
    ///
    /// # Errors
    ///
    /// Bytes that are not a bars state, another call, or a group that does
    /// not fit the call (a damaged state).
    pub fn load_bytes(bytes: &[u8], cfg: BarsConfig, input: &Schema) -> Result<Self, String> {
        const START_OVER: &str =
            "Load an undamaged copy of the state file, or leave out load_state to start over.";
        let opened = crate::envelope::open(bytes, crate::envelope::Kind::Bars).map_err(|r| {
            use crate::envelope::{Kind, Refused};
            match r {
                Refused::Damaged | Refused::Unsealed => {
                    format!("{WHO}: the state is {}. {START_OVER}", r.what())
                }
                Refused::OtherKind(Kind::Checkpoint) => format!(
                    "{WHO}: the state is a pipeline checkpoint, not a bars run's state. Read its \
                     stages with rheon.polars.stream.load_checkpoint and pass the bars run's stage \
                     as load_state."
                ),
                Refused::OtherKind(_) | Refused::UnknownKind(_) => format!(
                    "{WHO}: the state is {}, not a bars run's. Pass a state file a bars run saved, \
                     or leave out load_state to start over.",
                    r.what()
                ),
            }
        })?;
        let crate::envelope::Opened::Enveloped { version } = opened else {
            return Err(format!(
                "{WHO}: the state is not a bars run's. {START_OVER}"
            ));
        };
        if version != BARS_VERSION {
            return Err(format!(
                "{WHO}: the state's version is {version}, but this version of rheon reads version \
                 {BARS_VERSION}. Load the state with the version of rheon that saved it, or leave \
                 out load_state to start over."
            ));
        }
        let file: BarsFile = rmp_serde::from_slice(bytes).map_err(|e| {
            format!("{WHO}: the state cannot be read ({e}), so it is damaged. {START_OVER}")
        })?;
        let mut differs: Vec<String> = Vec::new();
        let (a, b) = (&file.config, &cfg);
        if a.kind != b.kind {
            differs.push(format!(
                "kind ({:?}, here {:?})",
                a.kind.name(),
                b.kind.name()
            ));
        }
        if a.call != b.call {
            for k in a.call.keys().chain(b.call.keys()) {
                if a.call.get(k) != b.call.get(k) && !differs.iter().any(|d| d.starts_with(k)) {
                    differs.push(format!(
                        "{k} ({}, here {})",
                        a.call.get(k).map_or("not given", String::as_str),
                        b.call.get(k).map_or("not given", String::as_str)
                    ));
                }
            }
        }
        for (key, x, y) in [
            ("gap_cap", a.gap_cap, b.gap_cap),
            ("max_span", a.max_span, b.max_span),
            (
                "restart_after_step_back",
                a.restart_after_step_back,
                b.restart_after_step_back,
            ),
        ] {
            if x.map(f64::to_bits) != y.map(f64::to_bits) {
                differs.push(format!("{key} ({x:?}, here {y:?})"));
            }
        }
        if a.threshold != b.threshold && !b.call.contains_key("threshold") {
            differs.push("threshold".into());
        }
        if a.monotone != b.monotone {
            differs.push("group_close".into());
        }
        if a.aggregates != b.aggregates && !b.call.contains_key("aggregates") {
            differs.push("the aggregates".into());
        }
        if !differs.is_empty() {
            return Err(format!(
                "{WHO}: the state was saved by another call: it differs in {}. Give the settings \
                 the state was saved with, or leave out load_state to start over.",
                differs.join(", ")
            ));
        }
        let mut bars = Self::new(cfg, input)?;
        let classes: Vec<Option<Class>> = bars
            .agg_dtypes
            .iter()
            .map(|d| d.as_ref().and_then(class_of))
            .collect();
        for (key, g) in &file.groups {
            let fits = g.aggs.len() == bars.cfg.aggregates.len()
                && g.aggs
                    .iter()
                    .zip(&bars.cfg.aggregates)
                    .zip(&classes)
                    .all(|((acc, a), c)| acc.fits(a.kind, *c))
                && Crossing::from_parts(bars.cfg.kind, g.crossing.running().clone()).is_ok()
                && (g.rows == 0) == g.time_open.is_none();
            if !fits || (bars.cfg.group.is_none() && key.as_str() != Some("")) {
                return Err(format!(
                    "{WHO}: the state is damaged: group {key} does not hold a bar of this call. \
                     {START_OVER}"
                ));
            }
        }
        bars.clock_kind = file.clock_kind;
        bars.high_water = file.high_water;
        for (key, g) in file.groups {
            bars.total_bytes += bars.group_bytes(key.as_str().map_or(0, str::len));
            bars.groups.insert(key, g);
        }
        Ok(bars)
    }

    /// One chunk, in clock order within each group, stopping once `limit`
    /// bars are out: the rows after the one that gave the last of them are
    /// not read, so the state is the state after that row, wherever the
    /// chunk ends.
    ///
    /// # Errors
    ///
    /// A row the call cannot take, named: a null clock, a step back, a
    /// weight other than 0 or 1, a negative quantity under `"sum"`, a
    /// missing or non-positive threshold, a group past the caps.
    pub fn feed(&mut self, df: &DataFrame, limit: Option<usize>) -> PolarsResult<DataFrame> {
        let n = df.height();
        let base = self.rows_fed;
        let instants = read_instants(df, &self.cfg.clock, base)?;
        let weights = match &self.cfg.weight {
            Some(w) => Some(df.column(w)?.cast(&DataType::Float64)?),
            None => None,
        };
        let weights = weights.as_ref().map(|c| c.f64()).transpose()?;
        let quantity = match &self.cfg.quantity {
            Some(q) => Some(Quantity::read(df.column(q)?)?),
            None => None,
        };
        let theta_col = match &self.cfg.threshold {
            Threshold::Column(c) => Some(df.column(c)?.cast(&DataType::Float64)?),
            Threshold::Value(_) => None,
        };
        let theta_col = theta_col.as_ref().map(|c| c.f64()).transpose()?;
        let keys = match &self.cfg.group {
            Some(g) => Some(crate::arrow::key_text(
                df.column(g)?.as_materialized_series(),
            )?),
            None => None,
        };
        let keys = keys.as_ref().map(|s| s.str()).transpose()?;
        let sessions = match &self.cfg.session {
            Some(s) => Some(crate::arrow::key_text(
                df.column(s)?.as_materialized_series(),
            )?),
            None => None,
        };
        let sessions = sessions.as_ref().map(|s| s.str()).transpose()?;
        let readers: Vec<Option<Reader>> = self
            .cfg
            .aggregates
            .iter()
            .map(|a| match &a.column {
                Some(c) => Reader::read(df.column(c)?).map(Some),
                None => Ok(None),
            })
            .collect::<PolarsResult<_>>()?;
        let integer = self.group_dtype.as_ref().is_some_and(integer_key);

        let mut out: Vec<Out> = Vec::new();
        let mut consumed = n;
        let full = |out: &Vec<Out>| limit.is_some_and(|l| out.len() >= l);
        for (row, &t) in instants.iter().enumerate() {
            if full(&out) {
                consumed = row;
                break;
            }
            let at = base + row;
            let Some(t) = t else {
                polars_bail!(ComputeError:
                    "{WHO}: clock column {:?} is null, NaN or infinite at row {at} (counting from \
                     0). Drop the rows whose clock has no value, or fill it in.", self.cfg.clock);
            };
            match self.clock_kind {
                Some(k) if k != t.kind() => polars_bail!(ComputeError:
                    "{WHO}: clock column {:?} is {}, but the state it resumes was saved on a {} \
                     clock. Resume the state on a clock of the kind it was saved with.",
                    self.cfg.clock, t.kind().name(), k.name()),
                _ => self.clock_kind = Some(t.kind()),
            }
            let key = GroupKey(match keys {
                Some(k) => k.get(row).map(str::to_string),
                None => Some(String::new()),
            });
            let weight = match weights {
                None => 1.0,
                Some(w) => match w.get(row) {
                    None => 0.0,
                    Some(x) if x == 0.0 || x == 1.0 => x,
                    Some(x) => polars_bail!(ComputeError:
                        "{WHO}: weight column {:?} is {x} at row {at} (counting from 0), but a \
                         bar's weight is 0 or 1: a row is in its bar or not, since a count or a \
                         high has no meaning for a fraction of a row. For a weighted count, sum \
                         the weight column with kind=\"sum\".",
                        self.cfg.weight.as_deref().unwrap_or("")),
                },
            };
            if self.cfg.monotone {
                if key.as_str().is_none() {
                    polars_bail!(ComputeError:
                        "{WHO}: row {at} (counting from 0) has a null group key, but \
                         group_close=\"monotone\" closes groups in the order of their keys, and a \
                         null has no place in that order. Give every row a group key.");
                }
                match &self.high_water {
                    Some(h) if key_cmp(&key, h, integer) == Ordering::Less => polars_bail!(
                        ComputeError:
                        "{WHO}: row {at} (counting from 0) has group {key} after group {h}, but \
                         group_close=\"monotone\" needs the keys in ascending order, as numbers \
                         for an integer column and as text otherwise. Sort the rows by {:?}.",
                        self.cfg.group.as_deref().unwrap_or("")),
                    Some(h) if key_cmp(&key, h, integer) == Ordering::Equal => {}
                    // A row of weight 0 closes nothing: it is in no clock.
                    _ if weight == 0.0 => {}
                    _ => {
                        // A greater key: every group before it is closed, its
                        // open bar cut and its state dropped.
                        let mut closed: Vec<GroupKey> = self.groups.keys().cloned().collect();
                        closed.sort();
                        for k in closed {
                            if let Some(mut g) = self.groups.remove(&k) {
                                self.total_bytes = self.total_bytes.saturating_sub(
                                    self.group_bytes(k.as_str().map_or(0, str::len)),
                                );
                                if let Some(o) = self.cut(&k, &mut g) {
                                    out.push(o);
                                }
                            }
                        }
                        self.high_water = Some(key.clone());
                        if full(&out) {
                            consumed = row;
                            break;
                        }
                    }
                }
            }
            if !self.groups.contains_key(&key) {
                if weight == 0.0 {
                    // A row of weight 0 in a group not yet seen: in no bar,
                    // and nothing to order it against.
                    continue;
                }
                self.admit(&key, at)?;
            }
            let cfg = &self.cfg;
            let g = self.groups.get_mut(&key).expect("admitted above");
            // The order check, for every row whatever its weight, against
            // the group's last row of any weight; and for a row of positive
            // weight, against its last row of positive weight too, since a
            // row of weight 0 is in no clock: a step back from it past
            // `restart_after_step_back` starts the stream over.
            let refuse = |prev: Instant| -> PolarsResult<bool> {
                if t.partial_cmp(&prev) != Some(Ordering::Less) {
                    return Ok(false);
                }
                if cfg
                    .restart_after_step_back
                    .is_some_and(|r| span_past(prev, t, r))
                {
                    return Ok(true);
                }
                polars_bail!(ComputeError:
                "{WHO}: clock column {:?} is {} at row {at} (counting from 0), below the \
                 previous row's {} in the same group. Sort the input by the clock{}.",
                cfg.clock, t.show(&self.clock_dtype), prev.show(&self.clock_dtype),
                if cfg.restart_after_step_back.is_some() {
                    ", or raise restart_after_step_back if the clock starts over here"
                } else {
                    ", or give restart_after_step_back if the clock starts over on purpose"
                });
            };
            if let Some(prev) = g.last_clock {
                refuse(prev)?;
            }
            g.last_clock = Some(t);
            if weight == 0.0 {
                continue;
            }
            let restart = match g.time_last {
                Some(last) => refuse(last)?,
                None => false,
            };
            let session = sessions.map(|s| session_hash(s.get(row)));
            // The cuts, each optional: the open bar ends before this row.
            let cut = restart
                || (session.is_some() && g.time_last.is_some() && session != g.session)
                || matches!((cfg.gap_cap, g.time_last), (Some(c), Some(last)) if span_past(t, last, c))
                || matches!((cfg.max_span, g.time_open), (Some(c), Some(open)) if span_past(t, open, c));
            if cut {
                let mut taken = std::mem::replace(g, Group::placeholder());
                let o = self.cut(&key, &mut taken);
                *self.groups.get_mut(&key).expect("present") = taken;
                if let Some(o) = o {
                    out.push(o);
                    if full(&out) {
                        consumed = row;
                        break;
                    }
                }
            }
            let cfg = &self.cfg;
            let g = self.groups.get_mut(&key).expect("present");
            let theta = match (&cfg.threshold, theta_col) {
                (Threshold::Value(v), _) => *v,
                (Threshold::Column(c), Some(col)) => {
                    let v = col.get(row).unwrap_or(f64::NAN);
                    check_threshold(cfg.kind, v).map_err(|why| {
                        polars_err!(ComputeError:
                            "{WHO}: threshold column {c:?} is {v} at row {at} (counting from 0), \
                             but {why}. Fill in or correct the threshold there.")
                    })?;
                    v
                }
                _ => unreachable!("a threshold column is read"),
            };
            let mut q = [0.0; 2];
            let parts = match &quantity {
                Some(qc) => qc.at(row, &mut q),
                None => 0,
            };
            let closed = g.crossing.step(&q[..parts], theta).map_err(|r| match r {
                CrossRefused::Negative => polars_err!(ComputeError:
                    "{WHO}: quantity {:?} is {} at row {at} (counting from 0), below 0, but \
                     kind=\"sum\" sums a quantity that only grows. Use kind=\"imbalance\" for a \
                     signed quantity, or correct the row.",
                    cfg.quantity.as_deref().unwrap_or(""), q[..parts].iter().sum::<f64>()),
                CrossRefused::TooMany => polars_err!(ComputeError:
                    "{WHO}: the row at {at} (counting from 0) crosses 2^53 thresholds or more: the \
                     threshold {theta} is below 2^-53 of the row's quantity. Raise the threshold."),
            })?;
            if g.rows == 0 {
                g.time_open = Some(t);
            }
            g.rows += 1;
            g.time_last = Some(t);
            g.session = session;
            for (acc, r) in g.aggs.iter_mut().zip(&readers) {
                acc.update(r.as_ref().map_or(Val::Null, |r| r.at(row)));
            }
            if let Some(m) = closed {
                let fresh = self.fresh_aggs();
                let g = self.groups.get_mut(&key).expect("present");
                out.push(Out {
                    key: key.clone(),
                    bar: g.bar,
                    open: g.time_open.take().expect("a bar holds a row"),
                    close: t,
                    thresholds: m,
                    carry: g.crossing.carry(),
                    complete: true,
                    aggs: std::mem::replace(&mut g.aggs, fresh),
                });
                g.bar += 1;
                g.rows = 0;
            }
        }
        self.rows_fed += consumed;
        self.frame(out)
    }

    /// The open bars, at the input's end: each emitted with `complete =
    /// false`, in the order of the groups' keys, and the state left with
    /// none open.
    pub fn finish(&mut self) -> PolarsResult<DataFrame> {
        let mut keys: Vec<GroupKey> = self.groups.keys().cloned().collect();
        keys.sort();
        let mut out = Vec::new();
        for k in keys {
            let mut g = self.groups.remove(&k).expect("listed");
            if g.rows > 0 {
                let fresh = self.fresh_aggs();
                out.push(Out {
                    key: k.clone(),
                    bar: g.bar,
                    open: g.time_open.take().expect("a bar holds a row"),
                    close: g.time_last.expect("a bar holds a row"),
                    thresholds: 0,
                    carry: g.crossing.carry(),
                    complete: false,
                    aggs: std::mem::replace(&mut g.aggs, fresh),
                });
                g.bar += 1;
                g.rows = 0;
            }
            g.crossing.reset();
            self.groups.insert(k, g);
        }
        self.frame(out)
    }

    fn fresh_aggs(&self) -> Vec<Acc> {
        self.cfg
            .aggregates
            .iter()
            .zip(&self.agg_dtypes)
            .map(|(a, d)| Acc::new(a.kind, d.as_ref().and_then(class_of)))
            .collect()
    }

    /// A new group, within the caps.
    fn admit(&mut self, key: &GroupKey, at: usize) -> PolarsResult<()> {
        let bytes = self.group_bytes(key.as_str().map_or(0, str::len));
        if bytes > self.cfg.max_state_bytes {
            polars_bail!(ComputeError:
                "{WHO}: group {key} at row {at} (counting from 0) holds up to {bytes} bytes, past \
                 max_state_bytes of {}: its key is {} bytes long. Raise max_state_bytes, or use a \
                 shorter group key.",
                self.cfg.max_state_bytes, key.as_str().map_or(0, str::len));
        }
        let total = self.total_bytes + bytes;
        if total > self.cfg.max_total_bytes {
            polars_bail!(ComputeError:
                "{WHO}: group {key} at row {at} (counting from 0) is group {} of the stream, and \
                 the groups would hold up to {total} bytes, past max_total_bytes of {}. Raise \
                 max_total_bytes, or close the groups that end (group_close=\"monotone\" for a key \
                 that only grows, such as a day).",
                self.groups.len() + 1, self.cfg.max_total_bytes);
        }
        if !self.warned_half && total > self.cfg.max_total_bytes / 2 {
            self.warned_half = true;
            self.notices.push(format!(
                "{WHO}: the groups hold up to {total} bytes, past half of max_total_bytes ({}), \
                 and each new group adds up to {bytes}. At {} bytes the run stops and refuses the \
                 group. Raise max_total_bytes, or close the groups that end, to stay clear of it.",
                self.cfg.max_total_bytes, self.cfg.max_total_bytes
            ));
        }
        self.total_bytes = total;
        let g = Group {
            crossing: Crossing::new(self.cfg.kind),
            bar: 0,
            rows: 0,
            time_open: None,
            time_last: None,
            last_clock: None,
            session: None,
            aggs: self.fresh_aggs(),
        };
        self.groups.insert(key.clone(), g);
        Ok(())
    }

    /// Cut the open bar of `g` short: emitted, unless empty or under
    /// `partial="drop"`, with `complete = false`; the remainder goes, since a
    /// cut ends the stretch the carry counts over. Done twice it does
    /// nothing the second time, which is what lets a limited feed stop
    /// between a cut and its row.
    fn cut(&self, key: &GroupKey, g: &mut Group) -> Option<Out> {
        g.crossing.reset();
        if g.rows == 0 {
            return None;
        }
        let aggs = std::mem::replace(&mut g.aggs, self.fresh_aggs());
        let open = g.time_open.take().expect("a bar holds a row");
        g.rows = 0;
        if !self.cfg.keep_partial {
            return None;
        }
        let o = Out {
            key: key.clone(),
            bar: g.bar,
            open,
            close: g.time_last.expect("a bar holds a row"),
            thresholds: 0,
            carry: 0.0,
            complete: false,
            aggs,
        };
        g.bar += 1;
        Some(o)
    }

    /// The bars as a frame of [`Self::schema`].
    fn frame(&self, out: Vec<Out>) -> PolarsResult<DataFrame> {
        let h = out.len();
        let mut cols: Vec<Column> = Vec::new();
        if let (Some(g), Some(d)) = (&self.cfg.group, &self.group_dtype) {
            let text: Vec<Option<&str>> = out.iter().map(|o| o.key.as_str()).collect();
            let s = Series::new(g.as_str().into(), text);
            let s = match d {
                DataType::Categorical(..) => s.cast(d)?,
                _ if d.is_integer() => s.strict_cast(&DataType::Int64)?.strict_cast(d)?,
                _ => s.strict_cast(d)?,
            };
            cols.push(s.into_column());
        }
        cols.push(Column::new(
            "bar".into(),
            out.iter().map(|o| o.bar as i64).collect::<Vec<_>>(),
        ));
        cols.push(instants_column(
            "time_open",
            out.iter().map(|o| o.open),
            &self.clock_dtype,
        )?);
        cols.push(instants_column(
            "time_close",
            out.iter().map(|o| o.close),
            &self.clock_dtype,
        )?);
        cols.push(Column::new(
            "thresholds".into(),
            out.iter()
                .map(|o| i64::try_from(o.thresholds).unwrap_or(i64::MAX))
                .collect::<Vec<_>>(),
        ));
        if self.cfg.kind == BarKind::Sum {
            cols.push(Column::new(
                "carry".into(),
                out.iter().map(|o| o.carry).collect::<Vec<_>>(),
            ));
        }
        cols.push(Column::new(
            "complete".into(),
            out.iter().map(|o| o.complete).collect::<Vec<_>>(),
        ));
        for (j, (a, d)) in self.cfg.aggregates.iter().zip(&self.agg_dtypes).enumerate() {
            let target = agg_dtype(a.kind, d.as_ref());
            cols.push(acc_column(
                &a.name,
                out.iter().map(|o| &o.aggs[j]),
                d.as_ref(),
                &target,
                &out,
            )?);
        }
        DataFrame::new(h, cols)
    }
}

impl Group {
    /// A stand-in while a group is out of the map.
    fn placeholder() -> Self {
        Self {
            crossing: Crossing::new(BarKind::Count),
            bar: 0,
            rows: 0,
            time_open: None,
            time_last: None,
            last_clock: None,
            session: None,
            aggs: Vec::new(),
        }
    }
}

/// Whether `t` is a threshold of `kind`: finite, above 0, at most 1e100, and
/// under `"count"` a whole number up to 2^53. The reason it is not, if not.
fn check_threshold(kind: BarKind, t: f64) -> Result<(), &'static str> {
    if t.is_nan() || t <= 0.0 || !usable(t) {
        return Err("a threshold must be a number above 0 and at most 1e100");
    }
    if kind == BarKind::Count && (t.fract() != 0.0 || t > 9_007_199_254_740_992.0) {
        return Err("a threshold of kind=\"count\" is a whole number of rows, up to 2^53");
    }
    Ok(())
}

fn agg_name(k: AggKind) -> &'static str {
    match k {
        AggKind::First => "first",
        AggKind::Last => "last",
        AggKind::Min => "min",
        AggKind::Max => "max",
        AggKind::Sum => "sum",
        AggKind::Count => "count",
        AggKind::Len => "len",
    }
}

/// The dtype Polars gives an aggregation of a column of `d`.
fn agg_dtype(k: AggKind, d: Option<&DataType>) -> DataType {
    use DataType::*;
    match (k, d) {
        (AggKind::Count | AggKind::Len, _) => IDX_DTYPE,
        (AggKind::Sum, Some(Int8 | Int16 | UInt8 | UInt16)) => Int64,
        (AggKind::Sum, Some(Boolean)) => IDX_DTYPE,
        (_, Some(d)) => d.clone(),
        (_, None) => Null,
    }
}

/// `later − earlier > span`, exactly, on either kind of clock.
fn span_past(later: Instant, earlier: Instant, span: f64) -> bool {
    match (later, earlier) {
        (Instant::Int(a), Instant::Int(b)) | (Instant::Nanos(a), Instant::Nanos(b)) => {
            rheon_core::cmp_int_f64(i128::from(a) - i128::from(b), span) == Some(Ordering::Greater)
        }
        _ => {
            let mut e = Expansion::new();
            for (x, sign) in [(later, 1i128), (earlier, -1i128)] {
                match x {
                    Instant::Number(v) => e.add(sign as f64 * v),
                    Instant::Int(v) | Instant::Nanos(v) => e.add_int(sign * i128::from(v)),
                }
            }
            e.cmp_f64(span) == Ordering::Greater
        }
    }
}

/// The clock of each row, as the order check compares it: an integer as
/// itself, a temporal value in integer nanoseconds, a float as itself;
/// `None` for a null, NaN or infinite value.
fn read_instants(df: &DataFrame, clock: &str, base: usize) -> PolarsResult<Vec<Option<Instant>>> {
    let col = df.column(clock)?;
    let dtype = col.dtype().clone();
    let n = df.height();
    if dtype.is_temporal() {
        let ns = crate::arrow::nanos_array(
            col.as_materialized_series(),
            &crate::arrow::InputRows::from_base(base),
            crate::arrow::NanosRole::Clock,
            WHO,
        )?;
        return Ok((0..n).map(|i| ns.get(i).map(Instant::Nanos)).collect());
    }
    if dtype == DataType::UInt64 {
        let s = col.as_materialized_series();
        return s
            .u64()?
            .iter()
            .enumerate()
            .map(|(row, v)| {
                v.map(|v| {
                    i64::try_from(v).map(Instant::Int).map_err(|_| {
                        polars_err!(ComputeError:
                            "{WHO}: clock column {clock:?} is {v} at row {} (counting from 0), \
                             past {}, the largest value an integer clock holds (an Int64's). Cast \
                             the clock to Int64 after subtracting an origin from it.",
                            base + row, i64::MAX)
                    })
                })
                .transpose()
            })
            .collect();
    }
    if dtype.is_integer() {
        let s = col.cast(&DataType::Int64)?;
        return Ok(s.i64()?.iter().map(|v| v.map(Instant::Int)).collect());
    }
    let s = col.cast(&DataType::Float64)?;
    Ok(s.f64()?
        .iter()
        .map(|v| v.filter(|t| t.is_finite()).map(Instant::Number))
        .collect())
}

/// Instants written back in the clock's dtype.
fn instants_column(
    name: &str,
    it: impl Iterator<Item = Instant>,
    dtype: &DataType,
) -> PolarsResult<Column> {
    let v: Vec<Instant> = it.collect();
    let s = if dtype.is_temporal() {
        let ns: Vec<i64> = v
            .iter()
            .map(|i| match *i {
                Instant::Nanos(x) | Instant::Int(x) => x,
                Instant::Number(x) => x as i64,
            })
            .collect();
        let s = Series::new(name.into(), ns);
        match dtype {
            DataType::Datetime(_, tz) => s
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, tz.clone()))?
                .cast(dtype)?,
            DataType::Duration(_) => s
                .cast(&DataType::Duration(TimeUnit::Nanoseconds))?
                .cast(dtype)?,
            _ => s
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))?
                .cast(dtype)?,
        }
    } else if dtype.is_integer() {
        let ints: Vec<i64> = v
            .iter()
            .map(|i| match *i {
                Instant::Int(x) | Instant::Nanos(x) => x,
                Instant::Number(x) => x as i64,
            })
            .collect();
        Series::new(name.into(), ints).strict_cast(dtype)?
    } else {
        let fl: Vec<f64> = v
            .iter()
            .map(|i| match *i {
                Instant::Number(x) => x,
                Instant::Int(x) | Instant::Nanos(x) => x as f64,
            })
            .collect();
        Series::new(name.into(), fl).cast(dtype)?
    };
    Ok(s.into_column())
}

/// One aggregation's column, from each bar's accumulator.
fn acc_column<'a>(
    name: &str,
    accs: impl Iterator<Item = &'a Acc>,
    input: Option<&DataType>,
    target: &DataType,
    out: &[Out],
) -> PolarsResult<Column> {
    let accs: Vec<&Acc> = accs.collect();
    let vals = |pick: &dyn Fn(&Acc) -> Option<Val>| -> Vec<Option<Val>> {
        accs.iter().map(|a| pick(a)).collect()
    };
    let physical = |vs: Vec<Option<Val>>| -> PolarsResult<Series> {
        let class = input.and_then(class_of);
        let s = match class {
            Some(Class::Float) => Series::new(
                name.into(),
                vs.iter()
                    .map(|v| match v {
                        Some(Val::F(x)) => Some(*x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            ),
            Some(Class::Int) => Series::new(
                name.into(),
                vs.iter()
                    .map(|v| match v {
                        Some(Val::I(x)) => Some(*x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            ),
            Some(Class::UInt) => Series::new(
                name.into(),
                vs.iter()
                    .map(|v| match v {
                        Some(Val::U(x)) => Some(*x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            ),
            Some(Class::Bool) => Series::new(
                name.into(),
                vs.iter()
                    .map(|v| match v {
                        Some(Val::B(x)) => Some(*x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            ),
            None => Series::full_null(name.into(), vs.len(), &DataType::Null),
        };
        match target {
            DataType::Datetime(..) | DataType::Date | DataType::Duration(_) | DataType::Time => {
                s.cast(&target.to_physical())?.cast(target)
            }
            _ => s.cast(target),
        }
    };
    let s = match accs.first() {
        None => Series::new_empty(name.into(), target),
        Some(Acc::First(_)) | Some(Acc::Last(_)) => physical(vals(&|a| match a {
            Acc::First(v) | Acc::Last(v) => v.filter(|v| *v != Val::Null),
            _ => None,
        }))?,
        Some(Acc::Min { .. }) | Some(Acc::Max { .. }) => physical(vals(&|a| match a {
            Acc::Min { best, nan } | Acc::Max { best, nan } => match best {
                Some(v) => Some(*v),
                None if *nan => Some(Val::F(f64::NAN)),
                None => None,
            },
            _ => None,
        }))?,
        Some(Acc::SumFloat { .. }) => Series::new(
            name.into(),
            accs.iter()
                .map(|a| match a {
                    Acc::SumFloat {
                        exact,
                        nan,
                        pos_inf,
                        neg_inf,
                        over,
                    } => {
                        if *nan || (*pos_inf && *neg_inf) {
                            f64::NAN
                        } else if *pos_inf {
                            f64::INFINITY
                        } else if *neg_inf {
                            f64::NEG_INFINITY
                        } else if *over != 0 {
                            f64::from(*over) * f64::INFINITY
                        } else {
                            exact.value()
                        }
                    }
                    _ => f64::NAN,
                })
                .collect::<Vec<_>>(),
        )
        .cast(target)?,
        Some(Acc::SumInt(_)) => {
            let sums: Vec<i128> = accs
                .iter()
                .map(|a| match a {
                    Acc::SumInt(s) => *s,
                    _ => 0,
                })
                .collect();
            let (lo, hi): (i128, i128) = match target {
                DataType::Int32 => (i32::MIN.into(), i32::MAX.into()),
                DataType::UInt32 => (0, u32::MAX.into()),
                DataType::UInt64 => (0, u64::MAX.into()),
                _ => (i64::MIN.into(), i64::MAX.into()),
            };
            if let Some(i) = sums.iter().position(|s| *s < lo || *s > hi) {
                polars_bail!(ComputeError:
                    "{WHO}: aggregate {name:?} of bar {} sums to {}, past the range of {}, the \
                     type Polars gives that sum. Cast the column to a wider type, such as \
                     Float64, before the call.",
                    out[i].bar, sums[i], crate::arrow::dtype_name(target));
            }
            if *target == DataType::UInt64 {
                Series::new(
                    name.into(),
                    sums.iter().map(|s| *s as u64).collect::<Vec<_>>(),
                )
            } else {
                Series::new(
                    name.into(),
                    sums.iter().map(|s| *s as i64).collect::<Vec<_>>(),
                )
                .strict_cast(target)?
            }
        }
        Some(Acc::Count(_)) | Some(Acc::Len(_)) => Series::new(
            name.into(),
            accs.iter()
                .map(|a| match a {
                    Acc::Count(n) | Acc::Len(n) => *n,
                    _ => 0,
                })
                .collect::<Vec<_>>(),
        )
        .strict_cast(target)?,
    };
    Ok(s.into_column())
}

/// A quantity column, read as one or two doubles a row.
enum Quantity {
    Float(Float64Chunked),
    Int(Int64Chunked),
    UInt(UInt64Chunked),
}

impl Quantity {
    fn read(c: &Column) -> PolarsResult<Self> {
        let d = c.dtype();
        Ok(if d.is_unsigned_integer() {
            Self::UInt(c.cast(&DataType::UInt64)?.u64()?.clone())
        } else if d.is_integer() {
            Self::Int(c.cast(&DataType::Int64)?.i64()?.clone())
        } else {
            Self::Float(c.cast(&DataType::Float64)?.f64()?.clone())
        })
    }

    /// The row's quantity into `q`, as the number of doubles it takes: 0 for
    /// a missing value (null, NaN, infinite or past 1e100), which adds 0; two
    /// for an integer past 2^53, its nearest double and the integer left
    /// over.
    fn at(&self, row: usize, q: &mut [f64; 2]) -> usize {
        let int = |v: i128, q: &mut [f64; 2]| {
            let hi = v as f64;
            let lo = (v - hi as i128) as f64;
            q[0] = hi;
            q[1] = lo;
            if lo == 0.0 { 1 } else { 2 }
        };
        match self {
            Self::Float(c) => match c.get(row) {
                Some(v) if usable(v) => {
                    q[0] = v;
                    1
                }
                _ => 0,
            },
            Self::Int(c) => c.get(row).map_or(0, |v| int(i128::from(v), q)),
            Self::UInt(c) => c.get(row).map_or(0, |v| int(i128::from(v), q)),
        }
    }
}

/// An aggregate's input column, read as its physical values.
enum Reader {
    Float(Float64Chunked),
    Int(Int64Chunked),
    UInt(UInt64Chunked),
    Bool(BooleanChunked),
    Null,
}

impl Reader {
    fn read(c: &Column) -> PolarsResult<Self> {
        let d = c.dtype();
        Ok(match class_of(d) {
            Some(Class::Float) => Self::Float(c.cast(&DataType::Float64)?.f64()?.clone()),
            Some(Class::Int) => Self::Int(
                c.as_materialized_series()
                    .to_physical_repr()
                    .cast(&DataType::Int64)?
                    .i64()?
                    .clone(),
            ),
            Some(Class::UInt) => Self::UInt(c.cast(&DataType::UInt64)?.u64()?.clone()),
            Some(Class::Bool) => Self::Bool(c.bool()?.clone()),
            None => Self::Null,
        })
    }

    fn at(&self, row: usize) -> Val {
        match self {
            Self::Float(c) => c.get(row).map_or(Val::Null, Val::F),
            Self::Int(c) => c.get(row).map_or(Val::Null, Val::I),
            Self::UInt(c) => c.get(row).map_or(Val::Null, Val::U),
            Self::Bool(c) => c.get(row).map_or(Val::Null, Val::B),
            Self::Null => Val::Null,
        }
    }
}
