//! `rp.stream.with_windows` over a frame (docs/PLAN.md tasks 78, 143 and
//! 144): formulas over the window operators, run a chunk at a time with the
//! window core, in input order, O(window) memory.
//!
//! A chunk goes through four passes. Polars evaluates every `increment`'s
//! input; this runner takes each increment one row back within its group's
//! session (a step back past `restart_after_step_back` starts it over), as
//! a hidden column of the chunk. Polars evaluates every operator's input --
//! an element-wise formula of the row, increments included -- into a
//! number per row. The core takes the rows, one kernel per distinct
//! setting, every operator on it reading its own input, and gives back
//! the rows it resolved, each with its operators' values. Polars then
//! evaluates the formulas over the held rows with the operators standing
//! in as hidden columns, and the hidden columns are dropped.

use std::borrow::Cow;
#[cfg(test)]
use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Cursor;

use polars::prelude::*;
use rheon_core::{
    ClockCfg, ClockDrift, ClockSteps, ClockValue, RunLength, RunValue, SessionGap, StepStat,
};
use serde::{Deserialize, Serialize};

use crate::arrow::{NanosRole, fits_64, key_text, nanos_array, wide_integer_clock};
use crate::formula::{Formula, Node, OpNode};
use crate::span::{Span, format_duration};
use crate::spec::{ClockPolicy, SessionGapSpec, clock_cfg_of};
use crate::stream::usable;
use crate::windows::{ChunkKeys, session_hashes};
use crate::windows::{
    Closed, CrossClosed, CrossDef, Direction, KernelDef, MomentsDef, OpDef, OpKind, Partial, Peek,
    PhaseSpan, PlainRows, ProfileDef, Refusal, RowIn, StateOf, StlDef, Windows,
};
use crate::wrong_type::{Refused, wrong_type};

/// The plan's definitions of the operators that are not window operators.
mod defs;
#[cfg(test)]
mod load_tests;
use defs::{list_keys, moments_def, neighbor_list, profile_def, stl_def};
mod rows;
use rows::{IncrPolicy, RowOp, RowState, step_increments};

/// The spec behind `like=`: its name, for messages, and the columns whose
/// values must all be usable for it to learn from a row -- its features and
/// its weight, the rule `stream.rs` applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Like {
    pub spec: String,
    pub accept: Vec<String>,
}

/// One `rp.stream.with_windows` call: its formulas and its clock policy, in
/// the spec's own words (`spec.rs`), which is how `like=` fills it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowsConfig {
    pub formulas: Vec<Formula>,
    #[serde(default)]
    pub clock: Option<String>,
    #[serde(default)]
    pub gap_cap: Option<Span>,
    #[serde(default)]
    pub restart_after_step_back: Option<Span>,
    #[serde(default, skip_serializing_if = "crate::spec::is_default_step_back")]
    pub on_step_back: Option<String>,
    #[serde(default)]
    pub session: Option<String>,
    #[serde(default)]
    pub session_gap: Option<SessionGapSpec>,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub like: Option<Like>,
}

impl WindowsConfig {
    /// Refuses a number or a duration as `session_gap`, which no window
    /// reads, since every window ends at a session change: the call a user
    /// makes is held to it, as Python's is. Not [`WindowsRun::new`], which a
    /// spec's formula targets build through with the spec's own
    /// `session_gap`, which its model does read; and under `like=` the spec's
    /// is lent. A state saved by 1.x with a number still loads, under the call
    /// without one.
    ///
    /// # Errors
    ///
    /// A numeric `session_gap` without `like`.
    pub fn refuse_unread_session_gap(&self) -> Result<(), Refused> {
        if let (None, Some(SessionGapSpec::Gap(g))) = (&self.like, &self.session_gap) {
            return Err(format!(
                "{WHO}: session_gap = {g} is no longer read: every window ends at a session \
                 change, so no window measures the clock's step across one. Leave session_gap \
                 out, or set it to \"reset\" to discard the windows still open at a session \
                 change rather than cut them."
            )
            .into());
        }
        Ok(())
    }
}

const WHO: &str = "with_windows";
/// The envelope's magic (`envelope.rs`), the first entry of a state of
/// version 12 or later, and its kind.
const WINDOWS_MAGIC: &str = crate::envelope::MAGIC;
const WINDOWS_KIND: &str = crate::envelope::Kind::Windows.tag();
/// The magic a state began with before the envelope, version 11 and
/// earlier.
const LEGACY_WINDOWS_MAGIC: &str = "polars-online windows";
/// 15 since the placeholder sum kernel's removal (task 78): a call of row
/// operators alone (`increment`, `tick_sign`, `run_length`, `clock_lag`,
/// `clock_drift`, `clock_step`) carries an empty window core (no kernels
/// and no operators) instead of the one-kernel, one-row sum stand-in that
/// versions 9 to 14 wrote. A version-14 (or earlier, down to 9) state of
/// any call with a window operator loads as a 15 unchanged; one of row
/// operators alone carries that stand-in and has it stripped on load
/// ([`LAST_PLACEHOLDER_VERSION`]), continuing to the bit and re-saving as
/// a 15 without it; an older build refuses a 15 by its number rather than
/// at the empty core;
/// 14 since task 564, PLAN 563 and PLAN 490: a group's increments' state may
/// keep a run length's run, a clock drift's sums or a clock step's moments
/// (`rows`), written only where there is a `run_length`, `clock_lag`,
/// `clock_drift` or `clock_step`; a formula may hold an `invert` node, its
/// four parts and its parameters written as the node's list form; and a
/// group may keep an STL or MSTL operator's ring (`seasonal`) and the core
/// its operators (`seasonals`), each written only where there is such an
/// operator. So a version-13 state is a version-14 state without any of them
/// and loads as one; an older build refuses a 14 by its number rather than
/// at the operator's, the node's or the ring's name;
/// 13 since task 561: a kernel may keep a `since` event (`since`), a queue
/// its extrema (`ext`) and its last event's stamp (`since_off`), and an
/// operator its `max_state_bytes`, each written only where there is an
/// extremum, so a version-12 state is a version-13 state without one and
/// loads as one; an older build refuses a 13 by its number rather than at
/// an extremum's name;
/// 12 since PLAN 542: the state is written in the envelope every state
/// file of this library shares (`envelope.rs`): its first entries are the
/// envelope's `magic`, `kind` (`"windows"`) and `version`, and the body is
/// 11's. A state of 9 to 11 begins with [`LEGACY_WINDOWS_MAGIC`] and loads
/// as it did ([`LAST_LEGACY_VERSION`]);
/// 11 since review P7: the state ends with `checksum`, the xxh3-64 of
/// every byte before it (`checksum.rs`), checked before anything is
/// decoded; a version-10 state has none and loads as one
/// ([`SEALED_FROM_VERSION`]);
/// 10 since task 272 (d): a queue keeps a quantile's sketches (`quant`) and
/// an operator its level (`quantile`), each written only where there is a
/// quantile, so a version-9 state is a version-10 state without one and
/// loads as one ([`MIN_WINDOWS_VERSION`]); an older build refuses a 10 by
/// its number rather than at the quantile's name;
/// 9 since task 212: a queue keeps which of its operators are variances
/// (`ewm_var`, `ewm_std`), its arenas six wide for one where every other
/// operator's are three, and an operator keeps its `bias`;
/// 8 since task 200: an integer clock rides in a row's `off` as its own
/// value, beside the form `off` holds, where a bool said nanoseconds or
/// the bits of a double, and an increment of an integer input keeps its
/// previous value as an integer; 7 since task 159 (F2): the group and
/// session columns' dtypes, which a resumed input must match; 6 since task 159 (W3): a number clock's raw
/// value rides in a row's `off`, where 5 left it 0; 5 since review R6
/// (D1): the last row read, beside the rows held, as a
/// sliced state's identity of its input; 4 since review R5 (C5): the skip
/// a sliced state carries counts the rows of its input consumed so far,
/// and the state knows that input by its first clock (round four changed
/// both under 3); 3 since review R2 (W1, W4, W5): a row's raw clock beside
/// its policy time in the queues, and the rows a resume skips; 2 since
/// task 143: formulas over operators, where 1 held descriptions. An older
/// state would load with defaults and misbehave. The bank's schema moves
/// with this number (review R4, A2; R6, D5).
const WINDOWS_VERSION: u32 = 15;

/// The last version whose row-only runs carried a one-kernel, one-sum
/// stand-in core (removed in version 15).
const LAST_PLACEHOLDER_VERSION: u32 = 14;

/// The last version written before the envelope.
const LAST_LEGACY_VERSION: u32 = 11;

/// The first version whose state ends with a checksum: a state of this
/// version or later without one is damaged.
const SEALED_FROM_VERSION: u32 = 11;

/// The oldest windows state this build reads: 9, whose layout is 10's
/// without a quantile (task 272 (d)). A bank file embeds a core per window
/// target, so a bank saved by a build that wrote 9 resumes on this one.
const MIN_WINDOWS_VERSION: u32 = 9;

/// What [`WindowsRun::save_bytes`] writes: the call, the core, the rows the
/// core holds as Arrow IPC with their increment columns, and each group's
/// increment state.
#[derive(Serialize)]
struct FileOut<'a> {
    magic: &'a str,
    kind: &'a str,
    version: u32,
    config: &'a WindowsConfig,
    core: &'a Windows,
    held: &'a [u8],
    increments: &'a [IncrState],
    stream_clock: Option<ClockValue>,
    resume_skip: usize,
    resume_first: Option<ClockValue>,
    /// That row's session hash: with its clock, what tells the saved input
    /// from a next file that starts at the same stamp (review R7, E2).
    resume_first_session: Option<u64>,
    /// The last row the run read, the input's columns as Arrow IPC: with
    /// the rows held, the identity of a sliced state's input (review R6,
    /// D1); empty unless saved under a slice (R7, E3).
    last_row: &'a [u8],
    /// The group and session columns' dtypes, by role (task 159, F2).
    key_dtypes: &'a [(String, String)],
    /// The xxh3-64 of every byte before it (version 11; `checksum.rs`),
    /// last; read from the file's last bytes, never by serde. None only in
    /// a test that encodes a file by hand and seals it after
    /// (`crate::reseal`).
    #[serde(skip_serializing_if = "Option::is_none")]
    checksum: Option<crate::checksum::Trailer>,
}

/// The rest of the file, after [`Header`]'s two fields.
#[derive(Deserialize)]
struct FileIn {
    config: WindowsConfig,
    core: Windows,
    held: Vec<u8>,
    increments: Vec<IncrState>,
    #[serde(default)]
    stream_clock: Option<ClockValue>,
    /// Rows of the input this state was saved under that the run consumed,
    /// for a run resumed on that input: see [`WindowsRun::consumed`]
    /// (review R2, W4; R4, B1).
    #[serde(default)]
    resume_skip: usize,
    /// That input's first row's clock: a resume on an input that starts
    /// elsewhere skips nothing (review R4, B3).
    #[serde(default)]
    resume_first: Option<ClockValue>,
    resume_first_session: Option<u64>,
    last_row: Vec<u8>,
    key_dtypes: Vec<(String, String)>,
}

/// The file's first two fields, read before the rest: a state another
/// version wrote is refused by its version, not by a field it lacks
/// (review R2, W5).
#[derive(Deserialize)]
struct Header {
    magic: String,
    version: u32,
}

/// A group's increment state: the previous row's inputs, session and clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct IncrState {
    key: Option<String>,
    /// Per increment: the previous value, NaN for none; a temporal input in
    /// nanoseconds; an integer input in integers (docs/PLAN.md task 200),
    /// whose step is taken in integers before it becomes a double. Then,
    /// per tick sign in the order of the increments, the sign it carries
    /// over a step of 0: +1, -1, or NaN while none is known. A
    /// run with no tick sign has none of these, so its state is the one an
    /// older build wrote; one with a tick sign names `tick_sign` in its
    /// formulas, which an older build refuses by name.
    prev: Vec<f64>,
    prev_ns: Vec<Option<i64>>,
    prev_int: Vec<Option<i128>>,
    session: Option<u64>,
    clock: Option<ClockValue>,
    /// Per row operator, where any is not an increment (task 564): a run
    /// length's run, or a clock drift's sums and the arrival they are
    /// centred on; empty where every row operator is an increment, and
    /// then not written, so such a state keeps its bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rows: Vec<RowState>,
}

/// The increments' state of a run, borrowed: a run's own, or a copy of it
/// that a caller reads the increments off without moving the run.
struct IncrTrack<'a> {
    state: &'a mut Vec<IncrState>,
    index: &'a mut HashMap<String, usize>,
    null: &'a mut Option<usize>,
    stream_clock: &'a mut Option<ClockValue>,
}

/// A chunk's increments' inputs, each read in the form its step is taken
/// in: integers, doubles or nanoseconds, one `Some` of the three per
/// increment.
#[derive(Clone)]
pub struct IncrInputs {
    rows: usize,
    integer: Vec<Option<Vec<Option<i128>>>>,
    numeric: Vec<Option<Vec<Option<f64>>>>,
    temporal: Vec<Option<Vec<Option<i64>>>>,
}

impl IncrInputs {
    /// The inputs at `rows` of the chunk, in that order.
    fn gather(&self, rows: &[usize]) -> Self {
        fn take<T: Copy>(all: &[Option<Vec<T>>], rows: &[usize]) -> Vec<Option<Vec<T>>> {
            all.iter()
                .map(|v| v.as_ref().map(|v| rows.iter().map(|&i| v[i]).collect()))
                .collect()
        }
        Self {
            rows: rows.len(),
            integer: take(&self.integer, rows),
            numeric: take(&self.numeric, rows),
            temporal: take(&self.temporal, rows),
        }
    }
}

/// What a chunk's groups' runs share, evaluated once for the chunk
/// ([`WindowsRun::share`]): each group is handed its rows' part
/// ([`Shared::of_group`]).
pub struct Shared {
    /// The operator inputs, a double a row, NaN for a null; none where each
    /// is a column of doubles a run reads for itself.
    inputs: Option<Vec<Vec<f64>>>,
    /// The increments' inputs.
    incr: Option<IncrInputs>,
}

impl Shared {
    /// The part of the chunk's values at a group's `rows`.
    pub fn of_group(&self, rows: &[usize]) -> Given {
        Given {
            inputs: self.inputs.as_ref().map(|all| {
                all.iter()
                    .map(|v| rows.iter().map(|&i| v[i]).collect())
                    .collect()
            }),
            incr: self.incr.as_ref().map(|r| r.gather(rows)),
        }
    }
}

/// One group's part of what its chunk shares ([`Shared::of_group`]), handed
/// to [`WindowsRun::feed_resolving_with`].
pub struct Given {
    inputs: Option<Vec<Vec<f64>>>,
    incr: Option<IncrInputs>,
}

/// The tick rule's sign of a row whose step is `step`: +1 up, -1 down, and
/// over a step of 0 the sign of the last step that was not 0, `carried`,
/// which it moves (NaN while no such step has been seen: null). A row with
/// no step -- the first of its group and session, or one whose input is
/// missing -- has no sign and leaves the carried one as it is.
fn tick_sign(step: Option<f64>, carried: &mut f64) -> Option<f64> {
    let step = step?;
    if step > 0.0 {
        *carried = 1.0;
    } else if step < 0.0 {
        *carried = -1.0;
    }
    (!carried.is_nan()).then_some(*carried)
}

#[cfg(test)]
thread_local! {
    /// A test's switch: the lazy engine for every input and formula, as
    /// before the short cuts, to hold the two to the same bits.
    static NO_SHORT_CUTS: Cell<bool> = const { Cell::new(false) };
}

/// Whether a bare column input and a bare operator formula skip the lazy
/// engine (always, but in a test that switches them off).
fn short_cuts() -> bool {
    #[cfg(test)]
    {
        !NO_SHORT_CUTS.with(Cell::get)
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// An operator's values as a column of doubles, null where the value is NaN:
/// the list itself, one pass to find a NaN and, only if there is one, a
/// validity mask built from it (a null's slot reads 0, as a column built from
/// options leaves it).
fn nullable_column(name: &str, mut v: Vec<f64>) -> Column {
    let ca = if v.iter().any(|x| x.is_nan()) {
        let valid = polars_arrow::bitmap::Bitmap::from_iter(v.iter().map(|x| !x.is_nan()));
        for x in &mut v {
            if x.is_nan() {
                *x = 0.0;
            }
        }
        Float64Chunked::from_vec_validity(name.into(), v, Some(valid))
    } else {
        Float64Chunked::from_vec(name.into(), v)
    };
    ca.into_column()
}

/// The hidden column an operator stands in as, in a formula.
/// The `@po:` prefix is kept: it is part of the serialized formula (a rename would break
/// formulas saved by 1.x) and is never shown.
fn hidden(op: &OpNode) -> String {
    format!("@po:{}", op.key())
}

/// The hidden column an operator's input is evaluated into.
fn input_column(i: usize) -> String {
    format!("@in:{i}")
}

/// What the plan made of the call.
struct Plan {
    kernels: Vec<KernelDef>,
    ops: Vec<OpDef>,
    /// Distinct operator inputs, each a formula of the row and the
    /// increments, by the index `OpDef::input` names.
    inputs: Vec<Node>,
    /// Distinct window operators: the hidden column and the core output.
    columns: Vec<(String, usize)>,
    /// Where every formula is nothing but one window operator (or an
    /// increment), each formula's hidden column: named once here, since
    /// naming one writes the operator out as JSON.
    bare: Option<Vec<String>>,
    /// Distinct row operators: increments, run lengths and clock drifts
    /// (task 564), each with its hidden column.
    increments: Vec<RowOp>,
    /// The row operators' distinct inputs: the formula, and whether it is
    /// temporal (an increment in seconds).
    slots: Vec<(Node, bool)>,
    /// Every span a kernel reads, with its parameter's name, for the scale
    /// check.
    spans: Vec<(&'static str, Span)>,
    /// The cross-group operators, after the window operators in output
    /// order, and each one's hidden column.
    cross: Vec<CrossDef>,
    cross_columns: Vec<String>,
    /// The list columns cross operators read their neighbors from, by the
    /// index `CrossDef::neighbors` names.
    lists: Vec<String>,
    /// The moments operators' terms, in operator order.
    moments: Vec<MomentsDef>,
    /// The profile operators, after the window operators in output order,
    /// and each one's hidden column.
    profiles: Vec<ProfileDef>,
    profile_columns: Vec<String>,
    /// The STL and MSTL operators, after the profiles in output order, and
    /// each one's hidden column.
    seasonals: Vec<StlDef>,
    seasonal_columns: Vec<String>,
}

fn dtype_of<'a>(input: &'a Schema, column: &str) -> &'a DataType {
    input.get(column).expect("checked by the plan")
}

fn numeric(dtype: &DataType) -> bool {
    dtype.is_primitive_numeric() || matches!(dtype, DataType::Boolean | DataType::Decimal(..))
}

/// The dtypes `exprs` give over an empty frame with `schema`.
fn dtypes_of(schema: &Schema, exprs: Vec<Expr>) -> Result<Schema, String> {
    DataFrame::empty_with_schema(schema)
        .lazy()
        .with_columns(exprs)
        .collect_schema()
        .map(|s| s.as_ref().clone())
        .map_err(|e| e.to_string())
}

/// The slot of a row operator's input `node` among `slots`, added where it
/// is new: its dtype must be a number or temporal, and `what` names it in a
/// refusal.
fn slot_of(
    slots: &mut Vec<(Node, bool)>,
    input: &Schema,
    name: &str,
    what: &str,
    node: &Node,
) -> Result<usize, Refused> {
    if let Some(i) = slots.iter().position(|(n, _)| n == node) {
        return Ok(i);
    }
    let dtype =
        dtypes_of(input, vec![node.to_expr(&|_| unreachable!()).alias("@t")]).map_err(|e| {
            format!(
                "{WHO}: {name}: Polars cannot compute {what} ({e}). Check that the input reads \
                 columns of the right type."
            )
        })?;
    let dtype = dtype.get("@t").expect("aliased").clone();
    let temporal = dtype.is_temporal();
    if !(numeric(&dtype) || temporal) {
        // "the increment's input" is "an increment's input" here, as it was,
        // and a tick sign's likewise.
        let what = if let Some(rest) = what.strip_prefix("the increment's") {
            format!("an increment's{rest}")
        } else if let Some(rest) = what.strip_prefix("the tick sign's") {
            format!("a tick sign's{rest}")
        } else {
            what.to_string()
        };
        return Err(Refused::wrong_type(format!(
            "{WHO}: {name}: {what} must be a number or a temporal column, but it is {}. Cast \
             the input to a number, such as with .cast(pl.Float64).",
            crate::arrow::dtype_name(&dtype)
        )));
    }
    slots.push((node.clone(), temporal));
    Ok(slots.len() - 1)
}

fn plan(config: &WindowsConfig, input: &Schema) -> Result<Plan, Refused> {
    if config.formulas.is_empty() {
        return Err(format!(
            "{WHO}: no expression (a column computation, such as pl.col(\"x\") + 1) was given. \
             Give at least one that holds a window operator or an increment, such as \
             rp.ops.ewm_mean(pl.col(\"x\"), half_life=10)."
        )
        .into());
    }
    let has = |role: &str, c: &str| -> Result<(), String> {
        if input.contains(c) {
            Ok(())
        } else {
            Err(format!(
                "{WHO}: the {role} column {c:?} is not in the input, whose columns are {:?}. \
                 Name a column the input has, or add {c:?} to it.",
                input.iter_names().map(|n| n.as_str()).collect::<Vec<_>>()
            ))
        }
    };
    let mut names: HashSet<&str> = HashSet::new();
    for f in &config.formulas {
        if f.name.is_empty()
            || f.name.starts_with("@po:")
            || f.name.starts_with("@in:")
            || f.name.contains('\0')
        {
            // A NUL cannot be in a column name of the Arrow export, which
            // panicked on one (review P11).
            return Err(format!(
                "{WHO}: an output cannot be named {:?}: a name must not be empty, start with \
                 \"@po:\" or \"@in:\", or hold a NUL character (\\0). Name the expression (a column computation, such as \
                 pl.col(\"x\") + 1) with .alias() or a keyword argument, such as \
                 with_windows(lf, m=...).",
                f.name
            )
            .into());
        }
        if input.contains(&f.name) {
            return Err(format!(
                "{WHO}: the output {:?} is already a column of the input. An expression (a \
                 column computation, such as pl.col(\"x\") + 1) given by position is named after \
                 its leftmost column unless .alias() names it. Give it another name with \
                 .alias() or a keyword argument, such as with_windows(lf, m=...).",
                f.name
            )
            .into());
        }
        if !names.insert(&f.name) {
            return Err(format!(
                "{WHO}: two expressions (column computations, such as pl.col(\"x\") + 1) are \
                 named {:?}. Give each a name of its own with .alias() or a keyword argument, such as \
                 with_windows(lf, m=...).",
                f.name
            ).into());
        }
        // Each formula holds an operator (one rule with the Python side,
        // review R1, N7), or an invert, the library's own node (PLAN 563),
        // and reads no column under a reserved prefix (N8).
        if f.tree.operators().is_empty() && !f.tree.contains_call("invert") {
            return Err(format!(
                "{WHO}: the expression {:?} (a column computation, such as pl.col(\"x\") + 1) \
                 holds no window operator or increment, and a window run computes only those. \
                 Compute an expression of the row alone with Polars' with_columns.",
                f.name
            )
            .into());
        }
        for c in f.tree.columns() {
            if c.starts_with("@po:") || c.starts_with("@in:") {
                return Err(format!(
                    "{WHO}: the expression {:?} (a column computation, such as pl.col(\"x\") + \
                     1) reads column {c:?}, but a name that starts with \"@po:\" or \"@in:\" is \
                     kept for the window run's own columns. Rename the column before the call.",
                    f.name
                )
                .into());
            }
            has("input", &c)?;
        }
    }
    if config
        .formulas
        .iter()
        .all(|f| f.tree.operators().is_empty())
    {
        return Err(format!(
            "{WHO}: no expression holds a window operator or increment, and an invert reads \
             the row alone, so the run has no window to compute. Add the inverts to a run that \
             computes a window operator, such as a running mean of a column the run reads."
        )
        .into());
    }
    let mut p = Plan {
        kernels: Vec::new(),
        ops: Vec::new(),
        inputs: Vec::new(),
        columns: Vec::new(),
        bare: None,
        increments: Vec::new(),
        slots: Vec::new(),
        spans: Vec::new(),
        cross: Vec::new(),
        cross_columns: Vec::new(),
        lists: Vec::new(),
        moments: Vec::new(),
        profiles: Vec::new(),
        profile_columns: Vec::new(),
        seasonals: Vec::new(),
        seasonal_columns: Vec::new(),
    };
    // Row operators first: an operator's input may read an increment.
    for f in &config.formulas {
        for op in f.tree.operators() {
            if !op.kind.is_row() {
                continue;
            }
            let key = hidden(op);
            if p.increments.iter().any(|r| r.key == key) {
                continue;
            }
            let (its, outer) = match op.kind {
                OpKind::TickSign => ("a tick sign's", "the tick sign"),
                _ => ("an increment's", "the increment"),
            };
            if let Some(inner) = op.input.operators().iter().find(|o| o.kind.is_step()) {
                let (holds, inner) = match (inner.kind == op.kind, inner.kind) {
                    (true, OpKind::TickSign) => ("another tick sign", "the inner tick sign"),
                    (true, _) => ("another increment", "the inner increment"),
                    (false, OpKind::TickSign) => ("a tick sign", "the tick sign"),
                    (false, _) => ("an increment", "the increment"),
                };
                return Err(format!(
                    "{WHO}: {}: {its} input holds {holds}, but it must be an expression (a \
                     column computation, such as pl.col(\"x\") + 1) of the row's own columns. \
                     Compute {inner} in one with_windows call, and {outer} of its output in a \
                     second.",
                    f.name
                )
                .into());
            }
            let what = match op.kind {
                OpKind::Increment => "the increment's input",
                OpKind::TickSign => "the tick sign's input",
                OpKind::RunLength => "the run length's input",
                OpKind::ClockStep => "the clock step's time",
                _ => "the arrival time",
            };
            let input_slot = slot_of(&mut p.slots, input, &f.name, what, &op.input)?;
            let event = match &op.event {
                Some(e) => {
                    let ei = slot_of(&mut p.slots, input, &f.name, "the event time", e)?;
                    if p.slots[ei].1 != p.slots[input_slot].1 {
                        return Err(format!(
                            "{WHO}: {}: {}: the arrival time and the event time must both be \
                             temporal or both be numbers, but one is temporal and the other is \
                             not. Give both as Datetime columns, or both as numbers in one unit.",
                            f.name,
                            op.kind.name()
                        )
                        .into());
                    }
                    Some(ei)
                }
                None => None,
            };
            let half_life = match &op.half_life {
                Some(h) => {
                    p.spans.push(("half_life", h.clone()));
                    h.value()
                }
                None => f64::INFINITY,
            };
            // A temporal input's run compares nanoseconds, so its `tol`,
            // given in seconds, is read in them.
            let tol = if p.slots[input_slot].1 {
                op.tol * 1e9
            } else {
                op.tol
            };
            p.increments.push(RowOp {
                key,
                kind: op.kind,
                input: input_slot,
                event,
                tol,
                half_life,
                step: op.step,
                bias: op.bias,
            });
        }
    }
    let incr_schema = {
        let mut s = input.clone();
        for r in &p.increments {
            s.insert(r.key.as_str().into(), DataType::Float64);
        }
        s
    };
    let resolve_incr = |op: &OpNode| col(hidden(op).as_str());
    // An input of a cross operator, its value or its weight: a number,
    // each distinct formula read once.
    let cross_input = |p: &mut Plan, f: &str, name: &str, what: &str, node: &Node| {
        if let Some(i) = p.inputs.iter().position(|n| n == node) {
            return Ok(i);
        }
        let dtype = dtypes_of(&incr_schema, vec![node.to_expr(&resolve_incr).alias("@t")])
            .map_err(|e| {
                format!(
                    "{WHO}: {f}: Polars cannot compute {name}'s {what} ({e}). Check that the \
                     {what} reads columns of the right type."
                )
            })?;
        let dtype = dtype.get("@t").expect("aliased").clone();
        if !numeric(&dtype) {
            return Err(Refused::wrong_type(format!(
                "{WHO}: {f}: {name}'s {what} must be a number, but it is {}. Cast it to a \
                 number, such as with .cast(pl.Float64).",
                crate::arrow::dtype_name(&dtype)
            )));
        }
        p.inputs.push(node.clone());
        Ok(p.inputs.len() - 1)
    };
    for f in &config.formulas {
        for op in f.tree.operators() {
            if op.kind.is_row() {
                continue;
            }
            let key = hidden(op);
            if p.columns.iter().any(|(k, _)| *k == key)
                || p.cross_columns.contains(&key)
                || p.profile_columns.contains(&key)
                || p.seasonal_columns.contains(&key)
            {
                continue;
            }
            if let Some(stat) = op.kind.cross() {
                let name = op.kind.name();
                if config.clock.is_none() {
                    return Err(format!(
                        "{WHO}: {}: {name} reduces every group's row at one stamp of the clock, \
                         but the call names no clock column, and a row count gives each row a \
                         stamp of its own. Name the clock column with clock=.",
                        f.name
                    )
                    .into());
                }
                let neighbors = match &op.neighbors {
                    Some(c) => Some(neighbor_list(&mut p, config, input, &f.name, name, c)?),
                    None => None,
                };
                let input = cross_input(&mut p, &f.name, name, "input", &op.input)?;
                let weight = match &op.weight {
                    Some(w) => Some(cross_input(&mut p, &f.name, name, "weight", w)?),
                    None => None,
                };
                let pick = match &op.pick {
                    Some(w) => Some(cross_input(&mut p, &f.name, name, "where", w)?),
                    None => None,
                };

                p.cross.push(CrossDef {
                    stat,
                    closed: if op.closed == Closed::Left {
                        CrossClosed::Left
                    } else {
                        CrossClosed::Right
                    },
                    input,
                    weight,
                    bias: op.bias,
                    cap: op
                        .max_state_bytes
                        .unwrap_or(crate::windows::CROSS_STATE_BYTES),
                    rank: op.order.rank,
                    descending: op.order.descending,
                    quantile: op.quantile.unwrap_or(0.5),
                    interpolation: op.order.interpolation,
                    neighbors,
                    pick,
                });
                p.cross_columns.push(key);
                continue;
            }
            if let Some(params) = &op.stl {
                let def = stl_def(&mut p, &f.name, op, params, &incr_schema)?;
                p.seasonals.push(def);
                p.seasonal_columns.push(key);
                continue;
            }
            if let Some(params) = &op.profile {
                let def = profile_def(config, input, &mut p, &f.name, op, params, &incr_schema)?;
                p.profiles.push(def);
                p.profile_columns.push(key);
                continue;
            }
            let direction = op.direction();
            let stat = op.kind.stat().expect("a window operator");
            let half_life = op.half_life.clone().expect("checked when read");
            if !stat.is_extremum() {
                p.spans.push(("half_life", half_life.clone()));
            }
            if let Some(w) = &op.window_size {
                p.spans.push(("window_size", w.clone()));
            }
            let kernel = KernelDef {
                direction,
                half_life: half_life.value(),
                window_size: op.window_size.as_ref().map(Span::value),
                window_ns: match &op.window_size {
                    Some(Span::Duration(d)) => Some(d.nanos),
                    _ => None,
                },
                closed: op.closed,
                since: match &op.since {
                    Some(e) => Some(since_input(&mut p, e, &incr_schema, &f.name, op)?),
                    None => None,
                },
            };
            kernel
                .check()
                .map_err(|e| format!("{WHO}: {}: {}: {e}", f.name, op.kind.name()))?;
            let ki = match p.kernels.iter().position(|k| *k == kernel) {
                Some(i) => i,
                None => {
                    p.kernels.push(kernel.clone());
                    p.kernels.len() - 1
                }
            };
            let mut input_of = |node: &Node| -> Result<usize, Refused> {
                if let Some(i) = p.inputs.iter().position(|n| n == node) {
                    return Ok(i);
                }
                let dtype = dtypes_of(&incr_schema, vec![node.to_expr(&resolve_incr).alias("@t")])
                    .map_err(|e| {
                        format!(
                            "{WHO}: {}: Polars cannot compute {}'s input ({e}). Check that the \
                         input reads columns of the right type.",
                            f.name,
                            op.kind.name()
                        )
                    })?;
                let dtype = dtype.get("@t").expect("aliased").clone();
                if !numeric(&dtype) {
                    return Err(Refused::wrong_type(format!(
                        "{WHO}: {}: {}'s input must be a number, but it is {}. Cast \
                         the input to a number, such as with .cast(pl.Float64).",
                        f.name,
                        op.kind.name(),
                        crate::arrow::dtype_name(&dtype)
                    )));
                }
                p.inputs.push(node.clone());
                Ok(p.inputs.len() - 1)
            };
            let input_i = match (&op.moments, &*op.input) {
                (Some(t), Node::Call(_, legs)) => {
                    let def = moments_def(f.name.as_str(), &kernel, t, legs, &mut input_of)?;
                    for lag in t.lags.iter().filter(|l| l.value() != 0.0) {
                        p.spans.push(("lag", lag.clone()));
                    }
                    if let Some(tol) = &t.lag_tolerance {
                        p.spans.push(("lag_tolerance", tol.clone()));
                    }
                    let first = def.inputs[0];
                    p.moments.push(MomentsDef {
                        op: p.ops.len(),
                        ..def
                    });
                    first
                }
                _ => input_of(&op.input)?,
            };
            p.ops.push(OpDef {
                kernel: ki,
                stat,
                input: input_i,
                min_samples: op.min_samples,
                partial: op.partial.unwrap_or(match direction {
                    Direction::Backward => Partial::Keep,
                    Direction::Forward => Partial::Null,
                }),
                bias: op.bias,
                quantile: op.quantile,
                max_state_bytes: op.max_state_bytes,
            });
            p.columns.push((key, p.ops.len() - 1));
        }
    }
    // The cross operators' outputs follow the window operators', and the
    // profile operators' follow the cross operators'.
    for (ci, key) in std::mem::take(&mut p.cross_columns).into_iter().enumerate() {
        p.columns.push((key, p.ops.len() + ci));
    }
    for (pi, key) in std::mem::take(&mut p.profile_columns)
        .into_iter()
        .enumerate()
    {
        p.columns.push((key, p.ops.len() + p.cross.len() + pi));
    }
    for (si, key) in std::mem::take(&mut p.seasonal_columns)
        .into_iter()
        .enumerate()
    {
        p.columns
            .push((key, p.ops.len() + p.cross.len() + p.profiles.len() + si));
    }
    debug_assert!(
        !(p.ops.is_empty()
            && p.increments.is_empty()
            && p.cross.is_empty()
            && p.profiles.is_empty()
            && p.seasonals.is_empty()),
        "checked per formula"
    );
    p.bare = config
        .formulas
        .iter()
        .map(|f| f.tree.bare_operator().map(hidden))
        .collect();
    Ok(p)
}

/// The row input an extremum's `since` reads (task 561): an element-wise
/// boolean, read as 1.0 where true, among the plan's inputs.
fn since_input(
    p: &mut Plan,
    e: &Node,
    schema: &Schema,
    name: &str,
    op: &OpNode,
) -> Result<usize, Refused> {
    if let Some(i) = p.inputs.iter().position(|n| n == e) {
        return Ok(i);
    }
    let dtype =
        dtypes_of(schema, vec![e.to_expr(&|_| unreachable!()).alias("@t")]).map_err(|err| {
            format!(
                "{WHO}: {name}: Polars cannot compute {}'s since ({err}). Check that since reads \
             columns of the right type.",
                op.kind.name()
            )
        })?;
    let dtype = dtype.get("@t").expect("aliased").clone();
    if dtype != DataType::Boolean {
        return Err(Refused::wrong_type(format!(
            "{WHO}: {name}: {}'s since must be a boolean, true at a row that starts the window \
             over, but it is {}. Compare the column to a value, such as pl.col(\"period\") != \
             pl.col(\"period\").shift() computed before the call.",
            op.kind.name(),
            crate::arrow::dtype_name(&dtype)
        )));
    }
    p.inputs.push(e.clone());
    Ok(p.inputs.len() - 1)
}

/// The kind of clock a run is fed. A state resumes only on the kind that
/// wrote it: the other cannot be ordered against its last row, and an
/// integer clock's edges are decided in integers, a float clock's in
/// doubles (docs/PLAN.md task 200).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClockKind {
    Numeric,
    Temporal,
    Integer,
}

impl ClockKind {
    fn name(self) -> &'static str {
        match self {
            ClockKind::Numeric => "a float",
            ClockKind::Temporal => "a temporal",
            ClockKind::Integer => "an integer",
        }
    }
}

fn kind_of(v: ClockValue) -> ClockKind {
    match v {
        ClockValue::F64(_) => ClockKind::Numeric,
        ClockValue::Ns(_) => ClockKind::Temporal,
        ClockValue::I64(_) => ClockKind::Integer,
    }
}

/// Whether the clock stepped back from `prev` to `now` by more than
/// `restart`, which the policy reads as a new start: on an integer clock
/// the step in integers against it, exactly (task 200).
fn back_past(prev: ClockValue, now: ClockValue, restart: f64) -> bool {
    if let Some(step) = now.int_step(prev) {
        return step < 0
            && rheon_core::cmp_int_f64(-step, restart) == Some(std::cmp::Ordering::Greater);
    }
    let back = match (prev, now) {
        (ClockValue::Ns(p), ClockValue::Ns(c)) => {
            rheon_core::seconds_of_ns(i128::from(p) - i128::from(c))
        }
        (p, c) => p.seconds() - c.seconds(),
    };
    back > 0.0 && back > restart
}

/// An integer column's values, as `i128` so an unsigned 64-bit one is
/// whole too, and an `Int128` (review round 5, B3); null where null.
fn integer_values(s: &Series) -> PolarsResult<Vec<Option<i128>>> {
    Ok(match s.dtype() {
        DataType::UInt64 => s.u64()?.iter().map(|v| v.map(i128::from)).collect(),
        DataType::Int128 => s.i128()?.iter().collect(),
        _ => s
            .cast(&DataType::Int64)?
            .i64()?
            .iter()
            .map(|v| v.map(i128::from))
            .collect(),
    })
}

/// Whether an increment's input is read as its integers: every integer
/// dtype [`integer_values`] holds whole, the 64-bit forms and an `Int128`.
fn integer_input(dtype: &DataType) -> bool {
    fits_64(dtype) || *dtype == DataType::Int128
}

/// `now − prev` as a double, rounded once: the difference of two `i128`s
/// fits a `u128` either way round, where `i128` subtraction overflows
/// between the type's two ends.
fn integer_step(now: i128, prev: i128) -> f64 {
    let size = now.abs_diff(prev) as f64;
    if now >= prev { size } else { -size }
}

/// Durations and numbers must match the clock, as a spec's must
/// (docs/PLAN.md task 88): a duration needs a temporal clock, a number a
/// numeric one, and a duration below the clock's resolution is refused.
fn check_scale(
    config: &WindowsConfig,
    spans: &[(&'static str, Span)],
    input: &Schema,
) -> Result<(), String> {
    let mut all: Vec<(&str, &Span)> = spans.iter().map(|(n, s)| (*n, s)).collect();
    all.extend(config.gap_cap.iter().map(|s| ("gap_cap", s)));
    all.extend(
        config
            .restart_after_step_back
            .iter()
            .map(|s| ("restart_after_step_back", s)),
    );
    if let Some(SessionGapSpec::Gap(g)) = &config.session_gap {
        all.push(("session_gap", g));
    }
    let duration = all.iter().find(|(_, s)| s.is_duration()).map(|(f, _)| *f);
    let number = all
        .iter()
        .find(|(_, s)| s.is_unit_bound_number())
        .map(|(f, _)| *f);
    let clock = config.clock.as_deref();
    let temporal = clock.is_some_and(|c| dtype_of(input, c).is_temporal());
    match (duration, number) {
        (Some(d), Some(n)) => {
            return Err(format!(
                "{WHO}: {d} is a duration and {n} is a number, but every parameter on one \
                 clock is written the same way. Give both as durations on a temporal clock, or \
                 both as numbers of clock units on a number clock."
            ));
        }
        (Some(d), None) if !temporal => {
            return Err(format!(
                "{WHO}: {d} is a duration, which needs a temporal clock column (Datetime, \
                 Date or Duration), but clock is {}. Give {d} as a number of clock units, or \
                 name a temporal clock column.",
                clock.map_or("not given".to_string(), |c| {
                    format!("{c:?} ({})", crate::arrow::dtype_name(dtype_of(input, c)))
                })
            ));
        }
        (None, Some(n)) if temporal => {
            return Err(format!(
                "{WHO}: {n} is a number, but clock column {:?} is {}, which measures time. Give \
                 {n} as a duration, such as \"10m\", a timedelta or pl.duration(minutes=10).",
                clock.unwrap_or(""),
                crate::arrow::dtype_name(dtype_of(input, clock.unwrap_or("")))
            ));
        }
        _ => {}
    }
    if let (Some(c), true) = (clock, temporal) {
        let dtype = dtype_of(input, c);
        let tick = match dtype {
            DataType::Date => 86_400.0,
            DataType::Datetime(unit, _) | DataType::Duration(unit) => match unit {
                TimeUnit::Milliseconds => 1e-3,
                TimeUnit::Microseconds => 1e-6,
                TimeUnit::Nanoseconds => 1e-9,
            },
            _ => 1e-9,
        };
        let step = match dtype {
            DataType::Date => "a day",
            DataType::Datetime(TimeUnit::Milliseconds, _)
            | DataType::Duration(TimeUnit::Milliseconds) => "a millisecond",
            DataType::Datetime(TimeUnit::Microseconds, _)
            | DataType::Duration(TimeUnit::Microseconds) => "a microsecond",
            _ => "a nanosecond",
        };
        for (param, s) in &all {
            let (why, or_zero) = match *param {
                "gap_cap" => (
                    "every step would be cut to it and the clock would count rows rather than \
                     measure time",
                    "",
                ),
                "restart_after_step_back" => (
                    "no step back could be as small and every one would reset the windows",
                    ", or to 0 to reset at every step back",
                ),
                _ => continue,
            };
            let v = s.value();
            if s.is_duration() && v > 0.0 && v < tick {
                return Err(format!(
                    "{WHO}: {param} is {s}, less than {step}, the smallest step clock column \
                     {c:?} ({}) can take, so {why}. Set {param} to at least {step}{or_zero}.",
                    crate::arrow::dtype_name(dtype)
                ));
            }
        }
    }
    Ok(())
}

/// The call a state resumes, as it compares: a numeric `session_gap` is no
/// part of it, since no window reads the number (every window ends at a
/// session change), so a state saved with one resumes the call without it,
/// or with another (task 278). `"reset"` acts, and stays.
fn same_call(config: &WindowsConfig) -> WindowsConfig {
    let mut c = config.clone();
    if matches!(c.session_gap, Some(SessionGapSpec::Gap(_))) {
        c.session_gap = None;
    }
    if c.on_step_back.as_deref() == Some("error") {
        c.on_step_back = None;
    }
    c
}

/// What differs between the call a state was saved by and this one, each
/// setting with both its values, for the refusal of another call.
fn call_differences(saved: &WindowsConfig, now: &WindowsConfig) -> String {
    fn text<T: std::fmt::Display>(v: Option<&T>) -> String {
        v.map_or_else(|| "not set".to_string(), |v| format!("\"{v}\""))
    }
    let reset = |c: &WindowsConfig| {
        if c.session_gap.is_some() {
            "\"reset\"".to_string()
        } else {
            "not set".to_string()
        }
    };
    let like = |c: &WindowsConfig| text(c.like.as_ref().map(|l| &l.spec));
    let mut out: Vec<String> = Vec::new();
    if saved.formulas != now.formulas {
        out.push("the expressions differ".into());
    }
    // Compared as the call compares them: a duration by its length.
    let settings = [
        (
            "clock",
            saved.clock != now.clock,
            text(saved.clock.as_ref()),
            text(now.clock.as_ref()),
        ),
        (
            "gap_cap",
            saved.gap_cap != now.gap_cap,
            text(saved.gap_cap.as_ref()),
            text(now.gap_cap.as_ref()),
        ),
        (
            "restart_after_step_back",
            saved.restart_after_step_back != now.restart_after_step_back,
            text(saved.restart_after_step_back.as_ref()),
            text(now.restart_after_step_back.as_ref()),
        ),
        (
            "on_step_back",
            saved.on_step_back != now.on_step_back,
            text(saved.on_step_back.as_ref()),
            text(now.on_step_back.as_ref()),
        ),
        (
            "session",
            saved.session != now.session,
            text(saved.session.as_ref()),
            text(now.session.as_ref()),
        ),
        (
            "session_gap",
            saved.session_gap != now.session_gap,
            reset(saved),
            reset(now),
        ),
        (
            "group",
            saved.group != now.group,
            text(saved.group.as_ref()),
            text(now.group.as_ref()),
        ),
        ("like=", saved.like != now.like, like(saved), like(now)),
    ];
    for (name, differs, was, is) in settings {
        if !differs {
            continue;
        }
        out.push(if was == is {
            format!("{name} differs")
        } else {
            format!("{name} was {was} when saved and is {is} now")
        });
    }
    match out.len() {
        0 => "the call differs".into(),
        1 => out.remove(0),
        _ => {
            let last = out.pop().unwrap_or_default();
            format!("{} and {last}", out.join(", "))
        }
    }
}

/// [`same_call`] for the core's clock rules: a session's gap compares as
/// any other number.
fn same_clock(cfg: &ClockCfg) -> ClockCfg {
    let mut c = *cfg;
    if matches!(c.session_gap, Some(SessionGap::Gap(_))) {
        c.session_gap = Some(SessionGap::Gap(0.0));
    }
    c
}

/// A session value as the core sees it: a hash of its text, so any dtype
/// can mark sessions and only a change is read.
/// The bank's session hash (`fnv1a`), which std's `DefaultHasher` is not: a
/// hash persisted in the state must be the same on every toolchain (review
/// R1, S6).
fn session_hash(v: Option<&str>) -> u64 {
    crate::bank::session_hash(v)
}

/// What a [`WindowsRun`] holds that no row changes: the call, its plan and
/// the schemas it was built against.
#[derive(PartialEq)]
struct Constants {
    config: WindowsConfig,
    inputs: Vec<Node>,
    columns: Vec<(String, usize)>,
    /// [`Plan::bare`].
    bare: Option<Vec<String>>,
    /// [`Plan::increments`] and [`Plan::slots`].
    increments: Vec<RowOp>,
    slots: Vec<(Node, bool)>,
    accept: Vec<String>,
    /// [`Plan::lists`].
    lists: Vec<String>,
    /// The input's schema, as the call was built against it.
    input: Schema,
    /// The group and session columns' dtypes, by role: a key is its value's
    /// text, so a column of another type on a resumed input would start
    /// every key over, silently (task 159, F2).
    key_dtypes: Vec<(String, String)>,
    /// The output's schema: the input's columns, then each formula.
    output: Schema,
}

/// A row of a frame by its place in it, sliced out only where it is read: a
/// chunk is fed far more often than a state is saved sliced.
#[derive(Clone)]
struct LastRow {
    frame: DataFrame,
    at: usize,
}

impl LastRow {
    fn row(&self) -> DataFrame {
        self.frame.slice(self.at as i64, 1)
    }
}

/// One `with_windows` run: the core, its plan, and the input rows it holds.
#[derive(Clone)]
pub struct WindowsRun {
    /// What the call fixes and no row changes, shared by every run of a spec
    /// (the bank's cores of one spec's groups): a copy of the run is then
    /// its state alone.
    k: Arc<Constants>,
    core: Windows,
    /// The rows the core holds, oldest first, as slices of the input with
    /// the increment columns beside them.
    /// Each behind an `Arc`, so that a copy of the run (the bank keeps one
    /// before a chunk that may be refused) shares the rows rather than
    /// copying a frame a chunk.
    held: VecDeque<Arc<DataFrame>>,
    /// Per group, the increments' state; the null key apart, so a row's
    /// key is looked up as text without a copy.
    incr_state: Vec<IncrState>,
    incr_index: HashMap<String, usize>,
    incr_null: Option<usize>,
    /// The previous row's clock on the stream, across groups: a step back
    /// past `restart_after_step_back` there restarts every group's windows
    /// in the core, so it restarts every group's increments too (review
    /// R1, S4).
    stream_clock: Option<ClockValue>,
    /// The core's row count at this run's first row: an error names the
    /// row of this run's input.
    run_base: u64,
    /// Rows at the start of the input to skip, from a state saved under a
    /// slice of the same input (review R2, W4; R4, B1), the first clock of
    /// the input that state was saved under (R4, B3), and how many rows
    /// this run has skipped, which an error's row number counts.
    resume_skip: usize,
    resume_first: Option<ClockValue>,
    resume_first_session: Option<u64>,
    skipped: u64,
    /// The identity of the input a sliced state resumes on (review R5, C3;
    /// R6, D1 and D2): the skip as loaded; the last rows the state read --
    /// the rows it holds that are this input's, or the last row read where
    /// it holds none -- and the skipped rows at those positions as they go
    /// by.
    resume_total: usize,
    resume_held: Option<DataFrame>,
    resume_rows: Vec<DataFrame>,
    /// The last row read, skipped or fed, as the chunk had it (the hidden
    /// columns beside the input's when fed): what a sliced state saves as
    /// its identity beside the rows it holds (review R6, D1).
    last_row: Option<LastRow>,
    /// The first row's clock and session of this run's input, and whether
    /// a row has been seen: what a state saved by this run remembers its
    /// input by (review R4, B3; R7, E2).
    first_clock: Option<ClockValue>,
    first_session: Option<u64>,
    started: bool,
}

/// Each text and binary column of `rows` with its views rebuilt into the
/// fewest data buffers, so the rows a state holds save to the same bytes
/// however they were laid out in memory (task 278, b32). A parquet read
/// gives a text column whose values all sit inline in their views a data
/// buffer anyway, where a frame built in memory has none, and the IPC
/// writer writes every buffer the array has: the command line saved 19
/// bytes more than `ModelBank.fit_predict` of the same rows. The values,
/// and so every number the state holds, are unchanged.
pub(crate) fn compact_views(rows: &mut DataFrame) -> PolarsResult<()> {
    let names: Vec<PlSmallStr> = rows.get_column_names_owned();
    for name in names {
        let col = rows.column(&name)?;
        let compact = match col.dtype() {
            DataType::String => StringChunked::from_chunk_iter(
                name.clone(),
                col.str()?.downcast_iter().map(|a| a.clone().gc()),
            )
            .into_column(),
            DataType::Binary => BinaryChunked::from_chunk_iter(
                name.clone(),
                col.binary()?.downcast_iter().map(|a| a.clone().gc()),
            )
            .into_column(),
            _ => continue,
        };
        rows.replace(&name, compact)?;
    }
    Ok(())
}

impl WindowsRun {
    /// Share `other`'s constants where they are the same as this run's: the
    /// bank builds one core a group, and the cores of one spec's groups hold
    /// one copy of the call and its plan, not one each. A run that differs
    /// (another input schema, say) keeps its own.
    pub fn share_with(&mut self, other: &WindowsRun) {
        if !Arc::ptr_eq(&self.k, &other.k) && *self.k == *other.k {
            self.k = Arc::clone(&other.k);
        }
        self.core.share_with(&other.core);
    }

    /// A run of `config` over an input with schema `input`.
    ///
    /// # Errors
    ///
    /// A formula or clock policy that cannot run, a column the input has
    /// not got, and an output name that collides.
    pub fn new(config: WindowsConfig, input: &Schema) -> Result<Self, Refused> {
        for (role, c) in [
            ("clock", &config.clock),
            ("session", &config.session),
            ("group", &config.group),
        ] {
            if let Some(c) = c
                && !input.contains(c)
            {
                return Err(format!(
                    "{WHO}: the {role} column {c:?} is not in the input, whose columns are {:?}. \
                     Name a column the input has, or add {c:?} to it.",
                    input.iter_names().map(|n| n.as_str()).collect::<Vec<_>>()
                )
                .into());
            }
        }
        // A clock of text was read through a cast that turns digits into
        // numbers and anything else into nulls, where every other surface
        // refuses it by its type (task 356).
        if let Some(c) = &config.clock {
            let dtype = dtype_of(input, c);
            if !(dtype.is_numeric()
                || dtype.is_temporal()
                || matches!(dtype, DataType::Boolean | DataType::Null))
            {
                return Err(Refused::wrong_type(format!(
                    "{WHO}: clock column {c:?} must be a number or temporal, but it is {}. Cast \
                     it, such as with pl.col({c:?}).cast(pl.Float64), or to a Datetime.",
                    crate::arrow::dtype_name(dtype)
                )));
            }
        }
        // A key that does nothing is refused, as a spec refuses it (task 278,
        // D3): each was accepted here and changed nothing.
        if config.session_gap.is_some() && config.session.is_none() {
            return Err(format!(
                "{WHO}: session_gap is set, but session is not, and session_gap says only what \
                 a change of session does. Set session to the session column, or leave out \
                 session_gap."
            )
            .into());
        }
        if config.restart_after_step_back.is_some() && config.clock.is_none() {
            return Err(format!(
                "{WHO}: restart_after_step_back is set, but clock is not, and without a clock \
                 column every row is one step forward, so no row steps back. Set clock to the \
                 clock column, or leave out restart_after_step_back."
            )
            .into());
        }
        let mut config = config;
        let cfg: ClockCfg = clock_cfg_of(&ClockPolicy {
            who: WHO,
            clock: config.clock.as_deref(),
            gap_cap: config.gap_cap.as_ref(),
            restart_after_step_back: config.restart_after_step_back.as_ref(),
            on_step_back: config.on_step_back.as_deref(),
            session: config.session.as_deref(),
            session_gap: config.session_gap.as_ref(),
            spec_closes_on_session: None,
            // A bank's window targets run one core per group, each built
            // with no group column (`resolvers.rs`): the spec refuses its
            // own groups without a clock, with this same rule (task 173).
            group: config.group.as_deref(),
            looks_ahead: config
                .formulas
                .iter()
                .any(|f| f.tree.has_forward_operator()),
        })?;
        if config.on_step_back.as_deref() == Some("error") {
            config.on_step_back = None;
        }
        let mut p = plan(&config, input)?;
        check_scale(&config, &p.spans, input)?;
        let accept = config
            .like
            .as_ref()
            .map_or_else(Vec::new, |l| l.accept.clone());
        for c in &accept {
            if !input.contains(c) {
                return Err(format!(
                    "{WHO}: the spec {:?} given as like= reads column {c:?}, which is not in \
                     the input. Add {c:?} to the input, or pass a spec that reads only its \
                     columns.",
                    config.like.as_ref().map_or("", |l| l.spec.as_str())
                )
                .into());
            }
        }
        // The output's dtypes: the formulas over the hidden columns, every
        // one a Float64, on an empty frame.
        let mut hidden_schema = input.clone();
        for r in &p.increments {
            hidden_schema.insert(r.key.as_str().into(), DataType::Float64);
        }
        for (k, _) in &p.columns {
            hidden_schema.insert(k.as_str().into(), DataType::Float64);
        }
        let exprs: Vec<Expr> = config
            .formulas
            .iter()
            .map(|f| {
                f.tree
                    .to_expr(&|op| col(hidden(op).as_str()))
                    .alias(f.name.as_str())
            })
            .collect();
        let evaluated = dtypes_of(&hidden_schema, exprs).map_err(|e| {
            format!(
                "{WHO}: Polars cannot compute an expression on the input's columns ({e}). \
                     Check that each expression reads columns of the right type."
            )
        })?;
        let mut output = input.clone();
        for f in &config.formulas {
            let dtype = evaluated.get(&f.name).expect("aliased").clone();
            output.insert(f.name.as_str().into(), dtype);
        }
        let lists = std::mem::take(&mut p.lists);
        let mut core = if !p.cross.is_empty() || !p.profiles.is_empty() || !p.seasonals.is_empty() {
            Windows::with_seasonals(
                p.kernels,
                p.ops,
                p.cross,
                p.moments,
                p.profiles,
                p.seasonals,
                cfg,
            )
            .map_err(|e| format!("{WHO}: {e}"))?
        } else {
            Windows::with_moments(p.kernels, p.ops, p.moments, cfg)
                .map_err(|e| format!("{WHO}: {e}"))?
        };
        if config.clock.is_none() {
            // On the row clock a window of `w` holds at most `w + 1` rows
            // read, and the row that joins before the oldest leaves.
            for (ki, row, cap) in core.extremum_caps() {
                let w = core.kernels()[ki]
                    .window_size
                    .expect("a capped kernel has a window");
                let rows = w.floor() + 2.0;
                let bytes = rows * row as f64;
                if bytes > cap as f64 {
                    return Err(format!(
                        "{WHO}: a rolling_max or rolling_min window of {w} rows holds up to \
                         {rows} rows of {row} bytes each a group, {bytes} bytes, past \
                         max_state_bytes of {cap}. Shorten window_size, or raise \
                         max_state_bytes (at most 64 MiB)."
                    )
                    .into());
                }
            }
        }
        core.set_grouped(config.group.is_some());
        let inputs = p.inputs;
        let key_dtypes: Vec<(String, String)> =
            [("group", &config.group), ("session", &config.session)]
                .into_iter()
                .filter_map(|(role, c)| {
                    let c = c.as_ref()?;
                    let dtype = input.get(c.as_str()).map(ToString::to_string)?;
                    Some((role.to_string(), dtype))
                })
                .collect();
        Ok(Self {
            k: Arc::new(Constants {
                config,
                inputs,
                columns: p.columns,
                bare: p.bare,
                increments: p.increments,
                slots: p.slots,
                accept,
                lists,
                input: input.clone(),
                key_dtypes,
                output,
            }),
            core,
            held: VecDeque::new(),
            incr_state: Vec::new(),
            incr_index: HashMap::new(),
            incr_null: None,
            stream_clock: None,
            run_base: 0,
            resume_skip: 0,
            resume_first: None,
            resume_first_session: None,
            skipped: 0,
            resume_total: 0,
            resume_held: None,
            resume_rows: Vec::new(),
            last_row: None,
            first_clock: None,
            first_session: None,
            started: false,
        })
    }

    /// The output's schema: the input's columns, then each formula.
    /// Switch the core's plain-run loop off or on again: for a test that
    /// holds it to the general row loop, and a timing of what it saves. Not
    /// state: a run read back from a file has it on.
    #[doc(hidden)]
    pub fn set_plain_run(&mut self, on: bool) {
        self.core.set_plain_run(on);
    }

    /// Rows the core's plain-run loop has taken, for a test to see that it ran.
    #[doc(hidden)]
    pub fn plain_rows(&self) -> u64 {
        self.core.plain_rows()
    }

    /// Bytes the core's plain-run loop keeps between rows, for a measurement.
    #[doc(hidden)]
    pub fn plain_scratch_bytes(&self) -> usize {
        self.core.plain_scratch_bytes()
    }

    pub fn schema(&self) -> Schema {
        self.k.output.clone()
    }

    /// Rows fed and not yet out.
    pub fn held(&self) -> usize {
        self.core.held()
    }

    /// Contributing rows the kernels' queues hold: what their sums cost.
    pub fn queued(&self) -> usize {
        self.core.queued()
    }

    /// The columns the formulas read, whatever else the output keeps: what a
    /// source must read however narrow the query's projection.
    pub fn needed(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut put = |c: &String| {
            if !out.contains(c) {
                out.push(c.clone());
            }
        };
        for c in [
            &self.k.config.clock,
            &self.k.config.session,
            &self.k.config.group,
        ]
        .into_iter()
        .flatten()
        {
            put(c);
        }
        for f in &self.k.config.formulas {
            f.tree.columns().iter().for_each(&mut put);
        }
        self.k.accept.iter().for_each(&mut put);
        out
    }

    /// The hidden columns a held chunk carries beside the input's.
    fn hidden_names(&self) -> Vec<String> {
        self.k.increments.iter().map(|r| r.key.clone()).collect()
    }

    /// Feed one chunk and return the rows it resolved, in input order, less
    /// any a `"drop"` operator was partial on. With `limit`, rows are fed
    /// only until `limit` rows are out, so the state is the state after
    /// the row that resolved the last of them, whatever the chunk size.
    ///
    /// # Errors
    ///
    /// A chunk whose columns are not the input's, and every row refusal the
    /// core makes, each naming the row.
    pub fn feed(&mut self, df: &DataFrame, limit: Option<usize>) -> PolarsResult<DataFrame> {
        self.feed_inner(df, limit, None, None, false)
    }

    /// Feed one chunk for a bank learning formula targets (docs/PLAN.md
    /// task 104): every row it resolved, dropped ones included, as a frame
    /// of the formulas' values -- null where a `"drop"` operator was partial
    /// -- beside `@po:seq`, the row's number as the caller counts it (the
    /// chunk's `@po:row` column), and `@po:at`, the number of the row that
    /// resolved it (the row that closed, cut or discarded its windows),
    /// read from `rows`, the chunk's rows' numbers.
    ///
    /// # Errors
    ///
    /// As [`Self::feed`].
    pub fn feed_resolving(&mut self, df: &DataFrame, rows: &[u64]) -> PolarsResult<DataFrame> {
        polars_ensure!(
            rows.len() == df.height() && df.get_column_names().iter().any(|c| c.as_str() == "@po:row"),
            ComputeError: "internal: feed_resolving: the chunk must carry a row number for each row and an \
                @po:row column, which is a defect in rheon. Open an issue at \
                https://github.com/rheon-dev/rheon/issues with this message."
        );
        self.feed_inner(df, None, Some(rows), None, false)
    }

    /// [`Self::feed_resolving`] for a caller that evaluates what the lazy
    /// engine would per chunk, once for every group it feeds (the bank's
    /// resolver, which hands the engine its whole chunk rather than one
    /// plan a group): `given` is what [`Self::share`] evaluated over the
    /// whole frame -- the chunk's operator inputs and its increments'
    /// inputs -- taken at this group's rows, and, where
    /// [`Self::formulas_pending`], the frame
    /// returned is the resolved rows before their formulas, which
    /// [`Self::resolved_formulas`] evaluates over any number of such frames
    /// stacked.
    ///
    /// # Errors
    ///
    /// As [`Self::feed_resolving`].
    pub fn feed_resolving_with(
        &mut self,
        df: &DataFrame,
        rows: &[u64],
        given: Option<Given>,
    ) -> PolarsResult<DataFrame> {
        polars_ensure!(
            rows.len() == df.height() && df.get_column_names().iter().any(|c| c.as_str() == "@po:row"),
            ComputeError: "internal: feed_resolving: the chunk must carry a row number for each row and an \
                @po:row column, which is a defect in rheon. Open an issue at \
                https://github.com/rheon-dev/rheon/issues with this message."
        );
        if let Some(given) = &given {
            let inputs_fit = given.inputs.as_ref().is_none_or(|all| {
                all.len() == self.k.inputs.len() && all.iter().all(|v| v.len() == df.height())
            });
            let incr_fits = given.incr.as_ref().is_none_or(|r| r.rows == df.height());
            polars_ensure!(
                inputs_fit && incr_fits,
                ComputeError: "internal: feed_resolving: the operator inputs handed over must be \
                    one value a row for each input, which is a defect in rheon. Open an \
                    issue at https://github.com/rheon-dev/rheon/issues with this message."
            );
        }
        self.feed_inner(df, None, Some(rows), given, true)
    }

    /// Whether the formulas of a resolving feed are evaluated by the lazy
    /// engine, over the resolved rows: unless each is only an operator's
    /// values.
    pub fn formulas_pending(&self) -> bool {
        self.bare_formulas().is_none()
    }

    /// The hidden column each formula is, where every formula is nothing else
    /// but one operator (and the short cuts are on).
    fn bare_formulas(&self) -> Option<&[String]> {
        short_cuts().then_some(self.k.bare.as_deref()).flatten()
    }

    /// Every operator input over a whole frame, as one plan of the lazy
    /// engine, each as doubles with a null (as a NaN) where the input has
    /// none: what a feed would evaluate for any rows of the frame, since a
    /// formula's inputs are element-wise. `None` where there is nothing to
    /// save a feed -- inputs that are all columns of doubles -- and where the
    /// plan refuses, which a feed reports for the rows it was handed. For a
    /// run with no increments; [`Self::share_increments`] does the rest.
    fn evaluate_inputs(&self, frame: &DataFrame) -> Option<Vec<Vec<f64>>> {
        if self.k.inputs.is_empty() || self.bare_inputs(frame).is_some() {
            return None;
        }
        self.evaluated_inputs(frame).ok()
    }

    /// The input columns, where each is a column of doubles.
    fn bare_inputs<'a>(&self, chunk: &'a DataFrame) -> Option<Vec<&'a Column>> {
        short_cuts()
            .then(|| {
                self.k
                    .inputs
                    .iter()
                    .map(|node| match node {
                        Node::Col(c) => chunk
                            .column(c.as_str())
                            .ok()
                            .filter(|s| *s.dtype() == DataType::Float64),
                        _ => None,
                    })
                    .collect()
            })
            .flatten()
    }

    /// Polars evaluates every operator's input over the chunk.
    fn evaluated_inputs(&self, chunk: &DataFrame) -> PolarsResult<Vec<Vec<f64>>> {
        if self.k.inputs.is_empty() {
            return Ok(Vec::new());
        }
        let resolve_incr = |op: &OpNode| col(hidden(op).as_str());
        // `with_columns` then `select`: a bare literal input broadcasts to the
        // chunk's rows.
        let evaluated = chunk
            .clone()
            .lazy()
            .with_columns(
                self.k
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        node.to_expr(&resolve_incr)
                            .cast(DataType::Float64)
                            .alias(input_column(i).as_str())
                    })
                    .collect::<Vec<_>>(),
            )
            .select(
                (0..self.k.inputs.len())
                    .map(|i| col(input_column(i).as_str()))
                    .collect::<Vec<_>>(),
            )
            .collect()?;
        (0..self.k.inputs.len())
            .map(|i| {
                let s = evaluated.column(input_column(i).as_str())?;
                Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            })
            .collect()
    }

    fn feed_inner(
        &mut self,
        df: &DataFrame,
        limit: Option<usize>,
        resolving: Option<&[u64]>,
        given: Option<Given>,
        defer: bool,
    ) -> PolarsResult<DataFrame> {
        // The clock of the whole chunk first: the row numbers an error
        // names count the rows skipped below.
        let mut clock = self.clock_values(df)?;
        // A state saved under a slice recorded the rows of its input the
        // run consumed; a run resumed on that input skips them (review R2,
        // W4; R4, B1). The input's first clock says whether this is that
        // input: one that starts elsewhere, the next file, skips nothing
        // (R4, B3). Without a clock column the rows are skipped either way.
        if !self.started && df.height() > 0 {
            // The verdict first, the run's record of its input after it: a
            // refused first chunk leaves the run as it was, so fed again it
            // is refused again, where a run marked started before the
            // verdict took the same chunk whole on the next call (task 159,
            // F1).
            let first_clock = clock.as_ref().map(|c| c[0]);
            let (key, session) = self.first_key_session(df)?;
            if self.resume_skip > 0
                && (first_clock != self.resume_first || session != self.resume_first_session)
            {
                // Not the same input from its start: its first row's clock or
                // session is not the saved input's (review R7, E2: a clock
                // that starts over at the same stamp each day gives the next
                // file the same first clock, and the session tells them
                // apart). The row is put to the clock policy against the
                // last row the state read: a step forward or a new start (a
                // step back past restart_after_step_back, a new session) is
                // the next file, which skips nothing; the same stamp is a
                // file boundary inside a tied stamp or the same input sliced
                // inside it, which cannot be told apart; a step back the
                // policy refuses is the same input sliced by hand or an
                // overlapping file (review R5, C4; R6, D3 and D4).
                match self.core.peek(key.as_deref(), first_clock, session) {
                    Ok(Peek::NewStart) => self.resume_skip = 0,
                    Ok(Peek::Forward) if self.k.config.clock.is_some() => self.resume_skip = 0,
                    // A row-count clock steps forward at every row, so a step
                    // forward is evidence of nothing there: without a clock
                    // column the next file begins with a new session (review
                    // R8, F1) -- of its first row's group under `group`, so a
                    // group the state has not read shows none (R9, G1).
                    Ok(Peek::Forward) => polars_bail!(ComputeError:
                        "{WHO}: the state was saved from a slice of its input, and this is \
                         another input: its first row continues the session last read, or is of \
                         a group the state has not read. Resume on the whole input the state was \
                         saved from, or begin the next file with a new session (and, under group, \
                         with a group the state has read)."
                    ),
                    Ok(Peek::SameStamp) => polars_bail!(ComputeError:
                        "{WHO}: the state was saved from a slice of its input, and this input \
                         starts at the last clock value the state read. A file that begins \
                         inside a run of tied clock values, or the same input sliced inside \
                         them, starts there too. Resume on the input the state was \
                         saved from, whole, or on the next file, which starts after the last \
                         clock value the state read."
                    ),
                    Ok(Peek::StepBack) => polars_bail!(ComputeError:
                        "{WHO}: the state was saved from a slice of its input, and this is \
                         another input: it starts before the last row the state read. Resume on \
                         the input the state was saved from, whole, or on the next file, which \
                         starts after the last row the state read."
                    ),
                    // The policy's own words beside the rule's (review R8, F2):
                    // under `group` a next-day file's step back is refused on
                    // the stream's clock whatever its session, as the
                    // unsliced path refuses it.
                    Err(r) => {
                        let why = self.refusal(r, self.core.next_seq()).to_string();
                        let why = why.strip_prefix(&format!("{WHO}: ")).unwrap_or(&why);
                        // Its first sentence, the step back: its remedy is
                        // not this one's.
                        let why = why.split_once(". ").map_or(why, |(first, _)| first);
                        let mut letters = why.chars();
                        let why: String = letters
                            .next()
                            .map(|c| c.to_uppercase().chain(letters).collect())
                            .unwrap_or_default();
                        polars_bail!(ComputeError:
                            "{WHO}: the state was saved from a slice of its input, and this is \
                             another input: it starts before the last row the state read. \
                             {why}. Resume on the input the state was saved from, whole, or on \
                             the next file, which starts after the last row the state read."
                        )
                    }
                }
            } else if self.resume_skip == 0 && self.k.config.clock.is_some() && resolving.is_none()
            {
                // A state saved without a slice goes on with the next file,
                // and refuses one that starts at the last stamp it read, as a
                // sliced state does: a file boundary inside a tied stamp
                // cannot be told from an input that repeats the rows read
                // there, which would come out twice (task 158, E13). A fresh
                // run has read no stamp, and a new session at that stamp is a
                // new start; a step back is the policy's to refuse, below.
                // Not for a core the bank's resolver feeds (`resolving`): it
                // gets the rows of one stream the bank's own clock has
                // checked, so a chunk boundary inside a tied stamp is that
                // stream going on, and the same words refused every bank
                // with a window target saved there (task 159, B1).
                if let Ok(Peek::SameStamp) = self.core.peek(key.as_deref(), first_clock, session) {
                    polars_bail!(ComputeError:
                        "{WHO}: this input starts at the last clock value the state read, so it \
                         may be an input that repeats rows the state read, which would come out \
                         twice. A file that begins inside a run of tied clock values cannot be \
                         told from one. Resume on the next file, which starts after the last \
                         clock value the state read, and cut files between clock values rather \
                         than inside a run of tied ones."
                    );
                }
            }
            self.started = true;
            self.first_clock = first_clock;
            self.first_session = session;
        }
        let skipped: DataFrame;
        let mut df = df;
        let mut resolving = resolving;
        let (mut given_inputs, mut given_incr) = given.map_or((None, None), |g| (g.inputs, g.incr));
        let skip = self.resume_skip.min(df.height());
        if skip > 0 {
            // The skipped rows inside the held window -- the last
            // `resume_held` of the `resume_total` rows consumed -- are kept
            // for the identity check once the skip completes (review R5, C3).
            let before = usize::try_from(self.skipped).expect("rows fit");
            let held = self.resume_held.as_ref().map_or(0, DataFrame::height);
            let lo = self.resume_total.saturating_sub(held).max(before);
            let hi = (before + skip).min(self.resume_total);
            if hi > lo {
                self.resume_rows
                    .push(df.slice((lo - before) as i64, hi - lo));
            }
            self.resume_skip -= skip;
            self.skipped += skip as u64;
            // Once the skip completes, the last row skipped is the last row
            // read; until then the loaded one still is, as `consumed` counts
            // the rows pending (review R6, D1; R7, E4).
            if self.resume_skip == 0 {
                self.last_row = Some(LastRow {
                    frame: df.clone(),
                    at: skip - 1,
                });
            }
            skipped = df.slice(skip as i64, df.height() - skip);
            df = &skipped;
            resolving = resolving.map(|r| &r[skip..]);
            for v in given_inputs.iter_mut().flatten() {
                v.drain(..skip);
            }
            // The increments' inputs of the rows after those skipped: not
            // handed over, the run reads them from its own rows.
            given_incr = None;
            if let Some(c) = &mut clock {
                c.drain(..skip);
            }
            if self.resume_skip == 0 {
                self.check_resumed_input()?;
            }
        }
        let n = df.height();
        let session: Option<Vec<u64>> = self.session_hashes(df)?;
        // The groups' keys as codes from the column's own values: a group is
        // made at the first row that reaches it, below. The increments (a
        // `diff` or `pct_change` in an input) keep their state by the key's
        // text, which only they read.
        let mut codes = match &self.k.config.group {
            None => ChunkKeys::ungrouped(),
            Some(c) => ChunkKeys::of(
                &mut self.core,
                df.column(c.as_str())?.as_materialized_series(),
            )?,
        };
        let group_col = match &self.k.config.group {
            Some(c) if !self.k.increments.is_empty() => {
                Some(key_text(df.column(c.as_str())?.as_materialized_series())?)
            }
            _ => None,
        };
        let groups = group_col.as_ref().map(|c| c.str()).transpose()?;
        let keys: Vec<Option<&str>> = if self.k.increments.is_empty() {
            Vec::new()
        } else {
            (0..n)
                .map(|i| match groups {
                    None => Some(""),
                    Some(g) => g.get(i),
                })
                .collect()
        };

        // Pass one: the increments, one row back within the group's
        // session, as hidden columns of the chunk. On a snapshot: the state
        // must end at the last row the core took, and a slice (`limit`) or
        // a refusal stops short of the chunk's end (review R1, B1).
        let snapshot = (
            self.incr_state.clone(),
            self.incr_index.clone(),
            self.incr_null,
            self.stream_clock,
        );
        let chunk = self.with_increments(df, &keys, &clock, session.as_deref(), given_incr)?;

        // Pass two: Polars evaluates every operator's input -- but an input
        // that is a column of doubles, as a price is, is its values already:
        // the cast to double of a double is none, and a plan of the lazy
        // engine is tens of microseconds a call, which a bank's group pays
        // every chunk. A bank's resolver evaluates the whole chunk's in one
        // plan, and hands each group its rows' values.
        // A column with no null in one piece is read where it lies: the copy
        // with a NaN for each null is for the column that has one.
        let values: Vec<Cow<'_, [f64]>> = if let Some(values) = given_inputs {
            values.into_iter().map(Cow::Owned).collect()
        } else if let Some(columns) = self.bare_inputs(&chunk) {
            columns
                .iter()
                .map(|s| {
                    let ca = s.f64()?;
                    Ok(match ca.cont_slice() {
                        Ok(whole) => Cow::Borrowed(whole),
                        Err(_) => Cow::Owned(ca.iter().map(|v| v.unwrap_or(f64::NAN)).collect()),
                    })
                })
                .collect::<PolarsResult<_>>()?
        } else {
            self.evaluated_inputs(&chunk)?
                .into_iter()
                .map(Cow::Owned)
                .collect()
        };
        let slices: Vec<&[f64]> = values.iter().map(|c| &**c).collect();
        let accepts: Vec<Vec<f64>> = self
            .k
            .accept
            .iter()
            .map(|c| {
                let s = df.column(c.as_str())?.cast(&DataType::Float64)?;
                Ok(s.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
            })
            .collect::<PolarsResult<_>>()?;

        // The neighbor lists cross operators read, as text.
        let lists = self
            .k
            .lists
            .iter()
            .map(|c| list_keys(df, c))
            .collect::<PolarsResult<Vec<_>>>()?;
        let mut row_lists: Vec<Option<&[Option<String>]>> = vec![None; lists.len()];

        // Pass three: the core.
        let chunk_base = self.core.next_seq();
        let mut row_values = vec![0.0; self.k.inputs.len()];
        let mut consumed = n;
        // Resolving: per resolved row, the number of the row that resolved
        // it, read off the core's ready count after each push. Nothing is
        // ready before the chunk: every resolving feed drains everything.
        let mut at: Vec<u64> = Vec::new();
        if let Some(rows) = resolving {
            at.resize(self.core.ready(), rows.first().copied().unwrap_or(0));
        }
        let plain_rows = limit.is_none() && resolving.is_none();
        let mut i = 0;
        while i < n {
            if let Some(l) = limit
                && self.core.ready_kept() >= l
            {
                consumed = i;
                break;
            }
            // The row's group: its code names it, or makes it where this is
            // the first row of a key the core had no group for.
            let group = codes.group(&mut self.core, i);
            // The rows the core's plain loop runs, taken by it as one block
            // after another; the first row that is not plain, and the rows
            // of a feed that stops at a row count or names the row that
            // resolved each, are `push`'s one at a time.
            if plain_rows && self.core.plain_may_run() {
                let taken = self.core.push_plain(&PlainRows {
                    group,
                    start: i,
                    clock: clock.as_deref(),
                    len: n,
                    session: session.as_deref(),
                    values: &slices,
                    accepts: &accepts,
                    keys: codes.is_grouped().then_some(&codes),
                });
                if taken > 0 {
                    i += taken;
                    continue;
                }
            }
            for (k, col) in slices.iter().enumerate() {
                row_values[k] = col[i];
            }
            for (l, (flat, ranges)) in lists.iter().enumerate() {
                row_lists[l] = ranges[i].map(|(a, b)| &flat[a..b]);
            }
            let row = RowIn {
                group,
                clock: clock.as_ref().map(|c| c[i]),
                session: session.as_ref().map(|s| s[i]),
                values: &row_values,
                accept: accepts.iter().all(|col| usable(col[i])),
                lists: &row_lists,
            };
            if let Err(r) = self.core.push(&row) {
                // The rows before it are in the core: hold them, and put the
                // increments' state where they end.
                if i > 0 {
                    self.held.push_back(Arc::new(chunk.slice(0, i)));
                    self.last_row = Some(LastRow {
                        frame: chunk.clone(),
                        at: i - 1,
                    });
                }
                self.rewind_increments(snapshot, df, &keys, &clock, session.as_deref(), i)?;
                return Err(self.refusal(r, chunk_base));
            }
            if let Some(rows) = resolving {
                at.resize(self.core.ready(), rows[i]);
            }
            i += 1;
        }
        if consumed > 0 {
            self.last_row = Some(LastRow {
                frame: chunk.clone(),
                at: consumed - 1,
            });
            // The whole chunk is held as it is: no slice of every column.
            let kept = if consumed == chunk.height() {
                chunk
            } else {
                chunk.slice(0, consumed)
            };
            self.held.push_back(Arc::new(kept));
        }
        if consumed < n {
            self.rewind_increments(snapshot, df, &keys, &clock, session.as_deref(), consumed)?;
        }
        let e = match limit {
            None => self.core.drain(),
            Some(l) => self.core.drain_kept(l),
        };
        if resolving.is_some() {
            debug_assert_eq!(at.len(), e.rows);
            self.assemble_resolved(e, &at, defer)
        } else {
            self.assemble(e)
        }
    }

    /// [`Self::assemble`] for [`Self::feed_resolving`]: the formulas over
    /// every resolved row, null on a dropped one, with the row numbers.
    fn assemble_resolved(
        &mut self,
        e: crate::windows::Emitted,
        at: &[u64],
        defer: bool,
    ) -> PolarsResult<DataFrame> {
        let resolved_at: UInt64Chunked = at.iter().map(|&a| Some(a)).collect();
        let mut frame = self.assemble_rows(e)?;
        frame.hstack_mut(&[resolved_at.with_name("@po:at".into()).into_column()])?;
        if defer && self.formulas_pending() {
            return Ok(frame);
        }
        self.resolved_formulas(frame)
    }

    /// The formulas over resolved rows, as [`Self::feed_resolving`] gives
    /// them: the rows with their operators' hidden columns, `@po:row` and
    /// `@po:at`, in; the formulas' columns, `@po:seq` and `@po:at` out. The
    /// formulas are element-wise, so the rows may be of any number of runs
    /// of one call, stacked.
    ///
    /// # Errors
    ///
    /// A formula that cannot be evaluated over the rows, such as a strict
    /// cast that fails.
    pub fn resolved_formulas(&self, frame: DataFrame) -> PolarsResult<DataFrame> {
        // A formula that is only a window operator's values is its hidden
        // column under the formula's name: no plan of the lazy engine for
        // it, which a bank's group would pay every chunk.
        if let Some(names) = self.bare_formulas() {
            let mut cols: Vec<Column> = Vec::with_capacity(names.len() + 2);
            for (f, name) in self.k.config.formulas.iter().zip(names) {
                cols.push(
                    frame
                        .column(name.as_str())?
                        .clone()
                        .with_name(f.name.as_str().into()),
                );
            }
            cols.push(frame.column("@po:row")?.clone().with_name("@po:seq".into()));
            cols.push(frame.column("@po:at")?.clone());
            return DataFrame::new(frame.height(), cols);
        }
        // An operator partial under `"drop"` gives no value, so a formula
        // over it is null of itself; the row's other formulas keep theirs
        // (review R1, D4: every formula of the row was nulled).
        let exprs: Vec<Expr> = self
            .k
            .config
            .formulas
            .iter()
            .map(|f| {
                f.tree
                    .to_expr(&|op| col(hidden(op).as_str()))
                    .alias(f.name.as_str())
            })
            .collect();
        let keep: Vec<Expr> = self
            .k
            .config
            .formulas
            .iter()
            .map(|f| col(f.name.as_str()))
            .chain([col("@po:row").alias("@po:seq"), col("@po:at")])
            .collect();
        frame.lazy().with_columns(exprs).select(keep).collect()
    }

    /// Rows of this run's input consumed so far: skipped as already read, a
    /// loaded skip not yet applied (review R5, C1), or fed to the core.
    /// Under a slice, what a run resumed on the same input must skip
    /// ([`Self::save_bytes_with`]). Rows the state held from an earlier run
    /// and returned are none of this input's (review R4, B1 and B2).
    pub fn consumed(&self) -> usize {
        let fed = usize::try_from(self.core.next_seq() - self.run_base).expect("rows fit");
        usize::try_from(self.skipped).expect("rows fit") + self.resume_skip + fed
    }

    /// The first row's group key and session hash, as the core reads them.
    fn first_key_session(&self, df: &DataFrame) -> PolarsResult<(Option<String>, Option<u64>)> {
        let head = df.slice(0, 1);
        let key = match &self.k.config.group {
            None => Some(String::new()),
            Some(c) => key_text(head.column(c.as_str())?.as_materialized_series())?
                .str()?
                .get(0)
                .map(str::to_string),
        };
        let session = match &self.k.config.session {
            None => None,
            Some(c) => {
                let s = key_text(head.column(c.as_str())?.as_materialized_series())?;
                Some(session_hash(s.str()?.get(0)))
            }
        };
        Ok((key, session))
    }

    /// The identity of the input a sliced state resumes on (review R5, C3
    /// and C4; R6, D1): the rows skipped where the state was cut must be
    /// the last rows the state read -- the rows it held that are the
    /// input's own, or the last row alone where it held none. Another input
    /// that starts at the same clock, or the same input sliced by hand, is
    /// refused by name rather than fed with rows skipped or doubled.
    fn check_resumed_input(&mut self) -> PolarsResult<()> {
        let rows = std::mem::take(&mut self.resume_rows);
        let names: Vec<&str> = self.k.input.iter_names().map(|n| n.as_str()).collect();
        let stack = |parts: &[DataFrame]| -> PolarsResult<DataFrame> {
            let mut out = DataFrame::empty_with_schema(&self.k.input);
            for part in parts {
                out.vstack_mut(&part.select(names.iter().copied())?)?;
            }
            Ok(out)
        };
        if let Some(want) = self.resume_held.take() {
            // The last rows the state read, as loaded: a run may have
            // drained some of the held ones before the skip completes.
            let got = stack(&rows)?;
            if got.height() != want.height() || !got.equals_missing(&want) {
                polars_bail!(ComputeError:
                    "{WHO}: the state was saved from a slice of its input, and the last {} rows \
                     it read are not this input's rows at that place, so this is another input. \
                     Resume on the input the state was saved from, whole.",
                    want.height()
                );
            }
        }
        Ok(())
    }

    /// The rows a sliced state still expects to skip: an input that ends
    /// first is another input (review R5, C3).
    fn resumed_input_ended_early(&self) -> Option<String> {
        (self.resume_skip > 0).then(|| {
            format!(
                "{WHO}: the state was saved from a slice of its input after reading {} rows, \
                 but this input ended after {}, so it is another input. Resume on the input the \
                 state was saved from, whole.",
                self.resume_total, self.skipped
            )
        })
    }

    /// The end of the input: every row still held goes out, each forward
    /// window still open over it null.
    pub fn finish(&mut self) -> PolarsResult<DataFrame> {
        if let Some(msg) = self.resumed_input_ended_early() {
            polars_bail!(ComputeError: "{}", msg);
        }
        let e = self.core.finish();
        self.assemble(e)
    }

    /// The whole input is fed and its state is about to be saved to go on
    /// ([`Self::save_bytes`]): the rows whose backward windows were waiting
    /// on the last stamp, and every row after them whose forward windows
    /// have closed, with the windows read as they stand. A run resumed on
    /// this state is refused an input that starts at that stamp, so no row
    /// can join them later, and the next run reads the same values one run
    /// over the whole stream gives.
    ///
    /// # Errors
    ///
    /// An input the state was to skip that ended first, as [`Self::finish`].
    pub fn settle(&mut self) -> PolarsResult<DataFrame> {
        if let Some(msg) = self.resumed_input_ended_early() {
            polars_bail!(ComputeError: "{}", msg);
        }
        self.core.settle();
        let e = self.core.drain();
        self.assemble(e)
    }

    /// Put the increments' state back to `snapshot` and run the first
    /// `rows` rows of `df` through it again, so it ends where the core
    /// stopped: the values the held rows carry are the same, since an
    /// increment reads only the rows before it.
    #[allow(clippy::type_complexity)]
    fn rewind_increments(
        &mut self,
        snapshot: (
            Vec<IncrState>,
            HashMap<String, usize>,
            Option<usize>,
            Option<ClockValue>,
        ),
        df: &DataFrame,
        keys: &[Option<&str>],
        clock: &Option<Vec<ClockValue>>,
        session: Option<&[u64]>,
        rows: usize,
    ) -> PolarsResult<()> {
        (
            self.incr_state,
            self.incr_index,
            self.incr_null,
            self.stream_clock,
        ) = snapshot;
        if rows > 0 {
            let clock = clock.as_ref().map(|c| c[..rows].to_vec());
            self.with_increments(
                &df.slice(0, rows),
                keys.get(..rows).unwrap_or(&[]),
                &clock,
                session.map(|s| &s[..rows]),
                None,
            )?;
        }
        Ok(())
    }

    /// The chunk with one hidden column per increment: `x_i - x_{i-1}`
    /// within the group's session, null on a session's first row and after
    /// a step back the policy reads as a new start; seconds for a temporal
    /// input. `given` is the increments' inputs where a caller has
    /// evaluated them (see [`Self::share`]); the lazy engine does it here
    /// otherwise.
    fn with_increments(
        &mut self,
        df: &DataFrame,
        keys: &[Option<&str>],
        clock: &Option<Vec<ClockValue>>,
        session: Option<&[u64]>,
        given: Option<IncrInputs>,
    ) -> PolarsResult<DataFrame> {
        if self.k.increments.is_empty() {
            return Ok(df.clone());
        }
        let inputs = match given {
            Some(read) => read,
            None => self.increment_inputs(df)?,
        };
        let policy = self.restart_policy();
        let out = step_increments(
            IncrTrack {
                state: &mut self.incr_state,
                index: &mut self.incr_index,
                null: &mut self.incr_null,
                stream_clock: &mut self.stream_clock,
            },
            (&self.k.increments, &self.k.slots),
            policy,
            &inputs,
            keys,
            clock,
            session,
        );
        let cols = self.increment_columns(out);
        let mut chunk = df.clone();
        chunk.hstack_mut(&cols)?;
        Ok(chunk)
    }

    /// The clock policy's reading of a step back: the span past which it is
    /// a new start, whether a step back starts anew or skips stale rows, the
    /// gap past which a drift's stretch ends, and whether groups are present.
    fn restart_policy(&self) -> IncrPolicy {
        let cfg = self.core.clock_cfg();
        IncrPolicy {
            restart: cfg.min_backwards_jump,
            restarts: cfg.on_clock_reset.resets(),
            skips: cfg.on_clock_reset.skips(),
            gap_cap: cfg.gap_cap,
            grouped: self.k.config.group.is_some(),
        }
    }

    /// The increments' values as the hidden columns they stand in as.
    fn increment_columns(&self, out: Vec<Vec<Option<f64>>>) -> Vec<Column> {
        self.k
            .increments
            .iter()
            .zip(out)
            .map(|(r, values)| {
                let ca: Float64Chunked = values.into_iter().collect();
                ca.with_name(r.key.as_str().into()).into_column()
            })
            .collect()
    }

    /// The increments' inputs over `df`, as one plan of the lazy engine.
    fn increment_inputs(&self, df: &DataFrame) -> PolarsResult<IncrInputs> {
        let inputs = df
            .clone()
            .lazy()
            .select(
                self.k
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(i, (node, _))| {
                        node.to_expr(&|_| unreachable!("no operator in a row operator's input"))
                            .alias(input_column(i).as_str())
                    })
                    .collect::<Vec<_>>(),
            )
            .collect()?;
        self.read_increment_inputs(&inputs)
    }

    /// Each increment's input read in the form its step is taken in.
    fn read_increment_inputs(&self, inputs: &DataFrame) -> PolarsResult<IncrInputs> {
        let width = self.k.slots.len();
        // An integer input's values as integers (task 200): its step is
        // taken in integers and then converted, where two doubles of an
        // epoch-nanosecond column near 1.8e18 tied at a step of 1.
        let integer: Vec<Option<Vec<Option<i128>>>> = (0..width)
            .map(|i| {
                let c = inputs.column(input_column(i).as_str())?;
                if self.k.slots[i].1 || !integer_input(c.dtype()) {
                    Ok(None)
                } else {
                    integer_values(c.as_materialized_series()).map(Some)
                }
            })
            .collect::<PolarsResult<_>>()?;
        let numeric: Vec<Option<Vec<Option<f64>>>> = (0..width)
            .map(|i| {
                if self.k.slots[i].1 || integer[i].is_some() {
                    Ok(None)
                } else {
                    let s = inputs
                        .column(input_column(i).as_str())?
                        .cast(&DataType::Float64)?;
                    Ok(Some(s.f64()?.iter().collect()))
                }
            })
            .collect::<PolarsResult<_>>()?;
        let temporal: Vec<Option<Vec<Option<i64>>>> = (0..width)
            .map(|i| {
                if self.k.slots[i].1 {
                    // A time of day is read too, in nanoseconds since
                    // midnight (review round 4, PD5).
                    let c = inputs.column(input_column(i).as_str())?;
                    let ns = nanos_array(
                        c.as_materialized_series(),
                        &crate::arrow::InputRows::default(),
                        NanosRole::Increment,
                        WHO,
                    )?;
                    Ok(Some((0..ns.len()).map(|r| ns.get(r)).collect()))
                } else {
                    Ok(None)
                }
            })
            .collect::<PolarsResult<_>>()?;
        Ok(IncrInputs {
            rows: inputs.height(),
            integer,
            numeric,
            temporal,
        })
    }

    /// What a chunk's groups' runs share, evaluated once over the whole
    /// chunk instead of once for each group (a bank's resolver, which feeds
    /// every group's run its own rows of one frame): the operator inputs
    /// and, where the runs have increments, the increments' inputs. Every
    /// one is an element-wise formula of the row, so a value does not
    /// depend on the rows beside it; the increments themselves are a group's
    /// own, which [`Self::preview_increments`] reads off each run's state
    /// without moving it, so that the operator inputs that read them can be
    /// evaluated over the chunk too. `groups` is each run's rows of `frame`,
    /// in `runs`' order. `None` where there is nothing to save, a feed
    /// plans for itself -- one group; inputs that are all columns of doubles
    /// -- or where a plan refuses, which a feed reports for the rows it was
    /// handed.
    pub fn share(runs: &[WindowsRun], frame: &DataFrame, groups: &[&[usize]]) -> Option<Shared> {
        let first = runs.first()?;
        if first.k.increments.is_empty() {
            return first.evaluate_inputs(frame).map(|inputs| Shared {
                inputs: Some(inputs),
                incr: None,
            });
        }
        first.share_increments(runs, frame, groups)
    }

    /// [`Self::share`] for runs with increments.
    fn share_increments(
        &self,
        runs: &[WindowsRun],
        frame: &DataFrame,
        groups: &[&[usize]],
    ) -> Option<Shared> {
        // A run keeps its rows' increments by group: only a bank's, whose
        // run is one group's, is read here. One that skips rows its input
        // read before it was saved is left to feed for itself.
        if self.k.config.group.is_some() || runs.iter().any(|r| r.resume_skip > 0) {
            return None;
        }
        let n = frame.height();
        let inputs = frame
            .clone()
            .lazy()
            .select(
                self.k
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(i, (node, _))| {
                        node.to_expr(&|_| unreachable!("no operator in a row operator's input"))
                            .alias(input_column(i).as_str())
                    })
                    .collect::<Vec<_>>(),
            )
            .collect()
            .ok()?;
        // A literal is one value, which a feed broadcasts.
        if inputs.height() != n {
            return None;
        }
        let read = self.read_increment_inputs(&inputs).ok()?;
        let clock = self.clock_values(frame).ok()?;
        let session = self.session_hashes(frame).ok()?;
        let mut values: Vec<Vec<Option<f64>>> = vec![vec![None; n]; self.k.increments.len()];
        for (run, rows) in runs.iter().zip(groups) {
            let keys = vec![Some(""); rows.len()];
            let clock = clock
                .as_ref()
                .map(|c| rows.iter().map(|&i| c[i]).collect::<Vec<_>>());
            let session: Option<Vec<u64>> = session
                .as_ref()
                .map(|s| rows.iter().map(|&i| s[i]).collect());
            let out = run.preview_increments(&read.gather(rows), &keys, &clock, session.as_deref());
            for (all, part) in values.iter_mut().zip(out) {
                for (&i, v) in rows.iter().zip(part) {
                    all[i] = v;
                }
            }
        }
        let mut chunk = frame.clone();
        chunk.hstack_mut(&self.increment_columns(values)).ok()?;
        let inputs = if self.k.inputs.is_empty() || self.bare_inputs(&chunk).is_some() {
            None
        } else {
            Some(self.evaluated_inputs(&chunk).ok()?)
        };
        Some(Shared {
            inputs,
            incr: Some(read),
        })
    }

    /// The increments of `rows` rows as [`Self::with_increments`] would give
    /// them, with this run's state left as it is.
    fn preview_increments(
        &self,
        inputs: &IncrInputs,
        keys: &[Option<&str>],
        clock: &Option<Vec<ClockValue>>,
        session: Option<&[u64]>,
    ) -> Vec<Vec<Option<f64>>> {
        let mut state = self.incr_state.clone();
        let mut index = self.incr_index.clone();
        let mut null = self.incr_null;
        let mut stream_clock = self.stream_clock;
        step_increments(
            IncrTrack {
                state: &mut state,
                index: &mut index,
                null: &mut null,
                stream_clock: &mut stream_clock,
            },
            (&self.k.increments, &self.k.slots),
            self.restart_policy(),
            inputs,
            keys,
            clock,
            session,
        )
    }

    /// The session column's values, each as the hash the core reads. Keys
    /// are text, a zoned Datetime's its instant, as the bank keys them: the
    /// cast to text failed on one (task 160, PA4b).
    fn session_hashes(&self, df: &DataFrame) -> PolarsResult<Option<Vec<u64>>> {
        self.k
            .config
            .session
            .as_ref()
            .map(|c| session_hashes(df.column(c.as_str())?.as_materialized_series()))
            .transpose()
    }

    /// Clock values of a chunk, in the form the column has them: a temporal
    /// clock as nanoseconds, a numeric one as numbers; null, NaN and
    /// infinity refused by row, and the other kind than the state's.
    fn clock_values(&self, df: &DataFrame) -> PolarsResult<Option<Vec<ClockValue>>> {
        let Some(c) = &self.k.config.clock else {
            return Ok(None);
        };
        let col = df.column(c.as_str())?;
        let base = self.core.next_seq() - self.run_base + self.skipped;
        let bad = |i: usize| {
            polars_err!(ComputeError:
                "{WHO}: clock column {c:?} is null, NaN or infinite at row {} (counting from \
                 0). Drop the rows whose clock has no value, or fill it in, before the call.",
                base + i as u64)
        };
        let values: Vec<ClockValue> = if col.dtype().is_temporal() {
            let ns = nanos_array(
                col.as_materialized_series(),
                &crate::arrow::InputRows::from_base(base as usize),
                NanosRole::Clock,
                WHO,
            )?;
            (0..ns.len())
                .map(|i| ns.get(i).map(ClockValue::Ns).ok_or_else(|| bad(i)))
                .collect::<PolarsResult<_>>()?
        } else if col.dtype().is_integer() && fits_64(col.dtype()) {
            // An integer clock as its integers (task 200), as the bank reads
            // one; an unsigned value past an `i64` is refused by row.
            integer_values(col.as_materialized_series())?
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    let v = v.ok_or_else(|| bad(i))?;
                    i64::try_from(v).map(ClockValue::I64).map_err(|_| {
                        polars_err!(ComputeError:
                            "{WHO}: clock column {c:?} is {v} at row {} (counting from 0), \
                             past {}, the largest value an integer clock holds (an Int64's). \
                             Cast the clock to Int64 after subtracting an origin from it.",
                            base + i as u64, i64::MAX)
                    })
                })
                .collect::<PolarsResult<_>>()?
        } else if col.dtype().is_integer() {
            return Err(wrong_type(wide_integer_clock(WHO, c, col.dtype())));
        } else {
            let s = col.cast(&DataType::Float64)?;
            s.f64()?
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.filter(|v| v.is_finite())
                        .map(ClockValue::F64)
                        .ok_or_else(|| bad(i))
                })
                .collect::<PolarsResult<_>>()?
        };
        if let (Some(first), Some(last)) = (values.first(), self.core.shared_clock().last_clock())
            && kind_of(*first) != kind_of(last)
        {
            polars_bail!(ComputeError:
                "{WHO}: clock column {c:?} is {}, but the state it resumes read {} one. Resume \
                 the state on a clock of the kind it was saved with.",
                crate::arrow::dtype_name(col.dtype()),
                kind_of(last).name()
            );
        }
        Ok(Some(values))
    }

    /// The cross-group operator `op` (its index among them) in the reader's
    /// words: its name and the expressions that read it.
    fn cross_name(&self, op: usize) -> String {
        let at = self.core.ops().len() + op;
        let Some((key, _)) = self.k.columns.iter().find(|(_, o)| *o == at) else {
            return "a cross-group operator".to_owned();
        };
        let mut kind = "a cross-group operator";
        let mut names = Vec::new();
        for f in &self.k.config.formulas {
            for o in f.tree.operators() {
                if hidden(o) == *key {
                    kind = o.kind.name();
                    if !names.contains(&f.name.as_str()) {
                        names.push(f.name.as_str());
                    }
                }
            }
        }
        format!("{kind} (in {})", names.join(", "))
    }

    /// A profile operator by the formulas it is in, as an error names it.
    fn profile_name(&self, op: usize) -> String {
        let out = self.core.ops().len() + self.core.cross_defs().len() + op;
        let key = self
            .k
            .columns
            .iter()
            .find(|(_, i)| *i == out)
            .map(|(k, _)| k);
        let names: Vec<&str> = self
            .k
            .config
            .formulas
            .iter()
            .filter(|f| key.is_some_and(|k| f.tree.operators().iter().any(|o| hidden(o) == *k)))
            .map(|f| f.name.as_str())
            .collect();
        format!("ewm_profile (in {})", names.join(", "))
    }

    /// The operator a state cap or warning is about, in the reader's words.
    fn state_name(&self, of: StateOf, op: usize) -> String {
        match of {
            StateOf::Cross => self.cross_name(op),
            StateOf::Profile => self.profile_name(op),
        }
    }

    /// Summary notices for stale rows skipped under `on_step_back = "skip"`,
    /// drained once per call.
    pub fn take_notices(&mut self) -> Vec<String> {
        let Some((rows, groups)) = self.core.take_step_back_counts() else {
            return Vec::new();
        };
        let row_word = if rows == 1 { "row" } else { "rows" };
        let msg = if self.k.config.group.is_some() {
            let grp_word = if groups == 1 { "group" } else { "groups" };
            format!(
                "{WHO}: skipped {rows} stale {row_word} across {groups} {grp_word} under \
                 on_step_back=\"skip\" (clock stepped backward; formula outputs left null and \
                 window state unchanged)."
            )
        } else {
            format!(
                "{WHO}: skipped {rows} stale {row_word} under on_step_back=\"skip\" (clock \
                 stepped backward; formula outputs left null and window state unchanged)."
            )
        };
        vec![msg]
    }

    /// Persisted stale-row skip total across all groups.
    pub fn total_skipped_step_back(&self) -> u64 {
        self.core.total_skipped_step_back()
    }

    /// Persisted stale-row skip counts by group key.
    pub fn skipped_step_back_by_group(&self) -> Vec<(Option<String>, u64)> {
        self.core.skipped_step_back_by_group()
    }

    /// The first row to take a cross-group operator or a profile past half
    /// of its `max_state_bytes`, once a run, as the warning a caller raises.
    pub fn take_warning(&mut self) -> Option<String> {
        let (of, op, bytes, cap) = self.core.take_state_warning()?;
        let (held, then, fix) = match of {
            StateOf::Cross => (
                "the rows of one stamp of the clock",
                "a stamp that holds twice as many rows will be refused",
                "Raise max_state_bytes, or use closed=\"left\", which holds no row for its \
                 output.",
            ),
            StateOf::Profile => (
                "one group",
                "a group whose state grows to twice that will be refused",
                "Raise max_state_bytes, or shorten window_size.",
            ),
        };
        Some(format!(
            "{WHO}: {} holds {bytes} bytes for {held}, past half of its max_state_bytes of \
             {cap}; {then}. {fix}",
            self.state_name(of, op)
        ))
    }

    /// A row the core refused, as the error naming it.
    fn refusal(&self, r: Refusal, _chunk_base: u64) -> PolarsError {
        let row = |seq: u64| seq - self.run_base + self.skipped;
        match r {
            Refusal::StateCap {
                seq,
                of,
                op,
                bytes,
                cap,
            } => {
                let name = self.state_name(of, op);
                match of {
                    StateOf::Cross => {
                        polars_err!(ComputeError:
                            "{WHO}: {name} would hold {bytes} bytes for the rows of one stamp of the clock \
                             at row {} (counting from 0), past its max_state_bytes of {cap}. Under \
                             closed=\"right\" each row of a stamp is held until the stamp closes; a \
                             rank, a quantile or a read over neighbors also keeps each row's value. \
                             Raise max_state_bytes, or use closed=\"left\", which holds no row for its \
                             output.",
                            row(seq)
                        )
                    }
                    StateOf::Profile => {
                        polars_err!(ComputeError:
                            "{WHO}: {name} would hold {bytes} bytes for one group at row {} (counting \
                             from 0), past its max_state_bytes of {cap}: its buckets, and under a \
                             window_size the rows each keeps, or a quantile's sketches. Raise \
                             max_state_bytes, or shorten window_size, or take fewer buckets.",
                            row(seq)
                        )
                    }
                }
            }
            Refusal::CrossRefTwice { seq, op, first } => {
                let name = self.cross_name(op);
                polars_err!(ComputeError:
                    "{WHO}: {name} picks row {} (counting from 0) as the reference row of its \
                     stamp, but it picked row {} at the same stamp already, and a stamp has one \
                     reference row. Change where= to pick one row a stamp, such as by naming \
                     one group's key.",
                    row(seq),
                    row(first)
                )
            }
            Refusal::Backwards {
                seq,
                back,
                prev,
                now,
                min_backwards_jump,
            } => {
                let column = self.k.config.clock.as_deref().unwrap_or("<row count>");
                // A step that is no number (the core refuses it like a step
                // back): two finite values whose difference is past the
                // largest double, since a value that is not finite was
                // refused by row before it (task 270, F-1).
                if !back.is_finite() {
                    let label = |v: Option<ClockValue>| {
                        v.map_or_else(String::new, |v| crate::spec::num_label(v.seconds()))
                    };
                    return polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} steps from {} to {} at row {} (counting \
                         from 0), a difference past the largest float, which cannot be a number \
                         of clock units. Shift or rescale the clock column so that the step between two \
                         of its values is finite.",
                        label(prev),
                        label(now),
                        row(seq)
                    );
                }
                let step = match (now, prev) {
                    (Some(ClockValue::Ns(c)), Some(ClockValue::Ns(p))) => {
                        i64::try_from(i128::from(p) - i128::from(c))
                            .map_or_else(|_| back.to_string(), format_duration)
                    }
                    // In integers, exactly (task 200).
                    (Some(ClockValue::I64(c)), Some(ClockValue::I64(p))) => {
                        (i128::from(p) - i128::from(c)).to_string()
                    }
                    _ => crate::spec::num_label(back),
                };
                match min_backwards_jump {
                    None => polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} steps back by {step} at row {} \
                         (counting from 0), and restart_after_step_back is not set, so every \
                         step back is refused. Sort the whole input by the clock, across groups, \
                         or set \
                         restart_after_step_back below the step to reset the windows at such a \
                         step back (0 resets them at every one).",
                        row(seq)
                    ),
                    Some(m) => polars_err!(ComputeError:
                        "{WHO}: clock column {column:?} steps back by {step} at row {} \
                         (counting from 0), no more than restart_after_step_back = {}, so the \
                         step back is refused. Sort the input by the clock, or set \
                         restart_after_step_back below the step to reset the windows at such a \
                         step back (0 resets them at every one).",
                        row(seq),
                        self.k.config.restart_after_step_back.as_ref().map_or_else(|| crate::spec::num_label(m), ToString::to_string)
                    ),
                }
            }
            Refusal::OverCap {
                seq,
                rows,
                row_bytes,
                cap,
            } => polars_err!(ComputeError:
                "{WHO}: at row {} (counting from 0) a group's rolling_max or rolling_min window \
                 holds {rows} rows of {row_bytes} bytes each, {} bytes, past max_state_bytes of \
                 {cap}. A window on a clock holds every row inside its span, so a burst of rows \
                 fills it. Shorten window_size, or raise max_state_bytes (at most 64 MiB).",
                row(seq),
                rows * row_bytes
            ),
            Refusal::Span { seq, origin, now } => {
                let column = self.k.config.clock.as_deref().unwrap_or("<row count>");
                polars_err!(ComputeError:
                    "{WHO}: clock column {column:?} has run from {} to {} at row {} (counting \
                     from 0) since its windows last started over, a distance past the largest \
                     float, which cannot be a number of clock units. Shift or rescale the clock \
                     column, or set gap_cap or a session so that the windows start over within \
                     a finite distance.",
                    crate::spec::num_label(origin.seconds()),
                    crate::spec::num_label(now.seconds()),
                    row(seq)
                )
            }
        }
    }

    /// The first `e.rows` held rows with the formulas evaluated beside
    /// them, less the dropped ones and the hidden columns.
    fn assemble(&mut self, e: crate::windows::Emitted) -> PolarsResult<DataFrame> {
        let drop = e.drop.clone();
        let mut stale = e.stale.clone();
        let mut frame = self.assemble_rows(e)?;
        if drop.iter().any(|&d| d) {
            let keep: BooleanChunked = drop.iter().map(|&d| Some(!d)).collect();
            frame = frame.filter(&keep)?;
            stale = stale
                .into_iter()
                .zip(&drop)
                .filter_map(|(s, &d)| (!d).then_some(s))
                .collect();
        }
        let any_stale = stale.iter().any(|&s| s);
        // A formula that is only an operator's values is its hidden column
        // under the formula's name: no plan of the lazy engine for it, which
        // costs 100 to 150 microseconds a chunk whatever the chunk holds.
        if let Some(names) = self.bare_formulas() {
            let mut cols: Vec<Column> = frame
                .columns()
                .iter()
                .filter(|c| self.k.input.contains(c.name().as_str()))
                .cloned()
                .collect();
            for (f, name) in self.k.config.formulas.iter().zip(names) {
                cols.push(
                    frame
                        .column(name.as_str())?
                        .clone()
                        .with_name(f.name.as_str().into()),
                );
            }
            return DataFrame::new(frame.height(), cols);
        }
        if any_stale {
            let mask: BooleanChunked = stale.iter().map(|&s| Some(s)).collect();
            frame.hstack_mut(&[mask.with_name("@po:stale".into()).into_column()])?;
        }
        // Pass four: the formulas, over the rows with their operators.
        let exprs: Vec<Expr> = self
            .k
            .config
            .formulas
            .iter()
            .map(|f| {
                let expr = f.tree.to_expr(&|op| col(hidden(op).as_str()));
                let expr = if any_stale {
                    when(col("@po:stale")).then(lit(NULL)).otherwise(expr)
                } else {
                    expr
                };
                expr.alias(f.name.as_str())
            })
            .collect();
        // The input's columns the chunk has (a projection may have narrowed
        // it), then the formulas; the hidden columns stay behind.
        let present: Vec<String> = frame
            .get_column_names()
            .iter()
            .map(|n| n.to_string())
            .filter(|n| self.k.input.contains(n.as_str()))
            .collect();
        let keep: Vec<Expr> = present
            .iter()
            .map(|n| col(n.as_str()))
            .chain(self.k.config.formulas.iter().map(|f| col(f.name.as_str())))
            .collect();
        frame.lazy().with_columns(exprs).select(keep).collect()
    }

    /// The emitted rows as one frame: the held input rows with their
    /// increment columns, and each operator's values as its hidden column.
    fn assemble_rows(&mut self, mut e: crate::windows::Emitted) -> PolarsResult<DataFrame> {
        let mut out: Option<DataFrame> = None;
        let mut parts = 0;
        let mut left = e.rows;
        while left > 0 {
            let h = self.held.front().map_or(0, |f| f.height());
            assert!(h > 0, "the core holds no row the frames do not");
            let part = if h <= left {
                let whole = self.held.pop_front().expect("checked");
                Arc::try_unwrap(whole).unwrap_or_else(|shared| (*shared).clone())
            } else {
                let front = self.held.front_mut().expect("checked");
                let part = front.slice(0, left);
                *front = Arc::new(front.slice(left as i64, h - left));
                part
            };
            left -= part.height();
            parts += 1;
            match &mut out {
                None => out = Some(part),
                Some(o) => {
                    o.vstack_mut(&part)?;
                }
            }
        }
        // No emitted row: an empty frame of the held rows' schema. Built
        // only then, not for every chunk.
        let mut frame = out.unwrap_or_else(|| {
            let mut held_schema = self.k.input.clone();
            for k in self.hidden_names() {
                held_schema.insert(k.as_str().into(), DataType::Float64);
            }
            DataFrame::empty_with_schema(&held_schema)
        });
        // Many one-row chunks make every later operation slow; a few are
        // what vstack is for.
        if parts > 8 {
            frame.rechunk_mut();
        }
        let mut cols: Vec<Column> = Vec::with_capacity(self.k.columns.len());
        for (name, o) in &self.k.columns {
            cols.push(nullable_column(name, std::mem::take(&mut e.values[*o])));
        }
        frame.hstack_mut(&cols)?;
        Ok(frame)
    }

    /// The state as bytes: the call, the core, the rows it holds, which the
    /// next run emits first, and the increments' state. Versioned msgpack,
    /// one state one file.
    pub fn save_bytes(&self) -> Result<Vec<u8>, String> {
        self.save_bytes_with(0, false)
    }

    /// [`Self::save_bytes`] for a state saved under a slice: a run resumed
    /// on the same input skips its first `skip_on_resume` rows, the rows
    /// consumed so far ([`Self::consumed`]); the state knows the input by
    /// its first row's clock and session, and by the last rows it read: the
    /// rows it holds that are the input's, or the last row alone (review
    /// R2, W4; R4, B1 to B3; R5, C1 to C4; R6, D1; R7, E2). `input_ended`
    /// says the input ran out rather
    /// than a slice being satisfied: a skip still pending is then another
    /// input, refused.
    pub fn save_bytes_with(
        &self,
        skip_on_resume: usize,
        input_ended: bool,
    ) -> Result<Vec<u8>, String> {
        // Only the caller knows whether the input ended or a slice was
        // satisfied first; a skip still pending is another input only in
        // the first case (review R5, C3).
        if input_ended && let Some(msg) = self.resumed_input_ended_early() {
            return Err(msg);
        }
        let held_schema = {
            let mut s = self.k.input.clone();
            for k in self.hidden_names() {
                s.insert(k.as_str().into(), DataType::Float64);
            }
            s
        };
        let mut rows = DataFrame::empty_with_schema(&held_schema);
        for part in &self.held {
            rows.vstack_mut(part).map_err(|e| e.to_string())?;
        }
        // One chunk: the IPC writer writes a record batch per chunk, so
        // held rows still in the chunks they came in saved to other bytes
        // after 600 chunks of one row than after one chunk (docs/PLAN.md
        // task 269, review round 8 B-1). The bytes are the rows' alone.
        rows.rechunk_mut();
        compact_views(&mut rows).map_err(|e| e.to_string())?;
        let mut held = Vec::new();
        IpcWriter::new(&mut held)
            .finish(&mut rows)
            .map_err(|e| e.to_string())?;
        // The last row read is a sliced state's identity and nothing else's
        // (review R7, E3): the bank saves every core unsliced.
        let mut last_row = Vec::new();
        if let (true, Some(r)) = (skip_on_resume > 0, &self.last_row) {
            let names: Vec<&str> = self.k.input.iter_names().map(|n| n.as_str()).collect();
            let mut r = r
                .row()
                .select(names.iter().copied())
                .map_err(|e| e.to_string())?;
            compact_views(&mut r).map_err(|e| e.to_string())?;
            IpcWriter::new(&mut last_row)
                .finish(&mut r)
                .map_err(|e| e.to_string())?;
        }
        let mut out = rmp_serde::to_vec_named(&FileOut {
            magic: WINDOWS_MAGIC,
            kind: WINDOWS_KIND,
            version: WINDOWS_VERSION,
            config: &self.k.config,
            core: &self.core,
            held: &held,
            increments: &self.incr_state,
            stream_clock: self.stream_clock,
            resume_skip: skip_on_resume,
            // A run that saw no row keeps the identity it loaded (review R5, C1).
            resume_first: if self.started {
                self.first_clock
            } else {
                self.resume_first
            },
            resume_first_session: if self.started {
                self.first_session
            } else {
                self.resume_first_session
            },
            last_row: &last_row,
            key_dtypes: &self.k.key_dtypes,
            checksum: Some(crate::checksum::Trailer::Patched),
        })
        .map_err(|e| e.to_string())?;
        crate::checksum::seal(&mut out);
        Ok(out)
    }

    /// [`Self::save_bytes`] to `path`, replacing it whole or not at all.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.save_with(path, 0, false)
    }

    /// [`Self::save_bytes_with`] to `path`, replacing it whole or not at
    /// all.
    pub fn save_with(
        &self,
        path: &std::path::Path,
        skip_on_resume: usize,
        input_ended: bool,
    ) -> std::io::Result<()> {
        let bytes = self
            .save_bytes_with(skip_on_resume, input_ended)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        crate::atomic::write(path, &bytes)
    }

    /// A run that goes on from a saved state: the call must be the one that
    /// saved it, and the input must have the columns it held. Rows are
    /// counted from the start of the new input, the rows the state says to
    /// skip included ([`Self::save_bytes_with`]).
    ///
    /// # Errors
    ///
    /// Bytes that are not a windows state, a version this build does not
    /// read, another call, or held rows whose columns the input has not got.
    pub fn load_bytes(
        bytes: &[u8],
        config: WindowsConfig,
        input: &Schema,
    ) -> Result<Self, Refused> {
        // The checksum first, then the envelope's head, before a byte is
        // decoded: damage that decodes would load in silence, or panic or
        // abort further in.
        const START_OVER: &str = "Load an undamaged copy of the state file, or leave out \
                                  load_state to start over.";
        let opened = crate::envelope::open(bytes, crate::envelope::Kind::Windows).map_err(|r| {
            use crate::envelope::Refused as Why;
            Refused::from(match r {
                Why::Damaged | Why::Unsealed => {
                    format!("{WHO}: the state is {}. {START_OVER}", r.what())
                }
                Why::OtherKind(crate::envelope::Kind::Checkpoint) => format!(
                    "{WHO}: the state is a pipeline checkpoint, not a window run's state. \
                         Read its stages with rheon.polars.stream.load_checkpoint and pass the \
                         window run's stage as load_state."
                ),
                Why::OtherKind(_) | Why::UnknownKind(_) => format!(
                    "{WHO}: the state is {}, not a window run's. Pass a state file a \
                         window run saved, or leave out load_state to start over.",
                    r.what()
                ),
            })
        })?;
        let (version, sealed, enveloped) = match opened {
            crate::envelope::Opened::Enveloped { version } => (version, true, true),
            crate::envelope::Opened::Legacy { sealed } => {
                let header: Header = rmp_serde::from_slice(bytes).map_err(|e| {
                    format!(
                        "{WHO}: the state cannot be read ({e}), so it is damaged or was not \
                         saved by a window run. Pass a state file a window run saved, or leave \
                         out load_state to start over."
                    )
                })?;
                if header.magic != LEGACY_WINDOWS_MAGIC {
                    return Err(format!(
                        "{WHO}: the state was not saved by a window run. Pass a state file a \
                         window run saved, or leave out load_state to start over."
                    )
                    .into());
                }
                (header.version, sealed, false)
            }
        };
        // A version the envelope was never written at, or one written
        // before it with its head, is refused as any version this build
        // does not read.
        let reads = if enveloped {
            LAST_LEGACY_VERSION + 1..=WINDOWS_VERSION
        } else {
            MIN_WINDOWS_VERSION..=LAST_LEGACY_VERSION
        };
        if !reads.contains(&version) {
            return Err(format!(
                "{WHO}: the state's version is {version}, but this version of rheon reads \
                 versions {MIN_WINDOWS_VERSION} to {WINDOWS_VERSION}. Load the state with the \
                 version of rheon that saved it, or leave out load_state to start over."
            )
            .into());
        }
        if version >= SEALED_FROM_VERSION && !sealed {
            return Err(format!(
                "{WHO}: the state is damaged (the checksum it ends with is missing or cut). Load \
                 an undamaged copy of the state file, or leave out load_state to start over."
            )
            .into());
        }
        let mut file: FileIn = rmp_serde::from_slice(bytes).map_err(|e| {
            format!(
                "{WHO}: the state cannot be read ({e}), so it is damaged or was saved by \
                 another version of rheon. Load it with the version that saved it, or \
                 leave out load_state to start over."
            )
        })?;
        // A duration compares by its length, so "5000ms" is the call that
        // saved "5s" (task 160, PC3).
        if same_call(&file.config) != same_call(&config) {
            return Err(format!(
                "{WHO}: the state was saved by another call, and it resumes only the call that \
                 saved it, but {}. Give the call as the state was saved with it, or leave out \
                 load_state to start over.",
                call_differences(&same_call(&file.config), &same_call(&config))
            )
            .into());
        }
        let mut run = Self::new(config, input)?;
        if file.key_dtypes != run.k.key_dtypes {
            let name = |role: &str| {
                if role == "group" {
                    run.k.config.group.clone()
                } else {
                    run.k.config.session.clone()
                }
                .unwrap_or_default()
            };
            let saved: Vec<String> = file
                .key_dtypes
                .iter()
                .map(|(r, t)| format!("{r} column {:?}: {t}", name(r)))
                .collect();
            let now: Vec<String> = run
                .k
                .key_dtypes
                .iter()
                .map(|(r, t)| format!("{r} column {:?}: {t}", name(r)))
                .collect();
            let what = match (run.k.config.group.is_some(), run.k.config.session.is_some()) {
                (true, true) => "group and session columns",
                (true, false) => "group column",
                _ => "session column",
            };
            return Err(format!(
                "{WHO}: the state was saved with a {what} of another type ({}), and this \
                 input's ({}) may not give the same keys. Cast the column to the type the state \
                 was saved with.",
                saved.join(", "),
                now.join(", ")
            )
            .into());
        }
        // Versions 9 to 14 wrote a one-kernel, one-sum stand-in core for a
        // call of row operators alone (task 78); version 15 writes an empty
        // clock-only core. Check the loaded core's invariants first so damage
        // in the legacy stand-in is still refused, then strip the stand-in so
        // the core matches the v15 call and re-saves without it.
        if version <= LAST_PLACEHOLDER_VERSION && run.core.is_clock_only() {
            if !file.core.is_legacy_placeholder()
                || same_clock(file.core.clock_cfg()) != same_clock(run.core.clock_cfg())
            {
                return Err(format!(
                    "{WHO}: the state's window operators are not those of the call that saved it, \
                     so the state is damaged. Load an undamaged copy of the state file, or leave \
                     out load_state to start over."
                )
                .into());
            }
            file.core.check().map_err(|_| {
                format!(
                    "{WHO}: the state is damaged. Load an undamaged copy of the state file, or \
                     leave out load_state to start over."
                )
            })?;
            file.core.strip_legacy_placeholder();
        }
        if file.core.kernels() != run.core.kernels()
            || file.core.ops() != run.core.ops()
            || file.core.cross_defs() != run.core.cross_defs()
            || file.core.moments() != run.core.moments()
            || file.core.profiles() != run.core.profiles()
            || file.core.seasonals() != run.core.seasonals()
            || same_clock(file.core.clock_cfg()) != same_clock(run.core.clock_cfg())
        {
            return Err(format!(
                "{WHO}: the state's window operators are not those of the call that saved it, \
                 so the state is damaged. Load an undamaged copy of the state file, or leave out \
                 load_state to start over."
            )
            .into());
        }
        // The core's own invariants, beyond the call's (review 2026-10-06,
        // PD1): a damaged core that still decoded panicked at the next call.
        // The check's own words name the core's parts, which are this
        // library's, not the reader's: the reader is told the file is
        // damaged, and what to do.
        file.core.check().map_err(|_| {
            format!(
                "{WHO}: the state is damaged. Load an undamaged copy of the state file, or leave \
                 out load_state to start over."
            )
        })?;
        // Polars' IPC reader unwraps one field of a record batch's header,
        // so bytes damaged there panic inside it rather than return its
        // error (review P7): the panic is taken as the damage it reports.
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            IpcReader::new(Cursor::new(file.held)).finish()
        }));
        let held = match read {
            Ok(read) => read.map_err(|e| {
                format!(
                    "{WHO}: the rows the state holds cannot be read ({e}), so it is damaged. \
                     Load an undamaged copy of the state file, or leave out load_state to \
                     start over."
                )
            }),
            Err(_) => Err(format!(
                "{WHO}: the rows the state holds cannot be read (a record batch is out of \
                 spec), so it is damaged. Load an undamaged copy of the state file, or leave \
                 out load_state to start over."
            )),
        }?;
        if held.height() != file.core.held() {
            return Err(format!(
                "{WHO}: the state holds {} rows but counts {}, so it is damaged. Load an undamaged \
                 copy of the state file, or leave out load_state to start over.",
                held.height(),
                file.core.held()
            )
            .into());
        }
        let hidden_names = run.hidden_names();
        let held_input: Vec<(String, DataType)> = held
            .schema()
            .iter()
            .filter(|(n, _)| !hidden_names.iter().any(|h| h == n.as_str()))
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect();
        let want: Vec<(String, DataType)> = input
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect();
        if held.height() > 0 && held_input != want {
            return Err(format!(
                "{WHO}: the state holds {} {} with columns {:?}, but the input has {:?}. \
                 Resume the state on an input with the columns and types it was saved with.",
                held.height(),
                if held.height() == 1 { "row" } else { "rows" },
                held_input
                    .iter()
                    .map(|(n, t)| format!("{n}: {}", crate::arrow::dtype_name(t)))
                    .collect::<Vec<_>>(),
                want.iter()
                    .map(|(n, t)| format!("{n}: {}", crate::arrow::dtype_name(t)))
                    .collect::<Vec<_>>()
            )
            .into());
        }
        run.run_base = file.core.next_seq();
        run.core = file.core;
        if held.height() > 0 {
            run.held.push_back(Arc::new(held));
        }
        if file.resume_skip > 0 && !file.last_row.is_empty() {
            let last = crate::envelope::read_ipc(&file.last_row).map_err(|e| {
                format!(
                    "{WHO}: the last row the state read cannot be read back ({e}), so it is \
                         damaged. Load an undamaged copy of the state file, or leave out \
                         load_state to start over."
                )
            })?;
            let have: Vec<(String, DataType)> = last
                .schema()
                .iter()
                .map(|(n, t)| (n.to_string(), t.clone()))
                .collect();
            if have != want {
                return Err(format!(
                    "{WHO}: the last row the state read has columns {:?}, but the input has \
                     {:?}. Resume the state on the input it was saved from, whole, or on the \
                     next file, with the columns and types it was saved with.",
                    have.iter()
                        .map(|(n, t)| format!("{n}: {}", crate::arrow::dtype_name(t)))
                        .collect::<Vec<_>>(),
                    want.iter()
                        .map(|(n, t)| format!("{n}: {}", crate::arrow::dtype_name(t)))
                        .collect::<Vec<_>>()
                )
                .into());
            }
            run.last_row = Some(LastRow { frame: last, at: 0 });
        }
        run.stream_clock = file.stream_clock;
        run.resume_skip = file.resume_skip;
        run.resume_first = file.resume_first;
        run.resume_first_session = file.resume_first_session;
        run.resume_total = file.resume_skip;
        if file.resume_skip > 0 {
            // The identity of this input: the rows the state holds that are
            // this input's -- the last `resume_skip` of them at most, since
            // a state saved on the next file under a slice still holds the
            // previous file's unresolved rows ahead of its own, rows going
            // out in order (review R6, D2) -- or, where it holds none (an
            // operator that resolves a row at its push), the last row it
            // read (R6, D1).
            let names: Vec<&str> = run.k.input.iter_names().map(|n| n.as_str()).collect();
            let mut want = DataFrame::empty_with_schema(&run.k.input);
            for part in &run.held {
                want.vstack_mut(
                    &part
                        .select(names.iter().copied())
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
            let extra = want.height().saturating_sub(file.resume_skip);
            if extra > 0 {
                want = want.slice(extra as i64, want.height() - extra);
            }
            if want.height() == 0
                && let Some(last) = &run.last_row
            {
                want = last.row();
            }
            if want.height() == 0 {
                // This build writes the last row read under every skip, so a
                // state with none has lost it (review R8, F5): refused rather
                // than resumed on anything by a check that compares nothing.
                return Err(format!(
                    "{WHO}: the state was saved from a slice of {} rows but holds no row to \
                     know its input by, so it is damaged. Load an undamaged copy of the state \
                     file, or leave out load_state to start over.",
                    file.resume_skip
                )
                .into());
            }
            run.resume_held = Some(want);
        }
        // Each group's increment state holds one previous value per
        // increment in each of its forms, then the sign each tick sign carries,
        // and one state per key: a list cut short was indexed past its end at
        // the next row, and a second state for a key hid the first (review
        // 2026-11-23, WIN-11).
        let width = run.k.increments.len();
        let carried = run
            .k
            .increments
            .iter()
            .filter(|r| r.kind == OpKind::TickSign)
            .count();
        // A group keeps a state for every row operator where any is a run
        // length or a drift (task 564), and none where every one is a step
        // (an increment or a tick sign): a state of another shape is not
        // this call's.
        let keeps_rows = run.k.increments.iter().any(|r| !r.kind.is_step());
        if file.increments.iter().any(|st| {
            st.prev.len() != width + carried
                || st.rows.len() != if keeps_rows { width } else { 0 }
                || st.rows.iter().zip(&run.k.increments).any(|(s, r)| {
                    std::mem::discriminant(s) != std::mem::discriminant(&RowState::of(r))
                })
        }) {
            return Err(format!(
                "{WHO}: the state is damaged: its row operators (increment, tick_sign, \
                 run_length, clock_lag, clock_drift, clock_step) do not match its own call's. \
                 Load an undamaged copy of the state file, or leave out load_state to start over."
            )
            .into());
        }
        for (i, st) in file.increments.into_iter().enumerate() {
            let fits = st.prev.len() == width + carried
                && st.prev_ns.len() == width
                && st.prev_int.len() == width;
            let first = match &st.key {
                Some(k) => run.incr_index.insert(k.clone(), i).is_none(),
                None => run.incr_null.replace(i).is_none(),
            };
            if !(fits && first) {
                return Err(format!(
                    "{WHO}: the state is damaged. Load an undamaged copy of the state file, or \
                     leave out load_state to start over."
                )
                .into());
            }
            run.incr_state.push(st);
        }
        Ok(run)
    }
}

#[cfg(test)]
mod tests;
