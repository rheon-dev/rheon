//! A formula: a Polars expression over the window operators, kept as this
//! library's own compact tree (docs/PLAN.md task 143, decided 2026-10-02).
//!
//! Python reads an expression's own tree (`expr.meta.serialize`) at build
//! time and walks seven node kinds into nested lists; this side reads the
//! lists and rebuilds the expression through Polars' public builders, which
//! are stable where the serialized tree is not:
//!
//! ```text
//! ["-", ["rewm_mean", ["col", "mid"], {"half_life": "10s", "window_size": "1m"}], ["col", "mid"]]
//! ```
//!
//! Leaves are `["col", name]` and `["lit", value]`; the element-wise nodes
//! are the binary operators `+ - * / ** == != < <= > >= & |`, the unary
//! `neg abs exp sqrt is_null is_not_null`, `["log", x, base]`,
//! `["clip", x, lo, hi]` (either bound `["lit"]`, a bare `null` read too), `["fill_null", x, y]`,
//! `["when", p, then, else]`, `["cast", x, "Float64"]` and
//! `["alias", x, name]`. An operator is `[name, input, {params}]` for
//! `ewm_mean`, `rewm_mean`, `ewm_sum`, `rewm_sum`, `ewm_rate`,
//! `rewm_rate`, `ewm_var` and `ewm_std` (these two with a `bias` too),
//! `ewm_quantile` and `rewm_quantile` (with a `quantile`, the level),
//! `["ewm_moments", ["legs", leg, ...], {params}]` (a leg's power and lag
//! among its parameters), `["increment", input]` and `["tick_sign", input]`. The row
//! operators of task 564 are
//! `["run_length", input]` (with `{"abs_tol": x}` after it where `abs_tol` is above
//! 0) and `["clock_lag", arrival, {"event": e, "half_life": h}]` and
//! `["clock_drift", ...]` of the same form, and
//! `["clock_step", time, {"half_life": h, "stat": s}]` (with a `bias`
//! where `s` is `"var"` or `"std"`). Nothing else is read, so a
//! `shift`, a `cum_sum`, a rolling window or an aggregation -- which would
//! depend on the chunking (hard rule 3) -- is refused by name, at build time.

use std::fmt;

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::span::Span;
use crate::windows::{
    Closed, CrossStat, Direction, Interp, OpKind, PROFILE_STATE_BYTES_CEILING, Partial,
    ProfileStat, RankMethod, Stat,
};

/// A literal, as JSON carries it.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl Literal {
    fn from_json(v: &Value) -> Result<Self, String> {
        Ok(match v {
            Value::Null => Literal::Null,
            Value::Bool(b) => Literal::Bool(*b),
            Value::Number(n) => match n.as_i64() {
                Some(i) => Literal::Int(i),
                None => Literal::Float(n.as_f64().ok_or_else(|| {
                    format!(
                        "formula: a literal must be a number a double can hold, but it is \
                             {n}. Write the literal as a smaller number."
                    )
                })?),
            },
            Value::String(s) => Literal::Str(s.clone()),
            other => {
                return Err(format!(
                    "formula: a literal must be a number, a string, true, false or null, but it \
                     is {}. Write the literal as one of those.",
                    short(other)
                ));
            }
        })
    }

    fn to_json(&self) -> Value {
        match self {
            Literal::Null => Value::Null,
            Literal::Bool(b) => json!(b),
            Literal::Int(i) => json!(i),
            Literal::Float(f) => json!(f),
            Literal::Str(s) => json!(s),
        }
    }

    fn to_expr(&self) -> Expr {
        match self {
            Literal::Null => lit(NULL),
            Literal::Bool(b) => lit(*b),
            Literal::Int(i) => lit(*i),
            Literal::Float(f) => lit(*f),
            Literal::Str(s) => lit(s.as_str()),
        }
    }
}

/// A window operator: what it computes, over which input, on which kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct OpNode {
    pub kind: OpKind,
    /// An element-wise formula of the row, `increment` and `tick_sign`
    /// nodes included; no window operator.
    pub input: Box<Node>,
    /// Required for a window operator; none for `increment` or `tick_sign`.
    pub half_life: Option<Span>,
    pub window_size: Option<Span>,
    pub closed: Closed,
    pub min_samples: u32,
    /// Unset takes the direction's default: `keep` backward, `null` forward.
    pub partial: Option<Partial>,
    /// A variance's (`ewm_var`, `ewm_std`): uncorrected for the weights'
    /// count, Polars' `bias`; false for every other operator.
    pub bias: bool,
    /// A quantile's level (`ewm_quantile`, `rewm_quantile`), in (0, 1);
    /// `None` for every other operator.
    pub quantile: Option<f64>,
    /// A cross-group operator's weight, an element-wise formula of the row;
    /// `None` weighs every row 1, and every window operator has none.
    pub weight: Option<Box<Node>>,
    /// A cross-group operator's cap on its state, or an extremum's on its
    /// window's bytes a group, where the call sets one; `None` otherwise.
    pub max_state_bytes: Option<u64>,
    /// A cross rank's tie rule and direction, and a cross quantile's
    /// interpolation; the defaults for every other operator.
    pub order: Order,
    /// A cross operator's list column of neighbor keys: it reads the rows
    /// of those groups alone. `None` reads every group.
    pub neighbors: Option<String>,
    /// A reference read's predicate, an element-wise formula of the row:
    /// the row where it holds is the stamp's reference row.
    pub pick: Option<Box<Node>>,
    /// The term of `ewm_moments`, whose input is `["legs", ...]`, a formula
    /// a leg; `None` for every other operator.
    pub moments: Option<Box<MomentsNode>>,
    /// A profile's (`ewm_profile`) own parameters; `None` for every other
    /// operator.
    pub profile: Option<Box<ProfileParams>>,
    /// An extremum's event (`rolling_max`, `rolling_min`): an element-wise
    /// boolean of the row, true where the window starts over at the row's
    /// stamp; `None` for every other operator.
    pub since: Option<Box<Node>>,
    /// A run length's tolerance (`run_length`), at least 0; 0 for every
    /// other operator.
    pub tol: f64,
    /// A clock drift's event time (`clock_lag`, `clock_drift`), whose
    /// arrival time is `input`; `None` for every other operator.
    pub event: Option<Box<Node>>,
    /// What a clock step (`clock_step`) reads of its time's steps; `None`
    /// for every other operator. A spread's `bias` is [`OpNode::bias`].
    pub step: Option<rheon_core::StepStat>,
    /// An STL or MSTL operator's (`stl`, `mstl`) own parameters; `None` for
    /// every other operator.
    pub stl: Option<Box<StlParams>>,
}

/// What an `stl` or `mstl` takes: statsmodels' settings, the window, the
/// lag, which ring it reads and the component it emits.
#[derive(Debug, Clone, PartialEq)]
pub struct StlParams {
    /// `(period, seasonal span)` per season, shortest period first.
    pub seasons: Vec<(u32, u32)>,
    pub iterate: u32,
    pub trend: Option<u32>,
    pub low_pass: Option<u32>,
    pub seasonal_deg: u8,
    pub trend_deg: u8,
    pub low_pass_deg: u8,
    pub seasonal_jump: u32,
    pub trend_jump: u32,
    pub low_pass_jump: u32,
    pub robust: bool,
    pub inner_iter: Option<u32>,
    pub outer_iter: Option<u32>,
    /// Unset: the embargo's `2 h + 1`.
    pub window: Option<u32>,
    /// Unset: `h`, plus 1 under `"left"`.
    pub lag: Option<u32>,
    pub closed: rheon_core::Closed,
    pub component: crate::windows::StlComponent,
    pub max_state_bytes: Option<u64>,
}

/// What an `ewm_profile` takes beside the window operators' parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileParams {
    pub period: Span,
    pub offset: Option<Span>,
    /// The bucketed form's buckets, or 0 for the harmonic form.
    pub buckets: u32,
    /// The harmonic form's harmonics, or 0 for the bucketed form.
    pub harmonics: u32,
    pub stat: ProfileStat,
    pub weight: Option<Node>,
    pub max_state_bytes: Option<u64>,
}

/// How a cross operator that reads values in order reads them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Order {
    pub rank: RankMethod,
    pub descending: bool,
    pub interpolation: Interp,
}

/// The term an `ewm_moments` operator reads (task 554): each leg's power
/// and lag, in the order of its input's legs, the statistic and its form.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentsNode {
    /// Each leg's power, a whole number from 1.
    pub powers: Vec<u32>,
    /// Each leg's lag: 0 for the row itself, else clock units, or a
    /// duration on a temporal clock (rows on a row-count clock).
    pub lags: Vec<Span>,
    pub kind: rheon_core::MomentKind,
    /// The plug-in form (`true`) or the unbiased one; `None` for the
    /// statistic's default.
    pub bias: Option<bool>,
    /// How far before `t - lag` a lag's as-of row may sit; `None` for the
    /// lag itself.
    pub lag_tolerance: Option<Span>,
}

/// The words `kind` takes, the default first.
const MOMENT_KINDS: [(&str, rheon_core::MomentKind); 4] = [
    ("central", rheon_core::MomentKind::Central),
    ("raw", rheon_core::MomentKind::Raw),
    ("standardized", rheon_core::MomentKind::Standardized),
    ("cumulant", rheon_core::MomentKind::Cumulant),
];

impl MomentsNode {
    /// The term's order: the sum of its powers.
    pub fn order(&self) -> u32 {
        self.powers.iter().sum()
    }

    /// The form the term is read in: `bias` as given, else the statistic's
    /// default (a variance unbiased, every other order the plug-in).
    pub fn bias(&self) -> bool {
        self.bias
            .unwrap_or_else(|| rheon_core::MomentTerm::default_bias(self.kind, self.order()))
    }

    fn kind_name(&self) -> &'static str {
        MOMENT_KINDS
            .iter()
            .find(|(_, k)| *k == self.kind)
            .map_or("central", |(n, _)| n)
    }
}

impl OpNode {
    /// The trees beside the input an operator reads: a cross operator's
    /// weight and a reference read's predicate, and a profile's weight.
    pub fn trees(&self) -> impl Iterator<Item = &Node> {
        self.weight
            .iter()
            .chain(self.pick.iter())
            .map(|b| &**b)
            .chain(self.profile.iter().filter_map(|p| p.weight.as_ref()))
    }

    /// A row operator's node over `input`, its other parameters unset.
    fn row(kind: OpKind, input: Node) -> Self {
        OpNode {
            kind,
            input: Box::new(input),
            half_life: None,
            window_size: None,
            closed: Closed::Right,
            min_samples: 1,
            partial: None,
            bias: false,
            quantile: None,
            weight: None,
            max_state_bytes: None,
            order: Order::default(),
            neighbors: None,
            pick: None,
            moments: None,
            profile: None,
            since: None,
            tol: 0.0,
            event: None,
            step: None,
        }
    }

    /// Backward or forward; a step (`increment`, `tick_sign`) reads back one
    /// row and is backward.
    pub fn direction(&self) -> Direction {
        self.kind.direction()
    }

    /// The compact JSON of this node alone: what a hidden column is named by,
    /// so two formulas asking for one operator share it.
    pub fn key(&self) -> String {
        Node::Op(self.clone()).to_json().to_string()
    }

    fn params_json(&self) -> Value {
        if let Some(p) = &self.stl {
            return stl_json(self.kind, p);
        }
        let mut m = Map::new();
        if let Some(e) = &self.event {
            m.insert("event".into(), e.to_json());
        }
        if self.kind == OpKind::RunLength {
            m.insert("abs_tol".into(), json!(self.tol));
            return Value::Object(m);
        }
        let ext = self.kind.stat().is_some_and(Stat::is_extremum);
        if let Some(h) = self.half_life.as_ref().filter(|_| !ext) {
            m.insert(
                "half_life".into(),
                serde_json::to_value(h).expect("a span is JSON"),
            );
        }
        if let Some(w) = &self.window_size {
            m.insert(
                "window_size".into(),
                serde_json::to_value(w).expect("a span is JSON"),
            );
        }
        if let Some(st) = self.step {
            m.insert("stat".into(), json!(st.name()));
            if st.is_spread() {
                m.insert("bias".into(), json!(self.bias));
            }
        }
        if self.kind.is_row() {
            return Value::Object(m);
        }
        m.insert("closed".into(), json!(self.closed.name()));
        m.insert("min_samples".into(), json!(self.min_samples));
        if let Some(p) = self.partial {
            m.insert("partial".into(), json!(p.name()));
        }
        if self.kind.stat().is_some_and(Stat::is_var) {
            m.insert("bias".into(), json!(self.bias));
        }
        if let Some(q) = self.quantile {
            m.insert("quantile".into(), json!(q));
        }
        if self.kind.cross().is_some() {
            // A cross operator has no kernel: its stamp and its weight.
            m.remove("min_samples");
            if self.kind == OpKind::CrossStd {
                m.insert("bias".into(), json!(self.bias));
            }
            if let Some(w) = &self.weight {
                m.insert("weight".into(), w.to_json());
            }
            if let Some(n) = &self.neighbors {
                m.insert("neighbors".into(), json!(n));
            }
            if let Some(p) = &self.pick {
                m.insert("where".into(), p.to_json());
            }
            match self.kind.cross() {
                Some(CrossStat::Rank) => {
                    m.insert("method".into(), json!(self.order.rank.name()));
                    m.insert("descending".into(), json!(self.order.descending));
                }
                Some(CrossStat::Quantile) => {
                    m.insert(
                        "interpolation".into(),
                        json!(self.order.interpolation.name()),
                    );
                }
                _ => {}
            }
        }
        if let Some(t) = &self.moments {
            m.remove("min_samples");
            m.insert("powers".into(), json!(t.powers));
            m.insert(
                "lags".into(),
                Value::Array(
                    t.lags
                        .iter()
                        .map(|l| serde_json::to_value(l).expect("a span is JSON"))
                        .collect(),
                ),
            );
            m.insert("kind".into(), json!(t.kind_name()));
            m.insert("bias".into(), json!(t.bias()));
            if let Some(tol) = &t.lag_tolerance {
                m.insert(
                    "lag_tolerance".into(),
                    serde_json::to_value(tol).expect("a span is JSON"),
                );
            }
        }
        if let Some(p) = &self.profile {
            // A profile has no `partial`: its `min_samples` counts a
            // bucket's rows.
            m.insert(
                "period".into(),
                serde_json::to_value(&p.period).expect("a span is JSON"),
            );
            if let Some(o) = &p.offset {
                m.insert(
                    "offset".into(),
                    serde_json::to_value(o).expect("a span is JSON"),
                );
            }
            if p.harmonics > 0 {
                m.insert("harmonics".into(), json!(p.harmonics));
            } else {
                m.insert("buckets".into(), json!(p.buckets));
            }
            m.insert("stat".into(), json!(p.stat.name()));
            if p.harmonics == 0 && matches!(p.stat, ProfileStat::Var | ProfileStat::Std) {
                m.insert("bias".into(), json!(self.bias));
            }
            if let Some(w) = &p.weight {
                m.insert("weight".into(), w.to_json());
            }
            if let Some(b) = p.max_state_bytes {
                m.insert("max_state_bytes".into(), json!(b));
            }
        }
        if let Some(e) = &self.since {
            m.insert("since".into(), e.to_json());
        }
        if let Some(b) = self.max_state_bytes {
            m.insert("max_state_bytes".into(), json!(b));
        }
        Value::Object(m)
    }
}

/// One node of a formula.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Col(String),
    Lit(Literal),
    /// An element-wise node: its name and its arguments, in the order the
    /// list form has them. `cast` and `alias` carry their text as a `Lit`.
    Call(String, Vec<Node>),
    Op(OpNode),
    /// The root in an unknown of a function of the row, solved row by row
    /// (`formula/scalar.rs`): an element-wise node the library evaluates
    /// itself, counted as one of its own where a formula must hold one.
    Invert(Box<scalar::Invert>),
}

/// The element-wise nodes and their arities.
const CALLS: &[(&str, usize)] = &[
    ("+", 2),
    ("-", 2),
    ("*", 2),
    ("/", 2),
    ("**", 2),
    ("==", 2),
    ("!=", 2),
    ("<", 2),
    ("<=", 2),
    (">", 2),
    (">=", 2),
    ("&", 2),
    ("|", 2),
    ("neg", 1),
    ("abs", 1),
    ("exp", 1),
    ("sqrt", 1),
    ("is_null", 1),
    ("is_not_null", 1),
    ("log", 2),
    ("clip", 3),
    ("fill_null", 2),
    ("when", 3),
    ("cast", 2),
    ("alias", 2),
];

/// The window operators as a formula's tree writes them, the looking-ahead
/// forms after the rest: the refusal of an operator lists them, and a unit
/// test holds them to `OpKind::parse`.
const WINDOW_HEADS: &[&str] = &[
    "ewm_mean",
    "ewm_sum",
    "ewm_rate",
    "ewm_var",
    "ewm_std",
    "ewm_quantile",
    "ewm_moments",
    "rolling_max",
    "rolling_min",
    "rewm_mean",
    "rewm_sum",
    "rewm_rate",
    "rewm_quantile",
    "rrolling_max",
    "rrolling_min",
];

/// The window operators looking ahead, as a formula writes them: "a, b or
/// c". A formula target's refusal of a formula with none names them all; it
/// named three of six (review 2026-11-23, SPEC-11).
pub(crate) fn forward_heads() -> String {
    let ahead: Vec<&str> = WINDOW_HEADS
        .iter()
        .copied()
        .filter(|h| OpKind::parse(h).is_some_and(|k| k.direction() == Direction::Forward))
        .collect();
    let (last, rest) = ahead.split_last().expect("operators looking ahead");
    format!("{} or {last}", rest.join(", "))
}

/// What a formula reads around the window operators, in the words a reader
/// looks for, each with the nodes it stands for. The refusal names their
/// nodes, as a tree writes them; a unit test holds their nodes to `CALLS` with `col` and `lit`, and
/// `tests/test_windows.py` holds this list to Python's `ACCEPTED`, so the
/// message cannot leave out a node that is read (task 238).
const ACCEPTED: &[(&str, &[&str])] = &[
    ("columns", &["col"]),
    ("literals", &["lit"]),
    ("arithmetic", &["+", "-", "*", "/"]),
    ("negation", &["neg"]),
    ("comparisons", &["==", "!=", "<", "<=", ">", ">="]),
    ("& and |", &["&", "|"]),
    ("log", &["log"]),
    ("exp", &["exp"]),
    ("abs", &["abs"]),
    ("sqrt", &["sqrt"]),
    ("pow", &["**"]),
    ("clip", &["clip"]),
    ("fill_null", &["fill_null"]),
    ("is_null", &["is_null"]),
    ("is_not_null", &["is_not_null"]),
    ("when/then/otherwise", &["when"]),
    ("cast", &["cast"]),
    ("alias", &["alias"]),
];

/// The heads of `ACCEPTED` as a tree writes them, as one English list: a
/// refusal of a head names the spellings the reader can write.
fn accepted_heads() -> String {
    let heads: Vec<&str> = ACCEPTED
        .iter()
        .flat_map(|(_, h)| h.iter().copied())
        .collect();
    match heads.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => heads.concat(),
    }
}

/// The dtypes a `cast` may name, as Polars spells them. `Date` and `Time`
/// are how a date or time-of-day literal keeps its dtype: Python carries one
/// as `["cast", ["lit", n], "Date"]`, `n` the days since 1970 (for `Time`,
/// the nanoseconds since midnight), where it was read as the bare integer
/// (review 2026-10-05, YB1).
const DTYPES: &[(&str, DataType)] = &[
    ("Float64", DataType::Float64),
    ("Float32", DataType::Float32),
    ("Int64", DataType::Int64),
    ("Int32", DataType::Int32),
    ("Int16", DataType::Int16),
    ("Int8", DataType::Int8),
    ("UInt64", DataType::UInt64),
    ("UInt32", DataType::UInt32),
    ("UInt16", DataType::UInt16),
    ("UInt8", DataType::UInt8),
    ("Boolean", DataType::Boolean),
    ("String", DataType::String),
    ("Date", DataType::Date),
    ("Time", DataType::Time),
];

fn dtype_of(name: &str) -> Result<DataType, String> {
    DTYPES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, t)| t.clone())
        .ok_or_else(|| {
            format!(
                "formula: a cast to {name:?} is not one a formula can hold. Cast to one of {:?}.",
                DTYPES.iter().map(|(n, _)| *n).collect::<Vec<_>>()
            )
        })
}

impl Node {
    /// Read the list form. An unknown node is refused by name: what is not
    /// element-wise never gets as far as a kernel.
    ///
    /// # Errors
    ///
    /// A node that is not a list headed by a name, an unknown name, a wrong
    /// arity, a bad literal, a `cast` to a dtype not read, an operator with
    /// bad parameters, or a window operator inside an operator's input.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let Value::Array(items) = v else {
            return Err(format!(
                "formula: a node must be a list whose first item names its kind, such as \
                 [\"col\", \"x\"], but it is {}. Write the node in that form.",
                short(v)
            ));
        };
        let Some(Value::String(head)) = items.first() else {
            return Err(format!(
                "formula: a node must be a list whose first item names its kind, such as \
                 [\"col\", \"x\"], but it is {}. Write the node in that form.",
                short(v)
            ));
        };
        let args = &items[1..];
        let arity = |n: usize| -> Result<(), String> {
            if args.len() == n {
                Ok(())
            } else {
                // A cast takes a third, `"non_strict"` (read below).
                let or = match head.as_str() {
                    "cast" => ", or 3 with \"non_strict\"",
                    "lit" => ", or none for null",
                    _ => "",
                };
                Err(format!(
                    "formula: {head:?} takes {n} argument{}{or}, but it has {}. Give {head:?} \
                     that many.",
                    if n == 1 { "" } else { "s" },
                    args.len()
                ))
            }
        };
        match head.as_str() {
            "invert" => Ok(Node::Invert(Box::new(scalar::Invert::from_json(v)?))),
            "col" => {
                arity(1)?;
                match &args[0] {
                    Value::String(s) => Ok(Node::Col(s.clone())),
                    other => Err(format!(
                        "formula: \"col\" takes a column's name as a string, but it has {}. \
                         Give \"col\" a column name, such as [\"col\", \"x\"].",
                        short(other)
                    )),
                }
            }
            "lit" => {
                // `["lit"]` is null, the form TOML can carry; `["lit", null]`
                // is read too.
                if args.is_empty() {
                    return Ok(Node::Lit(Literal::Null));
                }
                arity(1)?;
                Ok(Node::Lit(Literal::from_json(&args[0])?))
            }
            "cast" | "alias" => {
                // A cast is strict unless a third element says `"non_strict"`
                // (review R1, B2: Python's default is strict, Rust's `cast`
                // is not, and a failed cast became a silent null).
                let non_strict = head == "cast" && args.len() == 3;
                if !non_strict {
                    arity(2)?;
                }
                let x = Node::from_json(&args[0])?;
                let Value::String(text) = &args[1] else {
                    return Err(format!(
                        "formula: {head:?} takes {} as its second argument, but it has {}. \
                         Give a string there.",
                        if head == "cast" {
                            "a dtype's name, such as \"Float64\","
                        } else {
                            "the output's name"
                        },
                        short(&args[1])
                    ));
                };
                if head == "cast" {
                    dtype_of(text)?;
                }
                let mut call = vec![x, Node::Lit(Literal::Str(text.clone()))];
                if non_strict {
                    match &args[2] {
                        Value::String(s) if s == "non_strict" => {
                            call.push(Node::Lit(Literal::Str(s.clone())));
                        }
                        other => {
                            return Err(format!(
                                "formula: \"cast\" takes \"non_strict\" as its third \
                                 argument, but it has {}. Give \"non_strict\" there, or leave \
                                 out the third argument for a strict cast.",
                                short(other)
                            ));
                        }
                    }
                }
                Ok(Node::Call(head.clone(), call))
            }
            name if CALLS.iter().any(|(n, _)| *n == name) => {
                let n = CALLS
                    .iter()
                    .find(|(c, _)| *c == name)
                    .map_or(0, |(_, a)| *a);
                arity(n)?;
                let nodes = args
                    .iter()
                    .map(|a| {
                        if name == "clip" && a.is_null() {
                            Ok(Node::Lit(Literal::Null))
                        } else {
                            Node::from_json(a)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Node::Call(name.to_string(), nodes))
            }
            "increment" | "tick_sign" => {
                arity(1)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator(head.as_str())?;
                Ok(Node::Op(OpNode::row(
                    OpKind::parse(head.as_str()).expect("a step"),
                    input,
                )))
            }
            "run_length" => {
                // `["run_length", x]`, or with its tolerance after it.
                if args.len() != 2 {
                    arity(1)?;
                }
                let input = Node::from_json(&args[0])?;
                input.no_operator("run_length")?;
                let mut op = OpNode::row(OpKind::RunLength, input);
                if let Some(p) = args.get(1) {
                    let Value::Object(params) = p else {
                        return Err(format!(
                            "formula: \"run_length\" takes its parameters as an object, but it \
                             has {}. Give them as an object, such as {{\"abs_tol\": 0.5}}.",
                            short(p)
                        ));
                    };
                    for (k, v) in params {
                        if k != "abs_tol" {
                            return Err(format!(
                                "run_length: {k:?} is not one of its parameters, which is abs_tol. \
                                 Remove {k:?}, or correct its name."
                            ));
                        }
                        op.tol = v
                            .as_f64()
                            .filter(|t| t.is_finite() && *t >= 0.0)
                            .ok_or_else(|| {
                                format!(
                                    "run_length: abs_tol must be a finite number of at least 0, but \
                                     it is {}. Set abs_tol to 0 for values exactly equal, or to the \
                                     largest difference two adjacent rows of one run may have.",
                                    short(v)
                                )
                            })?;
                    }
                }
                Ok(Node::Op(op))
            }
            "clock_lag" | "clock_drift" => {
                let kind = OpKind::parse(head).expect("named above");
                let name = head.as_str();
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_operator(name)?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "formula: {name:?} takes its parameters as an object, but it has {}. \
                         Give them as an object, such as {{\"event\": [\"col\", \"e\"], \
                         \"half_life\": 10}}.",
                        short(&args[1])
                    ));
                };
                let mut op = OpNode::row(kind, input);
                for (k, v) in params {
                    match k.as_str() {
                        "event" => {
                            let e = Node::from_json(v)?;
                            e.no_operator(name)?;
                            op.event = Some(Box::new(e));
                        }
                        "half_life" => op.half_life = Some(span_of(name, k, v)?),
                        other => {
                            return Err(format!(
                                "{name}: {other:?} is not one of its parameters, which are event \
                                 and half_life. Remove {other:?}, or correct its name."
                            ));
                        }
                    }
                }
                if op.event.is_none() {
                    return Err(format!(
                        "{name}: event is not set, but the lag between two clocks needs the \
                         time each row's event happened. Set event to a column or an \
                         expression of the row."
                    ));
                }
                match &op.half_life {
                    None => {
                        return Err(format!(
                            "{name}: half_life is not set, but the lag's decayed sums need it. \
                             Set half_life to a positive number of clock units or a duration, \
                             or to inf for no decay."
                        ));
                    }
                    Some(h) if h.value().is_nan() || h.value() <= 0.0 => {
                        return Err(format!(
                            "{name}: half_life must be above 0, but it is {h}. Set half_life to \
                             a positive number of clock units or a duration, or to inf for no \
                             decay."
                        ));
                    }
                    Some(_) => {}
                }
                Ok(Node::Op(op))
            }
            "clock_step" => {
                // `["clock_step", time, {"half_life": h, "stat": s}]`, and a
                // spread's `bias` (task 564).
                let name = "clock_step";
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_operator(name)?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "formula: \"clock_step\" takes its parameters as an object, but it has \
                         {}. Give them as an object, such as {{\"half_life\": 10, \"stat\": \
                         \"mean\"}}.",
                        short(&args[1])
                    ));
                };
                let mut op = OpNode::row(OpKind::ClockStep, input);
                let mut bias = None;
                for (k, v) in params {
                    match k.as_str() {
                        "half_life" => op.half_life = Some(span_of(name, k, v)?),
                        "stat" => {
                            let st = v.as_str().and_then(rheon_core::StepStat::parse);
                            op.step = Some(st.ok_or_else(|| {
                                format!(
                                    "{name}: stat must be \"mean\", \"var\", \"std\", \
                                     \"zeros\" or \"backs\", but it is {}. Set stat to one \
                                     of those words.",
                                    short(v)
                                )
                            })?);
                        }
                        "bias" => {
                            bias = Some(v.as_bool().ok_or_else(|| {
                                format!(
                                    "{name}: bias must be true or false, but it is {}. Set \
                                     bias to true or false.",
                                    short(v)
                                )
                            })?);
                        }
                        other => {
                            return Err(format!(
                                "{name}: {other:?} is not one of its parameters, which are \
                                 half_life, stat and bias. Remove {other:?}, or correct its \
                                 name."
                            ));
                        }
                    }
                }
                let st = op.step.get_or_insert(rheon_core::StepStat::Mean);
                match bias {
                    Some(b) if st.is_spread() => op.bias = b,
                    Some(_) => {
                        return Err(format!(
                            "{name}: bias is set, but only stat \"var\" or \"std\" takes it. \
                             Remove bias, or set stat to \"var\" or \"std\"."
                        ));
                    }
                    None => {}
                }
                match &op.half_life {
                    None => {
                        return Err(format!(
                            "{name}: half_life is not set, but the steps' decayed statistics \
                             need it. Set half_life to a positive number of clock units or a \
                             duration, or to inf for no decay."
                        ));
                    }
                    Some(h) if h.value().is_nan() || h.value() <= 0.0 => {
                        return Err(format!(
                            "{name}: half_life must be above 0, but it is {h}. Set half_life to \
                             a positive number of clock units or a duration, or to inf for no \
                             decay."
                        ));
                    }
                    Some(_) => {}
                }
                Ok(Node::Op(op))
            }
            name if OpKind::parse(name).is_some_and(|k| k.cross().is_some()) => {
                let kind = OpKind::parse(name).expect("checked");
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator(name)?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "formula: {name:?} takes its parameters as an object, but it has {}. \
                         Give them as an object, such as {{\"closed\": \"right\"}}.",
                        short(&args[1])
                    ));
                };
                let mut op = OpNode {
                    kind,
                    input: Box::new(input),
                    half_life: None,
                    window_size: None,
                    closed: Closed::Right,
                    min_samples: 2,
                    partial: None,
                    bias: false,
                    quantile: None,
                    weight: None,
                    max_state_bytes: None,
                    order: Order::default(),
                    neighbors: None,
                    pick: None,
                    moments: None,
                    profile: None,
                    since: None,
                    tol: 0.0,
                    event: None,
                    step: None,
                    stl: None,
                };
                let std = kind == OpKind::CrossStd;
                let stat = kind.cross().expect("checked");
                let mut interpolation = None;
                for (k, v) in params {
                    match k.as_str() {
                        "closed" => {
                            op.closed = match v.as_str() {
                                Some("right") => Closed::Right,
                                Some("left") => Closed::Left,
                                _ => {
                                    return Err(format!(
                                        "{name}: closed must be \"right\" (the row's own stamp) \
                                         or \"left\" (the stamp before it), but it is {}. Set \
                                         closed to one of those words.",
                                        short(v)
                                    ));
                                }
                            };
                        }
                        "bias" if std => {
                            op.bias = v.as_bool().ok_or_else(|| {
                                format!(
                                    "{name}: bias must be true or false, but it is {}. Set bias \
                                     to true or false.",
                                    short(v)
                                )
                            })?;
                        }
                        "where" if stat == CrossStat::Ref => {
                            let w = Node::from_json(v)?;
                            w.no_window_operator(name)?;
                            op.pick = Some(Box::new(w));
                        }
                        "neighbors" if stat != CrossStat::Ref => {
                            let col = v.as_str().filter(|c| !c.is_empty()).ok_or_else(|| {
                                format!(
                                    "{name}: neighbors must name a list column, but it is {}. \
                                     Set neighbors to the name of a column whose rows list \
                                     the group keys each row reads.",
                                    short(v)
                                )
                            })?;
                            op.neighbors = Some(col.to_string());
                        }
                        "weight" if stat == CrossStat::Ref => {
                            return Err(format!(
                                "{name}: a reference read takes no weight: it reads one row's \
                                 value. Remove weight."
                            ));
                        }
                        "weight" if stat == CrossStat::Rank => {
                            return Err(format!(
                                "{name}: a rank takes no weight: a row's rank counts the rows \
                                 ranked before it, and a weight has no place in that count. \
                                 Remove weight, or use cross_quantile with weight= for the \
                                 weighted share of the cross-section below a level."
                            ));
                        }
                        "weight" => {
                            let w = Node::from_json(v)?;
                            w.no_window_operator(name)?;
                            op.weight = Some(Box::new(w));
                        }
                        "method" if stat == CrossStat::Rank => {
                            op.order.rank =
                                v.as_str().and_then(RankMethod::parse).ok_or_else(|| {
                                    format!(
                                        "{name}: method must be \"average\", \"min\", \
                                         \"max\", \"dense\" or \"ordinal\", but it is {}. \
                                         Set method to one of those words.",
                                        short(v)
                                    )
                                })?;
                        }
                        "descending" if stat == CrossStat::Rank => {
                            op.order.descending = v.as_bool().ok_or_else(|| {
                                format!(
                                    "{name}: descending must be true or false, but it is {}. \
                                     Set descending to true or false.",
                                    short(v)
                                )
                            })?;
                        }
                        "quantile" if stat == CrossStat::Quantile => {
                            op.quantile =
                                Some(v.as_f64().filter(|q| (0.0..=1.0).contains(q)).ok_or_else(
                                    || {
                                        format!(
                                            "{name}: quantile must be a number from 0 to 1, but \
                                             it is {}. Set quantile within that range, such as \
                                             0.5 for the median.",
                                            short(v)
                                        )
                                    },
                                )?);
                        }
                        "interpolation" if stat == CrossStat::Quantile => {
                            interpolation =
                                Some(v.as_str().and_then(Interp::parse).ok_or_else(|| {
                                    format!(
                                        "{name}: interpolation must be \"nearest\", \
                                         \"higher\", \"lower\", \"midpoint\", \"linear\" \
                                         or \"equiprobable\", but it is {}. Set interpolation \
                                         to one of those words.",
                                        short(v)
                                    )
                                })?);
                        }
                        "max_state_bytes" => {
                            let ceiling = crate::windows::CROSS_STATE_BYTES_CEILING;
                            op.max_state_bytes =
                                Some(v.as_u64().filter(|&b| b >= 1 && b <= ceiling).ok_or_else(
                                    || {
                                        format!(
                                            "{name}: max_state_bytes must be a whole number from 1 \
                                         to {ceiling} (64 MiB), but it is {}. Set it within \
                                         that range, or leave it out for 262144 (256 KB).",
                                            short(v)
                                        )
                                    },
                                )?);
                        }
                        other => {
                            return Err(format!(
                                "{name}: {other:?} is not one of its parameters, which are \
                                 {}. Remove {other:?}, or correct its name.",
                                match stat {
                                    CrossStat::Std =>
                                        "closed, weight, neighbors, max_state_bytes and bias",
                                    CrossStat::Rank =>
                                        "closed, method, descending, neighbors and \
                                         max_state_bytes",
                                    CrossStat::Quantile =>
                                        "quantile, closed, interpolation, weight, neighbors \
                                         and max_state_bytes",
                                    CrossStat::Mean =>
                                        "closed, weight, neighbors and max_state_bytes",
                                    CrossStat::Ref => "where, closed and max_state_bytes",
                                }
                            ));
                        }
                    }
                }
                if stat == CrossStat::Ref && op.pick.is_none() {
                    return Err(format!(
                        "{name}: where is required: the predicate that picks the stamp's \
                         reference row. Set where, such as pl.col(\"symbol\") == \"SPY\"."
                    ));
                }
                if stat == CrossStat::Quantile {
                    if op.quantile.is_none() {
                        return Err(format!(
                            "{name}: quantile is required: the level to read, from 0 to 1. \
                             Set quantile, such as 0.5 for the median."
                        ));
                    }
                    // A weighted cross-section has one reading between two
                    // values: the first whose cumulative weight share reaches
                    // the level.
                    op.order.interpolation = match (interpolation, op.weight.is_some()) {
                        (None, false) => Interp::Nearest,
                        (None, true) | (Some(Interp::Equiprobable), _) => Interp::Equiprobable,
                        (Some(i), false) => i,
                        (Some(i), true) => {
                            return Err(format!(
                                "{name}: interpolation {:?} has no weighted form: with weight= \
                                 the quantile is the first value whose cumulative weight share \
                                 reaches the level, \"equiprobable\". Set interpolation to \
                                 \"equiprobable\" or leave it out, or remove weight.",
                                i.name()
                            ));
                        }
                    };
                }
                Ok(Node::Op(op))
            }
            "ewm_moments" => moments_from_json(args),
            "stl" | "mstl" => {
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator(head.as_str())?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "formula: {head:?} takes its parameters as an object, but it has {}. \
                         Give them as an object, such as {{\"period\": 24}}.",
                        short(&args[1])
                    ));
                };
                stl_of(head.as_str(), input, params).map(Node::Op)
            }
            "ewm_profile" => {
                let name = "ewm_profile";
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator(name)?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "formula: {name:?} takes its parameters as an object, but it has {}. \
                         Give them as an object, such as {{\"period\": \"1d\"}}.",
                        short(&args[1])
                    ));
                };
                profile_of(input, params).map(Node::Op)
            }
            name if OpKind::parse(name).is_some() => {
                let kind = OpKind::parse(name).expect("checked");
                arity(2)?;
                let input = Node::from_json(&args[0])?;
                input.no_window_operator(head.as_str())?;
                let Value::Object(params) = &args[1] else {
                    return Err(format!(
                        "formula: {name:?} takes its parameters as an object, but it has {}. \
                         Give them as an object, such as {{\"half_life\": 10}}.",
                        short(&args[1])
                    ));
                };
                // A sum's default is Polars' own, 0: a window with no value
                // sums to 0.0. Every other statistic needs one value.
                let sum = kind.stat() == Some(Stat::Sum);
                let mut op = OpNode {
                    kind,
                    input: Box::new(input),
                    half_life: None,
                    window_size: None,
                    closed: Closed::Right,
                    min_samples: u32::from(!sum),
                    partial: None,
                    bias: false,
                    quantile: None,
                    weight: None,
                    max_state_bytes: None,
                    order: Order::default(),
                    neighbors: None,
                    pick: None,
                    moments: None,
                    profile: None,
                    since: None,
                    tol: 0.0,
                    event: None,
                    step: None,
                    stl: None,
                };
                let var = kind.stat().is_some_and(Stat::is_var);
                let quantile = kind.stat() == Some(Stat::Quantile);
                let ext = kind.stat().is_some_and(Stat::is_extremum);
                if ext {
                    // An extremum weighs no row: its kernel is the even one.
                    op.half_life = Some(Span::Units(f64::INFINITY));
                }
                for (k, v) in params {
                    match k.as_str() {
                        "half_life" if !ext => {
                            op.half_life = Some(span_of(name, k, v)?);
                        }
                        "since" if ext => {
                            let e = Node::from_json(v)?;
                            if !e.operators().is_empty() {
                                return Err(format!(
                                    "{name}: since holds a window operator or an increment, but \
                                     it may read only an expression of the row itself, such as \
                                     pl.col(\"new_period\"). Add the operator's result as a \
                                     column of the input before the run, and read that column."
                                ));
                            }
                            op.since = Some(Box::new(e));
                        }
                        "max_state_bytes" if ext => {
                            op.max_state_bytes =
                                Some(v.as_u64().filter(|&b| b > 0).ok_or_else(|| {
                                    format!(
                                        "{name}: max_state_bytes must be a whole number of \
                                         bytes above 0, but it is {}. Set max_state_bytes to the \
                                         bytes a group's window may hold.",
                                        short(v)
                                    )
                                })?);
                        }
                        "window_size" => {
                            op.window_size = Some(span_of(name, k, v)?);
                        }
                        "closed" => {
                            op.closed =
                                Closed::parse(v.as_str().unwrap_or("")).ok_or_else(|| {
                                    format!(
                                        "{name}: closed must be \"right\", \"left\", \"both\" or \
                                         \"none\", but it is {}. Set closed to one of those words.",
                                        short(v)
                                    )
                                })?;
                        }
                        "min_samples" => {
                            // The ceiling is named: a count past a `u32` was
                            // refused as one below 1 (review round 4, PD4).
                            op.min_samples = v
                                .as_u64()
                                .and_then(|n| u32::try_from(n).ok())
                                .filter(|&n| n >= u32::from(!sum))
                                .ok_or_else(|| {
                                    format!(
                                        "{name}: min_samples must be a whole number from {} to \
                                         {}, but it is {}. Set min_samples within that range.",
                                        u32::from(!sum),
                                        u32::MAX,
                                        short(v)
                                    )
                                })?;
                        }
                        "partial" => {
                            op.partial = Some(
                                Partial::parse(v.as_str().unwrap_or("")).ok_or_else(|| {
                                    format!(
                                        "{name}: partial must be \"keep\", \"null\" or \"drop\", \
                                         but it is {}. Set partial to one of those words.",
                                        short(v)
                                    )
                                })?,
                            );
                        }
                        "bias" if var => {
                            op.bias = v.as_bool().ok_or_else(|| {
                                format!(
                                    "{name}: bias must be true or false, but it is {}. Set bias \
                                     to true or false.",
                                    short(v)
                                )
                            })?;
                        }
                        "quantile" if quantile => {
                            op.quantile = Some(
                                v.as_f64().filter(|q| *q > 0.0 && *q < 1.0).ok_or_else(|| {
                                    format!(
                                        "{name}: quantile must be a number above 0 and below \
                                         1, but it is {}. Set quantile within that range, such as \
                                         0.5 for the median.",
                                        short(v)
                                    )
                                })?,
                            );
                        }
                        other if ext => {
                            return Err(format!(
                                "{name}: {other:?} is not one of its parameters, which are \
                                 window_size, closed, min_samples, partial, since and \
                                 max_state_bytes. Remove {other:?}, or correct its name.{}",
                                if other == "half_life" {
                                    " An extremum weighs no row, so it has no half_life."
                                } else {
                                    ""
                                }
                            ));
                        }
                        other => {
                            return Err(format!(
                                "{name}: {other:?} is not one of its parameters, which are \
                                 half_life, window_size, closed, min_samples{}. Remove \
                                 {other:?}, or correct its name.",
                                if var {
                                    ", partial and bias"
                                } else if quantile {
                                    ", partial and quantile"
                                } else {
                                    " and partial"
                                }
                            ));
                        }
                    }
                }
                if op.since.is_some() && kind.direction() == Direction::Forward {
                    return Err(format!("{name}: {}.", crate::windows::SINCE_AHEAD));
                }
                if quantile && op.quantile.is_none() {
                    return Err(format!(
                        "{name}: quantile is not set, but a quantile operator needs it. Set \
                         quantile to a number above 0 and below 1, such as 0.5 for the median."
                    ));
                }
                let Some(h) = &op.half_life else {
                    return Err(format!(
                        "{name}: half_life is not set, but every window operator needs it. Set \
                         half_life to a positive number of clock units or a duration, or to inf \
                         for no decay."
                    ));
                };
                if h.value().is_nan() || h.value() <= 0.0 {
                    return Err(format!(
                        "{name}: half_life must be above 0, but it is {h}. Set half_life to a \
                         positive number of clock units or a duration, or to inf for no decay."
                    ));
                }
                match &op.window_size {
                    Some(w) if !(w.value().is_finite() && w.value() > 0.0) => {
                        return Err(format!(
                            "{name}: window_size must be finite and above 0, but it is {w}. \
                             Set window_size to a finite positive number of clock units or a \
                             duration."
                        ));
                    }
                    None if kind.direction() == Direction::Forward => {
                        return Err(format!(
                            "{name}: window_size is not set, but an operator that reads the \
                             rows after each row needs one, since those rows have no end without \
                             it. Set window_size to a finite positive number of clock units or a \
                             duration."
                        ));
                    }
                    None if h.value().is_infinite()
                        && kind.stat().is_some_and(Stat::weighs_held) =>
                    {
                        return Err(format!(
                            "{name}: half_life is inf and window_size is not set, so a value \
                             held from before the first row would have no finite weight. Set \
                             window_size, or give half_life a finite value."
                        ));
                    }
                    _ => {}
                }
                Ok(Node::Op(op))
            }
            other => Err(format!(
                "formula: {other:?} is not one of the operators a formula can hold. It can hold \
                 increment (the step from a column's previous value), tick_sign (the sign of \
                 that step, the last nonzero one held over a step of 0), run_length (the rows a \
                 value has held), clock_lag and clock_drift (the lag between two clocks and its \
                 slope), clock_step (a decayed statistic of a time's steps), the window \
                 operators {}, \
                 and these operators within one row: {}. If the operator is misspelled, correct it; \
                 otherwise, write the formula with the operators listed.",
                {
                    let (last, rest) = WINDOW_HEADS.split_last().expect("fifteen heads");
                    format!("{} and {last}", rest.join(", "))
                },
                accepted_heads()
            )),
        }
    }

    /// Read the list form from its text.
    ///
    /// # Errors
    ///
    /// As [`Node::from_json`], and text that is not JSON.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| {
            format!(
                "formula: the text is not JSON ({e}). Write the formula as a JSON list, such as \
                 [\"col\", \"x\"]."
            )
        })?;
        Node::from_json(&v)
    }

    /// The list form.
    pub fn to_json(&self) -> Value {
        match self {
            Node::Col(c) => json!(["col", c]),
            // A null literal is `["lit"]`: TOML has no null (review R2, P3).
            Node::Lit(Literal::Null) => json!(["lit"]),
            Node::Lit(l) => json!(["lit", l.to_json()]),
            Node::Call(name, args) => {
                let mut items = vec![json!(name)];
                match name.as_str() {
                    "cast" | "alias" => {
                        items.push(args[0].to_json());
                        if let Node::Lit(Literal::Str(s)) = &args[1] {
                            items.push(json!(s));
                        }
                        // A non-strict cast carries its third argument
                        // (review R2, F2: it was dropped, and a saved spec
                        // ran a strict cast).
                        if let Some(Node::Lit(Literal::Str(s))) = args.get(2) {
                            items.push(json!(s));
                        }
                    }
                    // A bound left out is the null literal, `["lit"]`, the
                    // form TOML can carry, as Python writes it; a bare
                    // `null` is still read (task 159, F5).
                    "clip" => items.extend(args.iter().map(Node::to_json)),
                    _ => items.extend(args.iter().map(Node::to_json)),
                }
                Value::Array(items)
            }
            Node::Op(op) => {
                let mut items = vec![json!(op.kind.name()), op.input.to_json()];
                // An increment or a tick sign has no parameters, and a run length under the
                // default tolerance writes none (task 564): its key, and so
                // its hidden column, is the one form of the one operator.
                let bare = op.kind.is_step() || (op.kind == OpKind::RunLength && op.tol == 0.0);
                if !bare {
                    items.push(op.params_json());
                }
                Value::Array(items)
            }
            Node::Invert(i) => i.to_json(),
        }
    }

    /// Every operator in the tree, outermost first, in reading order.
    pub fn operators(&self) -> Vec<&OpNode> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect<'a>(&'a self, out: &mut Vec<&'a OpNode>) {
        match self {
            Node::Col(_) | Node::Lit(_) | Node::Invert(_) => {}
            Node::Call(_, args) => args.iter().for_each(|a| a.collect(out)),
            Node::Op(op) => {
                out.push(op);
                op.input.collect(out);
                for w in op.trees() {
                    w.collect(out);
                }
                if let Some(e) = &op.event {
                    e.collect(out);
                }
            }
        }
    }

    /// The columns of the input the tree reads, `increment` inputs included.
    pub fn columns(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_columns(&mut out);
        out
    }

    fn collect_columns(&self, out: &mut Vec<String>) {
        match self {
            Node::Col(c) => {
                if !out.contains(c) {
                    out.push(c.clone());
                }
            }
            Node::Lit(_) => {}
            Node::Invert(i) => {
                for c in i.columns() {
                    if !out.contains(c) {
                        out.push(c.clone());
                    }
                }
            }
            Node::Call(_, args) => args.iter().for_each(|a| a.collect_columns(out)),
            Node::Op(op) => {
                op.input.collect_columns(out);
                for w in op.trees() {
                    w.collect_columns(out);
                }
                if let Some(n) = &op.neighbors
                    && !out.contains(n)
                {
                    out.push(n.clone());
                }
                if let Some(e) = &op.since {
                    e.collect_columns(out);
                }
                if let Some(e) = &op.event {
                    e.collect_columns(out);
                }
            }
        }
    }

    /// Whether the tree holds a call of `name` anywhere (an operator's
    /// input included).
    pub fn contains_call(&self, name: &str) -> bool {
        match self {
            Node::Col(_) | Node::Lit(_) => false,
            Node::Invert(_) => name == "invert",
            Node::Call(n, args) => n == name || args.iter().any(|a| a.contains_call(name)),
            Node::Op(op) => {
                op.input.contains_call(name)
                    || op.trees().any(|w| w.contains_call(name))
                    || op.since.as_ref().is_some_and(|e| e.contains_call(name))
                    || op.event.as_ref().is_some_and(|e| e.contains_call(name))
            }
        }
    }

    /// [`Self::columns`], borrowed from the tree.
    pub fn columns_ref(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_columns_ref(&mut out);
        out
    }

    fn collect_columns_ref<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Node::Col(c) => {
                if !out.contains(&c.as_str()) {
                    out.push(c.as_str());
                }
            }
            Node::Lit(_) => {}
            Node::Invert(i) => {
                for c in i.columns() {
                    if !out.contains(&c.as_str()) {
                        out.push(c.as_str());
                    }
                }
            }
            Node::Call(_, args) => args.iter().for_each(|a| a.collect_columns_ref(out)),
            Node::Op(op) => {
                op.input.collect_columns_ref(out);
                for w in op.trees() {
                    w.collect_columns_ref(out);
                }
                if let Some(n) = &op.neighbors
                    && !out.contains(&n.as_str())
                {
                    out.push(n.as_str());
                }
                if let Some(e) = &op.since {
                    e.collect_columns_ref(out);
                }
                if let Some(e) = &op.event {
                    e.collect_columns_ref(out);
                }
            }
        }
    }

    /// Whether the tree holds an operator an operator's input may not: any
    /// but a step (`increment`, `tick_sign`). A window operator, and the row
    /// operators of task 564, whose values no operator reads in the run that
    /// makes them.
    pub fn has_window_operator(&self) -> bool {
        self.operators().iter().any(|op| !op.kind.is_step())
    }

    /// Refuse an input that holds any operator at all: a row operator's
    /// (task 564) reads the row's own columns, as an increment's does.
    fn no_operator(&self, who: &str) -> Result<(), String> {
        if let Some(inner) = self.operators().first() {
            return Err(format!(
                "{who}: its input holds the operator {}, but it may read only an expression (a \
                 column computation, such as pl.col(\"x\") + 1) of the row's own columns. Add \
                 the inner operator's result as a column of the input first, such as with one \
                 rp.stream.with_windows call, and read that column in a second.",
                inner.kind.name()
            ));
        }
        Ok(())
    }

    /// Whether the tree holds a forward operator.
    pub fn has_forward_operator(&self) -> bool {
        self.operators()
            .iter()
            .any(|op| op.direction() == Direction::Forward)
    }

    fn no_window_operator(&self, who: &str) -> Result<(), String> {
        if self.has_window_operator() {
            let such_as = if who == "increment" || who == "tick_sign" {
                "a column"
            } else {
                "a column or an increment of one"
            };
            return Err(format!(
                "{who}: its input holds a window operator, but it may read only an expression \
                 (a column computation, such as pl.col(\"x\") + 1) of the row itself, such as \
                 {such_as}. Add the inner operator's result as a column of the input before the \
                 run, such as with rp.stream.with_windows in Python, and read that column."
            ));
        }
        Ok(())
    }

    /// The Polars expression, with every operator resolved by `resolve` --
    /// to the hidden column that carries it, in a run.
    pub fn to_expr(&self, resolve: &dyn Fn(&OpNode) -> Expr) -> Expr {
        match self {
            Node::Col(c) => col(c.as_str()),
            Node::Lit(l) => l.to_expr(),
            Node::Op(op) => resolve(op),
            Node::Invert(i) => i.to_expr(),
            Node::Call(name, args) => {
                let e = |i: usize| args[i].to_expr(resolve);
                match name.as_str() {
                    "+" => e(0) + e(1),
                    "-" => e(0) - e(1),
                    "*" => e(0) * e(1),
                    // Python's `/`: a true division, whatever the dtypes.
                    "/" => binary_expr(e(0), Operator::TrueDivide, e(1)),
                    "**" => e(0).pow(e(1)),
                    "==" => e(0).eq(e(1)),
                    "!=" => e(0).neq(e(1)),
                    "<" => e(0).lt(e(1)),
                    "<=" => e(0).lt_eq(e(1)),
                    ">" => e(0).gt(e(1)),
                    ">=" => e(0).gt_eq(e(1)),
                    "&" => e(0).and(e(1)),
                    "|" => e(0).or(e(1)),
                    "neg" => -e(0),
                    "abs" => e(0).abs(),
                    "exp" => e(0).exp(),
                    "sqrt" => e(0).sqrt(),
                    "is_null" => e(0).is_null(),
                    "is_not_null" => e(0).is_not_null(),
                    "log" => e(0).log(e(1)),
                    "clip" => match (&args[1], &args[2]) {
                        (Node::Lit(Literal::Null), Node::Lit(Literal::Null)) => e(0),
                        (Node::Lit(Literal::Null), _) => e(0).clip_max(e(2)),
                        (_, Node::Lit(Literal::Null)) => e(0).clip_min(e(1)),
                        _ => e(0).clip(e(1), e(2)),
                    },
                    "fill_null" => e(0).fill_null(e(1)),
                    "when" => when(e(0)).then(e(1)).otherwise(e(2)),
                    "cast" => match &args[1] {
                        Node::Lit(Literal::Str(t)) => {
                            let dtype = dtype_of(t).expect("checked when read");
                            if args.len() == 3 {
                                e(0).cast(dtype)
                            } else {
                                e(0).strict_cast(dtype)
                            }
                        }
                        _ => unreachable!("cast carries its dtype as text"),
                    },
                    "alias" => match &args[1] {
                        Node::Lit(Literal::Str(n)) => e(0).alias(n.as_str()),
                        _ => unreachable!("alias carries its name as text"),
                    },
                    other => unreachable!("{other} is not a node kind this reads"),
                }
            }
        }
    }

    /// The operator this node is, where it is nothing else but, under any
    /// `alias`es at the root: a formula that only names a window operator's
    /// values.
    pub fn bare_operator(&self) -> Option<&OpNode> {
        match self {
            Node::Op(op) => Some(op),
            Node::Call(name, args) if name == "alias" => args[0].bare_operator(),
            _ => None,
        }
    }

    /// The name an `alias` at the root gives, if any.
    pub fn alias(&self) -> Option<&str> {
        match self {
            Node::Call(name, args) if name == "alias" => match &args[1] {
                Node::Lit(Literal::Str(n)) => Some(n.as_str()),
                _ => None,
            },
            _ => None,
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json())
    }
}

impl Serialize for Node {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.to_json().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        Node::from_json(&v).map_err(serde::de::Error::custom)
    }
}

/// A named formula: one output column of a `with_windows` call, or a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Formula {
    pub name: String,
    pub tree: Node,
}

/// `["ewm_moments", ["legs", leg, ...], {params}]`: a term of moments of
/// the legs, each an element-wise formula of the row, with each leg's power
/// and lag among the parameters.
fn moments_from_json(args: &[Value]) -> Result<Node, String> {
    const NAME: &str = "ewm_moments";
    if args.len() != 2 {
        return Err(format!(
            "formula: \"ewm_moments\" takes 2 arguments, its legs and its parameters, but it \
             has {}. Give it [\"legs\", ...] and an object of parameters.",
            args.len()
        ));
    }
    let legs = match &args[0] {
        Value::Array(items)
            if items.len() > 1 && items.first() == Some(&Value::String("legs".into())) =>
        {
            items[1..]
                .iter()
                .map(|v| {
                    let n = Node::from_json(v)?;
                    n.no_window_operator(NAME)?;
                    Ok(n)
                })
                .collect::<Result<Vec<_>, String>>()?
        }
        other => {
            return Err(format!(
                "{NAME}: its first argument is its legs, a list headed by \"legs\" with a \
                 formula a leg, such as [\"legs\", [\"col\", \"x\"]], but it is {}. \
                 Give the legs in that form.",
                short(other)
            ));
        }
    };
    let Value::Object(params) = &args[1] else {
        return Err(format!(
            "{NAME}: it takes its parameters as an object, but it has {}. Give them as an \
             object, such as {{\"half_life\": 10, \"powers\": [3]}}.",
            short(&args[1])
        ));
    };
    let mut op = OpNode {
        kind: OpKind::EwmMoments,
        input: Box::new(Node::Call("legs".into(), legs)),
        half_life: None,
        window_size: None,
        closed: Closed::Right,
        min_samples: 1,
        partial: None,
        bias: false,
        quantile: None,
        weight: None,
        max_state_bytes: None,
        order: Order::default(),
        neighbors: None,
        pick: None,
        moments: None,
        profile: None,
        since: None,
        tol: 0.0,
        event: None,
        step: None,
        stl: None,
    };
    let mut powers: Option<Vec<u32>> = None;
    let mut lags: Option<Vec<Span>> = None;
    let mut kind = rheon_core::MomentKind::Central;
    let mut bias = None;
    let mut lag_tolerance = None;
    for (k, v) in params {
        match k.as_str() {
            "half_life" => op.half_life = Some(span_of(NAME, k, v)?),
            "window_size" => op.window_size = Some(span_of(NAME, k, v)?),
            "closed" => {
                if v.as_str() != Some("right") {
                    return Err(format!(
                        "{NAME}: closed must be \"right\", but it is {}. The moments of a \
                         window are read once its stamp's rows are all in. Set closed to \
                         \"right\", or leave it out.",
                        short(v)
                    ));
                }
            }
            "partial" => {
                op.partial = Some(Partial::parse(v.as_str().unwrap_or("")).ok_or_else(|| {
                    format!(
                        "{NAME}: partial must be \"keep\", \"null\" or \"drop\", but it is \
                         {}. Set partial to one of those words.",
                        short(v)
                    )
                })?);
            }
            "powers" => {
                let bad = || {
                    format!(
                        "{NAME}: powers must be a list of whole numbers from 1, one a leg, but \
                         it is {}. Give each leg's power, such as [3].",
                        short(v)
                    )
                };
                let Value::Array(items) = v else {
                    return Err(bad());
                };
                powers = Some(
                    items
                        .iter()
                        .map(|p| {
                            p.as_u64()
                                .and_then(|p| u32::try_from(p).ok())
                                .filter(|&p| p >= 1)
                        })
                        .collect::<Option<Vec<u32>>>()
                        .ok_or_else(bad)?,
                );
            }
            "lags" => {
                let Value::Array(items) = v else {
                    return Err(format!(
                        "{NAME}: lags must be a list of spans of the clock, one a leg, 0 for \
                         the row itself, but it is {}. Give each leg's lag, such as [0, 1].",
                        short(v)
                    ));
                };
                let mut out = Vec::with_capacity(items.len());
                for l in items {
                    let span = span_of(NAME, "a lag", l)?;
                    if !(span.value() >= 0.0 && span.value().is_finite()) {
                        return Err(format!(
                            "{NAME}: a lag must be finite and at least 0, but it is {span}. \
                             Give 0 for the row itself, or a span of the clock back."
                        ));
                    }
                    out.push(span);
                }
                lags = Some(out);
            }
            "kind" => {
                kind = MOMENT_KINDS
                    .iter()
                    .find(|(n, _)| v.as_str() == Some(n))
                    .map(|(_, k)| *k)
                    .ok_or_else(|| {
                        format!(
                            "{NAME}: kind must be \"central\", \"raw\", \"standardized\" or \
                             \"cumulant\", but it is {}. Set kind to one of those words.",
                            short(v)
                        )
                    })?;
            }
            "bias" => {
                bias = Some(v.as_bool().ok_or_else(|| {
                    format!(
                        "{NAME}: bias must be true or false, but it is {}. Set bias to true or \
                         false, or leave it out for the statistic's own default.",
                        short(v)
                    )
                })?);
            }
            "lag_tolerance" => {
                let span = span_of(NAME, k, v)?;
                if !(span.value() >= 0.0 && span.value().is_finite()) {
                    return Err(format!(
                        "{NAME}: lag_tolerance must be finite and at least 0, but it is \
                         {span}. Give how far before t - lag the row a lag pairs with may sit."
                    ));
                }
                lag_tolerance = Some(span);
            }
            other => {
                return Err(format!(
                    "{NAME}: {other:?} is not one of its parameters, which are half_life, \
                     window_size, closed, partial, powers, lags, kind, bias and lag_tolerance. \
                     Remove {other:?}, or correct its name."
                ));
            }
        }
    }
    let n = match &*op.input {
        Node::Call(_, legs) => legs.len(),
        _ => 0,
    };
    let powers = powers.ok_or_else(|| {
        format!(
            "{NAME}: powers is not set, but each leg needs its power. Give powers, a whole \
             number from 1 a leg, such as [3]."
        )
    })?;
    let lags = lags.unwrap_or_else(|| vec![Span::Units(0.0); n]);
    if powers.len() != n || lags.len() != n {
        return Err(format!(
            "{NAME}: it has {n} legs, but {} powers and {} lags. Give one power and one lag a \
             leg.",
            powers.len(),
            lags.len()
        ));
    }
    let Some(h) = &op.half_life else {
        return Err(format!(
            "{NAME}: half_life is not set, but every window operator needs it. Set half_life to \
             a positive number of clock units or a duration, or to inf for no decay."
        ));
    };
    if h.value().is_nan() || h.value() <= 0.0 {
        return Err(format!(
            "{NAME}: half_life must be above 0, but it is {h}. Set half_life to a positive \
             number of clock units or a duration, or to inf for no decay."
        ));
    }
    match &op.window_size {
        Some(w) if !(w.value().is_finite() && w.value() > 0.0) => {
            return Err(format!(
                "{NAME}: window_size must be finite and above 0, but it is {w}. Set window_size \
                 to a finite positive number of clock units or a duration."
            ));
        }
        None if h.value().is_infinite() => {
            return Err(format!(
                "{NAME}: half_life is inf and window_size is not set, so a value held from \
                 before the first row would have no finite weight. Set window_size, or give \
                 half_life a finite value."
            ));
        }
        _ => {}
    }
    op.moments = Some(Box::new(MomentsNode {
        powers,
        lags,
        kind,
        bias,
        lag_tolerance,
    }));
    Ok(Node::Op(op))
}

/// A whole number of an `stl` or `mstl` parameter, at least `min`.
fn stl_uint(name: &str, k: &str, v: &Value, min: u32) -> Result<u32, String> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n >= min)
        .ok_or_else(|| {
            format!(
                "{name}: {k} must be a whole number of at least {min}, but it is {}. Give {k} \
                 as such a number.",
                short(v)
            )
        })
}

/// An `stl` or `mstl` node from its parameters, each checked by name; the
/// spans, degrees and the window are checked against statsmodels' rules
/// when the run is built.
fn stl_of(name: &str, input: Node, params: &Map<String, Value>) -> Result<OpNode, String> {
    let mstl = name == "mstl";
    let mut periods: Option<Vec<u32>> = None;
    let mut spans: Option<Vec<u32>> = None;
    let mut season: Option<u32> = None;
    let mut component = "trend".to_string();
    let mut p = StlParams {
        seasons: Vec::new(),
        iterate: 2,
        trend: None,
        low_pass: None,
        seasonal_deg: 1,
        trend_deg: 1,
        low_pass_deg: 1,
        seasonal_jump: 1,
        trend_jump: 1,
        low_pass_jump: 1,
        robust: false,
        inner_iter: None,
        outer_iter: None,
        window: None,
        lag: None,
        closed: rheon_core::Closed::Left,
        component: crate::windows::StlComponent::Trend,
        max_state_bytes: None,
    };
    let list = |k: &str, v: &Value| -> Result<Vec<u32>, String> {
        match v {
            Value::Array(a) if !a.is_empty() => a.iter().map(|x| stl_uint(name, k, x, 1)).collect(),
            _ => Err(format!(
                "{name}: {k} must be a list of whole numbers, but it is {}. Give {k} as a list, \
                 such as [24, 168].",
                short(v)
            )),
        }
    };
    let deg = |k: &str, v: &Value| -> Result<u8, String> {
        match v.as_u64() {
            Some(0) => Ok(0),
            Some(1) => Ok(1),
            _ => Err(format!(
                "{name}: {k} must be 0 or 1, but it is {}. Give 0 for a local mean or 1 for a \
                 local line.",
                short(v)
            )),
        }
    };
    for (k, v) in params {
        match (k.as_str(), mstl) {
            ("period", false) => periods = Some(vec![stl_uint(name, k, v, 2)?]),
            ("seasonal", false) => spans = Some(vec![stl_uint(name, k, v, 3)?]),
            ("periods", true) => periods = Some(list(k, v)?),
            ("windows", true) => spans = Some(list(k, v)?),
            ("iterate", true) => p.iterate = stl_uint(name, k, v, 1)?,
            ("season", true) => season = Some(stl_uint(name, k, v, 2)?),
            ("trend", _) => p.trend = Some(stl_uint(name, k, v, 3)?),
            ("low_pass", _) => p.low_pass = Some(stl_uint(name, k, v, 3)?),
            ("seasonal_deg", _) => p.seasonal_deg = deg(k, v)?,
            ("trend_deg", _) => p.trend_deg = deg(k, v)?,
            ("low_pass_deg", _) => p.low_pass_deg = deg(k, v)?,
            ("seasonal_jump", _) => p.seasonal_jump = stl_uint(name, k, v, 1)?,
            ("trend_jump", _) => p.trend_jump = stl_uint(name, k, v, 1)?,
            ("low_pass_jump", _) => p.low_pass_jump = stl_uint(name, k, v, 1)?,
            ("inner_iter", _) => p.inner_iter = Some(stl_uint(name, k, v, 1)?),
            ("outer_iter", _) => p.outer_iter = Some(stl_uint(name, k, v, 0)?),
            ("window", _) => p.window = Some(stl_uint(name, k, v, 1)?),
            ("lag", _) => p.lag = Some(stl_uint(name, k, v, 0)?),
            ("max_state_bytes", _) => {
                p.max_state_bytes = Some(v.as_u64().filter(|&b| b > 0).ok_or_else(|| {
                    format!(
                        "{name}: max_state_bytes must be a whole number above 0, but it is {}. \
                         Give the most bytes a group's ring may hold.",
                        short(v)
                    )
                })?);
            }
            ("robust", _) => {
                p.robust = v.as_bool().ok_or_else(|| {
                    format!(
                        "{name}: robust must be true or false, but it is {}. Set robust to true \
                         or false.",
                        short(v)
                    )
                })?;
            }
            ("closed", _) => {
                p.closed = match v.as_str() {
                    Some("left") => rheon_core::Closed::Left,
                    Some("right") => rheon_core::Closed::Right,
                    _ => {
                        return Err(format!(
                            "{name}: closed must be \"left\" or \"right\", but it is {}. Set \
                             closed to \"left\", which reads no row at or after the row's own, \
                             or \"right\".",
                            short(v)
                        ));
                    }
                };
            }
            ("component", _) => {
                component = match v.as_str() {
                    Some(c @ ("seasonal" | "trend" | "resid")) => c.to_string(),
                    _ => {
                        return Err(format!(
                            "{name}: component must be \"seasonal\", \"trend\" or \"resid\", \
                             but it is {}. Set component to one of those words.",
                            short(v)
                        ));
                    }
                };
            }
            _ => {
                return Err(format!(
                    "{name}: {k:?} is not one of its parameters. Remove it, or check its \
                     spelling."
                ));
            }
        }
    }
    let Some(periods) = periods else {
        let k = if mstl { "periods" } else { "period" };
        return Err(format!(
            "{name}: {k} is needed. Give the season's length in rows."
        ));
    };
    // statsmodels' defaults: 7 for STL; 7 + 4 i for MSTL's i-th shortest.
    let mut order: Vec<usize> = (0..periods.len()).collect();
    order.sort_by_key(|&i| periods[i]);
    let spans = match spans {
        Some(s) if s.len() != periods.len() => {
            return Err(format!(
                "{name}: windows has {} spans for {} periods. Give one span a period.",
                s.len(),
                periods.len()
            ));
        }
        Some(s) => s,
        None if mstl => {
            let mut s = vec![0; periods.len()];
            for (rank, &i) in order.iter().enumerate() {
                s[i] = 7 + 4 * (rank as u32 + 1);
            }
            s
        }
        None => vec![7],
    };
    p.seasons = order.iter().map(|&i| (periods[i], spans[i])).collect();
    p.component = match component.as_str() {
        "trend" => crate::windows::StlComponent::Trend,
        "resid" => crate::windows::StlComponent::Resid,
        _ if !mstl => crate::windows::StlComponent::Seasonal(0),
        _ => {
            let Some(want) = season else {
                return Err(format!(
                    "{name}: component \"seasonal\" needs season, the period whose seasonal \
                     to read. Give season as one of the periods."
                ));
            };
            let Some(i) = p.seasons.iter().position(|&(per, _)| per == want) else {
                return Err(format!(
                    "{name}: season {want} is not one of the periods. Give season as one of \
                     them."
                ));
            };
            crate::windows::StlComponent::Seasonal(i as u32)
        }
    };
    let kind = if mstl { OpKind::Mstl } else { OpKind::Stl };
    Ok(OpNode {
        kind,
        input: Box::new(input),
        half_life: None,
        window_size: None,
        closed: Closed::Right,
        min_samples: 1,
        partial: None,
        bias: false,
        quantile: None,
        weight: None,
        max_state_bytes: None,
        order: Order::default(),
        neighbors: None,
        pick: None,
        moments: None,
        profile: None,
        since: None,
        stl: Some(Box::new(p)),
    })
}

/// An `stl` or `mstl` node's parameters as the list form writes them.
fn stl_json(kind: OpKind, p: &StlParams) -> Value {
    let mut m = Map::new();
    if kind == OpKind::Mstl {
        m.insert(
            "periods".into(),
            json!(p.seasons.iter().map(|s| s.0).collect::<Vec<_>>()),
        );
        m.insert(
            "windows".into(),
            json!(p.seasons.iter().map(|s| s.1).collect::<Vec<_>>()),
        );
        m.insert("iterate".into(), json!(p.iterate));
    } else {
        m.insert("period".into(), json!(p.seasons[0].0));
        m.insert("seasonal".into(), json!(p.seasons[0].1));
    }
    for (k, v) in [("trend", p.trend), ("low_pass", p.low_pass)] {
        if let Some(v) = v {
            m.insert(k.into(), json!(v));
        }
    }
    m.insert("seasonal_deg".into(), json!(p.seasonal_deg));
    m.insert("trend_deg".into(), json!(p.trend_deg));
    m.insert("low_pass_deg".into(), json!(p.low_pass_deg));
    m.insert("seasonal_jump".into(), json!(p.seasonal_jump));
    m.insert("trend_jump".into(), json!(p.trend_jump));
    m.insert("low_pass_jump".into(), json!(p.low_pass_jump));
    m.insert("robust".into(), json!(p.robust));
    for (k, v) in [
        ("inner_iter", p.inner_iter),
        ("outer_iter", p.outer_iter),
        ("window", p.window),
        ("lag", p.lag),
    ] {
        if let Some(v) = v {
            m.insert(k.into(), json!(v));
        }
    }
    let closed = if p.closed == rheon_core::Closed::Left {
        "left"
    } else {
        "right"
    };
    m.insert("closed".into(), json!(closed));
    match p.component {
        crate::windows::StlComponent::Trend => {
            m.insert("component".into(), json!("trend"));
        }
        crate::windows::StlComponent::Resid => {
            m.insert("component".into(), json!("resid"));
        }
        crate::windows::StlComponent::Seasonal(i) => {
            m.insert("component".into(), json!("seasonal"));
            if kind == OpKind::Mstl {
                m.insert("season".into(), json!(p.seasons[i as usize].0));
            }
        }
    }
    if let Some(b) = p.max_state_bytes {
        m.insert("max_state_bytes".into(), json!(b));
    }
    Value::Object(m)
}

/// An `ewm_profile` node from its parameters, each checked by name.
/// The most harmonics a profile takes: a period cut finer than this is
/// what `buckets` is for.
const MAX_HARMONICS: u32 = 64;

fn profile_of(input: Node, params: &Map<String, Value>) -> Result<OpNode, String> {
    let name = "ewm_profile";
    let mut op = OpNode {
        kind: OpKind::EwmProfile,
        input: Box::new(input),
        half_life: None,
        window_size: None,
        closed: Closed::Right,
        min_samples: 1,
        partial: None,
        bias: false,
        quantile: None,
        weight: None,
        max_state_bytes: None,
        order: Order::default(),
        neighbors: None,
        pick: None,
        moments: None,
        profile: None,
        since: None,
        tol: 0.0,
        event: None,
        step: None,
        stl: None,
    };
    let (mut period, mut offset, mut buckets, mut stat) = (None, None, None, ProfileStat::Mean);
    let mut harmonics: Option<u32> = None;
    let (mut weight, mut max_state_bytes) = (None, None);
    let mut bias = None;
    for (k, v) in params {
        match k.as_str() {
            "half_life" => op.half_life = Some(span_of(name, k, v)?),
            "window_size" => op.window_size = Some(span_of(name, k, v)?),
            "quantile" => {
                op.quantile =
                    Some(v.as_f64().filter(|q| *q > 0.0 && *q < 1.0).ok_or_else(|| {
                        format!(
                            "{name}: quantile must be a number above 0 and below 1, but it is {}. \
                         Set quantile within that range, such as 0.5 for the median.",
                            short(v)
                        )
                    })?);
            }
            "closed" => {
                op.closed = Closed::parse(v.as_str().unwrap_or("")).ok_or_else(|| {
                    format!(
                        "{name}: closed must be \"right\", \"left\", \"both\" or \"none\", \
                         but it is {}. Set closed to one of those words.",
                        short(v)
                    )
                })?;
            }
            "period" => period = Some(span_of(name, k, v)?),
            "offset" => offset = Some(span_of(name, k, v)?),
            "buckets" => {
                buckets = Some(
                    v.as_u64()
                        .and_then(|b| u32::try_from(b).ok())
                        .filter(|&b| b >= 1)
                        .ok_or_else(|| {
                            format!(
                                "{name}: buckets must be a whole number from 1 to {}, but it \
                                 is {}. Set buckets to the number of phases of the period, \
                                 such as 24 for the hours of a day.",
                                u32::MAX,
                                short(v)
                            )
                        })?,
                );
            }
            "harmonics" => {
                harmonics = Some(
                    v.as_u64()
                        .and_then(|k| u32::try_from(k).ok())
                        .filter(|&k| (1..=MAX_HARMONICS).contains(&k))
                        .ok_or_else(|| {
                            format!(
                                "{name}: harmonics must be a whole number from 1 to \
                                 {MAX_HARMONICS}, but it is {}. Set harmonics to the Fourier \
                                 terms the profile takes, such as 3 for the daily shape of an \
                                 hourly series.",
                                short(v)
                            )
                        })?,
                );
            }
            "stat" => {
                stat = v.as_str().and_then(ProfileStat::parse).ok_or_else(|| {
                    format!(
                        "{name}: stat must be \"mean\", \"var\", \"std\" or \"quantile\", but \
                         it is {}. Set stat to one of those words.",
                        short(v)
                    )
                })?;
            }
            "bias" => {
                bias = Some(v.as_bool().ok_or_else(|| {
                    format!(
                        "{name}: bias must be true or false, but it is {}. Set bias to true or \
                         false.",
                        short(v)
                    )
                })?);
            }
            "min_samples" => {
                op.min_samples = v
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n >= 1)
                    .ok_or_else(|| {
                        format!(
                            "{name}: min_samples must be a whole number from 1 to {}, but it is \
                             {}. Set min_samples to the rows a phase needs before it is read.",
                            u32::MAX,
                            short(v)
                        )
                    })?;
            }
            "weight" => {
                let w = Node::from_json(v)?;
                w.no_window_operator(name)?;
                weight = Some(w);
            }
            "max_state_bytes" => {
                let ceiling = PROFILE_STATE_BYTES_CEILING;
                max_state_bytes = Some(v.as_u64().filter(|&b| b >= 1 && b <= ceiling).ok_or_else(
                    || {
                        format!(
                            "{name}: max_state_bytes must be a whole number from 1 to {ceiling} \
                             (64 MiB), but it is {}. Set it within that range, or leave it out \
                             for 262144 (256 KB).",
                            short(v)
                        )
                    },
                )?);
            }
            other => {
                return Err(format!(
                    "{name}: {other:?} is not one of its parameters, which are period, buckets, \
                     harmonics, half_life, window_size, stat, quantile, closed, offset, weight, \
                     min_samples, bias and max_state_bytes. Remove {other:?}, or correct its name."
                ));
            }
        }
    }
    let Some(h) = &op.half_life else {
        return Err(format!(
            "{name}: half_life is not set, but a profile needs it. Set half_life to a positive \
             number of clock units or a duration, or to inf for no decay."
        ));
    };
    if h.value().is_nan() || h.value() <= 0.0 {
        return Err(format!(
            "{name}: half_life must be above 0, but it is {h}. Set half_life to a positive \
             number of clock units or a duration, or to inf for no decay."
        ));
    }
    let Some(period) = period else {
        return Err(format!(
            "{name}: period is not set, but a profile needs it: the span the phase repeats \
             over. Set period, such as \"1d\" for the time of day."
        ));
    };
    if !(period.value().is_finite() && period.value() > 0.0) {
        return Err(format!(
            "{name}: period must be finite and above 0, but it is {period}. Set period to a \
             positive number of clock units or a duration, such as \"1d\"."
        ));
    }
    if let Some(o) = &offset
        && !o.value().is_finite()
    {
        return Err(format!(
            "{name}: offset must be finite, but it is {o}. Set offset to where the period \
             starts after its default start, such as \"9h30m\"."
        ));
    }
    let (buckets, harmonics) = match (buckets, harmonics) {
        (Some(b), None) => (b, 0),
        (None, Some(k)) => {
            if stat == ProfileStat::Quantile || bias.is_some() {
                return Err(format!(
                    "{name}: harmonics takes stat \"mean\", \"var\" or \"std\" and no bias: \
                     a Fourier fit's variance is the fit of x² less the square of the fit of \
                     x, and a quantile has no one-pass fit. Remove bias or quantile, or take \
                     buckets."
                ));
            }
            (0, k)
        }
        (None, None) => {
            return Err(format!(
                "{name}: buckets or harmonics is needed: how many equal phases the period is \
                 cut into, or how many Fourier terms fit it. Set one, such as buckets=24 for \
                 the hours of a day."
            ));
        }
        (Some(_), Some(_)) => {
            return Err(format!(
                "{name}: buckets and harmonics are both given, but a profile is one form or \
                 the other. Remove one of them."
            ));
        }
    };
    if let Some(w) = &op.window_size
        && !(w.value().is_finite() && w.value() > 0.0)
    {
        return Err(format!(
            "{name}: window_size must be finite and above 0, but it is {w}. Set window_size to \
             a finite positive number of clock units or a duration."
        ));
    }
    match (stat, op.quantile) {
        (ProfileStat::Quantile, None) => {
            return Err(format!(
                "{name}: quantile is not set, but stat=\"quantile\" needs it. Set quantile to a \
                 number above 0 and below 1, such as 0.5 for the median."
            ));
        }
        (s, Some(_)) if s != ProfileStat::Quantile => {
            return Err(format!(
                "{name}: quantile is given, but only stat=\"quantile\" takes it. Remove \
                 quantile, or set stat to \"quantile\"."
            ));
        }
        _ => {}
    }
    if bias.is_some() && matches!(stat, ProfileStat::Mean | ProfileStat::Quantile) {
        return Err(format!(
            "{name}: bias is given, but only a variance or a standard deviation takes it. \
             Remove bias, or set stat to \"var\" or \"std\"."
        ));
    }
    op.bias = bias.unwrap_or(false);
    op.profile = Some(Box::new(ProfileParams {
        period,
        offset,
        buckets,
        harmonics,
        stat,
        weight,
        max_state_bytes,
    }));
    Ok(op)
}

fn span_of(name: &str, key: &str, v: &Value) -> Result<Span, String> {
    serde_json::from_value::<Span>(v.clone()).map_err(|_| {
        format!(
            "{name}: {key} must be a number of clock units or a duration, but it is {}. \
                 Give {key} as a number, or as a duration such as \"10m\".",
            short(v)
        )
    })
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    // On a character boundary: a slice at byte 60 panicked inside a
    // multibyte name (review R1, B3).
    match s.char_indices().nth(60) {
        Some((cut, _)) => format!("{}...", &s[..cut]),
        None => s,
    }
}

pub(crate) mod scalar;

#[cfg(test)]
mod tests {
    /// Review R1, B3: the message's cut of a long value fell inside a
    /// multibyte character and panicked.
    #[test]
    fn a_long_name_is_cut_on_a_character_boundary() {
        let name = "€".repeat(70);
        let err = super::Node::from_json(&serde_json::json!({ "co": name })).unwrap_err();
        assert!(err.contains("..."), "{err}");
    }

    use super::*;

    fn tree(text: &str) -> Node {
        Node::parse(text).unwrap()
    }

    /// Every node kind survives the list form in both directions.
    #[test]
    fn every_node_kind_round_trips() {
        for text in [
            r#"["col","a"]"#,
            r#"["lit",1.5]"#,
            r#"["lit",2]"#,
            r#"["lit","x"]"#,
            r#"["lit",true]"#,
            r#"["lit"]"#,
            r#"["+",["col","a"],["col","b"]]"#,
            r#"["/",["col","a"],["lit",2]]"#,
            r#"["**",["col","a"],["lit",2]]"#,
            r#"["==",["col","s"],["lit","x"]]"#,
            r#"["&",[">",["col","a"],["lit",0]],["is_null",["col","b"]]]"#,
            r#"["neg",["col","a"]]"#,
            r#"["log",["col","a"],["lit",2.718281828459045]]"#,
            r#"["clip",["col","a"],["lit",0],["lit"]]"#,
            r#"["clip",["col","a"],["lit"],["lit",1]]"#,
            r#"["fill_null",["col","a"],["lit",0.0]]"#,
            r#"["when",[">",["col","a"],["lit",0]],["col","b"],["lit"]]"#,
            r#"["cast",["col","a"],"Float64"]"#,
            r#"["cast",["col","a"],"Int8","non_strict"]"#,
            r#"["fill_null",["col","a"],["lit"]]"#,
            r#"["alias",["col","a"],"z"]"#,
            r#"["ewm_mean",["col","mid"],{"closed":"right","half_life":"10s","min_samples":1,"window_size":"1m"}]"#,
            r#"["rewm_sum",["*",["col","p"],["col","q"]],{"closed":"both","half_life":5.0,"min_samples":3,"partial":"keep","window_size":60.0}]"#,
            r#"["ewm_rate",["increment",["col","cum"]],{"closed":"right","half_life":30.0,"min_samples":1}]"#,
            r#"["increment",["col","cum"]]"#,
        ] {
            let node = tree(text);
            assert_eq!(node.to_json().to_string(), text, "{text}");
            assert_eq!(Node::parse(&node.to_json().to_string()).unwrap(), node);
        }
        // A null written the old way, `["lit", null]`, still reads, and
        // writes back as `["lit"]`, the form TOML can carry (review R2, P3).
        let old = tree(r#"["fill_null",["col","a"],["lit",null]]"#);
        assert_eq!(old, tree(r#"["fill_null",["col","a"],["lit"]]"#));
        assert_eq!(
            old.to_json().to_string(),
            r#"["fill_null",["col","a"],["lit"]]"#
        );
    }

    /// The rebuilt expression computes what the formula says, on a frame.
    #[test]
    fn the_expression_computes_the_formula() {
        let df = df!("a" => [1.0, 4.0, f64::NAN], "b" => [2.0, 0.5, 1.0], "s" => ["x", "y", "x"])
            .unwrap()
            .lazy()
            .with_columns([col("a").fill_nan(lit(NULL))])
            .collect()
            .unwrap();
        let cases = [
            (
                r#"["-",["col","a"],["lit",1.5]]"#,
                vec![Some(-0.5), Some(2.5), None],
            ),
            (
                r#"["/",["col","a"],["col","b"]]"#,
                vec![Some(0.5), Some(8.0), None],
            ),
            (
                r#"["fill_null",["col","a"],["lit",0.0]]"#,
                vec![Some(1.0), Some(4.0), Some(0.0)],
            ),
            (
                r#"["when",["==",["col","s"],["lit","x"]],["col","b"],["lit",null]]"#,
                vec![Some(2.0), None, Some(1.0)],
            ),
            (r#"["sqrt",["col","a"]]"#, vec![Some(1.0), Some(2.0), None]),
            (
                r#"["log",["col","a"],["lit",2.0]]"#,
                vec![Some(0.0), Some(2.0), None],
            ),
            (
                r#"["clip",["col","a"],["lit",2.0],null]"#,
                vec![Some(2.0), Some(4.0), None],
            ),
            (
                r#"["neg",["col","b"]]"#,
                vec![Some(-2.0), Some(-0.5), Some(-1.0)],
            ),
            (
                r#"["cast",["**",["col","b"],["lit",2]],"Float64"]"#,
                vec![Some(4.0), Some(0.25), Some(1.0)],
            ),
        ];
        for (text, want) in cases {
            let e = tree(text)
                .to_expr(&|_| unreachable!("no operator here"))
                .alias("out");
            let got = df.clone().lazy().select([e]).collect().unwrap();
            let got: Vec<Option<f64>> = got.column("out").unwrap().f64().unwrap().iter().collect();
            assert_eq!(got, want, "{text}");
        }
    }

    /// Review 2026-10-05 (YB1): a date or a time-of-day literal arrives as
    /// `["cast", ["lit", n], "Date"]` (`"Time"`), and is that dtype, as
    /// Polars' own literal is: its text is the day's (the time's), and a date
    /// column less it is a duration, where the bare integer was refused.
    #[test]
    fn a_date_or_time_literal_is_its_own_dtype() {
        let df = df!("d" => [19723i32, 19725])
            .unwrap()
            .lazy()
            .with_columns([col("d").cast(DataType::Date)])
            .collect()
            .unwrap();
        let run = |text: &str| {
            let e = tree(text)
                .to_expr(&|_| unreachable!("no operator here"))
                .alias("out");
            let got = df.clone().lazy().select([e]).collect().unwrap();
            got.column("out").unwrap().clone()
        };
        let day = run(r#"["cast",["cast",["lit",19725],"Date"],"String"]"#);
        assert_eq!(day.str().unwrap().get(0), Some("2024-01-03"));
        let time = run(r#"["cast",["cast",["lit",45000000000000],"Time"],"String"]"#);
        assert_eq!(time.str().unwrap().get(0), Some("12:30:00"));
        let gap = run(r#"["-",["col","d"],["cast",["lit",19723],"Date"]]"#);
        let DataType::Duration(unit) = gap.dtype() else {
            panic!("a date less a date is a duration, got {}", gap.dtype());
        };
        let per_day: i64 = match unit {
            TimeUnit::Milliseconds => 86_400_000,
            TimeUnit::Microseconds => 86_400_000_000,
            TimeUnit::Nanoseconds => 86_400_000_000_000,
        };
        let stored = gap.cast(&DataType::Int64).unwrap();
        let days: Vec<Option<i64>> = stored
            .i64()
            .unwrap()
            .iter()
            .map(|v| v.map(|v| v / per_day))
            .collect();
        assert_eq!(days, vec![Some(0), Some(2)]);
    }

    /// An operator is resolved to whatever the caller hands over, so a run
    /// can stand a hidden column in for it.
    #[test]
    fn operators_are_resolved_by_the_caller_and_collected_once_each() {
        let f = tree(
            r#"["/",["rewm_sum",["col","n"],{"half_life":1.0,"window_size":2.0}],["rewm_sum",["col","q"],{"half_life":1.0,"window_size":2.0}]]"#,
        );
        let ops = f.operators();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].kind, OpKind::RewmSum);
        assert_eq!(ops[0].direction(), Direction::Forward);
        assert_ne!(ops[0].key(), ops[1].key());
        assert_eq!(f.columns(), vec!["n".to_string(), "q".to_string()]);
        let df = df!("@0" => [1.0], "@1" => [4.0]).unwrap();
        let e = f
            .to_expr(&|op| {
                col(if op.key().contains("\"n\"") {
                    "@0"
                } else {
                    "@1"
                })
            })
            .alias("out");
        let got = df.lazy().select([e]).collect().unwrap();
        assert_eq!(got.column("out").unwrap().f64().unwrap().get(0), Some(0.25));
    }

    /// The refusal's list of what is read names every node that is read,
    /// and nothing else: it once left out `pow`, negation, `&`, `|`,
    /// `is_null` and `is_not_null` (task 238).
    #[test]
    fn the_refusal_lists_every_node_that_is_read() {
        let mut listed: Vec<&str> = ACCEPTED
            .iter()
            .flat_map(|(_, n)| n.iter().copied())
            .collect();
        let mut read: Vec<&str> = CALLS.iter().map(|(n, _)| *n).collect();
        read.extend(["col", "lit"]);
        listed.sort_unstable();
        read.sort_unstable();
        assert_eq!(listed, read);
        for head in WINDOW_HEADS {
            assert!(OpKind::parse(head).is_some_and(|k| !k.is_row()), "{head}");
        }
        assert_eq!(
            WINDOW_HEADS.len(),
            15,
            "every window operator but the row operators"
        );
        let err = Node::from_json(&json!(["shift", ["col", "a"]])).unwrap_err();
        assert!(
            err.contains(
                "these operators within one row: col, lit, +, -, *, /, neg, ==, !=, <, <=, >, >=, \
                 &, |, log, exp, abs, sqrt, **, clip, fill_null, is_null, is_not_null, when, cast \
                 and alias."
            ),
            "{err}"
        );
    }

    /// What is not element-wise, or not a formula at all, is refused by
    /// name before any kernel exists.
    #[test]
    fn what_is_not_read_is_refused_by_name() {
        for (text, says) in [
            (
                r#"["shift",["col","a"],["lit",1]]"#,
                "\"shift\" is not one of the operators a formula can hold",
            ),
            (
                r#"["cum_sum",["col","a"]]"#,
                "\"cum_sum\" is not one of the operators a formula can hold",
            ),
            (
                r#"["mean",["col","a"]]"#,
                "\"mean\" is not one of the operators a formula can hold",
            ),
            (r#"{"col":"a"}"#, "a list whose first item names its kind"),
            (r#"["col",1]"#, "takes a column's name as a string"),
            (
                r#"["+",["col","a"]]"#,
                "\"+\" takes 2 arguments, but it has 1",
            ),
            (
                r#"["cast",["col","a"],"Datetime"]"#,
                "a cast to \"Datetime\" is not one a formula can hold",
            ),
            (
                r#"["lit",[1,2]]"#,
                "a literal must be a number, a string, true, false or null",
            ),
            (r#"["ewm_mean",["col","a"],{}]"#, "half_life is not set"),
            (
                r#"["ewm_mean",["col","a"],{"half_life":0}]"#,
                "half_life must be above 0",
            ),
            (
                r#"["rewm_mean",["col","a"],{"half_life":1}]"#,
                "window_size is not set, but an operator that reads the rows after",
            ),
            (
                r#"["ewm_mean",["col","a"],{"half_life":"inf"}]"#,
                "half_life is inf and window_size is not set",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"window_size":0}]"#,
                "window_size must be finite and above 0",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"closed":"up"}]"#,
                "closed must be",
            ),
            (
                r#"["ewm_mean",["col","a"],{"half_life":1,"min_samples":0}]"#,
                "min_samples must be a whole number from 1 to",
            ),
            (
                r#"["rewm_sum",["col","a"],{"half_life":1,"window_size":2,"min_samples":-1}]"#,
                "min_samples must be a whole number from 0 to",
            ),
            // Review round 4, PD4: past a `u32` the refusal names the
            // ceiling, where it said "at least 1" of a count far above it.
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"min_samples":4294967296}]"#,
                "min_samples must be a whole number from 0 to 4294967295, but it is 4294967296",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"partial":"maybe"}]"#,
                "partial must be",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"horizon":3}]"#,
                "\"horizon\" is not one of its parameters",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1},3]"#,
                "takes 2 arguments, but it has 3",
            ),
            (
                r#"["ewm_mean",["ewm_sum",["col","a"],{"half_life":1}],{"half_life":1}]"#,
                "its input holds a window operator",
            ),
            (r#"not json"#, "formula: the text is not JSON"),
        ] {
            let err = Node::parse(text).unwrap_err();
            assert!(err.contains(says), "{text}: {err}");
        }
        // A step inside an operator's input is the one nesting allowed: an
        // increment or a tick sign.
        assert!(Node::parse(r#"["ewm_sum",["increment",["col","c"]],{"half_life":1}]"#).is_ok());
        let signed = r#"["ewm_sum",["*",["tick_sign",["col","p"]],["col","q"]],{"closed":"right","half_life":1.0,"min_samples":1}]"#;
        let node = Node::parse(signed).unwrap();
        assert!(node.has_window_operator());
        assert_eq!(
            node.to_json().to_string(),
            signed,
            "it writes no parameters"
        );
        let sign = tree(r#"["tick_sign",["col","p"]]"#);
        assert!(!sign.has_window_operator());
        assert!(OpKind::TickSign.is_step() && OpKind::Increment.is_step());
        assert!(!OpKind::EwmMean.is_step() && !OpKind::RewmSum.is_step());
        assert_eq!(OpKind::TickSign.stat(), None);
        assert_eq!(OpKind::TickSign.direction(), Direction::Backward);
        assert_eq!(sign.columns(), ["p"]);
        let err =
            Node::parse(r#"["tick_sign",["ewm_sum",["col","p"],{"half_life":1}]]"#).unwrap_err();
        assert!(
            err.contains("tick_sign: its input holds a window operator")
                && err.contains("such as a column."),
            "{err}"
        );
        let err = Node::parse(r#"["tick_sign",["col","p"],{"half_life":1}]"#).unwrap_err();
        assert!(err.contains("takes 1 argument"), "{err}");
        let f = tree(r#"["alias",["-",["col","a"],["col","b"]],"spread"]"#);
        assert_eq!(f.alias(), Some("spread"));
        assert!(!f.has_window_operator());
    }

    /// Task 252: the refusals no other test reached, each by its words --
    /// a list headed by no name, a name that is not text, a cast's third
    /// argument that is not `"non_strict"`, parameters that are not an
    /// object, and a variance's `bias` that is not a boolean.
    #[test]
    fn each_malformed_node_is_refused_in_its_own_words() {
        for (text, says) in [
            (
                r#"[]"#,
                "a list whose first item names its kind, such as [\"col\", \"x\"], but it is []",
            ),
            (
                r#"[1,["col","a"]]"#,
                "a list whose first item names its kind, such as [\"col\", \"x\"], but it is [1,",
            ),
            (
                r#"["alias",["col","a"],3]"#,
                "\"alias\" takes the output's name as its second argument, but it has 3",
            ),
            (
                r#"["cast",["col","a"],7]"#,
                "\"cast\" takes a dtype's name, such as \"Float64\", as its second argument, but it has 7",
            ),
            (
                r#"["cast",["col","a"],"Float64","strict"]"#,
                "\"cast\" takes \"non_strict\" as its third argument, but it has \"strict\"",
            ),
            (
                r#"["ewm_mean",["col","a"],3]"#,
                "\"ewm_mean\" takes its parameters as an object, but it has 3",
            ),
            (
                r#"["ewm_var",["col","a"],{"half_life":1,"bias":"yes"}]"#,
                "ewm_var: bias must be true or false, but it is \"yes\"",
            ),
            (
                r#"["ewm_mean",["col","a"],{"half_life":1,"bias":true}]"#,
                "ewm_mean: \"bias\" is not one of its parameters",
            ),
        ] {
            let err = Node::parse(text).unwrap_err();
            assert!(err.contains(says), "{text}: {err}");
        }
        // The variance's own `bias` reads, and is kept.
        let Node::Op(op) = tree(r#"["ewm_std",["col","a"],{"half_life":1,"bias":true}]"#) else {
            panic!("an operator");
        };
        assert!(op.bias);
    }

    /// Task 272 (d): a quantile's level is required, a number above 0 and
    /// below 1, and no other operator's; it reads, is kept, and is written
    /// back in the list form, so one quantile asked for twice is one column
    /// and two levels are two.
    #[test]
    fn a_quantiles_level_is_read_kept_and_refused_in_its_own_words() {
        for (text, says) in [
            (
                r#"["ewm_quantile",["col","a"],{"half_life":1}]"#,
                "ewm_quantile: quantile is not set",
            ),
            (
                r#"["rewm_quantile",["col","a"],{"half_life":1,"window_size":2}]"#,
                "rewm_quantile: quantile is not set",
            ),
            (
                r#"["ewm_quantile",["col","a"],{"half_life":1,"quantile":1}]"#,
                "ewm_quantile: quantile must be a number above 0 and below 1, but it is 1",
            ),
            (
                r#"["ewm_quantile",["col","a"],{"half_life":1,"quantile":0}]"#,
                "but it is 0",
            ),
            (
                r#"["ewm_quantile",["col","a"],{"half_life":1,"quantile":"half"}]"#,
                "but it is \"half\"",
            ),
            (
                r#"["ewm_sum",["col","a"],{"half_life":1,"quantile":0.5}]"#,
                "ewm_sum: \"quantile\" is not one of its parameters",
            ),
            (
                r#"["ewm_quantile",["col","a"],{"half_life":1,"quantile":0.5,"bias":true}]"#,
                "\"bias\" is not one of its parameters, which are half_life, window_size, closed, \
                 min_samples, partial and quantile",
            ),
            (
                r#"["rewm_quantile",["col","a"],{"half_life":1,"quantile":0.5}]"#,
                "window_size is not set, but an operator that reads the rows after",
            ),
        ] {
            let err = Node::parse(text).unwrap_err();
            assert!(err.contains(says), "{text}: {err}");
        }
        let text = r#"["ewm_quantile",["abs",["col","r"]],{"half_life":"inf","quantile":0.25}]"#;
        let Node::Op(op) = tree(text) else {
            panic!("an operator");
        };
        assert_eq!((op.kind, op.quantile), (OpKind::EwmQuantile, Some(0.25)));
        let again = Node::Op(op.clone()).to_json().to_string();
        assert!(again.contains(r#""quantile":0.25"#), "{again}");
        assert_eq!(Node::parse(&again).unwrap(), Node::Op(op.clone()));
        let other =
            tree(r#"["ewm_quantile",["abs",["col","r"]],{"half_life":"inf","quantile":0.75}]"#);
        let Node::Op(other) = other else {
            panic!("an operator");
        };
        assert_ne!(op.key(), other.key());
        let Node::Op(fwd) =
            tree(r#"["rewm_quantile",["col","a"],{"half_life":1,"window_size":2,"quantile":0.9}]"#)
        else {
            panic!("an operator");
        };
        assert_eq!(fwd.direction(), Direction::Forward);
    }

    /// Task 252: the expression a `clip` with no bound at all, a `clip` with
    /// only an upper bound and a boolean literal compute; and the text a
    /// tree shows (`Display`) is its list form, its root's alias `None`
    /// unless the root is an `alias`.
    #[test]
    fn a_clip_without_a_lower_bound_and_a_boolean_literal_compute_the_formula() {
        let df = df!("a" => [1.0, 4.0, -3.0], "b" => [2.0, 0.5, 1.0]).unwrap();
        let cases = [
            (
                r#"["clip",["col","a"],["lit"],["lit"]]"#,
                vec![Some(1.0), Some(4.0), Some(-3.0)],
            ),
            (
                r#"["clip",["col","a"],null,null]"#,
                vec![Some(1.0), Some(4.0), Some(-3.0)],
            ),
            (
                r#"["clip",["col","a"],["lit"],["lit",2.0]]"#,
                vec![Some(1.0), Some(2.0), Some(-3.0)],
            ),
            (
                r#"["clip",["col","a"],["lit",0.0],["lit",2.0]]"#,
                vec![Some(1.0), Some(2.0), Some(0.0)],
            ),
            (
                r#"["when",["lit",false],["col","a"],["col","b"]]"#,
                vec![Some(2.0), Some(0.5), Some(1.0)],
            ),
            (
                r#"["when",["&",[">",["col","a"],["lit",0]],["lit",true]],["col","a"],["lit"]]"#,
                vec![Some(1.0), Some(4.0), None],
            ),
        ];
        for (text, want) in cases {
            let e = tree(text)
                .to_expr(&|_| unreachable!("no operator here"))
                .alias("out");
            let got = df.clone().lazy().select([e]).collect().unwrap();
            let got: Vec<Option<f64>> = got.column("out").unwrap().f64().unwrap().iter().collect();
            assert_eq!(got, want, "{text}");
        }
        let text = r#"["-",["ewm_mean",["col","a"],{"closed":"right","half_life":2.0,"min_samples":1}],["col","a"]]"#;
        let f = tree(text);
        assert_eq!(f.to_string(), text);
        assert_eq!(format!("{f}"), f.to_json().to_string());
        assert_eq!(f.alias(), None);
        assert_eq!(tree(r#"["col","a"]"#).alias(), None);
    }
}
