//! `invert`: the root in `x` of `f(x, row) = equals(row)` on a bracket
//! `[lo(row), hi(row)]`, solved row by row by `rheon_core::find_root`, as
//! one Polars expression over the columns the four parts read.
//!
//! The node is read from the compact tree a formula is written in:
//!
//! ```text
//! ["invert", f, equals, lo, hi, {"increasing": true, "xtol": 2e-12, "rtol": 8.88e-16, "detail": false}]
//! ```
//!
//! `f` holds the unknown, the leaf `["unknown"]`; `equals`, `lo` and `hi`
//! do not. Each part may hold `["col", name]`, a numeric `["lit", v]`, the
//! binary `+ - * / **`, the unary `neg abs exp sqrt`, `["log", x, base]`
//! and `["cast", x, "Float64"]`; anything else is refused by name when the
//! node is read. `increasing` is required: it is the direction the caller
//! declares for `f` on the bracket, which names the side of the range a
//! target lies on and catches a function that is not what was declared.
//! `xtol` (above 0) and `rtol` (at least four machine epsilons) default to
//! `2e-12` and four machine epsilons.
//!
//! # The interpreter
//!
//! Each part is compiled once into a postfix program over `f64` slots, the
//! row's columns, and evaluated a row at a time with no allocation a row.
//! Beside each value it carries a bound on its rounding error (Higham's
//! running error analysis): an operation's result `v` adds `u·|v|` (`u` half
//! a machine epsilon) to the bound propagated from its operands, and `2^-1074`
//! besides, so that a result that underflows to 0 is not read as exact:
//!
//! ```text
//! a ± b   e_a + e_b
//! a · b   |a| e_b + |b| e_a + e_a e_b
//! a / b   (e_a + |a/b| e_b) / (|b| − e_b)          infinite where |b| <= e_b
//! exp a   |v| (exp(e_a) − 1)                       plus 2 ulps of v
//! ln a    −ln(1 − e_a/a)                           infinite where e_a >= a; plus 2 ulps
//! √a      e_a / (√a + √max(a − e_a, 0))
//! a ** b  |v| (exp(|b| l + |ln|a|| e_b + l e_b) − 1),  l = −ln(1 − e_a/|a|); plus 2 ulps
//! ```
//!
//! The solve reads a sign as certain only where the value exceeds its bound,
//! so the bound decides which roots are certified (`rheon_core::find_root`).
//!
//! # What a row gives
//!
//! The root, or null where it is withheld, with the reason under `detail`:
//! `input_missing` where a column the node reads, `equals` or a bracket end
//! is null, NaN, infinite or beyond `1e100` in magnitude (the library's one
//! rule for a value); otherwise the solver's own reasons. A row's answer is
//! a function of that row alone, so the node holds no state and gives the
//! same bits however the stream is chunked. A solve that takes its whole
//! stated bound without meeting the tolerance would be a defect of the
//! solver, and is raised as an error naming the row, never a null.

use std::sync::Arc;

use polars::prelude::*;
use rheon_core::{Bounded, RootReport, RootTolerance, Withheld, find_root, usable};
use serde_json::Value;

/// Half a machine epsilon: the relative rounding of one operation.
const U: f64 = f64::EPSILON / 2.0;

/// The least positive double, `2^-1074`: each operation's absolute floor,
/// so that a result that underflowed carries a nonzero bound.
const ETA: f64 = f64::MIN_POSITIVE * f64::EPSILON;

/// What the platform's `exp`, `ln` and `powf` may be off by, relative to
/// their result: two units in the last place.
const LIBM: f64 = 2.0 * f64::EPSILON;

/// One instruction of a compiled part, in postfix order.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Col(usize),
    Lit(f64),
    Unknown,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Neg,
    Abs,
    Exp,
    Sqrt,
}

/// The binary nodes, as the tree writes them.
const BINARY: &[(&str, Op)] = &[
    ("+", Op::Add),
    ("-", Op::Sub),
    ("*", Op::Mul),
    ("/", Op::Div),
    ("**", Op::Pow),
    ("log", Op::Log),
];

/// The unary nodes, as the tree writes them.
const UNARY: &[(&str, Op)] = &[
    ("neg", Op::Neg),
    ("abs", Op::Abs),
    ("exp", Op::Exp),
    ("sqrt", Op::Sqrt),
];

/// A part compiled to postfix: its instructions and the most values they
/// hold at once.
#[derive(Debug, Clone, PartialEq)]
struct Program {
    ops: Vec<Op>,
    depth: usize,
}

impl Program {
    /// Compile `v`, registering each column it reads in `cols`.
    fn compile(v: &Value, cols: &mut Vec<String>, part: &str) -> Result<Self, String> {
        let mut ops = Vec::new();
        emit(v, cols, &mut ops, part)?;
        let (mut held, mut depth) = (0usize, 0usize);
        for op in &ops {
            match op {
                Op::Col(_) | Op::Lit(_) | Op::Unknown => held += 1,
                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow | Op::Log => held -= 1,
                Op::Neg | Op::Abs | Op::Exp | Op::Sqrt => {}
            }
            depth = depth.max(held);
        }
        Ok(Self { ops, depth })
    }

    fn holds_unknown(&self) -> bool {
        self.ops.contains(&Op::Unknown)
    }

    /// The part's value and rounding bound at `x` on the row `row`, `stack`
    /// a buffer reused from row to row.
    fn eval(&self, row: &[f64], x: f64, stack: &mut Vec<Bounded>) -> Bounded {
        stack.clear();
        for op in &self.ops {
            let r = match *op {
                Op::Col(i) => Bounded::exact(row[i]),
                Op::Lit(v) => Bounded::exact(v),
                Op::Unknown => Bounded::exact(x),
                Op::Neg | Op::Abs | Op::Exp | Op::Sqrt => {
                    let a = stack.pop().expect("compiled with its operand");
                    unary(*op, a)
                }
                _ => {
                    let b = stack.pop().expect("compiled with its operands");
                    let a = stack.pop().expect("compiled with its operands");
                    binary(*op, a, b)
                }
            };
            stack.push(r);
        }
        stack.pop().expect("a compiled part leaves one value")
    }
}

/// `v` with the bound `propagated` from its operands, plus its own rounding.
fn rounded(v: f64, propagated: f64, libm: bool) -> Bounded {
    let own = if libm {
        (U + LIBM) * v.abs()
    } else {
        U * v.abs()
    };
    Bounded::new(v, propagated + own + ETA)
}

/// A bound on `|ln a' − ln a|` over `|a' − a| <= e`, `a > 0`.
fn ln_spread(a: f64, e: f64) -> f64 {
    if e == 0.0 {
        0.0
    } else if e < a {
        -(-e / a).ln_1p()
    } else {
        f64::INFINITY
    }
}

fn unary(op: Op, a: Bounded) -> Bounded {
    let (x, e) = (a.value, a.error);
    match op {
        Op::Neg => Bounded::new(-x, e),
        Op::Abs => Bounded::new(x.abs(), e),
        Op::Exp => {
            let v = x.exp();
            rounded(v, v.abs() * e.exp_m1(), true)
        }
        Op::Sqrt => {
            let v = x.sqrt();
            let spread = if e == 0.0 {
                0.0
            } else {
                let below = (x - e).max(0.0).sqrt();
                if v + below > 0.0 {
                    e / (v + below)
                } else {
                    e.sqrt()
                }
            };
            rounded(v, spread, false)
        }
        _ => unreachable!("not a unary node"),
    }
}

fn binary(op: Op, a: Bounded, b: Bounded) -> Bounded {
    let (x, ex, y, ey) = (a.value, a.error, b.value, b.error);
    match op {
        Op::Add => rounded(x + y, ex + ey, false),
        Op::Sub => rounded(x - y, ex + ey, false),
        Op::Mul => rounded(x * y, x.abs() * ey + y.abs() * ex + ex * ey, false),
        Op::Div => {
            let v = x / y;
            if y.abs() > ey {
                rounded(v, (ex + (x.abs() / y.abs()) * ey) / (y.abs() - ey), false)
            } else {
                Bounded::new(v, f64::INFINITY)
            }
        }
        Op::Log => {
            // Polars' `log(base)` is `x.ln() / base.ln()`, as here.
            let num = rounded(x.ln(), ln_spread(x, ex), true);
            let den = rounded(y.ln(), ln_spread(y, ey), true);
            let mut q = binary(Op::Div, num, den);
            q.value = x.log(y);
            q
        }
        Op::Pow => {
            let v = x.powf(y);
            let spread = if ex == 0.0 && ey == 0.0 {
                0.0
            } else if (x < 0.0 && ey > 0.0) || x == 0.0 {
                // A negative base with an uncertain exponent may have no
                // real power; a base of 0 with an uncertain one, no bound.
                f64::INFINITY
            } else {
                let l = ln_spread(x.abs(), ex);
                v.abs() * (y.abs() * l + x.abs().ln().abs() * ey + l * ey).exp_m1()
            };
            rounded(v, spread, true)
        }
        _ => unreachable!("not a binary node"),
    }
}

/// The nodes a part may hold, as a refusal lists them.
fn accepted() -> String {
    let mut heads = vec!["unknown", "col", "lit"];
    heads.extend(BINARY.iter().map(|(n, _)| *n));
    heads.extend(UNARY.iter().map(|(n, _)| *n));
    heads.push("cast");
    heads.join(", ")
}

fn emit(v: &Value, cols: &mut Vec<String>, ops: &mut Vec<Op>, part: &str) -> Result<(), String> {
    let shape = || {
        format!(
            "invert: each node of its {part} must be a list whose first item names its kind, \
             such as [\"col\", \"x\"], but one is {}. Write the node in that form.",
            short(v)
        )
    };
    let Value::Array(items) = v else {
        return Err(shape());
    };
    let Some(Value::String(head)) = items.first() else {
        return Err(shape());
    };
    let args = &items[1..];
    let arity = |n: usize| {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!(
                "invert: {head:?} in its {part} takes {n} argument{}, but it has {}. Give \
                 {head:?} that many.",
                if n == 1 { "" } else { "s" },
                args.len()
            ))
        }
    };
    match head.as_str() {
        "unknown" => {
            arity(0)?;
            ops.push(Op::Unknown);
        }
        "col" => {
            arity(1)?;
            let Value::String(name) = &args[0] else {
                return Err(format!(
                    "invert: \"col\" in its {part} takes a column's name as a string, but it \
                     has {}. Give \"col\" a column name, such as [\"col\", \"x\"].",
                    short(&args[0])
                ));
            };
            let slot = cols.iter().position(|c| c == name).unwrap_or_else(|| {
                cols.push(name.clone());
                cols.len() - 1
            });
            ops.push(Op::Col(slot));
        }
        "lit" => {
            arity(1)?;
            let x = args[0].as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                format!(
                    "invert: a literal in its {part} must be a finite number, but it is {}. \
                     Write the literal as a number.",
                    short(&args[0])
                )
            })?;
            ops.push(Op::Lit(x));
        }
        "cast" => {
            arity(2)?;
            if args[1] != Value::String("Float64".into()) {
                return Err(format!(
                    "invert: a cast in its {part} may only be to \"Float64\", since the solve \
                     reads every value as a double, but it is to {}. Cast to Float64, or \
                     leave the cast out.",
                    short(&args[1])
                ));
            }
            emit(&args[0], cols, ops, part)?;
        }
        name => {
            if let Some((_, op)) = BINARY.iter().find(|(n, _)| *n == name) {
                arity(2)?;
                emit(&args[0], cols, ops, part)?;
                emit(&args[1], cols, ops, part)?;
                ops.push(*op);
            } else if let Some((_, op)) = UNARY.iter().find(|(n, _)| *n == name) {
                arity(1)?;
                emit(&args[0], cols, ops, part)?;
                ops.push(*op);
            } else {
                return Err(format!(
                    "invert: {name:?} in its {part} is not a node the solve can evaluate a row \
                     at a time. It can hold {}. Compute the value as a column first and read \
                     that column.",
                    accepted()
                ));
            }
        }
    }
    Ok(())
}

/// The `invert` node: the function, the target, the bracket, the declared
/// direction and the tolerance, compiled once.
#[derive(Debug, Clone, PartialEq)]
pub struct Invert {
    f: Program,
    equals: Program,
    lo: Program,
    hi: Program,
    increasing: bool,
    tol: RootTolerance,
    detail: bool,
    /// The columns the four parts read, in slot order.
    columns: Vec<String>,
}

/// The fields of the struct `detail` gives, in order.
pub const INVERT_DETAIL_FIELDS: [&str; 6] = ["root", "lo", "hi", "evaluations", "bound", "reason"];

impl Invert {
    /// Read the node from its list form.
    ///
    /// # Errors
    ///
    /// A head other than `invert`, a wrong arity, a part holding a node the
    /// interpreter does not evaluate, `f` without the unknown or another
    /// part with it, a parameter missing, unknown or out of range, or a node
    /// that reads no column.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let items = match v {
            Value::Array(items) if items.first() == Some(&Value::String("invert".into())) => items,
            _ => {
                return Err(format!(
                    "invert: the node must be a list headed \"invert\", but it is {}. Write it \
                     as [\"invert\", f, equals, lo, hi, {{\"increasing\": true}}].",
                    short(v)
                ));
            }
        };
        if items.len() != 6 {
            return Err(format!(
                "invert: the node takes 5 arguments (f, equals, lo, hi and its parameters), but \
                 it has {}. Write it as [\"invert\", f, equals, lo, hi, {{\"increasing\": true}}].",
                items.len() - 1
            ));
        }
        let mut columns = Vec::new();
        let f = Program::compile(&items[1], &mut columns, "function")?;
        let equals = Program::compile(&items[2], &mut columns, "target (equals)")?;
        let lo = Program::compile(&items[3], &mut columns, "bracket's lower end")?;
        let hi = Program::compile(&items[4], &mut columns, "bracket's upper end")?;
        if !f.holds_unknown() {
            return Err(
                "invert: its function does not hold the unknown, so there is nothing to \
                 solve for. Write the function in terms of the unknown."
                    .into(),
            );
        }
        for (p, part) in [
            (&equals, "target (equals)"),
            (&lo, "bracket's lower end"),
            (&hi, "bracket's upper end"),
        ] {
            if p.holds_unknown() {
                return Err(format!(
                    "invert: its {part} holds the unknown, but only the function may: the \
                     target and the bracket are values of the row. Write the {part} without it."
                ));
            }
        }
        if columns.is_empty() {
            return Err(
                "invert: the node reads no column, so it has no rows to solve on. Read \
                 at least one column in the function, the target or the bracket."
                    .into(),
            );
        }
        let Value::Object(params) = &items[5] else {
            return Err(format!(
                "invert: its parameters must be an object, such as {{\"increasing\": true}}, but \
                 they are {}. Give them as an object.",
                short(&items[5])
            ));
        };
        let (mut increasing, mut detail) = (None, false);
        let (mut xtol, mut rtol) = (rheon_core::DEFAULT_XTOL, rheon_core::MIN_RTOL);
        for (k, val) in params {
            let flag = || {
                val.as_bool().ok_or_else(|| {
                    format!(
                        "invert: {k} must be true or false, but it is {}. Set {k} to true or \
                         false.",
                        short(val)
                    )
                })
            };
            let number = || {
                val.as_f64().ok_or_else(|| {
                    format!(
                        "invert: {k} must be a number, but it is {}. Set {k} to a number.",
                        short(val)
                    )
                })
            };
            match k.as_str() {
                "increasing" => increasing = Some(flag()?),
                "detail" => detail = flag()?,
                "xtol" => xtol = number()?,
                "rtol" => rtol = number()?,
                other => {
                    return Err(format!(
                        "invert: {other:?} is not one of its parameters, which are increasing, \
                         xtol, rtol and detail. Remove {other:?}, or correct its name."
                    ));
                }
            }
        }
        let increasing = increasing.ok_or_else(|| {
            "invert: increasing is not set. Set it to true where the function rises with the \
             unknown over the bracket and false where it falls: the direction names the side a \
             target out of range lies on, and catches a function that is not as declared."
                .to_string()
        })?;
        let tol = RootTolerance::new(xtol, rtol).map_err(|e| format!("invert: {e}."))?;
        Ok(Self {
            f,
            equals,
            lo,
            hi,
            increasing,
            tol,
            detail,
            columns,
        })
    }

    /// Read the node from its text.
    ///
    /// # Errors
    ///
    /// As [`Invert::from_json`], and text that is not JSON.
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| {
            format!("invert: the text is not JSON ({e}). Write the node as a JSON list.")
        })?;
        Self::from_json(&v)
    }

    /// The columns the node reads, in the order a row's values are given to
    /// [`Invert::solve_row`].
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Whether the node gives the `detail` struct rather than the root.
    pub fn detail(&self) -> bool {
        self.detail
    }

    /// The solve on one row, its values in [`Invert::columns`]' order (a
    /// null as NaN); `stack` is a buffer reused from row to row.
    pub fn solve_row(&self, row: &[f64], stack: &mut Vec<Bounded>) -> RootReport {
        let missing = RootReport {
            root: Err(Withheld::InputMissing),
            lo: f64::NAN,
            hi: f64::NAN,
            evaluations: 0,
            bound: 0,
        };
        if !row.iter().all(|&v| usable(v)) {
            return missing;
        }
        let target = self.equals.eval(row, f64::NAN, stack);
        let lo = self.lo.eval(row, f64::NAN, stack).value;
        let hi = self.hi.eval(row, f64::NAN, stack).value;
        if !(usable(target.value) && usable(lo) && usable(hi)) {
            return missing;
        }
        let f = |x: f64| {
            let y = self.f.eval(row, x, stack);
            rounded(y.value - target.value, y.error + target.error, false)
        };
        find_root(f, lo, hi, self.increasing, &self.tol)
    }

    /// The output's dtype: `Float64`, or the `detail` struct.
    pub fn dtype(&self) -> DataType {
        if !self.detail {
            return DataType::Float64;
        }
        let types = [
            DataType::Float64,
            DataType::Float64,
            DataType::Float64,
            DataType::UInt32,
            DataType::UInt32,
            DataType::String,
        ];
        DataType::Struct(
            INVERT_DETAIL_FIELDS
                .iter()
                .zip(types)
                .map(|(n, t)| Field::new((*n).into(), t))
                .collect(),
        )
    }

    /// The solve over columns of equal length (or of length 1, broadcast),
    /// in [`Invert::columns`]' order, named `name`.
    ///
    /// # Errors
    ///
    /// A column that does not cast to `Float64`, lengths that do not agree,
    /// or a row whose solve took its whole bound without meeting the
    /// tolerance.
    pub fn apply(&self, name: PlSmallStr, cols: &[Column]) -> PolarsResult<Column> {
        let n = cols.iter().map(Column::len).max().unwrap_or(0);
        let mut values = Vec::with_capacity(cols.len());
        for c in cols {
            if c.len() != n && c.len() != 1 {
                polars_bail!(ShapeMismatch: "invert: its columns have {} and {n} rows. Give \
                    it columns of one length.", c.len());
            }
            let c = c.cast(&DataType::Float64)?;
            let v: Vec<f64> = c.f64()?.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            values.push(v);
        }
        let mut stack = Vec::with_capacity(
            [&self.f, &self.equals, &self.lo, &self.hi]
                .iter()
                .map(|p| p.depth)
                .max()
                .unwrap_or(0),
        );
        let mut row = vec![0.0; cols.len()];
        let mut reports = Vec::with_capacity(n);
        for i in 0..n {
            for (slot, v) in row.iter_mut().zip(&values) {
                *slot = if v.len() == 1 { v[0] } else { v[i] };
            }
            let r = self.solve_row(&row, &mut stack);
            if r.root == Err(Withheld::BoundReached) {
                polars_bail!(ComputeError: "invert: a solve took all {} evaluations its \
                    bracket allows without meeting the tolerance, on the row whose values ({}) \
                    are {row:?}: a defect of the solver, which the method rules out, not of the \
                    data. Send those values with a report of the defect.", r.bound, self.columns.join(", "));
            }
            reports.push(r);
        }
        let roots: Float64Chunked = reports.iter().map(|r| r.root.ok()).collect();
        if !self.detail {
            return Ok(roots.with_name(name).into_column());
        }
        let end = |x: f64| (!x.is_nan()).then_some(x);
        let lo: Float64Chunked = reports.iter().map(|r| end(r.lo)).collect();
        let hi: Float64Chunked = reports.iter().map(|r| end(r.hi)).collect();
        let evals: UInt32Chunked = reports.iter().map(|r| Some(r.evaluations)).collect();
        let bound: UInt32Chunked = reports.iter().map(|r| Some(r.bound)).collect();
        let reason: StringChunked = reports
            .iter()
            .map(|r| r.root.err().map(Withheld::name))
            .collect();
        let fields = [
            roots.into_series(),
            lo.into_series(),
            hi.into_series(),
            evals.into_series(),
            bound.into_series(),
            reason.into_series(),
        ];
        let fields: Vec<Series> = fields
            .into_iter()
            .zip(INVERT_DETAIL_FIELDS)
            .map(|(s, f)| s.with_name(f.into()))
            .collect();
        Ok(StructChunked::from_series(name, n, fields.iter())?.into_column())
    }

    /// The node as one Polars expression: the columns it reads, cast to
    /// `Float64`, and the solve over them as an element-wise function, so a
    /// streaming plan may hand it any slice of the rows.
    pub fn to_expr(&self) -> Expr {
        let inputs: Vec<Expr> = self.columns.iter().map(|c| col(c.as_str())).collect();
        let me = Arc::new(self.clone());
        let dtype = self.dtype();
        map_multiple(
            move |cols: &mut [Column]| {
                let name = cols.first().map_or(PlSmallStr::EMPTY, |c| c.name().clone());
                me.apply(name, cols)
            },
            inputs,
            move |_, fields: &[Field]| {
                let name = fields
                    .first()
                    .map_or(PlSmallStr::EMPTY, |f| f.name().clone());
                Ok(Field::new(name, dtype.clone()))
            },
        )
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    match s.char_indices().nth(60) {
        Some((cut, _)) => format!("{}...", &s[..cut]),
        None => s,
    }
}

#[cfg(test)]
mod tests;
