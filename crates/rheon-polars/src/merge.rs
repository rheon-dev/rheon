//! An N-way merge of streams by their clock (docs/PLAN.md task 541).
//!
//! `rp.stream.merge` interleaves N inputs, each in clock order, into one
//! stream in clock order, every row kept, each tagged with the input it came
//! from. It is what a user with quotes, trades and news, each a stream of its
//! own, needs before one bank reads them all: Polars' `merge_sorted` takes
//! two frames at a time, so N streams need N - 1 chained merges, and a sort
//! of their concatenation holds every row at once.
//!
//! # The order
//!
//! A row's place is its key `(restarts, clock, input)`: the number of times
//! its input started again under `restart_after_step_back` (0 without one),
//! its clock, and its input's position in the call. Rows of one input keep
//! their own order, so a tie on the whole key is broken by it. That is
//! exactly the order of `pl.concat(inputs).sort([clock, input, row],
//! maintain_order=True)` and of `merge_sorted` chained from the first input
//! (whose left frame wins a tie).
//!
//! # Memory: one chunk per input
//!
//! Each input is in clock order, so its first unmerged row (its head) is the
//! least of its rows still to come, and the least head is the least row of
//! every input still to come: it can be emitted. The merge keeps a heap of
//! the heads, pops the least, and moves that input's head on. When an
//! input's chunk is used up the merge stops and asks for that input's next
//! chunk ([`Merge::pull`] returns its index), since its next row may be the
//! least. So it holds one chunk of each input and nothing more, however long
//! the streams are, and the output is a function of the rows alone, never of
//! where the chunks fall: chunk invariance by construction.
//!
//! # Disorder
//!
//! The heap's argument needs each input in clock order across all its rows:
//! an input that steps back puts a later row of its own ahead of rows of the
//! other inputs the merge has already emitted. So each input's clock is read
//! as a bank reads a group's: a step back is refused, naming the input and
//! its row, or, under `restart_after_step_back`, one larger than it starts
//! the input again -- its `restarts` count goes up, and its later rows sort
//! after every row of the earlier count, as a new day's rows after the last
//! day's. And the merged rows are read as a bank grouped by `by` will read
//! them (one group without `by`): a step back in a group's merged rows is
//! refused by the bank's rule too. Within one run of sorted inputs it never
//! fires; it catches a resumed run whose inputs were cut at different
//! clocks, and an input that did not start again where the others did.
//!
//! # Resuming
//!
//! The state is each input's last row taken (its clock and restarts), the
//! merged rows' last clock in each group, and the rows held at a tie: a run
//! that saves its state at the end of its inputs cannot know whether the
//! next part of an earlier input brings rows at the last clock, which would
//! come first, so it holds the rows at that last clock and the next run
//! emits them in their place. Under a slice nothing is held: the state is
//! the state after the rows returned, and each input resumes after its rows
//! the slice returned.

mod state;
#[cfg(test)]
mod tests;

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};

use polars::prelude::*;
use rheon_core::FixedState;

use crate::bank::CLOCK_ORDER_HOW_TO;
use crate::figures::{Round, three_figures};

pub(crate) const WHO: &str = "rp.stream.merge";

/// The call's settings, which every run of a state must share.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeConfig {
    /// The clock column, in every input.
    pub clock: String,
    /// The column added to name each row's input.
    pub source: String,
    /// The column a bank downstream groups by: the merged rows of each of its
    /// groups are checked as that bank checks them.
    pub by: Option<String>,
    /// `restart_after_step_back` in the clock's units -- integer nanoseconds
    /// on a temporal clock -- or `None`, which refuses every step back.
    pub restart: Option<f64>,
    /// `restart_after_step_back` as the caller wrote it, for the messages.
    pub restart_text: Option<String>,
}

/// How a clock column's values are ordered and stepped.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// An integer column, compared and stepped exactly.
    Int,
    /// A float column, its key the order-preserving bits of the value.
    Float,
    /// A temporal column in its physical integer; `unit_ns` nanoseconds a
    /// step, so a step back is measured in nanoseconds.
    Temporal { unit_ns: i64 },
}

impl Mode {
    fn of(dtype: &DataType) -> Option<Self> {
        Some(match dtype {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => Self::Int,
            DataType::Float32 | DataType::Float64 => Self::Float,
            DataType::Datetime(u, _) | DataType::Duration(u) => Self::Temporal {
                unit_ns: match u {
                    TimeUnit::Nanoseconds => 1,
                    TimeUnit::Microseconds => 1_000,
                    TimeUnit::Milliseconds => 1_000_000,
                },
            },
            DataType::Date => Self::Temporal {
                unit_ns: 86_400 * 1_000_000_000,
            },
            _ => return None,
        })
    }
}

/// A float's key: its bits, with the magnitude's flipped below zero, so the
/// keys' integer order is the values' order. `-0.0` is `0.0` first, as the
/// two compare equal in a sort.
pub(crate) fn float_key(x: f64) -> i64 {
    let x = if x == 0.0 { 0.0 } else { x };
    let b = x.to_bits() as i64;
    b ^ ((((b >> 63) as u64) >> 1) as i64)
}

/// The value [`float_key`] keyed.
pub(crate) fn float_of(k: i64) -> f64 {
    f64::from_bits((k ^ ((((k >> 63) as u64) >> 1) as i64)) as u64)
}

/// A step back, measured: whether the rule refuses it, and its size as a
/// message prints it.
struct Back {
    refused: bool,
    text: String,
}

/// One chunk of an input (or the rows a state held), with each row's key.
struct Block {
    df: DataFrame,
    keys: Vec<i64>,
    /// Each row's group, under `by`; empty without it.
    gids: Vec<u32>,
    /// The rows that start the input again, ascending.
    bumps: Vec<usize>,
    /// The input's row number of row 0, or `None` for rows a state held.
    first_row: Option<u64>,
    /// The next row to emit, its restarts, and the next bump at or past it.
    pos: usize,
    restarts: u64,
    next_bump: usize,
}

impl Block {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn head(&self) -> (u64, i64) {
        (self.restarts, self.keys[self.pos])
    }

    fn last(&self) -> (u64, i64) {
        let more = (self.bumps.len() - self.next_bump) as u64;
        (self.restarts + more, self.keys[self.len() - 1])
    }

    /// Past the head; `false` once the block is used up.
    fn advance(&mut self) -> bool {
        self.pos += 1;
        if self.bumps.get(self.next_bump) == Some(&self.pos) {
            self.restarts += 1;
            self.next_bump += 1;
        }
        self.pos < self.len()
    }
}

/// One input stream.
struct Input {
    name: String,
    schema: Schema,
    queue: VecDeque<Block>,
    ended: bool,
    /// The last row read, for the order check of the next: key, restarts.
    read: Option<(i64, u64)>,
    /// The last row taken -- emitted, or held at the end -- which the state
    /// saves: key, restarts.
    taken: Option<(i64, u64)>,
    /// Rows of this run's input read so far, for the messages' row numbers.
    rows_read: u64,
}

impl Input {
    fn head(&self) -> Option<(u64, i64)> {
        self.queue.front().map(Block::head)
    }
}

/// A run of output rows from one block, as a pull collects them.
struct Part {
    input: usize,
    df: DataFrame,
    start: usize,
    end: usize,
}

/// The merge: fed chunks of each input, it gives the merged rows in order.
pub struct Merge {
    cfg: MergeConfig,
    mode: Mode,
    clock_dtype: DataType,
    inputs: Vec<Input>,
    out_schema: Schema,
    /// Group ids under `by`, by the key's text; the null key apart.
    gid_of: HashMap<String, u32, FixedState>,
    null_gid: Option<u32>,
    /// Each group's key, by id: the text, or `None` for the null key.
    group_keys: Vec<Option<String>>,
    /// The merged rows' last clock key in each group (one group without
    /// `by`).
    group_last: Vec<Option<i64>>,
    /// The heads of the inputs whose queue holds a row: (restarts, key,
    /// input), least first.
    heap: BinaryHeap<Reverse<(u64, i64, usize)>>,
    /// Set once every input ended and a run that saves its state holds the
    /// rows at the last clock: the state then holds every row still queued.
    held_end: bool,
    /// The greatest (restarts, key) read, of every input.
    max_read: Option<(u64, i64)>,
    /// The inputs' names, by index, for the `source` column.
    names: StringChunked,
}

/// A column a dtype refusal names, in Polars' words.
fn dtype_text(dtype: &DataType) -> String {
    crate::arrow::dtype_name(dtype)
}

impl Merge {
    /// A merge of inputs named `names`, of the columns `schemas` declare.
    ///
    /// # Errors
    ///
    /// No input or a name twice; an input without the clock or `by` column
    /// (`ColumnNotFound`); a clock that is neither numeric nor temporal, the
    /// inputs' clocks of different dtypes, or a column of one name with two
    /// dtypes (each a wrong type); `source` already an input's column.
    pub fn new(names: Vec<String>, schemas: Vec<Schema>, cfg: MergeConfig) -> PolarsResult<Self> {
        if names.is_empty() {
            polars_bail!(ComputeError:
                "{WHO}: no input is given. Pass the frames to merge, such as \
                 rp.stream.merge(quotes, trades, clock=\"ts\")."
            );
        }
        if names.len() != schemas.len() {
            polars_bail!(ComputeError:
                "{WHO}: {} names for {} inputs. Name each input once.",
                names.len(), schemas.len()
            );
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(dup) = names.iter().find(|n| !seen.insert(n.as_str())) {
            polars_bail!(ComputeError:
                "{WHO}: the input name {dup:?} is given twice, and each input's rows carry its \
                 name in the {:?} column. Name each input once.",
                cfg.source
            );
        }
        if let Some(r) = cfg.restart
            && (r.is_nan() || r < 0.0)
        {
            polars_bail!(ComputeError:
                "{WHO}: restart_after_step_back must be 0 or more, but it is {}. Set it to the \
                 smallest step back that starts an input again, or leave it out to refuse every \
                 step back.",
                cfg.restart_text.clone().unwrap_or_else(|| r.to_string())
            );
        }
        for (name, schema) in names.iter().zip(&schemas) {
            for (role, col) in [("clock", Some(&cfg.clock)), ("by", cfg.by.as_ref())] {
                if let Some(col) = col
                    && schema.get(col).is_none()
                {
                    let have: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
                    return Err(PolarsError::ColumnNotFound(
                        format!(
                            "{WHO}: input {name:?} has no {role} column {col:?}, only {have:?}. \
                             Give every input the {role} column, renamed if need be."
                        )
                        .into(),
                    ));
                }
            }
            if schema.get(&cfg.source).is_some() {
                polars_bail!(Duplicate:
                    "{WHO}: input {name:?} already has a column named {:?}, the column the merge \
                     adds to name each row's input. Pass another source=, or rename that column.",
                    cfg.source
                );
            }
        }
        let clock_dtype = schemas[0].get(&cfg.clock).expect("checked above").clone();
        let Some(mode) = Mode::of(&clock_dtype) else {
            return Err(crate::wrong_type(format!(
                "{WHO}: clock column {:?} of input {:?} is {}, which is no clock: a clock is an \
                 integer, a float, a Datetime, a Date or a Duration. Cast it, such as with \
                 pl.col({:?}).cast(pl.Float64) or pl.col({:?}).str.to_datetime().",
                cfg.clock,
                names[0],
                dtype_text(&clock_dtype),
                cfg.clock,
                cfg.clock
            )));
        };
        for (name, schema) in names.iter().zip(&schemas).skip(1) {
            let d = schema.get(&cfg.clock).expect("checked above");
            if d != &clock_dtype {
                return Err(crate::wrong_type(format!(
                    "{WHO}: the clocks are not of one kind: input {:?}'s clock column {:?} is {} \
                     and input {name:?}'s is {}. Cast every input's clock to one dtype, with one \
                     unit and one time zone, so that their values compare.",
                    names[0],
                    cfg.clock,
                    dtype_text(&clock_dtype),
                    dtype_text(d)
                )));
            }
        }
        // The columns unified by name, in the order they first appear.
        let mut out_schema = Schema::default();
        let mut first_in: HashMap<String, usize> = HashMap::new();
        for (i, schema) in schemas.iter().enumerate() {
            for (col, dtype) in schema.iter() {
                match out_schema.get(col) {
                    None => {
                        out_schema.insert(col.clone(), dtype.clone());
                        first_in.insert(col.to_string(), i);
                    }
                    Some(have) if have != dtype => {
                        return Err(crate::wrong_type(format!(
                            "{WHO}: column {:?} is {} in input {:?} and {} in input {:?}, and a \
                             merged column holds one dtype. Cast it to one dtype in every input, \
                             or rename it in one.",
                            col.as_str(),
                            dtype_text(have),
                            names[first_in[col.as_str()]],
                            dtype_text(dtype),
                            names[i]
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
        out_schema.insert(cfg.source.as_str().into(), DataType::String);
        let inputs: Vec<Input> = names
            .into_iter()
            .zip(schemas)
            .map(|(name, schema)| Input {
                name,
                schema,
                queue: VecDeque::new(),
                ended: false,
                read: None,
                taken: None,
                rows_read: 0,
            })
            .collect();
        let grouped = cfg.by.is_some();
        let names = StringChunked::from_iter_values(
            "".into(),
            inputs.iter().map(|i: &Input| i.name.as_str()),
        );
        Ok(Self {
            cfg,
            mode,
            clock_dtype,
            inputs,
            out_schema,
            gid_of: HashMap::default(),
            null_gid: None,
            group_keys: if grouped {
                Vec::new()
            } else {
                vec![Some(String::new())]
            },
            group_last: if grouped { Vec::new() } else { vec![None] },
            heap: BinaryHeap::new(),
            held_end: false,
            max_read: None,
            names,
        })
    }

    /// The merged rows' columns: every input's, unified by name in the order
    /// they first appear, then `source`.
    pub fn schema(&self) -> &Schema {
        &self.out_schema
    }

    /// How many inputs.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    /// Whether there are no inputs (never: [`Self::new`] refuses none).
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }

    /// A step back from key `prev` to key `now`, measured and judged by
    /// `restart_after_step_back`'s rule: refused when no rule is set or the
    /// step is no larger than it, a restart past it.
    fn back(&self, prev: i64, now: i64) -> Back {
        let r = self.cfg.restart;
        let late_int = |back: i128| match r {
            None => true,
            Some(r) => matches!(
                rheon_core::cmp_int_f64(back, r),
                Some(Ordering::Less | Ordering::Equal)
            ),
        };
        match self.mode {
            Mode::Int => {
                let back = i128::from(prev) - i128::from(now);
                Back {
                    refused: late_int(back),
                    text: back.to_string(),
                }
            }
            Mode::Temporal { unit_ns } => {
                let back = (i128::from(prev) - i128::from(now)) * i128::from(unit_ns);
                Back {
                    refused: late_int(back),
                    text: i64::try_from(back)
                        .map_or_else(|_| format!("{back} ns"), crate::span::format_duration),
                }
            }
            Mode::Float => {
                let back = float_of(prev) - float_of(now);
                Back {
                    refused: r.is_none_or(|r| back <= r),
                    text: three_figures(back, Round::Down),
                }
            }
        }
    }

    /// The rule's half of a step-back message: why it was refused.
    fn why(&self) -> String {
        match &self.cfg.restart_text {
            None => "and restart_after_step_back is not set, so every step back is refused".into(),
            Some(r) => format!("no more than restart_after_step_back = {r}, so it is refused"),
        }
    }

    /// The rows of chunk `df` of input `i`'s clock as keys, each checked
    /// against the row before it; with the rows that start the input again.
    fn keys(&self, i: usize, df: &DataFrame) -> PolarsResult<(Vec<i64>, Vec<usize>)> {
        let col = df.column(&self.cfg.clock)?.as_materialized_series();
        let name = &self.inputs[i].name;
        let base = self.inputs[i].rows_read;
        let bad = |row: usize, what: &str| {
            polars_err!(ComputeError:
                "{WHO}: input {name:?}: clock column {:?} is {what} at row {} (counting from 0), \
                 which has no place in clock order. Drop the rows with no clock, or fill them.",
                self.cfg.clock, base + row as u64
            )
        };
        let keys: Vec<i64> = match self.mode {
            Mode::Float => {
                let c = col.cast(&DataType::Float64)?;
                c.f64()?
                    .iter()
                    .enumerate()
                    .map(|(row, v)| match v {
                        None => Err(bad(row, "null")),
                        Some(x) if x.is_nan() => Err(bad(row, "NaN")),
                        Some(x) if x.is_infinite() => Err(bad(row, "infinite")),
                        Some(x) => Ok(float_key(x)),
                    })
                    .collect::<PolarsResult<_>>()?
            }
            Mode::Int if col.dtype() == &DataType::UInt64 => col
                .u64()?
                .iter()
                .enumerate()
                .map(|(row, v)| match v {
                    None => Err(bad(row, "null")),
                    Some(x) => i64::try_from(x).map_err(|_| {
                        polars_err!(ComputeError:
                            "{WHO}: input {name:?}: clock column {:?} is {x} at row {} (counting \
                             from 0), past the largest Int64, {}, which an integer clock is held \
                             in. Cast the clock to Float64, or subtract an origin from it.",
                            self.cfg.clock, base + row as u64, i64::MAX
                        )
                    }),
                })
                .collect::<PolarsResult<_>>()?,
            Mode::Int | Mode::Temporal { .. } => {
                let c = col.to_physical_repr().cast(&DataType::Int64)?;
                c.i64()?
                    .iter()
                    .enumerate()
                    .map(|(row, v)| v.ok_or_else(|| bad(row, "null")))
                    .collect::<PolarsResult<_>>()?
            }
        };
        let mut bumps = Vec::new();
        let mut prev = self.inputs[i].read.map(|(k, _)| k);
        for (row, &k) in keys.iter().enumerate() {
            if let Some(p) = prev
                && k < p
            {
                let back = self.back(p, k);
                if back.refused {
                    polars_bail!(ComputeError:
                        "{WHO}: input {name:?}: clock column {:?} steps back by {} at row {} \
                         (counting from 0), {}, and the merge holds one chunk of each input, so \
                         it needs each in clock order across all its rows, its groups included. \
                         Sort the input by the clock, or set restart_after_step_back below \
                         {} to start the input again at such a step back (0 starts it again at \
                         every one). See {CLOCK_ORDER_HOW_TO} for which to choose.",
                        self.cfg.clock, back.text, base + row as u64, self.why(), back.text
                    );
                }
                bumps.push(row);
            }
            prev = Some(k);
        }
        Ok((keys, bumps))
    }

    /// Each row's group id under `by`, new groups given the next ids.
    fn gids(&mut self, df: &DataFrame) -> PolarsResult<Vec<u32>> {
        let Some(by) = &self.cfg.by else {
            return Ok(Vec::new());
        };
        let text = crate::arrow::key_text(df.column(by)?.as_materialized_series())?;
        let text = text.str()?;
        let mut out = Vec::with_capacity(text.len());
        for v in text.iter() {
            let id = match v {
                Some(s) => match self.gid_of.get(s) {
                    Some(&id) => id,
                    None => {
                        let id = self.new_group(Some(s.to_owned()));
                        self.gid_of.insert(s.to_owned(), id);
                        id
                    }
                },
                None => match self.null_gid {
                    Some(id) => id,
                    None => {
                        let id = self.new_group(None);
                        self.null_gid = Some(id);
                        id
                    }
                },
            };
            out.push(id);
        }
        Ok(out)
    }

    fn new_group(&mut self, key: Option<String>) -> u32 {
        let id = u32::try_from(self.group_keys.len()).expect("fewer than 2^32 groups");
        self.group_keys.push(key);
        self.group_last.push(None);
        id
    }

    /// The next chunk of input `i`, in its order after the last.
    ///
    /// # Errors
    ///
    /// A chunk of other columns than the input's; a clock that is null, not
    /// finite, or steps back where the rule refuses it, naming the row.
    pub fn push(&mut self, i: usize, df: DataFrame) -> PolarsResult<()> {
        let input = &self.inputs[i];
        if input.ended {
            polars_bail!(ComputeError:
                "{WHO}: input {:?} was fed after its end. Give each input its chunks before its end.",
                input.name
            );
        }
        if df.schema().as_ref() != &input.schema {
            polars_bail!(SchemaMismatch:
                "{WHO}: a chunk of input {:?} has the columns {:?}, not the input's {:?}. Give \
                 each input chunks of its own columns.",
                input.name, df.schema(), input.schema
            );
        }
        if df.height() == 0 {
            return Ok(());
        }
        let (keys, bumps) = self.keys(i, &df)?;
        let gids = self.gids(&df)?;
        let input = &mut self.inputs[i];
        let restarts = input.read.map_or(0, |(_, r)| r);
        let first_bumps = usize::from(bumps.first() == Some(&0));
        let block = Block {
            first_row: Some(input.rows_read),
            pos: 0,
            restarts: restarts + first_bumps as u64,
            next_bump: first_bumps,
            df,
            keys,
            gids,
            bumps,
        };
        let last = block.last();
        input.rows_read += block.len() as u64;
        input.read = Some((last.1, last.0));
        let was_empty = input.queue.is_empty();
        input.queue.push_back(block);
        self.max_read = self.max_read.max(Some(last));
        if was_empty {
            let (r, k) = input.head().expect("a block was pushed");
            self.heap.push(Reverse((r, k, i)));
        }
        Ok(())
    }

    /// Input `i` has no more rows in this run.
    pub fn end(&mut self, i: usize) {
        self.inputs[i].ended = true;
    }

    /// The first input whose rows are needed before any more can be merged:
    /// one not ended whose queue is empty.
    fn waiting(&self) -> Option<usize> {
        self.inputs
            .iter()
            .position(|inp| !inp.ended && inp.queue.is_empty())
    }

    /// The refusal of a step back in a group's merged rows, at row `row` of
    /// input `i` (`None` for a row a state held).
    #[cold]
    fn group_refusal(&self, i: usize, gid: usize, back: &Back, row: Option<u64>) -> PolarsError {
        let at = match row {
            Some(r) => format!(
                "input {:?}'s row {r} (counting from 0)",
                self.inputs[i].name
            ),
            None => format!("a row of input {:?} the state held", self.inputs[i].name),
        };
        let whose = match (&self.cfg.by, &self.group_keys[gid]) {
            (None, _) => "the merged rows step".to_string(),
            (Some(by), Some(k)) => format!("the merged rows of {by} = {k:?} step"),
            (Some(by), None) => format!("the merged rows of the null {by} step"),
        };
        polars_err!(ComputeError:
            "{WHO}: {whose} back by {} at {at}, {}, as a bank reading the merged rows would \
             refuse it: a run resumed on inputs cut at different clocks, or an input that did not \
             start again where the others did. Cut \
             every input at the same clock, start every input again at the same step back, or \
             set restart_after_step_back below {}. See {CLOCK_ORDER_HOW_TO} for which to \
             choose.",
            back.text, self.why(), back.text
        )
    }

    /// Merged rows: at most `limit`, stopping where an input's next chunk is
    /// needed, whose index comes back beside them. `None` beside fewer than
    /// `limit` rows means every input has ended and its rows are out -- but,
    /// with `hold`, the rows at the last clock, which a state saved then
    /// holds for the next run; beside `limit` rows, call again for more. A
    /// caller that takes a chunk's rows a pull holds one chunk of output
    /// beside one chunk of each input.
    ///
    /// # Errors
    ///
    /// A step back in a group's merged rows that the rule refuses.
    pub fn pull(
        &mut self,
        limit: Option<usize>,
        hold: bool,
    ) -> PolarsResult<(DataFrame, Option<usize>)> {
        let mut parts: Vec<Part> = Vec::new();
        // Each input's part for its front block in this pull.
        let mut open: Vec<Option<usize>> = vec![None; self.inputs.len()];
        // Each output row: its part and its row there.
        let mut picks: Vec<(u32, u32)> = Vec::new();
        // An input with no row queued and more to come holds every other
        // back: its next row may be the least.
        let mut want = self.waiting();
        let all_ended = self.inputs.iter().all(|inp| inp.ended);
        // A run that saves its state at its inputs' end holds the rows at
        // the last clock read: the next part of an earlier input may bring
        // rows at that clock, which come first. So a row at the last clock
        // read so far waits until a later one is read, or every input ends.
        let hold_at = if hold { self.max_read } else { None };
        let budget = limit.unwrap_or(usize::MAX);
        'merge: while want.is_none() && picks.len() < budget {
            let Some(&Reverse((r, k, i))) = self.heap.peek() else {
                break;
            };
            if hold_at.is_some_and(|h| (r, k) >= h) {
                // Every row queued is at the last clock read: read on, from
                // the first input not ended (beside the chunk it holds), or
                // hold them.
                match self.inputs.iter().position(|inp| !inp.ended) {
                    Some(j) => want = Some(j),
                    None => self.held_end = true,
                }
                break;
            }
            self.heap.pop();
            // The least head of the other inputs, which stays put while input
            // `i`'s rows go out below it.
            let bound = self.heap.peek().map(|&Reverse(b)| b);
            // Input `i`'s rows, block by block, while its head stays the least.
            loop {
                let input = &mut self.inputs[i];
                let block = input.queue.front_mut().expect("a head on the heap");
                let p = match open[i] {
                    Some(p) => p,
                    None => {
                        parts.push(Part {
                            input: i,
                            df: block.df.clone(),
                            start: block.pos,
                            end: block.pos,
                        });
                        open[i] = Some(parts.len() - 1);
                        parts.len() - 1
                    }
                };
                let start = parts[p].start;
                let from = block.pos;
                let mut last_out = (0, 0);
                // A step back in a group's merged rows, for the rule to judge.
                let mut stepped: Option<(usize, i64, i64)> = None;
                // Whether the rows stopped inside the block, its head no
                // longer the least.
                let mut stopped = false;
                loop {
                    let row = block.pos;
                    let key = block.keys[row];
                    let gid = block.gids.get(row).map_or(0, |&g| g as usize);
                    let last = &mut self.group_last[gid];
                    if let Some(l) = *last
                        && key < l
                    {
                        stepped = Some((gid, l, key));
                        break;
                    }
                    *last = Some(key);
                    last_out = (key, block.restarts);
                    picks.push((p as u32, (row - start) as u32));
                    if !block.advance() {
                        break;
                    }
                    let head = block.head();
                    if picks.len() >= budget
                        || hold_at.is_some_and(|h| head >= h)
                        || bound.is_some_and(|b| (head.0, head.1, i) > b)
                    {
                        stopped = true;
                        break;
                    }
                }
                let used_up = !stopped && stepped.is_none();
                parts[p].end = block.pos;
                let row_number = block.first_row.map(|f| f + block.pos as u64);
                if block.first_row.is_some() && block.pos > from {
                    input.taken = Some(last_out);
                }
                if let Some((gid, l, key)) = stepped {
                    let back = self.back(l, key);
                    if back.refused {
                        return Err(self.group_refusal(i, gid, &back, row_number));
                    }
                    // A step back past the setting: the group starts again,
                    // as a bank's would; the row goes out on the next turn.
                    self.group_last[gid] = Some(key);
                    continue;
                }
                let input = &mut self.inputs[i];
                if used_up {
                    input.queue.pop_front();
                    open[i] = None;
                }
                let Some(head) = input.head() else {
                    // Used up: its next chunk is needed, unless it ended.
                    if !input.ended {
                        want = Some(i);
                        break 'merge;
                    }
                    break;
                };
                if stopped
                    || picks.len() >= budget
                    || hold_at.is_some_and(|h| head >= h)
                    || bound.is_some_and(|b| (head.0, head.1, i) > b)
                {
                    self.heap.push(Reverse((head.0, head.1, i)));
                    if picks.len() >= budget {
                        break 'merge;
                    }
                    break;
                }
            }
        }
        if want.is_none() && hold && all_ended {
            self.held_end = true;
        }
        let out = self.assemble(parts, &picks)?;
        Ok((out, want))
    }

    /// The output frame of a pull: each part's rows aligned to the output's
    /// columns, then gathered in the order the picks took them.
    fn assemble(&self, parts: Vec<Part>, picks: &[(u32, u32)]) -> PolarsResult<DataFrame> {
        if picks.is_empty() {
            return Ok(DataFrame::empty_with_schema(&self.out_schema));
        }
        let source = PlSmallStr::from(self.cfg.source.as_str());
        let mut base = Vec::with_capacity(parts.len());
        let mut stacked: Option<DataFrame> = None;
        let mut total = 0usize;
        for part in &parts {
            let len = part.end - part.start;
            base.push(total as IdxSize);
            total += len;
            let cols: Vec<Column> = self
                .out_schema
                .iter()
                .filter(|(n, _)| **n != source)
                .map(|(n, d)| match part.df.column(n) {
                    Ok(c) => c.slice(part.start as i64, len),
                    Err(_) => Column::full_null(n.clone(), len, d),
                })
                .collect();
            let df = DataFrame::new(len, cols)?;
            match &mut stacked {
                None => stacked = Some(df),
                Some(s) => {
                    s.vstack_mut(&df)?;
                }
            }
        }
        let stacked = stacked.expect("a pick has a part");
        // In order already when the parts follow one another whole: a pull
        // that took one input's rows alone, the common case of a burst.
        let in_order = picks
            .iter()
            .enumerate()
            .all(|(n, &(p, r))| base[p as usize] + r == n as IdxSize);
        // One copy of the rows, the gather's: the parts are slices of the
        // inputs' chunks, stacked without a copy.
        let mut out = if in_order {
            stacked
        } else {
            let idx = IdxCa::from_vec(
                "".into(),
                picks.iter().map(|&(p, r)| base[p as usize] + r).collect(),
            );
            stacked.take(&idx)?
        };
        // Each row's input's name, gathered from the names, one each.
        let which = IdxCa::from_vec(
            "".into(),
            picks
                .iter()
                .map(|&(p, _)| parts[p as usize].input as IdxSize)
                .collect(),
        );
        let src = self.names.take(&which)?;
        out.with_column(src.with_name(source).into_column())?;
        Ok(out)
    }
}
