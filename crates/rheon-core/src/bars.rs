//! The crossing rule of threshold-triggered bars, over plain values.
//!
//! A bar is one output row that aggregates a run of consecutive input rows.
//! It closes at the row where a running quantity reaches a threshold `θ`;
//! this module decides which row that is, and nothing else. The columns, the
//! groups, the aggregates and the clock live with the stream operator in
//! `rheon-polars`.
//!
//! # The three kinds
//!
//! | kind | a row adds | the bar closes when | after a close |
//! |---|---|---|---|
//! | [`BarKind::Count`] | 1 | `R >= θ` | 0 |
//! | [`BarKind::Sum`] | its quantity `q >= 0` | `R >= θ` | `R − m·θ`, `m = ⌊R/θ⌋` |
//! | [`BarKind::Imbalance`] | its signed quantity | `\|R\| >= θ` | 0 |
//!
//! `R` is the running quantity and `m`, the bar's thresholds, the number of
//! thresholds the closing row crossed: 1 almost always, more for a row
//! larger than `θ`, which stays whole in one bar. Under [`BarKind::Sum`] the
//! part of `R` past the last multiple crossed is carried into the next bar,
//! so the bars are the multiples of `θ` on the running total
//! `S_t = Σ_{s ≤ t} q_s`: row `t` closes a bar when `⌊S_t/θ⌋ > ⌊S_{t−1}/θ⌋`,
//! with `m` the difference. Under [`BarKind::Imbalance`] the sum starts from 0
//! at each bar, López de Prado's imbalance bar, and `m = ⌊|R|/θ⌋`. Under
//! [`BarKind::Count`] a constant threshold makes the two the same.
//!
//! # Exactly
//!
//! `R` is an [`Expansion`]: the decision of which rows make a bar is
//! discrete, and a running total in one double drifts across a multiple of
//! `θ` and moves a row from one bar to the next. A count is an integer.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::exact::{Expansion, FloorDiv};

/// What the running quantity counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarKind {
    /// Rows: tick bars.
    Count,
    /// A quantity of at least 0, the remainder carried: volume and dollar
    /// bars.
    Sum,
    /// A signed quantity, reset at each bar: imbalance bars.
    Imbalance,
}

impl BarKind {
    /// The kind as the call names it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Count => "count",
            Self::Sum => "sum",
            Self::Imbalance => "imbalance",
        }
    }
}

/// The running quantity of one open bar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Running {
    /// The rows of the open bar.
    Count(u64),
    /// The exact running sum: the remainder under [`BarKind::Sum`], the
    /// bar's signed sum under [`BarKind::Imbalance`].
    Exact(Expansion),
}

/// Why a row cannot be counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// A quantity below 0 under [`BarKind::Sum`].
    Negative,
    /// The row crosses `2^53` thresholds or more: a threshold below `2^-53`
    /// of the row.
    TooMany,
}

/// The crossing rule for one group's bars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Crossing {
    kind: BarKind,
    running: Running,
}

impl Crossing {
    /// No row yet.
    pub fn new(kind: BarKind) -> Self {
        let running = match kind {
            BarKind::Count => Running::Count(0),
            BarKind::Sum | BarKind::Imbalance => Running::Exact(Expansion::new()),
        };
        Self { kind, running }
    }

    /// A crossing rule from a saved running quantity, which must be of the
    /// kind's form: a count for [`BarKind::Count`], an exact sum otherwise,
    /// and, under [`BarKind::Sum`], a remainder of at least 0.
    pub fn from_parts(kind: BarKind, running: Running) -> Result<Self, String> {
        match (&running, kind) {
            (Running::Count(_), BarKind::Count) => {}
            (Running::Exact(e), BarKind::Sum) if e.signum() != Ordering::Less => {}
            (Running::Exact(_), BarKind::Imbalance) => {}
            _ => {
                return Err(format!(
                    "the running quantity is not one a {} bar holds",
                    kind.name()
                ));
            }
        }
        Ok(Self { kind, running })
    }

    pub fn kind(&self) -> BarKind {
        self.kind
    }

    pub fn running(&self) -> &Running {
        &self.running
    }

    /// One row of positive weight: `q` is its quantity as one or two
    /// doubles whose exact sum it is (an integer past `2^53` is two), empty
    /// for a missing quantity, which adds 0; ignored under
    /// [`BarKind::Count`]. `theta` is the row's threshold: finite and above
    /// 0, and under [`BarKind::Count`] a whole number. Returns the
    /// thresholds crossed when the row closes the bar.
    ///
    /// # Errors
    ///
    /// [`Refused`], with the state as it was before the row.
    pub fn step(&mut self, q: &[f64], theta: f64) -> Result<Option<u64>, Refused> {
        match (&mut self.running, self.kind) {
            (Running::Count(n), _) => {
                let r = n.saturating_add(1);
                // `θ` is a whole number up to 2^53, so the comparison of
                // `r` as a double is exact below 2^53 and past it `r` is
                // above every `θ`.
                if (r as f64) < theta {
                    *n = r;
                    return Ok(None);
                }
                *n = 0;
                Ok(Some((r as f64 / theta).floor() as u64))
            }
            (Running::Exact(r), BarKind::Sum) => {
                if q.iter().sum::<f64>() < 0.0 {
                    return Err(Refused::Negative);
                }
                for &x in q {
                    r.add(x);
                }
                if r.cmp_f64(theta) == Ordering::Less {
                    return Ok(None);
                }
                match r.floor_div(theta) {
                    Ok(m) => Ok(Some(m)),
                    Err(FloorDiv::TooMany) => {
                        // Exact, and the form is unique, so this is the
                        // state before the row to the bit.
                        for &x in q {
                            r.add(-x);
                        }
                        Err(Refused::TooMany)
                    }
                }
            }
            (Running::Exact(r), _) => {
                for &x in q {
                    r.add(x);
                }
                let past =
                    r.cmp_f64(theta) != Ordering::Less || r.cmp_f64(-theta) != Ordering::Greater;
                if !past {
                    return Ok(None);
                }
                let negative = r.signum() == Ordering::Less;
                if negative {
                    r.negate();
                }
                match r.floor_div(theta) {
                    Ok(m) => {
                        // The reset: the next bar starts from 0.
                        r.clear();
                        Ok(Some(m))
                    }
                    Err(FloorDiv::TooMany) => {
                        if negative {
                            r.negate();
                        }
                        for &x in q {
                            r.add(-x);
                        }
                        Err(Refused::TooMany)
                    }
                }
            }
        }
    }

    /// The remainder carried into the next bar, rounded once to nearest:
    /// under [`BarKind::Sum`] the part of the total past the last multiple
    /// crossed; 0 otherwise after a close.
    pub fn carry(&self) -> f64 {
        match &self.running {
            Running::Count(n) => *n as f64,
            Running::Exact(e) => e.value(),
        }
    }

    /// Start over at 0: a bar cut short by the clock ends the stretch the
    /// carry counts over.
    pub fn reset(&mut self) {
        match &mut self.running {
            Running::Count(n) => *n = 0,
            Running::Exact(e) => e.clear(),
        }
    }

    /// The running quantity's bytes in memory.
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.running {
                Running::Count(_) => 0,
                Running::Exact(e) => e.bytes() - std::mem::size_of::<Vec<f64>>(),
            }
    }
}

#[cfg(test)]
mod tests;
