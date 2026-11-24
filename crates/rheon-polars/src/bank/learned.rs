//! What a loaded bank has learned, as the rows of a resumed input to drop:
//! the command line's `--skip-learned` (docs/PLAN.md task 196, N26), the
//! rule `ModelBank.skip_learned` applies in Python.

use std::collections::HashMap;

use polars::prelude::*;
use rheon_core::ClockValue;

use super::{Bank, GroupKey, group_indices};
use crate::arrow::{ClockArray, chunk_from_frame};
use crate::spec::Spec;

/// Per spec that reads a clock, each group's last clock as the bank held it
/// when this was taken: the position a resumed input is measured against.
/// Taken once, after the load and before the first row, so every chunk is
/// filtered against the same positions and the chunking moves nothing (hard
/// rule 3).
#[derive(Debug, Clone)]
pub struct Learned {
    specs: Vec<Spec>,
    lasts: Vec<Option<HashMap<GroupKey, ClockValue>>>,
    /// Per spec under `group_close = "monotone"`, the largest group key the
    /// bank has read and whether its keys order as numbers: a group below
    /// it was learned and closed, and is gone from `lasts` (review
    /// 2026-11-23, BANK-1).
    marks: Vec<Option<(GroupKey, bool)>>,
}

impl Bank {
    /// What this bank has learned ([`Learned`]).
    ///
    /// # Errors
    ///
    /// When no spec reads a clock: a row-count bank resumes at the next
    /// row, which only the caller knows. And, unless `clock_runs_on`, when a
    /// spec that reads a clock sets `session` or `restart_after_step_back`:
    /// its group's clock may start again lower, so the last clock is not a
    /// place in the stream (`ModelBank.skip_learned` refuses the same; review
    /// fable2, Y2, the user's "refuse unless declared", 2026-11-20).
    pub fn learned(&self, clock_runs_on: bool) -> Result<Learned, String> {
        if !clock_runs_on {
            refuse_a_clock_that_may_start_again(self.specs())?;
        }
        let lasts: Vec<Option<HashMap<GroupKey, ClockValue>>> = self
            .specs()
            .iter()
            .zip(self.last_clocks())
            .map(|(spec, groups)| {
                spec.clock.as_ref().map(|_| {
                    groups
                        .into_iter()
                        .filter_map(|(key, last)| last.map(|c| (key, c)))
                        .collect()
                })
            })
            .collect();
        if lasts.iter().all(Option::is_none) {
            return Err(
                "config: skip_learned = true (--skip-learned) needs a spec that reads a clock, but \
                 no spec does, so the bank has no position to resume from: it goes on at the row \
                 after the last one it read, which only you know. Remove the rows the saved state \
                 has learned from the input, and leave out skip_learned (--skip-learned)."
                    .into(),
            );
        }
        let marks = self.group_marks();
        Ok(Learned {
            specs: self.specs().to_vec(),
            lasts,
            marks,
        })
    }
}

/// The first spec that reads a clock and lets its groups' clocks start again
/// lower, refused with what to do.
fn refuse_a_clock_that_may_start_again(specs: &[Spec]) -> Result<(), String> {
    for spec in specs.iter().filter(|s| s.clock.is_some()) {
        let (setting, how) = if let Some(session) = &spec.session {
            (format!("session = {session:?}"), "a session change")
        } else if spec.restart_after_step_back.is_some() {
            (
                "restart_after_step_back".to_string(),
                "a step back larger than restart_after_step_back",
            )
        } else {
            continue;
        };
        return Err(format!(
            "config: skip_learned = true (--skip-learned), but spec {:?} sets {setting}, so a \
             group's clock may start again lower at {how}, and then its last clock is not a \
             place in the stream: the rows after it are not the rows the state has not \
             learned. If, within each group, the clock never goes back (a timestamp that runs \
             on across sessions), set clock_runs_on = true (--clock-runs-on). Otherwise remove \
             the rows the saved state has learned from the input, by a column that orders the \
             whole stream such as a date or a row number, and leave out skip_learned \
             (--skip-learned).",
            spec.name
        ));
    }
    Ok(())
}

/// What one spec says of a row, from its group's last clock.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Strictly before the last clock: the bank learned past it.
    Before,
    /// At the last clock, with a weight the spec learns from.
    Tied,
    /// At the last clock, with weight 0: it learns nothing and counts in no
    /// clock, so replaying it is harmless.
    TiedAtZero,
    /// After the last clock, or in a group the spec has not seen, or on a
    /// null clock or one that does not compare (a NaN): kept for the bank to
    /// judge.
    New,
}

impl Learned {
    /// True on the rows of `df` the bank has not learned. A row that some
    /// spec reads strictly before that spec's group's last clock is dropped:
    /// the bank is past it. Of the rest, a row is kept when at least one spec
    /// has it after its group's last clock, in a group the spec has not seen,
    /// or on a null clock (which is kept for the bank to judge), so that one
    /// spec's tie at its last clock does not hide a row another spec has not
    /// learned; and when no spec has it so, kept too if it is a row of weight
    /// 0 at a last clock and no spec learned it at a positive weight. A row
    /// at its group's last clock otherwise counts as learned. A temporal
    /// clock is compared in the integer nanoseconds the bank keeps, and an
    /// integer clock in integers (task 200).
    ///
    /// # Errors
    ///
    /// A column a spec reads is missing or has a dtype the bank refuses, as
    /// [`Bank::fit_predict`] says it; and a clock temporal in the frame and
    /// numeric in the bank, or the other way round.
    pub fn unlearned(&self, df: &DataFrame) -> PolarsResult<BooleanChunked> {
        let n = df.height();
        let (mut before, mut new) = (vec![false; n], vec![false; n]);
        let (mut tied, mut tied_zero) = (vec![false; n], vec![false; n]);
        let chunk = chunk_from_frame(df, &self.specs)?;
        for ((spec, lasts), mark) in self.specs.iter().zip(&self.lasts).zip(&self.marks) {
            let (Some(name), Some(lasts)) = (&spec.clock, lasts) else {
                continue;
            };
            let clocks = chunk.clock(spec, name)?;
            let weights = match &spec.weight {
                Some(w) => Some(chunk.f64(spec, "weight", w)?),
                None => None,
            };
            let mut seen = vec![false; n];
            for (key, rows) in group_indices(&chunk, spec)?.iter() {
                let key = key.to_key();
                let Some(last) = lasts.get(&key) else {
                    // A group below the largest key a monotone close has
                    // read was learned and closed, whatever its clock.
                    if let Some((top, integer)) = mark
                        && super::key_cmp(&key, top, *integer) == std::cmp::Ordering::Less
                    {
                        for &i in rows {
                            seen[i] = true;
                            before[i] = true;
                        }
                    }
                    continue;
                };
                for &i in rows {
                    seen[i] = true;
                    let now = match clocks {
                        ClockArray::F64(a) => a.get(i).map(ClockValue::F64),
                        ClockArray::Nanos(a) => a.get(i).map(ClockValue::Ns),
                        // An integer clock, compared in integers (task 200).
                        ClockArray::I64(a) => a.get(i).map(ClockValue::I64),
                        // Past `i64::MAX` is after every clock an integer
                        // bank holds: kept, for the bank to refuse by row,
                        // as `ModelBank.skip_learned` keeps it.
                        ClockArray::U64(a) => match a.get(i).map(i64::try_from) {
                            None => None,
                            Some(Ok(v)) => Some(ClockValue::I64(v)),
                            Some(Err(_)) => {
                                new[i] = true;
                                continue;
                            }
                        },
                    };
                    // A comparison that is undefined -- a NaN clock -- is
                    // neither before nor at the last clock, and keeps the row
                    // for the bank to refuse by row, as a null is kept and as
                    // `ModelBank.skip_learned` keeps it; read as "not after",
                    // it dropped the row in silence (review round 5, C3).
                    let order = match (now, last) {
                        (None, _) => None,
                        // An infinite clock is no place on the clock either,
                        // and is kept for the bank to refuse by row, as a NaN
                        // is: read as before the last clock, `-inf` was
                        // dropped in silence (review 2026-11-23, BANK-2).
                        (Some(ClockValue::F64(now)), _) if !now.is_finite() => None,
                        (Some(ClockValue::F64(now)), ClockValue::F64(last)) => {
                            now.partial_cmp(last)
                        }
                        (Some(ClockValue::Ns(now)), ClockValue::Ns(last)) => Some(now.cmp(last)),
                        (Some(ClockValue::I64(now)), ClockValue::I64(last)) => Some(now.cmp(last)),
                        // An integer and a float clock are both numbers,
                        // compared as doubles, as `ModelBank.skip_learned`
                        // compares them; the bank refuses the chunk by its
                        // dtype (task 200).
                        (Some(ClockValue::I64(now)), ClockValue::F64(last)) => {
                            (now as f64).partial_cmp(last)
                        }
                        (Some(ClockValue::F64(now)), ClockValue::I64(last)) => {
                            now.partial_cmp(&(*last as f64))
                        }
                        (Some(now), _) => {
                            let (is, was) = match now {
                                ClockValue::Ns(_) => ("temporal", "numeric"),
                                ClockValue::F64(_) | ClockValue::I64(_) => ("numeric", "temporal"),
                            };
                            polars_bail!(ComputeError:
                                "spec {:?}: clock column {:?} is {is} in the frame and was {was} \
                                 in the bank, so its rows cannot be compared with the clock the \
                                 bank has learned up to. Cast the clock column back to a {was} \
                                 type.",
                                spec.name, name
                            );
                        }
                    };
                    let verdict = match order {
                        Some(std::cmp::Ordering::Less) => Verdict::Before,
                        Some(std::cmp::Ordering::Equal) => {
                            if weights.is_some_and(|w| w.get(i) == Some(0.0)) {
                                Verdict::TiedAtZero
                            } else {
                                Verdict::Tied
                            }
                        }
                        _ => Verdict::New,
                    };
                    match verdict {
                        Verdict::Before => before[i] = true,
                        Verdict::Tied => tied[i] = true,
                        Verdict::TiedAtZero => tied_zero[i] = true,
                        Verdict::New => new[i] = true,
                    }
                }
            }
            // A group the spec has not seen is new to it.
            for (i, seen) in seen.iter().enumerate() {
                new[i] |= !seen;
            }
        }
        let keep: Vec<bool> = (0..n)
            .map(|i| !before[i] && (new[i] || (tied_zero[i] && !tied[i])))
            .collect();
        Ok(BooleanChunked::from_slice("unlearned".into(), &keep))
    }
}
