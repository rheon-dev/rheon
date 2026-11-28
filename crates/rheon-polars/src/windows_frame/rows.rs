//! The row operators' own state and their pass (task 564): a run length's
//! run, a clock drift's sums and a clock step's moments, kept per group
//! beside an increment's previous value, and the pass that computes every
//! row operator one row at a time from its group's previous rows, before
//! the window core.

use super::*;

/// A row operator's own state in a group (task 564), beside an increment's
/// previous value, which [`IncrState`] keeps for every row operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) enum RowState {
    /// An increment's: its previous value is in [`IncrState`].
    Step,
    /// A run length's run.
    Run(RunLength),
    /// A clock lag's or drift's sums, and the arrival they are centred on:
    /// the first arrival taken since the sums last started.
    Drift {
        sums: ClockDrift,
        origin: Option<RunValue>,
    },
    /// A clock step's statistics, and the time its next step is taken
    /// from: the last time read since the chain of steps last started.
    Steps {
        stats: ClockSteps,
        prev: Option<RunValue>,
    },
}

impl RowState {
    pub(super) fn of(op: &RowOp) -> Self {
        match op.kind {
            OpKind::RunLength => {
                RowState::Run(RunLength::new(op.tol).expect("checked by the plan"))
            }
            OpKind::ClockLag | OpKind::ClockDrift => RowState::Drift {
                sums: ClockDrift::new(op.half_life).expect("checked by the plan"),
                origin: None,
            },
            OpKind::ClockStep => RowState::Steps {
                stats: ClockSteps::new(
                    op.half_life,
                    op.step.expect("a clock step's statistic"),
                    op.bias,
                )
                .expect("checked by the plan"),
                prev: None,
            },
            _ => RowState::Step,
        }
    }

    /// Start over at a new session or a reset; `gap` too at a gap past
    /// `gap_cap`, which ends a drift's stretch as it ends a window's, and
    /// leaves a run as it is.
    pub(super) fn restart(&mut self, gap: bool) {
        match self {
            RowState::Run(r) if !gap => r.restart(),
            RowState::Drift { sums, origin } => {
                sums.restart();
                *origin = None;
            }
            RowState::Steps { stats, prev } => {
                stats.restart();
                *prev = None;
            }
            _ => {}
        }
    }
}

/// A row operator of the plan: an increment, a run length, a clock lag or a
/// clock drift (task 564), each computed one row at a time from its group's
/// previous rows before the window core, as the hidden column `key`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RowOp {
    pub(super) key: String,
    pub(super) kind: OpKind,
    /// The slot of its input: the value, or a drift's arrival time.
    pub(super) input: usize,
    /// A drift's event time's slot.
    pub(super) event: Option<usize>,
    /// A run length's tolerance in the units the input is read in, a
    /// temporal input's in nanoseconds.
    pub(super) tol: f64,
    /// A drift's or a clock step's half-life, in policy-clock units.
    pub(super) half_life: f64,
    /// A clock step's statistic, and a spread's `bias`.
    pub(super) step: Option<StepStat>,
    pub(super) bias: bool,
}

/// A row value as a run reads it, from slot `i` of `inputs` at row `r`: an
/// integer or temporal input whole, a number that is not [`usable`] missing.
pub(super) fn row_value(inputs: &IncrInputs, i: usize, r: usize) -> RunValue {
    if let Some(v) = &inputs.integer[i] {
        return v[r].map_or(RunValue::Missing, RunValue::Int);
    }
    if let Some(v) = &inputs.numeric[i] {
        return match v[r] {
            Some(x) if usable(x) => RunValue::Float(x),
            _ => RunValue::Missing,
        };
    }
    if let Some(v) = &inputs.temporal[i] {
        return v[r].map_or(RunValue::Missing, |ns| RunValue::Int(i128::from(ns)));
    }
    RunValue::Missing
}

/// `a − b` in the units a drift reports: two integers in integers and two
/// temporal values in nanoseconds, each rounded once (seconds for the
/// second); a double's difference as it is. `None` where either is missing.
pub(super) fn value_step(a: RunValue, b: RunValue, temporal: bool) -> Option<f64> {
    match (a, b) {
        (RunValue::Int(x), RunValue::Int(y)) if temporal => Some(rheon_core::seconds_of_ns(x - y)),
        (RunValue::Int(x), RunValue::Int(y)) => Some(integer_step(x, y)),
        (RunValue::Missing, _) | (_, RunValue::Missing) => None,
        (x, y) => Some(as_f64(x) - as_f64(y)),
    }
}

pub(super) fn as_f64(v: RunValue) -> f64 {
    match v {
        RunValue::Int(i) => i as f64,
        RunValue::Float(x) => x,
        RunValue::Missing => f64::NAN,
    }
}

/// Clock policy slice that [`step_increments`] reads.
#[derive(Clone, Copy)]
pub(super) struct IncrPolicy {
    pub(super) restart: f64,
    pub(super) restarts: bool,
    pub(super) skips: bool,
    pub(super) gap_cap: f64,
    pub(super) grouped: bool,
}

/// One value a row operator gives, per row of `keys`: an increment's
/// `x_i - x_{i-1}` within each key's session, a run length's run and a
/// clock drift's lag or slope (task 564), moving `track` as it goes.
#[allow(clippy::too_many_lines)]
pub(super) fn step_increments(
    track: IncrTrack<'_>,
    (ops, slots): (&[RowOp], &[(Node, bool)]),
    policy: IncrPolicy,
    inputs: &IncrInputs,
    keys: &[Option<&str>],
    clock: &Option<Vec<ClockValue>>,
    session: Option<&[u64]>,
) -> Vec<Vec<Option<f64>>> {
    let IncrPolicy {
        restart,
        restarts,
        skips,
        gap_cap,
        grouped,
    } = policy;
    let IncrTrack {
        state,
        index,
        null,
        stream_clock,
    } = track;
    let IncrInputs {
        rows: n,
        integer,
        numeric,
        temporal,
    } = inputs;
    let width = ops.len();
    // Only a run length or a drift keeps a state beyond the previous value.
    let keeps_rows = ops.iter().any(|o| !o.kind.is_step());
    // Each tick sign's carried sign, past the increments' previous values in
    // `prev`: +1, -1, or NaN while none is known.
    let carry: Vec<Option<usize>> = ops
        .iter()
        .scan(width, |next, o| {
            Some((o.kind == OpKind::TickSign).then(|| {
                *next += 1;
                *next - 1
            }))
        })
        .collect();
    let carried = carry.iter().flatten().count();
    let mut out: Vec<Vec<Option<f64>>> = vec![Vec::with_capacity(*n); width];
    for r in 0..*n {
        let found = match keys[r] {
            Some(k) => index.get(k).copied(),
            None => *null,
        };
        let gi = match found {
            Some(gi) => gi,
            None => {
                let gi = state.len();
                state.push(IncrState {
                    key: keys[r].map(str::to_string),
                    prev: vec![f64::NAN; width + carried],
                    prev_ns: vec![None; width],
                    prev_int: vec![None; width],
                    session: None,
                    clock: None,
                    rows: if keeps_rows {
                        ops.iter().map(RowState::of).collect()
                    } else {
                        Vec::new()
                    },
                });
                match keys[r] {
                    Some(k) => {
                        index.insert(k.to_string(), gi);
                    }
                    None => *null = Some(gi),
                }
                gi
            }
        };
        // A step back on the stream's clock past the threshold restarts
        // every group in the core, so every group's increments start
        // over too (review R1, S4); under `on_step_back = "skip"`, a step back
        // that does not restart skips the stale row and leaves every state
        // untouched.
        let now_clock = clock.as_ref().map(|c| c[r]);
        let now_session = session.map(|s| s[r]);
        let stream_session_changed =
            !grouped && state[gi].session != now_session && state[gi].session.is_some();
        if let (Some(now), Some(prev)) = (now_clock, *stream_clock)
            && now.is_before(prev)
            && !stream_session_changed
        {
            if restarts && back_past(prev, now, restart) {
                for st in state.iter_mut() {
                    st.prev.iter_mut().for_each(|p| *p = f64::NAN);
                    st.prev_ns.iter_mut().for_each(|p| *p = None);
                    st.prev_int.iter_mut().for_each(|p| *p = None);
                    st.rows.iter_mut().for_each(|s| s.restart(false));
                    st.clock = None;
                }
            } else if skips {
                for col in &mut out {
                    col.push(None);
                }
                continue;
            }
        }
        let st = &mut state[gi];
        let mut new_start = st.session != now_session && st.session.is_some();
        if let (Some(now), Some(prev)) = (now_clock, st.clock)
            && now.is_before(prev)
            && !new_start
        {
            if restarts && back_past(prev, now, restart) {
                new_start = true;
            } else if skips {
                for col in &mut out {
                    col.push(None);
                }
                continue;
            }
        }
        *stream_clock = now_clock;
        if new_start {
            st.prev.iter_mut().for_each(|p| *p = f64::NAN);
            st.prev_ns.iter_mut().for_each(|p| *p = None);
            st.prev_int.iter_mut().for_each(|p| *p = None);
            st.rows.iter_mut().for_each(|s| s.restart(false));
        }
        // A drift's step of the policy clock since the group's last row,
        // and its stretch's end at a gap past `gap_cap`, as a window's
        // (task 564). On the row-count clock a row is one unit.
        let (d_clock, gap) = match (now_clock, st.clock) {
            (Some(now), Some(prev)) => (
                crate::windows::elapsed(now, prev).max(0.0),
                crate::windows::past_cap(now, prev, gap_cap),
            ),
            (Some(_), None) => (0.0, false),
            (None, _) => (1.0, false),
        };
        if gap && !new_start {
            st.rows.iter_mut().for_each(|s| s.restart(true));
        }
        st.session = now_session;
        st.clock = now_clock;
        for (j, op) in ops.iter().enumerate() {
            let i = op.input;
            if !op.kind.is_step() {
                let value = match &mut st.rows[j] {
                    RowState::Run(run) => run.step(row_value(inputs, i, r), 1.0).map(|c| c as f64),
                    RowState::Drift { sums, origin } => {
                        let t = slots[i].1;
                        let a = row_value(inputs, i, r);
                        let e = row_value(inputs, op.event.expect("a drift's event"), r);
                        match value_step(a, e, t) {
                            Some(d) => {
                                let o = *origin.get_or_insert(a);
                                let ac = value_step(a, o, t).expect("both present");
                                sums.step(d_clock, ac, d, 1.0);
                            }
                            // A row missing either time learns nothing, and
                            // the clock still decays the sums (rule 9).
                            None => sums.step(d_clock, 0.0, 0.0, 0.0),
                        }
                        if op.kind == OpKind::ClockLag {
                            sums.lag()
                        } else {
                            sums.slope()
                        }
                    }
                    RowState::Steps { stats, prev } => {
                        let t = row_value(inputs, i, r);
                        if t == RunValue::Missing {
                            // A missing time has no step and reports none,
                            // and the next time starts the chain again;
                            // the clock still decays the statistics.
                            stats.step(d_clock, None, 0.0);
                            *prev = None;
                            None
                        } else {
                            let s = prev.and_then(|p| value_step(t, p, slots[i].1));
                            stats.step(d_clock, s, 1.0);
                            *prev = Some(t);
                            stats.value()
                        }
                    }
                    RowState::Step => unreachable!("a row operator keeps its own state"),
                };
                out[j].push(value);
                continue;
            }
            let step = if let Some(vals) = &integer[i] {
                let v = vals[r];
                let step = match (v, st.prev_int[j]) {
                    (Some(c), Some(p)) => Some(integer_step(c, p)),
                    // A previous value a float input of another chunk left.
                    (Some(c), None) if !st.prev[j].is_nan() => Some(c as f64 - st.prev[j]),
                    _ => None,
                };
                if let Some(c) = v {
                    st.prev_int[j] = Some(c);
                    st.prev[j] = c as f64;
                }
                step
            } else if let Some(vals) = &numeric[i] {
                let v = match vals[r] {
                    Some(x) if usable(x) => x,
                    _ => f64::NAN,
                };
                let step = if v.is_nan() || st.prev[j].is_nan() {
                    None
                } else {
                    Some(v - st.prev[j])
                };
                if !v.is_nan() {
                    st.prev[j] = v;
                    st.prev_int[j] = None;
                }
                step
            } else if let Some(vals) = &temporal[i] {
                let v = vals[r];
                let step = match (v, st.prev_ns[j]) {
                    (Some(c), Some(p)) => {
                        Some(rheon_core::seconds_of_ns(i128::from(c) - i128::from(p)))
                    }
                    _ => None,
                };
                if v.is_some() {
                    st.prev_ns[j] = v;
                }
                step
            } else {
                None
            };
            out[j].push(match carry[j] {
                None => step,
                Some(c) => tick_sign(step, &mut st.prev[c]),
            });
        }
    }
    out
}
