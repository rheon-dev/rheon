# Run length and clock health: design

Design record for run length of unchanged values, clock-step statistics, and the
drift between two clocks. Part 1 of the build: no code, questions for the review.

## What is measured

Three quantities a stream's health is read from, each a running form over the
rows of a group, with the window form where a window applies:

1. **Run length.** For a column, the number of consecutive rows, ending at the
   current row, whose value is unchanged. A frozen feed (a countdown that stops
   at 40, a sensor holding its last reading) shows as a run that keeps growing.
2. **Clock-step statistics.** The distribution of the step `dt` between
   consecutive accepted clock values: its decayed mean and spread, the count of
   zero steps, and the count of steps refused or reset by `min_backwards_jump`.
3. **Two-clock drift.** When each row carries an event time and an arrival time,
   the lag `arrival − event` and its rate of change against arrival time.

## Where each piece lives

- **Run length and clock-step statistics: a window operator** in
  `rheon-polars`'s window engine, a formula target of the same kind as
  `rolling_max`, because it needs the clock and the group's row order. Running
  form when no `window_size` is given, the window form when one is.
- **Drift: the same operator**, reading two columns (`event`, `arrival`), so a
  stream needs no extra pass.
- **Audit: no new output.** `audit` already counts a stream's columns (its
  `Counts` and `Moments`). The clock-health numbers are a stream property, and
  they are read from the operator's own output, not from audit, so audit stays
  as it is. (Whether audit already reports step-backs is to verify in part 2;
  if it does, the operator reports the rest and the two are cross-checked, not
  merged.)

## "Unchanged" defined

- **Exact** by default: two values are equal when their `f64` bits are equal
  (`+0.0` and `-0.0` are unequal under bit equality, so define it on the value
  with `==` and test `-0.0 == +0.0` as equal; NaN is never equal to itself).
- **Null** (and NaN, per the engine's non-finite rule) ends the run: the row
  reports null, and the next non-null value starts a run of 1.
- **Tolerance** is an option `tol=` (absolute, `|x − prev| ≤ tol`), default 0.
  Stated in the output name only when set.

## Clock-step statistics: decay and window

- Decayed moments of `dt` on the clock: the same half-life and `lam` rules as the
  other decayed operators, so `min_weight` and `n_eff` follow rule 8 (the weight
  before the row's own decay).
- The spread reuses the moments the model layer already uses (554, B1, when it
  lands). Until then, a Welford-type update in `rheon-core`, no polars types.
- Zero steps and refused steps are counts, with decay on the same clock.

## Two-clock drift and its estimator

- Lag `d_t = arrival_t − event_t`. Output: decayed mean of `d` and the decayed
  slope of `d` on arrival time, from the two decayed sums
  `Σw·a`, `Σw·a²`, `Σw·d`, `Σw·a·d` (centred on the group's first arrival to
  keep the sums small; the centring offset is state).
- Slope undefined when the arrival spread is zero: withheld with a reason, not
  a guessed value.

## State bytes

- Run length: the last value (8 bytes), the run count (8 bytes), and a flag.
  O(1), running form.
- Window form: the window's values held in the queue already used for extrema
  (112 bytes a row measured in 561). Bounded by `window_size` and `max_state_bytes`.
- Clock steps and drift: a fixed set of decayed sums, O(1), no window.
- Every new operator is refused at build when its state cannot fit
  `max_state_bytes`, with its byte cost.

## Oracles

- **Run length:** Polars' `rle` on the same series, which gives each run's
  length; the running value is read from the run lengths. Third-party, not
  hand-coded.
- **Clock-step statistics:** `numpy` (or `statistics`) over the same steps,
  for the unweighted case; weighted and decayed forms checked against an exact
  rational implementation in Python (the same method as B14's decayed DDSketch).
- **Drift slope:** `numpy.polyfit` (or `statsmodels` OLS) on the same window,
  weights included where the operator uses them.
- **Tests on both sides** of every threshold: `tol` at equality, a run of
  exactly the window length, a zero arrival spread.

## Rules

- Chunk invariance: 1 chunk and 1000 chunks give identical output.
- Save and load to the bit: state is versioned; a new layout bumps the schema
  under the pre-1.0 rule and gets frozen fixtures.
- Zero-weight rows count in no clock and learn nothing (rule 9). A zero-weight
  row does not extend or break a run: to be decided (question 4).
- No fixed row counts: a run or a slope is withheld with a reason until the
  condition holds, never seeded with a value.

## Questions, with recommendations

1. **Operator or audit?** Recommend the window operator for all three; audit
   unchanged. Yes/no.
2. **Tolerance default.** Recommend exact (`tol=0`), with `tol=` opt-in. Yes/no.
3. **Null breaks the run.** Recommend yes: a null row reports null and restarts
   the count at the next non-null. Alternative: nulls carry the run through.
4. **Zero-weight rows and runs.** Recommend: a zero-weight row neither counts
   nor breaks the run, consistent with rule 9 (it learns nothing). Alternative:
   it breaks the run, as a row the feed did report.
5. **Step statistics decay.** Recommend the same half-life rules as the other
   decayed operators, with the window form as an option. Yes/no.
6. **Drift slope.** Recommend the decayed two-sum estimator with centring,
   withheld when arrival spread is zero. Yes/no.
7. **Dependency on 554 (B1 moments).** Recommend not waiting: a Welford update
   now, switched to B1's moments when it lands, with the oracle test unchanged.

## Part 2: the answers, and what was built

The review answered the questions (2026-11-23): an operator, not audit (1);
exact equality, `tol=` opt-in (2); nulls as Polars' `rle` has them (3, below);
a zero-weight row neither counts nor breaks a run (4); the drift by a centred
two-sum decayed estimator, withheld where the arrivals have no spread (6);
and the clock-step statistics on 554's moments (7), so they wait (below).

**Built:** three row operators, `rp.run_length(x, tol=0.0)`,
`rp.clock_lag(arrival, event, half_life=)` and
`rp.clock_drift(arrival, event, half_life=)`. Each is a formula node read by
`with_windows` (and by a formula target, beside a window looking ahead), each
a hidden `Float64` column computed one row at a time from the group's own
previous rows, in the pass that computes `increment`, before the window core.
The recursions are in `rheon-core` (`run_length.rs`, `clock_drift.rs`), with
no Polars type; the plumbing is `windows_frame.rs`'s increment pass, which now
carries any row operator and its inputs ("slots"), a drift's two among them.

### Nulls, measured

Polars' `rle` on `[1, 1, null, null, 1, NaN, NaN, null, 0.0, -0.0, inf, inf,
2, null]` under 1.38.1, 1.41.1 and 2.1.0 (the locked one) gives the same runs
on all three: lengths `2, 2, 1, 2, 1, 2, 2, 1, 1`. So:

- consecutive nulls are one run, of their own; a null beside a value ends the
  run on both sides (`[1, null, 1]` is three runs);
- `0.0` and `-0.0` are one value; NaN equals NaN, and `inf` equals `inf`;
- integers: `[1, null, null, 2, 2]` gives `1, 2, 2`.

The operator mirrors this, with the engine's rule for a value that is not
usable (NaN, an infinity, beyond `1e100`): it is missing, as a null is, so it
joins a run of nulls, where `rle` keeps a run of NaN apart. A null row reports
its place in the null run, as `rle` read row by row does, not null (the first
draft said null; the answer to question 3 replaced it). The oracle is `rle` on
the series with the unusable values set to null. Under `tol > 0` two adjacent
numbers are one value when `|x − prev| ≤ tol`, and a run can creep.

### Restarts and zero-weight rows

A run restarts where an increment does: at a new session and at a step back
past `restart_after_step_back`, the group's own or the stream's. A gap past
`gap_cap` leaves it, since a value held across a gap is still held. A drift's
sums restart there and at a gap past `gap_cap` too, which ends a stretch, as
for the window operators with no `window_size`.

`with_windows` has no row weight, so the surface has no zero-weight row; the
core takes one, and rule 9 is held there: a row of weight 0 neither extends
nor ends a run and reports it as it stands (`None` before the first row), and
a drift's row of weight 0 decays the sums and moves no mean, the first row
included, with no `0/0`. A row missing either time of a drift is a row of
weight 0.

### The drift's estimator

Per group, the decayed weight `W`, the weighted means of the centred arrival
and of the lag, and the two sums about those means, `C_aa` and `C_ad`, by
West's weighted update with the clock's decay scaling `W` and both sums. The
arrivals are centred on the stretch's first arrival, subtracted in integer
nanoseconds on a temporal column. The record's first draft kept `Σw·a`,
`Σw·a²`, `Σw·d`, `Σw·a·d` about that origin; sums about the running means are
the same two-sum estimator without the cancellation of `S·Saa − Sa²`, which
grows with the stretch's age next to the half-life, and make `C_aa` 0 exactly
while every arrival is one value, so the slope is withheld on that condition
alone. The decay is on the policy clock (one unit a row on the row-count
clock); the slope is in lag units per arrival unit (seconds a second on
temporal times). Withheld means null in the column, with the reason in the
docstrings: no row with both times (the lag), or no spread in the arrivals
(the slope).

### State bytes

Per group and operator: a run length's tolerance, last value and count, 48
bytes; a drift's half-life and five doubles, 56 bytes, and its origin, 32.
Each sits beside the increment's previous value every row operator keeps (56
bytes). None grows with the stream: no window, so `max_state_bytes` has
nothing to bound and is not taken (a fixed cost, as an increment's is). The
windows state moves to version 13: a group's state may carry `rows`, written
only where there is a run length or a drift, so a version-12 state loads as a
13 without one; an older build refuses a 13 by number.

### Deferred

- **Clock-step statistics.** They are to reuse 554's moments, which are not
  on `main` (`vm/task-554-moments`). They wait until 554 merges; then this
  branch is rebased, or they are added on top. Note for that part: `audit`
  already keeps, on a `clock` column, each step's class (a duplicate stamp,
  a gap at or past `gap_cap`, a regular step) and the regular steps' mean,
  population spread, coefficient of variation and largest step. It does not
  count step-backs itself: it refuses one as every spec does, and the bank's
  `summary()` (`clock_backwards`) and `check()` (`step_back`) count them
  under `restart_after_step_back`. So the operator's step statistics would
  repeat audit's on a stream, and their value is the running column form;
  the cross-check the first draft proposed is audit's table against the
  operator's last row.
- **A run length's window form.** It is the run truncated to the window,
  `min(run_length, rows in the window)`, which a formula writes from
  `rp.run_length` and an `rp.ewm_sum` of ones under `half_life=inf`; the
  queue the first draft pointed at is 561's, which is not on `main` either.

### Tests

Rust: `run_length.rs` and `clock_drift.rs` in `rheon-core` (the definition,
both sides of `tol` and of the spread, rule 9, restarts, save and load to
the bit, state size); `windows_frame.rs` (the row operators through a run,
chunking, a save at every row, a refused state of another shape); the
windows fixture `with_windows_rows`. Python: `tests/test_run_length.py`,
against Polars' `rle` and statsmodels' `WLS` (see its docstrings).

## Index row (for the Mac to add at merge)

```text
| [records/CLOCK-HEALTH-DESIGN.md](records/CLOCK-HEALTH-DESIGN.md) | record | Run length and clock health: the design of run length of unchanged values, clock-step statistics and two-clock drift (closes the design part of PLAN 564) |
```
