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

## Index row (for the Mac to add at merge)

| [records/CLOCK-HEALTH-DESIGN.md](records/CLOCK-HEALTH-DESIGN.md) | record | Run length and clock health: the design of run length of unchanged values, clock-step statistics and two-clock drift (closes the design part of PLAN 564) |
