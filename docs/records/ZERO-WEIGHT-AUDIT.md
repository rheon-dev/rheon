<!-- kind: record -->
# Rows of weight 0 in no clock: the audit (task 539)

The user, 2026-11-21: "Zero weight rows should not count in the clock the
next row should decay as if they were not present. This is always true on
every model." And: "All clocks should work the same way, if the clock is a
time then 5m should work."

This record is the audit task 539 asked for first: every model kind, window
kind and diagnostic where a row of weight 0 moved a later row's output, found
by testing, not by reading; then what changed, and what it moved.

## How it was tested

`crates/rheon-polars/tests/zero_weight.rs`, `a_zero_weight_row_counts_in_no_clock_on_every_kind`.
One stream of 420 rows: features `x0` and `x1` with nulls, a target, a 0/1
target and a label, a weight of 0 on 66 rows -- the first row, a scatter
(`i % 11 == 4`), runs of four (63-66, 100-103), the three rows about a gap of
7,200 clock units past every `gap_cap` (259-261), the first row of each
40-row session, and row 300, where the clock column `tb` steps back by 0.5.
Every spec runs twice, on the stream and on the stream with those rows taken
out, in the same chunks (37 rows of the first, the same rows of positive
weight in the second), on a row clock, a number clock in seconds and a
`Datetime` clock at the same instants with every span a duration. Every
output field of every row of positive weight, and the saved state but for
the counts of rows fed (`rows_processed`, `rows_fed`, the data summary, a
closed group's clock range, the checksum), are compared to the bit.
`RHEON_ZERO_WEIGHT_AUDIT=1` reports every case that parts instead of failing
at the first. The table below is that report on a3554b27, the build before
the change (a copy of the test run in an export of that commit).

35 specs over the clocks each runs on, 106 cases: every kind a bank takes a
weight for, a windowed form of `ewridge`, `ew_cov`, `ew_class` and `hawkes`
(events, and clock units at either edge), every diagnostic and `coef_every`
in one spec, a grid with `emit_averaged` and `emit_selected`, sessions with a
gap and with a reset, `embargo`, `label_clock`, and `restart_after_step_back`.
`audit` and `seqtest` are not in it: a bank refuses a weight column for
either.

## What a row of weight 0 moved on a3554b27

106 of 106 cases parted. "State" is the first field of the saved state that
differed; "outputs" the fields that differed on a row of positive weight.

| kind | row clock | number and temporal clock | how |
|---|---|---|---|
| `bocpd` (`hazard` a count) | state only (the clock's row count) | state only (the clock's removed time) | its recursion reads rows, not the clock; the stream's clock counted the rows |
| `bocpd` (`hazard` a duration) | -- | `p_change`, `run_mode`, `run_mean`, `pred_*`, `loglik` | each step's hazard `1 - exp(-d/hazard)`: the zero-weight row took a step, so the next row's chance of a break was split in two |
| `corrchange` | state only | state only | as `bocpd` with a count |
| `deco` | `u`, `rho`, `loglik`, `weight_sum`, `settled_frac`, `coef` | the same | the decay of its moments and the weight |
| `ew_class`, windowed | `p_*`, `weight_sum`, `settled_frac`, `coef`; windowed also `class`, `withheld_reason` | the same | decay, and the window's edge read from a step it had taken |
| `ew_cov`, windowed with `pca` | every `mean_`, `std_`, `corr_`, `pc0_*` field, `weight_sum`, `settled_frac` | the same, and `withheld_reason` | decay; the window's edge |
| `ewridge`, windowed | `pred`, `resid`, `weight_sum`, `settled_frac`, `coef`, `support_coef` | the same | decay; a window's edge |
| `ewridge`, a grid with `emit_averaged` and `emit_selected` | every instance's fields, `pred_y__selected`, `selected_y`, `pred_y__averaged` | the same | each instance's decay, and the selection read from them |
| `ewridge` with every diagnostic | 41 fields: `pred`, `resid`, `error_inflation`, `sigma`, `zscore`, the conformal `lo`/`hi`/`coverage`, `ic`, `r2`, `hit_rate`, the residual quantiles, `autocorr`, the calibration, `studentized`, `cusum`, `cusum_sq`, `break_wald`, `ljung_box`, `breusch_pagan`, `ramsey_reset`, `skew`, `kurtosis`, `jarque_bera`, `influence`, the feature health, `weight_sum`, `settled_frac`, `coef` (on its cadence), `se_coef`, `se_coef_hc0` | 39 of the same | each diagnostic aged its memory on the row (`sigma`, the quantiles, the autocorrelation, the drift detector, the metrics, the conformal multiplier, the checks), the readiness waits started or ended on it, and the `coef` cadence counted it |
| `ewridge` with a session (`session_gap` a span, or `"reset"`) | `pred`, `resid`, `weight_sum`, `settled_frac`, `coef` | the same, and `withheld_reason` | a zero-weight row that began a session took the session's rule (the gap, or a reset), and the row after it did not |
| `ewridge` under `embargo` | `pred`, `resid`, `sigma`, the conformal fields, `weight_sum`, `settled_frac`, `coef` | the same | the row was held and replayed as a step, and its place released rows |
| `ewridge` under `label_clock` | -- | `pred`, `resid`, `sigma`, `weight_sum`, `settled_frac`, `coef` | the row was held (rule 5's twenty-first exception) and learned at its label time as a step of the clock of labels learned |
| `ewridge` with `restart_after_step_back` | -- | refused: the zero-weight row stepped back by 0.5, under the bound of 1, a late row | the step-back rule read the row |
| `ftrl`, `glm`, `holt`, `huber`, `kalman`, `lasso`, `pa`, `quantile`, `rls` | `pred`, `resid`, `weight_sum`, `settled_frac`, `coef` (`lasso` also `penalty_selected`, `huber` and `quantile` `support_coef`) | the same | decay; `holt`'s level carried along its trend for the step, `kalman`'s process noise added for it, `glm`'s refit counted the row |
| `hawkes` (no window) | `intensity_*`, `weight_sum`, `settled_frac`, `coef` | the same | its own carry held the zero-weight rows' steps, but the stream's row clock counted them, and the emitted `coef` sat on a zero-weight row |
| `hawkes`, a window of events | `intensity_*`, `coef` | the same | as above |
| `hawkes`, a window of clock units, either edge | `intensity_*`, `weight_sum`, `coef` | `coef` only, and the stream's clock | its carry kept a number or temporal clock right; the row clock and the `coef` row did not |
| `hmm`, `kmeans`, `sgd` (squared and logistic) | `weight_sum`, `settled_frac` (`sgd` also `coef`) | the same | the shared weight's decay; the fit itself did not move on this stream |
| `marginal` | `weight_sum` | the same | the weight's decay |
| `micro` | `cluster`, `dist`, `micro_id`, `outlier`, `n_clusters`, `n_micro`, `weight_sum`, `settled_frac`, `coef` | `dist`, `weight_sum`, `settled_frac`, `coef` | decay, which on a row clock changed which micro-clusters faded |
| `rcov` | state only (the clock's row count) | state only (the clock) | its blocks count the model's own rows, which it was handed at weight 0 as gaps; its outputs did not move on this stream |
| every kind, `coef` without a cadence | written on the chunk's last accepted row, a zero-weight row among them | the same | `last_accepted` took a row of weight 0 |
| every kind | `settled_frac` | `settled_frac` | the first row weighed 0 and started the clock; the stream without it started at the next row |

The mechanisms, each now one place:

1. **The stream's clock.** A zero-weight row was an accepted row: the
   `ClockState` advanced on it, so on a row clock it was a step, under
   `gap_cap` its gap was capped on its own, a session change or a step back
   on it took the session's or the step-back rule, and a zero-weight first
   row started the clock.
2. **The models' own memories.** Every model was stepped with `w = 0` and
   decayed over the step (hard rule 9's old reading, "advance the clock,
   learn nothing"). On a time clock with plain exponential decay that is
   invisible but for rounding (`λ^a λ^b` against `λ^(a+b)`); everywhere a gap
   is not simply multiplicative it moved outputs: a row clock, `gap_cap`,
   per-row recursions (`holt`, `kalman`'s process noise, `hmm`), row windows
   and row counts (`glm`'s refit, warm-ups), and `n_eff`.
3. **The diagnostics.** `sigma`, the quantiles, the autocorrelation, the
   metrics, the conformal multiplier, the drift detector and the checks aged
   on a learned zero-weight row.
4. **The held-row buffers.** Under `embargo` and `label_clock` the row was
   held and replayed as a step.
5. **The notices' waits.** A zero-weight row started or ended the readiness
   gates' waits, and so moved when a "not met" notice came.
6. **`coef`.** Without a cadence `coef` sat on the chunk's last accepted row,
   a zero-weight row among them; with one, the cadence counted the row.

## The change

One place, the stream (`crates/rheon-polars/src/stream.rs`, `Stream::schedule`):
a row whose weight is 0 is scored as `predict` scores a row -- from the state
as it stands, over its step read on a copy of the clock
(`ClockState::advance_scoring`), a session change that would start a fresh
stream or blend toward the long run read on such a copy of the model -- and
the stream's clock, its row count, the `coef` cadence and the fresh fold
stay where they were. So no model, window, diagnostic, held-row buffer or
notice wait sees the row, and the next row steps from the last row that
counted. It releases the held rows whose wait or label time it has seen out
before it is scored, as the next row would (so under `label_clock` it is
scored from the labels known at its clock, which the doubled-stream oracle
in `label_clock.rs` holds), and it is held for nothing. The clock check
(`Stream::check_clock`) and the label check skip it, and `coef` without a
cadence sits on the chunk's last row of positive weight. The readiness
notices' waits neither start nor end on it, and under `label_clock` it is no
`labels_missing` (it waits for no label). A zero-weight row whose features
are null is skipped, and counts in no clock either. A state saved before the
change continues under the new rule with no loader: rule 5's twenty-fifth
exception, with its measurement (docs/records/STATE-CONTINUATION.md).

Only the core changed besides: `hawkes` loses its own `carry` and `seen`
(the stream does it now; the state is two fields shorter, `8 (k + D (1 + k)
+ D (D + 1) / 2 + 2)` bytes), and a row of weight 0 handed to it directly
ages its state as every model's does; `audit` counts nothing of a row of
weight 0, its clock step included (rule 5's twenty-fourth exception said the
step was counted).

On this build every one of the 106 cases is equal to the bit, outputs and
state, and so are 36 more on `tn`, a number clock whose steps no double sums
exactly (tenths and a third): the gap after a row of weight 0 is read once,
from the last row that counted, never as a sum of the steps either side of
it, so the equality holds to the bit on every OS. The three clocks agree: `a_number_clock_and_a_temporal_one_agree_on_every_kind`
runs each spec on the number clock and on the `Datetime` clock with every
span a duration, to the bit.

## After the merge: the chunk's checks and the session close (W9)

The full gate's property test found what the 106 cases, all on one chunking
of 37 rows, did not: three chunk-level readers of the clock still saw a row
of weight 0. `Stream::check_steps` (a step too large for a float) read each
step from the row before, where the stream's clock holds the last row that
counts; `session_bounds` under `group_close = "session"` made such a row a
session boundary; and `last_session`, the closed row's session, was the
span's last row whatever it weighed. Each now reads the last row that
counts, so a row of weight 0 is no boundary and the step past it is the
one the schedule takes; it is still checked over its own step, which it is
scored over. Beside them, the chunk was refused at whichever refusal a
stream or the pool found first; it is now refused at its first refused
input row. The tests: `zero_weight.rs`'s five `under_every_cutting` cases,
which fail on the merged build, and `test_properties.py`'s
`test_chunking_never_changes_the_outcome_behind_zero_weight_rows` and
`test_a_step_back_behind_zero_weight_rows_is_refused_at_one_row_whatever_the_chunking`.
The failure itself was the test's: its streams stepped back by
`MINIMUM + 2` from the row before, which past rows of weight 0 can be
`MINIMUM` or less from the last row that counts, and the bank refused such
a stream at the same row under every chunking.

## What the audit leaves as it was

- **The window formula kinds** (`rp.ewm_mean` and the rest, and a formula
  target's window). They mirror Polars' `ewm_mean_by` and `rolling_*_by`,
  which take no weights: `WindowsConfig` has no weight column, and a formula
  target's window core reads every row of the frame, a zero-weight row's
  value included. Their Polars-matching semantics are unchanged, so a
  formula target's value on a row of positive weight can read a zero-weight
  row's value.
- **The data summary and the counts of rows fed** (`rows_processed`,
  `rows_fed`, `rows_zero_weight`, the clock column's range, a closed group's
  range). They describe the input, every row fed.
- **`bocpd`'s `hazard` as a number** is a count of rows on every clock, and
  a duration is time; a number clock cannot say "time between changepoints"
  (task 538's half).
- **The notices' rounding bound** (`settled_weight`'s count of decay
  steps) counts processed rows, a zero-weight row among them: a bound, and
  no row's output.

## One span type

Every span, rate and decay parameter a spec takes is read once in the spec
layer: the clock spans as `Span` / `SpanList` (`crates/rheon-polars/src/span.rs`),
a number of the clock's units or a duration, and every count of rows as an
integer. None is parsed by a model. The counts of rows (`warm_rows`,
`refit_rows`, `window_rows`, `block_rows`, `window_events`,
`max_rows_between_*`, the lags, `horizon_rows`, `max_held_rows`) count the
rows a model or a diagnostic learns, so a zero-weight row is in none of them
now by construction. Two stand out:

- `hawkes`' `kernel_decays` is a rate per clock unit, refused on a temporal
  clock for its duration form `kernel_half_lives`, as `kalman`'s `q` is.
- `bocpd`'s `hazard`, above.

Which of the counts should also take a duration is task 538's audit.
