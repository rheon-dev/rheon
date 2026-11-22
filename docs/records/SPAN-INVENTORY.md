<!-- kind: record -->
# Every span a spec takes, in rows and on the clock (task 538)

The user, 2026-11-21: "All models should work on rows and clock if
possible", and "All clocks should work the same way, if the clock is a time
then 5m should work". This is the audit task 538 asked for: every parameter
of every kind, diagnostic and option that is a span (a window, a half-life,
a lag, a cadence, a warm-up, a start, a refit, a decay or a rate), what it
takes today, and whether a form is missing and why. Branch
`task-538-rows-and-clock`, on integration 4ddb6462 (schema 72), 2026-11-22.

## How a span is read

There is one span type and one clock. A spec has at most one clock column;
without one, each row is one unit of clock. A clock parameter is a `Span`
(`crates/rheon-polars/src/span.rs`), read once in the spec layer and listed
in `CLOCK_FIELDS` (`spec/clock.rs`), so it meets every rule a clock
parameter does: a number is clock units (rows without a clock, the
column's units on a numeric clock), a duration such as `"5m"` is refused
without a temporal clock, a plain number is refused on one (it would take
the column's storage unit), and a spec that mixes the two is refused. The
stream's `ClockState` turns stamps into each learned row's step and exact
`Stamp` (integer nanoseconds on a temporal clock, raw values on a number
clock, the count of rows that counted on the row clock); a row of weight 0
counts in no clock (task 539). A model with a span that holds rows reads
the span's end off those stamps (`Stamp::cmp_span`, and from this task
`clock_span::span_reached`) and bounds what it holds as it runs
(`clock_span::HeldCap`, task 537's rule: a warning past half of the cap,
once a group, and a refusal at the cap naming the group and the bytes).

So "works on rows and on the clock" means: a parameter that is a `Span`
works on both, in the spec's one clock. A parameter that is an integer
count of rows counts rows on every spec, a clocked one too; such a count
needs a clock form beside it (`<name>_span`, or `window_size` beside
`window_rows`) wherever the quantity has a meaning on the clock. The table
says, for each count, whether it has one, gained one here, or has no
meaning on the clock and why.

**Forms.** *rows*: a count of rows (rows of positive weight, task 539).
*number*: a number of a numeric clock's units, or of rows without a clock.
*duration*: on a temporal clock. *both* below means a `Span` (number and
duration) and, where noted, a row-count sibling.

## Every spec

| parameter | takes | holds rows the data sizes | status |
|---|---|---|---|
| `half_life` (and each diagnostic's `*_half_life`: `calibration`, `breaks`, `robust_se`, `specification`, `tails`, `influence`, `feature_health`) | number, duration | no | both |
| `lam` | a rate per clock unit (number) | no | no duration form by definition (a decay over one unit): refused on a temporal clock, whose form is `half_life` |
| `gap_cap`, `restart_after_step_back` | number, duration | no | clock only: each reads a clock's step, and a row has no step to cap; refused without a clock |
| `session_gap` | number, duration (or a mode word) | no | both |
| `embargo` | number (rows without a clock), duration | held rows, capped by `max_held_rows` | both |
| `coef_every` / `max_rows_between_coefs` | number, duration / rows | no | both |
| `drift_threshold` (Page-Hinkley, CUSUM) | number, duration | no | both |
| `resid_autocorr_lag`, `robust_se_lags`, `ljung_box_lags` | rows | a ring of lags | rows: each is a lag of a series of observations (an autocorrelation, Newey and West's and Ljung and Box's lags). A clock lag ("the residual one minute ago") is task 536's exact clock lag, whose ring holds a data-sized count of rows under task 537's cap: open there |
| `horizon_rows` | rows | no | rows: the count of overlapping residuals the HAC and CUSUM variances read; without a clock it defaults to `embargo` in rows, and a clock form would count the rows inside `embargo`, data-sized: open, with task 536 |
| `max_held_rows` | rows | is the cap | a bound on state, not a span |
| `conformal_rate` | a step per scored row | no | rows: adaptive conformal inference's step per observation (Gibbs and Candès) |
| `min_weight`, `average_eta`, `min_settled_frac` | a weight, a sharpness, a share | no | not spans |
| `drift_options` (`clock`, `min_window_length`, `grace_period`, `warm_start`, `window_size`, `stat_size`; `max_width`, `max_buckets`) | rows | ADWIN's and KSWIN's windows, capped | rows: each detector's test is over counts of observations (ADWIN's bound, KSWIN's two samples, DDM's and EDDM's error counts); `max_width` and `max_buckets` bound state |

## Window kinds (`with_windows`, formula targets)

| parameter | takes | status |
|---|---|---|
| `half_life`, `window_size` | number (rows without a clock), duration | both, held to Polars' `ewm_mean_by` and `rolling_*_by` |
| `min_samples` | rows | rows, as Polars' |

## Each kind

| kind | parameter | takes | holds rows the data sizes | status |
|---|---|---|---|---|
| `ewridge` | `long_half_life` | number, duration | no | both |
| | `solve_every` / `max_rows_between_solves` | number, duration / rows | no | both |
| | `window_size`, `snapshot_every` / `max_rows_between_snapshots` | number, duration / rows | snapshots, under `window_budget` | both |
| `lasso` | `select_half_life`, `solve_every` / `max_rows_between_solves`, `window_size`, `snapshot_every` / `max_rows_between_snapshots` | as `ewridge` | as `ewridge` | both |
| `kalman` | `coef_half_life`, `revert_half_life` | number, duration (a list) | no | both |
| | `q` | a variance per clock unit | no | no duration form (a rate): refused on a temporal clock, whose form is `coef_half_life` |
| `huber` | `huber_scale_half_life`, `solve_every` / `max_rows_between_solves` | number, duration / rows | no | both |
| | `huber_refit_rows` / **`huber_refit_span`** | rows / number, duration | the first rows; a count's buffer held at full size from the first row, a span's grown as it fills | **both, from part 2**: the span's buffers held to 256 KB a group as they grow |
| `quantile` | `solve_every` / `max_rows_between_solves` | number, duration / rows | no | both |
| | `quantile_refit_rows` / **`quantile_refit_span`** | rows / number, duration | as `huber`'s | **both, from part 2**, at least three rows a coefficient |
| `sgd` | `huber_scale_half_life` | number, duration | no | both |
| | `huber_refit_rows` / **`huber_refit_span`** | rows / number, duration | as `huber`'s | **both, from part 2** |
| | `schedule`, `power`, `learning_rate` | a step count | no | rows: a stochastic-approximation schedule `η_t` is over steps |
| `ftrl`, `pa`, `rls`, `seqtest` | none | | | |
| `ew_cov` | `pca_every` / `max_rows_between_pcas`, `window_size`, `snapshot_every` / `max_rows_between_snapshots` | number, duration / rows | snapshots | both |
| | `lags` | rows | a ring | rows; a clock lag is task 536's (above) |
| `holt` | `trend_half_life` | number, duration | no | both |
| `kmeans` | `warm_rows` / **`warm_span`** | rows / number, duration | the seeding rows, capped at 256 MiB | **both, from part 2**: the span's rows checked against the 256 MiB as they come |
| | `update_every_rows` / **`update_every`**, `split_merge_every_rows` / **`split_merge_every`** | rows / number, duration | no (a mini-batch is sums a center) | **both, from part 2**: the cadences on the shared `Since`, counted from the seeding |
| `micro` | `prune_every` / `max_rows_between_prunes` | number, duration / rows | no | both |
| | `max_clusters` | count | is the cap | a bound on state |
| `ew_class` | `window_size`, `snapshot_every` / `max_rows_between_snapshots` | number, duration / rows | snapshots | both |
| `marginal` | `window_size`, `snapshot_every` / `max_rows_between_snapshots` | number, duration / rows | snapshots | both |
| | `lags`, `cross_lags` | rows | a ring | rows; task 536 |
| | `bin_warm_rows` | rows | the rows that set the quantile bins' edges | **open**: a `bin_warm_span` |
| | `bin_refit_rows` | rows | the first rows under `"huber"` | **open**, as `huber_refit_rows` |
| `deco` | `alpha`, `beta` | per observation | no | rows: DCC's recursion is per observation, with no continuous-time form |
| `rcov` | `bandwidth`, `jitter`, `block_rows`, `max_bandwidth`, `preavg_rows`, `noise_stride`, `iv_stride` | rows | a ring sized from `block_rows` | rows: the realized kernel's and pre-averaging's lags and windows are counts of returns, in tick time by their definition; the block itself is closed on the clock, by `group_close` |
| `hmm` | `warm_rows` / **`warm_span`** | rows / number, duration | the seeding rows | **both, from part 2**, held to `kmeans`' 256 MiB (the count has no limit of its own: a 537 point) |
| | `transition`, `transition_prior`, `prior_rows` | per row; a pseudo-count | no | rows: the transition matrix is per observation (a continuous-time chain needs a generator, which no option offers); `prior_rows` is a weight, not a span |
| `corr_change` | `window_rows`, `monitor_rows` | rows | a ring of rows | **open**: a clock window (`window_size`), its rows checked as they come; Wied's and Dette's statistics are defined over the rows of a span, so the clock form is the same test over a data-sized count |
| | `permute_every_rows` / **`permute_every`** | rows / number, duration | no | **both, from part 2** (window kind) |
| | `bandwidth`, `perm_block`, `n_perm` | rows; a count | no | rows: Bartlett lags and a block bootstrap's block are counts of observations |
| `bocpd` | `hazard` | a number of rows on every clock, or a duration | no | both forms, but a number is rows on a numeric clock too, where every other `Span` is clock units: **the coordinator's decision** (below) |
| | `max_run` | rows | is the cap | a bound on state (run lengths are counted in rows) |
| | `warm_rows` / **`warm_span`** | rows / number, duration | the prior's rows | **both, from part 2**, held to its 256 MiB |
| `audit` | `distinct_cap` | count | is the cap | a bound on state |
| `glm` | `refit_rows` / **`refit_span`** | rows / number, duration | the first rows | **both, from this task**: checked as it fills against 256 KB a group |
| | `step_cap` | a bound on a step | no | not a span |
| `hawkes` | `kernel_half_lives`, `window_events` / `window_size` | number, duration / rows | events in the window | both |
| | `kernel_decays` | a rate per clock unit | no | no duration form (a rate): refused on a temporal clock, whose form is `kernel_half_lives` |
| `expectile` | `refit_rows` / **`refit_span`**, `window_rows` / `window_size` | rows / number, duration | the first rows; a window's rows | **both, from this task** for the refit; checked against `max_state_bytes` |
| `garch` | `start_rows` / `start_span` | rows / number, duration | the start's rows | both |
| | `truncation`, `horizon` | rows | FIGARCH's ring | rows: FIGARCH's lags and a forecast's steps are of a recursion per observation |

## What this task built

- **`rheon_core::clock_span`**: `span_reached(now, first, span)`, the end of
  a span begun at one row, on the exact stamps; and `HeldCap`, the run-time
  check over a model's `over`, `warned` and `warning`. `garch`'s start,
  `hawkes`' and `expectile`'s clock windows read both from there, to the
  bit (every golden, fixture and contract test unchanged).
- **`refit_span` for `glm` and `expectile`**: the first rows' refit over a
  span of the clock. The rows from the target's first learned row through
  the first row at least the span after it, once they hold a row per
  coefficient (`garch`'s `start_span` convention: the row that reaches the
  span is the span's last). So on a clock of one unit a row, a span of `D`
  holds `D + 1` rows and is `refit_rows = D + 1` to the bit. The other
  convention -- refit before the first row past the span, on the rows
  strictly inside -- would make `D` equal `refit_rows = D`, but needs a
  prediction read ahead to run the refit on a copy, an exact IRLS fit a
  read, so it was not taken. Checked as the buffers fill (`expectile`
  against `max_state_bytes`, `glm` against its fixed 256 KB a group).
- **Tests**: the core's `refit_clock_tests.rs` in each model (both sides of
  the end to the nanosecond, the wait for a row per coefficient, `D + 1`
  rows on a row and a temporal clock under decay and without, weight-0
  rows, the cap's warning and refusal, a state through the file's
  encoding, the refusals); `tests/test_refit_span.py` (the definition by
  Polars' datetime arithmetic on an irregular clock, with spans a
  millisecond either side of a row; the three clocks; zero-weight rows;
  chunking; a saved state; both caps through the bank; the refusals);
  fixtures `glm_refit_clock`, `expectile_refit_clock`, `bank_refit_span`.
- **No output moves** where a spec gives what it gave before, and no state
  file moves a byte: the new fields are last and written only when set.
  No `SCHEMA_VERSION` bump: both models came after the last release
  (2.0.0, schema 67), within schema 72, which no release has written.
- **Cost**: per row unchanged (arm64 release, best of 7, 60,000 rows at 20
  features under the default refit: `glm` 1826 base against 1821 ns,
  `expectile` 1838 against 1821 ns, load 6-10).

## Part 2 (2026-11-22)

- **Shared**: `clock_span::start_due` (a start's end, its first stamp the
  first row's, and its least rows) and `clock_span::Place` (a model's place
  on the clock: the stamp handed, or the clock summed from the steps for a
  caller that hands none), beside `span_reached` and `HeldCap`. The cadences
  use the existing `since::Since`, as `solve_every` and `prune_every` do.
  The refit buffer shared by `huber`, `quantile` and `sgd` takes an `OPEN`
  cap under a span: not reserved at a count's size, grown as rows come,
  sized by `held_bytes` against `SPAN_REFIT_CAP` (256 KB a group).
- **Gained**: `kmeans` `warm_span`, `update_every`, `split_merge_every`;
  `hmm` and `bocpd` `warm_span`; `corrchange` `permute_every`; `huber` and
  `sgd` `huber_refit_span`; `quantile` `quantile_refit_span`. Each: core
  tests (`D + 1` or `D` rows on the row clock and a 1 s temporal clock to
  the bit, both sides of the end to the nanosecond, the wait for the least
  rows, the cap's warning and refusal, a state through the named encoding,
  the refusals, a state without the form writing none of its keys), an entry
  in `tests/test_rows_and_clock.py` (the Polars datetime oracle for the
  starts, a duration against its seconds on a number clock, zero-weight
  rows, chunking, a save and load inside the span, the refusals), the
  sweeps (CLI, temporal clock, temporal property streams), a fixture.
- **Released names**: task 196 renamed `update_every`, `split_merge_every`
  and `permute_every` to `_rows` "so that a clock form can come later under
  the plain name"; the plain names are now those clock forms, out of
  `RENAMED` and `test_renames.py`.
- **Cost**: the count's paths are unchanged (still reserved at full size);
  per row level (arm64 release, best of 7, 20 features, refit 500: `huber`
  179.8 against 179.0 ns, `sgd` 134.1 against 131.6 ns).
- **Not built, and why**:
  - `marginal`'s `bin_warm_rows` and `bin_refit_rows`: one held list serves
    the warm-up and then the refit (the refit runs at the freeze when its
    count is the warm-up's or less), the sharded path defers rows, and the
    hold's bound is the spec's `bin_budget` (256 MiB by default) where the
    refit's reservation against `refit_budget` is a count's. A span form
    needs a cap and a reservation that agree (256 MiB a group would spend
    the default `refit_budget` of 320 MiB on one group); that choice is
    the coordinator's before it is built.
  - `corrchange`'s `window_rows` and `monitor_rows`: the statistics are
    defined over a span's row count `T` (the Bartlett bandwidth `⌊ln T⌋`,
    Wied and Galeano's simulated critical value for a history of `m` rows
    and a monitoring period, the permutation null over two windows of
    equal rows). A clock span gives each span its own `T`, so the sequential
    kind's critical value would be solved again per cycle (seconds each at
    `boundary_gamma` near its bound), and the window kind's two windows
    would differ in size. Whether to build a clock form, and with which of
    those, is a design question, not a missing parameter.

## Open, for the coordinator

1. **`bocpd`'s `hazard` as a number on a numeric clock** is a count of rows,
   where every other `Span` is the clock's units there. Making it clock
   units would move every output of a numeric-clock `bocpd` that gives a
   number, so it is the user's call: keep it (and say in the docs that a
   number is rows on every clock, a duration time), or move it with a
   CHANGELOG entry.
   The coordinator, 2026-11-22: left as it is for now, taken to the user.
2. **What remains** (part 2's "Not built, and why"): `marginal`'s
   `bin_warm_rows` and `bin_refit_rows` (a cap and a refit reservation that
   agree), and `corrchange`'s `window_rows` and `monitor_rows` (statistics
   defined over a span's row count). The lags (`ew_cov`, `marginal`, the
   diagnostics' and `horizon_rows`) are task 536's exact clock lags.
3. **Settled (the coordinator, 2026-11-22):** `glm`, `huber`, `quantile`
   and `sgd` take `max_state_bytes` under 537's rule (256 KB a group by
   default, at most 64 MiB, warned of at half and refused at the cap),
   replacing the fixed span cap; `glm`'s count is held to it too, its
   default count unchanged at the default cap.
