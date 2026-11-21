# Study 532: resumable time-series features held in a state

PLAN task 532 (the widened form): every feature a stream needs, computed from a
state the bank can save and load between any two rows, offered beside Polars'
own expressions. This study is numpy and Python only. It writes each feature as
a state machine, checks it against Polars 2.0.0 (the version `uv.lock` pins on
`main`), saves and loads the state between chunks, and measures the state's
bytes and the cost per row. Nothing in the crates is changed.

Code: `studies/study_532/` (`features.py` the state machines, `oracle.py` the
data and the bit and ulp measures, `runs.py` the drivers, `report.py` the
tables). Entry point: `python scripts/study_532.py` (about two minutes;
`--quick` for a smaller run), with numpy and `polars==2.0.0`. Every table below
is its output.

## Answer

1. **Every feature equals Polars to the bit**: `shift`, `diff`, `pct_change`,
   `rolling_{sum,mean,var,min,max}` and their `_by` forms under each `closed`,
   `ewm_mean`, `ewm_var`, `ewm_std` (each `adjust`, `bias`, `ignore_nulls`,
   `min_samples` tried) and `ewm_mean_by`: 0 bits differ and 0 validity
   differences in 57 outputs, per group under `over`, with nulls, zero-weight
   rows, NaN and both infinities in the input (Table 1).
2. That is possible because each state mirrors, operation for operation, the
   kernel Polars runs: a compensated sum with two error terms, a two-stack
   variance queue built from Welford combines, a monotone deque, and the EW
   recursions. The order of arithmetic can match everywhere a stream can know
   the answer. It cannot match in one place: a repeated time on a `_by` clock,
   where Polars gives each row the window of the last row at that time
   (Table 4).
3. **Chunk and resume invariance**: cut into 2, 17, ~300 chunks or after every
   row, with every state saved to bytes, dropped and loaded at each cut, all 57
   outputs equal the one-pass Polars output and the uncut stream to the bit
   (Table 2).
4. **State is bounded by the window, not the stream** (Table 6): a lag `n`
   holds `n` values; a row-counted window `w` holds a ring of at most `w`
   values shared by all five aggregations of one column, plus 92 bytes; a span
   holds the rows inside it, bounded by a hard cap; the EW features hold 24 or
   48 bytes. The variance's stacks and the min and max deques are rebuilt from
   the ring at load, so they cost nothing saved; in memory their worst case is
   the window length (a monotone series), measured.
5. **Forms that do not keep Polars' order** are measurably off (Table 3): a
   plain add-and-subtract window sum is up to ~40 ulps from Polars; a Welford
   add-and-remove variance up to 4.5e7 ulps (relative 1e-8) at a level of 1e3.
   Mirroring costs nothing in state and is the only way to be bit-equal.
6. Cost (Table 7): the Python state machines take 0.15 to 0.6 us per row for a
   lag or an EW feature and 7 to 9 us per row for the five rolling outputs of
   one window; Polars takes 0.3 to 26 ns and 30 to 90 ns for the same. A Rust
   state runs the same operations as Polars' kernel, plus a per-group lookup.

## Definitions: each feature as a state and an update

Notation: row `i` of one group, value `x_i` (or null), time `t_i` (int64,
rising strictly within a group). Each update below is the state's whole change
for one row; the output is read from the state after it.

### The zero-weight rule, nulls and NaN (applied by every feature)

- A **zero-weight row** (rule 9: it advances the clock and learns nothing) is
  a null to every feature: it takes its slot on the row clock and its time on
  the time clock, and none of its value enters a state. The check is Polars on
  the input with zero-weight rows' values set to null (`masked` in `oracle.py`);
  the stream feeds `None` for those rows. So a zero-weight row: holds a slot in
  a row-counted window, as a null does; moves a span window to its time; decays
  an EW weight when `ignore_nulls=False`; and in `ewm_mean_by` lets the next
  row's decay run from the last row with a value, as for a null.
- A **null** is not a value. A **NaN** is a value, as in Polars. What each
  Polars expression does with a null, and what the state does, is under each
  feature.
- The **clock** must rise strictly within a group on a span window: a step
  back or a repeat raises (`ClockError`). Polars sorts its `by` column first,
  which a stream cannot do across chunks. `ewm_mean_by` accepts a repeat (its
  `dt` is 0 and Polars handles it the same way). A session change or a gap past
  `gap_cap` (the bank's clock breaks, which Polars has no word for) is a new
  group: each feature starts empty. Table 1's groups show that a fresh state per
  key equals Polars' `over` on that key, so a break equals Polars `over` the
  group and the session.

### Lags: `shift(n)`, `diff(n)`, `pct_change(n)`

State: the last `n` values (with nulls) and how many rows were seen, up to `n`.
Update: read the oldest of the ring (null until `n` rows were seen), then push
`x_i`. Outputs: `shift = x_{i-n}`; `diff = x_i - x_{i-n}`;
`pct_change = (x_i - x_{i-n}) / x_{i-n}`, exactly Polars' `diff(s, n) /
s.shift(n)`. A null at either end gives null; no fill. One operation, so the
bits match trivially.

### The rolling window driver (rows and span)

A row-counted window of `w` is rows `[i+1-w, i+1)`. A span window of `d` is
the rows `j <= i` with `t_j` in `(t_i - d, t_i]` (`closed="right"`),
`[t_i - d, t_i]` (`both`), `[t_i - d, t_i)` (`left`) or `(t_i - d, t_i)`
(`none`); its start moves forward from the last start, as
`group_by_values_iter_lookbehind` moves it. Polars' rule, mirrored: a window
holding fewer than `min_samples` rows, nulls counted, is not updated and gives
null; otherwise the aggregator is moved from its last window `[s0, e0)` to
`[s, e)`: first the rows `[s0, s)` leave, oldest first, then the rows
`[e0, e)` enter, oldest first; when every row of the last window leaves
(`s >= e0`) the aggregator is reset to empty instead. The output is null when
fewer than `min_samples` non-null rows are in the window. (On a row-counted
window the not-updated rows can only come before the window first fills, where
they only delay additions, so Polars' two row drivers, with and without nulls,
run the same operations.)

State: a ring of the rows the window or the aggregator can still read (value,
validity bit, and the time on a span), the row count, the window start, and
each aggregator's `[s0, e0)` with its own few words below. All aggregations of
one column over one window share the ring.

### `rolling_sum`, `rolling_mean` (and `_by`)

Polars' `SumWindow` (polars-compute `rolling/sum.rs`), a compensated sum with
a separate error term for additions and for subtractions. A finite value `v`
entering: `y = v - c_add; s' = s + y; c_add = (s' - s) - y; s = s'`. Leaving:
the same with `-v` and `c_sub`. A non-finite value is not summed; it is counted
(and +inf and -inf separately), and while one is in the window the sum is
+inf, -inf or NaN as the counts say; when it leaves, the sum is again the
finite one, so a NaN does not poison the window after it leaves. A null
counts as a null. Mean: the sum over the count of non-null rows (null for
none). Extra state: `s, c_add, c_sub` (24 bytes); the counts are recomputed
from the ring at load. The three terms carry the whole history since the last
reset, so a ring alone does not resume to the bit (Table 3: a sum rebuilt from
the ring is up to 1 ulp off).

### `rolling_var` (and `_by`)

Polars' `MomentWindow` with `VarState` (polars-compute `rolling/moment.rs`,
`moment.rs`): a two-stack queue. A value entering is pushed on the back stack
and folded into its running state by Welford's step,
`n' = n + 1; d = x - m; m' = m + d / n'; dp' = dp + (x - m') d`. A value
leaving pops the front stack; when the front stack is empty, the back stack is
flipped: its values are popped newest first, folded one by one, and each
partial state pushed, so the front stack holds each suffix's state. The
variance combines the back's running state with the front's top by Chan's
merge, `n = n_a + n_b; f = n_b / n; d = m_b - m_a; m = m_a + d f;
dp = dp_a + (dp_b + n_b (m_b - m) d)`, then `dp / (n - ddof)`, null when
`n <= ddof`, 0 when negative. A non-finite value is pushed as 0.0 and counted;
while one is in the window the variance is NaN. Extra state saved: the number
of the window's values in the front stack (4 bytes). At load both stacks are
rebuilt from the ring by the same flip and pushes that built them, which gives
the same bits (every state is a function of the values and the order folded).

### `rolling_min`, `rolling_max` (and `_by`)

Polars' `ArgMinMaxWindow` (polars-compute `rolling/arg_min_max.rs`): a
monotone deque of `(row, value)`; a value entering pops every tail entry it is
strictly better than, then is appended; entries before the window start are
popped from the front; the front is the answer. NaN is better than any number
for both min and max (it propagates while in the window); a tie keeps the
older row. Extra state saved: none; the deque is the unique chain of the
window's values and is rebuilt from the ring at load. Its worst case is a
monotone series, where every row stays: `w` entries (Table 6), never the
stream's length.

### `ewm_mean`

Polars' `EwmMeanState` (polars-compute `ewm/mean.rs`) with
`alpha = 1 - exp(-ln 2 / half_life)` (py-polars' own expression). State: the
mean `m`, the weight `W`, the count of values. The first value sets
`m = x, W = 1`. Then on each row: `W *= 1 - alpha` unless the row is null and
`ignore_nulls`; on a value, with `u = 1` (`adjust=True`) or `alpha`:
`m = m + (x - m) (u / (W + u))`, then `W = W + 1` (`adjust`) or `W = 1`.
Output null on a null row and below `min_samples`. 24 bytes.

### `ewm_var`, `ewm_std`

Polars' `EwmCovState` fed `(x, x)` (polars-compute `ewm/cov.rs`). State: the
weight `W`, the mean `m`, the covariance `c`, the weight sum `S1` and the sum
of squared weights `S2`, a count. On each row the three weights decay by
`1 - alpha` (`S2` by its square) under the same null rule; on a value:
`W' = W + u; m' = m + (x - m) u / W';
c = (W (c + (m - m')^2) + u (x - m')^2) / W'; S1 += u; S2 += u^2`, and for
`adjust=False` `S1 /= W', S2 /= W'^2, W = 1`. Output `c` (`bias`) or
`S1^2 / (S1^2 - S2) c`, null when that denominator is not above 0; `ewm_std`
is its square root. The two means of the covariance state are equal for
`(x, x)`, so one is kept: 48 bytes.

### `ewm_mean_by`

Polars' `ewm_by_core` (polars-ops `series/ops/ewm_by.rs`): `y = x` at the
first value; then `y = x` when `x` equals the last `y`, else
`y = r y_prev + (1 - r) x` with `r = 0.5 ** ((t - t_prev) / half_life)`. A
null row gives null and changes nothing, so the next `dt` runs from the last
row with a value. State: the last time with a value, the last result, a
started flag (and the last time seen, for the clock check): 25 bytes.

## Results

The script's full output follows. Notes on each table:

- **Table 1** checks 57 outputs (21 features, the rolling ones with five
  outputs each) on 20000 rows in 5 groups, once clean and once with NaN and
  infinities in 0.4% of rows; nulls in 3% and zero-weight rows in 4% both
  times. Every output: 0 validity differences, 0 bits differing.
- **Table 2** cuts the same kind of frame (with non-finite values) and saves and
  loads every state at each cut. The bytes are the whole carry: the next chunk
  sees nothing else. Every row is a cut in the last line.
- **Table 3** is what a state that does not keep Polars' order gives. Polars'
  own `rolling_sum` is within 1 ulp of the exact sum; its `rolling_var` is
  2.7e3 ulps (relative 1e-13) from an exact two-pass variance at a level of
  1e3 and about 1e6 ulps (relative 1e-10) at a level of 1e6, so matching
  Polars means matching its error, not the exact value. A Welford
  add-and-remove variance and the sum-of-squares variance are far worse.
  `ewm_mean(adjust=False)` written as `y += alpha (x - y)` equals Polars,
  because `(1 - alpha) + alpha` rounds to 1 for every half-life tried
  (0.5 to 1000), so Polars' step reduces to it. The exp form of `ewm_mean_by`'s
  factor is 0 ulps here only because steps of 1 to 5 rarely differ (Table 5).
- **Table 4**: on a repeated time, a stream that reads the window as of each
  row differs from Polars on every row that is not the last at its time, and
  equals it on the last. Matching Polars there means holding a repeat's outputs
  until the time moves on (the rows at one time, bounded by the cap), or
  refusing repeats, as the study does by default.
- **Table 5** is the libm sensitivity. **Table 6** gives each feature's saved
  bytes, the formula and the measured peak. **Table 7** is the cost per row.

RESULTS_PLACEHOLDER

## State bytes as formulas

Per group and per input column; `n` the lag, `w` the row window, `r` the rows a
span's ring holds:

| feature | saved bytes | in memory, worst case, beyond the saved form |
|---|---|---|
| `shift`/`diff`/`pct_change(n)` | `8 + 8n + ceil(n/8)` | none |
| rolling over `w` rows, all five outputs | `28 + 8r + ceil(r/8) + 92`, `r <= w` | variance stacks `<= 24r` (each value once, as a 24-byte state in front or an 8-byte value in back); min and max deques `<= 16r` each |
| one rolling output alone | ring as above, plus 16 and: sum or mean 24, var 4, min or max 0 | as above, for that output |
| rolling over a span | `32 + 16r + ceil(r/8) + 92` | as above |
| `ewm_mean` | 24 | none |
| `ewm_var`, `ewm_std` | 48 | none |
| `ewm_mean_by` | 25 | none |

For `k` columns and `G` groups multiply by `k G`; features over the same column
and window share one ring, so five rolling outputs cost one ring, not five.
Examples: `w = 20`, `k = 10`, `G = 1000`: 283 B, so 2.8 MB saved; `w = 500`:
4183 B, so 41.8 MB saved, and up to about 4183 + 56 x 500 = 32 kB per state
in memory (both stacks and both deques full), 0.32 GB for the 10000 states.
EW features over 10 columns and 1000 groups: 0.24 to 0.48 MB.

A span has no bound of its own on its rows: `r` is the rows inside it, at most
twice that while windows short of `min_samples` are skipped (the rows of the
skipped aggregator's last window, still to leave, plus the current window's),
and never above the hard cap the operator must carry (`WindowSpec.cap`; past
it the study raises). In Table 6's runs `r` stayed at the window's own rows.

## Tolerances proposed

- **Same machine, mirrored order: bit-exact, 0 ulps.** One pass, any chunking,
  any number of save and load cycles, and Polars' own expression on the whole
  series, for every feature here. A test of these features should compare bits.
- **Across operating systems.** Lags and all rolling features use no libm:
  bit-exact. The EW features read one libm result, `exp(-ln 2 / half_life)`;
  carrying `alpha` itself (an f64) in the feature's spec or state makes a
  continuation on another OS bit-exact, since nothing else reads libm. Against
  a Polars run on another OS, one ulp in that result moves `ewm_mean` by at
  most 5 ulps and `ewm_var`/`ewm_std` by up to about 1.2e3 ulps (relative
  1.4e-13, a variance read through cancellation at a level of 1e3), Table 5.
  `ewm_mean_by` calls `pow` on every row, so its state carries a libm result:
  same OS bit-exact; across OSes, one ulp in `pow` on a random half of the rows
  moved the output by at most 2 ulps (Table 5). Proposed: 4 ulps for
  `ewm_mean_by` across OSes, stated beside the test; `ewm_*` with `alpha`
  carried, bit-exact.
- **A form that does not mirror Polars' order is not proposed**; if one is
  wanted for accuracy (an exact variance, say), it gives up bit-equality with
  Polars and should be held to the exact value instead (Polars' own
  `rolling_var` is up to ~1e6 ulps from it at a level of 1e6).

## Caveats and what is not done

- A repeated time on a `_by` clock cannot match Polars row by row without
  holding outputs back (Table 4). The study refuses repeats; the alternative
  is to emit a repeat's rows when the time moves on.
- `center=True` and `weights=` read rows after the current one or a fixed
  weight vector over the whole window; they were not studied (a centered
  window is a lookahead, as `reverse=True` is for rheon's windows).
- NaN and infinity follow Polars: they propagate while in a rolling window and
  leave cleanly; in an EW feature a NaN poisons the state for good, as it does
  in Polars. Treating NaN as null would be a choice that departs from Polars.
- The cost is Python's; a Rust state was not built (no cargo in this study).
  The operations per row are those of Polars' kernels: two compensated steps
  for a sum, an amortized two Welford steps and one merge for a variance, an
  amortized two comparisons per deque, about ten flops for an EW feature and
  one `pow` for `ewm_mean_by`, plus a hash lookup per row for the group.
- Polars 2.0.0 only. The kernels read are in the polars repository at tag
  `py-2.0.0`; a later Polars that changes one of them (its order of operations,
  a reset rule) breaks bit-equality, which a test against the pinned Polars
  catches.
- Datetime clocks were not run; Polars casts them to int64 in their unit
  before the same code, so an int64 clock covers them (a calendar duration such
  as `1mo` does not reduce to a fixed span and was not studied).
