# Study 532: resumable time-series features held in a state

PLAN task 532 (the widened form): every feature a stream needs, computed from a
state the bank can save and load between any two rows, offered beside Polars'
own expressions. This study is numpy and Python only. It writes each feature as
a state machine, checks it against Polars 2.0.0 (the version `uv.lock` pins on
`main`), saves and loads the state between chunks, and measures the state's
bytes and the cost per row. Nothing in the crates is changed.

Code: `studies/study_532/` (`features.py` the state machines, `oracle.py` the
data and the bit and ulp measures, `runs.py` the drivers, `report.py` the
tables). Entry point: `python scripts/study_532.py` (about three minutes;
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
   Mirroring costs at most two f64 more state than the plain forms (the sum's
   two error terms) and is the only way to be bit-equal.
6. Cost (Table 7): the Python state machines take 0.16 to 0.74 us per row for
   a lag or an EW feature and 6.2 to 10 us per row for the five rolling outputs
   of one window; Polars takes 0.3 to 14 ns and 30 to 81 ns for the same on one
   group, 25 to 158 ns under `over` with 5 groups. A Rust
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
  row's decay run from the last row with a value, as for a null. Its own
  output is what Polars gives a null row: the window's value for a rolling
  feature, null for an EW feature. A bank that wants a value there can read the
  EW state as it stands (the mean does not move on a null), an output rule that
  changes no state and no bits.
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
`n <= ddof`, 0 when negative. A null is counted and not pushed. A non-finite value is pushed as 0.0 and counted;
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
older row; a null is counted and not inserted. Extra state saved: none; the deque is the unique chain of the
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

### Versions

| python | numpy | polars | machine |
|---|---|---|---|
| 3.12.3 | 2.5.5 | 2.0.0 | Linux x86_64 |

### Table 1: each output against Polars, one pass, 20000 rows, 5 groups (`over`)

Nulls about 3% of x, zero-weight rows about 4% (masked to null), and in the
second column set NaN, +inf and -inf about 0.4% of x.

| output | valid | validity diff | bit diff | max ulp | valid (non-finite) | validity diff | bit diff |
|---|---|---|---|---|---|---|---|
| shift(1) | 18602 | 0 | 0 | 0 | 18614 | 0 | 0 |
| shift(5) | 18583 | 0 | 0 | 0 | 18596 | 0 | 0 |
| diff(1) | 17308 | 0 | 0 | 0 | 17335 | 0 | 0 |
| diff(5) | 17288 | 0 | 0 | 0 | 17308 | 0 | 0 |
| pct_change(2) | 17298 | 0 | 0 | 0 | 17312 | 0 | 0 |
| rolling_sum[w=20,m=20] | 4496 | 0 | 0 | 0 | 4628 | 0 | 0 |
| rolling_mean[w=20,m=20] | 4496 | 0 | 0 | 0 | 4628 | 0 | 0 |
| rolling_var[w=20,m=20] | 4496 | 0 | 0 | 0 | 4628 | 0 | 0 |
| rolling_min[w=20,m=20] | 4496 | 0 | 0 | 0 | 4628 | 0 | 0 |
| rolling_max[w=20,m=20] | 4496 | 0 | 0 | 0 | 4628 | 0 | 0 |
| rolling_sum[w=500,m=1] | 20000 | 0 | 0 | 0 | 20000 | 0 | 0 |
| rolling_mean[w=500,m=1] | 20000 | 0 | 0 | 0 | 20000 | 0 | 0 |
| rolling_var[w=500,m=1] | 19995 | 0 | 0 | 0 | 19995 | 0 | 0 |
| rolling_min[w=500,m=1] | 20000 | 0 | 0 | 0 | 20000 | 0 | 0 |
| rolling_max[w=500,m=1] | 20000 | 0 | 0 | 0 | 20000 | 0 | 0 |
| rolling_sum[w=7,m=3,ddof=0] | 19990 | 0 | 0 | 0 | 19990 | 0 | 0 |
| rolling_mean[w=7,m=3,ddof=0] | 19990 | 0 | 0 | 0 | 19990 | 0 | 0 |
| rolling_var[w=7,m=3,ddof=0] | 19990 | 0 | 0 | 0 | 19990 | 0 | 0 |
| rolling_min[w=7,m=3,ddof=0] | 19990 | 0 | 0 | 0 | 19990 | 0 | 0 |
| rolling_max[w=7,m=3,ddof=0] | 19990 | 0 | 0 | 0 | 19990 | 0 | 0 |
| rolling_sum_by[50i,right,m=1] | 19984 | 0 | 0 | 0 | 19986 | 0 | 0 |
| rolling_mean_by[50i,right,m=1] | 19984 | 0 | 0 | 0 | 19986 | 0 | 0 |
| rolling_var_by[50i,right,m=1] | 19759 | 0 | 0 | 0 | 19761 | 0 | 0 |
| rolling_min_by[50i,right,m=1] | 19984 | 0 | 0 | 0 | 19986 | 0 | 0 |
| rolling_max_by[50i,right,m=1] | 19984 | 0 | 0 | 0 | 19986 | 0 | 0 |
| rolling_sum_by[200i,left,m=3] | 19336 | 0 | 0 | 0 | 19340 | 0 | 0 |
| rolling_mean_by[200i,left,m=3] | 19336 | 0 | 0 | 0 | 19340 | 0 | 0 |
| rolling_var_by[200i,left,m=3] | 19336 | 0 | 0 | 0 | 19340 | 0 | 0 |
| rolling_min_by[200i,left,m=3] | 19336 | 0 | 0 | 0 | 19340 | 0 | 0 |
| rolling_max_by[200i,left,m=3] | 19336 | 0 | 0 | 0 | 19340 | 0 | 0 |
| rolling_sum_by[30i,both,m=2] | 19759 | 0 | 0 | 0 | 19761 | 0 | 0 |
| rolling_mean_by[30i,both,m=2] | 19759 | 0 | 0 | 0 | 19761 | 0 | 0 |
| rolling_var_by[30i,both,m=2] | 19759 | 0 | 0 | 0 | 19761 | 0 | 0 |
| rolling_min_by[30i,both,m=2] | 19759 | 0 | 0 | 0 | 19761 | 0 | 0 |
| rolling_max_by[30i,both,m=2] | 19759 | 0 | 0 | 0 | 19761 | 0 | 0 |
| rolling_sum_by[30i,none,m=1] | 19776 | 0 | 0 | 0 | 19778 | 0 | 0 |
| rolling_mean_by[30i,none,m=1] | 19776 | 0 | 0 | 0 | 19778 | 0 | 0 |
| rolling_var_by[30i,none,m=1] | 19556 | 0 | 0 | 0 | 19559 | 0 | 0 |
| rolling_min_by[30i,none,m=1] | 19776 | 0 | 0 | 0 | 19778 | 0 | 0 |
| rolling_max_by[30i,none,m=1] | 19776 | 0 | 0 | 0 | 19778 | 0 | 0 |
| rolling_sum_by[12i,right,m=4] | 13933 | 0 | 0 | 0 | 13976 | 0 | 0 |
| rolling_mean_by[12i,right,m=4] | 13933 | 0 | 0 | 0 | 13976 | 0 | 0 |
| rolling_var_by[12i,right,m=4] | 13933 | 0 | 0 | 0 | 13976 | 0 | 0 |
| rolling_min_by[12i,right,m=4] | 13933 | 0 | 0 | 0 | 13976 | 0 | 0 |
| rolling_max_by[12i,right,m=4] | 13933 | 0 | 0 | 0 | 13976 | 0 | 0 |
| rolling_sum_by[3i,left,m=0] | 20000 | 0 | 0 | 0 | 20000 | 0 | 0 |
| rolling_mean_by[3i,left,m=0] | 11143 | 0 | 0 | 0 | 11135 | 0 | 0 |
| rolling_var_by[3i,left,m=0] | 2029 | 0 | 0 | 0 | 2049 | 0 | 0 |
| rolling_min_by[3i,left,m=0] | 11143 | 0 | 0 | 0 | 11135 | 0 | 0 |
| rolling_max_by[3i,left,m=0] | 11143 | 0 | 0 | 0 | 11135 | 0 | 0 |
| ewm_mean[hl=10,adjust=True,ignore_nulls=False,m=1] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 |
| ewm_mean[hl=10,adjust=False,ignore_nulls=False,m=1] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 |
| ewm_mean[hl=4,adjust=True,ignore_nulls=True,m=5] | 18587 | 0 | 0 | 0 | 18598 | 0 | 0 |
| ewm_std[hl=10,adjust=True,bias=False,ignore_nulls=False] | 18602 | 0 | 0 | 0 | 18613 | 0 | 0 |
| ewm_var[hl=10,adjust=False,bias=True,ignore_nulls=False] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 |
| ewm_std[hl=3,adjust=False,bias=False,ignore_nulls=True] | 18602 | 0 | 0 | 0 | 18613 | 0 | 0 |
| ewm_mean_by[25i] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 |

### Table 2: chunked, the state saved to bytes and loaded at every cut (5000 rows, 5 groups)

A cut saves every group's state of every feature to bytes, drops it and loads it
from the bytes. Compared with one Polars pass over the whole frame, and with the
uncut stream.

| chunking | chunks | outputs | equal to Polars to the bit | equal to the uncut stream | seconds |
|---|---|---|---|---|---|
| 1 chunk | 1 | 57 | 57 | 57 | 0.5 |
| 2 chunks | 2 | 57 | 57 | 57 | 0.5 |
| 17 random chunks | 17 | 57 | 57 | 57 | 0.5 |
| ~300 random chunks | 293 | 57 | 57 | 57 | 1.8 |
| every row | 5000 | 57 | 57 | 57 | 21.8 |

### Table 3: forms that do not keep Polars' order of operations, in ulps (20000 rows, one series, no nulls)

Against the Polars expression (the state machines above are 0 ulps from it), and
against the exact value where one is computable (fsum). `level` is the series' mean
level; the steps are the same, so a higher level means more cancellation.

| level | window | form | max ulp | median ulp | max relative |
|---|---|---|---|---|---|
| 1000 | 20 | rolling_sum: Polars vs exact (fsum) | 1 | 0.0 | 2.0e-16 |
| 1000 | 20 | rolling_sum: plain add/sub vs Polars | 34 | 8.0 | 4.9e-15 |
| 1000 | 20 | rolling_sum: plain add/sub vs exact | 34 | 8.0 | 4.9e-15 |
| 1000 | 20 | rolling_sum: Kahan rebuilt from the ring every 97 rows vs Polars | 1 | 0.0 | 2.0e-16 |
| 1000 | 20 | rolling_mean: plain sum / w vs Polars | 28 | 9.0 | 5.0e-15 |
| 1000 | 20 | rolling_var: Polars vs exact | 2676 | 142.0 | 3.5e-13 |
| 1000 | 20 | rolling_var: Welford add/remove vs Polars | 44897707 | 263685.0 | 8.5e-09 |
| 1000 | 20 | rolling_var: Welford add/remove vs exact | 44897896 | 264037.0 | 8.5e-09 |
| 1000 | 20 | rolling_var: sum and sum of squares vs Polars | 493954378 | 6624971.0 | 5.9e-08 |
| 1000 | 500 | rolling_sum: Polars vs exact (fsum) | 1 | 0.0 | 2.2e-16 |
| 1000 | 500 | rolling_sum: plain add/sub vs Polars | 42 | 19.0 | 8.1e-15 |
| 1000 | 500 | rolling_sum: plain add/sub vs exact | 41 | 19.0 | 7.9e-15 |
| 1000 | 500 | rolling_sum: Kahan rebuilt from the ring every 97 rows vs Polars | 1 | 0.0 | 2.2e-16 |
| 1000 | 500 | rolling_mean: plain sum / w vs Polars | 43 | 19.0 | 8.1e-15 |
| 1000 | 500 | rolling_var: Polars vs exact | 951 | 115.0 | 1.2e-13 |
| 1000 | 500 | rolling_var: Welford add/remove vs Polars | 156753 | 9963.0 | 1.8e-11 |
| 1000 | 500 | rolling_var: Welford add/remove vs exact | 156803 | 9977.0 | 1.8e-11 |
| 1000 | 500 | rolling_var: sum and sum of squares vs Polars | 4123430 | 341071.0 | 6.1e-10 |
| 1000 | hl=10 | ewm_mean(adjust=False): y += a(x - y) vs Polars | 0 | 0.0 | 0.0e+00 |
| 1000 | hl=25i | ewm_mean_by: exp(dt ln 0.5 / hl) for 0.5 ** (dt/hl) vs Polars | 0 | 0.0 | 0.0e+00 |
| 1e+06 | 20 | rolling_sum: Polars vs exact (fsum) | 1 | 0.0 | 1.9e-16 |
| 1e+06 | 20 | rolling_sum: plain add/sub vs Polars | 35 | 21.0 | 6.5e-15 |
| 1e+06 | 20 | rolling_sum: plain add/sub vs exact | 35 | 21.0 | 6.5e-15 |
| 1e+06 | 20 | rolling_sum: Kahan rebuilt from the ring every 97 rows vs Polars | 1 | 0.0 | 1.9e-16 |
| 1e+06 | 20 | rolling_mean: plain sum / w vs Polars | 56 | 34.0 | 6.5e-15 |
| 1e+06 | 20 | rolling_var: Polars vs exact | 1740320 | 97291.0 | 2.0e-10 |
| 1e+06 | 20 | rolling_var: Welford add/remove vs Polars | 91761845381 | 531574778.0 | 1.1e-05 |
| 1e+06 | 20 | rolling_var: Welford add/remove vs exact | 91762246041 | 531619197.0 | 1.1e-05 |
| 1e+06 | 20 | rolling_var: sum and sum of squares vs Polars | 198672273016490 | 5131978818905.0 | 2.8e-02 |
| 1e+06 | 500 | rolling_sum: Polars vs exact (fsum) | 1 | 0.0 | 1.2e-16 |
| 1e+06 | 500 | rolling_sum: plain add/sub vs Polars | 22 | 5.0 | 2.6e-15 |
| 1e+06 | 500 | rolling_sum: plain add/sub vs exact | 22 | 5.0 | 2.6e-15 |
| 1e+06 | 500 | rolling_sum: Kahan rebuilt from the ring every 97 rows vs Polars | 1 | 0.0 | 1.2e-16 |
| 1e+06 | 500 | rolling_mean: plain sum / w vs Polars | 23 | 5.0 | 2.7e-15 |
| 1e+06 | 500 | rolling_var: Polars vs exact | 670390 | 82320.0 | 9.5e-11 |
| 1e+06 | 500 | rolling_var: Welford add/remove vs Polars | 417190507 | 15938546.0 | 4.6e-08 |
| 1e+06 | 500 | rolling_var: Welford add/remove vs exact | 416893091 | 15923786.0 | 4.6e-08 |
| 1e+06 | 500 | rolling_var: sum and sum of squares vs Polars | 578840126721 | 43250860963.0 | 1.0e-04 |
| 1e+06 | hl=10 | ewm_mean(adjust=False): y += a(x - y) vs Polars | 0 | 0.0 | 0.0e+00 |
| 1e+06 | hl=25i | ewm_mean_by: exp(dt ln 0.5 / hl) for 0.5 ** (dt/hl) vs Polars | 0 | 0.0 | 0.0e+00 |

### Table 4: repeated times on the clock (`rolling_*_by`, closed right)

Polars gives each row of a repeated time the window of the last row at that time,
rows after it included. Read as of the row instead (`asof_ties`), the rows of a
repeat that are not its last differ; the last row of each repeat is equal.

| output | rows | rows not last of their time | bit diff, all rows | bit diff, last rows of each time | largest relative gap |
|---|---|---|---|---|---|
| rolling_sum_by | 20000 | 7997 | 7997 | 0 | 5.00e-01 |
| rolling_mean_by | 20000 | 7997 | 7997 | 0 | 2.00e-03 |
| rolling_var_by | 20000 | 7997 | 7996 | 0 | 6.67e-01 |
| rolling_min_by | 20000 | 7997 | 976 | 0 | 5.20e-03 |
| rolling_max_by | 20000 | 7997 | 975 | 0 | 5.30e-03 |

### Table 5: the features that read a libm result, and what one ulp of it moves

`ewm_*` read `alpha = 1 - exp(-ln 2 / half_life)`, computed once; `ewm_mean_by`
reads `0.5 ** (dt / half_life)` at every row. The same libm gives the same bits (the
tables above, on one machine). Another libm may differ in the last bit; the rows
below put one ulp into that result and measure the outputs.

| output | parameters | perturbation | max ulp | median ulp | max relative |
|---|---|---|---|---|---|
| ewm_mean adjust=True | hl=3 | exp(-ln 2 / hl) +1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_mean adjust=False | hl=3 | exp(-ln 2 / hl) +1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_std adjust=True | hl=3 | exp(-ln 2 / hl) +1 ulp | 419 | 1.0 | 7.0e-14 |
| ewm_var adjust=False bias | hl=3 | exp(-ln 2 / hl) +1 ulp | 915 | 2.0 | 1.3e-13 |
| ewm_mean adjust=True | hl=3 | exp(-ln 2 / hl) -1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_mean adjust=False | hl=3 | exp(-ln 2 / hl) -1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_std adjust=True | hl=3 | exp(-ln 2 / hl) -1 ulp | 606 | 1.0 | 9.2e-14 |
| ewm_var adjust=False bias | hl=3 | exp(-ln 2 / hl) -1 ulp | 1230 | 2.0 | 1.4e-13 |
| ewm_mean adjust=True | hl=10 | exp(-ln 2 / hl) +1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_mean adjust=False | hl=10 | exp(-ln 2 / hl) +1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_std adjust=True | hl=10 | exp(-ln 2 / hl) +1 ulp | 390 | 2.0 | 6.9e-14 |
| ewm_var adjust=False bias | hl=10 | exp(-ln 2 / hl) +1 ulp | 566 | 5.0 | 7.0e-14 |
| ewm_mean adjust=True | hl=10 | exp(-ln 2 / hl) -1 ulp | 2 | 0.0 | 2.2e-16 |
| ewm_mean adjust=False | hl=10 | exp(-ln 2 / hl) -1 ulp | 1 | 0.0 | 2.2e-16 |
| ewm_std adjust=True | hl=10 | exp(-ln 2 / hl) -1 ulp | 647 | 2.0 | 7.2e-14 |
| ewm_var adjust=False bias | hl=10 | exp(-ln 2 / hl) -1 ulp | 1022 | 5.0 | 1.4e-13 |
| ewm_mean adjust=True | hl=100 | exp(-ln 2 / hl) +1 ulp | 4 | 1.0 | 5.8e-16 |
| ewm_mean adjust=False | hl=100 | exp(-ln 2 / hl) +1 ulp | 3 | 1.0 | 4.4e-16 |
| ewm_std adjust=True | hl=100 | exp(-ln 2 / hl) +1 ulp | 226 | 47.0 | 3.0e-14 |
| ewm_var adjust=False bias | hl=100 | exp(-ln 2 / hl) +1 ulp | 443 | 76.0 | 5.2e-14 |
| ewm_mean adjust=True | hl=100 | exp(-ln 2 / hl) -1 ulp | 3 | 1.0 | 4.3e-16 |
| ewm_mean adjust=False | hl=100 | exp(-ln 2 / hl) -1 ulp | 5 | 1.0 | 5.8e-16 |
| ewm_std adjust=True | hl=100 | exp(-ln 2 / hl) -1 ulp | 173 | 48.0 | 2.4e-14 |
| ewm_var adjust=False bias | hl=100 | exp(-ln 2 / hl) -1 ulp | 460 | 99.0 | 5.7e-14 |
| ewm_mean_by, mirrored (pow) | hl=25i, dt 1..399 | none | 0 | 0.0 | 0.0e+00 |
| ewm_mean_by, exp(dt ln 0.5 / hl) | hl=25i, dt 1..399 | the formula | 2 | 0.0 | 4.0e-16 |
| ewm_mean_by, each pow result moved 1 ulp at random | hl=25i, dt 1..399 | +-1 ulp, p=1/2 a row | 2 | 0.0 | 4.1e-16 |
| 1 - alpha itself: pow vs exp form | hl=25i, dt 1..399 | the formula | 16 | 2.0 | 2.4e-15 |

### Table 6: state bytes, saved form, per group and per column

`r` is the rows the ring holds: at most `w` for a row-counted window; for a span,
the rows inside it, at most twice that while windows short of `min_samples` are
skipped, and never above the hard cap. Measured: the largest saved size seen while
streaming 20000 rows with no nulls through one group, at a save every 37 rows.

| feature | window | bytes (formula) | formula at full window | measured peak |
|---|---|---|---|---|
| shift(1) | n=1 | 8 + 8n + ceil(n/8) | 17 | 17 |
| shift(5) | n=5 | 8 + 8n + ceil(n/8) | 49 | 49 |
| diff(1) | n=1 | 8 + 8n + ceil(n/8) | 17 | 17 |
| diff(5) | n=5 | 8 + 8n + ceil(n/8) | 49 | 49 |
| pct_change(2) | n=2 | 8 + 8n + ceil(n/8) | 25 | 25 |
| rolling[w=20,m=20] | w=20 | 28 + 8r + ceil(r/8) + 4 x 16 + 24 + 4 | 283 | 283 |
| rolling[w=500,m=1] | w=500 | 28 + 8r + ceil(r/8) + 4 x 16 + 24 + 4 | 4183 | 4183 |
| rolling[w=7,m=3,ddof=0] | w=7 | 28 + 8r + ceil(r/8) + 4 x 16 + 24 + 4 | 177 | 177 |
| rolling_by[50i,right,m=1] | 50i | 32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4 | (r varies) | 495 |
| rolling_by[200i,left,m=3] | 200i | 32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4 | (r varies) | 1398 |
| rolling_by[30i,both,m=2] | 30i | 32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4 | (r varies) | 399 |
| rolling_by[30i,none,m=1] | 30i | 32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4 | (r varies) | 399 |
| rolling_by[12i,right,m=4] | 12i | 32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4 | (r varies) | 253 |
| rolling_by[3i,left,m=0] | 3i | 32 + 16r + ceil(r/8) + 4 x 16 + 24 + 4 | (r varies) | 189 |
| ewm_mean[hl=10,adjust=True,ignore_nulls=False,m=1] |  | 24 | 24 | 24 |
| ewm_mean[hl=10,adjust=False,ignore_nulls=False,m=1] |  | 24 | 24 | 24 |
| ewm_mean[hl=4,adjust=True,ignore_nulls=True,m=5] |  | 24 | 24 | 24 |
| ewm_std[hl=10,adjust=True,bias=False,ignore_nulls=False] |  | 48 | 48 | 48 |
| ewm_var[hl=10,adjust=False,bias=True,ignore_nulls=False] |  | 48 | 48 | 48 |
| ewm_std[hl=3,adjust=False,bias=False,ignore_nulls=True] |  | 48 | 48 | 48 |
| ewm_mean_by[25i] |  | 25 | 25 | 25 |

In memory the derived structures add, at most: the variance's two stacks hold
each window value once, as a (weight, mean, dp) triple of 24 bytes in the front
stack or an 8-byte value in the back one; the min and max deques hold at most `r`
(index, value) pairs each. Their worst case, a monotone series, measured:

| series | w | min deque peak | max deque peak | front stack peak | back stack peak |
|---|---|---|---|---|---|
| 0, 1, 2, ... (rising) | 500 | 500 | 1 | 499 | 500 |

### Table 7: cost per row (best of 3; Python state machine over 50000 rows, Polars over 1000000 rows, one group, then 5 groups with `over`)

| feature | outputs | Python state, ns/row | Polars, ns/row | Polars over 5 groups, ns/row |
|---|---|---|---|---|
| shift(1) | 1 | 255 | 0.3 | 29.2 |
| shift(5) | 1 | 254 | 0.3 | 25.1 |
| diff(1) | 1 | 289 | 1.5 | 28.1 |
| diff(5) | 1 | 157 | 2.1 | 27.7 |
| pct_change(2) | 1 | 262 | 3.0 | 30.1 |
| rolling[w=20,m=20] | 5 | 7,016 | 29.9 | 98.0 |
| rolling[w=500,m=1] | 5 | 8,928 | 32.5 | 70.2 |
| rolling[w=7,m=3,ddof=0] | 5 | 6,583 | 30.1 | 84.1 |
| rolling_by[50i,right,m=1] | 5 | 9,971 | 81.2 | 146.9 |
| rolling_by[200i,left,m=3] | 5 | 9,021 | 59.6 | 134.0 |
| rolling_by[30i,both,m=2] | 5 | 8,242 | 78.7 | 139.4 |
| rolling_by[30i,none,m=1] | 5 | 7,272 | 63.7 | 143.8 |
| rolling_by[12i,right,m=4] | 5 | 6,233 | 73.8 | 157.6 |
| rolling_by[3i,left,m=0] | 5 | 6,237 | 58.3 | 113.3 |
| ewm_mean[hl=10,adjust=True,ignore_nulls=False,m=1] | 1 | 242 | 8.8 | 27.8 |
| ewm_mean[hl=10,adjust=False,ignore_nulls=False,m=1] | 1 | 247 | 8.2 | 28.7 |
| ewm_mean[hl=4,adjust=True,ignore_nulls=True,m=5] | 1 | 248 | 8.5 | 33.3 |
| ewm_std[hl=10,adjust=True,bias=False,ignore_nulls=False] | 1 | 608 | 11.4 | 38.2 |
| ewm_var[hl=10,adjust=False,bias=True,ignore_nulls=False] | 1 | 518 | 11.8 | 42.9 |
| ewm_std[hl=3,adjust=False,bias=False,ignore_nulls=True] | 1 | 743 | 14.4 | 33.0 |
| ewm_mean_by[25i] | 1 | 230 | 14.2 | 43.4 |

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
