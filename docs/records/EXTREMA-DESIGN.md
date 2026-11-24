<!-- kind: record -->
# Extrema on the clock: the design record

The design for task 561, part 1, written 2026-11-23 on branch
`vm/task-561-extrema` from `main` at 555b45e5, then built in parts 2 to 4
the same day. The coordinator's answers to the questions at the end, and
where the build differs from the design, are in "What was built", last.

## What exists, and what this adds

**The window engine has no extremum today.** `rheon.polars.ops` has
`ewm_mean`, `ewm_sum`, `ewm_rate`, `ewm_quantile`, `ewm_var`, `ewm_std` and
`increment`; the core (`crates/rheon-polars/src/windows.rs`, `OpKind`) has
the same list, and a grep of the crates, the package and the tests for
`rolling_min`, `rolling_max`, `cum_max`, `running_max` and `drawdown` finds
nothing. The window operators still listed as open work (min, max, an exact
quantile, a rank, the count-weighted mean) are unbuilt. So this design *is*
the min and max of that list, and the build should close both items' min
and max at once rather than build them twice.

**What the engine already gives, and this reuses rather than rebuilds:**

- the *kernel queue*: a two-stack queue per (direction, half-life, window,
  `closed`), shared by every operator with those settings, each row
  carrying a partial aggregate so a row leaves the window without a
  subtraction;
- which rows a window holds: Polars' `rolling_*_by` rule, every `closed`,
  edges decided on integer nanoseconds on a temporal clock, every row at a
  stamp sharing one backward window;
- the clock events: a gap past `gap_cap` or a session change *cuts* every
  window (the operator's `partial` says what a cut window gives); a reset
  (a step back past `restart_after_step_back`, or a session change under
  `session_gap="reset"`) *discards* them (null);
- looking ahead under `reverse=True`, with the rows held until their
  windows close;
- no `window_size`: the window is every row since the stretch began, kept
  in O(1) as one running aggregate (`Queue::total`), with no rows held;
- formulas around the operators (element-wise only), formula targets in a
  spec, `like=`, save and resume, the chunk-invariance tests.

**What is new:** a min and a max statistic on that queue, and one new
event, `since`, an element-wise boolean of the row that starts an
operator's window over. The running extremum since an event (the high-water
mark of a drawdown) is the max with no `window_size` and a `since`. The gap
below it is a formula around it, not an operator.

## (a) The interface

Two operators, beside the others in `rheon.polars.ops`, named for the
Polars functions they are held to without the `_by`, as `ewm_mean` is
`ewm_mean_by`'s (the clock is the call's, not the operator's):

```python
rp.rolling_max(
    input,                       # column name or element-wise expression, as every operator
    *,
    window_size=None,            # clock units, a duration on a temporal clock; None: since the stretch began
    closed="right",              # "right" | "left" | "both" | "none", Polars' rule
    min_samples=1,               # fewest rows with a value; 1 to 2**32 - 1
    partial=None,                # "keep" | "null" | "drop"; needs window_size
    reverse=False,               # look ahead; needs window_size
    since=None,                  # element-wise boolean: a true row starts the window over
)
rp.rolling_min(...)              # the same parameters
```

- **No `half_life`.** An extremum has no weights to decay ((j) below). The
  core keys the operator's kernel with an infinite half-life, so it shares
  a queue with the even-weighted sums of the same window, `closed` and
  direction, and adds no queue of its own where one exists.
- **The window span is on the call's clock**, as every operator's:
  numbers on a numeric clock, durations (`"10s"`, a `timedelta`,
  `pl.duration`) on a temporal one, and rows where the call has no clock
  column. An edge is decided from the difference of two rows' raw clocks,
  exact in nanoseconds, so the window holds what `rolling_max_by` holds.
- **Output** is a `Float64` column, as every operator's: the extremum of
  the values the window holds, or null where it holds fewer than
  `min_samples` of them, where `partial="null"` meets a cut window, where a
  forward window is still open at the input's end, and after a reset. Every
  row at one stamp gets the same backward value, so under `closed="right"`
  the value is emitted at the next distinct stamp, as for the other
  operators. An integer input is read as a double, so an integer beyond
  2**53 in magnitude is rounded, where Polars' `rolling_max_by` returns an
  `Int64` (a documented difference; the oracle tests use integers within
  2**53).
- **`window_size=None` is the running form**: every row since the stretch
  began, or since the last `since` event, whichever is later. Polars'
  `rolling_max_by` requires a window; rheon's operators all accept none, and
  this keeps that. Polars' own running form is `cum_max`, which differs in
  three places ((c) below), so the running form is held to `rolling_max_by`
  over each run, not to `cum_max`.
- **Drawdown is composed, not built in.** With a stream's `equity` column:

  ```python
  peak = rp.rolling_max("equity", since=pl.col("new_period"))   # the high-water mark
  out = rp.stream.with_windows(
      lf,
      peak=peak,
      gap=pl.col("equity") - peak,            # 0 at a new high, negative below it
      clock="ts", gap_cap="7d", group="account",
  )
  ```

  The gap is element-wise around one operator, which `with_windows`
  accepts. A relative drawdown (`equity / peak - 1`) and a trailing-window
  drawdown (`rolling_max(..., window_size="30d")`) are the same composition.
  The running *maximum drawdown* (the most negative gap so far) is an
  extremum of an operator's output, so it is two calls: the gap as a
  column in one `with_windows`, and `rp.rolling_min("gap", since=...)` in a
  second, as the guide already says for windowing an operator's output.

## (b) The `since` event

**What triggers it.** `since` is an element-wise boolean expression of the
row (a column name or a `pl.Expr`, held to the same rule as an operator's
input: no `shift`, no cumulative or rolling function, no `over`, no other
operator). A row where it is true is an event; null reads as false; a
non-boolean dtype is refused by name when the query is built. It acts on
that operator alone, unlike a session change, which ends every window in
the call: a drawdown's high-water mark can restart each period while a
volatility window in the same call runs on. Under `group` each group's
events are its own, which a session column built outside the call cannot
give (a session derived by a cumulative sum over the whole stream would
change for every group when one group's event arrives).

**The name.** `reset` already means something in the engine: a discard,
whose windows give null. An event here does the opposite, it starts a run
that gives a value at once, so the parameter is `since` (the window holds
the rows *since* the last event). See the questions.

**Events are read by stamp.** Every row at one stamp shares one backward
window, and an event keeps that: a stamp that holds an event, at any of its
rows, starts the window at that stamp. Every row of the stamp is in the new
run, and rows at earlier stamps are out. So the value never depends on
where within a stamp the event row falls, and the rows of one stamp never
get two values.

**What the running extremum does on an event.** It discards the previous
run's extremum and count, and the new run holds the event stamp's rows. On
the event stamp the max under `closed="right"` or `"both"` is the stamp's
own max (so the gap is 0 at the stamp's max row). Under `"left"` or
`"none"` the stamp's own rows are outside its window, and the new run holds
no row yet, so the event stamp's value is null (fewer than `min_samples`
values), never the previous run's extremum.

**Before the first value after an event** the run holds no value, so every
row until the first row with a value gives null under `min_samples=1`.
The previous run's extremum is never carried across an event.

**With a `window_size`**, an event cuts the backward window as the start of
a stretch does: the window holds the rows in `(t - w, t]` at or after the
last event's stamp, and a window that the event cut before its full span is
partial, under the operator's `partial` (`"keep"` gives the extremum since
the event; `"null"` withholds it until a whole window has passed since the
event, checked by the span, not by a count of rows).

**Clock breaks and resets still act.** A gap past `gap_cap` or a session
change starts a new stretch, and the running form starts over there as
every operator without a window does; a reset discards it (null). A user
who wants the high-water mark to survive a long gap sets a `gap_cap` that
covers it. See the questions.

**Looking ahead.** In the first build `since` is for backward windows and
the running form only, and `since` with `reverse=True` is refused by name.
A forward window cut at the next event is well defined (it closes when the
event's stamp arrives), but it needs its own rule for whether that window
is whole or partial, and nobody has asked for it. See the questions.

## (c) Ties and nulls, as Polars treats them

The pin: `pyproject.toml` declares `polars>=1.37.0,<3`; `uv.lock` builds
and tests against py-polars 2.1.0 (and Rust polars 0.55.2 in `Cargo.toml`).
Checked on 2026-11-23 by scripts run against py-polars 2.1.0 (the
repository's environment) and 1.37.0, 1.38.1, 1.39.3, 1.40.1 and 1.41.1
(scratch environments), on hand-built frames and on 400 random frames of
values drawn from {0.0, -0.0, 1.0, -1.0, null} with repeated stamps and
windows of 1 to 4 under `closed="right"` and `"left"`, each compared with
two candidate rules by bits:

| case | Polars `rolling_{min,max}_by` | Polars `cum_max` / `cum_min` | rheon's operators |
|---|---|---|---|
| a null row | skipped in the aggregate; the row gets the window's value (2.1.0; from 1.38.1, the earliest checked that accepts one) | null at that row | skipped; the row gets the window as it stands (the engine's rule): Polars' rolling rule |
| a null in the input on 1.37.0 | refused: `InvalidOperationError`, "not yet supported for series with null values" | | the oracle tests with nulls skip below 1.38.1 by `needs_polars`, naming why |
| NaN | propagates: a NaN in the window makes the result NaN | skipped by comparison, except a leading NaN, which is NaN | read as missing, as for every operator: equal to Polars on the input with the NaN set to null |
| `inf`, or beyond 1e100 | included as a value | included | read as missing, as for every operator (the engine's bound, so a later product stays finite): equal to Polars on the input with it set to null |
| equal values, distinguishable only as `0.0` and `-0.0` | the **earliest** row of the window wins, for min and max alike, on 1.37.0 and 2.1.0 (0 of 400 random frames differ from that rule; the latest-row rule differs on 139 to 159) | the latest row wins on 2.1.0, the earliest on 1.37.0: it changed inside the declared range | the earliest row of the window, Polars' rolling rule, to the bit |
| rows at one stamp | one window for every row of the stamp | row by row | one window a stamp, Polars' rolling rule |
| empty window | null, under any `min_samples` including 0 | | null; `min_samples` is 1 or more, as every operator's |
| `window_size` 0 | refused | | refused, as every operator's |

**The tie rule follows from the merge.** The aggregate is combined oldest
first and keeps the older value unless the newer is strictly beyond it.
That is associative, so the two stacks' partial aggregates give the
earliest row of the window wherever the stacks turned over, and it is the
whole of Polars' rule, since the only equal doubles with different bits are
the two zeros (NaN is missing here). Looking ahead, the rule is the mirror
(the row farthest from the read row, which is the latest), so a forward
window equals the backward one over the reversed stream to the bit.

**`cum_max` is not the oracle** for the running form, for three reasons the
table shows: it gives null at a null row, it treats NaN by comparison, and
its tie rule moved between 1.37.0 and 2.1.0. Polars' `rolling_max_by` over
each run, with a window longer than the run, is the running form's Polars
oracle.

## (d) State bytes, against `max_state_bytes`

The extremum's aggregate fits the queue's narrow accumulator (three
doubles: the extremum, an unused mass, the count of values), so it adds no
width and no new field to a queue holding only sums, means, rates or
extrema. Per group, in memory, with `m` extremum operators on one kernel:

| form | bytes a group | example |
|---|---|---|
| running (`window_size=None`), with or without `since` | `24 m`: one aggregate per operator, no rows held | one operator over 10,000 groups: 240 KB in all |
| windowed, backward, its own kernel | `R (40 + 64 m) + 24 m`, `R` the rows inside the window: 40 a row for the shared row fields (sequence, clock, raw clock, interval end, anchor), 64 a row per operator (5 values and 3 partial sums) | `R` = 1,000, one operator: 104 KB; the 256 KB default cap is reached at `R` = 2,520 |
| windowed, sharing a kernel with other operators | `64 m` more a row on the queue already there | |
| windowed, on a queue that also holds a variance (six wide) | `112` a row per operator: the queue's width is one for all its operators | |
| `reverse=True` | as windowed; the rows waiting for their windows to close are held by the engine as for every forward operator | |

A saved state writes each double as 9 bytes of msgpack, so the file is at
most `9/8` of these. A `Vec` that grew may hold up to twice its length in
capacity until the engine releases it. `R` on a clock is the data's to
number: a burst of rows inside one `window_size` is held whole, as for every
windowed operator today. On the row clock `R` is the window.

**The engine has no `max_state_bytes` today.** The models whose spans hold
rows check their bytes against a cap (256 KB a group by default, 64 MiB at
most, a warning at half, a refusal past it); `with_windows` and the formula
targets do not, for any operator. The extremum is bounded by its window, as
the rule allows, but a burst can make that window large. The design checks
the extremum's queue against `max_state_bytes` by the same rule as the
models, and recommends that the check be the engine's, for every
operator, so one cap covers a call. See the questions; this is the one
point that changes what existing users see.

**The alternative not taken: a monotone deque.** It holds only the rows no
later row dominates, about `ln R + 0.58` of them on exchangeable data, at
32 bytes each, but all `R` on a falling (for a max) stream. A bound is set
by the worst case, so it bounds a max no lower than `32 R` against the
queue's `64 R` an operator, and it needs its own copy of every edge rule
(`closed`, the cut and discard events, the nanosecond edges, the forward
close) that the queue already has under test. `windows.rs` is 249,591 bytes
of its 250 KB cap, so new code goes into its own module either way. The
build measures both on the budget harness and takes the deque only if the
memory it saves is wanted.

## (e) Chunk invariance

The engine reads rows one at a time in input order. The extremum's merge
is exact (it copies a value, never computes one) and associative, and it
depends on the rows' order, not on where a chunk ends. A `since` event is a
row's own value, read with the row. So no chunking can change a bit. The
build holds it by:

- `windows.rs::chunking_changes_no_bit`, extended with both extrema in every
  form (windowed back and ahead, running, with `since`), at chunks of 1, 2,
  3, 7, 50 and 499 rows, compared by `to_bits`, so a `-0.0` turning into
  `0.0` fails;
- `test_windows.py::test_chunking_changes_nothing`, extended the same way,
  with streams that put an event as the first and as the last row of a
  chunk, and a stamp split across two chunks with its event in the second;
- a hypothesis property over random streams (repeated stamps, nulls, NaN,
  both zeros, gaps, sessions, groups, events) at random chunk sizes against
  one chunk, by bits.

## (f) Save and load to the bit

The state is the queue's arenas, as for every operator, so a save writes
the extremum and its count as they are, and a load restores them. No
arithmetic touches a value and no libm call is made (the kernel's
half-life is infinite, so its discount is exactly 1), so a resumed run
equals the unbroken one to the bit on any OS, with no tolerance.

- `test_a_save_and_load_at_every_row_is_one_run`, and its Rust twin,
  extended to both extrema in every form, including a save between an
  event and the first value after it, and a save inside a stamp split by
  the event.
- **The layout changes**, since a state may now name the new operators and
  carry a kernel's `since`. The precedent is the quantile's: the windows
  state's version moves from 12 to 13, the new fields are written only
  where an operator uses them (so a state without an extremum keeps its
  bytes, and a version-12 state loads as a 13 without one), and the bank's
  schema moves with it, as its tripwire test requires. From 1.0 the current
  fixture sets are copied to `v<N>/`, the loader is the identity for a
  12, and the current sets are regenerated.
- **A new fixture**, `with_windows_extrema`, freezes every form a layout
  can take: a windowed max and min backward, a forward max with rows
  waiting, a running max with a `since` event mid-run and a run holding no
  value yet, a window holding both zeros, and a running form under groups.
  Its test decodes the file and holds each form non-empty.

## (g) Oracles

| form | oracle | how | to |
|---|---|---|---|
| windowed, backward | Polars `rolling_max_by` / `rolling_min_by` | every `closed`; windows from below one step to above the stream; integer, float and `Datetime` clocks with duration windows; repeated stamps; nulls (`needs_polars("1.38.1")`); NaN and `inf` against Polars on the input with them nulled; both zeros | bits |
| windowed, across gaps, sessions, groups | Polars, per stretch | `.over([group, stretch])`, the stretch numbered by the gaps and sessions the test placed, the brute-force tests' construction | bits |
| `partial="null"` and `"drop"` | Polars and the span | Polars' value where the window's span lies inside its stretch, null or dropped where it reaches before it | bits |
| windowed, forward | Polars over the reversed stream | rows and clock reversed, `closed` mirrored, the identity the other operators are held to | bits |
| running, with `since` | Polars `rolling_max_by` over each run | the run numbered by a cumulative sum of the events read per stamp, a window longer than the run | bits |
| running, a second opinion | numpy `np.fmax.accumulate` over each run's values, and pandas `groupby(run).cummax()` | missing values as `-inf` for numpy (then null), forward-filled; compared at each stamp's last row, since both run row by row where the operator gives a stamp one value | values (both zeros compare equal) |
| every form | the definition | `windows.rs`'s brute force: the extremum of the values in each row's window, computed from the whole stream with no queue | bits |

**No third-party library computes the running extremum since an arbitrary
event as one call**, so its oracles are a library over each run (Polars,
numpy, pandas) with the run cut by hand, which is the definition. Drawdown
libraries (`empyrical`, `quantstats`) compute a maximum drawdown of a return
series, a different quantity and a scalar, and would add a dependency for
nothing the above misses. In Rust the brute force is the oracle; the
`crate::oracle` helpers are for matrix results and have nothing to add.

## (h) Rows of weight 0

`with_windows` takes no weight column, so it has no row of weight 0: every
row is in the clock and every value is read, as Polars' rolling functions
read them, and the extrema keep that.

Inside a spec, a formula target's window core reads every row of the
frame today, a row of weight 0 included, and on the row clock that row
counts a unit; the window operators were left Polars-matching when the
models' clocks stopped counting such rows. The rule for new blocks says a
row of weight 0 counts on no clock. For an extremum the natural reading is:
a row of weight 0 is read as missing (it moves no extremum and gets the
window as it stands), counts no unit on the row clock, and its `since` is
not an event. But an extremum in a formula beside an `ewm_mean` must read
the same clock, so the rule belongs to the engine, for every operator, in
one change, not to this operator alone. See the questions.

## (i) Early data: no row counts

An extremum of the values seen is exact, not an estimate: it does not
converge, and it has no finite-sample error to wait out. So it needs no
warm-up and has none. The two conditions that withhold it are conditions,
not counts:

- **no value yet**: fewer than `min_samples` values in the window (1 by
  default, the definition's own condition: the extremum of nothing is
  undefined); null;
- **a window not yet whole**, under `partial="null"`: the window's span
  reaches before its stretch began or before the last event, read from the
  clock; null, or the row dropped under `"drop"`.

The reason for each null is one of the documented cases in (a), as for
every operator; a column carries no reason. A test of a feature that
"settles" after so many rows would be a row count tuned on test data and is
not written; the tests check the conditions at their edges (a window
exactly one span after the event, one nanosecond short of it).

## (j) Decay and window

**The window is feasible and built**: a trailing or leading span on the
clock, with every `closed`, `partial` and `reverse`.

**Decay is not, and there is no `ewm_max` or `ewm_min`.** No definition is
standard, and each candidate fails a property an extremum must keep:

- *An exponentially weighted order statistic* collapses to the plain one.
  A weight moves a quantile by shifting mass between values, but the
  maximum is the largest value with any weight at all: numpy's weighted
  `inverted_cdf` quantile at level 1 is the window's max under every
  finite half-life. So "decayed max" so read is the window's max, which
  `rolling_max` already gives.
- *A decayed peak* (a peak meter's `y_t = max(x_t, λ^Δt · y_{t−1})`, or a
  linear release `y_t = max(x_t, y_{t−1} − r Δt)`) is not shift-equivariant:
  `max` of `x + c` is `max(x) + c` for any `c`, but a peak decayed toward 0
  of `x + c` is not the decayed peak of `x` plus `c`. Its answer depends on
  where the column's zero is and on its sign (a negative peak decays
  *upward*), and the release rate is a free parameter in the column's
  units. Audio meters and some trading indicators use one or another, with
  no agreed form, and no library computes one we could hold it to.
- *A soft maximum* (`τ log Σ w_j e^{x_j/τ}`) is a different quantity with a
  free temperature, and it calls `exp` and `log` on every row.

A reader who wants a high that forgets smoothly has two exact tools: the
window (`rolling_max` with a `window_size`, which forgets at a known age),
and `ewm_quantile` at a level below 1 (such as 0.99), where decay does move
the answer, since mass then matters.

## Build outline (for part 2)

In order, each with its tests written first: the `since` input and kernel
key; `Stat::Max` and `Stat::Min` on the queue, in a module of their own;
the Python operators and their docstrings (the math stated, Raises listed);
the oracles of (g); chunking and resume (e), (f); the windows state
version, the bank schema, the kept fixture sets and the new fixture; a
budget case (time a row and peak memory, against `ewm_sum` on the same
window) with its measured cost; the guide's table of operators
(`docs/WINDOWS.md`) and the ops module's table; the CHANGELOG.

## Questions for the coordinator

1. Names: `rp.rolling_max` / `rp.rolling_min` (Polars' names without `_by`, as `ewm_mean`) with `window_size=None` as the running form; recommend yes, over a separate `rp.cum_max`, whose Polars namesake differs on nulls, NaN and ties.
2. The event parameter's name: `since=` (a true row starts the window over); recommend `since`, since `reset` already means a discard to null in the engine.
3. Events read by stamp (an event anywhere in a stamp starts the run at that stamp, every row of it in); recommend yes, so a stamp never gets two values.
4. Under `closed="left"`/`"none"` the event stamp's value is null (its own rows are outside); recommend yes, Polars' rule applied to the new run.
5. A gap past `gap_cap` or a session change restarts the running form, as every operator without a window; recommend yes, documented, with `since` and a wide `gap_cap` as the control.
6. `since` with `reverse=True`: recommend refused by name in this build, added when a forward use is asked for.
7. Tie rule: the earliest row of the window (Polars' rolling rule, both zeros only), the mirror looking ahead; recommend yes, to the bit.
8. NaN and `inf` read as missing, unlike Polars' rolling (NaN propagates, `inf` counts); recommend yes, the engine's rule for every operator, documented as a difference.
9. `max_state_bytes`: recommend one engine-wide cap on `with_windows` and formula targets (256 KB a group, warn at half, refuse past it), landed as its own change first with a measurement of which existing documented runs it would refuse; if the default would refuse any, the default is the user's call.
10. Rows of weight 0 in a spec's formula target (read as missing, no unit on the row clock, no event): recommend one engine-wide change for every operator, raised with the user, not an extremum-only rule.
11. Two-stack queue (shared with the engine, 64 bytes a row an operator) over a monotone deque (32 bytes a kept row, the same worst case); recommend the queue, measure both in the build.
12. An age-of-extremum output (clock since the peak: a drawdown's duration), which Polars has no rolling form of; recommend deferring to its own row, held to the definition.
13. Close the open window-operator row's min and max with this build (one implementation); recommend yes, leaving its quantile, rank and count-weighted mean open.

## What was built (parts 2 to 4)

**The coordinator's answers** (2026-11-23), which override the design where
they differ: the names `rp.rolling_max` / `rp.rolling_min`, no
`window_size` being the running form; the event keyword `since=`, acting on
its whole stamp, its own stamp null under `closed="left"` and `"none"`; a
gap past `gap_cap` or a session change restarts the running form, as for
every operator; `since` with `reverse=True` refused, saying why; ties to the
earliest row, held by the 400-frame test against Polars; NaN and infinities
by the engine's rule, the difference from Polars stated and tested on both
sides; `max_state_bytes` **not** engine-wide, only on the new operators'
queues; the existing two-stack queue, measured against a monotone deque;
no age-of-extremum output (its own task) and no decayed extremum; drawdown a
formula; rows of weight 0 left as the engine reads them, recorded by a
failing test if the engine breaks the rule; this build closes the **min
and max of task 518** (its exact quantile, rank and count-weighted mean
stay open).

**The core** (`crates/rheon-polars/src/windows/extrema.rs`). `Stat::Max`
and `Stat::Min`, `OpKind::{RollingMax, RollingMin, RrollingMax,
RrollingMin}`, on a kernel whose half-life is infinite (refused otherwise).
The extremum is kept beside the queue's sums, as a quantile's sketch is,
not in the sums themselves: each extremum keeps one double per `front` row
(the extremum from that row through `front`'s newest, built at the
turnover), one for `back`, and for the running form the run's extremum and
count. The queue's sums still count the values for `min_samples` in a
windowed queue; the running form counts its own, since an event lets the
run go without touching the shared sums. The join is `pick`: the older
value unless the newer is strictly beyond it, looking back; the newer
unless the older is strictly beyond, looking ahead.

**`since`.** A kernel carries the row input of its event (`KernelDef.since`,
part of the kernel's key, so an event never acts on another operator's
queue). An event read at any row of a stamp: with a window, the rows of the
earlier stamps are evicted; with none, the run before the stamp is let go
and the stamp's own rows, kept apart (`cur`), stay. Under `"left"` and
`"none"` the stamp's value, read as it began, becomes empty. Because an
event may come at any row of its stamp, a `since` kernel's rows wait for
the next distinct stamp under every `closed` (`KernelDef::waits`), so a
stamp never gets two values. A window is whole once its span has passed
since the stretch began and since the last event (`extrema::complete`,
decided on the raw clocks). The plain loop does not take a core with a
`since` kernel or a windowed extremum (`Derived::push_only`); every such
row is `push`'s.

**`max_state_bytes`** is an extremum's own parameter (default 256 KB a
group, at most 64 MiB), bounding the queue it sits on. Without a clock
column the window's rows are known (`floor(w) + 2` at most), so a window
past the cap is refused when the query is built, its cost stated; on a
clock the row that takes a group's queue past it is refused
(`Refusal::OverCap`), naming the row, the rows held, the bytes a row and
the cap. No existing operator is bounded.

**State bytes, measured** (`extrema::row_bytes`, held by the Python test of
the bound on both sides): a windowed row is `8 (5 + 8 n + e)` bytes for `n`
operators on the queue, `e` of them extrema, or `8 (5 + 14 n + e)` on a
queue holding a variance: **112 bytes** a row for one extremum on its own
window (the design's 104 plus the front double), so the 256 KB default
holds 2,340 rows a group; 184 for a maximum and a minimum sharing one. The
running form is about 70 bytes an extremum in memory (five doubles, two
flags and an empty list) plus the queue's three running sums, whatever the
stream's length.

**Two-stack queue against a monotone deque** (a standalone Rust probe,
`--release`, 2,000,000 rows, best of 5, the queue in the engine's row
layout of 14 doubles, the deque 32 bytes a kept row; x86-64, load about 2-4):

| window, data | two-stack ns a row | its peak bytes | deque ns a row | its peak bytes |
|---|---|---|---|---|
| 1,000 rows, uniform | 9.4 | 112,000 | 12.4 | 704 |
| 1,000 rows, falling | 9.3 | 112,000 | 5.6 | 32,000 |
| 10,000 rows, uniform | 17.1 | 1,120,000 | 12.8 | 864 |
| 10,000 rows, falling | 14.8 | 1,120,000 | 4.8 | 320,000 |

The deque holds far less on exchangeable data and 3.5 times less in its
worst case, and is faster on a long window; the queue keeps the engine's
one tested copy of every edge rule (`closed`, cuts, resets, nanosecond
edges, the forward close, the event). Kept as the coordinator decided; the
deque is the change to make if the memory is wanted.

**Windows state 13** within schema 72, which no release has written: a
kernel's `since`, a queue's `ext` and `since_off`, an operator's
`max_state_bytes`, each written only where there is an extremum, so a 12
loads as a 13 without one. The committed fixtures were moved from 12 to 13
by their version byte and checksums alone (the bank files that embed a
windows state too), keeping their bytes otherwise; `with_windows_extrema`
is new and freezes each form.

**Oracles as built**: Polars' `rolling_max_by` / `rolling_min_by` bit for
bit (every `closed`, integer and datetime clocks, repeated stamps, the
400 frames of both zeros, nulls from 1.38.1), the reversed stream for a
forward window, Polars over each run for `since`, numpy's
`fmax.accumulate` as a second opinion; in Rust the brute force of
`windows.rs` (the extrema added to `all_kernels`, every direction and
`closed`, across gaps, sessions, resets, groups, chunking and resumes) and
`extrema_tests.rs`'s definition of `since` by stamp, resumed at every row.

**Rows of weight 0.** A spec's formula target's window core reads every
row, a row of weight 0 included, for every operator: an extremum there
moves with such a row's value. Recorded as a strict expected failure,
`test_a_weight_zero_row_moves_no_formula_targets_extremum`; the fix is the
engine's, for every operator at once, and is not in this build.
