<!-- kind: record -->
# Threshold-triggered bars: the design (task 562)

Status, 2026-11-23: built as `rp.stream.bars`. The questions in the last
section are answered: each as recommended but the tenth, where the number of
groups is bounded by bytes, not by a count ([Groups and their
bound](#groups-and-their-bound)). The third-party libraries are checked as
oracles ([Oracles](#oracles)), and [As built](#as-built) states the choices
the build made that the design left open.

A **bar** here is one output row that aggregates a run of consecutive input
rows. It closes at the row where a running quantity reaches a threshold.
Counting rows gives tick bars. Summing a size gives volume bars, and summing
price times size gives dollar bars. Summing a signed size gives imbalance
bars, and VPIN is a rolling statistic of the bars' imbalance. This record
says where such an operator sits in the library, the rule that closes a bar,
the arithmetic that keeps the rule exact, the state it carries, its clock,
and the oracles that would hold it.

| section | what it settles |
|---|---|
| [The operator in one example](#the-operator-in-one-example) | the call, as proposed, and the rows it gives |
| [Where it lives](#where-it-lives) | a stream operator beside `refresh_time` and `merge`, not a window operator, and what each chunk gives |
| [The crossing rule](#the-crossing-rule) | the three quantities, the threshold, `>=`, a row that crosses several thresholds, and the carried remainder |
| [Exact arithmetic](#exact-arithmetic) | how the running total is kept exactly, and why a plain double is not enough |
| [The aggregates](#the-aggregates) | which Polars aggregations a bar takes, and how each matches Polars |
| [Groups and their bound](#groups-and-their-bound) | one bar stream per group, the bytes a group holds, and the caps |
| [State across chunks and on save](#state-across-chunks-and-on-save) | what is carried, the chunk boundary, a slice of the output, and the state file |
| [The clock of a bar](#the-clock-of-a-bar) | the close time, ties, rows of weight 0, gaps, sessions and resets |
| [Decay and windows](#decay-and-windows) | why neither applies to the bars, and where they apply instead |
| [Oracles](#oracles) | the Polars construction, the exact one, the imbalance loop, VPIN, and the libraries scanned |
| [Tests and budgets](#tests-and-budgets) | the tests and measurements the build owes |
| [As built](#as-built) | the choices the build made that the design left open |
| [Questions](#questions) | each open choice, with a recommendation, and the answers |

## The operator in one example

**A bar stream is a query that reads a query, one chunk at a time.** The call
below is the proposal, not an existing function, so its names may change.
It builds dollar bars of 50,000 per symbol from `trades` (the example data's
quotes with trades between them; `price`, `quantity` and `side` are null on
a quote):

```python
bars = rp.stream.bars(
    pl.scan_parquet("trades.parquet"),
    clock="ts",
    group="symbol",
    kind="sum",                                            # "count", "sum" or "imbalance"
    quantity=pl.col("price") * pl.col("quantity"),         # dollar bars; null on a quote adds 0
    threshold=50_000.0,
    open=pl.col("mid").first(),                            # each aggregate is a Polars aggregation
    high=pl.col("mid").max(),
    low=pl.col("mid").min(),
    close=pl.col("mid").last(),
    volume=pl.col("quantity").sum(),
    buys=pl.when(pl.col("side") == "buy").then(pl.col("quantity")).otherwise(0.0).sum(),
    vwap=(pl.col("price") * pl.col("quantity")).sum() / pl.col("quantity").sum(),
)
```

Each output row is one closed bar, in the order of the rows that closed
them:

| column | what it holds |
|---|---|
| `symbol` | the group, in its input dtype |
| `bar` | the bar's index within its group, from 0, continued across a resume |
| `time_open`, `time_close` | the clock of the bar's first row and of the row that closed it, in the clock's dtype |
| `thresholds` | how many thresholds the closing row crossed: 1 almost always, more for a row larger than the threshold |
| `carry` | under `kind="sum"`, the part of the running total past the last threshold crossed, carried into the next bar, rounded once to a double |
| `complete` | `false` for a bar cut short by a session change, a gap, a span limit or a reset ([The clock of a bar](#the-clock-of-a-bar)) |
| `open`, `high`, ..., `vwap` | the aggregates, in the order given |

## Where it lives

**A bar changes the number of rows, so it cannot be a window operator.** A
window operator, such as `rp.ewm_sum` under `with_windows`, gives one value
per input row, and its formula reads each row alone. A bar gives one row per
run of input rows, and most rows give none. Two other places were considered
and rejected:

- **A model in the bank, with the bars as a side output**, as `group_close`
  writes one row per closed group beside the per-row output. Bars are wanted
  *before* a bank, as the rows its features and targets are computed on, and
  a side output cannot feed the bank that writes it.
- **A Polars query** of `cum_sum`, `over` and `group_by`, which is the
  oracle below. Under `over` and a `group_by` over the whole input, Polars
  holds every row, which is what a stream must avoid. It also has no form
  for a rule that resets at each bar, which imbalance bars use.

**It is a stream operator, as `refresh_time` and `merge` are.** Each of
those is a Rust operator fed one chunk at a time and read as a lazy source.
Each also changes the row count, saves and loads its state, and resumes under
`.head(n)`. Bars would follow the same shape, in four layers:

| layer | what it holds |
|---|---|
| `crates/rheon-core/src/exact.rs` | the exact running sum ([Exact arithmetic](#exact-arithmetic)): no Polars, no `unsafe`, unit-tested and under `cargo mutants` |
| `crates/rheon-core/src/bars.rs` | one group's bar: the running total, the aggregates' accumulators, the crossing test and the cut, over plain values; the clock arrives as the shared clock state's verdict on each row (a step, a session change, a reset), never as a column |
| `crates/rheon-polars/src/bars.rs` | the columns read from each chunk, the groups, the output frame and its schema, the limited feed, and the state file in the envelope every state file shares, as a new kind |
| `python/rheon/polars/stream.py` | `rp.stream.bars`, which takes a `LazyFrame` or a `DataFrame` and gives back the same kind, read through Polars' IO plugin interface (`register_io_source`), as `refresh_time` is |

A model's checklist (`docs/EXTENDING.md`) does not apply: a bar is not a
model, has no prediction and has no place in a spec.

**Downstream, a bank reads the bars as any other query.** The bars'
`LazyFrame` goes to `lf.rheon.fit_predict` with `clock="time_close"` and the
same `group`. A bar is known only once its closing row has arrived, and
every value in it comes from rows at or before that row. So a bank that
learns on bars at `time_close` still predicts each row before learning from
it. `rp.stream.save_checkpoint` takes the bars' state beside the bank's, so
a crash leaves both at the same point.

**A chunk gives the bars its rows closed, and a chunk that closes none gives
nothing.** Bars are emitted at the closing row, in the chunk that holds that
row. A chunk with no crossing adds its rows to the open bars and yields no
batch: the source reads on to the next chunk. So the operator's output for a
chunk can be empty, and a `.head(n)` reads chunks until it has `n` bars or
the input ends. Memory is one input chunk, the bars it closes, and the state
of the open bars.

## The crossing rule

**The running quantity is one of three kinds.** Each is a configuration of
one operator, not a separate function:

| `kind` | a row adds | a bar closes when | after a close | gives |
|---|---|---|---|---|
| `"count"` | 1 | the bar holds `threshold` rows | 0 | tick bars |
| `"sum"` | the row's `quantity`, at least 0 | the running total reaches the next multiple of `threshold` | the total past the last multiple crossed is carried | volume bars, dollar bars, any sum |
| `"imbalance"` | the row's signed `quantity` | the bar's running sum reaches `threshold` in either direction, `\|sum\| >= threshold` | 0 | tick, volume and dollar imbalance bars |

`quantity` is a column name or an element-wise expression of the row, such
as `pl.col("price") * pl.col("quantity")`. It is checked as a window
formula's input is checked, since Polars computes it chunk by chunk.
A **missing** quantity follows the rule every spec column follows: null, NaN,
infinite or past 1e100 in magnitude. Such a row adds 0, and it is still a
row of the bar, so its other columns enter the aggregates. A quote row in
`trades` is the case: it adds no volume, and its `mid` still moves the bar's
high, low and close. A row of weight 0 is a different thing: it is not in the
bar at all ([The clock of a bar](#the-clock-of-a-bar)).

**The comparison is `>=`.** A volume bar of 10 closes at the trade that
brings the total to exactly 10. `mlfinpy`'s volume bars read it so (its
example closes a bar at a total of exactly 20), as does the inequality of
López de Prado's imbalance bars, `|θ_T| ≥ E₀[T]·|2P[b = 1] − 1|`. It makes
a count bar of `n` hold `n` rows, not `n + 1`.

**A row larger than the threshold stays whole, in one bar.** A trade of 35
against a threshold of 10 closes the open bar with the whole trade in it,
and the bar's `thresholds` is 3 or 4, by where the total stood. The
alternative splits the row across bars, as Easley, López de Prado and
O'Hara's volume buckets do, so that each bucket holds exactly the threshold.
Splitting is not recommended, for four reasons:

- **Every value in the output was observed.** A split invents rows with
  fractional sizes, all at one clock, and an open, high, low or close of a
  fraction of a trade has no meaning. `refresh_time` keeps the same rule.
- **The rate of thresholds stays exact without it.** Under `kind="sum"` the
  remainder is carried, so a row's excess counts toward the next threshold
  rather than being lost. The thresholds crossed over a stream equal the
  floor of the total over the threshold, whatever the row sizes. `thresholds`
  reports a large row's share, so a statistic that wants equal buckets can
  weight by it.
- **One row closes at most one bar.** That keeps the limited feed simple,
  with no bars held back between feeds ([State across chunks and on
  save](#state-across-chunks-and-on-save)).
- **VPIN does not need it.** Over bars of unequal size, VPIN is the sum of
  the bars' absolute imbalances over the sum of their volumes, which is the
  paper's formula when every bar holds exactly the threshold
  ([Oracles](#oracles)).

**Under `kind="sum"`, the remainder is carried, which makes the bars the
multiples of the threshold on the running total.** Let `S_t` be the total of
the group's quantities through row `t`, and `θ` the threshold. Row `t` closes
a bar when `floor(S_t / θ) > floor(S_{t−1} / θ)`. The bar's `thresholds` is
the difference, and its `carry` is `S_t − floor(S_t / θ)·θ`. The operator
keeps only that remainder, `R`, not `S_t`. Each row adds its quantity to
`R`, and a close subtracts `m·θ` with `m = floor(R / θ)`. In exact
arithmetic the two descriptions give the same bars, row for row, which is
what lets a `cum_sum` serve as the oracle.

**A quantity under `kind="sum"` that is below 0 is refused, naming the
row.** On a total that can fall, the multiples rule closes a bar each time the
total crosses a multiple downward and back, a bar a row while it hovers. A
signed quantity belongs to `kind="imbalance"`, whose rule does not do that.

**Under `kind="imbalance"`, the bar's sum starts from 0 at each bar.** That
is López de Prado's imbalance bar: the bar closes once the signed flow inside
it is large enough, either way. A carried excess would start the next bar
part-way toward the side the last one closed on, mixing two bars' flow. The
`thresholds` of an imbalance bar is `floor(|sum| / θ)`, and it has no `carry`.

**`threshold` is a positive number, or a column read at each row.** A
number must be finite and exactly a double (an integer up to 2^53), and under
`kind="count"` a whole number of at least 1. A column lets a threshold change
with the data, such as a day's expected volume over 50 joined from a small
daily table; each row's crossing test reads that row's threshold, and a
close subtracts multiples of it. The multiples description above, and its
`cum_sum` oracle, hold for a constant threshold only, so a column's oracle
is the loop of [Oracles](#oracles), run with the carry. A missing
or non-positive threshold in the column is refused, naming the row. López de
Prado's adaptive threshold, an exponentially weighted estimate of the bars'
own length and imbalance, feeds back from the bars to the threshold. No
column given from outside can express that, so it is a question, not part of
this design.

## Exact arithmetic

**A running total in one double drifts, and the crossing test reads the
drift.** Summed row by row, each addition rounds to the double nearest, and
the error grows with the number of rows and the size of the total. At a
dollar total of 1e12 a double resolves about 1e-4, so a crossing that
should fall on one row can fall on the next. The decision of which rows make
a bar is discrete: a rounding that moves the total across a multiple moves a
row from one bar to the next, and every later bar with it under `"sum"`.
Carrying only the remainder `R` bounds the size of the number, but each bar's
additions still round, and the carry passes their error on.

**So `R` is kept exactly, as a short sum of doubles.** An *expansion* is a
list of doubles whose exact sum is the value. Adding a double to an
expansion is exact by Knuth's two-sum, which splits `a + b` into its
rounded sum and the rounding error, both doubles (Shewchuk 1997, "Adaptive
precision floating-point arithmetic and fast robust geometric predicates").
The operations the crossing rule needs are all exact:

| operation | how |
|---|---|
| add a row's quantity | two-sum into the expansion; an `Int64` or `UInt64` value past 2^53 enters as two doubles, its rounded value and the exact integer left over |
| test `R >= θ`, or `\|R\| >= θ` | add `−θ` (and, for the second, `+θ`) to a copy and read the sign of its largest part, which is the sign of the exact sum |
| find `m = floor(R / θ)` | estimate it from `R`'s rounded value over `θ`, then step it by one, up or down, until the exact test holds; the estimate is off by at most two |
| subtract `m·θ` | `m` in binary: each `2^j·θ` is exact, a scaling by a power of two, so `m·θ` enters as at most 53 exact parts, and only for a row that crossed more than one threshold; `m` at or past 2^53 is refused, naming the row, since it means a threshold under 2^-53 of one row |
| emit `carry` | round the exact sum once, to nearest, as Python's `math.fsum` rounds |

**The expansion stays short and bounded.** After each change the expansion
is renormalized, so that each part is at most one unit in the last place of
the part above it (the renormalization of Joldes, Marty, Muller and Popescu,
"Arithmetic algorithms for extended precision using floating-point
expansions", IEEE Transactions on Computers, 2016). Each part is then at
least 2^52 below the one above. A value here lies between 2^-1074, the
smallest double, and 2^334, above twice the input bound of 1e100, which
bounds a quantity and the threshold alike. So an expansion has at most 27 parts, 216 bytes, whatever the input
(`⌊(334 + 1074) / 53⌋ + 1`), and a float sum among the aggregates, whose
values may reach the largest double, at most 40. On
realistic data it holds one to three. The build must test the property
after every operation, not assume it.

**Only addition, subtraction and scaling by a power of two are used.** No
product rounds, so no fused multiply-add and no library function is
involved, and every operation gives the same bits on every system. A state
saved on macOS continues to the bit on Windows and Linux, with no field
compared within a tolerance.

**A count is an integer.** Under `kind="count"`, `R` is a `u64`, and the
test is an integer comparison.

## The aggregates

**A bar's aggregates are Polars aggregations, written as Polars writes
them.** Each named argument is an expression whose aggregations are among
`first`, `last`, `min`, `max`, `sum`, `count` and `pl.len()`. An aggregation's
input may be any element-wise expression of the row, such as
`pl.when(...).then(...).otherwise(...)`. Element-wise arithmetic of the
aggregations is allowed around them, as in `vwap` above, and is computed on
the bar's row. The parser is the one `with_windows` uses for its formulas.
It reads Polars' serialized expression, which carries no stability promise
from Polars, so a new Polars may need it mended, as the window formulas may.
The same expressions, run by Polars' own `group_by`, are the oracle
([Oracles](#oracles)).

**Each aggregation matches Polars' own, dtype and nulls included.**

| aggregation | accumulator | bytes | matches Polars |
|---|---|---|---|
| `first`, `last` | the value, null included, as Polars' `first` and `last` take it | 16 | to the bit |
| `min`, `max` | the value; nulls skipped, NaN as Polars orders it | 16 | to the bit |
| `count`, `pl.len()` | the non-null count, the row count | 8 | to the bit, in Polars' index dtype |
| `sum` of an integer column | an `i128`, emitted as Polars' sum dtype; an overflow of that dtype refused, naming the bar | 16 | to the bit |
| `sum` of a float column | an expansion, rounded once at the close; a NaN or an infinity kept apart in plain arithmetic and given as Polars gives it | 32 + 8·p, p ≤ 40 | to the bit against `math.fsum`; within the bound in [Oracles](#oracles) against Polars, whose float sum is not correctly rounded |

`mean`, `std` and `var` are left out of the first build. A mean is
`sum() / count()`, written out, and its rounding is then stated. A
variance over a bar's rows is a second pass or a further accumulator, and no
case has asked for one. The input columns may be numeric, temporal or
Boolean. A string's `first` or `last`, such as a trade identifier, has no
bound on its size, so it is left out until a case asks.

## Groups and their bound

**Each value of `group` has its own bars.** Its state is its own open bar,
its remainder, its bar index and its clock. Rows of one group never touch
another's state, and the bars of all groups come out in the order of their
closing rows, interleaved.

**A group's bytes are fixed by the call, so they are checked when the query
is built.** In memory, a group holds about:

```text
B_group = 48 + len(key)                   the key in the group map
        + 128                             the clock state: last clock, session, pending gap
        + 40                              bar index, row count, time_open, flags
        + Q                               the running quantity: 8 for "count",
                                          24 + 8·p for "sum" and "imbalance"
        + Σ a_j                           each aggregate, by the table above
```

`p` is an expansion's parts: at most 27 for the running quantity and 40
for a float sum, one to three in practice. The
example above, with a dollar quantity, four `first`/`last`/`min`/`max`, and
three float sums, holds about 48 + 8 + 128 + 40 + 48 + 64 + 3·56 ≈ 500 bytes
a group in practice, and under 2 KB at the bounds. The three
sums are `volume`, `buys` and the dollar sum: `vwap`'s `quantity` sum is
`volume`'s, and an aggregation written twice is kept once. That bound
is checked against `max_state_bytes`, which means what it means on a spec:
a group's bytes, 256 KB by default and at most 64 MiB. A call past it is
refused when the query is built, naming the bytes. The state file holds the
same fields in msgpack, a little smaller.

**The number of groups is the only unbounded factor, so the groups
together are bounded by bytes too.** No cap on groups exists in the library
today: a bank's groups stay until `drop_groups` or `group_close` removes
them. A query has no point at which to call a method, so the operator takes
`max_total_bytes`, the most bytes the groups together may hold, 1 GiB by
default. Each new group adds its worst case, key included; a group that would
take the total past it is refused, naming the group, the bytes and the
keyword, and a `ReadinessWarning` comes once at half of it, the bank's
notice for a state past half of its cap. A count of groups was considered
and not taken: the bytes are what bound memory, and a group's bytes vary
with the call's aggregates and its key. Under `group_close="monotone"`, with
the bank's meaning, a group is closed when a row of a greater key arrives:
its open bar is cut ([The clock of a bar](#the-clock-of-a-bar)) and its state
dropped. That bounds the groups of a key that only grows, such as a trading
day or a block identifier.

## State across chunks and on save

**What is carried is each group's open bar, and nothing about closed
ones.** For each group: the bar index, the open bar's rows so far and its
`time_open`, the remainder `R`, each aggregate's accumulator, and the clock
state (the last clock of a row of positive weight, the session, and any gap
pending from rows of weight 0). Beside them, once for the operator: the kind
of clock, whether the input was grouped, the count of rows fed for error
messages, and the call's settings.

**A bar open at the end of a chunk is not emitted until a later row closes
it.** The rule is the same at every chunk boundary, so feeding a stream as
one chunk or a thousand gives the same bars, to the bit. Every decision is a
function of the rows up to it, and no row waits for a later one. Ties on the
clock are broken by row order ([The clock of a bar](#the-clock-of-a-bar)),
so, unlike `merge` and `with_windows`, no clock stamp is held back.

**A bar open at the end of the input is not emitted either.** It has not
reached its threshold, so it is not a bar of the rule. Emitting it would also
make the output depend on where the input was cut: a run resumed from the
state would emit the same rows again, merged into a different bar. It is
carried in the state. Whether a caller who knows the input has ended may ask
for the open bars, marked `complete = false`, is a question.

**A slice of the output saves the state after the input behind the rows it
returned.** Under `.head(n)`, the feed stops after the row that closed the
`n`th bar, and the state saved is the state after that row, wherever the
chunk ends, as `refresh_time` does. A row closes at most one bar of the
rule. It can also cut the open bar first, when it follows a gap or a session
change, and then close a bar of its own. A limit that falls between the two
stops after the cut, before the row is added. The resumed feed reads the row
again; the cut finds the bar empty, and an empty bar is never emitted, so the
cut does nothing the second time. So no bar is held between feeds, where
`refresh_time` must hold the points of one row.

**A save and a resume continue to the bit.** The remainder and the float
sums are their expansions' parts, saved as doubles, so nothing is rounded
by a save. The state file is a new kind in the shared envelope, at version 1,
with frozen fixtures under the state rules: one file for each form the layout
can take. That means ungrouped and grouped, each `kind`, an expansion of
several parts, each aggregate's accumulator including a null `first` and a
non-finite sum, a threshold column, and a clock state holding a gap pending
from rows of weight 0. A load is refused, naming the setting, when the call
differs from the saved one in `kind`, `quantity`, `threshold`, the
aggregates, `group`, the kind of clock, or the session, gap and reset
settings. It is refused too when a resumed group's first row is before its
saved clock.

## The clock of a bar

**A bar's clock is its close: the clock of the row that closed it.** That is
when the bar becomes known, so it is the clock a bank downstream reads, and
reading a bar at its open would let the bank see rows after the open.
`time_open` is given beside it for the bar's duration.

**Ties on the clock are broken by row order.** Rows that share the closing
row's clock but come after it in the input belong to the next bar. So two
bars can share a `time_close`, and a bar's `time_open` can equal the last
one's `time_close`. A bank downstream accepts equal clocks, since only a step
back is refused. This is `refresh_time`'s rule, and it is what lets a bar be
emitted at its closing row, not when a later stamp arrives. A live feed gets
each bar at the row that closes it.

**A row of weight 0 is in no bar and in no clock.** The library's rule is
that the stream behaves as if the row were not there, and here it means:

- it adds nothing to the count, the sum or the imbalance;
- it enters no aggregate, so its price is not the bar's close;
- it is never a bar's `time_open` or `time_close`;
- its time folds into the gap of the next row of positive weight, through the
  clock state every stream shares, so it cannot end a gap or start a
  session.

It is still read for the order check, since an input out of order is
refused whatever its weights. The test feeds a stream with and without its
rows of weight 0 and holds the bars equal to the bit. `weight` is optional,
and takes only 0 and 1, so it is a mask; any other value is refused,
naming the row. A weight of 2.5 has no meaning for a count or a high. A weighted count is
`kind="sum"` over the weight column. A row with a quantity of 0 differs: it
is a row of the bar, as the quote rows above are.

**The clock does not close a bar by default.** A bar is measured in the
quantity, not in time, and a volume bar may span a lunch break or a night.
Four settings close one early, each optional, each
read as a spec reads it:

| setting | the open bar is cut |
|---|---|
| `session` | at the first row of a new session |
| `gap_cap` | at a row whose step from the group's last row of positive weight is past `gap_cap` |
| `max_span` | at a row more than `max_span` after the bar's `time_open`: a bar of at most that span, as in mixed time-and-volume bars |
| `restart_after_step_back` | at a row whose clock steps back by more than it; without it, any step back is refused, naming the row |

A cut bar is emitted at the row that cut it, before that row is added, with
`complete = false` and `thresholds = 0`. Under `partial="drop"` it is not
emitted; `"keep"` is the default, as for a window that looks back. A cut
resets the remainder to 0, and the cutting row opens the next bar. The carry
keeps the thresholds' count exact over one unbroken stretch of the stream,
and a cut ends the stretch. A gap is seen only when the row after it arrives,
so a cut bar comes out at that row, in that row's chunk. That is still
streaming, and still independent of where chunks end.

## Decay and windows

**Decay does not apply to a bar.** A bar partitions the stream by an
undecayed total: a bar of 10,000 shares holds 10,000 shares however long they
took. A decayed total settles near the rate of flow times the half-life. A
threshold on it then fires when the rate is high, not when an amount has
traded, which makes it an event sampler on the intensity, such as a
CUSUM filter. That is a different block, and a question below.

**A window applies to a bar only as a limit on its span.** The quantity
already decides a bar's extent. `max_span` is the window form that makes
sense, a cap on a bar's clock span, and it is in the design above.

**Decay and windows apply to the statistics computed over bars.** The bars
are a stream like any other, so `with_windows` reads them. With no `clock`,
one unit is one row, so one unit is one bar. VPIN over the last 50 bars is
an evenly weighted window of 50 rows:

```python
flow = bars.with_columns(imbalance=(2 * pl.col("buys") - pl.col("volume")).abs())
vpin = rp.stream.with_windows(
    flow,
    vpin=rp.ewm_sum("imbalance", half_life=float("inf"), window_size=50, partial="null")
    / rp.ewm_sum("volume", half_life=float("inf"), window_size=50, partial="null"),
    group="symbol",
)
```

A finite `half_life` gives an exponentially weighted VPIN on the same bars,
and a `clock="time_close"` with a duration gives one over a span of time.
VPIN is a recipe in Polars, not a function of the library: the bars are the
block, and the statistic is composed from them. This uses the trades' own
`side`. VPIN's other form, bulk volume classification, splits each bar's
volume by the normal distribution function of its standardized price change.
Polars has no element-wise normal distribution function, so that form needs
one, which is a question below.

## Oracles

**For `"count"` and `"sum"`, the oracle is a cumulative sum and a
`group_by` in Polars.** With `S_{t−1}` the total before the row, the rows of
bar `k` are those with `floor(S_{t−1} / θ) = k`. The closing row is the last
row of that run, and it is the first with `floor(S_t / θ) > k`:

```python
theta = 50.0                                               # the threshold
aggregates = {"close": pl.col("mid").last()}              # the operator's own aggregates
q = pl.col("quantity").fill_nan(None).fill_null(0)        # missing adds 0
s_before = q.cum_sum().shift(1, fill_value=0).over("symbol")
s_after = q.cum_sum().over("symbol")
expected = (
    trades.with_row_index("_row")
    .filter(pl.col("w") > 0)                               # with a weight column w: rows of weight 0 are absent
    .with_columns(_bar=s_before // theta, _after=s_after // theta, _s=s_after)
    .group_by("symbol", "_bar", maintain_order=True)
    .agg(
        time_open=pl.col("ts").first(),
        time_close=pl.col("ts").last(),
        thresholds=pl.col("_after").last() - pl.col("_bar").first(),
        carry=pl.col("_s").last() - pl.col("_after").last() * theta,
        _close=pl.col("_row").last(),
        **aggregates,                                     # the same expressions the operator took
    )
    .filter(pl.col("thresholds") > 0)                      # the open bar has crossed nothing
    .sort("_close")
    .with_columns(bar=pl.int_range(pl.len()).over("symbol"))
)
```

A session, a gap cut or a reset adds the stretch's index to the group keys,
and gives the cut bars by the same `group_by` with `thresholds = 0`.

**The agreement expected depends on the arithmetic.**

| case | operator against oracle | why |
|---|---|---|
| `"count"`; `"sum"` of an integer column with an integer threshold | equal to the bit: every bar, aggregate, `thresholds` and `carry` | Polars' integer `cum_sum` and floor division are exact |
| `"sum"` of floats on a dyadic grid, values `j / 1024` with `\|j\| < 2^40` and totals under 2^52 | equal to the bit, aggregates included | every partial sum is a double, so Polars' `cum_sum` is exact too |
| `"sum"` of general floats, against the same construction in Python's `fractions.Fraction` | bars, `thresholds` and `carry` equal to the bit | both are exact; `carry` is the exact remainder rounded once |
| the same, against Polars' `cum_sum` in doubles | bars equal except at a row where Polars' total is within its rounding of a multiple, `\|S_t − jθ\| ≤ t·ε·S_t`; the test finds every row that differs and checks each is such a row | the double total drifts, which is why the operator does not use one |
| a float `sum` aggregate | equal to the bit against `math.fsum` over the bar's rows; within `(n − 1)·ε·Σ\|x\|` of Polars' `sum`, for `n` rows | the operator rounds once; Polars sums in an order of its own |

**For `"imbalance"`, the oracle is the definition, written as a loop.** The
reset at each bar makes each bar's start depend on the previous close, so no
`cum_sum` expresses it, in Polars or elsewhere. The loop runs in
`fractions.Fraction`, so it is exact:

```python
def imbalance_bars(values, theta):
    total, start, bars = Fraction(0), 0, []
    for i, v in enumerate(values):
        total += Fraction(v)
        if abs(total) >= theta:
            bars.append((start, i, int(abs(total) // theta)))
            total, start = Fraction(0), i + 1
    return bars
```

The bars agree to the bit, and their aggregates run through Polars'
`group_by` on the rows each bar holds, as above. Since the loop is Python's
alone, the exact arithmetic is also tested in Rust against
`crate::oracle::sum`, Shewchuk's partials rounded as `math.fsum` rounds.
That oracle shares no code with `exact.rs`.

**VPIN is checked as a statistic of the bars, which Polars computes too.**
`rolling_sum(50)` of the imbalance over `rolling_sum(50)` of the volume, on
the bars frame, agrees with the `with_windows` recipe within the rounding of
two running sums of 50 terms. It is equal to the bit for integer volumes.
The library's part is the bars, which the oracles above hold.

**The third-party libraries, checked as oracles.** Each library below was
installed at the version named, its bar code read, and run on seeded
streams against the three rules of [The crossing rule](#the-crossing-rule)
written out in `fractions.Fraction`: the count, the sum with its remainder
carried as multiples of the threshold, and the signed sum that resets to 0
at each bar. The streams are 4,000 rows on a dyadic grid (sizes the
integers 1 to 100, prices `j / 64` in [10, 20)), so every double sum a
library makes is exact and its own arithmetic cannot be what differs; three
seeds without and three with rows past the threshold (sizes 600 to 2,500
against a threshold of 500). A rule "matches" when the closing rows agree
on all six streams, and each library's comparison at exactly the threshold
was checked on a stream whose running total lands on it. Checked
2026-11-23:

| library | version, licence, upkeep | count | sum, remainder carried | imbalance, reset |
|---|---|---|---|---|
| `bar-forge` (PyPI) | 1.0.1, MIT, two releases in August 2026; depends on NumPy alone | matches (`tick_bars`) | differs: its `volume_bars` and `dollar_bars` reset the total to 0 at each close, so the excess of the closing row is lost; that reset rule is this design's `"imbalance"` on a quantity that is never negative, and matches it | matches, with a fixed threshold (`tick_imbalance_bars`, `volume_imbalance_bars`: `\|θ\| >= threshold`, then 0); it signs each trade itself by the tick rule, +1 before the first price change and the last sign carried over a step of 0 |
| `ml4t-engineer` (PyPI) | 0.2.1, MIT, 28 releases from January to October 2026; requires `polars>=2.0`, `pydantic`, `structlog`, `numba` and `ml4t-specs` | matches (`TickBarSampler`) | differs as `bar-forge` does: `VolumeBarSampler` and `DollarBarSampler` reset to 0 | matches (`FixedTickImbalanceBarSampler`, `FixedVolumeImbalanceBarSampler`, signs given as a column); its default `ImbalanceBarSampler` is López de Prado's adaptive threshold, which this design leaves out |
| `mlfinpy` (PyPI; the successor of `mlfinlab`) | 0.1.2, MIT, two releases in October 2024 and none since; requires `numpy<1.27`, `pandas<3`, `numba<0.61` | matches (`get_tick_bars`) | differs: `get_volume_bars` and `get_dollar_bars` reset to 0 | differs: its imbalance bars take only the adaptive threshold, and compare with `>`, not `>=` |
| `finmlkit` (PyPI) | 0.1.11, MIT, six releases from August to September 2025 and none since; requires `numpy~=2.2.0`, `pandas~=2.2.0` | matches in the closing rows; its aggregates leave the stream's first row out of the first bar, though that row counts toward the first threshold | differs: volume bars reset to 0; dollar bars carry, but subtract the threshold once, so they match this design only while every row is below the threshold, and after a row past it they close a bar on each of the following rows until the excess is spent | not built: its imbalance and run bar functions raise `NotImplementedError` |
| `PINstimation` (CRAN) | 0.2.2, GPL-3 or later, November 2026; R | does not build bars from rows | differs: VPIN's volume buckets are the multiples of the bucket size on the running total, as here, but are built from time bars, not from rows, and a time bar that straddles a bucket's edge, or is larger than a bucket, is split across buckets by volume; the bucket size is the average daily volume over a count of buckets, read from the whole input | does not build them |

Every library compares with `>=` where it takes a fixed threshold, as this
design does, and every one keeps the crossing row whole in one bar (no
library splits a row except `PINstimation`). None keeps its running total
in anything but one double.

**What the operator is held to.** `bar-forge` computes the same count bar
and the same fixed-threshold imbalance bar, and its reset volume and dollar
bars are the `"imbalance"` rule on a quantity that is never negative. It
depends on NumPy alone, so it is a test-only dependency, and a test holds
the operator to it on a seeded stream: the closing rows, and the open, high,
low, close, row count, volume and notional of each bar, to the bit (its
sums are plain doubles in row order, exact on the dyadic grid). The signs of
its imbalance bars are its own `tick_rule_signs`, fed to the operator as the
quantity's sign. `ml4t-engineer` computes the same count and imbalance bars
as `bar-forge`, row for row on the streams above. It cannot be a test
dependency: it requires `polars>=2.0`, which the floor of this library's
Polars range cannot install beside it, so it was checked once, outside the
suite, at the version above. `mlfinpy` and `finmlkit` require NumPy and
pandas versions older than the ones the tests run on, so they too were
checked outside the suite, each in an environment of its own. Once the
operator was built, its closing rows on the same six streams were compared
with each of the three there: equal for every rule the table marks as
matching (`ml4t-engineer`'s tick, volume, dollar and both fixed imbalance
bars; `mlfinpy`'s tick, volume and dollar bars; `finmlkit`'s tick and volume
bars), `finmlkit`'s dollar bars equal to `"sum"` on the three streams with no
row past the threshold and different on the three with one.

**No library carries the remainder as multiples of the threshold**, which
is the `"sum"` rule. Its oracles of record are Polars' `cum_sum` and
`group_by` construction above, and the same construction in `Fraction`.
`finmlkit`'s dollar bars are the one library rule that carries, and they
agree with `"sum"` only on a stream whose every row is below the threshold.

## Tests and budgets

The build owes these, each in its tier under the testing rules:

- **Oracles**: the table above, on generated streams of several sizes, and
  on `trades` for volume, dollar and imbalance bars per symbol.
- **Chunk invariance**: one chunk against any chunking, by Hypothesis over
  chunk sizes from 1 row, the bars and the saved state equal to the bit.
- **Rows of weight 0**: a stream with and without them, the bars equal to
  the bit, a group whose first row has weight 0 included.
- **Save and resume** after every row of a short stream, and after a
  `.head(n)` at every `n`: the concatenated bars equal one run's.
- **Degenerate cases**: a total that reaches a multiple exactly; a row of
  quantity 0; a stream whose quantities are all 0 and that closes no bar; one
  row of a million thresholds; a threshold near the smallest double and one
  near 1e100; a negative quantity refused under `"sum"`; nulls and NaN in the
  quantity, the threshold column and each aggregate's input; a step back;
  each cut; the group cap reached.
- **The expansion's invariant** after every operation, and its bound of 28
  parts on values spread over the whole exponent range.
- **State fixtures** for every form, continued and re-saved byte for byte on
  each system.
- **Budgets**: an instruction count per row for `"count"` with no
  aggregates, and for dollar bars with the example's aggregates, beside a
  memory case of the state per group. A sum through the expansion is a few
  two-sums a row, an estimate to be measured, not a measurement.

## As built

The build follows the sections above; where they left a choice open, it
made these:

- **A group is an integer, a String, a Categorical or a Date.** The key is
  held as its text, and a bar written at another group's row (a monotone
  close) or at the end (`final=True`) has no row of its own to copy the key
  from, so the key is cast back from its text: these four are the dtypes
  whose text casts back to the same value.
- **A bar's index counts the bars emitted.** A bar cut short under
  `partial="drop"` takes no index, so the indices of a group's bars are
  0, 1, 2, ... with no gap.
- **A cut bar's `time_close` is its last row's clock**, not the clock of the
  row that cut it: every value in it comes from rows at or before its last
  row. It is emitted at the cutting row, so under a group the output is in
  the order the bars became known, not sorted by `time_close` across groups.
- **`final=True` emits every open bar, whatever `partial`**, in the order of
  the groups' keys, since the caller asked for them by name.
- **A null weight is 0**, as a bank skips a row of null weight.
- **A `Float32` sum is the exact sum rounded to a double, then to a
  `Float32`**: two roundings, which can differ from one in a tie of the
  second. A `Float64` sum is rounded once.
- **Under `kind="count"` with a threshold column**, a threshold that falls
  below the bar's rows closes the bar at once, with `thresholds` the rows
  over the threshold, rounded down.
- **The state file** holds the call (its settings and the text of its
  expressions, compared whole on a load), each group's open bar, and the
  high-water key of a monotone close. Its three frozen fixtures cover an
  ungrouped `"sum"` on a float clock (a running quantity of three parts, a
  null first value, a float sum of two parts and a NaN), a grouped
  `"imbalance"` of integers past 2^53 on an integer clock (a threshold
  column, a weight, a session and the cuts), and a `"count"` on a temporal
  clock under a monotone close.
- **The tick test is `rp.tick_sign`, a window operator** (question 12).
  It reads the step `rp.increment` reads and carries the sign of the last
  step that was not 0 over a step of 0. A row with no step has no sign: a
  session's first row, a row whose price is missing (skipped, as an
  increment skips it), and the rows at the first price until it first
  moves. Lee and Ready's convention, and bar-forge's, signs those +1;
  `.fill_null(1.0)` gives it, and the tests hold the filled signs to
  bar-forge's `tick_rule_signs` row for row, and the unfilled ones to
  Polars' `sign` of `diff`, 0 read as null, filled forward `.over` the
  group, to the bit. A zero-weight row under a weighted spec's policy
  (`like=`) moves the sign as any row does: the sign looks back only, and
  no window looking back reads a weight.
  The carried sign is a number a group per tick sign. It rides in the
  windows state's `prev`, past the increments' previous values, rather
  than in a new field: the state's layout is unchanged, so its version and
  the bank's schema stay where they are, every existing state is the same
  bytes, and a new fixture, `with_windows_tick`, freezes the form with
  carried signs. A build without `tick_sign` refuses such a state by the
  operator's name in its formulas, not by a version number.
- **The budgets are two instruction cases and a resident-memory case.**
  `bars_volume` counts one `Bars::feed` of 1,000 rows of volume bars of
  1,000 over sizes of 1 to 100, about 20 rows a bar, with the open, high,
  low and close of the price, the volume and the rows: the exact running
  total, the crossing test and six aggregates a row, about 50 bars closed.
  `bars_volume_groups100` is the same over 100 groups, one row of each in
  turn after 20 a group, so a row finds its group's open bar and about half
  the groups close one. Each takes CI's first count as its budget, as a new
  case does; locally (Callgrind, gungraun 0.20.0, rustc 1.100.0, x86-64
  Linux, not CI's pinned toolchain) they counted 1,735,679 and
  2,276,447 instructions, about 1,736 and 2,276 a row, beside the 1,538,670
  budget of `window_ewm_sum_w10k`, a decayed window sum over the same rows.
  `rss.bars_5m` streams five million rows from a parquet scan through count
  bars of 100 rows with six aggregates to a parquet sink, as the bank's
  `streaming_5m` does, and takes CI's floor-leg readings as its budgets.
  Linux read 136.7 and 136.3 MiB locally, the bank's run 130.9 and 131.2
  beside it. The peak is not flat in the stream's length: 56, 98 and 136
  MiB at 1M, 3M and 5M rows, where the bank's run read 123 at 1M. The bars'
  state is not the cause: it saves to 825 bytes at 100,000 rows and at a
  million, and the core's live heap is flat over 40 chunks of 100,000
  rows. The same rows fed from memory hold flat (12.7 MiB over 30 chunks),
  and fed from the scan the process's memory rises over the first chunks
  and falls back as they are consumed, so the growth points to the parquet
  scan's decoded batches held ahead of the operator, under
  `POLARS_ROW_GROUP_PREFETCH_SIZE=1` and four threads on py-polars 2.1.0;
  why the bank's slower consumer is not read as far ahead is not traced.
  It is reported, not fixed here: the budget holds the 5M-row figure, so
  it would catch the growth getting worse.
  The design's case of the state per group is `Bars::group_bytes`, which
  the caps already check, rather than a budget.

## Questions

Each choice the design makes that the user may want otherwise, with the
recommendation:

1. **A row larger than the threshold: whole or split?** Recommended:
   whole, in one bar, with `thresholds` counting what it crossed. A split
   invents fractional rows, and the carry keeps the count of thresholds
   exact without one.
2. **Carry or reset after a close?** Recommended: carry under `"sum"`,
   which makes the bars the multiples of the threshold on the running total
   and gives the `cum_sum` oracle; reset under `"imbalance"`, as López de
   Prado defines it. Under `"count"` the two coincide.
3. **`>=` or `>`?** Recommended: `>=`, a bar closing when its total reaches
   the threshold.
4. **The running total in exact arithmetic?** Recommended: yes, as an
   expansion of at most 28 doubles, typically one to three, with only
   addition and exact scaling, so the same bits on every system. The
   alternative, a plain double, drifts with the stream's length, and moves
   a row between bars when it does.
5. **Float sums among the aggregates rounded once?** Recommended: yes,
   through the same expansion, equal to the bit to `math.fsum`. The
   alternative, a plain running sum, takes 8 bytes and matches neither
   `math.fsum` nor Polars to the bit.
6. **A negative quantity under `"sum"`?** Recommended: refused, naming the
   row; a signed quantity uses `"imbalance"`.
7. **`weight`?** Recommended: optional, 0 or 1 only, a row of weight 0 in
   no bar and no clock, and a weighted count written as a sum of the weight
   column.
8. **The open bars at the end of the input?** Recommended: never emitted
   by default, and carried in the state. A keyword such as `final=True` for
   a caller who knows the input has ended would emit them with `complete =
   false` and save a state with no open bars.
9. **Cuts by `session`, `gap_cap`, `max_span` and a reset?** Recommended:
   all four offered, none on by default, the cut bar kept with `complete =
   false` unless `partial="drop"`, and the remainder reset at a cut.
10. **The bound on groups?** Recommended: `max_state_bytes` for a group,
    checked when the query is built from the worst case; `max_groups`, 100,000
    by default, with a warning at half; and `group_close="monotone"`, which
    cuts and drops a group once a greater key arrives. **Answered:** no count
    of groups; `max_state_bytes` for a group and `max_total_bytes` for the
    stream, refused and warned as the bank refuses and warns, and
    `group_close="monotone"` kept.
11. **The threshold: a number, a column, or adaptive?** Recommended: a
    number or a column read at each row. López de Prado's adaptive
    threshold, fed back from the bars themselves, not in this build.
12. **The tick test for unsigned trades?** Recommended: a separate window
    operator, `rp.tick_sign("price")` under `with_windows`, rather than a
    setting of the bars. It reads the last price change, which no
    element-wise expression can, and a sign is useful beyond bars.
    `rp.increment` already gives the step; the tick test also carries the
    last nonzero sign across a step of 0.
13. **Which aggregations?** Recommended: `first`, `last`, `min`, `max`,
    `sum`, `count` and `pl.len()` over element-wise inputs, with element-wise
    arithmetic around them, parsed as `with_windows` parses its formulas.
    `mean`, `std`, `var` and a string's `first` or `last` are left out until
    a case needs them.
14. **VPIN?** Recommended: a recipe over the bars with `with_windows`, not
    a function. Bulk volume classification needs an element-wise normal
    distribution function, which Polars lacks; recommended as a small block
    of its own, if wanted.
15. **An event sampler on a decayed quantity, such as a CUSUM filter?**
    Recommended: a separate block, since its threshold reads an intensity,
    not an amount.
16. **The command line?** Recommended: not in the first build, as
    `refresh_time` and `merge` are not there either; the Python and Rust
    surfaces first.
17. **The name and the keywords?** Recommended: `rp.stream.bars(lf, *,
    clock, kind, threshold, quantity=None, group=None, weight=None,
    session=None, gap_cap=None, max_span=None, restart_after_step_back=None,
    partial="keep", group_close=None, max_state_bytes=..., max_groups=100_000,
    chunk_size=None, load_state=None, save_state=None, **aggregates)`,
    `quantity` required under `"sum"` and `"imbalance"` and refused under
    `"count"`. **Built** with `max_total_bytes=1 << 30` in place of
    `max_groups`, and `final=False`.

The answers, in short: questions 1 to 9 and 11 to 16 as recommended. The
method of question 4 is Shewchuk's expansion sums, and its bits are tested
on each system. Question 7's reason is in the docstring: a weight only
includes or excludes a row, since a bar's count or high has no meaning for a
fraction of a row. Question 9's cuts are each tested on both sides of the
setting. The tick test of question 12 is its own small part, with its own
oracle, and VPIN and a CUSUM filter (questions 14 and 15) are later work,
not this build.
