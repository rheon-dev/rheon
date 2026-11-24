<!-- kind: explanation -->
# How a window operator weighs the rows near each row

This page explains, for anyone who makes features or targets with the
window operators of [`rp.ops`](https://rheon-dev.github.io/rheon/ops.html),
which rows each operator's window holds and how it weighs them on a clock
whose steps are uneven.

**A window operator returns a Polars expression (a column computation,
such as `pl.col("x") + 1`) whose value on each row is computed over the rows near it on the [clock](GLOSSARY.md#time): those
before it, or those after it.** A window that looks back makes a feature,
such as the last minute's time-weighted mid. A window that looks ahead,
under `reverse=True` and with the `window_size` it then needs, makes a target, such as the next minute's VWAP
(volume-weighted average price). Each operator updates running sums, or
for a quantile a sketch, as each row arrives, where Polars' `rolling`
gathers the window's rows again for every row ([What a window operator
costs](PERFORMANCE.md#what-a-window-operator-costs) has the timings).

**Given a `window_size`, `rp.ewm_mean` gives a *bounded EWMA*:** an
exponentially weighted moving average whose history has a maximum age. Its
weights decay along the clock as any EWMA's do, but rows farther than
`window_size` from the row drop out instead of fading forever.

[Make features that look back over the
stream](how-to/make-window-features.md) adds windows as columns before the
bank with `rp.stream.with_windows`, and [Make a target that looks
ahead](how-to/make-a-target-that-looks-ahead.md) gives a model one as its
target. [Scale a feature by its spread](how-to/scale-a-feature-by-its-spread.md)
fits a model on a feature's z-score, made with `rp.ewm_mean` and
`rp.ewm_std`. Each operator's parameters, their defaults and what each refuses
are in its docstring under
[`rp.ops`](https://rheon-dev.github.io/rheon/ops.html).

## A row's weight on an uneven clock

**A row's weight falls with its distance on the clock from the row the
window is read at, and not with its count of rows.** A row `half_life`
away weighs half as much, and `half_life=float("inf")` weighs every row of
the window the same. `half_life` and `window_size` are in the clock's
units: numbers on a numeric clock, and durations such as `"10s"` on a
temporal one. Each operator then reads the weighted rows its own way:

| operator | how it weighs the window's rows | what it is held to |
|---|---|---|
| `rp.ewm_mean` | each value is held over the interval ending at its row and weighed by that interval's decayed time, so a burst of rows does not outweigh a quiet period, and a row at a repeated clock value moves no mean. Looking ahead, each value is held until the next row | `ewm_mean_by` |
| `rp.ewm_sum` | `Σ 0.5 ** (age / half_life) · x`, each row counted once at its own time. Every row at a repeated clock value carries that value's total. A window with no row with a value sums to 0.0 under the default `min_samples=0`, and is null under 1 or more | `ewm_sum_by` on distinct clock values, whose sum at a repeated one runs row by row; `rolling_sum_by`, empty windows included |
| `rp.ewm_rate` | the sum divided by the decayed time the window covers: a quantity per unit of clock | the sum and the decayed time, each as above |
| `rp.ewm_quantile` | each row weighed as `rp.ewm_sum` weighs it ([Quantiles and a robust scale](#quantiles-and-a-robust-scale)) | numpy's weighted quantile (`inverted_cdf`), within 0.78% |
| `rp.rolling_max`, `rp.rolling_min` | every row of the window counts once, whatever its age, so there is no `half_life`; with no `window_size`, every row since the stretch began or since the last `since` event ([The largest and smallest value, and since an event](#the-largest-and-smallest-value-and-since-an-event)) | `rolling_max_by` and `rolling_min_by`, bit for bit, a NaN or an infinity read as missing where Polars lets a NaN make the window NaN and counts an infinity |
| `rp.ewm_var`, `rp.ewm_std` | about `rp.ewm_mean`, each value weighed as the mean weighs it, and looking back only. With `bias=False`, the default, the variance is corrected for the weights' number and inequality, so a window that one row's weight fills is null. A window that holds one value gives 0 exactly | `ewm_var` and `ewm_std` with `adjust=False`, on a clock that steps by 1 a row with `half_life` in rows and an input with no nulls: a null row moves nothing here and the last value is held across the longer step to the next value, which no `ignore_nulls` setting of Polars' does. Polars before 1.44 gave the first row 0.0 under `bias=False`, where it now gives null, as these do |
| `rp.ewm_moments` | one term of the moments of one or more columns, each perhaps lagged (`{"x": 3}`, `{"x": 1, "x@1": 1}`), each row weighed by its held interval's decayed time as `rp.ewm_var` weighs it, and looking back only. The moments are merged a row at a time by Pébay's rule, so a column far from zero keeps its higher moments. Under `window_size` a row is in the window by its clock value and weighs its whole interval, where `rp.ewm_var` counts only the oldest row's part inside the window. A lag pairs each row with the latest row at or before `t - k`, within `lag_tolerance` | numpy's weighted averages of the deviations' products over the window's rows; scipy's `moment`, `skew`, `kurtosis` and `kstat` at equal weights; `rp.spec.ewm_moments` fed the same weights, to the bit |

`rp.increment`, which is not a window operator, gives an operator an
input: `x_t − x_{t−1}`, back to the last row with a value in the group and
session. It is null on the first row of each group and of each session, on a row
whose input is null, and after a reset, and in seconds when `x` is a
temporal column.

[Check the result](how-to/make-window-features.md#check-the-result)
compares `rp.ewm_mean` and `rp.ewm_sum` with Polars' `ewm_mean_by` and
`rolling_sum_by`, on one symbol's rows.

**A row whose value is missing adds nothing, and gets the window's value
as it stands.** A null is missing, and so is a NaN, an infinite value or
one beyond 1e100 in magnitude. Polars' `ewm_mean_by` and `ewm_sum_by`
give a null row null.

**A time-weighted mean weighs each value by how long it held.** Here `x`
is 1 for ten clock units, and then 0 on ten rows in the next unit. A mean
that counts each row once reads the burst of zeros as most of the window.
`rp.ewm_mean` reads the time each value held, and the first value of a
[stretch](#which-rows-a-window-holds) is held from before the stretch
begins. This code builds its own rows:

```python
# x is 1 from t = 0 to t = 10, then 0 on ten rows in the next clock unit.
burst = [10.1 + 0.1 * i for i in range(10)]
rows = pl.DataFrame({"t": [0.0, 10.0, *burst], "x": [0.0, 1.0, *[0.0] * 10], "one": 1.0})
# The last ten clock units, every row weighed evenly, by its count and by its time.
even = dict(half_life=float("inf"), window_size=10.0)
out = rp.stream.with_windows(
    rows,
    rows_in=rp.ewm_sum("one", **even),                                  # how many rows the window holds
    by_count=rp.ewm_sum("x", **even) / rp.ewm_sum("one", **even),       # each row counted once
    by_time=rp.ewm_mean("x", **even),                                   # each value weighed by the time it held
    clock="t", gap_cap=20.0,
)
last = out.row(-1, named=True)
print(f"at t = {last['t']:.1f}: {last['rows_in']:.0f} rows in the window, x = 1 on one of them, for 9 of its 10 clock units")
print(f"mean by count {last['by_count']:.3f}, time-weighted mean {last['by_time']:.3f}")
```

```text
at t = 11.0: 11 rows in the window, x = 1 on one of them, for 9 of its 10 clock units
mean by count 0.091, time-weighted mean 0.900
```

The window at `t = 11` is `(1, 11]`. The row at 10 holds its value of 1
over `(0, 10]`, of which 9 units are inside the window, and the zeros hold
the last unit. So the time-weighted mean is 0.9, where a mean by count is
1 in 11.

## Which rows a window holds

**A window holds the rows Polars' `rolling_*_by` would give it.** Windows
are taken over clock values, so every row that shares a clock value (its
*stamp*) gets the same window, the rows after it at that stamp included.
With `w` the `window_size`, `closed` sets which ends are in, and so when a
window that looks back has its value:

| `closed` | looking back, row *t*'s window | its value is known | looking ahead |
|---|---|---|---|
| `"right"`, the default | `(t − w, t]`: the rows at *t*'s own stamp are in | at the next distinct stamp | `(t, t + w]`: a row exactly `w` after *t* is in |
| `"left"` | `[t − w, t)` | as the stamp arrives | `[t, t + w)`: every row at *t*'s stamp, *t* itself included |
| `"both"` | both ends | at the next distinct stamp | both ends |
| `"none"` | neither end | as the stamp arrives | neither end |

Counting the rows in each window shows the rule. An evenly weighted sum of
a column of ones is the number of rows in the window. Five rows on a
numeric clock, two of them at the stamp 1.0, with a window of 2:

```python
stamps = pl.DataFrame({"t": [0.0, 1.0, 1.0, 2.0, 3.5], "one": 1.0})
# Count the rows in each window, back and ahead, under both closed rules.
counts = rp.stream.with_windows(
    stamps,
    **{f"back_{c}": rp.ewm_sum("one", half_life=float("inf"), window_size=2.0, closed=c) for c in ["right", "left"]},
    **{f"ahead_{c}": rp.ewm_sum("one", half_life=float("inf"), window_size=2.0, closed=c, reverse=True) for c in ["right", "left"]},
    clock="t", gap_cap=10.0,
)
# The counts as a table, a line per row.
show = lambda v: "null" if v is None else f"{v:.0f}"
print("   t  back_right  back_left  ahead_right  ahead_left")
for r in counts.iter_rows(named=True):
    print(f"{r['t']:4.1f}  {show(r['back_right']):>10}  {show(r['back_left']):>9}"
          f"  {show(r['ahead_right']):>11}  {show(r['ahead_left']):>10}")
```

```text
   t  back_right  back_left  ahead_right  ahead_left
 0.0           1          0            3           3
 1.0           3          1            1           3
 1.0           3          1            1           3
 2.0           3          3         null        null
 3.5           2          1         null        null
```

Both rows at 1.0 get one window. Looking back under `"right"`, it is
`(−1, 1]`: the row at 0.0 and both rows at 1.0. Under `"left"` it is
`[−1, 1)`, the row at 0.0 alone, and the row at 0.0 has an empty window,
so a sum of 0, as Polars' `rolling_sum_by` gives. A window with fewer rows
with a value than `min_samples` is null instead, and a sum's default is 0.
Looking ahead from 0.0 under `"right"`, `(0, 2]` holds both rows at 1.0 and
the row at 2.0. The last two rows' windows ahead are still open when the
input ends, so they are null.

**A row is in a window by its stamp, but a mean weighs it by the part of
its held interval inside the window.** So under `"left"` or `"both"`, a row
exactly one window old counts toward `min_samples`, the fewest rows with a
value a window must hold, and weighs nothing in a mean, since its interval
ends at its row.

**Without a `window_size`, a window holds every row since its stretch
began.** A *stretch* is the rows between two clock breaks: it starts at the
stream's first row, or at the first row after a gap longer than `gap_cap`
or a session change. A reset starts the window over in the same way. Those rows are weighed by their decay alone.
An evenly weighted mean, variance or standard deviation
(`half_life=float("inf")`) needs a `window_size`, and `with_windows`
refuses one without it.

**`with_windows` returns the rows in input order, each once every window
over it has its value.** A window that looks back, under the default
`closed="right"`, has its value at the next row with a later clock value,
and a window that looks ahead once its span has passed. So the output
trails the input by one clock step for a window that looks back, and by
the longest look-ahead window for one that looks ahead. A group that falls
silent holds back the rows after it for at most `gap_cap` of the stream's time. Only a clock column has a `gap_cap`, so a
window that looks ahead under `group` needs `clock`, and `with_windows`
raises `ValueError` without it. Each group's clock would count only its own
rows, and a silent group's windows would hold back every later row until
the input ended.

## Why a formula around the operators is element-wise

**`with_windows` computes a formula one chunk of rows at a time, so the
Polars around the operators must read each row alone.** A step that reads
another row, such as a `shift`, would give a chunk's first row a different
value from the one it gets in the middle of a chunk. So the call refuses
it by name, and chunking the stream cannot change its output:

| | in a formula |
|---|---|
| allowed, around at least one operator | columns, literals, arithmetic, negation, comparisons, `&` and `\|`, `log`, `exp`, `abs`, `sqrt`, `pow`, `clip`, `fill_null`, `is_null`, `is_not_null`, `when/then/otherwise`, `cast` and `alias` |
| refused by name when the [query](GLOSSARY.md#the-model-bank-and-what-it-reads) is built, since each would depend on where a chunk ended | a `shift`, a cumulative or rolling function, `over` or an aggregation |
| refused by name when the query is built: write it with Polars' own `with_columns` | a formula with no operator |

This code tries both refusals on `trades`, a stream of quotes with trades
between them. Its `mid` is on every row, and `side`, `quantity` and `price`
are null on a quote. It uses `trades` from [Example data](EXAMPLE-DATA.md):

```python
clock = dict(clock="ts", gap_cap="5m", group="symbol")
lagged = rp.ewm_mean("mid", half_life="5s").shift(1)       # reads the row before: a chunk's first row has none
spread = pl.col("price") - pl.col("mid")                    # reads the row alone, and no operator
for formula in [lagged, spread]:
    try:
        rp.stream.with_windows(trades, out=formula, **clock)
    except ValueError as e:
        print(str(e).split(". ")[0])
```

```text
with_windows: a formula over window operators reads each row alone, but its expression (a column computation, such as pl.col("x") + 1) holds the method shift, which it cannot read
with_windows: 'out' holds no window operator, so it reads only its own row
```

**An operator's input can be a formula of the same kind, `rp.increment`
included, but not another operator.** To window an operator's output, make
it a column in one `with_windows` call, and window that column in a second
([Window an operator's
output](how-to/make-window-features.md#window-an-operators-output)).

## The edges of a window: gaps, sessions and resets

**The operators read a spec's clock keywords, given once to `with_windows`
for every formula in the call.** These are `clock`, `gap_cap`,
`restart_after_step_back`, `session`, `session_gap` and `group`, and
`gap_cap` is required with a clock. A [formula
target](GLOSSARY.md#fitting-a-model) reads them from its spec. With `group`, each group keeps its own sessions and windows.

**Unlike a spec, `with_windows` needs no `session_gap` for a session, since
every window ends at a session change.** So no window measures the clock's
step across one. A number or a duration given for `session_gap` is refused, since
it would change nothing. Only `session_gap="reset"` acts, on the windows looking ahead that a session change leaves open.

**A gap longer than `gap_cap`, or a session change, ends every window open
across it, and the operator's `partial` sets what the cut window gives.**
A window that looks back is cut short in the same way where it reaches back
past the stream's first row, or past the first row after a gap or a reset.
A [reset](GLOSSARY.md#time) (a step back larger than
`restart_after_step_back`, or a session change under
`session_gap="reset"`) discards the open windows instead, so their rows
come out null, never dropped:

| `partial` | a cut window gives | for a formula target, the row is | the default for |
|---|---|---|---|
| `"keep"` | the value over what it saw, the window ending at the last row seen | learned from what the window saw | looking back |
| `"null"` | null | not learned | looking ahead |
| `"drop"` | nothing: the row leaves the output | not learned, as under `"null"`, while its other targets still are | |

`partial` needs a `window_size`, since without one no window is cut short.
On `trades`, a window of a minute looking back is cut short on each
symbol's first minute of rows, 102 rows in all.
This code uses `trades` from [Example data](EXAMPLE-DATA.md):

```python
for partial in ["keep", "null", "drop"]:
    out = rp.stream.with_windows(
        trades,
        mid_mean=rp.ewm_mean("mid", half_life="5s", window_size="1m", partial=partial),
        clock="ts", gap_cap="5m", group="symbol",
    )
    print(f'partial="{partial}": {out.height} rows, {out["mid_mean"].null_count()} null')
```

```text
partial="keep": 3000 rows, 0 null
partial="null": 3000 rows, 102 null
partial="drop": 2898 rows, 0 null
```

**A window that looks ahead under `closed="right"` or `"both"` gives its
value under any `partial` when its far edge falls on the last row before a
gap or a reset, or on its group's last row when the input ends.** That row is
in the window and no later row can join it, so the window is whole. A window
whose far edge lies past the last row of the input is null, since rows past
the input would have joined it. For a formula target, a reset still starts the model
over and drops every row waiting to be learned, a whole one with the rest.

## Quantiles and a robust scale

**`rp.ewm_quantile` is the quantile of the window's values, each weighed as
`rp.ewm_sum` weighs it.** Row *j* weighs `0.5 ** (age / half_life)`, its age
taken from the row the window is read at. The quantile at `quantile=q` is
the smallest value whose share of the weight, counting it and every value
below it, reaches `q`. That is numpy's weighted quantile,
`np.quantile(x, q, weights=w, method="inverted_cdf")`. Under
`half_life=float("inf")` every row of the window weighs the same, and it is
the plain quantile of the window's values.

**It reads each value as a bucket, so it is within 0.78% of the exact
quantile, relatively.** A value `m · 2^e`, with `m` in `[1, 2)`, lands in
bucket `⌊64 (e + m − 1)⌋`; a negative value lands below zero by its
magnitude's bucket. So each octave (the range from one power of two to the
next) holds 64 buckets. A bucket spans a ratio of at most `e^(1/64)`, and the
value reported for it is within `tanh(1/128)`, about 0.0078, of every value
in it, at any scale and either sign. A magnitude below the smallest normal
double, about 2.2e-308, reads as 0. These are DDSketch's buckets (Masson,
Rim and Lee, 2019), the ones `rheon.spec.huber`'s MAD scale reads too.

**A row leaves a window that `window_size` cuts off without a subtraction.** The windowed
operators keep a window's rows in two stacks: new rows join the newer one,
and when the older one is empty the newer one turns over into it. A row
leaving must take its weight out of its bucket. A subtraction would leave a
rounding behind in a bucket that should be empty, and the roundings would
build up over a long stream. So at the turnover each row records three
weights, taken newest row first: what its bucket, its bucket's octave and
the whole sketch held over the rows newer than it. A row leaving puts those three weights back, and the sketch is
then exactly the one built from the rows that are left. A sketch kept per
row, the textbook form of a two-stack queue, would cost the whole sketch a
row: about 11 KB for a window of 1,000 rows, where this costs 24 bytes.
A read walks the buckets an octave at a time, then the one octave's 64
buckets, so a quantile costs about as much as its window's values span
octaves.

| the window | what it keeps | its sketches, on standard normal values | time a row, against the same window's `rp.ewm_mean` |
|---|---|---|---|
| no `window_size` | one sketch of every row since the stretch began, decayed, folding the decay in every 64 half-lives and then dropping an end octave that holds at most 1e-12 of the weight | 1,985 buckets, about 16 KB, at a half-life of 500 rows | 1.9 times |
| a `window_size` | the window's rows, as every windowed operator keeps them, three numbers a row, and two sketches | 1,345 buckets for a window of 1,000 rows | 1.7 times |
| `reverse=True` | as under a `window_size` | as under a `window_size` | 1.9 times `rp.ewm_sum`'s |

The times are the best of five runs over 2,000,000 rows of one group,
interleaved, through `rp.stream.with_windows`, measured on 2026-10-24 on a
shared laptop at a load average of 8 to 12.

[Weight rows by a robust scale of the target](how-to/weight-rows-by-a-robust-scale.md)
shows how to make a MAD scale with `rp.ewm_quantile`, and weight each row
by it without reading targets not yet known.

## Across groups at one stamp

**A cross operator reduces the rows of every group at one stamp of the
clock and gives the result to each of those rows.** `rp.cross_mean`,
`rp.cross_std`, `rp.cross_rank` and `rp.cross_quantile` read the stamp's
cross-section: its weighted mean, its spread, a row's rank within it, or a
level of it. Each gives the value Polars gives over the clock on the whole
frame, such as `pl.col("r").mean().over("t")` for the mean, while the
stream arrives in chunks of any size. A row less its stamp's mean is a
*cross-sectional residual*: what a symbol did beyond what every symbol did.
This code builds its own rows:

```python
# Three symbols' returns at three timestamps, the rows in clock order.
panel = pl.DataFrame(
    {
        "t": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "symbol": ["a", "b", "c"] * 3,
        "r": [0.010, 0.030, -0.010, 0.020, 0.000, 0.040, 0.000, -0.030, 0.030],
    }
)
out = rp.stream.with_windows(
    panel,
    resid=pl.col("r") - rp.cross_mean("r"),                         # less every symbol's mean at its stamp
    rank=rp.cross_rank("r"),                                        # 1 for the smallest at its stamp
    resid_before=pl.col("r") - rp.cross_mean("r", closed="left"),   # less the stamp before's mean
    clock="t", gap_cap=10, group="symbol",
)
print(out.select("t", "symbol", pl.col("resid", "resid_before").round(3), "rank"))
```

```text
shape: (9, 5)
┌─────┬────────┬───────┬──────────────┬──────┐
│ t   ┆ symbol ┆ resid ┆ resid_before ┆ rank │
│ --- ┆ ---    ┆ ---   ┆ ---          ┆ ---  │
│ i64 ┆ str    ┆ f64   ┆ f64          ┆ f64  │
╞═════╪════════╪═══════╪══════════════╪══════╡
│ 1   ┆ a      ┆ 0.0   ┆ null         ┆ 2.0  │
│ 1   ┆ b      ┆ 0.02  ┆ null         ┆ 3.0  │
│ 1   ┆ c      ┆ -0.02 ┆ null         ┆ 1.0  │
│ 2   ┆ a      ┆ 0.0   ┆ 0.01         ┆ 2.0  │
│ 2   ┆ b      ┆ -0.02 ┆ -0.01        ┆ 1.0  │
│ 2   ┆ c      ┆ 0.02  ┆ 0.03         ┆ 3.0  │
│ 3   ┆ a      ┆ 0.0   ┆ -0.02        ┆ 2.0  │
│ 3   ┆ b      ┆ -0.03 ┆ -0.05        ┆ 1.0  │
│ 3   ┆ c      ┆ 0.03  ┆ 0.01         ┆ 3.0  │
└─────┴────────┴───────┴──────────────┴──────┘
```

**Under `closed="right"`, the default, a row reads its own stamp, so it
waits until the stamp closes.** A run's rows are in clock order across its
groups, so a stamp's rows arrive together, and the first row of a later
stamp closes the stamp, or the end of the input does. A row whose stamp
comes before the one already open is refused, naming the row. Under
`closed="left"` a row reads the stamp before its own, which is known when
the row arrives. That form is causal: a feature it makes reads no row at
the row's own time, so a model can learn from it without the row's own
cross-section. Above, `resid_before` at `t = 2` subtracts the mean at
`t = 1`, 0.01, and the first stamp has none to subtract.

**A row of weight 0, or with no value, counts in no stamp. A cross-section
of fewer than two rows of weight is null.** One row has no spread, and
there is nothing to rank it against. A stamp that holds only such rows is
no stamp: under `"left"` the next row reads the stamp before it. A gap past
`gap_cap`, a session change or a reset ends the stamp before, as it ends
every window.

**`neighbors=` names a list column of group keys: each row reads the rows
of the groups its own list names, in place of every group's.** A transit
stop can read the stops beside it, and a stock the other stocks in its
sector. A row's own group counts only where its list names it. Each key
counts once, and a key with no row at the stamp counts for nothing.
`rp.cross_ref` reads one reference row's value at each stamp, the row its
`where=` predicate picks, such as an index's return beside each symbol's.
A stamp with two such rows is refused, naming the second row:

```python
# Four stops on a line, each reading the delay at the stops beside it, and the line's first stop.
delays = pl.DataFrame(
    {
        "t": [0, 0, 0, 0, 1, 1, 1, 1],
        "stop": ["s1", "s2", "s3", "s4"] * 2,
        "delay": [2.0, 4.0, 6.0, 9.0, 3.0, 3.0, 7.0, 11.0],
        "beside": [["s2"], ["s1", "s3"], ["s2", "s4"], ["s3"]] * 2,
    }
)
out = rp.stream.with_windows(
    delays,
    beside_mean=rp.cross_mean("delay", neighbors="beside"),          # null where one stop is beside
    first_stop=rp.cross_ref("delay", where=pl.col("stop") == "s1"),
    clock="t", gap_cap=10, group="stop",
)
print(out.select("t", "stop", "beside_mean", "first_stop"))
```

```text
shape: (8, 4)
┌─────┬──────┬─────────────┬────────────┐
│ t   ┆ stop ┆ beside_mean ┆ first_stop │
│ --- ┆ ---  ┆ ---         ┆ ---        │
│ i64 ┆ str  ┆ f64         ┆ f64        │
╞═════╪══════╪═════════════╪════════════╡
│ 0   ┆ s1   ┆ null        ┆ 2.0        │
│ 0   ┆ s2   ┆ 4.0         ┆ 2.0        │
│ 0   ┆ s3   ┆ 6.5         ┆ 2.0        │
│ 0   ┆ s4   ┆ null        ┆ 2.0        │
│ 1   ┆ s1   ┆ null        ┆ 3.0        │
│ 1   ┆ s2   ┆ 5.0         ┆ 3.0        │
│ 1   ┆ s3   ┆ 7.0         ┆ 3.0        │
│ 1   ┆ s4   ┆ null        ┆ 3.0        │
└─────┴──────┴─────────────┴────────────┘
```

`s1` and `s4` each list one stop, and one row is no cross-section, so each
reads null. `s3` lists `s2` and `s4`: at `t = 0` it reads (4 + 9) / 2.

**The state holds the rows of the open stamp, within `max_state_bytes`.**
Under `"right"` a stamp's rows wait for it, so the state grows with the
rows one stamp holds, never with the stream's length. The run warns once,
as a `RheonWarning`, when a stamp takes it past half of the cap, and
refuses the row that would take it past the cap. A rank, a quantile and a
read over neighbors also keep each row's value, and under `"left"` the
stamp before's; each operator's docstring gives its bytes a row.

**A cross-section needs every group's rows in one place.** A stream
partitioned by group, as a Kafka topic keyed by group is, keeps each
group's rows in order but splits the stamp across partitions. A cross
operator therefore runs in one consumer that reads every group's rows,
such as from a topic with a single partition, after the per-group work.

## A profile by the phase of a period

**`rp.ewm_profile` reads a column's typical value at each row's phase of a
period: its hour of the day, its day of the week.** It keeps a decayed
mean, variance or quantile for each phase of each group, or one smooth
Fourier curve over the whole period (`harmonics=`), and gives each row the
value at its own phase. A seasonal adjustment is an expression around it:
`x - profile` for an additive one, `x / profile` or `x / std_profile` for a
multiplicative one. Under `closed="left"` a row's profile holds no row of
its own time, so the adjustment reads only what was known before the row.
This code builds its own rows:

```python
from datetime import datetime, timedelta

# Delays every 6 hours for three days, worse at the morning rush (06:00),
# with the clocks of Amsterdam.
start = datetime(2026, 3, 27)
delays = pl.DataFrame(
    {
        "t": [start + timedelta(hours=6 * i) for i in range(12)],
        "delay": [1.0, 8.0, 3.0, 2.0, 2.0, 10.0, 4.0, 2.0, 1.0, 9.0, 3.0, 3.0],
    }
).with_columns(pl.col("t").dt.replace_time_zone("Europe/Amsterdam"))
out = rp.stream.with_windows(
    delays,
    usual=rp.ewm_profile("delay", period="1d", buckets=4, half_life="7d", closed="left"),
    clock="t", gap_cap="1d",
)
out = out.with_columns(excess=pl.col("delay") - pl.col("usual"))  # less what that time of day had shown
print(out.tail(4).select(pl.col("t").dt.strftime("%m-%d %H:%M"), "delay", pl.col("usual", "excess").round(2)))
```

```text
shape: (4, 4)
┌─────────────┬───────┬───────┬────────┐
│ t           ┆ delay ┆ usual ┆ excess │
│ ---         ┆ ---   ┆ ---   ┆ ---    │
│ str         ┆ f64   ┆ f64   ┆ f64    │
╞═════════════╪═══════╪═══════╪════════╡
│ 03-29 00:00 ┆ 1.0   ┆ 1.52  ┆ -0.52  │
│ 03-29 06:00 ┆ 9.0   ┆ 9.05  ┆ -0.05  │
│ 03-29 12:00 ┆ 3.0   ┆ 3.52  ┆ -0.52  │
│ 03-29 18:00 ┆ 3.0   ┆ 2.0   ┆ 1.0    │
└─────────────┴───────┴───────┴────────┘
```

The 06:00 rows read the rush hour's own level, about 9, so a delay of 9
there is no excess, where the same delay at midnight would be one.

**A row's phase is read on the wall clock in the clock column's own time
zone, as Polars' `dt` namespace reads a zoned value.** The rows above are
at 00:00, 06:00, 12:00 and 18:00 Amsterdam time on each day, and the third
day is the one the clocks go forward: its 06:00 is five hours after its
midnight, and still the rush hour's phase. A phase is `t − t.dt.truncate(period)`
less an `offset`, so a week starts on Monday, as Polars truncates one. A
calendar period (`"1mo"`) has no fixed length, and is refused.

**A phase is read once it holds `min_samples` rows, 1 by default**, counted
as `rolling_*_by` counts them, so a phase no row has reached is null.
Each phase decays by the clock's age, seen or not, so a phase unseen for
long weighs little against a fresh one. Under a `window_size` each phase
keeps only its rows inside the window, in two stacks, so a row leaves
without a subtraction, as it leaves `rp.ewm_mean`'s window. The harmonic
form is null where its rows do not pin the curve down (fewer distinct
phases than its `2 · harmonics + 1` terms), and its variance where the fit
of `x²` falls below the square of the fit of `x`.

**A group's state holds its phases, bounded by the period's shape and the
window, never by the stream's length**, and held to `max_state_bytes`:
56 bytes a bucket, and under a window each row inside it; the operator's
docstring gives each form's bytes. A call whose buckets pass the cap is
refused before it reads a row; a state the data grows (a window's rows, a
quantile's sketches) warns once at half the cap and refuses the row that
would pass it.

## The largest and smallest value, and since an event

**`rp.rolling_max` and `rp.rolling_min` are the largest and smallest value
the window holds.** The window is the one `rolling_max_by` holds under each
`closed`, and the value is Polars' to the bit. On equal values that differ
only as `0.0` and `-0.0` the earliest row of the window wins, as in Polars.
Looking ahead the latest wins, so a forward window is the backward one over
the reversed stream. A NaN, an infinity or a value beyond 1e100 is read as
missing, as every operator reads it, where Polars' `rolling_max_by` lets a
NaN make the window NaN and counts an infinity.

**`since` starts the window over at an event.** It is an element-wise
boolean of the row, such as `pl.col("new_period")`. A stamp that holds a
true row, at any of its rows, starts the window at that stamp: every row of
the stamp is in, every earlier row is out, so the rows of one stamp never
get two values. Under `closed="left"` or `"none"` the event's stamp is null,
since its own rows are outside its window and the event emptied the rest.
With a `window_size`, a window the event cut before its full span is
partial, under `partial`. The event acts on its operator alone, and under
`group` each group's events are its own. A gap past `gap_cap` or a session
change starts the running form over too, as it does every operator; a
`gap_cap` wide enough keeps a high-water mark across a long gap. `since`
does not combine with `reverse=True` yet: a window looking ahead would end
at the next event, and whether such a window is whole has no rule yet.

**The drawdown subtracts the high-water mark from the value.** This code makes
both from a column of equity:

```python
import polars as pl
import rheon.polars as rp

rows = pl.DataFrame({
    "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "equity": [1.0, 3.0, 2.0, 4.0, 1.0, 2.0],
    "new_period": [False, False, False, False, True, False],
})
peak = rp.rolling_max("equity", since="new_period")
out = rp.stream.with_windows(
    rows, peak=peak, drawdown=pl.col("equity") - peak, clock="t", gap_cap=10.0
)
print(out["drawdown"].to_list())
```

```text
[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
```

Its running worst, the maximum drawdown so far, is `rp.rolling_min` of the
`drawdown` column in a second call, since an operator's input holds no
other operator.

**What it holds.** With no `window_size`, one running extremum and its
count a group: a few doubles, whatever the stream's length. With a
`window_size`, the window's rows, as every windowed operator keeps them,
and one double a row more for the extremum. A row costs 112 bytes for one
extremum on its own window: 40 for the row, 64 for the operator's value
and sums, 8 for the extremum. Each other operator sharing the window adds
64 bytes a row, and another extremum 72. A window on a clock holds every
row inside its span, so a burst of rows fills it. `max_state_bytes` bounds
it, 256 KB a group by default and at most 64 MiB. Without a clock column
the window's rows are known, and a window past the cap is refused when the
query is built. On a clock, the row that takes a group's window past it is
refused.

## Saving and resuming a window run

**A window run's state holds the rows still waiting for their windows to
close, so a run resumed from it returns those rows before its own.** A
window that looks back and waits for the next stamp is not one of them. At
the end of the input it is read as it stands. A resumed run refuses an input
that starts at the last stamp, so no row can join it. Two runs then give what
one run gives. [Resume a window
run](how-to/resume-a-window-run.md) shows how to save and load the state,
and which inputs a resumed run takes or refuses.
