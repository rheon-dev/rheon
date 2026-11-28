<!-- kind: how-to -->
# Merge streams by the clock

This how-to is for anyone with several streams that each tick at their own
times, such as quotes and trades read from files of their own, who wants
them as one stream in clock order before a bank or a window run. It merges
them with `rp.stream.merge`, which keeps every row at its own clock and
tags it with its stream, and resumes the merge when the streams arrive in
parts. To put such series on a common grid instead, one row per instant
every series has ticked by, see [Align series that tick at their own
times](align-series-that-tick-at-their-own-times.md).

## Prerequisites

These steps need the library and the streams:

- rheon installed with Polars.
- Each stream in [clock](../GLOSSARY.md#time) order across all its rows,
  its groups included, as a file or a query. [Put rows in clock
  order](put-rows-in-clock-order.md) shows how to sort one.
- One clock column, of one name and one dtype in every stream: the same
  unit and time zone on a `Datetime`. A column that two streams share must
  have one dtype in both.

## Merge the streams

The merge reads a chunk of each stream at a time, and gives the rows of all
of them in clock order. A stream in clock order has its earliest row still
to come at its head, so the earliest of the heads is the next row out. The
memory it holds is one chunk of each stream, however long they are.

1. **Call `rp.stream.merge` with the streams and the `clock`.** Pass the
   streams in a `dict`, each under the name its rows will carry, or by
   position, named `"0"`, `"1"` and on. Given a query, it returns one, so
   the merge runs as the query streams.
2. **Give `group` the column a bank downstream groups by, if it has one.**
   The merged rows of each group are then checked as that bank checks
   them, as [Check that it worked](#check-that-it-worked) says.
3. **Read the merged stream.** Its columns are every stream's, matched by
   name, then `source`, the name of the stream each row came from. A column
   a stream does not have is null on its rows. At one clock value the
   stream given first comes first, and each stream's rows keep their order.

This code writes quotes and trades of two symbols to files of their own,
and merges them:

```python
from datetime import datetime

import polars as pl
from rheon import polars as rp

def at(second):                                        # a time on the morning of 2 May 2024
    return datetime(2024, 5, 2, 9, 30, second)

# Write the quotes and the trades to files of their own.
pl.DataFrame({                                         # quotes, in clock order
    "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA"],
    "ts": [at(0), at(1), at(3), at(3), at(6)],
    "bid": [10.00, 20.00, 10.01, 19.99, 10.02],
}).write_parquet("quotes.parquet")
pl.DataFrame({                                         # trades, in clock order
    "symbol": ["AAA", "BBB", "AAA"],
    "ts": [at(2), at(3), at(5)],
    "price": [10.01, 20.00, 10.02],
    "size": [100, 50, 200],
}).write_parquet("trades.parquet")

# Merge them by the clock, the rows of each symbol checked together.
streams = {"quotes": pl.scan_parquet("quotes.parquet"), "trades": pl.scan_parquet("trades.parquet")}
merged = rp.stream.merge(streams, clock="ts", group="symbol")    # 1. and 2.: a query
# Read the merged stream.
out = merged.collect()
print(out.columns)                                             # 3.
for ts, symbol, source, bid, price in out.select("ts", "symbol", "source", "bid", "price").rows():
    print(ts.time(), symbol, source, bid, price)
```

```text
['symbol', 'ts', 'bid', 'price', 'size', 'source']
09:30:00 AAA quotes 10.0 None
09:30:01 BBB quotes 20.0 None
09:30:02 AAA trades None 10.01
09:30:03 AAA quotes 10.01 None
09:30:03 BBB quotes 19.99 None
09:30:03 BBB trades None 20.0
09:30:05 AAA trades None 10.02
09:30:06 AAA quotes 10.02 None
```

At 09:30:03 the two quotes come before the trade, since `quotes` is given
first. A bank chained after `merged` with `merged.rheon.fit_predict(specs)`
reads the merged stream in the same pass.

## Carry each stream's last value forward

Use this when a row of one stream needs another stream's latest value, such
as the bid at the time of each trade, and how old that value is, so that a
value too old to trust is set aside.

1. **Give `carry` the columns to fill forward.** On a row where such a
   column is null, the merge fills in its last value in the merged rows
   before it, of the row's `group`. A null in a stream is not a value,
   so the last value before it is carried on, as Polars' `forward_fill`
   carries it. Before a group has had a value, the column stays null.
2. **Read the age beside each carried column.** The column `<c>_age` is the
   row's clock less the clock of the row the value came from: 0 on a row
   with a value of its own, a `Duration` on a temporal clock, and null
   while the value is.
3. **Write the threshold in Polars.** A value older than you trust is a
   condition on its age, such as `pl.col("bid_age") <= pl.duration(seconds=1)`.

This code carries the bid of each symbol to its trades, and keeps it only
where it is at most a second old:

```python
carried = rp.stream.merge(streams, clock="ts", group="symbol", carry="bid")   # 1.
trades = carried.filter(pl.col("source") == "trades").with_columns(
    pl.when(pl.col("bid_age") <= pl.duration(seconds=1))                   # 2. and 3.
    .then(pl.col("bid"))
    .alias("fresh_bid"),
)
for ts, symbol, price, bid, age, fresh in trades.select(
    "ts", "symbol", "price", "bid", "bid_age", "fresh_bid"
).collect().rows():
    print(ts.time(), symbol, price, bid, age, fresh)
```

```text
09:30:02 AAA 10.01 10.0 0:00:02 None
09:30:03 BBB 20.0 19.99 0:00:00 19.99
09:30:05 AAA 10.02 10.01 0:00:02 None
```

The trades of AAA meet a bid two seconds old, and the threshold sets it
aside. The BBB trade at 09:30:03 meets a bid of its own instant, since
`quotes` is given first. Under `restart_after_step_back` a value is carried
within its day: a stream that starts again carries nothing from before,
since a clock that started again does not compare with the one before it.
A state saved with `carry` keeps each group's last value of each column,
one row a group however long the streams, so a resumed run carries what
the last run left.

## Merge streams that arrive in parts

Use this when each stream arrives in parts, such as a file a day, and the
parts must give what one merge over the whole streams gives.

1. **Cut every stream at the same clock.** A part of one stream that ends
   earlier than another's would let the next part's rows land before rows
   the last run already returned.
2. **Give the first part's run `save_state=` a file name.** The run
   returns its rows, but the rows at the last clock it read, which it
   saves in the state. The next part of a stream given before theirs in the
   call may bring more rows at that clock, and those rows come first.
3. **Give the next part's run `load_state=` the same file.** It returns the
   rows the state held in their place, then its own.

This code cuts both streams after 09:30:03 and compares the two runs with
the one above:

```python
cut = at(3)                                                    # 1. every stream cut at one clock
first = {k: pl.scan_parquet(f"{k}.parquet").filter(pl.col("ts") <= cut) for k in streams}
rest = {k: pl.scan_parquet(f"{k}.parquet").filter(pl.col("ts") > cut) for k in streams}
part1 = rp.stream.merge(first, clock="ts", group="symbol", save_state="merge.state").collect()  # 2.
part2 = rp.stream.merge(rest, clock="ts", group="symbol", load_state="merge.state").collect()   # 3.
print([str(ts.time()) for ts in part1["ts"]])                  # the rows at 09:30:03 are held
print([str(ts.time()) for ts in part2["ts"]])                  # and come first in the next part
print(pl.concat([part1, part2]).equals(out))                   # the two parts give the one run
```

```text
['09:30:00', '09:30:01', '09:30:02']
['09:30:03', '09:30:03', '09:30:03', '09:30:05', '09:30:06']
True
```

The state is one small file: each stream's last clock, the merged rows'
last clock in each group, and the rows held, those of one clock value.
`rp.stream.save_checkpoint` saves it beside a bank's state as one file, as
[Resume a window run](resume-a-window-run.md) does with a window run's.

## Check that it worked

**Read `source` and the clock of the first rows of each part.** Each row
carries the stream it came from, and the merged clock never steps back.

| what you see | what it means | what to do |
|---|---|---|
| an error naming a stream and a row whose clock steps back | that stream is not in clock order across all its rows: the merge needs each stream in order to hold one chunk of it at a time | sort the stream by the clock; or, for a clock that starts again each session, give `restart_after_step_back`, below which a step back is still refused and past which the stream starts again |
| an error that the clocks are not of one kind, naming two streams | their clock columns differ in dtype, unit or time zone, so their values do not compare | cast every stream's clock to one dtype, such as `pl.col("ts").dt.cast_time_unit("us")` |
| an error that a column is of two dtypes in two streams | a merged column holds one dtype | cast the column to one dtype in every stream, or rename it in one |
| an error that the merged rows step back, naming a stream and its row | a resumed run's streams were cut at different clocks, or, under `restart_after_step_back`, a stream did not start again where the others did | cut every stream at the same clock, and start every stream again at the same places, such as each day's open |
| fewer rows than the streams hold, from a run with `save_state` | the rows at the last clock read are held for the next part | load the state in the next part's run, which returns them first |
| a carried column null on rows after the stream had a value | the row is of another `group`, or of a day that started again, which has had no value yet | carry a column within the groups the bank reads; under `restart_after_step_back` the first rows of a day wait for that day's first value |

Every parameter is in `merge`'s
[docstring](https://rheon-dev.github.io/rheon/stream.html#rheon.polars.stream.merge).
