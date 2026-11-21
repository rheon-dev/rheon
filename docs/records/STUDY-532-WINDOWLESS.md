# Study 532b: rolling statistics that keep no window of rows

PLAN task 532, the windowless part. The user decided that no resumable feature
may keep a window of rows, so study 532's ring and monotone-deque forms
(branch `vm/study-532`, `docs/records/STUDY-532-FEATURES.md`) are not built or
proposed here. This study asks what can stand in for a rolling statistic with
a state bounded by the feature's shape alone: (1) a time span cut into `B`
fixed buckets of the clock, each bucket holding a few sums, never a row;
(2) the exponentially weighted forms, including a decayed min and max
envelope; (3) lags, which cannot be windowless, and how to serve them anyway.

numpy, scipy and Python only, against Polars 2.0.0 (the version `uv.lock`
pins); nothing in the crates is changed. Code: `studies/study_532b/`
(`bucket.py`, `ew.py`, `lag.py` the states; `runs_*.py` the checks;
`exact.py` the exact references; `report.py` the tables). Entry point:
`python scripts/study_532b.py` (75 s; `--quick`, `--part=N`).
Every table below is its output.

## Answer

1. **Bucketed windows match Polars to the bit wherever they compute the same
   quantity, for count, min and max.** On every window of `group_by_dynamic`
   (`every = b`, `period = B b`, `offset = -(B - 1) b`) and on every row of
   `rolling_*_by(window_size = B b)` whose time is the last tick of its bucket,
   count, min and max have 0 bit differences and 0 validity differences, with
   nulls, zero-weight rows, NaN and both infinities (Tables B1, B3; B3 itself
   is on finite data, and the non-finite rows are checked in "Sonnet rerun"
   and in Table R1), when the time rises strictly within a group. Two limits,
   found in the rerun: at `B = 1`, min and max differ on the windows that hold
   a NaN (14 windows of 242; Polars' tumbling min and max skip a NaN; the
   count is exact); and where a time repeats, a row that is not the last of
   its time sees only the rows before it, where Polars gives every row of the
   time the window of the last one (118 of 1211 aligned rows differ in count).
2. **Sum, mean and variance cannot match Polars' bits and do not need to:**
   Polars sums a window's rows in its own order (an add-and-subtract
   compensated sum for overlapping windows), which a state with no rows cannot
   repeat. The bucketed sum (a compensated sum per bucket, combined with
   Neumaier's step) was the correctly rounded sum on every window measured,
   0 ulps from `math.fsum`; Polars' was up to 62 ulps from it, and that is the
   whole gap between the two (Table B2). Mean: 1 ulp from exact against
   Polars' 62. Variance (centred sums per bucket, Chan's merge): up to
   1.2e4 ulps from exact (relative 2.4e-12 at a level of 1e3; 9.0e-12 on
   another seed) against Polars' 1.9e4, with a median of 57 to 117 against
   108 to 119 (lower on four of the five overlapping shapes, 2 ulps higher at
   `b = 1`, `B = 60`), on overlapping windows; on windows that do not overlap (`B = 1`) Polars computes the
   variance from the rows in two passes and is within 2 ulps of exact, the
   buckets 9277. The plain sum
   of squares the request named cancels: 2.6e8 ulps (relative 4e-8) at a level
   of 1e3 and relative 0.026 at 1e6, so the bucket keeps the centred form
   instead, at the same 8 bytes (Table B2, second part).
3. **The one genuine approximation is the window's old edge inside a bucket.**
   At a row of time `t` in bucket `k`, the `B` buckets cover `[(k-B+1) b, t]`,
   between `(B-1) b + 1` and `B b` ticks; the span `(t - B b, t]` reaches
   into bucket `k - B`. With 12 buckets across the span, the mean is within
   0.10 to 0.14 of the span's standard deviation on 95% of rows (largest
   0.39), the standard deviation within 7 to 11% relative, and min and max
   differ on 3.7 to 5.7% of rows; with 4 buckets, 0.30 to 0.35 sd and 7 to 15%
   of rows;
   with 1 bucket the read is a tumbling window and the gap is large (Table B4).
   Reading `B + 1` buckets covers the span instead, so the span's max lies
   between the two reads (0 violations), and an interpolated read (the bucket
   `k - B` weighted by the share of it inside the span, as if its rows were
   spread evenly) cuts the count and mean gaps by a factor of 2 to 8 on 95% of
   rows. Min, max and the variance have no interpolated form.
4. **State, with no row stored (Table B5):** `8 + 48 (B + p)` bytes per
   column and group, `p = 1` when the bucket before the window is kept for the
   `B + 1` and interpolated reads: 200 bytes for `B = 4`, 584 for 12, 2888 for
   60. It does not move with the rows per bucket. A bucket costs as much as
   six rows of a ring, so buckets pay when a bucket holds more than about six
   rows; finer than that, the bucketed state is larger than a ring would be.
5. **The EW features equal Polars to the bit**: `ewm_mean`, `ewm_var`,
   `ewm_std` under each `adjust`, `bias`, `ignore_nulls`, `min_samples`
   tried, and `ewm_mean_by` on two half-lives: 0 bit and 0 validity
   differences on 11 outputs (Table E1), with alpha carried in the state.
6. **The carried alpha.** alpha is computed once, when the feature is made,
   from Polars' own `1 - exp(-ln 2 / half_life)`, and from then on is read from
   the saved bytes (8 bytes: `ewm_mean` 32 bytes, `ewm_var`/`ewm_std` 56). A
   state loaded on another OS therefore continues to the bit (0 ulps, Table E2;
   Table R1 loads every EW state into an object made with alpha 0.5 and still
   matches). Recomputing alpha on a libm one ulp off moves the continuation by
   up to 612 ulps (`ewm_var`, `adjust=False`), 172 (`ewm_std`), 1
   (`ewm_mean`). `ewm_mean_by` calls `pow` on every row and cannot carry it;
   it stays at study 532's 4 ulps across OSes.
7. **Decayed min and max envelope.** The literature has one that suits a
   level: the branching peak detector with instantaneous attack (Giannoulis,
   Massberg and Reiss, "Digital Dynamic Range Compressor Design -- A Tutorial
   and Analysis", J. Audio Eng. Soc. 60(6), 2012): jump to a new extreme,
   otherwise release toward the value, `y = r y + (1 - r) x`. Built, 17 bytes.
   It is not an estimate of `rolling_max`: on the walk and on white noise it
   sits on average 0.66 to 1.41 rolling standard deviations below
   `rolling_max(w)` for half-lives `w/8` to `w/2` (smaller for longer
   half-lives), is above it on at most 1.5% of rows, and tracks it closely on a
   trending series (correlation 0.996 to 0.9997) but loosely on noise (0.22 to
   0.61) (Table E3). The time-decayed max of Cormode et al. ("Forward Decay",
   ICDE 2009), `max(x, r y)`, decays toward 0, not toward the data: on a
   series about 1e3 it is the current value on every row, and on a
   non-negative series it sits 1.5 sd below `rolling_max`. No O(1) state can
   give the exact windowed max: feed `-inf` after a falling window and the
   outputs read back every value in it, so such a state holds `w` f64.
8. **Lags hold `k` values; nothing smaller can serve them** (the next `k`
   outputs are the last `k` values, any f64 each, so the state determines
   `8k` bytes). The state equals Polars' `shift`, `diff`, `pct_change` to the
   bit (Table L1) and saves `8 + 8k + ceil(k/8)` bytes: 17, 49 and 171 for
   `k = 1, 5, 20` (Table L2). **Recommended: the bank keeps no lag; a lag is an
   input column the user computes with Polars before the bank.** See "Lags"
   below for the trade-off and the cost of each way.
9. **Chunk and resume invariance** (Table R1): 21 features, every state saved
   to bytes and loaded at every cut, cut into 2, 17, ~300 chunks and after
   every row: all 21 equal the uncut stream to the bit, and the 18 with a
   Polars counterpart equal one Polars pass to the bit, on the clean frame and
   on the one with NaN and infinities.
10. **Cost** (Table C1): the Python states take 0.15 to 0.65 us a row for an
    EW feature, an envelope or a lag, and 9 to 27 us for the six bucketed
    outputs (the read combines `B` buckets: 3.8 us at `B = 4`, 48 us at 240).
    Polars takes 11 to 17 ns for the EW features and 51 to 70 ns for the five
    `rolling_*_by` over the same span.

## Definitions

Notation: a group's rows in arrival order; value `x` (or null); time `t`
(int64). Each update is the state's whole change for one row; the output is
read after it.

### Conventions (every feature)

- A **zero-weight row** is a null to every feature, as in study 532: it keeps
  its slot on the row clock and its time on the time clock (so it moves a
  bucketed window to its bucket, decays an EW weight when
  `ignore_nulls=False`, and takes a lag's slot), and none of its value enters
  a state. The check is Polars on the input with zero-weight rows' values set
  to null; the stream feeds `None` for them.
- A **null** is not a value; a **NaN** is, as in Polars. In a bucketed window a
  NaN or an infinity propagates through every statistic of every window that
  holds its bucket, and leaves with the bucket: a bucket is never subtracted,
  so plain IEEE arithmetic does what Polars' counters do. In an EW feature or
  an envelope a NaN poisons the state for good, as it does in Polars.
- The **clock**: a bucketed window needs only `floor(t / b)` not to fall
  within a group; a repeated time, or times out of order inside the current
  bucket, land in the same bucket. So a read after the last row of a bucket is
  the same whichever order the bucket's rows came in. The repeated-time
  problem of study 532's span windows (Polars gives each row of a repeated
  time the window of the last row at that time) does remain row by row: a row
  that is not the last of its time is read from the rows up to it, so a
  per-row comparison with `rolling_*_by` holds only at the last row of each
  time (Sonnet rerun, check h). `ewm_mean_by` takes a repeat (`dt = 0`).

### Bucketed windows

Bucket `k` of width `b` holds the rows with `floor(t / b) = k`. Per bucket and
column, five numbers and a count:

- `n`, the values that entered;
- `(s, c)`, a compensated sum (Neumaier): entering `x`, `s' = s + x`, and
  `c += (s - s') + x` when `|s| >= |x|`, else `c += (x - s') + s`;
- `m2`, the centred sum of squares by Welford's step with the bucket's mean
  read from its sum, `m2 += (x - mean) (x - mean')`, `mean = (s + c) / n`;
- `min` and `max`, NaN-propagating (a NaN beats any number, as in Polars'
  `rolling_min`/`rolling_max` and the overlapping `group_by_dynamic`).

The state is a ring of `B` slots (`B + 1` with the bucket before the window)
and the current bucket id. Moving the clock to bucket `k'` clears the slots of
the buckets passed, at most `B + 1` of them. **Reads**, at a row in bucket
`k`:

- **`B` read**: combine buckets `k - B + 1 .. k`, oldest first: `count = sum
  n`; the sum by Neumaier over the buckets' `s` and `c` (a bucket whose `s` is
  not finite contributes `s` alone); `mean = sum / count`; the variance by
  Chan's merge of `(n, mean, m2)`, `d = mean_b - mean_a`,
  `mean = mean_a + d n_b / n`, `m2 = m2_a + m2_b + d^2 n_a n_b / n`, then
  `m2 / (count - ddof)`, null when `count <= ddof`; min and max over the
  buckets. Null below `min_samples` values.
- **`B + 1` read**: the same over `k - B .. k`.
- **Interpolated read** (count, sum, mean): the `B` read plus `f` times bucket
  `k - B`'s count and sum, `f = (b - 1 - t mod b) / b`, the share of that
  bucket's ticks inside `(t - B b, t]`. It assumes the bucket's rows are spread
  evenly over its ticks (the "sliding window counter" estimate).

**What it computes and what it approximates.** Read at the end of bucket `k`,
it is exactly the statistic of the rows in `[(k-B+1) b, (k+1) b)`: Polars'
`group_by_dynamic` window, and Polars' `rolling_*_by` span at any row whose
time is the bucket's last tick. Read at another row, it is exactly the
statistic of `[(k-B+1) b, t]`, a window whose length moves between
`(B-1) b + 1` and `B b` ticks with the row's place in its bucket: a
well-defined feature (a span aligned to the clock grid), not a noisy estimate,
but not Polars' sliding span. The gap to the sliding span is Table B4's; it
shrinks as `B` grows for a fixed span, at 48 bytes a bucket.

### EW features

The updates are study 532's, which mirror Polars' kernels
(`polars-compute/src/ewm/{mean,cov}.rs`, `polars-ops/src/series/ops/ewm_by.rs`)
and are in `ew.py`'s docstrings. The one change: alpha is a field of the saved
state, read back at load, never recomputed. A spec carries `half_life`; the
first state made from it computes alpha with Polars' expression, which is what
Polars itself would compute on the same OS.

### Decayed envelopes

- **Max envelope** (min: reverse the comparison): first value `y = x`; then
  `y = x` if `x > y`, else `y = r y + (1 - r) x`, `r = 1 - alpha` carried.
  A null row moves nothing and gives null. Translation- and scale-equivariant.
  Its bias against `rolling_max(w)` is Table E3's: below it on average, by
  less for longer half-lives; it is a smoother, different feature, not an
  estimate of the windowed max.
- **Decayed max** (Cormode et al.): `y = max(x, r y)`, decayed on every row
  (null rows too). Only for non-negative quantities whose 0 means "none".

### Lags

Study 532's `Lag`: the last `k` values with their validity, and how many rows
were seen up to `k`; `shift`, `diff = x - x_{-k}`,
`pct_change = (x - x_{-k}) / x_{-k}`.

## Lags: can they be served with no row in the bank?

Not by the bank: any state that serves a lag of `k` determines the last `k`
values, so it is at least `8k` bytes, and the ring above is that plus a few
bytes. The ways to serve a lag with the bank keeping nothing all move the `k`
values somewhere else:

| way | exact | bank state | who holds the `k` values | cost |
|---|---|---|---|---|
| the bank keeps a lag state | yes, chunk and resume invariant (Table R1) | `8 + 8k + ceil(k/8)` bytes a column and group (17, 49, 171) | the bank | a window of `k` rows in the state, which the user's rule excludes |
| the user lags with Polars `shift(k).over(g)` over the whole frame before the bank (one `LazyFrame` plan) | yes | none | Polars | Polars holds the frame for `over`: about 50 bytes a row measured, 2.2 GB at 40M rows (Table L4); no bound by `k` |
| the user lags each chunk before the bank, carrying the last `k` rows of each group to the next chunk | yes, to the bit (Table L3) | none | the user's loop (`k` rows a group) | the user writes the carry; forgetting it gives wrong lags at every chunk start (up to 14274 of 20000 rows for `k = 20` and 300 chunks, Table L3) and breaks chunk invariance silently |
| the lag is a column of the source, computed once offline | yes | none | the source | an offline pass sorted by group; a lag fixed at write time |
| a windowless substitute: the value's deviation from its EW mean before the update, `x - m_prev` | not a lag | the EW mean's 32 bytes | nobody | a different feature: a decayed "change" with no fixed horizon |

**Recommendation: the bank keeps no lag.** Lags enter the bank as input
columns: written as `shift(k).over(g)` in the frame the bank reads when it
fits (Polars computes them exactly), and precomputed in the source when it
does not. A chunk-fed caller carries the last `k` rows a group itself, and the
bank's documentation must show that carry, because a lag computed chunk by
chunk without it is wrong at every chunk start and nothing fails. If a
chunk-fed lag inside the bank is wanted after all, the lag state above is the
smallest that can exist (`8k` bytes plus 9 for `k <= 8`), and it is the user's
call as an exception to the no-window rule.

## Results

The script's full output follows. Notes:

- **Table B1** checks every `group_by_dynamic` window, including windows whose
  newest bucket has no row (the state is moved to that bucket with no row and
  read; that changes nothing a later row sees). Windows whose rows are all
  null are held apart: there Polars' overlapping-window sum is the residue of
  its add-and-subtract sum, up to 1.5e-12 instead of 0, which the bucketed
  sum does not have. With `B = 1` (windows that do not overlap) Polars'
  `min`/`max` skip NaN, as `Series.max` does, while with `B > 1` they
  propagate it, as `rolling_max` does; the bucketed state propagates, so the
  14 windows holding a NaN differ at `B = 1` only.
- **Table B2** is against exact arithmetic. The bucketed sum's 0 ulps is
  measured, not guaranteed: Neumaier's step is not a correctly rounded sum in
  general, but its error is a few ulps of the sum's magnitude at worst, far
  below Polars' add-and-subtract drift.
- **Table B4**'s units: count gaps relative to the span's count; mean, min and
  max gaps in units of the span's standard deviation (so a random walk and
  white noise read alike); sd relative. Rows with fewer than 3 values in the
  span are left out.
- **Table B5**'s sizes are those of this study's packed struct, read from
  `to_bytes()`; the size is fixed by the layout (the ring is a fixed number of
  slots), so "does not move with the rows" holds by construction, not by a
  growth measurement, and the last column is computed, not measured. A msgpack
  form writes each f64 as 9 bytes (a tag and the 8), unless the slots are
  written as one binary blob.
- **Table E3** compares with Polars' `rolling_max(w)` and `rolling_min(w)`; the
  envelopes have no Polars counterpart.
- **Table L4** measures the process's peak resident memory, each case in its
  own process. Every case grows with the rows in Polars 2.0.0's streaming sink
  here, the plain copy too; `over(g)` grows at about 50 bytes a row (the frame
  is held), and plain `shift` faster than the copy.

### Versions

| python | numpy | scipy | polars | machine |
|---|---|---|---|---|
| 3.13.18 | 2.5.2 | 1.18.1 | 2.0.0 | Linux x86_64 |

### Table B1: bucketed windows against `group_by_dynamic`, every window (10000 rows, 5 groups)

Each cell: validity differences / bit differences / largest ulp gap, over the windows that hold a value. `sum` and `mean` read the compensated sums; `var` the centred form. Second block: NaN, +inf, -inf in 0.4% of x.

| non-finite | b | B | windows | count | sum | mean | var | min | max | windows with no value (Polars sum not 0, largest |sum|) |
|---|---|---|---|---|---|---|---|---|---|---|
| no | 1 | 60 | 36852 | 0/0/0 | 0/9435/16 | 0/9055/16 | 0/35258/18785 | 0/0/0 | 0/0/0 | 50 (21, 1.5e-12) |
| no | 5 | 12 | 7366 | 0/0/0 | 0/1729/16 | 0/1660/16 | 0/7042/20683 | 0/0/0 | 0/0/0 | 8 (3, 1.5e-12) |
| no | 15 | 4 | 2455 | 0/0/0 | 0/453/13 | 0/428/13 | 0/2304/20683 | 0/0/0 | 0/0/0 | 1 (0, 0.0e+00) |
| no | 20 | 12 | 2905 | 0/0/0 | 0/633/62 | 0/608/62 | 0/2778/18555 | 0/0/0 | 0/0/0 | 0 (0, 0.0e+00) |
| no | 60 | 4 | 971 | 0/0/0 | 0/177/22 | 0/171/22 | 0/906/18555 | 0/0/0 | 0/0/0 | 0 (0, 0.0e+00) |
| no | 240 | 1 | 243 | 0/0/0 | 0/136/3 | 0/132/5 | 0/233/9278 | 0/0/0 | 0/0/0 | 0 (0, 0.0e+00) |
| yes | 1 | 60 | 36852 | 0/0/0 | 0/9171/15 | 0/8817/15 | 0/32839/18785 | 0/0/0 | 0/0/0 | 58 (11, 1.1e-12) |
| yes | 5 | 12 | 7366 | 0/0/0 | 0/1662/16 | 0/1596/16 | 0/6572/20683 | 0/0/0 | 0/0/0 | 13 (4, 1.1e-12) |
| yes | 15 | 4 | 2455 | 0/0/0 | 0/438/8 | 0/419/8 | 0/2148/20683 | 0/0/0 | 0/0/0 | 3 (0, 0.0e+00) |
| yes | 20 | 12 | 2905 | 0/0/0 | 0/606/53 | 0/585/53 | 0/2345/18555 | 0/0/0 | 0/0/0 | 1 (0, 0.0e+00) |
| yes | 60 | 4 | 971 | 0/0/0 | 0/183/15 | 0/172/15 | 0/761/18555 | 0/0/0 | 0/0/0 | 1 (0, 0.0e+00) |
| yes | 240 | 1 | 243 | 0/0/0 | 0/126/3 | 0/120/6 | 0/203/9278 | 0/14/0 | 0/14/0 | 1 (0, 0.0e+00) |

### Table B2: sum, mean and variance against the exact value, on the `group_by_dynamic` windows (10000 rows, finite data)

Largest and median ulp gap to the exact value (sum: `math.fsum`; mean: the exact rational mean rounded once; var: exact rational arithmetic rounded once), the bucketed state and Polars side by side.

| b | B | sum: buckets | sum: Polars | mean: buckets | mean: Polars | var: buckets | var: Polars |
|---|---|---|---|---|---|---|---|
| 1 | 60 | 0 (0) | 16 (0) | 1 (0) | 16 (0) | 4787 (117) | 18785 (115) |
| 5 | 12 | 0 (0) | 16 (0) | 1 (0) | 16 (0) | 12470 (108.5) | 16266 (119) |
| 15 | 4 | 0 (0) | 13 (0) | 1 (0) | 13 (0) | 10342 (101) | 10341 (116) |
| 20 | 12 | 0 (0) | 62 (0) | 1 (0) | 62 (0) | 9277 (69) | 9278 (110) |
| 60 | 4 | 0 (0) | 22 (0) | 1 (0) | 22 (0) | 9277 (57) | 9278 (108.5) |
| 240 | 1 | 0 (0) | 3 (1) | 1 (0) | 5 (1) | 9277 (24.5) | 2 (0) |

The variance's two bucket forms against the exact value (b = 20, B = 12):

| level | form | max ulp | median ulp | max relative |
|---|---|---|---|---|
| 1000 | centred m2 (Chan merge) | 7081 | 62 | 1.05e-12 |
| 1000 | sum of squares | 255010944 | 42813 | 3.79e-08 |
| 1e+06 | centred m2 (Chan merge) | 6410399 | 60465 | 8.36e-10 |
| 1e+06 | sum of squares | 154895670160108 | 4.57957e+10 | 0.0263 |

### Table B3: bucketed windows against `rolling_*_by(window_size = B b, closed = "right")` on the rows with `t mod b = b - 1` (10000 rows)

On these rows the B buckets are the span exactly. Cell: validity differences / bit differences / largest ulp gap.

| b | B | rows | count | sum | mean | var | min | max |
|---|---|---|---|---|---|---|---|---|
| 1 | 60 | 10000 | 0/0/0 | 0/1737/1 | 0/1672/2 | 0/9727/14967 | 0/0/0 | 0/0/0 |
| 5 | 12 | 2021 | 0/0/0 | 0/376/1 | 0/362/2 | 0/1965/8133 | 0/0/0 | 0/0/0 |
| 15 | 4 | 664 | 0/0/0 | 0/123/1 | 0/119/2 | 0/639/1614 | 0/0/0 | 0/0/0 |
| 20 | 12 | 486 | 0/0/0 | 0/31/1 | 0/31/2 | 0/464/2059 | 0/0/0 | 0/0/0 |
| 60 | 4 | 153 | 0/0/0 | 0/9/1 | 0/9/2 | 0/146/1614 | 0/0/0 | 0/0/0 |
| 240 | 1 | 43 | 0/0/0 | 0/3/1 | 0/3/2 | 0/42/1491 | 0/0/0 | 0/0/0 |

### Table B4: the window edge inside a bucket, every row against `rolling_*_by` over `(t - B b, t]` (10000 rows, rows with 3 or more values)

Each gap is median / 95th percentile / largest. Count: relative to the span's count. Mean, min, max: in units of the span's standard deviation. sd: relative. `B` reads buckets `k-B+1..k` (inside the span); `B+1` reads `k-B..k` (covers it); `interp` adds the fraction of bucket `k-B` inside the span, as if its rows were spread evenly.

| b | B | span | read | rows whose count differs | count | mean (sd units) | sd (relative) | min (sd units) | max (sd units) | rows whose min / max differ |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 12 | 60 | B | 40.0% | 0 / 0.1 / 0.222 | 0 / 0.135 / 0.39 | 2.81e-14 / 0.106 / 0.724 | 0 / 0.0183 / 3.79 | 0 / 0.0627 / 3.04 | 5.2% / 5.7% |
| 5 | 12 | 60 | B+1 | 56.0% | 0.0476 / 0.118 / 0.267 | 0.0138 / 0.182 / 1.3 | 0.00657 / 0.158 / 2.68 | 0 / 0.219 / 11.5 | 0 / 0.251 / 11.3 | 7.9% / 8.0% |
| 5 | 12 | 60 | interp | 62.1% | 0.0105 / 0.04 / 0.0923 | 0.00803 / 0.0649 / 0.578 | - | - | - | - |
|  |  |  | bracket violations | 0 |  |  |  |  |  |  |
| 20 | 3 | 60 | B | 72.9% | 0.111 / 0.318 / 0.476 | 0.0888 / 0.455 / 0.817 | 0.0368 / 0.398 / 0.852 | 0 / 1.08 / 3.79 | 0 / 1.06 / 3.54 | 17.0% / 18.3% |
| 20 | 3 | 60 | B+1 | 73.6% | 0.111 / 0.357 / 0.733 | 0.0789 / 0.557 / 2.9 | 0.0334 / 0.502 / 5.03 | 0 / 1.04 / 13.4 | 0 / 1.13 / 11.8 | 16.4% / 17.3% |
| 20 | 3 | 60 | interp | 74.3% | 0.0167 / 0.065 / 0.147 | 0.0251 / 0.159 / 0.971 | - | - | - | - |
|  |  |  | bracket violations | 0 |  |  |  |  |  |  |
| 15 | 4 | 60 | B | 68.3% | 0.087 / 0.25 / 0.444 | 0.0631 / 0.35 / 0.737 | 0.026 / 0.313 / 0.823 | 0 / 0.813 / 3.79 | 0 / 0.792 / 3.54 | 14.7% / 15.4% |
| 15 | 4 | 60 | B+1 | 71.5% | 0.0952 / 0.278 / 0.476 | 0.0633 / 0.419 / 2.07 | 0.0258 / 0.38 / 3.48 | 0 / 0.821 / 11.5 | 0 / 0.82 / 11.3 | 13.9% / 14.5% |
| 15 | 4 | 60 | interp | 71.0% | 0.0167 / 0.0588 / 0.16 | 0.0196 / 0.12 / 1.07 | - | - | - | - |
|  |  |  | bracket violations | 0 |  |  |  |  |  |  |
| 20 | 12 | 240 | B | 33.5% | 0 / 0.0714 / 0.2 | 0 / 0.0994 / 0.354 | 2.36e-14 / 0.0711 / 0.368 | 0 / 0 / 2.47 | 0 / 0 / 2.71 | 3.7% / 4.9% |
| 20 | 12 | 240 | B+1 | 33.8% | 0 / 0.0741 / 0.333 | 0 / 0.105 / 0.639 | 2.40e-14 / 0.0731 / 0.618 | 0 / 0 / 2.57 | 0 / 0 / 3.29 | 3.8% / 4.8% |
| 20 | 12 | 240 | interp | 34.4% | 0 / 0.0125 / 0.2 | 0 / 0.0215 / 0.426 | - | - | - | - |
|  |  |  | bracket violations | 0 |  |  |  |  |  |  |
| 60 | 4 | 240 | B | 40.0% | 0 / 0.214 / 0.8 | 0 / 0.296 / 1.08 | 2.85e-14 / 0.199 / 0.648 | 0 / 0.249 / 2.57 | 0 / 0.433 / 2.92 | 7.3% / 9.9% |
| 60 | 4 | 240 | B+1 | 35.7% | 0 / 0.212 / 0.667 | 0 / 0.285 / 0.889 | 2.43e-14 / 0.195 / 1.23 | 0 / 0.245 / 3.75 | 0 / 0.25 / 6.36 | 6.7% / 8.5% |
| 60 | 4 | 240 | interp | 39.8% | 0 / 0.0271 / 0.215 | 0 / 0.0771 / 0.34 | - | - | - | - |
|  |  |  | bracket violations | 0 |  |  |  |  |  |  |
| 240 | 1 | 240 | B | 65.9% | 0.239 / 0.911 / 0.988 | 0.193 / 1.34 / 3.92 | 0.0786 / 0.789 / 1 | 0 / 2.43 / 5.78 | 0 / 2.35 / 6.14 | 31.0% / 32.2% |
| 240 | 1 | 240 | B+1 | 36.8% | 0 / 0.741 / 13.3 | 0 / 0.834 / 14.1 | 2.35e-14 / 0.557 / 14.9 | 0 / 1.01 / 15.6 | 0 / 1.52 / 38.9 | 9.8% / 14.7% |
| 240 | 1 | 240 | interp | 66.5% | 0.0173 / 0.186 / 3.61 | 0.0294 / 0.436 / 9.97 | - | - | - | - |
|  |  |  | bracket violations | 0 |  |  |  |  |  |  |

### Table B5: bucketed state bytes, saved form, per group (no row is stored)

Per bucket and column: `n` (u64), the sum and its compensation, the centred sum of squares, min, max: 48 bytes. Per column: the current bucket id (8 bytes; a bank shares one across the columns of a group). Measured: the saved size after streaming 20000 rows through one group (it does not move with the rows).

| B | keeps bucket k-B (p) | formula | bytes, one column (formula) | saved size, peak | 10 columns, one shared id (computed) |
|---|---|---|---|---|---|
| 1 | no | 8 + 48 (B + p) | 56 | 56 | 488 |
| 1 | yes | 8 + 48 (B + p) | 104 | 104 | 968 |
| 4 | no | 8 + 48 (B + p) | 200 | 200 | 1928 |
| 4 | yes | 8 + 48 (B + p) | 248 | 248 | 2408 |
| 12 | no | 8 + 48 (B + p) | 584 | 584 | 5768 |
| 12 | yes | 8 + 48 (B + p) | 632 | 632 | 6248 |
| 60 | no | 8 + 48 (B + p) | 2888 | 2888 | 28808 |
| 60 | yes | 8 + 48 (B + p) | 2936 | 2936 | 29288 |
| 240 | no | 8 + 48 (B + p) | 11528 | 11528 | 115208 |
| 240 | yes | 8 + 48 (B + p) | 11576 | 11576 | 115688 |

### Table E1: exponentially weighted features against Polars, one pass (20000 rows, 5 groups, `over`)

Nulls about 3% of x, zero-weight rows about 4% (masked to null); the second column set adds NaN, +inf and -inf in 0.4% of x. alpha is carried in the state.

| output | valid | validity diff | bit diff | max ulp | valid (non-finite) | validity diff | bit diff | max ulp |
|---|---|---|---|---|---|---|---|---|
| ewm_mean[hl=10,adjust=True,ignore_nulls=False,m=1] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 | 0 |
| ewm_mean[hl=10,adjust=False,ignore_nulls=False,m=1] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 | 0 |
| ewm_mean[hl=4,adjust=True,ignore_nulls=True,m=5] | 18587 | 0 | 0 | 0 | 18598 | 0 | 0 | 0 |
| ewm_mean[hl=100,adjust=False,ignore_nulls=True,m=1] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 | 0 |
| ewm_std[hl=10,adjust=True,bias=False,ignore_nulls=False,m=1] | 18602 | 0 | 0 | 0 | 18613 | 0 | 0 | 0 |
| ewm_var[hl=10,adjust=False,bias=True,ignore_nulls=False,m=1] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 | 0 |
| ewm_std[hl=3,adjust=False,bias=False,ignore_nulls=True,m=1] | 18602 | 0 | 0 | 0 | 18613 | 0 | 0 | 0 |
| ewm_var[hl=25,adjust=True,bias=False,ignore_nulls=True,m=4] | 18592 | 0 | 0 | 0 | 18603 | 0 | 0 | 0 |
| ewm_std[hl=50,adjust=True,bias=True,ignore_nulls=False,m=2] | 18602 | 0 | 0 | 0 | 18613 | 0 | 0 | 0 |
| ewm_mean_by[25i] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 | 0 |
| ewm_mean_by[3i] | 18607 | 0 | 0 | 0 | 18618 | 0 | 0 | 0 |

### Table E2: the carried alpha, a state saved at row 10000 of 20000 and continued (one series, hl = 10)

Largest ulp gap over the second half to the uncut stream: loaded from the bytes (alpha carried), and with alpha recomputed by a libm that differs by one ulp, in alpha itself or in `exp(-ln 2 / hl)` (the libm result).

| feature | saved bytes | alpha carried | alpha +1 ulp | alpha -1 ulp | exp +1 ulp | exp -1 ulp |
|---|---|---|---|---|---|---|
| ewm_mean adjust=True hl=10 | 32 | 0 | 0 | 0 | 1 | 0 |
| ewm_mean adjust=False hl=10 | 32 | 0 | 1 | 1 | 1 | 1 |
| ewm_var adjust=False bias hl=10 | 56 | 0 | 312 | 278 | 363 | 612 |
| ewm_std adjust=True hl=10 | 56 | 0 | 0 | 0 | 172 | 2 |
| max envelope hl=10 | 17 | 0 | 0 | 0 | 7 | 6 |

### Table E3: the decayed envelopes against Polars' rolling extremes (20000 rows, after a start-up of 5 w)

Max envelope minus `rolling_max(w)`, in units of `rolling_std(w)`: mean (the bias), 5th and 95th percentiles, share of rows above; the min envelope's mean gap to `rolling_min(w)`; the correlation of the max envelope with `rolling_max(w)`.

| series | w | half-life | max: mean gap | p5 | p95 | rows above | min: mean gap | corr(max env, rolling_max) |
|---|---|---|---|---|---|---|---|---|
| random walk | 20 | 2.5 | -1.3 | -2.78 | 0 | 0.0% | 1.28 | 0.999 |
| random walk | 20 | 5 | -1.04 | -2.34 | 0 | 0.0% | 1.02 | 0.999 |
| random walk | 20 | 10 | -0.66 | -1.65 | 0 | 1.5% | 0.655 | 1 |
| random walk | 100 | 12.5 | -1.41 | -2.98 | 0 | 0.0% | 1.4 | 0.996 |
| random walk | 100 | 25 | -1.15 | -2.52 | 0 | 0.0% | 1.14 | 0.997 |
| random walk | 100 | 50 | -0.786 | -1.81 | 0 | 0.1% | 0.773 | 0.999 |
| white noise | 20 | 2.5 | -1.22 | -2.36 | -0.00371 | 0.0% | 1.23 | 0.283 |
| white noise | 20 | 5 | -0.966 | -1.96 | -0.00371 | 0.0% | 0.974 | 0.419 |
| white noise | 20 | 10 | -0.707 | -1.51 | -0.00289 | 0.0% | 0.714 | 0.606 |
| white noise | 100 | 12.5 | -1.27 | -2.22 | -0.321 | 0.0% | 1.26 | 0.219 |
| white noise | 100 | 25 | -1.02 | -1.89 | -0.204 | 0.0% | 1.01 | 0.334 |
| white noise | 100 | 50 | -0.773 | -1.53 | -0.121 | 0.0% | 0.774 | 0.493 |

Cormode et al.'s decayed max `max(x, r y)` against `rolling_max(w)` (same units); the last column is the share of rows where it is just the current value:

| series | w | half-life | mean gap | p5 | p95 | rows equal to x |
|---|---|---|---|---|---|---|
| abs(white noise) | 20 | 5 | -1.48 | -2.74 | -0.0112 | 30.4% |
| abs(white noise) | 100 | 25 | -1.58 | -2.84 | -0.34 | 10.9% |
| random walk | 20 | 5 | -1.72 | -3.68 | 0 | 100.0% |
| random walk | 100 | 25 | -1.98 | -4.14 | 0 | 100.0% |

### Table L1: the lag state against Polars, one pass (20000 rows, 5 groups, `over`)

| output | valid | validity diff | bit diff | valid (non-finite) | validity diff | bit diff |
|---|---|---|---|---|---|---|
| shift(1) | 18602 | 0 | 0 | 18614 | 0 | 0 |
| shift(5) | 18583 | 0 | 0 | 18596 | 0 | 0 |
| shift(20) | 18510 | 0 | 0 | 18525 | 0 | 0 |
| diff(1) | 17308 | 0 | 0 | 17335 | 0 | 0 |
| diff(5) | 17288 | 0 | 0 | 17308 | 0 | 0 |
| diff(20) | 17233 | 0 | 0 | 17238 | 0 | 0 |
| pct_change(1) | 17308 | 0 | 0 | 17335 | 0 | 0 |
| pct_change(5) | 17288 | 0 | 0 | 17308 | 0 | 0 |
| pct_change(20) | 17233 | 0 | 0 | 17238 | 0 | 0 |

### Table L2: lag state bytes, saved form

| k | bytes (formula) | formula | measured | lower bound 8k | 10 columns x 1000 groups |
|---|---|---|---|---|---|
| 1 | 8 + 8k + ceil(k/8) | 17 | 17 | 8 | 0.17 MB |
| 5 | 8 + 8k + ceil(k/8) | 49 | 49 | 40 | 0.49 MB |
| 20 | 8 + 8k + ceil(k/8) | 171 | 171 | 160 | 1.71 MB |

### Table L3: the lag computed by the user with Polars before the bank, chunk by chunk (20000 rows, 5 groups)

Rows that differ from one Polars pass of `shift(k).over("g")`: each chunk lagged alone, and each chunk lagged with the last k rows of each group carried from the chunks before (the user holds them, not the bank).

| k | chunks | rows wrong, no carry | rows wrong, k rows carried | rows the user carries (k x groups) |
|---|---|---|---|---|
| 1 | 2 | 5 | 0 | 5 |
| 1 | 17 | 75 | 0 | 5 |
| 1 | 300 | 1311 | 0 | 5 |
| 5 | 2 | 25 | 0 | 25 |
| 5 | 17 | 371 | 0 | 25 |
| 5 | 300 | 5707 | 0 | 25 |
| 20 | 2 | 94 | 0 | 100 |
| 20 | 17 | 1451 | 0 | 100 |
| 20 | 300 | 14274 | 0 | 100 |

### Table L4: what Polars' streaming engine holds for a lag (`scan_parquet(...).select(g, t, y = expr).sink_parquet(engine = "streaming")`, 1000 groups, peak resident memory of the process)

| rows | copy x | shift(1) | shift(20) | shift(1).over(g) | shift(20).over(g) |
|---|---|---|---|---|---|
| 10M | 283 MB (0.4 s) | 310 MB (0.3 s) | 316 MB (0.3 s) | 679 MB (0.8 s) | 676 MB (0.6 s) |
| 20M | 452 MB (0.6 s) | 506 MB (0.6 s) | 471 MB (0.7 s) | 1221 MB (1.8 s) | 1252 MB (1.4 s) |
| 40M | 494 MB (1.2 s) | 873 MB (1.5 s) | 794 MB (1.5 s) | 2204 MB (3.3 s) | 2187 MB (3.9 s) |

### Table R1: chunked, every state saved to bytes and loaded at every cut (5000 rows, 5 groups)

21 features: 3 bucketed windows (6 outputs each, 9 with the interpolated read), 11 EW features, 3 envelopes, 4 lags. A cut saves every group's state of every feature, drops it and loads it from the bytes; the EW states are loaded into an object made with a wrong alpha (0.5), so the carried alpha must be read back. Compared with the uncut stream (all 21, to the bit) and with one Polars pass (the 11 EW features and 4 lags on every row; the bucketed count, min and max on the rows where the buckets are the span exactly).

| frame | chunking | chunks | features | equal to the uncut stream | equal to Polars to the bit | seconds |
|---|---|---|---|---|---|---|
| nulls, zero weights | 1 chunk | 1 | 21 | 21 | 18 of 18 | 0.4 |
| nulls, zero weights | 2 chunks | 2 | 21 | 21 | 18 of 18 | 0.4 |
| nulls, zero weights | 17 random chunks | 17 | 21 | 21 | 18 of 18 | 0.3 |
| nulls, zero weights | ~300 random chunks | 300 | 21 | 21 | 18 of 18 | 0.4 |
| nulls, zero weights | every row | 5000 | 21 | 21 | 18 of 18 | 1.6 |
| NaN and infinities | 1 chunk | 1 | 21 | 21 | 18 of 18 | 0.3 |
| NaN and infinities | 2 chunks | 2 | 21 | 21 | 18 of 18 | 0.3 |
| NaN and infinities | 17 random chunks | 17 | 21 | 21 | 18 of 18 | 0.3 |
| NaN and infinities | ~300 random chunks | 300 | 21 | 21 | 18 of 18 | 0.4 |
| NaN and infinities | every row | 5000 | 21 | 21 | 18 of 18 | 1.6 |

### Table C1: cost per row (best of 3; the Python state machine over 50000 rows of one group; Polars over 1000000 rows of one group, the matching expression, or for the envelopes the nearest, `rolling_max(20)`; for the bucketed rows, the five `rolling_*_by` over the same span)

| feature | Python state, ns/row | Polars, ns/row |
|---|---|---|
| bucketed b=5, B=12: 6 outputs | 9,829 | 68.3 |
| bucketed b=20, B=12: 6 outputs | 8,754 | 50.6 |
| bucketed b=5, B=60: 6 outputs | 26,848 | 70.1 |
| ewm_mean[hl=10,adjust=True,ignore_nulls=False,m=1] | 308 | 10.8 |
| ewm_std[hl=10,adjust=True,bias=False,ignore_nulls=False,m=1] | 660 | 12.5 |
| ewm_mean_by[25i] | 229 | 17.4 |
| max envelope[hl=10] | 214 | 24.1 |
| decayed max[hl=10] | 130 | 22.8 |
| shift(1) | 200 | 0.1 |
| shift(5) | 194 | 0.2 |
| bucketed read alone, B=4 | 3,831 | - |
| bucketed read alone, B=12 | 8,203 | - |
| bucketed read alone, B=60 | 10,540 | - |
| bucketed read alone, B=240 | 48,128 | - |


## State bytes as formulas

Per group and per input column; no row of the stream is stored by any feature
below except the lag, whose `k` values are the lag itself.

| feature | saved bytes | examples | grows with |
|---|---|---|---|
| bucketed window, all six outputs | `8 + 48 (B + p)`; `p = 1` keeps bucket `k - B` for the `B + 1` and interpolated reads; the 8-byte bucket id is shared by a group's columns | `B = 4`: 200; 12: 584; 60: 2888 | `B` only |
| `ewm_mean` (alpha carried) | 32 | | nothing |
| `ewm_var`, `ewm_std` (alpha carried) | 56 | | nothing |
| `ewm_mean_by` | 25 | | nothing |
| max or min envelope, decayed max (`r` carried) | 17 | | nothing |
| lag `k` (`shift`, `diff`, `pct_change`) | `8 + 8k + ceil(k/8)`, lower bound `8k` | `k = 1`: 17; 5: 49; 20: 171 | `k` |

For `c` columns and `G` groups multiply by `c G` (the bucketed id once per
group): `B = 12`, 10 columns, 1000 groups: 5.8 MB; the EW features: 0.3 to
0.6 MB. In memory nothing is added beyond the saved form: the bucketed read
combines the ring in place, and the EW and envelope updates are a handful of
f64.

A bucket costs 48 bytes against 8 for a row in a ring, so the bucketed state
is smaller than the ring it replaces only when a bucket holds more than six
rows on average; Table B5's `b = 1, B = 60` is the counter-example (about 0.3
rows a tick, 2.9 kB against about 160 bytes of rows). The choice of `b` is
the trade between Table B4's edge gap and this.

## Tolerances proposed

- **Bucketed count, min, max**: bit-exact against `group_by_dynamic` with
  `B > 1`, and against `rolling_*_by` on rows at a bucket's last tick, NaN and
  infinities included. With `B = 1`, against finite data only (Polars' `min`
  and `max` skip NaN on windows that do not overlap).
- **Bucketed sum and mean**: hold them to the exact value, not to Polars:
  sum within 1 ulp of `math.fsum` and mean within 2 ulps of the exact mean
  (measured 0 and 1). Against Polars the gap is Polars' own error, up to 62
  ulps here; a test against Polars should allow its distance from exact plus
  2 ulps, or a relative 1e-14 on this generator.
- **Bucketed variance**: against exact arithmetic, relative 3e-11 at a level
  of 1e3 with unit steps (measured 2.4e-12 on the study's frame and 9.0e-12 on
  another seed, so 1e-11 is on the edge). It grows with `level^2 / variance`,
  and on overlapping windows Polars' error is of the same size: at a level of
  1e6, up to 6.1e-9 for both across window shapes (8.4e-10 for `b = 20`,
  `B = 12`, the shape first quoted here); at 1e9, 6.3e-6 for the buckets and
  3.2e-5 for Polars. At `B = 1` Polars is within 5 ulps of exact at 1e3 and
  1e6 (2.8e-12 at 1e9) and the buckets are not (1.7e-10 at 1e6, 1.7e-6 at
  1e9).
- **The window edge inside a bucket is not a tolerance**: the bucketed form is
  exact for its own window, `[(k-B+1) b, t]`. A test should check it against
  that window (Polars `rolling_*_by` on bucket-end rows, or
  `group_by_dynamic`), and the gap to the sliding span belongs in the
  documentation as Table B4's numbers.
- **EW features**: same machine, bit-exact against Polars and across any
  chunking and resume, with NaN equal to NaN whatever its sign: the states'
  NaNs have the sign bit set (0xfff8..., x86's default NaN from `inf - inf`)
  and Polars' on the same rows do not (0x7ff8...), 47440 outputs on the
  non-finite frame, and every non-NaN output is equal by raw bits. A test
  compares NaN with NaN, not raw bits. Across operating systems, with alpha carried, a
  continuation is bit-exact. Against Polars run on another OS, one ulp of
  `exp` moves `ewm_mean` by at most 1 ulp here (5 in study 532) and
  `ewm_var`/`ewm_std` by up to 612 ulps (relative about 1e-13; study 532
  saw 1230). `ewm_mean_by` across OSes: 4 ulps (study 532), since it reads
  `pow` every row.
- **Envelopes**: no Polars counterpart; bit-exact resume; across OSes
  bit-exact with `r` carried.
- **Lags**: bit-exact everywhere.

## Caveats and what is not done

- No Rust state was built (no cargo in this study); the cost is Python's. A
  Rust bucketed read need not combine `B` buckets per row: the `B - 1` closed
  buckets change only when the clock enters a new bucket, so their combined
  sums can be cached at that moment and each row's read is that cache merged
  with the current bucket, `O(1)` a row and `O(B)` a bucket. The cache is
  derived and need not be saved. Not measured here.
- The bucket grid is aligned to 0 (`floor(t / b)`), as `group_by_dynamic`'s
  default `start_by = "window"` with no offset; other `start_by` values and
  calendar durations (`1mo`) were not run. Datetime clocks were not run;
  Polars casts them to int64 in their unit, so an int64 clock covers them.
- Table B4's numbers are for this generator (a heavy-tailed random walk at 1 to
  5 ticks a row, with gaps). The edge gap depends on how many rows a bucket
  holds and on how the series moves within a span; a stream with bursts at
  bucket starts would make the interpolated read worse than its evenly-spread
  assumption.
- The bucketed sum's 0 ulps to `fsum` is measured on these windows, not a
  bound; Neumaier's step is not a correctly rounded sum in general.
- With NaN or infinities in the frame, a group's EW states are NaN after its
  first NaN (as in Polars), so Table R1's second block exercises the EW resume
  only up to that row; the first block is the test that bites for them, and a
  deliberate fault in `EwmMean.load` (alpha not read back) fails it at 2
  chunks. Study 532's Table 2 ran on a frame with NaN only, so its EW resume
  checks are weaker than they read.
- The byte formulas are for this study's packed struct, not rheon's msgpack
  (see Table B5's note); a layout change in the library changes the constants,
  not the dependence on `B`, `k` and the rows.
- The decayed envelope is a different feature from a rolling max, not an
  approximation of it; no `O(1)` state gives the exact windowed max (the
  argument is in the Answer, item 7).
- Polars 2.0.0 only. The kernels behind Tables B1 and B3 (the overlapping
  `group_by_dynamic` aggregations, `rolling_*_by`) and the EW kernels are
  Polars'; a later Polars that changes one of them moves the sum and variance
  gaps or breaks the EW bit-equality, which a test against the pinned Polars
  catches.

## Sonnet rerun (2026-11-21)

A second worker re-ran `scripts/study_532b.py` from commit 669e0f3 and re-checked
each claim against the code. Parts: 1 the re-run, 2 the claim checks
(`studies/study_532b/recheck.py`), 3 corrections.

### Part 1: the re-run

| python | numpy | scipy | polars | machine | venv |
|---|---|---|---|---|---|
| 3.13.18 | 2.5.2 | 1.18.1 | 2.0.0 | Linux x86_64 | `uv venv` + `uv pip install numpy==2.5.2 scipy polars==2.0.0` |

Runtime of `scripts/study_532b.py`: 75 s (the header of this record says
"about two minutes"). Every deterministic table (B1 to B5, E1 to E3, L1 to L3)
is identical to the one above, line for line. Table R1's counts are identical;
only its seconds column moved (0.3 to 0.4 on four rows). Table L4 (peak
resident memory and seconds) and Table C1 (ns per row) are measurements of this
machine and moved by noise: L4 within 12% a cell, the C1 Python states by
-41% to +7% (`max envelope` 126 ns against 214, `ewm_mean` 230 against 308),
the Polars side by -23% to +28% (bucketed b=5, B=12: 87.6 ns against 68.3),
and no ordering of features changed.

### Part 2: each claim checked another way (`python -m studies.study_532b.recheck`, 44 to 48 s)

Unlike the tables, these compare raw bits (a NaN's sign and the sign of a zero
count), use two other seeds, and use a reference that shares no code with the
state. Verdicts:

| claim | verdict | measured |
|---|---|---|
| (a) bucketed count, min, max equal `group_by_dynamic` to the bit | holds for `B > 1`; at `B = 1` min and max differ on NaN windows | 3 seeds x finite and non-finite x 8 shapes: 0 validity and 0 raw-bit differences in count, min and max, and the two sides have the same windows (0 missing either way). Exception: `B = 1`, non-finite: every window that holds a NaN differs in min and max (14, 14 and 7 windows on the three seeds) and no other window does, the windows holding only infinities (21, 14, 18) included. Aligned rows against `rolling_*_by`, 2 seeds x 8 shapes x both frames: 0/0 (this includes `B = 1`). |
| (a') the `B` read is exactly the statistic of `[(k-B+1) b, t]` | holds | brute-force numpy window, every row of 6000, 4 shapes x both frames: 0 mismatches in count, min, max (NaN windows give NaN min and max); the sum is 0 ulps from `math.fsum` on every finite window |
| (a'') the same on rows of a repeated time | does not hold for every row | with the times rounded down to even, so that some repeat, aligned rows: count differs on 118, min on 25, max on 21 of 1211; on the 1082 that are the last row of their time: 0, 0, 0 |
| (b) bucketed sum about 0 ulps from `fsum`, Polars up to 62 | holds | 0 ulps on every window of 3 seeds at level 1e3, and of seed 532 at 1e6 and 1e9; Polars 15 to 62 at b = 5 to 60, B > 1 (3 to 4 at `B = 1`); the mean 1 ulp against Polars' 62. Adversarial sums (values of 1e16 with 1 and 1e-3, 300 series): 0 error |
| (c) variance | numbers hold; two statements were loose (corrected above) | max 12470 ulps, 2.4e-12 relative at 1e3 (Polars 16266); 5532 to 52747 ulps (7.4e-13 to 9.0e-12) on the other seeds; the "lower median" holds on 4 of 5 overlapping shapes (57 to 117 against 108 to 119), not at `b = 1`; at 1e6 ours up to 6.1e-9 and Polars' the same; at 1e9, 6.3e-6 against 3.2e-5; at `B = 1` Polars is within 5 ulps at levels 1e3 and 1e6 (2.8e-12 relative at 1e9) and the buckets 1.7e-10 and 1.7e-6 relative |
| (d) window-edge table | holds | the re-run reproduces every cell of Table B4 (0 differing lines); 12 buckets: mean 95th percentile 0.0994 and 0.135 sd, sd 0.0711 and 0.106, min and max differ on 3.7% to 5.7% of rows, bracket violations 0 on all six shapes; interpolated read cuts the count and mean gaps by 2.1x to 7.9x |
| (e) EW features equal Polars to the bit, alpha carried | holds except for the sign of a NaN | 3 seeds, clean frame: 0 validity and 0 raw-bit differences on 11 outputs. Non-finite frame: 0 validity differences, and 47440, 40330 and 49610 raw-bit differences, every one a NaN against a NaN: the state's has the sign bit set (0xfff8...) and Polars' not (0x7ff8...). Every non-NaN output is equal. Lags (9 outputs, both frames): 0 and 0 by raw bits |
| (f) state bytes | holds | read from `to_bytes()`, fresh and after 3000 rows: `ewm_mean` 32, `ewm_var` and `ewm_std` 56, `ewm_mean_by` 25, envelope and decayed max 17; lag `8 + 8k + ceil(k/8)` at k = 1, 5, 8, 9, 20, 64 (17, 49, 73, 82, 171, 528); bucketed `8 + 48 (B + p)` at B = 1, 4, 12, 60 and p = 0, 1 (56 to 2936). These are the study's packed struct, not msgpack (note under Table B5) |
| (g) chunk and resume invariance | holds | Table R1 identical; by raw bits too (NaN sign included): 21 of 21 equal to the uncut stream at 2, 17, 300 chunks and every row, on both frames. A fault injected into `EwmMean.load` (alpha not read back) fails 4 of 21 features on the clean frame at 2 chunks, and goes unseen on the non-finite frame at 2 chunks (21 of 21 pass), caught only from 17 chunks |
| (h) study 532's EW resume checks ran only on NaN data | holds | `studies/study_532/report.py` on `origin/vm/study-532` builds `df = frame(n, nonfinite=True)` for its Table 2 and has no clean-frame chunking. Here, after a group's first NaN or infinity every EW output of the group is NaN (4068 rows valid, 0 finite, all 11 EW outputs), and the first non-finite rows are 66, 566, 655, 785 and 879 of 5000, so a cut at row 2500 resumes NaN states for every group |

### Part 3: what the rerun changed in this record

The earlier numbers stay; these statements were corrected where they stand:

- Answer 1 and the Definitions' clock paragraph: bit equality with
  `rolling_*_by` needs a time that rises strictly, or the last row of a
  repeated time; at `B = 1` min and max differ from Polars on NaN windows
  (the note under Table B1 said so; the Answer did not).
- Answer 2: the variance medians (57 to 117 against 108 to 119 on the
  overlapping shapes, not 24 to 117) and the relative error at a level of 1e3
  (2.4e-12 here, 9.0e-12 on another seed).
- Tolerances: the variance bound (3e-11, not 1e-11) and the level-1e6 figure
  (up to 6.1e-9 across window shapes, not 8e-10, which was one shape); the EW
  tolerance says NaN equals NaN whatever its sign (the state's NaNs carry the
  sign bit, Polars' do not).
- Table B5: the two middle column headings were swapped (the formula text sat
  under "bytes" and the number under "formula"); the note says the sizes are
  fixed by the layout, are this study's packed struct and not msgpack, and
  that the last column is computed.
- The header's run time: 75 s, not two minutes.

Not changed: every table's numbers; no bucketed, EW or lag code.
