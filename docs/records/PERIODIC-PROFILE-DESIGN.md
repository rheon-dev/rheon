<!-- kind: record -->
# A periodic profile on the clock: the design

A design for PLAN 560 (building block B5), written 2026-11-23 on branch
`vm/task-560-periodic` off main `c4e7c7d`, before any code. It covers where
the block lives, how a row's phase is read off its clock (time zones and
daylight saving included), the bucketed and harmonic forms, how a phase not
yet seen is withheld, decay and a window, the state's bytes, and the
oracles. The questions it leaves for the user are at the end.

## What the block computes

An exponentially weighted estimate of a statistic of `x` (its mean, its
variance or a quantile) as a function of the row's **phase** in a period
given as a span: the hour of the day, the minute of the trading session,
the day of the week. A row reads the profile at its own phase. A seasonal
adjustment is then a Polars expression around it, additive
(`x - profile`) or multiplicative (`x / profile`, or `x / sqrt(var
profile)` for a volatility profile). PLAN 554's rule is that business
concepts are composed from the blocks in Polars, not built in.

It sits beside two other tasks. Fourier terms (489) are stateless features
of the clock that a regression weighs. STL (490) is a trailing-window
decomposition that needs an embargo. B5 is the online, decayed estimate
that neither gives.

## Where it lives: a window operator (recommended), not a spec kind

**Recommended: one window operator, `rp.ewm_profile`, in the window runner
beside `ewm_mean` and the cross operators**, computed per group by
`rp.stream.with_windows`. Its output feeds a bank as a feature or as an
adjusted target, as PLAN 557's cross operators do.

- **Chunk invariance, save and load, refusals and the clock policy come
  from the runner.** Gaps, sessions, resets, `restart_after_step_back` and
  zero-weight rows are already the runner's. A spec kind would rebuild
  them in a model.
- **The oracle is direct.** On one chunk, Polars computes the same profile
  with `group_by` over the phase, so every value can be held to it.
- **A spec kind (`rheon.spec.profile`) is the alternative.** It would hold
  the profile in the bank's state, beside the models of the same call, and
  emit `profile` and `n_eff` fields. That buys a one-call pipeline where a
  model's target is deseasonalised inside `fit_predict`, at the cost of a
  new model kind: a `ModelState` variant, frozen fixtures, the field
  registry, and the bank's sweeps (EXTENDING.md's fifteen steps).
- **Both is possible later.** The core of the block (phase, bucket and
  harmonic accumulators) would sit in `rheon-core`, behind no Polars, so a
  spec kind could reuse it. Recommendation: the operator now, and the spec
  kind only if a model must read the profile inside one bank call (Q1).

## The phase on the clock

A row's phase is `φ ∈ [0, 1)`, its place in the period, read from its own
clock value and nothing else. That makes the phase chunk-invariant and
stateless.

| clock | phase | default origin |
|---|---|---|
| numeric (float or integer) | `((t − origin) mod P) / P`, in integers where the clock and `P` are integers | 0 |
| row count | the group's rows of weight above 0, counted from its first: `(n mod P) / P` | the group's first row |
| `Date` | days since the origin, mod `P` in days | Monday 1970-01-05 for a whole number of weeks, otherwise 1970-01-01 |
| naive `Datetime` | `(t − truncate(t, P)) / P`, Polars' `dt.truncate(P)` | Polars' `truncate`: the epoch, and Monday for weeks |
| zoned `Datetime` | the same on the **local wall clock** in the column's own zone: `local = t` converted to the zone with the zone dropped, then `(local − truncate(local, P)) / P` | as above, in local time |

**A zoned clock's phase is wall-clock time, not elapsed time.** The zone
is the Datetime dtype's own: no parameter names one, and a user who wants
another zone converts the column with `dt.convert_time_zone` first. On the
day the clocks go forward, a period of `"1d"` has 23 hours of elapsed
time. Measured in Polars 2.1.0 on Europe/Amsterdam, 2026-03-29: at
05:00 CEST the wall-clock phase is 5h, `naive − naive.dt.truncate("1d")`,
and the elapsed phase is 4h, `t − t.dt.truncate("1d")` (Polars truncates
a zoned Datetime in local time, but the difference is elapsed). Rush hour
and the market open are wall-clock events, so the wall clock is the
default:

- The skipped hour (02:00 to 03:00 on that day) holds no row.
- The repeated hour on the day the clocks go back feeds its phases twice,
  as two rows in the same hour always do.
- An elapsed-time phase (`phase="elapsed"`) is the alternative for a
  period that is physical, not civil (Q3).

**The time zone rules are the ones Polars uses.** `chrono-tz` 0.10.4 is
already linked into the wheel and the CLI through `polars-arrow`
(`cargo tree -i chrono-tz`), so reading local time in `rheon-polars` links
nothing new (CLAUDE.md rule 12). The zone database is the same one the
Polars oracle reads, so the two cannot differ by a database version.

**The period is a fixed span on the wall clock:** `"15m"`, `"1h"`,
`"1d"`, `"1w"`, or a number on a numeric clock. A calendar period of
uneven length (`"1mo"`, `"1q"`, `"1y"`) has no fixed bucket width. It is
refused at first, and a later form could read the phase as the fraction
of the month elapsed (Q4). `origin=` takes a clock value, for a session
that starts at 09:30 or a week that starts on Sunday.

## Bucketed and harmonic forms

**Bucketed (`buckets=B`).** The period is cut into `B` equal phase
intervals, and each bucket keeps its own exponentially weighted
accumulator, updated only by the rows that fall in it. A row reads its own
bucket: a step function of the phase. `interpolate=True` would read
linearly between the two nearest bucket centres instead (Q5). Every
statistic has a bucketed form:

| statistic | per bucket |
|---|---|
| mean | `S_w = Σ w d x`, `S = Σ w d`, read `S_w / S` |
| variance and std | weighted Welford, `V1`, `V2`, `m`, `M2`, with `ewm_var`'s `bias` correction `V1² / (V1² − V2)` |
| quantile | `ewm_quantile`'s sketch per bucket, within 0.78% relatively |

Here `d = 0.5^(age / half_life)`, and the decay is folded in lazily at a
bucket's own update (see "Decay and a window").

**Harmonic (`harmonics=K`).** The profile is a Fourier series in the
phase, `f(φ) = a₀ + Σ_{k≤K} a_k cos 2πkφ + b_k sin 2πkφ`. It is fitted by
exponentially weighted least squares on the basis `z(φ)` of `p = 2K + 1`
terms, with a Gram `G = Σ w d z zᵀ` and `r = Σ w d z x`, solved by the
library's Cholesky (`solve.rs`) at read time.

- It shares strength across neighbouring phases. A bucket seen rarely is
  smoothed by the rest of the period, which is the point of a harmonic
  form on thin data, such as a weekly profile of an hourly series.
- The mean and the variance have harmonic forms. The variance is the
  harmonic fit of `x²` less the square of the fit of `x`, clamped at 0,
  or, for a multiplicative volatility profile, the fit of `ln x²` (Q6).
- **A quantile has no harmonic form here.** The quantile regression of a
  Fourier basis is an iterative fit, not a one-pass sum. A harmonic
  quantile is refused.

## A phase not yet seen is withheld by its own weight

One rule covers both forms: **the profile at phase `φ` is withheld (null,
with a reason) while the weight behind it is below `min_weight`**, where
the weight behind it is

```text
n_eff(φ) = 1 / (z(φ)ᵀ G⁻¹ z(φ))
```

the reciprocal of the fit's variance factor at `φ`, in units of row
weight.

- **For buckets, `z` is the indicator of the row's bucket and `G` is
  diagonal**, so `n_eff(φ)` is exactly the bucket's decayed weight sum
  `S_b`, with no formula to trust.
- **For harmonics**, it is the effective weight the fit has at that phase.
  A phase the data never visited has a large variance factor and a small
  `n_eff`, and a singular `G` gives `n_eff = 0`.
- It is a weight, never a count of rows. It follows CLAUDE.md rule 8:
  before this row's update and before its own decay, and inside the window
  where a `window_size` is given. So `min_weight` means the same here as in
  every model.
- The default `min_weight` is the smallest above 0: a bucket with any
  weight is read, and an unseen one is null.
- A variance also needs `V1² − V2 > 0`, as `ewm_var` does: a bucket that
  one row's weight fills is null.
- A row of weight 0, or with no value, reads its phase's profile and adds
  nothing, and counts in no clock (rule 9).

## Decay and a window

- **`half_life` decays by the clock's age, not by the bucket's own
  rows.** A bucket last seen a week ago weighs a week's decay less. That is
  what "exponentially weighted" means on every other operator, and it lets
  a bucket go stale and be withheld after a long absence.
  - The alternative, decaying a bucket once per period, is the classic
    seasonal smoothing (Holt-Winters' γ). It equals this on a regular
    grid and is wrong on an uneven one (Q7).
  - In rows, a `half_life` counts the group's rows of weight.
- **Decay is lazy.** Each bucket stores its last update time and decays by
  `0.5^((t − t_b)/h)` when it is next read or updated. A harmonic Gram
  decays as a whole at each row, since every row touches every term.
- **A window (`window_size`, on the clock or in rows) is a hard cutoff:**
  rows older than it leave.
  - For buckets, each bucket keeps its window's rows in the two stacks the
    windowed operators use, so a row leaves without a subtraction. The
    state grows with the rows in the window and is held to
    `max_state_bytes`.
  - For harmonics, a leaving row would be a rank-1 downdate of `G`, whose
    roundings build up over a long stream. A harmonic window is refused at
    first, or kept as two stacks of Grams at `p²` doubles each (Q8).
- `closed` takes the window operators' meaning. The useful default for an
  adjustment is that a row reads the profile **before its own update**, so
  a row's own value is not in its own deseasonalising term. That is
  `closed="left"` on this operator (Q9).

## State: bytes per group

`B` buckets, `K` harmonics, `p = 2K + 1`, and `R` rows in a window.

| form | statistic | bytes per group |
|---|---|---|
| buckets, decay | mean | `24 B` (`S_w`, `S`, `t_b`): 576 at 24 hourly buckets, 4,032 at 168 hourly-weekly |
| buckets, decay | variance or std | `48 B` (`V1`, `V2`, `m`, `M2`, `t_b`, spare) |
| buckets, decay | quantile | one sketch a bucket: about 16 KB at a half-life of 500 rows (as measured for `ewm_quantile`), so 24 buckets is about 384 KB, past the default cap |
| buckets, window | any | the decayed form plus 24 bytes a row in the window (three numbers a row, as the windowed operators keep) |
| harmonics, decay | mean | `8 (p(p+1)/2 + p + 2)`: 304 at `K = 3` |
| harmonics, decay | variance | the mean's plus `8 p` for the second moment: 360 at `K = 3` |

Every form is bounded by the profile's shape (or the window), never by
the stream's length. Each is checked against the operator's
`max_state_bytes` (default 256 KB, ceiling 64 MiB), with a warning at half
and a refusal at the cap that names the group and the bytes. The
per-bucket quantile is the one form whose default could pass the default
cap. The options are a coarser sketch for a profile (Q10), or a cap the
user raises.

## Oracles

- **The phase.**
  - Polars on one chunk: `dt.convert_time_zone`, `dt.replace_time_zone(None)`
    and `dt.truncate(P)`, as measured above, across both daylight-saving
    changes of a year in two zones (Europe/Amsterdam, America/New_York)
    and on a zone with a half-hour offset (Asia/Kolkata).
  - A second opinion from the standard library's `zoneinfo`, row by row,
    which uses the system's database rather than chrono-tz's, so it checks
    the rules and not a shared copy.
- **The bucketed profile.**
  - Polars on one chunk: the bucket as `(phase * B).floor()`, and per row
    the decayed sums over the earlier rows of its bucket. Within the
    group and bucket, that is `(w · x · 2^(t/h)).cum_sum().over(g,
    bucket)` times `2^(−t/h)` on frames short enough that `2^(t/h)` stays
    finite.
  - The end-of-stream profile as `group_by(g, bucket).agg(...)` with the
    weights written out.
  - The variance against the weighted variance written out, as for
    `ewm_var`.
  - The quantile against NumPy's weighted `inverted_cdf` per bucket,
    within the sketch's 0.78%.
- **The harmonic profile.**
  - statsmodels: `DeterministicProcess` with `CalendarFourier` (or
    `Fourier`) for the basis, and `WLS` with the decay weights for the
    coefficients and the fitted profile at each row's phase.
  - `n_eff(φ)` against `1 / (zᵀ (ZᵀWZ)⁻¹ z)` from the same fit.
  - Rust tests use `crate::oracle`'s LU for `G⁻¹`.
- **What a window run owes:** chunk invariance (1 to 10,000 rows a chunk),
  a save and load at every row, zero weights in no clock, the clock events,
  a state bound, and the cap's warning and refusal.
- **Realistic and degenerate cases (PLAN 550).**
  - Realistic: an intraday volatility U-shape with weekends missing, and
    a transit series with rush hours across a daylight-saving change.
  - Degenerate: one bucket only ever seen, a constant series (variance 0),
    a period shorter than the clock's step, and a phase grid that aliases
    the period.

A scan of PyPI and CRAN found no streaming, exponentially weighted
seasonal profile to hold the operator to as a whole. statsmodels'
`seasonal_decompose`, STL and MSTL are batch decompositions (490's
oracles). pandas' `groupby` over the phase is the Polars oracle again. R's
`forecast::fourier` builds the same basis as statsmodels. So the operator
is held to its definition, computed by Polars and statsmodels, as PLAN
557's operators are.

## Questions for the user

1. **Where:** a window operator `rp.ewm_profile` only (recommended), a
   spec kind `rheon.spec.profile` only, or both, the spec kind for a model
   that must deseasonalise inside one bank call?
2. **One operator or a parameter:** `rp.ewm_profile(x, period=,
   half_life=, buckets= | harmonics=, stat="mean"|"var"|"std"|"quantile",
   ...)` as a new operator (recommended), or `period=` and `buckets=`
   added to `ewm_mean`, `ewm_var`, `ewm_std` and `ewm_quantile`? The
   latter reads well, but `ewm_mean` weighs a value by the time it held,
   which a profile cannot use: a bucket's rows are a period apart, and the
   time between them belongs to other buckets.
3. **The phase of a zoned clock:** the wall clock in the column's own
   zone (recommended), with an elapsed-time option, or elapsed time only?
4. **Calendar periods** (`"1mo"`, `"1q"`, `"1y"`): refused at first
   (recommended), or read as the fraction of the period elapsed?
5. **Reading between buckets:** the row's bucket alone (recommended
   default), with `interpolate=True` reading linearly between bucket
   centres?
6. **A harmonic variance:** the fit of `x²` less the squared fit of `x`,
   clamped at 0 (recommended), or the fit of `ln x²` (always positive, a
   multiplicative model, biased by `E ln χ²`)?
7. **Decay:** by the clock's age, so a bucket goes stale while unseen
   (recommended), or once per period, as Holt-Winters' seasonal smoothing
   does?
8. **A window on the harmonic form:** refused at first (recommended), or
   two stacks of Grams at `8 p²` bytes a row?
9. **The default `closed`:** `"left"`, so a row's profile excludes its own
   value (recommended for an adjustment), or the window operators'
   `"right"`, which includes it?
10. **The quantile per bucket:** `ewm_quantile`'s sketch, about 16 KB a
    bucket (past the default cap at 24 buckets), or a coarser sketch for
    profiles, such as 16 buckets an octave (within about 3%, about 4 KB)?
11. **Pooled profiles:** a profile per group only (recommended), or also
    one pooled across groups (every symbol's intraday volatility in one
    profile)? That would be a cross-group operator, with PLAN 557's
    partition note.
12. **`min_weight`'s default:** any weight above 0 (recommended, so an
    unseen bucket is null and a seen one is read), or a default in units
    of the half-life's weight, such as half of one row's?
