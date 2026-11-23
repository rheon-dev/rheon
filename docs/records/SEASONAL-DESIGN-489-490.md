<!-- kind: record -->
# Fourier terms, and STL and MSTL behind an embargo: the design

A design for PLAN 489 and 490, written 2026-11-23 on branch
`vm/task-489-490-seasonal` off main `555b45e`, before any code. It is
checked against the study `docs/records/STUDY-490-489.md` and its script
`scripts/study_490_489.py` (branch `vm/study-490`; "the study", "§n" below
are its sections). It sits beside the periodic profile `rp.ewm_profile`
(PLAN 560, branch `vm/task-560-periodic`, `PERIODIC-PROFILE-DESIGN.md`).
The questions for the coordinator are at the end.

## What is built, and where

| name | what it computes | state | home |
|---|---|---|---|
| `rp.fourier(clock, period=, order=, origin=)` | `sin` and `cos` of `2π i φ`, `i = 1..order`, `φ` the row's phase in a fixed period | none | Polars expressions (Q1) |
| `rp.calendar_fourier(clock, freq=, order=)` | the same with `φ` the fraction of the calendar hour, day, week, month, quarter or year elapsed | none | Polars expressions (Q1) |
| `rp.stl(x, period=, component=, ...)` | the trend, seasonal or remainder of the row `lag` learned rows back, from STL on the last `W` learned rows | a ring of `W` values per group | a window operator in `rp.stream.with_windows` |
| `rp.mstl(x, periods=, component=, ...)` | the same for MSTL with several seasons | the same ring | the same |

The decomposition (the loess smoothers, the inner and outer loops, MSTL's
composition, the kernel) lives in `rheon-core` (`stl.rs`, `stl/loess.rs`,
`stl/kernel.rs`), with no Polars; the ring, the clock and the save and load
live in `rheon-polars/src/windows/` beside 560's `profile.rs`. These are
window operators, not spec kinds: they transform one column per group and
feed a bank as features, as `ewm_profile` does, and the window runner
already owns chunk invariance, groups, sessions, resets, save and load and
`max_state_bytes`. The common paragraph of 486-493 ("a model in
`rheon-core` behind `OnlineModel`, the bank, the CLI config") does not fit
an operator: there is no `ModelState` variant, no field registry entry, and
the CLI runs no window operators today (Q1).

## Fourier terms (489)

**The form: reduced in integers, phase from a fixed origin.** For a clock
`t` in integer units (nanoseconds for `Datetime`, days for `Date`, the
integer itself for an integer clock), a period `P` in the same units and an
origin `t0` (default the epoch; Monday 1970-01-05 when `P` is a whole number
of weeks, as `ewm_profile` and Polars' `truncate` have it):

```text
b   = (t - t0) mod P                  floor mod, in [0, P)
r_i = (i b) mod P                     exact in i64 while P * order < 2^63
u_i = f64(r_i) / f64(P)               one conversion each, then one division
x_i = (2π) u_i                        the double 6.283185307179586 times u_i
sin(x_i), cos(x_i)
```

- The angle never exceeds `2π`, so its error stays under 1e-15 rad at any
  stream length (§3.5); a form that holds epoch time in a double is off by
  1e-6 rad at a one-second period and 1e-3 at a millisecond from the first
  row, and is not built.
- No origin is kept per group: no state, and chunk-invariant by construction.
- **Why expressions:** every step above is a Polars expression (Polars 2.1.0's
  integer `%` is the floor mod, checked: `[-7, 7, -1] % 3 = [2, 1, 2]`;
  `sin` and `cos` are Rust's, the platform libm, as a port's would be). So the
  terms stream in any `LazyFrame`, inside or outside `with_windows`, with no
  plugin. The terms on a row clock (a period in rows) would need the
  group's learned-row count, which is state; they are refused, and a
  regular grid's period is given as its duration instead.
- **The clock it reads:** a naive `Datetime` as is; a zoned `Datetime` on its
  **wall clock** in its own zone (`dt.replace_time_zone(None)` after the
  zone), as `ewm_profile` reads a phase, so a term and a profile of one
  period agree on the phase (Q4). A `Date` in days. An integer clock in its
  units. **A float clock is refused**: no exact reduction exists, and the
  clock's own spacing sets the phase error (epoch seconds in a double: 1.7e-6
  rad at a one-second period, §3.5); cast it to integer units first.
- `period` a fixed span (`"1d"`, `"1w"`, `"90m"`) or an integer; `"1mo"`,
  `"1q"`, `"1y"` are refused here and named in the message as
  `calendar_fourier`'s. `P * order < 2^63`, so `i b` cannot overflow
  (a year in nanoseconds allows order 291).
- Names: `{clock}_sin{i}_{period}`, `{clock}_cos{i}_{period}`, in
  statsmodels' column order (`sin(1)`, `cos(1)`, `sin(2)`, ...).
- The unreduced form (`u = f64(t - t0) / f64(P)`, then `sin(i x)`) equals
  statsmodels' `Fourier.in_sample` to the bit while `t - t0` is exact in a
  double, and drifts as `eps 2π cycles` after; it is not offered (Q2).

**CalendarFourier.** The ratio is statsmodels' `_compute_ratio`, converted
then divided (§3.3, a year is past 2^53 ns and an exact quotient would
differ by 1 ulp in a fifth of the rows):

```text
start = t.dt.truncate(freq);  end = start.dt.offset_by(freq)
ratio = f64(t - start) / f64(end - start), in the column's unit;  x = (2π) ratio
sin(i x), cos(i x)           i x after x is rounded, as statsmodels does
```

`freq` in `"1h"`, `"1d"`, `"1w"` (Monday to Sunday, pandas' `W-SUN`),
`"1mo"`, `"1q"`, `"1y"`; a multiple such as `"2d"` is refused (a fixed span
belongs to `fourier`). Here `i x` is taken after rounding `x`, not reduced
per harmonic: the ratio is below 1, so `i x < 2π i` and the drift of §3.5
does not arise; this keeps CalendarFourier bit-equal to statsmodels.

## STL and MSTL (490)

**The port.** STL is written from the authors' Fortran as statsmodels'
`_stl.pyx` has it, every choice of the study's §4 table: the span `q > n`
rule `h = lambda_n(x) + floor((q - n)/2)` (the Fortran's, not the
paper's scaled radius, which is a different quantity, 0.09 to 0.21 absolute
in every case of §5); the tricube's 1 below `0.001 h` and 0 above `0.999
h`; the degree-1 fall-back to the weighted mean; a point whose weights all
vanish takes its own value; the bisquare robustness weights with the
median of an even count as the mean of the two middle values; the window
`[1, q]` up to `(q + 2) // 2` then sliding; sums in row order; the moving
average as a running sum; cubes through `powf(3.0)` (the platform's `pow`),
squares as `x * x`. MSTL is statsmodels' composition (§7): seasons sorted
shortest first, `iterate` passes, each STL with its own period and seasonal
span, the trend the last STL's, `R = D - T`.

Parameters, all statsmodels': `period` (`periods` for MSTL), `seasonal`
(`windows`; default 7, and `7 + 4 i` for MSTL), `trend` and `low_pass`
(statsmodels' defaults from the period and span), `seasonal_deg`,
`trend_deg`, `low_pass_deg` in {0, 1}, the three jumps, `robust`,
`inner_iter`, `outer_iter` (the counts; `robust` sets their defaults, as
in statsmodels, §6.3), `iterate`. Not built: `lmbda` (Box-Cox is a pointwise
transform composed in Polars before the operator; `"auto"` is a
per-window likelihood fit, Q10), and OnlineSTL (Mishra et al. 2022, an
approximation with no oracle, not run by the study).

**The online form.** Per group, a ring of the last `W` learned rows. A row
is learned when its input is not null; a null row is not pushed and counts
in no lag (rule 9's meaning: the operators take no `weight=`, and STL has
no observation weights, so a weight would have no oracle). On each row the
operator emits the chosen component of the row `lag` learned rows back:

- `closed="left"` (recommended default, Q5): the ring as it stood before
  this row, so the emitted value reads no row at or after the current one;
  `lag >= 1`, the emitted position `W - lag` of the ring (0-based).
- `closed="right"`: the ring after pushing this row (the study's form);
  `lag >= 0`, position `W - 1 - lag`. A row's own value is then in its
  output, which leaks a target used as its own feature.
- The output name says the lag: `{x}_stl_{component}_lag{lag}`; MSTL's
  seasonal names its period, `{x}_mstl_seasonal{p}_lag{lag}`.
- Null until the ring holds `W` learned rows (the oracle's window is then
  always exactly `W` rows). A null input row emits what the row before it
  emitted, since the ring did not move.
- A session change, a reset or a refused step back empties the ring, as
  the runner's other operators restart. With `step=` given, a learned row
  whose clock is not the last learned row's plus `step` also empties it
  (Q8): STL is positional, and a gap would shift every phase after it.

**The embargo (the study's §6.1 and §7, not half the trend span).** The
default lag is the seasonal smoother's reach, `h = max_i p_i (s_i - 1) /
2`, plus 1 under `closed="left"`; the default window is `W = 2h + 1`, so
the emitted row is the window's centre, where the gap to the whole
history's decomposition is smallest. `(trend - 1)/2` is the worst lag for
the seasonal: one row before its largest revision (0.23 RMS, 14% of its
sd, at `h = 23`, §6.1). Defaults this gives:

| series | periods, spans | `h` (the lag under `"right"`; `h + 1` under `"left"`) | `W` |
|---|---|---|---|
| hourly, daily season | 24, 7 | 72 | 145 |
| hourly, weekly season | 168, 7 | 504 | 1,009 |
| minutes, daily season | 1440, 7 | 4,320 | 8,641 |
| hourly, MSTL daily and weekly | 24/168, 11/15 | 1,176 | 2,353 |

A smaller `lag` is allowed (the trend alone settles at `(trend - 1)/2`; lag
0 under `closed="right"` is the window's one-sided end), and the docs give
the study's measured gaps for each so the user sees what a short lag costs.
Refused: `lag` outside the window; `W < 2 p` for STL (statsmodels' two
cycles); for MSTL any period `>= W / 2` (statsmodels drops it with a
warning; a stream cannot, so it refuses); every statsmodels validation of
spans, degrees and jumps (odd spans, `trend > p`, `low_pass > p`).

**Two methods (Q6).** Without the robust loop every step is linear in the
ring, so the emitted component is one row of a `W x W` operator dotted with
the ring.

- `method="kernel"` (recommended default): the kernel row of each
  component at the emitted position, computed once per operator from the
  settings and `W` (never per group, never state), then `(m + 1)` dot
  products of `W` per emission (`m` seasons; the remainder is `x - S - T`).
  It is built by the **adjoint** of the port (the emitted position's unit
  vector run back through the transposed smoothers), about one fit's cost;
  the study's port on the identity is `W` fits, about half an hour at
  MSTL's `W = 2,353` (182 ms a statsmodels fit at `W = 504`, scaled). Its
  cubes are `x * x * x`: no libm, so the same bits on every OS, and the
  cube moves results by 1e-13 at most (§5), inside the 1e-12 tolerance.
- `method="exact"`: the port per emission, bit-equal to statsmodels on the
  same OS (both call its `pow`); required for `robust` or `outer_iter > 0`.
  One full fit per emission: 0.4 to 9 ms in statsmodels' compiled code at
  `W <= 260` (§5), about a second at MSTL's hourly defaults (extrapolated).
  Offered with the cost measured and stated, and no cadence (Q7).

## State, bytes and bounds

`k` the distinct `(input, W)` pairs of a run's operators (components and
lags of one input and `W` share a ring), `G` groups, `m` seasons, `L` lags
emitted per operator.

| item | bytes | per | counted in |
|---|---|---|---|
| Fourier, CalendarFourier | 0 | | |
| the ring: `W` values, head, count, last clock | `8 W + 24` | group and `(input, W)` | state, `max_state_bytes` |
| kernel rows | `8 W (m + 1) L` (STL: `16 W L`) | operator, shared by groups | memory, checked against the same cap at build |
| exact-mode scratch | `8 (7 (W + 2 p) + 3 W)` for STL, about `m + 1` times for MSTL | one fit, freed after | neither (not state) |

So the state is `G k (8 W + 24)` bytes, bounded by the window, never the
stream. At the defaults above: 1,184 B (hourly, daily), 8,096 B (hourly,
weekly), 69,152 B (minutes, daily), 18,848 B (MSTL hourly); MSTL's kernels
at `W = 2,353` are 56,472 B per lag, once. A group's rings are checked
against the operator's `max_state_bytes` (256 KB default, 64 MiB ceiling,
as the other operators) **when the run is built**, since `W` is known: the
refusal names the operator, `W` and the bytes. MSTL of daily and weekly
seasons on minute rows asks `W = 141,121`, 1,128,992 B, and is refused
under the default cap. The saved form is msgpack, about `9 W` bytes a ring.

**A layout change under rule 5** (2.0.0 is past 1.0): the window state
gains rings, so `WINDOWS_VERSION` moves (on top of 560's), and with it the
bank's schema (a bank file embeds a window core per formula target); the
current sets are copied to `v<N>/`, a loader reads an older state as one
with no rings, and the fixtures freeze each form a ring takes: empty,
filling, full and wrapped, and emptied by a break. A formula target does
not take `stl` or `mstl` at first.

## Oracles

| quantity | oracle | tolerance |
|---|---|---|
| Fourier, a whole number `n = P / s` of steps `s` on an even clock | statsmodels `Fourier(n, 1).in_sample(RangeIndex(n))` read at row `r_i / s`: the same `u_i` (both correctly rounded quotients of one rational), the same `(2π) u`, the same `sin` | **0 ulps** (the ratio on every OS; the terms where numpy's `sin` is the platform libm, measured on Linux only) |
| Fourier, any period, against statsmodels' unreduced columns | `Fourier(P, order).in_sample` | `4 eps 2π i n / P + 2 ulp of 1` (the study's bound; measured at 0.32 of it) |
| Fourier, against the exact value | mpmath at 50 digits (a dev dependency, declared by name) | 4 ulps of 1 |
| CalendarFourier, `h D W ME QE YE`, ns and us indexes | statsmodels `CalendarFourier(freq, order).in_sample` | **0 ulps** |
| a zoned clock's terms | statsmodels on the naive wall-clock index | 0 ulps |
| STL, `method="exact"`, any settings, robust or not | statsmodels `STL(window, period, ...).fit()` at the emitted position | **0 ulps**, every component, each OS against its own statsmodels |
| MSTL, `method="exact"` | statsmodels `MSTL(window, periods, windows, iterate, stl_kwargs)` | **0 ulps** |
| STL or MSTL, `method="kernel"` | the same | `1e-12` relative to `max \|window\|` (measured 70 times inside) |
| MSTL on a Box-Cox column made in Polars | `MSTL(lmbda=l)` | `16 eps max \|transformed\|` (a libm result) |
| the kernel's adjoint | the port run on the identity, in Rust, at small `W` (faer is no oracle here: no matrix is solved) | measured, then fixed in the ledger |

The window's value against the whole history's is a documented gap (§6.1,
§7), not a tolerance.

## Test plan

- **Python oracles** (`tests/test_fourier.py`, `tests/test_stl.py`, and the
  statsmodels cases in `tests/test_second_opinion.py`): the study's cases,
  rerun through `with_windows` on each OS: §3.1's even clocks, §3.2's
  irregular ones, §3.3's 200,289 calendar instants; §5's six STL settings,
  §6.3's robust and iteration settings, §7's MSTL cases with Box-Cox
  composed in Polars; one MSTL case at the hourly defaults (`h = 1,176`,
  `W = 2,353`, kernel, a few hundred emissions; extended tier). The kernel
  and Box-Cox bounds join the oracle ledger as grids.
- **The future is not read:** every row after `t` rewritten to `1e6` leaves
  row `t`'s output unchanged; under `closed="left"` row `t` itself too.
- **What a window run owes:** chunks of 1 to 1,000 rows give one chunk's
  bits; a save and load at every row (filling, full, wrapped, after a
  break); groups; a session change, a reset and a step back empty the ring;
  `step=`'s break; a null row emits the row before's values and counts in
  no lag; a streaming `LazyFrame` plan.
- **Refusals**, each by name: the spans, degrees and jumps; `W < 2 p`; an
  MSTL period `>= W / 2`; `lag` outside the window; the cap at build; a
  float clock, a row clock and a calendar unit for `fourier`; a multiple
  for `calendar_fourier`; `robust` under `method="kernel"`.
- **Rust (`rheon-core`, seen by `cargo mutants`):** the loess window and
  the `q > n` radius against the Fortran's formula; the cut-offs at `0.001
  h` and `0.999 h`; the degree-1 fall-back; the all-zero-weight fall-back;
  the even median; the cube is `powf`, checked on values where `x * x * x`
  differs, so a refactor to `powi` fails; the adjoint kernel against the
  forward operator on the identity; linearity; a constant series' seasonal
  is 0 within the rounding.
- **Realistic and degenerate:** hourly load with daily and weekly seasons;
  a constant series; period 2; `W = 2 p`; a robust point whose weights vanish.
- **State fixtures** as above; **budgets** (`rheon-bench`): a kernel
  emission at `W = 145` and at MSTL's `W = 2,353`, an exact STL emission at
  `W = 145`, and the kernel's construction at MSTL's defaults, each with
  its memory.

## When to use which (the docs)

A section of `docs/WINDOWS.md`, after `ewm_profile`'s, and rows in
`docs/FEATURES.md`; the docstrings state the update equations; nothing
reader-facing cites a plan task or a record. The section's table:

| you want | use | because |
|---|---|---|
| seasonality as inputs to a regression, learned with its other features | `fourier`, `calendar_fourier` | stateless, no warm-up, no delay; the model weighs the terms and decays them with everything else; a smooth shape in a few terms; month, quarter and year periods |
| to adjust a series by its typical value, spread or quantile at this hour or weekday, now | `ewm_profile` | an online estimate per phase, read on the row itself, with no delay; any shape (buckets) or a smooth one (harmonics); a variance or quantile profile, which a regression on Fourier terms does not give |
| the trend, the seasonal and the remainder separately, as statsmodels' STL or MSTL computes them | `stl`, `mstl` | the standard decomposition, exact against statsmodels, but each value arrives `lag` rows late (three seasons of the longest period by default: 72 hours for a daily season, seven weeks for a weekly one), needs a regular step, and holds `W` values per group |

The section adds: a decomposition's remainder is for review after the lag,
not a feature for the next row; the gap at shorter lags is tabled.

## What the study left open, or got wrong

1. **Its online form reads the row it emits on** (§5: the output on row
   `t` reads rows up to `t`). As a feature of the series it decomposes, that
   reads the row's own target, which hard rule 2 forbids for a prediction;
   hence `closed="left"` by default here (Q5).
2. **Its kernel construction does not scale**: the port on the identity is
   `W` fits; at MSTL's defaults, about half an hour per operator. The adjoint
   is the design's answer, not yet measured.
3. **The exact method's cost at the defaults it recommends was not
   measured**: MSTL at `W = 2,353` was not timed; extrapolated, about a
   second a row. The robust cadence was not built (§6.3).
4. **Cross-OS was not measured.** The 0-ulp claims hold on one OS because
   both sides call that OS's libm, but only if numpy's `sin` is the libm's
   (it was, on Linux x86-64 with AVX-512); macOS and Windows are open.
5. It used statsmodels' private `_get_terms`; the 0-ulp oracle here uses
   the public `in_sample`, looked up at the reduced position.
6. "No oracle for local time" is too strong: statsmodels on the naive
   wall-clock index is the oracle for the wall-clock reading adopted here.
7. Its ring returned nothing on a skipped row; here a null row is still
   answered (rule 9) with the ring's current values.

## Questions for the coordinator

1. **Homes:** `fourier` and `calendar_fourier` as Polars expressions, and
   `stl` and `mstl` as window operators, with no spec kind and no CLI
   (recommended); or spec kinds in the bank, as the common paragraph of
   486-493 reads, at the cost of a `ModelState` variant each and a ring
   per target in the bank's state?
2. **One Fourier form:** reduced in integers only (recommended: accurate at
   any length, held to statsmodels at 0 ulps by the lookup and within the
   study's bound otherwise), or also the unreduced form for users who want
   statsmodels' columns at any row?
3. **Build order:** part 2 waits for 560 to merge and is rebased on main, to
   share its clock reading and its `WINDOWS_VERSION` (recommended)?
4. **A zoned clock's phase:** the wall clock, as `ewm_profile` (recommended),
   or UTC?
5. **`closed` for `stl` and `mstl`:** `"left"` by default (recommended: the
   delayed value reads no row at or after its own row's), against the window
   operators' `"right"`, which the coordinator ruled for `ewm_profile`?
6. **The default method:** `kernel` (recommended: microseconds a row, the
   same bits on every OS, within 1e-12 of statsmodels), with `exact`
   (bit-equal, a fit a row) opt-in and required for robust?
7. **Robust:** `method="exact"` per emission with its measured cost and no
   cadence (recommended), or a cadence now, which the study did not test?
8. **Gaps on the clock:** an optional `step=` whose break empties the ring
   (recommended), or a required regular clock refused otherwise?
9. **Provisional values:** only the `lag` parameter, lag 0 or 1 being the
   window's one-sided end, its name saying so (recommended), or a separate
   provisional output?
10. **Box-Cox:** composed in Polars before the operator, no `lmbda`
    (recommended), or `lmbda` as a number?
11. **The kernel's cube as `x * x * x`** (recommended: no libm in the kernel
    path, so the default output is the same on every OS), or `powf` as in
    the exact port?
12. **The docs row in `docs/README.md`'s Records table** for this note is
    owed (`tests/test_doc_structure.py`); it was left out of part 1, which
    adds this file only. Add it in part 2 (recommended)?
