<!-- kind: record -->
# Study 508: ADWIN, Page-Hinkley, DDM, EDDM and KSWIN against river

A numpy prototype for PLAN task 508 ("CUSUM, Page-Hinkley, ADWIN and the other
change detectors ... Oracle: `river`'s drift detectors on the same stream"),
following task 494's scan (`vm-tools/research/494-MODELS-SCAN.md` §14 on
`vm-tools/oracle-gap-scan`: every detector becomes an option of `drift`, and
ADWIN's bucket list is the only new structure). Python only, no Rust: the
options in `rheon-core`, `docs/EXTENDING.md`'s places and the measured Rust
cost are the task proper. Nothing here is collected by the repo's pytest
(`testpaths = ["tests"]`) and nothing is added to `pyproject.toml`: river is
not in the dev group, so the study runs in a venv of its own.

## Files and how to run them

- `scripts/study_508.py`: runs every check and prints the tables below (about
  15 minutes at 200 seeds; `--seeds 20` for about 3, `--no-cost` skips the
  timings).
- `studies/study_508/detectors.py`: the five detectors (`Adwin`,
  `PageHinkley`, `DDM`, `EDDM`, `KSWIN`), one value a row, with `save`,
  `load` and `state_arrays`. The update equations are in its module docstring
  and below.
- `studies/study_508/data.py`: seeded synthetic streams (no data files).
- `studies/study_508/compare.py`: each check.

Environment: a standalone venv, not `uv sync` (which would build the Rust
extension): Python 3.13.18, numpy 2.5.5, scipy 1.18.1, **river 0.27.0**, on
Linux x86-64 (4 cores). River 0.27.0's ADWIN is compiled Rust
(`river._river_rust.drift.AdaptiveWindowing`; the scan read 0.26.1's
`adwin.py`); its source was read from the sdist (`rust_src/adwin.rs`). The
other four are Python (`river/drift/page_hinkley.py`, `kswin.py`,
`binary/ddm.py`, `binary/eddm.py`). River's argument names in 0.27.0:
`ADWIN(delta=0.002, clock=32, max_buckets=5, min_window_length=5,
grace_period=10)`, `PageHinkley(min_instances=30, delta=0.005,
threshold=50.0, alpha=1 - 0.0001, mode="both")`, `binary.DDM(warm_start=30,
warning_threshold=2.0, drift_threshold=3.0)`, `binary.EDDM(warm_start=30,
alpha=0.95, beta=0.9)`, `KSWIN(alpha=0.005, window_size=100, stat_size=30,
seed=None, window=None)`.

## The detectors, in short

Each reads one value a row and answers, on that row, whether it flags a
drift (DDM and EDDM also a warning). The flag for row `t` uses row `t` itself:
it says the stream has changed by now, not what row `t` will be, so rule 2
has nothing to hold here (a detector fed a model's out-of-sample residual
inherits that residual's rule 2). Every detector in river, and here, **starts
from an empty state on the row after a flag** (river's `update` opens with
`if self.drift_detected: self._reset()`).

**ADWIN** (Bifet and Gavalda 2007, ADWIN2). The window is an exponential
histogram: level `i` holds at most `M` (`max_buckets`) buckets of `2^i` rows,
each the pair (sum of its values, sum of squared deviations about its own
mean). A row enters level 0 as `(x, 0)`; when a level holds `M + 1` buckets its
two oldest merge into one bucket of the next level,
`(t0 + t1, v0 + v1 + n (t0/n - t1/n)^2 / 2)`, `n = 2^i`, and the merge
cascades upward. The window's totals are kept as the rows come:

```text
width += 1;   V += (width - 1) (x - S/(width - 1))^2 / width;   S += x
```

Every `clock` rows (counted since the last reset), once `width >
grace_period`, each split of the window at a bucket boundary into an older
`W0` (`n0` rows) and a newer `W1` (`n1`) is tested, oldest boundary first:

```text
cut when  |S0/n0 - S1/n1| > eps_cut,  both n0, n1 >= L  (L = min_window_length)
eps_cut = sqrt(2 (1/m) (V/width) d') + (2/3) d' (1/m)
d'      = ln(2 ln(width) / delta)
1/m     = 1/(n0 - L + 1) + 1/(n1 - L + 1)
```

A cut drops the oldest bucket (`width -= n; S -= u; V -= v + n width (u/n -
S/width)^2 / (n + width)`) and restarts the scan on the shrunken window; the
row is flagged if any cut was made.

**Page-Hinkley** (Page 1954; Hinkley 1971; two-sided as Mouss et al. 2004).
With `xbar_t` the running mean of the rows since the reset, this one included:

```text
m_t = alpha m_{t-1} + (x_t - xbar_t - delta),   M_t = min_{s <= t} m_s,   up:   m_t - M_t > lambda
u_t = alpha u_{t-1} + (x_t - xbar_t + delta),   U_t = max_{s <= t} u_s,   down: U_t - u_t > lambda
```

tested once `t >= min_instances`; `mode` picks up, down or either.

**DDM** (Gama et al. 2004). `x_t = 1` for an error; `p_t` the error rate of
the rows since the reset, `s_t = sqrt(p_t (1 - p_t) / t)`. Once `t >
warm_start`: if `p_t + s_t <= p_min + s_min` the minimum moves to `(p_t,
s_t)`; warning when `p_t + s_t > p_min + 2 s_min`, drift when `p_t + s_t >
p_min + 3 s_min`.

**EDDM** (Baena-Garcia et al. 2006). On each error, the distance in rows
since the previous error (the first from the reset) enters a running mean
`p'` and sample standard deviation `s'` (Welford, `ddof = 1`). Once `t >
warm_start`: a new maximum of `p' + 2 s'` is recorded; otherwise, once more
than `warm_start` errors have been seen, `level = (p' + 2 s') / max`, drift
when `level < beta`, warning when `level < alpha`. The warning is refreshed
only on error rows.

**KSWIN** (Raab et al. 2020). A ring of the last `window_size` values. Once
full, each row draws `r = stat_size` values without replacement from the
older `window_size - r` and compares them with the newest `r` by the
two-sample Kolmogorov-Smirnov statistic `D = max_x |F_old(x) - F_new(x)|`.
With equal sizes `D = h / r` for an integer `h`, and its exact null tail is
the Gnedenko-Korolyuk sum

```text
P(D_{r,r} >= h/r) = 2 sum_{j >= 1} (-1)^(j+1) C(2r, r - j h) / C(2r, r)
```

which the prototype computes in integers and rounds once (`ks_pvalue_equal`).

### Where river departs from the papers

The prototype's defaults are river's, so the comparison can be exact; each
departure is an option whose paper form is measured in §2.

| detector | river 0.27.0 | the paper |
|---|---|---|
| ADWIN | Bernstein bound on the window's variance, `d' = ln(2 ln(width)/delta)` | Theorem 3.1 is proved for the Hoeffding form, values in [0, 1]: `eps = sqrt((1/(2m)) ln(4/delta'))`, `delta' = delta/n`, `1/m = 1/n0 + 1/n1`; the Bernstein form is the paper's "in practice" variant (§4), with no theorem |
| ADWIN | `1/m` shifted by `min_window_length` (MOA's form) | `1/m = 1/n0 + 1/n1` |
| ADWIN | tests every `clock = 32` rows, none until `width > grace_period = 10` | every row |
| ADWIN | after a flag, the whole window is dropped on the next row | the window keeps `W1` |
| ADWIN | no cap on the width | none either; the library's state rule needs one (§4) |
| Page-Hinkley | fading `alpha = 1 - 1e-4` on both sums | `alpha = 1` |
| Page-Hinkley | mean kept as `m += (x - m)/n` | `sum / n` (same flags, §2) |
| DDM | `p += (x - p)/n` | `errors / n`, one rounding (same flags, §2; up to 265 ulps apart, §8) |
| KSWIN | flag when scipy's exact `p <= alpha` and `D > 0.1` | flag when `D > sqrt(-ln(alpha) / r)`, whose size is about `2 alpha` (§1b) |
| KSWIN | the draw is CPython's `random.Random(seed).sample`, **re-seeded at every reset**, so the same index sequence repeats after each flag | a uniform draw |
| KSWIN | on a flag the window is cut to its newest `r` values, but the next row's reset empties it: the cut is dead code | the window keeps the newest `r` |

The `D > 0.1` clause never binds at river's defaults: `p <= 0.005` at `r =
30` needs `h >= 14`, `D >= 0.467`.

## Results

### 1. Flag for flag and bit for bit against river

Each case on three stream kinds (no drift; a step of +1 sd, or of the error
rate from 0.10 to 0.25, at row 5,000; a ramp to the same over 1,000 rows),
three seeds of 10,000 rows each, fed row by row to the prototype and to river.
Compared on every row: the flag (and DDM's and EDDM's warning), and each
internal field river exposes, by its bits: ADWIN's `width`, `total`, sum of
squares (`variance`) and `estimation`; Page-Hinkley's mean, both sums and
their extremes; DDM's `p`, `p_min`, `s_min`, `p_min + s_min`; EDDM's distance
mean, sum of squares and maximum; KSWIN's p-value.

| case | stream | rows | flags (ours) | flags (river) | flag rows differing | warning rows differing | field values not bit-equal | max ulps |
|---|---|---|---|---|---|---|---|---|
| ADWIN | none | 30000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ADWIN | step | 30000 | 3 | 3 | 0 | 0 | 0 | 0 |
| ADWIN | gradual | 30000 | 6 | 6 | 0 | 0 | 0 | 0 |
| ADWIN clock=1, delta=0.05 | none | 30000 | 1 | 1 | 0 | 0 | 0 | 0 |
| ADWIN clock=1, delta=0.05 | step | 30000 | 4 | 4 | 0 | 0 | 0 | 0 |
| ADWIN clock=1, delta=0.05 | gradual | 30000 | 8 | 8 | 0 | 0 | 0 | 0 |
| ADWIN on bits, M=3 | none | 30000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ADWIN on bits, M=3 | step | 30000 | 3 | 3 | 0 | 0 | 0 | 0 |
| ADWIN on bits, M=3 | gradual | 30000 | 3 | 3 | 0 | 0 | 0 | 0 |
| Page-Hinkley, threshold 25 | none | 30000 | 69 | 69 | 0 | 0 | 0 | 0 |
| Page-Hinkley, threshold 25 | step | 30000 | 69 | 69 | 0 | 0 | 0 | 0 |
| Page-Hinkley, threshold 25 | gradual | 30000 | 78 | 78 | 0 | 0 | 0 | 0 |
| Page-Hinkley up, alpha=1 | none | 30000 | 37 | 37 | 0 | 0 | 0 | 0 |
| Page-Hinkley up, alpha=1 | step | 30000 | 41 | 41 | 0 | 0 | 0 | 0 |
| Page-Hinkley up, alpha=1 | gradual | 30000 | 52 | 52 | 0 | 0 | 0 | 0 |
| DDM | none | 30000 | 0 | 0 | 0 | 0 | 0 | 0 |
| DDM | step | 30000 | 5 | 5 | 0 | 0 | 0 | 0 |
| DDM | gradual | 30000 | 3 | 3 | 0 | 0 | 0 | 0 |
| EDDM | none | 30000 | 68 | 68 | 0 | 0 | 0 | 0 |
| EDDM | step | 30000 | 87 | 87 | 0 | 0 | 0 | 0 |
| EDDM | gradual | 30000 | 100 | 100 | 0 | 0 | 0 | 0 |
| KSWIN | none | 30000 | 30 | 30 | 0 | 0 | 25536 | 3 |
| KSWIN | step | 30000 | 34 | 34 | 0 | 0 | 25176 | 3 |
| KSWIN | gradual | 30000 | 33 | 33 | 0 | 0 | 25254 | 3 |
| KSWIN W=300, r=50 | none | 30000 | 39 | 39 | 0 | 0 | 13148 | 4 |
| KSWIN W=300, r=50 | step | 30000 | 39 | 39 | 0 | 0 | 13125 | 4 |
| KSWIN W=300, r=50 | gradual | 30000 | 41 | 41 | 0 | 0 | 12740 | 4 |

**Every flag and warning row agrees** (0 of 810,000 rows differ), and every
ADWIN, Page-Hinkley, DDM and EDDM field is bit-equal on every row. ADWIN's
bucket compression and its check cadence, which the request expected might
not match, are deterministic and reproduce exactly: a full level merges its
two oldest buckets by the same formula in the same order, the cascade stops
at the same level, and the cadence is the row count since the last reset
(`tick % clock == 0`), which is state, not a clock. The only `libm` call,
`ln` in `d'`, is the platform's `log` on both sides (Python's `math.log`,
Rust's `f64::ln`); it decides a comparison and enters no state (§8 has how
close the comparison ever came).

KSWIN's flags agree because the prototype draws with CPython's
`random.Random(seed).sample` exactly as river does, re-seeding at every
reset. Its p-value differs from river's on most tested rows by 1 to 4 ulps:
river's comes from scipy's evaluation of the tail, the prototype's from the
integer sum rounded once (1b). The decision depends on `D` only through the
integer `h = r D`, which agrees on every pair (1c); the double `D` does not
(1c), and river does not keep it.

### 1b. KSWIN's exact p-value, and the paper's critical value

For every `h` from 1 to `r`, scipy's `ks_2samp(method="exact")` on two
samples built to have `D = h/r`, against the Gnedenko-Korolyuk sum in Python
integers divided once (`fractions.Fraction`, rounded to nearest).

| r | max ulps | at h | smallest h with p <= 0.005 | its size P(D >= h/r) | paper's smallest h | its size |
|---|---|---|---|---|---|---|
| 10 | 2 | 9 | 8 | 2.06e-03 | 8 | 2.06e-03 |
| 30 | 3 | 8 | 14 | 2.53e-03 | 13 | 6.55e-03 |
| 50 | 4 | 17 | 18 | 2.83e-03 | 17 | 5.84e-03 |
| 100 | 11 | 99 | 25 | 3.73e-03 | 24 | 6.13e-03 |
| 300 | 11 | 96 | 43 | 4.16e-03 | 40 | 9.59e-03 |

scipy's value is within 11 ulps of the correctly rounded one up to `r = 300`.
River's rule rejects at `h >= 14` at `r = 30`, a size of 2.53e-3: the
discreteness of `D` leaves half of `alpha = 0.005` unused. The paper's rule,
`D > sqrt(-ln(alpha)/r)`, rejects one `h` lower and has size 6.55e-3 at `r =
30`, 9.59e-3 at `r = 300`. That threshold is the asymptotic Smirnov value with
`alpha` where the two-sided limit, `P(sqrt(r/2) D > c) -> 2 exp(-2 c^2)`, needs
`alpha / 2`: the paper's rule tends to size `2 alpha`, river's stays below
`alpha`.

### 1c. The statistic `D` against scipy's

Pairs of samples rounded to one decimal (so with ties), the second shifted by
a uniform amount; the prototype's `D` (`h / r` for equal sizes, `max |i/na -
j/nb|` otherwise) against `ks_2samp(...).statistic`.

| n old | n new | pairs | D not bit-equal | max ulps | h = round(r D) differing |
|---|---|---|---|---|---|
| 30 | 30 | 300 | 178 | 6 | 0 |
| 50 | 50 | 300 | 170 | 8 | 0 |
| 100 | 100 | 300 | 193 | 7 | 0 |
| 300 | 300 | 300 | 209 | 10 | 0 |
| 70 | 30 | 300 | 55 | 8 | - |
| 970 | 30 | 300 | 45 | 8 | - |
scipy returns `D` as the difference of two `linspace` values (`k (1/n)`
each), not the `h / r` its exact p-value is computed from, so its double sits
up to 10 ulps from the correctly rounded `h / r`. The integer `h` agrees on
every pair, ties included: `ks_h` advances both sorted samples past each
pooled value before measuring the gap, as scipy's `searchsorted(side="right")`
does.

### 2. The paper's forms where river departs from them

Each paper form against the river form (the prototype's default) on five
seeds of the same streams; "first flag after the change" is the median over
the seeds of the rows from the change to the first flag at or after it (the
change row is 1), whatever caused it.

| form | stream | flags, river's form | flags, paper's form | rows differing | first flag after the change, river's (median) | paper's (median) |
|---|---|---|---|---|---|---|
| ADWIN keeps W1 after a cut | none | 0 | 0 | 0 | - | - |
| ADWIN keeps W1 after a cut | step | 5 | 16 | 11 | 24 | 24 |
| ADWIN keeps W1 after a cut | gradual | 10 | 42 | 42 | 472 | 472 |
| ADWIN clock=1 (every row) | none | 0 | 0 | 0 | - | - |
| ADWIN clock=1 (every row) | step | 5 | 5 | 10 | 24 | 23 |
| ADWIN clock=1 (every row) | gradual | 10 | 10 | 20 | 472 | 466 |
| ADWIN Hoeffding bound (Thm 3.1), bits | none | 0 | 0 | 0 | - | - |
| ADWIN Hoeffding bound (Thm 3.1), bits | step | 5 | 5 | 10 | 152 | 280 |
| ADWIN Hoeffding bound (Thm 3.1), bits | gradual | 5 | 5 | 10 | 664 | 1048 |
| Page-Hinkley alpha=1 | none | 123 | 128 | 91 | - | - |
| Page-Hinkley alpha=1 | step | 123 | 131 | 148 | 16 | 10 |
| Page-Hinkley alpha=1 | gradual | 133 | 137 | 148 | 180 | 138 |
| Page-Hinkley mean as sum/n | none | 123 | 123 | 0 | - | - |
| Page-Hinkley mean as sum/n | step | 123 | 123 | 0 | 16 | 16 |
| Page-Hinkley mean as sum/n | gradual | 133 | 133 | 0 | 180 | 180 |
| DDM p = errors/n | none | 0 | 0 | 0 | - | - |
| DDM p = errors/n | step | 7 | 7 | 0 | 431 | 431 |
| DDM p = errors/n | gradual | 7 | 7 | 0 | 985 | 985 |
| KSWIN D > sqrt(-ln(alpha)/r) | none | 53 | 122 | 159 | - | - |
| KSWIN D > sqrt(-ln(alpha)/r) | step | 60 | 125 | 171 | 23 | 19 |
| KSWIN D > sqrt(-ln(alpha)/r) | gradual | 59 | 113 | 156 | 329 | 274 |
| KSWIN newest r vs all older (no draw) | none | 53 | 25 | 76 | - | - |
| KSWIN newest r vs all older (no draw) | step | 60 | 29 | 81 | 23 | 19 |
| KSWIN newest r vs all older (no draw) | gradual | 59 | 29 | 84 | 329 | 595 |

- **ADWIN keeping `W1`** flags the same change again as the old rows left in
  `W1` (a cut is tested only at bucket boundaries) are found and dropped:
  16 flags for 5 changes on the step. River's reset avoids the repeats and
  discards the post-change rows already seen; the first flag is the same row.
- **ADWIN at `clock = 1`** moves the first flag earlier, by up to 31 rows
  (§3: a median of 33 rows against 56 at `clock = 32`), at 10 to 20 times the
  cost (§5).
- **ADWIN's Hoeffding bound** (Theorem 3.1, for values in [0, 1]) is looser
  than the Bernstein bound on a stream whose variance is well below the
  range's 1/4: on bits at a rate of 0.1 to 0.25 it waits 280 rows against 152.
- **Page-Hinkley at `alpha = 1`** flags earlier and slightly more often; the
  mean kept as `sum / n` instead of `m += (x - m)/n` moves no flag.
- **DDM's `errors / n`** moves no flag (the two rates are up to 265 ulps apart, §8).
- **KSWIN's paper rule** flags 2.3 times as often on the null (122 against
  53), as its size (6.55e-3 against 2.53e-3) says. The deterministic draw
  (newest `r` against all 70 older values, asymptotic critical value) flags
  half as often on the null and is slower on the ramp (595 against 329 rows):
  its old sample spans the ramp's start.

### 3. False alarms and delay

TABLE3

TABLE3A

- **ADWIN is far inside its bound.** No false cut in 62,400 checks at `delta =
  0.002`, Bernstein or Hoeffding, at either cadence: the 95% upper limit, 4.8e-5
  a check at `clock = 32` and 1.5e-6 at `clock = 1`, is 2.4% and 0.08% of
  `delta`. At `delta = 0.1` the rate is 1.5e-3 a check, 1.5% of `delta`. The
  bound is loose by design: it is a union over every split with `delta' =
  delta / ln(width)`, and the Bernstein term `(2/3) d' / m` alone keeps short
  sub-windows from cutting.
- **ADWIN's delay** sits where the expected gap first crosses `eps_cut` (34
  rows for a +1 sd step at the split, 151 for the error-rate step), rounded up
  to the next check: at `clock = 32` the flags fall on rows 24 or 56 after the
  change (5,024 and 5,056 are multiples of 32), at `clock = 1` the median is 33.
  Every seed is within Theorem 3.1's factor-2 `n1` plus the clock's wait.
- **Page-Hinkley alarms often on a unit-variance null at river's defaults**:
  one every 1,670 rows at `lambda = 50`, one every 390 at `lambda = 25`. Its
  sums integrate deviations from a running mean, so under no change `m_t -
  M_t` is the range of a random walk with memory `1/(1 - alpha) = 10,000` rows;
  `lambda` has to scale with the stream's standard deviation times the square
  root of the memory, which river's defaults do not do. The paper gives no
  false-alarm bound for the running-mean form (Page's average run length is
  for a known mean, the tabular CUSUM the scan leaves for later).
- **DDM** alarms 181 times in 200 null streams of 10,000 rows and misses the step on 2 seeds of 200; its delay is long
  (median 361 rows) because `p_min + 3 s_min` is set by the best stretch
  seen since the reset, which the 5,000 pre-change rows make tight only slowly.
  The 3-sigma rule is a normal-approximation band, not a sequential bound.
- **EDDM** alarms 3,392 times on the null (17 a stream): at an error rate of
  0.1 the distances between errors are geometric with a large spread, and the
  ratio to the running maximum of `p' + 2 s'` falls below 0.9 by chance. The
  paper gives no bound.
- **KSWIN** alarms about once every 780 rows on the null: each test rejects
  with probability 2.5e-3 and, at river's `window_size = 100`, a fresh window
  is tested on every row once full. 3b has the test's size.

TABLE3B

Without resets each test rejects at the exact size (2.5e-3 measured, 2.53e-3
exact); the tests overlap (each shares `r - 1` new rows and most of the old
sample's pool with the next), so the `z` computed as if independent overstates
the precision. River's reset turns each cluster of rejections into one alarm,
so its alarms a row are fewer than its tests' size. With values rounded to
one decimal or to integers (ties) the continuous tail is conservative: the
rejection rate falls to RATE1 and RATE0.

### 4. State size

TABLE4

**ADWIN**, in river's layout (each level a fixed `M + 1` totals and `M + 1`
variances, and a count; six scalars):

```text
levels(W) <= 1 + floor(log2((W + M - 1) / M))
bytes(W)  <= levels(W) (16 (M + 1) + 8) + 48          (M = 5: 104 levels(W) + 48)
buckets   <= M levels(W)
```

Every level below the top holds at least `M - 1` buckets once the level above
it exists (a merge leaves `M - 1`, and a cut only ever removes the oldest
bucket, from the top level), so a window of `L` levels has at least `M
2^(L-1) - (M - 1)` rows. The bound is reached: counted levels equal it at
every mark from 1,000 to 1,000,000 rows, and no row of the million exceeded it.
At `M = 5`: 880 bytes at 1,000 rows, 1,192 at 10,000, 1,920 at 1,000,000 and
4,000 at `2^40`. A compact layout (`M` slots, a one-byte count) is
`levels (16 M + 1) + 48`: 3,126 bytes at `2^40`, the scan's "3.3 kB". River's
own pickle (1,233 to 2,129 bytes) grows by the same per-level step. With
`max_width` the levels stop at the cap's bound (8 at 1,000, 11 at 10,000).

**Page-Hinkley** 7 doubles (count, mean, two sums, their extremes, the flag),
56 bytes; **DDM** 7, 56 bytes; **EDDM** 9, 72 bytes: none depends on the
stream. **KSWIN** is its ring, `8 W` bytes, plus 32 for its scalars, plus, for
river's random draw, the Mersenne Twister's 624 words and position, 2,500
bytes: `8 W + 2,532` (10,532 at `W = 1,000`, 82,532 at `W = 10,000`), or
`8 W + 32` for the deterministic draw. The window is the user's choice and
bounds it.

### 5. Per-row cost

TABLE5

Pure Python against river (Rust for ADWIN; Python and scipy for the others),
the means of 20,000 rows (KSWIN 2,000 or fewer) after the window is filled.
These rank the work; they do not predict the Rust option's cost. ADWIN's row
is `O(1)` amortized plus, every `clock` rows, a scan of its `O(M log W)`
buckets: going from 1,000 to 10,000 rows adds three levels, and the cost
grows by a tenth at `clock = 32`, by about half to double at `clock = 1`
(the scan every row). KSWIN's row is the draw and two sorts, `O(r log r)`,
independent of `W` at fixed `r` (24 to 26 us here); with `r = W / 10` it is
`O(W log W)`. River's KSWIN spends about half a millisecond a test in
`scipy.stats.ks_2samp`'s overhead. Page-Hinkley, DDM and EDDM are a few
flops a row.

### 6. Zero-weight rows

TABLE6

Every detector skips a zero-weight row entirely: its state's bytes before and
after are equal (a pickle of `save()`), it flags nothing, and on the rows of
positive weight its flags are river's on the stream with the zero-weight rows
removed, the first row included. That includes the counters a chunking could
move (ADWIN's `tick`, KSWIN's draw).

### 7. Chunk invariance

TABLE7

One pass against 201 chunks at random boundaries, the state saved (a pickled
`save()`) and loaded into a fresh detector at every boundary, on a stream with
5% zero-weight rows: the flags are identical on every row and the final
states' bytes equal. Nothing in a detector depends on where a chunk ends: each
reads one row at a time, and its counters (ADWIN's `tick`, the rows since the
reset, KSWIN's random state) are in the saved state.

### 8. Numerical risk

TABLE8

**ADWIN's sums.** The window total `S` and sum of squares `V` are updated by
addition on insert and subtraction on each dropped bucket, so their rounding
accumulates over the stream, not the window. Against exact rationals over the
window (always a suffix of the stream), after 200,000 rows with a cap that
drops a bucket every few rows: `S` is off by 1e-12 absolute at a zero mean,
by 9e-9 at a level of 1e4 and 1.4e-4 at 1e8 (relative 1e-15: the rounding of a
total of 1e11); `V` is off by 2e-14 relative at a zero mean, 6e-13 at 1e4 and
6e-9 at 1e8. The loss in `V` grows as the square of level over spread: the
merge term `n (t0/n - t1/n)^2 / 2` and the insert term `(x - S/(width - 1))^2`
subtract two means of size `level` to leave a difference of size `spread`. At
a level 1e8 times the spread, 6e-9 relative is still far below anything the
cut test resolves: §8's last table shows the test never came within 6.6e-5
of a tie. A Rust option that needs more can centre the stream on its first
value (a shift leaves every difference of means unchanged).

**Running means.** River's `m += (x - m)/n` and `sum / n` drift apart from
the exact mean by tens to hundreds of ulps over 10^4 to 10^6 rows (at a
level of 1e3, 68 and 227 ulps after a million rows; the ulps at a level of 0
are of a mean near 1e-3, so they look larger than they are). Neither form is
the better here, and neither moves a flag in §2; matching river bit for bit
requires river's form.

**Page-Hinkley's sum.** At `alpha = 1` the sum `m_t` integrates `-delta` a row
and drifts to -4,468 over a million rows; its rounding is 8.8e-8 absolute
there (2e-11 relative), five hundred million times below `lambda = 50`. With
river's fading `alpha = 1 - 1e-4` the sum stays near its 10,000-row memory
and the gap to a `long double` run is 1.8e-11.

**DDM.** The recursive rate and `errors / n` differ by up to 265 ulps over a
million rows and by 75 at the end; DDM compares `p + s` to `p_min + k s_min`
with gaps of order `s_min`, about 3e-4 at a million rows, so 265 ulps (about
4e-15 at a rate of 0.1) cannot move a flag unless the two sides are equal to 14 digits, which
they were in none of the runs.

**KSWIN's ties.** `D` is decided by `==` on the doubles, as scipy decides it,
and `h` matched scipy's on every tied pair (1c). The exact p-value assumes
continuous data; with ties it overstates the tail (3b), so the test is
conservative, never anti-conservative. Its double differs from scipy's by up
to 11 ulps (1b); the decision sits on a discrete set of p-values whose
nearest neighbours either side of 0.005 at `r = 30` are 2.53e-3 and 6.55e-3.

## Proposed tolerances for the Rust options' tests

| check | tolerance | reason |
|---|---|---|
| ADWIN against river: flag rows, `width`, `total`, the sum of squares, `estimation`, on every row | flags exact; fields bit-equal | measured 0 rows and 0 ulps (§1). The state takes only `+ - * /`, so it is bit-equal on every OS; the one libm call (`ln` in `eps_cut`) decides a comparison and enters no state, and the closest of 637,520 tests was 6.6e-5 relative from a tie (§8), eleven orders above an ulp |
| Page-Hinkley, DDM, EDDM against river: flags, warnings, every internal field | flags exact; fields bit-equal | measured 0 (§1); no libm call (`sqrt` is correctly rounded by IEEE 754) |
| KSWIN flags against river, same seed | exact, **only if** the Rust model reproduces CPython's draw (MT19937 and `random.sample`'s selection, re-seeded on reset) | the flags match with the draw reproduced (§1); without it, the deterministic form (`draw="all"`) is checked instead, against `scipy.stats.ks_2samp` on the two parts |
| KSWIN statistic `D` against `ks_2samp` | `h = round(r D)` exact; the double within 32 ulps | measured: `h` equal on every pair, the double up to 10 ulps off (1c), because scipy differences two `linspace` values; 32 leaves room for windows up to 10,000 |
| KSWIN p-value | 16 ulps against scipy's `ks_2samp(method="exact")`; 0 against the integer sum rounded once | scipy's Horner evaluation was 2 to 11 ulps from the correctly rounded value for `r` from 10 to 300 (§1b); the decision `p <= alpha` never sat that close: the p-values are a discrete set and the nearest to 0.005 at `r = 30` is 2.53e-3 |
| ADWIN false cuts on the null, per check | at most `delta`; the ledger's kind (statistical, tightening only) | measured 0 in 62,400 checks at `delta = 0.002` (95% upper limit 4.8e-5, 2.4% of `delta`) and 94 at `delta = 0.1` (1.5e-3, 1.5% of `delta`) (§3a) |
| ADWIN step delay | every seed within Theorem 3.1's `n1` (factor 2) plus `clock - 1` | measured 200 of 200 seeds in every case (§3a) |
| KSWIN test size, no reset | within 4 standard errors of the exact size, the standard error inflated by the tests' overlap (each window shares `r - 1` new rows with the next) | measured 2.5e-3 against the exact 2.53e-3 (§3b) |
| a zero-weight row (the first included): flags against river on the stream without it, the state before and after it | exact; bit-equal | measured 0 (§6) |
| chunk invariance, save and load mid-stream | bit-equal flags and final state | rule 3; measured 0 over 201 chunks (§7) |

## Caveats and what was not done

- **The weighted form has no oracle.** River's five detectors take a value,
  not a weight (`update(x)`); `stats.Mean` underneath takes a weight, but no
  detector passes one. The checkable form is the scan's: a zero weight is a
  row that never happened (it advances no count, no `clock` tick and no draw)
  and any positive weight is one row, checked against river on the stream
  with the zero-weight rows removed. A fractional weight as a fraction of a
  row (an ADWIN bucket of weight `w`, a weighted error rate in DDM) would be
  this library's own definition with nothing to check it; integer frequency
  weights could be checked against river on repeated rows, which only
  restates the definition. Not built.
- **"Advance nothing" differs from rule 9's "advance the clock, learn
  nothing".** These detectors have no clock but their row count, so a
  zero-weight row cannot advance one without learning. `emit_drift`'s
  clock-integrated Page-Hinkley (the existing option) is the place a clock
  belongs.
- **ADWIN needs a cap for the state rule.** Uncapped, its levels grow with the
  logarithm of the rows since the last flag (§4): bounded by
  `1 + floor(log2((W + M - 1)/M))` levels at width `W`, so 18 levels and
  1,920 bytes at a million rows and 38 levels, 4,000 bytes, at `2^40`.
  `max_width` drops the oldest bucket while the width exceeds the cap; river
  has no cap, so the comparison with river holds only while the width stays
  under it.
- **KSWIN's random draw** is the one piece whose exact match needs a foreign
  algorithm in the state: CPython's Mersenne Twister (2,500 bytes) and
  `random.sample`'s choice between its two selection methods. Either the
  Rust option reproduces both (then river is an exact oracle) or it takes the
  deterministic form (newest `r` against all older values; scipy is then the
  oracle, river only a statistical one). The study does not choose.
- **No CUSUM.** River has no CUSUM class (the scan's point 3); the tabular
  CUSUM's oracle is its published average run lengths, not river, and is not
  in this study. Neither are `FHDDM`, `HDDM_A`, `HDDM_W`, which river has.
- **Delays here are on one stream design** (a +1 sd step or ramp at row
  5,000 of 10,000; an error rate 0.10 to 0.25). They rank the detectors on
  that design; they are not general claims.
- The cost is pure Python (ours) or river's (Rust for ADWIN, Python with
  scipy for KSWIN) on this machine; the Rust option's cost is the task's to
  measure.
- The paper's Theorem 3.1 bounds the Hoeffding form; river's Bernstein form,
  which uses the window's estimated variance, has no theorem. Both are
  measured against `delta` (§3a); both are far inside it.
