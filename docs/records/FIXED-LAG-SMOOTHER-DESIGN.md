<!-- kind: record -->
# A fixed-lag smoother beside `kalman`, with an embargo: the design record

The design for task 496, part 1, written 2026-11-24 on branch
`vm/task-496-fixed-lag` from `main` at 13f5ffe4. Nothing here is built. The
questions in section 8 are for the user before part 2 starts; section 9 lists
what part 2 would build and test.

Input: the study of the same task on branch `vm/study-496` (head 03e010c7),
its record `docs/records/STUDY-496-SMOOTHER.md` and its script
`scripts/study_496.py` there. Neither is merged; this record cites the study
by section ("study §8") and says in section 10 where this design departs from
it and why. The study's numbers quoted here are its own; the ones this record
measured are said to be.

## 1. The problem

`kalman` is a regression whose coefficients drift as random walks. On each
row it emits the filtered estimate: the coefficients given the rows up to
and including this one. A **fixed-lag smoother** emits, a lag of `L`
observations later, the estimate of an earlier row's coefficients given the
`L` observations after it as well:

```text
filtered:   b_{s|s} = E[beta_s | y_1 .. y_s]        what kalman's coef is
smoothed:   b_{s|t} = E[beta_s | y_1 .. y_t],  t = s + L observations
```

The smoothed value is a better estimate of where the coefficients *were*
(study §3: at a lag of three times the smoother's memory it is within 0.05 of
a posterior standard deviation of the full-stream smoother), at the cost of
the delay. It is a research
and attribution output, never a prediction.

**The embargo** is two guarantees, which this design must hold together:

1. **Nothing future-dependent reaches a prediction.** A smoothed value for
   row `s` reads targets after `s`. The model's own prediction `pred` must
   never read one, so the smoother sits beside the filter and feeds nothing
   back: every prediction with the smoother on is the filter's, to the bit.
2. **The delayed value is placed where it was known, and named so.** It is
   written on the line of the row that completed its lag, under names that
   say it describes another row, with that row's stamp beside it. It is never
   written back onto the described row's own line, where a reader or a join
   would take it for something known then. A downstream model that uses it
   as a target for the described row must then wait for it, as the spec's
   `embargo` already makes a model wait for a target that looks ahead.

The library's existing `embargo` (a delay on learning a row until its target
is known; `docs/explanation/out-of-sample.md`, "A target that looks ahead")
and the look-ahead window operators (which hold a row's output until the
window ahead has closed) are the two mechanisms this is set beside; section 6
says how it relates to each.

`docs/records/BEYOND-O-STATE.md` (B4) named the fixed-lag smoother in 2026-08
and set the condition this record answers: "it breaks out-of-sample-ness by
design ... so it would need to be a clearly separated output that cannot be
mistaken for `pred`. Worth doing only with that separation designed first."

## 2. The model the smoother runs on

The smoother is the Rauch-Tung-Striebel (RTS) recursion run backward over the
filter's own states, so it is only as right as its picture of the filter.
`kalman` is not the textbook filter the study ran (study §1: one clock unit a
row, `Phi = I`, `Q` added every row, a known prior, unstandardized). From the
header of `crates/rheon-core/src/kalman.rs`, as it stands at 13f5ffe4:

- **The noise is charged per observation, not per row.** A covariance's
  process noise is `Q D²` (on a reverting slot `q_i g_i(D)²`), added once, on
  the row that next observes its target, `D` being the clock since it last
  did. A row that observes nothing (a null target, a weight of 0, a refused
  row) only adds its `d` to `D`. The noise is not additive over a split gap:
  `q (d₁² + d₂²) < q (d₁ + d₂)²`.
- **The transition runs every row.** `Phi = diag(2^(-d/r_i))` under
  `revert_half_life`, the identity without it; it is multiplicative over a
  split gap.
- **The coordinates are an anchor's.** Under `standardize`, `b` and `P` are
  held in the coordinates of the moments at the last re-map. The noise
  `Q D²`, diagonal in the current coordinates, is an arrowhead in the
  anchor's, and a re-map is a change of coordinates `b <- Mᵀ b`,
  `P <- Mᵀ P M`, after a row.
- **Slots are sized late.** A slope's prior `p0` is set on the first
  observing row on which its feature's scale is usable; until then its
  coefficient and variance are 0. In units of the noise the intercept's prior
  is diffuse: the first observing row takes the exact diffuse step.
- **A covariance can be sized again.** One whose diagonal leaves the numbers
  of at least 0 is re-sized from the noise at the end of the row.
- **`share_p`** keeps one covariance for every target; the means stay per
  target.
- **Units of the noise.** `P` and `Q` are held over the noise `σ²`; the
  readouts (`pred_var`, `se_coef`) multiply by `σ²` as it stands.
- **The square-root form** (`covariance_form="square_root"`) holds `P = C Cᵀ`
  and updates `C` by Carlson's step.

So the smoother's time index is **the covariance's observations**: the rows
`r_1 < r_2 < ...` that observe its target at a positive weight (under
`share_p`, any target). Between two observations `n - 1` and `n`:

```text
F_n = the product of the rows' transitions from r_{n-1} to r_n, each re-map's Mᵀ among them
Q_n = the noise charged on r_n (Q D_n², carried into the anchor's coordinates)
      + the prior p0 of any slot first sized on r_n
```

and the filter is, at observation `n`,

```text
prior:    b⁻_n = F_n b_{n-1},       P⁻_n = F_n P_{n-1} F_nᵀ + Q_n
pred_n  = z_nᵀ b⁻_n                 (before y_n: what kalman emits as pred)
update:   s_n = z_nᵀ P⁻_n z_n + 1/w_n,  g_n = P⁻_n z_n / s_n
          b_n = b⁻_n + g_n (y_n − pred_n),  P_n = P⁻_n − g_n z_nᵀ P⁻_n
```

This is the linear Gaussian state-space model whose smoother part 2 builds.
A slot first sized at `n` enters as noise: its coefficient was held at 0 with
variance 0, and then becomes `N(0, p0)`. Rows between observations carry the
transition alone, with no noise until the next observation; section 5.5 says
what that means for what such a row emits.

## 3. The smoother

### 3.1 The lag-L RTS pass

After observation `t`'s update, run the RTS recursion back `L` observations:

```text
m <- b_t,  V <- P_t
for n = t-1 down to t-L:
    G_n = P_n F_{n+1}ᵀ (P⁻_{n+1})⁻¹                    (fixed once observation n+1's prior is formed)
    m   <- b_n + G_n (m − F_{n+1} b_n)
    V   <- P_n + G_n (V − P⁻_{n+1}) G_nᵀ
emit for observation t−L:  (m, V) = (b_{t-L|t}, P_{t-L|t})
```

Everything in a step but the running `(m, V)` is fixed once observation
`n + 1`'s prior is formed, so a slot can hold the step's constants and never
`F`:

```text
c_n = b_n − G_n b⁻_{n+1}             (per target)
K_n = P_n − G_n P⁻_{n+1} G_nᵀ        (per covariance, symmetric)
pass:  m <- c_n + G_n m,   V <- K_n + G_n V G_nᵀ
```

A slot is `(c_n, G_n, K_n)`, the study's count (`b`, `P`, `G`). A re-map
inside the window changes the coordinates between two slots; mapping every
slot to the new anchor at the re-map (`c <- Mᵀ c`, `G <- Mᵀ G M⁻ᵀ`,
`K <- Mᵀ K M`, each `O(m²)` with `M` an arrowhead) keeps the pass in one
coordinate system, at `O(L m²)` a re-map. `kalman`'s header measured 3 to 45
re-maps in 30,000 rows.

**It is exact.** The RTS step at `n` reads only observation `n`'s filtered pair
and the smoothed pair at `n + 1`, never anything before `n`, so stopping after
`L` steps changes nothing: the emitted pair is the full smoother of the
prefix, read at `t − L`. The study measured it so against statsmodels'
`KalmanSmoother` and filterpy's `rts_smoother` run on each prefix: within
2e-16 at `k = 2, L = 1` and 4.4e-13 at `k = 20, L = 100` (study §6, §15).
This record re-ran the study's sections `full`, `exact`, `forms`,
`contraction`, `oos` and `forget` in its `--quick` mode with the lockfile's
statsmodels 0.15.0 and filterpy 1.4.5: the same numbers to the printed digit.

**The gap to the full smoother** has a closed form (study §1): with
`A = G_s G_{s+1} ... G_{s+L-1}`,

```text
b_{s|s+L} − b_{s|N} = A (b_{s+L|s+L} − b_{s+L|N}),     P_{s|s+L} − P_{s|N} = A (P_{s+L|s+L} − P_{s+L|N}) Aᵀ
```

so it shrinks with the product of the gains, a memory of about
`sqrt(R / (q x²))` rows on one feature and longer as `k` grows (study §3, §6).
`L` is the user's choice against that memory, not a number with a universal
good value.

### 3.2 Which form holds the slots

The study compared three (study §5, §9, §11 to §13):

| form | per row, mean | per row, mean and covariance | state |
|---|---|---|---|
| ring of `(c_n, G_n, K_n)`, the pass on every row | `L m²` + one `m³` solve | `2 L m³` + one solve | `L (m + m(m+1)/2 + m²)` floats |
| reduced augmented: `a_j`, `C_j = cov(β_{t-j}, β_t)`, `D_j` | `2 L m²` | `3 L m²` | the same count |
| full augmented filter on `(β_t .. β_{t-L})` | `(m(L+1))²` | `(m(L+1))²` | `m(L+1)(m(L+1)+3)/2`: 16 MB at `k = 20, L = 100` |

The full augmented form is ruled out (study §9). Between the other two:

- **The reduced form** is the cheapest when every row emits a mean and a
  covariance (study §13, Python: 187 µs a row against 1,074 for the cached
  ring at `k = 20, L = 100`), needs no solve, and keeps `C_L`, which the
  bound of section 5.8 reads. But its `L` blocks move on every observation,
  emitted or not, and on every re-map.
- **The ring** stores what the filter already computed and moves one slot an
  observation. Its backward pass is needed only on a row that emits, so with
  the mean alone the work an observation is the one solve for `G_n`, plus
  `L m²` on a row that emits. This laziness is this record's observation; the
  study timed the pass on every row and did not measure it.

**Recommended: the ring** (question 2). It is the form the oracles compute
(the RTS recursion), its slot is the filter's own pair, and its cost falls
with how often the fields are read. The reduced form stays the check it was
in the study, in a test.

## 4. State and cost

With `m` slots a covariance (the features and the intercept), `T` targets
and `c` covariances (`T`, or 1 under `share_p`), in f64:

| part | floats |
|---|---|
| `kalman` as it is | `T m` means, `c m²` covariances (held as full row-major squares), the standardizer's `O(m)`, the noise's `O(T)` |
| the ring, mean only | `L (c m² + T m)`: `G_n` per covariance, `c_n` per target |
| the ring, with standard errors | `L (c (m² + m(m+1)/2) + T m)`: and `K_n`, stored as a triangle |
| each slot's stamp and break mark | `L` stamps of 16 bytes and a flag |

For one target (`c = T = 1`) with standard errors, the study's figures
(study §12, the filter's mean and triangle included; the study's `k` is the
slot count):

| `m` | `L` | the filter's pair and the ring, bytes | over `kalman`'s own `b` and `P` |
|---|---|---|---|
| 5 | 10 | 3,760 | about 16 |
| 5 | 100 | 36,160 | about 150 |
| 20 | 10 | 52,240 | about 16 |
| 20 | 100 | 505,840 | about 150 |

(The study's ratio column, 24 to 275, divides by a symmetric `P`;
`kalman` holds a full square, `m + m²` floats with its mean, so the ratio to
its actual state is lower, as above.) All of it is bounded by `m`, `T` and `L`, never by the stream's
length. Every size is known when the spec is built, so the bank can hold it to
`max_state_bytes` (256 KB a group by default, 64 MiB at most) then, before a
row is read: at `m = 20, L = 100` one target is over the default (question 17).
Under the state rule a lag of 100 at 20 slots is a significant growth over
`kalman`'s state and is the user's to allow; the study left it open (study
§16) and so does this record.

**Cost per observation**, multiply-adds, the filter's own `O(m²)` aside:
one Cholesky of `P⁻_{n+1}` on the sized slots and two triangular solves for
`G_n`, about `m³ / 3 + m³`; on a row that emits, `L m²` for the mean and
`2 L m³` for the covariance. At `m = 21` the gain is about 12,000 an
observation against about 1,300 for the filter's own update (study §13's
count at 20): near ten times a row's work whatever `L`. A covariance pass at
`L = 100` is about 1.9 million on the rows that emit (question 18). No Rust has been timed; the study's
microseconds are one interpreted Python step a row and say nothing about
Rust.

## 5. How it meets the library's rules

### 5.1 Predictions are out of sample (hard rule 2)

The smoother reads the filter's states and writes only its own slots and
fields. Nothing it computes reaches `b`, `P`, the noise, the standardizer or
any gate, so every `pred`, `withheld_reason`, `weight_sum` and every other
field the filter writes is the same, bit for bit, with `smooth_lag` set or
not. The study held it (study §4): with the targets after row 1,000 changed,
every emission up to row 1,000 and every prediction was bit-identical, in all
three forms at every `L`, while later emissions moved. Part 2 holds the
equality on `kalman` itself across `tests/model_contract.rs`'s streams.

### 5.2 Chunk invariance (hard rule 3)

The ring moves one slot an observation and the pass reads only slots; nothing
depends on where a chunk ends. One chunk and 1,000 give the same fields to the
bit, and a state saved mid-ring loads and continues to the bit. Part 2 tests
1, 7, 333 and 1,000 rows a chunk, one thread against eight.

### 5.3 State files (hard rule 5)

The ring is new state, so `kalman`'s layout changes: `SCHEMA_VERSION` rises
(72 at 13f5ffe4) and, past 1.0, the current fixture sets are copied to
`state_fixtures/v<N>/`, a loader reads a state of the previous schema as a
`kalman` with no smoother, and the current sets are regenerated. The spec
gains fields, so the bank's schema and fixture move with it. The fixtures
freeze every form the ring can take (question 21; section 9).

### 5.4 `n_eff` (hard rule 8)

The smoother adds no weight and does not redefine one: `weight_sum` stays the
filter's, the accumulated weight before the row's update and before its own
decay. A lag counted in observations is a count of positive-weight rows, a
span the user sets, not a weight.

### 5.5 A row of weight 0 (hard rule 9)

A row of weight 0, a null target or a refused row observes nothing. It takes
no slot, counts in no lag, completes no lag and emits nulls in the smoothed
fields; the clock it adds reaches the next observation through `D`, as in the
filter. So the smoothed fields on every other row are those of the same
stream with that row removed and its clock carried into the next row. Part 2
holds that to the bit on an integer clock, with the dropped row's target NaN
and 1e300 (the study's check, section 8).

**What a non-observing row's coefficients are.** Under `kalman`'s model the
noise for a gap is charged at the next observation, so at a row `r` between
observations `n` and `n + 1` the state is `β_r = Phi_{n→r} β_n` with no noise
yet, and its smoothed value is `Phi_{n→r} b_{n|t}`, the previous observation's
carried by the transition: a step, not an interpolation. The study's answer
for a zero-weight row (study §8: the midpoint of its neighbours, a Brownian
bridge) is right for noise charged a row at a time, which `kalman` stopped
doing so that a zero-weight row would move nothing. This design does not emit
a value for such a row; a reader who wants one reads the previous
observation's (question 8).

### 5.6 The math library

The pass is arithmetic and triangular solves; it calls no `exp`, `ln`,
`exp2` or `powf`. Its slots hold `c_n`, `G_n` and `K_n`, which carry `Phi`'s `exp2`
under `revert_half_life` and the standardizer's decays, as `kalman`'s own `P`
does. So a state fixture compares those slot fields across platforms within
the tolerance `kalman`'s `P` is compared within (none new is proposed), and
every other slot field exactly.

### 5.7 Guards by condition, not by row count

- **A slot not sized at the described observation** emits null for that
  coefficient: the filter held it at 0 with variance 0, not an estimate
  (question 14). The diffuse intercept before its first observation is such a
  slot.
- **A gain whose solve fails.** `P⁻_{n+1}` restricted to the sized slots is
  factored by Cholesky; a factor that does not exist at the rounding (a
  non-positive pivot) cuts the chain there (section 5.9) rather than emitting
  a value the arithmetic does not support.
- **No burn-in row count.** The study's tolerances start at row 500 (study
  §14, §16); the model withholds nothing by row number, and part 2's tests
  replace that cut with a self-normalised statistic (question 20).
- **`min_weight`** gates the prediction; the smoothed fields are a report, as
  `se_coef` is, and are not gated by it (question 15).

### 5.8 The remaining revision, in stream

The study's "RMS predicted by the covariances" (study §3) reads
`P_{s|s+L} − P_{s|N}`, which needs the stream's end, so it is not an in-stream
quantity, as the study's reading of it suggests. What is in stream is a bound:
`P_{t|N} ≥ 0`, so

```text
Var(b_{s|N} − b_{s|t} | y_1..y_t) = A (P_{t|t} − P_{t|N}) Aᵀ  ≤  A P_{t|t} Aᵀ = C_L P_t⁻¹ C_Lᵀ,
C_L = cov(β_s, β_t | y_1..y_t) = A P_t
```

a statistic valid at every sample size that says how far later rows could
still move the emitted value. It is unmeasured: neither its tightness nor its
cost (`A` is `L` products of `m x m` in the ring; `C_L` is free in the
reduced form) has been run. Question 16 recommends measuring it in part 2
before exposing a field.

### 5.9 Breaks in the chain

The RTS recursion is valid only across steps the filter's model connects.
Where the filter is not a continuation of itself, the smoother cannot pass
back across: a covariance re-sized at the end of a row, a re-map refused whole
(the state keeps its numbers in new coordinates), and a gain whose Cholesky
fails. At a break the pass stops: the observations before it are smoothed
with what came before the break, and are still emitted when their lag
completes. A session change and a capped gap call `clear_lags`, which clears a
model's row-lagged inputs; `kalman`'s filter continues through both, with its
noise charged for the capped gap, so the smoother continues too
(question 13).

## 6. The embargo: where the values go and what reads them

- **Fed back nowhere** (section 5.1).
- **Emitted on the line of the observation that completes the lag**, after
  that row's update, as `coef` is, describing observation `t − L`, with that
  observation's stamp (the clock value, or the row number with no clock
  column) beside it. Each observation is described once. The model knows no
  clock column (hard rule 4): it keeps each slot's `Stamp`, which the stream
  already hands it before a step (`stamp_next`), and the bank renders it.
- **Not written back onto the described row's line.** Doing so would hold
  every row's output back `L` observations, as a look-ahead window holds
  `with_windows`' output, and would move the timing of every other field the
  bank writes (question 6).
- **Used as a feature** on the line it appears, it carries that row's target,
  exactly as `coef` does. Emitting before the row's update instead (the value
  of observation `t − 1 − L`, from rows before `t`) would make it safe there,
  one observation older (question 7).
- **Used as a target** for the described row, joined back by its stamp, the
  downstream spec needs an `embargo` at least the lag's span on the clock. A
  lag in observations has no fixed span on a clock column; with no clock
  column it is at least `L` rows. Part 2 documents this in the how-to beside
  the look-ahead target, and the bank does not try to enforce a join it does
  not see.
- **Under the spec's own `embargo`** the filter learns a row only when it is
  released, so its observations are released rows, with the described row's
  own stamp: the delay is the embargo's plus the lag's (question 12).

## 7. The interfaces it touches (nothing changed now)

- **`rheon-core`.** A module `kalman/smooth.rs` for the ring, the pass and the
  breaks; an optional ring inside `Kalman`; one provided method on
  `OnlineModel` that writes the smoothed fields into the caller's buffers and
  returns `false` by default, as `support_coef_into` does, so no other model
  changes. `ModelState::Kalman` keeps its name (the variant names are the
  file's tags); its layout changes, hence the schema (section 5.3).
- **`rheon-polars`.** The spec's fields (`smooth_lag`, `emit_smoothed_se`),
  their checks (a lag of at least 1; a refusal under the square-root form, if
  question 4 says so; `max_state_bytes` at build), the output fields in
  `bank/fields.rs`, and the bank's schema and fixture.
- **The Python surface.** `rp.spec.kalman(..., smooth_lag=, emit_smoothed_se=)`,
  `tests/api_surface.txt`, `docs/MODELS.md`, `docs/OUTPUTS.md`, `CHANGELOG.md`.
  The output struct gains fields, which the PyO3 `ModelBank` and the Arrow
  output carry as they carry `coef`; no new interface of the four.
- **The command line.** The TOML spec reads the same two fields.
- **`docs/EXTENDING.md`'s steps** apply where a model gains a parameter and
  fields: the contract, the golden, the per-model sweeps, the registry.

## 8. Questions for the user, each with a recommendation

### A. The form

1. **Attach the smoother as an option of `kalman` (`smooth_lag=L`), not as a
   new kind?** Recommended: an option; the predictions stay `kalman`'s to the
   bit and the spec surface is one parameter.
2. **The ring of filtered slots with cached gains and a backward pass only on
   rows that emit, or the reduced augmented form?** Recommended: the ring
   (section 3.2).
3. **Count the lag in the covariance's observations, or in clock units?**
   Recommended: observations; a lag on the clock holds however many rows the
   clock packs into it, which no shape bounds.
4. **Refuse `smooth_lag` under `covariance_form="square_root"` in the first
   build?** Recommended: refuse; forming `P = C Cᵀ` would spend the digits that
   form exists to keep, and a square-root smoother is its own study.
5. **Smoothed standard errors opt-in (`emit_smoothed_se`), the mean always?**
   Recommended: yes; the mean needs no `K_n` slots, which are a third of
   the state at 20 slots.

### B. Emission and the embargo

6. **Emit on the line of the observation that completes the lag, never on the
   described row's own line?** Recommended: yes (section 6).
7. **After the row's update, as `coef` is, or before it, safe as a feature on
   its own line and one observation older?** Recommended: after, documented.
8. **A row that observes nothing (weight 0, null target, refused): no slot,
   no lag count, null fields, never described?** Recommended: yes; not the
   study's bridge value, which assumes noise charged a row at a time
   (section 5.5).
9. **Emit on every observing row whatever `coef_every` says?** Recommended:
   yes; a value describes one row once, and `coef`'s cadence would drop most.
10. **Field names `smoothed_coef`, `smoothed_se_coef` and `smoothed_for` (the
    described row's clock value or row number), laid out like `coef`, with
    `L` not in the names?** Recommended: yes.
11. **No end-of-stream flush in the output; the last `L` observations'
    values read from the saved state by a bank method?** Recommended: yes;
    the stream has no row after its last to carry them.
12. **Allowed with the spec's `embargo`, observations being released rows?**
    Recommended: yes.

### C. Breaks, guards and withholding

13. **Cut the chain where the filter does not continue itself (a re-sized
    covariance, a refused re-map, a failed Cholesky), and not at a session
    change or capped gap?** Recommended: yes (section 5.9).
14. **A slot not yet sized at the described observation emits null, rather
    than an exact diffuse smoother's value?** Recommended: null.
15. **`min_weight` does not gate the smoothed fields?** Recommended: yes; they
    are a report, as `se_coef` is.
16. **The in-stream bound on the remaining revision (section 5.8): measure in
    part 2 before any field?** Recommended: measure first, no field yet.

### D. State and cost

17. **Hold the ring to `max_state_bytes` when the spec is built, so one
    target at 20 slots and `L = 100` (about 506 KB) needs a raised cap?**
    Recommended: yes; this record states the bytes (section 4) for the state
    rule's decision.
18. **Accept one Cholesky a covariance an observation, `O(m³)`, for the cached
    gain?** Recommended: yes, with a bench case and its budget entry.

### E. Tests and the state file

19. **The oracle: statsmodels' `KalmanSmoother` on each prefix (intercept
    diffuse, slopes known at `p0`), read at `t − L`, within 1e-11 relative;
    filterpy only if the lockfile's 1.4.5 is raised?** Recommended: yes.
20. **Replace the study's burn-in of 500 rows with a self-normalised gap
    test?** Recommended: yes (section 9).
21. **The next schema version, the fixture sets copied, and a loader that
    reads an older `kalman` as one with no smoother?** Recommended: yes.

## 9. What part 2 would build, and the tests it needs

In three cuts, each mergeable alone:

1. **`rheon-core`: the ring.** `kalman/smooth.rs`; the ring in `Kalman`; the
   gain formed with each observation's prior; the pass; the breaks; the
   trait method; the schema bump and loader.
2. **`rheon-polars`: the spec and the fields.** `smooth_lag`,
   `emit_smoothed_se`, the checks, `smoothed_coef`, `smoothed_se_coef`,
   `smoothed_for`, the bank's schema, the state read of the last `L`
   observations.
3. **Python, docs, oracle, bench.** `rp.spec.kalman`'s parameters, the
   docstring's update equations (the RTS step above), `docs/MODELS.md`,
   `docs/OUTPUTS.md`, the how-to's note on joining the value back, the
   second-opinion test, the bench case and its budget.

The tests:

- **Exactness against a library (the oracle).** `tests/test_second_opinion_smoother.py`:
  statsmodels' `KalmanSmoother` with a time-varying design (the rows'
  features), `obs_cov = 1/w`, `state_cov = (ln 2 / coef_half_life)²`,
  transition the identity, and a mixed initialization (the intercept
  diffuse, the slopes known at `p0`), as
  `test_kalmans_square_root_at_a_vague_prior_is_statsmodels_exact_diffuse_filter`
  maps `kalman` unstandardized in units of the noise. Run on each prefix
  `y_1 .. y_t`, read at `t − L`, the mean and covariance within 1e-11
  relative (the study's threshold, 23 times its largest gap at `k = 20,
  L = 100`); `k` 2, 5 and 20, `L` 1, 10 and 100. statsmodels is already a dev
  dependency.
- **The Rust oracle.** A helper in `crates/rheon-core/src/oracle.rs`: the full
  RTS smoother on a prefix, its gains from faer's partial-pivot LU, which
  shares no arithmetic with the ring's Cholesky; the ring's emissions held to
  it within a bound derived from both computations' rounding.
- **The forms against each other.** The reduced augmented form, written in a
  test, against the ring within 1e-12 (study §10's threshold), as a second
  derivation of a different algorithm.
- **Out of sample.** Every `pred` and filter field bit-identical with the
  smoother on and off, on `model_contract.rs`'s streams; the study's
  perturbation test (targets after row `t0` changed, every emission up to `t0`
  unchanged, a later one moved).
- **A row of weight 0.** Inserted rows of weight 0 with targets NaN and 1e300
  against the stream with them dropped and the clock carried: every
  smoothed field on the kept rows to the bit, the inserted rows null; one
  such row first in the stream.
- **`kalman`'s paths.** `standardize` with re-maps inside the window
  (the anchor's coordinates mapped across them), `revert_half_life`, slots
  sized late, the diffuse intercept, `share_p` with targets null on different
  rows, `obs_var` and learned noise, clock gaps: each against the prefix
  oracle where one maps, and against the same stream smoothed offline from
  the filter's logged states where none does.
- **Breaks.** A forced re-size and a refused re-map: the observations before
  it emitted with the smoothing before it.
- **The gap, self-normalised.** In place of the study's rows from 500: the
  gap `b_{s|s+L} − b_{s|N}` over its own predicted standard deviation
  `sqrt(diag(P_{s|s+L} − P_{s|N}))`, on every row, data drawn from the model's
  own prior so the statistic is standard normal at every sample size; its
  mean square within a stated chi-squared interval. Whether that holds through
  the prior-dominated rows is not yet measured.
- **Chunks and threads.** 1, 7, 333 and 1,000 rows a chunk and one thread
  against eight, to the bit; a save mid-ring that loads, re-saves byte for
  byte and continues to the bit.
- **State fixtures** (hard rule 5): cases for a ring empty, partly filled,
  full and wrapped, with a break inside, with an unsized slot, under
  `share_p`, mean only and with standard errors, under the spec's `embargo`;
  a test that decodes each file and holds the new forms non-empty; the
  previous schema's sets loaded through the loader and re-saved as the
  current schema's bytes of the same case.
- **The state cap.** A spec over `max_state_bytes` refused at build, naming
  the bytes.
- **Bench.** A `kalman` case with `smooth_lag` (one target, `m` 6 and 21,
  `L` 10), its instruction and memory budgets.

## 10. The study against this design, and what was not verified

- **Noise per row against noise per observation.** The study charges `Q` on
  every row, a zero-weight row included (study §8, §11), and its dropped-stream
  oracle adds `Q` per clock unit. `kalman` charges `Q D²` once, at the next
  observation. The two agree on a stream observed every clock unit and part
  wherever a row observes nothing or the clock skips. The study's zero-weight
  results, its bridge midpoint among them, are for its model, not `kalman`'s.
- **The lag counted zero-weight rows.** The study's lag counts every row ("the
  dropped stream's lag `L − m` value when `m` zero-weight rows fall inside
  the window"); under hard rule 9 a row of weight 0 counts in no row-counted
  span, so here the lag counts observations.
- **A burn-in by row count.** The study's tolerances hold from row 500 on;
  the rule against row counts as answers to early-data problems names "first
  tested row N", hence question 20.
- **The in-stream gap.** The study says the gap's size "is known in-stream
  from `P_{s|s+L}` and `G`"; its prediction reads `P_{s|N}`, which is not. Only
  the bound of section 5.8 is in stream.
- **filterpy.** The study ran filterpy 2.0.0; the lockfile has 1.4.5, already
  a dev dependency (the study says it is not in `pyproject.toml`; it is).
  Re-run here at 1.4.5, the `zero` section stops in filterpy's
  `batch_filter`, which cannot take `z = None` in a list beside arrays; the
  sections before it reproduce. The sections `cost` and `seeds` (one seed,
  `--quick`) ran and passed their checks.
- **Only `kalman`'s simplest path was studied.** Unstandardized, `R = 1`, a
  known prior `10 I`, `Phi = I`, unit clock; no anchor, re-map, late sizing,
  diffuse intercept, learned noise, `share_p`, reversion or square-root form,
  and nothing run against `kalman`'s own output. Section 2's general recursion
  is this record's derivation from the source's header, unmeasured.
- **Forgetting** (study §7) is not in `kalman`, whose forgetting is its `Q`;
  it is out of scope here.
- **The plan's named oracle.** The task names statsmodels'
  `UnobservedComponents`; it has no regression on a time-varying row (study
  §15), so the oracle is `KalmanSmoother`, the state-space smoother the task
  also names.
- **Not verified here:** the study's 50-seed tolerance table (study §16) and
  its full-size timings; the lazy pass's saving; the bound of section 5.8;
  any Rust cost.
