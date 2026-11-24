<!-- kind: record -->
# Seasonal exponential smoothing (Holt-Winters and ETS): the design note

*Task 488, part 1: design only, 2026-11-24, from `main` at `555b45e5`. Nothing
is built. Sources: the time-series oracle scan (`vm-tools/research/
485-TIME-SERIES-SCAN.md` §4, on `vm-tools/oracle-gap-scan`), the estimator
study (`docs/records/ORACLE-STUDY-485.md` §13, §24, §25, on `vm/study-485`),
`crates/rheon-core/src/holt.rs`, and statsmodels 0.15.0's sources as installed
in the project's environment (`tsa/exponential_smoothing/ets.py`,
`_ets_smooth.pyx`, `initialization.py`, `tsa/holtwinters/`), read, not run.
Section (g) holds the questions for the Mac; part 2 waits on its answers.*

## (a) The variants, and their update equations

**The family.** ETS is an error type `E ∈ {A, M}`, a trend `T ∈ {N, A, Ad,
M, Md}` and a season `S ∈ {N, A, M}`: 30 forms. The scan classes every one,
with fixed smoothing parameters and known initial states, as *online
exactly*: a one-pass recursion over `O(1)` level and trend and `m` seasonal
values. That is the set this task builds.

**The error type does not enter the state recursion.** With the same
parameters, `(A, T, S)` and `(M, T, S)` move the states by the same numbers.
statsmodels' own module docstring says so, and it follows from the
definition: in `(M, N, M)`, `ŷ = l·s`, `ε = (y − ŷ)/ŷ` and `l' = l(1 + αε) =
l + α(y − ŷ)/s`, which is `(A, N, M)`'s `l' = l + αe/s`. Each of the 15 pairs
works out the same way. The error type changes only the likelihood (so the
learned parameters, section (c)), the fit statistics (`llf`, `aic`) and the
prediction intervals. So there are **15 distinct recursions**: five trends
by three seasons.

**The definition, one step on an even clock, unit weight** (Hyndman et al.,
*Forecasting with Exponential Smoothing*, 2008, ch. 2; the statsmodels
docstring's notation). `⊕` is the trend's operation (`+` for A, `×` for M), `⊙`
the season's, `⊘` the season's inverse; `φ` the damping (1 when undamped);
`p` the row's phase; `B` the damped trend step:

```text
B  = φ·b              (A, Ad)      B = b^φ          (M, Md)
L  = l ⊕ B                                           the level predicted for the row
ŷ  = L ⊙ S[p]                                        pred, before the row's target (hard rule 2)
l' = α·(y ⊘ S[p]) + (1 − α)·L
b' = β*·(l' − l) + (1 − β*)·φ·b                      (A, Ad)
b' = β*·(l' / l) + (1 − β*)·b^φ                      (M, Md)
S'[p] = γ·(y ⊘ L) + (1 − γ)·S[p]                     the other m − 1 values are untouched
```

`β*` is the classical trend gain: `ExponentialSmoothing`'s `smoothing_trend`.
`ETSModel`'s `smoothing_trend` is `β = α·β*` (the scan's "parameter trap"); `γ`
is the same in both (each subtracts or divides the *predicted* level `L`;
`ETSModel`'s smoother computes it as `γ/(1 − α)` on the new level, which is
the same number in another order). With no season, `S ≡ 0` (A) or `1` (M);
with no trend, `b ≡ 0` and `B = 0`.

**On rheon's clock.** `holt` today is this recursion's `(·, A, N)` and
`(·, N, N)` rows with three generalizations, each of which the seasonal forms
keep:

1. *Gains from weighted means over the clock*, not fixed. With `s` the
   (decayed) clock since the target was last observed and `w` the row's
   weight, the level's gain is `g = w/(λ_l^s·W + w)` and `W' = λ_l^s·W + w`,
   `λ_l` the level's decay per clock unit; likewise `h` for the trend on `V`.
   At `w = 1`, `s = 1` and `W` saturated at `1/(1 − λ_l)`, `g = 1 − λ_l = α`.
2. *The step over `s`*: `L = l + φ_s·b` with `φ_s = Σ_{i=1..s} φ^i =
   φ(1 − φ^s)/(1 − φ)` (`= s` undamped, holt's `l + b·s` today); `B` over `s`
   for M trends is `b^{φ_s}`. The trend's update is the weighted mean of the
   trend carried over the step, `φ^s·b` (A) or `b^{φ^s}` (M), and the slope the
   level moved at, `(l' − l)/s` (A) or `(l'/l)^{1/s}` (M).
3. *The seasonal gain from the clock since that phase was last learned*,
   `Δ_p = T − τ[p]` on the decayed clock: `g_s = w/(λ_S^{Δ_p}·U + w)`, with
   `λ_S^P = 1 − γ` for a season of `P` clock units. On an even clock a phase is
   learned once a season, `Δ_p = P`, and at a saturated `U` the gain is `γ`
   exactly. A phase learned twice in one bucket (an uneven clock) gains little
   the second time; one skipped for several seasons gains more. `U` is one
   number shared by the phases, not one each (section (e)).

The seasonal update is `S'[p] = (λ_S^{Δ_p}·U·S[p] + w·(y ⊘ L))·r`, `r = 1/(λ_S^{Δ_p}·U + w)`,
the same weighted-mean shape as the level's. `w = 0` never reaches the model
(the stream scores such a row through `predict`; hard rule 9), and every
reciprocal is taken of a sum with `w ≥ 2⁻¹⁰²²` in it, as `holt` guards today.

**Forecasts at horizons** (the `horizons` interface on `vm/task-559-horizons`):
`f(Δ) = L(s + Δ) ⊙ S[phase(c + Δ)]`, the level and trend carried `s + Δ` clock
units and the season read at the phase the horizon lands in. The integral
`F(Δ)` crosses up to `min(m, Δ/δ)` buckets of a piecewise-constant season, so
it costs `O(min(m, Δ/δ))` per horizon; a damped trend's integral has a closed
form in `φ^{s+Δ}`.

**Left out, with the reason** (the scan's classes): `initialization_method=
"estimated"` (the initial states optimized over every row: not online);
Box-Cox with `λ` estimated (a batch likelihood); simulated intervals and
`simulate` (randomness on the output path); `AutoETS`, BATS, TBATS
(searches over forms); `remove_bias` (statsmodels subtracts the *final* mean
residual from every fitted value). Box-Cox at a given `λ` (a per-row
transform), Theta without deseasonalizing, and Croston, SBA and TSB are
online and found by the scan, but not asked for here; the last three belong
to the models scan.

## (b) How the seasonal component is held

**Values.** `S[0..m)` per target: offsets (A, starting near 0) or factors
(M, starting near 1). `m` is `seasonal_periods`; `P`, `season_length`, is the
season's span in clock units (a duration on a temporal clock); each phase is a
bucket of `δ = P/m` clock units. Without a clock column the clock is the
row, `P = m` and `δ = 1`.

**A row's phase comes from its own clock value, never from a count of rows**:

```text
phase(c) = ⌊m·(c − c₀)/P⌋ mod m
```

with `c` the row's place on the clock as the clock column gives it (the raw
value: not the decayed clock, which `gap_cap` and `session_gap` shorten, and
not the elapsed stamp, which counts a session's gap in place of the time that
passed), and `c₀` the phase origin: `season_origin` if given, else the first
learned row's clock, so that row is phase 0, as in statsmodels. On a temporal
or an integer clock the expression is computed in `i128` integers (the stamps'
lesson, task 175: a phase decided in summed doubles drifts across a bucket
edge); on a float clock in `f64`, where a row exactly on an edge may round to
either side, which the docstring states. Without a clock column, `c` is the
row's place among the rows that count, 1 for the first.

**Why the phase does not depend on the number of rows.** The phase is a
function of the row's own clock value and `c₀` alone. Nothing between two
rows moves it: not a null target (the phase advances with the clock and the
skipped phase is not learned, which is what statsmodels' state-space
`ExponentialSmoothing` does with a `NaN`), not a run of skipped rows, not a
row of weight 0, which never reaches the model and counts in no clock (hard
rule 9). Chunk invariance follows (the phase reads no running count), and so
does save and continue: the state holds `c₀`, never a position. A row-counted
phase would shift every later phase by one at each missing row, which is
right only on a perfectly even series with nothing missing.

**One consequence to document.** Without a clock column a row of weight 0
does not count, so the row after it takes its place and its phase (rule 9:
"as if the row were not there"). A user who means "skip this row but keep its
place in the season" nulls its target instead. Question 3 asks whether that
is the rule wanted.

**Gaps and breaks.** `gap_cap` and `session_gap` shorten the decayed clock,
which ages the level, trend and each phase's `Δ_p`; they do not move phases.
A clock break (a capped gap, a session change) does not empty the seasonal
values, unlike a lag ring: they are keyed by the clock, not by adjacency in
rows, so `clear_lags` leaves them. A step back that restarts the model resets
them with everything else.

**An uneven clock** puts any number of rows in a bucket. Each row predicts
with `S[p]` as the row before left it and learns with the gain of section
(a) item 3, so a second row in the same bucket moves it by little. An empty
bucket is not learned and is aged by its `Δ_p` when next observed.

**Plumbing.** A model sees `d_clock` and, with a window, the decayed stamp.
The raw clock value is new to the trait: a `clock_next(c)` hook beside
`stamp_next`, which the stream calls before each learned row with the row's
exact raw value (integer nanoseconds, the integer, the double, or the counted
row). A model without a season ignores it. The model still reads no column
(hard rule 4).

## (c) Which parameters are fixed, which learned

**Fixed by the spec, never learned:** the form (trend, damped, season, error
type), `m`, `season_length`, `season_origin`, the start (below), and the
initial states under `start = "known"`.

**The smoothing parameters** `α, β*, γ, φ` are given in one of two ways,
refused side by side: per step, `alpha`, `beta`, `gamma`, `phi` (on a row
clock exactly statsmodels' `smoothing_level`, `smoothing_trend`,
`smoothing_seasonal`, `damping_trend`; on a clock column, per clock unit), or
as half-lives in clock units, as `holt` takes them today (`α = 1 − 2^{−1/h}`
per unit), with `seasonal_half_life` and `damping_half_life` beside them.
Fixed is the default and the exact class.

**The start** decides the gains' weights before the first row:

| start | what it does | oracle |
|---|---|---|
| `"warm"` (holt today) | `W`, `V` from 0, so the first gains are 1 and fall to the fixed ones; with a trend the gains are the larger of these and the weighted line fit's | statsmodels' `Holt` once the weights saturate (held so today from row 1,000) |
| `"known"` | the initial level, trend and `m` seasonal values given in the spec; `W`, `V`, `U` held at saturation, so every gain is the fixed one from row 1 | `ETSModel(initialization_method="known")`, `ExponentialSmoothing(initialization_method="known")`, from row 1 |
| `"heuristic"` | statsmodels' `_initialization_heuristic` on the first `k` seasons buffered (`k = 5`; 6 at `m = 2`; the first 10 rows without a season): a centred moving average over a season (`2×m` when `m` is even), the detrended rows' mean per phase normalized, and a least-squares line through the first 10 deseasonalized values. Then the buffered rows are replayed through the recursion once, `O(k·m)`, and every row after is learned as it comes. Each buffered row emits a null prediction with the reason "buffering the start" | `initialization_method="heuristic"`, from row `k·m + 1` (a series of at least `5m` rows, where statsmodels also takes 5 seasons) |

**Learned parameters, two routes** (the scan §4; the study §13, §24, §25):

- *A grid of fixed members with `emit_selected`.* Each member is exact; the
  choice is the argmin of the members' EW out-of-sample error. It exists
  today and needs nothing new; the oracle is each member against
  statsmodels and the argmin against statsmodels' SSE on the same grid.
- *The recursive prediction-error method* (Ljung and Söderström 1983):
  each row, after its prediction, one Gauss-Newton step on `θ = (α, β, γ, φ)`
  with the sensitivities `∂(l, b, S[p])/∂θ` carried by their own exact linear
  recursion (only phase `p`'s row of the seasonal sensitivities moves, so a
  row costs `O(k²)` for `k` parameters, not `O(m·k)`), gain `1/n`, projected
  onto statsmodels' box (`α ∈ [10⁻⁴, 1 − 10⁻⁴]`, `β/α` the same, `γ < 1 − α`,
  `φ ∈ [0.8, 0.98]`). The objective is `½e²` for the A error, the
  concentrated M likelihood for the M error. The study adopted a **500-row
  batch start** fitted by a core-held optimizer (remedy (a)): with it,
  ETS(A,A,N) passed 50 of 50 seeds within 2.5 SE at `α` 0.1, 0.3 and 0.6;
  without it `α` 0.6 failed (6 of 50). That optimizer is `garch`'s on `main`
  (`garch/optim.rs`). **Not measured:** damped, seasonal and multiplicative
  forms, the half-life form, an uneven clock. The initial states are never
  learned online.

## (d) The oracle plan

**Fixed parameters, `start = "known"`, a row clock, nothing missing**: the
predictions and each state against `ETSModel(y, error, trend, damped_trend,
seasonal, seasonal_periods=m, initialization_method="known", initial_level,
initial_trend, initial_seasonal).smooth(params)` and against
`ExponentialSmoothing(...).fit(smoothing_level=α, smoothing_trend=β*,
smoothing_seasonal=γ, damping_trend=φ, optimized=False)`, every one of the 15
recursions, A and M error both (the recursions must agree; a test holds the
two error types' states equal to the bit in rheon). The order of
`initial_seasonal` (statsmodels sorts it so the last column holds `s₋ₘ`) is
pinned by a test on the first prediction before any grid trusts it.

**`start = "heuristic"`**: the start's states against statsmodels' heuristic
on the same rows, then predictions from row `k·m + 1`. **`"warm"`**: as
`holt` is held now, from the saturation row. **A null target** (A season
only): the state-space `ExponentialSmoothing`, which takes `NaN`, on the
seasonal values and on the row after the gap. **Forecasts**:
`ETSResults.forecast(h)` at integer horizons on a row clock. **The grid
route**: each member as above, the selection against statsmodels' SSE.
**The estimator**: against `ETSModel.fit` with known initial states, by the
study's floors (48 of 50 seeds within 2.5 SE on `α, β`, prediction RMS gap
≤ 0.04 error SDs) for (A,A,N); each other form gets its own grid first.

**Tolerance, and why it is not the bit.** Both sides run the same recursion
in a different order (statsmodels forms `γ/(1 − α)` and uses the new level;
its `b^φ` goes through C's `pow`; rheon's gains through the weighted mean and,
on a clock, `exp2` and `powf`). Each row adds a few units of rounding `u`.
The level and the seasonal values are contractions (by `1 − α` per row, `1 − γ`
per visit), so their rounding does not build up past about `1/α` rows; an
additive trend feeds the level every row, and the level's rounding then
walks: the study measured 8.7e-11 absolute over 10,000 rows on wandering
levels for (A,A,N). So: `rtol 1e-12` plus `atol 1e-12·max|y|` for the additive
forms (the scan's proposal, which the study held); for M seasons and M trends,
whose factors multiply the errors, `1e-10` relative as a proposal until
measured. Each test derives its bound from `crate::oracle`'s helpers and
writes the measured headroom beside it (docs/TESTING.md, "The math library's
last bits"), and each grid of 50 seeds is recorded in `oracles.toml`. Bits
hold only a build against itself: one chunk against many, a save continued.

**What statsmodels cannot match**, each held by a Python longhand written
from section (a)'s equations in `tests/test_second_opinion.py` instead ("where
the library is Python's alone, write both"):

- an uneven clock, or a clock column at all: statsmodels indexes by position;
- the row after a gap: rheon's gain grows with the clock since the last
  observation (as `holt`'s does today, by design); statsmodels' fixed gain
  does not;
- several rows in one bucket, and an empty bucket;
- weights other than 1 (statsmodels has none);
- the warm start before saturation;
- a null target on an M form (the state-space form is additive only);
- the heuristic start on an uneven clock or with weights;
- the estimator, which is statistical against `fit`, never exact;
- simulated intervals, `"estimated"` initial states, `use_boxcox=True`.

## (e) The state, against `max_state_bytes`

Per target, today: level, trend, `W`, `V`, the clock since the last
observation and the line fit's three sums, 64 bytes, and a flag. Added:

| part | bytes per target | at `m` = 24 / 168 / 8,760 |
|---|---|---|
| the seasonal values `S` | `8m` | 192 B / 1.3 KB / 70 KB |
| each phase's last learned clock `τ` | `8m` | 192 B / 1.3 KB / 70 KB |
| `U`, `φ`'s and the start's scalars, `c₀` | about 32 | |
| the heuristic start's buffer, freed after the start | `16·k·m` (value and clock), `k = 5` | 1.9 KB / 13 KB / 700 KB |
| the estimator, if shipped: sensitivities, Gauss-Newton matrix, 500-row start buffer | `8(m + 2)k + 8k² + 4,000`, `k ≤ 4` | 5 KB / 9.5 KB / 285 KB |

The per-phase weights are not kept: with a saturated start the weight is the
constant `1/(1 − λ_S^P)` for every phase, and the warm start is not offered
with a season (question 4). Every size is fixed by the spec (`m`, `k`,
targets), so the spec is checked against `max_state_bytes` (256 KB a group
by default, at most 64 MiB) when it is built, as `garch`'s start is, with an
error naming the bytes. An hourly season over a year (`m = 8,760`) fits one
target at 140 KB, but its heuristic start (700 KB) is refused at the
default cap. No window: a nonlinear recursion is not a difference of sums,
and its exact window form re-runs the window every row (the scan §1.4).
Rule 8: `weight_sum` stays `holt`'s shared `w_sum` and each target's weight
its level's, the weight of the rows it learned, never the saturated weight a
gain is formed from.

## (f) Saving and loading

One model, so one tag: the `ModelState::Holt` variant stays (its name is
frozen). The new fields go at the **end** of `Holt` and of `HoltCfg`, after
`horizons` (task 559), with `#[serde(default)]`, so a state written as an
array keeps its places. That makes this branch rebase onto 559 first.

From 1.0 a layout change copies the current fixture sets to `v<N>/`, ships a
loader and regenerates the current sets (hard rule 5). Here the loader is
the defaults: a `holt` state of the schema before loads with no season, no
damping and the warm start, continues to the bit as it was fitted and
re-saves as the new schema's bytes of the same case. `SCHEMA_VERSION` moves
by one (72 to 73 if 559 moves none). The new fixtures freeze every form the
layout can take, each checked non-empty by decoding: an A and an M season,
an Ad and an Md trend, an M trend, the known and the heuristic starts, a
heuristic start halfway through its buffer, a state that has seen several
rows in one bucket and an empty one, and the estimator's state if it ships.
The M trend's `b^φ`, `(l'/l)^{1/s}` and the clock-scaled gains pass through
`powf` and `exp2`: those fields continue to the bit on one OS and compare
across OSes within the stated ulps (CLAUDE.md, the libm rule); every other
field to the bit. The bank fixture is regenerated with the package version,
as at every bump.

## (g) Questions for the Mac

1. **One model or two?** Extend `holt` (same tag, new fields at the end) or
   add an `ets` kind. *Recommend extending `holt`*: with a saturated start
   its gains are the fixed ETS gains exactly, today's forms stay to the bit,
   and the loader is the defaults.
2. **The phase from the row's raw clock value** (section (b)), through a new
   `clock_next` hook. Alternatives: by counted rows (shifts at every missing
   row), or from the elapsed stamp (a session's gap stands in for the night).
   *Recommend the raw clock.*
3. **A zero-weight row without a clock column** does not count, so it shifts
   the phase of every row after it. *Recommend keeping rule 9 as it is and
   documenting "null the target to keep the row's place in the season".*
4. **The starts.** *Recommend `"known"` and `"heuristic"` for the seasonal
   and damped forms, and `"warm"` only for the forms `holt` has today*: no
   oracle and no clean definition of the warm start for a season, whose
   first season confounds the level with the seasonal values.
5. **The error type**: accept `error = "add" | "mul"` now, or only with the
   estimator and the fit statistics, where it changes anything? *Recommend
   with the estimator.*
6. **Learned parameters.** *Recommend part 2 builds the fixed forms and
   relies on the existing `emit_selected` grid; the recursive estimator
   follows after a numpy extension of the study to (A,Ad,N), (A,A,A) at
   `m = 12` and (A,N,M), and with `garch`'s optimizer moved to a module both
   models share.*
7. **A non-positive target on an M form** (statsmodels refuses the whole
   series). *Recommend: predicted, not learned, counted in a stated reason;
   and an update that would make a level, a factor or an M trend 0 or below
   is not taken, the same way.*
8. **Infinite half-lives with a season.** The exact form is a least-squares
   fit on `m + 2` terms, `O(m²)` state. *Recommend refusing it and pointing to
   `ewridge` with seasonal or Fourier features.*
9. **The tolerance.** The plan says bit-level; statsmodels' arithmetic is in
   another order, so bits are impossible against it. *Recommend the derived
   rounding bound (about 1e-12 relative, additive) and bits only for a build
   against itself.*
10. **Size.** Season `16m` bytes a target, the heuristic start's buffer
    `80m` while it fills. *Recommend the default 256 KB cap, so `m = 8,760`
    needs a raised `max_state_bytes` for the heuristic start.*
11. **Renormalizing the seasonal values** (good practice, but not
    statsmodels'). *Recommend not in part 2; an option later if asked.*
12. **Math library.** *Recommend the platform's `powf` and `exp2`, as `holt`
    uses today, with the cross-OS ulps test; the `libm` crate only if the
    estimator and its optimizer ship, for the reason `garch` took it.*

## (h) Part 2 as built (2026-11-24, from `4f688d99`)

The Mac's answers bound part 2; the fixed forms are built in `holt`
(`crates/rheon-core/src/holt/ets.rs`), the stream hands a seasonal `holt` its
row's raw clock (`OnlineModel::clock_next`, `predict_at_into` for a shared
model), and the spec, the Python builder and the docs take the new keys.

**Built.** Seasons `add` and `mul` by trends none, additive and damped
additive (nine of the fifteen recursions); the `known` and `heuristic`
starts, `warm` only for today's forms; `max_state_bytes` (256 KB, ceiling
64 MiB) refused at build with the bytes needed; `not_positive`; schema 73
with schema 72's sets kept.

**The oracle, measured on Linux x86-64.** `ExponentialSmoothing`, known
start: 0 ulps at every row of all nine forms (`holt` follows `model.py`'s
`_predict` operation for operation; held at 5 ulps). Heuristic start: at
most 6 ulps over 8 seeds of 600 rows (held at 16): statsmodels fits the line
by `pinv` and pandas keeps a running sum for the moving average, `holt`
solves the normal equations and sums afresh. `ETSModel`, known start, both
error types: 0-1 ulps without a season, at most 6 with an additive one
(held at 16), since it updates the season from the new level with `γ/(1 −
α)`.

**Departures from the design, each with its reason.**

1. *(a) is wrong for `ETSModel`'s multiplicative season.* Its smoother
   (`_ets_smooth.pyx`) sets `s' = γ*·y/l_t + (1 − γ*)·s`, `γ* = γ/(1 −
   α)`, on the new level; Hyndman's table and `ExponentialSmoothing` divide
   by the level carried to the row, `γ·y/L`. These agree for an additive
   season and not for a multiplicative one (3e-5 to 1e-5 relative on the
   test's series). `holt` is Hyndman's; that form is held to
   `ExponentialSmoothing` alone (`tests/test_second_opinion_ets.py`).
2. *The fixed gains on a clock.* (a) item 1 held `W` at `1/(1 − λ_l)` per
   clock unit, which on a clock column whose rows come every `δ` units
   gives a gain near `1 − λ` per unit, not per row. The gain is instead the
   weighted mean's whose held weight is the one rows at this step saturate
   at, `1/(1 − λ^s)`: `w·a/(λ^s + w·a)`, `a = 1 − λ^s` (Wright's irregular
   smoothing at weight 1), `a` to the bit at `w = 1`. (a) item 3's seasonal
   gain is the same form over `Δ_p`. So a row at the clock of the last one
   teaches nothing (`1 − λ⁰ = 0`).
3. *The initial states stand one step before the first row*: a season's
   bucket `P/m`, or one clock unit without a season, as statsmodels' are on
   a row clock.
4. *The multiplicative trends (`M`, `Md`) are not built.* `b^φ` and
   `(l'/l)^{1/s}` need a power, a platform function not yet in `holt`'s
   state, which question 12's answer bars; the `libm` crate's `pow` would
   give the same bits everywhere, but `docs/TESTING.md` records it declined
   for speed outside `garch`. A question for the user.
5. *`season_origin` is not built*: the origin is the first row of positive
   weight. *The seasonal values are not in `coef`*, which stays `[level,
   trend]`; the Rust accessor `Holt::seasonal` has them.
6. *`not_positive` stands beside a prediction.* The answer asked for a
   named `withheld_reason` on a row predicted and not learned; every other
   reason names a null prediction. It is the eighth reason, last in
   precedence, and every model's `withheld_reason` Enum gains it.
7. *Schema.* The task asked for the pre-1.0 path (regenerate, raise the
   minimums); the library is past 1.0 (2.0.0 holds schema 67), so CLAUDE.md
   rule 5's from-1.0 path is taken: schema 72's sets copied to `v72/`, the
   loader the new fields' defaults, the minimums unchanged. The current
   sets of both crates are schema 72's macOS bytes loaded and re-saved by
   this build (the bytes the Mac's regeneration writes for every case the
   change does not touch), and the four new `holt` cases were written on
   Linux x86-64. The Mac regenerates both crates' sets with
   `PRINT_STATE_FIXTURES=1` before the push.
8. *Grids.* No 50-seed grid is in `oracles.toml`; the essentials hold three
   seeds of each form and the measurements above are from 8.
