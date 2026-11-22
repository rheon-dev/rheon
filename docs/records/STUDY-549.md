<!-- kind: record -->
# GARCH across math libraries, and `bocpd`'s hazard in rows (PLAN 549)

The user, 2026-11-22: "Add to the plan to revisit garch and bopcd. The
garch test seems to be a very degenerate case and I don't understand why
bopcd was working the way it was", and "it's fine to include degenerate
cases but I want to be sure that all our model tests have non degenerate
cases as well". This study changes no library behaviour. It answers PLAN
549 (a) and (b), folds in 548, and lists what 550 needs for `garch`.

## The decisions for the user

1. **Keep the `libm` crate for `garch` (recommended), or drop it.** On
   realistic data the platform's own library still moves `garch`'s outputs
   by far more than a few ulps. It does not need a ridge to do that. The t
   forms' starts take another path in 1 to 4 of 100 series a start size.
   When they reach the same optimum, they still differ at the optimizer's
   stopping tolerance. Dropping the crate would need two optimizer fixes
   (below) and then a cross-OS tolerance of about 10³ ulps on the t forms'
   outputs and state, and a few hundred on EGARCH's. CLAUDE.md's libm rule
   calls that "not tiny", so it would come back to you. The crate costs
   1.4x to 1.8x a call, felt only in the start's fit and in the skewed t's
   and EGARCH's row step (4% to 14%).
2. **Fix the start's soundness test whatever is chosen (recommended).** On
   realistic data every path switch between the libraries was the same
   event. A first fit stopped `NoAscent` on one library and `Converged` on
   the other, at an optimum whose Newton decrement was at most 1.6e-12
   nats. "Unsound" then doubled the start on one side only. Deciding
   soundness by the decrement, below 1e-6 nats say, would have removed all
   25 switches.
3. **PLAN 548's premise does not hold on realistic data.** It holds on
   degenerate data only. At starts of 1,000 and 2,000 rows, every one of
   1,350 start fits reached `arch`'s optimum to 2e-12 nats. At 500 rows,
   4 of 650 converged to a lower local maximum (0.5 to 5 nats). `NoAscent`
   stops, nearly all on 50-row starts, fall short in 58 of 67 cases
   (median 1.3 nats).
   A better optimizer is worth having for the two fixes above and for local
   maxima at 500 rows. It does not make the fit better than `arch` at the
   default start.
4. **A ridge guard is cheap and decisive on degenerate data, and inert on
   realistic data.** A likelihood-ratio statistic against constant variance
   of at least 13.8 (χ²₂ at 0.999) withheld 100% of the no-GARCH series at
   every start length. It withheld 0% of realistic series at 2,000 rows,
   1.8% at 1,000 and 15% at 500 (the persistence-0.90 truths). Its decision
   never differed between the libraries. It is a design question of its own
   (`garch` would wait, as other models wait for rank). It does not answer
   the platform question.
5. **`bocpd`'s `hazard`: rows was the paper's reading.** It was kept
   deliberately for compatibility. No recorded reason covers a numeric
   clock. PLAN 547's oracle "to the bit" cannot hold for `p_change`, which
   has to be restated (part (b)).

## Method

The harness (`scripts/study_549/`) ran each series twice through
`Garch` itself:

- once with `garch::math`'s `ln`, `exp` and `ln_1p` switched back to
  `f64::ln`, `exp` and `ln_1p` (Apple's library, this Mac);
- once on the `libm` crate, as committed.

The `libm` crate stands in for "another platform". Both runs are on one
machine, so nothing else differs. The switch and the harness are a scratch
patch, `scripts/study_549/harness.patch`, never committed to the library.
The patch logs each start fit's stop, iterations, θ, f and gradient.

The data are 50 seeds of 10,000 rows each:

- **Realistic, "hi" truths.** The oracle grid's own processes and seeds
  (`tests/test_second_opinion_garch.py`): `garch` (0.05, 0.08, 0.90),
  `gjr` (0.03, 0.05, 0.10, 0.87), `egarch` β 0.97 and 0.985, the t forms
  with ν = 6, the skewed t with λ = −0.2, and `figarch`. Persistence is
  0.97 to 0.985.
- **Realistic, "lo" truths.** Persistence 0.90: `garch` (0.1, 0.1, 0.8),
  `gjr` (0.1, 0.05, 0.1, 0.8), `egarch` (0, 0.2, −0.1, 0.9), and the t
  forms on them.
- **Degenerate, "iid".** The same innovations with no GARCH effect.
- **Start sizes.** Realistic series start on 50, 500, 1,000 and 2,000
  rows, iid series on 50, 500 and 2,000, and `figarch` on 2,000 only. The
  start is the model's default otherwise: 2 doublings while the cap allows.

That is 3,550 run pairs. "Another path" means the start θ apart by more
than 1e-3, relative. "Flip" means the libraries doubled the start a
different number of times. ulps are distances between the two runs' f64
outputs.

The goldens' own degenerate stream (60 rows of a linear target repeated,
50-row start) reproduces CI's parting on this one machine:

| golden | path, platform / libm | signature apart (relative, ulps) |
|---|---|---|
| `garch` | cap50 / cap50 | 0, the bits |
| `gjr_t` | noasc9 / noasc9 | 5.0e-14, 440 |
| `skewt` | conv50 / cap50 | 3.0e-8, 1.7e8 |
| `egarch` | conv47 / conv49 | 2.0e-6, 1.4e10 |

The observed information there is not positive definite, the
likelihood-ratio statistic against constant variance is about 0, and the
Gauss-Newton matrix's smallest normalized eigenvalue is 1e-6 to 1e-3.

## (a) GARCH

### 1. How far the two libraries part

The realistic sets (hi and lo pooled, 100 series a cell; `egarch_985` and
`figarch` 50). The columns are:

- *bits*: series whose outputs are identical on every row;
- *flip*: doubling decisions that differ;
- *other*: start θ apart by more than 1e-3;
- *apart*: start θ apart by more than 1e-9;
- *last ulps*: the variance at the last row, median and p90;
- *max rel*: the largest relative difference over every row and output;
- *end θ*: the parameters at the end, median ulps.

| form | start | bits | flip | other | apart | last ulps med / p90 | max rel | end θ ulps |
|---|---|---|---|---|---|---|---|---|
| garch | 500 / 1000 / 2000 | 98 / 100 / 98 | 0 | 0 | ≤1 | 0 / 0 | ≤5.8e-9 | 0 |
| gjr | 500 / 1000 / 2000 | 100 / 100 / 100 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| figarch | 2000 | 50 | 0 | 0 | 0 | 0 / 0 | 0 | 0 |
| egarch | 500 | 0 | 0 | 0 | 4 | 2 / 18 | 7.3e-9 | 31 |
| egarch | 1000 | 0 | 0 | 0 | 3 | 2 / 10 | 6.1e-9 | 30 |
| egarch | 2000 | 0 | 0 | 0 | 4 | 2 / 15 | 3.4e-8 | 26 |
| egarch_985 | 2000 | 0 | 0 | 0 | 3 | 3 / 15 | 7.3e-8 | 64 |
| garch_t | 500 | 0 | 4 | 4 | 41 | 4.0e3 / 2.0e7 | 0.20 | 2.2e4 |
| garch_t | 1000 | 0 | 4 | 4 | 32 | 9.0e2 / 5.7e6 | 0.20 | 7.4e3 |
| garch_t | 2000 | 0 | 3 | 3 | 36 | 1.2e4 / 1.3e7 | 0.036 | 1.4e5 |
| garch_skewt | 500 | 0 | 2 | 2 | 33 | 4.1e3 / 1.5e7 | 0.17 | 2.6e4 |
| garch_skewt | 1000 | 0 | 3 | 3 | 39 | 8.3e3 / 6.2e7 | 0.36 | 4.4e4 |
| garch_skewt | 2000 | 0 | 4 | 4 | 43 | 3.9e4 / 2.7e7 | 0.036 | 4.0e5 |
| gjr_t | 500 | 0 | 3 | 3 | 36 | 1.7e3 / 1.8e7 | 0.076 | 2.1e4 |
| gjr_t | 1000 | 0 | 1 | 1 | 38 | 1.7e3 / 6.7e6 | 0.088 | 1.2e4 |
| gjr_t | 2000 | 0 | 1 | 1 | 39 | 3.1e4 / 1.7e7 | 0.071 | 1.6e5 |

At 50-row starts on the same realistic series, the t forms and EGARCH
take another path in 9 to 22 of 100. On the no-GARCH series, they do so in
6 to 20 of 50 at every start length, 2,000 rows included. That is the ridge
the goldens sit on. `garch` and `gjr` under the normal stay bit-identical
there too.

**This confirms the fix worker's report, with its cause.** At the default
2,000-row start on the "hi" truths, 2, 2 and 1 of 50 `garch_t`, `skewt` and
`gjr_t` series took another start. 12 to 16 of 50 ended their first fit at
another iteration, and 30 to 38 of 50 started more than 1e-12 apart.

**Three sources, each measured alone:**

1. **The doubling decision.** Every "other" cell above is a flip, 25 of 25.
   One library's first fit stopped `NoAscent` and the other's `Converged`,
   both after 5 to 12 iterations. The `NoAscent` side was marked unsound
   and doubled its start, so the two fit different rows. The Newton
   decrement `gᵀM⁻¹g/2` at those `NoAscent` stops was at most 1.6e-12
   nats, median 5e-25. They are optima to the rounding: no step gains more
   than f's last bits. On realistic data no flip came from a ridge. A flip
   moves the outputs by up to 0.2 to 0.36, relative. That gap shrinks only
   as the start's weight in the running estimate does, about `n₀/n`: 3e-3
   to 1e-2 at the last row.
2. **Where a converged fit stops.** The optimizer stops once a step moves
   every parameter by at most 1e-8 of its unit. Same optimum or not, the
   last iterate depends on the path, so the libraries' starts part by 2e-11
   to 7e-11, relative (median), about 10⁵ ulps. A difference of that size
   then shrinks over the stream: median 7e-12 over the first 100 rows, 5e-13
   at row 10,000.
3. **The row's step.** Two runs were started from the same bits (the
   `libm` crate's fit), then one switched to the platform's library from
   the first stepped row on:

   | form | last-row ulps, median / p90 / max | max rel, every row and output |
   |---|---|---|
   | garch, gjr, figarch | 0 | 0 |
   | egarch, egarch_985 | 1 to 3 / 7 to 8 / 22 | 1.1e-14 |
   | t forms | 20 to 33 / 60 to 107 / 157 to 347 | 6e-14 to 8e-14 |

   The row's step accumulates a difference slowly and does not amplify
   it. The median difference is 4e-16 over the first 100 rows, 2e-14
   over the rest, and at most 8e-14 in 8,000 rows.

`garch` and `gjr` under the normal, and `figarch`, call `ln` only in the
start's likelihood value, never in its gradient, curvature or filter.
Apple's `ln` and the `libm` crate's do differ: by 1 ulp on 4.2% of
arguments over 10⁷ draws, `exp` on 9.8% and `ln_1p` on 4.6%
(`study549_function_agreement`). But f is read only by the line search's
Armijo comparison. Unless that comparison sits at the rounding, every
iterate is the same bits. It sat there in 7 of 850 realistic normal-form
runs (4 of 650 at starts of 500 or more, up to 5.8e-9 apart), and in 1 of
300 iid ones.

### 2. A minimum start or an identifiability check

Each candidate was read at the start's final fit:

- the observed information, by central differences of the gradient, its
  smallest eigenvalue scaled to a unit diagonal (`obs`);
- the Gauss-Newton or BHHH matrix, scaled the same way (`gn`);
- the likelihood-ratio statistic against the constant-variance model with
  the same innovations (`lr`).

Each row gives the share withheld, how many decisions differ between the
libraries, and, among the series that pass, how many are bit-identical and
how many took another path:

| set, start | guard | withheld | decisions differ | pass: bits / another path |
|---|---|---|---|---|
| iid, 50 / 500 / 2000 | none | 0 | 0 | 99, 100, 100 of 300 / 88, 56, 66 |
| iid, 50 / 500 / 2000 | today's `sound` | 69% / 48% / 38% | 2 / 1 / 1 | 27 / 9; 55 / 3; 71 / 17 |
| iid, 50 / 500 / 2000 | `obs ≥ 1e-3` | 60% / 54% / 69% | 3 / 2 / 1 | 40 / 17; 41 / 9; 16 / 20 |
| iid, 50 / 500 / 2000 | `lr ≥ 6` | 94% / 94% / 93% | 0 | 2 / 6; 4 / 2; 4 / 1 |
| iid, 50 / 500 / 2000 | `lr ≥ 13.8` | 100% | 0 | none pass |
| real, 50 | `lr ≥ 13.8` | 77% | 0 | 39 / 37 of 150 |
| real, 500 | `lr ≥ 13.8` | 15% | 0 | 160 / 14 of 554 |
| real, 1000 | `lr ≥ 13.8` | 1.8% | 0 | 193 / 12 of 638 |
| real, 2000 | `lr ≥ 13.8` | 0 | 0 | 248 / 10 of 700 |
| real, 2000 | `obs ≥ 1e-2` | 4.6% | 0 | 228 / 10 of 668 |

**Findings:**

- The likelihood-ratio guard is the one that tells a series with no GARCH
  effect from a real one. It withholds every iid start, no realistic
  2,000-row start, and the short or low-persistence realistic starts as
  the evidence warrants. Its decision is the same on both libraries,
  since 13.8 is far from the rounding of `2 Δf`.
- The information-matrix guards withhold realistic series too, from 4% at
  2,000 rows (they read correlations among α, β and ω, which are high in
  any persistent GARCH). They let through about half of the iid series,
  and their decision differs between the libraries in a few cases.
- **No guard makes the platform library bit-identical on realistic data,
  nor within a few ulps.** The passing t-form series keep their 10 another
  paths, from the flips, and their 10³ to 10⁷ ulps. The ridge is not what
  parts them there.

**With the platform library, what would be needed.** Fix 1 is soundness
by decrement, which removes the flips. Fix 2 is a Newton run to the
rounding floor, which removes the stop's tolerance. With both, the starts'
θ came within these distances (eight full Newton steps from each library's
own evaluations, the last step 2e-12 of a unit; `study549_polished`):

| form | θ apart, ulps: median / p90 / max | max rel |
|---|---|---|
| garch, gjr | 0 | 0 |
| egarch, egarch_985 | 6 / 31 to 43 / 304 | 4.8e-14 |
| t forms | 245 to 288 / 700 to 990 / 2,132 | 3.8e-13 |

The row's step adds at most about 350 ulps. A cross-OS state and golden
tolerance would then be about **2,500 ulps for the t forms' θ and outputs,
and about 350 for EGARCH's**, to be confirmed on Linux and Windows. That
assumes `ucrtbase.dll` stays within its measured 3 ulps a call. `garch`,
`gjr` and `figarch` under the normal, whose row step calls no `ln` or
`exp`, would hold to the bit only where the platform's `ln` agrees on the
start's rows. Their goldens would need a stated tolerance too. Every number
here is above CLAUDE.md's "few ulps". Under its libm rule, each goes to the
user.

### 3. The start's optimizer against `arch` (PLAN 548)

Each start's final fit θ (the `libm` build) was scored by `arch`'s own
likelihood on the same rows (`am.fix(θ)`). That agreed with the model's f
to 1.2e-10 nats. It was compared with the best of `arch`'s fits from its
default start and from θ. "Short" means `arch`'s best minus ours.

| set | start | fits | short by > 0.01 nats | worst |
|---|---|---|---|---|
| realistic | 2000 | 700 | 0 | 1.8e-12 |
| realistic | 1000 | 650 | 0 | 6.8e-13 |
| realistic | 500 | 650 | 4 | 5.0 (egarch lo), 2.2 (gjr_t lo), 1.5 (garch lo), 0.49 |
| realistic | 50 | 650 | 108 | 12 |
| iid | 50 | 300 | 105 | 11 |

By how the fit stopped, over all realistic starts:

| stop | fits | short by > 0.01 | median short |
|---|---|---|---|
| Converged | 2,567 | 46 | 0 |
| NoAscent | 67 | 58 | 1.3 nats |
| at the cap | 16 | 8 | 0.009 |

**Findings:**

- At the default start the optimizer is not stopping short. The four 500-row
  cases are converged, "sound" fits at a lower local maximum: the grid's
  best point chose the basin.
- The shortfalls PLAN 548 found (gjr_t −75.530 against −75.560) are 50-row,
  no-GARCH starts. There a `NoAscent` stop at a non-optimum is common. On
  EGARCH, some `NoAscent` stops sit on the invertibility guard's edge, with
  a decrement of 0.07 to 12 nats if that guard is ignored.

**What to do:** keep 548, retargeted.

- Decide soundness by the decrement, which is fix 1.
- Run to the rounding floor, which is fix 2 (the constrained Newton
  prototype of 7c03f405).
- Optionally, start from more than the grid's single best point, for the
  500-row local maxima.

Hold it to `arch` on the realistic grids above, where it must not lose. On
50-row starts it should gain, not be held to `arch`. Whichever library
`garch` uses, fix 1 is a correctness fix: a start at its optimum should not
wait for twice the rows because of its last bits.

### 4. `garch`'s tests: degenerate and realistic

| test | data | degenerate? |
|---|---|---|
| `crates/rheon-core/tests/golden.rs`, the six `garch_*` goldens | the golden stream: 60 rows of `1.5x₀ − 0.75x₁ + 0.25 + 0.1u` repeated, weights 0.5 to 1.5, a null; start 50, a cap too small to double; `figarch` 2,000 rows of the same repeat, truncation 20 | **degenerate**: no conditional variance, a period of 60 rows, a nonzero mean, a 50-row start |
| `tests/state_fixtures/garch*.rs` (and `v71/`): `garch`, `garch_started`, `garch_targeted`, `garch_skewt`, `garch_egarch`, `garch_figarch` | `cases/mod.rs`'s `Targets::Linear(2)`: a linear target of features that shift level every 12 rows; start 50; saved at 40 or 300 rows (`figarch` at 2,700 on a 2,000-row start) | **degenerate**, the same way |
| `src/garch/tests.rs` (`the_fit_finds_the_process`, `a_saved_state_goes_on_to_the_bit`, the equivariance, forecast, zero-weight and bound tests) | `gjr_series(0.05, 0.08, 0, 0.9)` and a t₆ GARCH, persistence 0.98; start 500, doubling; `figarch` 2,000 | realistic (`a_start_on_a_bound_doubles_and_refits` is degenerate by design) |
| `tests/test_garch.py` | a GJR at persistence 0.98, `start_rows=100`, random weights | realistic process, short start (weakly identified) |
| `tests/test_second_opinion_garch.py` | the oracle truths, 10,000 and 100,000 rows, start 2,000, 50 seeds, against `arch` | realistic |
| `tests/model_contract.rs` (`garch` kind) | the contract's shared stream | degenerate (no GARCH effect); it checks the contract, not numbers |

**Realistic cases to add beside them** (not added; PLAN 550). Each draws
its innovations without a platform `ln`: the sum of twelve uniforms less
six, or a t as a normal over the square root of a sum of squared normals.
`sqrt` is IEEE-exact, and EGARCH's `exp` comes from the `libm` crate, so
the data are the same bits on every OS.

- **Goldens.** One a form, the oracle's "hi" truth, 2,500 rows, the
  default 2,000-row start. The signature is the variance, z and the
  forecast at rows 2,100, 2,300 and 2,499, then θ. Add a "lo" truth
  (persistence 0.90) for `garch_t`.
- **State fixtures.** `garch_realistic` and `garch_t_realistic`, saved at
  row 2,300 after a 2,000-row start on those series (a few hundred bytes
  of state; the buffer is gone).
- **Oracle.** Extend `test_second_opinion_garch.py`'s grid to starts of
  500 and 1,000 and to the "lo" truths, where 500-row local maxima and
  flips live. Add a test that no realistic start's fit is short of `arch`
  by more than 1e-6 nats at 1,000 and 2,000 rows.
- **Python.** A `test_garch.py` case at the default `start_rows` on a
  realistic series.

## (b) `bocpd`'s `hazard`: why a number counts rows on a numeric clock

**The paper.** Adams and MacKay (2007, arXiv 0710.3742) are discrete in
the observations. Section 2 defines `r_t` as the length of the current run
"at time t", where t indexes the observations `x₁ … x_T`. The gap prior
`P_gap(g)` is "discrete". The hazard (their eq. 5) is
`H(τ) = P_gap(g = τ) / Σ_{t ≥ τ} P_gap(g = t)`. They note: "the hazard
function is constant at H(τ) = 1/λ". The recursion applies `H` once per
observation. Their examples first put the data on a regular grid, so that
an observation is a unit of time:

- the Dow Jones returns are daily, and "the time axis is in business days";
- the coal-mine disasters were "modelled ... as a Poisson process by
  weeks", with `λ_gap = 1000`;
- the well-log data are 4,050 equally spaced measurements.

Nothing in the paper treats irregular spacing. One step a row is the
paper's own reading.

**The history**, from `git log -S hazard` across the rename (the crate was
`online-core`):

- **2026-09-05, `a48b9282`.** ENHANCEMENTS E61 specifies `bocpd` with
  `H = 1/hazard`, "`hazard` a number or a column (time-varying)", and
  `hazard=250.0` by default, the paper's `λ_gap = 250` trading days.
- **2026-09-06, task 55, `47d3b35e`.** Builds it. The config field says
  "Expected rows between changepoints; `H = 1/hazard`". ANSWERS-E54-E64
  (task 55) records "Verified (Adams & MacKay 2007). The recursion, the
  hazard ... (constant `1/λ` for a geometric gap prior)". A numeric clock
  then carried decay only, so there was no clock reading to choose.
- **2026-10-06, PLAN §17.** The parameters "counted in rows, where a clock
  would fit". The table's row reads: Today "the per-row chance of a break",
  `1/hazard`. A clock-aware form "`H(d) = 1 - exp(-d / tau)` ... in clock
  units". The recommendation was "worth building as an option (a duration
  `hazard` on a temporal clock): on irregular rows the per-row hazard says
  breaks follow the row rate". The user: "Follow your reco on all".
- **2026-10-06, task 179, `8d5784fc`.** Built that option. The guide's
  table says a number is "the expected rows between changepoints, on any
  clock". The duration form is refused on "no temporal clock, a
  plain-number clock parameter beside it". Its record says "A number
  `hazard` is byte-identical in outputs and state over five
  configurations".
- **2026-11-22, task 538, `e786463c`.** Made every other number span clock
  units on a numeric clock. SPAN-INVENTORY, "Open, for the coordinator"
  1: "`bocpd`'s `hazard` as a number on a numeric clock is a count of
  rows, where every other `Span` is the clock's units there". Then: "The
  coordinator, 2026-11-22: left as it is for now, taken to the user".
  ZERO-WEIGHT-AUDIT gives the one stated reason: "a number clock cannot
  say 'time between changepoints'". The user answered "Clock units", which
  is PLAN 547.

**Verdict.** Counting rows was a **deliberate** reading, though not
argued for numeric clocks:

- It was the paper's recursion, one step an observation, taken as written
  (task 55).
- It was kept as the number's meaning when a clock form was added, so that
  existing specs stayed byte-identical. §17 scoped the clock form to "a
  duration ... on a temporal clock" (task 179).
- No record weighs a numeric clock's units against rows. It was not an
  oversight in the sense of a bug. When 538 made numbers clock units
  elsewhere, it became an inconsistency, flagged and taken to you.

**What clock units can mean on an irregular clock.** There are two
readings:

- **Once a row, with λ in clock units.** `H = 1/λ` after every row, whatever
  the step. This only renames the unit: the gaps still do not enter. It is
  rows by another name, and not a clock reading.
- **A per-step hazard** `h = 1 − exp(−Δt/λ)` for the step `Δt` into each
  row, applied before the row is read. This is task 179's duration path,
  `hazard_on_clock`, with `τ = λ`. It is the only reading in which the
  clock's spacing changes anything. A gap of `Δt ≫ λ` makes a break near
  certain. A step of 0 (a shared stamp) makes one impossible. A capped gap
  counts as the cap, a session change as its `session_gap`, and a
  zero-weight row counts in no clock.

**What the per-step form does to the outputs** (`scripts/study_549/
bocpd_hazard.py`; 5 seeds × 4,000 rows, `λ = 200`, levels N(0, 3²) between
breaks and N(0, 1) noise):

| clock | breaks arrive | loglik a row, rows / clock | mean \|Δ p_change\| | `run_mode` differs |
|---|---|---|---|---|
| regular, 1 s a row | either | −1.47826 / −1.47826 | 5.0e-3 | 0% |
| irregular, gaps Exp(1 s) | per row | −1.47826 / −1.48020 | 6.6e-3 | 15% of rows |
| irregular | per second | −1.47724 / −1.47625 | 6.5e-3 | 19% |
| sessions of 2,000 rows, 1 h apart | per row | −1.47826 / −1.47996 | 5.3e-3 | 8% |
| sessions | per second | −1.48142 / −1.48010 | 5.1e-3 | 0.1% |

On irregular data, each form wins when the breaks come the way it assumes,
by 1 to 2 thousandths of a nat a row. So the choice is a model of where
breaks come from. Neither form is a correction of the other.

**PLAN 547's oracle needs restating.** On a unit-step clock, the clock form
is not the per-row form with the same number:

- Against `hazard=200` read as rows, `run_mean` moved by up to 0.15 and
  `loglik` by up to 2.5e-3.
- It is the per-row form with `H = 1 − e^{−1/λ}`, which is
  `hazard = 1/(1 − e^{−1/λ})`: `loglik` agreed to 1.6e-14, `run_mean` to
  5.5e-13, and `run_mode` exactly.
- Even then `p_change` differs by H on every row (4.99e-3). The per-row
  form leaves exactly H at `r = 0` after the row. The clock form puts the
  break before the next row, and leaves 0 there.

So "the span written in rows on a unit clock, to the bit" holds for no
output under the same number. With the matched H, it holds for `run_mode`
only (and `run_mean` and `loglik` to rounding). `p_change` changes meaning
by H.

547 should declare this in its `Moves:` line. Its oracle should be:

- the clock form on a unit clock against the per-row form at
  `hazard = 1/(1 − e^{−1/λ})`, `loglik`, `run_mode` and `run_mean`
  to rounding, `p_change` less H;
- plus task 179's enumeration oracle (`clock_oracle`) on irregular steps.

## Reproduce

```
uv run python scripts/study_549/gen.py          # 3,550 run pairs' data under .cache/study549
git apply scripts/study_549/harness.patch       # the scratch switch and harness (never commit)
cargo test -p rheon-core --release --lib garch::study549 -- --ignored --nocapture
uv run python scripts/study_549/analyze.py q1   # also q1b, growth, q2, q3
uv run python scripts/study_549/arch_q3.py      # the arch comparison (about a minute, 4 processes)
uv run python scripts/study_549/bocpd_hazard.py
git apply -R scripts/study_549/harness.patch
```

The harness's tests are:

- `study549_runs`, the 3,550 pairs, 82 s on 6 threads;
- `study549_goldens`;
- `study549_noascent`, the decrement at each `NoAscent` stop;
- `study549_rows_only`, the row's step alone;
- `study549_polished`, the optima at the rounding floor;
- `study549_function_agreement`.

Machine: Apple M4 Pro, macOS 15.7.3, load 5 to 15 during the runs.
