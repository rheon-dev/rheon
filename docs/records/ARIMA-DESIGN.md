<!-- kind: record -->
# ARIMA and SARIMA online: design

Design record for PLAN 487. Part 1 set out the model's equations, its
state, how it keeps hard rules 2, 8 and 9, and eight questions for the
review. Part 2 (this revision) carries the coordinator's answers to those
questions into the design and builds the first piece: the fixed-parameter
filter in `rheon-core`, held to statsmodels' `SARIMAX.filter`. Part 3
builds the estimator: the start, its gate and the per-row update.

The measurements behind it are `docs/records/STUDY-487-488.md` on branch
`vm/study-487` ("the study", §3 to §6 and §10 to §11), which extends the
ARMA(1,1) estimator of `ORACLE-STUDY-485.md` (branch `vm/study-485`), and
the library scan `vm-tools/research/485-TIME-SERIES-SCAN.md` (branch
`vm-tools/oracle-gap-scan`, §1 and §3, "the scan"). The start and its
buffer follow `garch` (`docs/records/GARCH-DESIGN.md`).

**Two changes from part 1.** The start is gated on a condition, not on a
count of rows or a weight threshold: the default `min_weight` stays the
library's default (part 1 had set it to `d + sD + max(500, 3s)`), and a
fit is reported once a start has been fitted and accepted ("The start,
and its gate"). And a row of weight 0, or a null target, keeps its slot
as a missing observation, where part 1 dropped it (`ar`'s rule), so a
season stays aligned.

## The model

statsmodels' signs and orders, so a spec reads as `SARIMAX`'s:

```text
phi(B) Phi(B^s) (w_t - xw_t' beta) = theta(B) Theta(B^s) e_t,   e_t ~ (0, sigma2)
w_t  = (1 - B)^d (1 - B^s)^D y_t          xw_t = (1 - B)^d (1 - B^s)^D x_t
phi(B)   = 1 - phi_1 B - ... - phi_p B^p           Phi(B^s)   = 1 - Phi_1 B^s - ... - Phi_P B^(sP)
theta(B) = 1 + theta_1 B + ... + theta_q B^q       Theta(B^s) = 1 + Theta_1 B^s + ... + Theta_Q B^(sQ)
A(B) = phi(B) Phi(B^s),   C(B) = theta(B) Theta(B^s),   delta(B) = (1 - B)^d (1 - B^s)^D
```

The spec names `order = (p, d, q)` and `seasonal_order = (P, D, Q, s)`;
ARMA is `d = D = 0` with no season, ARIMA is no season. The regressors are
the row's features, shared by every target, entering as `SARIMAX`'s
regression with SARIMA errors under simple differencing: each regressor
differenced as the target is, `xw_t' beta` in the observation. Each target
is its own series, with its own rings, parameters and start.

`B` steps one **slot**: `spacing` clock units, 1 on a clock that counts
rows. A slot with no learned row (a row of weight 0, a null target, a slot
the clock stepped over) is a missing observation, not a dropped row
(question 3).

## The fixed-parameter filter (part 2, built)

With `params` given, each target runs the exact Kalman filter of the
state-space form `SARIMAX` builds with `simple_differencing=True`, built
inside the model, not through `kalman` (question 4), whose transition is
diagonal where this one is a companion matrix.

- **The differencing**: statsmodels' `diff`, `D` seasonal differences of
  period `s` and then `d` ordinary ones, each a subtraction of two values,
  over a ring of the last `d + sD` slots. The first `d + sD` slots of a
  series have no differenced value and report none, as simple differencing
  drops them.
- **The state** (Harvey's form, as `SARIMAX`'s): `r = max(p + sP, q + sQ +
  1)` states, `T`'s first column `-A_1, ..., -A_r` with ones on its
  superdiagonal, `R = (1, C_1, ..., C_{r-1})'`, `Z = (1, 0, ..., 0)`, no
  measurement noise.
- **The start**: `a_1 = 0` and `P_1` solving `P = T P T' + R sigma2 R'`
  (`SARIMAX`'s `stationary` initialization), by Smith's doubling, `O(r^3)`
  a step for about `log2` of the slowest root's memory in steps, run when
  the model is built and at each restart.
- **A slot** `t`, with `d_t = xw_t' beta`:

  ```text
  ŵ_t = d_t + a_0,   ŷ_t = ŵ_t - L_t       L_t: w_t's differencing with y_t at 0
  F = P_00,  v = w_t - ŵ_t                  only where w_t is observed
  a <- a + P_.0 v / F,   P <- P - P_.0 P_0. / F
  a <- T a,              P <- T P T' + R sigma2 R',  then (P + P') / 2
  ```

  `ŷ_t` is the row's `pred`, formed before `y_t` is read (hard rule 2).
  `w_t` is observed when the slot and every slot its differencing reads
  were observed; otherwise the update is skipped and the filter predicts
  through the slot, as `SARIMAX` does with a `NaN`. The ring keeps a
  missing slot's prediction `ŷ_t` in its place, so the rows after it still
  predict. A skipped slot has no features, and a row of weight 0 keeps
  none of its own (the contract every model keeps: nothing of a zero-weight
  row's values enters the state), so with regressors their value in the
  ring is NaN and the rows whose differencing reads it report none.
- **`T P T'` costs `O(r^2)`**, not `O(r^3)`: `T` is a companion matrix, so
  each entry of each product is two terms, formed in place with one saved
  row. The state holds `P`'s upper triangle (it is symmetric after each
  step); a row steps it in full in a working matrix of `r^2` doubles that
  every target of the group shares and the state does not hold.
- **Stationarity**: each autoregressive factor is checked when the model
  is built by the step-down (Schur-Cohn) recursion, whose partial
  autocorrelations lie inside `(-1, 1)` exactly when the factor's roots lie
  outside the unit circle (the inverse of Monahan's transform; arithmetic
  only). A factor outside is refused: it has no stationary start. The
  moving-average factors need no check for the filter, which runs at any
  `theta`, as `SARIMAX.filter` does.

**Measured** against `SARIMAX(order, seasonal_order, trend="n",
simple_differencing=True).filter(params)` with `ssm.tolerance = 0`
(`tests/test_sarima_filter.py`): every row from the first, in units in the
last place of the largest `|w|`; the bound is the study's 8 ulps for
`s <= 24`. The numbers are in "Part 2's results" at the end.

## The estimator (part 3, designed)

### The recursion

Write `beta` for the stacked parameters `(phi, theta, Phi, Theta)`, `n` of
them. After the start, each learned row steps them by recursive
prediction error (Gauss-Newton; Ljung and Söderström 1983), with the row's
weight `w` and the fit's weight `V`:

```text
V'     = V + w,          a = w / V'
R'     = R + a (psi psi' - R)
beta'  = Proj(beta + a R'^{-1} psi e)
```

`e` is the a-priori error and `psi = -de/dbeta` comes from four filtered
signals at the current parameters (`x / C(B)` is the recursion `xf_t =
x_t - sum C_j xf_{t-j}`): `u = Phi(B^s) w / C(B)` for `phi`, `v = phi(B)
w / C(B)` for `Phi`, `r = Theta(B^s) e / C(B)` for `theta`, `z = theta(B)
e / C(B)` for `Theta`. The study held these to a central difference of the
conditional sum of squares to 2.0e-11 and 2.3e-11 relative.

**`Proj` for every order** (question 2): each polynomial is carried in
the unconstrained parameters `SARIMAX` optimizes, through its stationarity
and invertibility transforms (Monahan 1984, through Jones's 1980 map of the
partial autocorrelations; the same transform on `-theta` for the moving
averages), with the chain rule through the gradient, and each held to
`enforce_stationarity` and `enforce_invertibility` as `SARIMAX` holds it:
switched off, the parameter steps unconstrained. Part 1's clip to
`[-0.99, 0.99]`, right only for polynomials of order one, goes.

**`sigma2`** (question 7): `SARIMAX`'s `sigma2` is a parameter of the
maximum-likelihood fit (with `concentrate_scale=False`, its default), not
the residuals' mean square. Part 3 either estimates it as that parameter
of the same likelihood, or states the difference and tests it (the
study's `s2` averaged the start's zero-start residuals and read up to 1.5
standard errors high at `s = 168`).

### The start, and its gate

No count of rows and no weight threshold opens the fit. The gate is a
condition on the buffer and on the fit itself:

1. **Buffer.** A target's learned rows are buffered as `(w_t, weight)`
   pairs, with their regressors and each slot's place, so a missing slot
   stays a gap in the likelihood, under `max_state_bytes`: a warning at
   half, a refusal at the cap, naming the row and the bytes. Each row
   reports a null `pred` with a `withheld_reason` naming the stage.
2. **Structural minimum** (question 1). Nothing is fitted until Kish's
   count of the buffer, `(sum w)^2 / sum w^2`, passes the parameter count
   plus `d + sD`: below it the likelihood has more unknowns than
   observations. No multiple of the parameter count is added; whether the
   rows identify the fit is the next step's question, answered by the fit.
   Reason: "buffering: Kish's count not past the parameters and the
   differencing".
3. **Fit and accept.** Past the minimum the start is fitted (the buffer's
   likelihood, by the projected Newton `garch` holds, from the start grid)
   and accepted only when all three hold:
   - it **converged**: its Newton decrement below a stated tolerance at
     Kish's count;
   - it sits on **no bound**: no partial autocorrelation within a stated
     distance of 1 where the transforms are on;
   - its **information matrix is conditioned**: every standard error
     finite and below a stated bound in the parameter's own units, the
     bound set when part 3 is built from the study's spread of standard
     errors at its 500-row point.

   Otherwise the target keeps buffering with the reason that failed
   ("start not converged", "start on a bound", "start not identified:
   standard errors past the bound") and the fit is tried again as the
   weight grows, each time Kish's count has grown by a stated factor, so a
   long wait costs a bounded number of fits.
4. **After acceptance** the buffer is replayed at the fitted `beta`
   through the filter and the recursion's rings, and dropped. The first
   row reported is the first predicted by an accepted fit.

`min_weight` keeps the library's default and its meaning (hard rule 8): a
floor on the weight before a row, checked against the target's own
weight, on top of the gate, never a stand-in for it.

**Both sides of the gate are tested**: a target withheld below the
structural minimum and through a fit that fails each condition (a buffer
too short to converge, one built to put the fit on a bound, one with a
flat likelihood, so its standard errors are past the bound), each with its
reason; and one accepted. The oracle test shows acceptance near the
study's 500-row point: on the study's specs at unit weights the gate opens
within a stated band around 500 differenced rows, and the accepted fit is
within 2 of `SARIMAX`'s standard errors there.

The study's plain start (zero parameters, steps with no buffer) is not
offered: it passes at `s = 7` and `24` but fails at `s = 168` (0 and 1 of
50 seeds), its slow approach measured in seasons (study §6.5).

### Decay

None in parts 2 and 3 (question 5): `n_eff` is the plain sum of weights.
Decay, a weighted likelihood, comes later, with a hand-written weighted
log-likelihood that scipy optimizes as its oracle, or is recorded as
infeasible with the reason.

## Rows, weights and the clock

- **Out of sample** (hard rule 2): `ŷ_t` reads the state and the rings
  only, before `y_t`. The core's test changes row 40 by 100, and no
  prediction up to and including it moves.
- **`n_eff`** (hard rule 8): the weight of every row before this one, with
  no decay, `n_eff' = n_eff + w`. Each target's own weight counts the rows
  that observed it, and its `min_weight` is checked against that
  (`target_n_eff_into`). With fixed parameters a row's weight enters these
  sums only: the filter conditions on an observed value whatever it
  weighs. In part 3 the weight also scales the row's term of the
  likelihood and the gain.
- **A row of weight 0** (hard rule 9, question 3) advances the clock and
  learns nothing: its slot is a missing observation, predicted through.
  Its `pred` is reported under the same gate. A zero-weight first row
  leaves every weight at 0; the filter divides only by `F = P_00`, which
  the stationary start and every step after it keep at least `sigma2 > 0`,
  and only on an observed slot.
- **A null target**: a missing observation the filter predicts through
  (question 3), as `SARIMAX` reads a `NaN`, so seasonal positions stay
  aligned.
- **The clock** (question 6). A step of `n` slots (`d_clock / spacing`,
  rounded; a step under half a slot is the next slot) leaves the `n - 1`
  slots between as missing observations. A gap past `gap_cap`, or a
  session change, is a break (`clear_lags`): each series starts again, its
  ring empty and its state at the stationary start, and in part 3 the
  start is fitted again from a new buffer; the weights are kept. A step of
  more than 4,096 slots is a break too, since predicting through costs one
  covariance step a slot. A step back restarts as `restart_after_step_back`
  says, the stream's rule, before the model sees the row.
- **Chunk invariance** (hard rule 3): every number is a function of the
  rows in order and their clock steps.
- **No window form.** A recursive estimator is not a difference of running
  sums, so `window_size` is refused (scan §1, finding 4).

## State

Per target, for the fixed-parameter filter: the ring (`d + sD` values and
flags), `a` (`r`), `P`'s upper triangle (`r (r + 1) / 2`) and the weight;
per regressor, a ring of `d + sD`. msgpack writes 9 bytes a double, and
`Sarima::state_bytes` bounds the file, checked against `max_state_bytes`
when the model is built. With `r = s + 2` for the first three forms below
and `r = s + 1` for the fourth:

| form | doubles a target (filter) | `s = 7` | `s = 24` | `s = 168` | `s = 1440` |
|---|---|---|---|---|---|
| `(0,1,1)(0,1,1,s)` airline | `r(r+1)/2 + r + s + 1` | 62 | 404 | 14,874 | 1,042,566 |
| `(1,1,1)(1,1,1,s)` | the same | 62 | 404 | 14,874 | 1,042,566 |
| `(1,0,1)(1,0,1,s)` | `r(r+1)/2 + r` | 54 | 377 | 14,705 | 1,041,124 |
| `(1,1,1)(1,1,0,s)` | `r(r+1)/2 + r + s + 1` | 52 | 375 | 14,704 | 1,041,125 |
| part 3's conditional recursion, `(1,1,1)(1,1,1,s)` (study §6.4) | `7 s + 27` | 76 | 195 | 1,203 | 10,107 |

At 9 bytes a double the filter's state fits the default `max_state_bytes`
of 256 KiB for one target up to about `s = 239` (`s = 168`: about 134 KB);
past it the user raises the cap in the spec, up to its ceiling of 64 MiB
(`s = 1440`: 9.4 MB a target). Beside the state, the working matrix holds
`8 r^2` bytes a group (231 KB at `s = 168`). A row costs `O(r^2)` a
target, flat in the number of rows. The filter's state is quadratic in
`s` where the recursion's is linear: for a season past a few hundred the
fixed-parameter form costs the user a raised cap, and part 3 measures
whether the estimator needs the filter's covariance at all after its start
or can run on the recursion alone.

The estimator of part 3 adds, per target, the recursion's rings (`u`,
`v`, `r`, `z`, each at most `max(p + sP, q + sQ)`), `R`'s `n (n + 1) / 2`
entries and its scalars, and until the start is accepted the buffer, 16
bytes a row (the differenced value and its weight) and its regressors.
The buffer is held under `max_state_bytes` as it runs.

The new `ModelState` variant `Sarima` lands with part 2's filter, at the
current schema with no bump, as `ar`'s did: a variant added is no change to
any layout a file already holds. Its fixtures freeze each form its state
takes: inside the differencing ring's fill (`sarima_filling`) and past it,
with slots predicted through, a null target, zero weights and clock steps
of 9 (`sarima`). Part 3's fields change `Sarima`'s layout from 1.0 on
(hard rule 5): the current fixture sets are copied to `v<N>/`, a loader
ships, and the sets are regenerated.

**Platforms.** The filter, its start and the step-down check are IEEE
arithmetic in a fixed order with no `ln`, `exp` or `powf`, so every field
is the same bits on every platform; no libm result sits in the state.

## Where each piece lives

- `rheon-core`: `sarima.rs` (the model, its settings, `OnlineModel`),
  `sarima/series.rs` (one target's ring and filter), `sarima/poly.rs` (the
  polynomials, the differencing, the state-space matrices, the stationary
  start, the step-down test), `sarima/tests.rs`. In part 3, one file for
  the recursion and one for the start, whose optimizer is `garch`'s
  projected Newton lifted into a module both share, not a second copy.
- `rheon-py`: `_rheon.sarima_filter`, the core's filter over one series
  with no clock, weight or group, which the comparison with `SARIMAX`
  reads.
- Later: the bank entry, the spec, the TOML config; `docs/MODELS.md` and
  each place `docs/EXTENDING.md` lists. Outputs per row: `pred`,
  `weight_sum`, `withheld_reason`; `coef()` reads the parameters by name
  (`phi_1`, ..., `sigma2`).

## Oracles

- **Fixed parameters** (exact; part 2): `SARIMAX(order, seasonal_order,
  trend="n", simple_differencing=True).filter(params)` with
  `model.ssm.tolerance = 0`, every row's `ŵ_t` within 8 ulps of the
  largest `|w|` from the first row, `s <= 24` in the essentials and
  `s = 168` in the extended tier (question 8); a missing value predicted
  through as `SARIMAX` predicts through a `NaN`; regressors through
  `exog=`; and the default filter (its steady-state switch on) within
  `5e-9` of the largest `|w|`. The fifty-seed grid of three specs is an
  `oracle_grid` test recorded in the oracle ledger (`oracles.toml`, group
  `sarima_filter.filter_grid`), its cost noted beside the group.
- **In Rust**, which `cargo mutants` sees: the filter against the Gaussian
  conditional expectation `gamma' Gamma^{-1} w` over the observed slots,
  the autocovariances from the MA(inf) weights and the solve by faer's LU
  (`crate::oracle`), through runs of missing slots; the stationary start
  against the Lyapunov equation and `gamma_0`; the step-down test against
  AR(2)'s triangle; the differencing, and the prediction put back on the
  series; the out-of-sample check; a zero weight, a null and a skipped slot
  alike; a zero-weight first row; `min_weight` on the target's own weight;
  a break and a long gap as a fresh start; `predict` as the step; scale
  equivariance; a save at every row continuing to the bit; the state's
  bound and cap; the refusals. `model_contract.rs` probes it with every
  model, and `state_fixtures.rs` freezes it.
- **The estimator** (statistical; part 3): `SARIMAX(...).fit(method=
  "lbfgs", cov_type="opg")` on the same rows, every parameter at the last
  row within 2 of its standard errors; floor 48 of 50 seeds at 10,000
  rows. At `s = 168` the oracle's memory-light path (`fit(low_memory=
  True, cov_type="none")`, then `filter(..., cov_type="opg")`). A `SARIMAX`
  fit at `s = 168` takes 106 to 183 s (the study's 100 fits at `s = 168`,
  with the `s = 24` grids, ran 110 minutes on four workers), so `s = 168`
  is in the extended tier and its grid in the ledger with its cost
  (question 8).

## Left out of parts 2 and 3

Forecasts past one slot; the information criteria and roots at emit;
decay (above); the method-of-moments estimators (Yule-Walker,
innovations), which the scan finds exact from kept autocovariances and
which belong beside `ar`; sharing the `r` and `z` rings (one can be
derived from the other; not tried); the exact diffuse treatment of the
differencing (`simple_differencing=False`), which keeps what a missing
value's neighbours say about it (with `d = 1`, `y_{t+1} - y_{t-1}` when
`y_t` is missing) at the cost of `d + sD` more states.

## The review's questions, answered

1. **The start's floor**: Kish's count against the structural minimum,
   the parameter count plus `d + sD`, with no `10 n` multiple; then the fit
   must converge, sit on no bound and have a conditioned information
   matrix. The default `min_weight` is the library's.
2. **Orders above one**: `SARIMAX`'s stationarity and invertibility
   transforms (Monahan, Jones) for every order, held to
   `enforce_stationarity` and `enforce_invertibility`.
3. **A zero-weight row or a null**: a zero-weight row advances the clock
   and learns nothing (rule 9); a null target is a missing observation the
   filter predicts through, as `SARIMAX` treats a `NaN`. Both keep the
   slot.
4. **The exact form for fixed parameters**: the state-space filter, built
   inside this model, held to `SARIMAX.filter`, not through `kalman`.
5. **Decay**: none now; later a weighted likelihood with scipy as oracle,
   or recorded as infeasible with the reason.
6. **The clock**: a step back restarts as `restart_after_step_back` says;
   a gap past `gap_cap` restarts the start; shorter gaps are missing slots.
7. **`sigma2`**: `SARIMAX`'s is a parameter of the likelihood; part 3
   matches its definition, or states and tests the difference.
8. **The grid's cost**: `s = 168` in the extended tier, the essentials at
   `s <= 24`; the grid's cost in the oracle ledger.

## Part 2's results

Measured 2026-11-24 on a Linux x86-64 VM, statsmodels 0.15.0. The
comparison in `tests/test_sarima_filter.py` runs through
`_rheon.sarima_filter`; these numbers were taken with the same core filter
driven by a scratch Rust harness on the same inputs, bit for bit (the
session's disk could not hold a wheel build, so the Python tests
themselves have not run yet; see "Not done" below).

**One seed, 2,000 rows**, the gap in ulps of the largest `|w|`, over every
row and past the first 500 differenced rows:

| spec | every row | past 500 |
|---|---|---|
| `(0,1,1)(0,1,1,7)` airline | 0 | 0 |
| `(1,1,1)(1,1,1,24)` | 2.1 | 0 |
| `(1,0,1)(1,0,1,24)` | 97 (1.3e-14 relative) | 1 |
| `(1,1,1)(1,1,0,24)` | 3.6 | 1 |
| `(2,1,2)`, no season | 0 | 0 |
| `(0,1,1)(0,1,2,12)` | 1.1 | 0 |
| `(1,1,1)(0,1,1,7)`, two regressors | 1 | 1 |
| `(1,1,1)(1,1,1,24)`, 57 values missing, a run of 50 among them | 2 | 0 |
| `(1,1,1)(1,1,1,168)` | 6.2 (3.5 over the first 50) | 1.7 |

`ŷ_t` against `ŵ_t + y_t - w_t` (`SARIMAX`'s `ŵ_t` put back): within 0.5
ulps of the largest `|y|` wherever the series is differenced. With
statsmodels' steady-state switch on (its default), the gap is 1.3e-10 of
the largest `|w|`, under the bound of 5e-9.

**The early rows of `(1,0,1)(1,0,1,24)`** are statsmodels' start, not the
filter: its stationary covariance (the bilinear transform and scipy's
Lyapunov solver) misses the Lyapunov equation by 1.6e-14 of its scale and
`gamma_0` from the MA(inf) weights by 5.4e-14, where the core's doubling
is within 2e-15 of both (`sarima/tests.rs`). The gap fades as the filter
forgets its start: 1 ulp by row 500. The tests therefore hold every row
within `1e-13` of the largest `|w|` and the rows past 500 within 8 ulps,
the study's bound for `s <= 24`.

**Settled against the exact filter** (2026-11-24, part 3's session). A
scratch reference in mpmath at 60 digits, sharing no arithmetic with
either filter (the stationary covariance as its defining series `sigma2
sum_j (T^j R)(T^j R)'`, 4,969 terms, its Lyapunov residual 8e-61 of its
scale; the Kalman filter in the same precision), on the grid's fifty seeds
of `(1,0,1)(1,0,1,24)`, 2,000 rows each, in ulps of the largest `|w|`:

| | every row: worst (median) | first 50 rows | past row 500 |
|---|---|---|---|
| statsmodels' `SARIMAX.filter` from the exact | 99.9 (96.5) | 99.9 | 2.4 |
| the core's filter from the exact | 3.5 (2.5) | 3.5 | 2.6 |
| the two from each other | 100 (97) | 100 | 2 |

statsmodels' stationary covariance is 5.4e-14 of its scale from the
exact one on every seed (the spec fixes it). So the core is the exact one
and statsmodels' start is off. `tests/test_sarima_filter.py` now holds the
core within 4 ulps of the exact filter over the first 200 rows of seed 1
(`test_the_first_rows_are_the_exact_filter`, extended: about 7 s), with
mpmath declared in the dev group by name; the comparison with statsmodels
keeps its bounds (`1e-13` from the first row, 8 ulps past row 500).

**The grid**, fifty seeds of each of three specs, 2,000 rows: past row 500
the worst seed at 0 (airline), 0.125 (`(1,1,1)(1,1,1,24)`) and 2 ulps
(`(1,0,1)(1,0,1,24)`); over every row 1, 4.5 and 100 ulps, at most 1.32e-14
of the largest `|w|`. It costs 62 s with its simulations (SARIMAX 0.1 to
0.8 s a filter at 26 states), recorded beside its group in `oracles.toml`.

**Cost at `s = 168`**: `SARIMAX.filter` took 492 s for 2,000 rows (its
products are dense, `O(r^3)` a row at 170 states); the core's filter took
3.9 s in a debug build. The extended test runs 600 rows.

### Not done in part 2, and why

- **The estimator** (the start, its gate and the per-row update) is part
  3: it needs the transforms of question 2, the likelihood's `sigma2`
  (question 7) and the gate's three conditions, each measured, and its own
  fixtures as a layout change. Nothing of it is in the code.
- **The bank, the spec, the CLI and the docs** for `sarima` come with the
  estimator, as `ar`'s came after its core.
- **The Python tests and the hash of the ledger's group** have not run in
  this session: the disk was at its floor of 15 GB free, so no wheel was
  built. `uv run python scripts/oracles.py rerun sarima_filter.filter_grid`
  runs the grid and writes the group's hash, and CI's oracle workflow
  fills its numbers.

## Part 3: the estimator, built

Built 2026-11-24 in `rheon-core` (`sarima/est.rs`, `sarima/transform.rs`,
`sarima/est_tests.rs`), behind `SarimaCfg::estimate` (`params` then
empty). Where it departs from "The estimator (part 3, designed)" above,
and why:

- **The start's likelihood is the conditional one**, the zero pre-sample
  recursion the steps themselves run, `sigma2` concentrated out
  (`-W/2 ln(S/W)`), not the exact (Kalman) likelihood. It is the study's
  batch start (§6.1), its gradient is cheap, and the per-row step after it
  minimizes the same criterion. The gradient is exact (forward-mode
  derivatives of the recursion, held to a central difference with missing
  slots and an unknown regressor); with no slot missing it equals the
  filtered signals' `psi` (held too).
- **After the start the prediction is the conditional recursion's**, not
  the Kalman filter's: no covariance is kept, so the state is linear in
  `s` (the table below). The fixed-parameter form keeps the filter.
- **`sigma2`** is the conditional likelihood's (the innovations' mean
  square), stepped as `sigma2 + a (e^2 - sigma2)` (question 7's "state the
  difference"): it is not `SARIMAX`'s exact-likelihood parameter, and the
  two differ by the start's terms, which fade as `1/n`. Measured against
  `SARIMAX.fit` at 10,000 rows, `sigma2` is among the parameters within
  0.4 of its standard errors in the table below.
- **The gate's order**: the bound first (a fit drifting to a bound has no
  optimum inside the region, so it would also fail to converge; the bound
  is the more telling reason), then convergence (Newton decrement at most
  `1e-6` at Kish's count, `garch`'s cut), then the standard errors.
- **The standard errors' bound is 0.1** in standardized units (a
  coefficient as it is, a regressor's coefficient times its regressor's
  standard deviation over the innovations', `sigma2` relative to itself),
  from the conditional information `sum w psi psi' / sigma2` at Kish's
  count. Chosen by measurement, not from a count of rows: five specs,
  eight seeds each, 10,000 rows, the worst coefficient's gap at the last
  row from `SARIMAX.fit(method="lbfgs", cov_type="opg")` in its standard
  errors, and the rows the gate opened at (scratch harness over the core,
  2026-11-24):

  | bound | `(0,1,1)(0,1,1,7)` | `(1,0,1)(1,0,1,24)` | `(1,1,1)(1,1,0,24)` | `(1,1,1)(1,1,1,24)` | `(1,0,1)` |
  |---|---|---|---|---|---|
  | 0.1 | 148-323, 0.23 | 548-822, 0.28 | 255-370, 0.24 | 6,240-9,348, 0.38 | 243-548, 0.20 |
  | 0.15 | 70-148, 0.26 | 243-548, 0.25 | 127-178, 1.24 | 1,866-6,240, **4.48** | 108-243, 0.29 |
  | 0.2 | 49-101, 0.25 | 162-365, 0.74 | 70-127, 1.17 | 267-4,168, **8.16** | 48-108, 0.51 |
  | 0.3 | 26-49, 0.33 | 108-243, **2.57** | 70, 0.27 | 267-2,787, **8.16** | 32-48, 0.96 |

  At 0.1 all forty land within 0.4 standard errors. Looser bounds open
  earlier but start some series from fits the per-row step does not
  recover from in 10,000 rows (the full seasonal spec, whose seasonal
  factors `1 - 0.4 B^24` and `1 - 0.5 B^24` nearly cancel, at 0.15 and
  above). The price is that spec's wait: 6,240 to 9,348 rows, a buffer of
  up to 168 KB a target without regressors, under the default
  `max_state_bytes` of 256 KiB for one target but past it for two, where
  the slot that would pass the cap is refused. The other specs open by
  822 rows. A bound of 0.5 (the 95% interval as wide as `(-1, 1)`) was
  tried first: it opened at 6 to 26 rows on the simple specs and ended up
  to 8 standard errors off on the full one.
- **Retries** at each growth of Kish's count by 1.5: the openings above
  fall on that grid (148, 218, 323, 548, 822, ...).
- **A row of weight 0** keeps none of its regressors in the rings or the
  buffer (the contract every model keeps), while its own prediction reads
  them.
- **The model contract's ready count** for `sarima` is 200 of its probe's
  400 rows, not 300: an estimated start needs Kish's count near 100
  before its standard errors pass 0.1.

**State** (msgpack, 9 bytes a double), per target past its start: the
differencing ring and the run, `9 (n + 2 + n^2 + L (6 + 2k))` for `n = k +
p + q + P + Q` and `L = max(p + sP, q + sQ, 1)`; before it, the buffer,
`9 (2 + k)` a slot, held under `max_state_bytes`. Beside the filter's
(part 2's table), at `(1,1,1)(1,1,1,s)` without regressors:

| | `s = 7` | `s = 24` | `s = 168` | `s = 1440` |
|---|---|---|---|---|
| filter at given parameters (doubles) | 62 | 404 | 14,874 | 1,042,566 |
| estimator's run (doubles) | 70 | 172 | 1,036 | 8,668 |

No decay, no window: the run is bounded by the orders and the season,
the buffer by the cap.

**Fixtures**: `sarima_estimated` (300 rows: the first target's run, the
second's buffer) and `sarima_buffering` (40 rows: both buffering), and the
two part-2 fixtures regenerated for the layout's new fields (`est`,
`over`, `warned`, `warning`, `estimate`). No release and no kept fixture
set holds a `Sarima` (schema 72 is unreleased, the variant new in it), so
the layout change needs no loader and no copy to `v<N>/`.

**At fad8ea9d, `test_a_season_of_168_is_sarimax_too` fails** (extended
tier; the Python tests had not run in part 2): `(1,1,1)(1,1,1,168)`, 600
rows, the gap from `SARIMAX.filter` is 12.4 ulps of the largest `|w|`
against its bound of 8 (part 2 had measured 6.2 through a scratch
harness). An extended-precision reference (x87 80-bit, `eps` 1.1e-19; the
stationary covariance as its defining series, 4,705 terms, Lyapunov
residual 3.1e-19 of its scale; the filter in the same precision) settles
it on this machine: the core is within 0.88 ulps of it on every row,
statsmodels 12.5 (7.2 over the first 50 rows, 8.7 past row 300), its
dense products at 170 states. The test is left failing as it stands: the
reference is not portable (macOS arm64's `long double` is a double), an
mpmath one at 170 states would take tens of minutes, and a bound loosened
to statsmodels' error is the coordinator's call, not this session's.

**At fad8ea9d** (part 2), `state_fixtures` failed two tests: the
committed `sarima` and `sarima_filling` fixtures were not what that build
wrote (`sarima`, row 3 after the save, target 0: 0.4624 against a frozen
0.7943). Part 2's fixtures were written before its last change; this
part's regeneration replaces them.

**Measured against `SARIMAX.fit`**, eight seeds of each of five specs at
10,000 rows (bound 0.1; the table above, measured before the start's `ln`
became the `libm` crate's): every parameter within 0.4 of `SARIMAX`'s OPG
standard errors. The fifty-seed oracle test of "Oracles" (48 of 50 seeds
within 2) is not yet in `tests/`: it needs a Python entry for the
estimated form, which comes with the bank entry.
