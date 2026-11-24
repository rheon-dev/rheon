<!-- kind: record -->
# ARIMA and SARIMA online: design

Design record for PLAN 487, part 1: the model's equations, its state, how it
keeps hard rules 2, 8 and 9, and the questions for the review. No code. The
measurements behind it are `docs/records/STUDY-487-488.md` on branch
`vm/study-487` ("the study", §3 to §6 and §10 to §11), which extends the
ARMA(1,1) estimator of `ORACLE-STUDY-485.md` (branch `vm/study-485`), and the
library scan `vm-tools/research/485-TIME-SERIES-SCAN.md` (branch
`vm-tools/oracle-gap-scan`, §1 and §3, "the scan"). The lag rule and the rule
for a row of weight 0 follow `ar` (task 486, branch `vm/task-486-ar-spectral`);
the batch start and its buffer follow `garch` (`docs/records/GARCH-DESIGN.md`).

**One change from the study.** The study started its fit after a fixed count
of `max(500, 3s)` differenced rows. This design replaces the count with a
condition on the weight: the start is fitted, and the fit reported, once the
accumulated weight `n_eff` passes the spec's `min_weight`, as every other
model's warm-up gate reads. The default `min_weight` is set so that at unit
weights it buffers exactly the rows the study measured (below, "The start").

## The model

statsmodels' signs and orders, so a spec reads as `SARIMAX`'s:

```text
phi(B) Phi(B^s) w_t = theta(B) Theta(B^s) e_t,     w_t = (1 - B)^d (1 - B^s)^D y_t
phi(B)   = 1 - phi_1 B - ... - phi_p B^p           Phi(B^s)   = 1 - Phi_1 B^s - ... - Phi_P B^(sP)
theta(B) = 1 + theta_1 B + ... + theta_q B^q       Theta(B^s) = 1 + Theta_1 B^s + ... + Theta_Q B^(sQ)
A(B) = phi(B) Phi(B^s),   C(B) = theta(B) Theta(B^s),   delta(B) = (1 - B)^d (1 - B^s)^D
```

The spec names `order = (p, d, q)` and `seasonal_order = (P, D, Q, s)`;
ARMA is `d = D = 0` with no season, ARIMA is no season. Each target is its
own series, with its own rings, parameters and start. `B` steps one
**learned row** of that target, not one unit of the clock (the lag rule,
below).

## The recursion, one learned row at a time

Write `beta` for the stacked parameters `(phi, theta, Phi, Theta)`, `n` of
them, and `ê` for the kept residuals. Row `t` of a target, in this order:

**1. Predict, before `y_t` is read** (hard rule 2):

```text
ŵ_t = - sum_{k>=1} A_k w_{t-k} + sum_{j>=1} C_j ê_{t-j}
ŷ_t = ŵ_t - sum_{k>=1} delta_k y_{t-k}
```

Every term reads the rings only. `ŷ_t` is the row's `pred`; it is reported
only while the target's start is fitted and `n_eff >= min_weight`.

**2. Read `y_t`** and form `w_t = delta(B) y_t` from the differencing ring,
the a-priori error `e_t = w_t - ŵ_t`, and the gradient `psi_t = -de_t/dbeta`
from four filtered signals, each a ring at the current parameters (`x / C(B)`
is the recursion `xf_t = x_t - sum_{j>=1} C_j xf_{t-j}`):

| signal | definition | its gradient entries |
|---|---|---|
| `u` | `Phi(B^s) w_t / C(B)` | `psi_{phi_i} = u_{t-i}` |
| `v` | `phi(B) w_t / C(B)` | `psi_{Phi_i} = v_{t-s i}` |
| `r` | `Theta(B^s) e_t / C(B)` | `psi_{theta_j} = r_{t-j}` |
| `z` | `theta(B) e_t / C(B)` | `psi_{Theta_j} = z_{t-s j}` |

At fixed parameters from a zero start these are the exact derivatives of the
conditional sum of squares; the study held them to a central difference of it
to 2.0e-11 and 2.3e-11 relative.

**3. Step** (recursive prediction error, Gauss-Newton; Ljung and Söderström
1983), with the row's weight `w` and the fit's weight `V`, the weight of the
target's rows learned into the fit (the start's buffer and every row since):

```text
V'     = V + w,          a = w / V'
R'     = R + a (psi psi' - R)
beta'  = Proj(beta + a R'^{-1} psi e)
s2'    = s2 + a (e^2 - s2)
```

At unit weights `a = 1/V'` is the study's gain `1/W_t`. `Proj` clips each
coefficient to `[-0.99, 0.99]`, which for polynomials of order at most one
(every spec the study measured) is the stationary and invertible region. A
higher order needs a projection that keeps the roots outside the unit circle
(question 2). The kept residual `ê_t` is the a-posteriori one, at `beta'`, as
in the study. The rings then take `y_t`, `w_t`, `ê_t`, `u_t`, `v_t`, `r_t`,
`z_t`.

**Fixed parameters.** With `params` given, step 3 is skipped and steps 1 and 2
run from the first differenced row: the conditional (zero-start) predictor at
those parameters. It equals `SARIMAX.filter(params)` once the zero start has
decayed: over the last 1,000 of 10,000 rows, to 3.5 ulps of the largest
`|w|` at `s <= 24`, and to 2.2e-11 absolute at `s = 168`, where the
start still decays as `|Theta|^(t/s)` (the study §4, §6.5). The exact Kalman
form from the first row is question 4.

## The start, and the warm-up condition

**The condition.** A target's fit is withheld until its accumulated weight
`n_eff` reaches `min_weight`. Until then its rows of positive weight with the
target present are buffered as `(w_t, weight)` pairs once the differencing
ring is full, and report a null `pred` with a `withheld_reason` naming the
stage: "filling the differencing ring (`d + sD` rows)" or "buffering rows for
the start until the weight reaches `min_weight`". On the row whose learning
first brings `n_eff` to `min_weight` or past it, after that row is learned:

1. the buffer's weighted conditional sum of squares, `sum w_t e_t(beta)^2`
   from a zero start, is minimized by the core's projected Newton with Armijo
   backtracking (the optimizer `garch` already holds; the study ran the same
   method in numpy), the gradient and Gauss-Newton matrix from the recursions
   above at fixed parameters, the projection `Proj`, from the start grid
   `{-0.5, 0, 0.5}` per coefficient;
2. the buffer is replayed at the fitted `beta`, held, building `R`, `s2` and
   every ring, with `V` the buffer's weight;
3. the buffer is dropped. Steps run from the next row, which is the first
   whose `n_eff` (before it) has reached `min_weight`, so the first row
   reported is the first row predicted by a fitted model: no row is reported
   before the fit and none is withheld after it.

**The default threshold**: `min_weight = d + sD + max(500, 3s)` (`max(500,
3s)` without a season). `n_eff` counts every learned row, the `d + sD` rows
that fill the differencing ring among them (hard rule 8, below), so at unit
weights this buffers exactly `max(500, 3s)` differenced rows, the start the
study measured: 50 of 50 seeds within 2 of `SARIMAX`'s standard errors on all
four sets at `s = 24` and `168`, every gap under 1.5 (study §6.2, §6.5). It
is a weight, read as every `min_weight` is: a stream whose rows weigh 0.5
takes twice the rows, one whose rows weigh 2 half as many.

**The guards on the condition**, each a condition, none a count of rows:

- a `min_weight` below the start's least weight is refused when the spec is
  built: the fit needs differenced rows to fit, so `min_weight` must exceed
  `d + sD` (recommended: refused below `d + sD + 10 n`, ten rows of weight 1
  a parameter; question 1);
- a start whose fit has not converged (its Newton decrement above a stated
  tolerance at Kish's count) or sits on the projection's boundary waits until
  the weight doubles and refits, at most twice, as `garch`'s start does; with
  no doubling left it is stepped from, and the start's report says so;
- the buffer is held under `max_state_bytes` as it runs: a warning at half,
  a refusal at the cap, naming the row and the bytes. With unit weights the
  bytes are known when the spec is built and checked then.

The study's plain start (zero parameters, a warm phase, steps with no buffer)
is not offered: it passes at `s = 7` and `24` but fails at `s = 168` (0 and 1
of 50 seeds), its slow approach measured in seasons (study §6.5).

## Rows, weights and the clock

- **Out of sample** (hard rule 2): step 1 reads only the rings, before `y_t`.
  The study's check changed row 1,500 by 100 and no prediction up to and
  including it moved; the build repeats it in Rust and through the bank.
- **`n_eff`** (hard rule 8): the weight of the target's learned rows before
  this row's update, `n_eff' = n_eff + w` on a learned row and unchanged on
  any other; 0 on the first row, 1 after one row of weight 1. There is no
  decay, so there is no decay before the row to exclude (question 5). The
  emitted `weight_sum` is the shared weight; each target's own gates it,
  through `target_n_eff_into`. `V`, the fit's weight, is a separate number
  used only for the gain: the weight of the rows in the start's buffer and
  of every row that has stepped the fit since, so `n_eff` less the weight of
  the rows that only filled a ring.
- **A row of weight 0** (hard rule 9) learns nothing and moves no ring:
  no residual, no step, no sum; the next row's lags are the ones before it,
  as if it were not there. This is `ar`'s rule and `garch`'s, and
  statsmodels' `missing="drop"`. Its `pred` is still computed from the rings
  and reported under the same gate. A zero-weight first row leaves every
  weight at 0 and divides by nothing: `a = w / V'` is reached only on a
  learned row, where `V' >= w > 0`. On a seasonal spec this drops the row
  from the season's count of rows (question 3).
- **A null target**: the same as a row of weight 0 (question 3 asks whether
  a seasonal spec should keep the slot instead).
- **The lag rule.** Lags, differencing and the season count learned rows, not
  clock units. A clock break (a session change, a gap past `gap_cap`, a
  reset) empties every ring (`clear_lags`): the rows either side are no longer
  adjacent. `beta`, `R`, `s2`, `V` and `n_eff` are kept. After a break the
  first `d + sD` learned rows refill the differencing ring and report null
  with that reason; then the recursion runs from zero residuals, and steps
  resume once the deepest ring (`max(p + sP, q + sQ)` rows) is full again,
  so no step reads a ring the break emptied (not measured; question 6). A
  break inside the start's buffer drops the buffer's rows before it from
  the replay's rings but keeps them in the fit, each stretch's conditional
  sum of squares from its own zero start.
- **Chunk invariance** (hard rule 3): every number above is a function of
  the rows in order. The start fires on a row's weight, not on a chunk's
  end, and a save inside the buffer holds the buffer.
- **No window form.** A recursive estimator is not a difference of running
  sums, so `window_size` is refused (scan §1, finding 4).

## State

Per target: the rings, each as long as the largest lag a recursion reads from
it, `beta`, the `n (n + 1) / 2` distinct entries of `R`, and the scalars
`s2`, `V`, `n_eff`, and the stage. With `L_e = q + sQ`:

| ring | length | what reads it |
|---|---|---|
| `y` | `d + sD` | the differencing |
| `w` | `p + sP` | the AR lags |
| `e` | `L_e` | the MA lags, and `r` and `z`'s numerators |
| `u` | `max(p, L_e)` if `p > 0` | `psi_phi` |
| `v` | `max(sP, L_e)` if `P > 0` | `psi_Phi` |
| `r` | `max(q, L_e)` if `q > 0` | `psi_theta` |
| `z` | `max(sQ, L_e)` if `Q > 0` | `psi_Theta` |

Bounded by the spec, never by the stream; linear in `s` for polynomials of
order at most one (the study §6.4, checked there against the floats its
implementation allocated):

| form | bytes a target | `s = 7` | `s = 24` | `s = 168` | `s = 1440` |
|---|---|---|---|---|---|
| `(0,1,1)(0,1,1,s)` airline | `32 s + 120` | 344 | 888 | 5,496 | 46,200 |
| `(1,1,1)(1,1,1,s)` | `56 s + 216` | 608 | 1,560 | 9,624 | 80,856 |
| `(1,0,1)(1,0,1,s)` | `48 s + 208` | 544 | 1,360 | 8,272 | 69,328 |
| `(1,1,1)(1,1,0,s)` | `24 s + 160` | 328 | 736 | 4,192 | 34,720 |
| the start's buffer, until fitted, unit weights | `16 max(500, 3s)` | 8,000 | 8,000 | 8,064 | 69,120 |
| Kalman state and covariance, `(1,1,1)(1,1,1,s)`, for comparison | `≈ 4 s²` | 432 | 3,016 | 117,640 | 8,334,760 |

The buffer is 16 bytes a row (the differenced value and its weight; the
study's 8 had unit weights) and is gone after the start. Under the default
`max_state_bytes` of 256 KiB, the full spec with its default start fits to
`s = 2,518` and the airline spec to `s = 3,275`; after the start alone, to
`s = 4,677` and `8,188`. A larger season needs a larger cap, raised by the
user in the spec. The state's size is checked when the model is built, as
`ar`'s is. A row costs `O(number of non-zero lags + n²)`, flat in `s` (0.24
to 0.32 s per 10,000 rows in the study's Python at every `s`).

A new `ModelState` variant is a layout change from 1.0 (hard rule 5): the
current fixture sets are copied to `v<N>/`, a loader ships, the sets are
regenerated, and the new fixture freezes each form the state can take: inside
the differencing ring's fill, inside the buffer, after the start, a fixed-
parameter spec, and a ring part-filled after a break.

**Platforms.** The recursion, the step, the projection and the start's
optimizer are IEEE arithmetic in a fixed order with no `ln`, `exp` or `powf`,
so every field is the same bits on every platform; no libm result sits in the
state.

## Where each piece lives

- `rheon-core`: the model behind `OnlineModel` (one file for the recursion
  and its rings, one for the start), with no Polars type. The start's
  optimizer is `garch`'s projected Newton lifted into a module both share,
  not a second copy (the scan's finding that one recursive estimator serves
  487, 488, 492 and 493).
- `rheon-polars`, `rheon-py`, the CLI: the bank entry, the spec, the TOML
  config, as for every model; `docs/MODELS.md` and each place
  `docs/EXTENDING.md` lists.
- Outputs per row: `pred`, `weight_sum`, `withheld_reason`; `coef()` reads
  `beta` and `s2` by name (`phi_1`, ..., `sigma2`).

## Oracles

- **Fixed parameters** (exact): `SARIMAX(order, seasonal_order,
  trend="n", simple_differencing=True).filter(params)` with
  `model.ssm.tolerance = 0` (statsmodels' steady-state switch off; with it on
  the gap is about 1e-9). Tolerances from the study §10.1: 8 ulps of the
  largest `|w|` at `s <= 24` after 500 rows; 1e-10 absolute at `s = 168`
  over the last 1,000 of 10,000 rows; 5e-9 against the default filter.
  `arma_innovations` is the second path, sharing no code with the filter.
- **The estimator** (statistical): `SARIMAX(...).fit(method="lbfgs",
  cov_type="opg")` on the same rows, every parameter at the last row within
  2 of its standard errors; floor 48 of 50 seeds at 10,000 rows, the study's
  per-seed bounds (§10.2) beside it. At `s = 168` the oracle's memory-light
  path (`fit(low_memory=True, cov_type="none")` then `filter(..., cov_type=
  "opg")`) gave the default path's numbers to the bit at `s = 7`.
- **In Rust**, which `cargo mutants` sees: the gradient against a central
  difference of the conditional sum of squares; the out-of-sample check; scale
  equivariance (`y` times 37: coefficients to 1e-12 relative, `sigma2` by
  `37²`); a zero-weight first row and a zero-weight row inside the buffer;
  `n_eff` in `model_contract.rs`; a save at every row continuing to the bit.
- Python: `tests/test_second_opinion.py` (or a file of its own, as `garch`'s)
  takes `SARIMAX`.

## Left out of the first build

Polynomial orders above one until question 2 is measured; exogenous
regressors and a trend (regression with SARIMA errors: one more filtered ring
per regressor, `x / C(B)` after `A(B) delta(B)`, not measured); forecasts past
one row; the information criteria and roots at emit; decay; the method-of-
moments estimators (Yule-Walker, innovations), which the scan finds exact
from kept autocovariances and which belong beside `ar`; sharing the `r` and
`z` rings (one can be derived from the other; not tried).

## Questions, with recommendations

1. **The start's floor.** Recommend refusing a `min_weight` below
   `d + sD + 10 n`, so a start always has some rows a parameter; and, so that
   one heavy row cannot open the start, also requiring Kish's count of the
   buffer, `(sum w)^2 / sum w^2`, to reach `10 n` before the fit. Neither
   number was measured; the study ran unit weights only.
2. **Orders above one.** Recommend carrying each polynomial's parameters as
   partial autocorrelations (statsmodels' `constrain_stationary_univariate`,
   Monahan's transform, which needs only `sqrt`) with the chain rule through
   the gradient, and measuring orders 2 and 3 against `SARIMAX` before
   lifting the refusal. Alternative: ship orders of at most one, the measured
   region. The start grid of `3^n` points needs a cap or a sparser grid past
   `n = 4`.
3. **A zero-weight row or a null on a seasonal spec.** Recommend `ar`'s rule
   (dropped, the season counted in learned rows), so every model in the
   library reads weight 0 one way. Alternative: the row keeps its slot, filled
   by its prediction with a residual of 0 and no step, which is what
   `SARIMAX` does with a `NaN` and keeps a daily season aligned when an hour
   is masked. Neither was measured.
4. **The exact Kalman form for fixed parameters.** PLAN 487 asks for it
   through `kalman`'s filter, but that filter's transition is diagonal and its
   covariance costs `≈ 4 s²` bytes (117 KB at `s = 168`, 8.3 MB at `1,440`).
   Recommend shipping the conditional predictor above for fixed parameters,
   exact against `SARIMAX` after the zero start's transient, and offering the
   Kalman form later under a cap near `r = 256` (263 KB), past which the
   conditional form is the only one.
5. **Decay.** Recommend none in the first build (as `garch`), so `n_eff` is
   the plain sum and the condition can always be met. With a half-life the
   weight tops out at `w / (1 - lambda)`, so a `min_weight` above it would
   never open the start and must be refused when the spec is built; the gain
   would take a floor of `1 - lambda`. Not measured.
6. **After a clock break.** Recommend holding steps until the deepest ring
   has refilled, as above. Alternative: step at once from zero residuals.
   Not measured; the oracle has no breaks, so the test is a longhand.
7. **`sigma2`'s bias.** `s2` reads high by up to 1.5 standard errors at
   `s = 168` because it averages the buffer's zero-start residuals. Recommend
   measuring `s2` over the rows after the start only before choosing; the
   study's form passes the 2-SE criterion as it stands.
8. **The test grid's cost.** A `SARIMAX` fit at `s = 168` takes 106 to 183 s
   (the 100 fits at `s = 168`, with the `s = 24` grids, ran 110 minutes on
   four workers). Recommend the
   `s = 7` and `24` grids in the extended tier and `s = 168` in the soak
   tier, with fewer seeds and the same floor's proportion.

## Index row (for the coordinator to add at merge)

```text
| [records/ARIMA-DESIGN.md](records/ARIMA-DESIGN.md) | record | ARIMA and SARIMA online: the recursion, the state, the start on a weight condition, the hard rules and the review's questions (the design part of PLAN 487) |
```
