# Task 485: the oracle-library scan for the time-series models (486 to 493)

*Status, 2026-11-20: research only. Nothing in the repository was changed and no
model was run. Written from the tree at `a5c4dcef` (`integration-w4`, slot 1).*

## 0. What this record is, and what stands behind each claim

For each family in PLAN 486 to 493 this record lists the features and parameters
the Python libraries offer, classifies each one, names the oracle and its
tolerance, lists the hard parts, and proposes a build order.

**How each claim was checked.** Three grades, marked where it matters:

| mark | meaning |
|---|---|
| (src) | read from the installed source or signature in the project's locked environment: statsmodels 0.15.0, scipy 1.18.1, river 0.26.1, under `.venv/lib/python3.14/site-packages/`. Signatures were dumped with `inspect.signature`; nothing else was executed |
| (web) | read from the library's documentation or its source on the default branch, fetched 2026-11-20. `arch`, `sktime`, `darts`, `pmdarima`, `statsforecast`, `pymgarch` and `mvgarch` are **not installed**, so none of their behaviour was run. PyPI lists `arch` 9.0.0; the docs pages do not state their version, so a parameter cited from them must be checked against the installed release before a test cites it (the standing rule of the 2026-11-20 scans) |
| (mem) | from memory of the literature. Every paper reference and every convergence statement about a recursive estimator is this grade. Confirm the reference before a docstring cites it |

**No tolerance below was measured.** Each is a proposal with its reasoning. The
project's rule is to measure a premise before believing it, so section 11 puts a
measurement study first.

**Classification** (PLAN 485's four classes):

| code | class |
|---|---|
| E | online exactly: a recursion that equals the batch answer on the same rows |
| A | online approximately: a recursive estimator that converges to the batch fit |
| D | online with an embargo: exact after a stated delay, over a bounded window |
| N | not online, with the obstacle |

**Decay and window columns.** "decay" is exponential decay on the clock
(`half_life`); "window" is a hard rolling window (`window_size`). Each says
whether the form is feasible and whether an oracle settles it, in
ORACLE-COVERAGE's marks: [T] a third-party library computes the quantity, [L] a
definition written by hand. Per the plan's rule, a form with no oracle is not
built until one exists.

## 1. Findings that cut across the families

1. **Every library model assumes an evenly spaced series indexed by position.**
   rheon's clocks are uneven. Lags, seasonal phase, differencing and the DFT
   are all defined by row position in the oracles. Each model therefore needs
   one stated rule: a step is one learned row, a clock break empties what is
   kept by position (the existing lag rule, `docs/MODELS.md` "The clock's
   policy"), and decay runs on the clock. On an even clock this equals the
   library; on an uneven clock it is rheon's own definition, held by a
   longhand only. The exceptions, which are defined on the clock itself and so
   keep a library oracle on uneven data: `CalendarFourier` (489) and
   Lomb-Scargle (491).
2. **Fixed parameters are exact almost everywhere; estimating them is where
   the approximation enters.** With given coefficients, the ARMA filter, every
   ETS recursion, every `arch` variance recursion and the DCC `Q` recursion
   are one-pass recursions with a library oracle at rounding tolerance. Batch
   maximum likelihood over all rows is never online. So each of 487, 488, 492
   and 493 splits in two: a filter (class E) and an estimator (class A).
3. **One recursive estimator serves 487, 488, 492 and 493.** The recursive
   prediction-error method (Ljung and Soderstrom 1983, (mem)): parameters
   `θ`, a Gauss-Newton matrix, the sensitivities `∂state/∂θ` carried by their
   own exact linear recursion, and a projection onto the admissible set at
   every step. ARMA's moving-average terms, ETS's smoothing parameters, GARCH's
   quasi-likelihood and DCC's `(a, b)` differ only in the model equations. It
   should be built once, on the hardest case.
4. **`window_size` today is a subtraction of running sums** (a ring of
   snapshots; `docs/how-to/forget-rows-past-a-window.md`). It carries over
   only to models whose state is a sum over rows: least squares (486),
   autocovariances (487's method-of-moments estimators), Lomb-Scargle and
   Welch sums (491). A nonlinear recursion (ETS states, a GARCH variance, a
   recursive estimator) is not a difference of sums; its exact window form is
   a re-run over the window's rows for every row, `O(window)` per row, which
   breaks the one-pass cost. Those forms are recorded as left out, with that
   obstacle. 490 and 491's sliding DFT are window models by nature (a ring of
   raw values).
5. **A library's fitted values are in-sample; rheon's `pred` is not** (hard
   rule 2). An oracle test compares one of: coefficients at the last row
   against the batch fit on all rows; predictions against the library run with
   the same fixed parameters; or predictions against the library refitted on
   prefixes ("run once on prefixes", as the existing diagnostics tests do).
6. **Discoveries that change a plan row:**
   - `scipy.signal.lombscargle` takes `weights` and `floating_mean` in 1.18.1
     (src), so a decayed Lomb-Scargle has an exact third-party oracle.
   - statsmodels' `levinson_durbin(s, isacov=True)` and `innovations_algo(acov)`
     take autocovariances directly (src), so Yule-Walker AR and the innovations
     MA estimate are **exact** online from kept lag sums, under decay too
     (weights written by hand, the library does the algebra).
   - `statsmodels.tsa.innovations.arma_innovations.arma_innovations(endog,
     ar_params, ma_params, sigma2)` (src) is a second fixed-parameter ARMA
     oracle that shares no code with `SARIMAX`'s Kalman filter.
   - Without its robustness iterations STL is a linear filter, so its value at
     a fixed lag inside a sliding window is one fixed kernel applied to the
     window: `O(window)` per row, not a full STL per row (section 6).
   - PLAN 493 says no mainstream Python library fits DCC. `arch` has none
     (web). But `pymgarch` 0.6.0 (MIT, first release 2026-08-31) lists DCC(1,1),
     ADCC, correlation targeting, a composite-likelihood option and a `filter()`
     for fixed parameters, and claims replication tests against R's `rmgarch`
     (web, PyPI page only; its `Q` recursion and targeting were not read). It
     is young and unvetted: a candidate second opinion for 493 and 514, not yet
     an oracle.
   - `arch`'s default start of the variance recursion (the "backcast") and its
     variance bounds both read the whole sample (web, `volatility.py`), so a
     bit-level match needs the backcast passed in and the bounds not binding
     (section 8).
7. **Earlier decisions this work touches.** `docs/ENHANCEMENTS.md` E25 recorded
   "deliberately no seasonal term: a seasonal index is a `group_by` on the
   phase"; 488 reverses it, and the `group_by` form stays as the cheap baseline
   (it keeps a level and trend per phase, where Holt-Winters shares them).
   §5.1's rejection code X (randomness on the output path) excludes `arch`'s
   simulation and bootstrap forecasts and ETS's simulated intervals. The
   2026-11-20 libm rule is what allows EGARCH's log variance and a Box-Cox
   state.

## 2. Task 486: AR(p)

Libraries: statsmodels `AutoReg`, `ar_select_order`, `AutoRegResults`
(`statsmodels/tsa/ar_model.py`), `yule_walker`, `burg`
(`regression/linear_model.py`), `pacf*`, `levinson_durbin`
(`tsa/stattools/_stattools.py`) (src). `AutoReg.fit` is `OLS` on the lag design
(`ar_model.py:451`, src), so it takes no weights.

`AutoReg(endog, lags, trend='c', seasonal=False, exog=None, hold_back=None,
period=None, missing='none', *, deterministic=None)`;
`fit(cov_type='nonrobust', cov_kwds=None, use_t=False)` (src).

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| `lags`: an int or a list of lags | E: a ring of the last `max(lags)` learned values per target, then the least squares rheon already has | yes [T]: statsmodels `WLS` on `tsatools.lagmat` with the decay's weights | yes [T]: `AutoReg` on the window's rows plus the `max(lags)` rows before them, `hold_back=max(lags)` | `AutoReg(...).fit().params`, no decay: propose 1e-9 relative on a conditioned design, aligned with whatever `ewridge`'s existing `OLS` oracle test uses (not checked here) |
| `trend`: `n`, `c`, `t`, `ct`, `ctt` | E on an even clock. statsmodels' `t` is the row position from 1; rheon's is the clock from the group's origin, so coefficients differ by a scale and a shift and predictions agree | yes [T] as above | yes [T] | predictions, and coefficients after the affine map. `t` without `c` is not shift-invariant: fix the origin and say so |
| `seasonal=True`, `period` (dummies) | E: `period − 1` more regressors from the row's phase | yes [T] | yes [T] | `AutoReg(seasonal=True, period=)` |
| `exog` | E (the features rheon already takes) | yes [T] | yes [T] | `AutoReg(exog=)` |
| `hold_back` | E: rows before the ring fills are scored null and not learned; a larger `hold_back` skips more | n/a | n/a | same sample as `AutoReg(hold_back=)` |
| `missing`: `none`, `drop`, `raise` | `drop` is E: a null or zero-weight target is not learned and the lags close up, which is the existing "a lag counts learned rows" rule | n/a | n/a | `AutoReg(missing='drop')` |
| `deterministic=DeterministicProcess` | E through 489's terms | yes | yes | `AutoReg(deterministic=)` |
| `cov_type`: `nonrobust` | E at no decay (`emit_se_coef`) | yes, the existing weighted form | yes | `AutoRegResults.bse` |
| `cov_type`: `HC0` to `HC3` | E/A as today: `emit_robust_se` is HC0 on out-of-sample residuals; HC1 to HC3 are PLAN 512 | as 512 | as 512 | statsmodels, as 512 |
| `cov_type`: `HAC` | A: the batch form needs residuals at the final coefficients against lagged regressors, which is not a running sum at reasonable size; with out-of-sample residuals it is a bounded `O(L·k²)` sum | [L] only | [L] only | statistical, against `fit(cov_type='HAC')` on the final row; propose leaving out of the first build |
| `ar_select_order(maxlag, ic, glob=False)` | E for nested orders: one Gram at `maxlag` on the common sample holds every nested order as a leading block, so each order's SSR and information criterion comes from one factorization | yes [L] | yes [T] | `ar_select_order(...).ar_lags`, and `aic`/`bic`/`hqic` per order |
| `ar_select_order(glob=True)` (all subsets) | E in principle, `2^maxlag` solves per emit; propose refusing above a small `maxlag` | — | — | — |
| results: `params`, `bse`, `sigma2`, `llf`, `aic`, `aicc`, `bic`, `hqic`, `fpe` | E at emit, from SSR, the row count and `k` (the `ln` is in the output, not the state) | [L] | [T] on the window | `AutoRegResults`, 1e-9 |
| results: `roots`, `arfreq` | E at emit: eigenvalues of the companion matrix, `O(p³)`; gives a "stationary" flag | n/a | n/a | `AutoRegResults.roots`; eigenvalue tolerance 1e-8 |
| `forecast(steps)`, `predict(dynamic=)`, `get_prediction` | E given the coefficients: iterate the recursion `h` steps, `O(h·p)` per row; interval from the ψ-weights. With `exog`, only `h = 1` unless future features are given | n/a | n/a | `AutoRegResults.forecast`, with the coefficients fixed to rheon's |
| `test_serial_correlation`, `test_normality`, `test_heteroskedasticity` | the first two exist (`ljung_box`, `jarque_bera`); ARCH-LM is a small regression of `e²` on its lags, E | as the diagnostics today | — | `acorr_ljungbox`, `het_arch` (src), run once |
| `append`, `apply` | native: a saved state continues | — | — | state continuation tests |
| `yule_walker(order, method, demean)`, `pacf_yw`, `levinson_durbin` | E at emit from kept lag sums (the lag cross-products, the sums, and the first and last `m` values reproduce `acovf`'s final-mean form exactly) | yes [T]: weighted autocovariances by hand into `levinson_durbin(isacov=True)` | yes [T]: on the window | `yule_walker`, `pacf_yw`: 1e-10 |
| `pacf_ols` | E from the nested Grams | yes | yes | `pacf_ols(efficient=)` |
| `burg`, `pacf_burg` | N exactly: each order's reflection coefficient needs every row's forward and backward errors at the final lower-order coefficients. A gradient lattice approximates it | — | — | propose leaving out |
| the AR spectrum (parametric) | E at emit from the coefficients | n/a | n/a | `arima_process.arma_periodogram` or `scipy.signal.freqz` |

**Oracle.** `AutoReg` on the same rows, exact with no decay; `WLS` on `lagmat`
under decay; `AutoReg` on the window. Tolerance 1e-9 relative on coefficients
for a conditioned design, loosening with the condition number as `ewridge`'s
does.

**Hard parts and risks.** (1) `ar` is `ewridge` on lags it keeps itself, so
the work is the ring and the clock-break rule, not the solver. (2) A zero-weight
row: under the existing rule it is not learned and so not a lag, which matches
`missing='drop'`. A user who means "observe it but do not learn from it" gets a
different series; this needs a decision (section 12). (3) `t` and `t²` grow
without bound with no decay; the Gram's conditioning falls. Centre the trend on
the weighted mean clock as the intercept handling already does. (4) Near a unit
root least squares is still exact, but the standard errors are not the usual
ones; the docstring should say it.

**State.** `8·max(lags)` bytes per target for the ring, plus the `ewridge`
state at `k = p + trend + (period − 1) + exog` (a `k×k` Gram: about 0.6 KB at
`k = 11`). Under a window, the existing snapshot ring and its 256 MiB default
cap apply unchanged.

## 3. Task 487: ARMA, ARIMA and SARIMA(X)

Libraries: statsmodels `ARIMA` (`tsa/arima/model.py`), `SARIMAX`
(`tsa/statespace/sarimax.py`), the estimators in `tsa/arima/estimators/`
(`yule_walker`, `burg`, `innovations`, `innovations_mle`, `hannan_rissanen`,
`durbin_levinson`, `gls`, `statespace`), `arma_innovations`, `MLEResults`
(src). Catalogues: `pmdarima.auto_arima` 2.1.1 (web), river `SNARIMAX(p, d, q,
m, sp, sd, sq, regressor)` (src), sktime `ARIMA`, `SARIMAX`, `AutoARIMA`,
`ARDL`, `ARARForecaster`, darts `ARIMA`, `VARIMA`, `AutoARIMA` (web).

`ARIMA(endog, exog=None, order=(0,0,0), seasonal_order=(0,0,0,0), trend=None,
enforce_stationarity=True, enforce_invertibility=True, concentrate_scale=False,
trend_offset=1, dates=None, freq=None, missing='none',
validate_specification=True, validate_exog=True)`; `fit(start_params,
transformed, includes_fixed, method, method_kwargs, gls, gls_kwargs, cov_type,
cov_kwds, return_params, low_memory)`. `SARIMAX` adds `measurement_error`,
`time_varying_regression`, `mle_regression`, `simple_differencing`,
`hamilton_representation`, `use_exact_diffuse` (src).

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| differencing `d`, seasonal `D` at period `s` | E: a ring of `d + s·D` values | n/a | n/a | `SARIMAX(simple_differencing=True)` sees the same differenced series; integrated predictions add the kept values back exactly |
| AR and MA terms (`p`, `q` or lag lists), **fixed parameters** | E: the Kalman filter of the state-space form; state dimension `r = max(p + s·P, q + s·Q + 1)` | n/a (parameters given); the noise scale is a running mean of `v²/F`, which decays as `kalman`'s does | n/a | `SARIMAX(...).filter(params)`: `forecasts`, `forecasts_error`, `forecasts_error_cov`, `llf_obs`; and `arma_innovations` as a second path. Two Kalman implementations agree to about 1e-10 relative, not to the bit |
| seasonal `(P, D, Q, s)`, fixed parameters | E: the multiplied polynomials in the same filter | n/a | n/a | as above |
| initial state: stationary | E: the discrete Lyapunov solve at the start (and at a reset), `O(r³)` | — | — | `SARIMAX(enforce_stationarity=True)` default initialization |
| initial state: approximate diffuse (`1e6`) | E, with digits lost to cancellation in the first `d + s·D` rows; statsmodels burns those rows from the likelihood | — | — | `SARIMAX(use_exact_diffuse=False)`, the default; 1e-6 in the burn rows, 1e-10 after |
| `use_exact_diffuse=True` | E in principle (Koopman's exact diffuse filter), a second filter to write | — | — | propose leaving out of the first build |
| AR by Yule-Walker (`method='yule_walker'`) | E at emit from kept lag sums (section 2) | yes [T] | yes [T] | `arima.estimators.yule_walker`, 1e-10 |
| MA(q) by the innovations algorithm (`method='innovations'`) | E at emit: `innovations_algo` on the kept autocovariances | yes [T]: weighted autocovariances by hand into `innovations_algo` | yes [T] | `arima.estimators.innovations`, 1e-10 |
| ARMA by Hannan-Rissanen (`method='hannan_rissanen'`) | A: the long AR is exact, but the batch method takes every residual at the final long-AR coefficients, and a stream has only each row's coefficients at the time | [L] | no: the residuals are not a sum | `hannan_rissanen` at the last row; converges as the long AR settles; statistical |
| ARMA and SARIMA by maximum likelihood (`method='statespace'`, `'innovations_mle'`) | A: recursive prediction error (Gauss-Newton with filtered gradients). Asymptotically it has the batch estimate's distribution (mem); the transient is what a test sees | yes, weakly: the weighted conditional sum of squares written by hand and minimized by `scipy.optimize`, with statsmodels' unweighted fit beside it at no decay | no: a recursive estimator is not a difference of sums; a refit per row over the window is `O(window × iterations)` | `ARIMA(...).fit()` at the last row: propose within 0.5 batch standard errors after 5,000 rows for roots inside 0.8, over fifty seeds; to be measured (section 11, step 0) |
| ARMA by recursive pseudo-linear regression (extended least squares) | A: simpler (RLS on lagged values and lagged residuals), but consistent only where the MA polynomial meets a positive-real condition, and not efficient (mem) | as above | no | the same oracle, a looser tolerance. Propose not shipping it: its failure region is a silent wrong answer |
| `burg` | N (section 2) | — | — | — |
| `trend`: `n`, `c`, `t`, `ct`, a polynomial; `exog` (regression with ARMA errors) | fixed ARMA parameters: E with the coefficients in the state (`mle_regression=False`, which statsmodels describes as recursive least squares through the filter). Estimated jointly: A | as the row above | no | `SARIMAX(mle_regression=False).filter(params)`. Note river's `SNARIMAX` puts `exog` inside the ARMA recursion, a different model |
| `time_varying_regression=True` | E for a fixed coefficient variance: the regression coefficients as random-walk states | n/a | n/a | `SARIMAX(time_varying_regression=True, mle_regression=False).filter` |
| `measurement_error=True` | E for a fixed variance | n/a | n/a | `SARIMAX(measurement_error=True).filter` |
| `enforce_stationarity`, `enforce_invertibility` | in the filter: a check. In the estimator: held at every step by carrying the parameters as partial autocorrelations (the transform statsmodels uses) or by projection | — | — | a property test (every row's roots admissible) plus the limit oracle |
| `concentrate_scale` | n/a: the scale is the running mean of `v²/F` | — | — | `SARIMAX(concentrate_scale=True)` scale at the last row |
| `hamilton_representation`, `trend_offset`, `validate_*`, `dates`, `freq`, `low_memory` | n/a: no number a user reads changes | — | — | — |
| `missing` | E: a null target runs the prediction step and skips the update, which is statsmodels' state-space handling of `NaN` | — | — | `SARIMAX` with `NaN` in `endog` |
| `forecast(steps)`, `get_forecast` (intervals) | E given parameters: `h` prediction steps, `O(h·r²)` | n/a | n/a | `MLEResults.forecast`, `get_forecast().var_pred_mean` |
| `cov_type`: `opg`, `oim`, `approx`, `robust`, `robust_approx` | A: the estimator's Gauss-Newton matrix is an outer-product estimate | [L] | no | `fit(cov_type='opg').bse`, statistical |
| `fix_params`, `start_params` | E as configuration: held coordinates and a warm start | — | — | — |
| results: `resid`, `llf`, `aic`, `bic`, `arroots`, `maroots`, `impulse_responses` | E at emit given parameters | — | — | `MLEResults` |
| `append`, `extend`, `apply` | native | — | — | `MLEResults.extend` on a split stream equals the unsplit filter |
| `smooth`, `news`, `simulate` | N in this task: smoothing needs later rows (PLAN 496's fixed-lag smoother), `news` needs revisions, `simulate` is random | — | — | — |
| `arma_order_select_ic`, `pmdarima.auto_arima` (stepwise search, KPSS/ADF/OCSB tests, about forty parameters) | N as a search: several fits per candidate. The online analogue is a bank of specs with `emit_selected` | — | — | catalogue only |

**Oracle.** Fixed parameters: `SARIMAX.filter(params)` and `arma_innovations`,
about 1e-10 relative after the diffuse rows. Method-of-moments estimates
(Yule-Walker, innovations): statsmodels on the same autocovariances, 1e-10,
under decay and in a window too. Maximum likelihood: the batch fit at the last
row, statistical, the number to be measured.

**Hard parts and risks.**
- *The MA terms online.* The residuals a moving-average term needs are
  themselves outputs of the model, so the regression is on quantities that move
  as the parameters move. Three routes, in order of preference: (a) the
  innovations algorithm on kept autocovariances, exact, MA and AR only, not
  efficient; (b) recursive prediction error, efficient in the limit, needs the
  projection and a stated transient; (c) pseudo-linear regression, which can
  fail to converge and is not recommended. Offer (a) and (b) and name which
  one a spec runs.
- *The filter is not `kalman`'s.* `kalman` has a diagonal transition and the
  row's features as its observation vector. An ARMA state-space form needs a
  companion transition and a fixed observation vector: a second filter (or a
  generalization of the first), plus the Lyapunov start.
- *Memory at a long season.* The filter's covariance is `r(r+1)/2` floats:
  3 KB at `s = 24`, 116 KB at `s = 168`, 8.3 MB per group and target at
  `s = 1440`. Propose a cap near `r = 256` (263 KB) and, past it, the
  conditional (innovations-form) recursion, whose state is a ring of about
  `2s` floats. That form equals the filter only after its transient, so its
  oracle is a longhand plus the library in the limit. Bring the cap to the
  user, as the bounded-state rule asks.
- *Near non-invertibility* the recursive estimate's transient lengthens without
  bound; the test grid should stay inside roots of 0.8 and the docstring should
  say what happens outside.

**State.** Ring `8·(d + s·D)`; filter `8·(r + r(r+1)/2)`; estimator
`8·(n + n² + n·(q + s·Q))` for `n` parameters. ARMA(2,1) with estimation: about
0.3 KB.

## 4. Task 488: Holt-Winters and ETS

Libraries: statsmodels `ExponentialSmoothing`, `Holt`, `SimpleExpSmoothing`
(`tsa/holtwinters/model.py`), `ETSModel` (`tsa/exponential_smoothing/ets.py`),
the state-space `ExponentialSmoothing`
(`tsa/statespace/exponential_smoothing.py`, additive only, takes `NaN`),
`ThetaModel` (src); river `HoltWinters(alpha, beta, gamma, seasonality,
multiplicative)` (src); statsforecast `SimpleExponentialSmoothing`, `Holt`,
`HoltWinters`, `AutoETS`, `AutoCES`, the Theta family, `CrostonClassic`,
`CrostonSBA`, `TSB`, `ADIDA`, `IMAPA` (web); sktime `ExponentialSmoothing`,
`AutoETS`, `BATS`, `TBATS`, `Croston` (web); darts `ExponentialSmoothing`,
`Theta`, `FourTheta`, `TBATS`, `Croston` (web).

`ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None, *,
seasonal_periods=None, initialization_method='estimated', initial_level=None,
initial_trend=None, initial_seasonal=None, use_boxcox=False, bounds=None,
dates=None, freq=None, missing='none')`; `fit(smoothing_level, smoothing_trend,
smoothing_seasonal, damping_trend, *, optimized=True, remove_bias=False,
start_params=None, method=None, minimize_kwargs=None, use_brute=True)`.
`ETSModel(endog, error='add', trend=None, damped_trend=False, seasonal=None,
seasonal_periods=None, initialization_method='estimated', initial_level,
initial_trend, initial_seasonal, bounds, dates, freq, missing)` (src).

**A parameter trap.** `ETSModel`'s `smoothing_trend` is `β`, with the trend
update `b_t = (β/α)(l_t ⊖ l_{t−1}) + (1 − β/α) b_{t−1}` (`ets.py` module
docstring, src). `ExponentialSmoothing`'s `smoothing_trend` is the classical
`β* = β/α`. An oracle test must map between them.

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| the recursions, fixed `α, β, γ, φ`, known initial states: `error` ∈ {add, mul} × `trend` ∈ {none, add, mul} × `damped_trend` × `seasonal` ∈ {none, add, mul} (30 forms) | E: level, trend and `m` seasonal values | rheon's half-life form (each component's gain from its own decayed weight, as `holt` today): [T] once the weights saturate on an even clock, where the gains are the fixed ones (how `holt` is held now, to 1e-12) | no: the states are not a difference of sums. The additive forms with no decay inside a window are a regression on (1, t, seasonal dummies), which 486 and 489 give with `window_size` | `ETSModel(..., initialization_method='known').smooth(params)` and `ExponentialSmoothing(..., initialization_method='known').fit(smoothing_*=..., optimized=False)`: 1e-12 relative for additive forms; 1e-10 for multiplicative forms over long series (products accumulate rounding) |
| additive seasonal with no decay (`half_life=inf`) | E: it is the least-squares fit on (1, t, dummies), as `holt` with no decay is the weighted line | — | yes [T] | statsmodels `OLS` on `DeterministicProcess(constant=True, order=1, seasonal=True)`, 1e-9 |
| multiplicative seasonal or trend with no decay | definable (gains of `1/n`), but no least-squares equivalent and no library form | [L] only | no | a longhand only; propose refusing `half_life=inf` on multiplicative forms until an oracle exists |
| `initialization_method='known'` (`initial_level`, `initial_trend`, `initial_seasonal`) | E | — | — | direct |
| `initialization_method='heuristic'`, `'legacy-heuristic'` | D-like: needs the first 10 rows, or two full seasons, before any state exists (`exponential_smoothing/initialization.py`, src). Bounded buffer `O(m)`; no prediction until it fills. statsmodels' fitted values inside that span have seen those rows, so only rows after it compare | — | — | `ETSModel(initialization_method='heuristic')` states after the warm-up |
| `initialization_method='estimated'` | N: the initial states are optimized over the whole series | — | — | — |
| `optimized=True` (learn `α, β, γ, φ`), `bounds`, `method`, `use_brute`, `start_params`, `minimize_kwargs` | A, two routes: (a) a bank of fixed-parameter specs with `emit_selected`, each member exact, the choice an argmin over a grid, which is what `use_brute` starts from; (b) the recursive prediction-error step on the smoothing parameters, with `bounds` as its projection | (a) yes, each member as above; (b) weakly, a weighted SSE by hand through `scipy.optimize` | no | (a) each member against statsmodels, the selection against statsmodels' SSE on the same grid; (b) `fit(optimized=True)` with known initial states, statistical |
| `use_boxcox`: a float or `'log'` | E: a per-row transform; the state then holds a libm result (allowed since 2026-11-20, with the cross-OS ulp test) | as the model | as the model | `ExponentialSmoothing(use_boxcox=λ)` |
| `use_boxcox=True` (estimate λ) | N: a batch likelihood over all rows | — | — | — |
| `remove_bias` | E for the forecast at each row: the running mean residual. statsmodels subtracts the final mean from every fitted value, so only the last row compares | [L] | no | the last row's forecast |
| `missing`: `drop`; a null target | E: with a clock the state carries across the gap and the phase advances by the clock | — | — | the state-space `ExponentialSmoothing`, which takes `NaN` (additive forms only; this is how `holt` is held across a missing value today) |
| `forecast(steps)` | E: the closed form per variant, `O(1)` per horizon | — | — | `HoltWintersResults.forecast`, `ETSResults.forecast` |
| `get_prediction` intervals | E for the additive-error forms with closed-form variance; the rest are simulated in statsmodels (`simulate_repetitions=1000`), which is randomness on the output path | — | — | closed-form cases only; elsewhere rheon's conformal intervals |
| `simulate` | N (random) | — | — | — |
| `llf`, `aic`, `aicc`, `bic`, `sse`, `mse`, `mae` | E at emit for fixed parameters (running sums of the residuals) | [L] | — | `ETSResults` at the last row |
| `Holt(exponential=True)` | E: the multiplicative-trend form above | — | — | `Holt(exponential=True)` |
| Theta without the deseasonalizing step | E: simple smoothing plus a recursive trend line | — | — | `ThetaModel(deseasonalize=False)`; found, small, could ride with this task |
| Croston, SBA, TSB (intermittent demand) | E: two exponential smoothers on the nonzero sizes and the intervals between them, `O(1)` state | [L] | no | statsforecast `CrostonClassic`, `CrostonSBA`, `TSB`; sktime `Croston`. Found, not asked for; belongs to 494's scan |
| BATS, TBATS, `AutoETS`, `AutoCES` | N as estimation (batch search over forms and parameters); a fixed-parameter TBATS is an innovations filter | — | — | catalogue only |

**Oracle.** `ETSModel.smooth(params)` and `ExponentialSmoothing.fit(...,
optimized=False)` with known initial states and the same fixed parameters,
1e-12 (additive) to 1e-10 (multiplicative). With no decay, `OLS` on the
deterministic terms for the additive forms.

**Hard parts and risks.**
- *The multiplicative seasonal state.* It divides by the seasonal factor and by
  the level. A non-positive target is refused by statsmodels ("endog must be
  strictly positive", `holtwinters/model.py:247`, `ets.py:476`, src); rheon
  must refuse the row or the spec the same way, and guard a factor that a
  large residual drives to zero or below, since one such division poisons the
  state for good (rule 9's lesson). Renormalizing the factors is good practice
  but departs from the oracle: an option, off by default.
- *The seasonal phase on an uneven clock.* By position, a missing row shifts
  every later phase. From the clock, a phase is `floor((clock − origin)/step)
  mod m` and needs a step. Choose one and state it; only the even-clock case
  has a library oracle.
- *Half-lives against smoothing parameters.* `holt` exposes half-lives and its
  gains depend on the clock step. A seasonal index then needs a decayed weight
  per phase (`m` more floats), and the library only agrees at saturation. A
  direct `alpha=`, `beta=`, `gamma=`, `phi=` form on a row clock would agree
  with statsmodels from the first row after known initial states: worth
  offering beside the half-life form.
- *A slow start.* A seasonal state forgets its start at `(1 − γ)` per season,
  not per row, so "after saturation" can be thousands of rows.

**State.** `8·(2 + m)` bytes per target, plus `8·m` for per-phase weights under
half-lives: 0.4 KB at `m = 24`, 2.7 KB at `m = 168`, 140 KB at `m = 8760`.
Learned parameters multiply it by `1 + n_params` for the sensitivities.

## 5. Task 489: Fourier seasonality terms

Libraries: statsmodels `Fourier(period, order)`, `CalendarFourier(freq, order)`,
`DeterministicProcess(index, *, period, constant, order, seasonal, fourier,
additional_terms, drop)`, `TimeTrend(constant, order)`, `Seasonality(period,
initial_period)`, `CalendarSeasonality(freq, period)`, `CalendarTimeTrend(freq,
constant, order, *, base_period)` (`tsa/deterministic.py`, src). Catalogues:
sktime `FourierFeatures`, darts' cyclic datetime encoders, Prophet's Fourier
seasonality on continuous time (web, mem).

| feature or parameter | class | decay and window | oracle and tolerance |
|---|---|---|---|
| `Fourier(period, order)`: columns `sin(i,period)`, `cos(i,period)`, `i = 1..order`, `2·order <= period` | E, stateless. statsmodels' argument is the **row position**, `2π·i·n/period` with `n = 0, 1, ...` (`deterministic.py:455-461, 528-533`, src), so it is the oracle on an even clock only. On the clock, the term is `sin(2π·i·(clock − origin)/period)` | n/a: features, not a model. The regression that reads them decays or windows | `Fourier.in_sample`. statsmodels forms `2π·i·n/period` in floating point, so its own phase error grows as `ε·2π·i·n/period`; a phase reduced modulo the period in integers first is more accurate than the oracle on a long series. Tolerance `4ε·(2π·i·n/period) + 2 ulp` |
| `CalendarFourier(freq, order)`: the fraction of the calendar period elapsed, from the timestamp (`_compute_ratio`, `deterministic.py:571-579`, src) | E, stateless, and defined on the clock: the library oracle holds on uneven timestamps | n/a | `CalendarFourier.in_sample` on a `DatetimeIndex`, 2 ulp plus the ratio's rounding. Months, quarters and years vary in length, so this needs civil-date arithmetic; and a local-time day is 23 or 25 hours at a daylight-saving change, which statsmodels' period arithmetic and rheon's must be compared on explicitly |
| several periods at once | E | n/a | `DeterministicProcess(additional_terms=[Fourier(...), ...])` |
| `Seasonality(period, initial_period)`, `CalendarSeasonality(freq, period)` (dummies) | E, stateless | n/a | `in_sample` |
| `TimeTrend(constant, order)`, `CalendarTimeTrend` | E; needs the group's clock origin, which the state already keeps | n/a | `in_sample`, after the affine map of section 2 |
| `drop=True` (drop collinear terms) | N as a data-dependent rank test over all rows; a fixed rule (drop the first dummy beside a constant) is E | n/a | the fixed rule only |
| `out_of_sample(steps)` | E: the terms at a future clock, which multi-step forecasts in 486 to 488 need | n/a | `out_of_sample` |
| Fourier terms with random-walk coefficients (a trigonometric seasonal) | E by composition: `kalman` over these features with a `coef_half_life`. In distribution this is `UnobservedComponents(freq_seasonal=[{'period', 'harmonics'}], stochastic_freq_seasonal=[True])` when the pair's noise is isotropic (mem: a rotation leaves isotropic noise unchanged) | as `kalman` | to verify in the build: `UnobservedComponents(...).filter(params)` predictions |

**Oracle.** statsmodels' columns on the same index. `CalendarFourier` is the
one that needs no even clock.

**Hard parts and risks.** (1) `sin` and `cos` are libm: the outputs differ in
the last bit between glibc and Apple, and numpy's vectorized `sin` is not the
platform libm either. The cross-platform tolerance is stated in ulps, not
bits. (2) Where the terms are computed: hard rule 4 keeps clocks-as-columns out
of the core, yet the CLI needs the terms too, so the natural home is the
feature plumbing in `rheon-polars`, reached from a spec parameter. (3) Calendar
periods and time zones, above.

**State.** None beyond the clock origin already kept. Output: `16·order` bytes
per period per row if emitted; nothing if fed straight to a model's features.

## 6. Task 490: STL and MSTL, with an embargo

Libraries: statsmodels `STL(endog, period=None, seasonal=7, trend=None,
low_pass=None, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False,
seasonal_jump=1, trend_jump=1, low_pass_jump=1)`, `fit(inner_iter=None,
outer_iter=None)` (`tsa/stl/_stl.pyx`, src); `MSTL(endog, *, periods=None,
windows=None, lmbda=None, iterate=2, stl_kwargs=None)` (`tsa/stl/mstl.py`,
src); `seasonal_decompose(x, model='additive', filt=None, period=None,
two_sided=True, extrapolate_trend=0)`; `STLForecast` (src). Catalogues:
statsforecast `MSTL`, sktime `STLForecaster`, `StatsForecastMSTL` (web).
OnlineSTL: Mishra, Sriharsha and Zhong, PVLDB 15(7), 2022, arXiv 2107.09110
(web).

**What "exact after a delay" can mean.** STL's value at a row depends on the
whole series it was given: each inner pass widens the reach, and the robust
weights use the median remainder of every row. So STL over a trailing window,
read `h` rows back, is **not** the full-history STL at that row at any `h`. It
**is** statsmodels' STL run on exactly the window's rows, read at the same
position. That is the oracle the plan names, and it is the only exact one. How
far the window's value sits from the full-history value, as a function of the
window and `h`, is a measurement for the build to make and the docstring to
state.

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| `STL`, `robust=False`: trend, seasonal and remainder of row `t − h` from a window of `W` rows | D. Without robustness the whole of STL is a linear map of the window, and the read position is fixed, so each component is one fixed kernel of length `W`: a dot product per row. The kernels depend only on the configuration and are computed once per spec | no: statsmodels' `STL` takes no observation weights, so a decayed STL has no oracle | this is the window form | `STL(window, ...).fit()` at position `W − 1 − h`, about 1e-12 relative (a linear-algebra identity, not the same order of operations) |
| `period`; `seasonal`, `trend`, `low_pass` (odd spans; defaults `7`, the smallest odd above `1.5·period/(1 − 1.5/seasonal)`, the smallest odd above `period`) | D: configuration of the kernel | — | — | same |
| `seasonal_deg`, `trend_deg`, `low_pass_deg` ∈ {0, 1} | D | — | — | same |
| `seasonal_jump`, `trend_jump`, `low_pass_jump` (fit every `j`-th point, interpolate) | D: still linear; reproduce the interpolation to match | — | — | same |
| `inner_iter` (default 5, or 2 when robust) | D | — | — | same |
| `robust=True`, `outer_iter` (default 15): bisquare weights from `6·median|remainder|` over the series (`_stl.pyx:631-655`, src) | D, but nonlinear: a full STL over the window for each emission, about `W·(spans)·inner·outer` operations. Affordable only at a cadence counted in learned rows (the `pca_every` device, which keeps chunk invariance) | no | the window form | `STL(window, robust=True).fit()`, including `weights`; 1e-10 if the port follows the same order of operations |
| provisional values for the newest `h` rows | D with `h = 0`: the window's own end values, which batch STL fits one-sided. One more kernel per lag emitted | no | — | the same `STL` at positions past `W − 1 − h`. Exact, unlike OnlineSTL |
| `MSTL(periods, windows, iterate=2, stl_kwargs)`; default `windows = 7 + 4·(1..n)` | D: nested STL passes, linear when not robust, so one kernel per seasonal component and one for the trend | no | — | `MSTL(window, ...).fit()` component by component |
| `MSTL(lmbda=float)` (Box-Cox first) | D: a per-row transform, then nonlinear only through it; the kernels apply to the transformed ring | no | — | `MSTL(lmbda=)` |
| `MSTL(lmbda='auto')` | N at reasonable cost: a likelihood maximization over the window per emission | — | — | — |
| `period=None` (inferred from a pandas index) | n/a: rheon requires it | — | — | — |
| `seasonal_decompose(two_sided=True)`: a centred moving-average trend, per-phase means | D: the trend is exact at a delay of `period/2` rows; the seasonal is a mean over the given rows, so on a window it is exact against the window | [L] (weights by hand) | yes [T] | `seasonal_decompose(window, model=, period=, filt=)`; additive and multiplicative |
| `seasonal_decompose(two_sided=False)` | E on a window: a trailing moving average, no delay | [L] | yes [T] | same |
| `seasonal_decompose(extrapolate_trend=)` | D: a line fit at the ends of the window | — | yes [T] | same |
| `STLForecast` | composition: a model on the deseasonalized series. Only the delayed components may feed a model, which the embargo enforces | — | — | not in the first build |
| OnlineSTL | A, and not STL: one-sided tri-cube filters over the last `4m` points and exponential smoothing (`γ = 0.7`) of the seasonal, `O(k·m)` per point, initialized by a batch STL on the first `4m` points; no robust option; the paper does not claim to reproduce STL, and a later paper reports its quality trails batch methods (web) | — | — | no Python library computes it, so a longhand only. Propose leaving it out: the window's own end values serve the same purpose with an exact oracle |

**Oracle.** statsmodels `STL` and `MSTL` on the same window, component by
component, at the delayed position: 1e-12 for the kernel form, 1e-10 for the
robust port.

**Hard parts and risks.**
- *Loess under an embargo.* The delay does not make the value final; it makes
  it the window's. State that, and measure the gap to the full-history value.
- *The port.* statsmodels' `_stl.pyx` follows Cleveland's Fortran. The robust
  path needs a faithful port, including the interpolation under `*_jump` and
  the end rules of the low-pass filter; the kernel path needs either `W` unit
  impulses through that port at setup (`O(W²·spans)`: by estimate about a second at
  `W = 672`, minutes at `W = 10⁴`) or the transposed passes (`O(W·spans)`).
- *Nulls and gaps.* statsmodels' `STL` takes no missing values. A null inside
  the window breaks the fixed kernel, and a loess with holes has no Python
  oracle. Propose: the window is by position, a clock break empties it, and
  the output is null until `W` rows are held.
- *Cost.* A dot product of `W` per component per row (an estimate, not measured: about 0.2 µs at
  `W = 672`), against roughly 100 ns for a whole regression row today. The
  robust form is three to four orders more per emission. State both.
- *Minimum window.* Two full periods of the largest season, and a few more for
  the seasonal smoother's span (`seasonal = 7` reaches three periods each
  way).

**State.** `8·W` bytes per group and target (5.4 KB at `W = 672`, 46 KB at
`W = 5760`), under the existing window cap. The kernels are per spec:
`8·W·(1 + seasons + lags emitted)`.

## 7. Task 491: spectral analysis

Libraries: `scipy.signal` 1.18.1 (src): `periodogram(x, fs=1.0,
window='boxcar', nfft=None, detrend='constant', return_onesided=True,
scaling='density', axis=-1)`; `welch(x, fs=1.0, window='hann_periodic',
nperseg=None, noverlap=None, nfft=None, detrend='constant',
return_onesided=True, scaling='density', axis=-1, average='mean')`; `csd`,
`coherence` (the same parameters over a pair); `spectrogram(..., window=
('tukey_periodic', 0.25), mode='psd')`; `stft(..., nperseg=256, detrend=False,
boundary='zeros', padded=True, scaling='spectrum')`; `ShortTimeFFT(win, hop,
fs, *, fft_mode='onesided', mfft=None, dual_win=None, scale_to=None,
phase_shift=0)`; `lombscargle(x, y, freqs, *, precenter (deprecated),
normalize=False, weights=None, floating_mean=False)`; `czt`, `zoom_fft`;
`get_window` and 25 windows. Note the default window names are
`'hann_periodic'` and `'tukey_periodic'` in this version. scipy has no Goertzel
function. Catalogue: darts `FFT` forecaster (web).

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| sliding DFT at chosen bins `k` of a window of `N` rows | E in exact arithmetic, `O(1)` per bin per row over a ring of `N`. In floating point the textbook rotation drifts without bound; the accumulate-and-subtract form does not multiply its error, and an exact recompute from the ring every `N` rows bounds it | see the next row | this is the window form | `ShortTimeFFT(win=boxcar(N), hop=1, fs).stft(x)` columns, or `stft(..., boundary=None, padded=False, noverlap=N−1)`; phase convention via `phase_shift`. Tolerance about `N·ε·max|x|` with the recompute; to be measured |
| the DFT at fixed angular frequencies with exponential decay: `X(ω) = Σ λ^(age) x e^(−iω·clock)` | E: one complex accumulator per frequency, no subtraction | yes [T]: `periodogram(x, window=<the decay's amplitude weights as an array>, detrend=False)` on an even clock, or `czt` for off-grid frequencies | n/a | 1e-12 relative on an even clock |
| off-grid frequencies (`nfft > N`, arbitrary `ω`; the generalized Goertzel) | E: the same recursion with a complex correction on the leaving sample | as above | yes | `czt` / `zoom_fft` over the window, or `periodogram(nfft=)` |
| `window`: `boxcar` | E | — | yes | above |
| `window`: `hann`, `hamming`, `blackman` and other cosine sums | E: a fixed combination of the neighbouring bins `k ± 1` (`± 2`), so three or five bins are tracked per frequency | — | yes | `ShortTimeFFT(win=get_window(...), hop=1)` |
| `window`: any other (`tukey`, `kaiser`, `dpss`, ...) | E at a hop: a direct windowed DFT from the ring every `hop` learned rows, `O(N·K/hop)` per row | — | yes | `ShortTimeFFT(win, hop)` at those rows |
| `periodogram`: `scaling` ∈ {density, spectrum}, `return_onesided`, `fs` | E: constants from `Σw²` and `Σw` | yes [T] | yes [T] | `periodogram` on the window |
| `detrend='constant'` (the default) | E: on-grid with a boxcar it changes only bin 0; with another window, subtract the window mean times the window's own transform at the bin (a constant) | [L]: scipy removes the unweighted mean, so the decayed form needs its mean stated | yes [T] | `periodogram(detrend='constant')` |
| `detrend='linear'` | E: the window's least-squares line from sliding sums, and the transform of a windowed ramp | [L] | yes [T] | `periodogram(detrend='linear')`; more algebra, propose second |
| `detrend=<callable>` | N | — | — | — |
| `welch`: `nperseg`, `noverlap`, `average='mean'` | E: every `nperseg − noverlap` learned rows fold one segment's periodogram into a mean. scipy drops a trailing partial segment, as a stream does | yes: each segment's periodogram is scipy's (`ShortTimeFFT.spectrogram`), and the decay's weighted mean across segments is written by hand | yes [T]: the mean over the last `M` segments, a ring of `M·K` floats | `welch` on the rows so far, 1e-12 |
| `welch(average='median')` | E in a window of segments (scipy's bias factor included); N over all history at `O(1)` state | no | yes [T] | `welch(average='median')` on the window |
| `csd`, `coherence` (a pair of series); phase lead | E: the same sums over a pair | as `welch` | as `welch` | `csd`, `coherence` |
| `spectrogram(mode=)`: `psd`, `complex`, `magnitude`, `angle` | E from the tracked bins | — | yes | `spectrogram` |
| `spectrogram(mode='phase')` (unwrapped) | E: one float per bin of state | — | yes | `spectrogram(mode='phase')` |
| `stft(boundary=)`: `zeros`, `even`, `odd`, `constant`; `padded` | D: a centred segment needs `N/2` later rows. With `boundary=None, padded=False` the segments trail and nothing is delayed | — | — | `stft(boundary=None, padded=False)` |
| `istft`, `dual_win`, `check_COLA`, `check_NOLA` | N in this task: reconstruction | — | — | — |
| `lombscargle(x, y, freqs)` on an uneven clock | E: per frequency it is a weighted least-squares fit of a sinusoid, so about nine running sums per frequency | yes [T]: `lombscargle(weights=<the decay's weights>)` | yes [T]: the sums subtract, so the existing snapshot ring serves; scipy on the window's rows | 1e-10 relative; the phase is taken from `clock − origin` |
| `lombscargle(floating_mean=)`, `normalize` ∈ {power, normalize, amplitude} | E: a three-parameter or two-parameter fit; the normalizations are functions of the same sums | yes [T] | yes [T] | same |
| dominant frequency, band power, spectral entropy | E at emit from the tracked bins | as the bins | as the bins | numpy over scipy's spectrum |
| a spectrum from AR coefficients | E at emit (section 2) | — | — | `arma_periodogram` |
| the whole-history periodogram on the DFT grid of a growing `N` | N: the grid moves with `N`. Fixed frequencies are the online form | — | — | — |
| darts `FFT` forecaster (keep the top frequencies of the whole series) | N; the online analogue is a regression on 489's terms | — | — | catalogue only |

**Oracle.** `scipy.signal` on the same window or the same weighted rows. Three
of the four quantities (decayed periodogram, Welch, Lomb-Scargle) have a
third-party oracle under decay as well.

**Hard parts and risks.** (1) Drift of the sliding DFT: choose the stable form
and the recompute, and hold it with a long-run test (10⁷ rows against a fresh
transform). (2) Cost grows with the frequencies kept: `O(K)` per row for the
sliding DFT; for Lomb-Scargle a `sin` and a `cos` per frequency per row, tens
of nanoseconds each, so sixteen frequencies cost about ten ordinary model rows.
(3) libm again: sums of `sin` and `cos` results sit in the state, so the
cross-OS ulp test applies to those fields. (4) Phase conventions differ between
`stft`, `ShortTimeFFT` and a sliding DFT; fix one and state it. (5) The sliding
DFT is by position: a clock break empties the ring.

**State.** Sliding DFT: `8·N + 16·K` (three times the bins for a cosine-sum
window). Welch: `8·nperseg + 8·K`, plus `8·M·K` for a window of `M` segments.
Lomb-Scargle: about `72·F` bytes per target, and that much per snapshot under a
window.

## 8. Task 492: the GARCH family

Library: `arch` (web; PyPI 9.0.0, NCSA licence, Python 3.10+).
`arch_model(y, x=None, mean='Constant', lags=0, vol='GARCH', p=1, o=0, q=1,
power=2.0, dist='normal', hold_back=None, rescale=None)`; `mean` ∈ {Constant,
Zero, LS, AR, ARX, HAR, HARX}; `vol` ∈ {GARCH, ARCH, EGARCH, FIGARCH, APARCH,
HARCH}; `dist` ∈ {normal, t, skewt, ged}. Classes: `ConstantVariance()`,
`ARCH(p=1)`, `GARCH(p=1, o=0, q=1, power=2.0)`, `EGARCH(p=1, o=0, q=1)`,
`FIGARCH(p=1, q=1, power=2.0, truncation=1000)`, `APARCH(p=1, o=1, q=1,
delta=None, common_asym=False)`, `HARCH(lags=1)`, `MIDASHyperbolic(m=22,
asym=False)`, `EWMAVariance(lam=0.94)` (`None` estimates it),
`RiskMetrics2006(tau0=1560, tau1=4, kmax=14, rho=1.414...)`,
`FixedVariance(variance, unit_scale=False)`; the default branch's source also
defines `FIAPARCH` and `NGARCH` (check they are in the installed release). Mean
classes: `ZeroMean`, `ConstantMean`, `ARX`, `HARX(lags, constant, use_rotated)`,
`LS`, `ARCHInMean`. Distributions: `Normal`, `StudentsT`, `SkewStudent`,
`GeneralizedError`. `fit(update_freq=1, disp='final', starting_values=None,
cov_type='robust', show_warning=True, first_obs=None, last_obs=None, tol=None,
options=None, backcast=None)` (SLSQP); `fix(params, first_obs, last_obs)`;
`forecast(params=None, horizon=1, start=None, align='origin',
method='analytic', simulations=1000, rng=None, random_state=None, *,
reindex=False, x=None)`. Catalogues: sktime `ARCH`, `StatsForecastGARCH`,
`StatsForecastARCH`; statsforecast `GARCH`, `ARCH` (web). river has none (src).

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| the variance recursion with **fixed parameters**: `GARCH(p, o, q)` with `power=2` (GARCH, GJR), `power=1` (TARCH, AVGARCH), `ARCH(p)` | E: a ring of the last `max(p, o)` residuals and `q` variances | n/a: parameters given | n/a | `GARCH(...).compute_variance(parameters, resids, sigma2, backcast, var_bounds)` with the backcast passed in and bounds that do not bind: 1e-13 relative, near the bit |
| `GARCH(power=)` other than 1 or 2; `APARCH(delta, common_asym)` | E, with `powf` on every row | n/a | n/a | the class's `compute_variance`; 1e-12, and the state now holds a libm result |
| `EGARCH(p, o, q)` | E: the log variance is the state (libm `exp` and `ln` each row). arch caps the log variance and re-clips through its bounds | n/a | n/a | `EGARCH.compute_variance`; 1e-12 on one OS; across OSes the recursion contracts (`Σβ < 1`), so last-bit differences should stay a few ulps: measure it, and bring it to the user if it does not |
| `FIGARCH(p, q, power, truncation=1000)`, `FIAPARCH` | E: an ARCH(∞) cut at `truncation` lags, so a ring of `truncation` squared residuals and `O(truncation)` work per row | n/a | n/a | `FIGARCH.compute_variance`; 8 KB per group and target at the default: state it |
| `HARCH(lags)` | E: sliding sums over a ring of `max(lags)` | n/a | n/a | `HARCH.compute_variance` |
| `MIDASHyperbolic(m=22, asym)` | E: a ring of `m`, `O(m)` per row | n/a | n/a | `MIDASHyperbolic.compute_variance` |
| `EWMAVariance(lam)` | E, and it exists: an exponentially weighted variance about zero on a row clock | is the decay | — | `EWMAVariance.compute_variance`; also `rp.ewm_var` |
| `RiskMetrics2006(tau0, tau1, kmax, rho)` | E: `kmax` EWMAs with fixed weights, no parameter | n/a | n/a | its `compute_variance` |
| `ConstantVariance`, `FixedVariance(variance, unit_scale)` | E: a running variance; a given variance column (the existing `weight`) | yes | yes | arch, and what rheon has |
| the start of the recursion: `backcast` | arch's default is a 0.94-weighted mean of the first `min(75, n)` squared residuals: it reads ahead of row 0. Three forms: (a) `backcast=` given, E; (b) buffer 75 rows, then replay them, E with 75 null rows; (c) rheon's own start from the residuals so far, which meets arch's path at rate `(α + β)^t` | — | — | (a) is the oracle path; (c) agrees to 1e-9 only after about 2,000 rows at a persistence of 0.99 |
| the variance bounds | arch clips each variance to bounds built from the **whole sample** (an EWMA scaled by `1e±6`, a floor of the sample variance over `1e8`). They bind only on pathological inputs; an online recursion cannot reproduce a clip that depends on later rows | — | — | the test asserts no bound binds, or passes infinite bounds |
| mean `Zero`, `Constant` | E for fixed parameters; estimated: a running mean (or jointly, below) | yes | — | `ZeroMean`, `ConstantMean` |
| mean `AR`, `ARX`, `LS` | two-step: 486's `ar` or `ewridge` for the mean, optionally weighted by `1/σ²` (the variance for the row is known before it, so it is a legal weight), then the variance on its out-of-sample residuals. arch estimates them jointly by likelihood, so coefficients differ in finite samples | as 486 | as 486 | fixed parameters: `ARX(...).fix(params)`. Estimated: statistical |
| mean `HAR`, `HARX(lags, use_rotated)` | E for the regressors: means of the last 1, 5, 22 values from a ring (overlapping, or `use_rotated` for non-overlapping) | as 486 | as 486 | `HARX` regressors exactly; coefficients as above |
| `ARCHInMean` | fixed parameters: E, since the row's variance is known before the row. Jointly estimated: A | — | — | `ARCHInMean.fix` |
| `dist`: normal, Student's t, skewed t, GED, with **fixed** shape | E: the shape changes the log-likelihood and the quantiles, not the variance path. Needs `ln Γ`, which Rust's standard library does not offer on stable: a new dependency or an own implementation, to raise under the linking rule | n/a | n/a | `ARCHModelFixedResult.loglikelihood`, 1e-10 |
| estimating the shape (`ν`, `η`, `λ`) | A: a small grid of shapes each with a running log-likelihood (exact per member, the choice is an argmax), or a score step. A moment estimate from the kurtosis is cheap and poor below `ν = 8` | grid: yes, each member | no | each member against arch at that shape; the choice against arch's fitted shape, to the grid's spacing |
| **estimating the variance parameters** (arch `fit`: batch quasi-likelihood by SLSQP) | A: recursive quasi-maximum likelihood, a Gauss-Newton step on the Gaussian score with the sensitivities `∂σ²/∂θ` carried exactly, projected onto positivity and stationarity at every step. Variants in the literature (mem): Kierkegaard et al. 2000; Aknouche and Guerbyenne 2006; Gerencser, Orlovits and Torma 2010; Hendrych and Cipra 2018 (self-weighted, robust); Werge and Wintenberger 2022 (AdaVol, adaptive steps and projection) | yes, weakly: the weighted Gaussian quasi-likelihood written by hand and maximized by `scipy.optimize`, with arch's unweighted fit beside it at no decay | no: the likelihood is not a difference of sums. A refit over the window is `O(window × iterations)` per row; possible only at a cadence with an optimizer in the core, a separate and larger piece | arch `fit()` at the last row. Proposed gates, to be measured: on simulated GARCH(1,1) with persistence at most 0.98, parameters within 2 batch standard errors after 10,000 rows; the conditional variance within 5% (median relative) of the batch-parameter path after 5,000 rows; QLIKE within 1e-3; fifty seeds |
| variance targeting (`ω` from the running variance) | A: one parameter fewer and a steadier recursion. arch's `GARCH` has no targeting switch (web, not confirmed in source) | [L] plus optimizer | no | the definition and `scipy.optimize` |
| constraints | held at every step by projection: `ω > 0`; `α, β ≥ 0`; `α + γ/2 + β < 1` (GJR); EGARCH `Σβ < 1`; FIGARCH `0 ≤ d ≤ 1`, `0 ≤ φ ≤ (1 − d)/2`, `0 ≤ β ≤ d + φ`; APARCH `δ` in (0.05, 4), `|γ| < 1` (arch's `bounds()` and `constraints()`, web) | — | — | a property test at every row |
| `cov_type`: `robust` (Bollerslev-Wooldridge), `classic` | A: from the estimator's two matrices | [L] | no | arch `std_err`, statistical |
| `forecast(horizon, method='analytic')` | E for GARCH, GJR, ARCH, HARCH, MIDAS, EWMA, RiskMetrics at `power=2`: the recursion toward the long-run level, `O(h·(p+q))` | n/a | n/a | `forecast(params=, horizon=h).variance`, 1e-12 |
| `forecast` for EGARCH, APARCH, FIAPARCH, TARCH (`power≠2`), FIGARCH at `power≠2`, past `h = 1` | arch raises for `analytic` and offers only `simulation` and `bootstrap`: randomness on the output path | — | — | `h = 1` only. EGARCH's expected log variance has a closed form but no library oracle |
| `method='simulation'`, `'bootstrap'`, `simulations`, `rng` | N (random) | — | — | — |
| outputs: conditional variance before the row, standardized residual, log-likelihood | E given parameters | — | — | `conditional_volatility`, `std_resid`, `loglikelihood` of `fix(params)` |
| `rescale`, `hold_back`, `first_obs`, `last_obs`, `update_freq`, `disp`, `tol`, `options`, `starting_values` | n/a or configuration (a warm start). `rescale` exists because the optimizer is scale-sensitive: the recursive step must be scale-equivariant, held by an equivariance test | — | — | — |
| IGARCH, component GARCH, realized GARCH (R's `rugarch`) | E as recursions, but no Python oracle | — | — | not built until one exists |

**Oracle.** Fixed parameters: each `arch` class's `compute_variance` with the
backcast given, 1e-13 to 1e-12. Estimation: the batch fit at the last row,
statistical, on the parameters and, more usefully, on the variance path and
QLIKE.

**Hard parts and risks.**
- *Likelihood estimation online.* The GARCH likelihood is nearly flat along
  `α + β = const`, the Gaussian score has the fourth power of the return in its
  variance, and `ω` is weakly identified. A recursive step therefore wanders
  along the ridge and jumps on outliers. Known mitigations: variance targeting,
  self-weighting or a Student-t score, a projection with a margin, and a
  warm-up on a short buffered batch. The theory gives the batch estimate's
  limit distribution (mem); nothing here says how many rows that takes on real
  returns. Measure first (section 11, step 0).
- *The start and the bounds*, above: without a given backcast there is no
  bit-level oracle before the transient dies.
- *libm in a feedback loop* (EGARCH, APARCH): the 2026-11-20 rule's cross-OS
  ulp test applies to a state that feeds itself.
- *Weights and the clock.* A GARCH step is one row. On an uneven clock the
  recursion has no library meaning; state the row-step rule. A zero-weight row
  needs the decision in section 12.
- *`arch` as a dev dependency:* check it publishes wheels for CPython 3.14 on
  macOS arm64 and Windows (CI's macOS leg runs 3.14; `hmmlearn` already builds
  from source there).

**State.** GARCH(1,1) filter: 24 to 48 bytes per target. With estimation:
`8·(n + n² + n·q)` for `n = 1 + p + o + q`, 120 bytes at (1,1). FIGARCH: 8 KB.
HARCH and MIDAS at 22 lags: 176 bytes. A shape grid of `G` members adds `8·G`.

## 9. Task 493 (and 514): DCC and ADCC

Libraries: `arch` has no multivariate model (web). `pymgarch` 0.6.0 (MIT,
Python 3.11+, built on `arch`; CCC, DCC(1,1), ADCC, GO-GARCH, copula-GARCH,
scalar and diagonal BEKK; Gaussian and Student-t; correlation targeting;
two-stage quasi-likelihood; `method="composite"`; a `filter()` for fixed
parameters; claims replication tests against `rmgarch`) and `mvgarch` 2.0.2
(MIT, a `DCCGARCH` class, Linux classifier only) and `mgarch` 0.3.0 exist on
PyPI (web, PyPI pages only). R's `rmgarch` and `tsmarch` are the reference
implementations. Definitions (mem): Engle 2002; Cappiello, Engle and Sheppard
2006 (ADCC); Aielli 2013 (cDCC); Pakel, Shephard, Sheppard and Engle 2021
(composite likelihood).

```
Q_t = (1 − a − b)·Q̄ + a·z_{t−1}z_{t−1}ᵀ + b·Q_{t−1}            R_t = diag(Q_t)^(−1/2) Q_t diag(Q_t)^(−1/2)
ADCC:  Q_t = (Q̄ − a·Q̄ − b·Q̄ − g·N̄) + a·zzᵀ + b·Q + g·nnᵀ      n = z·1[z < 0],  N̄ = E[nnᵀ]
second-stage log-likelihood per row:  −½ (ln|R_t| + z_tᵀR_t⁻¹z_t − z_tᵀz_t)
```

| feature or parameter | class | decay | window | oracle and tolerance |
|---|---|---|---|---|
| the `Q` recursion with given `(a, b)`, `Q̄` and a start (task 514) | E: `O(k²)` state and work per row | n/a: parameters given | n/a | the recursion from Engle (2002) in numpy [L]; `pymgarch`'s `filter()` as a second opinion if it accepts a given `Q̄` and start (not yet read). 1e-12 |
| ADCC's `g` and `N̄` | E | n/a | n/a | the same, from Cappiello, Engle and Sheppard |
| cDCC (Aielli's correction) | E: one rescaling of `z` per row | n/a | n/a | [L]; `pymgarch` does not list it |
| `R_t`, the covariance `D_t R_t D_t`, pair correlations | E; the output is `k(k−1)/2` fields, so a `pairs=` selection as `ew_cov` has | — | — | above |
| the per-row log-likelihood | E: a Cholesky of `R_t` per row, `O(k³)` (about 10 µs at `k = 50`), and `ln` | — | — | `scipy.stats.multivariate_normal.logpdf` on the dense `R_t`, as `deco`'s `loglik` is held today |
| correlation targeting: `Q̄` from the data | batch targeting uses the whole sample's correlation of `z`. Online: (a) a given `Q̄`, E; (b) the running or decayed mean of `zzᵀ` over earlier rows, which is rheon's own definition and meets the batch one only in the limit | (b) is the decayed form, [L] | (b) on a window: `Q̄` subtracts, the `Q` recursion does not | (a) is the oracle path |
| estimating `(a, b)` and `g` (two-stage: on the standardized residuals of 492) | A, two routes: (a) a grid of `(a, b)` members, each an exact recursion with a running log-likelihood, the choice an argmax, coarse then fine; (b) a recursive Gauss-Newton step with `∂Q/∂a`, `∂Q/∂b` carried (two or three more `k×k` matrices). For `k` above about 10, the composite likelihood over adjacent pairs makes the score `O(k)` instead of `O(k³)` and avoids the known large-`k` bias of the full likelihood (mem) | (a) each member exactly; (b) weakly: a weighted likelihood by hand through `scipy.optimize` | no, as GARCH | the definition's batch maximum by `scipy.optimize` on the same rows (PLAN 493), and `pymgarch` or `rmgarch` beside it. Proposed, to be measured: `(a, b)` within 2 batch standard errors after 5,000 rows; correlations within 0.02 of the batch-parameter path after the burn-in. Route (a)'s tolerance is the grid's spacing |
| constraints | `a, b ≥ 0`, `a + b < 1` keep `Q_t` positive definite. ADCC needs `a + b + δ·g < 1` with `δ` the largest eigenvalue of `Q̄^(−1/2) N̄ Q̄^(−1/2)`, which moves with an online `Q̄` and `N̄`: an eigen-solve at a cadence | — | — | a property test: `Q_t` positive definite at every row |
| Student-t DCC | fixed degrees of freedom: E (a different per-row likelihood); estimated: the shape grid of section 8 | — | — | `pymgarch`, unvetted |
| multi-step correlation forecast | the usual approximation `R_{t+h} ≈ R̄ + (a + b)^(h−1)(R_{t+1} − R̄)` (mem) | — | — | [L] only |
| CCC | E: the running correlation of `z`, which `ew_cov` gives | yes | yes | numpy; `pymgarch` |
| DECO on the DCC `Q` | E: the mean off-diagonal of `R_t`; makes `deco` the true DECO-DCC | — | — | R's `dccmidas` (ORACLE-COVERAGE §3b names it) |
| GO-GARCH, copula-GARCH, BEKK | not asked for; batch estimation | — | — | catalogue only |

**Oracle.** The recursion by hand from the papers and a batch maximum by
`scipy.optimize`, as PLAN 493 says. `pymgarch` is a possible third-party second
opinion once its `Q̄`, start and recursion are read against the papers; it is
three months old, so it should not be the only thing a test rests on.

**Hard parts and risks.** (1) `O(k³)` per row for the full likelihood; the
composite form or the grid bounds it. (2) Targeting with a moving `Q̄` is not
the batch estimator: say so. (3) The model owns `k` univariate GARCH states
(as `ew_class` owns one `EwCov` per class), so every risk of 492 is inside it.
(4) ADCC's admissible set moves with the data.

**State.** `k` GARCH states, plus `Q` and `Q̄` at `8·k(k+1)/2` each, `N̄` for
ADCC, and two or three `k×k` sensitivities: about 2.5 KB at `k = 10`, 60 to
120 KB at `k = 50`, about 2 MB at `k = 200` per group. A grid of `G` members
holds `G` copies of `Q`. Bring the `k = 200` figure to the user.

## 10. What the catalogues add

| library | what it adds beyond the above | disposition |
|---|---|---|
| river 0.26.1 `time_series` (src) | `SNARIMAX`, `HoltWinters`, and horizon metrics. `SNARIMAX` is stochastic gradient descent on lagged values and lagged errors with any regressor: a different estimator from every form above | a behavioural reference only; it cannot hold a number |
| sktime (web) | `ARDL`, `ARARForecaster`, `Croston`, `TSB`, `BATS`, `TBATS`, `UnobservedComponents`, `DynamicFactor`, `VAR`, `VARMAX`, `VECM`, `FourierFeatures`, `UpdateEvery` and `UpdateRefitsEvery` (refit on a cadence) | Croston and TSB are exactly online (section 4). `VAR`, `VARMAX`, `VECM`, `DynamicFactor`, `UnobservedComponents` belong to 494 and 495; with fixed parameters each is a Kalman filter |
| darts (web) | `KalmanForecaster` (subspace identification, batch), `FFT`, `FourTheta`, the naive baselines (`NaiveSeasonal`, `NaiveDrift`, `NaiveMovingAverage`) | the naive baselines are exactly online and nearly free: a ring of one season. Worth a row in 488 |
| statsforecast (web) | `AutoARIMA`, `AutoETS`, `AutoCES`, `AutoTheta`, `MSTL`, `GARCH`, `ARCH`, `ADIDA`, `IMAPA`, window averages | `GARCH(p, q)` is a second batch opinion for the normal case; `MSTL` a second for 490 |
| pmdarima 2.1.1 (web) | `auto_arima`'s search and its unit-root and seasonal tests | not online (section 3) |
| statsmodels `DynamicFactor(endog, k_factors, factor_order, exog, error_order, error_var, error_cov_type, enforce_stationarity)` (src) | a factor state-space model | fixed parameters: a filter. Estimation is batch. Not in 486 to 493; note for 494 |

## 11. A proposed build order

The plan's order is 489, 486, 488, 487, 490, 491, 492, 493. The dependencies
are: 487 needs 486; 492's AR and HAR means need 486; 493 needs 492 and 514.
Nothing else depends on anything. Two lanes fit the two-worker limit and put
the hardest work first where the dependencies allow.

**Step 0, before any Rust: a measurement study in numpy** (2 to 3 days, one
worker, no build). Recursive quasi-likelihood for GARCH(1,1) and GJR against
`arch`, and recursive prediction error for ARMA(1,1) and MA(2) against
statsmodels, fifty seeds, a grid of row counts, with and without variance
targeting and a robust score. It sets the tolerances this record could only
propose, and it decides whether the recursive estimators ship at all or the
grid forms stand in. Everything in class A waits on it.

| order | task | why here | state per group and target | effort |
|---|---|---|---|---|
| A1 | 492, the GARCH family | the hardest, and independent with a zero or constant mean. Fixed-parameter recursions first (exact oracle), then the estimator from step 0, which 487, 488 and 493 reuse | 24 to 48 B filter; 120 B with estimation at (1,1); 8 KB FIGARCH | very large: 10 to 14 days (eleven processes, four distributions, the estimator, forecasts) |
| A2 | 514, then 493, DCC and ADCC | needs A1 | 2.5 KB at `k = 10`; about 100 KB at `k = 50`; 2 MB at `k = 200` | large: 6 to 8 days |
| A3 | 487, ARMA, ARIMA, SARIMAX | needs B2 and A1's estimator; a new filter, the Lyapunov start, differencing, seasonal polynomials | 0.3 KB at ARMA(2,1); 3 KB at `s = 24`; 116 KB at `s = 168`; capped near 263 KB | very large: 10 to 14 days |
| B1 | 489, Fourier terms | smallest, exact, and 486 and 488 use it | none | small: 1 to 2 days |
| B2 | 486, `ar` | exact; unlocks A3 and A1's AR and HAR means | `8·max(lags)` plus the Gram, about 0.6 KB at `k = 11` | medium: 3 to 4 days |
| B3 | 490, STL and MSTL | independent and the hardest of the exact lane: the port, the kernels, the robust path | `8·W`: 5.4 KB at `W = 672`, 46 KB at `W = 5760` | large: 7 to 9 days |
| B4 | 491, spectral | independent, exact, several quantities | `8·N + 16·K`; about `72·F` for Lomb-Scargle | large: 6 to 8 days |
| B5 | 488, Holt-Winters and ETS | fixed-parameter forms are exact; learned parameters reuse A1's estimator or the existing `emit_selected` | `8·(2 + m)`: 0.4 KB at `m = 24`, 2.7 KB at `m = 168` | medium to large: 5 to 7 days |

Each row also carries the standing costs the plan lists: a new `ModelState`
variant is a layout change (rule 5: fixtures copied, a loader, regenerated
sets), every place `docs/EXTENDING.md` names, the CLI config, the docs, chunk
invariance, rules 8 and 9, and the measured memory and speed. Eight models are
eight schema bumps unless a wave batches them.

Against today's caps: a window ring is refused past 256 MiB by default, and an
`ew_cov` over 20 columns with a 1,000-row window holds about 3 MB per group.
Every state above is small beside that except a long-season SARIMA covariance
and DCC at large `k`, which is why those two are flagged for the user.

## 12. Decisions this scan could not make

1. **A zero-weight row in a time-series model.** Today it advances the clock
   and learns nothing, and a lag counts learned rows. For a series model that
   means the row's value is not a lag and does not drive the variance
   recursion, which matches statsmodels' `missing='drop'` for 486 and a
   skipped update for the state-space forms. A user may instead mean "observe
   the value, hold the parameters still". The two give different numbers.
2. **Row steps against the clock** for lags, phases and the GARCH step
   (section 1, item 1): by position with a break emptying the ring, or
   bucketed from the clock.
3. **Caps to bring to the user:** the SARIMA filter's covariance past about
   `r = 256`, FIGARCH's 8 KB ring, DCC past about `k = 50`.
4. **New dependencies:** `arch` in the dev group (and `pymgarch` if it is
   vetted); a `ln Γ` for the Student-t and GED likelihoods in the core, under
   the linking rule.
5. **Whether class A ships where the grid form exists.** For ETS parameters,
   the distribution's shape and DCC's `(a, b)`, a grid of exact members with a
   running loss is deterministic and fully checkable; the recursive step is
   cheaper and only statistically checkable. Step 0's numbers should decide.
6. **`half_life=inf` on multiplicative ETS**: refuse it until an oracle
   exists, or ship it on a longhand.

## 13. Sources

Installed, read as source or signature (statsmodels 0.15.0, scipy 1.18.1, river
0.26.1, under `.venv/lib/python3.14/site-packages/`):
`statsmodels/tsa/ar_model.py`; `tsa/arima/model.py`; `tsa/arima/estimators/`;
`tsa/statespace/sarimax.py`, `mlemodel.py`, `structural.py`,
`dynamic_factor.py`, `exponential_smoothing.py`, `initialization.py`;
`tsa/holtwinters/model.py`, `results.py`; `tsa/exponential_smoothing/ets.py`,
`initialization.py`; `tsa/stl/_stl.pyx`, `mstl.py`; `tsa/seasonal/`;
`tsa/deterministic.py`; `tsa/stattools/_stattools.py`; `tsa/_innovations`,
`tsa/innovations/arma_innovations.py`; `tsa/arima_process.py`;
`tsa/forecasting/theta.py`, `stl.py`; `stats/diagnostic.py`; `scipy/signal/`
(`periodogram`, `welch`, `csd`, `coherence`, `spectrogram`, `stft`,
`ShortTimeFFT`, `lombscargle`, `czt`, `zoom_fft`, `windows`);
`river/time_series/`.

Fetched 2026-11-20:
- arch: <https://arch.readthedocs.io/en/latest/univariate/introduction.html>,
  `.../univariate/volatility.html`, `.../univariate/mean.html`,
  `.../univariate/distribution.html`,
  `.../univariate/generated/generated/arch.univariate.base.ARCHModel.fit.html`,
  `.../arch.univariate.base.ARCHModelResult.forecast.html`,
  `.../univariate/generated/arch.univariate.HARX.html`,
  `.../univariate/univariate_volatility_forecasting.html`; source
  <https://raw.githubusercontent.com/bashtage/arch/main/arch/univariate/volatility.py>
  and `.../recursions_python.py`; <https://pypi.org/pypi/arch/json> (9.0.0).
- <https://pypi.org/project/pymgarch/> (0.6.0), <https://pypi.org/project/mvgarch/> (2.0.2).
- sktime: <https://www.sktime.net/en/stable/api_reference/forecasting.html>.
- darts: <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html>.
- statsforecast: <https://nixtlaverse.nixtla.io/statsforecast/src/core/models_intro.html>.
- pmdarima 2.1.1: <https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html>.
- statsmodels `MLEResults.append`: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.append.html>.
- OnlineSTL: <https://arxiv.org/abs/2107.09110> (read through ar5iv).

From memory, to confirm before any docstring cites them: Ljung and Soderstrom,
*Theory and Practice of Recursive Identification* (1983); Cleveland, Cleveland,
McRae and Terpenning, "STL" (1990); Engle (2002); Cappiello, Engle and Sheppard
(2006); Aielli (2013); Pakel, Shephard, Sheppard and Engle (2021); Engle and
Kelly (2012); Kierkegaard, Nielsen, Jensen and Madsen (2000); Aknouche and
Guerbyenne (2006); Gerencser, Orlovits and Torma (2010); Hendrych and Cipra
(2018); Werge and Wintenberger (2022); Zechmeister and Kurster (2009); Hyndman
and Athanasopoulos, *Forecasting: Principles and Practice*.
