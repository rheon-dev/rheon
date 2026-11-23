<!-- kind: record -->
# Multi-horizon forecasts: the design note (task 559)

The building block PLAN 559 names (B4): one `horizons=[...]` interface for
every model with a forecast recursion (`garch`, `ar`, `holt`, `kalman`,
`dcc`), giving per horizon the h-step value and the integrated sum; exogenous
inputs in `garch`'s variance equation (GARCH-X); and a score-driven (GAS)
family covering Beta-t-EGARCH. Branch `vm/task-559-horizons`, cut from
`main` at `c4e7c7d` (request A24, 2026-11-23). This is part 1: the scan, the
design and its questions. No model code is written yet.

`ar` (PLAN 486) and the DCC `Q` recursion and its estimation (PLAN 514, 493)
are not on `main`. Their parts are designed here against the studies
(`STUDY-486-491.md` on `vm/study-486`, `STUDY-493-488.md` on `vm/study-493`)
and are **built last, on the tree those tasks merge into**: the interface,
the field names, the cap and the tests land first with `garch`, `holt` and
`kalman`, so `ar` and `dcc` only add their recursion.

## 1. The oracle-library scan

Each convention marked *measured* was run here (Linux x86-64; the
repository's `.venv`; R 4.3.3, with `betategarch` installed in a scratch
library) against the definition written by hand. The rest were read from the
library's source, named by file.

| library | version | licence | what it computes at horizon `h` | its convention |
|---|---|---|---|---|
| Python `arch`, `ARCHModel.forecast(params, horizon, start, method)` | 9.0.0 (dev group) | NCSA | variance forecasts `h.1 ... h.H`; `analytic` for GARCH, GJR, FIGARCH (and HARCH, ARCH); `simulation` and `bootstrap` for any process | *measured*: `h.1` at `start = t − 1` is the conditional variance of row `t` (gap 0.0); `h.k` is `ω + (α + γ/2 + β) f_{k−1}` (≤ 1 ulp from the recursion by hand); EGARCH `analytic` past 1 raises "Analytic forecasts not available for horizon > 1"; no exogenous term in any volatility process (`GARCH.__init__` takes no `x`; `forecast(x=)` is the mean model's); no integrated sum (a `cumsum` of the columns) |
| Python `statsmodels`, `AutoReg.predict/forecast`, `AutoRegResults.get_prediction` | 0.15.0 (dev) | BSD-3-Clause | AR(p) forecasts and their error variance | *measured*: the forecast is the recursion with forecasts in place of future lags (≤ 2 ulps); `var_pred_mean` is `σ² Σ_{j<h} ψ_j²` with `σ² = RSS/nobs` (ratio 1.0); future `exog` must be given (`exog_oos`) |
| `statsmodels`, `Holt.forecast(k)`, `ETSModel.get_prediction` | 0.15.0 | BSD-3-Clause | Holt's and ETS forecasts and ETS's forecast variance | *measured*: `Holt` is `l + k b` (0 ulps); ETS(A,A,N)'s `forecast_variance` is `σ² (1 + Σ_{j=1}^{h−1} (α + β j)²)` (ratio 1.0) |
| `statsmodels`, `MLEModel`/`SARIMAX`/`UnobservedComponents` `get_forecast` | 0.15.0 | BSD-3-Clause | state-space forecasts, mean and variance | future exogenous values must be given; a second opinion for `kalman` beside `filterpy` |
| Python `filterpy`, `KalmanFilter.predict()` repeated | 2.0.0 (dev, `>=1.4.5`) | MIT | the filter's time update with no measurement: `x ← F x`, `P ← F P Fᵀ + Q` | `kalman`'s existing second opinion (`tests/test_second_opinion.py`, `_filterpy_run`) |
| Python `scipy`, `special.ndtr`, `special.hyp1f1`, `integrate.quad` | 1.18.1 (dev) | BSD-3-Clause | the expectations the nonlinear forecasts need | *measured*: Nelson's EGARCH factor `E e^{c g(z)}` in closed form against `quad`: 0 to 7e-16 relative; the Beta-t score's `E e^{c u}` as `e^{−c} ₁F₁(½; (ν+1)/2; c(ν+1))` against `quad` over the Beta and over the t: ≤ 9e-15 |
| R `rugarch`, `ugarchspec(variance.model = list(external.regressors = v))`, `ugarchfilter`, `ugarchforecast(external.forecasts = list(vregfor =))` | 1.5-6 | GPL-3 | GARCH-X, GJR-X, EGARCH-X at fixed parameters, and their forecasts | read from `src/filters.c` (`sgarchfilter`): `h_t = ω + Σ ζ_j v_{j,t} + α ε²_{t−1} + β h_{t−1}`, the regressor **on the row**, as the caller aligned it; `.forcregressors` fills a missing `vregfor` with zeros, so a test passes every step's value |
| R `rmgarch`, `dccforecast` | 1.4-3 | GPL-3 | DCC and ADCC correlation and covariance forecasts | read from `R/rdcc-main.R` (`.dccforecastn`): `R_{t+i} = (1 − P^{i−1}) R̄ + P^{i−1} R_{t+1}`, `R̄` the normalized `Q̄`, `P = Σa + Σb` (ADCC's `g` not in it), `H = D R D` with the margins' variance forecasts |
| Python `mvgarch`, `DCCGARCH.forecast(n_ahead)`, `aggregate_forecasts` | 2.0.2 (2026-07) | MIT | DCC forecasts by the same form; the covariance of the summed returns | read from `mvgarch/mgarch.py`: Engle and Sheppard's `R` solved forward, as `rmgarch`; the aggregate is Hlouskova's (2015), `Σ_h H_h` when the mean has no ARMA terms |
| Python `mgarch`, `predict(ndays)` | 0.3.0 (2022) | MIT | `H_T √ndays` | not a forecast: **not an oracle** |
| R `gasmodel`, `gas`, `gas_filter`, `gas_forecast(method = "mean_path" \| "simulated_paths")` | 0.6.2 | GPL-3 (pure R) | GAS(p, q) for 50 families, exogenous `x` in the recursion, scalings `unit`, `fisher_inv`, `fisher_inv_sqrt`, links | read from `R/main_estimate.R`, `R/main_forecast.R`, `R/distr_t_meanvar.R`: `f_t = ω + Σ β x_t + α S ∇_{t−1} + φ f_{t−1}` (the intercept form); `t_meanvar`'s `var` is the t's scale squared, log-linked, so its `f = 2λ` and its score is `u/2`; `mean_path` is the path of zero scores, which is `E[f_{t+h}]` in the link's space exactly; `simulated_paths` the Monte Carlo means in natural units |
| R `betategarch`, `tegarchRecursion`, `tegarch`, `predict.tegarch` | 3.4 | GPL-2 | Beta-t-EGARCH with leverage (and skew), its filter at given parameters, and forecasts of `σ` | *measured*: `predict` at `h = 1` is the definition (`stdev = σ √(ν/(ν − 2))`); past 1 it is **not an oracle**: its `h = 2` value moves with `n.ahead` (1.134701 at `n.ahead = 6`, 1.134622 at 3), and sits 6 standard errors from a simulation of the definition (1.134160 ± 0.000077, 10⁶ paths), where the closed form of section 2 gives 1.134270 (1.2 SE); the code raises `phi1` to `n.ahead − j` where the horizon is due. It forecasts `E σ`, not `E σ²`. Its filter is an oracle for the recursion |
| R `GAS`, `UniGASFilter`, `UniGASFor` | 0.4.3 (2026-10-30) | GPL-3 | univariate GAS, multistep by simulation | not run (compiled; `Rcpp`, `RcppArmadillo`); a candidate second opinion beside `gasmodel` |
| R `forecast`, `holt`, `ets` | 8.21.1 (installed) | GPL-3 | Holt's and ETS forecasts | not needed beside `statsmodels` |
| Python `pyflux` | 0.4.15 (2017) | BSD | GAS models | unmaintained, not run: not an oracle |

**The oracles, by quantity** (each test-only, the definition written by
hand beside it; the library computes the quantity, we write only the
definition):

- **GARCH, GJR, FIGARCH variance at `h`**: `arch`'s analytic forecast at the
  row's parameters (`fix(params)`, `start = t − 1`), column `h.h`. The
  integrated sum: the definition, against `math.fsum` of `arch`'s columns.
- **EGARCH variance at `h > 1`** (normal): Nelson's closed form (section 2),
  its factor computed by `scipy` (`ndtr`, and `quad` of the expectation as
  the second), and `arch`'s `simulation` forecast within its Monte Carlo
  error (*measured*: the closed form sat within one standard error of
  `arch`'s 200,000-path means at `h = 2 ... 10`).
- **GARCH-X**: `rugarch`'s `ugarchfilter` and `ugarchforecast` at fixed
  parameters, every future `vregfor` given (the `r-oracles` job, which
  already installs `rugarch` and `rmgarch`, `ORACLE-SEARCH-2026-11-21.md`),
  and the definition in numpy in every Python run.
- **AR**: `AutoReg(...).predict(params, start, end, dynamic=True)` at the
  online coefficients, and `get_prediction`'s `var_pred_mean` for the error
  variance.
- **Holt**: `statsmodels` `Holt` with the gains the warm-up settles to,
  `forecast(s + Δ)` (section 3's mapping).
- **Kalman**: `filterpy`'s `predict` with `F = Φ(Δ)` and the noise `kalman`
  charges, then `H x` and `H P Hᵀ + R`; `statsmodels`' state space as the
  second.
- **DCC**: `rmgarch` (`r-oracles`) and `mvgarch` (dev group, to be added),
  and the definition by hand.
- **GAS**: `gasmodel` (`r-oracles`, to be added) for the filter at fixed
  parameters, `mean_path` for `E[λ_{t+h}]`, `simulated_paths` for `E[σ²_{t+h}]`
  within Monte Carlo error; `betategarch::tegarchRecursion` for the
  leverage form's filter; `scipy.special.hyp1f1` and `quad` for the factor.
- The Rust tests hold every form to its definition, computed in
  double-double where the gap is a rounding question; no matrix result, so
  `crate::oracle` is not needed.

## 2. The h-step recursions

Notation: row `t` is read before it is learned (hard rule 2). `θ` are the
parameters as they stand, `w` weights, `P` a persistence. Every horizon is
counted **from the row**: horizon 1 is the row's own value, the one the model
already reports (`variance`, `pred`), so the forecasts extend that value and
never replace it. `f_h` is the value at horizon `h`, `F_h` the integrated
sum. `S_n = Σ_{i=0}^{n−1} P^i`, `T_n = Σ_{m=0}^{n−1} S_m`.

**GARCH(1,1), GJR** (`garch`, `gjr`): `s_t` the row's variance.

```text
f_1 = s_t,    f_h = ω + P f_{h−1}           P = α + β;   GJR: α + γ/2 + β (symmetric z, as now)
f_h = P^{h−1} s_t + ω S_{h−1}               F_h = Σ_{i=1}^{h} f_i = s_t S_h + ω T_h
```

`S` and `T` by doubling, every term positive (no `1 − P` cancels):
`S_{n+1} = 1 + P S_n`, `T_{n+1} = T_n + S_n`, `S_{2n} = (1 + P^n) S_n`,
`T_{2n} = (1 + P^n) T_n + n S_n`. The closed form `v̄ + P^{h−1}(s − v̄)`, `v̄ =
ω/(1 − P)`, is not used: at the bound `P = 1 − 10⁻⁴`, `1 − P` carries 1e-12
of relative error.

**FIGARCH** (as `garch/filter.rs` forecasts now): `f_h = ω/(1 − β) + Σ_{j=1}^{M}
λ_j ê_{t+h−1−j}`, `ê` the ring's `e²` for learned rows and `f` for rows at
or after `t`.

**EGARCH** (normal `z`; Nelson 1991): `g(z) = α(|z| − √(2/π)) + γ z`,

```text
ln s_{t+h−1} = ω S_{h−1}(β) + β^{h−1} ln s_t + Σ_{j=1}^{h−1} β^{j−1} g(z_{t+h−1−j})
f_h = exp(ω S_{h−1}(β) + β^{h−1} ln s_t) · Π_{j=1}^{h−1} M(β^{j−1})
M(c) = E e^{c g(z)} = e^{−cα√(2/π)} [ e^{(cα+cγ)²/2} Φ(cα + cγ) + e^{(cα−cγ)²/2} Φ(cα − cγ) ]
```

This lifts `garch`'s refusal of `horizon > 1` for EGARCH: the forecast has
this closed form for normal innovations, which `arch` estimates only by
simulation. With t innovations `E e^{c|z|}` is infinite and so is every
variance forecast past 1: an EGARCH-t (not offered today) would refuse
horizons, which is the reason Beta-t-EGARCH exists.

**GARCH-X** (section 10): `s_t = ω + π'x_t + α e²_{t−1} + β s_{t−1}`, so `f_1 =
s_t` and `f_h = ω̃ + P f_{h−1}` with `ω̃ = ω + π'x̄` past the row; EGARCH-X puts
`π'x` in `ln s` and takes `ω̃` in place of `ω` above.

**AR(p)** (`ar`, PLAN 486; `c` the intercept, with the trend, seasonal and
exogenous terms 486 builds in `c_h`):

```text
ŷ_1 = pred_t = c_1 + Σ_{i=1}^{p} φ_i y_{t−i}
ŷ_h = c_h + Σ_{i=1}^{p} φ_i ŷ_{h−i},   ŷ_{h−i} = y_{t+h−1−i} where that row is learned (a lag in the ring)
F_h = Σ_{i=1}^{h} ŷ_i
error variance (optional, question 9): σ² Σ_{j=0}^{h−1} ψ_j²,  ψ_0 = 1,  ψ_j = Σ_{i=1}^{min(j,p)} φ_i ψ_{j−i}
```

`c_h`'s trend counts learned rows (`AutoReg`'s observations), and a seasonal
dummy the learned row's position in its period, so both are known ahead.

**Holt** (`holt`, on the clock): `s` the clock since the target's last
observation, this row's delta included, so `pred = l + b s`.

```text
f(Δ) = l + b (s + Δ)          Δ ≥ 0 clock units past the row; f(0) = pred
F(Δ) = ∫_0^Δ f(u) du = l Δ + b (s Δ + Δ²/2)          trend = false: f = l, F = l Δ
```

**Kalman** (`kalman`, on the clock): the forecast at `Δ` is the prediction
`kalman` would make for a row with this row's features at the row's clock
plus `Δ`, with no row between. With `d` the row's delta and `D` the clock
since the covariance last took an observation:

```text
b(Δ) = Φ(d + Δ) b,   Φ(τ) = diag(2^{−τ/r_i})
P(Δ) = Φ(d + Δ) P Φ(d + Δ) + Q g(D + Δ)²            (g as the module defines it; D² at r = ∞)
f(Δ) = z'b(Δ),   v(Δ) = σ² (z' P(Δ) z + 1)           (pred_var's definition, in units of the noise)
F(Δ) = ∫_0^Δ f(u) du = Σ_i z_i b⁻_i (r_i/ln 2)(1 − 2^{−Δ/r_i})      (z_i b⁻_i Δ at r_i = ∞; b⁻ = Φ(d) b)
```

At `revert_half_life = inf` (the default) the mean does not move with `Δ`;
only the variance grows, which is the forecast's content there.

**DCC** (`dcc`, PLAN 514 and 493): `R_t` the row's correlation, `R̄ =
diag(Q̄)^{−1/2} Q̄ diag(Q̄)^{−1/2}`, `P = a + b` (ADCC too: with `E[z z'] ≈ Q`
and `E[n n'] ≈ N̄` its `g` terms cancel, as `rmgarch` does), `f_h^{(i)}` the
margin `i`'s variance forecast:

```text
R^{(h)} = (1 − P^{h−1}) R̄ + P^{h−1} R_t           (Engle and Sheppard's R solved forward)
H^{(h)}_{ij} = √(f_h^{(i)} f_h^{(j)}) R^{(h)}_{ij}
cov_sum = Σ_{h'=1}^{h} H^{(h')},   corr_sum_{ij} = cov_sum_{ij} / √(cov_sum_{ii} cov_sum_{jj})
```

`corr_sum` is the correlation of the returns summed over the `h` rows, the
term structure of correlation. The alternative, `Q` solved forward and then
normalized, is not used: both oracle libraries solve `R`.

**GAS** (section 10): location `μ_{t+h−1}` and log-scale `λ_{t+h−1}`:

```text
location:   E μ_{t+h−1} = ω + φ^{h−1} (μ_t − ω)                     (zero-mean score; gasmodel's mean_path)
log-scale:  λ_{t+h−1} = ω + φ^{h−1}(λ_t − ω) + Σ_{j=1}^{h−1} φ^{j−1} k(ε_{t+h−1−j})
            k(ε) = κ u + κ* sgn(−ε)(u + 1)
f_h = E e^{2λ_{t+h−1}} · v_ε = e^{2ω + 2φ^{h−1}(λ_t − ω)} Π_{j=1}^{h−1} K(2φ^{j−1}) · v_ε
K(c) = ½ [e^{cκ*} m(c(κ + κ*)) + e^{−cκ*} m(c(κ − κ*))],   m(a) = E e^{a u}
t:      u = (ν + 1) b − 1, b ~ Beta(½, ν/2):  m(a) = e^{−a} ₁F₁(½; (ν + 1)/2; a(ν + 1)),   v_ε = ν/(ν − 2)
normal: u = z² − 1:  m(a) = e^{−a}/√(1 − 2a)  (a < ½),   v_ε = 1
```

`|sgn|` and `u` are independent for a symmetric `ε`, which gives `K`. The
closed form agreed with a 10⁶-path simulation of the definition at the
parameters `betategarch` fitted, within 1.2 standard errors at `h = 2 ... 6`
(section 1).

## 3. Spans: rows and the clock

A horizon is counted in the unit the model's recursion steps in, and each
kind takes one:

| kind | steps per | horizon | `h = 1` / `Δ = 0` is |
|---|---|---|---|
| `garch` (every `vol`), GARCH-X, `gas`, `ar`, `dcc` | learned row (observation) | rows, `h ≥ 1` | the row's own value |
| `holt`, `kalman` | clock | clock span `Δ ≥ 0` past the row's own time | the row's `pred` |

- **Per-observation kinds take rows only.** Their recursion steps once per
  learned row, whatever the clock between (`GARCH-DESIGN.md`: no parameter
  of the process has a duration form), so a clock span holds a number of
  steps the data decides and has no forecast. A clock span is refused with
  that reason (PLAN 538 asks the docs to say so). The way to it is a row
  count the user computes in Polars; a rate-based mapping (rows = `Δ` over
  the mean gap) would add an estimate the model does not have
  (question 2).
- **Clock kinds take clock spans only.** `holt` extrapolates `l + b·(clock)`
  and `kalman`'s transition and noise are on the clock; the clock of future
  rows is not known, so a row count has no forecast on a time clock. On the
  row clock (no clock column, `d = 1` a row), a clock unit is a row: `Δ = 4`
  is the value four rows past this one.
- **The mapping to the libraries**: a per-observation horizon `h` at row `t`
  is `arch`'s `h.h` at `start = t − 1` and `statsmodels`' step `h` from the
  rows before `t`. A clock horizon `Δ` is `statsmodels`' step `s + Δ` from the
  last observation (`s = 1` on a regular row clock), and `filterpy`'s
  transition over `d + Δ`.
- **Spans are the user's, not capped by `gap_cap`**: a cap shortens the
  past's clock as it does now (`s`, `d`, `D` are the capped values), never
  the horizon asked for.
- A clock horizon is a number on a numeric clock or a duration on a
  temporal one (`_duration.py`'s three forms), as `half_life` is.

## 4. Outputs and field names

Per horizon `L` and target `t`, two fields (three for `kalman`):

| kind | value at `L` | integrated sum | more |
|---|---|---|---|
| `garch`, `gas` (log-scale) | `variance_<L>_<t>` | `variance_sum_<L>_<t>` | |
| `ar`, `holt`, `gas` (location) | `forecast_<L>_<t>` | `forecast_sum_<L>_<t>` | `forecast_var_<L>_<t>` for `ar` (question 9) |
| `kalman` | `forecast_<L>_<t>` | `forecast_sum_<L>_<t>` | `forecast_var_<L>_<t>` |
| `dcc` | `corr_<L>_<a>_<b>`, `cov_<L>_<a>_<b>`, `variance_<L>_<a>` | `cov_sum_<L>_<a>_<b>`, `corr_sum_<L>_<a>_<b>`, `variance_sum_<L>_<a>` | the pair naming follows 514's |

- **The label `L`**: `h<n>` for a row horizon (`variance_h21_ret`); the
  duration text for a clock horizon on a temporal clock
  (`forecast_30d_price`); `c<number>` on a numeric clock (`forecast_c2.5_y`,
  since a bare number would read as a row count); or the user's own name when
  `horizons` is a dict, `{"1m": 21, "3m": 63}`, as `power_variation`'s terms
  name their fields (`<term>_<target>`). The block's pattern,
  `<quantity>_<label>_<target>`, is `power_variation`'s with the quantity in
  front.
- **Not `pred_`**: `pred_` marks the row's own prediction for `eval.unpack`
  (`fields.rs`, the conformal comment), so a horizon's value is `forecast_`.
- **`garch`'s `horizon = h` and `forecast_<t>` are replaced** by `horizons =
  [h]` and `variance_h<h>_<t>`. Both are unreleased (CHANGELOG
  `[Unreleased]`), so no deprecation path; the CHANGELOG entry is rewritten.
- `weight_sum` is unchanged; no `n_<target>` is needed, since nothing here
  is divided by a count.
- Every horizon emits both fields; a per-horizon switch would save columns
  the user can drop in Polars.

## 5. Decay, windows, zero-weight rows

- **The state at the row, nothing more.** A forecast is a function of the
  state `pred` reads on the row (after the last learned row, and for the
  clock kinds after the row's own clock step) and of the row's features. It
  is computed at the read, writes nothing, and adds no decay across the
  horizon: decay weighs the evidence behind the parameters; it is not a
  dynamic of the forecast. Kalman's `revert_half_life` *is* a transition,
  so it moves the mean across `Δ`, and its process noise grows the variance.
- **Windows**: a windowed fit (`ar`'s window, from the 485-510 decay and
  window rule) forecasts from the windowed coefficients and the lag ring's
  last learned values, which the window does not cut.
- **`min_weight`**: every horizon field is null while the row's own value
  is (the target's weight under `min_weight`, rule 8), with the same reason.
- **A zero-weight row** (rule 9) is read and steps nothing:
  - per-observation kinds: its horizons are those the next learned row would
    read on the same state, to the bit (the next learned row is horizon 1);
    a horizon counts learned rows, so a zero-weight row is in no horizon;
  - clock kinds: its horizons are at its own clock (`s` and `d` carry its
    delta, folded into the next row's as now), so the next learned row
    forecasts as if the zero-weight row were not there;
  - GARCH-X reads the zero-weight row's own `x` for its horizon 1 and
    leaves `x̄` alone (a weight of 0 adds nothing to a weighted mean).
- **A null target** is that target's zero-weight row; the others go on.
- **A clock break that empties `ar`'s lags** withholds `pred` until the ring
  refills, and so every horizon.

## 6. State and per-row cost

The forecasts add **no state**: each is computed from the state at the read
into the thread-local scratch the FIGARCH forecast already uses. The spec's
horizons are configuration (8 bytes each, at most 64). The parts that add
state are GARCH-X and GAS, each bounded by the spec's shape:

| part | bytes a target (`garch`'s formula, `8 (2k + m + n(n+1)/2 + 9)`) | beside it |
|---|---|---|
| any horizons on any kind | 0 | `8 K` in the spec, `K ≤ 64` |
| GARCH(1,1)-X, `q` inputs | `k = 3 + q`, `n = 3 + q`: 272 at `q = 2` (168 without) | `x̄`, `8 q` a group; the start's buffer `16 + 8 q` a row (32 at `q = 2`) |
| EGARCH-X | as above, plus the 512 of the guard's bins | |
| `gas`, log-scale t with leverage (`ω, κ, φ, κ*`, `ν`) | `k = 4`, `m = 1`, `n = 5`: 264 | the guard's 64 bins, 512; the start's buffer, 16 a row |
| `gas`, location t (`ω, κ, φ`, `σ`, `ν`) | `k = 3`, `m = 2`, `n = 5`: 256 | the start's buffer, 16 a row |
| `ar`, `dcc` | 0 beyond 486's and 514's | `R̄` is read from `Q̄`, which 514 keeps |

Against the 256 KB default (`max_state_bytes`): only the start's buffer
moves. GARCH-X at `q = 2` doubles it from 32,000 to 64,000 bytes at the
default 2,000 rows, so the start's two doublings reach 256,000 (inside the
cap); at `q = 3` the cap holds one doubling, and the existing rule ("doubles
while the cap holds the doubled buffer") stops there. Nothing grows with the
stream's length.

**Per row**, a target: the closed forms (`garch`, `gjr`, GARCH-X, `holt`,
`kalman`'s mean, `dcc`'s `R`, `gas` location) cost `O(K log H)`
multiplications; the forward pass (EGARCH and GAS log-scale products,
FIGARCH, `ar`, `dcc`'s covariance sums) costs `O(c H_max)` (section 9).

## 7. Chunk invariance, save and load, platforms

- **Chunk invariance** follows from the forecasts being pure functions of
  the state and the row: the bank's chunk-invariance test runs every kind
  with a horizons spec (1 chunk against 1,000, to the bit).
- **Save and load**: a stream saved at row `t` and continued reads the same
  forecasts to the bit. `holt` and `kalman` are released kinds: their
  `horizons` is a `#[serde(default, skip_serializing_if = "Vec::is_empty")]`
  field, as `kalman`'s `covariance_form` is, so their present fixtures load
  and re-save byte for byte, and a state with horizons is a new form with
  its own fixture case. `garch` (and GARCH-X), `gas`, `ar` and `dcc` are
  unreleased; their layout changes inside the unreleased window take no
  schema bump of their own, by `POWER-VARIATION.md`'s precedent, and their
  fixture cases are regenerated (rule 5). A fixture freezes each new form:
  horizons on each kind, GARCH-X, each GAS form.
- **Platforms**: the closed forms and the passes are `+ × ÷` in a fixed
  order, the same bits everywhere. EGARCH's and GAS's factors read `exp`
  and `erfc` from the `libm` crate (already linked, `garch/math.rs`) and
  `₁F₁` from a series in plain arithmetic, so they are the same bits too.
  `kalman`'s `2^{−τ/r}` is what its `pred` already reads.

## 8. Out of sample by construction

A forecast at row `t` reads the state before row `t` is learned, so it uses
rows before `t` and the row's own features, never its target (rule 2).
Tests: (1) horizon 1 equals the row's `variance` or `pred` to the bit, on
every kind; (2) changing row `t`'s target by any amount moves no field of
any row up to `t` (to the bit), and changing row `t + 1`'s features moves
none either (GARCH-X); (3) the oracle alignment of section 3 holds at
`start = t − 1`, never `t`.

## 9. The integrated sum and the work cap

`F` is the sum of the values over the rows the horizon spans, the row's own
included (rows), or their integral over the span past the row (clock). It is
never a re-sum of horizons: the closed forms give `F_h` directly (`T_h` for
the GARCH family; Holt's and Kalman's integrals), and the forward pass to
`H_max = max(horizons)` carries one running sum and reads it at each
requested horizon, one addition a step.

The forward pass is `O(c H_max)` a target a row, with `c` the work of one
step: 1 for EGARCH and GAS (plus their `exp`/`erfc`/`₁F₁` calls), `p` for
`ar`, `M` for FIGARCH (its present forecast is already `O(h M)`, uncapped),
`m(m + 1)/2` for `dcc`. Proposed cap: `c · H_max ≤ 2²⁰` multiply-adds a
target a row, a libm call counted as 32, checked when the spec is built and
refused naming the largest horizon that fits. That admits FIGARCH at `M =
1,000` to `H = 1,048`, `ar` at `p = 20` to 52,428, EGARCH and GAS to about
30,000. The closed forms need no cap beyond `H < 2³¹` and `K ≤ 64`. EGARCH's
and GAS's products converge as `β^{2j}` shrinks; stopping when the rest of
the product is 1 to the rounding would bound their pass by the parameters
instead of `H_max`, a refinement the build may measure.

## 10. GARCH-X and the GAS family

**GARCH-X.** The exogenous inputs are the spec's `features`, read on the row
as every regression in the bank reads them: known before the row's target by
the bank's contract, so `x_t` enters `s_t` (`rugarch`'s alignment: the caller
lags `x` when it is the previous day's measure).

```text
garch-X    s_t = ω + π'x_t + α e²_{t−1} + β s_{t−1}                π ≥ 0, x_t ≥ 0
gjr-X      s_t = ω + π'x_t + α e²_{t−1} + γ 1[e_{t−1} < 0] e²_{t−1} + β s_{t−1}
egarch-X   ln s_t = ω + π'x_t + α(|z_{t−1}| − √(2/π)) + γ z_{t−1} + β ln s_{t−1}     π any sign
sensitivity (garch-X)   ∂s_t/∂π = x_t + β ∂s_{t−1}/∂π           (the filter's recursion, one more column a input)
past the row            x̄ = Σ w x / Σ w over learned rows     (question 4)
```

The positivity of `s` under GARCH and GJR needs `π ≥ 0` with `x ≥ 0`; a row
whose `x` is negative is refused as a non-finite feature is (the rule every
model keeps). The start's likelihood-ratio test against constant variance
takes `2 + q` restrictions (`χ²` at 0.999). `π = 0` is where an input that
explains nothing sits, so `π`'s bound does not count among the "on a bound"
conditions that make a start wait and refit; the process's bounds still do.
FIGARCH-X is not offered.

**The GAS family** (Creal, Koopman and Lucas 2013; Harvey 2013). One
time-varying parameter, GAS(1,1), in Harvey's mean form so that `ω` is the
parameter's long-run level and `κ` is `betategarch`'s:

```text
log-scale:  y_t = e^{λ_t} ε_t,   λ_{t+1} = ω + φ(λ_t − ω) + κ s_t + κ* sgn(−y_t)(s_t + 1)
location:   y_t = μ_t + σ ε_t,   μ_{t+1} = ω + φ(μ_t − ω) + κ s_t
s_t = I^{−d} ∇_t,   ∇_t = ∂ ln p(y_t | f_t)/∂f_t,   d ∈ {0, ½, 1}  (unit, inverse square root, inverse information)
```

| `ε` | parameter | `∇` | information `I` |
|---|---|---|---|
| t, `ν` (Beta-t-EGARCH) | `λ` | `u = (ν+1) y²/(ν e^{2λ} + y²) − 1`, in `[−1, ν]` | `2ν/(ν + 3)` |
| normal | `λ` | `z² − 1`, `z = y e^{−λ}` | 2 |
| t, `ν` | `μ` | `(ν+1)(y − μ)/(ν σ² + (y − μ)²)` | `(ν + 1)/((ν + 3) σ²)` |
| normal | `μ` | `(y − μ)/σ²` (with `d = 1`, exponential smoothing) | `1/σ²` |

**The score's scaling.** In each row of the table `I` does not move with the
time-varying parameter, so the three scalings differ by a constant factor,
absorbed in `κ`: the choice changes which `κ` the user reads, never the
filter. `d = 0` on `λ` is Harvey's `u` (`betategarch`'s `kappa1`);
`gasmodel`'s `t_meanvar` runs on `f = 2λ`, whose unit score is `u/2`, so its
`α` is `4κ` and its intercept `2ω(1 − φ)`.

**Estimation** is `garch`'s, reused: a start fitted by the core's projected
Newton (`garch/optim.rs`), the recursive prediction-error step after it, the
projection in `H`'s metric, `ν` stepped as `ln(ν − 2)`. Admissible set: `0 ≤ φ
≤ 1 − 10⁻⁴`, `κ ≥ 0`, `|κ*| ≤ κ`; for the normal log-scale form also `κ + |κ*| <
¼` (section 11).

## 11. Guards

Each guard checks the condition itself and withholds with a reason; none is a
row count (PLAN 552).

| condition | where | withholds | reason |
|---|---|---|---|
| the row's own value is withheld (not started, `not_identified`, under `min_weight`, `ar`'s ring empty, `kalman`'s readiness) | every kind | every horizon | the row's own reason |
| a start's likelihood ratio against the static model below `χ²` at 0.999 (`garch`: 2, 3; GARCH-X: `2 + q`; GAS: 2, 3 with leverage) | `garch`, GARCH-X, `gas` | the start's fit, as now | `not_identified` |
| the start's information singular in `π` (an input constant or collinear over the start's rows: its smallest eigenvalue, scaled to unit diagonal, at the rounding) | GARCH-X | the start | `not_identified`, naming the inputs |
| the start's Newton decrement over 10⁻⁶ nats | `garch`, `gas` | as now | as now |
| the filter not contracting: `E ln abs(φ − 2(κ + κ* sgn(−y))(ν + 1) b(1 − b)) ≥ 0` (t; `∂u/∂λ = −2(ν + 1) b(1 − b)`, the normal's `−2z²`) read from 64 bins of `b` (of `z²`) and the sign at the current parameters (Blasques et al. 2018), held below 0 by cutting the step, as EGARCH's guard | `gas` log-scale | the step | (no withholding) |
| a variance forecast that does not exist: `E e^{a(z² − 1)}` needs `a < ½`, so `κ + abs(κ*) < ¼` | `gas` normal log-scale | held in the admissible set (question 7) | (no withholding) |
| `R̄` not positive definite (`Q̄`'s Cholesky fails) | `dcc` | horizons past 1 | `not_identified` |
| a value not finite (overflow at a long horizon of an explosive `ar`, an `exp` past the double range) | every kind | that field | null |

An explosive `ar` fit is forecast as its recursion, as `statsmodels` does;
no stationarity guard is proposed, since the row's own `pred` has none and
the check is an eigenvalue problem a row.

## 12. Build order

1. The interface: `horizons` (list or dict), the labels, the cap, the fields,
   the refusals; `garch` on it (closed forms for `garch`/`gjr`, FIGARCH's
   pass, EGARCH's product), `horizon` and `forecast_<t>` removed.
2. `holt` and `kalman` (clock spans, integrals, `kalman`'s variance).
3. GARCH-X (`rugarch` and the definition).
4. `gas` (Beta-t-EGARCH first, then the normal log-scale and the location
   forms; `gasmodel`, `betategarch`, `scipy`).
5. `ar`, then `dcc`, on the tree where 486, 514 and 493 have merged.

Each part: chunk invariance, zero-weight rows, out-of-sample perturbation,
fixtures, realistic and degenerate cases (PLAN 550), the docs (`MODELS.md`,
every place `EXTENDING.md` lists), and the budgets' memory and speed cells.

## 13. Open questions

1. **Counting**: a row horizon counts the row (`h = 1` its own value, `arch`'s
   `h.1` at `t − 1`); a clock horizon is measured from the row's own time
   (`Δ = 0` its `pred`). *Recommended.*
2. **One unit per kind**: rows for `garch`, `gas`, `ar`, `dcc`; the clock for
   `holt`, `kalman`; the other refused with its reason, under the one name
   `horizons`. *Recommended*, over a rate-based clock form for the
   per-observation kinds.
3. **Names**: `<quantity>_<label>_<target>` and `<quantity>_sum_…`, label
   `h<n>`, the duration text, or the dict's key; `garch`'s unreleased
   `horizon`/`forecast_<t>` replaced outright. *Recommended.*
4. **Exogenous inputs past the row** (GARCH-X, `ar`'s exogenous terms, GAS-X
   if wanted): `x̄`, their learned weighted mean, so the term structure
   reverts to `(ω + π'x̄)/(1 − P)`; or held at the row's value; or columns the
   user supplies. *Recommended `x̄`*; `kalman` holds the row's features, its
   forecast being the regression at them.
5. **GARCH-X's form**: `x` on the row, `π ≥ 0` and `x ≥ 0` for GARCH and GJR,
   any sign in EGARCH-X, no FIGARCH-X, `π`'s bound out of the refit rule.
   *Recommended.*
6. **GAS's shape**: a new kind `gas` on `garch`'s internals; log-scale (t,
   normal) with leverage and location (t, normal); GAS(1,1); `λ` the log of
   the scale (Harvey's, `betategarch`'s `κ`) with `variance = e^{2λ} ν/(ν −
   2)`; `scaling` unit by default. Or Beta-t-EGARCH alone as a `vol` of
   `garch`. *Recommended the kind.*
7. **The normal log-scale GAS's moment**: hold `κ + |κ*| < ¼` in the
   admissible set (the condition for its variance and its forecasts to
   exist), rather than withhold horizons. *Recommended.*
8. **The cap**: `c · H_max ≤ 2²⁰` a target a row, a libm call as 32, refused
   at the call; `K ≤ 64`. Or `2¹⁶`, which holds FIGARCH at `M = 1,000` to 65.
   *Recommended `2²⁰`.*
9. **Error variances for the value kinds**: `kalman` (its own `P`) and `ar`
   (`σ² Σ ψ²`, `statsmodels`' `var_pred_mean`) emit `forecast_var_<L>_<t>`;
   `holt` none, its gains not being ETS's fixed `α, β`. *Recommended.*
10. **DCC**: Engle and Sheppard's `R` solved forward, `P = a + b` for ADCC,
    as `rmgarch` and `mvgarch`. *Recommended.*
11. **EGARCH past one row** by Nelson's closed form, lifting today's refusal.
    *Recommended.*

## 14. Part 2 as built: the interface and `garch` (request A24, 2026-11-23)

**The answers** (the user on the Exchange page, relayed by the
coordinator): all eleven recommendations taken, with these changes: (1)
horizon 1 is the row's own value and a test holds the equality; (5) a
negative input under GARCH or GJR is withheld with a reason, never clipped;
(7) the bound is cited; (8) the work cap is a named limit,
`max_horizon_work`, refused at build with the work it would take; (9) `holt`
gets a forecast variance if `ETSModel`'s applies to ETS(A,A,N), else none
with the reason; (10) ADCC's persistence is `a + b + δ g` (Cappiello, Engle
and Sheppard), checked against `rmgarch`; (11) EGARCH's closed form is
checked against `arch`'s simulation with draws enough to resolve the
difference.

**What part 2 built.** `garch` takes `horizons` (a list of row counts,
labelled `h<n>`, or a dict naming them) and `max_horizon_work` (2²⁰ by
default, at most 2³⁰), in place of `horizon` and `forecast_<t>`. Per target
and horizon, `variance_<L>_<t>` and `variance_sum_<L>_<t>`. One pass from
horizon 1 to the largest (`garch/horizons.rs`): GARCH and GJR run the
recursion with the terms in `Filter::predict`'s order; FIGARCH's pass is the
existing forecast's arithmetic, unchanged to the bit; EGARCH's is Nelson's
product, `exp` and `erfc` from the `libm` crate. The sum is TwoSum-
compensated. Every process takes horizons; the EGARCH refusal is gone. Step
work: 4 (GARCH, GJR), `M + 2` (FIGARCH), `4 + 5·32` (EGARCH), so the default
holds horizons of 262,144, 1,046 at `M = 1,000`, and 6,393.

**Not to the bit against `arch`, and why.** The brief asked for `arch`'s
forecast to the bit. `arch` squares each forecast's square root through
numpy's vectorized `power` with an array of exponents; on this x86-64
(AVX-512, numpy 2.5.2) that differs from the correctly rounded product in
5.7% of 10⁷ draws, and glibc's scalar `pow(x, 2)` from `x·x` in 189 of
200,000. `sqrt(f)²` is not `f` either. So `arch`'s last bits are its
platform's. The bank is held to the recursion written out to the bit (Rust
`the_forecasts_are_their_definitions_written_out`, Python
`test_the_forecasts_are_the_recursion_run_ahead`) and to `arch`'s analytic
forecast, run from the bank's own variance and residual
(`tests/test_second_opinion_garch_horizons.py`), within a bound in units of
the last place set from the measured gap.

**State and fixtures.** The forecasts add no state. `GarchCfg.horizons`
reads a state's `horizon` (schema 71's form): `h > 1` as `[h]`, 1 as none;
the v71 garch fixtures load through it, convert to the current bytes, and
continue to the bit with the new sum slots dropped from the comparison
(`narrowed`). Both fields are written even when empty: the contract's probe
saves a state as an array (`rmp_serde::to_vec`), where a skipped field
shifts every later one; `holt`'s and `kalman`'s `horizons` are the last
field of their configuration, so skipping it when empty shifts nothing. No
schema bump: `garch` is unreleased. The realistic cases carry three
horizons out of order, and FIGARCH's its own `max_horizon_work`.

**Revised after R104** (the Mac's answers (d) and (e)): the first build took
GARCH's and GJR's forecast terms in the row's own order, which moved the
forecasts by up to 4 units in the last place and needed a tolerance on the
v71 continuation; the forecast is `ω + P f` again, the single horizon's
arithmetic, so the goldens' forecasts equal `main`'s to the bit and the v71
continuation needs none. The writer check on the kept sets is strict again;
the garch fixtures this build writes differ from `main`'s (`horizons` for
`horizon`), and a set written here carries the writer `linux-x86_64`, so
the strict check passes only once they are written on the platform the v71
set was, macOS arm64.

**Measured against `arch`** (Linux x86-64, 2026-11-23,
`tests/test_second_opinion_garch_horizons.py`). The analytic forecast from
the bank's own variance and residual, over 79 rows (15 for FIGARCH) from
row 2,050 to 2,599, largest gap in units of the last place at horizons 1,
2, 5, 22 and 100: GARCH 0, 1, 3, 6, 9; GJR 0, 1, 2, 4, 8; FIGARCH 19, 15, 16,
26, 20 (a 1,000-term dot product in BLAS's order); sums at most 7, 4 and
19. The bound held is `2 √h + 2` (64 for FIGARCH). EGARCH against `arch`'s
simulation with 10⁶ draws at rows 2,100, 2,300 and 2,599, horizons 2, 3, 5
and 10: the bank within 1.9 standard errors everywhere (relative standard
error 1.3e-4 to 3.7e-4), while `exp(E ln s)`, the forecast without
Nelson's convexity, sits 57 to 149 standard errors away, so the draws
resolve the difference the closed form makes. The 20,000-draw form runs in
the essentials, the 10⁶ one (about 6 s, 200 MB) in the extended tier.

## 15. Part 3: `holt` and `kalman` on the clock (2026-11-23)

**Answer 9, measured.** `statsmodels`' `ETSModel` forecast variance for
ETS(A,A,N), `σ² (1 + Σ_{j=1}^{h−1} (α + β j)²)` (checked in section 1), is
the variance of `holt`'s forecast errors where `holt` is ETS(A,A,N): unit
row steps, unit weights, gains past their warm-up (`α = 1 − λ_l`, `β = (1 −
λ_b) α`). On 200,000 simulated ETS rows at `(α, β) = (0.3, 0.06)`, from row
5,000, the empirical variance of the `h`-step errors over the formula was
1.0023, 1.0014, 0.9964 and 0.9930 at `h` = 1, 2, 5 and 10. Elsewhere it does
not apply: on a clock the rows a span holds are not known, the gains move
with the spacing and the weights and through the line fit's warm-up, and
`holt` keeps no noise variance. So `holt` reports none, and its docstring
and MODELS.md say why.

**Built.** Both take `horizons`, spans of the clock `Δ ≥ 0` past the row
(the polars spec's `SpanHorizon`: a span labelled `h<number>` or by the
duration's text, or a `[name, span]` pair), at most 64, distinct, finite.
The forecasts ride after the predictions in the model's `pred` (each
target's slots together), as `Source::Stat` fields; the bank's slot stride
for a regression is now `m·nc` plus `Spec::forecast_slots()`, and the
residual machinery treats the extra slots as having no target (their
residuals are NaN and fold nowhere).

- `holt`: `f(Δ) = l + b (s + Δ)`, `F(Δ) = Δ (l + b (s + Δ/2))`, from the
  level and trend before the row; `Δ = 0` is `pred` to the bit.
  `statsmodels`' `Holt.forecast` from the rows before (after the gains stop
  moving) agrees to 1e-12, and scipy's `quad` of it gives the integral.
- `kalman` (`kalman/horizons.rs`): the mean `zᵀ T(Δ) b`, the integral
  `zᵀ T̄(Δ) b` with each factor `2^(−u/r)` integrated, and the variance from
  `T(Δ) P T(Δ)ᵀ` plus the drift of `D + Δ` carried into the anchor's
  coordinates as `covariance_now` carries it, scaled as `pred_var` is.
  `predict` reads what `step` writes to the bit in both covariance forms,
  with and without the standardizer, reversion and `share_p`. Against
  `filterpy` (unstandardized, a given noise and `q`, per-slot reversion,
  rows of weight 0): largest gaps 2.6e-16 (mean), 1.4e-15 (integral) and
  1.3e-16 (variance) of `1 + |value|`.

**State and fixtures.** No state is added. `horizons` is the last field of
`HoltCfg` and `KalmanCfg`, skipped when empty, so every existing fixture
loads and re-saves its bytes (none was regenerated for part 3), and a
state written as an array keeps its places. `kalman.rs` stays under the
250 KB cap by keeping the forecasts in `kalman/horizons.rs`, and the Python
`kalman` builder moved to `_spec_kalman.py` for `_spec.py`'s cap.

**A bug the forecast slots found.** Under an `embargo` a held row keeps a
record of its scores, one entry a slot in each part, and its release checks
the record's length. `kalman`'s row inflation (which a conformal interval
reads) is one a prediction, so with forecast slots the record came up short
and every released row scored nothing: `sigma`, the interval, the metrics
and the calibration went null. The stream now pads the row inflation to
every slot (the forecasts' read none), and
`test_a_short_delay_is_the_undelayed_run_with_horizons` holds both models to
the undelayed run under a short delay.

## 16. Part 4 (GARCH-X): the oracle's convention, measured (2026-11-23)

`rugarch` 1.5-6 installed here (R 4.3.3, beside `Rcpp` 1.1.1 and
`RcppArmadillo` in a scratch library; the system's `Rcpp` 1.0.12 is too old
for it). At fixed parameters, zero mean, normal innovations, an input `v ≥
0` on each of 1,500 rows, its filter against the definition written in R:

| model | `rugarch`'s variance equation, measured | largest gap (rows 2 on) |
|---|---|---|
| `sGARCH` | `s_t = ω + ζ v_t + α e²_{t−1} + β s_{t−1}`, the input on the row itself | 6.4e-16 |
| `gjrGARCH` | `s_t = ω + ζ v_t + (α + γ 1[e_{t−1} < 0]) e²_{t−1} + β s_{t−1}` | 5.6e-16 |
| `eGARCH` | `ln s_t = ω + ζ v_t + α z_{t−1} + γ (\|z_{t−1}\| − √(2/π)) + β ln s_{t−1}` | 7.2e-16 |

So `rugarch` reads the input of the row whose variance it forms (the caller
lags it), as section 10 proposes and the user's answer 5 confirmed.
`rugarch`'s `eGARCH` names the sign term `α` and the size term `γ`, the
reverse of `arch`'s and `garch`'s (`α` on `|z| − √(2/π)`, `γ` on `z`): a
test maps them. `ugarchforecast` takes the future inputs as
`external.forecasts$vregfor`, one a step, and fills a missing one with 0;
its first step from the last row reads `vregfor[1]` for the next row
(measured to 1.4e-16 against the recursion by hand). The bank's forecast
holds the input at its learned weighted mean `x̄` past the row (answer 4),
so a test hands `rugarch` `x̄` for every step but the first, which is the
row's own input.

The oracles for the build: the definition in numpy (every Python run), a
batch maximum-likelihood fit by `scipy.optimize` on the same rows for the
estimates, and `rugarch` where R is present. The repository's CI has no R
job on `main` yet (the `r-oracles` group and job are on task 499's and
555's branches), so `rugarch`'s test would be marked for that job and
skipped elsewhere.
