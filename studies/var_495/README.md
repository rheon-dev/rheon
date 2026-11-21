# Study 495: an online VAR(p) with Granger-causality tests

A numpy prototype of PLAN task 495 ("a vector autoregression by exponentially
weighted least squares on lags the model keeps, exact against `statsmodels` `VAR`
with no decay; Granger tests read from the same sums (Wald, as
`VAR.test_causality`)"), checked against `statsmodels` 0.15.0. Python only, no
Rust: the model in `rheon-core` behind `OnlineModel`, `docs/EXTENDING.md`'s places
and the measured Rust cost are the task proper, not this study. Nothing here is
collected by the repo's pytest (`testpaths = ["tests"]`), and nothing is added to
`pyproject.toml`: `statsmodels` and `scipy` are already in the dev group.

## Files

- `online_var.py`: the model (`OnlineVAR`) and a feeder (`run`). The update
  equations, the fit, the Granger statistics and the state size are in its module
  docstring.
- `data.py`: seeded synthetic VAR data (no data files).
- `compare.py`: every check against statsmodels, printing the table below.
- `test_online_var.py`: the same checks as asserts (21 tests, about 3 s):
  `uv run --no-project --with numpy,scipy,statsmodels,pytest pytest studies/var_495`
  (or any env with those four).
- `calibration.py`: size of the online Granger test under decay, and detection
  after a break (Monte Carlo; results below).

## The model, in short

The model keeps the last `p` rows itself, so a row is just `y_t` (k values) and a
weight. A target row's regressors are `x_t = [y_{t-1}; ...; y_{t-p}]` (`m = k p`).
It keeps exponentially weighted means of `x` and `y` and their centred co-moments
`Sxx` (m x m), `Sxy` (m x k), `Syy` (k x k), updated in mean form (a West/Welford
weighted update), plus `W = sum w` and `V = sum w^2`. Everything is read from these
sums at any time:

- slopes `B = Sxx^{-1} Sxy` (Cholesky), constant `c = ybar - B' xbar`;
- `SSE = Syy - Sxy' B`, `n_eff = W^2 / V` (the row count with no decay);
- `sigma_u = SSE n_eff / (W df)`, `df = n_eff - (m + 1)`; `sigma_u_mle = SSE / W`;
- `cov_params = (W / n_eff) (Z'Z)^{-1} kron sigma_u`, with `(Z'Z)^{-1}` in block
  form from `Sxx^{-1}` and `xbar` (no raw `Z'Z` is ever formed);
- Granger Wald for caused set `E`, causing set `J` (lag columns `R`, `q = p|J||E|`):
  `lambda_W = trace(Sigma_EE^{-1} beta' G^{-1} beta)`, `G = (Sxx^{-1})_RR`,
  `beta = B[R, E]`, `Sigma = SSE / df`: the restricted block of a Kronecker product
  is the product of the blocks, so no `(k m) x (k m)` matrix is built;
- `grangercausalitytests`' single-equation `ssr_ftest`, `ssr_chi2test`, `lrtest`
  (and `params_ftest`, equal to `ssr_ftest`): each is a function of
  `lambda_W = (SSR_r - SSR_u) / (SSR_u / df)`, so the restricted fit is never run;
- information criteria for every `p' <= p` from the leading `k p'` block of the
  same sums (`select_order`'s common sample).

Rules held: `pred` is computed before the row's update (2); one row at a time, so
any chunking is bit-identical (3); the emitted `weight_sum` is `W` before the row
and before its decay (8); a zero-weight row, also the first, advances the lag
buffer and decays the sums with no 0/0 (9). A row with a non-finite value is no
target and its value enters the lag buffer, so the next `p` rows are not targets
either (OLS on the complete rows of the lag design).

State: `8 (p k + m + k + m^2 + m k + k^2 + 3)` bytes, `m = k p`, bounded by the
shape: 168 B (k=2, p=1), 648 B (3, 2), 3.0 KB (4, 4), 17.5 KB (10, 4), 101 KB
(20, 5). Per row: `O(m^2 + m k)` for the update; a fit (for `pred`) is one
`m x m` Cholesky, `O(m^3 / 3)`, or `O(m^2)` if the Rust model keeps the factor
up to date by rank-one updates (not studied). `Sxx` is symmetric, so half of it
need be stored.

## Results (no decay, against `VAR(y).fit(p)`)

Relative error `max|a - b| / max|b|` (p-values: worst `|a/b - 1|` over every test):

| T | k | p | params | sigma_u | sigma_u_mle | cov_params | Granger stat | Granger p | tests |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 2 | 1 | 4.7e-16 | 1.4e-16 | 1.9e-16 | 1.2e-16 | 5.2e-15 | 4.9e-15 | 4 |
| 40 | 3 | 2 | 2.2e-15 | 3.5e-16 | 3.5e-16 | 4.8e-16 | 1.4e-14 | 3.5e-14 | 24 |
| 300 | 2 | 1 | 4.0e-16 | 1.6e-15 | 1.6e-15 | 1.5e-15 | 1.2e-14 | 8.3e-15 | 4 |
| 300 | 3 | 2 | 1.2e-15 | 7.9e-16 | 1.0e-15 | 1.6e-15 | 9.8e-15 | 3.9e-13 | 24 |
| 500 | 5 | 3 | 9.8e-16 | 1.6e-15 | 1.6e-15 | 3.1e-15 | 4.3e-14 | 4.2e-13 | 360 |
| 3000 | 4 | 4 | 4.3e-15 | 3.9e-15 | 3.9e-15 | 2.7e-15 | 1.5e-14 | 4.6e-12 | 100 |
| 3000 | 2 | 8 | 1.3e-15 | 1.8e-15 | 1.9e-15 | 2.3e-14 | 2.2e-14 | 1.0e-13 | 4 |

The tests are every (caused set, causing set) pair of disjoint non-empty subsets,
each as `kind="f"` and `kind="wald"`; degrees of freedom are equal. The largest
p-value error is on p-values of 1e-80 to 1e-177, where the F tail amplifies the
statistic's last bits; it is the statistic's 1e-14 seen through `F.sf`.

Other checks (all in `test_online_var.py`):

- **Out of sample**: `pred[t]` equals `VAR(y[:t]).fit(p).forecast(...)` to 7.2e-15,
  from the first row with `m + 1` targets (before it, NaN).
- **Chunk invariance**: 1 chunk and 1000 give bit-identical predictions, weight
  sums and state (with decay 0.995).
- **Decay and weights**: against per-equation `statsmodels` `WLS` with weights
  `w_t lam^(T-1-t)`, row weights uniform on [0, 2] with 10% zeros (and a zero first
  target row): params 1.3e-15 / 2.8e-15 / 4.9e-16 and SSE 3.1e-15 / 3.8e-16 /
  2.2e-16 at `lam` 1 / 0.99 / 0.9.
- **Missing rows**: NaN or inf rows: params equal OLS on the complete rows of the lag
  design to 7.1e-16, and `n_eff` equals their count.
- **Lag order**: `info_criteria(p')` for p' = 1..6 from one VAR(6)'s sums equals
  `VAR.select_order(6).ics` (AIC, BIC, HQIC, FPE) to 1.4e-14.
- **`grangercausalitytests`** (k = 2, T = 60, p = 3): its `ssr_ftest`,
  `ssr_chi2test`, `lrtest` and `params_ftest` statistics and p-values agree to
  2.2e-14. Its `ssr_ftest` *statistic* equals `VAR.test_causality`'s F to 5e-15,
  but the p-values differ, by design: `grangercausalitytests` refers it to
  F(q, df), `test_causality` to F(q, k df) (Lutkepohl's choice: the restriction is
  tested inside the system of `k` equations). At T = 60: 0.00146 against 0.00087
  for the true `0 -> 1` effect; 0.85 against 0.85 under the null. The model offers
  both (`kind="ssr_f"` and `kind="f"`). The gap shrinks as `df` grows.

### Series far from zero (conditioning)

A VAR with a constant is shift-invariant, so the reference here is statsmodels on
`y - level` (an exact subtraction), and both fits of the raw `y` are compared with
it (T = 300, k = 3, p = 3):

| level | online slopes | statsmodels slopes | online Granger stat | statsmodels Granger stat |
|---|---|---|---|---|
| 1e2 | 1.2e-14 | 1.9e-14 | 1.2e-14 | 1.7e-11 |
| 1e4 | 8.8e-13 | 5.3e-12 | 8.1e-13 | 1.4e-07 |
| 1e6 | 9.6e-11 | 1.5e-10 | 1.5e-10 | 8.5e-04 |

statsmodels' slopes come from an SVD least-squares solve and are fine; its
`cov_params` inverts the raw `Z'Z`, whose condition number grows as `level^2`, so its
Granger statistic loses about `level^2 * 1e-16`. The mean-form sums never form
`Z'Z`, and the online error is the running mean's own rounding (an ulp of `level`
against the series' spread), about `level * 1e-16`. This is why the
"exact against statsmodels" test needs data near zero, or a looser tolerance that
grows with the level: past a level of about 1e3 the online model is the more
accurate of the two.

### Under decay (no oracle: statsmodels has no forgetting)

CALIBRATION

## Differences from statsmodels, and why

1. **`test_causality(caused)` with `causing=None` fails for k > 2** in statsmodels
   0.15.0 (`IndexError`): it sets `causing = [names[c] for c in caused_ind]` (the
   caused variables, not the rest), so the restriction count is wrong. For k = 2 it
   happens to give the right count. The study passes `causing` explicitly; the
   online model's `causing=None` means "all the others".
2. **The F test's denominator**: `k df` (`test_causality`) against `df`
   (`grangercausalitytests`); see above. Same statistic.
3. **Decay changes what `sigma_u` and `cov_params` mean** (none of this exists in
   statsmodels): with no decay both reduce to statsmodels' exactly.
4. **Conditioning** far from zero: the online model is more accurate (above).
5. Not compared: `test_inst_causality` (it reads only `sigma_u`, so it follows from
   the same sums; in statsmodels 0.15.0 its `sig_mask[causing_ind, caused_ind] = 1`
   pairs the indices elementwise, so with two or more variables on each side it
   fails: k = 4, `causing=[0, 1]` raises `IndexError`, while `causing=[0]` runs),
   impulse responses, forecast-error variance decomposition,
   `trend` other than `"c"`, exogenous regressors.

## Caveats and what was not done

- The Rust model, `docs/EXTENDING.md`'s places, state fixtures, the measured Rust
  speed and memory, and the PLAN entry are the task proper. PLAN puts 495 after 494
  (the oracle-library scan), which this study does not wait for.
- `p` is fixed per model; a lag order chosen online reads `info_criteria` from a
  VAR(pmax)'s sums, which is exact for `select_order`'s common sample.
- `pred` refits from the sums each row (a Cholesky of `m x m`); a Rust model
  would cache the factor or update it.
- Row weights are importance weights read through Kish's `n_eff`; frequency
  weights (a weight of 2 as two rows) would want `n = W` instead. Under no decay
  with unit weights the two agree.
