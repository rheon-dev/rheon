<!-- kind: record -->
# Study 495: an online VAR(p) with Granger-causality tests

A numpy prototype of PLAN task 495 ("a vector autoregression by exponentially
weighted least squares on lags the model keeps, exact against `statsmodels` `VAR`
with no decay; Granger tests read from the same sums (Wald, as
`VAR.test_causality`)"), checked against `statsmodels` 0.15.0. Python only, no
Rust: the model in `rheon-core` behind `OnlineModel`, `docs/EXTENDING.md`'s places
and the measured Rust cost are the task proper, not this study. Nothing here is
collected by the repo's pytest (`testpaths = ["tests"]`), and nothing is added to
`pyproject.toml`: `statsmodels` and `scipy` are already in the dev group.

## Files and how to run them

- `scripts/study_495.py`: runs every comparison and prints the tables below
  (about 15 s; `--drift` adds the window's long-stream drift, about 10 minutes).
- `studies/var_495/online_var.py`: the model (`OnlineVAR`), its rolling-window
  variant (`WindowVAR`) and a feeder (`run`). The update equations, the fit, the
  Granger statistics, the impulse responses and the state size are in its
  docstrings.
- `studies/var_495/data.py`: seeded synthetic VAR data (no data files).
- `studies/var_495/compare.py`: each check against statsmodels, returning its
  worst error.
- `studies/var_495/test_online_var.py`: the same checks as asserts (33 tests,
  about 12 s): `python -m pytest studies/var_495` in any env with numpy, scipy,
  statsmodels and pytest.
- `studies/var_495/calibration.py`: size of the online Granger test under decay,
  and detection after a break (Monte Carlo; results below).

Environment used: a standalone venv (Python 3.13, numpy 2.5.5, scipy 1.18.1,
statsmodels 0.15.0), not `uv sync`, which would build the Rust extension.

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
  same sums (`select_order`'s common sample);
- impulse responses `Phi_i = sum_{j<=min(i,p)} Phi_{i-j} A_j`, their orthogonalised
  form `Phi_i P` (`P` the lower Cholesky factor of `sigma_u`, statsmodels'
  ordering), the FEVD, and the asymptotic covariance of `vec(Phi_i)`
  (`G_i Sigma_alpha G_i'`, Lutkepohl 3.7.5, `Sigma_alpha` the slope block of
  `cov_params`).

Rules held: `pred` is computed before the row's update (2); one row at a time, so
any chunking is bit-identical (3); the emitted `weight_sum` is `W` before the row
and before its decay (8); a zero-weight row, also the first, advances the lag
buffer and decays the sums with no 0/0 (9). A row with a non-finite value is no
target and its value enters the lag buffer, so the next `p` rows are not targets
either (OLS on the complete rows of the lag design).

## State size

No window: `8 (p k + m + k + m^2 + m k + k^2 + 3)` bytes, `m = k p`, bounded by
the shape.

A window of `W` rows stores the rows themselves, `W k` values and `W` weights;
the lag buffer is the window's last `p` rows, and one counter (targets of
positive weight, so an emptied window resets exactly) is added:
`8 (W k + W + m + k + m^2 + m k + k^2 + 4)` bytes, `8 (W (k + 1) - p k + 1)` more
than with no window. With unit weights only, the `W` weights need not be stored.
`Sxx` and `Syy` are symmetric, so about half of `m^2 + k^2` need be stored.

| k | p | no window | W = 250 | W = 1000 |
|---|---|---|---|---|
| 2 | 1 | 168 B | 6.2 KB | 24.2 KB |
| 2 | 5 | 1.2 KB | 7.1 KB | 25.1 KB |
| 5 | 2 | 1.6 KB | 13.6 KB | 49.6 KB |
| 5 | 5 | 6.7 KB | 18.5 KB | 54.5 KB |
| 10 | 1 | 2.7 KB | 24.6 KB | 90.6 KB |
| 10 | 5 | 25.7 KB | 47.3 KB | 113.3 KB |

Per row: `O(m^2 + m k)` for the update (twice that under a window: an add and a
remove); a fit (for `pred`) is one `m x m` Cholesky, `O(m^3 / 3)`, or `O(m^2)` if
the Rust model keeps the factor up to date by rank-one updates (not studied).
Impulse responses and the FEVD add no state: they are reads of the fit, `O(H p
k^3)` for `H` periods; the covariance of the impulse responses is
`O(H k^6 p^2)`, a read on demand, not a per-row cost.

## Results against statsmodels (no decay)

Relative error `max|a - b| / max|b|` (Granger: worst `|a/b - 1|` over every test).

### The coordinator's grid: k = 2, 5, 10 by p = 1, 2, 5 (T = 400)

`VAR(y).fit(p)` (`trend="c"`, the default) on the same rows:

| k | p | params | sigma_u | sigma_u_mle | cov_params | Granger stat | Granger p | tests |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 7.5e-16 | 1.4e-15 | 1.2e-15 | 1.3e-15 | 2.1e-14 | 1.1e-14 | 4 |
| 2 | 2 | 2.5e-15 | 8.1e-16 | 8.2e-16 | 1.3e-15 | 2.3e-15 | 2.0e-15 | 4 |
| 2 | 5 | 1.5e-15 | 7.0e-16 | 7.2e-16 | 1.4e-15 | 2.4e-15 | 4.8e-15 | 4 |
| 5 | 1 | 9.2e-16 | 1.5e-14 | 1.4e-14 | 1.2e-14 | 1.6e-13 | 4.3e-12 | 360 |
| 5 | 2 | 4.2e-15 | 9.5e-16 | 1.0e-15 | 1.7e-15 | 3.0e-14 | 4.1e-13 | 360 |
| 5 | 5 | 2.0e-15 | 1.8e-15 | 1.9e-15 | 1.0e-14 | 4.4e-14 | 5.7e-13 | 360 |
| 10 | 1 | 4.3e-15 | 6.3e-15 | 6.4e-15 | 5.0e-15 | 1.2e-13 | 3.6e-12 | 204 |
| 10 | 2 | 1.5e-15 | 2.3e-15 | 2.3e-15 | 8.9e-15 | 6.0e-14 | 6.3e-13 | 204 |
| 10 | 5 | 2.2e-15 | 1.7e-15 | 1.8e-15 | 3.1e-14 | 4.2e-14 | 2.6e-13 | 204 |

The tests are every (caused set, causing set) pair of disjoint non-empty subsets
up to k = 5, each as `kind="f"` and `kind="wald"`; at k = 10 (3^10 pairs) every
single pair, each variable against all the others, and two splits. Degrees of
freedom are equal in every test.

### Impulse responses and FEVD (10 periods, same fits)

Against `VARResults.irf(10)` (`.irfs`, `.orth_irfs`, `.cov()`) and
`VARResults.fevd(10).decomp`:

| k | p | irf | orth irf | irf cov | fevd |
|---|---|---|---|---|---|
| 2 | 1 | 4.4e-16 | 8.6e-16 | 1.3e-15 | 3.3e-16 |
| 2 | 2 | 1.1e-15 | 9.1e-16 | 2.2e-15 | 5.6e-16 |
| 2 | 5 | 3.8e-16 | 3.8e-16 | 1.4e-15 | 2.2e-16 |
| 5 | 1 | 2.1e-15 | 7.8e-15 | 5.4e-15 | 4.8e-15 |
| 5 | 2 | 1.1e-15 | 1.0e-15 | 2.1e-15 | 8.9e-16 |
| 5 | 5 | 7.1e-16 | 9.6e-16 | 3.4e-15 | 6.7e-16 |
| 10 | 1 | 2.0e-15 | 3.6e-15 | 6.2e-15 | 1.6e-15 |
| 10 | 2 | 1.8e-15 | 1.6e-15 | 8.2e-15 | 1.1e-15 |
| 10 | 5 | 2.5e-15 | 2.5e-15 | 1.2e-14 | 8.2e-16 |

These are cheap reads of the state: the IRF needs only the slopes, the orthogonal
form and the FEVD add `sigma_u`'s Cholesky factor, and the IRF covariance adds
`cov_params`, all of which the sums already give. Not done: the covariance of the
orthogonalised IRF (`irf.cov(orth=True)`, Lutkepohl 3.7.8), which also needs
the covariance of `vech(sigma_u)`, `2 D+ (Sigma kron Sigma) D+' / n`: a function
of `sigma_u` and `n_eff` alone, so equally cheap, but not written; and
statsmodels' Monte Carlo error bands (`irf_errband_mc`), which simulate and are no
read of any state. Under decay the IRF and FEVD are those of the decayed fit
(no oracle).

### Earlier grid

| T | k | p | params | sigma_u | sigma_u_mle | cov_params | Granger stat | Granger p | tests |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 2 | 1 | 4.7e-16 | 1.4e-16 | 1.9e-16 | 1.2e-16 | 5.2e-15 | 4.9e-15 | 4 |
| 40 | 3 | 2 | 2.2e-15 | 3.5e-16 | 3.5e-16 | 4.8e-16 | 1.4e-14 | 3.5e-14 | 24 |
| 300 | 2 | 1 | 4.0e-16 | 1.6e-15 | 1.6e-15 | 1.5e-15 | 1.2e-14 | 8.3e-15 | 4 |
| 300 | 3 | 2 | 1.2e-15 | 7.9e-16 | 1.0e-15 | 1.6e-15 | 9.8e-15 | 3.9e-13 | 24 |
| 500 | 5 | 3 | 9.8e-16 | 1.6e-15 | 1.6e-15 | 3.1e-15 | 4.3e-14 | 4.2e-13 | 360 |
| 3000 | 4 | 4 | 4.3e-15 | 3.9e-15 | 3.9e-15 | 2.7e-15 | 1.5e-14 | 4.6e-12 | 100 |
| 3000 | 2 | 8 | 1.3e-15 | 1.8e-15 | 1.9e-15 | 2.3e-14 | 2.2e-14 | 1.0e-13 | 4 |

### Rolling window

`WindowVAR` fits the last `W` rows of the stream: a target counts while it and its
`p` lags are inside the window, so the fit is `VAR(y[t-W+1 : t+1]).fit(p)`, with
`W - p` targets. Each row is added by the update above, then the target that has
just lost its oldest lag is removed by the reverse update (`W' = W - w`,
`e = x - xbar`, `xbar' = xbar - (w / W') e`, `S' = S - w (W / W') e e'`). When
the window holds no target of positive weight the sums are reset to zero, so an
emptied window leaves no residue and no 0/0. No decay with a window in this study:
combining them needs the leaving row's weight times `lam` to its age, a scalar per
row, not studied.

W = 120, k = 3, p = 2, T = 700, against `VAR(window rows).fit(2)`:

| pred (every row) | params | sigma_u | cov_params | Granger stat | n_eff |
|---|---|---|---|---|---|
| 1.4e-14 | 6.1e-15 | 4.5e-15 | 4.8e-15 | 1.2e-14 | exact (`W - p`) |

`pred[t]` is checked at every row past the first full window against
`VAR(y[t-W:t]).fit(2).forecast(...)`. With row weights uniform on [0, 2], 10%
zeros, a zero-weight first target and NaN/inf rows inside the window, params equal
statsmodels `WLS` on the window's complete rows to 2.0e-14 (W = 150, checked every
25 rows).

### Other checks (all in `test_online_var.py`, all passing)

- **Out of sample**: `pred[t]` equals `VAR(y[:t]).fit(p).forecast(...)` to 7.2e-15,
  from the first row with `m + 1` targets (before it, NaN).
- **Chunk invariance**: 1 chunk and 1000 give bit-identical predictions, weight
  sums and state (with decay 0.995).
- **Decay and weights**: against per-equation `statsmodels` `WLS` with weights
  `w_t lam^(T-1-t)`, row weights uniform on [0, 2] with 10% zeros (and a zero first
  target row): params 1.3e-15 / 2.8e-15 / 4.9e-16 and SSE 3.1e-15 / 3.8e-16 /
  2.2e-16 at `lam` 1 / 0.99 / 0.9.
- **Zero-weight first rows**: ten zero-weight rows to open a decayed stream leave
  every sum finite and `weight_sum` 0 through them.
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

## Numerical risk

**Level offsets.** A VAR with a constant is shift-invariant, so the reference is
statsmodels on `y - level` (an exact subtraction), and both fits of the raw `y`
are compared with it (T = 300, k = 3, p = 3):

| level | online slopes | statsmodels slopes | online Granger stat | statsmodels Granger stat |
|---|---|---|---|---|
| 1e2 | 1.2e-14 | 1.9e-14 | 1.2e-14 | 1.7e-11 |
| 1e4 | 8.8e-13 | 5.3e-12 | 8.1e-13 | 1.4e-07 |
| 1e6 | 9.6e-11 | 1.5e-10 | 1.5e-10 | 8.5e-04 |

statsmodels' slopes come from an SVD least-squares solve and are fine; its
`cov_params` inverts the raw `Z'Z`, whose condition number grows as `level^2`, so
its Granger statistic loses about `level^2 * 1e-16`. The mean-form sums never
form `Z'Z`; the online error is the running mean's rounding (an ulp of `level`
against the series' spread), about `level * 1e-16`. So "exact against
statsmodels" holds for data near zero; past a level of about 1e3 the online
model is the more accurate of the two, and a test there needs the shifted
reference or a tolerance that grows with the level.

**Far-tail p-values.** The p-value's relative error is the statistic's
times `|d ln p / d ln stat|`, which is large in the far tail. The largest
p-value gaps are on vanishing p-values: 4.6e-12 on a p-value of 1.5e-274
(T = 3000, k = 4, p = 4), 4.3e-12 on 8.4e-125 (k = 5, p = 1), 3.6e-12 on 3.7e-230
(k = 10, p = 1), each with the statistic agreeing to 7e-15 to 1.5e-14. Over the
p-values above 1e-10, which are the ones a decision reads, the worst gap is
6.5e-13 (2e-13 at T = 400).

**Window down-dating.** Removing a row subtracts `w (W / W') e e'` from the sums:
a cancellation whose rounding is never washed out (no decay forgets it). Measured
drift of the final fit against statsmodels on the last window (W = 200, k = 3,
p = 2; reference shifted by the level as above; `--drift`):

| level | T | slopes | sigma_u | Granger stat |
|---|---|---|---|---|
| 0 | 1e4 | 2.7e-14 | 3.8e-14 | 2.7e-15 |
| 0 | 1e5 | 6.7e-14 | 3.7e-14 | 1.8e-13 |
| 0 | 1e6 | 2.7e-13 | 5.8e-13 | 3.7e-13 |
| 1e2 | 1e4 | 3.9e-13 | 2.6e-13 | 1.6e-13 |
| 1e2 | 1e5 | 5.1e-13 | 4.0e-13 | 3.9e-13 |
| 1e2 | 1e6 | 4.0e-12 | 2.6e-12 | 2.2e-13 |
| 1e4 | 1e4 | 6.2e-11 | 4.2e-11 | 2.4e-11 |
| 1e4 | 1e5 | 1.3e-10 | 7.8e-11 | 5.0e-11 |
| 1e4 | 1e6 | 3.2e-10 | 2.0e-10 | 1.3e-10 |

The drift grows slowly with the stream, about tenfold over a hundredfold more
rows (a random walk of rounding), and about in proportion to the level. At a
level the window is the weaker of the two online forms: 6.2e-11 at 1e4 after
1e4 rows, against the add-only model's 8.8e-13 (T = 300, above), because each
removal subtracts a rank-one term of the level's size from sums of the spread's
size. A Rust model that must stay exact over very long streams, or far from zero,
can rebuild the sums from the stored window every `N` rows (`O(W m^2)`,
amortised `O(m^2)` a row when `N ~ W`), at no state cost: the rows are already
stored. That remedy is not measured here.

**Not measured.** A near-singular `Sxx` (collinear series, a constant series, a
cointegrated system in levels) fails the Cholesky; the Rust model needs a guard
there (withhold the fit, as below `m + 1` targets).

## Under decay (no oracle: statsmodels has no forgetting)

`python studies/var_495/calibration.py 1000` (about 4 minutes): a bivariate
VAR(1) with N(0, I) noise, `1 -> 0` absent, the test read once at the stream's
end, 1000 streams each (Monte Carlo standard error of a 5% rate: 0.007):

| lam | T | n_eff | size of `f` at 5% | size of `ssr_f` at 5% | mean `sigma_u` diagonal (truth 1) |
|---|---|---|---|---|---|
| 1 | 500 | 499 | 0.057 | 0.056 | 1.001 |
| 0.99 | 2000 | ~199 | 0.046 | 0.046 | 0.999 |
| 0.95 | 2000 | ~39 | 0.035 | 0.031 | 1.001 |

- `sigma_u = SSE n_eff / (W df)` is unbiased for the noise under decay. The first
  draft used statsmodels' `SSE / df` there, which estimates `Sigma V / W`, about
  `Sigma / (1 + lam)`: half the noise. The Wald statistic was right either way (it
  needs `SSE / df` times `Sxx^{-1}`, which is what the decayed fit's variance is),
  so only the reported `sigma_u`, `sigma_u_mle` and `cov_params` moved.
- With Kish's `n_eff` for the degrees of freedom the test holds its size at
  `n_eff ~ 200` and is conservative at `n_eff ~ 40` (0.031 to 0.035 against 0.05):
  a decayed sample is not `n_eff` independent rows, and the F reference is an
  approximation there. A `min_weight` of a few hundred keeps the p-values honest.
- Detection after a break (one stream; the `1 -> 0` coefficient switches from 0 to
  0.25 at row 1000; rows until the 1% test rejects at every later row): 148 rows
  with no decay, 66 at `lam = 0.995`, 435 at 0.99, never at 0.98 (with `n_eff ~
  100` the test's power at this effect is well below 1, so it dips above 1% now and
  then). One path only: an illustration, not a power study.

## Proposed tolerances for the Rust model's tests

Relative to the reference's largest entry, data near zero (level 0, unit-scale
noise), no decay:

| quantity | worst measured | proposed | why |
|---|---|---|---|
| params, sigma_u, sigma_u_mle, cov_params | 3.1e-14 | 1e-12 | 30x the worst over 16 fits up to k = 10, p = 8; the m x m solve's rounding grows with `m` |
| IRF, orthogonal IRF, FEVD, IRF covariance | 1.2e-14 | 1e-12 | the same sums through a short matrix recursion |
| Granger statistic (`f`, `wald`, single-equation forms) | 1.6e-13 | 1e-11 | a quadratic form in a small difference; 60x margin |
| Granger p-value, relative | 4.6e-12 | 1e-10 | far-tail amplification (p down to 1e-177); or compare `log p` absolutely |
| decay and weights against WLS | 2.8e-15 | 1e-12 | the same arithmetic as the fit |
| window, short stream | 2.0e-14 | 1e-12 | the add and remove each round once |
| window, long stream near zero | 5.8e-13 (1e6 rows) | 1e-11 | drift grows about tenfold per hundredfold rows; at a level of 1e4 it is 3.2e-10, so a level needs the shifted reference |
| level offsets | `level * 1e-16` | shifted reference | statsmodels' own error at a level is `level^2 * 1e-16` |

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
   the orthogonalised IRF's covariance, `trend` other than `"c"`, exogenous
   regressors.

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
- This record has no row in `docs/README.md`'s Records table yet, which
  `tests/test_doc_structure.py` requires; the study's branch does not edit that
  index.
