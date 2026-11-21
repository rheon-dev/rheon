# Study 500: online GLMs by recursive IRLS

**Status (2026-11-21): parts 1 and 2 of 3 done; part 3 missing.** Part 1
(sections 1 to 6): Poisson and Gamma, log link, no decay, no window, 10 seeds.
Part 2 (section 7): the log-link binomial and separation, decay, a rolling
window, zero-weight rows and zero counts, and a step cap in place of the
refit. Missing, for part 3: the row-by-row exact oracle through statsmodels'
family objects, 50 seeds, and a sweep of `refit_rows` and the ridge. The
scripts are `scripts/study_500.py` (part 1) and `scripts/study_500_part2.py`
(part 2, which imports part 1's generator); both generate their own data from
seeded numpy generators and read nothing from disk. The proposal is task 500
in the models scan (branch `vm-tools/oracle-gap-scan`,
`vm-tools/research/494-MODELS-SCAN.md`, section 6).

Environment: `~/rheon/.venv` with numpy, scipy, statsmodels 0.15.0, glum
3.5.0 (glum installed without trouble).

## 1. The math

Row `t` is `(x_t, y_t, w_t)`, `x_t` in R^k with an intercept column. Log link:
`eta = x'beta`, `mu = exp(eta)`, `dmu/deta = mu`. Variance `V(mu) = mu`
(Poisson), `V(mu) = mu^2` (Gamma, dispersion `phi` aside).

Per-row half deviance (the negative log likelihood at scale 1, up to a term
free of beta): Poisson `mu - y*eta`; Gamma `y/mu + eta`.

**Score** of row t at beta (the gradient of `-halfdev`):
`s_t x_t` with `s_t = (y_t - mu_t)(dmu/deta)/V(mu_t)`, which is `y - mu`
(Poisson) and `(y - mu)/mu` (Gamma).
**Fisher information** of row t: `W_t x_t x_t'` with
`W_t = (dmu/deta)^2 / V(mu_t)`: `W = mu` (Poisson), `W = 1` (Gamma).
**Working response**: `z_t = eta_t + (y_t - mu_t)/(dmu/deta) = eta_t + (y_t - mu_t)/mu_t`.

**The recursion implemented** (`OnlineGLM.step`). State: `G` (k x k), `h`
(k), `beta` (k), the ridge `R` (a constant), `weight_sum`. Start `G = 0`,
`h = 0`, `beta = 0`. For each row, with `eta_t = x_t' beta_{t-1}` and `mu_t`,
`W_t`, `z_t` evaluated at `beta_{t-1}`:

```
pred_t = mu_t                                   (emitted before the update)
if w_t == 0: stop                               (nothing advances, nothing is divided)
G_t    = lambda * G_{t-1} + w_t W_t x_t x_t'     (lambda = 1 in this part)
h_t    = lambda * h_{t-1} + w_t W_t z_t x_t
beta_t = (G_t + R I)^{-1} h_t
```

This is one Fisher-scoring (Newton, for the canonical Poisson link exactly
Newton) step per row on the online objective
`F_t(beta) = sum_i w_i halfdev_i(beta) + R/2 |beta|^2`, with each past row's
curvature frozen at the fit that scored it. Because `h_{t-1} = (G_{t-1} + R I)
beta_{t-1}` holds by construction, the sums form is the same as the increment
form `beta_t = beta_{t-1} + (G_t + R I)^{-1} w_t s_t x_t` (checked by the
script: max difference 4.5e-15 Poisson, 9e-16 Gamma over 500 rows). With
`R > 0`, `G + R I` is positive definite from the first row, so the first row
and every row before `k` rows have arrived solve without a division by zero.
A zero-weight row (first or later) leaves `G`, `h`, `beta` bit-identical (the
script checks it against the same stream with those rows removed).
`|eta|` is clipped at 30 before `exp` and the clips counted.

**Cost and state.** Per row: the outer product `O(k^2)`, a dense solve
`O(k^3)` here (a Cholesky of `G + R I`; a rank-one update of the factor or of
the inverse makes it `O(k^2)`, as the increment form shows). State:
`k(k+1)/2 + 2k + 2` floats per target, `huber`'s size; `W` depends on the
target, so `T` targets need `T` Grams. In numpy (a Python loop) the 100k-row
pass takes 1.9 s at k = 5 and 3.0 to 3.6 s at k = 20.

**Weights.** The batch fits get no weights here (all `w = 1`). A row weight
in this recursion means "this row counts `w` times" in the likelihood, which
is statsmodels' `freq_weights`, not `var_weights`: the point estimates agree
either way, but `var_weights` keeps `nobs` at the row count and so gives a
different scale estimate and standard errors for Gamma. glum's
`sample_weight` normalises by its sum and gives the same coefficients.

## 2. The batch fits and how the ridge is matched

Both libraries minimise a mean-form objective,
`(1/sum w) sum w_i halfdev_i(beta) + alpha/2 |beta|^2`. The online objective
`F_t` divided by `sum w` is that with **`alpha = R / sum w`**, so at a
checkpoint of `n` rows the batch fits use `alpha = R/n` on the same `n` rows.
The intercept is column 0 of `X` and is penalised like the rest
(`fit_intercept=False` in glum).

- **statsmodels**: `GLM(y, X, family, Log()).fit_regularized(alpha=R/n,
  L1_wt=0)`, which routes to a BFGS minimisation of `-loglike/nobs +
  alpha/2 |beta|^2`. For Gamma, statsmodels' `loglike` profiles the scale at
  each beta, which is not the penalised deviance; the script pins `scale=1`
  on the model's `loglike` and `score`, which makes the objective the one
  above. BFGS stops at a gradient of about 1e-5 (largest seen 9.7e-6), so
  the statsmodels ridge fit is accurate to about 1e-5; the unpenalised
  `fit(method="IRLS", tol=1e-12)` supplies the standard errors (`bse`; for
  Gamma with the Pearson scale) and the gap to the truth.
- **glum**: `GeneralizedLinearRegressor(family, link="log", alpha=R/n,
  l1_ratio=0, fit_intercept=False, gradient_tol=1e-10)`.
- The two batch fits agree to within 0.0024 SE everywhere (the floor of the
  comparison; most of it is BFGS's stopping rule).

## 3. Data

`X`: intercept plus `k - 1` standard normal features. `beta_true[0] = 0.5`,
the rest `N(0, (0.5/sqrt(k-1))^2)`, so `eta` has sd about 0.5. Poisson
`y ~ Poisson(mu)`; Gamma `y ~ Gamma(shape 2, scale mu/2)` (`phi = 0.5`).
Generator `numpy.random.default_rng([500, seed, k, family])`, seeds 0 to 9.

## 4. Results

The gap is `max_j |beta_online_j - beta_batch_j|` (abs) and the same divided
by the batch `bse_j` (SE), each the max over the k coefficients; mean and max
over the 10 seeds. Three variants of the online fit:

- **A**: `R = 1`, no refit (the plain recursion).
- **B**: `R = 10`, no refit.
- **C**: `R = 1` with `refit_rows = 200`: after 200 rows the buffer is fitted
  to convergence by penalised IRLS, and `G`, `h` are rebuilt at that fit;
  the buffer is then dropped (the scan's proposed start-up remedy,
  `glm_refit_rows`, as `huber` and `quantile` have).

### 4.1 The plain recursion fails at start-up (A)

| family | k | seeds failed (gap > 10 SE at 100k) |
|---|---|---|
| Poisson | 5 | 0 / 10 (but max 3.4 SE at 100k) |
| Poisson | 20 | 6 / 10 (stuck: gap 2.2 abs, 894 SE at 100k) |
| Gamma | 5 | 2 / 10 (overflow) |
| Gamma | 20 | 5 / 10 (overflow) |

Two mechanisms. **Gamma** overflows: before `k` rows have arrived the fit is
underdetermined, an early step sends `eta` strongly negative on some row,
`(y - mu)/mu` is then of order `y e^{|eta|}`, and the next step is enormous;
`eta` hits the clip and the fit never returns (gap 1e9 to 1e11 abs).
**Poisson** does not overflow but sticks: a row scored at a too-large `mu`
enters `G` with weight `W = mu` frozen at that fit, so `G` is huge in that
direction and every later step along it is tiny; with no decay it never
washes out (gap 2 to 3 abs, unchanged from 1k to 100k rows). This is the
frozen-weight defect the scan named as the first risk, and it is not a
statistical small-sample effect: it does not shrink with `n`.

### 4.2 Gaps to the batch fits with the start-up remedied

**C (`R = 1`, `refit_rows = 200`): every seed converges.** Gap to statsmodels
(ridge matched); glum's is the same to the 3rd digit (in brackets where it
differs):

| family | k | n | abs, max | SE, mean | SE, max |
|---|---|---|---|---|---|
| Poisson | 5 | 1k | 5.3e-3 | 0.098 | 0.203 |
| Poisson | 5 | 10k | 9.9e-4 | 0.065 | 0.119 |
| Poisson | 5 | 100k | 1.19e-4 (1.21e-4) | 0.031 | 0.047 (0.048) |
| Poisson | 20 | 1k | 1.36e-2 | 0.376 | 0.522 |
| Poisson | 20 | 10k | 3.0e-3 | 0.272 | 0.372 |
| Poisson | 20 | 100k | 4.2e-4 | 0.137 | 0.164 (0.165) |
| Gamma | 5 | 1k | 4.2e-3 | 0.113 | 0.172 (0.173) |
| Gamma | 5 | 10k | 5.7e-4 | 0.069 | 0.080 |
| Gamma | 5 | 100k | 8.1e-5 (8.6e-5) | 0.033 (0.035) | 0.036 (0.038) |
| Gamma | 20 | 1k | 1.08e-2 | 0.417 | 0.492 (0.491) |
| Gamma | 20 | 10k | 2.3e-3 | 0.298 | 0.323 |
| Gamma | 20 | 100k | 3.4e-4 (3.5e-4) | 0.144 (0.146) | 0.153 (0.155) |

**B (`R = 10`, no refit)**: Poisson converges on every seed (max SE gap
0.294 / 0.149 / 0.055 at k = 5 and 1.72 / 1.00 / 0.385 at k = 20, for 1k /
10k / 100k); Gamma k = 5 converges (max 2.05 / 0.65 / 0.22 SE); Gamma k = 20
still overflows on 1 seed of 10. A larger ridge alone is not a remedy for
Gamma at k = 20, and it costs accuracy where it works (B's gaps are 1.5 to 3
times C's).

Reading the table: the abs gap falls 4.5 to 8x per decade of rows (between
`1/sqrt(n)` and `1/n`, nearer `1/n` from 10k on) while `bse` falls 3.2x, so
the SE gap falls 1.4 to 2.6x per decade. At k = 20 it is about 0.15 SE at 100k
rows. The online fit's error is dominated by the rows' frozen weights, which
is an `O(1/n)` bias in beta against an `O(1/sqrt(n))` noise.

### 4.3 Gap to the truth at 100k rows (C)

Max over coefficients of `|beta - beta_true| / bse`, mean over seeds, online
vs batch IRLS: Poisson k = 5 1.65 vs 1.66; Poisson k = 20 2.17 vs 2.16;
Gamma k = 5 1.72 vs 1.73; Gamma k = 20 1.89 vs 1.87. (The expected max of k
absolute standard normals is about 1.6 for k = 5 and 2.2 for k = 20, so
these are the expected sizes.) The online fit is statistically as good as the
batch fit at 100k rows: its excess over the batch fit's own error is under
0.03 SE on average.

## 5. Findings for part 1

1. The recursion as proposed (one IRLS step per row on running `X'WX`,
   `X'Wz`, no decay) **is not usable without a start-up remedy**: half the
   k = 20 runs fail with `R = 1`, by overflow (Gamma) or a permanently stuck
   direction (Poisson).
2. With the refit of the first 200 rows (state bounded by the buffer, dropped
   after) it converges on every seed and stays within 0.52 SE of both
   statsmodels and glum at 1k rows, 0.37 SE at 10k and 0.165 SE at 100k (the
   worst seed, k = 20); at k = 5 within 0.21, 0.12 and 0.05 SE.
3. statsmodels and glum agree with each other to 0.0024 SE; the online gap is
   20 to 200 times that floor, so a ledger tolerance has to be statistical
   (a multiple of `bse` after a stated count), not exact.
4. The two forms of the step (sums vs increment) agree to 1e-15, and a
   zero-weight row, first or later, changes nothing.

## 6. Open questions (for parts 2 and 3)

- How small can `refit_rows` be? 200 rows for k = 20 worked on 10 seeds;
  is `10 k` enough, and does the failure probability with no refit fall
  with a smarter start (intercept at `log(mean y)`, which needs a buffer
  anyway)?
- A guard short of a refit: a trust region on the step (cap `|delta eta|`
  per row) or a cap on the working response. Would it make the no-refit
  recursion safe, and at what cost in accuracy?
- With decay (`lambda < 1`) the frozen weights wash out: does the Poisson
  "stuck" failure then recover without a refit, and does the gap to
  `GLM(freq_weights=lambda^age)` stay below the same SE multiples?
- The row-by-row exact oracle (the same recursion with `W`, `z` from
  statsmodels' family objects, to 1e-10) is not yet written.
- 50 seeds: the max over 10 seeds understates the tail, and the no-refit
  failure rates above are rough.
- The batch reference precision: statsmodels' ridge stops at gradient 1e-5;
  for a ledger, glum or a Newton polish is the tighter reference.

## 7. Part 2: binomial, separation, decay, window, zero weights, step cap

**Baseline: variant C of part 1** (`R = 1`, `refit_rows = 200`) throughout,
except in 7.6, which asks whether a step cap can replace the refit. Run:
`.venv/bin/python scripts/study_500_part2.py checks binomial separation --out a.json`
(about 1 minute) and `... decay window --procs 2` (3 minutes), `... cap
--procs 2` (7 minutes), each with `nohup`. Each job is about 550 MB at its
peak; four workers in parallel were killed for memory once, so the long runs
use two. The batch reference in 7.3 to 7.6 is a penalised IRLS with step
halving (`irls_ridge2`, the refit's solver, tolerance 1e-13), with glum
(`sample_weight`, `alpha = R / sum w`) as the second fit: the two agree to
within 6e-7 SE everywhere, so part 1's 1e-5 BFGS floor of statsmodels' ridge
path is gone. statsmodels' unpenalised IRLS supplies `bse`.

The recursion with decay `lambda` (constant ridge, not decayed):

```
G_t = lambda G_{t-1} + w_t W_t x_t x_t'      (decays on every row, zero weight included)
h_t = lambda h_{t-1} + w_t W_t z_t x_t
beta_t = (G_t + R I)^{-1} h_t
```

Its converged target is the batch fit of `sum_i w_i lambda^(T-i) halfdev_i +
R/2 |beta|^2`. The refit of the first 200 rows weights them `lambda^age`.

### 7.1 Binomial with the log link is not well posed; logit is an extra

`mu = exp(eta)` must stay below 1, and nothing in the recursion or in IRLS
keeps it there. The Fisher weight is `W = mu^2 / (mu (1 - mu)) = mu / (1 - mu)`,
infinite at `mu = 1` and negative above it.

- **The recursion's own start is the boundary**: `beta = 0` is `mu = 1`, so the
  first row's `W` is `1/0`. The plain recursion (A) is NaN from row 1 and never
  recovers (19,999 of 19,999 rows have a non-finite weight).
- With the refit (C), the first 200 rows are NaN too, but the refit (IRLS with
  step halving, which rejects any step where some `mu >= 1`, whose deviance is
  NaN) replaces the whole state, and from there it runs clean.
  - (a) A truth inside the domain (`beta = (-1.5, .3, -.2, .15, .1)`, k = 5,
    19,999 rows, max true `mu` 0.97 after dropping the one row above 1):
    no prediction above 1, gap 9e-4 abs (0.07 SE) to the ridge-matched batch fit.
  - (b) A truth that crosses 1 (`mu >= 1` on 3,996 of 20,000 rows, `y`
    drawn at `min(mu, 1)`): 2,518 predictions above 1 and 2,718 rows with a
    negative or infinite `W`; `beta` stays finite and means nothing.
- **The batch libraries fail on (a)**, a well-specified problem:
  statsmodels (DomainWarning "the Log link function does not respect the
  domain of the Binomial family") raises `ValueError: NaN, inf or invalid
  value detected in weights` from its default start; started at
  `(log mean y, 0, ...)` it converges to a boundary point (max fitted `mu`
  0.99999999993, half deviance 10476.09 against the optimum's 10475.69),
  0.85 SE from the optimum. glum returns all-zero coefficients (`mu = 1`
  everywhere) with only divide-by-zero RuntimeWarnings. On (b) statsmodels
  returns a fit with max `mu` 1.0000000000002, glum only RuntimeWarnings.

So the log-link binomial (the relative-risk model) needs a constrained
solver and is out of scope for this recursion. **Labelled extra, logit link**
(`W = mu (1 - mu)`, `z = eta + (y - mu)/W`, same rows as (a)): variant C is
within 0.076 SE of statsmodels at 20k rows, k = 5, with no clipping.

### 7.2 Separation

**Poisson, log link (in scope)**: `x1` is a 0/1 indicator and `y = 0` on every
row with `x1 = 1`, so the MLE's coefficient on `x1` is minus infinity. k = 5.

| n | online (C) | ridge-matched batch (R = 1; glum identical) | statsmodels IRLS | glum, alpha 1e-12 |
|---|---|---|---|---|
| 1k | -4.77 | -5.08 | -26.37 (bse 11,291) | -24.47 |
| 10k | -6.02 | -7.11 | -29.38 (bse 15,512) | -24.52 |
| 100k | -7.16 | -9.15 | -31.35 (bse 13,330) | -24.51 |

statsmodels reports `converged=True` with **no warning**; glum gives no warning
either; both stop wherever the deviance stops moving (`exp(-25)` is 1e-11).
The ridge-matched fit grows like `log n` (the ridge is fixed while the data
push), and the online fit lags it by 0.30, 1.10 and 1.99 abs: the rows with
`x1 = 1` enter `G` with `W = mu = exp(eta)`, which is ever smaller, so the
direction is starved of curvature and the online coefficient trails the
batch one. In fitted means the difference is negligible (`exp(-7.2)` against
`exp(-9.2)`, both under 1e-3), the online state stays finite, and nothing
clips.

**Logit (labelled extra)**: `y = 1` exactly when `x1 > 0` (complete
separation), k = 5. statsmodels: `x1` coefficient 3.7e15 at 1k
(`converged=True`, an overflow RuntimeWarning only), 5.3e4 and 5.9e5 at 10k
and 100k with `converged=False`. glum (alpha 1e-12): 6,509 and 34,151 with
"The fitted values are numerically 0 or 1 for 99.5% (99.9%) of the
observations", and `LinAlgError: A singular matrix` at 100k. Ridge-matched
batch: 8.84, 18.67, 40.24. Online (C): 7.66, 12.37, 19.78, finite, with 7,584
rows clipped at `|eta| = 30`. With a ridge every fit is finite; without one,
the batch libraries return garbage, and the online fit is bounded by its
ridge but trails the ridge-matched fit by a factor of 2 at 100k.

### 7.3 Decay: half-lives 500 and 5000 rows

Gap of variant C (with decay) to the decay-weighted ridge batch fit, max over
coefficients, in units of the batch `bse` from `GLM(freq_weights =
lambda^(T-t))`; mean / max over 10 seeds. "none" is part 1's variant C (4.2).

| family | k | half-life | 1k | 10k | 100k |
|---|---|---|---|---|---|
| Poisson | 5 | 500 | 0.072 / 0.155 | 0.039 / 0.060 | 0.028 / 0.047 |
| Poisson | 5 | 5000 | 0.095 / 0.199 | 0.034 / 0.053 | 0.011 / 0.015 |
| Poisson | 5 | none | 0.098 / 0.203 | 0.065 / 0.119 | 0.031 / 0.047 |
| Poisson | 20 | 500 | 0.280 / 0.368 | 0.117 / 0.159 | 0.122 / 0.166 |
| Poisson | 20 | 5000 | 0.363 / 0.500 | 0.141 / 0.174 | 0.037 / 0.054 |
| Poisson | 20 | none | 0.376 / 0.522 | 0.272 / 0.372 | 0.137 / 0.164 |
| Gamma | 5 | 500 | 0.095 / 0.123 | 0.067 / 0.074 | 0.065 / 0.075 |
| Gamma | 5 | 5000 | 0.110 / 0.166 | 0.046 / 0.050 | 0.021 / 0.021 |
| Gamma | 5 | none | 0.113 / 0.172 | 0.069 / 0.080 | 0.033 / 0.036 |
| Gamma | 20 | 500 | 0.398 / 0.429 | 0.259 / 0.269 | 0.263 / 0.279 |
| Gamma | 20 | 5000 | 0.412 / 0.483 | 0.191 / 0.203 | 0.083 / 0.084 |
| Gamma | 20 | none | 0.417 / 0.492 | 0.298 / 0.323 | 0.144 / 0.153 |

glum's gap is the same to the third digit in every cell (glum vs the IRLS
reference: at most 6e-7 SE). Abs gaps at 100k, max: Poisson 1.4e-3 / 5.2e-3
(k = 5 / 20) at half-life 500, 1.4e-4 / 5.2e-4 at 5000; Gamma 2.1e-3 / 7.6e-3
and 1.8e-4 / 7.1e-4.

Reading it: under decay the gap reaches a steady state once the stream is a
few half-lives long (half-life 500: the 10k and 100k columns match), because
the effective sample stops growing; at half-life 500 that steady state (0.12
SE Poisson, 0.26 SE Gamma at k = 20) is *below* the no-decay gap at a
comparable count (1k rows: 0.38, 0.42), because the rows scored by early,
poor fits decay away instead of staying in `G` for ever. Half-life 5000 at
100k is below the no-decay gap at 100k, for the same reason. Gamma's gap at
half-life 500 is nearly seed-free (0.263 mean, 0.279 max): a systematic
frozen-weight bias, about a quarter of an SE at k = 20.

**Which statsmodels weights are right.** The point estimates of
`freq_weights` and `var_weights` agree to 1e-15. The standard errors do not:
- For Poisson (scale fixed at 1) the two `bse` are the same.
- For Gamma, `var_weights` divides the Pearson statistic by the row count
  `n - p`, `freq_weights` by `sum w - p`. Under decay `sum w` is about
  `1 / (1 - lambda)` (721 rows at half-life 500), so `var_weights` treats rows
  weighted 1e-60 as observations and its `bse` is too small by
  `sqrt(sum w / n)`: ratio 0.085 at 100k rows and half-life 500 (measured
  0.084 to 0.085). **`freq_weights` is the right one** for the point estimate
  and the dispersion, and it is the one this record's SE units use.
- Neither is the sampling SE of a decay-weighted estimate. The sandwich
  (`A^{-1} (sum w_i^2 s_i^2 x_i x_i') A^{-1}`, `A = X' diag(w W) X`) is, and
  under stationarity its ratio to the `freq_weights` `bse` is
  `sqrt(sum w^2 / sum w) = sqrt(1 / (1 + lambda))`, about `1/sqrt(2)` once the
  stream is long: measured
  0.68 to 0.71 at 100k rows. In sandwich units every decay gap above is about
  1.4 times larger (Gamma k = 20, half-life 500: 0.39 SE).

### 7.4 Rolling window, W = 2000 rows

The window keeps its rows: `(x, y, w)` and the `(W_i, z_i)` each was scored
at (or `eta_i`, from which they follow), and the row that leaves has exactly
its own contribution `w W x x'`, `w W z x` subtracted. **The removal is exact
(the state is always the sum over the rows in the window), so the window's
state needs the window's rows**: `(k + 4) W` floats plus `G`, `h`, `beta`
(k = 20, W = 2000: 384 KB, of which `x` is 320 KB; bounded by the window,
not the stream; part 1's state without a window is under 2 KB). The *fit* is approximate in the same way as without a window: each
row's weight is frozen at the fit that scored it, not at the window's
optimum. Without the rows there is no exact window (decay is the
approximation). Subtraction leaves a rounding drift; a periodic rebuild from
the stored rows (which the state has anyway) removes it and was not needed
here. The refit's 200 rows are re-scored at the refit's fit in the window
too, so they leave with what they added.

Gap to the ridge-matched batch fit on the window's 2000 rows (alpha = R/2000),
SE from that window's `bse`; mean / max over 10 seeds. At 1k rows the window
is not yet full (the 1k column is part 1's no-window case).

| family | k | 1k | 10k | 100k |
|---|---|---|---|---|
| Poisson | 5 | 0.098 / 0.203 | 0.043 / 0.080 | 0.027 / 0.071 |
| Poisson | 20 | 0.376 / 0.522 | 0.137 / 0.224 | 0.150 / 0.266 |
| Gamma | 5 | 0.113 / 0.173 | 0.079 / 0.142 | 0.100 / 0.174 |
| Gamma | 20 | 0.417 / 0.491 | 0.323 / 0.409 | 0.318 / 0.390 |

Like decay at half-life 500 (an effective 721 rows) it reaches a steady
state; per row of effective sample the window is slightly worse than decay
(Gamma k = 20: 0.32 SE at 2000 rows vs 0.26 SE at 721), because a row stays at
full weight in `G` for 2000 rows with its frozen `W` and `z` instead of
fading. A periodic exact refit of the stored window (it is all there) would
remove the frozen-weight gap entirely at `O(W k^2)` per refit; not measured.

### 7.5 Zero-weight rows and zero counts

Checked on 3000 rows, k = 5, 20% of rows at weight 0 including the first two
(their `y` set to 7 to show it is not read):
- No decay, plain (A) and with refit (C): **bit-identical** `beta` and `G` to
  the stream with those rows dropped, Poisson and Gamma; no NaN. The refit
  buffer skips zero-weight rows, so it fills after 200 weighted rows.
- Decay (half-life 500): a zero-weight row advances the clock (`G`, `h` decay)
  and learns nothing, so the target is the batch fit with those rows at
  weight 0 and the ages still counted, *not* the fit with the rows dropped.
  Gap to that target: 0.14 SE Poisson, 0.09 SE Gamma; no NaN. `beta` is
  re-solved after the decay (the ridge does not decay), and `G + R I` is
  positive definite, so nothing divides by zero.
- Window (W = 1000, the zero-weight rows occupy slots): gap to the batch fit
  on the window's weighted rows 0.28 SE Poisson, 0.21 SE Gamma; no NaN.
- **Zero counts**: the Poisson stream above has 87% zeros (mean about
  `exp(-2)`) and a zero count in its first row; `z = eta - 1` at `y = 0`, no
  NaN anywhere. A Gamma `y = 0` (outside Gamma's support, 1 row in 50): the
  online fit stays finite (`z = eta - 1`); statsmodels fits it with no error and
  no warning; glum raises `ValueError: Some value(s) of y are out of the valid
  range`. An online Gamma should reject `y <= 0` as glum does.

### 7.6 A step cap instead of the refit (k = 20, no refit, R = 1)

The cap: the change in this row's own `eta`, `x_t'(beta_t - beta_{t-1})`, is
scaled down to at most `c` in absolute value, and `h` is reset to `(G + R I)
beta` so the next row steps from the capped fit (the increment form). No
buffer, so no extra state. Failure as in part 1: gap above 10 SE to the
ridge-matched batch fit at 100k rows. 10 seeds each; gap of the converged
seeds mean / max, in SE at 100k.

| family | variant | seeds failed | gap, SE mean / max | rows capped (of 100k) |
|---|---|---|---|---|
| Poisson | A, no cap (control) | 6 / 10 (0, 1, 2, 3, 4, 6) | 3.93 / 8.08 | 0 |
| Poisson | cap 1 | **0 / 10** | 0.220 / 0.298 | 10 to 16 |
| Poisson | cap 2 | **0 / 10** | 0.498 / 0.878 | 3 to 7 |
| Poisson | C (part 1, refit 200) | 0 / 10 | 0.137 / 0.164 | |
| Gamma | A, no cap (control) | 5 / 10 (0, 4, 6, 7, 8; overflow, ~99,970 rows clipped) | 0.806 / 2.253 | 0 |
| Gamma | cap 1 | **0 / 10** | 0.214 / 0.242 | 6 to 11 |
| Gamma | cap 2 | **0 / 10** | 0.264 / 0.317 | 2 to 7 |
| Gamma | C (part 1, refit 200) | 0 / 10 | 0.144 / 0.153 | |

The controls reproduce part 1's counts (6 and 5 of 10). With decay (half-life
5000, no refit): **decay alone does not rescue the start-up**: Gamma still
fails on the same 5 seeds (overflow), Poisson on 1 (seed 6, which clips 37
rows and ends 1,188 SE away), although decay does wash out Poisson's frozen
weights on the other 5 seeds that failed without it. **Cap 1 with decay
5000**: 0 / 10 fail in both families, and the gap is the same as C with decay
(Poisson 0.037 / 0.054, Gamma 0.083 / 0.084 SE at 100k, the 7.3 table's
values to the third digit).

So **on these seeds the cap replaces the refit**: it acts on only the first
2 to 16 rows (the underdetermined start, where one row can move `eta` by
tens), keeps no buffer, and leaves no failure in either family at cap 1 or 2.
Without decay it costs accuracy for good: the frozen weights of the capped
early rows stay in `G`, and the gap at 100k is about 1.5 times C's at cap 1
(0.22 against 0.14 SE) and up to 5 times at cap 2 (0.88 against 0.16, max).
With decay that cost disappears, since those rows decay away. Cap 1 beats
cap 2 in every cell here. Ten seeds cannot bound a failure rate below about
1 in 4 (with 0 of 10 failing, the 95% upper bound is 26%); part 3's 50 seeds
should repeat 7.6 before the cap is called a replacement.

### 7.7 Findings for part 2

1. **The log-link binomial is not well posed** for this recursion (its own
   start `beta = 0` is `mu = 1`, where the weight is infinite; with a truth
   near 1 it predicts above 1 on 2,518 of 20,000 rows) and statsmodels and
   glum fail on it too. Logit is a clean extra (0.076 SE at 20k rows).
2. **Separation**: no batch library warns on the Poisson case (statsmodels says
   converged, coefficient -31 with `bse` 13,000); the online fit stays finite,
   bounded by its ridge, and trails the ridge-matched fit (by 2.0 abs at 100k;
   negligible in fitted means). Logit: statsmodels returns 1e15 or fails to
   converge, glum warns and then fails with a singular matrix at 100k.
3. **Decay**: the gap to the `lambda^(T-t)`-weighted batch fit reaches a
   steady state of 0.12 (Poisson) and 0.26 (Gamma) SE at k = 20, half-life
   500, and 0.04 and 0.08 SE at half-life 5000 (100k rows); at k = 5 under
   0.07 SE. Use `freq_weights`, not `var_weights` (whose Gamma `bse` is 12
   times too small at 100k rows, half-life 500); the true sampling SE is the
   sandwich, about 0.7 of `freq_weights`' `bse`.
4. **Window (W = 2000)**: steady-state gap 0.15 / 0.27 SE (Poisson k = 20, mean
   / max) and 0.32 / 0.39 (Gamma k = 20). Removal is exact and needs the
   window's rows in the state (384 KB at k = 20); the fit is approximate
   through the frozen weights.
5. **Zero weights**: bit-identical to dropping the rows without decay (first
   row included, with and without the refit); under decay or a window a
   zero-weight row advances the clock or takes a slot, and matches the batch
   fit with that row at weight 0. No NaN anywhere, 87% zero counts included.
   A Gamma `y = 0` is silently accepted by the recursion and by statsmodels.
6. **The step cap (1 or 2) replaces the 200-row refit on all 10 seeds** of
   both families at k = 20 (0 failures with the cap, 6 and 5 without), with no
   buffer; it costs accuracy without decay (0.22 against 0.14 SE at cap 1) and
   nothing with decay.

Harness note: one long-lived Python process that ran the failing (overflowing)
seeds one after another grew to 12 GB; one process per job stays at 0.7 to
1.4 GB. The cause was not chased (the online state is `O(k^2)`); runs of the
script should use `--procs 2` at most or a process per job.
