# Study 500: online GLMs by recursive IRLS

**Status (2026-11-21): complete (parts 1 to 3).** Part 1 (sections 1 to 6):
Poisson and Gamma, log link, no decay, no window, 10 seeds. Part 2 (section
7): the log-link binomial and separation, decay, a rolling window, zero-weight
rows and zero counts, and a step cap in place of the refit. Part 3 (section
8): Tweedie (p = 1.5), a periodic refit of a stored window, Newton sweeps per
row, the row-by-row oracle through statsmodels' family objects, 50 seeds, and
section 9, which collects the recursion of every variant, the state sizes, the
numerical risks, the oracles and the **proposed test tolerances**. Not done:
a sweep of `refit_rows` and of the ridge (part 1's open question).

One script, `scripts/study_500.py`, runs every variant and check (parts 1 and
2's scripts are merged into it; `--quick` runs every section on short streams
in about 30 seconds). It generates its own data from seeded numpy generators
and reads nothing from disk. The proposal is task 500 in the models scan
(branch `vm-tools/oracle-gap-scan`, `vm-tools/research/494-MODELS-SCAN.md`,
section 6).

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

(Answered since: the cap, 7.6 and 8.6; decay, 7.3; the row-by-row oracle,
8.5; 50 seeds, section 8; the batch reference's precision, 7 and 8. Still
open: how small `refit_rows` can be, and a sweep of the ridge.)

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
except in 7.6, which asks whether a step cap can replace the refit. Run (part
2 ran these as `scripts/study_500_part2.py`, now merged into
`scripts/study_500.py` with the same functions and seeds):
`.venv/bin/python scripts/study_500.py checks binomial separation --out a.json`
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

The script now runs one job per worker process (`max_tasks_per_child=1`): in
part 3 a pool worker that had run about 20 jobs at 100k rows had grown to
5 GB and was killed by the memory limit. Each result is also appended to
`<out>.partial.jsonl` as it arrives.

## 8. Part 3: Tweedie, a periodic window refit, Newton sweeps, 50 seeds

**Baseline: variant C** (`R = 1`, refit of the first 200 rows) throughout.
Run (each under `nohup`):
`.venv/bin/python scripts/study_500.py variants --seeds 50 --procs 3 --variants C periodic replace sweeps --out v50.json`
(43 minutes on 3 cores), `... variants --seeds 10 --families tweedie
--variants A B`,
`... dispersion` (5 seeds), `... cap --seeds 50 --procs 1`, `... checks`.

**Seeds: 50** (0 to 49) for every family, k (5, 20) and variant in 8.1 to
8.4, and for the cap (8.6); 10 for Tweedie's A and B; 5 for the dispersion
check. Part 1's 10-seed values for C are reproduced (Poisson k = 20 at 100k,
seeds 0 to 9: 0.137 / 0.164 SE, as in 4.2).

**The references at each checkpoint** (all on the same rows, all weights 1):

- `ref`: penalised IRLS with step halving, sum form, ridge `R`, tolerance
  1e-13 (the refit's solver);
- statsmodels `GLM(freq_weights=1).fit(method="IRLS", tol=1e-12)`,
  unpenalised: its `params` ("statsmodels, unpenalised") and its `bse`, the SE
  unit of every gap below (Pearson scale for Gamma and Tweedie);
- statsmodels' ridge path, `fit_regularized(alpha=R/n, L1_wt=0)` (BFGS),
  with the model's `loglike` replaced by `-deviance/2` from statsmodels' own
  family object (Gamma's `loglike` profiles the scale, Tweedie's is a series
  evaluation of the density) and `score` at scale 1 ("statsmodels"; started
  at the unpenalised fit);
- glum, `GeneralizedLinearRegressor(family, link="log", alpha=R/n,
  l1_ratio=0, fit_intercept=False, gradient_tol=1e-10)`, Tweedie as
  `TweedieDistribution(1.5)`.

Floors over the 50 seeds: glum against `ref` at most 1.1e-05 SE; the
statsmodels ridge path against `ref` at most 3.7e-03 SE (BFGS stops at a
gradient of 1e-5, as in part 1); statsmodels unpenalised against `ref` at
most 0.038 SE (the ridge itself, largest at 1k rows). So "statsmodels"
and "glum" below are interchangeable to the third decimal of an SE.

Gap = `max_j |beta_online_j - beta_batch_j| / bse_j`; "mean / max" over seeds.

### 8.1 Tweedie, p = 1.5, log link

`V(mu) = mu^1.5`, so `W = mu^(2-p) = mu^0.5`, `z = eta + (y - mu)/mu` (as for
every log link), the score `W (z - eta) = (y - mu) mu^(-0.5)`, and the half
deviance up to a term free of beta `-y mu^(1-p)/(1-p) + mu^(2-p)/(2-p) =
2 y mu^(-1/2) + 2 mu^(1/2)`. Data: the same design and seeds as Poisson and
Gamma (generator `[500, seed, k, 2]`), `y` compound Poisson-Gamma: `N ~
Poisson(mu^(2-p) / (phi (2-p)))` amounts, each `Gamma(shape (2-p)/(p-1) = 1,
scale phi (p-1) mu^(p-1))`, `phi = 2`, so `E y = mu`, `Var y = phi mu^1.5`,
and 27 to 28% of `y` are exactly 0. `y = 0` is legal (`z = eta - 1`).

Variant C, 50 seeds:

| k | n | abs, max | SE mean / max | statsmodels unpenalised, SE max | glum, SE mean / max |
|---|---|---|---|---|---|
| 5 | 1k | 1.10e-02 | 0.154 / 0.270 | 0.263 | 0.154 / 0.270 |
| 5 | 10k | 2.10e-03 | 0.113 / 0.166 | 0.163 | 0.113 / 0.166 |
| 5 | 100k | 3.89e-04 | 0.057 / 0.097 | 0.097 | 0.057 / 0.098 |
| 20 | 1k | 3.74e-02 | 0.644 / 0.950 | 0.942 | 0.644 / 0.950 |
| 20 | 10k | 7.67e-03 | 0.486 / 0.609 | 0.606 | 0.486 / 0.609 |
| 20 | 100k | 1.25e-03 | 0.241 / 0.311 | 0.311 | 0.242 / 0.312 |

**A and B (10 seeds)**: with `R = 1` (A), 2 of 10 seeds at k = 5 end above 10
SE at 100k rows (stuck, without a clip; the other 8: 1.5 / 6.4 SE mean / max)
and 9 of 10 at k = 20 (in 7 the fit runs away within the first rows, clipping
77 to 108 rows, and stays about 23 abs away; 2 are stuck without a clip); with
`R = 10` (B), k = 5 converges on all 10 (0.16 / 0.80 SE) and k = 20 sticks on
2 of 10 (the other 8: 0.82 / 2.15 SE). So Tweedie has both failures, the Gamma
runaway and the Poisson stick. Tweedie's step is `(y - mu) mu^(-1/2) x`, which
for a row scored at a very negative `eta` is of order `y e^(|eta|/2)`: the same
runaway as Gamma's `y e^(|eta|)`, half as steep. Variant C's refit also
rescues every Tweedie seed: in 43 of 50 seeds at k = 20 some rows clip
(`|eta| = 30`) before the refit replaces the state, and all 50 converge.

**Dispersion.** Tweedie's gap is about 1.7 times Poisson's at k = 20. It is
the dispersion: with `phi` = 0.5 / 2 / 8 (1%, 28%, 72% zeros), variant C's gap
at 100k rows, k = 20, 5 seeds, is 0.119 / 0.241 / 0.527 SE mean (0.156 /
0.264 / 0.606 max): doubling per factor 4 in `phi`, i.e. the gap in SE grows
like `sqrt(phi)`. The frozen-weight error is a bias of order `phi / n`
(the early fits' errors are of order `sqrt(phi / i)` and enter squared),
while the SE is of order `sqrt(phi / n)`. Across families the dispersion
is not the only factor: Gamma (`phi = 0.5`, `W = 1`) has 0.143 SE and Poisson (1) 0.137 at
k = 20, against 0.119 for Tweedie at `phi = 0.5`. **A test tolerance in SE
therefore holds for a stated dispersion**, and scales by `sqrt(phi / phi_0)`.

### 8.2 Periodic refit of a stored window (W = 2000, every M = 1000 rows)

Two versions, both on top of C:

- **replace** (the literal one): every 1000 rows, fit the last 2000 rows alone
  (penalised IRLS from the online estimate) and **replace** beta by it (`G`
  kept, `h = (G + R I) beta`). The estimate is then a 2000-row fit, whose SE is
  `sqrt(n / 2000)` times the full fit's: 7.1 at 100k rows. Measured at 100k
  rows, SE mean / max: Poisson 11.4 / 20.6 (k = 5) and 15.7 / 24.0 (k = 20),
  Gamma 11.5 / 18.4 and 15.4 / 22.2, Tweedie 10.9 / 21.0 and 15.1 / 23.4 (the
  max over coefficients of a 2000-row fit's error in units of the 100k-row
  SE, as expected). **It makes the gap far
  worse**; it estimates a different thing (the last 2000 rows), which is what a
  window is for (section 7.4), not a remedy for the full-stream fit.
- **periodic** (re-score the window): the rows that have left the window are
  kept only as their frozen quadratic (`G_out`, `h_out`); every 1000 rows,
  Newton with step halving from the online estimate minimises
  `F(b) = 1/2 b'G_out b - h_out'b + sum_window w_i halfdev_i(b) + R/2 |b|^2`
  to convergence (mean 2.1 to 4.4 iterations a refit), and every window row is
  re-scored at that fit: `G = G_out + sum_window w W_i(b) x x'`, `h = (G + R I)
  b`. Between refits the plain recursion runs. A row's quadratic is thus
  frozen at the fit of the last refit while it was in the window, 1000 to 2000
  rows after it arrived, instead of at the fit before it.

### 8.3 Newton sweeps on the last 200 rows

**sweeps**: after C's start-up, every row: the row leaving the 200-row window is
frozen at the current fit (added to `G_out`, `h_out`), then 2 Newton steps
with step halving on the same `F(b)` over the last 200 rows. No `G` beyond
`G_out` and no per-row recursion step: the row's information enters through
its exact half deviance.

### 8.4 Results: gap at 1k, 10k and 100k rows, 50 seeds

SE gap to statsmodels' ridge path, mean / max over 50 seeds, glum's max in
brackets; 99.5k is half-way between two window refits (the refits fall on
multiples of 1000, so the 1k, 10k and 100k checkpoints see the state just
after one; at 1k the window holds every row and periodic and replace equal the
batch fit). Last column: the numpy online pass, seconds for 100k rows (a
Python loop; mean over seeds).

| family | k | variant | 1k | 10k | 99.5k | 100k | numpy, s |
|---|---|---|---|---|---|---|---|
| Poisson | 5 | C | 0.085 / 0.245 (0.245) | 0.065 / 0.144 (0.145) | 0.032 / 0.057 (0.058) | 0.032 / 0.057 (0.058) | 1.4 |
| Poisson | 5 | periodic | 0.000 / 0.000 (0.000) | 0.017 / 0.038 (0.038) | 0.017 / 0.035 (0.036) | 0.017 / 0.034 (0.035) | 1.9 |
| Poisson | 5 | sweeps | 0.040 / 0.150 (0.150) | 0.047 / 0.104 (0.104) | 0.027 / 0.050 (0.050) | 0.027 / 0.049 (0.050) | 13.8 |
| Poisson | 20 | C | 0.389 / 0.945 (0.945) | 0.278 / 0.466 (0.467) | 0.137 / 0.198 (0.199) | 0.137 / 0.198 (0.199) | 1.8 |
| Poisson | 20 | periodic | 0.000 / 0.000 (0.000) | 0.068 / 0.115 (0.115) | 0.066 / 0.105 (0.106) | 0.066 / 0.107 (0.107) | 2.4 |
| Poisson | 20 | sweeps | 0.180 / 0.414 (0.414) | 0.199 / 0.321 (0.322) | 0.111 / 0.174 (0.174) | 0.111 / 0.173 (0.174) | 16.1 |
| Gamma | 5 | C | 0.111 / 0.305 (0.305) | 0.071 / 0.116 (0.116) | 0.033 / 0.043 (0.043) | 0.033 / 0.043 (0.043) | 1.4 |
| Gamma | 5 | periodic | 0.000 / 0.000 (0.000) | 0.025 / 0.047 (0.047) | 0.020 / 0.028 (0.029) | 0.020 / 0.029 (0.030) | 1.9 |
| Gamma | 5 | sweeps | 0.066 / 0.159 (0.159) | 0.053 / 0.085 (0.085) | 0.031 / 0.042 (0.041) | 0.031 / 0.042 (0.041) | 14.4 |
| Gamma | 20 | C | 0.405 / 0.492 (0.491) | 0.296 / 0.330 (0.330) | 0.144 / 0.157 (0.159) | 0.143 / 0.157 (0.159) | 1.8 |
| Gamma | 20 | periodic | 0.000 / 0.000 (0.000) | 0.079 / 0.115 (0.114) | 0.074 / 0.091 (0.089) | 0.074 / 0.091 (0.089) | 2.5 |
| Gamma | 20 | sweeps | 0.206 / 0.325 (0.325) | 0.207 / 0.252 (0.252) | 0.118 / 0.134 (0.132) | 0.118 / 0.134 (0.132) | 17.0 |
| Tweedie | 5 | C | 0.154 / 0.270 (0.270) | 0.113 / 0.166 (0.166) | 0.057 / 0.097 (0.098) | 0.057 / 0.097 (0.098) | 1.4 |
| Tweedie | 5 | periodic | 0.000 / 0.000 (0.000) | 0.020 / 0.051 (0.051) | 0.013 / 0.037 (0.037) | 0.013 / 0.037 (0.037) | 1.8 |
| Tweedie | 5 | sweeps | 0.058 / 0.157 (0.157) | 0.036 / 0.108 (0.108) | 0.017 / 0.051 (0.051) | 0.017 / 0.051 (0.051) | 15.5 |
| Tweedie | 20 | C | 0.644 / 0.950 (0.950) | 0.486 / 0.609 (0.609) | 0.242 / 0.313 (0.314) | 0.241 / 0.311 (0.312) | 1.8 |
| Tweedie | 20 | periodic | 0.000 / 0.000 (0.000) | 0.063 / 0.109 (0.109) | 0.034 / 0.068 (0.068) | 0.034 / 0.067 (0.068) | 2.5 |
| Tweedie | 20 | sweeps | 0.159 / 0.268 (0.268) | 0.102 / 0.159 (0.159) | 0.043 / 0.077 (0.077) | 0.043 / 0.076 (0.076) | 18.1 |

Gap to the truth at 100k rows (max over coefficients in batch SE, mean over
seeds), online C, periodic, sweeps and the batch fit: Poisson 1.54, 1.54,
1.54, 1.54 (k = 5) and 2.17 for all four (k = 20); Gamma 1.58, 1.58, 1.59,
1.58 and 2.19, 2.18, 2.19, 2.18; Tweedie 1.69, 1.69, 1.69, 1.68 and 2.25,
2.24, 2.24, 2.23. Every variant but replace is statistically as good as the batch fit.

**Does the periodic refit close the gap?** It cuts it, does not close it.
At 100k rows, k = 20: Poisson 0.137 to 0.066 (mean; max 0.198 to 0.107),
Gamma 0.143 to 0.074 (max 0.157 to 0.091), Tweedie 0.241 to 0.034 (max 0.311
to 0.067): to a half for Poisson and Gamma, to a seventh for Tweedie. At k = 5:
Poisson 0.032 to 0.017, Gamma 0.033 to 0.020, Tweedie 0.057 to 0.013. It is
the same just after a refit and half-way between two
(99.5k), so what remains is not the rows since the last refit. What remains
is the rows frozen at the early refits: a row is frozen at a fit 1000 to 2000
rows later, and for the first few thousand rows that fit is itself poor. In SE
the remaining gap is flat from 10k to 100k (Poisson k = 20: 0.068, 0.066),
while C's halves (0.278, 0.137): at a few million rows C would catch up.

**Cost of the periodic refit.** State: the window's rows `(x, y, w)` and each
one's last scoring `(W_i, z_i)`, `(k + 4) W` floats, plus `G_out` and `h_out`
beside C's `G`, `h`, `beta`: 144 KB at k = 5 and 388 KB at k = 20 (C: 216 B and
2 KB), bounded by the window. Work: a refit costs about `(I + 1) W k(k+1)/2`
multiply-adds for the Grams (I = Newton iterations, on average 2.1 for
Poisson and 3.6 to 4.4 for Gamma and Tweedie; the +1 is the re-scoring) and
`3 I W k` for `eta`, scores and objective values: 1.5M (Poisson) to 2.8M
multiply-adds per refit at k = 20. Amortised over M = 1000 rows, `2(I + 1)
k(k+1)/2 + 6 I k`: 1,550 to 2,800 a row at k = 20 against C's 2,000 (one
Cholesky of `G + R I`, `k^3/6`, two triangular solves, the rank-one `G`
update), so 1.8 to 2.4 times C's per-row arithmetic in total; at k = 5, 160
to 250 a row against 70, 3.2 to 4.5 times. The numpy pass takes 1.3 to 1.4
times C's (a Python loop's overhead hides the arithmetic). The work comes as
a spike: the row that triggers a refit does 1000 rows' worth of it.

**Do the sweeps close it?** No, except for Tweedie. At 100k rows, k = 20:
Poisson 0.137 to 0.111, Gamma 0.143 to 0.118 (about a fifth less), Tweedie
0.241 to 0.043 (under a fifth of it); at 10k rows the Poisson k = 20 sweeps
gap (0.199) is still seven tenths of C's. A row's quadratic is frozen 200 rows after it
arrives, which for Poisson and Gamma moves the fit it is frozen at too little;
Tweedie, whose gap is dispersion-driven early noise, gains most. **Cost**:
2 Newton steps on 200 rows per row, `2 (200 k(k+1)/2 + 3 * 200 k + k^3/6)`
multiply-adds: about 111,000 a row at k = 20 (56 times C) and 12,000 at k = 5
(170 times C); the numpy pass about 10 times C's (14 to 18 s per 100k rows
against 1.4 to 1.8). State: `(k + 2) 200` floats for the window plus `G_out`,
`h_out`, `beta` (37 KB at k = 20, 11 KB at k = 5).

So for Poisson and Gamma neither remedy takes the gap much below half of C's,
the periodic refit is the cheaper and the better of the two, and both buy a
fraction of an SE that is already statistically invisible (8.4's gap to the
truth). For Tweedie at `phi = 2` the periodic refit removes six sevenths of a
gap that is the largest of the three.

### 8.5 The row-by-row oracle through statsmodels' family objects

The same recursion (with C's refit) with `W` and `z` taken from statsmodels'
family objects instead of the closed forms: `W = family.weights(mu) = 1 /
(g'(mu)^2 V(mu))`, `z = eta + (y - mu) g'(mu)` with `g' = link.deriv`. Over
2000 rows, k = 5, max `|dbeta|` along the path: Poisson 2.9e-15, Gamma
8.3e-16, Tweedie 7.2e-14 (`checks`). The sums form against the increment
form (500 rows, no refit): 2.3e-15, 1.0e-15 and 4.0e-15 (Tweedie on seed 997,
because seed 998 overflows without the refit). The zero-weight checks of 7.5
pass for Tweedie too (bit-identical to the stream with the rows dropped, plain
and with the refit; decay 500 gap 0.10 SE, window 1000 gap 0.12 SE; 28% zeros
in `y`).

### 8.6 The step cap at 50 seeds (repeat of 7.6, k = 20, no refit)

Failure: gap above 10 SE to the ridge-matched (decay-weighted) batch fit at
100k rows; gap of the other seeds, mean / max, in SE; rows capped per seed.

| family | variant | seeds failed (of 50) | gap, SE mean / max | rows capped |
|---|---|---|---|---|
| Poisson | A, no cap (control) | 23 | 2.09 / 8.08 | 0 |
| Poisson | cap 1 | **0** | 0.239 / 0.578 | 6 to 18 |
| Poisson | cap 2 | **0** | 0.612 / 4.19 | 1 to 11 |
| Poisson | decay 5000, no cap | 5 | 0.626 / 8.52 | 0 |
| Poisson | cap 1, decay 5000 | **0** | 0.039 / 0.065 | 6 to 18 |
| Poisson | C (8.4) | 0 | 0.137 / 0.198 | |
| Gamma | A, no cap (control) | 34 | 1.22 / 9.27 | 0 |
| Gamma | cap 1 | **0** | 0.208 / 0.284 | 5 to 15 |
| Gamma | cap 2 | **0** | 0.285 / 0.388 | 1 to 8 |
| Gamma | decay 5000, no cap | 32 | 0.083 / 0.084 | 0 |
| Gamma | cap 1, decay 5000 | **0** | 0.083 / 0.084 | 5 to 15 |
| Gamma | C (8.4) | 0 | 0.143 / 0.157 | |

Part 2's 10 seeds understated the plain recursion's failure rate (6 and 5 of
10; now 23 and 34 of 50, i.e. 46% and 68%). **Cap 1 fails on none of the 50
seeds** in either family, with or without decay: the 95% upper bound on its
failure rate is now 6% (3 / 50), against 26% after part 2. Its cost without
decay is larger in the tail than 10 seeds showed: Poisson max 0.578 SE
against C's 0.198 (2.9 times), Gamma 0.284 against 0.157 (1.8 times). Cap 2
leaves one Poisson seed at 4.2 SE: cap 1 is the cap. With decay (half-life
5000) cap 1 costs nothing (the capped rows decay away) and equals C's decay
figures of 7.3.

### 8.7 Findings for part 3

1. **Tweedie** (`p = 1.5`, `phi = 2`) behaves as Poisson and Gamma do: the
   plain recursion fails (2 of 10 seeds at k = 5, 9 of 10 at k = 20, by both the
   Gamma runaway and the Poisson stick); variant C converges on all 50 seeds;
   its gap at 100k rows is at most 0.311 SE to statsmodels and 0.312 SE to
   glum (k = 20; 0.097 / 0.098 at k = 5), the largest of the three families
   because the gap in SE grows like `sqrt(phi)`.
2. **The largest gap at 100k rows** (C, 50 seeds, k = 20), statsmodels /
   glum: Poisson 0.198 / 0.199, Gamma 0.157 / 0.159, Tweedie 0.311 / 0.312
   SE. At 10k: 0.466 / 0.467, 0.330 / 0.330, 0.609 / 0.609. Every variant's
   error against the truth equals the batch fit's to 0.01 SE on average.
3. **A periodic refit of the last 2000 rows that replaces beta** is 15 SE off
   at 100k rows (it is a 2000-row fit): not a remedy.
4. **The periodic refit that re-scores the window** (2000 rows, every 1000,
   from the online estimate) halves the gap for Poisson and Gamma and cuts
   Tweedie's by seven (k = 20 at 100k, max: 0.107, 0.091, 0.068 SE); it does
   not close it, and its gap is flat in SE from 10k rows on. Cost: 388 KB of
   state at k = 20 (144 KB at k = 5) against C's 2 KB, and 1.8 to 2.4 times C's
   per-row arithmetic at k = 20 (3 to 4.5 times at k = 5), paid as a spike every
   1000 rows.
5. **Two Newton sweeps per row on the last 200 rows** cut the gap by a fifth
   for Poisson and Gamma and by four fifths for Tweedie, at 56 times C's
   per-row arithmetic at k = 20 (170 times at k = 5; about 10 times in numpy)
   and 37 KB of state: dominated by the periodic refit everywhere.
6. **The cap** (part 2's alternative to the refit) holds on 50 seeds: cap 1
   fails on none in either family (the no-cap control fails on 46% and 68%),
   at up to 2.9 times C's gap without decay and none with it.
7. **The recursion is exact against statsmodels' own family objects**
   (max `|dbeta|` 7.2e-14 over 2000 rows, Tweedie) and against its increment
   form (4e-15); zero-weight rows stay bit-identical for Tweedie too.
8. **Proposed tolerances** (9.5): Poisson 0.75 / 0.30 SE, Gamma 0.50 / 0.25,
   Tweedie 0.95 / 0.50 at 10k / 100k rows, for k up to 20 at the stated
   dispersions.

## 9. For an implementation and its tests

### 9.1 The recursion, per variant (log link)

Per family: `W(mu) = mu` (Poisson), `1` (Gamma), `mu^(2-p)` (Tweedie);
`z = eta + (y - mu)/mu` for all three; score `s = W (z - eta)`. Row `t`:
`eta_t = x_t' beta_{t-1}` (clipped to `|eta| <= 30`), `mu_t = exp(eta_t)`,
`pred_t = mu_t` emitted first.

- **A** (`R = 1`) / **B** (`R = 10`): `G_t = lambda G_{t-1} + w_t W_t x_t x_t'`,
  `h_t = lambda h_{t-1} + w_t W_t z_t x_t`, `beta_t = (G_t + R I)^{-1} h_t`
  (equivalently `beta_t = beta_{t-1} + (G_t + R I)^{-1} w_t s_t x_t`).
  `lambda = 1` without decay; a zero-weight row only decays.
- **C** (baseline): A until 200 weighted rows have arrived; then `b* =
  argmin sum_{i<=200} w_i halfdev_i(b) + R/2 |b|^2` (penalised IRLS with step
  halving, from `b_0 = (log ybar, 0, ...)`), `G = sum_{i<=200} w_i W_i(b*) x_i x_i'`
  (times `lambda^age` under decay), `h = (G + R I) b*`, `beta = b*`; the buffer
  is dropped and A continues.
- **periodic** (`W = 2000`, `M = 1000`): C, plus the window; when a row leaves
  the window, `G_out += w W_i x x'`, `h_out += w W_i z_i x` at its stored
  scoring; every `M` rows `b* = argmin F(b)` (Newton, step halving, from
  `beta`), each window row re-scored at `b*`, `G = G_out + sum_window w W_i(b*)
  x x'`, `h = (G + R I) b*`, `beta = b*`.
- **replace**: C, plus the window; every `M` rows `beta = argmin
  sum_window w halfdev + R/2 |b|^2`, `h = (G + R I) beta`.
- **sweeps** (window 200): C until its refit; then per row: the leaving row
  frozen at `beta` (`G_out`, `h_out` += its quadratic scored at `beta`), the
  new row stored, then `beta <- beta + t d` twice, `d = (G_out + sum_window
  w W_i(beta) x x' + R I)^{-1} grad`, `grad = h_out - G_out beta + sum_window w
  s_i(beta) x - R beta`, `t` halved until `F` does not rise.
- **cap c** (7.6): A, and if `|x_t'(beta_t - beta_{t-1})| > c` the step is
  scaled to `c` and `h = (G + R I) beta_t`.

### 9.2 State, in bytes (8-byte floats, symmetric matrices packed)

| variant | floats | k = 5 | k = 20 |
|---|---|---|---|
| A, B, C after the start-up | `k(k+1)/2 + 2k + 2` (`G`, `h`, `beta`, `weight_sum`, `R`) | 216 B | 2.0 KB |
| C during the start-up | plus the buffer `200 (k + 2)` | +11.2 KB | +35.2 KB |
| decay | as C | | |
| exact window `W` (7.4) | C plus `(k + 4) W` (`x`, `y`, `w`, `W_i`, `z_i`) | +144 KB at W = 2000 | +384 KB |
| periodic (`W` = 2000) | C plus `G_out`, `h_out` plus `(k + 4) W` | 144 KB | 388 KB |
| replace (`W` = 2000) | C plus `(k + 2) W` | 112 KB | 354 KB |
| sweeps (window 200) | `G_out`, `h_out`, `beta`, plus `(k + 2) 200` | 11.4 KB | 37 KB |

Per target: `W` depends on the target, so `T` targets keep `T` copies. All of
it is bounded by `k` and the window, none by the stream's length.

### 9.3 Numerical risks

- **Overflow of the Gamma (and Tweedie) recursion at start-up.** Before `k`
  weighted rows the fit is underdetermined and one row can move `eta` by tens;
  a later row scored at a very negative `eta` has a score of order
  `y e^(|eta|)` (Gamma) or `y e^(|eta|/2)` (Tweedie), the next step is
  enormous, `eta` hits the clip and the fit never returns (34 of 50 Gamma
    seeds at k = 20 with `R = 1` and no remedy, 8.6; Tweedie 9 of 10, 8.1).
  Remedy: C's refit (no failure in 50
  seeds of any family) or the cap (7.6, 8.6). Keep the `|eta|` clip as a
  guard, and count it.
- **The stuck Poisson case.** A row scored at a too-large `mu` enters `G` with
  `W = mu` frozen at that fit; without decay that direction's curvature is
  never corrected and the fit stops moving along it (23 of 50 seeds at
    k = 20, 8.6; up to 894 SE in part 1). Same remedies; decay alone recovers
  most but not all
  (7.6).
- **Separation.** Poisson with a zero-count indicator: the MLE is at minus
  infinity; the online fit stays finite, bounded by its ridge, and trails the
  ridge-matched fit (7.2). statsmodels reports convergence with no warning and
  glum warns nothing; a test must not use a separated design against an
  unpenalised oracle.
- **Domain.** `y < 0` for any of the three, `y = 0` for Gamma: reject
  (statsmodels accepts a Gamma `y = 0` silently, glum raises). The log-link
  binomial is out of scope (7.1).
- **Zero weights.** A zero-weight row (first included) must not divide: `R >
  0` keeps `G + R I` positive definite (7.5).
- **The frozen weights are a bias, not noise**: of order `phi / n` in beta,
  so `sqrt(phi / n)` in SE; it does not wash out without decay, and grows with
  the dispersion (8.1).
- **Window subtraction drift** (7.4): an exact window subtracts each row's own
  contribution; a periodic rebuild from the stored rows (which the periodic
  variant does at every refit) bounds the rounding.

### 9.4 The oracles

| oracle | what it checks | agreement measured |
|---|---|---|
| the same recursion with statsmodels' family objects (`weights`, `link.deriv`) | the per-row `W`, `z` and the recursion, exactly | largest change in beta 7.2e-14 over 2000 rows (8.5) |
| the increment form `beta += (G + R I)^{-1} w s x` | the sums form | 4.0e-15 (8.5) |
| the stream with zero-weight rows dropped | zero weights | bit-identical (7.5, 8.5) |
| penalised IRLS, tolerance 1e-13 (`ref`) | the batch target `argmin sum w halfdev + R/2 b'b` | glum within 1.1e-05 SE |
| glum, `alpha = R / sum w` | the batch target, third party | the reference for a ledger: within 1.1e-5 SE of `ref` |
| statsmodels `GLM(freq_weights)` IRLS | `bse` (the SE unit; Pearson scale), the unpenalised fit | `freq_weights`, not `var_weights` (7.3) |
| statsmodels `fit_regularized(L1_wt=0)` with `-deviance/2` | the batch target, second third party | within 3.7e-03 SE of `ref` (BFGS stops at gradient 1e-5) |

A Rust test cannot call these; it can carry the batch target as a penalised
IRLS through `crate::oracle`'s LU (as `ref` does here) and check it once
against glum in Python (`tests/test_second_opinion.py`).

### 9.5 Proposed test tolerances

What a test checks: variant C (`R = 1`, 200-row refit, no decay), the design
of section 3 (intercept plus `k - 1` standard normal features, `eta` sd 0.5
around 0.5), `max_j |beta_online_j - beta_batch_j| / bse_j <= tol` at a
stated row count, where `beta_batch` is the ridge-matched batch fit (`alpha =
R / n`) and `bse` statsmodels' unpenalised `GLM(freq_weights)` SE (Pearson
scale for Gamma and Tweedie). The data's dispersion is fixed: Poisson 1,
Gamma 0.5 (shape 2), Tweedie `p = 1.5`, `phi = 2`.

The measured basis (C, 50 seeds, k = 5 and 20; the max is at k = 20 in every
cell), in SE:

| family | n | statsmodels, max | unpenalised, max | glum, max | glum, mean / sd (k = 20) | mean + 4 sd |
|---|---|---|---|---|---|---|
| Poisson | 10k | 0.466 | 0.464 | 0.467 | 0.279 / 0.054 | 0.494 |
| Poisson | 100k | 0.198 | 0.198 | 0.199 | 0.138 / 0.022 | 0.225 |
| Gamma | 10k | 0.330 | 0.323 | 0.330 | 0.296 / 0.017 | 0.364 |
| Gamma | 100k | 0.157 | 0.157 | 0.159 | 0.146 / 0.006 | 0.168 |
| Tweedie | 10k | 0.609 | 0.606 | 0.609 | 0.486 / 0.053 | 0.699 |
| Tweedie | 100k | 0.311 | 0.311 | 0.312 | 0.242 / 0.024 | 0.340 |

The proposed tolerance is **1.5 times the max over 50 seeds, rounded up to
0.05 SE**, for any k up to 20:

| family (dispersion) | tolerance at 10k rows | at 100k rows |
|---|---|---|
| Poisson (1) | **0.75 SE** (max 0.467) | **0.30 SE** (max 0.199) |
| Gamma (0.5) | **0.50 SE** (max 0.330) | **0.25 SE** (max 0.159) |
| Tweedie (2) | **0.95 SE** (max 0.609) | **0.50 SE** (max 0.312) |

The margin, and why 1.5:
- **The seed tail.** The gap is a bias, not a noise: its spread over seeds is
  narrow (standard deviation 0.006 to 0.054 SE at k = 20), so mean + 4 sd
  (the last column; about 1 seed in 30,000 would pass it if the spread were
  normal) is only 1.05 to 1.15 times the 50-seed max. The tail is heavier than
  normal in places (Poisson at 10k: the worst seed is 3.5 sd above the mean,
  0.467 against the next 0.373), so the margin is set on the max, not on
  the sd: 1.5 times the max is 7.8 sd above the mean in the widest cell
  (Poisson, 10k) and covers the unseen seeds with room.
- **The batch reference.** glum and `ref` agree within 1.1e-5 SE and
  statsmodels' ridge path within 3.7e-3 SE, a negligible share of any tolerance; the
  unpenalised fit differs from the ridge-matched one by at most 0.04 SE at 1k
  rows, 0.012 at 10k: a test should compare with the ridge-matched fit.
- **The implementation.** A different solver order (Cholesky against LU,
  another summation order) moves `beta` by about 1e-14 relative: nothing.
- **What 1.5 does not cover**: a larger dispersion (the gap in SE grows like
  `sqrt(phi)`: at Tweedie `phi = 8` the max was 0.61 SE at 100k, over the
  Tweedie tolerance), a larger `k` (k = 20 is 3 to 4 times k = 5), a stronger
  signal (`eta` with sd above 0.5 means wider swings of `W`), or no refit. A test
  outside these must be re-measured, not extrapolated. A k = 5 test can carry
  its own, by the same rule (1.5 times the k = 5 max, rounded up to 0.05):
  Poisson 0.25 / 0.10, Gamma 0.20 / 0.10, Tweedie 0.25 / 0.15 SE at 10k /
  100k (maxima 0.145 / 0.058, 0.116 / 0.043, 0.166 / 0.098).
- **The periodic refit**, if it is built, gets its own by the same rule:
  Poisson 0.20 / 0.20, Gamma 0.20 / 0.15, Tweedie 0.20 / 0.15 SE at 10k /
  100k (maxima 0.115 / 0.107, 0.115 / 0.091, 0.109 / 0.068).

A test at 1k rows is not proposed: at 1k rows, k = 20, the max over seeds is
0.49 (Gamma) to 0.95 SE (Poisson, Tweedie) and the worst Poisson seed is 2.4
times the mean; the spread is too wide for a tolerance that would catch a
real defect. 10k is the first count at which the bias dominates the spread.
