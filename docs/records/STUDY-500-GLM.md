# Study 500: online GLMs by recursive IRLS

**Status (2026-11-21): part 1 of 3 done; parts 2 and 3 missing.** Poisson and Gamma, log link, no
decay, no window, 10 seeds (the target is 50; part 3 extends it if time
allows). Missing, for parts 2 and 3: decay (`lambda < 1`, the converged
target `GLM(freq_weights=lambda^age)`), the row-by-row exact oracle through
statsmodels' family objects, 50 seeds, and a sweep of `refit_rows` and the
ridge. The script is `scripts/study_500.py`; it generates its own data from
seeded numpy generators and reads nothing from disk. The proposal is task 500
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
