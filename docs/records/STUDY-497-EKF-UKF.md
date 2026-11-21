# Extended and unscented Kalman filters for GLM observations: a numpy study (PLAN task 497)

*Status, 2026-11-21: a record, in progress. Request A13, part 1: the recursions
in numpy, the linear Gaussian case against filterpy's `KalmanFilter` and
`ExtendedKalmanFilter`, and the Poisson EKF against filterpy's
`ExtendedKalmanFilter` (§3, §4). Part 2: the UKF for every family and the EKF
for binomial and Gamma against filterpy (§5). Part 3: the second oracle (a
grid filter, a particle filter and statsmodels' static posterior, §6), `k =
20` and zero-weight rows (§7), and divergence (§8). Nothing in the package or the Rust crates was
changed. The scan this builds on is `vm-tools/research/494-MODELS-SCAN.md` on
branch `vm-tools/oracle-gap-scan`, section 3 ("scan §3"). The runnable study is
`scripts/study_497.py`.*

## 1. The question

Scan §3 proposes family A: the `kalman` model's random-walk coefficients with an
observation from a GLM family (Poisson with the log link, binomial with the
logit link, Gamma with the log link), filtered by an extended Kalman filter
(EKF, linearised at the prior mean) or an unscented Kalman filter (UKF, sigma
points), as an option of `kalman`. Before any Rust:

1. Do the recursions, written row by row in numpy, equal filterpy's
   `ExtendedKalmanFilter` and `UnscentedKalmanFilter` on the same rows with the
   same noise, and under which settings exactly?
2. How far are they from the exact filtering posterior, which no library
   computes for a dynamic GLM? (The second oracle, §6.)
3. Where do they diverge: extreme linear predictors, separation, wide priors?
4. What do they cost per row, and how large is their state as a function of `k`?

## 2. The recursion

State per target: the coefficient mean `b` (k), its covariance `P` (k x k,
symmetric), `D` (the clock since `P` last took an observation) and the weight
sum `n`. Per row, with clock step `d` (1 here) and weight `w`:

```text
D    <- D + d
P^-  = P / lam^D + q D^2 I           (forgetting factor, then random-walk noise)
pred = g^{-1}(x' b)                  (EKF; the UKF emits its sigma-point mean)
n_eff emitted = n,  n <- lam^d n + w
```

and only on a row that observes its target (positive weight, finite `y`):

```text
EKF   eta = x'b, mu = g^{-1}(eta), m = dmu/deta, R = phi V(mu) / w
      s = m^2 x'P^-x + R,  K = (m / s) P^- x,  b <- b + K (y - mu)
      P <- P^- - s K K'                               ("standard", rheon's form)
      P <- (I - K H) P^- (I - K H)' + R K K', H = m x' ("joseph", filterpy's form)
UKF   chi = b, b +- columns of chol(c P^-),  z_i = g^{-1}(x' chi_i)
      zbar = sum Wm_i z_i,  R = phi sum Wm_i V(z_i) / w   ("expected")
                          or phi V(g^{-1}(x'b)) / w       ("plugin", as the EKF)
      s = sum Wc_i (z_i - zbar)^2 + R,  C = sum Wc_i (chi_i - b)(z_i - zbar)
      K = C / s,  b <- b + K (y - zbar),  P <- P^- - s K K'
then  D <- 0
```

The prior counts as observed at clock 0, so the first row has `D = 1`. A row
that observes nothing (weight 0, null target) only adds its clock to `D`: the
noise `q D^2` is charged once per observation for the whole gap, as `kalman`
charges it, and `lam^D` is the same over one gap as over any cut of it. The
prediction is computed before the row's target is read.

Families: `gaussian` (identity, `V = 1`, `phi = sigma^2`), `poisson` (log,
`V = mu`, `phi = 1`), `binomial` (logit, `V = mu(1 - mu)`, `phi = 1`), `gamma`
(log, `V = mu^2`, `phi = 1/shape`, fixed).

**The clamp.** `eta` is clamped to `|eta| <= 30` for the log link and `36` for
the logit link, and the EKF is linearised at the clamped value: `mu`, `dmu/deta`
and `V` are all read at `clamp(eta)`. The first version used the clamp's own
derivative (0 beyond the bound); on the forgetting stream of §4 (`k = 2`,
`q = 0`, `lam = 0.99`) one overshoot to the bound then froze the filter for the
remaining 1,137 rows, every prediction `e^30`. Linearised at the bound, a row
beyond it pulls `eta` back by about 1 a row (`Δeta ≈ -x'P^-x / (x'P^-x + 1/mu)`).

**Equivalence with the working-response form.** Linearising in `eta` space
(observation `z = eta + (y - mu) / m`, variance `R / m^2`, `H = x'`) gives the
same gain times innovation: multiply numerator and denominator of `K (y - mu)`
by `1/m^2`. The two differ in rounding only. With `q = 0` and `lam = 1` the EKF
is one Fisher-scoring step per row on the posterior, the recursive IRLS of the
GLM study (task 500), with the prior's `P0^{-1}` as its ridge.

## 3. Oracle 1, the settings that make filterpy compute the same thing

filterpy 2.0.0 (numpy 2.5.5, scipy 1.18.1). The numpy filter and filterpy are
fed the same rows; filterpy's `predict`/`update` pair is called only on rows
that observe their target, with the clock gathered as above.

- **Transition**: `F = I` (EKF) and `fx(b, dt) = b` (UKF).
- **Process noise and forgetting**: `Q = (lam^{-D} - 1) P + q D^2 I`, set
  before each `predict`, so that filterpy's `F P F' + Q` is the numpy filter's
  `P / lam^D + q D^2 I` (equal in exact arithmetic; the rounding differs). For
  the linear case, filterpy's `KalmanFilter` also has its own fading factor:
  `alpha = lam^{-D/2}`, with `Q = q D^2 I`; both forms were run.
- **EKF**: `Hx(b) = g^{-1}(clamp(x'b))`, `HJacobian(b) = (dmu/deta)(clamp(x'b)) x'`,
  `R = phi V(Hx(b)) / w`, all evaluated at filterpy's `x` when `update` is
  called, which is the prior mean (the linearisation point). filterpy's EKF
  updates `P` by the Joseph form; the numpy filter's `form="joseph"` is the
  same formula.
- **UKF**: `hx(b) = g^{-1}(clamp(x'b))` with the row's `x`, the sigma points
  `MerweScaledSigmaPoints(k, alpha, beta, kappa)` or `JulierSigmaPoints(k,
  kappa)` with filterpy's default root (scipy's upper Cholesky of `c P`, whose
  rows are the numpy filter's lower-Cholesky columns), and `R` passed to
  `update` per row: `phi V(hx(x)) / w` ("plugin") or `phi sum Wm_i V(z_i) / w`
  over filterpy's own `sigmas_f` ("expected"). filterpy 2.0's `predict` pushes
  the points through `fx`, takes their weighted mean and covariance plus `Q` as
  the prior, and **redraws** `sigmas_f` from that prior; `update` uses the
  redrawn points. With `fx = I` that prior is `P + Q` up to rounding: the
  unscented transform of `P`'s own points returns `P` exactly only in exact
  arithmetic, and `x` comes back as `sum Wm_i chi_i`, not `b`. The "exact
  prior" variant replaces `x`, `P` and `sigmas_f` after `predict` by the numpy
  filter's `b`, `P^-` (symmetrised) and its sigma points, which isolates the
  update from that rounding. Earlier filterpy versions (1.4) did not redraw
  and used the points propagated from the previous posterior, which
  ignores `Q` in the cross-covariance: a different filter for a random walk.

## 4. Part 1 results: the linear case and the Poisson EKF

Cases: `k` in {2, 5}; `(q, lam)` in {(1e-4, 1), (0, 0.99), (1e-4, 0.995)}; with
and without a zero-weight first row; 2,000 rows each, 10% of rows at weight 0
and 10% at a weight drawn from U(0.5, 2). Gaussian: `phi = 0.5`, `P0 = I`.
Poisson: two priors, "calibrated" `P0 = (0.25/k) I` (an `eta` prior sd of about
0.5) and "wide" `P0 = I`. Gaps are relative: `pred` as `|a - r| / |r|` on each
row; `P` as `max |a - r| / max |r|` over its entries on each row; `b` as the
run's largest `|a_j - r_j|` over the run's largest `|r_j|` (a `b` that passes
near 0 on some row makes a per-row ratio for `b` meaningless). The **band** is the numpy filter against itself with
`P0` moved by one ulp: the stream's own amplification of a rounding
difference, which a second implementation's gap is read against.

| comparison | pred | b | P |
|---|---|---|---|
| Gaussian, numpy (Joseph) vs `KalmanFilter` (fading `alpha`) | 6.2e-15 abs | 1.2e-15 | 8.9e-15 |
| Gaussian, numpy (Joseph) vs `ExtendedKalmanFilter` | 3.6e-15 abs | 1.0e-15 | 8.0e-15 |
| Gaussian, numpy (standard) vs `KalmanFilter` | 6.2e-15 abs | 1.2e-15 | 9.0e-15 |
| Gaussian band | | 4.4e-16 | 2.3e-15 |
| Poisson EKF, 19 cases that never reach the clamp, Joseph vs filterpy | 6.0e-14 | 5.6e-15 | 3.3e-14 |
| same, standard form vs filterpy | 1.1e-13 | 1.0e-14 | 3.8e-14 |
| same, band | 9.8e-14 | 1.1e-14 | 3.8e-14 |
| Poisson EKF, 5 cases that reach the clamp, Joseph vs filterpy | 2.1e-8 | 5.1e-10 | 9.2e-9 |
| same, band | 1.9e-8 | 3.6e-10 | 8.8e-9 |
| same, standard form vs filterpy (the wide prior) | 6.9e-6 | 1.9e-7 | 3.0e-6 |

(The Gaussian `pred` gap is given in absolute terms: a prediction near 0 makes
the relative one meaningless.)

Read:

1. **The linear case is the same filter**: numpy, filterpy's `KalmanFilter`
   (with its own fading factor) and its `ExtendedKalmanFilter` (with the
   forgetting as `Q`) agree to 1e-14, within a few times the band.
2. **The Poisson EKF matches filterpy to rounding wherever the stream is well
   conditioned** (5.6e-15 on `b`). On the five streams where the EKF
   overshoots to the clamp, both its gap to filterpy and the 1-ulp band grow
   to 1e-10 on `b` together: the filterpy gap is between 0.3 and 2.0 times the
   band. The amplification is the stream's (an EKF that swings to `e^30` and
   back), not a difference between the implementations.
3. **The standard and Joseph covariance forms part by up to 1.9e-7 on `b`** in
   those streams (a cancellation in `P^- - s K K'` at a large gain), 500 times
   the band. A Rust EKF matched against filterpy must use the Joseph form, or
   be held to the band of its own form; `kalman`'s covariance form today is the
   standard one (or Carlson's square root).
4. Out of sample: changing row `t`'s target (+17) and weight (3) moved no
   prediction up to and including row `t` (0.0, rows 0, 1, 50, 299).
5. Zero-weight rows: the run equals the same stream with those rows removed
   and their clock folded into the next observed row's `d`, to the bit
   (0.0). A zero-weight first row leaves `b` at the prior, to the bit.

**Finding, carried to part 3: the EKF diverges on the wide-ranging streams.**
Seed 1020 (`k = 2`, true `eta` from -7.6 to 9.8, slope 2.1) under forgetting
`lam = 0.99`, `q = 0`, lost track at row 237 even from the calibrated prior: `P`
had shrunk to 3e-6 on the high-count rows, a predicted `eta` of -15.5 against
a true 0.4 followed, and the next rows swung between the bounds (93 rows at the
clamp). Six of the twelve wide-prior Poisson streams reached a prediction of
`e^20` or more. This is the mean-space linearisation's known weakness: at a
small `mu` and a large count the step in `eta` is about `x'P^-x · y`.

Run: `.venv-497/bin/python scripts/study_497.py part1 --out <dir>` (10 s).

## 5. Part 2 results: the UKF for every family, the EKF for binomial and Gamma

The same twelve stream settings per family as §4 (`k` in {2, 5}, three `(q,
lam)`, with and without a zero-weight first row), calibrated prior, 2,000
rows. Binomial: Bernoulli rows; Gamma: `phi = 0.5` (shape 2). Sigma points:
Merwe `(alpha, beta, kappa) = (1, 2, 0)` for every family, and also Merwe
`(0.5, 2, 0)` (a negative centre weight `Wm_0 = -3`) and Julier `kappa = 1`
for Poisson and binomial; each non-Gaussian family with both `R` forms. No
stream in part 2 reached the clamp. Largest gap over the twelve streams of
each row of the table (metrics as §4):

| comparison | pred | b | P |
|---|---|---|---|
| EKF binomial, Joseph vs filterpy | 1.8e-14 | 6.1e-15 | 1.8e-14 |
| EKF binomial, band | 3.0e-15 | 8.1e-16 | 2.5e-15 |
| EKF Gamma, Joseph vs filterpy | 7.9e-15 | 1.3e-15 | 7.2e-15 |
| EKF Gamma, band | 3.6e-15 | 3.9e-16 | 2.5e-15 |
| UKF Gaussian, Merwe (1, 2, 0), vs filterpy `KalmanFilter` | 5.3e-15 abs | 1.9e-15 | 9.5e-15 |
| UKF, every family, point set and `R` form, vs filterpy as it runs | 1.9e-13 | 5.1e-14 | 4.1e-14 |
| same, vs filterpy with the exact prior | 7.4e-15 | 1.7e-15 | 7.2e-15 |
| same, band | 9.8e-15 | 2.5e-15 | 9.7e-15 |

By family, UKF against filterpy as it runs (`b`): Gaussian 7.0e-15, Poisson
2.2e-14, binomial 5.1e-14, Gamma 7.1e-15; the largest are with the Merwe
`alpha = 0.5` set, whose centre weight of -3 costs some digits in the weighted
sums. With the exact prior every one of the 180 cases is within 9e-16 of its
own band on `b` (as a ratio, up to 10 times, where the band is itself near
1e-16).

Read:

1. **The UKF is filterpy's UKF**, to 5e-14 as filterpy runs it, and to the
   1-ulp band once filterpy is handed the same prior. The remaining 5e-14 is
   filterpy's prior recomputed by the unscented transform (`sum Wc_i (chi_i -
   xbar)(chi_i - xbar)'` for `P`, `sum Wm_i chi_i` for `b`), which a Rust UKF
   need not copy: with `fx = I` the prior is `P / lam^D + q D^2 I` exactly.
2. **On the linear case the UKF is the Kalman filter** (1.9e-15 on `b`): the
   unscented transform of a linear function is exact.
3. **Under forgetting without process noise, `P` must be kept symmetric.**
   The "exact prior" variant first passed filterpy the numpy filter's `P^-`
   unsymmetrised, and the gap grew steadily to 8.5e-7 on `P` by row 2,000 at
   `lam = 0.99`, `q = 0`: the update `P^- - s K K'` leaves the antisymmetric
   part of `P` (rounding) untouched, `1/lam^D` multiplies it every row
   (`0.99^{-2000} = e^{20}`), and the Cholesky reads one triangle only. The
   numpy filter symmetrises after each update; a packed (upper-triangle) store
   in Rust is symmetric by construction. filterpy's own `predict` hides this,
   since its prior is rebuilt symmetric from the points every row.
4. The EKF results of §4 hold for binomial and Gamma: Joseph form against
   filterpy at 6e-15 on `b`, a few times the band.

Run: `.venv-497/bin/python scripts/study_497.py part2 --out <dir>` (about 200 s;
filterpy's UKF is a Python loop over the sigma points).

## 6. Part 3: the second oracle

No library fits a dynamic GLM, so the second oracle is the exact filtering
posterior, computed three ways, each where it can be computed well:

- **k = 2: a grid filter** (`grid_filter`), written from the definition: the
  posterior density on an `n x n` grid; the forgetting as power discounting
  (`density^(lam^D)`, which is exactly `P / lam^D` for a Gaussian and the
  standard non-Gaussian generalisation); the random walk as a convolution with
  `N(0, q D^2 I)` applied in Fourier space; Bayes' rule with the row's
  likelihood to the power of its weight. The grid spans the prior, the true
  path and both filters' paths with 8 sd to spare; its spacing is at most 0.28
  of the narrowest posterior sd the UKF reports (`n` up to 500 a side). It is
  checked twice: on the Gaussian family against filterpy's `KalmanFilter`
  (the exact posterior there), and against itself at 0.75 of the resolution
  (the grid's band).
- **k = 5: a bootstrap particle filter** (`particle_filter`), 50,000
  particles, systematic resampling under an effective sample size of N/2, four
  independent replicates; the band is 4 Monte Carlo sd of the replicates'
  mean. Random walk only (`lam = 1`): power discounting has no particle form
  without a further approximation. The first 50 rows are left out, where the
  wide prior leaves the particles few.
- **k = 2, 5, 20 with `q = 0`, `lam = 1`: statsmodels.** The static posterior's
  mode, from statsmodels' `GLM(..., var_weights=w)` score and observed Hessian
  at the fixed scale `phi`, plus the prior's `-b/p0` and `-I/p0`, by Newton's
  method to a gradient under 6e-13; its Laplace covariance is the inverse of
  the negative Hessian there. The mode is not the mean, by `O(1/n)` sd; at
  2,000 rows that difference is small beside the gaps reported.

Units: `z` is `|b_filter - b_exact|` over the exact posterior sd, per
coefficient and row; "sd ratio" is the filter's sd over the exact sd; "pred" is
`|pred - E[mu]| / E[mu]`, the exact predictive mean of the row taken under the
row's prior (the EKF's prediction is the plug-in `g^{-1}(x'b)`, the UKF's the
sigma-point mean). Streams of 1,000 rows, `Var(x'beta) ≈ 0.25`, calibrated
prior, the same weights as §4.

**The grid is exact to 1e-9 sd.** Gaussian: grid against `KalmanFilter`, `z`
at most 1.2e-9 (`q > 0`; the Fourier convolution) and 7.6e-15 (`q = 0`, `lam =
0.99`), sd ratio 1 to 1e-10; the grid's own band at most 2.4e-9 sd in every
family.

| family | `(q, lam)` | EKF `z` max / median | UKF `z` max / median | EKF sd ratio | UKF sd ratio | EKF pred median / max | UKF pred median / max |
|---|---|---|---|---|---|---|---|
| Poisson | (1e-4, 1) | 0.94 / 0.059 | 0.55 / 0.011 | 0.95 – 1.21 | 0.96 – 1.34 | 0.0038 / 0.16 | 0.0012 / 0.20 |
| Poisson | (0, 0.99) | 0.79 / 0.062 | 0.44 / 0.017 | 0.95 – 1.18 | 0.96 – 1.22 | 0.0035 / 0.14 | 0.0016 / 0.16 |
| Poisson | (1e-4, 0.995) | 0.96 / 0.068 | 0.59 / 0.011 | 0.95 – 1.21 | 0.96 – 1.35 | 0.0044 / 0.17 | 0.0013 / 0.21 |
| binomial | (1e-4, 1) | 0.077 / 0.023 | 0.040 / 0.006 | 0.96 – 1.01 | 0.97 – 1.01 | 0.0011 / 0.052 | 0.0007 / 0.012 |
| binomial | (0, 0.99) | 0.24 / 0.035 | 0.14 / 0.016 | 0.89 – 1.03 | 0.90 – 1.04 | 0.0034 / 0.046 | 0.0024 / 0.039 |
| binomial | (1e-4, 0.995) | 0.15 / 0.034 | 0.051 / 0.008 | 0.95 – 1.02 | 0.95 – 1.02 | 0.0022 / 0.079 | 0.0010 / 0.012 |
| Gamma | (1e-4, 1) | 0.36 / 0.067 | 0.23 / 0.032 | 0.92 – 1.08 | 0.93 – 1.17 | 0.0047 / 0.21 | 0.0035 / 0.075 |
| Gamma | (0, 0.99) | 0.37 / 0.017 | 0.23 / 0.058 | 0.92 – 1.08 | 0.95 – 1.18 | 0.0037 / 0.21 | 0.0055 / 0.078 |
| Gamma | (1e-4, 0.995) | 0.36 / 0.062 | 0.23 / 0.039 | 0.91 – 1.08 | 0.93 – 1.17 | 0.0045 / 0.21 | 0.0047 / 0.075 |

Particle filter, `k = 5`, `q = 1e-4`, rows 50 to 1,000:

| family | ESS min | PF mc sd / post sd, median / max | EKF `z` max (where PF sharp) | EKF beyond the band | UKF `z` max (where PF sharp) | UKF beyond the band |
|---|---|---|---|---|---|---|
| Poisson | 176 | 0.012 / 1.33 | 2.67 (0.29) | 0.76 | 2.15 (0.25) | 0.78 |
| binomial | 17,008 | 0.020 / 0.086 | 0.21 (0.21) | 0.14 | 0.13 (0.094) | 0.030 |
| Gamma | 3,099 | 0.009 / 0.15 | 4.09 (3.72) | 3.80 | 0.37 (0.36) | 0.28 |

("where PF sharp": the cells whose Monte Carlo sd is under 0.05 posterior sd,
80% to 92% of them; "beyond the band": the largest gap past 4 Monte Carlo sd,
in posterior sd.) The Poisson stream's PF collapses to an ESS of 176 on one
row, where its own error reaches 1.3 posterior sd; the EKF and UKF maxima of
2.7 and 2.2 sd sit on such rows, and where the PF is sharp both are within 0.3
sd. On the Gamma stream the EKF is 3.7 sd from a sharp PF: it has lost track
(§8), the UKF has not.

statsmodels, the static posterior's mode at row 2,000 (`z` in Laplace sd; the
sd ratio is the filter's sd over the Laplace sd):

| family | k | EKF `z` max | EKF sd ratio | UKF `z` max | UKF sd ratio |
|---|---|---|---|---|---|
| Poisson | 2 | 0.10 | 1.01 | 0.052 | 1.01 |
| Poisson | 5 | 0.25 | 1.01 | 0.19 | 1.01 – 1.02 |
| Poisson | 20 | 0.70 | 1.00 – 1.02 | 0.26 | 1.01 – 1.03 |
| binomial | 2 | 0.083 | 0.98 – 0.99 | 0.027 | 0.98 – 0.99 |
| binomial | 5 | 0.85 | 0.90 – 0.96 | 0.68 | 0.90 – 0.96 |
| binomial | 20 | 0.17 | 0.97 – 0.98 | 0.10 | 0.98 – 0.99 |
| Gamma | 2 | 0.24 | 1.00 | 0.14 | 1.00 |
| Gamma | 5 | 0.61 | 0.99 – 1.01 | 0.51 | 1.00 – 1.02 |
| Gamma | 20 | 2.59 | 0.98 – 1.02 | 1.04 | 0.99 – 1.03 |

Read:

1. **Both filters are approximations, and the gap is a statistic, not a
   rounding error.** Against the exact posterior the EKF's mean is typically
   0.02 to 0.07 sd off on the grid and at worst about 1 sd (Poisson, `k = 2`);
   the UKF's typically 0.006 to 0.06 and at worst 0.6. The UKF is the closer
   of the two on 8 of the 9 grid settings by the median and on all 9 by the
   maximum (the exception is Gamma under pure forgetting by the median), on
   all three PF streams, and on all 9 static cases at row 2,000 (7 of 9 at row
   200).
2. **Neither filter's sd is far off** on the grid: 0.89 to 1.35 of the exact
   sd. The UKF errs wide (up to 1.35), the EKF both ways.
3. **The UKF's prediction is the better one for Poisson and binomial**: its
   median relative error in `E[mu]` is 0.07% to 0.24% against the EKF's 0.11%
   to 0.44% on those six settings; the plug-in `g^{-1}(x'b)` misses the
   `exp(x'Px / 2)` of a log link. For Gamma the medians are even (0.35% to
   0.55% against 0.37% to 0.47%) and the UKF's maximum is the smaller (7.5%
   against 21%). The largest Poisson errors (14% to 21%) are as large in
   both filters.
4. **With `q = 0` the one-pass EKF does not converge to the static posterior**
   as `k` grows: 0.70 sd (Poisson) and 2.6 sd (Gamma) from the mode at `k =
   20` after 2,000 rows, where the UKF is 0.26 and 1.0. The early rows'
   linearisation error is never revisited; the batch fit relinearises at the
   end. This is task 500's finding for recursive IRLS, which the EKF with `q =
   0` is.

Run: `.venv-497/bin/python scripts/study_497.py part3 --out <dir>` (about 620 s:
the grids take 30 to 40 s each). The particle-filter rows were recomputed with
the final `_pf_case` (the band columns were added after the first run; same
seeds, same draws).

## 7. Part 3: k = 20 against filterpy, and zero-weight rows

`k = 20`, `q = 1e-4`, `lam = 0.995`, a zero-weight first row, 2,000 rows,
calibrated prior:

| family | EKF (Joseph) vs filterpy, `b` | band | UKF vs filterpy as it runs | UKF vs filterpy, exact prior | band |
|---|---|---|---|---|---|
| Poisson | 7.1e-16 | 4.3e-16 | 3.3e-15 | 4.5e-16 | 5.3e-16 |
| binomial | 5.8e-15 | 8.6e-16 | 9.6e-15 | 9.6e-16 | 8.0e-16 |
| Gamma | 5.8e-16 | 7.2e-16 | 5.8e-15 | 5.2e-16 | 5.9e-16 |

(`pred` at most 5.8e-14 and `P` 1.8e-14, per row.) The match does not degrade
with `k`.

**Zero-weight rows**, 18 cases (three families, EKF and UKF, `k` = 2, 5 and
20), each with its first 25 rows at weight 0 and a null target among them,
`q = 1e-3`, `lam = 0.99`: through the run of zero-weight rows `b` and `P` stay
at the prior to the bit, `n_eff` is 0 on every row up to and including the
first row after the run, nothing is NaN, and the whole run equals the stream with
those rows removed and their clock folded into the first observing row's `d`,
to the bit. Half of the streams of §4 and §5, and every `k = 20` stream
above, start with a zero-weight row, and those match filterpy as the others
do. No division in either filter is by a weight sum: the weight
divides `R` only on a row whose weight is positive, so the first-row `0/0` of
a mean-form update does not arise.

## 8. Part 3: divergence

Four probes (`divergence` in the script; 1,000 rows unless stated).

**D1, overshoot over seeds.** 50 seeds per cell, `q = 1e-4`, `lam = 1`,
`Var(x'beta) ≈ 1`, at the calibrated and the wide prior. "Clamp": the filter's
linear predictor reached the bound on some row; "lost": at the last row some
coefficient is more than 4 of the filter's own sd from the truth.

| family | k | prior | EKF clamp / lost | UKF clamp / lost |
|---|---|---|---|---|
| Poisson | 2 | calibrated | 2 / 0 | 1 / 0 |
| Poisson | 2 | wide | 1 / 1 | 0 / 0 |
| Poisson | 5 | calibrated | 1 / 0 | 1 / 0 |
| Poisson | 5 | wide | 2 / 1 | 0 / 0 |
| binomial | 2, 5 | both | 0 / 0 | 0 / 0 |
| Gamma | 2 | calibrated | 0 / 0 | 0 / 0 |
| Gamma | 2 | wide | 2 / 2 | 0 / 0 |
| Gamma | 5 | calibrated | 1 / 1 | 0 / 0 |
| Gamma | 5 | wide | 10 / 10 | 0 / 0 |

Of 600 streams per filter the EKF lost track on 15 and the UKF on none; the
UKF reached the clamp twice and came back both times.

**D2, feature spikes.** `beta = (1, 0.1)` fixed; `x1 = 60, 300, 1000` on rows
250, 500 and 750 (true `eta` 7, 31 and 101; past 30 the target is the true
mean, rounded); the filter starts with `b1 = 0.4` at the calibrated prior,
`q = 1e-4`.

| filter | clamp | `eta` predicted at the three spikes | rows until within 2 sd of the truth after each |
|---|---|---|---|
| EKF | yes | 0.5, 809, 248 | 480, 230, not by row 1,000 |
| EKF | no | 0.5, 809, then NaN | fails at row 500 (`e^809` overflows) |
| UKF | yes | 0.6, 60, 135 | 0, 76, 0 |
| UKF | no | 0.6, 60, 131 | 0, 0, 0 |

At the first spike the EKF's prior predicts `e^0.5` against a count near
`e^7`, and its step in `eta` is about `x'P^-x · (y - mu) / (mu x'P^-x + 1)`
with `x'P^-x` large on a row with `x1 = 60`: `b1` jumps to 10.9, and the
next spike puts `eta` at 809. After the third spike the clamped EKF's `b`
reaches 1e27 (finite only because every later `eta` is clamped). The UKF's sigma points span the curvature of
`exp` and its step is bounded by the spread of the `z_i`. **The clamp is
necessary for the EKF** (without it the state is NaN at the next spike) **and
not sufficient** (it then takes hundreds of rows to recover). Linearising at
the clamped value, not differentiating the clamp, is what lets it recover at
all (§2).

**D3, separation (binomial logit).** `y = 1[x1 > 0]`, 10,000 rows:

| `(q, lam)` | filter | `b1` at rows 100 / 1,000 / 5,000 / 9,999 | sd of `b1` at the same rows | rows at the clamp |
|---|---|---|---|---|
| (0, 1) | EKF | 1.39 / 2.99 / 4.52 / 5.32 | 0.20 / 0.12 / 0.088 / 0.076 | 0 |
| (0, 1) | UKF | 1.43 / 3.04 / 4.58 / 5.39 | 0.21 / 0.13 / 0.090 / 0.078 | 0 |
| (1e-4, 1) | EKF | 1.43 / 4.09 / 10.5 / 16.4 | 0.22 / 0.26 / 0.49 / 0.67 | 49 |
| (1e-4, 1) | UKF | 1.48 / 4.14 / 10.6 / 16.5 | 0.22 / 0.26 / 0.49 / 0.67 | 51 |
| (0, 0.99) | EKF | 1.71 / 16.9 / 3.0e5 / 2.1e14 | 0.30 / 3.7 / 2.0e4 / 7.3e6 | 8,103 |
| (0, 0.99) | UKF | 1.77 / 18.4 / 2.9e5 / 1.1e11 | 0.31 / 4.0 / 1.6e5 / 8.9e10 | 8,411 |

Without forgetting the coefficient grows like `log t` (the prior bounds it)
or, under a random walk, like `√t`, and stays finite. **Under pure forgetting
it does not**: once the fitted probabilities saturate, a row carries no
information in the separating direction, and `P / lam^D` grows as `lam^{-t}`
there without bound. Neither filter had overflowed by row 10,000; at that rate
`P` would pass `1e308` near row 70,000 (an extrapolation, not run). A
forgetting factor without process noise needs a bound on `P` (for example
`P <= P0` in the order of positive semi-definite matrices, or forgetting only
the directions a row informs); the random walk alone does not have this
failure. Rheon's `kalman` uses the random walk.

**D4, the UKF's centre weight.** Merwe `(alpha, 2, 0)` at `k = 5` has `Wm_0 =
0, -3, -99, -999,999` for `alpha = 1, 0.5, 0.1, 0.001`. A negative centre
weight can make the "expected" `R` or the innovation variance negative for a
variance function that is not convex (binomial). On 10 Poisson and 10
binomial streams each, no setting failed (no non-positive-definite `P`, no
non-finite state). With `alpha` small the points sit close to `b` and the UKF
tends to a second-order EKF; the weights of `-1e6` cost about six digits in
`zbar` (not measured beyond "no failure").

Run: `.venv-497/bin/python scripts/study_497.py divergence --out <dir>` (80 s).
