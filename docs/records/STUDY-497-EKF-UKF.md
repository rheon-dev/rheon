# Extended and unscented Kalman filters for GLM observations: a numpy study (PLAN task 497)

*Status, 2026-11-21: a record, in progress. Request A13, part 1: the recursions
in numpy, the linear Gaussian case against filterpy's `KalmanFilter` and
`ExtendedKalmanFilter`, and the Poisson EKF against filterpy's
`ExtendedKalmanFilter` (§3, §4). Nothing in the package or the Rust crates was
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

## 4. Part 1 results: the linear case and the Poisson EKF

Cases: `k` in {2, 5}; `(q, lam)` in {(1e-4, 1), (0, 0.99), (1e-4, 0.995)}; with
and without a zero-weight first row; 2,000 rows each, 10% of rows at weight 0
and 10% at a weight drawn from U(0.5, 2). Gaussian: `phi = 0.5`, `P0 = I`.
Poisson: two priors, "calibrated" `P0 = (0.25/k) I` (an `eta` prior sd of about
0.5) and "wide" `P0 = I`. Gaps are relative to the reference's own size on that
row: `pred` as `|a - r| / |r|`, `b` as `max_j |a_j - r_j| / max_j |r_j|`, `P`
the same over its entries. The **band** is the numpy filter against itself with
`P0` moved by one ulp: the stream's own amplification of a rounding
difference, which a second implementation's gap is read against.

| comparison | pred | b | P |
|---|---|---|---|
| Gaussian, numpy (Joseph) vs `KalmanFilter` (fading `alpha`) | 6.2e-15 abs | 2.2e-15 | 8.9e-15 |
| Gaussian, numpy (Joseph) vs `ExtendedKalmanFilter` | 3.6e-15 abs | 1.5e-15 | 8.0e-15 |
| Gaussian, numpy (standard) vs `KalmanFilter` | 6.2e-15 abs | 2.2e-15 | 9.0e-15 |
| Gaussian band | | 5.7e-16 | 2.3e-15 |
| Poisson EKF, 19 cases that never reach the clamp, Joseph vs filterpy | 6.0e-14 | 7.7e-15 | 3.3e-14 |
| same, standard form vs filterpy | 1.1e-13 | 1.9e-14 | 3.8e-14 |
| same, band | 9.8e-14 | 2.3e-14 | 3.8e-14 |
| Poisson EKF, 5 cases that reach the clamp, Joseph vs filterpy | 2.1e-8 | 5.2e-10 | 9.2e-9 |
| same, band | 1.9e-8 | 3.8e-10 | 8.8e-9 |
| same, standard form vs filterpy (the wide prior) | 6.9e-6 | 2.0e-7 | 3.0e-6 |

(The Gaussian `pred` gap is given in absolute terms: a prediction near 0 makes
the relative one meaningless.)

Read:

1. **The linear case is the same filter**: numpy, filterpy's `KalmanFilter`
   (with its own fading factor) and its `ExtendedKalmanFilter` (with the
   forgetting as `Q`) agree to 1e-14, within a few times the band.
2. **The Poisson EKF matches filterpy to rounding wherever the stream is well
   conditioned** (7.7e-15 on `b`). On the five streams where the EKF
   overshoots to the clamp, both its gap to filterpy and the 1-ulp band grow
   to 1e-10 on `b` together: the filterpy gap is between 0.3 and 6.3 times the
   band. The amplification is the stream's (an EKF that swings to `e^30` and
   back), not a difference between the implementations.
3. **The standard and Joseph covariance forms part by up to 2e-7 on `b`** in
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
