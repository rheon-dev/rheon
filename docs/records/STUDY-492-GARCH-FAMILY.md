# Study 492, part 1: online GJR-GARCH, EGARCH and GARCH(2,2), Gaussian

Python and numpy only (`scripts/study_492.py`); no Rust, no source change. The
question: can the online GARCH estimator of study 485 (requests A9 to A12:
variance-targeted GARCH(1,1), a 500-row batch start by a small projected-Newton
optimizer, a Huber-clipped score) extend to GJR-GARCH (leverage), EGARCH and
GARCH(2,2), and how near do the extensions stay to `arch`'s batch fits on the
same rows? This part is Gaussian innovations. Student-t, skew-t and FIGARCH
come in parts 2 and 3.

Status: part 1 complete for the three models at 10,000 rows: 50 seeds (492000 to 492049) for GJR and EGARCH, 49 for GARCH(2,2) (the 50th was still running at the commit; the figures are over the 49 finished). Open:
EGARCH has no variance-targeted form (see 1.3); the GARCH(2,2) oracle
comparison is weak for a reason that belongs to the model, not the estimator
(see 4).

## 1. The math

Notation: `e_t` the zero-mean residual, `s_t = sigma2_t` computed before row
`t`'s residual is read (so it is an out-of-sample prediction of `e_t^2`),
`z_t = e_t / sigma_t`. The log-likelihood of one row is
`l_t = -1/2 (ln 2 pi + ln s_t + e_t^2 / s_t)`.

### 1.1 The three models

**GJR-GARCH(1,1)** (arch: `p=1, o=1, q=1`):

    s_t = omega + (alpha + gamma 1[e_{t-1} < 0]) e_{t-1}^2 + beta s_{t-1}

Constraints (arch's, plus the base's stationarity margin): `omega >= floor`,
`alpha >= 0`, `alpha + gamma >= 0`, `beta >= 0`,
`alpha + gamma/2 + beta <= 1 - 1e-4`. For symmetric `z`, the persistence is
`alpha + gamma/2 + beta`.

**EGARCH(1,1,1)** (arch: `vol='EGARCH', p=1, o=1, q=1`), with `c = E|z| = sqrt(2/pi)`:

    log s_t = omega + beta log s_{t-1} + alpha (|z_{t-1}| - c) + gamma z_{t-1}

No sign constraints on `omega, alpha, gamma`; `0 <= beta <= 1 - 1e-4`. `s_t` is
positive by construction, so there is no variance floor.

**GARCH(2,2)** (arch: `p=2, o=0, q=2`):

    s_t = omega + alpha_1 e_{t-1}^2 + alpha_2 e_{t-2}^2 + beta_1 s_{t-1} + beta_2 s_{t-2}

Constraints: `omega >= floor`, all `alpha_i, beta_j >= 0`,
`alpha_1 + alpha_2 + beta_1 + beta_2 <= 1 - 1e-4`.

### 1.2 The online step

Every model is written `s_t = s_t(theta)` with a sensitivity
`d_t = d s_t / d theta` carried by its own recursion at the current parameters.

Score and information of one row, with `dl_t = d log s_t / d theta`:

    g_t = 1/2 (z_t^2 - 1) dl_t          (the Gaussian quasi-likelihood score)
    I_t = 1/2 dl_t dl_t'                (its expected-information form)

The recursions:

* **GJR and GARCH(p,q)**: `d_t = x_t + sum_k beta_k d_{t-k}` with the regressor
  `x_t = (1, e_{t-1}^2 .. e_{t-p}^2, 1[e<0] e^2 lags, s_{t-1} .. s_{t-q})`;
  `dl_t = d_t / s_t`.
* **EGARCH**: `d log s_t / d log s_{t-1}` is `beta - (alpha |z| + gamma z)/2`,
  because `dz/d log s = -z/2` and `d|z|/d log s = -|z|/2`; so
  `dl_t = (1, |z_{t-1}| - c, z_{t-1}, log s_{t-1}) + (beta - (alpha |z_{t-1}| + gamma z_{t-1})/2) dl_{t-1}`.
  This is settled (it is a chain-rule step) and `check` compares it to a
  central difference of the log-likelihood.

The step, with `gamma_t = 1/(t + 500)` after the 500-row start:

    H_t   = H_{t-1} + gamma_t (I_t - H_{t-1})
    theta = Proj(theta + gamma_t H_t^{-1} g_t)

`Proj` is the Euclidean projection onto the constraint set of 1.1, computed by
Dykstra's algorithm over its half-spaces (the GARCH forms); for EGARCH it is
the clip of `beta`. The Huber variant replaces `z^2 - 1` by the base's clipped
residual at `c = 2` with running centring and slope.

**The start** (base A10 and A12): the first 500 rows are buffered and emit
nothing; `batch_core.maximize` (the base's core-held projected Newton) fits them
from a small grid of starting points; the buffer is replayed at that fit to
build `H` as an equal-weight mean; steps run from row 501.

**Variance targeting** (base A9) is available for the GARCH forms:
`omega = v (1 - c.phi)`, `v` the running mean of `e^2` and `phi` the other
parameters, with weights `c = 1` on `alpha, beta` and `1/2` on `gamma`. The free
parameters' sensitivity is `d_phi - v c d_omega`, so no new recursion is
carried. The batch start's `v` is the buffer's mean.

### 1.3 What is not derived

EGARCH has no variance target: the unconditional `E[log s]`, not `E[s]`, is
what `omega` fixes, and `E[s] = exp(omega/(1-beta)) E[exp(...)]` has no
closed form in `(alpha, gamma)` for the Gaussian case that this part
derived. The EGARCH estimator is therefore the unrestricted-`omega` form,
which the base found sticks on the persistence ridge without a batch start.
The part-3 Student-t and skew-t forms need their own score and are not
attempted here.

### 1.4 State

Floats per series, `k = 1 + p + o + q` parameters (`k - 1` free when targeted):

    theta k, H k(k+1)/2, d (q sensitivity lags, k each) q k,
    residual lags max(p, o), asymmetric-term lags o, variance lags q,
    running mean of e^2, its count, step count, two Huber means  (5 scalars)

so `floats = k + k(k+1)/2 + max(p,o) + o + q + q k + 5`, bytes `8 * floats`,
bounded by the model's shape, never by the stream (plus the 500-row buffer,
4 KB, until the start). EGARCH (`k = 4`) carries `z`, `log s`, `dl`.

| model | k | floats | bytes |
|---|---|---|---|
| GARCH(1,1) (base, targeted) | 3 | 16 | 128 |
| GJR(1,1) | 4 | 26 | 208 |
| EGARCH(1,1) | 4 | 25 | 200 |
| GARCH(2,2) | 5 | 39 | 312 |

## 2. Verification of the work-in-progress script (step 0)

* `study_492.py check`: the analytic score against a central difference of
  the log-likelihood, at the truth on 800 rows: max relative gap 5e-8 (GJR),
  8e-7 (EGARCH), 3e-7 (GARCH(2,2)). Passes.
* The fixed-parameter filter against `arch`'s conditional variance at `arch`'s
  own parameters: after row 500 within 1e-15 for all three; **from row 1,
  GJR was off by 5.7%** and decaying: the script read the pre-sample
  asymmetric term as zero, while `arch` uses `0.5 * backcast`. Fixed
  (`GarchPOQ.init` keeps an asymmetric-term lag, `a`, initialised at
  `0.5 * start`); now within 8e-16 on every row, all three models.
* GARCH(1,1) (`GarchPOQ(1,0,1)` with the WIP's unrestricted `omega`) against the
  base's `garch.OnlineGarch(target='running', batch_start=True,
  batch_opt='core', robust_c=2, warm=500, t0=500)` on three seeds: both
  within 1.1 SE of `arch` on every parameter; their last-1000 predictions
  differ by 5 to 7% at the worst row (the base is targeted, the WIP is not,
  so they are different estimators, as expected). Consistent, kept.
* One difference from the base remains by design: the WIP's default is the
  unrestricted `omega`; the targeted form was added in this part (`target=True`).

## 3. Results against `arch` (Gaussian, 10,000 rows, 50 seeds)

Truth: GJR `(0.03, 0.05, 0.10, 0.87)`; EGARCH `(0, 0.15, -0.08, 0.97)`;
GARCH(2,2) `(0.03, 0.05, 0.05, 0.45, 0.42)`. Oracle: `arch_model(y, mean='Zero',
dist='normal', rescale=False, vol='GARCH'/'EGARCH', p, o, q)`, its default
backcast, on the same rows; gaps are in `arch`'s (robust-default) standard
errors. The path gap is `|s_online - s_arch| / s_arch` over rows 9,000 to
9,999, the online value being the run's own one-step prediction (made before
the row's update, parameters still moving), `arch`'s the in-sample path at its
batch fit. Variants: `plain` (unrestricted omega), `huber2` (clipped score,
c = 2), `target` (omega at the running variance), `target_huber2`. Seeds
492000 and up; the grid is `study_492.py grid DIR 50` (resumable, sharded).

| model, variant | seeds within 2 SE (all params / without omega) | worst seed | path gap, last 1000 rows: max, median over seeds / worst seed | max abs persistence gap |
|---|---|---|---|---|
| GJR plain | 49 / 49 | beta +2.10 SE | 3.4% / 11.1% | 0.005 |
| GJR huber2 | 50 / 50 | omega -1.99 | 6.8% / 15.7% | 0.010 |
| GJR target | 45 / 49 | omega -3.21 | 6.6% / 18.4% | 0.012 |
| GJR target_huber2 | 47 / 49 | omega -3.56 | 6.7% / 21.3% | 0.012 |
| EGARCH plain | 46 / 46 | omega +8.91 | 3.4% / 61.4% | 0.014 |
| EGARCH huber2 | 21 / 46 | omega +9.90 | 7.4% / 47.9% | 0.011 |
| GARCH(2,2) plain | 12 / 15 | omega +527 | 140% / 5.1e6% | 0.085 |
| GARCH(2,2) huber2 | 11 / 13 | omega +596 | 120% / 6.0e6% | 0.041 |
| GARCH(2,2) target | 29 / 32 | beta1 +13.4 | 13.8% / 141% | 0.037 |
| GARCH(2,2) target_huber2 | 26 / 30 | beta1 +10.1 | 15.4% / 185% | 0.039 |

(The table is read from the script's `summary` output.)

Reading it:

* **GJR extends.** The plain estimator with the batch start, unrestricted
  omega: 49 of 50 within 2 SE on every parameter, all 50 with the clipped
  score; median-seed path gap 3.4% to 6.8%, worst seed 11% to 16%. The batch
  start converged on 50 of 50 (at most 9 iterations). Targeting `omega` does
  not help GJR here (45 of 50, omega the outlier: `omega` is then a function
  of the other parameters and the running variance, while `arch`'s is free);
  the unrestricted form matches an unrestricted oracle better. The targeted
  form was not compared to a targeted batch fit, which is what the base's
  oracle was; that is not measured.
* **EGARCH extends on most seeds and fails on a few.** 46 of 50 plain; 3 of 50
  batch starts hit the iteration cap (50 iterations), and one seed, 492006, is
  8.9 SE off on `omega` with a path gap of 61%. `omega` is the weak parameter
  (the mean of `log s` times `1 - beta`, with `beta` near 1). The Huber clip
  biases `omega` for EGARCH (21 of 50 within 2 SE, 46 once `omega` is dropped,
  median `omega` gap 2.2 SE): the clipped score's centring is for the variance
  scale, not for EGARCH's log scale. Plain is the better choice. Not derived:
  a score clip fitted to EGARCH.
* **GARCH(2,2) does not extend as built.** Unrestricted, the step leaves the
  admissible region on nearly every row of most seeds (137,000 projected rows
  over 49 seeds) and on seeds 492002 and 492007 settles on `beta_1 = 1`, the
  others 0, a path gap of 1e3 to 5e4. Targeting rescues most of those seeds
  (29 of 50 within 2 SE; median-seed path gap 13.8%) but not all: seed 492000
  is 13 SE from `arch` on `beta_1`, path gap up to 141% at the worst seed.
  Part of that is identification: `arch`'s own fits of the same rows split the
  `beta` mass differently seed to seed (the arch fit of seed 492000 has
  `beta_1 = 0` and `beta_2 = 0.84`, the truth is `(0.45, 0.42)`), so the gap in
  SE is partly `arch`'s own ridge, and the persistence gap, 0.037 at most, is
  the better figure. Open: a parametrisation that steps on the identified
  combinations (the sum of the `beta`s and the filter's roots) and not on
  `beta_1` and `beta_2` separately. Not tried.

## 4. State and per-row cost

State bytes are in 1.4: 208 (GJR), 200 (EGARCH), 312 (GARCH(2,2)), against the
base's 128, each plus the 4 KB buffer until the start. All are bounded by `k`,
`p` and `q`, not by the stream.

Per-row cost in numpy (one row's predict, push and step, Python overhead
included, a 4-core VM, median over seeds, microseconds per row): GJR 41
(unrestricted), 62 (targeted); EGARCH 19; GARCH(2,2) 58 (unrestricted), 65
(targeted). The GARCH forms' figures include Dykstra's projection, which runs
over the half-spaces whenever an iterate leaves the set; a production version
would carry a closed-form projection for each shape. The start (one batch fit
of 500 rows, Python objective) costs 0.3 to 0.4 s for GJR, 0.26 s for EGARCH
and 1.3 to 1.9 s for GARCH(2,2), against `arch`'s whole 10,000-row fit at 0.04
to 0.06 s. These are numpy costs and say nothing about Rust.

## 5. Questions for the coordinator

1. GJR and EGARCH extend with the base's start and step; is the unrestricted
   `omega` (which the base declined to ship for GARCH(1,1)) acceptable for them
   if the batch start is kept, given the measured 49 of 50 and 46 of 50?
2. GARCH(2,2): do we want a reparametrised step (roots or the `beta` sum), as in
   section 3, before part 2?
3. Is `arch`'s unrestricted fit the right oracle for a targeted online form,
   or is a targeted batch fit (as the base used) required? Not run here.
