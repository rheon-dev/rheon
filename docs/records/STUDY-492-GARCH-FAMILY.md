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

---

# Study 492, part 2: Student-t and skew-t innovations (GARCH(1,1), GJR-GARCH(1,1))

Python and numpy only (`scripts/study_492_t.py`, built on part 1's `GarchPOQ`); no Rust, no source
change. The shape parameters (`nu`, and `lambda` for the skew-t) are fitted in the batch start and
then either held or updated online. Oracle: `arch_model(y, mean='Zero', vol='GARCH', dist='t'|'skewt',
rescale=False)`, batch fit on the same rows. Seeds 492100 to 492149, 10,000 rows, truth
GARCH(1,1) `(0.05, 0.08, 0.90)`, GJR `(0.03, 0.05, 0.10, 0.87)`, `nu = 6`, `lambda = -0.2`.

Status: part 2 complete for GARCH(1,1) t, GARCH(1,1) skew-t and GJR t, 10 seeds (the first ten) and
50 seeds both reported. GARCH(2,2) t is wired in (`MODELS['g22']`) but was **not run** here. The
estimator has a failure mode on a few seeds (section 5); it is recorded, not fixed.

## 1. The math

Standardised innovation `z` (mean 0, variance 1), `e_t = sigma_t z_t`, `s_t = sigma_t^2`.

**Student-t (nu > 2).** The standardised density is
`f(z) = c (1 + z^2/(nu-2))^(-(nu+1)/2)`, `c = G((nu+1)/2) / (sqrt(pi (nu-2)) G(nu/2))`
(the scaling `sqrt((nu-2)/nu)` makes the variance 1). Row log-likelihood
`l_t = ln c - (nu+1)/2 ln(1 + z_t^2/(nu-2)) - 1/2 ln s_t`.

Score in the GARCH parameters. With `dz^2/ds = -z^2/s`,

    d l_t / d s_t = w_t / (2 s_t),     w_t = (nu+1) z_t^2 / (nu - 2 + z_t^2) - 1
    g_t(theta)    = 1/2 w_t d ln s_t / d theta

(`w` replaces the Gaussian `z^2 - 1`; it is bounded by `nu`, so a Student-t score is already
clipped by the likelihood, which is why part 1's Huber clip is not used here). Score in `nu`:

    d l_t / d nu = 1/2 [psi((nu+1)/2) - psi(nu/2) - 1/(nu-2)] - 1/2 ln(1 + z^2/(nu-2))
                   + (nu+1) z^2 / (2 (nu-2) (nu-2+z^2))

Checks (`study_492_t.py check`): `E w^2 = 2 nu/(nu+3)` to 3 digits at nu = 4, 6, 12 (4M draws);
shape score against a central difference of `ln f`, 1.6e-10 absolute; the density integrates to
1, mean 0, variance 1 (quad); the log-likelihood at arch's own parameters equals arch's to 2e-13.

**Skew-t (Hansen 1994; arch's `SkewStudent`, parameters `eta = nu`, `lambda` in (-1, 1)).**

    a = 4 lambda c (nu-2)/(nu-1),   b^2 = 1 + 3 lambda^2 - a^2
    f(z) = b c (1 + m^2/(nu-2))^(-(nu+1)/2),   m = (b z + a) / (1 + sg lambda),   sg = sign(z + a/b)

(`a` is the mean and `b^2` the variance correction, so `z` has mean 0 and variance 1; `lambda < 0`
puts the long tail on the left.) The GARCH score has the same form with
`dm/dz = b/(1 + sg lambda)`:

    w_t = (nu+1) m b z / ((1 + sg lambda)(nu - 2 + m^2)) - 1,    g_t(theta) = 1/2 w_t d ln s_t / d theta

This closed form agrees with a central difference of the full likelihood to 1.1e-7 relative
(`check`). **Left open: the skew-t's scores in `nu` and `lambda` are not derived in closed form.**
`a` and `b` depend on `nu` and `lambda` through `c`, and the result was not settled by hand;
they are a central difference (steps `1e-5 nu`, `1e-6`) of the closed-form density. At the truth,
the mean shape score over 400,000 draws is `(2.5e-5, -9.0e-4)` with SEs `(5.8e-5, 1.2e-3)`, i.e. zero
within noise, and the full-vector score matches a central difference of the likelihood (a weaker
check for the shape rows, which compare a difference to the same difference). A production form
would carry the closed forms.

**The batch start.** The first 500 rows are buffered; `batch_core.maximize` (part 1's projected
Newton, BHHH information) maximises the likelihood jointly in `(theta, shape)` from a small grid
(shape starts `nu` in {5, 10} and, for the skew-t, `(7, -0.2)`), with `nu` in `[2.05, 300]` and
`|lambda| < 0.99` (arch's bounds). Fitting `nu` on the start's standardised residuals separately
was not tried; the joint fit is the same Newton step in one more dimension.

**The online step.** Two modes, both measured:

* `online`: the shape is stepped with the GARCH parameters, `x = (theta, shape)`,
  `H <- H + gamma (s s' - H)`, `x <- Proj(x + gamma H^-1 s)`, `gamma = 1/(t+500)`, `s` the whole
  per-row score (BHHH, the running mean of its outer product; the Gaussian part's expected
  information is not used because the shape rows have none that is as cheap).
* `fixed`: `nu` (and `lambda`) held at the start's values.

Stability: held shape does not work (section 3): a 500-row estimate of `nu` has an SE near 1, and the
miss sets the tail weight of every later score. The online step tracks arch (median `nu` gap 0.1 on
an arch SE of 0.35) but is not safe everywhere (section 5).

**State.** Floats `k + m + nf(nf+1)/2 + max(p,o) + o + q + q k + 3`, `k` GARCH parameters, `m`
shape parameters (1 or 2), `nf` the stepped parameters (`k` or `k-1` if targeted, plus `m` if the
shape is online), the 3 scalars being the running mean of `e^2`, its count and the step count.
Bounded by the shape, not the stream, plus the 500-row buffer (4 KB) until the start. Bytes = 8 x:

| model | omega | shape | floats | bytes | vs Gaussian (part 1) |
|---|---|---|---|---|---|
| GARCH(1,1) t | free | online | 22 | 176 | 128 (base) |
| GARCH(1,1) t | free | fixed | 18 | 144 | |
| GARCH(1,1) t | targeted | online | 18 | 144 | |
| GARCH(1,1) skew-t | free | online | 28 | 224 | |
| GARCH(1,1) skew-t | targeted | online | 23 | 184 | |
| GJR(1,1) t | free | online | 30 | 240 | 208 |
| GJR(1,1) t | targeted | online | 25 | 200 | |

## 2. Verification of the work-in-progress file (step 0)

`scripts/study_492_t.py` was left untracked by an earlier agent, unverified. Verified, then kept:

* It imports part 1's `GarchPOQ`, `backcast` (so inherits the 8e-16 filter match to arch and the GJR
  `0.5 * backcast` pre-sample term) and the base's `batch_core`.
* `check` passes (section 1): scores against central differences, log-likelihood against arch's at
  arch's parameters (2e-13 t, 5e-13 skew-t), densities normalised, `E w^2` identity.
* A smoke case (`one g11 skewt 492100`) ran: batch start converged, all four variants within 2 SE.
* The earlier agent's cache of 199 cases (`.cache/study492t`, version of the script unknown, includes
  GARCH(2,2)) was **not used**; the grid below was rerun from scratch with this file.
* One thing I tried and removed: a per-row cap on the shape's move plus `nu` in `[3, 100]` online. It
  did not fix the failures of section 5 (the free-`omega` runs left the stable region regardless), so the
  file is as it was verified.

## 3. Results against arch (10,000 rows)

Gap in arch's (robust-default) standard errors on `(omega, alpha[, gamma], beta, shape)`; path gap
`|s_online - s_arch|/s_arch` over rows 9,000 to 9,999 (online: the run's one-step prediction made
before the update; arch: its in-sample path). "free" is `plain_*` (unrestricted `omega`), "targeted" is
`target_*` (`omega` at the running variance). Seeds within 2 SE on every parameter, `omega` and the
shape included.

| model, variant | 10 seeds | 50 seeds | median shape gap to arch (arch SE) | path gap, max over last 1000: median / 90th pct (50 seeds) | |persistence gap| 90th pct |
|---|---|---|---|---|---|
| GARCH(1,1) t, free, online | 10 | 46 | nu 0.095 (0.35) | 3.5% / 11.5% | 0.004 |
| GARCH(1,1) t, targeted, online | 9 | 43 | nu 0.118 | 5.5% / 19.7% | 0.007 |
| GARCH(1,1) t, free, fixed nu | 4 | 19 | nu 0.88 | 6.0% / n/a | |
| GARCH(1,1) skew-t, free, online | 9 | 43 | eta 0.12 (0.34), lambda 0.002 (0.013) | 4.1% / 16.8% | 0.004 |
| GARCH(1,1) skew-t, targeted, online | 10 | 44 | eta 0.21, lambda 0.002 | 6.8% / 17.8% | 0.005 |
| GARCH(1,1) skew-t, free, fixed shape | 2 | 10 | eta 0.85, lambda 0.031 | 6.5% / n/a | |
| GJR t, free, online | 10 | 48 | nu 0.100 (0.35) | 5.5% / 8.8% | 0.004 |
| GJR t, targeted, online | 9 | 43 | nu 0.149 | 7.3% / 20.5% | 0.010 |
| GJR t, free, fixed nu | 4 | 18 | nu 0.88 | 7.0% / n/a | |

On the first 10 seeds the free online form is at or near arch: GARCH(1,1) t 10 of 10, skew-t 9 of 10
(seed 492108, `alpha` 4.4 SE, a batch start that hit the iteration cap), GJR t 10 of 10; median path
gap 3.8%, 3.6% and 5.8%; worst path gap 9% (t), 26% (skew-t), 8% (GJR t); persistence gap at most
0.005. Over 50 seeds the counts fall to 46, 43 and 48; the median seed stays near arch, the loss is a
tail of seeds (section 5), not a general drift. Fixing the shape at the start's value costs most
seeds (19, 10, 18 of 50): the start's `nu` from 500 rows misses arch's by a median 0.9 (3 SE) and
`lambda` by 0.03 (2 SE), and the log-likelihood gap is -3.9 (t) and -10.6 (skew-t) against -0.4 and
-0.5 for the online shape.

The `ll gap` is the log-likelihood of the online parameters on the full 10,000 rows minus arch's
maximum, a median of -0.4 to -1.4 (nats): the online fit is within a nat or so of the batch maximum
on the median seed.

## 4. State and per-row cost

State bytes in section 1. Per-row cost in numpy (predict, push, step, Python overhead included; 4-core
VM; median over 50 seeds; microseconds per row): GARCH(1,1) t 55 (free), 75 (targeted); skew-t 106
(free), 127 (targeted), which includes the four difference evaluations of the density for the shape
scores; GJR t 58 (free), 75 (targeted); fixed shape 52 to 54 for t, 101 for skew-t. The batch start
(one joint fit of 500 rows) costs 0.2 s (t), 0.3 s (skew-t, up to the 50-iteration cap on one seed),
0.34 s (GJR t); arch's whole 10,000-row fit 0.03 to 0.06 s. These are numpy costs and say nothing
about Rust.

## 5. Where it fails

Counting the seeds out of the 2 SE band, by cause (50 seeds, free, online):

* **A step that leaves the stable region, GARCH(1,1) t seeds 492111 and 492134** (and 492111 for GJR t):
  with unrestricted `omega` the step takes `omega` to 33 to 60, `beta` to 0 and `nu` to 3 or 66 to 290,
  with 9,000 projected rows; the path gap exceeds 1,000%. These are the two cases that show up in the
  "worst seed" columns of the 50-seed summary. The targeted form on the same seeds does not blow up
  but is 7 to 9 SE off on `nu` (492111: `nu` 2.9 against arch's 6.3) or on `omega`. A heavy-tailed run of
  outliers shifts the `nu` estimate and the free parameters together through the BHHH matrix.
  Not derived: a step that cannot do that; the cap on the shape move (removed, section 2) did not
  help. Candidates not tried: a `log(nu - 2)` or `1/nu` parametrisation, a scale of `H` fixed after the
  start, or holding `nu` fixed after a larger start.
* **The skew-t's iteration-capped batch start** (492108, 50 iterations) and a few moderate misses
  (2 to 10 SE; the worst is 492144, `alpha` 10.5 SE with 87 projected rows): the cause was not
  isolated. A weakly identified `omega` and `alpha` on 500 rows is the suspect, as in part 1; not tested.
* Targeting `omega` does not improve the within-2 SE count here (43 to 44 of 50): `omega` is
  then a function of the other parameters and the running variance while arch's is free.

## 6. Questions for the coordinator

1. The free-`omega` online form is the better match to arch on the median seed but blows up on 2
   of 50 GARCH(1,1) t seeds; the targeted form is safer but 43 of 50. Which is wanted?
2. Shape update in the online step needs a stabiliser (section 5). Is a `log(nu-2)` step worth a
   part 3 slot, or should `nu` be held after a start long enough that its SE is small (the start
   would have to be several thousand rows; the base's start is 500)?
3. The skew-t shape scores need a closed form before any Rust port.
