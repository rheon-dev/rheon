# Study 492, part 1: online GJR-GARCH, EGARCH and GARCH(2,2), Gaussian

Python and numpy only (`scripts/study_492.py garch`); no Rust, no source change. The
question: can the online GARCH estimator of study 485 (requests A9 to A12:
variance-targeted GARCH(1,1), a 500-row batch start by a small projected-Newton
optimizer, a Huber-clipped score) extend to GJR-GARCH (leverage), EGARCH and
GARCH(2,2), and how near do the extensions stay to `arch`'s batch fits on the
same rows? This part is Gaussian innovations. Student-t, skew-t and FIGARCH
come in parts 2 and 3.

Status: part 1 complete for the three models at 10,000 rows: 50 seeds (492000 to 492049) for GJR, EGARCH and GARCH(2,2) (part 3 read the finished 50th GARCH(2,2) seed: the counts are unchanged, 12 and 15 within 2 SE plain, 29 and 32 targeted; the tables below keep part 1's text). Open:
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

Python and numpy only (`scripts/study_492.py t`, built on part 1's `GarchPOQ`; it was `scripts/study_492_t.py` until part 3 merged the two scripts); no Rust, no source
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

Checks (`study_492.py check t`): `E w^2 = 2 nu/(nu+3)` to 3 digits at nu = 4, 6, 12 (4M draws);
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

`scripts/study_492_t.py` (now merged into `scripts/study_492.py`, part 3) was left untracked by an earlier agent, unverified. Verified, then kept:

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

---

# Study 492, part 3: FIGARCH with a capped truncation, the state cap, one script, tolerances

Python and numpy only; one entry point now, `scripts/study_492.py` (the part-2 script is merged into it and
deleted). No Rust, no source change. Oracle: arch 9.0.0. Seeds: FIGARCH 492200 to 492219 (20 seeds; the first 10
are reported beside the 20), Gaussian GARCH(1,1) and GJR 492000 to 492049 (rerun with the merged script), the t
and skew-t figures are part 2's (the merged script reproduces seed 492111 bit for bit).

```
python scripts/study_492.py --quick            # every family, 2 to 3 seeds on 3000 rows, smoke bounds: 17 to 20 s
python scripts/study_492.py figarch            # 10 seeds at 10,000 rows (the grid below ran 20); garch, t: 50
python scripts/study_492.py grid figarch DIR --seeds 20     # resumable; SHARD and NSHARDS spread it over cores
python scripts/study_492.py summary figarch DIR              # the tables below, with the tolerance verdicts
python scripts/study_492.py one figarch 492201 --ms 500,1000  # one case, printed
```

## 1. The FIGARCH form

**Which FIGARCH arch fits.** `arch.univariate.FIGARCH(p=1, q=1, truncation=1000)`: the fractionally integrated
GARCH(1,1) written as an ARCH(infinity) in the squared residual and truncated at a fixed lag `M`, with the
pre-sample squared residuals all equal to arch's backcast (the exponentially weighted mean of the first 75 squared
rows, `0.94^i` weights). arch's documentation names no source for it; it is the fractionally integrated recursion
`h_t = omega + [1 - beta L - (1 - phi L)(1 - L)^d] e_t^2 + beta h_{t-1}` solved for the weights, then cut at `M`
(the fixed-truncation estimation form). arch's own `figarch_weights`, filter and bounds are what the script
compares with, and its default `M` is 1000.

    h_t = omega/(1 - beta) + sum_{j=1}^{M} lam_j e_{t-j}^2          (e_s^2 = backcast for s < 0)
    lam_1 = phi - beta + d,   delta_1 = d,
    delta_j = (j - 1 - d)/j delta_{j-1},   lam_j = beta lam_{j-1} + delta_j - phi delta_{j-1}

Constraints (arch's, which keep every `lam_j >= 0`): `omega > 0`, `phi >= 0`, `0 <= d <= 1`, `2 phi + d <= 1`,
`0 <= beta <= phi + d`; the script adds `d <= 1 - 1e-4` and an `omega` floor, as the GARCH forms do. The weights
sum to less than 1 and the untruncated sum is exactly 1 (the process is integrated: strictly stationary, with no
finite variance), so the truncated model's long-run variance, `omega/(1 - beta)/(1 - sum_{j<=M} lam_j)`, is set
by the tail mass `1 - sum_{j<=M} lam_j ~ M^-d`. That is why `M` is part of the model and why a short `M` moves
`omega` (section 4).

**The online step, state and sensitivity.** The state is a ring of the last `M` squared residuals plus the usual
`theta` (4) and `H` (4 x 4, symmetric). Unlike the GARCH forms, the sensitivity of `h_t` is not a finite
recursion; it is a dot product of the ring with the weight derivatives, at the current `theta`:

    dh/domega = 1/(1 - beta)
    dh/dphi, dh/dd = dlam_phi . ring,  dlam_d . ring
    dh/dbeta = omega/(1 - beta)^2 + dlam_beta . ring

The weights and their derivatives are not state: they are a pure function of `theta` and one O(M) pass produces
them. With `P_j = prod_{i=2}^{j} (i - 1 - d)/i` (`P_1 = 1`), `delta_j = d P_j` (no division by `d`), and
`u_1 = d - beta + phi`, `u_j = delta_j - phi delta_{j-1}`: `lam` is the linear recursion `lam_j = beta lam_{j-1} +
u_j`, and its three derivatives are the same recursion driven by `du/dphi = (1, -delta_1, ..., -delta_{M-1})`,
`du/dd = (1, d delta_j/dd - phi d delta_{j-1}/dd)` with `d delta_j/dd = P_j + d dP_j`, `dP_j = -P_j sum_{i=2}^{j}
1/(i - 1 - d)`, and `du/dbeta = (-1, lam_1, ..., lam_{M-1})`. A Rust pass runs the four recursions in lock step
along `j` against the ring with O(1) scratch; the numpy script materialises four M-vectors per row.

The step is part 1's: `g = (z^2 - 1)/(2 h) dh`, `H <- H + gamma (dh dh'/(2 h^2) - H)`, `theta <- Proj(theta +
gamma H^-1 g)`, `gamma = 1/(t + 500)`, the first 500 rows buffered and fitted by `batch_core.maximize` from arch's
own starting grid (`d` in .2, .5, .7; `phi`, `beta` as fractions of their bounds; `omega` from the variance
target of the truncated weights), the buffer replayed at that fit to build `H`.

**Two projections of the step.** `euclid` is part 1's Euclidean projection onto the constraint set. `metric`
projects in the metric of the step's own `H` (`argmin (y - x)' H (y - x)` over the set: Dykstra's algorithm in the
coordinates `u = L' y`, `H = L L'`). The Euclidean projection of a Newton step is not an ascent direction of the
likelihood when `H` is far from the identity: the constraints bind on `phi`, `d` and `beta`, which the likelihood
barely separates, and a cut `phi` leaves `d` and `omega` with moves computed for an uncut `phi`. Seed 492201 shows it
(section 3).

## 2. Verification (`study_492.py check figarch`, seconds)

* The weights against arch's `figarch_weights`: 1.7e-18 absolute at M = 1000.
* The analytic weight derivatives against central differences: 1.6e-10 (phi), 1.2e-9 (d), 9.3e-11 (beta), relative.
* The fixed-parameter filter against arch's conditional variance at arch's own fitted parameters, every row of
  3000: 5.3e-15 relative (so the backcast handling, including the lags that reach before the sample, is arch's).
* The likelihood's score against a central difference of the likelihood: 5.6e-9 relative.

## 3. Results against arch (Gaussian, 10,000 rows, 20 seeds, truth `(omega, phi, d, beta) = (0.03, 0.2, 0.4, 0.5)`)

Data come from the truncation-1000 process (arch's default), by arch's recursion. arch is fitted at the same
`M` as the online form ("same M") and at `M = 1000` (the reference). The gap is in arch's (robust-default)
standard errors, as in parts 1 and 2; the path gap is `|s_online - s_arch|/s_arch` over rows 9,000 to 9,999.
arch's median standard errors at M = 1000: `(0.007, 0.039, 0.033, 0.052)`; its fits converged on 20 of 20 at
every `M` (0.1 s at M = 100, 0.3 s at M = 1000).

| online form, M | vs arch at | seeds within 2 SE, every parameter (first 10 / 20) | median abs gap in SE: omega, phi, d, beta | path gap, max over the last 1000 rows: median / worst seed | log-lik gap: median / worst |
|---|---|---|---|---|---|
| euclid, 100 | M = 100 | 7 / 14 | 0.75, 0.52, 0.85, 0.74 | 8.3% / 1800% | -1.65 / -6439 |
| euclid, 500 | M = 500 | 8 / 16 | 0.40, 0.39, 0.58, 0.43 | 5.5% / 1480% | -0.61 / -5281 |
| euclid, 1000 | M = 1000 | 8 / 16 | 0.41, 0.47, 0.44, 0.53 | 5.9% / 667% | -0.64 / -3541 |
| metric, 100 | M = 100 | 8 / 18 | 0.53, 0.39, 0.83, 0.59 | 6.6% / 19.0% | -0.90 / -9.3 |
| metric, 100 | M = 1000 | 1 / 1 | 4.72, 0.34, 0.89, 0.86 | 11.6% / 22.5% | -6.64 / -23.3 |
| metric, 500 | M = 500 | 8 / 17 | 0.48, 0.28, 0.57, 0.57 | 5.0% / 15.4% | -0.43 / -10.5 |
| metric, 500 | M = 1000 | 8 / 15 | 1.66, 0.28, 0.57, 0.58 | 5.5% / 15.6% | -0.95 / -9.4 |
| **metric, 1000** | M = 1000 | **8 / 17** | 0.42, 0.32, 0.47, 0.56 | **5.0% / 16.8%** | **-0.33 / -11.7** |

* **FIGARCH extends, with the metric projection.** At M = 1000, 17 of 20 seeds are within 2 SE of arch on all four
  parameters, the median absolute gaps are at most 0.56 SE (omega 0.0029, phi 0.013, d 0.017, beta 0.027 in
  value), the median seed's variance path is 5.0% from arch's and the worst seed's 16.8%, and the likelihood is
  0.33 nats below arch's maximum on the median seed (a nat or so, as in part 2). No seed leaves the region.
* **The Euclidean projection blows up on 1 seed in 20.** Seed 492201: the 500-row start lands at `(0.18, 0.045,
  0.178, 0)`; from row 600 the step is projected on nearly every row (8,658 of 9,500) and ends at `(2.11, 0, 1,
  0)`, the likelihood 3,541 nats below arch's, a variance path 667% from arch's. It is part 1's step and the
  mechanism of part 2's two blown-up t seeds (section 5 of part 2); the metric projection on the same seed ends at
  `(0.042, 0.171, 0.308, 0.346)`, 4.7 nats below arch's maximum. The metric form still misses 2 SE on
  three seeds (492201, 492203, 492213), by 2.5 to 4.0 SE at the worst parameter, `omega` high and `beta` low in
  each, with starts that took 12 to 17 iterations (492203: "no ascent" at 17).
* **The online estimate is biased down in `d` by about one SE and slightly in `beta`.** Median online
  `(omega, phi, d, beta)` at M = 1000, metric: `(0.036, 0.193, 0.365, 0.480)` against arch's mean `(0.032,
  0.205, 0.400, 0.500)`. A 500-row start with M = 1000 sees almost no row whose ring is not mostly the backcast
  (a lag beyond `t` carries `lam_j x backcast`), so `d` is identified by the stream, at `1/(t + 500)`.
  Not tried: a longer start; fitting `H` by a second pass.
* **A short `M` is a different model and the gap says so.** M = 100 matches arch at M = 100 as well as M = 1000
  does at its own (18 of 20), but against the M = 1000 reference it is 1 of 20, `omega` 4.7 SE high (arch's own
  M = 100 fit has mean `omega` 0.062 against 0.032): the shorter truncation loses tail mass and `omega` compensates.
  M = 500 is 15 of 20 against the M = 1000 reference, the median `omega` gap 1.7 SE.

Cost (numpy, 4-core VM, median over seeds): 103, 119 and 141 microseconds per row at M = 100, 500 and 1000
(Python overhead is about 100; the slope is about 0.04 microseconds per unit of `M`), the start 0.13 s.
Extrapolation was measured separately: 261 microseconds at M = 4000, 1.2 ms at 16,000, 2.6 ms at 32,751.

## 4. Truncation length against accuracy

Two measurements, both at the truth `(0.03, 0.2, 0.4, 0.5)`:

**The filter's truncation error alone** (fixed parameters, data from a truncation-32,000 process, relative gap
of the `M`-filter to the 32,000-filter's variance over the last 1000 rows; three seeds, range): tail mass
`1 - sum_{j<=M} lam_j` and the gap (the reference is itself cut: its own tail mass is 1.7%, so each gap below
understates the gap to the untruncated process).

| M | state bytes | % of the 256 KiB cap | tail mass | median gap | max gap |
|---|---|---|---|---|---|
| 100 | 936 | 0.4% | 17.1% | 16% to 25% | 43% to 54% |
| 500 | 4,136 | 1.6% | 9.0% | 6.5% to 14% | 17% to 32% |
| 1000 | 8,136 | 3.1% | 6.8% | 4.6% to 9.4% | 12% to 23% |
| 4000 | 32,136 | 12.3% | 3.9% | 1.8% to 3.6% | 4.5% to 8.5% |
| 16,000 | 128,136 | 48.9% | 2.2% | 0.3% to 0.65% | 0.8% to 1.6% |
| 32,000 | 256,136 | 97.7% | 1.7% | (reference) | (reference) |

The tail mass falls like `M^-d` with `d = 0.4`, a factor 3 per factor 16 in `M`; the gap to the 32,000-filter
falls faster (a factor 14 from M = 1000 to 16,000) because the reference holds the weights out to 32,000, so the
rows near the bottom of the table are the part between `M` and 32,000, not a distance to the untruncated process.
At n = 10,000 rows every `M` above the sample length reads the backcast for the whole pre-sample, so the 16,000 and
32,000 rows also partly measure the backcast treatment.

**The fit when the data are longer-memoried than the filter** (6 seeds, data from a truncation-4000 process,
metric projection, arch fitted at the same `M`): arch's own mean `omega` is 0.068 (M = 500), 0.054 (M = 1000) and
0.035 (M = 4000) against the truth 0.03; `phi`, `d` and `beta` barely move (`d` 0.412, 0.414, 0.417). The online
form at M = 1000 is within 2 SE of arch at M = 1000 on 4 of 6 seeds, within 2 SE of arch at M = 4000 on 2 of 6
(`omega` 7 SE high on the worst), against 4 of 6 for the online form at M = 4000 and arch at M = 4000: the
estimator is as good as arch at the same truncation, and a truncation shorter than the data's costs `omega`, not
the dynamics.

**Chosen M: 1000.** It is arch's default and the oracle's truncation (so the comparison is like for like), it fits
17 of 20 seeds, and it costs 8,136 bytes. It is not the accurate `M`: for the variance path to be within about
5% of the long-memory process on the median row the truncation must be about 4000 (a 2 to 3.5% median gap), and
within about 1% about 16,000. Both are inside the cap (section 5); the cost that binds is the per-row pass, linear
in `M`.

## 5. The state cap

The coordinator's rule: every model's data-dependent state needs a stated cap, 256 KB per group by default. Here
256 KB is read as 256 KiB = 262,144 bytes; at 256,000 bytes the numbers below move by under 2%.

What FIGARCH stores: the ring of `M` squared residuals (the newest first), `theta` (4 floats), `H` (4 x 4
symmetric, 10), the running mean of `e^2`, its count and the step count (3):

    floats = M + 17,    bytes = 8 M + 136

Before the start the 500-row buffer (4,000 bytes) is the only state (the ring is built when the buffer is
replayed), so the peak is the larger of the two, not the sum. The weights and their derivatives are not stored
(one pass per row, O(1) scratch in a Rust port). A port that cached them to avoid the pass would store five
M-vectors (`40 M` bytes) and reach the cap at `M = 6,553`; that is the layout to avoid.

**The cap is reached at `M = 32,751`**: `8 M + 136 <= 262,144` gives `M <= 32,751`, which is 262,144 bytes exactly (at
256,000 bytes, `M <= 31,983`). The values proposed: M = 1000 (8,136 bytes, 3.1% of the cap, the oracle's
truncation), M = 4000 (32,136 bytes, 12.3%) for a 5% variance-path accuracy against the long-memory process, and
M = 16,000 (128,136 bytes, 48.9%) for a 1% accuracy. **Every M that the accuracy measurements call for stays within
the cap, plainly: the cap does not bind before accuracy does.** What binds first is the pass: about 30 M floating-point
operations per row (four recursions of two operations, four dot products, the cumulative product and sum) and
2.6 ms per row in numpy at the cap's largest M, against 0.14 ms at M = 1000. The bytes of the other models are
at most 312 (section 6), 0.1% of the cap. A bank holding FIGARCH for many groups pays `8 M + 136` bytes and `O(M)` time per group.

## 6. Tolerances

Each is a count or a bound from the measured distribution over seeds; a bound on a gap is 1.5 times the measured
median or worst, rounded up to a clean figure. They are in `TOL` in `scripts/study_492.py`, applied by
`summary` and by the full run; `--quick` applies only smoke bounds (a few seeds on short series, structural, so it
cannot flake on a 2 SE count: path gap median under 100%, shape gap median under 3 SE, FIGARCH state equals
`M + 17`).

| family | quantity (unit) | measured | bound | tight or loose |
|---|---|---|---|---|
| Gaussian GARCH(1,1), 50 seeds | seeds within 2 SE of arch on every parameter (fraction) | 49/50 | >= 0.90 (45) | tight: one seed of slack in 4 |
| | path gap, max over the last 1000 rows, median over seeds (relative) | 0.025 | <= 0.05 | tight |
| | same, worst seed | 0.363 | <= 0.6 | **loose**: one seed, 492026, `omega` 3.7 SE high (the next worst is 0.105, the 98th percentile 0.11) |
| GJR(1,1), 50 seeds | within 2 SE | 49/50 | >= 0.90 | tight |
| | path gap median / worst | 0.034 / 0.111 | <= 0.05 / <= 0.2 | tight / loose by the 1.5 factor |
| EGARCH(1,1), 50 seeds | within 2 SE | 46/50 | >= 0.90 (45) | **one seed of slack**; three batch starts hit the 50-iteration cap |
| | path gap median | 0.034 | <= 0.05 | tight; no bound on the worst (0.61, seed 492006, `omega` 8.9 SE) |
| GARCH(2,2), 50 seeds | none | 12/50 plain, 29/50 targeted within 2 SE | **no pass tolerance** | the failure is the expected result: `beta_1` and `beta_2` are weakly identified (arch's own fits split their sum differently seed to seed), the persistence gap is the better figure (0.085 plain, 0.037 targeted) |
| t, skew-t, GJR t, 50 seeds | within 2 SE, every parameter (fraction) | 46, 43, 48 of 50 | >= 0.80 (40) | loose: 43/50 is the worst, 6 seeds of slack |
| | shape within 2 SE (fraction) | 47, 46, 49 of 50 | >= 0.85 (43) | loose |
| | shape gap, median over seeds (arch SE; in value: nu 0.095 to 0.12, eta 0.12, lambda 0.002) | 0.25 (nu), 0.34 and 0.12 (eta, lambda), 0.29 | <= 0.5 | tight |
| | path gap, median over seeds of the max | 0.035, 0.041, 0.055 | <= 0.10 | tight |
| | same, 90th percentile over seeds | 0.115, 0.168, 0.088 | <= 0.25 | loose; **no bound on the worst**: two GARCH(1,1) t seeds (492111, 492134) blow up (7,880% and 4,580%), GJR t 492111 is off by 99% |
| FIGARCH, M = 1000, metric, 20 seeds | within 2 SE of arch at the same M | 17/20 | >= 0.75 (15) | tight |
| | parameter gap, median over seeds, each parameter (arch SE) | 0.42, 0.32, 0.47, 0.56 | <= 1.0 | loose |
| | path gap median / worst | 0.050 / 0.168 | <= 0.10 / <= 0.25 | tight / loose |
| | log-likelihood gap, worst seed | -11.7 | > -50 | loose: it separates the blow-up (-3541) from the rest |

The Student-t and skew-t counts and the shape gap are the loosest because the shape parameter's standard
error is wide (0.35 for `nu` at 10,000 rows), the 500-row start misses it by a median 0.9 (section 3 of part 2),
and a heavy-tailed run moves the shape and the GARCH parameters together. The Euclidean-projection FIGARCH is
not given a tolerance: it is the baseline whose 1 in 20 blow-up the metric projection removes.

## 7. The family, summarised

### Recursions (`l_t = log s_t`, `s_t = sigma_t^2`, `z_t = e_t/sigma_t`)

| model | conditional variance | sensitivity carried | step |
|---|---|---|---|
| GARCH(p, q), GJR(p, o, q) | `s_t = omega + sum a_i e_{t-i}^2 + sum g_j 1[e_{t-j} < 0] e_{t-j}^2 + sum b_k s_{t-k}` (GJR's pre-sample asymmetric term is `0.5 x backcast`) | `d_t = x_t + sum b_k d_{t-k}`, `dl = d_t/s_t` | `g = (z^2 - 1)/2 dl`, `I = dl dl'/2`; Gaussian quasi-likelihood |
| EGARCH(1,1,1) | `l_t = omega + beta l_{t-1} + alpha (abs z_{t-1} - sqrt(2/pi)) + gamma z_{t-1}` | `dl_t = (1, abs z - c, z, l_{t-1}) + (beta - (alpha abs z + gamma z)/2) dl_{t-1}` | same; no variance target |
| t, skew-t on the above | the same `s_t` | the same | `g = w dl/2`, `w = (nu + 1) z^2/(nu - 2 + z^2) - 1` (t), the skew-t's `w` of part 2; shape stepped with BHHH information |
| FIGARCH(1,1), truncation M | `h_t = omega/(1 - beta) + sum_{j<=M} lam_j e_{t-j}^2` | `dh` by the ring and the weight derivatives (section 1) | `g = (z^2 - 1)/(2h) dh`, `I = dh dh'/(2h^2)`, metric projection |

All: `H <- H + gamma (I - H)`, `theta <- Proj(theta + gamma H^-1 g)`, `gamma = 1/(t + 500)`, a 500-row batch start,
out-of-sample predictions (the row's prediction is made before its update).

### State bytes, as formulas

`k = 1 + p + o + q`; `m` shape parameters (1 for t, 2 for skew-t); `nf` the stepped parameters.

| model | floats | bytes (example) |
|---|---|---|
| GARCH(p,o,q) Gaussian | `k + k(k+1)/2 + max(p,o) + o + q + q k + 5` | GARCH(1,1) 152, GJR 208, GARCH(2,2) 312 |
| EGARCH(1,1,1) | `k + k(k+1)/2 + 1 + 1 + k + 5`, `k = 4` | 200 |
| t / skew-t | `k + m + nf(nf+1)/2 + max(p,o) + o + q + q k + 3` | GARCH(1,1) t 176, skew-t 224, GJR t 240 |
| FIGARCH(1,1) | `M + 4 + 10 + 3` | `8 M + 136`: 8,136 at M = 1000 |

Every one adds the 500-row buffer (4,000 bytes) until the start. Against the 256 KiB cap (262,144 bytes) every
model but FIGARCH is under 0.1%; FIGARCH reaches it at `M = 32,751`.

### Cost per row

Floating-point operations: GARCH-type forms are `O(k^2 + q k + k^3/3)` (the solve) with `k <= 5`, and the
projection only when an iterate leaves the set (Dykstra over about `k + 2` half-spaces, hundreds of sweeps in the worst
case; a production form wants a closed-form projection per shape); the t adds two digamma values and a
`log1p` per row, the skew-t four more density evaluations for the shape scores (central differences) or the
closed forms; FIGARCH is about `30 M`. Measured in numpy, with Python overhead (4-core VM, median over seeds, microseconds per
row): GJR 41 (free), 62 (targeted); EGARCH 19; GARCH(2,2) 58 and 65; GARCH(1,1) t 55 (free), skew-t 106; GJR t 58;
FIGARCH 103, 119, 141 and 261 at M = 100, 500, 1000 and 4000. These say nothing about Rust.

### Numerical risks

* **The two blown-up t seeds, 492111 and 492134 (GARCH(1,1) t, free `omega`).** The step takes `omega` to 33 to 60,
  `beta` to 0 and `nu` to 67 or 293, with 9,200 to 9,500 projected rows, a likelihood 9,900 to 12,600 nats below arch's.
  Part 3 found the mechanism: it is the Euclidean projection of a Newton step, as in FIGARCH seed 492201. Projecting
  in the step's own metric (`H`, with the shape eliminated by the Schur complement) removes the blow-up on both
  (likelihood gap -190 and -151 nats, `omega` 0.19 and 0.14), but leaves them off: `nu` 2.7 and 2.9 against arch's
  6.3. Over seeds 492100 to 492119 the metric form changes nothing where the projection was inactive (17 of 20
  seeds within 2 SE with the Euclidean projection, 18 with the metric one) and the worst likelihood gap goes from -12,644 to -190.
  Spot check on one family only (GARCH(1,1) t, 20 seeds); not run on skew-t, GJR t or the 50-seed grid. The shape
  stabiliser of part 2's question 2 (a `log(nu - 2)` step, or holding `nu`) is still open.
* **The GJR pre-sample term.** arch reads the pre-sample asymmetric term as `0.5 x backcast`; the script read it as
  zero, which put the first rows' variance 5.7% off, decaying. Fixed in part 1; the filter now matches arch at its own
  parameters to 8e-16 on every row, all three Gaussian models, and to 5.3e-15 for FIGARCH.
* **The EGARCH iteration cap.** The batch start hit its 50-iteration cap on 3 of 50 seeds; one of them, 492006, is
  `omega` 8.9 SE off with a 61% path gap. `omega` times `1 - beta` is the mean of `log s` and `beta` is near 1: the
  ridge. The Huber clip biases EGARCH (21 of 50 within 2 SE): plain is the form.
* **FIGARCH's start.** 500 rows against `M = 1000` give a start whose ring is mostly backcast; seed 492201's start
  `(0.18, 0.045, 0.178, 0)` is a local optimum, 492203 stops with "no ascent". The metric projection recovers the
  first; the second ends 11.7 nats below arch's maximum and 4 SE off on `beta`.
* **Libm.** FIGARCH uses no libm function in the weights (a cumulative product, a cumulative sum and divisions);
  the GARCH forms use none either; the t and skew-t use `log1p`, `digamma` and `lgamma`, so their `theta` is a libm
  field in the sense of the cross-OS rule: continue to the bit on one OS, a tolerance across OSes. Not measured here.
* **Variance target.** None for EGARCH (see 1.3 of part 1) or FIGARCH (the untruncated variance is infinite; the
  truncated one is a function of `M`).

### Oracles

arch 9.0.0 throughout: `arch_model(..., mean='Zero', dist='normal'|'t'|'skewt', vol='GARCH'|'EGARCH', rescale=False)`
and `arch.univariate.FIGARCH(p=1, q=1, truncation=M)` in `ZeroMean`, each fitted by arch's default optimizer on the
same rows; arch's own filter, weights and backcast are compared to ours at arch's fitted parameters (8e-16 to 5.3e-15
relative, every row). The `check` subcommands compare each analytic score to a central difference of the
log-likelihood. scipy is used for the special functions and `lfilter`.

### Open derivations (not settled by hand or by a test)

1. **The skew-t's closed-form score in `nu` and `lambda`**: closed in part 4 (section 1 there), verified against
   central differences; the script now uses it.
2. **The EGARCH variance target**: a log-variance target is derived in part 4 (section 2), exact under the normal;
   it does not beat the free `omega` against arch. Still open: a score clip fitted to EGARCH's log scale (the
   variance-scale Huber centring biases `omega`), and the online step's stability on a start that hit the
   iteration cap (seed 492006, part 4).
3. **A root-based step for GARCH(2,2)**: step on the sum of the `beta`s and the filter's roots, not on `beta_1` and
   `beta_2` separately; not tried. GARCH(2,2) carries no pass tolerance until it is.
4. Also open: a shape stabiliser for the t's online `nu` after the metric projection (section 7, risks); and the
   FIGARCH weight-derivative pass as one fused loop with O(1) scratch (derived and verified here in vector form).

## 8. Notes for the merge

* `scripts/study_492_t.py` is deleted; its code lives in `scripts/study_492.py` under the `t` subcommand, names
  suffixed `_t` where they would collide (`MODELS` became `T_MODELS` for the t family; the part-1 `compare`,
  `check`, `grid` and `summary` became `compare_garch` and so on, and one resumable `grid` and `summary` serve the
  three families). A record written by the old `grid` has the same fields and is read by the new `summary`.
* The script gained the metric projection (`FigOnline(metric=True)`, `OnlineT(metric=True)`, both off by default
  for the GARCH and t forms, so part 1 and 2's results are unchanged) and one file-level lint exemption. The
  repository's `ruff check .` and `ruff format --check .` are clean with it.
* **`docs/README.md` needs a row for this record at merge**; it was not edited here.
* Untracked `.claude/` is not committed.

## 9. Questions for the coordinator

1. FIGARCH at `M = 1000` is within the cap by a factor of 32 and matches arch's default; is the oracle's truncation
   the right target, or do we want `M = 4000` (a 5% accuracy against the long-memory process, 12% of the cap, about
   twice the per-row time)?
2. The H-metric projection removes the blow-ups in FIGARCH and in the t forms; should it become the step's projection
   for the whole family (part 1's GARCH(2,2) projected rows, 137,000, are the same mechanism)? Untested on GARCH(2,2).
3. The skew-t closed form (open derivation 1) is the one item before a port.

# Study 492, part 4: the open derivations, skew-t score and EGARCH target (B11, part 1 of 2)

Python and numpy only (`scripts/study_492.py`); no Rust, no source change. Two of section 7's open derivations:
the skew-t's score in its shape parameters, and a variance target for EGARCH. Status: both derived and verified;
the skew-t closed form is now the script's score; the EGARCH target is an option (`Online(target=True,
level="start"|"running")` on `egarch`), measured, not adopted.

## 1. The skew-t score in closed form

**Density** (Hansen 1994, arch's `SkewStudent`; `nu > 2`, `-1 < lambda < 1`), for the standardised `z`:

    c   = G((nu+1)/2) / (sqrt(pi (nu-2)) G(nu/2))
    a   = 4 lambda c (nu-2)/(nu-1),        b = sqrt(1 + 3 lambda^2 - a^2)
    sg  = sign(z + a/b),   q = 1 + sg lambda,   m = (b z + a)/q
    log f(z) = ln b + ln c - (nu+1)/2 ln(1 + m^2/(nu-2))

**Step 1, `c`.** `ln c = lgamma((nu+1)/2) - lgamma(nu/2) - 1/2 ln pi - 1/2 ln(nu-2)`, so

    (ln c)_nu  = 1/2 [psi((nu+1)/2) - psi(nu/2) - 1/(nu-2)],     (ln c)_lam = 0

(the Student-t's constant term, part 2).

**Step 2, `a`.** `a` is `lambda` times `4 c (nu-2)/(nu-1)`; `d/d nu [(nu-2)/(nu-1)] = 1/(nu-1)^2`:

    a_lam = 4 c (nu-2)/(nu-1)                  (= a/lambda, written without the division)
    a_nu  = a (ln c)_nu + 4 lambda c/(nu-1)^2

**Step 3, `b`.** From `b^2 = 1 + 3 lambda^2 - a^2`, `2 b b_x = d(3 lambda^2)/dx - 2 a a_x`:

    b_lam = (3 lambda - a a_lam)/b,            b_nu = -a a_nu / b

**Step 4, `m`.** `q` depends on `lambda` only (`q_lam = sg`); `sg` is constant away from the kink `z = -a/b`:

    m_nu  = (b_nu z + a_nu)/q,                 m_lam = (b_lam z + a_lam)/q - sg m/q

**Step 5, the kink.** The branch point `z = -a/b` moves with `(nu, lambda)`, but `m = 0` there on both
branches, so `log f` is continuous in the parameters at every `z`; its derivative exists at every `z` except
the one point `z = -a/b`, a set of probability zero. No term from the moving branch point enters the score.

**Step 6, the log-density.** With `D = nu - 2 + m^2` and `d/dx ln(1 + m^2/(nu-2))`:

    d log f/d lam = b_lam/b - (nu+1) m m_lam / D
    d log f/d nu  = b_nu/b + (ln c)_nu - 1/2 ln(1 + m^2/(nu-2))
                    - (nu+1)(2 m m_nu (nu-2) - m^2) / (2 (nu-2) D)

At `lambda = 0` (`a = 0`, `b = 1`, `m = z`, `m_nu = 0`) the `nu` score is part 2's Student-t score, term for term.

**Step 7, the GARCH parameters.** The shape does not enter the variance recursion, so the chain rule runs only
through `z = e/sqrt(s)`: `dz/d theta = -z/2 dl`, `d log f/dz = -(nu+1) m b/(q D)`, and
`d l_t/d theta = d log f/dz (-z/2) dl - 1/2 dl = 1/2 w dl` with
`w = (nu+1) m b z/(q D) - 1`, part 2's form (already closed, unchanged). The shape score has no recursion: it
is the row's `(d log f/d nu, d log f/d lam)` at the row's `z`. (For the record: an EGARCH skew-t would also
carry the shape through `E|z|` in the news term; arch's EGARCH keeps `sqrt(2/pi)` for every distribution,
so with arch's form it would not.)

**Verification** (`study_492.py check t`; `skewt_shape_score` against differences of `logf`, 2,401 values of `z`
in `[-8, 8]` and `N(0, 4)` at each of 8 points `(nu, lambda)` = (6, -0.2), (6, 0), (3, 0.5), (4.5, -0.7),
(12, 0.3), (30, -0.05), (2.3, 0.9), (100, -0.95), excluding `|z + a/b| < 0.01`):

| reference | max abs error | max rel error |
|---|---|---|
| part 2's central difference (steps `1e-5 nu`, `1e-6`) | 3.4e-6 | 5.2e-6 |
| Richardson extrapolation of two central differences | 6.1e-9 | 9.9e-8 |

The error against the central difference is the difference's own (it falls 500-fold when the reference is
extrapolated); the worst points are the edges, `nu = 2.3, lambda = 0.9` and `nu = 100, lambda = -0.95`. Within
the default points it is 4e-9 absolute. The full score now matches a central difference of the likelihood to
5.2e-8 relative in the shape rows, a check that is no longer a difference against the same difference. The
mean score at the truth over 400,000 draws is unchanged, zero within its SE.

## 2. The skew-t comparison with the closed form

GARCH(1,1) skew-t, seeds 492100 to 492109, 10,000 rows, against arch, before (central difference) and after:

| variant | score | within 2 SE (every parameter) | median abs `nu` gap (arch SE 0.36) | median abs `lambda` gap | worst gap (SE) | us/row |
|---|---|---|---|---|---|---|
| free, online | difference | 9 of 10 | 0.086 | 0.0010 | 4.39 (`alpha`, 492108) | 114 |
| free, online | closed form | 9 of 10 | 0.086 | 0.0010 | 4.55 (`alpha`, 492108) | 94 |
| targeted, online | difference | 10 of 10 | 0.087 | 0.0013 | 1.91 | 129 |
| targeted, online | closed form | 10 of 10 | 0.087 | 0.0013 | 1.91 | 115 |

No change in the counts or the shape gap: the same seed (492108, the start that hit the iteration cap) is the one
outside 2 SE. The largest end-of-run parameter difference between the two scores is 0.018, on that seed
(a 1e-9 score change amplified over 9,500 steps on a poorly started run). Part 2 quoted 0.12 for the free
form's `nu` gap; that is its 50-seed median, the first ten seeds' is 0.086 either way. The closed form is
17% faster per row in numpy (two `digamma` calls and about 40 flops against four density evaluations, eight
`lgamma`).

## 3. An EGARCH variance target

**What `omega` fixes.** `l_t = log s_t = omega + beta l_{t-1} + alpha (|z_{t-1}| - c) + gamma z_{t-1}` with
`c = sqrt(2/pi)`. For iid `z` with `E z = 0` and `E|z| = c`, the news term has mean 0, so for `0 <= beta < 1`
the stationary mean is exactly

    E[log s] = omega/(1 - beta),       whatever alpha and gamma.

**The target.** `log e_t^2 = l_t + log z_t^2` with `z_t` independent of `l_t`, so
`E[log e^2] = E[log s] + E[log z^2]`. Under the normal, `log z^2` is `log chi2_1`, whose mean is
`psi(1/2) + ln 2 = -gamma_E - ln 2 = -1.27036`. The rule:

    L     = mean(log e_t^2) + gamma_E + ln 2           (an unbiased estimate of E[log s])
    omega = L (1 - beta)

**Closed form under the normal: yes, exact**, and `alpha, gamma` do not enter. It is the GARCH forms' targeting
with `v = L`, `phi = (alpha, gamma, beta)`, `c = (0, 0, 1)`; the free parameters' sensitivity is
`d_phi l - L (0, 0, 1) d_omega l`, no new recursion. Checked by simulation (4M rows): at `(0, 0.15, -0.08, 0.97)`
the mean of `log s` is +0.0000 against 0, `mean log e^2 + 1.2704` is -0.0003; at `(-0.05, 0.2, -0.1, 0.9)`,
-0.4997 and -0.4994 against -0.5.

**Why part 1 found no closed form.** Part 1 targeted `E[s]` (the sample variance). `ln E[s] = E[ln s] + K`,
with `K = sum_j [-alpha beta^j c + ln(exp(p_j^2/2) Phi(p_j) + exp(n_j^2/2) Phi(n_j))]`,
`p_j, n_j = beta^j (alpha +- gamma)` (Nelson 1991; the stationary `l` is a sum of independent news terms and
`E exp(u|z| + g z)` is closed under the normal). That is a convergent series (terms fall as `beta^(2j)`), not a
finite closed form; `egarch_log_mean_gap` sums it (0.1294 at the truth, simulated 0.1298). Targeting the
log-variance avoids it.

**Under the t (approximation).** Two terms change. (i) `E[log z^2]` for the standardised t is closed:
`z^2 = (nu-2)/nu F(1, nu)`, so `E[log z^2] = ln(nu-2) + psi(1/2) - psi(nu/2)` (-1.500 at `nu = 6`; simulated
-1.501). (ii) arch's EGARCH keeps `c = sqrt(2/pi)` for every distribution, while the t's
`E|z| = sqrt((nu-2)/pi) G((nu-1)/2)/G(nu/2)` (0.750 at `nu = 6`, against 0.798), so the news term has mean
`alpha (E|z| - c)` and `E[log s] = (omega + alpha (E|z| - c))/(1 - beta)`. The t rule is
`omega = (1 - beta) L - alpha (E|z| - c)` with `L = mean(log e^2) - E[log z^2]`, both at the start's `nu`.
It is exact given `nu`; the approximation is that `nu` is the start's estimate (500 rows, an SE near 1), and
`alpha` now enters. The `E[s]` target does not exist under the t: `E exp(u|z|)` is infinite for the t, so the
unconditional variance is infinite whenever `alpha` or `gamma` is not 0. Both t moments are in the script
(`e_logz2_t`, `e_abs_t`) and checked by simulation; the EGARCH-t estimator is not built (the study's t forms
are on `GarchPOQ` only), so the t rule is not run against arch.

**Results** (EGARCH(1,1), Gaussian, seeds 492000 to 492009, 10,000 rows, truth `(0, 0.15, -0.08, 0.97)`,
arch's `omega` SE median 0.0013):

| variant | within 2 SE (every parameter) | median abs `omega` gap (SE) | worst `omega` gap (SE) | median abs `omega` gap (value) | path gap max, median | us/row |
|---|---|---|---|---|---|---|
| plain (free `omega`) | 9 of 10 | 0.21 | 8.91 (492006) | 0.0003 | 3.8% | 17 |
| target, `L` held at the first 500 rows (as specified) | 4 of 10 | 2.16 | 5.10 | 0.0027 | 10.3% | 23 |
| target, `L` running | 8 of 10 (one blew up) | 0.41 | 2.01 | 0.0005 | 3.8% | 23 |

The held target is the rule as stated, `alpha, beta, gamma` at the start's fit and `L` from the first 500 rows;
it is worse than the free form because a 500-row mean of `log s` with `beta = 0.97` has an SD near 0.2 (the
AR(1) long-run variance `0.25 x 65.7/500` plus `pi^2/2/500` for `log z^2`), i.e. `omega` off by about
`0.03 x 0.2 = 0.006`, several arch SEs. The running `L` removes that and lands near the free form (0.41 SE
against 0.21), but does not beat it. Seed 492006, whose batch start hit the 50-iteration cap at
`(alpha, gamma, beta) = (-0.06, -0.04, 0.988)`, defeats every form: plain is 8.9 SE off on `omega`, held is
10.7 SE off on `alpha`, and the running target's step left float range at row 746 (`alpha` -7.3, `gamma` 16.4;
EGARCH projects `beta` only, `alpha` and `gamma` are unconstrained). The run records it as failed. The cause is
the start, not the target; a guard on the EGARCH step (a trust region, or a box on `alpha` and `gamma`) is not
derived here.

Recommendation: keep the free-`omega` EGARCH. The log target is exact and cheap, and it is the right target if
one is wanted (the `E[s]` one is a series under the normal and does not exist under the t), but it buys no
accuracy here.

## 4. State and cost

* **Skew-t closed form: state unchanged** (GARCH(1,1) skew-t free online 28 floats, 224 bytes). It adds no
  state: `a`, `b`, `c` and their derivatives are recomputed from `(nu, lambda)` each row, as the density already
  was. Per row: 2 `digamma`, 1 `log1p`, about 40 flops against 4 extra density evaluations (8 `lgamma`, 4 `log1p`),
  94 against 114 us/row in numpy. `digamma` is a libm-class result in the state's path, as before.
* **EGARCH target: state smaller.** `omega` is no longer stepped, so `H` is 3x3 (6 floats) instead of 4x4 (10);
  `L` takes the slot of the running mean of `e^2`, which EGARCH does not use (its projection has no scale);
  `held` keeps the start's mean of `log e^2`, `running` carries the running one in the same slot. 25 to 21 floats,
  200 to 168 bytes (section 7's EGARCH formula with `k(k+1)/2` replaced by `(k-1)k/2`).
  Per row: one `log` and the `omega = L (1 - beta)` product, 17 to 23 us/row in numpy (the target's extra
  array work, not its math). A zero residual has `log e^2 = -inf`; the script floors `e^2` at the smallest
  normal float, a production form needs a stated floor or a skip.

## 5. Open after this part

* The EGARCH-t target rule (section 3) is derived and its moments checked, not run against arch: there is no
  EGARCH-t estimator in the study.
* A step guard for EGARCH's unconstrained `alpha, gamma` after a capped start (seed 492006).
* Part 2 of B11 (the coordinator's second half) is not this part's.
