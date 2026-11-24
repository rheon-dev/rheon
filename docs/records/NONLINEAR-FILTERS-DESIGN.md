<!-- kind: record -->
# Nonlinear state-space filters (EKF, UKF, particle filter): the design record

The design for task 497, part 1, written 2026-11-24 on branch
`vm/task-497-nonlinear-filters` from `main` at 13f5ffe4. Nothing here is
built, and no build or test was run for it. Section 9 holds the questions for
the coordinator; section 10 lists what part 2 would build and test.

Inputs, none merged:

- the study of the same task, branch `vm/study-497` (head e4473d1a), its
  record `docs/records/STUDY-497-EKF-UKF.md` and script `scripts/study_497.py`
  there, cited as "study §N". Its numbers are its own; a number this record
  derives is said to be derived, and none here was measured;
- task 494's scan, `vm-tools/research/494-MODELS-SCAN.md` on branch
  `vm-tools/oracle-gap-scan`, section 3 ("scan §3");
- `crates/rheon-core/src/family.rs` (task 500's family objects) and
  `crates/rheon-core/src/kalman.rs` as they stand at 13f5ffe4;
- the fixed-lag smoother's design, `docs/records/FIXED-LAG-SMOOTHER-DESIGN.md`
  on branch `vm/task-496-fixed-lag` (task 496).

## 0. Decisions at a glance

| # | question | decision | reason, in one line |
|---|---|---|---|
| 1 | families | linear transitions only (`kalman`'s random walk and reversion); scalar observations through one index `η = z'β`, with the `family.rs` families and links, a Gaussian with a log or logit mean, and (particle filter only) a Student t; no expressions, no range-bearing | closed forms only, so a spec is data and every row is deterministic; the prediction step stays exact |
| 2 | EKF Jacobian | closed form per link, `H = (dμ/dη)(η̃) z'` at the clipped `η̃`; no finite differences | every offered pair has one; a difference step is a tuned constant with cross-platform rounding |
| 3 | UKF points | Merwe scaled points under filterpy's names; `β = 2`, `κ = 0` from Wan and van der Merwe (2000), `α = 1` in place of their `1e-3`; a scalar-index rule to be measured against it before the default is fixed | `α = 1e-3` puts a weight of about `-1e6` at the centre; the full rule's accuracy on a one-index observation depends on `k` (section 3.3) |
| 4 | resampling | when `ESS < τ N`, `τ = 0.5` by default; systematic by default, the four filterpy schemes offered; the random stream re-keyed on every observing row from `(key, n)`, `n` in the state | a condition on the cloud, never a row count; the draws of row `n` depend on nothing else, so any chunking and any save and load give the same bits |
| 5 | state | `kalman` gains configuration fields only, at the end, `serde(default)`; the particle filter is a new `ModelState` variant appended last; within unreleased schema 72 | no existing byte moves; the particles are `8 (k + 1)` bytes each per target, bounded by the spec, held to 256 KB a group |
| 6 | oracles | filterpy's `ExtendedKalmanFilter` and `UnscentedKalmanFilter` (dev group already, 1.4.5, no bump); filterpy's four resamplers index for index; the particle filter held to the Kalman filter by the formula of section 6.3 and to a grid filter on Gordon, Salmond and Smith's (1993) growth model | third-party wherever one computes the quantity; the grid is the only exact reference for the benchmark |
| 7 | cost, smoother | EKF `O(k²)`, UKF `O(k³)`, particle filter `O(N k)`; the EKF and UKF compose with task 496's ring unchanged (linear transition), lifted after 496 merges; no particle smoother | the backward pass never reads the observation; ancestor paths would cost 8 times the per-group cap |
| 8 | platforms | every `exp`, `ln`, `lgamma` in the three filters from the `libm` crate, as `garch` does | a resampling index is a step function of the weights: no tolerance is tiny there; the EKF's clipped streams amplify a last bit to 1e-10 |

## 1. The transition and observation families a spec can express

### 1.1 The model the three filters share

A row has features `z` (k, the intercept first under `fit_intercept`), a
target `y_j` per target, a weight `w` and a clock step `d`. Each target has
its coefficient state `β_j` (k). All three filters run on the same state-space
model, which is `kalman`'s with the observation generalized:

```text
transition   β_n = Φ(D_n) β_{n-1} + u_n,   u_n ~ N(0, Q(D_n))
             Φ(D) = diag(2^(-D/r_i)),  Q(D) = diag(q_i g_i(D)²)          (kalman's, unchanged)
observation  η_n = z_n' β_n,   μ_n = g⁻¹(η_n),
             y_n | η_n ~ family(μ_n, φ V(μ_n) / w_n)
```

`D_n` is the clock since the target last took an observation; the noise is
charged once on the row that observes, over the whole gap, as `kalman`
charges it, and `Φ` multiplies over any cut of a gap. A row that observes
nothing (weight 0, a null target) adds only its clock to `D`; under the
shared `ClockState` a zero-weight row's time is carried into the next row
as one gap, so no filter sees it as a step (task 539).

### 1.2 Transitions: linear in the state, additive Gaussian noise, only

**Decision.** The transition is `kalman`'s and nothing else: the random walk
with `Q D²` noise, and per-slot reversion. No nonlinear transition is offered
as a product family in task 497.

**Reasons.** (1) The state is a regression's coefficients; no family of
nonlinear coefficient dynamics has been asked for. (2) With a linear
transition the EKF's and UKF's prediction step is exact (`P⁻ = Φ P Φ + Q`),
so the only approximation is in the update, and the UKF need not push sigma
points through `f` (study §5, read 1: filterpy's recomputed prior is the
only part of its gap to a direct filter). (3) The fixed-lag smoother of
task 496 reads only the transition, so it applies unchanged (section 7.3).
(4) The particle filter's engine is generic over an internal trait (a
propagate step and a log-likelihood), so a named nonlinear transition can be
added later as one enum variant with its own oracle. The candidate is
stochastic volatility (an AR(1) log-variance, a scalar latent state), whose
case overlaps task 492; it waits for a user.

### 1.3 Observations: one index, a fixed set of families

**Decision.** A target is a scalar, observed through the single index
`η = z'β`, under one of:

| family | links | filters | note |
|---|---|---|---|
| gaussian | identity | linear (today's `kalman`), EKF, UKF, particle | the EKF and UKF are the Kalman filter here, to rounding (study §4, §5) |
| gaussian | log, logit | EKF, UKF, particle | a mean curve with constant variance: the "saturating" logistic `L/(1 + e^(-η))` with `L` as a scale on the target, an exponential trend. `obs_var` is required: the UKF and particle filter read `P` absolutely (section 5.1) |
| poisson | log | EKF, UKF, particle | |
| binomial | logit | EKF, UKF, particle | "logistic" in the classification sense; Bernoulli rows (no trial counts, study §11) |
| gamma | log, inverse | EKF, UKF, particle | |
| tweedie, `var_power` p | log, power | EKF, UKF, particle | |
| student_t, `nu`, `scale` | identity | particle only | heavy-tailed noise: its Gaussian approximation in a Kalman gain is a Gaussian with a larger variance, which `kalman` already is |

The families and links are `family.rs`'s, whose variance, link and
derivative are held to statsmodels' family objects
(`tests/test_second_opinion_glm.py`); the filters read the same functions,
through the `libm` crate (section 8.5).

**Expressible through the features, with no family of its own.**
A power law `y = a Π x_j^(b_j)` is the log link with `ln x_j` as features. A
saturating Michaelis-Menten curve `y = V s / (K + s)` is the inverse link
with `1/s` as a feature: `μ = 1/(β_0 + β_1/s) = s / (β_0 s + β_1)`, so
`V = 1/β_0` and `K = β_1/β_0`; under the Gamma family its noise has a constant
coefficient of variation. The user computes these features in Polars.

**Refused, with the reason.**

- **An arbitrary expression for `f` or `h`** (scan §3, family D). The spec is
  serialized and must give the same bits on every platform; an expression
  needs an evaluator and its derivatives in the core, and a closure breaks
  hard rule 4. Raised, not built.
- **Range-bearing.** It observes one state through two coordinates at once,
  one an angle whose residual wraps. The bank's targets are scalars observed
  independently, with no shared observation noise. It would need a joint
  update of two targets and a wrapped residual: a separate model with its own
  oracle (filterpy's filters take a vector `z`), raised only if a user asks.
- **Observations nonlinear in more than one index**, such as a
  four-parameter logistic in its parameters: none asked for, and each breaks
  the one-index reduction of section 3.3.

## 2. The EKF and its Jacobian

**Decision.** Closed form per link; no finite differences.

```text
η̃ = clip(z'β⁻),  μ̃ = g⁻¹(η̃),  m = dμ/dη at η̃          (identity 1, log μ, logit μ(1 − μ),
                                                         inverse −μ², power from family.rs)
H = m z',  R = φ V(μ̃) / w
s = m² z'P⁻z + R,   K = (m / s) P⁻ z
β ← β⁻ + K (y − μ̃),   P ← P⁻ − s K K'
pred = μ̃                                                 (before y is read)
```

- **The clip.** `family.rs`'s `ETA_CLIP = 30` for the log and logit links
  (the study used 36 for the logit; one clip shared with `glm` is simpler
  and the logit's mean is within `1e-13` of its bound at 30). The EKF is
  linearized *at* the clipped value, never through the clip's own derivative,
  which is 0 past the bound and froze the study's filter for 1,137 rows
  (study §2). The clip is necessary and not sufficient (study §8, D2).
- **Inverse and power links at `η ≤ 0`.** No mean: the prediction is
  withheld with a reason and the row learns nothing, as `glm` does; it adds
  its clock to `D` as any row that observes nothing.
- **No finite differences.** Every offered pair has its derivative in
  `family.rs`. A difference step is a constant tuned to the data's scale,
  and its rounding differs by platform. If a future family has no closed
  form, its difference is measured against a neighbour's closed form and its
  error bounded in the test before it lands; none is needed now.
- **The covariance forms are `kalman`'s own**, not filterpy's Joseph matrix
  product: `"covariance"` (the update above, about `5k²` flops) and
  `"square_root"` (Carlson's scalar step with `h = m z`). The study found the
  standard and expanded-Joseph forms part from filterpy's matrix product only
  on streams that reach the clip, where the EKF has already lost track
  (`b` 1.9e-7 and 7.1e-7 against a band of 3.6e-10, study §4, §9); the matrix
  form costs `4k³`. The tests hold well-conditioned streams at 1e-12
  relative, and clipped streams at 10 times the filter's own one-ulp band
  (study §10). The square-root form is the remedy for that cancellation and
  stays offered.

## 3. The UKF

### 3.1 The recursion

```text
prior      b⁻ = Φ b,  P⁻ = Φ P Φ + Q            (exact: the transition is linear)
points     L = chol((k + λ) P⁻),  χ_0 = b⁻,  χ_{±i} = b⁻ ± L_i,   λ = α²(k + κ) − k
weights    Wm_0 = λ/(k + λ),  Wc_0 = Wm_0 + 1 − α² + β,  W_i = 1 / (2(k + λ))
update     ζ_i = g⁻¹(clip(z'χ_i)),  ζ̄ = Σ Wm_i ζ_i
           R = φ Σ Wm_i V(ζ_i) / w           ("expected", default)
             or φ V(g⁻¹(z'b⁻)) / w            ("plugin", as the EKF)
           s = Σ Wc_i (ζ_i − ζ̄)² + R,  C = Σ Wc_i (χ_i − b⁻)(ζ_i − ζ̄)
           K = C / s,  b = b⁻ + K (y − ζ̄),  P = P⁻ − s K K',  then P ← (P + P')/2
pred = ζ̄                                       (before y is read)
```

`P` is symmetrized after each update: the Cholesky reads one triangle, and an
antisymmetric rounding part grew to 8.5e-7 in the study under a forgetting
factor (study §5, read 3). The prediction is the points' mean, which carries
the log link's `e^(x'Px/2)` that the EKF's plug-in misses (study §6, read 3).

### 3.2 The parameters and their defaults

Exposed under filterpy's `MerweScaledSigmaPoints` names, `alpha`, `beta`,
`kappa`, so a test passes them through unchanged.

- **`beta = 2`, `kappa = 0`: Wan and van der Merwe (2000).** `β = 2` is
  their optimum for a Gaussian prior; `κ = 0` their usual secondary scaling.
- **`alpha = 1`, not their worked `1e-3`.** At `α = 1e-3` the centre weight
  is `1 − 1/α² ≈ −1e6` at every `k`, and the weighted sums lose about six
  digits to cancellation (study §8, D4; the loss was not measured beyond
  "no failure"). A small `α` also closes the points on the mean, which makes
  the UKF a second-order EKF and gives up the spread that kept it on track
  where the EKF was lost (study §8: 0 of 600 streams against 15; the feature
  spikes of D2). At `α = 1`, `κ = 0` the centre weight is 0 at every `k`:
  no negative weight. It is the setting the study measured against filterpy
  and the exact posterior.

### 3.3 The full rule depends on `k` on a one-index observation (derived, not measured)

Every offered observation reads `β` only through `η = z'β`. The points then
matter only through their projections `z'(χ_i − b) = ±sqrt(k + λ) z'L_i`.
With `s² = z'P⁻z = Σ_i (z'L_i)²/(k + λ)`, the fourth central moment the
rule gives `η` is `Σ_i (z'L_i)⁴ / (k + λ)`, between `(k + λ) s⁴ / k` (the
projections spread evenly over the `k` columns) and `(k + λ) s⁴` (all on
one column); a Gaussian's is `3 s⁴`. Which it is depends on how `z` sits
against the Cholesky columns, so on the ordering of the features and, under
`standardize`, on the anchor's coordinates. At `(1, 2, 0)`, `k + λ = k`, so the
moment lies between `s⁴` and `k s⁴`: too light by up to 3 times, or too
heavy by up to `k/3` times.

Worked, with all the projection on one column and `s = 0.5`: the true
`E[e^η] / e^(η̄) = e^(s²/2) = 1.1331`; the three-point projected rule gives
1.1303 at `k = 2` and 1.1866 at `k = 20` (+4.7%, points at `±4.5 s`); at
`k + λ = 3` it gives 1.1330.

**A scalar-index rule removes the dependence.** Since `β` given `η` is
linear-Gaussian, `Cov(β, h(η)) = P⁻z · Cov(η, h(η)) / s²` exactly, so the
update needs only a one-dimensional rule on `η ~ N(z'b⁻, s²)`. The
three-point Gauss-Hermite rule (points `0, ±sqrt(3) s`, weights `2/3, 1/6,
1/6`) is exact for polynomials in `η` up to degree 5. It costs `O(k²)` (no
Cholesky, three link calls), is free of the coordinates, and is, derived
here, filterpy's `UnscentedKalmanFilter` with `JulierSigmaPoints(k, kappa =
3 − k)` and a `sqrt_method` whose first row lies along `P⁻z`: the other
`2(k − 1)` points then project onto the centre, their cross-covariance terms
cancel in pairs, and the centre's weight `(3 − k)/3` and theirs sum to `2/3`.
So it has an exact third-party oracle.

**Decision.** Part 2 builds the full rule as above, then measures it at
`(1, 2, 0)` against the scalar-index rule (and Julier `κ = 3 − k`) on the
study's second oracle: the grid at `k = 2`, the particle filter at `k = 5`,
statsmodels' static posterior at `k = 20`. The default is the one closer to
the exact posterior; the other stays an option. **For the coordinator to
rule (section 9, Q1).**

### 3.4 Guards

- **A Cholesky that fails** (`P⁻` not positive definite in rounding): that
  row takes the EKF's update instead, on the same state, and emits
  `filter_step = "ekf"` so the fall-back is counted. No jitter is added to
  `P`: a jitter is a constant tuned to the data that moves the fit.
- **The clip and the inverse links** as the EKF (section 2).

## 4. The particle filter

### 4.1 Scope

**Decision.** A new kind, `particle`, over `kalman`'s regression state and
transition, with the observation families of section 1.3, Student t
included. Not an option of `kalman`: its state and its outputs differ.

Refused, each by a condition on the spec, not by a row count:

- **No process noise** (`q_i = 0` on some slot, or no `q`). The bootstrap
  filter rejuvenates a coefficient only through its noise; without it the
  distinct values in the cloud can only fall after each resampling. The UKF
  and `glm` serve static coefficients.
- **A diffuse intercept.** A cloud cannot hold an improper prior, and the
  posterior after a single Poisson zero or binomial row under a diffuse
  intercept is itself improper. Every coefficient, the intercept with it, has
  the proper prior `N(0, p0)`. The same holds for the EKF and UKF under a
  family other than the Gaussian (section 5.1).
- **`standardize = True`** in its first build. The particles live in the
  user's coordinates, as `p0` and `q` are written. A re-map of the cloud is
  `O(N k²)` a re-map, bounded, and can be added later.

### 4.2 The row

On a row that observes target `j` (positive weight, finite target), with
`n` the target's observing rows so far and `D` its gap:

```text
1. stream    r = SplitMix64(mix(key, n))                  (section 4.4)
2. start     if n = 0: β_i ~ N(0, p0 I) from r, ℓ_i = 0    (i = 1..N)
3. propagate β_i ← Φ(D) β_i + sqrt(q) ∘ g(D) ∘ ε_i,  ε_i ~ N(0, I) from r
4. predict   pred = Σ w̄_i μ(z'β_i);  pred_var = Σ w̄_i (φ V(μ_i) + μ_i²) − pred²
             (emitted now: y has not been read)
5. weigh     ℓ_i += w · log p(y | z'β_i)                   (the likelihood to the row's weight, as R/w)
6. normalize w̄_i = exp(ℓ_i − max ℓ) / Σ;  ESS = 1 / Σ w̄_i²;
             loglik = log Σ w̄_i^prev p(y | z'β_i)^w     (emitted)
7. resample  if ESS < τ N: indices from r by the scheme; β ← β[indices]; ℓ ← 0
8. close     D ← 0, n ← n + 1
```

A row that observes nothing (weight 0, a null target) is predicted from the
stream row `n` will use, propagated over the gap so far, and nothing is
stored: no draw is consumed, `n`, `ℓ` and the cloud are unmoved, and `D`
follows the clock as in `kalman`. So the next observing row draws the same
numbers as in the stream without that row (rule 9), and its prediction is
the next observation's prior at the time of the row.

Outputs: `pred`, `pred_var`, `coef` (`Σ w̄_i β_i`), `se_coef` (the weighted
sd), `ess` (before resampling), `resampled`, `loglik`, `weight_sum`.

### 4.3 Resampling: the condition is the ESS

**Decision.** Resample when `ESS < τ N`, `τ = resample_ess` in `(0, 1]`,
default `0.5` (Liu and Chen 1995; Doucet and Johansen's tutorial, 2009).
`τ = 1` resamples whenever the weights are not all equal. There is no
cadence and no row count anywhere in the rule: the trigger is the cloud's
own effective size.

Schemes: `"systematic"` (default: one uniform a resampling, `O(N)`),
`"stratified"` (the one with a proved variance no larger than multinomial's,
Douc, Cappé and Moulines 2005), `"residual"`, `"multinomial"`: the four
filterpy has, each its oracle. The cumulative sum is taken in particle
order, sequentially, as numpy's `cumsum`; its last entry is set to 1, as
filterpy's multinomial and residual do. filterpy's systematic and stratified
do not, and can index past the end on round-off, where they raise; ours
clamps, a stated difference.

**Collapse.** Every family's log-likelihood is finite on its support at a
finite mean, and a target outside the support is refused by the bank before
the model (as `glm`), so `max ℓ` is finite and `ESS ≥ 1` on every row. A
cloud that degenerates shows it in `ess`. Nothing resets it and nothing
jitters it: either is a tuned constant.

### 4.4 Determinism under a seed

Chunk invariance, save and load, and every platform must give the same
bits. So:

- **The stream is keyed per observing row.** Row `n`'s draws come from
  SplitMix64 (`rheon_core::cluster::SplitMix64`, already in the core for
  `kmeans`) started at `mix(key, n)`, `mix` being SplitMix64's finalizer.
  The state holds `n` (u64) per target and `key` (u64) per group. However
  many draws a row consumes (a resampling or not, a scheme's count), the
  next row's draws do not move. A chunk boundary, a save and a load all sit
  between rows, so none can move a draw.
- **`key = mix(seed, hash of the group key)`**, the hash being the bank
  file's stable one, computed by the bank when it creates the group and kept
  in the state. Each group then has its own stream: a group's draws depend
  on neither the order groups first appear nor on a partition of the input
  across processes. (`kmeans` uses one seed for every group; for Monte Carlo
  that would correlate the errors of groups. Section 9, Q5.)
- **Normals by inverse CDF.** Wichura's AS241 (`PPND16`), one uniform a
  normal, `u = (top 53 bits + 0.5) · 2⁻⁵³` in `(0, 1)`: a rational function in
  the central 85%, `ln` and `sqrt` only in the tails. A row's draw count is
  fixed by `(N, k, scheme)`, and each normal is addressable.
- **Fixed order.** Sums over particles in particle order, on one thread per
  target (groups and targets may run in parallel; a cloud is never split).
- **Same bits on every platform**: section 8.5.

## 5. The state and its layout

### 5.1 EKF and UKF: options of `kalman`, no new state field

The persisted state per target stays `kalman`'s: `b`, `P`, `D`, the weight
sum. The UKF's points are scratch, rebuilt every row (`8 (3k² + 5k + 2)`
bytes, study §9). The configuration gains, appended last, each
`#[serde(default)]` and written only when it differs from the default (task
538's convention), so every existing `kalman` state keeps its bytes:

| field | default | values |
|---|---|---|
| `family` | gaussian | section 1.3 |
| `link` | the family's | section 1.3 |
| `var_power` | none | Tweedie only |
| `dispersion` (φ) | 1 | positive; fixed |
| `filter` | linear | linear, ekf, ukf |
| `alpha`, `beta`, `kappa` | 1, 2, 0 | filterpy's ranges |
| `ukf_noise` | expected | expected, plugin |

Under a family other than the Gaussian identity:

- **`P` is absolute and φ fixed.** The noise-units form (`P` over a learned
  noise, which cancels from the gain) holds for the EKF, whose gain does not
  read `P`'s scale through the linearization point, but not for the UKF or
  the particle filter, whose points spread by `P`'s absolute size. One rule
  for all three: φ is the spec's `dispersion`, `obs_var` is refused (required
  for the Gaussian with a log or logit link), and a learned φ is left out
  (study §11).
- **The process noise is the explicit `q`** (η's scale per clock unit²).
  `coef_half_life` is refused there: its matching to exponential forgetting
  assumes a row's information is `z z'` over the noise, and a GLM row's is
  `W z z' / φ`, with `W` read at the prediction. A matching at the row's own
  `W` is possible but unmeasured (section 9, Q6).
- **The intercept's prior is proper** (`p0`), as in section 4.1.
- **`standardize`** stays offered: `z'β` is invariant under the anchor's
  re-maps. The full UKF's points are not (section 3.3), so its output then
  depends on the anchor, deterministically; the filterpy tests run unstandardized.

### 5.2 The particle filter: a new variant, appended

`ModelState::Particle`, appended after every existing variant (the variant
names are the file's tags, frozen; appending moves no other layout, as
`ewm_moments` did at 72). Per group: the configuration, `key` (u64); per
target: `n` (u64), `D` and the weight sum (f64), `particles` (`N k` f64,
row-major, empty before the first observation) and `logw` (`N` f64, empty
before it).

**Bytes, bounded by the spec.** Persisted: `8 (k + 1)` bytes a particle a
target, so `8 T N (k + 1) + 24 T + 8` a group plus the configuration.
Scratch, not persisted: the resampling copy and the ancestor indices, `8k + 4`
bytes a particle.

| N | k = 2 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|
| 1,000 | 24,000 | 48,000 | 88,000 | 168,000 |
| 5,000 | 120,000 | 240,000 | refused | refused |
| 10,000 | 240,000 | refused | refused | refused |

(one target; bytes). The cap is task 537's per-group default, 256 KB:
`8 T N (k + 1) ≤ 262,144`, checked in `validate`, where `N`, `T` and `k` are
all known, refusing with the bytes and the largest `N` that fits. Nothing in
the state grows with the stream. `N` defaults to 1,000 (section 6.3 gives
its Monte Carlo error).

### 5.3 Schema and fixtures

Schema 72 is unreleased, so both changes ride on it: no
`SCHEMA_VERSION` bump. A `kalman` state of any schema loads with the new
fields at their defaults (the linear Gaussian filter), to the bit; no loader
is needed. The fixtures are regenerated at the merge
(`PRINT_STATE_FIXTURES=1`, the coordinator's step). New cases, each
decoded by a test that holds its new forms non-empty:

- `kalman_ekf_poisson`, `kalman_ekf_binomial_square_root`,
  `kalman_ukf_gamma_inverse`, `kalman_ukf_gaussian_logit`: each
  configuration form;
- `particle_unobserved` (empty clouds), `particle_weighted` (non-uniform
  `logw`), `particle_resampled` (`logw` all 0), `particle_two_targets` (one
  observed, one not), and one per family (`gaussian`, `poisson`,
  `binomial`, `student_t`) and per scheme.

If 72 ships first, the change becomes 73, schema 72's sets are copied to
`v72/` and the loader is the `serde` defaults above.

## 6. Oracles

### 6.1 EKF: filterpy's `ExtendedKalmanFilter`

filterpy 1.4.5 is already in the dev group (`uv.lock`); no new dependency.
Per observing row: `F = Φ(D)`, `Q = Q(D)` set before `predict`; `HJacobian =
m(η̃) z'`, `Hx = μ(η̃)`, `R = φ V(μ̃)/w`, at the prior mean. Tolerances, from
study §10: 1e-12 relative on `b`, `P`, `pred` on streams that never reach
the clip (measured worst 2.4e-14, 3.8e-14, 2.1e-13 for the standard form);
on clipped streams, 10 times the filter's own band. On the linear Gaussian
case the EKF and UKF are held to filterpy's `KalmanFilter` at 1e-13
(measured 2.0e-15).

### 6.2 UKF: filterpy's `UnscentedKalmanFilter`, the prior handed over

`MerweScaledSigmaPoints(k, alpha, beta, kappa)` with scipy's Cholesky,
`fx = Φ`, `hx = μ(clip(z'·))`, `R` passed per row. filterpy 1.4.5 does not
redraw the points after `predict` and 2.0 does (study §3), so the test hands
filterpy the prior on every row: `x = b⁻`, `P = P⁻` and `sigmas_f` from its
own `points_fn` on them. The test is then the same under either version, and
the lock stays at 1.4.5. Tolerance 2e-14 relative on `b`, 5e-14 on `P` and
`pred` (measured 1.7e-15 and 7.2e-15). The scalar-index rule, if built, is
held to `JulierSigmaPoints(k, 3 − k)` with the aligned `sqrt_method`
(section 3.3), the same way.

Against the exact posterior (approximation error, a statistic of the
stream, on the study's seeds): the bounds of study §10's second table, the
grid at `k = 2`, the particle filter at `k = 5`, statsmodels' static
posterior mode at `k = 20`.

### 6.3 The particle filter

**(i) The resamplers, index for index.** filterpy's
`monte_carlo.resampling` `systematic_resample`, `stratified_resample`,
`residual_resample`, `multinomial_resample`, on the same normalized weights
and the same uniforms: the test patches the module's `random` to return the
uniforms the Rust drew, and reaches the Rust resampler through an internal
function of the extension (as the Gram helpers are), not a public name.

**(ii) The linear Gaussian case, held to the Kalman filter.** Gaussian
identity, `q > 0`, the same rows through `particle` and through filterpy's
`KalmanFilter`, which gives the exact posterior `(b_t, P_t)`. For particle
mean `m̂_t,j`, the bootstrap filter's central limit theorem (Chopin 2004; Del
Moral 2004) gives

```text
e_t,j = (m̂_t,j − b_t,j) / sqrt(P_t,jj / N)  →  N(0, ρ_t,j),   ρ ≥ 1
```

with `ρ` the asymptotic variance over the posterior variance (1 for
independent draws from the posterior; larger from weighting and resampling),
bounded in time under the model's mixing (Del Moral and Guionnet 2001). The
test runs `R` seeds on the same rows and, over all `M = T_rows · k` cells,
with `ē` and `s` the replicates' mean and sd of `e`:

```text
bias        |ē_t,j| ≤ t_{R−1}(1 − α/(2M)) · s_t,j / sqrt(R)       (Bonferroni, α = 1e-3)
efficiency  s²_t,j  ≤ ρ_max
one run     |e_t,j| ≤ sqrt(ρ_max) · Φ⁻¹(1 − α/(2M))                (the essentials tier)
```

The self-normalized estimator's bias is `O(1/N)`, below `sqrt(ρ P / N)` for
`N ≥ 100`. `ρ_max` is set from part 2's measurement with room, as the
oracle ledger sets every grid's bound, and only tightens. For scale: at
`M = 5,000`, `Φ⁻¹(1 − 1e-7) = 5.2`; at `ρ_max = 10` and `N = 1,000` the one-run
bound is `16.4 sqrt(P/N) = 0.52` posterior sd, and the Monte Carlo sd itself
is `sqrt(ρ/N) = 0.03` to `0.1` posterior sd for `ρ` from 1 to 10. The
posterior variance is held the same way, `(v̂ − P)/P` against
`sqrt(2ρ_v/N)`. The autocorrelation of `e` over rows makes Bonferroni
conservative.

**(iii) A published nonlinear benchmark: Gordon, Salmond and Smith (1993).**
The univariate nonstationary growth model (from Kitagawa 1987 and Netto et
al. 1978 before it):

```text
x_t = x_{t−1}/2 + 25 x_{t−1} / (1 + x_{t−1}²) + 8 cos(1.2 t) + v_t,   v_t ~ N(0, 10)
y_t = x_t² / 20 + w_t,                                                 w_t ~ N(0, 1)
```

Its squared observation leaves the sign of `x_t` ambiguous, so the posterior
is bimodal: the case a particle filter exists for and the EKF and UKF fail on
by construction. Its transition is nonlinear, so it is no product family
(section 1.2): a Rust test implements the engine's internal particle-model
trait for it (propagate, log-likelihood; the forcing `8 cos(1.2 t)` passed
as an input, so the model reads no clock) and drives the same resampling,
stream and weights the product uses.

**Where the reference values come from.** Not from the paper: its figures
are from its own random draws and give no row-by-row values. The reference
is the exact filtering density of the test's own seeded stream, computed by a
point-mass (grid) filter written from the definition in the same test: a
one-dimensional state on a grid of `n` points spanning the prior and the
path with 8 sd to spare, the transition density applied as an `n × n` matrix,
Bayes' rule pointwise. The grid is checked twice, as the study's was (study
§6): against itself at 0.75 of its resolution (its band), and, with the
`25x/(1 + x²)` term dropped and `y = x + w`, against the Kalman recursion.
Held: the filtered mean and `P(x_t > 0)`, by the formula of (ii) with the
grid's posterior variance for `P_t`, and `sqrt(ρ p (1 − p) / N)` for the
probability. The prior's variance is taken as the paper states it, checked
against the paper before the test cites it (the scan's citations are from
memory).

**(iv) The other families.** The GLM families: the particle filter against
the study's grid at `k = 2` by the formula of (ii), the grid's variance in
place of `P`. Student t: the grid at `k = 1` and `2`, and each row's
log-likelihood against `scipy.stats.t.logpdf` (third-party).

## 7. Cost per row, and the smoother

### 7.1 Arithmetic per row

| filter | an observing row | a row that observes nothing | transcendental calls an observing row |
|---|---|---|---|
| linear (today) | about `5k²` | `z'b` | none |
| EKF | about `5k²` (covariance form); Carlson `O(k²)` without noise, `O(k³)` with | `z'b`, one link | one link, one `V` |
| UKF, full rule | about `k³/3 + 13k²` | the Cholesky and `2k + 1` links | `2k + 1` links and `V` |
| UKF, scalar-index rule | about `6k²` | `O(k)` after `P⁻z` | 3 links and `V` |
| particle | `N` (`k` normals, `3k` flops, one link, one log-likelihood), `O(N)` to normalize, `O(N k)` to resample when the ESS says | `N` (`k` normals, `k` flops, one link) | `N` links, `N` `exp`, `ln` on about 15% of the `N k` normals |

The coordinator's "EKF `O(k³)`" holds for filterpy's Joseph matrix product,
not for a scalar observation with a diagonal `Φ`: there the EKF is
`kalman`'s own update, `O(k²)` (study §9). The UKF's `k³/3` is its Cholesky.
At `k = 20`: EKF about 2,100 flops and one `exp`; UKF about 8,500 and 41
(study §9). The particle filter at `N = 1,000`, `k = 5`: about `10⁵` flops,
5,000 normals and 1,000 `exp` a row, two to three orders above `kalman`'s
150 (derived, not measured). Part 2 adds bench cases for each and measures
the `libm` cost (`garch` measured its `exp` at 1.8 times Apple's, 3.2 ns).

### 7.2 Where it runs

The particle filter's rows are its cost, and its cloud is one target's
sequential work; the bank's parallelism is over groups and targets, as
today. `N` and `resample_ess` are the user's levers.

### 7.3 The fixed-lag smoother (task 496)

Task 496's `smooth_lag` runs the RTS pass over a ring of the last `L`
observations' `(c_n, G_n, K_n)`, built from the filtered `(b_n, P_n)`, the
prior `P⁻_{n+1}` and `F_{n+1}`, at `O(L k²)` for the means and `O(L k³)`
with the covariances (its design, section 3.1).

- **EKF and UKF: the same smoother, no new code.** The backward pass never
  reads the observation, and the transition is linear, so the extended RTS
  smoother and the unscented one (Särkkä 2008; with linear dynamics its
  cross-covariance is `P_n Φ'`, which makes its gain RTS's) are task 496's
  pass on the filter's own stored steps. The smoothed value is then exact
  RTS of an approximate filter; the filter's approximation carries into it.
  **Decision:** task 497 refuses `smooth_lag` with `filter ≠ "linear"` until
  task 496 has merged; a follow-up lifts the refusal with one test against
  filterpy's `rts_smoother` on the stored steps (and
  `UnscentedKalmanFilter.rts_smoother` with `fx = Φ` as a second).
- **Particle filter: no smoother, deferred.** A fixed-lag smoother by
  ancestor paths keeps `L` generations: `8 L N k` bytes, 2 MB at `L = 50`,
  `N = 1,000`, `k = 5`, eight times the per-group cap; and path degeneracy
  leaves few distinct ancestors `L` rows back. Forward-filtering
  backward-smoothing avoids that at `O(N² L)` a row. Not asked for; not built.

## 8. The rules, held

1. **Out of sample (rule 2).** Each filter emits `pred` before reading `y`:
   the EKF's `μ̃`, the UKF's `ζ̄`, the particle filter's weighted mean of the
   propagated cloud, whose draws do not read `y`. Held by the probe the study
   ran (a row's target and weight moved no prediction up to and including
   it, study §4).
2. **A zero-weight row (rule 9).** EKF and UKF: `kalman`'s path, unchanged
   (study §7: `b`, `P`, `n_eff` held to the bit, the run equal to the stream
   without those rows). Particle filter: section 4.2; no draw consumed.
3. **No row counts (the user, 2026-11-22).** The resampling trigger is the
   ESS; the clip and the missing mean are conditions on `η`; a failed
   Cholesky takes the EKF step and says so; a spec whose process noise
   cannot rejuvenate a cloud is refused. No warm-up count anywhere.
4. **Chunk invariance (rule 3).** Every quantity is per row, and the
   particle stream is keyed on `(key, n)` (section 4.4).
5. **The same bits on every platform.** Every `exp`, `ln`, `log1p` and
   `lgamma` in the three filters is the `libm` crate's, as `garch`'s are
   (`garch/math.rs`): it is already linked, so nothing new is linked (rule
   12), and every state field is then bit-exact across platforms, with no
   tolerance in ulps to state. For the particle filter there is no
   alternative: a resampling index is a step function of the weights, so a
   last-bit difference picks another ancestor and the clouds part entirely.
   For the EKF and UKF it is the cheaper choice too: on clipped streams the
   filter amplifies a one-ulp difference to 1e-10 relative (study §4), no
   tiny tolerance. `family.rs` gains a math parameter (its functions generic
   over an `exp`/`ln` provider), so `glm` keeps the platform's functions and
   its outputs do not move.
6. **`n_eff` (rule 8)** is `kalman`'s weight sum before the row, in all three.
7. **f64, no `unsafe` (rule 6)**; the models in `rheon-core` behind
   `OnlineModel` (rule 4); the bank only computes the group's key.

## 9. Questions for the coordinator

1. **The UKF's default rule** (section 3.3). Build the full Merwe rule at
   `(α, β, κ) = (1, 2, 0)` and the scalar-index Gauss-Hermite rule, measure
   both on the study's second oracle, and default to the closer? The
   scalar-index rule is `O(k²)` against `O(k³)`, free of the coordinates, and
   has filterpy as its exact oracle; the full rule's error on a one-index
   observation grows with `k` (derived: +4.7% in `E[e^η]` at `k = 20`,
   `s = 0.5`). And is `α = 1` in place of Wan and van der Merwe's `1e-3`
   accepted (section 3.2)?
2. **The `libm` crate for every transcendental call in the three filters**
   (section 8.5): required for the particle filter, proposed for the EKF and
   UKF, at a cost part 2 measures. The alternative for the EKF and UKF is the
   platform's functions with a tolerance in ulps that the clipped streams may
   not meet.
3. **The particle filter's scope.** As designed, a regression state under a
   256 KB cap: `N (k + 1) ≤ 32,768` for one target, so `N ≤ 5,461` at `k = 5`.
   The study needed 50,000 particles at `k = 5` and still saw one row's ESS
   fall to 176 (study §6); its value is in low dimensions and in the Student
   t. Build it now (part 2b, after the EKF and UKF), or build only the engine
   with the growth-model benchmark and wait for a user model (stochastic
   volatility) before a product kind?
4. filterpy stays at 1.4.5: the UKF test hands the prior over (section 6.2).
5. The particle stream keyed per group from a hash of the group key (section
   4.4), where `kmeans` uses one seed for all groups.
6. Under a family other than the Gaussian, `coef_half_life` refused and `q`
   required (section 5.1), or a matching at the row's working weight, to be
   measured first?
7. Section 7.3's smoother: refused with a nonlinear filter until task 496
   merges, then lifted with one test.

## 10. Part 2: what would be built and tested

**2a, the EKF and UKF as options of `kalman`.** `rheon-core`: the update of
section 2 and 3 beside `kalman`'s, `family.rs`'s functions generic over the
math provider. `rheon-polars`: the spec fields and their refusals (section
5.1). Python: `po.spec.kalman(family=, link=, var_power=, dispersion=,
filter=, alpha=, beta=, kappa=, ukf_noise=)`. Tests: filterpy (6.1, 6.2),
the second oracle's bounds, the zero-weight and out-of-sample probes, the
contract (`model_contract.rs`), chunk invariance, the fixtures (5.3), bench
cases at `k = 5` and `20`.

**2b, the particle filter.** `rheon-core`: the engine generic over the
internal particle-model trait, the stream, AS241, the four resamplers, the
`particle` model. `rheon-polars`: the kind, the cap in `validate`, the group
key. Python: `po.spec.particle(...)`. Tests: 6.3 (i) to (iv), a save and load
mid-stream and between resamplings to the bit, a chunking of 1 and of 1,000
rows to the bit, the cap driven past its limit, bench cases at `N = 1,000`,
`k = 2` and `5`.

## 11. Not verified

- Nothing here was built or run. Every cost is the study's flop count or an
  estimate; the particle filter's per-row time is not measured.
- Section 3.3 (the full rule's `k` dependence, the scalar-index rule and its
  filterpy equivalence) is derived on paper, not run.
- `ρ` for the particle filter on these models is not known; `ρ_max` comes
  from part 2.
- The growth model's prior variance, and the parameter names of `particles`
  (Chopin's library, a possible second opinion), are from memory.
- AS241's speed beside Box-Muller under `libm` is not measured.
- The cross-platform claims rest on `garch`'s precedent and the `libm`
  crate's being compiled from source, not on a run of these filters on
  macOS and Windows.
