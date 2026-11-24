<!-- kind: record -->
# Nonlinear state-space filters (EKF and UKF): the design record

The design for task 497, part 1, first written 2026-11-24 on branch
`vm/task-497-nonlinear-filters` from `main` at 13f5ffe4, and revised the
same day after the coordinator's rulings (section 9). Nothing in this record
was built when it was written; part 2a builds sections 1 to 8. Section 10
lists what part 2a builds and tests.

**The particle filter was dropped (the user, 2026-11-24).** The first
version of this record also designed a bootstrap particle filter: a new
`particle` kind over `kalman`'s regression state, resampling on the
effective sample size (ESS), a random stream keyed per observing row, a
state of `8 (k + 1)` bytes a particle a target held to the 256 KB per-group
cap, and Gordon, Salmond and Smith's (1993) growth model against a grid
filter as its benchmark. The user dropped it from this release, on two
measurements set side by side: the cap allows `N (k + 1) ≤ 32,768` for one
target, so at most 5,461 particles at `k = 5`, where the study's particle
filter needed 50,000 particles at `k = 5` and still saw one row's ESS fall
to 176 (study §6). A regression state of that size is beyond what the
bootstrap filter does well under the cap. Task 497 is now the EKF and UKF
only: there is no particle kind, no `ModelState` variant for one, no engine
and no benchmark. The earlier text is in this file's history at 0e724dc8.

Inputs, none merged:

- the study of the same task, branch `vm/study-497` (head e4473d1a), its
  record `docs/records/STUDY-497-EKF-UKF.md` and script `scripts/study_497.py`
  there, cited as "study §N". Its numbers are its own; a number this record
  derives is said to be derived;
- task 494's scan, `vm-tools/research/494-MODELS-SCAN.md` on branch
  `vm-tools/oracle-gap-scan`, section 3 ("scan §3");
- `crates/rheon-core/src/family.rs` (task 500's family objects) and
  `crates/rheon-core/src/kalman.rs` as they stand at 13f5ffe4;
- the fixed-lag smoother's design, `docs/records/FIXED-LAG-SMOOTHER-DESIGN.md`
  on branch `vm/task-496-fixed-lag` (task 496).

## 0. Decisions at a glance

| # | question | decision | reason, in one line |
|---|---|---|---|
| 1 | families | linear transitions only (`kalman`'s random walk and reversion); scalar observations through one index `η = z'β`, with the `family.rs` families and links and a Gaussian with an identity, log or logit mean; no expressions, no range-bearing | closed forms only, so a spec is data and every row is deterministic; the prediction step stays exact |
| 2 | EKF Jacobian | closed form per link, `H = (dμ/dη)(η̃) z'` at the clipped `η̃`; no finite differences | every offered pair has one; a difference step is a tuned constant with cross-platform rounding |
| 3 | UKF points | Merwe scaled points under filterpy's names; `β = 2`, `κ = 0` from Wan and van der Merwe (2000), `α = 1` in place of their `1e-3`; and a scalar-index Gauss-Hermite rule, both built and measured, the closer one the default | `α = 1e-3` puts a weight of about `-1e6` at the centre; the full rule's accuracy on a one-index observation depends on `k` (section 3.3) |
| 5 | state | `kalman` gains configuration fields only, at the end, `serde(default)`; within unreleased schema 72 | no existing byte moves; the state stays `kalman`'s, bounded by `k` |
| 6 | oracles | filterpy's `ExtendedKalmanFilter` and `UnscentedKalmanFilter` (dev group already, 1.4.5, no bump), the UKF handed its prior each row | third-party wherever one computes the quantity |
| 7 | cost, smoother | EKF `O(k²)`, full UKF `O(k³)`, scalar-index UKF `O(k²)`; both compose with task 496's ring unchanged (linear transition), refused until 496 merges | the backward pass never reads the observation |
| 8 | platforms | every `exp`, `ln` and `powf` in the two filters from the `libm` crate, as `garch` does; `family.rs` generic over the math provider, `glm` bit-identical | the EKF's clipped streams amplify a last bit to 1e-10; the same bits everywhere need no tolerance |

(The first version's item 4, the particle filter's resampling, is gone with
it.)

## 1. The transition and observation families a spec can express

### 1.1 The model the two filters share

A row has features `z` (k, the intercept first under `fit_intercept`), a
target `y_j` per target, a weight `w` and a clock step `d`. Each target has
its coefficient state `β_j` (k). Both filters run on the same state-space
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
as one gap, so neither filter sees it as a step (task 539).

### 1.2 Transitions: linear in the state, additive Gaussian noise, only

**Decision.** The transition is `kalman`'s and nothing else: the random walk
with `Q D²` noise, and per-slot reversion. No nonlinear transition is offered.

**Reasons.** (1) The state is a regression's coefficients; no family of
nonlinear coefficient dynamics has been asked for. (2) With a linear
transition the EKF's and UKF's prediction step is exact (`P⁻ = Φ P Φ + Q`),
so the only approximation is in the update, and the UKF need not push sigma
points through `f` (study §5, read 1: filterpy's recomputed prior is the
only part of its gap to a direct filter). (3) The fixed-lag smoother of
task 496 reads only the transition, so it applies unchanged (section 7.3).

### 1.3 Observations: one index, a fixed set of families

**Decision.** A target is a scalar, observed through the single index
`η = z'β`, under one of:

| family | links | filters | note |
|---|---|---|---|
| gaussian | identity | linear (today's `kalman`), EKF, UKF | the EKF and UKF are the Kalman filter here, to rounding (study §4, §5) |
| gaussian | log, logit | EKF, UKF | a mean curve with constant variance: the "saturating" logistic `L/(1 + e^(-η))` with `L` as a scale on the target, an exponential trend. `obs_var` is required: the UKF reads `P` absolutely (section 5.1) |
| poisson | log | EKF, UKF | |
| binomial | logit | EKF, UKF | "logistic" in the classification sense; Bernoulli rows (no trial counts, study §11) |
| gamma | log, inverse | EKF, UKF | |
| tweedie, `var_power` p | log, power | EKF, UKF | |

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
- **Heavy-tailed (Student t) noise.** Its Gaussian approximation in a Kalman
  gain is a Gaussian with a larger variance, which `kalman` already is; it
  was offered only with the particle filter, and went with it.
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
  withheld and the row learns nothing, as `glm` does; it adds its clock to
  `D` as any row that observes nothing.
- **No finite differences.** Every offered pair has its derivative in
  `family.rs`. A difference step is a constant tuned to the data's scale,
  and its rounding differs by platform. If a future family has no closed
  form, its difference is measured against a neighbour's closed form and its
  error bounded in the test before it lands; none is needed now.
- **The covariance form is `kalman`'s own** update above (about `5k²`
  flops), not filterpy's Joseph matrix product. The study found the
  standard and expanded-Joseph forms part from filterpy's matrix product only
  on streams that reach the clip, where the EKF has already lost track
  (`b` 1.9e-7 and 7.1e-7 against a band of 3.6e-10, study §4, §9); the matrix
  form costs `4k³`. The tests hold well-conditioned streams at 1e-12
  relative, and clipped streams at 10 times the filter's own one-ulp band
  (study §10).

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
- **`alpha = 1`, not their worked `1e-3`** (accepted, the coordinator,
  2026-11-24). At `α = 1e-3` the centre weight is `1 − 1/α² ≈ −1e6` at every
  `k`, and the weighted sums lose about six digits to cancellation (study §8,
  D4; the loss was not measured beyond "no failure"). A small `α` also closes
  the points on the mean, which makes the UKF a second-order EKF and gives up
  the spread that kept it on track where the EKF was lost (study §8: 0 of 600
  streams against 15; the feature spikes of D2). At `α = 1`, `κ = 0` the
  centre weight is 0 at every `k`: no negative weight. It is the setting the
  study measured against filterpy and the exact posterior. The docstring
  gives this reason beside the default.

### 3.3 The full rule depends on `k` on a one-index observation (derived, not measured)

Every offered observation reads `β` only through `η = z'β`. The points then
matter only through their projections `z'(χ_i − b) = ±sqrt(k + λ) z'L_i`.
With `s² = z'P⁻z = Σ_i (z'L_i)²/(k + λ)`, the fourth central moment the
rule gives `η` is `Σ_i (z'L_i)⁴ / (k + λ)`, between `(k + λ) s⁴ / k` (the
projections spread evenly over the `k` columns) and `(k + λ) s⁴` (all on
one column); a Gaussian's is `3 s⁴`. Which it is depends on how `z` sits
against the Cholesky columns, so on the ordering of the features. At
`(1, 2, 0)`, `k + λ = k`, so the moment lies between `s⁴` and `k s⁴`: too
light by up to 3 times, or too heavy by up to `k/3` times.

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

**Decision (ruled, the coordinator, 2026-11-24).** Part 2a builds both
rules, measures them on the study's second oracle (the grid at `k = 2`, the
study's own numpy particle filter at `k = 5`, statsmodels' static posterior
at `k = 20`), and makes the one closer to the exact posterior the default;
the other stays an option. The measurement goes into the docstring.

### 3.4 Guards

- **A Cholesky that fails** (`P⁻` not positive definite in rounding): that
  row takes the EKF's update instead, on the same state. No jitter is added
  to `P`: a jitter is a constant tuned to the data that moves the fit.
- **The clip and the inverse links** as the EKF (section 2).

## 4. (The particle filter: dropped)

See the note at the top. The section numbers after this one are kept, so
that the citations of them in the exchange stay right.

## 5. The state and its layout

### 5.1 Options of `kalman`, no new state field

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
| `ukf_rule` | the measured default (section 3.3) | merwe, scalar_index |
| `alpha`, `beta`, `kappa` | 1, 2, 0 | filterpy's ranges |
| `ukf_noise` | expected | expected, plugin |

Under a family other than the Gaussian identity:

- **`P` is absolute and φ fixed.** The noise-units form (`P` over a learned
  noise, which cancels from the gain) holds for the EKF, whose gain does not
  read `P`'s scale through the linearization point, but not for the UKF,
  whose points spread by `P`'s absolute size. One rule for both: φ is the
  spec's `dispersion`, `obs_var` is refused (required for the Gaussian with
  a log or logit link), and a learned φ is left out (study §11).
- **The process noise is the explicit `q`** (η's scale per clock unit²).
  `coef_half_life` is refused there (ruled, the coordinator, 2026-11-24): its
  matching to exponential forgetting assumes a row's information is `z z'`
  over the noise, and a GLM row's is `W z z' / φ`, with `W` read at the
  prediction.
- **The intercept's prior is proper** (`p0`): the posterior after a single
  Poisson zero or binomial row under a diffuse intercept is improper.

### 5.3 Schema and fixtures

Schema 72 is unreleased, so the change rides on it: no `SCHEMA_VERSION`
bump. A `kalman` state of any schema loads with the new fields at their
defaults (the linear Gaussian filter), to the bit; no loader is needed. The
fixtures are regenerated at the merge (`PRINT_STATE_FIXTURES=1`, the
coordinator's step). New cases, each decoded by a test that holds its new
forms non-empty: `kalman_ekf_poisson`, `kalman_ukf_gamma_inverse`,
`kalman_ukf_gaussian_logit`, and one with the other UKF rule. If 72 ships
first, the change becomes 73, schema 72's sets are copied to `v72/` and the
loader is the `serde` defaults above.

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
the lock stays at 1.4.5 (ruled). Tolerance 2e-14 relative on `b`, 5e-14 on
`P` and `pred` (measured 1.7e-15 and 7.2e-15). The scalar-index rule is held
to `JulierSigmaPoints(k, 3 − k)` with the aligned `sqrt_method` (section
3.3), the same way.

Against the exact posterior (approximation error, a statistic of the
stream, on the study's seeds): the bounds of study §10's second table.

## 7. Cost per row, and the smoother

### 7.1 Arithmetic per row

| filter | an observing row | a row that observes nothing | transcendental calls an observing row |
|---|---|---|---|
| linear (today) | about `5k²` | `z'b` | none |
| EKF | about `5k²` | `z'b`, one link | one link, one `V` |
| UKF, full rule | about `k³/3 + 13k²` | the Cholesky and `2k + 1` links | `2k + 1` links and `V` |
| UKF, scalar-index rule | about `6k²` | `O(k²)` for `z'P⁻z` | 3 links and `V` |

The coordinator's "EKF `O(k³)`" holds for filterpy's Joseph matrix product,
not for a scalar observation with a diagonal `Φ`: there the EKF is
`kalman`'s own update, `O(k²)` (study §9). The UKF's `k³/3` is its Cholesky.
At `k = 20`: EKF about 2,100 flops and one `exp`; UKF about 8,500 and 41
(study §9). Part 2a adds bench cases and measures the `libm` cost (`garch`
measured its `exp` at 1.8 times Apple's, 3.2 ns).

### 7.3 The fixed-lag smoother (task 496)

Task 496's `smooth_lag` runs the RTS pass over a ring of the last `L`
observations' `(c_n, G_n, K_n)`, built from the filtered `(b_n, P_n)`, the
prior `P⁻_{n+1}` and `F_{n+1}`, at `O(L k²)` for the means and `O(L k³)`
with the covariances (its design, section 3.1).

The backward pass never reads the observation, and the transition is
linear, so the extended RTS smoother and the unscented one (Särkkä 2008;
with linear dynamics its cross-covariance is `P_n Φ'`, which makes its gain
RTS's) are task 496's pass on the filter's own stored steps. The smoothed
value is then exact RTS of an approximate filter; the filter's
approximation carries into it. **Decision (ruled):** `smooth_lag` is refused
with `filter ≠ "linear"` until task 496 has merged; a follow-up lifts the
refusal with one test against filterpy's `rts_smoother` on the stored steps.
`smooth_lag` is not on `main` yet, so the refusal is written where the two
branches meet.

## 8. The rules, held

1. **Out of sample (rule 2).** Each filter emits `pred` before reading `y`:
   the EKF's `μ̃`, the UKF's `ζ̄`. Held by the probe the study ran (a row's
   target and weight moved no prediction up to and including it, study §4).
2. **A zero-weight row (rule 9).** `kalman`'s path, unchanged (study §7:
   `b`, `P`, `n_eff` held to the bit, the run equal to the stream without
   those rows).
3. **No row counts (the user, 2026-11-22).** The clip and the missing mean
   are conditions on `η`; a failed Cholesky takes the EKF step. No warm-up
   count anywhere.
4. **Chunk invariance (rule 3).** Every quantity is per row.
5. **The same bits on every platform (ruled).** Every `exp`, `ln` and
   `powf` in the two filters is the `libm` crate's, as `garch`'s are
   (`garch/math.rs`): it is already linked, so nothing new is linked (rule
   12), and every state field is then bit-exact across platforms, with no
   tolerance in ulps to state. On clipped streams the filter amplifies a
   one-ulp difference to 1e-10 relative (study §4), which no tiny tolerance
   covers. `family.rs` gains a math parameter (its functions generic over an
   `exp`/`ln`/`powf` provider), so `glm` keeps the platform's functions and
   its outputs do not move: held by a digest of its outputs and states taken
   before the change.
6. **`n_eff` (rule 8)** is `kalman`'s weight sum before the row.
7. **f64, no `unsafe` (rule 6)**; the models in `rheon-core` behind
   `OnlineModel` (rule 4).

## 9. The coordinator's rulings (2026-11-24)

1. **The UKF's rule:** build both the full Merwe rule and the scalar-index
   Gauss-Hermite rule, measure both on the study's oracles, default to the
   closer one, and write the measurement in the docstring; `α = 1` accepted,
   with the centre-weight reason beside the default.
2. **`libm`** for every transcendental call in the filters; `family.rs`
   generic over the math provider, `glm`'s outputs bit-identical, proven by
   digest.
3. **The particle filter:** first sent to the user, then dropped from this
   release by the user (the note at the top).
4. filterpy stays at 1.4.5; `coef_half_life` refused under families other
   than the Gaussian; `smooth_lag` refused with a nonlinear filter until task
   496 merges. (The group-keyed random stream, also accepted, went with the
   particle filter.)

## 10. Part 2a: what is built and tested

`rheon-core`: `family.rs` generic over the math provider, with the digest;
the EKF and both UKF rules beside `kalman`'s update. `rheon-polars`: the spec
fields and their refusals (section 5.1). Python:
`po.spec.kalman(family=, link=, var_power=, dispersion=, filter=, ukf_rule=,
alpha=, beta=, kappa=, ukf_noise=)`. Tests: filterpy (6.1, 6.2), the second
oracle's measurement for the default, the zero-weight and out-of-sample
probes, the contract (`model_contract.rs`), chunk invariance, the fixtures
(5.3), bench cases at `k = 5` and `20`.

## 11. Not verified

- Every cost here is the study's flop count; no Rust was timed when this was
  written.
- Section 3.3 (the full rule's `k` dependence, the scalar-index rule and its
  filterpy equivalence) is derived on paper, not run.
- The cross-platform claims rest on `garch`'s precedent and the `libm`
  crate's being compiled from source, not on a run of these filters on
  macOS and Windows.
