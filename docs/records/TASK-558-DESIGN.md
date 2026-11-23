<!-- kind: record -->
# The online distribution fit, PLAN entry 558, part 1 (design note)

Building block B2 of exchange request F1. Part 1 is design only: no code
changes on this branch. Written 2026-11-23 at base 5878a47.

This note says "entry 558", never the citation form
`tests/test_plan_references.py` checks against the plan's task lines: the
plan has no line 558 yet, so that form would fail the essentials until it
does. Once the entry is written, the title may say it plainly.

**The task text is missing.** `docs/PLAN.md` at 5878a47 has no entry 558
(its last task is 548), and commit 1ecda80a, which the coordinator named, is
not in this repository. This note works from the coordinator's request
alone, quoted here:

> Fit a chosen family (normal, Student t, skew-t, GED, Beta-t and what the
> scan finds) by exponentially weighted maximum likelihood or moments; emit
> the parameters, quantiles and tail measures; share family code with
> garch. Part 1, design first, stop and reply.

The PLAN entry should be written (by the coordinator or the user) before
part 2 starts; question 1 below.

## 1. What `garch` already has

`crates/rheon-core/src/garch/dist.rs` holds the standardized innovation
densities (mean 0, variance 1) as `arch` writes them:

- normal (inline in the filter and fit, `w = z² − 1`);
- Student's t, `ν ∈ [2.05, 300]` (`NU_LO`, `NU_HI`, `arch`'s bounds);
- Hansen's (1994) skewed t, `λ ∈ [−0.99, 0.99]`; `λ = 0` is the t.

with `log_density(z, ν, λ)`, `scores(z, ν, λ, skew)` (the factor `w` of the
variance score and the closed shape scores `∂/∂ν`, `∂/∂λ`), and hand-written
`ln_gamma` and `digamma` (recurrence to `x ≥ 20`, asymptotic series). All
`ln`/`exp`/`ln_1p` go through `garch::math`, the `libm` crate (the user's
choice of 2026-11-22: the same bits on every OS). The estimator is a
recursive Newton (prediction-error) step `θ ← Proj_H(θ + a H⁻¹ g)` with `H`
the expected information (normal) or BHHH (t forms), `ν` stepped on
`ln(ν − 2)`, after a batch start fitted by projected Newton (`garch::optim`)
over a buffered start whose rows are a fixed count or clock span. There is
no GED in `garch` today, though `arch` has `GeneralizedError`.

`crate::family` is already taken by the GLM's exponential families
(variance function, link, deviance); the new shared module needs another
name (section 6).

## 2. Oracle scan

Checked in a throwaway env with the lockfile's versions (scipy 1.18.1,
arch 9.0.0, statsmodels 0.15.0; `arch` and `scipy` are already in the dev
group, `statsmodels` too) and from the R packages' reference pages.

| Library | Weighted fit? | What it does compute |
|---|---|---|
| `scipy.stats.<dist>.fit` | **No.** `t.fit(x, weights=w)` raises `TypeError: Unknown arguments` (verified); `scipy.stats.fit(dist, data, bounds, method=)` takes no weights either (signature verified). | `logpdf`, `cdf`, `ppf`, `expect` for `norm`, `t`, `gennorm` (GED), `jf_skew_t` (Jones-Faddy skew t), `skewnorm`, `nct`, `johnsonsu`, `norminvgauss`, `genhyperbolic`. |
| `arch.univariate.distribution` | No fit of its own (it is a GARCH innovation); `loglikelihood(params, resids, sigma2, individual=True)` returns **each row's** log density (verified for `GeneralizedError`). | `Normal`, `StudentsT`, `SkewStudent` (Hansen), `GeneralizedError` (`ν ∈ [1.01, 500]`); `cdf`, `ppf`, `moment`, `partial_moment`, `bounds`, `starting_values`. The standardization is the one `garch::dist` already matches. |
| `statsmodels` | `DescrStatsW(x, weights=w)`: weighted mean, variance, quantiles (real weights). `GenericLikelihoodModel.__init__` takes **no** `weights` (verified); a `loglikeobs` multiplied by `ω` is hand-plumbing. | The normal's exact EW fit. |
| R `sn::selm` | Has `weights`, but "Weights are supposed to represent frequencies, hence must be non-negative integers" (reference page). Exponential weights are not integers. | Azzalini SN / ST / SC. |
| R `gamlss` | `weights` "for a weighted likelihood analysis where the contribution of the observations to the likelihood differs according to weights" (reference page); whether non-integer weights are accepted is not stated. | NO, TF, ST1-ST5, SST, PE (GED), JSU, SHASH, BCT and more. |
| R `fGarch` | `stdFit`, `sstdFit`, `gedFit`, `sgedFit`: no weights argument as far as I know (not verified). | Fernández-Steel skewed t / GED. |
| R `betategarch` | Beta-t-EGARCH (Harvey's score-driven volatility); availability on CRAN not verified. | |

R is not installed on this machine and is not part of the test toolchain
(tests may use a library "declared in the dev group", which is Python's);
bringing R in is question 4.

**Conclusion: no third-party library computes the exponentially weighted
MLE of any non-normal family.** What third parties do compute is each row's
log density, CDF, quantile and partial moments. The oracle therefore is:

- the definition written by hand, from the paper: the objective
  `L(θ) = Σ_i ω_i log f(y_i; θ)` with `ω_i` the window's weights (section 3);
- each `log f` from the third-party library (`arch`'s `individual=True`
  log-likelihood for normal, t, skewed t, GED; `scipy.stats.<dist>.logpdf`
  as the second opinion where scipy has the same family: `t`, `gennorm`);
- its maximum found by a third-party optimizer (`scipy.optimize.minimize`,
  `trust-constr` with `arch`'s bounds), from several starts;
- quantiles and tail measures at the fitted parameters from `arch`'s `ppf`
  and `partial_moment` and `scipy.stats.<dist>.ppf` / `.expect`.

This is "write only the definition by hand; let the library compute it".
In Rust (`crate::oracle`), only the normal's EW moments have an oracle
already; for the others a Rust test would need a hand-written maximizer
over a hand-written density, so the Rust tests should test properties
(scores are the density's derivatives, the density is standardized, chunk
invariance, zero-weight rows, the start's fit is a stationary point) and
leave the third-party agreement to `tests/test_second_opinion*.py`.

## 3. The estimand: the weighted window each oracle checks

Per target `y`, rows `i ≤ t` learned (weight `w_i > 0`, target present), the
decay on the model's clock as everywhere else (`crate::Decay`, `λ_i` the
product of the factors since row `i`):

```text
ω_i = w_i · λ_i                     (under window_size: 0 for rows outside the window)
L_t(θ) = Σ_i ω_i log f(y_i; θ),     θ*_t = argmax over the family's admissible set
f(y; μ, σ, s) = f_z((y − μ)/σ; s) / σ,   f_z standardized: mean 0, variance 1
```

`W_t = Σ ω_i` is `n_eff` (before the row's update and decay, rule 8);
Kish's `n = W²/Σω²` is the `n` of every standard error and test below.
With `w ≡ 1` and no decay, `θ*` is the batch MLE that `arch`'s densities
and `scipy.optimize` reproduce exactly; the oracle test reads `ω_i` off the
same rows, stamps and decay with pandas' `ewm(times=)` weights (already the
temporal clock's oracle) and maximizes. A row of weight 0 contributes
nothing and moves no clock (rule 9).

Families parameterized standardized (`σ` is the standard deviation), as
`arch` and `garch` are, so the t needs `ν > 2`. Question 3 asks whether a
loc/scale form (scipy's, `ν > 0`, `σ` not the sd) is wanted as well.

## 4. The online estimator

There is no bounded sufficient statistic for the exact `θ*_t` of any family
here but the normal. Three options, measured against the rules:

1. **Recursive Newton / Fisher scoring** (`garch`'s method, and `glm`'s):
   per learned row, with the score `g_i = ∇ log f(y_i; θ_{i−1})` and the
   information `I_i`,

   ```text
   H ← λ H + w I_i
   θ ← Proj_H(θ + H⁻¹ w g_i)
   ```

   Each past row's score is frozen at the fit it was scored at: the
   recursive prediction-error estimate, which agrees with `θ*_t` to first
   order once started near it, and tracks it under decay with a lag of
   order `1/n_eff`. Bounded state (section 7), `O(p³)` a row, `p ≤ 4`.
   `I_i` is the **expected** (Fisher) information where it is closed
   (normal; t and GED in `(μ, ln σ, shape)` — closed forms in ψ and ψ′),
   i.e. a natural-gradient step; BHHH (`g gᵀ`) for Hansen's skewed t as
   `garch` does. The observed Hessian is not used: GED's location term has
   `|z|^(ν−2)`, unbounded at `z = 0` for `ν < 2`, so observed Newton is
   unstable exactly where GED is wanted.
2. **Moment matching**: exact for the window's definition with bounded
   state (EW mean and the central sums `M_2..M_4` by Pébay's update,
   already in `tails.rs`), shape solved from the implied kurtosis (t:
   `κ = 6/(ν − 4)`; GED: `Γ(5/ν)Γ(1/ν)/Γ(3/ν)² − 3`, a 1-D Newton) and,
   for the skewed t, skewness and kurtosis together (a 2-D solve). Its
   estimand is a different one (the moment estimator, not the MLE), it is
   undefined for the t with `ν ≤ 4` (the kurtosis is infinite and its EW
   estimate never settles), and it is inefficient in heavy tails. Useful as
   the **start's initial point** and as a fallback estimand the user may
   choose (`method = "moments"`); its oracle is `DescrStatsW` / scipy's
   weighted moments and `scipy.optimize.brentq` on the same equations.
3. **Exact MLE on a binned sketch**: keep an EW histogram of the
   standardized `z` (as `garch::guard` keeps 64 bins of `z`, or `EwQuantile`'s
   DDSketch buckets) and run Newton to convergence on the binned likelihood
   each row. The binned MLE is its own estimand (not `θ*`), the bins'
   standardization moves with `(μ, σ)`, and it costs `O(B · iters)` a row.
   Not recommended for part 2; listed because it is the only bounded way to
   a true Newton decrement of the whole objective.

**Recommendation: option 1, started by a batch fit, option 2 as the start's
initial point.** The start: the first rows are buffered (16 bytes a row,
`[y, w]`, capped by `max_state_bytes` as `garch`'s is) and fitted by
projected Newton with step halving (`garch::optim::maximize`, shared) from
the moment estimate, at their decayed weights; the buffer is replayed to
build `H`, then freed. Per the rule that early data is guarded by the
condition and not by a fixed row count, the start is **not** a row count:
it is tried on a geometric schedule (each time the buffer doubles, so `O(1)`
amortized per row) and accepted when its convergence and identifiability
checks pass (below); until then every output but `weight_sum` is withheld
with a reason. If the buffer's cap is reached before they pass, the fit is
accepted on the cap with the shape held at the simpler family (question 6).

### Convergence check (per row, after the start)

The recursive estimate never solves `∇L_t = 0`; what can be checked with
bounded state is whether it has caught up with the curvature it has seen.
Keep, decayed like `H`, the frozen-score sums `S_g = Σ ω g_i(θ_{i−1})`,
`S_H = Σ ω I_i` (this is `H`) and `S_Hθ = Σ ω I_i θ_{i−1}`. The quadratic
surrogate's gradient at the current `θ` is `G(θ) = S_g + S_H θ − S_Hθ`, and
its **Newton decrement** `δ² = Gᵀ H⁻¹ G` is twice the surrogate log-likelihood
a full Newton step would gain. Withhold (reason `not_converged`) while
`δ²` exceeds what the noise alone gives at this weight: `δ²` against the
`χ²_p` quantile scaled by `W/n_kish` (a score test of "θ is the surrogate's
maximum", whose null law is `χ²_p` with the Kish correction for unequal
weights), at a level the user sets (default question 7). A projection onto
a bound makes `G` non-zero by construction there, so `G` is read in the
free coordinates only.

### Identifiability check

- **Information conditioning**: withhold (reason `not_identified`) while
  `H`'s reciprocal condition number, after the diagonal scaling the step
  already uses, is below a floor (`1e-10`, the same order as `STEP_RIDGE`
  in `garch::series`), or `H` is not positive definite. This catches a
  constant stream, `σ → 0`, and a shape the data cannot tell.
- **The shape against the simpler family**: the t's `ν → ∞` and the skewed
  t's `λ → 0` are boundaries (or near-boundaries) where the shape is not
  identified from near-normal data. Parameterize the t's tail by `η = 1/ν`
  (`η ∈ [0, 1/2.05]`): the normal is then the point `η = 0`, where the
  information in `η` is finite and non-zero, so the Wald statistic
  `η̂² / [H⁻¹]_ηη` is regular and its null law is the boundary mixture
  `½χ²_0 + ½χ²_1` (Self and Liang 1987). Hansen's skew is regular at
  `λ = 0` (two-piece families have non-singular information at symmetry,
  unlike Azzalini's skew-normal, whose information is singular there —
  a reason to keep Hansen's). While the test does not reject the simpler
  family, the shape is reported with the reason `shape_not_identified`
  (question 8: report the simpler family's quantiles, or withhold the
  quantiles too). The test uses `H` and `θ`, no new state.
- **A second, out-of-sample opinion**: the decayed sum of the prequential
  log-score difference against the simpler family run beside it (the
  normal's exact EW fit, 3 words), `Σ ω (log f_family − log f_normal)` at
  each row's pre-update fits. It is emitted, not gated on.

Note that `garch` steps the t on `ln(ν − 2)` and would step it on `1/ν`
here: the density code is shared, the coordinate is not (question 9).

## 5. Per-family table

`p` parameters per target, `k` targets; state per target in 8-byte words
(section 7 gives the sum). Oracle: `arch` density + weighted sum by hand +
`scipy.optimize` unless stated.

| family | `θ` stepped | estimator | convergence | identifiability | outputs beyond `loc`, `scale` | oracle |
|---|---|---|---|---|---|---|
| `normal` | `μ, σ²` | **exact** EW Welford (as `EwCov`); no recursion | none needed (closed form) | `W > 0` and `M_2 > 0`; Kish `n ≥ 2` for a variance | — | `DescrStatsW` / `np.average`; `arch` `Normal`; `scipy.stats.norm.ppf` |
| `t` | `μ, ln σ, η = 1/ν` | recursive Fisher scoring, expected info (closed, ψ′) | Newton decrement on the surrogate | `H` conditioning; Wald on `η = 0` (½χ²₀+½χ²₁) | `nu`, `tail_index = ν` | `arch` `StudentsT`; `scipy.stats.t` second opinion (rescaled by `√((ν−2)/ν)`) |
| `skewt` (Hansen) | `μ, ln σ, η, λ` | recursive, BHHH (as `garch`) | as above | `H` conditioning; Wald on `λ = 0` and on `η = 0`; joint 2-df test against the normal | `nu`, `lambda` | `arch` `SkewStudent` (`ppf`, `partial_moment`) |
| `ged` | `μ, ln σ, ln ν` | recursive Fisher scoring, expected info (closed, ψ, ψ′) | as above | `H` conditioning; Wald on `ν = 2` (interior: the normal), plain `χ²_1` | `nu` (2 normal, 1 Laplace) | `arch` `GeneralizedError`; `scipy.stats.gennorm` second opinion (`β = ν`, scale `σ √(Γ(1/ν)/Γ(3/ν))`) |
| `beta_t` | — | see below | | | | none in Python; R `betategarch` (unverified) or Harvey (2013) by hand |

**Beta-t is not a static family.** Harvey and Chakravarty's Beta-t-EGARCH
is Student's t innovations with a score-driven (DCS) log-scale,
`ln σ_{t+1} = ω + φ ln σ_t + κ u_t`, `u_t = (ν + 1) z²/(ν − 2 + z²) − 1`,
whose `b = z²/(ν − 2 + z²)` is Beta(½, ν/2)-distributed. Its static
distribution is the t, and `u_t` is exactly `garch::dist::scores`'s `w`. As
a *static* fit it adds nothing to `t`; as a *dynamic* one it is a new
`garch` process (`vol = "beta_t"` beside `egarch`), not part of this model
(question 5).

**What the scan adds** (candidates, not proposed for part 2 without the
user's word, question 2): Johnson's SU (`scipy.stats.johnsonsu`, gamlss
`JSU`; closed-form quantiles, four parameters, skew and tails separately),
Fernández-Steel's skewed GED (fGarch `sged`, no Python oracle), the
normal inverse Gaussian (`scipy.stats.norminvgauss`; its density needs a
Bessel `K₁` the crate does not have), and Jones-Faddy's skew t
(`scipy.stats.jf_skew_t`, which would give the skewed t a scipy oracle of a
different family).

## 6. Sharing family code with `garch`

Move `garch/dist.rs` and `garch/math.rs` up to a crate module
(`crate::innovation`, since `crate::family` is the GLM's), behind which each
family is a standardized density with:

```text
log_density(z, shape)          scores(z, shape) -> (w, ∂/∂shape)
expected_info(shape)           (where closed; BHHH otherwise)
cdf(z, shape), ppf(p, shape)   lower partial moment ∫_{−∞}^{q} z f(z) dz  (for ES)
shape bounds and the coordinate each model steps
```

`garch` keeps calling the same functions, so the move is **bit-identical
for `garch`** (its golden and its state fixtures unchanged; the test:
`garch`'s goldens and the cross-OS digests re-run with no diff). It also
gives `garch` GED for free later (`arch`'s `GeneralizedError`), a separate
task. The new model and `garch` then share the `libm`-crate math and its
cross-OS guarantee.

New special functions the families need and the crate lacks:

- trigamma `ψ′` (the t and GED expected information): the same recurrence
  plus asymptotic series as `digamma`;
- the normal CDF: `libm::erfc` (already linked); its inverse: the existing
  `conformal::norm_ppf` (moved or re-exported);
- the t's CDF and quantile (Hansen's skewed t's too, which is a two-piece
  rescaled t): the regularized incomplete beta and its inverse;
- GED's CDF and quantile: the regularized incomplete gamma and its inverse.

The last two are not in `libm`. Either they are written by hand (continued
fractions, Lentz; Newton/Halley inversion; tested against `scipy.special`
`betainc`, `betaincinv`, `gammainc`, `gammaincinv`) or a crate such as
`statrs` is added, which would statically link a new library into the wheel
and the CLI (rule 12: question 10). The quantile is per row per level, so
its cost matters: a Halley inversion is a few incomplete-beta calls.

## 7. Outputs, naming and state size

Model name `dist_fit` (question 11), config: `family` (`normal`, `t`,
`skewt`, `ged`), `decay` (the crate's `Decay`), `min_weight`, `quantiles`
(levels, default `[0.01, 0.05, 0.95, 0.99]`), `es_levels` (default
`[0.01, 0.05]`, lower tail; upper by symmetry of the call), `demean` (fit
`μ`, default true; false fixes `μ = 0` as `garch` does),
`max_state_bytes`, `test_level`. Per target, each **before** the row is
learned (rule 2):

| field | meaning |
|---|---|
| `loc`, `scale` | `μ`, `σ` (the sd) |
| `nu`, `lambda` | the shape, `arch`'s names (as `garch`'s `coefficients`) |
| `q_<level>` | `μ + σ F_z⁻¹(level)` |
| `es_<level>` | expected shortfall `E[y | y ≤ q]` from the lower partial moment |
| `skew`, `excess_kurtosis` | implied by the shape (NaN where the moment does not exist, `ν ≤ 3`, `ν ≤ 4`) |
| `pit` | `F(y_t)` under the fit before the row: the calibration stream |
| `log_score` | `log f(y_t)` under that fit, and `log_score_vs_normal` (section 4) |
| `weight_sum` | `n_eff`, the shared weight |
| `withheld_reason` | `start`, `not_converged`, `not_identified`, `shape_not_identified`, `min_weight` (the bank's existing field) |

**State per target** (8-byte words, `p` stepped parameters): `θ` `p`; `H`
`p(p+1)/2`; the surrogate's `S_g` `p` and `S_Hθ` `p`; `W`, `Σω²`; the
normal beside it (mean, `M_2`, its `W` shared) 2; the clock 2; plus the
start's buffer, 16 bytes a row until fitted, capped by `max_state_bytes`.

| family | `p` | words | bytes after the start |
|---|---|---|---|
| normal | 2 (exact form: mean, `M_2`) | 6 | 48 |
| t, ged | 3 | 3 + 6 + 6 + 6 = 21 | 168 |
| skewt | 4 | 4 + 10 + 8 + 6 = 28 | 224 |

So `k` targets cost `k` times that plus 16 for the model, and the start's
buffer `16 · rows` per target until fitted (32 KB at 2,000 rows), the same
as `garch`'s. No growth with the stream.

**Under `window_size`** the exact windowed MLE needs the window's rows: a
ring of `[y, w, stamp]`, 24 bytes a row, and a refit cost of `O(N)` a row.
A recursive form needs each row's frozen score and information to subtract
when it leaves: `8 (2 + p + p(p+1)/2)` bytes a row, 136 for the skewed t, a
window of 10,000 rows 1.3 MB per target. Both are bounded by the window
but well above today's caps, so the proposal is **no `window_size` in part
2** (decay only), raised as question 12 per the state-size rule.

**libm in the state.** `θ`, `H` and the sums carry `ln`, `exp`, `ln_1p`,
`ln Γ`, `ψ`, `ψ′` results (and the start's fit). Through the `libm` crate
they are the same bits on every OS (the `garch` precedent), so no cross-OS
tolerance is needed; question 13 confirms the user wants the `libm` crate
here too, at its 1.4x/1.8x cost on arm64 (measured for `garch`).

**State schema.** Schema 72 is unreleased: a new `ModelState` variant
(its name frozen, it is the file's tag) with frozen fixtures for every
form (each family, before and after the start, `demean` both ways), no
bump. The `garch` module move changes no serialized layout.

## 8. Rules checked

- Model in `rheon-core` behind `OnlineModel`, no polars or Python; f64; no
  `unsafe`.
- Out of sample: every output is read at the fit before the row's update.
- Zero-weight row: predicted (`pit`, `log_score`, quantiles), learned not
  at all, clock not moved; the first row of weight 0 leaves `W = 0` and the
  normal's update guards its `0/0`.
- `n_eff`: `W` before the row's update and decay; `min_weight` per target.
- Chunk invariance: row-at-a-time recursion; the start's schedule depends
  on rows learned, not chunks.
- Bounded state, one pass (the start's buffer is the one exception, capped
  as `garch`'s is, and freed).
- No fixed row-count guard for early data: the start is accepted by its
  checks, not a count.
- Reader-facing docs (`docs/MODELS.md`, the docstrings) will cite the
  papers (Hansen 1994; Self and Liang 1987; Harvey 2013), not this note.

## 9. Open design questions for the user

1. There is no PLAN entry 558 (last is 548) and commit 1ecda80a is not here.
   Who writes the task text, and does it match this note's scope?
2. The family set for part 2: normal, t, Hansen skewed t, GED. Add Johnson
   SU, skewed GED or NIG now, later, or never?
3. Standardized parameterization only (`σ` the sd, `ν > 2` for the t, as
   `arch` and `garch`), or also scipy's loc/scale (`ν > 0`)?
4. Oracles: Python only (`arch` densities + `scipy.optimize`, the weighted
   sum by hand), or bring R (`gamlss` with weights) into the test
   toolchain as a second opinion?
5. Beta-t: confirm it is a `garch` process (score-driven scale with t
   innovations) for a separate task, not a family here.
6. A start whose checks have not passed when its buffer hits the cap: fall
   back to the simpler family's fit, or keep withholding?
7. The convergence test's level (and the identifiability test's): one
   `test_level` (default 0.05?) or two settings?
8. While the shape is not identified: withhold only the shape, or also
   the quantiles and ES (or report the simpler family's)?
9. Step the t's tail on `1/ν` here (regular at the normal) while `garch`
   keeps `ln(ν − 2)`? Or move `garch` to `1/ν` too (outputs move)?
10. The incomplete beta and gamma functions and their inverses: hand-write
    them (tested against `scipy.special`), or add `statrs` (a new
    statically linked library, rule 12)?
11. Names: `dist_fit` for the model; `loc`/`scale`/`nu`/`lambda`,
    `q_<level>`, `es_<level>`, `pit`, `log_score`?
12. `window_size`: leave it out (proposed), or accept 24-136 bytes a row of
    window and which form (ring and refit, or recursive downdate)?
13. The `libm` crate for this model's math (same bits on every OS, slower),
    as for `garch`?
14. Moment matching as a user-selectable estimand (`method = "moments"`),
    or only as the start's initial point?
15. The location: fit `μ` by default, with `demean = false` for returns?

## 10. Not verified

- No PLAN entry 558; commit 1ecda80a absent. The request's intent beyond
  the quoted sentence is unknown.
- R: not installed. `sn::selm`'s integer-only weights and `gamlss`'s
  weighted likelihood are from their reference pages; whether `gamlss`
  accepts non-integer weights is not stated there. `fGarch` having no
  weights and `betategarch` being on CRAN are from memory.
- The closed forms of the t's and GED's expected information in
  `(μ, ln σ, shape)` are from memory, not yet derived and checked against
  a numerical Fisher information; Hansen's skewed t's is assumed not closed.
- Self and Liang's mixture for `η = 1/ν` at 0, and the regularity of
  Hansen's information at `λ = 0`, are from the literature as recalled,
  not simulated here.
- The Newton-decrement test's `χ²_p` law with the Kish correction under
  the recursive estimator is a proposal; its size needs a seed study in
  part 2 before it gates anything.
- Costs (per row, the quantile inversions, the `libm` crate) are not
  measured; the state sizes are counted, not measured.
- The dev env was not synced in the checkout; the API facts above come from
  a throwaway env at the lockfile's versions.
