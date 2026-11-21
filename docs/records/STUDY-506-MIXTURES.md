# Study 506: Gaussian mixtures by online EM on `hmm` (parts 1 and 2 of 3)

Question: can a Gaussian mixture be fit by exponentially weighted online EM
(Cappé and Moulines 2009, sufficient statistics), built as an `hmm` whose
transition rows are all equal, and does it reach scikit-learn's batch
`GaussianMixture` on stationary data? Part 1: the recursion, the `hmm`
mapping, and the stationary comparison at k = 2 features, K = 2 components.

Script: `scripts/study_506.py` (numpy, scipy, scikit-learn 1.9.1; seeded
generators, no data files). `python scripts/study_506.py` runs everything in
about 80 s; `--quick` runs N = 2000 only in about 10 s. Notation: k features,
K components (rheon's `k` is the feature count).

Status: part 1 done; part 2 done (sections 6 to 11, script
`scripts/study_506_part2.py`) except two cells of section 7 (full
covariance, K = 8, N = 20000, k = 5 and 20), not finished in the time box:
`python scripts/study_506_part2.py sklearn full 5` (and `full 20`) fills
them. Part 3 is not started.

## 1. The model and the recursion

p(x) = Σⱼ πⱼ N(x; μⱼ, Σⱼ), full covariances.

Row t, features xₜ, weight wₜ ≥ 0, λ = the decay over the row's clock
(λ = 1: no forgetting). Parameters are read from the state **before** the row
(so the row's responsibilities and log-likelihood are out of sample, hard
rule 2):

```text
E:  rₜⱼ = πⱼ N(xₜ; μⱼ, Σⱼ) / Σₗ πₗ N(xₜ; μₗ, Σₗ)      (log-sum-exp about the largest term)
    S0ⱼ ← λ S0ⱼ + wₜ rₜⱼ
    S1ⱼ ← λ S1ⱼ + wₜ rₜⱼ xₜ
    S2ⱼ ← λ S2ⱼ + wₜ rₜⱼ xₜ xₜᵀ
M (read on demand, never stored):
    πⱼ = S0ⱼ / Σₗ S0ₗ
    μⱼ = S1ⱼ / S0ⱼ
    Σⱼ = S2ⱼ / S0ⱼ − μⱼ μⱼᵀ + reg_covar · I
```

**State**, per component: S0ⱼ (1), S1ⱼ (k), S2ⱼ (k(k+1)/2, symmetric):
`K·(1 + k + k(k+1)/2)` f64, plus the fallback parameters below if they are
kept. Σₗ S0ₗ is the model's `n_eff` (the responsibilities sum to 1, so it is
`λ·n_eff + w`, the shared recursion of hard rule 8).

**The mean form** (what `hmm`'s `EwCov` holds, and the one to build: it has
no `S2/S0 − μμᵀ` cancellation). With wⱼ = w rⱼ, W' = λW + wⱼ, a = λW/W',
b = wⱼ/W', δ = x − μ:

```text
μ' = μ + b δ,    C' = a C + a b δ δᵀ,    W' as above
```

S0 = W, S1 = Wμ, S2 = W(C + μμᵀ). The script runs both forms on every
comparison stream: their parameters agree to 6e-14 at N = 2000 and 4e-13 at
N = 20000, after one pass.

**Guards** (hard rule 9):
- w = 0 (or a refused row): no E-step at all; S0, S1, S2 (W) decay by λ and
  nothing is added. μ and C, being ratios, do not move (the mean form does
  not touch them).
- S0ⱼ = 0 (a component never attributed, or an empty state): μⱼ and Σⱼ read
  the fallback (the initial) parameters; Σ S0 = 0 reads the fallback weights.
  In the mean form W' ≤ 0 skips the component's update.
- The densities are combined in log space about the largest, so a row far
  from every component gives a finite posterior, not 0/0.
- reg_covar > 0 keeps Σⱼ factorizable when C is rank-deficient (a component
  that has seen fewer than k+1 distinct rows).

**Per-row cost.** E-step: per component one k×k Cholesky (k³/6) and one
triangular solve (k²/2), or O(k²) with the factor held and moved by a rank-1
update (`hmm`'s factor form, used at k ≥ 13); K exps and logs. Update:
O(K k²) (a rank-1 term per component). So O(K k³) refactoring, O(K k²) with
held factors. At k = 2, K = 2 it is about 60 flops, 2 exp, 2 log, 2 sqrt per
row (Python measures 60 to 70 µs a row, all interpreter overhead; no claim
about a Rust row).

## 2. The `hmm` special case

`hmm` (crates/rheon-core/src/hmm.rs) filters p̃ₗ = Σₖ pₖ Πₖₗ, then
pₗ ∝ p̃ₗ fₗ, and feeds each state's `EwCov` the row at weight w·pₗ. Set every
row of Π to the same π:

- p̃ₗ = Σₖ pₖ πₗ = πₗ, whatever the previous posterior: the filtered posterior
  **is** the mixture responsibility rₜₗ. **Exact.**
- State l's `EwCov` is the mean form above: `w_sum` = S0ₗ, `m` = S1ₗ/S0ₗ,
  `c` (packed centred co-moments) = S2ₗ/S0ₗ − μμᵀ. **Exact** (algebraically;
  6e-14 numerically here). `hmm`'s `n_eff` = Σ S0. Its order (read the
  densities, then update; the zero-weight row ages the states by `update(x,
  λ, 0)`, which moves no mean or co-moment) is this recursion's.
- The weights: **not exact in today's `hmm`.** `hmm` learns Π from the joint
  counts Aₖₗ ← λAₖₗ + w ξₖₗ and reads each row separately,
  Πₖₗ = (Aₖₗ + τ)/Σₗ(Aₖₗ + τ). Under equal rows ξₖₗ = pₖ(t−1) rₜₗ, so the
  column sums Σₖ Aₖₗ equal S0ₗ, but the per-row normalisation gives K
  different rows that are not π. An `iid` form must read
  πₗ = (W_l + τ)/(n_eff + Kτ) from the states' weights instead and keep no A
  (K² cells fewer); with τ = 0 it is this recursion exactly (with the
  Σ S0 = 0 guard), with τ > 0 a Dirichlet-smoothed π that scikit-learn has
  no counterpart for.
- The ridge: **not exact.** `hmm` adds `precision_prior · precision_scale`,
  where `precision_scale` is the product of the state's mixing factors a
  since its first row, so the ridge fades like W₁/W (a prior). scikit-learn's
  `reg_covar` is a fixed floor. At reg = 1e-6 against variances near 1 the
  difference is numerically invisible here, but the two are different
  definitions; a fixed `reg_covar` would be a new option (the scan notes
  task 525 raises the same for `ew_class`).
- The start: `hmm` seeds from `warm_rows` buffered rows (k-means) or takes
  `means=`/`covs=` at weight 1. This study shares scikit-learn's own
  initialisation instead (section 3), entered at weight 1 like `means=`.

So the mixture is `hmm` with a new weight read (and no transition counts);
the E-step and the per-state statistics are `hmm`'s unchanged.

## 3. The comparison

Initialisation, shared: scikit-learn's own `init_params='kmeans'` step,
reproduced: `KMeans(n_clusters=2, n_init=1, random_state=seed)` on the N
rows, hard labels, one M-step. Those weights, means and covariances (+
reg_covar) are passed to `GaussianMixture(covariance_type='full',
reg_covar=1e-6, tol=1e-12, max_iter=5000, weights_init, means_init,
precisions_init)`, and the online fit starts from the same parameters at
weight W0 = 1 (S0ⱼ = πⱼ). Using all N rows for the k-means start is not
streaming; it is shared here only so the two fits differ in the EM alone.

Data: two components in k = 2, π = (0.35, 0.65), Σ₁ = [[1, .5], [.5, 1]],
Σ₂ = [[1.5, −.4], [−.4, .8]]; **separated**: μ₂ = (4, 3) (μ₁ = 0);
**overlapping**: μ₂ = (1.5, 1). Online at λ = 1, w = 1, over the same N rows
in order: after one pass; after a second pass with the statistics carried on
(pass 1 keeps half the weight); after a second pass restarted from pass 1's
parameters at W0 = 1. Components matched by the Hungarian assignment on
squared mean distances.

Units: batch standard errors, with Nⱼ = N πⱼ: mean sqrt(Σₐₐ/Nⱼ); covariance
entry sqrt((ΣₐₐΣ_bb + Σₐ_b²)/Nⱼ) (Gaussian); weight sqrt(π(1−π)/N). Each
figure is the max over components and entries (4 means, 6 covariance
entries, 1 weight), then median / max over 5 seeds. These SEs assume known
labels, so they understate the true SE when components overlap; the
`batch vs truth` row calibrates them.

| scenario | N | row | means (med / max) | covariances | weights |
|---|---|---|---|---|---|
| separated | 2000 | batch vs truth | 2.26 / 3.17 | 1.89 / 2.53 | 0.89 / 1.58 |
| separated | 2000 | online, 1 pass | **0.48 / 0.59** | **0.64 / 0.90** | **0.32 / 0.36** |
| separated | 2000 | online, 2 passes carried | 0.33 / 0.43 | 0.47 / 0.64 | 0.22 / 0.25 |
| separated | 2000 | online, pass 2 restarted | 0.50 / 0.59 | 0.66 / 0.89 | 0.31 / 0.36 |
| separated | 20000 | batch vs truth | 1.61 / 2.56 | 1.41 / 1.61 | 0.66 / 2.71 |
| separated | 20000 | online, 1 pass | **0.57 / 0.74** | **0.84 / 0.97** | **0.31 / 0.44** |
| separated | 20000 | online, 2 passes carried | 0.38 / 0.52 | 0.55 / 0.71 | 0.20 / 0.31 |
| separated | 20000 | online, pass 2 restarted | 0.56 / 0.72 | 0.84 / 0.94 | 0.30 / 0.43 |
| overlapping | 2000 | batch vs truth | 3.26 / 5.97 | 2.04 / 4.16 | 3.26 / 6.77 |
| overlapping | 2000 | online, 1 pass | 6.68 / 8.41 | 6.14 / 10.31 | 4.96 / 6.86 |
| overlapping | 2000 | online, 2 passes carried | 6.39 / 8.27 | 5.67 / 9.61 | 4.83 / 6.65 |
| overlapping | 20000 | batch vs truth | 6.22 / 8.11 | 4.37 / 6.88 | 7.21 / 7.76 |
| overlapping | 20000 | online, 1 pass | 14.15 / 25.08 | 9.51 / 13.34 | 15.16 / 44.66 |
| overlapping | 20000 | online, 2 passes carried | 14.22 / 24.16 | 8.91 / 12.51 | 14.74 / 42.88 |
| overlapping | 20000 | online, pass 2 restarted | 20.58 / 48.44 | 17.68 / 34.36 | 22.04 / 85.42 |

Absolute, one pass, max over seeds: separated N = 2000 means 0.023, covariance
entries 0.042, weight 0.0038; N = 20000: 0.0089, 0.016, 0.0015. Overlapping
N = 2000: 0.34, 0.54, 0.073; N = 20000: 0.29, 0.23, 0.15.

Repeated carried-on passes, N = 2000, median of the max gap (SE) and of the
batch's mean log-likelihood lead per row:

| scenario | pass | means | covariances | weights | loglik lead |
|---|---|---|---|---|---|
| separated | 1 | 0.48 | 0.64 | 0.32 | 2.8e-4 |
| separated | 2 | 0.33 | 0.47 | 0.22 | 1.5e-4 |
| separated | 4 | 0.23 | 0.34 | 0.15 | 7.8e-5 |
| separated | 8 | 0.15 | 0.24 | 0.10 | 3.5e-5 |
| overlapping | 1 | 6.68 | 6.14 | 4.96 | 8.5e-3 |
| overlapping | 8 | 5.83 | 4.78 | 4.52 | 4.9e-3 |

Reading:
- **Separated components: yes.** One pass lands within one batch SE of
  scikit-learn's fit in every parameter, at both N, and the gap in SE units
  holds steady as N grows by 10 (so the absolute gap shrinks like 1/√N).
  Carried-on passes approach batch EM's fixed point at about 1/passes (the
  log-likelihood lead halves each doubling); restarting the statistics at
  weight 1 buys nothing over one pass.
- **Overlapping components: no, not within the batch SE.** One pass is
  5 to 15 SE (median) from batch EM, the gap in SE units grows with N, and
  eight passes close little of it. This is the known slow convergence of
  online EM at step 1/t when the fraction of missing information is high: the
  stochastic-approximation error from early attributions decays like
  t^−(1−ρ), ρ the EM map's contraction rate, near 1 here. The batch fit itself
  sits 3 to 8 known-label SEs from the truth, so the problem is weakly
  identified at these N. Any oracle test should use separated components and
  say why (as the scan proposed); a ledger band for overlap would be loose.

## 4. Zero-weight rows

Both forms, from `check_zero_weight`:
- an empty state (W0 = 0), its first row of weight 0: S0, S1, S2 (and W) stay
  exactly zero, no NaN; 49 rows after it give finite parameters.
- λ = 1, a zero-weight row after every third row: the state is bit-identical
  to the stream without them.
- λ = 0.99: a zero-weight row only decays the statistics; the parameters it
  reads move by at most 1.1e-16 (the weights' S0ⱼ/ΣS0 after both are scaled
  by λ; the mean form's μ and C are not touched).

## 5. State size and cost at k = 2, K = 2

The mixture state proper: K·(1 + k + k(k+1)/2) = 2·6 = 12 f64 = **96 bytes**
(plus `n_eff`, and 12 more f64 if the initial parameters are kept as the
S0ⱼ = 0 fallback). Generally `8·K·(1 + k + k(k+1)/2)` bytes, the scan's
figure. For comparison, today's `hmm` at the same shape keeps per state
`w_sum`, `q_sum`, `prior_scale`, `precision_scale`, `m` (k) and the packed
`c` (3): 9 f64; plus `p` (K), the counts `A` (K²) and `n_eff`: about 25 f64,
200 bytes, before the encoding's framing. An `iid` form drops the K² counts.

Per row: about 60 flops, 2 exp, 2 log, 2 sqrt (section 1).

## Open questions

1. Weight read for the `iid` form: τ = 0 (scikit-learn's estimate, needs the
   Σ S0 = 0 guard) or keep `transition_prior` as a Dirichlet smoothing?
2. Ridge: `hmm`'s fading `precision_prior` or a fixed `reg_covar` floor? They
   agree at no decay only as reg → 0; under decay they differ in kind.
3. Initialisation in a stream: `warm_rows` k-means (as `hmm`) versus the
   batch k-means shared here; part 2 should measure the cost of the streaming
   start on separated data.
4. Overlapping components: is a step-size schedule (t^−α, α < 1, with Polyak
   averaging, as Cappé and Moulines recommend) worth offering beside λ, or is
   the separated-component guarantee enough to document?

# Part 2: decay, larger k and K, collapse, zero-weight rows, state

Script: `scripts/study_506_part2.py`, one subcommand per section (`decay`
~70 s, `sklearn` ~6 min, `collapse` ~45 s, `zero` and `state` ~1 s; run the
long ones under `nohup`). The recursion is part 1's sum form vectorised over
components, with a diagonal option (S2ⱼ keeps the k squares only), and the
mean form for the collapse case. Data: `make_mixture(k, K, seed)`, K
components with covariance eigenvalues in [0.5, 1.5], means at least 4
apart (3.3 or more standard deviations, near part 1's separated pair),
weights ∝ 1 + j/(K−1). Overlapping components are not run here: part 1
showed they are a limit (5 to 15 SE), not a pass. A first run with means at
least 8 apart had every responsibility 0 or 1 to rounding, and the online
fit equalled the decay-weighted longhand to 1e-15: online EM is exact when
the E-step is a hard assignment, so the informative regime is the one used.

## 6. Decay against the weighted EM longhand

Oracle (`weighted_em`): batch EM over all N rows with fixed row weights
vₜ = λ^(N−1−t), responsibilities recomputed each iteration from the current
parameters, run until no parameter moves by more than 1e-11; at λ = 1 it
equals scikit-learn's fit to 8.9e-16. Both start from the same k-means
parameters (part 1's shared start; the online fit enters them at W0 = 1,
whose weight λ^N·W0 is at most 0.14 of the fit's 865 at λ = 0.999,
N = 2000, and nothing at the others). SE units are the batch SEs of part 1
with n the Kish size of the decay-weighted sample, (Σv)²/Σv²: 1523 and
1999 at λ = 0.999 (N = 2000, 20000), 199 at λ = 0.99. Gap at the stream's
end, K = 2, full covariance, max over entries, median / max over 5 seeds:

| k | λ | N | means | covariances | weights | with stabilised τ = 0.01: means / cov / weights (median / max) |
|---|---|---|---|---|---|---|
| 2 | 0.999 | 2000 | 0.012 / 0.23 | 0.028 / 0.28 | 0.004 / 0.13 | 0.011 / 0.20, 0.027 / 0.25, 0.005 / 0.12 |
| 2 | 0.999 | 20000 | 0.017 / 0.22 | 0.031 / 0.31 | 0.008 / 0.12 | the same to 2 digits |
| 2 | 0.99 | 2000 | 0.020 / 0.17 | 0.065 / 0.27 | 0.005 / 0.08 | the same to 2 digits |
| 2 | 0.99 | 20000 | 0.081 / 0.18 | 0.13 / 0.32 | 0.044 / 0.12 | the same to 2 digits |
| 5 | 0.999 | 2000 | 0.010 / **24.8** | 0.018 / **68** | 0.004 / **11.1** | 0.010 / 23.3, 0.018 / 64, 0.004 / 10.4 |
| 5 | 0.999 | 20000 | 0.015 / 0.038 | 0.029 / 0.060 | 0.007 / 0.019 | the same |
| 5 | 0.99 | 2000 | 0.0009 / **5.9** | 0.0024 / **14.3** | 0.0003 / **2.9** | 0.0010 / **0.31**, 0.0026 / 0.68, 0.0012 / 0.16 |
| 5 | 0.99 | 20000 | 0.0003 / 0.025 | 0.0008 / 0.038 | 0.0001 / 0.012 | 0.0007 / 0.025, 0.0019 / 0.038, 0.0012 / 0.013 |

Reading: **under decay the online fit tracks the longhand within 0.32 SE
of the decay-weighted sample** in every cell that reached the longhand's
fixed point (median 0.0003 to 0.13 SE). The two bold maxima are one seed
each at N = 2000 where the online path and the longhand's iteration ended at
different fixed points: at λ = 0.999 the window still holds the start (the
early rows, read against a W0 = 1 start, fixed a different attribution); at
λ = 0.99 a component was briefly starved, and the stabilised τ (section 8)
kept it alive and brought that seed to 0.31 SE. Neither recurs at N =
20000, where the start has decayed away. The gap does not grow with
forgetting: it is a constant-step stochastic approximation whose error is
a fraction of the window's own SE.

## 7. Larger k and K against scikit-learn

λ = 1, one pass, against `GaussianMixture(covariance_type = 'diag' | 'full',
reg_covar = 1e-6, tol = 1e-10)` from the same k-means start (part 1's
shared start; `best5 same` counts the seeds where scikit-learn's own best of
`n_init = 5` starts reached the same fit, so the shared start is not a
worse local optimum). Components matched by the Hungarian assignment on
means. The online fit starts at W0 = 1 row (part 1) and at **W0 = 20·K
rows** (20 rows per component). Max over entries, median / max over 5 seeds,
in batch SE (n = N):

| cov | k | K | N | W0 = 1: means / cov / weights (median / max) | W0 = 20K: means / cov / weights (median / max) | best5 same |
|---|---|---|---|---|---|---|
| diag | 5 | 2 | 4000 | 0.0006 / 0.57, 0.0017 / 1.1, 0.0002 / 0.34 | 0.0007 / 0.17, 0.0018 / 0.27, 0.0002 / 0.10 | 5/5 |
| diag | 5 | 2 | 20000 | 0.0068 / 0.062, 0.014 / 0.12, 0.0013 / 0.041 | 0.0066 / 0.059, 0.013 / 0.097, 0.0009 / 0.039 | 5/5 |
| diag | 5 | 4 | 4000 | 0.35 / 66, 0.58 / 186, 0.26 / 19 | **0.16 / 0.25, 0.26 / 0.36, 0.064 / 0.15** | 5/5 |
| diag | 5 | 4 | 20000 | 0.42 / 1.0, 0.70 / 3.4, 0.22 / 0.45 | **0.084 / 0.15, 0.13 / 0.21, 0.040 / 0.063** | 5/5 |
| diag | 5 | 8 | 4000 | 67 / 116, 76 / 174, 18 / 50 | 0.34 / 11, 0.58 / 10, 0.12 / 8.8 | 4/5 |
| diag | 5 | 8 | 20000 | 20 / 197, 116 / 176, 25 / 276 | 0.37 / 33, 0.47 / 96, 0.27 / 140 | 4/5 |
| diag | 20 | 2 | 4000 | 8e-14 / 1e-13 (all) | 3e-13 / 6e-12 (all) | 5/5 |
| diag | 20 | 2 | 20000 | 2e-12 / 7e-11 (all) | 7e-12 / 7e-11 (all) | 5/5 |
| diag | 20 | 4 | 4000 | 1e-12 / 6e-12 (all) | 2e-12 / 6e-12 (all) | 5/5 |
| diag | 20 | 4 | 20000 | 4e-11 / 3e-7 (all) | 3e-11 / 3e-7 (all) | 5/5 |
| diag | 20 | 8 | 4000 | 30 / 86, 10 / 154, 20 / 24 | 6e-8 / 2.6, 2e-7 / 5.3, 2e-8 / 2.6 | 4/5 |
| diag | 20 | 8 | 20000 | 142 / 432, 324 / 817, 44 / 110 | **2e-8 / 0.001, 9e-8 / 0.002, 8e-9 / 0.0004** | 5/5 |
| full | 5 | 2 | 4000 | 0.078 / 4.6, 0.15 / 16, 0.031 / 2.2 | **0.026 / 0.12, 0.046 / 0.18, 0.013 / 0.061** | 5/5 |
| full | 5 | 2 | 20000 | 0.0073 / 0.33, 0.015 / 0.73, 0.0031 / 0.19 | **0.0086 / 0.062, 0.017 / 0.092, 0.0035 / 0.032** | 5/5 |
| full | 5 | 4 | 4000 | 48 / 72, 105 / 154, 23 / 26 | **0.048 / 0.24, 0.12 / 0.27, 0.025 / 0.12** | 5/5 |
| full | 5 | 4 | 20000 | 92 / 108, 204 / 255, 40 / 62 | **0.12 / 0.32, 0.15 / 0.39, 0.076 / 0.14** | 5/5 |
| full | 5 | 8 | 4000 | 65 / 93, 134 / 247, 38 / 51 | **0.25 / 0.74, 0.39 / 1.0, 0.12 / 0.66** | 5/5 |
| full | 5 | 8 | 20000 | not finished in the time box | — |
| full | 20 | 2 | 4000 | 108 / 113, 389 / 484, 43 / 44 | **3e-13 / 4e-12, 2e-11 / 2e-11, 0 / 1e-12** | 5/5 |
| full | 20 | 2 | 20000 | 3e-13 / 247, 8e-11 / 822, 0 / 100 | **5e-13 / 1e-11, 6e-11 / 9e-11, 3e-14 / 4e-12** | 5/5 |
| full | 20 | 4 | 4000 | 126 / 144, 200 / 337, 58 / 91 | **2e-13 / 3e-12, 7e-12 / 1e-11, 7e-15 / 8e-13** | 5/5 |
| full | 20 | 4 | 20000 | 219 / 311, 413 / 625, 122 / 202 | 3e-12 / 19, 4e-11 / 9.6, 8e-14 / 179 | 4/5 |
| full | 20 | 8 | 4000 | 87 / 108, 149 / 202, 137 / 164 | **1e-7 / 0.0004, 3e-7 / 0.001, 3e-8 / 0.0002** | 5/5 |
| full | 20 | 8 | 20000 | not finished in the time box | — |

Python time a row (interpreter bound): diag 36 to 49 µs, full k = 5 66 to
75 µs, full k = 20 85 µs (K = 2) to 129 µs (K = 8).

Reading:

- **The start's weight decides it, not k or K.** At W0 = 1 row (part 1's
  start) the fit is fine at K = 2 and k = 5 but from K = 4 (and at k = 20
  full, K = 2) it lands 20 to 400 SE away: with one row of weight, the first
  rows a component takes set its mean and covariance alone (a full k = 20
  covariance from a handful of rows), and the stream then settles into a
  different attribution. Entering the same start at **20 rows per component**
  brings every cell's median within 0.37 SE, and the maxima within 1 SE in
  all but four cells.
- Those four (diag k = 5, K = 8, both N; diag k = 20, K = 8, N = 4000; full
  k = 20, K = 4, N = 20000) are one seed each at a different fixed point
  (10 to 180 SE). In three of them `best5 same` is 4/5: scikit-learn's own
  best of five starts also left the shared start's fit there, so the shared
  start was itself a poor local optimum on that seed and the comparison
  there is between two local optima, not a tracking error.
- **By component count** (W0 = 20K, median of the max gap over means and
  covariances): K = 2 at most 0.05 SE; K = 4 at most 0.26 SE; K = 8 0.25 to
  0.58 SE at k = 5. The gap grows with K at k = 5, as the components get
  closer (minimum spacing 3.3 SD) and the E-step's overlap grows.
- At k = 20 the minimum distance of 4 is no constraint (random means in 20
  dimensions sit about 15 apart): the responsibilities are 0 or 1 to
  rounding and the online fit equals scikit-learn's to 1e-7 SE or better,
  the hard-assignment case noted above. Separation, not k, sets the gap.
- Diagonal and full covariance behave alike once started well.

## 8. Collapse: a component's weight going to zero

k = 2, K = 3, λ = 0.99: 20,000 rows from all three components, 80,000 rows
from which the third is absent, 20,000 rows with it back. Per variant: the
missing component's S0 (W in the mean form), and what happens.

| variant | missing S0, min | what happens | on return |
|---|---|---|---|
| sum form, part 1 guard (S0 = 0 reads the start) | 2.2e-321 | **fails at row 93,835**: Σ = S2/S0 − μμᵀ not positive definite | — |
| sum form, no guard | 2.2e-321 | the same row, the same failure | — |
| sum form, fading τ = 0.01 (decays with the rows, `hmm`'s ridge) | 1.7e-321 | **fails at row 93,860**, the same way | — |
| mean form (`hmm`'s `EwCov`), part 1 guard | 2.4e-322 | finite throughout; μ, C frozen | **dead**: π = 0.000 (true 0.444), never takes a row |
| sum form, stabilised τ = 0.01 (does not decay) | 6.9e-16 | finite; π ≥ τ/(n_eff + Kτ) = 1.0e-4 | **revives**: π 0.392, μ within 0.13 |
| sum form, reseed (π < 1e-6 and the row an outlier to every component, χ²ₖ 1 − 1e-6) | 1.1e-173 | finite; 2 reseeds | **revives**: π 0.392, μ within 0.13 |
| S0 = 0 from the start, no guard | 0 | μ = 0/0 = NaN at the first read; every row after is NaN | — |
| S0 = 0 from the start, part 1 guard or τ | 0 | finite (reads the start's parameters) | guard alone: π = 0 exactly, log π = −∞, r = 0: absorbing |

What happens, in order:
- S0 of the absent component decays by λ a row, but **never reaches 0**:
  once it is a few dozen subnormal units, λ·S0 rounds back to S0 (at λ = 0.99
  below 50 units, 2.5e-322). The S0 = 0 guard of part 1 never fires.
- The sum form's M-step loses its precision first: S1 and S2 sink into the
  subnormals with S0, so μ = S1/S0 keeps a handful of significant bits and
  S2/S0 − μμᵀ (|μ| ≈ 9, μμᵀ ≈ 80 against variances near 1) is no longer
  positive definite: the Cholesky refuses the row 73,835 rows into the gap.
  Under exact zero (the forced case) it is 0/0 = NaN instead. Either way an
  unguarded division in the M-step poisons the fit.
- The mean form does not fail (μ and C are ratios it never divides out), but
  a component at π ≈ 1e-322 has no route back: it is absorbed for good.
- A fading prior is no floor: it decays with the component and fails with it.

The fix: **a stabilised (non-decaying) pseudo-count τ** at a prior,
μ⁰ⱼ and C⁰ⱼ (here the start's parameters):

```text
πⱼ = (S0ⱼ + τ) / (Σ S0 + Kτ)
μⱼ = (S1ⱼ + τ μ⁰ⱼ) / (S0ⱼ + τ)
Σⱼ = (S2ⱼ + τ (C⁰ⱼ + μ⁰ⱼμ⁰ⱼᵀ)) / (S0ⱼ + τ) − μⱼμⱼᵀ + reg I
```

No division by less than τ, no NaN in any case run; π keeps a floor of
τ/(n_eff + Kτ), so a returning component is taken up at once (its density
there dwarfs the others'). Equivalently (Kulhavý's stabilised form) the
statistics take `+ (1 − λ) τ · prior` each row, decay included, and the
read is part 1's; the two agree in steady state. **Cost:** state, the prior,
K(k + k(k+1)/2) f64 (the fallback part 1 already keeps; k + 1 if one shared
prior μ⁰, d·I serves every component); flops, K(1 + k + k(k+1)/2) adds a
read (k = 2, K = 3: 18 against about 100; k = 20, K = 8: 1,848 against
28,848, 6%); fit, at τ = 0.01 the stationary gaps of section 6 move by at
most 0.001 SE (median), the weights' bias being Kτ/n_eff = 1e-4 relative at
n_eff = 199. Reseeding also revives, costs an O(K) test a row (the E-step's
own Mahalanobis distances) and an O(k²) reseed when it fires, and no state,
but it only fires on an outlier row and does not by itself prevent the sum
form's subnormal failure (it did here because the reseed came first); it is
a complement to τ, not a substitute.

## 9. Zero-weight rows under decay

λ = 0.99, k = 3, K = 2, 400 rows (`zero`), sum and mean forms:
- **Empty state, a zero-weight first row**: S0, S1, S2 (W) stay exactly 0;
  the stream after it is **bit-identical** to the stream without it (0·λ = 0).
- **Initialised state, a zero-weight first row: the clock advances.** Each
  S0ⱼ (Wⱼ) is λ times its value **to the bit**, so n_eff = λ·n_eff; the
  parameters it reads move by 1.1e-16 (mean form: only the weights'
  Wⱼ/ΣW after both are scaled; μ and C are not touched) and 7.1e-15 (sum
  form, S1/S0 after both are scaled).
- **A zero-weight row after every third row** equals the stream without
  them with λ² over each gap (the clock form) to 9.4e-15 (sum) and 8.0e-15
  (mean) relative, worst entry.
- **Under the stabilised τ** the clock advances the same way, but the
  parameters move (2.9e-5 here, n_eff = 75): the statistics decay and τ
  does not, so a gap hands share back to the prior. That is the floor
  doing its job, and it is what `hmm`'s Π already does against its
  pseudo-count τ (the zero-weight row decays the counts before the read).

## 10. State and cost

Bytes = 8·K·(1 + k + k(k+1)/2) full, 8·K·(1 + 2k) diagonal (S0, S1, S2 per
component), plus n_eff, plus the prior/fallback (section 8) if kept. Flops a
row: full E-step K(k³/3 + k² + 4k) refactoring each Σⱼ, or K(2k² + 4k) with
held factors moved by rank-1 updates (`hmm`'s factor form); update K(k² + 3k);
diagonal E-step and update 8Kk; plus K exp and K log either way.

| k | K | full bytes | diag bytes | full flops a row (refactor / held factor) | diag flops |
|---|---|---|---|---|---|
| 5 | 2 | 336 | 176 | 252 / 220 | 80 |
| 5 | 4 | 672 | 352 | 504 / 440 | 160 |
| 5 | 8 | 1,344 | 704 | 1,008 / 880 | 320 |
| 20 | 2 | 3,696 | 656 | 7,212 / 2,680 | 320 |
| 20 | 4 | 7,392 | 1,312 | 14,424 / 5,360 | 640 |
| 20 | 8 | 14,784 | 2,624 | 28,848 / 10,720 | 1,280 |

Bounded by the shape, never by the stream. Python time a row (interpreter
bound, no claim about a Rust row) is in section 7's run log.

## 11. Should `hmm`'s stabilised P0 = d prior carry over to the mixture?

**Yes**, as a non-decaying floor on all three reads (τ on the weights, τ
pseudo-rows at μ⁰ and C⁰ = d·I or the start on the means and covariances).
Evidence from this part: (1) under decay a component that stops being
sampled is the one case where the mixture breaks, and only a floor that
does not decay holds it: the fading prior (today's `hmm` ridge form) failed
at the same row as no prior at all, and the mean form survived only by
freezing the component dead; (2) the stabilised τ = 0.01 kept every run
finite, revived the component on its return, and rescued the one starved
seed of section 6 (5.9 → 0.31 SE); (3) it costs at most 0.001 SE (median) on
stationary data and about 6% of the flops at k = 20, K = 8, and no state
beyond the fallback part 1 keeps. Unlike `rls`'s stabilised forgetting,
which was set aside because its uncentred floor pulls like δ·level², this
floor is in centred, per-component coordinates: C⁰ is a covariance, so its
pull does not grow with the data's level. The one visible effect is section
9's: under τ a zero-weight row moves the parameters toward the prior, by the
clock, as `hmm`'s Π does.

## Open questions (part 2)

5. The start: W0 = 1 row is too light from K = 4 (section 7); `hmm`'s
   `warm_rows` start should enter at its rows' weight, not at one row's.
6. τ's default: 0.01 served here at n_eff 75 to 2000; in units of rows, it
   should be stated against `min_weight`.
