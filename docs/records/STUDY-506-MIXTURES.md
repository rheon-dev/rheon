# Study 506: Gaussian mixtures by online EM on `hmm` (parts 1 to 3)

Question: can a Gaussian mixture be fit by exponentially weighted online EM
(Cappé and Moulines 2009, sufficient statistics), built as an `hmm` whose
transition rows are all equal, and does it reach scikit-learn's batch
`GaussianMixture` on stationary data? Part 1: the recursion, the `hmm`
mapping, and the stationary comparison at k = 2 features, K = 2 components.

Script: `scripts/study_506.py`, the one study script since part 3 (numpy
2.5, scipy 1.18, scikit-learn 1.9.1; seeded generators, no data files).
`python scripts/study_506.py --quick` runs every section shortened and
checks each finding below (48 checks, about 80 s);
`python scripts/study_506.py` runs every section in full (about 25 min; run
it under `nohup`); naming sections runs only those (`compare`, `passes`,
`zero`, `timing`, `decay`, `sklearn`, `collapse`, `state`, `tolerances`);
`--cell full 20 8 20000` runs one cell of section 7. Seeds: 5 (0 to 4) in
every table, as in the earlier studies; `--seeds 50` widens it. Notation: k
features, K components (rheon's `k` is the feature count).

Status: done. Parts 1 and 2 (sections 1 to 11) and part 3 (sections 12 to
18: the two cells section 7 lacked, the script consolidated, the design in
one place, the proposed test tolerances). The design summary a builder
needs is section 13 onward; sections 1 to 11 are the evidence.

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

Script: `scripts/study_506.py` (part 2 was a second script until part 3
merged it), one section each (`decay` ~70 s, `sklearn` ~15 min with the two
K = 8, N = 20000 full cells, `collapse` ~45 s, `zero` and `state` ~1 s). The recursion is part 1's sum form vectorised over
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
| full | 5 | 8 | 20000 | 121 / 143, 265 / 332, 76 / 108 | 0.46 / 12, 0.59 / 7.3, 0.31 / 25 | 4/5 |
| full | 20 | 2 | 4000 | 108 / 113, 389 / 484, 43 / 44 | **3e-13 / 4e-12, 2e-11 / 2e-11, 0 / 1e-12** | 5/5 |
| full | 20 | 2 | 20000 | 3e-13 / 247, 8e-11 / 822, 0 / 100 | **5e-13 / 1e-11, 6e-11 / 9e-11, 3e-14 / 4e-12** | 5/5 |
| full | 20 | 4 | 4000 | 126 / 144, 200 / 337, 58 / 91 | **2e-13 / 3e-12, 7e-12 / 1e-11, 7e-15 / 8e-13** | 5/5 |
| full | 20 | 4 | 20000 | 219 / 311, 413 / 625, 122 / 202 | 3e-12 / 19, 4e-11 / 9.6, 8e-14 / 179 | 4/5 |
| full | 20 | 8 | 4000 | 87 / 108, 149 / 202, 137 / 164 | **1e-7 / 0.0004, 3e-7 / 0.001, 3e-8 / 0.0002** | 5/5 |
| full | 20 | 8 | 20000 | 198 / 253, 355 / 414, 284 / 407 | **2e-8 / 8e-6, 6e-8 / 2e-5, 4e-9 / 2e-7** | 5/5 |

Python time a row (interpreter bound): diag 36 to 49 µs, full k = 5 66 to
75 µs, full k = 20 85 µs (K = 2) to 138 µs (K = 8). The two K = 8,
N = 20000 full rows were filled in part 3 (section 12).

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

# Part 3: the missing cells, one script, the design in one place, tolerances

Script: part 2's script merged into `scripts/study_506.py` (header). The
recursion classes are part 2's vectorised ones (`Mix`, the sum form;
`MixMean`, the mean form, now with the stabilised read too), so part 1's
tables are re-run on them: every figure of sections 3, 6 and 7 came out
the same to the digits printed (section 8's did not, and part 2's own
script does not reproduce them either: section 12, corrections). `--quick` runs each section shortened
(`compare` and `decay` at N = 2000; `sklearn` at k = 5, K = 2 and 4,
N = 4000, without the best-of-5 fits; `collapse` at λ = 0.98 over 50,000
rows, which reaches the subnormals in half the rows; `tolerances` at K = 2
and 4 on seeds 0 and 1) and checks each finding as it goes; it exits
non-zero if one fails. Seeds: 5 (0 to 4) throughout, as in parts 1 and 2;
section 18 adds a 50-seed run of the tolerance cells, which changes the
proposal.

## 12. The two missing cells

Section 7's format (W0 = 20K is 160 rows here), λ = 1, one pass, batch SE,
median / max over 5 seeds:

| cov | k | K | N | W0 = 1: means / cov / weights (median / max) | W0 = 20K: means / cov / weights (median / max) | best5 same |
|---|---|---|---|---|---|---|
| full | 5 | 8 | 20000 | 121 / 143, 265 / 332, 76 / 108 | 0.46 / 12, 0.59 / 7.3, 0.31 / 25 | 4/5 |
| full | 20 | 8 | 20000 | 198 / 253, 355 / 414, 284 / 407 | **2e-8 / 8e-6, 6e-8 / 2e-5, 4e-9 / 2e-7** | 5/5 |

They took 341 s and 292 s (5 seeds; almost all of it scikit-learn at
tol = 1e-10 and its best-of-5 fits, not the online pass at 75 and 138 µs a
row), which is why part 2's time box missed them. Both read as their
neighbours: at k = 20 the responsibilities are hard and the fit equals
scikit-learn's; at k = 5, K = 8 the median is within 0.6 SE and one seed
(best5 4/5) sits 7 to 25 SE away, as in the diagonal K = 8 cells.

That seed's kind, looked at for the first time here (diagonal, k = 5, K = 8,
N = 20000, seed 0, the 33 / 96 / 140 SE seed of section 7): scikit-learn's
fit from the shared start has a starved component (weight 0.005, true
weights 0.083 to 0.167) and is not its best of five starts; the online fit
shares that pair's weight 0.077 / 0.077. It is not a separate fixed point:
batch EM restarted from the online fit returns to scikit-learn's fit, but
in 4,224 iterations (2 to 9 on every other seed), a near-flat ridge of the
likelihood (the online fit's mean log-likelihood is 0.0015 a row lower).
So section 7's "one seed each at a different fixed point" should read: the
shared start's batch fit there is a poor optimum on a flat ridge, and the
online pass did not follow it onto the ridge. A test excludes such seeds by
a criterion fixed in advance (section 18's regular seeds), not by the gap.

**Corrections to section 8.** Re-run here, part 2's own script (as
committed) and the merged one agree with each other, but not with section
8's table: the sum form fails at row 96,749, not 93,835 (S0 4.0e-316 at the
failing row, not 2.2e-321); the fading prior at 97,009; the reseed **never
fires** (0 reseeds, not 2) and fails at the sum form's row, it does not
revive; under the stabilised τ the missing component's least S0 is 0.058,
not 6.9e-16 (it keeps taking a share of the tails during the gap, π 0.0077
at the gap's end). The table was evidently from an earlier state of the
script. What section 8 concluded stands, and is firmer for the reseed:
the stabilised τ revives (π 0.391, mean within 0.14, as before), the mean
form survives dead, the fading prior fails with no prior; the reseed is no
remedy (section 16, risk 1, has the λ = 0.98 run as well).

## 13. The recursion, complete

Row t: features x, weight w ≥ 0, λ the decay over the row's clock. The
parameters are read from the state before the row (the row's
responsibilities and log-likelihood are out of sample).

```text
E:  log ρⱼ = log πⱼ − ½ (k log 2π + log det Σⱼ + (x − μⱼ)ᵀ Σⱼ⁻¹ (x − μⱼ))
    rⱼ = exp(log ρⱼ − maxₗ log ρₗ) / Σₗ exp(log ρₗ − maxₗ log ρₗ)
    (log Σₗ ρₗ, the same sum, is the row's predictive log-density)
S:  S0ⱼ ← λ S0ⱼ + w rⱼ,   S1ⱼ ← λ S1ⱼ + w rⱼ x,   S2ⱼ ← λ S2ⱼ + w rⱼ x xᵀ
M:  πⱼ = (S0ⱼ + τ) / (Σₗ S0ₗ + K τ)
    μⱼ = (S1ⱼ + τ μ⁰ⱼ) / (S0ⱼ + τ)
    Σⱼ = (S2ⱼ + τ (C⁰ⱼ + μ⁰ⱼ μ⁰ⱼᵀ)) / (S0ⱼ + τ) − μⱼ μⱼᵀ + reg · I
```

- The statistics are decayed sums: S0ⱼ = Σₛ λ^(t−s) wₛ rₛⱼ, so the fit at
  row t is one EM step's M-step over the decay-weighted sample, each row's
  responsibilities frozen at the parameters it was read with. At λ = 1 it
  is Cappé and Moulines' online EM with step 1/t (a stochastic
  approximation of batch EM); at λ < 1 a constant step 1 − λ.
- n_eff = Σⱼ S0ⱼ follows the shared recursion n_eff ← λ n_eff + w (the
  responsibilities sum to 1); read before the row, it is the weight before
  the row's update and before its decay.
- τ is the prior's weight in rows, **not decayed**: τ pseudo-rows per
  component at μ⁰ⱼ, C⁰ⱼ (the start's parameters, or one shared μ⁰ and
  d · I). With τ = 0 the read is part 1's, which then needs the guards
  (S0ⱼ = 0 reads the start; Σ S0 = 0 reads the start's weights). With τ > 0
  no read divides by less than τ and πⱼ ≥ τ / (n_eff + K τ).
- The fading alternative, τ λ^t (how a prior fades in a model whose ridge
  is scaled by the product of the mixing factors), is no floor: it decays
  with the component it should hold up and fails with it (section 8).
- The stabilised-forgetting alternative adds (1 − λ) τ · prior to the
  statistics each row (S0 ← λ S0 + (1 − λ) τ + w r, and so on); its steady
  state is the read above, but it has no floor at λ = 1 and its floor
  builds up only over 1 / (1 − λ) rows. The read form is exact at every
  λ, costs nothing in the per-row update, and leaves the statistics plain
  sums (so zero-weight rows decay them exactly).
- reg is a fixed floor on the diagonal, as scikit-learn's `reg_covar`.

The form to build is the **mean form** (Wⱼ, μⱼ, Cⱼ), each component an
exponentially weighted mean and covariance fed the row at weight w rⱼ
(section 1), with the τ read in centred coordinates so that nothing
cancels (T = Wⱼ + τ, d = μⱼ − μ⁰ⱼ):

```text
πⱼ = (Wⱼ + τ) / (Σ W + K τ)
μ̂ⱼ = (Wⱼ μⱼ + τ μ⁰ⱼ) / T
Σ̂ⱼ = (Wⱼ Cⱼ + τ C⁰ⱼ) / T + (Wⱼ τ / T²) d dᵀ + reg · I
```

This is the sum form's read rearranged (the pooled covariance of two
groups); the script runs both through the collapse case and they agree to
4e-14 (`collapse`, check "the stabilised read agrees").

**Zero-weight rows** (a refused row is one: negative or non-finite weight,
non-finite features): no E-step; the statistics (W) decay by λ and nothing
is added. With τ = 0 the read does not move (the mean form's μ and C are
untouched; the weights move by rounding). With τ > 0 the clock hands share
back to the prior, by design (section 9). The first row of an empty state
(W = 0) at weight 0 leaves every statistic exactly zero.

## 14. The `hmm` mapping

With every transition row equal to π:

- **Exact: the posterior.** The predicted state probabilities p̃ₗ = Σₖ pₖ πₗ
  = πₗ whatever the previous posterior, so `hmm`'s filtered posterior is
  the mixture responsibility rₜₗ, and its predictive density is the
  mixture's.
- **Exact: the per-state statistics.** Each state's exponentially weighted
  mean and covariance, fed the row at weight w · pₗ, is the mean form
  above (6e-14 to 4e-13 against the sum form, section 1). The order (read
  the densities, then update) and the zero-weight row (decay the weights,
  move no mean or covariance) are the same.
- **Not exact: the weights.** `hmm` learns the transition counts
  Aₖₗ ← λ Aₖₗ + w pₖ(t−1) rₜₗ and normalises each row separately; under
  equal rows the column sums of A are S0ₗ, but the K row reads differ and
  none is π. The iid form keeps **no A** and reads πₗ = (Wₗ + τ) /
  (n_eff + K τ) from the states' own weights. That drops K² counts and the
  previous posterior (K) from the state, and the K² predict step from the
  row.
- **Not exact: the ridge.** `hmm` adds a prior precision scaled by the
  product of the state's mixing factors, which fades like W₁ / W.
  The iid form takes the fixed `reg` floor (scikit-learn's `reg_covar`,
  which the oracle needs matched) plus the stabilised τ pseudo-rows of
  section 13, which do the fading prior's job (a covariance from few rows
  stays positive definite) without fading.
- **Not exact: the start.** `hmm` enters a start (k-means over `warm_rows`
  buffered rows, or given means and covariances) at one row's weight. The
  mixture needs it entered at its rows' weight, at least 20 rows a
  component (section 16, risk 3).

So the iid form is `hmm`'s E-step and per-state statistics unchanged, a new
weight read, a fixed ridge, the stabilised floor, and a heavier start.

## 15. State and cost

State, bytes: **8 · K · (1 + k + k(k+1)/2)** with full covariance (S0 or W,
S1 or μ, the packed symmetric S2 or C), **8 · K · (1 + 2k)** diagonal. n_eff
is Σ S0 and need not be stored. The prior μ⁰ⱼ, C⁰ⱼ adds 8 · K · (k +
k(k+1)/2) full (8 · K · 2k diagonal), or 8 · (k + 1) for one shared μ⁰ and
d · I. Bounded by the shape, never by the stream. The script checks the
formula against the arrays the recursion holds (`state`).

| k | K | full | full + prior | diag | diag + prior | `hmm` today (part 1's field list) |
|---|---|---|---|---|---|---|
| 2 | 2 | 96 | 176 | 80 | 144 | 200 |
| 5 | 2 | 336 | 656 | 176 | 336 | 440 |
| 5 | 4 | 672 | 1,312 | 352 | 672 | 936 |
| 5 | 8 | 1,344 | 2,624 | 704 | 1,344 | 2,120 |
| 20 | 2 | 3,696 | 7,376 | 656 | 1,296 | 3,800 |
| 20 | 4 | 7,392 | 14,752 | 1,312 | 2,592 | 7,656 |
| 20 | 8 | 14,784 | 29,504 | 2,624 | 5,184 | 15,560 |

(`hmm` today: per state 4 scalars, the mean and the packed covariance, plus
the posterior K, the counts K² and n_eff, before the encoding's framing.)
A per-component prior doubles the full state; one shared μ⁰ and d · I
costs k + 1 f64.

Cost a row, flops: full E-step K (k³/3 + k² + 4k) refactoring each Σⱼ, or
K (2k² + 4k) with held factors moved by rank-1 updates (`hmm`'s factor
form); update K (k² + 3k); the τ read K (1 + k + k(k+1)/2) more; diagonal
8Kk. Plus K exp, K log and a log-sum-exp. k = 5, K = 4: 504 (refactor) /
440 (held); k = 20, K = 8: 28,848 / 10,720, diagonal 1,280. Python's time
a row (38 to 138 µs) is interpreter overhead and says nothing about a
compiled row.

## 16. Numerical risks

1. **Collapse, and the subnormal sticking.** Under decay a component that
   stops being sampled loses S0 by λ a row, but S0 **never reaches 0**: below
   about 0.5 / (1 − λ) subnormal units (50 at λ = 0.99, about 2.5e-322)
   λ · S0 rounds back to S0, so it sticks near 1e-321 (the mean form's W
   stuck at 2.4e-322) and an S0 = 0 guard never fires. In the sum form S1
   and S2 sink with S0, S1 / S0 keeps a handful of bits, and S2 / S0 − μμᵀ
   stops being positive definite before S0 sticks: the Cholesky refuses the
   row 76,749 rows into the gap at λ = 0.99 (S0 4e-316) and 37,930 at
   λ = 0.98 (the quick run). The fading prior fails 260 rows later at
   λ = 0.99. The mean form survives, but the component is absorbed for good
   (π = 0 to the print, takes no row on its return). **The stabilised τ**
   prevents all of it: π ≥ τ / (n_eff + K τ), no division by less than τ,
   and the component is taken up at once on its return (π 0.39 at λ = 0.99,
   0.45 at λ = 0.98, true 0.44; mean within 0.14 and 0.20), in the sum form
   and in the mean form's centred read alike. **The reseed** (a starved
   component restarted on a row that is an outlier to every component) is
   no substitute: at λ = 0.99 no such row came before the sum form's failure
   (0 reseeds, the same failing row); at λ = 0.98 it fired once in the gap,
   but the returning rows fell inside the χ² cut of a live component, which
   took them, and the component stayed dead.
2. **The sum form is not positive definite by construction.** S2 / S0 −
   μμᵀ cancels: relative error about ε |μ|² / σ², harmless at the levels
   run here (|μ| ≤ 9, σ² ≈ 1) until S0 goes subnormal, but it grows with the
   data's level (|μ| / σ = 1e6 leaves about 1e-4 relative). The mean form
   holds C as a convex combination of positive semi-definite terms and the
   centred τ read adds only positive semi-definite terms: build that.
3. **The start.** Entered at one row's weight (W0 = 1), the start fails from
   K = 4 (and at k = 20 full from K = 2): the first rows a component takes
   set its covariance alone and the stream settles into another
   attribution, 20 to 400 SE from batch EM (section 7). Entered at 20 rows a
   component, every cell's median is within 0.6 SE. A streaming start
   (`warm_rows` k-means) should enter at its rows' weight; the study shared
   a batch k-means start so the two fits differ in the EM alone.
4. **S0 = 0 exactly** (a component given zero start weight): μ = 0/0 = NaN
   at the first read without a guard, and every row after is NaN. The guard
   or τ keeps it finite; the guard alone leaves it absorbing (π = 0, log π
   = −∞).
5. **Far rows.** The E-step in log space about the largest term gives a
   finite posterior for a row far from every component (no 0/0).
6. **Overlapping components** are a limit of online EM, not a numerical
   fault (section 18).

## 17. The oracles

1. **scikit-learn's `GaussianMixture`** at λ = 1: `covariance_type`
   matched ('full' or 'diag'), `reg_covar` = the online fit's reg (1e-6,
   added to the diagonal by both), `tol` 1e-10 (1e-12 at k = 2),
   `max_iter` 5000, and the shared start passed as `weights_init`,
   `means_init`, `precisions_init` (the start's covariances plus reg,
   inverted). Labels matched by the Hungarian assignment on squared mean
   distances. Gaps in batch SE (section 3), max over components and
   entries. A seed is well posed when the shared start's fit is also
   scikit-learn's best of `n_init = 5` own starts (mean log-likelihood
   within 1e-6). scikit-learn's `GaussianMixture` takes no sample weights,
   so it cannot check decay.
2. **The weighted EM longhand** under decay: batch EM over all N rows with
   fixed weights vₜ = λ^(N−1−t), responsibilities recomputed each iteration,
   run until no parameter moves by 1e-11. It equals scikit-learn's fit at
   λ = 1 to 8.9e-16 (checked each run). SE units use the Kish size
   (Σv)² / Σv² of the decay-weighted sample. No third-party library fits a
   row-weighted mixture, so the definition is written by hand, as the
   project's rule for oracles allows where no library computes the
   quantity; at λ = 1 it is tied to the library.

Largest gaps for separated components, well-posed seeds: against
scikit-learn, one pass at N = 20000, k = 5, the start at 20 rows a
component, regular seeds (section 18): 0.90 / 1.43 / 0.73 SE (means /
covariances / weights, full, K = 8) over 5 seeds, 1.74 / 2.72 / 1.34 SE
over 50; at K = 2, 0.06 / 0.10 / 0.04 over 5 seeds and 0.75 / 0.99 / 0.43
over 50. Against the longhand at
N = 20000, 0.22 / 0.31 / 0.12 SE (means / covariances / weights, k = 2,
λ = 0.999), median 0.0003 to 0.13 SE.

## 18. Proposed test tolerances

Rule: 1.5 times the max over seeds, rounded up to 0.05, in batch SE (the
earlier studies' rule); separated components (`make_mixture`, means 4 or
more apart, 3.3 SD), k = 5, N = 20000, one pass at λ = 1, the start entered
at 20 rows a component, against scikit-learn from the same start, over the
**regular** seeds: well posed (the shared start's batch fit is scikit-learn's
best of 5 own starts) and EM restarted from the online fit back at
scikit-learn's in under 100 iterations. The split is stark: regular seeds
come back in 2 to 22 iterations, the others in 550 to 5,000 (the online pass
ended on a ridge of the likelihood, another attribution of rows to
components). Seeds 0 to 4 as in parts 1 and 2, then 0 to 49
(`tolerances --seeds 50`; about 45 min split over four processes):

| cov | K | 5 seeds: max (means / cov / weights) | tolerance from 5 | 50 seeds: regular | median | max | **proposed tolerance** | regular seeds over the 5-seed tolerance |
|---|---|---|---|---|---|---|---|---|
| diag | 2 | 0.059 / 0.097 / 0.039 | 0.10 / 0.15 / 0.10 | 50 | 0.006 / 0.013 / 0.002 | 0.71 / 0.92 / 0.42 | **1.10 / 1.40 / 0.65** | 4 |
| diag | 4 | 0.148 / 0.211 / 0.063 | 0.25 / 0.35 / 0.10 | 48 | 0.095 / 0.150 / 0.053 | 0.79 / 0.93 / 0.70 | **1.20 / 1.45 / 1.05** | 15 |
| diag | 8 | 0.526 / 0.859 / 0.289 (seed 0 out) | 0.80 / 1.30 / 0.45 | 34 | 0.25 / 0.33 / 0.14 | 0.83 / 1.26 / 0.95 | **1.25 / 1.90 / 1.45** | 3 |
| full | 2 | 0.061 / 0.092 / 0.032 | 0.10 / 0.15 / 0.05 | 50 | 0.011 / 0.021 / 0.005 | 0.75 / 0.99 / 0.43 | **1.15 / 1.50 / 0.70** | 7 |
| full | 4 | 0.323 / 0.394 / 0.135 | 0.50 / 0.60 / 0.25 | 46 | 0.21 / 0.36 / 0.11 | 1.21 / 1.36 / 0.82 | **1.85 / 2.05 / 1.25** | 8 |
| full | 8 | 0.897 / 1.426 / 0.725 (seed 2 out) | 1.35 / 2.15 / 1.10 | 35 | 0.38 / 0.55 / 0.23 | 1.74 / 2.72 / 1.34 | **2.65 / 4.10 / 2.05** | 1 |

Reading:

- **Five seeds understate the tail.** The gap is heavy-tailed: at K = 2 the
  median is 0.01 SE and the 90th percentile 0.1, but one seed in 50 sits at
  0.7 to 1 SE (diagonal seed 31, full seed 33: well-posed fits, EM back
  in 8 and 12 iterations, so genuine tracking gaps, not another optimum),
  and 3 or 4 seeds in 50 above 0.3 SE. Seeds 0
  to 4 happen to miss it, and their tolerances would fail 1 to 15 of the 50
  (last column). The proposed tolerances are therefore the 50-seed ones; the
  5-seed column is kept to show the rule's sensitivity. A test with fixed
  seeds passes either way, but only the 50-seed band survives a change of
  generator or seed list.
- **Not every seed is a test case.** The shared k-means start is itself a
  poor optimum on 2 of 50 seeds at K = 4 diag, 3 at K = 4 full, 16 at K = 8
  diag and 14 at K = 8 full (scikit-learn's best of 5 starts does better);
  there the comparison is between two fits of a weakly posed batch problem.
  And on 2 seeds of 300 (full K = 4 seed 7, full K = 8 seed 18) the batch fit
  was well posed but the online pass ended elsewhere (246 and 227 SE, EM
  back in 3,270 and 550 iterations): with K ≥ 4 a single pass occasionally
  settles two components on one cluster. A test should fix regular seeds and
  state the criterion; the script computes both.
- The tolerances grow with K (closer components, more overlap at the
  minimum spacing) and are about 1.3 to 1.5 times larger for full than for
  diagonal covariance at K ≥ 4.
- k = 20 needs no row: its responsibilities are hard and every regular cell
  of section 7 is within 1e-3 SE; the k = 5 tolerances cover it.

Under decay, against the longhand (N = 20000, K = 2, the start at one row,
which has decayed away by then; 5 seeds):

| k | λ | max means / cov / weights | tolerance |
|---|---|---|---|
| 2 | 0.999 | 0.218 / 0.307 / 0.117 | 0.35 / 0.50 / 0.20 |
| 2 | 0.99 | 0.183 / 0.315 / 0.115 | 0.30 / 0.50 / 0.20 |
| 5 | 0.999 | 0.038 / 0.060 / 0.019 | 0.10 / 0.10 / 0.05 |
| 5 | 0.99 | 0.025 / 0.038 / 0.012 | 0.05 / 0.10 / 0.05 |

And k = 2, K = 2 at a one-row start (part 1's table, N = 20000, max 0.74 /
0.97 / 0.44): 1.15 / 1.50 / 0.70.

**Overlapping components do not pass, and no tolerance is proposed for
them.** At means one SD apart (k = 2, part 1) one pass sits 5 to 15 SE
(median) from batch EM, the gap in SE units grows with N (6.7 at N = 2000,
14 at 20000), and eight passes close little of it. Online EM's error after
an early misattribution decays like t^−(1−ρ), ρ the EM map's contraction
rate (the fraction of missing information), near 1 when components
overlap, while the batch SE falls like t^−½: the ratio grows without
bound. A band wide enough to pass would pass a wrong fit. It is a limit of
the method to document, not a tolerance to set; a step-size schedule with
averaging (open question 4) is the remedy to study if it matters.

## Open questions (part 3)

7. The τ default in rows and how it reads against `min_weight` (question 6)
   stays open: the study ran τ = 0.01 only.
8. Whether to keep a per-component prior (doubles the full state) or one
   shared μ⁰, d · I (k + 1 f64): the collapse case was run with the start's
   parameters per component.
