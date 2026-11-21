# Study 506: Gaussian mixtures by online EM on `hmm` (part 1 of 3)

Question: can a Gaussian mixture be fit by exponentially weighted online EM
(Cappé and Moulines 2009, sufficient statistics), built as an `hmm` whose
transition rows are all equal, and does it reach scikit-learn's batch
`GaussianMixture` on stationary data? Part 1: the recursion, the `hmm`
mapping, and the stationary comparison at k = 2 features, K = 2 components.

Script: `scripts/study_506.py` (numpy, scipy, scikit-learn 1.9.1; seeded
generators, no data files). `python scripts/study_506.py` runs everything in
about 80 s; `--quick` runs N = 2000 only in about 10 s. Notation: k features,
K components (rheon's `k` is the feature count).

Status: part 1 done. Parts 2 and 3 (decay and drift, larger k and K, the
other covariance shapes) are not started.

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
