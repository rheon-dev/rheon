<!-- kind: record -->
# Gaussian mixtures by online EM: the design

A design for PLAN 506, written 2026-11-24 on branch `vm/task-506-gmm` off
main `13f5ffe4`, before any code. It builds on the study
`docs/records/STUDY-506-MIXTURES.md` and its script `scripts/study_506.py`
(branch `vm/study-506`, `64fdda0b`; "the study" and "§n" below are its
sections), and on the clustering survey `docs/CLUSTERING.md` §6.4 and §8.
Every number here is the study's unless it says otherwise. The questions
for the coordinator are at the end.

## What is built, and where

A new model kind `gmm` in `rheon-core` (`cluster/gmm.rs`, beside `kmeans`
and `micro`), a `GaussianMixture` fitted one row at a time by
exponentially weighted online EM (Cappé and Moulines 2009): each component
an exponentially weighted mean and covariance fed the row at weight
`w r_j`, `r_j` its responsibility read from the parameters before the row.
It is `hmm`'s E-step and per-state statistics with every transition row
equal (§2, §14: the posterior and the per-state statistics are exact), but
a model of its own: the weights are read from the components' own weights,
not from transition counts, the ridge is a fixed floor, the start is
gated, and a stabilised prior holds a starved component up. Q1 asks
whether it should be an option of `hmm` instead.

## 1. Which of scikit-learn's forms have an online form

`GaussianMixture`'s M-step is a function of per-component sums of the
rows' responsibilities, `S0_j = Σ w r_j`, `S1_j = Σ w r_j x`, and a second
moment. Every sum is additive over rows, so decaying them and adding the
row (each row's responsibilities frozen at the parameters it was read
with) is online EM. The covariance types differ only in which second
moment they keep:

| `covariance_type` | online | second moment kept | state, bytes (no prior) | oracle |
|---|---|---|---|---|
| `full` | **yes**, measured (§3, §6, §7, §18) | `C_j`, packed `k(k+1)/2` | `8 K (1 + k + k(k+1)/2)` | scikit-learn, and the longhand under decay |
| `diag` | **yes**, measured (§7, §18) | the diagonal of `C_j`, `k` | `8 K (1 + 2k)` | the same |
| `spherical` | **yes**, not measured | `tr C_j`, one number: `σ²_j = tr C_j / k` | `8 K (2 + k)` | scikit-learn (`spherical`); tolerances to be measured in the build |
| `tied` | **yes**, not measured | one `C` for all: `Σ_j W_j C_j / Σ W` | `8 K (1 + k) + 8 k(k+1)/2` (Q3) | scikit-learn (`tied`); tolerances to be measured |

`tied` needs care. Its covariance is the pooled within-component scatter,
`Σ_j W_j C_j / Σ_j W_j`. Held as each component's `C_j` (the full state),
it is a convex combination of positive semi-definite terms and needs no
subtraction. Held as one total covariance less the between-component
scatter (`C_total − Σ_j π_j (μ_j − μ̄)(μ_j − μ̄)ᵀ`, the smaller state), it
cancels when components are far apart, the same failure as the study's
sum form (§16, risk 2). Q3 chooses.

**The priors.** `GaussianMixture` has none beyond `reg_covar`, a fixed
floor on each covariance's diagonal, which is built as `reg` (§13).
`BayesianGaussianMixture`'s priors split:

- **As a MAP floor: yes.** A Dirichlet prior on the weights and a
  Normal-inverse-Wishart prior on each component's mean and covariance are
  conjugate, so their posterior mode is the M-step read with pseudo-rows
  added. That is exactly the study's stabilised `τ` (§8, §11, §13): `τ`
  pseudo-rows per component at a prior mean `μ⁰` and covariance `C⁰`. Its
  hyperparameters map onto `BayesianGaussianMixture`'s
  `weight_concentration_prior`, `mean_prior`, `mean_precision_prior` and
  `covariance_prior` at one shared `τ` (Q4).
- **The variational posterior: not built.** `BayesianGaussianMixture`'s
  E-step reads expected log densities (digamma terms) under a variational
  posterior and is meant to drive unneeded components to zero weight. Its
  online form, stochastic variational inference (Hoffman et al. 2013), is
  online *approximately*. scikit-learn has no online form to hold it to,
  and its purpose (pruning components) is what the stabilised floor is
  there to prevent. The `dirichlet_process` prior's stick-breaking order
  also ties each component's prior to its label, so label order would
  change the fit. Not built; listed as not online exactly.

The other `GaussianMixture` parameters map to the start (`init_params`,
`n_init`, `random_state`, `max_iter` and `tol` of the start's batch EM) or
have no online meaning (`warm_start`: a stream always continues;
`verbose`).

## 2. The EM step and its decay

Row `t`, features `x`, weight `w`, `λ` the decay over the row's clock step.
The parameters are read before the row, so its responsibilities and
log-density are out of sample (hard rule 2). The mean form (§13), which
never subtracts:

```text
E:  log ρ_j = log π_j − ½ (k log 2π + log det Σ̂_j + (x − μ̂_j)ᵀ Σ̂_j⁻¹ (x − μ̂_j))
    r_j = exp(log ρ_j − max_l log ρ_l) / Σ_l exp(log ρ_l − max_l log ρ_l)
    log f(x) = max_l log ρ_l + log Σ_l exp(log ρ_l − max_l log ρ_l)
update, each j, with v = w r_j:
    W_j' = λ W_j + v,  a = λ W_j / W_j',  b = v / W_j',  δ = x − μ_j
    μ_j' = μ_j + b δ,  C_j' = a C_j + a b δ δᵀ      (W_j' = 0: nothing moves)
read (T = W_j + τ, d = μ_j − μ⁰_j):
    π_j = (W_j + τ) / (Σ_l W_l + K τ)
    μ̂_j = (W_j μ_j + τ μ⁰_j) / T
    Σ̂_j = (W_j C_j + τ C⁰_j) / T + (W_j τ / T²) d dᵀ + reg I
```

`n_eff = Σ_j W_j` follows the shared recursion `n_eff ← λ n_eff + w`, read
before the row (hard rule 8). The decay is the library's: `half_life` on
the clock, or `lam`, with `gap_cap`.

**A zero-weight row** (and a refused row: a negative or non-finite weight,
or a non-finite feature) is still read (its responsibilities and
log-density are emitted) but learns nothing and counts in no clock (hard
rule 9): no `W`, `μ` or `C` moves, and the next row decays as if it were
not there. The study's §9 decayed the statistics on such a row, as the
library did before rule 9 was stated (2026-11-21); under the rule a
zero-weight row moves nothing, `τ` or not, so §9's 2.9e-5 drift toward the
prior on a gap of zero-weight rows does not arise. The first row of an
empty state at weight 0 leaves every statistic at exactly 0 (§4, §9).

**`window_size`.** The fit over the last `W` of the clock (or rows): each
row's contribution is frozen at its read, so the window's statistics are
the sums of the contributions of the rows inside it, and a row leaves by
removing its own. This rides the library's window machinery (snapshots
under a memory budget, `window.rs`), as the other windowed models do,
with its bytes bounded by the window. A window is checked two ways: the
state against the sums of its rows' frozen contributions, to rounding (a
definition, no oracle needed); and the fit against the weighted EM
longhand on the window's rows (below), within the study's decay
tolerances, since the longhand recomputes responsibilities and online EM
does not.

**Each form's oracle:**

| setting | oracle | tolerance |
|---|---|---|
| no decay, `full` and `diag`, `k = 5`, `K` 2, 4, 8 | scikit-learn's `GaussianMixture` on the same rows from the same start, one pass, `N = 20000`, regular seeds (§18) | §18's 50-seed band, in batch SE: full `K = 8` 2.65 / 4.10 / 2.05 (means / covariances / weights), down to diag `K = 2` 1.10 / 1.40 / 0.65 |
| no decay, `spherical`, `tied` | scikit-learn, the same way | measured in the build by §18's rule (1.5 times the 50-seed max of the regular seeds) |
| `half_life` | the weighted EM longhand from the paper, weights `λ^age` (§6, §17): no library fits a row-weighted mixture; at `λ = 1` it equals scikit-learn to 8.9e-16 | §18: `k = 2` 0.35 / 0.50 / 0.20, `k = 5` 0.10 / 0.10 / 0.05 SE, in the Kish size of the weighted sample |
| `window_size` | the frozen-contribution sums (exact) and the longhand on the window's rows | rounding; and the `half_life` band, to be confirmed in the build |
| overlapping components | none: a limit of online EM, documented (§3, §18) | none set |

## 3. The start: gated on identifiability and the likelihood, not a count

Online EM needs a start, and its weight decides the fit (§7, §16 risk 3):
entered at one row's weight it lands 20 to 400 SE from batch EM from
`K = 4`, and entered at about 20 rows a component every cell's median is
within 0.6 SE. `kmeans` and `hmm` start after `warm_rows` buffered rows, a
count. Under the user's rule of 2026-11-22 (no fixed row count against
early data) the mixture starts when its buffered rows identify `K`
components, tested as follows.

Learned rows are buffered. The buffer is bounded by `max_state_bytes`,
not by a count: a run whose buffer passes the cap before the start is
refused by name, saying how many components the rows held. After each
row (or at each doubling of the buffer, Q5), the start fits batch EM on
the buffer from a seeded k-means++ start, `n_init` times (scikit-learn's
`init_params="k-means++"`, best likelihood kept), and starts when all three
pass:

1. **Every component is identified.** Its responsibility mass holds
   Kish's effective count `(Σ r)² / Σ r²` above the parameters it carries
   (`k + k(k+1)/2` full, `2k` diag, `k + 1` spherical, `k` tied plus the
   shared covariance's share). Its covariance before `reg` has its
   smallest eigenvalue above `reg`, so the floor is not what holds it
   positive definite.
2. **`K` components are told apart.** The buffer's fit with `K` components
   beats the fit with `K − 1` by BIC, `ΔBIC > 10` (Kass and Raftery's
   "very strong"). The likelihood-ratio test of `K` against `K − 1` is not
   χ²-distributed for a mixture (the null sits on the boundary), so BIC,
   which is consistent for the number of components, is the test. A
   stream with fewer than `K` clusters never starts, and its rows read
   `withheld_reason = "not_separated"`, as `pca`'s components do.
3. **The likelihood has stopped moving.** The fit is stable as rows
   arrive: the rows added since the last try score, under the new fit, at
   most `3 p / n` above their score under the last try's fit (`p` the free
   parameters, `n` the buffer's Kish size). *Corrected in the build
   (2026-11-24):* this note first set the bound at `p / (2 n)`, AIC's
   optimism of one in-sample likelihood. That is the wrong comparison.
   At a fixed point the last try's fit, from half the rows, loses about
   `p / n` on the new rows out of sample, and the new fit's in-sample
   optimism on them adds `p / (2 n)`. A converged fit therefore gains
   about `3 p / (2 n)`, three times the first bound, and the bound is set
   at twice that, `3 p / n`. Measured with the first bound and seeded
   starts alone: a one-component fit on stationary rows did not start in
   500 rows (it started at 1,024), and a separated pair stayed `"moving"`
   through 4,000 rows. With `3 p / n` and the warm start below, they
   start at 32 and 128 rows. A fit that has not converged
   gains a constant a row, which no bound of order `p / n` admits as `n`
   grows. Each try also runs EM from the last try's fit beside its seeded
   starts: from a seeded start alone, successive tries on one stream
   landed in different local optima, and the gain between them never
   fell.

At the start, the buffer's fit enters at its rows' weight (`W_j` the
buffer's responsibility mass, decayed to the start's clock), which the
study found decisive (§7), and the buffered rows are not replayed through
online EM: they are in the start. The start's components are put in a
fixed order (Q6) that never changes after, so a label means the same
component for the rest of the run. Until the start, every row is
withheld. The start's cost is a few batch EM fits on a bounded buffer, at
the rows where the gate is tried.

## 4. Labels, the outputs, the state and its bound

**Label switching.** Online EM keeps each component's statistics from row
to row, so a component's label does not switch: its parameters move a
row's worth at a time. A label can change meaning in two places, each
said in the docs. At the start, the order is fixed by Q6. After a
collapse a revived component may take a different cluster (§8, §16 risk
1). The oracle matches components by the Hungarian assignment on squared
mean distances (§17), and a test holds that the assignment is the same at
every check (no switching).

**Outputs**, each row's read from the state before the row:

| field | type | meaning |
|---|---|---|
| `cluster` | `i64` | `argmax_j r_j`; null while withheld |
| `membership` | `list[f64]`, length `K` | the responsibilities `r_j` |
| `log_density` | `f64` | the row's predictive log-density `log f(x)`, out of sample |
| `dist` | `f64` | the Mahalanobis distance to `cluster`'s component |
| `n_eff` (`weight_sum`) | `f64` | rule 8, unchanged |
| `coef` | `list[list[f64]]` | per component, its weight, mean and covariance (packed), on `coef_every` rows, through the existing `coef` plumbing |
| `withheld_reason` | as every model | `"warm_up"` before the gate is tried, `"not_separated"` while gate 2 fails |

The fields follow `kmeans`' (`cluster`, `dist`, `membership`), so a user
switches between the two without renaming.

**State**, bounded by the shape and never by the stream (the study's §15):

| `k` | `K` | full | full, shared prior | diag | spherical | tied (per-component `C_j`) |
|---|---|---|---|---|---|---|
| 5 | 4 | 672 | 720 | 352 | 224 | 672 |
| 20 | 8 | 14,784 | 14,952 | 2,624 | 1,408 | 14,784 |

The shared prior `μ⁰`, `d I` adds `8 (k + 1)` bytes; a per-component prior
at the start's parameters doubles the full state (Q4). The start's buffer
is held only until the start and counts against `max_state_bytes`. A
window adds its snapshots, bounded by the window. Held Cholesky factors of
each `Σ̂_j` (`hmm`'s factor form, `O(K k²)` a row in place of `O(K k³)`) are
derived state, rebuilt on load and not saved (Q7). The cap is checked
when the run is built for the fixed state, and at the row that would pass
it for the buffer and the window, refused with the bytes named.

**Cost a row** (§15): full `K (k³/3 + k² + 4k)` flops refactoring, `K (2k² +
4k)` with held factors; diag `8 K k`; plus `K` exp and log. `k = 20`,
`K = 8` full: 28,848 or 10,720 flops.

## 5. What the study leaves open, and what this design adds

- **The start.** The study shared a batch k-means start over every row,
  so the two fits differed in the EM alone, and it measured that the
  start's weight decides the fit. The gate of section 3 is new and
  unmeasured: the build measures how many rows it takes on §7's and §18's
  streams, and that the fits after it stay within §18's bands.
- **`spherical` and `tied`** were not run; the build measures them by
  §18's rule.
- **`window_size`** was not run; the build measures it against the
  longhand.
- **Zero-weight rows** follow rule 9, not the study's §9 (section 2).
- **Overlapping components** stay a documented limit (§3, §18): at means
  one SD apart one pass sits 5 to 15 SE from batch EM and the gap grows
  with `N`. A step-size schedule with averaging (the study's open question
  4) is not built.
- **`τ`'s default** in rows is open (the study's questions 6 and 7; Q4).

## Questions for the coordinator

1. **Home:** a new kind `gmm` in `rheon-core`'s `cluster/` (recommended:
   its own weight read, ridge, gate and prior, and `kmeans`' outputs), or
   an `iid` option of `hmm` (its E-step and statistics are exact there, but
   `hmm`'s weights, ridge and start would each need a second form)?
2. **Covariance types:** all four in the first build, or `full` and
   `diag` first (measured), with `spherical` and `tied` after their
   tolerances are measured?
3. **`tied`'s state:** each component's `C_j`, pooled at the read
   (recommended: no cancellation, the full state's size), or one total
   covariance less the between-component scatter (smaller, cancels)?
4. **The prior:** one shared `μ⁰` and `d I` (`k + 1` f64), or a
   per-component prior at the start's parameters (doubles the full
   state; the study's collapse runs used it)? And `τ`'s default: 0.01
   rows, as the study ran, or a value stated against `min_weight`? Expose
   `BayesianGaussianMixture`'s hyperparameter names, or `τ` alone?
5. **The gate's cadence:** try the start after every buffered row (exact,
   one batch EM a row until it fires) or at each doubling of the buffer
   (`O(log n)` fits, the start up to twice as late)?
6. **The start's order:** by weight, descending (recommended: the
   largest component is label 0), or by the first mean coordinate?
7. **Held Cholesky factors:** rebuilt on load and not saved (recommended,
   state as §15 counts it), or saved, for an `O(K k²)` first row after a
   load?
8. **BIC's threshold** for gate 2: 10 (recommended), or a significance
   level the user sets?
