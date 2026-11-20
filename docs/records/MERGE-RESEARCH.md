# Merging saved states where merging is not exact: research notes

Read-only research for PLAN task 529 (branch vm/research-529). No code changes.
Each family: merge rule(s) in the literature or libraries, exactness (exact /
bound with its value / none), state and memory cost against bounded state,
an oracle that could check it, and a recommendation. Citations to papers and
library docs.

## Status

- Part 0: skeleton.
- Part 1 (huber, quantile, robust covariance, sgd, pa, ftrl): done, 2026-11-20.
- Parts 2-3 pending.

## Parts

1. huber, quantile, robust covariance; sgd, pa, ftrl
2. kmeans, hmm online EM, mixtures; held_median, tails sketches
3. bocpd, seqtest, drift, breaks, corrchange, adaptive conformal

## Notes by family

### Part 1: what "merge" is taken to mean here

PLAN tasks 528 and 529 are not in `docs/PLAN.md` on this branch (searched
2026-11-20), so these notes assume the plain definition: two saved states of
one configuration, A and B, learned from disjoint sets of rows, are combined
into one state without re-reading a row. Two cases are kept apart throughout:

- **concatenation**: B's rows all follow A's (B started cold at the seam).
  Under a half-life A's sums are aged by the decay over B's span first.
- **shards**: A and B cover the same span, rows interleaved.

"Exact" means the merged state is the state of one pass over the union (in
real arithmetic; a sum taken in another order differs in its last bits, which
is a separate question from this one). Every family in this part is
**path-dependent**: a row's contribution depends on the fit before it, so a
state merged from two cold starts cannot be the one-pass state. The useful
questions are then (i) what the merged state *is* exactly, and (ii) how far it
is from the converged estimator on the union, which is what an oracle
computes.

Citations marked *(from memory)* were not re-checked in this session; the
others had author, title, venue and year confirmed by a search on 2026-11-20,
but theorem statements were read from abstracts, not the full text, unless
said otherwise.

### huber (`robust.rs`, `huber_refit.rs`)

State today, per target: the weighted Gram (`EwCov`, k(k+1)/2 floats), the
cross-moment (k), target mean and scale sums, the scale sketch (`EwQuantile`,
log buckets), and until the refit fires a buffer of the first `refit_rows`
rows (default max(500, 10k) rows of k+2 floats: 88 KB at k = 20).

Merge rules:

1. **Add the weighted sums** (Gram, cross-moment, weights), then solve. This
   is the Hessian-weighted combination of the two fits,
   `b = (G_A + G_B)^-1 (G_A b_A + G_B b_B)`, the "aggregated estimating
   equation" estimator: Lin and Xi (2011), "Aggregated estimating equation
   estimation", Statistics and Its Interface 4:73-83 *(from memory)*. What it
   is exactly: the weighted least squares over the union with each row at the
   robust weight its own shard gave it. It is associative and order-free.
2. **Average the coefficients** (divide and conquer). Zhang, Duchi and
   Wainwright (2013), "Communication-efficient algorithms for statistical
   optimization", JMLR 14:3321-3363: mean-squared error
   `O(1/N + 1/n^2)`, N rows in all, n a shard. Their conditions ask a smooth
   loss; Huber's second derivative jumps at the cut, so the theorem applies to
   the population loss only where the noise has a density there. I have not
   checked that their assumptions hold verbatim for Huber.
3. **One more round on the data** from the merged point: Jordan, Lee and Yang
   (2019), "Communication-efficient distributed statistical inference", JASA
   114(526); Luo, Sun and Zhou (2022), "Distributed adaptive Huber
   regression", Computational Statistics and Data Analysis 169:107419
   (gradients exchanged each round, a Berry-Esseen bound for the result).
   Both need another pass over the rows, so neither is a merge of saved
   states.
4. **While both states still hold their refit buffers** and the union fits
   the cap: concatenate the buffers and run the existing IRLS to convergence.
   This is the converged M-estimator on the union, by construction.
5. **The scale**: `sig2`/`wsig` are sums and add. The MAD sketch is a
   log-bucket sketch of the DDSketch kind (Masson, Rim and Lee 2019,
   "DDSketch: a fast and fully-mergeable quantile sketch with relative-error
   guarantees", PVLDB 12(12):2195-2205): bucket weights add, and the merged
   sketch is the sketch of the union when both share a grid. `EwQuantile`'s
   grid has an anchor; two states anchored differently are up to a bucket
   apart, `2a/(1 - a)` in the read by the file's own account. The residuals
   sketched are each shard's own, not the merged fit's.

Exactness: **exact** for rule 4 (both in the refit phase). Otherwise **none**
against the one-pass state; against the converged estimator **bounded only in
rate**, `O(1/N + 1/n^2)` for rule 2: at N = 20,000 in two shards of 10,000
the second term is `1e-8` against `5e-5`, a 0.02% addition to the mean-squared
error, constants unstated by the theorem. Rule 1 has no published finite
bound I can cite; it reduces to rule 2's order when the shards' Grams agree.

Cost: rules 1, 2 and 5 add no state. Rule 4 is the existing buffer. All
independent of stream length.

Oracle: `statsmodels.RLM(M=HuberT, scale_est="mad")` on the union for the
target of rules 1, 2 and 4 (already the refit's oracle);
`sklearn.linear_model.HuberRegressor` as a second opinion. Rule 1's algebra is
checked by `numpy.linalg.solve` on the summed systems.

Recommendation: offer rule 4 as exact where it applies and rule 1 elsewhere,
documented as "the weighted least squares of both shards' rows at their own
weights", with the distance to `RLM` on the union measured over seeds rather
than promised. Do not call it exact.

### quantile (`robust.rs`, `robust/quantile_refit.rs`)

State: as huber's, with the Gram holding only the in-band rows and the
cross-moment the nudges `2h psi(r) z`.

Merge rules:

1. **Add the sums.** rheon's row is the linear form of the smoothed check
   loss, and that is the form the literature merges: Chen, Liu and Zhang
   (2019), "Quantile regression under memory constraint", Annals of Statistics
   47(6):3244-3273. Their estimator is a ratio of two sums evaluated at an
   initial fit; the sums add across batches exactly, and a few rounds of
   aggregation reach the full-data estimator's efficiency while n grows
   polynomially in the batch size. One merge of saved states is one such
   round, with the difference that each shard linearised at its own running
   fit and band, not at a common one. (The search confirmed the paper's claim
   and method in outline; "linear-type estimator" as their name for it is from
   memory.)
2. **Average the coefficients.** The same paper: the naive average is valid
   only when `n = o(m^2)` (n all rows, m a batch). Volgushev, Chao and Cheng
   (2019), "Distributed inference for quantile regression processes", Annals
   of Statistics 47(3):1634-1662, give the conditions on the number of shards
   under which averaging loses nothing; for two shards they hold trivially.
3. **Refit buffers**, as huber's rule 4, through `quantile_refit`.
4. **The scale sketch**, as huber's rule 5. For an *unconditional* quantile
   the mergeable-summary results apply directly: Agarwal, Cormode, Huang,
   Phillips, Wei and Yi (2013), "Mergeable summaries", ACM TODS 38(4):26;
   Karnin, Lang and Liberty (2016), "Optimal quantile approximation in
   streams", FOCS, pp. 71-78 (additive rank error `eps n`, kept under merges);
   DDSketch (relative value error `a`, kept under merges, bucket counts add).
   KLL is unweighted and has no decay; a log-bucket sketch takes both, which
   is why `EwQuantile` is one. t-digest (Dunning and Ertl 2019, arXiv
   1902.04023, *from memory*) merges but states no worst-case bound.

Exactness: **exact** for rule 3; the sketch merge is **bounded**, value error
`a` on a shared grid and `2a/(1 - a)` across two anchors (about 2% at
`a = 0.01`); the coefficients **none** against one pass, and against
`QuantReg` on the union a rate only (rule 2's condition holds for two shards;
no constant).

Cost: nothing added. A DDSketch-style sketch at `a = 1%` over twelve decades
is about 1,380 buckets, 11 KB; `EwQuantile` is already that kind of state.

Oracle: `statsmodels.QuantReg` on the union (already the family's oracle) for
the coefficients; `numpy.quantile` on the union's residuals for the sketch,
within `a`.

Recommendation: the same as huber: rule 3 where it applies, else add the
sums, stated as one aggregation round and measured against `QuantReg`.

### robust covariance (`rcov.rs`; and the M-estimators it is not)

`rcov` is robust to microstructure noise, not to outliers. Its estimators are
sums over lags of products of **adjacent** returns: Barndorff-Nielsen, Hansen,
Lunde and Shephard, "Multivariate realised kernels", Journal of Econometrics
2011, and Christensen, Kinnebrock and Podolskij, "Pre-averaging estimators of
the ex-post covariance matrix in noisy diffusion models with non-synchronous
data", Journal of Econometrics 2010 (the search found both as working papers
only; journal and year are as `rcov.rs` cites them, unverified here).

Merge rules, concatenation:

1. **`plain`**: `sum x x'` adds. Exact.
2. **`kernel`, seam as a break**: add `Gamma_h`, `n` and the subsampled
   grids, as the model already does across a `gap_cap` break (stretches). The
   result is exactly the estimator with a break at the seam. Against the
   unbroken estimator it lacks the products that straddle the seam,
   `sum_h h k(h/(H+1))` of them out of about `n (2H+1)`, and jitters two extra
   end points: at H = 30 and n = 23,400 that is under 0.07% of the lag
   products, each of mean zero for an efficient price beyond the noise's own
   lag.
3. **`kernel`, seam closed exactly**: A already keeps its last
   `max_bandwidth + m` returns (the ring); B would have to keep its first
   `max_bandwidth + m` as well, where it keeps `m` today. Then the straddling
   products and the un-jittered seam can be formed at the merge. Extra state:
   `8 k (max_bandwidth)` bytes a group (k = 4, H = 30: 960 bytes), bounded by
   the configuration.
4. **`preavg`**: the same two choices, with `k_n - 1` straddling windows; the
   exact form needs B's first `k_n - 1` returns.

Shards: **none**. A return is a difference of adjacent prices and every lag
product pairs neighbours; two interleaved shards hold different returns from
the union's. The literature's answer to interleaved clocks is refresh-time
sampling upstream, not a merge.

For outlier-robust scatter (not a rheon model today): Maronna (1976), "Robust
M-estimators of multivariate location and scatter", Annals of Statistics 4,
and Tyler (1987), "A distribution-free M-estimator of multivariate scatter",
Annals of Statistics 15 *(both from memory)*, are fixed points whose weights
depend on the current estimate, so they merge as huber does: weighted sums
add, exactness none, averaging at the divide-and-conquer rate. The minimum
covariance determinant (Rousseeuw and Van Driessen 1999, "A fast algorithm for
the minimum covariance determinant estimator", Technometrics 41, *from
memory*) selects a subset and has no mergeable summary. Shrinkage estimators
are functions of sums and merge exactly: OAS (Chen, Wiesel, Eldar and Hero
2010, IEEE Trans. Signal Processing, *from memory*) needs only S, n and k;
Ledoit and Wolf (2004, Journal of Multivariate Analysis 88, *from memory*)
also needs `sum |x|^4` and, centred, `sum |x|^2 x`: k + 1 more floats (my
expansion of their intensity, to be checked against scikit-learn before use).

Exactness: `plain` **exact**; `kernel`/`preavg` **exact as a break at the
seam** with no new state, **exact unbroken** with B's leading rows kept;
shards **none**.

Oracle: the existing `rcov` oracles computed on the concatenated returns
(with and without a break at the seam); `sklearn.covariance.OAS`,
`LedoitWolf` and `MinCovDet` for the scatter estimators.

Recommendation: concatenation as a break is free and exact for what it says;
offer it first. The unbroken form is a small bounded state growth (state its
bytes to the user before it lands). Refuse shards.

### sgd (`sgd.rs`)

State per target: `beta` (k), the per-coordinate squared-gradient sums `g2`
(k), scaler moments, target spread, scale sums and sketch, the Huber refit
buffer under `loss = "huber"`.

Merge rules:

1. **One-shot parameter averaging**, weighted by each state's weight.
   Zinkevich, Weimer, Li and Smola (2010), "Parallelized stochastic gradient
   descent", NIPS 23; Mann, McDonald, Mohri, Silberman and Walker (2009),
   "Efficient large-scale distributed training of conditional maximum entropy
   models", NIPS 22 *(from memory)*; the rate in Zhang, Duchi and Wainwright
   (2013): `O(1/N + 1/n^2)` for a strongly convex smooth loss on identically
   distributed shards. Federated averaging (McMahan, Moore, Ramage, Hampson
   and Aguera y Arcas 2017, "Communication-efficient learning of deep networks
   from decentralized data", AISTATS, PMLR 54) is the same step repeated.
   A bound that needs no distribution: every loss here is convex in the
   coefficients, so the merged fit's expected loss is at most the weighted
   mean of the two fits' losses (Jensen). It is never worse than the worse
   shard; it is not thereby close to one pass.
2. **Averaging weighted per coordinate by the adaptive accumulators**,
   `b_i = (g2_A,i b_A,i + g2_B,i b_B,i) / (g2_A,i + g2_B,i)`: what Vowpal
   Wabbit's cluster mode does. Agarwal, Chapelle, Dudik and Langford (2014),
   "A reliable effective terascale linear learning system", JMLR 15 *(from
   memory; the per-feature weighting by the adaptive diagonal is as I recall
   the paper, unverified)*. `g2` itself adds.
3. **Exact, for the squared loss with a step that does not read the state**:
   the iterate is affine in its start, `b_AB = b_B(cold) + P_B b_A` with
   `P_B = prod (I - eta_t (w z z' + l2 I))` over B's rows. This is my
   derivation, not a citation, and it does not survive what `sgd` does:
   adaptive steps (`g2` reads the gradients), gradient clipping, the Huber,
   quantile, epsilon-insensitive, Poisson and logistic losses, or the
   standardizing map. It also costs `8 k^2` bytes a target (k = 20: 3.2 KB;
   k = 100: 80 KB) and O(k^2) work a row, which gives up the reason `sgd`
   exists.
4. Scaler moments and the target spread are sums and merge exactly; during
   the scaler's warm-up `beta` is in the moments' coordinates, so each state
   must be read out to the caller's units before averaging.

Under drift (concatenation of two regimes) averaging has no guarantee: the
average of two regimes' fits is neither's.

Exactness: **none** against one pass. **Bounded** in rate for rule 1 on
identically distributed shards (the 0.02% figure of the huber section at
N = 20,000, n = 10,000, same caveat on constants), and by convexity always.

Cost: rules 1, 2 and 4 add nothing. Rule 3 is rejected on cost and scope.

Oracle: no library merges two SGD states; the arithmetic of rules 1 and 2 is
`numpy`. The statistical check is the union's optimum from
`sklearn.linear_model` (`Ridge`, `HuberRegressor`, `PoissonRegressor`,
`LogisticRegression`) or `statsmodels.QuantReg`, with the merged fit's
distance compared to each shard's over seeds.

Recommendation: weighted averaging (rule 2 where `g2` is live, else rule 1),
documented as an approximation with the convexity guarantee and a measured
distance, and concatenation steered to resuming A's state on B's rows, which
is exact today.

### pa (`pa.rs`)

State per target: `beta` (k), the target spread, scaler moments. No learning
rate, no accumulators.

Merge rules:

1. **Parameter averaging**, weighted by weight. There is no PA-specific
   result I know of. The nearest is McDonald, Hall and Mann (2010),
   "Distributed training strategies for the structured perceptron", NAACL:
   a single mix of independently trained perceptrons can fail to separate
   separable data, while mixing every epoch converges. Their bounds are for
   the iterated form, which needs passes over the rows.
2. Crammer, Dekel, Keshet, Shalev-Shwartz and Singer (2006), "Online
   passive-aggressive algorithms", JMLR 7:551-585, bound cumulative loss, not
   the distance of the last iterate from anything, so averaging two last
   iterates inherits no bound from them. The convexity bound of the sgd
   section holds (the epsilon-insensitive loss is convex).
3. **Averaged iterates.** Online-to-batch conversion (Cesa-Bianchi, Conconi
   and Gentile 2004, "On the generalization ability of on-line learning
   algorithms", IEEE Trans. Information Theory 50, *from memory*) gives its
   guarantee to the mean of the iterates over the run. A running sum of
   iterates and its weight add across states exactly (k + 1 floats a target),
   but predicting from it is a different model from `pa`.
4. The spread and the scaler merge exactly, as sums.

Exactness: **none**; the only bound is convexity's.

Cost: rule 1 nothing; rule 3 `8 (k + 1)` bytes a target.

Oracle: none computes a PA merge. `sklearn.linear_model.
PassiveAggressiveRegressor` and river's `PARegressor` check a single pass
only (with the intercept difference `pa.rs` already maps); the merged fit can
be scored against `sklearn.svm.LinearSVR` on the union, the batch problem PA
approximates, as a distance, not an equality.

Recommendation: weighted averaging, labelled an approximation, or no merge at
all for `pa`; concatenation is served by resuming. Rule 3 only if the user
wants an averaged-PA model in its own right.

### ftrl (`ftrl.rs`)

State per target: `n` (sum of squared gradients), `zz` (the adjusted gradient
sum) and the proximal weight `d`, k floats each. McMahan et al. (2013), "Ad
click prediction: a view from the trenches", KDD *(venue from memory; the
search did not return it)*; McMahan (2017), "A survey of algorithms and
analysis for adaptive online learning", JMLR 18 *(from memory)*.

Merge rules:

1. **Add `zz`, `n` and `d`.** FTRL-proximal's coefficients are the minimizer
   of a sum of per-row terms (the linearised loss and the row's proximal
   term) plus one fixed penalty, and the state is those sums, so the sum of
   two states is exactly the state whose coefficients minimize the sum of the
   two objectives, with `l1`, `l2` and `beta/alpha` counted once. This is
   exact for that statement, associative and order-free, and under a
   half-life A's three sums are aged first. It is not the one-pass state: B's
   gradients were taken at iterates that had not seen A. That is FTRL with
   delayed gradients: Langford, Smola and Zinkevich (2009), "Slow learners
   are fast", NIPS 22, and McMahan and Streeter (2014), "Delay-tolerant
   algorithms for asynchronous distributed online learning", NIPS 27 *(both
   from memory)*; their regret bounds grow with the delay (of order
   `sqrt(tau T)` in the first), and a merge is a delay of a whole shard, so
   the bound says nothing useful here.
2. **A caveat on `d` without a half-life.** The code then reads the proximal
   weight as `sqrt(n)/alpha`. After a merge `sqrt(n_A + n_B)` is less than
   `sqrt(n_A) + sqrt(n_B)`, the weight `zz` was built with (by a factor of up
   to sqrt(2) for two equal shards), so the closed form no longer matches the
   sums. A merged state has to carry `d` explicitly, as the half-life path
   does, or its coefficients are not the minimizer rule 1 describes.
3. **Averaging the coefficients** (federated averaging) does not map back to
   a state: a coefficient at zero under `l1` has no unique `zz`. Averaging is
   a rule for the read-out, not for the state. I know of no library that
   merges FTRL states; river's `FTRLProximal` has none.

Exactness: **exact for the sum-of-objectives statement** (rule 1 with `d`
kept); **none** against one pass, with no bound of use.

Cost: nothing added: 24k bytes a target today, independent of stream length.
Rule 2 may move a no-half-life state onto the explicit-`d` path, which is a
behaviour change to state, not a size change.

Oracle: the definition, minimized by `scipy.optimize.minimize` over the
summed objective rebuilt in the test from both shards' rows, iterates and
steps (coordinate-wise it is a closed form); river's `FTRLProximal` for each
shard's own pass (already T-R1); `sklearn` `Lasso`/`LogisticRegression` on the
union for the statistical distance.

Recommendation: add the three sums, keep `d` explicit after a merge, and
document it as "the fit of both shards' linearised losses", not as the
one-pass fit. Of this part's stateful learners it has the cleanest rule.

### Part 1 summary

| family | rule | exactness |
|---|---|---|
| huber | add weighted sums; refit buffers concatenated | exact while both in the refit phase; else none vs one pass, rate `1/N + 1/n^2` vs the converged fit |
| quantile | the same; sketch buckets add | as huber; sketch bounded by `a` (`2a/(1-a)` across anchors) |
| rcov | add lag sums as a break at the seam | exact (plain; as a break); unbroken with B's leading rows; shards none |
| sgd | weighted parameter averaging | none; convexity bound, rate on i.i.d. shards |
| pa | weighted parameter averaging | none; convexity bound only |
| ftrl | add `zz`, `n`, `d` | exact for the summed objective; none vs one pass |
