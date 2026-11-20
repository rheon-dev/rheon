# Merging saved states where merging is not exact: research notes

Read-only research for PLAN task 529 (branch vm/research-529). No code changes.
Each family: merge rule(s) in the literature or libraries, exactness (exact /
bound with its value / none), state and memory cost against bounded state,
an oracle that could check it, and a recommendation. Citations to papers and
library docs.

## Status

- Part 0: skeleton.
- Part 1 (huber, quantile, robust covariance, sgd, pa, ftrl): done, 2026-11-20.
- Part 2 (kmeans, hmm online EM, mixtures, held_median, tails): done, 2026-11-20.
- Part 3 (bocpd, seqtest, drift, breaks, corrchange, adaptive conformal): done, 2026-11-20. Summary of all 17 families at the end of the file.

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

### Part 2: what changes from part 1

The definitions of part 1 stand (concatenation, shards, "exact"). Two things
are new. First, a **label** has no meaning across two states: cluster 3 of A
is not cluster 3 of B, so every rule below starts by matching components, and
the matching is itself a choice (the assignment problem on the centre
distances, `scipy.optimize.linear_sum_assignment`; label switching in
mixtures: Stephens 2000, "Dealing with label switching in mixture models",
JRSS B 62, *from memory*). Second, two of this part's families are not
path-dependent at all (`tails`' moments, `held_median`'s held rows) and merge
exactly.

There is no model named `mixture` in `crates/rheon-core/src` on this branch
(searched 2026-11-20). The mixtures section is written for the family in
general and says where it meets `hmm`'s emission states, `micro` and
`ew_class`. Searches were made on 2026-11-20; a citation is marked *(from
memory)* where no search confirmed it, and theorem constants were not read
from full texts unless said.

### kmeans (`cluster/kmeans.rs`, `cluster/summary.rs`)

State: three sets of k summaries (`clusters`, `batch`, `far`), each
`(n, c, c_lo, r2)` = `8 (2p + 2)` bytes, so about `48 k (p + 1)` bytes
(k = 8, p = 20: 8 KB), the feature moments and metric (O(p)), a handful of
scalars and counters, and until seeding the warm-up buffer (`warm_rows` rows
of p + 1 floats).

Merge rules:

1. **A summary is a clustering feature and two of them add.** `(n, c, r2)` in
   mean form carries the same information as BIRCH's `(N, LS, SS)`: Zhang,
   Ramakrishnan and Livny (1996), "BIRCH: an efficient data clustering method
   for very large databases", SIGMOD (the paper and its CF concept confirmed;
   the additivity theorem itself is *from memory*). The pooled weight, mean
   and radius² of two row sets are functions of the two summaries (the
   parallel-axis identity), which `ClusterSummary::merge_welford` already
   computes. Exact for the statement "the summary of the union of these two
   row sets", associative, order-free; a decay ages `n` first.
2. **Which summaries to pool: recluster the 2k weighted centres.** Guha,
   Meyerson, Mishra, Motwani and O'Callaghan (2003), "Clustering data
   streams: theory and practice", IEEE TKDE 15(3) (title and year confirmed;
   venue *from memory*): cluster each piece, keep the k centres weighted by
   their counts, cluster the weighted centres. The same last step closes
   k-means|| (Bahmani, Moseley, Vattani, Kumar and Vassilvitskii 2012,
   "Scalable k-means++", PVLDB 5:622-633; the reclustering step is *from
   memory*). What it is exactly: by the identity
   `sum_i |x_i - C|^2 = sum_i |x_i - m|^2 + n |m - C|^2`, weighted k-means on
   the 2k centres minimizes the union's inertia **over the assignments that
   keep each shard cluster whole**. The within-shard term is a constant. So
   the result is the constrained optimum, and its excess over the
   unconstrained one comes only from rows that would leave their shard
   cluster.
3. **The worst-case factor.** For k-median Guha et al. give: a b-approximate
   clustering of each piece, then a c-approximate clustering of the weighted
   centres, is a `c (1 + 2b) + 2b` approximation of the whole *(from memory;
   the search returned only a secondhand `2^(1/eps)` for the multi-level
   form)*. At b = c = 1 that is 5; for squared distances the relaxed triangle
   inequality makes the constant larger. It is a bound on the inertia against
   the optimum, not on the distance to the one-pass state.
4. **Mergeable coresets.** The union of an eps-coreset of A and one of B is
   an eps-coreset of the union, and a coreset of a coreset compounds the
   error ("merge and reduce"): Har-Peled and Mazumdar (2004), "On coresets for
   k-means and k-median clustering", STOC, size `O(k eps^-d log n)`
   (confirmed); Feldman, Schmidt and Sohler (2013), "Turning big data into
   tiny data: constant-size coresets for k-means, PCA and projective
   clustering", SODA (title confirmed, venue *from memory*), size free of n;
   Bachem, Lucic and Krause (2018), "Scalable k-means clustering via
   lightweight coresets", KDD (title confirmed). This gives `(1 + eps)` on the
   inertia of *any* k centres, which is the strongest merge guarantee in the
   literature, at the price of keeping a weighted sample: the lightweight
   coreset is of order `d k log k / eps^2` points *(from memory)*, thousands
   of rows where the state today is 3k.
5. **Distributed form**: Balcan, Ehrlich and Liang (2013), "Distributed
   k-means and k-median clustering on general topologies", NIPS (title
   confirmed, venue *from memory*): a coreset built from the sites' local
   solutions, total size `O~(kd + nk)` over n sites. It samples the rows, so
   it is not a merge of saved states.
6. **Both states still in the warm-up buffer** and the union within the cap:
   concatenate the buffers and seed once. The seeding rules are
   deterministic given the buffer and `seed`, so this is the one-pass state.
7. What does not merge: the metric each shard assigned under was its own
   (the feature moments add exactly; the assignments made under the old
   metric do not move); `far` summaries pool by rule 1 but each was "far"
   against its own shard's cut; `r2_typical`, `far_cut`, the split-merge and
   checkpoint counters are functions of the path. Mini-batch k-means
   (Sculley 2010, "Web-scale k-means clustering", WWW, pp. 1177-1178) has no
   merge of its own; its centres are per-centre running means, which is
   rule 1.

Exactness: **exact** for rule 6, and for rule 1 as "the summary of the pooled
rows". The merged clustering against one pass: **none**. Against the union's
optimal inertia: **bounded** by a constant factor (5 for k-median at
b = c = 1, larger for k-means, *from memory*), and by `(1 + eps)` only with a
coreset in the state. For well-separated clusters both shards found, the
constrained optimum of rule 2 is the unconstrained one and the excess is 0;
that is a measurement to make, not a theorem to cite.

Cost: rules 1 to 3 add no state. Rule 4 would add a weighted sample, of the
order of `8 (p + 1) d k log k / eps^2` bytes (k = 8, p = 20, eps = 0.1: about
5.6 MB against 8 KB, constants unstated), bounded and independent of stream
length in the constant-size forms but a large growth: the user's call before
it lands. Independent of stream length otherwise.

Oracle: `sklearn.cluster.KMeans(n_clusters=k).fit(centres,
sample_weight=n)` on the 2k centres for rule 2, and `KMeans` on the union's
rows for the inertia ratio; `numpy` mean and variance of the pooled rows for
rule 1; `scipy.optimize.linear_sum_assignment` for the matching;
`sklearn.cluster.Birch` holds the same CF arithmetic as a second opinion.

Recommendation: offer rule 6 as exact, else rule 2 (weighted k-means over the
2k summaries, pooled by `merge_welford`), documented as "the best clustering
that keeps each shard's clusters whole", with the inertia ratio to `KMeans`
on the union measured over seeds. No coreset unless the user wants the
`(1 + eps)` promise at its memory.

### hmm online EM (`hmm.rs`)

State: K emission accumulators (`EwCov`, about `8 (d (d + 1) / 2 + d + 3)`
bytes each, or `EwCovDiag`), the filtered `p` (K), the transition counts `A`
(K^2), `n_eff`, and a seeding buffer before the first fit. K = 3, d = 4:
about 0.5 KB. Learning is from the filtered joint `xi` of consecutive states
at the running parameters.

Merge rules:

1. **Expected counts add at fixed parameters.** Baum-Welch over several
   independent sequences sums each sequence's expected transition and
   emission statistics before the M-step (Rabiner 1989, "A tutorial on hidden
   Markov models and selected applications in speech recognition", Proc. IEEE
   77, *from memory*; `hmmlearn`'s `fit(X, lengths=[...])` does this). That
   sum is exact, but its premise is one parameter value for every sequence.
2. **Online EM keeps running averages of those statistics**, each row's
   responsibilities taken at the parameters of the moment: Cappé (2011),
   "Online EM algorithm for hidden Markov models", Journal of Computational
   and Graphical Statistics, pp. 728-749 (confirmed; volume 20(3) *from
   memory*), built on a recursion for the smoothing functionals; Mongillo and
   Deneve (2008), "Online learning with hidden Markov models", Neural
   Computation 20 *(from memory)*; Le Corff and Fort (2013), "Online
   expectation maximization based algorithms for inference in hidden Markov
   models", Electronic Journal of Statistics 7 (confirmed), which processes
   blocks. None of them states a merge of two runs. rheon's `A` uses the
   filtered joint, not Cappé's smoothing recursion, so his consistency
   argument is not a statement about this model either.
3. **The merge the state allows**: match the states (centre distances under
   the emission covariances, assignment problem), then add `A` and pool each
   matched pair of emission accumulators (weight, mean and co-moments are
   sums: the `EwCov` pooling, exact as the moments of the pooled weighted
   rows). This is rule 1's sum with each shard's responsibilities taken at
   its own running parameters: one aggregation round, in the sense of the
   quantile section.
4. **Concatenation loses one transition.** B cold starts its filter from the
   uniform `1/K`, where one pass would carry A's last `p` through `Pi`. The
   merged `A` lacks the seam's `xi` and B's first rows were filtered from the
   wrong prior; a Gaussian HMM's filter forgets its start geometrically under
   the usual mixing conditions (Le Gland and Mevel 2000, "Exponential
   forgetting and geometric ergodicity in hidden Markov models", Mathematics
   of Control, Signals and Systems 13, *from memory*), so the count error is
   a few rows' weight out of B's. The larger error is B's parameters having
   been learned from a cold seed. Resuming A's state on B's rows has neither
   and is exact today.
5. **Shards: none for the transitions.** `A` counts consecutive rows. A shard
   holding every second row of a chain sees the chain at lag 2, whose
   transition matrix is `Pi^2`, not `Pi`; an irregular interleaving sees a
   mixture of powers. The emission moments still pool (rule 3), because a
   row's emission does not depend on which rows are its neighbours, though
   its responsibility does.
6. **Mixture reduction** (next section) applies if the states cannot be
   matched one to one: 2K emission components reduced to K by moment-matched
   merges, with `A`'s rows and columns summed for merged states.

Exactness: **exact** for the emission pooling as moments of the pooled
weighted rows, and for `A` as a sum of the two shards' counts; **none**
against one pass; **none** for shards' transitions, by construction. No
published bound for a merged online-EM HMM that I could find.

Cost: nothing added; `8 K^2` for `A`, independent of stream length.

Oracle: `hmmlearn.hmm.GaussianHMM` (not a dev dependency today; it would be
declared in the dev group): one EM iteration from fixed parameters over
`lengths=[n_A, n_B]` checks that summed statistics give the pooled M-step
(rule 1), and a converged fit on the concatenation is the statistical target;
`statsmodels.tsa.regime_switching.MarkovRegression` as a second opinion for
K = 2, d = 1. The matching by `scipy.optimize.linear_sum_assignment`.

Recommendation: no shard merge; say why (rule 5). For concatenation, steer to
resuming. If a cold-B merge is wanted, rule 3 with the matching made explicit
and reported, labelled one aggregation round, with the distance to a
converged `hmmlearn` fit measured.

### mixtures (no model of this name; `hmm`'s emissions, `micro`, `ew_class`)

Merge rules:

1. **Sufficient statistics add under one set of responsibilities.** EM's
   E-step for an exponential-family mixture is a sum over rows of
   `(r_ij, r_ij x_i, r_ij x_i x_i')`; Neal and Hinton (1998), "A view of the
   EM algorithm that justifies incremental, sparse, and other variants", in
   Learning in Graphical Models (title as recalled; the search returned only
   secondary citations) show that the statistics may be updated a block at a
   time. Across shards the sum is the full E-step **when every shard
   evaluates the responsibilities at the same parameters**: Nowak (2003),
   "Distributed EM algorithms for density estimation and clustering in sensor
   networks", IEEE Trans. Signal Processing 51 *(from memory)*. That is exact
   and needs a pass per round.
2. **Online EM**: Cappé and Moulines (2009), "On-line
   expectation-maximization algorithm for latent data models", JRSS B
   71(3):593-613 (confirmed): a stochastic-approximation average of the
   statistics, converging to a stationary point of the Kullback-Leibler
   divergence at the maximum-likelihood rate; Cappé (2011), "Online
   expectation-maximisation", in Mixtures: Estimation and Applications
   (confirmed as a chapter, arXiv 1011.1745). Each row's responsibilities are
   at the parameters of the moment, so two cold runs' statistics, added, are
   one aggregation round and not the one-pass state.
3. **Hard labels make it exact.** Where the component is observed
   (`ew_class`: one accumulator per label), responsibilities are 0 or 1 and
   do not depend on the fit, so the per-label sums add and the merged state
   is the one-pass state. (Read from the module's description, not checked in
   the code.)
4. **The union mixture is exact, at 2K components.** With weights
   `n_A pi_A / n` and `n_B pi_B / n` the 2K-component mixture is exactly the
   weight-pooled density of the two fits. Nothing is lost; the state doubles
   at each merge, so it is bounded only if reduced.
5. **Reduction to K by moment-matched merges.** Two components
   `(w_i, mu_i, P_i)` and `(w_j, mu_j, P_j)` become one with the pair's
   weight, mean and covariance (`P = (w_i P_i + w_j P_j) / w + w_i w_j
   (mu_i - mu_j)(mu_i - mu_j)' / w^2`), exact in the first two moments. Which
   pair: Runnalls (2007), "Kullback-Leibler approach to Gaussian mixture
   reduction", IEEE Trans. Aerospace and Electronic Systems 43 *(from
   memory)*: merge the pair of least
   `B(i, j) = 1/2 [(w_i + w_j) ln det P_ij - w_i ln det P_i - w_j ln det P_j]`,
   an upper bound on the Kullback-Leibler divergence the merge adds; earlier,
   Salmond's joining algorithm and West (1993), "Approximating posterior
   distributions by mixtures", JRSS B 55 *(both from memory)*. The bound is
   computable at merge time from the state alone.
6. **Micro-clusters** (`micro.rs`): each is a clustering feature and pools
   exactly (kmeans rule 1); CluStream (Aggarwal, Han, Wang and Yu 2003, "A
   framework for clustering evolving data streams", VLDB) and DenStream (Cao,
   Ester, Qian and Zhou 2006, "Density-based clustering over an evolving data
   stream with noise", SDM) *(both from memory)* merge the closest pair when
   the budget is full, which is the 2m to `max_clusters` reduction. The
   `eps` radius bound may be broken by a pooled pair, so a merge either pools
   only pairs whose pooled radius stays within the bound or keeps both.

Exactness: **exact** for rule 1 (common parameters), rule 3 (labels) and
rule 4 (2K components); **bounded** for rule 5, by `sum B(i, j)` over the
merges made, in nats of Kullback-Leibler divergence: for two equal-weight
unit-variance components one standard deviation apart in one dimension,
`P_ij = 1.25`, `B = 1/2 (w_i + w_j) ln 1.25 = 0.112 (w_i + w_j)` nats (my
arithmetic from the formula as recalled); **none** for rule 2 against one
pass.

Cost: rule 4 doubles the component state per merge (`8 K (d (d + 1) / 2 + d
+ 1)` bytes a copy); rules 1, 3, 5 and 6 add nothing. Independent of stream
length.

Oracle: `sklearn.mixture.GaussianMixture` on the union for the statistical
target, and one M-step from summed statistics checked against its
`_estimate_gaussian_parameters` on the pooled responsibilities (a private
function: prefer `numpy` weighted mean and `numpy.cov(aweights=...)`);
`scipy.stats.multivariate_normal` to evaluate the Kullback-Leibler divergence
of rule 5 by quadrature or Monte Carlo against the bound.

Recommendation: rule 3 where labels exist; for a soft mixture, match and pool
(rule 5), reporting the summed bound `B` with the merged state so the caller
sees the cost of each merge. Do not grow K.

### held_median (`held_median.rs`)

State: the residuals a refit buffer holds, as `(|e_i|, r_i)` with the decay
kept in parts; the tree is rebuilt, not saved. Bounded by the refit buffer's
cap (`refit_rows`, default max(500, 10k) rows, part 1), 16 bytes a held pair
before the rows themselves.

Merge rules:

1. **Union of the held rows.** The read is defined on the multiset of
   `(|e|, r)`, so the merge is the multiset union with B's weights put on A's
   decay scale (concatenation: A's `r` are unchanged and B's are divided by
   A's final decay product, a multiplication in the same `m 2^k` parts;
   shards: the decays are common). The module says a restored buffer replays
   its rows' decays in order to the bit, so concatenated buffers replayed in
   stream order give the one-pass tree. Exact, and bit-exact for
   concatenation by that replay (as I read the module docs; the replay path
   was not traced in code).
2. **Two medians do not make a median**, but they bracket it: the weighted
   median of a union lies between the two weighted medians (at most half the
   weight of each part is below the smaller one, at least half of each at or
   below the larger). That is the only bound available from the two reads
   alone, and it is the whole gap between the shards' scales.
3. **Exact selection needs the data.** Munro and Paterson (1980), "Selection
   and sorting with limited storage", Theoretical Computer Science 12 *(from
   memory)*: one pass and exact selection need memory linear in n. So past
   the buffer the choice is a sketch:
   - KLL (Karnin, Lang and Liberty 2016, part 1): additive rank error, kept
     under merges. Apache DataSketches' table gives, at the default k = 200,
     a normalized rank error of 1.33% single-sided and 1.65% double-sided,
     the fit to the 99th percentile of the measured maximum error
     (https://datasketches.apache.org/docs/KLL/KLLAccuracyAndSize.html,
     confirmed). Unweighted, no decay.
   - DDSketch (part 1): relative value error `a`, exact bucket addition;
     takes weights and decay. This is what `EwQuantile` already is, and it is
     what the scale reads once the refit has fired.
   - t-digest (Dunning and Ertl, "Computing extremely accurate quantiles
     using t-digests", arXiv 1902.04023, confirmed): the ordered union of two
     digests is a valid digest under the scale function, but the merged
     digest need not equal the one built in one pass, and the guarantee is on
     size, not accuracy; Cormode, Mishra, Ross and Veselý (2021), "Theory
     meets practice at the median: a worst case comparison of relative error
     quantile algorithms", KDD (title confirmed, venue *from memory*) build
     inputs on which its error is large.
   - Greenwald and Khanna (2001), "Space-efficient online computation of
     quantile summaries", SIGMOD: deterministic `eps n`, but not fully
     mergeable (Agarwal et al. 2013, part 1) *(from memory)*.
4. The residuals held are each shard's own fit's, as in part 1; while both
   refits wait there is no fit yet for them to differ by if the residuals are
   about a common location, and otherwise the refit on the concatenated
   buffer recomputes them (huber rule 4).

Exactness: **exact** (rule 1), within the cap; from the two medians alone
**bounded** by the bracket `[min, max]`; beyond the buffer, a sketch's bound
(`a` for the log buckets, 1.33% of rank for KLL at k = 200).

Cost: rule 1 holds `n_A + n_B` pairs, which must fit the existing cap; when
it does not, the refit should fire on the union instead (huber rule 4). No
new state, independent of stream length.

Oracle: `numpy.quantile(x, 0.5, weights=w, method="inverted_cdf")` on the
union where its definition matches the module's "first value at which the
running weight reaches half" (to be checked for ties); the existing
`weighted_median_sorted` is the definition; `statsmodels.stats.weightstats.
DescrStatsW.quantile` as a second opinion.

Recommendation: merge by union and replay, exact; if the union passes the
cap, fire the refit on it. No sketch is needed here: the model already
changes to `EwQuantile` at the point a sketch would start.

### tails (`tails.rs`)

State: `W`, `sum w^2`, the mean and the central sums `M_2, M_3, M_4`: six
floats, 48 bytes a slot. No sketch: these are moments.

Merge rules:

1. **Pairwise central-moment formulas.** Pébay, Terriberry, Kolla and Bennett
   (2016), "Numerically stable, scalable formulas for parallel and online
   computation of higher-order multivariate central moments with arbitrary
   weights", Computational Statistics 31(4):1305-1325 (confirmed), the paper
   the module already cites; the row update there is this merge with a set of
   one. With `d = mu_B - mu_A`, `W = W_A + W_B`:

   ```text
   M_2 = M_2A + M_2B + d² W_A W_B / W
   M_3 = M_3A + M_3B + d³ W_A W_B (W_A − W_B) / W² + 3 d (W_A M_2B − W_B M_2A) / W
   M_4 = M_4A + M_4B + d⁴ W_A W_B (W_A² − W_A W_B + W_B²) / W³
         + 6 d² (W_A² M_2B + W_B² M_2A) / W² + 4 d (W_A M_3B − W_B M_3A) / W
   mu  = mu_A + d W_B / W          sum w² adds
   ```

   Written from the standard form and checked by hand against `tails.rs`'s
   row update (B a single row: all three reduce to the module's lines); not
   copied from the paper's text. Concatenation ages A first (`W`, `M_j` by
   the decay over B's span, `sum w^2` by its square), exactly `Tails::age`.
2. Skewness, kurtosis and Jarque-Bera are functions of the six numbers, so
   they follow. Kish's size is `W^2 / sum w^2` and follows too.
3. **What is merged is the residuals as scored.** They are recursive
   residuals of each shard's own fit, so a cold B's early residuals are not
   the ones a single pass would have scored. The moment merge is exact; its
   input is path-dependent upstream.
4. If tail *quantiles* were wanted and not moments: DDSketch for relative
   error in the tails, or the relative-rank-error sketch of Cormode, Karnin,
   Liberty, Thaler and Veselý (2021), "Relative error streaming quantiles",
   PODS *(from memory)*, both mergeable; and the moments themselves are a
   mergeable sketch of the quantiles (Gan, Ding, Tai, Sharan and Bailis 2018,
   "Moment-based quantile sketches for efficient high cardinality aggregation
   queries", PVLDB 11 *(from memory)*), with no worst-case error bound.

Exactness: **exact**, shards and concatenation alike, in real arithmetic; the
sums are taken in another order, so the last bits differ from one pass (part
1's separate question), and at `d` large against the spread the `d^4` term
carries the cancellation the module's own note describes for raw sums, to a
smaller degree.

Cost: nothing added: 48 bytes, independent of stream length.

Oracle: `scipy.stats.skew` and `scipy.stats.kurtosis` (biased, the module's
existing oracle) and `statsmodels.stats.stattools.jarque_bera` on the
concatenated residuals at unit weights; for weights, the definition in
`numpy` on the union.

Recommendation: implement rule 1; it is the one family of this part with an
exact merge for both cases and no caveat of its own.

### Part 2 summary

| family | rule | exactness |
|---|---|---|
| kmeans | pool summaries (`merge_welford`) chosen by weighted k-means on the 2k centres | exact in the warm-up buffer and as pooled summaries; clustering none vs one pass, constant factor vs the optimum, `1 + eps` only with a coreset |
| hmm | match states, add `A`, pool emission moments | sums exact; none vs one pass; shards none for transitions (`Pi^2`) |
| mixtures | add statistics; else moment-matched reduction | exact with labels, common parameters or 2K components; reduction bounded by Runnalls' `B` |
| held_median | union of the held rows, replayed | exact within the cap; two medians alone only bracket |
| tails | Pébay's pairwise moment formulas | exact (last bits aside) |

### Part 3: what changes from parts 1 and 2

The definitions of part 1 stand. These six are **sequential**: each state is
a statistic of a path in time order (a run length, a wealth, a cumulative
excess, a prefix correlation, a tracked radius), so two things recur. First,
**shards have no one-pass meaning** for most of them: interleaved rows carry
no order in the saved state, and a quantity counted in rows or integrated
over the clock is counted twice. Second, a cold B is a **restart at the
seam**, and a restart is itself a statement: "a new run began here". Where
the statistic composes over a restart (a wealth, a cumulative sum against a
fixed reference) the merge is exact or valid; where B's rows were scored
against a quantity learned from the past (a running mean, a scale, a fit, a
radius) it is not, and a change exactly at the seam is absorbed by the cold
start and seen by neither state. Resuming A's state on B's rows is the exact
concatenation for all six today.

Searches were made on 2026-11-20. Citations marked *(from memory)* were not
confirmed by a search; the bounds marked "derived here" are this note's own
algebra from the module's stated recursions, not a published result, and were
not run.

### bocpd (`bocpd.rs`)

State: the run vector (per run: row count, weight, mean, scatter; `d` or
`d*d` scatter) and each run's log joint `P(r_t = r, x_1:t)`, pruned below
`prune_below` and capped at `max_run`.

Merge rules:

1. **No merge of two run-length posteriors is offered** by Adams and MacKay
   (2007), "Bayesian online changepoint detection", arXiv 0710.3742
   (confirmed), nor by Fearnhead and Liu (2007), "On-line inference for
   multiple changepoint problems", JRSS B 69(4):589-605 (confirmed), whose
   resampling is a rule for *dropping* hypotheses, not combining filters. The
   Rust `changepoint` crate and the Python `bayesian_changepoint_detection`
   package expose no merge that I know of *(from memory, not checked)*.
2. **The straddling runs merge exactly** (derived here from the product
   partition structure: Barry and Hartigan 1992, "Product partition models
   for change point problems", Annals of Statistics 20 *(from memory)*). A
   one-pass run of length `n_B + j` is A's run `j` continued through all of
   B. Its statistics are A's run `j` pooled with B's oldest run (the one
   holding all `n_B` rows), a conjugate sum, and its joint is

   ```text
   P(r_T = n_B + j, x_1:T) = P_A(r_s = j, x_1:s) · (1 − H)^{n_B} · m(A_j ∪ B) / m(A_j)
   ```

   with `m` the conjugate marginal likelihood (closed form for the
   normal-inverse-Wishart and normal-inverse-gamma pairs), since the product
   of a run's posterior predictives telescopes to that ratio. Exact for
   `gaussian` and `diag` with a constant hazard, provided B's oldest run was
   not pruned or folded.
3. **The runs that began inside B do not.** B's joint for a run begun at
   `tau` is the one-pass joint with the evidence `Z(tau) = P(x_1:tau,
   change at tau)` replaced by B's own, which assumed a change at the seam.
   The ratio `Z(tau) / Z_B(tau)` varies with `tau`, and computing it needs,
   for every `tau` in B, the statistics of B's prefix up to `tau` to extend
   each of A's runs: `O(n_B)` state B did not keep. Taking the ratio as
   constant is the approximation "B's posterior, conditional on a change at
   the seam"; it is close when B contains a decisive changepoint (then
   `Z(tau)` for later `tau` is dominated by paths through it and the ratio
   is nearly constant) and arbitrary otherwise. No published bound found.
4. `robust`: a row's weight is read from the run's own predictive, so B's
   rows were weighted against B's runs; neither rule 2's pooled statistics
   nor its marginal ratio hold. None.
5. Shards: a run length counts rows in time order; none. (Several streams
   with *common* changepoints is a different model, a product of emissions
   on one clock, not a merge of two states.)

Exactness: **none** overall. Rule 2 is exact for its runs; rule 3 has no
bound. If B's oldest run is gone (a decisive change inside B, or `max_run`
reached), the straddling runs cannot be rebuilt at all, and the merged state
is B's alone, which is what one pass also holds up to mass below
`prune_below` in B's own accounting, not a bound on the one-pass mass.

Cost: nothing added; the merged vector is at most `runs_A + runs_B`, folded
at `max_run` as a row would fold it. A run is about `8(2d + 4)` bytes
(`diag`) or `8(d^2 + d + 4)` (`gaussian`) plus its log joint, an estimate
from the struct's fields; bounded by `max_run`, independent of stream length.

Oracle: the unpruned recursion on the concatenated stream in `numpy`, with
predictives from `scipy.stats.t` / `scipy.stats.multivariate_t` (the
definition by hand, the density by the library); the straddling-run mass of
rule 2 against the product of those predictives.

Recommendation: no merge; steer concatenation to resuming. If one is wanted,
rule 2 plus B's runs as they are, named "B's posterior given a change at the
seam, plus A's runs carried through", `gaussian` and `diag` only.

### seqtest (`seqtest.rs`)

State: per target `n_pos`, `n_neg`, `log_e_pos`, `log_e_neg`; no decay.

Merge rules:

1. **Add the log wealths, add the counts.** B's stakes are predictable (a
   function of B's earlier rows, in `[0, 1)`), so `E_A · E_B` is a
   non-negative supermartingale under the module's null (conditional on the
   whole past) over the concatenated stream, and Ville's inequality holds
   for it at every row. The same argument covers shards read in time order:
   at each row one factor moves, by a stake fixed before the row. This is
   the product rule for sequential e-values: Vovk and Wang (2021),
   "E-values: calibration, combination and applications", Annals of
   Statistics 49(3):1736-1754 (confirmed), and their "Merging sequential
   e-values via martingales", arXiv 2007.06382 (confirmed to exist; the
   theorem was not read); Ramdas, Grünwald, Vovk and Shafer (2023),
   "Game-theoretic statistics and safe anytime-valid inference",
   Statistical Science 38 *(from memory)*.
2. **The average `(E_A + E_B) / 2`** is an e-value under any dependence
   (Vovk and Wang 2021: the arithmetic mean is essentially the only
   admissible symmetric merge without independence, as reported by a later
   paper of Wang's). Valid, weaker than the product, and not needed here.
3. **Unclipped, the wealth is a function of the counts**: the module's
   closed form `E_n = 2^n B(n⁺ + ½, n⁻ + ½) / B(½, ½)`, the
   Krichevsky-Trofimov mixture (Krichevsky and Trofimov 1981, "The
   performance of universal encoding", IEEE Trans. Information Theory 27
   *(from memory)*). So the two-sided unclipped wealth of the union follows
   from the summed counts exactly, shards and concatenation alike. The
   shipped one-sided processes clip the stake at 0, which makes each a
   function of the path (the clip binds whenever the running difference has
   the other sign), and their one-pass value is not recoverable.

Exactness: rule 1 is **exact as a test** (level `alpha` holds at every row,
no approximation) and **not the one-pass wealth**. For the unclipped
mixture the gap is bounded on one side: one pass is at most `½ ln n_A + ½ ln
n_B + 2 ln 2` nats above the product (10.6 nats at `n_A = n_B = 10^4`; from
`P_KT ≥ P_ML / (2√n)` *(from memory)* and the maximum likelihood of the
union being at most the parts' sum), and unboundedly below it when the sign
rate differs between the parts, where the restart helps. Rule 3 is
**exact**, for a statistic the model does not emit.

Cost: nothing added: 32 bytes a target, independent of stream length.

Oracle: `scipy.special.betaln` for rule 3 on the summed counts; for rule 1,
the recursion replayed with a restart at the seam, and the level by
simulation under the null (rejections at `1/alpha` at most `alpha`).

Recommendation: implement rule 1, documented as "a valid test of the same
null, restarted at the seam; not the wealth of one pass". Consider emitting
the two-sided mixture of rule 3 at a merge, since it is exact from counts.

### drift (`drift.rs`, Page-Hinkley)

State: `n`, `w`, `mean`, `cum`, `min_cum` (40 bytes).

Merge rules:

1. **A cumulative sum and its running minimum compose**: for increments
   that do not depend on the past,

   ```text
   cum = cum_A + cum_B          min_cum = min(min_A, cum_A + min_B)
   ```

   which is the Lindley / max-plus composition of Page's CUSUM (Page 1954,
   "Continuous inspection schemes", Biometrika 41(1/2):100-115, confirmed;
   Hinkley 1971, "Inference about the change-point from cumulative sum
   tests", Biometrika 58 *(from memory)*). Exact for a CUSUM against a
   *fixed* reference level.
2. **Page-Hinkley's reference is the running mean**, so B's increments
   `e_t − mean_t − delta` used B's cold mean. The one-pass excess differs
   from rule 1's by `Σ_B d_t (mean_t^B − mean_t^one)`, which is not in
   either state. The mean itself merges exactly (age A's weight, pool).
   Derived here, at unit spacing with A at its steady weight `1/(1 − λ)`:
   `mean_j^B − mean_j^one = λ^j (mean_j^B − mean_A)`, so

   ```text
   |cum_merged − cum_one| ≤ Δ · λ / (1 − λ),    Δ = sup_j |mean_j^B − mean_A|
   ```

   At a half-life of 100 rows `λ/(1 − λ)` is 143.8, so `Δ = 0.1` error
   units gives at most 14.4 error-rows, to be read against `threshold`.
   With no decay the factor is `n_A ln(1 + n_B / n_A)` and does not settle.
3. **Two more numbers make rule 2 exact in that case**: B keeping `C0 = Σ
   d_j D_j` and `C1 = Σ d_j D_j mean_j^B` (`D_j` the decay since B's start)
   gives the correction `C1 − mean_A · C0` (16 bytes). Away from A's steady
   weight, or on an irregular clock, it is an approximation again.
4. Libraries: `river.drift.PageHinkley` has no merge *(from memory)*. ADWIN
   (Bifet and Gavaldà 2007, "Learning from time-changing data with adaptive
   windowing", SIAM SDM *(from memory)*) keeps an exponential histogram
   (Datar, Gionis, Indyk and Motwani 2002, "Maintaining stream statistics
   over sliding windows", SIAM J. Computing 31 *(from memory)*), whose
   buckets can be concatenated; that is a different detector.
5. Shards: the literature's rule is to run a CUSUM per stream and combine
   the statistics (sum or maximum) against a threshold of the combination's
   own: Mei (2010), "Efficient scalable schemes for monitoring a large
   number of data streams", Biometrika 97; Tartakovsky and Veeravalli
   (2008), "Asymptotically optimal quickest change detection in distributed
   sensor systems", Sequential Analysis 27 *(both from memory)*. That is a
   new detector, not the one-pass statistic: each shard integrates its
   excess over the same clock, so the sum counts the span twice.

Exactness: **bounded** for concatenation under decay (rule 2's bound,
`Δ λ/(1 − λ)`; 14.4 error-rows in the stated case), exact with rule 3's two
numbers when A is at steady weight on a unit clock; **none** with no decay
and for shards. A change at the seam is under-counted: B's cold mean starts
at the new level.

Cost: nothing (rules 1-2), or 16 bytes (rule 3); independent of stream
length.

Oracle: `numpy.cumsum` and `numpy.minimum.accumulate` on the concatenated
errors for rule 1 and for the one-pass statistic (the module's own test
does this); `river.drift.PageHinkley` at unit spacing as a second opinion
would be a new dev-group library, and its forgetting factor is on the sum,
not the mean *(from memory)*, so it would need care to match.

Recommendation: steer concatenation to resuming. If a cold-B merge is
wanted, rule 1 with the pooled mean, labelled approximate with the bound
reported; rule 3 only if the user wants 16 bytes for exactness in one case.
Refuse shards.

### breaks (`breaks.rs`)

State, per slot: the scale's sums (`v1`, `v2`, `vq`), `S1`, `S2`, `Z1`,
`Z2`, the fourth-moment pair, under a horizon the lag products `P1_l`,
`P2_l` and a ring of the last `L = 2h` studentized residuals; and the twin
fits (`TwinFit`: two `EwCov` and cross terms, `O(k^2)`).

Merge rules:

1. **Age and add the sums** (`S1`, `S2`, `Z1`, `Z2`, the scale's, the
   fourth moment's; the twin fits' Gram sums as `EwCov` merges). Exact as
   the pooled sums of what each state scored, the same rule as `tails`.
2. **What was scored is path-dependent twice**: `v` is a recursive residual
   of each state's own fit, and `z = v / s` reads the scale of the rows
   before, so B's `z` were studentized by B's young scale after B's own
   wait of `SCALE_ROWS`. Neither can be redone from the sums.
3. **Under the null the merged CUSUM keeps its distribution.** Brown,
   Durbin and Evans (1975), "Techniques for testing the constancy of
   regression relationships over time", JRSS B 37(2):149-192 *(from memory;
   the search did not return it)*: recursive residuals are independent
   `N(0, σ²)` under a constant relationship with Gaussian errors. B's cold
   recursive residuals are a different orthogonal transform of B's errors
   than the one-pass continuation's, equally independent and independent of
   A's. So run once, the merged `Z1` is a sum of `n_A + n_B − 2k` such
   terms where one pass has `n − k`: a valid CUSUM with `k` fewer terms
   (and `SCALE_ROWS` fewer `z`), not the same path.
4. **A break at the seam is invisible to rule 1**: B's fit starts in the
   new regime, and its residuals are clean. The statistic that sees it is
   the comparison of the two fits, which the two states give exactly: Chow
   (1960), "Tests of equality between sets of coefficients in two linear
   regressions", Econometrica 28 *(from memory)*, from the two Gram sums;
   it is `break_wald`'s idea with A and B as the slow and fast fits.
5. **Under a horizon the seam's lag products are missing**: `P_l` needs
   `z_t z_{t−l}` across the seam, A's ring has its last `L` values, B's
   first `L` are gone. At most `L(L + 1)/2` products per sum are omitted, of
   `(n_A + n_B) L`; treating the seam as a break in the lag sums (as `rcov`
   in part 1) is exact for what it says.
6. Related: Ploberger and Krämer (1992), "The CUSUM test with OLS
   residuals", Econometrica 60; Chu, Hornik and Kuan (1995), "MOSUM tests
   for parameter constancy", Biometrika 82; Zeileis, Leisch, Hornik and
   Kleiber (2002), "strucchange: an R package for testing for structural
   change in linear regression models", J. Statistical Software 7 *(all
   from memory)*. None gives a rule for merging two partial CUSUM states.
7. Shards: the sums add and `Z1 / sqrt(S2)` keeps its null, but it is no
   longer a path in time, and the lag products pair rows that were not
   neighbours. None under a horizon.

Exactness: **none** against one pass; rule 1 is exact as pooled sums, rule
3 gives the null distribution exactly in the run-once Gaussian case, rule 5
is a bounded omission (`L(L + 1)/2` products, zero with no horizon).

Cost: nothing added; about 72 bytes plus `16 L` for the lag sums and `8 L`
for the ring per slot, and the twin fits' `O(k^2)`; independent of stream
length.

Oracle: `statsmodels.stats.diagnostic.recursive_olsresiduals` on the
concatenated rows and on each part (the recursive residuals and the CUSUM
path); the Chow statistic from two `statsmodels` OLS fits against the
pooled one; the level by simulation.

Recommendation: add the sums with the seam as a break in the lag products,
documented as "a CUSUM over both parts' own residuals", and report the
A-against-B Wald (Chow) distance at the merge, since the seam is where the
summed statistic is blind.

### corrchange (`corrchange.rs`)

State: `window`: a ring of the `2 * window_rows` latest rows, the held
permutation critical value, its counter and the RNG. `monitor`: the span's
rows, or after the history fills, the history's level and scale per pair
with the monitored rows and their running moments.

Merge rules:

1. **`window`: the rows are the state**, so concatenation is the last
   `2 * window_rows` rows of A's ring followed by B's: all B's when `n_B`
   reaches the ring. The block summaries are rebuilt from the ring.
   Exact in the statistic. The held critical value and RNG position are
   not the one-pass ones (A's draws would have advanced the stream), so
   the merged critical value is another valid draw of the same null,
   equal in distribution, not in value; redraw on the merged ring.
2. **Correlations of a union merge exactly from co-moment summaries**, the
   rule `window_moments` already uses: Pébay (2008), "Formulas for robust,
   one-pass parallel computation of covariances and arbitrary-order
   statistical moments", Sandia report SAND2008-6212 *(from memory)*.
3. **`monitor`'s statistic does not**: `Q = max_j (j/√T)|ρ̂_j − ρ̂_T|/D̂`
   reads every prefix correlation of the span and a long-run variance
   centred at the span's means (Wied, Krämer and Dehling 2012, "Testing for
   a change in correlation at an unknown point in time using an extended
   functional delta method", Econometric Theory 28(3):570-589, confirmed).
   It needs the span's rows in order. So the merge is exact when the seam
   falls on a span boundary (B's spans are then the one-pass spans), or
   when both parts lie inside one span whose rows are still held (union,
   replay). Otherwise B's spans are phase-shifted from one pass's: each is
   a valid closed test at its level, on other rows.
4. **The sequential form would merge where the closed one does not**: Wied
   and Galeano (2013), "Monitoring correlation change in a sequence of
   random variables", J. Statistical Planning and Inference 143 *(from
   memory; the module reads their SFB 823 preprint)*: the detector at step
   `k` compares the history's correlation with the correlation of the `k`
   monitored rows, two co-moment summaries (rule 2), scaled by `D̂` from the
   history. A's state past its history and B's monitored moments would give
   it exactly, if B is taken as monitored rows against A's history.
5. Shards: a ring holds rows without their times; none.

Exactness: `window` **exact** (the statistic; the critical value in
distribution); `monitor` **exact at a span boundary or inside one held
span, else none** (a valid test on shifted spans). A change at the seam is
seen by `window` (the windows straddle it) and by `monitor` only if the
seam is inside a span.

Cost: nothing added: `16 k window_rows` bytes for the ring, `8 k
span_rows` for a span, bounded by the window, independent of stream length.

Oracle: `numpy.corrcoef` on the union for rule 2; `Q` by the paper's
definition in `numpy` on the concatenated span, with
`scipy.stats.kstwobign.ppf(0.95)` (1.3581) for the boundary;
`scipy.stats.permutation_test` for the window's null. No Python library
computing the Wied-Krämer-Dehling statistic is known to me.

Recommendation: `window`: merge the rings and redraw the critical value,
exact. `monitor`: merge only at a span boundary or within a held span, and
refuse otherwise with the reason.

### adaptive conformal (`conformal.rs`)

State, per slot: the multiplier `m`, the EW coverage pair, the EW mean
weight pair, the held-rows mean; about 8 numbers. The residual spread
`σ` is the slot's, outside this struct.

Merge rules:

1. **No state merge in the online literature.** Gibbs and Candès (2021),
   "Adaptive conformal inference under distribution shift", NeurIPS
   (confirmed); Angelopoulos, Candès and Tibshirani (2023), "Conformal PID
   control for time series prediction", NeurIPS (confirmed); Gibbs and
   Candès (2024), "Conformal inference for online prediction with arbitrary
   distribution shifts", JMLR (confirmed), and Zaffran, Féron, Goude, Josse
   and Dieuleveut (2022), "Adaptive conformal predictions for time series",
   ICML *(from memory)*, *aggregate experts* run side by side at several
   step sizes, which is two trackers kept, not one merged.
2. **Concatenation: take B's `m`.** Derived here by coupling: two trackers
   fed the same scaled scores at a constant step `η` never move apart, and
   close by `η` each time a score falls between them, so

   ```text
   |m_B − m_one| ≤ max(|m_0 − m_A|, η)      and  ≤ η once they have met
   ```

   with `m_0 = Φ⁻¹(1 − α/2)` B's start. At `α = 0.1` (`m_0 = 1.645`) and a
   rate of 0.05, the gap is at most `|1.645 − m_A|` at once and 0.05 after
   about `|1.645 − m_A| / 0.05` scores between the two. With weights the
   step is `η w / w̄` and `η` reads as the largest step. It assumes the
   same `σ` and residuals in both, which a cold B's model does not give
   until its spread has caught up.
3. **The coverage guarantee survives exactly**: each part obeys the
   module's telescoped bound, so the rows shown intervals over both miss
   at a rate within `2(B + η)/(η T)` of `α`, for every sequence. The EW
   coverage and mean-weight pairs merge as EW means (age, pool), exact as
   scored.
4. **Shards: a weighted mean of the two `m`**, by analogy with part 1's
   parameter averaging; it lies between two trackers of the same quantile.
   No bound against one pass; under stationary scores each `m` sits within
   the tracker's own wander of the `1 − α` quantile of `s / σ`.
5. **The mergeable alternative is a quantile of the scores, not a
   tracker**: federated conformal prediction merges clients' calibration
   scores through a quantile sketch (Lu, Yu, Karimireddy, Jordan and
   Raskar 2023, "Federated conformal predictors for distributed uncertainty
   quantification", arXiv 2305.17564, confirmed; ICML *from memory*), or
   takes a quantile of the clients' quantiles (Humbert, Le Bars, Bellet and
   Arlot 2023, "One-shot federated conformal prediction", ICML *(from
   memory)*). Coverage then rests on (partial) exchangeability, not on the
   every-sequence bound, and the sketch's error adds (the log buckets' `a`
   of part 1). `EwQuantile` of `s / σ` would be this.

Exactness: concatenation **bounded** (rule 2: `max(|m_0 − m_A|, η)`, then
`η`; 0.05 in the stated case once met), with the coverage bound **exact**
at twice its constant; shards **none**.

Cost: nothing added: about 64 bytes a slot, independent of stream length.

Oracle: the recursion replayed in `numpy` on the concatenated scores (the
coupling bound asserted on every row), and `numpy.quantile` of `s / σ` at
`1 − α` on the union for where both should sit. MAPIE's adaptive conformal
option steps `α_t`, not the radius *(from memory)*, so it is no oracle for
this form.

Recommendation: concatenation takes B's `m` with the pooled EW pairs,
documented with rule 2's bound, or resumes A (exact). For shards, the
weighted mean labelled a heuristic, or refuse; if mergeable intervals are
wanted in their own right, rule 5 is a separate model.

### Part 3 summary

| family | rule | exactness |
|---|---|---|
| bocpd | carry A's runs through B (pooled conjugate statistics, marginal ratio); B's runs as they are | straddling runs exact (`gaussian`, `diag`); runs begun in B none; `robust` and shards none |
| seqtest | add log wealths and counts | exact as a test; not the one-pass wealth (unclipped: within `½ ln n_A + ½ ln n_B + 2 ln 2` above); unclipped mixture exact from counts |
| drift | `cum` adds, `min` composes, mean pools | bounded under decay by `Δ λ/(1 − λ)`; exact with 16 bytes more in one case; no decay and shards none |
| breaks | age and add the sums; seam as a break in the lag products | sums exact; none vs one pass; null kept run once; blind at the seam (use Chow) |
| corrchange | `window`: merge the rings; `monitor`: only at a span boundary | `window` exact; `monitor` exact at a boundary or in a held span, else none |
| conformal | take B's `m`, pool the EW pairs | bounded by `max(|m_0 − m_A|, η)`, then `η`; coverage bound exact at twice the constant; shards none |

## Summary

All 17 families. The verdict is against "the state of one pass over the
union" (part 1's definition); "resume" means continuing A's state on B's
rows, which is exact today for a concatenation and needs no merge.

| family | verdict | recommendation |
|---|---|---|
| huber | exact while both refit buffers are live; else none | concatenate buffers where it applies; else add sums, labelled, distance to `RLM` measured |
| quantile | as huber; sketch bounded by `a` | same as huber; measure against `QuantReg` |
| robust covariance | exact as a break at the seam; shards none | offer the break form; unbroken form needs a small state growth (ask); refuse shards |
| sgd | none (convexity bound only) | weighted averaging, labelled approximate; resume for concatenation |
| pa | none (convexity bound only) | weighted averaging labelled approximate, or no merge; resume |
| ftrl | exact for the summed objective; none vs one pass | add `zz`, `n`, `d`; document as the fit of both linearised losses |
| kmeans | exact in warm-up and as pooled summaries; clustering none | weighted k-means over the 2k summaries; measure the inertia ratio |
| hmm online EM | sums exact; none vs one pass; shards none | no shard merge; resume; cold-B merge only with explicit matching |
| mixtures | exact with labels or 2K components; reduction bounded (Runnalls) | add statistics where labelled; else match and pool, report the bound |
| held_median | exact within the cap | union and replay; refit on the union past the cap |
| tails | exact (last bits aside) | implement Pébay's pairwise formulas |
| bocpd | none (straddling runs exact) | no merge; resume; if wanted, "B given a change at the seam plus A's runs carried through" |
| seqtest | exact as a test; not the one-pass wealth | add log wealths and counts; say it is a restart at the seam |
| drift | bounded under decay (`Δ λ/(1 − λ)`); else none | resume; cold-B merge labelled with its bound; refuse shards |
| breaks | none (sums exact, null kept) | add sums with a break at the seam; report the A-against-B Chow distance |
| corrchange | `window` exact; `monitor` exact only at a span boundary | merge rings and redraw; refuse `monitor` off a boundary |
| adaptive conformal | bounded (`max(|m_0 − m_A|, η)`, then `η`); shards none | take B's `m`, pool the EW pairs; shards a labelled heuristic or refused |

Exact for both cases with no caveat: `tails`, `held_median` (within its
cap), `corrchange`'s `window`. Exact for a stated reading: `ftrl` (summed
objective), `rcov` (break at the seam), `seqtest` (validity), the labelled
or buffered cases of `huber`, `quantile`, `kmeans` and mixtures. Bounded:
`drift` under decay, `conformal`, the sketches, the mixture reduction. No
one-pass meaning: `sgd`, `pa`, `hmm`, `bocpd`, `breaks`, and every shard
merge of part 3. The bounds marked "derived here" in part 3 are this note's
algebra and should be checked by the oracle named beside each before any is
promised in a docstring.
