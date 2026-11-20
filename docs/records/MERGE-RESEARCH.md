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
- Part 3 pending.

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
