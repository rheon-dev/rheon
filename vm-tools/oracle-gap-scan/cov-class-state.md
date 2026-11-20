# Oracle gap scan: covariance and dependence, classification, smoothing, clustering, state models

*2026-11-20. Read-only scan of rheon 2.0.0 (`.claude/slots/1`, `5122b435`). Nothing was built or run
except `inspect.signature` on the dev venv's oracle classes (sklearn 1.9.1, statsmodels 0.15.0,
hmmlearn, river 0.26.1, polars 2.0.0) and a read of `river/covariance/ewa.py` and
`bayesian_changepoint_detection/*.py`. Scope: options, outputs and variants of existing models
only. New model families (PLAN tasks 485-510) are out of scope and are named only where an item
overlaps one.*

Checked against: `docs/ENHANCEMENTS.md`, `docs/PLAN.md` (the main checkout's copy, which has
tasks 483-510; the slot's copy at 2.0.0 stops at 481), `docs/CLUSTERING.md`,
`docs/records/ORACLE-COVERAGE.md`.

**Feasibility key.** *exact*: a recursion equal to the batch answer on the same rows (under
decay, equal to the weighted definition). *on read*: a function of state rheon already keeps,
computed when a row is emitted. *checkpoint*: an O(k³) or iterative solve on the state at a
cadence, as `pca_every` does. Like `pca`, it is chunk-invariant on a learned-row schedule.
*embargo*: exact after a fixed delay of L rows. *approx*: a recursive estimator that only
converges. *passes*: needs more than one pass. Not online.

## Two findings before the tables

1. **The shrinkage rejection's premise no longer holds.** ENHANCEMENTS' list of river exclusions
   ("Shrinkage covariance estimators") says Ledoit-Wolf, OAS and Shrunk have "no streaming form
   that keeps our exactness guarantee". But:
   - OAS's intensity needs only `tr(S)`, `tr(S²)`, `p` and `n`, all on read from `ew_cov`'s state
     (with `n_kish` for `n` under decay).
   - Ledoit-Wolf's `π̂ = (1/n)·Σ‖dᵢdᵢᵀ − S‖²_F` reduces to `Σ‖dᵢ‖⁴ − n‖S‖²_F`. This is exact from
     one more scalar fourth moment and one k-vector third moment (`Σ‖d‖²d`). sklearn's own
     `ledoit_wolf_shrinkage` computes the same `Σᵢ‖xᵢ‖⁴` term.
   - river 0.26.1 now ships `LedoitWolfCovariance`, `OASCovariance` and `ShrunkCovariance`, each
     with a `fading_factor` (`river/covariance/ewa.py`). Its LW uses an EW `π̄` heuristic with
     `n ≈ 1/f`, so river is a reference here, not an exact oracle. The exact oracle is sklearn on
     a window at `half_life=inf`.

   Recommend reopening the item.
2. **Task 493's premise is wrong at 2.0.0.** Task 493 says "the DCC correlation recursion is in
   `corr` today with given parameters". It is not. `corr.py` holds no DCC function. The only
   DCC-like recursion is `deco(dynamics="linear", alpha, beta)`, on the scalar equicorrelation,
   and its docstring says it departs from the paper's DECO-DCC `Q` recursion. A full-matrix DCC
   `Q` recursion with given `(a, b)` (and ADCC's `g`) has to exist before 493 can estimate it. It
   is listed under `ew_cov`/`corr` below.

## ew_cov, ewdiag, ew_lag_cov, and `rheon.corr`

What rheon has now:
- `stats`: mean, var, std, cov, corr, partial_corr, mahal, lag_corr.
- Parameters: `bias`, `precision_prior` (a fading ridge), `mahal_quantiles`, `pca`/`pca_every`,
  `lags`, `window_size`/`closed`, plus the common clock, decay and weight parameters.
- `ewdiag` and `ew_lag_cov` are internal accumulators, the diagonal and the lag ring, and have no
  surface of their own.
- `rheon.corr` offers `nearest(w=)`, `shrink(target="identity"|"constant", alpha=, x=)`,
  `fisher_se`, `equicorr*`, `absorption`, `spectral`, `mp_*`, `epps_invert` and others, all
  offline on a matrix.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| sklearn.covariance.EmpiricalCovariance(assume_centered=True) (also on LedoitWolf, OAS, ShrunkCovariance, GraphicalLasso) [1] | none: `ew_cov` always centers at the EW mean | exact: `Σw·x xᵀ` with no mean, the uncentered second moment | **high**: RiskMetrics-style EWMA covariance of returns is uncentered, and so is `rcov(kind="plain")`. The two cannot be compared today | sklearn `EmpiricalCovariance(assume_centered=True).fit(window rows)` at `half_life=inf`, windowed; numpy `(X.T*w)@X/Σw` under decay | new |
| sklearn.covariance.OAS() / river.covariance.OASCovariance(fading_factor) [1][8] | none on the stream. `corr.shrink` has no OAS target | on read: `tr S`, `tr S²`, `n_kish` | **high**: a well-conditioned covariance and precision at `p` near `n_kish`, the regime a short half-life puts every wide bank in | sklearn `oas(X_window)` covariance and `shrinkage_` to 1e-12 at `half_life=inf`; the weighted definition under decay | rejected (ENHANCEMENTS, river exclusions, "Shrinkage covariance estimators"); premise refuted above |
| sklearn.covariance.LedoitWolf() / river LedoitWolfCovariance [1][8] | `corr.shrink(target="identity")` offline, given the raw rows `x` | exact: +1 scalar `Σ‖d‖⁴` and +1 k-vector `Σ‖d‖²d` beside the co-moments (weighted Pébay-style update) | **high**: as above; also feeds LDA's `shrinkage="auto"` | sklearn `ledoit_wolf(X_window)` (covariance and intensity) exact in a window | rejected (same entry); premise refuted |
| sklearn.covariance.ShrunkCovariance(shrinkage=δ); river ShrunkCovariance(delta, target="constant_correlation") [1][8] | `precision_prior` adds `s·prior·I` that fades; `corr.shrink(alpha=)` offline | on read | medium: a fixed convex shrink toward `μI` or toward constant correlation does not wash out as the ridge does | sklearn `shrunk_covariance(S, δ)` on the emitted matrix; river `ShrunkCovariance` with matching fading | rejected (same entry) |
| sklearn.covariance.GraphicalLasso(alpha, mode, tol, max_iter) [2] | none (`partial_corr` is dense) | checkpoint: glasso on the EW `S` every `pca_every`-style cadence | medium: sparse partial correlations and a sparse precision for wide `p` | sklearn `graphical_lasso(emp_cov=S_emitted, alpha)` on the same `S`: identical input, so exact to solver tolerance | new |
| sklearn.covariance.EmpiricalCovariance.score / `score_samples`-style Gaussian log-likelihood [1] | `mahal` only. `deco`, `hmm` and `bocpd` emit `loglik`, but `ew_cov` does not | on read: logdet from the Cholesky that `mahal` already factors | medium: an outlier/regime score in nats, comparable with `hmm`'s and `bocpd`'s `loglik` | scipy `multivariate_normal.logpdf(x, mean_before, cov_before + ridge)` per row | new |
| sklearn.covariance.MinCovDet(support_fraction) / EllipticEnvelope(contamination) [1] | none | **passes** (C-steps on subsets) | low | n/a | not online |
| river.stats.Skew(bias) / Kurtosis(bias); polars rolling_skew(bias) / rolling_kurtosis(fisher, bias) [8][9] | none | exact: weighted third and fourth central moments (Pébay), O(k) per feature | medium: tail and asymmetry features beside `std` | scipy `stats.skew/kurtosis(bias=)` in a window at `half_life=inf`; polars `rolling_skew` on a row clock; the weighted definition under decay | new |
| river.stats.SEM(ddof) [8] | none | on read: `std / sqrt(n_kish)` | low | river `SEM` at no decay | new |
| pandas ExponentialMovingWindow.cov(other, pairwise, bias) / .corr(times=) as a window *operator* [10] | the `ew_cov` model in the bank. No `rp.ewm_cov`/`ewm_corr` expression in `ops.py` | exact | medium: a pairwise EW correlation as a column inside a query, beside `ewm_var` | pandas `ewm(times=).cov/corr` on a row clock and an irregular one | new |
| statsmodels.tsa.stattools.pacf (Durbin-Levinson from acf) [3] | `lag_corr` at the given lags | on read when lags 1..L are all kept | low-medium | statsmodels `pacf(method="ldb")` | planned (task 485 scans `acf`/`pacf`) |
| DCC `Q` recursion with given `(a, b)`; ADCC `g` (Engle 2002; R `rmgarch::dccfilter` with fixed pars) [11] | none. `deco(dynamics="linear")` is scalar-only | exact: O(k²) a row on `ew_cov`'s standardized row | **high**: the per-pair dynamic correlation that task 493 assumes exists | longhand from Engle (2002) eq. 6-8 with numpy, and R `rmgarch` `dccfilter` at fixed parameters as a second opinion | new as a prerequisite; estimation is planned (task 493, whose premise is wrong) |
| sklearn.covariance.oas(X) as a `rheon.corr.shrink(target="oas")` target [1] | `shrink` identity/constant (LW) | offline helper | low-medium | sklearn `oas` | new |
| statsmodels.stats.correlation_tools.corr_clipped / cov_nearest [3] | `corr.nearest` (Higham) | offline | low | statsmodels `corr_clipped` | new |

## marginal, marglag

What rheon has now:
- The `ModelBank.marginal()` table, one row per pair: `n_kish`, the means and variances, `cov`,
  `corr`, `beta`, `t_stat`, `lag_corr_*`, `n_serial`, `t_stat_serial`, `phi_*`, the bins and
  `split_gain`.
- Parameters: `lags`, `cross_lags`, `serial_rule`, the bin rules and `window_size`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| scipy.stats.linregress → `intercept`, `intercept_stderr`, `stderr`, `pvalue` [4] | `beta` and `t_stat`; no intercept, no p-value | on read: `mean_y − beta·mean_x`, Student-t on `n_kish − 2` | low-medium: saves every user the same three lines, and the p-value is what a screen thresholds | scipy `linregress` on window rows at `half_life=inf`; statsmodels `WLS` under decay | new |
| scipy.stats.pearsonr(...).confidence_interval(confidence_level) [4] | `corr`; `rheon.corr.fisher_se` offline | on read (Fisher z, with `n_serial` for a serial-corrected interval) | low-medium | scipy `pearsonr(...).confidence_interval()` in a window | new |
| statsmodels.tsa.stattools.acf(qstat=True, alpha=) (Ljung-Box Q and Bartlett bands on `lag_corr_xx`/`yy`) [3] | `lag_corr_xx`, `lag_corr_yy` at the lags given | on read over the kept lags | low | statsmodels `acf(qstat=True)` in a window | new (partly task 485) |
| scipy.stats.spearmanr / kendalltau [4] | none | approx (binned) or window | — | — | planned (task 510) |

## deco

What rheon has now: `dynamics="ew"|"linear"`, `alpha`, `beta`, `blocks`; outputs `u`, `rho`,
`loglik`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| Engle & Kelly (2012), the less biased alternative to `u`: `1 − (1/(n−1))·Σ(ζᵢ − ζ̄)²` [12] | none: the docstring says "which this does not compute" | exact, O(n) a row | medium: `u` is biased low (E[u] ≈ 0.20 at a true 0.30, n = 6), which the docstring itself warns about | the definition, written from the paper, against `u` on simulated equicorrelated draws (scipy `multivariate_normal.rvs`) | new |
| DECO through the DCC `Q` recursion (Engle & Kelly eq. 21; R `dccmidas`) [12][13] | `"linear"` is a reparameterized scalar recursion | exact with given parameters | low-medium: matches the published model and dccmidas | R `dccmidas` at fixed parameters, beside the existing scipy `logpdf` | new (estimating its parameters belongs with task 493) |

## rcov

What rheon has now: `kind="kernel"|"preavg"|"plain"`, `kernel="parzen"` only, `bandwidth`,
`max_bandwidth`, `jitter`, `theta`, `psd`, `block_rows`, `preavg_rows`, `noise_stride`,
`iv_stride`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| R highfrequency `rKernelCov(kernelType = "bartlett", "tukeyhanning", "cubic", "qs", ...)` [14] | `kernel="parzen"` only | exact per block | medium (Parzen is the PSD-guaranteed one; the others are for comparison with the literature) | R `rKernelCov` on the same block; hfhd `krvm` where its kernel matches | planned (task 499, "realized kernels") |
| highfrequency `rBPCov`, `rSemiCov`, `rTSCov`, realized beta [14] | none | exact per block | medium | R highfrequency | planned (task 499) |

## ew_class

What rheon has now: `covariance_type="full"|"tied"|"diag"` (QDA, LDA, naive Bayes),
`precision_prior` (a fading ridge), `classes`, windows. Outputs `class`, `p_<c>`, and `coef`,
which holds the class means.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| GaussianNB(priors=) / LDA(priors=) / QDA(priors=) [5] | none: `π_c` is always the decayed class counts | exact (constant `π`) | medium: under class imbalance or a known base rate, the decayed counts move the boundary with the label mix | sklearn `GaussianNB(priors=)`, `QuadraticDiscriminantAnalysis(priors=)` at `half_life=inf`, windowed | new |
| QDA(reg_param=r): `(1−r)Σ_c + r·I`; GaussianNB(var_smoothing=ε·max var) [5] | `precision_prior`: `+ ω_c I`, which fades as the class fills | exact | medium: a regularizer that does not wash out, scaled to the data | sklearn `QDA(reg_param=)` and `GaussianNB(var_smoothing=)` in a window (today's tests set the prior to 1e-9 to dodge the mismatch) | new |
| LDA(shrinkage="auto"\|float, covariance_estimator=OAS()) [5] | none | LW/OAS on read on the tied matrix (see `ew_cov`) | medium | sklearn `LDA(solver="lsqr", shrinkage="auto")` in a window | new; depends on reopening the shrinkage rejection |
| LDA.transform / n_components (discriminant coordinates) [5] | none | checkpoint: the eigenproblem of `S_w⁻¹S_b` from class means and the tied matrix | low-medium: a supervised low-dimensional score per row | sklearn `LDA(solver="eigen").transform`, up to sign | new |
| predict_log_proba / decision_function [5] | `p_<c>` only | on read (already computed as `l_c`) | low: no underflow at extreme posteriors | sklearn `predict_log_proba` | new |
| river.naive_bayes.MultinomialNB(alpha) / BernoulliNB(alpha, true_threshold) / ComplementNB [8] | none: Gaussian only | exact (decayed counts) | low: count or indicator features | river with no decay; sklearn `MultinomialNB.partial_fit` | new |

## holt

What rheon has now: `half_life` (the level), `trend_half_life`, `trend` (False gives SES), a
weighted line-fit start. Output `pred`, `resid`, `coef=[level, trend]`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| statsmodels Holt(damped_trend=True) + fit(damping_trend=φ) [6] | none | exact | medium | statsmodels `Holt(damped_trend=True).fit(..., optimized=False)` | planned (task 488) |
| Holt(exponential=True) (multiplicative trend) [6] | none | exact | low-medium | statsmodels | planned (task 488, "every ETS variant") |
| Holt(initialization_method="known", initial_level=, initial_trend=) [6] | none: always the line-fit start | exact | low-medium: deploys from a known state, and gives an exact oracle from row 1 rather than "once the weight saturates" | statsmodels `Holt(initialization_method="known").fit(smoothing_level=1−2^(−1/h), smoothing_trend=..., optimized=False).fittedvalues` to 1e-12 from the first row | new |
| HoltWintersResults.forecast(h) (`l + h·b`) [6] | one-step `pred` only | on read | medium: an h-step forecast is the usual baseline a horizon target is compared with | statsmodels `.forecast(h)` at each row's state | new |
| fit(optimized=True): smoothing parameters by MLE [6] | fixed half-lives | approx (recursive gradient) | — | — | planned (task 488) |

## kmeans, micro (the cluster module)

What rheon has now:
- `kmeans`: `n_clusters`, `warm_rows`, `init="first"|"farthest"|"k-means++"|"lloyd"` (lloyd is the
  best by inertia of ten k-means++ starts), `seed`, `update_every_rows`, `split_merge`,
  `dead_frac`, `standardize`, `scale_floor`. Outputs `cluster`, `dist`, `dist_second`.
- `micro`: `eps`, `beta_mu`, `max_clusters`, `prune_every`, `macro_link`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| sklearn KMeans(init=ndarray), MiniBatchKMeans(init=ndarray) [7] | none (`hmm` takes `means=`; `kmeans` does not) | exact | medium: deploys with known centers, skips the warm-up, and gives an exact oracle from row 1 | sklearn `KMeans(init=C, n_init=1, max_iter=1)` assignments; sklearn `MiniBatchKMeans.partial_fit` one row at a time with no decay | new |
| KMeans.transform (the distance to every center) [7] | `dist`, `dist_second` | on read, O(k·p) | medium: soft cluster features for a downstream regression | sklearn `.transform` at the frozen centers | new |
| KMeans.inertia_ / score (an EW SSQ) [7] | none | exact, O(1) a row | medium: the elbow across several `k` in one bank | sklearn `KMeans.score(X, sample_weight=decay)` at frozen centers | planned (task 118, "k chosen by EW SSQ"; CLUSTERING.md §8 and §12) |
| sklearn.metrics.calinski_harabasz_score [7] | none | exact from `(n, Σx, Σ‖x‖²)` per cluster (CLUSTERING.md §8) | low-medium | sklearn `calinski_harabasz_score` on window rows and labels | new (CLUSTERING.md §8 marks it streamable, not built) |
| river.cluster.KMeans(p=) (Minkowski metric); spherical (cosine) k-means; fuzzy c-means `fuzzifier`; Huber [8] | Euclidean on the standardized scale | exact | low-medium | river `KMeans(p=1)` has its own update, so its labels serve as a reference only; for fuzzy, scikit-fuzzy `cmeans` at fixed centers | designed, not built (CLUSTERING.md §6.2 "Variants") |
| KMeans(init="random") [7] | the four rules | exact | low | sklearn | new |
| river.cluster.DBSTREAM(intersection_factor) (macro links by shared density) [8] | `micro` links by center distance (`macro_link·eps`) | exact, O(M²) at checkpoints | low-medium: shared density follows a manifold better than center distance | river `DBSTREAM` (reference, not exact) | new |
| river.cluster.CluStream(n_macro_clusters) (a weighted k-means macro step over the micro-clusters) [8] | linkage macro step only | checkpoint | low-medium | sklearn `KMeans(sample_weight=micro weights).fit(micro centers)` at the checkpoint: exact | new |
| MiniBatchKMeans(reassignment_ratio) [7] | `dead_frac` | — | — | — | covered |

## hmm

What rheon has now:
- `n_components`, `covariance_type="full"|"tied"|"diag"`, `precision_prior`, `learn` (all or
  nothing), `transition_prior`, `transition`, `means`/`covs` (given; entered at one row's weight),
  `init`, `seed`, `warm_rows`, `exog_tvtp`/`tvtp_coef`.
- Outputs: `filtered_state<j>`, `predicted_state<j>`, `state`, `loglik`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| hmmlearn GaussianHMM.predict_proba (smoothed posteriors); statsmodels MarkovRegression smoothed_marginal_probabilities [15][6] | filtered and predicted only | embargo: a fixed-lag forward-backward over the last L rows, emitting row t−L's smoothed probabilities at row t | medium-high: smoothed regimes are what users label history with | hmmlearn `predict_proba(prefix)[t−L]` at fixed parameters (`learn=False`), which is exact | new (task 496 does this for `kalman` only) |
| GaussianHMM.decode(algorithm="viterbi") [15] | `state` = argmax of the predicted probabilities | embargo or approx (online Viterbi with a fixed lag is exact once the survivor paths coalesce) | medium | hmmlearn `decode(prefix)` at fixed parameters | new |
| GaussianHMM(params="t", init_params=...) (learn some parameters, hold the others) [15] | `learn` True or False for everything | exact | medium: a domain-fixed transition matrix with learned emissions, or the reverse | composite, as `hmm(learn=True)`'s test is now; hmmlearn at `learn=False` for the parameters held | new |
| GaussianHMM(covariance_type="spherical") [15] | full, tied, diag | exact | low-medium | hmmlearn `"spherical"` at fixed parameters (`TestHmmIsHmmlearns`) | new |
| GaussianHMM(means_prior, means_weight, covars_prior, covars_weight, min_covar) [15] | `precision_prior` (a ridge); given states count as one row | exact (the conjugate MAP update in the online EM) | medium: given states that hold for N rows rather than one, and a floor that stops a state collapsing | hmmlearn's M-step formulas, with a one-step `fit(n_iter=1)` on a block as the reference | new |
| GaussianHMM(startprob_prior, startprob) [15] | uniform start | exact | low: matters for the first rows only | hmmlearn at fixed parameters with a given `startprob_` | new |
| GaussianHMM.get_stationary_distribution(); MarkovRegression expected_durations `1/(1−p_kk)` [15][6] | none | on read from the transition counts | low-medium | hmmlearn `get_stationary_distribution()`; statsmodels `expected_durations` at fixed parameters | new |
| statsmodels MarkovRegression(exog, switching_exog=True, switching_trend) (a regime-dependent regression) [6] | none: `hmm` takes no targets | approx (online EM with a weighted RLS per state) | medium | statsmodels `MarkovRegression(...).filter(params)` at fixed parameters, as the TVTP test does now | partly idea (task 118, "clusterwise regression"); new as an `hmm` option |
| hmmlearn GMMHMM(n_mix), PoissonHMM [15] | Gaussian emission | approx (online EM) | low | hmmlearn at fixed parameters | new |
| MarkovRegression(switching_variance=False) [6] | `covariance_type="tied"` | — | — | — | covered |

## bocpd

What rheon has now:
- `hazard` (a constant, or a duration on the clock), `hazard_col`, `covariance_type="diag"|"full"`
  (normal-inverse-gamma or normal-inverse-Wishart), the four conjugate priors, `robust_beta`,
  `prune_below`, `max_run`, `warm_rows`.
- Outputs: `p_change`, `run_mode`, `run_mean`, `pred_<f>`, `loglik`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| A hazard that depends on run length: `online_changepoint_detection(hazard_func(r))`; the offline `geometric_prior`/`neg_binominal_prior` gap priors as hazards; Turner et al. (2009) logistic hazard [16][17] | constant, per-row (`hazard_col`) or on the clock | exact, O(1) per run | medium: real regimes have a minimum length, and a negative-binomial or Weibull gap expresses "rarely shorter than N rows" | `bayesian_changepoint_detection.online_changepoint_detection` with `hazard_func = H(r)` (it is called with `np.arange(t+1)`, so it accepts run lengths): exact with pruning off | new |
| A Gaussian with known mean (variance change only); a Gaussian with known variance (mean change only) | NIG/NIW (both unknown); `mean_precision_prior→∞` only approximates the first | exact, conjugate | medium: zero-mean returns whose break is in the volatility | hildensia's StudentT longhand with a fixed mean; scipy `t`/`norm` predictive | new |
| R ocp `onlineCPD(probModel = "p")` (Poisson-Gamma) and Bernoulli-Beta observation models [18] | Gaussian only | exact, conjugate | low-medium: count streams (trades per interval) | R `ocp` | new |
| The predictive variance and quantiles of the run-length mixture [16] | `pred_<f>` (the mean) only | on read: `Σ p(r)(var_r + μ_r²) − μ²`; quantiles need an approximate mixture inversion | medium: an interval for the next row | longhand from scipy `t` per run, weighted by hildensia's posterior | new |
| ocp `minRlength`, `minsep`, `cpthreshold`, `getR` (a top-k run-length vector) [18] | `p_change`, `run_mode`, `run_mean` | exact | low | R `ocp` | new |
| Hyperparameter learning (Turner, Saatçi & Rasmussen 2009) [17] | fixed priors and hazard | approx (online gradient) | low | the definition | new |
| `covariance_type="spherical"` | diag, full | exact | low | scipy | new |

## Window operators and window moments (`ops.py`, `window*.rs`)

What rheon has now:
- Operators: `ewm_mean`, `ewm_sum`, `ewm_rate`, `ewm_quantile` (a decaying sketch accurate to
  0.78%), `ewm_var` and `ewm_std` (`bias`), `increment`.
- Parameters on every operator: `half_life` (`inf` weighs the window evenly), `window_size`,
  `closed` (all four values), `min_samples`, `partial`, `reverse`.
- The windows follow Polars' `rolling_*_by` edges and `ewm_mean_by`.

| oracle feature | rheon equivalent | online feasibility | value | oracle test it enables | status |
|---|---|---|---|---|---|
| polars rolling_min_by / rolling_max_by; river RollingMin/Max/PeakToPeak [9][8] | none | exact: a monotone deque over the rows the window already keeps | **high**: the commonest window feature after the mean and the sum | polars `rolling_min_by/max_by` at every `closed`, exact | new |
| polars rolling_quantile_by(interpolation="nearest"\|"higher"\|"lower"\|"midpoint"\|"linear"), rolling_median_by [9] | `ewm_quantile` at `half_life=inf`, within 0.78%, `inverted_cdf` only | exact in a hard window (an order-statistic structure over the kept rows) | medium: an exact median and the interpolation Polars users expect | polars `rolling_quantile_by` / `rolling_median_by`, exact | new |
| polars rolling_rank_by(method="average"\|"min"\|"max"\|"dense"\|"random", seed) [9] | none | exact, O(log n) with an order-statistic tree | medium: a rank or percentile feature that one wild value does not move | polars `rolling_rank_by` | new |
| polars ewm_mean / ewm_var (adjust=True, the default); pandas `ewm(times=)` (the count-weighted EW mean) [9][10] | time-weighted `ewm_mean`; var/std held to `adjust=False` only; the docstring says no operator offers the count-weighted form | exact (it equals `ewm_sum(x)/ewm_sum(1)` on distinct stamps) | medium: it is Polars' default `ewm_mean` and pandas' `times=` form | polars `ewm_mean(adjust=True, half_life=h)` on a row clock; pandas `ewm(times=, halflife=)` | new |
| polars ewm_*(com=, span=, alpha=) [9] | `half_life` only | on read (`h = ln 2 / −ln(1−α)` on a row clock) | low (the "match the Polars API" rule argues for the aliases) | polars `ewm_mean(alpha=)` against the mapped `half_life` | new |
| polars ewm_*(ignore_nulls=True) [9] | the clock advances over a missing value (`ignore_nulls=False`-like) | exact (a clock that counts only rows with a value) | low | polars `ewm_mean(ignore_nulls=True)` on a row clock | new |
| polars rolling_skew(bias) / rolling_kurtosis(fisher, bias) (row windows only) [9] | none | exact (weighted third and fourth moments) | low-medium | polars on a row clock at `half_life=inf`; scipy `skew/kurtosis` | new |
| polars rolling_std_by(ddof=int) [9] | `bias` (bool) | — | — | — | covered (ddof 0 or 1 is what `bias` maps to) |
| river.stats.IQR / RollingIQR, MAD [8] | two `ewm_quantile` calls give the IQR | the IQR on read; MAD approx (a quantile of the deviations from a moving median) | low | river `RollingIQR` | new (MAD) |
| river.stats.BayesianMean(prior, prior_weight) [8] | none | exact (prior pseudo-weight in the ratio) | low | river `BayesianMean` | new |

## Ranked top ten

1. **Shrinkage on `ew_cov`: OAS, Ledoit-Wolf and fixed `δ` (and as LDA's `shrinkage`).**
   OAS is a read of state rheon already keeps, and LW is exact with one scalar and one k-vector.
   sklearn is exact in a window. Status: rejected in ENHANCEMENTS on a premise this scan refutes,
   and river 0.26.1 ships EW versions.
2. **`ew_cov` uncentered (`assume_centered=True`).** RiskMetrics' EWMA covariance, comparable with
   `rcov(kind="plain")`. sklearn is exact. New.
3. **`rolling_min_by` / `rolling_max_by` operators.** Exact with a monotone deque, and Polars is an
   exact oracle at every `closed`. New.
4. **The DCC `Q` recursion with given `(a, b)`, plus ADCC `g`.** Task 493 assumes it exists, and it
   does not. New, as a prerequisite to 493.
5. **`ew_cov` Gaussian `loglik`.** Nearly free next to `mahal`, and scipy `logpdf` is the oracle.
   New.
6. **`hmm` fixed-lag smoothed probabilities, under an embargo.** hmmlearn's `predict_proba` on the
   prefix is an exact oracle. New; task 496 covers `kalman` only.
7. **`bocpd` run-length hazards (negative-binomial, logistic) and known-mean/known-variance
   emissions.** The hildensia package is an exact oracle for the hazard. New.
8. **Exact windowed quantile/median with `interpolation=`, and `rolling_rank_by`.** Polars is an
   exact oracle. New.
9. **`hmm` partial learning (`params=`), `spherical`, and prior weights for given states.**
   hmmlearn at fixed parameters is the oracle. New.
10. **`kmeans` given centers (`init=array`), `transform` (the distance to every center) and EW
    inertia.** sklearn is exact from row 1. The inertia is planned as an idea in task 118.

Close behind:
- `ew_class` fixed `priors` and a non-fading `reg_param`/`var_smoothing`.
- `holt` `forecast(h)` and a known initial level and trend (this one makes statsmodels exact from
  row 1).
- `deco`'s less biased alternative to `u`.
- The count-weighted `ewm_mean` (Polars `adjust=True`).
- GraphicalLasso at checkpoints.

Not online: MinCovDet and EllipticEnvelope, which need passes. Already planned: damped and
exponential `holt` and fitted smoothing parameters (task 488); more `rcov` kernels and estimators
(task 499); `pacf` (task 485); rank correlations (task 510); DCC estimation (task 493).

## Sources

1. scikit-learn covariance: https://scikit-learn.org/stable/modules/covariance.html; the classes
   at https://scikit-learn.org/stable/api/sklearn.covariance.html (EmpiricalCovariance,
   LedoitWolf, OAS, ShrunkCovariance, MinCovDet, EllipticEnvelope; signatures read from
   sklearn 1.9.1).
2. https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html
3. statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html,
   `.pacf.html`, and https://www.statsmodels.org/stable/stats.html#correlation-and-covariance-matrices
   (`corr_clipped`, `cov_nearest`).
4. scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html and
   `scipy.stats.pearsonr.html`.
5. https://scikit-learn.org/stable/api/sklearn.discriminant_analysis.html (LDA: `solver`,
   `shrinkage`, `priors`, `n_components`, `covariance_estimator`; QDA: `priors`, `reg_param`,
   `shrinkage`) and
   https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
   (`priors`, `var_smoothing`).
6. statsmodels Holt: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.Holt.html
   (`exponential`, `damped_trend`, `initialization_method`, `initial_level`, `initial_trend`);
   MarkovRegression:
   https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html
   (`switching_trend`, `switching_exog`, `switching_variance`, `exog_tvtp`; results
   `smoothed_marginal_probabilities`, `expected_durations`).
7. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html,
   `MiniBatchKMeans.html`, and
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
8. river 0.26.1: https://riverml.xyz/latest/api/overview/ (covariance: EwaCovariance,
   LedoitWolfCovariance, OASCovariance, ShrunkCovariance; cluster: KMeans(p), DBSTREAM,
   DenStream, CluStream; naive_bayes; stats: Skew, Kurtosis, SEM, IQR, MAD, Rolling*,
   BayesianMean). The source read is `river/covariance/ewa.py`.
9. Polars 2.0.0:
   https://docs.pola.rs/api/python/stable/reference/expressions/computation.html (ewm_mean,
   ewm_var and ewm_std with `com`, `span`, `half_life`, `alpha`, `adjust`, `ignore_nulls`,
   `bias`, `min_samples`; rolling_*_by with `min_samples`, `closed`, `ddof`, `interpolation`;
   rolling_rank_by; rolling_skew and rolling_kurtosis). Signatures read locally.
10. pandas: https://pandas.pydata.org/docs/reference/api/pandas.core.window.ewm.ExponentialMovingWindow.cov.html
    and `.corr.html`.
11. Engle (2002), "Dynamic Conditional Correlation", JBES 20(3); R rmgarch
    https://cran.r-project.org/package=rmgarch (`dccspec`, `dccfilter` with fixed parameters).
12. Engle & Kelly (2012), "Dynamic Equicorrelation", JBES 30(2) (the alternative estimator, and
    eq. 21).
13. R dccmidas: https://cran.r-project.org/package=dccmidas
14. R highfrequency: https://cran.r-project.org/package=highfrequency (`rKernelCov`
    `kernelType`, `rBPCov`, `rSemiCov`, `rTSCov`).
15. hmmlearn: https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.hmm.GaussianHMM
    (`covariance_type` spherical, diag, full or tied; `min_covar`, `startprob_prior`,
    `transmat_prior`, `means_prior`, `means_weight`, `covars_prior`, `covars_weight`,
    `algorithm`, `params`, `init_params`; `predict_proba`, `decode`,
    `get_stationary_distribution`; GMMHMM, PoissonHMM).
16. https://github.com/hildensia/bayesian_changepoint_detection
    (`online_changepoint_detection(data, hazard_func, observation_likelihood)`, `constant_hazard`,
    `StudentT`; offline `const_prior`, `geometric_prior`, `neg_binominal_prior`). Adams & MacKay
    (2007), arXiv:0710.3742.
17. Turner, Saatçi & Rasmussen (2009), "Adaptive Sequential Bayesian Change Point Detection"
    (NIPS workshop): the logistic hazard and hyperparameter learning.
18. R ocp: https://cran.r-project.org/package=ocp (`onlineCPD`: `hazard_func`, `probModel`
    "g" or "p", `minRlength`, `minsep`, `cpthreshold`, `getR`).
