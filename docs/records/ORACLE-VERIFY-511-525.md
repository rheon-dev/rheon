# Oracle verification for PLAN tasks 511-525

Read-only check of every oracle library the tasks cite, against current official docs or
source (fetched 2026-11-20). Verdicts: CONFIRMED (name, spelling and meaning as cited),
CORRECTED (the right name or setting is given), MISSING (the library has no such thing; an
alternative is given). "Exact for the comparison" says whether the cited setting makes the
oracle equal to rheon's proposed quantity, or only a reference.

Versions seen: scikit-learn 1.9.1 docs, SciPy 1.18.0 docs, river 0.26.1 source and 0.27.0 docs
(riverml.xyz/latest), filterpy source on master and PyPI 2.0.0 (the PyPI JSON read).

## Task 511. Shrinkage covariance on `ew_cov` and `ew_class`

- scikit-learn `OAS(*, store_precision=True, assume_centered=False)`: CONFIRMED (1.9.1).
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
  Exact for the comparison only if rheon follows sklearn's formula, which is NOT the paper's:
  the docs say it "differs from Eq. 23" of Chen et al. (2010): the `2/p` factor on `tr(S^2)` is
  omitted. Source (`_oas`, 1.9.X `sklearn/covariance/_shrunk_covariance.py`):
  `alpha = mean(S**2)` (= tr(S^2)/p^2 for symmetric S), `mu = tr(S)/p`,
  `num = alpha + mu^2`, `den = (n+1)*(alpha - mu^2/p)`, `shrinkage = 1.0 if den == 0 else min(num/den, 1.0)`;
  `p == 1` returns `S` with shrinkage 0.0. `S` is the biased (divisor n) covariance, centered
  unless `assume_centered=True`. The task must say "OAS as scikit-learn computes it" and
  replicate the omitted factor; an implementation from Eq. 23 will differ at small p.
- scikit-learn `LedoitWolf(*, store_precision=True, assume_centered=False, block_size=1000)`:
  CONFIRMED. `block_size` is memory-only. `shrinkage_` attribute. Source
  (`ledoit_wolf_shrinkage`): divisor n throughout, `X - X.mean(0)` unless `assume_centered`,
  `beta = min(beta, delta)`, `shrinkage = 0 if beta == 0 else beta/delta`; a single feature
  returns 0.0. Target `mu*I`, `mu = tr(S)/p`. Exact in a window and at `half_life=inf`
  (same divisor n). The scan's reduction to `sum||d||^4 - n||S||_F^2` is the same
  term sklearn uses (`X2` sums); not independently re-derived here.
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
- scikit-learn `ShrunkCovariance(*, store_precision=True, assume_centered=False, shrinkage=0.1)`:
  CONFIRMED. `(1-shrinkage)*S + shrinkage*mu*I`, `mu = tr(S)/p`. Exact.
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html
- river `LedoitWolfCovariance(fading_factor=0.5)`, `OASCovariance(fading_factor=0.5)`,
  `ShrunkCovariance(fading_factor=0.5, delta=0.1, target="constant_correlation")`: CONFIRMED at
  0.26.1 (`river/covariance/ewa.py`; docs at 0.27.0 show the same signature).
  CORRECTED detail: river's ShrunkCovariance uses `delta` (not `shrinkage`) and defaults to
  `target="constant_correlation"` (other: `"identity"`); only `target="identity"` compares to
  sklearn's `ShrunkCovariance`. River's LW is a heuristic (n ~ 1/fading_factor), its OAS sets
  n = max(1/fading_factor, 2): references only, as the task says. Not exact.
  https://raw.githubusercontent.com/online-ml/river/0.26.1/river/covariance/ewa.py
  https://riverml.xyz/latest/api/covariance/LedoitWolfCovariance/

## Task 517. `ew_cov` outputs: uncentered covariance and a Gaussian log-likelihood

- scikit-learn `assume_centered=True` (on `EmpiricalCovariance`, also OAS, LedoitWolf,
  ShrunkCovariance): CONFIRMED. `empirical_covariance`: `X.T @ X / n` when centered-assumed,
  `np.cov(X.T, bias=1)` otherwise: divisor n. Exact vs an undecayed or windowed uncentered
  second moment `sum w x x^T / sum w` (with sklearn's n, i.e. unit weights).
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html
- scipy `scipy.stats.multivariate_normal.logpdf(x, mean=None, cov=1, allow_singular=False)`:
  CONFIRMED (SciPy 1.18.0). Exact per row: one row's `logpdf(x, mean, cov)` equals the
  Gaussian log-density `-0.5*(p*ln(2 pi) + ln|S| + mahal^2)`. Use `allow_singular=True` for a
  rank-deficient early `S`.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
- The scan's "scikit-learn `score_samples`-style" log-likelihood: MISSING as a per-row method.
  `EmpiricalCovariance` has `score(X_test, y=None)`, which returns the log-likelihood of the
  test set's empirical covariance under the fit, a single average per-sample value
  (`log_likelihood(emp_cov, precision) = (-sum(emp_cov*precision) + logdet(precision) - p*ln(2 pi))/2`),
  and `mahalanobis(X)` (squared distances). No `score_samples`. Use scipy `logpdf` per row
  as the oracle; `score` on a one-row `X_test` equals the per-row log-density only when the
  row is centered at `location_` (it uses `X_test - location_`), which holds, since a
  one-row test covariance is `d d^T`: `-d^T P d` equals `-sum(emp_cov*precision)`. So `score`
  on one row is a valid second check.
  https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.X/sklearn/covariance/_empirical_covariance.py

## Task 519. `kalman`: `coef_prior`, innovation log-likelihood and Mahalanobis distance

- filterpy `KalmanFilter` attributes `x`, `log_likelihood`, `mahalanobis`: CONFIRMED
  (master `filterpy/kalman/kalman_filter.py`; PyPI reports 2.0.0 released 2026-10-26).
  Read-only, lazily computed from the last `update()`: `log_likelihood = logpdf(x=y, cov=S)`
  (innovation `y` and its covariance `S = H P H^T + R`, both before the update): exact for
  the proposed innovation log-likelihood. CORRECTED detail: `mahalanobis` is
  `sqrt(y^T S^-1 y)`, the distance, NOT the squared form. The test must square it (or rheon
  emit the distance) before comparing with `mahal`-style squared outputs. `update(z, R=None, H=None)`.
  `x`/`P` are the initial state and covariance for `coef_prior`'s comparison (set `x`, `P`
  before the first update).
  https://raw.githubusercontent.com/rlabbe/filterpy/master/filterpy/kalman/kalman_filter.py
  https://pypi.org/pypi/filterpy/json

## Task 512. Standard-error types (HC1-3, `huber`, `quantile`, cluster)

All statsmodels pages are the 0.15.0 docs (https://www.statsmodels.org/stable/).

- `RegressionResults.get_robustcov_results(cov_type='HC1', use_t=None, **kwargs)` and
  `OLS.fit(method='pinv', cov_type='nonrobust', cov_kwds=None, use_t=None)`: CONFIRMED.
  `cov_type` accepts `'HC0'`..`'HC3'`, `'HAC'` (kernels `'bartlett'` default, `'uniform'`;
  `maxlags` required), `'cluster'` (CRV1), `'cluster-crv3'`, `'cluster-jk'`,
  `'hac-groupsum'` (Driscoll-Kraay), `'hac-panel'`, `'fixed scale'`.
  Definitions from source (`regression/linear_model.py`): HC0 `e^2`; HC1 `n/df_resid * e^2`
  (`df_resid = n - rank`, not literally `n - k`); HC2 `e^2/(1-h)`; HC3 `(e/(1-h))^2`, with
  `h` the leverage `x_i' (X'X)^-1 x_i` from the FINAL fit. Exact for the comparison only
  against final-fit residuals and leverages; rheon's per-row sandwich uses out-of-sample
  residuals at each row's fit, so the comparison is statistical (a ratio band), as the HC0
  test is today. The task's "HC2/HC3 from `error_inflation`" is a rheon identity, not
  something statsmodels offers. CORRECTED (scan): `'nw-panel'` is not a `cov_type`; the name is
  `'hac-panel'`.
- `RLM.fit(maxiter=50, tol=1e-08, scale_est='mad', cov='H1', update_scale=True, conv='dev',
  start_params=None, start_scale=None)`: CONFIRMED (the `cov` parameter and `'H1'` default
  exist). CORRECTED on what "the M-estimator sandwich" is: per `RLMResults.bcov_scaled`
  (source `robust/robust_linear_model.py`, docstring quoted), `H1` is NOT a sandwich: it is
  `k^2 * (sum(psi^2) * scale^2 / df_resid) / (mean(psi'))^2 * (X'X)^-1`; `H2` is
  `k * (sum(psi^2) scale^2/df_resid) / mean(psi') * W^-1`; only `H3` is a sandwich:
  `(1/k) * (sum(psi^2) scale^2/df_resid) * W^-1 (X'X) W^-1`, with
  `W = sum psi'(r_i) x_i x_i'`, `k = 1 + (p/n) var(psi')/mean(psi')^2` (`p = df_model + 1`).
  None of the three equals the per-row-weighted Huber-White form `A^-1 [sum psi^2 z z'] A^-1`
  that the task describes (EW sums of `psi' z z'` and `psi^2 z z'`); `H3` is the closest, and
  its middle term is `X'X` times a scalar, not `sum psi^2 z z'`. Oracle verdict: statistical
  only (ratio band), and the test should name `cov="H3"` for the sandwich shape and may
  also compare `H1`. Source: https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/robust/robust_linear_model.py
- `QuantReg.fit(q=0.5, vcov='robust', kernel='epa', bandwidth='hsheather', max_iter=1000,
  p_tol=1e-06, **kwargs)`: CONFIRMED. `vcov` in `'robust'` (default), `'iid'`; `kernel` in
  `'biw'`, `'cos'`, `'epa'` (default), `'gau'`, `'par'`; `bandwidth` in `'hsheather'`
  (Hall-Sheather 1988, default), `'bofinger'`, `'chamberlain'`. Exact for the comparison only
  at no decay and the final fit; with an EW kernel density it is statistical.

## Task 516. Linear equality constraints

- statsmodels `RecursiveLS(endog, exog, constraints=None, **kwargs)`: CONFIRMED (0.15.0,
  https://www.statsmodels.org/stable/generated/statsmodels.regression.recursive_ls.RecursiveLS.html).
  `constraints` takes a 2-D `r x k` array (each row `= 0`), a string, or a tuple `(R, q)` with
  `q` a scalar or length-p vector. The string form is parsed by `FormulaManager().get_linear_constraints`
  with names from the data (`x1`, `x2`, ... for bare arrays), per the source
  (`regression/recursive_ls.py`); "x1 + x2 = 1" is the same grammar as `t_test`.
  Per-row outputs: `res = RecursiveLS(y, X, constraints=...).fit(); res.recursive_coefficients.filtered`
  (k by nobs, the estimate AFTER row t's update) and `.filtered_cov`. Exact for the comparison
  with caveats the task must state: (1) it is expanding-window OLS, so the oracle matches
  `ridge=0`, `half_life=inf`, no window; (2) the filter is diffusely initialized, so
  coefficients are not identified until `k - n_constraints` rows (compare from there on);
  (3) constraints are imposed as "fake" zero-variance observations (source comment: "Linear
  constraints are technically imposed by adding fake endog variables") through the
  Kalman filter, so equality holds to filter precision, not by a closed form; the tolerance
  should be set accordingly (1e-8 is plausible, 1e-12 is not); (4) `filtered` at row t uses row t,
  while rheon's `pred` at row t uses the fit through t-1, so align by one row.
- The scan's second citation `OLS.fit_regularized` for equality restrictions: CORRECTED.
  `fit_regularized` does not impose equalities. `GLM.fit_constrained(constraints,
  start_params=None, **fit_kwds)` does (formula/string or `(R, q)`; it transforms the design
  matrix, so with `sm.families.Gaussian()` it is the closed-form restricted OLS: a better
  oracle for `gram.solve(constraints=)`, final fit, exact to solver tolerance).
  https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_constrained.html

## Task 524. Regression variants

- scikit-learn `BayesianRidge`: CONFIRMED (1.9.1, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html).
  Signature `(*, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
  lambda_2=1e-6, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True,
  copy_X=True, verbose=False)`; `fit(X, y, sample_weight=None)` (weights since 0.20);
  `predict(X, return_std=False)`; attributes `coef_`, `intercept_`, `alpha_`, `lambda_`,
  `sigma_`, `scores_` (with `compute_score=True`), `n_iter_`. CORRECTED on exactness: the
  default is NOT exact evidence maximization. (a) `compute_score` defaults to False (the scan
  wrote True); (b) the four Gamma hyperpriors default to `1e-6`, so the update is MAP over
  the hyperparameters, not pure ML-II: pass all four as `0.0` (allowed by the parameter
  constraint) or give rheon the same pseudo-counts; (c) the loop stops when the change in `coef_`
  is below `tol=1e-3`, so an exact comparison needs a tiny `tol` and large `max_iter`;
  (d) the algorithm is Tipping (2001) Appendix A with MacKay (1992) updates, and the docs
  say these rules "do not guarantee that the marginal likelihood is increasing"; (e) the
  intercept is unpenalized and handled by centering. `sample_weight` rescales rows by
  `sqrt(w)`, so n in the fixed point is the row count, not a Kish n: "decayed weights scaled to
  Kish n" is a rheon choice, which sklearn matches only when it is given the same scaled
  weights.
- scikit-learn `SGDRegressor(learning_rate="optimal")` with `eta = 1/(alpha*(t + t0))`, `t0` from
  "a heuristic proposed by Leon Bottou": CONFIRMED as a setting; CORRECTED on the scan's remark
  "`optimal` is sklearn's default schedule": for `SGDRegressor` the default is
  `learning_rate='invscaling'` (`eta0=0.01`, `power_t=0.25`); `'optimal'` is `SGDClassifier`'s
  default. `t0` is not a user parameter, so a row-by-row oracle needs sklearn's heuristic (read
  `_sgd_fast`/`sag` source or inspect `t_` after `partial_fit`). Also `average=True|int`
  (average begins after that many samples), `penalty in {'l2','l1','elasticnet',None}`,
  `l1_ratio=0.15`, `loss in {'squared_error','huber','epsilon_insensitive',
  'squared_epsilon_insensitive'}`; `learning_rate` also has `'pa1'`, `'pa2'` (added 1.8, only with
  `epsilon_insensitive`). https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
  The task says "averaged `sgd` and `pa`" with scikit-learn: `PassiveAggressiveRegressor(*, C=1.0,
  ..., average=False)` exists with `average` (added 0.19) but is DEPRECATED since 1.8 and
  removed in 1.10; the replacement is `SGDRegressor(loss="epsilon_insensitive", penalty=None,
  learning_rate="pa1" or "pa2", eta0=C)`. Pin the oracle to the SGD spelling.
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html
  Averaging is a uniform average over all iterates (or after the cutoff); the decay-weighted
  version is rheon's, so the oracle is exact only at `half_life=inf`.
- statsmodels `RLM` norms: CONFIRMED names in `statsmodels.robust.norms` (0.15.0, https://www.statsmodels.org/stable/rlm.html):
  `HuberT(t)`, `Hampel(a, b, c)`, `TukeyBiweight(c)`, `TukeyQuartic(c, k)`, `AndrewWave(a)`,
  `RamsayE(a)`, `TrimmedMean(c)`, `StudentT(c, df)`, `LeastSquares()`. Use `TukeyBiweight`
  (not "TukeyBisquare"). Tukey is redescending (non-convex), so IRLS from a different start
  can reach another local minimum: an oracle comparison must start RLM from the same fit.
- Huber's proposal-2 scale: statsmodels `statsmodels.robust.scale.HuberScale(d=2.5, tol=1e-08,
  maxiter=30)` is CONFIRMED to exist and is what `RLM(scale_est=HuberScale())` takes, but its
  docs call it "Huber's scaling for fitting robust linear models" and say it "differs slightly"
  from `scale.Huber`; it includes an `h` (degrees-of-freedom) constant depending on `n, p, d`.
  The class the docs label "Huber's proposal 2 for joint location and scale" is
  `statsmodels.robust.scale.Huber(c=1.5, tol=1e-08, maxiter=30, norm=None)` (location/scale
  only). For a regression scale comparison use `HuberScale`, and state that it is not
  identical to the textbook proposal 2. https://www.statsmodels.org/stable/generated/statsmodels.robust.scale.HuberScale.html
- statsmodels `QuantReg` for "several quantiles in one spec": CONFIRMED (one fit per `q`).
- scikit-learn `Lasso(alpha=1.0, *, fit_intercept=True, ..., positive=False, ...)` (and
  `ElasticNet`): `positive=True` CONFIRMED (1.9.1,
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).
  `fit(X, y, sample_weight=None, check_input=True)` (weights since 0.23) rescales the weights to
  sum to `n_samples`, so the objective is `(1/(2 sum w)) sum w r^2 + alpha ||b||_1`: the same
  per-weight normalization as a decayed fit, so `alpha` is comparable. Exactness: coordinate
  descent to `tol=1e-4`; set `tol` small for a KKT-level (1e-10) comparison.
  Per-feature penalty weights: MISSING in scikit-learn (no penalty-factor parameter on `Lasso`,
  `ElasticNet` or `enet_path`; glmnet's `penalty.factor` is R's). Alternative: fit on columns
  `X_j / p_j` with a uniform `alpha` and divide the fitted coefficient `g_j` by `p_j`
  (`b_j = g_j / p_j`), which solves penalty `alpha * p_j * |b_j|`. The scan's `enet_path` on
  rescaled columns is that same device.
- Vowpal Wabbit "importance-invariant step": CONFIRMED. `--invariant` is a real option
  (`make_option("invariant", invariant)` in `vowpalwabbit/core/src/reductions/gd.cc` on master; the
  wiki lists it as "use safe/importance aware updates", on by default and switched off by
  `--sgd`). Theory: Karampatziakis and Langford, "Online Importance Weight Aware Updates"
  (arXiv 1011.1576). The wiki page is flagged out of date by its own banner. Per
  scan/wiki: `--initial_t` (0), `--power_t` (0.5), `--decay_learning_rate` (1), `--l1`, `--l2`,
  `--ftrl --ftrl_alpha (0.005) --ftrl_beta (0.1)`.
  https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-Line-Arguments

## Task 513. Scores for intervals and losses

MAPIE 1.6.0 (PyPI 2026-10-19; also on master) and scikit-learn 1.9.1.

- MAPIE regression metrics in `mapie.metrics.regression` (source
  https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/mapie/metrics/regression.py):
  `regression_coverage_score(y_true, y_intervals)`, `regression_mean_width_score(y_intervals)`,
  `regression_ssc(y_true, y_intervals, num_bins=3)`, `regression_ssc_score(y_true, y_intervals,
  num_bins=3)`, `regression_mwi_score(y_true, y_pis, confidence_level)`,
  `coverage_width_based(...)`, `hsic(...)`: CONFIRMED (the scan's `intervals` spelling for the
  Winkler function is CORRECTED to `y_pis`, shape `(n, 2, 1)`; `y_intervals` for the others, shape
  `(n, 2)` or `(n, 2, n_levels)`).
  Exactness caveats: (1) `regression_mwi_score` is
  `(sum(width) + (2/(1 - confidence_level)) * sum(penalties)) / n`, i.e. the Winkler score
  with `alpha = 1 - confidence_level`, and it first un-crosses the bounds with min/max: exact
  for a streaming Winkler mean. (2) `regression_ssc` ranks intervals by size ascending and
  splits them into `num_bins` EQUAL-COUNT groups: that needs the whole sorted set, so a
  streaming "size-stratified coverage" with fixed-width or sketch bins matches it only
  approximately; exact only on a finished window through `rp.eval`.
- scikit-learn `mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5,
  multioutput='uniform_average')`: CONFIRMED. `d2_pinball_score(y_true, y_pred, *, sample_weight=None,
  alpha=0.5, multioutput='uniform_average')` (added 1.1): CONFIRMED. CORRECTED on streaming
  exactness: `D2 = 1 - sum rho(y - yhat) / sum rho(y - q_alpha)` where `q_alpha` is the
  empirical alpha-quantile of the WHOLE `y_true` (the null model; NaN when `n < 2`), a
  hindsight quantity: a streaming D2 can only use a running quantile, so it equals sklearn
  only at the end of a window or when the quantile is the batch one. The pinball itself
  (mean) is exact. Docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_pinball_score.html
- scikit-learn `mean_absolute_error(y_true, y_pred, *, sample_weight=None,
  multioutput='uniform_average')`: CONFIRMED. river `metrics.MAE()` with `update(y_true, y_pred,
  w=1.0)`, `revert(...)`, `get()`: CONFIRMED (no rolling variant on the page; use `revert` or
  `utils.Rolling`). Exact. https://riverml.xyz/latest/api/metrics/MAE/ (docs 0.27.0)
- Streaming log loss: scikit-learn `log_loss(y_true, y_proba=None, *, normalize=True,
  sample_weight=None, labels=None, y_pred='deprecated')`: CORRECTED: in 1.9.1 the second argument is
  `y_proba` (`y_pred` is deprecated, removal in 1.11) and there is no `eps` argument: probabilities
  are clipped to `[eps, 1-eps]` with `eps` the dtype's machine epsilon (so a rheon clip must use
  `np.finfo(float64).eps` to match to the bit). river `metrics.LogLoss()` with
  `update(y_true: bool, y_pred, w=1.0)`: CONFIRMED; the page documents no clipping, so a
  comparison with extreme probabilities may differ from sklearn's clip.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
  https://riverml.xyz/latest/api/metrics/LogLoss/

## Task 515. Conformal variants

- MAPIE `SplitConformalRegressor(estimator=LinearRegression(), confidence_level=0.9,
  conformity_score="absolute", prefit=True, n_jobs=None, verbose=0)` with `confidence_level`
  `Union[float, Iterable[float]]`: CONFIRMED (several levels; `predict_interval` returns
  `(n, 2, n_levels)`). Source: `mapie/regression/regression.py` on master (1.6.0).
- `AbsoluteConformityScore(sym=True)` in `mapie/conformity_scores/bounds/absolute.py`:
  CONFIRMED, `sym=False` gives asymmetric bounds; `conformity_score` accepts a string or a
  `BaseRegressionScore` instance, so use `conformity_score=AbsoluteConformityScore(sym=False)`.
  With `sym=False` the lower and upper quantile levels are `beta` and `1 - alpha + beta`
  (`beta` defaults to `alpha/2` unless `optimize_beta`): matches "two signed trackers at
  alpha/2 and 1 - alpha/2".
- `ConformalizedQuantileRegressor(symmetric_correction=False)`: CORRECTED. The class is
  `ConformalizedQuantileRegressor(quantile_estimator, confidence_level: float = 0.9, ...)`
  (single level only; `CrossConformalizedQuantileRegressor` takes `Iterable[float]`), and
  `symmetric_correction` is a parameter of `predict_interval(X, minimize_interval_width=False,
  allow_infinite_bounds=False, symmetric_correction=False)`, not of the constructor. Its default
  `False` already means separate corrections for the lower and upper quantile regressors.
  https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/mapie/regression/quantile_regression.py
- crepes (PyPI 0.9.1, 2026-06-12): `ConformalRegressor.predict_p_online(y_hat, y, t=None,
  sigmas=None, bins=None, smoothing=True, seed=None, warm_start=True)`: CONFIRMED (also on
  `ConformalPredictiveSystem`; `ConformalClassifier.predict_p_online(alphas, classes, y, bins=None,
  all_classes=True, smoothing=True, seed=None, warm_start=True)`). Exactness caveats: the
  calibration set grows with every test object when `warm_start=True` (expanding, no decay, so
  equal to rheon only at `half_life=inf` with an exact rank, not a sketch); `smoothing=True` makes
  the p-value randomized (needs `seed`), so compare with `smoothing=False`. Also confirmed:
  `predict_int_online(y_hat, y, sigmas=None, bins=None, confidence=0.95, y_min=-inf, y_max=inf,
  warm_start=True)`; `ConformalPredictiveSystem.predict_percentiles_online(...)`.
  The scan's `ConformalClassifier.predict_set_online(class_cond=)`: CORRECTED: no `class_cond`
  argument; the signature is `(alphas, classes, y, bins=None, labels=None, confidence=0.95,
  smoothing=True, seed=None, warm_start=True)` (class-conditional is a Mondrian category via `bins`).
  https://crepes.readthedocs.io/en/latest/crepes.html
- MAPIE master's `exchangeability_testing` (re-check item): CONFIRMED present since v1.4.0 (absent
  at v1.3.0; present at v1.6.0). `mapie.exchangeability_testing` exports `OnlineMartingaleTest`,
  `OnlineExchangeabilityTest`, `FixedDatasetExchangeabilityTest`, `RiskMonitoring`,
  `PermutationTest`, `PValuePermutationTest`, `SequentialMonteCarloTest`.
  `OnlineMartingaleTest(test_method="jumper_martingale"|"plugin_martingale", test_level=0.05,
  jump_size=0.01, burn_in=100)`: CONFIRMED (scan spelling right). `_venn_abers`: CONFIRMED as a private
  module `mapie/_venn_abers.py` (present at v1.3.0 through v1.6.0) with `VennAbers().fit(p_cal, y_cal,
  precision=None)`, `.predict_proba(p_test)`, `VennAbersCV`, `VennAbersMultiClass`; being
  underscore-private, it carries no API stability promise (a dev-only oracle).
  https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/mapie/exchangeability_testing/martingales.py

## Task 520. Specification and influence tests

All statsmodels 0.15.0 (https://www.statsmodels.org/stable/).

- `statsmodels.stats.diagnostic.het_arch(resid, nlags=None, store=False, ddof=0, *,
  result_object=None)`: CONFIRMED. It is `acorr_lm(resid**2, nlags=nlags, ...)`, the LM test of
  `e^2` on its own lags, verified against R `FinTS::ArchTest` per the docs. CORRECTED details:
  the default `nlags=None` becomes `min(10, nobs // 5)` (source of `acorr_lm`), so the test
  must pass `nlags` explicitly; the return is a plain 4-tuple `(lm, lmpval, fval, fpval)` unless
  `result_object=True` (the plain-tuple default is deprecated; changes after 0.16.0 or July
  2027); there is no `autolag`. Exact against rheon's recursive ARCH-LM at the final row of a
  `half_life=inf` run.
- `het_white(resid, exog, interaction_terms=True)`: CONFIRMED, with a CORRECTION to the task's cost
  note: `interaction_terms=False` drops the cross products and keeps only the squares (about `k`
  regressors instead of `k(k+1)/2`), so the k^4 Gram applies only to the default full form.
  `exog` must contain a constant ("assumes x contains constant (for counting dof)"). Returns
  `(lm, lm_pvalue, fvalue, f_pvalue)` with `lm = nobs * R^2` and `chi2.sf(lm, df_model)`. Exact on a
  window.
- `OLSInfluence(results)` with `dfbetas` (a METHOD in the docs: `dfbetas()`, a leave-one-out
  loop), `cooks_distance` (property, uses the original fit with no leave-one-out loop),
  `dffits`, `hat_matrix_diag`, `resid_studentized_external`, `cov_ratio`: CONFIRMED. Caveat:
  `dfbetas` is computed per observation from the final fit on all rows (not at each prefix),
  so the per-row comparison is "OLSInfluence on the prefix, newest row", as the scan says for DFFITS.
  https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html
- Spiegelhalter's calibration test (MAPIE `mapie.metrics.calibration.spiegelhalter_statistic(y_true,
  y_score)` and `spiegelhalter_p_value(y_true, y_score)`): CONFIRMED to exist. Statistic
  `z = sum((y - p)(1 - 2p)) / sqrt(sum((1 - 2p)^2 p (1 - p)))`, matching the three EW sums.
  CORRECTED: `spiegelhalter_p_value = 1 - norm.cdf(z)`, ONE-SIDED (upper tail), not the two-sided
  p-value the test is usually quoted with; a two-sided rheon p-value differs from MAPIE's by
  construction, so compare the statistic and then `2 * min(p, 1 - p)` or the one-sided value.
  https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/mapie/metrics/calibration.py

## Task 521. Forecast comparison

- `dieboldmariano` (PyPI 1.1.0, 2025-01-09; source `src/dieboldmariano/dieboldmariano.py` on master):
  `dm_test(V, P1, P2, *, loss=lambda u, v: (u - v)**2, h=1, one_sided=False,
  harvey_correction=True, variance_estimator="acf")`, returning `(statistic, p_value)`:
  CONFIRMED for `loss`, `one_sided`, `harvey_correction`. Note `variance_estimator` is on master;
  whether 1.1.0 has it was not checked. CORRECTED/important details from the source:
  (1) `loss(actual, predicted)` is a scalar function called per point; `d = L(P1) - L(P2)`.
  (2) The p-value ALWAYS comes from Student's t with `n - 1` df (regularized incomplete beta),
  whether or not `harvey_correction` is set; there is no normal-reference option, so "a t
  reference" is built in, and a normal-reference DM needs another oracle (the HAC t of statsmodels
  with a normal tail).
  (3) The long-run variance uses lags `0..h-1`: `"acf"` is a RECTANGULAR sum
  `(g0 + 2 sum_{i<h} g_i)/n` (autocovariances divided by `n`, not `n - i`), and `"bartlett"` weights
  `2(1 - i/h)`. The scan's "h <-> lags + 1" map holds for the Bartlett kernel:
  `variance_estimator="bartlett"`, `h = lags + 1` equals Bartlett Newey-West with `lags` lags;
  the default `"acf"` is NOT Bartlett.
  (4) HLN factor `sqrt((n + 1 - 2h + h(h-1)/n)/n)`, applied to the statistic.
  (5) `one_sided=True` returns `p/2` if `stat < 0` else `1 - p/2` (small when P1 has the lower
  loss): the alternative is "P1 more accurate".
  (6) raises on a zero variance or a negative variance estimate (suggests `"bartlett"`).
  https://github.com/edoannunziata/dieboldmariano
- statsmodels HAC for Giacomini-White and the HAC t: `OLS.fit(cov_type="HAC", cov_kwds={"maxlags":
  L, "kernel": "bartlett"|"uniform", "use_correction": ...})` CONFIRMED (`maxlags` required);
  `RegressionResults.wald_test(r_matrix, cov_p=None, invcov=None, use_f=None, df_constraints=None,
  scalar=True)` CONFIRMED; with `use_f=None` it uses F when the model's `use_t` is True and
  chi-square otherwise, so pass `use_f=False` for the asymptotic Wald (Giacomini-White's
  chi-square with q df). `f_test` exists as well. Giacomini-White itself: MISSING as a library
  function (the scan says no mainstream Python or `arch` implementation; `arch` offers
  `SPA`, `StepM`, `MCS`, `RealityCheck` in `arch.bootstrap`, bootstrap-based; the docs page
  fetched did not list them, so that part is unverified here).
- MAPIE `RiskMonitoring(risk, test_level=0.05, tolerance=0.05, tolerance_type="absolute",
  threshold=None, reference_data=None, warn=True)`: CONFIRMED to exist (since v1.4.0) but CORRECTED as an
  oracle for "an anytime-valid test of the mean loss DIFFERENCE": it monitors ONE model's risk
  against a threshold built from a reference sample: `threshold = hoeffding_upper_bound(reference
  risks, test_level/2) + tolerance`, then flags a harmful shift when the time-uniform LOWER
  bound `conjugate_mixture_empirical_bernstein_bound(history, v_opt=1, alpha=test_level/2,
  bound_side="lower")` exceeds it; built-in risks are binary (`"precision"`, `"recall"`,
  `"accuracy"`, `"fpr"`, ...) and continuous `"mae"`, `"mse"`; the level is split in two, and the
  bound is recomputed from the full history on each `update`. It is not a two-forecast test. The
  usable oracle is the bound function itself: `mapie.exchangeability_testing.bounds.
  conjugate_mixture_empirical_bernstein_bound(empirical_risk_sequence, v_opt, alpha=0.05,
  bound_side="upper"|"lower", running_intersection=True)` (a Gamma-exponential mixture,
  Howard et al. 2021, via the confseq construction): feed it a loss difference rescaled into
  `[0, 1]`. Also `hoeffding_bound(sequence, delta, bound_side)`.
  https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/mapie/exchangeability_testing/risk_monitoring.py
- `confseq` (`betting_cs`, empirical-Bernstein CS; https://github.com/gostevehoward/confseq):
  not re-fetched here (the scan says not installable).

## Task 514. The DCC `Q` recursion with given `(a, b)` and ADCC's `g`

- "A Python library to compute it where one exists": MISSING for a recursion with GIVEN
  parameters. `arch` (PyPI 9.0.0, 2026-11-11) is univariate only (per third-party READMEs;
  arch's own docs were not located, so this is secondhand). Candidates that exist on PyPI or
  GitHub: `mgarch` 0.3.0 (2022-07-07, "DCC-GARCH(1,1)", `mgarch.mgarch(dist).fit(rt)` and
  `.predict(ndays)`, no filter with given parameters, no ADCC); `pymgarch` (`mg.DCC`, `mg.ADCC`
  with `.fit(returns, marginals=...)` and `res.filter(new_returns)`, which reuses FITTED parameters
  on new data, Cappiello-Engle-Sheppard ADCC; a young project, 0 stars; no documented way to pass
  your own `(a, b, g)`); `multigarch` and `mvarch` (DCC/ADCC, not inspected). None is a mainstream
  oracle. Verdict: no mainstream Python library does it.
  https://pypi.org/pypi/mgarch/json , https://github.com/marwinsteiner/pymgarch
- R `rmgarch` 1.4-3 (CRAN, 2026-09-10): `dccspec(uspec, VAR=FALSE, ..., dccOrder=c(1,1),
  model=c("DCC","aDCC","FDCC"), groups, distribution="mvnorm", start.pars, fixed.pars)` and
  `dccfilter(spec, data, out.sample=0, filter.control=list(), ...)`: CONFIRMED to exist, and
  `fixed.pars` is "required for the filter" methods, so a filter at given parameters is
  possible. The DCC coefficient names in the package's own tests are `dcca1`, `dccb1`, `dccg1`
  (`[Joint]` prefix in a fit's `matcoef`); the exact `fixed.pars` spelling was not confirmed in a
  doc page. It is an R second opinion, not a Python oracle, and its Q starts from the
  full-sample covariance of the standardized residuals (`Qbar`) unless `filter.control$n.old` is
  set, so rheon's online start must be set to the same `Qbar` for a bit comparison.
  https://cran.r-project.org/web/packages/rmgarch/refman/rmgarch.html
- The scan's own recommendation (the recursion written longhand in numpy from Engle 2002 and
  Cappiello-Engle-Sheppard 2006) is the right primary oracle; no library claim to check.

## Task 518. Window operators

Polars (the docs are the stable page; PyPI latest 2.1.0, 2026-10-25; the scan used 2.0.0). All
four methods below are marked UNSTABLE in the docs ("may change without being treated as a
breaking change"), so a Polars upgrade can break the oracle without a deprecation.

- `Expr.rolling_min_by(by, window_size, *, min_samples=1, closed='right')` and
  `rolling_max_by`: CONFIRMED (`min_periods` was renamed `min_samples` in 1.21.0; `by` must be
  Date, Datetime or an integer column, `"3i"` for integer windows; `closed` in `left`, `right`,
  `both`, `none`). Exact.
  https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_min_by.html
- `Expr.rolling_quantile_by(by, window_size, *, quantile, interpolation='nearest', min_samples=1,
  closed='right')`: CONFIRMED. `interpolation` takes `'nearest'` (default), `'higher'`,
  `'lower'`, `'midpoint'`, `'linear'` and also `'equiprobable'` (the scan's list omitted the
  last). `quantile` is keyword-only. Exact for a hard window.
  https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_quantile_by.html
- `Expr.rolling_rank_by(by, window_size, method='average', *, seed=None, min_samples=1,
  closed='right')`: CONFIRMED. `method` in `average` (default; Float64 result), `min`, `max`,
  `dense`, `random` (`seed` used only with `random`; with `seed=None` a seed is generated per
  operation, so `random` is not reproducible unless seeded). Result is the index type unless
  `average`.
- `Expr.ewm_mean(*, com=None, span=None, half_life=None, alpha=None, adjust=True, min_samples=1,
  ignore_nulls=False)`: CONFIRMED. `adjust=True` (default) uses weights `(1-alpha)^i` over
  positions; `adjust=False` is the recursion `y_t = (1-alpha) y_{t-1} + alpha x_t`;
  `half_life` gives `alpha = 1 - exp(-ln 2 / tau)`; `ignore_nulls=False` counts a null as a time
  step. Note: this is a row-position operator, so the "count-weighted" comparison is against
  `ewm_mean` on a row clock (the time-weighted operator is `ewm_mean_by`, whose signature the
  fetched page did not show).
  https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.ewm_mean.html

## Task 522. `hmm`

hmmlearn (PyPI 0.3.3, 2024-10-31; docs `latest`, which carries no version string and may be ahead
of the release).

- `GaussianHMM(n_components=1, covariance_type='diag', min_covar=0.001, startprob_prior=1.0,
  transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=0.01, covars_weight=1,
  algorithm='viterbi', random_state=None, n_iter=10, tol=0.01, verbose=False, params='stmc',
  init_params='stmc', implementation='log')`: CONFIRMED. `covariance_type` in `'spherical'`,
  `'diag'`, `'full'`, `'tied'` (so `spherical` is available as the oracle; the docs' prose says
  spherical is the default while the signature says `'diag'`: trust the signature). `params` selects
  what EM updates and `init_params` what is initialized, both letters of `s`, `t`, `m`, `c`
  (`params="t"` would learn only the transition matrix). `predict_proba(X, lengths=None)` returns the
  SMOOTHED state posteriors, shape `(n_samples, n_components)`: CONFIRMED, so
  `predict_proba(X[:t+1])[t-L]` is the fixed-lag smoothed probability of row `t-L` given rows up to `t`,
  exact at fixed parameters. `decode(X, lengths=None, algorithm=None)`, `score(X, lengths=None)`,
  `get_stationary_distribution()`: CONFIRMED. Caveat: with `params=""` the model must have
  `startprob_`, `transmat_`, `means_`, `covars_` set by hand; the start distribution
  `startprob_` enters the prefix smoother, so rheon's must equal it.
  https://hmmlearn.readthedocs.io/en/latest/api.html

## Task 523. `bocpd`

The scan's reference is `hildensia/bayesian_changepoint_detection`, which has since been rewritten
on PyTorch and published as `bayesian-changepoint` 1.2.2 (PyPI, 2026-11-06; renamed from `bayescd`
in 1.1.0; requires `torch>=2.0`). Names (master, `bayesian_changepoint_detection/`):

- `online_changepoint_detection(data, hazard_function, likelihood_model, device=None)` returning a
  tuple of tensors: CORRECTED (the scan wrote `hazard_func` and `observation_likelihood`).
- Hazards: `constant_hazard(lam, r, device=None)` and `negative_binomial_hazard(k, p, r, device=None)`
  (CONFIRMED: `H(r) = P(len = r+1)/P(len >= r+1)` for `P(len = t) = C(t-1,k-1) p^k (1-p)^(t-k)`,
  `H = p` at `k = 1`; zero for `r + 1 < k`). So "negative-binomial" hazard has an exact oracle.
  A logistic hazard: MISSING (Turner et al. 2009's logistic form is not in the library).
  The offline priors are spelled `const_prior`, `geometric_prior`, `negative_binomial_prior`
  (the scan's `neg_binominal_prior` is CORRECTED).
- Emissions (`online_likelihoods`): `StudentT(alpha=0.1, beta=0.1, kappa=1.0, mu=0.0)`,
  `MultivariateT`, `Poisson`, `NormalKnownVariance(variance=1.0, mu=0.0, prior_variance=1.0)`.
  So "known variance" exists (CONFIRMED); "known mean" (variance-only change) is MISSING:
  write it longhand (a Gamma prior on the precision, a Student-t predictive with a fixed mean,
  via scipy `t`).
- Exactness caveat: run-length state (`alpha`, `beta`, `kappa`, and the centered mean) is
  float32 (`torch.float32`; the data shift is done in float64 and cast back), so agreement with
  rheon's f64 is to roughly 1e-6 relative, not bit-level; `torch` becomes a dev dependency of the
  test group. Also the library carries pruning (`BaseLikelihood.prune(n)`); compare with pruning off.
  https://github.com/hildensia/bayesian_changepoint_detection (raw files under `master`),
  https://pypi.org/pypi/bayesian-changepoint/json

## Task 525. Smaller options

scikit-learn 1.9.1 unless noted.

- `KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0,
  random_state=None, copy_x=True, algorithm='lloyd')`: CONFIRMED. `init` may be an array of shape
  `(n_clusters, n_features)`; with `n_init='auto'` an array `init` runs once (`n_init=1`). It runs
  full Lloyd to convergence (`max_iter=300`), so the oracle for "given centers" row by row is
  `KMeans(init=C, n_init=1, max_iter=1)` or `.predict`/`.transform` at frozen centers; and
  `MiniBatchKMeans.partial_fit` for the incremental update (not re-fetched). `transform(X)`: CONFIRMED,
  distances to every center, `(n_samples, n_clusters)`, Euclidean (not squared). `score(X, y=None,
  sample_weight=None)`: CONFIRMED, returns the NEGATIVE inertia on `X` at the fitted centers
  (`inertia_` is the weighted sum of squared distances), so an exponentially weighted inertia at
  frozen centers = `-score(X, sample_weight=w)`.
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- `GaussianNB(*, priors=None, var_smoothing=1e-09)`: CONFIRMED (`priors` fixes the class priors,
  "not adjusted according to the data"; `class_prior_` attribute). `var_smoothing`: CORRECTED
  detail: the docs only say "portion of the largest variance of all features added to variances";
  in source `epsilon_ = var_smoothing * max(var(X, axis=0))` taken over the `X` of each `fit` or
  `partial_fit` call (so under `partial_fit` it is per call, not on the stream), with
  `var_` carrying `+ epsilon_`. A rheon `var_smoothing` that scales to the running data needs
  that nuance pinned. `partial_fit(X, y, classes=None, sample_weight=None)` uses Chan-Golub-LeVeque
  updates.
  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- `QuadraticDiscriminantAnalysis(*, solver='svd', shrinkage=None, priors=None, reg_param=0.0,
  store_covariance=False, tol=0.0001, covariance_estimator=None)`: CONFIRMED (`priors`,
  `reg_param`). `reg_param` regularizes each class covariance as `(1 - reg_param) * S + reg_param * I`
  (applied to the per-class scalings, i.e. the eigenvalues, which is rotation-invariant, so it
  equals the matrix form). Also: QDA has `solver='eigen'` with `shrinkage` (`None`, `'auto'`
  Ledoit-Wolf, or a float) and `covariance_estimator`, which is also the oracle for task
  511's "shrinkage on `ew_class`" (per-class shrunk covariance).
  `LinearDiscriminantAnalysis(solver, shrinkage, priors, ...)` for the tied case (not re-fetched).
  https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
- statsmodels `Holt(endog, exponential=False, damped_trend=False, initialization_method=None,
  initial_level=None, initial_trend=None)`: CONFIRMED (0.15.0); `initialization_method` in `None`,
  `'estimated'`, `'heuristic'`, `'legacy-heuristic'`, `'known'` (`'known'` requires `initial_level`
  and, with a trend, `initial_trend`). `Holt.fit(smoothing_level=None, smoothing_trend=None, *,
  damping_trend=None, optimized=True, start_params=None, use_brute=True, remove_bias=False,
  method=None, minimize_kwargs=None)`: CONFIRMED; `optimized=False` with fixed smoothing values is
  the exact filter. CORRECTED: `fit` has no `initial_level`/`initial_trend`/`use_boxcox`
  arguments in 0.15.0; the known initial state goes on the constructor. `HoltWintersResults.forecast(steps=1)`:
  CONFIRMED, out-of-sample from the end of the sample; the docs do not state the formula
  (`l + h*b`, damped otherwise) nor the smoothing convention (`smoothing_trend` is the
  beta of the error-correction form, to be pinned by a test), so the formula must be checked
  against the source.
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.Holt.html
- Holm beside Bonferroni: `statsmodels.stats.multitest.multipletests(pvals, alpha=0.05, method='hs',
  maxiter=1, is_sorted=False, returnsorted=False)`: CONFIRMED; `method` strings include `'bonferroni'`,
  `'sidak'`, `'holm-sidak'`, `'holm'`, `'simes-hochberg'`, `'hommel'`, `'fdr_bh'`, `'fdr_by'`,
  `'fdr_tsbh'`, `'fdr_tsbky'`, `'fdr_gbs'`, `'lfdr'`. CORRECTED trap: the DEFAULT is `'hs'`
  (Holm-Sidak), so Holm must be passed as `method='holm'`. Returns `(reject, pvals_corrected,
  alphacSidak, alphacBonf)`.
  https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
- river `drift.KSWIN(alpha=0.005, window_size=100, stat_size=30, seed=None, window=None)`:
  CONFIRMED (docs 0.27.0). It compares the latest `stat_size` values with `stat_size` values drawn AT
  RANDOM from the older part of the window and flags drift when the KS distance exceeds a threshold
  from `alpha` and `stat_size`; `update(x)`, `drift_detected`. Not exact vs a deterministic rheon
  test: the random draw is why it is a reference; compare with `scipy.stats.ks_2samp` on the same
  two samples. https://riverml.xyz/latest/api/drift/KSWIN/
- `deco`'s less biased alternative to `u` (Engle and Kelly 2012): MISSING as a library
  function; the paper's formula is the only oracle (as the scan says).

## Summary table

| Task | Oracle claim | Verdict |
|---|---|---|
| 511 | sklearn OAS | CONFIRMED |
| 511 | sklearn LedoitWolf | CONFIRMED |
| 511 | sklearn ShrunkCovariance | CONFIRMED |
| 511 | river LedoitWolf/OAS/Shrunk (delta, target) | CORRECTED |
| 512 | statsmodels HC0-HC3 via get_robustcov_results / OLS.fit cov_type | CONFIRMED |
| 512 | statsmodels cov_type cluster | CONFIRMED |
| 512 | statsmodels RLM cov="H1" as the sandwich (it is H3) | CORRECTED |
| 512 | statsmodels QuantReg vcov robust hsheather | CONFIRMED |
| 513 | MAPIE regression metrics (coverage, width, ssc, mwi) | CONFIRMED |
| 513 | sklearn mean_pinball_loss | CONFIRMED |
| 513 | sklearn d2_pinball_score | CONFIRMED |
| 513 | sklearn mean_absolute_error / river MAE | CONFIRMED |
| 513 | sklearn log_loss (y_proba, no eps) / river LogLoss | CORRECTED |
| 514 | Python library for DCC filter with given a,b / ADCC g | MISSING |
| 514 | R rmgarch dccfilter fixed.pars | CONFIRMED |
| 515 | MAPIE AbsoluteConformityScore sym=False | CONFIRMED |
| 515 | MAPIE SplitConformalRegressor confidence_level list | CONFIRMED |
| 515 | MAPIE ConformalizedQuantileRegressor symmetric_correction (constructor) | CORRECTED |
| 515 | crepes predict_p_online | CONFIRMED |
| 515 | MAPIE exchangeability_testing and _venn_abers | CONFIRMED |
| 516 | statsmodels RecursiveLS constraints | CONFIRMED |
| 516 | statsmodels OLS.fit_regularized for equalities | CORRECTED |
| 517 | sklearn assume_centered=True | CONFIRMED |
| 517 | scipy multivariate_normal.logpdf | CONFIRMED |
| 517 | sklearn per-row score_samples | MISSING |
| 518 | Polars rolling_min_by / rolling_max_by | CONFIRMED |
| 518 | Polars rolling_quantile_by interpolation | CONFIRMED |
| 518 | Polars rolling_rank_by | CONFIRMED |
| 518 | Polars ewm_mean adjust=True | CONFIRMED |
| 519 | filterpy x, log_likelihood | CONFIRMED |
| 519 | filterpy mahalanobis (sqrt, not squared) | CORRECTED |
| 520 | statsmodels het_arch | CONFIRMED |
| 520 | statsmodels het_white (interaction_terms) | CONFIRMED |
| 520 | statsmodels OLSInfluence dfbetas, cooks_distance | CONFIRMED |
| 520 | MAPIE spiegelhalter_statistic / p_value (one-sided p) | CORRECTED |
| 521 | dieboldmariano dm_test loss, one_sided, harvey_correction | CONFIRMED |
| 521 | dieboldmariano t reference as an option (always t; h<->lags map) | CORRECTED |
| 521 | MAPIE RiskMonitoring as mean loss difference test | CORRECTED |
| 521 | statsmodels HAC Wald for Giacomini-White | CONFIRMED |
| 521 | Giacomini-White as a library function | MISSING |
| 522 | hmmlearn predict_proba on prefix | CONFIRMED |
| 522 | hmmlearn params / init_params | CONFIRMED |
| 522 | hmmlearn spherical covariance | CONFIRMED |
| 523 | bayesian_changepoint_detection online API names | CORRECTED |
| 523 | negative-binomial hazard | CONFIRMED |
| 523 | logistic hazard | MISSING |
| 523 | known-variance emission | CONFIRMED |
| 523 | known-mean emission | MISSING |
| 524 | sklearn BayesianRidge (defaults not evidence max) | CORRECTED |
| 524 | sklearn SGDRegressor average=True | CONFIRMED |
| 524 | sklearn SGD learning_rate optimal (not the regressor default) | CORRECTED |
| 524 | sklearn PassiveAggressiveRegressor average (deprecated 1.8) | CORRECTED |
| 524 | statsmodels RLM TukeyBiweight etc. | CONFIRMED |
| 524 | statsmodels HuberScale as proposal 2 | CORRECTED |
| 524 | sklearn Lasso/ElasticNet positive=True | CONFIRMED |
| 524 | sklearn per-feature penalty weights | MISSING |
| 524 | Vowpal Wabbit --invariant | CONFIRMED |
| 525 | sklearn KMeans init=array, transform, score | CONFIRMED |
| 525 | sklearn GaussianNB priors | CONFIRMED |
| 525 | sklearn GaussianNB var_smoothing formula (per call) | CORRECTED |
| 525 | sklearn QDA priors, reg_param | CONFIRMED |
| 525 | statsmodels Holt known init, forecast | CONFIRMED |
| 525 | statsmodels Holm via multipletests (default is holm-sidak) | CORRECTED |
| 525 | river KSWIN | CONFIRMED |
| 525 | deco less biased u (library) | MISSING |

Totals over the 65 rows above: CONFIRMED 42, CORRECTED 16, MISSING 7. A CONFIRMED row can still carry an exactness caveat in its section (for example 511's OAS differs from the paper, 513's D2 uses a hindsight quantile); the row verdict is about names and meaning.
