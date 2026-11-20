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
- scikit-learn `Lasso/ElasticNet(positive=True)` (from the scan; the task text names no
  oracle): not re-fetched beyond the scan's claim; per-feature penalty weights: scikit-learn
  has no `penalty_factor` option (glmnet's name); the scan's workaround (divide columns by w)
  is the standard equivalence. Not verified against docs here (MISSING as a library option).
- Vowpal Wabbit "importance-invariant step": CONFIRMED. `--invariant` is a real option
  (`make_option("invariant", invariant)` in `vowpalwabbit/core/src/reductions/gd.cc` on master; the
  wiki lists it as "use safe/importance aware updates", on by default and switched off by
  `--sgd`). Theory: Karampatziakis and Langford, "Online Importance Weight Aware Updates"
  (arXiv 1011.1576). The wiki page is flagged out of date by its own banner. Per
  scan/wiki: `--initial_t` (0), `--power_t` (0.5), `--decay_learning_rate` (1), `--l1`, `--l2`,
  `--ftrl --ftrl_alpha (0.005) --ftrl_beta (0.1)`.
  https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-Line-Arguments
