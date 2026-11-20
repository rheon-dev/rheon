# Oracle-library gap scan: linear and robust regressions and their inference

*2026-11-20. Read-only scan of rheon 2.0.0 (slot 1, `5122b435`). Nothing was built or run.
Scope: `ewridge`, `rls`, `kalman`, `sgd`, `pa`, `ftrl`, `lasso` (elastic net via `l1_ratio`),
`huber`, `quantile`, the stream's inference fields (`emit_se_coef`, `emit_robust_se`,
`emit_influence`, `emit_breaks`, `conformal`, `emit_error_inflation`), and `rheon.gram`
(`solve`, `lasso_path`, `lars_path`, `coef_stats`, `vif`, `condition`, `merge`).
New model families are out of scope (PLAN tasks 485-510).*

**Sources read in rheon:** `python/rheon/_spec.py` (`ewridge`, `rls`, `lasso`, `kalman`, `ftrl`,
`pa`, `_common`), `_spec_gradient.py` (`sgd`), `_spec_robust.py` (`huber`, `quantile`),
`spec.py` (the common fields table), `gram.py`; checked against `docs/ENHANCEMENTS.md`,
`docs/PLAN.md` (main checkout, which carries tasks 483-510), `docs/records/PLAN-ARCHIVE-*.md`,
`docs/records/PERFORMANCE-LOG.md`, `docs/records/ORACLE-COVERAGE.md`.

**Citation caveat:** the oracle features are cited to the libraries' documentation pages
below from the libraries' published APIs (scikit-learn 1.x, statsmodels 0.14/0.15, river 0.2x,
VW 9.x, filterpy 1.4.5, pykalman 0.9/0.10); the pages were not re-fetched in this pass, so a
parameter's exact spelling should be confirmed on the page before a task cites it.

| key | URL |
|---|---|
| SK-SGD | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html |
| SK-PA | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html |
| SK-RIDGE | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html |
| SK-RIDGECV | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html |
| SK-LASSO | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html |
| SK-EN | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html |
| SK-LARSIC | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html |
| SK-HUBER | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html |
| SK-QR | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html |
| SK-BR | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html |
| SK-ARD | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html |
| SK-LR | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html |
| SM-ROBCOV | https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html |
| SM-PRED | https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_prediction.html |
| SM-RLS | https://www.statsmodels.org/stable/generated/statsmodels.regression.recursive_ls.RecursiveLS.html |
| SM-QR | https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.fit.html |
| SM-RLM | https://www.statsmodels.org/stable/generated/statsmodels.robust.robust_linear_model.RLM.fit.html and https://www.statsmodels.org/stable/rlm.html (norms) |
| SM-HSCALE | https://www.statsmodels.org/stable/generated/statsmodels.robust.scale.HuberScale.html |
| SM-INFL | https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html |
| SM-CONSTR | https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html and `GLM.fit_constrained` (https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.fit_constrained.html) |
| RV-LR | https://riverml.xyz/latest/api/linear-model/LinearRegression/ |
| RV-BLR | https://riverml.xyz/latest/api/linear-model/BayesianLinearRegression/ |
| RV-SCHED | https://riverml.xyz/latest/api/optim/schedulers/Optimal/ (and `InverseScaling`, `Constant`) |
| VW-ARGS | https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Command-Line-Arguments |
| VW-LOSS | https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Loss-functions |
| FP-KF | https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html |
| PK | https://pykalman.github.io/ (`KalmanFilter.em`, `loglikelihood`, `filter_update`) |

Status words: **new** (not in ENHANCEMENTS, PLAN or ORACLE-COVERAGE), **planned (task N)**,
**rejected (ref)**, **idea (ref)** for PLAN task 118's "ideas waiting on a need" (not
planned, not rejected). Feasibility: **one pass**, **decay/window** (one pass, and meaningful
under a half-life or window), **needs passes** (no faithful streaming form).

---

## 1. Current surface (step 1), condensed

| model | own parameters | outputs beyond `pred`/`resid`/`coef`/`weight_sum`/`settled_frac`/`withheld_reason` |
|---|---|---|
| `ewridge` | `ridge` (scalar or grid), `feature_sets`, `standardize`, `ridge_scale` mean/sum, `coef_prior`, `session_shrink`, `long_half_life`, `solve_every`, `max_rows_between_solves`, `gram_block_rows`, `gram_threads`, `target_gaps`, `window_size`, `closed`, snapshots, `window_budget` | `support_coef`; every common diagnostic; `se_coef`, `se_coef_hc0`, `se_coef_hac`, `error_inflation`, `influence` |
| `rls` | `delta`, `coef_prior` | same as ewridge incl. robust SE |
| `kalman` | `coef_half_life` (per coef) or `q` (diagonal), `obs_var`, `p0` (scalar), `share_p`, `revert_half_life`, `standardize` | `se_coef` (posterior), `error_inflation`, `influence` |
| `lasso` | `penalties` (path), `l1_ratio`, `select_half_life`, cadence, window, `max_iter`, `tol`, `target_gaps` | `penalty_selected_<t>`; no `se_coef` (refused, post-selection) |
| `huber` | `huber_delta`, `huber_scale` mad/rms, `huber_scale_half_life`, `ridge`, `standardize`, cadence, `feature_sets`, `ridge_scale`, `emit_huber_scale`, `huber_refit_rows`, `refit_budget` | `huber_scale`; no `se_coef` (refused: "an M-estimator's covariance is a sandwich") |
| `quantile` | `quantile` (one tau), `ridge`, `standardize`, cadence, `quantile_eps`, `quantile_refit_rows`, `refit_budget` | no `se_coef` |
| `sgd` | `loss` squared/huber/quantile/epsilon_insensitive/poisson/logistic, Huber scale options, `eps`, `learning_rate`, `schedule` constant/inv_scaling/adagrad, `power`, `l2`, `clip_gradient`, `standardize`, `coef_min/max/sum`, `strict_binary` | `withheld_reason="unstable_rate"` |
| `pa` | `mode` pa/pa1/pa2, `c`, `eps`, `coef_min/max/sum`, `standardize` | — |
| `ftrl` | `alpha`, `beta`, `l1`, `l2`, `loss` logistic/squared, `strict_binary` | — |
| common | `fit_intercept`, `weight`, decay/clock rules, `min_weight`, `min_settled_frac`, `max_error_inflation`, `emit_sigma/zscore/metrics/selected/averaged`, `conformal`, `resid_quantiles`, `emit_autocorr/drift/calibration/breaks/specification/tails/influence/feature_health`, `emit_robust_se` (HC0 + Bartlett HAC, `robust_se_lags`), `embargo`, `label_clock` | |
| `rheon.gram` | `solve(ridge, standardize)`, `lasso_path(penalties, l1_ratio, penalty_weights, max_iter, tol)`, `lars_path`, `solve_subsets`, `coef_stats` (classical se, t, R²), `vif`, `condition`, `merge(within=)` | |

---

## 2. Gap tables per model

### 2.1 `ewridge` (and `rheon.gram.solve`)

| oracle feature | rheon equivalent | online feasibility | value | oracle test it would enable | status |
|---|---|---|---|---|---|
| sklearn.BayesianRidge(alpha_1, alpha_2, lambda_1, lambda_2, compute_score=True) — ridge and noise precision by evidence maximization [SK-BR] | none; `ridge` is fixed, chosen only by `emit_selected` over a grid | one pass, decay/window: MacKay's fixed-point iterations read only the Gram, `X'y`, `y'y` and n (Kish n under decay), so they run at solve time on the running sums | **high** — removes the one tuning knob of the default model, and gives a per-row predictive std with no grid | `ewridge(ridge="evidence", half_life=inf)` coef, alpha, lambda, `scores_` == `BayesianRidge(fit_intercept=True).fit` on the rows (1e-8); with decay, == BayesianRidge with `sample_weight` = decayed weights scaled to Kish n | new |
| sklearn.ARDRegression(threshold_lambda, ...) — one prior precision per coefficient [SK-ARD] | none | one pass at solve time (the updates need `(diag(lambda) + alpha X'X)^-1` and `X'y` only) | medium — sparse-ish Bayesian alternative to `lasso` with no penalty path; pruning threshold changes the support | `ARDRegression().fit` coef and `lambda_` at `half_life=inf` | new |
| sklearn.BayesianRidge.predict(return_std=True); statsmodels get_prediction().summary_frame(obs=True) `obs_ci_lower/upper` [SK-BR, SM-PRED] | `sigma` and `error_inflation` exist separately; `conformal` is `pred ± m*sigma` | one pass: `pred_std = sigma * error_inflation`, interval with Student t at Kish n − k | medium — a parametric interval that widens on a high-leverage row, which the conformal interval does not (it scales `sigma` only) | run once, `lo/hi` == `get_prediction(exog_row).conf_int(obs=True, alpha)` from the fit before the row (1e-10) | new |
| sklearn.RidgeCV(alphas, gcv_mode, alpha_per_target=True) — efficient LOO/GCV ridge choice [SK-RIDGECV] | `emit_selected` picks a grid point by EW out-of-sample error (prequential, not LOO) | one pass at solve time from an eigendecomposition of the Gram (LOO residuals from `X'X`, `X'y`, `y'y` need the per-row hat diagonal, so exact LOO needs passes; GCV needs only the trace and is one pass) | low-medium — prequential selection already covers the use; GCV adds a batch-comparable criterion | `rheon.gram.solve(ridge="gcv")` == `RidgeCV(alphas, gcv_mode="eigen")` with `alpha_per_target` at no decay | new (GCV one pass; LOO needs passes) |
| sklearn.Ridge(alpha=array per target) [SK-RIDGE] | a `ridge` list is a grid shared by all targets, not one value per target | one pass | low — a grid with `emit_selected` covers it, at k² extra solves | `Ridge(alpha=[a1, a2])` on two targets | new |
| sklearn.Ridge(positive=True), LinearRegression(positive=True) [SK-RIDGE, SK-LR] | none on the exact solvers; box/simplex only on `sgd`/`pa` | one pass at solve time (active-set NNLS on the Gram per solve; O(k³) worst case) | medium — portfolio and mixing weights with an exact fit | `LinearRegression(positive=True)` / `scipy.optimize.nnls` on the rows | rejected for exact solvers (ENHANCEMENTS E40: "non-negative least squares is an active-set solve per row ... a different model"; "no use case named") |
| statsmodels.RecursiveLS(constraints="x1 + x2 = 1") and OLS fit with linear equality restrictions [SM-RLS, SM-CONSTR] | `coef_sum` on `sgd`/`pa` only (projection) | one pass: equality-constrained least squares is a closed form on the Gram (KKT / null-space), exact at every solve, cost O(k²·m) | **high** — exact sum-to-one / linear restrictions on the default exact fit, with a closed form and a direct oracle | `ewridge(..., constraints=...)`, `half_life=inf`, `ridge=0`: coef == `RecursiveLS(y, X, constraints=...)`.fit() filtered params at every row (1e-9) | new (E40 rejected bounds, not equalities) |
| statsmodels.RegressionResults: `tvalues`, `pvalues`, `conf_int` [SM-ROBCOV] | `se_coef` per row; t and p only through `rheon.gram.coef_stats` after the fact | one pass | low — arithmetic on fields already emitted | `t_coef` == OLS `tvalues` run once | new |

### 2.2 `rls`

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| statsmodels.RecursiveLS(constraints=...) [SM-RLS] | none | one pass (constrained update of `P` and the state) | high (as for ewridge; RecursiveLS is the direct oracle for this model) | `rls(constraints=...)` == RecursiveLS filtered params and `cov_params` per row | new |
| statsmodels.RecursiveLS results `llf`, `loglikelihood_burn` [SM-RLS] | none | one pass, decay (EW prediction-error log-likelihood) | medium — model comparison and a likelihood for choosing `half_life`/`delta` | `loglik` run once == RecursiveLS `llf` after the burn-in | new |
| padasip FilterRLS(mu, eps, w) | `half_life`, `delta`, `coef_prior` | — | covered | (ORACLE-COVERAGE §2.1, COVERED) | — |
| square-root-free Givens for `rls` | — | — | numerics only | — | idea (PLAN task 118) |

### 2.3 `kalman`

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| filterpy.KalmanFilter.x — initial state mean [FP-KF]; pykalman `initial_state_mean` [PK] | none: the prior mean is 0 (no `coef_prior`, which `ewridge` and `rls` have) | one pass | **high** — the warm start every other exact model offers; restarting from yesterday's fit without a saved state | `kalman(coef_prior=c0)` == filterpy with `kf.x = c0` (1e-12), standardize off | new |
| filterpy.KalmanFilter.P (full matrix), pykalman `initial_state_covariance` | `p0` scalar (same for every slope) | one pass | low-medium — per-coefficient prior confidence | filterpy with `kf.P = diag(p0_i)` / full | new |
| filterpy `Q` full matrix; pykalman `transition_covariance` | `q` diagonal, or `coef_half_life` per coefficient | one pass | low-medium — correlated drift (e.g. two betas moving together) | filterpy with a full `Q` | new |
| filterpy.KalmanFilter.log_likelihood / likelihood / mahalanobis [FP-KF]; pykalman `loglikelihood` [PK] | none (`zscore` is resid/sigma, not the innovation's own variance) | one pass, decay | **medium-high** — the innovation likelihood is how `coef_half_life`/`q` are chosen; a per-row innovation z-score `e / sqrt(S)` is the filter's own outlier test | `loglik_<t>` and `innovation_z` == filterpy `kf.log_likelihood`, `kf.mahalanobis` per row (1e-12) | new |
| filterpy.KalmanFilter.alpha — fading memory (`P = alpha² F P F' + Q`) [FP-KF] | none; forgetting is via `coef_half_life`/`q` on the clock | one pass, decay | low — the clock-aware `coef_half_life` does the job; alpha is per-step, not per-clock-unit | filterpy `alpha` at unit clock | new (low) |
| pykalman KalmanFilter.em(em_vars=["transition_covariance","observation_covariance"]) [PK] | `obs_var` estimated as EW residual variance; `Q` fixed | needs passes (EM over the whole series); an online EM / ML gradient of Q is possible but approximate | medium — automatic `q` | pykalman `em` on the rows, then compare the filtered states | new, **needs passes** (an online ML variant is one pass but not exact) |
| filterpy rts_smoother / FixedLagSmoother; pykalman `smooth` | none | fixed-lag: one pass with an embargo | — | — | planned (task 496) |
| filterpy predictive variance with `Q·d` | `pred_var` documented as filtered; `Q·d` not added | — | — | — | rejected (REVIEW-2026-09-18 D2, signed off 2026-09-29, task 115(e)) |

### 2.4 `sgd`

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| sklearn.SGDRegressor(average=True / average=n) — Polyak/ASGD [SK-SGD]; sklearn PA `average` [SK-PA] | none | one pass; under decay an EW average of the iterates (the form PERFORMANCE-LOG §19 names) | **medium-high** — measured best sklearn setting on a stationary stream (0.9830); removes most of the rate-tuning noise | `sgd(average=True, half_life=inf)` `pred` == `SGDRegressor(average=True, learning_rate="constant").partial_fit` row by row (1e-12) | idea (PLAN task 118, "`sgd`'s iterate average"; PERFORMANCE-LOG §19: "not built") |
| sklearn.SGDRegressor(penalty="l1"/"elasticnet", l1_ratio) [SK-SGD]; river LinearRegression(l1=) [RV-LR]; VW `--l1` (truncated gradient) [VW-ARGS] | `sgd` has `l2` only; L1 lives in `ftrl` and `lasso` | one pass (truncated/cumulative-penalty gradient, as sklearn does) | medium — an O(k) sparse fit under the robust and Poisson losses that `ftrl` lacks | `SGDRegressor(penalty="elasticnet")` partial_fit row by row; river `LinearRegression(l1=...)` | new (PLAN-ARCHIVE-1 task-comparison note: "`l1`/elastic net lives in `lasso`" — a stance, not a rejection) |
| sklearn.SGDRegressor(loss="squared_epsilon_insensitive") [SK-SGD] | none (`epsilon_insensitive` only) | one pass | low | `SGDRegressor(loss="squared_epsilon_insensitive")` row by row | new (noted as lacking in PLAN-ARCHIVE-1; never planned) |
| sklearn learning_rate="optimal" (`1/(alpha (t + t0))`); river `optim.schedulers.Optimal` [SK-SGD, RV-SCHED]; VW `--initial_t`, `--power_t`, `--decay_learning_rate` [VW-ARGS] | `constant`, `inv_scaling` (no `t0` offset), `adagrad` | one pass, decay (count = `weight_sum`) | low-medium — `optimal` is sklearn's default schedule, so the default comparison is not reproducible today | `sgd(schedule="optimal", l2=a)` == `SGDRegressor(learning_rate="optimal", alpha=a)` row by row | new |
| sklearn learning_rate="adaptive", early_stopping, n_iter_no_change, validation_fraction, max_iter, tol, shuffle [SK-SGD] | none | **needs passes** (epochs, held-out fraction) | low | — | new, needs passes (out of one-pass scope) |
| river LinearRegression(intercept_lr=, intercept_init=) [RV-LR] | intercept takes the running mean's step during warm-up, the gradient after | one pass | low | river with `intercept_lr` a separate scheduler | new (low) |
| VW `--invariant` importance-aware updates (Karampatziakis & Langford) [VW-ARGS] | weights multiply the step; `pa` caps the weight at 1 | one pass (closed form for squared, logistic, hinge, quantile) | medium — large row weights overshoot under a constant rate; the closed form makes a weight-w row equal w unit rows in the limit | `vw --sgd --invariant` vs `vw --sgd` predictions with weighted examples | new |
| VW `--normalized` (scale-invariant NAG) [VW-ARGS] | `standardize` (EW moments) | one pass | low — standardize covers the need | — | new (low) |
| optim.Adam/RMSProp/Momentum/Nadam/... | adagrad only | — | — | — | rejected (ENHANCEMENTS §4, "Momentum-family optimizers") |
| VW `--loss_function expectile` [VW-LOSS] | none | — | — | — | planned (task 501, expectile regression) |
| statsmodels GLM families beyond Poisson (Gamma, Tweedie) | `loss="poisson"`, `"logistic"` | — | — | — | planned (task 500, online GLMs) |

### 2.5 `pa`

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| sklearn.PassiveAggressiveRegressor(average=True) [SK-PA] | none | one pass | medium (same argument as sgd) | `PassiveAggressiveRegressor(average=True)` partial_fit row by row, `eps=0`, no intercept | idea (task 118 names `sgd`'s only) |
| sklearn PA(loss="squared_epsilon_insensitive") | `mode="pa2"` (Crammer's, = river) | — | — | — | rejected as an oracle mismatch: sklearn's PA-II squares the loss in the numerator (REVIEW-2026-09-12-pass10 V24); rheon follows Crammer/river |
| sklearn PA sample weight uncapped | rheon caps the step's weight at 1 | — | — | — | documented difference (REVIEW-2026-09-12-pass10 D10) |
| early_stopping, max_iter, tol | none | needs passes | low | — | new, needs passes |

### 2.6 `ftrl`

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| river FTRLProximal(alpha, beta, l1, l2); VW `--ftrl --ftrl_alpha --ftrl_beta --l1 --l2` | all four | — | covered | (COVERED, river and VW) | — |
| VW `--loss_function quantile/hinge/poisson` with `--ftrl` [VW-LOSS] | `loss` logistic/squared | one pass (the gradient is the only change) | low-medium — sparse quantile/Poisson fits | `vw --ftrl --loss_function quantile --quantile_tau t` per row | new |
| VW `--coin` / `--pistol` / `--freegrad` (parameter-free) [VW-ARGS] | none | one pass, but per-coordinate wealth state with no clean decay semantics | low | `vw --coin` per row | new; likely falls under ENHANCEMENTS §4's reason for momentum optimizers (state with no clock-decay meaning) |

### 2.7 `lasso` / elastic net (and `rheon.gram.lasso_path`, `lars_path`)

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| sklearn.Lasso / ElasticNet(positive=True) [SK-LASSO, SK-EN] | none | one pass at solve time: coordinate descent clips each coordinate at 0, no active-set solve, same cost | **medium-high** — sign-constrained sparse fit at no extra cost; unlike NNLS it is not "a different model" | `lasso(positive=True)` == `ElasticNet(positive=True, alpha, l1_ratio).fit` with sample weights (KKT 1e-10) | E40 scoped bounds out of the exact solvers; the lasso case (a clip in CD) is not argued there — report as new, flag E40 |
| sklearn per-feature penalty factors (glmnet `penalty.factor`; adaptive lasso via rescaled columns) | `rheon.gram.lasso_path(penalty_weights=)` only; the streaming `lasso` spec has none | one pass | medium — adaptive lasso (weights from a ridge fit) and unpenalized controls in the live model | `lasso(penalty_weights=w)` == sklearn `enet_path` on columns divided by w (the test `gram` already runs) | new for the spec (gram has it, COVERED) |
| sklearn.LassoLarsIC(criterion="aic"/"bic", noise_variance) [SK-LARSIC] | selection by EW out-of-sample error (`select_half_life`) | one pass at solve time (df = active-set size, RSS from the Gram, n = Kish n) | low-medium — an in-sample criterion that needs no warm-up of errors | `lasso(select="bic")` penalty == `LassoLarsIC(criterion="bic")` `alpha_` at no decay | new |
| statsmodels OLS.fit_regularized(method="sqrt_lasso") | none | one pass at solve time | low | `fit_regularized(method="sqrt_lasso")` | new (low) |
| post-selection inference | refused by design | — | — | — | rejected (spec.py: `se_coef` refused for `lasso`, post-selection) |
| `gram_block_rows` on `lasso` | — | — | — | — | idea (task 118) |

### 2.8 `huber` (and RLM)

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| statsmodels.RLM(...).fit(cov="H1"/"H2"/"H3") `bse` — the M-estimator sandwich [SM-RLM] | none: `se_coef` refused for `huber` | one pass, decay: EW sums of `psi'(r_i) z z'` and `psi(r_i)² z z'` at the row's residual, read like `emit_robust_se`'s sandwich | **high** — the robust model has no inference today; the machinery (`sandwich.rs`) exists for LS | run once, `se_coef` ≈ `RLM(y, X, M=HuberT(t)).fit(cov="H1").bse` statistically (ratio band over seeds, as HC0 is held today) | new |
| statsmodels RLM norms: TukeyBiweight, Hampel, AndrewWave, RamsayE, TrimmedMean [SM-RLM] | Huber only (`huber_delta`) | one pass by the same reweighting on running sums, but redescending norms are non-convex: the warm-up refit (`huber_refit_rows`) is what keeps them from capture | **medium-high** — heavy-tailed data where a Huber fit still leans on gross outliers; Tukey bisquare is the common choice | `huber(norm="tukey", c=4.685)` with a refit buffer == `RLM(M=TukeyBiweight(c)).fit` on the buffer at the refit row (as `huber_refit_rows` is held today), then statistical | new |
| statsmodels HuberScale / RLM(scale_est=HuberScale()) — Huber's proposal 2 joint scale [SM-HSCALE]; sklearn HuberRegressor's concomitant scale `scale_` [SK-HUBER] | `huber_scale` = `"mad"` or `"rms"` | one pass, decay (an EW fixed-point of the proposal-2 equation, one step a row) | medium — the scale sklearn's and RLM's default fits estimate; a closer oracle match at a finite delta | `huber(huber_scale="proposal2")` scale == `HuberScale()(resid)` run once; coef vs `HuberRegressor(epsilon, alpha=0)` (whose scale is joint) | new |
| sklearn HuberRegressor.outliers_ (|r| > epsilon·scale mask) [SK-HUBER] | `zscore` and `emit_huber_scale` give the pieces | one pass | low | `outliers_` == `abs(resid) > delta * huber_scale` | new (low) |
| `huber(feature_sets=)`, `ridge_scale="sum"`, cadence | — | — | — | — | PARTIAL in ORACLE-COVERAGE §3 rank 15 (no oracle beyond reductions) |

### 2.9 `quantile` (and QuantReg / QuantileRegressor)

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| statsmodels.QuantReg.fit(vcov="robust"/"iid", kernel="epa"/"gau"/..., bandwidth="hsheather"/"bofinger"/"chamberlain") `bse` [SM-QR] | none: `se_coef` refused | one pass, decay: the sparsity `1/f(0)` from an EW kernel density of the residuals near 0 at the chosen bandwidth (Kish n in the bandwidth rule), and EW sums of `z z'` and of the kernel-weighted `z z'` | **high** — quantile fits are used for tails and medians and have no inference | run once, `se_coef` vs `QuantReg(y, X).fit(q, vcov="robust", kernel="epa", bandwidth="hsheather").bse` statistically | new |
| sklearn.QuantileRegressor(alpha=) — an L1 penalty [SK-QR] | `ridge` (L2) only | one pass at solve time (CD on the smoothed check loss's weighted Gram) | low-medium — sparse quantile fit | `QuantileRegressor(quantile, alpha, solver="highs")` at a small `quantile_eps` (statistical) | new |
| several quantiles in one spec (QuantReg fitted per q; no-crossing) | one `quantile` per spec | one pass (one state per tau; a list as a grid like `ridge`) | medium — the interval use (0.05/0.5/0.95) needs three specs today; a grid could also enforce non-crossing by sorting | per tau as today's `TestQuantileIsQuantReg` | new |
| expectile regression | — | — | — | — | planned (task 501) |

### 2.10 Robust / sandwich standard errors (`emit_robust_se`, `coef_stats`)

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| statsmodels get_robustcov_results(cov_type="HC1") [SM-ROBCOV] | HC0 only | one pass: HC0 × n/(n−k) at Kish n | **high** (cheap) — HC1 is the default robust SE in most packages (Stata `robust`) | run once `se_coef_hc1` vs `OLS.fit(cov_type="HC1").bse`, same ratio band as HC0 today | new |
| cov_type="HC2"/"HC3" [SM-ROBCOV] | HC0 on out-of-sample residuals | one pass: the in-sample residual is `e_oos / (1 + h_pre)` (`h_pre = error_inflation² − 1`), so HC0/HC2/HC3 weights `e²(1−h)^-a` are all reachable from fields already kept; note the OOS-residual HC0 rheon emits is already close to a jackknife (HC3-like) form | medium — lets the user pick the textbook form and makes the "reads above statsmodels' HC0" note precise | run once, each vs `OLS.fit(cov_type="HC2"/"HC3")` (statistical; residuals are at the row's fit, not the final fit) | new |
| cov_type="HAC" with kernel="uniform" (and use_correction) [SM-ROBCOV] | Bartlett only | one pass (same lag slots, other weights) | low | `OLS.fit(cov_type="HAC", cov_kwds={"maxlags":L,"kernel":"uniform"})` | new (low) |
| cov_type="cluster" (groups=), "hac-groupsum" (Driscoll-Kraay), "nw-panel" [SM-ROBCOV] | none; `group` runs independent models | one pass where clusters arrive contiguous (a session/date column): accumulate the cluster's score `sum z e`, add its outer product when the cluster closes (`group_close`-style). Clustering across groups of a pooled fit, or on `rheon.gram.merge` output, **needs passes** (scores need the pooled fit's residuals) | medium-high for panels with time clusters (cross-sectional regressions per date); low elsewhere | run once, `se_coef_cluster` vs `OLS.fit(cov_type="cluster", cov_kwds={"groups": date})` | new (contiguous one pass; pooled needs passes) |
| `use_t`, `df_correction` [SM-ROBCOV] | — | one pass | low | — | new (low) |
| robust SE for `kalman`, `huber`, `quantile` | `ewridge` and `rls` only | see 2.8/2.9 for the M-estimators; for `kalman` the posterior is the model's own | — | — | huber/quantile new (above) |

### 2.11 Influence and the recursive diagnostics (`emit_influence`, `emit_breaks`)

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| OLSInfluence.cooks_distance [SM-INFL] | DFFITS only | one pass: `D = e_stud² h / (k (1−h))` from the same leverage and residual | medium — the influence measure most users know; free given DFFITS | run once on prefixes, == `OLSInfluence(res).cooks_distance[0][-1]` (as DFFITS is held) | new |
| OLSInfluence.dfbetas [SM-INFL] | none | one pass, O(k²) a row: `(X'X)^-1 z e / (1−h)` scaled by `s·sqrt(diag)` — the gain vector `rls`/`kalman` already form | medium — which coefficient a row moved | == `OLSInfluence.dfbetas[-1]` on prefixes | new |
| OLSInfluence.cov_ratio, resid_studentized_internal | `studentized` (external, recursive) in `emit_breaks` | one pass | low | — | new (low) |
| RecursiveLS cusum / cusum_squares / recursive residuals | `emit_breaks` | — | covered | (COVERED) | — |

### 2.12 `rheon.gram` helpers

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| statsmodels robust covariances on a Gram fit | `coef_stats` classical only | HC*, HAC and cluster need per-row residuals against the Gram's fit: **needs passes** (or the stream's `emit_robust_se`) | low | — | new, needs passes |
| BayesianRidge / GCV / BIC on a Gram (see 2.1, 2.7) | none | one pass (Gram-only) | medium — the natural first home: no state change, Python only | as in 2.1/2.7 | new |
| `lasso_path(positive=True)`, `solve(constraints=)` | none | Gram-only | medium | sklearn `enet_path(positive=True)`; constrained LS closed form vs statsmodels | new |

### 2.13 Warm-up

| oracle feature | rheon equivalent | online feasibility | value | oracle test | status |
|---|---|---|---|---|---|
| sklearn `warm_start` / `coef_init`, `intercept_init` (SGD, PA, Huber, Lasso) | `coef_prior` on `ewridge`/`rls`; `load_state` for all | one pass | medium for `kalman` (2.3) and `sgd`/`pa`/`huber`/`quantile` (a starting fit without a saved state); low elsewhere | `sgd(coef_init=c0)` == `SGDRegressor().partial_fit(..., coef_init=c0)` | new |
| river `optim.initializers` | zeros | — | low | — | new (low) |

---

## 3. Ranked top ten

| rank | item | model(s) | why first | status |
|---|---|---|---|---|
| 1 | M-estimator sandwich SEs (RLM `cov="H1"`) | `huber` (and `sgd` huber loss) | the robust model emits no inference at all; the sandwich code exists for LS; one pass with EW sums | new |
| 2 | Quantile SEs (QuantReg `vcov="robust"`, Hall-Sheather bandwidth) | `quantile` | same gap for quantile fits; one EW kernel density at 0 plus two Gram-like sums | new |
| 3 | Linear equality constraints (RecursiveLS `constraints=`) | `ewridge`, `rls`, `gram.solve` | closed form on the Gram, exact every row; a direct statsmodels oracle row for row; covers the sum-to-one ask on the exact models without NNLS | new (E40 rejected bounds only) |
| 4 | HC1 (and HC2/HC3) in `emit_robust_se` | `ewridge`, `rls` | HC1 is a scalar on HC0; HC2/HC3 come from `error_inflation` already kept; matches the default robust SE users compare against | new |
| 5 | Evidence-maximized ridge (BayesianRidge), with `pred_std` | `ewridge`, `gram.solve` | removes the default model's one tuning knob; one pass at solve time; exact oracle at `half_life=inf` | new |
| 6 | Kalman `coef_prior` (filterpy `x`) and innovation log-likelihood / Mahalanobis | `kalman` | the one exact model without a warm start; the likelihood is how its noise is chosen; filterpy holds both per row | new |
| 7 | Averaged SGD / PA (`average=True`), EW-weighted | `sgd`, `pa` | sklearn's best stationary setting; measured gap recorded in PERFORMANCE-LOG §19 | idea (task 118) |
| 8 | `positive=True` and per-feature `penalty_weights` on the streaming `lasso` | `lasso` | a clip in coordinate descent, and weights `gram.lasso_path` already takes; both held by `enet_path` | new (positive: flag E40) |
| 9 | Redescending norms (Tukey bisquare) with the existing refit buffer | `huber` | heavy tails beyond Huber; RLM is the oracle at the refit row | new |
| 10 | Cluster-robust SEs for contiguous clusters (date/session) | `ewridge`, `rls` | panel and cross-sectional finance use; one pass when clusters close in order | new (pooled across groups needs passes) |

Close behind: Cook's distance and DFBETAS beside DFFITS; a quantile grid in one spec;
SGD elastic-net penalty; sklearn's `optimal` schedule; VW `--invariant` importance-aware steps;
Huber proposal-2 scale.

**Already planned, so not proposed:** expectile loss (task 501), Gamma/Tweedie GLM families
(task 500), the fixed-lag Kalman smoother (task 496). **Rejected:** momentum-family optimizers
(ENHANCEMENTS §4), NNLS bounds on the exact solvers (E40), Kalman predictive variance with `Q·d`
(REVIEW-2026-09-18 D2), post-selection SEs on `lasso`, sklearn's PA-II form (V24).
**Needs passes, out of scope:** SGD/PA early stopping and `learning_rate="adaptive"`, pykalman
EM for `Q`/`R`, exact LOO for ridge, robust covariances on a Gram fit after the fact, cluster SEs
on a pooled fit across groups.
