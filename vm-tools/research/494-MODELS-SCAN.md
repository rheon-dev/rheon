# Task 494: the oracle-library scan for the models of tasks 495 to 510

*2026-11-20. Read-only research, written against `integration-w4` at `2ab5add5`
(slot 1). Nothing was built and nothing in the repository was changed. One table
per model family; each task (495 to 510) takes its feature list, its
decay and window decision, and its oracle from its section here.*

## 0. How to read this

**Sources and how each was checked.**

| tag | library, version | how it was read |
|---|---|---|
| [sm] | statsmodels 0.15.0 | `inspect.signature` and the source in the repo's `.venv` (`.venv/lib/python3.12/site-packages/statsmodels/…`); `file:line` given |
| [sk] | scikit-learn 1.9.1 | the same, `sklearn/…` |
| [rv] | river 0.26.1 | the same, `river/…`; module contents from each module's `__all__` |
| [sp] | scipy 1.18.1 | the same, `scipy/…` |
| [fp] | filterpy 1.4.5 | the same, `filterpy/…` |
| [hf] | hfhd 0.1.4 | the same, `hfhd/hf.py` |
| [vw] | vowpalwabbit 9.11.9 (installed) | options from the project wiki, <https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms> |
| [tick] | tick 0.8.0.2 (not installed) | <https://x-datainitiative.github.io/tick/modules/hawkes.html> and the class pages under it; wheels from <https://pypi.org/pypi/tick/json> |
| [hawkes] | `hawkes` (Omi; not installed) | <https://github.com/omitakahiro/Hawkes>, `Hawkes/model.py` |
| [pk] | pykalman 0.11.2 (not installed) | <https://github.com/pykalman/pykalman> README |
| [arch] | arch (not installed) | <https://arch.readthedocs.io/en/latest/api.html> |
| [glum] | glum (not installed) | <https://glum.readthedocs.io/en/latest/glm.html> |
| [ds] | datasketch (not installed) | <https://ekzhu.com/datasketch/documentation.html> |
| [ads] | Apache `datasketches` (not installed) | <https://apache.github.io/datasketches-python/main/index.html> and the per-sketch pages |
| [pyod] | PyOD 3.6.8 (not installed) | <https://pyod.readthedocs.io/en/latest/> |
| [rrcf] | rrcf (not installed) | <https://klabum.github.io/rrcf/> |
| [rup] | ruptures (not installed) | <https://centre-borelli.github.io/ruptures-docs/user-guide/> |
| [dcor] | dcor (not installed) | <https://dcor.readthedocs.io/en/latest/apilist.html> |
| [mab] | MABWiser (not installed; found by this scan) | <https://fidelity.github.io/mabwiser/api.html> |
| (memory) | anything so marked | recalled, not read today; check the name against the library before a test cites it (the rule beneath task 510's siblings, PLAN "Each task: …") |

**Online class** (task 494's four words): **E** online exactly (one pass, the
library's batch number reproduced to rounding at the stated settings);
**A** online approximately (what it gives up is said); **M** online with an
embargo (a value for row `t − L` emitted at row `t`); **N** not online (why).

**Decay / window columns.** "yes" means feasible *and* checkable in the sense of
the rule beneath task 510: for decay, the library's fit with the decay's weights,
or the definition with the weights by hand beside the library's unweighted answer;
for a window, the library on exactly the window's rows. "no (obstacle)" or
"no (no oracle)" says which half fails. "n.a." means the form has no meaning for
the item.

**Tolerances.** "exact" means the kind of agreement the suite already holds
elsewhere: 1e-10 to 1e-12 relative for a closed form against a library (the
filterpy, sklearn `SGDRegressor` and scipy comparisons in ORACLE-COVERAGE §2).
"statistical" means a recursive estimate against a converged batch fit: the row
states the band and it goes in the oracle ledger (`oracles.toml`), where it only
tightens.

**State sizes** are per group and per spec, in doubles ×8 bytes, `k` features,
`T` targets. For scale: `ewridge`'s Gram is about `8·(k+1)(k+2)/2` bytes
(1.8 kB at k = 20).

**Effort** is in worker-days for a high-effort worker under this repository's
workflow (core model, bank, CLI, docstring with the update equations, MODELS.md,
EXTENDING's list, state fixtures, oracle tests): **S** up to 2, **M** 3 to 5,
**L** 6 to 10, **XL** more than 10.

## 0.1 Verdicts at a glance

| task | family | verdict | online | decay | window | oracle (exact unless said) | state | effort |
|---|---|---|---|---|---|---|---|---|
| 495 | VAR, Granger | new spec `var`, built on `ew_cov`'s Gram of the stacked lags and `solve.rs`; needs a lag buffer (task 532's, or its own) | E | yes | yes | [sm] `VAR` at no decay; `WLS` per equation under decay; `RollingOLS` in a window | `(k(p+1)+1)²/2` | M |
| 496 | fixed-lag Kalman smoother | **extend `kalman`** (`smooth_lag=L`) | M | inherits `q` | n.a. | [fp] `rts_smoother` on the prefix; [sm] state-space smoother as a second | `L·(1.5k² + 2.5k)·T` | M to L |
| 497 | EKF, UKF, particle | EKF/UKF: **extend `kalman`** with task 500's families; particle filter new | E (EKF/UKF), A (PF) | via `q`; [fp] fading | n.a. | [fp] `ExtendedKalmanFilter`, `UnscentedKalmanFilter`; PF: definition, [fp] resamplers | `k²` ; PF `N·k` | M + L |
| 498 | Hawkes | new spec `hawkes`; least-squares fit with given decays is a Gram integrated over the clock | E (intensity, LS fit), A (likelihood, decays) | yes (composite) | yes (composite) | [tick] `ModelHawkesExpKernLeastSq`, `HawkesExpKern`, `HawkesSumExpKern` | `(1+D·U)²/2 + D(1+D·U)` | L |
| 499 | realized volatility | **extend `rcov`** (new kinds) and the window operators | E | yes (EW sums) | yes | [hf] `tsrc`, `msrc`, `hayashi_yoshida`; bipower, MedRV and jump tests have **no Python library** | ring of `K` returns · `k` | M |
| 500 | GLMs | new spec `glm` in the reweight family beside `huber`/`quantile` | A (recursive IRLS) | yes | no (no oracle) | [sm] `GLM` row by row through its family objects (exact), and converged (statistical); [sk], [glum] seconds | `T·(k+1)(k+2)/2` | M to L |
| 501 | expectiles | new spec `expectile` **sharing `quantile`'s code** | A (one pass), E at the refit | yes | no (as `quantile`) | [sp] `scipy.stats.expectile(weights=)`; [sm] `RLM(M=MQuantileNorm(q, LeastSquares()))` | as `quantile` | S |
| 502 | kernel, random features | RFF, Nyström, splines: **feature helpers into `ewridge`/`rls`**; KRLS new | E (features), A (KRLS past its budget) | yes (features); no (KRLS) | yes (features); no (KRLS) | [sk] `RBFSampler`, `Nystroem`, `SplineTransformer` through `Ridge`; KRLS: [sk] `KernelRidge` until the budget binds | `D²/2` ; KRLS `m² + m·k` | S + L |
| 503 | online trees | new family; **state size must be raised with the user first** | E (Hoeffding tree, given order), A (ensembles: RNG) | no (no oracle) | no (a split cannot be unlearned) | [rv] `HoeffdingTreeRegressor/Classifier` node for node; ensembles statistical | leaves · k · bins · 3 | XL |
| 504 | contextual bandits | **policy layer on existing regressions** (`rls`/`kalman`/`ewridge` per arm) | E (LinUCB, greedy), A (sampling policies: RNG) | yes | yes | [mab] `LinUCB`/`LinGreedy` expectations; [rv] `LinUCBDisjoint`; [vw] only a regret band | `arms · k²/2` | M to L |
| 505 | streaming PCA | **`ew_cov(pca=)` already is exact PCA with decay and window**; build only the `O(r·k)` forms for wide inputs | A | no (unless `r = k`) | no (no downdate) | [sk] `IncrementalPCA.partial_fit` batch for batch | `r·k` | M |
| 506 | Gaussian mixtures | **extend `hmm`** (state-independent transitions) | A (online EM), E (posteriors at given parameters) | yes (composite) | yes (composite) | [sk] `GaussianMixture.predict_proba`/`score_samples` at given parameters; batch EM statistical | `K·(k²/2 + k + 1)` | S to M |
| 507 | anomaly detectors | half-space trees new; four of river's nine are **already covered** | E (HST given trees) | no (no oracle) | HST's own two windows | [rv] `HalfSpaceTrees` with the trees handed over; RRCF has no exact oracle | `n_trees·2^(h+1)·2` | M (+ L for RRCF) |
| 508 | change detectors | **all become options of `drift`**; ADWIN is the only new structure | E | n.a. (ADWIN), yes (PH, exists) | n.a. | [rv] `ADWIN`, `PageHinkley`, `DDM`… flag for flag; CUSUM: published ARL; [rup] for placement | ADWIN ≤ `5·log2(n)·2` | S to M |
| 509 | sketches | heavy hitters: **`audit` has Misra-Gries already**; HLL: extend `audit`; count-min new only for a per-row key count | A (within bounds) | CM yes; HLL no | CM yes; HLL no (no oracle) | [ds] `HyperLogLog(hashfunc=)` register for register; truth from Polars within the bound; **[ds] has no count-min**, [ads] and [rv] do | HLL `2^p` bytes; CM `w·d` | S to M |
| 510 | nonlinear dependence | mutual information: **extend `marginal`** (joint bins); ranks: windowed only, new ring | E (binned MI; windowed ranks) | MI yes; ranks no | yes | [sk] `mutual_info_score(contingency=)` on integer counts, [sp] `entropy` under decay; [sp] `spearmanr`, `kendalltau`, `chatterjeexi`; [dcor] | bins² per pair; ranks `W·k` | S + M |

## 0.2 Corrections to the plan's text that this scan found

1. **Task 509: `datasketch` has no count-min sketch.** Its documented classes are
   MinHash and its variants, the LSH indexes, `HyperLogLog`, `HyperLogLogPlusPlus`
   and `HNSW` [ds]. Count-min is in Apache `datasketches` (`count_min_sketch`) [ads]
   and river (`sketch.Counter`, `river/sketch/counter.py:12`) [rv].
2. **Task 497: filterpy has no particle filter.** `filterpy/monte_carlo/` holds
   only `resampling.py` (`multinomial_resample`, `residual_resample`,
   `stratified_resample`, `systematic_resample`, each taking `weights`) [fp]. The
   plan's wording, "a seeded resampler checked against filterpy's", is the most
   filterpy can give. pykalman has no EKF and no particle filter either [pk].
3. **Task 508: river has no CUSUM class** and no two-window test other than KSWIN.
   `river.drift` is `ADWIN`, `KSWIN`, `PageHinkley`, `DriftRetrainingClassifier`,
   `DummyDriftDetector`, `NoDrift` and `binary.{DDM, EDDM, FHDDM, HDDMA, HDDMW}`
   [rv]. `PageHinkley`'s docstring calls itself the CUSUM chart.
4. **Task 507: river 0.26.1 has no random cut forest.** `river.anomaly` is
   `GaussianScorer`, `HalfSpaceTrees`, `LODA`, `OneClassSVM`, `QuantileFilter`,
   `StandardAbsoluteDeviation`, `ThresholdFilter`, `LocalOutlierFactor`,
   `PredictiveAnomalyDetection` [rv]. PyOD documents only `fit`,
   `decision_function`, `predict`; none of its detectors learns incrementally [pyod].
5. **Task 505: river has no streaming PCA** (no `decomposition` module in 0.26.1).
   scikit-learn's `IncrementalPCA` is the only library oracle; CCIPCA and Oja have
   none in Python (R's `onlinePCA`, memory).
6. **Task 499: `arch` has no realized measure** [arch]: volatility processes, mean
   models (`HARX` among them), distributions, unit-root and cointegration tests,
   bootstraps and HAC kernels only. hfhd, already a dev dependency, has
   `tsrc` (two-scale), `msrc` (multi-scale), `mrc`, `krvm`, `hayashi_yoshida`,
   `preaverage` and `ensemble` (`hfhd/hf.py:733, 442, 874, 1393, 1637, 239, 2000`)
   [hf]. Nothing in Python computes bipower or tripower variation, MedRV/MinRV or
   the jump tests (R's `highfrequency` does; memory).
7. **Task 496: `UnobservedComponents` is not the oracle for `kalman`.** It is a
   level/trend/seasonal/cycle model of one series
   (`statsmodels/tsa/statespace/structural.py:48`) [sm], the oracle for a smoothed
   `holt`. `kalman` is a regression with random-walk coefficients, whose
   observation row is the feature row: its oracle is filterpy's `rts_smoother`
   (`filterpy/kalman/kalman_filter.py:1688`) on the filter's own stored steps, with
   statsmodels' `KalmanSmoother` under a time-varying design as a second.
8. **Task 501: two libraries compute expectiles.** `scipy.stats.expectile(a,
   alpha=0.5, *, weights=None)` (`scipy/stats/_stats_py.py`, wrapped at
   `_axis_nan_policy.py`) [sp] and statsmodels' `MQuantileNorm`
   (`statsmodels/robust/norms.py:1760`: "Expectiles are M-quantiles with the
   LeastSquares as base norm") [sm]. The plan expected a hand-written batch fit.
9. **Task 498: tick installs.** 0.8.0.2 (2026-05-04) publishes wheels for CPython
   3.11 to 3.14 on macOS arm64 and x86-64, manylinux x86-64 and Windows amd64
   [tick, PyPI JSON]; BSD-3-Clause. The plan's fallback ("where `tick` cannot be
   installed") should not be needed. Check Linux aarch64 if CI gains one: no wheel.
10. **Task 504: an exact LinUCB oracle exists.** MABWiser's `LinUCB(alpha=1.0,
    l2_lambda=1.0, scale=False)` is ridge per arm with the bound
    `x'β + α·sqrt(x'(X'X + λI)⁻¹x)` and `predict_expectations` returns it per arm
    [mab]. River's `LinUCBDisjoint` is built on its `BayesianLinearRegression`
    (`river/bandit/lin_ucb.py:10`), which `kalman` already reproduces
    (`test_river.py::TestKalmanIsBayesianLinearRegression`).

## 0.3 Shared pieces to build once, before the models that need them

| piece | needed by | note |
|---|---|---|
| A lag or row ring in the state | 495 (lags), 496 (the last `L` filter steps), 510 (windowed ranks), 499 (lagged returns; `rcov` has one) | task 532 builds lag operators with the buffer in the state; 495 should use it or share its ring |
| A counter-based random stream whose position is in the state | 497 (particles), 502 (random features, if drawn inside), 503 (bagging, ARF), 504 (Thompson, ε-greedy), 507 (tree construction) | keyed on (seed, group, learned-row count) so a chunking cannot move a draw and a saved state continues. `docs/ENHANCEMENTS.md` §4 rejected "randomness on the prediction path" for bandit model selection; the user's "add them all" reopens it, and the reply should say draws are reproducible, not that they are absent |
| The embargo's delayed-emission naming | 496, and tasks 490 and 522 | one rule for "the value of row `t − L`, emitted at `t`" |
| Family objects (variance, link, deviance) | 500, then 497's dynamic GLM, 498's likelihood step | write once in `rheon-core` |

## 1. Task 495: VAR and Granger causality

**In rheon today.** `ewridge` solves several targets from one Gram and has decay,
windows and `feature_sets`; `ew_cov(lags=)` and `marginal(lags=)` keep lagged
co-moments, from which Yule-Walker coefficients could be read, but Yule-Walker is
not the conditional least squares statsmodels' `VAR` fits. Nothing keeps lagged
*regressors*: a Polars `shift` is refused inside the stream (task 532's premise).

**Library surface** [sm], `statsmodels/tsa/vector_ar/var_model.py`:
`VAR(endog, exog=None, dates=None, freq=None, missing='none')` (:570);
`fit(maxlags=None, method='ols', ic=None, trend='c', verbose=False)` (:676);
`select_order(maxlags=None, trend='c')` (:846);
`VARResults.test_causality(caused, causing=None, kind='f', signif=0.05)` (:2119);
`test_inst_causality(causing, signif=0.05)` (:2249); `irf(periods=10,
var_decomp=None, var_order=None)` (:2047); `fevd(periods=10, var_decomp=None)`
(:2068); `forecast(y, steps, exog_future=None)` (:1211);
`forecast_interval(y, steps, alpha=0.05, exog_future=None)` (:1310);
`test_whiteness(nlags=10, signif=0.05, adjusted=False)` (:2370);
`test_normality(signif=0.05)` (:2464); `is_stable(verbose=False)` (:965).
Beside it: `grangercausalitytests(x, maxlag, addconst=True)`
(`tsa/stattools/_stattools.py:2368`), `VARMAX(endog, exog, order=(1, 0),
trend='c', error_cov_type='unstructured', measurement_error=False,
enforce_stationarity=True, enforce_invertibility=True, trend_offset=1)`
(`statespace/varmax.py:37`), `VECM(endog, exog, exog_coint, k_ar_diff=1,
coint_rank=1, deterministic='n', seasons=0, first_season=0)`
(`vector_ar/vecm.py:823`), `coint_johansen(endog, det_order, k_ar_diff)` (:594),
`select_coint_rank(endog, det_order, k_ar_diff, method='trace', signif=0.05)`
(:531), `SVAR(endog, svar_type, A=None, B=None)` (`svar_model.py:33`),
`DynamicFactor(endog, k_factors, factor_order, …)` (`dynamic_factor.py:33`).
scikit-learn, river and scipy have no VAR.

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| coefficients, `maxlags=p` given, `method='ols'` | [sm] `VAR.fit` | E: one Gram of `(1, y_t, y_{t−1}, …, y_{t−p})`, solved as `ewridge` solves | yes: `WLS(weights=λ^age)` per equation on the lag-embedded rows (equal to system GLS, the regressors being common) | yes: `RollingOLS`/`RollingWLS` on the lag-embedded frame, as `TestEwRidgeIsRollingWLS` | `VAR(...).fit(p).params` at `half_life=inf`, 1e-9 relative |
| `trend='n'/'c'/'ct'/'ctt'` | [sm] `fit` | E for `n`, `c`; `ct`, `ctt` need the row index as a feature (task 489's deterministic terms) | yes | yes | the same fit |
| `exog=` | [sm] `VAR` | E: extra features | yes | yes | the same fit |
| residual covariance `sigma_u` (df-adjusted) and `sigma_u_mle` | [sm] | E: from the Gram's `y y'` block; this is why the Gram is `ew_cov`'s on the stacked vector and not `ewridge`'s, which keeps no cross-target moment | yes: numpy `cov(aweights=)` of the WLS residuals | yes | 1e-9 |
| Granger test, `kind='f'` or `'wald'`, one caused variable | [sm] `test_causality` | E: a Wald form in the coefficients and `sigma_u ⊗ (Z'Z)⁻¹`, all in the Gram | yes: `WLS(...).fit().f_test(R)` on the caused equation (the restriction touches one equation) | yes | statistic 1e-8, p-value 1e-8 |
| Granger test, several caused variables | [sm] `test_causality` | E | by hand beside the unweighted library answer | yes | at no decay 1e-8 |
| instantaneous causality | [sm] `test_inst_causality` | E: a Wald test on `sigma_u`'s off-diagonal block; needs fourth-moment-free normal theory as statsmodels uses | by hand | yes | at no decay 1e-8 |
| bivariate `grangercausalitytests` (ssr F, ssr χ², LR, params F) | [sm] | E: the two-variable case of the above | as above | yes | 1e-8 |
| order selection: AIC, BIC, HQIC, FPE over `0..maxlags` | [sm] `select_order` | E: every lower order is a sub-block of the `maxlags` Gram (`feature_sets` does this for `ewridge`) | by hand (the criteria use a row count; under decay it is `n_eff`, a definition to state) | yes | at no decay, the same selected order and criteria to 1e-8 |
| `forecast(y, steps)`, `forecast_interval` | [sm] | E: `h` products of the companion matrix at emit time; out of sample by rule 2 | follows the coefficients | follows | 1e-9 against `forecast` from the same coefficients |
| `irf`, orthogonalized and cumulative; `fevd` | [sm] | E at read time from `coef` and `sigma_u` (a helper beside `rp.gram`, not a per-row output): cost `periods·k³` | follows | follows | 1e-9 |
| IRF standard errors (asymptotic, Monte Carlo) | [sm] `irf_errband_mc` and kin | asymptotic E at read time; Monte Carlo N (simulation) | by hand | yes | asymptotic only |
| `is_stable`, roots | [sm] | E at read time: eigenvalues of the companion matrix, non-symmetric | follows | follows | numpy `eigvals`, 1e-8 |
| `test_whiteness` (portmanteau), `test_normality` | [sm] | whiteness A: needs residual autocovariances at `nlags`, which `emit_specification`'s Ljung-Box keeps per slot but not across slots; normality E from `emit_tails`' sums per slot, the multivariate form needs cross fourth moments | no (no oracle under weights) | yes | leave to the per-slot diagnostics that exist; say so |
| `VARMAX` (moving-average terms, measurement error) | [sm] | N: a state-space likelihood maximized over the whole sample | | | not built |
| `VECM`, `coint_johansen` | [sm] | E in principle: Johansen's moment matrices `S00, S01, S11` are sums over the same stacked vector, and the rank test an eigenproblem at read time | by hand | yes | `coint_johansen(...).lr1`, `.eig` at no decay, 1e-8. **A candidate extension, not in task 495's text: propose it as its own task** |
| `SVAR` (A/B identification) | [sm] | N for the fit (a numerical likelihood on `sigma_u`); the short-run Cholesky case is the orthogonalized IRF above | | | not built |
| `DynamicFactor` | [sm] | N (state-space likelihood) | | | not built |

**Hard parts and risks.** (1) The lag buffer: `p` rows of `k` values per group in
the state, with a stated meaning for a null in the buffer (the row cannot be
learned), a zero-weight row (it advances the clock; does it enter the buffer? It
should, the value was observed), a session change and a step back (clear it).
Lags are in rows, as `ew_cov(lags=)`'s are, so on an uneven clock a "lag" is the
previous row, not a time offset: say it. (2) statsmodels drops the first
`maxlags` rows as presample; a windowed comparison must hand statsmodels the
lag-embedded frame, not the raw window. (3) Under decay the F form's denominator
degrees of freedom need a definition; reuse `emit_se_coef`'s. (4) Width: the Gram
is `(k(p+1)+1)²/2` doubles.

**State.** `8·m(m+1)/2 + 8·p·k` bytes with `m = k(p+1)+1`: 15 kB at k = 10,
p = 5; 1.2 MB at k = 50, p = 10. Cap `k·p` and state it; above `ewridge`'s
largest Gram in use it is a growth to raise.

**Effort.** M (4 to 5 days), S less if task 532's buffer lands first.

## 2. Task 496: a fixed-lag Kalman smoother

**In rheon today.** `kalman(coef_half_life, q, obs_var, p0, share_p,
revert_half_life, standardize, covariance_form)`: a regression whose coefficients
are a random walk or revert, filtered row by row; held to filterpy.

**Library surface.** [fp] `KalmanFilter.rts_smoother(Xs, Ps, Fs=None, Qs=None,
inv=…)` (`kalman_filter.py:920`), module `rts_smoother(Xs, Ps, Fs, Qs)` (:1688),
`FixedLagSmoother(dim_x, dim_z, N=None)` with `smooth(z, u=None)` and
`smooth_batch(zs, N, us=None)` (`fixed_lag_smoother.py:27`; its docstring:
"Implementation based on Wikipedia article as it existed on November 18, 2014",
and it returns smoothed means only), `UnscentedKalmanFilter.rts_smoother(Xs, Ps,
Qs=None, dts=None, UT=None)` (`UKF.py:614`). [sm] `KalmanSmoother.smooth(
smoother_output=None, smooth_method=None, results=None, run_filter=True, …)`
(`statespace/kalman_smoother.py:374`) with `set_smoother_output` (:192: state,
state covariance, state autocovariance, disturbances and their covariances) and
`set_smooth_method` (:266: conventional, classical, alternative, univariate);
`SimulationSmoother` (`simulation_smoother.py:23`); `UnobservedComponents(...)`
(`structural.py:48`); `RecursiveLS(endog, exog, constraints=None)`
(`regression/recursive_ls.py:34`), an `MLEModel` with `q = 0` whose results carry
`smoothed_state`. [pk] `KalmanFilter.smooth`, `filter_update`, `em(n_iter)`,
masked arrays for missing rows, `CholeskyKalmanFilter`; no fixed-lag smoother.

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| smoothed coefficients of row `t − L` given rows up to `t` | [fp] `rts_smoother`; [sm] `smoothed_state`; [pk] `smooth` | M, **exact**: the Rauch-Tung-Striebel backward pass from `t` to `t − L` reads only the filtered means and covariances of those `L + 1` rows, so the fixed-lag value equals the fixed-interval smoother run on the prefix and read at `t − L` | inherits the filter's `q` and `coef_half_life` | n.a. (`L` is the window) | [fp] `rts_smoother` on the prefix's stored `(x, P, F, Q)`, read at `t − L`: 1e-9. Second: [sm] a time-varying-design `MLEModel` (or `RecursiveLS` at `q = 0`), `smoothed_state[:, t − L]` on the prefix |
| smoothed covariance, standard errors | [fp] returns `P`; [sm] `smoothed_state_cov` | M, exact; costs `L·k³` a row where the mean costs `L·k²` | inherits | n.a. | the same, 1e-8 |
| smoothed fitted value `x_{t−L}'β_{t−L|t}` and residual | [sm] smoothed signal | M. **In-sample by construction** (it has seen `y_{t−L}`): must not be named `pred` (rule 2); name it by the lag, as task 490 does | inherits | n.a. | the same |
| lag `N` / `L` | [fp] `FixedLagSmoother(N=)` | parameter | | | filterpy's `FixedLagSmoother` itself is **not** the oracle: means only, an old transcription; use `rts_smoother` |
| lag-one autocovariance of the smoothed state | [sm] `smoothed_state_autocov` | M, exact; only an EM step needs it | | | not built unless task 497 or an EM asks |
| smoothed disturbances | [sm] `smoothed_state_disturbance`, `smoothed_measurement_disturbance` | M, exact (Durbin-Koopman backward recursion over the same `L` rows) | | | [sm], 1e-8; low value, propose to leave out |
| `smooth_method` (conventional, classical, alternative, univariate) | [sm] | one method suffices: the results agree to rounding | | | n.a. |
| missing observations | [pk] masked arrays; [sm] NaN | E: a zero-weight row is a predict-only step (rule 9), and its smoother gain is still defined | | | filterpy with the update skipped |
| parameter learning by EM | [pk] `em`; [sm] `fit` | N: iterates over the whole sample. An online EM is task 497's question, with no oracle | | | not built |
| simulation smoother | [sm] | N for a spec (draws); not asked | | | not built |
| a smoothed `holt` (level and trend) | [sm] `UnobservedComponents('lltrend')` | M: the same pass on `holt`'s two-state form. A candidate, since `holt` is a steady-state Kalman filter only once its weight saturates | | | [sm] `UnobservedComponents(...).smooth(params).smoothed_state` |

**Hard parts and risks.** (1) **`standardize=True` is `kalman`'s default**, and
it filters rows standardized by moments read before each row, so the coefficient
vector's coordinates move from row to row. The backward pass mixes rows `t − L`
to `t`: either smooth in raw coordinates (re-express each stored step), or refuse
`smooth_lag` with `standardize=True` at first. Decide before building. (2)
`share_p=True` (one covariance for all targets) saves `T`-fold on the stored
gains. (3) `embargo` (labels that arrive late) composes with `L`: the smoothed
row is `L` learned rows back. (4) The per-row cost: `O(L·k²)` for means with the
gains `C_s = P_s F' P_{s+1|s}⁻¹` stored at filter time, `O(L·k³)` with
covariances; give a cadence (`smooth_every`) as `coef_every` does. (5) A group's
first `L` rows and its last `L` (never emitted unless the group closes: emit them
at `group_close`).

**State.** Per target (or once under `share_p`): `L` × (gain `k²`, filtered
covariance `k(k+1)/2`, two means `2k`) doubles = `8·L·(1.5k² + 2.5k)` bytes:
260 kB at k = 20, L = 50, against the filter's 1.7 kB. **A significant growth:
raise it with the user before it lands**, with a cap on `L·k²`.

**Effort.** M to L (5 to 7 days), most of it the standardization question and the
state fixtures.

## 3. Task 497: EKF, UKF and particle filters

**Library surface.** [fp] `ExtendedKalmanFilter(dim_x, dim_z, dim_u=0)` with
`update(z, HJacobian, Hx, R=None, args=(), hx_args=(), residual=np.subtract)` and
`predict_update(z, HJacobian, Hx, args=(), hx_args=(), u=0)` (`EKF.py:32, 254,
172`); `UnscentedKalmanFilter(dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None,
x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None)` (`UKF.py:32`);
sigma points `MerweScaledSigmaPoints(n, alpha, beta, kappa, sqrt_method=None,
subtract=None)`, `JulierSigmaPoints(n, kappa=0.0, …)`, `SimplexSigmaPoints(n,
alpha=1, …)` (`sigma_points.py:24, 211, 386`); `CubatureKalmanFilter(dim_x,
dim_z, dt, hx, fx, …)`; `EnsembleKalmanFilter(x, P, dim_z, dt, N, hx, fx)`;
`IMMEstimator(filters, mu, M)`; `MMAEFilterBank(filters, p, dim_x, H=None)`;
`SquareRootKalmanFilter`; `InformationFilter`; `FadingKalmanFilter(alpha, dim_x,
dim_z, dim_u=0)`; `HInfinityFilter(dim_x, dim_z, dim_u, gamma)`; `GHFilter(x, dx,
dt, g, h)`; the four resamplers in `monte_carlo/resampling.py` (:27, :80, :117,
:153). [pk] `UnscentedKalmanFilter`, `AdditiveUnscentedKalmanFilter`; no EKF.
[sm] nothing nonlinear (its switching models are the Hamilton and Kim filters,
`hmm`'s territory).

**What a spec can express without user code** (rule 4 forbids a closure in the
core). The plan asks which transition and observation families fit. Proposed:

| family | transition | observation | filter |
|---|---|---|---|
| A. dynamic GLM | `kalman`'s own (random walk, or reverting) | `y ~ family(link⁻¹(x'β_t))`, the families and links of task 500 | EKF or UKF: **an option of `kalman`** (`family=`, `link=`, `filter=`) |
| B. stochastic volatility | `h_t = μ + φ(h_{t−1} − μ) + σ_η η_t` | `y_t = exp(h_t/2)·ε_t` | particle filter; or the linear filter on `ln y²` (Harvey, Ruiz and Shephard), which is `kalman` with a known non-Gaussian noise |
| C. heavy-tailed noise | `kalman`'s | Student-t observation noise | particle filter |
| D. a formula for `f` and `h` | a serialized expression | a serialized expression | would need an expression evaluator and derivatives in the core: **raise it, do not build it under 497** |

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| EKF predict and update, family A | [fp] `ExtendedKalmanFilter` | E: a recursion by definition, `O(k²)` state as `kalman` | via `q`/`coef_half_life`, as `kalman`; [fp] `FadingKalmanFilter(alpha)` is the library's fading form, checkable | n.a. | [fp] with `Hx = link⁻¹(x'β)`, `HJacobian = dμ/dη · x'`, `R = V(μ)·φ` evaluated at the prior mean: 1e-10 per row, as `TestKalmanSettingsAreFilterpy` |
| iterated EKF | none in [fp] | E | | | definition only: leave out |
| UKF, `MerweScaledSigmaPoints(alpha, beta, kappa)` | [fp] | E, `2k+1` points a row, `O(k³)` for the root | as above | n.a. | [fp] `UnscentedKalmanFilter` with the same points: 1e-9 (the matrix root: filterpy's default is scipy's Cholesky; match it) |
| `JulierSigmaPoints(kappa)`, `SimplexSigmaPoints(alpha)` | [fp] | E | | | [fp], 1e-9 |
| cubature filter | [fp] `CubatureKalmanFilter` | E | | | [fp], 1e-9; low value beside the UKF |
| additive-noise UKF | [pk] `AdditiveUnscentedKalmanFilter` | E | | | [pk] as a second opinion |
| square-root and information forms | [fp] | E; `kalman(covariance_form=)` already offers forms | | | exists |
| `HInfinityFilter(gamma)` | [fp] | E | | | [fp]; not asked, leave out |
| IMM, MMAE (banks of filters with switching) | [fp] `IMMEstimator(filters, mu, M)` | E: `K` filters and a `K×K` mixing step; it is `hmm` over `kalman`s | | | [fp], 1e-9. **A candidate of its own (regime-switching regression): propose as a task** |
| ensemble Kalman filter | [fp] `EnsembleKalmanFilter(N=)` | A: draws | | | statistical only; leave out |
| bootstrap particle filter, `N` particles | none ([fp] has resamplers only) | A: Monte Carlo error `∝ 1/√N`; reproducible given the counter-based stream | via the transition noise | n.a. | the definition, with (i) each resampling step equal to [fp] `systematic_resample`/`stratified_resample`/`residual_resample`/`multinomial_resample` on the same weights and the same uniforms (index for index), and (ii) on a linear-Gaussian model the filtered mean within `4·sd/√N` of `kalman`'s. A second opinion to look at: the `particles` package (Chopin; memory) |
| resampling scheme and effective-sample-size threshold | [fp] four schemes | E given the uniforms | | | index for index |
| auxiliary, guided proposals; SMC² parameter learning | `particles` (memory) | A; parameter learning N in one pass without a further approximation | | | no oracle: not built |
| filter log-likelihood | [fp] `log_likelihood` on the Kalman classes | E (EKF/UKF Gaussian approximation), A (PF) | | | [fp]; task 519 adds it to `kalman` first |
| RTS smoother for the UKF | [fp] `UnscentedKalmanFilter.rts_smoother` | M, as task 496 | | | [fp]; after 496 |

**Hard parts and risks.** (1) **Cross-platform continuation of a particle
filter.** Weights pass through `exp`, and a resampling index is a step function of
them: a last-bit difference between macOS and Windows libm can pick a different
ancestor, which is not a small tolerance (the libm rule's "tiny" cannot hold).
Either compute `exp`/`ln` in software (the pure-Rust `libm` crate gives the same
bits everywhere; a new dependency to raise under rule 12's spirit) or state that a
particle state continues to the bit only on the OS that saved it. Decide first.
(2) Normal draws without platform libm (a ziggurat or a rational inverse CDF in
plain arithmetic). (3) Family A's EKF with a Poisson row at a large linear
predictor overflows: clamp `η` as statsmodels' links do. (4) Family A overlaps
task 500: the EKF with `q = 0` *is* recursive IRLS. Build 500's family code first
and let both use it.

**State.** EKF/UKF: `kalman`'s. Particle filter: `8·N·d` bytes plus the stream's
counter (`d` the state dimension): 80 kB at N = 10,000, d = 1 per group. Raise.

**Effort.** EKF/UKF as `kalman` options after 500: M (3 to 4 days). Particle
filter with family B: L (6 to 8 days), most of it determinism.

## 4. Task 498: Hawkes and other point processes

**In rheon today.** `rp.ewm_sum`/`rp.ewm_rate` keep an exponentially decayed sum
on the clock, which is the exponential kernel's state `R(t) = Σ e^{−β(t − t_j)}`.
No model has an event intensity, a compensator or a fit.

**Library surface.** [tick] learners: `HawkesExpKern(decays,
gofit='least-squares', penalty='l2', C=1000.0, solver='agd', step=None,
tol=1e-05, max_iter=100, verbose=False, print_every=10, record_every=10,
elastic_net_ratio=0.95, random_state=None)` (`gofit` also `'likelihood'`;
`penalty` `'l1'`, `'l2'`, `'elasticnet'`, `'nuclear'`, `'none'`; `solver` `'gd'`,
`'agd'`, `'bfgs'`, `'svrg'`; attributes `baseline`, `adjacency`, `coeffs`);
`HawkesSumExpKern(decays, penalty='l2', C=1000.0, n_baselines=1,
period_length=None, solver='agd', …)` (`n_baselines > 1`: a piecewise-constant
periodic baseline); `HawkesADM4(decay, …)` (lasso plus nuclear norm);
`HawkesSumGaussians(max_mean_gaussian, …)`; `HawkesEM(kernel_support,
kernel_size, …)`; `HawkesBasisKernels(kernel_support, …)`;
`HawkesConditionalLaw(delta_lag, min_lag, …)` (marked processes);
`HawkesCumulantMatching(integration_support)`. Models:
`ModelHawkesExpKernLeastSq(decays)`, `ModelHawkesExpKernLogLik(decay)`,
`ModelHawkesSumExpKernLeastSq(decays)`, `ModelHawkesSumExpKernLogLik(decays)`.
Simulation: `SimuHawkesExpKernels(adjacency, decays, …)`,
`SimuHawkesSumExpKernels`, `SimuHawkes`, kernels `HawkesKernelExp(intensity,
decay)`, `HawkesKernelSumExp`, `HawkesKernelPowerLaw(multiplier, cutoff, …)`,
`HawkesKernelTimeFunc`, `SimuPoissonProcess`, `SimuInhomogeneousPoisson`.
[hawkes] one-dimensional: `estimator().set_kernel(type, **kwargs)` with `'exp'`
(`num_exp`), `'pow'`, `'nonpara'` (`support`, `num_bin`); `set_baseline(type)`
with `'const'`, `'loglinear'`, `'plinear'`, `'pconst'` (`num_basis`), `'custom'`
(`l_custom`); `fit(T, itv, prior=[], opt=[], merge=[])`; after it `para`, `L`,
`AIC`, `br` (branching ratio), `ste`; `tl()` (the intensity path), `t_trans()`
(time rescaling), `predict(en_f, num_seq=1)`; `simulator().simulate(itv)`. MIT.

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| intensity before each event, exponential kernel, given `(μ, α, β)`; `D` event types | [tick] `HawkesKernelExp`, `estimated_intensity` (memory); [hawkes] `tl()` | E: `R_ij ← e^{−βΔ}R_ij`, then add the event; out of sample by reading before the add | n.a. (given parameters) | n.a. | [tick] the log-likelihood model's value at the given coefficients, or [hawkes] `tl()` in one dimension: 1e-10 |
| sum of `U` exponentials, given decays | [tick] `HawkesSumExpKern`, `HawkesKernelSumExp`; [hawkes] `'exp', num_exp` | E: `U` states per pair | n.a. | n.a. | the same |
| compensator between events; time-rescaled residuals; branching ratio | [hawkes] `t_trans()`, `br`; [tick] kernel norms | E in closed form | | | [hawkes] `t_trans()`, 1e-9; a KS check is then scipy's |
| **least-squares fit, decays given** | [tick] `HawkesExpKern(gofit='least-squares')`, `ModelHawkesExpKernLeastSq` | **E**: the contrast `∫λ_i² dt − 2Σ_{events of i} λ_i` is quadratic in `(μ_i, α_i·)`. Its matrix `G = ∫φφ' dt`, `φ = (1, R_·)`, is one Gram shared by every node and its linear terms are sums at the events: the shape of `ewridge` with several targets, integrated over the clock between events in closed form | yes: the same integrals with `e^{−ρ(T − s)}` inside, checked by scipy `quad` beside tick's unweighted loss | yes by the composite (a snapshot ring of `G`, as `ewridge`'s windows); tick on the window's events alone starts its kernels empty, so it is not the oracle for the window | tick's loss is quadratic, so its value at a handful of coefficient vectors recovers `G` and `b`: 1e-9; the minimizer against `HawkesExpKern(penalty='none', solver='bfgs', tol` tightened`)`: 1e-5 |
| penalties: `l2`; `l1`, `elasticnet`; `nuclear` | [tick] | `l2` E (a ridge on the Gram); `l1` E through the lasso solver on the Gram; nuclear N (a matrix prox per iteration; no use here) | as the fit | as the fit | [tick] at the same `C` (its penalty is `1/C`): 1e-5 |
| non-negativity of `μ`, `α` | [tick] projects onto it | E: a small non-negative least squares at solve time | | | the same |
| periodic piecewise-constant baseline, `n_baselines`, `period_length` | [tick] `HawkesSumExpKern` | E: `n_baselines` indicator features on the clock's phase (task 489's terms) | yes | yes | [tick], 1e-5 |
| likelihood fit, decays given | [tick] `gofit='likelihood'`, `ModelHawkesExpKernLogLik`; [hawkes] `fit` | A: the log-likelihood is concave in `(μ, α)` but its event term `Σ ln λ` is not a sum of fixed statistics; one Newton step per event at the fit so far (the reweight pattern of `huber`) | yes (weights in the same sums) | no (no oracle for a windowed likelihood with kernel memory) | [tick] or [hawkes] converged: statistical, within 3 standard errors (`ste` from [hawkes]) after a stated number of events |
| estimating the decay `β` | [hawkes] (quasi-Newton over all parameters); [tick] takes it as given | A: non-convex. Offer a grid of decays in one spec and select by out-of-sample log-likelihood, as `ewridge` selects a ridge | follows | follows | [hawkes] `para` in one dimension: statistical |
| power-law kernel | [tick] `HawkesKernelPowerLaw`; [hawkes] `'pow'` | A: not Markov; a fixed sum of exponentials approximates it, with a stated error | | | [hawkes] `'pow'` likelihood, statistical; propose to offer the sum of exponentials only |
| non-parametric kernels: `HawkesEM`, `HawkesBasisKernels`, `'nonpara'`, `HawkesSumGaussians` | [tick], [hawkes] | N as they stand (each needs every pair of events within the support). A histogram kernel over a bounded support `S` is E with a ring of the events inside `S`: window memory | | | [tick] `HawkesEM` statistical; not in the first build |
| `HawkesConditionalLaw`, `HawkesCumulantMatching` | [tick] | N (moment estimators over the whole sample; cumulants could be accumulated but the inversion is not checked row by row) | | | not built |
| marks (size-dependent excitation) | [tick] `HawkesConditionalLaw` | E for a mark that scales the jump (`R += g(mark)`): it is the row weight | | | definition; [tick] has no parametric marked learner |
| simulation, `predict` | [tick] `Simu*`; [hawkes] `predict`, `simulate` | not a streaming output; tests use them to generate data | | | n.a. |
| Poisson and inhomogeneous Poisson rates | [tick] `SimuPoissonProcess`, `SimuInhomogeneousPoisson` | the `α = 0` case; `rp.ewm_rate` already estimates a rate | | | exists |

**Hard parts and risks.** (1) Mapping to rows: a row is an event of type `d`
(a declared set, as `ew_class`'s `classes`), and **a zero-weight row is an
observation time with no event**, which rule 9 already makes legal and which the
integrals need at a group's end. (2) Simultaneous events (`Δ = 0`) and a
nanosecond clock: the decay's unit must be stated, as `ewm_rate`'s is. (3) The
closed-form integrals of products of exponentials lose digits when two decays are
close or `βΔ` is small: use `expm1` forms; every such value sits in the state and
is a libm result (the 2026-11-20 rule: compare across OSes within a stated few
ulps). (4) `n_eff` (rule 8): the decayed event count before the row. (5)
Stationarity: a fit with branching ratio at or above 1 must be reported, not
clamped silently.

**State.** `8·[(1+DU)(2+DU)/2 + D(1+DU) + D·D·U]` bytes: 0.4 kB at D = 2, U = 3;
9 kB at D = 10, U = 3.

**Effort.** L (7 to 9 days): intensity and LS fit first (the exact part), the
likelihood step and the decay grid second.

## 5. Task 499: realized-volatility estimators

**In rheon today.** `rcov(kind="plain"|"kernel"|"preavg", kernel="parzen",
bandwidth, jitter, theta, psd, block_rows, max_bandwidth, preavg_rows,
noise_stride, iv_stride)`: realized covariance per block, reported at the group's
close, with noise-variance, integrated-variance and quarticity estimates from
subsampling; `rp.stream.refresh_time`; hfhd holds the kernel and the pre-averaged
estimate. An exponentially weighted realized variance is `rp.ewm_sum` of squared
returns already.

**Library surface.** [hf] `tsrc(tick_series_list, J=1, K=None)` (:733),
`msrc(tick_series_list, M=None, N=None, pairwise=True)` (:442),
`mrc(tick_series_list, theta=None, g=None, bias_correction=True, …)` (:874),
`krvm(tick_series_list, H, pairwise=True, kernel=quadratic_spectral_kernel)`
(:1393), `hayashi_yoshida(tick_series_list, theta=None, k=None)` (:1637),
`preaverage(data, K=None, g=None, return_K=False)` (:239), `get_bandwidth(n,
var_ret, var_noise, kernel)` (:1305), `parzen_kernel`,
`quadratic_spectral_kernel`, `ensemble(estimates, var_weights, cov_weights)`
(:2000), `refresh_time` (:24). [arch] none (0.2 item 6); its `HARX` mean model is
the oracle for a HAR regression *on* realized variances, which is `ewridge` on
three window means.

| estimator | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| realized variance and covariance | [hf] none needed; `rcov(kind="plain")` | E; exists | yes: `rp.ewm_sum(r²)` exists | yes: a clock window of the sum | numpy, exists |
| bipower variation `μ₁⁻² Σ|r_i||r_{i−1}|` | none in Python (R `highfrequency::rBPCov`, memory) | E: one lagged return | yes (an EW sum of the products) | yes | **definition only** (Barndorff-Nielsen and Shephard 2004) plus known truth: a simulated jump diffusion whose integrated variance is known. No third-party Python oracle |
| tripower, quadpower; realized quarticity (tri- and quad-power) | none in Python | E: two or three lagged returns | yes | yes | definition only |
| MedRV, MinRV (Andersen, Dobrev and Schaumburg) | none in Python | E: a median or minimum of 3 or 2 adjacent returns | yes | yes | definition only |
| realized semivariance (up, down), signed jump variation | none in Python | E | yes | yes | numpy on the signed squares (the definition is a masked sum: numpy *is* the library) |
| two-scale estimator | [hf] `tsrc(J, K)` | E: `K` subsampled sums and the full-grid sum, a ring of `K` returns | by hand beside hfhd | yes | [hf] `tsrc`: 1e-10, with hfhd's grid conventions matched as `mrc`'s were (ORACLE-COVERAGE, fifth pass) |
| multi-scale estimator | [hf] `msrc(M, N)` | E: `M` scales, a ring of `M` returns | by hand | yes | [hf] `msrc(pairwise=False)`: 1e-10 |
| realized kernel, Parzen | [hf] `krvm` | E; exists | no (the end-point jitter has no decayed form) | block only | exists |
| other kernels: quadratic spectral, Tukey-Hanning, cubic | [hf] `quadratic_spectral_kernel` | E for a finite lag window; QS has infinite support, so it needs every lag: N as a stream unless truncated | | | [hf] `krvm(kernel=)` |
| pre-averaged (modulated) covariance | [hf] `mrc`, `preaverage` | E; exists | no | block only | exists |
| Hayashi-Yoshida (asynchronous covariance) | [hf] `hayashi_yoshida(theta, k)` | E: each pair keeps its two open intervals; with pre-averaging (`theta`, `k`) a ring | by hand | yes | [hf]: 1e-10. Needs rows that carry one asset's tick at a time (a long frame), which is a different input shape from `rcov`'s refresh-time grid: say so |
| realized range (Parkinson, Garman-Klass per bar) | none here | E from bar highs and lows | yes | yes | definition only; low priority |
| jump tests: BNS ratio and difference; Lee-Mykland per row; Andersen-Bollerslev-Dobrev; Aït-Sahalia-Jacod | none in Python (R `highfrequency::BNSjumpTest`, `intradayJumpTest`, `AJjumpTest`, memory) | E: each is a function of RV, BPV and tri-power quarticity over a block or a trailing window; Lee-Mykland reads a trailing bipower window before the row, so it is out of sample | n.a. (a test over a stated span) | yes (the span is the window) | definition only, plus size and power on simulated paths (as `corrchange`'s tests are held) |
| automatic bandwidth and noise variance | [hf] `get_bandwidth` | exists in `rcov` | | | exists |
| `ensemble` of estimators | [hf] | a weighted mean at read time | | | trivial |
| HAR-RV forecasting | [arch] `HARX` | E: `ewridge` on daily, weekly and monthly means of the block values; no new model | yes | yes | [arch] `HARX(lags=[1, 5, 22]).fit()` at no decay, if arch is added to the dev group |

**Verdict: extend `rcov`, do not build a model.** New `kind`s (`bipower`,
`medrv`, `minrv`, `semi`, `twoscale`, `multiscale`, `hy`), a `jump_test=` output
at the block's close, and the per-row trailing forms (Lee-Mykland, a rolling
bipower) as window operators or a per-row output of `rcov`.

**Hard parts and risks.** (1) **The oracle gap.** Bipower, tripower, MedRV/MinRV
and every jump test have no Python library. By the rule beneath task 510 a form
with no oracle is not built; ORACLE-COVERAGE counts a longhand from the paper as
an oracle of the weaker kind ([L]), and `rcov`'s own kinds began that way. Ask
the user which reading applies here; a frozen table of R `highfrequency` outputs
on a seeded series would be a published-value oracle ([P]) if R may be run once
off-line. (2) `rcov` reports at close; per-row trailing outputs change what it
emits. (3) Small-sample constants (`μ₁`, the finite-sample corrections `n/(n−1)`)
differ between papers and packages: state which.

**State.** One to three lagged returns per column for the power variations;
`K` (or `M`) returns per column for the scale estimators: `8·K·k` bytes.

**Effort.** M (4 to 5 days) for the kinds with hfhd oracles and the power
variations; S more for the jump tests.

## 6. Task 500: online GLMs

**In rheon today.** `sgd(loss="poisson")` and `sgd`/`ftrl` logistic are
first-order steps. `huber` and `quantile` are the "reweight" family: running
sums, each row entering with a weight computed from the fit before it.
`docs/ENHANCEMENTS.md` §5.1 recorded "`sgd` carries the GLM losses"; the task
reopens it for the second-order fit.

**Library surface.** [sm] `GLM(endog, exog, family=None, offset=None,
exposure=None, freq_weights=None, var_weights=None, missing='none')`
(`genmod/generalized_linear_model.py:66`); `fit(start_params=None, maxiter=100,
method='IRLS', tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None,
use_t=None, full_output=True, disp=False, max_start_irls=3)` (:1161);
`fit_regularized(method='elastic_net', alpha=0.0, start_params=None, refit=False,
opt_method='bfgs')` (:1509); `fit_constrained(constraints, …)` (:1687);
`estimate_tweedie_power(mu, method='brentq', low=1.01, high=5.0)` (:964).
Families (`genmod/families/family.py`) and the links each allows:
`Poisson` (Log, Identity, Sqrt); `Gamma` (Log, Identity, InversePower);
`Tweedie(link=None, var_power=1.0, eql=False)` (Log, Power);
`NegativeBinomial(link=None, alpha=None)` (Log, CLogLog, Identity,
NegativeBinomial, Power); `Gaussian` (Log, Identity, InversePower);
`InverseGaussian` (InverseSquared, InversePower, Identity, Log); `Binomial`
(Logit, Probit, Cauchy, Log, LogC, CLogLog, LogLog, Identity). Likelihood count
models: `discrete_model.Poisson`, `NegativeBinomial(loglike_method='nb2')`,
`NegativeBinomialP(p=2)`, `GeneralizedPoisson(p=1)`,
`count_model.ZeroInflatedPoisson(exog_infl, inflation='logit')`,
`truncated_model.HurdleCountModel(dist='poisson', zerodist='poisson')`.
[sk] `PoissonRegressor(alpha=1.0, fit_intercept=True, solver='lbfgs',
max_iter=100, tol=0.0001, warm_start=False)`, `GammaRegressor(…)`,
`TweedieRegressor(power=0.0, alpha=1.0, link='auto', …)`
(`linear_model/_glm/glm.py:490, 629, 761`); `fit(X, y, sample_weight)`.
[glum] `GeneralizedLinearRegressor(alpha=None, l1_ratio=0, P1='identity',
P2='identity', fit_intercept=True, family='normal', link='auto', solver='auto',
max_iter=100, gradient_tol=None, step_size_tol=None, hessian_approx=0.0,
warm_start=False, alpha_search=False, n_alphas=100, start_params=None,
selection='cyclic', scale_predictors=False, lower_bounds=None,
upper_bounds=None, A_ineq=None, b_ineq=None, drop_first=False, robust=True,
expected_information=False, …)`; families `'binomial'`, `'gamma'`,
`'gaussian'`/`'normal'`, `'inverse.gaussian'`, `'poisson'`, `'tweedie (p)'`,
`'negative.binomial (theta)'`; links `'auto'`, `'identity'`, `'log'`, `'logit'`,
`'cloglog'`; solvers `'irls-cd'`, `'irls-ls'`, `'lbfgs'`, `'trust-constr'`,
`'closed-form'`. [rv] `linear_model.LinearRegression(loss=optim.losses.Poisson())`
only (first order).

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| Poisson, log link | [sm], [sk], [glum] | A: one IRLS (Fisher scoring) step per row at the fit before the row; the sums are `X'WX` and `X'Wz`, `W = (dμ/dη)²/V(μ)`, `z` the working response. With no decay it converges to the maximum likelihood fit at the batch rate | yes: the same step with decayed sums; the converged target is `GLM(freq_weights=λ^age)` | no: a row's `W` is frozen at the fit that scored it, so the window's sums are not the window's IRLS; no library computes them | (i) **row by row, exact**: a longhand recursion whose `W`, `z`, variance and link derivative come from statsmodels' own family object (`family.weights(mu)`, `family.link.deriv(mu)`, `family.variance(mu)`): 1e-10. (ii) **converged, statistical**: `GLM(...).fit()` on the same rows, coefficients within a stated multiple of their standard errors after a stated count, in the ledger (as `convergence.huber`). (iii) [sk] `PoissonRegressor(alpha=0)` and [glum] as seconds |
| a refit of the first rows to convergence | none | E on the buffer (as `huber_refit_rows`, `quantile_refit_rows`) | | | [sm] `GLM.fit()` on exactly the buffer: 1e-8. This is the exact library check of the IRLS arithmetic |
| Gamma (log; inverse power, identity) | [sm], [sk], [glum] | A, the same step | yes | no | as above; log link first. Identity and inverse links can leave `μ ≤ 0`: guard and withhold, never clamp silently |
| Tweedie, `var_power` given, `1 < p < 2` (and the other ranges) | [sm] `Tweedie(var_power)`, [sk] `TweedieRegressor(power)`, [glum] `'tweedie (p)'` | A | yes | no | as above. `powf` enters the working weight: a libm result in the state (the 2026-11-20 rule) |
| Tweedie power estimated | [sm] `estimate_tweedie_power` (a root-find over the fitted means) | N as the library does it. A grid of powers in one spec, selected by out-of-sample deviance, is the form this library has | follows | | [sm] at each fixed power; the selection is the library's own rule (as `lasso`'s) |
| negative binomial, `alpha` given | [sm] `families.NegativeBinomial(alpha)`, [glum] `'negative.binomial (theta)'` | A | yes | no | as above |
| negative binomial, `alpha` estimated | [sm] `discrete_model.NegativeBinomial` (joint likelihood) | A: a moment estimate from the running Pearson statistic; not the library's estimator | yes | no | statistical only, against [sm] `NegativeBinomial(...).fit().params[-1]`; mark it approximate, or offer the grid as for Tweedie |
| binomial (logit, probit, cloglog) | [sm], [glum] | A; comes free with the family code; overlaps `ftrl` and `sgd(loss="logistic")`, which stay the first-order choices | yes | no | as above |
| inverse Gaussian | [sm], [glum] | A; comes free | yes | no | as above |
| links per family (the lists above) | [sm] | each is a pair of functions; the safe ones first (`Log` for the count and positive families) | | | [sm] link objects in the composite oracle |
| `offset=`, `exposure=` | [sm]; [glum] `fit(offset=)` (memory) | E: a column added to the linear predictor | | | [sm] |
| `freq_weights`, `var_weights`, `sample_weight` | [sm], [sk], [glum] | the row `weight` is `freq_weights` | | | [sm]: pin which of the two a row weight means (they differ in the scale and the standard errors) |
| `scale`: Pearson `X2`, deviance, fixed | [sm] `fit(scale=)` | E: running weighted sums of Pearson residuals read before each row | yes | no | [sm] `scale` at the converged fit: statistical; row by row by the composite |
| standard errors `cov_type='nonrobust'`, `HC0`… | [sm]; [glum] `robust=True` | A: `(X'WX)⁻¹·scale` from the state; the sandwich needs the score's outer product, one more Gram per target | yes | no | [sm] `bse` statistical; ties to task 512 |
| deviance, Pearson χ², log-likelihood, AIC | [sm] results | E as out-of-sample running metrics of each row's unit deviance (an extension of `emit_metrics`); `ln` at read time | yes | yes (a sum of per-row values) | [sm] `family.deviance`, `family.loglike_obs` per row: 1e-10 |
| ridge and elastic net: `alpha`, `l1_ratio`, `P1`, `P2`, `fit_regularized` | [sk] `alpha`, [glum], [sm] | ridge E within the step (as `huber(ridge=)`); L1 through the lasso solver on the weighted Gram: A | yes | no | [sk] `alpha`, [glum]: statistical |
| bounds and linear constraints | [glum] `lower_bounds`, `A_ineq`; [sm] `fit_constrained` | equality E on the Gram (task 516); bounds N (E40's rejection) | | | task 516 |
| zero-inflated, hurdle, generalized Poisson, `NegativeBinomialP` | [sm] `discrete`, `count_model`, `truncated_model` | N by IRLS: extra parameters outside the exponential family; only a gradient step fits them | | | not built; say so |
| `GEE` (clustered) | [sm] | N: needs the cluster's rows together | | | not built |

**Hard parts and risks.** (1) Start-up: the first rows' weights are computed at a
poor fit and never revisited, the defect `huber`'s refit buffer exists for; take
the same remedy (`glm_refit_rows`). (2) `W` depends on the target, so several
targets cannot share one Gram: state is `T` Grams, as `huber`'s. (3) Overflow of
`exp(η)`; statsmodels clips, and the composite oracle must clip identically.
(4) A zero-weight first row (rule 9) and a row with `y` outside the family's
support (a negative count): refuse or skip, and say which. (5) The windowed form
is left out because no oracle exists; record that reason in the docstring's
refusal.

**State.** `8·T·(k+1)(k+2)/2` bytes plus `T·(k+1)` cross-moments and the scale's
sums: `huber`'s size.

**Effort.** M to L (5 to 7 days) for Poisson, Gamma, Tweedie, negative binomial
with given dispersion, log link, offsets, scale, the refit and the two oracles;
S more for the remaining links and families.

## 7. Task 501: expectiles

**In rheon today.** `quantile(quantile, ridge, standardize, solve_every,
quantile_eps, quantile_refit_rows)`: one Newton step per row on a smoothed check
loss. `huber` reweights rows by the fit before them.

**Library surface.** [sp] `scipy.stats.expectile(a, alpha=0.5, *, weights=None,
axis=None, nan_policy='propagate', keepdims=False)`. [sm]
`robust.norms.MQuantileNorm(q, base_norm)` (`norms.py:1760`) under
`RLM(endog, exog, M=…)` (`robust/robust_linear_model.py:37`); with
`base_norm=LeastSquares()` it is Newey and Powell's asymmetric least squares.
[sk], [rv], [glum]: none (river's losses are `Absolute`, `Cauchy`,
`EpsilonInsensitiveHinge`, `Huber`, `Poisson`, `Quantile`, `Squared`).

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| expectile of one column at level `τ` (no features) | [sp] `expectile(a, alpha, weights=)` | A in one pass (the row's side of the fit is judged before the row); E at the refit | yes: `weights=λ^age` | yes: scipy on the window's values (a windowed form would need the ring to re-judge sides: offer it only if `quantile` gains one) | [sp]: statistical for the running value, 1e-10 at the refit |
| expectile regression | [sm] `RLM(M=MQuantileNorm(τ, LeastSquares()))` | A: reweighted least squares, weight `|τ − 1{r < 0}|` from the residual before the row. The loss is piecewise quadratic, so unlike `quantile` no smoothing bandwidth is needed and the fixed point is exact | yes: `RLM` has no weights, but the loss is homogeneous of degree 2, so `RLM` on rows scaled by `√w` (the constant column included) is the weighted fit (run today, intercept only, `τ = 0.8`, weights `0.5^(age/100)`: `RLM` 2.6668528196795602 against scipy's weighted `expectile` 2.66685281967956) | no (as `quantile`) | [sm]: statistical for the one-pass fit; 1e-8 on the refit buffer (`expectile_refit_rows`), as `quantile_refit_rows` is held to `QuantReg` |
| `ridge`, `standardize`, solve cadence | none | as `quantile` | | | the stationarity condition by hand, as `TestTheQuantileFitsDefinition` |
| several levels in one spec | none | E to share the Gram? No: the weights differ by level, so one Gram per level. Task 524 asks the same of `quantile` | | | per level |
| standard errors | [sm] `RLM` `cov='H1'`… | A (task 512's sandwich for M-estimators covers it) | | | [sm] `bse`, statistical |
| expectile-implied tail measures (the quantile level an expectile matches; expected shortfall) | none | a read-time transform needing the residual distribution (`resid_quantiles` has it) | | | no oracle: not built |

**Verdict.** A new spec name (`expectile`) on `quantile`'s code path: the same
state layout with a different row rule. Not a new model family.

**Hard parts and risks.** The `quantile` review finding (N9: a secant weight
unbounded near zero) does not arise: the expectile weight is bounded in
`[min(τ, 1−τ), max(τ, 1−τ)]`. Extreme `τ` (0.99) makes the effective sample small
on one side: report `n_eff` per side or warn.

**State.** `quantile`'s. **Effort.** S (2 days).

## 8. Task 502: kernel and random-feature regression

**In rheon today.** `docs/ENHANCEMENTS.md` §5.1, "Two that need no build":
random Fourier features drawn once and written into the query as constants, fed
to `ewridge`. Nothing in the package draws them.

**Library surface.** [sk] `RBFSampler(gamma=1.0, n_components=100,
random_state=None)` (`kernel_approximation.py:253`; fitted attributes
`random_weights_`, `random_offset_`); `Nystroem(kernel='rbf', gamma=None,
coef0=None, degree=None, kernel_params=None, n_components=100,
random_state=None, n_jobs=None)` (:859); `AdditiveChi2Sampler(sample_steps=2,
sample_interval=None)`; `SkewedChi2Sampler(skewedness=1.0, n_components=100,
random_state=None)`; `PolynomialCountSketch(gamma=1.0, degree=2, coef0=0.0,
n_components=100, random_state=None)`; `KernelRidge(alpha=1, kernel='linear',
gamma=None, degree=3, coef0=1, kernel_params=None)` (`kernel_ridge.py:21`);
`GaussianProcessRegressor(kernel=None, alpha=1e-10, …)`;
`SplineTransformer(n_knots=5, degree=3, knots='uniform',
extrapolation='constant', include_bias=True, …)`
(`preprocessing/_polynomial.py:586`); `PolynomialFeatures(degree=2,
interaction_only=False, include_bias=True)`. [rv]
`feature_extraction.RBFSampler(gamma=1.0, n_components=100, seed=None)`
(`kernel_approx.py:15`), `PolynomialExtender`.

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| random Fourier features, RBF kernel: `gamma`, `n_components`, seed | [sk] `RBFSampler`; [rv] `RBFSampler` | E: the map is stateless once drawn, and `ewridge`/`rls` on it is exact for that draw | yes (`ewridge`'s) | yes (`ewridge`'s) | hand scikit-learn's `random_weights_` and `random_offset_` to the helper (it must accept given frequencies and phases as well as a seed), then [sk] `Ridge(sample_weight=)` on `RBFSampler.transform(X)`: 1e-10. The seeded draw itself is this library's and is tested for its distribution only |
| other shift-invariant kernels (Laplacian: Cauchy frequencies; Matérn: Student-t) | none in [sk] | E, a different draw | yes | yes | definition (Rahimi and Recht); the kernel approximation error against scipy's exact kernel, statistical |
| `SkewedChi2Sampler`, `AdditiveChi2Sampler`, `PolynomialCountSketch` | [sk] | E (stateless maps) | yes | yes | [sk] with its fitted attributes handed over; for histogram-like inputs, low value here |
| Nyström features: `kernel`, `gamma`, `degree`, `coef0`, `n_components` | [sk] `Nystroem` | E after the landmarks are fixed (the first `m` learned rows, or given); before that the model is warming up | yes | yes | [sk] `Nystroem(n_components=m).fit(landmarks)` then `Ridge`: predictions 1e-8 (the feature basis is defined up to a rotation; predictions are not) |
| splines: `n_knots`, `degree`, `knots`, `extrapolation` | [sk] `SplineTransformer` | E with given knots (stateless); knots from the first rows as `marginal`'s learned edges | yes | yes | [sk] `SplineTransformer` with the same knots: 1e-12 |
| polynomial features | [sk] `PolynomialFeatures`; Polars expressions | exists upstream (Polars) | | | n.a. |
| kernel ridge, exact | [sk] `KernelRidge` | N: its state is every row | | | the target the approximations are compared with |
| kernel recursive least squares with a dictionary budget (approximate linear dependence; Engel, Mannor and Meir 2004) | none | A: order-dependent; once the budget `m` binds, rows are projected onto the dictionary | no (a forgetting KRLS exists in the literature, with no library) | no | **until the budget binds and at threshold 0 it equals [sk] `KernelRidge` on the same rows: 1e-8.** After that only a bound on the excess error against `KernelRidge`: statistical |
| kernel LMS, quantized KLMS, NORMA | none here | A | | | no oracle: not built |
| Gaussian-process regression (predictive variance) | [sk] `GaussianProcessRegressor` | N exactly; the random-feature ridge gives a predictive variance through `kalman` on the features (`p0`, `obs_var`): E for that draw | | | [sk] GPR, statistical |

**Verdict.** Random features, Nyström and splines are **feature helpers** (a
builder that returns the Polars expressions or a spec's `feature_map=`), feeding
models that exist; no new model state beyond the drawn constants, which must be
saved with the spec so a reload reproduces them. KRLS is a new model with a weak
oracle past its budget: build it last or not at all.

**Hard parts and risks.** (1) Where the map runs: as Polars expressions it costs
nothing new but `D = 1,000` cosines per row is a wide query; inside the core it
needs `cos` (libm, but nothing of it is kept in the state except through the
Gram, which is a sum of libm results: across OSes the Gram then differs by ulps.
The 2026-11-20 rule covers it; state the tolerance). (2) `D` features cost a
`D²/2` Gram: `D = 500` is 1 MB per group. (3) The bandwidth `gamma` wants the
features standardized first: order of operations with `standardize=True`.

**State.** Features: `8·D²/2` (the regression's Gram) plus `D·(k+1)` constants in
the spec. KRLS: `8·(m² + m·k)`.

**Effort.** S (2 days) for the helpers and their oracles; L for KRLS.

## 9. Task 503: online trees and ensembles

**In rheon today.** None in the bank. `docs/BOOSTED-TREES.md` holds measured
prototypes (decision open); `rp.gbdt` builds XGBoost's and LightGBM's training
data (batch). `docs/ENHANCEMENTS.md` §4 excluded trees for unbounded state,
randomness and no clock decay.

**Library surface** [rv].
`tree.HoeffdingTreeRegressor(grace_period=200, max_depth=None, delta=1e-07,
tau=0.05, leaf_prediction='adaptive', leaf_model=None, model_selector_decay=0.95,
nominal_attributes=None, splitter=None, min_samples_split=5, binary_split=False,
max_size=500.0, memory_estimate_period=1000000, stop_mem_management=False,
remove_poor_attrs=False, merit_preprune=True)`
(`tree/hoeffding_tree_regressor.py:15`);
`HoeffdingTreeClassifier(grace_period=200, max_depth=None,
split_criterion='info_gain', delta=1e-07, tau=0.05, leaf_prediction='nba',
nb_threshold=0, nominal_attributes=None, splitter=None, binary_split=False,
min_branch_fraction=0.01, max_share_to_split=0.99, max_size=100.0, …)`
(`hoeffding_tree_classifier.py:13`); the adaptive trees add
`bootstrap_sampling=True, drift_window_threshold=300, drift_detector=None,
switch_significance=0.05, seed=None`; `ExtremelyFastDecisionTreeClassifier` adds
`min_samples_reevaluate=20`; `SGTRegressor(delta=1e-07, grace_period=200,
init_pred=0.0, max_depth=None, lambda_value=0.1, gamma=1.0,
feature_quantizer=None)`; `LASTClassifier`; `ISOUPTreeRegressor`. Splitters:
`EBSTSplitter()`, `TEBSTSplitter(digits=1)`, `QOSplitter(radius=0.25,
allow_multiway_splits=False)`, `HistogramSplitter(n_bins=256, n_splits=32)`,
`GaussianSplitter(n_splits=10)`, `ExhaustiveSplitter()`,
`DynamicQuantizer(radius=0.5, std_prop=0.25)`, `StaticQuantizer(n_bins=64,
warm_start=100)`. Forests: `forest.ARFRegressor(n_models=10,
max_features='sqrt', aggregation_method='median', lambda_value=6, metric=None,
disable_weighted_vote=True, drift_detector=None, warning_detector=None,
grace_period=50, max_depth=None, delta=0.01, tau=0.05,
leaf_prediction='adaptive', …, seed=None)`
(`forest/adaptive_random_forest.py:716`), `ARFClassifier`,
`AMFRegressor(n_estimators=10, step=1.0, use_aggregation=True, max_nodes=None,
seed=None)` (`aggregated_mondrian_forest.py:244`), `AMFClassifier(…,
dirichlet=0.5, split_pure=False)`, `OXTRegressor(…,
resampling_strategy='subbagging', resampling_rate=0.5, detection_mode='all', …)`.
Ensembles: `BaggingRegressor(model, n_models=10, seed=None)`
(`ensemble/bagging.py:100`), `BaggingClassifier`, `ADWINBaggingClassifier`,
`LeveragingBaggingClassifier(model, n_models=10, w=6, adwin_delta=0.002,
bagging_method='bag', seed=None)`, `AdaBoostClassifier`,
`ADWINBoostingClassifier`, `BOLEClassifier(…, error_bound=0.5)`,
`SRPClassifier/SRPRegressor(model, n_models=10, subspace_size=0.6,
training_method='patches', lam=6, drift_detector, warning_detector, …)`,
`EWARegressor(models, loss=None, learning_rate=0.5)` (`ensemble/ewa.py:12`),
`StackingClassifier`, `VotingClassifier`. [sk] has no incremental tree.

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| Hoeffding tree regressor: `grace_period`, `delta`, `tau`, `max_depth`, `min_samples_split`, `binary_split` | [rv] | E given the row order: no randomness; a row updates one leaf; `grace_period` counted in learned weight so a chunking cannot move a split | no: decayed leaf sums are possible (BOOSTED-TREES measured them) but river has no decayed tree, so no oracle | no: a split cannot be undone | [rv] `HoeffdingTreeRegressor` with the same splitter on the same rows: the same splits at the same rows, thresholds 1e-12, predictions 1e-9 at `leaf_prediction='mean'`. River's trees are pure Python: keep oracle runs to a few thousand rows |
| splitters | [rv] | `QOSplitter(radius)`: E, bounded by range/radius bins; `TEBSTSplitter(digits)`: E, bounded by distinct rounded values; `HistogramSplitter(n_bins, n_splits)` and `GaussianSplitter(n_splits)` (classification): E, bounded; **`EBSTSplitter` and `ExhaustiveSplitter`: N (they keep every distinct value: state grows with the stream)** | | | [rv], per splitter |
| leaf prediction: `'mean'`, `'model'`, `'adaptive'` (`leaf_model`, `model_selector_decay`) | [rv] | `'mean'` E; `'model'` E with a linear leaf (river's default leaf is its SGD `LinearRegression`; here a leaf could hold `sgd`'s state); `'adaptive'` E (a faded error per choice) | | | [rv] with `leaf_model` set to the river model `sgd` is already held to |
| Hoeffding tree classifier: `split_criterion` (`'info_gain'`, `'gini'`, `'hellinger'`), `leaf_prediction` (`'mc'`, `'nb'`, `'nba'`), `nb_threshold`, `min_branch_fraction`, `max_share_to_split` | [rv] | E; `ln` in the gain is computed at split attempts, not kept | no | no | [rv], node for node |
| memory management: `max_size`, `memory_estimate_period`, `stop_mem_management`, `remove_poor_attrs`, `merit_preprune` | [rv] | river bounds memory in megabytes of Python objects, which cannot be matched; replace by a node cap and a per-leaf bin cap, and say so | | | the oracle runs with river's management off and the cap not binding |
| Hoeffding adaptive trees (ADWIN in the nodes, alternate subtrees), `bootstrap_sampling`, `seed` | [rv] | E with `bootstrap_sampling=False`; needs task 508's ADWIN | n.a. (ADWIN is the forgetting) | no | [rv], node for node, after 508 |
| extremely fast decision tree (split re-evaluation) | [rv] | E | | | [rv]; later |
| stochastic gradient trees (`SGTRegressor`) | [rv] | E; gradient statistics per bin | | | [rv]; later |
| Mondrian forests (`AMFRegressor`, `AMFClassifier`) | [rv] | A: random splits (seed), exact aggregation over prunings; bounded by `max_nodes` | no | no | [rv]: structure only if river's draws are reproduced (Python's `random.Random`); otherwise an accuracy band |
| online bagging (`BaggingRegressor`: Poisson(1) replicate weights) | [rv] | A: draws. Over this library's own models it needs no tree: the draw is a row weight per replicate | follows the base model | follows | [rv] statistical (the same base model, error within a band); the draws from the shared stream, tested for their distribution |
| leveraging bagging (`w`, `adwin_delta`, `bagging_method`), ADWIN bagging and boosting, AdaBoost, BOLE, streaming random patches | [rv] | A; need 508's ADWIN and the stream | | | [rv] statistical |
| adaptive random forest: `n_models`, `max_features`, `lambda_value`, `aggregation_method`, drift and warning detectors, background trees | [rv] | A: Poisson(6) weights and random feature subsets at every node | n.a. | no | [rv]: an accuracy band on river's datasets; a node-for-node match would need Python's Mersenne Twister and river's draw order reproduced, which is brittle: do not promise it |
| `EWARegressor` (exponentially weighted average of models) | [rv] | **exists**: `emit_averaged` is this, with a different weight rule | | | note the overlap; no build |
| stacking, voting | [rv] | composition across specs: a second-stage spec on first-stage predictions, which the bank can already express with two specs | | | no build |

**Hard parts and risks.** (1) **State.** A regression leaf keeps, per feature, a
table of bins each with a count, a sum and a sum of squares: about
`leaves × k × bins × 24` bytes. 1,000 leaves, 20 features, 64 bins: 31 MB per
group, four orders above any model today. **This must go to the user before any
build**, with caps (nodes, bins) in the spec and their bytes in the docstring.
(2) A static output schema is fine (`pred`), but `coef` has no meaning; the tree
must be readable some other way (a table from the state, as `bank.audit()`).
(3) Rule 5: a tree in msgpack with a frozen fixture for every node form. (4)
Rule 8: `n_eff` is the weight seen, but leaves have their own; `min_weight`
applies to the root. (5) No decay and no window have an oracle, so by the rule
beneath task 510 neither is built: a tree model would be the bank's first with
no forgetting other than a drift detector. Say it plainly in the docs. (6)
BOOSTED-TREES' prototype is a different algorithm (decayed histograms, boosted);
task 503 as written is river's family. Decide which is wanted before building
either.

**State.** As (1). **Effort.** XL: Hoeffding regressor and classifier with two
bounded splitters and the river oracle, 10 to 15 days; each ensemble on top, M.

## 10. Task 504: contextual bandits

**In rheon today.** Per-group regressions with `emit_error_inflation` (the
prediction's variance over the noise, `sqrt(1 + x'(X'X)⁻¹x)`), `emit_se_coef`,
`kalman`'s posterior, and `ModelBank.predict` (scores without learning).
`docs/ENHANCEMENTS.md` §4 excluded bandit model selection for randomness.
`docs/records/BEYOND-O-STATE.md`: "Contextual bandits. Vowpal Wabbit owns this."

**Library surface.** [rv] `bandit.LinUCBDisjoint(alpha=1.0, beta=1.0,
smoothing=None, reward_obj=None, burn_in=0, seed=None)` (`bandit/lin_ucb.py:10`,
one `BayesianLinearRegression` per arm; its docstring: "as of yet it is too slow
to realistically be used in practice"); `ThompsonSampling(reward_obj=None,
burn_in=0, seed=None)`; `UCB(delta, reward_obj=None, reward_scaler=None,
burn_in=0, seed=None)`; `BayesUCB(…)`; `EpsilonGreedy(epsilon, decay=0.0, …)`;
`Exp3(gamma, …)`; `RandomPolicy`; `evaluate_offline(policy, history,
reward_stat=None)`; `model_selection.BanditRegressor`. Only `LinUCBDisjoint`
reads a context. [vw] `--cb_explore K` and `--cb_explore_adf` with `--epsilon`
(default 0.05), `--first tau`, `--bag m`, `--cover m`, `--softmax --lambda`,
`--rnd N` (`--rnd_alpha`, `--rnd_invlambda`), and `--regcb`, `--squarecb` (named
on the page, their options not read today); `--cb_type` `ips`, `dm`, `dr`,
`mtr` (default), `sm`; labels `action:cost:probability`. [mab] `MAB(arms,
learning_policy, neighborhood_policy=None, seed=123456, n_jobs=1, backend=None)`
with `fit`, `partial_fit(decisions, rewards, contexts)`, `predict`,
`predict_expectations`; policies `LinUCB(alpha=1.0, l2_lambda=1.0, scale=False)`,
`LinTS(alpha=1.0, l2_lambda=1.0, scale=False)`, `LinGreedy(epsilon=0.1,
l2_lambda=1.0, scale=False)`, `EpsilonGreedy(epsilon=0.1)`, `Softmax(tau=1)`,
`ThompsonSampling(binarizer=None)`, `UCB1(alpha=1)`, `Popularity`, `Random`.
Not read today (memory): `contextualbandits` (Cortes), `obp` (off-policy
estimators).

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| LinUCB, one ridge per arm: `alpha`, `l2_lambda` | [mab] `LinUCB`; [rv] `LinUCBDisjoint` | E: per arm it is `rls`/`ewridge`, and the bonus is `α·sqrt(x'A⁻¹x)`, which `error_inflation` holds; deterministic (ties by a stated rule) | yes: a decayed Gram per arm ([sk] `Ridge(sample_weight=)` for the mean, numpy for the bonus); note discounted LinUCB in the literature uses a second Gram for the bonus: state which is built | yes: `ewridge`'s window per arm | [mab] `predict_expectations` per arm after `partial_fit` on the same rows: 1e-9; [rv] `LinUCBDisjoint` through the existing `kalman`-river mapping |
| greedy and ε-greedy on a linear model: `epsilon`, `decay` | [mab] `LinGreedy`; [rv] `EpsilonGreedy`; [vw] `--epsilon` | greedy E; ε A (a draw) | yes | yes | expectations as above; the exploration rate statistical |
| linear Thompson sampling: `alpha`, `l2_lambda` | [mab] `LinTS` | A: one multivariate normal draw per arm per row from the stream | yes | yes | the posterior mean and covariance exact (as LinUCB); the draw: hand the test's standard normals in and compare `β̂ + α·L·z` with numpy, 1e-10; the policy's choice frequencies statistical |
| softmax / Boltzmann: `tau`, `--lambda` | [mab] `Softmax`; [vw] `--softmax` | probabilities E (`exp` at emit time), the draw A | yes | yes | scipy `softmax` on the scores, 1e-12 |
| explore-first, bagging, cover, RND, RegCB, SquareCB | [vw] | bagging A (online bagging, task 503's piece); the rest are tied to VW's reduction stack | | | [vw] cannot be matched number for number (its learners are hashed SGD); a regret band on a simulated problem is all it gives |
| non-contextual policies: UCB1, Bayes-UCB, Thompson (Beta, Gaussian), Exp3 | [rv]; [mab] | E for the indexes (running means and counts per arm: `ew_cov` per group); draws A | yes (discounted UCB) | yes (sliding-window UCB) | [rv] index for index, 1e-12, with river's `burn_in` and tie rule matched |
| action-dependent features (one feature row per action) | [vw] `--cb_explore_adf` | E for a shared-coefficient LinUCB ("hybrid"); needs several candidate rows per decision: an input shape the bank does not have | | | definition (Li et al. 2010); later |
| propensities out; off-policy value: IPS, DM, DR, SNIPS; replay evaluation | [vw] `--cb_type`; [rv] `evaluate_offline` | E: running weighted means of logged rewards over the logged propensity; each is a ratio of sums | yes | yes | definitions; [rv] `evaluate_offline` for replay; `obp` (memory) as the library for IPS/DR, to verify |

**Verdict.** No new estimator: a **policy layer** over per-arm regressions that
exist. What is new is the row protocol: score every arm for the row's context
before anything is learned (rule 2 gives this for free), emit the chosen action,
its score and its probability, then learn the *logged* arm's reward. It needs an
`action` column, a `reward` column and optionally a logged `propensity`.

**Hard parts and risks.** (1) A stream cannot act: on logged data the policy's
choice and the logged action differ, and what is learned must be said (always the
logged row; replay evaluation counts only agreements). (2) Randomized policies put
a draw on the output path, which ENHANCEMENTS rejected once; with the
counter-based stream it is reproducible and chunk-invariant, but it is still the
user's call. (3) Arms as declared classes (a static schema), as `ew_class`.
(4) State is `arms ×` the base model's.

**State.** `8·arms·(k+1)(k+2)/2`. **Effort.** M (LinUCB, greedy, the indexes, the
off-policy sums) to L (with sampling policies and the stream).

## 11. Task 505: streaming PCA

**In rheon today.** `ew_cov(pca=r, pca_every=, max_rows_between_pcas=)`: the top
`r` eigenpairs of the decayed (or windowed) covariance, refreshed on a cadence of
learned rows, scores per row on the frozen components, a fixed sign rule; held to
numpy `eigh` at every refresh. Exact, with decay and window, in `O(k²)` state and
`O(k³)` per refresh.

**Library surface.** [sk] `IncrementalPCA(n_components=None, whiten=False,
copy=True, batch_size=None)` with `partial_fit(X)`
(`decomposition/_incremental_pca.py:19, 254`; each batch must hold at least
`n_components` rows, :310); `PCA(n_components=None, whiten=False,
svd_solver='auto', tol=0.0, iterated_power='auto', n_oversamples=10,
power_iteration_normalizer='auto', random_state=None)` (`_pca.py:113`);
`TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, …)`;
`MiniBatchSparsePCA`, `MiniBatchDictionaryLearning`, `MiniBatchNMF(…,
forget_factor=0.7, …)`, `FactorAnalysis`, `FastICA`, `KernelPCA`. [sm]
`multivariate.pca.PCA(data, ncomp=None, standardize=True, demean=True,
normalize=True, gls=False, weights=None, method='svd', missing=None, …)`. [rv]
none.

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| exact PCA of the running covariance | [sk] `PCA`; numpy | E; **exists** (`ew_cov(pca=)`) | yes, exists | yes, exists | exists |
| incremental SVD, block by block (Ross et al.), `n_components`, block size | [sk] `IncrementalPCA.partial_fit` | A: exact only when `n_components = k`; a truncated basis loses what it dropped. State `O(r·k)`. Blocks must be cut on learned-row counts, not chunks (rule 3) | no: scikit-learn has no forgetting factor (the paper has); under decay it is checkable only at `r = k`, where `ew_cov` is already exact | no: an SVD cannot drop rows | [sk] `partial_fit` on the same blocks: components up to sign 1e-9, `explained_variance_`, `singular_values_`, `mean_`, `var_` 1e-9 |
| `whiten` | [sk] | E at emit time | | | [sk] `transform` |
| explained variance and ratio, noise variance | [sk] | E from the kept singular values and the total variance (one more running sum) | | | [sk] |
| scores per row, reconstruction error | [sk] `transform`, `inverse_transform` | E, out of sample (projected on the basis before the row) | | | [sk] on the basis of the previous block |
| Oja's rule, generalized Hebbian (Sanger): learning-rate schedule | none | A: a stochastic approximation; converges to the leading subspace at a rate set by the eigengap | a constant rate is a forgetting; no oracle | no | definition; convergence to [sk] `PCA`'s subspace by principal angles, statistical, on separated spectra only (the lesson of `hmm`'s online EM, ORACLE-COVERAGE fourth pass) |
| CCIPCA (Weng, Zhang and Hwang 2003): amnesic parameter | none in Python (R `onlinePCA::ccipca`, memory) | A | the amnesic parameter is its forgetting; no oracle | no | definition; subspace angle, statistical |
| randomized, ARPACK solvers; `TruncatedSVD` | [sk] | batch solvers, not streaming forms | | | n.a. |
| sparse PCA, dictionary learning, NMF (mini-batch), ICA, factor analysis, kernel PCA | [sk] | mini-batch forms are A with shuffling and random initial values; not asked by 505 | | | not built |
| weighted or GLS PCA, missing values by EM | [sm] `pca.PCA(weights=, gls=, missing=)` | row weights E through `ew_cov` (exists); EM for missing values N | | | [sm] for the weighted case, already numpy's |

**Verdict.** For any `k` whose `k²/2` Gram is affordable, **task 505 is already
done** by `ew_cov(pca=)`, exactly and with both forgetting forms. What is missing
is the `O(r·k)`-state form for inputs too wide for a Gram. Build the block
incremental SVD (the one form with a library oracle) as that option; build Oja
and CCIPCA only if a statistical oracle is accepted for them.

**Hard parts and risks.** (1) The crossover: at k = 2,000 a Gram is 16 MB per
group and the incremental form (r = 10) 160 kB; below a few hundred columns the
exact form wins on every count. State the crossover measured, not guessed. (2)
`IncrementalPCA`'s mean correction row and its `n_samples ≥ n_components` rule
per block. (3) Signs: reuse `ew_cov`'s rule. (4) No decay and no window pass the
checkable rule for the truncated forms.

**State.** `8·(r·k + r + 2k)`. **Effort.** M (3 to 4 days) for the block form.

## 12. Task 506: Gaussian mixtures by online EM

**In rheon today.** `hmm(n_components, precision_prior, covariance_type
("full", "tied", "diag"), learn, transition_prior, transition, means, covs,
warm_rows, init, seed, …)`: a Gaussian hidden Markov model filtered per row, its
means, covariances and transitions learned by an online EM that attributes each
row by its filtered posterior; held to hmmlearn at fixed parameters. `kmeans`,
`micro`, `ew_class` (the same Gaussians with the label given).

**Library surface.** [sk] `GaussianMixture(n_components=1,
covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,
init_params='kmeans', weights_init=None, means_init=None, precisions_init=None,
random_state=None, warm_start=False, verbose=0, verbose_interval=10)`
(`mixture/_gaussian_mixture.py:556`; `covariance_type` `'full'`, `'tied'`,
`'diag'`, `'spherical'`; `init_params` `'kmeans'`, `'k-means++'`, `'random'`,
`'random_from_data'`); methods `predict_proba`, `score_samples`, `score`, `aic`,
`bic`, `sample`. `BayesianGaussianMixture(n_components=1, covariance_type='full',
…, weight_concentration_prior_type='dirichlet_process',
weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None,
degrees_of_freedom_prior=None, covariance_prior=None, …)`
(`_bayesian_mixture.py:74`). Neither has `partial_fit`. [rv] none (its `cluster`
module is `CluStream`, `DBSTREAM`, `DenStream`, `KMeans`, `ODAC`, `STREAMKMeans`,
`TextClust`).

| feature / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| posteriors and log-density at given weights, means, covariances | [sk] `predict_proba`, `score_samples` | E | n.a. | n.a. | [sk] with `weights_`, `means_`, `precisions_cholesky_` set: 1e-10 |
| online EM of weights, means, covariances (Cappé and Moulines 2009) | [sk] batch EM | A: each row is attributed once, at the parameters before it; the fixed point is batch EM's, reached slowly on overlapping components | yes by composite: the sums are decayed, responsibility-weighted moments; given the responsibilities (scipy `multivariate_normal.logpdf`), numpy's weighted mean and `cov(aweights=)` reproduce them per row | yes by the same composite (a ring of the sums, as `ew_class`'s windows); scikit-learn fitted on the window is a different quantity (it re-attributes) | composite: 1e-10 per row. Batch: [sk] `GaussianMixture(...).fit` on the same rows, on separated components, parameters within a stated band after a stated count (ledger) |
| `covariance_type`: full, tied, diag | [sk] | exists in `hmm`, `ew_class` | | | as above |
| `covariance_type='spherical'` | [sk] | E; task 522 adds it to `hmm` | | | [sk] |
| `reg_covar` | [sk] | `precision_prior` plays this part; under decay a fading prior and a fixed floor differ (task 525 raises the same for `ew_class`) | | | [sk] at no decay |
| initial values: `init_params`, `*_init`, `n_init`, `random_state`, `warm_start` | [sk] | `warm_rows`, `init`, `seed`, `means=`, `covs=` exist on `hmm`; `n_init` (several restarts) N in one pass | | | [sk] `kmeans_plusplus` already holds the seeding |
| `aic`, `bic`, per-row log-likelihood | [sk] | E as running out-of-sample sums (`ln` at emit) | yes | yes | [sk] `score_samples` |
| the number of components chosen from the data | [sk] `BayesianGaussianMixture` (variational, Dirichlet process) | N in one pass as the library does it; `kmeans`'s split-merge is this library's answer | | | not built |
| variational priors (`mean_prior`, `covariance_prior`, `degrees_of_freedom_prior`, weight concentration) | [sk] | a MAP form with fixed priors is E to add to the sums; the variational fit N | | | [sk] cannot check a MAP online EM: definition only |

**Verdict.** A Gaussian mixture is `hmm` with every row of the transition matrix
equal to the mixing weights. **Extend `hmm`** (a `transition="iid"` form, or a
thin `gmm` spec over the same core) rather than write a second model.

**Hard parts and risks.** Label switching and slow convergence on overlapping
components, both already documented for `hmm` (ORACLE-COVERAGE fourth pass,
finding 2): test on separated components and say why. A component whose weight
decays to nothing needs `hmm`'s existing rule.

**State.** `8·K·(k(k+1)/2 + k + 1)` (full), `hmm`'s without the transition
counts. **Effort.** S to M (2 to 4 days).

## 13. Task 507: streaming anomaly detectors

**In rheon today, against river's nine.**

| river `anomaly` | in rheon |
|---|---|
| `GaussianScorer(window_size=None, grace_period=100)` | `emit_zscore` (E21) |
| `PredictiveAnomalyDetection(predictive_model, horizon=1, n_std=3.0, warmup_period=0)` | `emit_zscore` against a threshold; adaptive `conformal` intervals |
| `StandardAbsoluteDeviation(sub_stat)` | `audit`'s `robust_z`; `resid_quantiles` |
| `QuantileFilter(anomaly_detector, q)`, `ThresholdFilter(anomaly_detector, threshold)` | `mahal_quantiles`, `resid_quantiles` |
| `OneClassSVM(nu=0.1, optimizer, intercept_lr=0.01, clip_gradient=1e12)` | not present |
| `LocalOutlierFactor(n_neighbors=10)` | not present (memory grows with its window) |
| `HalfSpaceTrees(n_trees=10, height=8, window_size=250, limits=None, seed=None)` | not present |
| `LODA(n_bins=10, n_random_cuts=100, seed=None)` | not present |

Also present: `ew_cov(stats=["mahal"])` (the Gaussian distance, with PCA scores
for a reconstruction error), `micro`'s outlier flag, `bocpd`.

**Other libraries.** [pyod] 50 batch detectors (ECOD, COPOD, HBOS, IForest,
LODA, MCD, PCA, kNN, LOF, OCSVM, GMM, KDE, MAD, … and neural ones); the API is
`fit`, `decision_function`, `predict`: nothing incremental. [rrcf] `RCTree(X,
index_labels=…)`, `insert_point(x, index)`, `forget_point(index)`,
`codisp(leaf)`, `codisp_with_cut_dimension`, `rrcf.shingle(data, size)`;
examples use 40 to 100 trees of 256 points; seeding is by `np.random.seed` in
the examples. [sk] `IsolationForest(n_estimators=100, max_samples='auto',
contamination='auto', max_features=1.0, bootstrap=False, …)` (batch),
`SGDOneClassSVM(nu=0.5, …)` (has `partial_fit`), `EllipticEnvelope`,
`LocalOutlierFactor`, `OneClassSVM`.

| detector / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| half-space trees: `n_trees`, `height`, `window_size`, `limits`, `seed` | [rv] (`anomaly/hst.py:52`) | E given the trees: they are drawn once from the seed and the feature ranges, then each row adds mass along one path per tree; the reference and latest masses swap every `window_size` learned rows. State is fixed at build | no (river has none: no oracle) | the detector's own two tumbling windows, exact against river; a clock window has no oracle | [rv]: build river's trees, read each node's feature and split value, hand them to the model (a test-only constructor), then scores 1e-12 on every row. The seeded construction is this library's own |
| the feature ranges (`limits`; river assumes [0, 1]) | [rv]; its `MinMaxScaler` | a declared range, or one read from `warm_rows`; a running min-max would move the trees: N | | | with `limits=` given to both |
| LODA: `n_bins`, `n_random_cuts`, `seed` | [rv]; [pyod] `LODA` (batch) | E given the projections (sparse random vectors drawn once) and fixed histogram edges; `ln` at emit | decayed bin counts are natural; no oracle | no | [rv] with its projections handed over: 1e-10 |
| robust random cut forest: `num_trees`, `tree_size`, shingle size | [rrcf] | A: every insertion draws a cut; state `num_trees × tree_size` points, bounded; forgetting is first-in-first-out by rows | no | its own (the last `tree_size` points) | **no exact oracle**: rrcf draws from numpy's global generator inside its insertion, so its trees cannot be reproduced from here. What can be exact: `codisp` on a tree built by rrcf and handed over. The insertion rule is then the definition (Guha et al. 2016), with a rank agreement against rrcf's scores as the statistical check |
| isolation forest | [sk], [pyod] | N: batch subsampling; HST is its streaming form | | | n.a. |
| one-class SVM by SGD: `nu`, optimizer | [rv] `OneClassSVM`; [sk] `SGDOneClassSVM.partial_fit` | E: it is `sgd` with a hinge-type loss and an offset | as `sgd` | no | [sk] `SGDOneClassSVM.partial_fit` row by row, as `TestSgdIsScikitLearnsSgd`: 1e-12. Linear only (a kernel needs task 502's features) |
| local outlier factor, kNN, COF, ABOD | [rv] LOF; [pyod]; [sk] | N: neighbours need the rows; windowed it is `O(W)` memory and `O(W)` work per row | | on a row ring only | [sk] `LocalOutlierFactor` on the window; propose not to build |
| ECOD, COPOD (empirical tail probabilities per feature) | [pyod] | A: the empirical CDFs come from the quantile sketch the library has (`EwQuantile`), so the score carries the sketch's rank error | yes (the sketch decays) | yes | [pyod] `ECOD` on the prefix within the sketch's stated rank error: statistical |
| HBOS (histogram score) | [pyod] | E with fixed edges (`marginal`'s bins hold the counts) | yes | yes | [pyod] `HBOS` with the same edges, if its edges can be set (to verify) |
| MCD and other robust-covariance scores | [pyod] `MCD`; [sk] `EllipticEnvelope` | N (a combinatorial search over subsets); `ew_cov`'s `mahal` is the non-robust form | | | n.a. |
| Gaussian-mixture, PCA scores | [pyod] `GMM`, `PCA` | task 506's log-density; `ew_cov(pca=)`'s residual | | | those tasks |
| neural detectors | [pyod] | out of scope | | | not built |

**Hard parts and risks.** (1) HST's ranges must be fixed before the first row;
a feature that leaves its range falls in an edge cell. Standardize first (the
running scale `kmeans` uses) or declare limits. (2) The random cut forest's only
exact check is a scoring function; by the rule beneath task 510 its build waits
on the user's reading of "no oracle" (as 0.2 item 6 for bipower). (3) Scores are
out of sample (the row is scored before it adds its mass): river does the same.

**State.** HST: `8·n_trees·(2^(h+1) − 1)·2` plus the splits: 82 kB at 10 trees of
height 8. RRCF: `8·num_trees·tree_size·(d + overhead)`: about 1 MB at 40 × 256 ×
d = 8. **Effort.** HST M (3 to 4 days); one-class SGD S; RRCF L.

## 14. Task 508: CUSUM, Page-Hinkley, ADWIN and the others

**In rheon today.** `emit_drift` (`drift.rs`): Page-Hinkley, upward only, on each
slot's `|resid|/sigma`, its mean decayed on the model's clock and its excess
integrated over the clock, with `drift_delta`, `drift_threshold`,
`drift_action="reset"`. `emit_breaks`: CUSUM and CUSUM of squares of recursive
residuals and a fast-slow Wald distance. `seqtest`: an anytime-valid sign test.
`bocpd`, `corrchange`, `emit_feature_health`. ENHANCEMENTS §4 excluded ADWIN
("adaptive-window") and KSWIN (window memory); tasks 521 and 525 already carry an
anytime-valid mean test and a window two-sample test.

**Library surface.** [rv] `drift.ADWIN(delta=0.002, clock=32, max_buckets=5,
min_window_length=5, grace_period=10)` (`drift/adwin.py:7`; properties `width`,
`estimation`, `variance`, `n_detections`); `PageHinkley(min_instances=30,
delta=0.005, threshold=50.0, alpha=0.9999, mode='both')`
(`page_hinkley.py:7`; `mode` `'up'`, `'down'`, `'both'`); `KSWIN(alpha=0.005,
window_size=100, stat_size=30, seed=None, window=None)` (`kswin.py:14`);
`binary.DDM(warm_start=30, warning_threshold=2.0, drift_threshold=3.0)`
(`binary/ddm.py:8`), `EDDM(warm_start=30, alpha=0.95, beta=0.9)`,
`FHDDM(sliding_window_size=100, confidence_level=1e-06,
short_window_size=None)`, `HDDMA(drift_confidence=0.001, warning_confidence=0.005,
two_sided_test=False)`, `HDDMW(drift_confidence=0.001, warning_confidence=0.005,
lambda_val=0.05, two_sided_test=False)` (so named in 0.26.1; river's docs write
`HDDM_A`, `HDDM_W`); `DriftRetrainingClassifier`;
`stats.KolmogorovSmirnov(statistic='ks')` (an incremental two-sample statistic).
[rup] offline search methods `Dynp`, `Pelt`, `Binseg`, `BottomUp`, `Window`,
`KernelCPD` over costs `CostL1`, `CostL2`, `CostNormal`, `CostRbf`,
`CostCosine`, `CostLinear`, `CostCLinear`, `CostRank`, `CostMl`, `CostAR`; each
takes the finished signal. [sm] `breaks_cusumolsresid(resid, ddof=0)`
(`stats/diagnostic.py:2108`), `recursive_olsresiduals(res, skip=None,
lamda=0.0, alpha=0.95, order_by=None)`, `breaks_hansen`. [sp] `ks_2samp`,
`cramervonmises_2samp`.

| detector / parameter | library | online | already in rheon? | decay | window | oracle and tolerance |
|---|---|---|---|---|---|---|
| Page-Hinkley, upward | [rv] `PageHinkley(mode='up')` | E | **yes**, `emit_drift` | exists (its own mean rule; river's `alpha` fades the sum instead, ORACLE-COVERAGE §5 item 1) | n.a. | exists |
| Page-Hinkley `mode='down'`, `'both'`; `min_instances`; river's `alpha` | [rv] | E | no: add `drift_mode` on the signed standardized residual, and `drift_min_rows` | as today | n.a. | [rv] flag for flag at `alpha=1` and unit steps; river's `alpha < 1` as a second form only if wanted |
| a detector on a raw column (not a model's residual) | [rv] any detector on any stream | E | no: `emit_drift` needs a model. A level-only `holt` gives one today, clumsily. Add a thin spec over `drift.rs` | | | the same oracles |
| tabular CUSUM (Page 1954): reference `k`, decision interval `h`, two-sided, known target mean and scale, restart after a signal | none in [rv] | E | no: it is Page-Hinkley with a fixed target in place of the running mean. `drift_kind="cusum"` with `drift_target` | n.a. (a fixed target) | n.a. | the recursion `C⁺ = max(0, C⁺ + z − k)` by definition, plus **published average run lengths** (two-sided, `k = 0.5`: ARL₀ about 168 at `h = 4`, 465 at `h = 5`; Montgomery's table, memory: verify the figures) simulated to within Monte Carlo error |
| CUSUM of recursive residuals, CUSUM of squares | [sm] `breaks_cusumolsresid`, `recursive_olsresiduals` | E | **yes**, `emit_breaks` | exists | exists (moving sum) | exists |
| ADWIN: `delta`, `clock`, `max_buckets`, `min_window_length`, `grace_period`; outputs width, mean, variance | [rv] | E: an exponential histogram of buckets; memory `max_buckets · log2(width)` pairs, so it grows with the logarithm of the stream and is capped by capping the width | no | n.a.: ADWIN *is* an adaptive window; a clock decay has no meaning in it | n.a. | [rv] `ADWIN` on the same values: `drift_detected`, `width`, `estimation` on every row, exact (river's is compiled, so long streams are cheap). `clock=32` counts rows: count learned rows so a chunking cannot move a check |
| ADWIN as the model's memory (shrink the fit to ADWIN's window) | [rv] ADWIN bagging, adaptive trees | A for a model with a snapshot ring: drop to the snapshot nearest the cut | | | | no oracle; `drift_action="reset"` on ADWIN's flag is the checkable form |
| DDM, EDDM (error rate and error distance of a binary stream) | [rv] | E, `O(1)` state | no: options for binary targets (`ftrl`, `sgd` logistic), fed the hit stream | n.a. | n.a. | [rv] flag and warning for flag, exact |
| `HDDMA`, `HDDMW` (Hoeffding bounds; W is exponentially weighted): `drift_confidence`, `warning_confidence`, `lambda_val`, `two_sided_test` | [rv] | E, `O(1)` | no | `HDDMW`'s `lambda_val` is its own weight | n.a. | [rv], exact |
| FHDDM: `sliding_window_size`, `confidence_level` | [rv] | E with a ring of `window` bits | no | n.a. | its own | [rv], exact |
| KSWIN: `alpha`, `window_size`, `stat_size`, `seed` | [rv] | A: it draws a random sample of the older part; state is the window | no; task 525 has it | n.a. | its own | deterministic form (the newest `stat_size` rows against the rest): [sp] `ks_2samp` on the two parts, 1e-12. River's seeded draw: statistical |
| incremental two-sample KS | [rv] `stats.KolmogorovSmirnov` | E over a ring (a treap in river) | no | | its own | [sp] `ks_2samp`, 1e-12 |
| Shewhart and EWMA control charts | none here | E | **yes** in effect: `emit_zscore` against a limit; an EW mean's z-score | | | exists |
| offline segmentation (PELT, binary segmentation, dynamic programming, kernel) | [rup] | N: each needs the whole signal. Their use is as a referee: on a finished stream, each online flag should fall within a stated distance after a breakpoint ruptures finds, and not elsewhere | | | | [rup] `Pelt(model="l2")` / `Dynp` breakpoints as the known truth for detection delay and false alarms: statistical, as `bocpd`'s TCPD floors are |
| Bayesian online changepoint | (`bayesian_changepoint_detection`) | E | **yes**, `bocpd` | | | exists |

**Verdict.** **No new model.** Every item is an option of `drift` (kind, mode,
target, binary kinds), reached from `emit_drift` and from one thin column-level
spec; ADWIN's bucket list is the only new structure, and CUSUM of residuals
exists as `emit_breaks`.

**Hard parts and risks.** (1) ADWIN's bound assumes values in a known range;
river's uses the window variance. Match river's formula exactly, including its
check cadence and `min_window_length`. (2) ADWIN's state is a list whose length
depends on the data: a state fixture must freeze its forms (rule 5), and its
maximum bytes must be stated: `max_buckets·(log2(cap) + 1)·2` doubles, 3.3 kB at
a cap of 2⁴⁰ rows. (3) Row weights: ADWIN and the binary detectors count rows;
a weight other than 0 or 1 has no library meaning. Treat any positive weight as
one row, as `seqtest` does. (4) The clock: these detectors are row-counted;
`emit_drift`'s clock integration is this library's own and stays the default.

**State.** `O(1)` each, ADWIN as above, the window tests their ring.
**Effort.** S (modes, CUSUM, DDM family) to M (with ADWIN and the column spec).

## 15. Task 509: count-min and HyperLogLog sketches

**In rheon today.** `audit(distinct_cap=256)`: per column, exact counters up to
the cap and a Misra-Gries summary past it (`top_value`, `top_count`,
`second_count`, each within `n/(distinct_cap + 1)`), and a digest for the median
and MAD. `EwQuantile` (a decaying DDSketch) behind the quantile outputs. A fixed
hasher (`hasher.rs`) for group keys.

**Library surface.** [ds] `HyperLogLog(p=8, reg=None, hashfunc=sha1_hash32,
hashobj=None)` with `update(b)`, `count()`, `merge(other)`, `digest()`;
`HyperLogLogPlusPlus(p=8, reg=None, hashfunc=sha1_hash64, hashobj=None)`;
`MinHash(num_perm=128, seed=1, …)`, `LeanMinHash`, `bBitMinHash`,
`WeightedMinHash`, the LSH indexes, `HNSW`. **No count-min.** [ads]
`hll_sketch(lg_k, tgt_type=HLL_8, start_max_size=False)` (`lg_k` 7 to 21;
`update`, `get_estimate`, `get_upper_bound(num_std_devs)`,
`get_lower_bound`), `hll_union(lg_max_k)`, `cpc_sketch`, theta and tuple
sketches; `count_min_sketch` (`num_hashes`, `num_buckets`, `seed`;
`update(item, weight=1.0)`, `get_estimate`, `get_upper_bound`,
`get_lower_bound`, `merge`, `suggest_num_buckets(relative_error)`,
`suggest_num_hashes(confidence)`); `frequent_items_sketch(lg_max_k)` and
`frequent_strings_sketch` ("a variant of the Misra-Gries algorithm";
`get_frequent_items(err_type, threshold=0)` with `NO_FALSE_POSITIVES` or
`NO_FALSE_NEGATIVES`; `(UB − LB) ≤ W·3.5/M`); KLL, REQ, t-digest quantile
sketches; VarOpt and EBPPS sampling. [rv] `sketch.Counter(epsilon=0.1,
delta=0.05, seed=None)` (count-min, `sketch/counter.py:12`),
`HeavyHitters(support=0.001, epsilon=0.005, fading_factor=0.999)` (lossy
counting with forgetting, `heavy_hitters.py:10`), `NUnique(error_rate=0.01,
seed=None)` (HyperLogLog, `n_unique.py:11`), `Histogram(max_bins=256)`,
`Set(capacity=2048, fp_rate=0.01, seed=None)` (a Bloom filter).

| sketch / parameter | library | online | already in rheon? | decay | window | oracle and tolerance |
|---|---|---|---|---|---|---|
| heavy hitters by counters (Misra-Gries / space-saving): capacity; bounds per item; the list above a threshold | [ads] `frequent_items_sketch`; [rv] `HeavyHitters` | A within `W/(cap+1)` | **yes**: `audit`'s counters. Missing: the top-`n` table with lower and upper bounds, and row weights | [rv] `fading_factor`; decayed counters are simple, and the exact decayed counts (a Polars group-by of `λ^age`) are the truth to check against | by ring of summaries: N (Misra-Gries summaries do not subtract) | truth from Polars `value_counts` within the stated bound (exact below the cap); [ads] `get_lower_bound`/`get_upper_bound` bracket the same truth |
| distinct count by HyperLogLog: precision `p` (registers `2^p`) | [ds] `HyperLogLog(p)`; [ads] `hll_sketch(lg_k)`; [rv] `NUnique(error_rate)` | A: relative standard error about `1.04/√(2^p)`; mergeable (task 528's shard merge gets it free) | no: `audit`'s distinct count is exact only up to the cap | no: a register is a maximum and cannot fade | sliding-window HyperLogLog exists in the literature, with no library: no oracle | **[ds] `HyperLogLog(p, hashfunc=h)` with this library's hash handed in: the register array (`reg`) equal entry for entry and `count()` 1e-12.** The estimate against Polars `n_unique`, within 3 standard errors |
| HyperLogLog++ (64-bit hash, bias correction, sparse form) | [ds] `HyperLogLogPlusPlus` | A, better at small counts; the bias tables are data to embed | no | no | no | [ds] with the hash handed in. Below the cap `audit` is exact already, so plain HyperLogLog past the cap suffices: propose not to build `++` |
| CPC, theta, tuple sketches | [ads] | A | no | | | not asked |
| count-min: width `w = ⌈e/ε⌉`, depth `d = ⌈ln(1/δ)⌉`, seed; point queries; weights | [ads] `count_min_sketch`; [rv] `Counter(epsilon, delta, seed)` | A: never below the truth, above it by at most `ε·W` with probability `1 − δ` | no | yes: scale every cell lazily, and the truth is the exact decayed count per key | yes: the sketch is linear, so a window is a difference of two snapshots, exactly; costs a ring of `w·d` tables | the hash cannot be handed to [ads] or [rv], so no cell-for-cell match: truth from Polars within the bound on every queried key, with [ads]' `get_upper_bound` as a second bracket. Conservative update is a variant to state or refuse |
| the row's key count so far, as a per-row output (a frequency encoding of a category, read before the row) | none as such | A (count-min) or A (counters) | no. **This is the online use**: a feature for the models, out of sample by rule 2 | yes | yes | truth: Polars `cum_count().over(key) − 1`, within the bound |
| Bloom filter (seen before?) | [rv] `Set` | A | no | no | no | truth from Polars `is_first_distinct`; not asked |
| quantile sketches: KLL, REQ, t-digest; streaming histogram | [ads]; [rv] `Histogram` | A | **yes** in kind: `EwQuantile` (DDSketch) | exists | | exists; [ads] KLL would be a second opinion for task 529's mergeable sketches |
| MinHash, LSH | [ds] | set similarity: no use here | | | | not built |
| reservoir and VarOpt sampling | [ads] | A: draws | no | | | not asked |

**Verdict.** Heavy hitters: **extend `audit`** (it has the summary; expose the
table and bounds). Distinct counts: **extend `audit`** with HyperLogLog past the
cap. Count-min: a new, small accumulator, worth building only for the per-row
key count.

**Hard parts and risks.** (1) Hashing values: floats by bit pattern after
canonicalizing `−0.0` and NaN; strings by bytes; the hash must be the same on
every platform and frozen for ever, since registers live in saved states (rule
5). `hasher.rs`'s function was chosen for table speed, not for sketch quality:
check its avalanche on sequential integers before using it for HyperLogLog, or
add a second, stronger one. (2) To hand the hash to datasketch the test needs it
from Python: expose it for tests, or reimplement it in a few lines of Python.
(3) `audit` refuses `half_life` and `weight` today: decayed counts would be its
first. Keep sketches undecayed inside `audit` and put decay only in the new
count-min accumulator. (4) Memory per column per group: HyperLogLog `2^p` bytes
(4 kB at p = 12); count-min `8·w·d` (22 kB at ε = 0.005, δ = 0.01; 109 kB at ε = 0.001).

**State.** As (4). **Effort.** `audit` table and bounds S; HyperLogLog S to M;
count-min with the per-row count M.

## 16. Task 510: nonlinear dependence

**In rheon today.** `marginal(bins=, bin_edges=, bin_rule="fixed"|"quantile",
bin_warm_rows=)`: per feature and target, the target's moments in each feature
bin (a response curve and `split_gain`), with decay and windows; the *joint*
count table over a binned target is not kept. `ew_cov` and `marginal`: Pearson
only. `EwQuantile` gives a decayed quantile, so a decayed rank of a new value.
ENHANCEMENTS declined "a dependence measure other than Pearson"; the user
reversed it.

**Library surface.** [sp] `spearmanr(a, b=None, axis=0,
nan_policy='propagate', alternative='two-sided')` (`_stats_py.py:5203`);
`kendalltau(x, y, *, nan_policy='propagate', method='auto', variant='b',
alternative='two-sided')` (:5574; `variant` `'b'` or `'c'`);
`weightedtau(x, y, rank=True, weigher=None, additive=True)`; `somersd(x, y=None,
alternative='two-sided')`; `chatterjeexi(x, y, *, axis=0, y_continuous=False,
method='asymptotic')`; `multiscale_graphcorr(x, y, compute_distance=…,
reps=1000, workers=1, is_twosamp=False, random_state=None)`; `entropy(pk,
qk=None, base=None)`; `differential_entropy(values, *, window_length=None,
base=None, method='auto')`; `contingency.association(observed, method='cramer',
correction=False, lambda_=None)`; `chi2_contingency(observed, correction=True,
lambda_=None)`; `rankdata(a, method='average')`. [sk]
`metrics.mutual_info_score(labels_true, labels_pred, *, contingency=None)`,
`normalized_mutual_info_score(…, average_method='arithmetic')`,
`adjusted_mutual_info_score`; `feature_selection.mutual_info_regression(X, y, *,
discrete_features='auto', n_neighbors=3, copy=True, random_state=None)` and
`mutual_info_classif` (nearest-neighbour estimators with added noise). [dcor]
`distance_covariance`, `distance_correlation`, their `_sqr` forms,
`u_distance_correlation_sqr` (bias-corrected), `distance_stats`,
`distance_correlation_af_inv`, `partial_distance_correlation(x, y, z)`,
`energy_distance`, `independence.distance_covariance_test`,
`independence.distance_correlation_t_test`, `homogeneity.energy_test`,
`rowwise`; a `DistanceCovarianceMethod` class selects the algorithm (its members
not read today; a fast one-dimensional method exists, memory). [rv]
`stats.Entropy(fading_factor=1, eps=1e-08)`.

| measure / parameter | library | online | decay | window | oracle and tolerance |
|---|---|---|---|---|---|
| mutual information from a joint histogram (plug-in) | [sk] `mutual_info_score(contingency=)` | E with fixed or warm-learned edges: one count per cell, `bx·by` per pair; `ln` at read time, never kept | yes: decayed counts. **scikit-learn cannot check it**: `mutual_info_score` casts the table's margins to `int64`, so a non-integer table gives a wrong number (run today: 0.098 against 0.059 on one table). The decayed oracle is [sp] `entropy` on the weighted margins and joint, `H(X) + H(Y) − H(X, Y)`, which normalizes any non-negative weights (the same table: 0.058891016495, equal to the definition to 1e-15) | yes: `marginal`'s ring of count tables (unit weights in a window keep the cells integers, so [sk] serves) | [sk] 1e-12 on integer counts (run today: equal to scipy's to 1e-15); [sp] `entropy` 1e-12 under decay or row weights |
| normalized and adjusted mutual information; `average_method` | [sk] | E at read time; the adjusted form's expected value needs integer counts (no meaning under decay) | normalized yes; adjusted no | yes | [sk] |
| bias corrections: Miller-Madow; others (Chao-Shen, shrinkage) | none in [sk]/[sp] | E at read time from the counts | by hand | yes | Miller-Madow is `(cells − 1)/(2n)` on scikit-learn's value: composite. The others have no Python oracle here: not built |
| entropy of one binned column | [sp] `entropy`; [rv] `Entropy(fading_factor)` | E | yes ([rv] fades) | yes | [sp] on the counts, 1e-12 |
| chi-square, Cramér's V, Tschuprow, contingency coefficient | [sp] `chi2_contingency`, `contingency.association` | E from the same table | by hand (a test needs counts) | yes | [sp], 1e-12 |
| quantile (equal-frequency) bins, i.e. the copula's mutual information | edges as `marginal(bin_rule="quantile")` | E after the warm rows fix the edges; edges that track a moving distribution would move rows between cells: N | yes | yes | [sk] on numpy's `digitize` with the same edges |
| nearest-neighbour mutual information (Kraskov) | [sk] `mutual_info_regression` | N: needs the rows and neighbour searches; scikit-learn adds random noise, so even on a window it is not reproducible to the digit | | | not built |
| differential entropy (spacing estimators) | [sp] `differential_entropy` | on a window's sorted values only | no | at a cadence | [sp]; low value |
| Spearman's rho | [sp] `spearmanr` | E over a window whose rows are kept: a new row shifts other rows' ranks, so work is `O(W)` a row (or `O(W log W)` at a cadence). Without a window the rank of every past row changes: N | **no**: a decayed form would rank a new value in the decayed distribution (`EwQuantile`) and take the EW correlation of those ranks: A, and no library has it. [sp] `weightedtau` is a different quantity (weights by rank, not by age) | yes | [sp] `spearmanr` on the window's rows: 1e-12, ties included |
| Kendall's tau: `variant='b'`, `'c'` | [sp] `kendalltau` | E over a kept window: the entering and leaving rows each change `O(W)` pairs | no (no oracle) | yes | [sp], 1e-12 |
| Somers' D | [sp] `somersd` | E from the same pair counts | no | yes | [sp] |
| Chatterjee's xi: `y_continuous` | [sp] `chatterjeexi` | E over a kept window, at a cadence (`O(W log W)` each) | no | yes | [sp], 1e-12 |
| distance correlation and covariance (biased, bias-corrected); `exponent` | [dcor] | E over a kept window, at a cadence: `O(W²)`, or `O(W log W)` for one column against one column | no | yes | [dcor] `distance_correlation`, `u_distance_correlation_sqr` on the window: 1e-10 |
| partial distance correlation; energy distance; their tests | [dcor] | as above; the permutation tests draw | no | yes | [dcor]; tests by their statistic only |
| multiscale graph correlation | [sp] | N in effect: `O(W²)` with 1,000 permutations | | | not built |
| p-values for the rank measures | [sp] | E at read time from the window's count | | yes | [sp]; under serial dependence they are wrong, as `marginal`'s `serial_rule` exists to say: carry the same warning |

**Verdict.** Binned mutual information and the table statistics: **extend
`marginal`** with a binned target (`target_bins=`) and joint counts, read back
through `bank.marginal()`. The rank and distance measures: a new windowed
accumulator that keeps the window's rows, since no decayed form has an oracle.

**Hard parts and risks.** (1) The joint table is `bx·by` cells per (feature,
target) pair: at 16 × 16 bins and 1,000 features, 2 MB per target per group on
top of `marginal`'s table; with a window, times the ring's snapshots. Raise it.
(2) Windowed ranks keep rows: `8·W·k` bytes and `O(W)` work per row per pair;
give a cadence (`rank_every`) and a cap on `W`. This is the row ring tasks 495
and 532 also need. (3) Plug-in mutual information is biased upward by about
`(bx−1)(by−1)/(2n)`: emit the Miller-Madow value beside it and say which is
which. (4) Pairs: all features against all targets (as `marginal`), not all
pairs of features, unless asked.

**State.** Mutual information: `8·bx·by` per pair. Ranks: `8·W·(columns)` plus
sorted indexes. **Effort.** Mutual information on `marginal` S (2 days);
windowed Spearman, Kendall and xi M (4 days); distance correlation S more.

## 17. Proposed build order

The plan's order (beneath task 510) is cheapest first. Asked here for hardest and
most valuable first where dependencies allow. The dependencies: 500's family code
before 497; 508's ADWIN before 503's adaptive trees and leveraging bagging; a row
ring (task 532) before 495 and 510's ranks; the counter-based stream before 497's
particles, 503's ensembles, 504's sampling policies and 507's tree construction;
the delayed-emission rule shared by 490, 496 and 522.

| wave | tasks | why here |
|---|---|---|
| 0 | the shared pieces of §0.3: the row ring (with 532), the counter-based stream, the family objects | each is needed by three or more tasks; building them inside the first consumer would fix their shape too early |
| 1 | **500** GLMs; **498** Hawkes; **495** VAR and Granger | the three most valuable new capabilities (counts and positive targets; event intensities on the Binance data; multivariate lags with a test), each with an exact library check, and independent of one another. 498 is the hardest of the three and has no dependency: start it first |
| 2 | **496** fixed-lag smoother; **499** `rcov` kinds; **508** drift options with ADWIN | 496 after its state growth is agreed; 499 is contained in `rcov`; 508 is cheap and unblocks 503 |
| 3 | **503** trees (Hoeffding regressor and classifier first); **497** EKF and UKF as `kalman` families | 503 is the largest build and waits on the user's word on its state size and on 508; 497's Kalman forms ride on 500 |
| 4 | **510** mutual information then windowed ranks; **501** expectiles; **506** mixtures on `hmm`; **502** feature helpers; **509** `audit` table and HyperLogLog; **505** block incremental PCA | small extensions of existing models; any of them fills a free slot at any time, and none blocks another |
| 5 | **504** bandits; **507** half-space trees, then the random cut forest; **497**'s particle filter; **502**'s KRLS; **509**'s count-min | each needs a decision first (randomness on the output path; an oracle that is a definition or a band; cross-OS continuation of a sampled state) |

## 18. For the user before the builds

1. **State growth to approve** (the 2026-11-20 rule): 503 trees (tens of MB per
   group without caps); 496 (`L·1.5k²` per target); 510's joint tables and row
   rings; 495 at wide `k·p`; 502 at `D` in the hundreds; 507's forests; 497's
   particles.
2. **Does a paper's definition count as the oracle** where no Python library
   computes the quantity: bipower and tripower variation, MedRV/MinRV, the jump
   tests (499); tabular CUSUM (508, with published run lengths); Oja and CCIPCA
   (505); the random cut forest's insertion (507); KRLS past its budget (502).
   The rule beneath task 510 says a form with no oracle is not built;
   ORACLE-COVERAGE counts a longhand as an oracle of the weaker kind.
3. **Randomness on the output path** (504's sampling policies, 497's particles,
   503's ensembles): reproducible through a counter-based stream in the state,
   but ENHANCEMENTS §4 rejected it once.
4. **A particle state across operating systems** (497): bit-identical only with
   software `exp`/`ln`, a new pure-Rust dependency, or else continued only on
   the OS that saved it.
5. **New test-only dependencies** (dev group, none in the package): `tick`
   (498), `datasketch` (509), `mabwiser` (504), `dcor` (510), `ruptures` (508),
   and optionally `pykalman` (496/497), `glum` (500), `datasketches` (509),
   `rrcf` (507), `arch` (499's HAR). Check each one's licence and its wheels for
   CI's Pythons before adding it; tick's are confirmed above.
6. **Candidates this scan met that are in no task**: Johansen cointegration and
   VECM from the VAR's Gram (§1); interacting-multiple-model filters, `hmm` over
   `kalman`s (§3); a smoothed `holt` (§2); the per-row key count as a feature
   (§15).
7. **Three plan sentences to correct** when 494 is ticked: task 509's
   "`datasketch` (HyperLogLog, count-min)"; task 496's oracle
   (`UnobservedComponents`); task 499's "`arch`'s realized measures if present"
   (absent). The rest of §0.2 refines without contradicting.
