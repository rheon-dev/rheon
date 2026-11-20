# Oracle gap scan: diagnostics and evaluation

Read-only scan of rheon 2.0.0 (`.claude/slots/1` at 5122b435), 2026-11-20.
Scope: the residual switches (`emit_calibration`, `emit_breaks`, `emit_specification`,
`emit_tails`, `emit_influence`, `emit_feature_health`, `emit_drift`, `conformal`,
`resid_quantiles`, `emit_metrics`, `emit_autocorr`, `emit_sigma/zscore`), the
`seqtest`, `corrchange` and `audit` specs, and `rp.eval` (`metrics`, `window_metrics`,
`compare_specs`, `sums`, `diebold_mariano`, `clark_west`, `seqtest`, `grid_search`).
New models are out of scope: the change detectors, anomaly detectors and sketches are
PLAN 494/507-509, and GARCH, AR and VAR are 486/492/495.

`runs`, `gaps`, `spread`, `since` and `pieces` (`crates/rheon-core/src/*.rs`) are
internal mechanics: a held-value run per slot, targets with nulls, the insensitivity
band's unit, cadence stamps, and thread-invariant matrix tiling. None of them is a
diagnostic a user reads, and no oracle library has an equivalent *feature*. Their
oracles are numpy, faer and the longhands that already hold them. They have no table
below. Audit's "runs" (the longest run and repeat share) is covered under `audit`.

Sources checked before marking a row new: `docs/ENHANCEMENTS.md`, `docs/PLAN.md` (main
checkout, through task 510), `docs/DIAGNOSTICS.md`, `docs/records/ORACLE-COVERAGE.md`,
`docs/records/CONFORMAL-STEP.md` and `docs/records/PLAN-ARCHIVE-1.md` (task 76).

The **online** column uses these codes:
- **1P**: one pass, O(state), and works under decay or a window.
- **1P-k²**: one pass, with cost that grows as stated.
- **sketch**: one pass, but only approximate, through rheon's decayed DDSketch or t-digest.
- **passes**: needs a second pass, the full sample, or a bootstrap. These rows are marked and not proposed.

---

## 1. Current surface: parameters and outputs

| diagnostic | parameters | outputs (per target `t`) |
|---|---|---|
| `emit_sigma`, `emit_zscore` | the model's half-life | `sigma_t`, `zscore_t` |
| `resid_quantiles` | levels list | `abs_resid_q<p>_t` (decayed DDSketch of \|resid\|, 0.78%) |
| `conformal` | `conformal` (one coverage), `conformal_rate` (float or `"auto"`) | `lo_t`, `hi_t`, `coverage_t`. Symmetric `pred ± m·sigma`, and `m` tracks the quantile of \|resid\|/sigma |
| `emit_metrics` | none | `ic_t`, `r2_t` (Brier skill on logistic), `hit_rate_t` (accuracy at 0.5 on logistic) |
| `emit_autocorr` | `resid_autocorr_lag` | `autocorr_t` (one lag) |
| `emit_drift` | `drift_delta`, `drift_threshold`, `drift_action` (`flag`/`reset`) | `drift_t`: Page-Hinkley on \|resid\|/sigma, upward only, with an EW mean and a clock-integrated excess |
| `emit_calibration` | `calibration_half_life`; NW lags under `horizon_rows` | `calibration_slope/intercept/wald_t` (Mincer-Zarnowitz, chi2(2)); on quantile fits `calibration_coverage_t` and a binomial Wald |
| `emit_breaks` | `breaks_half_life` | `studentized_t`, `cusum_t` (BDE run once, EW MOSUM-like windowed), `cusum_sq_t` (sum of z²−1 over measured E[z⁴]−1, deliberately not BDE's), `break_wald_t` (fast vs 4× slow fit) |
| `emit_specification` | `specification_half_life`, `ljung_box_lags`, `horizon_rows` | `ljung_box_t` (one Q at L lags), `breusch_pagan_t` (Koenker), `ramsey_reset_t` (LM, powers 2 and 3 of the fitted value) |
| `emit_tails` | `tails_half_life` | `skew_t`, `kurtosis_t`, `jarque_bera_t` |
| `emit_influence` | `influence_half_life` | `influence_t` (online DFFITS at the newest row). Only on ewridge, rls and kalman |
| `emit_feature_health` | `feature_health_half_life` | `spread_ratio_x`, `mean_shift_x` (fast vs 4× slow EW moments) |
| `emit_robust_se` | `robust_se_half_life`, `robust_se_lags` | `se_coef_hc0`, `se_coef_hac` |
| `seqtest` (spec and `rp.eval.seqtest`) | `a`, `b`, suffixes, `min_weight` | `log_e_pos/neg_t` or `log_e_a/b_t`, the counts. A KT-mixture sign bet, two one-sided gamblers |
| `corrchange` | `kind` monitor/sequential/window, `window_rows`, `alpha`, `alpha_adjust` (bonferroni/none), `bandwidth`, `scalar`, `crit`, `n_perm`, `permute_every_rows`, `perm_block`, `norm` (l1/linf), `seed`, `change_action`, `monitor_rows`, `boundary_gamma` | `stat`, `crit`, `flag`, `since_flag`, `since_change` |
| `audit` | `columns`, `pairs`, `distinct_cap` | via `bank.audit()`/`check()`: null/NaN/inf counts, moments, Misra-Gries repeats, runs, smallest change, lag-1 autocorr, Dickey-Fuller with a constant and no lags, median/MAD, pairs, clock steps |
| `rp.eval.metrics` / `window_metrics` / `compare_specs` / `sums` | `group`, `targets`, `min_samples`, `binary`, `weight` (sums), `clock`/`every` | `n`, `r2`/`brier_skill`, `ic`, `hit_rate`/`accuracy`, `log_loss` (binary, eval only), `mse`, `rmse` |
| `rp.eval.diebold_mariano` / `clark_west` | `a`/`b` or `big`/`small`, suffixes, `lags`, `group`, `half_life`, `min_samples` | `n`, `mean`, `dm`/`cw`, `p_value`. Squared loss only, Bartlett NW, no HLN correction |
| `rp.eval.grid_search` | `n_splits`, `gap`, `scoring` (`mse`/`loglik`) | per-point scores |

---

## 2. Gaps per diagnostic

### 2.1 Conformal intervals (`conformal`)

| oracle feature | rheon equivalent | online | value | oracle test it would enable | status |
|---|---|---|---|---|---|
| `mapie.regression.SplitConformalRegressor(confidence_level=[0.8, 0.9, 0.95])`, and crepes `predict_int(confidence=)` per call ([MAPIE regression.py](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/regression/regression.py)) | one float `conformal` | 1P (one multiplier per level, O(levels)) | **high**: fan charts and nested intervals are the common ask, and today they need one spec per level | each level's `m` sequence equals a single-level run, bit for bit; nesting `lo_0.8 ≥ lo_0.9` holds | new |
| asymmetric intervals: `AbsoluteConformityScore(sym=False)` ([bounds/absolute.py](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/conformity_scores/bounds/absolute.py)) and `ConformalizedQuantileRegressor(symmetric_correction=False)` | symmetric \|resid\|/sigma only | 1P (two trackers on the signed residual at α/2 and 1−α/2) | **high**: on skewed residuals (delays, prices, counts) a symmetric band wastes width on one side | per-side miss rates telescope to α/2 each (the published bound); each tracker equals the replayed recursion | new |
| `GammaConformityScore` (\|y−ŷ\|/\|ŷ\|), and the `ResidualNormalisedScore` / `StdConformityScore` choice of scale | always normalized by `sigma` | 1P | low-medium: a multiplicative band for positive targets (Poisson `sgd`) | the normalized score replayed per row | new |
| crepes Mondrian categories, `ConformalRegressor.fit(..., bins=)` / `WrapRegressor.calibrate(mc=)` ([crepes API](https://crepes.readthedocs.io/en/latest/crepes.html)) | none inside a spec (a `group` splits the model too) | 1P, one multiplier per bin of a categorical column, O(bins) | medium: coverage per regime or category without splitting the fit | per-bin coverage within the bound; a one-bin run equals today's | new |
| crepes `ConformalRegressor.predict_p_online(y_hat, y, sigmas=)`: the conformal p-value of the realized target | none | sketch (the rank of \|resid\|/sigma in the decayed DDSketch, within 0.78%) | **medium-high**: a per-row PIT/p-value is the input for uniformity checks, martingale tests and CRPS | uniformity with `scipy.stats.kstest`; against crepes `predict_p` exactly at `half_life=inf` with a large sketch | new |
| crepes `ConformalPredictiveSystem.predict_percentiles_online` / `predict_cpds` (the full predictive distribution) | `resid_quantiles` of \|resid\| only (unsigned) | sketch (signed-residual quantiles plus `pred`) | medium: predictive quantiles for any level after the fact | crepes CPS percentiles at `half_life=inf` within the sketch's error | new |
| crepes `predict_int(y_min=, y_max=)` clipping | none | 1P, trivial | low: a known support, such as a non-negative target | clip identity | new |
| Conformal PID: integrator `KI`, `Csat`, and the scorecaster ([conformal-time-series methods.py](https://github.com/aangelopoulos/conformal-time-series/blob/main/core/methods.py)) | the quantile-tracking (P) step only. CONFORMAL-STEP §2.1 names it but builds no I-term | 1P (the saturating integral of the coverage error is one f64) | medium: faster recovery after a level shift of the noise, which CONFORMAL-STEP §6 names as the delay rule's weak spot | `conformal-time-series` `pid` run on the bank's scores with `sigma` held constant | new |
| MAPIE `TimeSeriesRegressor(method="aci").adapt_conformal_inference(gamma=)` (level tracking over an empirical quantile) | the quantile tracker | sketch | low: the same guarantee by another route | none, since ORACLE-COVERAGE §4.2 already says it differs | rejected as an oracle (ORACLE-COVERAGE §4.2). As a feature it is new and low |
| AgACI / SAOCP (and DtACI, the same family of experts over rates) | `conformal_rate="auto"` | 1P, K experts | low | the replay in `scripts/conformal_step/` | **rejected** (CONFORMAL-STEP.md §0 and §6: they swing or under-cover under label delay). DtACI was not tried |
| MAPIE `TimeSeriesRegressor(method="enbpi")` | none | **passes** (a bootstrap ensemble) | — | — | not proposed |
| MAPIE `SplitConformalClassifier(conformity_score="lac"/"aps"/"raps"/"top_k")` and crepes `ConformalClassifier.predict_set_online(class_cond=)` | none for logistic `sgd`/`ftrl` or `ew_class` | 1P (track a threshold on 1−p_true; class-conditional means one tracker per class) | medium: prediction sets with a guarantee for the classifiers rheon ships | `classification_coverage_score` / `mean_width` at the tracked threshold; crepes `predict_set_online` at `half_life=inf` | new |
| MAPIE `OnlineMartingaleTest(test_method="jumper_martingale", jump_size=, burn_in=)` ([martingales.py](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/exchangeability_testing/martingales.py)) | none | sketch (p-values from the decayed DDSketch rank; exact needs history) | medium | MAPIE's jumper on the same p-values | overlaps planned **task 508** (change detectors): propose it there as an option |

### 2.2 Calibration (`emit_calibration`) and the probability scores

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `mapie.metrics.calibration.spiegelhalter_statistic` / `_p_value` ([calibration.py](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/metrics/calibration.py)) | Mincer-Zarnowitz on a probability | 1P (three EW sums: Σ(y−p)(1−2p) and Σ(1−2p)²p(1−p)) | **high**: the proper z-test of a binary forecast's calibration. MZ's linear slope on a 0/1 target is not | MAPIE run once; under a memory, at the Kish size by definition | new |
| `sklearn.calibration.calibration_curve(n_bins=, strategy="uniform")` ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)) and `mapie...expected_calibration_error(num_bins=)`, `top_label_ece` | none | 1P with fixed uniform bins (EW count and mean y per bin). `strategy="quantile"` is sketch | medium: a reliability curve and ECE per row for `sgd`/`ftrl` logistic and `ew_class` | sklearn `calibration_curve` and MAPIE ECE at `half_life=inf` | new |
| MAPIE `kolmogorov_smirnov_statistic` / `kuiper_statistic` (cumulative-difference calibration) | none | **passes** (sorted by score) | low | — | not proposed |
| `sklearn.metrics.brier_score_loss` | `r2` as Brier skill (stream), and `mse` in eval | — | — | already covered (`TestStreamingMetrics`) | covered |
| `sklearn.metrics.log_loss`, streaming | eval only | 1P, but the state holds a `ln` | medium | sklearn `log_loss` | **rejected** (PLAN-ARCHIVE-1, task 76, "log loss lives only in `po.eval`": no libm result in the state). Task 449 revisits the libm rule and may reopen it |
| Brier decomposition (Murphy: reliability, resolution, uncertainty); sklearn has no such function, but it is the binned ECE's companion | none | 1P with fixed bins | low-medium | the sum of the three parts equals `brier_score_loss` | new (rides with the reliability curve) |
| `sklearn.metrics.mean_pinball_loss(alpha=)`, `d2_pinball_score` ([doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html)) for `quantile` and `sgd(loss="quantile")` | coverage and binomial Wald only | 1P | **high**: the proper score of a quantile fit. Coverage alone cannot rank two calibrated quantile fits | sklearn `mean_pinball_loss` / `d2_pinball_score`, weighted | new |
| `properscoring.crps_gaussian(y, mu=pred, sig=sigma)` ([properscoring](https://github.com/properscoring/properscoring)) | none (`grid_search(scoring="loglik")` has the Gaussian log score) | 1P (closed form, but it needs erf/exp, so it is reported and never stored) | medium: a proper score of `pred ± sigma` | properscoring per row | new. The value comes from libm, so it can be emitted but not accumulated in the state (the same rule as log loss) |
| Interval score: `mapie.metrics.regression.regression_mwi_score(y, intervals, confidence_level)` (Winkler) | none | 1P | medium: the proper score that ranks two conformal intervals, which coverage and width alone cannot | MAPIE `regression_mwi_score` run once | new |

### 2.3 Breaks (`emit_breaks`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| Chow test at a known break date (statsmodels has no function. Build it from two `OLS` fits and `RegressionResults.compare_f_test`) | `break_wald` (fast vs slow fit, no fixed date) | 1P-k² (Grams before and after a given clock/date; F at the end, or running after the date) | medium: event studies and known regime dates are common | statsmodels OLS on each side and the F-test | new |
| `statsmodels.regression.recursive_ls.RecursiveLSResults` CUSUM-of-squares with BDE boundary; `recursive_olsresiduals(alpha=)` → `rcusumci` | `cusum_sq` is a different statistic (deliberately). The `cusum` boundary is computed by hand | BDE CUSUMSQ needs the run's total, so **passes** unless the horizon T is fixed | low | `RecursiveLS.cusum_squares` | rejected by design (DIAGNOSTICS "The CUSUM of squares is not BDE's"). The `cusum` boundary for a given T could be emitted (1P, low value) |
| `breaks_cusumolsresid` (Ploberger-Krämer OLS-CUSUM), `breaks_hansen` (Nyblom-Hansen), Andrews sup-F | `cusum`, `break_wald` | **passes** (full-sample residuals, or every candidate split) | — | — | not proposed |
| `linear_harvey_collier` (t-test on the mean of recursive residuals) | `cusum` run once is the same sum, scaled by the prior spread | 1P | low: it is the same idea | Harvey-Collier t equals `cusum·sqrt(r)/sd(v)` within the scale's definition: an extra oracle for `cusum` run once | new (test only, no field) |
| rectangular MOSUM (Chu, Hornik and Kuan) | EW CUSUM (a MOSUM with an exponential window) | 1P only with a stored window, O(W) | low | — | not proposed: the EW form is by design (DIAGNOSTICS "With a memory, the CUSUM is a moving sum") |

### 2.4 Specification (`emit_specification`, `emit_autocorr`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `statsmodels.stats.diagnostic.het_white(resid, exog)` ([doc](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html)) | `breusch_pagan` (linear in the features) | 1P-k⁴ (a Gram of the k(k+3)/2 squares and cross products) | **high** for small k: DIAGNOSTICS itself shows BP missing a spread symmetric in a feature (\|x0\| read 3.0 < 5.99), which White's squares catch | `het_white` run once; WLS at Kish under a memory, as for BP | new |
| `statsmodels.stats.diagnostic.het_arch(resid, nlags=)` (Engle's ARCH LM) | none: `ljung_box` is on the residuals, not their squares | 1P-p² | **high**: volatility clustering in residuals is what BP misses, and the most common finance failure | `het_arch` run once | new. GARCH itself is task 492; this is the diagnostic only |
| `acorr_breusch_godfrey(res, nlags=)` | `ljung_box` | 1P-(k+p)² (the residual on the features and p lagged residuals) | medium-high: valid when the features include lagged targets (AR terms, task 486), where Ljung-Box is not | `acorr_breusch_godfrey` run once | new (gains value with task 486) |
| `acorr_ljungbox(lags=[...], return_df)`: Q at every lag 1..L | one Q at L | 1P (the lag products are already kept) | low-medium | `acorr_ljungbox` table | new |
| `acorr_ljungbox(model_df=)` | none | 1P | low now, medium after ARMA (task 487) | `acorr_ljungbox(model_df=)` | new. Fold it into 486/487 |
| `acorr_ljungbox(period=)` seasonal lags, `boxpierce=True` | Box-Pierce under a memory and LB run once, automatically | 1P | low | — | new (low) |
| `linear_reset(power=, test_type="exog"/"princomp", use_f=)` ([doc](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_reset.html)) | `power=3`, `fitted`, LM | 1P (`exog`: k·(powers) more regressors, O(k²p²)) | low-medium: `exog` points to *which* feature is nonlinear | `linear_reset(test_type="exog")` | new |
| `het_breuschpagan(robust=False)` (original BP) | Koenker form | 1P | low | statsmodels with `robust=False` | new (low) |
| `statsmodels.stats.stattools.durbin_watson` | `autocorr` (≈ 1 − DW/2) | 1P | low | DW ≈ 2(1−ρ₁) | new (test only) |
| `het_goldfeldquandt`, `linear_rainbow`, `spec_white` | — | **passes** (sorting or subsamples) | — | — | not proposed |

### 2.5 Tails (`emit_tails`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `statsmodels.stats.stattools.robust_skewness` / `robust_kurtosis` (Bowley, Moors, Crow-Siddiqui quantile forms) ([doc](https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.robust_kurtosis.html)) | moment skew/kurtosis, which DIAGNOSTICS calls noisy under heavy tails | sketch (signed-residual EW quantiles) | medium: a stable tail read for setting `drift_threshold`, exactly where the moment kurtosis fails | statsmodels at `half_life=inf` within the sketch's error | new |
| `statsmodels.stats.stattools.omni_normtest` / `scipy.stats.normaltest` (D'Agostino K²) | Jarque-Bera | 1P (from the same moments, at Kish n) | low | scipy `normaltest` run once | new (low) |
| `medcouple`, `normal_ad`, `lilliefors` | — | **passes** | — | — | not proposed |

### 2.6 Influence (`emit_influence`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `OLSInfluence.dfbetas` ([doc](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.html)) | `influence` (DFFITS: total move only) | 1P-k² (Δβ = A⁻¹x·e/(1+h) is already formed, scaled by se_coef) | **medium-high**: names *which* coefficient a row moved | `OLSInfluence.dfbetas` at the newest row on prefixes, as DFFITS is held | new |
| `OLSInfluence.cooks_distance` | derivable from `influence` | 1P | low (a rescale: D ≈ influence²/(k+1) with the external scale) | statsmodels on prefixes | new (low; could be a doc formula) |
| `OLSInfluence.hat_matrix_diag` (leverage), `cov_ratio` | `error_inflation` = sqrt(1+h) | 1P | low | hat = error_inflation² − 1 | covered in effect (document the identity) |
| `OLSInfluence.resid_studentized_external` | `studentized` (prior spread) | — | — | — | covered |

### 2.7 Drift (`emit_drift`, river's detector of the same kind)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `river.drift.PageHinkley(mode="both"/"down")` ([source](https://github.com/online-ml/river/blob/main/river/drift/page_hinkley.py)) | upward only | 1P | low-medium: "down" flags a model that suddenly fits better or a feed that went flat | river flag for flag with `mode=` at nothing decaying | the user's planned **task 508** ("the rest become options of those") |
| `PageHinkley(min_instances=)` | readiness gates (4.3 half-lives, `min_weight`) | — | — | — | covered |
| `PageHinkley(alpha=)` (fading the sum) | the EW mean, a different form | — | — | — | rejected as an oracle (ORACLE-COVERAGE §4.1 row 1: it differs) |

### 2.8 Feature health (`emit_feature_health`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| a two-sample distribution shift: `scipy.stats.ks_2samp` / PSI between fast and slow memories | spread ratio and mean shift only | sketch (two decayed DDSketches per feature) | medium: catches a change of shape or a new sentinel that leaves the mean and sd alone | `ks_2samp` on the exact windows at `half_life=inf` splits | related to the rejected **KSWIN** (ENHANCEMENTS §4: O(window)). The sketch form is O(state), so it is new and needs a user decision |

### 2.9 Sequential tests (`seqtest`, `rp.eval.seqtest`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| an anytime-valid test of the **mean** loss difference: confseq `betting_cs` / empirical-Bernstein CS ([confseq](https://github.com/gostevehoward/confseq)), and MAPIE `RiskMonitoring(test_level=, tolerance=)` ([risk_monitoring.py](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/exchangeability_testing/risk_monitoring.py)) | a sign test only (DIAGNOSTICS: "a different question" from DM) | 1P (betting on a bounded or clipped differential, O(1)) | **high**: it gives DM's question with seqtest's anytime validity. Today a user must pick one or the other | confseq is not installable (ORACLE-COVERAGE §3b), so a longhand plus Ville's bound by simulation; MAPIE `RiskMonitoring` for the bounded-risk form | new. Task 508 may claim it ("the rest become options of seqtest") |
| `scipy.stats.binomtest(k, n, p=p0, alternative=)` generalized: a null of P(+) ≤ p0, or a test that the q-th quantile is at c | null p0 = 1/2 only | 1P | medium: tests a quantile fit's coverage anytime-validly | `binomtest` fixed-n p-value is ≥ the e-value's 1/E at stopping; the KT closed form at p0 | new |
| confidence sequences (anytime-valid interval on the win rate or the mean) | e-values only | 1P | medium | confseq longhand | new |
| Wald SPRT with a lower (accept) boundary | none: the betting test never accepts the null | 1P | low | the definition | new (low) |
| `statsmodels.stats.multitest.multipletests(method="holm"/"fdr_bh")` across targets, or e-BH on the e-values | none | 1P per row (sort T values) | low-medium | `multipletests` | **rejected** (ENHANCEMENTS §9: "multiple-testing corrections ... is analysis over the library's output") |
| statsmodels SPRT or group-sequential tools | — | — | — | — | none exist in statsmodels; nothing to compare |

### 2.10 Correlation change (`corrchange`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `multipletests(method="holm"/"sidak"/"fdr_bh")` for the per-pair p-values | `alpha_adjust="bonferroni"/"none"` | 1P (per due row, sort the P pair p-values) | low-medium: Holm is uniformly more powerful than Bonferroni at the same FWER | statsmodels `multipletests` on the pair p-values | new as an option, but ENHANCEMENTS §9 rejects corrections in general. `corrchange` already carries Bonferroni, so this one needs a user decision |
| the `bandwidth` kernel choice (`statsmodels.stats.sandwich_covariance` kernels: Bartlett, Parzen, QS) | Bartlett only | 1P | low | `S_hac_simple` with weights | new (low) |
| no Python library implements Wied-Galeano. The published table is the oracle | — | — | — | — | covered |

### 2.11 Audit (`audit`, `check()`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `statsmodels.tsa.stattools.adfuller(maxlag=, autolag=None, regression="c"/"ct"/"n")` ([doc](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)) | DF with a constant and 0 lags | 1P-p² (regression on p lagged differences, per column) | medium: the 0-lag DF over-rejects a unit root on a column with autocorrelated changes, so `random_walk` can be missed | `adfuller(maxlag=p, autolag=None, regression=)` statistic | new |
| `adfuller(autolag="AIC")` | — | **passes** (fits for every lag) | — | — | not proposed |
| `acorr_ljungbox` on a column (beyond lag 1) | lag-1 autocorr only | 1P-L | low | `acorr_ljungbox` | new (low) |
| `kpss` | — | **passes** (full-sample residuals) | — | — | not proposed |

### 2.12 Evaluation and metrics (`emit_metrics`, `rp.eval`)

| oracle feature | rheon equivalent | online | value | oracle test | status |
|---|---|---|---|---|---|
| `sklearn.metrics.mean_absolute_error`, river `metrics.MAE` | none (only mse/rmse in eval; nothing streaming) | 1P | **high**: the most-asked regression metric, and robust to the tails rheon documents | sklearn `mean_absolute_error(sample_weight=)`; river `MAE` row by row | new |
| `sklearn.metrics.mean_pinball_loss` in eval | none | 1P (a sum, so mergeable) | high (see §2.2) | sklearn | new |
| `sklearn.metrics.mean_poisson_deviance` / `mean_tweedie_deviance(power=)`, `d2_tweedie_score` | `r2` and `ic` only. `hit_rate` is null on Poisson `sgd` | 1P (needs `ln`: report it, do not store it; eval is fine) | medium: the proper score and skill for `sgd(loss="poisson")` | sklearn | new (eval). Streaming hits the libm rule (task 76/449) |
| `sklearn.metrics.mean_absolute_percentage_error`, river `SMAPE` | none | 1P | low-medium | sklearn | new |
| `sklearn.metrics.median_absolute_error`, `max_error` | none | sketch / 1P | low | sklearn | new (low) |
| `sklearn.metrics.roc_auc_score`; river `metrics.ROCAUC(n_thresholds=)` / `RollingROCAUC` ([river metrics](https://riverml.xyz/latest/api/overview/#metrics)) | accuracy at 0.5 | sketch (river's threshold grid; exact AUC needs a sort) | medium: threshold-free ranking quality for logistic fits | river `ROCAUC` with the same thresholds, exactly; sklearn within the grid's error | new |
| `sklearn.metrics.precision_score` / `recall_score` / `f1_score` / `matthews_corrcoef`; river `F1`, `MCC`, `BalancedAccuracy` | accuracy only | 1P (the EW confusion matrix) | medium for imbalanced binary targets | sklearn with `sample_weight` | new |
| MAPIE `regression_coverage_score`, `regression_mean_width_score`, `regression_ssc(num_bins=)` (coverage stratified by width), `coverage_width_based` ([regression.py](https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/metrics/regression.py)) | `coverage_t` streaming. `rp.eval` scores no interval columns | 1P (SSC needs fixed bins or a sketch) | **high**: an interval evaluated after the run, per group or window, beside `metrics` | MAPIE's functions on `lo`/`hi` | new |
| MAPIE `hsic` (independence of coverage and width) | none | **passes** (kernel, O(n²)) | — | — | not proposed |
| DM with a loss choice (absolute or power), as in `dieboldmariano.dm_test(..., one_sided=, harvey_correction=True)` ([dieboldmariano](https://github.com/edoannunziata/dieboldmariano)) | squared loss only, two-sided p, no HLN | 1P | **medium-high**: HLN fixes DM's over-rejection on short keys, which the docstring itself warns about. An absolute loss pairs with MAE | `dm_test(harvey_correction=True)` and statsmodels HAC t times the HLN factor; check the h ↔ lags+1 map (ORACLE-COVERAGE §4.2) | new |
| a DM p-value from Student's t(n−1) (part of HLN) | normal | 1P | (in the row above) | scipy `t.sf` | new |
| Giacomini-White (2006) conditional predictive ability: a Wald test of `E[d_t · h_{t−1}] = 0` for test functions `h` (constant, lagged d, a regime column). No mainstream Python or arch implementation | none | 1P-q² (EW Gram of `h·d`, HAC at lags) | **medium-high**: says *when* one model beats another, not just on average, and its unconditional form justifies DM between rheon's limited-memory fits even when they are nested | the definition with statsmodels `OLS(d, h).fit(cov_type="HAC")` Wald (`f_test` / `wald_test`) run once | new |
| `arch.bootstrap.SPA`, `StepM`, `MCS` (superior predictive ability, model confidence set) ([arch](https://arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparisons.html)) | `compare_specs`, `emit_selected` | **passes** (a block bootstrap over the loss series) | — | — | not proposed |
| `statsmodels.stats.diagnostic.compare_encompassing` / `compare_j` (non-nested OLS) | `clark_west` (nested), DM | 1P-k² | low | statsmodels | new (low) |

---

## 3. Ranked top ten

1. **Asymmetric and multi-level conformal intervals** (§2.1): two signed trackers per level, `conformal=[0.8, 0.9]`. MAPIE `sym=False` and `confidence_level=[...]`. 1P, O(levels). The replayed recursion and the per-side telescoped bound hold it.
2. **Interval scores in `rp.eval`** (§2.12, §2.2): coverage, mean width, Winkler/MWI and size-stratified coverage over `lo`/`hi`, per group and window, mergeable through `sums`. MAPIE `regression_*` functions are the oracle.
3. **Pinball loss for quantile fits** (§2.2, §2.12), streaming and in eval. sklearn `mean_pinball_loss` and `d2_pinball_score`.
4. **MAE, streaming and in eval** (§2.12). sklearn and river `MAE`.
5. **ARCH LM test** in `emit_specification` (§2.4): `het_arch` run once. This is the diagnostic, separate from GARCH (task 492).
6. **Spiegelhalter's calibration test** for logistic fits (§2.2). MAPIE `spiegelhalter_statistic`. Three EW sums.
7. **Diebold-Mariano with the HLN correction, a t reference, a loss choice and one-sided p** (§2.12). `dieboldmariano.dm_test(harvey_correction=True)`.
8. **White's heteroskedasticity test** (§2.4): it fixes the BP blind spot that DIAGNOSTICS documents. `het_white` run once. Cost grows as k⁴, so it is for small k.
9. **An anytime-valid test of the mean loss difference** (§2.9), beside the sign `seqtest`. confseq is not installable, so a longhand; MAPIE `RiskMonitoring` covers the bounded form. Task 508 may absorb it.
10. **DFBETAS beside `influence`** (§2.6): `OLSInfluence.dfbetas` on prefixes, as DFFITS is held today.

Next after these: Giacomini-White conditional test, the conformal p-value/PIT per row (crepes `predict_p_online`), Breusch-Godfrey (once task 486 lands), the reliability curve and ECE, ROC AUC on river's threshold grid, the Chow test at a known date, ADF with lags in `audit`, and robust skewness and kurtosis.

**Already decided, so not ranked:**
- Streaming log loss: task 76, the libm rule. Task 449 may reopen it.
- Multiple-testing corrections: ENHANCEMENTS §9.
- KSWIN-style tests over a window: ENHANCEMENTS §4.
- AgACI and SAOCP: CONFORMAL-STEP §6.
- BDE's CUSUM of squares: DIAGNOSTICS, a different statistic by design.
- Page-Hinkley `mode=`: task 508.
- The conformal test martingale: task 508.
