<!-- kind: record -->
# Test realism: which tests run on a well-posed problem, and which on a degenerate one (task 550)

*Status, 2026-11-22: a record, written by reading the test sources at
`5878a47d`; nothing was run. It is the audit half of task 550: every model
of `rheon-core` and each kind of test that holds it, classified as realistic
or degenerate, with what is missing. The realistic cases the audit calls for
are added one model a commit, and each adds a row's "missing" cell to the
log at the end. The degenerate tests stay: they hold edges a realistic
stream never reaches.*

## 1. Definitions and method

**Degenerate** means the test's problem is not identified or not well
posed: no signal, a flat likelihood, a feature collinear with the
intercept, too few rows to identify the model, a constant (or noiseless)
target, or a start shorter than the model needs. **Realistic** means
identified and well posed: a seeded stream with a known truth or a
reference's answer, a signal above its noise, enough rows for the model's
parameters. A third class, **R\***, is realistic but short: well posed at 40
to 60 rows, so it pins arithmetic and forms and does not show that a fit
converges.

Degenerate is not a fault. A golden number, a frozen state or a contract
probe is meant to move when the arithmetic moves, and an adversarial stream
is meant to find the input a model mishandles. The audit asks only whether
each model also has a realistic test of each kind, so a mistake that a
well-posed problem exposes and a degenerate one hides cannot pass.

The four kinds of test, and what each one's streams are (read from the
source):

| kind | where | the streams |
|---|---|---|
| **G**, golden | `crates/rheon-core/tests/golden.rs`, 58 tests | one fixed stream: 60 rows, two iid uniform features on [-1, 1], `y = 1.5 x0 - 0.75 x1 + 0.25 + 0.1 u` (a signal of variance about 0.9 over noise of variance 0.003), one null target, one clock gap of 25, weights 0.5 to 1.5. The same rows feed every model, so a model that looks for structure (clusters, regimes, a correlation break, self-excitation) is given none |
| **F**, state fixtures | `crates/rheon-core/tests/state_fixtures/` (114 cases) from `tests/cases/mod.rs`; the bank's in `crates/rheon-polars/tests/state_fixtures/` (22 files) | 40 rows before the save and 16 after (a few 2 to 8 before it, on purpose: a model still warming; four garch cases 300 to 2,700), the same signal, features shifted by 2.0 every 12 rows (two regimes, three alternations), nulls, zero weights, gaps. They freeze forms; each case is a form, not a fit |
| **O**, oracle comparison | `tests/test_second_opinion*.py`, `tests/test_oracles*.py`, `tests/test_convergence*.py` (`tests/convergence.py`'s registry), `tests/test_truth_recovery.py`, `tests/test_river.py`, `tests/test_learning_rate.py`, and the 69 ledger groups in `oracles.toml` | seeded numpy streams with a known truth (the convergence grids: 50 or more seeds a case, every regression model) or a third-party library's answer on the same rows; public data where a test downloads it (Franti's sets, TCPDBench, NIST StRD) |
| **C**, contract and property | `crates/rheon-core/tests/model_contract.rs`; `tests/test_semantics_all_models.py`, `tests/test_properties*.py`, `tests/test_model_registry.py`; chunk invariance and resume in `tests/test_bank.py` on `tests/data.py::synthetic` | the Rust probe: 40 rows of an exactly linear target with no noise, a bounded script (the value bound in every position, 300 ordinary rows), and proptest streams (arbitrary values, a target independent of the features, a repeat that holds a column still). The Python sweeps: 40 rows of `y = 2 x0 - x1` with no noise (a coin-flip label for the classifiers), and hypothesis streams of arbitrary values. `synthetic` is realistic: 3 groups of 400 rows, a random-walk coefficient, noise of 0.5 |

Counts below are test functions or cases found by reading the files; a
parametrized test counts once, and a ledger group's cases are named where
they matter. Where a cell says "the file", the whole file's tests share one
data constructor.

## 2. Findings in one place

1. **Every regression model has a realistic oracle comparison** (`ewridge`,
   `rls`, `lasso`, `kalman`, `huber`, `quantile`, `sgd`, `pa`, `ftrl`,
   `holt`, `glm`, `expectile`): `tests/convergence.py` registers a
   known-truth generator and a statsmodels or scikit-learn oracle for each,
   and a 50-seed grid holds the worst seed. The extra rows below name an
   option that only reductions or a longhand hold.
2. **Every model that looks for structure has a realistic oracle or recovery
   test** (`kmeans`, `micro`, `hmm`, `bocpd`, `corrchange`, `deco`, `rcov`,
   `ew_class`, `hawkes`, `audit`, `marginal`, `ew_cov`), but **its golden and
   most of its contract and fixture streams carry no structure**: the golden
   stream is iid uniform noise, so a clusterer, a regime model and a change
   detector are pinned on data with nothing to find. Those goldens are
   degenerate by the definition above, and no realistic Rust-side test
   stands beside them (the realistic ones run through the bank in Python).
3. **No model lacks a realistic oracle comparison outright.** The models
   whose realistic comparison is **not third party**: `seqtest` (a scalar
   replay, the published KT closed form through `math.lgamma`, and Monte
   Carlo of Ville's guarantee: confseq does not build), `deco` (the pair sum
   written longhand and a scipy density for `loglik`; the level is recovered
   on a one-factor stream but no library computes it), and the diagnostics
   `conformal` (a replayed recursion: no library has this step). Every one
   has a realistic test; none has a library's answer for its own number.
4. **The contract probe's target has no noise** (`model_contract.rs` and
   `test_semantics_all_models.py`), so every residual-scale, likelihood or
   significance output is read at a zero spread there. The realistic reading
   of those outputs is held in the oracle files, not beside the contract.
5. **Option-level gaps** (an option held by a reduction or a longhand only,
   or by a degenerate stream), the ones task 550's realistic cases start
   from, are ranked in section 5.
6. **GARCH is out of this task's scope** (task 548 reworks its start
   optimizer on the same finding: the golden is the 60-row stream repeated,
   a periodic series rather than a variance process, and the start
   buffers 50 rows). Its rows are classified and left.

## 3. The models

Each model has a row for each kind. *Class* is R (realistic), R\*
(realistic and short), D (degenerate, with the reason) or a mix. *Missing*
is what a realistic test of that kind would add; "none" means the kind has a
realistic test and nothing is left to add.

### 3.1 Regression models

| model | kind | tests | class | missing |
|---|---|---|---|---|
| `ewridge` | G | 5 (`ewridge`, `_standardized`, `_windowed`, `_origin_standardized`, `_paths`) | R\* | 60 rows at a signal-to-noise of about 300: the fit is exact to the noise; no ill-conditioned or correlated pair of features |
| | F | 7 (`ewridge`, `_window`, `_leverage`, `_output_only`, `_first_period`, `_never`, `_blocked`), 2 targets | R\* | forms only; none |
| | O | `convergence.ewridge` (19 cases, 50 seeds: scikit-learn Ridge, statsmodels WLS), `convergence_conditioning.nist_ewridge_rls_gram` (41, NIST StRD), `test_second_opinion` (32 specs: sklearn Ridge, statsmodels WLS, ridge and elastic net), `_diagnostics`, `_grids`, `_restarts`, `_windows`, `test_oracles_ewridge_paths` | R | none |
| | C | Rust probe D (noiseless line); proptest D; sweeps D (noiseless, 40 rows); `test_bank` chunk invariance and `test_predict` on `synthetic` R | D / R | none for the contract; sigma, `r2` and `se_coef` at a nonzero noise are in O |
| `rls` | G | 1 | R\* | as `ewridge` |
| | F | 1 (`rls`) | R\* | none |
| | O | `convergence.rls` (19, statsmodels RLS, sklearn), `nist_ewridge_rls_gram`, `test_second_opinion_rls_errors` (leverage, sandwich), `test_oracles_rls_kalman_paths` | R | none |
| | C | as `ewridge` | D / R | none |
| `lasso` | G | 2 (`lasso`, `_windowed`) | R\* | two features: no stretch where the penalty path selects among many; no collinear pair |
| | F | 4 (`lasso`, `_window`, `_floor`, `_first_period`) | R\* | none |
| | O | `convergence.lasso` (18, sklearn Lasso), `convergence_iterations.lasso_descent`, `test_oracles` (KKT), `test_oracles_lasso_paths`, `test_second_opinion_gram` (`lars_path`, `enet_path`) | R | none |
| | C | as `ewridge` | D / R | none |
| `kalman` | G | 4 (`kalman_plain`, `kalman`, `_revert`, `_noise`) | R\* | the truth is a constant coefficient; the filter is never run on a drifting one |
| | F | 5 (`kalman`, `_gap`, `_noise_units` (3 rows before), `_output_only`, `_square_root`) | R\*; `_noise_units` D (3 rows: shorter than the filter's start) | none (the 3-row case freezes the form before the start, on purpose) |
| | O | `convergence.kalman` (19, statsmodels), `long_run.kalman_p`, `test_second_opinion` (10: filterpy, statsmodels exact diffuse), `_extremes`, `_grids`, `test_river` | R | none |
| | C | as `ewridge` | D / R | none |
| `huber` | G | 2 (`huber`, `_origin_standardized`) | R\*, but the stream has no outlier | no contaminated row, so the loss is never in its linear region by design |
| | F | 13 (`huber`, `_first_period`, `_joining`, `_mad`, `_mad_anchored`, `_mad_awaiting`, `_mad_awaiting_first_value`, `_sets`, `_sum`, `_refit`, `_refit_first_value`, `_refit_done`, `_refit_span`) | R\*; the two `_awaiting` cases (2 rows before) D, on purpose | none |
| | O | `convergence.huber` (26, statsmodels RLM, contaminated streams), `huber_scale.huber_sgd_cuts`, `test_second_opinion` (4: `HuberRegressor`, MAD), `test_huber_options` | R | `feature_sets=`, `ridge_scale="sum"` and the solve cadence: by reduction to `ewridge` and `rls` only (section 5, item 3) |
| | C | as `ewridge`; the robust contamination sweep (`test_convergence_soak`, soak tier) is R | D / R | none |
| `quantile` | G | 1 | R\* | the stream's noise is uniform, symmetric: the median and the mean coincide, so a quantile at 0.5 reads the same as least squares; no skewed noise |
| | F | 5 (`quantile`, `_mad`, `_refit`, `_refit_done`, `_refit_span`) | R\* | none |
| | O | `convergence.quantile` (26, statsmodels QuantReg), `quantile_robust_sets`, `oracles.quantile_eps`, `_huge_ridge`, `_mixed_scales`, `test_second_opinion` (QuantReg, skewed noise), `test_quantile_refit` | R | none |
| | C | as `ewridge` | D / R | none |
| `sgd` | G | 3 (`sgd`, `_squared`, `_simplex`) | R\* | losses other than squared and the default, run for 60 rows: the learning-rate schedule has barely left its start |
| | F | 14 (`sgd`, `_warming` (20 rows), `_huber`, `_huber_mad`, `_huber_mad_anchored`, `_huber_refit`, `_huber_refit_done`, `_huber_refit_span`, `_spread`, `_spread_adagrad`, `_unstable`, `_resting` (300), `_eps`, `_logistic`) | R\*; `_unstable` D (a step size past stability, by design) | none |
| | O | `convergence.sgd_squared` (18), `_huber` (26), `_quantile` (8), `_epsilon_insensitive` (8, scipy), `_logistic` (6), `_poisson` (6), `_rates`, `learning_rate.*` (15, scikit-learn), `test_oracles_gradient_paths` (10), `test_river`, `test_second_opinion_gradient` | R | `standardize=True` and the constraints (a simplex or a box): `sgd_ref` and one SLSQP projection (section 5, item 4) |
| | C | as `ewridge`; `sgd_unstable` is D by design | D / R | none |
| `pa` | G | 2 (`pa`, `_box`) | R\* | none |
| | F | 2 (`pa`, `_warming`) | R\* | none |
| | O | `convergence.pa` (18, statsmodels), `learning_rate.pa_noisy` (sklearn PassiveAggressive), `test_oracles_gradient_paths` (4), `test_second_opinion_gradient` | R | `standardize=True` and the constraints, as `sgd` |
| | C | as `ewridge` | D / R | none |
| `ftrl` | G | 2 (`ftrl`, `_squared`) | R\* | the sparse regime the model exists for: two dense features, no irrelevant ones |
| | F | 1 | R\* | none |
| | O | `convergence.ftrl_logistic` (7), `ftrl_squared` (9) (statsmodels), `learning_rate` (3), `test_oracles_gradient_paths` (2), `test_second_opinion` | R | a finite half-life: `ftrl_ref` only (no library forgets) |
| | C | as `ewridge` | D / R | none |
| `holt` | G | 1 | R\* | the stream has no trend: the series `y` is a function of random features, so the level and trend states are driven by noise |
| | F | 2 (`holt`, `_inf`) | D: same stream, no trend | a trending series; the fixtures freeze forms only |
| | O | `convergence.holt` (8, statsmodels), `test_second_opinion` (4: statsmodels Holt, missing rows, row after a gap) | R | none |
| | C | as `ewridge` (`holt` takes no features) | D / R | none |
| `glm` | G | 2 (`glm` Gamma/log under decay, `glm_cap` Tweedie) | D: a positive target `y^2 + 0.1` of a linear index, not drawn from the family, so the dispersion is near 0 and the working response is exact | a Gamma or Poisson draw with a known mean |
| | F | 4 (`glm_refit`, `_refit_done`, `_refit_clock`, `glm_cap`) | D: same stream | none for forms |
| | O | `convergence.glm` (6, statsmodels), `test_second_opinion_glm` (statsmodels `GLM` with `freq_weights`, glum; 5 coefficients at 10,000 rows, 20 at 10,000 and 100,000) | R | none |
| | C | as `ewridge` | D / R | none |
| `expectile` | G | 2 (`expectile`, `_window`) | R\* | skewed noise: with symmetric noise an expectile at 0.8 is a small shift of the mean |
| | F | 6 (`expectile`, `_window`, `_clock_window`, `_refit`, `_refit_done`, `_refit_clock`) | R\* | none |
| | O | `convergence.expectile` (8, statsmodels), `test_second_opinion_expectile` (8: statsmodels, scipy `expectile`) | R | none |
| | C | as `ewridge` | D / R | none |

### 3.2 Moments, classification and counts

| model | kind | tests | class | missing |
|---|---|---|---|---|
| `ew_cov` | G | 2 (`ew_cov`, `_paths`) | R\* | the two features are independent: no correlation to estimate away from 0 |
| | F | 3 (`ew_cov`, `_window`, `ew_cov_accumulator`) | R\* | none |
| | O | `test_second_opinion` (7: pandas `ewm` mean, var, cov, corr, bias both ways, offset 1e8, 400 rows; irregular clock with `times=`; numpy weighted cov; statsmodels `ccf`/`acf` for `lags=`; scipy for `mahal`), `test_river` (2) | R, but the pandas comparison draws two independent features (true correlation 0) | a correlated, weighted stream with a gap held to numpy at nonzero cov and corr, on the clock |
| | C | Rust probe D; sweeps D | D | none |
| `marginal` | G | 2 (`marginal`, `_paths`) | R\* | none |
| | F | 10 (`marginal_lags`, `_bins`, `_bins_huber`, `_bins_refit`, `_bins_refit_done`, `_bins_refit_first_value`, `_bins_refit_first_value_frozen` (8 and 16 rows before), `_bins_huber_anchored`, `_window`, `_window_lags`) | R\* | none |
| | O | `test_second_opinion` (7: pandas pairwise, numpy window and decay, `binned_statistic`), `_grids.marglag_acf` (5) and `marglag_ccf` (statsmodels `acf`, `ccf`), `marginal_bin_huber`, `marginal_bin_refit` (statsmodels) | R (`y = 0.7 x + noise`, 600 rows) | `serial_rule="geometric"` and the bin edge rule: a longhand only (no library places edges this way) |
| | C | Rust probe D; sweeps D | D | none |
| `kmeans` | G | 2 (`kmeans`, `_seeding_and_metric`) on iid uniform features (D: no clusters, a flat objective), and `golden_structured.rs`'s `kmeans_blobs` (three blobs of 60 rows, the cluster and the distance at six rows, the blobs asserted found) | D and R\* | none |
| | F | 4 (`kmeans`, `_warming` (6 rows), `kmeans_spans`, `_spans_warming` (6)) | R\* (two regimes shifted by 2.0 are separated); the warming cases D on purpose | none |
| | O | `truth_recovery.kmeans_blobs` (50 seeds: sklearn `KMeans` inertia), `kmeans_franti` (7 sets, Fränti's A and S), `second_opinion_seeding.seeding_kmeanspp` (sklearn `kmeans_plusplus`), `convergence_iterations.kmeans_lloyd` | R | `init="farthest"`: no library has it |
| | C | Rust probe D (no structure); proptest D; a three-blob Lloyd test in `model_contract.rs` R | D / R | none |
| `micro` | G | 2 (`micro`, `_on_the_clock`) | D: iid uniform, so every row is its own micro-cluster or none | blobs and a dense core |
| | F | 1 | R\* | none |
| | O | `truth_recovery.micro_blobs` (50 seeds, sklearn ARI), `micro_franti` (S1), `test_micro` (scipy hierarchy, `csgraph`, `spatial` for the macro step) | R | none |
| | C | Rust probe D | D | none |
| `ew_class` | G | 2 (`ew_class`, `_windowed`); the first runs full, shared and diagonal | R\* (a label cut from `y`, so the classes differ) | none |
| | F | 3 (`ew_class`, `_diag`, `_window`) | R\* | none |
| | O | `test_ew_class` (classes drawn from multivariate normals: sklearn `GaussianNB`, scipy `multivariate_normal` at numpy's weighted covariance), `test_second_opinion` (2), `_windows` (full, diag, tied at `half_life=inf`) | R | none |
| | C | Rust probe D; sweeps use a coin-flip label (D: no signal) | D | none |
| `seqtest` | G | 1 (both slots) | R\* | none |
| | F | 1 | R\* | none |
| | O | `test_seqtest`: `seqtest_ref` (a scalar replay), the closed form `2^n B(n_pos + 1/2, n_neg + 1/2) / pi` through `math.lgamma`, Type I error over 20,000 fair-coin and 20,000 dependent-null streams, power at 60% and 55% | R, **no third-party library** (confseq does not build) | the KT closed form through `scipy.special.betaln` instead of `math.lgamma`, on a realistic residual-sign stream |
| | C | Rust probe D (no sign structure by design); `test_seqtest` edge cases | D | none |
| `deco` | G | 2 (`deco`, `_loglik`) | D: iid uniform features, a true equicorrelation of 0 (no correlation to track) | a one-factor stream at a nonzero level |
| | F | 1 | R\* | none |
| | O | `test_deco` (longhand pair sum, `ew_cov`'s mean form to the bit, scipy `multivariate_normal` for `loglik`, dense and blocked: a one-factor stream of 800 rows at a given rho), `truth_recovery.deco_level` (50 seeds) | R, **third party only for `loglik`** | the estimate and its dynamics: a longhand (no library computes DECO's `u`; `dccmidas` runs another model) |
| | C | Rust probe D | D | none |
| `corrchange` | G | 2 (`corrchange`, `_sequential`) | D: no correlation break, the detector reports the statistic of a null stream | a stream with a break |
| | F | 4 (`_monitor`, `_window`, `_permute_every`, `_sequential`) | R\* (two regimes) | none |
| | O | `truth_recovery.corrchange_break` (50 seeds: size and power), `test_family_f_grids` (4), `test_corrchange` (statsmodels `S_hac_simple` for `bandwidth`), `corrchange.n_perm` (numpy), `second_opinion_regimes` | R | `perm_block=` and `change_action="reset"`: a limit and self-comparison only |
| | C | Rust probe D | D | none |
| `bocpd` | G | 3 (`bocpd`, `_on_the_clock`, `_emissions`) | D: no change point in the stream | a stream with a level shift |
| | F | 3 (`bocpd`, `_warming` (6), `_warm_span` (6)) | R\*; the warming cases D on purpose | none |
| | O | `second_opinion_bocpd.bocpd_pruning`, `second_opinion_regimes.bocpd_priors` (bayesian-changepoint-detection), `truth_recovery.bocpd_shift`, `bocpd_tcpd`, `bocpd_tcpdbench`, `_robust` (TCPDBench's published F1), `test_bocpd` (scipy `multivariate_t` for `covariance_type="full"`) | R | `max_run` fold (no library truncates this way) |
| | C | Rust probe D | D | none |
| `hmm` | G | 2 (`hmm`, `_tvtp`) | D: one Gaussian cloud, two states with a flat likelihood, `warm_rows = 10` | a two-regime stream |
| | F | 6 (`hmm`, `_diag`, `_full`, `_full_narrow`, `_full_warming`, `_warm_span` (8)) | R\* (two regimes of 12 rows); `_warm_span` D on purpose | none |
| | O | `second_opinion_hmm` (hmmlearn: filter, loglik, batch limit, online EM), `second_opinion_regimes.hmm_tvtp` (statsmodels `MarkovRegression`), `truth_recovery.hmm_regimes` (50 seeds), `test_family_f_grids` | R | none (overlapping states converge slowly; documented, not a gap) |
| | C | Rust probe D | D | none |
| `rcov` | G | 3 (`rcov`, `_preavg`, `_psd`) | D: iid uniform returns, nothing to estimate but a diagonal | none beyond a correlated tick stream |
| | F | 1 | R\* | none |
| | O | `test_rcov` (hfhd `mrc` for `preavg`, `krvm` for the kernel at every jitter; numpy for the plain block), `truth_recovery.rcov_plain` | R | the subsampled and two-scale noise and IV estimates: a longhand |
| | C | Rust probe D | D | none |
| `audit` | G | 1 | R\* with edge values injected (a null, an infinity) | none |
| | F | 3 (`audit` (300), `_exact`, `_scaled`) | R\* | none |
| | O | `test_second_opinion_audit` (numpy `corrcoef` and `==` over the rows both columns hold: 3,000 dirty rows), `test_second_opinion` (scipy moments, statsmodels Dickey-Fuller, numpy median and MAD from exact counts), `test_audit` | R | none |
| | C | Rust probe D | D | none |
| `hawkes` | G | 1 (+ a window case) | D: events on a unit lattice, no self-excitation | a simulated self-exciting stream |
| | F | 3 (`hawkes` (40), `_window` (45), `_clock_window`) | D: same stream | none for forms |
| | O | `test_second_opinion_hawkes` (tick's least-squares contrast on tick-simulated streams, 3.9e-14) | R | the weighted fit: tick takes no weights, so the decay form is a longhand |
| | C | Rust probe D | D | none |
| `garch` | G | 6 (`garch`, `_targeted`, `_gjr_t`, `_skewt`, `_egarch`, `_figarch`) | D: the 60-row stream repeated, not a variance process | task 548 |
| | F | 7 (`garch`, `_started`, `_targeted`, `_skewt`, `_egarch`, `_figarch`, `_span`) | `garch` and `_span` D (40 rows, before the start); the rest R\* | task 548 |
| | O | `test_second_opinion_garch` (arch's recursions) | R | left to task 548 |
| | C | Rust probe D | D | left to task 548 |

## 4. The diagnostics and helpers of `rheon-core`

These are not `ModelState` variants. Each is held by an oracle in
`tests/test_second_opinion_diagnostics.py` or beside it, on a seeded,
identified stream (`_calibration_rows`, `_break_rows`, `_sandwich_rows`,
`_spec_rows`); their goldens, where they have one, are the same
60-row stream and their contract tests the noiseless line. One line each,
without a row a kind:

| component | realistic oracle | missing |
|---|---|---|
| `Calibration` | statsmodels OLS for Mincer-Zarnowitz, sklearn `r2_score` | none |
| `Breaks` (CUSUM, Wald) | `online_diagnostics.cusum_size` (7 cases), `breaks_memory`; the Wald distance by a longhand | no library computes the online CUSUM of the predictive residual |
| `Sandwich` | statsmodels HAC and HC covariances | none |
| `Influence` | statsmodels `OLSInfluence`, `WLS` leverage | none |
| `Specification`, `Tails` | statsmodels `acorr_ljungbox`, `jarque_bera`, scipy skew and kurtosis | none |
| `Drift` | river Page-Hinkley, ADWIN, DDM, EDDM, KSWIN (`test_second_opinion_drift*`) | none |
| `Conformal` | a replayed recursion only (`conformal.coverage_rates`: coverage over 50 streams) | no library has this step: realistic coverage held, no library's number |
| `FeatureHealth`, `Runs`, `Spread`, `Gaps`, `Window` | definitions: numpy, pandas and the brute force (`windows.rs`'s `brute()`) | none |

## 5. What the realistic cases start from

In order, the options and models whose only tests are degenerate, or whose
realistic test is not an oracle:

1. **The golden file's structure-finding models** (`kmeans`, `micro`, `hmm`,
   `bocpd`, `corrchange`, `deco`, `rcov`, `hawkes`): a second signature on
   a stream with structure (blobs, a level shift, a correlation break, a
   self-exciting stream), so a mutant that moves a number only on data with
   something to find moves a number `cargo mutants` sees.
2. **`ew_cov` at a nonzero correlation** on a weighted, gapped stream, held
   to numpy.
3. **`huber(feature_sets=, ridge_scale="sum")`** and the cadence of `huber`
   and `quantile`: held by reduction only.
4. **`sgd` and `pa` with `standardize=True` or constraints**: `sgd_ref` and
   one projection.
5. **`seqtest`'s closed form through scipy**, **`deco`'s estimate**,
   **`corrchange(perm_block=)`**: no library found; a longhand stays.
6. **A noisy contract line**: the contract probe's rows with noise, beside
   the noiseless line, so the spread outputs are read at a nonzero value.

## 6. Cases added

Each realistic case lands as one commit per model and is listed here with
its sha.

(none yet)
