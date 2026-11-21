# Oracle search for the queued builds, 2026-11-21 (request C12)

The standing instruction is "Always look for oracles". This record covers every OPEN task
numbered 485 to 544 on `origin/main` (`docs/PLAN.md`, read 2026-11-21) whose oracle is written as
"the definition", "by hand", "from the paper", or names no library. For each quantity it
names the best third-party oracle found, run on ONE small case against the paper's
definition written by hand. The entry point is `scripts/oracle_search.py`; each check is one
function, run with `python scripts/oracle_search.py <check> ...` (no argument: every check,
17 s). Nothing here changes the package, its dependencies or any config.

## 1. The open tasks with no library named, and the quantity of each

| Task | Plan text | Quantity to be checked |
|---|---|---|
| 493, 514 | DCC / ADCC: "the definition from Engle (2002) written by hand and a batch maximum likelihood of it by scipy" | the `Q` recursion with given `(a, b, g)`, `R_t`, the Gaussian log-likelihood |
| 499 | "the definitions from the papers and any library 494 finds" | RV, bipower and tripower variation, MedRV, quarticity, realized kernel, two-scale RV, pre-averaged RV, BNS jump test, Hayashi-Yoshida covariance, time-shifted HY lead-lag |
| 501 | "the definition and a batch fit (scipy.optimize), and any library 494 finds" | expectile regression (asymmetric least squares) |
| 504 | "the definitions and any library 494 finds" | LinUCB, Thompson sampling scores |
| 507 | "for the random cut forest, the definition with a seeded tree and a library 494 finds (rrcf)" | CoDisp of a tree, the forest score |
| 508 | "river's drift detectors on the same stream" (river has no tabular CUSUM) | the tabular two-sided CUSUM (Page 1954), Page-Hinkley |
| 520 | "Spiegelhalter's calibration test for logistic fits" (no library named) | the Spiegelhalter z statistic and p-value |
| 521 | "the definition plus statsmodels' HAC Wald" | Giacomini-White conditional test; Diebold-Mariano with HLN (`dieboldmariano` is named); the anytime-valid test (MAPIE `RiskMonitoring` named) |
| 523 | "the reference implementations the scan names" | BOCPD run-length posterior, constant and run-length-dependent hazards |
| 543 | not on `origin/main` (the last task there is 539); read as "chained models" | one model's pre-update prediction fed to the next as a feature |

Tasks 485 and 494 are scans (no quantity). Tasks 528 to 539 are infrastructure or policy; the
oracle of a merge (528) is the single-process run itself, of a gate (537 to 539) the contract
tests; no library computes them, and none was sought. The other open tasks 486 to 497, 502,
503, 505, 506, 509 to 513, 515 to 519, 522, 524, 525 and 536 name a library already.

## 2. Result table

"Gap" is the largest gap to the paper definition on the one case (relative unless stated).
Licence and release date are PyPI's (the GitHub API is not reachable from this session, so
commit dates are not recorded; the release date is the maintenance signal).

| Task | Quantity | Best oracle | Version | Licence | Released | Gap to the definition |
|---|---|---|---|---|---|---|
| 508 (a) | tabular two-sided CUSUM, fixed target `mu0`, `k` | R `qcc::cusum` (also `pyspc.cusum`) | qcc 2.7; pyspc 0.4 | GPL-2; GPLv3 | pyspc 2016-09 | 0 (both, C+ and C-, 300 rows) |
| 508 | Page-Hinkley (running mean, both sides) | `river.drift.PageHinkley` | 0.27.0 | BSD-3 | 2026-11-13 | 3.6e-15 (the statistic, from its private sums) |
| 508 | one-sided CUSUM, running-mean reference | `frouros` CUSUM | 0.9.0 | BSD-3 | 2024-10-05 | 0 |
| 499 | RV, BPV, MedRV, TPQ, Parzen realized kernel, two-scale RV | R `highfrequency` (`rRVar`, `rBPCov`, `rMedRVar`, `rTPQuar`, `rKernelCov`, `rTSCov`) | 1.0.3 | GPL | CRAN | RV 0, BPV 1e-16, MedRV 1e-16, TPQ 1.2e-15, RK 2e-16, TSRV 1.3e-6 (with its `1/(1 - nbar/n)` factor) |
| 499 | the same, in Python | `realized-library` | 0.1.2 | MIT | 2025-08-05 | RV 0, BPV and TPV 1e-15 with `correct_scaling_bias=False` (default adds `n/(n-1)`-type factors: 2e-3, 4e-3), MedRV 8e-6, Parzen RK 1.2e-2 (matches no plain convention; the `k((h-1)/H)` form is closest) |
| 499 | the same, in Python | `mfe-toolbox` | 0.1.2 | MIT | 2026-09-03 | RV 0, BPV 1e-16, MedRV 1e-16; TPQ off by a factor `n` (gap 0.998); TSRV returns the noise variance (gap 0.9985). Not an oracle for those two |
| 499 | BNS jump test | R `highfrequency::BNSjumpTest` | 1.0.3 | GPL | CRAN | `linear` 5e-15; `ratio` 0.35 on one case (its ratio form differs from the BNS 2006 text; unresolved) |
| 499 (b) | Hayashi-Yoshida covariance | `mfe-toolbox` `realized_hayashi_yoshida` | 0.1.2 | MIT | 2026-09-03 | 8e-15 (60 and 45 ticks) |
| 499 (b) | time-shifted HY lead-lag (Hoffmann-Rosenbaum-Yoshida 2013) | `lead-lag` (`LeadLag`, Cython `shifted_modified_hy_estimator`) | 2.3 | MIT (GitHub; PyPI says "Open Source") | 2022-12-20 | the argmax lag agrees (21 ms; truth 20 ms, one-ms grid); the contrast curve differs by up to 5.2e-2 (correlation 0.9993), cause not found |
| 501 | expectile regression | `pygam.ExpectileGAM` with linear terms, `lam=0` | 0.12.0 | Apache-2.0 | 2025-12-18 | 2.1e-10 (coefficients) |
| 501 | expectile, intercept only | `scipy.stats.expectile` | 1.14.1 | BSD-3 | | 4.7e-9 |
| 493, 514 | DCC / ADCC `Q` recursion, given `(a, b, g)` | R `rmgarch::dccfit` (+ `rugarch`) | 1.4.3 | GPL | CRAN | 6e-16 (the recursion constant is the same at every `t`); its intercept `Qbar` differs from the sample moment by 1.1e-5 (DCC) and 2.1e-3 (ADCC), and its `Q_1` is not `Qbar` |
| 493 | DCC log-likelihood, `mgarch` (PyPI) | `mgarch` | 0.3.0 | MIT | 2022-07-07 | does not agree (3.96e3 on a log-likelihood of 2.59e3): `log(D_t.sum())`, `reshape` for `Qbar`. Not an oracle |
| 504 | LinUCB (disjoint) scores | `mabwiser` | 2.7.4 | Apache-2.0 | 2024-08-30 | 8.9e-16 |
| 504 | LinUCB in `river` | `river.bandit.LinUCBDisjoint` | 0.27.0 | BSD-3 | 2026-11-13 | the mean is exact (4e-16); its bonus is the predictive sigma, which carries the `1/beta` noise term, so it is not Li et al.'s `alpha sqrt(x'A^-1 x)` (gap 1.0 in sigma squared) |
| 507 | CoDisp of an RRCF tree | `rrcf` | 0.4.4 | MIT | 2023-04-30 | 0 (60 points); the random cuts are not compared; needs `setuptools<81` |
| 520 | Spiegelhalter z and p | R `rms::val.prob` (`S:z`, `S:p`) | 6.8-1 | GPL | archive | 4e-16 (z), 8e-17 (p) |
| 521 | Diebold-Mariano, HLN, t reference, one-sided | `dieboldmariano` | 1.1.0 | MIT | 2025-01-09 | 5e-15 (statistic), 7e-15 (p), h = 3 |
| 521 | Giacomini-White, `q = 1` | `feval` `gw` | 0.0.2 (GitHub only) | MIT | | 3e-14 |
| 521 | Giacomini-White, `q = 2` (instruments `1`, `d_{t-1}`) | `feval` `gw` | 0.0.2 | MIT | | wrong: S = -8.9 against 9.06 (it keeps only the first coefficient of the regression); its `tau = 2` with `covar_style="sample"` raises. Not an oracle for `q > 1` |
| 521 | anytime-valid test of the mean loss difference | `confseq` | 0.0.11 | MIT | 2023-01-26 | NOT RUN: the C++ build failed in this venv |
| 523 | BOCPD, run-length-dependent hazard | `bayesian-changepoint-detection` (PyPI) | 0.2.dev1 | MIT | 2019-08-12 | 2.2e-16 (logistic hazard, Gaussian normal-gamma) |
| 523 | negative-binomial hazard | the same package, GitHub master (torch rewrite) | master | MIT | | 7.7e-10 (float32 output) |
| 523 | BOCPD, `bocd` (PyPI) | `bocd` | 0.1.2 | MIT | 2019-09-19 | 0.39 in the run-length posterior: its `beta` update reads `kappaT` where Murphy has `betaT`. Not an oracle |
| 543 | chain: A's pre-update prediction feeds B | `statsmodels.RecursiveLS` run twice | 0.15.0 | BSD-3 | | 1.7e-6 (A), 1.9e-6 (B): the diffuse prior `P_0 = 1e6 I` against the library's, so pass the same prior |

### Tasks with NONE (no library computes the quantity at all)

| Task | Quantity | Closest alternative and gap |
|---|---|---|
| 499 | pre-averaged RV (Jacod et al. 2009), as a library function | `mfe-toolbox` `realized_preaveraged_variance` and `realized-library` `utils.preaverage` exist but were not checked here; `hfhd.preaverage` fails on numpy 2 / pandas 3 (last release 2020-08). R `highfrequency` 1.0.3 was not searched for a pre-averaged variance. The definition by hand stays the oracle until one of these is checked |
| 499 | the BNS `ratio` statistic | R `highfrequency` differs from the BNS 2006 text by 0.35 on the case; use the `linear` form (5e-15) |
| 507 | the forest score of an RRCF with the same seed | none; `rrcf`'s tree is stochastic and `river` has no RRCF. Per-tree CoDisp is exact (above) |
| 521 | Giacomini-White with `q > 1` | none correct in Python; R `forecastdom::gw_test` is the candidate (named by a web search, not run); closest: the definition by hand, and `feval` at `q = 1` |
| 543 | a "chained model" as a library object | none; the chain is two oracles in sequence (above) |
| 528 to 539 | merges, gates, policies | not oracle questions |

## 3. State each online form needs (inferred; the task texts give none)

- Tabular CUSUM: `C+`, `C-` (two f64) plus the given `mu0`, `k` (and a reset rule); Page-Hinkley: running mean, count, two sums, `min`, `max` (six f64).
- Realized measures: the last one or two absolute returns (BPV, TPQ, MedRV: two), the running sums; a kernel of bandwidth `H`: the last `H` returns and `H + 1` autocovariances; TSRV: `K` offset grids.
- HY covariance: per asset the last tick time and price, and the open intervals of the other asset still overlapping (bounded by a window, PLAN 537); the lead-lag form keeps one set of open intervals per lag on the grid.
- Expectile: the weighted Gram (`k (k + 1) / 2 + k` numbers) as in `rls`.
- DCC: `Q` (`N (N + 1) / 2`), the last `eps` and `n`, `Qbar`, `Nbar`.

## 4. Proposed tolerances

| Check | Tolerance | Observed |
|---|---|---|
| closed-form sums (RV, BPV, MedRV, TPQ, Parzen RK, CUSUM, Page-Hinkley, DM, Spiegelhalter, BOCPD, rrcf, LinUCB, HY covariance) | relative 1e-12 | at most 7e-15 |
| `Q` recursion against `rmgarch` | absolute 1e-12 on the implied constant | 6e-16 |
| expectile regression against `pygam` | 1e-8 on the coefficients (a convergence tolerance, set both sides to 1e-12) | 2e-10 |
| TSRV against R | 1e-5 (small-sample factor) | 1.3e-6 |
| chain against `RecursiveLS` | 1e-5 absolute when the prior is `P_0 = 1e6 I`; 1e-12 when `rheon`'s prior is set equal and the first `2 k` rows are skipped | 1.9e-6 |
| HRY lead-lag | the argmax lag equal; contrast 6e-2 against `lead-lag`, 1e-12 against the hand double sum | 5.2e-2 |

## 5. Method

One seeded case per quantity (sizes in the script). The definition is written from the paper in
a few lines in `scripts/oracle_search.py` (`rv_def`, `bpv_def`, `tpq_def`, `medrv_def`,
`tsrv_def`, `hy_def`, `shifted_hy_contrast`, `cusum_def`, `page_hinkley_def`, `bocpd_def`,
`dcc_loglik_def`, `linucb_def`, `dm_def`, `spiegelhalter_def`, `rls_def`) and compared to the
library's output on the same data; where a library has a convention that differs (a
small-sample factor, a kernel argument), the variants are shown rather than one picked. R
runs by `Rscript` from the script (no rpy2 install was needed; `rpy2` would call the same
packages).

Environment (kept at `/home/user/venv-c12`, R libraries at `/home/user/Rlib`): Python 3.12.3;
numpy 2.x, scipy 1.14.1, pandas 3.1.1, statsmodels 0.15.0; R 4.3.3 from Ubuntu with
`highfrequency` 1.0.3, `qcc` 2.7, `spc` 0.7.2, `rugarch`, `rmgarch` 1.4.3, `expectreg` 0.54,
`rms` 6.8-1.

Install quirks that a dev-group addition would meet:
`lead-lag` needs `--no-deps --no-build-isolation` and Cython (its `pyrasta` dependency needs GDAL);
`realized-library` imports `scipy.integrate.simps` and `trapz`, removed from current scipy (the
script shims them); `rrcf` imports `pkg_resources` (`setuptools<81`); `feval` is on GitHub only;
`hfhd` fails on numpy 2 / pandas 3; `lead-lag` reads the index as nanoseconds, which pandas 3
breaks unless cast to `datetime64[ns]`; `rmgarch` needs Rcpp 1.1.1 or newer (Ubuntu has 1.0.12);
`rms` current needs a newer R, so 6.8-1 from the CRAN archive was used; the torch rewrite of
`bayesian-changepoint-detection` (GitHub master) needs torch.

## 6. Sources

- Page 1954, "Continuous inspection schemes"; Montgomery, *Introduction to Statistical Quality Control* 6e, 9.1.2 (tabular CUSUM). `qcc`: https://cran.r-project.org/package=qcc ; `pyspc`: https://pypi.org/project/pyspc/ ; `river` Page-Hinkley: https://riverml.xyz/ ; `frouros`: https://pypi.org/project/frouros/
- Hayashi and Yoshida 2005; Hoffmann, Rosenbaum and Yoshida 2013 (arXiv 1303.4871); `lead-lag`: https://github.com/philipperemy/lead-lag (MIT) and https://pypi.org/project/lead-lag/ ; `mfe-toolbox`: https://pypi.org/project/mfe-toolbox/ ; `realized-library`: https://pypi.org/project/realized-library/ ; `hfhd`: https://hfhd.readthedocs.io/ ; R `highfrequency`: https://cran.r-project.org/package=highfrequency
- Barndorff-Nielsen and Shephard 2004, 2006; Andersen, Dobrev and Schaumburg 2012; Barndorff-Nielsen, Hansen, Lunde and Shephard 2008; Zhang, Mykland and Ait-Sahalia 2005; Jacod et al. 2009.
- Engle 2002; Cappiello, Engle and Sheppard 2006; `rmgarch`: https://cran.r-project.org/package=rmgarch ; `mgarch`: https://pypi.org/project/mgarch/
- Newey and Powell 1987; `pygam`: https://pygam.readthedocs.io/en/stable/api/expectilegam.html ; `scipy.stats.expectile`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expectile.html ; `expectreg`: https://cran.r-project.org/package=expectreg
- Li, Chu, Langford and Schapire 2010; `mabwiser`: https://github.com/fidelity/mabwiser ; `contextualbandits`: https://pypi.org/project/contextualbandits/
- Guha, Mishra, Roy and Schrijvers 2016; `rrcf`: https://pypi.org/project/rrcf/
- Spiegelhalter 1986; `rms::val.prob`: https://search.r-project.org/CRAN/refmans/rms/html/val.prob.html
- Diebold and Mariano 1995; Harvey, Leybourne and Newbold 1997; `dieboldmariano`: https://pypi.org/project/dieboldmariano/ ; Giacomini and White 2006; `feval`: https://www.github.com/ogrnz/feval ; R `forecastdom`: https://cran.r-project.org/package=forecastdom ; `confseq`: https://github.com/gostevehoward/confseq
- Adams and MacKay 2007; Murphy 2007 (conjugate Gaussian); `bayesian_changepoint_detection`: https://github.com/hildensia/bayesian_changepoint_detection ; `bocd`: https://pypi.org/project/bocd/

## 7. Caveats

- One case per quantity: a gap of 1e-15 says the library and the hand definition agree there, not that every option of the library does. Options differ (small-sample factors, kernel arguments, the sign of `C-`); the script prints the variants.
- GPL and GPLv3 oracles (`pyspc`; the R packages) are test-only and never linked, but the dependency policy (`tests/test_dependency_policy.py`) decides whether they may enter the dev group, and R is a new system requirement for CI. The MIT, BSD and Apache candidates above need no such decision. Every R check has a Python or by-hand twin or a closed form, so R can be dropped without losing a quantity except the DCC `Q` recursion, the Spiegelhalter test and the `highfrequency` realized kernel and TSRV, which have R only.
- The R `rHYCov` was tried on merged returns and did not match the double sum (28%); its input convention was not resolved, so no HY claim rests on it.
- Not run: `confseq` (build failed), `contextualbandits`, `vowpalwabbit`, `hfvol`, `datasketch`, `hmmlearn`, `filterpy` (installed where they built, but the tasks that name them did not need a new oracle), R `forecastdom`, `ocp`, `changepoint`.
- Release dates stand for maintenance; GitHub commit dates were not reachable. `bocd` (2019), `bayesian-changepoint-detection` on PyPI (2019), `pyspc` (2016), `hfhd` (2020) are unmaintained.
- `lead-lag`'s contrast gap of 5.2e-2 to the hand double sum has no found cause; before the library is relied on for the HRY contrast, compare its Cython kernel to the hand sum on a grid with no tied ticks.
- PLAN 543 does not exist on `origin/main`; the chained-model check is the reading stated in section 1.
