# Bipower variation and the jump tests against third-party oracles (PLAN task 499, request A19)

*Status, 2026-11-21: a record, in progress for request A19 (branch
`vm/study-499b`). Part 1 is done: the oracle search (§2), the paper's
definitions (§3) and bipower variation and tripower quarticity against every
candidate that computes them (§4). Parts 2 (the ratio test and the Lee and
Mykland statistic), 3 (the online forms and their state) and 4 (the `tick`
Hawkes check, the proposed tolerances, the conclusions) are not yet run; their
sections say so. Nothing in the package or the Rust crates was changed. The
runnable study is `scripts/study_499_jumps.py` (`part1`); it is Python, plus R
through rpy2. This is not the realized-kernel study of the same task number on
branch `vm/study-499`; the two share nothing but the task.*

## 1. The question

Task 499 asks for realized-volatility estimators and the jump tests built on
them. The user's rule is "always look for oracles". Before any Rust:

1. Which third-party code computes bipower variation (BV), tripower quarticity
   (TQ), the Barndorff-Nielsen and Shephard (BNS) ratio jump test and the Lee
   and Mykland (LM) per-return statistic, and under which licence?
2. Does each equal the definition in the paper, to floating precision, and
   where it does not, is the departure a convention, a scaling constant or a bug?
3. What are the online forms (one pass; decay; a window in rows and in clock
   time; a zero-weight row learns nothing), and how large is their state?

## 2. The oracle search

Searched (2026-11-21, from this VM): PyPI by exact-name probe (about 30 guessed names
such as `highfrequency`, `hf-econometrics`, `jumptest`, `lee-mykland`,
`realized-variance`) and by web search ("bipower variation realized jump test
Python", "Lee Mykland jump test Python implementation github", "hf.econometrics
YalDan", "CRAN highfrequency rBPCov BNSjumpTest", "CRAN realized bipower
tripower quarticity jump test alternatives"); CRAN through the same searches;
GitHub through `git clone` (the GitHub API and tarballs are closed to this
session). PyPI has no search endpoint that works from here, so a package whose
name and summary no probe or search hit would be missed: the list is what was
found, not a proof of absence.

| Candidate | Version | Licence | What it has of the four | Run? |
|---|---|---|---|---|
| R `highfrequency` (CRAN) | 1.0.3 (2026-01-08) | GPL (>= 2) | `rBPCov` (BV), `rTPQuar` (TQ), `BNSjumpTest` (BNS test, linear and ratio), `intradayJumpTest` (spot-statistic jump test of the form L(t) = (R(t) - mu(t)) / sigma(t); there is no function called `LMtest` in 1.0.3) | yes, through rpy2 |
| R `yuima` (CRAN) | 1.15.34 (2025-12-14) | GPL-2 | `bns.test` (BNS ratio test), `lm.jumptest` (LM 2008), `minrv.test`, `medrv.test` | found after the first install; part 2 |
| R `hf.econometrics` (GitHub YalDan) | 0.2.0, commit `7322015` (2026-11-09) | MIT + file LICENSE | `iv_bipower` (BV, with an `adjust` flag); no TQ and no BNS test; its `lm_test` is Lee and Mykland (2012), the noise-robust test with pre-averaging, a different statistic from LM (2008) | BV yes; LM (2012) not comparable, see §2.1 |
| PyPI `hfvol` | 0.1.0 | MIT | `bipower_variation`, `jump_variation` (RV - BV, floored at 0); no TQ, no BNS test, no LM statistic (its sktime transformers are wrappers); also the Hawkes code (part 4) | yes |
| PyPI `mfe-toolbox` | 0.1.2 | MIT | `realized_bipower_variation`, `realized_tripower_quarticity`, `bns_jump_test`; a port of the Oxford MFE Toolbox's measures (the port is a different codebase) | yes |
| PyPI `realized-library` | 0.1.2 | MIT | `bipower_variation`, `multipower_variation` (TQ as m = 3, r = 4), `bns_test` (linear, ratio, adjusted ratio), `lm_test` (LM 2008, max statistic), also ABD, JO and ASJ tests | yes; needs a scipy shim (§2.2) |
| PyPI `hffe` | 0.0.13 | MIT | `Stock.getRV`, `Stock.getBV`; a threshold jump separation (`separateReturns`, alpha n^-0.49 sqrt(BV x time-of-day)), which is neither BNS nor LM | BV yes; the separation is a different method |
| PyPI `highfrequency` | 0.1.0 | MIT | tick-data cleaning only (exchange selection, refresh time); no estimator | no estimator to run |
| PyPI `arch` | 9.0.0 | NCSA | no realized measures or jump tests (a search of its installed source for bipower, realized variance, tripower and jump-test finds nothing; it came in as a dependency) | no |
| PyPI `jumpdiff`, `jumpmodels`, `pyjump`, `hfhd` | 0.4.2, 0.1.1, 1.3, 0.1.4 | MIT, not stated, not stated, MIT | jump-diffusion estimation, a regime-switching model, an unrelated menu utility, high-dimensional integrated covariance (summaries only; not opened); none is these quantities | no |
| MATLAB MFE Toolbox (Sheppard) | - | - | the origin of `mfe-toolbox`'s functions; needs MATLAB | not run |
| R `RTAQ`, `realized` | - | - | removed from CRAN (the CRAN pages say so) | not run |

Other tools installed to run these: R 4.3.3 (Ubuntu 24.04 apt, GPL), rpy2
3.5.17 (GPLv2+; 3.6 needs R >= 4.5), numpy 2.5.5, scipy 1.18.1, mpmath 1.4.1
(BSD-3-Clause; it computes the exact reference), polars 2.1.0 and pyarrow
26.0.1 (the cached data), `tick` 0.8.0.2 (BSD-3-Clause; part 4).

The R packages are loaded into a private library (`/root/Rlib`), the
Python ones into `.venv-499` (excluded in `.git/info/exclude`). Nothing of
either is committed.

### 2.1 `hf.econometrics`: what it is for

Its description says so itself: the noise-robust Lee and Mykland (2012) test,
the Ait-Sahalia and Jacod tests and pre-averaged integrated variance. The LM
(2012) statistic standardises blocks of pre-averaged returns and is built for
prices with microstructure noise; LM (2008), the per-return statistic that
rheon's task asks for, is a different statistic with a different limit law
(Gumbel on the maximum of |L|). It is therefore not an oracle for the LM
statistic of §1, and it is not run on it. Its `iv_bipower` is the one piece
that is an oracle (§4). It takes log prices, not prices: passing prices gives
a bipower variation of the wrong series, which is how the first run of this
study went wrong (a unit error in the harness, not in the package).

### 2.2 The `realized-library` scipy shim

`realized-library` 0.1.2 imports `scipy.integrate.simps` and `trapz`, which
scipy removed in 1.14. The study aliases them to `simpson` and `trapezoid`
before the import (`_shim_scipy` in the script). Neither function is used by
the quantities here, so the shim does not touch a result; it is stated because
the package does not import without it on a current scipy.

## 3. The paper's definitions

Barndorff-Nielsen and Shephard (2004, "Power and bipower variation with
stochastic volatility and jumps", JFE 2(1)) and (2006, "Econometrics of testing
for jumps in financial economics using bipower variation", JFE 4(1)), `n`
returns `r_j` of a unit interval, `mu_p = E|Z|^p = 2^(p/2) Gamma((p+1)/2) /
Gamma(1/2)`:

```text
RV = sum_j r_j^2
BV = mu_1^-2 sum_{j=2}^{n} |r_j| |r_{j-1}|                         mu_1 = sqrt(2/pi)
TQ = n mu_{4/3}^-3 sum_{j=3}^{n} |r_j|^{4/3} |r_{j-1}|^{4/3} |r_{j-2}|^{4/3}
```

Neither paper has a small-sample factor such as `n/(n-1)` or `n/(n-2)`; the
`(1/n)^(1 - (r+s)/2)` factor of the general definition is 1 for BV and is the
`n` of TQ. BV converges to the integrated variance in the presence of finite-
activity jumps, TQ to the integrated quarticity.

The reference for every comparison below is these formulas evaluated in
40-digit arithmetic (mpmath) on the exact float64 returns the candidate is
given. `ulps` is `|candidate - exact| / ulp(exact)`. The paper's formula in
float64 numpy is itself a row of every table, so the floor of what "agrees to
floating precision" means is visible: summing `n` products and dividing by a
rounded constant costs several ulps.

Constants (`mu_1` and `mu_{4/3}` by their closed form agree with quadrature of
the definition to the 40 digits used):

| Constant | Source | Error in ulps |
|---|---|---|
| `mu_1` | `math.gamma` form | 1.45 |
| `mu_1` | `hfvol` (`sqrt(2/pi)`), `mfe-toolbox` | 0.45 |
| `mu_1` | `realized-library` (`mu_x(1)`) | 1.45 |
| `mu_{4/3}` | `math.gamma` form | 1.20 |
| `mu_{4/3}` | `mfe-toolbox` (`_MU43`) | 0.80 |
| `mu_{4/3}` | `realized-library` (`mu_x(4/3)`) | 0.20 |
| `mu_{4/3}^-3` | `highfrequency` (`(gamma(.5)/(2^(2/3) gamma(7/6)))^3`) | 6.46 |

## 4. Bipower variation and tripower quarticity (part 1)

*Data (all from `scripts/study_499_jumps.py part1`, 36 series, 7 s):*
synthetic returns of one day, `n` in {78, 390, 2340, 23400}, three seeds, each
without and with five planted jumps (a stochastic-volatility diffusion with an
intraday U-shape and a log-OU volatility factor, 1% a day; jumps at 10, 30, 50,
70 and 90% of the day of sizes +8, -10, +6, +12, -7 local standard deviations);
BTCUSDT 1-minute closes, ten days in January 2024 (1,439 returns a day), and
the ETCUSDT trades of 2024-01-02 resampled to the last price in 10 seconds
(8,639 returns) and in 1 minute (1,439). The candidates that take prices get
`100 exp(cumsum r)`, and every candidate and the reference see the returns
`diff(log(p))` of those same prices.

### 4.1 Bipower variation

| Candidate | Factor in its source | Largest gap to the paper | Largest gap after dividing out its factor |
|---|---|---|---|
| paper, float64 numpy | 1 | 4.4 ulps | 4.4 ulps |
| `hfvol` `bipower_variation` | 1 | 1.9 ulps | 1.9 ulps |
| `mfe-toolbox` `.value` | 1 | 1.9 ulps | 1.9 ulps |
| `mfe-toolbox` `.debiased_value` | `m/(m-1)`, `m = n-1` products: `(n-1)/(n-2)` | 1.3e-2 relative at `n` = 78 | 1.9 ulps |
| `realized-library` `bipower_variation`, default | `n/(n-1)` | 1.3e-2 relative at `n` = 78 | 3.8 ulps |
| `realized-library` `bipower_variation`, `correct_scaling_bias=False` | 1 | 4.0 ulps | 4.0 ulps |
| `realized-library` `multipower_variation`, m = 2, r = 2, default | `n/(n-1)` | 1.3e-2 relative | 4.9 ulps |
| `hffe` `Stock.getBV` | 1 | 1.9 ulps | 1.9 ulps |
| R `highfrequency` `rBPCov` | 1 | 1.4 ulps | 1.4 ulps |
| R `hf.econometrics` `iv_bipower`, `adjust = FALSE` | 1 | 1.4 ulps | 1.4 ulps |
| R `hf.econometrics` `iv_bipower`, `adjust = TRUE` | `n/(n-1)` | 1.3e-2 relative | 2.5 ulps |

Every candidate computes the paper's `BV`, with `mu_1^-2 = pi/2`, to within
5 ulps on all 36 series, real data included; the six candidates with no factor
in their source differ from one another by at most 5 ulps (the widest pair is
`hfvol` and `realized-library` without its factor: summation order and the
constant, `sqrt(2/pi)` against `Gamma` form). Three departures, all
conventions:

* `realized-library` applies `n/(n-1)` **by default** (`correct_scaling_bias
  = True`), so its default is above the paper's by 1/(n-1): 1.3% at 78 returns,
  0.07% at 1,439, 0.004% at 23,400. `hf.econometrics` has the same factor
  behind `adjust = TRUE` and says in its documentation that BNS (2004) does not
  have it.
* `mfe-toolbox`'s `.debiased_value` multiplies by `m/(m-1-skip)` with `m = n-1`
  the number of products, giving `(n-1)/(n-2)`, not the `n/(n-1)` of the other
  two (the sum has `n-1` terms, and `n/(n-1)` rescales it to `n`). Its `.value`
  is the paper's.
* `hffe` and `hfvol` (and the R functions) apply none.

### 4.2 Tripower quarticity

Four candidates compute it; `hfvol`, `hffe` and `hf.econometrics` have none.

| Candidate | Factor in its source | Largest gap to the paper | After dividing out its factor |
|---|---|---|---|
| paper, float64 numpy | 1 | 14.7 ulps | 14.7 ulps |
| `realized-library` `multipower_variation`, m = 3, r = 4, default | `n/(n-2)` | 2.6e-2 relative at `n` = 78 | 17.7 ulps |
| `realized-library` ..., `correct_scaling_bias=False` | 1 | 16.7 ulps | 16.7 ulps |
| R `highfrequency` `rTPQuar` | `n/(n-2)` | 2.6e-2 relative at `n` = 78 | 23.7 ulps |
| `mfe-toolbox` `realized_tripower_quarticity` | **1/(n-2)** | the value is 1/(n-2) of the paper's | 18.4 ulps |

* `realized-library` and R `highfrequency` follow the paper with an extra
  `n/(n-2)` (`realized-library`'s by default); the residual, 17.7 and 23.7
  ulps, is the cost of the 4/3 powers and the cube of the constant (the
  `highfrequency` constant alone is 6.5 ulps off).
* **`mfe-toolbox`'s tripower quarticity is wrong by a factor of `n-2`.** Its
  source is `n * mu_{4/3}^-3 * mean(...)` where the paper has `n * sum(...)`;
  the mean divides by `n-2`. Dividing the paper's value by `n-2` reproduces it to
  18.4 ulps on every series. The scale check below makes the same point without
  the paper: a quarticity estimator is of the order of the realized quarticity
  `RQ = (n/3) sum r^4` on a series without jumps, and `TQ / RQ` is 0.94 to 1.00
  for the paper, `realized-library` and `highfrequency` and 4.0e-5 to 4.3e-5
  (= 1/23,398) for `mfe-toolbox`. Its own `bns_jump_test` uses this value in
  `max(1, TQ / BV^2)`, where `TQ / BV^2` is about `1/n`: the maximum is always 1
  (part 2 measures what that does to the test).

Scale check, `TQ / RQ`, no jumps, n = 23,400, three seeds:

| Candidate | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| paper | 0.9869 | 0.9373 | 1.001 |
| `realized-library` (default) | 0.987 | 0.9374 | 1.001 |
| R `highfrequency` | 0.987 | 0.9374 | 1.001 |
| `mfe-toolbox` | 4.2e-5 | 4.0e-5 | 4.3e-5 |

### 4.3 What jumps do to the paper's estimators

The same diffusion with and without the five jumps (paper definitions; the
candidates agree with them above, so this is a property of the estimators):

| n | RV, with / without | BV, with / without | TQ, with / without |
|---|---|---|---|
| 390 | 2.05, 2.30, 2.14 | 1.32, 1.41, 1.29 | 1.99, 2.45, 1.54 |
| 2,340 | 1.16, 1.15, 1.20 | 1.027, 1.022, 1.055 | 1.04, 1.03, 1.30 |
| 23,400 | 1.018, 1.013, 1.017 | 1.003, 1.004, 1.006 | 1.002, 1.008, 1.015 |

(Three seeds each.) BV is not unmoved by jumps at a finite `n`: a jump of 10
standard deviations adds `|jump| |r|` twice, and at 390 returns the five jumps
raise BV by 29 to 41%, a sixth to a third of what they add to RV (RV rises by
105 to 130%), at 2,340 by
2 to 5% and at 23,400 by 0.3 to 0.6%. Tripower quarticity, which cubes the
neighbours' 4/3 powers, is moved more at small `n` (a factor 1.5 to 2.5 at
390). Both converge to their integrated targets as `n` grows, which is the
paper's result and not a defect of an implementation.

### 4.4 Findings of part 1

1. Six implementations (R `highfrequency` and `hf.econometrics`; `hfvol`,
   `hffe`, `mfe-toolbox`'s `.value` and `realized-library` without its
   factor) agree with the paper's BV to within 5 ulps on 36 series, so a
   rheon BV that is the paper's formula has an oracle at a tolerance of a few
   ulps of the sum (the proposal waits for the forms of part 3).
2. The conventions differ and must be chosen, not inherited: `n/(n-1)`
   (`realized-library` default, `hf.econometrics` `adjust`), `(n-1)/(n-2)`
   (`mfe-toolbox` debiased) or none (BNS 2004; R `highfrequency`, `hfvol`,
   `hffe`). For TQ, `n/(n-2)` (R `highfrequency`, `realized-library`) or none
   (BNS 2006). The paper's has none; the online form (§6, part 3) has to say
   which it is, since a factor of `n/(n-2)` is not available on a decayed
   window of weights.
3. `mfe-toolbox`'s tripower quarticity and so its jump test are not usable as
   an oracle: they are off by `n-2`.
4. Not verified in part 1: the BNS ratio test and the LM statistic (part 2),
   the online forms (part 3) and the Hawkes check (part 4).

## 5. The BNS ratio test and the Lee and Mykland statistic (part 2)

Not yet run.

## 6. The online forms and their state (part 3)

Not yet run.

## 7. Hawkes intensity against `tick` (part 4)

Not yet run.

## 8. Proposed tolerances and conclusions (part 4)

Not yet written.
