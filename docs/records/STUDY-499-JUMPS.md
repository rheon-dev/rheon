# Bipower variation and the jump tests against third-party oracles (PLAN task 499, request A19)

*Status, 2026-11-21: a record, in progress for request A19 (branch
`vm/study-499b`). Parts 1 and 2 are done: the oracle search (§2), the paper's
definitions (§3), bipower variation and tripower quarticity against every
candidate that computes them (§4), and the Barndorff-Nielsen and Shephard
tests and the Lee and Mykland statistic against theirs, with and without
planted jumps and on the Binance data (§5). Parts 3 (the online forms and
their state) and 4 (the `tick` Hawkes check, the proposed tolerances, the
conclusions) are not yet run; their sections say so. Nothing in the package or
the Rust crates was changed. The runnable study is `scripts/study_499_jumps.py`
(`part1`, `part2`); it is Python, plus R through rpy2. This is not the realized-kernel study of the same task number on
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
| R `yuima` (CRAN) | 1.15.34 (2025-12-14) | GPL-2 | `bns.test` (BNS linear, ratio and adjusted-ratio tests, any quarticity by its powers `r`), `lm.jumptest` (LM 2008), `minrv.test`, `medrv.test` | yes, through rpy2 (§5) |
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
  (§5.3 measures what that does to the test).

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
4. Not verified in part 1: the BNS ratio test and the LM statistic (§5),
   the online forms (part 3) and the Hawkes check (part 4).

## 5. The BNS tests and the Lee and Mykland statistic (part 2)

*Run: `scripts/study_499_jumps.py part2` (46 s: the 36 series of §4, 13 BNS
candidate settings against exact references, 10,000 Monte Carlo days for the
`mfe-toolbox` effect, 28 Lee and Mykland series and windows, 3,200 Monte Carlo
days for the Lee and Mykland windows). The study ran twice from the same seeds
and gave identical output.*

### 5.1 The definitions

**BNS** (the Nuffield working paper of 2 September 2004, eqs. 10-14, which the
2006 article revises; the article itself was not read), `n` returns, `delta =
1/n`, `vartheta = pi^2/4 + pi - 5 = 0.6090`, `BV` and `RV` as §3, and `QP = n
mu_1^-4 sum_{j>=4} |r_j||r_{j-1}||r_{j-2}||r_{j-3}|` the quadpower quarticity:

```text
G (linear)   = delta^-1/2 (BV - RV) / sqrt(QP)             -> N(0, vartheta)
H (ratio)    = delta^-1/2 (BV/RV - 1) / sqrt(QP / BV^2)    -> N(0, vartheta)
J (adjusted) = H with max(1, QP / BV^2) in place of QP / BV^2
```

All three are significantly *negative* under jumps, and divided by
`sqrt(vartheta)` they are N(0,1). The study's `Z` is the negative of that, so a
jump makes it positive (the sign of R `highfrequency`, `yuima` and
`realized-library`): `Z_lin = sqrt(n)(RV - BV)/sqrt(vartheta Q)`, `Z_rat =
sqrt(n)(1 - BV/RV)/sqrt(vartheta Q/BV^2)`, `Z_adj` with the `max`. `Q` is `QP`
(the paper) or the tripower quarticity `TQ` of §3 (Huang and Tauchen's choice,
the default of R `highfrequency`). No small-sample factor anywhere. The exact
reference is these formulas in 40 digits on the exact float64 returns, with the
conventions of the candidate's source applied to it (a "convention-adjusted"
reference), so that what is left is the candidate's arithmetic.

**Lee and Mykland** (2008, RFS 21(6), Definition 1, eqs. 7, 8, 12 and 13, read
in the author's PDF), window `K`, `c = sqrt(2/pi)`, `n` returns:

```text
L(i)    = |r_i| / sigma(i),   sigma(i)^2 = 1/(K-2) sum_{j=i-K+2}^{i-1} |r_j||r_{j-1}|
          (the K-1 returns just before i; the current return is outside the window)
C_n     = sqrt(2 log n)/c - (log pi + log log n) / (2 c sqrt(2 log n)),  S_n = 1 / (c sqrt(2 log n))
T       = (max_i |L(i)| - C_n) / S_n  ->  xi,  P(xi <= x) = exp(-e^-x)
reject at significance alpha when T > beta* = -log(-log(1 - alpha))      (alpha = 1%: 4.6001)
```

`|L(i)|` alone is flagged at `i` when it exceeds `beta* S_n + C_n`. The paper
recommends `K` between `sqrt(252 n)` and `252 n` for `n` observations a day;
every `K` below is smaller than that, because a one-day series cannot be longer
than itself (stated where it matters, §5.5). `alpha = 5%` here, `beta* = 2.970`.

### 5.2 The BNS tests against the candidates (36 series)

Every row is the largest over the 36 series (synthetic without and with the
five jumps, `n` from 78 to 23,400; the Binance series). "Relative gap after its
conventions" is the gap to the paper's formula with the candidate's own
conventions applied, as §4's "after dividing out its factor". The
statistic is a difference, so near `RV = BV` its relative error is amplified by
`cond = (RV + BV)/|RV - BV|`, at most 3.4e3 here.

| Candidate (all positive under jumps) | Conventions found in its source | Departure from the paper | Relative gap after its conventions |
|---|---|---|---|
| R `highfrequency` `BNSjumpTest`, linear / ratio / ratio with `max = TRUE` (J), `IQestimator = "TP"` | quarticity is `rTPQuar`, with the extra `n/(n-2)`; BV with none | Z is smaller by `sqrt((n-2)/n)` (1.3% at 78, 0.26% at 390) when the `max` does not pick 1 | 4.1e-14, 9.3e-14, 9.3e-14 |
| R `yuima` `bns.test`, `type = "standard"` (G), `"ratio"` with `adj = FALSE` (H) and `adj = TRUE` (J); `r = rep(1, 4)` (QP, the default) and `r = rep(4/3, 3)` (TQ) | the factor in front is `sqrt` of the number of *prices* (`n + 1`), not of returns; quarticity and BV as the paper | Z is larger by `sqrt((n+1)/n)` (0.64% at 78, 0.13% at 390, 0.002% at 23,400) | 5.3e-14 to 9.4e-14 |
| `realized-library` `bns_test`, `"linear"` | the time step is taken from the timestamps in fractions of a 24-hour day (here `round(day/n)/day`); QP | 4.4e-13 from the rounding of the step to a nanosecond; otherwise the paper's G | 4.4e-13 |
| `realized-library` ..., `"adjusted-ratio"` | numerator `RV/BV - 1` (the reciprocal of the paper's `BV/RV`, a different statistic with the same limit); `max(1/t, .)` with `t` the span in days | Z is the paper's J (with QP) times `RV/BV`: 0.86 to 1.17 on the synthetic series, up to 3.3 on the Binance days | 5.1e-13 |
| `realized-library` ..., `"ratio"` | numerator `RV/BV - 1` and `(QP_raw/BPV_raw)^2` where the paper has `QP/BV^2` | **wrong by a factor of order `n`: Z is 110 to 4.2e4 times the paper's H** (7,500 to 31,000 on the synthetic days without jumps), so the test rejects every series | 8.4e-11 (to the formula as coded) |
| `mfe-toolbox` `bns_jump_test` | numerator `RV/BV - 1`; quarticity is the shipped TQ, which is `TQ/(n-2)` (§4.2), so `TQ/BV^2 ~ 1/n` and `max(1, .)` is always 1; two-sided p-value | see §5.3 | 4.0e-13 (to the formula as coded) |
| R `highfrequency` `BNSjumpTest`, linear with `max = TRUE` | not run: its source replaces `IQ` by `max(1, IQ/IV^2)` in the linear statistic's denominator, where the paper has `QP` | - | - |

What agrees: with its conventions stated, every candidate follows the paper to
1e-13 relative (well inside the conditioning of the difference), so the BNS
statistics of R `highfrequency` and `yuima` are oracles for a rheon test written
from the paper, `yuima`'s with no convention to undo apart from `n + 1`.
What does not: `realized-library`'s ratio test is wrong by a factor of order
`n`; `mfe-toolbox`'s (§5.3); and the sign and the numerator differ (`RV/BV - 1`
against `1 - BV/RV`), so the `Z` of `realized-library`'s adjusted test and of
`mfe-toolbox` are the paper's J times `RV/BV`, not the paper's J. The `log`
type of `yuima`, `log(RV/BV)`, is not in the paper and was not run.

Some values (`Z`):

| Series | R `highfrequency` J | `yuima` J, TQ | `mfe-toolbox` | `realized-library` adjusted |
|---|---|---|---|---|
| synthetic, no jumps, n = 390, seed 0 | 1.105 | 1.109 | 1.318 | 1.172 |
| synthetic, 5 jumps, n = 390, seed 0 | 8.080 | 8.111 | 15.906 | 14.04 |
| synthetic, 5 jumps, n = 23,400, seed 0 | 2.046 | 2.046 | 2.373 | 2.074 |
| BTCUSDT 1 min, 2024-01-02 | 1.675 | 1.677 | 2.128 | 1.689 |

(`realized-library` uses QP and the reciprocal numerator, so its column is not
comparable with the others to better than the quarticity estimators and the
numerators differ; the run prints every value.)

### 5.3 What the tripower-quarticity error does to `mfe-toolbox`'s test

Its `bns_jump_test` computes `Z = sqrt(n)(RV/BV - 1)/sqrt(vartheta max(1, TQ/BV^2))`
with `TQ` from `realized_tripower_quarticity`, which is the paper's `TQ/(n-2)`
(§4.2). `TQ/BV^2` is then about `1/n` and the `max` is always 1, so the test
divides by `sqrt(vartheta)` alone, the variance of a series with constant
volatility, instead of by the quarticity ratio (0.7 to 1.9 on the synthetic
series, up to 26 on the Binance days). Three comparisons, using the library's
own code with `TQ` multiplied by `n - 2` for "TQ fixed":

*On the series* (Z shipped, Z with TQ fixed, the paper's J with TQ):

| Series | n | shipped | TQ fixed | paper J | TQ/BV^2 |
|---|---|---|---|---|---|
| synthetic, no jumps, seed 0 | 390 | 1.318 | 1.165 | 1.108 | 1.28 |
| synthetic, 5 jumps, seed 0 | 390 | 15.91 | 13.19 | 8.10 | 1.45 |
| synthetic, 5 jumps, seed 0 | 23,400 | 2.373 | 2.070 | 2.046 | 1.31 |
| BTCUSDT 1 min, 2024-01-03 | 1,439 | 3.376 | 0.662 | 0.619 | 26 |
| BTCUSDT 1 min, 2024-01-05 | 1,439 | 5.775 | 1.333 | 1.192 | 18.8 |
| BTCUSDT 1 min, 2024-01-09 | 1,439 | 8.535 | 2.866 | 2.438 | 8.87 |
| ETCUSDT trades, 1 min | 1,439 | 1.179 | 0.707 | 0.690 | 2.78 |

*Decisions on the ten BTCUSDT days* (two-sided 5%, `|Z| > 1.96`, the library's
own rule): shipped, 7 days flagged; TQ fixed, 2 days (a third, 2024-01-04, is at
1.959); the paper's J (a one-sided test), 2 days. On the days with a big jump
(TQ/BV^2 of 9 to 26) the shipped statistic is 3.0 to 5.1 times the TQ-fixed one.

*Monte Carlo* (stochastic-volatility days of §4, five planted jumps of 6 to 12
local standard deviations, nominal 5%; 2,000 days at n = 78 and 390, 1,000 at
2,340; standard error of a rate about 0.5 points):

| n | rate under H0, two-sided: shipped / TQ fixed / paper J | one-sided (`Z > 1.645`): shipped / TQ fixed / paper J | sd of Z under H0: shipped / fixed / paper J | rate with five jumps |
|---|---|---|---|---|
| 78 | 11.4% / 10.1% / 7.4% | 15.2% / 14.2% / 10.3% | 1.29 / 1.20 / 1.05 | 100% for all |
| 390 | 8.5% / 6.5% / 5.6% | 10.2% / 8.0% / 6.8% | 1.15 / 1.04 / 1.02 | 100% for all |
| 2,340 | 8.0% / 4.9% / 5.1% | 8.3% / 6.5% / 5.7% | 1.12 / 1.00 / 1.00 | 99.7% / 99.5% / 99.5% |

The shipped test over-rejects by 1.5 to 1.6 times at 390 and 2,340 returns (8.5%
and 8.0% against 5.6% and 5.1%) because its denominator is too small;
correcting the quarticity brings the sd of Z to 1.00 at 2,340 returns and the
rate to 4.9%. What is left of the gap between "TQ fixed" and the paper's J
(6.5% against 5.6% at 390) is the numerator: `RV/BV - 1` against `1 - BV/RV`
(the reciprocal-numerator statistic with the correct TQ rejects exactly as
often as the "TQ fixed" one: 8.0% one-sided at 390, to the digit). The paper's
J itself is 5.6% two-sided at 390 and 5.1% at 2,340. Power against these
large jumps is 100% for every version, so this test cannot tell the versions
apart on power; smaller jumps were not run. **`mfe-toolbox`'s test is not
used as an oracle**: until its quarticity is corrected, it has neither the
paper's statistic nor its size.

### 5.4 The Lee and Mykland statistic against the candidates

Candidates: R `yuima` `lm.jumptest(K)` (returns `T` for the maximum);
`realized-library` `lm_test.compute(prices, K)` (returns `T`; `flags` gives the
per-return flags); R `highfrequency` `intradayJumpTest` (per-return `L'(i) =
c L(i)`-form statistics in `ztest` and a critical value; `T` is formed from
them). Series: the synthetic series of §4 at `n` = 78, 390, 2,340 and 23,400
(two seeds, without and with the five jumps, `K` in {10}, {10, 30}, {30, 100}
and {100, 300}: 28 cases), and the Binance series (`K` = 30 and 100: 24 cases,
paper, `yuima` and `realized-library` only: `highfrequency` needs trading hours).

*The paper's `T` in float64* is within 1.7e-14 of the 40-digit value on all 28
synthetic cases (exact `max` over the five largest `L(i)`), and unchanged by
rescaling the returns (1.4e-14).

*R `yuima` `lm.jumptest`.* Its source gives `L(i) = |r_i|/sqrt((pi/2) v_i)` with
`v_i` the mean of the `K-2` products ending at `|r_{i-1}||r_i|`: the current
return is **inside** its own volatility window (the products run to
`|r_i||r_{i-1}|`), the window is shifted by one against the paper's, and the
first `K-3` windows are filled from the first full one. The constants `C_n`,
`S_n` are the paper's, absorbed into `c`. Written from that source in numpy the
statistic reproduces `yuima`'s `T` to 1.8e-13 on all 52 cases (a relative 9e-15),
so this is what it computes. The departure from the paper's `T`:

* with the five jumps, `T` is 0.38 to 0.97 of the paper's (14.6 to 40.9 against
  21.6 to 48.6), because the jump sits in its own denominator (the shrinkage is
  strongest at small `K`);
* without jumps, `T` is lower too (-1.06 to 3.4 against 0.46 to 5.7);
* on the Binance series `T` is 0.42 to 1.06 of the paper's (24 cases).

All 14 jump series are still rejected, all at the 5% level; of the 14 no-jump
series the paper rejects 4 (small `K`, §5.5) and `yuima` 1. It cannot be used as
an oracle for the per-return statistic; it can for its own window if the rheon
statistic offers that window.

*R `highfrequency` `intradayJumpTest`.* Its spot volatility is `(pi/2) sum` of
the `K-1` products over `K` returns **including the current one** by default,
divided by `K-2`, not by the `K-1` it summed. With `dontIncludeLast = TRUE` the
window is `j-K+1 .. j-1`, which is the paper's: then the per-return statistic
equals the paper's `c L(i)` to at most 4 ulps over every return of every one of
the 28 cases (the paper side uses libm's `log`, because numpy's SIMD `log` differs
from libm's in the last bit at one price in 2,341 and a tiny return amplifies
that to 1e-11; R and `math.log` agree on all 2,341 prices of the case checked), and its `T`, formed with its
own `C_n`, `S_n` (it takes `n` as the number of prices, `n + 1`), differs from
the paper's by up to 0.042 (n = 78) and, with `n` returns, by 2e-14. At
`alpha = 0.05` its critical value is the paper's 5% threshold: it flags exactly
the same returns as the paper in all 28 cases (planted jumps found, and the
false flags of the paper's own small-`K` size, the same ones). So with that
option `highfrequency` is a second oracle, to the bit, for the per-return
statistic and the flags.

Two departures that are not options:

* the **default window includes the current return**: `T` is 0.38 to 0.96 of the
  paper's on the series with jumps, and it flags 3 or 4 of the 5 planted jumps at
  `K = 10` (n = 78 and 390) where the paper flags 4 or 5; its `T` never
  exceeds 2.97 on the 14 no-jump series (the paper's rejects 4 at 5%);
* the **critical value's argument**: `critical = C_n + S_n (-log(-log(1 - alpha)))`
  with the default `alpha = 0.95` ("confidence level" in its documentation) is
  `-log(-log(0.05)) = -1.097`, the point that 95% of the null distribution lies
  *above*, not the 5% point 2.970 the paper's formula gives for `alpha = 0.05`.
  Its default critical values are 2.14, 2.71, 3.26 and 3.86 (n = 78, 390, 2,340
  and 23,400, in the `c L` scale) against 3.52, 3.89, 4.29 and 4.76 for the 5%
  test: it would flag the largest return on about 95% of null days.

*`realized-library` `lm_test`.* The window is the paper's (the previous
`K`-price window, current return outside), the constants `C_n` and `S_n` are the
paper's, but the denominator is `bpv(window)`, which is `(pi/2) (K-1)/(K-2) sum`
of the products, a *variance* summed over the window, not the volatility
`sigma(i) = sqrt(mean)`: `L_i = r_i / BPV` has the units of one over a variance.
Its `T` is therefore not scale-free: 700 to 15,000 on all 28 synthetic cases (the
paper's 0.46 to 48), `T` falls by a factor of 10.1 to 12.2 when the returns are
multiplied by 10 where the paper's `T` does not move, and it rejects all 14
no-jump series. On the Binance days it is 260 to 6,800. It is not an oracle.

### 5.5 Size, with and without a constant volatility

The paper's `T` under H0 needs the volatility to be nearly constant over the
window. Rates of `T > 2.970` (nominal 5%), paper's window / `yuima`'s window /
`highfrequency`'s default window, and the mean of `T` (the Gumbel mean is 0.577):

| days | n | K | no jumps | mean of `T` | five jumps |
|---|---|---|---|---|---|
| constant volatility, 400 | 2,340 | 200 | 6.0% / 5.8% / 4.5% | 0.80 / 0.74 / 0.63 | 100% for all |
| constant volatility, 200 | 23,400 | 1,000 | 7.0% / 6.5% / 6.5% | 0.68 / 0.66 / 0.63 | 100% for all |
| stochastic volatility, 500 | 2,340 | 100 | 25.2% / 20.0% / 18.6% | 2.05 / 1.73 / 1.63 | 100% for all |
| stochastic volatility, 500 | 390 | 30 | 42.8% / 29.0% / 24.8% | 3.03 / 2.21 / 1.93 | 100% for all |

With constant volatility the paper's test is at 6.0% and 7.0% (standard errors
1.1 and 1.5 points) and `T` has the mean the limit law gives within the
finite-window bias, which is the check that the paper's formula is implemented
as written; the windows of `yuima` and `highfrequency` are within the same
noise. With the stochastic volatility of this study's days (log-volatility with
a half-life of `n/10` returns, so the window is far from constant) and a `K`
below the paper's range the test rejects 25% to 43% of null days, all three
windows alike (a window that includes the current return rejects less often, because the
denominator is inflated by the return itself): that is the
statistic's size distortion under volatility that moves faster than its window,
not an implementation error, and it is why a rheon test needs `K` stated beside
its null.

### 5.6 What part 2 settles

1. **BNS test.** R `highfrequency` (with its `n/(n-2)` on the tripower
   quarticity) and R `yuima` (with `n + 1` prices under the square root) follow the
   paper's G, H and J to 1e-13 relative; the paper has no small-sample factor, so
   a rheon test written from the paper differs from `highfrequency` by
   `sqrt((n-2)/n)` and from `yuima` by `sqrt(n/(n+1))` where the `max` does not
   pick 1, and the oracle test states which it checks against. `yuima` takes the
   quarticity as powers, so it gives both the paper's QP and Huang and
   Tauchen's TP.
2. `realized-library`'s ratio test and `mfe-toolbox`'s test are not oracles
   (a factor of order `n`; the tripower error of §5.3, with 7 of 10 BTCUSDT days
   flagged against 2). The numerator `RV/BV - 1` that `realized-library`'s
   adjusted test and `mfe-toolbox` use is a different statistic from the
   paper's `BV/RV - 1`.
3. **Lee and Mykland.** R `highfrequency` with `dontIncludeLast = TRUE` and
   `alpha = 0.05` is an oracle for the per-return statistic and the flags to a
   few ulps; its defaults are not (window, critical value). R `yuima`'s
   `lm.jumptest` computes a shifted-window statistic, `realized-library`'s is
   scale-dependent. For the maximum `T` the paper's formula in numpy is the
   reference, and its size under a constant volatility is the check.
4. Not run: the 2012 noise-robust Lee and Mykland test (`hf.econometrics`), a
   different statistic; `yuima`'s `log` and the `minrv`/`medrv` tests;
   smaller jumps than 6 to 12 standard deviations; `K` in the paper's range
   `sqrt(252 n)` to `252 n` for a single day (a day cannot supply it); several
   days of data per test (the paper's range needs them).

## 6. The online forms and their state (part 3)

Not yet run.

## 7. Hawkes intensity against `tick` (part 4)

Not yet run.

## 8. Proposed tolerances and conclusions (part 4)

Not yet written.
