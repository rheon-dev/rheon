# Bipower variation and the jump tests against third-party oracles (PLAN task 499, request A19)

*Status, 2026-11-22: a record, complete for request A19 (branch `vm/study-499b`).
Part 1: the oracle search (§2), the paper's definitions (§3), bipower variation and
tripower quarticity against every candidate that computes them (§4). Part 2: the
Barndorff-Nielsen and Shephard tests and the Lee and Mykland statistic against
theirs, with and without planted jumps and on the Binance data (§5). Part 3: the
online forms of all four, their state in bytes and their agreement with the batch
oracles (§6). Part 4: hfvol's Hawkes intensity against `tick` and the online
Hawkes form (§7), the proposed tolerances, the state against a per-group cap and
the conclusions (§8). Nothing in the package or the Rust crates was changed. The
online forms are Python prototypes of the arithmetic, not Rust. The runnable study
is `scripts/study_499_jumps.py` (`part1` to `part4`); it is Python, plus R through
rpy2. This is not the realized-kernel study of the same task number on branch
`vm/study-499`; the two share nothing but the task.*

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

*Run: `scripts/study_499_jumps.py part3` (21 s: four series, 23,400 rows at the
longest; the four every table below uses: synthetic with five jumps at
n = 23,400 ("J23400"), synthetic without at n = 2,340 ("N2340"), BTCUSDT 1 minute
of 2024-01-05 ("BTC", 1,439 rows) and the ETCUSDT trades of 2024-01-02 in 10
seconds ("ETC10s", 8,639 rows)). The engines are `JumpEngine` (bipower variation,
tripower and quadpower quarticity, realized variance and the BNS ratio as one row
machine) and `LeeMyklandEngine` in the script; they are Python prototypes of the
arithmetic, so nothing here is said of speed.*

### 6.1 The forms

A row is a clock `c`, a return `r` and a weight `w >= 0`. The rules, which the
tests of §6.5 hold:

* **Out of sample, and `weight` as everywhere.** The outputs of a row are read
  from the state *before* its update; `weight` (the emitted `weight_sum`) is the
  accumulated weight before the row's update and before its own decay, and
  under a window the weight inside the window.
* **A row of weight 0, or a return that is not finite, advances the clock and
  learns nothing.** The sums decay by the clock step and the window edge moves; no
  sum takes a term; the memory of the last returns is not touched, so the row is
  as if absent, a product reaching across it to the last row that was observed.
  It takes no snapshot and enters no ring. A stream that opens with such rows
  leaves the state free of NaN.
* **Sums.** With `a = |r|`, `b = a^(4/3)`, a term taking the weight `w` of the
  *later* row of its product and the decay `lam^(clock age)`:
  `nw = sum w`, `rv = sum w r^2`, `bv = sum w a_j a_{j-1}`,
  `tq = sum w b_j b_{j-1} b_{j-2}`, `qp = sum w a_j a_{j-1} a_{j-2} a_{j-3}`.
  Outputs `BV = (pi/2) bv`, `TQ = nw tq / mu_{4/3}^3`, `QP = nw qp (pi/2)^2` and
  the adjusted ratio `Z = sqrt(nw)(1 - BV/rv) / sqrt(vartheta max(1, TQ/BV^2))` of
  §5.1. The paper's `n` becomes the weight `nw`. With `lam = 1`, no window and
  unit weights they are §3's `BV`, `TQ`, `QP` and §5's `Z` on the rows so far
  (§6.3 checks this against the oracles).
* **A hard window** of `W` clock units, closed on the right (a row exactly `W`
  old has no weight; as in the repo's windows). It holds a term for every row in
  it, the first one's with its predecessor outside the window: **W returns, W
  products**, one product more than the paper's batch formula on `W` returns, which
  has `W - 1`. The batch oracle for the window is the paper's formula on the `W + 1`
  returns that end at the row (§6.4).
* **Lee and Mykland.** The window of `K` rows is the paper's (the `K - 2` products
  before the row, §5.1); the decay form is the exponentially weighted mean
  `sigma^2 = sum lam^age w p / sum lam^age w`; the clock form the mean over the
  products of the rows within `W` clock units. `L` is read before the row's update,
  so the statistic is out of sample by construction. `T = (max |L| - C_n)/S_n` is
  the running maximum over the rows tested so far; `n` is the count of rows
  tested, or a fixed `n_ref` (the tests below use the batch's `n`, which the
  paper's flags need: with the running count the early rows get a smaller `C_n`
  and more flags).

### 6.2 State in bytes

Every number held is 8 bytes (f64 sums, `|r|`s, i64 clock and counts). Measured
on the engines (`state_bytes` counts the numbers held, as the repo's `Footprint`
does), and equal to the formulas over 23,400 rows:

| Form | Bytes, as a function of the window or decay |
|---|---|
| BNS engine, one pass or any decay (5 sums, 3 returns, clock, count) | **89**, whatever `lam`, `n` or the half-life |
| the same with Neumaier-compensated sums | **137** (the compensations, 48 more) |
| BNS engine, hard window, **snapshot ring** (the repo's design: a snapshot of the 5 sums, the clock and the count a row) | **89 + 56 (rows in the window + 1)**: 5,745 at 100 rows, 56,145 at 1,000, 560,145 at 10,000 |
| BNS engine, hard window, **term ring** (each row's clock and 5 terms, compensated running sums) | **137 + 48 (rows in the window)**: 4,937 at 100 rows, 48,137 at 1,000, 480,137 at 10,000 |
| the snapshot ring with a snapshot every `W/20` clock units | at most `W/s + 1 = 21` snapshots: 1,265 bytes whatever `W` |
| Lee-Mykland, decay | **80** |
| Lee-Mykland, window of `K` rows, Neumaier-compensated sum (a ring of `K - 2` products) | **8 K + 48**: 848 at `K` = 100, 2,448 at 300 |
| the same with a plain running sum / two stacks / differences of cumulative sums | `8 K + 40` / `16 K + 24` (1,624 at 100, 4,824 at 300) / `8 K + 48` |
| Lee-Mykland, clock window (ring of clock, product and weight; compensated sums) | **80 + 24 (rows in the window)** |

Each output needs fewer numbers than the engine that serves all of them
(formulas, not measured): bipower variation alone 32 bytes and 16 per row in a
ring; tripower quarticity 48 and 24; the BNS ratio with the tripower 64 and 40, with
the quadpower 72 and 40; Lee-Mykland over `K` rows 48 and 8 per row. A **clock**
window holds as many rows as its clock units contain, which no window length
bounds: on the irregular clock of this study (exponential gaps, mean 1) a window
of 100 units held up to 126 rows and one of 1,000 up to 1,035. So a clock window
needs a cap on its rows (the repo's `window_budget` and `max_rows_between_snapshots`
are that). Over 23,400 rows the rings stayed at exactly the sizes above (checked
at every 1,000th row) and did not grow.

**To raise before it lands** (the rule on state growth): a window of `W` rows
holds 48 to 56 bytes a row per group, 480 to 560 KB a group at `W` = 10,000,
480 to 560 MB at 1,000 groups, against 89 to 137 bytes a group without a window.
A snapshot cadence bounds the snapshot ring at `W/s + 1` snapshots (1.3 KB at
`s = W/20`), at the price of an edge that is `s` wide (a window of 1,000 units
held 1,113 rows when sampled). The Lee-Mykland window is small: 848 bytes at the
paper-like `K` = 100.

### 6.3 One pass and decay, against the exact values and the oracles

References: for the one pass, decay and windows, each term in float64 summed
exactly (`fsum`), with the outputs formed in 40 digits; its own error is the
rounding of the terms, about one ulp. `Z` is a difference, so its gap is in
units of `cond x 2^-52`, `cond = (RV + BV)/|RV - BV|`.

*One pass, 25 prefixes of each of the four series (largest gap, ulps):*

| Sums | BV | TQ | QP | RV | `weight` | `Z` (cond units) |
|---|---|---|---|---|---|---|
| plain running sums (89 bytes) | 40.3 | 132 | 32.2 | 30 | 0 | 29.6 |
| Neumaier-compensated (137 bytes) | 0.7 | 5.1 | 1.4 | 0 | 0 | 1.5 |

The plain sum's error grows with the rows (a random walk of the rounding of each
add: 40 to 130 ulps at 23,400 rows, 5 to 17 at 1,439); the compensated sum stays
within the rounding of the terms.

*Against the oracles of §4 and §5 on the whole series* (the online value is the
paper's, so the gap is the sum's): BV against `hfvol` and R `rBPCov`: 22 ulps
plain, 0 compensated (largest over the four series); TQ against `rTPQuar` with
its `n/(n-2)` divided out: 41 plain, 10 compensated (the oracle's own error is
17 to 24 ulps, §4.2); `Z` (adjusted, tripower) against `yuima` `bns.test` ratio
with `r = rep(4/3, 3)` and `adj = TRUE`, its `sqrt(n+1)` divided out, and against
R `highfrequency` `BNSjumpTest` ratio with `max = TRUE`, its `n/(n-2)` on the
quarticity applied to the online sums: relative 2.6e-13 plain, 6.5e-16 and
8.7e-16 compensated.

*Decay* (half-lives of 50 and 500 per clock unit, on the row clock and on an
irregular clock, weights in [0.5, 1.5], 25 prefixes of J23400 and BTC), largest
gap in ulps: plain BV 23, TQ 31, QP 45, RV 17, `weight` 15, `Z` 16 (cond units);
compensated 20, 27, 30, 19, 14, 7. The compensation does not help the decay,
whose error is the rounding of the decay multiplication at every step (about 15
ulps for a 500-unit half-life); state stays at 89 or 137 bytes.

### 6.4 The hard window

References as §6.3, with the window's terms as defined in §6.1 (so the window
holds `W` products). Largest gap in ulps over J23400 and BTC, windows of 100 and
1,000 units, on the row clock and on the irregular clock, 25 prefixes each:

| Window | no decay: BV | TQ | RV | `Z` (cond) | decay, half-life `W/4`: BV | TQ | RV | `Z` |
|---|---|---|---|---|---|---|---|---|
| snapshot ring, every row (rows / clock) | 2,118 / 1,871 | 14,245 / 23,799 | 2,487 / 2,504 | 2,739 / 2,414 | 18 / 24 | 244 / 103 | 19 / 12 | 11 / 10 |
| snapshot ring, every `W/20` (clock) | 1,678 | 26,696 | 2,377 | 2,114 | 22 | 93 | 12 | 9 |
| term ring, compensated (rows / clock) | 0.8 / 0.8 | 4.8 / 5.2 | 0 / 0 | 0.7 / 0.7 | 13 / 12 | 88 / 161 | 11 / 18 | 6 / 7 |

**The snapshot ring without a decay loses 3 to 4 digits** because the window is
the difference of two cumulative sums that have grown with the stream: the error
is about the rounding times rows so far over window (234 for the 100-row window
on J23400's 23,400 rows; BTC at `W` = 100 gives 152 ulps for BV and 13,000 for
TQ). With a decay the live sums are bounded and the gap falls to a few tens of
ulps (TQ 100 to 250). **The term ring with compensated sums has no such growth**:
0.8 ulps for BV and 5 for TQ in the undecayed window, the rounding of the terms.
With a decay the term ring is bounded by the decay's multiplication and by
subtracting the expired term decayed by its `lam^age`, which does not reproduce
the live product of the per-row factors bit for bit: BV 12 to 13, TQ 88 to 161
ulps.

*Against the batch oracles, a window of `W` rows, no decay* (`hfvol`, R `rBPCov` on
the `W + 1` returns ending at the row; R `rTPQuar` on the `W + 2`, its `N N/(N-2)`
rescaled to `W`; every 997th row; ulps):

| Window | `hfvol` BV | `rBPCov` BV | `rTPQuar` TQ |
|---|---|---|---|
| snapshot ring, J23400, W = 100 / 1,000 | 2,168 / 495 | 2,169 / 495 | 5,479 / 4,207 |
| snapshot ring, BTC, W = 100 / 1,000 | 15 / 0 | 15 / 0 | 635 / 0 |
| term ring, J23400, W = 100 / 1,000 | 1 / 1 | 0 / 0 | 12 / 13 |
| term ring, BTC, W = 100 / 1,000 | 0 / 0 | 0 / 0 | 10 / 0 |

*The edge.* With a snapshot or a term per row, the rows in the window are exactly
the rows whose clock is above `c - W`, on the row clock and the irregular clock
(checked every 331st row, all true). With a snapshot every `W/20`, the window is
the rows after the snapshot at or below the cut, between `W` and `W + W/20`
clock units; the engine reports the cut and the reference uses it.

### 6.5 Zero-weight rows

A stream of the real rows with a third as many zero-weight rows inserted at
random, each carrying garbage in its return (NaN, infinity, +-1e300, 0.37,
-1e-300), against the same real rows without them (N2340: 2,340 real rows, 780
zero-weight; BTC: 1,439 and 479):

| Form | the zero-weight row | result |
|---|---|---|
| `lam = 1`, no window | at its neighbour's clock | outputs **bit for bit** those of the stream without (both series); 5.1 and 6.5 ulps to the exact |
| decay, `lam = 0.5^(1/100)` | half a clock unit later | within 14 (N2340) and 44 (BTC) ulps of the stream without, whose one decay of a unit and a half became two (`lam^0.5`, `lam^1`); 22 and 51 ulps to the exact |
| window of 200 units, snapshot ring | half a unit later | **bit for bit**; 16 and 844 ulps to the exact (the ring's difference, §6.4) |
| window of 200 units, term ring | half a unit later | **bit for bit**; 3.0 and 3.4 ulps to the exact |

The state is finite at every row in all of them, and a stream whose first 5 rows
have weight 0 leaves a finite state and finite outputs (no division is made
before a positive weight).

### 6.6 Lee and Mykland online

The window of `K` rows with four running sums of the products (24 runs: J23400 at
`K` = 100 and 300, N2340 and BTC at 30 and 100, `n_ref` the batch's `n`):

| Running sum | `L` against the paper's batch `L`, largest, ulps | `T` against the paper's | flags | bytes at `K` = 100 / 300 |
|---|---|---|---|---|
| Neumaier-compensated ring | **3** | 0 | identical to the paper's in all | 848 / 2,448 |
| two stacks | 7 | 4e-14 | identical | 1,624 / 4,824 |
| plain running sum (add the new, subtract the old) | 62 | 1.6e-13 | identical | 840 / 2,440 |
| differences of cumulative sums (the snapshot form) | 2,312 | 1.3e-12 | identical | 848 / 2,448 |

(The paper's own `L` in float64 agrees with R `highfrequency`'s to 4 ulps, §5.4, so
the online `L` with the compensated or two-stack sum is at the oracle's floor.)
When an outlier leaves the window, the relative error of its sum (`K` = 100, an
outlier of `m` standard deviations at row 500, read from row 700 on):

| `m` | Neumaier | two stacks | plain running sum | cumulative differences |
|---|---|---|---|---|
| 1 | 0 | 5e-16 | 6e-15 | 2e-14 |
| 1e3 | 0 | 5e-16 | 5e-14 | 1e-13 |
| 1e6 | 0 | 5e-16 | 4e-10 | 1e-10 |
| 1e9 | 0 | 5e-16 | 1.5e-7 | 1.9e-7 |

The plain running sum is fine for returns without absurd outliers and wrong after
a price glitch of 1e6 standard deviations (4e-10 relative, kept after the glitch
has left the window); the compensated ring is exact in all four and half the bytes of
the two stacks. The decay form agrees with the exact weighted mean to 3.0 eps
(J23400) and 2.2 (BTC); the clock form (term ring) to 0.8 and 0.3.

### 6.7 State is bounded, and a save at a chunk edge changes no bit

Over the 23,400 rows of J23400 the snapshot ring held 101 and 1,001 snapshots at
`W` = 100 and 1,000, the term ring 100 and 1,000 rows, the bytes at every 1,000th
row between the empty state's (145 / 185) and the formula's (5,745 / 56,145;
4,937 / 48,137) and never above. Each of eight forms (the BNS engine one pass,
decay, snapshot window, term window, decay and window; Lee and Mykland over `K`
rows, clock window, decay) was saved, loaded and continued at a third and two
thirds of 2,340 rows: the outputs of every row are bit for bit those of the
uninterrupted run (a Python object round trip: it shows the state is complete, not
that a Rust layout is stable).

### 6.8 What part 3 settles

1. **The forms exist, in the state sizes of §6.2, and agree with the batch oracles
   to the rounding of the sums:** BV and TQ to 0 to 10 ulps and `Z` to 1e-15
   relative with compensated sums on a whole series (22, 41 ulps and 2.6e-13 with
   plain ones at 23,400 rows); the Lee and Mykland `L` to 3 ulps and its flags
   identical to the paper's.
2. **Choices that matter, which part 4's tolerances depend on:** compensated sums
   cost 48 bytes and take the one-pass error from a random walk in the rows to the
   rounding of the terms; for a hard window, the repo's snapshot ring loses 3 to 4
   digits without a decay (an error growing with rows over window) where a term ring
   with compensated sums costs 8 bytes a row less and keeps 1 to 5 ulps; the Lee
   and Mykland window's best sum is the compensated ring (8 K + 48 bytes, 3 ulps).
   A window in clock time needs a cap on its rows.
3. **Conventions to fix before any Rust:** the windowed bipower variation has `W`
   products over `W` returns, not the paper's `W - 1`; the term of a product takes
   the weight of its later row; a zero-weight row advances the clock and is
   otherwise absent.
4. **Not done:** speed and memory traffic (the prototypes are Python); a bucketed
   ring (a window held as `W/s` buckets of compensated sums, which would bound the
   bytes and not lose the digits; not built, so not measured); non-unit weights
   inside a hard window, `min_weight`, many groups; Lee and Mykland's `T` as the
   maximum over a moving set of rows (a monotone deque, bounded by the window; not
   built); `weight` and `min_weight` across a bank (rule 8 is implemented, not
   tested against the other models).

## 7. Hawkes intensity: hfvol against tick (part 4)

*Run: `scripts/study_499_jumps.py part4` (16 s; the run twice gave identical
output). PLAN task 498 (Hawkes and other point processes) has no model yet; this
is the check of the two Python oracles it names, before a model: hfvol 0.1.0
(MIT) and tick 0.8.0.2 (BSD-3-Clause).*

### 7.1 The definitions and the conventions

Univariate Hawkes process, exponential kernel (Hawkes 1971; Ozaki 1979):
`lambda(t) = mu + alpha sum_{t_j < t} exp(-beta (t - t_j))`, the left limit,
strictly earlier events; `log L = sum_i log lambda(t_i) - mu T - (alpha/beta)
sum_i (1 - exp(-beta (T - t_i)))`. hfvol's `HawkesParams(mu, alpha, beta)` has
`alpha` the jump in intensity per event; tick's kernel is `a beta exp(-beta t)`,
`a` the adjacency (the branching ratio `alpha/beta`): **`alpha = a beta`**. Other
conventions found, each of which a test must state:

* tick's `score` with `gofit="likelihood"` is **`(log L + T)/n`**, not `log L`:
  `log L = n score - T` (checked at five parameter sets, §7.2).
* tick's `estimated_intensity` at an event time is the **left limit**; hfvol's
  `hawkes_intensity_path` defaults to the **right limit** (`include_current=True`:
  an event at `t` counts), which differs at every event by `alpha`; the left
  limit, `include_current=False`, is the predictable intensity and the one to use as
  a feature. tick's *simulator's* `tracked_intensity` is the right limit (it
  agrees with hfvol's `include_current=True` to 4e-15 and differs from the
  left by 0.6 = `alpha` at event times).
* **Tied time stamps**: hfvol lets only strictly earlier events excite. tick's
  `estimated_intensity` at tied events is sequential in array order: the k-th
  event of a tie group sees the `k - 1` earlier ones, so it is the left limit plus
  `(k - 1) alpha` (on a stream with 257 tied events, the largest relative
  difference is 1.76; the first event of each group agrees). tick's log-likelihood
  agrees with hfvol's on the same stream (2.7e-12 absolute) so it is strict.
* tick's learners need `numpydoc` and `six`, which its wheel does not declare
  (without `numpydoc` every learner fails with "has no settable attribute
  'events'": tick builds its attribute table from docstrings); `fit(events, start)`
  has no `end_time` (the end is the last event; a float passed as the second
  argument is the *start point*, which is how a first attempt failed with "the
  sum of the influence on someone cannot be negative"); the likelihood fit works
  with `solver="bfgs"` and `penalty="l2"` at a huge `C` (a negligible penalty) and
  raises that error with `agd` and `gd` and with `penalty="none"` in this build.
  The decay is fixed (given), not estimated.

### 7.2 Agreement

Three streams of 2,000 time units: one simulated by hfvol (`mu, alpha, beta =
0.5, 0.6, 1`; 2,523 events), one by tick's `SimuHawkes` (2,624), and the first
rounded to 0.1 (2,523 events, 257 tied). Parameters for the intensity are tick's
own likelihood fit of the stream.

*The intensity* at tick's 42,500 times (a 0.05 grid and the event times), tick's
`estimated_intensity` against hfvol's left limit: maximum absolute 6.2e-15,
relative **1.7e-15** (hfvol-simulated stream) and 4.9e-15 and 1.8e-15 (tick's
stream). Against the definition in 40 digits at 60 sampled points: hfvol
**1.7 ulps**, tick **8.2 ulps** (5.9 and 1.5 on the second stream). The right limit
differs at the event times, as §7.1 says (by `alpha` at all 2,522 event times).

*The log-likelihood* at five parameter sets (including `alpha = 0`, and `beta`
other than 1), against the definition in 40 digits:

| | hfvol | tick (mapped, `n score - T`) |
|---|---|---|
| largest absolute gap to the definition, `|log L|` of 1,500 to 2,200 | 4.1e-13 | 5.6e-12 |
| tick against hfvol | 5.5e-12 absolute, **3e-15** relative to `|log L|` | |

(The same on the tied stream: 5.5e-12.) Both implement the strict-tie definition.

*The fit.* tick's likelihood fit (`bfgs`) and the maximum of hfvol's
log-likelihood found by Nelder-Mead at the same `beta = 1`: the same maximum to
**5.6e-9** in `mu` and **2.2e-9** in `alpha` (3.7e-9 and 2.2e-9 on the second
stream), which is the optimisers' tolerance; with `beta` free, hfvol's
`fit_hawkes_mle` and tick's likelihood profiled over `beta` give `beta` = 0.91089966 and 0.91089965
(0.98756930 and 0.98756918 on the second stream) with log-likelihoods equal to
7e-12 and 5e-12. tick's default least-squares fit is a different estimator:
`mu, alpha` = 0.4845, 0.6167 against 0.4906, 0.6118 for the likelihood on the first stream.
hfvol's own code is correct against the definition (above), including its tie rule.

### 7.3 The online form

`HawkesEngine`: three numbers of state (`E`, the excitation of the strictly earlier
events; `m`, the weight of the events at the last time; the last time), 24 bytes,
whatever the length of the stream and the half-life `ln 2/beta`. The intensity
before each event is `mu + alpha E` with `E <- exp(-beta dt)(E + m)` when the time
moves and `m` kept while it does not (events at one time do not excite each
other: hfvol's rule); an event of weight 0 advances the clock and adds nothing to
`m`. On the hfvol stream: **3 ulps** to the definition in float64, **2 ulps** to
hfvol's left limit (bit for bit at 1,958 of 2,523 events), 1.4 ulps to the exact
values at 60 events; on the tied stream 2 ulps to hfvol (strict ties, 257 tied
events). 800 zero-weight events inserted at random times leave the intensity at
every real event within **4 ulps** of the stream without them and the state finite. A
hard window on a kernel that already decays is not needed; a truncated kernel
would hold one time stamp (8 bytes) per event in the window, not built.

## 8. Proposed tolerances and conclusions (part 4)

### 8.1 Tolerances

Each is what a rheon test could assert, with the measurement it comes from. Ulps
are of the quantity, for deterministic arithmetic, with the oracle's own error
included; a statistic that is a difference carries its conditioning
(`cond = (RV + BV)/|RV - BV|`, at most 3.4e3 here) in units of `cond x 2^-52`.

| Quantity | Against | Proposed tolerance | Measured (largest) |
|---|---|---|---|
| bipower variation, batch | `hfvol`, R `rBPCov`, `hffe`, `highfrequency`, `hf.econometrics` (no factor) | 8 ulps | 5 (all pairs); the paper's formula in numpy 4.4 |
| bipower variation with `n/(n-1)` (`realized-library` default, `hf.econometrics adjust`) | the same, times `n/(n-1)` | 8 ulps after the factor | 3.8 to 4.9 |
| tripower quarticity, batch | `highfrequency` `rTPQuar` (its `n/(n-2)` divided out), `realized-library` `multipower_variation(m=3, r=4)` | 32 ulps | 17.7 to 23.7; the paper's in numpy 14.7 |
| `mfe-toolbox` tripower quarticity | none: wrong by `n - 2` | not an oracle | - |
| BNS `Z` (G, H, J), batch | `highfrequency` (with `n/(n-2)` on TQ), `yuima` (`sqrt(n+1)`), with the conventions applied | relative 1e-12, and 64 in units of `cond x 2^-52` | 9.4e-14 relative; 50 (cond units) |
| BNS `Z`, the paper's against `realized-library` ratio, `mfe-toolbox` | none | not an oracle (a factor of order `n`; the `n - 2`; the numerator `RV/BV - 1`) | - |
| Lee and Mykland per-return `L` | `highfrequency` with `dontIncludeLast = TRUE`, `alpha = 0.05` | 8 ulps; the flags identical | 4 (all 28 cases); flags identical in all |
| Lee and Mykland `T` | the paper's formula | 1e-11 absolute; same `n` in `C_n`, `S_n` | 1.3e-12 |
| `yuima` `lm.jumptest`, `realized-library` `lm_test` | none | not oracles (a shifted window; scale-dependent) | - |
| online one pass, compensated sums (137 bytes): BV, TQ, QP, RV | the exact paper value | 4, 16, 8, 2 ulps | 0.7, 5.1, 1.4, 0 |
| online one pass, plain sums (89 bytes) | the same | `0.5 sqrt(n)` ulps for BV, QP, RV and `1.5 sqrt(n)` for TQ | 40, 30, 32 and 132 at n = 23,400 (0.26 sqrt(n), 0.9 sqrt(n)) |
| online `Z`, one pass | `yuima`, `highfrequency` with conventions | relative 1e-12 plain, 1e-14 compensated; `cond` units 64 plain, 8 compensated | 2.6e-13 plain, 8.7e-16 compensated; 29.6 and 1.5 |
| online decay (any half-life) | the exact weighted sums | 48 ulps BV, 64 TQ and QP, 32 RV and `weight` | 23, 31, 45, 17, 15 |
| online hard window, term ring (compensated) | the paper's formula on the `W + 1` returns (`hfvol`, `rBPCov`); `rTPQuar` rescaled | 4 ulps BV, 32 TQ, 4 RV | 1, 13, 0 |
| the same with a decay | the exact weighted window | 32 ulps BV, 256 TQ, 32 RV | 13, 161, 18 |
| online hard window, snapshot ring, no decay | the same | **not acceptable**: error is `eps x rows so far / W` times a constant (2,100 ulps BV, 14,000 TQ at 23,400 rows over 100) | - |
| online Lee-Mykland `L`, window of `K` rows | the paper's `L` | 8 ulps (compensated ring), 8 (two stacks); 128 (plain running sum) | 3, 7, 62 |
| online Lee-Mykland, decay and clock windows | the exact weighted mean | 8 eps | 3.0, 0.8 |
| zero-weight rows, `lam = 1` or any window | the same stream without them | **bit for bit** | bit for bit in all |
| zero-weight rows, decay | the same stream without them | 64 ulps (the decay rounds once per row) | 44 |
| Hawkes intensity, hfvol left limit | the definition (40 digits) | 4 ulps | 1.7 |
| Hawkes intensity, tick `estimated_intensity` | hfvol left limit | relative 1e-14; ties excluded (they differ by convention) | 1.8e-15 |
| Hawkes log-likelihood, hfvol, tick (mapped `n score - T`) | the definition | 1e-14 of `|log L|` | 3e-15 |
| Hawkes fit, `beta` fixed | tick `bfgs` likelihood fit | 1e-7 in each parameter (the optimisers') | 5.6e-9 |
| Hawkes online intensity | the definition; hfvol | 4 ulps; 4 ulps | 3; 2 |

A recursive estimator of the parameters (PLAN 498) is compared with the batch
likelihood by a statistical tolerance, not a floating one; not run.

### 8.2 The state of each form against a per-group cap

Cap 256 KB (262,144 bytes) a group, a warning at 128 KB (131,072), a refusal at
the cap, no cap on a bank's total. The rows a window can hold:

| Form | Bytes | Rows at the warning | Rows at the cap |
|---|---|---|---|
| BNS engine (BV, TQ, QP, RV, ratio), one pass or decay | 89 (137 compensated) | no limit | no limit |
| BNS engine, hard window, snapshot ring | `145 + 56 W` | 2,337 | **4,678** |
| BNS engine, hard window, term ring | `137 + 48 W` | 2,727 | **5,458** |
| BNS ratio with the tripower, minimal term ring | `64 + 40 W` | 3,275 | **6,552** |
| bipower variation alone, minimal term ring | `32 + 16 W` | 8,190 | 16,382 |
| Lee-Mykland, `K` rows, compensated ring | `8 K + 48` | 16,378 | 32,762 |
| Lee-Mykland, clock window, term ring | `80 + 24 W` | 5,458 | 10,919 |
| Lee-Mykland, decay | 80 | no limit | no limit |
| Hawkes, exponential kernel | 24 | no limit | no limit |

**A hard window on the BNS engine beyond about 4,500 rows (4,678 with the
repo's snapshot ring, 5,458 with a term ring) exceeds the per-group cap, and the
warning is reached at about 2,300 to 2,700 rows. That needs the user's decision:**
refuse such a window; or take a snapshot cadence (`W/20`: 21 snapshots, 1.3 KB,
but the snapshot ring loses the digits, §6.4) or a bucketed ring of compensated
sums (not built; bounds the bytes and keeps the digits); or hold a smaller
number of outputs per group. A clock-time window is bounded by the rows its
window can hold, so it needs a row cap in any case. The Lee-Mykland window of
the paper's range (`K` up to a few hundred) and every decay form are far under the
warning.

### 8.3 Conclusions

1. **Bipower variation and tripower quarticity** have six independent oracles
   that agree to 5 ulps (BV) and to the conventions' factors (`n/(n-1)`, `n/(n-2)`)
   and 24 ulps (TQ); `mfe-toolbox`'s TQ is wrong by `n - 2` (§4).
2. **BNS test.** R `highfrequency` and R `yuima` are oracles to 1e-13 relative once
   their conventions are applied (`n/(n-2)` on the quarticity; `sqrt(n+1)`); the
   paper's statistics have no small-sample factor. `realized-library`'s ratio test
   and `mfe-toolbox`'s test are not oracles; the latter over-rejects (8.5% and
   8.0% against 5.6% and 5.1% at 5%) and flags 7 of 10 BTCUSDT days where the
   paper's flags 2 (§5).
3. **Lee and Mykland.** R `highfrequency` with `dontIncludeLast = TRUE` and
   `alpha = 0.05` is an oracle for the per-return statistic and the flags to 4 ulps
   (its defaults include the current return and use a critical value that is for a
   95% test); `yuima` and `realized-library` are not (§5).
4. **Online forms exist** for all four and agree with the batch oracles to the
   rounding of the sums (§6): compensated sums cost 48 bytes; a hard window is
   best held as a term ring (48 bytes a row, 1 to 13 ulps) rather than the
   snapshot ring (56 bytes a row, up to 14,000 ulps without a decay); the
   windowed bipower variation has `W` products over `W` returns; a zero-weight row
   advances the clock and is otherwise absent, bit for bit under `lam = 1` and
   under a window.
5. **Hawkes.** hfvol and tick agree to 2e-15 relative on the intensity and 3e-15
   on the log-likelihood once the conventions of §7.1 are stated (the left limit,
   `alpha = a beta`, `(log L + T)/n`, the tie rule); both are second oracles for a
   rheon `hawkes`, and hfvol is correct against the definition. The online
   intensity is 24 bytes and within 3 ulps.
6. **For the user:** the state of a hard window beyond about 4,500 rows (§8.2).

### 8.4 Not verified

Speed and memory traffic of any online form (Python prototypes only); a Rust
layout and its state fixtures; a bucketed ring (a window held as buckets of
compensated sums); non-unit weights inside a hard window; `min_weight` and
`weight` across a bank; Lee and Mykland's `T` as the maximum over a moving set
of rows; jumps smaller than 6 to 12 local standard deviations (power);
`K` in the paper's recommended range, which a single day cannot supply; the BNS
tests on several days of Binance data (one day each, ten for BTCUSDT); the
published JFEC article of BNS (2006) (the working paper of 2004 was read);
oracles reachable only through MATLAB (the MFE Toolbox); tick's least-squares
learner beyond one stream; tick's multivariate Hawkes and non-exponential kernels;
a recursive parameter estimator for Hawkes against the batch likelihood; any
part of the search that PyPI's missing search endpoint hid.
