<!-- kind: record -->
# Study 511b (C11): the rest of tasks 511-525 (517 to 525), as numpy against their corrected oracles

Python only, no Rust: each feature's online form in numpy, beside the recursion the existing
model keeps (read from `crates/rheon-core` on `origin/main`), against the oracle review C1
confirmed or corrected (`docs/records/ORACLE-VERIFY-511-525.md` on `vm/research-511`, names as
C1 spells them). Method and format as `docs/records/STUDY-511-FEATURES-1.md` (C10, branch
`vm/study-511`). Nothing here is collected by pytest and nothing is added to `pyproject.toml`.

## Files and how to run them

- `scripts/study_511b.py`: prints every table below (`--only 517 518 ...`, `--seeds N`).
- `studies/study_511b/`: `common.py` (seeded streams, decay convention, the `ew_cov` mean-form
  accumulator `EwMoments`, two-pass oracle), one module per task, `t517.py` ... `t525.py`; each
  module's docstring holds the update equations.

Environment: a standalone venv (`/home/user/venv-c11`), not `uv sync`: Python 3.12, numpy 2.5.5,
scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, MAPIE 1.6.0, polars 2.1.0, hmmlearn 0.3.3,
filterpy 2.0.0, dieboldmariano 1.1.0 on Linux x86-64 (torch and `bayesian-changepoint` for 523 are
listed in that section). Decay convention, as the models': a gap `dt` decays everything by
`lam = 0.5 ** (dt / half_life)`, then the row enters at weight `w`; a zero-weight row advances the
clock and learns nothing (rule 9), first row included. Where a feature takes a decay, it is checked
with weights in [0.5, 2], interleaved zero-weight rows and a zero-weight first row.

PLAN task 537 is not in `docs/PLAN.md` on `origin/main` (the plan ends at 534); the growth rule used
is CLAUDE.md's "State stays bounded, of reasonable size, and the fit stays one pass", and the
bank-level total state cap the coordinator relayed (a per-group cap of 256 KB by default; a settable
bank total, warning at half and refusing at the cap, infinity allowed; each feature's bytes per
group feed that total, so each section states its contribution).

SELECTION_PLACEHOLDER

## 517. `ew_cov` outputs: the uncentered covariance and a Gaussian log-likelihood

Beside `ew_cov`'s mean-form recursion (`W' = lam W + w`, `a = lam W / W'`, `b = w / W'`, `d = x - m`,
`m' = m + b d`, `C' = a C + a b d d'`, `Q' = lam^2 Q + w^2`, ridge scale `s' = a s`), both outputs
are reads of what the accumulator keeps:

- **Uncentered second moment** `R = C + m m'` (divisor `W`, the weighted mean of `x x'`; the
  accumulator already offers it as `EwCov::raw`). sklearn's `assume_centered=True` is `X'X / n`;
  `rcov(kind="plain")` is `sum x x' = n R` at `lam = 1`, unit weights.
- **Gaussian log-likelihood** `ll = -0.5 (k ln 2 pi + ln|S| + q)`, `S = c (C + s prior I)`,
  `q = d' S^-1 d` with `d = x - m`, read before the row folds in (rule 2): `q` is the square of
  `mahal`, `c` the unbiased factor `W^2 / (W^2 - Q)` (1 where `bias`), exactly the matrix `mahal`
  inverts. Needs `precision_prior`, as `mahal` does.

Against the corrected oracles (largest gap over 20 seeds):

| case | gap |
|---|---|
| `R` vs `EmpiricalCovariance(assume_centered=True)`, `lam = 1`, k = 2 / 5 / 20 | 2.1e-15 / 2.4e-15 / 2.6e-15 (relative to the largest entry); `n R` vs `X'X`: same |
| `R` vs two-pass `sum omega x x' / sum omega`, half-life 20, weights, 10% zero-weight rows, zero first row, offset 1e3, k = 2 / 5 / 20 | 1.2e-15 / 8e-16 / 1.1e-15 |
| `R` over a window of 60 by retraction vs sklearn on the window, k = 2 / 5 / 20 | 2.1e-14 each |
| `ll` vs `scipy.stats.multivariate_normal.logpdf(x, m, S, allow_singular=True)`, `bias` on/off, prior 0 / 1e-3, `lam = 1` and half-life 30 | at most 1.3e-12 (relative, floor 1) |
| `q` vs `-2 ll_ref - k ln 2 pi - ln|S|` (the square of `mahal`) | at most 4.2e-12 |
| `ll` vs sklearn `EmpiricalCovariance.score` on the one row (C1's second check; fitted on the prefix, `lam = 1`) | 2.4e-13 |

The 1e-12 is the oracle's: scipy's `allow_singular=True` takes the log-density through an
eigendecomposition and a pseudo-inverse, rheon's solve is Cholesky; they differ by the condition
number times the unit roundoff on the generic random covariance used.

**RiskMetrics' EWMA** (`S' = lam S + (1 - lam) x x'`) is the same recursion with the row's weight
fixed at `1 - lam`; `R` gives the row `1/W`, with `W -> 1/(1-lam)`. They agree only after the
transient `lam^t`: at `lam = .94`, seeded with `x_1 x_1'`, the relative gap is 1.5, 0.38, 0.038,
2.0e-3, 3.5e-6, 2e-15 at t = 5, 20, 50, 100, 200, 1000. Seeded with the weight `W_0 = 1/(1-lam)` on a zero
matrix the two are equal to 1.7e-16 from row 1. So `R` **is** RiskMetrics' estimate after a warm-up and
is the unbiased-from-row-1 form before it; the comparison is exact only with that seeding.
`rcov(kind="plain")` is compared as `n R` (above).

Decay and zero weight: `R` and `ll` read the same `C`, `m`, `W`, `Q`, so they inherit the accumulator's
rule 9. A zero-weight row only ages `W` and `Q`; against the same stream with the zero-weight rows
removed and their gaps folded into the next kept row, `R` differs by 7 ulps (the rounding of
`lam^a lam^b` against `lam^(a+b)`); a zero-weight first row leaves `W = 0`, every moment finite, and the
log-likelihood is withheld (no fit) until `min_weight`.

**State added: 0 bytes** for `R` and `ll` (both read the accumulator). Emitted slots, not state:
`R` is `k (k+1)/2` values a row if asked for, `ll` is 1. Nothing grows with the stream. Per group it adds
0 to the bank's total cap.

**Changes a default or an existing output: no.** `Cov` (centered) and `Mahal` stay as they are; the
two are new `stats` entries. One thing to name rather than change: `Mahal` is a distance, the square root
of `q`; the log-likelihood's quadratic form is its square.

## 518. Window operators

Polars' own `rolling_min_by`, `rolling_max_by`, `rolling_quantile_by`, `rolling_rank_by` and
`ewm_mean` are the oracle (all four `*_by` are marked UNSTABLE in Polars' docs, so an upgrade can
move the oracle without a deprecation: C1). The windows are hard windows on the stamps: with
`closed="right"` the window is `(t - window, t]`, `"left"` `[t - window, t)`, `"both"` `[t - window, t]`, `"none"`
`(t - window, t)`. A zero-weight row advances the clock (it evicts rows) and enters no window: a null in
the oracle. Weights and decay do not apply to a hard window (each row counts once); a weighted quantile
would be a different quantity with no oracle, and was not studied.

Online forms: `min`/`max` a monotone deque of `(stamp, value)` (amortised O(1)); `quantile` and `rank` a
ring of `(stamp, value)` plus a sorted list of the window's values (`O(log R + R)` insert and delete in
this numpy form; a Rust form would use a tree). The quantile's read, matched to Polars on `m` values
in the window, `h = (m - 1) q`: `nearest` the value at `floor(h + 1/2)` (halves go up), `higher` `ceil h`,
`lower` `floor h`, `midpoint` the mean of the two, `linear` `v[lo] + (h - lo)(v[hi] - v[lo])`,
`equiprobable` `v[max(ceil(m q) - 1, 0)]`. The rank, `lo` and `hi` the bisect positions of the row's own
value in the sorted window: `average` `(lo + 1 + hi)/2`, `min` `lo + 1`, `max` `hi`, `dense` the number of
distinct values at or below it. `method="random"` is not reproducible (Polars seeds it per operation) and
was not compared.

| operator, against Polars (integer stamps with gaps 1-3, window 25, 10% zero-weight rows as nulls, zero first row, ties on a 0.1 grid; 20 seeds for min/max, 5 for the rest) | largest gap | null-pattern mismatches |
|---|---|---|
| `rolling_min_by` / `rolling_max_by`, all four `closed`, `min_samples` 1 and 3 | 0 | 0 |
| `rolling_quantile_by`, six interpolations, q in {.1, .5, .9}, closed right / both / left | 0 | 0 |
| `rolling_rank_by`, `average`, `min`, `max`, `dense`, closed right / both | 0 | 0 |

**Count-weighted `ewm_mean`.** With `lam = 0.5 ** (1/half_life)` per row (a decay in rows, not clock units)
the mean form `W' = lam W + 1`, `m' = m + (1/W')(x - m)` **is** Polars' `ewm_mean(adjust=True)` (the weights
`(1-alpha)^i`); a zero-weight row is a null with `ignore_nulls=False` (a time step that learns nothing). Gap
0 (bit-equal) at half-lives 3 and 40 rows. `adjust=False` is `y' = (lam^g y + alpha x)/(lam^g + alpha)`, `g` the
rows since the last value (`g = 1` is Polars' `(1-alpha) y + alpha x`), `alpha = 1 - lam`: gap 5e-16 and 1.9e-15
relative. The first value seeds the mean; both forms are null until the first non-zero-weight row. So Polars'
`adjust=True` is an option on rheon's mean (decay by row count where the model's own decays by clock), the
existing clock decay unchanged; on a unit-gap clock the two are the same numbers.

**State added** (rows held `R` inside the window; a clock window holds as many rows as the arrival rate puts
in it, so the form needs a `max_rows` that bounds it): `min`/`max` `16 R` bytes at most (the deque; random data
holds about `ln R`, monotone data holds all `R`: 13 against 200 measured at a 200-row window); `quantile` and
`rank` `24 R` (ring `16 R` and the sorted values `8 R`); count-weighted `ewm_mean` `8 (1 + k)` per column set
(`adjust=False`: 16 bytes a column: `y` and `g`). Per group: `R = 100 / 1000 / 10000`: 1.6 / 16 / 160 KB for `min`
and `max`, 2.4 / 24 / 240 KB for `quantile` and `rank`. The 256 KB per-group cap bounds `R` at 16384
(`min`/`max`) and 10922 (`quantile`/`rank`); a window that fills past `max_rows` must refuse or say it is no
longer exact, and the bank total counts `16 R` or `24 R` per group.

**Changes a default: no.** Four new operators and one new decay option; the clock-decayed means are untouched.

## 519. `kalman`: `coef_prior`, the innovation log-likelihood and the Mahalanobis distance

Beside `kalman`'s recursion, in units of the noise (a random-walk-beta filter; per target the mean `b`,
the covariance `P`, and the clock `D` since `P` last took an observation). A row of clock step `d`, weight `w`,
features `z` (intercept first) does `D += d` (every row), and on a row that observes (`w > 0`)
`P += q D^2 I`, `D = 0`, `s = z'Pz + 1/w`, `nu = y - z'b`, gain `Pz/s`, `b += (Pz/s) nu`, `P -= (Pz/s)(Pz)'`; the
first observation is the exact diffuse step on the intercept (`b_0 += nu`, `P_00 = s`, `P_0i = -(Pz)_i`, the slopes' block
unmoved). New, read on the out-of-sample innovation before the update:

- **`coef_prior`**: the initial `b` (as `ewridge` and `rls` have it; for them it is also the ridge's centre).
- **Innovation log-likelihood** `ll = -0.5 (ln 2 pi + ln(R s) + nu^2/(R s))`, `R` the model's noise estimate.
- **Mahalanobis distance** `|nu| / sqrt(R s)` (the square root: C1's correction, filterpy's `mahalanobis` is `sqrt(y' S^-1 y)`;
  rheon's `ew_cov` `mahal` is a distance too, so the two agree in kind; a squared form would be the `ll`'s quadratic).
  It is not the existing `zscore`: that divides by the residual spread `sqrt(R)`, this by the innovation spread
  `sqrt(R s)`, which carries the coefficients' own uncertainty `z'Pz`.

Against filterpy `KalmanFilter` (`F = I`, `predict(Q = R q D^2 I)`, `update(z, R = R/w, H = z')`; `x`, `log_likelihood`,
`mahalanobis`; `P` in absolute units `R P`), 10-20 seeds, k = 2 and 8, 200 rows:

| case | coefficients | `pred` | `log_likelihood` | `mahalanobis` |
|---|---|---|---|---|
| no intercept, `coef_prior` on all, `q = 0`, unit weights | 8e-14 abs | 3.6e-13 abs | 2.7e-13 rel | 4.2e-13 rel |
| `q = 1e-4`, weights in [.5, 2] | 3.0e-14 | 9.1e-14 | 8.9e-14 | 1.2e-13 |
| `q = 1e-4`, weights, 10% zero-weight rows, zero first row | 2.6e-14 | 8.1e-14 | 6.9e-14 | 9.0e-14 |

**Where the oracle is not exact: the diffuse intercept.** filterpy has no diffuse start; its stand-in is a large prior
variance `kappa` on the intercept, which converges to the exact step as `1/kappa`: coefficient gap 3.2e-7, 3.2e-9, 3.2e-11, 3.2e-13
at `kappa` = 1e4 / 1e6 / 1e8 / 1e10; the log-likelihood's relative gap 2.3e-3, 2.3e-5, 2.3e-7, 2.3e-9 (the mahalanobis 3.5e-3 to 3.5e-9).
The comparison is therefore exact only for the no-intercept case (the table above) and a limit for the diffuse case. On the diffuse
row itself the innovation variance is infinite in the limit: the log-likelihood and distance are **withheld on a target's first
observed row** (the likelihood is the proper, non-diffuse part, as in Durbin and Koopman) and from the next row on they are the above.

**`coef_prior` and the intercept.** Because the first observation takes the whole innovation into `b_0`, the intercept's prior
mean is discarded: setting it to 123 against 0 moves the final coefficients by 1.1e-16. `coef_prior` is the slopes' prior, and for
a diffuse intercept the entry for the intercept is moot: say so in the docstring, or refuse a non-zero one (a decision for the
Rust side; it is not a default change, since `coef_prior` is new). With `q = 0` the filter is ridge toward the prior,
`argmin sum w (y - z'b)^2 + (1/p0) |b_slopes - prior|^2` with the intercept unpenalized, to 2e-14, 8e-14, 8e-13 at `p0` = 0.5, 4, 100 (a
second, closed-form oracle).

Decay and zero weight: the clock `D` carries the decay (process noise `q D^2`); a zero-weight row only adds to `D` and has no
innovation (its `s` is infinite), so the log-likelihood and distance are **null on a zero-weight row** (a decision to confirm: scoring a row at the weight
it would have had is the other reading, and it would make the output depend on a weight the row does not carry). Against the same stream with the zero-weight rows' gaps folded into the next
row (10% zero-weight rows, a zero-weight first row): `b` and `P` bit-equal (gap 0), since `D` is additive.

**State added: 0 bytes.** `nu`, `s` and the noise estimate are computed by the step. `coef_prior` is configuration, `8 k_total`
bytes per target if the state file carries it (k = 20: 168). Nothing grows with the stream; per group it adds 0 to the total cap.

**Changes a default: no.** `coef_prior` defaults to none (zeros, as today); the two outputs are new.

## 520. Specification and influence tests

Beside `specification.rs`: a test of what a fit is missing is an auxiliary regression read from EW centred moments of the out-of-sample
residual `e` and regressors `phi`: `LM = n R^2` at Kish's `n = (sum w)^2 / sum w^2`, `R^2 = c' S_xx^-1 c / S_yy` from the centred `C` of the vector
`(e^2, phi)`; a scored row (`w > 0`) enters once, a zero-weight row ages. Oracles are statsmodels 0.15.0 and MAPIE 1.6.0.

- **ARCH LM** (`het_arch`): `phi` = the last `L` scored squared residuals (a ring of `8 L` bytes); a row enters once the ring is full. C1: `nlags` must be
  passed (the default is `min(10, nobs // 5)`) and the result is a plain 4-tuple.
- **White** (`het_white(resid, exog, interaction_terms)`, `exog` with a constant): `phi = (x_i, x_i x_j, i <= j)`, `q = k + k(k+1)/2` regressors (the constant's product with
  `x_i` is `x_i`). **A correction to C1's note**: with `interaction_terms=False` statsmodels regresses `e^2` on the constant and the squares `x_i^2` **alone** (`exog = x**2`), so `q = k`, not `2k`.
- **Spiegelhalter's z** (MAPIE `spiegelhalter_statistic`): `z = sum w (y-p)(1-2p) / sqrt(sum w^2 (1-2p)^2 p(1-p))`: three EW sums.
  The squared weights in the denominator are a choice: at unit weights it is MAPIE's formula; under decay and weights it keeps `Var z = 1` (below).
- **DFBETAS and Cook's distance** beside `influence`: the row's leverage `h = w x' G^-1 x` against the Gram `G` before it (the `error_inflation^2 - 1` the model already
  computes), the out-of-sample residual `e`, `Delta b = G^-1 x w e / (1+h)` (the move the row makes, equal to `b - b_(i)` for the newest row), the recursive residual
  `v^2 = w e^2/(1+h)` (what `influence` folds): `DFBETAS_j = Delta b_j / (s_(i) sqrt(((G + w x x')^-1)_jj))`, `(G + w x x')^-1_jj = (G^-1)_jj - w (G^-1 x)_j^2/(1+h)`,
  `s_(i)^2 = Q_c/W_c` (the spread before the row, as `influence`), and `Cook = (w e^2 h/(1+h)) / (k s^2)` with `s^2 = (Q_c + v^2)/(W_c + w)` the full fit's.

Against the corrected oracles (largest gap over 20 seeds; relative unless said):

| case | gap |
|---|---|
| ARCH LM vs `het_arch(nlags=L)`, `lam = 1`, L = 1 / 4 / 10, 300 rows | 5.4e-15 / 6.4e-15 / 3.8e-15 |
| ARCH LM over a window of 100 by retraction, L = 4, vs OLS on the window's rows (lags cross the window's edge, as in the stream) | 9.8e-13 |
| ARCH LM under decay (half-life 40), weights, 10% zero-weight rows, zero first row, vs the two-pass weighted regression (statsmodels has no weights) | 3.9e-15 |
| White vs `het_white`, `interaction_terms=True`, k = 2 / 4 / 8 | 3.2e-15 / 2.1e-15 / 2.4e-15 |
| White vs `het_white`, `interaction_terms=False` (the squares only), k = 2 / 4 / 8 | 3.2e-15 / 5.7e-15 / 3.3e-15 |
| White over a window of 80 by retraction vs `het_white` on the window, k = 3 | 7.2e-14 |
| Spiegelhalter `z` vs MAPIE (calibrated and miscalibrated `p`, 400 rows) | 1.3e-15 |
| one-sided `p = 1 - Phi(z)` vs MAPIE `spiegelhalter_p_value` (C1: one-sided); two-sided `2 sf(abs z)` vs the same on MAPIE's `z` | 4.4e-16 abs; 7.8e-16 abs |
| DFBETAS / Cook's D of the newest row vs `OLSInfluence` on the prefix, `lam = 1`, unit weights, k = 3, rows 40, 79, 119 | 3.3e-12 (to the row's largest entry) / 3.3e-14 |
| same, weights in [.5, 2] (divisor corrected, below) | 3.4e-12 / 5.2e-14 |
| same, half-life 40, weights, 10% zero-weight rows (divisor corrected) | 7.0e-13 / 3.5e-14 |

**Where the oracle is not exact.** (1) statsmodels' own WLS influence is not the weighted leverage, so the weighted rows are checked against `OLSInfluence` on the
whitened rows (`sqrt(omega) x`, `sqrt(omega) y`), as in C10. (2) With weights or decay, statsmodels' `s^2` divides by `n - k` rows, where rheon's `influence` divides the recursive
residuals' weighted sum by their weight sum `W_c` (rows, at unit weights: the same). The two differ by the known factor `n_full / W_c` for Cook's and `sqrt(n_ext / W_c,ext)` for DFBETAS;
the table above is after that correction, so the **comparison is exact at unit weights and `lam = 1` and exact up to that stated divisor elsewhere**: rheon's convention stays the
weighted mean square it already uses for `influence` (not a default change). (3) ARCH under a window: the lags of a row near the window's edge are the stream's, not
the window's, so the windowed test equals OLS on the window's own rows, not `het_arch` on the window's residuals alone. (4) `OLSInfluence.dfbetas` is a leave-one-out loop on the final fit, so the per-row
comparison is the prefix's newest row (the `influence` model's own oracle convention).

**Size, and the choice on `w^2`.** Under H0 (iid homoskedastic residuals, k = 3, q = 9), half-life 100 rows, Kish `n`: White's test passed its 5% value on 4.7% of 300 streams (binomial sd 1.3%). Spiegelhalter's z
under H0 (calibrated `p`, half-life 30 rows, weights 1 / 1.5 / 2, 300 streams of 500): sd 1.03 with the squared weights in the denominator (stderr of the sd 0.04), 0.93 with the unit-weight formula read
naively (`sum w (1-2p)^2 p(1-p)`): conservative by the ratio `sum w^2 b / sum w b`. The squared form is proposed.

Decay and zero weight: a zero-weight row is not scored (the ARCH ring takes scored residuals only, as Ljung-Box's lags do) and only ages the sums. Against the same stream with the zero-weight rows removed and their gaps folded: Spiegelhalter's
`z` 1.4e-15; the ARCH and White accumulators inherit the accumulator's 7 ulps (517); a zero-weight first row leaves every sum zero and the tests withheld.

**State added** (per slot, bytes): ARCH LM `8 ((L+1)(L+2)/2 + (L+1) + 2 + L)`: 64 / 208 / 712 / 4472 at L = 1 / 4 / 10 / 30; White's full form `8 ((q+1)(q+2)/2 + q + 3)` with `q = k + k(k+1)/2`: 2.0 KB at
k = 5, 18 KB at k = 10, **216 KB at k = 20**, 259 KB at k = 21 (the 256 KiB cap, 262,144 bytes, is passed at k = 22; at 256,000 bytes, at k = 21): **White's full form is the quartic one, and the cap is
`k <= 21`** (the task's "a cap on k"). `interaction_terms=False`: `8 ((k+1)(k+2)/2 + k + 3)`, 88 bytes at k = 2, 448 at k = 8. Spiegelhalter 24 bytes (three sums). DFBETAS and Cook's: `8 k` for the diagonal of the inverse Gram
if refreshed with the solve (otherwise a per-row solve), and `W_c`, `Q_c` shared with `influence`. Nothing grows with the stream; each adds its bytes per group to the bank's total, and White's at k near 21 uses
the whole per-group allowance by itself.

**Changes a default: no.** All are new tests; `influence`'s convention (weighted mean square) is kept, and the weighted DFBETAS and Cook's are stated in it.

