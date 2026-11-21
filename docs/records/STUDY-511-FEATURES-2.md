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

## 521. Forecast comparison

Beside `rp.eval.diebold_mariano`'s recursion (the squared-loss differential, positive when `a` is closer; Newey-West variance with
Bartlett weights over `lags`; a normal reference; two-sided; `half_life=` gives EW sums). Everything below is an option or a new output; the defaults are unchanged.

**Diebold-Mariano: HLN, a t reference, any loss, a one-sided p.** With `d` the loss differential, `m` its EW mean, `W = sum w`, `n = W^2/sum w^2`
(Kish), `h = lags + 1`: `S = sum w (d-m)^2 + 2 sum_l (1 - l/h) sum_{i>=l} w_i (d_i-m)(d_{i-l}-m)` (a pair weighted by its later row), `dm = m sqrt(n) / sqrt(S/W)`,
HLN `dm * sqrt((n + 1 - 2h + h(h-1)/n)/n)`, p from the normal (today) or from t with `n - 1` df, one-sided `sf(dm)` for "`a` is closer". The centred lag products
expand about the one global `m`: `sum w (d_i-m)(d_{i-l}-m) = P_l - m A_l - m B_l + m^2 N_l`, four EW sums per lag over the rows that have a lag-`l` partner (`A_l = sum w d_i`,
`B_l = sum w d_{i-l}`, `N_l = sum w`, `P_l = sum w d_i d_{i-l}`) and a ring of the last `lags` differentials.

Against `dieboldmariano.dm_test` 1.1.0 (`variance_estimator="bartlett"`, `h = lags + 1`: C1: `"acf"`, its default, is a rectangular sum, not Bartlett; `harvey_correction` on and off;
the oracle's differential is `loss_a - loss_b`, so its statistic has the opposite sign; 200 rows, 10 seeds; losses squared, absolute, LINEX (a = 0.5), pinball (.8)):

| quantity | largest gap |
|---|---|
| statistic, HLN on and off, lags 0 / 2 / 5, four losses | 8.8e-16 relative |
| two-sided p with the t reference (`n-1` df), same cases | 7.3e-11 absolute |
| one-sided p ("a closer"), same cases | 3.7e-11 absolute |
| statistic under half-life 40, weights, 10% zero-weight rows, zero first row vs the two-pass weighted definition (lags 3, HLN) | 4.9e-15 |
| same vs the stream with zero-weight rows removed and gaps folded | 5.5e-16 |

The p gap is the oracle's: it takes the t tail from its own incomplete-beta routine (accurate to about 1e-11). C1's points hold: the oracle's p-value is always from the t, so a normal reference needs
another oracle, and here it is scipy's normal tail on the same statistic. **Size** (share of 400 null streams of 400 rows, iid differentials, final-row two-sided test at 5%, binomial sd 0.011; `n` = Kish):

| case | normal (today) | normal + HLN | t, n-1 | t + HLN |
|---|---|---|---|---|
| half-life 10, lags 0 / 4 | 0.068 / 0.095 | 0.063 / 0.053 | 0.060 / 0.078 | 0.053 / 0.048 |
| half-life 30, lags 0 / 4 | 0.065 / 0.075 | 0.063 / 0.053 | 0.058 / 0.063 | 0.055 / 0.048 |
| half-life 100, lags 0 / 4 | 0.065 / 0.065 | 0.065 / 0.055 | 0.065 / 0.063 | 0.065 / 0.055 |

So at a short effective sample and some lags, HLN with the t is what keeps the size near 5% (today's form: 9.5% at half-life 10, lags 4). Making it the default would be a change; **not proposed here**, offered as options.

**Anytime-valid test of the mean loss difference, beside `seqtest`** (task 508, "CUSUM, Page-Hinkley, ADWIN and the other change detectors", does not take it on its text: it tests a change in one stream, not
`E[d] <= 0` between two forecasts; `seqtest` is a sign test on one stream). C1: MAPIE `RiskMonitoring` monitors one model's risk against a reference threshold, so it is no oracle; the usable oracle is its bound function.
Online form, bounded differences `r = (d/B + 1)/2 in [0, 1]`: the conjugate-mixture empirical-Bernstein confidence sequence (Howard et al. 2021) as MAPIE spells it, `S_t = sum r`, `V_t = sum (r_i - mu_{i-1})^2`
(`mu_{i-1}` the running mean of the rows before, 1/2 at the first), `lower_t = max(mu_t - B(V_t, alpha/2)/t, 0)` with the running maximum, reject "a is not better" when the lower bound passes 1/2. No decay and a zero weight skips the row, as `seqtest`
(a forgetting test is not an e-process). Against `conjugate_mixture_empirical_bernstein_bound(r, v_opt, alpha, bound_side)` (150 rows, running intersection): **0 gap** lower and upper, to the bit (it is the same boundary function).
A second, longhand form: the betting e-process `E_t = prod (1 + lam_t d_t)`, `lam_t = clip(mu/(var + mu^2), 0, 1/2)` from the rows before (4 floats), valid by Ville's inequality; no library oracle, so it is checked by its guarantee.
Share of 100 streams of 1500 rows that **ever** rejected (CM-EB read every 10 rows, the betting process every row; differentials in [-1, 1], noise sd .35):

| stream | CM-EB | betting | median stop, CM-EB / betting |
|---|---|---|---|
| E[d] = 0 (the null's edge) | 0 | 0.01 | - / 210 |
| E[d] = -0.05 | 0 | 0 | - |
| E[d] = +0.10 | 1.00 | 1.00 | 299 / 87 |
| E[d] = +0.20 | 1.00 | 1.00 | 129 / 37 |

Both hold their level (the CS is conservative: 0 of 100 at the edge); the betting form stops about three times earlier. **Two things to settle on the Rust side:** the differential must be bounded
(`B` configured; squared loss is not bounded, and clipping `d` to `[-B, B]` is symmetric, so it keeps `E[d] = 0` only where the differential is symmetric under the null: a different test); and the boundary is a root of a gamma-function
equation (`gammainc`, `gammaln`: libm-grade), so a stored running bound is a libm result in the state (CLAUDE.md's cross-OS rule), which dropping the running intersection avoids (the CS stays valid without it).

**Giacomini-White's conditional test**: `Z_t = h_t d_t` for instruments `h_t = (1, d_{t-1}, ...)`, `stat = n Zbar' Omega^-1 Zbar`, `Omega = Gamma_0 + sum_l (1 - l/(L+1))(Gamma_l + Gamma_l')`, `Gamma_l = sum w Z_i Z_{i-l}' / W`, `chi2(q)`
(uncentred: the null is `E[Z] = 0`). Against statsmodels (relative gap, 300 rows, 20 seeds), q = 1 / 2 / 3 instruments with lags 0 / 3 / 5:

| oracle | iid, H0 true | AR(.3), H0 false |
|---|---|---|
| the definition by `S_hac_simple(Z, nlags=L, weights_bartlett)` (exact) | 8e-14 / 4e-16 / 1e-15 | 6e-14 / 5e-16 / 7e-16 |
| `OLS(d, h).fit(cov_type="HAC", maxlags=L, kernel="bartlett").wald_test(I, use_f=False)` (C1's) | 0.042 / 0.19 / 0.29 | 0.074 / 0.70 / 0.69 |

**Where the oracle is not exact.** C1's HAC Wald is asymptotically equivalent but not the same number: the Wald regresses `d` on `h` and builds the HAC from the fitted residuals `u`, the definition builds it from `d`
(`u = d - h'delta`); they differ by that plug-in, 4% to 29% on iid data and up to 70% where `d` is predictable. So the exact oracle used is statsmodels' own `S_hac_simple` on `Z`, and the HAC Wald is
reported as the second opinion, with this gap named. Giacomini-White itself is MISSING as a library function (C1). Size, 500 null streams at half-life 100 rows (binomial sd 0.010): 0.060 (q = 1, lags 0) and 0.052 (q = 2, lags 3).

Zero weight: a zero-weight row is skipped by all of the above (no differential, no instrument, no lag), the ageing is the clock's; verified for DM (above).

**State added** (per slot): DM `8 (4 + 4 L + L)` bytes for `L` lags (the lag-0 sums fold into the four scalars): 32 / 192 / 432 / 1232 at L = 0 / 4 / 10 / 30; CM-EB 48 (six scalars; the running bounds are the libm part); betting 32; GW
`8 ((L+1) q^2 + q L + q + (q-1) + 2)`: 32 (q = 1, L = 0), 216 (q = 2, L = 3), 1.8 KB (q = 4, L = 10), 17.9 KB (q = 8, L = 30). Each grows with its configured `L`, `q` only; nothing with the stream; each adds its bytes per group to the bank's total.

**Changes a default: no.** (The HLN-and-t form would be a better default by the size table; that is the coordinator's call, not made here.)

## 522. `hmm`: fixed-lag smoothing, `params=`, `spherical`, prior weights for given states

Beside `hmm.rs`'s Hamilton filter (`p~ = p Pi` from the filtered `p` the previous row left, uniform before the first; `f_l = N(x | mu_l, S_l)`; `loglik = ln sum p~ f`; `p = p~ f/sum`; transition counts
`A += w xi`, `xi_kl = p_k(t-1) Pi_kl f_l / sum`, `Pi = (A + tau)/sum(A + tau)`; each state's accumulator takes the row at weight `w p_l`). A zero-weight row leaves the filtered `p` where it was (rule 9, and the Rust test
`a_zero_weight_row_is_clock_alone`), so the filter, the counts and the smoother's ring skip it; its clock ages the counts and the states as the gap would. Oracle: hmmlearn 0.3.3 `GaussianHMM`.

**Fixed-lag smoothing under an embargo.** `gamma_{t-L|t} ~ p_{t-L} * beta`, `beta_t = 1`, `beta_{s-1} = Pi (f_s * beta_s)` for `s = t .. t-L+1` (normalised): the smoothed posterior of row `t - L` given rows to `t`,
emitted at row `t` (so the output is `L` rows late by construction: the embargo is the lag; it is available as a feature or label only for rows `L` back). State: the last `L` likelihood vectors and **one** filtered vector `alpha_{t-L}`,
advanced a row at a time by `alpha_{t-L} <- norm((alpha_{t-L-1} Pi) * f_{t-L})` (the two-ring form, a ring of filtered vectors as well, is `2x` larger and equal to 1e-16), `8 K (L + 2)` bytes with `p`. At **fixed parameters** it is exact; under learning the
stored `f` and `alpha` were made at the then-current parameters and `Pi` at the read is the current one, so the read is the smoother of a drifting model (an approximation: nothing to compare to; hmmlearn has no online mode).

Against hmmlearn at fixed parameters (`params=""`, `startprob_ = (1/K) 1'Pi`: C1: the start distribution enters the prefix smoother, and rheon's first `p~` is `(1/K) 1'Pi`; 10% zero-weight rows skipped, zero first row; lags 1, 3, 8):

| case | `score` (summed `ln p(x_t | x_<t)`), rel | filtered `p` = `predict_proba(X[:t+1])[t]`, abs | smoothed `p` = `predict_proba(X[:t+1])[t-L]`, abs | advance form vs two rings |
|---|---|---|---|---|
| full, K = 2, d = 2 | 8.3e-16 | 2.0e-14 | 2.8e-14 | 1.1e-16 |
| full, K = 3, d = 3 | 1.3e-15 | 5.5e-14 | 6.3e-14 | 1.1e-16 |
| spherical, K = 2, d = 2 | 1.0e-15 | 2.8e-14 | 3.0e-14 | 1.1e-16 |
| spherical, K = 3, d = 3 | 1.1e-15 | 4.0e-14 | 5.4e-14 | 2.2e-16 |

**`spherical` covariance.** `S_l = s_l I`, the scalar `s_l' = a s_l + a b |delta|^2/d` (the mean-form step with the mean squared deviation), 8 bytes a state against `8 d(d+1)/2` (full: 24 / 440 / 10200 at
d = 2 / 10 / 50) and `8 d` (diag). The density (above) and the learned variance both match hmmlearn: against `GaussianHMM._do_mstep` (`covariance_type="spherical"`, `params="mc"`) on the **same recorded responsibilities**, 400 rows, K = 2 / 3:
means 6.8e-16 / 8.0e-16, spherical variance 2.5e-15 / 2.6e-15 (relative). The comparison is of the update formulas given the responsibilities (exact); the responsibilities themselves are the filter's, not EM's smoothed ones.
hmmlearn's `spherical` takes `covars_` as `(K,)` for scoring but EM stores `(K, d)`; setting `_covars_` as the tiled form is what EM needs (a library quirk met, not worked around beyond that).

**Prior weights for given states.** Given means enter as a prior worth `kappa` rows (the mean form started at `W = kappa`, `m = mu_0`; today's pair is `kappa = 1`), given variances with hmmlearn's `covars_prior`
and `covars_weight`: `mu = (kappa mu_0 + sum g x)/(kappa + sum g)`, `s = (covars_prior + W C)/(max(covars_weight - 1, 0) + sum g)` with `W C` the centred sum of squares about the posterior mean (which
carries the `kappa (mu - mu_0)^2` term hmmlearn writes out) and `sum g` the data mass without the prior's. Against `_do_mstep` with `means_prior`, `means_weight`, `covars_prior`, `covars_weight`
(`half_life = inf`; same responsibilities), (`kappa`, `covars_prior`, `covars_weight`) = (1, .01, 1) / (5, .2, 3) / (20, 1, 10): means 8.6e-16 / 1.9e-15 / 2.3e-15, variance 2.3e-15 / 4.5e-15 / 2.9e-15. **Under a decay the prior fades**
as `kappa 2^(-t/h)` (after 600 rows at half-life 30 the prior is 2e-5 of a state's weight), where hmmlearn's is permanent: a starting point, as the given states already are; the exactness is at `half_life = inf`. The default
`kappa = 1` is today's behaviour, so the option changes nothing until set.

**Partial learning, `params=`.** Letters `s t m c` gate which parts learn (`s`: the start vector is not learned online, as rheon's is the uniform `p`; `t`, `m`, `c` gate the counts, the means and the variances). The frozen parts are asserted
**bit-equal** to what was given. An online pass learns with the filtered, not the smoothed, responsibilities, so there is **no exact oracle** for the learned values: hmmlearn's EM (`params=letters`, 100 iterations, same perturbed start) is the
statistical one. Largest error against the generating values after 4000 rows, 2 states, d = 2, mean over 4 streams:

| letters | online pass | hmmlearn EM |
|---|---|---|
| `t` | 0.0111 | 0.0107 |
| `m` | 0.0395 | 0.0396 |
| `c` | 0.0416 | **defect: diverges** |
| `mc` | 0.184 | 0.0418 |
| `tmc` | 0.465 | 0.0423 |

One part to name: hmmlearn 0.3.3 accumulates its `post` and `obs` statistics only when `'m'` is in `params` (`_needs_sufficient_statistics_for_mean`), so **`params="c"` alone diverges** (variances to 1e8 in the first
M-step); C1's CONFIRMED stands for names and meaning, with this defect. Single-parameter learning (`t`, `m`) matches EM; learning jointly from a perturbed start, one online pass trails batch EM (0.18 and 0.47 against 0.04): the usual
cost of the filtered responsibilities, and a reason `params=` is useful (learn the transition matrix with the emissions given, which is the case that matches).

**State added** (per `hmm` group): the fixed-lag smoother `8 K (L + 2)` bytes: K = 2, L = 5: 112; K = 3, L = 20: 528; K = 5, L = 100: 4 KB; K = 8, L = 1000: 64 KB; the 256 KB per-group cap bounds `L` at
`262144/(8K) - 2` (K = 8: 4094; K = 2: 16382). A spherical state `8` bytes of variance in place of `8 d (d+1)/2`: it **reduces** state. Prior weights and `params=`: 0 (configuration: `8` bytes each). Nothing grows with the stream.

**Changes a default: no.** Four options; `kappa = 1`, all letters, a full or diag covariance and no smoother remain what they are.

## 523. `bocpd`: a run-length hazard and a known-variance emission

Selected in part. C1: the reference is `bayesian-changepoint` 1.2.2 (formerly `bayesian_changepoint_detection`; PyTorch, `torch` 2.15.0+cpu here, its run-length state **float32**): `online_changepoint_detection(data, hazard_function, likelihood_model)`,
`negative_binomial_hazard(k, p, r)` CONFIRMED, `NormalKnownVariance(variance, mu, prior_variance)` CONFIRMED. **Excluded, oracle MISSING (C1):** the **logistic hazard** (Turner et al. 2009; not in the library) and the **known-mean
(variance-change) emission** (no library class). Both would have an exact definition-based oracle (the all-paths enumeration used in 523b below works for any hazard and emission), so they can be reopened on the coordinator's word; they were not built.

Beside `bocpd.rs` (Adams and MacKay, per-row hazard, `r` the learned rows in a run before the row): growth `P(r+1) = P(r) pi(r) (1 - H(r))`, changepoint `P(0) = sum_r P(r) pi(r) H(r)`; today `H(r) = 1/hazard` for every `r`. New:

- **Negative-binomial hazard**: `P(L = t) = C(t-1, k-1) p^k (1-p)^(t-k)`, `H(r) = P(L = r+1)/P(L >= r+1)`; 0 for `r + 1 < k` (a run shorter than `k` cannot end), `p` at `k = 1`, rising to `p` as the run grows. The recursion changes in one place: `H` is a vector read at each held run's `len`.
- **Known-variance normal emission**: the segment mean `~ N(mu_0, tau^2)`, the observation variance `s2` known; a run keeps its mean (`bocpd.rs`'s `len`, `n`, mean and mean's low part) and no scatter; `v = 1/(1/tau^2 + n/s2)`,
  `m = v (mu_0/tau^2 + n xbar/s2)`, predictive `N(m, v + s2)`.

Against the oracles (hazard exactness in float64; the recursion against the library's float32 `R`):

| check | gap |
|---|---|
| `H(r)` from scipy's `nbinom` pmf and survival vs the pmf ratio from a long cumulative sum, k = 1 / 3 / 5 / 10 | 2.8e-17 / 5.3e-16 / 9.4e-16 / 2e-18 abs |
| same vs the library's `negative_binomial_hazard` table (float32) | 4.5e-10 / 1.9e-9 / 3.7e-9 / 5.7e-11 abs |
| `H(r) = 0` for `r + 1 < k`; `H(0) - p` at `k = 1` | exactly 0; -1e-17 |
| the recursion vs summing all `2^T` run-length paths from the definition (T = 10; constant `lam = 6`, NB (2, .3), NB (3, .2)) | 2.8e-16 / 1.1e-16 / 1.1e-16 abs in the posterior |
| the whole run-length posterior `R[r, t]` vs `online_changepoint_detection` with `NormalKnownVariance(1, 0, 4)`, 160 rows with two level shifts; constant `lam = 100`, NB (3, .02), NB (10, .05) | 9.2e-7 / 1.3e-6 / 1.6e-6 abs; 0 MAP run-length mismatches |

**Where the oracle is not exact:** the library keeps its run-length statistics in float32 (C1), so against it the posterior agrees to float32's 1e-6, not to bit level; the all-paths enumeration and scipy are the float64 oracles, and
the recursion is exact against them. Library pruning (`BaseLikelihood.prune`) was off. A `torch` dependency would enter the test group only if the library were an oracle there; the enumeration needs none.

**What the hazard buys** (60 streams: 100 rows `N(0, 1)` then a +2 shift for 100; known variance; `P(r <= 1) > 0.5` as an alarm): constant `lam = 100`: median delay 3 rows, 56/60 detected, 1/60 early alarm; NB(5, .05)
(mean 100): median delay 2 rows, 58/60 detected, 1/60 early alarm. A small, honest gain on this stream: the NB hazard's value is in streams whose segments are rarely short.

Decay and zero weight: a per-row hazard counts learned rows, so a zero-weight row is a **no-op** for the recursion: no run grows and no hazard applies; against the stream without those rows (20%) the posterior is bit-equal (gap 0). The NB hazard
is therefore defined **per learned row**; a run-length hazard in *clock* units (`hazard_on_clock`'s sense) is not defined by this form and is not proposed. Row weights enter as `bocpd` already does, `w / w_bar` in the statistics and as the power on the predictive; not changed by either feature.

**State added.** The NB hazard adds **no group state**: its table `H(r)`, `r < max_run`, is `8 max_run` bytes derived from `(k, p)` (80 KB at the default 10,000) and shared by every group of the spec; the 256 KB cap is for state, so count it once per spec at most. The known-variance emission
needs `8 (3 + 2d)` bytes a held run (log-mass, `len`, `n`, mean, mean's low part) against `diag`'s `8 (3 + 3d)` (it drops the scatter): d = 1 / 5 / 20: 40 / 104 / 344 against 48 / 144 / 504; the runs the 256 KB per-group cap holds:
6553 / 2520 / 762 against 5461 / 1820 / 520. **A pre-existing point to state**: the default `max_run` is 10,000, so a stationary stream that fills it holds 480 KB at `d = 1` under `diag`, over the per-group cap; the Rust doc measures far fewer runs held in practice
(2,001 after 2,000 rows, 7,036 at most by 20,000 on a stationary stream, a few dozen after a shift), so what is capped is the worst case. Neither feature changes it; the cap, not the hazard, is what bounds `max_run`.

**Changes a default: no.** `hazard` stays the constant per-row hazard; the NB hazard and the known-variance emission are options.

## 524. Regression variants

Task 524 lists nine items. Selection by C1's verdicts: **built** evidence-maximized ridge (`BayesianRidge`, CORRECTED on exactness), averaged `sgd` and `pa` (`SGDRegressor(average=True)`, CONFIRMED, `pa` spelling CORRECTED),
`positive=True` (`Lasso`, `ElasticNet`, CONFIRMED), and (next section) the redescending norms and Huber's scale (`RLM`), the `sgd` elastic net with the `optimal` schedule, and Vowpal Wabbit's importance-invariant step;
**excluded, oracle MISSING (C1): per-feature penalty weights on `lasso`** (scikit-learn has no such parameter; the column-rescaling device is an identity, not an independent oracle). "Several quantiles in one spec" is plumbing (one `quantile`
model per level; `QuantReg` is one fit per `q`, C1): its state is the sum of the levels' and it needs no new recursion, so it is listed, not studied.

### 524 (a) Evidence-maximized ridge at solve time with `pred_std`

At a solve, from the accumulator over `(x, y)` (the centred Gram `G = n C_xx`, `X'y = n c_xy`, `y'y = n C_yy`, with `n` Kish's count and the weights scaled to sum to `n`): the MacKay / Tipping fixed point, eigen-decomposing `G`
once (`e_i`), iterating `coef = V ((V' X'y)/(e + lam/alpha))`, `sse = y'y - 2 coef'X'y + coef'G coef`, `gamma = sum alpha e_i / (lam + alpha e_i)`, `lam = (gamma + 2 l_1)/(|coef|^2 + 2 l_2)`, `alpha = (n - gamma + 2 a_1)/(sse + 2 a_2)`, the
intercept unpenalised (centring), `pred_std = sqrt(1/alpha + (x - m)' Sigma (x - m))`, `Sigma = (alpha G + lam I)^-1`, `alpha_0 = 1/(var y)`, `lam_0 = 1`; the same stopping rule as sklearn (sum of absolute coefficient changes under `tol`).

**A correction to C1's note on `n`.** C1 wrote that `sample_weight` rescales rows by `sqrt(w)` so that `n` in the fixed point is the row count. In scikit-learn 1.9.1's source the `alpha` update uses `sw_sum = sample_weight.sum()` (and the weighted `y_var`
for `alpha_0`), not the row count. So rheon's "weights scaled to Kish `n`" **is** `BayesianRidge`'s own convention when it is handed weights that sum to Kish's `n`: the comparison is exact under decay and row weights, not only at unit weights.

Against `BayesianRidge(alpha_1, alpha_2, lambda_1, lambda_2, tol=1e-14, max_iter=5000).fit(X, y, sample_weight=w')` (C1: the defaults are MAP with `1e-6` hyperpriors and `tol=1e-3`, so the comparison gives rheon the same four pseudo-counts and a tiny `tol`; also with
all four 0, pure evidence maximization), k = 6, 160 rows, 20 seeds, relative gaps:

| case | coef | intercept | `alpha_` | `lambda_` | `pred_std` |
|---|---|---|---|---|---|
| `lam = 1`, unit weights, priors `1e-6` | 1.3e-13 | 5.4e-15 | 4.4e-14 | 9.3e-14 | 6.8e-14 |
| `lam = 1`, unit weights, priors 0 | 1.3e-13 | 5.2e-15 | 4.6e-14 | 9.2e-14 | 6.7e-14 |
| `lam = 1`, weights in [.5, 2], priors `1e-6` | 1.1e-13 | 1.0e-14 | 2.7e-14 | 5.2e-14 | 1.4e-14 |
| half-life 50, weights, 10% zero-weight rows, zero first row, priors `1e-6` | 9.8e-14 | 6.6e-15 | 2.5e-14 | 2.6e-14 | 1.9e-14 |

At sklearn's defaults (`tol = 1e-3`, `max_iter = 300`) the same rule gives the same iteration count on every seed and the same fit (7e-14). C1's caveat stands: the update "does not guarantee that the marginal likelihood is increasing"; the loop is sklearn's, so the guarantee is the same.
Warm-starting `(alpha, lam)` from the previous solve (one row later) takes 3.4 iterations against 4.2 cold (the fit differs only within `tol`). A zero-weight row adds nothing to `G`, `X'y` or `n`; a zero-weight first row leaves `n = 0`, the solve withheld.
**State added: 16 bytes** (`alpha`, `lam`, the warm start); the eigendecomposition `V, e` and `Sigma` are solve scratch, `8 k^2` bytes if cached for per-row `pred_std` (derived, rebuilt on load, not persisted); cost `O(k^3)` per solve, `O(k^2)` per `pred_std` row.

### 524 (b) Averaged `sgd` and `pa`, weighted by the decay

The average is the EW mean of the iterates in the mean form: `W' = lam W + w`, `bbar += (w/W')(b' - bbar)`, one more k_total-vector. At `half_life = inf` and unit weights it is the uniform (Polyak-Ruppert) average of every iterate. Against
`SGDRegressor(average=True)` fed one row at a time by `partial_fit(shuffle=False)`, 300 rows, 20 seeds (rheon's plain step `b -= lr (p - y) z w`; penalty `None`, constant rate), absolute gap in the averaged coefficients:

| case | coefficients | intercept |
|---|---|---|
| `sgd`, k = 3 / 10, `lr = .01`, `half_life = inf`, unit weights | 2.7e-15 / 4.0e-15 | 5.6e-17 / 2.2e-16 |
| `pa` (PA-I, `C = 1`, `eps = .1`, **no intercept**), k = 3 / 10, `half_life = inf` | 3.6e-15 / 5.8e-15 | |

C1: `PassiveAggressiveRegressor` is deprecated since 1.8 (removed in 1.10), so the `pa` oracle is `SGDRegressor(loss="epsilon_insensitive", penalty=None, learning_rate="pa1", eta0=C)`. **Where it is not exact:** rheon's `pa` keeps the intercept inside `|z|^2` (the augmented form,
Crammer et al.) where sklearn keeps it outside, so the `pa` comparison is on the no-intercept case (the one where the two steps are the same step); and rheon's `pa` row weight (`min(w, 1) tau`) has no sklearn counterpart. The decay-weighted average is rheon's: sklearn's is
uniform, so the oracle is exact only at `half_life = inf` (C1), and under decay, weights and zero-weight rows (half-life 40, 10% zero-weight rows, zero first row) it is checked against the two-pass `sum omega b_i / sum omega` over the recorded iterates (7.7e-16) and
against the stream with the zero-weight rows removed and their gaps folded (1.6e-16). Zero weight: no new iterate, the average only ages (its `W`).
**State added: `8 k_total + 8` bytes** a target (k = 20: 176; k = 100: 816), doubling an `sgd` target's coefficient state; nothing grows with the stream.

### 524 (c) `positive=True` on the streaming lasso

`lasso.rs`'s coordinate step on the standardised correlation Gram, `b_i = soft(rho_i, l l1) / (C_ii + l (1 - l1))`, with `positive`: `b_i = max(rho_i - l l1, 0)/(C_ii + l (1 - l1))`. Against `Lasso(positive=True)` and `ElasticNet(l1_ratio, positive=True)`
fitted on the standardised features with `sample_weight` = the EW weights (C1: sklearn rescales the weights to sum `n`, so `alpha` is on the same per-weight scale as a decayed fit; `tol=1e-14`), alpha .15, k = 8, 200 rows, 20 seeds; largest gap in the standardised coefficients and the
intercept; the unconstrained step against sklearn as the control; zeros per fit are the coefficients the constraint holds at 0:

| case | `positive` gap | control (unconstrained) | zeros per fit |
|---|---|---|---|
| `lam = 1`, unit weights, lasso | 2.7e-13 | 6.2e-13 | 4.2 |
| `lam = 1`, unit weights, elastic net `l1_ratio` .5 | 8.9e-14 | 1.5e-13 | 4.0 |
| half-life 50, weights, 10% zero-weight rows, zero first row, lasso | 2.9e-13 | 1.2e-12 | 4.4 |
| half-life 50, weights, zero weight rows, elastic net .5 | 1.1e-13 | 1.4e-13 | 4.2 |

Positivity in the standardised coordinates is positivity in the raw ones (the scale is positive), so one constraint serves both. Decay and zero weight change only the Gram. **State added: 0** (the warm start is the same coefficient vector). It changes no default (`positive = False`).

### 524 (d) Redescending norms for `huber`

`huber` reweights each row by the robust weight of its **prior** (out-of-sample) residual, `w(u) = psi(u)/u`, `u = r/(c s)`, into the mean-form Gram. A redescending norm changes the weight function only: Tukey's biweight `(1 - u^2)^2` for `|u| < 1`, else **exactly 0**.
**Exact part:** each weight function against `statsmodels.robust.norms.*.weights` (C1: `TukeyBiweight`, not `TukeyBisquare`; `HuberT(t)`, `Hampel(a, b, c)`, `AndrewWave(a)`, `RamsayE(a)`), 4801 points in [-12, 12]: gap 0 for Huber, Tukey (c = 4.685), Hampel (2, 4, 8)
and Ramsay (a = .3), 1.7e-16 for Andrews (a = 1.339); 2927 / 3118 / 1602 of the points have weight exactly 0 for Tukey / Andrews / Hampel. A row at weight 0 is a legal zero-weight row for the Gram (it advances the clock and learns nothing, rule 9); whether it still counts in `n_eff`
and `min_weight` is a decision to make on the Rust side (today's Huber weight is never 0, so the question is new).

**Statistical part, where the oracle is not exact.** `RLM` is converged IRLS; rheon's `huber` is one pass. C1: Tukey is non-convex, so the oracle must start from the same fit. With the scale **fixed** (a `FixedRLM` subclass returns the clean noise's 1; `update_scale=False`), the first `3 (k+1)`
rows at weight 1, k = 3, mean over 10 seeds of the largest coefficient error against the truth, and the gap between the one-pass and the `RLM` coefficients:

| stream | OLS | Huber one-pass / RLM | Tukey one-pass / RLM | gap Huber | gap Tukey |
|---|---|---|---|---|---|
| n = 400, clean | 0.069 | 0.076 / 0.076 | 0.075 / 0.076 | 0.011 | 0.014 |
| n = 400, 15% one-sided outliers (sd 8, mean +6) | 0.199 | 0.129 / 0.086 | **0.573** / 0.087 | 0.086 | **0.54** |
| n = 3000, clean | 0.027 | 0.030 / 0.028 | 0.030 / 0.028 | 0.0045 | 0.0056 |
| n = 3000, 15% one-sided outliers | 0.089 | 0.039 / 0.034 | 0.040 / 0.031 | 0.018 | 0.025 |

So the one-pass redescending fit is within 0.025 of the converged `RLM` at n = 3000 and **fails at n = 400 with heavy contamination** (error 0.57, worse than Huber's 0.13): the first rows set a biased fit, and a redescending weight then gives the outliers near the fit weight 0 and the clean rows far from the biased
fit weight 0 too, so it cannot leave the basin. It is the non-convexity C1 named, arising online: a redescending norm needs a robust start (the `huber_refit` of the first rows, or a Huber warm-up) before it is switched on. **Huber's proposal-2 scale was not built**: the oracle C1 found is
`statsmodels.robust.scale.HuberScale`, which "differs slightly" from the textbook proposal 2 (an `h` constant of `n, p, d`) and `scale.Huber` is location and scale only, so there is no oracle for the quantity; recorded, not studied. **State added: 0** (a weight function; the scale state `huber` keeps is unchanged). It changes no default (`huber_norm = "huber"`).

### 524 (e) `sgd`: the elastic-net penalty and scikit-learn's `optimal` schedule

`sgd`'s L2 today is `g_i = d z_i w + l2 b_i`. The addition is scikit-learn's: the L2 as a shrinkage `b <- b (1 - (1 - l1_ratio) eta alpha)` taken before the gradient step, and the L1 by **Tsuruoka et al.'s cumulative penalty** (`u += l1_ratio eta alpha`, each `b_j` truncated by `u + q_j` or `u - q_j`, `q_j` the penalty already
applied), with the schedules `constant`, `invscaling` (`eta_0 / t^power_t`) and `optimal` (`eta_t = 1/(alpha (t_0 + t - 1))`, `t_0 = 1/(eta_0 alpha)`, `eta_0 = sqrt(1/sqrt(alpha))` for the squared loss: the source's `typw / max(1, gradient(1, -typw))`). I read the update from scikit-learn's `_sgd_fast.pyx.tp` (shipped in the wheel).
Row weight multiplies the update; a zero-weight row is skipped, and `t` counts learned rows. Against `SGDRegressor(penalty, alpha = .05, learning_rate, shuffle=False).partial_fit(X, y, sample_weight)`, 300 rows, 20 seeds, row weights in [.5, 2], largest absolute gap (coef / intercept):

| schedule | l2 | l1 | elasticnet (l1_ratio .4) |
|---|---|---|---|
| `optimal` | 5.3e-15 / 4.4e-16 | 2.7e-15 / 4.2e-16 | 6.2e-15 / 4.4e-16 |
| `invscaling` | 1.0e-15 / 8.3e-17 | 5.6e-17 / 1.1e-16 | 7.2e-16 / 8.3e-17 |
| `constant` | 3.3e-15 / 2.2e-16 | 1.1e-16 / 5.6e-17 | 2.2e-15 / 2.2e-16 |

(C1's CORRECTED holds: `optimal` is `SGDClassifier`'s default, not `SGDRegressor`'s, whose default is `invscaling`; `t0` is not a parameter, and the source's rule above is what the exact comparison used.) **A trap met:** scikit-learn re-creates the cumulative penalty `u, q` on **every** `partial_fit` call, so
a row-by-row `partial_fit` loses the L1 memory and cannot be the oracle for an L1 stream; one call over all rows is the stream, and is the comparison used. Zero-weight rows (15%, a zero-weight first row) against the stream without them: gap 0 (the schedule and the penalty both count learned rows). `sgd` has no coefficient decay, so decay does not enter.
With a large `eta_0` at small `alpha` the `optimal` schedule's first steps overshoot (`eta_1 = sqrt(1/sqrt(alpha))`: 3.2 at `alpha = .01`) on unscaled features, in scikit-learn as in the numpy form; its tests used features of norm below 1. **State added:** `q` (one `k`-vector), `u` and `t`: `8 (k + 2)` bytes a target under an L1 part (k = 20: 176; k = 100: 816), 16 bytes (`t` and a flag) under L2 with `optimal` alone.

### 524 (f) Vowpal Wabbit's importance-invariant step

Karampatziakis and Langford's importance-aware update, for the squared loss: instead of `b += lr w (y - p) z`, `b += (y - p) (1 - exp(-lr w |z|^2))/|z|^2 z`, so `p' = y + (p - y) exp(-lr w |z|^2)`: a heavy row moves the prediction to its target and stops, where the plain step overshoots. Row weight `w` is the importance.
Against VW 9.11.9 `--sgd --invariant` (C1: `--invariant` is real and on by default; `--sgd` switches off the adaptive and normalized updates, leaving the plain-plus-invariant step; VW's squared loss is `(p - y)^2`, so its `-l` is half of the `1/2 (p - y)^2` rate used here; explicit constant feature; predictions clipped off with `--min_prediction/--max_prediction`;
the example text must carry plain floats, a numpy 2 `repr` silently zeroes the features), row-by-row predictions and final weights, 200 rows, 10% zero-weight rows, 5 seeds (relative; **VW keeps float32 weights, so the floor is float32's**):

| case | predictions | weights |
|---|---|---|
| `lr` .05, importance in [.5, 2] | 8.3e-7 | 1.3e-7 |
| `lr` .05, importance in [.5, 20] | 8.0e-7 | 1.2e-7 |
| `lr` .5, importance in [.5, 2] | 7.6e-7 | 1.9e-7 |
| `lr` .5, importance in [.5, 20] | 9.4e-7 | 2.2e-7 |

At `lr w |z|^2` near 1e-4 the two steps differ by 3.8e-7 in the coefficients (first order); at a huge importance (1e6) the plain step overshoots the target on 50/50 rows, the invariant one leaves a residual of at most 4.4e-16. A zero-weight row is a no-op (`1 - e^0 = 0`). Decay: none (no coefficient decay).
**State added: 0.** It is an option (`invariant = false` by default), and agrees with the plain step to first order where `lr w |z|^2` is small, so a default of today is unchanged; **if it were made the default, outputs would move** at large weights, which is the coordinator's call, not proposed here.

**Not built in 524, and why:** per-feature penalty weights (oracle MISSING, above); Huber's proposal-2 scale (oracle not the same quantity); several quantiles in one spec (plumbing). **Changes a default: no**, for any of the built items.

