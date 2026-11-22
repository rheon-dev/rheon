# EWM moments: oracle scan and design (PLAN 554, part 1)

A record, written 2026-11-22 before any implementation: which third-party
library computes each quantity the block names, how far each sits from the
definition written longhand, the online update and its state, the window
forms, the numerical-stability argument, and the choices the request leaves
open. The scan is `scripts/oracle_scan_554.py`; every number below is its
output (python 3.13.18, numpy 2.5.5, scipy 1.18.1, statsmodels 0.15.0,
polars 1.37.1, river 0.27.0, all in the dev group).

## 1. The quantities, defined

A row carries a value (or a pair), its own weight `w_t >= 0`, and the decay
`lam_t` the clock gives it (`lam^(dt/h)` on a clock, `lam` a row). After row
`T` row `t` holds the weight `omega_t = w_t * prod_{s=t+1..T} lam_s`, and with
`W = sum omega_t` every moment is a **frequency-weighted plug-in** moment,
normalised by `W`:

```text
raw            mu'_p      = sum omega_t x_t^p / W
central        mu_p       = sum omega_t (x_t - m)^p / W,     m = mu'_1
cross          mu_{a,b}   = sum omega_t (x_t - m_x)^a (y_t - m_y)^b / W
lagged cross   mu^k_{a,b} = the cross moment of the pair stream (x_t, y_{t-k})
cumulant       kappa_n    = mu_n - sum_{j=2}^{n-2} C(n-1, j-1) kappa_j mu_{n-j}   (n >= 2)
standardized   s_p        = mu_p / mu_2^(p/2)
```

The cumulant recursion, from central moments (`kappa_1 = m`, `mu_1 = 0`),
gives `kappa_2..kappa_4 = mu_2, mu_3, mu_4 - 3 mu_2^2`,
`kappa_5 = mu_5 - 10 mu_3 mu_2`, `kappa_6 = mu_6 - 15 mu_4 mu_2 - 10 mu_3^2 + 30 mu_2^3`.
The special cases: skew `g1 = s_3`; excess kurtosis `g2 = s_4 - 3 = kappa_4 / kappa_2^2`;
autocovariance `mu^k_{1,1}` with `y = x`; autocorrelation the same over the
legs' standard deviations; the Roll covariance `mu^1_{1,1}` of price changes
(the change taken upstream in Polars). With `w = 1` and `lam = 1` these are
scipy's `moment`, `skew(bias=True)` and `kurtosis(bias=True)`.

This is the convention `ew_cov` keeps under `bias=True` and Polars'
`ewm_var(bias=True)` computes. The unbiased forms (Fisher's k-statistics,
the adjusted `G1` and `G2`) are different estimators, not different
roundings; see question Q6.

## 2. The oracle scan

Small case: 40 rows of `3 + Gamma(2, 1.5)`, `y = 0.6 x + N(0, 1)`, seed 554;
EW weights `lam = 0.9` with row 17's own weight 0. "Gap" is
`|library - longhand| / max(|longhand|, 1)`, the longhand over Python floats
with `math.fsum`, sharing no arithmetic with the library.

| Quantity | Call | Version | Exact? | Gap |
|---|---|---|---|---|
| raw moment, p = 2..6 | `scipy.stats.moment(x, order=p, center=0)` | scipy 1.18.1 | exact | 1.7e-16 |
| central moment, p = 2..6 | `scipy.stats.moment(x, order=p)` | scipy 1.18.1 | exact (1/n, plug-in) | 2.4e-16 |
| EW raw moment, p = 2..6 | `numpy.average(x**p, weights=w)` | numpy 2.5.5 | exact; numpy sums, the definition is by hand | 1.4e-16 |
| EW central moment, p = 2..6 | `numpy.average((x-m)**p, weights=w)` | numpy 2.5.5 | exact, same | 2.0e-16 |
| EW variance | `DescrStatsW(x, weights=w).var` (ddof 0) | statsmodels 0.15.0 | exact, order 2 only | 6.0e-16 |
| EW covariance | `numpy.cov(x, y, aweights=w, bias=True)` | numpy 2.5.5 | exact | 0 |
| EW mean, decay only | `pl.Series.ewm_mean(alpha=0.1, adjust=True)` | polars 1.37.1 | exact; no per-row weights | 1.4e-16 |
| EW variance, decay only | `pl.Series.ewm_var(alpha=0.1, adjust=True, bias=True)` | polars 1.37.1 | exact under `bias=True`; `False` is Kish's correction | 2.0e-16 |
| cumulants k1..k6, plug-in | `statsmodels...moment_helpers.mnc2cum(raw moments)` | statsmodels 0.15.0 | exact, but fed raw moments it cancels | 5.0e-13 |
| cumulants k2..k6, plug-in | `moment_helpers.mc2cum([0, mu_2..mu_6])` | statsmodels 0.15.0 | exact; fed central moments, no cancellation | 3.5e-16 |
| k-statistics k1..k4 | `scipy.stats.kstat(x, n)` | scipy 1.18.1 | exact for Fisher's unbiased k-statistic | 0 |
| k4 against the plug-in kappa_4 | `scipy.stats.kstat(x, 4)` | scipy 1.18.1 | **not** the plug-in cumulant | 1.1e-1 |
| variance of k2 | `scipy.stats.kstatvar(x, 2)` | scipy 1.18.1 | exact: `(2n k2^2 + (n-1) k4) / (n(n+1))` | 0 |
| skew g1 | `scipy.stats.skew(x, bias=True)` | scipy 1.18.1 | exact | 0 |
| skew G1 | `scipy.stats.skew(x, bias=False)` | scipy 1.18.1 | exact for `G1 = g1 sqrt(n(n-1))/(n-2)`, not g1 | 1.1e-16 |
| excess kurtosis g2 | `scipy.stats.kurtosis(x, bias=True)` | scipy 1.18.1 | exact | 0 |
| excess kurtosis G2 | `scipy.stats.kurtosis(x, bias=False)` | scipy 1.18.1 | exact for G2, not g2 | 0 |
| standardized, p = 3..5 | `moment(x, p) / moment(x, 2)**(p/2)` | scipy 1.18.1 | exact | 1.1e-16 |
| skew and kurtosis | `moment_helpers.mc2mvsk([m, mu2, mu3, mu4])` | statsmodels 0.15.0 | exact (g1, excess g2) | 0 |
| rolling skew, 10 rows | `pl.Series.rolling_skew(10, bias=True)` | polars 1.37.1 | exact (g1) | 4.1e-15 |
| rolling excess kurtosis, 10 rows | `pl.Series.rolling_kurtosis(10, bias=True)` | polars 1.37.1 | exact (g2) | 1.6e-14 |
| skew, online | `river.stats.Skew(bias=True)` | river 0.27.0 | exact (g1); no weights | 5.6e-16 |
| excess kurtosis, online | `river.stats.Kurtosis(bias=True)` | river 0.27.0 | exact (g2); no weights | 8.9e-16 |
| cross central `E[dx^a dy^b]` | `numpy.mean((x-mx)**a * (y-my)**b)` | numpy 2.5.5 | exact | 7.6e-16 |
| autocovariance, lag 1..3 | `statsmodels.tsa.stattools.acovf(x, fft=False)[k]` | statsmodels 0.15.0 | exact: 1/n, full-sample mean | 5.6e-17 |
| autocovariance, adjusted | `acovf(x, adjusted=True, fft=False)[k]` | statsmodels 0.15.0 | exact: 1/(n-k), full-sample mean | 1.1e-16 |
| autocorrelation, lag 1..3 | `acf(x, nlags=3, fft=False)[k]` | statsmodels 0.15.0 | exact: `acovf_k / acovf_0` | 1.4e-17 |
| autocorrelation as Pearson of the pairs | (longhand) against `acf` | - | **not** `acf`: pairs' own means and variances | 1.1e-3 to 2.6e-3 |
| autocorrelation, online | `river.stats.AutoCorr(lag=k)` | river 0.27.0 | exact for Pearson of the pairs; not `acf` | 1.3e-16 |
| cross-correlation, lag 1..3 | `statsmodels.tsa.stattools.ccf(x, y, adjusted=False, fft=False)[k]` | statsmodels 0.15.0 | exact, with **x leading**: `x_{t+k}` with `y_t`; 1/n, full means; `adjusted` defaults to True | 5.6e-17 |
| lagged cross `E[dx_t dy_{t-k}]` | `numpy.cov(x[k:], y[:-k], bias=True)[0, 1]` | numpy 2.5.5 | exact for the pairs' own means; not `ccovf` | 9.7e-17 |
| EW variance, online | `river.stats.EWVar(fading_factor=0.1)` | river 0.27.0 | **not** exact: unadjusted, seeded at the first row | 7.6e-3 |
| EW central co-moments to (6,0), (2,2), (3,1), online | the update of section 3, longhand | - | exact to rounding, a zero-weight row inside | 6.6e-16 |
| the same with a zero-weight **first** row | the update of section 3, longhand | - | exact, no NaN | 5.1e-16 |

What the scan says for the tests:

- **No library computes a weighted moment above order 2.** scipy's
  `moment`, `kstat`, `skew` and `kurtosis` take no weights; statsmodels'
  `DescrStatsW` stops at the covariance; river's `Skew`, `Kurtosis` and
  `AutoCorr` take none. The weighted and decayed oracle is the definition
  of section 1 written by hand, with numpy's `average` doing the sum, and
  the unweighted special case (`w = 1`, `lam = 1`) against scipy, river and
  statsmodels, which do compute it.
- **Cumulants**: `mc2cum` fed central moments, not `mnc2cum` fed raw ones
  (the latter loses 5e-13 here, at a level of 3 over a spread of 2.2; it
  grows with the level).
- **Three autocorrelations, not one.** `acf` (full-sample mean, `1/n`,
  over `acovf_0`), the Pearson correlation of the pairs `(x_t, x_{t-k})`
  (river), and the existing `ew_cov` `lag_corr` recursion (both legs
  against the pre-row mean of the single stream) are different statistics.
  They differ by 1e-3 on 40 rows here and by 4.5e-2 at lag 2 on an AR(1)
  stream (the `ew_lag_cov` module's own measurement). Which one the block
  computes is Q2.
- **statsmodels' `ccf` has x leading** and `adjusted=True` by default; a
  test that reads `ccf(x, y)[k]` as `E[x_t y_{t-k}]` reads the wrong
  direction and the wrong normalisation.
- **Windows**: Polars' `rolling_skew` and `rolling_kurtosis` with
  `bias=True` are the oracle for the row window, unweighted.

## 3. The online update

**Decay is a scaling and a row is a merge.** Decaying every past weight by
`lam` multiplies `W` and every unnormalised co-moment `M_{i,j} = W mu_{i,j}`
by `lam`; a row is then the merge of the decayed summary with the one-row
summary `(w, x, y)`. Pébay, Terriberry, Kolla and Bennett (2016) give the
merge for arbitrary weights and orders; with one side a single row it is the
binomial shift of every past deviation to the new mean. With
`W' = lam W + w`, `a = lam W / W'`, `b = w / W'` (`a + b = 1`),
`dx = x - m_x`, `dy = y - m_y` (both against the pre-row means):

```text
M'_{p,q} = sum_{i<=p, j<=q} C(p,i) C(q,j) lam M_{p-i,q-j} (-b dx)^i (-b dy)^j
           + w (a dx)^p (a dy)^q
           (terms with M_{1,0} or M_{0,1} are 0 and skipped; M_{0,0} = W)
m'_x = m_x + b dx,   m'_y = m_y + b dy
```

In the normalised form the state keeps (`c_{p,q} = M_{p,q} / W`, as
`EwDiag` and `EwCov` keep theirs, which keeps a long gap from underflowing
`W`'s scale into every moment):

```text
c'_{p,q} = sum C(p,i) C(q,j) a c_{p-i,q-j} (-b dx)^i (-b dy)^j  +  b (a dx)^p (a dy)^q
```

Univariate (`q = 0`): `c'_p = a c_p + sum_{i=1}^{p-2} C(p,i) a c_{p-i} (-b d)^i + d^p (a (-b)^p + b a^p)`.
At `p = 2` this is `c'_2 = a c_2 + a b d^2`, `EwDiag`'s update. Part 2
must decide whether order 2 is computed by `EwDiag`'s exact expressions,
so a block's variance is `ew_cov`'s to the bit (proposed: yes).

Every output kind is read from this one state: central directly; raw by
`mu'_p = sum_j C(p,j) m^(p-j) mu_j` (to be measured against
`numpy.average(x**p)` in part 2, and given its own power sums only if the
gap is not tiny); cumulants by the recursion of section 1; standardized by
`mu_p / mu_2^(p/2)`.

**A zero-weight row** (`w = 0`): `b = 0`, `a = 1` when `lam W > 0`, so the
moments only decay with the clock and learn nothing. When `W' = 0` (a
zero-weight first row, or every row so far zero-weight), `a` and `b` are
0/0: the update is skipped by that condition, not computed, and the state
stays empty. The scan feeds both and finds no NaN.

**Withholding early output, by the condition itself, never by a row count.**

- `W = 0`: no moment; reason "no weight yet".
- standardized moments, skew, kurtosis and correlations need `mu_2 > 0` in
  each leg: a leg held at one value withholds them with "the spread is
  zero". The test is the existing one `ew_cov` uses for "no digit of
  spread left" (`EwCov`'s rounding bound), not `== 0`.
- finite-sample uncertainty: the delta-method variance of a sample central
  moment is `(mu_{2p} - mu_p^2 - 2p mu_{p-1} mu_{p+1} + p^2 mu_2 mu_{p-1}^2) / n`,
  at Kish's effective size `n = W^2 / sum omega_t^2`. Withholding on it
  needs the moments to order `2P` and `sum omega^2` in the state (Q5).
- an order above what the data can identify: a two-point stream has every
  standardized odd moment fixed by the two values; nothing withholds that,
  and the uncertainty gate above is what covers it.

## 4. Lags

A lagged cross-moment `E[x_t^a y_{t-k}^b]` is the cross-moment of the pair
stream `z_t = (x_t, y_{t-k})`, which starts at the `k`-th learned row. Its
own summary is a section 3 pair state. What it needs besides is the past
values of `y`, so a **ring of the last `K` learned values of each column
that appears as a lagged leg**, `K` the deepest lag asked of that column,
bounded by the lag count and never by the stream (and held to
`ew_lag_cov`'s `MAX_LAG`). A zero-weight row is not pushed (it taught
nothing, so it is no row to be lagged against), as `EwLagCov` does.

Each lag's pair stream has its **own means** under the "pairs' own means"
convention (numpy's `cov(x[k:], y[:-k])`, river's `AutoCorr`), so the
autocovariance at lag `k` and the variance are read from different
summaries; under `EwLagCov`'s convention both legs are centred at the
single stream's pre-row mean and no extra mean is kept. These are the two
candidates of Q2.

## 5. State size

`P` the highest order, one double each, `S_pair` a pair summary.

| Form | State | Bytes |
|---|---|---|
| one column, central to order `P` | `W`, `m`, `m_lo` (compensated mean, as `EwDiag`), `c_2..c_P` | `8 (P + 2)`; `P = 4`: 48 |
| + Kish's `sum omega^2` (Q5, Q6) | one more double | `+ 8` |
| + uncertainty gate (Q5) | orders to `2P` | `8 (2P + 2)`; `P = 4`: 80 |
| one pair, orders `(A, B)` | `W`, `m_x`, `m_y`, two `m_lo`, `c_{i,j}` for `i <= A`, `j <= B` but `(0,0)`, `(1,0)`, `(0,1)` | `8 ((A+1)(B+1) + 2)`; `(2,2)`: 88 |
| several cross terms on one pair | the union's downward closure, as above | at most `(A_max, B_max)`'s |
| each lag `k` of a pair | its own pair summary | `S_pair` each |
| lag rings | `K_y` values per lagged column `y` (+ `K_y` stamps on a clock lag, Q3) | `8 K_y` (or `16 K_y`) per column |
| row window of `w` rows | the window's rows of the columns read, the merged block sums rebuilt on load (as `window_moments`) | `8 w q` stored; working `S (2w/b + 4)` |
| clock window under decay | a ring of snapshots, at most `window / spacing + 2` | `(S + 16)` each |

`S` is the summary of everything the spec asks for: the sum of the single
column, pair and lag summaries. Example: two columns, orders to 4, lags
1..5 at `(1,1)`, a 1000-row window: `2 * 48 + 5 * 48 + 2 * 8 * 5 + 8 * 1000 * 2`
≈ 16.4 KB, against the default `max_state_bytes` of 256 KB, which the
spec checks before the first row from these formulas.

## 6. Window forms

**Row window: merge, never subtract.** `window_moments.rs` already slides
two row windows of centred co-moments by merging block summaries (Chan,
Golub and LeVeque's rule, van Herk / Gil-Werman aggregation), with no
inverse; Pébay's merge is that rule at every order, so the same scheme
carries over unchanged in structure. The alternative, subtracting the
summary of the rows that left (the inverse merge), measured on
`1e6 + N(0,1)`, 10,000 rows, a 500-row window, `mu_4`:

| Window `mu_4` by | Relative error |
|---|---|
| inverse merge (total less prefix) | 3.2e-9 |
| merge of the window's own rows | 1.3e-11 |

240 times worse at order 4, and the gap grows with the order and with the
level over the spread, because each order's inverse needs every lower
order's already-subtracted value.

**Clock window under decay.** The existing EW window (`window.rs`)
subtracts a decayed snapshot from the live accumulator, which at order 2 is
guarded and measured. At orders above 2 the subtraction is the inverse
merge above. Q4 asks which of three to build.

## 7. The numerical-stability argument

Every term of the update is a product of quantities at the scale of the
spread: `dx = x - m` (one subtraction, rounding `eps * |x|`), `a`, `b` in
[0, 1], and the past's central moments. Nothing carries the level, so the
relative error of `mu_p` is of order `p * eps * level / spread` per row and
does not accumulate beyond the decayed sum of those. The raw-power-sum form
(`mu_4 = s_4 - 4 m s_3 + 6 m^2 s_2 - 3 m^4`) carries `level^4` in each term
and cancels to the spread:

| `mu_4` of `1e6 + N(0,1)`, 10,000 rows | Relative error |
|---|---|
| raw power sums | 1.0 (no digit left) |
| the update of section 3 | 7.5e-11 |

The mean is held as `m + m_lo` (compensated, as `EwDiag`), which keeps
`dx`'s error at the spread's scale when the level is large. A high order
can overflow: `|dx|^p` for `p = 8` at a spread of `1e40` is past `f64`'s
range; the order cap (Q1) bounds this, and an overflowed moment is withheld
by its non-finiteness, never emitted.

## 8. Open questions (for the user)

- **Q1. The orders and the API's spelling.** `orders=[...]` for one column
  is clear. For cross terms: a list of `(a, b)` pairs per column pair, or
  every `(a, b)` with `a + b <= P`? What is the highest order allowed
  (proposed: 8, so `|dx|^p` and the `C(p, i)` stay well inside `f64`)? Is
  it a new spec (`ewm_moments`) or new `stats` of `ew_cov`, which already
  has the means, the second moments, `bias`, `lags` and the windows?
- **Q2. Which lagged moment.** The pair stream's own means (numpy, river;
  section 4) or `ew_lag_cov`'s single-stream pre-row mean (which makes
  lag 0 equal `ew_cov` to the bit)? And the autocorrelation's denominator:
  the pair legs' own variances (river), or the stream's variance (`acf`)?
  These differ by 1e-3 here and 4.5e-2 on AR(1). Reconciling with
  `ew_lag_cov` means choosing one for both.
- **Q3. A lag on the clock, and gaps.** A row lag is "the `k`-th learned row
  back". A clock lag `k` could be: the latest row at or before `t - k`
  (as-of), exactly `t - k` (else no pair), or as-of within a tolerance.
  After a gap the as-of row may be far older than `k`; today
  `EwLagCov::clear` empties the ring on a session change and a capped
  clock gap. Proposed: row lags only in the first version, with the ring
  cleared on those two events as now. And is a lag's ring per column
  (shared by every lag and every pair that reads that column, proposed)
  or per pair?
- **Q4. The clock window at orders above 2.** (a) Snapshot subtraction as
  order 2 does, accepting the error measured in section 6 with a guard
  that withholds when the subtraction cancels; (b) a ring of block
  summaries merged (no subtraction), spaced on the clock, which costs
  `window / spacing` summaries of working memory; (c) no clock window above
  order 2 in the first version. Proposed: (b).
- **Q5. The finite-sample gate.** Withholding a moment whose delta-method
  standard error is large needs orders to `2P` and `sum omega^2` (state
  `8 (2P + 3)` bytes per column instead of `8 (P + 2)`). Keep them, and
  what threshold, stated how (relative standard error above a parameter)?
- **Q6. Unbiased forms.** Order 2 has `ew_cov`'s `bias=False` (Kish's
  correction, Polars' `ewm_var(bias=False)`). Above order 2 the unbiased
  weighted forms are Rimoldini's (2014, "Weighted skewness and kurtosis
  unbiased by sample size and Gaussian uncertainties", needing
  `sum omega^2`, `sum omega^3`, `sum omega^4`). Offer `bias=False` above
  order 2, or plug-in only, as the scan's oracles (scipy `bias=True`)
  compute?
- **Q7. The weight of a lagged pair.** `w_t` alone (the row that closes the
  pair), or `w_t * w_{t-k}`? A zero-weight `t - k` is not in the ring under
  either.

## 9. The next part

Part 2: the update of section 3 in numpy (Rust-free), the reference the
Rust is checked against: single column, pair and lag summaries, the row
window by block merge, the zero-weight and constant-leg cases, the oracle
tests of section 2 at `w = 1, lam = 1` and the longhand weighted
definitions otherwise, and the raw-from-central gap measured. Part 3: the
state in `rheon-core` behind `OnlineModel`, its size checked against
`max_state_bytes`, fixtures per the state rules, then the spec.
