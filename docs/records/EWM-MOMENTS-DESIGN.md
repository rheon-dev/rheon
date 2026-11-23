# EWM moments: oracle scan and design (PLAN 554)

A record, begun 2026-11-22 before any implementation (part 1) and
completed with the Rust state (parts 3 to 5, section 11): which third-party
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
range; with no order cap (Q1), an overflowed moment is withheld by its
non-finiteness, with a reason, never emitted.

## 8. Open questions (for the user; answered in section 9)

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

## 9. The answers (the user, 2026-11-22)

- **Q1.** A new spec kind, `ewm_moments`, and the window operator
  `rp.ewm_moments`, beside `ewm_mean` and `ewm_var`; not new statistics of
  `ew_cov`. Terms are named by the user, as `power_variation`'s:
  `terms={"m3": {"x": 3}, "xy2": {"x": 1, "y": 2}, "ac1": {"x": 1, "x@1": 1}}`,
  a lag written on the column, each output named after its term. No order
  cap: the state is checked against `max_state_bytes`; tests run to order
  6, the oracles' range.
- **Q2.** Exact central moments about the lagged pairs' own weighted means
  (Pebay over the pairs); the autocorrelation is that over both legs'
  variances on the same pairs. It differs from `ew_lag_cov`'s recursion and
  from `acf`, and the tests state by how much (section 10).
- **Q3.** One lag buffer per column, shared by every pair that reads it. A
  clock lag pairs the row with the latest row at or before `t - lag`
  (as-of); a break clears the buffer; a gap longer than the lag leaves no
  pair (the row adds nothing to the lagged terms and still counts for the
  others).
- **Q4.** The merged-block window, on the clock as on rows, its edges from
  the raw clocks as Polars' `rolling_*_by`.
- **Q5.** Guard by the condition: `sum omega^2` is kept always (Kish's
  `n`). A term is withheld with a reason where it is undefined: no weight;
  zero variance for a standardized or correlation term; a bias-correction
  denominator at or below 0. Orders to `2P` are kept only when the spec asks
  for `se=True`, which emits `se_<term>`. No row counts anywhere.
- **Q6.** One `bias` parameter, its default each statistic's in Polars
  (`ewm_var`: unbiased; `skew`, `kurtosis`: plug-in). Unbiased orders 3 and
  4 by Rimoldini's forms for reliability weights; `bias=False` above order 4
  is refused by name.
- **Q7.** `ew_lag_cov`'s pair weight, `w_t`, the weight of the row that
  closes the pair, so the two weigh a pair alike at order (1,1). A
  zero-weight row on either side gives no pair: a zero-weight row is never
  learned, so it is neither a pair's row nor in a buffer.

## 10. The reference (part 2)

`tests/reference_ewm_moments.py` is the update in Python floats, Rust-free,
and `tests/test_reference_ewm_moments.py` holds it to the oracles (28 tests,
3 s, the essentials tier). It is exact against itself (chunks, a saved
state) and held to the Rust within a tolerance, not to the bit: it keeps the
mean whole where `EwDiag` compensates it, and folds a window's blocks left
to right where `window_moments` aggregates them by van Herk's scheme.

**The summary.** Over a term's `d` legs: `W`, the `d` means, the normalised
central co-moments `c_a = M_a / W` for every `a` in the downward closure of
the terms' indices (order 2 and up), and `r_k = sum omega^k / W^k` for
`k = 2` always (`1 / r_2` is Kish's `n`) and `k = 3, 4` where an unbiased
third or fourth moment is asked for. Decay multiplies `W` alone: `c` and
`r` are ratios every factor scales alike, so a long gap moves no moment and
underflows nothing. A row is Pebay's merge with the one-row summary
`(w, z, c = 0, r = 1)`, and a block is the same merge with a block's
summary, so one function does both:

```text
W = W_A + W_B    a = W_A / W    b = W_B / W    delta = m_B - m_A    m = m_A + b delta
c_a = sum_{beta <= a} C(a, beta) [ a c^A_{a-beta} (-b delta)^beta + b c^B_{a-beta} (a delta)^beta ]
r_k = a^k r^A_k + b^k r^B_k
```

Powers are formed by repeated multiplication, not `pow`, so the Rust can
form the same numbers. An empty side (`W = 0`) returns the other unchanged:
the guard is the condition, so a zero-weight first row is no 0/0.

**The unbiased forms.** For i.i.d. rows under weights with ratios `r_k`,
derived exactly (rational enumeration over two discrete distributions, the
coefficients fitted in `1, r2, r3, r4, r2^2` and checked on further weight
vectors):

```text
E[m2]   = (1 - r2) mu2
E[m3]   = (1 - 3 r2 + 2 r3) mu3
E[m4]   = (1 - 4 r2 + 6 r3 - 3 r4) mu4 + (6 r2 - 6 r3 + 9 r4 - 9 r2^2) mu2^2
E[m2^2] = (r2 - 2 r3 + r4) mu4 + (1 - 3 r2 + 2 r3 - 3 r4 + 3 r2^2) mu2^2
```

Inverting gives Rimoldini's unbiased `M2`, `M3`, `M4` and an unbiased
`mu2^2`; Fisher's `k4` is `M4 - 3 (mu2^2)_unbiased`. At equal weights these
are `kstat`, `skew(bias=False)` (`M3 / M2^1.5`) and
`kurtosis(bias=False, fisher=False)` (`k4 / M2^2 + 3`). Cross terms of order
3 and 4 use the same coefficients, polarised (`mu2^2` becomes
`(c_ij c_kl + c_ik c_jl + c_il c_jk) / 3` over the four leg indices). Each
denominator (`1 - r2`, `1 - 3 r2 + 2 r3`, the 2x2 determinant) withholds its
term when it keeps no digit (within `64 eps` of its terms' sizes): one row
carrying all the weight, or the weight on too few rows, by the weights, not
a count.

**The standard error.** Each kind's influence function is a polynomial in
the legs' deviations (`d^a - mu_a - sum_j a_j mu_{a-e_j} d_j` for a central
moment, the chain rule for standardized and cumulant terms, the expansion
of `(m + d)^a` for raw ones); its mean square, read from moments to twice
the order, times `r_2`, is the delta-method variance at Kish's `n`.

**Choices the reference makes that the answers left open.**

- A zero-weight row, or one with a non-finite value or weight in a column
  the spec reads, is no row: no clock advance, no decay, no buffer, no
  window. Every output is a ratio that a uniform decay leaves alone, so on
  a clock this changes nothing; on a row clock it means a zero-weight row
  is not a row of decay.
- "A gap longer than the lag", for a clock lag `L`: the as-of row is more
  than `L` before the as-of point `t - L` (older than `t - 2L`). An as-of
  match within `L` of the point pairs; jittered rows at a spacing near `L`
  still pair, which a rule on the spacing between consecutive rows would not
  allow.
- `kind` is one kind for every term or a map from term name to kind
  (`central`, `raw`, `standardized`, `cumulant`). A cumulant is of one leg;
  a joint cumulant is refused by name.
- A break clears the lag buffers only; the accumulators and the window ring
  go on, as `ew_lag_cov`'s `clear`. The window operator's surface resets at
  a break as `ewm_var` does; that is the surface's, not the state's.
- Output is read after the row's update (`emit="after"`, a window
  operator's convention, closed on the right) or before it (`"before"`, a
  model's, out of sample).

**Results** (seed 554, 60 rows of `3 + Gamma(2, 1.5)`, unequal weights with
one zero; "gap" is `|ref - oracle| / max(|oracle|, 1)`, worst over every row
and order):

| Held to | Gap |
|---|---|
| `scipy.stats.moment`, central, orders 2 to 6, equal weights | 2.1e-14 |
| `scipy.stats.moment(center=0)`, raw read from central and the mean | 1.1e-15 |
| `numpy.average`, central, orders 2 to 6, decayed on a clock, weighted | 8.0e-14 |
| `numpy.average`, raw, the same | 1.3e-15 |
| `scipy.stats.skew` / `kurtosis`, `bias=True` | 3.0e-15 / 1.7e-15 |
| `scipy.stats.skew` / `kurtosis`, `bias=False` | 3.2e-15 / 6.4e-15 |
| `scipy.stats.kstat`, orders 2 to 4 | 8.3e-13 (five rows, `k4`'s cancellation) |
| `moment_helpers.mc2cum`, plug-in cumulants 2 to 6 | 2.6e-13 |
| `pl.Series.ewm_var(adjust=True, bias=False)`, decay only | under 1e-12 |
| exact expectation, unequal weights, `M2`, `M3`, `M4`, `k4` | under 1e-12 relative |
| `numpy.cov(x[k:], x[:-k], bias=True)`, lags 1 to 3 | 8.9e-16 |
| `acovf` through `(n-k)/n (cov_k + (m_a - m)(m_b - m))` | 8.9e-16 |
| `acf` through that over `acovf_0` | under 1e-13 |
| the pairs' Pearson correlation against `acf` (the convention gap) | 2.3e-3, 2.8e-3, 1.1e-3 at lags 1, 2, 3 |
| cross and lagged cross `(1,1)`, `(1,2)`, `(2,2)`, `(3,1)` against the longhand | under 1e-12 |
| `pl.Series.rolling_skew` / `rolling_kurtosis(bias=True)`, 10 rows | 2.2e-14 / 4.1e-14 |
| a clock window, decayed and weighted, against the longhand | under 1e-12 |
| delta-method se of `mu_3`, and of `mu_2` at Kish's `n` | under 1e-12 |
| se of the skew, 3,000 normal rows, against `sqrt(6/n)` | within 10% |
| `mu_4` of `1e6 + N(0, 1)`, a 500-row window of 3,000 rows | 2.3e-11 |

And to the bit: 1 chunk, random chunks of 1 to 6 rows and chunks of one row
give the same outputs and reasons (EW on a clock and on rows, a row window,
a clock window, output before the row; a break inside); a state saved
halfway (as JSON) continues to the same outputs and the same final state;
a zero-weight first row leaves every later output what it is without it.

**The state, in doubles** (8 bytes each):

```text
summary over d legs     1 + d + |closure| + |{r_k}|
  one column, order P   P + 2            (P = 4: 48 bytes; se=True: 2P + 2)
  a pair, orders (A, B) (A+1)(B+1) + 1   ((2,2): 80 bytes)
  + r3, r4              + 1 each, only for an unbiased order 3 or 4
lag buffer, row lags    K per column     (K the column's deepest lag)
lag buffer, clock lags  2 per row inside the last 2 K of clock time
row window              n (3 + L) ring   + (floor(n / b) + 2) summaries, rebuilt on load
clock window            (3 + L) per row inside the window's span
counters                3
```

`L` is the number of distinct legs. Nothing grows with the stream's
length: the summaries with the spec, a row lag's buffer with the lag, a row
window's ring with the window. A clock lag's buffer and a clock window's
ring hold the rows inside a span of the clock, so their size is the data's
rate times the span; the reference checks them against `max_state_bytes`
as they fill and refuses the run by name when they pass it.

## 11. The Rust state (parts 3 to 5)

`crates/rheon-core/src/ewm_moments.rs` (with `layout.rs` beside it) keeps
the state behind `OnlineModel`, as the module docs state it. It is held to
the reference of section 10 within 5e-14 relative to each number's size
(or to 1 below it), withholding exactly where the reference withholds:

| Held to the reference | Gap (2026-11-23) |
|---|---|
| every kind, plug-in and unbiased; stream A (`lam = 0.9`, a zero-weight first row) / B (half-life 7, uneven clock) | 2.6e-15 / 3.5e-15 |
| row lags and cross terms, plug-in and unbiased, `lam = 0.9`, zero-weight rows | 1.8e-15 |
| clock lags as-of, a gap of 9.5 past both lags; the default tolerance / 0.5 | 4.4e-16 / 4.4e-16 |
| a window of 12 rows / a clock window of 9 (two stacks; the block merge was 2.3e-15 / 4.3e-16) | 4.7e-15 / 4.3e-16 |
| each kind's standard error, central to lagged | 6.4e-15 |
| the definition longhand, orders 2 to 6 and raw, stream B | under 1e-13 |
| order 2 against `EwDiag` (`ew_cov`'s variance) | to the bit |

They part by rounding alone: the Rust keeps each mean as a pair (`m +
m_lo`, `crate::comp`) where the reference keeps it whole, forms order 2 by
`EwDiag`'s expression, `r_3` and `r_4` by products where the reference
calls `pow`, decays a zero-weight row's clock step on its own where the
reference folds it into the next row's (the same weights: a decay over two
steps is the product of the two), and reads a window through two stacks
where the reference folds blocks left to right. To the bit, in Rust: one
chunk against chunks of any size, a state saved at any row (each form:
row lags, clock lags, a row window, a clock window, the standard errors),
`predict` against the step's read, a zero-weight first row against none.

**The state, final.** In doubles, `d` a summary's legs, `L` the distinct
legs, `K_c` column `c`'s deepest row lag, `b` a window's block rows:

```text
the weight           1
a summary            1 + 2d + |kept| + |r_k|    (W, each mean's pair, the moments, r_2 [, r_3, r_4])
the summaries        their sum                  (none under a window)
a lag buffer         1 + K_c                    (rows)
                     1 + 3 per row held         (clock: a stamp is two)
the clock's place    1                          (on a clock)
a window's ring      3 + (2 + L) a row          (+ 2 a row for its stamp on a clock)
its two stacks       (rows / b + b + 2)(Σ summaries + 2), not saved: a cache
```

One column to order `P` is `P + 4` doubles (order 4: 64 bytes): section
10's `P + 2`, the mean's low part (8 bytes, the compensated mean that keeps
the deviation at the spread's scale under a large level) and the model's
own weight. Under `se` the summary keeps the moments to `2P`. The ring's
three counters are its first row's number, the decay rows of weight 0 have
passed since the last learned row, and the stacks' boundary `F` (below).
`b` is `⌊√w⌋` for a row window and 16 on a clock. The settings' part is
checked against `max_state_bytes` before the first row; a clock lag's
buffer and a clock window's ring as each row goes in (a warning past half,
the run refused at the cap).

**The window: two stacks against the block merge.** Part 4 read a window
by merging the summaries of its full blocks of `b` rows and folding the
partial blocks at either end from the rows, `w / b` merges and up to `2b`
row folds a read. Part 5 tried two stacks over the same blocks (the
module docs give the scheme): the back is the running fold of the rows
from a block boundary `F`; the front holds the merge from each block's
start to `F`, formed right to left once needed; the head the same from
each row of the block the window starts in; a read is one merge, the
head's summary then the back. `F` moves to the last block boundary when the
window's first row reaches it and a block has filled since. Each row is
folded once into the back, once into its block and once into the head, so
a read's cost does not grow with the window.

Measured 2026-11-23 (both forms in one release test binary, a test
removed with the block merge; `window_timing` below times the shipped one;
the minimum of three rounds of 20,000 steps after three windows of warm-up;
skew and kurtosis of `x` and its correlation with `y`, `x` at a level of
`1e3` with a spread of about 1, weights in `[1/4, 5/4)`, a half-life equal
to the window, the clock stepping `[1/2, 3/2)`; the gap is to a fresh left
fold of the window's rows at the end):

| Window | Two stacks | Block merge | Ratio | Gap, two stacks | Gap, block merge |
|---|---|---|---|---|---|
| 1,000 rows | 4.25 µs a row | 53.7 µs | 12.7 | 1.2e-15 | 1.9e-15 |
| 1,000 clock units | 4.35 µs | 85.8 µs | 19.7 | 3.3e-16 | 2.1e-15 |
| 10,000 rows | 4.23 µs | 178 µs | 42.1 | 5.7e-15 | 1.9e-15 |
| 10,000 clock units | 4.23 µs | 907 µs | 214 | 4.5e-15 | 4.6e-15 |

The two stacks are kept. Both forms are within the same tolerance of the
reference and of a fresh fold (rounding, at a few times `1e-15`), and the
two stacks are 12.7 to 214 times faster, flat in the window's length. They
round differently (55,539 of 60,000 outputs at the 1,000-row window part
in their last bits), so the window fixtures' outputs moved with them.

Their cost: the stacks' working memory is `(rows / b + b + 2)` parts where
the block merge's was `(rows / b + 2)`, the head's `b` row summaries.
For one column to order 4 and a 10,000-row window that is 1,818 doubles
(14.5 KB) against 918 (7.3 KB), beside the ring's 30,000 (240 KB): 3% more
state. A summary is a function of the rows and of `F`, so `F` is saved: one
more number in the ring (8 bytes), an addition to schema 72's layout, which
is unreleased, so no bump; the fixtures `ewm_moments_window` and
`ewm_moments_clock_window` are regenerated and `F` held non-empty in each.

**The standard errors' order.** Every term in the order the spec names
them, then `se_<term>` for each, in the same order, after every term. No
other kind emits a standard error per term; the nearest patterns agree:
`power_variation` emits each target's terms and then that target's `n`,
and the bank emits `se_coef` after `coef` as a block. Part 4's order, so no
reference, test or fixture moved.

**Measurements kept as tests.** `window_timing` (ignored; `cargo test
--release -p rheon-core window_timing -- --ignored --nocapture`) times the
shipped form at these four windows (on the final code: 4.25, 4.32, 4.22
and 4.70 µs a row, the last while a debug build ran beside it); `the_two_stacks_read_the_windows_rows`
holds every read of windows of 1 to 50 rows and clock windows of 0.75 to
40 units (a gap of 30 every 101 rows, emptying the window or taking several
blocks at once, zero-weight rows) to a fresh fold within 1e-12 (measured
2.9e-15), and a state saved every 13th row to the bit.

## 12. What the entry still lacks

The state is complete in `rheon-core`. Not built yet:

- the spec kind `ewm_moments` in `rheon-polars` (parsing the terms map with
  `@` lags, a kind per term, `bias`, `se`, `lag_tolerance`, `window_rows`
  and `window_size`, `max_state_bytes`), its output fields (`<term>`,
  `se_<term>`, `weight_sum`) and the CLI's TOML;
- the Python builder and the window operator `rp.ewm_moments` beside
  `ewm_mean` and `ewm_var` (Q1), with docstrings stating the update;
- the tests through the bank: the oracles of section 2 (scipy, statsmodels,
  Polars' `rolling_skew` and `rolling_kurtosis`), chunk invariance, and
  zero-weight rows through the stream's shared clock (`model_contract.rs`
  runs the kind already);
- the reader-facing docs (the outputs and models guides, CHANGELOG,
  `llms.txt`), `docs/EXTENDING.md`'s list, bench cells and budgets.
