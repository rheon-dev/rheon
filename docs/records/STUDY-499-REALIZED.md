# Study 499: realised-volatility estimators against hfhd

*2026-11-21. A numpy study, nothing built in the library. Branch
`vm/study-499`; entry point `scripts/study_499.py` (every table below is its
output); code in `studies/study_499/` (`data.py` the seeded ticks,
`estimators.py` the batch definitions from the papers, `streaming.py` the
one-pass forms, `compare.py` the checks). Machine: x86_64 Linux, 4 cores.*

## 1. The answer

Every estimator the task names is reproduced from its paper and checked
against hfhd 0.1.4 on the same ticks. Where hfhd's conventions are written
in, the two agree to floating precision: the largest gap is 3.0e-13 of the
estimate's largest entry (the realised kernel on the one-second grid, whose
sum cancels a noise term about 36 times the estimate), and every other gap
is 8.4e-15 or less.
The refresh-time grid agrees to the bit, and the pre-averaged
Hayashi-Yoshida with hfhd's window convention to 6e-16. Where hfhd departs from its paper the
study sets the paper's form beside it and measures the difference
(Section 4).

| estimator | hfhd function | largest gap to hfhd (hfhd's convention) | paper's form vs hfhd | one pass? | state (bytes; d assets) | at hfhd's defaults, n = 23,400, d = 2 | one pass, ns a tick (d = 2 / 10) |
|---|---|---|---|---|---|---|---|
| refresh time | `refresh_time` | identical stamps and values | same definition | yes (one stamp of look-ahead) | `8 (3 + d) + d` | 42 | 7.7 / - |
| two scales | `tsrc(J, K)` | 5.3e-16 | 2e-7 (hfhd counts prices for `n`) | yes | `8 (K d + 2 d^2 + 1)` | 13,160 (K = 818) | 30 / 97 |
| multi scale | `msrc(M, N, pairwise)` | 4.9e-15 | 2e-6 (the edge term's `n`) | yes | `8 ((N + M) d + 3 d^2 + 1)` | 5,000 (M = N = 153) | 1,210 / 6,900 |
| pre-averaged (MRC) | `mrc(theta, bias_correction, pairwise)` | 8.4e-15 | 1e-3 to 1e-2 with the bias term, 4e-4 to 4e-3 without | yes | `8 ((k - 1) d + d + 2 d^2 + 2)` | 1,072 (k = 62) | 220 / 374 |
| realised kernel, Parzen | `krvm(H, kernel=parzen_kernel)`, `get_bandwidth` | 3.0e-13 | a different weight function: 4% to 7% (Section 4) | yes at a fixed H; with H chosen at close, a ring of `H_max` | `8 (H d + d + d^2 + 1)`; `8 (H_max d + d + (H_max + 1) d^2 + 1)` | 936; 70,616 (H_max = 1,470) | 910 / 5,040; 34,400 / 74,300 |
| realised kernel, quadratic spectral | `krvm(H, kernel=quadratic_spectral_kernel)` | 3.0e-13 | as Parzen | **no**: every lag carries weight; cut at `L` lags it is one pass | `8 (L d + d + d^2 + 1)` | 1,176 (L = 10 H = 70; gap 1.1% to all lags) | 1,290 / 6,470 |
| Hayashi-Yoshida | `hayashi_yoshida(theta=None)` | 2.7e-15 | same definition | yes, on a long frame | `8 * 18 + 2` a pair | 146 | 4.6 |
| Hayashi-Yoshida, pre-averaged | `hayashi_yoshida(theta)` | 6.3e-16 | up to 10% on a day (which pairs of windows meet at the edges; 2% to 3% of the truth on average over 40 days, beside the estimator's own 26% to 32%) | one pass, but its pending queue is bounded only by the ratio of the two assets' tick rates | not bounded by the shape | - | not built (batch numpy 959) |

Two passes are needed only where a choice needs the block's own length
before its first row: `K = n^(2/3)`, `M = N = sqrt(n)`, `k = theta sqrt(n)`
and the kernel's ring. With the block's length given in advance, as
`rcov`'s `block_rows` gives it, every one of them is one pass; the
closing arithmetic (`nbar_K`, the edge term, the finite-sample factor)
reads the block's actual count at close. The automatic kernel bandwidth is
the one that costs: it is chosen at close from the block's own noise and
variance, so every lag up to the largest it could choose must be kept
apart (rcov does exactly this).

Not studied: weighted (decayed) forms. None of the hfhd functions takes
weights, so a weighted form would have only the definition by hand as its
oracle, and a zero-weight row (the library's rule that such a row learns
nothing) never arises in the forms studied here.

## 2. Setup

**Versions.** python 3.13.18, numpy 2.5.5, scipy 1.18.1, pandas 3.1.1,
numba 0.68.1, hfhd 0.1.4 (a venv of its own; the repository's lock has
numpy 2.5.2, numba 0.68.1, pandas 3.0.5 and hfhd >= 0.1.4).

**Ticks** (`data.py`, seeded; no file). One day is 6.5 hours, cut into
93,600 fine steps of a quarter second. Two efficient log-prices diffuse
with a U-shaped deterministic volatility (1.6 times its level at the open
and close, 0.8 at midday; level 2% a day) and correlation 0.5, so the
day's integrated variance (4.78e-4) and covariance are known exactly.
Each observation adds independent Gaussian noise of standard deviation
6e-4 (noise-to-signal `xi^2 = omega^2 / IV` = 7.5e-4: the plain realised
variance at one second is 36 times the truth). From one path:
`sync`, both assets every second (23,400 returns); `async`, asset A at
each fine step with probability 1/8 (about 11,700 ticks) and B with 1/20
(about 4,700), independently, which refresh time reduces to about 4,150
returns; `irregular`, A's ticks alone. About 585 stamps a day are shared
by A and B, which tests the tie rules of refresh time and
Hayashi-Yoshida. A stamp never repeats within one asset.

**The gap** is `max |mine - hfhd| / max |hfhd|` over the matrix. Every
input is a log-price (hfhd's input); every difference of log-prices is
taken once, as hfhd takes it.

**hfhd under numpy 2.** hfhd 0.1.4's `_get_indeces_and_values` fills a
`uint64` stamp matrix with `np.nan`, which numpy 2 refuses ("cannot
convert float NaN to integer"). So `hayashi_yoshida` and every
`pairwise=True` path (`msrc`, `mrc`, `krvm`) raise under numpy 2 -- under
the repository's lock as here. `compare.py` replaces that one helper with
one that pads with 0; hfhd cuts each row at the count of non-NaN *values*,
so the padding is never read and nothing else changes. An oracle test of
any pairwise path or of Hayashi-Yoshida needs the same shim (or an hfhd
release that fixes it).

## 3. The definitions and hfhd's settings

Prices `Y_0 .. Y_n` (`n` returns `r_i = Y_i - Y_{i-1}`), `d` assets,
`x y'` an outer product.

**Refresh time** (Barndorff-Nielsen, Hansen, Lunde and Shephard 2011,
Definition 1; the previous-tick sampling of Ait-Sahalia, Mykland and
Zhang): `tau_1 = max_i t_i[0]`, `tau_{j+1} = max_i min{t in t_i : t >
tau_j}`, each asset's value at `tau_j` its last price stamped at or
before it. hfhd: `refresh_time([sa, sb])`.

**Two scales** (Zhang, Mykland and Ait-Sahalia 2005, Eq. 55, with a fast
scale `J` after Ait-Sahalia, Mykland and Zhang 2011; covariances by
Zhang 2011):

    [Y,Y]^(K) = (1/K) sum_{i=K}^{n} (Y_i - Y_{i-K})(Y_i - Y_{i-K})'
    TSRV      = [Y,Y]^(K) - (nbar_K / nbar_J) [Y,Y]^(J),   nbar_K = (n - K + 1) / K

hfhd: `tsrc(series, J=1, K=None)`, `K = int(N_p^(2/3))` with `N_p` the
count of refresh-time *prices*, and checked also at `J = 2` and `(J, K) =
(5, K/2)`. hfhd takes `N_p = n + 1` for the paper's `n` in `nbar`; ZMA's
small-sample factor `(1 - nbar_K / nbar_J)^-1` is not applied.

**Multi scale** (Zhang 2006, with the shifted scales and end term of Tao,
Wang and Chen 2013, which is what hfhd writes):

    K_m = N + m,  a_m = K_m (K_m - Kbar) / (M var K) = 12 (m + N)(m - M/2 - 1/2) / (M (M^2 - 1))
    MSRV = sum_{m=1}^{M} a_m [Y,Y]^(K_m) + zeta ([Y,Y]^(K_1) - [Y,Y]^(K_M)),
    zeta = (M + N)(N + 1) / ((n + 1)(M - 1))

`sum a_m = 1` and `sum a_m / K_m = 0`, which cancels every scale's noise
term `2 n omega^2 / K`. The two closed forms of `a_m` agree to 1e-16.
hfhd: `msrc(series, M=None, N=None, pairwise=False)` (both
`ceil(sqrt(N_p))`), and `(M, N) = (10, 0)` and `M = 1` (the plain realised
covariance; `N` ignored), and `pairwise=True` on the asynchronous pair
(each variance on the asset's own ticks, the covariance on the pair's
refresh time). hfhd's `n` in `zeta` is the count of prices; read as the
count of returns the estimate moves by 2e-7 to 2e-6. I did not have Tao,
Wang and Chen's paper to hand, so which reading is theirs is open; the
difference is immaterial.

**Pre-averaging** (Christensen, Kinnebrock and Podolskij 2010, Eq. 8, 9,
12, footnote 1), `g(x) = min(x, 1 - x)`:

    Ybar_i = sum_{j=1}^{k-1} g(j/k) r_{i+j},  i = 0 .. n - k + 1   (n - k + 2 windows)
    MRC    = n/(n - k + 2) * 1/(psi_2 k) * sum_i Ybar_i Ybar_i'
    bias-corrected: MRC - psi_1/(theta^2 psi_2) * (1/(2n)) sum_j r_j r_j',  then / (1 - psi_1/(2 psi_2 k^2))
    psi_1 = k sum_{i=1}^{k} (g(i/k) - g((i-1)/k))^2,  psi_2 = (1/k) sum_{i=1}^{k-1} g(i/k)^2

Window: `k = ceil(theta sqrt(n))` with the bias term, `ceil(theta
n^0.6)` without. hfhd: `mrc(series, theta=0.4, bias_correction=True,
pairwise=False)` (its default), `theta=0.6`, and `theta=1.0,
bias_correction=False` (`rcov`'s default `psd = true` form), plus
`pairwise=True` on the asynchronous pair. hfhd parts from the paper in
four places, each a flag in `estimators.mrc` and measured in Section 4:
its first window starts at the second return (`n - k + 1` windows,
scaled as if there were `n - k + 2`); its `psi_1` sums to `k - 1`, leaving
out the last term `1/k^2`; it does not rescale by footnote 1; and its bias
term reads the `theta` it was given, not the window's own `k / sqrt(n)`.

**Realised kernels.** BNHLS 2011's form (their Eq. 2) on the jittered
prices (the first and last price each replaced by the mean of `m` at that
end):

    K = sum_{h=-n}^{n} k(h / (H + 1)) Gamma_h,   Gamma_h = sum_{j=h+1}^{n} x_j x_{j-h}',  Gamma_{-h} = Gamma_h'

hfhd's `krvm` is another form: no jitter (its docstring says so), and
`Gamma_0 + sum_{h>=1} k((h - 1)/H)(Gamma_h + Gamma_h')`, the flat-top
weights of Barndorff-Nielsen, Hansen, Lunde and Shephard (2008), which
give lag 1 the full weight and are not positive semi-definite by
construction. Both are checked: hfhd's own `parzen_kernel` and
`quadratic_spectral_kernel` against the flat-top form, and BNHLS's form
by passing hfhd a numba kernel `u -> k((u + 1)/(H + 1))` with `H_hfhd = 1`
(the device `tests/test_rcov.py` uses), at `m = 1` and on prices jittered
at `m = 2`. Kernels: Parzen, and BNHLS's quadratic spectral `k(x) = 3/x^2
(sin x / x - cos x)` (their Table 1; not Andrews' rescaled one).
Bandwidth: `get_bandwidth(n, var_ret, var_noise, kernel)` =
`int(c* xi^{4/5} n^{3/5})`, `c*` = 3.51 or 0.46, with `var_noise =
RV / (2n)` on every return and `var_ret` the realised variance on a
20-return grid averaged over its 20 offsets (BNHLS Section 4.1, and
`rcov`'s `noise_stride = 1`, `iv_stride = 20`), asset A's.
`pairwise=False` throughout (the pairwise kernel is not checked).

**Hayashi-Yoshida** (Hayashi and Yoshida 2005):

    HY = sum_i sum_j dX_i dY_j 1{(ta_{i-1}, ta_i] and (tb_{j-1}, tb_j] meet}

checked three ways: the double sum over every pair on 400 and 200 ticks
(the definition itself, `hy_bruteforce`), the `O(n log n)` form that
telescopes B's returns over each A interval to `Y(first tb >= e) -
Y(last tb <= s)`, and the one-pass form. hfhd: `hayashi_yoshida([sa, sb])`
(theta and k `None`: `k = 1`). The diagonal is each asset's realised
variance on its own ticks.

**Pre-averaged Hayashi-Yoshida** (CKP 2010 Section 4; Christensen,
Podolskij and Vetter 2013):

    HY^theta = 1/(psi_HY k)^2 sum_i sum_j Abar_i Bbar_j 1{I_i and J_j meet},  Abar_i = sum_{l=1}^{k-1} g(l/k) dX_{i+l}

the paper's windows `i = 0 .. n - k` with `I_i = (t_i, t_{i+k}]`. hfhd
(`hayashi_yoshida(theta=0.4)` and `0.6`, `k = ceil(theta sqrt(nbar))`,
`nbar` the mean count of prices of the two assets) differs in its
indexing: the asset with fewer ticks is "A"; A's windows run from `k`
(its first `k` windows are dropped); each window's interval is one return
earlier, `(t_{i-1}, t_{i+k-1}]`; B's first `k` windows have no lower end;
and `psi_HY = 1/4` (the limit; exact for even `k`, `(k^2 - 1)/(4k^2)` for
odd). `estimators.hy_preavg(convention="hfhd")` writes these choices out
and gives hfhd's number to the bit.

## 4. Agreement with hfhd, and where the paper parts from it

{{AGREEMENT}}

Reading it:

* **Floating precision** holds everywhere hfhd's convention is written
  in. The kernels' 3e-13 is the cancellation: `Gamma_0` is about 36 times
  the estimate on the one-second grid, so a summation order that differs
  in the last bit of `Gamma_0` shows at 1e-13 of the estimate; on the
  refresh-time grid (`Gamma_0` 7 times the estimate) it is 2e-14, and on
  one asset's own ticks, where both sides run the same one-dimensional
  arithmetic, 4e-16 or exactly 0.
* **The one-pass forms** agree with hfhd to 8.3e-14 or better and give the
  same bits at 1 chunk and 997 chunks for every estimator: the state
  carries everything (the ring, the accumulators, the counts, and for the
  tick-level ones the open stamp).
* **tsrc and msrc**: the paper's `n` against hfhd's moves the estimate by
  1e-7 to 2e-6; ZMA's small-sample adjustment by 0.1% to 0.4%; msrc
  without the end term by 0.5% to 0.9%.

{{MRCFLAGS}}

Each of hfhd's four departures moves the estimate by 3e-4 to 9e-3 alone;
the largest is the missing rescale on the refresh-time grid, where the
window is short (`k` = 26) and the bias term's share of the integrated
covariance, `psi_1 / (2 psi_2 k^2)`, is larger. They partly cancel. `rcov(kind="preavg", psd=False)` is the paper's form (`tests/
test_rcov.py::test_the_preaveraged_estimate_is_its_definition`); its
`psd = true` default, which has no bias term, matches hfhd's
`bias_correction=False` to 3e-15 once hfhd's first window is mapped (the
existing `test_the_default_preaveraged_estimate_is_hfhds_mrc`).

{{HYGAPS}}

The paper's windows and hfhd's give different numbers on a day (up to 10%
over five days) because the boundary pairs differ: each
window's interval carries one zero-weight return, at the front in hfhd's
form and at the back in the paper's, so the two sets of pairs differ by
pairs whose weighted returns do not overlap in time. Those products have
mean zero; over 40 days the mean difference is 2% to 3% of the truth, a
fifth of a standard error of it, beside the estimator's own spread of 26%
to 32%, and the `psi_HY` choice (limit or exact) moves the mean by about
0.1%.

{{BANDWIDTH}}

**`c*`.** BNHLS's `c* = (k''(0)^2 / k_.^{0,0})^{1/5}`, `k_.^{0,0} =
int_0^inf k(x)^2 dx`, is 3.5117 for Parzen (`k_.^{0,0}` = 151/560 =
0.26964) and 0.4628 for the quadratic spectral (`k''(0) = -1/5`,
`k_.^{0,0} = 3 pi / 5`). hfhd rounds them to 3.51 and 0.46 and
truncates `H` to an integer; `rcov` uses 0.269 for `k_.^{0,0}`, so
3.5134, and rounds `H` up. Changing `H` by those amounts moves an estimate
by 0.2% to 2% (the table's last column): the bandwidth is a statistical
choice, and an oracle test should pass `H` explicitly. The noise ratio
`xi^2` the rule reads is itself estimated: the 20-return grid's realised
variance still carries `2 (n/20) omega^2` of noise here, so `xi^2` comes
out at 2.8e-4 against the true 7.5e-4 and `H` about a third smaller than
the truth would give.

**Flat-top against BNHLS.** At the same `H` they differ by 4% to 7% on
a day (the agreement table's "paper's form" column for `krvm`). They are
different estimators, not a convention: `rcov(kind="kernel")` is BNHLS's,
with the jitter, and its oracle is hfhd's `krvm` with the weights passed
in, as the existing test does. hfhd's default kernel is the quadratic
spectral, not Parzen.

{{QS}}

The quadratic spectral kernel never reaches zero: hfhd's `krvm` loops
over every lag (39 microseconds a tick at n = 23,400, against 0.7 for
Parzen), and an exact one-pass form would keep every return and every
`Gamma_h`, which is the stream's length. Cut at `L` lags it is one pass
in `8 (L d + d + d^2 + 1)` bytes; the cut costs 1% at `L = 10 H` and 1e-4
of the estimate at `L = 100 H`. Its weights fall as `x^-2` with an
oscillation, so the gap does not fall evenly with `L`.

## 5. Against the truth

{{ACCURACY}}

On this design's noise (`xi^2` = 7.5e-4) the plain realised variance is
36 times the truth on the one-second grid and 7 times on the refresh-time
grid; every noise-robust estimator removes that. Over 40 days:

* The bias-corrected pre-averaged estimate and the flat-top kernels are
  unbiased within 2% with a root mean square of 5% (one-second grid) and
  10% (refresh time) for the variance, 8% to 17% for the covariance.
* BNHLS's kernel with the jitter, at the bandwidth from the rule, is 2% to
  7% high for the variance (the price of a positive semi-definite kernel,
  whose bias in the noise falls as `1/H^2`), with a root mean square of 7%
  to 11%.
* Two and multi scales at hfhd's default scales have 4 to 5 times that
  root mean square (22% to 34%), and on the refresh-time grid, about 4,150
  returns, are 13% to 17% low: their scales `n^(2/3)` and `sqrt(n)` leave
  few subsamples at that length.
* `rcov`'s default pre-averaged form (`psd = true`: `theta` 1, the longer
  window `n^0.6`, no bias term) has 3.4 times the bias-corrected form's
  root mean square for the variance (18% against 5% on the one-second
  grid): positive semi-definiteness is bought at that price here.
* The plain Hayashi-Yoshida on all the ticks has a root mean square of 22%
  for the covariance; pre-averaging it (theta 0.4) does not improve on
  that at this noise.

These are one design's numbers, a yardstick for a future test's
statistical bands, not a ranking.

## 6. State and cost

{{STATE}}

**Counting.** Every byte count is `sum(a.nbytes)` over the state dict
after a day's rows, and equals its formula. The configuration (the scales
and their weights, the pre-averaging weights, the kernel weights) is fixed
before the first row and is not state: it is a function of the settings.
The accumulators are stored as full `d x d` matrices; each is symmetric
except the per-lag `Gamma_h`, so a library form would keep a triangle,
`d (d + 1) / 2`, and halve that term. The refresh-time stamp sits in the
state as an `int64`; the Hayashi-Yoshida stamp as an `f64` (exact below
2^53 nanoseconds, 104 days).

**Per estimator:**

* **tsrc**: a ring of the last `K` price vectors (`Y_{i-K}` for row `i`)
  and two sums. `K` is fixed before the first row (`int((n + 1)^(2/3))`
  needs `n`); `nbar_K`, `nbar_J` read the actual count at close. Rows are
  prices; on returns, `rcov`'s rows, the ring would hold the running sum
  of returns, whose differences are the same to the rounding of the sum.
* **msrc**: a ring of `N + M` prices and three sums: the weighted sum over
  the `M` scales (the weights `a_m / K_m` are known before the first row)
  and the two end scales' raw sums for the end term, whose `zeta` needs
  `n` at close. Cost per tick is `M` outer products, so it grows as
  `sqrt(n)`: 1.2 microseconds at `M = 153`, `d = 2`.
* **mrc**: a ring of the last `k - 1` returns, the last price, the sums
  of `Ybar Ybar'` and of `r r'`, two counts. Cost `k d` for each window
  plus `d^2`.
* **krvm, Parzen**: at a fixed `H` one accumulator suffices, each return
  adding `w_h (x y' + y x')` for each lag its ring holds. With `H` chosen
  at close (BNHLS's rule reads the block's own noise and variance), every
  lag up to the largest possible `H` keeps its own `Gamma_h`: `rcov`'s
  design, whose ring is `ceil(c* n^{3/5})` at `xi = 1`, 1,470 lags at `n`
  = 23,400, and that is 34 microseconds a tick at `d = 2`. The jitter is
  `rcov`'s (a return enters only once it cannot be part of the end
  jitter) and is not rebuilt here.
* **Refresh time**: each asset's last price and a seen flag, the open
  stamp. A stamp is decided only when a later stamp arrives (or at close),
  because hfhd's grid, and the definition, take every tick at the
  refresh stamp, including a second asset's tick at the same stamp: one
  stamp of look-ahead, not one row.
* **Hayashi-Yoshida**: per pair, A's anchor (B's path as of A's last
  tick), the first pending A return and its anchor, the sum of the later
  pending ones (anchored at B's current path), a pending count; per asset,
  the last price, the running path, the tick at the open stamp. A closed A
  return waits for B's next tick, which completes it. Within a stamp: A's
  return closes, then B's tick completes what is pending, then A's anchor
  is read. For `d` assets pairwise this is `8 (6 d (d - 1)/2 + 5 d + 2)`
  bytes or so (one pair built here). Rows are a long frame, one asset's
  tick a row: a different input from `rcov`'s refresh-time grid.
* **Pre-averaged Hayashi-Yoshida**: not built as one pass. An A window
  `(s, e]` needs every B window that ends after `s` (a running sum of B's
  pre-averaged returns read at `s` gives that) up to the last that starts
  before `e`, which closes `k - 1` of B's ticks after `e`. So each A
  window waits `k - 1` B ticks, and the queue holds every A window closing
  meanwhile: about `k * lambda_A / lambda_B` windows, bounded by the
  shape only if the tick-rate ratio is capped. With the slower asset as A
  (hfhd's choice) the queue is about `k`. A library form needs that cap or
  a refusal past it.

**Cost.** One pass costs what its formula's inner products cost: `O(d^2)`
a row for tsrc, `O(M d^2)` for msrc, `O(k d + d^2)` for mrc, `O(H d^2)`
for the kernel. The batch numpy (vectorised over rows) is faster where
the work is a few large matrix products (tsrc 5.8 ns a tick, the Parzen
kernel 105) and slower where it loops (mrc's window at `k = 419`, 5.6
microseconds); hfhd's own calls carry about 0.4 to 0.8 microseconds a
tick of pandas and refresh-time overhead at `d = 2`. The one-pass loops
are numba-compiled and not tuned: a Rust form would be of the same order.
Each timed run took under 2 seconds; the longest is hfhd's quadratic
spectral kernel, 0.9 seconds a call.

## 7. Proposed tolerances for an oracle test

| check | tolerance (relative to the largest entry) | basis |
|---|---|---|
| refresh time | equal (stamps and values) | measured equal |
| `tsrc`, `msrc` (hfhd's `n`) | 1e-13 (1e-12 for a one-pass `msrc` at large `M`) | measured at most 7.4e-14 (one pass, whose `M` scales are summed row by row) and 4.9e-15 (batch) |
| `mrc` with hfhd's four conventions mapped | 1e-13 | measured at most 1.2e-14 |
| `krvm`, either form, explicit `H` | 1e-12 | measured at most 3.0e-13: the noise cancellation amplifies rounding by `Gamma_0 / K`, about 36 here; a noisier day needs more, so scale the bound by that ratio |
| `hayashi_yoshida`, `k = 1` | 1e-13 | measured at most 2.8e-15 |
| pre-averaged HY, hfhd's indexing | 1e-13 | measured at most 6.3e-16 |
| any form against the paper where hfhd differs | not an oracle: state the convention and map it (as `tests/test_rcov.py` maps `mrc`'s first window) | Section 4 |

`rcov`'s existing bounds (`KRVM_TOL` and `HFHD_TOL`, both 1e-13) hold for its own tests, which run 400 to 1,000 returns with milder cancellation (measured 1.5e-15 and 2.0e-15 there); a kernel test at this study's length and noise needs 1e-12.
For a covariance, take the bound relative to the matrix's largest entry,
not to the entry itself, which can be near zero.

## 8. Caveats and what was not done

* **hfhd's pairwise paths and `hayashi_yoshida` fail under numpy 2**
  without the padding shim (Section 2). This holds in the repository's
  environment too.
* **Repeated stamps within one asset**: hfhd's `refresh_time` finds a
  stamp with `searchsorted(side="left")` and so takes the *first* price of
  a repeated stamp, where the previous-tick rule would take the last; the
  ticks here never repeat a stamp within an asset, so that is untested.
  hfhd's search also reads one element past an asset's last stamp (numba
  does not check bounds) before testing the position; the comparison
  that follows makes the read harmless.
* **Not checked**: `krvm(pairwise=True)`; `ensemble`; `preaverage` on its
  own (it is inside `mrc` and `hayashi_yoshida`, both checked); a
  one-pass jitter (`rcov` has it); a one-pass pre-averaged
  Hayashi-Yoshida (Section 6); weighted forms (Section 1).
* **The paper's edge term in `msrc`**: Tao, Wang and Chen 2013 was not at
  hand, so whether their `n` counts prices or returns is unverified; the
  two readings differ by 2e-6 at most.
* **Accuracy figures** are for one simulated design (constant noise,
  deterministic volatility, no jumps, Poisson-like ticks); they say
  nothing about real quotes.
* The one-pass forms take prices, as hfhd does. `rcov` takes returns; a
  price rebuilt as a running sum of returns differs from the quoted price
  in its last bits, so a test of a returns-fed form against hfhd on prices
  sees 1e-16-relative differences in each difference of prices (log-prices
  near 4.6, returns near 1e-4: up to 5e-12 of a single return), which the
  tolerances above absorb only after summation; build the test's prices
  as the running sum of the returns it feeds.
