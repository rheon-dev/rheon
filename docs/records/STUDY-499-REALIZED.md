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
| refresh time | `refresh_time` | identical stamps and values | same definition | yes (one stamp of look-ahead) | `8 (3 + d) + d` | 42 | 6.7 / - |
| two scales | `tsrc(J, K)` | 5.3e-16 | 2e-7 (hfhd counts prices for `n`) | yes | `8 (K d + 2 d^2 + 1)` | 13,160 (K = 818) | 51 / 142 |
| multi scale | `msrc(M, N, pairwise)` | 4.9e-15 | 2e-6 (the edge term's `n`) | yes | `8 ((N + M) d + 3 d^2 + 1)` | 5,000 (M = N = 153) | 1,830 / 7,050 |
| pre-averaged (MRC) | `mrc(theta, bias_correction, pairwise)` | 8.4e-15 | 1e-3 to 1e-2 with the bias term, 4e-4 to 4e-3 without | yes | `8 ((k - 1) d + d + 2 d^2 + 2)` | 1,072 (k = 62) | 318 / 604 |
| realised kernel, Parzen | `krvm(H, kernel=parzen_kernel)`, `get_bandwidth` | 3.0e-13 | a different weight function: 4% to 7% (Section 4) | yes at a fixed H; with H chosen at close, a ring of `H_max` | `8 (H d + d + d^2 + 1)`; `8 (H_max d + d + (H_max + 1) d^2 + 1)` | 936 (H = 55); 70,616 (H_max = 1,470) | 1,320 / 5,380; 44,500 / 84,100 |
| realised kernel, quadratic spectral | `krvm(H, kernel=quadratic_spectral_kernel)` | 3.0e-13 | as Parzen | **no**: every lag carries weight; cut at `L` lags it is one pass | `8 (L d + d + d^2 + 1)` | 1,176 (L = 10 H = 70; gap 1.1% to all lags) | 1,160 / 6,420 |
| Hayashi-Yoshida | `hayashi_yoshida(theta=None)` | 2.7e-15 | same definition | yes, on a long frame | `8 * 18 + 2` a pair | 146 | 5.4 / - |
| Hayashi-Yoshida, pre-averaged | `hayashi_yoshida(theta)` | 6.3e-16 | up to 10% on a day; over 40 days hfhd's form is 1.9% to 2.7% of the truth lower (3 to 4 standard errors), beside the estimator's own spread of 26% to 32% | one pass, but its pending queue is bounded only by the ratio of the two assets' tick rates | not bounded by the shape | - | not built (batch: hfhd 446, numpy 786) |

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
numpy 2.5.2, numba 0.68.1, pandas 3.0.5 and hfhd 0.1.4).

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
returns; `irregular`, A's ticks alone. On average about 585 stamps a day are shared
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

**Agreement with hfhd (largest gap over seeds 0-4, relative to the largest entry of hfhd's matrix)**

| estimator | hfhd setting | grid | batch, hfhd convention | one pass | 1 = 997 chunks | paper's form vs hfhd | state bytes |
|---|---|---|---|---|---|---|---|
| refresh_time | - | async | 0 | 0 | yes | - | 42 |
| hayashi_yoshida | theta=None (k=1) | async | 2.69e-15 | 2.80e-15 | yes | - | 146 |
| hayashi_yoshida | diagonal | async | 0 | - | yes | - | - |
| hayashi_yoshida | theta=0.4 (k=37) | async | 6.30e-16 | - | yes | 0.09576 | - |
| hayashi_yoshida | theta=0.6 (k=55) | async | 2.91e-16 | - | yes | 0.1024 | - |
| tsrc | J=1, K=None (int(N_p^(2/3))) | sync | 8.78e-17 | 1.01e-14 | yes | 8.74e-08 | 13160 |
| tsrc | J=2, K = int(N_p^(2/3)) | sync | 8.79e-17 | 9.84e-15 | yes | 9.04e-08 | 13160 |
| tsrc | J=5, K = int(N_p^(2/3)) // 2 | sync | 2.23e-16 | 1.00e-14 | yes | 8.92e-08 | 6616 |
| msrc | M=N=None (default), pairwise=False | sync | 1.89e-15 | 7.39e-14 | yes | 2.03e-07 | 5000 |
| msrc | M=10, N=0, pairwise=False | sync | 4.86e-15 | 1.60e-14 | yes | 6.63e-08 | 264 |
| msrc | M=1, pairwise=False | sync | 9.82e-16 | 7.65e-15 | yes | 9.82e-16 | 120 |
| mrc | theta=0.4, bias_correction=True, pairwise=False | sync | 6.21e-15 | 8.11e-15 | yes | 0.002559 | 1072 |
| mrc | theta=0.6, bias_correction=True, pairwise=False | sync | 8.37e-15 | 1.03e-14 | yes | 1.00e-03 | 1552 |
| mrc | theta=1.0, bias_correction=False, pairwise=False | sync | 5.00e-15 | 1.20e-14 | yes | 4.41e-04 | 6784 |
| krvm | parzen, flat-top k((h-1)/H), H=get_bandwidth | sync | 3.01e-13 | 8.26e-14 | yes | 0.0681 | 936 |
| krvm | parzen, BNHLS k(h/(H+1)), jitter m=1 (kernel passed) | sync | 2.84e-13 | - | yes | - | - |
| krvm | parzen, BNHLS k(h/(H+1)), jitter m=2 (kernel passed) | sync | 2.73e-13 | - | yes | - | - |
| krvm | quadratic_spectral, flat-top k((h-1)/H), H=get_bandwidth | sync | 2.98e-13 | - | yes | 0.06149 | - |
| krvm | quadratic_spectral, BNHLS k(h/(H+1)), jitter m=1 (kernel passed) | sync | 2.77e-13 | - | yes | - | - |
| krvm | quadratic_spectral, BNHLS k(h/(H+1)), jitter m=2 (kernel passed) | sync | 2.71e-13 | - | yes | - | - |
| tsrc | J=1, K=None (int(N_p^(2/3))) | async | 1.65e-16 | 2.56e-15 | yes | 6.35e-07 | 4200 |
| tsrc | J=2, K = int(N_p^(2/3)) | async | 1.66e-16 | 2.73e-15 | yes | 7.16e-07 | 4200 |
| tsrc | J=5, K = int(N_p^(2/3)) // 2 | async | 3.00e-16 | 3.90e-15 | yes | 8.58e-07 | 2136 |
| msrc | M=N=None (default), pairwise=False | async | 2.62e-15 | 3.55e-14 | yes | 2.10e-06 | 2184 |
| msrc | M=10, N=0, pairwise=False | async | 5.81e-16 | 1.12e-14 | yes | 4.17e-07 | 264 |
| msrc | M=1, pairwise=False | async | 4.84e-16 | 2.78e-15 | yes | 4.84e-16 | 120 |
| mrc | theta=0.4, bias_correction=True, pairwise=False | async | 1.87e-15 | 5.36e-15 | yes | 0.01045 | 496 |
| mrc | theta=0.6, bias_correction=True, pairwise=False | async | 3.53e-15 | 5.71e-15 | yes | 0.005614 | 704 |
| mrc | theta=1.0, bias_correction=False, pairwise=False | async | 3.58e-15 | 5.64e-15 | yes | 0.003743 | 2464 |
| krvm | parzen, flat-top k((h-1)/H), H=get_bandwidth | async | 1.64e-14 | 2.52e-14 | yes | 0.05282 | 504 |
| krvm | parzen, BNHLS k(h/(H+1)), jitter m=1 (kernel passed) | async | 1.51e-14 | - | yes | - | - |
| krvm | parzen, BNHLS k(h/(H+1)), jitter m=2 (kernel passed) | async | 1.61e-14 | - | yes | - | - |
| krvm | quadratic_spectral, flat-top k((h-1)/H), H=get_bandwidth | async | 1.48e-14 | - | yes | 0.06778 | - |
| krvm | quadratic_spectral, BNHLS k(h/(H+1)), jitter m=1 (kernel passed) | async | 1.58e-14 | - | yes | - | - |
| krvm | quadratic_spectral, BNHLS k(h/(H+1)), jitter m=2 (kernel passed) | async | 1.63e-14 | - | yes | - | - |
| tsrc | J=1, K=None (int(N_p^(2/3))) | irregular | 4.81e-16 | 2.72e-15 | yes | 2.06e-07 | 4136 |
| tsrc | J=2, K = int(N_p^(2/3)) | irregular | 4.82e-16 | 2.41e-15 | yes | 2.16e-07 | 4136 |
| tsrc | J=5, K = int(N_p^(2/3)) // 2 | irregular | 5.28e-16 | 4.39e-15 | yes | 2.28e-07 | 2080 |
| msrc | M=N=None (default), pairwise=False | irregular | 1.16e-15 | 2.66e-14 | yes | 5.20e-07 | 1776 |
| msrc | M=10, N=0, pairwise=False | irregular | 2.45e-15 | 1.51e-14 | yes | 1.44e-07 | 112 |
| msrc | M=1, pairwise=False | irregular | 6.00e-16 | 4.31e-15 | yes | 6.00e-16 | 40 |
| mrc | theta=0.4, bias_correction=True, pairwise=False | irregular | 4.10e-16 | 6.31e-15 | yes | 0.004344 | 384 |
| mrc | theta=0.6, bias_correction=True, pairwise=False | irregular | 6.17e-16 | 2.88e-15 | yes | 0.001771 | 552 |
| mrc | theta=1.0, bias_correction=False, pairwise=False | irregular | 3.67e-16 | 7.21e-15 | yes | 3.66e-04 | 2240 |
| krvm | parzen, flat-top k((h-1)/H), H=get_bandwidth | irregular | 2.02e-16 | 1.79e-14 | yes | 0.05876 | 368 |
| krvm | parzen, BNHLS k(h/(H+1)), jitter m=1 (kernel passed) | irregular | 1.21e-16 | - | yes | - | - |
| krvm | parzen, BNHLS k(h/(H+1)), jitter m=2 (kernel passed) | irregular | 3.88e-16 | - | yes | - | - |
| krvm | quadratic_spectral, flat-top k((h-1)/H), H=get_bandwidth | irregular | 0 | - | yes | 0.06019 | - |
| krvm | quadratic_spectral, BNHLS k(h/(H+1)), jitter m=1 (kernel passed) | irregular | 0 | - | yes | - | - |
| krvm | quadratic_spectral, BNHLS k(h/(H+1)), jitter m=2 (kernel passed) | irregular | 0 | - | yes | - | - |
| msrc | M=N=None, pairwise=True | async | 1.82e-15 | - | yes | - | - |
| mrc | theta=0.4, bias_correction=True, pairwise=True | async | 1.03e-15 | - | yes | - | - |

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

**Bias-corrected MRC (theta 0.4): each step from hfhd's convention to the paper's, alone (seed 0)**

| grid | k | change | moves the estimate by |
|---|---|---|---|
| sync | 62 | windows from the first return (n - k + 2 of them) | 3.35e-04 |
| sync | 62 | psi_1 summed to k, not k - 1 | 9.32e-04 |
| sync | 62 | rescaled by 1 / (1 - psi_1 / (2 psi_2 k^2)) | 0.001578 |
| sync | 62 | theta in the bias term = k / sqrt(n), not the input 0.4 | 0.001479 |
| sync | 62 | all four (the paper's form) | 0.002454 |
| async | 26 | windows from the first return (n - k + 2 of them) | 3.63e-04 |
| async | 26 | psi_1 summed to k, not k - 1 | 0.00263 |
| async | 26 | rescaled by 1 / (1 - psi_1 / (2 psi_2 k^2)) | 0.008743 |
| async | 26 | theta in the bias term = k / sqrt(n), not the input 0.4 | 0.001198 |
| async | 26 | all four (the paper's form) | 0.007547 |

Each of hfhd's four departures moves the estimate by 3e-4 to 9e-3 alone;
the largest is the missing rescale on the refresh-time grid, where the
window is short (`k` = 26) and the bias term's share of the integrated
covariance, `psi_1 / (2 psi_2 k^2)`, is larger. They partly cancel. `rcov(kind="preavg", psd=False)` is the paper's form (`tests/
test_rcov.py::test_the_preaveraged_estimate_is_its_definition`); its
`psd = true` default, which has no bias term, matches hfhd's
`bias_correction=False` to 3e-15 once hfhd's first window is mapped (the
existing `test_the_default_preaveraged_estimate_is_hfhds_mrc`).

**Pre-averaged Hayashi-Yoshida: the paper's windows against hfhd's (seeds 0-39; estimate / true covariance)**

| theta | k | hfhd mean | paper mean | paper, exact psi | paper - hfhd, mean | paper - hfhd, sd | hfhd's own sd |
|---|---|---|---|---|---|---|---|
| 0.4 | 36-37 | 0.9451 | 0.9639 | 0.9652 | 0.0188 | 0.03616 | 0.2582 |
| 0.6 | 54-55 | 0.9348 | 0.9615 | 0.9621 | 0.02668 | 0.04001 | 0.3187 |

The paper's windows and hfhd's give different numbers on a day (up to 10%
over five days) because the boundary pairs differ: each
window's interval carries one zero-weight return, at the front in hfhd's
form and at the back in the paper's, so the two sets of pairs differ by
pairs whose weighted returns do not overlap in time. Those products alone
would have mean zero, but over 40 days the paper's form sits 1.9%
(theta 0.4) and 2.7% (theta 0.6) of the truth above hfhd's, 3.3 and 4.2
standard errors of that mean: hfhd's form loses something systematic as
well, most likely the slower asset's first `k` windows it drops (about
`k / n`, 0.8% and 1.2% of them here) and the pairs at the ends. I did not
split the four index differences apart. Beside the estimator's own spread
of 26% to 32% it is small, and both forms are 3% to 7% low on average
here. The `psi_HY` choice (limit or exact) moves the mean by about 0.1%.

**Bandwidth: hfhd's get_bandwidth against BNHLS's rule (seed 0, asset A's noise and variance)**

| grid | kernel | n | xi^2 | c* hfhd | c* exact | H hfhd | H exact | ceil | BNHLS estimate moves |
|---|---|---|---|---|---|---|---|---|---|
| sync | parzen | 23400 | 2.78e-04 | 3.51 | 3.512 | 55 | 55.53 | 56 | 0.003288 |
| sync | quadratic_spectral | 23400 | 2.78e-04 | 0.46 | 0.4628 | 7 | 7.317 | 8 | 0.02156 |
| async | parzen | 4148 | 7.25e-04 | 3.51 | 3.512 | 28 | 28.86 | 29 | 0.002423 |
| async | quadratic_spectral | 4148 | 7.25e-04 | 0.46 | 0.4628 | 3 | 3.804 | 4 | 0.0065 |

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

**Quadratic spectral cut at L lags against every lag (seed 0)**

| grid | form | H | L / H | L | gap | state bytes, d = 2 |
|---|---|---|---|---|---|---|
| sync | bnhls | 7 | 1 | 7 | 0.32 | 168 |
| sync | bnhls | 7 | 2 | 14 | 0.2736 | 280 |
| sync | bnhls | 7 | 5 | 35 | 0.002702 | 616 |
| sync | bnhls | 7 | 10 | 70 | 0.01111 | 1176 |
| sync | bnhls | 7 | 20 | 140 | 0.00176 | 2296 |
| sync | bnhls | 7 | 50 | 350 | 7.77e-04 | 5656 |
| sync | bnhls | 7 | 100 | 700 | 1.64e-04 | 11256 |
| sync | bnhls | 7 | 400 | 2800 | 3.30e-06 | 44856 |
| sync | flat_top | 7 | 1 | 7 | 0.3308 | 168 |
| sync | flat_top | 7 | 2 | 14 | 0.2761 | 280 |
| sync | flat_top | 7 | 5 | 35 | 0.01409 | 616 |
| sync | flat_top | 7 | 10 | 70 | 0.009256 | 1176 |
| sync | flat_top | 7 | 20 | 140 | 0.001147 | 2296 |
| sync | flat_top | 7 | 50 | 350 | 3.83e-04 | 5656 |
| sync | flat_top | 7 | 100 | 700 | 9.65e-05 | 11256 |
| sync | flat_top | 7 | 400 | 2800 | 8.14e-06 | 44856 |
| async | bnhls | 3 | 1 | 3 | 0.2068 | 104 |
| async | bnhls | 3 | 2 | 6 | 0.08467 | 152 |
| async | bnhls | 3 | 5 | 15 | 0.02342 | 296 |
| async | bnhls | 3 | 10 | 30 | 0.003918 | 536 |
| async | bnhls | 3 | 20 | 60 | 0.002391 | 1016 |
| async | bnhls | 3 | 50 | 150 | 4.25e-04 | 2456 |
| async | bnhls | 3 | 100 | 300 | 2.96e-04 | 4856 |
| async | bnhls | 3 | 400 | 1200 | 1.16e-05 | 19256 |
| async | flat_top | 3 | 1 | 3 | 0.2018 | 104 |
| async | flat_top | 3 | 2 | 6 | 0.0635 | 152 |
| async | flat_top | 3 | 5 | 15 | 0.01477 | 296 |
| async | flat_top | 3 | 10 | 30 | 0.007791 | 536 |
| async | flat_top | 3 | 20 | 60 | 0.001392 | 1016 |
| async | flat_top | 3 | 50 | 150 | 2.24e-04 | 2456 |
| async | flat_top | 3 | 100 | 300 | 1.14e-04 | 4856 |
| async | flat_top | 3 | 400 | 1200 | 8.17e-06 | 19256 |

The quadratic spectral kernel never reaches zero: hfhd's `krvm` loops
over every lag (33 microseconds a tick at n = 23,400, against 1.0 for
Parzen), and an exact one-pass form would keep every return and every
`Gamma_h`, which is the stream's length. Cut at `L` lags it is one pass
in `8 (L d + d + d^2 + 1)` bytes; the cut costs 1% at `L = 10 H` and 1e-4
of the estimate at `L = 100 H`. Its weights fall as `x^-2` with an
oscillation, so the gap does not fall evenly with `L`.

## 5. Against the truth

**Against the truth (seeds 0-39): mean of estimate / true - 1, and its root mean square**

| grid | estimator | variance A, bias | variance A, rmse | covariance, bias | covariance, rmse |
|---|---|---|---|---|---|
| sync | realised covariance (msrc M=1) | 35.31 | 35.31 | -0.0415 | 0.5875 |
| sync | tsrc, K = n^(2/3), J = 1 | -0.08999 | 0.2453 | -0.1305 | 0.3692 |
| sync | tsrc, the same, ZMA's small-sample adjustment | -0.08891 | 0.2452 | -0.1295 | 0.3692 |
| sync | msrc, M = N = sqrt(n) | -0.01087 | 0.2237 | -0.0518 | 0.36 |
| sync | mrc, theta 0.4, bias-corrected (hfhd) | -0.004882 | 0.05238 | 0.005465 | 0.08782 |
| sync | mrc, theta 0.4, bias-corrected (paper) | -0.002647 | 0.05229 | 0.007181 | 0.08807 |
| sync | mrc, theta 1.0, no bias term (rcov's default) | -0.004377 | 0.1781 | -0.03528 | 0.2703 |
| sync | krvm parzen, flat-top (hfhd), H = get_bandwidth | 0.002914 | 0.05066 | 0.01101 | 0.08744 |
| sync | krvm parzen, BNHLS, jitter 2, H = ceil(exact) | 0.06513 | 0.0823 | 0.01071 | 0.08623 |
| sync | krvm quadratic_spectral, flat-top (hfhd), H = get_bandwidth | 0.001313 | 0.04665 | 0.009101 | 0.08345 |
| sync | krvm quadratic_spectral, BNHLS, jitter 2, H = ceil(exact) | 0.04565 | 0.06942 | 0.01412 | 0.09348 |
| async | realised covariance (msrc M=1) | 6.33 | 6.333 | -0.327 | 0.4459 |
| async | tsrc, K = n^(2/3), J = 1 | -0.1683 | 0.3204 | -0.232 | 0.4579 |
| async | tsrc, the same, ZMA's small-sample adjustment | -0.1653 | 0.3196 | -0.2292 | 0.4578 |
| async | msrc, M = N = sqrt(n) | -0.1334 | 0.3355 | -0.1732 | 0.4785 |
| async | mrc, theta 0.4, bias-corrected (hfhd) | -0.01712 | 0.09716 | -0.00766 | 0.1506 |
| async | mrc, theta 0.4, bias-corrected (paper) | -0.00819 | 0.09712 | 0.001241 | 0.1517 |
| async | mrc, theta 1.0, no bias term (rcov's default) | -0.03839 | 0.2368 | -0.08321 | 0.3627 |
| async | krvm parzen, flat-top (hfhd), H = get_bandwidth | 0.001253 | 0.1041 | 0.00739 | 0.1684 |
| async | krvm parzen, BNHLS, jitter 2, H = ceil(exact) | 0.04068 | 0.1112 | 0.003208 | 0.1622 |
| async | krvm quadratic_spectral, flat-top (hfhd), H = get_bandwidth | 0.004264 | 0.09535 | 0.02202 | 0.1529 |
| async | krvm quadratic_spectral, BNHLS, jitter 2, H = ceil(exact) | 0.02098 | 0.1098 | -0.004835 | 0.1751 |
| ticks | hayashi_yoshida, k = 1 | - | - | -0.0439 | 0.2193 |
| ticks | hayashi_yoshida, theta 0.4 (hfhd) | - | - | -0.05486 | 0.2608 |
| ticks | hayashi_yoshida, theta 0.4 (paper) | - | - | -0.03606 | 0.2592 |

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

**State and cost per tick (ns; best of 3; hfhd and numpy are batch, the one pass is numba-compiled)**

| estimator | setting | d | rows | state formula | bytes by formula | bytes counted | hfhd | numpy | one pass |
|---|---|---|---|---|---|---|---|---|---|
| tsrc | K = 818, J = 1 | 2 | 23400 | 8 (K d + 2 d^2 + 1) | 13160 | 13160 | 847.2 | 7.901 | 51.02 |
| msrc | M = N = 153 | 2 | 23400 | 8 ((N + M) d + 3 d^2 + 1) | 5000 | 5000 | 721.6 | 561.6 | 1834 |
| mrc | theta 0.4, bias-corrected, k = 62 | 2 | 23400 | 8 ((k - 1) d + d + 2 d^2 + 2) | 1072 | 1072 | 742.6 | 1017 | 317.9 |
| mrc | theta 1.0, no bias term, k = 419 | 2 | 23400 | 8 ((k - 1) d + d + 2 d^2 + 2) | 6784 | 6784 | 776.7 | 6394 | 1807 |
| krvm | Parzen, H = 55 fixed, one accumulator | 2 | 23400 | 8 (H d + d + d^2 + 1) | 936 | 936 | 983.1 | 137.6 | 1317 |
| krvm | Parzen, H chosen at close, ring 1470 | 2 | 23400 | 8 (H_max d + d + (H_max + 1) d^2 + 1) | 70616 | 70616 | 974 | 102.9 | 4.449e+04 |
| krvm | QS, H = 7, cut at L = 10 H = 70 | 2 | 23400 | 8 (L d + d + d^2 + 1) | 1176 | 1176 | 3.302e+04 | 2.422e+04 | 1158 |
| tsrc | K = 818, J = 1 | 10 | 23400 | 8 (K d + 2 d^2 + 1) | 67048 | 67048 | 1693 | 45.57 | 141.9 |
| msrc | M = N = 153 | 10 | 23400 | 8 ((N + M) d + 3 d^2 + 1) | 26888 | 26888 | 4401 | 3197 | 7048 |
| mrc | theta 0.4, bias-corrected, k = 62 | 10 | 23400 | 8 ((k - 1) d + d + 2 d^2 + 2) | 6576 | 6576 | 2075 | 1522 | 603.8 |
| mrc | theta 1.0, no bias term, k = 419 | 10 | 23400 | 8 ((k - 1) d + d + 2 d^2 + 2) | 35136 | 35136 | 2121 | 1.038e+04 | 1968 |
| krvm | Parzen, H = 56 fixed, one accumulator | 10 | 23400 | 8 (H d + d + d^2 + 1) | 5368 | 5368 | 2707 | 911.1 | 5378 |
| krvm | Parzen, H chosen at close, ring 1470 | 10 | 23400 | 8 (H_max d + d + (H_max + 1) d^2 + 1) | 1294488 | 1294488 | 3145 | 1022 | 8.405e+04 |
| krvm | QS, H = 7, cut at L = 10 H = 70 | 10 | 23400 | 8 (L d + d + d^2 + 1) | 6488 | 6488 | - | - | 6419 |
| refresh_time | 2 assets, a long frame | 2 | 16243 | 8 (3 + k) + k | 42 | 42 | 353.6 | 560.2 | 6.712 |
| hayashi_yoshida | one pair, k = 1 | 2 | 16243 | 8 (13 + 2 + 3) + 2 | 146 | 146 | 91.61 | 20.99 | 5.36 |
| hayashi_yoshida | pre-averaged, theta 0.4 (k = 37) | 2 | 16243 | not one pass with a bound (see text) | - | - | 445.6 | 786 | - |

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
  `sqrt(n)`: 1.8 microseconds at `M = 153`, `d = 2`.
* **mrc**: a ring of the last `k - 1` returns, the last price, the sums
  of `Ybar Ybar'` and of `r r'`, two counts. Cost `k d` for each window
  plus `d^2`.
* **krvm, Parzen**: at a fixed `H` one accumulator suffices, each return
  adding `w_h (x y' + y x')` for each lag its ring holds. With `H` chosen
  at close (BNHLS's rule reads the block's own noise and variance), every
  lag up to the largest possible `H` keeps its own `Gamma_h`: `rcov`'s
  design, whose ring is `ceil(c* n^{3/5})` at `xi = 1`, 1,470 lags at `n`
  = 23,400, and that is 44 microseconds a tick at `d = 2`. The jitter is
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
the work is a few large matrix products (tsrc 7.9 ns a tick, the Parzen
kernel 138) and slower where it loops (mrc's window at `k = 419`, 6.4
microseconds); hfhd's own calls carry about 0.4 to 1.0 microseconds a
tick of pandas and refresh-time overhead at `d = 2`. Timings move by up
to 50% between runs on this machine (two full runs; the tables are the
second); read them as orders of magnitude. The one-pass loops
are numba-compiled and not tuned: a Rust form would be of the same order.
Each timed run took under 2 seconds; the longest is hfhd's quadratic
spectral kernel, 0.8 seconds a call (2 seconds for the one-pass
kernel with a ring of 1,470 lags at `d = 10`). The whole script takes
about 4 minutes.

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
