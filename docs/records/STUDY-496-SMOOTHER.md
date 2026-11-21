# Study 496: a fixed-lag smoother beside `kalman` (part 1 of 3)

Status: part 1 done (the recursion, its exactness, the k = 2 comparison against
statsmodels and filterpy, the out-of-sample check, cost and state size). Parts 2
and 3 are not started.

Script: `scripts/study_496.py` (Python and numpy only; seeded generators, no data
files). `uv run python scripts/study_496.py` runs everything in about four
minutes; `--quick` runs one seed at N = 2000 in about 20 seconds. Versions:
numpy 2.5.5, scipy 1.18.1, statsmodels 0.15.0, filterpy 2.0.0 (installed into
the dev venv for the study; it is not in `pyproject.toml`).

## Question

`kalman` is a regression whose coefficients follow a random walk. A fixed-lag
smoother would emit, at row `t`, the coefficient estimate for row `t - L`
smoothed with the `L` rows after it. Is that exact, does it leak the future into
a row's own prediction, and how does it approach the full (fixed-interval)
smoother as `L` grows?

## 1. The model and the two forms

### The model (`kalman`'s state form, part 1's case)

From `crates/rheon-core/src/kalman.rs`'s header, with one clock unit between
rows, unit weights, no reverting slots (`Phi = I`) and the noise given:

```text
beta_t = beta_{t-1} + eta_t,      eta_t ~ N(0, Q)        (random walk; Q = q D², D = 1)
y_t    = x_t' beta_t + eps_t,     eps_t ~ N(0, R)
beta_1 ~ N(0, P0)                                         (the prior at the first row)
```

`kalman` has no exponential forgetting: its forgetting *is* `Q` (a per-factor
half-life is turned into `q = (ln 2 / h)²`), so part 1 needs none. Where the
noise is learned, `kalman` holds `P` and `Q` in units of `R`; the study takes
`R = 1`, so its `P` and `Q` are `kalman`'s numbers in those units.

The filter, per row `t`:

```text
P_{t|t-1} = P_{t-1|t-1} + Q          (not on row 1: P_{1|0} = P0)
b_{t|t-1} = b_{t-1|t-1}
pred_t    = x_t' b_{t|t-1}           (before y_t: out of sample)
s_t       = x_t' P_{t|t-1} x_t + R
g_t       = P_{t|t-1} x_t / s_t
b_{t|t}   = b_{t|t-1} + g_t (y_t - pred_t)
P_{t|t}   = P_{t|t-1} - g_t x_t' P_{t|t-1}
```

### (a) The ring: a lag-L RTS pass

Keep the last `L` filtered pairs `(b_{s|s}, P_{s|s})`, `s = t-L .. t-1`; row
`t`'s own pair is the filter's state. After row `t`'s update run the
Rauch-Tung-Striebel recursion backward from `t` to `t - L`:

```text
start:  b_{t|t},  P_{t|t}
for s = t-1 down to t-L:
    P_{s+1|s} = P_{s|s} + Q                    (predicted covariance; b_{s+1|s} = b_{s|s})
    G_s       = P_{s|s} P_{s+1|s}^{-1}
    b_{s|t}   = b_{s|s} + G_s (b_{s+1|t} - b_{s|s})
    P_{s|t}   = P_{s|s} + G_s (P_{s+1|t} - P_{s+1|s}) G_s'
emit (row t-L): b_{t-L|t}, P_{t-L|t}
```

This is the fixed-interval RTS smoother run on the prefix `y_1 .. y_t`, stopped
after `L` steps. The RTS recursion at `s` reads only `(b_{s|s}, P_{s|s})` and
the smoothed pair at `s + 1`, never anything before `s`, so stopping early
changes nothing: the emitted pair *is* `E[beta_{t-L} | y_1..y_t]` and its
covariance, exactly. `G_s` depends only on `P_{s|s}`, so it can be computed once
when row `s` is filtered and kept in the ring instead of `P_{s|s}` being
re-factored each row.

### (b) The augmented state

Stack `z_t = (beta_t, beta_{t-1}, ..., beta_{t-L})`, of size `k(L+1)`:

```text
z_t = F z_{t-1} + (eta_t, 0, ..., 0)
F   = [ I 0 ... 0 0 ]       (beta_t = beta_{t-1} + eta_t)
      [ I 0 ... 0 0 ]       (the old beta_{t-1} moves to slot 1)
      [ 0 I ... 0 0 ]       (and so on: a shift; the oldest slot falls off)
      [ 0 0 ... I 0 ]
Q_z = blockdiag(Q, 0, ..., 0),    H_t = (x_t', 0, ..., 0),    R as before
```

An ordinary Kalman filter on `z_t` gives `E[z_t | y_1..y_t]`; its last block is
`E[beta_{t-L} | y_1..y_t]`, the same quantity as (a). The full covariance is
`k(L+1) x k(L+1)`.

**(b') reduced augmented.** The filter on `z_t` only ever reads the first block
column of its covariance (`P H_t'` touches column block 0), so the blocks it
needs are

```text
a_j = E[beta_{t-j} | Y_t],   C_j = cov(beta_{t-j}, beta_t | Y_t),   D_j = cov(beta_{t-j} | Y_t)
shift:  a_j <- a_{j-1},  C_j <- C_{j-1},  D_j <- D_{j-1},  C_0 <- C_0 + Q,  D_0 = C_0
        (cov(beta_{t-j}, beta_t) = cov(beta_{t-j}, beta_{t-1}): eta_t is independent of the past)
update: h_j = C_j x_t,  s = x_t' h_0 + R,  a_j += h_j (y_t - x_t' a_0) / s,
        C_j -= h_j h_0' / s,  D_j -= h_j h_j' / s
```

`D_j` is needed only to emit the variance; the mean alone needs `a_j` and `C_j`.
This is the classic fixed-lag smoother by state augmentation with the unused
off-diagonal blocks dropped, `O(L k²)` a row.

### The emission rule

At row `t`, after the filter's update with `y_t`, emit the smoothed estimate for
row `t - L` (rows `t < L` emit nothing; the last `L` rows of a stream get theirs
only at a later row, or at an end-of-stream flush, which is not a row). The
emitted pair is a function of `y_1 .. y_t` alone: (a) reads the filter's states
up to `t`, (b) is a filter on rows up to `t`. Row `t`'s own prediction is
`x_t' b_{t|t-1}`, formed before `y_t`; neither form feeds anything back into the
filter, so every prediction is the plain filter's, bit for bit. The emitted
value on row `t`'s line does depend on `y_t`, as a post-update coefficient
does: it describes row `t - L`, not row `t`, and must be named so (the
"value of row `t - L` emitted at `t`" naming the scan asks to share with the
embargo).

The gap to the full smoother has a closed form. Subtracting the two RTS
recursions, which share every `b_{s|s}` and `G_s`,

```text
b_{s|s+L} - b_{s|N} = G_s G_{s+1} ... G_{s+L-1} (b_{s+L|s+L} - b_{s+L|N})
```

so the gap shrinks with the product of the smoother gains, whose spectral radius
is about `1 - sqrt(q x²/ R)` in steady state: a memory of roughly
`sqrt(R / (q x²))` rows. And given `y_1 .. y_{s+L}` the difference is a zero-mean
Gaussian with covariance `P_{s|s+L} - P_{s|N}`, which predicts its size.

## 2. Checks against statsmodels and filterpy

Variance mapping. statsmodels: `KalmanSmoother(k_endog=1, k_states=k,
k_posdef=k)` with a time-varying `design` of shape `(1, k, N)` holding `x_t'`,
`obs_cov = R`, `transition = selection = I`, `state_cov = Q`,
`initialize_known(0, P0)` (statsmodels' `a_1, P_1` is the prior at row 1, so no
`Q` before it). `UnobservedComponents` is not used: it is a level/trend model of
one series with no regression on a time-varying row (the scan's correction 7).
filterpy: `KalmanFilter(dim_x=k, dim_z=1)` with `F = I`, `Q`, `R = [[R]]`, one
`H_t = x_t'` per row through `batch_filter(..., Hs=, update_first=True)` (so
row 1 updates before any predict, matching `P_{1|0} = P0`), then
`rts_smoother`. Both libraries installed and ran. Data: `x_t ~ N(0, I_2)`,
`beta` a random walk with the same `Q = q I`, `R = 1`, `P0 = 10 I`, seeds 0 to 4.

### Full smoother: own numpy RTS against both libraries (k = 2, N = 2000, seed 0)

Largest difference relative to the largest magnitude:

```text
q=0.01: filtered vs sm 1.8e-16; smoothed mean vs sm 5.6e-16, cov vs sm 3.3e-15; mean vs filterpy 2.8e-16, cov vs filterpy 1.8e-15
q=0.001: filtered vs sm 1.6e-16; smoothed mean vs sm 7.5e-16, cov vs sm 4.4e-15; mean vs filterpy 5.0e-16, cov vs filterpy 5.3e-15
q=1e-05: filtered vs sm 1.6e-16; smoothed mean vs sm 2.6e-15, cov vs sm 6.0e-15; mean vs filterpy 9.9e-16, cov vs filterpy 1.8e-15
```

### Exactness: the emitted value against a library smoother on the prefix

For rows `t` in {L, 500, 1234, 1999}, statsmodels and filterpy each ran the
full RTS smoother on `y_1 .. y_t` and were read at `t - L`; the ring's emitted
mean and covariance for row `t - L` match both to rounding (q = 1e-3):

```text
L=1: max rel diff (mean and cov, 4 rows) vs statsmodels 1.7e-16, vs filterpy 2.2e-16
L=10: max rel diff (mean and cov, 4 rows) vs statsmodels 3.6e-15, vs filterpy 3.6e-15
L=100: max rel diff (mean and cov, 4 rows) vs statsmodels 6.2e-15, vs filterpy 7.1e-15
```

**The lag-L smoother is exact**: what it emits is the full smoother of the
prefix, to the last few ulps.

## 3. The two forms agree; the gap to the full smoother (k = 2)

"ring vs aug" and "ring vs reduced" are the largest difference over seeds, rows,
means and covariances, relative to the largest magnitude; "ring stream vs
vectorized" checks the streaming class against the batched arithmetic used for
the large runs (run at N = 2000 only, where the streaming ring is cheap enough).
The gap is `|b_{s|s+L} - b_{s|N}| / sd_{s|N}` per row and coefficient, `sd` the
full smoother's posterior standard deviation; "max" is over rows `s = 1 .. N-L`,
both coefficients and the 5 seeds; "RMS" is over rows `s >= 500` (past the
prior's burn-in), worst seed; the last column is the RMS the covariances predict,
`sqrt(mean((P_{s|s+L} - P_{s|N})_ii / (P_{s|N})_ii))`, worst seed.

| q | N | L | ring vs aug | ring vs reduced | ring stream vs vectorized | gap max, all rows | gap max, rows >= 500 | gap RMS, rows >= 500 | RMS predicted by the covariances |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 1 | 4.2e-16 | 4.2e-16 | 1.4e-16 | 8.49 | 3.89 | 0.95 | 0.9 |
| 0.01 | 2000 | 10 | 2.9e-15 | 2.4e-15 | 2.2e-16 | 2.42 | 1.87 | 0.421 | 0.399 |
| 0.01 | 2000 | 100 | 2.7e-15 | 2.6e-15 | 2.2e-16 | 0.000977 | 0.000851 | 0.000129 | 0.000128 |
| 0.01 | 10000 | 1 | 4.2e-16 | 4.2e-16 | n/a | 5.78 | 4.19 | 0.913 | 0.9 |
| 0.01 | 10000 | 10 | 2.9e-15 | 2.4e-15 | n/a | 2.03 | 2.03 | 0.403 | 0.4 |
| 0.01 | 10000 | 100 | 2.7e-15 | 2.6e-15 | n/a | 0.00122 | 0.00122 | 0.000123 | 0.000124 |
| 0.001 | 2000 | 1 | 2.8e-16 | 2.8e-16 | 1.7e-16 | 17 | 3.34 | 1.08 | 0.96 |
| 0.001 | 2000 | 10 | 2.9e-15 | 3.5e-15 | 2.6e-16 | 4.83 | 2.46 | 0.818 | 0.728 |
| 0.001 | 2000 | 100 | 3.7e-15 | 5.5e-15 | 3.6e-16 | 0.243 | 0.177 | 0.051 | 0.0487 |
| 0.001 | 10000 | 1 | 2.8e-16 | 2.8e-16 | n/a | 11.3 | 4.38 | 1.01 | 0.965 |
| 0.001 | 10000 | 10 | 2.9e-15 | 3.5e-15 | n/a | 3.57 | 3.21 | 0.769 | 0.733 |
| 0.001 | 10000 | 100 | 3.7e-15 | 5.5e-15 | n/a | 0.291 | 0.291 | 0.0485 | 0.0474 |
| 1e-05 | 2000 | 1 | 8.5e-16 | 8.5e-16 | 1.6e-16 | 59.1 | 2.83 | 0.913 | 0.928 |
| 1e-05 | 2000 | 10 | 4.7e-15 | 4.7e-15 | 5.2e-16 | 20.4 | 2.75 | 0.884 | 0.902 |
| 1e-05 | 2000 | 100 | 4.5e-15 | 4.5e-15 | 6.5e-16 | 3.26 | 1.78 | 0.636 | 0.685 |
| 1e-05 | 10000 | 1 | 8.5e-16 | 8.5e-16 | n/a | 35.4 | 3.22 | 0.964 | 0.987 |
| 1e-05 | 10000 | 10 | 4.7e-15 | 4.7e-15 | n/a | 13.3 | 3.13 | 0.938 | 0.959 |
| 1e-05 | 10000 | 100 | 4.5e-15 | 4.5e-15 | n/a | 2.63 | 2.36 | 0.709 | 0.724 |

The memory of the smoother, from its gains (seed 0, N = 2000):

```text
q=0.01: L=1: 0.927, L=10: 0.441, L=100: 0.00012; median spectral radius of one G 0.9264 (1/(1-rho) = 14 rows)
q=0.001: L=1: 0.973, L=10: 0.755, L=100: 0.0528; median spectral radius of one G 0.9727 (1/(1-rho) = 37 rows)
q=1e-05: L=1: 0.997, L=10: 0.97, L=100: 0.74; median spectral radius of one G 0.9970 (1/(1-rho) = 333 rows)
```

Read together:

- **The forms agree to floating precision at every L** (largest relative
  difference 5.5e-15, means and covariances), and the streaming ring equals its
  batched arithmetic to 6.5e-16.
- **The gap does not depend on N** (2000 and 10000 give the same RMS; the max
  grows a little with N because more rows draw more extremes). It is local: by
  the identity in section 1 it is the product `G_s .. G_{s+L-1}` applied to the
  filtered-vs-smoothed difference at `s + L`.
- **The gap goes to zero as L passes the smoother's memory**, about
  `sqrt(R / q)` rows here (14, 37 and 333 rows measured for q = 1e-2, 1e-3,
  1e-5). L = 1 is about 1 posterior SD off in RMS at every q (the lag-1 smoother
  has roughly twice the full smoother's variance); L = 10 is 0.4 SD at q = 1e-2
  and 0.8 to 0.9 SD at the slower drifts; L = 100 is 1.2e-4 SD at q = 1e-2,
  0.05 SD at q = 1e-3 and still 0.7 SD at q = 1e-5, where 100 rows are a third
  of the memory. The RMS tracks the median norm of the gain product (1.2e-4,
  0.053, 0.74 at L = 100) and the covariances' own prediction of it to within
  about 10 %, so the size of the gap is known in-stream from `P_{s|s+L}` and
  `G`: a model can report it.
- **The "all rows" max is the prior's burn-in**: in the first few hundred rows
  `P0 = 10 I` dominates and a lag-L estimate made from a handful of rows can be
  tens of SDs from the full smoother's (59 at L = 1, q = 1e-5). Past row 500 the
  max is 3 to 4 SD at small L, what an RMS near 1 over thousands of rows gives.
- So L is a choice the user makes against the drift's memory, not a number with
  a universal good value: `L ≈ 3 sqrt(R / q)` rows brings the gap to about 0.05
  SD. Under a half-life `h` (`q = (ln 2 / h)²` on standardized features) the
  memory is about `h / ln 2` clock units, so `L ≈ 4.3 h` rows at one row per unit.

## 4. Out of sample

`y_1001 .. y_2000` were replaced by `y` plus `5 N(0, 1)` noise (N = 2000,
q = 1e-3); separately `y_1000` alone was moved by 5.

```text
ring L=1: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
ring L=10: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
ring L=100: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
augmented L=1: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
augmented L=10: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
augmented L=100: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
reduced L=1: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
reduced L=10: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
reduced L=100: emitted rows <= t0 bit-identical under future perturbation: True; later emissions move: True; emission at t0 moves with y_t0: True; pred at t0 unaffected by y_t0: True
```

Every value emitted on rows up to 1000, and every prediction up to 1000, is
bit-identical under the change of later targets, in all three forms and at every
L; later emissions move (the test has teeth). The emission at row 1000 (for row
`1000 - L`) moves with `y_1000`, as it must, while row 1000's prediction does
not: the prediction is formed before `y_t` and the smoother never feeds back.

## 5. Per-row cost and state size (k = 2)

Measured in Python with one interpreted step per row, so the microseconds are
interpreter overhead more than arithmetic and say nothing about Rust; the
operation counts are the portable figure. The ring as written re-solves
`P_{s+1|s}` for each slot on each row; with `G_s` cached in the ring at the row
`s` was filtered, its mean pass is `L` matrix-vector products.

| form | L | us/row | floats in state | bytes |
|---|---|---|---|---|
| ring | 1 | 22 | 10 | 80 |
| augmented | 1 | 16 | 14 | 112 |
| reduced | 1 | 18 | 14 | 112 |
| ring | 10 | 112 | 55 | 440 |
| augmented | 10 | 15 | 275 | 2200 |
| reduced | 10 | 18 | 95 | 760 |
| ring | 100 | 1059 | 505 | 4040 |
| augmented | 100 | 141 | 20705 | 165640 |
| reduced | 100 | 29 | 905 | 7240 |

| form | per row (mean only) | per row (mean and covariance) | state, floats |
|---|---|---|---|
| (a) ring, `G` cached | `L k²` | `L k³` | `L (k + k(k+1)/2 + k²)` plus the filter's `k + k(k+1)/2` |
| (a) ring as written | `L k³` | `L k³` | `L (k + k(k+1)/2)` plus the filter's |
| (b) augmented, full | `(k(L+1))²` | `(k(L+1))²` | `k(L+1) + k(L+1)(k(L+1)+1)/2` |
| (b') reduced augmented | `L k²` | `L k²` | `k(L+1) + k(k+1)/2 + L (k² + k(k+1)/2)` |

For k = 2 the state is, in bytes (f64):

| L | ring (as written) | ring with `G` cached | augmented (symmetric) | reduced |
|---|---|---|---|---|
| 1 | 80 | 112 | 112 | 112 |
| 10 | 440 | 760 | 2,200 | 760 |
| 100 | 4,040 | 7,240 | 165,640 | 7,240 |

The full augmented form grows as `L²` and is ruled out past small L. The ring
and the reduced augmented form are both linear in `L` and `k²`; the reduced
form is `O(L k²)` a row with the covariance included and needs no solve, so it
is the cheaper of the two for emitting a variance. The ring is the cheaper
state for the mean alone, and it is the form whose structure is the RTS
smoother the oracles compute. Both are bounded by the model's shape and `L`,
never by the stream's length.

## Open for parts 2 and 3

Not studied here: `k` beyond 2 and correlated features; `kalman`'s other paths
(learned noise, the diffuse intercept, standardized coordinates and their
re-anchoring, reverting slots, weighted and zero-weight rows, clock gaps,
several targets, the square-root covariance form) and whether each commutes
with the lag; chunk invariance of the ring; and the output naming of the
delayed value.
