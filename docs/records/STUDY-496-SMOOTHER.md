# Study 496: a fixed-lag smoother beside `kalman`

Status: done (parts 1 to 3). Part 1: the recursion, its exactness, the k = 2
comparison against statsmodels and filterpy, the out-of-sample check, cost and
state size. Part 2 (sections 6 to 9): k = 5 and 20, exponential forgetting,
zero-weight rows, cost and state at k = 5 and 20. Part 3 (sections 10 to 16):
one script, the recursions and the emission rule restated, state and cost
formulas, numerical risks, the oracles, and the proposed test tolerances from
50 seeds at N = 10000.

Script: `scripts/study_496.py`, the single script for all three parts (Python
and numpy only; seeded generators, no data files). `uv run python
scripts/study_496.py --quick` runs every check of parts 1 and 2 and a one-seed
tolerance table in about 50 seconds; without `--quick` it runs the recorded
sizes; `--section <name>` (repeatable) runs chosen sections. Every section
records pass/fail checks and the run exits 1 if any fails (section 10).
Versions: numpy 2.5.5, scipy 1.18.1, statsmodels 0.15.0, filterpy 2.0.0
(installed into the dev venv for the study; it is not in `pyproject.toml`).

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

# Part 2: k = 5 and 20, forgetting, zero-weight rows, costs

Script: part 2 was written as `scripts/study_496_part2.py`, which generalized
the filter and the three forms (ring, full augmented, reduced augmented) to a
forgetting factor and row weights; part 3 merged it into `scripts/study_496.py`
(sections `forms`, `forget`, `zero`, `cost` there) and deleted it. The
generalized classes reduce to part 1's arithmetic bit for bit at `lam = 1` and
unit weights, so part 1's sections now run on them.
Same versions as part 1; same data (`x_t ~ N(0, I_k)`, `Q = q I`, `R = 1`,
`P0 = 10 I`, seeds 0 to 4, N = 2000). The augmented form is run while
`k(L+1) <= 600`, so not at k = 20, L = 100.

## 6. k = 5 and k = 20

### Exactness on the prefix (q = 1e-3, seed 0)

The emitted mean and covariance for row `t - L` against statsmodels'
`KalmanSmoother` and filterpy's `rts_smoother`, each run on `y_1 .. y_t`, for
`t` in {L, 500, 1234, 1999}; largest relative difference:

| k | L | vs statsmodels | vs filterpy |
|---|---|---|---|
| 5 | 1 | 2.8e-16 | 3.7e-16 |
| 5 | 10 | 5.3e-15 | 3.2e-14 |
| 5 | 100 | 1.5e-14 | 3.4e-14 |
| 20 | 1 | 4.3e-16 | 3.1e-15 |
| 20 | 10 | 1.5e-15 | 2.4e-14 |
| 20 | 100 | 2.9e-13 | 4.4e-13 |

Both libraries handled k = 20 without trouble (each call under 0.2 s); nothing
failed. The difference grows with k and L as rounding through `L` backward
steps of `k x k` solves does (filterpy inverts `P_{s+1|s}` explicitly), and
stays at the 1e-13 level at k = 20, L = 100: the emitted value is the prefix's
full smoother.

### The forms agree; the gap to the full smoother

Columns as in section 3 (5 seeds; "ring stream vs vectorized" on seed 0).

| k | q | L | ring vs aug | ring vs reduced | ring stream vs vectorized | gap max, rows >= 500 | gap RMS, rows >= 500 | RMS predicted by the covariances |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.01 | 1 | 3.8e-16 | 3.8e-16 | 4.2e-17 | 4.53 | 1 | 0.943 |
| 5 | 0.01 | 10 | 5.4e-15 | 5.7e-15 | 1.7e-16 | 2.47 | 0.49 | 0.46 |
| 5 | 0.01 | 100 | 6.4e-15 | 6.4e-15 | 1.7e-16 | 0.00229 | 0.000355 | 0.000348 |
| 5 | 0.001 | 1 | 3.8e-16 | 3.8e-16 | 1.3e-16 | 3.8 | 1.04 | 0.971 |
| 5 | 0.001 | 10 | 8.3e-15 | 9.6e-15 | 3.1e-16 | 3.08 | 0.801 | 0.747 |
| 5 | 0.001 | 100 | 8.4e-15 | 9.0e-15 | 4.0e-16 | 0.266 | 0.0567 | 0.0538 |
| 20 | 0.01 | 1 | 3.6e-16 | 3.6e-16 | 1.4e-16 | 5.57 | 1.1 | 1.1 |
| 20 | 0.01 | 10 | 2.8e-15 | 2.8e-15 | 2.9e-16 | 3.26 | 0.718 | 0.719 |
| 20 | 0.01 | 100 | not run | 3.0e-14 | 2.9e-16 | 0.0581 | 0.0088 | 0.00874 |
| 20 | 0.001 | 1 | 2.7e-16 | 2.7e-16 | 1.6e-16 | 4.15 | 1.03 | 1.03 |
| 20 | 0.001 | 10 | 2.8e-15 | 2.5e-15 | 2.4e-16 | 3.33 | 0.833 | 0.83 |
| 20 | 0.001 | 100 | not run | 3.1e-14 | 5.4e-16 | 0.421 | 0.0937 | 0.0925 |

"not run": the full augmented state at k = 20, L = 100 has `k(L+1) = 2020`
slots, 2,043,230 floats with the covariance stored symmetric (16,345,840 bytes;
32,659,360 as a full square), from the formula in section 5.

- **The forms agree to 3e-14 at every k and L**, and the streaming ring equals
  the batched arithmetic to 5.4e-16.
- **The gap grows with k at fixed q and L**: at q = 1e-2, L = 100 the RMS is
  1.3e-4 SD at k = 2, 3.6e-4 at k = 5 and 8.8e-3 at k = 20; at q = 1e-3, L = 100
  it is 0.051, 0.057 and 0.094. One target spread over k coefficients informs
  each of them less per row (the others' uncertainty is noise to it), so the
  smoother's memory is longer than part 1's `sqrt(R / q)` and the same L closes
  less of the gap. At L = 1 the gap stays about 1 SD at every k. The
  covariances still predict the RMS to within 7 % (0 to 1 % at k = 20), so a
  model can report the gap in-stream at any k.

## 7. Exponential forgetting

`kalman` has no forgetting factor (part 1, section 1); this adds one per row,
`lam` in (0, 1]. The change to the recursion is the prediction step alone:

```text
P_{t|t-1} = P_{t-1|t-1} / lam + Q        (was P_{t-1|t-1} + Q; the mean is unchanged)
```

That is the random walk with row-varying noise
`Q_t = (1/lam - 1) P_{t-1|t-1} + Q`. `P_{t-1|t-1}` depends on the rows' `x`
and weights, never on `y`, so this is still a linear Gaussian state-space model
and its RTS smoother is the full smoother "under the same forgetting". In the
smoother only the predicted covariance changes:

```text
P_{s+1|s} = P_{s|s} / lam + Q,   G_s = P_{s|s} P_{s+1|s}^{-1},   the rest as in section 1
```

In the augmented forms the extra noise enters `beta_t` alone, so the head
block is the only one that is inflated: `C_0 <- C_0 / lam + Q`, with
`C_j` and `D_j` (j >= 1) moved unchanged (`cov(beta_{t-j}, beta_t) =
cov(beta_{t-j}, beta_{t-1})` still holds, the noise being independent of the
past). Dividing every block by `lam` is a different model and does not agree
(column "all blocks / lam").

Oracles: (1) a hand-written full RTS under the same forgetting on the prefix
`y_1 .. y_t`; (2) statsmodels' `KalmanSmoother` with a time-varying
`state_cov[..., t] = (1/lam - 1) P_{t|t} + Q` (from this filter's `P_{t|t}`;
statsmodels' own filtered covariance then matches it to 5.4e-16 and its
filtered mean to 1.7e-15 at lam = 0.99); (3) filterpy's own fading-memory
filter (`alpha = 1/sqrt(lam)`, matching this filter to 8.5e-16), whose
`rts_smoother` has no `alpha` and is given `Qs[t+1] = (alpha² - 1) P_t + Q`
from filterpy's filtered covariances. k = 5, N = 2000, rows `t` in
{L, 500, 1234, 1999} for the prefix columns (seed 0); gap over 5 seeds. The
q = 0 rows forget only, on data that still drift at q = 1e-3.

| q | lam | L | ring vs hand RTS on prefix | vs statsmodels on prefix | vs filterpy on prefix | ring vs reduced | ring vs aug | all blocks / lam, vs ring | gap RMS, rows >= 500 | RMS predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 0.999 | 1 | 0 | 3.2e-16 | 4.1e-16 | 1.8e-16 | 1.8e-16 | 0.00054 | 1.03 | 0.97 |
| 0.001 | 0.999 | 10 | 0 | 8.0e-15 | 1.8e-14 | 7.7e-15 | 7.9e-15 | 0.0031 | 0.788 | 0.743 |
| 0.001 | 0.999 | 100 | 0 | 2.3e-14 | 2.3e-14 | 4.7e-15 | n/a | 0.0043 | 0.0533 | 0.0512 |
| 0.001 | 0.99 | 1 | 0 | 6.7e-16 | 4.0e-16 | 3.7e-16 | 3.7e-16 | 0.0054 | 0.931 | 0.962 |
| 0.001 | 0.99 | 10 | 0 | 4.7e-15 | 2.0e-14 | 1.1e-14 | 9.9e-15 | 0.03 | 0.683 | 0.705 |
| 0.001 | 0.99 | 100 | 0 | 1.0e-14 | 2.0e-14 | 6.0e-15 | n/a | 0.041 | 0.0294 | 0.0314 |
| 0 | 0.999 | 1 | 0 | 1.1e-14 | 1.5e-14 | 1.8e-16 | 1.8e-16 | 0.00074 | 13.6 | 0.926 |
| 0 | 0.999 | 10 | 0 | 2.0e-14 | 2.4e-14 | 3.9e-15 | 3.1e-15 | 0.0043 | 13.5 | 0.914 |
| 0 | 0.999 | 100 | 0 | 1.1e-14 | 2.5e-14 | 3.2e-15 | n/a | 0.0084 | 12.1 | 0.806 |
| 0 | 0.99 | 1 | 0 | 5.7e-16 | 7.6e-16 | 1.8e-16 | 1.8e-16 | 0.0058 | 2.44 | 0.964 |
| 0 | 0.99 | 10 | 0 | 1.1e-14 | 2.1e-14 | 5.1e-15 | 5.1e-15 | 0.032 | 2.23 | 0.881 |
| 0 | 0.99 | 100 | 0 | 1.1e-14 | 1.6e-14 | 3.6e-15 | n/a | 0.099 | 0.918 | 0.357 |

("n/a": the full augmented form is not run at L = 100 in this section.)

- **The ring is still exact under forgetting**: bit-identical to the
  hand-written full RTS on the prefix (the same arithmetic), and within 2.5e-14
  of statsmodels and filterpy. The reduced and full augmented forms agree with
  it to 1.1e-14 when only the head block is inflated.
- **With q = 0 the smoother's gain is `lam I` exactly** (2.7e-15 measured):
  `P_{s+1|s} = P_{s|s} / lam`, so the mean pass is
  `b_{s|t} = (1 - lam) b_{s|s} + lam b_{s+1|t}` with no solve, and the gap
  shrinks as `lam^L` (memory `1/(1 - lam)` rows: 100 at 0.99, 1000 at 0.999).
- **Forgetting alone on drifting data is a misspecified model**, and there the
  gap in posterior SD (12 to 14 at lam = 0.999) is far above what the
  covariances predict (0.8 to 0.9): the forgetting is too slow for the drift,
  the filtered estimate lags, and the posterior SD does not see the bias. The
  agreement columns are unaffected; the in-stream gap estimate is only as good
  as the model.

## 8. Zero-weight rows

Row weights enter as `R_t = R / w_t`. A zero-weight row advances the clock and
learns nothing: the prediction step runs (`P_{t|t-1} = P_{t-1|t-1} + Q`, or
`P0` on the first row), the prediction `x_t' b_{t|t-1}` is emitted, and the
update is **skipped**, so `b_{t|t} = b_{t|t-1}` and `P_{t|t} = P_{t|t-1}`. The
update must be skipped rather than run with a zero gain: `0 * (y_t - pred_t)`
is NaN when the zero-weight row's target is NaN or infinite, and no division by
`w` happens on such a row. Data: k = 5, q = 1e-3, N = 2000, 221 zero-weight
rows: row 0 (the first row), 200 scattered at random, and a run at rows
700 to 720; their targets set to NaN.

| L | any NaN (emitted, predictions; all forms) | NaN vs 1e300 targets on zero rows | ring vs reduced | ring vs aug | vs statsmodels (y missing) | vs filterpy (z = None) | kept rows vs dropped stream, clock carried | isolated zero row vs neighbours' midpoint |
|---|---|---|---|---|---|---|---|---|
| 1 | none | bit-identical | 1.8e-16 | 1.8e-16 | 2.2e-16 | 7.7e-16 | 4.4e-16 | 2.4e-17 |
| 10 | none | bit-identical | 7.4e-15 | 7.2e-15 | 1.5e-14 | 5.5e-14 | 8.9e-16 | 9.0e-17 |
| 100 | none | bit-identical | 7.3e-15 | n/a | 2.4e-14 | 5.4e-14 | 1.8e-15 | 2.3e-16 |

The library oracles mark the row missing (statsmodels: `y = NaN`; filterpy:
`z = None`), the prefix rows checked include the first-row emission (`t = L`)
and rows inside the run. The "dropped stream" removes the zero-weight rows and
carries the clock: the next kept row's prediction adds `Q` times the clock
units since the last kept row (and the first kept row's prior is `P0` plus `Q`
per zero row before it). Its filter matches the kept rows to 1.2e-16 (`P + Q +
Q` against `P + 2Q` is the only rounding), and its smoother, run from the
last kept row at or before `t` back to `r`, matches every kept row's emission.

What a zero-weight row emits, exactly. For a zero-weight row `r`, at
`t = r + L` the ring emits

```text
b_{r|t} = E[beta_r | y_1 .. y_t] = b_{r|r} + G_r (b_{r+1|t} - b_{r|r}),
b_{r|r} = b_{r-1|r-1},  P_{r|r} = P_{r-1|r-1} + Q,  G_r = P_{r|r} (P_{r|r} + Q)^{-1}
```

with its covariance from the same RTS step: the coefficient at row `r`'s clock
time given everything up to row `t`. Between two kept rows (unit clock steps,
equal `Q` either side) it is exactly the midpoint of the neighbours' smoothed
means given the same rows, `(b_{r-1|t} + b_{r+1|t}) / 2` (measured to 2.3e-16):
a linear interpolation in clock time, as the random walk's bridge gives. On the
first row (`b_{0|0} = 0`, `P_{0|0} = P0`) it is `P0 (P0 + Q)^{-1} b_{1|t}`
(0 difference measured at L = 10). A kept row's emission is unchanged by the
zero-weight rows except through the clock: it equals the dropped stream's
smoothed value conditioned on the same targets, which is that stream's
lag `L - m` value when `m` zero-weight rows fall inside the window. No NaN at
any L in any form.

## 9. Per-row cost and state size (k = 5 and 20)

As in section 5: one interpreted Python step per row, so these microseconds are
interpreter overhead plus numpy dispatch and say nothing about Rust; timed after
the first L rows (q = 1e-3, seed 0), on 1000 rows, or 300 for the two rings
and the augmented form at L = 100. Bytes are f64 from section 5's formulas (the ring with `G`
cached and the reduced form have the same count,
`(k + k(k+1)/2) + L (k + k(k+1)/2 + k²)`).

| k | L | ring us/row | bytes | ring, G cached us/row | bytes | augmented us/row | bytes | reduced us/row | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1 | 24 | 320 | 25 | 520 | 15 | 520 | 18 | 520 |
| 5 | 10 | 142 | 1,760 | 83 | 3,760 | 23 | 12,760 | 20 | 3,760 |
| 5 | 100 | 1258 | 16,160 | 673 | 36,160 | 2641 | 1,026,160 | 34 | 36,160 |
| 20 | 1 | 34 | 3,680 | 38 | 6,880 | 20 | 6,880 | 21 | 6,880 |
| 20 | 10 | 230 | 20,240 | 141 | 52,240 | 213 | 196,240 | 38 | 52,240 |
| 20 | 100 | 2228 | 185,840 | 879 | 505,840 | not run | 16,345,840 (formula) | 143 | 505,840 |

Formula only: the augmented form at k = 20, L = 100 (not run; 32,659,360 bytes
if its covariance were stored as a full square). Every other entry was
measured, and every byte count is the formula.

- The reduced augmented form is the cheapest per row at every k and L, with
  the covariance included (143 us at k = 20, L = 100 against 879 for the ring
  with `G` cached and 2228 for the ring as written, which solves `L` systems a
  row). The full augmented form is `(k(L+1))²` a row and in state: 2.6 ms and
  1 MB at k = 5, L = 100, 16 MB at k = 20, L = 100. It stays ruled out.
- State: the reduced form and the cached ring are bounded by `k²` and `L`,
  never by the stream. At k = 20, L = 100 that is 506 KB against the filter's
  own 1,840 bytes (`k + k(k+1)/2` floats), 275 times; at k = 5, L = 100,
  36 KB against 160 bytes. A lag of 100 at k = 20 is a significant growth in
  state over `kalman`'s and would go to the user with these numbers before it
  lands; the ring as written (186 KB at k = 20, L = 100, mean and covariance,
  `L` solves a row) is the smallest state that still emits a covariance.

# Part 3: one script, the recursions, state, cost, risks, tolerances

## 10. The consolidated script

`scripts/study_496.py` now holds parts 1 and 2: one filter (`filter_gen`, with
forgetting and row weights), one full RTS (`full_rts_gen`), the batched ring
(`ring_vec_gen`), the three streaming forms (`Ring`, with `cache_g`;
`Augmented`; `Reduced`), the two library oracles (`sm_smoother`,
`fp_smoother`) and nine sections: `full`, `exact`, `forms`, `contraction`,
`oos` (part 1), `forget`, `zero`, `cost` (part 2, `exact`, `forms` and `cost`
also covering k = 5 and 20) and `seeds` (part 3). Part 1's own classes were
dropped: at `lam = 1` and `w = 1` the generalized ones do the same operations
(`P / 1.0 + Q`, `R / 1.0`), and the full run reproduces every recorded
number of sections 2, 3 and 6 to 8 to the printed digit (the timings of
sections 5 and 9 aside), except the "ring stream vs vectorized" column at
k = 2, which is now taken on seed 0 alone (8.1e-17 to 6.5e-16, was 1.4e-16 to
6.5e-16 over five seeds).

Each section records checks with a threshold, and the run exits 1 if one
fails: the library comparisons below 1e-11 relative, the forms against each
other below 1e-12, the streaming ring against its batched arithmetic below
1e-13, the forgetting ring against the hand-written full RTS below 1e-13, the
out-of-sample and zero-weight properties exactly (bit-identical where section
4 or 8 says so), the gain products shrinking with L, the "all blocks / lam"
foil differing by more than 1e-6 (the test has teeth), and the covariances'
predicted gap RMS within a factor 0.7 to 1.4 of the measured one.

| run | checks | wall time (4 cores) |
|---|---|---|
| `--quick` (one seed, N = 2000, prefix rows {L, 1999}, `lam` 0.99 only, the streaming ring where L <= 10 or k = 2, the augmented form where `k(L+1) <= 250`, a one-seed tolerance table at N = 2000) | 129 of 129 passed | 49 s |
| sections `full` to `cost` at the recorded sizes | 151 of 151 passed | 195 s |
| `--section seeds --seeds 50 --n 10000 --workers 3` (under `nohup`, logged) | 12 of 12 passed | 375 s |

## 11. The recursions, together

With a forgetting factor `lam` in (0, 1] (1: none) and row weights `w_t >= 0`:

```text
filter, row t:
    P_{t|t-1} = P_{t-1|t-1} / lam + Q           (row 1: P_{1|0} = P0; mean unchanged)
    pred_t    = x_t' b_{t|t-1}                  (before y_t)
    if w_t > 0:
        s_t = x_t' P_{t|t-1} x_t + R / w_t,  g_t = P_{t|t-1} x_t / s_t
        b_{t|t} = b_{t|t-1} + g_t (y_t - pred_t),  P_{t|t} = P_{t|t-1} - g_t x_t' P_{t|t-1}
    else:  b_{t|t} = b_{t|t-1},  P_{t|t} = P_{t|t-1}     (skipped, not a zero gain)
```

**(a) Ring (lag-L RTS).** State: the filter's `(b, P)` and `L` slots
`(b_{s|s}, P_{s|s})` for `s = t-L .. t-1`, optionally with
`G_s = P_{s|s} (P_{s|s} / lam + Q)^{-1}`, which is fixed once row `s` is
filtered. After row `t`'s update:

```text
m <- b_{t|t},  V <- P_{t|t}
for s = t-1 down to t-L:
    P_{s+1|s} = P_{s|s} / lam + Q
    m <- b_{s|s} + G_s (m - b_{s|s})
    V <- P_{s|s} + G_s (V - P_{s+1|s}) G_s'
emit for row t-L: (m, V);   then slot t mod L <- (b_{t|t}, P_{t|t}[, G_t])
```

**(b') Reduced augmented.** State: `a_j = E[beta_{t-j} | Y_t]` (j = 0..L),
`C_j = cov(beta_{t-j}, beta_t | Y_t)` (j = 1..L, `k x k`, not symmetric),
`D_j = cov(beta_{t-j} | Y_t)` (j = 1..L, symmetric), and `P = C_0 = D_0`.

```text
shift (row t > 1):  a_j <- a_{j-1},  C_j <- C_{j-1},  D_j <- D_{j-1}   (j = L down to 1)
                    C_0 <- C_0 / lam + Q,  D_0 = C_0                    (head block only)
update (w_t > 0):   h_j = C_j x_t,  s = x_t' h_0 + R / w_t,  e = y_t - x_t' a_0
                    a_j += h_j e / s,  C_j -= h_j h_0' / s,  D_j -= h_j h_j' / s
emit for row t-L:   (a_L, D_L)
```

The shift is a relabelling: stored as a ring of `L` slots it moves no memory.
The full augmented form (b) is the same filter on the stacked state with its
whole `k(L+1)` square covariance, and agrees with both to 1e-14 (sections 3, 6,
7, 8); it is a check, not a candidate.

**Emission rule.** Row `t` emits the value for row `t - L`, after row `t` is
seen: the pair `E[beta_{t-L} | y_1 .. y_t]` and its covariance. Nothing from a
row after `t` enters it. Both forms read only the filter's states up to `t`
(or are a filter on rows up to `t`), and neither feeds anything back into the
filter, so row `t`'s prediction is the plain filter's `x_t' b_{t|t-1}`, formed
before `y_t`, bit for bit. Rows `t < L` emit nothing for an earlier row; the
last `L` rows' values come only at a later row or an end-of-stream flush. The
value on row `t`'s line depends on `y_t`, as a post-update coefficient does,
and describes row `t - L`, so it must be named as row `t - L`'s value emitted
at `t`. Section 4 holds every emission up to row 1000 bit-identical when the
targets after row 1000 change, in all three forms and at every L.

## 12. State size

Floats are f64, 8 bytes. With `p = k + k(k+1)/2` (a mean and a symmetric
covariance), the filter's own `(b, P)` is `8p = 4k(k+3)` bytes, and:

| form | bytes | over the filter's own |
|---|---|---|
| (a) ring as written (`b`, `P` a slot) | `8 (L+1) p = 4 (L+1) k (k+3)` | `4 L k (k+3)` |
| (a) ring with `G` cached | `8 (p + L (p + k²)) = 4k(k+3) + 12 L k(k+1)` | `12 L k (k+1)` |
| (b') reduced augmented | `8 (k(L+1) + k(k+1)/2 + L (k² + k(k+1)/2))`, the same number | `12 L k (k+1)` |
| (b) augmented, full (symmetric) | `8 (m + m(m+1)/2) = 4 m (m+3)`, `m = k(L+1)` | |
| mean only: ring with `G`, no `P` slots | `8 (p + L (k + k²))` | `8 L k (k+1)` |

The cached ring and the reduced form store the same count: the ring's slot is
`(b_s, P_s, G_s)`, the reduced form's `(a_j, D_j, C_j)`.

| k | L | filter alone | ring as written | ring, `G` cached | reduced | augmented (full) | reduced / filter |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 40 | 80 | 112 | 112 | 112 | 3 |
| 2 | 10 | 40 | 440 | 760 | 760 | 2,200 | 19 |
| 2 | 100 | 40 | 4,040 | 7,240 | 7,240 | 165,640 | 181 |
| 5 | 1 | 160 | 320 | 520 | 520 | 520 | 3 |
| 5 | 10 | 160 | 1,760 | 3,760 | 3,760 | 12,760 | 24 |
| 5 | 100 | 160 | 16,160 | 36,160 | 36,160 | 1,026,160 | 226 |
| 20 | 1 | 1,840 | 3,680 | 6,880 | 6,880 | 6,880 | 4 |
| 20 | 10 | 1,840 | 20,240 | 52,240 | 52,240 | 196,240 | 28 |
| 20 | 100 | 1,840 | 185,840 | 505,840 | 505,840 | 16,345,840 | 275 |

All are bounded by `k` and `L`, never by the stream's length. Per target: a
bank with several targets of the same features holds one such state each.

## 13. Cost per row

Multiply-adds, leading terms (the filter's own `O(k²)` step excluded):

| form | mean only | mean and covariance |
|---|---|---|
| (a) ring as written | `L (k³ solve + k²)` | `L (k³ solve + 2k³)` |
| (a) ring, `G` cached | `L k²` (+ one `k³` solve a row) | `2 L k³` (+ one solve) |
| (b') reduced | `2 L k²` (`h_j`, `C_j`) | `3 L k²` (and `D_j`) |
| (b) augmented, full | `(k(L+1))²` | `(k(L+1))²` |

The reduced form is the only one whose covariance costs `O(L k²)`; the ring's
covariance pass is two `k x k` products per slot. Measured in Python (one
interpreted step a row, so interpreter and numpy dispatch more than arithmetic;
no figure for Rust), from the full run of section 10:

| k | L | ring us/row | ring, `G` cached | augmented | reduced |
|---|---|---|---|---|---|
| 2 | 1 | 25 | 26 | 24 | 18 |
| 2 | 10 | 131 | 145 | 28 | 29 |
| 2 | 100 | 1537 | 689 | 159 | 25 |
| 5 | 1 | 24 | 29 | 17 | 23 |
| 5 | 10 | 153 | 106 | 28 | 24 |
| 5 | 100 | 1603 | 694 | 1432 | 37 |
| 20 | 1 | 40 | 39 | 21 | 22 |
| 20 | 10 | 248 | 124 | 267 | 38 |
| 20 | 100 | 2619 | 1074 | not run | 187 |

The flop count at k = 20, L = 100 is about 120,000 multiply-adds a row for the
reduced form with its covariance and about 1.6 million for the cached ring,
against about 1,300 for the filter's own update: a factor near 100 on the
per-row work of the filter for the cheaper form.

## 14. Numerical risks

- **The prior-dominated first rows.** While `P0` dominates, a lag-L estimate
  built from a handful of rows can sit tens of posterior SDs from the full
  smoother's (59 SD at k = 2, L = 1, q = 1e-5; up to 21 SD at k = 5 and 20 in
  section 15's 50 seeds). This is the model being honest about a prior, not
  rounding, but it rules out a gap tolerance over all rows: the tolerances below
  are for rows 500 on, and the burn-in a test excludes must grow with the
  smoother's memory (longer for larger k and smaller q).
- **The gap grows with k.** One target informs each of k coefficients less per
  row, so the memory is longer than `sqrt(R / q)` and the same L closes less
  (L = 100, q = 1e-3: RMS 0.051, 0.057, 0.095 SD at k = 2, 5, 20). Rounding
  grows with k and L too: the emitted value is within 4.4e-13 of the libraries
  at k = 20, L = 100, against 2e-16 at k = 2, L = 1 (L backward steps through
  `k x k` solves; filterpy inverts `P_{s+1|s}` explicitly).
- **Small q.** The gains approach `I` (`G = P (P + Q)^{-1}`), so the gap
  shrinks only as about `(1 - sqrt(q x² / R))^L`: at q = 1e-5 an L of 100 still
  leaves 0.7 SD. The solve itself stays well conditioned (`P + Q` is as well
  conditioned as `P`). The covariance pass forms `V - P_{s+1|s}`, a difference
  of nearly equal matrices when `G` is near `I`; with q = 0 and `lam = 1`
  (a fixed regression) every lag's value is the latest filtered one. The
  in-stream gap estimate `P_{s|t} - P_{s|N}` is a difference of close numbers
  and should be read as an order of magnitude below about 1e-8 relative.
- **Symmetry and definiteness.** Neither `V` in the ring nor `D_j` in the
  reduced form stays symmetric under rounding; the study symmetrizes on
  emission and keeps `P` symmetric each row. Whether the square-root covariance
  form the filter can run in carries over to the lag is not studied.
- **Zero-weight rows** must skip the update (section 8): a zero gain times a
  NaN target is NaN, and it would never wash out.
- **Forgetting** inflates the head block only (section 7); dividing every block
  by `lam` is a different model and differs by up to 0.1 relative in the
  emitted means.

## 15. Oracles used

- statsmodels `KalmanSmoother` (time-varying `design`, `obs_cov` per row,
  `state_cov` per row for forgetting, `y = NaN` for a zero-weight row,
  `initialize_known(0, P0)`), run on each prefix `y_1 .. y_t` and read at
  `t - L`.
- filterpy `KalmanFilter.batch_filter(..., Hs=, Rs=, update_first=True)` and
  `rts_smoother(..., Qs=)` (fading memory `alpha = 1/sqrt(lam)`, `z = None` for
  a zero-weight row), on each prefix likewise.
- A hand-written full RTS (`full_rts_gen`), checked against both libraries on
  all rows to 6e-15, used for the gap to the full smoother and as the prefix
  oracle under forgetting (bit-identical to the ring there).
- Internal second derivations: the full augmented filter (a different
  algorithm: a Kalman filter on the stacked state), the dropped stream with the
  clock carried (zero-weight rows), the random-walk bridge midpoint, and
  `G_0 b_{1|t}` for a zero-weight first row.
- statsmodels' `UnobservedComponents` is not used: it has no regression on a
  time-varying row.

Largest relative difference of the emitted mean and covariance to each library
on the prefix (q = 1e-3, seed 0, N = 2000, rows `t` in {L, 500, 1234, 1999}):

| k | L | statsmodels | filterpy |
|---|---|---|---|
| 2 | 1 | 1.7e-16 | 2.2e-16 |
| 2 | 10 | 3.6e-15 | 3.6e-15 |
| 2 | 100 | 6.2e-15 | 7.1e-15 |
| 5 | 1 | 2.8e-16 | 3.7e-16 |
| 5 | 10 | 5.3e-15 | 3.2e-14 |
| 5 | 100 | 1.5e-14 | 3.4e-14 |
| 20 | 1 | 4.3e-16 | 3.1e-15 |
| 20 | 10 | 1.5e-15 | 2.4e-14 |
| 20 | 100 | 2.9e-13 | 4.4e-13 |

A test against a library smoother on the prefix can therefore hold 1e-11
relative at k <= 20, L <= 100 (23 times the largest seen), which is the
script's threshold.

## 16. The gap over 50 seeds, and the proposed test tolerances

50 seeds (0 to 49) fitted the budget: 375 s on 3 processes, run under `nohup`
with a log. N = 10000, `x_t ~ N(0, I_k)`, `Q = q I`, `R = 1`, `P0 = 10 I`. The
gap is `|b_{s|s+L} - b_{s|N}| / sd_{s|N}`, the full smoother's posterior SD;
"max" is over rows `s >= 500` (to `N - L - 1`) and the k coefficients of a seed.
The tolerance is 1.5 times the max over seeds, rounded up to a multiple of
0.05.

| k | q | L | gap max, rows >= 500, max over 50 seeds (seed) | median over seeds | gap RMS, rows >= 500, worst seed | RMS predicted, worst seed | gap max, all rows | proposed tolerance (SD) |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.01 | 1 | 5.35 (44) | 4.54 | 0.967 | 0.945 | 10.8 | 8.05 |
| 5 | 0.01 | 10 | 2.91 (5) | 2.36 | 0.472 | 0.46 | 4.55 | 4.40 |
| 5 | 0.01 | 100 | 0.0054 (37) | 0.00323 | 0.000355 | 0.000348 | 0.0054 | 0.05 |
| 5 | 0.001 | 1 | 5.27 (21) | 4.07 | 1.02 | 0.978 | 20.6 | 7.95 |
| 5 | 0.001 | 10 | 3.94 (21) | 3.14 | 0.783 | 0.751 | 9.35 | 5.95 |
| 5 | 0.001 | 100 | 0.337 (33) | 0.269 | 0.057 | 0.0542 | 0.337 | 0.55 |
| 20 | 0.01 | 1 | 7.81 (26) | 5.47 | 1.13 | 1.1 | 10.2 | 11.75 |
| 20 | 0.01 | 10 | 4.58 (26) | 3.64 | 0.736 | 0.72 | 8.08 | 6.90 |
| 20 | 0.01 | 100 | 0.0841 (24) | 0.0638 | 0.00838 | 0.0081 | 0.0841 | 0.15 |
| 20 | 0.001 | 1 | 6.16 (37) | 4.79 | 1.07 | 1.04 | 19.7 | 9.25 |
| 20 | 0.001 | 10 | 4.93 (37) | 3.86 | 0.865 | 0.835 | 17.8 | 7.40 |
| 20 | 0.001 | 100 | 0.586 (46) | 0.484 | 0.0953 | 0.0915 | 0.625 | 0.90 |

**PROPOSED TEST TOLERANCE** for the gap to the full smoother at rows 500 on,
N = 10000, in posterior SD, per k and L (the larger over the two q; a test that
runs one q can take that q's row above):

| k | L | tolerance (SD) | binding q |
|---|---|---|---|
| 5 | 1 | 8.05 | 1e-2 |
| 5 | 10 | 5.95 | 1e-3 |
| 5 | 100 | 0.55 | 1e-3 |
| 20 | 1 | 11.75 | 1e-2 |
| 20 | 10 | 7.40 | 1e-3 |
| 20 | 100 | 0.90 | 1e-3 (**open: the state rule**) |

Read together:

- These bound a maximum over about 10,000 k draws, so at L = 1 and 10 they are
  extreme-value numbers (an RMS near 1 SD gives a max of 5 to 8 over the
  47,500 to 190,000 row-coefficient draws of a seed), and they grow slowly with
  N and k. A test at another N or
  burn-in needs its own run; the script's `--seeds` and `--n` give it.
- At small L the gap is the model's, not an error: the lag-1 value is a
  different estimate with about twice the full smoother's variance, so a test of
  L = 1 against the full smoother checks only that nothing is grossly wrong.
  The sharp tests are the exactness on the prefix (1e-11 relative, section 15)
  and the covariances' prediction of the gap's RMS (worst-seed RMS within 6 %
  of the worst-seed prediction over the 50 seeds, above).
- At L = 100 the gap is small and its tolerance is a real check (0.05 SD at
  k = 5, q = 1e-2).
- **k = 20, L = 100 is the state rule's decision, which remains open.** The
  tolerance (0.90 SD; 0.15 at q = 1e-2) is proposed for the case it would test,
  but the state it needs, 505,840 bytes a target for either the reduced form
  or the ring with `G` cached (185,840 for the ring as written, which solves
  `L` systems a row), is 275 times the filter's own 1,840 bytes and is
  "significant growth" under the state rule: it goes to the user with these
  numbers before any lag that large lands at that k. Nothing here decides it;
  smaller `L` or `k` (36 KB at k = 5, L = 100; 52 KB at k = 20, L = 10) is the
  same question at a smaller size.

## Still open after part 3

The filter's other paths and whether each commutes with the lag: learned noise
(the noise scale changes `P`'s units mid-stream), the diffuse intercept,
standardized coordinates and their re-anchoring, reverting slots (`Phi != I`:
`P_{s+1|s} = Phi P Phi' + Q` and `G = P Phi' P_{s+1|s}^{-1}`), clock gaps
(section 8's dropped stream is the evidence that a gap enters as `Q` per clock
unit), several targets, and the square-root covariance form. Also chunk
invariance of the ring (it holds per row, so it should be exact), the output
naming of the delayed value, and the end-of-stream flush.
