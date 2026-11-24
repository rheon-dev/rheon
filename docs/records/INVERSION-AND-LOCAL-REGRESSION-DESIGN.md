<!-- kind: record -->
# Function inversion and local regression: the design record

The design for PLAN entry 563 (building block B10), part 1, written
2026-11-23 on branch `vm/task-563-inversion` from `main` at 555b45e. Nothing
here is built. The entry's text is on the coordinator's plan branch
(`vm/plan-550`), not yet on `main`, so this record cites it as "entry 563"
and never in the form the plan-reference test resolves against `main`'s
task lines. The questions at the end are for the user before the build
starts.

The entry asks for two blocks:

- **(A) a guarded monotone root-finder** that inverts a closed-form function
  row by row, the function given as a Polars expression, never as a per-row
  Python callback;
- **(B) decayed local-polynomial regression** on two or more features,
  evaluated at query points.

A pricing formula or a surface parametrization (Black-76, SVI) stays on the
user's side. Black-76 appears below only as the worked example a user would
write, and as the test case the motivating use needs.

## 1. What exists, and what this adds

- **No root-finder.** A grep of the crates and the package for `brent`,
  `bisect`, `illinois`, `regula` and `root` finds none. `garch` has a
  projected Newton optimizer for its start (`garch/optim.rs`), which is a
  minimizer, not a bracketing root-finder.
- **The formula grammar** (`python/rheon/polars/_formula.py`,
  `crates/rheon-polars/src/formula.rs`): an element-wise Polars expression
  over the window operators is read once, when the query is built, from
  `expr.meta.serialize(format="json")` into the library's own compact tree
  (nested lists: `col`, `lit`, `+ - * / **`, comparisons, `& |`, `neg abs
  exp sqrt log`, `clip`, `fill_null`, `is_null`, `when`, `cast`, `alias`,
  and the operators as `[name, input, {params}]`). The Rust side rebuilds
  each formula through Polars' public builders. Anything else (`shift`, a
  cumulative or rolling function, `over`, an aggregation) is refused by
  name. The tree, not Polars' serialized form, is what a state holds.
- **A one-sided local fit along one feature** already exists as a recipe
  (`docs/how-to/fit-locally-along-a-feature.md`): sort by the feature, give
  it as the clock, and the half-life becomes a one-sided exponential
  bandwidth. It is one feature, one-sided, and needs the rows sorted in
  memory. Block (B) is the two-sided kernel fit in any number of features
  on a stream in clock order.
- **Windows on a model's sums** are the snapshot ring of
  `crates/rheon-core/src/window.rs`: snapshots spaced on the clock
  (`snapshot_every`), subtracted to cut the window, the boundary chosen so
  nothing older than the window is kept, the ring thinned under
  `max_state_bytes`.
- **`max_state_bytes`**: 256 KB a group by default, 64 MiB at most
  (`crates/rheon-core/src/clock_span.rs`).
- **The `libm` crate** is already linked (rheon-core depends on it; `garch`
  calls it so its stream is the same bits on every platform). It has `erf`
  and `erfc`. Rust's standard library has neither on stable, and Polars
  0.55.2 (the Rust crate this workspace pins) has no `erf` expression.
  py-polars 2.1.0 (the lockfile's) has `Expr.erf` and `Expr.erfc`; whether
  every py-polars in the declared range `>=1.37.0,<3` has them was not
  checked.

## Part A: the root-finder

### A1. The interface

Three new names in `rheon.polars`, each returning a genuine `pl.Expr`, as
the window operators do:

```python
x = rp.unknown()                                # the quantity solved for
rp.norm_cdf(expr)                               # Phi, the standard normal CDF
rp.invert(
    f,                    # element-wise expression in x and the row's columns
    *,
    equals=0.0,           # the value f must take: a column, an expression or a number
    bracket=(lo, hi),     # each a number or an element-wise expression of the row
    increasing=True,      # the declared direction of f in x over the bracket
    xtol=2e-12,           # absolute tolerance in x's units, above 0
    rtol=4 * 2.0**-52,    # relative tolerance, at least 4 machine epsilons
    detail=False,         # True: a struct with the bracket, the evaluations and the reason
)
```

A user's Black-76 implied volatility, written on the user's side:

```python
s = rp.unknown()
v = s * pl.col("tau").sqrt()
d1 = ((pl.col("F") / pl.col("K")).log() + 0.5 * v * v) / v
call = pl.col("df") * (pl.col("F") * rp.norm_cdf(d1) - pl.col("K") * rp.norm_cdf(d1 - v))
out = rp.stream.with_windows(
    lf,
    iv=rp.invert(call, equals=pl.col("mid"), bracket=(1e-4, 5.0), increasing=True),
)
```

`rp.unknown()` is a column reference under the reserved prefix the operators
already use (`@po:unknown`), so it composes with every Polars operator and
function, and the reader below finds it by name. `rp.norm_cdf` is a node of
the library's own, `Phi(z) = erfc(-z / sqrt(2)) / 2`, accurate in both tails
(`1 - erf` loses the left tail); Polars' own `.erf()` and `.erfc()` are read
as the same nodes where the caller's Polars serializes them.

### A2. Where it runs

**Recommended: as a node of the formula grammar, so it runs inside
`rp.stream.with_windows`** (and, later, in a spec's formula target). On the
Rust side the node is rebuilt as one Polars expression: its inputs (the
columns, the `equals` and the bracket ends, each an element-wise expression
not holding the unknown) through Polars' public builders, and the solve as a
Rust function over those input columns, built with Polars' public
`map`-style builder. That is the existing streaming `LazyFrame` plan, under
the window engine's chunking, with no new interface.

**Not recommended: an expression plugin**, which would let `rp.invert` run
in a plain `with_columns`. The plugin was removed because a stateful
expression is handed its whole column; a stateless element-wise solve has no
such problem, but the plugin's `derive` feature turned on
`polars-plan/python` and with it `lazy` in the extension's build, and their
removal was measured as the larger gain. Re-adding it is the user's call
(question 1).

**Not recommended: `map_batches` with a Python function.** It is a Python
callback a batch, not a row, but it holds the GIL, cannot be serialized
with the plan, and is a second path to test.

Inside `with_windows` today "each formula must hold at least one operator";
`invert` counts as one, since it is the library's node, not Polars'.

### A3. How the function is given, and how it is serialized

The function is read exactly as a formula is: once, when the query is built,
from `f.meta.serialize(format="json")` under the caller's Polars, walked
into the compact tree. The unknown is the leaf `["unknown"]`. The node is

```text
["invert", f_tree, equals_tree, lo_tree, hi_tree,
 {"increasing": true, "xtol": 2e-12, "rtol": 8.881784197001252e-16, "detail": false}]
```

- **What `f` may hold.** Every subtree that holds the unknown is evaluated
  by the library's own scalar interpreter, a row at a time, so it may hold
  only the grammar's element-wise nodes (columns, literals, arithmetic,
  `**`, `neg`, `abs`, `exp`, `log`, `sqrt`, `clip`, `when`, comparisons)
  plus `norm_cdf`, `erf` and `erfc`. A subtree that does not hold the
  unknown is an input column; in the first build it too is held to the
  grammar, which keeps the whole function in the library's versioned form.
  A user with a more complex input computes it as a column first. Anything
  else is refused by name when the query is built, as for a formula.
- **Rule 13.** Polars' serialized form is read once, in the caller's
  process, and never stored; the compact tree is what a window run's state
  and a formula target's state hold. Adding the nodes `unknown`, `invert`,
  `norm_cdf`, `erf` and `erfc` is a change to the forms that tree can take,
  so it bumps the window state's version and the bank's schema (if a
  formula target may hold it) with fixtures that hold each new node (rule
  5). The stable, versioned serialized form that PLAN entry 577 asks for a
  formula target would cover these nodes with the rest.
- **The interpreter** walks the tree in Rust over `f64`, a node at a time,
  with no allocation a row: the tree is compiled once into a flat postfix
  program of slots.

### A4. The algorithm: ITP, stopped by a tolerance, with a stated bound

**Recommended: the ITP method** (Oliveira and Takahashi, "An enhancement of
the bisection method average performance preserving minmax optimality",
ACM TOMS 47(1), 2021). Each step interpolates (regula falsi), truncates the
interpolant toward the midpoint, and projects it into a band around the
midpoint whose radius shrinks so that the bracket after `j` steps is never
wider than bisection's after `j - n0` steps. Brent's method (scipy's
`brentq`) has no comparable worst case: its guarantee is of the order of the
square of bisection's step count.

- **Convergence is stated by the tolerance.** With
  `eps = xtol + rtol * min(|lo|, |hi|)` (and `xtol` alone when the bracket
  holds 0), the solve stops when the bracket `[a, b]` holding the sign
  change has `b - a <= 2 eps` (plus one rounding of each end), or when an
  evaluation is exactly `equals`. The answer is the bracket's midpoint, so
  it is within `eps` of a sign change of the computed function.
- **The bound is a consequence of the tolerance, not a cap.** ITP needs at
  most `n_half + n0` steps, `n_half = ceil(log2((hi - lo) / (2 eps)))`,
  each one evaluation, plus the two at the bracket's ends and the two of the
  certificate (A5). With `n0 = 1`, `kappa1 = 0.2 / (hi - lo)` and
  `kappa2 = 2` (the paper's recommendation), a bracket of `[1e-4, 5]` at
  `xtol = 2e-12` is at most 44 evaluations before the certificate. Since
  `xtol > 0` and `rtol >= 4 eps_machine`, `n_half` is finite for any finite
  bracket (at most about 2,100). The bound is computed per row; reaching it
  without the tolerance met is a defect, raised as an error naming the row,
  never a withheld value, and a test drives the solve over hundreds of
  thousands of brackets asserting it is never reached.
- **The bound holds in floating point only with the projection's radius
  rounded down.** A prototype of the paper's method in scipy's arithmetic
  (`scratchpad` script, not kept: 19,935 Black-76 calls, strikes
  `F exp(N(0, 0.25))`, expiries 1 day to 2 years, volatilities 0.05 to 1.5,
  bracket `[1e-4, 5]`, `xtol = 2e-12`) overran the bound by one step on 5
  rows, each where the bracket's width came out a few ulps above `2 eps`.
  With the radius rounded down by four ulps and one ulp of the bracket's
  ends, none overran: at most 44 evaluations, mean 12.9, median 11. scipy's
  `brentq` on the same rows: mean 10.3, median 9, at most 41. ITP spends
  about 25% more evaluations on average for a worst case that is stated
  (question 3).

### A5. The guards: what is withheld, and why

**A root is withheld, never clipped or guessed.** A value outside the
bracket, or the bracket's end, would be a number the function does not
produce at the target: a volatility for a price below the forward's
intrinsic value is not a small volatility, it is no volatility, and a
consumer reading the bracket's end would read a false one. So the output is
null, and under `detail=True` the reason says which:

| reason | when | why it is withheld |
|---|---|---|
| `input_missing` | `equals`, a bracket end or a column `f` reads is null, or not usable (NaN, infinite, past 1e100: the library's one rule for a value) | no function to invert on this row |
| `bracket_invalid` | `lo >= hi` | no interval to search |
| `below_range` / `above_range` | no sign change: `f(lo) - equals` and `f(hi) - equals` have one sign; named by which side the target lies, using the declared direction | no root in the bracket. With a monotone function, none anywhere in it |
| `wrong_direction` | a sign change against the declared direction | the function is not what the caller declared on this row; a root found would be of another function |
| `not_finite` | NaN at a bracket end or at a step | a domain error in the function (a log of a negative, `0/0`); an infinity is not this case: it has a sign, and the step it came from falls back to the midpoint |
| `not_monotone` | a step's value lies outside the values at the bracket's two ends | the declared monotonicity is broken inside the bracket, so the root may not be unique; checked at no extra evaluation |
| `not_determined` | the certificate fails (below) | the root's position is not fixed to the tolerance by the function's own rounding |

**The certificate.** The function's computed value carries rounding, and
where it is flat (a deep in-the-money option near expiry had a vega of
3e-15 in the prototype) its sign near the root is noise: any point within `eta / |f'|` of the
root is a root of the computed function, `eta` the rounding. The prototype
above found 138 of 19,935 rows where ITP and `brentq` disagreed by more than
both tolerances, by up to 0.04 in volatility, every one with a vega below
about 0.1 and each solver as far from the true value as the other. So the
interpreter carries a running error bound beside each value (Higham's running
error analysis: each operation adds its result's rounding, `u |result|`, to
the bounds propagated from its operands; a call to `exp`, `log` or `pow`
adds the platform's few ulps, `erf` and `erfc` the `libm` crate's), and a
root is reported only where the final bracket's ends have signs larger than
their bounds. Where an end does not, the two points `m - 2 eps` and
`m + 2 eps` around the midpoint `m` are evaluated, and if their signs are
certain the root is reported with that bracket; otherwise it is withheld as
`not_determined`. In the prototype (with a rough stand-in for the running
bound, `16 u (F Phi(d1) + K Phi(d2) + price)`) this withheld 291 rows,
1.5%, all 138 disagreements among them, and no row with a vega above 0.14.

**No zero-weight rule applies**: the solve learns nothing and has no clock;
a row's weight is a model's input, not the function's. The solve runs on
every row, as a row of weight 0 is still predicted.

### A6. Outputs

`Float64`: the root, or null where withheld. Under `detail=True`, a struct
with `root`, `lo` and `hi` (the certified bracket), `evaluations`
(`UInt32`), `bound` (the stated bound for the row) and `reason` (an `Enum`
of the table's names, null where a root is given). A test holds `bound >=
evaluations` on every row it produces.

### A7. State, chunk invariance, save and load

The solve holds no state: each row's answer is a function of that row's
values alone, so chunk invariance holds by construction and is tested
anyway, at 1, 7 and 1,000 chunks and one thread against eight, to the bit.
A window run that holds an `invert` node saves and loads its formula tree as
today, with the new node kinds (A3).

### A8. The math library

The interpreter calls the platform's `exp`, `ln`, `powf` and `sqrt`, which
are the functions Polars itself calls for the same expression, so the solve
and Polars evaluating `f` at the root agree on one machine to the bit.
`erf`, `erfc` and `norm_cdf` come from the `libm` crate: the same bits on
every OS, and the only choice, since std and Polars have none. Nothing goes
into a state. Across OSes a root may differ in its last bits, or by up to
the tolerance where a different last bit flips a step's side: the
cross-platform goldens are therefore on functions that call no platform
math (polynomials, `sqrt`, the `libm` crate's `erfc`), as the testing guide
asks of a value pinned to its bits.

### A9. Oracles and tests

- **scipy.optimize.brentq** on the same function written in numpy and
  `scipy.special.ndtr`, row by row: the two roots within
  `2 eps + xtol_s + rtol_s |x|` on every row the certificate passes, on
  realistic Black-76 and Bachelier grids (moneyness, expiries from a day to
  years, volatilities 1% to 300%) and a polynomial family.
- **The definition, from Polars:** the returned bracket's ends, substituted
  for the unknown and evaluated by Polars' own expression engine, have
  opposite signs, and the bracket is no wider than `4 eps` plus rounding.
  Polars computes `f` independently of the interpreter, so this also holds
  the interpreter to Polars node by node.
- **Rust (what `cargo mutants` sees):** the bound never reached, the
  certificate's signs, each guard's reason, an exact zero at an end and
  inside, infinite values at the ends, and the interpreter against direct
  Rust arithmetic on every node.
- **Degenerate cases beside the realistic ones:** a price below intrinsic
  and above the forward (no sign change), a flat function (the deep
  in-the-money call), a NaN inside the bracket (a log of a negative), a
  non-monotone function with a sign change, a bracket of one ulp, `lo = hi`.

### A10. Cost

To be measured by the build. An estimate: the Black-76 tree is about 30
nodes, two of them `erfc`; at 12 to 15 evaluations a row that is of the
order of a microsecond a row. A bench cell (`invert.black76`) and its budget
come with the build.

## Part B: decayed local-polynomial regression

### B1. The interface

A model kind in the bank, so it has the bank's groups, clocks, weights,
`min_weight`, `embargo`, save and load:

```python
rp.spec.local_poly(
    target,                     # one or more targets
    features=["k", "tau"],      # the features, d >= 1 (the entry asks for two or more)
    degree=1,                   # 0 (Nadaraya-Watson), 1 (local linear) or 2 (local quadratic)
    kernel="tricube",           # "tricube" | "epanechnikov" | "gaussian": product kernels
    bandwidth=[0.05, 0.25],     # one per feature, in its own units, above 0
    query=[[-0.2, 0.25], ...],  # the query points, Q x d; or rp.grid(k=[...], tau=[...])
    half_life=..., lam=...,     # the decay on the clock, as every model
    window_size=..., window_rows=..., snapshot_every=...,
    min_weight=...,             # read against each query point's local weight (B6)
    solve_every=..., coef_every=...,
    max_state_bytes=...,
)
```

### B2. The estimand: the weighted window each oracle checks

For a group at row `t`, over the rows `i < t` learned before it (weight
`w_i > 0`, target present, inside the window):

```text
omega_i  = w_i * lambda_i                 lambda_i the decay since row i on the clock (0 outside the window)
u_ij     = (x_ij - q_j) / h_j             j = 1..d
K_q(x_i) = prod_j k(u_ij)                 k(0) = 1
P(z)     = the monomials of z of degree <= p, m = C(d + p, p) of them, 1 first
beta(q)  = argmin_b  sum_i omega_i K_q(x_i) (y_i - P(x_i - q) b)^2
surface(q) = beta_0(q)                    the fit's value at q
gradient(q) = beta_1..d(q)                its slope in each feature (degree >= 1)
```

`k` is `(1 - |u|^3)^3` (tricube), `1 - u^2` (Epanechnikov), each 0 past
`|u| = 1`, or `exp(-u^2 / 2)` (Gaussian). With `omega = 1` and no window this
is the textbook local polynomial estimator (Fan and Gijbels, 1996) with a
product kernel; scaling `k(0)` to 1 changes no fit, and makes the local
weight below comparable to the shared one.

**The surface at row `t` is computed before row `t` is learned**, as every
prediction is (rule 2).

### B3. Kernel and bandwidth

- **Product kernels**, one bandwidth per feature, because features come
  in different units (moneyness and years), and because statsmodels'
  `KernelReg` is a product-kernel local linear estimator, which makes it an
  independent oracle (B9). LOESS's radial kernel on scaled distance is the
  alternative (question 8).
- **Tricube by default**: compact support (a row touches only the query
  points within one bandwidth, which is the cost's saving), and no libm call,
  so the kernel's weights in the state are the same bits on every OS.
  Epanechnikov likewise. Gaussian calls `exp`, so its weights are a libm
  result in the state (B11).
- **The bandwidth is fixed**, in each feature's units. A nearest-neighbor
  span (LOESS's `frac`) depends on every row's distance to the query point,
  and cross-validation on every row's leave-one-out fit: neither is one pass
  in bounded state. A bandwidth in units of each feature's running spread
  would change the weight of rows already summed; it is not offered
  (question 9).

### B4. Decay and window

The decay is every model's: `half_life` or `lam` on the clock, `gap_cap`,
sessions, steps back, and rule 9 for a zero-weight row. `window_size` (on
the clock) or `window_rows` (a count) cut the sums with the snapshot ring of
`window.rs`, each snapshot holding every query point's sums, spaced by
`snapshot_every`, thinned under `max_state_bytes` as the ring is today. A
window with no decay (`half_life = inf`) is the plain kernel fit on the
window's rows.

### B5. Query points

**Recommended for the first build: a fixed set of query points**, given at
the spec as a `Q x d` list or as the Cartesian product of per-axis lists
(`rp.grid`), the same for every group, part of the spec (a changed set is
another spec, refused on load like any changed parameter). Each point keeps
its own sums, so the state is bounded by `Q` and the shape, never by the
stream.

**The row's own point as a query** (the fit at this quote's moneyness and
tenor, its residual the quote's richness) needs the rows themselves: with
decay alone that is every row ever seen, so it is offered only under a
window, holding the window's rows (B10) and solving a local fit a row
(question 5). It would give the model a `pred`, and with it the residual
diagnostics; the fixed-point form predicts no target and refuses them, as
`ew_cov` does.

### B6. The update, the solve, and what is withheld

- **Per learned row**, for each query point with `K_q(x_i) > 0`: add
  `omega K` times the outer product of `P(x_i - q)` to the point's Gram, its
  products with each target, and the point's local weight. Every point
  decays by the row's factor, as every model's sums do.
- **Solved on `solve_every`**: a Cholesky of the point's Gram without the
  jittered-diagonal fallback, since a ridge would bias a local fit silently.
  `surface` and `gradient` are written on `coef_every`'s rows.
- **Withheld, per query point, with a reason**:
  - `below_min_weight`: the point's local weight `W_q = sum omega_i K_q(x_i)`
    is below `min_weight` (recommended default: `m`, the weight of `m` rows
    at the point's centre; question 10);
  - `not_identified`: the Gram is not positive definite, or the solve's
    normwise forward error bound `2 kappa eta / (1 - kappa eta)` (the bound
    the tests' `forward_bound` helper states, computed here in the model,
    with `kappa` the Gram's condition number after scaling to unit diagonal) is larger than a stated relative accuracy (question 7). Rows
    on a line through `q`, or all on one side of it under degree 2, fail
    this, which is the condition itself, not a count of rows standing in for
    it (PLAN entry 552's rule);
  - a feature or target that is not usable: the model trait's one rule
    (the row is learned at weight 0, a missing target learns nothing).

### B7. `n_eff` and the zero-weight row

- **`weight_sum`** is the shared weight `W = sum omega_i`, before the row's
  update and before its own decay, and the weight inside the window under
  one (rule 8): it is not kernel-weighted, so it is the same number every
  other model in a bank emits.
- **Each query point's `min_weight` gate reads its own `W_q`**, as each
  target's gate reads that target's own weight (rule 8, S2); `W_q` is
  emitted per point as `local_weight`.
- **A row of weight 0** is predicted (its surface is the state's) and
  handed to no model, so the next row decays as if it were not there, and it
  counts in no window or clock (rule 9). The first row of a stream at weight
  0 divides nothing: every update is a sum, and a point's solve is gated by
  `W_q`, so no 0/0 is formed.

### B8. Outputs

Per row, beside `weight_sum`: `surface_<target>` (`Array(Float64, Q)`, null
where withheld), `local_weight` (`Array(Float64, Q)`), and on request
`gradient_<target>` (`Array(Float64, Q * d)`) and `surface_reason`
(`Array(UInt8, Q)` codes). The arrays are written on `coef_every`'s rows and
null on the others, so an output frame at `Q = 100` is not 800 bytes a row
on every row.

### B9. Oracles and tests

- **statsmodels `WLS`** (and scikit-learn's `LinearRegression` with
  `sample_weight`, as the second opinion), at each query point, on the rows
  the window holds: design `P(x_i - q)`, weights `omega_i K_q(x_i)` with
  `omega_i` read off the same rows and stamps by pandas' `ewm(times=)`
  weights, as the temporal clock's tests already do. The surface is the
  intercept, the gradient the next `d` coefficients. Every kernel, every
  degree, decay and window.
- **statsmodels `KernelReg(reg_type="ll", ckertype="gaussian")`** at unit
  weights, no decay and no window: an independent local linear estimator.
  Checked here: on 400 rows in two features it equals `WLS` with the
  Gaussian product kernel to 1.1e-15 at two query points.
- **Not statsmodels' `tricube`**: its kernel sets `u = 0` where `|u| > 1`
  and so gives a row outside the support the kernel's *largest* weight
  (`statsmodels.nonparametric.kernels.tricube`, 0.15.0). Checked here: on
  the same 400 rows, `KernelReg`'s tricube fit is 0.20 from `WLS` with the
  tricube kernel. Compact kernels are held to `WLS` alone.
- **Rust:** each solve against its Gram built from the raw rows and solved
  by `crate::oracle::solve` (faer's LU), held to the derived bound.
- **Realistic and degenerate cases:** a smooth surface in two features
  (an SVI-like smile drifting in level) with noise, on an irregular clock
  with gaps and groups; and beside it a query outside the data, all rows on
  one line through a query point, a constant feature, one row, zero
  weights first and throughout, a window that empties.

### B10. State bytes, against `max_state_bytes`

With `d` features, degree `p`, `m = C(d + p, p)`, `T` targets and `Q`
query points, each point keeps

```text
a = m(m+1)/2   the Gram's upper triangle
  + m T        its products with the targets
  + T          sum omega K y^2, for a local residual scale
  + 2          W_q and sum (omega K)^2, the local weight and its Kish size
```

doubles. A group holds, in memory:

| form | bytes a group |
|---|---|
| fixed points, decay only | `8 Q a` + about 64 for the shared weight and clock |
| fixed points under a window | `8 Q a (1 + S)` + 24 a snapshot for its stamps, `S` the snapshots held: at most `window / snapshot_every + 2`, fewer when thinned |
| the row's own point (B5, if built) | `8 R (d + T + 2)`, `R` the rows inside the window |

and `G` groups `G` times that. The query points and bandwidths are the
spec's, held once. A saved state writes a double as 9 bytes of msgpack.
Examples (`T = 1`, a 10 x 10 grid unless named):

| features, degree | `m` | `a` | decay only | window, `S = 10` | 1,000 groups, decay only |
|---|---|---|---|---|---|
| 2, linear | 3 | 12 | 9.6 KB | 106 KB | 9.6 MB |
| 2, quadratic | 6 | 30 | 24 KB | 264 KB: over the 256 KB default | 24 MB |
| 3, linear, 5 x 5 x 5 | 4 | 17 | 17 KB | 187 KB | 17 MB |
| 3, quadratic, 5 x 5 x 5 | 10 | 68 | 68 KB | 748 KB | 68 MB |
| the row's own point, 2 features | | | | 40 B a row: 6,553 rows in 256 KB | |

For fixed points every term is known when the spec is built, so a spec over
its `max_state_bytes` is refused then, naming the bytes and what to lower
(`Q`, `degree`, `snapshot_every`). The row form is checked as its rows come,
as `expectile`'s held rows are. Nothing grows with the stream's length.

### B11. Chunk invariance, save and load, the math library

- **Chunk invariance**: the rows are learned one at a time in input order,
  each point's sums in a fixed order, the solves on the clock's schedule; so
  any chunking gives the same bits, held by the bank's 1, 7 and 100 chunk
  tests and `tests/test_every_kind.py`.
- **Save and load to the bit**: a new `ModelState` variant (`LocalPoly`, its
  name frozen once released), a schema bump, a loader for the previous
  schema (which holds no such variant), and fixtures freezing every form:
  each kernel and degree, decay only and under a window with snapshots held,
  two targets, a point withheld. If the row form is built, a fixture with
  rows held.
- **libm**: the decay factor is every model's (`exp2`, already in every
  state). The Gaussian kernel's `exp` puts a libm result in the state, so
  its fixtures compare those sums across OSes within the smallest tolerance
  that passes, stated in ulps beside the test (the libm exception);
  tricube and Epanechnikov call nothing and stay bit-exact across OSes.

### B12. Cost

Per learned row: `d` kernel factors for each of `Q` points, then a rank-one
update of `a` doubles for each point inside the support; a decay of `8 Q a`
bytes. At a 10 x 10 grid, `d = 2`, degree 1, and a bandwidth covering a
quarter of the grid, that is about 25 updates of 12 doubles and 1,200
multiplies of decay, of the order of a microsecond. The decay of every point
every row dominates at large `Q`; a lazily applied common scale removes it
but is another layout, so the build measures both before choosing. A bench
cell (`local_poly.history`, and one for `Q`) and their budgets come with the
build.

## 2. The build, in slices

1. **(A) the solver in `rheon-core`** (`root.rs`: ITP over a closure, the
   bound, the certificate's logic; no Polars), with its Rust tests.
2. **(A) the interpreter and the node** in `rheon-polars` (a new
   `formula/scalar.rs`: `formula.rs` and `windows_frame.rs` are near the
   source cap), the Python names, the brentq and Polars oracles, the window
   state's version bump and fixtures.
3. **(B) the model** through `docs/EXTENDING.md`'s seventeen steps, fixed
   query points only.
4. **(B) the row form**, if the user wants it (question 5).

The two blocks share nothing but the record; they can be built and merged
apart. Several building blocks bump the schema at once; each rebases its
fixture sets onto the last one merged.

## 3. Open questions, each with a recommendation

1. **Where the solve runs.** Recommended: a node of the formula grammar,
   inside `with_windows` (A2), with no new interface. Not an expression
   plugin, which would bring back the build features removed with the old
   plugin. Ask: is `with_windows` alone enough, or is a plain `with_columns`
   form wanted at that cost?
2. **`rp.invert` inside a spec's formula target.** Recommended: not in the
   first build; a user computes the root as a column first. It would bump
   the bank's schema for a form no one has asked for yet.
3. **ITP or Brent.** Recommended: ITP, for its stated worst case (bisection's
   step count plus one); it spent about 25% more evaluations than `brentq`
   on the prototype's Black-76 rows. Brent's method would match the oracle's
   algorithm and be faster on average, with a worst case of the order of the
   square of bisection's.
4. **The certificate (`not_determined`).** Recommended: build it, with the
   interpreter's running error bound, and withhold where it fails; it
   withheld 1.5% of the prototype's rows, every disagreement with `brentq`
   among them. The alternative is to report every root and give the
   certified bracket's width under `detail=True` only.
5. **The row's own point as a query** (a `pred` and a residual for each
   quote). Recommended: a second slice, only under a window, after the
   fixed points have landed and been measured.
6. **Declaring the direction.** Recommended: `increasing` required, so the
   reasons can say which side of the range a target lies on
   (`below_range`, `above_range`) and a wrong direction is caught. The
   alternative is `increasing=None`, accepting any sign change.
7. **The identifiability bound for a local fit.** Recommended: withhold a
   point where the solve's forward error bound passes `1e-8`, relative, so
   a surface value is never reported with fewer than about 8 correct
   digits from the arithmetic alone; the user may prefer another number,
   or Belsley's condition index of 30 that the Gram guide reads, which is a
   statistical rather than an arithmetic threshold.
8. **Product or radial kernel.** Recommended: product, with a bandwidth per
   feature; it has a third-party oracle (`KernelReg`). Radial (LOESS's)
   is a later option.
9. **The bandwidth's units.** Recommended: fixed, in each feature's own
   units. A bandwidth scaled by each feature's running spread cannot be
   summed in one pass.
10. **`min_weight`'s default for a local fit.** Recommended: `m`, the
    number of coefficients, in kernel-centre rows, read against each
    point's own `W_q`. This is a weight, not a row count, and the
    identifiability check (question 7) is the condition that decides
    whether the fit is determined.
11. **Names.** Recommended: `rp.invert`, `rp.unknown`, `rp.norm_cdf`,
    `rp.spec.local_poly`, `rp.grid`; the output fields `surface_<target>`,
    `gradient_<target>`, `local_weight`, `surface_reason`.
12. **`erf` and `norm_cdf` from the `libm` crate.** Recommended: yes; it is
    already linked, gives the same bits on every OS, and neither std nor
    Polars 0.55.2 has an `erf`. No new static linking (rule 12).

## 4. Slice 1 as built (part 2a): the solver in `rheon-core`

Built 2026-11-23 on this branch, under the coordinator's rulings on the
questions above (1, 2, 4, 5, 8, 9 and 11 as recommended; ITP held to
`brentq`; `increasing` required; the forward-error threshold a parameter;
identifiability by the Gram's condition; `norm_cdf` behind one function on
the `libm` crate until the shared special functions land). Code:
`crates/rheon-core/src/root.rs`, its tests in `root/tests.rs` with a
bisection oracle of their own (not in `oracle.rs`: a module that names
`crate::oracle` must be listed in the CPU-paths workflow's SDE filters,
a CI file this build leaves to the coordinator), and the example
`crates/rheon-core/examples/root_black76` that `tests/test_second_opinion_root.py`
holds to scipy's `brentq`.

Decisions taken in the build, beside the record:

- **The core API** is `find_root(f, lo, hi, increasing, &RootTolerance)`,
  `f` a closure returning `Bounded { value, error }`: the certificate
  needs each value's rounding bound, so a closure on `f64` alone could not
  carry it. The interpreter of slice 2 supplies the bound by its running
  error analysis; a caller with an exact function passes `Bounded::exact`.
  The core solves `f(x) = 0`; `equals` is the caller's subtraction.
- **The stop allows the ends' rounding.** A cubic on `[-3, 4]` ran its
  `n_max` steps and stopped a few units in the last place of its ends
  wider than `2 eps`: rounding the radius down covers the projection, not
  the midpoint's own rounding. The stop is therefore `b - a <= 2 eps + 8
  eps_mach max(|a|, |b|)`, or adjacent doubles, and the answer is within
  `eps + 5 eps_mach |root|` of a sign change. 5,000,000 random brackets
  (roots, widths and tolerances across the doubles' range; six monotone
  families, a step among them) never reached the bound; on the step
  family the solve comes within `n0` of it.
- **`bound_reached` is a reason** in the core, as the coordinator's
  contract lists it, documented as a defect for the caller to raise (A4).
  A test reaches it by handing the inner solve a smaller `n_max`.
- **`input_missing`** in the core is a NaN or infinite bracket end;
  `bracket_invalid` is `lo >= hi` or a width that overflows.
- **Range and direction verdicts need certain signs.** `below_range`,
  `above_range` and `wrong_direction` are given only where the ends'
  signs (or, for the direction, their order) exceed their error bounds;
  otherwise `not_determined`. A deep in-the-money quote whose time value is
  below the price's rounding was read as `below_range` before this, a
  false claim that no volatility gives the price.
- **The certificate's points** are `m -+ 2 eps`, each at least one double
  beyond the final bracket and clamped to `[lo, hi]` (an end's value is
  reused, not evaluated). An exact zero at an end with a nonzero bound is
  withheld: the root may lie outside the bracket.
- **Underflow in the running bound.** The tests' Black-76 bound adds
  `2^-1074` to every operation (Higham's `eta`): `erfc` far in its tail
  returns 0, and without the term a deep out-of-the-money price was 0 with
  a zero bound, an exact root the function does not have. Slice 2's
  interpreter needs the same term.
- **The grid's moneyness scales with the volatility** (`ln(K/F)` within
  `±3 sigma sqrt(tau)`): at a fixed `±0.8`, a fifth of the rows had a time
  value below the price's rounding, which no solver can invert. Measured:
  the Rust grid withholds 441 of 20,000; the scipy grid certifies 3,850 of
  4,000, within 2.08e-12 of `brentq` (bound 6e-12), at a mean of 18
  evaluations to `brentq`'s 12.4, at most 46 (the bound at `[1e-4, 5]`,
  reached).
- **The example binary is left out of the `oracle` mark**, as the command
  line's tests are (`tests/oracle_marks.py`): a CPU-paths leg does not
  build it, and the Rust solver's numbers move with no CPU path.

## 5. Slice 2, first cut (part 2b): the node in `rheon-polars`

Built 2026-11-24 on this branch. Code: `crates/rheon-polars/src/formula/scalar.rs`
(`Invert`, exported from the crate), its tests in `formula/scalar/tests.rs`,
and the example `crates/rheon-polars/examples/invert_rows`, which
`tests/test_second_opinion_invert.py` holds to scipy's `brentq` on a seeded
monotone closed form (`x^3 + a x + exp(b x)`, 1,000 rows, every in-range row
certified and within `2 eps + xtol + rtol |x|`).

- **Not yet a node of the formula grammar.** `Invert` reads its own list
  form, `["invert", f, equals, lo, hi, {params}]` with the leaf
  `["unknown"]`, and is not read by `Node::from_json`, so a window run's
  state cannot hold it and the window state's version and fixtures are
  untouched. Wiring it into `Node` (and so into `with_windows`) is the next
  cut, with the version bump and fixtures that holding the new forms needs.
- **The interpreter's nodes** are `col`, numeric `lit`, `unknown`,
  `+ - * / **`, `neg abs exp sqrt`, `log` with a base and a cast to
  `Float64`. `clip`, `when`, comparisons, `norm_cdf`, `erf` and `erfc` are
  refused by name for now.
- **All four parts run in the interpreter**, the target and the bracket
  included, so the target's own rounding enters the certificate's bound.
  Each operation's running bound adds `u |v|` and `2^-1074`; `exp`, `ln` and
  `powf` add two ulps of their result besides.
- **As a Polars expression** the node is `map_multiple` over the columns it
  reads (element-wise, so a streaming plan may hand it any slice), giving
  `Float64` or, under `detail`, the struct of A6 with `reason` as a
  `String` (an `Enum` waits for the grammar's cut). A solve that reaches its
  bound raises an error naming the row's values.
- **Deferred:** `rp.invert`, `rp.unknown`, `rp.norm_cdf` in Python; the
  Polars-definition oracle (A9); the bench cell (A10).
