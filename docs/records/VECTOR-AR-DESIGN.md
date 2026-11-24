<!-- kind: record -->
# The vector autoregression: design and the decisions it took

Status (2026-11-24): built as `vector_ar` (`crates/rheon-core/src/vector_ar.rs`
and its `fit`, `reads` and `tails` modules), from a numpy prototype measured
against statsmodels 0.15.0 and a design note answered by the coordinator.
This records what was built, the alternatives weighed and the measurements
behind the defaults.

## The model

A `k`-variate VAR(p) with a constant and optional exogenous features `x`
(`e` of them), fitted by exponentially weighted least squares on lags the
model keeps:

```text
y_t = c + B_x x_t + A_1 y_{t−1} + … + A_p y_{t−p} + u_t
```

The equations share their inputs `z_t = [x_t; y_{t−1}; …; y_{t−p}]`
(`d = e + kp`, features first, then the lags lag by lag, statsmodels'
`params` order), so least squares equation by equation is the system's
estimate. With that order, VAR(q) for any `q ≤ p` is the leading block of
VAR(p), which the order choice reads.

## The carrier: one `ewridge` over `[z, y]`

The fit is one `EwRidge` with `d + k` features and the `k` targets, its
solve restricted to the first `d` (a feature set). Its running means then
hold `S_zz`, `S_zy` and `S_yy` in one centered Gram, the targets'
co-moments included, which the multi-equation tests, `Σ_u`, the order
criteria and the impulse responses need and which `ewridge`'s per-target
moments alone do not keep. The carrier gives, already tested: decay on the
clock, the snapshot window with its budget, edge and cadence, Kish's sums
(inside the window too), and the solve schedule. Its price is `2(d+1)k + k²`
more numbers in the Gram than `d + 1` slots alone would hold.

**Alternatives weighed.** A second, windowed `EwCov` of the targets beside
the carrier would hold `S_yy`, at the cost of a second snapshot ring and
budget kept in step with the first; the prototype's own row store with
down-dating would be smaller under a window but brings a second window
mechanism and rounding that is never washed out (3.2e-10 at a level of 1e4
after 1e6 rows). The coordinator chose the snapshot window.

## Rows, the lag ring and the clocks

A learned row has a positive weight and every target present; only it
enters the ring and, once the ring is full, the fit. A row of weight 0
learns nothing and moves no ring. A row with a target null (an incomplete
row) teaches nothing and moves no ring, ages the fit by its clock step, and
is predicted wherever its lags are complete: statsmodels'
`VAR(missing="drop")`, which the tests hold exactly. (The prototype broke
the chain at such a row instead; the coordinator chose `ar`'s rule.) Lags
count learned rows; the half-life and the window are on the clock. A
session change or a capped gap empties the ring and keeps the fit.
`embargo` is refused, as for `ar`.

## The solve schedule and what a solve reads

The coefficients are the carrier's, solved on `ewridge`'s schedule and
defaults (by weight share under a finite half-life, every row under
`half_life = inf` or `lam`). The carrier counts its solves (a count kept out
of the state); on each step that solved, the model reads the moments once:
one Cholesky factor of `S_zz` plus the ridge, stopping at the first pivot at
or below `1e-10` of its diagonal. From it:

- **identification** (all pivots of the order's block passed): the fit is
  withheld as `not_identified` until it holds; outside a window it stays
  once passed, under a window it is read again at every solve;
- **the order** under `select`, by `ln det Σ_mle(q)` plus the criterion's
  penalty over the identified orders, the first of equal values (`libm`'s
  logarithms, so the choice is the same on every platform);
- **each test's Wald statistic** `λ = df · tr(SSE_E⁻¹ βᵀ G⁻¹ β)`;
- **the coefficients' variances**, statsmodels' `bse²`, under
  `emit_se_coef`.

These are kept in the state beside the fit and read by every row until the
next solve, as the coefficients are. A per-row read would cost one factor a
row; the schedule's default, under a finite half-life about one solve every
`half_life / 50`, amortizes it. No value kept is a logarithm or a tail: the
p-values are formed for output only.

## Inference under decay: Kish's count

With `n = W² / Σw²` over the learned rows, `df = n − (d + 1)`,
`Σ_u = SSE · n / df` and `Cov(b) = (S_zz⁻¹ / n) ⊗ Σ_u`, which are
statsmodels' with no decay and unit weights. The prototype measured
`SSE / df` alone at half the noise at steady state. The tests and the
standard errors are null while `df ≤ 0`, the estimate's own finite-sample
condition rather than a count of rows.

Calibration of the built model (`tests/test_vector_ar.py`, extended tier): a
VAR(1) of two series with unit noise and no causation, read at the end of
each of 1,000 streams of 2,000 rows:

| `λ` | Kish's `n` | `f` at 5% | `ssr_f` at 5% | mean `Σ_u[0,0]` |
|---|---|---|---|---|
| 0.99 | ~199 | 0.044 | 0.044 | 1.002 |
| 0.95 | ~39 | 0.036 | 0.031 | 1.001 |

The prototype measured 0.046 and 0.035 on the same design with its own
streams.

## The tails

No model reported a p-value before. The F tail is the regularized
incomplete beta, `I_x(d₂/2, d₁/2)` at `x = d₂ / (d₂ + d₁ f)`, by its
continued fraction (Lentz) on the side that keeps a small value relative;
the χ² tail the regularized upper incomplete gamma, by its series below
`x = a + 1` and its continued fraction above. Both are built on the `libm`
crate's `lgamma`, `exp`, `ln` and `ln_1p`, so a p-value is the same bits on
every platform. Rust tests hold them to closed forms (χ²(2), integer shapes,
F with two degrees of freedom); the Python tests hold every emitted p-value
to scipy's `f.sf` and `chi2.sf` at the emitted statistic within 1e-10.

## The reads

`ModelBank.vector_ar` computes, from the moments as they stand: `Φ_i` and
`Θ_i = Φ_i P`; their standard errors, Lütkepohl's 3.7.5 for `Φ_i` and 3.7.8
for `Θ_i` (the second term through `H = ∂vec P / ∂vech Σ_u`, formed from the
Cholesky factor's derivative `dP = P Φ(P⁻¹ dΣ P⁻ᵀ)` rather than
statsmodels' elimination and commutation matrices, and only the diagonal of
each covariance, by the Kronecker structure); the FEVD; the companion
matrix's eigenvalues (faer); the long-run mean; every order's criteria; and
a test of any two sets. All are held to statsmodels' `irf`, `fevd`, `roots`,
`mean`, `select_order` and `test_causality` with no decay, and to
`VARProcess` of the fit's own parameters under decay.

## The state

Per group: the ring, `8kp` bytes; the carrier over `K = d + k + 1` slots,
at most `8 (4K² + 17K + 64) + 8 (k − 1)(4K + 16)`; the last solve's read,
at most `8 (2k(d + 1) + T + 10)` for `T` tests; and 2 KiB of field names.
Measured (named msgpack, after 400 rows, `select` and the variances on):

| k | p | K | bytes written | bound |
|---|---|---|---|---|
| 1 | 1 | 3 | 1,709 | 3,376 |
| 2 | 5 | 13 | 4,972 | 10,792 |
| 5 | 5 | 31 | 20,337 | 44,368 |
| 10 | 5 | 61 | 71,972 | 157,288 |
| 10 | 7 | 81 | 109,453 | 260,008 |

The default 256 KB cap takes up to 41 lags of 2 series, 15 of 5 and 7 of 10;
ten series at ten lags need 462,088 bytes and are refused at the build,
naming the bytes. A snapshot of the window holds about `K²/2 + kK` numbers.

## Left out

Instantaneous causality (statsmodels' own fails for sets), Monte Carlo
error bands, multi-step forecasts and a `ct` trend, which `ar` lacks too.
