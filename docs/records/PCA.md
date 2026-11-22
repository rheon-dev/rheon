<!-- kind: record -->
# `pca`: streaming principal components (PLAN 505)

The design note of `pca` (`crates/rheon-core/src/streampca.rs`,
`rheon.spec.pca`): what was built, the alternatives weighed, and the
measurements behind the defaults. The study it starts from is
`docs/records/STUDY-505-502-PCA-KRLS.md` on `vm/study-505` (part 1, the
block incremental SVD against scikit-learn), and the oracle-library scan's
section 11 (`vm-tools/research/494-MODELS-SCAN.md`).

## A new kind, not an `ew_cov` option

The scan found `ew_cov(pca=)` already the exact PCA of the decayed or
windowed covariance, in `O(k^2)` state, and that what 505 adds is the
`O(q k)` form for inputs too wide for a Gram. That form keeps no Gram, so as
an option of `ew_cov` it would have had to drop the accumulator every other
`ew_cov` output reads (means, variances, correlations, `mahal`, the window)
and refuse them all. A kind of its own keeps both models whole: `ew_cov`'s
components stay the exact, windowed form, and `pca` is the bounded one.
**For the coordinator's review**: the scan's verdict could be read as "an
option of `ew_cov`".

## The three recursions

All three keep the mean form every moment model keeps (`EwDiag`: the
weight, the mean as its pairs, the variances whose sum is the trace) and
`q = min(n_components + 1, k)` directions with the variance along each.

- **`svd`** (default): the incremental SVD of the study, one row a block.
  The stack `[sqrt(a) diag(s) V; sqrt(ab) d']` is rotated in pairs
  (Hestenes' one-sided Jacobi) until its rows are orthogonal, the `q`
  longest kept. Only `+ - * /` and `sqrt`, in a fixed order, so the
  directions are the same bits on every platform; orthogonality is made
  afresh each row, so it does not drift (`the_svd_directions_stay_orthonormal`:
  1e-14 after 20,000 rows). Rejected: Brand's small-core form (an
  `(q+1)^2` SVD and a rotation of the basis), cheaper for large `k` but its
  basis loses orthogonality row by row and needs a periodic Gram-Schmidt;
  faer's SVD, whose run-time SIMD dispatch can part the bits across CPUs.
- **`ccipca`**: Weng, Zhang and Hwang (2003), in the mean form, with the
  library's decay as its forgetting. The amnesic parameter is not offered:
  it is a second forgetting beside `half_life`, and no oracle computes it.
- **`oja`**: Oja's rule for `q` directions as the stochastic orthogonal
  iteration, `V' = GramSchmidt(V + eta (V d) d')`, with `eta =
  learning_rate * b / tr'`: the row's mean-form share of the weight over the
  total variance, so the rate is scale-free, `1/n` without decay and
  `1 - lambda` in steady state under it. The Gram-Schmidt keeps it stable at
  any gain, where the plain Hebbian form can blow up.

A direction not yet started takes the residual of the first deviation the
started ones leave, above `k` roundings of `|d|`. That is CCIPCA's own
start (`v_i(i) = u_i(i)`), put in the mean form's units.

## The gate: no count of rows

The user's rule of 2026-11-22: no fixed row count against early data. A
component is written while the likelihood-ratio test of the equality of
two eigenvalues (Anderson 1963), `(n - 1) ln(((l + m)/2)^2 / (l m))` at
Kish's `n`, passes 9.2103 (chi-squared, 2 df, at 1%) against both
neighbours. A neighbour of no variance is told apart once `n` exceeds the
rows that could leave it at none. The extra kept direction gives the last
emitted component its neighbour. `min_weight` defaults to 0. The rows a
component is withheld on read `withheld_reason = "not_separated"`, a new
seventh category of the shared enum.

North, Bell, Cahalan and Moeng's rule of thumb (a gap of one standard error,
`l sqrt(2/n)`) was tried first and rejected on measurement: on an
isotropic three-column stream it told the sorted sample eigenvalues apart
on 3,848 of 5,000 rows, since sorted sample eigenvalues of a tie repel by
more than one standard error. The test above read none of the 5,000 as
separated. On a stream with variances 100, 1 and 0.5 it first wrote the
leading component on the fifth row. The decision uses the `libm` crate's
`log`, so a state gives the same decision on every platform.

## Oja's gain

Measured over 6 seeds of 20,000 rows, six columns, two components, the
worst seed's largest principal angle against weighted PCA:

| spectrum | half-life | gain 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| 9, 4, 1, ... | inf | 0.69 | 0.30 | 0.048 | 0.0045 | 0.0069 | 0.012 |
| 9, 4, 1, ... | 2,000 | 0.44 | 0.10 | 0.0088 | 0.0082 | 0.019 | 0.033 |
| 9, 4, 1, ... | 200 | 0.040 | 0.033 | 0.021 | 0.022 | 0.045 | 0.069 |
| 9, 7, 5, ... | inf | 0.89 | 0.26 | 0.11 | 0.099 | 0.019 | 0.021 |
| 9, 7, 5, ... | 2,000 | 0.40 | 0.15 | 0.090 | 0.042 | 0.015 | 0.041 |

The theory's condition for a `c/t` rate, a gain above about `tr / (2 gap)`,
says a closer spectrum wants a larger gain; 4 is the default, the best or
near it on the separated spectrum and not past it on the close one.

## The oracles and their gaps

- `IncrementalPCA.partial_fit`, a first batch of `q` rows then one a time,
  no decay: 7.0e-13 rad, 5.4e-13 relative in `explained_variance_` and the
  ratio (tolerance 1e-11).
- `PCA(svd_solver="full")` on `[Y; -Y]` of the decay-weighted rows and
  `scipy.linalg.eigh` of `np.cov(aweights=)`, every direction kept: 1.4e-14
  rad, 3.3e-13 (a row of weight 0 counts in no clock: each row's age is the
  rows of weight after it).
- CCIPCA and Oja against their definitions in numpy: to rounding.
- Their convergence to PCA's subspace, ten seeds of 20,000 rows: CCIPCA
  0.0032 (no decay) and 0.0029 (half-life 2,000); Oja 0.0045 and 0.0082.
- In Rust, faer's self-adjoint eigensolver on the covariance written out
  longhand (`crate::oracle`, which this model does not call): 1e-12.

## State and cost

`8 (2 + 3k + q k + q)` bytes, fixed by the spec, checked at the call
against `max_state_bytes` (256 KB default): 2,000 features and 12
components, 256,120 bytes, against `ew_cov`'s 16 MB covariance. A row
costs a few sweeps of `O(q^2 k)` under `svd`, `O(q k)` under `ccipca`,
`O(q^2 k)` under `oja` (`crates/rheon-bench`'s `pca.*` cells).

## Not built

- A window: none of the three can remove a row. `ew_cov(pca=,
  window_size=)` is the exact windowed form.
- Whitening and the reconstruction error (`IncrementalPCA.transform` with
  `whiten`, `inverse_transform`): read off the emitted fields, not asked for.
- Closed groups: a `pca` group that closes reports no components there.
