<!-- kind: record -->
# Study 505 / 502: block incremental PCA, and kernel RLS

A Python-only numpy study; nothing here changes the package. The script is
`scripts/study_505_502.py` (`uv run python scripts/study_505_502.py pca`, about
14 s); every input comes from seeded generators and no data file is read or
written. Part 1 (task 505, block incremental PCA) is below. Part 2 (task 502,
kernel RLS) and part 3 are not written yet.

**Status:** part 1 done; parts 2 and 3 missing.

Notation: `k` columns, `n` the components compared (1, 2, 5 for k = 2, 5, 20),
`r` the rows of a truncated basis (`n + 2` capped at `k`, or `k`), `b` the
block's rows (200 throughout), `lam` the decay per row, `w` the row weight.

## Part 1: block incremental PCA (task 505)

### The question

The scan (`vm-tools/research/494-MODELS-SCAN.md` on `vm-tools/oracle-gap-scan`,
section 11) says `ew_cov(pca=)` already is exact PCA of the decayed or windowed
covariance, in `O(k^2)` state, and that what 505 could add is the `O(r k)` form
(the block incremental SVD of Ross et al. 2008, which scikit-learn's
`IncrementalPCA` runs) for inputs too wide for a Gram. This part builds both
forms in numpy, in `ew_cov`'s mean form, and holds them to scikit-learn.

### The math

**Per row** (`ew_cov`, `crates/rheon-core/src/ewcov.rs`), mean form, centred:

```text
W'  = lam W + w,      a = lam W / W',   beta = w / W'   (a + beta = 1)
m'  = m + beta (x - m)
C'  = a C + a beta (x - m)(x - m)'
```

`C` is the weighted covariance with the running mean, `C = (1/W) sum_i u_i
(x_i - m)(x_i - m)'`, `u_i = w_i lam^(t-i)` each row's weight now (the biased
form; the uncentred moment is `C + m m'`, not used here).

**Per block** of `b` rows (Chan's pairwise combination with a decay). Decay
each row to the end of its block, `u_i = w_i lam^(b-1-i)`, and form the block's
weight `U = sum u_i`, mean `m_B = sum u_i x_i / U` and covariance
`C_B = (1/U) sum u_i (x_i - m_B)(x_i - m_B)'`. Then

```text
W'  = lam^b W + U,    a = lam^b W / W',   beta = U / W'
m'  = m + beta (m_B - m)
C'  = a C + beta C_B + a beta (m - m_B)(m - m_B)'
```

which is the per-row recursion applied `b` times, up to rounding (measured
below: 7.4e-15 relative). An all-zero block (`U = 0`) skips `m_B` and `C_B`
(0/0) and sets `W' = lam^b W` only; at `W = 0` it leaves everything at 0.

**Read**: `eigh(C)`, sorted descending; the top `n` eigenvectors are the
components, `lambda_j` the explained variance (mean form, biased), and
`lambda_j / trace(C)` the explained-variance ratio.

**Truncated form** (the incremental SVD). Keep `C ~ V' diag(s^2) V` with `V`
the top `r` directions (`r x k`, orthonormal rows) and `s` their singular values
in the mean form, plus `tr = trace(C)` exactly (one scalar). A block stacks

```text
S = [ sqrt(a) diag(s) V                       ]   r rows
    [ sqrt(beta u_i / U) (x_i - m_B)          ]   b rows
    [ sqrt(a beta) (m - m_B)                  ]   1 row
```

so that `S'S` is exactly the `C'` above with `C` replaced by `V' diag(s^2) V`.
The thin SVD `S = P diag(s') V'` gives the new `V`, `s` (keep the top `r`);
`tr' = a tr + beta trace(C_B) + a beta |m - m_B|^2`; `m` and `W` as above. At
`lam = 1`, `w = 1` this is `IncrementalPCA.partial_fit` divided through by `W`
(scikit-learn's stack is `[diag(S) V; X_B - mean_B; sqrt(n_old n_B / n) (mean -
mean_B)]` in the sum form). It is exact when `r = k` (or the data's rank is at
most `r`); for `r < k` it drops, at every block, the mass outside the top `r`,
and never gets it back.

**State** (mean form, f64):

| form | fields | bytes |
|---|---|---|
| exact | `W`, `m` (k), `C` (packed lower triangle) | `8 (1 + k + k(k+1)/2)` |
| truncated | `W`, `m` (k), `V` (r k), `s` (r), `tr` | `8 (2 + k + r k + r)` |
| either, while a block is held | the rows, `lam`, `w` | `+ 8 b (k + 2)` |
| exact, windowed | plus the window's rows and weights | `+ 8 W_rows (k + 1)` |

The truncated form has no window: an SVD cannot give back a row it has folded
in. Both are bounded by the shape (and the window), never by the stream.

**Per-block cost** (flops, leading terms):

| step | flops | per row |
|---|---|---|
| exact fold | `2 b k^2` (the block scatter; half of it by symmetry) `+ O(k^2)` | `2 k^2` |
| exact read | `eigh`: about `10 k^3` for all pairs; `O(r k^2)` for the top `r` by a partial solver | per read, not per row |
| truncated fold | thin SVD of `(r+b+1) x k`: about `4 k (r+b+1)^2 + 8 (r+b+1)^3` | `~4 k (r+b)^2 / b`, smallest near `b ~ r`: `~16 r k` |
| truncated read | none: `V` and `s` are the answer | |

On flops per row the truncated fold wins once `k` passes about `8 r`; on state
once `k` passes about `2 r`. Measured below on this VM: at `k = 2000, r = 12,
b = 200` the truncated fold took 153 ms a block against the exact fold's 82 ms
(LAPACK's SVD runs nearer its peak less than the GEMM does), but the exact read
took 483 ms and its state is 16.0 MB against 208 kB.

### The oracles

- **`sklearn.decomposition.IncrementalPCA.partial_fit`** on the same 200-row
  batches, at `lam = 1` with unit weights, the only case it covers (it has no
  forgetting and no weights). `n_components=None` (all `k`) for the exact
  form; `n_components=r` for the truncated form at the same `r`.
- **`sklearn.decomposition.PCA(svd_solver='full')`** on the decay-weighted rows.
  PCA takes no sample weights and centres by the unweighted mean, so it is
  given `Y = sqrt(u_i / U) (x_i - m_u)` stacked with `-Y`: the stack's
  unweighted mean is exactly 0, so PCA's centring changes nothing, and its
  scatter is `2 Y'Y = 2 C`, so its components and ratios are those of `C`.
  Rows at zero weight are dropped first. No randomized solver is used.
- **`scipy.linalg.eigh`** of `np.cov(x, aweights=u, bias=True)`, a second
  library path to the same `C`.

Comparisons: the largest principal angle between the top-`n` spaces
(`scipy.linalg.subspace_angles`, radians); components matched by the largest
absolute inner product and reported as the worst `1 - |cos|`; the largest gap
in the explained-variance ratio of matched components.

Data: `k` columns with a geometric spectrum `9 * 0.6^j`, a mean offset near 5,
and a regime change half way (a new random rotation and mean), so the decay
has something to forget. Decay runs draw weights from U(0.5, 1.5) with one row
in ten at zero. Grid: `k` in {2, 5, 20}, `lam` in {1, 0.999, 0.99}, `N` in
{2000, 20000}, seeds 0 to 4; each cell is the worst over the seeds.

### Results

**Exact form.** Against `IncrementalPCA` at `lam = 1`: largest angle 9.1e-14
rad, ratio gap 6.9e-15. Against PCA on the weighted rows under decay
(`lam < 1`): largest angle 1.6e-14 rad, ratio gap 1.9e-15. Rounding, at every
`k`, `lam` and `N`.

**Truncated form.** Against `IncrementalPCA(n_components=r)` on the same
batches: largest angle 9.5e-14 rad, ratio gap 8.4e-15, so the decayed,
weighted form reduces to scikit-learn's exactly at `lam = 1`. At `r = k` it
matches PCA on the weighted rows under decay to 2.4e-14 rad (ratio gap
3.2e-15). At `r = n + 2 < k` it is an approximation, and a coarse one on these
streams: up to 0.46 rad against weighted PCA, and scikit-learn's own
`IncrementalPCA(r)` is 0.57 rad from batch PCA on the same rows (column "IPCA(r)
vs PCA"), so the loss belongs to the method, not this implementation. It is
worst at `lam = 1` and short streams, where the first regime's dropped
directions still carry weight; under `lam <= 0.999` at `N = 20000`, where the
first regime has decayed away, it is 2e-3 to 4e-3 rad. Under decay no library
checks `r < k`.

**Window** (`W = 2000` rows, `N = 20000`, exact form: each block added, the
block that left removed at its current decayed weight by the inverse of the
combination, read every tenth block): against PCA on the window's rows, largest
angle 3.3e-14 rad, ratio gap 2.6e-15, at `lam = 1` and `0.999`. The removal
subtracts, so a mean far from the spread loses digits there; not measured here
(offset 5, spread 3).

**Zero-weight rows** (the first 300 rows, so the whole first block and part of
the second, and two later blocks, `k = 5`, `lam = 0.999`): no NaN in either
state; every all-zero block leaves `m`, `C` and `m`, `V`, `s`, `tr` bit for bit
as they were and sets `W' = lam^b W` exactly (from `W = 0` too); at the end
the exact form matches PCA on the weighted rows to 5.5e-16 rad.

**State bytes and cost** (the last table below): `k = 20, r = 7`: 1,848 bytes
exact, 1,352 truncated; `k = 200, r = 12`: 162 kB and 21 kB; `k = 2000, r =
12`: 16.0 MB and 208 kB.

### Verdict

The exact block form is `ew_cov(pca=)` with the rows held in blocks; it adds
nothing but the blocking, and it matches both oracles to rounding with decay,
weights, zero weights and a window. The truncated form is the only new thing:
`8 (2 + k + r k + r)` bytes, no window, exact only at `r = k`, checkable
against a library only at `lam = 1` (to 1e-13, batch for batch) or at `r = k`
under decay. Below some hundreds of columns the exact form wins on every count;
above, the truncated form trades an error that depends on the spectrum's gap
and on drift (0.002 to 0.57 rad on these streams) for a state 77 times smaller
at `k = 2000`.

### The run (`uv run python scripts/study_505_502.py pca`)

#### lambda = 1, unit weights: against IncrementalPCA, same 200-row batches

| k | n compared | r | N | exact vs IPCA(all): angle, 1-cos, ratio | truncated(r) vs IPCA(r): angle, 1-cos, ratio | IPCA(r) vs PCA: angle |
|---|---|---|---|---|---|---|
| 2 | 1 | 2 | 2000 | 2.4e-15, 0.0e+00, 2.4e-15 | 2.4e-15, 1.1e-16, 3.3e-15 | 2.4e-15 |
| 2 | 1 | 2 | 20000 | 3.2e-14, 2.2e-16, 6.3e-15 | 2.3e-14, 2.2e-16, 6.4e-15 | 3.1e-14 |
| 5 | 2 | 4 | 2000 | 1.5e-14, 4.4e-16, 1.1e-15 | 5.0e-14, 4.4e-16, 1.6e-15 | 3.1e-01 |
| 5 | 2 | 4 | 20000 | 9.1e-14, 4.4e-16, 6.9e-15 | 2.2e-14, 6.7e-16, 2.9e-15 | 1.3e-01 |
| 20 | 5 | 7 | 2000 | 1.9e-14, 5.6e-16, 2.2e-15 | 1.3e-14, 1.8e-15, 2.6e-15 | 5.7e-01 |
| 20 | 5 | 7 | 20000 | 3.4e-14, 8.9e-16, 5.1e-15 | 9.5e-14, 7.8e-16, 8.4e-15 | 4.4e-01 |

worst exact vs IPCA(all): angle 9.1e-14, 1-cos 8.9e-16, ratio gap 6.9e-15
worst truncated vs IPCA(r): angle 9.5e-14, 1-cos 1.8e-15, ratio gap 8.4e-15

#### Decay, weights U(0.5, 1.5) with 10% zeros: against PCA on the weighted rows

| k | n | lambda | N | exact vs PCA: angle, 1-cos, ratio | exact vs eigh: angle | truncated r=n+2 vs PCA: angle, ratio | truncated r=k vs PCA: angle, ratio |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 1.0 | 2000 | 1.1e-15, 0.0e+00, 2.2e-16 | 2.4e-15 | 3.1e-15, 1.7e-15 | 3.1e-15, 1.7e-15 |
| 2 | 1 | 1.0 | 20000 | 2.6e-15, 0.0e+00, 3.3e-16 | 2.2e-14 | 8.5e-15, 3.8e-15 | 8.5e-15, 3.8e-15 |
| 2 | 1 | 0.999 | 2000 | 5.6e-16, 1.1e-16, 2.2e-16 | 4.2e-15 | 3.1e-15, 1.8e-15 | 3.1e-15, 1.8e-15 |
| 2 | 1 | 0.999 | 20000 | 1.1e-15, 2.2e-16, 3.3e-16 | 1.8e-15 | 1.6e-15, 6.7e-16 | 1.6e-15, 6.7e-16 |
| 2 | 1 | 0.99 | 2000 | 8.0e-16, 2.2e-16, 2.2e-16 | 2.0e-15 | 1.2e-15, 5.6e-16 | 1.2e-15, 5.6e-16 |
| 2 | 1 | 0.99 | 20000 | 8.0e-16, 1.1e-16, 3.3e-16 | 1.2e-15 | 5.0e-16, 2.2e-16 | 5.0e-16, 2.2e-16 |
| 5 | 2 | 1.0 | 2000 | 6.5e-15, 3.3e-16, 2.8e-16 | 3.7e-15 | 3.4e-01, 2.4e-02 | 6.0e-15, 1.8e-15 |
| 5 | 2 | 1.0 | 20000 | 2.2e-15, 7.8e-16, 3.9e-16 | 4.9e-15 | 1.3e-01, 6.1e-03 | 5.0e-14, 4.5e-15 |
| 5 | 2 | 0.999 | 2000 | 3.8e-15, 2.2e-16, 5.6e-16 | 9.7e-15 | 2.0e-01, 9.8e-03 | 2.4e-14, 8.3e-16 |
| 5 | 2 | 0.999 | 20000 | 2.2e-15, 6.7e-16, 9.4e-16 | 1.7e-15 | 4.2e-03, 3.1e-05 | 4.6e-15, 1.8e-15 |
| 5 | 2 | 0.99 | 2000 | 1.6e-14, 3.3e-16, 6.7e-16 | 1.1e-15 | 2.8e-03, 6.5e-05 | 2.6e-15, 5.6e-16 |
| 5 | 2 | 0.99 | 20000 | 2.9e-15, 5.6e-16, 4.4e-16 | 1.2e-15 | 1.8e-03, 2.8e-05 | 5.5e-15, 4.4e-16 |
| 20 | 5 | 1.0 | 2000 | 4.2e-15, 1.1e-15, 7.2e-16 | 3.5e-15 | 4.4e-01, 8.8e-03 | 7.5e-15, 1.6e-15 |
| 20 | 5 | 1.0 | 20000 | 3.4e-15, 5.6e-16, 6.7e-16 | 2.1e-15 | 4.6e-01, 1.3e-02 | 4.1e-14, 7.0e-15 |
| 20 | 5 | 0.999 | 2000 | 5.7e-15, 6.7e-16, 5.6e-16 | 4.7e-15 | 4.5e-01, 7.0e-03 | 1.8e-14, 1.4e-15 |
| 20 | 5 | 0.999 | 20000 | 3.0e-15, 1.4e-15, 1.9e-15 | 1.9e-15 | 3.3e-03, 1.4e-05 | 3.4e-15, 3.2e-15 |
| 20 | 5 | 0.99 | 2000 | 2.2e-15, 1.2e-15, 9.4e-16 | 2.6e-15 | 3.0e-03, 1.5e-05 | 1.6e-15, 7.2e-16 |
| 20 | 5 | 0.99 | 20000 | 2.1e-15, 1.1e-15, 9.2e-16 | 1.6e-15 | 3.5e-03, 1.5e-05 | 2.5e-15, 7.2e-16 |

worst (lambda < 1) exact vs weighted PCA: angle 1.6e-14, 1-cos 1.4e-15, ratio gap 1.9e-15
worst (lambda < 1) truncated r = k vs weighted PCA: angle 2.4e-14, ratio gap 3.2e-15

#### Window of the last 2000 rows (N = 20000): against PCA on the window rows

| k | n | lambda | angle | 1-cos | ratio gap |
|---|---|---|---|---|---|
| 2 | 1 | 1.0 | 2.2e-15 | 3.3e-16 | 1.4e-15 |
| 2 | 1 | 0.999 | 1.1e-15 | 3.3e-16 | 4.4e-16 |
| 5 | 2 | 1.0 | 2.5e-14 | 6.7e-16 | 1.0e-15 |
| 5 | 2 | 0.999 | 1.0e-14 | 1.1e-15 | 8.3e-16 |
| 20 | 5 | 1.0 | 3.3e-14 | 1.4e-15 | 2.6e-15 |
| 20 | 5 | 0.999 | 4.3e-15 | 1.8e-15 | 1.5e-15 |

worst window: angle 3.3e-14, 1-cos 1.8e-15, ratio gap 2.6e-15

#### Zero-weight rows

- NaN anywhere in either state: False
- all-zero blocks leave m, C (exact) and m, V, s, tr (truncated) bit-identical: True
- all-zero blocks only decay W (W' == lam^b W exactly, also from 0): True
- end of stream vs PCA on the weighted rows: exact angle 5.5e-16, ratio gap 3.3e-16; truncated (r = 4 of k = 5) angle 3.4e-03, ratio gap 7.9e-05

#### Block size: b = 200 against b = 1 (exact form, k = 20, N = 2000)

- largest |C(b=200) - C(b=1)| / max|C|: 7.4e-15 (rounding, not bit-equal)

#### Cost per 200-row block (this VM, numpy's default BLAS threads, median of 5)

| k | r | exact fold | exact eigh read | truncated fold (SVD) | exact state bytes | truncated state bytes |
|---|---|---|---|---|---|---|
| 20 | 7 | 0.05 ms | 0.11 ms | 0.19 ms | 1,848 | 1,352 |
| 200 | 12 | 3.93 ms | 7.83 ms | 12.30 ms | 162,408 | 20,912 |
| 2000 | 12 | 81.72 ms | 483.13 ms | 153.14 ms | 16,024,008 | 208,112 |

