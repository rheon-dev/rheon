<!-- kind: record -->
# Study 505 / 502: block incremental PCA, and kernel RLS

A Python-only numpy study; nothing here changes the package. One script is the
entry point, `scripts/study_505_502.py`, with the subcommands `pca`, `krls`,
`tol` (the tolerance tables) and `all` (the default), and `--quick` for a short
run of every check (see part 3 for the times). Every input comes from seeded
generators and no data file is read or written; nothing random is on the
output path. Part 1 (task 505, block incremental PCA), part 2 (task 502,
kernel RLS) and part 3 (the single script, the proposed tolerances and the
summary an implementation starts from) are below.

**Status:** done (parts 1 to 3). This record needs a row in `docs/README.md`'s
Records table: until the coordinator adds it at the merge,
`tests/test_doc_structure.py::test_every_document_under_docs_has_a_row_in_the_index`
fails on this branch. The row is left to the merge on purpose.

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

## Part 2: kernel recursive least squares (task 502)

### The question

The scan (section 8 of the same file) lists KRLS with a dictionary budget as
"A: order-dependent", with no library, and says that "until the budget binds and
at threshold 0 it equals [sk] `KernelRidge` on the same rows: 1e-8", only a
statistical bound after. This part builds approximate-linear-dependence (ALD)
KRLS with a fixed cap, and the sliding-window KRLS, in numpy; finds which fits
are the same definition as a scikit-learn fit (and checks them to rounding);
measures the others; and states the state's size and the numerical risk.

### The math

**Model.** `f(x) = sum_{i=1..m} alpha_i k(d_i, x)` over a dictionary
`D = {d_1..d_m}` of past rows (atoms), `m <= M`. Write `K_D` for the `m x m`
kernel matrix of the atoms and `kv(x) = k(D, x)` (length `m`). Kernels:
RBF `exp(-gamma |x - y|^2)` and polynomial `(gamma <x, y> + 1)^3`, both with
`gamma = 1/k` (scikit-learn's `rbf` and `polynomial`, `degree=3, coef0=1`).

**The ALD test** (Engel, Mannor and Meir 2004, "The kernel recursive
least-squares algorithm", IEEE Transactions on Signal Processing 52(8),
2275-2285; cited from memory, not checked here). The residual of `phi(x)`
projected on the span of the atoms' features is

```text
a = K_D^-1 kv(x),     delta = k(x, x) - kv(x)' a     (>= 0 in exact arithmetic)
```

and `x` joins the dictionary when `delta > nu`. Admission extends the inverse
by the block (Schur) formula, `O(m^2)`:

```text
K_D^-1 <- [[K_D^-1 + a a'/delta, -a/delta], [-a'/delta, 1/delta]]
```

**The coefficients, with a ridge `r`.** The fit is the subset-of-regressors
ridge on the dictionary:

```text
min_alpha  sum_t w_t (y_t - kv_t' alpha)^2 + r alpha' K_D alpha
H = sum_t w_t kv_t kv_t' + r K_D,   P = H^-1,   alpha = P sum_t w_t kv_t y_t
```

Each row is a weighted RLS step on the feature `kv_t` (`O(m^2)`):

```text
g = P kv / (1/w + kv' P kv),   alpha += g (y - kv' alpha),   P -= g (P kv)'
```

When an atom is admitted, `H` gains a column whose past entries need
`sum_s w_s k(x_s, x_new) kv_s`, which a bounded state cannot hold. Engel's
approximation replaces each past row's `k(x_s, x_new)` by its projection
`kv_s' K_D^-1 kv(x_new) = kv_s' a` (exact for a past row that is itself an
atom; off by at most `sqrt(delta_s delta_new)` by Cauchy-Schwarz otherwise).
With it the new column is `H a` and the new diagonal `a' H a + r delta`, so the
Schur complement is `r delta` and

```text
P <- [[P + a a'/(r delta), -a/(r delta)], [-a'/(r delta), 1/(r delta)]],   alpha <- [alpha; 0]
```

then the row's own RLS step with `kv <- [kv; k(x, x)]`. With `r = 0` this is
Engel's unregularised KRLS; `r > 0` is what makes the comparison with
`KernelRidge(alpha=r)` a matter of definition. It follows that the online fit is
**exactly** the batch fit `min sum w (y - kv' alpha)^2 + r alpha' K_D alpha` on
the final dictionary whenever no row was learned by projection before an atom
admitted after it: in particular (i) when every learned row is an atom, which
is `KernelRidge(alpha=r)` on those rows (`alpha = (K_D + r W^-1)^-1 y_D`), and
(ii) at `nu = 0` under the stop rule, where the first `M` rows are the atoms and
every later row is learned on the frozen dictionary. Otherwise it differs from
that batch fit by the projection of the rows learned during the growth phase.

**The cap's rule: stop admitting.** When `m = M`, no row is admitted any more;
every row is still learned, on the frozen dictionary, exactly (its `kv` is
exact). The dictionary is therefore the first `M` rows that pass the ALD test,
in arrival order: deterministic, no randomness on the output path. Rejected
alternatives: replacing the least useful atom (the pruning of later work, e.g.
the fixed-budget KRLS of Van Vaerenbergh et al., ICASSP 2010, cited from memory) needs a downdate of `P` that
loses the dropped atom's share of the past rows, which is no longer a fit of
any batch definition; it is left out.

**Sliding-window KRLS** (Van Vaerenbergh, Via and Santamaria 2006, "A
sliding-window kernel RLS algorithm and its application to nonlinear channel
identification", ICASSP; cited from memory). Every learned row of the last `W`
rows is an atom, `M = W`, and the state keeps

```text
Q = (K_W + c diag(1/w))^-1,    alpha = Q y_W
add (s = k(x, x) + c/w - kv' Q kv,  q = Q kv):  Q <- [[Q + q q'/s, -q/s], [-q'/s, 1/s]]
drop the oldest (e = Q_00, f = Q_1:,0):          Q <- Q_1:,1: - f f'/e
```

which is `KernelRidge(alpha=c, sample_weight=w)` on the window's rows by
definition (scikit-learn's weighted solve is `(K + c W^-1)^-1 y`). The window
is counted in rows; a zero-weight row advances it and is not an atom. The
Schur complement is at least `c/w`, so the regularised window cannot meet the
loss of definiteness below.

**Zero-weight rows**, including the first: `w = 0` returns before the ALD test,
so it is neither admitted nor learned and the state is bit for bit unchanged
(no clock here: there is no decay). The first learned row meets an empty
dictionary (`a` empty, `delta = k(x, x)`), so no division by a zero weight or a
zero count ever happens; `1/w` is only formed for `w > 0`.

**State** (f64, `M` atoms, `k` columns):

| form | fields | bytes, dense | bytes, symmetric halves packed |
|---|---|---|---|
| ALD-KRLS | `D` (`M k`), `K_D^-1` (`M^2`), `P` (`M^2`), `alpha` (`M`) | `8 (M k + 2 M^2 + M)` | `8 (M k + M (M+1) + M)` |
| window | rows (`W k`), `y` (`W`), row index (`W`), `Q` (`W^2`) | `8 (W k + 2 W + W^2)` | `8 (W k + 2 W + W (W+1)/2)` |

The scan's `8 (m^2 + m k)` counts one `M x M` matrix; ALD-KRLS needs two
(`K_D^-1` for the test, `P` for the fit). Bounded by `M` and `k`, never by the
stream. For `M` in {20, 50, 200} and `k` in {2, 5}: 6,880 / 7,360, 41,200 /
42,400, 644,800 / 649,600 bytes dense (3.8 to 331 kB packed); the window
3,840 / 4,320, 21,600 / 22,800, 326,400 / 331,200 bytes dense. Grouped, that
is per group: `M = 200` costs 0.65 MB a group.

**Per-row cost.** ALD test `2 M^2` (`K_D^-1 kv`) plus `M k` kernel
evaluations; RLS step `2 M^2` (`P kv`) plus `2 M^2` (the rank-one update; `M^2`
by symmetry); an admission adds `2 M^2` twice. About `6` to `8 M^2` flops a row,
`O(M^2)`. Window: add `~4 W^2`, drop `~2 W^2`, `alpha = Q y` `2 W^2` at a read.
Measured on this VM (numpy, one row at a time, so mostly call overhead below
`M = 50`): 19 to 32 us a row at `M <= 50`, 99 to 129 us at `M = 200`; window
34 to 50 us and 265 to 290 us.

### The oracles

- **`sklearn.kernel_ridge.KernelRidge(alpha=r, kernel=..., gamma=1/k[, degree=3,
  coef0=1])`**, `sample_weight=w`, `r = 0.1` throughout, fit (a) on the
  dictionary's atoms (their own `x`, `y`, `w`), (b) on the window's rows, (c)
  on all `N` rows for context. Coefficients (`dual_coef_` against `alpha`) are
  compared only where the atoms and their order are the same, cases (a) and (b);
  for every other pair the basis differs and only predictions are compared.
- **`sklearn.kernel_approximation.Nystroem(n_components=m)` fit on the atoms**
  (with `n_components` equal to the number of landmarks its seeded permutation
  only reorders them), then **`Ridge(alpha=r, fit_intercept=False,
  sample_weight=w)`** on every learned row: `phi = K_D^-1/2 kv`, so `|v|^2 =
  alpha' K_D alpha` and this is the subset-of-regressors fit above on the same
  dictionary. Nystroem clips the atoms' eigenvalues at 1e-12, so it is not an
  oracle when `K_D` is that ill-conditioned (the `nu = 0` rows below).
- Predictions are compared on 500 held-out rows; "gap" is the largest absolute
  difference (the target's spread is about 1.5).

Data: `x ~ N(0, I_k)`, `y = sin(1.5 x.u) + 0.3 (x.v)^2 + 0.1 eps`, `u`, `v`
seeded unit vectors; `N = 2000` rows, 500 held out, seeds 0 to 4, `nu = 1e-3`
for the grid, `r = c = 0.1`. Every row has unit weight except in the
zero-weight run.

### Results

**Same definition, to rounding.** Atoms-only KRLS (every learned row is an
atom, `nu = 1e-3, M = 50`) against KernelRidge on the atoms: predictions within
6.7e-11 (RBF) and 8.6e-11 (polynomial), coefficients within 4.4e-10 relative.
Over the grid (`M` = 20, 50, 200): within 2.5e-10. At `nu = 0` and `M = 20`
under the stop rule, against Nystroem + Ridge on all 2000 rows: 1.3e-8 (RBF
k = 2), 2.2e-14 (RBF k = 5), 8.7e-13 (poly k = 5). Uncapped at `nu = 0` on 150
rows, RBF `k = 5`, every row admitted: 9.0e-12 against KernelRidge on all 150
rows (coefficients 1.7e-11): the scan's "equals KernelRidge until the budget
binds at threshold 0" holds, but only while `K_D` stays well conditioned (RBF
`k = 2` and polynomial do not; see the numerical risk).

**Dictionary growth** (`nu = 1e-3`, per 1000 rows, mean of 5 seeds):

| kernel | k | uncapped ALD, all 2000 rows | M = 20 | M = 50 | M = 200 |
|---|---|---|---|---|---|
| RBF | 2 | 90.6 | 20 / 0 | 50 / 0 | 79.6 / 11.0 |
| RBF | 5 | 984.4 | 20 / 0 | 50 / 0 | 200 / 0 |
| poly | 2 | 10.0 (the feature space's dimension, `C(5,3)`) | 10 / 0 | 10 / 0 | 10 / 0 |
| poly | 5 | 56.0 (`C(8,3)`) | 20 / 0 | 50 / 0 | 56 / 0 |

(first 1000 rows / second 1000.) Every cap except RBF `k = 2, M = 200` and the
polynomial's saturation binds inside the first 1000 rows (at `M` rows when it
binds: the first rows are nearly always new). RBF `k = 5` at `gamma = 1/k`
admits about one row in two: ALD at `nu = 1e-3` hardly sparsifies there, and
the cap does all the work.

**Gaps on the same dictionary, and the held-out error** (worst gap over seeds,
mean MSE; noise variance 0.01):

| kernel | k | M | KRLS vs Nystroem+Ridge, same D, all rows | KRLS vs KernelRidge on D | MSE: KRLS / KernelRidge on D / batch KernelRidge, all rows |
|---|---|---|---|---|---|
| RBF | 2 | 20 | 3.1e-4 | 1.7 | 0.080 / 0.187 / 0.016 |
| RBF | 2 | 50 | 5.2e-3 | 2.3 | 0.023 / 0.054 / 0.016 |
| RBF | 2 | 200 | 1.2e-2 | 0.17 | 0.016 / 0.019 / 0.016 |
| RBF | 5 | 20 | 2.2e-14 | 0.97 | 0.263 / 0.374 / 0.019 |
| RBF | 5 | 50 | 6.8e-13 | 1.2 | 0.121 / 0.224 / 0.019 |
| RBF | 5 | 200 | 1.1e-3 | 1.7 | 0.032 / 0.077 / 0.019 |
| poly | 2 | any | 1.3e-9 | 13 | 0.079 / 0.575 / 0.079 |
| poly | 5 | 20 | 8.7e-13 | 2.6 | 0.312 / 0.611 / 0.073 |
| poly | 5 | 50 | 6.9e-11 | 3.8 | 0.095 / 0.253 / 0.073 |
| poly | 5 | 200 | 3.9e-5 | 3.9 | 0.073 / 0.220 / 0.073 |

Read across: against **KernelRidge on the same dictionary** the full KRLS is
not the same estimator (it also learns the 1,800 to 1,990 rows it did not
admit), so the gap is large (0.17 to 13) and favours KRLS: its held-out MSE is
0.14 to 0.87 times KernelRidge-on-atoms'. The exact library check of the
arithmetic is the atoms-only row above (2.5e-10). Against **Nystroem + Ridge
on the same dictionary and all rows** (the closest batch definition) the gap
is rounding where no projected row preceded an admission (1e-14 to 1e-9:
caps that bind at once, and the polynomial, whose projections are exact once
the span is full), and 3e-4 to 1.2e-2 where rows were projected during growth
(RBF `k = 2`, RBF `k = 5, M = 200`, poly `k = 5, M = 200`): Engel's
approximation, bounded by `nu`, not a bug. Against **batch KernelRidge on all
rows**, KRLS reaches it where the dictionary holds the function's span (poly
at saturation: identical MSE; RBF `k = 2, M = 200`: 0.0161 against 0.0161)
and pays for the cap elsewhere (RBF `k = 5`: 0.032 at `M = 200`, 0.26 at
`M = 20`, against 0.019).

**Window** (`W` = 20, 50, 200; checked every 100 rows once full, 5 seeds):
against KernelRidge on the window's rows, predictions within 7.3e-13 (RBF) and
1.0e-8 (polynomial, `k = 2, W = 200`, the worst-conditioned `K_W + c I`), and
coefficients within 2.8e-9 relative, after 2000 add-and-drop steps with no
re-factorisation. The window's held-out MSE is worse than ALD-KRLS at the same
`M` (RBF `k = 2, W = 200`: 0.038 against 0.016), since it forgets by design.

**Zero-weight rows** (the first 50 rows, then one in ten, at `w = 0`; others
U(0.5, 1.5)): no NaN in any state; every zero-weight row leaves `D`, `alpha`,
`P`, `K_D^-1` bit-identical and the window's atoms unchanged apart from the
drop of its oldest by the row clock; the first atom is row 50. With weights,
the window matches `KernelRidge(sample_weight=w)` to 2.5e-13 (RBF) and 6.3e-10
(poly), and RBF `nu = 0, M = 20` matches Nystroem + Ridge(`sample_weight`) to
1.3e-11.

**Numerical risk** (RBF and poly, `k = 2, M = 200`, seed 0, table below). The
test's `delta` is a difference of nearly equal numbers, and both `K_D^-1` and
`P` carry `1/delta` (and `1/(r delta)`) at each admission:

- `nu >= 1e-3`: no failure. `K_D^-1` stays positive definite and within
  1.4e-7 of `inv(K_D)` (RBF; `cond(K_D)` 1.1e7) or 3.8e-10 (poly); `P` stays
  positive definite. Rounding-negative `delta` values appear (218 for the
  polynomial once its 10-dimensional span is full, where every true `delta` is
  0; 11 for RBF at `nu = 1e-4`) and are harmless while they are rejected.
- `nu = 1e-6` (RBF): `cond(K_D)` 5.7e15, `K_D^-1` 3.4 relative from the true
  inverse, `P` indefinite (smallest eigenvalue -1.1e4), predictions 0.40 from
  the batch fit on the same dictionary. `nu <= 1e-9` and `nu = 0`: `K_D^-1`
  indefinite too (-8.8e2), predictions 22 off.
- Polynomial at `nu <= 1e-9`: rounding residuals of about 1e-12 pass the test,
  so atoms 11 to 20 enter a space that has only 10 dimensions; `K_D` is
  singular (`cond` 4e17) and the fit is 1.7 to 18 off. The same failure is the
  1.3 gap of the `nu = 0` zero-weight run (poly) and the 1.3 to 19 gaps of the
  uncapped `nu = 0` rows (RBF `k = 2`, poly) in the exact-cases table.
- The cap bounds the damage only by bounding `m`: near the cap with `nu`
  small, the last atoms are the most nearly dependent, and they are the ones
  whose `1/delta` breaks the inverse.

So `nu` must be a floor on the conditioning, not a sparsity knob that may go
to 0: about `1e-3 * k(x, x)` held `K_D^-1` to 1e-7 here; a library default
should refuse `nu` below about `1e-8 * k(x,x)` (or keep a Cholesky of `K_D`
rather than its inverse, at the same `M^2/2` state, and re-test `delta`
against it). The window with `c > 0` showed no such failure: its Schur
complement is at least `c/w`.

### Verdict

Two fits are exact definitions a library checks to 1e-8 or better:
atoms-only KRLS (KernelRidge on the dictionary's rows, 2.5e-10) and the
sliding window (KernelRidge on the window's rows, 1e-8 worst, 7e-13 for RBF);
ALD-KRLS learning every row is Nystroem + Ridge on the final dictionary
exactly when no row was projected before a later admission, and within the
growth phase's projection error otherwise (up to 1.2e-2 here at `nu = 1e-3`).
The model that learns every row is the useful one: its held-out error is 0.14
to 0.87 times KernelRidge-on-atoms' and reaches batch KernelRidge when the
dictionary covers the function. The cost is `8 (M k + 2 M^2 + M)` bytes (two
`M x M` matrices, not the scan's one: 0.65 MB a group at `M = 200`) and
`O(M^2)` per row; the risk is `nu`, which must stay well above rounding or the
inverse loses definiteness. ALD at `nu = 1e-3` does not sparsify an RBF in
five dimensions (one row in two admitted), so there the cap is the
dictionary, chosen as the first `M` rows.

### The run (`uv run python scripts/study_505_502.py krls`)

The grid took 88 s and the window, zero-weight and numerical parts 56 s (each
run with nohup and logged); the exact cases and the cost table 2.4 s.


#### Exact cases: the same definition, library oracle

| kernel | k | case | rows learned | atoms | max abs pred gap (held-out) | max abs coef gap / max abs coef |
|---|---|---|---|---|---|---|
| rbf | 2 | uncapped nu=0, first 150 rows | 150 | 73 | 1.3e+00 | n.a. (only 73 atoms: rank) |
| rbf | 2 | atoms only, nu=1e-3, M=50 | 2000 | 50 | 6.7e-11 | 4.4e-10 |
| rbf | 2 | nu=0, M=20, stop rule, all 2000 rows | 2000 | 20 | 1.3e-08 | n.a. (basis differs) |
| rbf | 5 | uncapped nu=0, first 150 rows | 150 | 150 | 9.0e-12 | 1.7e-11 |
| rbf | 5 | atoms only, nu=1e-3, M=50 | 2000 | 50 | 1.3e-13 | 3.8e-13 |
| rbf | 5 | nu=0, M=20, stop rule, all 2000 rows | 2000 | 20 | 2.2e-14 | n.a. (basis differs) |
| poly | 2 | uncapped nu=0, first 150 rows | 150 | 17 | 9.9e+00 | n.a. (only 17 atoms: rank) |
| poly | 2 | atoms only, nu=1e-3, M=50 | 2000 | 10 | 8.6e-11 | 2.2e-11 |
| poly | 2 | nu=0, M=20, stop rule, all 2000 rows | 2000 | 17 | 1.9e+01 | n.a. (basis differs) |
| poly | 5 | uncapped nu=0, first 150 rows | 150 | 63 | 5.7e+00 | n.a. (only 63 atoms: rank) |
| poly | 5 | atoms only, nu=1e-3, M=50 | 2000 | 50 | 7.9e-12 | 4.9e-12 |
| poly | 5 | nu=0, M=20, stop rule, all 2000 rows | 2000 | 20 | 8.7e-13 | n.a. (basis differs) |

#### State bytes and per-row cost

| M | k | ALD-KRLS: D + Kinv + P + alpha, dense | packed (two triangles) | window: rows + y + t + Q, dense | flops per row, ALD (about 8 M^2) | measured us/row (ALD, cap full) | measured us/row (window, full) |
|---|---|---|---|---|---|---|---|
| 20 | 2 | 6,880 | 3,840 | 3,840 | 3,200 | 22 | 40 |
| 20 | 5 | 7,360 | 4,320 | 4,320 | 3,200 | 21 | 40 |
| 50 | 2 | 41,200 | 21,600 | 21,600 | 20,000 | 25 | 48 |
| 50 | 5 | 42,400 | 22,800 | 22,800 | 20,000 | 26 | 61 |
| 200 | 2 | 644,800 | 326,400 | 326,400 | 320,000 | 137 | 274 |
| 200 | 5 | 649,600 | 331,200 | 331,200 | 320,000 | 124 | 286 |


#### Grid: nu = 0.001, r = 0.1, N = 2000, 5 seeds (worst gap, mean MSE)

| kernel | k | M | atoms in rows 1-1000 / 1001-2000 (mean) | uncapped ALD atoms (mean) | pred gap: KRLS vs Nystroem+Ridge, same D, all rows | pred gap: atoms-only KRLS vs KernelRidge on D | pred gap: KRLS vs KernelRidge on D | held-out MSE: KRLS / KR on D / batch KR on all rows |
|---|---|---|---|---|---|---|---|---|
| rbf | 2 | 20 | 20.0 / 0.0 | 90.6 | 3.1e-04 | 3.2e-12 | 1.7e+00 | 0.0800 / 0.1868 / 0.0161 |
| rbf | 2 | 50 | 50.0 / 0.0 | 90.6 | 5.2e-03 | 6.7e-11 | 2.3e+00 | 0.0234 / 0.0544 / 0.0161 |
| rbf | 2 | 200 | 79.6 / 11.0 | 90.6 | 1.2e-02 | 2.5e-10 | 1.7e-01 | 0.0161 / 0.0186 / 0.0161 |
| rbf | 5 | 20 | 20.0 / 0.0 | 984.4 | 2.2e-14 | 4.4e-15 | 9.7e-01 | 0.2632 / 0.3736 / 0.0188 |
| rbf | 5 | 50 | 50.0 / 0.0 | 984.4 | 6.8e-13 | 1.3e-13 | 1.2e+00 | 0.1207 / 0.2239 / 0.0188 |
| rbf | 5 | 200 | 200.0 / 0.0 | 984.4 | 1.1e-03 | 1.6e-11 | 1.7e+00 | 0.0319 / 0.0770 / 0.0188 |
| poly | 2 | 20 | 10.0 / 0.0 | 10.0 | 1.3e-09 | 8.6e-11 | 1.3e+01 | 0.0791 / 0.5754 / 0.0791 |
| poly | 2 | 50 | 10.0 / 0.0 | 10.0 | 1.3e-09 | 8.6e-11 | 1.3e+01 | 0.0791 / 0.5754 / 0.0791 |
| poly | 2 | 200 | 10.0 / 0.0 | 10.0 | 1.3e-09 | 8.6e-11 | 1.3e+01 | 0.0791 / 0.5754 / 0.0791 |
| poly | 5 | 20 | 20.0 / 0.0 | 56.0 | 8.7e-13 | 5.5e-14 | 2.6e+00 | 0.3116 / 0.6106 / 0.0727 |
| poly | 5 | 50 | 50.0 / 0.0 | 56.0 | 6.9e-11 | 7.9e-12 | 3.8e+00 | 0.0945 / 0.2529 / 0.0727 |
| poly | 5 | 200 | 56.0 / 0.0 | 56.0 | 3.9e-05 | 2.5e-10 | 3.9e+00 | 0.0727 / 0.2195 / 0.0727 |

(noise variance 0.0100: the floor for every held-out MSE)


#### Window: last W rows, c = r = 0.1, against KernelRidge on the window rows

| kernel | k | W | max abs pred gap (held-out, every 100 rows) | max coef gap / max coef | held-out MSE at the end |
|---|---|---|---|---|---|
| rbf | 2 | 20 | 2.2e-14 | 7.1e-14 | 0.2095 |
| rbf | 2 | 50 | 8.1e-14 | 3.9e-13 | 0.1149 |
| rbf | 2 | 200 | 7.3e-13 | 2.5e-12 | 0.0382 |
| rbf | 5 | 20 | 9.1e-15 | 1.6e-14 | 0.3845 |
| rbf | 5 | 50 | 3.8e-14 | 8.8e-14 | 0.2294 |
| rbf | 5 | 200 | 2.9e-13 | 6.7e-13 | 0.0956 |
| poly | 2 | 20 | 1.3e-10 | 2.3e-11 | 0.4915 |
| poly | 2 | 50 | 6.7e-10 | 3.4e-10 | 0.1223 |
| poly | 2 | 200 | 1.0e-08 | 2.8e-09 | 0.0916 |
| poly | 5 | 20 | 1.6e-13 | 1.2e-13 | 0.5567 |
| poly | 5 | 50 | 3.0e-12 | 9.9e-13 | 0.3495 |
| poly | 5 | 200 | 7.7e-11 | 5.6e-11 | 0.1307 |

#### Zero-weight rows (first 50 rows, then one row in ten, at w = 0)

- rbf: NaN in any state: False; zero-weight rows leave the ALD states bit-identical and the window's only by its clock: True; first atom at row 50; nu = 0, M = 20 vs Nystroem+Ridge(sample_weight): 1.3e-11; window W = 200 vs KernelRidge(sample_weight): 2.5e-13
- poly: NaN in any state: False; zero-weight rows leave the ALD states bit-identical and the window's only by its clock: True; first atom at row 50; nu = 0, M = 20 vs Nystroem+Ridge(sample_weight): 1.3e+00; window W = 200 vs KernelRidge(sample_weight): 6.3e-10

#### Numerical risk: Kinv and P against nu (k = 2, M = 200, N = 2000, seed 0)

| kernel | nu | atoms | smallest admitted delta | delta < 0 seen | min eig sym(Kinv) | rel err Kinv vs inv(K_D) | min eig sym(P) | cond K_D | pred gap vs Nystroem+Ridge |
|---|---|---|---|---|---|---|---|---|---|
| rbf | 0.01 | 66 | 1.0e-02 | 0 | 1.1e-01 | 4.2e-11 | 1.8e-04 | 8.6e+04 | 6.5e-02 |
| rbf | 0.001 | 90 | 1.0e-03 | 0 | 7.5e-02 | 1.4e-07 | 1.2e-04 | 1.1e+07 | 6.4e-03 |
| rbf | 0.0001 | 113 | 1.0e-04 | 11 | 5.8e-02 | 5.3e-05 | 9.6e-05 | 8.8e+08 | 4.1e-03 |
| rbf | 1e-06 | 141 | 1.0e-06 | 1705 | 3.3e-02 | 3.4e+00 | -1.1e+04 | 5.7e+15 | 4.0e-01 |
| rbf | 1e-09 | 75 | 2.0e-08 | 1925 | -8.8e+02 | 6.4e+03 | -5.3e+03 | 2.8e+16 | 2.2e+01 |
| rbf | 0 | 75 | 2.0e-08 | 1925 | -8.8e+02 | 6.4e+03 | -5.3e+03 | 2.8e+16 | 2.2e+01 |
| poly | 0.01 | 10 | 1.1e-02 | 453 | 1.4e-02 | 1.3e-11 | 2.3e-06 | 3.1e+05 | 1.6e-04 |
| poly | 0.001 | 10 | 2.7e-03 | 218 | 1.3e-02 | 3.8e-10 | 2.1e-06 | 1.9e+05 | 4.2e-11 |
| poly | 0.0001 | 10 | 2.7e-03 | 218 | 1.3e-02 | 3.8e-10 | 2.1e-06 | 1.9e+05 | 4.2e-11 |
| poly | 1e-06 | 10 | 2.7e-03 | 218 | 1.3e-02 | 3.8e-10 | 2.1e-06 | 1.9e+05 | 4.2e-11 |
| poly | 1e-09 | 20 | 1.2e-09 | 1962 | 1.0e-02 | 1.0e+00 | -2.1e-03 | 3.9e+17 | 1.7e+00 |
| poly | 0 | 15 | 3.4e-12 | 1985 | 1.3e-02 | 1.0e+00 | -6.1e+05 | 4.4e+17 | 1.8e+01 |

## Part 3: the single script, proposed tolerances, and the summary

### The script

`scripts/study_505_502.py` is the one entry point (the parts' separate `main`s
are gone; no other study script was left behind):

```text
uv run python scripts/study_505_502.py [pca | krls [exact grid window zero numerics cost] | tol | all] [--quick] [--krls-seeds N]
```

Each run ends with a `## Checks` list and exits 1 if any check fails. The
checks: exact block PCA against IncrementalPCA (`< 1e-12` rad, the proposed
tolerance), the truncated form against `IncrementalPCA(r)` and, at `r = k`,
against weighted PCA under decay, the exact form under decay and in a window
against PCA (each `< 1e-12` rad), block size 200 against 1 (`< 1e-12`
relative), the zero-weight invariants of both parts (no NaN, state unchanged,
`W` only decayed, the first atom is the first weighted row), atoms-only KRLS
against KernelRidge on the atoms (`< 1e-8`, the proposed tolerance), the
window KRLS against KernelRidge on the window (`< 1e-7`), and, with `tol`, the
two checks on the full KRLS given in the tolerances below.

`--quick` runs 2 seeds, `N = 2000`, `lambda` in {1, 0.99}, a 1000-row window
over 4000 rows, KRLS caps and windows 20 and 50, KRLS parts `exact`, `window`
and `zero`, and the tolerance tables at 2 seeds with the full `M` grid (20, 50,
200); it skips the timing tables, the KRLS growth grid and the `nu` sweep. Times on this VM:
`--quick` 9.7 s (21 checks, all passing, through `uv run`); `pca` 13 s; `tol`
72 s at 20 KRLS seeds; `all` 124 s, 21 checks passing (`krls` is the rest:
about 50 s alone, 150 s when part 2 ran it beside other work). The long runs
went under nohup, logged.

### Proposed tolerances

The rule: 1.5 times the largest gap over the seeds, rounded up to a power of
ten, per cell and over the grid. The script holds every run to the grid's
value (`TOL_*` at the top of its part 3).

**(a) Block PCA, exact form, `lambda = 1`, unit weights, against
`IncrementalPCA(n_components=None).partial_fit` on the same 200-row batches**,
in radians of the largest principal angle between the top-`n` spaces
(`n` = 1, 2, 5 for `k` = 2, 5, 20), 5 seeds:

| k | N = 2000: largest angle (rad) | proposed | N = 20000: largest angle (rad) | proposed |
|---|---|---|---|---|
| 2 | 2.4e-15 | 1e-14 | 3.2e-14 | 1e-13 |
| 5 | 1.5e-14 | 1e-13 | 9.1e-14 | 1e-12 |
| 20 | 1.9e-14 | 1e-13 | 3.4e-14 | 1e-13 |

**Proposed: `1e-12` rad** over the grid (largest 9.1e-14). The same bound
covers the other exact comparisons of part 1 (decay and weights against PCA on
the weighted rows, 1.6e-14; the window, 3.3e-14; the truncated form against
`IncrementalPCA(r)`, 9.5e-14), and the script holds them to it as well.

**(b) Atoms-only KRLS against `KernelRidge(alpha=0.1, sample_weight=w)` on the
same dictionary (its atoms' `x`, `y`, `w`)**, in the largest absolute
difference of the predictions on 500 held-out rows (the target's spread is
about 1.5), `nu = 1e-3`, `N = 2000`, **20 seeds**:

| kernel | k | M = 20 | proposed | M = 50 | proposed | M = 200 | proposed |
|---|---|---|---|---|---|---|---|
| RBF | 2 | 4.6e-12 | 1e-11 | 8.0e-11 | 1e-9 | 6.1e-10 (95 atoms) | 1e-9 |
| RBF | 5 | 7.4e-15 | 1e-13 | 1.9e-13 | 1e-12 | 1.6e-11 | 1e-10 |
| poly | 2 | 1.1e-10 (10 atoms) | 1e-9 | same | 1e-9 | same | 1e-9 |
| poly | 5 | 3.0e-13 | 1e-12 | 1.1e-11 | 1e-10 | 1.5e-9 (56 atoms) | 1e-8 |

**Proposed: `1e-8`** absolute over the grid (largest 1.5e-9, polynomial
`k = 5, M = 200`). The 5 seeds of part 2 gave 2.5e-10 and so 1e-9; seed 5 to
19 raised it by a factor of 6, which is why the table rests on 20. The gap
grows with `cond(K_D)` (polynomial, and RBF at `k = 2`), not with the stream.

**The full KRLS (every row learned) is not held to KernelRidge on its
dictionary**: it is a different estimator (it also learns the rows it did not
admit), and the gap is 0.17 to 13 with no tolerance to propose. Its held-out
MSE over KernelRidge-on-atoms' is 0.05 to 0.95 in every cell but one, and
reaches 1.29 in that one (polynomial `k = 5, M = 200`, where batch KernelRidge
on all rows is worse than the 56 atoms alone by the same factor), so "beats
KernelRidge on its atoms" is not a valid test either. What its test can check, from the
20-seed measurements:

1. **Against `Nystroem(landmarks = its dictionary)` + `Ridge(alpha=0.1,
   fit_intercept=False, sample_weight=w)` on every learned row, where no row
   was learned by projection before the last admission** (the full form's
   batch definition holds exactly there: a cap that binds before any rejection,
   `nu = 0` under the stop rule, or a polynomial that saturates without a
   rejection): largest gap 1.9e-8 (polynomial `k = 5, M = 200`; 3.3e-14 to
   1.9e-9 elsewhere), **proposed `1e-7`**. Where rows were projected during
   the growth phase the gap is Engel's approximation, 4.6e-5 to 1.7e-2 here,
   bounded by `nu`, and not a tolerance.
2. **The held-out error against batch KernelRidge on all rows, where the cap
   did not bind** (the dictionary stopped growing by the ALD test: polynomial
   at its feature dimension, RBF `k = 2` at `M = 200` with 90 to 95 atoms):
   `|MSE(KRLS) / MSE(batch) - 1|` at most 2.0e-2 (RBF) and 1.1e-5 (polynomial),
   **proposed `0.05`** (1.5 times 0.020 is 0.030; a power of ten, 0.1, would
   be too coarse to catch a regression). This is a statistical bound on these
   streams, not rounding. Where the cap binds, the ratio is 0.53 to 17.9 and
   depends on `M` and the data; no bound was measured that a test could hold.

The 20-seed table (`uv run python scripts/study_505_502.py tol`):

| kernel | k | M | atoms (max) | atoms-only: largest abs pred gap | 1.5 x, up to a power of 10 | full vs Nystroem+Ridge, same D, all rows: seeds with no row projected before the last admission, gap there / elsewhere | full: MSE ratio to KernelRidge on D (min / max) | full: MSE ratio to batch KernelRidge (min / max) | full, cap not bound: largest abs(MSE ratio to batch - 1) |
|---|---|---|---|---|---|---|---|---|---|
| rbf | 2 | 20 | 20 | 4.6e-12 | 1e-11 | 3 of 20: 4.9e-11 / 8.2e-04 | 0.205 / 0.721 | 2.786 / 9.815 | - |
| rbf | 2 | 50 | 50 | 8.0e-11 | 1e-09 | 0 of 20: - / 5.2e-03 | 0.220 / 0.610 | 1.024 / 1.817 | - |
| rbf | 2 | 200 | 95 | 6.1e-10 | 1e-09 | 0 of 20: - / 1.7e-02 | 0.590 / 0.954 | 0.997 / 1.020 | 2.0e-02 |
| rbf | 5 | 20 | 20 | 7.4e-15 | 1e-13 | 20 of 20: 3.3e-14 / - | 0.536 / 0.806 | 7.954 / 17.851 | - |
| rbf | 5 | 50 | 50 | 1.9e-13 | 1e-12 | 20 of 20: 8.4e-13 / - | 0.429 / 0.718 | 4.008 / 8.938 | - |
| rbf | 5 | 200 | 200 | 1.6e-11 | 1e-10 | 0 of 20: - / 2.6e-03 | 0.289 / 0.490 | 1.403 / 1.924 | - |
| poly | 2 | 20 | 10 | 1.1e-10 | 1e-09 | 13 of 20: 1.9e-09 / 4.6e-05 | 0.047 / 0.337 | 1.000 / 1.000 | 6.2e-06 |
| poly | 2 | 50 | 10 | 1.1e-10 | 1e-09 | 13 of 20: 1.9e-09 / 4.6e-05 | 0.047 / 0.337 | 1.000 / 1.000 | 6.2e-06 |
| poly | 2 | 200 | 10 | 1.1e-10 | 1e-09 | 13 of 20: 1.9e-09 / 4.6e-05 | 0.047 / 0.337 | 1.000 / 1.000 | 6.2e-06 |
| poly | 5 | 20 | 20 | 3.0e-13 | 1e-12 | 20 of 20: 8.7e-13 / - | 0.349 / 0.705 | 0.895 / 7.419 | - |
| poly | 5 | 50 | 50 | 1.1e-11 | 1e-10 | 20 of 20: 6.9e-11 / - | 0.171 / 0.567 | 0.527 / 3.216 | - |
| poly | 5 | 200 | 56 | 1.5e-09 | 1e-08 | 14 of 20: 1.9e-08 / 7.9e-05 | 0.111 / 1.293 | 1.000 / 1.000 | 1.1e-05 |

The window KRLS (not asked for a tolerance here) was within 1.0e-8 of
KernelRidge on the window's rows over 5 seeds; the script holds it to 1e-7.

**Seeds.** Part 1's tables and tolerance (a): 5 seeds (0 to 4). Part 2's tables:
5 seeds. Tolerance (b) and the full-KRLS bounds: 20 seeds (0 to 19). `--quick`:
2 seeds.

### Summary for an implementation

Everything below is restated from parts 1 and 2 so that an implementation can
start from this section alone. `k` columns, `b` rows a block, `r` the rows of a
truncated basis, `M` the dictionary cap (or the window's rows), `lam` the decay
per row, `w` a row's weight.

**Block PCA, exact form** (mean form, the decayed weighted covariance). A block
decays each row to its end, `u_i = w_i lam^(b-1-i)`, `U = sum u_i`,
`m_B = sum u_i x_i / U`, `C_B = (1/U) sum u_i (x_i - m_B)(x_i - m_B)'`; then

```text
W' = lam^b W + U,   a = lam^b W / W',   beta = U / W'
m' = m + beta (m_B - m)
C' = a C + beta C_B + a beta (m - m_B)(m - m_B)'
read: eigh(C), descending; components, variances lambda_j, ratios lambda_j / trace(C)
```

It equals the per-row recursion to rounding (7.4e-15 relative at `b = 200`
against `b = 1`), so the block size is not part of the result beyond rounding.
A window removes a leaving block at its current decayed weight by the
inverse combination (`W_r = W - U`, `m_r = (W m - U m_B) / W_r`,
`C_r = (W C - U C_B - (W_r U / W) d d') / W_r`, `d = m_r - m_B`).

**Block PCA, truncated form** (`C ~ V' diag(s^2) V`, `V` the top `r` rows). The
thin SVD of the `(r + b + 1) x k` stack `[sqrt(a) diag(s) V; sqrt(beta u_i / U)
(x_i - m_B); sqrt(a beta) (m - m_B)]` gives the new `V`, `s` (top `r` kept);
`tr' = a tr + beta trace(C_B) + a beta |m - m_B|^2`; `m`, `W` as above. Exact at
`r = k` or when the data's rank is at most `r`; for `r < k` it drops the mass
outside the top `r` at every block and never recovers it. No window.

**ALD-KRLS** (dictionary `D` of `m <= M` atoms, ridge `r`, threshold `nu`). Per
row with `w > 0`:

```text
kv = k(D, x),  a = Kinv kv,  delta = k(x, x) - kv' a
if delta > nu and m < M:                       (admit; the cap's rule is: stop admitting)
    Kinv <- [[Kinv + a a'/delta, -a/delta], [-a'/delta, 1/delta]]
    P    <- [[P + a a'/(r delta), -a/(r delta)], [-a'/(r delta), 1/(r delta)]]
    alpha <- [alpha; 0],  kv <- [kv; k(x, x)],  D <- [D; x]
g = P kv / (1/w + kv' P kv),  alpha += g (y - kv' alpha),  P -= g (P kv)'
predict: k(D, x)' alpha
```

`w = 0` returns before the test: the state is unchanged, bit for bit. The
atoms-only form skips the RLS step for a row it does not admit.

**Window KRLS** (every learned row of the last `M` rows is an atom, ridge `c`):

```text
drop oldest while its row index <= t - M:   Q <- Q[1:,1:] - f f'/e,   e = Q_00, f = Q[1:,0]
add (w > 0):  q = Q kv,  s = k(x, x) + c/w - kv' q,  Q <- [[Q + q q'/s, -q/s], [-q'/s, 1/s]]
alpha = Q y_W
```

**State** (f64 bytes; all bounded by the shape, the cap or the window, never by
the stream):

| form | fields | bytes |
|---|---|---|
| block PCA, exact | `W`, `m`, `C` (packed triangle) | `8 (1 + k + k (k+1)/2)` |
| block PCA, truncated | `W`, `m`, `V`, `s`, `tr` | `8 (2 + k + r k + r)` |
| block PCA, a block held | its rows, weights, `lam` | `+ 8 b (k + 2)` |
| block PCA, exact, windowed | the window's rows and weights | `+ 8 W_rows (k + 1)` |
| ALD-KRLS | `D`, `Kinv`, `P`, `alpha` | `8 (M k + 2 M^2 + M)` dense; `8 (M k + M (M+1) + M)` with both symmetric matrices packed |
| window KRLS | rows, `y`, row index, `Q` | `8 (M k + 2 M + M^2)` dense; `8 (M k + 2 M + M (M+1)/2)` packed |

At `k = 2000, r = 12`: 16.0 MB exact against 208 kB truncated. At `M = 200,
k = 5`: 650 kB ALD-KRLS (331 kB packed), 331 kB window, per group.

**Cost.** Exact block fold `2 b k^2` flops (`2 k^2` a row), read `~10 k^3`
(`eigh`, per read, not per row). Truncated fold: the thin SVD of
`(r + b + 1) x k`, `~4 k (r + b + 1)^2 + 8 (r + b + 1)^3` a block, no read.
Measured at `b = 200`: `k = 20`: 0.05 ms exact fold, 0.2 ms truncated;
`k = 2000, r = 12`: 82 ms exact fold plus 483 ms per read, 153 ms truncated.
ALD-KRLS: `M k` kernel evaluations plus about `6` to `8 M^2` flops a row
(`O(M^2)`); window KRLS about `6 M^2` a row (`4 M^2` to add, `2 M^2` to drop), `2 M^2` a read.
Measured (numpy, one row at a time): 20 to 30 us a row at `M <= 50`, 100 to
140 us at `M = 200`; the window 40 to 60 and 270 to 290 us.

**Numerical risks.**

- `nu` and positive definiteness. Both `Kinv` and `P` carry `1/delta` at
  each admission, and `delta` is a difference of nearly equal numbers. At
  `nu >= 1e-3 k(x, x)` both stay positive definite (`Kinv` within 1.4e-7 of
  `inv(K_D)`, `cond(K_D)` 1.1e7, RBF `k = 2, M = 200`). At `nu = 1e-6`, `P`
  is indefinite (smallest eigenvalue -1.1e4) and predictions are 0.40 off the
  batch fit on the same dictionary; at `nu <= 1e-9` (and 0) `Kinv` is
  indefinite too and predictions are 22 off; the polynomial admits rounding
  residuals into a saturated span (atoms 11 to 20 in 10 dimensions). An
  implementation refuses `nu` below about `1e-8 k(x, x)` (or keeps a Cholesky
  of `K_D` and re-tests `delta` against it). The window KRLS with `c > 0` never
  met this: its Schur complement is at least `c / w`.
- The truncated PCA at `r < k` is an approximation: up to 0.46 rad from PCA
  on the weighted rows on these streams (and scikit-learn's own
  `IncrementalPCA(r)` is 0.57 rad from batch PCA), 2e-3 to 4e-3 rad once the
  first regime has decayed away. Only `r = k` or `lam = 1` (against
  `IncrementalPCA(r)`) can be held to a library.
- Dictionary saturation. RBF at `k = 5, gamma = 1/k` admits about one row in
  two at `nu = 1e-3` (984 atoms of 2000 uncapped): ALD does not sparsify there,
  so every cap binds at once (in the first `M` rows) and the dictionary is
  simply the first `M` rows; the held-out error then rests on `M` (MSE 0.26 at
  `M = 20`, 0.032 at `M = 200`, batch 0.019). The polynomial saturates at its
  feature dimension (`C(k+3, 3)`: 10 at `k = 2`, 56 at `k = 5`), after which
  every `delta` is rounding, and only `nu` keeps those rows out.
- The window PCA's removal subtracts; a mean far from the spread loses digits
  there (not measured beyond offset 5, spread 3).

**Oracles.**

- Block PCA: `sklearn.decomposition.IncrementalPCA.partial_fit` on the same
  200-row batches (`lam = 1`, unit weights; `n_components=None` for the exact
  form, `r` for the truncated); `sklearn.decomposition.PCA(svd_solver='full')`
  on `[Y; -Y]`, `Y = sqrt(u_i / U) (x_i - m_u)`, under decay and weights;
  `scipy.linalg.eigh` of `np.cov(aweights=u, bias=True)`. Measure: the largest
  principal angle (`scipy.linalg.subspace_angles`), and the ratio gap.
- KRLS: `sklearn.kernel_ridge.KernelRidge(alpha=r, sample_weight=w)` on the
  atoms (atoms-only form) and on the window rows (window form);
  `sklearn.kernel_approximation.Nystroem(n_components=m)` on the atoms then
  `Ridge(alpha=r, fit_intercept=False, sample_weight=w)` on every learned row
  (the full form's batch definition on the same dictionary); batch
  `KernelRidge` on all rows for the held-out error. Measure: the largest
  absolute prediction difference on 500 held-out rows.
