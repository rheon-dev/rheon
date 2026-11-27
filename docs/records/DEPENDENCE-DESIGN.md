<!-- kind: record -->
# Nonlinear dependence (`dependence`) — design and measurements

Status (2026-11-27): built as `dependence` (`crates/rheon-core/src/dependence.rs`,
`crates/rheon-polars/src/spec/dependence.rs`, `crates/rheon-polars/src/bank/dependence.rs`,
`python/rheon/_spec_dependence.py`, `ModelBank.dependence`).

## 1. Scope and API surface

`rp.spec.dependence` adds a dedicated model kind alongside `marginal` (leaving
released 2.0.0 `marginal` untouched) for online nonlinear dependence between
every `(feature, target)` pair:

- **Measures** (`measures`, default `["mi"]`):
  - `"mi"`: plug-in binned mutual information in nats, computed in ratio form
    with `libm::log` (`(w_ab / w_a) * (N / w_b)`, falling back to log-difference
    only if the product overflows or underflows at extreme weight scales) and
    clamped at `0.0` against rounding noise.
  - `"mi_mm"`: Miller-Madow bias-corrected MI subtracting
    `(B_xy - B_x - B_y + 1) / (2 * N)`. Defined for unit weights without decay;
    returns NaN/null whenever the active sample holds any positive weight other
    than `1.0`, and recovers automatically once non-unit rows leave a window.
  - `"spearman"`, `"kendall"`, `"xi"`: exact rank correlations over a window
    (`window_rows` or `window_size` with `closed = "right" | "both"`). Refused
    beside finite decay or without a window.
- **Dual output surface:**
  - Per-row struct fields (out-of-sample, before learning the row):
    `<measure>_<feature>_<target>` ordered by `(measures, features, targets)`,
    plus `weight_sum`, optional `weight_<target>` (`emit_target_weight`),
    `settled_frac`, and `withheld_reason` (`"below_warm_rows"` during MI bin
    warm-up).
  - `ModelBank.dependence(spec, group=None)`: tidy Polars DataFrame with columns
    `spec`, `group`, `instance`, `half_life`, `lam`, `feature`, `target`,
    `weight_sum`, and one column per active measure in `measures` order. Always
    computes fresh exact Spearman and Chatterjee xi from the current window even
    when `rank_every` or `rank_every_rows` throttles per-row output.

## 2. Algorithms, numerics, and invariants

- **Binning & warm-up:** Fixed edges (`bin_edges`, `target_bin_edges`, as dict
  or list-of-lists) skip warm-up completely. Learned edges (`bin_rule = "quantile"`
  or `"fixed"`) buffer `bin_warm_rows` (default `min(1000, window_rows).max(2)`
  under `window_rows`, else `1000`) or `bin_warm_span` clock units, then freeze
  edges once and replay every buffered row into the joint tables (trimmed to the
  active window when windowed).
- **Windowed MI exactness:** Each cell keeps a `u32` row counter beside its
  `f64` weight sum. When an evicted row clears a cell's count to zero, the
  weight is zeroed exactly; when `w_out > 1e8 * remaining`, the pair table is
  resummed from the active ring so an extreme `1e100` weight leaves no residual
  cancellation error once it exits the window.
- **Incremental Kendall & periodic Spearman / xi:** Kendall's tau-b maintains
  exact integer counts `(n, n_c, n_d, ties_x, ties_y, ties_xy)` in `O(W)` per
  entering/leaving row per pair. Spearman and Chatterjee xi compute average
  ranks and upper/lower tie ranks over `O(W log W)` scratch buffers (zero steady-state
  allocations per row), with ties in `x` broken deterministically by arrival
  order (oldest first). Periodic cadence (`rank_every_rows` / `rank_every`)
  performs its first refresh as soon as 2 positive-weight rows exist and restarts
  both row and clock cadences from that row, so `rank_every = D` and
  `rank_every_rows = D` agree to the bit on every unit clock.
- **Decay & zero-weight rules:** Under decay, subnormal weights below
  `f64::MIN_POSITIVE` are zeroed so forgotten histories do not produce scale-invariant
  stale MI. Zero-weight rows touch no clock, no window, and no warm-up.
- **Bounded state & atomic `HeldCap`:** Fixed state sizes are checked against
  `max_state_bytes` (default 256 KiB; ceiling 64 MiB) at build time. Dynamic
  windows (`window_size` or `bin_warm_span`) check projected bytes before any
  state mutation, emit a `ReadinessWarning` once past half the cap, and refuse
  atomically before exceeding the cap.

## 3. Verification and oracles

- `sklearn.metrics.mutual_info_score`: max gap `< 5e-16` across 50 seeds
  (2,000-row bivariate Gaussians, `rho = 0.8`, 8x8 octile bins).
- Plain-Python Miller-Madow & Gauss-Legendre binned population MI: exact to
  `< 1e-13`; Miller-Madow reduces finite-sample positive bias across all 50 seeds.
- Decayed MI vs `scipy.stats.entropy` / longhand: exact to `< 1e-12`.
- `scipy.stats.spearmanr`, `scipy.stats.kendalltau(..., variant="b")`, and
  `scipy.stats.chatterjeexi`: max gap `< 6e-17` across 50 seeds (noisy
  parabolas, `W = 200`), plus exact arrival-order tie longhand for `xi`.
- Frozen state fixtures at schema 72 cover decayed MI, warming MI, row-windowed
  MI + ranks, clock-windowed MI + ranks, and full bank save/load + `ModelBank.dependence()`.

## 4. Arm64 release timings, state bytes, and scaling

Measured on macOS arm64 (`--release`, single thread, ABBA interleaved across 4
passes, `k = 10`, `T = 1`, `16 × 16` bins, `W = 1,000`, zero steady-state heap
allocations per row):

| configuration (`k = 10, T = 1`) | min ns/row | med ns/row | exact `state_bytes()` |
|---|---:|---:|---:|
| MI-only decayed (`16 × 16`, `half_life = 100`) | 9,465 | 9,521 | 23,840 B (23.3 KiB) |
| MI + Miller-Madow windowed (`16 × 16`, `W = 1,000`) | 9,262 | 9,289 | 64,080 B (62.6 KiB) |
| Ranks-only (`spearman + kendall + xi`, `W = 1,000`, `rank_every_rows = 10`) | 44,991 | 45,317 | 98,168 B (95.9 KiB) |
| Ranks-only (`spearman + kendall + xi`, `W = 1,000`, `rank_every_rows = 1`) | 260,490 | 262,696 | 98,168 B (95.9 KiB) |
| Combined MI + MM + ranks (`16 × 16`, `W = 1,000`, `rank_every_rows = 10`) | 54,544 | 54,666 | 130,560 B (127.5 KiB) |
| Combined MI + MM + ranks (`16 × 16`, `W = 1,000`, `rank_every_rows = 1`) | 269,280 | 269,771 | 130,560 B (127.5 KiB) |

Scaling (`rheon-bench --release --test scaling`, 8× dimension step):

- `dependence.k` (`2 → 16`, `O(n)` work and state): CPU `3.40×`, allocs `1.00× (0 → 0)`, saved `5.30×`, heap `3.97×`.
- `dependence.window` (`50 → 400`, `O(n)` work and state): CPU `4.85×`, allocs `1.00× (0 → 0)`, saved `4.27×`, heap `2.84×`.
- `dependence.history` (`1,000 → 8,000`, `O(1)` work and state): CPU `1.00×`, allocs `1.00× (0 → 0)`, saved `1.00×`, heap `1.00×`.

