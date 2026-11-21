# Study 544: do the streaming tail measures agree with independent oracles?

Status: parts 1 and 2 of 3 complete (oracle search; P-squared port against river;
expected shortfall from a DDSketch-style histogram; t-digest rank error against
pytdigest, crick and tdigest). Missing, for part 3: the comparison with rheon's own
`EwQuantile` and tail outputs.

Python only. No Rust or existing source changed; no data file committed (every
stream is generated from a seed). Script: `scripts/study_544.py`.

Environment: a separate venv, `~/study544-venv` (CPython 3.12.3, built with
`uv venv`), not the project's `.venv`. Versions below are read from the
installed packages (`importlib.metadata`, `uv pip show`).

## 1. Oracle search

| Oracle | PyPI name | Installed | Licence (package metadata) | URL | Installed? |
| --- | --- | --- | --- | --- | --- |
| P-squared | `river` (`stats.Quantile`) | 0.27.0 | BSD-3-Clause | https://riverml.xyz/ , https://github.com/online-ml/river/ | yes |
| t-digest (C++ binding) | `pytdigest` | 0.1.4 | Apache Software License (classifier) | https://github.com/protivinsky/pytdigest | yes |
| t-digest (pure Python) | `tdigest` | 0.5.2.2 | MIT | https://github.com/CamDavidsonPilon/tdigest | yes |
| DDSketch | `ddsketch` (Datadog, `sketches-py`) | 3.0.1 | Apache Software License (classifier) | http://github.com/datadog/sketches-py | yes |
| exact quantile, tail mean | `numpy` | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org | yes |
| (extra) t-digest, Cython | `crick` | 0.0.8 | BSD-3-Clause | https://github.com/dask/crick | yes |
| (extra) KLL, Apache DataSketches | `datasketches` | 5.2.0 | Apache License 2.0 | http://datasketches.apache.org | yes |
| (extra) | `scipy` | 1.18.1 | BSD License (classifier) | https://scipy.org | yes (river dependency) |

Last-release dates, from the PyPI JSON API on 2026-11-21: river 0.27.0
(2026-11-13), pytdigest 0.1.4 (2023-02-03), tdigest 0.5.2.2 (2019-05-07),
ddsketch 3.0.1 (2024-04-01), crick 0.0.8 (2024-12-23), datasketches 5.2.0
(2025-03-01). `tdigest` has been unmaintained since 2019; `pytdigest` since
2023.

R through rpy2: not named in the brief as available, and not available.
`uv pip install rpy2` failed (its `rpy2-rinterface` build needs an R
installation); `R` and `Rscript` are not on the PATH. R was not installed.

### Other streaming VaR or expected-shortfall libraries

A web search and the PyPI JSON API found no package that offers a streaming
(one-pass, bounded-memory) VaR or expected shortfall. What exists is batch,
over a stored return series:

| Name | Latest | Licence | Last release | What it is |
| --- | --- | --- | --- | --- |
| `pyriskmgmt` | 1.1.3 | MIT | 2023-09-15 | VaR and ES, quantile or bootstrap, EVT; batch |
| `varpy` | 2.0.1 | MIT (classifier) | 2025-05-18 | VaR and CVaR from fitted distributions, GARCH; batch |
| `var` (ibaris/VaR) | 2024.3.0 | not stated | 2024-03-08 | historical, parametric, Monte Carlo VaR; batch |
| `rmpy` | 1.2.4 | MIT | 2023-05-23 | risk measures; batch |
| `empyrical-reloaded` | 0.7.3 | Apache-2.0 | 2026-11-16 | performance and risk statistics (historical VaR, CVaR); batch |
| `pyfolio-reloaded` | 0.10.2 | Apache-2.0 | 2026-11-02 | portfolio tear sheets; batch |
| `arch` | 9.0.0 | NCSA | 2026-11-11 | ARCH/GARCH models, forecast quantiles; batch |

Streaming building blocks only (no VaR/ES): `opstats` 1.2.0 (GPLv3, 2023-12),
`numpy-onlinestats` 0.1.1 (GPL, 2025-04), `streaming-statistics` 1.0.2
(2024-07), `livestats` 1.0 (licence UNKNOWN, 2013; the P-squared code river's
was inspired by), `hdrhistogram` 0.10.8 (2026-10), `tdigest-rs` 1.0.1
(Apache-2.0, 2026-11-19; a Rust t-digest with a Polars binding, not
installed). Expected shortfall from a stream is therefore built from a streaming
quantile (P-squared, t-digest, DDSketch) plus a tail mean; the oracle for ES
is numpy over the stored stream, and the third-party sketches are oracles for
the quantile only. river has no VaR or ES either (its `stats` has `Quantile`,
`RollingQuantile`, `IQR`).

## 2. P-squared (Jain and Chlamtac 1985)

For the quantile `p`, keep five markers: heights `q_1..q_5` (estimates of the
min, the `p/2`, `p`, `(1+p)/2` quantiles and the max), integer positions
`n_1..n_5` (the rank each height is believed to have), desired positions
`n'_1..n'_5` and fixed increments `dn'`:

```text
init (after 5 observations, sorted):  q_i = x_(i),  n_i = i
n'  = [1, 1 + 2p, 1 + 4p, 3 + 2p, 5]         dn' = [0, p/2, p, (1+p)/2, 1]
```

For each new `x`:

1. Find the cell `k`: if `x < q_1`, set `q_1 = x`, `k = 1`; else the `k` in
   1..4 with `q_k <= x < q_{k+1}`; if `x >= q_5`, set `q_5 = max(q_5, x)`,
   `k = 4`.
2. Positions `n_i += 1` for `i = k+1..5`; desired `n'_i += dn'_i` for all `i`.
3. For `i = 2, 3, 4`, let `d_i = n'_i - n_i`. If `d_i >= 1` and
   `n_{i+1} - n_i > 1`, or `d_i <= -1` and `n_{i-1} - n_i < -1`, set
   `s = sign(d_i)` and try the parabolic (piecewise-parabolic, "P-squared")
   prediction

   ```text
   q'_i = q_i + s / (n_{i+1} - n_{i-1}) *
          [ (n_i - n_{i-1} + s)(q_{i+1} - q_i)/(n_{i+1} - n_i)
          + (n_{i+1} - n_i - s)(q_i - q_{i-1})/(n_i - n_{i-1}) ]
   ```

   If `q_{i-1} < q'_i < q_{i+1}` it replaces `q_i`; otherwise use the linear
   prediction `q_i + s (q_{i+s} - q_i) / (n_{i+s} - n_i)`. Then `n_i += s`.
4. The estimate of the `p`-quantile is `q_3`.

The port is `P2` in `scripts/study_544.py` (numpy float64 arrays and scalars,
the arithmetic in the order written above).

River 0.27.0 runs this in Rust (`river._river_rust.stats.RsQuantile`), so its
source is not in the wheel. The last pure-Python version (river commit
`40c3190`, linked from river's own docstring) was read as the reference for
the order of operations. Where river differs from the paper, the port follows
river: the first five observations are only appended (they are sorted on the
sixth update, not at the fifth), and `get()` before the sixth update returns
the order statistic at index `int(min(max(m-1, 0), m*p))` of the `m` values
seen so far.

### Result against river 0.27.0

Four distributions, seeded `default_rng([seed, dist_index, 544])`: normal,
lognormal(0, 1), Student-t with 3 degrees of freedom, classical Pareto with
shape 2.5 (`pareto(2.5) + 1`). Quantiles 0.05, 0.5, 0.95; seeds 0..49.
Every comparison is exact float equality of `stats.Quantile(p).get()` and the
port's `get()`, after every update ("trajectory").

| Run | n per stream | Streams | Values compared | Mismatches | Largest difference |
| --- | --- | --- | --- | --- | --- |
| trajectory | 1,000 | 600 | 600,000 | 0 | 0 ulps |
| trajectory | 20,000 | 600 | 12,000,000 | 0 | 0 ulps |

The port is bit-equal to river on all 12.6 million comparisons, including
the first five updates (both `None`, then the same order statistic), so river's
Rust code follows the pure-Python order of operations. Negative control: a
port with `p` perturbed by `1e-12` disagrees with river, so the comparison
is sensitive to the last bits. Accuracy is not the question here (P-squared
at n = 5000, Student-t(3), p = 0.05: -2.680 against the exact -2.592), but
note that P-squared's error is not bounded, which parts 2 and 3 measure.

Reproduce: `python scripts/study_544.py p2 --n 1000 --seeds 50 --trajectory`
(3 s) and `--n 20000` (about 40 s, run under `nohup`).

## 3. P-squared state per quantile

All values are 8 bytes (f64; the positions are integers in the paper and
fit in an i64).

| Part | Count | Bytes |
| --- | --- | --- |
| marker heights `q` | 5 | 40 |
| marker positions `n` | 5 | 40 |
| desired positions `n'` | 5 | 40 |
| increments `dn'` | 5 | 40 |
| `p` and observation count | 2 | 16 |
| **total** | 22 | **176** |

Markers and positions alone are 80 bytes; `n'` and `dn'` are re-derivable
from `p` and the count, so a minimal serialization is 96 bytes (heights,
positions, `p`, count). The state is bounded by the algorithm, independent of
the stream length. For comparison, rheon's current `EwQuantile` (an
exponentially weighted DDSketch) replaced P-squared because P-squared never
forgets (`crates/rheon-core/src/stats.rs`); its state is the decayed bucket
set, bounded by the value range, not a fixed size. Parts 2 and 3 compare it.

## 4. Expected shortfall from a DDSketch-style histogram (part 2)

Script: `scripts/study_544_tails.py` (`es`, `decay`, `window`, `buckets`, `tdigest`), run in
`~/study544-venv` (the project `.venv` was not used). Streams from `study_544.stream`, seeds 0..19.

### 4.1 The sketch and the bound

`alpha = tanh(1/128) = 0.0078123` (the `EwQuantile` setting), `gamma = (1 + alpha)/(1 - alpha)
= 1.0157477`. A value `x > 0` goes to bucket `i = ceil(log(x)/log(gamma))`, which covers
`(gamma^(i-1), gamma^i]`; the representative is `r_i = 2 gamma^i/(gamma + 1)`. Negative values
use a second store on `|x|` (representative `-r_i`), zero has a count, so the buckets sort in
value order. Derivation: `r_i/gamma^i = 2/(gamma+1) = 1 - alpha` and `r_i/gamma^(i-1) =
2 gamma/(gamma+1) = 1 + alpha`, so `r_i` is within `alpha` relative of both ends of the bucket
and `|r_i - x| <= alpha x` for every `x` in it.

Tail at level `q` (mass `m = q N`, or `q * sum(w)` weighted; lower tail = the `m` smallest,
upper tail = the `m` largest, computed as the lower tail of `-x` in a separate sketch). The
VaR is the representative of the bucket where the cumulative mass first reaches `m`. ES is
`(sum over whole buckets below the VaR bucket of mass * r + (m - mass below) * r_VaR) / m`: the
VaR bucket contributes only the mass the tail still needs ("partial mass"). The exact side is
numpy on the stored values with the same rule (the `m` smallest, fractional weight at the VaR
value). Pair the sorted stored values with the sorted sketch slots: each pair lies in one
bucket (bucket order is value order and the masses match), so `|ES_hat - ES| <= (1/m) sum
alpha |x| = alpha * mean(|x|)` over the tail, which is `alpha |ES|` when the tail has one sign
(true of all four levels here: every lower 0.01/0.05 and upper 0.95/0.99 tail of these four
distributions is one-signed). The whole-bucket variant (take the entire VaR bucket) has no such
bound; it is shown for contrast. Float rounding adds about 1e-15.

### 4.2 ES against the exact tail mean (counts; 20 seeds x 4 levels)

Largest relative gap `|ES_hat - ES|/|ES|` over 20 seeds and the four levels (upper 0.95 means
the mean of the largest 5%), with the largest `gap / (alpha mean|x|)` (the bound is 1; no run
exceeded it, 0 violations in 640 runs). `alpha` = 7.81e-03.

| dist | N = 10,000 gap | gap/bound | N = 100,000 gap | gap/bound |
| --- | --- | --- | --- | --- |
| normal | 1.32e-03 | 0.17 | 2.51e-04 | 0.03 |
| lognormal | 1.32e-03 | 0.17 | 3.75e-04 | 0.05 |
| Student-t(3) | 1.38e-03 | 0.18 | 3.76e-04 | 0.05 |
| Pareto(2.5) | 6.31e-03 | 0.81 | 5.95e-03 | 0.76 |

Pareto(2.5) is the outlier, at the lower 0.01 level only (0.0063 and 0.0059; its 0.05 lower
gap is 1.5e-03, upper 0.95 and 0.99 are 5.4e-04 and 8.8e-04 at N = 10,000). Its support starts
at `x_m = 1`, the right edge of bucket 0 `(1/gamma, 1]`, whose representative is `1 - alpha`:
the lowest 1% of the values all sit just above 1, so every one is read `alpha` low and the
errors do not cancel. That is the bound nearly attained (0.76 to 0.81), by a hard edge at a
bucket boundary, not by the heavy tail. The heavy upper tails behave like the light ones:
t(3) upper 0.99 is 1.2e-03 (N = 10,000) and 3.3e-04 (N = 100,000); Pareto upper 0.99 is
8.8e-04 and 4.5e-04. The gap falls with N for the smooth tails and does not for a tail pressed
on a bucket edge (Pareto's lower 0.01: 6.3e-03 and 5.9e-03).

VaR (the bucket representative) against the exact VaR: the largest relative error is 7.8e-03
over all runs (below `alpha`, as it must be). The whole-bucket ES variant reaches 3.0e-02 at
N = 10,000 and 2.0e-02 (t(3) upper 0.99) at N = 100,000, up to about 3.8 `alpha`; the partial
mass is what keeps ES inside the bound.

### 4.3 Decay (lambda = 0.999) and a window (W = 5,000)

Decay: the weight of row `t` is `lambda^(N-1-t)` on the row clock, each bucket holds its decayed
weight, the tail mass is `q * sum(w)` (`n_eff` = 1,000), the exact side is numpy on the decayed
weights with the same fractional rule. The batch weights equal the streaming recursion (every
bucket times lambda, then +1) to 1e-14 relative (N = 10,000, seed 0, all four distributions).
Largest gap over 20 seeds and four levels: N = 10,000: normal 2.27e-03, lognormal 2.64e-03,
t(3) 2.13e-03, Pareto 6.86e-03; N = 100,000: normal 1.96e-03, lognormal 2.08e-03, t(3) 2.39e-03,
Pareto 6.36e-03; gap/bound at most 0.34 except Pareto's lower 0.01 tail (0.81 to 0.88, the edge
effect above). No violation of the bound in 640 runs. The gaps are larger than the unweighted
ones because the effective sample is 1,000 rows, so the 1% tail holds about 10 rows' worth of
mass in few buckets and fewer errors cancel.

Window: a rolling count store of the last 5,000 rows: add the new row's bucket, subtract the
bucket of the row that left. Subtracting an integer count is exact, so the window store equals
the bincount of the window: checked at every one of the 20 queries per stream (every 5,000 rows
from row 5,000) for each stream, level and seed, 6,400 of 6,400 checks equal. ES against numpy
on exactly those 5,000 rows, 400 queries per cell: largest gap normal 2.55e-03, lognormal
1.94e-03, t(3) 2.43e-03, Pareto 6.56e-03 (Pareto lower 0.01, gap/bound 0.84); gap/bound at most
0.38 outside that cell; 0 violations in 6,400 runs. Largest live-bucket count in a window: normal 687,
lognormal 377, t(3) 792, Pareto 163 (12,712 bytes at most, `live * 16 + 40`).

### 4.4 Buckets and state bytes

State of one signed sketch: live buckets (both stores and the zero count) times 16 bytes (an
i64 key and an f64 count) plus a 40-byte header (alpha, zero count, total weight, two store
lengths; the header size is my choice). The mirrored sketch for the upper tail holds the same
buckets with the sign flipped, so one signed sketch serves both tails; bytes are for one.
Live buckets, 20 seeds (2 at N = 1,000,000):

| dist | N = 10,000 mean (max) | N = 100,000 mean (max) | bytes at N = 100,000 (max) | N = 1,000,000 mean | buckets holding the top 1% of mass, N = 100,000 |
| --- | --- | --- | --- | --- | --- |
| normal | 758 (775) | 1,078 (1,098) | 17,608 | 1,390 | 36 |
| lognormal | 389 (400) | 479 (489) | 7,864 | 551 | 91 |
| Student-t(3) | 879 (909) | 1,277 (1,292) | 20,712 | 1,671 | 98 |
| Pareto(2.5) | 167 (176) | 227 (235) | 3,800 | 286 | 110 |

One bucket covers `ln(gamma) = 0.0156` of log-range of `|x|`, so the count is the log-range of
the data over that width, not a function of the stream's length: it grows by about 300 to 400 a
decade of N for the normal and t(3), 50 to 60 for lognormal and Pareto, and reaches at most
1,677 buckets (27 KB) at N = 10^6. The normal's and t(3)'s counts include the buckets of small
`|x|` toward 0 (both signs), which the positive-only lognormal and Pareto lack. Heavy tails (t(3) and Pareto) put 2.7 to 3 times as many buckets in the top 1%
of mass as the normal does at N = 100,000, 98 and 110 against 36, each holding a few values
(N = 10^6: 149 and 169 against 46); the ES gap there is still inside the bound, see 4.2.

Oracle: `ddsketch` 3.0.1 (`LogarithmicMapping(alpha)`, unbounded `DenseStore`) fed the same
streams (5 seeds x 4 distributions, N = 100,000): its bucket key and count agree with this
sketch's on all 15,330 buckets, none mismatching, and its quantile is within `alpha` of the
exact quantile (largest 7.80e-03).

## 5. t-digest rank error (part 2)

Rank error `|F(q_hat) - p|`, `F` the empirical CDF of the stream, N = 100,000, 20 seeds, four
distributions, `p` in 0.01, 0.05, 0.5, 0.95, 0.99. The digests: `mine`, a Dunning-Ertl merging
digest (`MergingDigest` in the script: scale function `k1 = delta/(2 pi) asin(2q - 1)`, a buffer
of 8 delta points merged into the centroids, Dunning's linear interpolation between centroid
centres, the end centroids interpolated to the stored min and max); `pytdigest` 0.1.4
(`TDigest.compute`, `inverse_cdf`); `crick` 0.0.8 (`TDigest(compression)`, `quantile`);
`tdigest` 0.5.2.2 (`TDigest(delta = 1/compression)`, `percentile`; an older tree-based
algorithm, not the merging digest, 3 seeds only because it takes 3 to 6 s a stream).

| compression | impl | largest rank error | at p = | centroids (mean, max) | state bytes (max, centroids x 16) |
| --- | --- | --- | --- | --- | --- |
| 100 | mine | 2.41e-03 | 0.5 | 57.3, 61 | 976 |
| 100 | pytdigest | 1.02e-02 | 0.5 (lognormal, Pareto) | 61.5, 64 | 1,024 |
| 100 | crick | 1.44e-03 | 0.5 | 128.3, 134 | 2,144 |
| 100 | tdigest | 1.70e-04 | 0.5 | 833.5, 886 | 14,176 |
| 200 | mine | 1.20e-03 | 0.5 | 113.9, 118 | 1,888 |
| 200 | pytdigest | 2.90e-03 | 0.5 | 102.6, 107 | 1,712 |
| 200 | crick | 6.80e-04 | 0.95 | 254.9, 261 | 4,176 |
| 200 | tdigest | 9.00e-05 | 0.5 | 1,572.6, 1,659 | 26,544 |

Per `p` the largest errors (mine, compression 100): 9.2e-04 at 0.01, 1.07e-03 at 0.05, 2.41e-03 at 0.5,
9.8e-04 at 0.95, 8.5e-04 at 0.99 (most digests have their largest error at the median, where the scale
function allows the widest centroids; `crick` at compression 200 peaks at 0.95). The tail ranks
are a factor of 2 to 3 finer than the median's for `mine`.

Agreement. The libraries do not agree with one another: the ranks of the quantiles from
`pytdigest` and `crick` differ by up to 1.03e-02 at compression 100 and 2.89e-03 at 200
(`pytdigest` against `mine`: 9.9e-03 and 2.6e-03), because they hold different numbers of
centroids for the same compression (61, 128 and 833 at compression 100) and interpolate
differently; `pytdigest`'s median is the worst of the four (1.0e-02 on lognormal and Pareto).
All four are within about 1e-2 of the exact rank; none is an exact oracle for the others, and
the error is not bounded by a stated number in any of them. The state of `crick` at compression
100 is 2.1 KB against 1.0 KB for `pytdigest`; `mine` at compression 100 holds 57 centroids.

## 6. Reading of parts 4 and 5

- The DDSketch ES has a derivable, deterministic error bound (`alpha` of the tail's mean
  magnitude, 0.78% here), met in every run with the partial-mass rule, with state of 4 to 27 KB
  bounded by the data's log-range. Decay and a window do not change the bound (a window removes
  counts exactly, so it is the same sketch on the last W rows).
- Its typical gap is 2% to 18% of the bound; the case that approaches it is a tail pressed on
  a bucket edge (Pareto's lower 0.01, 76% to 88%), not the heavy tail.
- The t-digest's rank error is a measurement, not a bound: 1e-3 to 1e-2 at compression 100 and
  7e-4 to 3e-3 at 200 (the largest at the median), in about 1 KB to 4 KB for the merging
  implementations, and the installed implementations do not agree with one another.

## Open questions

- river's Rust source was not readable offline (the wheel is compiled and
  GitHub access for the source repository was not enabled in this session);
  the bit-equality above is empirical, and the order of operations came from
  the last pure-Python version. If river's Rust code later diverges, the
  trajectory test will show it.
- No R oracle (R absent).
- Part 2: the header size of the sketch (40 bytes) and the buffer size of `mine` (8 delta) are
  choices of this study; `tdigest` was run on 3 seeds only.
