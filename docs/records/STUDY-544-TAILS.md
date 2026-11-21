# Study 544: do the streaming tail measures agree with independent oracles?

Status: complete (parts 1 to 3). Part 1: oracle search and the P-squared port against river.
Part 2: expected shortfall from a DDSketch-style histogram, and t-digest rank error. Part 3:
one entry point, the t-digest comparison on 20 seeds for every library, proposed tolerances,
the oracle table, the state of each online form, and the open items below.

Python only. No Rust or existing source changed; no data file committed (every stream is
generated from a seed). Script: `scripts/study_544.py`, one entry point:

```text
python scripts/study_544.py psquared [--n 1000] [--seeds 50] [--trajectory]
python scripts/study_544.py tails    [--seeds 20] [--only es|decay|window|buckets]
python scripts/study_544.py tdigest  [--seeds 20] [--td-seeds 20] [--workers 4]
python scripts/study_544.py all --quick      # every part, small; checks the tolerances; 8 s
python scripts/study_544.py oracles | state  # package table | bytes per online form
```

Each part checks its result against the tolerances of section 7 (the constants are at the top
of the script) and exits 1 on a failure. `--quick` is 3 seeds of P-squared at n = 1,000, 2 seeds
of the tail measures at N = 10,000 (the window at 20,000, the bucket oracle at 100,000) and
1 seed of the t-digests at N = 100,000: 8 seconds on 4 cores.

Environment: a separate venv, `~/study544-venv` (CPython 3.12.3, built with `uv venv`), not the
project's `.venv`. Versions below are read from the installed packages (`importlib.metadata`).
The full runs of this part: `tails` 32 s, `psquared` 57 s (n = 1,000 and 20,000), `tdigest`
on 20 seeds 175 s (3 worker processes).

Reader-facing note: the study's own implementations are named by what they are (the P-squared
port `P2`, the merging digest `MergingDigest`, shown as `mine` in the script's output).

## 1. Oracle search

| Oracle | PyPI name | Installed | Licence (package metadata) | URL |
| --- | --- | --- | --- | --- |
| P-squared | `river` (`stats.Quantile`) | 0.27.0 | BSD-3-Clause | https://riverml.xyz/ , https://github.com/online-ml/river/ |
| t-digest (C++ binding) | `pytdigest` | 0.1.4 | Apache Software License (classifier) | https://github.com/protivinsky/pytdigest |
| t-digest (pure Python) | `tdigest` | 0.5.2.2 | MIT | https://github.com/CamDavidsonPilon/tdigest |
| DDSketch | `ddsketch` (Datadog, `sketches-py`) | 3.0.1 | Apache Software License (classifier) | http://github.com/datadog/sketches-py |
| exact quantile, tail mean | `numpy` | 2.5.5 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| (extra) t-digest, Cython | `crick` | 0.0.8 | BSD-3-Clause | https://github.com/dask/crick |
| (extra) KLL, Apache DataSketches | `datasketches` | 5.2.0 | Apache License 2.0 | http://datasketches.apache.org (installed; not compared) |
| (extra) | `scipy` | 1.18.1 | BSD License (classifier) | https://scipy.org (river dependency; not compared) |

Last-release dates, from the PyPI JSON API on 2026-11-21: river 0.27.0 (2026-11-13), pytdigest
0.1.4 (2023-02-03), tdigest 0.5.2.2 (2019-05-07), ddsketch 3.0.1 (2024-04-01), crick 0.0.8
(2024-12-23), datasketches 5.2.0 (2025-03-01). `tdigest` has been unmaintained since 2019;
`pytdigest` since 2023.

R through rpy2: not available (`uv pip install rpy2` failed, its `rpy2-rinterface` build needs
an R installation; `R` and `Rscript` are not on the PATH). R was not installed.

### Other streaming VaR or expected-shortfall libraries

A web search and the PyPI JSON API found no package that offers a streaming (one-pass,
bounded-memory) VaR or expected shortfall. What exists is batch, over a stored return series:

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
`numpy-onlinestats` 0.1.1 (GPL, 2025-04), `streaming-statistics` 1.0.2 (2024-07), `livestats`
1.0 (licence UNKNOWN, 2013; the P-squared code river's was inspired by), `hdrhistogram` 0.10.8
(2026-10), `tdigest-rs` 1.0.1 (Apache-2.0, 2026-11-19; a Rust t-digest with a Polars binding,
not installed). Expected shortfall from a stream is therefore built from a streaming quantile
(P-squared, t-digest, DDSketch) plus a tail mean; the oracle for ES is numpy over the stored
stream, and the third-party sketches are oracles for the quantile only. river has no VaR or ES
either (its `stats` has `Quantile`, `RollingQuantile`, `IQR`).

### The oracle table: each oracle against its definition and against the others

"Definition" is the published algorithm the oracle claims to implement. The gap to the
definition is measured against a definition written from the paper in this study (`P2`, the
bucket rule of section 3.2, `MergingDigest`) or against exact numpy values. N = 100,000 and
20 seeds x 4 distributions for the t-digests; sections 4 to 6 give the runs.

| Oracle (version, licence) | Definition compared against | Largest gap to the definition | Largest gap to the other oracles |
| --- | --- | --- | --- |
| river 0.27.0 (BSD-3-Clause) | Jain and Chlamtac 1985, `P2` port | 0 ulps over 12.6 million values | no other P-squared oracle exists |
| ddsketch 3.0.1 (Apache) | Masson, Rim and Lee 2019, the bucket rule of 3.2 | 0 of 15,330 bucket counts differ; quantile within alpha of exact (7.799e-3 against alpha = 7.812e-3) | the study sketch is identical to it, so 0 |
| numpy 2.5.5 (BSD-3-Clause and others) | the empirical quantile (`inverted_cdf`) and the tail mean of the `m` smallest values | 0 (it is the reference) | the reference for all of them |
| crick 0.0.8 (BSD-3-Clause) | Dunning and Ertl 2019, the k1 size rule (centroid k-span <= 1) | k-span 0.50 (rule met); rank error 1.44e-3 (compression 100), 6.8e-4 (200) | 1.03e-2 / 2.9e-3 from pytdigest; 1.5e-3 / 6.7e-4 from tdigest |
| pytdigest 0.1.4 (Apache) | the same k1 size rule | k-span 5.49 at compression 100 and 3.79 at 200 (rule broken); rank error 1.02e-2 / 2.9e-3 | 1.03e-2 / 2.9e-3 from crick; 1.02e-2 / 2.85e-3 from tdigest |
| tdigest 0.5.2.2 (MIT) | Dunning's first (tree-based, 2013) t-digest, not the 2019 merging digest | not a k1 digest (k-span 0.31, 14.6 KB at compression 100); rank error 2.7e-4 / 1.7e-4 | 1.5e-3 / 6.7e-4 from crick; 1.02e-2 / 2.85e-3 from pytdigest |
| `MergingDigest` (this study) | the same k1 size rule | k-span 1.00 by construction; rank error 2.41e-3 / 1.20e-3 | 2.6e-3 / 9.7e-4 from crick; 9.9e-3 / 2.6e-3 from pytdigest; 2.3e-3 / 1.2e-3 from tdigest |

The gaps between t-digest implementations are differences in the rank of the value they
return for the same quantile, on the same stream (compression 100 / compression 200).

## 2. P-squared (Jain and Chlamtac 1985)

For the quantile `p`, keep five markers: heights `q_1..q_5` (estimates of the min, the `p/2`,
`p`, `(1+p)/2` quantiles and the max), integer positions `n_1..n_5` (the rank each height is
believed to have), desired positions `n'_1..n'_5` and fixed increments `dn'`:

```text
init (after 5 observations, sorted):  q_i = x_(i),  n_i = i
n'  = [1, 1 + 2p, 1 + 4p, 3 + 2p, 5]         dn' = [0, p/2, p, (1+p)/2, 1]
```

For each new `x`:

1. Find the cell `k`: if `x < q_1`, set `q_1 = x`, `k = 1`; else the `k` in 1..4 with
   `q_k <= x < q_{k+1}`; if `x >= q_5`, set `q_5 = max(q_5, x)`, `k = 4`.
2. Positions `n_i += 1` for `i = k+1..5`; desired `n'_i += dn'_i` for all `i`.
3. For `i = 2, 3, 4`, let `d_i = n'_i - n_i`. If `d_i >= 1` and `n_{i+1} - n_i > 1`, or
   `d_i <= -1` and `n_{i-1} - n_i < -1`, set `s = sign(d_i)` and try the parabolic prediction

   ```text
   q'_i = q_i + s / (n_{i+1} - n_{i-1}) *
          [ (n_i - n_{i-1} + s)(q_{i+1} - q_i)/(n_{i+1} - n_i)
          + (n_{i+1} - n_i - s)(q_i - q_{i-1})/(n_i - n_{i-1}) ]
   ```

   If `q_{i-1} < q'_i < q_{i+1}` it replaces `q_i`; otherwise use the linear prediction
   `q_i + s (q_{i+s} - q_i) / (n_{i+s} - n_i)`. Then `n_i += s`.
4. The estimate of the `p`-quantile is `q_3`.

The port is `P2` in `scripts/study_544.py` (numpy float64 arrays and scalars, the arithmetic in
the order written above). River 0.27.0 runs this in Rust (`river._river_rust.stats.RsQuantile`),
so its source is not in the wheel. The last pure-Python version (river commit `40c3190`, linked
from river's own docstring) was read as the reference for the order of operations. Where river
differs from the paper, the port follows river: the first five observations are only appended
(they are sorted on the sixth update, not at the fifth), and `get()` before the sixth update
returns the order statistic at index `int(min(max(m-1, 0), m*p))` of the `m` values seen so far.

P-squared has no error bound: the markers are heuristic and nothing limits how far `q_3` strays
from the quantile (n = 5000, Student-t(3), p = 0.05: -2.680 against the exact -2.592), and it
never forgets old rows.

## 3. The recursions

### 3.1 Decay and window, common to every sketch below

Row `t` carries weight `w_t = 1`. Decay with factor `lambda` on the row clock: each stored mass is
multiplied by `lambda` before the new row is added, so after `N` rows the row `t` has weight
`lambda^(N-1-t)` and the total mass is `n_eff = (1 - lambda^N)/(1 - lambda)`, 1,000 for
`lambda = 0.999` once `N` is large. The tail mass is `q * (total mass)`. A window of `W` rows
adds the new row and subtracts the row that left; the stored counts then equal those of a sketch
built from the last `W` rows. Subtracting an integer count is exact in floating point (counts
are below 2^53), so the window's counts carry no rounding.

### 3.2 DDSketch bucket rule

`alpha = tanh(1/128) = 0.0078123`, `gamma = (1 + alpha)/(1 - alpha) = 1.0157477`. A value `x > 0`
goes to bucket `i = ceil(ln x / ln gamma)`, which covers `(gamma^(i-1), gamma^i]`; the
representative is `r_i = 2 gamma^i/(gamma + 1)`. Negative values use a second store on `|x|`
(representative `-r_i`), and zero has its own count, so the buckets sort in value order. Then
`r_i/gamma^i = 2/(gamma+1) = 1 - alpha` and `r_i/gamma^(i-1) = 2 gamma/(gamma+1) = 1 + alpha`:
the representative is within `alpha` relative of both ends of the bucket, hence
`|r_i - x| <= alpha x` for every `x` in it.

### 3.3 VaR and expected shortfall from the buckets (the partial-mass rule)

Tail at level `q`, mass `m = q * (total mass)`. The lower tail is the `m` smallest values; the
upper tail is the lower tail of `-x`, taken in a mirrored sketch (the same buckets with the sign
flipped, so one signed sketch serves both). Walk the buckets in value order with cumulative mass
`C_j`. The VaR bucket is the first `j` with `C_j >= m`, and VaR is `r_j`. Then

```text
ES = ( sum_{k<j} c_k r_k  +  (m - C_{j-1}) r_j ) / m
```

The VaR bucket contributes only the mass the tail still needs (`m - C_{j-1}`), not its whole
count. The exact reference applies the same rule to the stored values: the `m` smallest, with
the fractional weight at the VaR value.

### 3.4 Merging t-digest (Dunning and Ertl 2019)

Scale function `k(q) = delta/(2 pi) asin(2q - 1)`, which maps `q` in [0, 1] to a range of
`delta/2`. A centroid (mean `c`, weight `w`) may take in its neighbour while the span of `k`
over its cumulative mass stays at most 1. State: centroids `(c_i, w_i)` sorted by mean, plus the
minimum, maximum and total weight. Merge pass (run when a buffer of `8 delta` points is full):
sort the old centroids and the buffer by mean; walk in order with the running centroid `(c, w)`
and the cumulative weight `W` before it, and the limit `q_lim = k^-1(k(W/N) + 1)`. For the next
item `(c', w')`: if `(W + w + w')/N <= q_lim`, merge it (`w += w'`, `c += (c' - c) w'/w`); else
emit `(c, w)`, set `W += w` and start a new centroid. Quantile `p`, `t = p N`: centre of
centroid `i` is `C_i = sum_{k<i} w_k + w_i/2`; for `C_i <= t < C_{i+1}` interpolate linearly
between the means, and interpolate the end centroids to the stored minimum and maximum.

Centroid count: `k` spans `delta/2` and every merged pair of neighbours exceeds a span of 1, so
the digest holds between `delta/2` and `delta` centroids (57 at `delta = 100`, 114 at 200, as
measured).

## 4. Results

Streams: normal, lognormal(0, 1), Student-t(3), classical Pareto with shape 2.5 (`pareto(2.5) + 1`),
seeded `default_rng([seed, dist_index, 544])`.

### 4.1 P-squared against river 0.27.0

Quantiles 0.05, 0.5, 0.95; seeds 0..49; every comparison is exact float equality of
`stats.Quantile(p).get()` and the port's `get()`, after every update ("trajectory").

| Run | n per stream | Streams | Values compared | Mismatches | Largest difference |
| --- | --- | --- | --- | --- | --- |
| trajectory | 1,000 | 600 | 600,000 | 0 | 0 ulps |
| trajectory | 20,000 | 600 | 12,000,000 | 0 | 0 ulps |

Bit-equal on all 12.6 million comparisons, including the first five updates (both `None`, then
the same order statistic). Negative control: a port with `p` perturbed by `1e-12` disagrees with
river, so the comparison is sensitive to the last bits.

### 4.2 ES against the exact tail mean (counts; 20 seeds x 4 levels)

Largest relative gap `|ES_hat - ES|/|ES|` over 20 seeds and the four levels (lower 0.01, 0.05;
upper 0.95, 0.99, the mean of the largest 5% and 1%), and the largest `gap / (alpha mean|x|)`.
`alpha` = 7.81e-03. 0 bound violations in 640 runs.

| dist | N = 10,000 gap | gap/bound | N = 100,000 gap | gap/bound |
| --- | --- | --- | --- | --- |
| normal | 1.32e-03 | 0.17 | 2.51e-04 | 0.03 |
| lognormal | 1.32e-03 | 0.17 | 3.75e-04 | 0.05 |
| Student-t(3) | 1.38e-03 | 0.18 | 3.76e-04 | 0.05 |
| Pareto(2.5) | 6.31e-03 | 0.81 | 5.95e-03 | 0.76 |

Pareto(2.5) is the outlier, at the lower 0.01 level only (its 0.05 lower gap is 1.5e-03, upper
0.95 and 0.99 are 5.4e-04 and 8.8e-04 at N = 10,000). Its support starts at `x_m = 1`, the right
edge of bucket 0 `(1/gamma, 1]`, whose representative is `1 - alpha`: the lowest 1% of the values
all sit just above 1, so every one is read `alpha` low and the errors do not cancel. That is the
bound nearly attained, by a hard edge at a bucket boundary, not by the heavy tail. The heavy upper
tails behave like the light ones (t(3) upper 0.99: 1.2e-03 at N = 10,000, 3.3e-04 at 100,000;
Pareto upper 0.99: 8.8e-04 and 4.5e-04). The gap falls with N for the smooth tails and does not
for a tail pressed on a bucket edge.

VaR (the representative) against the exact VaR: largest relative error 7.8e-03 over all runs,
below `alpha`. The whole-bucket ES variant (the entire VaR bucket's mean) reaches 3.0e-02 at
N = 10,000 and 2.0e-02 at 100,000, up to about 3.8 `alpha`; the partial mass is what keeps ES
inside the bound.

### 4.3 Decay (lambda = 0.999) and a window (W = 5,000)

Decay: the exact side is numpy on the decayed weights with the same fractional rule. The batch
weights equal the streaming recursion (every bucket times lambda, then +1) to 9.5e-15 relative
(N = 10,000, seed 0, all four distributions). Largest gap over 20 seeds and four levels:
N = 10,000: normal 2.27e-03, lognormal 2.64e-03, t(3) 2.13e-03, Pareto 6.86e-03; N = 100,000:
normal 1.95e-03, lognormal 2.08e-03, t(3) 2.39e-03, Pareto 6.36e-03. gap/bound at most 0.34 except
Pareto's lower 0.01 tail (0.81 to 0.88, the edge effect above). No bound violation in 640 runs.
The gaps exceed the unweighted ones because the effective sample is 1,000 rows, so the 1% tail
holds the mass of about 10 rows in few buckets and fewer errors cancel.

Window: a rolling count store of the last 5,000 rows. The window store equals the bincount of
the window at every one of the 20 queries per stream (every 5,000 rows from row 5,000), for each
stream, level and seed: 6,400 of 6,400 checks equal. ES against numpy on exactly those 5,000
rows (400 queries per cell): largest gap normal 2.55e-03, lognormal 1.94e-03, t(3) 2.43e-03,
Pareto 6.56e-03 (Pareto lower 0.01, gap/bound 0.84); gap/bound at most 0.38 elsewhere; 0
violations in 6,400 runs. Largest live-bucket count in a window: normal 687, lognormal 377, t(3)
792, Pareto 163.

### 4.4 Buckets

Live buckets (both stores and the zero count) of one signed sketch, 20 seeds (2 at
N = 1,000,000):

| dist | N = 10,000 mean (max) | N = 100,000 mean (max) | N = 1,000,000 mean | buckets holding the top 1% of mass, N = 100,000 |
| --- | --- | --- | --- | --- |
| normal | 758 (775) | 1,078 (1,098) | 1,390 | 36 |
| lognormal | 389 (400) | 479 (489) | 551 | 91 |
| Student-t(3) | 879 (909) | 1,277 (1,292) | 1,671 | 98 |
| Pareto(2.5) | 167 (176) | 227 (235) | 286 | 110 |

One bucket covers `ln(gamma) = 0.0156` of log-range of `|x|`, so the count is the log-range of
the data over that width, not a function of the stream's length: about 300 to 400 more buckets a
decade of N for the normal and t(3), 50 to 60 for lognormal and Pareto, at most 1,677 at
N = 10^6. Heavy tails put 2.7 to 3 times as many buckets in the top 1% of mass as the normal does
(98 and 110 against 36). The ddsketch 3.0.1 oracle (`LogarithmicMapping(alpha)`, unbounded
`DenseStore`), fed the same streams (5 seeds x 4 distributions, N = 100,000), agrees with the
study sketch on all 15,330 buckets (key and count; none differ), and its quantile is within
`alpha` of the exact quantile (largest 7.80e-03).

### 4.5 t-digest rank error and agreement (20 seeds for every library)

Rank error `|F(q_hat) - p|`, `F` the empirical CDF of the stream, N = 100,000, 20 seeds, four
distributions, `p` in 0.01, 0.05, 0.5, 0.95, 0.99. The `tdigest` package, which part 2 ran on 3
seeds, now runs on all 20 (80 streams per compression, about 6 s a stream).

| compression | implementation | largest rank error | at p = | centroids (mean, max) | state bytes (max centroids x 16) | largest k1-span |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | study digest | 2.41e-03 | 0.5 | 57.3, 61 | 976 | 1.00 |
| 100 | pytdigest | 1.02e-02 | 0.5 (lognormal) | 61.5, 64 | 1,024 | 5.49 |
| 100 | crick | 1.44e-03 | 0.5 | 128.3, 134 | 2,144 | 0.50 |
| 100 | tdigest | 2.70e-04 | 0.5 | 834.5, 912 | 14,592 | 0.31 |
| 200 | study digest | 1.20e-03 | 0.5 | 113.9, 118 | 1,888 | 1.00 |
| 200 | pytdigest | 2.90e-03 | 0.5 | 102.6, 107 | 1,712 | 3.79 |
| 200 | crick | 6.80e-04 | 0.95 | 254.9, 261 | 4,176 | 0.50 |
| 200 | tdigest | 1.70e-04 | 0.5 | 1,546.6, 1,694 | 27,104 | 0.32 |

The k1-span is the paper's size rule (3.4) applied to each library's own centroids at the same
compression: the rule needs span <= 1. `tdigest` is called with `delta = 1/compression` and
`K = 25` (its default); it is an older tree-based algorithm.

Agreement. The largest rank difference between two libraries' answers for the same quantile of
the same stream (compression 100 / 200): pytdigest against crick 1.03e-02 / 2.89e-03; pytdigest
against tdigest 1.02e-02 / 2.85e-03; pytdigest against the study digest 9.9e-03 / 2.6e-03; crick
against tdigest 1.47e-03 / 6.7e-04; crick against the study digest 2.6e-03 / 9.7e-04; tdigest
against the study digest 2.3e-03 / 1.2e-03. So pytdigest and tdigest do not agree with each
other (1.0e-02 and 2.9e-03 in rank), nor with exact ranks to better than pytdigest's own error;
`tdigest` agrees with exact ranks to 2.7e-04 and with `crick` to 1.5e-03. The error order tracks
the centroid count: `tdigest` (835 centroids at compression 100) < `crick` (128) < the study
digest (57) < `pytdigest` (62, but with centroids up to 5.5 k-units wide, so up to 5.5 times the
mass the rule allows at the places it breaks the rule). Per `p`, the study digest's largest
errors at compression 100: 9.2e-04 at 0.01, 1.07e-03 at 0.05, 2.41e-03 at 0.5, 9.8e-04 at 0.95,
8.5e-04 at 0.99; most digests have their largest error at the median, where the scale function
allows the widest centroids.

Which library is closest to the paper definition. The definition is the k1 size rule of 3.4.
`crick` meets it (span 0.50 at both compressions: a finer digest of the same family, 2.2 times
as many centroids as the rule's minimum) and has a small rank error (1.44e-03 and 6.8e-04);
`tdigest` meets it only by being a different, much finer algorithm (span 0.31, 14.6 KB at
compression 100); `pytdigest` breaks it (span 5.49 and 3.79), and has the largest error (1.02e-02
and 2.9e-03). Among the installed libraries `crick` is the closest to the paper definition; the
study digest is the definition itself (span exactly 1.00).

## 5. Error bounds and their derivations

**DDSketch ES.** Sort the `m` mass units of the tail and the `m` units of the sketch's slots; the
`i`-th smallest stored value and the `i`-th slot lie in the same bucket, because bucket order is
value order and the masses match. By 3.2 each pair differs by at most `alpha |x_i|`, so

```text
|ES_hat - ES| = |(1/m) sum_i (r_(i) - x_(i))| <= (1/m) sum_i alpha |x_(i)| = alpha * mean|x| over the tail
```

which is `alpha |ES|` when the tail has one sign (true of every level here: each lower 0.01 and
0.05 and upper 0.95 and 0.99 tail of the four distributions is one-signed). The partial-mass rule
is what makes the slot count equal `m` exactly; the whole-bucket variant takes mass the tail
does not hold and has no such bound (it reached 3.8 `alpha`). Float rounding adds about 1e-15.
VaR: the representative is within `alpha` relative of the true VaR for the same reason. With
`alpha = 0.78%` the bound is a relative 7.8e-03 of the tail mean; the measured fraction of it is
at most 0.18 with no decay, 0.34 with decay and 0.33 in a window for the light tails, and 0.76
to 0.88 for Pareto's lower 0.01 tail.

**Decay and window.** The bound is per mass unit, so it does not depend on the weights: decay
changes which values hold the mass, not the pairing argument, and a window's counts are exactly
those of the last `W` rows (4.3). The gaps grow only because fewer buckets hold the tail.

**t-digest rank error.** A quantile read between two centroid centres can be wrong by at most
the mass between those centres, which is at most the larger centroid's mass fraction. With the
k1 rule that fraction is `dq` such that `dk = 1` at `q`, and `dk/dq = delta/(2 pi sqrt(q(1-q)))`,
so

```text
rank error <= 2 pi sqrt(q (1 - q)) / delta        (3.1e-02 at q = 0.5, delta = 100; 1.6e-02 at delta = 200;
                                                    6.3e-03 at q = 0.01, delta = 100)
```

This worst-case bound holds only for a digest that meets the rule. The measured errors are
7.7% (2.41e-03 of 3.1e-02) and 7.6% (1.20e-03 of 1.6e-02) of it at the median: the true CDF is
smooth over a centroid and linear interpolation takes care of most of the mass. `pytdigest`'s
1.02e-02 is 32% of the bound for a digest meeting the rule, and its centroids break the rule by
a factor of up to 5.5, so its own worst case would be 5.5 times larger, which is consistent with
its error. The measured values are for i.i.d. streams; the bound assumes the digest meets the rule
after every merge pass.

**P-squared.** No bound exists (section 2).

## 6. State per online form

| Form | Bytes | Derivation |
| --- | --- | --- |
| P-squared, one quantile | 176 (96 re-derived) | 22 f64 or i64 values: heights 5, positions 5, desired positions 5, increments 5, `p`, count. Heights, positions, `p` and the count (12 values) are 96 bytes; `n'` and `dn'` are re-derivable from `p` and the count |
| DDSketch histogram (one signed sketch, both tails) | 16 per live bucket + 40 header; 227 to 1,292 buckets at N = 100,000, at most 20,712 bytes (20.7 KB) | an i64 key and an f64 count per live bucket; header (alpha, zero count, total weight, two store lengths: 5 f64) is an assumption of this study. The bucket count is the data's log-range over `ln(gamma) = 0.0156` (mean 227 for Pareto, maximum 1,292 for Student-t(3)), at most 1,677 buckets (26.9 KB) at N = 10^6 |
| merging t-digest, compression 100 | 976 to 1,024 | 61 to 64 centroids x 16 bytes (mean, weight); add 24 bytes if the minimum, maximum and total are stored |
| merging t-digest, compression 200 | 1,712 to 1,888 | 107 to 118 centroids x 16 bytes |

All three are bounded by the algorithm or by the data's log-range, never by the stream's length.
A decayed sketch has the same layout (f64 counts). A window needs the bucket key of each of the
last `W` rows to subtract the row that leaves: the keys are below 2^17 in magnitude for finite
f64 at this `alpha`, so 4 bytes each fit, 20 KB at `W` = 5,000 (8 bytes: 40 KB), on top of the
live buckets (12.7 KB at most, 163 to 792 live). See the open items.

## 7. Proposed tolerances

| | Quantity | Tolerance | Unit | Measured | Tight or loose |
| --- | --- | --- | --- | --- | --- |
| (a) | P-squared port against river | 0 | ulps (bit equality) | 0 ulps over 12.6 million values | tight; exact |
| (b) | ES from the sketch against the exact tail mean, light tails (normal, lognormal, Student-t) | 0.5 x alpha x mean`|x|` = 3.9e-03 relative (for a one-signed tail) | fraction of the deterministic bound | at most 0.18 (no decay), 0.34 (decay), 0.33 (window) of the bound | loose against the measurement (about 1.5 to 3 times), tight against the guarantee |
| (b) | the same, Pareto and any tail pressed on a bucket edge | 1.0 x the bound = 7.8e-03 relative | fraction of the bound | 0.76 to 0.88 | loose: the full guarantee; a smaller fraction would fail a run that is correct |
| (c) | window counts against the bincount of the window | exact | counts | 6,400 of 6,400 equal | tight; exact |
| (d) | merging t-digest rank error against exact ranks, N = 100,000 | 3.8e-03 at compression 100 (2.5e-03 x 1.5); 1.8e-03 at compression 200 (1.2e-03 x 1.5) | absolute rank | 2.41e-03 and 1.20e-03 (the measured maxima, rounded up to two figures, with a margin of 1.5) | tight against the measurement, loose (about 12%) against the derived worst case of section 5 |

Why (b) has two tolerances. For light tails the measurement sits at a fifth to a third of the
bound, so half the bound passes them with a margin of 1.5 to 3 and still catches an error twice
as large as measured. Pareto's lower 0.01 tail is the exception: the support's edge at `x_m = 1`
is a bucket edge, every value reads `alpha` low and the errors do not cancel, so 0.76 to 0.88 of
the bound is correct behaviour; the guarantee itself is the only defensible tolerance there.
Neither is tighter than the proof, and neither is looser: a gap above the bound is a bug.

Why (d) uses the study digest and not a library. `pytdigest`'s error (1.02e-02 and 2.9e-03) is
4 times the study digest's and its centroids break the paper's size rule (5.5 k-units wide), so
it should not be the tolerance reference; `tdigest` (2.7e-04) is a different, much finer
algorithm (14.6 KB at compression 100), and `crick` (1.44e-03) a finer member of the same family.
Any of them would give a different tolerance, 4 to 40 times apart, because the libraries do not
agree with each other (differences in rank up to 1.0e-02). A digest must be checked against its
own size rule and against exact ranks, not against another library's quantiles. (d) applies to
N = 100,000: at N = 20,000 one stream at compression 200 reached 1.95e-03 > 1.8e-03, so a
smaller N needs its own measurement.

## 8. Open items

- The 40-byte header of the sketch is an assumption (five f64: alpha, zero count, total weight,
  two store lengths); the byte counts of section 6 are only as good as it. The study digest's
  buffer of `8 delta` points is also an assumption.
- river's Rust source was not readable offline (the wheel is compiled and GitHub access for the
  source repository was not enabled in this session). The P-squared port follows river's last
  pure-Python version for the order of operations; the bit-equality is empirical (12.6 million
  comparisons). If river's Rust code later diverges, the trajectory test will show it.
- `tdigest` now runs on 20 seeds (part 2 used 3); the results above are on 20. pytdigest's and
  crick's internal scale functions were not read: the k1-span is a measurement of their
  centroids, not a statement about their source.
- The streams are i.i.d. A sorted, trending or adversarially ordered stream was not tested for
  the t-digest or the P-squared markers.
- The window's ring of `W` bucket keys is not in the part 2 byte counts (4.3 reports the live
  buckets only); at 4 bytes a row that is 20 KB at `W` = 5,000, against at most 12.7 KB of live
  buckets. A window memory above the current caps needs the user's decision before it lands.
- The proposed tolerances of section 7 are proposals; none has been wired into a test of the
  Rust code. The comparison with rheon's own `EwQuantile` and tail outputs needs the compiled
  extension, which this Python-only study did not build.
- No R oracle (R absent). `datasketches` (KLL) and `scipy` are installed but were not used as
  oracles.
- `docs/README.md` needs a row for this record when the study is merged (it was not edited
  here).
