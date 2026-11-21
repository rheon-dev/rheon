# Study 544: do the streaming tail measures agree with independent oracles?

Status: part 1 of 3 complete (oracle search; P-squared port against river).
Missing, for parts 2 and 3: t-digest and DDSketch quantiles against exact
quantiles, VaR and expected shortfall (the tail mean) against numpy, and
the comparison with rheon's own `EwQuantile` and tail outputs.

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

## Open questions

- river's Rust source was not readable offline (the wheel is compiled and
  GitHub access for the source repository was not enabled in this session);
  the bit-equality above is empirical, and the order of operations came from
  the last pure-Python version. If river's Rust code later diverges, the
  trajectory test will show it.
- No R oracle (R absent).
