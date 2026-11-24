<!-- kind: record -->
# `ddsketch`: the design note

The bank's `ddsketch` kind (PLAN 544, part 7; the study on `vm/study-544`,
`docs/records/STUDY-544-TAILS.md` there, sections 3.1 to 3.3 and 4.2 to
4.4). The core sketch is `crates/rheon-core/src/ddsketch.rs`; the model over
it, one sketch a target, is `ddsketch_model.rs`.

## The update

For target `t`, level `p`, a row of weight `w` and value `y`:

```text
q_p,t  = the representative of the first bucket whose cumulative weight reaches p W_t
es_p,t = (sum_{k<j} c_k r_k + (m - C_{j-1}) r_j) / m     m = p W_t (p < 0.5) or (1 - p) W_t (p > 0.5)
D_t   <- D_t + (y, w)          w > 0 and y usable
W_t   <- W_t + w,   S_t <- S_t + w^2
```

The mapping is the `ddsketch` package's `LogarithmicMapping` (no offset):
`g = 2 alpha / (1 - alpha)`, `gamma = 1 + g`, `mu = ln 2 / ln(1 + g)`,
`key(x) = ceil(log2(x) mu)`, `value(k) = 2^(k / mu) 2 / (1 + gamma)`, a
negative value by its magnitude in a second store, a magnitude at or under
`f64::MIN_POSITIVE gamma` in the zero count. `log2`, `ln(1 + g)` and `2^y`
are summed from series in plain arithmetic (within 4 ulps of the math
library's), so a key and a representative are the same bits on every OS and
no libm result is in the state (only the keys and weights are).

## Decisions (the coordinator, 2026-11-23)

- Kind `ddsketch`, the output shape of `tdigest`: `alpha` (default
  `tanh(1/128)`), `targets=[...]`, `quantiles=[...]`; per target the levels'
  `q<level>_<target>` then their `es<level>_<target>`, beside `weight_sum`,
  `settled_frac` and `withheld_reason`; no `pred` or `coef`; the builder in
  `python/rheon/_spec_ddsketch.py`.
- The plain sketch takes no span. The windowed sketch is a ring of bucket
  keys in the state: `window_rows` (the last rows of positive weight) or
  `window_size` (the rows of the last span of the clock, under `closed`),
  the library's two spellings of a window (as `expectile`'s).
- A decay: refused in the first build, naming the windowed form (a decayed
  sketch scales every bucket at every row, so its weights are rounded and no
  row can be taken out exactly). The coordinator's answer to parts 6-7:
  build the decayed form as a rounded form, the rounding stated and checked
  against a reference; the windowed ring stays exact. A decay beside a
  window is refused. Each row, after its read (`n_eff` before its own decay,
  hard rule 8), multiplies every bucket, the zero count, `n_eff` and `W_t`
  by the factor (`S_t` by its square), then adds the row. A bucket formed
  `k` rows ago holds its exact decayed sum within a relative
  `2 k u / (1 - 2 k u)`, `u = 2^-53` (one rounding a multiply, one an add);
  `tests/test_ddsketch.py` holds every bucket of 2,000 rows to the exact sum
  in `fractions` from the same factor within it, and the Rust test to a
  double-double sum. A bucket that underflows to 0 is removed. The factor is
  the platform's `exp2`/`powf`, so the decayed fixture is held to the
  golden tolerance across platforms, not to the bit.
- The Kish withholding rule of `tdigest` (`level_identified`), with its
  reason `level_beyond_sample`.
- State bounded and checked against `max_state_bytes`: the fixed part
  (header, levels, a row window's ring at its fullest) at the call; the
  buckets, which the data's log-range decides, and a clock window's ring as
  the rows go in, by `clock_span::HeldCap` (said past half of the cap, the
  run refused at it).

## Choices this part made

- **The quantile's rank rule** is the study's and `ewm_quantile`'s: the
  first bucket whose cumulative weight reaches `p W` (numpy's
  `inverted_cdf`). The package reads rank `p (n - 1)`, which is not
  invariant to a weight's scale (the weight-scale rule); the two pick the
  same order statistic whenever `p n` is not whole and `frac(p n) >= p`, so
  the agreement test reads both at `n = 100,001`.
- **The tail** is the lower one below 0.5 and the upper above it, 0.5
  refused by name, as `tdigest`'s (the coordinator's part 5 exchange B); the partial-mass rule (the study's 3.3) makes the tail's
  mean within `alpha` times the tail's mean magnitude of the exact one.
- **`n_eff` under a window** is the weight of the rows the row reads: a
  clock window's rows that have left by the row leave before the read
  (hard rule 8: the weight the fit is read from).
- **`alpha`'s floor** is `1e-6`: a key is kept in 32 bits in the ring (the
  key times 4 and a tag), and at `alpha >= 1e-6` every finite double's key
  is under `2^29` in magnitude. No ceiling but 1.

- **Sparse stores.** A dense store (one weight a key from the smallest to
  the largest held) took 345 KB for a stream holding 1.0 and 5.5e-80, past
  the 256 KB default cap, in the property tests' first draw; the stores keep
  their live buckets alone, sorted by key (a binary search a row), as the
  study's byte count (16 a live bucket) assumed.

## State

Per target: `12` bytes a live bucket (a 32-bit key and an `f64` weight),
`16` under a window (a `u32` row count beside each), and `32` bytes of
sums; a header of the model and the levels. The live buckets are the data's
distinct ones, at most its log-range over `ln gamma` (at the default
`alpha`, 64 a factor of `e`, about 1,470 a store for values spanning `1e-5`
to `1e5`); the study counted 167 to 1,292 live buckets (both stores) at
100,000 rows of its four streams, so 2.0 to 15.5 KB a target. A row window
adds `window_rows (8 + 4 m)` bytes for `m` targets at the call; a clock
window `8 + 4 m + size_of(Stamp)` bytes a row it holds, as the rows go in.
At `alpha` halved the buckets double.

## Measured

`tests/test_ddsketch.py`, against the `ddsketch` package 3.0.1 and numpy:
every bucket key and count equal to the package's at `alpha` = tanh(1/128),
0.01 and 0.05 on the four streams (zeros included); each representative and
each quantile at N = 100,001 within 2 ulps of the package's; each quantile
within `alpha` of numpy's `inverted_cdf` and each tail mean within
`alpha` times the tail's mean magnitude of the exact partial-mass tail mean,
20 seeds, with weights of 1 and drawn from [0.1, 3]; `crick` 0.0.8's
t-digest at compression 1,000 within `2 alpha` plus its own rank error at
the quartiles and median.
