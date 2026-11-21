<!-- kind: record -->
# Study 508: ADWIN, Page-Hinkley, DDM, EDDM and KSWIN against river

A numpy prototype for PLAN task 508 ("CUSUM, Page-Hinkley, ADWIN and the other
change detectors ... Oracle: `river`'s drift detectors on the same stream"),
following task 494's scan (`vm-tools/research/494-MODELS-SCAN.md` §14 on
`vm-tools/oracle-gap-scan`: every detector becomes an option of `drift`, and
ADWIN's bucket list is the only new structure). Python only, no Rust: the
options in `rheon-core`, `docs/EXTENDING.md`'s places and the measured Rust
cost are the task proper. Nothing here is collected by the repo's pytest
(`testpaths = ["tests"]`) and nothing is added to `pyproject.toml`: river is
not in the dev group, so the study runs in a venv of its own.

## Files and how to run them

- `scripts/study_508.py`: runs every check and prints the tables below (about
  15 minutes at 200 seeds; `--seeds 20` for about 3, `--no-cost` skips the
  timings).
- `studies/study_508/detectors.py`: the five detectors (`Adwin`,
  `PageHinkley`, `DDM`, `EDDM`, `KSWIN`), one value a row, with `save`,
  `load` and `state_arrays`. The update equations are in its module docstring
  and below.
- `studies/study_508/data.py`: seeded synthetic streams (no data files).
- `studies/study_508/compare.py`: each check.

Environment: a standalone venv, not `uv sync` (which would build the Rust
extension): Python 3.13.18, numpy 2.5.5, scipy 1.18.1, **river 0.27.0**, on
Linux x86-64 (4 cores). River 0.27.0's ADWIN is compiled Rust
(`river._river_rust.drift.AdaptiveWindowing`; the scan read 0.26.1's
`adwin.py`); its source was read from the sdist (`rust_src/adwin.rs`). The
other four are Python (`river/drift/page_hinkley.py`, `kswin.py`,
`binary/ddm.py`, `binary/eddm.py`). River's argument names in 0.27.0:
`ADWIN(delta=0.002, clock=32, max_buckets=5, min_window_length=5,
grace_period=10)`, `PageHinkley(min_instances=30, delta=0.005,
threshold=50.0, alpha=1 - 0.0001, mode="both")`, `binary.DDM(warm_start=30,
warning_threshold=2.0, drift_threshold=3.0)`, `binary.EDDM(warm_start=30,
alpha=0.95, beta=0.9)`, `KSWIN(alpha=0.005, window_size=100, stat_size=30,
seed=None, window=None)`.

## The detectors, in short

Each reads one value a row and answers, on that row, whether it flags a
drift (DDM and EDDM also a warning). The flag for row `t` uses row `t` itself:
it says the stream has changed by now, not what row `t` will be, so rule 2
has nothing to hold here (a detector fed a model's out-of-sample residual
inherits that residual's rule 2). Every detector in river, and here, **starts
from an empty state on the row after a flag** (river's `update` opens with
`if self.drift_detected: self._reset()`).

**ADWIN** (Bifet and Gavalda 2007, ADWIN2). The window is an exponential
histogram: level `i` holds at most `M` (`max_buckets`) buckets of `2^i` rows,
each the pair (sum of its values, sum of squared deviations about its own
mean). A row enters level 0 as `(x, 0)`; when a level holds `M + 1` buckets its
two oldest merge into one bucket of the next level,
`(t0 + t1, v0 + v1 + n (t0/n - t1/n)^2 / 2)`, `n = 2^i`, and the merge
cascades upward. The window's totals are kept as the rows come:

```text
width += 1;   V += (width - 1) (x - S/(width - 1))^2 / width;   S += x
```

Every `clock` rows (counted since the last reset), once `width >
grace_period`, each split of the window at a bucket boundary into an older
`W0` (`n0` rows) and a newer `W1` (`n1`) is tested, oldest boundary first:

```text
cut when  |S0/n0 - S1/n1| > eps_cut,  both n0, n1 >= L  (L = min_window_length)
eps_cut = sqrt(2 (1/m) (V/width) d') + (2/3) d' (1/m)
d'      = ln(2 ln(width) / delta)
1/m     = 1/(n0 - L + 1) + 1/(n1 - L + 1)
```

A cut drops the oldest bucket (`width -= n; S -= u; V -= v + n width (u/n -
S/width)^2 / (n + width)`) and restarts the scan on the shrunken window; the
row is flagged if any cut was made.

**Page-Hinkley** (Page 1954; Hinkley 1971; two-sided as Mouss et al. 2004).
With `xbar_t` the running mean of the rows since the reset, this one included:

```text
m_t = alpha m_{t-1} + (x_t - xbar_t - delta),   M_t = min_{s <= t} m_s,   up:   m_t - M_t > lambda
u_t = alpha u_{t-1} + (x_t - xbar_t + delta),   U_t = max_{s <= t} u_s,   down: U_t - u_t > lambda
```

tested once `t >= min_instances`; `mode` picks up, down or either.

**DDM** (Gama et al. 2004). `x_t = 1` for an error; `p_t` the error rate of
the rows since the reset, `s_t = sqrt(p_t (1 - p_t) / t)`. Once `t >
warm_start`: if `p_t + s_t <= p_min + s_min` the minimum moves to `(p_t,
s_t)`; warning when `p_t + s_t > p_min + 2 s_min`, drift when `p_t + s_t >
p_min + 3 s_min`.

**EDDM** (Baena-Garcia et al. 2006). On each error, the distance in rows
since the previous error (the first from the reset) enters a running mean
`p'` and sample standard deviation `s'` (Welford, `ddof = 1`). Once `t >
warm_start`: a new maximum of `p' + 2 s'` is recorded; otherwise, once more
than `warm_start` errors have been seen, `level = (p' + 2 s') / max`, drift
when `level < beta`, warning when `level < alpha`. The warning is refreshed
only on error rows.

**KSWIN** (Raab et al. 2020). A ring of the last `window_size` values. Once
full, each row draws `r = stat_size` values without replacement from the
older `window_size - r` and compares them with the newest `r` by the
two-sample Kolmogorov-Smirnov statistic `D = max_x |F_old(x) - F_new(x)|`.
With equal sizes `D = h / r` for an integer `h`, and its exact null tail is
the Gnedenko-Korolyuk sum

```text
P(D_{r,r} >= h/r) = 2 sum_{j >= 1} (-1)^(j+1) C(2r, r - j h) / C(2r, r)
```

which the prototype computes in integers and rounds once (`ks_pvalue_equal`).

### Where river departs from the papers

The prototype's defaults are river's, so the comparison can be exact; each
departure is an option whose paper form is measured in §2.

| detector | river 0.27.0 | the paper |
|---|---|---|
| ADWIN | Bernstein bound on the window's variance, `d' = ln(2 ln(width)/delta)` | Theorem 3.1 is proved for the Hoeffding form, values in [0, 1]: `eps = sqrt((1/(2m)) ln(4/delta'))`, `delta' = delta/n`, `1/m = 1/n0 + 1/n1`; the Bernstein form is the paper's "in practice" variant (§4), with no theorem |
| ADWIN | `1/m` shifted by `min_window_length` (MOA's form) | `1/m = 1/n0 + 1/n1` |
| ADWIN | tests every `clock = 32` rows, none until `width > grace_period = 10` | every row |
| ADWIN | after a flag, the whole window is dropped on the next row | the window keeps `W1` |
| ADWIN | no cap on the width | none either; the library's state rule needs one (§4) |
| Page-Hinkley | fading `alpha = 1 - 1e-4` on both sums | `alpha = 1` |
| Page-Hinkley | mean kept as `m += (x - m)/n` | `sum / n` (same flags, §2) |
| DDM | `p += (x - p)/n` | `errors / n`, one rounding (same flags, §2; up to 265 ulps apart, §8) |
| KSWIN | flag when scipy's exact `p <= alpha` and `D > 0.1` | flag when `D > sqrt(-ln(alpha) / r)`, whose size is about `2 alpha` (§1b) |
| KSWIN | the draw is CPython's `random.Random(seed).sample`, **re-seeded at every reset**, so the same index sequence repeats after each flag | a uniform draw |
| KSWIN | on a flag the window is cut to its newest `r` values, but the next row's reset empties it: the cut is dead code | the window keeps the newest `r` |

The `D > 0.1` clause never binds at river's defaults: `p <= 0.005` at `r =
30` needs `h >= 14`, `D >= 0.467`.

## Results

RESULTS

## Proposed tolerances for the Rust options' tests

| check | tolerance | reason |
|---|---|---|
| ADWIN against river: flag rows, `width`, `total`, the sum of squares, `estimation`, on every row | flags exact; fields bit-equal | measured 0 rows and 0 ulps (§1). The state takes only `+ - * /`, so it is bit-equal on every OS; the one libm call (`ln` in `eps_cut`) decides a comparison and enters no state, and the closest of 637,520 tests was 6.6e-5 relative from a tie (§8), eleven orders above an ulp |
| Page-Hinkley, DDM, EDDM against river: flags, warnings, every internal field | flags exact; fields bit-equal | measured 0 (§1); no libm call (`sqrt` is correctly rounded by IEEE 754) |
| KSWIN flags against river, same seed | exact, **only if** the Rust model reproduces CPython's draw (MT19937 and `random.sample`'s selection, re-seeded on reset) | the flags match with the draw reproduced (§1); without it, the deterministic form (`draw="all"`) is checked instead: `D` against `scipy.stats.ks_2samp` on the two parts, bit-equal (`D = h/r` is one division in both) |
| KSWIN p-value | 16 ulps against scipy's `ks_2samp(method="exact")`; 0 against the integer sum rounded once | scipy's Horner evaluation was 2 to 11 ulps from the correctly rounded value for `r` from 10 to 300 (§1b); the decision `p <= alpha` never sat that close: the p-values are a discrete set and the nearest to 0.005 at `r = 30` is 2.53e-3 |
| ADWIN false cuts on the null, per check | at most `delta`; the ledger's kind (statistical, tightening only) | measured 0 in 62,400 checks at `delta = 0.002` (95% upper limit 4.8e-5, 2.4% of `delta`) and 94 at `delta = 0.1` (1.5e-3, 1.5% of `delta`) (§3a) |
| ADWIN step delay | every seed within Theorem 3.1's `n1` (factor 2) plus `clock - 1` | measured 200 of 200 seeds in every case (§3a) |
| KSWIN test size, no reset | within 4 standard errors of the exact size, the standard error inflated by the tests' overlap (each window shares `r - 1` new rows with the next) | measured 2.5e-3 against the exact 2.53e-3 (§3b) |
| a zero-weight row (the first included): flags against river on the stream without it, the state before and after it | exact; bit-equal | measured 0 (§6) |
| chunk invariance, save and load mid-stream | bit-equal flags and final state | rule 3; measured 0 over 201 chunks (§7) |

## Caveats and what was not done

- **The weighted form has no oracle.** River's five detectors take a value,
  not a weight (`update(x)`); `stats.Mean` underneath takes a weight, but no
  detector passes one. The checkable form is the scan's: a zero weight is a
  row that never happened (it advances no count, no `clock` tick and no draw)
  and any positive weight is one row, checked against river on the stream
  with the zero-weight rows removed. A fractional weight as a fraction of a
  row (an ADWIN bucket of weight `w`, a weighted error rate in DDM) would be
  this library's own definition with nothing to check it; integer frequency
  weights could be checked against river on repeated rows, which only
  restates the definition. Not built.
- **"Advance nothing" differs from rule 9's "advance the clock, learn
  nothing".** These detectors have no clock but their row count, so a
  zero-weight row cannot advance one without learning. `emit_drift`'s
  clock-integrated Page-Hinkley (the existing option) is the place a clock
  belongs.
- **ADWIN needs a cap for the state rule.** Uncapped, its levels grow with the
  logarithm of the rows since the last flag (§4): bounded by
  `1 + floor(log2((W + M - 1)/M))` levels at width `W`, so 18 levels and
  1,920 bytes at a million rows and 38 levels, 4,000 bytes, at `2^40`.
  `max_width` drops the oldest bucket while the width exceeds the cap; river
  has no cap, so the comparison with river holds only while the width stays
  under it.
- **KSWIN's random draw** is the one piece whose exact match needs a foreign
  algorithm in the state: CPython's Mersenne Twister (2,500 bytes) and
  `random.sample`'s choice between its two selection methods. Either the
  Rust option reproduces both (then river is an exact oracle) or it takes the
  deterministic form (newest `r` against all older values; scipy is then the
  oracle, river only a statistical one). The study does not choose.
- **No CUSUM.** River has no CUSUM class (the scan's point 3); the tabular
  CUSUM's oracle is its published average run lengths, not river, and is not
  in this study. Neither are `FHDDM`, `HDDM_A`, `HDDM_W`, which river has.
- **Delays here are on one stream design** (a +1 sd step or ramp at row
  5,000 of 10,000; an error rate 0.10 to 0.25). They rank the detectors on
  that design; they are not general claims.
- The cost is pure Python (ours) or river's (Rust for ADWIN, Python with
  scipy for KSWIN) on this machine; the Rust option's cost is the task's to
  measure.
- The paper's Theorem 3.1 bounds the Hoeffding form; river's Bernstein form,
  which uses the window's estimated variance, has no theorem. Both are
  measured against `delta` (§3a); both are far inside it.
