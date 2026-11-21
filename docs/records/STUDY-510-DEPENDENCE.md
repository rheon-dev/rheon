<!-- kind: record -->
# Study 510: online mutual information and windowed rank correlations

A numpy prototype of PLAN task 510 ("online mutual information from binned
counts and rank correlations (Spearman, Kendall) over windows, beside `ew_cov`
and `marginal` ... Oracle: `scipy.stats` (`spearmanr`, `kendalltau`) and
`scikit-learn` `mutual_info_score` on the same window"), following task 494's
scan (`vm-tools/research/494-MODELS-SCAN.md` §16 on `vm-tools/oracle-gap-scan`).
Python only, no Rust: the model in `rheon-core` behind `OnlineModel`,
`docs/EXTENDING.md`'s places and the measured Rust cost are the task proper.
Nothing here is collected by the repo's pytest (`testpaths = ["tests"]`), and
nothing is added to `pyproject.toml`: scipy and scikit-learn are already in the
dev group.

## Files and how to run them

- `scripts/study_510.py`: runs every check and prints the tables below (about
  75 s; `--no-cost` skips the timings).
- `studies/study_510/dependence.py`: the prototypes. `BinnedMI` (joint bin
  tables, decay or a window), `RankWindow` (a ring of rows; Kendall's tau-b kept
  incrementally, Spearman and Chatterjee's xi read at a cadence), `edges_from`
  (`margbins::edges_from` line for line). The update equations are in its module
  docstring.
- `studies/study_510/data.py`: seeded synthetic streams (no data files).
- `studies/study_510/compare.py`: each check, returning its worst gap.
- `studies/study_510/test_checks.py`: the same checks as asserts (26 tests, about
  18 s): `python -m pytest studies/study_510` in any env with numpy, scipy,
  scikit-learn and pytest.

Environment: a standalone venv, not `uv sync` (which would build the Rust
extension): Python 3.13.18, numpy 2.5.5, **scipy 1.18.1**, scikit-learn 1.9.1,
on Linux x86-64 (4 cores; numpy's BLAS held to one thread).

## The measures, in short

Every pair is a feature `j` against a target `t`, as `marginal` pairs them.

**Bins** are `marginal`'s: a value goes to `edges.partition_point(e <= v)`
(numpy's `searchsorted(edges, v, side="right")`; a value on an edge goes to the
bin above), a non-finite value to no bin; edges `"fixed"` or `"quantile"` from a
warm-up sample, as `margbins::edges_from`. The target is binned the same way
(the scan's `target_bins=`).

**Mutual information.** Per pair a joint table of weights `n_ab`; a row of
weight `w > 0` whose two values both fall in a bin adds `w` to one cell. Read,
with `ln` only at read time:

```text
MI    = sum_{n_ab > 0} (n_ab / N) ln((n_ab / n_a) (N / n_b))          (nats)
MI_MM = MI - (B_xy - B_x - B_y + 1) / (2N)
```

`B_xy`, `B_x`, `B_y` are the nonzero joint cells and margins. Miller-Madow is
`H_MM = H + (m - 1) / (2N)` on each of `H(X)`, `H(Y)`, `H(X, Y)`, and `MI = H(X)
+ H(Y) - H(X, Y)` gives the line above: **the correction is subtracted**. The
request's form, `MI + (B_xy - B_x - B_y + 1) / (2N)`, has the sign reversed (it
adds the bias instead of removing it); the table below measures it against the
definition, 0.07 to 0.72 nats off. The scan's "`(cells - 1)/(2n)` on
scikit-learn's value" is the one-entropy form; for MI it is the line above.
Miller-Madow is given for unit weights only (a window, or no decay): under decay
the `N` it needs is not the table's weight (Kish's `W^2/V` is one candidate) and
no library settles it.

Decay (`half_life` `h` rows, `lam = 2^(-1/h)`) keeps the table undecayed against
one scale `s_t`, as `margbins.rs` does: a row adds `w / s_t`, the scale folds into
the table when it falls below `1e-50`. A window of `W` rows keeps a ring of each
row's bin indices (`uint16`) and weight; the row leaving subtracts what it added.

**Ranks.** A ring of the last `W` rows (values and weights). Pair `(j, t)`'s
sample is the ring's rows with `w > 0` and both values finite; a weight only
gates, each row counts once. Kendall's tau-b is kept incrementally in four
integer counters per pair: `S = n_c - n_d`, the pair count `n_0`, the pairs tied
in `x` (`n_1`) and in `y` (`n_2`). A row entering compares itself with the ring
(`O(W)` per pair), the row leaving subtracts its pairs with the rows that stay,
and `tau_b = S / sqrt((n_0 - n_1)(n_0 - n_2))` is read every row in `O(1)`.
Spearman (Pearson of average ranks) and Chatterjee's xi (2021, eq. 1.2 with ties
in `y`, 1.1 without) are read every `rank_every` rows (`O(W log W)` per pair) and
held between reads. Chatterjee breaks ties in `x` at random; this prototype
breaks them by arrival (a stable sort of the ring, oldest row first), which is
deterministic and needs no random stream in the state.

**Rules held.** A row's emitted numbers are read before the row is learned (2).
One row at a time, so any chunking is bit-identical (3; checked, 1 chunk against
13). The emitted weight is the table's `N` before the row (8): under a window the
weight inside it. A zero-weight row, the first included, takes a ring slot and
adds to no cell; a read of an empty table or ring is NaN, never a 0/0 in the state (9).
A window is the last `W` rows of the stream: zero-weight rows and rows with a
NaN or an inf take a slot (they advance the clock) and give no sample.

## Results

Gaps are absolute; every compared quantity is of order one or smaller.

### 1, 2. Plug-in MI against `sklearn.metrics.mutual_info_score`; Miller-Madow against its definition

A 4,000-row stream, 4 features by 2 targets, with ties, zero weights (the first
row too), NaN and inf cells and whole NaN rows; edges from the first 500 rows;
read every 97 rows. The oracle's labels come from `np.digitize` on the window's
rows (`pair_sample`), independent of the model's ring. The definition is plain
Python (`collections.Counter`, `math.fsum`).

| edges | bx | by | W | reads | `mutual_info_score` | rel (MI > 1e-3) | MM vs definition | the `+` sign vs definition |
|---|---|---|---|---|---|---|---|---|
| fixed | 4 | 4 | none | 328 | 1.3e-15 | 6.5e-13 | 8.7e-16 | 0.074 |
| quantile | 16 | 8 | none | 328 | 9.3e-16 | 6.6e-14 | 1.6e-15 | 0.500 |
| quantile | 16 | 16 | 400 | 328 | 8.9e-16 | 2.3e-15 | 1.8e-15 | 0.586 |
| fixed | 32 | 8 | 400 | 328 | 7.8e-16 | 3.8e-15 | 1.3e-15 | 0.359 |
| quantile | 32 | 32 | 2000 | 328 | 1.1e-15 | 2.7e-15 | 1.8e-15 | 0.719 |

The relative 6.5e-13 is an absolute 1.3e-15 on an MI of 2e-3. Against an
`np.longdouble` read of the same table (below) the model's read is the more
accurate of the two: the gap here is scikit-learn's rounding.

### 3. The Gaussian band

A standard bivariate normal, `I = -1/2 ln(1 - rho^2)`; `b x b` quantile bins
learned from 2,000 warm rows; 40 seeds of `N = 20,000` rows each. `I_bin` is the
MI of the binned *population*: the cell probabilities by Gauss-Legendre
quadrature (96 nodes a bin) of `phi(x) [Phi((y_hi - rho x)/s) - Phi((y_lo - rho x)/s)]`,
`s = sqrt(1 - rho^2)`, good to about 1e-14 (its value at `rho = 0`).

| rho | bins | I | I_bin | I - I_bin | plug-in mean | (b-1)^2/2N | MM mean | MM sd | MM z vs I_bin | MM worst \|. - I_bin\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | 0 | 0 | 0 | 0.0012 | 0.0012 | -5.6e-05 | 2.0e-04 | -1.7 | 5.3e-04 |
| 0 | 16 | 0 | 0 | 0 | 0.0056 | 0.0056 | -6.8e-05 | 4.1e-04 | -1.0 | 0.0010 |
| 0 | 32 | 0 | 0 | 0 | 0.0244 | 0.0240 | 3.4e-04 | 0.0011 | 2.0 | 0.0027 |
| 0.3 | 8 | 0.0472 | 0.0419 | 0.0053 | 0.0429 | 0.0012 | 0.0417 | 0.0020 | -0.5 | 0.0042 |
| 0.3 | 16 | 0.0472 | 0.0450 | 0.0022 | 0.0504 | 0.0056 | 0.0448 | 0.0023 | -0.6 | 0.0045 |
| 0.3 | 32 | 0.0472 | 0.0463 | 8.6e-04 | 0.0705 | 0.0240 | 0.0464 | 0.0025 | 0.4 | 0.0052 |
| 0.6 | 8 | 0.2231 | 0.1961 | 0.0270 | 0.1971 | 0.0012 | 0.1959 | 0.0036 | -0.4 | 0.0078 |
| 0.6 | 16 | 0.2231 | 0.2120 | 0.0111 | 0.2175 | 0.0056 | 0.2119 | 0.0041 | -0.1 | 0.0072 |
| 0.6 | 32 | 0.2231 | 0.2187 | 0.0045 | 0.2426 | 0.0240 | 0.2194 | 0.0043 | 1.1 | 0.0091 |
| 0.9 | 8 | 0.8304 | 0.6953 | 0.1350 | 0.6963 | 0.0012 | 0.6953 | 0.0042 | 0.0 | 0.0085 |
| 0.9 | 16 | 0.8304 | 0.7713 | 0.0590 | 0.7759 | 0.0056 | 0.7718 | 0.0048 | 0.6 | 0.0116 |
| 0.9 | 32 | 0.8304 | 0.8054 | 0.0249 | 0.8244 | 0.0240 | 0.8084 | 0.0057 | 3.3 | 0.0140 |

Two biases with opposite signs, and the band is stated from both:

- **Discretization** pulls every binned value *below* `I` (a function of the
  data carries no more information than the data): `I - I_bin` is 0.005, 0.027
  and 0.135 nats at `rho` = 0.3, 0.6, 0.9 with 8 bins, and falls about 2.3-fold
  for each doubling of the bins (0.135, 0.059, 0.025 at `rho = 0.9`). It is
  deterministic given the edges and the distribution, so the oracle is `I_bin`,
  not `I`; `I` bounds it from above.
- **Sampling** pushes the plug-in *above* `I_bin` by about `(b-1)^2 / (2N)`
  (measured 0.0012, 0.0056, 0.0244 at `rho = 0` against 0.0012, 0.0056, 0.0240).
  Miller-Madow removes it to within two standard errors in eleven of twelve
  rows. At `rho = 0.9`, 32 bins it leaves 0.003 (z = 3.3), an eighth of the
  correction: 1,024 cells at about 20 rows a cell, many of them nearly empty,
  where the `O(1/N^2)` terms the correction ignores are no longer small.

So the band, for one stream: `I_bin - 4 sd <= MI_MM <= I_bin + 4 sd + (b-1)^2/(10N)`,
and so `MI_MM <= I + 4 sd + (b-1)^2/(10N)`; `sd` is about `sqrt(Var[ln p(x,y)/(p(x)p(y))]/N)`,
measured above (2e-4 to 6e-3 at `N = 20,000`). The asserts hold every one of the
seeds to `|MI_MM - I_bin| < 4 sd + 0.2 (b-1)^2/(2N)`.

### 4. Decay: the table and MI against a plain-Python longhand

A 6,000-row stream (3 by 2 pairs, 8 by 6 bins) with fractional weights on [0.2, 2],
zero weights and gaps. The longhand is plain Python lists: every cell multiplied
by `lam` every row, then `w` added. MI read from the longhand table in plain
Python (`math.log`, `math.fsum`), and by scipy's `entropy` on the weighted
margins and joint, `H(X) + H(Y) - H(X, Y)` (the scan's decayed oracle;
`mutual_info_score` casts a non-integer table to integers and cannot serve).
Read every 250 rows.

| half-life | rows | scale folds | table rel | MI vs longhand | MI vs scipy `entropy` |
|---|---|---|---|---|---|
| 5 | 6000 | 7 | 3.5e-16 | 4.4e-16 | 1.1e-15 |
| 50 | 6000 | 0 | 9.1e-16 | 4.2e-16 | 1.3e-15 |
| 1000 | 6000 | 0 | 3.5e-15 | 3.3e-16 | 1.1e-15 |

At `h = 5` the scale folds seven times (below `1e-50` every 166 half-lives). The
"table rel" growth with `h` is the longhand's: it rounds each cell once a row,
the model once a fold.

### 5. The rings

A 2,000-row stream, `W = 60`, the first row of zero weight, a run of 120
zero-weight rows (twice the window, so the window empties), a 30-row gap of NaN
features, scattered NaN and inf cells and whole NaN rows. Before every row: each
pair's rank sample (oldest first) against the stream's last `W` rows filtered by
hand, element for element; each Kendall pair count against `n(n-1)/2` of that
sample; the MI table against one rebuilt from those rows.

| rows | rank samples differing | Kendall pair counts differing | MI tables differing | emptied window reads NaN, counters 0 | first row zero-weight |
|---|---|---|---|---|---|
| 2000 | 0 | 0 | 0 | yes | yes |

All exact (bit-equal). With fractional weights the MI window's subtraction is
not exact: after 100,000 rows at `W = 500` (16 x 16 bins), the worst cell is
7.1e-14 from a rebuilt table and **52 cells that should be empty are not**
(residue of either sign). MI moves by 2.9e-15 (the cells' `c ln c` is tiny), but
`B_xy` counts them: one more reason Miller-Madow is withheld for fractional
weights. Unit weights keep every cell an integer, so a window there is exact
forever.

### 6. Ranks against scipy

A stream of 4 features by 2 targets with ties in a feature and a target, zero
weights and gaps; every pair at every read against scipy on the window's rows.
Kendall is checked on every row (it is emitted every row); `held` checks that a
row between reads emits the last read's Spearman and xi; `chunks` that 13 chunks
give the same bits as one.

| W | rank_every | reads | `spearmanr` | `kendalltau` (b) | `chatterjeexi`, distinct x | `chatterjeexi`, tied x | xi longhand | `chatterjeexi(y_continuous=True)` | held | chunks |
|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 1 | 6368 | 3.3e-16 | 2.2e-16 | 0 | 0 | 0 | 0 | yes | yes |
| 200 | 7 | 1711 | 2.2e-16 | 2.2e-16 | 0 | 0 | 0 | 0 | yes | yes |
| 500 | 25 | 472 | 3.3e-16 | 2.2e-16 | 0 | 0 | 0 | 0 | yes | yes |

`scipy.stats.chatterjeexi` exists in scipy 1.18.1 and is the oracle. Its xi is
exact here (0, not merely close): the numerator and denominator are integers.
With ties in `x` the statistic depends on the order the tied rows are visited:
scipy sorts with `argsort`'s default kind, which numpy does not promise is
stable. It agreed with this prototype's arrival order on every read here (0),
and an earlier draft that passed the ring in slot order instead of arrival order
differed from scipy by up to 0.13. So for tied `x` the oracle is the longhand
(plain Python, eq. 1.2, Python's stable `sorted`), checked on every seventh read.

### 7. Cost

`k = 10` features, `T = 1` target (10 pairs), a full ring, numpy on one thread;
the means of 200 rows and 5 reads (MI: 3,000 rows, 200 reads). Prototype
numbers, 10 to 20% apart from run to run: they rank the work, they do not
predict the Rust model's.

| W | ranks: per row | Spearman + xi read, all pairs | Kendall read | fill the ring |
|---|---|---|---|---|
| 1,000 | 0.20 ms | 6.3 ms | 7.4 us | 0.09 s |
| 10,000 | 4.4 ms | 67 ms | 7.8 us | 19 s |

| W | bins | MI: per row | MI read, all pairs |
|---|---|---|---|
| none | 16 | 45 us | 56 us |
| 1,000 | 16 | 63 us | 74 us |
| 10,000 | 16 | 61 us | 47 us |

The ranks' row cost is Kendall's incremental update: `O(W k T)` (two comparisons
of the row against the ring and four `(k x W)(W x T)` products of `0/±1`
matrices). It grew 22-fold for a 10-fold `W` here (the ring falls out of cache),
which a Rust loop over the ring would not; at about 2 ns a compared pair,
`W = 10,000` by 10 pairs is some 0.2 ms a row (an estimate, not measured). A
read is `O(k T W log W)`: 67 ms at `W = 10,000`, so `rank_every` should be about
`W / 10` or more there. Kendall's read is `O(k T)`. MI's row cost does not depend
on `W` (one cell per pair in, one out); its read is `O(k T bx by)`.

## State size

Counted from the arrays (`state_arrays`, summed `nbytes`) and equal to the
formula at `k = 10`, `T = 1`, 16 x 16 bins, `W = 1,000`:

| form | formula (bytes) | k = 10, T = 1, 16 x 16 bins, W = 1,000 |
|---|---|---|
| MI, decay or none | `8 k T bx by + 8 (k (bx-1) + T (by-1)) + 24` | 21,824 (counted 21,824) |
| MI, window | the above `+ 2 W (k + T) + 8 W + 8` | 51,832 (counted 51,832) |
| ranks, window | `8 W (k + T) + 8 W + 48 k T + 16` | 96,496 (counted 96,496) |

The table is `8 bx by` per pair (2 kB at 16 x 16). The MI ring is `uint16` bin
indices; the rank ring is the values in doubles; `48 k T` is Kendall's four
`int64` counters and the two held reads per pair. At `W = 10,000` the rank ring
is 960,496 bytes, linear in `W`. Both are bounded by the shape and `W`, never the
stream. Two savings for the Rust model, not taken here: the weights need only a
bit each in the rank ring (a weight only gates); and when both measures run on
one pair set, the MI window needs no ring of its own, since its edges are fixed
and the leaving row's bins are `bin_of(edges, value)` from the rank ring's values,
saving `2 W (k + T) + 8 W` bytes. The rank ring is the row ring tasks 495 and 532
also need.

## Numerical risk

**The MI read: one log of a ratio, not a sum of logs.** The first draft read
`ln n_ab + ln N - ln n_a - ln n_b`. Under decay the stored weights are undecayed
against the scale and run to `1/1e-50`, so each log is up to about 115 and their
difference, of order one, kept only about 1e-14: measured 3.5e-15 (`h = 5`) and
1.2e-14 (`h = 50`) from an `np.longdouble` longhand, against 1e-17 and 4e-16 for
the same table read in `long double`. Read as `ln((n_ab/n_a)(N/n_b))`, whose
argument is of order one and carries three roundings, the decay rows above fell
to 4e-16, and against `np.longdouble` the read is now:

| N | bins | MI | abs gap | rel gap | `H + H - H` form | `mutual_info_score` |
|---|---|---|---|---|---|---|
| 200,000 | 32 | 0.0024 | 1.9e-18 | 7.6e-16 | 6.7e-16 | 3.0e-16 |
| 2,000,000 | 64 | 9.7e-04 | 1.2e-17 | 1.3e-14 | 1.5e-15 | 3.5e-16 |

(The first draft's read: 3.5e-16 at both.) The entropy form `H(X) + H(Y) - H(X, Y)`
subtracts entropies of a few nats to leave an MI of 1e-3, and loses the most;
the Rust model should read the ratio form. The plug-in MI's own sampling bias,
`(b-1)^2/(2N)`, is 1e3 to 1e12 times any of these: rounding never matters beside
it.

**Decay's scale.** The fold multiplies each cell by `s` once (one rounding),
where a cell decayed every row rounds once a row (`3.5e-15` relative after 6,000
rows in the longhand); a fold leaves cells as small as `1e-50` of the largest
without underflow.

**Window subtraction.** Exact for unit weights (integers below `2^53`).
Fractional weights leave a residue of about `1e-14` a cell after 1e5 rows and
false nonzero cells (§5); MI moves by 3e-15. A Rust model wanting exact
fractional-weight windows can keep a per-cell row count beside the weight (`4 bx
by` bytes a pair in `u32`) and zero a cell when its count does, or rebuild the
table from the ring every `W` rows (`O(W k T)`, amortized `O(k T)` a row).

**Ranks.** Kendall's counters are integers held exactly to `2^53`: `n_0 =
W(W-1)/2` stays exact for `W` up to about 1.3e8, and each row's products sum at
most `W` terms of `0/±1`, exact. Spearman's centred ranks are multiples of 1/2,
their products sum to at most `W^3/12` (8e10 at `W = 10,000`), exact; only the
final division and square root round, hence 2e-16 to 3e-16 from scipy. Xi is a
ratio of integers. Ties are decided by `==` on the doubles, as scipy decides
them: two values a rounding apart are distinct to both. **No rank measure loses
precision with the stream's length or level**; what costs is time, not digits.

## Proposed tolerances for the Rust model's tests

| check | tolerance | reason |
|---|---|---|
| plug-in MI vs `mutual_info_score` (unit weights, window or none) | 1e-13 absolute | measured 1.3e-15, all of it scikit-learn's rounding; 1e-13 leaves 75x for a libm `ln` a few ulps apart (rule: libm results compared within the smallest tolerance that passes) |
| Miller-Madow vs the plain-Python definition | 1e-13 absolute | measured 1.8e-15; the correction itself is a ratio of integers |
| decayed MI vs scipy `entropy` on the weighted table, or a longhand | 1e-12 absolute; table 1e-12 relative | measured 1.3e-15 and 3.5e-15; the longhand rounds once a row, so allow for long streams |
| Gaussian band | each seed `\|MI_MM - I_bin\| < 4 sd + 0.2 (b-1)^2/(2N)`; `0 <= I - I_bin` below 0.14, 0.06, 0.026 at 8, 16, 32 bins | statistical: the ledger's kind (`oracles.toml`), tightening only; `I_bin` by quadrature to 1e-14 |
| `spearmanr`, `kendalltau` (b), `chatterjeexi` on the window's rows | 1e-14 absolute | measured 3.3e-16, 2.2e-16, 0; integers until the last division |
| xi with ties in `x` | 0 against the stable-order longhand | scipy's sort order is not promised; the longhand is the oracle there |
| ring contents, Kendall pair counts, unit-weight MI table vs a rebuilt one | bit-equal | integers and copies, nothing rounds |
| chunk invariance, held values between reads | bit-equal | rule 3 |

## Caveats and what was not done

- **Decayed ranks are not built**, as the scan says: a decayed Spearman or
  Kendall has no oracle (`weightedtau` weighs by rank, not by age).
- **Miller-Madow under decay or fractional weights is not built**: which `N`
  (the table's weight, or Kish's `W^2/V`) is a choice no library checks, and §5
  shows a fractional window miscounts nonzero cells.
- **Weighted ranks**: a positive weight only admits a row. Integer frequency
  weights could be checked against scipy on `np.repeat`ed rows; not done.
- **xi with ties in `x`** is checked against the arrival-order longhand; the
  paper's random tie-break is not reproduced (it would need a random stream in
  the state).
- Normalized and adjusted MI, the contingency statistics (chi-square, Cramér's
  V), Somers' D and distance correlation (the scan's further rows) are not in
  this study. Distance correlation's oracle, `dcor`, is not installed.
- The cost is numpy's, single-threaded, on this machine; the Rust model's is the
  task's to measure. The `W = 10,000` Rust row cost above is an estimate.
- p-values for the rank measures are not studied; under serial dependence they
  are wrong, as `marginal`'s `serial_rule` warns.
- MI's bins come from a warm-up and then stay fixed; edges that track a moving
  distribution would move rows between cells (the scan: not online).
