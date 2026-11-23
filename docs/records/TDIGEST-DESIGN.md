<!-- kind: record -->
# `tdigest`: the design note

The bank's `tdigest` kind (PLAN 544, parts 2 to 5; the study on
`vm/study-544`, `docs/records/STUDY-544-TAILS.md` there). The core digest is
`crates/rheon-core/src/tdigest.rs`; the model over it, one digest a target,
is `tdigest_model.rs`.

## The update

For target `t`, level `p`, a row of weight `w` and value `y`:

```text
q_p,t  = D_t.quantile(p)       read before the row (hard rule 2)
es_p,t = the centroids' mean past q_p,t, the boundary one split
D_t   <- D_t + (y, w)          w > 0 and y usable
W_t   <- W_t + w,   S_t <- S_t + w^2
```

`D_t` is Dunning and Ertl's merging digest (2019) on `k_1`, its buffer
`8 delta` rows; a read merges a copy, so the state is the rows' alone. The
merge limit `k^-1(k(q) + 1)` goes through the sine addition formula with
`sin` and `cos` of `2 pi / delta` from their Taylor series and a correctly
rounded `sqrt`, so no libm result decides a merge: the state and the reads
are the same bits on every OS, and the fixtures are held to the bit across
platforms (`state_fixtures.rs`, as `garch`'s).

## Decisions (the coordinator, 2026-11-22, from the user's rules)

- Kind `tdigest`; `compression` (default 100) and `quantiles=[...]` at the
  spec; one output a level and a column, `q<level>_<column>`, as the
  residual quantiles name theirs (`abs_resid_q<level>_<target>`).
- One digest a target column in each group. The columns are `targets`, as
  `garch`'s returns are; the kind takes no `features`.
- The levels, `weight_sum` and `withheld_reason` (with `settled_frac`, which
  every row-writing model carries); no `pred` or `coef`.
- The builder in its own file, `python/rheon/_spec_tdigest.py`. `_spec.py`
  stood 159 bytes under the source cap, so its table of keys skipped when
  absent moved to `_spec_skipped.py` to make room for the registration.
- No window state. A `half_life`, `lam`, `window_size`, `window_rows` or
  `span` is refused at spec build naming the windowed DDSketch
  (`ewm_quantile` with a `window_size`): a merging digest cannot take a row
  out. `half_life` and `lam` are refused in `Spec::validate`; a window key,
  which the variant does not have, is refused by serde as unknown, and
  `name_renamed` adds the reason.

## Withholding a level (the user's rule: no fixed row count)

A level is withheld while the rows do not identify it, by the condition
itself: with Kish's `n_t = W_t^2 / S_t`, the `i`-th order statistic of `n`
rows sits at level `i / (n + 1)` in expectation (the Weibull plotting
position, Hyndman and Fan's definition 6), so a level outside
`[1 / (n + 1), n / (n + 1)]` lies beyond the smallest or largest row and the
digest can only read that extreme. Kish's size, not `W_t`, so that scaling
every weight changes nothing but `weight_sum` (the weight-scale rule); for
rows of weight 1 it is the row count. Levels of 0.25 and 0.75 read from the
4th row, 0.05 and 0.95 from the 20th, 0.01 and 0.99 from the 100th.

The model writes NaN for such a level, and the stream names it with a new
`withheld_reason`, `level_beyond_sample`, appended after `unstable_rate` (a
code's number is kept in a state's last row, so a new reason takes the next
one). It is named only where no gate before it withheld anything, as
`ill_conditioned` and `unstable_rate` are. Open for the coordinator: the
reason's name, and whether a row where some levels are reported and some
withheld should carry the reason (it does, as every gate's reason is "the
first that withheld anything").

`compression` keeps the floor the algorithm needs, 10 (the merge limit's
sine and cosine of `2 pi / delta` are summed from series that converge
within 30 terms there, and under it the digest's few centroids are not
worth a sketch), and has no ceiling: `max_state_bytes` bounds it (the
coordinator, 2026-11-23, part 5's exchange B).

## Expected shortfall (PLAN 544, part 6)

The coordinator's answer (2026-11-23, part 5's exchange B): emit
`es<level>_<column>` only, no `var<level>`, since `q<level>_<column>` is the
value at risk already; the tail is chosen by the level, below 0.5 the lower
(`E[X | X <= q]`), above it the upper (`E[X | X >= q]`), exactly 0.5 refused
by name; no sign flip; the same Kish condition withholds it.

The first build took the mean of the read's interpolated curve over the
tail (knots at the minimum, each centroid's centre and the maximum,
integrated a trapezoid a piece). Measured over 20 seeds at N = 100,000 it
was within 2.1e-2 of the exact tail mean on a normal stream but 0.97
relative on Pareto's upper 0.99 tail (0.37 at compression 200): the end
pieces run from the last centroid's mean to the largest row over half that
centroid's weight, so a heavy tail's mean carried the maximum. The
coordinator's second answer (2026-11-23, parts 6-7) chose the partial-mass
rule over the centroids, keeping the boundary split and the 0.5 refusal:

```text
ES_p = (sum_{i<j} w_i c_i + (m - W_{j-1}) c_j) / m,   m = p N (p < 0.5) or (1 - p) N (p > 0.5)
```

Measured over 20 seeds at N = 100,000, largest over six levels, relative to
the exact tail mean: compression 100 normal 1.15e-2, lognormal 4.32e-2, t3
2.13e-2, Pareto 1.94e-2; compression 200 3.28e-3, 1.19e-2, 5.13e-3,
4.32e-3. The `tdigest` package's `trimmed_mean` (the same rule over its
much finer digest) is within 6.5e-5 of exact. The tests hold the
definition in fractions from the centroids and the measurements with a
margin of 1.5 (`tests/test_tdigest.py`).

## State and cost

At most `delta + 1` centroids and `8 delta` buffered rows a target, 16
bytes each, plus the header: about 14.5 KB a target at 100 and 29 KB at 200,
checked at the call against `max_state_bytes` (256 KB by default). A read
merges a copy of the buffer, `O(delta log delta)` a target a row and an
allocation of the copy; the scaling cells declare it (`tdigest.compression`
at `n^(5/4)`). A cheaper read that leaves the numbers to the bit is a
candidate for PLAN 545's performance round.
