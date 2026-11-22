<!-- kind: record -->
# `tdigest`: the design note

The bank's `tdigest` kind (PLAN 544, parts 2 to 5; the study on
`vm/study-544`, `docs/records/STUDY-544-TAILS.md` there). The core digest is
`crates/rheon-core/src/tdigest.rs`; the model over it, one digest a target,
is `tdigest_model.rs`.

## The update

For target `t`, level `p`, a row of weight `w` and value `y`:

```text
q_p,t = D_t.quantile(p)        read before the row (hard rule 2)
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
rows of weight 1 it is the row count. The median reads from the second row,
0.05 and 0.95 from the 20th, 0.01 and 0.99 from the 100th.

The model writes NaN for such a level, and the stream names it with a new
`withheld_reason`, `level_beyond_sample`, appended after `unstable_rate` (a
code's number is kept in a state's last row, so a new reason takes the next
one). It is named only where no gate before it withheld anything, as
`ill_conditioned` and `unstable_rate` are. Open for the coordinator: the
reason's name, and whether a row where some levels are reported and some
withheld should carry the reason (it does, as every gate's reason is "the
first that withheld anything").

`compression` takes the core's range, 10 to 100,000, not any positive
value: under 10 the limit's series and the digest's few centroids are no
longer worth a sketch (part 2's choice, held by its tests).

## State and cost

At most `delta + 1` centroids and `8 delta` buffered rows a target, 16
bytes each, plus the header: about 14.5 KB a target at 100 and 29 KB at 200,
checked at the call against `max_state_bytes` (256 KB by default). A read
merges a copy of the buffer, `O(delta log delta)` a target a row and an
allocation of the copy; the scaling cells declare it (`tdigest.compression`
at `n^(5/4)`). A cheaper read that leaves the numbers to the bit is a
candidate for PLAN 545's performance round.
