# `hawkes`: the design note (task 498)

The model, its alternatives and the measurements behind its settings. The
update equations are in `crates/rheon-core/src/hawkes.rs` and in
[MODELS.md](../MODELS.md#hawkes--how-much-does-each-event-excite-the-next);
the study that derived them is `docs/records/STUDY-498-HAWKES.md` on the
branch `vm/study-498` (numpy, tick and scipy; 2026-11-20).

## What was built

- **Exact online least squares** for a Hawkes process with exponential
  kernels at given decays, any number of nodes (the targets) and decays: the
  contrast `Σ_i ∫ λ_i² dt − 2 ∫ λ_i dN_i` is a quadratic form in the
  parameters, solved at every row for the intensities the row reports.
- **The state is the study's form C, with one change.** `G` is read from
  `b` at each read, so the state is `R` (`k`), `b` (`D p`), `G₀₀` and the
  weight. Form C kept the count of each node's events for `G`'s diagonal
  blocks, which is right only when no two nodes have an event on one row.
  The jump of `R_a R_b` at a row of events has the term `m_j(a) m_j(b) β_a
  β_b`, so the state keeps `Q = Σ m mᵀ` over the rows, `D (D + 1) / 2`
  numbers. A first build without it failed the longhand on every stream
  with rows of several nodes' events (`G` off by 27% of its largest entry).
  The state is `8 (k + D (1 + k) + D (D + 1) / 2 + 2)` bytes: 64 for one
  node and two decays, 200 for two nodes and three, 3,176 for ten nodes and
  three.
- **Marks.** A target's value above 0 is an event of that mark (a count of
  simultaneous events, or a size); `Q` makes marks free. The row weight
  only gates a row's events (0 learns none): a weight that scaled the
  excitation would square in `Q` and pass the input bound's square at a
  weight and a mark of `1e100`, and one that scaled only the contrast's
  event term would break the identity that reads `G` from `b`. So scaling
  every weight moves nothing but `weight_sum`, and `test_weight_scale`
  needs no exception.
- **Forgetting per clock unit**, the study's choice (per-event forgetting
  would cost `O(k²)` a row). `γ` within 0.1% of a decay or a sum of two is
  refused; the divisions lose `log10(β / |β − γ|)` digits there.
- **The window counts rows with an event** (`window_events`), the task's
  bound, so its size is known at the call. Its removal is exact through the
  forward sums of the dropped row's excitation, in two blocks (the study's
  section 7, the two-block form it derived and did not run). Every row holds
  `8 (1 + D + k)` bytes in either block: a first layout held the newer
  block's rows without their sums, which made the state's size depend on
  where in the block's turnover it was read (the scaling cell measured
  10.8x for an 8x window). `R`, `b` and `Q` are formed afresh from the rows
  each time a block turns over, so the window's rounding is one window's.
- **A fit the events cannot identify is withheld**: a pivot of `G` scaled to
  a unit diagonal below `1e-10`. The contract's zero-weight probe, at a
  level of 1,000, feeds two nodes events on every row with marks a constant
  apart; `G`'s scaled condition number is 3.5e14 there, and the exact
  solutions of two streams differing only in rounding gave intensities
  1e-5 apart and coefficients of ±450,000.
- **Unconstrained.** tick's learners project onto `θ ≥ 0`; this fit does
  not, so a negative `α` (inhibition) is reported as the data give it.

## The state bound (PLAN 537)

The one part of the state whose size the data could set is the window's
rows, and `window_events` bounds it, so a group's whole state is
`Hawkes::state_bytes(D, U, W)` bytes, known at the call. The spec's
`max_state_bytes` (default 256 KB) is checked by `Spec::validate`: past it
the spec is refused naming the bytes, and the Python builder warns past
half of it. `the_state_never_passes_the_bytes_its_settings_give` holds the
encoded state to the formula exactly once the window is full, and
`a_hawkes_window_past_its_state_cap_is_refused` and
`test_the_window_is_held_to_the_groups_state_cap` drive the window past
the cap. Nothing else grows: the solve's scratch is a thread's, bounded by
the shapes it solves.

## The oracles and what they measured

| check | where | worst gap |
|---|---|---|
| `G`, `b` against the definition's double sums, 9 cases, `γ` from 0 to 12 | `hawkes/tests.rs` | 1.2e-13 (at `γ = 12`, past every decay sum); 1e-14 elsewhere |
| the fit against the longhand's LU solve | `hawkes/tests.rs` | 1.0e-12 (at `γ = 12`); 3e-14 elsewhere |
| the window's `G`, `b` against the longhand on its rows, 4 shapes, every third row | `hawkes/tests.rs` | 1.2e-13 |
| tick's `ModelHawkesSumExpKernLeastSq` minimiser, 1 to 3 nodes, 2 or 3 decays | `test_second_opinion_hawkes.py` | 3.9e-14 of the largest coefficient |
| tick's `ModelHawkesExpKernLeastSq`, one shared decay | the same | 1.2e-15 |
| tick refit on the window's events, 15 reads over 1,855 rows | the same | 1.1e-14 |
| tick's loss times its event count, against the fit's own contrast | the same | within 1e-11 |
| a row's intensity against tick refit on the rows before it | the same | within 1e-10 |

A parameter recovery from Ogata's thinning (`μ = 0.5`, `α = 0.6`, `β = 2`,
25,178 events) fits 0.507 and 0.597.
