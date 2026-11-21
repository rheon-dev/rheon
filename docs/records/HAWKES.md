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
  `8 (2 + D + k)` bytes in either block (its clock, its weight, `D` marks
  and `k` sums): a first layout held the newer block's rows without their
  sums, which made the state's size depend on where in the block's turnover
  it was read (the scaling cell measured 10.8x for an 8x window). `R`, `b`
  and `Q` are formed afresh from the rows each time a block turns over, so
  the window's rounding is one window's.
- **The window of clock units** (`window_size`, PLAN 538: every span a model
  takes accepts a count of rows and a duration on the clock where it has
  meaning in both) holds the events of the last `D` clock units. It is the
  same removal and the same two blocks; what changes is when a row leaves
  and what is observed.
  - *When a row leaves.* At each row, before its intensity is read, every
    held row `D` old or older leaves (`Right`, the default) or older than `D`
    (`Both`): the window kinds' edge (`WindowClosed`, Polars'
    `rolling_*_by(closed=)`), decided on the rows' exact stamps
    (`Stamp::cmp_span`: integer nanoseconds on a temporal clock, one
    subtraction of raw values on a number clock) that the stream hands each
    learned row, and on the model's own summed clock where none is handed.
    A row's stamp is held with it (32 bytes). A clock summed in doubles
    misreads an edge: steps of 100 ms against a span of 700 ms put a row
    exactly 700 ms old on the wrong side in the unit test's stream
    (`the_clock_window_reads_its_edge_off_the_stamps`).
  - *What is observed.* The interval `(T − D, T]`, or from the first row
    until there are `D` units of clock: `G₀₀ = min(D, the clock since the
    first row)`, which is exactly tick's observation window over the span's
    events (`test_the_clock_window_is_tick_refit_on_the_events_of_its_span`,
    1.5e-14 of the largest coefficient). The alternative, observing from the
    clock of the last row dropped as `window_events` does, makes the window
    longer than `D` by every stretch of quiet after its oldest event, so it
    is not "the last `D`".
  - *`predict`* cuts the window at the row's clock on a copy, by the step's
    own routine, so a read ahead is the step's answer to the bit; it needs
    the last stamp handed, advanced by the step, which the state keeps.
  - *Both windows together are refused*: the observation start of the one
    is the dropped row's clock and the other's is `T − D`, and what the pair
    would mean is not a fit anyone asked for.
- **The weight under a window is the weight inside it** (hard rule 8,
  "under a `window_size`, `n_eff` is the weight inside the window"). A first
  build kept every row's weight, undecayed, which is not the window's. The
  model holds each event row's weight (8 bytes a row) and sums them; a row
  without an event is no part of the fit, and the model keeps no record of
  it, so its weight is not counted: the one deviation from "every row", made
  because counting it would need every row of the span (an unbounded count
  with no event in it). Read before the row's own clock step, as every
  `n_eff` is, so a row that leaves at this row is still in what the row
  reports and the accessor equals the step's `n_eff`. A `min_weight` above
  the weight a window can hold reports nothing: the contract's windowed
  cases set 3 for this.
- **Every clock works the same way** (PLAN 538: "if the clock is a time then
  5m should work"). On a temporal clock `window_size` is a duration, read off
  the exact stamps, and the kernels are given by their half-lives
  (`kernel_half_lives`, numbers or durations, `β = ln 2 / half-life` in the
  clock's unit: seconds on a temporal clock), mutually exclusive with
  `kernel_decays`, as Polars' `half_life` gives a decay. A rate has no
  duration form, so `kernel_decays` stays refused there. The core takes
  decays in the clock's unit and the spec converts them, so the temporal
  clock is the library's own: integer nanoseconds, a step's seconds by
  `seconds_of_ns`. Terms are named by the kernel as given: `beta2`, `h500ms`.
  On a temporal clock the intensity and `mu` are events per second (the
  library's convention: a quantity measured on a temporal clock reaches the
  output in seconds, as `holt`'s trend). Tests: the same model on a Datetime
  clock and on a numeric clock of seconds with the decays converted by hand,
  to the bit on steps of a quarter second and within 1e-13 where a window
  holds 5 events or more, on millisecond steps (a numeric step is a
  difference of rounded seconds, a temporal one the seconds of an exact
  integer; a span on a row's exact edge is read the other way by the numeric
  clock, so those tests' spans sit between two rows' ages).
- **A zero-weight row counts in no clock** (hard rule 9, PLAN 539). The model
  reads such a row from the last row that counted (a read ahead on a copy)
  and keeps only the clock it stepped over, `carry`, which the next counted
  row adds to its own step: one advance by the sum, so the other rows'
  outputs are the stream-without's to the bit. Before any row has counted the
  zero-weight rows' steps are dropped, and the first row to count is the
  model's first. The row is in no window and none of `n_eff`. The model
  sums the steps it is handed; a stream that hands each row the step from
  the last counted row must drop the carry.
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
rows. `window_events` bounds it, so a group's whole state is
`Hawkes::state_bytes(D, U, W)` bytes, known at the call: the spec's
`max_state_bytes` (default 256 KB) is checked by `Spec::validate`, past it
the spec is refused naming the bytes, and the Python builder warns past half
of it. A `window_size` window holds the events its span has, so it is
checked as the rows go in, by `Hawkes::clock_state_bytes(D, U, rows)`: the
first row to take the state past half of the cap sets a flag (held in the
state, so a group says it once across a save and load) and the model hands
the stream `(bytes, cap)` (`OnlineModel::take_state_warning`), which becomes
a notice the bank says with the spec and the group; a row that would take
the state past the cap is not learned, nor any after it, and the model
reports `(bytes, cap)` (`OnlineModel::state_over_cap`), on which the bank
stops the run with an error naming the spec, the group and the bytes, as a
window ring past its refusing budget does (the bank has learned part of the
chunk, so it is `broken`). The check is a refusal at the cap and not a
thinning: dropping the oldest events would change the fit the span names.
`the_state_never_passes_the_bytes_its_settings_give`,
`a_clock_windows_numbers_are_the_bytes_it_counts` and
`a_burst_of_events_is_warned_of_at_half_the_cap_and_stopped_at_it` hold the
formulas to the encoded state and the bound to a burst;
`a_burst_warns_once_and_is_refused_at_the_cap` and
`test_a_burst_in_a_clock_window_is_warned_of_once_and_refused_at_the_cap`
drive it through the bank. Nothing else grows: the solve's scratch is a
thread's, bounded by the shapes it solves.

## The oracles and what they measured

| check | where | worst gap |
|---|---|---|
| `G`, `b` against the definition's double sums, 9 cases, `γ` from 0 to 12 | `hawkes/tests.rs` | 1.2e-13 (at `γ = 12`, past every decay sum); 1e-14 elsewhere |
| the fit against the longhand's LU solve | `hawkes/tests.rs` | 1.0e-12 (at `γ = 12`); 3e-14 elsewhere |
| the window's `G`, `b` against the longhand on its rows, 4 shapes, every third row | `hawkes/tests.rs` | 1.2e-13 |
| tick's `ModelHawkesSumExpKernLeastSq` minimiser, 1 to 3 nodes, 2 or 3 decays | `test_second_opinion_hawkes.py` | 3.9e-14 of the largest coefficient |
| tick's `ModelHawkesExpKernLeastSq`, one shared decay | the same | 1.2e-15 |
| tick refit on the window's events, 15 reads over 1,855 rows | the same | 1.1e-14 |
| tick refit on the events of the last 25 to 80 clock units (`window_size`), observed over the span, 355 reads, three stream shapes, both edges | the same | 4.6e-14 |
| the clock window's `G`, `b` against the longhand's, every row, both edges, steps in quarters so rows sit on the edge | `hawkes/tests.rs` | 1.8e-15 |
| tick's loss times its event count, against the fit's own contrast | the same | within 1e-11 |
| a row's intensity against tick refit on the rows before it | the same | within 1e-10 |

A parameter recovery from Ogata's thinning (`μ = 0.5`, `α = 0.6`, `β = 2`,
25,178 events) fits 0.507 and 0.597.
