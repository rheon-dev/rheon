<!-- kind: record -->
# The exceptions to state continuation (CLAUDE.md hard rule 5)

Each is the user's decision, numbered in the order given; a code comment or
test that cites "rule 5's Nth exception" means the Nth below. Moved
verbatim from CLAUDE.md on 2026-11-14 (the user: cut the size of the docs
read most). A new exception is added here, numbered next, and counted in
CLAUDE.md's rule 5.

One exception, the user's (2026-10-24, task 275): a *windowed* state of
schema 61 or 62 continues within the golden tolerance, not to the bit,
once a snapshot taken after its load is the window's edge (those states
never recorded the per-row decay factors); its first read is exact.
A second, the user's (2026-10-25, task 278): a `kmeans` state of schema
61-63 saved before it seeded seeds by the new rule after loading -- the
same centres and distances to the bit, the cluster labels possibly
permuted, held to one consistent relabelling.
A third, the user's (2026-10-25, task 278): a `lasso` state of schema
61-63 continues within 1e-9 relative, not to the bit (the stopping rule
became relative to the target's spread), and still loads and re-saves the
same bytes.
A fourth, the user's (2026-10-25, task 278): a diagnosed state of schema
61-63 continues to the current values in its windowed `cusum_*` and
`cusum_sq_*` fields while their sums are young (their null's share is now
read at the sums' age); every other field continues to the bit.
A fifth, the user's (2026-10-26, task 278): a `holt` state of schema
61-63 loads with no line fit and continues with the textbook gains it
was fitted under, within 1e-14 in its predictions and state, and its
diagnostics derived over many rows within the golden 1e-12 (to the bit
until task 291a's kept reciprocals, the user's, 2026-11-01); its re-saved bytes cannot be the current
fixture's (which carries the line fit), so it is held, beside `kmeans`,
to re-saving bytes that load back the same.
A sixth, the user's (2026-11-01, task 294): the co-moment matrix every
`EwCov` keeps is now stored as its lower triangle and read through its
mirror, so a state of schema 61-63 continues within rounding, not to the
bit, where it read an upper entry: an `ew_cov` state's covariances and
correlations within 1e-15 by |Δ|/(1+|x|), and a diagnosed state's
specification, calibration, break and `ic` statistics (and a closed
group's exported `comoments`) within the golden 1e-12; every prediction
continues to the bit, `lasso` stays inside its own 1e-9, and each state
still re-saves the current fixture's bytes exactly.
A seventh, the user's (2026-11-03, task 311): an `ewridge` state saved
by an earlier build no longer runs a solve its weight schedule brings due
while the weight is below `(1 − s)(1 − 2⁻¹⁶)` of its lowest floor. It
continues with every residual, spread, weight and Gram to the bit, and
every prediction to the bit except where a target's weight has decayed to
exactly 0 below the floor or a fit's factorization fails at every rung:
there a row may read the last fit solved rather than the skipped one.
Until the next solve run, its `coef`, `support_coef` and `se_coef` report
the file's fit and `solve_failures` counts only the solves run, so the
state it re-saves after those rows is not the old build's. No frozen
fixture reaches it.
An eighth, the user's (2026-11-05, task 323; amended 2026-11-05, task
340): a state of schema 64 or earlier that holds a MAD scale sketch
(`huber`, `quantile`, `sgd(loss="huber")`, `marginal`'s huber bins)
keeps the rule it was saved under -- a sketch awaiting its anchor takes
its first value, a warm-up with no held residuals reads its sketch --
marked in its bytes (`anchor_by_first_value`, `before_65`). It continues
to the bit, but re-saves to its own marked bytes, not the current
fixture's, so it is held, beside `kmeans` and `holt`, to re-saving bytes
that load back the same (`kept_scale_rule`). A converted `marginal`
state re-saves so only until its histogram is built: `before_65` acts at
the freeze and is cleared there (schema 67), so from the freeze it
re-saves the current form; a state of 65 or 66 saved past its freeze
still carries the mark and loads without it, converting to the current
fixture's bytes exactly (`marginal_bins_refit_first_value_frozen`, kept
under `v66/` by the 66 build).
A ninth, the user's (2026-11-05, task 328): a `corrchange(kind="window")`
state of schema 65 or earlier loads numbered from its ring's oldest row
(it never recorded `ring_rows`), and where its windows span four blocks
or more its statistic is merged from block moments rather than read from
the rows, so `stat` continues within the merged form's derived rounding
bound (held at 1e-14 of `1 + |stat|`, measured 4e-16), not to the bit;
every other output continues to the bit, a kept `permute_every_rows`
stays as saved, and its re-saved state equals the current fixture's but
for `ring_rows` (`RENUMBERED`).
A tenth, the user's (2026-11-05, task 327): a windowed `ew_class` under
`covariance_type="diag"` now counts each snapshot class at its k
variances, not k² co-moments, against `window_budget`, so a state of
schema 65 or earlier that had reached its budget thins or refuses later
than it would have, and from that row does not continue to the bit. No
frozen fixture reaches it; every state below its budget continues to the
bit.
An eleventh, the user's (2026-11-05, task 331): `refit_budget`'s
default rose from 256 to 320 MiB, and the spec keeps only a budget the
user gave, so a bank saved under the old default reads its saved
reservation against 320 MiB when loaded. The groups it gave up stay
given up and every group it holds continues to the bit, but a group it
first sees after the load refits until 320 MiB is reserved, where the
old build would have given it up. No frozen fixture reaches the
default budget.
A twelfth, the user's (2026-11-08, task 364): an `ewridge` state
saved by an earlier build no longer runs a solve its row cap brings due
under a weight share (the default schedule under a finite half-life)
while the weight is below `(1 − s)(1 − 2⁻¹⁶)` of its lowest floor, as it
already ran none the weight brought due there. It continues with every
residual, spread, weight and Gram to the bit, and every prediction to
the bit except in the seventh exception's two cases. Until the next
solve run, its `coef`, `support_coef` and `se_coef` report the file's
fit and `solve_failures` counts only the solves run, so the state it
re-saves after those rows is not the old build's. A given
`solve_every`'s clock and row cap still run every solve they bring due.
No frozen fixture reaches it.
A thirteenth, the user's (2026-11-09, task 368): a conformal tracker
now keeps a multiplier `m` of the residual spread, the interval being
`pred ± m·σ`, where a state of schema 66 or earlier holds the radius
`q`. Such a state loads with `q` kept, re-saves it, and shows it to the
bit on its next row; that row, if it reads an error scale, converts it
to `m = q / σ` and steps `m`, so from there its intervals and coverage
go on under the multiplier, not to the old build's, and the state it
re-saves is not the old build's. Every other field continues to the
bit. No frozen fixture holds a conformal tracker;
`a_conformal_radius_of_schema_66_is_shown_then_converted` holds the form,
written from a current bank, and `tests/test_released_state.py` each
release's file (`CONVERTED`: each group's first row to the bit).
A fourteenth, the user's (2026-11-15, task 405): a `lasso` coordinate
descent that reaches `max_iter` while its Gram has too few effective rows
(Kish n) for its features to settle in that many sweeps on any data -- the
smallest eigenvalue such rows can give, by the Marchenko-Pastur edge, is
below what `max_iter` sweeps need to reach `tol` -- is no longer counted in
`solve_failures`. A `lasso` state saved by an earlier build loads and
continues with every prediction, coefficient and other field to the bit;
its `solve_failures` grows by the counted cap hits only, so where a later
descent of its warm-up runs out of sweeps the count it re-saves is lower
than the old build's would be. The current fixture `lasso_window`
regenerated by one byte (its count 2 to 0); no schema bump.
A fifteenth, the user's (2026-11-17, task 457, landed in task 461): a
window operator's held mass -- the time a row's value is held over, `c (1 -
2^-x)` of the gap `x` half-lives before it, the partner of that row's
discount -- is now taken from the discount's own `exp2` for a gap of at
least a hundredth of a half-life (one `exp2` for a row's step and its mass),
where it was `exp_m1` at every gap; below a hundredth of a half-life it is
`exp_m1` still, so every row of such a gap is as it was. The two forms agree
to a few ulps of the mass (at most `2^-52 / (1 - d)`, 1.6e-14 at the edge and
less above), which is the whole difference: outputs of `ewm_mean`,
`ewm_std`, `ewm_var` and the mean and variance of a windowed operator move by
a median of a few ulps over the rows that move, and by at most about 1e-13
relative (1.05e-13 in the landing's re-measurement over 188 cases and 90M
rows, at a floor of 0.01 on the value, the largest at a 50 ns half-life and
1 ns steps; 52 cases move, and none flips a null, a non-finite value or a
decision), where `ewm_sum`, `ewm_rate` and `ewm_quantile` do not move.
A window run's state, or a bank's holding a window formula, saved by an
earlier build loads and re-saves the same bytes; its queue's sums hold the
bits the old form made for the rows before the save, so it continues within
that bound of the old build's run, not to the bit, once a row after the
load has a gap of a hundredth of a half-life or more, and the sums it
re-saves after that row are not the old build's. The user took the change
for its speed on 2026-11-17 (task 457: a window operator on an irregular
clock 14-16% faster on arm64 with one operator, 0.83-0.96 of the time on
x86; outputs moving by a median of 1-2 ulps and at most about 7e-14), and
the regeneration below is under this exception. The window digest's pinned
bits and the current `with_windows` and `with_windows_int` fixtures were
regenerated (no layout change, no `SCHEMA_VERSION`); no other fixture moved
(`with_windows_quantile` and every bank fixture re-save as before). The kept
previous-schema fixtures of window runs (`v62/`, `v63/`, `v66/`'s
`with_windows*`) are held to the golden 1e-12 against their frozen frames
and to re-saving bytes that load back the same, where the current set stays
to the bit (`moved_by_the_held_mass` in `state_fixtures.rs`).

A sixteenth, the user's (2026-11-17, task 457; built in task 462):
`bocpd` under the diagonal covariance, `covariance_type = "diag"` (with
`robust_beta` as well), and two features or more, takes the logarithms of
its predictive differently. It adds the features' `ln ψₙ` and
`ln(1 + z²/νₙ)` as the logs of products of blocks of 8 values (a block's
logarithm once, where its product is a normal number far from overflow, else
one at a time), and holds `ln(π(κₙ+1)/κₙ)`, the term `ln(νₙπ·var)` splits
into beside `ln ψₙ`, once for each run length (keyed by the run's weight),
where it took two logarithms for each feature of each run. The forms are
equal in exact arithmetic and differ by rounding, which the weight
`(νₙ+1)/2` of the second term scales. A diagonal `bocpd` state of two
features or more saved by an earlier build loads with every stored number as
it was and continues with `loglik`, `p_change`, `run_mean` and the
predictions parting from the old build's in their last digits, not to the
bit. Measured on 16 diagonal cases (2 to 200 features, 1, 100 and 10,000
groups, up to a million rows, a save and load, nulls) against the build
before: `pred` moved at most 2.0e-6 relative (a changepoint row; p99.9 at
most 5.0e-10, median at most 6e-14), `loglik` at most 2.0e-10 (median at
most 9.7e-14), `p_change` and `run_mean` at most 9.3e-10, with `robust_beta`
at most 1.1e-7 (p99.9 3.4e-10); no `run_mode`, `weight_sum`, withheld reason
or null changed, and no row crossed `p_change > 0.5` or `> 0.1`. The state it
re-saves is not the old build's. `bocpd` with one feature, with
`covariance_type = "full"`, and every other model continue to the bit, and
their states are the old build's byte for byte. The current fixtures `bocpd`
and `bocpd_warming` regenerated (their continuations moved by at most
1.4e-15 relative, each saved state by the last bit of one float); no schema
bump. The kept sets' two cases are held to that bound by `MOVED`
(`crates/online-core/tests/state_fixtures.rs`).

A seventeenth, the user's (2026-11-17, task 457: "keep the corrchange
improvement", after the per-seed difference below was put to them; built in
task 464): `corrchange(kind = "window")` draws the swaps of its permutation
critical value by Lemire's multiply-shift reduction (the high word of a
64-bit draw times the bound, with a draw whose low word is under
`2^64 mod n` redrawn, so it is exactly uniform) where it took the draw
modulo the bound (`x % n`, a 64-bit division a draw, biased by up to
`n / 2^64`). The generator is called once a draw as before, and a redraw has
probability under `n / 2^64`, so a state's saved generator is the old
build's after the same draws and nothing about the layout changes. What
moves is which permutations a seed visits: for a given `seed` and rows, every
redrawn critical value is a different draw from the same distribution, so it
parts from the old build's, and a flag whose statistic sits near the old or
the new critical value can differ. A window-kind state saved by an earlier
build loads with every stored number as it was, including the critical value
it holds until its next redraw, and its next redraw, from its saved
generator, takes the new draws; so it continues to the bit until that
redraw and parts from the old build's critical values from there, and the
state it re-saves is not the old build's. `corrchange` with `kind = "monitor"`
or `"sequential"`, which draw nothing, and every other model continue to
the bit. Measured against the build before (2026-11-18, arm64, the harness of
task 452: a window of 20, 100 and 400 rows, 4
columns, `n_perm = 200`, `perm_block` 1 and 10, 8 pools of correlated
Gaussian rows with 2,000 seeds each, 16,000 critical values a case): the
distribution of the critical value over seeds is the same (two-sample
Kolmogorov-Smirnov distance median 0.020-0.030 and at most 0.044 where the
5% critical value at 2,000 and 2,000 draws is 0.043; mean difference at most
0.076 of a standard deviation; standard deviation ratio 0.96-1.05; each
0.13-0.15 standard deviations above the critical value of 20,000 draws, the
bias of an order statistic of 200 draws, equal for both); a given seed's
critical value, paired against the old build's, moves by a median of 3.6% to
4.1% of its value (none identical), p99 15% to 16%, at most 32%. Over 3,000
streams of 3,000 rows (3 columns, correlation 0.3, and 0.8 from the middle for
the alternative): per-row flag rate 0.0511 against 0.0512 at window 100 (null,
standard error 0.0005), 0.0935 against 0.0933 (alternative); 0.0522 against
0.0522 and 0.0685 against 0.0685 at window 50; 0.0566 against 0.0567 and
0.0928 against 0.0928 at window 100 with `perm_block` 10; the median delay to
the first flag after the change 33 against 33 rows (22 against 22, 32 against
31). 1.60% to 1.71% of rows carry another `flag` than the old build gave in
the same stream; no stream changes whether it ever flags. The current fixtures
`corrchange_window` (core) and `bank_diagnosed` (bank), whose saved critical
values and redrawn ones moved, regenerated, with the two oracle hash guards
`corrchange.n_perm` and `truth_recovery.corrchange_break` re-run and
re-pinned; no schema bump. The kept sets are held by bounds: core's
`corrchange_window` in `v65/` and `v66/` is in `MOVED` at 0.25 of `1 + |x|`
(measured: the saved `perm_crit` 0.115, the redrawn `crit` 0.140; a flipped
`flag` would read 0.5, so none may), and the bank's `corr_window` `crit` in
`v65/` and `v66/` `bank_diagnosed` is in `CONTINUES_WITHIN` at 0.25 (measured
0.063), every other field staying to the bit.

An eighteenth, the user's (2026-11-18, "Yes option 1", the review of the
proposal to take faer's product from order 24; built in task 469, which
replaces that proposal's branch): the break
distance `break_wald` (`emit_breaks`) is formed in another way and moves at
every order. The variance of the two fits' difference was
`A_f H_ff A_f + A_s H_ss A_s − A_f H_fs A_s − A_s H_fs' A_f`, four terms
that are 500 times the sum or more over the first rows past the `2k` wait
and cancel to it. It is now `D H_ff D' + A_s E₂ A_s + C + C'` with
`D = A_f − A_s`, `E_ff = H_ff − H_fs`, `E₂ = H_ff + H_ss − H_fs − H_fs'` and
`C = D E_ff A_s`, each term the size of the sum: the same matrix in exact
arithmetic, nearer the exact one in floating point (against the exact
distance of the stored Grams and sums, to a thousand bits: the 12 largest
movers of 14,000 distances at order 32 with an intercept, relative error
geometric mean 1.4e-5 to 2.3e-7 and largest 2.5e-4 to 1.6e-5; ordinary rows
3.4e-11 to 2.7e-12; without an intercept 5.3e-5 to 1.1e-5 and 4.1e-10 to
7.4e-11; the first-row-past-the-gate rows of orders 4 and 20, 6.9e-6 to
4.7e-7 and 7.2e-6 to 9.9e-8; order 2's 12 largest, geometric mean 9.8e-6 to
4.3e-6 but largest 9.2e-5 to 1.9e-4, the one case where its largest is
not smaller; after a 4,000-step gap at order 24 the form gains little).
The products run as the plain loops' sums to the bit under 22 coefficients
and through faer's matrix multiply from 22 (the three full products; the
three half ones stay on the register tile), which add in blocks and fuse
where the machine does, so from 22 the distance also differs by the kernel
(about 1e-13 relative on ordinary rows, 1e-6 on the first rows past the
gate, and more near a singular variance), the same on every thread count
and chunking, and another platform's kernel can round it differently.
Measured by `scripts/compare_outputs.py`, main's release wheel against this
build's (arm64, polars 2.0.0), over 42 cases (1 to 200 features, 1 to
1,000 groups, 1 and 3 targets, chunks of 1 and 1,000 rows, a million rows,
nulls, gaps, a shift, a save and load): only `break_wald` moves, in every
case; 0 flips of its thresholds at `chi2(k+1)`'s 0.9, 0.99, 0.999 and
0.9999 values and of `zscore`, `drift`, `cusum` and `cusum_sq`, and 0 nulls,
NaNs or infinities appearing or vanishing. Relative movement, max / p99.9 /
median over the case's rows: one group of 100,000 to 200,000 rows, k = 4 1.8e-11 / 5.0e-15 /
5.2e-16, k = 12 8.2e-11 / 4.1e-14 / 1.6e-15, k = 20 1.5e-9 / 1.6e-13 /
3.6e-15, k = 32 2.3e-10 / 2.0e-13 / 3.8e-15, k = 64 6.6e-9 / 1.2e-11 /
1.5e-14, k = 100 6.8e-5 / 9.7e-11 / 2.5e-14, k = 200 1.8e-7 / 1.8e-10 /
1.4e-13, a million rows at k = 32 and 64 1.5e-10 / 1.5e-13 / 3.7e-15 and
5.1e-10 / 6.4e-13 / 1.2e-14; 1,000 groups of 300 to 400 rows (most rows
young, the first past the gate large) k = 1 1.7e-6 / 2.8e-11 / 1.2e-15,
k = 2 1.2e-5 / 6.2e-11 / 2.2e-15, k = 4 8.1e-7 / 6.9e-11 / 4.6e-15, k = 8
2.3e-5 / 2.4e-10 / 1.6e-14, k = 12 7.7e-4 / 5.2e-10 / 4.1e-14, k = 20
2.3e-4 / 8.6e-10 / 8.6e-14, k = 24 1.4e-4 / 1.2e-9 / 1.4e-13, k = 32
7.1e-4 / 2.4e-9 / 3.2e-13, k = 64 1.4e-7 / 1.4e-8 / 1.5e-11, k = 100
8.5e-6 / 7.0e-8 / 7.5e-11. Native, over 1,180,000 distances of random designs at
orders 1 to 48 (a decay with a half-life of 200 rows, 800 to 3,000 groups
each, the first seven rows past the gate, with and without an intercept, and
after a gap) D refuses no distance the four-term form reads and the
four-term form refuses 2 that D reads, and the four thresholds above flip on
none. On designs whose features are
collinear (to 1e-4 or 1e-7) the two forms refuse different rows, each where
its rounding breaks a factorization of a variance whose exact pivots span
fewer than six digits, in both directions in about equal number (a number
where the four-term form had none, and none where it had a number); where
the exact variance of the stored sums is not positive definite either form
can read a number. A saved state of an earlier build loads and continues
with every field to the bit except `break_wald`; its last row's
`break_wald`, kept in the state's last-row part, is the new build's, so a
state saved by the new build after a row with a distance differs from the
old build's by that float. The current fixture `bank_diagnosed` of
`online-polars` regenerated (17 lines: its `break_wald_y` fields move by rounding, 1 ulp
on the first row, within the golden 1e-12 throughout; no schema bump); every other fixture, the whole of
`online-core`'s, is byte for byte. The kept previous-schema fixtures
(`v61/` and later `bank_diagnosed`) are held to the golden tolerance
against their frozen frames in `break_wald_` (`quantiled`'s entry of
`CONTINUES_WITHIN`, 4.7e-16 relative here) and re-save the current
fixture's bytes.

A nineteenth, the user's (2026-11-20, task 526: "Accept as exception 19"):
a windowed `lasso` state saved by 2.0.0 or earlier continues to the bit in
every field except `solve_failures`. Once rows have aged out of the window,
a descent that runs out of `max_iter` sweeps now counts, where the window's
effective rows alone used to excuse it, so a continued state counts its cap
hits under the new rule (the rebuild after a clock gap longer than the
window included). No emitted column moves and the layout is unchanged. The
current fixture `lasso_window` of `rheon-core` regenerated: one byte, its
`solve_failures` 0 -> 1, after its clock gap at row 36; every other fixture
is byte for byte, and the kept `v61` to `v66` fixtures load and continue as
before.

A twentieth, the user's (2026-11-20, task 479: "Land it from k ≥ 13"):
`hmm` under `covariance_type = "full"` with 13 features or more keeps each
state, once seeded, as its weight, its means and the Cholesky factor of its
matrix, `M = σ·L Lᵀ` with `σ` in `[1/4, 1]`, and a learned row moves the
factor by plane rotations, where the co-moments were updated and factorized
afresh on the next row (schema 69). An `hmm` full-covariance state of 13 or
more features saved before schema 69, once seeded, holds co-moments: it
loads, converts its co-moments to their factor -- the factor its next row
would have read, to the bit -- and continues in the factor form. Its first
row reads as the old build's, to the bit; from the first row it learns, it
continues within the measured bound rather than to the bit: on 13, 20 and
32 features under half-lives of 20 and 400 rows and none, 2,000 rows a run
saved after 600 (`a_state_of_schema_68_loads_over_the_grid`,
`crates/rheon-core/src/hmm/tests_factor.rs`, macOS arm64), a filtered or
predicted posterior moved by at most 1.9e-13 from the old build's and
`loglik` by at most 6.8e-14 of `1 + |loglik|`, and no reported `state`
changed; the test holds them to 2e-12 and 1e-12. Every weight, mean and
ridge scale continues to the bit until the posteriors that weigh the rows
part. Such a state re-saves in the factor form: not the old build's bytes,
nor this build's for the same rows, whose factor was taken up at the
seeding and rotated since, but bytes that load back the same; its matrices
are within 4.7e-16 of the largest diagonal entry of this build's (the kept
`v68/hmm_full`, held to 1e-14, its 16-row continuation to 1e-13 of
`1 + |x|`, measured 5.3e-15, by `TAKEN_UP` in
`crates/rheon-core/tests/state_fixtures.rs`). A state saved inside its
warm-up loads in co-moments and takes its factor up at its seeding, so it
continues to the bit of this build's run. Narrower `full` states, `tied` and
`diag` keep their layout and continue to the bit, and re-save the current
fixture's bytes (`v68/hmm_full_narrow`, 12 features). The current fixture
`hmm_full` (13 features) is the factor form; every other fixture moved only
its schema number.

A twenty-first, the user's (2026-11-20, review P1: "Accept exception 21";
its holding of a row of weight 0 superseded by the twenty-fifth):
under `label_clock` a row of weight 0 with a label time is held and
learned at that time as a zero-weight step of the clock of the labels
learned, as `embargo` learns it, where a build before scored it and held
nothing; and a stream whose first row is skipped starts that clock at the
skipped row's label time (P1b), as `embargo`'s starts at its clock. The
layout is unchanged (a held row of weight 0 is a `PendingRow` with `w = 0`)
and there is no schema bump. A `label_clock` state of schema 67 to 69 saved
by a build before the fix holds none of the zero-weight rows it scored whose
label times had not passed at the save, and the rows it holds, those a full
buffer dropped among them, were chosen without them; no loader can restore
rows never written. It loads, re-saves its own bytes, and continues by the
new rule from the rows it holds, every later zero-weight row held: so its
continuation is neither the old build's nor a fresh run's of this build, and
the state it re-saves after those rows is neither. Exactly, it continues as
this build continues the same stream with the label times of the
zero-weight rows before the save null (scored and not held, as the old
build did), to the bit, but for `labels_missing`, which counts those nulls.
A state saved after a skipped first row and before its first label was
learned holds no start for that clock, and continues without it, as the old
build did.
A stream with no zero-weight row and no skipped first row, every `embargo`
and every spec without a delay continue to the bit. Measured by
`scripts/compare_outputs.py` against `a5c4dcef` (arm64, polars 2.0.0; 12
models in four cases of 4,000 to 30,000 rows, 1 to 200 groups, chunks of 7
to a whole stream, gaps, nulls, a save and load): only the weighted
`label_clock` specs moved, by up to the whole value with nulls appearing and
vanishing at `min_weight`, and the unweighted `label_clock` `rls`, `sgd` and
`pa` where a group's first row was skipped; `embargo`, no delay, and the
smoke, `kalman` and `lasso` cases 0 ulps. The current fixtures
`bank_label_clock` and `bank_conformal_auto` of `rheon-polars` regenerated
(their `label_frame` scores a zero-weight row on rows 7, 17, 27 and 37, and
row 37's label time, 94, is in flight at the save at row 40;
`bank_conformal_auto`'s unbounded buffer holds it, which
`a_label_buffer_is_frozen_holding_a_zero_weight_row` holds non-empty); every
other fixture, the whole of `rheon-core`'s, is byte for byte. The kept
`v67/` and `v68/` sets of those two kinds are held by `HELD_ZERO_WEIGHT`
(`crates/rheon-polars/tests/state_fixtures.rs`): each re-saves bytes that
load back the same, and continues to the bit of its twin, a bank of this
build fed the rows before the save with the zero-weight rows' label times
null, `labels_missing` aside.

A twenty-second, the user's (2026-11-20, review fable2's C4: stabilized
forgetting "Bd" as `hmm`'s default for `covariance_type` `"full"` and
`"diag"`, at `τ = 1/16`, in the deficit-triggered form; schema 70): an
`hmm` under `"full"` or `"diag"` in a model that learns forgets each
state's covariance toward a stabilized prior (Kulhavý and Zarrop, 1993),
`Σ = (W·M + P₀·diag(ψ))/(W + P₀)` after the row's decay, with `P₀` the new
spec option `prior_rows` (`d` by default) and `ψ` the features' pooled
within-state variance `ŝ`, tracked at ten half-lives with weight `V`, where
it kept a ridge that decayed with the state's data. At 13 features or more
the factor form holds the prior as factored, `f`, decaying with the data
and moved back to `(1 + τ/2)·P₀ψ` by one factorization of each state when
any coordinate is more than `τ` from `P₀ψ`. The model keeps `ŝ`, `V` and
`f` (`2d + 1` numbers a model in the factor form, `d + 1` under `"diag"`
and under `"full"` below 13 features) in `stabilized`, its last field,
written only where it has a prior; `factor_states` is no longer the last
field and is written always; the configuration gains `prior_rows`. Where
the states were given (`means` and `covs`) and are not one diagonal matrix
for every state, each state's prior is the matrix it was given, `G_s`,
rescaled, `diag(√ψ)·G_s·diag(√ψ)`, and `ŝ` is in units of the given
variances (1 as given; a state of 69 or earlier loads with its variances
over the given ones, pooled by weight); the given matrices are read from
the configuration, so the layout is the same and nothing more is written
(review fable2's C4-BD-GIVEN: under the diagonal prior a learning filter
given two regimes that differ in correlation alone put 0.792 of the rows in
the true state, where 0.948 were under the ridge; under each state's own
matrix, 0.976). A state
of schema 69 or earlier keeps none of them, and no loader can replay the
rows it learned under the ridge. It loads as this build would continue
from its moments: `prior_rows` absent (the default, `d`), `ŝ` its states'
variances pooled by weight and `V` their total weight, as the seeding seeds
them; a factor form of schema 69, held under the ridge, is read back as its
co-moments, `σ·L Lᵀ` less the ridge it has reached, and taken up again from
them under the prior with `f` at its target `P₀ψ`; a co-moment state of
schema 68 or earlier at 13 features or more is taken up from its
co-moments the same way (where the twentieth exception took it up under
the ridge). From there it goes on under the prior: neither to the bit nor
within rounding of the old build's continuation, nor as this build's own
run from the same rows, whose prior acted from the seeding. Its
`weight_sum` goes on to the bit, and its other fields are numbers wherever
the old build's were. A state saved inside its warm-up holds no moments to
seed from: it loads with the prior at nothing, seeds it at its seeding, and
goes on to the bit of this build's run. `"tied"`, a model that does not
learn, and every other model continue to the bit, and re-save the current
fixture's bytes.
Measured on macOS arm64: over 4 and 20 features under `"diag"` and 12,
13, 20 and 32 under `"full"`, half-lives of 20 and 400 rows and none,
2,000 rows a run saved after 60 (in the warm-up) and after 600, at schema
68 and 69 (`a_state_before_schema_70_loads_under_the_prior_over_the_grid`,
`crates/rheon-core/src/hmm/tests_stabilized.rs`; 27 runs saved seeded, 18
of them in the factor form, and 27 in the warm-up), the converted `ŝ` is
the states' variances pooled by weight within 1e-13 and `V` their weight
to the bit, a factor form's matrix is `(W·C + P₀ψ)/(W + P₀)` within 1e-13
of its largest entry, each state's `σ` 1 and `f` its target; the
continuation keeps `weight_sum` to the bit and a number wherever the old
build had one, and a state saved inside the warm-up continues to the bit
of this build's run. The posteriors part from the old build's by up to the
whole probability, and the reported `state` differs on 17% of the rows
after the load: on these streams of three blobs the decaying ridge loses
a state that the prior keeps. A state re-saves bytes that load back the
same and continue to the bit. Measured by `scripts/compare_outputs.py`
against `6a594f88` (arm64, polars 2.0.0; the smoke cases, and eight cases
of 6,000 and 8,000 rows over 4 and 14 features, 2 and 3 groups, chunks of
700 and 900 rows, gaps, a shift, nulls, a save and load at half, holding
`hmm` under `"full"`, `"diag"`, `"tied"` and with `learn = False` beside
every other kind but `seqtest` and `rcov`): only `hmm` under `"full"` and
`"diag"` moved, in its posteriors, `state`, `loglik` and `coef`; its
`weight_sum`, `settled_frac` and `withheld_reason`, the `"tied"` and
not-learning `hmm`, and every other model moved 0 ulps. The current
fixtures `hmm_diag`, `hmm_full` and `hmm_full_narrow` of `rheon-core`
regenerated (their continuations are the prior's), `hmm` (`"tied"`) by
its configuration's `prior_rows` and its empty factor list alone (its
continuation to the bit), the new case `hmm_full_warming` (saved inside
its warm-up with `prior_rows = 6.5`) written; every other fixture of
both crates moved only its schema number, or, where a bank holds an
`hmm` spec, by its `prior_rows` and its model's empty factor list. The
kept `v69/` sets are the schema-69 build's; the kept `hmm_diag`,
`hmm_full` and `hmm_full_narrow` of `v65/` to `v69/` are held by
`UNDER_THE_PRIOR` (`crates/rheon-core/tests/state_fixtures.rs`): each
loads, re-saves bytes that load back the same and hold the prior, keeps
its frozen `weight_sum` to the bit and a number wherever the frozen
continuation has one. A released file of `hmm` (the release comparison's
workload holds one under `"full"`) is held the same way against its
release's continuation (`UNDER_THE_PRIOR` in `tests/test_released_state.py`).

A twenty-third, the user's (2026-11-20, review C1: "Accept exception 23";
the twenty-second is `hmm`'s stabilized forgetting, written on its own
branch): `sgd` and `pa` under `standardize` read each row in the stream's
own unit of weight, one row's `ŵ = Σw²/Σw` over the rows the scaler
learned until the switch (Kish's weight per row, held in `Warmup`'s two
sums, frozen with the count), where they read every row at weight 1. While
the scaler warms up a row is admitted to the moments at `ŵ`, and once the
fit is held at `min(ŵ, w)`, its own weight and at most one row's; and the
scaler's history keeps at least `ŵ·n/(n + 1)` of weight into a row before
the switch (`n` Kish's count) and `ŵ` after it, its decay taken as
`max(lam, min(1, floor/W))`. The layout is unchanged (the two sums were
already in the state, only counted) and there is no schema bump. On a
stream whose weights are all 1, `ŵ` is exactly 1 and the read is the old
build's, so a unit-weight state continues to the bit (where a later gap
leaves the history's decayed weight under one row, the floor binds and the
row is read as this build reads any stream there: the fix of the gap, a
change to the outputs the CHANGELOG declares, not of the state). A state
saved by an earlier build
whose rows were not all of weight 1 loads, re-saves the same bytes, and
continues by the new read from the old read's moments and fit, not to the
bit: at a half-life of 20 rows and weights of 0.5 to 1.5, its predictions,
residuals and coefficients move by a median of 2e-3 to 5e-3 relative from
the first row whose step is read at another weight, with no `weight_sum`,
`min_weight` gate or null moved. There is no fix of the weighted read
without it: the defect was in how every weighted row was read. The current
fixtures of `rheon-core` that hold a weighted `sgd` or `pa`, every `sgd_*`
and `pa*` case (15: `sgd`, `sgd_warming`, `sgd_huber`, `sgd_huber_mad`,
`sgd_huber_mad_anchored`, `sgd_huber_refit`, `sgd_huber_refit_done`,
`sgd_spread`, `sgd_unstable`, `sgd_resting`, `sgd_spread_adagrad`, `sgd_eps`,
`sgd_logistic`, `pa`, `pa_warming`), regenerated; every other fixture, and
the whole of `rheon-polars`'s, is byte for byte. The kept `v61/` to `v68/`
sets of those cases are held by `READ_ANEW`
(`crates/rheon-core/tests/state_fixtures.rs`): each re-saves bytes that load
back the same, and continues within a measured bound of its frozen
continuation (the largest move of any number after the save, measured
2026-11-20 on macOS arm64 over every kept set: 1.1e-4 to 1.9e-3 of
`1 + |x|` for a held fit, from the third row after the save, and 3.1e-2 and
8.6e-2 for `sgd_warming` and `pa_warming`, whose predictions read the
moments, from the first; `sgd_unstable`'s outputs did not move). The
kept `v69/` set of those cases, copied when the twenty-second exception
bumped the schema to 70, holds this read's bytes at schema 69 (C1 changed
no layout, and no release carried schema 69 before it), so it converts to
the current fixture's bytes exactly and continues to the bit. A
release's own files (`tests/test_released_state.py`) of the 14 `sgd` and
`pa` specs of its probe, all on weighted rows, are declared moved since
2.0.0 and held to the release's own continuation within `WITHIN`'s 0.06 of
`1 + |v|` (measured: 2.9e-3 to 2.1e-2 in predictions, residuals and
coefficients) and 0.25 for the two diagnosed specs (6.6e-2 and 8.0e-2, in a
Jarque-Bera statistic and a hit rate), every release from 1.0.0 to 2.0.0,
with no non-float field, null or withheld row moved; a 1.3.0 Huber `sgd`
state goes on within 3e-3 of 1.3.0's continuation (8.1e-4 measured).

A twenty-fourth, the user's (2026-11-21: "Accept exception 24"; hard
rule 9 for `audit`): an `audit` learns nothing from a row of weight 0.
Such a row's clock step is counted, as every row's is (until the
twenty-fifth, under which it is not), but its values
reach no count, moment, distinct-value counter, digest, run, lag or
pair, where they were counted like any other row's. The layout is
unchanged and there is no schema bump of its own (it landed beside
schema 71's checksum). An `audit` state of schema 61 to 70 keeps the
zero-weight values it already counted, which cannot be told apart from
the rest. It loads as it was and continues to the bit: an audit's one
output, `n_eff` (`weight_sum`), never took such a row's weight. From its
load it counts no zero-weight row. It re-saves bytes that load back the
same, which differ from the current fixture's in the columns' and pairs'
counts alone; its `n_eff`, clock, configuration and other fields are the
current fixture's. The current fixtures `audit`, `audit_exact` and
`audit_scaled` of `rheon-core` regenerated, because their streams hold
zero-weight rows before the save. The kept `v61/` to `v70/` sets of those
cases are held by `COUNTED_ZERO_WEIGHT` (`counted_zero_weight_rule`,
`crates/rheon-core/tests/state_fixtures.rs`): each loads, continues to
the bit, re-saves bytes that load back the same, matches the current
fixture outside its columns and pairs, and reads at least as many rows in
each column, more in one at least. A bank's `audit` refuses a `weight`
column, so every row it reads weighs 1 and no bank state, output or
`ModelBank.audit()` table moves. Measured by `scripts/compare_outputs.py`
against 2875e1f9 (arm64; the smoke cases, and two `audit` cases with
pairs beside a weighted `ewridge` on streams with zero-weight rows, one
saved and loaded at half): 0 ulps on every field, and the three audit
tables hash the same on both builds. No released file holds a weighted
`audit`, so `tests/test_released_state.py` meets none.

A twenty-fifth, the user's (2026-11-21: "539 exception approved"; hard
rule 9 as task 539 restates it, "the next row should decay as if they were
not present"): a row of weight 0 counts in no clock. The stream scores it as
`predict` scores a row and steps no model, window, diagnostic, held-row
buffer or notice wait on it, so the next row steps from the last row that
counted; it supersedes the twenty-first's holding of such a row under
`label_clock`, and the twenty-fourth's counting of its clock step in
`audit`. No bank layout changed and there is no schema bump. `hawkes` loses
its own `carry` and `seen`, which no released or pushed build wrote, and its
current fixtures were regenerated. A state saved by an earlier build after
rows of weight 0 holds what they did: its clock stands at the last such
row's clock (on a row clock, a row further on; its session that row's), its
models' memories decayed over their steps, every diagnostic aged over them,
under `embargo` and `label_clock` such rows held and waiting, and an
`audit`'s clock counted their steps. No loader can tell those steps from
the rest. It loads as it was and continues under the new rule, with no
loader: the next row steps from the clock the state holds, a held row of
weight 0 is released and replayed as the old build held it, and every row
of weight 0 after the load counts in nothing. So it continues neither to its
frozen continuation nor to a fresh run of this build. Measured on macOS
arm64 (`continuation.py` of the task's scratch, 2026-11-21): 20 kinds
weighted, 3% of the rows of weight 0 and the three rows before the save
among them, a gap past the cap before the save, three groups, chunks of
500, saved by a3554b27 at row 3,000 of 6,000 and continued by this build.
On a number clock, every fit field (predictions, residuals, intervals,
probabilities, distances, states) went on within 3.4e-15 of `1 + |x|` of
both the old build's continuation and this build's uninterrupted run;
`weight_sum` and `settled_frac` parted by up to 0.39 of `1 + |x|` from the
old continuation and 0.30 from the fresh run, on the rows after the load,
and the parting decays with the half-life; `hawkes`' intensities by 8.4e-2
of either. On the row clock the fits parted by 2.8e-3 to 9.1e-2 from the
old continuation and 1.6e-3 to 3.1e-2 from the fresh run (`quantile`'s
residual 0.57 and 0.31, `hawkes` 0.19 and 2.9e-2), the weights by 0.15 and
5.5e-2; `bocpd`, which steps on rows, by nothing; no null flipped anywhere.
The kept bank sets of `rheon-polars` whose frames hold rows of weight 0
(`bank`, `bank_gated`, `bank_label_clock` and `bank_conformal_auto`, `v61/`
to `v70/`) are held by `ZERO_WEIGHT_CLOCK`
(`crates/rheon-polars/tests/state_fixtures.rs`): each loads, re-saves bytes
that load back the same, and continues within the measured bound of its
frozen continuation, relative to `1 + |frozen|` over every number of every
frame, with no other cell moved: `bank` 3.77e-2 (`int_clock`'s `pred_y`),
`bank_gated` 1.67e-1, `bank_label_clock` 2.86e-3 for `v67/` and `v68/` and
1.64e-1 for `v69/` and `v70/`, `bank_conformal_auto` 2.86e-3 and 4.53e-2,
bounded at 0.04, 0.17, 0.17 and 0.05. The current fixtures `bank`,
`bank_gated`, `bank_label_clock` and `bank_conformal_auto` of
`rheon-polars` and `audit`, `audit_scaled`, `hawkes`, `hawkes_window` and
`hawkes_clock_window` of `rheon-core` regenerated. `rheon-core`'s kept
`audit` sets (`COUNTED_ZERO_WEIGHT`) match the current fixture outside their
columns, pairs and now their clock, which counted the steps of the rows of
weight 0 before the save, at least as many as the current one's. No bank's
`audit` takes a weight, so no bank state of one moves. A stream with no row
of weight 0 continues to the bit: every output of `scripts/compare_outputs.py`'s
unweighted cases moved 0 ulps against a3554b27, and a bank state saved
without one is the same bytes but for `hawkes`'.

A twenty-sixth, the user's (2026-11-22: "Switch, restarting the level",
asked whether a saved `deco` state of an earlier schema keeps the old form
of its level or switches to the new one; PLAN 551): a `deco` state of
schema 61 to 71 switches to the ratio level on load, its level restarted.
From schema 72 `deco`'s level `rho` is the ratio of the exponentially
weighted means of the row's numerator and denominator, `EW(A)/EW(V)` (or
their averaged DCC recursion, `N/D`, under `"linear"`), kept in the state's
`ratio`; a state of 71 or earlier kept `rho_bar` and `rho_bar_lo`, the
weighted mean of the row estimate `u` and what it leaves out, from which
neither mean can be had, and no `Σw²/(Σw)²` of the standardiser. What
continues: the configuration, the blocks and the standardiser (its weight,
means and variances) load as saved, so `u`, `weight_sum` and `settled_frac`
go on to the bit of the old build's continuation, and `loglik` is read
under the new level as soon as it has one. What restarts: the level. Every
value's `rho` is NaN, its weight 0, its means of `A`, `V` (and a pair's
second `V`) and `"linear"`'s `N` and `D` empty, and `rho_bar` is let go;
the standardiser's share `Σw²/(Σw)²`, which the warm-up reads, is taken as
1, one effective row, the cold side. The level then relearns from the next
rows under the same warm-up as a new model's (a row teaches it only once
the standardiser's Kish size `(Σw)²/Σw²` passes 5), and until it has a
value `rho` and `loglik` are null and `withheld_reason` reads
`below_warm_rows`. From the share of 1, the share falls below a fifth in
about 1.2 half-lives of rows at a steady rate (24 rows at a half-life of
20), where a state of this build that was warm carries on at once. So such
a state continues neither to its frozen continuation nor to a fresh run of
this build. The state re-saves in the current layout, bytes that load back
the same. No layout of any other model moved, and there is no schema bump
of its own (schema 72 was unreleased). The kept `deco` files of `v61/` to
`v71/` (`rheon-core`) are held by `RESTARTED_LEVEL`
(`restarted_level_rule`, `crates/rheon-core/tests/state_fixtures.rs`):
each loads, re-saves bytes that load back the same with the kept
configuration, blocks and standardiser, every `rho` NaN, every weight 0,
empty means and a share of 1, in the current fixture's layout key for key;
and continues with `weight_sum` and `u` to the bit of its frozen
continuation, and `rho` and `loglik` null until the level relearns and
numbers after, the relearning reached inside the 16 rows. `deco.rs`'s
`a_state_from_before_the_ratio_restarts_its_level` holds the restarted
model to this build's model with the same standardiser and its level
emptied by hand, to the bit, under both dynamics and with blocks. A
release's own `deco` file (`tests/test_released_state.py`,
`RESTARTED_LEVEL`) is declared moved since 2.0.0 and held, every release
from 1.0.0 to 2.0.0, to the release's own continuation in `u`,
`weight_sum` and `settled_frac` to the bit, `rho` and `loglik` null on a
run of first rows and numbers after it, and `withheld_reason`
`below_warm_rows` on that run and the release's value after. No bank
fixture of `rheon-polars` holds a `deco`. The current fixtures `deco`,
`deco_linear` and `deco_warming` of `rheon-core` are the ratio's.

A twenty-seventh, the user's (2026-11-22, PLAN 553: "Switch to the new
statistics", asked whether a state saved before keeps the papers'
statistics or switches; the twenty-sixth is `deco`'s, on its own branch):
`corrchange`'s `monitor` is the linear CUSUM `max_j |Σ_{t≤j} ξ_t|/(√T·D̂)`
against the Kolmogorov quantile less `0.5826/√T`, a pair's flag withheld
where the same CUSUM of `x̃² − ỹ²` passes it too, and its `sequential`
detector compares the monitored rows' correlation with the history's on
Fisher's scale, `((k − 3)/√(m − 3))·|atanh ρ̂_k − atanh ρ̂_h|·(1 − ρ̂_h²)/D̂`,
its critical value at `(K − 3)/(m − 3)`. No layout changed and there is no
schema bump: the state holds no number either old statistic computed (a
span's rows, a history's rows or its correlation and long-run deviation, the
monitored rows and their sums), so a state of schema 61 to 71 loads as it
was, re-saves the same bytes, and goes on under the new statistics from its
first row after the load. What it keeps of the old statistics is where they
flagged: `since_flag` counts from the last flag the old statistic made, and a
`sequential` cycle in progress began at the row after an old flag (or an old
period's end), so it monitors the rows of that cycle against that history
with the new detector, where this build from the same rows would be in
another cycle. The outputs from the load on are as this build computes them
from the state: `stat`, `crit`, `flag`, `since_change` and, after the next
flag, `since_flag`; under `monitor` a state goes on exactly as this build
from the same rows wherever the two statistics flagged alike before the
save. No loader is needed or could do more: the rows before an old flag are
not in the state. The kept sets `v61/` to `v71/` of `corrchange_monitor` and
`corrchange_sequential` are held by `NEW_STATISTICS`
(`crates/rheon-core/tests/state_fixtures.rs`): no span of the monitor's rows
flagged under either statistic before its save, so a kept `monitor` state
converts to the current fixture's bytes exactly and goes on to the current
fixture's frozen continuation to the bit; a kept `sequential` state (the old
detector flagged 8 learned rows before the save, the new one 19) saves bytes
that load back to the same and goes on with its frozen continuation's
`weight_sum` to the bit, its reports the new detector's over the old cycle.
The current fixtures `corrchange_monitor` and `corrchange_sequential`
regenerated (`corrchange_window` and `corrchange_permute_every` did not
move). A release's own file of `corrchange` (`tests/test_released_state.py`,
the probe's `monitor` over spans of 40 rows, every release from 1.0.0 to
2.0.0) is held by `NEW_STATISTICS` there: it goes on as this build goes on
from both halves, every field to the bit but `since_flag`; its release's own
continuation differs in `stat` from the first span closed after the load
(row 89: 0.704 against the release's 0.849).

A twenty-ninth, the user's (2026-11-23, review 2026-11-23's REG-18:
"Exception 29: start fresh", after the decision "Scales from every row"):
under `standardize=True`, `huber` and `quantile` read each feature's scale
from its moments over every row the target is learned on with a positive
weight, decayed as the model's own (`Robust::spread`), refreshed at each
row that moves the target's band (`Robust::spread_scale`), where they read
them from the band's Gram. The band holds only the rows near the fit under
the quantile loss, and a 0/1 feature constant on them had no variance
there, was dropped and held at 0 (the review's case read `[16.67, 0.29,
0.0]` where the unstandardized fit and statsmodels' `QuantReg` read `[0.0,
2.0, 20.0]`). A state of schema 71 or earlier holds no such moments, and
the rows before its save are not in it, so no loader can build them. It
loads with them empty and starts them at its first row after the load;
until they give a feature a scale (a usable variance over the rows since
the load, at least two rows apart), the feature's scale is the band's, as
it was, and a feature with neither is dropped, as it was. So its
continuation is neither the old build's nor this build's from the same
rows. No layout of any other model moved, and there is no schema bump of
its own (schema 72 was unreleased). The kept `huber*` and `quantile*`
files of `v61/` to `v71/` (`rheon-core`) are held by `started_fresh`
(`started_fresh_rule`, `crates/rheon-core/tests/state_fixtures.rs`): each
loads; re-saves bytes that load back the same, with every key the kept
state held, in the current fixture's layout key for key, each target's
moments empty under `standardize` (no weight, every moment 0) and its
scales none, and neither without it; and continues with `weight_sum` to the
bit of its frozen continuation, each prediction a number where the frozen
one is, and within `STARTED_FRESH_BOUND`, 2e-5 relative to the larger of
the frozen value and 1, of it (measured: at most 1.56e-5, `huber`'s fourth
row after the load in every kept set; 3,540 of 4,554 predictions moved,
every null kept). The kept bank files of `rheon-polars` that hold such a
model, `bank_diagnosed`, `bank_refit` and `bank_refit_64` of `v61/` to
`v71/`, are held by `STARTED_FRESH` there: each loads, saves bytes that load
back the same, and its frames go on to the bit of its frozen ones (the
fresh start moves no cell of them over their rows). The current fixtures
of every `huber*` and `quantile*` case and of those three banks were
regenerated, and `the_scales_from_every_row_are_written` holds the new form:
each standardized case's moments holding weight and its scales written,
neither in a case without `standardize`.
