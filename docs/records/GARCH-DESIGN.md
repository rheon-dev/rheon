<!-- kind: record -->
# `garch`: the design note

The GARCH family as built for PLAN 492, on branch `task-492-garch`,
2026-11-21. The studies behind it are `docs/records/STUDY-492-GARCH-FAMILY.md`
and `ORACLE-STUDY-485.md` on the `vm/study-492` branch; the review that
reshaped it is `.claude/coord/reviews/FABLE-GARCH.md` and its adjudication
`FABLE-GARCH-BOUNDS.md`.

## The estimator

Each target is its own series. Row `t`'s log-likelihood is `l_t = log f(z_t)
− ½ ln s_t`; its score in the process's parameters is `½ w_t ∂ln s_t/∂θ`,
the sensitivity carried by its own recursion (`garch/filter.rs`). Each
learned row steps `H ← H + a(I − H)`, `θ ← Proj_H(θ + a H⁻¹ g)` with `a =
w/W`: the recursive prediction-error method (Ljung and Söderström), first
order equivalent to the batch fit after a batch start.

- **The start.** 2,000 rows by default (at least 2,000 for FIGARCH, whose
  ring of 1,000 lags is mostly the backcast on fewer), fitted by the
  model's own projected Newton with Armijo backtracking (`garch/optim.rs`,
  the core-held optimizer study 485 measured against the libraries'), from
  a small grid. A fit that did not converge, or sits on a bound of the
  process's set, doubles its rows and refits at most twice, while the cap
  holds the doubled buffer. The 500-row start of the studies was where
  every recorded failure began (EGARCH 492006, skew-t 492108, FIGARCH
  492203); at 2,000 the worst seed of a 50-seed grid was within 2 SE.
- **The projection in `H`'s metric** for every form (Dykstra's algorithm
  in `u = Lᵀ y`; a t's shape eliminated by the Schur complement). The
  Euclidean projection of a Newton step was the mechanism of every
  blow-up the studies recorded.
- **Free `ω` by default.** Variance targeting is an option of `garch` with
  normal innovations, with the Huber-clipped score; the clip, centred on its
  own running mean, has mean 0 at every variance level, so it is used only
  where the target fixes the level (a model started at half the truth stayed
  at half with the clip and a free `ω`).
- **The t's `ν` in `s = ln(ν − 2)`**, clamped to `arch`'s `[2.05, 300]`.
  The outer-product `H` is kept for the t; the closed-form information of
  the GARCH block, `ν/(2(ν + 3)) ∂ln s ∂ln sᵀ`, is a later item (the t's
  `n D / L²` runs three times the normal's, the likely cause).
- **EGARCH's guard** (`garch/guard.rs`; the user, 2026-11-21: "Go with
  fables reco on garch"): the box `α ≥ |γ|`, `α ≤ 1`; `arch`'s variance
  clamp; and the mean of `ln |β − (α |z| + γ z)/2|` below 0, read at the
  current parameters from 64 bins of the standardized residuals and held
  by cutting the step back along itself. A refusal on a running mean of
  each row at the parameters of its time ratcheted (seed 492022: 53% of
  steps refused, `β` driven from 0.968 to 0.951 while `arch`'s rose).
  Wintenberger's stable likelihood was weighed and not taken: it is
  sufficient, not necessary, and above 0 at `β = 0.98` for the study's
  `(α, γ)`, so it excludes persistent processes `arch` fits.
- **GJR with the skewed t is refused**: under skewed innovations GJR's
  persistence is `α + γ κ₋ + β`, `κ₋ = E[z² 1(z < 0)]` (0.573 at `(ν, λ) =
  (6, −0.2)`), not `α + γ/2 + β`.

## Rows, weights, the clock

The recursion is per observation, so no parameter of the process has a
duration form (PLAN 538); only the start is a span, of rows or of the clock.
A row of weight 0 or a null return steps nothing (PLAN 539), held to the bit
by `a_zero_weight_row_steps_no_recursion` in the core and
`test_a_zero_weight_row_steps_no_recursion` through the bank. A weight
scales a row's score, information and term in the start's likelihood.

## State (PLAN 537)

Per target `8 (2k + m + n(n + 1)/2 + 9)` bytes, EGARCH's bins 512 beside
them, plus the larger of the start's buffer (16 a row) and FIGARCH's ring
(`8 M`); 16 for the model. Every part is bounded by the settings except a
`start_span` start's buffer, which is checked as it runs (a warning at half
of `max_state_bytes`, a refusal at it); the rest is checked at the call.
`max_state_bytes` is 256 KB by default and refused above 64 MiB.

| form | bytes a target, after the start |
|---|---|
| `garch` | 168 (144 targeted) |
| `gjr` | 216 |
| `egarch` | 728 |
| `garch`, t / skewed t | 208 / 256 |
| `gjr`, t | 264 |
| `figarch`, `M = 1000` | 8,216 |

## Against `arch`

50 seeds a form, 10,000 rows (the probe grid of 2026-11-21, the study's
seeds; `tests/test_second_opinion_garch.py` repeats it on its own seeds in
the extended tier):

| form | within 2 SE | excess / T, median (max) | D, 90th pct (bound) | at 100k: median D, median gap, ratio |
|---|---|---|---|---|
| `garch` | 50 | 0.92 (1.92) | 0.09 (3.91) | 0.007, 0.06, 0.54 |
| `garch`, targeted (vs `arch`'s targeted fit) | 50 | 1.37 (5.20) | 1.55 (3.00) | not held |
| `gjr` | 50 | 0.96 (1.78) | 0.07 (4.74) | 0.008, 0.06, 0.65 |
| `egarch`, `β` 0.97 | 50 | 0.93 (2.14) | 0.07 (4.74) | 0.013, 0.05, 0.72 |
| `egarch`, `β` 0.985 | 50 | 0.88 (1.78) | 0.07 (4.74) | 0.04 |
| `garch`, t | 50 | 0.88 (2.16) | 0.26 (4.74) | 0.024, 0.08, 0.73 |
| `garch`, skewed t | 50 | 0.90 (1.88) | 0.37 (5.54) | 0.045, 0.07, 0.55 |
| `gjr`, t | 50 | 0.91 (2.20) | 0.23 (5.54) | 0.029, 0.07, 0.62 |
| `figarch` | 50 | 0.87 (1.64) | 0.31 (4.74) | 0.052, 0.14, 0.72 |

`T = (k/2)(ln(n/n0) + 1)`. The ratio (median gap at 100k over 10k) is
reported, not gated: after a 2,000-row start its expected value lies
between 0.49 and 0.77, and its spread over 50 seeds is 0.06 to 0.13. The
targeted form's worst excess is its level's cost, which an expanding-window
targeted refit pays too; it is held to its excess over that refit instead.

Per row, steady state (a release build of the probe at 100,000 rows,
arm64, load about 14): `garch` 0.42 µs, `gjr` 0.47, `egarch` 0.55 before the
guard's bins (about 1.2 after), the t 0.75 to 0.88, the skewed t 1.11,
`figarch` 5.8.

## Left out

GARCH(2,2) (a cloud study is testing it on separated roots); a t on EGARCH
or FIGARCH, APARCH, the GED and a mean model (none checked against `arch`);
EGARCH's forecast past one row (no closed form); decay as a gain floor and a
cadence-refit window; the health outputs (the start's report and the guard's
mean are Rust accessors, `start_report` and `invertibility`).
