<!-- kind: record -->
# Study 511b (C11): the rest of tasks 511-525 (517 to 525), as numpy against their corrected oracles

Python only, no Rust: each feature's online form in numpy, beside the recursion the existing
model keeps (read from `crates/rheon-core` on `origin/main`), against the oracle review C1
confirmed or corrected (`docs/records/ORACLE-VERIFY-511-525.md` on `vm/research-511`, names as
C1 spells them). Method and format as `docs/records/STUDY-511-FEATURES-1.md` (C10, branch
`vm/study-511`). Nothing here is collected by pytest and nothing is added to `pyproject.toml`.

## Files and how to run them

- `scripts/study_511b.py`: prints every table below (`--only 517 518 ...`, `--seeds N`).
- `studies/study_511b/`: `common.py` (seeded streams, decay convention, the `ew_cov` mean-form
  accumulator `EwMoments`, two-pass oracle), one module per task, `t517.py` ... `t525.py`; each
  module's docstring holds the update equations.

Environment: a standalone venv (`/home/user/venv-c11`), not `uv sync`: Python 3.12, numpy 2.5.5,
scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, MAPIE 1.6.0, polars 2.1.0, hmmlearn 0.3.3,
filterpy 2.0.0, dieboldmariano 1.1.0 on Linux x86-64 (torch and `bayesian-changepoint` for 523 are
listed in that section). Decay convention, as the models': a gap `dt` decays everything by
`lam = 0.5 ** (dt / half_life)`, then the row enters at weight `w`; a zero-weight row advances the
clock and learns nothing (rule 9), first row included. Where a feature takes a decay, it is checked
with weights in [0.5, 2], interleaved zero-weight rows and a zero-weight first row.

PLAN task 537 is not in `docs/PLAN.md` on `origin/main` (the plan ends at 534); the growth rule used
is CLAUDE.md's "State stays bounded, of reasonable size, and the fit stays one pass", and the
bank-level total state cap the coordinator relayed (a per-group cap of 256 KB by default; a settable
bank total, warning at half and refusing at the cap, infinity allowed; each feature's bytes per
group feed that total, so each section states its contribution).

SELECTION_PLACEHOLDER

## 517. `ew_cov` outputs: the uncentered covariance and a Gaussian log-likelihood

Beside `ew_cov`'s mean-form recursion (`W' = lam W + w`, `a = lam W / W'`, `b = w / W'`, `d = x - m`,
`m' = m + b d`, `C' = a C + a b d d'`, `Q' = lam^2 Q + w^2`, ridge scale `s' = a s`), both outputs
are reads of what the accumulator keeps:

- **Uncentered second moment** `R = C + m m'` (divisor `W`, the weighted mean of `x x'`; the
  accumulator already offers it as `EwCov::raw`). sklearn's `assume_centered=True` is `X'X / n`;
  `rcov(kind="plain")` is `sum x x' = n R` at `lam = 1`, unit weights.
- **Gaussian log-likelihood** `ll = -0.5 (k ln 2 pi + ln|S| + q)`, `S = c (C + s prior I)`,
  `q = d' S^-1 d` with `d = x - m`, read before the row folds in (rule 2): `q` is the square of
  `mahal`, `c` the unbiased factor `W^2 / (W^2 - Q)` (1 where `bias`), exactly the matrix `mahal`
  inverts. Needs `precision_prior`, as `mahal` does.

Against the corrected oracles (largest gap over 20 seeds):

| case | gap |
|---|---|
| `R` vs `EmpiricalCovariance(assume_centered=True)`, `lam = 1`, k = 2 / 5 / 20 | 2.1e-15 / 2.4e-15 / 2.6e-15 (relative to the largest entry); `n R` vs `X'X`: same |
| `R` vs two-pass `sum omega x x' / sum omega`, half-life 20, weights, 10% zero-weight rows, zero first row, offset 1e3, k = 2 / 5 / 20 | 1.2e-15 / 8e-16 / 1.1e-15 |
| `R` over a window of 60 by retraction vs sklearn on the window, k = 2 / 5 / 20 | 2.1e-14 each |
| `ll` vs `scipy.stats.multivariate_normal.logpdf(x, m, S, allow_singular=True)`, `bias` on/off, prior 0 / 1e-3, `lam = 1` and half-life 30 | at most 1.3e-12 (relative, floor 1) |
| `q` vs `-2 ll_ref - k ln 2 pi - ln|S|` (the square of `mahal`) | at most 4.2e-12 |
| `ll` vs sklearn `EmpiricalCovariance.score` on the one row (C1's second check; fitted on the prefix, `lam = 1`) | 2.4e-13 |

The 1e-12 is the oracle's: scipy's `allow_singular=True` takes the log-density through an
eigendecomposition and a pseudo-inverse, rheon's solve is Cholesky; they differ by the condition
number times the unit roundoff on the generic random covariance used.

**RiskMetrics' EWMA** (`S' = lam S + (1 - lam) x x'`) is the same recursion with the row's weight
fixed at `1 - lam`; `R` gives the row `1/W`, with `W -> 1/(1-lam)`. They agree only after the
transient `lam^t`: at `lam = .94`, seeded with `x_1 x_1'`, the relative gap is 1.5, 0.38, 0.038,
2.0e-3, 3.5e-6, 2e-15 at t = 5, 20, 50, 100, 200, 1000. Seeded with the weight `W_0 = 1/(1-lam)` on a zero
matrix the two are equal to 1.7e-16 from row 1. So `R` **is** RiskMetrics' estimate after a warm-up and
is the unbiased-from-row-1 form before it; the comparison is exact only with that seeding.
`rcov(kind="plain")` is compared as `n R` (above).

Decay and zero weight: `R` and `ll` read the same `C`, `m`, `W`, `Q`, so they inherit the accumulator's
rule 9. A zero-weight row only ages `W` and `Q`; against the same stream with the zero-weight rows
removed and their gaps folded into the next kept row, `R` differs by 7 ulps (the rounding of
`lam^a lam^b` against `lam^(a+b)`); a zero-weight first row leaves `W = 0`, every moment finite, and the
log-likelihood is withheld (no fit) until `min_weight`.

**State added: 0 bytes** for `R` and `ll` (both read the accumulator). Emitted slots, not state:
`R` is `k (k+1)/2` values a row if asked for, `ll` is 1. Nothing grows with the stream. Per group it adds
0 to the bank's total cap.

**Changes a default or an existing output: no.** `Cov` (centered) and `Mahal` stay as they are; the
two are new `stats` entries. One thing to name rather than change: `Mahal` is a distance, the square root
of `q`; the log-likelihood's quadratic form is its square.

## 518. Window operators

Polars' own `rolling_min_by`, `rolling_max_by`, `rolling_quantile_by`, `rolling_rank_by` and
`ewm_mean` are the oracle (all four `*_by` are marked UNSTABLE in Polars' docs, so an upgrade can
move the oracle without a deprecation: C1). The windows are hard windows on the stamps: with
`closed="right"` the window is `(t - window, t]`, `"left"` `[t - window, t)`, `"both"` `[t - window, t]`, `"none"`
`(t - window, t)`. A zero-weight row advances the clock (it evicts rows) and enters no window: a null in
the oracle. Weights and decay do not apply to a hard window (each row counts once); a weighted quantile
would be a different quantity with no oracle, and was not studied.

Online forms: `min`/`max` a monotone deque of `(stamp, value)` (amortised O(1)); `quantile` and `rank` a
ring of `(stamp, value)` plus a sorted list of the window's values (`O(log R + R)` insert and delete in
this numpy form; a Rust form would use a tree). The quantile's read, matched to Polars on `m` values
in the window, `h = (m - 1) q`: `nearest` the value at `floor(h + 1/2)` (halves go up), `higher` `ceil h`,
`lower` `floor h`, `midpoint` the mean of the two, `linear` `v[lo] + (h - lo)(v[hi] - v[lo])`,
`equiprobable` `v[max(ceil(m q) - 1, 0)]`. The rank, `lo` and `hi` the bisect positions of the row's own
value in the sorted window: `average` `(lo + 1 + hi)/2`, `min` `lo + 1`, `max` `hi`, `dense` the number of
distinct values at or below it. `method="random"` is not reproducible (Polars seeds it per operation) and
was not compared.

| operator, against Polars (integer stamps with gaps 1-3, window 25, 10% zero-weight rows as nulls, zero first row, ties on a 0.1 grid; 20 seeds for min/max, 5 for the rest) | largest gap | null-pattern mismatches |
|---|---|---|
| `rolling_min_by` / `rolling_max_by`, all four `closed`, `min_samples` 1 and 3 | 0 | 0 |
| `rolling_quantile_by`, six interpolations, q in {.1, .5, .9}, closed right / both / left | 0 | 0 |
| `rolling_rank_by`, `average`, `min`, `max`, `dense`, closed right / both | 0 | 0 |

**Count-weighted `ewm_mean`.** With `lam = 0.5 ** (1/half_life)` per row (a decay in rows, not clock units)
the mean form `W' = lam W + 1`, `m' = m + (1/W')(x - m)` **is** Polars' `ewm_mean(adjust=True)` (the weights
`(1-alpha)^i`); a zero-weight row is a null with `ignore_nulls=False` (a time step that learns nothing). Gap
0 (bit-equal) at half-lives 3 and 40 rows. `adjust=False` is `y' = (lam^g y + alpha x)/(lam^g + alpha)`, `g` the
rows since the last value (`g = 1` is Polars' `(1-alpha) y + alpha x`), `alpha = 1 - lam`: gap 5e-16 and 1.9e-15
relative. The first value seeds the mean; both forms are null until the first non-zero-weight row. So Polars'
`adjust=True` is an option on rheon's mean (decay by row count where the model's own decays by clock), the
existing clock decay unchanged; on a unit-gap clock the two are the same numbers.

**State added** (rows held `R` inside the window; a clock window holds as many rows as the arrival rate puts
in it, so the form needs a `max_rows` that bounds it): `min`/`max` `16 R` bytes at most (the deque; random data
holds about `ln R`, monotone data holds all `R`: 13 against 200 measured at a 200-row window); `quantile` and
`rank` `24 R` (ring `16 R` and the sorted values `8 R`); count-weighted `ewm_mean` `8 (1 + k)` per column set
(`adjust=False`: 16 bytes a column: `y` and `g`). Per group: `R = 100 / 1000 / 10000`: 1.6 / 16 / 160 KB for `min`
and `max`, 2.4 / 24 / 240 KB for `quantile` and `rank`. The 256 KB per-group cap bounds `R` at 16384
(`min`/`max`) and 10922 (`quantile`/`rank`); a window that fills past `max_rows` must refuse or say it is no
longer exact, and the bank total counts `16 R` or `24 R` per group.

**Changes a default: no.** Four new operators and one new decay option; the clock-decayed means are untouched.

