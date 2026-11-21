<!-- kind: record -->
# Study 511 (C10): the first five additive features of tasks 511-525, as numpy against their corrected oracles

Python only, no Rust: each feature's online form in numpy, beside the recursion the
existing model keeps, against the oracle review C1 confirmed or corrected
(`docs/records/ORACLE-VERIFY-511-525.md` on `vm/research-511`, names as C1 spells
them). Nothing here is collected by pytest and nothing is added to `pyproject.toml`.

## Files and how to run them

- `scripts/study_511.py`: prints every table below (`--only 511 512 513 515 516`,
  `--seeds N`; about 45 s at 20 seeds). The tables in this record are its output at
  the default 20 seeds (a few cases use fewer; each table says so where it does).
- `studies/study_511/`: `common.py` (seeded streams, decay convention, table printer),
  `shrink.py` (511), `hc.py` (512), `scores.py` (513), `conformal.py` (515),
  `constrain.py` (516). Each module's docstring holds the update equations.

Environment: a standalone venv, not `uv sync` (which builds the Rust extension):
Python 3.12, numpy 2.5.5, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1,
MAPIE 1.6.0, crepes 0.9.1, on Linux x86-64. Decay convention, as the models':
a gap `dt` decays everything by `lam = 0.5 ** (dt / half_life)`, then the row enters at
weight `w`; a zero-weight row advances the clock and learns nothing (rule 9), first row
included. Every feature is checked with decay, weights in [0.5, 2], interleaved
zero-weight rows and a zero-weight first row.

PLAN task 537 is not in `docs/PLAN.md` on `origin/main` (the plan ends at 534, with 532
last); the growth rule used is CLAUDE.md's "State stays bounded, of reasonable size, and
the fit stays one pass", and the bank-level total cap the coordinator relayed.

## Selection

C1's verdicts by task, with the additive test (a new option or output column, no change
to a default):

| task | C1 verdict | selected | why |
|---|---|---|---|
| 511 shrinkage covariance | CONFIRMED (sklearn OAS, LW, Shrunk), river CORRECTED | **yes** | option on `ew_cov` / `ew_class`; the default stays unshrunk |
| 512 standard-error types | CONFIRMED (HC0-HC3, QuantReg), `RLM` `H1` CORRECTED | **yes** (HC1-HC3 only; see below) | new `emit_robust_se` types; the existing output is unchanged |
| 513 interval and loss scores | CONFIRMED (MAPIE, sklearn), `log_loss` CORRECTED | **yes** | new outputs |
| 514 DCC recursion, ADCC | **MISSING** as a Python library (R `rmgarch` `dccfilter` CONFIRMED) | no | no Python oracle to compare with; Python only here |
| 515 conformal variants | CONFIRMED (MAPIE `sym=False`, `confidence_level` list, crepes), `ConformalizedQuantileRegressor` CORRECTED | **yes** (asymmetric, several levels, p-value) | new options and outputs |
| 516 equality constraints | CONFIRMED (`RecursiveLS`), `OLS.fit_regularized` CORRECTED to `GLM.fit_constrained` | **yes** | new option |
| 517-525 | | no | the five are 511, 512, 513, 515, 516: the first five by number |

Not done inside the selected tasks, and why: task 512 also lists the `huber` M-estimator
sandwich (C1: `RLM` `cov="H3"` is the sandwich-shaped one, `H1` is not), `quantile`'s
robust standard errors (`QuantReg` `vcov="robust"`) and cluster-robust errors. Each is its
own recursion (a `psi'`-weighted bread for `huber`, an EW kernel density at 0 for
`quantile`); none was built, for time. Task 511's `ew_class` use was not built either:
the shrinkage is one more call on a class's `ew_cov` moments, so the per-class cost is
the per-accumulator cost below. Task 515's crepes oracle covers the p-value only for
`smoothing=False`; the smoothed (randomised) p-value is not studied. Task 513's
size-stratified coverage and D2 are exact only over a row window (below); the
sketch-binned streaming form was not built.

**Defaults check: none of the five changes an existing default or an existing output.**
The one place a name could mislead: the out-of-sample HC0 that `emit_robust_se` emits today
is, read at each row's own time, what the textbook calls HC3 (512). The new types are
options beside it; the existing column keeps its name and its numbers. Renaming it would
be a change, and is not proposed.

## 511. Shrinkage covariance on `ew_cov`

Beside `ew_cov`'s mean-form recursion (`W' = lam W + w`, `a = lam W / W'`, `b = w / W'`,
`d = x - m`, `m' = m + b d`, `C' = a C + a b d d'`, `Q' = lam^2 Q + w^2`, Kish `n = W^2/Q`),
three estimators, `S = C` (divisor `W`, as sklearn's divisor `n`), `mu = tr S / p`:

- **Shrunk**: `(1-s) S + s mu I`, `s` given.
- **OAS as scikit-learn computes it** (C1: not the paper's Eq. 23): `alpha = |S|_F^2/p^2`,
  `s = min((alpha + mu^2) / ((n+1)(alpha - mu^2/p)), 1)`, 1 where the denominator is 0, 0 at `p = 1`.
  Reads `tr S`, `|S|_F^2` and `n` from what `ew_cov` keeps: **no new state**.
- **Ledoit-Wolf**: `delta = (|S|_F^2 - p mu^2)/p`, `beta = (F - |S|_F^2)/(p n)`,
  `s = min(beta, delta)/delta`, with `F = E[|z|^4]`, `z = x - m`. `F` needs two more
  centered quantities in the same mean form, `T = E[|z|^2 z]` (a k-vector) and `F`
  (scalar), updated before `C` is overwritten, with `u = b d`:

```text
T' = a (T - tr(C) u - 2 C u - |u|^2 u)            + b (1-b)^3 |d|^2 d
F' = a (F + 4 u'Cu + |u|^4 - 4 T.u + 2 |u|^2 trC) + b (1-b)^4 |d|^4
```

  The identities are polynomial in the weights, so the same step at weight `-w` retracts a row;
  the study windows that way (rheon subtracts a snapshot: the same quantity, not built here).
  The intensity under weights uses `n` = Kish (the task's choice); at integer weights
  `n = W` is what makes it sklearn's on repeated rows, and both are one line.

Against the corrected oracle (largest gap over 20 seeds; `s` absolute, covariances relative):

| case | OAS s | LW s | OAS cov | LW cov | Shrunk cov |
|---|---|---|---|---|---|
| expanding, `lam = 1`, k = 2 / 5 / 20 | 9e-16 / 7e-17 / 6e-17 | 7e-16 / 1e-16 / 1e-16 | 3e-15 / 6e-14 / 2e-13 | 3e-15 / 6e-14 / 2e-13 | 2e-15 / 6e-14 / 2e-13 |
| window 60 by retraction, k = 2 / 5 / 20 | 1e-14 / 1e-15 / 3e-15 | 6e-15 / 4e-15 / 5e-15 | 8e-14 / 7e-13 / 6e-11 | 7e-14 / 7e-13 / 6e-11 | |

Integer weights with `n = W`, against sklearn on the repeated rows: intensity gap
9e-16 (k=3), 9e-17 (k=8). **Where it is not exact:** under a decay or non-integer weights
sklearn has no `sample_weight`, so the oracle does not apply; the recursion is checked
against two-pass weighted moments instead (half-life 20, weights in [.5,2], zero-weight rows
and a zero-weight first row, features at offset 1e3; C / F / Kish relative, T relative to
`sqrt(F tr C)`): k=2: 3e-13 / 1.6e-13 / 1.5e-15 (T 5e-13); k=5: 4e-9 / 1.3e-13 / 1.3e-15
(T 3e-13); k=20: 7e-11 / 1e-14 / 2e-15 (T 6e-14). The 4e-9 is one off-diagonal of `C` near
zero at offset 1e3, measured elementwise-relatively. River's `OASCovariance`,
`LedoitWolfCovariance` and `ShrunkCovariance(target="identity")` are references only (C1:
heuristic `n`); they were not run.

Zero weight: interleaved zero-weight rows against the same stream with their gaps folded
into the next row: `C` 7 ulps, `F` 0, LW `s` 4 ulps (the rounding of `lam^a lam^b` against
`lam^(a+b)`); a zero-weight first row leaves `W = 0` and every moment finite (the 0/0 is
guarded).

**State added** (per `ew_cov` accumulator; per class for `ew_class`): Shrunk 0 (the intensity
is configuration, 8 bytes); OAS 0; **LW `8 (k + 1)` bytes** (`T` and `F`): k = 5: 48,
k = 20: 168, k = 100: 808, against `ew_cov`'s packed `C, m, W, Q` of 184, 1864, 41224.
Nothing grows with the stream. **Under a `window_size`** each snapshot of the ring carries
`T` and `F` too: `8 (k + 1) S` bytes, `S <= window/spacing + 2` snapshots, so it grows with
the window's parameters, not the stream: k = 20, 1000 snapshots: 168 KB. That is the one
part to cap: `8 (k + 1) S` against the 256 KB per-group cap bounds `S` at `32768/(k+1)` (1560
at k = 20). Per group it adds `8 (k+1) (1 + S)` to the bank's total; the total cap is the
bank-level check.

## 512. HC0-HC3 from the out-of-sample residual

`emit_robust_se`'s sandwich keeps `B = sum omega z z'` and `M = sum omega^2 e^2 z z'` with
`e` the out-of-sample residual. Before the row folds, `g = w z'(lam B)^-1 z`; the fold
gives `h = g/(1+g)` and, by the rank-one identity, the in-fit residual
`e_in = e/(1+g) = e(1-h)`. Read at the row's own time the four types are the one residual
`e` times a factor:

| type | factor on `e^2` | meaning at the row's time |
|---|---|---|
| HC0* | `1/(1+g)^2 = (1-h)^2` | `e_in^2` |
| HC1* | HC0* times `n/(n-p)`, `n` = Kish `W^2/Q` | scalar correction |
| HC2* | `1/(1+g) = 1-h` | `e_in^2/(1-h)` |
| HC3* | `1` | `(e_in/(1-h))^2 = e^2`: **what is emitted today** |

So the out-of-sample HC0 is the row-time HC3, the sense of "already near HC3". Against
statsmodels 0.15.0 (C1: HC1 `n/df_resid`, HC2 `e^2/(1-h)`, HC3 `(e/(1-h))^2`, final-fit residual and
leverage):

- **Exact building blocks** (WLS on the prefix, whitened OLS for the weighted leverage;
  statsmodels' own `WLS` influence is not the weighted leverage): `e_in` relative gap
  4e-13 (`lam=1`) and 2e-12 (half-life 30, weights, zero-weight rows); `h` absolute 5e-15 and
  8e-14; `e` against the WLS fit on the prefix without the row 5e-13 and 2e-12.
- **Exact assembled sandwich** (statsmodels' `cov_hc0` on whitened rows, fed the same
  row-time residuals; HC1 by `n/(n-p)` at Kish `n`): relative gap 1e-12 (`lam=1`), 3e-12 (half-life 40).
- **Statistical only against the final fit** (C1: out-of-sample residuals at each row's fit,
  not the final fit's; a ratio band). SE ratio rheon type / statsmodels same type, 20 seeds,
  k = 3, unit weights, no decay, meat starting after 20 rows, homoskedastic (heteroskedastic
  within 0.5 point of it):

| n | HC0* | HC1* | HC2* | HC3* (= existing) |
|---|---|---|---|---|
| 100 | 0.867 [0.720, 0.986] | same | 0.898 [0.744, 1.039] | 0.934 [0.771, 1.173] |
| 500 | 0.970 [0.942, 0.993] | same | 0.982 [0.954, 1.010] | 0.996 [0.967, 1.033] |
| 2000 | 0.990 [0.982, 0.998] | same | 0.995 [0.988, 1.004] | 1.001 [0.994, 1.011] |

  HC1* is HC0* in the ratio because both sides carry the same `n/(n-p)`. Largest
  `|ratio - 1|` at n = 2000: 1.8% (HC0*), 1.2% (HC2*), 1.1% (HC3*).
- **Caveat that matters: the meat must start after a warm-up.** Started at the first row
  with a fit (`k+1` rows), one huge early out-of-sample residual never washes out at `lam = 1`:
  HC3* against statsmodels reads up to 68x at n = 100 and 58x at n = 2000 (mean 3.0). The
  table above is with 20 rows folded into the bread first; the existing emission has the
  same property, so its min-weight warm-up is what keeps it near 1 in the field.

Decay and zero weight: the leverage and residual identities hold under decay and weights
(row 2 of the exact block above); a zero-weight row only ages (as `Sandwich::update`).
`min_weight`'s `n_eff` rule is untouched.

**State added:** the bread is shared; `hc0` and `hc1` share one meat; one type alone adds
nothing. Each additional distinct meat (`hc2`, `hc3`) is `8 d^2` bytes, `d = k + 1`, and
`8 (d^2 (L+1) + d L)` with `L` HAC lags: `d = 4`: 128 bytes per extra type (928 at `L = 5`);
`d = 21`: 3.5 KB (22 KB); `d = 101`: 82 KB (494 KB). Bounded by `k` and `L`, per slot
(targets x combos). At `k = 100` with `L = 5` a third type nearly fills 256 KB beside the
existing sandwich: that combination should be refused or flagged under the per-group cap.

## 513. Scores for intervals and losses

One EW mean per score in mean form (`m' = a m + b s`), read on the row's out-of-sample
output before the row folds in: coverage, width, Winkler
`(u-l) + (2/alpha)((l-y)+ + (y-u)+)` (bounds un-crossed with min/max, as MAPIE's `regression_mwi_score`
does), MAE, pinball per level, log loss with the clip at `np.finfo(float64).eps` (C1: sklearn
1.9.1 has no `eps` argument; a larger clip would not match). Largest relative gap over 6
seeds (250 rows, 5% crossed bounds); MAPIE for coverage, width, Winkler (`y_pis`,
`1 - confidence_level`; no weights, so its per-row values are averaged at the weights),
scikit-learn with `sample_weight` for the rest:

| case | coverage | width | Winkler | MAE | pinball .1 / .5 / .9 |
|---|---|---|---|---|---|
| `lam=1`, unit w | 6e-16 | 1e-15 | 1e-15 | 2e-15 | 9e-16 / 2e-15 / 9e-16 |
| `lam=1`, w in [.5,2] | 6e-16 | 1e-15 | 1e-15 | 1e-15 | 9e-16 / 1e-15 / 1e-15 |
| half-life 60, w, 12% zero-weight rows | 4e-16 | 7e-16 | 4e-16 | 5e-16 | 6e-16 / 5e-16 / 1e-15 |

Log loss (probabilities include exactly 1.0): 1.3e-15 (`lam=1`), 8e-16 (decay, zero-weight rows).
**A difference to state, not a gap:** MAPIE's `regression_coverage_score` does not un-cross
bounds, so on raw crossed bounds it counts a miss: 0.048 to 0.060 below the un-crossed
coverage on 5% crossed rows. Un-crossing is what the Winkler score does; coverage should do the same.

**Not one scalar:**
- *Size-stratified coverage* (`regression_ssc`, equal-count bins by width): exact over a row
  window from a ring of `(width, hit)` (gap 0 at windows 50 and 200). A fixed-bin streaming form
  would be approximate; not built.
- *D2 pinball*: C1's caveat holds: `D2 = 1 - sum rho(y-yhat)/sum rho(y-q)` with `q` the
  whole-`y_true` quantile (sklearn's `_weighted_percentile(average=True)`: inverted CDF averaged
  where `alpha n` is whole). Exact over a window from a ring of `(y, rho_t)` and a sorted read
  (gap 4e-16). Over the stream the null can only be a running quantile (8 bytes, step = rate x EW
  mean absolute deviation), a different quantity: against sklearn's whole-stream D2, on a
  stationary stream, mean gap 0.085 (n = 2000, rate .02) falling to 8e-4 (n = 20000), max 0.16
  to 0.013; on a stream whose mean wanders the gap is -0.25 to -1.0 (the quantities differ, not
  only the estimate). So: pinball is exact and streaming; D2 is exact only over a window.

**State added:** one `f64` per score plus a shared weight, `8 n + 8` bytes (7 scores: 64).
Window forms, per slot: SSC ring `16 W`, D2 ring `16 W`, one shared order index `8 W`
(together `40 W`: 2 KB at W = 50). `W` is the row window: a ring over 256 KB at `W` = 6553 (both rings)
or 16384 (one): cap `W` by the per-group bytes.

## 515. Asymmetric and multi-level conformal intervals, per-row p-value

The existing tracker is `m' = max(0, m + eta (w/w_bar)(err - alpha))` on `|r|/sigma`. Added:
one pair of **signed** trackers per level `alpha_l`, each a quantile tracker of the signed
scaled residual `s = r/sigma` at tail level `alpha_l/2` (MAPIE `AbsoluteConformityScore(sym=False)`, `beta = alpha/2`):
`m_hi' = m_hi + eta rel (1{s > m_hi} - alpha_l/2)`, `m_lo' = m_lo + eta rel (1{-s > m_lo} - alpha_l/2)`,
interval `[pred - m_lo sigma, pred + m_hi sigma]`, no clamp at 0, both starting at
`Phi^-1(1 - alpha_l/2)`. Levels are independent recursions, so nested levels can cross (2.2%
of rows at rate .05, 0.03% at .02, none at .005 for levels .8/.9/.95); sort at read, no state.
**The per-row conformal p-value** (crepes `predict_p_online`, `smoothing=False`):
`p_t = (sum_i omega_i 1{a_i >= a_t} + 1)/(sum_i omega_i + 1)` over a ring of the last `W`
scores `a = |r|/sigma` with their clock stamps (`omega_i = w_i lam^age`), read before the row is stored.

Against the corrected oracles (8 seeds, 40000-row stream, skewed gamma noise with sd 0.99;
MAPIE calibrated on 40000 fresh rows with the same model; tracked offsets are the mean over the
second half):

- **Tracker against MAPIE `SplitConformalRegressor(AbsoluteConformityScore(sym=False), confidence_level=[.8,.9,.95])`**:
  statistical, not exact: a stochastic tracker against an empirical quantile. Largest gap, in units
  of the noise: lower side 0.007 to 0.012, upper side 0.07 to 0.10 across rates .05 / .02 / .005.
  The upper tail is the gap's source: against the true noise quantile, MAPIE's own calibrated offset
  is off by up to 0.045 (sampling error at 40000 rows) and the tracker by 0.04 to 0.07, so the
  tracker is within the oracle's own sampling error of the truth.
- **Exact identities** (the guarantee): `m_{T+1} - m_1 = eta sum rel (err - alpha_l/2)`, gap 8e-12;
  realized coverage within 1.4e-3 of the level at every rate; each side's miss rate within
  `2(B+eta)/(eta T)` of `alpha_l/2`: observed 0.012 (T = 1000, rate .05) against a bound of
  0.59, 5e-4 against 0.018 (T = 20000).
- **crepes `predict_p_online`** (expanding, `smoothing=False`, `warm_start=True`): exact, gap 0
  (5 seeds, 60 warm-up rows plus 400 online rows with row-varying sigma); a 100-row window
  against `predict_p` on that calibration set: 0; the decayed, weighted, zero-weight-row form
  against the explicit weighted share: 2e-16 (crepes has no decay).
- Decay and zero weight: signed multipliers with zero-weight rows (first row included) against the
  same stream with the gaps folded forward: 4e-16, all finite. A zero-weight row only ages the
  realized coverage and the row count, as the symmetric tracker.

**State added** per slot: signed trackers `24 L + 24` bytes at `L` levels (`m_hi`, `m_lo` and the
realized-coverage mean per level, plus `cov_w`, `w_mean`, `w_rows` shared): 48, 96, 144 at
`L` = 1, 3, 5, against 40 for today's symmetric tracker. The p-value ring is `16 W` bytes
(score and stamp; `+8 W` where row weights are not all 1): 4 KB at W = 256, 16 KB at 1024, 128 KB at
8192. The ring is the only part that scales with a parameter; it is a row window of a rank, which
has no bounded exact form without it. Cap `W` so `16 W` stays under the per-group cap (16384 alone).

## 516. Linear equality constraints on a Gram fit

In mean form (`A = (C + rho I)^-1`, `beta = A c`), on the slopes (the intercept is the free
parameter that keeps the centered system exact), `R beta = r`, `R` `q x k`:

```text
beta_c = beta - A R' (R A R')^-1 (R beta - r)        beta_0 = ybar - m' beta_c
```

One extra `q x q` solve at read time on the Gram; decay, weights and zero-weight rows change
nothing but the Gram. `R A R'` singular (dependent or unreachable constraints) is refused
(tested). Against C1's corrected oracle `GLM(Gaussian, var_weights).fit_constrained((R, q))`
(closed-form restricted WLS, final fit), three constraint sets (sum = 1; `b1 - 2 b2 = 0`;
two rows), 20 seeds, n = 400, k = 4: relative gap 1e-12 at `lam = 1` and 4e-13 at half-life 50 with weights
in [.5,2] and zero-weight rows (first row included); `|R beta - r|` at most 3e-14. With a ridge
penalty (0, .05, 2) against a second derivation, the KKT block system: 9e-14, 2e-14, 6e-15. Row
by row against `RecursiveLS(constraints=(R, q)).filtered` (expanding, unit weights, ridge 0, rows
aligned on the same row, 8 seeds, rows 5-119): absolute gap 1e-12 to 5e-12 for one constraint,
1.8e-10 for two (C1: the filter imposes constraints as fake observations; 1e-8 plausible, 1e-12 not).
A statsmodels 0.15 defect met: `RecursiveLS(constraints=...)` with a numpy 1-D `endog` raises
`IndexError` (it indexes `endog[:, 1:]`); the study passes `endog` with its `q` constraint columns,
which the constructor overwrites. Zero-weight rows against the gaps folded forward: 8e-15.

**State added:** none that accumulates. `(R, r)` is configuration, `8 q (k+1)` bytes if the
state file carries it (k = 20, q = 5: 840 bytes); the `k x q` solve scratch is not state.

## Proposed tolerances (for the Rust ports' tests against the same oracles)

| feature | quantity | tolerance | observed worst |
|---|---|---|---|
| 511 | intensity vs sklearn, expanding or integer weights | 1e-12 abs | 9e-16 |
| 511 | shrunk covariance vs sklearn, expanding | 1e-10 rel | 2e-13 (k=20) |
| 511 | shrunk covariance vs sklearn, window by retraction | 1e-9 rel (snapshot subtraction may differ) | 6e-11 (k=20) |
| 511 | decayed moments vs two-pass, offset 1e3 | 1e-7 rel | 4e-9 |
| 512 | `e_in`, `h`, assembled sandwich vs statsmodels on the same residuals | 1e-10 rel | 3e-12 |
| 512 | SE vs statsmodels final fit | ratio in [0.97, 1.03] at n >= 2000, k = 3, after a 20-row warm-up; no tolerance below | [0.982, 1.011] |
| 513 | EW means vs MAPIE / sklearn | 1e-12 rel | 2e-15 |
| 513 | windowed SSC / D2 | 1e-12 abs | 0 / 4e-16 |
| 515 | tracker vs MAPIE offsets | statistical: 0.15 sd at rate .02, 40000 rows | 0.10 |
| 515 | telescoping identity | 1e-9 abs | 8e-12 |
| 515 | p-value vs crepes (exact), expanding or window | 1e-12 abs | 0 |
| 516 | final fit vs `GLM.fit_constrained` | 1e-10 rel | 1e-12 |
| 516 | row by row vs `RecursiveLS` (from row k + 8) | 1e-8 abs | 1.8e-10 |

## State summary, against the caps

| feature | adds, per slot or accumulator | grows with | needs a cap |
|---|---|---|---|
| 511 OAS / Shrunk | 0 | | no |
| 511 LW | `8 (k+1)`; windowed `8 (k+1) S` | `S <= window/spacing + 2` | yes, on `S` |
| 512 HC types | `8 d^2` per extra distinct meat (`8(d^2(L+1)+dL)` with lags) | `k`, `L` | flag `k = 100`, `L = 5` |
| 513 means | `8 n + 8` | | no |
| 513 SSC, D2 windows | `40 W` | window `W` | yes, on `W` |
| 515 signed trackers | `24 L + 24` | levels | no |
| 515 p-value ring | `16 W` (`24 W` weighted) | window `W` | yes, on `W` |
| 516 constraints | 0 (config `8 q (k+1)`) | | no |

Nothing grows with the stream's length. The parts that scale with a configured size (LW's
snapshots, the windowed rings) add `bytes per group` as the formulas give; the settable total
state cap (warn at half, refuse at the cap, infinity allowed) is a bank-level check on the sum
over groups of these bytes plus the existing state, not a per-feature one, and the per-group
256 KB cap binds first for a single ring above `W = 16384` or an LW window above `S = 32768/(k+1)`.
