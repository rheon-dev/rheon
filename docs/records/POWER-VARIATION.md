# Power variation and kernel-weighted autocovariance: the design note (task 555)

The building block PLAN 555 names (B3): multipower variation and
kernel-weighted autocovariance sums, with subsampling over offsets, of which
`realized`'s estimators (task 499), the realized kernel, two-scale realized
variance and the HAC long-run variance are instances. Branch
`vm/task-555-power-variation`, cut from `vm/task-499-realized-vol` at
`50a10c5` (request A23, 2026-11-22). This note is part 1 of the build: the
oracle-library scan and what will be built, with the questions the PLAN entry
and the studies do not settle. Nothing in it is built yet.

## 1. The oracle-library scan

Each library was run here on seeded returns against the definition written
by hand, so each row's convention is measured, not read from its docs.
Versions are those installed (R 4.3.3; the repository's `.venv`).

| library | version | licence | what it computes | its convention, measured |
|---|---|---|---|---|
| R `yuima`, `mpv(r = c(p_1, ..., p_m), normalize)` | 1.15.34 | GPL-2 | multipower variation with **any** powers | `n^(Σp/2 − 1) Π μ_{p_i}^-1 Σ_t Π_i \|r_{t−m+i}\|^{p_i}`: `r[1]` on the oldest return of the run, `n` the returns; gap 0 to 2e-16 relative |
| R `highfrequency`, `rMPVar(m, p)` | 1.0.3 | GPL (>= 2) | multipower variation, `m` equal powers `p/m` | with `N/(N − m + 1)` (task 499 measured it for `m = 3`) |
| R `highfrequency`, `rKernelCov(kernelType, kernelParam = H, kernelDOFadj)` | 1.0.3 | GPL (>= 2) | the realized kernel; 12 kernels (rectangular, Bartlett, second to eighth order, Epanechnikov, cubic, Parzen, Tukey-Hanning, modified Tukey-Hanning) | **flat-top**: `γ_0 + Σ_{h=1}^{H} k((h − 1)/H)(γ_h + γ_−h)`, uncentred, not divided by `n`; `kernelDOFadj = TRUE` multiplies `γ_h` by `n/(n − h)`; gap 0 and 4e-16 |
| R `highfrequency`, `rTSCov`, `TSRV(K, J)` | 1.0.3 | GPL (>= 2) | two-scale realized variance on prices | Zhang, Mykland and Aït-Sahalia's form with their small-sample factor `(1 − n̄_K/n̄_J)^-1`, `n̄_K = (N − K + 1)/K`, `N` the **prices**; gap 0 |
| R `highfrequency`, `rAVGCov(alignBy, alignPeriod, k)` | 1.0.3 | GPL (>= 2) | the subsampled average realized variance on a clock grid | not probed yet (a clock grid with offsets of `k` units) |
| R `sandwich`, `kernHAC(bw, kernel, prewhite = FALSE, adjust = FALSE)`, `NeweyWest`, `lrvar` | 3.1-0 | GPL-2 \| GPL-3 | the HAC long-run variance; Bartlett, Parzen, Tukey-Hanning, quadratic spectral, truncated | **centred** (about the sample mean), divided by `n`, weights `k(h/bw)`; `NeweyWest(lag = L)` is Bartlett at `bw = L + 1`; gap 0 and 2e-16 |
| R `hf.econometrics` (YalDan) | 0.2.0 | MIT | `aj_power_variation(x, p, truncation, k)`: one power `p` on `k`-step returns, Aït-Sahalia and Jacod's; pre-averaged power variation | single power only; a second opinion for `p` alone |
| Python `arch`, `arch.covariance.kernel.{Bartlett, Parzen, TukeyHanning, QuadraticSpectral, ...}(x, bandwidth, center, df_adjust, force_int=False)` | 9.0.0 (dev group already) | NCSA | the kernel long-run covariance | divided by `n`, centring optional; weights `k(h/(bw + 1))` for the kernels of compact support, `k(h/bw)` for the quadratic spectral; gap 0 to 4e-16 |
| Python `statsmodels`, `sandwich_covariance.S_hac_simple`, `weights_bartlett`, `tsa.stattools.acovf` | 0.15.0 (dev) | BSD-3-Clause | Newey-West with Bartlett or uniform weights; autocovariances | `weights_bartlett(L)` = `1 − h/(L + 1)` |
| Python `hfhd`, `krvm`, `tsrc`, `msrc` | 0.1.4 (dev) | MIT | realized kernel (flat-top, Parzen and quadratic spectral), two- and multi-scale on prices | task 499's study: `krvm` is the flat-top form, `tsrc` takes `n` as the prices and no small-sample factor |
| Python `hfvol` | 0.1.0 (dev) | MIT | realized variance, bipower and jump variation | task 499 uses it |
| R `realized`, `sde`; PyPI `realized-library` | - | - | `realized` and `sde` are not installed (`realized` is archived on CRAN; `sde` has no power variation); task 499's study found `realized-library`'s ratio test off by a factor of order `n` | not oracles |

**The oracles, by quantity.** Each is test-only, the definition written by
hand from the paper beside it:

- **Multipower variation**: `yuima::mpv`, the one library with arbitrary
  powers, in the `r_oracle` tests (task 499's `r-oracles` group and marker);
  `highfrequency::rMPVar` for equal powers, its `N/(N − m + 1)` divided out.
- **Kernel sums**: `arch`'s kernels (already in the dev group, so in every
  Python run, not only the R job) for the centred and uncentred long-run
  forms at `bw = H`; `highfrequency::rKernelCov(kernelDOFadj = FALSE)` and
  `hfhd.krvm` for the flat-top form; `sandwich::kernHAC` as the R second
  opinion of the centred form, at `bw = H + 1`.
- **Subsampling and two scales**: `highfrequency`'s `TSRV`/`rTSCov` with its
  factor and its count of prices divided out, and `hfhd.tsrc`, on prices
  built as the running sum of the returns fed (task 499's study, section 8).
- The Rust tests hold every form to the definition in double-double, as
  `realized/paper.rs` does; no matrix result, so `crate::oracle` is not
  needed.

No dependency is added: `arch`, `statsmodels`, `hfhd` are in the dev group,
`rpy2` in `r-oracles`; `yuima`, `highfrequency` and `sandwich` are R
packages the `r-oracles` job installs (`sandwich` would be added to its
install line).

## 2. What will be built (the proposal)

A **new kind** beside `realized`, which is left untouched so none of its
outputs, goldens or fixtures can move (its tests run before and after each
part to show it). Working name `power_variation` (question 1). Per target, a
list of terms the spec names, each one output a target:

1. **Multipower term** `powers = [p_1, ..., p_m]`, oldest first (`yuima`'s
   order), each `p_i > 0`:
   `MPV = n^(Σp/2 − 1) Π μ_{p_i}^-1 Σ_t w_t Π_i |r_{t−m+i}|^{p_i}`,
   `μ_p = 2^(p/2) Γ((p + 1)/2)/Γ(1/2)`, `normalize = false` dropping the
   `n` and `μ` factors. `n` is task 499's: the returns a window holds, or
   Kish's `(Σw)²/Σw²` under a decay. So `[2]` is `rv`, `[1, 1]` is `bv`,
   `[2/3]*3` is `tv`, `[4/3]*3` is `tq`, each to `realized`'s definition.
2. **Kernel term** `kernel, H`: `γ_0 + 2 Σ_{h=1}^{H} k(x_h) γ_h`, `γ_h =
   Σ_t w_t r_t r_{t−h}`, with the weight argument `x_h` and the centring as
   questions 3 and 4 settle.
3. **Scale `K`** on either term (subsampling over `K` offsets): the term
   taken on the `K`-step returns `R_t = r_t + ... + r_{t−K+1}`, every `t`,
   divided by `K`, which is the average over the `K` offset grids
   (`[2]` at scale `K` is Zhang, Mykland and Aït-Sahalia's `[Y,Y]^(K)`).

Shared with task 499 by precedent, not reopened: one pass; a row reads the
state before its own return and its decay; decay (default), `window_rows` or
`window_size` with `closed`, a window taking no decay; a window of `W`
returns holds the terms whose every return is in it; zero-weight rows in no
clock; a null return is that target's alone, its terms reaching across;
gaps and sessions are clock steps; TwoSum-compensated sums set to 0 when a
ring empties; lags, `m`, `H` and `K` count returns, with no clock form (as
`lm_window`); no small-sample factor (`kernelDOFadj`'s `n/(n − h)`, ZMA's
`(1 − n̄_K/n̄_J)^-1`) since a factor of `n` has no meaning on decayed
weights; a pure-Rust `libm` `pow` for non-integer powers, the same bits on
every OS, with exact paths for powers 1 and 2.

**State, per target, under a decay**: a ring of the last `max(mK − 1, (H + 1)K − 1)`
returns (lags on the `K`-step grid are `K` returns apart) (8 bytes each; the powers are recomputed from
it), the weights `W`, `V` and `Σ w r` (16 bytes each), and per term one
compensated sum (16 bytes; four for a centred kernel term). One target with
`rv`, `bv`, `tq` and a Parzen term at `H = 50` is about 700 bytes. A window
adds per row held `16` bytes and per target `8 + 16 T` for `T` terms (each
row's share of every term, kernel products credited to their older return so
they leave with it), checked against `max_state_bytes` (256 KB) as
`realized` checks it: 2 targets, 6 terms, `W = 1,000` is about 225 KB.

**Not proposed here**: rewiring `realized`, `rcov`, `corrchange` or
`sandwich` onto the block (each would move outputs or need its own
bit-identity proof; question 8); an automatic bandwidth (it needs the
block's `n` before its first row, or a ring of every lag it could choose:
70 KB at `n` = 23,400 in task 499's study); the quadratic spectral kernel,
which has no finite `H`; terms across targets.

## 3. Questions the PLAN entry and the studies do not settle

1. **The shape**: a new kind (name?) with `realized` untouched, as above;
   or the terms added to `realized`; or `realized` rebuilt on the block,
   bit-identical. And the spec: a dict of user-named terms, e.g.
   `terms={"rv": [2], "bv": [1, 1], "lrv": {"kernel": "parzen", "H": 20}}`,
   naming the outputs `<name>_<target>`?
2. **Multipower's scale**: `yuima`'s `n^(Σp/2 − 1) Π μ^-1` by default (so
   `realized`'s four are instances exactly), with `normalize=False` for the
   raw sum?
3. **The kernel's argument and "flat-top"**: the PLAN writes `k(h/H)`. The
   libraries use three forms: `k(h/(H + 1))` (BNHLS 2011, `rcov`, `arch`;
   Newey-West's Bartlett at lag `H` is `1 − h/(H + 1)`), `k(h/bw)`
   (`sandwich`), and the flat-top realized kernel `k((h − 1)/H)` (BNHLS
   2008, `highfrequency`, `hfhd`), where lag 1 takes full weight. Is
   "flat-top" in the PLAN that last form (proposed: `flat_top=True` beside
   any kernel, default `k(h/(H + 1))`), or Politis' trapezoidal flat-top
   lag window as a kernel of its own? Which kernels: Bartlett, Parzen,
   Tukey-Hanning and rectangular proposed.
4. **Centring and scale for the HAC form**: the long-run variance is about
   the mean and divided by `n` (`arch`, `sandwich`, `corrchange`); the
   realized kernel is neither. Proposed: `center` (default false), the
   centred sum taken about the current mean exactly, `S − m A − m B + m² C`
   from three more kernel-weighted sums (not `ew_lag_cov`'s recursion about
   each row's earlier mean, a different statistic); and the output is the
   sum, with `n` emitted per target so `/ n` is a Polars division. Or emit
   the long-run variance itself?
5. **Subsampling's scope**: a scale `K` on every term (proposed), or on the
   squared return only? Is two-scale RV composed in Polars from two terms
   and `n` (`[Y,Y]^(K) − (n̄_K/n̄_J)[Y,Y]^(J)`, `n̄_K` from the window's or
   Kish's count), or emitted by the block, and with or without ZMA's factor?
   Is a clock form of the offsets (`rAVGCov`'s 5-minute returns at 1-minute
   offsets) wanted now, or `K` in returns only?
6. **`n` as an output**: emit each target's `n` (the count the scales read)
   as a field, so Polars can compose the scale-dependent forms?
7. **Across targets**: univariate per target only (proposed), or the cross
   terms too (realized covariance kernels, a long-run covariance matrix for
   robust SEs), at `m²` sums a term?
8. **Rewiring**: leave `realized`, `rcov`, `corrchange` and `sandwich` as
   they are in this task (proposed), with a follow-up to rebuild them on the
   block where it is bit-identical?

## 4. The 499 baseline

Before any change, on this branch at `50a10c5` (Linux x86-64): `cargo test
-p rheon-core --lib realized -- --include-ignored` and the `golden`,
`model_contract` and `state_fixtures` integration tests: 26, 59, 148 and 27
passed, none failed. Each later part runs them again; a new kind leaves
`realized`'s source untouched, and its fixtures and goldens unchanged byte
for byte.
